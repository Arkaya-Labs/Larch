//! Folio: the chat's attached documents read whole, Research v2's way. A document is its text and
//! its pages (`Doc`), lettered for the life of the chat by its source record (`source`). Their
//! text opens the turn's system prompt (`block`: the rules, then each document under its letter
//! with a mark where each page starts), so the engine reads it once and every turn after it
//! costs only its own words; the agent's usual system text (persona, memory, date) and tools
//! follow it. The model quotes the words that show a claim and cites the page after them,
//! `"..." [p. 12]` or `"..." [B, p. 12]`; Rust finds the quote on that page (`cite`) and adds a
//! note under the answer for each that is not there. It never rewrites the answer.
//!
//! The pieces are plain functions other skills may call: the store (`store`, `store_url`,
//! `load`), the cite check (`cite::check`, `notes`, `checked`), the map of a document
//! (`map::map`, `map::outline`) and the reader's tools (`tools::find`, `tools::read`,
//! `tools::place`). `route` sizes a set against the loaded context and the device: small
//! documents are plain chat text, medium ones pinned, large ones read by section.

use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

pub mod cache;
pub mod cite;
pub mod held;
pub mod map;
pub mod ocr;
pub mod source;
pub mod table;
pub mod tools;
pub mod visible;

use larch::{ChatMessage, EngineError, Finish, Larch, Role, SessionOptions, Timings};
use serde::{Deserialize, Serialize};

use crate::{
    config::Thinking,
    parse::Family,
    runtime::inference::SeatAdmission,
    skill::SkillError,
    skills::{
        ask_docs::{self, Log, NOT_FOUND},
        read_papers, web_answer,
    },
    tool::{ToolCtx, ToolError},
    web::fetch,
    workers::answer,
};

pub use larch::api::TURN_ROOM;

/// The part of `TURN_ROOM` kept free for the answer, which `trim` must leave out of the room it
/// gives the history and the question: `TURN_ROOM` counts all three, so alone it would fill the
/// room with history and leave the answer none.
// ponytail: the engine has no constant for it; take it from `larch` if one lands
pub const ANSWER_ROOM: u32 = 4_096;

/// Tokens of thinking a turn on documents takes. With none, a question across a datasheet's
/// tables (which pins each package puts a signal on) reasoned ~1,000 tokens in the answer itself,
/// "Wait, let's re-read the table" included (24 Nov). With 2,048, a 21-page report whose whole
/// text was in the prompt thought its way into search_files and read_file instead of answering,
/// and the first word came up to ~35 s later (24 Nov). A short block checks a footnote against a
/// table; if it runs out, the answer still follows. A Mac A/B at 0, 512 and 1,024 (24 Nov) was
/// equally correct at all three, 1,024 used no more than 512, and 0 gave the first word soonest:
/// 512 is a choice for the answer's look (the self-correction stays out of it, and the hard turn
/// ended ~30 s sooner), not for correctness. A question it answers wrong only sometimes is the
/// test that would move it.
pub const DOC_THINKING: u32 = 512;

/// The thinking of a turn on documents: `DOC_THINKING` whatever `chat`'s budget (0, thinking
/// turned off, stays off), and the answer after it the rest of `ANSWER_ROOM`, or less when `chat`
/// caps it, so the block and the answer fit the room the plan keeps.
pub fn thinking(chat: Thinking) -> Thinking {
    let budget = match chat.budget {
        0 => 0,
        _ => DOC_THINKING,
    };
    Thinking {
        budget,
        answer_max_tokens: match chat.answer_max_tokens {
            0 => ANSWER_ROOM - budget,
            n => n.min(ANSWER_ROOM - budget),
        },
    }
}

/// The skill's name in the audit log.
const NAME: &str = "pinned-doc";

/// The rules at the head of the turn's system prompt; the documents follow them. The model may
/// answer anything: only a question about the documents that they do not answer gets `NOT_FOUND`.
const RULES: &str = "You answer questions about the user's documents, which follow. Each document \
has a letter, and a mark such as [p. 3] starts each page and recurs within it: the words after \
it are on that page. When a question is about the \
documents, answer from them and nothing else: their whole text is here, so use no tool to \
open or search them. Quote the exact words that show each claim, in \
quotation marks, and cite their page in brackets right after the quote: [p. 12], or [B, p. 12] \
for document B; with one document, [p. 12]; a quote across a page break, [pp. 12-13]. Quote only \
what the page says, and never give a number you did not read. Be brief. If the question is about the documents and they do not \
answer it, reply with only this sentence: ";

/// Bytes of a page's text after which its next paragraph repeats the page mark ([`render`]).
const MARK_EVERY: usize = 1_000;

/// What follows `NOT_FOUND` in the rules: the rest of what a chat on documents may do.
const OTHER: &str = "\nIf the question is not about the documents (the weather, a greeting, \
something from an earlier answer), answer it as you normally would, without citing.\n";

/// How a document's text was made. A citation names it, as one document has different offsets
/// and pages in each.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Rep {
    /// Text a PDF holds, from its text layer.
    Pdf,
    /// Text read from a page's pictures.
    Ocr,
    /// An arXiv paper's LaTeX source.
    Latex,
    /// A web page's text.
    Html,
    /// A plain text or Markdown file.
    Text,
}

/// One page of a document's text: its byte range, and the label the page prints when one was
/// found ("iv", "12"). A page with no text is an empty range, so page indices never shift.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page {
    /// Where the page starts in the text.
    pub start: usize,
    /// Where it ends, exclusive.
    pub end: usize,
    /// The number or label printed on the page.
    pub label: Option<String>,
}

/// A document a chat read, in one representation: its whole text and its pages in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Doc {
    /// Its letter in the chat, from 0: 0 is "A".
    pub index: usize,
    /// The file name or URL, as the user gave it.
    pub name: String,
    /// The SHA-256 of the original bytes, lowercase hex.
    pub sha256: String,
    /// How the text was made.
    pub rep: Rep,
    /// The whole text.
    pub text: String,
    /// Its pages, in order; one page for a document that has none.
    pub pages: Vec<Page>,
}

impl Doc {
    /// The index of the page that holds byte `at` of the text: the last page that starts at or
    /// before it, so an empty page is never one.
    pub fn page_of(&self, at: usize) -> usize {
        self.pages
            .iter()
            .rposition(|p| p.start <= at && p.start < p.end)
            .unwrap_or(0)
    }

    /// The letter the user sees for this document: "A", "B", ... "Z", then "AA".
    pub fn letter(&self) -> String {
        let (mut n, mut out) = (self.index, Vec::new());
        loop {
            out.push(char::from(b'A' + (n % 26) as u8));
            if n < 26 {
                break;
            }
            n = n / 26 - 1;
        }
        out.iter().rev().collect()
    }
}

/// A byte range of one document's text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Range {
    /// The document's index in the chat.
    pub doc: usize,
    /// The first byte.
    pub start: usize,
    /// The end, exclusive.
    pub end: usize,
}

/// Where a citation points inside one representation of a document.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Loc {
    /// The PDF page index, from 0; the user sees it plus one.
    pub page: Option<usize>,
    /// A printed page label, a LaTeX label or an item label ("Table 3").
    pub label: Option<String>,
    /// The section the text is in, by its heading.
    pub section: Option<String>,
}

/// A citation as Rust keeps it: which document, which snapshot, which representation, and
/// where. The user sees only the short form ("[B, p. 12]").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cite {
    /// The document's index in the chat.
    pub doc: usize,
    /// The snapshot's SHA-256.
    pub sha256: String,
    /// The representation the location is in.
    pub rep: Rep,
    /// Where in it.
    pub loc: Loc,
}

/// The pages of `docs` together: the pages with text.
pub fn pages(docs: &[Doc]) -> u32 {
    let n = docs
        .iter()
        .flat_map(|d| &d.pages)
        .filter(|p| p.start < p.end);
    u32::try_from(n.count()).unwrap_or(u32::MAX)
}

/// The documents as the prompt shows them: each under "Document B: name", and `[p. N]` before the
/// text of each page that has some, N from 1 as a cite counts.
pub fn render(docs: &[Doc]) -> String {
    let mut out = String::new();
    for d in docs {
        out.push_str(&format!("\nDocument {}: {}\n", d.letter(), d.name));
        for (i, p) in d.pages.iter().enumerate().filter(|(_, p)| p.start < p.end) {
            // A paragraph carries its page again once MARK_EVERY bytes passed since the last mark:
            // deep in a long block the model cited the page a sentence or list began on (128k: 3 of
            // 8 quotes on the page), and a mark on every paragraph cost 7% more tokens (figures'
            // axis labels are paragraphs too).
            let mark = format!("[p. {}]", i + 1);
            let text = d.text.get(p.start..p.end).unwrap_or_default();
            out.push_str(&format!("{mark}\n"));
            let mut since = 0;
            for para in text.split_inclusive("\n\n") {
                if since >= MARK_EVERY && !para.trim().is_empty() {
                    out.push_str(&format!("{mark} "));
                    since = 0;
                }
                out.push_str(para);
                since += para.len();
            }
        }
    }
    out
}

/// The Folio block: the rules and the documents, and nothing else (no date, no memory, no tools),
/// the first thing of every turn's system prompt, so these are the same bytes for every turn on
/// the documents, today and any day after, and the engine's cache of this prefix serves each one.
/// The agent's own system text follows it.
pub fn block(docs: &[Doc]) -> String {
    format!("{RULES}{NOT_FOUND}{OTHER}{}", render(docs))
}

/// The system prompt of a turn on `docs` whose agent's own system text is `plain`: the Folio block
/// first, so a change in `plain` (the date, a saved memory) re-reads only what follows it.
pub fn system(docs: &[Doc], plain: &str) -> String {
    before(block(docs), plain)
}

/// `head`, then the chat's own system text `plain` on the next line when it has any.
pub fn before(head: String, plain: &str) -> String {
    match plain.is_empty() {
        true => head,
        false => format!("{head}\n{plain}"),
    }
}

/// The file at `file`, which the user named `name`, as the chat `chat` reads it: its text from the
/// extract cache (`source::text`), entered in the chat's source record, which gives it its letter
/// for the life of the chat. `None` when the file gives no text (a scan, a file the readers
/// cannot open, or one over read_papers' size cap); the caller then lets ask_docs say so.
/// `file` is the real path `ask_docs::documents` gives (no link in it): the record and the
/// extract cache are kept under it, and `forget` looks them up by it.
// ponytail: no OCR here (`source::text` with no `pictures`): a scan is `None` until the app asks
// for its pictures to be read and shows their progress
pub fn store(
    data: &Path,
    chat: &str,
    name: &str,
    file: &Path,
    stop: &AtomicBool,
) -> Result<Option<Doc>, SkillError> {
    let io = |e| SkillError::Tool(ToolError::Io(file.display().to_string(), e));
    if fs::metadata(file).map_err(io)?.len() > read_papers::MAX_FILE_BYTES {
        return Ok(None);
    }
    // A file the chat's record read with this size and change time is opened from the extract
    // cache, unread: its bytes are neither read nor hashed again.
    let here = file.display().to_string();
    let stamp = fs::metadata(file).ok().and_then(|m| source::Stamp::of(&m));
    let known = source::load(data, chat).unwrap_or_default();
    let same = |e: &source::Entry| e.source == here && stamp.is_some() && e.stamp == stamp;
    if let Some((index, entry)) = known.iter().enumerate().find(|(_, e)| same(e))
        && let Ok(opened) = source::open(data, index, entry, stop)
    {
        return Ok(Some(opened.doc));
    }
    let read = match source::text(data, file, stop, None) {
        Ok(read) => read,
        Err(_) if stop.load(Ordering::Relaxed) => {
            return Err(SkillError::Engine(EngineError::Cancelled));
        }
        Err(_) => return Ok(None),
    };
    // A PDF of no text is only its form feeds.
    if read
        .text
        .trim_matches(|c: char| c.is_whitespace())
        .is_empty()
    {
        return Ok(None);
    }
    let index = source::add(data, chat, source::Entry::file(name, file, &read)).map_err(io)?;
    Ok(Some(source::doc(
        index,
        name,
        &read.sha256,
        read.rep,
        read.text,
    )))
}

/// The page at `url` as a document of chat `chat`: a PDF with its pages (`Rep::Pdf`), any other
/// page as its text (`Rep::Html`), fetched as every web tool fetches (robots.txt, the size cap,
/// the https upgrade, a Drive or Dropbox share link made a direct one, up to 32 MiB) and entered
/// in the chat's source record, a PDF's bytes kept beside it. Named by its final URL. `None` when
/// the page has no text.
pub fn store_url(
    data: &Path,
    chat: &str,
    url: &str,
    stop: &AtomicBool,
) -> Result<Option<Doc>, SkillError> {
    if stop.load(Ordering::Relaxed) {
        return Err(SkillError::Engine(EngineError::Cancelled));
    }
    let mut seen = fetch::Seen::default();
    let page = (fetch::fetch_documents(&[url], data, &mut seen, true)
        .into_iter()
        .next())
    .expect("one URL, one result")
    .map_err(SkillError::Tool)?;
    if stop.load(Ordering::Relaxed) {
        return Err(SkillError::Engine(EngineError::Cancelled));
    }
    // A page of less text than a shell (an app page, a login wall) is no article: the normal tools
    // take its link.
    let bare = page.text.trim().len() < web_answer::SHELL_BYTES;
    if page.text.trim().is_empty() || (!page.pdf && bare) {
        return Ok(None);
    }
    let io = |e| SkillError::Tool(ToolError::Io(url.to_owned(), e));
    let name = page.final_url.clone();
    let index = source::add(data, chat, source::Entry::link(&name, &page, false)).map_err(io)?;
    let rep = if page.pdf { Rep::Pdf } else { Rep::Html };
    Ok(Some(source::doc(
        index,
        &name,
        &page.sha256,
        rep,
        page.text,
    )))
}

/// The chat's documents, in letter order: the files attached as themselves, each stored (`store`),
/// and the links its source record holds, each reopened as it was read (`source::open`). `None`
/// when there are none or more files than read_papers reads at once, so that ask_docs answers.
/// A file that gives no text, and the oldest links past that cap, are left out and named in the
/// second list, so one scan or a ninth link does not turn off the rest (#590 F5, F6). An attached folder is documents here only when it is a folder of papers
/// (`ask_docs::papers_folder`): a repo or a folder of notes and code is attached for its files,
/// which ask_docs and the file tools read, not to be pinned for its README.
// ponytail: at most read_papers::MAX_PAPERS documents, each opened from the extract cache per
// question while its size and change time hold; keep the set in memory per chat when that shows
// in a question's time
pub fn load(
    ctx: &ToolCtx,
    chat: &str,
    stop: &AtomicBool,
) -> Result<(Option<Vec<Doc>>, Vec<LeftOut>), SkillError> {
    let attached = ToolCtx {
        allowed: (ctx.allowed.iter())
            .filter(|p| p.is_file() || ask_docs::papers_folder(p))
            .cloned()
            .collect(),
        ..ctx.clone()
    };
    let files = ask_docs::documents(&attached)?;
    let record = source::load(&ctx.data_dir, chat)
        .map_err(|e| SkillError::Tool(ToolError::Io(chat.to_owned(), e)))?;
    let mut links: Vec<(usize, &source::Entry)> = (record.iter().enumerate())
        .filter(|(_, e)| e.stamp.is_none())
        .collect();
    if files.len() + links.len() == 0 || files.len() > read_papers::MAX_PAPERS {
        return Ok((None, Vec::new()));
    }
    let (mut docs, mut skipped) = (Vec::new(), Vec::new());
    // Past the cap, the latest links stay: an old link leaves the set, not every document.
    let over = (files.len() + links.len()).saturating_sub(read_papers::MAX_PAPERS);
    let cap = "the chat has more documents than are read at once, and this link is the oldest";
    skipped.extend(links.drain(..over).map(|(_, e)| (e.source.clone(), cap)));
    for (name, file) in files {
        match store(&ctx.data_dir, chat, &name, &file, stop)? {
            Some(doc) => docs.push(doc),
            None => skipped.push((name, NO_TEXT_LEFT_OUT)),
        }
    }
    for (index, entry) in links {
        match source::open(&ctx.data_dir, index, entry, stop) {
            Ok(opened) => docs.push(opened.doc),
            Err(_) if stop.load(Ordering::Relaxed) => {
                return Err(SkillError::Engine(EngineError::Cancelled));
            }
            Err(_) => skipped.push((entry.source.clone(), NO_TEXT_LEFT_OUT)),
        }
    }
    docs.sort_by_key(|d| d.index);
    Ok(((!docs.is_empty()).then_some(docs), skipped))
}

/// A document `load` leaves out: its name and why.
pub type LeftOut = (String, &'static str);

/// Why `load` leaves out a document that gives no text.
const NO_TEXT_LEFT_OUT: &str =
    "it gives no text (a scan, or a file too big or that cannot be opened)";

/// How a question on documents is answered, by the size of the set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Route {
    /// Small: the block fits the context already loaded with a turn's room to spare
    /// (`small_room`), so it goes in the chat as plain text and nothing is pinned.
    Small(Vec<Doc>),
    /// Medium: the block fits the device under the documents plan: read whole and pinned;
    /// `host_layers` is the fit's, for `Documents::ready`.
    Pinned(Vec<Doc>, u32),
    /// Large: never read whole; the prompt names them (`large_note`) and ask_docs reads the parts
    /// a question matches. `fits_pages` is how many pages of these documents the device can hold,
    /// when the documents have pages: the line "Fits up to N pages on this computer".
    Large {
        /// The documents.
        docs: Vec<Doc>,
        /// Pages that fit, by these documents' tokens a page.
        fits_pages: Option<u32>,
    },
}

/// The room a turn keeps beside a plain-text set in a loaded `context`: `TURN_ROOM`, the documents
/// plan's, where the context is big enough for it to leave a quarter, and three quarters of a
/// smaller context. At the default 8192 a note of up to 2,048 tokens goes in as text, not through
/// a reload into the documents plan (#590 F2); from about 22k on the room is the plan's own.
pub fn small_room(context: u32) -> u32 {
    TURN_ROOM.min(context / 4 * 3)
}

/// The size of `docs` against `context`, the tokens of the context loaded now, and the device.
/// `count` is the tokens of a text (`Larch::token_ids`) and `fit` is `Larch::document_fit`, for
/// a number of tokens.
pub fn route(
    docs: Vec<Doc>,
    context: u32,
    count: impl Fn(&str) -> u32,
    fit: impl Fn(u32) -> larch::DocFit,
) -> Route {
    if docs.is_empty() {
        return Route::Large {
            docs,
            fits_pages: None,
        };
    }
    let tokens = count(&block(&docs));
    let room = small_room(context);
    if tokens.saturating_add(room) <= context {
        return Route::Small(docs);
    }
    let fit = fit(tokens);
    if fit.fits {
        return Route::Pinned(docs, fit.host_layers);
    }
    let pages = u64::from(pages(&docs));
    // The loaded context holds a shorter set as plain text even where the plan holds none.
    let most = fit.max_tokens.max(context.saturating_sub(room));
    let fits_pages = (pages > 0 && tokens > 0)
        .then(|| u32::try_from(u64::from(most) * pages / u64::from(tokens)).unwrap_or(u32::MAX));
    Route::Large { docs, fits_pages }
}

/// Seconds a first read may take: a set the device holds but reads slower than this goes to
/// ask_docs instead of a pin (#590 F8).
pub const READ_SECS: f64 = 120.0;

/// `fit`, the device's answer for `tokens`, narrowed to what reads in `READ_SECS` at `rate`, the
/// tokens a second of this model's last first read (`cache::read_rate`). With no rate yet, `fit`.
// ponytail: linear in tokens, though attention grows with length, so a set much longer than the
// one measured reads slower than this says; scale by length if reads past ~3 min show up.
pub fn in_time(mut fit: larch::DocFit, tokens: u32, rate: Option<f64>) -> larch::DocFit {
    if let Some(rate) = rate {
        // `as` saturates: a rate past u32 tokens in the time is no cap. Rounded, since a rate
        // of `tokens / READ_SECS` times `READ_SECS` can come back a hair under `tokens`.
        let most = (rate * READ_SECS).round() as u32;
        fit.fits &= tokens <= most;
        fit.max_tokens = fit.max_tokens.min(most);
    }
    fit
}

/// What the document chip shows (`AgentEvent::Documents`), and the follow-up's speed.
#[derive(Clone, Debug, PartialEq)]
pub enum Documents {
    /// The engine reloads to hold the documents, which takes about `secs` seconds (0: not
    /// measured yet).
    MakingRoom {
        /// `Larch::load_secs`: how long the last load took.
        secs: u32,
    },
    /// The engine has read `done` of `total` tokens of the documents' `pages` (0 for documents
    /// with no pages).
    Reading {
        /// Tokens read.
        done: u32,
        /// Tokens to read.
        total: u32,
        /// `pages(docs)`.
        pages: u32,
    },
    /// Read, and kept: a question costs only its own tokens.
    Ready,
    /// `Ready`, with part of the model in host memory to make room: other chats may be slower.
    FitsSlower {
        /// Layers in host memory (`larch::DocFit::host_layers`).
        host_layers: u32,
    },
    /// The documents do not fit and ask_docs searches them: the pages that would fit, from
    /// `Route::Retrieval`.
    FitsUpTo {
        /// Pages the device can hold.
        pages: u32,
    },
    /// A question was answered: the time to first token and the prompt tokens served from cache.
    Answered {
        /// Time to first token in milliseconds.
        ttft_ms: f64,
        /// Prompt tokens the engine did not read again.
        cached_tokens: u32,
    },
}

impl Documents {
    /// `Ready`, or `FitsSlower` when the plan put `host_layers` layers in host memory.
    pub fn ready(host_layers: u32) -> Documents {
        match host_layers {
            0 => Documents::Ready,
            host_layers => Documents::FitsSlower { host_layers },
        }
    }

    /// `Reading`, or `Ready` once `done` reaches `total`.
    pub fn reading(done: u32, total: u32, pages: u32) -> Documents {
        match done >= total {
            true => Documents::Ready,
            false => Documents::Reading { done, total, pages },
        }
    }
}

// ponytail: the page is by tokens, as if each page had as many; a page of figures' captions
// reads as a short one. Map pages to token offsets when the chip is seen to be off
impl std::fmt::Display for Documents {
    /// The chip's text: "Reading p. 40 of 120" (a percentage without pages), "Ready", "Fits up
    /// to 80 pages on this computer", "First word in 1.2 s".
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Documents::Reading {
                done,
                total,
                pages: 0,
            } => {
                write!(
                    f,
                    "Reading {}%",
                    u64::from(done) * 100 / u64::from(total.max(1))
                )
            }
            Documents::Reading { done, total, pages } => {
                let page = (u64::from(done) * u64::from(pages)).div_ceil(u64::from(total.max(1)));
                write!(f, "Reading p. {} of {pages}", page.max(1))
            }
            Documents::MakingRoom { secs: 0 } => f.write_str("Making room for the document"),
            Documents::MakingRoom { secs } => {
                write!(f, "Making room for the document (about {secs} s)")
            }
            Documents::Ready => f.write_str("Ready"),
            Documents::FitsSlower { .. } => f.write_str("Ready; other chats may be slower"),
            Documents::FitsUpTo { pages } => write!(f, "Fits up to {pages} pages on this computer"),
            Documents::Answered { ttft_ms, .. } => {
                write!(f, "First word in {:.1} s", ttft_ms / 1e3)
            }
        }
    }
}

/// The messages of a turn on `docs`: the system prompt (`system`, the Folio block then `plain`,
/// the agent's own text), the chat's `history` (earlier questions and answers, whatever system
/// message it holds left out), then the new `question`. The prefix up to the end of the documents
/// is the same for every turn, so the engine's cache of it serves them all; the pin's ids are the
/// common prefix of two such renderings.
pub fn messages(
    docs: &[Doc],
    plain: &str,
    history: &[ChatMessage],
    question: &str,
) -> Vec<ChatMessage> {
    let message = |role, content: &str| ChatMessage {
        role,
        content: content.into(),
        ..ChatMessage::default()
    };
    let mut out = vec![message(Role::System, &system(docs, plain))];
    out.extend(history.iter().filter(|m| m.role != Role::System).cloned());
    out.push(message(Role::User, question));
    out
}

/// The notes under an answer to a question on `docs`: that it stopped at the token limit when
/// `cut`, one for each quote not on the page it cites (`cite::check`), and one for the numbers no
/// document has (`cite::numbers`). The answer is never rewritten.
pub fn notes(answer: &str, docs: &[Doc], cut: bool) -> Vec<String> {
    let mut notes = Vec::new();
    if cut {
        notes.push("_The answer stops at the token limit._".into());
    }
    notes.extend(cite::check(answer, docs, &map::labels(docs)).notes);
    notes.extend(cite::numbers(answer, docs));
    notes
}

/// `answer` as the chat shows and keeps it: its own text, then its notes (`notes`).
pub fn checked(answer: &str, docs: &[Doc], cut: bool) -> String {
    std::iter::once(answer.trim().to_owned())
        .chain(notes(answer, docs, cut))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// `history` (the system message first) without its oldest exchanges when the rest of the turn,
/// `extra` tokens (the agent's own system text and the new message), and what `history` holds
/// after the system message pass `room`: whole exchanges from the oldest, never the last one, and
/// never the system message, whose Folio block is the pinned prefix. `None` when it fits. `count`
/// is the tokens of a text.
pub fn trim(
    history: &[ChatMessage],
    extra: u32,
    room: u32,
    count: &dyn Fn(&str) -> u32,
) -> Option<Vec<ChatMessage>> {
    let weight = |m: &ChatMessage| {
        let calls: u32 = (m.tool_calls.iter())
            .map(|c| count(&c.name).saturating_add(count(&c.arguments.to_string())))
            .sum();
        count(&m.content).saturating_add(calls)
    };
    let head = usize::from(history.first().is_some_and(|m| m.role == Role::System));
    // Each exchange starts at a user message.
    let mut starts: Vec<usize> = (head..history.len())
        .filter(|&i| history[i].role == Role::User)
        .collect();
    let mut left: u32 = history[head..]
        .iter()
        .map(weight)
        .fold(extra, u32::saturating_add);
    let mut from = head;
    while left > room && starts.len() > 1 {
        let next = starts[1];
        left = history[from..next]
            .iter()
            .map(weight)
            .fold(left, u32::saturating_sub);
        from = next;
        starts.remove(0);
    }
    (from > head).then(|| {
        let mut kept = history[..head].to_vec();
        kept.extend_from_slice(&history[from..]);
        kept
    })
}

/// What a pinned turn gives back.
#[derive(Clone, Debug, PartialEq)]
pub struct Pinned {
    /// The answer in Markdown, its notes under it (`checked`).
    pub markdown: String,
    /// What each quote of the answer came to.
    pub results: Vec<cite::Verdict>,
    /// The turn's timings: time to first token, and the prompt tokens served from cache.
    pub timings: Timings,
}

/// One turn on `docs` for a caller that is no chat turn (a skill): the Folio block alone as the
/// system prompt, then `history`, then `question`, with thinking off, its quotes checked. A chat's
/// own turn is an ordinary agent turn whose system prompt starts with the same block
/// (`Agent::pin_documents`) and whose answers pass the same check. `family` is the
/// model's tool-call family, which says whether its request has a thinking field to turn off.
pub fn run(
    larch: &Larch,
    ctx: &ToolCtx,
    family: Option<Family>,
    docs: &[Doc],
    history: &[ChatMessage],
    question: &str,
    stop: &AtomicBool,
) -> Result<Pinned, SkillError> {
    let mut log = Log::new(ctx, NAME);
    let start = std::time::Instant::now();
    let system = block(docs);
    let opts = SessionOptions {
        system: Some(system.clone()),
        // One turn, then dropped: the engine's cache of the prefix serves the next.
        keep_kv: false,
        ..SessionOptions::default()
    };
    let mut session = larch.session(opts.clone());
    let id = session.id();
    if !history.is_empty() {
        let mut all = messages(docs, "", history, "");
        all.pop();
        session.restore(opts, all).map_err(SkillError::Engine)?;
    }
    // Thinking off: a lookup in the text needs none, and its tokens would delay the first word.
    let thinking = Thinking {
        budget: 0,
        ..Thinking::default()
    };
    let p = ask_docs::answer_params(family, thinking, crate::namespace(&ctx.data_dir));
    let reply = SeatAdmission::of(larch).seat(p.class, stop, || session.send(question, p));
    let a = reply.and_then(|s| answer(id, s, stop)).or_else(|e| {
        log.write(ask_docs::failed("answer", id, start, &e))?;
        Err(SkillError::Engine(e))
    })?;
    log.write(ask_docs::turn("answer", &a, None))?;
    Ok(Pinned {
        markdown: checked(&a.text, docs, a.finish == Finish::Length),
        results: cite::check(&a.text, docs, &map::labels(docs)).results,
        timings: a.timings,
    })
}

/// The system prompt's line for `docs` too long to hold whole (`Route::Large`): their text is not
/// in the prompt, so it names them and sends a question about them to the tool that can read
/// them. ask_docs reads only attached files, so a set from links goes to read_papers with the
/// links (#590 F5), which a chat with nothing attached still has.
pub fn large_note(docs: &[Doc]) -> String {
    let listed: Vec<String> = (docs.iter())
        .map(|d| {
            let n = d.pages.len();
            let pages = if n == 1 { "page" } else { "pages" };
            format!("{}. {}, {n} {pages}", d.letter(), d.name)
        })
        .collect();
    let is_link = |d: &&Doc| d.name.starts_with("http://") || d.name.starts_with("https://");
    let links: Vec<&str> = docs
        .iter()
        .filter(is_link)
        .map(|d| d.name.as_str())
        .collect();
    let how = match (links.len(), docs.len()) {
        (0, _) => "call ask_docs with the question, and in search the terms the documents likely \
                   use: it reads the parts that match"
            .to_owned(),
        (l, n) if l == n => format!(
            "call read_papers with papers {} and the question as focus: it reads them for it",
            links.join(", ")
        ),
        _ => format!(
            "call ask_docs with the question, and in search the terms the documents likely use, \
             for the attached files, and read_papers with papers {} and the question as focus \
             for the links",
            links.join(", ")
        ),
    };
    // ask_docs reads the parts a question matches; "read through this report" asked for all of
    // it, and got two small sections and no page (Mac, 24 Nov). The model picks.
    let whole = match links.len() < docs.len() {
        true => {
            " To read an attached document through or summarize it, call read_papers with its \
                 name and the question as focus: it reads all of it, in parts."
        }
        false => "",
    };
    format!(
        // "Their text is not here" alone read as nothing attached: the model told a user with a
        // PDF attached "I don't see any file attached" (Mac, 24 Nov).
        "These documents are attached to this chat: {}. They are too long for this computer to \
         hold whole, so their text is not in this prompt. To answer a question about them, \
         {how}.{whole} An earlier summary may say where to look, but quote a figure as the \
         documents' only from what those tools read this turn, never from memory, and say when \
         an answer comes from only the parts the tools read, not all of the documents.",
        listed.join("; ")
    )
}

/// The web links in the user's own message `text`, without the punctuation a sentence puts after
/// them: each is a document when it is a PDF, a Drive or Dropbox share link (`is_document_link`),
/// or an article, which `store_url` decides by what text the page has.
pub fn links(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        let url = word.trim_matches(|c: char| "()<>[]\"'.,;:!?".contains(c));
        let web = url.starts_with("http://") || url.starts_with("https://");
        if web && url.len() > "https://".len() && !out.iter().any(|u| u == url) {
            out.push(url.to_owned());
        }
    }
    out
}

/// Whether `url` is a PDF or a share link: a document whose failure to read is worth a notice,
/// where a failed article link is left to the normal tools.
pub fn is_document_link(url: &str) -> bool {
    let path = url.split(['?', '#']).next().unwrap_or_default();
    path.to_ascii_lowercase().ends_with(".pdf") || read_papers::direct(url).is_some()
}

/// The sha-256 of the file at `file`.
pub fn sha(file: &Path) -> io::Result<String> {
    Ok(larch::bench::hex(&larch::bench::sha256(&fs::read(file)?)))
}

/// Deletes what the chat kept of the documents under `path` (a file or folder taken off a chat,
/// or each one of a deleted chat's) unless some chat still attaches the same file: the extract
/// cache's text of each (`source::forget`) and the pin files read from it
/// (`folio::cache::forget`). Call it after the attachment is saved or the chat's own list is
/// deleted, so `<data>/chat/*.attached.json` holds what is still attached; a deleted chat's
/// source record goes with `source::forget_chat`. A document that cannot be read is skipped; the
/// user's files are never touched. Returns how many documents' copies went.
pub fn forget(data: &Path, path: &Path) -> io::Result<usize> {
    let ctx = |paths: Vec<PathBuf>| ToolCtx {
        allowed: paths,
        data_dir: data.to_path_buf(),
        source: String::new(),
    };
    let files = |paths: Vec<PathBuf>| -> Vec<PathBuf> {
        let docs = ask_docs::documents(&ctx(paths)).unwrap_or_default();
        docs.into_iter().map(|(_, f)| f).collect()
    };
    let kept = files(attached(data));
    let mut gone = Vec::new();
    for file in files(vec![path.to_path_buf()]) {
        if !kept.contains(&file) {
            gone.extend(source::forget(data, &file)?);
        }
    }
    gone.sort();
    gone.dedup();
    cache::forget(&cache::dir(data), &gone)?;
    Ok(gone.len())
}

/// Every path attached to any chat: the app's `<data>/chat/<id>.attached.json` files.
fn attached(data: &Path) -> Vec<PathBuf> {
    let Ok(dir) = fs::read_dir(data.join("chat")) else {
        return Vec::new();
    };
    (dir.flatten())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".attached.json"))
        .filter_map(|e| serde_json::from_slice::<Vec<PathBuf>>(&fs::read(e.path()).ok()?).ok())
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    /// A turn on documents thinks `DOC_THINKING` whatever the chat's budget, and its answer has
    /// the rest of the room, or less when the chat caps it.
    #[test]
    fn a_turn_on_documents_thinks_in_the_answers_room() {
        let t = |budget, answer_max_tokens| Thinking {
            budget,
            answer_max_tokens,
        };
        assert_eq!(
            thinking(t(100, 0)),
            t(DOC_THINKING, ANSWER_ROOM - DOC_THINKING)
        );
        assert_eq!(thinking(t(0, 0)), t(0, ANSWER_ROOM), "off stays off");
        assert_eq!(
            thinking(t(1 << 20, 0)),
            t(DOC_THINKING, ANSWER_ROOM - DOC_THINKING),
            "a big budget stays short"
        );
        assert_eq!(thinking(t(100, 500)), t(DOC_THINKING, 500));
        let room = ANSWER_ROOM - DOC_THINKING;
        assert_eq!(thinking(t(100, room)), t(DOC_THINKING, room));
        assert_eq!(
            thinking(t(100, ANSWER_ROOM)),
            t(DOC_THINKING, room),
            "a cap past the room"
        );
    }

    use super::*;

    fn doc_at(index: usize, name: &str, text: &str) -> Doc {
        source::doc(
            index,
            name,
            &format!("sha{index}{name}"),
            Rep::Text,
            text.into(),
        )
    }

    fn doc(name: &str, text: &str) -> Doc {
        doc_at(0, name, text)
    }

    /// Three pages, pdftotext's way: a form feed after each page.
    fn pdf() -> Doc {
        let text = "one\ntwo\n\u{c}three\nfour\n\u{c}five\n";
        source::doc(0, "a.pdf", "sha", Rep::Pdf, source::between(text).into())
    }

    /// A repo is not a document set for its README, and a file that gives no text is left out by
    /// name while the others stay documents (#590 F6).
    #[test]
    fn a_repo_is_no_document_set_and_a_scan_leaves_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let (repo, papers, data) = (
            dir.path().join("repo"),
            dir.path().join("papers"),
            dir.path().join("data"),
        );
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(repo.join("README.md"), "A repo.\n").unwrap();
        fs::write(repo.join("Cargo.toml"), "[package]\n").unwrap();
        fs::write(repo.join("src/main.rs"), "fn main() {}\n").unwrap();
        fs::create_dir_all(&papers).unwrap();
        fs::write(papers.join("good.md"), "Rent is $5.\n").unwrap();
        fs::write(papers.join("scan.pdf"), b"not a pdf").unwrap();
        let ctx = |allowed: PathBuf| ToolCtx {
            allowed: vec![allowed],
            data_dir: data.clone(),
            source: String::new(),
        };
        let stop = AtomicBool::new(false);
        let (docs, skipped) = load(&ctx(repo), "c", &stop).unwrap();
        assert!(docs.is_none() && skipped.is_empty());
        let (docs, skipped) = load(&ctx(papers), "c", &stop).unwrap();
        let names: Vec<_> = docs.unwrap().into_iter().map(|d| d.name).collect();
        assert_eq!(names, ["good.md"]);
        assert_eq!(skipped, [("scan.pdf".to_owned(), NO_TEXT_LEFT_OUT)]);
    }

    /// A large set names the tool that can read it: ask_docs for attached files, and read_papers
    /// to read one through, read_papers with the links for documents from links, which ask_docs
    /// cannot reach.
    #[test]
    fn a_large_set_is_sent_to_the_tool_that_can_read_it() {
        let file = doc_at(0, "manual.md", "text");
        let link = doc_at(1, "https://a.org/p.pdf", "text");
        // Attached files: ask_docs for a question, read_papers to read one through.
        let files = large_note(std::slice::from_ref(&file));
        assert!(files.contains("call ask_docs with the question"), "{files}");
        assert!(
            files.contains("To read an attached document through"),
            "{files}"
        );
        let links = large_note(std::slice::from_ref(&link));
        assert!(links.contains("read_papers with papers https://a.org/p.pdf"));
        assert!(!links.contains("ask_docs"));
        let both = large_note(&[file, link]);
        assert!(both.contains("ask_docs") && both.contains("read_papers"));
    }

    fn say(role: Role, content: &str) -> ChatMessage {
        ChatMessage {
            role,
            content: content.into(),
            ..ChatMessage::default()
        }
    }

    #[test]
    fn a_byte_is_on_its_page_and_never_on_an_empty_one() {
        let d = Doc {
            pages: [(0, 10), (10, 10), (10, 20)]
                .map(|(start, end)| Page {
                    start,
                    end,
                    label: None,
                })
                .to_vec(),
            ..doc("a.pdf", &"x".repeat(20))
        };
        let on = |at| d.page_of(at);
        assert_eq!((on(0), on(9), on(10), on(19)), (0, 0, 2, 2));
    }

    #[test]
    fn letters_run_a_to_z_then_aa() {
        let l = |i| doc_at(i, "a", "x\n").letter();
        assert_eq!(
            (l(0), l(1), l(25), l(26), l(27)),
            ("A".into(), "B".into(), "Z".into(), "AA".into(), "AB".into())
        );
    }

    #[test]
    fn the_block_marks_each_page_and_leaves_out_one_with_no_text() {
        let d = source::doc(1, "n.pdf", "s", Rep::Pdf, "one\n\u{c}\u{c}three\n".into());
        let b = block(&[d]);
        assert!(
            b.contains("\nDocument B: n.pdf\n[p. 1]\none\n[p. 3]\nthree\n"),
            "{b}"
        );
        assert!(!b.contains("[p. 2]"), "{b}");
        assert_eq!(pages(&[pdf()]), 3);
    }

    #[test]
    fn a_stored_file_keeps_its_letter_and_a_second_chat_has_its_own() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path();
        let (a, b) = (dir.path().join("a.md"), dir.path().join("b.md"));
        fs::write(&a, "alpha\n").unwrap();
        fs::write(&b, "beta\n").unwrap();
        let stop = AtomicBool::new(false);
        let at = |chat, name: &str, f| store(data, chat, name, f, &stop).unwrap().unwrap();
        assert_eq!(
            (at("c1", "a.md", &a).index, at("c1", "b.md", &b).index),
            (0, 1)
        );
        // Read again, in any order: the same letters.
        assert_eq!(
            (at("c1", "b.md", &b).letter(), at("c1", "a.md", &a).letter()),
            ("B".into(), "A".into())
        );
        assert_eq!(at("c2", "b.md", &b).letter(), "A");
        // The text is in the extract cache, with no form feed in it.
        assert_eq!(at("c1", "a.md", &a).text, "alpha\n");
    }

    #[test]
    fn an_unchanged_file_is_opened_from_the_cache_and_a_changed_one_is_read_again() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.md");
        fs::write(&file, "alpha\n").unwrap();
        let stop = AtomicBool::new(false);
        let at = || {
            store(dir.path(), "c", "a.md", &file, &stop)
                .unwrap()
                .unwrap()
        };
        assert_eq!(at().text, "alpha\n");
        // The extractor's text is replaced in the cache: a second turn that read the file again
        // would not show it.
        let stamp = source::Stamp::of(&fs::metadata(&file).unwrap()).unwrap();
        source::save(dir.path(), &file, stamp, "cached\n").unwrap();
        assert_eq!(at().text, "cached\n");
        // Another size and change time is another file: read again, under the same letter.
        fs::write(&file, "beta beta\n").unwrap();
        let again = at();
        assert_eq!((again.text.as_str(), again.index), ("beta beta\n", 0));
    }

    #[test]
    fn a_file_with_no_text_is_no_document() {
        let dir = tempfile::tempdir().unwrap();
        let blank = dir.path().join("blank.md");
        fs::write(&blank, "\n \n").unwrap();
        let stop = AtomicBool::new(false);
        assert_eq!(
            store(dir.path(), "c", "blank.md", &blank, &stop).unwrap(),
            None
        );
    }

    #[test]
    fn the_block_is_the_same_bytes_for_every_turn_whatever_the_date_or_the_memory() {
        let docs = [pdf(), doc_at(1, "n.md", "alpha\n")];
        let block = block(&docs);
        // Nothing of today, of memory or of tools is in it.
        assert!(!block.contains(&crate::run::today()) && !block.contains("Today is"));
        let (a, b) = (
            crate::run::system_prompt("", Some("a saved fact"), "1 Jan 2030", None, ""),
            crate::run::system_prompt("", Some("another"), "2 Jan 2030", None, ""),
        );
        for plain in [&a, &b] {
            let system = system(&docs, plain);
            assert!(system.starts_with(&block) && system.ends_with(plain.as_str()));
        }
        assert_eq!(
            messages(&docs, &a, &[], "who?")[0].content[..block.len()],
            block
        );
        assert_eq!(
            messages(&docs, &b, &[], "when?")[0].content[..block.len()],
            block
        );
    }

    #[test]
    fn a_second_turn_is_the_system_text_then_the_first_exchange_then_the_new_question() {
        let docs = [doc("n.md", "alpha\n")];
        let first = messages(&docs, "plain", &[], "Q1");
        let history = [
            say(Role::System, "an ordinary system prompt"),
            say(Role::User, "Q1"),
            say(Role::Assistant, "A1"),
        ];
        let second = messages(&docs, "plain", &history, "Q2");
        let roles: Vec<_> = second
            .iter()
            .map(|m| (m.role, m.content.as_str()))
            .collect();
        assert_eq!(
            roles,
            [
                (Role::System, system(&docs, "plain").as_str()),
                (Role::User, "Q1"),
                (Role::Assistant, "A1"),
                (Role::User, "Q2"),
            ]
        );
        assert_eq!(second[0], first[0]);
    }

    /// The paths compared are the real ones `ask_docs::documents` gives, whichever way a chat
    /// named the file: here one chat attaches the files through a link to their folder (as
    /// macOS's `/var` and `/tmp` are) and the other by the folder's own path (#590 Mac review).
    #[test]
    fn forgetting_a_document_deletes_its_text_and_pins_unless_another_chat_holds_it() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        fs::create_dir_all(data.join("chat")).unwrap();
        let real = dir.path().join("real");
        fs::create_dir_all(&real).unwrap();
        #[cfg(unix)]
        let linked = {
            let linked = dir.path().join("linked");
            std::os::unix::fs::symlink(&real, &linked).unwrap();
            linked
        };
        #[cfg(not(unix))]
        let linked = real.clone();
        fs::write(real.join("a.md"), "alpha\n").unwrap();
        fs::write(real.join("b.md"), "beta\n").unwrap();
        let (a, b) = (linked.join("a.md"), linked.join("b.md"));
        let stop = AtomicBool::new(false);
        // Stored as a turn stores them (`load`): under the path each resolves to.
        let ctx = ToolCtx {
            allowed: vec![a.clone(), b.clone()],
            data_dir: data.clone(),
            source: String::new(),
        };
        let Ok((Some(docs), _)) = load(&ctx, "c", &stop) else {
            panic!("no documents");
        };
        let [da, db] = &docs[..] else {
            panic!("{} documents", docs.len());
        };
        let cache = cache::dir(&data);
        for (key, doc) in [("a.pin", da), ("b.pin", db)] {
            fs::create_dir_all(&cache).unwrap();
            fs::write(cache.join(key), "pin").unwrap();
            fs::write(cache.join(key).with_extension("state"), "s").unwrap();
            cache::tag(&cache, key, std::slice::from_ref(&doc.sha256)).unwrap();
        }
        // Another chat still attaches b.md, by its folder's own path.
        fs::write(
            data.join("chat/other.attached.json"),
            serde_json::to_vec(&[real.join("b.md")]).unwrap(),
        )
        .unwrap();
        // The extract cache is keyed by the real path, as `store` was given it.
        let extract = |f: &Path| {
            let stamp = source::Stamp::of(&fs::metadata(f).unwrap()).unwrap();
            source::cached(&data, &fs::canonicalize(f).unwrap(), stamp)
        };
        assert_eq!(forget(&data, &a).unwrap(), 1);
        assert_eq!(forget(&data, &b).unwrap(), 0);
        assert!(extract(&a).is_none() && extract(&b).is_some());
        assert!(!cache.join("a.pin").exists() && !cache.join("a.state").exists());
        assert!(cache.join("b.pin").exists() && cache.join("b.state").exists());
        // The user's files are never touched, and the chat's letters stay.
        assert!(a.exists() && b.exists());
        assert_eq!(source::load(&data, "c").unwrap().len(), 2);
    }

    /// A page starts with its mark on a line of its own, and a paragraph repeats it once
    /// `MARK_EVERY` bytes passed since the last; a page with no text gets none.
    #[test]
    fn a_long_page_repeats_its_mark() {
        let long = "x".repeat(MARK_EVERY);
        let text = format!("Intro.\n\n{long}\n\nShort.\n\nNext.\nTop of two.\n\nMore.\n");
        let mut d = doc("a.pdf", &text);
        let (cut, len) = (d.text.find("Top").unwrap(), d.text.len());
        d.pages = vec![
            Page {
                start: 0,
                end: cut,
                label: None,
            },
            Page {
                start: cut,
                end: cut,
                label: None,
            },
            Page {
                start: cut,
                end: len,
                label: None,
            },
        ];
        assert_eq!(
            render(&[d]),
            format!(
                "\nDocument A: a.pdf\n[p. 1]\nIntro.\n\n{long}\n\n[p. 1] Short.\n\nNext.\n\
                 [p. 3]\nTop of two.\n\nMore.\n"
            )
        );
    }

    #[test]
    fn a_set_is_small_medium_or_large_by_the_context_and_the_device() {
        let docs = vec![pdf()];
        // A thousand tokens a byte: the set is long enough that the room is `TURN_ROOM`.
        let count = |t: &str| u32::try_from(t.len()).unwrap() * 1000;
        let tokens = count(&block(&docs));
        let fit = |fits, max_tokens, host_layers| larch::DocFit {
            fits,
            max_tokens,
            host_layers,
            seats: 2,
        };
        // Room to spare in the context loaded: plain text, and the device is not asked.
        let loaded = tokens + TURN_ROOM;
        assert_eq!(small_room(loaded), TURN_ROOM);
        assert_eq!(
            route(docs.clone(), loaded, count, |_| unreachable!()),
            Route::Small(docs.clone())
        );
        // One token short of it: pinned, with the fit's host layers.
        assert_eq!(
            route(docs.clone(), loaded - 1, count, |t| fit(
                t <= tokens,
                tokens / 2,
                7
            )),
            Route::Pinned(docs.clone(), 7)
        );
        // Half the tokens fit, so half of the 3 pages do, rounded down.
        let large = route(docs.clone(), 0, count, |t| fit(t < tokens, tokens / 2, 0));
        assert_eq!(
            large,
            Route::Large {
                docs,
                fits_pages: Some(1)
            }
        );
        // A plan that holds none (its experts would read on the CPU): the loaded context's pages.
        let context = TURN_ROOM + tokens - 1;
        let none = route(vec![pdf()], context, count, |_| fit(false, 0, 0));
        assert_eq!(
            none,
            Route::Large {
                docs: vec![pdf()],
                fits_pages: Some(2)
            }
        );
    }

    /// A set the device holds but would read for longer than `READ_SECS` is large, and its pages
    /// are the ones that read in time; with no rate measured yet the device alone decides (#590 F8).
    #[test]
    fn a_set_that_reads_too_slowly_is_large() {
        let docs = vec![pdf()];
        let count = |t: &str| u32::try_from(t.len()).unwrap() * 1000;
        let tokens = count(&block(&docs));
        let device = |_| larch::DocFit {
            fits: true,
            max_tokens: tokens * 4,
            host_layers: 3,
            seats: 2,
        };
        let at = |rate: Option<f64>| route(docs.clone(), 0, count, |n| in_time(device(n), n, rate));
        assert_eq!(at(None), Route::Pinned(docs.clone(), 3));
        // Reads in exactly the time: still pinned.
        let just = f64::from(tokens) / READ_SECS;
        assert_eq!(at(Some(just)), Route::Pinned(docs.clone(), 3));
        // Half that speed: half the 3 pages read in time.
        assert_eq!(
            at(Some(just / 2.0)),
            Route::Large {
                docs: docs.clone(),
                fits_pages: Some(1)
            }
        );
    }

    /// At the default 8192 a short note is plain text, not a reload into the documents plan
    /// (#590 F2).
    #[test]
    fn a_note_at_the_default_context_is_plain_text() {
        let note = vec![doc("note.txt", "Rent is $5.")];
        let count = |t: &str| u32::try_from(t.len()).unwrap();
        let held = |_| larch::DocFit {
            fits: true,
            max_tokens: 100_000,
            host_layers: 0,
            seats: 2,
        };
        assert_eq!(route(note.clone(), 8192, count, held), Route::Small(note));
        assert_eq!(small_room(8192), 6144);
    }

    #[test]
    fn the_chip_reads_by_page_and_then_is_ready_or_ready_but_slower() {
        let chip = |d, t, p| Documents::reading(d, t, p).to_string();
        assert_eq!(chip(40_000, 120_000, 120), "Reading p. 40 of 120");
        assert_eq!(chip(1, 120_000, 120), "Reading p. 1 of 120");
        assert_eq!(chip(60, 120, 0), "Reading 50%");
        assert_eq!(chip(120, 120, 120), "Ready");
        let fits = Documents::FitsUpTo { pages: 80 }.to_string();
        assert_eq!(fits, "Fits up to 80 pages on this computer");
        let room = |secs| Documents::MakingRoom { secs }.to_string();
        assert_eq!(room(0), "Making room for the document");
        assert_eq!(room(7), "Making room for the document (about 7 s)");
        assert_eq!(Documents::ready(0), Documents::Ready);
        assert_eq!(
            Documents::ready(12).to_string(),
            "Ready; other chats may be slower"
        );
    }

    #[test]
    fn the_oldest_exchanges_go_first_and_never_the_system_message_or_the_last_exchange() {
        let count = |t: &str| u32::try_from(t.len()).unwrap();
        let h = [
            say(Role::System, &"s".repeat(1000)),
            say(Role::User, "q1"),
            say(Role::Assistant, "a1"),
            say(Role::User, "q2"),
            say(Role::Assistant, "a2"),
            say(Role::User, "q3"),
            say(Role::Assistant, "a3"),
        ];
        // The system message is the documents: its size is no part of the room.
        assert_eq!(trim(&h, 0, 12, &count), None);
        let kept = trim(&h, 2, 8, &count).unwrap();
        let said: Vec<_> = kept.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(said[1..], ["q3", "a3"]);
        assert_eq!(kept[0], h[0]);
        // Even a last exchange over the room stays.
        let kept = trim(&h, 100, 1, &count).unwrap();
        assert_eq!(kept.len(), 3);
    }
}
