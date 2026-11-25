//! The read-papers skill: one reader per paper, all sent at once so the engine batches them, each
//! reading only its own paper and writing a summary of it. Rust reads the files, cuts each paper to
//! the reader's budget (references off; the abstract, the headings, the captions and the key
//! paragraphs kept), and checks every number a summary gives against its paper's text. The summaries are the turn's answer as they are: the
//! readers work alone, and no model step merges them.

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use larch::{Finish, GenParams, Larch, TextEvent};
use serde_json::{Value, json};

use crate::{
    audit::{Audit, AuditEvent},
    folio::{Doc, Rep, ocr, source},
    parse::Family,
    run::{self, Kind},
    skill::SkillError,
    skills::{
        ask_docs::{self, Chunk, ms},
        doc::reader::{Ask, Followup},
    },
    tool::{Registry, ToolCtx, ToolError},
    web::fetch::{self, Seen},
    workers::Workers,
};

/// The skill's name as a chat tool.
pub const TOOL: &str = "read_papers";

/// What the chat's model reads to choose the tool.
pub const TOOL_DESCRIPTION: &str = "Read several papers, long documents or web pages at once, one \
reader per paper, and get a summary of each; a long one is read whole, in parts. Use it to \
summarize, compare or find something across papers, and to read a link the user gives.";

/// `TOOL_DESCRIPTION` when follow-ups are on (`doc_followups`): the tool also answers questions
/// about documents this chat already read, which a model otherwise pages through with read_file.
pub const FOLLOWUP_DESCRIPTION: &str = "Read several papers, long documents or web pages at \
once and get a summary of each, or answer a question about documents this chat already read, \
from their whole text with quotes and pages: give the documents again and the question as focus. \
Use it to summarize, compare or find something across papers, for any question about a \
document already read, and to read a link the user gives.";

/// The `focus` argument's description: the summaries' answer is the chat model's.
const FOCUS: &str = "The user's question about the papers, word for word, when they asked one; \
you answer it from the summaries";

/// `FOCUS` when follow-ups are on: the readers answer from the text, and the user sees that.
const FOCUS_READ: &str = "The user's question about the papers, word for word, when they asked \
one; readers answer it from the papers' own text, and their answer is shown to the user as it is";

/// What a read_file result on a document read_papers read in this chat ends with, when
/// follow-ups are on: the model paged such a file 4 to 7 times (Mac baseline, 13 Nov).
pub const READ_FILE_HINT: &str = "[agent] read_papers read this document in this chat: \
read_papers with it and the question as focus answers from its whole text, with pages.";

/// What the chat's model reads with read_papers' summaries when follow-ups are on: on 11 of 20
/// gate 1 follow-ups it answered from the summaries and called no tool (Mac, 13 Nov).
pub const SUMMARY_NOTE: &str = "[agent] These are short summaries. Details, numbers, dates and \
quotes are only in the documents' own text: for a question about them, call read_papers again \
with the documents and the question as focus.";

/// Papers one call reads, and parts a lone long document is read in.
pub const MAX_PAPERS: usize = 8;

/// The first line of a call's answer when it read one document in parts: the model writes one
/// summary from them, as it answers a focus.
pub const PARTS: &str = "[agent] These are summaries of the parts of one document, in order. \
Write one summary of the whole document from them.";

/// Tokens of a paper a reader sees by default (config.toml's `paper_tokens`): a usual paper's
/// extract, so four readers prefill at once on the M5 (a whole 4.5k-token paper took ~50 s).
pub const PAPER_TOKENS: u32 = 1500;

/// The fewest `paper_tokens`: a page, about what an abstract and a conclusion take.
pub const MIN_PAPER_TOKENS: u32 = 500;

/// Bytes a token takes, as the agent counts everywhere.
pub(crate) const TOKEN_BYTES: usize = 4;

/// The share of a cut paper's budget its start gets; its end, where the conclusion is, gets the
/// rest.
const HEAD_SHARE: f64 = 0.7;

/// The line that stands for a part of a paper a reader does not see.
pub const LEFT_OUT: &str = "[...]";

/// The share of an extract's budget its front (title, authors, abstract) may take.
const FRONT_SHARE: f64 = 0.4;

/// Most of a figure's or a table's caption an extract keeps.
const CAPTION_BYTES: usize = 400;

/// Most of a table's rows, or of a section's first paragraph, an extract keeps.
const PARAGRAPH_BYTES: usize = 600;

/// The shortest cut part worth keeping.
const MIN_PART_BYTES: usize = 80;

/// Paragraphs that best match the user's focus an extract keeps first, after the front.
const FOCUS_PARAGRAPHS: usize = 3;

/// Headings an extract finds below which a text is not a paper's sections; it keeps the start and
/// the end then.
const MIN_HEADINGS: usize = 3;

/// A references heading, lower case: a paper's sources start below it.
pub(crate) const REFERENCES: [&str; 3] = ["references", "bibliography", "works cited"];

/// Sections a heading may name without a number.
const SECTIONS: [&str; 22] = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "discussion",
    "conclusion",
    "conclusions",
    "concluding remarks",
    "limitations",
    "summary",
    "future work",
    "acknowledgments",
    "acknowledgements",
    "appendix",
];

/// Sections a summary needs most after the abstract: the problem, and what came of it.
const KEY_SECTIONS: [&str; 6] = [
    "introduction",
    "discussion",
    "conclusion",
    "conclusions",
    "concluding remarks",
    "limitations",
];

/// Sections that report results: their first two paragraphs come after the key sections'.
const RESULT_SECTIONS: [&str; 4] = [
    "results",
    "experiments",
    "evaluation",
    "experimental results",
];

/// Sections a summary does not need: only their heading is kept.
const SKIPPED_SECTIONS: [&str; 4] = [
    "background",
    "related work",
    "acknowledgments",
    "acknowledgements",
];

/// Tokens of one summary.
const SUMMARY_TOKENS: u32 = 512;

/// The largest file read: a long paper's PDF with its figures is a few tens of MiB.
pub const MAX_FILE_BYTES: u64 = 64 << 20;

/// The character between a PDF's pages in `text_of`'s text: pdftotext's, which every reader here gives.
pub const PAGE_BREAK: char = '\u{c}';

/// Files a folder gives: papers and plain text.
const EXTENSIONS: [&str; 5] = ["pdf", "txt", "md", "tex", "text"];

/// Every reader's system prompt; the user's focus, when given, follows it.
pub const READER_SYSTEM: &str = "You read one research paper, essay, article or other long \
document and summarize it for someone who has not read it. Give its title. For a research paper, \
give the problem it works on, its method, its main results, with their numbers where the text \
gives them, and its limits. For an essay, article or web page, give its main point and its key \
points, in its order, not a paper's parts. Use only what the text \
says: never add a number, name or claim it does not have, and leave out what it does not cover \
instead of saying it is missing. The text \
may be an extract, with [...] where parts are left out: its start, its section headings, its \
table and figure captions and its key paragraphs; summarize what you see. Write at most 200 words, in plain sentences.";

/// The sentence follow-ups (`doc_followups`) add to `READER_SYSTEM`, before its "Use only"
/// (plan 7.14).
const KINDS: &str = "For a slide deck, give its pitch and its key numbers; for a report or a \
datasheet, give its subject and its key figures. ";

/// The skill's settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Papers {
    /// The chat's tool-call family, which sets the readers' parameters.
    pub family: Option<Family>,
    /// Tokens of a paper a reader sees.
    pub paper_tokens: u32,
}

impl Default for Papers {
    fn default() -> Self {
        Self {
            family: None,
            paper_tokens: PAPER_TOKENS,
        }
    }
}

/// A paper as a reader gets it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paper {
    /// The file, as the user's folder names it, or the page's URL; a part of a lone long document
    /// has "(part 2 of 5)" after its name.
    pub file: PathBuf,
    /// The whole text, references and all, which the numbers are checked against.
    pub text: String,
    /// What the reader sees: the text without its references, cut to the budget.
    pub shown: String,
    /// The text was cut to the budget.
    pub cut: bool,
}

/// One reader's work.
#[derive(Clone, Debug, PartialEq)]
pub struct Reader {
    /// The paper's file.
    pub file: PathBuf,
    /// The summary, or why there is none.
    pub summary: Result<String, String>,
    /// Numbers the summary gives that its paper's text does not have.
    pub unchecked: Vec<String>,
    /// Prompt tokens, as the engine counted them; 0 without a reply.
    pub prompt_tokens: u32,
    /// Generated tokens.
    pub completion_tokens: u32,
    /// Time to the first token, ms.
    pub ttft_ms: f64,
    /// Whether the paper was cut to the budget.
    pub cut: bool,
}

/// What a run gives back.
#[derive(Clone, Debug, PartialEq)]
pub struct PapersRun {
    /// The call's answer: each summary under its paper's name.
    pub markdown: String,
    /// Each reader, in the order the papers were given.
    pub readers: Vec<Reader>,
    /// Readers the engine ran at once.
    pub at_once: usize,
    /// The KV pool's tokens, which with `reader_tokens` set `at_once`.
    pub pool_tokens: u64,
    /// The tokens of the longest reader: its prompts and its summary.
    pub reader_tokens: u64,
    /// Each document given, in order, as the chat's source record adds it; `None` for one that
    /// could not be read.
    pub read: Vec<Option<Added>>,
    /// Reading the files, ms.
    pub read_ms: f64,
    /// The readers, from the first request to the last token, ms.
    pub readers_ms: f64,
}

/// The tool's answer to a name that is not a web link in a chat with nothing attached.
pub const NOTHING_ATTACHED: &str = "No files are attached to this chat, so read_papers opens \
only web links. Answer from what you know, or use web_answer";

/// The files `paths` name, in order: a file as it is, a folder as its papers sorted by name, each
/// file once, at most `MAX_PAPERS`; more is an error, so the model hears it and picks.
pub fn gather(ctx: &ToolCtx, paths: &[&str]) -> Result<Vec<PathBuf>, ToolError> {
    // A paper's title with nothing attached is no file: the model hears what to do instead.
    if ctx.allowed.is_empty() {
        return Err(ToolError::Invalid(format!(
            "{NOTHING_ATTACHED} ({})",
            paths.join(", ")
        )));
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for path in paths {
        let real = ctx.resolve(path, false)?;
        let io = |e| ToolError::Io((*path).into(), e);
        if real.is_dir() {
            // Each entry is checked as a path of its own, so a link out of the folder is refused.
            let mut inside = Vec::new();
            for entry in fs::read_dir(&real).map_err(io)?.filter_map(Result::ok) {
                let entry = entry.path();
                if !is_paper(&entry) {
                    continue;
                }
                let entry = ctx.resolve(&entry.to_string_lossy(), false)?;
                if entry.is_file() {
                    inside.push(entry);
                }
            }
            inside.sort();
            files.extend(inside);
        } else {
            // A pipe or a device would block the read: only a plain file is a paper.
            if !fs::metadata(&real).map_err(io)?.is_file() {
                return Err(ToolError::Invalid(format!("{path} is not a file")));
            }
            files.push(real);
        }
    }
    let mut seen = Vec::new();
    files.retain(|f| {
        let new = !seen.contains(f);
        seen.push(f.clone());
        new
    });
    match files.len() {
        0 => Err(ToolError::Invalid(format!(
            "no papers there: give files, or a folder with .{} files",
            EXTENSIONS.join(", .")
        ))),
        n if n > MAX_PAPERS => Err(ToolError::Invalid(format!(
            "{n} papers; one call reads at most {MAX_PAPERS}: give the ones to read"
        ))),
        _ => Ok(files),
    }
}

fn is_paper(path: &Path) -> bool {
    (path.extension().and_then(|e| e.to_str()))
        .is_some_and(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
}

/// The longest `pdftotext` runs on one file: a long paper takes it about a second.
const PDFTOTEXT_LIMIT: Duration = Duration::from_secs(60);

/// The text of `file`: a PDF through poppler's `pdftotext` when it is installed, else the
/// built-in reader, then macOS's PDFKit, each page followed by a form feed as pdftotext gives
/// it; any other file as UTF-8 (lossy). `stop` ends `pdftotext` and PDFKit.
pub fn text_of(file: &Path, stop: &AtomicBool) -> Result<String, String> {
    text_in(file, &bytes_of(file)?, stop)
}

/// `file`'s bytes, at most `MAX_FILE_BYTES`; a longer file is an error.
pub(crate) fn bytes_of(file: &Path) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    // Bounded by what is read, not the metadata, which a growing file outruns.
    (fs::File::open(file).and_then(|f| f.take(MAX_FILE_BYTES + 1).read_to_end(&mut bytes)))
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(format!(
            "the file is over {} MiB, the largest read",
            MAX_FILE_BYTES >> 20
        ));
    }
    Ok(bytes)
}

/// Whether `file`, which holds `bytes`, is a PDF: by its bytes too, as read_file decides, so a
/// PDF saved without its extension is still one.
pub(crate) fn is_pdf(file: &Path, bytes: &[u8]) -> bool {
    bytes.starts_with(b"%PDF-")
        || (file.extension().and_then(|e| e.to_str()))
            .is_some_and(|e| e.eq_ignore_ascii_case("pdf"))
}

/// `text_of` for `file`, whose bytes `bytes_of` read.
// ponytail: the built-in reader runs in this thread and does not see `stop`; its input is at
// most MAX_FILE_BYTES. Move it to a child process if a stop ever waits on it.
pub(crate) fn text_in(file: &Path, bytes: &[u8], stop: &AtomicBool) -> Result<String, String> {
    if !is_pdf(file, bytes) {
        return Ok(String::from_utf8_lossy(bytes).into_owned());
    }
    let mut poppler = || {
        let mut cmd = Command::new("pdftotext");
        crate::tool::no_window(&mut cmd)
            .args(["-q", "-enc", "UTF-8"])
            .arg(file)
            .arg("-");
        output_until(cmd, PDFTOTEXT_LIMIT, stop)
            .map(|out| String::from_utf8_lossy(&out).into_owned())
    };
    // poppler is not installed everywhere, and PDFKit is the Mac's alone: the built-in reader's
    // error is the one that says what is wrong with the file. PDFKit's none is no reason.
    let mut built_in = || pdf_text(bytes);
    let mut mac = || pdfkit(file, stop).ok_or_else(String::new);
    // The model read a refusal as a corrupt file (5 Nov 2026) when every viewer opened it.
    first_text(&mut [&mut poppler, &mut built_in, &mut mac], stop).map_err(|e| {
        match bytes.starts_with(b"%PDF-") {
            true => format!(
                "{e}; the file itself may be fine, as this app's readers refuse some PDFs that \
                 viewers open"
            ),
            false => e,
        }
    })
}

/// The text of the first of `reads`, tried in order, that has any: one that gives none, or
/// fewer letters than pages, missed it as one that fails did (pdf-extract gave no text of
/// NVIDIA's FY2025 10-K, PDFKit 503 KB in 0.5 s, 13 Nov 2026). With none, the first such text,
/// whose pages a scan has; else the last error that gives a reason. `stop` ends the tries at the
/// reader it stopped.
fn first_text(
    reads: &mut [&mut dyn FnMut() -> Result<String, String>],
    stop: &AtomicBool,
) -> Result<String, String> {
    let (mut blank, mut error) = (None, String::new());
    for read in reads {
        match read() {
            Ok(text)
                if text.chars().filter(|c| !c.is_whitespace()).count()
                    >= text.matches('\u{c}').count().max(1) =>
            {
                return Ok(text);
            }
            Ok(text) => {
                blank.get_or_insert(text);
            }
            Err(e) if stop.load(Ordering::Relaxed) => return Err(e),
            Err(e) if !e.is_empty() => error = e,
            Err(_) => {}
        }
    }
    blank.ok_or(error)
}

/// The JavaScript for Automation script that prints a PDF's text by macOS's PDFKit, page by
/// page, each followed by a form feed as pdftotext gives it; a page with no text is an empty one,
/// so the pages after it keep their numbers.
#[cfg(target_os = "macos")]
const PDFKIT: &str = r#"ObjC.import("PDFKit");
function run(argv) {
  var d = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(argv[0]));
  if (d.isNil()) { throw new Error("PDFKit could not open the file"); }
  var out = [];
  for (var i = 0; i < d.pageCount; i++) {
    var s = d.pageAtIndex(i).string;
    out.push(s.isNil() ? "" : s.js + "\n");
  }
  return out.join("\f") + (out.length ? "\f" : "");
}"#;

/// `file`'s text by macOS's PDFKit, which every Mac has, when it reads any: the built-in
/// reader panics on some PDFs that every viewer opens (pdf-extract's `current_point` on a `v`
/// curve with no path, in an arXiv paper's figure, 5 Nov 2026). In a child process, as
/// pdftotext is. It runs after the built-in reader has failed, which can take seconds on a long
/// paper.
// ponytail: the file is an absolute path from every caller, so no "-" can start it
#[cfg(target_os = "macos")]
fn pdfkit(file: &Path, stop: &AtomicBool) -> Option<String> {
    let mut cmd = Command::new("osascript");
    cmd.args(["-l", "JavaScript", "-e", PDFKIT]).arg(file);
    let out = output_until(cmd, PDFTOTEXT_LIMIT, stop).ok()?;
    // osascript ends what `run` returns with a line end of its own, after the last form feed.
    let out = out.strip_suffix(b"\n").unwrap_or(&out);
    let text = String::from_utf8_lossy(out).into_owned();
    (!text.trim().is_empty()).then_some(text)
}

/// No other OS has a PDF reader to ask.
#[cfg(not(target_os = "macos"))]
fn pdfkit(_: &Path, _: &AtomicBool) -> Option<String> {
    None
}

/// `cmd`'s output, at most `MAX_FILE_BYTES`, when it exits with success within `limit`; past
/// the limit, or once `stop` is set, it is killed and this is an error.
pub fn output_until(
    mut cmd: Command,
    limit: Duration,
    stop: &AtomicBool,
) -> Result<Vec<u8>, String> {
    let mut child = (cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null()))
    .spawn()
    .map_err(|e| e.to_string())?;
    let out = child.stdout.take().expect("stdout is piped");
    // Read on a thread of its own: a full pipe would stall the child while it is waited on.
    let reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        out.take(MAX_FILE_BYTES)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let start = Instant::now();
    let ended = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if stop.load(Ordering::Relaxed) => break Err("it was stopped".to_owned()),
            Ok(None) if start.elapsed() > limit => {
                break Err(format!("it ran past {} s", limit.as_secs()));
            }
            Ok(None) => thread::sleep(Duration::from_millis(20)),
            Err(e) => break Err(e.to_string()),
        }
    };
    if ended.is_err() {
        // Its end of the pipe closes with it, so the reader returns.
        let _ = child.kill();
        let _ = child.wait();
    }
    let bytes = reader.join().map_err(|_| "its reader failed".to_owned())?;
    match ended? {
        status if status.success() => bytes.map_err(|e| e.to_string()),
        status => Err(format!("it failed: {status}")),
    }
}

/// A PDF's text by the built-in reader, each page followed by a form feed as pdftotext gives it.
/// Its parser panics on some malformed files; a panic is that file's error, never the chat's.
pub fn pdf_text(bytes: &[u8]) -> Result<String, String> {
    // Page by page, as `extract_text_from_mem` reads them, but a page's error is the file's:
    // `extract_text_from_mem_by_pages` ends quietly at the first one, and the pages shift.
    let pages = || -> Result<String, pdf_extract::OutputError> {
        let mut doc = pdf_extract::Document::load_mem(bytes)?;
        // The load already decrypts a file whose user password is empty (it then holds the
        // state); decrypting again fails AES files on the block length and garbles RC4 ones.
        if doc.is_encrypted() && doc.encryption_state.is_none() {
            doc.decrypt("")?;
        }
        let mut out = String::new();
        for n in doc.get_pages().into_keys() {
            let mut page = String::new();
            pdf_extract::output_doc_page(
                &doc,
                &mut pdf_extract::PlainTextOutput::new(&mut page),
                n,
            )?;
            out.push_str(&page);
            if !page.is_empty() && !page.ends_with('\n') {
                out.push('\n');
            }
            out.push('\u{c}');
        }
        Ok(out)
    };
    match std::panic::catch_unwind(pages) {
        Ok(Ok(text)) => Ok(text),
        Ok(Err(e)) => Err(format!("the PDF could not be read: {e}")),
        Err(_) => Err("the PDF could not be read: its reader failed on this file".into()),
    }
}

/// `text` without its references: everything from the last line that is only a references
/// heading, when that line is in the second half; a paper's appendix after it goes too.
pub fn without_references(text: &str) -> &str {
    let mut at = None;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let word = line
            .trim()
            .trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' ')
            .trim_end_matches(':')
            .to_ascii_lowercase();
        if REFERENCES.contains(&word.as_str()) {
            at = Some(offset);
        }
        offset += line.len();
    }
    match at {
        Some(at) if at >= text.len() / 2 => &text[..at],
        _ => text,
    }
}

/// `text` with runs of blank lines made one and each line's ends trimmed: PDF text has many.
pub fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = false;
    for line in text.lines().map(str::trim) {
        if line.is_empty() {
            blank = !out.is_empty();
            continue;
        }
        if blank {
            out.push('\n');
            blank = false;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// `text` cut to `tokens`: whole when it fits, else its start and its end, at line ends, with
/// `LEFT_OUT` between; and whether it was cut.
pub fn fit(text: &str, tokens: u32) -> (String, bool) {
    let budget = (tokens as usize).saturating_mul(TOKEN_BYTES);
    if text.len() <= budget {
        return (text.to_owned(), false);
    }
    let head = (budget as f64 * HEAD_SHARE) as usize;
    let tail = budget.saturating_sub(head);
    let head_end = text[..text.floor_char_boundary(head)]
        .rfind('\n')
        .map_or(text.floor_char_boundary(head), |i| i + 1);
    let tail_start = text.ceil_char_boundary(text.len() - tail);
    // To the next line's start, unless that is the end: a long last line keeps its end.
    let tail_start = (text[tail_start..].find('\n'))
        .map(|i| tail_start + i + 1)
        .filter(|&start| start < text.len())
        .unwrap_or(tail_start);
    let tail_start = tail_start.max(head_end);
    (
        format!("{}{LEFT_OUT}\n{}", &text[..head_end], &text[tail_start..]),
        true,
    )
}

/// What a block of a paper is, which sets what an extract keeps first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    /// Before the first heading, or under "Abstract": title, authors, abstract.
    Front,
    Heading,
    Caption,
    /// The rows after a table's caption.
    Table,
    Body,
}

/// A paragraph, heading or caption of a paper, with its section's name and its place there.
pub(crate) struct Block<'a> {
    /// Its text, to its last line's end.
    pub(crate) text: &'a str,
    /// Where it starts in the text.
    pub(crate) start: usize,
    /// What it is.
    pub(crate) part: Part,
    section: String,
    nth: usize,
}

/// The section a line heads, lower case and without its number: "3.2 Results" and "RESULTS"
/// head "results"; a numbered line heads one only when it is short, starts with a capital and
/// does not end a sentence.
pub(crate) fn heading(line: &str) -> Option<String> {
    let line = line.trim().trim_end_matches(':');
    if line.len() > 80 || line.split_whitespace().count() > 12 || line.ends_with(['.', ',', ';']) {
        return None;
    }
    let (number, name) = match line.split_once(' ') {
        Some((n, rest))
            if n.trim_end_matches('.').split('.').all(|d| {
                !d.is_empty()
                    && (d.bytes().all(|b| b.is_ascii_digit())
                        || d.bytes().all(|b| b"IVXL".contains(&b))
                        || (d.len() == 1 && d.bytes().all(|b| b.is_ascii_uppercase())))
            }) =>
        {
            (true, rest.trim())
        }
        _ => (false, line),
    };
    let lower = name.to_lowercase();
    let first_word_is_a_word = name
        .split_whitespace()
        .next()
        .is_some_and(|w| w.chars().filter(|c| c.is_alphabetic()).count() >= 2);
    let numbered = number && name.starts_with(|c: char| c.is_uppercase()) && first_word_is_a_word;
    (numbered || SECTIONS.contains(&lower.as_str())).then_some(lower)
}

/// Whether a line starts a caption: "Table 2: ...", "Figure 3. ...", "Fig. 1 ...", "TABLE IV".
fn caption(line: &str) -> bool {
    let lower = line.to_lowercase();
    let Some(rest) = ["table ", "figure ", "fig. "]
        .iter()
        .find_map(|p| lower.strip_prefix(p))
    else {
        return false;
    };
    let label = rest.trim_start();
    let end = label
        .find(|c: char| !(c.is_ascii_digit() || "ivxl".contains(c)))
        .unwrap_or(label.len());
    // "Table 2 shows ..." is a sentence about the table, not its caption.
    end > 0 && matches!(label[end..].chars().next(), None | Some(':' | '.' | '|'))
}

/// `text`'s blocks: paragraphs, each heading and each caption on its own, the appendix too.
pub(crate) fn blocks(text: &str) -> Vec<Block<'_>> {
    let mut out: Vec<Block> = Vec::new();
    let mut section = String::new();
    let mut nth = 0;
    let mut headed = false;
    let mut open: Option<(usize, Part)> = None;
    let mut offset = 0;
    fn close<'a>(
        text: &'a str,
        out: &mut Vec<Block<'a>>,
        open: &mut Option<(usize, Part)>,
        end: usize,
        section: &str,
        nth: &mut usize,
    ) {
        if let Some((start, part)) = open.take() {
            let block = &text[start..end];
            let part = match part {
                // Rows mostly of numbers right after a table's caption are its table.
                Part::Body
                    if out.last().is_some_and(|b| {
                        b.part == Part::Caption
                            && b.text
                                .get(..5)
                                .is_some_and(|t| t.eq_ignore_ascii_case("table"))
                    }) && block.bytes().filter(u8::is_ascii_digit).count() * 4
                        >= block.bytes().filter(|b| !b.is_ascii_whitespace()).count() =>
                {
                    Part::Table
                }
                part => part,
            };
            out.push(Block {
                text: block,
                start,
                part,
                section: section.to_owned(),
                nth: *nth,
            });
            if part == Part::Body {
                *nth += 1;
            }
        }
    }
    for line in text.split_inclusive('\n') {
        let end = offset + line.len();
        if line.trim().is_empty() {
            close(text, &mut out, &mut open, offset, &section, &mut nth);
        } else if let Some(name) = heading(line) {
            close(text, &mut out, &mut open, offset, &section, &mut nth);
            headed = true;
            section = name;
            nth = 0;
            out.push(Block {
                text: line,
                start: offset,
                part: Part::Heading,
                section: section.clone(),
                nth,
            });
        } else if caption(line) {
            close(text, &mut out, &mut open, offset, &section, &mut nth);
            open = Some((offset, Part::Caption));
        } else if open.is_none() {
            let front = !headed || section == "abstract";
            open = Some((offset, if front { Part::Front } else { Part::Body }));
        }
        offset = end;
    }
    close(text, &mut out, &mut open, offset, &section, &mut nth);
    out
}

/// `text` cut to `tokens` as an extract: whole when it fits; else its front, every heading and
/// caption, the key sections' first paragraphs, tables, the results' first two paragraphs and
/// each other section's first, in that order while they fit, then more of the key and results
/// sections; related work, acknowledgments and appendices are left out. With a `focus`, the
/// `FOCUS_PARAGRAPHS` paragraphs that match it best (BM25, as ask_docs ranks) come with the
/// headings, wherever they are. Shown in the paper's order with `LEFT_OUT` where text is left
/// out. A text with fewer than `MIN_HEADINGS` headings is cut as `fit` cuts it. And whether it
/// was cut.
pub fn extract(text: &str, tokens: u32, focus: Option<&str>) -> (String, bool) {
    let budget = (tokens as usize).saturating_mul(TOKEN_BYTES);
    if text.len() <= budget {
        return (text.to_owned(), false);
    }
    let mut blocks = blocks(text);
    // An appendix holds proofs and extra tables; a summary stops before it.
    if let Some(at) = (blocks.iter()).position(|b| {
        b.part == Part::Heading && matches!(b.section.as_str(), "appendix" | "appendices")
    }) {
        blocks.truncate(at);
    }
    if blocks.iter().filter(|b| b.part == Part::Heading).count() < MIN_HEADINGS {
        return fit(text, tokens);
    }
    // The question's answer may sit in any section, e.g. a method paragraph the key sections
    // would crowd out.
    let bodies: Vec<usize> = (0..blocks.len())
        .filter(|&i| blocks[i].part == Part::Body)
        .collect();
    let focused: Vec<usize> = match focus {
        Some(focus) => {
            let chunks: Vec<Chunk> = (bodies.iter())
                .map(|&i| Chunk {
                    file: PathBuf::new(),
                    line: 0,
                    heading: Arc::from(vec![blocks[i].section.clone()]),
                    start: blocks[i].start,
                    text: blocks[i].text.to_owned(),
                })
                .collect();
            // An in-memory table that fails leaves the extract as it is without a focus.
            (ask_docs::rank(&chunks, focus, FOCUS_PARAGRAPHS).unwrap_or_default())
                .into_iter()
                .filter(|(_, score)| score.is_some())
                .map(|(c, _)| bodies[c])
                .collect()
        }
        None => Vec::new(),
    };
    // (rank, the most of the block kept); blocks from rank 4 on fill what is left, whole.
    let rank = |i: usize, b: &Block| -> Option<(u8, usize)> {
        let section = b.section.as_str();
        let key = KEY_SECTIONS.contains(&section);
        let results = RESULT_SECTIONS.contains(&section);
        Some(match b.part {
            Part::Front => (0, (budget as f64 * FRONT_SHARE) as usize),
            Part::Heading => (1, usize::MAX),
            Part::Caption => (1, CAPTION_BYTES),
            Part::Body if focused.contains(&i) => (1, PARAGRAPH_BYTES),
            Part::Body if SKIPPED_SECTIONS.contains(&section) => return None,
            // Twice a paragraph: a PDF's paragraphs may run together as one block.
            Part::Body if key && b.nth < 2 => (2, 2 * PARAGRAPH_BYTES),
            Part::Table => (3, PARAGRAPH_BYTES),
            Part::Body if b.nth == 0 || (results && b.nth < 2) => (3, PARAGRAPH_BYTES),
            Part::Body if key || results => (4, usize::MAX),
            Part::Body => (5, usize::MAX),
        })
    };
    let mut order: Vec<(u8, usize, usize)> = (blocks.iter().enumerate())
        .filter_map(|(i, b)| rank(i, b).map(|(r, most)| (r, i, most)))
        .collect();
    order.sort_unstable();
    let mut front = 0;
    let mut used = 0;
    let mut kept: Vec<(usize, &str)> = Vec::new();
    for (rank, i, most) in order {
        let b = &blocks[i];
        let room = budget.saturating_sub(used + LEFT_OUT.len() + 1);
        let most = match b.part {
            Part::Front => most.saturating_sub(front),
            _ => most,
        };
        let take = b.text.len().min(most).min(room);
        let part = if take == b.text.len() {
            b.text
        } else if rank >= 4 || b.part == Part::Heading || take < MIN_PART_BYTES {
            continue;
        } else {
            let cut = b.text.floor_char_boundary(take);
            &b.text[..b.text[..cut].rfind('\n').map_or(cut, |n| n + 1)]
        };
        if part.len() < MIN_PART_BYTES.min(b.text.len()) {
            continue;
        }
        if b.part == Part::Front {
            front += part.len();
        }
        used += part.len() + LEFT_OUT.len() + 1;
        kept.push((b.start, part));
    }
    kept.sort_by_key(|&(start, _)| start);
    let mut out = String::with_capacity(budget);
    let mut at = 0;
    for (start, part) in kept {
        if !text[at..start].trim().is_empty() {
            out.push_str(LEFT_OUT);
            out.push('\n');
        } else if start > at {
            out.push('\n');
        }
        out.push_str(part);
        at = start + part.len();
    }
    if !text[at..].trim().is_empty() {
        out.push_str(LEFT_OUT);
        out.push('\n');
    }
    (out, true)
}

/// Each paper of `files`, read through the extract cache under `data_dir` and fitted to `tokens`
/// for the user's `focus`, or why it could not be read. With `pictures`, a PDF's thin pages are
/// read from their pictures too, and `pictures` hears each one (`source::text`).
pub fn read(
    data_dir: &Path,
    files: &[PathBuf],
    tokens: u32,
    focus: Option<&str>,
    stop: &AtomicBool,
    pictures: Option<&dyn Fn(&str)>,
) -> Vec<Result<Paper, String>> {
    files
        .iter()
        .map(|file| {
            if stop.load(Ordering::Relaxed) {
                return Err(larch::EngineError::Cancelled.to_string());
            }
            let text = tidy(&source::text(data_dir, file, stop, pictures)?.text);
            if text.trim().is_empty() {
                return Err(match (ocr::available(), pictures.is_some()) {
                    (true, true) => "the file has no text, and none was read from its pictures",
                    (true, false) => "the file has no text (a scanned PDF needs OCR first)",
                    (false, _) => {
                        "the file has no text (a scanned PDF needs OCR first). This machine \
                         cannot read text in pictures."
                    }
                }
                .into());
            }
            Ok(paper(file.clone(), text, tokens, focus))
        })
        .collect()
}

/// `text` as a reader gets it, fitted to `tokens` for the user's `focus`.
fn paper(file: PathBuf, text: String, tokens: u32, focus: Option<&str>) -> Paper {
    let (shown, cut) = extract(without_references(&text), tokens, focus);
    Paper {
        file,
        text,
        shown,
        cut,
    }
}

/// `text` cut at paragraph ends into parts of about `tokens`, at most `max`: past that, the parts
/// share the text out, and the last takes the rest, which its reader's extract then cuts.
pub fn parts(text: &str, tokens: u32, max: usize) -> Vec<&str> {
    let budget = (tokens as usize * TOKEN_BYTES).max(text.len().div_ceil(max.max(1)));
    let (mut out, mut start, mut end) = (Vec::new(), 0, 0);
    for p in text.split_inclusive("\n\n") {
        if end > start && end + p.len() - start > budget && out.len() + 1 < max {
            out.push(&text[start..end]);
            start = end;
        }
        end += p.len();
    }
    out.push(&text[start..]);
    out
}

/// The URL of a paper `papers` names when it is a web link, which is fetched, not a file: an
/// http or https URL, or a word that starts "www.", which a user writes for an https one. An
/// arXiv abstract page holds the abstract alone, so its paper is its PDF (a user, 5 Nov 2026: a
/// summary of /abs/ read as if it were the whole paper). A Google Drive or Dropbox share link
/// opens a viewer, so its file is its direct download (`direct`).
fn link(path: &str) -> Option<String> {
    let url = written(path)?;
    Some(match arxiv(&url) {
        Some(id) => format!("https://arxiv.org/pdf/{id}"),
        None => direct(&url).unwrap_or(url),
    })
}

/// The direct download link of a Google Drive (`drive.google.com/file/d/<id>/view`) or Dropbox
/// (`?dl=0`) share link, whose page is a viewer with no document text.
pub fn direct(url: &str) -> Option<String> {
    let mut url = url::Url::parse(url).ok()?;
    let mut path = url.path_segments()?;
    match url.host_str()? {
        "drive.google.com" => {
            let id = match (path.next(), path.next(), path.next()) {
                (Some("file"), Some("d"), Some(id)) if !id.is_empty() => id.to_owned(),
                _ => return None,
            };
            Some(format!(
                "https://drive.google.com/uc?export=download&id={id}"
            ))
        }
        "dropbox.com" | "www.dropbox.com" if matches!(path.next(), Some("s" | "scl" | "sh")) => {
            let kept: Vec<(String, String)> = (url.query_pairs())
                .filter(|(k, _)| k != "dl")
                .map(|(k, v)| (k.into_owned(), v.into_owned()))
                .collect();
            url.query_pairs_mut()
                .clear()
                .extend_pairs(kept)
                .append_pair("dl", "1");
            Some(url.into())
        }
        _ => None,
    }
}

/// Marks seen the URL `link` reads for each of `paths` that `seen` lets open as written: an
/// arXiv link the user wrote opens as its PDF, the same host and paper.
pub fn seen_as_read(paths: &[&str], seen: &mut Seen) {
    for path in paths {
        if let (Some(url), Some(read)) = (written(path), link(path))
            && seen.refuses(&url).is_none()
        {
            seen.read(&read);
        }
    }
}

/// `path` as the URL the user wrote, before `link` makes an arXiv one the paper's PDF.
fn written(path: &str) -> Option<String> {
    let lower = path.to_ascii_lowercase();
    if lower.starts_with("www.") {
        return Some(format!("https://{path}"));
    }
    (lower.starts_with("http://") || lower.starts_with("https://")).then(|| path.into())
}

/// The arXiv id in `url`, a link to its abstract, PDF or HTML page: "2610.39302v1" in
/// "https://arxiv.org/pdf/2610.39302v1.pdf", "hep-th/9901001" in an old-style one.
pub fn arxiv(url: &str) -> Option<String> {
    let lower = url.to_ascii_lowercase();
    let rest = ["https://", "http://"]
        .iter()
        .find_map(|s| lower.strip_prefix(s))?;
    let rest = rest.strip_prefix("www.").unwrap_or(rest);
    let path = rest.strip_prefix("arxiv.org/")?;
    let id = ["abs/", "pdf/", "html/"]
        .iter()
        .find_map(|k| path.strip_prefix(k))?;
    // The id as written: arXiv's ids have no capitals but an old archive name may.
    let id = url[url.len() - id.len()..].split(['?', '#']).next()?;
    let id = id.trim_end_matches('/');
    let id = match id.to_ascii_lowercase().ends_with(".pdf") {
        true => &id[..id.len() - 4],
        false => id,
    };
    (!id.is_empty()).then(|| id.to_owned())
}

/// The numbers `summary` gives, as written, that `text` does not have: each run of digits with
/// its decimal point, at least two characters, with a point, or one digit before "%" or "x" (a
/// result, where a lone digit is mostly a list's "1." or a "Table 3"); a number is found only
/// whole, give or take trailing zeros: "93.4" in "93.40" and "93.4%", "95" in "95.0", never "95"
/// in "950" or "193.4".
pub fn unchecked(summary: &str, text: &str) -> Vec<String> {
    let squash = |s: &str| s.replace([',', '\u{202f}', '\u{a0}'], "");
    let text = squash(text);
    let summary = squash(summary);
    let mut out: Vec<String> = Vec::new();
    for number in numbers(&summary) {
        let found = text.match_indices(&number).any(|(i, _)| {
            let before = text[..i].chars().next_back();
            let after: String = (text[i + number.len()..].chars())
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            // A point that ends a sentence is not part of the number.
            let after = after.strip_suffix('.').unwrap_or(&after);
            let zeros = match number.contains('.') {
                true => after.bytes().all(|b| b == b'0'),
                false => {
                    after.is_empty()
                        || after
                            .strip_prefix('.')
                            .is_some_and(|z| z.bytes().all(|b| b == b'0'))
                }
            };
            !before.is_some_and(|c| c.is_ascii_digit() || c == '.') && zeros
        });
        if !found && !out.contains(&number) {
            out.push(number);
        }
    }
    out
}

/// Whether a summary's `number` is checked: two characters or more, or a `result` ("3x", "5%"); a
/// lone digit is mostly a list's "1." or a "Table 3".
pub(crate) fn checkable(number: &str, result: bool) -> bool {
    number.len() >= 2 || result
}

fn numbers(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() || (i > 0 && (bytes[i - 1].is_ascii_alphanumeric())) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1].is_ascii_digit() {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        let number = &text[start..i];
        // "3x" and "2.5x" are results; "x" then a letter is a word.
        let times = bytes.get(i) == Some(&b'x')
            && !bytes.get(i + 1).is_some_and(u8::is_ascii_alphanumeric)
            || text[i..].starts_with('\u{d7}');
        // A digit run glued to letters ("GPT4", "3B") names a thing, not a result.
        let glued = bytes.get(i).is_some_and(u8::is_ascii_alphabetic) && !times;
        let result = times || bytes.get(i) == Some(&b'%');
        if checkable(number, result) && !glued {
            out.push(number.to_owned());
        }
    }
    out
}

/// Cuts `paper` again when its extract, `real` tokens long, runs past `tokens`: the extract
/// counts four bytes a token, and a paper's symbols and numbers take more tokens than that
/// (about a fifth more on Qwen3), which would let fewer readers fit in the KV pool at once.
pub fn refit(paper: &mut Paper, tokens: u32, real: u64, focus: Option<&str>) {
    if real <= u64::from(tokens) {
        return;
    }
    // Scaled from the extract being cut again, not from `tokens`: a second pass that scaled the
    // first budget by a smaller overshoot would cut longer than the pass before.
    let shown = (paper.shown.len() / TOKEN_BYTES) as u64;
    let scaled = shown * u64::from(tokens) / real;
    let (shown, cut) = extract(without_references(&paper.text), scaled as u32, focus);
    paper.shown = shown;
    paper.cut |= cut;
}

/// A reader's prompt: its paper under its file's name.
pub fn prompt(paper: &Paper) -> String {
    let name = paper.file.file_name().map_or_else(
        || paper.file.display().to_string(),
        |n| n.to_string_lossy().into(),
    );
    format!("File: {name}\n\n{}", paper.shown)
}

/// The readers' system prompt, with the user's `focus` when there is one; with `followups`
/// (`doc_followups`), `KINDS` too.
pub fn system(focus: Option<&str>, followups: bool) -> String {
    let system = match followups {
        true => READER_SYSTEM.replacen("Use only", &format!("{KINDS}Use only"), 1),
        false => READER_SYSTEM.into(),
    };
    match focus.map(str::trim).filter(|f| !f.is_empty()) {
        Some(focus) => format!("{system}\n\nThe user wants to know: {focus}"),
        None => system,
    }
}

/// A reader's parameters: a worker step's (no thinking, greedy), seeded, `SUMMARY_TOKENS` long.
pub(crate) fn params(family: Option<Family>, namespace: u64) -> GenParams {
    let worker = run::params(
        &Registry::builtin(),
        &[],
        family,
        Kind::Worker,
        Default::default(),
        namespace,
    );
    GenParams {
        max_tokens: SUMMARY_TOKENS,
        seed: Some(0),
        ..worker
    }
}

/// Tokens the chat template adds around a reader's system prompt and prompt, with room to spare:
/// Llama 3.1's adds about 30 (its date lines), Qwen3's about 15.
// ponytail: an estimate, as the facade renders no template for the agent; count the rendered
// request when larch exposes its token count
pub(crate) const TEMPLATE_TOKENS: u64 = 96;

/// Tokens of one KV block: the engine gives each sequence whole blocks.
pub const KV_BLOCK_TOKENS: u64 = 16;

/// Readers that fit at once: one per paper, at most `workers.max_workers` (the chat waits on the
/// call, so its slot is free), and no more than a pool of `pool_blocks` KV blocks holds at `each`
/// tokens a reader (its longest prompt and its summary), each rounded up to whole blocks, so the
/// engine does not preempt one reader for another. A pool of 0 is not known, and caps nothing.
pub fn at_once(papers: usize, workers: Workers, pool_blocks: u64, each: u64) -> usize {
    let each = each.max(1).div_ceil(KV_BLOCK_TOKENS);
    let fits = match pool_blocks {
        0 => usize::MAX,
        pool => usize::try_from(pool / each).unwrap_or(usize::MAX),
    };
    papers.min(workers.max_workers).min(fits).max(1)
}

/// A document a call read, as the chat's source record adds it (`doc::source`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Added {
    /// A local file, by its real path.
    File(PathBuf),
    /// A link, by its entry and its page's text.
    Link(source::Entry, String),
}

/// What `Papers::texts` read.
struct Texts {
    /// Each file and link, files first.
    files: Vec<PathBuf>,
    /// Each one's text, or why it could not be read.
    papers: Vec<Result<Paper, String>>,
    /// What the chat's source record adds for each one read.
    read: Vec<Option<Added>>,
    /// What a link's page is when it may not be the document (`fetch::note`).
    notes: Vec<String>,
}

impl Papers {
    /// The files and links `paths` names, files first, each read whole and fitted to the reader's
    /// budget for `focus`, or why it could not be read; links only through `web`'s provenance
    /// check. `pictures` comes only with follow-ups on: then thin PDF pages are read from their
    /// pictures (`read`, `source::link_pictures`), and a linked PDF's bytes are kept.
    fn texts(
        &self,
        ctx: &ToolCtx,
        paths: &[&str],
        focus: Option<&str>,
        web: Option<&mut Seen>,
        stop: &AtomicBool,
        pictures: Option<&dyn Fn(&str)>,
    ) -> Result<Texts, ToolError> {
        let links: Vec<String> = paths.iter().filter_map(|p| link(p)).collect();
        let local: Vec<&str> = paths
            .iter()
            .copied()
            .filter(|p| link(p).is_none())
            .collect();
        let mut files = match local.is_empty() {
            true => Vec::new(),
            false => gather(ctx, &local)?,
        };
        if files.len() + links.len() > MAX_PAPERS {
            return Err(ToolError::Invalid(format!(
                "{} papers; one call reads at most {MAX_PAPERS}: give the ones to read",
                files.len() + links.len()
            )));
        }
        source::prune(&ctx.data_dir);
        let tokens = self.paper_tokens;
        let mut papers = read(&ctx.data_dir, &files, tokens, focus, stop, pictures);
        let mut added: Vec<Option<Added>> = (files.iter().zip(&papers))
            .map(|(f, p)| p.is_ok().then(|| Added::File(f.clone())))
            .collect();
        // What the links' pages are when they may not be the document, after every summary, so
        // the summaries shown while they were written stay the answer's start.
        let mut notes = Vec::new();
        if !links.is_empty() {
            let Some(seen) = web else {
                return Err(ToolError::Invalid(format!(
                    "{TOOL} reads web links only where the web is on"
                )));
            };
            seen_as_read(paths, seen);
            let urls: Vec<&str> = links.iter().map(String::as_str).collect();
            let pages = fetch::fetch_documents(&urls, &ctx.data_dir, seen, pictures.is_some());
            for (link, page) in links.iter().zip(pages) {
                files.push(PathBuf::from(link));
                let page = page.map_err(|e| e.to_string()).and_then(|mut p| {
                    let ocr = match pictures {
                        Some(progress) if p.pdf && ocr::available() => {
                            let (text, ocr) =
                                source::link_pictures(&ctx.data_dir, &p, stop, progress)?;
                            p.text = text;
                            ocr
                        }
                        _ => false,
                    };
                    match p.text.trim().is_empty() {
                        true => Err("the page has no text".to_owned()),
                        false => Ok((p, ocr)),
                    }
                });
                added.push(page.as_ref().ok().map(|(p, ocr)| {
                    Added::Link(source::Entry::link(link, p, *ocr), p.text.clone())
                }));
                papers.push(match page {
                    Err(e) => Err(e),
                    Ok((p, _)) => {
                        if let Some(note) = fetch::note(&p) {
                            notes.push(format!("[{}] {note}", files.len()));
                        }
                        // An essay's title can be in <title> alone (paulgraham.com's are).
                        let titled = p.text.starts_with("# ") || p.title.trim().is_empty();
                        let text = match titled {
                            true => p.text,
                            false => format!("# {}\n\n{}", p.title.trim(), p.text),
                        };
                        Ok(paper(p.final_url.into(), tidy(&text), tokens, focus))
                    }
                });
            }
        }
        Ok(Texts {
            files,
            papers,
            read: added,
            notes,
        })
    }

    /// Reads the papers `paths` names on `larch`'s model, inside `ctx`'s folders, and the web
    /// links through `web`'s provenance check, files first; a caller without the web passes
    /// `None` and gets no link read. A lone document longer than a reader's budget is read in
    /// parts at once, and the answer starts with `PARTS`. `stop` ends the readers. `show` gets
    /// the answer's markdown while the readers write it, each summary as it comes; a paper that
    /// cannot be read, a reader that fails, or parts, leave the rest unshown.
    #[allow(clippy::too_many_arguments)]
    pub fn run_until(
        &self,
        larch: &Larch,
        ctx: &ToolCtx,
        workers: Workers,
        paths: &[&str],
        focus: Option<&str>,
        show: &mut dyn FnMut(&str),
        web: Option<&mut Seen>,
        stop: &AtomicBool,
    ) -> Result<PapersRun, ToolError> {
        self.run_with(larch, ctx, workers, paths, focus, show, web, stop, None)
    }

    /// `run_until`, with `pictures` as `texts` takes it, given only with follow-ups on (research
    /// v2): they read a PDF's thin pages from their pictures and say so on the chat's progress
    /// line, and the summaries' prompt gets `KINDS`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_with(
        &self,
        larch: &Larch,
        ctx: &ToolCtx,
        workers: Workers,
        paths: &[&str],
        focus: Option<&str>,
        show: &mut dyn FnMut(&str),
        web: Option<&mut Seen>,
        stop: &AtomicBool,
        pictures: Option<&dyn Fn(&str)>,
    ) -> Result<PapersRun, ToolError> {
        let start = Instant::now();
        let Texts {
            mut files,
            mut papers,
            read,
            notes,
        } = self.texts(ctx, paths, focus, web, stop, pictures)?;
        let tokens = self.paper_tokens;
        // One document past the budget is read whole, a reader a part, rather than cut.
        // ponytail: only a lone document, so a call over several papers keeps one reader each
        // (the demo's time); share the parts out when a summary of several misses a long one's
        // middle
        let parted = match &papers[..] {
            [Ok(p)] => Some(parts(without_references(&p.text), tokens, MAX_PAPERS))
                .filter(|parts| parts.len() > 1)
                .map(|parts| {
                    let n = parts.len();
                    // Every part's reader sees the document's title first, as the first part's does.
                    let title = p.text.lines().next().filter(|l| l.starts_with("# "));
                    (1..)
                        .zip(parts)
                        .map(|(i, text)| {
                            let named = format!("{} (part {i} of {n})", name(&p.file));
                            let text = match title {
                                Some(title) if i > 1 => format!("{title}\n\n{text}"),
                                _ => text.into(),
                            };
                            paper(p.file.with_file_name(named), text, tokens, focus)
                        })
                        .collect::<Vec<_>>()
                }),
            _ => None,
        };
        if let Some(parted) = &parted {
            files = parted.iter().map(|p| p.file.clone()).collect();
            papers = parted.iter().cloned().map(Ok).collect();
        }
        // Counted, not estimated from bytes: a paper's text runs past four bytes a token.
        let count = |text: &str| larch.token_ids(text).len() as u64;
        // The scaled cut can still run a little over (a paper's parts tokenize unevenly), and a
        // reader one KV block over its share keeps another reader waiting.
        for paper in papers.iter_mut().flatten() {
            for _ in 0..3 {
                let real = count(&paper.shown);
                if real <= u64::from(self.paper_tokens) {
                    break;
                }
                refit(paper, self.paper_tokens, real, focus);
            }
        }
        let read_ms = ms(start);
        if stop.load(Ordering::Relaxed) {
            return Err(ToolError::Skill(larch::EngineError::Cancelled.to_string()));
        }
        let ok: Vec<&Paper> = papers.iter().filter_map(|p| p.as_ref().ok()).collect();
        let prompts: Vec<String> = ok.iter().map(|p| prompt(p)).collect();
        let system = system(focus, pictures.is_some());
        let reader_tokens = count(&system)
            + prompts.iter().map(|p| count(p)).max().unwrap_or(0)
            + u64::from(SUMMARY_TOKENS)
            + TEMPLATE_TOKENS;
        let pool_blocks = u64::from(larch.deployment().kv_pool_blocks);
        let pool_tokens = pool_blocks * KV_BLOCK_TOKENS;
        let n = at_once(ok.len(), workers, pool_blocks, reader_tokens);
        // Workers keeps one slot for the chat; this call's chat waits on it, so the readers get it.
        let pool = Workers {
            max_workers: n + 1,
            ..workers
        };
        let p = params(self.family, crate::namespace(&ctx.data_dir));
        let start = Instant::now();
        let mut live = Live {
            show,
            on: papers.iter().all(Result::is_ok) && parted.is_none(),
            at: None,
            text: String::new(),
            shown: 0,
        };
        // Each reader is a helper the chat shows, by its file's or page's name.
        let tasks: Vec<String> = (ok.iter())
            .map(|p| {
                let name = p.file.file_name().unwrap_or(p.file.as_os_str());
                format!("Reading {}", name.to_string_lossy())
            })
            .collect();
        let replies = pool.run_helpers(
            larch,
            &system,
            &prompts,
            &tasks,
            &p,
            |_| None,
            |j, e| live.event(ok[j], j, e),
            stop,
        );
        let readers_ms = ms(start);
        let mut replies = replies.into_iter();
        let readers: Vec<Reader> = (files.iter().zip(&papers))
            .map(|(file, paper)| {
                let paper = match paper {
                    Ok(paper) => paper,
                    Err(e) => return Reader::failed(file, e.clone(), false),
                };
                let (answers, error) = replies.next().unwrap_or_default();
                let Some(a) = answers.into_iter().next() else {
                    let e = error.map_or_else(|| "no reply".into(), |e| e.to_string());
                    return Reader::failed(file, e, paper.cut);
                };
                Reader {
                    prompt_tokens: a.usage.prompt_tokens,
                    completion_tokens: a.usage.completion_tokens,
                    ttft_ms: a.timings.ttft_ms,
                    ..Reader::of(paper, &a.text, a.finish)
                }
            })
            .collect();
        if stop.load(Ordering::Relaxed) {
            return Err(ToolError::Skill(larch::EngineError::Cancelled.to_string()));
        }
        if readers.iter().all(|r| r.summary.is_err()) {
            let why: Vec<(String, &str)> = (readers.iter())
                .map(|r| (name(&r.file), r.summary.as_ref().unwrap_err().as_str()))
                .collect();
            return Err(ToolError::Skill(unread(&why)));
        }
        let mut markdown = match parted {
            Some(_) => format!("{PARTS}\n\n{}", markdown(&readers)),
            None => markdown(&readers),
        };
        if !notes.is_empty() {
            markdown = format!("{markdown}\n\n{}", notes.join("\n"));
        }
        Ok(PapersRun {
            markdown,
            readers,
            at_once: n,
            pool_tokens,
            reader_tokens,
            read,
            read_ms,
            readers_ms,
        })
    }
}

/// The answer's markdown as the readers write it: the readers' events come in paper order, the
/// first reader's while it writes, so each summary shows as it comes.
struct Live<'a> {
    show: &'a mut dyn FnMut(&str),
    /// No paper failed to read, so the markdown starts with paper 1's section.
    on: bool,
    /// The reader writing, by its prompt's index.
    at: Option<usize>,
    /// Its text so far.
    text: String,
    /// The bytes of its section shown.
    shown: usize,
}

impl Live<'_> {
    fn event(&mut self, paper: &Paper, j: usize, e: &TextEvent) {
        if !self.on {
            return;
        }
        if self.at != Some(j) {
            if self.at.is_some_and(|at| at + 1 != j) || (self.at.is_none() && j != 0) {
                // A reader failed between: its section is not one this shows.
                self.on = false;
                return;
            }
            if self.at.is_some() {
                (self.show)("\n\n");
            }
            self.at = Some(j);
            self.text.clear();
            self.shown = 0;
        }
        let (text, finish) = match e {
            TextEvent::Prefilled { .. } | TextEvent::Thinking { .. } => return,
            TextEvent::Text { delta, .. } => {
                self.text.push_str(delta);
                (self.text.trim(), None)
            }
            TextEvent::Done { finish, .. } => (self.text.as_str(), Some(*finish)),
            TextEvent::Error(_) => {
                self.on = false;
                return;
            }
        };
        let section = match finish {
            // Trailing white space waits: a summary's ends are trimmed.
            None => section(j + 1, &Reader::of(paper, text, Finish::Eos), false),
            Some(finish) => section(j + 1, &Reader::of(paper, text, finish), true),
        };
        let section = section.trim_end_matches('\n');
        if let Some(more) = section.get(self.shown..) {
            (self.show)(more);
            self.shown = section.len();
        }
    }
}

impl Reader {
    /// A reader's summary of `paper`, from its reply `text`.
    fn of(paper: &Paper, text: &str, finish: Finish) -> Reader {
        let mut summary = text.trim().to_owned();
        if finish == Finish::Length {
            summary.push_str(" [cut at the length limit]");
        }
        Reader {
            file: paper.file.clone(),
            unchecked: unchecked(&summary, &paper.text),
            summary: Ok(summary),
            prompt_tokens: 0,
            completion_tokens: 0,
            ttft_ms: 0.0,
            cut: paper.cut,
        }
    }

    fn failed(file: &Path, why: String, cut: bool) -> Reader {
        Reader {
            file: file.into(),
            summary: Err(why),
            unchecked: Vec::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            ttft_ms: 0.0,
            cut,
        }
    }
}

/// What the model hears when no paper could be read, before the paper-by-paper reasons.
pub const UNREAD: &str = "Tell the user you could not read these papers and why. Do not describe \
them from memory: an id or a title is not what a paper says.";

/// The longest first line `unread` gives: the chat's step line shows it after "failed: the
/// skill failed: ", on one line.
const UNREAD_LINE: usize = 130;

/// The error of a call that read none of the papers `why` names: a first line the chat shows as
/// the step's, each reason once without the URL each paper repeats, then each paper's whole
/// reason, for the model and the audit log, and `UNREAD`.
fn unread(why: &[(String, &str)]) -> String {
    // "HTTP 404" of "https://a.org/x: HTTP 404", the detail after a reason's first ": " cut.
    let short = |why: &str| {
        let why = match why.starts_with("http://") || why.starts_with("https://") {
            true => why.split_once(": ").map_or(why, |(_, rest)| rest),
            false => why,
        };
        why.split(": ").next().unwrap_or(why).to_owned()
    };
    let mut reasons: Vec<String> = Vec::new();
    for (_, w) in why {
        let w = short(w);
        if !reasons.contains(&w) {
            reasons.push(w);
        }
    }
    let reasons = reasons.join("; ");
    let mut line = match why.len() {
        1 => format!("could not read the paper: {reasons}"),
        n => format!("could not read any of the {n} papers: {reasons}"),
    };
    if line.len() > UNREAD_LINE {
        line.truncate(line.floor_char_boundary(UNREAD_LINE - 3));
        line.push('…');
    }
    let each: Vec<String> = why.iter().map(|(n, w)| format!("{n}: {w}")).collect();
    format!("{line}\n{}\n{UNREAD}", each.join("\n"))
}

/// `file`'s name, or the whole of it when it has none (a link's).
pub(crate) fn name(file: &Path) -> String {
    file.file_name().map_or_else(
        || file.display().to_string(),
        |n| n.to_string_lossy().into(),
    )
}

/// The call's answer: each paper's summary under its number and name, and the numbers not found
/// in its text.
pub fn markdown(readers: &[Reader]) -> String {
    let sections: Vec<String> = (1..)
        .zip(readers)
        .map(|(n, r)| section(n, r, true))
        .collect();
    sections.join("\n").trim_end().into()
}

/// Paper `n`'s part of the markdown: its heading, its summary and, when `done`, the numbers to
/// check, which a summary still being written does not have yet.
fn section(n: usize, r: &Reader, done: bool) -> String {
    let cut = if r.cut { " (key parts read)" } else { "" };
    let mut out = format!("## [{n}] {}{cut}\n", name(&r.file));
    match &r.summary {
        Ok(summary) => out.push_str(summary),
        Err(e) => out.push_str(&format!("Not read: {e}")),
    }
    out.push('\n');
    if done && !r.unchecked.is_empty() {
        out.push_str(&format!(
            "Check these numbers, not found in the paper's text: {}\n",
            r.unchecked.join(", ")
        ));
    }
    out
}

/// The skill as a chat tool, run on the chat's engine.
pub(crate) struct PapersTool {
    /// The engine and the settings; `None` in `Registry::builtin`, which runs no model.
    pub(crate) engine: Option<(Larch, Papers, Workers)>,
    /// The chat's seen URLs when the call was made, which its links are checked against; `None`
    /// when the turn has no web, which reads no link.
    pub(crate) seen: Option<Seen>,
    /// Ends the run when set: a chat sets it when its turn is dropped.
    pub(crate) stop: Arc<AtomicBool>,
    /// Where the answer goes while the readers write it, for the chat to show.
    pub(crate) show: Option<std::sync::mpsc::Sender<String>>,
    /// With follow-ups on (`doc_followups`): a call with a focus is the follow-up reader's, and
    /// each document a summary reads goes in the chat's source record.
    pub(crate) followup: Option<Followup>,
}

impl crate::tool::Tool for PapersTool {
    fn spec(&self) -> crate::tool::ToolSpec {
        crate::tool::ToolSpec {
            name: TOOL.into(),
            description: match self.followup {
                Some(_) => FOLLOWUP_DESCRIPTION.into(),
                None => TOOL_DESCRIPTION.into(),
            },
            schema: json!({
                "type": "object",
                "properties": {
                    "papers": {"type": "array", "items": {"type": "string"},
                               "minItems": 1, "maxItems": MAX_PAPERS,
                               "description": "Paper files, one folder of papers, or web links"},
                    "focus": {"type": "string",
                              "description": match self.followup {
                                  Some(_) => FOCUS_READ,
                                  None => FOCUS,
                              }}
                },
                "required": ["papers"],
                "additionalProperties": false
            }),
            permission: crate::tool::Permission::Read,
            // A link's page is the web's words.
            untrusted_results: true,
            // Never cut: MAX_PAPERS summaries of SUMMARY_TOKENS bound it.
            max_result_tokens: u32::MAX,
            max_args_tokens: crate::tool::ARGS_TOKENS,
        }
    }

    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        let Some((larch, papers, workers)) = &self.engine else {
            return Err(ToolError::Invalid(format!(
                "{TOOL} runs only in a chat, which has the model's engine"
            )));
        };
        let paths: Vec<&str> = (args["papers"].as_array().into_iter().flatten())
            .filter_map(Value::as_str)
            .collect();
        let focus = args["focus"].as_str();
        // A chat that stopped reading has dropped its end: the answer still comes back.
        let mut show = |text: &str| {
            let _ = self.show.as_ref().map(|s| s.send(text.into()));
        };
        let mut seen = self.seen.clone();
        let question = focus.filter(|f| !f.trim().is_empty());
        let progress = |text: &str| {
            let tell = self.followup.as_ref().and_then(|f| f.progress.as_ref());
            let _ = tell.map(|p| p.send(text.into()));
        };
        if let (Some(f), Some(question)) = (&self.followup, question) {
            let (docs, notes) = self.docs(f, papers, ctx, &paths, seen.as_mut(), &progress)?;
            let ask = Ask {
                question,
                last: f.last.as_ref().map(|(q, a)| (q.as_str(), a.as_str())),
                router: f.router,
                chat: &f.chat,
                ..Ask::default()
            };
            let mut readers = (f.readers.lock()).unwrap_or_else(std::sync::PoisonError::into_inner);
            return match readers.answer(larch, ctx, &docs, ask, &self.stop, &mut show, &progress) {
                Ok(markdown) => Ok([markdown]
                    .into_iter()
                    .chain(notes)
                    .collect::<Vec<_>>()
                    .join("\n\n")),
                Err(SkillError::Tool(e)) => Err(e),
                Err(e) => Err(ToolError::Skill(e.to_string())),
            };
        }
        // Follow-ups read a scan's pictures: their answers quote them.
        let pictures = self
            .followup
            .is_some()
            .then_some(&progress as &dyn Fn(&str));
        let run = papers.run_with(
            larch,
            ctx,
            *workers,
            &paths,
            focus,
            &mut show,
            seen.as_mut(),
            &self.stop,
            pictures,
        )?;
        if let Some(f) = &self.followup {
            registered(f, ctx, &run, &self.stop)?;
        }
        Ok(run.markdown)
    }
}

impl PapersTool {
    /// The documents `paths` names, from chat `f`'s source record, or read now and added to it
    /// when it has none of them, a scan's pages from their pictures as `progress` hears, in the
    /// chat's letter order; with what their readers should hear of a moved or changed original.
    // ponytail: a moved file fails `gather` before its entry is tried; match entries by name
    // when the trace shows such a follow-up
    fn docs(
        &self,
        f: &Followup,
        papers: &Papers,
        ctx: &ToolCtx,
        paths: &[&str],
        web: Option<&mut Seen>,
        progress: &dyn Fn(&str),
    ) -> Result<(Vec<Doc>, Vec<String>), ToolError> {
        let links: Vec<String> = paths.iter().filter_map(|p| link(p)).collect();
        let local: Vec<&str> = (paths.iter().copied())
            .filter(|p| link(p).is_none())
            .collect();
        let mut sources: Vec<String> = match local.is_empty() {
            true => Vec::new(),
            false => (gather(ctx, &local)?.iter())
                .map(|f| f.display().to_string())
                .collect(),
        };
        sources.extend(links);
        let entries = source::load(&ctx.data_dir, &f.chat)
            .map_err(|e| ToolError::Io("the chat's source record".into(), e))?;
        let (mut docs, mut notes, mut unread) = (Vec::new(), Vec::new(), Vec::new());
        for s in &sources {
            let entry = (entries.iter().enumerate()).find(|(_, e)| e.source == *s);
            match entry.map(|(i, e)| source::open(&ctx.data_dir, i, e, &self.stop)) {
                Some(Ok(opened)) => {
                    docs.push(opened.doc);
                    notes.extend(opened.note);
                }
                // Not read in this chat, or its text is no longer kept: read now, which keeps
                // its letter.
                _ => unread.push(s.as_str()),
            }
        }
        if !unread.is_empty() {
            // ponytail: a link's viewer note (`fetch::note`) is the summary's; show it here too
            // when a first read of a viewer's link comes as a follow-up
            let read = papers.texts(ctx, &unread, None, web, &self.stop, Some(progress))?;
            for ((file, paper), added) in read.files.iter().zip(read.papers).zip(read.read) {
                paper.map_err(|e| ToolError::Skill(format!("{}: {e}", name(file))))?;
                let added = added.expect("a document read has what its record adds");
                docs.push(
                    add(&ctx.data_dir, &f.chat, &added, &self.stop).map_err(ToolError::Skill)?,
                );
            }
        }
        docs.sort_by_key(|d| d.index);
        Ok((docs, notes))
    }
}

/// Puts each document `run` read in chat `f`'s source record, with its summary for the follow-up
/// reader; one that does not go in is a trace line, and the summary stands.
fn registered(
    f: &Followup,
    ctx: &ToolCtx,
    run: &PapersRun,
    stop: &AtomicBool,
) -> Result<(), ToolError> {
    let parted = run.markdown.starts_with(PARTS);
    let summaries = |i: usize| -> Option<String> {
        let ok = |r: &Reader| r.summary.as_ref().ok().cloned();
        match parted {
            true => Some(
                run.readers
                    .iter()
                    .filter_map(ok)
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            ),
            false => run.readers.get(i).and_then(ok),
        }
    };
    let mut readers = (f.readers.lock()).unwrap_or_else(std::sync::PoisonError::into_inner);
    for (i, read) in (run.read.iter().enumerate()).filter_map(|(i, r)| Some((i, r.as_ref()?))) {
        match add(&ctx.data_dir, &f.chat, read, stop) {
            Ok(doc) => {
                if let Some(summary) = summaries(i) {
                    readers.summarized(&doc.sha256, summary);
                }
            }
            Err(e) => {
                let event = AuditEvent {
                    skill: Some(crate::skills::doc::send::SKILL.into()),
                    error: Some(e),
                    ..ask_docs::step("register", 0.0, None)
                };
                (Audit::new(&ctx.data_dir).append(event))
                    .map_err(|e| ToolError::Io("audit log".into(), e))?;
            }
        }
    }
    Ok(())
}

/// Adds `read` to chat `chat`'s source record under `data`: the document as its reader gets it,
/// at its letter, the text as `source::open` gives it again, a scan's pages from their pictures
/// (read just before, so from the OCR cache).
fn add(data: &Path, chat: &str, read: &Added, stop: &AtomicBool) -> Result<Doc, String> {
    let (entry, text) = match read {
        Added::File(path) => {
            let x = source::text(data, path, stop, Some(&|_| {}))
                .map_err(|e| format!("{}: {e}", name(path)))?;
            (source::Entry::file(&name(path), path, &x), x.text)
        }
        Added::Link(entry, text) => (entry.clone(), text.clone()),
    };
    let index = (source::add(data, chat, entry.clone()))
        .map_err(|e| format!("{}: the chat's source record: {e}", entry.name))?;
    let rep = entry.reps.first().map_or(Rep::Text, |r| r.0);
    Ok(source::doc(index, &entry.name, &entry.sha256, rep, text))
}

#[cfg(test)]
mod tests {

    /// An AES-256 (R6) file with an empty user password, as manuals ship: the load decrypts it,
    /// and its text reads once, not decrypted twice into an error.
    #[test]
    fn an_aes_256_pdf_with_no_password_reads() {
        use pdf_extract::encryption::crypt_filters::{Aes256CryptFilter, CryptFilter};
        use pdf_extract::{Document, EncryptionState, EncryptionVersion, Permissions};
        use std::sync::Arc;
        let stream = "BT /F1 12 Tf 72 720 Td (USB offline update) Tj ET";
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
            "<< /Type /Pages /Kids [4 0 R] /Count 1 >>".to_owned(),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R \
             /Resources << /Font << /F1 3 0 R >> >> >>"
                .to_owned(),
            format!(
                "<< /Length {} >>\nstream\n{stream}\nendstream",
                stream.len()
            ),
        ];
        let mut plain = b"%PDF-1.7\n".to_vec();
        let mut offsets = Vec::new();
        for (i, o) in objects.iter().enumerate() {
            offsets.push(plain.len());
            plain.extend(format!("{} 0 obj\n{o}\nendobj\n", i + 1).bytes());
        }
        let xref = plain.len();
        plain.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
        for o in offsets {
            plain.extend(format!("{o:010} 00000 n \n").bytes());
        }
        plain.extend(
            format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").bytes(),
        );
        let mut doc = Document::load_mem(&plain).unwrap();
        let filter: Arc<dyn CryptFilter> = Arc::new(Aes256CryptFilter);
        let state = EncryptionState::try_from(EncryptionVersion::V5 {
            encrypt_metadata: true,
            crypt_filters: [(b"StdCF".to_vec(), filter)].into(),
            file_encryption_key: &[7; 32],
            stream_filter: b"StdCF".to_vec(),
            string_filter: b"StdCF".to_vec(),
            owner_password: "owner",
            user_password: "",
            permissions: Permissions::all(),
        })
        .unwrap();
        doc.encrypt(&state).unwrap();
        let mut encrypted = Vec::new();
        doc.save_to(&mut encrypted).unwrap();
        assert!(
            !encrypted.windows(7).any(|w| w == b"offline"),
            "the text is encrypted"
        );
        let text = pdf_text(&encrypted).unwrap();
        assert!(text.contains("USB offline update"), "{text:?}");
    }
    use super::*;

    /// `first_text` over fake readers that give `reads`, with `stop` set or not, and how many
    /// were tried.
    fn first(reads: &[Result<&str, &str>], stop: bool) -> (Result<String, String>, usize) {
        let tried = std::cell::Cell::new(0);
        let tried = &tried;
        let mut fakes: Vec<_> = (reads.iter())
            .map(|r| {
                move || {
                    tried.set(tried.get() + 1);
                    r.map(str::to_owned).map_err(str::to_owned)
                }
            })
            .collect();
        let mut reads: Vec<&mut dyn FnMut() -> Result<String, String>> =
            fakes.iter_mut().map(|f| f as _).collect();
        let got = first_text(&mut reads, &AtomicBool::new(stop));
        (got, tried.get())
    }

    #[test]
    fn the_summary_prompt_names_decks_and_reports_only_with_followups_on() {
        assert_eq!(system(None, false), READER_SYSTEM);
        assert!(!READER_SYSTEM.contains("slide deck"));
        let on = system(Some("the rent"), true);
        let kinds = "not a paper's parts. For a slide deck, give its pitch and its key numbers; \
                     for a report or a datasheet, give its subject and its key figures. Use only";
        assert!(on.contains(kinds), "{on}");
        assert!(on.ends_with("\n\nThe user wants to know: the rent"), "{on}");
    }

    #[test]
    fn a_reader_that_gives_no_text_leads_to_the_next() {
        // pdf-extract's empty text of a 10-K that PDFKit reads: PDFKit's text.
        let got = first(
            &[Err("not installed"), Ok(" \n"), Ok("Revenue"), Ok("x")],
            false,
        );
        assert_eq!(got, (Ok("Revenue".into()), 3));
        // Fewer letters than pages is no text either.
        let pages = "Page one\n\u{c}two\n\u{c}";
        assert_eq!(
            first(&[Ok("x\u{c}\u{c}\u{c}"), Ok(pages)], false).0,
            Ok(pages.into())
        );
        // A scan: no reader has text, and the first one's pages stay.
        let scan = first(&[Ok("\u{c}\u{c}"), Ok(""), Err("")], false);
        assert_eq!(scan, (Ok("\u{c}\u{c}".into()), 3));
        // A broken file: the last reason given, the built-in reader's, not PDFKit's none.
        let broken = first(
            &[Err("exit 1"), Err("the PDF could not be read"), Err("")],
            false,
        );
        assert_eq!(broken.0, Err("the PDF could not be read".into()));
        // A stop ends the tries.
        assert_eq!(
            first(&[Err("it was stopped"), Ok("text")], true),
            (Err("it was stopped".into()), 1)
        );
    }
}
