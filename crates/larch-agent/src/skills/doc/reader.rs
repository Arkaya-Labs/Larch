//! The follow-up reader: one question about a chat's documents, answered from their text.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Instant,
};

use larch::{EngineError, Finish, GenParams, Larch, Overflow, Prompt, Session, SessionOptions};
use regex::Regex;
use serde_json::{Value, json};

use super::{
    all, compare,
    record::{Progress, Searches},
    send::{self, SKILL, To},
};
use crate::{
    audit::{Audit, AuditEvent},
    config::Thinking,
    folio::{Doc, Page, Range, Rep, cite, map, ocr, source, table, tools, visible::Visible},
    parse::{self, Family, ToolCall, ValidateError},
    router::Intent,
    run,
    skill::SkillError,
    skills::{
        ask_docs::{self, Chunk, ms},
        read_papers::{self, KV_BLOCK_TOKENS, LEFT_OUT, TEMPLATE_TOKENS, TOKEN_BYTES},
    },
    tool::{ARGS_TOKENS, Permission, ToolCtx, ToolError, ToolSpec},
    workers::Workers,
};

/// Tokens of a question's documents a reader gets whole (config.toml's `followup_tokens`); past
/// them it gets a first read and its tools.
// ponytail: the Mac read-size sweep (13 Nov) found the 35B's first word at 6.9 s at 6k and 15.8 s
// at 12k and the 8B's at 9.5 s at 6k, every read right with thinking off, so 6,000 stays; raise it
// when a harder question set separates the sizes on accuracy
pub const FOLLOWUP_TOKENS: u32 = 6000;

/// Every reader's system prompt. Each sentence is A/B tested on the Mac, so it stays short.
pub const SYSTEM: &str = "You answer a question about the documents below from their text \
alone. Quote the exact sentence, formula or table row that answers it, in quote marks, each \
quote followed by its place as the text marks it, such as [p. 12] or [B, p. 12]. Name the \
section it is in. If the text does not answer the question, start with \"Not found in the \
text.\" and say what it does cover. You cannot see pictures: if the answer needs one, say what \
text was read from it, if any, and that you cannot see it. Answer briefly.";

/// The line of a reader's prompt that has tools, after `SYSTEM`: what this machine can read of a
/// PDF's pictures and tables, so a question about a chart or a figure finds `read`'s options
/// (plan 4).
fn can() -> String {
    let pictures = match ocr::available() {
        true => {
            "For a question about a chart, figure or scan, read its page with pictures=true to \
             add the text inside its pictures; you cannot see what a picture shows."
        }
        false => "This machine cannot read text inside pictures.",
    };
    match table::available() {
        true => format!(
            "{pictures} For a table, read its page with tables=true to add its rows and columns."
        ),
        false => pictures.into(),
    }
}

/// The system prompt of a reader that has `tools` or has none.
fn system(tools: bool) -> String {
    match tools {
        true => format!("{SYSTEM}\n{}", can()),
        false => SYSTEM.into(),
    }
}

/// How an answer the text does not give starts (`SYSTEM`).
pub const NOT_FOUND: &str = "Not found in the text.";

/// The one retry after an empty answer, sent without thinking.
const RETRY: &str = "[agent] Your answer was empty. Answer the question from the text above.";

/// Every call's result once not even a small one fits the context: the next step has no tools.
const NO_ROOM: &str = "[agent] No room is left in this context for more text: answer from what \
you have read, and say what you could not read.";

/// Pages one `read` reads the pictures and tables of: each OCR takes about a second.
// ponytail: 4 by hand; set it from the Mac's Vision time per page when a question about a
// deck's charts asks for more in one call
const PICTURE_PAGES: usize = 4;

/// What ends the pictures' and tables' part of a result that the room cuts.
const CUT: &str = "[agent] No more of what was read from the pictures and tables fits the room \
left: read fewer pages.\n";

/// Tokens of the smallest tool result worth sending: a few lines `find` matched.
const SMALL_RESULT: u64 = 256;

/// Tokens of the shortest answer a full context still gets: a few quoted sentences.
const MIN_ANSWER: u64 = 256;

/// Tokens a step without thinking may render past the one it replaces (Qwen3's empty think
/// block).
const MARGIN: u64 = 16;

/// Bytes of the chat's last answer a reader gets: its start names what "that" means.
const LAST_BYTES: usize = 400;

/// Bytes of a first read's piece of a page with no breaks (a web page, a text file): about a
/// PDF page's text.
const PAGE_BYTES: usize = 3000;

/// Bytes of each document `bytes_per_token` counts.
const SAMPLE: usize = 16 * 1024;

/// Characters of a text the trace keeps.
const PREVIEW: usize = 200;

/// The search tool's name.
const FIND: &str = "find";

/// The reading tool's name.
const READ: &str = "read";

/// A page the question names: "page 12", "p. 12", "pp. 3-5", "pages 12 to 14".
static NAMED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:pages?|pp?)\.?\s*\d+(?:\s*(?:-|\u{2013}|\u{2014}|to)\s*\d+)?")
        .expect("a valid pattern")
});

/// How readers run: the chat's model family, its thinking, and the whole-document limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    /// The chat's tool-call family.
    pub family: Option<Family>,
    /// The chat's thinking budget and answer cap.
    pub thinking: Thinking,
    /// config.toml's `followup_tokens`.
    pub followup_tokens: u32,
    /// Part readers that run at once: config.toml's `max_workers`.
    pub max_workers: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            family: None,
            thinking: Thinking::default(),
            followup_tokens: FOLLOWUP_TOKENS,
            max_workers: Workers::default().max_workers,
        }
    }
}

/// One question as a reader gets it, with what the chat said before it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ask<'a> {
    /// The user's question, as the chat's model wrote it in `focus`.
    pub question: &'a str,
    /// The documents' summaries from the call that read them; empty when there are none.
    pub summary: &'a str,
    /// The chat's last question and its answer, so "that" has a meaning.
    pub last: Option<(&'a str, &'a str)>,
    /// The router's label of the turn, for the trace.
    pub router: Option<Intent>,
    /// Ranges the reader gets in place of a first read: the paragraphs a comparison checks.
    pub evidence: &'a [Range],
    /// The chat's ID, whose source record says where each document's original is, for `read`'s
    /// pictures and tables; empty when there is none.
    pub chat: &'a str,
}

/// What a read_papers call needs for follow-ups: the chat, its last exchange, its kept readers,
/// and where its progress lines go.
#[derive(Clone, Default)]
pub struct Followup {
    /// The chat's ID, which names its source record.
    pub chat: String,
    /// The chat's readers, kept between its questions.
    pub readers: Arc<Mutex<Readers>>,
    /// The chat's last question and answer before this turn.
    pub last: Option<(String, String)>,
    /// The router's label of the turn.
    pub router: Option<Intent>,
    /// Where the call says what it is doing (`AgentEvent::Progress`).
    pub progress: Option<mpsc::Sender<String>>,
}

/// A chat's readers: one per set of documents a question was about, each kept with its session
/// between the chat's questions, and each document's summary from the call that read it.
#[derive(Default)]
pub struct Readers {
    /// How they run.
    pub settings: Settings,
    kept: HashMap<Vec<String>, Reader>,
    summaries: HashMap<String, String>,
}

/// One reader's session and records, kept between the chat's questions about its documents.
#[derive(Default)]
pub struct Reader {
    session: Option<Session>,
    /// The session holds each document whole, so a later question sends only itself.
    whole: bool,
    /// Tokens the session holds after its last answer.
    held: u64,
    /// Calls made in the session, which name their results.
    calls: u64,
    visible: Visible,
    searches: Searches,
    /// The task's progress, which a reset keeps (`read_all`'s).
    pub progress: Progress,
    /// The task `progress` is for.
    task: String,
    /// The last answer, whose cited pages a reset gives again.
    last_answer: String,
}

/// A reader's reply before its check: the text, how it ended, and what it read.
#[derive(Clone, Debug, PartialEq)]
pub struct Reply {
    /// The answer, trimmed.
    pub text: String,
    /// How its last step ended.
    pub finish: Finish,
    /// The ranges the reader was given for this question, in the order it got them.
    pub ranges: Vec<Range>,
    /// The answer came back empty and was asked for again without thinking.
    pub retried: bool,
    /// Not even a small tool result fitted, so its last step had no tools.
    pub no_room: bool,
    /// Its last step's answer says the text does not answer (`NOT_FOUND`).
    pub not_found: bool,
}

/// A reader's answer: what the user sees and what the trace and a workflow step keep.
#[derive(Clone, Debug, PartialEq)]
pub struct Answered {
    /// The reader's words, never rewritten.
    pub answer: String,
    /// The check's notes and Rust's own, shown under the answer.
    pub notes: Vec<String>,
    /// The ranges the reader read.
    pub ranges: Vec<Range>,
    /// The quote check ran on at least one cited quote.
    pub checked: bool,
    /// The reader said the text does not answer.
    pub not_found: bool,
}

impl Answered {
    /// The answer with its notes under it.
    pub fn markdown(&self) -> String {
        let parts = [self.answer.as_str()]
            .into_iter()
            .chain(self.notes.iter().map(String::as_str));
        parts
            .filter(|p| !p.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Whether `answer` is the reader's "not found" (`NOT_FOUND`), in any case.
pub fn not_found(answer: &str) -> bool {
    let stem = NOT_FOUND.trim_end_matches('.');
    (answer.trim_start().get(..stem.len())).is_some_and(|s| s.eq_ignore_ascii_case(stem))
}

impl Readers {
    /// Readers that run on `settings`.
    pub fn new(settings: Settings) -> Readers {
        Readers {
            settings,
            ..Readers::default()
        }
    }

    /// Keeps `summary` as the summary of the document with SHA-256 `sha256`.
    pub fn summarized(&mut self, sha256: &str, summary: String) {
        self.summaries.insert(sha256.into(), summary);
    }

    /// Answers `ask` from `docs` and gives the markdown the user sees: one reader on them all
    /// when they are within the whole-document rule together, else each document's reader under
    /// its name, then one comparison of their answers (plan 7.7) under `compare::HEADING`. `show`
    /// gets a lone reader's answer as it is written; `progress` hears "Reading 3 documents" for
    /// several, and `read_all`'s parts.
    #[allow(clippy::too_many_arguments)]
    pub fn answer(
        &mut self,
        larch: &Larch,
        ctx: &ToolCtx,
        docs: &[Doc],
        ask: Ask,
        stop: &AtomicBool,
        show: &mut dyn FnMut(&str),
        progress: &dyn Fn(&str),
    ) -> Result<String, SkillError> {
        if docs.len() > 1 {
            progress(&format!("Reading {} documents", docs.len()));
        }
        let s = self.settings;
        let summary = |docs: &[Doc]| -> String {
            let each = docs.iter().filter_map(|d| self.summaries.get(&d.sha256));
            each.cloned().collect::<Vec<_>>().join("\n\n")
        };
        let all = summary(docs);
        let together = Ask {
            summary: &all,
            ..ask
        };
        if docs.len() == 1 || whole(larch, &s, docs, &together) {
            let mut shown = Trimmed::new(show);
            let reader = self.kept.entry(key(docs)).or_default();
            let mut live = |t: &str| shown.push(t);
            let answered =
                reader.answer(larch, ctx, &s, docs, &together, stop, &mut live, progress)?;
            return Ok(answered.markdown());
        }
        let (mut sections, mut answers) = (Vec::new(), Vec::new());
        for doc in docs {
            let one = std::slice::from_ref(doc);
            let summary = summary(one);
            let ask = Ask {
                summary: &summary,
                ..ask
            };
            let reader = self.kept.entry(key(one)).or_default();
            let answered = reader.answer(larch, ctx, &s, one, &ask, stop, &mut |_| {}, progress)?;
            let name = read_papers::name(Path::new(&doc.name));
            sections.push(format!(
                "## {}. {name}\n{}",
                doc.letter(),
                answered.markdown()
            ));
            answers.push(answered);
        }
        progress("Comparing the answers");
        let question = compare::question(ask.question, docs, &answers);
        let evidence = compare::evidence(docs, &answers);
        let ask = Ask {
            question: &question,
            evidence: &evidence,
            ..together
        };
        // Its own reader: one on the documents together would read them whole.
        let mut k = key(docs);
        k.push("compare".into());
        let reader = self.kept.entry(k).or_default();
        let compared = reader.answer(larch, ctx, &s, docs, &ask, stop, &mut |_| {}, progress)?;
        sections.push(format!("{}\n{}", compare::HEADING, compared.markdown()));
        Ok(sections.join("\n\n"))
    }
}

/// A reader's key: its documents' snapshots.
fn key(docs: &[Doc]) -> Vec<String> {
    docs.iter().map(|d| d.sha256.clone()).collect()
}

/// Shows a reply as it streams, trimmed as its answer is, so the answer starts with what was
/// shown and a later step's notes follow it.
struct Trimmed<'a> {
    show: &'a mut dyn FnMut(&str),
    text: String,
    shown: usize,
}

impl<'a> Trimmed<'a> {
    fn new(show: &'a mut dyn FnMut(&str)) -> Trimmed<'a> {
        Trimmed {
            show,
            text: String::new(),
            shown: 0,
        }
    }

    fn push(&mut self, delta: &str) {
        self.text.push_str(delta);
        // Trailing white space waits: the answer's end is trimmed.
        let text = self.text.trim();
        if let Some(more) = text.get(self.shown..).filter(|m| !m.is_empty()) {
            (self.show)(more);
            self.shown = text.len();
        }
    }
}

/// What goes to the session next.
enum Next {
    User(String),
    Results(Vec<(String, String)>),
}

impl Reader {
    /// `reply`, then the quote check (`cite::check`), whose notes go under the answer, and its
    /// trace line.
    #[allow(clippy::too_many_arguments)]
    pub fn answer(
        &mut self,
        larch: &Larch,
        ctx: &ToolCtx,
        s: &Settings,
        docs: &[Doc],
        ask: &Ask,
        stop: &AtomicBool,
        live: &mut dyn FnMut(&str),
        progress: &dyn Fn(&str),
    ) -> Result<Answered, SkillError> {
        let r = self.reply(larch, ctx, s, docs, ask, stop, live, progress)?;
        let start = Instant::now();
        let checked = cite::check(&r.text, docs, &map::labels(docs));
        let mut notes = checked.notes;
        notes.extend(cite::numbers(&r.text, docs));
        if r.text.is_empty() {
            notes.push("The model gave an empty answer.".into());
        } else if r.finish == Finish::Length {
            notes.push("The answer stops at its length limit.".into());
        }
        if r.no_room {
            notes.push("The reader ran out of room in its context before it read more.".into());
        }
        let answered = Answered {
            not_found: r.not_found,
            answer: r.text,
            notes,
            ranges: r.ranges,
            checked: checked.ran,
        };
        let detail = json!({
            "ran": checked.ran,
            "cited": checked.cited,
            "matched": checked.matched,
            "notes": answered.notes.iter().map(|n| preview(n)).collect::<Vec<_>>(),
            "not_found": answered.not_found,
        });
        let event = AuditEvent {
            skill: Some(SKILL.into()),
            ..ask_docs::step("check", ms(start), Some(detail))
        };
        Audit::new(&ctx.data_dir)
            .append(event)
            .map_err(SkillError::Log)?;
        Ok(answered)
    }

    /// The reader's reply to `ask` about `docs`, before its check. In a fresh session the
    /// documents go whole when the whole-document rule lets them, else a first read with `find`
    /// and `read`, whose calls run until the model answers or no small result fits (plan 7.3,
    /// 7.4). A kept session gets only the question and what it does not hold yet. A prompt too
    /// long for the context resets the session once (7.5). An empty answer is asked for once
    /// more without thinking, never after `stop` (7.11). `live` gets the reply as it is written:
    /// a step that offers tools shows its text once it is prose with no call's opener, as the
    /// chat does (`run::streamed`), and what showed before a call stays the answer's start, so
    /// the reply is all that showed and nothing shows twice. `progress` hears `read_all`'s parts.
    /// A Stop or any other error resets the session (`reset`): it may hold a call with no result,
    /// and the text it was to get never reached it.
    #[allow(clippy::too_many_arguments)]
    pub fn reply(
        &mut self,
        larch: &Larch,
        ctx: &ToolCtx,
        s: &Settings,
        docs: &[Doc],
        ask: &Ask,
        stop: &AtomicBool,
        live: &mut dyn FnMut(&str),
        progress: &dyn Fn(&str),
    ) -> Result<Reply, SkillError> {
        (self.replied(larch, ctx, s, docs, ask, stop, live, progress)).inspect_err(|_| self.reset())
    }

    /// `reply`, which may leave the session as its error found it.
    #[allow(clippy::too_many_arguments)]
    fn replied(
        &mut self,
        larch: &Larch,
        ctx: &ToolCtx,
        s: &Settings,
        docs: &[Doc],
        ask: &Ask,
        stop: &AtomicBool,
        live: &mut dyn FnMut(&str),
        progress: &dyn Fn(&str),
    ) -> Result<Reply, SkillError> {
        let audit = Audit::new(&ctx.data_dir);
        let namespace = crate::namespace(&ctx.data_dir);
        let turn = all::Turn {
            larch,
            s,
            audit: &audit,
            namespace,
            stop,
            progress,
        };
        let names: Vec<String> = (docs.iter())
            .map(|d| read_papers::name(Path::new(&d.name)))
            .collect();
        let (mut thinking, mut reset, mut retried, mut no_room) = (s.thinking, false, false, false);
        // The answer's cap once the context leaves less than an answer step's.
        let mut cap: Option<u32> = None;
        let mut ranges = Vec::new();
        let mut given = Vec::new();
        let mut next = Next::User(self.opening(larch, s, docs, ask, false, &mut given)?);
        let mut step = "read";
        let mut said = Vec::new();
        let ratio = bytes_per_token(larch, docs);
        // What showed so far, and whether a step that showed text then called a tool, so the
        // next text shows after a blank line.
        let (mut out, mut gap) = (String::new(), false);
        loop {
            let tools = !self.whole && !no_room;
            let specs = specs(docs.len() > 1, !self.progress.parts.is_empty());
            let mut p = params(s, thinking, namespace, tools.then_some(&specs[..]));
            p.max_tokens = cap.map_or(p.max_tokens, |c| c.min(p.max_tokens));
            let detail = json!({
                "docs": names,
                "question": preview(ask.question),
                "router": ask.router,
                "ranges": given,
                "results": std::mem::take(&mut said),
            });
            let session = (self.session).get_or_insert_with(|| {
                larch.session(SessionOptions {
                    system: Some(system(!self.whole)),
                    keep_kv: true,
                    // Text never leaves a reader's session unseen: a reset says what it gives.
                    on_overflow: Overflow::Fail,
                })
            });
            let to = match &next {
                Next::User(text) => To::User(session, text),
                Next::Results(results) => To::Results(session, results.clone()),
            };
            let family = s.family.filter(|_| tools);
            let (mut text, mut shown, mut pieces) = (String::new(), 0, 0);
            let mut seen = |t: &str| {
                text.push_str(t);
                pieces += 1;
                let end = run::streamed(&text, shown, pieces, family);
                let more = &text[shown..end];
                shown = end;
                show(&mut out, &mut gap, live, more)
            };
            let sent = match send::send(larch, to, p.clone(), step, detail, &audit, stop, &mut seen)
            {
                // The room is the one hard stop: a session too full for the question starts
                // again with the text it needs (7.5); a step after the reads, or after that
                // reset, answers in what room is left, without thinking or tools, rather than
                // not at all.
                Err(SkillError::Engine(EngineError::PromptTooLong { tokens, limit })) => {
                    let prompt = u64::from(tokens).saturating_sub(answer_room(larch, &p));
                    let left = u64::from(limit).saturating_sub(prompt + MARGIN);
                    if !reset && (!no_room || left < MIN_ANSWER) {
                        (reset, step) = (true, "reset");
                        given.clear();
                        self.reset();
                        next = Next::User(self.opening(larch, s, docs, ask, true, &mut given)?);
                    } else if left >= MIN_ANSWER && cap.is_none() {
                        (no_room, step) = (true, "no room");
                        thinking.budget = 0;
                        cap = u32::try_from(left).ok();
                    } else {
                        return Err(SkillError::Engine(EngineError::PromptTooLong {
                            tokens,
                            limit,
                        }));
                    }
                    continue;
                }
                sent => sent?,
            };
            ranges.append(&mut given);
            let a = sent.answer;
            self.held = u64::from(a.usage.prompt_tokens) + count(larch, &a.text);
            let parsed = run::read(s.family.filter(|_| tools), &a.text, false);
            let mut calls: Vec<ToolCall> =
                (parsed.call.iter().chain(&parsed.later)).cloned().collect();
            if calls.is_empty() {
                let answer = parsed.text_before.as_str();
                if answer.trim().is_empty() && !retried && !stop.load(Ordering::Relaxed) {
                    (retried, step) = (true, "retry");
                    thinking.budget = 0;
                    next = Next::User(RETRY.into());
                    continue;
                }
                // The rest of the answer the stream held back; the answer is trimmed, what showed
                // is not.
                let rest = answer.strip_prefix(a.text[..shown].trim_start());
                show(&mut out, &mut gap, live, rest.unwrap_or_default());
                self.last_answer = out.trim().into();
                return Ok(Reply {
                    text: self.last_answer.clone(),
                    finish: a.finish,
                    ranges,
                    retried,
                    no_room,
                    not_found: not_found(answer),
                });
            }
            // What this step showed stays; the rest before its call is not the answer's.
            gap |= shown > 0;
            let session = self.session.as_mut().expect("the step was sent on it");
            let dropped = !session.takes_several_calls() && calls.len() > 1;
            if dropped {
                calls.truncate(1);
            }
            let ids: Vec<String> = (calls.iter())
                .map(|_| {
                    self.calls += 1;
                    format!("{:09}", self.calls)
                })
                .collect();
            let model: Vec<_> = (calls.iter().zip(&ids))
                .map(|(c, id)| run::model_call(c, id))
                .collect();
            (session.set_several_tool_calls(&parsed.text_before, model)).map_err(|e| {
                (e.downcast::<EngineError>()).map_or_else(
                    |e| SkillError::Tool(ToolError::Skill(e.to_string())),
                    SkillError::Engine,
                )
            })?;
            let room = limit(larch).saturating_sub(self.held + reserve(larch, &p));
            let n = calls.len() as u64;
            let mut results: Vec<(String, String)> = match room < SMALL_RESULT * n {
                true => {
                    no_room = true;
                    ids.into_iter().map(|id| (id, NO_ROOM.into())).collect()
                }
                false => {
                    let share = room / n;
                    let mut left = room;
                    (calls.iter().zip(ids))
                        .map(|(call, id)| {
                            left -= share;
                            // A tenth spare for the places and the result's lines.
                            let bytes = (share as f64 * ratio * 0.9) as usize;
                            let (out, got) = self.call(
                                &turn,
                                docs,
                                call,
                                bytes,
                                left as usize,
                                (ctx, ask, stop),
                            );
                            given.extend(got);
                            (id, out)
                        })
                        .collect()
                }
            };
            if dropped && let Some((_, last)) = results.last_mut() {
                last.push_str("\n[agent] One call per reply runs here; the others did not.");
            }
            said = (calls.iter().zip(&results))
                .map(
                    |(c, (_, out))| json!({"call": c.name, "args": c.args, "result": preview(out)}),
                )
                .collect();
            step = "after tools";
            next = Next::Results(results);
        }
    }

    /// The message that starts a question. A fresh session gets the documents (whole when the
    /// rule lets them, else the first read and the pages it gives), the summary and the chat's
    /// last exchange, then the question; a kept one gets the question, any part of its first read
    /// it does not hold, and the chat's last exchange when it is not the session's own. After a
    /// `reset`, the pages the last answer cited come first in the first read. All of it fits the
    /// room: a fresh request the count says is too long is not sent. `given` gets the ranges sent.
    fn opening(
        &mut self,
        larch: &Larch,
        s: &Settings,
        docs: &[Doc],
        ask: &Ask,
        reset: bool,
        given: &mut Vec<Range>,
    ) -> Result<String, SkillError> {
        let fresh = self.session.is_none();
        // The chat may have asked about other documents since this session's answer (7.3).
        let own = |a: &str| !self.last_answer.is_empty() && a.contains(self.last_answer.as_str());
        let ask = &Ask {
            last: ask.last.filter(|&(_, a)| fresh || !own(a)),
            ..*ask
        };
        if fresh {
            self.whole = whole(larch, s, docs, ask);
        }
        if self.whole {
            if !fresh {
                return Ok(message("", ask, false));
            }
            given.extend(docs.iter().map(all));
            return Ok(message(&whole_text(docs), ask, true));
        }
        let ran = !self.progress.parts.is_empty();
        let p = params(s, s.thinking, 0, Some(&specs(docs.len() > 1, ran)));
        // A fresh session's request is counted as the engine renders it, tools and all; a kept
        // one's count holds them already.
        let tokens = |text: &str| match fresh {
            true => first_request(larch, &system(!p.tools.is_empty()), &p, text),
            false => self.held + count(larch, text) + reserve(larch, &p),
        };
        let limit = limit(larch);
        let mut budget = (limit.saturating_sub(tokens(&message("", ask, fresh))))
            .min(u64::from(s.followup_tokens));
        let cited = match reset {
            true => cite::cited(&self.last_answer, docs, &map::labels(docs)),
            false => Vec::new(),
        };
        // The places and the left-out marks are the budget's too: a read they push past the
        // context is taken again in less.
        loop {
            let picked = first_read(larch, docs, ask, budget, &cited)?;
            let (above, new): (Vec<Range>, Vec<Range>) =
                (picked.into_iter()).partition(|r| self.visible.already(*r) == [*r]);
            let mut read = pieces(docs, &new, fresh);
            if !above.is_empty() {
                let several = docs.len() > 1;
                let places: Vec<String> = (above.iter())
                    .filter_map(|r| docs.iter().find(|d| d.index == r.doc).map(|d| (d, r)))
                    .map(|(d, r)| tools::place(d, several, *r))
                    .collect();
                read.push_str(&format!("{} already above.", places.join(", ")));
            }
            let text = message(read.trim_end(), ask, fresh);
            let n = tokens(&text);
            let over = n.saturating_sub(limit);
            if over > 0 && budget > 0 {
                budget = budget.saturating_sub(over);
                continue;
            }
            // A fresh count is the engine's own, but for a template that cannot render, whose
            // send says why; a kept one's is an estimate, and a refusal resets the session.
            if over > 0 && fresh && n != u64::MAX {
                let at_most = |t: u64| u32::try_from(t).unwrap_or(u32::MAX);
                return Err(SkillError::Engine(EngineError::PromptTooLong {
                    tokens: at_most(n),
                    limit: at_most(limit),
                }));
            }
            for r in &new {
                self.visible.seen(*r);
            }
            given.extend(new);
            return Ok(text);
        }
    }

    /// The session no longer fits: a new one, which holds nothing; the task's progress stays.
    fn reset(&mut self) {
        (self.session, self.whole, self.held, self.calls) = (None, false, 0, 0);
        self.visible.clear();
    }

    /// Runs the reader's `call` on `docs`: its result for the model, `bytes` of text at most,
    /// ending with `room` tokens left, and the ranges it gave; `read_all`'s parts run on `turn`.
    /// `here` is the turn's context, its ask and its stop, which `read`'s pictures and tables
    /// need (`extras`).
    fn call(
        &mut self,
        turn: &all::Turn,
        docs: &[Doc],
        call: &ToolCall,
        bytes: usize,
        room: usize,
        here: (&ToolCtx, &Ask, &AtomicBool),
    ) -> (String, Vec<Range>) {
        let specs = specs(docs.len() > 1, !self.progress.parts.is_empty());
        let functions: Vec<Value> = specs.iter().map(ToolSpec::function).collect();
        let mut call = call.clone();
        parse::coerce(&mut call, &functions);
        let Some(spec) = specs.iter().find(|s| s.name == call.name) else {
            let available = specs.iter().map(|s| s.name.clone()).collect();
            let name = call.name.clone();
            return (
                ValidateError::UnknownTool { name, available }.to_string(),
                Vec::new(),
            );
        };
        if let Err(e) = parse::validate(&call, spec) {
            return (e.to_string(), Vec::new());
        }
        let from = (call.args["from"].as_u64()).and_then(|f| usize::try_from(f).ok());
        // read_all's text never reaches the reader, only the items its part readers list.
        if call.name == all::READ_ALL {
            let task = call.args["task"].as_str().unwrap_or_default();
            let (kept, progress) = (&mut self.task, &mut self.progress);
            let out = all::read_all(turn, docs, kept, progress, task, bytes, room);
            return (out, Vec::new());
        }
        if call.name == all::ITEMS {
            let out = all::items(docs, &self.progress, from.unwrap_or(0), bytes, room);
            return (out, Vec::new());
        }
        if call.name == FIND {
            let words = call.args["words"].as_str().unwrap_or_default();
            let found = tools::find(docs, words, from.unwrap_or(0), bytes);
            let mut out = found.render(docs, words, room);
            if let Some(note) = self.searches.note(words, found.from, found.total) {
                out = format!("{out}\n{note}");
            }
            let got: Vec<Range> = found.hits.iter().map(|h| h.range).collect();
            for r in &got {
                self.visible.seen(*r);
            }
            return (out, got);
        }
        let doc = match call.args["doc"].as_str() {
            Some(l) => docs
                .iter()
                .find(|d| d.letter().eq_ignore_ascii_case(l.trim())),
            None => docs.first(),
        };
        let Some(doc) = doc else {
            let letters: Vec<String> = docs.iter().map(Doc::letter).collect();
            return (
                format!("no such document; the documents are {}", letters.join(", ")),
                Vec::new(),
            );
        };
        let part = call.args["part"].as_str().unwrap_or_default();
        let labels = map::labels(std::slice::from_ref(doc));
        let wants = |k: &str| call.args[k].as_bool() == Some(true);
        let (pictures, tables) = (wants("pictures"), wants("tables"));
        // What the pictures and tables add shares the room with the text.
        let text_bytes = if pictures || tables { bytes / 2 } else { bytes };
        match tools::read(doc, &labels, part, from, text_bytes) {
            Ok(piece) => {
                let mut out = piece.render(docs, &self.visible, room);
                if pictures || tables {
                    let mut more = extras(doc, piece.range, pictures, tables, here);
                    let left = bytes.saturating_sub(out.len());
                    if more.len() > left {
                        let cut = more.floor_char_boundary(left.saturating_sub(CUT.len()));
                        more.truncate(more[..cut].rfind('\n').map_or(0, |n| n + 1));
                        more.push_str(CUT);
                    }
                    // The room left stays the result's last line.
                    let at = out.rfind('\n').map_or(0, |n| n + 1);
                    out.insert_str(at, &more);
                }
                self.visible.seen(piece.range);
                (out, vec![piece.range])
            }
            Err(near) => (near, Vec::new()),
        }
    }
}

/// Shows `more` of a reply after `out`, what showed before it, and adds it there: after a blank
/// line when a step that showed text called a tool (`gap`). Whether anything showed.
fn show(out: &mut String, gap: &mut bool, live: &mut dyn FnMut(&str), more: &str) -> bool {
    if more.is_empty() {
        return false;
    }
    let newlines = out.len() - out.trim_end_matches('\n').len();
    let pad = match std::mem::take(gap) {
        true => &"\n\n"[newlines.min(2)..],
        false => "",
    };
    let piece = format!("{pad}{more}");
    live(&piece);
    out.push_str(&piece);
    true
}

/// What `read` adds for `pictures` and `tables` on the first `PICTURE_PAGES` pages of `r`, a
/// piece of `doc`: the text OCR reads in each page's pictures, marked, then the pages' tables as
/// Markdown, or their lines when the layout is uncertain. Each part says plainly what this
/// machine or the original cannot give. The original is the one the chat's source record names
/// (`turn`'s ask), unchanged since it was read; `turn`'s stop ends the reads.
fn extras(
    doc: &Doc,
    r: Range,
    pictures: bool,
    tables: bool,
    (ctx, ask, stop): (&ToolCtx, &Ask, &AtomicBool),
) -> String {
    // An empty piece is a page with no text, which starts where the next one does.
    let on = |p: &Page| match r.start == r.end {
        true => (p.start, p.end) == (r.start, r.end),
        false => p.start < r.end && r.start < p.end,
    };
    let mut pages: Vec<usize> = (doc.pages.iter().enumerate())
        .filter(|(_, p)| on(p))
        .map(|(i, _)| i)
        .collect();
    let mut out = String::new();
    if pages.len() > PICTURE_PAGES {
        out = format!(
            "Pictures and tables are read for at most {PICTURE_PAGES} pages a call: read from \
             page {} for the rest.\n",
            pages[PICTURE_PAGES] + 1
        );
        pages.truncate(PICTURE_PAGES);
    }
    let original = (source::load(&ctx.data_dir, ask.chat).ok())
        .and_then(|entries| entries.into_iter().nth(doc.index))
        .filter(|e| doc.rep == Rep::Pdf && e.sha256 == doc.sha256)
        .and_then(|e| source::original(&ctx.data_dir, &e));
    let Some(pdf) = original else {
        return match doc.rep {
            Rep::Pdf => format!(
                "The original PDF of {} is not there as it was read, so its pictures and tables \
                 cannot be read.\n",
                doc.name
            ),
            _ => format!(
                "{} is not a PDF: pictures and tables are read only from a PDF's pages.\n",
                doc.name
            ),
        };
    };
    if pictures {
        out += &pictures_of(doc, &pages, &pdf, &ctx.data_dir, stop);
    }
    if tables {
        out += &tables_of(&pages, &pdf, stop);
    }
    out
}

/// The text OCR reads in the pictures of `pages` of `doc`, whose original is `pdf`, each marked;
/// a page whose text was read from its pictures with the document says so.
fn pictures_of(
    doc: &Doc,
    pages: &[usize],
    pdf: &Path,
    data_dir: &Path,
    stop: &AtomicBool,
) -> String {
    if !ocr::available() {
        return format!("{}.\n", ocr::UNAVAILABLE);
    }
    let mut out = String::new();
    for &i in pages {
        let p = &doc.pages[i];
        out += &if doc.text[p.start..p.end].contains(&ocr::mark(i)) {
            format!(
                "Page {}'s text above already has what was read from its pictures.\n",
                i + 1
            )
        } else {
            match ocr::read(Some(pdf), &doc.sha256, i, data_dir, stop) {
                Ok(read) if read.lines.is_empty() => {
                    format!("No text was read from the pictures on page {}.\n", i + 1)
                }
                Ok(read) => format!("{}\n", read.marked()),
                Err(e) => format!("{e}\n"),
            }
        };
    }
    out
}

/// The tables on `pages` of `pdf`, from their words' places, joined where one runs on to the next
/// page (`table::join`).
fn tables_of(pages: &[usize], pdf: &Path, stop: &AtomicBool) -> String {
    let mut found = Vec::new();
    for &i in pages {
        match table::words(pdf, i, stop) {
            Ok(words) if words.is_empty() => {}
            Ok(words) => found.push(table::table(&words)),
            Err(e) => return format!("Tables cannot be read: {e}.\n"),
        }
    }
    let shown: Vec<String> = table::join(found)
        .iter()
        .map(table::Table::markdown)
        .collect();
    let (Some(first), Some(last)) = (pages.first(), pages.last()) else {
        return String::new();
    };
    let on = match first == last {
        true => format!("page {}", first + 1),
        false => format!("pages {} to {}", first + 1, last + 1),
    };
    match shown.is_empty() {
        true => format!("No words with places were found on {on}, so no table was read.\n"),
        false => format!(
            "[tables read from the places of the words on {on}]\n{}\n",
            shown.join("\n\n")
        ),
    }
}

/// The reader's tools: `find` and `read`, with the document's letter when it reads `several`; on
/// one document `read_all` too, and `items` once `read_all` `ran`. Several documents' reader is
/// the comparison, which checks answers rather than lists them.
fn specs(several: bool, ran: bool) -> Vec<ToolSpec> {
    let spec = |name: &str, description: &str, schema: Value| ToolSpec {
        name: name.into(),
        description: description.into(),
        schema,
        permission: Permission::Read,
        untrusted_results: true,
        // The loop fits each result to the room itself.
        max_result_tokens: u32::MAX,
        max_args_tokens: ARGS_TOKENS,
    };
    let from = json!({"type": "integer", "minimum": 0,
                      "description": "The from= a result's \"more:\" line gives"});
    let mut read = json!({
        "type": "object",
        "properties": {
            "part": {"type": "string",
                     "description": "Pages (\"p. 12\", \"pages 12-14\") or a heading"},
            "from": from,
            "pictures": {"type": "boolean",
                         "description": "Also the text inside the pages' pictures: a chart's or \
                                         a figure's words and numbers"},
            "tables": {"type": "boolean",
                       "description": "Also the pages' tables, as rows and columns"},
        },
        "required": ["part"],
        "additionalProperties": false
    });
    if several {
        read["properties"]["doc"] = json!({"type": "string",
                                           "description": "The document's letter: A, B, ..."});
    }
    let mut out = vec![
        spec(
            FIND,
            "Exact search of the documents' whole text: the lines that hold the words, each with \
             its place, and how many match. No match means these words are not in the text, not \
             that the idea is absent.",
            json!({
                "type": "object",
                "properties": {
                    "words": {"type": "string", "description": "The exact words to find"},
                    "from": from,
                },
                "required": ["words"],
                "additionalProperties": false
            }),
        ),
        spec(
            READ,
            "Read pages or a section of a document, with their places.",
            read,
        ),
    ];
    if several {
        return out;
    }
    out.push(spec(
        all::READ_ALL,
        "Read the whole text part by part for a task that needs all of it, such as \"list every \
         risk factor\": a reader of each part lists the items it finds, with quotes and pages. \
         It takes long on a long document.",
        json!({
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What to list, in the user's words"},
            },
            "required": ["task"],
            "additionalProperties": false
        }),
    ));
    // ponytail: items joins the tools once read_all ran, so the session prefills once more; offer
    // it from the start if the trace shows that prefill in the time
    if ran {
        out.push(spec(
            all::ITEMS,
            "The next of the items read_all found, from the from= its last line gives.",
            json!({
                "type": "object",
                "properties": {"from": from},
                "required": ["from"],
                "additionalProperties": false
            }),
        ));
    }
    out
}

/// A reader step's parameters: an answer step's (`ask_docs::answer_params`: greedy, its thinking
/// on `thinking`, its own `max_tokens`), with `tools` shown.
pub(super) fn params(
    s: &Settings,
    thinking: Thinking,
    namespace: u64,
    tools: Option<&[ToolSpec]>,
) -> GenParams {
    GenParams {
        tools: tools
            .into_iter()
            .flatten()
            .map(ToolSpec::function)
            .collect(),
        ..ask_docs::answer_params(s.family, thinking, namespace)
    }
}

/// Tokens of `text`.
pub(super) fn count(larch: &Larch, text: &str) -> u64 {
    larch.token_ids(text).len() as u64
}

/// Bytes of `docs`' text a token holds, as their first `SAMPLE` bytes each count, so a result's
/// bytes fit its tokens on any tokenizer and any script.
fn bytes_per_token(larch: &Larch, docs: &[Doc]) -> f64 {
    let each = docs
        .iter()
        .map(|d| &d.text[..d.text.floor_char_boundary(SAMPLE)]);
    let sample: String = each.collect();
    match count(larch, &sample) {
        0 => TOKEN_BYTES as f64,
        n => sample.len() as f64 / n as f64,
    }
}

/// Tokens one request may hold: the context, and no more than the KV pool when it is known.
pub(super) fn limit(larch: &Larch) -> u64 {
    let d = larch.deployment();
    let pool = u64::from(d.kv_pool_blocks) * KV_BLOCK_TOKENS;
    match pool {
        0 => u64::from(d.ctx_max),
        pool => u64::from(d.ctx_max).min(pool),
    }
}

/// Tokens a request on `p` keeps free for its reply, as the session counts them: `max_tokens`,
/// and the thinking close a budget cut feeds.
fn answer_room(larch: &Larch, p: &GenParams) -> u64 {
    let cut = (p.thinking_budget).filter(|&b| p.max_tokens == 0 || b < p.max_tokens);
    let close = cut.map_or(0, |_| count(larch, larch::text::THINKING_CLOSE));
    u64::from(p.max_tokens) + close
}

/// Tokens a request on `p` needs past what its session's last request counted: its reply's room
/// and the template around the new messages.
fn reserve(larch: &Larch, p: &GenParams) -> u64 {
    answer_room(larch, p) + TEMPLATE_TOKENS
}

/// Tokens a fresh session's first request on `p` with the system prompt `system` and the message
/// `text` holds, as the engine counts them against the context: the rendered prompt, its tools
/// too, and the reply's room. A template that cannot render it fits nothing; the send then says
/// why.
pub(super) fn first_request(larch: &Larch, system: &str, p: &GenParams, text: &str) -> u64 {
    let prompt = Prompt::Messages(run::messages(system, text, false));
    let rendered = larch.prompt_len(prompt, &p.tools, p.thinking);
    (rendered.map_or(u64::MAX, |n| n as u64)).saturating_add(answer_room(larch, p))
}

/// The whole-document rule (plan 7.3): `docs` go whole only within `followup_tokens` together and
/// within the room of one request with their message.
fn whole(larch: &Larch, s: &Settings, docs: &[Doc], ask: &Ask) -> bool {
    let text = whole_text(docs);
    let p = params(s, s.thinking, 0, None);
    count(larch, &text) <= u64::from(s.followup_tokens)
        && first_request(
            larch,
            &system(!p.tools.is_empty()),
            &p,
            &message(&text, ask, true),
        ) <= limit(larch)
}

/// A question's message: `read`, the text the session does not hold, then the summary in a
/// `fresh` session, the last exchange `ask` gives (its answer's first `LAST_BYTES`), and the
/// question.
fn message(read: &str, ask: &Ask, fresh: bool) -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.extend((!read.is_empty()).then(|| read.to_owned()));
    if fresh && !ask.summary.is_empty() {
        parts.push(format!("Summary from the first read:\n{}", ask.summary));
    }
    if let Some((q, a)) = ask.last {
        let a = match a.len() > LAST_BYTES {
            true => format!("{} {LEFT_OUT}", &a[..a.floor_char_boundary(LAST_BYTES)]),
            false => a.into(),
        };
        parts.push(format!("Last question: {q}\nLast answer: {a}"));
    }
    parts.push(format!("Question: {}", ask.question));
    parts.join("\n\n")
}

/// The whole of `doc`.
fn all(doc: &Doc) -> Range {
    Range {
        doc: doc.index,
        start: 0,
        end: doc.text.len(),
    }
}

/// `doc`'s heading line: its letter with several, its name and its page count.
pub(super) fn header(doc: &Doc, several: bool) -> String {
    let n = doc.pages.len();
    let s = if n == 1 { "" } else { "s" };
    match several {
        true => format!("Document {}: {} ({n} page{s})", doc.letter(), doc.name),
        false => format!("Document: {} ({n} page{s})", doc.name),
    }
}

/// Every document whole, each under its header, its place at each page.
fn whole_text(docs: &[Doc]) -> String {
    let several = docs.len() > 1;
    let each = docs.iter().map(|d| {
        let mut out = format!("{}\n", header(d, several));
        shown(&mut out, d, several, all(d));
        out
    });
    each.collect::<Vec<_>>().join("\n").trim_end().into()
}

/// Appends `r` of `doc` with its place at each page it reaches, as `tools::place` cites it; a
/// page with no text is left out.
pub(super) fn shown(out: &mut String, doc: &Doc, several: bool, r: Range) {
    for p in &doc.pages {
        let (a, b) = (p.start.max(r.start), p.end.min(r.end));
        let Some(text) = doc.text.get(a..b).filter(|t| !t.trim().is_empty()) else {
            continue;
        };
        let at = Range {
            doc: doc.index,
            start: a,
            end: a + 1,
        };
        out.push_str(&tools::place(doc, several, at));
        out.push('\n');
        out.push_str(text.replace('\x0c', "\n").trim_end());
        out.push('\n');
    }
}

/// The first read's `ranges` (merged, in document order) of `docs`, `LEFT_OUT` where text is left
/// out; in a `fresh` session under each document's header, with the pages it gives and the
/// map's headings.
fn pieces(docs: &[Doc], ranges: &[Range], fresh: bool) -> String {
    let several = docs.len() > 1;
    let mut out = String::new();
    for d in docs {
        let own: Vec<Range> = ranges
            .iter()
            .filter(|r| r.doc == d.index)
            .copied()
            .collect();
        if fresh {
            out.push_str(&header(d, several));
            out.push('\n');
        }
        let mut at = 0;
        for r in &own {
            if !d
                .text
                .get(at..r.start)
                .unwrap_or_default()
                .trim()
                .is_empty()
            {
                out.push_str(LEFT_OUT);
                out.push('\n');
            }
            shown(&mut out, d, several, *r);
            at = r.end;
        }
        if !d.text.get(at..).unwrap_or_default().trim().is_empty() {
            out.push_str(LEFT_OUT);
            out.push('\n');
        }
        if fresh {
            let places: Vec<String> = own.iter().map(|r| tools::place(d, false, *r)).collect();
            // A comparison's evidence can quote none of a document.
            let given = match places.is_empty() {
                true => "none".into(),
                false => places.join(", "),
            };
            out.push_str(&format!(
                "This read gives {given} of its {} pages; find and read reach the rest.\n{}\n",
                d.pages.len(),
                map::outline(d)
            ));
        }
    }
    out
}

/// The first read over the limit (plan 7.3): a comparison's evidence and `cited` first, the
/// evidence standing in for the rest; then the pages and labels the question names, then the
/// page-sized pieces that best match its words and the last exchange's (BM25), each whole, while
/// `budget` tokens hold them; merged, in document order.
fn first_read(
    larch: &Larch,
    docs: &[Doc],
    ask: &Ask,
    budget: u64,
    cited: &[Range],
) -> Result<Vec<Range>, SkillError> {
    let mut picked: Vec<Range> = Vec::new();
    let mut used = 0;
    let mut take = |r: Range, text: &str| {
        let n = count(larch, text);
        let overlaps =
            (picked.iter()).any(|p| p.doc == r.doc && p.start < r.end && r.start < p.end);
        if !overlaps && used + n <= budget {
            used += n;
            picked.push(r);
        }
    };
    for r in ask.evidence.iter().chain(cited) {
        let doc = docs.iter().find(|d| d.index == r.doc);
        take(
            *r,
            doc.and_then(|d| d.text.get(r.start..r.end))
                .unwrap_or_default(),
        );
    }
    if !ask.evidence.is_empty() {
        return Ok(merged(picked));
    }
    let bytes = usize::try_from(budget).unwrap_or(usize::MAX) * TOKEN_BYTES;
    let labels = map::labels(docs);
    let named = (NAMED.find_iter(ask.question).map(|m| m.as_str().to_owned()))
        .chain(map::named(ask.question));
    for named in named {
        for d in docs {
            if let Ok(piece) = tools::read(d, &labels, &named, None, bytes) {
                take(piece.range, &d.text[piece.range.start..piece.range.end]);
            }
        }
    }
    let (chunks, ranges) = page_chunks(docs);
    // "That" in the question means the last exchange's subject; its answer's start names it.
    // ponytail: the last answer's first LAST_BYTES; weigh its words when the trace shows the
    // first read missing what "that" means
    let query = match ask.last {
        Some((q, a)) => format!(
            "{} {q} {}",
            ask.question,
            &a[..a.floor_char_boundary(LAST_BYTES)]
        ),
        None => ask.question.into(),
    };
    // One below the count, as `rank` gives every chunk unranked when it may give them all.
    let top = chunks.len().saturating_sub(1).max(1);
    for (i, _) in ask_docs::rank(&chunks, &query, top)? {
        take(ranges[i], &chunks[i].text);
    }
    Ok(merged(picked))
}

/// `docs`' pages as chunks for the ranker, with their ranges; a page past `PAGE_BYTES` (a web
/// page or a text file is one page) is cut at line ends into pieces about that long.
fn page_chunks(docs: &[Doc]) -> (Vec<Chunk>, Vec<Range>) {
    let (mut chunks, mut ranges) = (Vec::new(), Vec::new());
    let heading: Arc<[String]> = Arc::from(Vec::new());
    for d in docs {
        for p in &d.pages {
            let mut start = p.start;
            while start < p.end {
                let mut end = p.end;
                if end - start > PAGE_BYTES {
                    let max = d.text.floor_char_boundary(start + PAGE_BYTES);
                    end = (d.text[start..max].rfind('\n')).map_or(max, |n| start + n + 1);
                }
                chunks.push(Chunk {
                    file: PathBuf::new(),
                    line: 0,
                    heading: Arc::clone(&heading),
                    start,
                    text: d.text[start..end].into(),
                });
                ranges.push(Range {
                    doc: d.index,
                    start,
                    end,
                });
                start = end;
            }
        }
    }
    (chunks, ranges)
}

/// `ranges` sorted, those that overlap or touch made one.
fn merged(mut ranges: Vec<Range>) -> Vec<Range> {
    ranges.sort();
    let mut out: Vec<Range> = Vec::new();
    for r in ranges {
        match out.last_mut() {
            Some(last) if last.doc == r.doc && r.start <= last.end => {
                last.end = last.end.max(r.end)
            }
            _ => out.push(r),
        }
    }
    out
}

/// The first `PREVIEW` characters of `text`, which the trace keeps.
pub(super) fn preview(text: &str) -> String {
    text.chars().take(PREVIEW).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::folio::{Page, Rep};

    fn doc(pages: &[&str]) -> Doc {
        let (mut text, mut out) = (String::new(), Vec::new());
        for (i, p) in pages.iter().enumerate() {
            if i > 0 {
                text.push('\x0c');
            }
            let start = text.len();
            text.push_str(p);
            out.push(Page {
                start,
                end: text.len(),
                label: None,
            });
        }
        Doc {
            index: 0,
            name: "/home/u/a.pdf".into(),
            sha256: "s".into(),
            rep: Rep::Pdf,
            text,
            pages: out,
        }
    }

    #[test]
    fn a_not_found_is_known_in_any_case_but_not_mid_answer() {
        assert!(not_found("  not found in the text. It covers rent."));
        assert!(!not_found(
            "The rent is \"$900\" [p. 2]. Not found in the text."
        ));
    }

    #[test]
    fn the_whole_text_marks_each_page_and_leaves_out_an_empty_one() {
        let d = doc(&["One.\n", "", "Three.\n"]);
        assert_eq!(
            whole_text(&[d]),
            "Document: /home/u/a.pdf (3 pages)\n[p. 1]\nOne.\n[p. 3]\nThree."
        );
    }

    #[test]
    fn a_first_read_shows_its_pieces_in_order_with_what_is_left_out() {
        let d = doc(&["One.\n", "Two.\n", "Three.\n", "Four.\n"]);
        let r = |p: usize| Range {
            doc: 0,
            start: d.pages[p].start,
            end: d.pages[p].end,
        };
        let got = pieces(std::slice::from_ref(&d), &[r(1), r(3)], true);
        assert_eq!(
            got,
            "Document: /home/u/a.pdf (4 pages)\n[...]\n[p. 2]\nTwo.\n[...]\n[p. 4]\nFour.\n\
             This read gives [p. 2], [p. 4] of its 4 pages; find and read reach the rest.\n\n"
        );
    }

    #[test]
    fn a_long_page_is_cut_at_line_ends_and_ranges_merge_when_they_touch() {
        let line = "x".repeat(99) + "\n";
        let d = doc(&[&line.repeat(70)]);
        let (chunks, ranges) = page_chunks(std::slice::from_ref(&d));
        assert_eq!(chunks.len(), 3);
        assert!(
            chunks
                .iter()
                .all(|c| c.text.len() <= PAGE_BYTES && c.text.ends_with('\n'))
        );
        assert_eq!(merged(ranges), [all(&d)]);
    }

    #[test]
    fn the_question_names_its_pages() {
        let named: Vec<&str> = (NAMED.find_iter("what do p. 12 and Pages 3 to 5 say, apps 7?"))
            .map(|m| m.as_str())
            .collect();
        assert_eq!(named, ["p. 12", "Pages 3 to 5"]);
    }

    #[test]
    fn a_fresh_message_has_the_summary_and_a_kept_one_only_the_last_exchange_it_is_given() {
        let ask = Ask {
            question: "And the deposit?",
            summary: "A lease.",
            last: Some(("The rent?", "$900 [p. 2].")),
            ..Ask::default()
        };
        assert_eq!(
            message("TEXT", &ask, true),
            "TEXT\n\nSummary from the first read:\nA lease.\n\nLast question: The rent?\nLast \
             answer: $900 [p. 2].\n\nQuestion: And the deposit?"
        );
        assert_eq!(
            message("", &ask, false),
            "Last question: The rent?\nLast answer: $900 [p. 2].\n\nQuestion: And the deposit?"
        );
        let own = Ask { last: None, ..ask };
        assert_eq!(message("", &own, false), "Question: And the deposit?");
        // A long last answer comes as its start.
        let long = "word ".repeat(200);
        let cut = message(
            "",
            &Ask {
                last: Some(("Q?", &long)),
                ..ask
            },
            false,
        );
        assert!(
            cut.contains(&format!("{} {LEFT_OUT}\n\nQuestion", &long[..LAST_BYTES])),
            "{cut}"
        );
    }
}
