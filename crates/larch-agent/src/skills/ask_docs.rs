//! The ask-docs skill: answer a question from a folder of the user's documents. Rust gathers and
//! ranks the chunks, one worker per chunk picks the numbered lines that answer it, Rust copies
//! each picked line as the fact's quote, and one session writes the answer from the numbered
//! facts, citing their numbers, which Rust turns into `file:line`.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs, io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Instant,
};

use larch::{
    EngineError, Finish, GenParams, Larch, SessionOptions,
    api::{Grammar, json_schema_regex},
};
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::runtime::inference::SeatAdmission;
use crate::{
    audit::{Audit, AuditEvent, EventKind},
    config::Thinking,
    folio::source,
    memory::{
        self,
        dates::{day_of_month, month_of, year_of},
        fts_query,
    },
    parse::Family,
    run::{self, Kind},
    skill::{Skill, SkillError, SkillInput, SkillRun, SkillSpec},
    skills::read_papers,
    tool::{Registry, ToolCtx, ToolError},
    workers::{Answer, Workers, answer},
};

/// Bytes in a chunk: about 256 tokens at four bytes a token. Small chunks rank sharper and keep
/// the workers' prefill short.
pub const CHUNK_BYTES: usize = 1024;
/// Bytes a token takes, as the agent counts everywhere: the window's unit.
const TOKEN_BYTES: usize = 4;
/// The bytes of a heading title a chunk keeps, cut at a character boundary: a one-line 1 MiB
/// heading would otherwise ride along with each of the file's 1,024 chunks.
pub const HEADING_BYTES: usize = 120;
/// The bytes of a chunk's header, cut at a character boundary.
pub const HEADER_BYTES: usize = 300;
/// Files read from one folder; the rest are counted, not read.
pub const MAX_FILES: usize = 200;
/// The largest file read; a larger one is counted, not read.
pub const MAX_FILE_BYTES: u64 = 1 << 20;
/// Entries `gather` and find_files look at before they stop: a home folder holds millions, and
/// search_files and find_files walk it on the chat's own thread.
pub const MAX_ENTRIES: usize = 20_000;

/// The most PDFs `gather` extracts for one question; one read before comes from the cache and is
/// not counted. The rest count as skipped, so a folder of 150 PDFs does not hold the answer for
/// minutes; the next question extracts the next ones.
pub const MAX_PDFS: usize = 20;

/// The most extensions a `NO_READABLE` line lists.
const MAX_SAW: usize = 5;
/// Chunks that go to the workers by default (config.toml's `docs_workers`): one group of the
/// default pool, which runs `max_workers - 1` = 7 sessions at once, so an eighth would wait for
/// the slowest of the first seven.
// ponytail: 7 by that arithmetic; the Mac bench's --docs-workers 7 against 14 picks the default
pub const DOCS_WORKERS: usize = 7;

/// Every worker's system prompt, under 200 tokens.
pub const WORKER_SYSTEM: &str = "You read one excerpt from the user's documents. It starts with \
its file and section, and each line starts with its number. List the facts in it that help answer \
the user's question. A fact helps only if it is about one of the things the question names: the \
same place, person, thing or year. A fact about something similar, such as another home, account \
or year, does not help; leave it out. A list the question asks for, such as ingredients, steps \
or prices, counts line by line, even where a line does not name the thing. Reply with only a JSON \
list and no other text: [{\"line\": 12, \"claim\": \"the fact in your own words, naming what it \
is about\"}], where line is the number of the line that shows the fact. If nothing in the excerpt \
helps answer the question, reply with only [].";

/// The orchestrator's system prompt, before `NOT_FOUND`. A worker can still pass on a near miss,
/// so the rule on what a fact must be about is said here again.
// Rust, not the model, says when two facts disagree (`conflicts`): asked to, the model called a
// rent and an insurance premium "conflicting values for the same thing".
const ANSWER_SYSTEM: &str = "Answer the user's question from the numbered facts below and \
nothing else. Each fact says the file and section it comes from. Use only facts about exactly what \
the question names: the same place, person, thing or year. A fact about something similar, such \
as another home, account or year, does not answer the question. After each fact you use, give its \
number in brackets, such as [2]. Be brief. If the question asks about two or more things and the \
facts answer only some of them, answer those and say which the documents do not answer. If they \
answer no part of it, reply with only this sentence: ";

/// The orchestrator's line for two facts, numbered `a` and `b`, that `conflicts` found to
/// disagree.
pub fn disagreeing(a: usize, b: usize) -> String {
    format!(
        "Facts [{a}] and [{b}] give different values for the same thing: give both, with their \
         numbers."
    )
}

/// The answer when no fact passed the line check, and the one the orchestrator is told to give
/// when the facts do not answer the question.
pub const NOT_FOUND: &str = "The documents do not answer this question.";

/// The starts of the first line of ask_docs's result in a chat (`head`), which Rust and the bench
/// read: what was read, or why nothing was.
pub const READ: &str = "Read ";
/// See `READ`: every PDF gave no text.
pub const NO_TEXT: &str = "No text in ";
/// See `READ`: files of other types only.
pub const NO_READABLE: &str = "No readable files in ";
/// See `READ`: no file at all.
pub const NO_MATCH: &str = "No files match ";

/// What follows `NO_READABLE` or `NO_MATCH`: what the user can do.
const ATTACH_DOCS: &str = "I can read .pdf, .md and .txt files. Attach a folder or files of \
those types, then ask again.";

/// What follows `NO_TEXT`: what the user can do.
const ATTACH_TEXT: &str = "I found no text in these PDFs: they look like scans, or locked or \
damaged files, which I can't read yet. Attach a copy with text (a searchable PDF, .md or .txt), \
then ask again.";

/// The first line of ask_docs's result for `g`, gathered from `folder`: `Read N files (k pdf, m
/// md, j txt)`, with `; no text in a.pdf` after it when a PDF gave none; else `No text in a.pdf`,
/// `No readable files in <folder> (saw: .docx)`, `No readable files in <folder> (3 skipped)` when
/// documents were there but not read, or `No files match <folder>`.
pub fn head(g: &Gathered, folder: &str) -> String {
    let blank: Vec<String> = g.blank.iter().map(|f| slashed(f)).collect();
    let blank = blank.join(", ");
    if g.files == 0 {
        return match (blank.is_empty(), g.skipped, g.others.is_empty()) {
            (false, _, _) => format!("{NO_TEXT}{blank}"),
            (true, 1.., _) => format!("{NO_READABLE}{folder} ({} skipped)", g.skipped),
            (true, 0, false) => {
                let mut saw: Vec<&str> = g.others.iter().map(String::as_str).collect();
                if saw.len() > MAX_SAW {
                    saw.truncate(MAX_SAW);
                    saw.push("…");
                }
                format!("{NO_READABLE}{folder} (saw: {})", saw.join(", "))
            }
            (true, 0, true) => format!("{NO_MATCH}{folder}"),
        };
    }
    let n = |e: &str| {
        (g.texts.keys())
            .filter(|f| extension(f).as_deref() == Some(e))
            .count()
    };
    let (s, _) = plural(g.files, "", "");
    let mut line = format!(
        "{READ}{} file{s} ({} pdf, {} md, {} txt)",
        g.files,
        n(".pdf"),
        n(".md"),
        n(".txt")
    );
    if !blank.is_empty() {
        line.push_str(&format!("; no text in {blank}"));
    }
    line
}

/// `result` without ask_docs's first line (`head`), and the line, if it has one.
pub(crate) fn split_head(result: &str) -> (Option<&str>, &str) {
    let (first, rest) = result.split_once("\n\n").unwrap_or((result, ""));
    let counted = (first.strip_prefix(READ))
        .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()) && r.contains(" file"));
    let head = counted
        || [NO_TEXT, NO_READABLE, NO_MATCH]
            .iter()
            .any(|h| first.starts_with(h));
    match head && !first.contains('\n') {
        true => (Some(first), rest),
        false => (None, result),
    }
}

/// The start of the line under each chat answer that names the folder searched.
pub(crate) const SEARCHED: &str = "\n\n_Searched the files in ";

/// The end of the note under `NOT_FOUND` when only the parts that best match the question were
/// read, not every part of the documents ("_Read 7 of 319 parts, those that best match the
/// question._"): `run` finds the partial read by it, so it is the note's whole fixed wording.
pub const PARTS_READ: &str = " parts, those that best match the question._";

/// The start of the note under `NOT_FOUND` when the answer was withheld for a name it lacks.
pub const WITHHELD: &str = "An answer was withheld";

/// The start of the note under a web answer whose cited pages never say a name of the question,
/// when the subject check did not run or failed.
pub const UNNAMED: &str = "The pages this answer cites never say";

/// The subject check's system prompt: whether the facts of a web answer whose pages never write a
/// name of the question are about it.
const SUBJECT_SYSTEM: &str = "You check whether facts from web pages are about what a question \
asks about. The pages never write some of the question's names: they may call the same thing \
another name (\"Texas\" or \"Longhorns\" for \"UT\", \"Federal Reserve\" for \"the Fed\"), or be \
about something else. Reply with only a JSON object: {\"about\": \"what the facts are about, as \
the pages name it\", \"verdict\": \"same\" if that is what the question asks about, \
\"different\" if it is something else, or \"unclear\" if the question could mean it or something \
else}.";

/// Tokens the subject check's reply may take.
const SUBJECT_TOKENS: u32 = 100;

/// The subject check's reply.
#[derive(Debug, Deserialize)]
struct Subject {
    about: String,
    verdict: Verdict,
}

/// Whether the facts are about what the question names.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Verdict {
    Same,
    Different,
    Unclear,
}

/// The error of a writer's turn whose thinking was cut and which then said "not found".
pub(crate) const RETRIED: &str = "thinking cut, then no answer: answered again without thinking";

/// The start of the note when some excerpts' workers gave no list of facts.
pub const NO_LIST: &str = "No list of facts came back";

/// A Llama 3 orchestrator's system prompt, before `NOT_FOUND`: its facts are the lines alone.
// The m1 audit at 97dc246 (PR #203): its workers' claims named the question's subject for a line
// about something else (the renters premium as "the Alder Street rent and the renters insurance
// together", the phone plan's line as "the gym membership"), its orchestrator answered from the
// claim over the line, and the part-answer sentence had it answer "which is cheaper" with "the
// documents do not answer".
// ponytail: Llama 3 only, from one model's audit; give the other families the lines alone when a
// bench of theirs shows the orchestrator taking a claim over its line
const LINES_ANSWER_SYSTEM: &str = "Answer the user's question from the numbered facts below and \
nothing else. Each fact is a line of the user's documents, with the file and section it comes \
from. Use only facts about exactly what the question names: the same place, person, thing or \
year. A fact about something similar, such as another home, account or year, does not answer the \
question. After each fact you use, give its number in brackets, such as [2]. When the question \
asks about two things, as a comparison or a sum does, give the fact for each, with its number, \
then the answer. Be brief. If no fact is about what the question asks, reply with only this \
sentence: ";

/// The answer turn's request: its system prompt, its prompt, and the facts the prompt numbers,
/// `[n]` being `facts[n - 1]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnswerRequest {
    /// The system prompt.
    pub system: String,
    /// The question, the numbered facts and a line for each pair that disagrees.
    pub prompt: String,
    /// The facts in the order the prompt numbers them.
    pub facts: Vec<Cited>,
}

/// The answer turn's request for `family`. Each fact comes with its file and the section its
/// line is in (`texts`, `section`), so that the router's password does not pass for the cabin's,
/// and each pair that disagrees (`facts.conflicts`) with `disagreeing`, so that the answer gives
/// both and an older status report's date does not pass for the newer one's. Every family but
/// Llama 3 gets each fact as its claim and line, every fact in `facts.cited` order; Llama 3 gets
/// each line once, as the line alone, and `LINES_ANSWER_SYSTEM`.
pub fn answer_request(
    family: Option<Family>,
    question: &str,
    facts: &Facts,
    texts: &BTreeMap<PathBuf, String>,
) -> AnswerRequest {
    request(
        family == Some(Family::Llama3),
        question,
        facts,
        texts,
        &BTreeMap::new(),
    )
}

/// `answer_request` with each fact as its line alone (`lines`, `LINES_ANSWER_SYSTEM`) or as its
/// claim and line, and the date of a file `dated` dates (a web page's).
pub(crate) fn request(
    lines: bool,
    question: &str,
    facts: &Facts,
    texts: &BTreeMap<PathBuf, String>,
    dated: &BTreeMap<PathBuf, String>,
) -> AnswerRequest {
    // The number each fact of `facts.cited` has in the prompt, by its index.
    let (mut shown, mut number): (Vec<Cited>, Vec<usize>) = (Vec::new(), Vec::new());
    for c in &facts.cited {
        match shown.iter().position(|s| lines && s.source() == c.source()) {
            Some(at) => number.push(at + 1),
            None => {
                shown.push(c.clone());
                number.push(shown.len());
            }
        }
    }
    let listed: String = (1..)
        .zip(&shown)
        .map(|(n, c)| {
            let text = texts.get(&c.file).map_or("", String::as_str);
            let from: Vec<_> = [slashed(&c.file)]
                .into_iter()
                .chain(section(text, c.line))
                .collect();
            let mut from = from.join(" > ");
            if let Some(day) = dated.get(&c.file) {
                from = format!("{from}, published {day}");
            }
            match lines {
                true => format!("[{n}] \"{}\" (from {from})\n", c.quote),
                false => format!("[{n}] {} (from {from}: \"{}\")\n", c.claim, c.quote),
            }
        })
        .collect();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for &(i, j) in &facts.conflicts {
        let pair = (number[i], number[j]);
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    }
    let disagree: String = (pairs.iter())
        .map(|&(a, b)| format!("{}\n", disagreeing(a, b)))
        .collect();
    let system = match lines {
        true => format!("{LINES_ANSWER_SYSTEM}{NOT_FOUND}"),
        false => format!("{ANSWER_SYSTEM}{NOT_FOUND}"),
    };
    AnswerRequest {
        system,
        prompt: format!("Question: {question}\n\nFacts:\n{listed}{disagree}"),
        facts: shown,
    }
}

/// Tokens of one reply, which has no thinking.
// ponytail: 1,024 per reply; raise it when the audit log shows replies cut at the limit
const REPLY_TOKENS: u32 = 1024;

/// The skill's name in the audit log and the daily notes.
const NAME: &str = "ask-docs";

/// Set to 1, the audit log holds each worker's whole reply, which quotes the documents; without
/// it, only the reply's length and hash.
pub const LOG_REPLIES: &str = "LARCH_AGENT_LOG_REPLIES";

/// A fact a worker reported: the excerpt's line that shows it and what it says. Other fields the
/// worker adds (a quote, a confidence) are not read: Rust copies the quote from the line.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    /// The line's number, as the excerpt shows it: the line of the file.
    pub line: u64,
    /// The fact, in the worker's words.
    pub claim: String,
}

/// Why a worker's reply is no list of facts; `Display` is the message the worker gets back.
#[derive(Debug)]
pub enum FactsError {
    /// The reply ended at `max_tokens` before the list was whole.
    Cut,
    /// The reply holds no JSON list of facts.
    Json(serde_json::Error),
}

impl fmt::Display for FactsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cut => f.write_str(
                "[agent] Your reply was cut off at the token limit before the list was complete. \
                 Reply with only the JSON list, or [], and no reasoning before it.",
            ),
            Self::Json(e) => write!(
                f,
                "[agent] Your reply was not a JSON list of facts: {e}. Reply with only the list, \
                 [{{\"line\": ..., \"claim\": ...}}], or with only [] if the excerpt does not help."
            ),
        }
    }
}

impl std::error::Error for FactsError {}

/// The facts in a worker's reply. Nothing found is the reply `[]` alone, blanks aside: any other
/// reply without facts is an error, so a broken list, a list in a checklist's "- [ ]" box or a
/// "[]" the model goes on to take back never pass for an excerpt that says nothing. Facts are the
/// first non-empty JSON list of facts that starts at a `[`, so a code fence, a sentence or
/// brackets around it do not matter. A `[` inside text that a failed parse already read starts no
/// list, and the error is the one of the parse that read furthest.
pub fn parse_facts(text: &str, finish: Finish) -> Result<Vec<Fact>, FactsError> {
    if text.trim() == "[]" {
        return Ok(Vec::new());
    }
    // The failed parse that read furthest: where it stopped, and why.
    let mut failed: Option<(usize, serde_json::Error)> = None;
    let mut empty = false;
    for (at, _) in text.match_indices('[') {
        if failed.as_ref().is_some_and(|(end, _)| at < *end) {
            continue;
        }
        let mut de = serde_json::Deserializer::from_str(&text[at..]);
        match Vec::<Fact>::deserialize(&mut de) {
            Ok(list) if !list.is_empty() => return Ok(list),
            Ok(_) => empty = true,
            Err(e) => failed = Some((at + stopped_at(&text[at..], &e), e)),
        }
    }
    if matches!(finish, Finish::Length) {
        return Err(FactsError::Cut);
    }
    let e = match failed {
        Some((_, e)) => e,
        // An empty list in a code fence, with words around it or a blank inside: say so.
        None if empty => serde::de::Error::custom(
            "an empty list is the reply [] alone, with no fence, words or blank around or in it",
        ),
        // No list at all: the parser says what it found instead.
        None => serde_json::from_str::<Vec<Fact>>(text)
            .err()
            .unwrap_or_else(|| serde::de::Error::custom("no list")),
    };
    Err(FactsError::Json(e))
}

/// What a worker may write: `[]` alone, or a list of `{"line", "claim"}` objects, which
/// `parse_facts` always reads. A small model held to nothing wrote a lone object, `{"line": 12,
/// ...}`, for most excerpts (the audit log of 28 Oct 2026, 40 of 53 failed replies), and a retry
/// fixed few of them.
pub fn facts_grammar() -> Grammar {
    let fact = json!({
        "type": "object",
        "properties": {"line": {"type": "integer"}, "claim": {"type": "string"}},
        "required": ["line", "claim"],
        "additionalProperties": false
    });
    let fact = json_schema_regex(&fact).expect("the fact's schema has a regex");
    // The engine's JSON whitespace (`grammar::WS`), but none in an empty list, which
    // `parse_facts` takes only as `[]`.
    let ws = r"(?: |\n {0,16})?";
    Grammar {
        trigger: None,
        pattern: format!(r"\[(?:\]|{ws}{fact}(?:{ws},{ws}{fact})*{ws}\])"),
    }
}

/// The byte of `text` where the parse that gave `e` stopped.
fn stopped_at(text: &str, e: &serde_json::Error) -> usize {
    let lines: usize = text
        .split_inclusive('\n')
        .take(e.line().saturating_sub(1))
        .map(str::len)
        .sum();
    lines + e.column()
}

/// What a worker reads: its chunk's header, then whole lines of the chunk's file, each with its
/// number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Excerpt {
    /// The file, relative to the folder asked about.
    pub file: PathBuf,
    /// The chunk's `Chunk::header`.
    pub header: String,
    /// The number of the first line.
    pub first: u32,
    /// The lines, without their line breaks.
    pub lines: Vec<String>,
}

impl Excerpt {
    /// The worker's prompt: the header, then each line after its number.
    pub fn render(&self) -> String {
        let mut out = format!("{}\n", self.header);
        for (n, line) in (self.first..).zip(&self.lines) {
            out.push_str(&format!("{n}: {line}\n"));
        }
        out
    }

    /// The fact as cited: its claim, and the text of the line it names as its quote, with every
    /// run of whitespace one space. A line the excerpt does not show cites nothing. A claim whose
    /// money amounts or dates its line lacks cites the nearest line of the excerpt that holds them
    /// all (`holding`), as when a worker names the line after the one with the date. A line with
    /// no text cites nothing.
    pub fn cite(&self, fact: &Fact) -> Option<Cited> {
        let at = usize::try_from(fact.line.checked_sub(u64::from(self.first))?).ok()?;
        self.lines.get(at)?;
        let at = self.holding(&fact.claim, at);
        let quote = squash(&self.lines[at]).0;
        (!quote.is_empty()).then(|| Cited {
            claim: fact.claim.clone(),
            quote,
            file: self.file.clone(),
            // A line the excerpt shows is a line of a file that `u32` counts.
            line: self
                .first
                .saturating_add(u32::try_from(at).unwrap_or(u32::MAX)),
        })
    }

    /// The index of the line that shows `claim`'s money amounts and dates: `at` when it holds
    /// them all or the claim has none, else the nearest line that holds them all (the earlier of
    /// two as near), else `at`.
    // ponytail: amounts and dates only; a claim that names the wrong thing for its line's value
    // (the renters premium as "the rent and the insurance together") keeps its line, and only the
    // Llama 3 orchestrator, which gets the line alone, is spared the claim
    fn holding(&self, claim: &str, at: usize) -> usize {
        let holds = |line: &String| holds(claim, line);
        if holds(&self.lines[at]) {
            return at;
        }
        (0..self.lines.len())
            .filter(|&i| holds(&self.lines[i]))
            .min_by_key(|&i| (i.abs_diff(at), i))
            .unwrap_or(at)
    }
}

/// Whether `text` shows all of `claim`'s money amounts and dates, or `claim` has none: the check a
/// cite must pass to support its sentence. Shared by every skill that cites lines.
pub fn holds(claim: &str, text: &str) -> bool {
    let (money, when) = (amounts(claim), dates(claim));
    let (m, d) = (amounts(text), dates(text));
    money.iter().all(|a| m.contains(a)) && when.iter().all(|a| d.iter().any(|b| same_date(a, b)))
}

/// The excerpt of `chunk` from `g`: its own text, and whole lines of its file on each side, up to
/// `window` tokens a side; a line that does not fit stops the side, as does the file's edge.
pub fn excerpt(g: &Gathered, chunk: &Chunk, window: u32) -> Excerpt {
    let budget = usize::try_from(window)
        .unwrap_or(usize::MAX)
        .saturating_mul(TOKEN_BYTES);
    // A chunk of no file of `g` reads alone.
    let file = g.texts.get(&chunk.file).map_or("", String::as_str);
    let end = chunk.start.saturating_add(chunk.text.len());
    let before = file.get(..chunk.start).unwrap_or_default();
    let back = fits(before.split_inclusive('\n').rev(), budget);
    let on = fits(
        file.get(end..).unwrap_or_default().split_inclusive('\n'),
        budget,
    );
    let text = file
        .get(chunk.start - back..end + on)
        .unwrap_or(&chunk.text);
    let first = chunk
        .line
        .saturating_sub(lines(&before[before.len() - back..]));
    Excerpt {
        file: chunk.file.clone(),
        header: chunk.header(),
        first,
        lines: text
            .split_inclusive('\n')
            .map(|l| l.trim_end_matches(['\n', '\r']).to_string())
            .collect(),
    }
}

/// The bytes of the first `pieces` that fit in `budget` together.
fn fits<'a>(pieces: impl Iterator<Item = &'a str>, budget: usize) -> usize {
    let mut taken = 0;
    for piece in pieces {
        if taken + piece.len() > budget {
            break;
        }
        taken += piece.len();
    }
    taken
}

/// `text` with its ends trimmed and every inner run of whitespace made one space, and for each
/// byte of the result the offset in `text` it came from.
pub(crate) fn squash(text: &str) -> (String, Vec<usize>) {
    let (mut out, mut from) = (String::new(), Vec::new());
    let mut gap = false;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            gap = !out.is_empty();
            continue;
        }
        if gap {
            out.push(' ');
            from.push(i);
            gap = false;
        }
        out.push(c);
        from.resize(out.len(), i);
    }
    (out, from)
}

/// The line breaks in `text`, as a line count.
fn lines(text: &str) -> u32 {
    u32::try_from(text.matches('\n').count()).unwrap_or(u32::MAX)
}

/// A piece of one file, cut at blank lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// The file, relative to the folder asked about.
    pub file: PathBuf,
    /// The line of the file the chunk starts on, counted from 1.
    pub line: u32,
    /// The titles of the `#` headings the chunk's first line is under, outermost first; a
    /// heading the chunk starts with is one of them. Each is at most `HEADING_BYTES`, and the
    /// chunks under the same headings share one list.
    pub heading: Arc<[String]>,
    /// The byte of the file the chunk starts at.
    pub start: usize,
    /// The file's text, exactly.
    pub text: String,
}

impl Chunk {
    /// The chunk's subject, which its indexed text and its excerpt start with: the file and the
    /// heading path, "# lease.md > Rent", at most `HEADER_BYTES`.
    pub fn header(&self) -> String {
        let path: Vec<_> = [slashed(&self.file)]
            .into_iter()
            .chain(self.heading.iter().cloned())
            .collect();
        let mut header = format!("# {}", path.join(" > "));
        header.truncate(header.floor_char_boundary(HEADER_BYTES));
        header
    }
}

/// What `gather` found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Gathered {
    /// Every chunk, in file order.
    pub chunks: Vec<Chunk>,
    /// The text of each file read, by its path relative to the folder asked about, which the
    /// excerpts' windows read around their chunks.
    pub texts: BTreeMap<PathBuf, String>,
    /// Each fetched page's title and the start of its text, where a page names its subject;
    /// empty for a folder.
    pub heads: BTreeMap<PathBuf, String>,
    /// Files read.
    pub files: usize,
    /// `.md` and `.txt` files not read: past `MAX_FILES`, over `MAX_FILE_BYTES`, not a plain
    /// file, a link out of the allowed folders, to nothing, to another type, to a hidden file or
    /// into a hidden folder or the agent's folder, a name that is not UTF-8, a read error.
    pub skipped: usize,
    /// Subfolders that could not be listed, relative to the folder asked about.
    pub unlisted: Vec<PathBuf>,
    /// The walk stopped after `MAX_ENTRIES` entries, so files past them were not looked at.
    pub stopped: bool,
    /// Read from a folder by `gather`: its chunks go by score alone (`rank`), as a folder of 25
    /// notes under `rank_pages` would give most of 7 slots to files that only share a word with
    /// the question. Fetched pages, each found for the question, go through `rank_pages`.
    pub folder: bool,
    /// The extensions of the files `gather` saw and does not read, such as `.docx`.
    pub others: BTreeSet<String>,
    /// PDFs that gave no text: a scan, or a file the reader could not open.
    pub blank: Vec<PathBuf>,
    /// `.md` and `.txt` files over `MAX_FILE_BYTES`, which are not read (counted in `skipped`
    /// too): the notes name them.
    pub oversized: Vec<PathBuf>,
    /// For each PDF read, the line each of its pages starts on, from the first page's 1:
    /// what turns a citation's line into its page (`Gathered::page`).
    pub pages: BTreeMap<PathBuf, Vec<u32>>,
}

impl Gathered {
    /// The page, from 1, that `line` of `file` is on, for a PDF whose pages are known.
    pub fn page(&self, file: &Path, line: u32) -> Option<usize> {
        let starts = self.pages.get(file)?;
        Some(starts.partition_point(|&s| s <= line).max(1))
    }
}

/// The line each page of `text` starts on, counted from 1 as `split` counts, for a text with a
/// form feed (`read_papers::PAGE_BREAK`) after each page: a feed does not end a line, so the
/// lines are the same with the feeds taken out.
pub fn page_lines(text: &str) -> Vec<u32> {
    let mut starts = vec![1];
    let mut line = 1u32;
    for c in text.chars() {
        match c {
            '\n' => line = line.saturating_add(1),
            read_papers::PAGE_BREAK => starts.push(line),
            _ => {}
        }
    }
    // The feed after the last page starts none.
    if text
        .trim_end_matches(['\n', '\r'])
        .ends_with(read_papers::PAGE_BREAK)
    {
        starts.pop();
    }
    starts
}

/// `markdown` with each `file:line` citation of a PDF whose pages `g` knows written as `file,
/// p. N`: the person reads a PDF by its pages, and a line of its extracted text ("pdf:4595")
/// names nothing they can find (Mac, 24 Nov). The audit log and the bench keep `file:line`.
/// Citations in one bracket join with "; ", since a page citation has its own comma
/// ("[a.pdf, p. 3; a.pdf, p. 4]", not "[a.pdf, p. 3, a.pdf, p. 4]").
pub fn with_pages(markdown: &str, facts: &Facts, g: &Gathered) -> String {
    let paged: BTreeMap<String, String> = (facts.cited.iter())
        .filter_map(|c| {
            let page = g.page(&c.file, c.line)?;
            Some((c.source(), format!("{}, p. {page}", slashed(&c.file))))
        })
        .collect();
    if paged.is_empty() {
        return markdown.to_owned();
    }
    bracketed(markdown, |inner| {
        let parts: Vec<&str> = inner.split(", ").collect();
        parts.iter().any(|p| paged.contains_key(*p)).then(|| {
            let mut shown: Vec<&str> = Vec::new();
            for p in parts
                .iter()
                .map(|p| paged.get(*p).map_or(*p, String::as_str))
            {
                // Two lines of one page cite it once.
                if !shown.contains(&p) {
                    shown.push(p);
                }
            }
            (shown.join("; "), 0)
        })
    })
    .0
}

/// Every `.pdf`, `.md` and `.txt` file under `folder` (or `folder` itself when it is one such
/// file), hidden ones skipped, cut into chunks of at most `CHUNK_BYTES`. The folder must resolve inside an allowed folder, which is checked before
/// anything is read, and so must each file: a link can lead out. The agent's own folder in the
/// data folder is left out wherever it is. A file or subfolder that cannot be read is counted, not
/// an error. The walk stops after `MAX_ENTRIES` entries, and `stopped` says so.
pub fn gather(ctx: &ToolCtx, folder: &str) -> Result<Gathered, ToolError> {
    let mut pdfs = MAX_PDFS;
    gather_until(ctx, folder, &AtomicBool::default(), &mut pdfs)
}

/// `gather` on `folder`, or with none on every folder and file attached, each folder's paths
/// under its name. One that no longer lists is counted in `unlisted`. `stop` ends each PDF's
/// extraction, which then counts as skipped.
pub fn gather_in(
    ctx: &ToolCtx,
    folder: Option<&str>,
    stop: &AtomicBool,
) -> Result<Gathered, ToolError> {
    let mut pdfs = MAX_PDFS;
    let all = match (folder, ctx.allowed.as_slice()) {
        (Some(folder), _) => return gather_until(ctx, folder, stop, &mut pdfs),
        (None, [] | [_]) => return gather_until(ctx, &first(ctx), stop, &mut pdfs),
        (None, all) => all,
    };
    let mut merged = Gathered {
        folder: true,
        ..Gathered::default()
    };
    for attached in all {
        let shown = attached.to_string_lossy();
        let g = match gather_until(ctx, &shown, stop, &mut pdfs) {
            Ok(g) => g,
            Err(_) => {
                merged.unlisted.push(attached.clone());
                continue;
            }
        };
        // An attached file is its folder's one document, already under its own name.
        let under = (attached.is_dir().then(|| attached.file_name()).flatten())
            .map_or_else(PathBuf::new, PathBuf::from);
        let at = |rel: PathBuf| under.join(rel);
        merged.chunks.extend(g.chunks.into_iter().map(|mut c| {
            c.file = at(c.file);
            c
        }));
        merged
            .texts
            .extend(g.texts.into_iter().map(|(f, t)| (at(f), t)));
        merged
            .pages
            .extend(g.pages.into_iter().map(|(f, p)| (at(f), p)));
        merged.files += g.files;
        merged.skipped += g.skipped;
        merged.unlisted.extend(g.unlisted.into_iter().map(at));
        merged.stopped |= g.stopped;
        merged.others.extend(g.others);
        merged.blank.extend(g.blank.into_iter().map(at));
    }
    Ok(merged)
}

/// The folder ask_docs reads when its call names none and one is attached: the first.
pub(crate) fn first(ctx: &ToolCtx) -> String {
    let first = ctx.allowed.first().and_then(|a| a.to_str());
    first.unwrap_or(".").into()
}

/// `gather`, which extracts at most `pdfs` PDFs and counts them off, and ends an extraction
/// once `stop` is set.
fn gather_until(
    ctx: &ToolCtx,
    folder: &str,
    stop: &AtomicBool,
    pdfs: &mut usize,
) -> Result<Gathered, ToolError> {
    let (root, found, mut gathered) = listed(ctx, folder)?;
    let (bases, data) = scope(ctx, &root);
    let data = data.as_deref();
    for rel in found {
        let text = (gathered.files < MAX_FILES)
            .then(|| read(ctx, &bases, &root.join(&rel), data, stop, pdfs))
            .flatten();
        let text = match text {
            Some(Some(text)) => text,
            Some(None) => {
                gathered.blank.push(rel);
                continue;
            }
            None => {
                gathered.skipped += 1;
                let path = root.join(&rel);
                if extension(&path).is_some_and(|e| e != ".pdf")
                    && fs::metadata(&path).is_ok_and(|m| m.len() > MAX_FILE_BYTES)
                {
                    gathered.oversized.push(rel);
                }
                continue;
            }
        };
        // A PDF's pages, from its form feeds, which no reader here needs in the text: a citation
        // of a PDF names its page (`with_pages`), not a line of the extracted text.
        let text = match text.contains(read_papers::PAGE_BREAK) {
            true => {
                gathered.pages.insert(rel.clone(), page_lines(&text));
                text.replace(read_papers::PAGE_BREAK, "")
            }
            false => text,
        };
        gathered.chunks.extend(split(&rel, &text));
        // ponytail: a second copy of each file beside its chunks, up to MAX_FILES MB, which the
        // windows read; hold ranges into it instead when a folder that size runs short of memory
        gathered.texts.insert(rel, text);
        gathered.files += 1;
    }
    Ok(gathered)
}

/// The folder `folder` resolves to, the documents under it as `walk` finds them, and what the walk
/// saw. An attached file is its own folder's one document.
fn listed(ctx: &ToolCtx, folder: &str) -> Result<(PathBuf, Vec<PathBuf>, Gathered), ToolError> {
    let root = ctx.resolve(folder, false)?;
    let mut g = Gathered {
        folder: true,
        ..Gathered::default()
    };
    if let (Some(parent), Some(name)) = (root.parent().filter(|_| root.is_file()), root.file_name())
    {
        let found = match is_doc(Path::new(name)) {
            true => vec![name.into()],
            false => {
                g.others.extend(extension(Path::new(name)));
                Vec::new()
            }
        };
        return Ok((parent.into(), found, g));
    }
    let data = scope(ctx, &root).1;
    let (mut found, mut left) = (Vec::new(), MAX_ENTRIES);
    walk(
        &root,
        Path::new(""),
        data.as_deref(),
        &mut left,
        &mut found,
        &mut g,
    )
    .map_err(|e| ToolError::Io(folder.into(), e))?;
    Ok((root, found, g))
}

/// The documents `gather` would read under `folder`, counted without reading them: what the app
/// shows when a folder is attached. The walk stops after `MAX_ENTRIES` entries.
pub fn count(ctx: &ToolCtx, folder: &str) -> Result<usize, ToolError> {
    Ok(listed(ctx, folder)?.1.len())
}

/// The documents `gather_in` would read from every attached folder and file, in the order they
/// were attached, each as its name there and its real path: what the folio module reads whole.
/// Which files are read is `gather`'s rule (`visible`), and an attached folder that no longer lists
/// is an error, which ask_docs says in its own words.
pub(crate) fn documents(ctx: &ToolCtx) -> Result<Vec<(String, PathBuf)>, ToolError> {
    let mut out = Vec::new();
    for attached in &ctx.allowed {
        let (root, found, _) = listed(ctx, &attached.to_string_lossy())?;
        let (bases, data) = scope(ctx, &root);
        // Under the folder's own name when more than one is attached, as `gather_in` names them.
        let under = (ctx.allowed.len() > 1 && attached.is_dir())
            .then(|| attached.file_name().map(PathBuf::from))
            .flatten()
            .unwrap_or_default();
        for rel in found {
            let seen = visible(ctx, &bases, &root.join(&rel), data.as_deref());
            if let Some((real, _)) = seen.filter(|(_, m)| m.is_file()) {
                out.push((slashed(&under.join(&rel)), real));
            }
        }
    }
    Ok(out)
}

/// `path`'s extension as the first line lists it: `.docx`, in lower case.
fn extension(path: &Path) -> Option<String> {
    let e = path.extension()?.to_str()?;
    Some(format!(".{}", e.to_lowercase()))
}

/// What `gather` and find_files check the paths under `root` against: the folders `visible` looks
/// for hidden folders below, and the agent's own folder, which they leave out.
pub(crate) fn scope(ctx: &ToolCtx, root: &Path) -> (Vec<PathBuf>, Option<PathBuf>) {
    // Where a file's path starts to be checked for hidden folders: `root`, or the allowed folder a
    // link leads into. The user may name a hidden folder, or one below a hidden folder (iCloud
    // Drive is below ~/Library), but not the hidden folders above it.
    let bases = (ctx.allowed.iter())
        .filter_map(|a| fs::canonicalize(a).ok())
        .chain([root.to_path_buf()])
        .collect();
    // Its notes and audit log are the agent's, not the user's documents; the models and chats
    // beside it are no `.md` or `.txt` file.
    (bases, fs::canonicalize(ctx.data_dir.join("agent")).ok())
}

/// The text of the file at `path`, if it is `visible`, a document (`is_doc`), and a plain file
/// of at most `MAX_FILE_BYTES` (a PDF: read_papers' cap) that reads; `Some(None)` for a PDF that
/// gives no text. A PDF past the `pdfs` left, or stopped, is not read.
fn read(
    ctx: &ToolCtx,
    bases: &[PathBuf],
    path: &Path,
    data: Option<&Path>,
    stop: &AtomicBool,
    pdfs: &mut usize,
) -> Option<Option<String>> {
    let (real, meta) = visible(ctx, bases, path, data)?;
    let pdf = extension(&real).is_some_and(|e| e == ".pdf");
    let cap = if pdf {
        read_papers::MAX_FILE_BYTES
    } else {
        MAX_FILE_BYTES
    };
    if !is_doc(&real) || !meta.is_file() || meta.len() > cap {
        return None;
    }
    if pdf {
        let text = pdf_text(ctx, &real, &meta, stop, pdfs)?;
        return Some(Some(text).filter(|t| !t.trim().is_empty()));
    }
    Some(Some(
        String::from_utf8_lossy(&fs::read(&real).ok()?).into_owned(),
    ))
}

/// The text of the PDF at `real`, from the extract cache (`source::cached`) when it was read
/// before, so a changed file misses by construction. Empty when the reader fails on it, which is
/// cached too, so a broken PDF is tried once. `None`, with nothing cached, when `pdfs` is 0 or
/// `stop` ends the extraction; one extracted counts off `pdfs`.
fn pdf_text(
    ctx: &ToolCtx,
    real: &Path,
    meta: &fs::Metadata,
    stop: &AtomicBool,
    pdfs: &mut usize,
) -> Option<String> {
    let stamp = source::Stamp::of(meta)?;
    let text = match source::cached(&ctx.data_dir, real, stamp) {
        Some(text) => text,
        None if *pdfs == 0 || stop.load(Ordering::Relaxed) => return None,
        None => {
            *pdfs -= 1;
            let text = match read_papers::text_of(real, stop) {
                Ok(text) => text,
                Err(_) if stop.load(Ordering::Relaxed) => return None,
                Err(_) => String::new(),
            };
            // A cache that cannot be written only costs the next question another extraction;
            // written whole or not at all, so a crash leaves no cut text behind.
            let _ = source::save(&ctx.data_dir, real, stamp, &text);
            text
        }
    };
    // The built-in reader starts with blank lines that pdftotext leaves out: without them, both
    // number the lines alike, so a citation's line is the same with poppler installed or not.
    // The cache keeps a form feed after each page, which `gather_until` reads the pages from.
    Some(text.trim_start_matches(['\n', '\r']).to_owned())
}

/// The real path of `path` and its metadata, if it resolves inside an allowed folder and outside
/// `data`, to something visible, as is each folder above it below the deepest of `bases` that
/// holds it (a link can lead into a hidden folder). A name that is not UTF-8 cannot go through
/// the check, so it is not visible either.
pub(crate) fn visible(
    ctx: &ToolCtx,
    bases: &[PathBuf],
    path: &Path,
    data: Option<&Path>,
) -> Option<(PathBuf, fs::Metadata)> {
    let real = ctx.resolve(path.to_str()?, false).ok()?;
    let meta = fs::metadata(&real).ok()?;
    let base = bases
        .iter()
        .filter(|b| real.starts_with(b))
        .max_by_key(|b| b.as_os_str().len())?;
    let hid = real
        .ancestors()
        .take_while(|a| *a != base.as_path())
        .any(|a| hidden(a, fs::metadata(a).ok().as_ref()));
    (!hid && !data.is_some_and(|d| real.starts_with(d))).then_some((real, meta))
}

/// The `.md` and `.txt` files under `dir`, as paths under `rel`, in name order. Hidden entries
/// and the data folder are skipped and links to folders are not followed; a subfolder that cannot
/// be listed goes to `g.unlisted`, an entry that cannot be read adds one to `g.skipped`. It looks
/// at `left` entries at most, counts them off, and sets `g.stopped` when there were more.
// ponytail: depth first in name order, as before the cap, so one large folder early in the tree
// can use the whole cap; walk breadth first, as find_files does, when a search of a home folder
// misses its later folders
fn walk(
    dir: &Path,
    rel: &Path,
    data: Option<&Path>,
    left: &mut usize,
    out: &mut Vec<PathBuf>,
    g: &mut Gathered,
) -> io::Result<()> {
    // Taken before the sort, so a folder of a million names is not read whole.
    let mut entries = (fs::read_dir(dir)?.take(*left + 1)).collect::<io::Result<Vec<_>>>()?;
    g.stopped |= entries.len() > *left;
    entries.truncate(*left);
    *left -= entries.len();
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        if hidden(&path, entry.metadata().ok().as_ref()) || data == Some(path.as_path()) {
            continue;
        }
        let rel = rel.join(entry.file_name());
        match entry.file_type() {
            Ok(t) if t.is_dir() => {
                if walk(&path, &rel, data, left, out, g).is_err() {
                    g.unlisted.push(rel);
                }
            }
            Ok(_) if is_doc(&rel) => out.push(rel),
            Ok(_) => g.others.extend(extension(&rel)),
            Err(_) => g.skipped += 1,
        }
    }
    Ok(())
}

/// Whether `folder` is a folder of papers: everything in it that is not hidden is a `.pdf`, `.md`
/// or `.txt` file, no subfolder. A repo or a folder of notes and code is not one, whatever
/// README it holds.
pub(crate) fn papers_folder(folder: &Path) -> bool {
    let Ok(entries) = fs::read_dir(folder) else {
        return false;
    };
    let mut any = false;
    for e in entries.flatten() {
        let (path, meta) = (e.path(), e.metadata().ok());
        if hidden(&path, meta.as_ref()) {
            continue;
        }
        if !(meta.is_some_and(|m| m.is_file()) && is_doc(&path)) {
            return false;
        }
        any = true;
    }
    any
}

/// Whether the path names a `.pdf`, `.md` or `.txt` file.
fn is_doc(path: &Path) -> bool {
    extension(path).is_some_and(|e| DOCS.contains(&e.as_str()))
}

/// The documents ask_docs reads, as the first line of its result counts them.
const DOCS: [&str; 3] = [".pdf", ".md", ".txt"];

/// Whether the file is hidden: its name starts with `.`, or macOS or Windows marks it hidden in
/// `meta`, as they mark ~/Library and AppData, where the agent's own data lives.
#[cfg_attr(not(any(target_os = "macos", windows)), allow(unused_variables))]
pub(crate) fn hidden(path: &Path, meta: Option<&fs::Metadata>) -> bool {
    let dot = path
        .file_name()
        .is_some_and(|n| n.to_string_lossy().starts_with('.'));
    #[cfg(target_os = "macos")]
    let flag = {
        use std::os::macos::fs::MetadataExt;
        const UF_HIDDEN: u32 = 0x8000;
        meta.is_some_and(|m| m.st_flags() & UF_HIDDEN != 0)
    };
    #[cfg(windows)]
    let flag = {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_HIDDEN: u32 = 0x2;
        meta.is_some_and(|m| m.file_attributes() & FILE_ATTRIBUTE_HIDDEN != 0)
    };
    #[cfg(not(any(target_os = "macos", windows)))]
    let flag = false;
    dot || flag
}

/// `text` in chunks of at most `CHUNK_BYTES`: whole paragraphs (runs of non-blank lines) packed
/// together, and a paragraph over the cap cut at the last line break below it, or inside a line
/// over the cap after its last sentence end in the second half (else its last space there, else
/// its last character boundary). Each chunk knows the
/// headings above its first line.
// ponytail: a `#` line inside a code fence reads as a heading; track fences when a user's notes
// with shell comments show wrong headers in the audit log
pub(crate) fn split(file: &Path, text: &str) -> Vec<Chunk> {
    split_at(file, text, CHUNK_BYTES)
}

/// `split` with chunks of at most `cap` bytes: a passage card's smaller ones.
pub(crate) fn split_at(file: &Path, text: &str, cap: usize) -> Vec<Chunk> {
    let mut pieces = Vec::new();
    let mut cut = |mut start: usize, end: usize| {
        while end - start > cap {
            // At the last line break that fits, so the excerpts' lines stay whole; inside the
            // line only when one line is over the cap.
            let limit = text.floor_char_boundary(start + cap);
            let piece = &text[start..limit];
            let at = match piece.rfind('\n') {
                Some(i) => start + i + 1,
                // A quote is copied from one chunk, so a cut mid-sentence or mid-word splits the
                // sentence a reader would quote.
                None => {
                    let half = |&i: &usize| i > cap / 2;
                    let sentence = (piece.rmatch_indices(['.', '!', '?']).map(|(i, _)| i + 1))
                        .take_while(half)
                        .find(|&i| piece[i..].starts_with(' '))
                        .map(|i| i + 1);
                    let space = piece.rfind(' ').map(|i| i + 1).filter(half);
                    start + sentence.or(space).unwrap_or(piece.len())
                }
            };
            pieces.push((start, at));
            start = at;
        }
        pieces.push((start, end));
    };
    let (mut paragraph, mut at) = (None, 0);
    for line in text.split_inclusive('\n') {
        match (line.trim().is_empty(), paragraph) {
            (false, None) => paragraph = Some(at),
            (true, Some(start)) => {
                cut(start, at);
                paragraph = None;
            }
            _ => {}
        }
        at += line.len();
    }
    if let Some(start) = paragraph {
        cut(start, at);
    }
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for (start, end) in pieces {
        match spans.last_mut() {
            Some(span) if end - span.0 <= cap => span.1 = end,
            _ => spans.push((start, end)),
        }
    }
    let (mut line, mut counted) = (1u32, 0);
    // The headings open at `read`, with their levels, and their titles as the chunks share them.
    let (mut open, mut read): (Vec<(usize, String)>, usize) = (Vec::new(), 0);
    let mut titles: Arc<[String]> = Arc::new([]);
    spans
        .into_iter()
        .map(|(start, end)| {
            line = line.saturating_add(lines(&text[counted..start]));
            counted = start;
            // Every line up to the end of the chunk's first one, which may be a heading.
            let first = text[start..]
                .find('\n')
                .map_or(text.len(), |i| start + i + 1);
            for l in text[read.min(first)..first].split_inclusive('\n') {
                if let Some((level, title)) = heading(l) {
                    open.retain(|&(l, _)| l < level);
                    open.push((level, title.into()));
                    titles = open.iter().map(|(_, t)| t.clone()).collect();
                }
            }
            read = read.max(first);
            Chunk {
                file: file.into(),
                line,
                heading: Arc::clone(&titles),
                start,
                text: text[start..end].into(),
            }
        })
        .collect()
}

/// The level and title of a Markdown heading line ("## Rent" is 2 and "Rent"), the title cut to
/// `HEADING_BYTES`.
pub(crate) fn heading(line: &str) -> Option<(usize, &str)> {
    let line = line.trim_end();
    let level = line.bytes().take_while(|&b| b == b'#').count();
    let title = line[level..].strip_prefix(' ')?.trim();
    let title = title[..title.floor_char_boundary(HEADING_BYTES)].trim_end();
    ((1..=6).contains(&level) && !title.is_empty()).then_some((level, title))
}

/// The indices of the chunks the workers read, best first, each with its BM25 score (higher is
/// better): every chunk, with no score, when there are at most `top`, else the best `top` of
/// those that share a word with the question (`rank_query`: stop words aside), by BM25 in an
/// in-memory FTS5 table over each chunk's header and text, so a chunk's file name and headings
/// name its subject.
// ponytail: FTS5's unicode61 tokens split at spaces, so text without them (Chinese, Japanese)
// matches nothing and the run says so; switch the table to the trigram tokenizer when users ask
// about documents in such languages
pub fn rank(
    chunks: &[Chunk],
    question: &str,
    top: usize,
) -> rusqlite::Result<Vec<(usize, Option<f64>)>> {
    if chunks.len() <= top {
        return Ok((0..chunks.len()).map(|i| (i, None)).collect());
    }
    scored(chunks, &rank_query(question), top)
}

/// `chosen` (best first) with each name the question gives (`question_names`) that a chunk holds
/// read at least once: the chunk holding the most names no chosen chunk holds yet joins, in a free
/// slot or in place of the worst chosen chunk whose names stay read without it (by another pick
/// or by the one joining), the better half of `top` kept.
/// A datasheet question naming its packages (WQFN, VSSOP) ranked the crystal's prose over the
/// package table, and the answer was withheld for want of a fact naming them (Windows, 24 Nov).
// ponytail: ties go to the earlier chunk, not the better scored; score them if a run picks a
// table of contents that lists every name over the table itself
pub fn cover_names(
    chunks: &[Chunk],
    mut chosen: Vec<(usize, Option<f64>)>,
    question: &str,
    top: usize,
) -> Vec<(usize, Option<f64>)> {
    let names: Vec<String> = question_names(question)
        .iter()
        .map(|n| n.to_lowercase())
        .collect();
    if names.is_empty() {
        return chosen;
    }
    let words: Vec<BTreeSet<String>> = (chunks.iter())
        .map(|c| {
            format!("{}\n{}", c.header(), c.text)
                .split(|ch: char| !ch.is_alphanumeric())
                .map(str::to_lowercase)
                .collect()
        })
        .collect();
    let keep = top.div_ceil(2);
    loop {
        let open: Vec<&String> = (names.iter())
            .filter(|n| !chosen.iter().any(|&(i, _)| words[i].contains(*n)))
            .collect();
        let held = |i: usize| open.iter().filter(|n| words[i].contains(**n)).count();
        let best = (0..chunks.len())
            .filter(|i| !chosen.iter().any(|c| c.0 == *i))
            .map(|i| (held(i), std::cmp::Reverse(i)))
            .filter(|&(n, _)| n > 0)
            .max();
        let Some((_, std::cmp::Reverse(best))) = best else {
            return chosen;
        };
        if chosen.len() < top {
            chosen.push((best, None));
            continue;
        }
        // Each swap reads one more name and loses none, so the loop ends.
        let needed = |k: usize| {
            (names.iter()).any(|n| {
                words[chosen[k].0].contains(n)
                    && !words[best].contains(n)
                    && !(chosen.iter().enumerate()).any(|(j, c)| j != k && words[c.0].contains(n))
            })
        };
        match (keep..chosen.len()).rev().find(|&k| !needed(k)) {
            Some(k) => chosen[k] = (best, None),
            None => return chosen,
        }
    }
}

/// `rank` with each word of the question as a prefix (`stems`): "weigh" finds "Weight" and
/// "headphones" finds "headphone", which a passage card's few chunks cannot afford to miss.
pub(crate) fn rank_stems(
    chunks: &[Chunk],
    question: &str,
    top: usize,
) -> rusqlite::Result<Vec<(usize, Option<f64>)>> {
    if chunks.len() <= top {
        return Ok((0..chunks.len()).map(|i| (i, None)).collect());
    }
    scored(chunks, &stems(question), top)
}

/// The FTS5 query of the question's `rank_words`, each exact and as a prefix, so the exact form
/// scores more: a word of 4 or more letters without its last 2 (at least 4 kept). A word with a
/// digit (a model number: "1000XM6" is not "1000XM5") or under 4 letters ("UT") stays exact.
fn stems(question: &str) -> String {
    let words: Vec<String> = (rank_words(question).into_iter())
        .map(|w| {
            let w = w.to_lowercase();
            let n = w.chars().count();
            match n < 4 || w.contains(|c: char| c.is_ascii_digit()) {
                true => format!("\"{w}\""),
                false => {
                    let stem: String = w.chars().take((n - 2).max(4)).collect();
                    format!("\"{w}\" OR \"{stem}\"*")
                }
            }
        })
        .collect();
    words.join(" OR ")
}

/// Chunks of one page `rank_pages` keeps among the chosen.
const PER_PAGE: usize = 2;

/// `rank` for fetched pages: each page keeps its best `PER_PAGE` chunks that share a word with
/// the question, so a page whose answer is plain prose is not ranked out by pages that repeat the
/// question's words; the slots left go by score. With more pages than `top / PER_PAGE`, each
/// page's best chunk goes first, by score, then each page's second.
pub fn rank_pages(
    chunks: &[Chunk],
    question: &str,
    top: usize,
) -> rusqlite::Result<Vec<(usize, Option<f64>)>> {
    if chunks.len() <= top {
        return rank(chunks, question, top);
    }
    let mut seen: BTreeMap<&Path, usize> = BTreeMap::new();
    let mut all = scored(chunks, &rank_query(question), usize::MAX)?;
    // A page whose every line scores about 0 (a word on all its lines, as "cookies" on a cookie
    // page or "Texas" on a Texas recap) has scores that say nothing, so its chunks keep the page's
    // order, a recap's lead first, after its longest list or table: the ranker sent intros and
    // comments.
    // Otherwise a page's longest list or table is its second pick: its intro and comments say
    // the question's words, the list asked for (ingredients, specs, prices, results) says few.
    let pages: BTreeSet<&Path> = chunks.iter().map(|c| c.file.as_path()).collect();
    for page in pages {
        let of_page = |c: &(usize, Option<f64>)| chunks[c.0].file == page;
        let best = (all.iter().filter(|c| of_page(c)))
            .filter_map(|c| c.1)
            .fold(0.0, f64::max);
        if best >= NEAR_ZERO {
            let first = all.iter().find(|c| of_page(c)).map(|c| c.0);
            if let Some(list) = longest_list(chunks, page).filter(|&l| first != Some(l)) {
                let entry = all.extract_if(.., |c| c.0 == list).next();
                let at = all.iter().position(of_page).map_or(all.len(), |p| p + 1);
                all.insert(at, entry.unwrap_or((list, None)));
            }
            continue;
        }
        let at = all.iter().position(of_page).unwrap_or(all.len());
        let mut own: Vec<_> = all.extract_if(.., |c| of_page(c)).collect();
        own.sort_by_key(|c| c.0);
        if let Some(list) = longest_list(chunks, page) {
            own.retain(|c| c.0 != list);
            own.insert(0, (list, None));
        }
        all.splice(at..at, own);
    }
    let mut by_turn: Vec<_> = (all.into_iter().enumerate())
        .map(|(at, chosen)| {
            let n = seen.entry(&chunks[chosen.0].file).or_default();
            *n += 1;
            ((*n - 1).min(PER_PAGE), at, chosen)
        })
        .collect();
    by_turn.sort_by_key(|&(turn, at, _)| (turn, at));
    by_turn.truncate(top);
    by_turn.sort_by_key(|&(_, at, _)| at);
    Ok(by_turn.into_iter().map(|(_, _, chosen)| chosen).collect())
}

/// A BM25 score below this matched only words on most of a page's lines: FTS5 gives such a word
/// an idf of 1e-6.
pub(crate) const NEAR_ZERO: f64 = 1e-3;

/// The chunk of `page` with the most list items and table rows, when one has at least three that
/// are not only a link (a share or menu list).
// ponytail: counts lines per chunk, so a list split across chunks counts as two; join them when a
// page's reader misses its list's end
pub(crate) fn longest_list(chunks: &[Chunk], page: &Path) -> Option<usize> {
    let listed = |l: &str| {
        let l = l.trim_start();
        let numbered = l
            .split_once(". ")
            .is_some_and(|(n, _)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
        let item = l.strip_prefix("- ").or_else(|| l.strip_prefix("* "));
        let link = item.is_some_and(|t| t.starts_with('[') && t.ends_with(')') && t.contains("]("));
        (item.is_some() || numbered || l.contains(" | ")) && !link
    };
    (chunks.iter().enumerate())
        .filter(|(_, c)| c.file == page)
        .map(|(i, c)| (c.text.lines().filter(|l| listed(l)).count(), i))
        .filter(|&(n, _)| n >= 3)
        .max_by_key(|&(n, i)| (n, std::cmp::Reverse(i)))
        .map(|(_, i)| i)
}

/// The best `top` of `chunks` that match the FTS5 `query`, by BM25, best first.
fn scored(
    chunks: &[Chunk],
    query: &str,
    top: usize,
) -> rusqlite::Result<Vec<(usize, Option<f64>)>> {
    if query.is_empty() {
        return Ok(Vec::new());
    }
    let conn = Connection::open_in_memory()?;
    conn.execute_batch("CREATE VIRTUAL TABLE chunks USING fts5(text, pointers UNINDEXED)")?;
    let mut insert =
        conn.prepare("INSERT INTO chunks (rowid, text, pointers) VALUES (?1, ?2, ?3)")?;
    for (i, chunk) in chunks.iter().enumerate() {
        let text = format!("{}\n{}", chunk.header(), chunk.text);
        insert.execute(params![i as i64, text, pointers(chunk)])?;
    }
    // Contents and index chunks rank after every other match, not out: one `pointers` takes for
    // them by mistake still ranks when nothing else matches. Ties go by position, so a folder
    // ranks the same way every time. FTS5's rank is BM25 negated.
    let mut best = conn.prepare(
        "SELECT rowid, rank FROM chunks WHERE chunks MATCH ?1 \
         ORDER BY pointers, rank, rowid LIMIT ?2",
    )?;
    let top = i64::try_from(top).unwrap_or(i64::MAX);
    best.query_map(params![query, top], |r| {
        Ok((r.get::<_, i64>(0)? as usize, Some(-r.get::<_, f64>(1)?)))
    })?
    .collect()
}

/// Lines of a PDF chunk that point to pages, past which `pointers` calls it a contents or index
/// page.
const POINTER_LINES: usize = 8;

/// Whether `chunk` is part of a PDF's contents or index, which names every subject of the book
/// and states none: it ranked first for a tax guide's questions and its workers found no fact
/// (Mac, 24 Nov), so it ranks after every other chunk that matches. At least `POINTER_LINES`
/// lines, and half its lines, point to pages: a contents line by dot leaders and a page ("Who Can
/// Open a Roth IRA? . . . 39"), an index line by its pages alone ("Conversion 29, 44",
/// "Contributions 39-44"), in a chunk with an index's lone letter heading ("R"), so a pin table's
/// "PA0 14" rows stay. A number with a thousands comma ("8,000") or a `$` is an amount, not a
/// page, so a limits table stays too. Only a PDF's: a note's "eggs 12" list is its content.
pub fn pointers(chunk: &Chunk) -> bool {
    if extension(&chunk.file).as_deref() != Some(".pdf") {
        return false;
    }
    let lines: Vec<&str> = (chunk.text.lines().map(str::trim))
        .filter(|l| !l.is_empty())
        .collect();
    let page = |p: &str| (1..=4).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_digit());
    // "39-44" is one reference; a money amount, "8,000" or a decimal is none.
    let refs = |r: &str| r.split('-').all(page);
    // "Conversion 29, 44" ends in "44", after a word: the line is no bare number.
    let ends_in_pages = |l: &str| {
        let (words, last) = l.rsplit_once(' ').unwrap_or(("", l));
        !words.is_empty() && !words.contains('$') && refs(last)
    };
    let contents = |l: &str| (l.contains(". . .") || l.contains("....")) && ends_in_pages(l);
    let letter = (lines.iter()).any(|l| l.len() == 1 && l.bytes().all(|b| b.is_ascii_uppercase()));
    let n = match letter {
        true => lines.iter().filter(|l| ends_in_pages(l)).count(),
        false => lines.iter().filter(|l| contents(l)).count(),
    };
    n >= POINTER_LINES && n * 2 >= lines.len()
}

/// The FTS5 query of the question's `rank_words`.
fn rank_query(question: &str) -> String {
    fts_query(&rank_words(question).join(" "))
}

/// The question's words for the ranking, stop words and question words left out, so that
/// "how much" in a question about a pet does not rank the washer manual's prices first; every
/// word when it has no other.
pub(crate) fn rank_words(question: &str) -> Vec<&str> {
    let words = question
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty());
    let kept: Vec<&str> = (words.clone())
        .filter(|w| {
            let w = w.to_lowercase();
            !STOP_WORDS.contains(&w.as_str()) && !QUESTION_WORDS.contains(&w.as_str())
        })
        .collect();
    match kept.is_empty() {
        true => words.collect(),
        false => kept,
    }
}

/// Words that ask rather than name what is asked about.
const QUESTION_WORDS: &[&str] = &[
    "can", "could", "did", "do", "does", "how", "i", "if", "me", "much", "many", "should", "what",
    "when", "where", "which", "who", "why", "would",
];

/// What the line check found: the run's JSON copy of the facts.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Facts {
    /// The facts whose line is a line of their excerpt with text, in the order the chunks were
    /// ranked: the orchestrator's fact `[n]` is `cited[n - 1]`.
    pub cited: Vec<Cited>,
    /// Facts whose line is not one of their excerpt's, or has no text.
    pub dropped: usize,
    /// Facts cited at another line than the worker named, one that holds the claim's amounts and
    /// dates (`Excerpt::cite`).
    #[serde(default)]
    pub moved: usize,
    /// Workers that gave no list of facts, after the retry too, or whose retry lost its excerpt.
    pub unparsed: usize,
    /// Pairs of `cited` indices whose facts disagree (`conflicts`), each said under the answer.
    #[serde(default)]
    pub conflicts: Vec<(usize, usize)>,
}

/// A fact that passed the line check: the worker's claim, and the line Rust copied as its quote.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cited {
    /// The fact, in the worker's words.
    pub claim: String,
    /// The line's text, every run of whitespace one space.
    pub quote: String,
    /// The file, relative to the folder asked about.
    pub file: PathBuf,
    /// The line.
    pub line: u32,
}

impl Cited {
    /// `file:line`, with `/` between folders on every system, as the answer cites it.
    pub fn source(&self) -> String {
        format!("{}:{}", slashed(&self.file), self.line)
    }
}

/// `file` with `/` between its parts.
pub(crate) fn slashed(file: &Path) -> String {
    let parts: Vec<_> = file.iter().map(|p| p.to_string_lossy()).collect();
    parts.join("/")
}

/// What a worker reading `file` shows as its helper's task: "Reading" and the file's name, or for
/// a fetched page (not of a `folder`), whose file is its URL without the scheme
/// (`web_answer::key`), its site without "www.".
pub(crate) fn reading(file: &Path, folder: bool) -> String {
    let name = match folder {
        true => file.file_name(),
        false => file.iter().next(),
    };
    let name = name.map_or_else(|| slashed(file), |n| n.to_string_lossy().into_owned());
    format!("Reading {}", name.strip_prefix("www.").unwrap_or(&name))
}

/// Words a claim or a question shares with any other, which say nothing of what it is about.
const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "each", "for", "from", "has", "have", "in",
    "is", "it", "its", "my", "of", "on", "or", "our", "per", "that", "the", "this", "to", "was",
    "will", "with", "you", "your",
];

/// A date in a claim: month (1 to 12), day, and year when it is written.
type Date = (i64, i64, Option<i64>);

/// The pairs of `cited` (indices, the first one lower) whose facts disagree: from different
/// files, their claims share three or more words beyond the stop words and numbers, their lines
/// name the same thing (`label`: a word of one line's text before its first digit is a word of
/// the other's), and both hold money amounts and no amount of one is in the other, or both hold
/// dates and no date of one is a date of the other. The label is read from the line Rust copied,
/// as a claim can repeat the question's words: a worker asked for the rent may call the insurance
/// premium's line "the monthly rent".
pub fn conflicts(cited: &[Cited]) -> Vec<(usize, usize)> {
    let parsed: Vec<_> = cited
        .iter()
        .map(|c| {
            let claim = (words(&c.claim), amounts(&c.claim), dates(&c.claim));
            (claim, label(&c.quote))
        })
        .collect();
    let mut out = Vec::new();
    for (i, ((wi, mi, di), li)) in parsed.iter().enumerate() {
        for (j, ((wj, mj, dj), lj)) in parsed.iter().enumerate().skip(i + 1) {
            let shared = wi.iter().filter(|w| wj.contains(w)).count();
            let named = li.iter().any(|w| lj.contains(w));
            let money = !mi.is_empty() && !mj.is_empty() && !mi.iter().any(|m| mj.contains(m));
            let date = !di.is_empty()
                && !dj.is_empty()
                && !di.iter().any(|a| dj.iter().any(|b| same_date(a, b)));
            if cited[i].file != cited[j].file && shared >= 3 && named && (money || date) {
                out.push((i, j));
            }
        }
    }
    out
}

/// Whether two dates are one day: the same month and day, and the same year when both say one.
fn same_date(a: &Date, b: &Date) -> bool {
    a.0 == b.0 && a.1 == b.1 && (a.2.is_none() || b.2.is_none() || a.2 == b.2)
}

/// The words of a line before its first digit, which name what its value is ("Monthly rent:
/// $1,850" names "monthly" and "rent"), without the stop words, weekdays and words that say when
/// rather than what ("next", "now").
fn label(quote: &str) -> Vec<String> {
    let before = quote
        .find(|c: char| c.is_ascii_digit())
        .unwrap_or(quote.len());
    let mut out = words(&quote[..before]);
    out.retain(|w| !WHEN_WORDS.contains(&w.as_str()) && !WEEKDAYS.contains(&w.as_str()));
    out
}

/// Words that say when a value holds, not what it is.
const WHEN_WORDS: &[&str] = &[
    "date", "day", "last", "new", "next", "now", "on", "since", "until",
];

/// The days of the week, which a question or a line can name without naming what it is about.
const WEEKDAYS: &[&str] = &[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
];

/// The claim's words, lower case, once each: no stop word and nothing with a digit.
fn words(claim: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for w in claim.split(|c: char| !c.is_alphanumeric()) {
        let w = w.to_lowercase();
        if !w.is_empty()
            && !w.chars().any(|c| c.is_ascii_digit())
            && !STOP_WORDS.contains(&w.as_str())
            && !out.contains(&w)
        {
            out.push(w);
        }
    }
    out
}

/// The money amounts in the claim, as their sign and digits: "$1,850.00" is "$1850",
/// "$1,850.50" "$1850.5", and "£1,850" "£1850", another amount. A magnitude word after it
/// scales it to digits: "$1.2 million" is "$1200000", as "$1,200,000" is.
// ponytail: amounts after $, € or £ only; read "1,850 dollars" when the audit log shows claims
// that write them so
fn amounts(claim: &str) -> Vec<String> {
    claim
        .match_indices(['$', '€', '£'])
        .filter_map(|(at, sign)| {
            let rest = &claim[at + sign.len()..];
            let n: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == ',' || *c == '.')
                .filter(|c| *c != ',')
                .collect();
            let after = rest[rest
                .find(|c: char| !(c.is_ascii_digit() || c == ',' || c == '.'))
                .unwrap_or(rest.len())..]
                .trim_start();
            // "1,850.50" and "1850.5" are one amount, and so are "1850.00" and "1850".
            // A full stop after the amount ends the sentence, not the amount.
            let n = n.trim_end_matches('.');
            let n = scaled(n, after);
            let n = if n.contains('.') {
                n.trim_end_matches('0').trim_end_matches('.').to_owned()
            } else {
                n
            };
            (!n.is_empty()).then(|| format!("{sign}{n}"))
        })
        .collect()
}

/// `digits` (an amount without separators) times the magnitude `after` starts with: thousand or k,
/// million or m, billion or bn, as digits; `digits` as it is when `after` starts with none.
fn scaled(digits: &str, after: &str) -> String {
    let word: String = after
        .chars()
        .take_while(|c| c.is_alphabetic())
        .collect::<String>()
        .to_lowercase();
    let zeros = match word.as_str() {
        "k" | "thousand" => 3,
        "m" | "mm" | "million" => 6,
        "bn" | "billion" => 9,
        _ => return digits.to_owned(),
    };
    let (whole, frac) = digits.split_once('.').unwrap_or((digits, ""));
    if frac.len() > zeros {
        return digits.to_owned();
    }
    format!("{whole}{frac}{}", "0".repeat(zeros - frac.len()))
}

/// The dates in the claim: a day before or after a month's name, with the year after them when
/// there is one ("3 November 2026", "November 3, 2026", "22 Oct").
// ponytail: no 2026-11-03 or 11/03 dates; read them when claims write dates so
fn dates(claim: &str) -> Vec<Date> {
    let tokens: Vec<String> = claim
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect();
    let day = |t: Option<&String>| {
        t.and_then(|t| day_of_month(t))
            .filter(|d| (1..=31).contains(d))
    };
    let year = |t: Option<&String>| t.and_then(|t| year_of(t));
    let mut out = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let Some(month) = month_of(t) else {
            continue;
        };
        let before = i.checked_sub(1).and_then(|b| tokens.get(b));
        if let Some(d) = day(before) {
            out.push((month, d, year(tokens.get(i + 1))));
        } else if let Some(d) = day(tokens.get(i + 1)) {
            out.push((month, d, year(tokens.get(i + 2))));
        }
    }
    out
}

/// Whether `answer` cites `source` (`file:line`) as Rust writes a citation: not as the end of a
/// longer file name, and not as the start of a longer line number.
pub(crate) fn cites(answer: &str, source: &str) -> bool {
    answer.match_indices(source).any(|(at, _)| {
        let before = answer[..at].chars().next_back();
        let after = answer[at + source.len()..].chars().next();
        !before.is_some_and(|c| c.is_alphanumeric() || "-_./".contains(c))
            && !after.is_some_and(|c| c.is_ascii_digit())
    })
}

/// The names a question gives, as it writes them, once each: its capitalized words that do not
/// start a sentence, other than "I", the months and the weekdays, and its years (1900 to 2099).
/// "What is the Wi-Fi password at the Pine Hollow cabin?" names "Wi", "Fi", "Pine" and "Hollow";
/// "What is the monthly rent for 17 Maple Avenue in 2027?" names "Maple", "Avenue" and "2027".
/// A question in lower case names nothing here; `file_words` reads what it names from the
/// folder's file names instead.
pub fn question_names(question: &str) -> Vec<String> {
    let (mut out, mut word, mut starts): (Vec<String>, String, bool) =
        (Vec::new(), String::new(), true);
    for c in question.chars().chain([' ']) {
        if c.is_alphanumeric() {
            word.push(c);
            continue;
        }
        if !word.is_empty() {
            let lower = word.to_lowercase();
            let name = !starts
                && word.starts_with(char::is_uppercase)
                && word != "I"
                && memory::dates::month_of(&lower).is_none()
                && !WEEKDAYS.contains(&lower.as_str());
            let year = word.len() == 4
                && word.bytes().all(|b| b.is_ascii_digit())
                && (word.starts_with("19") || word.starts_with("20"));
            if (name || year) && !out.contains(&word) {
                out.push(word.clone());
            }
            starts = false;
            word.clear();
        }
        if matches!(c, '.' | '?' | '!') {
            starts = true;
        }
    }
    out
}

/// Words of file names that say what kind of file it is, not what it is about.
// ponytail: a fixed list; a user folder whose names repeat a word across many files (say "2024 taxes")
// makes a question with that word withhold answers from other files; drop file words that most
// files share when a chat shows such a withheld answer
const GENERIC_FILE_WORDS: &[&str] = &[
    "checklist",
    "document",
    "documents",
    "draft",
    "file",
    "files",
    "home",
    "info",
    "letter",
    "list",
    "log",
    "manual",
    "notes",
    "plan",
    "readme",
    "record",
    "records",
    "setup",
    "status",
    "summary",
];

/// The question's words, in lower case and once each, that are a word of a file's or folder's
/// name in `files` (or that word with an "s"), past the stop words and `GENERIC_FILE_WORDS`:
/// "what's the wifi password at the cabin?" names "cabin" through cabin-weekend-plan.md, however
/// it is typed.
pub fn file_words(question: &str, files: &[&Path]) -> Vec<String> {
    let named: BTreeSet<String> = files
        .iter()
        .flat_map(|f| {
            f.with_extension("")
                .to_string_lossy()
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .filter(|w| w.len() >= 3 && !w.bytes().all(|b| b.is_ascii_digit()))
        .filter(|w| !STOP_WORDS.contains(&w.as_str()) && !GENERIC_FILE_WORDS.contains(&w.as_str()))
        .collect();
    let mut out: Vec<String> = Vec::new();
    for w in question
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
    {
        let hit = named.contains(w) || w.strip_suffix('s').is_some_and(|s| named.contains(s));
        if hit && !out.iter().any(|o| o == w) {
            out.push(w.to_owned());
        }
    }
    out
}

/// The question's names (`question_names`, then `file_words` of `files`) that none of `texts`
/// holds as a word or the start of one, case aside ("St" is held by "Street"): when the excerpts
/// an answer cites never name the cabin, its Wi-Fi password is another place's, so the answer
/// does not stand. A name with no small letter ("UT", "M3", "2027") is held only as a whole word
/// or its plural: "UTSA" is another school.
pub fn missing_names(question: &str, files: &[&Path], texts: &[String]) -> Vec<String> {
    let held: BTreeSet<String> = texts
        .iter()
        .flat_map(|t| t.split(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    let mut names = question_names(question);
    for w in file_words(question, files) {
        if !names.iter().any(|n| n.to_lowercase() == w) {
            names.push(w);
        }
    }
    names.retain(|n| {
        let whole = !n.chars().any(char::is_lowercase);
        let n = n.to_lowercase();
        match whole {
            true => !held.contains(&n) && !held.contains(&format!("{n}s")),
            false => !(held.range(n.clone()..).next()).is_some_and(|w| w.starts_with(&n)),
        }
    });
    names
}

/// `answer` cut into sentences, each with the blank after it: a sentence ends at ".", "?" or
/// "!" before a blank or the end of the text.
pub(crate) fn sentences(answer: &str) -> Vec<&str> {
    let mut ends: Vec<usize> = answer
        .char_indices()
        .filter(|&(i, c)| {
            matches!(c, '.' | '?' | '!')
                && answer[i + 1..]
                    .chars()
                    .next()
                    .is_none_or(char::is_whitespace)
        })
        .map(|(i, _)| i + 1)
        .collect();
    ends.push(answer.len());
    let mut start = 0;
    ends.into_iter()
        .map(|end| {
            let s = &answer[start..end];
            start = end;
            s
        })
        .filter(|s| !s.is_empty())
        .collect()
}

/// Stems of the words a sentence that says facts disagree uses.
const CONFLICT_WORDS: &[&str] = &["conflict", "contradict", "disagree", "inconsistent"];

/// `answer` without the sentences that say facts disagree, and how many went. Only Rust says that
/// (`conflicts`); the run calls this when it found no conflict, so such a sentence is the model's
/// mistake, as when it called a rent and an insurance premium "conflicting values for the same
/// thing".
// ponytail: the words are four stems; widen them when the audit log shows the model saying it
// another way
pub fn without_conflict_talk(answer: &str) -> (String, usize) {
    let (mut kept, mut dropped) = (String::new(), 0);
    for sentence in sentences(answer) {
        let lower = sentence.to_lowercase();
        if CONFLICT_WORDS.iter().any(|w| lower.contains(w)) {
            dropped += 1;
        } else {
            kept.push_str(sentence);
        }
    }
    (kept.trim().to_string(), dropped)
}

/// The headings open at `line` of `text` (counted from 1), outermost first, each cut as a chunk's
/// are: the section the line is in, which its chunk's header, the headings at the chunk's first
/// line, may not be.
fn section(text: &str, line: u32) -> Vec<String> {
    let mut open: Vec<(usize, &str)> = Vec::new();
    for l in text
        .lines()
        .take(usize::try_from(line).unwrap_or(usize::MAX))
    {
        if let Some((level, title)) = heading(l) {
            open.retain(|&(l, _)| l < level);
            open.push((level, title));
        }
    }
    open.into_iter().map(|(_, t)| t.to_string()).collect()
}

/// `answer` with each citation of fact numbers ("[2]", "[1, 3]") written as their sources
/// ("[lease.md:14]"), and how many numbers named no fact of `cited`: those are dropped, and a
/// citation of only such numbers goes.
// ponytail: numbers and commas only; a range "[1-3]" stays as text, read it when the audit log
// shows answers that cite so
pub fn cite(answer: &str, cited: &[Cited]) -> (String, usize) {
    citations(answer, |n| cited.get(n.checked_sub(1)?).map(Cited::source))
}

/// `answer` with each number of each citation ("[2]", "[1, 3]") as `f` writes it, and how many
/// numbers `f` dropped (`None`): a citation of only such numbers goes.
pub(crate) fn citations(
    answer: &str,
    mut f: impl FnMut(usize) -> Option<String>,
) -> (String, usize) {
    bracketed(answer, |inner| {
        let numbers = !inner.is_empty()
            && inner.chars().any(|c| c.is_ascii_digit())
            && inner
                .chars()
                .all(|c| c.is_ascii_digit() || c == ',' || c == ' ');
        if !numbers {
            return None;
        }
        let mut sources = Vec::new();
        let mut unknown = 0;
        for n in inner.split([',', ' ']).filter(|n| !n.is_empty()) {
            match n.parse::<usize>().ok().and_then(&mut f) {
                Some(s) => sources.push(s),
                None => unknown += 1,
            }
        }
        Some((sources.join(", "), unknown))
    })
}

/// `answer` with the text of each `[...]` that `f` reads as a citation replaced by what `f` gives
/// for it, and the sum of the counts `f` gives of what it dropped. `f` gets the text inside the
/// brackets and returns `None` for one that is no citation, which stays as it is, or the new text
/// and the count dropped; a citation whose new text is empty goes, with the space before it.
pub(crate) fn bracketed(
    answer: &str,
    mut f: impl FnMut(&str) -> Option<(String, usize)>,
) -> (String, usize) {
    let (mut out, mut rest, mut unknown) = (String::new(), answer, 0);
    while let Some(open) = rest.find('[') {
        let Some(close) = rest[open..].find(']').map(|c| open + c) else {
            break;
        };
        let Some((shown, dropped)) = f(&rest[open + 1..close]) else {
            out.push_str(&rest[..=open]);
            rest = &rest[open + 1..];
            continue;
        };
        unknown += dropped;
        let before = &rest[..open];
        if shown.is_empty() {
            // The space before a citation that goes goes with it.
            out.push_str(before.strip_suffix(' ').unwrap_or(before));
        } else {
            out.push_str(before);
            out.push_str(&format!("[{shown}]"));
        }
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    (out, unknown)
}

/// The ask-docs skill: gather, rank, workers, the line check, then the answer.
#[derive(Clone, Copy, Debug)]
pub struct AskDocs {
    /// The model's tool-call family, which says whether its requests get a thinking field.
    pub family: Option<Family>,
    /// The answer's thinking budget, as a chat step's; the workers never think.
    pub thinking: Thinking,
    /// Chunks the workers read, one each (config.toml's `docs_workers`).
    pub docs_workers: usize,
    /// Tokens of the chunk's file a worker reads on each side of its chunk (config.toml's
    /// `docs_window`); 0 is the chunk alone.
    pub docs_window: u32,
    /// The workers sample under `facts_grammar` (config.toml's `tool_grammar`, the switch the
    /// calls' grammar has).
    pub facts_grammar: bool,
}

impl Default for AskDocs {
    fn default() -> Self {
        Self {
            family: None,
            thinking: Thinking::default(),
            docs_workers: DOCS_WORKERS,
            docs_window: 0,
            facts_grammar: true,
        }
    }
}

impl Skill for AskDocs {
    fn spec() -> SkillSpec {
        SkillSpec {
            name: NAME.into(),
            description:
                "Answers a question from the .pdf, .md and .txt files in a folder, citing \
                          file:line for each fact."
                    .into(),
            tools: Vec::new(),
        }
    }

    fn run(
        &self,
        larch: &Larch,
        ctx: &ToolCtx,
        workers: Workers,
        input: &SkillInput,
    ) -> Result<SkillRun, SkillError> {
        self.run_until(larch, ctx, workers, input, false, &AtomicBool::default())
    }
}

impl AskDocs {
    /// `run`, which `stop` ends with `EngineError::Cancelled`: no worker group starts once it is
    /// set, and every stream of the workers and of the answer is dropped before its next token.
    pub(crate) fn run_until(
        &self,
        larch: &Larch,
        ctx: &ToolCtx,
        workers: Workers,
        input: &SkillInput,
        all: bool,
        stop: &AtomicBool,
    ) -> Result<SkillRun, SkillError> {
        let mut log = Log::new(ctx, NAME);
        let gathered = log.step(
            "gather",
            || gather_in(ctx, (!all).then_some(input.folder.as_str()), stop),
            |g| {
                json!({"files": g.files, "chunks": g.chunks.len(), "skipped": g.skipped,
                       "unlisted": g.unlisted.len(), "head": head(g, &input.folder)})
            },
        )?;
        let request = |f: &Facts| answer_request(self.family, &input.question, f, &gathered.texts);
        let source = Source {
            gathered: &gathered,
            question: &input.question,
            extract: &input.question,
            today: "",
            request: &request,
            not_found: NOT_FOUND,
            documents: true,
        };
        let Answered {
            answer: answered,
            mut notes,
            facts,
            copy,
            chosen,
        } = {
            let ranked_by = match input.search.trim() {
                "" => input.question.clone(),
                search => format!("{}\n{search}", input.question),
            };
            self.answer_ranked(larch, ctx, workers, &source, &ranked_by, &mut log, stop)?
        };
        if gathered.files == 0 {
            notes.push(format!(
                "_No .pdf, .md or .txt file was read from {}._",
                input.folder
            ));
        } else if gathered.chunks.is_empty() {
            let (s, hold) = plural(gathered.files, "holds", "hold");
            notes.push(format!(
                "_The {} file{s} read {hold} no text._",
                gathered.files
            ));
        } else if chosen.is_empty() {
            notes.push(format!(
                "_None of the {} excerpts shares a word with the question, so none was read._",
                gathered.chunks.len()
            ));
        } else if answered == NOT_FOUND && chosen.len() < gathered.chunks.len() {
            // "Do not answer" is the parts' that were read: the chat may ask again with other
            // words (`run::NOT_IN_PARTS`), and the person knows the rest was not read.
            notes.push(format!(
                "_Read {} of {}{PARTS_READ}",
                chosen.len(),
                gathered.chunks.len()
            ));
        }
        notes.extend(unread(&gathered));
        let markdown = [answered.clone()]
            .into_iter()
            .chain(notes)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        let markdown = with_pages(&markdown, &facts, &gathered);
        let line = format!(
            "{NAME} on {}: \"{}\": {} facts cited, {} dropped, {} workers without a list",
            input.folder,
            input.question,
            facts.cited.len(),
            facts.dropped,
            facts.unparsed
        );
        memory::note(&ctx.data_dir, &line).map_err(SkillError::Log)?;
        Ok(SkillRun {
            markdown,
            answer: answered,
            facts: copy,
            steps: log.events,
        })
    }

    /// The pipeline both sources share: rank `source`'s chunks, one worker per chunk, the line
    /// check, then the writer's turn and the checks on its answer. The notes it returns end with
    /// the excerpts that gave no list; the source adds its own after them.
    pub(crate) fn answer(
        &self,
        larch: &Larch,
        ctx: &ToolCtx,
        workers: Workers,
        source: &Source,
        log: &mut Log,
        stop: &AtomicBool,
    ) -> Result<Answered, SkillError> {
        self.answer_ranked(larch, ctx, workers, source, source.extract, log, stop)
    }

    /// `answer`, whose chunks are ranked by `ranked_by`'s words rather than `source.extract`'s:
    /// the question with the words the model said the documents may use (`SkillInput::search`).
    #[allow(clippy::too_many_arguments)]
    fn answer_ranked(
        &self,
        larch: &Larch,
        ctx: &ToolCtx,
        workers: Workers,
        source: &Source,
        ranked_by: &str,
        log: &mut Log,
        stop: &AtomicBool,
    ) -> Result<Answered, SkillError> {
        let gathered = source.gathered;
        let Extracted {
            facts,
            copy,
            excerpts,
            origin,
            chosen,
        } = self.extract(
            larch,
            ctx,
            workers,
            gathered,
            source.extract,
            ranked_by,
            source.today,
            log,
            stop,
        )?;
        let namespace = crate::namespace(&ctx.data_dir);

        // Said under the answer, so a cut reply or a failed worker does not pass for an answer.
        let mut notes = Vec::new();
        // No fact passed: Rust abstains, and no model turn can talk its way past that.
        let answered = if facts.cited.is_empty() {
            source.not_found.to_string()
        } else {
            let start = Instant::now();
            // ponytail: every fact that passed goes to the orchestrator; cap the list by tokens
            // when a run fails with the prompt too long for the context
            let request = (source.request)(&facts);
            let write = |thinking: Thinking, log: &mut Log| {
                let mut session = larch.session(SessionOptions {
                    system: Some(request.system.clone()),
                    // One turn, then dropped, which frees any park: nothing could reuse its KV.
                    keep_kv: false,
                    ..SessionOptions::default()
                });
                let id = session.id();
                let p = answer_params(self.family, thinking, namespace);
                let reply = SeatAdmission::of(larch)
                    .seat(p.class, stop, || session.send(&request.prompt, p));
                reply.and_then(|s| answer(id, s, stop)).or_else(|e| {
                    log.write(failed("answer", id, start, &e))?;
                    Err(SkillError::Engine(e))
                })
            };
            let mut thinking = self.thinking;
            let mut a = write(thinking, log)?;
            // A think block cut at its budget, then "not found" with no citation, though facts
            // passed: the thinking stopped before it weighed them, so the writer answers again
            // without it.
            let text = a.text.trim();
            if a.timings.thinking_cut
                && !text.contains('[')
                && text.starts_with(source.not_found.trim_end_matches('.'))
            {
                log.write(AuditEvent {
                    wall_ms: Some(ms(start)),
                    detail: Some(json!({"thinking_tokens": thought(larch, &a)})),
                    ..turn("answer", &a, Some(RETRIED.into()))
                })?;
                thinking.budget = 0;
                a = write(thinking, log)?;
            }
            let max_tokens = answer_params(self.family, thinking, namespace).max_tokens;
            let limit = format!("the limit of {max_tokens} tokens");
            let (text, unknown) = cite(a.text.trim(), &request.facts);
            let (text, unfounded) = match facts.conflicts.is_empty() && source.documents {
                true => without_conflict_talk(&text),
                false => (text, 0),
            };
            // The names the question gives that no excerpt the answer cites holds: an answer
            // from facts about something else, which does not stand. An answer that cites
            // nothing, as the one that says the documents do not answer, is left as it is. A web
            // page names its subject once, in its title or byline, so a cited page's head holds
            // names too, but not a year: a page's dates and footer are not the answer's. Not the
            // whole page, whose sidebars name other places; a file's other chunks may be about
            // another home.
            let cited: Vec<_> = (facts.cited.iter().zip(&origin))
                .filter(|(c, _)| cites(&text, &c.source()))
                .collect();
            let held: Vec<String> = cited.iter().map(|&(_, &k)| excerpts[k].render()).collect();
            let heads: Vec<String> = (cited.iter())
                .filter_map(|(c, _)| gathered.heads.get(&c.file).cloned())
                .collect();
            let missing = match held.is_empty() {
                true => Vec::new(),
                false => {
                    let files: Vec<&Path> = match source.documents {
                        true => gathered.texts.keys().map(PathBuf::as_path).collect(),
                        false => Vec::new(),
                    };
                    let not_in_heads = missing_names(source.question, &files, &heads);
                    (missing_names(source.question, &files, &held).into_iter())
                        .filter(|n| {
                            n.bytes().all(|b| b.is_ascii_digit()) || not_in_heads.contains(n)
                        })
                        .collect()
                }
            };
            let error = if a.finish == Finish::Length {
                notes.push(format!("_The answer stops at {limit}._"));
                Some(format!("cut at {limit}"))
            } else if text.trim().is_empty() {
                notes.push("_The model gave an empty answer._".into());
                Some("empty answer".into())
            } else {
                None
            };
            if unknown > 0 {
                let (s, were) = plural(unknown, "was", "were");
                notes.push(format!(
                    "_{unknown} citation{s} named no fact and {were} dropped._"
                ));
            }
            if unfounded > 0 {
                let (s, was) = plural(unfounded, "was", "were");
                notes.push(format!(
                    "_{unfounded} sentence{s} that said the facts disagree {was} dropped: the \
                     facts' values are for different things._"
                ));
            }
            log.write(AuditEvent {
                wall_ms: Some(ms(start)),
                detail: Some(json!({
                    "unknown_citations": unknown,
                    "conflict_sentences_dropped": unfounded,
                    "names_not_cited": missing,
                    "thinking_tokens": thought(larch, &a),
                })),
                ..turn("answer", &a, error)
            })?;
            if missing.is_empty() {
                text.trim().to_string()
            } else if source.documents {
                // "Do not answer" is the facts', not the documents': say when the files do name
                // them, so the person knows to ask after that part.
                let in_files: Vec<&str> = (missing.iter())
                    .filter(|n| {
                        (gathered.texts.values())
                            .flat_map(|t| t.split(|c: char| !c.is_alphanumeric()))
                            .any(|w| w.eq_ignore_ascii_case(n))
                    })
                    .map(String::as_str)
                    .collect();
                notes.push(format!(
                    "_{WITHHELD}: no fact it cited names {}, which the question names.{}_",
                    missing.join(", "),
                    match in_files.is_empty() {
                        true => String::new(),
                        false => format!(
                            " The documents do name {}, in parts the facts did not come from: \
                             ask about that part.",
                            in_files.join(", ")
                        ),
                    }
                ));
                source.not_found.to_string()
            } else {
                // A page spells a name its own way ("Texas" for "UT", "Federal Reserve" for "the
                // Fed"), so the model judges whether the cited facts are about the names; a year
                // has no other spelling, and only gets the note.
                let (mut unnamed, words): (Vec<String>, Vec<String>) =
                    (missing.into_iter()).partition(|n| n.bytes().all(|b| b.is_ascii_digit()));
                let judged = match words.is_empty() {
                    true => None,
                    false => {
                        let listed: String = (cited.iter().enumerate())
                            .map(|(i, (c, _))| {
                                let page = (gathered.heads.get(&c.file))
                                    .and_then(|h| h.lines().find(|l| !l.trim().is_empty()))
                                    .unwrap_or_default();
                                format!("[{}] {} (\"{}\"; page: {page})\n", i + 1, c.claim, c.quote)
                            })
                            .collect();
                        let prompt = format!(
                            "Question: {}\nNames the pages never write: {}\n\nFacts:\n{listed}",
                            source.question,
                            words.join(", ")
                        );
                        subject(larch, self.family, namespace, &prompt, log, stop)?
                    }
                };
                let names = words.join(", ");
                let stands = match judged {
                    Some(Subject {
                        about,
                        verdict: Verdict::Different,
                    }) => {
                        notes.push(format!(
                            "_{WITHHELD}: its pages are about {about}, not {names}._"
                        ));
                        false
                    }
                    Some(Subject {
                        about,
                        verdict: Verdict::Same,
                    }) => {
                        notes.push(format!(
                            "_The pages never say {names}; they say {about}, taken as the same \
                             thing._"
                        ));
                        true
                    }
                    Some(Subject {
                        about,
                        verdict: Verdict::Unclear,
                    }) => {
                        notes.push(format!(
                            "_I took {names} to mean {about}. If you meant something else, say \
                             which._"
                        ));
                        true
                    }
                    None => {
                        unnamed.splice(0..0, words);
                        true
                    }
                };
                if stands && !unnamed.is_empty() {
                    notes.push(format!(
                        "_{UNNAMED} {}, which the question names: check that they are about the \
                         same thing._",
                        unnamed.join(", ")
                    ));
                }
                match stands {
                    true => text.trim().to_string(),
                    false => source.not_found.to_string(),
                }
            }
        };
        for &(i, j) in &facts.conflicts {
            let (a, b) = (&facts.cited[i], &facts.cited[j]);
            notes.push(format!(
                "_Conflict: \"{}\" [{}] against \"{}\" [{}]._",
                a.quote,
                a.source(),
                b.quote,
                b.source()
            ));
        }
        if facts.unparsed > 0 {
            // The reply, or a session that dropped the excerpt to fit the context: the log's
            // turns say which.
            notes.push(format!(
                "_{NO_LIST} for {} of {} excerpts, so the answer may miss what they say; the \
                 audit log says why._",
                facts.unparsed,
                excerpts.len()
            ));
        }
        Ok(Answered {
            answer: answered,
            notes,
            facts,
            copy,
            chosen,
        })
    }

    /// The first half of `answer`, which research runs alone each round: rank `gathered`'s chunks
    /// by `ranked_by`'s words, one worker per chunk on `question`, told `today` when the chunks
    /// are pages, then the line check. Each worker is a helper the chat shows, as research's are:
    /// without them a chat's readers ran unseen behind a still tree.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn extract(
        &self,
        larch: &Larch,
        ctx: &ToolCtx,
        workers: Workers,
        gathered: &Gathered,
        question: &str,
        ranked_by: &str,
        today: &str,
        log: &mut Log,
        stop: &AtomicBool,
    ) -> Result<Extracted, SkillError> {
        self.extract_until(
            larch, ctx, workers, gathered, question, ranked_by, today, log, stop, stop, true,
        )
    }

    /// `extract`, whose workers `cut` ends too: a cut that is not `stop` (research's wall time)
    /// keeps the facts of the workers done by then. With `helpers`, each worker is a helper the
    /// chat shows (`Workers::run_helpers`), named by what it reads (`reading`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn extract_until(
        &self,
        larch: &Larch,
        ctx: &ToolCtx,
        workers: Workers,
        gathered: &Gathered,
        question: &str,
        ranked_by: &str,
        today: &str,
        log: &mut Log,
        stop: &AtomicBool,
        cut: &AtomicBool,
        helpers: bool,
    ) -> Result<Extracted, SkillError> {
        // The chunks the workers read, so a run that found nothing shows what it looked at.
        let place = |c: &Chunk| json!({"file": slashed(&c.file), "line": c.line});
        let chosen = log.step(
            "rank",
            || match gathered.folder {
                true => rank(&gathered.chunks, ranked_by, self.docs_workers)
                    .map(|c| cover_names(&gathered.chunks, c, ranked_by, self.docs_workers)),
                false => rank_pages(&gathered.chunks, ranked_by, self.docs_workers),
            },
            |chosen| {
                let chosen: Vec<_> = chosen
                    .iter()
                    .map(|&(i, score)| {
                        let mut entry = place(&gathered.chunks[i]);
                        entry["score"] = json!(score);
                        entry
                    })
                    .collect();
                json!({ "chosen": chosen, "ranked_by": ranked_by })
            },
        )?;
        let chunks: Vec<_> = chosen.iter().map(|&(i, _)| &gathered.chunks[i]).collect();
        let excerpts: Vec<_> = (chunks.iter())
            .map(|c| excerpt(gathered, c, self.docs_window))
            .collect();
        // The question in the system prompt, so every worker of the run shares its KV. A page's
        // reader gets the day too, for its "latest" and "last" (a user's football score, 27 Oct).
        let system = match gathered.folder {
            true => format!("{WORKER_SYSTEM}\n\nQuestion: {question}"),
            false => format!("{WORKER_SYSTEM}\n\nToday is {today}.\nQuestion: {question}"),
        };
        let prompts: Vec<_> = excerpts.iter().map(Excerpt::render).collect();
        let namespace = crate::namespace(&ctx.data_dir);
        let p = GenParams {
            grammar: self.facts_grammar.then(facts_grammar),
            ..params(self.family, namespace)
        };
        let start = Instant::now();
        let check = |a: &Answer| parse_facts(&a.text, a.finish).err().map(|e| e.to_string());
        let tasks: Vec<String> = match helpers {
            true => (chunks.iter())
                .map(|c| reading(&c.file, gathered.folder))
                .collect(),
            false => Vec::new(),
        };
        let replies =
            workers.run_helpers(larch, &system, &prompts, &tasks, &p, check, |_, _| {}, cut);
        let workers_ms = ms(start);

        let start = Instant::now();
        let (mut facts, mut turns) = (Facts::default(), Vec::new());
        // A debugging switch: the documents' text stays out of the log unless it is asked for.
        let log_replies = std::env::var_os(LOG_REPLIES).is_some_and(|v| v == "1");
        let (mut retried, mut failed) = (0, None);
        // The excerpt each fact of `facts.cited` came from.
        let mut origin = Vec::new();
        for (k, ((chunk, ex), (answers, error))) in
            chunks.iter().zip(&excerpts).zip(replies).enumerate()
        {
            retried += answers.len().saturating_sub(1);
            // The last reply counts: the first when it parsed, else the retry.
            let mut list = None;
            for (i, a) in answers.iter().enumerate() {
                // A retry the session fit into the context by dropping the excerpt and the first
                // reply was about nothing; its "[]" would pass for an excerpt that says nothing.
                // ponytail: such a retry still decodes up to REPLY_TOKENS that go unread; fail it
                // up front (Overflow::Fail) once the audit log shows retries that lost the excerpt
                let parsed = if a.timings.dropped_exchanges > 0 {
                    Err("the session dropped the excerpt to fit the context".to_string())
                } else {
                    parse_facts(&a.text, a.finish).map_err(|e| e.to_string())
                };
                let step = if i == 0 { "worker" } else { "retry" };
                let mut reply = place(chunk);
                // The lines the worker read, its chunk's and its window's.
                let last = ex.first as usize + ex.lines.len().saturating_sub(1);
                reply["lines"] = json!([ex.first, last]);
                reply["reply_bytes"] = json!(a.text.len());
                reply["reply_sha256"] =
                    json!(larch::bench::hex(&larch::bench::sha256(a.text.as_bytes())));
                if log_replies {
                    reply["reply"] = json!(a.text);
                }
                turns.push(AuditEvent {
                    detail: Some(reply),
                    ..turn(step, a, parsed.as_ref().err().cloned())
                });
                list = parsed.ok();
            }
            // The turns that came are logged before the run fails.
            if let Some(e) = error {
                failed.get_or_insert(e);
                continue;
            }
            let Some(list) = list else {
                facts.unparsed += 1;
                continue;
            };
            // ponytail: with docs_window over 0, two chosen chunks' windows can show one line, and
            // both workers' facts on it go to the answer; drop repeats by file:line when the Mac
            // run at --docs-window 128 shows them
            for fact in list {
                match ex.cite(&fact) {
                    Some(c) => {
                        facts.moved += usize::from(u64::from(c.line) != fact.line);
                        facts.cited.push(c);
                        origin.push(k);
                    }
                    None => facts.dropped += 1,
                }
            }
        }
        facts.conflicts = conflicts(&facts.cited);
        let verify_ms = ms(start);
        for t in turns {
            log.write(t)?;
        }
        log.write(AuditEvent {
            error: failed.as_ref().map(|e| e.to_string()),
            ..step(
                "workers",
                workers_ms,
                Some(
                    json!({"workers": prompts.len(), "at_once": workers.at_once(),
                            "retried": retried}),
                ),
            )
        })?;
        // Ignored only when the cut came and the user did not stop: an engine's own cancel
        // stays an error.
        let timed_out = |e: &EngineError| {
            *e == EngineError::Cancelled
                && cut.load(Ordering::Relaxed)
                && !stop.load(Ordering::Relaxed)
        };
        if let Some(e) = failed.filter(|e| !timed_out(e)) {
            return Err(SkillError::Engine(e));
        }
        let copy = serde_json::to_value(&facts).map_err(|e| SkillError::Log(e.into()))?;
        log.write(step("verify", verify_ms, Some(copy.clone())))?;
        Ok(Extracted {
            facts,
            copy,
            excerpts,
            origin,
            chosen,
        })
    }
}

/// Where `AskDocs::answer` takes its chunks from and how its writer is asked: a folder's
/// documents (ask-docs) or fetched pages (web-answer).
pub(crate) struct Source<'a> {
    /// The chunks, and the texts their windows read.
    pub gathered: &'a Gathered,
    /// The user's question, whose names the cited excerpts must hold.
    pub question: &'a str,
    /// What the readers extract facts for: `question`, with the days a relative date names.
    pub extract: &'a str,
    /// The day a page's readers are told, `YYYY-MM-DD`; a folder's are told none.
    pub today: &'a str,
    /// The writer's request for the facts that passed the line check.
    pub request: &'a dyn Fn(&Facts) -> AnswerRequest,
    /// The answer when no fact passed, and when an answer is withheld.
    pub not_found: &'a str,
    /// The chunks are the user's documents, so the answer checks tuned on them run: a name the
    /// question shares with a file's name must be in what the answer cites, and a sentence that
    /// says facts disagree goes when Rust found none. Web pages' URLs name more than their pages
    /// do, and a news answer says "conflict" of a war.
    pub documents: bool,
}

/// What `AskDocs::extract` gives back.
pub(crate) struct Extracted {
    /// What the line check found.
    pub facts: Facts,
    /// `facts` as the audit log has it.
    pub copy: Value,
    /// What each worker read, in rank order.
    pub excerpts: Vec<Excerpt>,
    /// The excerpt each fact of `facts.cited` came from.
    pub origin: Vec<usize>,
    /// The chunks the workers read, best first.
    pub chosen: Vec<(usize, Option<f64>)>,
}

/// What `AskDocs::answer` gives back.
pub(crate) struct Answered {
    /// The writer's answer, its citations as sources, or `Source::not_found`.
    pub answer: String,
    /// Said under the answer.
    pub notes: Vec<String>,
    /// What the line check found.
    pub facts: Facts,
    /// `facts` as the audit log has it.
    pub copy: Value,
    /// The chunks the workers read, best first.
    pub chosen: Vec<(usize, Option<f64>)>,
}

/// The run's audit events, each written to the log when its step ends and kept for the caller.
/// The workers' turns come back together from `Workers::run`, so they are written with the
/// workers step and carry its end time; a run killed during that step leaves none of them.
pub(crate) struct Log {
    audit: Audit,
    pub(crate) events: Vec<AuditEvent>,
    /// The skill's name on each event.
    name: &'static str,
    /// Where the skill says what it is doing, for the chat to show (`AgentEvent::Progress`).
    pub(crate) progress: Option<mpsc::Sender<String>>,
}

impl Log {
    /// An empty log of the skill `name`, in `ctx`'s data folder.
    pub(crate) fn new(ctx: &ToolCtx, name: &'static str) -> Log {
        Log {
            audit: Audit::new(&ctx.data_dir),
            events: Vec::new(),
            name,
            progress: None,
        }
    }

    /// Says `text` on `progress`, if there is one and its chat still listens.
    pub(crate) fn progress(&self, text: &str) {
        if let Some(p) = &self.progress {
            let _ = p.send(text.into());
        }
    }

    pub(crate) fn write(&mut self, event: AuditEvent) -> Result<(), SkillError> {
        let event = AuditEvent {
            skill: Some(self.name.into()),
            ..event
        };
        self.audit.append(event.clone()).map_err(SkillError::Log)?;
        self.events.push(event);
        Ok(())
    }

    /// Runs `f` as the step `name`, logging its wall time and `detail` of what it returned, or
    /// its error.
    pub(crate) fn step<T, E: fmt::Display + Into<SkillError>>(
        &mut self,
        name: &str,
        f: impl FnOnce() -> Result<T, E>,
        detail: impl FnOnce(&T) -> Value,
    ) -> Result<T, SkillError> {
        let start = Instant::now();
        let done = f();
        let wall_ms = ms(start);
        self.write(match &done {
            Ok(t) => step(name, wall_ms, Some(detail(t))),
            Err(e) => AuditEvent {
                error: Some(e.to_string()),
                ..step(name, wall_ms, None)
            },
        })?;
        done.map_err(Into::into)
    }
}

/// The notes that say what `gather` left out: the files it counted but did not read, the folders
/// it could not list, and the rest of a walk it stopped.
pub(crate) fn unread(gathered: &Gathered) -> Vec<String> {
    let mut notes = Vec::new();
    let skipped = gathered.skipped;
    if !gathered.oversized.is_empty() {
        let names: Vec<_> = gathered.oversized.iter().map(|f| slashed(f)).collect();
        let (s, were) = plural(names.len(), "was", "were");
        notes.push(format!(
            "_{} {were} over {} MB, which is more than a text file is read at once, so {} not \
             read. Split {} into smaller files and attach those._",
            names.join(", "),
            MAX_FILE_BYTES >> 20,
            if names.len() == 1 {
                "it was"
            } else {
                "they were"
            },
            if s.is_empty() { "it" } else { "them" },
        ));
    }
    if skipped > 0 {
        let (s, were) = plural(skipped, "was", "were");
        notes.push(format!(
            "_{skipped} file{s} {were} not read: past the first {MAX_FILES}, past the \
             {MAX_PDFS} PDFs one question reads anew (ask again for the next), stopped, over {} \
             MB, unreadable, or a link to a file outside the allowed folders, hidden, of another \
             type or in the agent's data folder._",
            MAX_FILE_BYTES >> 20
        ));
    }
    if !gathered.unlisted.is_empty() {
        let names: Vec<_> = gathered.unlisted.iter().map(|f| slashed(f)).collect();
        let these = if names.len() == 1 {
            "This folder"
        } else {
            "These folders"
        };
        notes.push(format!(
            "_{these} could not be listed: {}._",
            names.join(", ")
        ));
    }
    if gathered.stopped {
        notes.push(format!(
            "_Stopped after {MAX_ENTRIES} files and folders; name a smaller folder to search the \
             rest._"
        ));
    }
    notes
}

/// The noun's `s` and the verb for `n` things.
pub(crate) fn plural(
    n: usize,
    one: &'static str,
    more: &'static str,
) -> (&'static str, &'static str) {
    if n == 1 { ("", one) } else { ("s", more) }
}

/// The parameters of the workers' requests and their retries: a worker step's, which does not
/// think (a list of facts needs no think block before it), greedy and seeded, as every bench
/// request is, so a run and its bench scores repeat, in the data folder's cache `namespace`.
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
        max_tokens: REPLY_TOKENS,
        seed: Some(0),
        ..worker
    }
}

/// The answer's parameters: an answer step's, which thinks on `thinking`'s budget (the abstention
/// is Rust's, so thinking only weighs the facts), greedy and seeded, with `REPLY_TOKENS` after
/// the think block, or `answer_max_tokens` when that sets a cap.
pub(crate) fn answer_params(
    family: Option<Family>,
    thinking: Thinking,
    namespace: u64,
) -> GenParams {
    let p = run::params(
        &Registry::builtin(),
        &[],
        family,
        Kind::Answer,
        thinking,
        namespace,
    );
    let reply = match thinking.answer_max_tokens {
        0 => REPLY_TOKENS,
        cap => cap,
    };
    GenParams {
        max_tokens: p.thinking_budget.unwrap_or(0).saturating_add(reply),
        seed: Some(0),
        ..p
    }
}

/// The subject check (`SUBJECT_SYSTEM`) on `prompt`: one turn with thinking off, held to its
/// reply's schema; `None` when the turn fails, which the log says.
fn subject(
    larch: &Larch,
    family: Option<Family>,
    namespace: u64,
    prompt: &str,
    log: &mut Log,
    stop: &AtomicBool,
) -> Result<Option<Subject>, SkillError> {
    let start = Instant::now();
    let mut session = larch.session(SessionOptions {
        system: Some(SUBJECT_SYSTEM.into()),
        keep_kv: false,
        ..SessionOptions::default()
    });
    let id = session.id();
    let schema = json!({
        "type": "object",
        "properties": {
            "about": {"type": "string"},
            "verdict": {"enum": ["same", "different", "unclear"]},
        },
        "required": ["about", "verdict"],
    });
    let pattern = json_schema_regex(&schema).expect("the subject schema is ours");
    let thinking = Thinking {
        budget: 0,
        answer_max_tokens: SUBJECT_TOKENS,
    };
    let p = GenParams {
        grammar: Some(Grammar {
            trigger: None,
            pattern,
        }),
        ..answer_params(family, thinking, namespace)
    };
    let sent = SeatAdmission::of(larch).seat(p.class, stop, || session.send(prompt, p));
    let a = match sent.and_then(|s| answer(id, s, stop)) {
        Ok(a) => a,
        Err(e) => {
            log.write(failed("subject", id, start, &e))?;
            // The user's stop stays a stop; any other failure leaves the answer with its note.
            return match e {
                EngineError::Cancelled => Err(SkillError::Engine(e)),
                _ => Ok(None),
            };
        }
    };
    let parsed = serde_json::from_str::<Subject>(a.text.trim()).map_err(|e| e.to_string());
    log.write(AuditEvent {
        wall_ms: Some(ms(start)),
        detail: Some(json!({ "reply": a.text })),
        ..turn("subject", &a, parsed.as_ref().err().cloned())
    })?;
    Ok(parsed.ok())
}

/// A Rust step's event.
pub(crate) fn step(name: &str, wall_ms: f64, detail: Option<Value>) -> AuditEvent {
    AuditEvent {
        kind: EventKind::Step,
        step: Some(name.into()),
        wall_ms: Some(wall_ms),
        detail,
        ..AuditEvent::default()
    }
}

/// A model turn's event, with the engine's counts and timings.
pub(crate) fn turn(step: &str, a: &Answer, error: Option<String>) -> AuditEvent {
    AuditEvent {
        session: a.session,
        kind: EventKind::Turn,
        step: Some(step.into()),
        error,
        prompt_tokens: Some(a.usage.prompt_tokens),
        cached_tokens: Some(a.usage.cached_tokens),
        generated_tokens: Some(a.usage.completion_tokens),
        ttft_ms: Some(a.timings.ttft_ms),
        tok_s: Some(a.timings.tok_s),
        ..AuditEvent::default()
    }
}

/// A model turn's event when it failed to run: its session and the engine's error, no counts.
pub(crate) fn failed(step: &str, session: u64, start: Instant, e: &EngineError) -> AuditEvent {
    AuditEvent {
        session,
        kind: EventKind::Turn,
        step: Some(step.into()),
        wall_ms: Some(ms(start)),
        error: Some(e.to_string()),
        ..AuditEvent::default()
    }
}

/// The tokens of `a`'s hidden think block, which its audit line's detail counts.
pub(crate) fn thought(larch: &Larch, a: &Answer) -> usize {
    (a.thinking.as_deref()).map_or(0, |t| larch.token_ids(t).len())
}

/// Milliseconds since `start`.
pub(crate) fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

/// The skill's name as a chat tool.
pub const TOOL: &str = "ask_docs";

/// The tool's description: a question about what the documents say goes here, a file the user
/// names to read_file.
const TOOL_DESCRIPTION: &str = "Answers a question about what the user's documents say, from the \
.pdf, .md and .txt files in a folder, citing file:line for each fact. Use it for a question about the \
content of the user's files; to open a file the user names, use read_file.";

/// The `question` argument's description. The user's words, as they gave them: a model told
/// "this year" asked about "2026" of a guide for 2025, which then "did not answer" (Mac, 24 Nov).
const QUESTION: &str = "The user's question, word for word";

/// The `search` argument's description: the ranker reads the documents by their words, which a
/// question in the user's words may not share. "Income limits for opening a Roth IRA" ranked a
/// tax guide's index pages first; its answer says "modified AGI" and "phased out" (Mac, 24 Nov).
const SEARCH: &str = "Optional: terms the documents may use";

/// ask-docs as a chat tool: the skill on the chat's engine, inside the chat's allowed folders.
pub(crate) struct DocsTool {
    /// The engine and the skill's settings; `None` in `Registry::builtin`, whose specs the step
    /// parameters and the bench read, and which runs no model.
    pub(crate) engine: Option<(Larch, AskDocs, Workers)>,
    /// Ends the run when set (`AskDocs::run_until`): a chat sets it when its turn is dropped.
    pub(crate) stop: Arc<AtomicBool>,
}

impl crate::tool::Tool for DocsTool {
    fn spec(&self) -> crate::tool::ToolSpec {
        crate::tool::ToolSpec {
            name: TOOL.into(),
            description: TOOL_DESCRIPTION.into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": QUESTION},
                    "search": {"type": "string", "description": SEARCH},
                    "folder": {"type": "string",
                               "description": "Folder or file of documents; the first allowed one if left out"}
                },
                "required": ["question"],
                "additionalProperties": false
            }),
            permission: crate::tool::Permission::Read,
            untrusted_results: false,
            // Never cut: the answer is the reply as it stands, its citations and notes whole,
            // and REPLY_TOKENS and the citations' paths bound it.
            max_result_tokens: u32::MAX,
            max_args_tokens: crate::tool::ARGS_TOKENS,
        }
    }

    /// The skill's answer in Markdown: the cited answer or `NOT_FOUND`, and the notes under it.
    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        let Some((larch, skill, workers)) = &self.engine else {
            return Err(ToolError::Invalid(format!(
                "{TOOL} runs only in a chat, which has the model's engine"
            )));
        };
        let question = crate::tool::str_arg(args, "question")?;
        let search = args["search"].as_str().unwrap_or_default();
        // A folder left out is every one attached, each read as when named.
        let all = args["folder"].is_null() && ctx.allowed.len() > 1;
        let shown = match all {
            true => (ctx.allowed.iter())
                .map(|a| a.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
            // The folder as the run's notes and the daily note name it: its real path.
            false => {
                let first = first(ctx);
                let folder = args["folder"].as_str().unwrap_or(&first);
                let real = crate::tool::plain(ctx.resolve(folder, false)?);
                real.to_str().unwrap_or(folder).into()
            }
        };
        let input = SkillInput {
            folder: shown,
            question: question.into(),
            search: search.into(),
        };
        match skill.run_until(larch, ctx, *workers, &input, all, &self.stop) {
            // The folder too, so "which folder did you read?" has an answer (a user's chat named
            // another one).
            Ok(run) => {
                let gathered = run
                    .steps
                    .iter()
                    .find(|e| e.step.as_deref() == Some("gather"));
                let detail = gathered.and_then(|e| e.detail.as_ref());
                let head = detail.and_then(|d| d["head"].as_str()).unwrap_or_default();
                let body = if head.starts_with(NO_TEXT) {
                    ATTACH_TEXT
                } else if head.starts_with(NO_MATCH)
                    || (head.starts_with(NO_READABLE) && !head.ends_with(" skipped)"))
                {
                    ATTACH_DOCS
                } else {
                    &run.markdown
                };
                Ok(format!("{head}\n\n{body}{SEARCHED}{}._", input.folder))
            }
            Err(SkillError::Tool(e)) => Err(e),
            Err(e) => Err(ToolError::Skill(e.to_string())),
        }
    }
}
