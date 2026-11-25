//! What the screens show, without the screens: the connection snippets, task prompts, research
//! parsing and paper checks of the old web UI's iris-core.js, the step lines of its app.js, a light
//! Markdown reader, and the number formats. Nothing here touches the window, so all of it is tested.

use std::path::PathBuf;
use std::sync::LazyLock;

use larch_agent::folio::{
    Doc,
    cite::{self, CiteResult},
    map,
};
use larch_app::contract::DocState;
use regex::Regex;
use serde_json::Value;

/// The knob changes of the engine now running. A loaded model is a new engine whose revisions
/// start again, so its ledger starts empty and an Undo the old engine answers is dropped.
#[derive(Default)]
pub struct Ledger {
    /// Newest first.
    pub changes: Vec<larch_app::contract::ConfigChange>,
    reverted: Vec<u64>,
    /// Counts engines, so a callback can tell whether its engine is still the one running.
    pub engine: u64,
}

impl Ledger {
    /// Forgets the old engine's changes.
    pub fn new_engine(&mut self) {
        self.changes.clear();
        self.reverted.clear();
        self.engine += 1;
    }

    /// Records that `rev` of `engine` was undone; false once that engine is gone.
    pub fn undone(&mut self, engine: u64, rev: u64) -> bool {
        let live = engine == self.engine;
        if live {
            self.reverted.push(rev);
        }
        live
    }

    /// The newest change to `knob`, and its revision while Undo can still put it back.
    pub fn last(&self, knob: &str) -> (Option<&larch_app::contract::ConfigChange>, Option<u64>) {
        let last = self.changes.iter().find(|c| c.knob == knob);
        let undo = last
            .filter(|c| c.cause != "inverse" && !self.reverted.contains(&c.rev))
            .map(|c| c.rev);
        (last, undo)
    }
}

/// Bytes in a GiB, which the app calls GB as the OS does.
pub const GIB: f64 = 1024. * 1024. * 1024.;

/// What the 35B card says of its faster version before it is made: plain words, its speed and
/// its cost, as measured (bench/results/35b-vs-llamacpp-2026-11-25.md).
pub const FASTER_NOTE: &str = "Keeps the parts of the model it uses least in less detail, so more of it fits on the GPU. About a third faster here; answers change very slightly. Takes a few minutes and 16 GB of disk.";

/// The faster version's progress line at `frac` (0 to 1) after `secs` seconds: "42 % · 3 min left".
pub fn making_left(frac: f64, secs: f64) -> String {
    let pct = format!("{:.0} %", frac * 100.);
    if frac < 0.02 || secs <= 0. {
        return pct;
    }
    let left = secs * (1. - frac) / frac;
    match left < 90. {
        true => format!("{pct} · {left:.0} s left"),
        false => format!("{pct} · {:.0} min left", left / 60.),
    }
}

/// `b` bytes as "13.1 GB".
pub fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / GIB)
}

/// A count as "2.1M", "640k" or "12".
pub fn fmt_k(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1000 {
        format!("{}k", (n as f64 / 1e3).round())
    } else {
        n.to_string()
    }
}

/// A knob's value: whole numbers plain, others to two places.
pub fn fmt_v(v: f64) -> String {
    if v.fract() == 0. {
        format!("{v:.0}")
    } else {
        let s = format!("{v:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_owned()
    }
}

/// How long ago `then_ms` was, at `now_ms` (Unix milliseconds).
pub fn ago(then_ms: u64, now_ms: u64) -> String {
    let s = now_ms.saturating_sub(then_ms) / 1000;
    match s {
        0..5 => "just now".into(),
        5..60 => format!("{s} s ago"),
        60..3600 => format!("{} min ago", s / 60),
        3600..86400 => format!("{} h ago", s / 3600),
        86400..172800 => "yesterday".into(),
        _ => format!("{} days ago", s / 86400),
    }
}

/// The greeting for the local hour `h`.
pub fn greeting(h: u32) -> &'static str {
    match h {
        5..12 => "Good morning",
        12..17 => "Good afternoon",
        _ => "Good evening",
    }
}

/// The last part of a path, either separator.
pub fn base_name(p: &str) -> &str {
    p.rsplit(['/', '\\']).find(|s| !s.is_empty()).unwrap_or(p)
}

/// A quote from a page as read: Markdown links keep their words and lose the address (a source's
/// quote showed "[name](https://…" raw).
pub fn plain_links(s: &str) -> std::borrow::Cow<'_, str> {
    static LINK: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"!?\[([^\]]*)\]\([^)\s]*\)?").expect("a valid regex")
    });
    LINK.replace_all(s, "$1")
}

/// `s` on one line, cut to `n` characters with an ellipsis.
pub fn clip(s: &str, n: usize) -> String {
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.chars().count() > n {
        s.chars().take(n - 1).collect::<String>() + "…"
    } else {
        s
    }
}

/// A scrolled view at offset `y` (gpui's: 0 at the top, `-max` at the bottom) shows its end, to a
/// pixel of rounding. Only such a chat follows a reply as it streams; one scrolled back stays put.
pub fn at_end(y: f32, max: f32) -> bool {
    y <= 1. - max
}

/// A chat scrolled past its tree pins it full size, helpers and all, only while a run streams into
/// it at its end (`y`, `max` as [`at_end`] takes them); otherwise it pins the one-line banner. Keyed
/// to where the chat is, never to when the wheel last moved: a tree that grew back whenever a scroll
/// paused dropped down over the text and folded up again on every step.
pub fn crest_pinned(live: bool, y: f32, max: f32) -> bool {
    live && at_end(y, max)
}

/// The parameter count a model's name gives, in billions ("Qwen3-30B-A3B" is 30).
pub fn params(name: &str) -> Option<f64> {
    let b = name.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit()
            && (i == 0 || !b[i - 1].is_ascii_alphanumeric() && b[i - 1] != b'.')
        {
            let s = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            let mut j = i;
            while j < b.len() && b[j] == b' ' {
                j += 1;
            }
            if j < b.len()
                && b[j].eq_ignore_ascii_case(&b'b')
                && !b.get(j + 1).is_some_and(|c| c.is_ascii_alphabetic())
            {
                return name[s..i].parse().ok();
            }
        } else {
            i += 1;
        }
    }
    None
}

/// An http(s) URL without credentials, as the browser would write it; anything else is None.
pub fn web_url(value: &str) -> Option<String> {
    let u = url::Url::parse(value).ok()?;
    (matches!(u.scheme(), "http" | "https") && u.username().is_empty() && u.password().is_none())
        .then(|| u.to_string())
}

/// `value` as one single-quoted POSIX shell word.
pub fn shell(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn ps_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// A TCP port typed by the user.
pub fn port(value: &str) -> Result<u16, String> {
    let bad = || "Use a port from 1 to 65535.".to_owned();
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad());
    }
    match value.parse::<u32>() {
        Ok(p @ 1..=65535) => Ok(p as u16),
        _ => Err(bad()),
    }
}

/// What Connect apps fills in.
pub struct SnippetOptions<'a> {
    pub port: &'a str,
    pub mcp_port: &'a str,
    pub model: &'a str,
    pub id: &'a str,
    pub folder: &'a str,
    pub powershell: bool,
}

impl Default for SnippetOptions<'_> {
    fn default() -> Self {
        SnippetOptions {
            port: "8080",
            mcp_port: "8090",
            model: "",
            id: "",
            folder: "",
            powershell: cfg!(windows),
        }
    }
}

/// Copyable commands for other apps to use this computer's model.
#[derive(Debug)]
pub struct Snippets {
    pub base: String,
    pub start: String,
    pub curl: String,
    pub powershell: String,
    pub python: String,
    pub anthropic: String,
    pub claude: String,
    pub mcp: String,
    pub mcp_http: String,
    pub health: String,
}

fn js(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}

/// The snippets for `o`; refused for a bad port or one port for both servers.
pub fn snippets(o: &SnippetOptions) -> Result<Snippets, String> {
    let quote = |v: &str| if o.powershell { ps_quote(v) } else { shell(v) };
    let p = port(o.port)?;
    let mcp_port = port(o.mcp_port)?;
    if p == mcp_port {
        return Err("Use different ports for the model API and the assistant's MCP server.".into());
    }
    let base = format!("http://127.0.0.1:{p}");
    let model = if o.model.is_empty() {
        "/absolute/path/to/model.gguf"
    } else {
        o.model
    };
    let id = if o.id.is_empty() { "larch" } else { o.id };
    let folder = if o.folder.is_empty() {
        "/absolute/path/to/documents"
    } else {
        o.folder
    };
    let body = serde_json::json!({"model": id, "messages": [{"role": "user", "content": "Hello."}], "max_tokens": 128})
        .to_string();
    let vars = [
        ("ANTHROPIC_BASE_URL", base.as_str()),
        ("ANTHROPIC_AUTH_TOKEN", "local"),
        ("ANTHROPIC_DEFAULT_OPUS_MODEL", id),
        ("ANTHROPIC_DEFAULT_SONNET_MODEL", id),
        ("ANTHROPIC_DEFAULT_HAIKU_MODEL", id),
        ("CLAUDE_CODE_ATTRIBUTION_HEADER", "0"),
    ];
    let claude = if o.powershell {
        vars.iter()
            .map(|(k, v)| format!("$env:{k}={}\n", ps_quote(v)))
            .collect::<String>()
            + "claude"
    } else {
        vars.iter()
            .map(|(k, v)| match *k {
                "ANTHROPIC_AUTH_TOKEN" | "CLAUDE_CODE_ATTRIBUTION_HEADER" => {
                    format!("export {k}={v}\n")
                }
                _ => format!("export {k}={}\n", shell(v)),
            })
            .collect::<String>()
            + "claude"
    };
    Ok(Snippets {
        start: format!("larch serve -m {} --host 127.0.0.1 --port {p} --name {}", quote(model), quote(id)),
        curl: format!(
            "curl {} \\\n  -H 'Content-Type: application/json' \\\n  -d {}",
            shell(&format!("{base}/v1/chat/completions")),
            shell(&body)
        ),
        powershell: format!(
            "$body = @'\n{body}\n'@\nInvoke-RestMethod -Uri {} -Method Post -ContentType 'application/json' -Body $body",
            quote(&format!("{base}/v1/chat/completions"))
        ),
        python: format!(
            "from openai import OpenAI\n\nclient = OpenAI(\n    base_url={},\n    api_key=\"local\",\n)\nreply = client.chat.completions.create(\n    model={},\n    messages=[{{\"role\": \"user\", \"content\": \"Hello.\"}}],\n    max_tokens=128,\n)\nprint(reply.choices[0].message.content)",
            js(&format!("{base}/v1")),
            js(id)
        ),
        anthropic: format!(
            "from anthropic import Anthropic\n\nclient = Anthropic(\n    base_url={},\n    api_key=\"local\",\n)\nreply = client.messages.create(\n    model={},\n    max_tokens=128,\n    messages=[{{\"role\": \"user\", \"content\": \"Hello.\"}}],\n)\nprint(reply.content[0].text)",
            js(&base),
            js(id)
        ),
        claude,
        mcp: serde_json::to_string_pretty(&serde_json::json!({"mcpServers": {"larch": {"command": "larch-agent", "args": ["mcp", "-m", model, "--allow", folder]}}}))
            .expect("json serializes"),
        mcp_http: format!("larch-agent mcp -m {} --allow {} --http {mcp_port}", quote(model), quote(folder)),
        health: format!("curl {}", shell(&format!("{base}/health"))),
        base,
    })
}

/// A task the composer turns into an agent prompt.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Task {
    Research,
    Web,
    Docs,
}

/// The agent prompt for `task`; refused when its inputs are missing.
pub fn prompt(task: Task, question: &str, folder: &str) -> Result<String, String> {
    let q = question.trim();
    if q.is_empty() {
        return Err("Add a question to get started.".into());
    }
    Ok(match task {
        Task::Docs => {
            let f = folder.trim();
            if f.is_empty() {
                return Err("Choose a folder the assistant is allowed to read.".into());
            }
            format!(
                "Use ask_docs with folder {} and question {}. Cite the exact file and line; flag conflicting facts.",
                js(f),
                js(q)
            )
        }
        Task::Research | Task::Web => format!(
            "Use {} with question {}. Keep source quotes, citations, uncertainties, and the reason research stopped.",
            if task == Task::Research {
                "research"
            } else {
                "web_answer"
            },
            js(q)
        ),
    })
}

/// A numbered source a research answer cites.
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    pub n: u32,
    pub title: String,
    pub url: String,
    pub quote: String,
}

/// A heading and the text under it.
#[derive(Clone, Debug, Default)]
pub struct Section {
    pub title: String,
    pub text: String,
    pub level: usize,
}

/// A paper a read_papers answer reports on: its section index, and whether only key parts were read.
#[derive(Clone, Debug)]
pub struct Paper {
    pub n: u32,
    pub file: String,
    pub cut: bool,
    pub section: usize,
}

/// What a research, paper or document answer holds. A quote parsed from assistant text is never
/// called verified.
#[derive(Clone, Debug, Default)]
pub struct Answer {
    pub sources: Vec<Source>,
    pub sections: Vec<Section>,
    pub warnings: Vec<String>,
    pub papers: Vec<Paper>,
    pub docs: Vec<(String, u32)>,
    pub stopped: String,
}

macro_rules! re {
    ($name:ident, $re:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($re).expect("a valid regex"));
    };
}
re!(FENCE, r"^\s*(```|~~~)");
re!(SOURCES, r"(?i)^(?:#{1,4}\s+)?Sources\s*:?\s*$");
re!(UNVERIFIED, r"(?i)^(?:#{1,4}\s+)?Could not verify\s*:?\s*$");
re!(
    SOURCE,
    r#"^\[(\d+)\]\s+(.*?)(https?://\S+?)(?:,?\s+["“]([\s\S]*)["”])?\s*$"#
);
re!(HEADING, r"^(#{1,4})\s+(.+)$");
re!(PAPER, r"^\[(\d+)\]\s+(.+?)( \(key parts read\))?$");
re!(
    WARN,
    r"(?i)could not verify|conflict:|conflicting facts|check these numbers|not read:|cut at the length limit|named no page"
);
// A web answer's note that a citation named no page: the panel lists it, the chat beside it not.
re!(
    DROPPED,
    r"^_\d+ citations? named no page and (?:was|were) dropped\._$"
);
re!(STOPPED, r"_Stopped:\s*([\s\S]*?)\._");
re!(DOC, r"(?:^|[\s`(\[])([^\s`<>()\[\]]+\.(?:md|txt)):(\d+)\b");

/// Parses a research, paper or document answer's Markdown.
pub fn parse_answer(text: &str) -> Answer {
    let mut a = Answer::default();
    let mut ambiguous = std::collections::BTreeSet::new();
    let (mut in_sources, mut in_warnings, mut fenced) = (false, false, false);
    let mut current: Option<usize> = None;
    for line in text.split('\n') {
        if FENCE.is_match(line) {
            fenced = !fenced;
            if let Some(c) = current {
                a.sections[c].text += &format!("{line}\n");
            }
            continue;
        }
        if !fenced && SOURCES.is_match(line) {
            (in_sources, in_warnings, current) = (true, false, None);
            continue;
        }
        if !fenced && UNVERIFIED.is_match(line) {
            (in_warnings, in_sources, current) = (true, false, None);
            continue;
        }
        if !fenced
            && in_sources
            && let Some(m) = SOURCE.captures(line)
        {
            let n: u32 = m[1].parse().unwrap_or(0);
            let quote = m.get(4).map_or("", |q| q.as_str()).to_owned();
            if let Some(url) = web_url(m[3].trim_end_matches(',')) {
                match a.sources.iter().find(|s| s.n == n) {
                    Some(p) if p.url != url || p.quote != quote => {
                        ambiguous.insert(n);
                    }
                    Some(_) => {}
                    None => {
                        let t = m[2].trim_end().trim_end_matches(',').trim();
                        let title = if t.is_empty() {
                            url::Url::parse(&url)
                                .ok()
                                .and_then(|u| u.host_str().map(str::to_owned))
                                .unwrap_or_default()
                        } else {
                            t.to_owned()
                        };
                        a.sources.push(Source {
                            n,
                            title,
                            url,
                            quote,
                        });
                    }
                }
            }
            continue;
        }
        if !fenced && let Some(h) = HEADING.captures(line) {
            (in_sources, in_warnings) = (false, false);
            a.sections.push(Section {
                title: h[2].to_owned(),
                text: String::new(),
                level: h[1].len(),
            });
            current = Some(a.sections.len() - 1);
            if let Some(p) = PAPER.captures(&h[2]) {
                a.papers.push(Paper {
                    n: p[1].parse().unwrap_or(0),
                    file: p[2].to_owned(),
                    cut: p.get(3).is_some(),
                    section: a.sections.len() - 1,
                });
            }
            continue;
        }
        if !fenced && (in_warnings || WARN.is_match(line)) && !line.trim().is_empty() {
            let w = line
                .trim_start_matches(['-', '*', '_', ' ', '\t'])
                .trim_end_matches(['_', '*']);
            a.warnings.push(w.to_owned());
        }
        if let Some(c) = current {
            a.sections[c].text += &format!("{line}\n");
        }
    }
    a.stopped = STOPPED
        .captures(text)
        .map(|m| m[1].to_owned())
        .unwrap_or_default();
    a.docs = DOC
        .captures_iter(text)
        .map(|m| (m[1].to_owned(), m[2].parse().unwrap_or(0)))
        .collect();
    for n in &ambiguous {
        a.warnings.push(format!(
            "Source [{n}] was reused for different passages. Inspect the full report."
        ));
    }
    a.sources.retain(|s| !ambiguous.contains(&s.n));
    a
}

/// The reply as the chat shows it beside the side panel: without a "Sources" list the panel shows
/// instead. Only a list whose every line is a source the panel takes from the whole reply (a web
/// address, no number reused anywhere in it) is left out; anything else in it, or after it, such
/// as a "could not verify" note, stays, but a note that citations named no page goes to the
/// panel's notes alone: it is about the sources, not the answer. Copy still takes the whole reply.
pub fn without_sources(text: &str) -> String {
    let panel = parse_answer(text).sources;
    let lines: Vec<&str> = text.split('\n').collect();
    let mut keep = vec![true; lines.len()];
    let mut fenced = false;
    let mut i = 0;
    while i < lines.len() {
        if FENCE.is_match(lines[i]) {
            fenced = !fenced;
        }
        if !fenced && DROPPED.is_match(lines[i].trim()) {
            keep[i] = false;
            // With a blank line beside it, so no gap is left where it was.
            if lines.get(i + 1).is_some_and(|l| l.trim().is_empty()) {
                keep[i + 1] = false;
            } else if i > 0 && lines[i - 1].trim().is_empty() {
                keep[i - 1] = false;
            }
        }
        if fenced || !SOURCES.is_match(lines[i]) {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < lines.len() && (lines[j].trim().is_empty() || SOURCE.is_match(lines[j])) {
            j += 1;
        }
        let listed = lines[i + 1..j]
            .iter()
            .filter(|l| !l.trim().is_empty())
            .count();
        let block = parse_answer(&lines[i..j].join("\n")).sources;
        if listed > 0 && block.len() == listed && block.iter().all(|s| panel.contains(s)) {
            keep[i..j].iter_mut().for_each(|k| *k = false);
        }
        i = j;
    }
    let kept: Vec<&str> = (lines.iter().zip(&keep))
        .filter_map(|(l, k)| k.then_some(*l))
        .collect();
    kept.join("\n").trim_end().to_owned()
}

/// The chats as the lists show them: pinned ones first, in pin order, then the rest as given
/// (newest first); archived ones apart.
pub fn arrange<'a>(
    chats: &'a [larch_app::contract::ChatEntry],
    pinned: &[String],
    archived: &[String],
) -> (
    Vec<&'a larch_app::contract::ChatEntry>,
    Vec<&'a larch_app::contract::ChatEntry>,
) {
    let kept = |c: &&larch_app::contract::ChatEntry| !archived.contains(&c.id);
    let mut shown: Vec<_> = pinned
        .iter()
        .filter_map(|p| chats.iter().find(|c| &c.id == p))
        .filter(kept)
        .collect();
    shown.extend(
        chats
            .iter()
            .filter(|c| !pinned.contains(&c.id))
            .filter(kept),
    );
    (shown, chats.iter().filter(|c| !kept(c)).collect())
}

/// For each of `sections` (in order), the index of its heading among `md_blocks(text)`.
pub fn heading_blocks(text: &str, sections: &[Section]) -> Vec<Option<usize>> {
    let blocks = md_blocks(text);
    let mut from = 0;
    sections
        .iter()
        .map(|s| {
            let b = (from..blocks.len())
                .find(|&b| matches!(&blocks[b], Block::Heading(_, t) if *t == s.title))?;
            from = b + 1;
            Some(b)
        })
        .collect()
}

re!(
    PAPER_MSG,
    r#"^Use read_papers with files (\[[^\n]*\]) and focus ("(?:[^"\\]|\\.)*")\."#
);

/// A saved paper request as the user typed it: the question and the files. Only chats saved by an
/// older version hold one: its paper chips sent each as this read_papers prompt.
pub fn paper_message(text: &str) -> Option<(String, Vec<String>)> {
    let m = PAPER_MSG.captures(text)?;
    let files: Vec<String> = serde_json::from_str(&m[1]).ok()?;
    let question: String = serde_json::from_str(&m[2]).ok()?;
    (1..=8).contains(&files.len()).then_some((question, files))
}

/// What "+ Attach" does with the picked `paths`: the images, which join the next message, and the
/// paths of every other file or folder, a PDF, Markdown or text file too, which are attached to
/// the chat: Folio reads its documents, ask_docs and the file tools the rest. A paper kept aside
/// for a read_papers prompt instead never reached Folio, and with nothing attached the agent
/// answered that it could not read the user's files.
pub fn attach_split(paths: Vec<PathBuf>) -> (Vec<PathBuf>, Vec<String>) {
    let (pictures, rest): (Vec<PathBuf>, Vec<PathBuf>) = paths
        .into_iter()
        .partition(|p| crate::images::image_path(p));
    (
        pictures,
        rest.iter().map(|p| p.display().to_string()).collect(),
    )
}

/// Whether papers "+ Attach" picked in an open chat wait for its next message (`attach_later`,
/// which the send attaches first): the chat is answering (`live`), or the app refused the attach
/// because a reply is being written (`refused`, its error). Dropped, they were lost (#591 review).
pub fn attach_waits(live: bool, refused: Option<&str>) -> bool {
    live || refused == Some(larch_app::BUSY)
}

/// A cite of a pinned-document answer, ready for the page view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CiteLink {
    /// What its button says: "p. 3", "B, p. 12", and ": quote not found" after it when it is not.
    pub label: String,
    /// The quote is in none of the documents; the view shows the page the cite names, unmarked.
    pub missing: bool,
    /// The page view's title: "report.pdf, p. 3", the page the quote is on.
    pub title: String,
    /// That page's text.
    pub page: String,
    /// The bytes of `page` the quote is, which the view highlights.
    pub hit: Option<std::ops::Range<usize>>,
}

/// The cites of `answer` to `docs`, as folio checks them (the answer is not changed), each once:
/// a cite whose quote is elsewhere opens the page where it is.
pub fn cite_links(answer: &str, docs: &[Doc]) -> Vec<CiteLink> {
    let mut out: Vec<CiteLink> = Vec::new();
    for v in cite::check(answer, docs, &map::labels(docs)).results {
        let (Ok(result), Some(at)) = (&v.result, v.at) else {
            continue;
        };
        let Some(doc) = docs.iter().find(|d| d.index == at.doc) else {
            continue;
        };
        let (n, missing) = (doc.page_of(at.start), *result == CiteResult::NotFound);
        let page = &doc.pages[n];
        let place = v.place.trim_start_matches("on ").trim_start_matches("in ");
        let mut label = match docs.len() {
            1 => place.to_owned(),
            _ => format!(
                "{}, {place}",
                v.doc.map_or(doc.letter(), |d| docs[d].letter())
            ),
        };
        if missing {
            label.push_str(": quote not found");
        }
        let link = CiteLink {
            label,
            missing,
            title: format!("{}, p. {}", base_name(&doc.name), n + 1),
            page: doc
                .text
                .get(page.start..page.end)
                .unwrap_or_default()
                .to_owned(),
            hit: (!missing).then(|| {
                let from = |i: usize| i.clamp(page.start, page.end) - page.start;
                from(at.start)..from(at.end)
            }),
        };
        if !out.contains(&link) {
            out.push(link);
        }
    }
    out
}

/// What the chip of a chat's documents says to add when other chats slow down for them.
pub const SLOWER: &str = "Other chats will be slower while this document is ready.";

/// What a chat's pinned documents show: the chip's words, and the last follow-up's first word.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Docs {
    /// "Document · Reading p. 40 of 120", "Document · Ready"; `None` until the agent reports.
    pub chip: Option<String>,
    /// `SLOWER`, while part of the model is in host memory for the documents.
    pub slower: Option<&'static str>,
    /// The last answer's speed in the agent's words ("First word in 1.2 s").
    pub first_word: Option<String>,
    /// The chip says a reload or a read is under way, which only a later report ends.
    pub busy: bool,
}

impl Docs {
    /// Takes the agent's next report. The chip says "Document" so it never reads as a pinned chat.
    pub fn apply(&mut self, state: &DocState, text: &str) {
        match state {
            DocState::Answered { .. } => self.first_word = Some(text.to_owned()),
            // The agent's words for it say "Ready; other chats may be slower": the line is ours.
            DocState::FitsSlower { .. } => {
                self.chip = Some("Document · Ready".into());
                self.slower = Some(SLOWER);
            }
            // The pin went: nothing is ready, and other chats are no longer slower (#591 review).
            DocState::Released => {
                self.chip = None;
                self.slower = None;
            }
            _ => {
                self.chip = Some(format!("Document · {text}"));
                self.slower = None;
            }
        }
        if !matches!(state, DocState::Answered { .. }) {
            self.busy = matches!(state, DocState::MakingRoom { .. })
                || matches!(state, DocState::Reading { done, total, .. } if done < total);
        }
    }

    /// The turn ended (an answer, an error or a Stop): a reload or read it left under way was
    /// refused or stopped, so the chip stops saying it is still going; the notice says why.
    pub fn ended(&mut self) {
        if std::mem::take(&mut self.busy) {
            self.chip = None;
            self.slower = None;
        }
    }
}

re!(
    RESEARCH,
    r"(?i)^\s*(?:(?:please|can you|could you)\s+)*(?:use research|do (?:some |deep )?research|(?:deep )?research|investigate)\b"
);
re!(
    WEB,
    r"(?i)^\s*(?:(?:please|can you|could you)\s+)*(?:use web_answer|search the web|look up)\b"
);

/// What a typed message asks for: research, a web answer, or a plain chat.
pub fn research_mode(text: &str) -> Option<Task> {
    if RESEARCH.is_match(text) {
        Some(Task::Research)
    } else if WEB.is_match(text) {
        Some(Task::Web)
    } else {
        None
    }
}

/// The task a send runs: the Research chip as chosen (on a project folder it reads its docs), and
/// with no chip what the words ask for, unless the chat has files attached or this
/// send attaches some (`files`): "Investigate the methods in this paper" is a question on the
/// documents, not one for research or the web (#591 review).
pub fn send_task(chip: Option<&str>, typed: &str, repo: bool, files: bool) -> Option<Task> {
    match chip {
        Some("Research") if repo => Some(Task::Docs),
        Some("Research") => Some(Task::Research),
        Some(_) => None,
        // Images with no words are a plain chat turn: the model reads them.
        None if typed.is_empty() || files => None,
        None => research_mode(typed),
    }
}

/// What to tell the user about a local server's HTTP status.
pub fn http_message(code: u16) -> String {
    match code {
        401 => "The API key is missing or incorrect.".into(),
        429 => "The engine is busy. Wait a moment, then retry.".into(),
        503 => "The model is loading or the GPU is unavailable. Check the server and retry.".into(),
        _ => format!("The server returned HTTP {code}."),
    }
}

/// A saved fact's sentence, without the user's words a capture keeps after it.
pub fn fact_sentence(text: &str) -> &str {
    text.split("\nSaid: ").next().unwrap_or(text)
}

/// Where a fact came from, as the Memory page says it: a capture's `chat <id> turn <n>` by the
/// chat's `title` while it is kept ("a deleted chat" once it is not), anything else as written.
pub fn fact_source(source: &str, title: impl Fn(&str) -> Option<String>) -> String {
    let chat = source
        .strip_prefix("chat ")
        .and_then(|r| r.rsplit_once(" turn "))
        .filter(|(_, n)| n.parse::<u32>().is_ok());
    match chat {
        Some((id, _)) => title(id).unwrap_or_else(|| "a deleted chat".into()),
        None if source.is_empty() => "you".into(),
        None => source.into(),
    }
}

/// A run's access as the Tasks page says it: an MCP server's grant (`mcp.<id>.<install>`) by the
/// server's `name` (its id once it is gone), the web's grants as "the web", any other as written.
/// The install's random token said nothing to the user ("mcp.notion.9XPnMFyRsBOCmgViX1B7Lw").
/// A run without the web says "no web": one saved with "Let tasks use the web" off made up the
/// listings it was asked to find, and nothing on the page said why. A run saved with it that the
/// switch, now off (`web_on`), takes it from says so: its next step gets neither web tool.
pub fn access(scope: &[String], name: impl Fn(&str) -> Option<String>, web_on: bool) -> String {
    let mut out: Vec<String> = Vec::new();
    for grant in scope {
        let said = match grant.strip_prefix("mcp.").and_then(|r| r.rsplit_once('.')) {
            Some((id, _)) => name(id).unwrap_or_else(|| id.to_owned()),
            None if grant.starts_with("network.") && web_on => "the web".into(),
            None if grant.starts_with("network.") => "the web (switched off)".into(),
            None => grant.clone(),
        };
        if !out.contains(&said) {
            out.push(said);
        }
    }
    if !scope.iter().any(|g| g.starts_with("network.")) {
        out.push("no web".into());
    }
    out.join(", ")
}

/// Columns of at least `min` wide, `gap` apart, that fit `width`: one at least, `max` at most.
pub fn columns(width: f32, min: f32, gap: f32, max: u16) -> u16 {
    (((width + gap) / (min + gap)).floor().max(1.) as u16).min(max)
}

/// What an Undo line says after the backend answered, and whether Undo may be tried again: a fact
/// changed since cannot be undone; any other failure (a locked file) can be.
pub fn undo_outcome(err: Option<&str>) -> (String, bool) {
    match err {
        None => ("Undone".into(), false),
        Some(e) if e.contains("changed since") => {
            ("Not undone: it changed since it was saved".into(), false)
        }
        Some(e) => (format!("Not undone: {e}"), true),
    }
}

/// A tool call as the user reads it: what it is doing now, and the step it leaves behind.
pub fn describe(name: &str, a: &Value) -> (String, String) {
    let s = |k: &str| a.get(k).and_then(Value::as_str).unwrap_or("");
    let f = base_name(s("path"));
    // A step a done call leaves keeps a relative path ("src/lib.rs"), whose folders say which
    // file; a full one names the user's home and folders, and shows as the file's name alone.
    let path = match std::path::Path::new(s("path")).is_absolute() {
        true => f,
        false => s("path"),
    };
    let q = ["query", "question", "q", "search"]
        .iter()
        .map(|k| s(k))
        .find(|v| !v.is_empty())
        .unwrap_or("");
    let br = if s("branch").is_empty() {
        "the branch"
    } else {
        s("branch")
    };
    let or = |v: &str, d: &str| {
        if v.is_empty() {
            d.to_owned()
        } else {
            v.to_owned()
        }
    };
    let range = match (
        a.get("start_line").and_then(Value::as_u64),
        a.get("end_line").and_then(Value::as_u64),
    ) {
        (Some(s), Some(e)) => format!(", lines {s}–{e}"),
        (Some(s), None) => format!(", lines {s} on"),
        _ => String::new(),
    };
    let r = |now: String, done: String| (now, done);
    match name {
        "research" => r(
            "Researching your question".into(),
            "Researched your question".into(),
        ),
        "web_answer" => r(
            "Finding an answer with sources".into(),
            "Found an answer with sources".into(),
        ),
        "ask_docs" => r(
            "Reading your documents".into(),
            "Read your documents".into(),
        ),
        "read_papers" => r(
            match a
                .get("papers")
                .and_then(Value::as_array)
                .map_or(0, Vec::len)
            {
                1 => "Reading 1 paper".into(),
                n => format!("Reading {n} papers"),
            },
            "Read your papers".into(),
        ),
        "list_dir" => r(
            format!("Looking in {}", or(f, "a folder")),
            format!("Looked in {}", or(path, "a folder")),
        ),
        "read_file" | "view" | "show_file" => r(
            format!("Reading {}", or(f, "a file")),
            format!("Read {}{range}", or(path, "a file")),
        ),
        "search_files" => r(
            format!("Searching files for “{}”", clip(q, 40)),
            format!("Searched files for “{}”", clip(q, 48)),
        ),
        "find_files" => r(
            format!("Looking for {}", clip(s("pattern"), 40)),
            format!("Looked for {}", clip(s("pattern"), 48)),
        ),
        "write_file" => r(
            format!("Writing {}", or(f, "a file")),
            format!("Wrote {}", or(path, "a file")),
        ),
        "changed_files" => r(
            format!("Checking what changed on {br}"),
            format!("Listed what changed on {br}"),
        ),
        "diff_file" => r(
            format!("Reading the changes to {}", or(f, "a file")),
            format!("Read the changes to {}", or(path, "a file")),
        ),
        "log" => r(
            format!("Reading the history of {br}"),
            format!("Read the history of {br}"),
        ),
        "recall" => r(
            "Checking what I remember".into(),
            format!("Looked in my memory for “{}”", clip(q, 40)),
        ),
        "remember" => r(
            "Remembering this".into(),
            format!("Remembered “{}”", clip(s("text"), 48)),
        ),
        "forget" => r(
            format!("Forgetting {}", or(s("slug"), "a note")),
            format!("Forgot {}", or(s("slug"), "a note")),
        ),
        "search" => r(
            format!("Searching the code for “{}”", clip(s("pattern"), 40)),
            format!(
                "Searched {} for “{}”",
                or(path, "the code"),
                clip(s("pattern"), 48)
            ),
        ),
        "edit" => r(
            format!("Editing {}", or(f, "a file")),
            format!("Edited {}", or(path, "a file")),
        ),
        "run_tests" => r("Running the tests".into(), "Ran the tests".into()),
        "bash" => r(
            format!("Running {}", clip(s("command"), 40)),
            format!("Ran {}", clip(s("command"), 64)),
        ),
        // An MCP server's tool (`<server>__<tool>`), before the web words its tool name may hold.
        _ if crate::mcp::outside_tool(name).is_some() => {
            let (server, tool) = crate::mcp::outside_tool(name).unwrap_or_default();
            let (server, tool) = (server.replace('_', " "), tool.replace('_', " "));
            r(
                format!("Asking {server}: {tool}"),
                format!("Asked {server}: {tool}"),
            )
        }
        _ if ["search", "web", "url", "browse", "fetch", "http", "page"]
            .iter()
            .any(|w| name.contains(w)) =>
        {
            let u = or(s("url"), s("href"));
            if u.is_empty() {
                r(
                    "Searching the web".into(),
                    format!("Searched the web for “{}”", clip(q, 48)),
                )
            } else {
                let host = u
                    .split("://")
                    .nth(1)
                    .unwrap_or(&u)
                    .split(['/', '?', '#'])
                    .next()
                    .unwrap_or("");
                r(
                    format!("Opening {}", host.trim_start_matches("www.")),
                    format!("Opened {}", clip(&u, 64)),
                )
            }
        }
        _ => {
            let w = name.replace('_', " ");
            r(format!("Running {w}"), format!("Ran {w}"))
        }
    }
}

re!(LINES, r" \((\d+) lines\)$");
re!(TESTS, r"(\d+) passed, (\d+) failed");
re!(AT_LINE, r" at line (\d+)");

re!(
    CALL_TAIL,
    r"(?:^|\n)[ \t]*(?:(?:<tool_call>|<function-call>|<\|python_tag\|>|\[TOOL_CALLS\])\s*(?:(?:[\[{]|<function=)[\s\S]*)?|<function=[^>\n]+>[\s\S]*)$"
);

/// A saved reply as words: a reply a retry followed keeps its call's markup, an opener the parser
/// reads (larch's `toolcall::Family::wrappers`) at a line's start, then a call's JSON, Qwen 3.5's
/// XML body (`<function=NAME>`) or nothing, or that XML body bare, as the parser also reads it;
/// one named in prose or a code block stays.
pub fn reply_words(text: &str) -> String {
    let Some(call) = CALL_TAIL.find(text) else {
        return text.trim_end().to_owned();
    };
    // An opener inside a code block that is still open is an example: the fences are
    // `md_blocks`'s, so a `~~~` block counts, and a block is closed only by its own fence.
    let mut open: Option<(u8, usize)> = None;
    for line in text[..call.start()].lines() {
        open = match open {
            Some((ch, n)) => {
                let l = line.trim();
                (l.len() < n || !l.bytes().all(|b| b == ch)).then_some((ch, n))
            }
            None => (MD_FENCE.captures(line)).map(|f| (f[1].as_bytes()[0], f[1].len())),
        };
    }
    match open {
        None => text[..call.start()].trim_end().to_owned(),
        Some(_) => text.trim_end().to_owned(),
    }
}

/// A saved tool result's step line: a count, as the live card's summary (larch-agent's
/// `run::summary`), since a file's first line, such as a markdown list item, says nothing of the
/// call.
pub fn saved_step(content: &str) -> String {
    let lines: Vec<&str> = content
        .strip_suffix('\n')
        .unwrap_or(content)
        .split('\n')
        .collect();
    match lines.len() {
        1 => lines[0].chars().take(80).collect(),
        n => format!("{n} lines"),
    }
}

/// Why a code run stopped, for its packet card.
pub fn stop_line(why: &str) -> String {
    if why.contains("nothing new") && why.ends_with("ended with the model's answer") {
        return "Stopped early: it ran out of room to hold the files it needed, so it could not \
                finish this"
            .into();
    }
    format!("Stopped: {why}")
}

/// A tool result's line as the user wants it: what came of the call, not the tool's words.
pub fn tidy_sum(name: &str, sum: &str) -> String {
    let lines = LINES.captures(sum).map(|m| m[1].to_owned());
    match name {
        "list_dir" => match sum.strip_suffix(" lines").or(sum.strip_suffix(" line")) {
            Some(k) if k.bytes().all(|b| b.is_ascii_digit()) => {
                format!("{k} {}", if k == "1" { "item" } else { "items" })
            }
            _ => sum.into(),
        },
        "write_file" if sum.starts_with("wrote ") && sum.contains(" bytes to ") => {
            format!("{} bytes", &sum[6..sum.find(" bytes").unwrap_or(6)])
        }
        "search" if sum.starts_with("no line matches ") => "no matches".into(),
        "search" => lines.map_or("1 match".into(), |n| format!("{n} matches")),
        "view" => lines.map_or("1 line".into(), |n| format!("{n} lines")),
        "edit" => AT_LINE
            .captures(sum)
            .map_or(String::new(), |m| format!("line {}", &m[1])),
        "run_tests" if TESTS.is_match(sum) => {
            let m = TESTS.captures(sum).expect("matched");
            if &m[2] != "0" {
                format!("{} failed, {} passed", &m[2], &m[1])
            } else {
                format!("all {} passed", &m[1])
            }
        }
        "bash" => LINES.replace(sum, "").into_owned(),
        _ => sum.into(),
    }
}

/// A block of a reply.
#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    Para(Vec<String>),
    List { ordered: bool, items: Vec<String> },
    Heading(usize, String),
    Code { lang: String, body: String },
}

re!(LI, r"^\s*([-*+]|\d+[.)])\s+");
re!(MD_FENCE, r"^\s*(`{3,}|~{3,})\s*([\w+#.-]*)");

fn block_start(t: &str) -> bool {
    MD_FENCE.is_match(t) || HEADING.is_match(t) && t.chars().take_while(|c| *c == '#').count() <= 4
}

/// Paragraphs, lists, headings and code fences. Links stay text.
pub fn md_blocks(text: &str) -> Vec<Block> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i];
        if t.trim().is_empty() {
            i += 1;
            continue;
        }
        if let Some(f) = MD_FENCE.captures(t) {
            let (ch, n) = (f[1].as_bytes()[0], f[1].len());
            let mut j = i + 1;
            let mut body = Vec::new();
            while j < lines.len() {
                let l = lines[j].trim();
                if l.len() >= n && l.bytes().all(|b| b == ch) {
                    break;
                }
                body.push(lines[j]);
                j += 1;
            }
            out.push(Block::Code {
                lang: f[2].to_owned(),
                body: body.join("\n"),
            });
            i = j + 1;
            continue;
        }
        if let Some(h) = HEADING.captures(t) {
            out.push(Block::Heading(h[1].len(), h[2].to_owned()));
            i += 1;
            continue;
        }
        let list = LI.is_match(t);
        let mut items: Vec<String> = Vec::new();
        let mut k = i;
        while k < lines.len() {
            let ln = lines[k];
            if ln.trim().is_empty() || k > i && block_start(ln) {
                break;
            }
            if !list {
                if k > i && LI.is_match(ln) {
                    break;
                }
                items.push(ln.to_owned());
            } else if LI.is_match(ln) {
                items.push(LI.replace(ln, "").into_owned());
            } else if ln.starts_with("  ") && !items.is_empty() {
                let last = items.len() - 1;
                items[last] = format!("{} {}", items[last], ln.trim());
            } else {
                break;
            }
            k += 1;
        }
        out.push(if list {
            Block::List {
                ordered: t.trim_start().starts_with(|c: char| c.is_ascii_digit()),
                items,
            }
        } else {
            Block::Para(items)
        });
        i = k.max(i + 1);
    }
    out
}

/// How a span of a line is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Span {
    Plain,
    Bold,
    Italic,
    Code,
}

re!(
    INLINE,
    r"`([^`\n]+)`|\*\*([^*\n]+)\*\*|(^|[\s(])[*_]([^*_\n]+)[*_]([\s).,;:!?]|$)|\[([^\]\n]+)\]\([^)\s]+\)"
);

/// A line split into spans: inline code, bold, italics; a link keeps only its text.
pub fn spans(s: &str) -> Vec<(String, Span)> {
    let mut out: Vec<(String, Span)> = Vec::new();
    let mut push = |t: &str, k: Span| {
        if t.is_empty() {
            return;
        }
        match out.last_mut() {
            Some((p, pk)) if *pk == k => p.push_str(t),
            _ => out.push((t.to_owned(), k)),
        }
    };
    let mut at = 0;
    for m in INLINE.captures_iter(s) {
        let whole = m.get(0).expect("whole match");
        push(&s[at..whole.start()], Span::Plain);
        let g = |i: usize| m.get(i).map_or("", |g| g.as_str());
        if m.get(1).is_some() {
            push(g(1), Span::Code);
        } else if m.get(2).is_some() {
            push(g(2), Span::Bold);
        } else if m.get(4).is_some() {
            push(g(3), Span::Plain);
            push(g(4), Span::Italic);
            push(g(5), Span::Plain);
        } else {
            push(g(6), Span::Plain);
        }
        at = whole.end();
    }
    push(&s[at..], Span::Plain);
    out
}

/// A unified diff's lines, each with its kind: a file header, an added or removed line, a hunk
/// header or context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffKind {
    File,
    Hunk,
    Add,
    Del,
    Same,
}

/// The lines of `diff` worth showing, at most `max`.
pub fn diff_lines(diff: &str, max: usize) -> Vec<(DiffKind, String)> {
    let mut out = Vec::new();
    for l in diff.lines() {
        if out.len() >= max {
            break;
        }
        let k = if l.starts_with("diff --git ") {
            out.push((
                DiffKind::File,
                l.rsplit(" b/").next().unwrap_or(l).to_owned(),
            ));
            continue;
        } else if l.starts_with("+++") || l.starts_with("---") || l.starts_with("index ") {
            continue;
        } else if l.starts_with("@@") {
            DiffKind::Hunk
        } else if l.starts_with('+') {
            DiffKind::Add
        } else if l.starts_with('-') {
            DiffKind::Del
        } else {
            DiffKind::Same
        };
        out.push((k, l.to_owned()));
    }
    out
}

/// The OS asks for less motion: GNOME's animations off, macOS's Reduce motion, or Windows's
/// "Animate controls and elements" off.
pub fn os_reduce_motion() -> bool {
    let out = |cmd: &str, args: &[&str]| {
        larch_agent::tool::no_window(&mut std::process::Command::new(cmd))
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    if cfg!(target_os = "macos") {
        out(
            "defaults",
            &["read", "com.apple.universalaccess", "reduceMotion"],
        )
        .is_some_and(|v| v == "1")
    } else if cfg!(windows) {
        out(
            "reg",
            &[
                "query",
                r"HKCU\Control Panel\Desktop\WindowMetrics",
                "/v",
                "MinAnimate",
            ],
        )
        .is_some_and(|v| v.trim_end().ends_with('0'))
    } else {
        out(
            "gsettings",
            &["get", "org.gnome.desktop.interface", "enable-animations"],
        )
        .is_some_and(|v| v == "false")
    }
}

/// `s` with its first letter a capital, as a service's lower-case error reads in a sentence.
pub fn capital(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map_or(String::new(), |f| f.to_uppercase().chain(c).collect())
}

/// A shortcut as Settings shows it: `M-n` with `m` "Ctrl" is Ctrl+N.
pub fn keys(k: &str, m: &str) -> String {
    let k = k.replace("M-", &format!("{m}-"));
    k.split('-').map(capital).collect::<Vec<_>>().join("+")
}

#[cfg(test)]
mod tests {
    #[test]
    fn making_left_reads_as_the_download_rows_do() {
        assert_eq!(super::making_left(0., 0.), "0 %");
        assert_eq!(super::making_left(0.01, 5.), "1 %");
        assert_eq!(super::making_left(0.5, 30.), "50 % · 30 s left");
        assert_eq!(super::making_left(0.25, 120.), "25 % · 6 min left");
        assert!(!super::FASTER_NOTE.to_lowercase().contains("repack"));
    }

    #[test]
    fn a_quoted_link_keeps_its_words() {
        let p = |s: &str| super::plain_links(s).into_owned();
        assert_eq!(
            p("by [VTECaddict](https://www.e90post.com/m.php?u=1) said"),
            "by VTECaddict said"
        );
        // A quote cut inside the address still loses it.
        assert_eq!(p("[skow](https://www.e90post.com/forums/mem"), "skow");
        assert_eq!(p("a [1] and [b] stay"), "a [1] and [b] stay");
    }

    #[test]
    fn a_facts_chat_is_named_by_its_title() {
        let title = |id: &str| (id == "17950-29476").then(|| "Create Notion Calendar".to_owned());
        let f = |s: &str| super::fact_source(s, title);
        assert_eq!(f("chat 17950-29476 turn 1"), "Create Notion Calendar");
        assert_eq!(f("chat 17946-19596 turn 12"), "a deleted chat");
        assert_eq!(f(""), "you");
        assert_eq!(f("chat about turnips"), "chat about turnips");
        assert_eq!(f("chat x turn two"), "chat x turn two");
    }

    #[test]
    fn a_runs_access_names_its_servers_and_the_web() {
        let name = |id: &str| (id == "notion").then(|| "Notion".to_owned());
        let scope = |g: &[&str]| g.iter().map(|g| g.to_string()).collect::<Vec<_>>();
        let a = |g: &[&str]| super::access(&scope(g), name, true);
        assert_eq!(a(&["mcp.notion.9XPnMFyRsBOCmgViX1B7Lw"]), "Notion, no web");
        let web = ["network.web_search", "network.fetch_page", "mcp.gone.abc"];
        assert_eq!(a(&web), "the web, gone");
        assert_eq!(a(&["repo"]), "repo, no web");
        // A task saved with "Let tasks use the web" off: its steps cannot look anything up.
        assert_eq!(a(&[]), "no web");
        // Saved with it on, then switched off: its next step gets neither web tool.
        let off = super::access(&scope(&web), name, false);
        assert_eq!(off, "the web (switched off), gone");
    }

    #[test]
    fn a_narrow_page_drops_grid_columns() {
        assert_eq!(super::columns(1700., 260., 16., 3), 3);
        // The window at its smallest: 880 less the sidebar and the page's margins.
        assert_eq!(super::columns(552., 260., 16., 3), 2);
        assert_eq!(super::columns(500., 260., 16., 3), 1);
        assert_eq!(super::columns(0., 260., 16., 3), 1);
    }

    #[test]
    fn an_mcp_tool_step_names_its_server_not_the_web() {
        let (now, done) = super::describe("github__search_issues", &serde_json::json!({"q": "x"}));
        assert_eq!(now, "Asking github: search issues");
        assert_eq!(done, "Asked github: search issues");
    }

    use super::*;

    /// Pinned chats lead in pin order, archived ones leave the list (pinned or not), and a pin
    /// for a deleted chat is skipped.
    #[test]
    fn pinned_lead_and_archived_go_apart() {
        let c = |id: &str| larch_app::contract::ChatEntry {
            id: id.into(),
            title: id.into(),
            updated_at: 0,
            turns: 1,
        };
        let chats = [c("a"), c("b"), c("c"), c("d")];
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let (shown, arch) = arrange(&chats, &s(&["c", "gone", "d"]), &s(&["d", "b"]));
        let ids = |v: Vec<&larch_app::contract::ChatEntry>| {
            v.into_iter().map(|c| c.id.clone()).collect::<Vec<_>>()
        };
        assert_eq!(ids(shown), ["c", "a"]);
        assert_eq!(ids(arch), ["b", "d"]);
    }

    /// Each outline entry finds its own heading, a repeated title included, and skips the
    /// Sources heading that is no section.
    #[test]
    fn outline_headings_map_to_blocks() {
        let text = "# A\n\npara\n\n## B\n\n```\n# not\n```\n\n## B\n\n## Sources\n";
        let a = parse_answer(text);
        assert_eq!(
            heading_blocks(text, &a.sections),
            [Some(0), Some(2), Some(4)]
        );
    }

    #[test]
    fn a_chat_follows_its_reply_only_from_the_end() {
        assert!(at_end(0., 0.), "a thread shorter than the view");
        assert!(at_end(-500., 500.));
        assert!(at_end(-499.5, 500.), "rounding");
        assert!(!at_end(-400., 500.), "scrolled back");
        assert!(!at_end(0., 500.), "at the top");
    }

    /// Reading back through a chat, paused or not, keeps the banner; only a run followed at the
    /// end shows the full tree.
    #[test]
    fn the_pinned_tree_stays_a_banner_while_reading_back() {
        assert!(
            !crest_pinned(true, -400., 500.),
            "scrolled back during a run"
        );
        assert!(
            !crest_pinned(false, -500., 500.),
            "at the end, nothing running"
        );
        assert!(crest_pinned(true, -500., 500.), "following a run");
    }

    fn opts<'a>(port: &'a str, mcp_port: &'a str) -> SnippetOptions<'a> {
        SnippetOptions {
            port,
            mcp_port,
            powershell: false,
            ..Default::default()
        }
    }

    #[test]
    fn a_new_engine_drops_the_old_ledger_and_its_late_undo() {
        let change = |rev, cause: &str| larch_app::contract::ConfigChange {
            engine: 1,
            rev,
            knob: "max_batch_seqs".into(),
            old: 16.,
            new: 20.,
            cause: cause.into(),
            at: 0,
        };
        let mut l = Ledger::default();
        l.changes.insert(0, change(3, "human"));
        assert_eq!(l.last("max_batch_seqs").1, Some(3));
        let old = l.engine;
        assert!(l.undone(old, 3));
        assert_eq!(l.last("max_batch_seqs").1, None);
        l.new_engine();
        assert!(l.last("max_batch_seqs").0.is_none());
        // The new engine numbers from the start again: its rev 3 is undoable, and an answer
        // from the old engine does not mark it.
        l.changes.insert(0, change(3, "gait"));
        assert!(!l.undone(old, 3));
        assert_eq!(l.last("max_batch_seqs").1, Some(3));
        l.changes.insert(0, change(4, "inverse"));
        assert_eq!(l.last("max_batch_seqs").1, None);
    }

    #[test]
    fn client_examples_use_the_right_base_paths_names_and_separate_mcp_ports() {
        let s = snippets(&SnippetOptions {
            port: "8123",
            mcp_port: "8124",
            model: "/my models/local.gguf",
            folder: "/documents",
            id: "local",
            powershell: false,
        })
        .unwrap();
        assert_eq!(s.base, "http://127.0.0.1:8123");
        assert!(s.python.contains("base_url=\"http://127.0.0.1:8123/v1\""));
        assert!(s.anthropic.contains("base_url=\"http://127.0.0.1:8123\""));
        assert!(
            s.claude
                .contains("ANTHROPIC_BASE_URL='http://127.0.0.1:8123'")
        );
        assert!(s.claude.contains("ANTHROPIC_DEFAULT_HAIKU_MODEL='local'"));
        assert!(s.start.contains("--name 'local'"));
        assert!(s.mcp_http.ends_with("--http 8124"));
        let mcp: Value = serde_json::from_str(&s.mcp).unwrap();
        assert_eq!(
            mcp["mcpServers"]["larch"]["args"],
            serde_json::json!([
                "mcp",
                "-m",
                "/my models/local.gguf",
                "--allow",
                "/documents"
            ])
        );
        assert!(!s.mcp.contains("--write"));
    }

    #[test]
    fn bad_ports_and_transport_collisions_cannot_produce_copyable_commands() {
        for p in ["0", "-1", "65536", "8080\n--host", "80.5", "abc", ""] {
            assert!(snippets(&opts(p, "8090")).is_err(), "{p:?}");
        }
        assert!(
            snippets(&opts("8090", "8090"))
                .unwrap_err()
                .contains("different ports")
        );
        assert_eq!(port("65535"), Ok(65535));
        assert_eq!(port("1"), Ok(1));
    }

    #[cfg(unix)]
    #[test]
    fn shell_snippets_keep_path_text_as_one_literal_argument() {
        let model = "/path with space/it's-$(printf EXECUTED)-`printf EXECUTED`.gguf";
        let s = snippets(&SnippetOptions {
            model,
            powershell: false,
            ..Default::default()
        })
        .unwrap();
        let out = std::process::Command::new("/bin/sh")
            .args([
                "-c",
                &format!("larch() {{ printf '%s\\n' \"$@\"; }}\n{}", s.start),
            ])
            .output()
            .unwrap();
        assert!(out.status.success());
        let stdout = String::from_utf8(out.stdout).unwrap();
        assert_eq!(stdout.split('\n').nth(2), Some(model));
        assert!(!stdout.contains("its-EXECUTED"));
        assert!(!s.curl.contains("\n+"));
        assert!(
            std::process::Command::new("/bin/sh")
                .args(["-n", "-c", &s.curl])
                .status()
                .unwrap()
                .success()
        );
    }

    #[test]
    fn task_prompts_preserve_the_folder_and_reject_empty_input() {
        assert!(
            prompt(Task::Research, " ", "")
                .unwrap_err()
                .contains("question")
        );
        assert!(
            prompt(Task::Docs, "What changed?", "")
                .unwrap_err()
                .contains("folder")
        );
        let p = prompt(Task::Docs, "Methods \"and\" limits", "/research/my papers").unwrap();
        assert!(p.contains(&js("/research/my papers")));
        assert!(p.contains(&js("Methods \"and\" limits")));
    }

    #[test]
    fn research_markdown_yields_source_quotes_sections_stop_reason_and_uncertainty() {
        let a = parse_answer(
            "# Question\n\n## Findings\n\nA claim [1, 2].\n\n## Sources\n\n[1] Paper, https://example.org/paper?a=1&b=2, \"Exact quote.\"\n[2] https://example.net/page, \"Another quote.\"\n\n## Could not verify\n\n- \"Missing claim\": no saved copy.\n\n_Stopped: the page budget was spent._",
        );
        assert_eq!(a.sources.len(), 2);
        assert_eq!(a.sources[0].quote, "Exact quote.");
        assert_eq!(a.sources[0].url, "https://example.org/paper?a=1&b=2");
        assert_eq!(a.sections[1].title, "Findings");
        assert!(a.warnings[0].contains("no saved copy"));
        assert_eq!(a.stopped, "the page budget was spent");
    }

    #[test]
    fn the_chat_leaves_out_a_sources_list_the_side_panel_shows() {
        let web = "It has 325 hp [1].\n\nSources:\n[1] Cayman, https://example.org/c\n[2] Guide, https://example.net/g\n\n_Could not verify \"x\": gone._";
        assert_eq!(
            without_sources(web),
            "It has 325 hp [1].\n\n_Could not verify \"x\": gone._"
        );
        // The note that a citation named no page is the panel's alone (a demo, 24 Nov).
        let dropped = "It has 325 hp [1].\n\nSources:\n[1] Cayman, https://example.org/c\n\n_1 citation named no page and was dropped._\n\n_Could not verify \"x\": gone._";
        assert_eq!(
            without_sources(dropped),
            "It has 325 hp [1].\n\n_Could not verify \"x\": gone._"
        );
        let last = "A [1].\n\nSources:\n[1] One, https://example.org/a\n\n_2 citations named no page and were dropped._";
        assert_eq!(without_sources(last), "A [1].");
        assert!(
            (parse_answer(dropped).warnings.iter())
                .any(|w| w == "1 citation named no page and was dropped.")
        );
        let report = "# Q\n\nA claim [1].\n\n## Sources\n\n[1] Paper, https://example.org/p, \"Quote.\"\n\n## Could not verify\n\n- \"y\": no copy.";
        assert_eq!(
            without_sources(report),
            "# Q\n\nA claim [1].\n\n## Could not verify\n\n- \"y\": no copy."
        );
        // Kept: a source the panel would not show, a reused number, a fenced example, prose.
        for t in [
            "A.\n\nSources:\n[1] Bad, https://u:p@example.org",
            "A.\n\nSources:\n[1] One, https://example.org/a\n[1] Two, https://example.org/b",
            "```\nSources:\n[1] One, https://example.org/a\n```",
            "Sources:\nthe two pages agree.",
            "No list here.",
            // [1] names a different page in each list, so the panel shows neither.
            "A [1].\n\nSources:\n[1] First, https://example.org/a\n\n## More\nB [1].\n\nSources:\n[1] Second, https://example.org/b",
        ] {
            assert_eq!(without_sources(t), t);
        }
    }

    #[test]
    fn code_fences_and_unsafe_urls_do_not_become_evidence() {
        let t = "~~~md\n## Sources\n[1] Example, https://example.org, \"Not evidence.\"\n~~~\n\nSources:\n[1] Credentials, https://user:password@example.org, \"No.\"\n[2] Real, https://example.org, \"Quoted.\"";
        assert_eq!(parse_answer(t).sources.len(), 1);
        for u in [
            "javascript:alert(1)",
            "file:///etc/passwd",
            "data:text/html,test",
            "https://u:p@example.org",
        ] {
            assert_eq!(web_url(u), None);
        }
        assert_eq!(
            web_url("https://example.org/a").as_deref(),
            Some("https://example.org/a")
        );
    }

    #[test]
    fn paper_renderers_retain_individual_errors_partial_reads_and_unchecked_numbers() {
        let a = parse_answer(
            "## [1] first.pdf (key parts read)\nSummary.\nCheck these numbers, not found in the paper's text: 42\n\n## [2] second.pdf\nNot read: unreadable PDF\n",
        );
        assert_eq!(a.papers.len(), 2);
        assert!(a.papers[0].cut);
        assert_eq!(a.papers[0].file, "first.pdf");
        assert!(a.sections[a.papers[1].section].text.contains("Not read:"));
        assert_eq!(a.warnings.len(), 2);
    }

    #[test]
    fn document_citations_and_conflicts_survive_presentation_parsing() {
        let a = parse_answer(
            "See (notes.md:12) and /project/draft.txt:8.\n_Conflict: \"A\" [notes.md:12] against \"B\" [draft.txt:8]._",
        );
        assert!(a.docs.contains(&("notes.md".into(), 12)));
        assert!(a.docs.contains(&("/project/draft.txt".into(), 8)));
        assert_eq!(a.warnings.len(), 1);
    }

    #[test]
    fn overload_auth_and_loading_errors_have_specific_recovery_language() {
        assert!(http_message(401).contains("key"));
        assert!(http_message(429).contains("retry"));
        assert!(http_message(503).contains("loading"));
        assert!(http_message(400).contains("400"));
    }

    /// "+ Attach" attaches a PDF, Markdown or text file to the chat like any other file, for
    /// Folio to read; only images go to the message.
    /// Papers picked while the chat answers, or that the app refused for it, wait for the next
    /// message; another refusal is the user's to read (#591 review).
    #[test]
    fn papers_picked_while_the_chat_answers_wait_for_its_next_message() {
        assert!(!attach_waits(false, None));
        assert!(attach_waits(true, None));
        assert!(attach_waits(false, Some(larch_app::BUSY)));
        assert!(!attach_waits(
            false,
            Some("cannot attach /d/a.pdf: not found")
        ));
    }

    #[test]
    fn attached_papers_go_to_the_chat_and_images_to_the_message() {
        let picked = [
            "/d/paper.PDF",
            "/d/notes.md",
            "/d/a.txt",
            r"C:\Users\Local User\Documents\one.pdf",
            "/d/repo",
            "/d/shot.png",
        ];
        let (pictures, attached) = attach_split(picked.iter().map(PathBuf::from).collect());
        assert_eq!(pictures, [PathBuf::from("/d/shot.png")]);
        assert_eq!(attached, picked[..5]);
    }

    /// A paper request an older version saved shows as the question it asked, Windows paths and
    /// quotes kept.
    #[test]
    fn a_saved_paper_request_shows_as_its_question() {
        let paths: Vec<String> = vec![
            r"C:\Users\Local User\Documents\one.pdf".into(),
            r"C:\Users\Local User\Documents\two.pdf".into(),
        ];
        let req = format!(
            "Use read_papers with files {} and focus {}. Keep each paper separate and show any reading limits or errors.",
            serde_json::to_string(&paths).unwrap(),
            js("Compare \"methods\" and limitations")
        );
        assert_eq!(
            paper_message(&req),
            Some(("Compare \"methods\" and limitations".into(), paths))
        );
        assert_eq!(paper_message("ordinary chat"), None);
        assert_eq!(
            paper_message(r#"Use read_papers with files ["a.pdf"] and focus "bad\uQQQQ"."#),
            None
        );
    }

    #[test]
    fn research_requests_are_recognized_without_changing_ordinary_chat() {
        assert_eq!(
            research_mode("Can you research the history of this method?"),
            Some(Task::Research)
        );
        assert_eq!(
            research_mode("Please do deep research on this topic"),
            Some(Task::Research)
        );
        assert_eq!(
            research_mode("Look up the latest release notes"),
            Some(Task::Web)
        );
        assert_eq!(research_mode("What does the word research mean?"), None);
    }

    /// With files attached, a message's words never send it to research or the web; the chips
    /// still do (#591 review).
    #[test]
    fn a_chat_with_files_attached_is_not_routed_to_research_or_the_web() {
        let route = |chip, typed, files| send_task(chip, typed, false, files);
        let paper = "Investigate the methods in this paper";
        assert_eq!(route(None, paper, false), Some(Task::Research));
        assert_eq!(route(None, paper, true), None);
        assert_eq!(route(None, "Look up the lease's end date", true), None);
        assert_eq!(route(Some("Research"), paper, true), Some(Task::Research));
        assert_eq!(
            send_task(Some("Research"), paper, true, true),
            Some(Task::Docs)
        );
        assert_eq!(route(Some("Write"), paper, false), None);
        assert_eq!(route(None, "", false), None);
    }

    #[test]
    fn windows_connection_examples_use_powershell_quoting_and_environment_variables() {
        let model = r"C:\Users\Local User\Models\author's model.gguf";
        let s = snippets(&SnippetOptions {
            model,
            powershell: true,
            ..Default::default()
        })
        .unwrap();
        assert!(s.start.contains("author''s model.gguf"));
        assert!(s.powershell.contains("Invoke-RestMethod"));
        assert!(
            s.powershell
                .contains("'http://127.0.0.1:8080/v1/chat/completions'")
        );
        assert!(
            s.claude
                .contains("$env:ANTHROPIC_BASE_URL='http://127.0.0.1:8080'")
        );
        assert!(!s.claude.contains("export "));
        assert!(s.mcp_http.contains("author''s model.gguf"));
    }

    #[test]
    fn reused_citation_numbers_cannot_silently_link_to_a_different_source() {
        let a = parse_answer(
            "## Sources\n[1] One, https://example.org/a, \"A\"\n[1] Two, https://example.org/b, \"B\"\n[2] Three, https://example.org/c, \"C\"",
        );
        assert_eq!(a.sources.len(), 1);
        assert_eq!(a.sources[0].n, 2);
        assert!(a.warnings[0].contains("reused"));
    }

    #[test]
    fn a_receipt_line_shows_the_fact_without_its_said_and_says_why_an_undo_failed() {
        assert_eq!(
            fact_sentence("The user likes tea.\nSaid: \"I like tea a lot.\""),
            "The user likes tea."
        );
        assert_eq!(fact_sentence("The user likes tea."), "The user likes tea.");
        assert_eq!(undo_outcome(None), ("Undone".into(), false));
        assert_eq!(
            undo_outcome(Some(
                "the fact [tea] has changed since it was saved, so it was not undone"
            )),
            ("Not undone: it changed since it was saved".into(), false)
        );
        assert_eq!(
            undo_outcome(Some("memory: Resource temporarily unavailable")),
            (
                "Not undone: memory: Resource temporarily unavailable".into(),
                true
            )
        );
    }

    #[test]
    fn a_saved_reply_drops_a_trailing_call_but_keeps_one_named_in_prose() {
        assert_eq!(
            reply_words("Let me look.\n<tool_call>\n{\"name\": \"read_file\"}"),
            "Let me look."
        );
        assert_eq!(reply_words("<tool_call>"), "");
        // Qwen 3.5's XML body, inside the marker or bare as Qwen3-Coder writes it at times.
        let xml = "<function=web_answer>\n<parameter=question>\n2015 Cayman S price\n</parameter>\n</function>";
        assert_eq!(
            reply_words(&format!("Sure.\n<tool_call>\n{xml}\n</tool_call>")),
            "Sure."
        );
        assert_eq!(reply_words(&format!("<tool_call>\n{xml}")), "");
        assert_eq!(reply_words(&format!("Sure.\n{xml}")), "Sure.");
        let prose = "Wrap a call in <tool_call> tags.";
        assert_eq!(reply_words(prose), prose);
        let example = "Here is the syntax:\n```xml\n<tool_call>\n{\"name\": \"read_file\"}\n</tool_call>\n```\nThis is only an example.";
        assert_eq!(reply_words(example), example);
        // A bare XML body in a `~~~` block, or in a four-backtick block that holds a three-backtick
        // line, is an example too; a call after the block closes is dropped.
        let tilde =
            "Example:\n~~~xml\n<function=demo>\n<parameter=x>1</parameter>\n</function>\n~~~\nEnd.";
        assert_eq!(reply_words(tilde), tilde);
        let four = "Example:\n````md\n```\n<function=demo>\n</function>\n```\n````\nEnd.";
        assert_eq!(reply_words(four), four);
        assert_eq!(
            reply_words(&format!("Example:\n~~~\nx\n~~~\nNow.\n{xml}")),
            "Example:\n~~~\nx\n~~~\nNow."
        );
        assert_eq!(saved_step("one\n"), "one");
        assert_eq!(saved_step("- item\n- two\n"), "2 lines");
        assert!(
            stop_line("the context filled twice with nothing new in between; ended with the model's answer")
                .contains("ran out of room")
        );
        assert_eq!(
            stop_line("the user stopped the run"),
            "Stopped: the user stopped the run"
        );
    }

    #[test]
    fn steps_and_results_read_as_what_they_did() {
        let a = serde_json::json!({"path": "/home/me/src/calc.py", "start_line": 3, "end_line": 9});
        // A full path names the user's folders: the step shows the file's name alone, and a
        // relative one as it is.
        assert_eq!(
            describe("read_file", &a),
            ("Reading calc.py".into(), "Read calc.py, lines 3–9".into())
        );
        let relative = serde_json::json!({"path": "src/calc.py"});
        assert_eq!(describe("edit", &relative).1, "Edited src/calc.py");
        assert_eq!(
            describe(
                "fetch_url",
                &serde_json::json!({"url": "https://www.example.org/a"})
            )
            .0,
            "Opening example.org"
        );
        let papers = serde_json::json!({"papers": ["2401.04088", "a.pdf", "b.pdf"]});
        assert_eq!(describe("read_papers", &papers).0, "Reading 3 papers");
        assert_eq!(describe("my_tool", &Value::Null).1, "Ran my tool");
        // A file tool whose name says search is not the web.
        assert_eq!(
            describe("search_files", &serde_json::json!({"query": "fn main"})).1,
            "Searched files for “fn main”"
        );
        assert_eq!(
            describe("find_files", &serde_json::json!({"pattern": "*.rs"})).0,
            "Looking for *.rs"
        );
        assert_eq!(tidy_sum("list_dir", "1 lines"), "1 item");
        assert_eq!(tidy_sum("write_file", "wrote 120 bytes to /a"), "120 bytes");
        assert_eq!(
            tidy_sum("run_tests", "4 passed, 1 failed"),
            "1 failed, 4 passed"
        );
        assert_eq!(tidy_sum("run_tests", "4 passed, 0 failed"), "all 4 passed");
        assert_eq!(tidy_sum("search", "src/a.rs:1: x (12 lines)"), "12 matches");
        assert_eq!(tidy_sum("bash", "ok (3 lines)"), "ok");
    }

    #[test]
    fn markdown_blocks_and_spans() {
        let b = md_blocks(
            "# Title\n\nOne\ntwo\n\n- a\n- b\n  more\n\n```rust\nfn x() {}\n```\n1. first",
        );
        assert_eq!(b[0], Block::Heading(1, "Title".into()));
        assert_eq!(b[1], Block::Para(vec!["One".into(), "two".into()]));
        assert_eq!(
            b[2],
            Block::List {
                ordered: false,
                items: vec!["a".into(), "b more".into()]
            }
        );
        assert_eq!(
            b[3],
            Block::Code {
                lang: "rust".into(),
                body: "fn x() {}".into()
            }
        );
        assert!(matches!(&b[4], Block::List { ordered: true, .. }));
        assert_eq!(
            spans("a **b** `c` _d_ [e](http://x)"),
            vec![
                ("a ".into(), Span::Plain),
                ("b".into(), Span::Bold),
                (" ".into(), Span::Plain),
                ("c".into(), Span::Code),
                (" ".into(), Span::Plain),
                ("d".into(), Span::Italic),
                (" e".into(), Span::Plain),
            ]
        );
    }

    #[test]
    fn numbers_read_as_the_design_writes_them() {
        assert_eq!(gb(14_066_017_894), "13.1 GB");
        assert_eq!(fmt_k(2_100_000), "2.1M");
        assert_eq!(fmt_k(640_400), "640k");
        assert_eq!(fmt_v(0.85), "0.85");
        assert_eq!(fmt_v(1024.), "1024");
        assert_eq!(ago(0, 7_300_000), "2 h ago");
        assert_eq!(params("Qwen3-30B-A3B-Instruct"), Some(30.));
        assert_eq!(params("Llama-3.2-3B"), Some(3.));
        assert_eq!(params("bert-base"), None);
        assert_eq!(base_name(r"C:\m\a.gguf"), "a.gguf");
        assert_eq!(greeting(19), "Good evening");
    }

    /// The documents of the cite tests: one page of text each, in the order given.
    fn docs(texts: &[&[&str]]) -> Vec<larch_agent::folio::Doc> {
        use larch_agent::folio::{Rep, source};
        (texts.iter().enumerate())
            .map(|(i, pages)| {
                let text = pages.join("\u{c}");
                source::doc(i, &format!("/d/{i}.pdf"), "", Rep::Pdf, text)
            })
            .collect()
    }

    /// The chip follows the agent's reports: each state's words, the slower line only while the
    /// documents make room, and a follow-up's speed leaves the chip as it was.
    #[test]
    fn the_chip_follows_the_agents_reports() {
        use larch_agent::folio::Documents as D;
        use larch_app::contract::ChatEvent;
        let mut docs = super::Docs::default();
        let mut chip = |d: D| {
            let ChatEvent::Documents { state, text } = ChatEvent::from(d) else {
                unreachable!()
            };
            docs.apply(&state, &text);
            (
                docs.chip.clone().unwrap_or_default(),
                docs.slower,
                docs.first_word.clone(),
            )
        };
        let mut said = |d: D| {
            let (c, s, _) = chip(d);
            (c, s.is_some())
        };
        assert_eq!(
            said(D::MakingRoom { secs: 12 }),
            (
                "Document · Making room for the document (about 12 s)".into(),
                false
            )
        );
        assert_eq!(
            said(D::reading(1, 3, 120)),
            ("Document · Reading p. 40 of 120".into(), false)
        );
        assert_eq!(
            said(D::reading(1, 3, 0)),
            ("Document · Reading 33%".into(), false)
        );
        assert_eq!(
            said(D::reading(3, 3, 120)),
            ("Document · Ready".into(), false)
        );
        assert_eq!(said(D::ready(2)), ("Document · Ready".into(), true));
        assert_eq!(
            said(D::FitsUpTo { pages: 80 }),
            (
                "Document · Fits up to 80 pages on this computer".into(),
                false
            )
        );
        let answered = D::Answered {
            ttft_ms: 1200.,
            cached_tokens: 9,
        };
        let (c, _, first) = chip(answered);
        assert_eq!(
            (c.as_str(), first.as_deref()),
            (
                "Document · Fits up to 80 pages on this computer",
                Some("First word in 1.2 s")
            )
        );
        // A turn that ends keeps a finished chip and drops one still reading (#590 F13).
        chip(D::ready(0));
        docs.ended();
        assert_eq!(docs.chip.as_deref(), Some("Document · Ready"));
        let ChatEvent::Documents { state, text } = ChatEvent::from(D::reading(1, 3, 120)) else {
            unreachable!()
        };
        docs.apply(&state, &text);
        docs.ended();
        assert_eq!((docs.chip.as_deref(), docs.slower), (None, None));
        // The app let the pin go: the chip and the slower line go with it, the speed stays.
        let ChatEvent::Documents { state, text } = ChatEvent::from(D::ready(2)) else {
            unreachable!()
        };
        docs.apply(&state, &text);
        assert_eq!(docs.slower, Some(SLOWER));
        docs.apply(&DocState::Released, "Released");
        assert_eq!(
            (docs.chip.as_deref(), docs.slower, docs.busy),
            (None, None, false)
        );
        assert_eq!(docs.first_word.as_deref(), Some("First word in 1.2 s"));
    }

    /// `"quote" [p. 3]` and `"quote" [B, p. 12]` map to their page and the quote's bytes in it.
    #[test]
    fn a_cite_opens_its_page_with_the_quote_marked() {
        let one = docs(&[&[
            "Intro.",
            "Costs.",
            "Net income fell sharply in the quarter.",
        ]]);
        let got = cite_links(
            r#"It "fell sharply in the quarter" [p. 3]. Again "fell sharply in the quarter" [p. 3]."#,
            &one,
        );
        assert_eq!(got.len(), 1, "each cite once");
        let c = &got[0];
        assert_eq!(
            (c.label.as_str(), c.title.as_str(), c.missing),
            ("p. 3", "0.pdf, p. 3", false)
        );
        assert_eq!(
            &c.page[c.hit.clone().unwrap()],
            "fell sharply in the quarter"
        );

        let two = docs(&[&["A one."], &["B one.", "B two has the longer rate text."]]);
        let got = cite_links(r#""the longer rate text" [B, p. 2]"#, &two);
        let c = &got[0];
        assert_eq!(
            (c.label.as_str(), c.title.as_str()),
            ("B, p. 2", "1.pdf, p. 2")
        );
        assert_eq!(&c.page[c.hit.clone().unwrap()], "the longer rate text");
    }

    /// A quote that is not on its page opens where it is, or is marked when it is nowhere.
    #[test]
    fn a_cite_that_misses_is_marked_or_opens_where_the_quote_is() {
        let d = docs(&[&["First page.", "Second page has words in it."]]);
        let c = &cite_links(r#""page has words in it" [p. 1]"#, &d)[0];
        assert_eq!((c.title.as_str(), c.missing), ("0.pdf, p. 2", false));
        assert_eq!(&c.page[c.hit.clone().unwrap()], "page has words in it");
        let c = &cite_links(r#""no such thing" [p. 1]"#, &d)[0];
        assert_eq!(
            (
                c.label.as_str(),
                c.title.as_str(),
                c.missing,
                c.hit.is_none()
            ),
            ("p. 1: quote not found", "0.pdf, p. 1", true, true)
        );
    }

    #[test]
    fn a_shortcut_shows_its_letter_in_capitals() {
        assert_eq!(keys("M-n", "Ctrl"), "Ctrl+N");
        assert_eq!(keys("Shift-Enter", "Ctrl"), "Shift+Enter");
        assert_eq!(keys("M-,", "⌘"), "⌘+,");
    }
}
