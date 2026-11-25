//! The larch agent: the loop over one chat session, tool-call parsing, tools, file memory, the
//! audit log, parallel workers, the compiled skills and the benchmark, over the `larch` library.
//! Synchronous; only `web_search` and `fetch_page` use the network.

pub mod audit;
pub mod bench;
pub mod code;
pub mod config;
mod context;
pub mod folio;
pub mod mcp;
pub mod memory;
pub mod parse;
pub mod probe;
mod retrieve;
pub mod router;
mod run;
pub mod runtime;
/// The command sandbox.
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
pub mod sandbox;
mod signals;
pub mod skill;
pub mod tool;
/// The web: `web_search`, `fetch_page` and its extraction.
pub mod web {
    pub mod cards;
    pub mod extract;
    pub mod fetch;
    pub mod search;
}
pub mod workers;

/// Compiled skills: Rust fixes the steps, the model fills them in.
pub mod skills {
    pub mod ask_docs;
    pub mod doc;
    pub mod read_papers;
    pub mod research;
    pub mod web_answer;
}

mod tools {
    pub(crate) mod code;
    pub(crate) mod files;
    pub(crate) mod git;
    pub(crate) mod memory;
}

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, OpenOptions},
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError, atomic::AtomicBool},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use larch::{
    ChatMessage, EngineError, Larch, Role, Session, SessionOptions,
    chat::{Batcher, Metrics, Terminal, Transcript, transcript_path},
};
use tempfile::NamedTempFile;

pub use context::{is_turn, turns};
pub use run::{AgentEvent, AnswerSource, LOOPED, NO_VISION, Turn, knows_until, user_text};
pub use signals::NO_FOLDER;

use crate::runtime::inference::SeatAdmission;
use crate::{
    audit::Audit,
    parse::Family,
    tool::{Registry, Tool, ToolCtx, ToolSpec},
};

/// The tools every chat offers: the files and memory tools, ask_docs, web_answer, read_papers and
/// research.
pub const CHAT_TOOLS: [&str; 12] = [
    "list_dir",
    "read_file",
    "find_files",
    "search_files",
    "write_file",
    "recall",
    "remember",
    "forget",
    "ask_docs",
    "web_answer",
    "read_papers",
    "research",
];

/// The tools of `CHAT_TOOLS` that read or write the user's files: a chat with nothing attached
/// does not offer them, so the model cannot reach for them. read_papers stays: it opens a pasted
/// paper's URL, and with no folder allowed it reads no local path.
pub const FILE_TOOLS: [&str; 6] = [
    "list_dir",
    "read_file",
    "find_files",
    "search_files",
    "write_file",
    "ask_docs",
];

/// The tools a chat offers as well when config.toml's `git_tools` is true.
pub const GIT_TOOLS: [&str; 4] = ["changed_files", "diff_file", "show_file", "log"];

/// The tools code mode offers only in free mode (CP-5); the registry has them all the same.
pub const FREE_TOOLS: [&str; 1] = ["bash"];

/// Where an agent keeps its chat, what its tools may touch and the model's architecture; the data
/// folder's config.toml adds its settings.
#[derive(Clone, Debug)]
pub struct AgentOptions {
    /// The engine's data folder: `chat/<id>.jsonl` and `agent/`.
    pub data_dir: PathBuf,
    /// Folders and files the file tools may use, before config.toml's: the chat's attachments.
    /// With none, the chat offers no file tools.
    pub allowed: Vec<PathBuf>,
    /// The model's architecture, GGUF `general.architecture`.
    pub arch: String,
    /// Who decides between a tool call and a plain answer (the CLI passes config.toml's `router`).
    pub router: router::Router,
    /// Whether the chat anchors its system prompt in the prefix cache, at its first turn: the
    /// engine's cache keeps blocks (`PrefixCacheStats::capacity_blocks` above 0). Without blocks an
    /// anchor would only cost a prefill.
    pub prefix_cache: bool,
    /// Whether the chat captures facts in the background after each turn (`memory::capture`),
    /// when config.toml's `memory_capture` is on too. The app and `larch-agent chat` turn it on;
    /// a test's scripted model is off, as a capture would read its scripted replies.
    pub memory_capture: bool,
}

impl AgentOptions {
    /// The options the desktop app opens every chat with, which `bench --app` opens its chats with
    /// too, so a bench number is the app's: `allowed` as the app gives it (the home folder), the
    /// model deciding alone, and no prefix cache anchor.
    pub fn app(data_dir: PathBuf, allowed: Vec<PathBuf>, arch: String) -> AgentOptions {
        AgentOptions {
            data_dir,
            allowed,
            arch,
            // A chat opens with the model deciding alone (larch-agent's rule).
            router: router::Router::Off,
            // `Agent::open` does no GPU work: the chat anchors its prompt prefixes at its first
            // turn, in the turn worker, not under the state lock that `cancel` takes. The app
            // turns it off when the engine's cache keeps no blocks.
            prefix_cache: true,
            memory_capture: true,
        }
    }
}

/// Why an agent could not open a chat or take a turn.
#[derive(Debug)]
pub enum AgentError {
    /// Another agent in this process has the chat open.
    ChatBusy(String),
    /// The turn waits for `confirm`, so another one cannot start.
    Parked,
    /// `confirm` without a write waiting for it.
    NotParked,
    /// The session refused the history.
    Engine(EngineError),
    /// The transcript could not be created, read or written.
    Transcript(anyhow::Error),
    /// config.toml could not be read.
    Config(config::ConfigError),
    /// The router is not in this build.
    Router(String),
    /// The chat's documents could not be read or held.
    Documents(skill::SkillError),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChatBusy(id) => write!(f, "chat {id} is already open"),
            Self::Parked => f.write_str("the turn waits for a confirm"),
            Self::NotParked => f.write_str("no write waits for a confirm"),
            Self::Engine(e) => e.fmt(f),
            Self::Transcript(e) => write!(f, "transcript: {e:#}"),
            Self::Config(e) => e.fmt(f),
            Self::Router(e) => f.write_str(e),
            Self::Documents(skill::SkillError::Engine(EngineError::InvalidRequest(code)))
                if let Some(words) = documents_refused(*code) =>
            {
                f.write_str(words)
            }
            Self::Documents(e) => write!(f, "documents: {e}"),
        }
    }
}

impl std::error::Error for AgentError {}

/// What the user reads when the engine refuses to hold a chat's documents, in place of its code.
fn documents_refused(code: u16) -> Option<&'static str> {
    use larch::api::invalid;
    match code {
        invalid::DOC_TOO_LONG => Some(
            "The documents are too long for this computer to hold right now, so this answer does not read them whole.",
        ),
        invalid::PINS_FULL => Some(
            "Other chats' documents fill the memory, so this answer does not read these whole.",
        ),
        invalid::NO_PREFIX_CACHE => Some(
            "The prefix cache is off, so the documents cannot be kept ready; turn it on to pin them.",
        ),
        _ => None,
    }
}

/// The chats open in this process: one agent per chat. Another process is refused by the
/// transcript's own file lock.
static OPEN: Mutex<BTreeSet<PathBuf>> = Mutex::new(BTreeSet::new());

/// Outside tools one step shows: the phase 0 bench saw each extra tool make wrong picks more likely
/// and the first token later. A chat takes every usable one; retrieval chooses which are shown.
pub const MAX_OUTSIDE: usize = 24;
/// Their tool JSON one step shows, in characters (about a quarter as many tokens), so they leave
/// room for a step.
pub const OUTSIDE_CHARS: usize = 8000;

/// The BM25 score a server's best tool needs for a message to select it (toolcall-opt item 6,
/// tuned on the MCP dev rows). While a chat has selected none, every outside tool is shown, up to
/// the step's caps: one server alone never scores this high, as its words are in every document.
const RETRIEVE_FLOOR: f32 = 1.5;

/// `tools` as one step shows them: every built-in, and the outside tools in order while they fit
/// [`MAX_OUTSIDE`] and [`OUTSIDE_CHARS`].
fn capped(tools: &[&'static str], registry: &Registry) -> Vec<&'static str> {
    let (mut count, mut chars) = (0, 0);
    (tools.iter().copied())
        .filter(|t| {
            if !is_outside(t) {
                return true;
            }
            let size = (registry.get(t)).map_or(0, |t| t.spec().function().to_string().len());
            let fits = count < MAX_OUTSIDE && chars + size <= OUTSIDE_CHARS;
            if fits {
                (count, chars) = (count + 1, chars + size);
            }
            fits
        })
        .collect()
}

/// `order` with each tool's needs (`Tool::needs`) that `order` holds right after it, unless they
/// came earlier: a write shown without the lookup its server says to call first got IDs made up
/// by a small model (Notion's create-view, called with a data source named after the page).
fn with_needs(order: &[&'static str], registry: &Registry) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::with_capacity(order.len());
    for &t in order {
        if out.contains(&t) {
            continue;
        }
        out.push(t);
        let needs = registry.get(t).map(|x| x.needs()).unwrap_or_default();
        for &n in order {
            if needs.iter().any(|x| x == n) && !out.contains(&n) {
                out.push(n);
            }
        }
    }
    out
}

/// An outside tool's name as `'static`: the allow lists hold `&'static str`, and an outside
/// server's names arrive at run time. Each distinct name is kept once, for the process's life.
fn intern(name: &str) -> &'static str {
    static NAMES: Mutex<BTreeSet<&'static str>> = Mutex::new(BTreeSet::new());
    let mut names = NAMES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(n) = names.get(name) {
        return n;
    }
    let n: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.insert(n);
    n
}

/// Item 6: the most servers one message adds to a chat's shown tools.
const RETRIEVED: usize = 2;

/// Whether `name` is an outside tool's: `<server>__<tool>`, a shape no built-in has.
pub(crate) fn is_outside(name: &str) -> bool {
    name.contains("__")
}

/// Why no chat takes an outside tool, whatever else the chat holds ([`left_out`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeftOut {
    /// The call grammar refuses its arguments' schema: one such tool would leave every step of
    /// the chat unconstrained. The grammar's reason.
    Grammar(String),
    /// Its tool JSON is this many characters, over [`OUTSIDE_CHARS`]: no step could show it.
    Size(usize),
}

impl fmt::Display for LeftOut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LeftOut::Grammar(e) => write!(f, "the call grammar cannot follow its arguments ({e})"),
            LeftOut::Size(n) => write!(
                f,
                "its description is {n} characters, over the {OUTSIDE_CHARS} a step shows"
            ),
        }
    }
}

/// Why no chat takes the outside tool `spec`; `None` when one can. Its server's page says so
/// beside the tool, and the app leaves such tools out before a chat opens, so its chats never
/// start on a notice about them.
pub fn left_out(spec: &ToolSpec) -> Option<LeftOut> {
    // The JSON body's schema is checked alike in every family's format, so any family answers.
    let grammar = larch::toolcall::grammar(
        Family::Qwen3,
        &[spec.function()],
        false,
        larch::toolcall::Mode::Auto,
    );
    if let Err(e) = grammar {
        return Some(LeftOut::Grammar(e.to_string()));
    }
    let size = spec.function().to_string().len();
    (size > OUTSIDE_CHARS).then_some(LeftOut::Size(size))
}

/// The outside tools a chat takes, in order, and a notice for each group left out. All of them
/// are kept for retrieval to choose from; [`capped`] limits what one step shows.
fn usable_outside(
    outside: Vec<Box<dyn Tool>>,
    registry: &Registry,
) -> (Vec<Box<dyn Tool>>, Vec<String>) {
    let (mut kept, mut refused, mut over): (Vec<Box<dyn Tool>>, Vec<String>, usize) =
        (Vec::new(), Vec::new(), 0);
    for t in outside {
        let spec = t.spec();
        let clash = registry.get(&spec.name).is_some()
            || !is_outside(&spec.name)
            || kept.iter().any(|k| k.spec().name == spec.name);
        match (clash, left_out(&spec)) {
            (true, _) | (_, Some(LeftOut::Grammar(_))) => refused.push(spec.name),
            (_, Some(LeftOut::Size(_))) => over += 1,
            _ => kept.push(t),
        }
    }
    let mut notes = Vec::new();
    if !refused.is_empty() {
        notes.push(format!(
            "outside tools left out (a name taken, or a schema the call grammar cannot follow): {}",
            refused.join(", ")
        ));
    }
    if over > 0 {
        notes.push(format!(
            "{over} more outside tools are left out: each one's description is over the {OUTSIDE_CHARS} characters a step shows"
        ));
    }
    (kept, notes)
}

/// A chat's place in `OPEN`, given back on drop.
struct OpenChat(PathBuf);

/// `path` from the real path of its nearest folder that exists, so every spelling of it is one.
fn real(path: &Path) -> io::Result<PathBuf> {
    let path = std::path::absolute(path)?;
    let base = path.ancestors().find(|a| a.exists()).unwrap_or(&path);
    let rest = path.strip_prefix(base).expect("an ancestor is a prefix");
    Ok(fs::canonicalize(base)?.join(rest))
}

/// The prefix cache's namespace of a data folder, its real path hashed: the requests of two data
/// folders never share KV, and every spelling of one folder shares it.
pub fn namespace(data_dir: &Path) -> u64 {
    // A path that does not resolve keeps its spelling: fewer hits at worst, never a share.
    let path = real(data_dir).unwrap_or_else(|_| data_dir.into());
    let hash = larch::bench::sha256(path.as_os_str().as_encoded_bytes());
    u64::from_le_bytes(hash[..8].try_into().expect("a SHA-256 has 8 bytes"))
}

/// The least context a load drops to when the model does not fit at more (`load_fitting`): a
/// chat's turn and a code step's call still fit it.
pub const LEAST_CONTEXT: u32 = 2048;

/// `load` on `opts`, and if the model does not fit the device, again at half the context each
/// time, down to `LEAST_CONTEXT`: on a 4 GB GPU, a model past about 1.4 GB fits only with less
/// than the default's KV beside it. A refusal whose `need` the halving left as it was is not the
/// context's (the weights or the scratch do not fit), so it ends the tries. The loaded model and
/// the context it got.
pub fn load_fitting<T>(
    mut opts: larch::LoadOptions,
    mut load: impl FnMut(larch::LoadOptions) -> Result<T, larch::LoadError>,
) -> Result<(T, u32), larch::LoadError> {
    let mut needed = None;
    loop {
        match load(opts.clone()) {
            Err(larch::LoadError::DoesNotFit { need, .. })
                if opts.max_context > LEAST_CONTEXT && needed != Some(need) =>
            {
                needed = Some(need);
                opts.max_context = (opts.max_context / 2).max(LEAST_CONTEXT);
            }
            got => return got.map(|loaded| (loaded, opts.max_context)),
        }
    }
}

impl OpenChat {
    fn claim(path: &Path, id: &str) -> Result<OpenChat, AgentError> {
        let path = real(path).map_err(|e| AgentError::Transcript(e.into()))?;
        let mut open = OPEN.lock().unwrap_or_else(PoisonError::into_inner);
        if !open.insert(path.clone()) {
            return Err(AgentError::ChatBusy(id.into()));
        }
        Ok(OpenChat(path))
    }
}

impl Drop for OpenChat {
    fn drop(&mut self) {
        OPEN.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.0);
    }
}

/// One chat with the model and its tools: a session, its transcript and the audit log.
pub struct Agent {
    session: Session,
    /// The logprob gate's own session, for the one token it reads before a step
    /// (`run::first_token`), so the chat's session keeps its KV. It keeps KV itself, as only such
    /// a session reads the prefix cache.
    gate: Session,
    /// The session's options, with the system prompt it was built with.
    opts: SessionOptions,
    transcript: Transcript,
    chat_id: String,
    family: Option<Family>,
    /// Whether tool steps sample under the call grammar (config.toml's `tool_grammar`).
    tool_grammar: bool,
    /// The model's template asks for the XML tool-call body (`set_xml_body`).
    xml_body: bool,
    /// Shared with a running web_answer, whose search and fetches go through it.
    registry: Arc<Registry>,
    /// The tools the chat offers: `CHAT_TOOLS`, and `GIT_TOOLS` when config.toml turns them on;
    /// without the memory tools while memory is off.
    tools: Vec<&'static str>,
    /// `tools` with the memory tools.
    base_tools: Vec<&'static str>,
    /// The memory's `rev` the system prompt was built at.
    rev: u64,
    /// The memory block of the system prompt (`run::memory_block`), `None` while memory is off.
    memory_block: Option<String>,
    /// The documents pinned in the system prompt (`pin_documents`) and the system prompt they
    /// replaced: while set, every turn is a turn on the documents.
    pinned: Option<(Vec<folio::Doc>, String)>,
    /// The line naming a set too long to hold (`folio::large_note`), before the chat's own system
    /// text while no documents are pinned.
    large: Option<String>,
    /// The documents `folio::load` last left out, told once until the list changes.
    left_out: Vec<folio::LeftOut>,
    /// The chat's documents and the pin that holds them (`read_documents`).
    held: Option<folio::held::Held>,
    ctx: ToolCtx,
    audit: Audit,
    /// The system prompt's folders block and listing, built once: a compaction's prompt keeps
    /// them.
    folders: run::Folders,
    /// `max_context` of the loaded engine.
    max_context: u32,
    /// config.toml's thinking budget and answer cap.
    thinking: config::Thinking,
    /// Tokens of the facade's `THINKING_CLOSE`, which a budget cut feeds.
    close_tokens: u32,
    /// The user's turns so far, for a fact's source.
    turns: u32,
    /// The prefix cache's namespace of the data folder.
    namespace: u64,
    /// The chat said that a fact it saved shows from the next chat on.
    told_remember: bool,
    /// Where each `remember` this chat runs sends its receipt, for the app's Undo.
    receipts: Option<std::sync::mpsc::Sender<memory::Receipt>>,
    /// Said at the start of the next turn.
    notices: Vec<String>,
    run: run::Run,
    /// The key turns, when config.toml's `memory_keys` is on.
    keys: Option<Keys>,
    /// What the chat has read (`untrusted`, `read_results`).
    marks: run::Marks,
    /// Background capture, when it is on (`AgentOptions::memory_capture` and config.toml's).
    capture: Option<Capture>,
    /// [`Agent::stop_loops`].
    pub(crate) stop_loops: bool,
    /// The router this chat runs: the probe's turns to the model alone when it has no probe.
    router: router::Router,
    /// config.toml's `router_margin`.
    margin: f32,
    /// The tokens a call of the chat's family starts with (`router::starts`).
    starts: Vec<u32>,
    /// The probe of the model, with the `probe` router (`set_model_sha256`).
    probe: Option<probe::Probe>,
    /// The model's layers, which the probe's must be below.
    n_layers: u16,
    /// A turn without the facade (`turn`) said the probe cannot run there.
    told_probe: bool,
    /// The BM25 index of the outside tools by server and the score floor (`RETRIEVE_FLOOR`);
    /// `None` shows every tool.
    retrieval: Option<(retrieve::Index, f32)>,
    /// The servers retrieval has selected in this chat, least recently asked for first (a
    /// server stays selected, so the prompt prefix grows rather than changes), and the tools that
    /// shows this turn.
    selected: Vec<&'static str>,
    pub(crate) shown: Vec<&'static str>,
    /// Each outside tool's BM25 for the latest message, so a server's matching tools come first
    /// to the step's caps.
    matched: BTreeMap<&'static str, f32>,
    /// Whether the latest message selected no server (no name, pick or match at the floor):
    /// the step then shows the other servers' tools too, after the selected ones.
    unmatched: bool,
    /// config.toml's `probe_low` and `probe_high`.
    probe_low: Option<f32>,
    probe_high: Option<f32>,
    /// The folders' top-level names the system prompt lists.
    clues: tool::Clues,
    /// When the model's own knowledge ends (`run::knows_until`).
    knows_until: Option<&'static str>,
    /// The engine, which the anchors and ask_docs's skill go to.
    larch: Larch,
    /// The skill ask_docs runs and its workers, as config.toml sets them.
    docs: (skills::ask_docs::AskDocs, workers::Workers),
    /// The helpers of the skills this turn ran, as their workers last said (`helpers`).
    helpers: std::sync::Arc<std::sync::Mutex<Vec<workers::Helper>>>,
    /// The skill web_answer runs, as config.toml sets it; it runs on `docs`'s workers.
    web: skills::web_answer::WebAnswer,
    /// The skill read_papers runs, as config.toml sets it; it runs on `docs`'s workers.
    papers: skills::read_papers::Papers,
    /// The skill research runs, as config.toml sets it; it runs on `docs`'s workers.
    research: skills::research::Research,
    /// `AgentOptions::prefix_cache`: the chat anchors its prompt prefixes (`warm`).
    prefix_cache: bool,
    /// The chat's follow-up readers, kept between its questions; `None` while config.toml's
    /// `doc_followups` is off.
    readers: Option<Arc<Mutex<skills::doc::reader::Readers>>>,
    /// config.toml's `outside_pick`: a turn asks the per-server question (`router::fits`) before
    /// its first step.
    outside_pick: bool,
    /// The first turn is still to come, which anchors first.
    unanchored: bool,
    _open: OpenChat,
}

/// A chat's background capture: the user's message of the turn that runs, the messages still to
/// capture, and the thread reading the capture that runs.
#[derive(Default)]
struct Capture {
    /// Whether the model's `replaces` replace (`memory::capture::full_replaces`).
    replace: bool,
    turn: Option<String>,
    waiting: memory::capture::Waiting,
    running: Option<memory::capture::Started>,
}

/// The session the key turns run in, and the turn for the last daily note line while it runs:
/// the engine answers it while the user reads, and the next turn or the drop saves its keys.
struct Keys {
    session: Session,
    /// Read on a thread of its own, so the turn's seat frees when the engine is done, not when the
    /// next turn looks.
    pending: Option<(
        memory::Noted,
        std::thread::JoinHandle<Result<workers::Answer, EngineError>>,
    )>,
}

impl Agent {
    /// The helpers of the skills this turn ran (a research report's sections, the papers it
    /// reads), as their workers last said: a handle the app reads between a turn's events, as
    /// the turn holds the agent. Each turn starts it empty, a confirm's too.
    pub fn helpers(&self) -> std::sync::Arc<std::sync::Mutex<Vec<workers::Helper>>> {
        self.helpers.clone()
    }

    /// Empties the helper board for a new turn, so the last turn's (a cancelled one's still
    /// working) never show as this one's.
    fn new_board(&self) {
        self.helpers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// Ends a reply that gets stuck repeating one short piece (`larch::text::looping`: `^^^^`
    /// under an error, a run of fences), with a notice, instead of running to the token limit.
    /// The app turns it on; off by default, as a scripted model's replies repeat on purpose.
    pub fn stop_loops(&mut self, on: bool) {
        self.stop_loops = on;
    }

    /// Opens a new chat, saved to `chat/<id>.jsonl` after every step. It sends the engine nothing:
    /// its first turn puts its prompt prefixes in the prefix cache first (`anchor_once`).
    pub fn new(larch: &Larch, opts: AgentOptions) -> Result<Agent, AgentError> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        Agent::start(
            larch,
            format!("{nanos}-{}", std::process::id()),
            opts,
            false,
            true,
            Vec::new(),
        )
    }

    /// `new` with outside tools, as [`Agent::open_attached_with`] takes them (the bench's turn rows
    /// with `--extra-tools`).
    pub fn new_with(
        larch: &Larch,
        opts: AgentOptions,
        outside: Vec<Box<dyn Tool>>,
    ) -> Result<Agent, AgentError> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        Agent::start(
            larch,
            format!("{nanos}-{}", std::process::id()),
            opts,
            false,
            true,
            outside,
        )
    }

    /// Reopens a saved chat: its history under a freshly built system prompt (today's memory and
    /// date). The parked KV is gone, so the first turn prefills the history once.
    pub fn open(larch: &Larch, chat_id: &str, opts: AgentOptions) -> Result<Agent, AgentError> {
        Agent::start(larch, chat_id.into(), opts, true, true, Vec::new())
    }

    /// `open`, reading only `opts.allowed`, the folders and files attached to the chat:
    /// config.toml's `allowed` is the CLI's, not every app chat's.
    pub fn open_attached(
        larch: &Larch,
        chat_id: &str,
        opts: AgentOptions,
    ) -> Result<Agent, AgentError> {
        Agent::start(larch, chat_id.into(), opts, true, false, Vec::new())
    }

    /// `open_attached` with tools from outside the agent (an MCP server's, `mcp::client::McpTool`;
    /// the bench's stand-ins). Each joins the chat's list after the built-ins, unless its name is a
    /// built-in's or [`left_out`] says why none can take it; a step shows at most [`MAX_OUTSIDE`]
    /// of them, chosen by retrieval; and none join when a step showing that many would have no
    /// room. The chat's first notices name what was left out.
    pub fn open_attached_with(
        larch: &Larch,
        chat_id: &str,
        opts: AgentOptions,
        outside: Vec<Box<dyn Tool>>,
    ) -> Result<Agent, AgentError> {
        Agent::start(larch, chat_id.into(), opts, true, false, outside)
    }

    fn start(
        larch: &Larch,
        chat_id: String,
        opts: AgentOptions,
        resume: bool,
        configured: bool,
        outside: Vec<Box<dyn Tool>>,
    ) -> Result<Agent, AgentError> {
        // Before the first search needs them, so it doesn't wait on the certificate load.
        web::search::warm();
        opts.router.check().map_err(AgentError::Router)?;
        let path = transcript_path(&opts.data_dir, &chat_id).map_err(AgentError::Transcript)?;
        let config = config::load(&opts.data_dir).map_err(AgentError::Config)?;
        let open = OpenChat::claim(&path, &chat_id)?;
        let family = parse::family_for(&opts.arch, config.tool_format.as_deref());
        // config.toml's tool format is one of the known ones, or it failed to load.
        let notices = family.is_none().then(|| {
            format!(
                "this chat has no tools: the model's architecture '{}' has no known tool-call format",
                opts.arch
            )
        });
        let mut allowed = opts.allowed;
        if configured {
            allowed.extend(config.allowed);
        }
        let mut tools = CHAT_TOOLS.to_vec();
        if config.git_tools {
            tools.extend(GIT_TOOLS);
        }
        // A chat without tools reads and writes nothing, so it names and checks no folder.
        let mut folders = match family {
            Some(_) => run::folders(&allowed),
            None => run::Folders::default(),
        };
        // Nothing usable attached: the prompt says so, and the file tools are not shown.
        // A folder that is gone counts as none, so a question about the files gets `NO_FOLDER`.
        if family.is_some() && folders.block.is_empty() {
            tools.retain(|t| !FILE_TOOLS.contains(t));
            folders.block = run::NO_FOLDERS.into();
            allowed.clear();
        }
        let (unusable, clues) = (
            std::mem::take(&mut folders.notices),
            std::mem::take(&mut folders.clues),
        );
        let knows_until = run::knows_until(&opts.arch);
        let mut registry = Registry::builtin();
        if config.doc_followups {
            // Its spec says it answers follow-ups, which the room and memory fit count.
            registry.replace(Box::new(skills::read_papers::PapersTool {
                engine: None,
                seen: None,
                stop: Default::default(),
                show: None,
                followup: Some(Default::default()),
            }));
        }
        let mut outside_notes = Vec::new();
        if family.is_some() && !outside.is_empty() {
            let (kept, notes) = usable_outside(outside, &registry);
            outside_notes = notes;
            let names: Vec<_> = kept.iter().map(|t| intern(&t.spec().name)).collect();
            for t in kept {
                registry.replace(t);
            }
            let mut with = tools.clone();
            with.extend(&names);
            // Without the memory block, which shrinks to the room left: if the tools alone leave
            // no room for a step, the chat opens without them rather than not at all.
            let bare = run::system_prompt(
                &folders.block,
                None,
                &run::today(),
                knows_until,
                &folders.listing,
            );
            let most = capped(&with, &registry);
            match run::room(larch, &bare, &registry, &most, family, config.thinking) {
                Ok(()) => tools = with,
                Err(_) if !names.is_empty() => outside_notes.push(format!(
                    "the {} outside tools are left out of this chat: with them, a step would not fit the model's context",
                    names.len()
                )),
                Err(_) => {}
            }
        }
        // Read before the memory, so a change between the two shows at the first turn.
        let rev = memory::rev(&opts.data_dir);
        let base_tools = tools.clone();
        if !memory::enabled(&opts.data_dir) {
            tools.retain(|t| !run::MEMORY_TOOLS.contains(t));
        }
        let registry = Arc::new(registry);
        // A step shows at most the capped list, so the memory and the room are fitted to it.
        let most = capped(&tools, &registry);
        let shown: &[&str] = if family.is_some() { &most } else { &[] };
        let today = run::today();
        let around = |block: Option<&str>| {
            run::system_prompt(&folders.block, block, &today, knows_until, &folders.listing)
        };
        let memory = run::fitted_memory_block(
            larch,
            &opts.data_dir,
            &registry,
            shown,
            family,
            config.thinking,
            |block| around(Some(block)),
        );
        let system = around(memory.as_deref());
        // A setting whose step cannot fit the context would fail every turn, so the chat does not
        // open with it.
        // ask_docs runs the skill on this engine, with the settings `larch-agent run ask-docs`
        // takes from config.toml.
        let skill = skills::ask_docs::AskDocs {
            family,
            thinking: config.thinking,
            docs_workers: config.docs_workers,
            docs_window: config.docs_window,
            facts_grammar: config.tool_grammar,
        };
        let workers = workers::Workers {
            max_workers: config.max_workers,
            prefix_cache: opts.prefix_cache,
        };
        run::room(larch, &system, &registry, shown, family, config.thinking).map_err(|e| {
            AgentError::Config(config::ConfigError {
                path: config::path(&opts.data_dir),
                line: 0,
                reason: e.reason(e.key),
            })
        })?;
        let close_tokens =
            u32::try_from(larch.token_ids(larch::text::THINKING_CLOSE).len()).unwrap_or(u32::MAX);
        let notices: Vec<String> = notices
            .into_iter()
            .chain(unusable)
            .chain(outside_notes)
            .collect();
        // A chat without tools has no choice to route. The probe comes with `set_model_sha256`.
        // The options' router, or config.toml's when the options leave it off (the app's do):
        // the per-model default reaches the app this way (toolcall-opt items 2 and 3).
        let router = family.map_or(router::Router::Off, |_| match opts.router {
            router::Router::Off => config.router,
            r => r,
        });
        let starts = family.map_or_else(Vec::new, |f| {
            router::starts(f, |text| larch.token_ids(text))
        });
        let mut session_opts = SessionOptions {
            system: Some(system.clone()),
            ..SessionOptions::default()
        };
        let mut session = larch.session(session_opts.clone());
        let gate = larch.session(session_opts.clone());
        let mut turns = 0;
        let mut lost = Vec::new();
        let transcript = if resume {
            let (transcript, saved) = Transcript::resume(&path).map_err(AgentError::Transcript)?;
            let mut history = saved.history;
            if history.first().is_some_and(|m| m.role == Role::System) {
                history.remove(0);
            }
            // ponytail: counts the turns since the last compaction, so a fact's source repeats a
            // turn number after one; carry the count in the summary exchange when sources must be
            // unique
            turns = history.iter().filter(|m| context::is_turn(m)).count() as u32;
            history.insert(
                0,
                ChatMessage {
                    role: Role::System,
                    content: system,
                    ..ChatMessage::default()
                },
            );
            session_opts = SessionOptions {
                system: session_opts.system,
                ..saved.options
            };
            // Files an interrupted cleanup left (the journal dropped their exchanges first).
            if let Err(e) = larch::chat::prune_images(&path, &history) {
                lost.push(format!("image cleanup: {e:#}"));
            }
            // D10: a model without a projector reads this chat's images as words, and says so once.
            if !session.sees_images() && history.iter().any(|m| !m.images.is_empty()) {
                lost.push(
                    "this model can't see images: the chat's images reach it as the descriptions \
                     written for them, or as [image]"
                        .into(),
                );
            }
            session
                .restore(session_opts.clone(), history.clone())
                .map_err(AgentError::Engine)?;
            // The images the history names, read back with their notes (a stored note stays one);
            // one that cannot be read is a notice, and its message shows its note or [image].
            lost.extend(larch::chat::resume_images(&mut session, &path, &history));
            transcript
        } else {
            Transcript::create(&path, session_opts.clone(), session.history())
                .map_err(AgentError::Transcript)?
        };
        // A chat reopened on the documents plan its pin was held on (`folio::cache::plan`) routes
        // its documents against the context it had then.
        let loaded = loaded_plan(larch);
        let max_context = match folio::cache::plan(&folio::cache::dir(&opts.data_dir), &chat_id) {
            Some((plan, route)) if resume && plan == loaded => route,
            _ => loaded.max_context,
        };
        // The outside tools' BM25 index, so a message is shown the servers it matches (Qwen3
        // 1.7B went from 53.5% to 58.6% right on 99 MCP rows, toolcall-opt item 6). It holds
        // every tool the chat took, so retrieval reaches those past the step's caps too.
        let docs: Vec<(String, String, String)> = (tools.iter())
            .filter_map(|t| {
                let (server, tool) = t.split_once("__")?;
                let spec = registry.get(t)?.spec();
                let params: Vec<&str> = (spec.schema["properties"].as_object())
                    .map(|o| o.keys().map(String::as_str).collect())
                    .unwrap_or_default();
                let text = format!(
                    "{} {} {} {}",
                    server.replace('_', " "),
                    tool.replace('_', " "),
                    spec.description,
                    params.join(" ")
                );
                Some((server.to_owned(), (*t).to_owned(), text))
            })
            .collect();
        // Only a chat with outside tools has servers to select.
        let retrieval = (!docs.is_empty()).then(|| (retrieve::Index::new(&docs), RETRIEVE_FLOOR));
        let mut agent = Agent {
            session,
            gate,
            opts: session_opts,
            transcript,
            chat_id,
            family,
            tool_grammar: config.tool_grammar,
            xml_body: false,
            registry,
            tools,
            base_tools,
            rev,
            memory_block: memory,
            pinned: None,
            large: None,
            left_out: Vec::new(),
            held: None,
            ctx: ToolCtx {
                allowed,
                data_dir: opts.data_dir.clone(),
                source: String::new(),
            },
            audit: Audit::new(&opts.data_dir),
            folders,
            max_context,
            thinking: config.thinking,
            close_tokens,
            turns,
            namespace: namespace(&opts.data_dir),
            told_remember: false,
            receipts: None,
            notices: notices.into_iter().chain(lost).collect(),
            run: run::Run::default(),
            keys: config.memory_keys.then(|| Keys {
                session: larch.session(SessionOptions {
                    system: Some(memory::KEY_SYSTEM.into()),
                    keep_kv: false,
                    ..SessionOptions::default()
                }),
                pending: None,
            }),
            marks: run::Marks::default(),
            stop_loops: false,
            capture: (opts.memory_capture && config.memory_capture).then(|| Capture {
                replace: memory::capture::full_replaces(larch.deployment().params_b),
                ..Capture::default()
            }),
            router,
            margin: config.router_margin,
            starts,
            probe: None,
            n_layers: larch.deployment().n_layers,
            told_probe: false,
            retrieval,
            selected: Vec::new(),
            matched: BTreeMap::new(),
            unmatched: false,
            shown: Vec::new(),
            probe_low: config.probe_low,
            probe_high: config.probe_high,
            clues,
            knows_until,
            larch: larch.clone(),
            docs: (skill, workers),
            helpers: Default::default(),
            web: skills::web_answer::WebAnswer {
                docs: skill,
                pages: config.web_pages,
                reader: config.web_reader,
                snippets: skills::web_answer::snippet_tier(),
                one_page: skills::web_answer::one_page_tier(),
                park: skills::web_answer::park(),
                race: skills::web_answer::race(),
                short: skills::web_answer::short_answer(),
                read_thinking: skills::web_answer::read_thinking(larch.deployment().params_b),
                dates: skills::web_answer::page_dates(larch.deployment().params_b),
            },
            papers: skills::read_papers::Papers {
                family,
                paper_tokens: config.paper_tokens,
            },
            research: skills::research::Research {
                docs: skill,
                knows_until,
                ..Default::default()
            },
            prefix_cache: opts.prefix_cache,
            readers: config.doc_followups.then(|| {
                Arc::new(Mutex::new(skills::doc::reader::Readers::new(
                    skills::doc::reader::Settings {
                        family,
                        thinking: config.thinking,
                        followup_tokens: config.followup_tokens,
                        max_workers: config.max_workers,
                    },
                )))
            }),
            outside_pick: config.outside_pick,
            unanchored: true,
            _open: open,
        };
        agent.reselect();
        agent.refresh_shown();
        agent.settle_calls().map_err(AgentError::Engine)?;
        Ok(agent)
    }

    /// Before a turn: notes for the images its prompt would show as words and that have none
    /// (`Session::images_wanting_notes`), so older images shrink only when the room is needed and
    /// no turn waits on notes it never uses. A failure is a notice; the image then stays rows.
    fn note_images(&mut self, incoming: &[Arc<larch_model::vision::Image>]) {
        let want = self.session.images_wanting_notes(incoming);
        if want.is_empty() {
            return;
        }
        self.notices.push(format!(
            "describing {} earlier image{} so the chat stays within memory",
            want.len(),
            if want.len() == 1 { "" } else { "s" }
        ));
        let failed = self.describe(&want);
        self.notices.extend(failed);
    }

    /// Deletes the stored images the history no longer names (exchanges dropped or summarized
    /// away; `larch::chat::prune_images`). A failure is a notice: the files only wait for the
    /// next cleanup or the chat's deletion.
    /// The turn being held (a compaction runs before it, or a withdrawn step waits) keeps its own
    /// images, which are not in the history yet.
    pub(crate) fn prune_images(&mut self) {
        let mut history = self.session.history();
        if !self.run.images.is_empty() {
            history.push(ChatMessage {
                images: self.run.images.iter().map(|i| i.hex()).collect(),
                ..ChatMessage::default()
            });
        }
        let pruned = transcript_path(&self.ctx.data_dir, &self.chat_id)
            .and_then(|path| larch::chat::prune_images(&path, &history));
        if let Err(e) = pruned {
            self.notices.push(format!("image cleanup: {e:#}"));
        }
    }

    /// Writes a note for each image `refs` names (see `note_images`).
    fn describe(&mut self, refs: &[String]) -> Vec<String> {
        if !self.session.sees_images() || refs.is_empty() {
            return Vec::new();
        }
        let path = match transcript_path(&self.ctx.data_dir, &self.chat_id) {
            Ok(p) => p,
            Err(e) => return vec![format!("image notes: {e:#}")],
        };
        let mut failed = Vec::new();
        for r in refs {
            let p = larch::GenParams {
                temperature: Some(0.0),
                ..larch::GenParams::default()
            };
            let mut note = String::new();
            match self.session.describe_image(r, p) {
                Ok(stream) => {
                    for event in stream {
                        match event {
                            larch::TextEvent::Prefilled { .. }
                            | larch::TextEvent::Thinking { .. } => {}
                            larch::TextEvent::Text { delta, .. } => note.push_str(&delta),
                            larch::TextEvent::Error(e) => {
                                failed.push(format!("image note: {e:?}"));
                                note.clear();
                            }
                            larch::TextEvent::Done { .. } => {}
                        }
                    }
                }
                Err(e) => {
                    failed.push(format!("image note: {e}"));
                    continue;
                }
            }
            let note = note.trim();
            if note.is_empty() {
                continue;
            }
            self.session.set_image_note(r, note);
            if let Err(e) = larch::chat::store_note(&path, r, note) {
                failed.push(format!("image note: {e:#}"));
            }
        }
        failed
    }

    /// The chat's ID: its transcript is `chat/<id>.jsonl` in the data folder.
    pub fn chat_id(&self) -> &str {
        &self.chat_id
    }

    /// The history the model sees, system prompt first.
    pub fn history(&self) -> Vec<ChatMessage> {
        self.session.history()
    }

    /// Where the last turn's answer came from, once its `Done` is out.
    pub fn source(&self) -> AnswerSource {
        self.run.source
    }

    /// Waits until this chat's background memory capture, if one runs, has written what it
    /// saves, so a caller that reads the memory next (the memory chat bench) sees it; a message
    /// whose capture reached its token cap is read again alone first, and waited for too.
    pub fn wait_for_capture(&mut self) {
        loop {
            self.join_capture();
            let alone = self.capture.as_ref().is_some_and(|c| {
                !(c.waiting.lock().unwrap_or_else(PoisonError::into_inner))
                    .alone
                    .is_empty()
            });
            if !alone || !self.capture_next(None) {
                return;
            }
        }
    }

    /// Waits for the capture that runs, if any: at a turn end the turn's start has cancelled it,
    /// so this is quick.
    fn join_capture(&mut self) {
        let running = self.capture.as_mut().and_then(|c| c.running.take());
        if let Some(Err(_)) = running.map(|r| r.thread.join()) {
            self.notices
                .push("the memory capture stopped with a panic".into());
        }
    }

    /// Starts the capture of the user messages not yet captured, this turn's last, at the end of
    /// a turn however it ended: a failed turn's user still said what they said. The capture
    /// before it has ended by now (the turn's start cancelled it), and a cancelled one left its
    /// messages to this one. A message cut at the token cap goes first, alone.
    pub(crate) fn start_capture(&mut self) {
        self.join_capture();
        let turn = self.capture.as_mut().and_then(|c| c.turn.take());
        self.capture_next(turn);
    }

    /// Adds `turn` to the waiting messages and starts the next capture: a message to read again
    /// alone if one waits, else the waiting messages. Whether one started.
    fn capture_next(&mut self, turn: Option<String>) -> bool {
        // A reply after a file, a page or a doc can repeat words that ask for a save; the capture
        // never sees it then, as a `remember` asks then.
        let read = self.capture.is_some() && (self.read_results() || self.untrusted());
        let Some(capture) = &mut self.capture else {
            return false;
        };
        let (messages, retry) = {
            let mut waiting = capture
                .waiting
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            // Memory off: what is said now is never captured, also once it is on again.
            if !memory::enabled(&self.ctx.data_dir) {
                *waiting = Default::default();
                return false;
            }
            // A message of 1 or 2 words ("yes", "ok thanks") states no fact of its own: what it
            // agrees to is the reply's.
            let short = |t: &String| memory::capture::words(t).len() < memory::capture::QUOTE_WORDS;
            waiting.messages.extend(turn.filter(|t| !short(t)));
            let over = (waiting.messages.len()).saturating_sub(memory::capture::MAX_MESSAGES);
            waiting.messages.drain(..over);
            match waiting.alone.pop_front() {
                Some(alone) => (vec![alone], true),
                None => (waiting.messages.clone(), false),
            }
        };
        if messages.is_empty() {
            return false;
        }
        let history = self.session.history();
        let reply = match read {
            true => "",
            false => (history.iter().rev())
                .find(|m| m.role == Role::Assistant)
                .map_or("", |m| m.content.as_str()),
        };
        let facts: Vec<(String, String)> = match memory::list(&self.ctx.data_dir) {
            Ok(facts) => facts.into_iter().map(|f| (f.slug, f.text)).collect(),
            Err(e) => {
                self.notices.push(format!("memory capture: {e}"));
                return false;
            }
        };
        let shown = memory::capture::shown(&self.ctx.data_dir, &facts, &messages);
        let job = memory::capture::Job {
            data_dir: self.ctx.data_dir.clone(),
            chat: self.chat_id.clone(),
            source: self.ctx.source.clone(),
            messages,
            shown: Vec::new(),
            waiting: Arc::clone(&capture.waiting),
            receipts: self.receipts.clone(),
            retry,
            replace: capture.replace,
        };
        match memory::capture::start(&self.larch, self.family, reply, &shown, job) {
            Ok(started) => {
                capture.running = Some(started);
                true
            }
            Err(e) => {
                self.notices.push(format!("memory capture: {e}"));
                false
            }
        }
    }

    /// Sends a receipt (`memory::Receipt`) for each fact this chat's `remember` saves, so the app
    /// can show it with Undo (`memory::undo`). A setter, as the app builds `AgentOptions` as a
    /// literal.
    pub fn set_memory_receipts(&mut self, to: std::sync::mpsc::Sender<memory::Receipt>) {
        self.receipts = Some(to);
    }

    /// Adds `user` and `answer` to the chat as one exchange, saved as a turn's is, with no step
    /// of the model: what a code run in the chat was asked and did, which the next turns read. A
    /// chat whose turn waits for `confirm` takes none.
    pub fn add_exchange(&mut self, user: &str, answer: &str) -> Result<(), AgentError> {
        if self.run.parked.is_some() {
            return Err(AgentError::Parked);
        }
        let said = |role, content: &str| ChatMessage {
            role,
            content: content.into(),
            ..ChatMessage::default()
        };
        let exchange = [said(Role::User, user), said(Role::Assistant, answer)];
        let mut history = self.session.history();
        history.extend_from_slice(&exchange);
        // No tokens: the model took no step. Saved first, as the transcript refuses what would
        // not replay.
        let metrics = Metrics {
            prompt_tokens: 0,
            completion_tokens: 0,
            cached_tokens: 0,
            ttft_ms: 0.0,
            tpot_ms: 0.0,
            tok_s: 0.0,
        };
        let saved = self
            .transcript
            .append_turn(Terminal::Eos, Some(metrics), 0, Some(exchange));
        saved.map_err(AgentError::Transcript)?;
        (self.session.restore(self.opts.clone(), history)).map_err(AgentError::Engine)
    }

    /// The tool steps' grammar takes the XML call body, as the model's chat template asks
    /// (`toolcall::xml_body`); a new agent takes the JSON body.
    pub fn set_xml_body(&mut self, on: bool) {
        self.xml_body = on;
    }

    /// Names the model by its file's SHA-256 in hex, which names its probe file (`probe::path`),
    /// and reads the probe the `probe` router decides with. Without it, or with no usable file,
    /// the next turn says so and the model decides alone. Other routers ignore it.
    // ponytail: a setter, not an AgentOptions field, because larch-app builds AgentOptions as a
    // literal; move it into the options once they have a Default
    pub fn set_model_sha256(&mut self, sha256: Option<String>) {
        if self.router != router::Router::Probe {
            return;
        }
        let Some(sha) = sha256 else {
            self.probe = None;
            return;
        };
        let path = probe::path(&self.ctx.data_dir, &sha);
        match probe::Probe::load(&path, self.n_layers) {
            Ok(probe) => self.probe = Some(probe),
            Err(e) => self.no_probe(&e.to_string()),
        }
    }

    /// Says why the probe router has no probe, once, and leaves its turns to the model: the
    /// logprob gate lost to the model alone on the 3050 (toolcall-opt item 2, Qwen3 1.7B T set:
    /// 79.4% against 89.0%), so it is no fallback.
    fn no_probe(&mut self, why: &str) {
        self.probe = None;
        self.router = router::Router::Off;
        self.notices.push(format!(
            "router probe: no usable probe for this model ({why}); the model decides alone. \
             `larch-agent probe train` makes one"
        ));
    }

    /// The router is `probe`, which reads the model's SHA-256 (`set_model_sha256`): a caller
    /// hashes the model file only then.
    pub fn wants_model_sha256(&self) -> bool {
        self.router == router::Router::Probe
    }

    /// Starts a turn with the user's text; its events end at `Done`, at a failure, or at
    /// `NeedsConfirm`, which `confirm` answers.
    pub fn turn(&mut self, user: &str) -> Result<Turn<'_>, AgentError> {
        self.note_images(&[]);
        self.start_turn(None, user, None, Vec::new())
    }

    /// `turn_for` with the user's images (design: image paste). They are stored beside the chat's
    /// transcript first, then go with the user's message; a model without a projector answers that
    /// it cannot see them. A message of images alone is a chat turn (no web skill).
    pub fn turn_images(
        &mut self,
        user: &str,
        images: Vec<Arc<larch_model::vision::Image>>,
        intent: Option<router::Intent>,
    ) -> Result<Turn<'_>, AgentError> {
        // A refused turn writes no files: the refusal `start_turn` makes first is checked before the
        // images are stored beside the transcript.
        if self.run.parked.is_some() {
            return Err(AgentError::Parked);
        }
        if !images.is_empty() && self.session.sees_images() {
            let path = transcript_path(&self.ctx.data_dir, &self.chat_id)
                .map_err(AgentError::Transcript)?;
            for image in &images {
                larch::chat::store_image(&path, image).map_err(AgentError::Transcript)?;
            }
        }
        let intent = match intent {
            None if !images.is_empty() && user.trim().is_empty() => Some(router::Intent::Chat),
            i => i,
        };
        self.note_images(&images);
        self.start_turn(None, user, intent, images)
    }

    /// `turn`, held to the mode the user picked (`router::Intent`): code, and a change, which
    /// reaches a turn only when no checkout runs it, are answered with no tool; chat may call no
    /// web skill; web is `turn`.
    pub fn turn_for(&mut self, user: &str, intent: router::Intent) -> Result<Turn<'_>, AgentError> {
        self.note_images(&[]);
        self.start_turn(None, user, Some(intent), Vec::new())
    }

    /// Selects for `user` the servers it names or the turn picked (`Run::picked`), and up to two whose best tool
    /// scores at least the floor, then sets `shown` (`refresh_shown`). A server asked for again
    /// moves to the end, as the most recent; a named one ends after a retrieved one. Without
    /// retrieval, nothing.
    fn select_servers(&mut self, user: &str) {
        // Retrieval would change the shown list per message, and so the pinned prefix.
        if self.pinned.is_some() {
            return;
        }
        let Some((index, floor)) = &self.retrieval else {
            return;
        };
        self.matched = (index.tools(user).into_iter())
            .filter_map(|(name, score)| Some((*self.tools.iter().find(|t| **t == name)?, score)))
            .collect();
        let servers = self.outside_servers();
        // Least important first: the retrieved (the best last), then the picked, then the named.
        let mut add: Vec<&'static str> = Vec::new();
        let retrieved = (index.servers(user).into_iter().take(RETRIEVED))
            .filter(|(_, score)| score >= floor)
            .filter_map(|(server, _)| servers.iter().find(|(s, _)| *s == server).map(|(s, _)| *s));
        add.extend(retrieved.collect::<Vec<_>>().into_iter().rev());
        // The server the turn picked for this message (config.toml `outside_pick`) too.
        if let Some(prefix) = self.run.picked.as_ref() {
            add.extend(
                (servers.iter())
                    .map(|(s, _)| *s)
                    .filter(|s| *prefix == format!("{s}__")),
            );
        }
        if let Some(n) = self.named_outside(user) {
            add.extend(
                (servers.iter())
                    .map(|(s, _)| *s)
                    .filter(|s| n.prefix.is_empty() || n.prefix == format!("{s}__")),
            );
        }
        self.unmatched = add.is_empty();
        for s in add {
            self.touch(s);
        }
        self.refresh_shown();
    }

    /// Makes `server` the chat's most recently selected one.
    fn touch(&mut self, server: &'static str) {
        self.selected.retain(|s| *s != server);
        self.selected.push(server);
    }

    /// Selects again the outside servers a reopened chat's history called, the latest call's
    /// last, and puts the tools it called first among their servers' (as a message's matches
    /// are): a call parked at its confirm resolves against them, past the step's caps too (a
    /// server's 31st tool), rather than being dropped as a tool the chat no longer has.
    fn reselect(&mut self) {
        if self.retrieval.is_none() {
            return;
        }
        let history = self.session.history();
        for call in history.iter().flat_map(|m| &m.tool_calls) {
            let Some(&tool) = self.tools.iter().find(|t| **t == call.name) else {
                continue;
            };
            let Some((server, _)) = tool.split_once("__") else {
                continue;
            };
            self.touch(server);
            self.matched.insert(tool, f32::MAX);
        }
    }

    /// Sets `shown` from the chat's tools: the built-ins and the selected servers' tools, or every
    /// outside tool while the chat has selected none or the latest message selected none (a
    /// message under the floor still reaches them, a change of topic too), up to the step's caps,
    /// shown in the chat's order. To the caps the most recently selected server comes first, then
    /// the other selected ones, then the unselected; within a server the tools the latest message
    /// matches come first, each followed by the tools it needs (`with_needs`), so neither the
    /// server the user asked for last nor a match past a big server's 24th tool, nor the lookup
    /// that match needs, is the one they cut. Run at open too, so a reopened chat's first step (a
    /// parked call's confirm) has the built-ins.
    fn refresh_shown(&mut self) {
        fn server(t: &str) -> Option<&str> {
            t.split_once("__").map(|(s, _)| s)
        }
        let builtins = self.tools.iter().copied().filter(|t| server(t).is_none());
        let mut outside: Vec<&'static str> = match self.selected.is_empty() || self.unmatched {
            true => self
                .tools
                .iter()
                .copied()
                .filter(|t| server(t).is_some())
                .collect(),
            false => (self.selected.iter().rev())
                .flat_map(|s| (self.tools.iter().copied()).filter(move |t| server(t) == Some(*s)))
                .collect(),
        };
        // A stable sort: by server as above, then by this message's match, then the chat's order.
        let rank = |t: &&'static str| {
            (self.selected.iter().rev())
                .position(|s| server(t) == Some(*s))
                .unwrap_or(usize::MAX)
        };
        let matched = |t: &&'static str| self.matched.get(t).copied().unwrap_or(0.0);
        outside.sort_by(|a, b| {
            rank(a)
                .cmp(&rank(b))
                .then(matched(b).total_cmp(&matched(a)))
        });
        let outside = with_needs(&outside, &self.registry);
        let order: Vec<&'static str> = builtins.chain(outside).collect();
        let keep = capped(&order, &self.registry);
        self.shown = (self.tools.iter().copied())
            .filter(|t| keep.contains(t))
            .collect();
    }

    /// The first outside server of this chat whose tools the model reads `user` as asking for
    /// (`router::fits`), as its `<server>__` prefix; each question is in the audit log.
    fn pick_outside(&mut self, user: &str) -> Option<String> {
        self.family?;
        for (server, tools) in self.outside_servers() {
            let started = Instant::now();
            let fits = router::fits(
                &self.larch,
                self.family,
                self.namespace,
                self.prefix_cache,
                server,
                &tools,
                user,
            );
            let event = audit::AuditEvent {
                session: self.session.id(),
                chat: Some(self.chat_id.clone()),
                kind: audit::EventKind::Step,
                step: Some("outside pick".into()),
                wall_ms: Some(started.elapsed().as_secs_f64() * 1e3),
                detail: Some(serde_json::json!({
                    "server": server,
                    "fits": fits.as_ref().is_ok_and(|f| *f),
                })),
                error: fits.as_ref().err().map(ToString::to_string),
                ..audit::AuditEvent::default()
            };
            if let Err(e) = self.audit.append(event) {
                self.notices.push(format!("audit log: {e}"));
            }
            if matches!(fits, Ok(true)) {
                return Some(format!("{server}__"));
            }
        }
        None
    }

    /// `turn`, with the facade the probe router reads the prompt's hidden state through.
    // ponytail: the facade comes with each turn because `Larch` is not Clone (#181); keep one in
    // the agent and drop this once it is
    pub fn turn_with(&mut self, larch: &Larch, user: &str) -> Result<Turn<'_>, AgentError> {
        self.note_images(&[]);
        self.start_turn(Some(larch), user, None, Vec::new())
    }

    fn start_turn(
        &mut self,
        larch: Option<&Larch>,
        user: &str,
        intent: Option<router::Intent>,
        images: Vec<Arc<larch_model::vision::Image>>,
    ) -> Result<Turn<'_>, AgentError> {
        self.new_board();
        // The agent's own facade is the same engine: a caller without one (the app's
        // `turn_for`) still gets the probe router.
        let own = self.larch.clone();
        let larch = Some(larch.unwrap_or(&own));
        memory::capture::cancel_all();
        if self.run.parked.is_some() {
            return Err(AgentError::Parked);
        }
        self.finish_keys();
        self.reread_memory()?;
        // A continuation's session refused an overflow; the chat drops old exchanges again.
        if self.run.strict {
            self.session
                .restore(self.opts.clone(), self.session.history())
                .map_err(AgentError::Engine)?;
        }
        self.turns += 1;
        let cut = self.run.cut.take();
        self.run = run::Run::default();
        self.run.images = images;
        self.ctx.source = format!("chat {} turn {}", self.chat_id, self.turns);
        // A message with images is never a continuation (`Turn::start` agrees): it routes.
        let continues = cut.is_some()
            && self.run.images.is_empty()
            && user.trim().eq_ignore_ascii_case("continue");
        let on = memory::enabled(&self.ctx.data_dir);
        if let Some(capture) = self.capture.as_mut().filter(|_| !continues && on) {
            capture.turn = Some(user.to_owned());
        }
        // "continue" goes on with an answer: there is no tool choice to make.
        if !continues {
            self.route(larch, user);
            // Each outside server is asked whether the message is for its tools, unless the
            // user picked a mode or the message names one; the first that fits holds the turn's
            // first step to a call of them, as a message that names the server does. A picked
            // mode (the Code chip's no tools, say) is the user's and no automatic pick lifts it.
            let ask = self.outside_pick && intent.is_none() && self.named_outside(user).is_none();
            self.run.picked = if ask { self.pick_outside(user) } else { None };
            self.select_servers(user);
            // A message that names an outside server, or that the turn picked one for, or that
            // follows a turn that called one, keeps the tools: it reads as chat, whatever the
            // mode.
            let outside = || {
                self.run.picked.is_some()
                    || self.named_outside(user).is_some()
                    || self.outside_before()
            };
            let intent = match intent {
                Some(router::Intent::Code | router::Intent::Change) if outside() => {
                    Some(router::Intent::Chat)
                }
                i => i,
            };
            // In a chat that read documents, a question the router takes for code is about them
            // as often (plan 7.8): it keeps read_papers, as chat does, and a set too long to hold
            // keeps ask_docs.
            let read = || {
                let calls = |m: &ChatMessage| {
                    (m.tool_calls.iter()).any(|c| c.name == skills::read_papers::TOOL)
                };
                self.large.is_some()
                    || (self.readers.is_some() && self.session.history().iter().any(calls))
            };
            match intent {
                // A change reaches a turn only without a checkout to change: it is answered as
                // code is.
                Some(router::Intent::Code | router::Intent::Change) if !read() => {
                    (self.run.masked, self.run.gate) = (true, false);
                }
                // A run is answered as chat, which reads the files: the app starts no run for it.
                Some(
                    router::Intent::Chat
                    | router::Intent::Run
                    | router::Intent::Code
                    | router::Intent::Change,
                ) => self.run.no_web = true,
                Some(router::Intent::Web) | None => {}
            }
        }
        // The gate reads a text-only side session: an image turn is not its to decide. An image
        // turn offers no web skill unless the user's own words ask for the web ("look it up"):
        // text in a picture must not send the chat's data to a URL it names.
        if !self.run.images.is_empty() {
            self.run.gate = false;
            if !signals::asks_web(user) {
                self.run.no_web = true;
            }
        }
        let notices = std::mem::take(&mut self.notices);
        Ok(Turn::start(self, notices, user, cut))
    }

    /// Brings a delete, an edit or an on/off made since the system prompt was built (in another
    /// chat, the app or this chat) into it, at a turn start: only when `memory::rev` moved, and
    /// then only when the memory block or the tools came out different. The history is kept
    /// whole; the cost is one prefill of the chat, which the anchor starts. A turn that runs in
    /// another chat goes on as it is.
    fn reread_memory(&mut self) -> Result<(), AgentError> {
        let data = &self.ctx.data_dir;
        let rev = memory::rev(data);
        if rev == self.rev {
            return Ok(());
        }
        let on = memory::enabled(data);
        let tools: Vec<&'static str> = (self.base_tools.iter().copied())
            .filter(|t| on || !run::MEMORY_TOOLS.contains(t))
            .collect();
        let memory = self.fitted_memory_block(&capped(&tools, &self.registry));
        if memory == self.memory_block && tools == self.tools {
            self.rev = rev;
            return Ok(());
        }
        // The rev, block and tools are the chat's only once the restore took them: a failed one
        // is tried again at the next turn.
        if memory != self.memory_block {
            let plain = run::system_prompt(
                &self.folders.block,
                memory.as_deref(),
                &run::today(),
                self.knows_until,
                &self.folders.listing,
            );
            let system = self.framed(plain);
            let opts = SessionOptions {
                system: Some(system.clone()),
                ..self.opts.clone()
            };
            let mut history = self.session.history();
            match history.first_mut() {
                Some(first) if first.role == Role::System => first.content = system,
                _ => history.insert(
                    0,
                    ChatMessage {
                        role: Role::System,
                        content: system,
                        ..ChatMessage::default()
                    },
                ),
            }
            self.session
                .restore(opts.clone(), history)
                .map_err(AgentError::Engine)?;
            self.opts = opts;
            self.memory_block = memory;
        }
        (self.rev, self.tools) = (rev, tools);
        self.refresh_shown();
        // A chat not yet anchored anchors at this turn anyway.
        if !self.unanchored {
            let failed = self.warm();
            self.notices.extend(failed);
        }
        Ok(())
    }

    /// Sets how the router decides this turn: the probe's verdict before any token, the gate on
    /// the first token when the router is `logprob`, or the model alone when the probe cannot run.
    fn route(&mut self, larch: Option<&Larch>, user: &str) {
        if self.router == router::Router::Probe && self.probe.is_none() {
            self.no_probe("the model's SHA-256 is not known");
        }
        self.run.gate = self.router == router::Router::Logprob;
        let Some(probe) = self
            .probe
            .as_ref()
            .filter(|_| self.router == router::Router::Probe)
        else {
            return;
        };
        let Some(larch) = larch else {
            if !std::mem::replace(&mut self.told_probe, true) {
                self.notices.push(
                    "router probe: this caller starts turns without the engine (Agent::turn), so \
                     the model decides alone"
                        .into(),
                );
            }
            return;
        };
        let started = Instant::now();
        let system = self.opts.system.as_deref().unwrap_or_default();
        let threshold = probe.threshold;
        let scored = match probe::features(larch, probe::prompt(system, user), &probe.layers) {
            // An engine failure passes: the model decides this turn, and the next tries the probe.
            Err(e) => Err((e.to_string(), false)),
            // A probe of another width than the model's features never fits: it is done for good.
            Ok(x) => probe.score(&x).map_err(|e| (e, true)),
        };
        let (detail, error) = match scored {
            Ok(p) => {
                // At or under `low` no tools, at or over `high` a held call, else the model's
                // choice; `LARCH_EXP_PROBE_DECISION` (auto, force or mask) takes one of the three
                // for every row, so a bench run records each row's outcome under it with its p.
                let low = self.probe_low;
                let high = self.probe_high.unwrap_or(f32::INFINITY);
                let decision = match std::env::var("LARCH_EXP_PROBE_DECISION").as_deref() {
                    Ok("auto") => "auto",
                    Ok("force") => "force",
                    Ok("mask") => "mask",
                    _ if low.map_or(p < threshold, |l| p <= l) => "mask",
                    _ if p >= high => "force",
                    _ => "auto",
                };
                self.run.masked = decision == "mask";
                self.run.probe_force = decision == "force";
                let tool = !self.run.masked;
                (
                    Some(serde_json::json!({"probability": p, "tool": tool, "decision": decision})),
                    None,
                )
            }
            Err((e, true)) => {
                self.no_probe(&e);
                (None, Some(e))
            }
            Err((e, false)) => {
                self.notices.push(format!(
                    "router probe: it did not run ({e}); the model decides alone this turn"
                ));
                (None, Some(e))
            }
        };
        let event = audit::AuditEvent {
            session: self.session.id(),
            chat: Some(self.chat_id.clone()),
            kind: audit::EventKind::Step,
            step: Some("router probe".into()),
            wall_ms: Some(started.elapsed().as_secs_f64() * 1e3),
            detail,
            error,
            ..audit::AuditEvent::default()
        };
        if let Err(e) = self.audit.append(event) {
            self.notices.push(format!("audit log: {e}"));
        }
    }

    /// Makes the chat's turns turns on `docs` (`Some`), or ordinary turns again (`None`). A turn
    /// on documents is an ordinary agent turn (every tool, the memory, the date) whose system
    /// prompt starts with the Folio block (`folio::system`: the rules and the documents), so the
    /// pinned prefix is only those bytes, and a new day or a saved memory re-reads only what
    /// follows them. The history is kept whole; the next turn prefills it after the documents.
    /// A pinned chat keeps its room by dropping its oldest exchanges (`run::Turn::send`), never
    /// by summarizing, which would replace the documents.
    pub fn pin_documents(&mut self, docs: Option<Vec<folio::Doc>>) -> Result<(), AgentError> {
        self.frame_documents(docs, None)
    }

    /// `pin_documents`, or with `large` and no `docs`, the line naming a set too long to hold
    /// before the chat's own system text.
    fn frame_documents(
        &mut self,
        docs: Option<Vec<folio::Doc>>,
        large: Option<String>,
    ) -> Result<(), AgentError> {
        let system = self.opts.system.clone().unwrap_or_default();
        let plain = match (&self.pinned, &self.large) {
            (Some((_, plain)), _) => plain.clone(),
            (None, Some(note)) => (system.strip_prefix(note.as_str()))
                .map(|rest| rest.strip_prefix('\n').unwrap_or(rest).to_owned())
                .unwrap_or(system),
            (None, None) => system,
        };
        let large = large.filter(|_| docs.is_none());
        let system = match (&docs, &large) {
            (Some(docs), _) => folio::system(docs, &plain),
            (None, Some(note)) => folio::before(note.clone(), &plain),
            (None, None) => plain.clone(),
        };
        let opts = SessionOptions {
            system: Some(system.clone()),
            ..self.opts.clone()
        };
        let mut history = self.session.history();
        match history.first_mut() {
            Some(first) if first.role == Role::System => first.content = system,
            _ => history.insert(
                0,
                ChatMessage {
                    role: Role::System,
                    content: system,
                    ..ChatMessage::default()
                },
            ),
        }
        self.session
            .restore(opts.clone(), history)
            .map_err(AgentError::Engine)?;
        self.opts = opts;
        self.pinned = docs.map(|d| (d, plain));
        self.large = large;
        Ok(())
    }

    /// Reads the chat's documents (its attached files, and `links`, the web links of the user's
    /// message, which `folio::links` finds: PDFs, share links and articles) and holds them by size (`folio::route`): small ones as
    /// plain text in the system prompt, medium ones pinned in the engine, with the chip's states
    /// told to `on` as the read goes, and large ones named in the system prompt and left to ask_docs. A set already held is not
    /// read again, and one released (`release_documents`) is imported back. `unpin_others` lets
    /// the other chats' pins go when the engine's pool is full. Call it before a turn.
    pub fn read_documents(
        &mut self,
        links: &[String],
        stop: &AtomicBool,
        on: &mut dyn FnMut(AgentEvent),
        unpin_others: &mut dyn FnMut(),
    ) -> Result<(), AgentError> {
        let data = self.ctx.data_dir.clone();
        for url in links {
            match folio::store_url(&data, &self.chat_id, url, stop) {
                Err(skill::SkillError::Engine(e)) => return Err(AgentError::Engine(e)),
                // An article that cannot be fetched is the normal tools' to report.
                Err(e) if folio::is_document_link(url) => {
                    on(AgentEvent::Notice(format!("could not read {url}: {e}")));
                }
                _ => {}
            }
        }
        let (docs, skipped) =
            folio::load(&self.ctx, &self.chat_id, stop).map_err(AgentError::Documents)?;
        if skipped != self.left_out {
            for (name, why) in &skipped {
                on(AgentEvent::Notice(format!(
                    "{name} is left out of the documents: {why}."
                )));
            }
            self.left_out = skipped;
        }
        let config = config::load(&data).map_err(AgentError::Config)?;
        let larch = self.larch.clone();
        // A context that shrank since the chat opened (a documents plan it opened in went back)
        // is what its set and its turns must fit now; a bigger one may go back again (#590 F7).
        self.max_context = self.max_context.min(max_context(&larch));
        let env = folio::held::Env {
            larch: &larch,
            data: &data,
            namespace: self.namespace,
            context: self.max_context,
            keep: config.docs_keep_ready,
            budget: config.docs_cache_bytes,
            stop,
            tools: self.pin_tools(),
        };
        let docs = match docs {
            Some(docs) => docs,
            None => {
                if let Some(mut held) = self.held.take() {
                    held.release(&env, on, false);
                    self.pin_documents(None)?;
                }
                self.keep_plan(on);
                return Ok(());
            }
        };
        if let Some(held) = &mut self.held
            && held.same(&docs, &env)
        {
            // A refused read keeps the record: the next open may load into that plan again.
            held.hold(&env, on, unpin_others)
                .map_err(AgentError::Documents)?;
            // A read too slow to wait for left the set to be read by section (`Kind::Large`).
            if held.kind == folio::held::Kind::Large && self.large.is_none() {
                let note = folio::large_note(&held.docs);
                self.frame_documents(None, Some(note))?;
            }
            self.keep_plan(on);
            return Ok(());
        }
        if let Some(mut old) = self.held.take() {
            old.release(&env, on, false);
        }
        let held =
            folio::held::Held::open(&env, docs, on, unpin_others).map_err(AgentError::Documents)?;
        let (shown, large) = match held.kind {
            folio::held::Kind::Large => (None, Some(folio::large_note(&held.docs))),
            _ => (Some(held.docs.clone()), None),
        };
        self.held = Some(held);
        self.frame_documents(shown, large)?;
        self.keep_plan(on);
        Ok(())
    }

    /// Records the plan the chat's pin is held on and the context it routed against (none when
    /// nothing is pinned), so a restart that reopens the chat loads straight into it
    /// (`folio::cache::plan`). A record that cannot be written only costs that restart a second
    /// load: a notice, not an error.
    fn keep_plan(&self, on: &mut dyn FnMut(AgentEvent)) {
        let pinned = self.held.as_ref().is_some_and(|h| h.pinned(&self.larch));
        let plan = pinned.then(|| (loaded_plan(&self.larch), self.max_context));
        let dir = folio::cache::dir(&self.ctx.data_dir);
        if let Err(e) = folio::cache::keep_plan(&dir, &self.chat_id, plan) {
            on(AgentEvent::Notice(format!("document cache: {e}")));
        }
    }

    /// The chat's documents as plain text in its system prompt, nothing pinned: the baseline
    /// `folio-bench --plain` measures the pin against.
    #[doc(hidden)]
    pub fn hold_plain_documents(&mut self, stop: &AtomicBool) -> Result<(), AgentError> {
        let (docs, _) =
            folio::load(&self.ctx, &self.chat_id, stop).map_err(AgentError::Documents)?;
        self.pin_documents(docs)
    }

    /// Exports the chat's pin and lets it go, as the chat loses focus to another: the next
    /// `read_documents` imports it back, or reads the documents again when the file is gone.
    /// With `covered_only` a pin its file does not hold whole stays (`folio::held::Held::release`).
    pub fn release_documents(
        &mut self,
        stop: &AtomicBool,
        on: &mut dyn FnMut(AgentEvent),
        covered_only: bool,
    ) -> folio::held::Released {
        let tools = self.pin_tools();
        let Some(held) = &mut self.held else {
            return folio::held::Released::Nothing;
        };
        let data = self.ctx.data_dir.clone();
        let config = config::load(&data).unwrap_or_default();
        let larch = self.larch.clone();
        let env = folio::held::Env {
            larch: &larch,
            data: &data,
            namespace: self.namespace,
            context: self.max_context,
            keep: config.docs_keep_ready,
            budget: config.docs_cache_bytes,
            stop,
            tools,
        };
        held.release(&env, on, covered_only)
    }

    /// `plain`, the chat's own system prompt, as the session holds it: after the Folio block when
    /// documents are pinned, or after the line naming a set too long to hold (`folio::large_note`).
    /// The pinned chat's own text becomes `plain`.
    pub(crate) fn framed(&mut self, plain: String) -> String {
        match &mut self.pinned {
            Some((docs, was)) => {
                let system = folio::system(docs, &plain);
                *was = plain;
                system
            }
            None => match &self.large {
                Some(note) => folio::before(note.clone(), &plain),
                None => plain,
            },
        }
    }

    /// The documents the chat's turns are on (`pin_documents`), in letter order; none when its
    /// turns are ordinary ones.
    pub fn pinned_documents(&self) -> Option<&[folio::Doc]> {
        self.pinned.as_ref().map(|(docs, _)| docs.as_slice())
    }

    /// The chat's system prompt as its session holds it.
    pub fn system_prompt(&self) -> &str {
        self.opts.system.as_deref().unwrap_or_default()
    }

    /// The call waiting for `confirm`, if any; a reopened chat may start with one.
    pub fn parked(&self) -> Option<&parse::ToolCall> {
        self.run.parked.as_ref().map(|p| &p.call)
    }

    /// Answers the write the turn stopped at and continues the turn: `yes` runs it, `no` tells
    /// the model the user declined.
    pub fn confirm(&mut self, yes: bool) -> Result<Turn<'_>, AgentError> {
        self.new_board();
        memory::capture::cancel_all();
        let parked = self.run.parked.take().ok_or(AgentError::NotParked)?;
        // A reopened chat may start with this.
        self.anchor_once();
        Ok(Turn::confirmed(self, parked, yes))
    }

    /// Puts the chat's prompt prefixes in the prefix cache, once, before its first chat step, and
    /// not while the chat opens, which a caller may do holding a lock: a first turn Rust answers
    /// or calls a skill for (a web question) does not wait for them. A cold cache only slows the
    /// turn, so a failure is a notice, the turn's first event.
    pub(crate) fn anchor_once(&mut self) {
        if std::mem::take(&mut self.unanchored) {
            let failed = self.warm();
            self.notices.extend(failed);
        }
    }

    /// Sets whether this chat's capture lets a `replaces` replace, which the model's size decides
    /// (`memory::capture::full_replaces`): a test's way to a large model on a tiny one.
    #[doc(hidden)]
    pub fn set_capture_replaces(&mut self, on: bool) {
        if let Some(c) = &mut self.capture {
            c.replace = on;
        }
    }

    /// The memory block (`run::fitted_memory_block`) for this chat's system prompt with the tools
    /// `tools`, as the chat shows them.
    pub(crate) fn fitted_memory_block(&self, tools: &[&str]) -> Option<String> {
        let shown: &[&str] = if self.family.is_some() { tools } else { &[] };
        let today = run::today();
        run::fitted_memory_block(
            &self.larch,
            &self.ctx.data_dir,
            &self.registry,
            shown,
            self.family,
            self.thinking,
            |block| {
                run::system_prompt(
                    &self.folders.block,
                    Some(block),
                    &today,
                    self.knows_until,
                    &self.folders.listing,
                )
            },
        )
    }

    /// Puts the chat's prompt prefixes in the prefix cache when the chat anchors
    /// (`AgentOptions::prefix_cache`): at its first turn, and after a compaction rebuilt the
    /// system prompt. A failure's notice comes back.
    pub(crate) fn warm(&self) -> Option<String> {
        if !self.prefix_cache {
            return None;
        }
        let e = self.anchor(&self.larch).err()?;
        Some(format!(
            "the system prompt did not reach the prefix cache: {e}"
        ))
    }

    /// Runs `tool` in place of the built-in tool of its name, or adds it: a test's way to a write
    /// whose audit line fails after the write, which no built-in tool can bring about, or to a
    /// fake of a tool web_answer calls.
    #[doc(hidden)]
    pub fn replace_tool(&mut self, tool: Box<dyn tool::Tool>) {
        Arc::get_mut(&mut self.registry)
            .expect("no skill runs between turns")
            .replace(tool);
    }

    /// Sends the key turn for the daily note line `noted`, after saving the last one's keys.
    // ponytail: a chat's note lines get keys, ask-docs's do not; send its line here too when
    // the memory bench shows ask-docs notes asked about
    fn start_keys(&mut self, noted: memory::Noted) {
        self.finish_keys();
        let Some(keys) = &mut self.keys else {
            return;
        };
        keys.session.reset();
        let (session, p) = (
            &mut keys.session,
            memory::key_params(self.family, self.namespace),
        );
        // A key turn is foreground (`run::params`), which seats never hold back, so a turn's
        // drop that starts it never waits.
        match SeatAdmission::of(&self.larch).seat(p.class, &AtomicBool::default(), || {
            session.send(&noted.text, p)
        }) {
            Ok(stream) => {
                let id = session.id();
                let read = move || workers::answer(id, stream, &AtomicBool::default());
                keys.pending = Some((noted, std::thread::spawn(read)));
            }
            Err(e) => {
                let session = keys.session.id();
                self.keys_done(session, &noted, None, Some(e.to_string()));
            }
        }
    }

    /// Waits for the key turn that runs, if any, and saves its keys.
    fn finish_keys(&mut self) {
        let Some(keys) = &mut self.keys else {
            return;
        };
        let Some((noted, read)) = keys.pending.take() else {
            return;
        };
        let session = keys.session.id();
        // A panic of the reader is the caller's, as when the caller read the stream itself.
        let read = read.join().unwrap_or_else(|p| std::panic::resume_unwind(p));
        let (answer, error) = match read {
            Ok(a) => {
                let saved = memory::parse_keys(&a.text, a.finish == larch::Finish::Length)
                    .and_then(|keys| memory::save_keys(&self.ctx.data_dir, &noted, &keys));
                (Some(a), saved.err().map(|e| e.to_string()))
            }
            Err(e) => (None, Some(e.to_string())),
        };
        self.keys_done(session, &noted, answer.as_ref(), error);
    }

    /// The key turn's audit line, and a failure's notice for the next turn. The line belongs to
    /// no chat, so a write's outcome before it still reads as its chat's last step.
    fn keys_done(
        &mut self,
        session: u64,
        noted: &memory::Noted,
        a: Option<&workers::Answer>,
        error: Option<String>,
    ) {
        if let Some(e) = &error {
            self.notices.push(format!(
                "no search keys for daily/{}.md line {}: {e}",
                noted.day, noted.line
            ));
        }
        let event = audit::AuditEvent {
            session,
            kind: audit::EventKind::Turn,
            step: Some("memory keys".into()),
            error,
            prompt_tokens: a.map(|a| a.usage.prompt_tokens),
            cached_tokens: a.map(|a| a.usage.cached_tokens),
            generated_tokens: a.map(|a| a.usage.completion_tokens),
            ttft_ms: a.map(|a| a.timings.ttft_ms),
            tok_s: a.map(|a| a.timings.tok_s),
            ..audit::AuditEvent::default()
        };
        if let Err(e) = self.audit.append(event) {
            self.notices.push(format!("audit log: {e}"));
        }
    }
}

impl Drop for Agent {
    /// The last note line's keys are saved before the chat closes, unless a panic is unwinding:
    /// waiting on the model then could hang the panic.
    fn drop(&mut self) {
        // A capture never outlives its chat: cancelled, then joined (quick once cancelled), so
        // it saves nothing more and its stream lets the engine go.
        if let Some(running) = self.capture.as_mut().and_then(|c| c.running.take()) {
            running
                .cancel
                .store(true, std::sync::atomic::Ordering::Relaxed);
            let _ = running.thread.join();
        }
        if !std::thread::panicking() {
            self.finish_keys();
        }
    }
}

/// `larch-agent chat`, on `larch`, which the probe router reads through: one prompt per line; text as it comes, a line per tool call and result,
/// `y/n` before a write, and the timings after each turn. `/quit` or the end of input stops it.
pub fn repl(
    larch: &Larch,
    agent: &mut Agent,
    input: &mut impl BufRead,
    out: &mut impl Write,
) -> io::Result<()> {
    let mut line = String::new();
    loop {
        let (mut user, mut answer) = (None, None);
        // A reopened chat may wait for the write it was closed at, or for a research plan, which
        // it shows again as its live turn did (R8).
        if let Some(call) = agent.parked() {
            if let Some(plan) = skills::research::shown_parked(call) {
                writeln!(out, "note: {plan}")?;
            }
            answer = Some(ask(input, out, call)?);
        } else {
            write!(out, "> ")?;
            out.flush()?;
            line.clear();
            if input.read_line(&mut line)? == 0 || line.trim() == "/quit" {
                return Ok(());
            }
            if line.trim().is_empty() {
                continue;
            }
            user = Some(line.trim().to_owned());
        }
        loop {
            let turn = match (user.take(), answer.take()) {
                (Some(user), _) => agent.turn_with(larch, &user),
                (None, Some(yes)) => agent.confirm(yes),
                (None, None) => break,
            };
            let turn = match turn {
                Ok(turn) => turn,
                Err(e) => {
                    writeln!(out, "note: {e}")?;
                    break;
                }
            };
            // Text shows a few words at a time, as `larch chat` shows it; tool lines start on a
            // line of their own after it.
            let mut batch = Batcher::default();
            let mut open_line = false;
            for event in turn {
                if let AgentEvent::Text(text) = &event {
                    if let Some(ready) = batch.push(text, Instant::now()) {
                        show(out, &ready, &mut open_line)?;
                    }
                    continue;
                }
                // The hidden reasoning stays hidden here: the CLI shows the answer.
                if let AgentEvent::Thinking(_) = event {
                    continue;
                }
                show(out, &batch.finish(), &mut open_line)?;
                if std::mem::take(&mut open_line) {
                    writeln!(out)?;
                }
                match event {
                    // Taken above.
                    AgentEvent::Text(_) | AgentEvent::Thinking(_) => {}
                    AgentEvent::ToolCall { name, args } => writeln!(out, "-> {name} {args}")?,
                    AgentEvent::ToolResult { name, summary } => {
                        writeln!(out, "<- {name}: {summary}")?
                    }
                    AgentEvent::NeedsConfirm { call } => answer = Some(ask(input, out, &call)?),
                    AgentEvent::Notice(text) => writeln!(out, "note: {text}")?,
                    AgentEvent::Progress(text) => writeln!(out, "... {text}")?,
                    AgentEvent::Documents(d) => writeln!(out, "... {d}")?,
                    AgentEvent::Done(t) => {
                        writeln!(
                            out,
                            "{:.2} tok/s · TTFT {:.2} ms · {} cached",
                            t.tok_s, t.ttft_ms, t.cached_tokens
                        )?;
                        if t.dropped_exchanges > 0 {
                            writeln!(out, "(dropped {} oldest exchanges)", t.dropped_exchanges)?;
                        }
                    }
                }
            }
            show(out, &batch.finish(), &mut open_line)?;
            if open_line {
                writeln!(out)?;
            }
        }
    }
}

/// Writes `text` at once; `open_line` is then whether it left a line unfinished.
fn show(out: &mut impl Write, text: &str, open_line: &mut bool) -> io::Result<()> {
    if !text.is_empty() {
        out.write_all(text.as_bytes())?;
        out.flush()?;
        *open_line = !text.ends_with('\n');
    }
    Ok(())
}

/// Asks y/n before a write; anything but yes, or the end of input, declines it.
fn ask(input: &mut impl BufRead, out: &mut impl Write, call: &parse::ToolCall) -> io::Result<bool> {
    write!(out, "allow {} {}? [y/n] ", call.name, call.args)?;
    out.flush()?;
    let mut line = String::new();
    input.read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes"))
}

/// Writes `text` to a temporary file beside `path`, syncs it and renames it over `path`, so a
/// crash or a full disk leaves the old file or the new one, never part of one. A file that was
/// there keeps its permissions, and one the user may not write is refused, as `fs::write`
/// refuses it. Without `clobber`, a file that appeared at `path` (a link, say) fails the write
/// instead of being replaced, except on a volume that cannot rename that way.
pub(crate) fn write_atomic(path: &Path, text: &str, clobber: bool) -> io::Result<()> {
    let old = match fs::metadata(path) {
        Ok(old) => Some(old),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => return Err(e),
    };
    if let Some(old) = &old {
        // Checked before the temporary file exists: beside an allowed folder itself, it would be
        // outside the folder. A FIFO or a device is never replaced either.
        if !old.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "not a regular file",
            ));
        }
        // The rename needs only a writable folder.
        OpenOptions::new().write(true).open(path)?;
    }
    let mut tmp = temp_beside(path, old.as_ref())?;
    tmp.write_all(text.as_bytes())?;
    if let Some(old) = old {
        tmp.as_file().set_permissions(old.permissions())?;
    }
    tmp.as_file().sync_all()?;
    let done = if clobber {
        tmp.persist(path)
    } else {
        // A volume with neither a rename that may not replace nor hard links (exFAT on macOS)
        // fails this. A plain rename onto a name that is still free does the same there, short of
        // a name made in between; exFAT has no symbolic links, the names this must not replace.
        // ponytail: any error but an existing name takes the plain rename; on a volume with links
        // and no exclusive rename, a link made in that instant is replaced. Refuse there by
        // error code when such a volume shows up.
        tmp.persist_noclobber(path).or_else(|e| {
            let free =
                fs::symlink_metadata(path).is_err_and(|err| err.kind() == io::ErrorKind::NotFound);
            if e.error.kind() != io::ErrorKind::AlreadyExists && free {
                e.file.persist(path)
            } else {
                Err(e)
            }
        })
    };
    done.map(drop).map_err(|e| e.error)
}

/// An empty temporary file in `path`'s folder, to be renamed over `old`. It holds the new text
/// before the rename, so from the start only those who may read `old` may read it.
// ponytail: on Windows the new file takes the folder's inherited ACL, not an ACL of the old
// file's own, and on macOS and Linux it loses the old file's ACL and extended attributes; replace
// through ReplaceFileW (Windows) or copy them onto the open file (fcopyfile, fsetxattr) in
// larch-sys when a skill writes such files
#[cfg_attr(not(unix), allow(unused_mut, unused_variables))]
fn temp_beside(path: &Path, old: Option<&fs::Metadata>) -> io::Result<NamedTempFile> {
    let dir = path.parent().ok_or(io::ErrorKind::InvalidInput)?;
    let mut builder = tempfile::Builder::new();
    // The old file's mode, less the umask until the caller sets it exactly; for a new file, what
    // `fs::write` gives (tempfile's 0600 would hide it from other users).
    #[cfg(unix)]
    builder.permissions(old.map_or_else(
        || std::os::unix::fs::PermissionsExt::from_mode(0o666),
        fs::Metadata::permissions,
    ));
    let tmp = builder.tempfile_in(dir)?;
    // A new file belongs to this process (and its group, or the folder's), and the rename would
    // hand the text and the file to them. Only the owner or an administrator may set them back;
    // anyone else's write is refused. The owner may set only a group they are in; for another
    // group the file takes the folder's, as a new file does, when the mode gives the group what
    // it gives anyone else: then no one gains or loses access. Otherwise (640, or 604, which
    // shuts the group out) the old group's members would get the others' access, and the new
    // group's the group's.
    #[cfg(unix)]
    if let Some(old) = old {
        use std::os::unix::fs::{MetadataExt, fchown};
        let new = tmp.as_file().metadata()?;
        let uid = (new.uid() != old.uid()).then_some(old.uid());
        let gid = (new.gid() != old.gid()).then_some(old.gid());
        if uid.is_some() || gid.is_some() {
            let group_only_harmless = uid.is_none() && (old.mode() >> 3) & 0o7 == old.mode() & 0o7;
            match fchown(tmp.as_file(), uid, gid) {
                Err(e) if e.kind() == io::ErrorKind::PermissionDenied && group_only_harmless => {}
                Err(e) => {
                    return Err(io::Error::new(
                        e.kind(),
                        format!("cannot keep the file's owner and group: {e}"),
                    ));
                }
                Ok(()) => {}
            }
        }
    }
    Ok(tmp)
}

/// The loaded engine's context and seats, as a documents plan names them.
fn loaded_plan(larch: &Larch) -> larch_engine::DocumentsPlan {
    let values = larch.knobs().effective().values;
    let at = |name| larch::knob::spec(name).map_or(0, |k| values[k.id as usize] as u32);
    larch_engine::DocumentsPlan {
        max_context: at("max_context"),
        seats: at("max_batch_seqs"),
    }
}

/// The loaded engine's `max_context`: the most tokens a session holds.
pub fn max_context(larch: &Larch) -> u32 {
    larch::knob::spec("max_context")
        .map(|k| larch.knobs().effective().values[k.id as usize] as u32)
        .expect("max_context is a registered knob")
}

#[cfg(test)]
mod tests {
    use std::fs;

    use larch::{LoadError, LoadOptions};
    use serde_json::json;

    use crate::tool::{Builtin, Permission, Registry, Tool};

    fn outside(name: &str, schema: serde_json::Value) -> Box<dyn Tool> {
        Builtin::boxed(
            name,
            "An outside tool.",
            Permission::Read,
            64,
            schema,
            |_, _| Ok(String::new()),
        )
    }

    /// Outside tools join in order; a built-in's name, a name not shaped `<server>__<tool>`, a
    /// repeated name, a schema the call grammar refuses and one too long for any step are left
    /// out, each group with a notice. Every other one joins, past `MAX_OUTSIDE` too: the cap is
    /// on what one step shows (`capped`), so retrieval can still reach the rest.
    #[test]
    fn outside_tools_are_filtered_before_a_chat_shows_them() {
        let object = json!({"type": "object", "properties": {"q": {"type": "string"}}});
        let mut tools = vec![
            outside("time__now", object.clone()),
            outside("read_file", object.clone()),
            outside("plain_name", object.clone()),
            outside("time__now", object.clone()),
            outside(
                "bad__schema",
                json!({"type": "object", "properties": {"a": {"allOf": [{"type": "string"}, {"type": "integer"}]}}}),
            ),
        ];
        let long = json!({"type": "object", "properties": {"q": {"type": "string",
            "description": "x".repeat(super::OUTSIDE_CHARS)}}});
        tools.push(outside("long__tool", long));
        tools.extend(
            (0..super::MAX_OUTSIDE).map(|i| outside(&format!("many__t{i}"), object.clone())),
        );
        let (kept, notes) = super::usable_outside(tools, &Registry::builtin());
        let names: Vec<_> = kept.iter().map(|t| t.spec().name).collect();
        assert_eq!(names.len(), super::MAX_OUTSIDE + 1);
        assert_eq!(names[0], "time__now");
        // A step shows the built-ins and the first `MAX_OUTSIDE` outside tools.
        let mut registry = Registry::builtin();
        let interned: Vec<&'static str> = names.iter().map(|n| super::intern(n)).collect();
        for t in kept {
            registry.replace(t);
        }
        let all: Vec<&'static str> = ["read_file"].into_iter().chain(interned).collect();
        let step = super::capped(&all, &registry);
        assert_eq!(step.len(), 1 + super::MAX_OUTSIDE);
        assert_eq!(step[0], "read_file");
        assert!(!step.contains(&"many__t23"));
        assert!(
            !names.contains(&"read_file".to_string()) && !names.contains(&"plain_name".to_string())
        );
        assert!(
            notes[0].contains("read_file") && notes[0].contains("plain_name"),
            "{notes:?}"
        );
        assert!(
            notes[1].contains("more outside tools are left out"),
            "{notes:?}"
        );
        assert!(super::is_outside("time__now") && !super::is_outside("read_file"));
    }

    /// A refused pin reads as words, not as the engine's code (#590 F1); other errors keep theirs.
    #[test]
    fn a_refused_pin_says_why_in_words() {
        use larch::api::invalid;
        let refused = |code| {
            super::AgentError::Documents(crate::skill::SkillError::Engine(
                larch::EngineError::InvalidRequest(code),
            ))
            .to_string()
        };
        assert!(refused(invalid::DOC_TOO_LONG).starts_with("The documents are too long"));
        assert!(refused(invalid::PINS_FULL).starts_with("Other chats' documents"));
        assert!(refused(invalid::NO_TOKENS).starts_with("documents: "));
    }

    /// A server's page names why a chat leaves a tool out: the grammar's reason, or its size.
    #[test]
    fn a_left_out_tool_says_why() {
        let object = json!({"type": "object", "properties": {"q": {"type": "string"}}});
        assert_eq!(super::left_out(&outside("time__now", object).spec()), None);
        let all_of =
            json!({"type": "object", "properties": {"a": {"allOf": [{"type": "string"}]}}});
        let why = super::left_out(&outside("bad__schema", all_of).spec()).unwrap();
        assert!(matches!(why, super::LeftOut::Grammar(_)), "{why:?}");
        assert!(why.to_string().contains("allOf"), "{why}");
        let long = json!({"type": "object", "properties": {"q": {"type": "string",
            "description": "x".repeat(super::OUTSIDE_CHARS)}}});
        let why = super::left_out(&outside("long__tool", long).spec()).unwrap();
        assert!(
            matches!(why, super::LeftOut::Size(n) if n > super::OUTSIDE_CHARS),
            "{why:?}"
        );
    }

    #[test]
    fn an_interned_name_is_one_string() {
        let a = super::intern("x__y");
        let b = super::intern(&String::from("x__y"));
        assert!(std::ptr::eq(a, b));
    }

    /// A model too big at the default context loads at the first half that fits; one too big at
    /// any is refused at `LEAST_CONTEXT`; one whose need a smaller context leaves as it was (the
    /// weights) after one halving; every other error at once.
    #[test]
    fn a_model_that_does_not_fit_loads_with_less_context_down_to_the_least() {
        let opts = LoadOptions::new("m.gguf".into(), larch_kernels::CPU);
        let fits_at = |most: u32| {
            let mut tried = Vec::new();
            let got = super::load_fitting(opts.clone(), |o| {
                tried.push(o.max_context);
                // The KV pool's need, which a smaller context shrinks.
                match o.max_context <= most {
                    true => Ok(o.max_context),
                    false => Err(LoadError::DoesNotFit {
                        need: o.max_context.into(),
                        have: most.into(),
                    }),
                }
            });
            (got, tried)
        };
        assert_eq!(fits_at(4096), (Ok((4096, 4096)), vec![8192, 4096]));
        let too_big = Err(LoadError::DoesNotFit {
            need: 2048,
            have: 1024,
        });
        let all = vec![8192, 4096, super::LEAST_CONTEXT];
        assert_eq!(fits_at(1024), (too_big, all));
        let mut tried = Vec::new();
        let weights = LoadError::DoesNotFit { need: 5, have: 1 };
        let got = super::load_fitting(opts.clone(), |o| -> Result<u32, _> {
            tried.push(o.max_context);
            Err(weights)
        });
        assert_eq!((got, tried), (Err(weights), vec![8192, 4096]));
        let mut tries = 0;
        let got = super::load_fitting(opts, |_| -> Result<u32, _> {
            tries += 1;
            Err(LoadError::Io)
        });
        assert_eq!((got, tries), (Err(LoadError::Io), 1));
    }

    #[test]
    fn the_temporary_file_is_no_easier_to_read_than_the_file_it_replaces() {
        // It holds the new text before the rename: with a wider mode, or another owner or group,
        // other users could open it in that window and keep reading. On macOS a new file takes
        // its folder's group, and /private/tmp's is wheel: a file made in a folder of the user's
        // group (staff) and moved there has another group than the temporary file beside it,
        // also without root. Both folders are in /private/tmp, so the move stays on one volume
        // wherever $TMPDIR is.
        let folder = || {
            if cfg!(target_os = "macos") {
                tempfile::tempdir_in("/private/tmp").unwrap()
            } else {
                tempfile::tempdir().unwrap()
            }
        };
        let (dir, made) = (folder(), folder());
        #[cfg(target_os = "macos")]
        {
            use std::os::unix::fs::{MetadataExt, chown};
            let home = fs::metadata(std::env::var_os("HOME").unwrap()).unwrap();
            chown(made.path(), None, Some(home.gid())).unwrap();
        }
        let path = dir.path().join(".env");
        fs::write(made.path().join(".env"), "KEY=old\n").unwrap();
        fs::rename(made.path().join(".env"), &path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::{PermissionsExt, chown};
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
            // Only an administrator (root, as in a container) may give the file away; for anyone
            // else it stays theirs, and the owner and group check below holds by itself.
            chown(&path, Some(65534), Some(65534)).ok();
        }
        let old = fs::metadata(&path).unwrap();
        let tmp = super::temp_beside(&path, Some(&old)).unwrap();
        let new = tmp.as_file().metadata().unwrap();
        // On Windows the permissions are the file's attributes, and tempfile marks the file
        // temporary until it persists it: only the read-only flag must match there, and the ACL
        // decides who may read (see temp_beside).
        assert_eq!(new.permissions().readonly(), old.permissions().readonly());
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(new.permissions(), old.permissions());
            assert_eq!((new.uid(), new.gid()), (old.uid(), old.gid()));
        }
    }
}
