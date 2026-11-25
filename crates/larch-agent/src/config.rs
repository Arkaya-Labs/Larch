//! `<data_dir>/agent/config.toml`, the agent's settings. A missing file is the defaults; a key the
//! agent does not know, or a value its key does not take, is an error that names the key and the
//! line, so a typo never passes for a default.

use std::{
    fmt, fs, io,
    path::{Path, PathBuf},
};

use serde_json::Value;

use crate::{
    code::task::Verify,
    memory, router,
    skills::{ask_docs, doc::reader, read_papers, web_answer},
    tool,
    web::search::{Key, Provider},
    workers::Workers,
};

/// The keys, as an error lists them.
const KEYS: &str = "tool_format, git_tools, allowed, mcp_allow, max_workers, docs_workers, \
docs_window, docs_keep_ready, docs_cache_bytes, web_pages, web_reader, research_keep, paper_tokens, \
doc_followups, followup_tokens, thinking_budget, answer_max_tokens, recall_hits, memory_keys, memory_capture, \
router, router_margin, tool_grammar, \
outside_pick, probe_low, probe_high, code_experts, code_verify, \
brave_key, serper_key, tavily_key and exa_key";
/// The most lines `recall_hits` lets a recall return.
pub const MAX_RECALL_HITS: usize = 10;

/// The largest `thinking_budget`.
pub const MAX_THINKING_BUDGET: u32 = 4096;

/// How the chat's model steps think, and how long an answer may run after its thinking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Thinking {
    /// Tokens a think block may take before the facade closes it (`GenParams::thinking_budget`);
    /// 0 turns thinking off.
    pub budget: u32,
    /// Tokens an answer may take after its thinking; 0 is no cap.
    pub answer_max_tokens: u32,
}

impl Default for Thinking {
    fn default() -> Self {
        Self {
            // ponytail: 100 is an estimate from the v1 bench rows; the Mac bench's multi-step rows
            // at --thinking-budget 0, 100, 256 and 4096 pick the default
            budget: 100,
            answer_max_tokens: 0,
        }
    }
}

/// The agent's settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// The tool-call format (qwen3, llama3 or mistral), which wins over the architecture's row.
    pub tool_format: Option<String>,
    /// Whether a chat offers the git tools with the files and memory tools.
    pub git_tools: bool,
    /// Folders the tools may use, after the ones `--allow` names; `~` is the home folder.
    pub allowed: Vec<PathBuf>,
    /// Folders MCP callers may read when `larch-agent mcp` names no `--allow`; never the chat's.
    pub mcp_allow: Vec<PathBuf>,
    /// Sessions ask-docs runs at once, one of them kept free for the chat.
    pub max_workers: usize,
    /// Chunks ask-docs gives its workers, one each.
    pub docs_workers: usize,
    /// Tokens of a chunk's file an ask-docs worker reads on each side of it; 0 is the chunk alone.
    pub docs_window: u32,
    /// Whether pinned documents are kept on disk (`folio::cache`), so a restart does not read them
    /// again: "Keep documents ready after restart".
    pub docs_keep_ready: bool,
    /// The bytes the document cache may hold; `None` is `folio::cache::budget`'s default.
    pub docs_cache_bytes: Option<u64>,
    /// Pages web-answer fetches, distinct hosts first.
    pub web_pages: usize,
    /// What answers web-answer's question from the pages it fetched.
    pub web_reader: web_answer::Reader,
    /// Deprecated: read so that an older config.toml still loads; research's planner now reads
    /// its own report instead of the last rounds (v3 R5).
    pub research_keep: usize,
    /// Tokens of a paper each read_papers reader sees; a longer paper keeps its start and end.
    pub paper_tokens: u32,
    /// Whether a read_papers call with a question is answered by the follow-up reader from the
    /// documents' own text (`doc_followups = "on"`), not by the chat's model from the summaries.
    pub doc_followups: bool,
    /// Tokens of a chat's documents the follow-up reader gets whole; past them it gets a first
    /// read and its tools.
    pub followup_tokens: u32,
    /// `thinking_budget` and `answer_max_tokens`.
    pub thinking: Thinking,
    /// Lines the recall tool returns, 1 to `MAX_RECALL_HITS`.
    pub recall_hits: usize,
    /// Whether a model turn extracts key phrases from each daily note line of a chat.
    pub memory_keys: bool,
    /// Whether the loaded model reads each chat turn's user message in the background after the
    /// turn and saves the lasting facts in it (`memory::capture`).
    pub memory_capture: bool,
    /// Who decides between a tool call and words before a chat step (`larch-agent chat`).
    pub router: router::Router,
    /// The logprob gate's margin in nats: a first token closer than this runs the step with
    /// thinking.
    pub router_margin: f32,
    /// web_search's provider keys; an empty one is not kept.
    pub search_keys: Vec<(Provider, Key)>,
    /// Whether a chat step that offers tools samples under the call grammar (`run::grammar`).
    pub tool_grammar: bool,
    /// Code mode's area experts, which start only for a repository too big for one context;
    /// off by default, and on to measure them against the single session.
    pub code_experts: bool,
    /// How code mode checks the work with the tests: full, or simple for an A/B run.
    pub code_verify: Verify,
    /// A turn asks the model, one yes/no step per installed MCP server, whether the
    /// message is for that server's tools (`router::fits`), and a yes holds the turn to a call of
    /// them. Experimental (toolcall-opt item 5): off until its bench says on.
    pub outside_pick: bool,
    /// The probe router's two thresholds (toolcall-opt item 2): at or under `low` the step
    /// shows no tools, at or over `high` it is held to a call, in between the model decides.
    /// `None` keeps the probe file's threshold for `low` and never forces.
    pub probe_low: Option<f32>,
    pub probe_high: Option<f32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tool_format: None,
            git_tools: false,
            allowed: Vec::new(),
            mcp_allow: Vec::new(),
            max_workers: Workers::default().max_workers,
            docs_workers: ask_docs::DOCS_WORKERS,
            // ponytail: 0, as 7 workers with a window prefill several times the tokens; the Mac
            // bench's --docs-window 0 against 128 picks the default
            docs_window: 0,
            docs_keep_ready: true,
            docs_cache_bytes: None,
            // ponytail: 5, a guess at enough sources for a short answer; the GPU check's total
            // time against answer quality at 3, 5 and 10 picks the default
            web_pages: web_answer::WEB_PAGES,
            // Plain reads 32 of the replay's 36 answerable questions right to chunks' 22, at 1.91x the
            // time and 2 of 5 abstain rows (agent-replay-m5-q30, 28 Oct); the owner took accuracy.
            web_reader: web_answer::Reader::Plain,
            research_keep: 3,
            paper_tokens: read_papers::PAPER_TOKENS,
            // ponytail: off until the v2 gate passes on the Mac; then on
            doc_followups: false,
            followup_tokens: reader::FOLLOWUP_TOKENS,
            thinking: Thinking::default(),
            recall_hits: memory::RECALL_HITS,
            // ponytail: off until the memory bench shows recall with keys at least as good as
            // without on the Mac's models; then on
            memory_keys: false,
            memory_capture: true,
            // ponytail: off until the Mac's bench of off, logprob and probe picks the default
            router: router::Router::Off,
            router_margin: router::MARGIN,
            search_keys: Vec::new(),
            tool_grammar: true,
            // ponytail: off until the Mac's SWE-bench Verified subset passes with them on
            code_experts: false,
            // ponytail: full until the SWE-bench A/B of full against simple picks the default
            code_verify: Verify::Full,
            outside_pick: false,
            probe_low: None,
            probe_high: None,
        }
    }
}

/// Why config.toml could not be read: the file, its line (0 for the whole file) and the reason.
#[derive(Debug)]
pub struct ConfigError {
    /// The file.
    pub path: PathBuf,
    /// The line, counted from 1; 0 when the file could not be read.
    pub line: usize,
    /// What is wrong, naming the key.
    pub reason: String,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            0 => write!(f, "{}: {}", self.path.display(), self.reason),
            line => write!(f, "{}:{line}: {}", self.path.display(), self.reason),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Where `data_dir`'s config.toml is.
pub fn path(data_dir: &Path) -> PathBuf {
    data_dir.join("agent").join("config.toml")
}

/// The settings in `data_dir`'s config.toml, or the defaults when there is none.
pub fn load(data_dir: &Path) -> Result<Config, ConfigError> {
    let path = path(data_dir);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Config::default()),
        Err(e) => {
            return Err(ConfigError {
                path,
                line: 0,
                reason: e.to_string(),
            });
        }
    };
    parse(&text).map_err(|(line, reason)| ConfigError { path, line, reason })
}

/// Reads one `key = value` per line, with `#` comments and blank lines between.
// ponytail: a TOML subset: each value is a JSON string, true or false, a whole number or a one-line
// list of strings, and what reads here TOML reads the same way (a JSON-only `\/`, a surrogate
// `\u` escape and a number past i64::MAX are refused). Valid TOML it refuses: tables, multi-line
// lists, 'literal strings', `\U` escapes, a comment right after a number or true/false (`4#c`),
// quoted or dotted keys, a trailing comma in a list, `1_000` and TOML's other number forms, and a
// tab inside a string. Parse with a TOML crate when a config.toml needs one of them.
fn parse(text: &str) -> Result<Config, (usize, String)> {
    let mut config = Config::default();
    let mut seen: Vec<&str> = Vec::new();
    for (i, line) in text
        .strip_prefix('\u{feff}')
        .unwrap_or(text)
        .lines()
        .enumerate()
    {
        let n = i + 1;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, rest)) = line.split_once('=') else {
            return Err((n, format!("'{line}' is not key = value")));
        };
        let key = key.trim();
        let wrong = |takes: &str| (n, format!("{key} takes {takes}"));
        let mut values = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
        let value = match values.next() {
            Some(Ok(value)) => value,
            Some(Err(e)) => {
                // serde_json's own position counts inside the value; the line is the one to say.
                let e = e.to_string();
                let e = e.split(" at line ").next().unwrap_or(&e);
                return Err((n, format!("{key}: the value does not read: {e}")));
            }
            None => return Err((n, format!("{key} has no value"))),
        };
        let end = values.byte_offset();
        if let Some(escape) = not_toml(&rest[..end]) {
            return Err((n, format!("{key}: TOML has no escape {escape}")));
        }
        let after = rest[end..].trim();
        if !after.is_empty() && !after.starts_with('#') {
            return Err((n, format!("{key}: '{after}' after the value")));
        }
        if seen.contains(&key) {
            return Err((n, format!("{key} is set twice")));
        }
        seen.push(key);
        match key {
            "tool_format" => {
                let formats = "qwen3, llama3 or mistral";
                let name = value.as_str().ok_or_else(|| wrong(formats))?;
                // The three names alone, not the architectures family_for also reads.
                if !["qwen3", "llama3", "mistral"].contains(&name) {
                    return Err(wrong(&format!("{formats}, not '{name}'")));
                }
                config.tool_format = Some(name.into());
            }
            "git_tools" => {
                config.git_tools = value.as_bool().ok_or_else(|| wrong("true or false"))?
            }
            "allowed" | "mcp_allow" => {
                let list = value.as_array().ok_or_else(|| wrong("a list of folders"))?;
                for folder in list {
                    let folder = folder.as_str().ok_or_else(|| wrong("a list of folders"))?;
                    let path = tool::expand_home(folder).map_err(|e| (n, format!("{key}: {e}")))?;
                    // Relative to the folder the agent started in, it would change with it.
                    if path.is_relative() {
                        return Err((
                            n,
                            format!(
                                "{key}: '{folder}' is not a full path or one that starts with ~"
                            ),
                        ));
                    }
                    match key {
                        "allowed" => config.allowed.push(path),
                        _ => config.mcp_allow.push(path),
                    }
                }
            }
            "max_workers" | "docs_workers" => {
                // TOML's whole numbers end at i64::MAX.
                let n = value
                    .as_u64()
                    .filter(|&w| w > 0 && i64::try_from(w).is_ok())
                    .and_then(|w| usize::try_from(w).ok())
                    .ok_or_else(|| wrong(&format!("a whole number from 1 to {}", i64::MAX)))?;
                if key == "max_workers" {
                    config.max_workers = n;
                } else {
                    config.docs_workers = n;
                }
            }
            "docs_window" => {
                config.docs_window = value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        wrong(&format!(
                            "a whole number from 0 (the chunk alone) to {}",
                            u32::MAX
                        ))
                    })?;
            }
            "docs_keep_ready" => {
                config.docs_keep_ready = value.as_bool().ok_or_else(|| wrong("true or false"))?
            }
            "docs_cache_bytes" => {
                config.docs_cache_bytes = Some(
                    value
                        .as_u64()
                        .filter(|&b| i64::try_from(b).is_ok())
                        .ok_or_else(|| {
                            wrong(&format!("a whole number of bytes, at most {}", i64::MAX))
                        })?,
                );
            }
            "web_pages" => {
                config.web_pages = value
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|n| (1..=web_answer::MAX_PAGES).contains(n))
                    .ok_or_else(|| {
                        wrong(&format!(
                            "a whole number from 1 to {}",
                            web_answer::MAX_PAGES
                        ))
                    })?;
            }
            "web_reader" => {
                let readers = "plain or chunks";
                let name = value.as_str().ok_or_else(|| wrong(readers))?;
                config.web_reader = clap::ValueEnum::from_str(name, false)
                    .map_err(|_| wrong(&format!("{readers}, not '{name}'")))?;
            }
            "research_keep" => {
                config.research_keep = value
                    .as_u64()
                    .filter(|&n| i64::try_from(n).is_ok())
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| {
                        wrong(&format!(
                            "a whole number from 0 (every round folded) to {}",
                            i64::MAX
                        ))
                    })?;
            }
            "paper_tokens" => {
                config.paper_tokens = value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|&n| n >= read_papers::MIN_PAPER_TOKENS)
                    .ok_or_else(|| {
                        wrong(&format!(
                            "a whole number from {} to {}",
                            read_papers::MIN_PAPER_TOKENS,
                            u32::MAX
                        ))
                    })?;
            }
            "doc_followups" => {
                config.doc_followups = match value.as_str() {
                    Some("on") => true,
                    Some("off") => false,
                    _ => return Err(wrong("\"on\" or \"off\"")),
                }
            }
            "followup_tokens" => {
                config.followup_tokens = value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|&n| n >= read_papers::MIN_PAPER_TOKENS)
                    .ok_or_else(|| {
                        wrong(&format!(
                            "a whole number from {} to {}",
                            read_papers::MIN_PAPER_TOKENS,
                            u32::MAX
                        ))
                    })?;
            }
            "recall_hits" => {
                config.recall_hits = value
                    .as_u64()
                    .and_then(|h| usize::try_from(h).ok())
                    .filter(|h| (1..=MAX_RECALL_HITS).contains(h))
                    .ok_or_else(|| wrong(&format!("a whole number from 1 to {MAX_RECALL_HITS}")))?;
            }
            "memory_keys" => {
                config.memory_keys = value.as_bool().ok_or_else(|| wrong("true or false"))?
            }
            "code_experts" => {
                config.code_experts = value.as_bool().ok_or_else(|| wrong("true or false"))?
            }
            "code_verify" => {
                let modes = "full or simple";
                let name = value.as_str().ok_or_else(|| wrong(modes))?;
                config.code_verify = clap::ValueEnum::from_str(name, false)
                    .map_err(|_| wrong(&format!("{modes}, not '{name}'")))?;
            }
            "memory_capture" => {
                config.memory_capture = value.as_bool().ok_or_else(|| wrong("true or false"))?
            }
            "tool_grammar" => {
                config.tool_grammar = value.as_bool().ok_or_else(|| wrong("true or false"))?
            }
            "probe_low" | "probe_high" => {
                let v = (value.as_f64())
                    .or_else(|| value.as_i64().map(|i| i as f64))
                    .filter(|v| (0.0..=1.0).contains(v))
                    .ok_or_else(|| wrong("a number from 0 to 1"))? as f32;
                match key {
                    "probe_low" => config.probe_low = Some(v),
                    _ => config.probe_high = Some(v),
                }
                // Below `probe_low` masks the call and above `probe_high` forces it, so a low above
                // the high would mask some of what it forces.
                if let (Some(low), Some(high)) = (config.probe_low, config.probe_high)
                    && low > high
                {
                    return Err((n, format!("probe_low ({low}) is above probe_high ({high})")));
                }
            }
            "outside_pick" => {
                config.outside_pick = value.as_bool().ok_or_else(|| wrong("true or false"))?
            }
            "thinking_budget" => {
                config.thinking.budget = value
                    .as_u64()
                    .and_then(|b| u32::try_from(b).ok())
                    .filter(|&b| b <= MAX_THINKING_BUDGET)
                    .ok_or_else(|| {
                        wrong(&format!(
                            "a whole number from 0 (thinking off) to {MAX_THINKING_BUDGET}"
                        ))
                    })?;
            }
            "answer_max_tokens" => {
                config.thinking.answer_max_tokens = value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        wrong(&format!("a whole number from 0 (no cap) to {}", u32::MAX))
                    })?;
            }
            "router" => {
                let routers = "off, logprob, probe or score";
                let name = value.as_str().ok_or_else(|| wrong(routers))?;
                config.router = clap::ValueEnum::from_str(name, false)
                    .map_err(|_| wrong(&format!("{routers}, not '{name}'")))?;
            }
            "router_margin" => {
                config.router_margin = value
                    .as_f64()
                    .map(|m| m as f32)
                    .filter(|m| *m > 0.0 && *m <= router::CERTAIN)
                    .ok_or_else(|| {
                        wrong(&format!(
                            "a number of nats above 0 and at most {}",
                            router::CERTAIN
                        ))
                    })?;
            }
            _ if let Some(provider) = Provider::for_key(key) => {
                let secret = value.as_str().ok_or_else(|| wrong("a string"))?;
                if !secret.is_empty() {
                    config.search_keys.push((provider, Key(secret.into())));
                }
            }
            _ => return Err((n, format!("unknown key '{key}'; the keys are {KEYS}"))),
        }
    }
    Ok(config)
}

/// The first escape in `value`, JSON text, that a TOML string does not have: `\/`, or a `\u`
/// that names half of a surrogate pair, which TOML refuses where JSON joins the pair.
fn not_toml(value: &str) -> Option<&str> {
    let mut at = 0;
    while let Some(i) = value[at..].find('\\').map(|i| at + i) {
        let escape = value.get(i..i + 2)?;
        at = i + 2;
        match escape {
            "\\/" => return Some(escape),
            "\\u" => {
                let hex = value.get(i..i + 6)?;
                if u16::from_str_radix(&hex[2..], 16).is_ok_and(|c| (0xD800..=0xDFFF).contains(&c))
                {
                    return Some(hex);
                }
                at = i + 6;
            }
            _ => {}
        }
    }
    None
}
