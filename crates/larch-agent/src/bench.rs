//! The benchmark: the tool-call prompts, their scorer and runner, the ask-docs validators and the
//! `larch-agent-bench/1` report.

// The bench's requests are foreground, which seats count but never hold back and whose groups
// keep their `at_once` size (`SeatAdmission::take`, `group`), so it measures the engine's
// batching as sent, as before seats.

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    process::Command,
    sync::{LazyLock, Mutex, atomic::AtomicBool},
    time::Instant,
};

use anyhow::Context;
use larch::{
    ChatMessage, EngineError, Finish, GenParams, Larch, Role, Session, SessionOptions, TextEvent,
    TextStream, Timings, Usage, api::Grammar, toolcall::Mode,
};
use larch_engine::PrefixCacheStats;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};

use crate::{
    Agent, AgentOptions,
    audit::{Audit, AuditEvent, EventKind},
    config::Thinking,
    memory::{self, Hit, dates},
    parse::{Family, ParseError, Parsed, ToolCall, ValidateError},
    probe::{self, Probe},
    router::{self, Router},
    run::{self, AgentEvent, Checked, Kind},
    skill::{Skill, SkillError, SkillInput, SkillRun},
    skills::{
        ask_docs::{self, AskDocs, Cited, Facts, Log, NOT_FOUND, cites, ms, squash},
        web_answer::{self, Park, Reader, WebAnswer},
    },
    tool::{Builtin, Permission, Registry, Tool, ToolCtx, ToolSpec},
    web::{
        fetch::{self, Page, Seen},
        search,
    },
    workers::Workers,
};

pub mod folio;
mod memory_chats;
pub mod workflows;

pub use memory_chats::*;

/// The report's schema name.
pub const SCHEMA: &str = "larch-agent-bench/1";
/// The date in the bench's system prompt: bench/README's `date_string`, as the agent writes dates.
pub const DATE: &str = "2024-07-26";
/// The seed of every request the bench sends, so a sampling setting cannot make a rerun differ.
const SEED: u64 = 0;

/// One prompt of `bench/agent/toolcalls.jsonl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchPrompt {
    /// `<kind>-<n>`: clear, none, near (a near-miss tool name in the text) or infer.
    pub id: String,
    /// The tools the model is shown.
    pub tools: Vec<String>,
    /// Earlier turns of the chat, as a saved chat holds them, before `user`; empty for a first
    /// turn. When the last is a tool result, the row's reply is the model's answer to it.
    #[serde(default)]
    pub history: Vec<ChatMessage>,
    /// The user's text; empty when `history` ends with a tool result.
    pub user: String,
    /// The right call.
    pub expect: Expect,
    /// The recorded MCP servers installed for this row (`--mcp-fixture`), whose tools join the
    /// chat's as outside tools.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub servers: Vec<String>,
}

/// The call a prompt needs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expect {
    /// The tool, or `None` when a plain answer is right.
    pub tool: Option<String>,
    /// When not empty, any of these is right in place of `tool`, `null` a plain answer.
    #[serde(default)]
    pub tool_any: Vec<Option<String>>,
    /// Arguments the call must have: `null` for any value, text the value must contain (case
    /// aside), a number it must equal, or a list of paths it must hold and no other (in any order;
    /// case, a leading `./` and a trailing `/` aside), as read_papers' `papers`.
    pub args_required: Map<String, Value>,
    /// A plain answer must be a refusal (`REFUSALS`) that states no number or date (`DATED`) and
    /// does not guess (`HEDGES`).
    #[serde(default)]
    pub refuse: bool,
    /// Text a plain answer must hold, each, case aside, on word boundaries (`holds`).
    #[serde(default)]
    pub answer_must_contain: Vec<String>,
    /// Text a plain answer must hold one of, when not empty, as `answer_must_contain` matches.
    #[serde(default)]
    pub answer_must_contain_any: Vec<String>,
    /// Text a plain answer must not hold, as `answer_must_contain` matches.
    #[serde(default)]
    pub answer_must_not_contain: Vec<String>,
}

impl Expect {
    /// Whether a reply calling `tool`, or answering with `None`, picked a right one.
    fn takes(&self, tool: Option<&str>) -> bool {
        match self.tool_any.is_empty() {
            true => tool == self.tool.as_deref(),
            false => self.tool_any.iter().any(|t| t.as_deref() == tool),
        }
    }
}

/// What a refusal says, lower case with a straight apostrophe: a plain answer holding one of these
/// is wrong where only an answer is right (`tool` is `None`, no `tool_any`, no `refuse`), since
/// the right reply there is the answer. The phrases naming a source or a failed search come first;
/// the rest are said in the first person, so a recipe's "you can't go wrong" is not one.
pub const REFUSALS: [&str; 25] = [
    "your files do not",
    "the files do not",
    "the documents do not",
    "not in your files",
    "files don't",
    "documents don't",
    "couldn't find",
    "could not find",
    "nothing about",
    "do not answer",
    "do not say",
    "do not contain",
    "no information about",
    "no record",
    "cannot provide",
    "i don't have",
    "i do not have",
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i can't",
    "i cannot",
    "i'm unable to",
    "i am unable to",
];

/// What an answer that says it cannot reach the web holds, lower case with a straight apostrophe:
/// false in a chat that has web_answer, and its own counter (`Counters::offline_claims`).
pub const OFFLINE: [&str; 12] = [
    "can't access the internet",
    "cannot access the internet",
    "don't have access to the internet",
    "do not have access to the internet",
    "no internet access",
    "can't browse",
    "cannot browse",
    "unable to browse",
    "can't search the web",
    "cannot search the web",
    "don't have access to real-time",
    "do not have access to real-time",
];

/// Words that state a number, a date or a time, which a refusal (`Expect::refuse`) must not hold:
/// "may" is left out, as a refusal says it as a verb, and "one", as in "no one" (a digit is caught
/// on its own).
pub const DATED: [&str; 53] = [
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "hundred",
    "thousand",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "year",
    "years",
    "month",
    "months",
    "week",
    "weeks",
    "noon",
    "morning",
    "afternoon",
    "evening",
    "tomorrow",
    "tonight",
    "a.m.",
    "p.m.",
];

/// Words that guess, which a refusal must not hold either: "I don't know, but it is likely O
/// positive" still states the fact.
pub const HEDGES: [&str; 5] = ["usually", "typically", "likely", "probably", "generally"];

/// Whether lower-case `text` holds `word` on word boundaries, so "not" is not in "notes"; a
/// trailing `*` lets it run on ("minute*" is in "minutes").
fn holds(text: &str, word: &str) -> bool {
    let word = word.to_lowercase();
    let (word, open) = word
        .strip_suffix('*')
        .map_or((&*word, false), |w| (w, true));
    let edge = |c: Option<char>| !c.is_some_and(char::is_alphanumeric);
    text.match_indices(word).any(|(i, _)| {
        edge(text[..i].chars().next_back()) && (open || edge(text[i + word.len()..].chars().next()))
    })
}

/// Why a plain answer fails `expect`'s answer checks (`REFUSALS` too where no tool is right), or
/// `None` when it passes; a call is scored on its tool alone.
pub fn score_answer(expect: &Expect, parsed: &Parsed) -> Option<String> {
    if parsed.call.is_some() {
        return None;
    }
    // A model writes the curly apostrophe as often as the straight one.
    let text = parsed.text_before.to_lowercase().replace('\u{2019}', "'");
    let has = |s: &String| holds(&text, s);
    let only_answer = expect.tool.is_none() && expect.tool_any.is_empty() && !expect.refuse;
    if let Some(r) = REFUSALS.iter().find(|r| only_answer && holds(&text, r)) {
        return Some(format!("refuses: '{r}'"));
    }
    if expect.refuse {
        if !REFUSALS.iter().any(|r| holds(&text, r)) {
            return Some("does not refuse".into());
        }
        if text.contains(|c: char| c.is_ascii_digit()) {
            return Some("has a digit".into());
        }
        if let Some(w) = DATED.iter().chain(&HEDGES).find(|w| holds(&text, w)) {
            return Some(format!("has '{w}'"));
        }
    }
    if let Some(s) = expect.answer_must_contain.iter().find(|s| !has(s)) {
        return Some(format!("lacks '{s}'"));
    }
    let any = &expect.answer_must_contain_any;
    if !any.is_empty() && !any.iter().any(has) {
        return Some(format!("lacks any of '{}'", any.join("', '")));
    }
    (expect.answer_must_not_contain.iter())
        .find(|s| has(s))
        .map(|s| format!("has '{s}'"))
}

/// How one reply scores.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolcallScore {
    /// The parser read a call or an answer from it: no format error.
    pub parsed_ok: bool,
    /// The call names the expected tool, or there is no call and none was expected.
    pub right_tool: bool,
    /// The right tool with every required argument.
    pub args_ok: bool,
}

/// Scores a parsed reply against the prompt's expected call.
pub fn score_toolcall(expect: &Expect, parsed: &Parsed) -> ToolcallScore {
    let parsed_ok = matches!(parsed.error, None | Some(ParseError::MultipleCalls));
    let call = parsed.call.as_ref();
    let right_tool = parsed_ok && expect.takes(call.map(|c| c.name.as_str()));
    let args_ok = right_tool
        && expect.args_required.iter().all(|(key, want)| {
            match (call.and_then(|c| c.args.get(key)), want) {
                (None, _) => false,
                (Some(_), Value::Null) => true,
                (Some(Value::String(have)), Value::String(want)) => {
                    have.to_lowercase().contains(&want.to_lowercase())
                }
                (Some(Value::Array(have)), Value::Array(want)) => {
                    let paths = |v: &[Value]| -> Option<BTreeSet<String>> {
                        let path = |p: &str| {
                            p.trim_start_matches("./")
                                .trim_end_matches('/')
                                .to_lowercase()
                        };
                        v.iter().map(|v| v.as_str().map(path)).collect()
                    };
                    have.len() == want.len() && paths(have).is_some() && paths(have) == paths(want)
                }
                // 5.0 is 5, as the schema check takes it.
                (Some(have), want) if have.is_number() && want.is_number() => {
                    have.as_f64() == want.as_f64()
                }
                (Some(have), want) => have == want,
            }
        });
    ToolcallScore {
        parsed_ok,
        right_tool,
        args_ok,
    }
}

/// One prompt's result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    /// The prompt's ID.
    pub id: String,
    /// The first reply's score.
    pub before: ToolcallScore,
    /// The score after the one retry; the first's when no retry was needed.
    pub after: ToolcallScore,
    /// The first reply had a parse or validate error, so the loop's retry ran.
    pub retried: bool,
    /// The tool the last reply called.
    pub tool: Option<String>,
    /// The engine error that ended the prompt.
    pub error: Option<String>,
    /// Prompt tokens over both steps.
    pub prompt_tokens: u32,
    /// Generated tokens over both steps.
    pub generated_tokens: u32,
    /// Prompt tokens reused from parked KV, over both steps.
    pub cached_tokens: u32,
    /// The first step's time to first token.
    pub ttft_ms: f64,
    /// The first step's decode speed.
    pub tok_s: f64,
    /// Both steps, end to end.
    pub wall_ms: f64,
    /// From the start of the turn to the first thing the user sees (text after the hidden block,
    /// the call line or its confirm); `None` when nothing showed, or in a v1 report.
    #[serde(default)]
    pub visible_ms: Option<f64>,
    /// Where each step that ended at the token limit was cut.
    #[serde(default)]
    pub cut: Vec<Cut>,
    /// The last reply was a valid call with the wrong tool or a wrong argument value.
    #[serde(default)]
    pub wrong_but_valid: bool,
    /// Why the last reply, a plain answer, failed the answer checks (`score_answer`).
    #[serde(default)]
    pub wrong_answer: Option<String>,
    /// Calls the turn made, results that came back failed, and (with `--whole`) the 1-based call
    /// that was first the expected one.
    #[serde(default)]
    pub calls_made: u32,
    /// The tools of every call the turn made, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<String>,
    #[serde(default)]
    pub failed_results: u32,
    #[serde(default)]
    pub right_at: Option<u32>,
    /// The probe router's reading of a turn row, from the chat's audit line: its probability and
    /// what it did (auto, force or mask).
    #[serde(default)]
    pub probe: Option<f32>,
    #[serde(default)]
    pub probe_decision: Option<String>,
    /// What the router did before and at the first step; `None` with the router off.
    #[serde(default)]
    pub routed: Option<Routed>,
    /// The last reply, a plain answer, says it cannot reach the web (`OFFLINE`).
    #[serde(default)]
    pub offline_claim: bool,
    /// Calls to a tool not in the list, over both steps.
    #[serde(default)]
    pub unknown_tools: u32,
    /// Calls that fail their tool's schema, over both steps.
    #[serde(default)]
    pub schema_fails: u32,
    /// A check took back the turn's first reply (v3 D5) and the message went again, held to a
    /// call; only a `--turn` row has one.
    #[serde(default)]
    pub withdrawn: bool,
    /// The last reply as the loop read it (`reply_of`).
    #[serde(default)]
    pub reply: Option<String>,
}

/// What the router did with a row's first step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Routed {
    /// The gate's reading of the first token: at least this many nats between a call and words.
    pub margin: Option<f32>,
    /// The gate read a call.
    pub call: Option<bool>,
    /// The margin was under `router_margin`, so the step ran with thinking.
    pub again: bool,
    /// The probe's probability that the row needs a tool.
    pub probe: Option<f32>,
    /// The probe said no tool, so the step was shown none.
    pub masked: bool,
    /// The gate's one-token request: its prompt tokens, those the prefix cache held, and its time
    /// end to end in ms, which the row's `wall_ms` and `visible_ms` include and its `ttft_ms` does
    /// not; `None` without the gate, or in a report from before them.
    #[serde(default)]
    pub gate_prompt_tokens: Option<u32>,
    #[serde(default)]
    pub gate_cached_tokens: Option<u32>,
    #[serde(default)]
    pub gate_ms: Option<f64>,
}

/// Where a step that ended at its token limit (`Finish::Length`) was cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Cut {
    /// In plain text.
    Text,
    /// Inside a tool call.
    Call,
    /// Inside the hidden think block.
    Think,
}

/// Where a reply that ended at its token limit was cut, from where its text ends: in the think
/// block, in a call (after the parser found one), or in text. The parser's error says what is
/// wrong with a call, not where the reply stopped: two whole calls, or a whole call without a
/// name, then text, were cut in the text.
fn cut(text: &str, parsed: &Parsed) -> Cut {
    if parsed.error == Some(ParseError::UnclosedThinking) {
        return Cut::Think;
    }
    if parsed.call.is_none() && parsed.error.is_none() {
        return Cut::Text;
    }
    // The reply after its think block, from where the parser's call starts.
    let rest = match parsed.thinking {
        Some(_) => text.split_once("</think>").map_or("", |(_, after)| after),
        None => text,
    };
    let call = (rest.trim_start())
        .get(parsed.text_before.len()..)
        .unwrap_or_default();
    if ends_open(call) {
        Cut::Call
    } else {
        Cut::Text
    }
}

/// The markers a call starts with: Qwen3's tag, which `</tool_call>` closes, and the Llama 3 and
/// Mistral markers, whose call is the JSON after them.
const CALL_MARKERS: [&str; 3] = ["<tool_call>", "<|python_tag|>", "[TOOL_CALLS]"];
const CALL_CLOSE: &str = "</tool_call>";

/// Whether `text`, from where a call starts, ends inside a call. It walks the calls by their
/// bounds: a marker, then the JSON value (its end found by its brackets, strings skipped), then
/// for a tag the closing tag. It ends inside one right after a marker, inside the JSON, or before
/// a tag's closing tag. Between calls is text: only a marker, or a call-shaped value after one
/// (`;` between Llama 3 calls), starts another, and text brackets and tags in strings are not
/// structure.
fn ends_open(text: &str) -> bool {
    let (mut rest, mut first) = (text, true);
    loop {
        let t = rest.trim_start();
        if t.is_empty() {
            return false;
        }
        let marker = CALL_MARKERS.iter().find(|m| t.starts_with(**m));
        let body = marker.map_or(t, |m| t[m.len()..].trim_start());
        if marker.is_some() && body.is_empty() {
            return true;
        }
        // Where the parser saw a call there may be bare JSON; after a call, only a call-shaped
        // value: `[the contents` in the text after one is not a call.
        let bare = body.starts_with('{')
            || body
                .strip_prefix('[')
                .is_some_and(|b| first || b.trim_start().starts_with('{'));
        let mut after = if bare {
            match json_len(body) {
                Some(n) => &body[n..],
                None => return true,
            }
        } else if marker.is_some() {
            body
        } else {
            // Text: the next call starts at the next marker.
            match CALL_MARKERS.iter().filter_map(|m| t.find(m)).min() {
                Some(at) => {
                    (rest, first) = (&t[at..], false);
                    continue;
                }
                None => return false,
            }
        };
        if marker == Some(&CALL_MARKERS[0]) {
            match after.find(CALL_CLOSE) {
                Some(at) => after = &after[at + CALL_CLOSE.len()..],
                None => return true,
            }
        }
        (rest, first) = (after.trim_start().trim_start_matches(';'), false);
    }
}

/// The length of the JSON object or array `text` starts with, by its brackets with strings
/// skipped, whether or not it is valid JSON; `None` when the text ends inside it.
fn json_len(text: &str) -> Option<usize> {
    let (mut depth, mut string, mut escaped) = (0u32, false, false);
    for (i, c) in text.char_indices() {
        if string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// What went wrong on the way, over a section's model steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counters {
    /// Steps that ended at the token limit.
    pub length_stops: u32,
    /// Of those, the ones cut inside a tool call.
    pub in_call: u32,
    /// Of those, the ones cut inside the hidden think block.
    pub in_think: u32,
    /// Calls that parsed and validated but named the wrong tool or a wrong argument value.
    pub wrong_but_valid: u32,
    /// Retries after a parse or validate error.
    pub retries: u32,
    /// Plain answers that say they cannot reach the web (`Row::offline_claim`).
    #[serde(default)]
    pub offline_claims: u32,
    /// Calls to a tool not in the list; the call grammar stops them, but not a Llama 3 call
    /// written without `<|python_tag|>`.
    #[serde(default)]
    pub unknown_tools: u32,
    /// Calls that fail their tool's schema; the call grammar does not hold bounds (`minimum`,
    /// `minLength`) or a Llama 3 call written without `<|python_tag|>`, so some get past it.
    #[serde(default)]
    pub schema_fails: u32,
    /// Replies a check took back (`Row::withdrawn`).
    #[serde(default)]
    pub withdrawn: u32,
}

impl Counters {
    /// Adds one row's cut steps, wrong-but-valid calls and retries.
    fn add(&mut self, cut: &[Cut], wrong_but_valid: u32, retries: u32) {
        let count = |c| cut.iter().filter(|&&x| x == c).count() as u32;
        self.length_stops += cut.len() as u32;
        self.in_call += count(Cut::Call);
        self.in_think += count(Cut::Think);
        self.wrong_but_valid += wrong_but_valid;
        self.retries += retries;
    }
}

/// Counts `checked`'s invalid call in `unknown` (a tool not in the list) or `schema` (a call that
/// fails its tool's schema, a tool's own broken schema included). JSON that does not parse is in
/// neither, and only a reply's first call is checked, as the loop runs it.
fn tally(checked: &Checked, unknown: &mut u32, schema: &mut u32) {
    match checked {
        Checked::Invalid(_, ValidateError::UnknownTool { .. }) => *unknown += 1,
        Checked::Invalid(..) => *schema += 1,
        _ => {}
    }
}

/// Nearest-rank percentiles, as the engine's bench takes them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Percentiles {
    /// The median.
    pub p50: f64,
    /// The 99th percentile.
    pub p99: f64,
}

/// The run's numbers, over the prompts that did not error; the counters over all of them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Summary {
    /// Prompts run.
    pub prompts: usize,
    /// Prompts an engine error ended.
    pub errors: usize,
    /// Share of first replies that parsed.
    pub parse_rate_before: f64,
    /// Share that parsed after the retry.
    pub parse_rate_after: f64,
    /// Share whose last reply parsed and called a tool other than the expected one.
    pub wrong_tool_rate: f64,
    /// Time to first token of the first step.
    pub ttft_ms: Percentiles,
    /// Mean decode speed of the first step.
    pub tok_s: f64,
    /// Prompt tokens reused from parked KV, all steps.
    pub cached_tokens: u64,
    /// All prompts, end to end.
    pub wall_s: f64,
    /// From the start of each turn to the first thing the user sees; `None` when nothing showed,
    /// or in a v1 report.
    #[serde(default)]
    pub visible_ms: Option<Percentiles>,
    /// Limit stops, wrong-but-valid calls and retries, over every prompt; `None` in a v1 report.
    #[serde(default)]
    pub counters: Option<Counters>,
    /// `wrong_tool_rate` of each id group (clear, near, infer, none), over its prompts that did not
    /// error; empty in a report from before it.
    #[serde(default)]
    pub wrong_tool_by_group: BTreeMap<String, f64>,
    /// The counts of each class (`class`), over every prompt; empty in a report from before it.
    #[serde(default)]
    pub by_class: BTreeMap<String, ClassCount>,
    /// The IDs of the rows that errored, missed the tool or failed the answer checks, in file
    /// order, so two runs diff by row; each row says why.
    #[serde(default)]
    pub failing: Vec<String>,
    /// Share of all prompts not in `failing`: no error, the right tool, a passing answer; 0 in a
    /// report from before it.
    #[serde(default)]
    pub right_rate: f64,
}

/// One class's rows and what went wrong in them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassCount {
    /// Rows of the class.
    pub rows: usize,
    /// Of those, the ones an engine error ended.
    pub errors: usize,
    /// The ones that did not error and whose last reply missed the expected tool or answer.
    pub wrong_tool: usize,
    /// The ones whose plain answer failed the answer checks (`Row::wrong_answer`).
    pub wrong_answer: usize,
}

/// The ask-docs measures, from `docs_metrics`; each is `None` when there was nothing to count.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Docs {
    /// Facts whose line is one of their excerpt's / facts emitted.
    pub precision: Option<f64>,
    /// Planted facts the answers cite / planted facts.
    pub recall: Option<f64>,
    /// Questions with no answer in the documents where the answer, notes aside, is only the
    /// sentence that says so / such questions.
    pub abstain_rate: Option<f64>,
    /// The error that ended each question it ended, by question ID; such a question recalls
    /// nothing and does not abstain.
    #[serde(default)]
    pub errors: BTreeMap<String, String>,
    /// Each question that did not error, in the set's order; empty in a v1 report.
    #[serde(default)]
    pub questions: Vec<DocsQuestion>,
}

/// One document question's answer and how it scored.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocsQuestion {
    /// The question's ID.
    pub id: String,
    /// The answer, notes aside.
    pub answer: String,
    /// The facts the answer cites, of those that passed the line check.
    pub cited: Vec<CitedQuote>,
    /// Facts the workers gave.
    pub facts_emitted: usize,
    /// Of those, the ones that passed the line check.
    pub facts_verified: usize,
    /// The planted facts the answer recalls, by their index in the question's `facts`.
    pub recalled: Vec<usize>,
    /// The answer is only the sentence that says the documents do not answer it.
    pub abstained: bool,
    /// The question's run, end to end; 0 when scored without a run.
    pub wall_ms: f64,
    /// What the run came to (`outcome`): right is every planted fact recalled, or an abstention
    /// where none was planted; `None` in a report from before it.
    #[serde(default)]
    pub outcome: Option<Outcome>,
    /// Where its time went; `None` in a report from before it.
    #[serde(default)]
    pub stages: Option<Stages>,
    /// The first line each of its `before` calls returned, in order.
    #[serde(default)]
    pub before: Vec<String>,
    /// Every planted fact recalled, or an abstention where none was planted; false in a report
    /// from before it.
    #[serde(default)]
    pub right: bool,
}

/// A cited fact's place and quote.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CitedQuote {
    /// The file, relative to the folder asked about.
    pub file: PathBuf,
    /// The line the quote starts on.
    pub line: u32,
    /// The words the fact quotes.
    pub quote: String,
}

/// The prompts file a report ran.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptsFile {
    /// Its path as given.
    pub file: String,
    /// SHA-256 of its bytes, in hex.
    pub sha256: String,
}

/// One `larch-agent bench` run: `bench/results/agent-<machine>-<alias>.json`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    /// `SCHEMA`.
    pub schema: String,
    /// The model's alias in bench/models.toml.
    pub model: String,
    /// The machine's name.
    pub machine: String,
    /// The commit checked out when the bench ran, in the folder the binary was built from (a
    /// commit made there after the build changes it), `-dirty` with local edits; `None` outside a
    /// git checkout.
    pub commit: Option<String>,
    /// When the run started, UTC.
    pub started_at: String,
    /// A `--quick` run over a fixed subset of each section; never kept in bench/results/.
    #[serde(default)]
    pub quick: bool,
    /// The tool-choice router the run used.
    #[serde(default)]
    pub router: Router,
    /// The logprob gate's margin in nats (`--router-margin`); `None` without the gate, or in a
    /// report from before it.
    #[serde(default)]
    pub router_margin: Option<f32>,
    /// The run showed the tools under fixed names that mean nothing (`--rename-tools`), a
    /// diagnostic whose rows go next to a run with the real names; the rows name the real tools.
    #[serde(default)]
    pub rename_tools: bool,
    /// Tools no row needs, shown after each row's own (`--extra-tools`): how far the tool choice
    /// falls as the list grows.
    #[serde(default)]
    pub extra_tools: usize,
    /// The rows started from `app_prompt` (`--app-prompt`), not the bare system prompt.
    #[serde(default)]
    pub app_prompt: bool,
    /// The tool rows ran as chat turns (`--turn`, `run_turns`).
    #[serde(default)]
    pub turn: bool,
    /// The chats opened with the desktop app's options (`--app`): each setting where they differ
    /// from the bench's (`differs`); `None` without `--app`.
    #[serde(default)]
    pub app: Option<Vec<String>>,
    /// Tool steps sampled under the call grammar (`--no-tool-grammar` turns it off), which
    /// `counters.unknown_tools` and `counters.schema_fails` are read by.
    #[serde(default)]
    pub tool_grammar: bool,
    /// The chat steps' thinking budget (`--thinking-budget`; 0 is off); `None` in a report from
    /// before the budget.
    #[serde(default)]
    pub thinking_budget: Option<u32>,
    /// The answers' cap (`--answer-max-tokens`; 0 is none); `None` in a report from before it.
    #[serde(default)]
    pub answer_max_tokens: Option<u32>,
    /// Chunks each document question gave its workers (`--docs-workers`); `None` without the
    /// docs section, or in a report from before it.
    #[serde(default)]
    pub docs_workers: Option<usize>,
    /// Tokens of a chunk's file a docs worker read on each side of it (`--docs-window`); `None`
    /// without the docs section, or in a report from before it.
    #[serde(default)]
    pub docs_window: Option<u32>,
    /// The prompts.
    pub prompts: PromptsFile,
    /// The numbers.
    pub summary: Summary,
    /// The multi-step rows; `null` when not measured.
    #[serde(default)]
    pub multistep: Option<MultiStep>,
    /// The ask-docs numbers; `null` when not measured.
    #[serde(default)]
    pub docs: Option<Docs>,
    /// The memory set's numbers; `null` when not measured.
    #[serde(default)]
    pub memory: Option<Memory>,
    /// The engine's prefix cache counters at the end of each section run, in run order.
    #[serde(default)]
    pub prefix_cache: Vec<PrefixCache>,
    /// The engine's KV pool in blocks of 16 tokens, which bounds the prefix cache; `None` in a
    /// report from before it.
    #[serde(default)]
    pub kv_pool_blocks: Option<u32>,
    /// The blocks the prefix cache may keep when the model loaded; 0 is off; `None` in a report
    /// from before it.
    #[serde(default)]
    pub prefix_cache_blocks: Option<usize>,
    /// The wall time of the anchors sent before the rows, which no row's timings hold; `null`
    /// when none ran, with the cache off.
    #[serde(default)]
    pub anchor_ms: Option<f64>,
    /// The replay set's answers (`--replay`), which run in place of the other sections; `null`
    /// when not measured.
    #[serde(default)]
    pub replay: Option<Replay>,
    /// The same questions on the live web (`--live`), kept apart from `replay` and never summed
    /// with it; `null` when not measured.
    #[serde(default)]
    pub live: Option<Replay>,
    /// The everyday set's runs (`--everyday`), which run in place of the other sections; `null`
    /// when not measured.
    #[serde(default)]
    pub everyday: Option<Everyday>,
    /// The memory chat set's cases (`--memory-chats`), which run in place of the other
    /// sections; `null` when not measured.
    #[serde(default)]
    pub memory_chats: Option<MemoryChats>,
    /// One per prompt, in file order.
    pub rows: Vec<Row>,
}

/// The engine's prefix cache counters when a section ended, totals since the engine started.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefixCache {
    /// `toolcalls`, `multistep`, `docs` or `memory`.
    pub section: String,
    /// Blocks of 16 tokens the cache may keep; 0 is off.
    pub capacity_blocks: usize,
    /// Prompts that looked in the cache.
    pub lookups: u64,
    /// Prompts that reused a cached prefix.
    pub hits: u64,
    /// Prompt tokens reused, from the cache or a session's own KV.
    pub reused_prompt_tokens: u64,
    /// Prompt tokens computed.
    pub computed_prompt_tokens: u64,
}

/// Records the prefix cache counters at the end of `section` in `report` and says them.
fn cache_stats(
    section: &str,
    stats: &impl Fn() -> Option<PrefixCacheStats>,
    report: &mut Report,
    progress: &mut impl FnMut(String),
) {
    let Some(s) = stats() else {
        progress(format!(
            "prefix cache after {section}: no counters, the engine failed"
        ));
        return;
    };
    progress(format!(
        "prefix cache after {section}: {} hits, {} reused prompt tokens",
        s.hits, s.reused_prompt_tokens
    ));
    report.prefix_cache.push(PrefixCache {
        section: section.into(),
        capacity_blocks: s.capacity_blocks,
        lookups: s.lookups,
        hits: s.hits,
        reused_prompt_tokens: s.reused_prompt_tokens,
        computed_prompt_tokens: s.computed_prompt_tokens,
    });
}

impl Report {
    /// Whether an error ended a row or a question of any section, or a note line got no keys.
    pub fn failed(&self) -> bool {
        self.summary.errors > 0
            || self
                .multistep
                .as_ref()
                .is_some_and(|m| m.summary.errors > 0)
            || self.docs.as_ref().is_some_and(|d| !d.errors.is_empty())
            || (self.memory.as_ref()).is_some_and(|m| {
                m.questions.iter().any(|r| r.error.is_some())
                    || m.keys.as_ref().is_some_and(|k| !k.errors.is_empty())
            })
            || (self.everyday.as_ref()).is_some_and(|e| e.rows.iter().any(|r| r.error.is_some()))
            || (self.memory_chats.as_ref())
                .is_some_and(|m| m.rows.iter().any(|r| r.error.is_some()))
            || [&self.replay, &self.live].iter().any(|section| {
                (section.as_ref()).is_some_and(|r| r.rows.iter().any(|r| r.error.is_some()))
            })
    }
}

/// Prompts per id group (clear, none, near, infer) in a `--quick` run.
pub const QUICK_PER_GROUP: usize = 10;
/// Multi-step rows in a `--quick` run.
pub const QUICK_STEPS: usize = 10;
/// Document questions in a `--quick` run.
pub const QUICK_DOCS: usize = 6;
/// Memory questions in a `--quick` run.
pub const QUICK_MEMORY: usize = 5;
/// Replay questions in a `--quick` run.
pub const QUICK_REPLAY: usize = 8;

/// `n` of `items`, spread evenly from the first, in order; all of them when there are no more.
pub fn spread<T: Clone>(items: &[T], n: usize) -> Vec<T> {
    if items.len() <= n {
        return items.to_vec();
    }
    (0..n).map(|i| items[i * items.len() / n].clone()).collect()
}

/// The quick subset of a prompts file: `QUICK_PER_GROUP` of each id group, spread over the
/// group, in file order; fixed for a frozen file.
// ponytail: a group is a run of ids with one prefix, as gen.py writes them; collect by prefix if a
// file ever interleaves groups
pub fn quick_prompts(prompts: &[BenchPrompt]) -> Vec<BenchPrompt> {
    let group = |p: &BenchPrompt| p.id.rsplit_once('-').map(|(g, _)| g.to_owned());
    (prompts.chunk_by(|a, b| group(a) == group(b)))
        .flat_map(|members| spread(members, QUICK_PER_GROUP))
        .collect()
}

/// What a run asks, read before the model loads so a broken input fails first.
#[derive(Debug)]
pub struct Inputs {
    /// The tool-call prompts.
    pub prompts: Vec<BenchPrompt>,
    /// The multi-step rows and their file.
    pub steps: Option<(Vec<StepPrompt>, PromptsFile)>,
    /// The document set.
    pub docs: Option<DocSet>,
    /// The memory set.
    pub memory: Option<MemorySet>,
}

impl Inputs {
    /// The `--quick` subset of each section.
    pub fn quick(mut self) -> Inputs {
        self.prompts = quick_prompts(&self.prompts);
        if let Some((rows, _)) = &mut self.steps {
            *rows = spread(rows, QUICK_STEPS);
        }
        if let Some(set) = &mut self.docs {
            set.questions = spread(&set.questions, QUICK_DOCS);
        }
        if let Some(set) = &mut self.memory {
            set.questions = spread(&set.questions, QUICK_MEMORY);
        }
        self
    }
}

/// The system prompt every bench row starts from, and the probe is fitted on: the chat's, with no
/// folders or memory, on `DATE`.
pub fn system_prompt() -> String {
    run::system_prompt("", Some(""), DATE, None, "")
}

/// Where `app_prompt`'s folders read as being, so every run sends the same bytes.
const APP_HOME: &str = "/Users/alex";
/// `app_prompt`'s allowed folders and the files in each.
const APP_FOLDERS: [(&str, &[&str]); 2] = [
    (
        "papers-ml",
        &[
            "attention-is-all-you-need.pdf",
            "lora-low-rank-adaptation.pdf",
            "flash-attention-2.pdf",
            "scaling-laws-for-neural-language-models.pdf",
        ],
    ),
    (
        "notes",
        &["groceries.md", "trip-ideas.md", "reading-list.md"],
    ),
];
/// `app_prompt`'s remembered facts.
const APP_FACTS: [&str; 3] = [
    "The user's name is Alex.",
    "The user lives in Chicago.",
    "The user is allergic to penicillin.",
];

/// The system prompt of a chat with folders and memory (`--app-prompt`): `APP_FOLDERS` and
/// `APP_FACTS` rendered by the chat's own code from a temporary home, which then reads as
/// `APP_HOME`, with the cutoff of a model of GGUF architecture `arch`, as a chat has it.
pub fn app_prompt(arch: &str) -> anyhow::Result<String> {
    let tmp = tempfile::tempdir()?;
    let home = tmp.path().join("home");
    let mut allowed = Vec::new();
    for (folder, files) in APP_FOLDERS {
        let dir = home.join(folder);
        fs::create_dir_all(&dir)?;
        for file in files {
            fs::write(dir.join(file), "")?;
        }
        allowed.push(dir);
    }
    for fact in APP_FACTS {
        memory::remember(tmp.path(), fact, "user", "bench")?;
    }
    // With the separator, so a folder reads /Users/alex/notes on Windows too.
    let real = format!(
        "{}{}",
        crate::tool::plain(fs::canonicalize(&home)?).display(),
        std::path::MAIN_SEPARATOR
    );
    // The listing holds the path as a JSON string, escaped where it has a backslash.
    let quoted = serde_json::to_string(&real)?;
    let to = format!("{APP_HOME}/");
    let shown = |text: &str| (text.replace(&quoted[1..quoted.len() - 1], &to)).replace(&real, &to);
    let f = run::folders(&allowed);
    let memory = run::memory_block(tmp.path());
    Ok(run::system_prompt(
        &shown(&f.block),
        memory.as_deref(),
        DATE,
        run::knows_until(arch),
        &shown(&f.listing),
    ))
}

/// Checks `--thinking-budget` and `--answer-max-tokens` against the loaded context, for a step
/// that shows every chat tool, as `config.toml`'s keys are checked when a chat opens; the error
/// names the flag.
pub fn check_room(larch: &Larch, family: Family, thinking: Thinking) -> Result<(), String> {
    let system = run::system_prompt("", Some(""), DATE, None, "");
    let registry = Registry::builtin();
    run::room(
        larch,
        &system,
        &registry,
        &crate::CHAT_TOOLS,
        Some(family),
        thinking,
    )
    .map_err(|e| e.reason(&format!("--{}", e.key.replace('_', "-"))))
}

/// Runs the sections into `report` in their order, tool rows, multi-step rows, docs, memory, so
/// the timed tool rows run on a GPU the rest has not warmed, and `save`s the report after each,
/// so a run that stops keeps what it measured. First every system prompt and tool list the rows
/// and the memory questions start from goes to the prefix cache in `namespace`, so no row prefills
/// it, unless `stats` shows the cache off; the anchors' time goes in `anchor_ms`. An engine error
/// ends only its row or question, or that anchor; `progress` gets a line for each as it is done,
/// and the `stats` of the prefix cache at the end of each section. The chat steps think on
/// `thinking`, and `routing`'s router goes before each tool and multi-step row.
#[allow(clippy::too_many_arguments)]
pub fn run_all(
    larch: &Larch,
    family: Family,
    thinking: Thinking,
    namespace: u64,
    routing: &Routing,
    inputs: &Inputs,
    report: &mut Report,
    stats: impl Fn() -> Option<PrefixCacheStats>,
    mut save: impl FnMut(&Report) -> anyhow::Result<()>,
    mut progress: impl FnMut(String),
) -> anyhow::Result<()> {
    let registry = routing.registry();
    let steps = inputs.steps.iter().flat_map(|(rows, _)| rows);
    // A turn's chat anchors its own prompt.
    let rows = inputs.prompts.iter().filter(|_| !routing.turn);
    // The rows' lists as the model sees them, renamed or not.
    let mut lists: BTreeSet<Vec<&str>> = (rows.map(|p| &p.tools))
        .chain(steps.map(|p| &p.tools))
        .map(|tools| {
            (tools.iter())
                .map(|t| match routing.rename {
                    true => shown_name(t),
                    false => t.as_str(),
                })
                .chain(routing.extra().iter().map(|e| e.0))
                .collect()
        })
        .collect();
    // A memory question shows no tools, and nor does a row the probe masks.
    if inputs.memory.is_some() || routing.router == Router::Probe {
        lists.insert(Vec::new());
    }
    let system = routing.system.clone();
    let on = stats().is_some_and(|s| s.capacity_blocks > 0);
    if !on {
        lists.clear();
    }
    let started = Instant::now();
    let anchors = lists.len();
    for allow in lists {
        // A row's first step and a memory question are chat steps: the same render as theirs.
        let p = step_params(
            &registry,
            &allow,
            Some(family),
            Kind::Chat,
            thinking,
            namespace,
        );
        if let Err(e) = run::anchor(larch, &system, &p) {
            progress(format!(
                "the system prompt did not reach the prefix cache: {e}"
            ));
        }
    }
    report.anchor_ms = (anchors > 0).then(|| ms(started));
    if let Some(t) = report.anchor_ms {
        progress(format!("{anchors} anchors: {t:.0} ms"));
    }
    let line = |row: &Row| match &row.error {
        Some(e) => format!("{} error: {e}", row.id),
        None => format!("{} {:?}", row.id, row.after),
    };
    // The turn rows' chats as `run_turns` opens them, the data folder aside.
    if routing.app && routing.turn {
        let home = crate::tool::expand_home("~")?;
        let mut lines = chat_options(Path::new(""), vec![home], routing, on)?.1;
        lines.extend(settings_differ(routing, thinking)?);
        for line in &lines {
            progress(format!("--app: {line}"));
        }
        report.app = Some(lines);
    }
    report.rows = match routing.turn {
        true => run_turns(larch, thinking, routing, on, &inputs.prompts, |row| {
            progress(line(row))
        })?,
        false => run_routed(
            larch,
            family,
            thinking,
            namespace,
            routing,
            &inputs.prompts,
            |row| progress(line(row)),
        ),
    };
    report.summary = summary(&report.rows);
    cache_stats("toolcalls", &stats, report, &mut progress);
    save(report)?;
    if let Some((rows, file)) = &inputs.steps {
        let rows = run_steps_routed(larch, family, thinking, namespace, routing, rows, |r| {
            progress(match &r.error {
                Some(e) => format!("{} error: {e}", r.id),
                None => format!("{} steps {:?} answered {}", r.id, r.steps, r.answered),
            })
        });
        report.multistep = Some(MultiStep {
            prompts: file.clone(),
            summary: step_summary(&rows),
            rows,
        });
        cache_stats("multistep", &stats, report, &mut progress);
        save(report)?;
    }
    if let Some(set) = &inputs.docs {
        let workers = Workers {
            prefix_cache: on,
            ..Workers::default()
        };
        let held = routing.tool_grammar;
        let docs = run_docs(
            larch,
            Some(family),
            thinking,
            held,
            set,
            workers,
            |q, run| {
                progress(match run {
                    Ok(run) => format!("{} {}", q.id, run.markdown.replace('\n', " ")),
                    Err(e) => format!("{} error: {e}", q.id),
                })
            },
        )?;
        report.docs = Some(docs);
        cache_stats("docs", &stats, report, &mut progress);
        save(report)?;
    }
    if let Some(set) = &inputs.memory {
        let keys = set
            .keys
            .then(|| memory_keys(larch, family, namespace, &Workers::default(), set));
        let memory = run_memory(Some((larch, family, namespace)), thinking, set, |r| {
            progress(match &r.error {
                Some(e) => format!("{} error: {e}", r.id),
                None => format!("{} found {} holds {:?}", r.id, r.found, r.holds_value),
            })
        });
        report.memory = Some(Memory { keys, ..memory });
        cache_stats("memory", &stats, report, &mut progress);
        save(report)?;
    }
    Ok(())
}

/// The commit checked out in `dir` now, not the one a binary was built from: the full ID,
/// `-dirty` with local edits, and `None` outside a git checkout.
pub fn commit(dir: &Path) -> Option<String> {
    // Tags excluded, so it is the full commit ID.
    let describe = [
        "describe",
        "--always",
        "--dirty",
        "--abbrev=40",
        "--exclude=*",
    ];
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(describe)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// Reads a prompts file, one JSON prompt per line, and its SHA-256; a file with none is an error.
pub fn load(path: &Path) -> anyhow::Result<(Vec<BenchPrompt>, PromptsFile)> {
    lines(path)
}

/// Reads a multi-step rows file as `load` reads a prompts file.
pub fn load_steps(path: &Path) -> anyhow::Result<(Vec<StepPrompt>, PromptsFile)> {
    lines(path)
}

fn lines<T: DeserializeOwned>(path: &Path) -> anyhow::Result<(Vec<T>, PromptsFile)> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = std::str::from_utf8(&bytes).context("the prompts file is not UTF-8")?;
    let prompts: Vec<T> = text
        .lines()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line).with_context(|| format!("prompt line {}", i + 1))
        })
        .collect::<anyhow::Result<_>>()?;
    anyhow::ensure!(!prompts.is_empty(), "{} has no prompts", path.display());
    let file = PromptsFile {
        file: path.display().to_string(),
        sha256: larch::bench::hex(&larch::bench::sha256(&bytes)),
    };
    Ok((prompts, file))
}

/// Runs each prompt on a fresh session with its tools, as the loop's first step and its one
/// retry, and scores both, with the router off; `progress` sees each row as it is done.
pub fn run(
    larch: &Larch,
    family: Family,
    thinking: Thinking,
    namespace: u64,
    prompts: &[BenchPrompt],
    progress: impl FnMut(&Row),
) -> Vec<Row> {
    let routing = Routing::default();
    run_routed(
        larch, family, thinking, namespace, &routing, prompts, progress,
    )
}

/// `run` with `routing`'s router in front of each row's first step and its tool names.
pub fn run_routed(
    larch: &Larch,
    family: Family,
    thinking: Thinking,
    namespace: u64,
    routing: &Routing,
    prompts: &[BenchPrompt],
    mut progress: impl FnMut(&Row),
) -> Vec<Row> {
    let registry = routing.registry();
    let starts = router::starts(family, |text| larch.token_ids(text));
    let mut rows = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let started = Instant::now();
        let mut row = Row {
            id: prompt.id.clone(),
            ..Row::default()
        };
        let prompt = routing.shown_prompt(prompt);
        let ask = Ask {
            registry: &registry,
            allow: prompt.tools.iter().map(String::as_str).collect(),
            family,
            thinking,
            namespace,
            routing,
            starts: &starts,
        };
        if let Err(e) = one(larch, &ask, &prompt, &mut row) {
            row.error = Some(e.to_string());
        }
        row.tool = row.tool.map(|t| real_name(&t).to_owned());
        row.wall_ms = ms(started);
        progress(&row);
        rows.push(row);
    }
    rows
}

fn one(larch: &Larch, ask: &Ask, prompt: &BenchPrompt, row: &mut Row) -> anyhow::Result<()> {
    let opts = SessionOptions {
        system: Some(ask.routing.system.clone()),
        ..SessionOptions::default()
    };
    let mut session = larch.session(opts.clone());
    let (earlier, result) = match prompt.history.split_last() {
        Some((last, earlier)) if last.role == Role::Tool => (earlier, Some(last)),
        _ => (prompt.history.as_slice(), None),
    };
    if !earlier.is_empty() {
        let mut history = session.history();
        history.extend(earlier.iter().cloned());
        session.restore(opts, history)?;
    }
    let started = Instant::now();
    let (mut step, masked) = match result {
        // ponytail: no router in front of a row that resumes from a tool result; add it when
        // such a row runs with the router on.
        Some(m) => {
            let p = ask.params(Kind::Answer, false, false);
            let stream = session.send_tool_result(&m.tool_call_id, &m.content, p);
            (read_step(stream, Some(ask.family))?, false)
        }
        None => ask.first(larch, &mut session, &prompt.user, &mut row.routed)?,
    };
    let allow: &[&str] = if masked { &[] } else { &ask.allow };
    let tools = !allow.is_empty();
    loop {
        row.prompt_tokens += step.usage.prompt_tokens;
        row.generated_tokens += step.usage.completion_tokens;
        row.cached_tokens += step.usage.cached_tokens;
        row.cut.extend(step.cut);
        let checked = run::check(&step.parsed, ask.registry, allow);
        tally(&checked, &mut row.unknown_tools, &mut row.schema_fails);
        row.visible_ms = row.visible_ms.or_else(|| step.seen(&checked, started));
        // Scored as the loop runs it: `start_line: "2"` is the 2 the tool gets.
        let ran = as_run(&step.parsed, &checked);
        row.after = score_toolcall(&prompt.expect, &ran);
        row.wrong_answer = score_answer(&prompt.expect, &ran);
        row.tool = ran.call.as_ref().map(|c| c.name.clone());
        row.reply = Some(reply_of(&ran));
        let said = ran.text_before.to_lowercase().replace('\u{2019}', "'");
        row.offline_claim = row.tool.is_none() && OFFLINE.iter().any(|o| said.contains(o));
        if !row.retried {
            (row.ttft_ms, row.tok_s) = (step.timings.ttft_ms, step.timings.tok_s);
            row.before = row.after;
        }
        let mode = run::retry_mode(&checked);
        let stream = match checked {
            Checked::Answer => return Ok(()),
            Checked::Call(..) => {
                row.wrong_but_valid = !row.after.args_ok;
                return Ok(());
            }
            // The loop's one retry is spent.
            _ if row.retried => return Ok(()),
            Checked::ParseError(e) => {
                session.send(&run::retry_text(&e, tools), ask.retry(mode, masked))
            }
            Checked::Invalid(call, e) => {
                let id = format!("{:09}", 1);
                session.set_tool_calls(&step.parsed.text_before, run::model_call(&call, &id))?;
                session.send_tool_result(&id, &e.to_string(), ask.retry(mode, masked))
            }
        };
        row.retried = true;
        step = read_step(stream, tools.then_some(ask.family))?;
    }
}

/// Runs each row as a chat's turn after its planted history (`Routing::turn`): the agent's own
/// system prompt, turn notes, forced calls and reply checks, in a throwaway data folder whose
/// config.toml holds the run's settings (with `routing.app`, the user's: `write_config`). A row
/// whose tools hold a file tool has the home folder allowed, and one without (`chat_tools`) is a
/// chat with nothing attached, whose right reply may say it cannot read the files, so it is scored
/// without the refusal check (with `routing.app`, the app's folders either way). A row is scored
/// on the turn's first call or its answer. The call does not run, unless it is a read the turn
/// runs before it shows the call (a file tool, on the home folder); `progress` sees each row as
/// it is done.
/// The model's probe, copied from the user's data folder into a bench run's new one, so
/// config.toml's `router = "probe"` finds it there.
fn copy_probe(data: &Path, routing: &Routing) -> anyhow::Result<()> {
    if let (Some(sha), Some(from)) = (&routing.model_sha256, &routing.config) {
        let src = probe::path(from, sha);
        if src.exists() {
            let dst = probe::path(data, sha);
            fs::create_dir_all(dst.parent().context("a probe path has a folder")?)?;
            fs::copy(&src, &dst).with_context(|| format!("copying {}", src.display()))?;
        }
    }
    Ok(())
}

pub fn run_turns(
    larch: &Larch,
    thinking: Thinking,
    routing: &Routing,
    prefix_cache: bool,
    prompts: &[BenchPrompt],
    mut progress: impl FnMut(&Row),
) -> anyhow::Result<Vec<Row>> {
    let data = tempfile::tempdir().context("making a data folder for the turn rows")?;
    let app = routing.config.as_deref().filter(|_| routing.app);
    write_config(data.path(), routing, thinking, app)?;
    copy_probe(data.path(), routing)?;
    let home = crate::tool::expand_home("~")?;
    let mut rows = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let allowed = match files(&prompt.tools) {
            true => vec![home.clone()],
            false => Vec::new(),
        };
        let opts = chat_options(data.path(), allowed, routing, prefix_cache)?.0;
        let started = Instant::now();
        let mut row = Row {
            id: prompt.id.clone(),
            ..Row::default()
        };
        if let Err(e) = turn(
            larch,
            &opts,
            routing.xml_body,
            prompt,
            &mut row,
            (routing.outside().into_iter())
                .chain(routing.mcp.iter().flat_map(|m| m.tools(&prompt.servers)))
                .collect(),
            routing.flow,
            routing.whole,
            routing.model_sha256.clone(),
        ) {
            row.error = Some(format!("{e:#}"));
        }
        row.wall_ms = ms(started);
        progress(&row);
        rows.push(row);
    }
    Ok(rows)
}

/// The options a bench chat opens with in `data_dir`: the bench's (`allowed`, `routing`'s
/// router, the prefix cache as `prefix_cache`), or with `routing.app` the desktop app's
/// (`AgentOptions::app`, with the home folder allowed as the app has it); and the settings where
/// the app's differ from the bench's (`differs`), empty without `routing.app`.
pub fn chat_options(
    data_dir: &Path,
    allowed: Vec<PathBuf>,
    routing: &Routing,
    prefix_cache: bool,
) -> anyhow::Result<(AgentOptions, Vec<String>)> {
    let ours = AgentOptions {
        data_dir: data_dir.into(),
        allowed: allowed.clone(),
        arch: routing.arch.clone(),
        router: routing.router,
        prefix_cache,
        memory_capture: routing.capture,
    };
    if !routing.app {
        return Ok((ours, Vec::new()));
    }
    // The app allows the chat's attached folders only; `--flow` passes the row's.
    let allowed = match routing.flow {
        true => allowed,
        false => vec![crate::tool::expand_home("~")?],
    };
    let app = AgentOptions::app(data_dir.into(), allowed, routing.arch.clone());
    let lines = differs(&ours, &app);
    Ok((app, lines))
}

/// Each setting where `app` differs from `bench`, one line each: `allowed`, `router`,
/// `prefix_cache` (the data folder and the architecture are the run's own).
pub fn differs(bench: &AgentOptions, app: &AgentOptions) -> Vec<String> {
    let mut lines = Vec::new();
    if app.allowed != bench.allowed {
        lines.push(format!(
            "allowed: {:?} (bench: {:?})",
            app.allowed, bench.allowed
        ));
    }
    if app.router != bench.router {
        lines.push(format!(
            "router: {:?} (bench: {:?})",
            app.router, bench.router
        ));
    }
    if app.prefix_cache != bench.prefix_cache {
        lines.push(format!(
            "prefix_cache: {} (bench: {})",
            app.prefix_cache, bench.prefix_cache
        ));
    }
    if app.memory_capture != bench.memory_capture {
        lines.push(format!(
            "memory_capture: {} (bench: {})",
            app.memory_capture, bench.memory_capture
        ));
    }
    lines
}

/// Whether `tools` hold one of `FILE_TOOLS`.
fn files(tools: &[String]) -> bool {
    tools
        .iter()
        .any(|t| crate::FILE_TOOLS.contains(&t.as_str()))
}

/// Whether a `--turn` row's `tools` are a chat's: `CHAT_TOOLS`, or those without `FILE_TOOLS`,
/// as a chat with nothing attached shows them (bench/agent/no-folders.jsonl).
pub fn chat_tools(tools: &[String]) -> bool {
    let shown = |attached: bool| {
        (crate::CHAT_TOOLS.iter())
            .filter(move |t| attached || !crate::FILE_TOOLS.contains(t))
            .copied()
    };
    let tools = || tools.iter().map(String::as_str);
    tools().eq(shown(true)) || tools().eq(shown(false))
}

/// The confirms a `--whole` turn answers before the bench stops it.
const MAX_CONFIRMS: usize = 8;

/// Recorded MCP servers (`fixture.json`: each server's id and its tools as `tools/list` gave them,
/// with the name and description the agent shows) and a scripted result per tool
/// (`results.json`), which `--mcp-fixture <dir>` reads; a row's `servers` installs them.
#[derive(Debug, Default)]
pub struct McpFixture {
    /// Server id -> its tools' specs, as `mcp::client::McpTool` builds them.
    pub servers: BTreeMap<String, Vec<ToolSpec>>,
    /// `<server>__<tool>` -> what a call returns, and the argument rules whose error it returns
    /// instead (the failure set's strict servers).
    pub results: BTreeMap<String, ScriptedResult>,
}

/// A scripted tool's reply: `result`, unless an argument breaks one of `rules`.
#[derive(Clone, Debug, Default)]
pub struct ScriptedResult {
    pub result: String,
    pub rules: Vec<(String, regex::Regex, String)>,
}

impl ScriptedResult {
    fn of(v: &Value) -> anyhow::Result<ScriptedResult> {
        if let Some(text) = v.as_str() {
            return Ok(ScriptedResult {
                result: text.into(),
                rules: Vec::new(),
            });
        }
        let rules = (v["rules"].as_array().into_iter().flatten())
            .map(|r| {
                let arg = r["arg"]
                    .as_str()
                    .context("a rule without an arg")?
                    .to_owned();
                let re = regex::Regex::new(r["regex"].as_str().context("a rule without a regex")?)?;
                let error = r["error"]
                    .as_str()
                    .unwrap_or("invalid arguments")
                    .to_owned();
                Ok((arg, re, error))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(ScriptedResult {
            result: v["result"].as_str().unwrap_or_default().to_owned(),
            rules,
        })
    }

    /// The reply to a call with `args`: the first broken rule's error, or the result.
    fn reply(&self, args: &Value) -> Result<String, crate::tool::ToolError> {
        for (arg, re, error) in &self.rules {
            let value = match &args[arg.as_str()] {
                Value::String(s) => s.clone(),
                Value::Null => String::new(),
                other => other.to_string(),
            };
            if !re.is_match(&value) {
                return Err(crate::tool::ToolError::Outside(
                    error.replace("{value}", &value),
                ));
            }
        }
        Ok(self.result.clone())
    }
}

impl McpFixture {
    pub fn load(dir: &Path) -> anyhow::Result<McpFixture> {
        let read = |name: &str| -> anyhow::Result<Value> {
            let path = dir.join(name);
            let text = fs::read_to_string(&path).with_context(|| format!("{}", path.display()))?;
            Ok(serde_json::from_str(&text)?)
        };
        let (fixture, results) = (read("fixture.json")?, read("results.json")?);
        let mut servers = BTreeMap::new();
        for s in fixture["servers"].as_array().into_iter().flatten() {
            let id = s["id"]
                .as_str()
                .context("a server without an id")?
                .to_owned();
            let specs = (s["tools"].as_array().into_iter().flatten())
                .map(|t| {
                    let mut schema = t["inputSchema"].clone();
                    if !schema.is_object() {
                        schema = json!({"type": "object", "properties": {}});
                    }
                    if let Some(o) = schema.as_object_mut() {
                        o.entry("type").or_insert_with(|| "object".into());
                    }
                    let read_only = t["annotations"]["readOnlyHint"] == json!(true)
                        && t["annotations"]["openWorldHint"] == json!(false);
                    // The compiled form of the server's own description, as McpTool makes it.
                    let description = crate::mcp::describe::compile(
                        t["description"].as_str().unwrap_or_default(),
                        crate::mcp::describe::MAX_CHARS,
                    );
                    let schema = crate::mcp::describe::compile_schema(&schema);
                    ToolSpec {
                        name: t["agent_name"].as_str().unwrap_or_default().to_owned(),
                        description,
                        schema,
                        permission: match read_only {
                            true => Permission::Read,
                            false => Permission::WriteExternal,
                        },
                        untrusted_results: true,
                        max_result_tokens: 2048,
                        max_args_tokens: 512,
                    }
                })
                .collect();
            servers.insert(id, specs);
        }
        let results = (results.as_object().into_iter().flatten())
            .map(|(k, v)| Ok((k.clone(), ScriptedResult::of(v)?)))
            .collect::<anyhow::Result<_>>()?;
        Ok(McpFixture { servers, results })
    }

    /// The outside tools of `servers`, in that order, each returning its scripted result.
    fn tools(&self, servers: &[String]) -> Vec<Box<dyn Tool>> {
        (servers.iter())
            .flat_map(|s| self.servers.get(s).into_iter().flatten())
            .map(|spec| {
                let result = self.results.get(&spec.name).cloned().unwrap_or_default();
                Box::new(Scripted {
                    spec: spec.clone(),
                    result,
                }) as Box<dyn Tool>
            })
            .collect()
    }
}

/// A recorded MCP tool that returns its scripted result, or its rules' error.
struct Scripted {
    spec: ToolSpec,
    result: ScriptedResult,
}

impl Tool for Scripted {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn call(
        &self,
        args: &Value,
        _: &crate::tool::ToolCtx,
    ) -> Result<String, crate::tool::ToolError> {
        self.result.reply(args)
    }
}

/// An `EXTRA` tool: no row's right call, so a call scores as the wrong tool; it reads and returns
/// nothing.
fn extra_tool(name: &str, description: &str, args: &[&str]) -> Box<dyn Tool> {
    let properties: Map<String, Value> = (args.iter())
        .map(|a| ((*a).into(), json!({"type": "string"})))
        .collect();
    let schema = json!({"type": "object", "properties": properties, "required": args});
    Builtin::boxed(name, description, Permission::Read, 64, schema, |_, _| {
        Ok(String::new())
    })
}

/// One row as a chat's turn: its planted history in a new chat, then its message; the turn stops
/// at its first call. `outside` joins the chat's tools (`--extra-tools`).
#[allow(clippy::too_many_arguments)]
fn turn(
    larch: &Larch,
    opts: &AgentOptions,
    xml_body: bool,
    prompt: &BenchPrompt,
    row: &mut Row,
    outside: Vec<Box<dyn Tool>>,
    flow: bool,
    whole: bool,
    model_sha256: Option<String>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        chat_tools(&prompt.tools),
        "a turn shows the chat's tools, not the row's"
    );
    anyhow::ensure!(
        prompt.history.last().is_none_or(|m| m.role != Role::Tool),
        "a turn starts from the user's message, not a tool result"
    );
    let mut agent = Agent::new_with(larch, opts.clone(), outside)?;
    agent.set_model_sha256(model_sha256);
    agent.set_xml_body(xml_body);
    agent.plant(prompt.history.clone())?;
    let sent = Instant::now();
    let turn = match flow {
        true => agent.turn(&prompt.user)?,
        false => agent.turn_with(larch, &prompt.user)?,
    };
    // After the chat's anchors, which a later turn of the app would not wait for; with `flow`
    // from the message, as the app's own steps before the turn count too.
    let mut heard = Heard::new();
    if flow {
        heard.started = sent;
    }
    let mut parked = listen(turn, whole, &mut heard);
    // With `whole`, an outside call (scripted) runs and a local write is declined, so the
    // fixture folder never changes; the turn goes on to its end.
    let mut confirms = 0;
    while let Some(call) = parked.filter(|_| whole && confirms < MAX_CONFIRMS) {
        confirms += 1;
        parked = listen(
            agent.confirm(crate::is_outside(&call.name))?,
            whole,
            &mut heard,
        );
    }
    (row.visible_ms, row.withdrawn) = (heard.visible_ms, heard.withdrawn);
    row.calls_made = heard.calls.len() as u32;
    row.calls = heard.calls.iter().map(|c| c.name.clone()).collect();
    row.failed_results = (heard.results.iter())
        .filter(|r| r.starts_with("failed"))
        .count() as u32;
    // A whole turn is scored on the first call that was the expected one, else its last call.
    let pick = match whole {
        true => {
            let right = (heard.calls.iter()).position(|c| {
                let ran = Parsed {
                    call: Some(c.clone()),
                    ..Parsed::default()
                };
                let s = score_toolcall(&prompt.expect, &ran);
                s.right_tool && s.args_ok
            });
            row.right_at = right.map(|i| i as u32 + 1);
            right.or(heard.calls.len().checked_sub(1))
        }
        false => Some(0),
    };
    let (said, call) = (
        heard.said,
        pick.and_then(|i| heard.calls.into_iter().nth(i)),
    );
    // A turn that failed ends with its notice and no answer.
    if call.is_none() && !heard.ended {
        anyhow::bail!(
            "{}",
            (heard.notice).unwrap_or_else(|| "the turn ended without an answer".into())
        );
    }
    // The turn's model steps, from the chat's audit lines: the speed of the first that ran to its
    // end (a withdrawn step has none), every step's tokens.
    let steps = steps_of(&agent)?;
    if let Some(e) = (logged(agent.audit.dir())?.into_iter()).find(|e| {
        e.step.as_deref() == Some("router probe") && e.chat.as_deref() == Some(agent.chat_id())
    }) && let Some(d) = e.detail
    {
        row.probe = d["probability"].as_f64().map(|p| p as f32);
        row.probe_decision = d["decision"].as_str().map(String::from);
    }
    if let Some(first) = steps.iter().find(|s| s.ttft_ms.is_some()) {
        row.ttft_ms = first.ttft_ms.unwrap_or_default();
        row.tok_s = first.tok_s.unwrap_or_default();
    }
    for s in &steps {
        row.prompt_tokens += s.prompt_tokens.unwrap_or_default();
        row.cached_tokens += s.cached_tokens.unwrap_or_default();
        row.generated_tokens += s.generated_tokens.unwrap_or_default();
    }
    let ran = Parsed {
        text_before: said,
        call,
        ..Parsed::default()
    };
    // With nothing attached, "I can't read your files; attach a folder" is right: the same tool
    // choice, without the refusal check.
    let mut expect = prompt.expect.clone();
    if !files(&prompt.tools) && expect.tool_any.is_empty() {
        expect.tool_any = vec![expect.tool.clone()];
    }
    row.before = score_toolcall(&expect, &ran);
    row.after = row.before;
    row.wrong_answer = score_answer(&expect, &ran);
    row.wrong_but_valid = ran.call.is_some() && !row.after.args_ok;
    row.reply = Some(reply_of(&ran));
    row.tool = ran.call.map(|c| c.name);
    let said = ran.text_before.to_lowercase().replace('\u{2019}', "'");
    row.offline_claim = row.tool.is_none() && OFFLINE.iter().any(|o| said.contains(o));
    Ok(())
}

/// What a chat turn showed, as `listen` heard it.
struct Heard {
    /// When the turn started, after its chat's anchors.
    started: Instant,
    /// The text shown, from the last correction on.
    said: String,
    /// The calls made, in order.
    calls: Vec<ToolCall>,
    /// The first line of each call's result, in order.
    results: Vec<String>,
    /// From `started` to the first text or call line shown.
    visible_ms: Option<f64>,
    /// A check took back a reply (`Notice("Correction: ...")`).
    withdrawn: bool,
    /// The turn reached `Done`.
    ended: bool,
    /// The last other notice: a failed turn's reason.
    notice: Option<String>,
}

impl Heard {
    fn new() -> Heard {
        Heard {
            started: Instant::now(),
            said: String::new(),
            calls: Vec::new(),
            results: Vec::new(),
            visible_ms: None,
            withdrawn: false,
            ended: false,
            notice: None,
        }
    }
}

/// Reads `turn`'s events into `heard`: up to its first call, or with `whole` to its end, which is
/// the write or research plan it parks on (returned, for `Agent::confirm`) or `Done`.
fn listen(
    turn: impl IntoIterator<Item = AgentEvent>,
    whole: bool,
    heard: &mut Heard,
) -> Option<ToolCall> {
    for event in turn {
        let shows = match &event {
            AgentEvent::Text(t) => !t.is_empty(),
            AgentEvent::ToolCall { .. } => true,
            _ => false,
        };
        if shows && heard.visible_ms.is_none() {
            heard.visible_ms = Some(ms(heard.started));
        }
        match event {
            AgentEvent::Text(t) => heard.said.push_str(&t),
            // The app replaces what showed with the answer after it.
            AgentEvent::Notice(n) if n.starts_with("Correction:") => {
                heard.withdrawn = true;
                heard.said.clear();
            }
            AgentEvent::Notice(n) => heard.notice = Some(n),
            // A write's call comes first too; its confirm never comes, as the turn stops here.
            AgentEvent::ToolCall { name, args } => {
                heard.calls.push(ToolCall { name, args });
                if !whole {
                    break;
                }
            }
            AgentEvent::ToolResult { summary, .. } => {
                heard
                    .results
                    .push(summary.lines().next().unwrap_or_default().into());
            }
            AgentEvent::NeedsConfirm { call } => return Some(call),
            AgentEvent::Done(_) => heard.ended = true,
            // A running tool's working line is the app's, not the reply.
            AgentEvent::Progress(_) | AgentEvent::Documents(_) | AgentEvent::Thinking(_) => {}
        }
    }
    None
}

/// The model steps `agent`'s chat logged, in order.
fn steps_of(agent: &Agent) -> anyhow::Result<Vec<AuditEvent>> {
    let mut steps = logged(agent.audit.dir())?;
    steps.retain(|e| e.kind == EventKind::Turn && e.chat.as_deref() == Some(agent.chat_id()));
    Ok(steps)
}

/// Every event of the audit log in `dir`, in order.
fn logged(dir: &Path) -> anyhow::Result<Vec<AuditEvent>> {
    let mut files: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(entries) => entries
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<_, _>>()?,
        // A turn Rust answered alone logs no step.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(e.into()),
    };
    // YYYY-MM-DD names sort by day.
    files.sort();
    let mut steps = Vec::new();
    for file in files {
        for line in fs::read_to_string(&file)?.lines() {
            // A torn line, from a write cut short, is skipped as the log's other readers do.
            if let Ok(event) = serde_json::from_str::<AuditEvent>(line) {
                steps.push(event);
            }
        }
    }
    Ok(steps)
}

/// What a bench run puts in front of each row's first step, as `larch-agent chat` does with
/// config.toml's `router`, whether the tools go by meaningless names, and the call grammar.
#[derive(Debug)]
pub struct Routing {
    /// The router.
    pub router: Router,
    /// The logprob gate's `router_margin`, in nats.
    pub margin: f32,
    /// The model's probe, which the probe router reads; without it, it gates on logprobs.
    pub probe: Option<Probe>,
    /// `--rename-tools`: the model sees `RENAMED`'s names, the rows the real ones.
    pub rename: bool,
    /// `--extra-tools`: this many of `EXTRA` follow each row's tools.
    pub extra_tools: usize,
    /// Tool steps sample under the call grammar, as config.toml's `tool_grammar`: off in
    /// `default`, whose callers script replies the grammar would not take; the CLI turns it on
    /// unless `--no-tool-grammar`, as the app has it.
    pub tool_grammar: bool,
    /// The template asks for the XML tool-call body (`Agent::set_xml_body`).
    pub xml_body: bool,
    /// The system prompt the rows start from: `system_prompt`, or `app_prompt`.
    pub system: String,
    /// `--turn`: the tool rows run as chat turns (`run_turns`), on the agent's own prompt.
    pub turn: bool,
    /// The model's architecture, which a chat's prompt and calls are built for.
    pub arch: String,
    /// `--app`: the chats open with the desktop app's options (`AgentOptions::app`), not the
    /// bench's.
    pub app: bool,
    /// `--flow`: each turn row goes as the desktop app sends a message with no mode picked:
    /// `Agent::turn` (with, when on, the outside-server pick), with only the folder a chat would
    /// have attached (the home folder for a row with file tools, none otherwise), and its times
    /// from the message.
    pub flow: bool,
    /// `--mcp-fixture`: recorded MCP servers a row's `servers` installs, with scripted results.
    pub mcp: Option<std::sync::Arc<McpFixture>>,
    /// `--whole`: a turn row runs to its end, outside calls allowed (scripted) and local writes
    /// declined, and scores right when any call it made was the expected one.
    pub whole: bool,
    /// The user's data folder, whose config.toml the everyday chats (and the turn chats with
    /// `app`) start from (`write_config`); `None` starts from nothing.
    pub config: Option<PathBuf>,
    /// The chats capture facts in the background (`AgentOptions::memory_capture`), unless the
    /// config.toml says `memory_capture = false`: on in the memory chat bench, as the app has it;
    /// off elsewhere, so a row's numbers hold no capture.
    pub capture: bool,
    /// The model file's SHA-256, which names its probe (`probe::path`): each turn chat gets it as
    /// the chat command gives it (`Agent::set_model_sha256`), so a config.toml `router = "probe"`
    /// (the app's way) reads the probe from `config`'s folder.
    pub model_sha256: Option<String>,
}

impl Default for Routing {
    fn default() -> Self {
        Self {
            system: system_prompt(),
            router: Router::Off,
            margin: router::MARGIN,
            probe: None,
            rename: false,
            extra_tools: 0,
            tool_grammar: false,
            xml_body: false,
            turn: false,
            arch: String::new(),
            app: false,
            flow: false,
            mcp: None,
            whole: false,
            config: None,
            capture: false,
            model_sha256: None,
        }
    }
}

/// The names a `--rename-tools` run shows for the real tools: fixed strings that mean nothing,
/// so a run says how much of the model's choice the names drive.
const RENAMED: [(&str, &str); 12] = [
    ("list_dir", "qv7_tarn"),
    ("read_file", "wx2_plom"),
    ("find_files", "gs3_vilo"),
    ("search_files", "kp5_rado"),
    ("write_file", "zk9_bire"),
    ("recall", "hj4_suto"),
    ("remember", "fd8_mave"),
    ("forget", "lc3_neki"),
    ("changed_files", "rp6_dova"),
    ("diff_file", "ty5_gume"),
    ("show_file", "bn1_kesa"),
    ("log", "mz0_fipu"),
];

/// The tools `--extra-tools` adds, in the order it takes them: name, description and required
/// string arguments. A calendar's, a mailbox's and other outside services', as an MCP server
/// would add them; none reads a file, memory or the web, so no row's right call is one of them.
const EXTRA: [(&str, &str, &[&str]); 44] = [
    (
        "calendar_list_events",
        "Lists the events on the user's calendar between two dates.",
        &["start", "end"],
    ),
    (
        "calendar_create_event",
        "Creates an event on the user's calendar.",
        &["title", "start", "end"],
    ),
    (
        "calendar_update_event",
        "Changes the title or the time of a calendar event.",
        &["event_id", "start", "end"],
    ),
    (
        "calendar_delete_event",
        "Deletes an event from the user's calendar.",
        &["event_id"],
    ),
    (
        "calendar_find_free_time",
        "Finds the times on a day when the user has no event.",
        &["date"],
    ),
    (
        "email_search",
        "Searches the user's mailbox for messages.",
        &["query"],
    ),
    ("email_read", "Reads one email message.", &["message_id"]),
    (
        "email_draft",
        "Saves a draft of an email.",
        &["to", "subject", "body"],
    ),
    ("email_send", "Sends an email.", &["to", "subject", "body"]),
    (
        "email_archive",
        "Moves an email message to the archive.",
        &["message_id"],
    ),
    (
        "contacts_search",
        "Finds a person in the user's contacts.",
        &["name"],
    ),
    (
        "contacts_add",
        "Adds a person to the user's contacts.",
        &["name", "phone"],
    ),
    (
        "contacts_update",
        "Changes a contact's phone number or email address.",
        &["contact_id", "phone"],
    ),
    (
        "tasks_list",
        "Lists the tasks on one of the user's to-do lists.",
        &["list"],
    ),
    (
        "tasks_add",
        "Adds a task to a to-do list.",
        &["list", "title"],
    ),
    ("tasks_complete", "Marks a task as done.", &["task_id"]),
    (
        "tasks_set_due",
        "Sets the day a task is due.",
        &["task_id", "date"],
    ),
    ("weather_now", "Gives the weather in a city now.", &["city"]),
    (
        "weather_forecast",
        "Gives the weather forecast for a city on a day.",
        &["city", "date"],
    ),
    (
        "maps_directions",
        "Gives directions between two places.",
        &["from", "to"],
    ),
    (
        "maps_travel_time",
        "Gives the travel time between two places.",
        &["from", "to"],
    ),
    (
        "maps_find_place",
        "Finds a place such as a shop or a restaurant near an address.",
        &["query", "near"],
    ),
    (
        "music_play",
        "Plays a song, an album or a playlist.",
        &["name"],
    ),
    ("music_pause", "Pauses the music that is playing.", &[]),
    (
        "music_queue",
        "Adds a song after the one that is playing.",
        &["name"],
    ),
    (
        "music_set_volume",
        "Sets the music's volume from 0 to 100.",
        &["level"],
    ),
    (
        "lights_set",
        "Turns a room's lights on or off.",
        &["room", "state"],
    ),
    (
        "thermostat_set",
        "Sets the temperature the heating keeps.",
        &["degrees"],
    ),
    (
        "door_lock",
        "Locks or unlocks a door of the home.",
        &["door", "state"],
    ),
    (
        "vacuum_start",
        "Starts the robot vacuum in a room.",
        &["room"],
    ),
    (
        "message_send",
        "Sends a text message to a person.",
        &["to", "text"],
    ),
    (
        "message_list",
        "Lists the latest text messages from a person.",
        &["from"],
    ),
    ("call_start", "Starts a phone call to a person.", &["to"]),
    (
        "bank_balance",
        "Gives the balance of one of the user's bank accounts.",
        &["account"],
    ),
    (
        "bank_transactions",
        "Lists an account's payments between two dates.",
        &["account", "start", "end"],
    ),
    (
        "bank_transfer",
        "Moves money between two of the user's accounts.",
        &["from", "to", "amount"],
    ),
    (
        "flight_search",
        "Searches for flights between two cities on a day.",
        &["from", "to", "date"],
    ),
    (
        "hotel_search",
        "Searches for hotels in a city for a stay.",
        &["city", "check_in", "check_out"],
    ),
    (
        "train_times",
        "Gives the train departures between two stations on a day.",
        &["from", "to", "date"],
    ),
    (
        "order_status",
        "Gives the delivery status of an online order.",
        &["order_id"],
    ),
    (
        "cart_add",
        "Adds a product to the shopping cart.",
        &["product"],
    ),
    (
        "steps_today",
        "Gives the steps the user's watch has counted today.",
        &[],
    ),
    (
        "sleep_last_night",
        "Gives how long the user's watch says they slept last night.",
        &[],
    ),
    (
        "translate_text",
        "Translates a text into another language.",
        &["text", "language"],
    ),
];

/// The most `--extra-tools` takes.
pub const MAX_EXTRA_TOOLS: usize = EXTRA.len();

/// The name `RENAMED` shows for `real`; any other name stays.
pub fn shown_name(real: &str) -> &str {
    RENAMED
        .iter()
        .find(|(r, _)| *r == real)
        .map_or(real, |&(_, s)| s)
}

/// The real name of `shown`, a `RENAMED` name; any other name stays.
pub fn real_name(shown: &str) -> &str {
    RENAMED
        .iter()
        .find(|(_, s)| *s == shown)
        .map_or(shown, |&(r, _)| r)
}

/// `expect` with its `tool` and `tool_any` shown by their `RENAMED` names.
fn shown_expect(expect: &mut Expect) {
    expect.tool = expect.tool.take().map(|t| shown_name(&t).into());
    for t in expect.tool_any.iter_mut().flatten() {
        *t = shown_name(t).into();
    }
}

/// `text` with each word that is a tool's real name shown by its `RENAMED` name.
fn rename_words(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut word = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            word.push(c);
        } else {
            out.push_str(shown_name(&word));
            word.clear();
            out.push(c);
        }
    }
    out.push_str(shown_name(&word));
    out
}

impl Routing {
    /// The tools the rows run with: the built-in ones, under `RENAMED`'s names with `rename`.
    fn registry(&self) -> Registry {
        let mut registry = match self.rename {
            true => Registry::builtin().renamed(rename_words),
            false => Registry::builtin(),
        };
        for &(name, description, args) in self.extra() {
            registry.replace(extra_tool(name, description, args));
        }
        registry
    }

    /// The `EXTRA` tools as a chat's outside tools, for the turn rows, which open real chats
    /// (`Agent::new_with`): named as an MCP server's would be, `calendar__list_events`.
    fn outside(&self) -> Vec<Box<dyn Tool>> {
        (self.extra().iter())
            .map(|&(name, description, args)| {
                extra_tool(&name.replacen('_', "__", 1), description, args)
            })
            .collect()
    }

    /// The `EXTRA` tools this run adds.
    fn extra(&self) -> &'static [(&'static str, &'static str, &'static [&'static str])] {
        &EXTRA[..self.extra_tools]
    }

    /// `prompt` as the model sees it: with `rename`, its tools and the tool it expects by their
    /// shown names.
    fn shown_prompt(&self, prompt: &BenchPrompt) -> BenchPrompt {
        let mut prompt = prompt.clone();
        if self.rename {
            prompt.tools = prompt.tools.iter().map(|t| shown_name(t).into()).collect();
            shown_expect(&mut prompt.expect);
        }
        prompt.tools.extend(self.extra().iter().map(|e| e.0.into()));
        prompt
    }

    /// `prompt` as `shown_prompt` shows a tool row.
    fn shown_steps(&self, prompt: &StepPrompt) -> StepPrompt {
        let mut prompt = prompt.clone();
        if self.rename {
            prompt.tools = prompt.tools.iter().map(|t| shown_name(t).into()).collect();
            for step in &mut prompt.steps {
                shown_expect(&mut step.expect);
            }
            prompt.r#final.tool = prompt.r#final.tool.take().map(|t| shown_name(&t).into());
        }
        prompt.tools.extend(self.extra().iter().map(|e| e.0.into()));
        prompt
    }
}

/// How a row sends its steps: the loop's parameters for its tools, family and thinking, and the
/// run's router.
struct Ask<'a> {
    registry: &'a Registry,
    allow: Vec<&'a str>,
    family: Family,
    thinking: Thinking,
    namespace: u64,
    routing: &'a Routing,
    /// The tokens a call of `family` starts with.
    starts: &'a [u32],
}

impl Ask<'_> {
    /// A step's parameters, seeded: a step the router decided (`decided`) thinks not, and one the
    /// probe masked shows no tools.
    fn params(&self, kind: Kind, decided: bool, masked: bool) -> GenParams {
        let thinking = match decided || masked {
            true => Thinking {
                budget: 0,
                ..self.thinking
            },
            false => self.thinking,
        };
        let allow: &[&str] = if masked { &[] } else { &self.allow };
        let family = Some(self.family);
        GenParams {
            grammar: self.grammar(allow, Mode::Auto),
            ..step_params(self.registry, allow, family, kind, thinking, self.namespace)
        }
    }

    /// The loop's one retry, held as `mode` (`run::retry_mode`).
    fn retry(&self, mode: Mode, masked: bool) -> GenParams {
        let allow: &[&str] = if masked { &[] } else { &self.allow };
        GenParams {
            grammar: self.grammar(allow, mode),
            ..self.params(Kind::Retry, false, masked)
        }
    }

    /// The run's call grammar of a step with `allow`'s tools, held as `mode` (`run::grammar`).
    fn grammar(&self, allow: &[&str], mode: Mode) -> Option<Grammar> {
        let (on, xml) = (self.routing.tool_grammar, self.routing.xml_body);
        let g = on.then(|| run::grammar(self.registry, allow, Some(self.family), xml, mode));
        // run's tests build one of every tool, in every family.
        g.transpose()
            .expect("the agent's tools fit a grammar")
            .flatten()
    }

    /// The first step as the chat sends it, read to its end, and whether the probe masked the
    /// tools. The probe decides before it; the logprob gate (or the probe router without a probe)
    /// reads its first token on a one-token request of its own (`run::first_token`) and, on a
    /// family that thinks, sends the step with thinking when the margin is under
    /// `router_margin`, else without. `routed` says what happened.
    fn first(
        &self,
        larch: &Larch,
        session: &mut Session,
        user: &str,
        routed: &mut Option<Routed>,
    ) -> anyhow::Result<(Step, bool)> {
        let tools = !self.allow.is_empty();
        let (router, family) = (self.routing.router, tools.then_some(self.family));
        if router == Router::Off || !tools {
            let step = read_step(
                session.send(user, self.params(Kind::Chat, false, false)),
                family,
            )?;
            return Ok((step, false));
        }
        let mut r = Routed::default();
        if router == Router::Probe
            && let Some(probe) = &self.routing.probe
        {
            let x = probe::features(
                larch,
                probe::prompt(&self.routing.system, user),
                &probe.layers,
            )?;
            let p = probe.score(&x).map_err(anyhow::Error::msg)?;
            r.probe = Some(p);
            r.masked = p < probe.threshold;
            *routed = Some(r);
            let fam = family.filter(|_| !r.masked);
            let p = self.params(Kind::Chat, r.masked, r.masked);
            return Ok((read_step(session.send(user, p), fam)?, r.masked));
        }
        // The gate reads one token on a session of its own; the step asks for no logprobs.
        let opts = SessionOptions {
            system: Some(self.routing.system.clone()),
            ..SessionOptions::default()
        };
        let mut gate = larch.session(opts.clone());
        let decided = self.params(Kind::Chat, true, false);
        let read = run::first_token(
            &crate::runtime::inference::SeatAdmission::of(larch),
            &mut gate,
            &opts,
            session.history(),
            user,
            &decided,
        )?;
        (r.gate_prompt_tokens, r.gate_cached_tokens) =
            (Some(read.prompt_tokens), Some(read.cached_tokens));
        r.gate_ms = Some(read.ms);
        let reading = read.first.and_then(|t| router::read(self.starts, &t.top));
        r.margin = reading.map(|g| g.margin);
        r.call = reading.map(|g| g.call);
        let close = reading.is_none_or(|g| g.margin < self.routing.margin);
        r.again = close && self.family == Family::Qwen3 && self.thinking.budget > 0;
        *routed = Some(r);
        let p = match r.again {
            true => self.params(Kind::Chat, false, false),
            false => decided,
        };
        Ok((read_step(session.send(user, p), family)?, false))
    }
}

/// A step's parameters: the loop's own for `family` and the step's kind in the cache `namespace`,
/// seeded.
fn step_params(
    registry: &Registry,
    allow: &[&str],
    family: Option<Family>,
    kind: Kind,
    thinking: Thinking,
    namespace: u64,
) -> GenParams {
    GenParams {
        seed: Some(SEED),
        ..run::params(registry, allow, family, kind, thinking, namespace)
    }
}

/// One model step as the loop reads it.
struct Step {
    parsed: Parsed,
    /// Where the step was cut, when it ended at its token limit.
    cut: Option<Cut>,
    usage: Usage,
    timings: Timings,
    /// When the loop would first have shown visible text while the step streamed.
    shown_at: Option<Instant>,
    /// When the step ended.
    done_at: Instant,
}

impl Step {
    /// Milliseconds from `started` to the first thing of this step the user saw: text as it
    /// streamed, else at the step's end the call line (with its confirm) or an answer the loop
    /// held back to the end. A failed step shows nothing more.
    fn seen(&self, checked: &Checked, started: Instant) -> Option<f64> {
        let at_end = match checked {
            Checked::Call(..) => true,
            Checked::Answer => !self.parsed.text_before.trim().is_empty(),
            Checked::ParseError(_) | Checked::Invalid(..) => false,
        };
        let at = self.shown_at.or(at_end.then_some(self.done_at))?;
        Some(at.duration_since(started).as_secs_f64() * 1e3)
    }
}

/// Reads a step to its end as the loop does: its parser (`None`, plain text, when the model was
/// shown no tools), and the time the loop would first have shown text.
fn read_step(
    stream: Result<TextStream, EngineError>,
    family: Option<Family>,
) -> Result<Step, EngineError> {
    let tools = family.is_some();
    let mut stream = stream?;
    let (mut text, mut shown_at) = (String::new(), None);
    while let Some(event) = stream.next() {
        match event {
            TextEvent::Prefilled { .. } | TextEvent::Thinking { .. } => {}
            TextEvent::Text { delta, .. } => {
                text.push_str(&delta);
                if shown_at.is_none() && !text[..run::shown(&text, tools)].trim().is_empty() {
                    shown_at = Some(Instant::now());
                }
            }
            TextEvent::Done {
                finish,
                usage,
                timings,
                ..
            } => {
                let parsed = run::read(family, &text, stream.in_thinking());
                return Ok(Step {
                    cut: (finish == Finish::Length).then(|| cut(&text, &parsed)),
                    parsed,
                    usage,
                    timings,
                    shown_at,
                    done_at: Instant::now(),
                });
            }
            TextEvent::Error(e) => return Err(e),
        }
    }
    Err(EngineError::Cancelled)
}

/// Nearest-rank p50 and p99; zeros for no values.
fn percentiles(mut values: Vec<f64>) -> Percentiles {
    values.sort_by(f64::total_cmp);
    Percentiles {
        p50: nearest(&values, 50.0),
        p99: nearest(&values, 99.0),
    }
}

/// The nearest-rank `p`th percentile of `sorted`; 0 for no values.
fn nearest(sorted: &[f64], p: f64) -> f64 {
    let at = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted
        .get(at.saturating_sub(1))
        .copied()
        .unwrap_or_default()
}

/// The percentiles of the values there are, `None` when there are none.
fn some_percentiles(values: impl Iterator<Item = Option<f64>>) -> Option<Percentiles> {
    let values: Vec<f64> = values.flatten().collect();
    (!values.is_empty()).then(|| percentiles(values))
}

/// The run's numbers over its rows: the counters over all of them, as a step an engine error
/// followed still stopped where it did, the rest over those that did not error.
pub fn summary(rows: &[Row]) -> Summary {
    let ok: Vec<&Row> = rows.iter().filter(|r| r.error.is_none()).collect();
    let share = |f: &dyn Fn(&Row) -> bool| {
        ok.iter().filter(|r| f(r)).count() as f64 / ok.len().max(1) as f64
    };
    let mut counters = Counters::default();
    for r in rows {
        counters.add(&r.cut, r.wrong_but_valid.into(), r.retried.into());
        counters.offline_claims += u32::from(r.offline_claim);
        counters.unknown_tools += r.unknown_tools;
        counters.schema_fails += r.schema_fails;
        counters.withdrawn += u32::from(r.withdrawn);
    }
    let failing: Vec<String> = (rows.iter())
        .filter(|r| r.error.is_some() || !r.after.right_tool || r.wrong_answer.is_some())
        .map(|r| r.id.clone())
        .collect();
    Summary {
        prompts: rows.len(),
        errors: rows.len() - ok.len(),
        parse_rate_before: share(&|r| r.before.parsed_ok),
        parse_rate_after: share(&|r| r.after.parsed_ok),
        wrong_tool_rate: share(&|r| wrong_tool(r)),
        ttft_ms: percentiles(ok.iter().map(|r| r.ttft_ms).collect()),
        tok_s: ok.iter().map(|r| r.tok_s).sum::<f64>() / ok.len().max(1) as f64,
        cached_tokens: ok.iter().map(|r| u64::from(r.cached_tokens)).sum(),
        wall_s: rows.iter().map(|r| r.wall_ms).sum::<f64>() / 1e3,
        visible_ms: some_percentiles(ok.iter().map(|r| r.visible_ms)),
        counters: Some(counters),
        wrong_tool_by_group: by_group(&ok),
        by_class: by_class(rows),
        right_rate: (rows.len() - failing.len()) as f64 / rows.len().max(1) as f64,
        failing,
    }
}

/// A row's class: the middle of `<set>-<class>-<n>` (`source-files-present-003` is
/// `files-present`), or `other`.
fn class(id: &str) -> &str {
    (id.split_once('-'))
        .and_then(|(_, rest)| rest.rsplit_once('-'))
        .map_or("other", |(class, _)| class)
}

/// `ClassCount` of each class.
fn by_class(rows: &[Row]) -> BTreeMap<String, ClassCount> {
    let mut classes: BTreeMap<String, ClassCount> = BTreeMap::new();
    for r in rows {
        let c = classes.entry(class(&r.id).into()).or_default();
        c.rows += 1;
        c.errors += usize::from(r.error.is_some());
        c.wrong_tool += usize::from(r.error.is_none() && !r.after.right_tool);
        c.wrong_answer += usize::from(r.wrong_answer.is_some());
    }
    classes
}

/// The row's last reply parsed and called a tool other than the expected one.
fn wrong_tool(r: &Row) -> bool {
    r.after.parsed_ok && r.tool.is_some() && !r.after.right_tool
}

/// `wrong_tool_rate` per id group: the id before its last `-`.
fn by_group(rows: &[&Row]) -> BTreeMap<String, f64> {
    let mut groups: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for r in rows {
        let group = r.id.rsplit_once('-').map_or(r.id.as_str(), |(g, _)| g);
        let (n, wrong) = groups.entry(group.into()).or_default();
        *n += 1;
        *wrong += usize::from(wrong_tool(r));
    }
    (groups.into_iter())
        .map(|(g, (n, wrong))| (g, wrong as f64 / n as f64))
        .collect()
}

/// One row of `bench/agent/multistep.jsonl` or `routing/multistep.jsonl`: a turn that takes its
/// steps' calls, then its final step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepPrompt {
    /// `<kind>-<n>`: one, chain (a later call's argument is in an earlier result), parallel
    /// (independent reads) or error (the result is a tool error); `gen_multistep.py` names the
    /// chat rows' kinds.
    pub id: String,
    /// The tools the model is shown.
    pub tools: Vec<String>,
    /// The user's text.
    pub user: String,
    /// The calls the turn needs, and what each gets back.
    pub steps: Vec<ScriptedStep>,
    /// The answer, or the skill call that ends the turn.
    pub r#final: Final,
}

/// A call a multi-step row needs and its scripted result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptedStep {
    /// The call, scored as a tool row's.
    pub expect: Expect,
    /// What the runner sends back as the call's result.
    pub result: String,
    /// A call may skip this step for a later one, which then counts it as made: a not-found
    /// error already lists the folder a listing step would.
    #[serde(default)]
    pub optional: bool,
}

/// The step that ends a multi-step row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Final {
    /// The tool it calls: `None`, an answer. A skill call ends the app's turn, so it comes here.
    pub tool: Option<String>,
    /// Arguments the call must have, as a step's `args_required`.
    #[serde(default)]
    pub args_required: Map<String, Value>,
}

/// The rows whose calls may all come in one reply.
const PARALLEL: &str = "parallel-";

/// One multi-step row's result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepRow {
    /// The row's ID.
    pub id: String,
    /// Per scripted step: a call with its tool, every required argument and its path came, or for
    /// an `optional` step a later call skipped it.
    pub steps: Vec<bool>,
    /// The turn ended with the final step: an answer with text when it names no tool.
    pub answered: bool,
    /// Model steps the turn took, retries included.
    pub steps_used: u32,
    /// Retries after a parse or validate error.
    pub retries: u32,
    /// A valid call fit no open step, which ended the row.
    pub wrong_but_valid: bool,
    /// Where each step that ended at the token limit was cut.
    pub cut: Vec<Cut>,
    /// On a parallel row, whether the first reply held more than one call (`false` for an answer,
    /// a broken call or an engine error first); `None` elsewhere.
    pub calls_in_one_step: Option<bool>,
    /// Prompt tokens over all steps.
    pub prompt_tokens: u32,
    /// Generated tokens over all steps.
    pub generated_tokens: u32,
    /// The first step's time to first token.
    pub ttft_ms: f64,
    /// From the start of the turn to the first thing the user sees.
    pub visible_ms: Option<f64>,
    /// The turn, end to end.
    pub wall_ms: f64,
    /// The engine error that ended the row.
    pub error: Option<String>,
    /// What the router did before and at the first step; `None` with the router off.
    #[serde(default)]
    pub routed: Option<Routed>,
    /// Calls to a tool not in the list, over all steps.
    #[serde(default)]
    pub unknown_tools: u32,
    /// Calls that fail their tool's schema, over all steps.
    #[serde(default)]
    pub schema_fails: u32,
    /// Each model step's reply as the loop read it (`reply_of`).
    #[serde(default)]
    pub replies: Vec<String>,
}

impl StepRow {
    /// Every step right and the turn answered.
    pub fn right(&self) -> bool {
        self.answered && self.steps.iter().all(|&s| s)
    }
}

/// The multi-step numbers, over the rows that did not error; the counters over all of them.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepSummary {
    /// Rows run.
    pub rows: usize,
    /// Rows an engine error ended.
    pub errors: usize,
    /// Share of rows with every step right and the answer.
    pub right_rate: f64,
    /// Mean model steps a row took.
    pub steps: f64,
    /// Mean generated tokens a row took.
    pub generated_tokens: f64,
    /// Share of parallel rows whose first reply held more than one call; `None` without any.
    pub calls_in_one_step: Option<f64>,
    /// Time to first token of the first step.
    pub ttft_ms: Percentiles,
    /// From the start of each turn to the first thing the user sees.
    pub visible_ms: Option<Percentiles>,
    /// Limit stops, wrong-but-valid calls and retries, over every row.
    pub counters: Counters,
    /// All rows, end to end.
    pub wall_s: f64,
}

/// The multi-step section of a report.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiStep {
    /// The rows file.
    pub prompts: PromptsFile,
    /// The numbers.
    pub summary: StepSummary,
    /// One per row, in file order.
    pub rows: Vec<StepRow>,
}

/// Runs each multi-step row on a fresh session with its tools, as the loop runs a turn, sending
/// each call the result of the step it fits, with the router off; `progress` sees each row as it
/// is done.
pub fn run_steps(
    larch: &Larch,
    family: Family,
    thinking: Thinking,
    namespace: u64,
    prompts: &[StepPrompt],
    progress: impl FnMut(&StepRow),
) -> Vec<StepRow> {
    let routing = Routing::default();
    run_steps_routed(
        larch, family, thinking, namespace, &routing, prompts, progress,
    )
}

/// `run_steps` with `routing`'s router in front of each row's first step and its tool names.
pub fn run_steps_routed(
    larch: &Larch,
    family: Family,
    thinking: Thinking,
    namespace: u64,
    routing: &Routing,
    prompts: &[StepPrompt],
    mut progress: impl FnMut(&StepRow),
) -> Vec<StepRow> {
    let registry = routing.registry();
    let starts = router::starts(family, |text| larch.token_ids(text));
    let mut rows = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let started = Instant::now();
        let mut row = StepRow {
            id: prompt.id.clone(),
            steps: vec![false; prompt.steps.len()],
            calls_in_one_step: prompt.id.starts_with(PARALLEL).then_some(false),
            ..StepRow::default()
        };
        let prompt = routing.shown_steps(prompt);
        let ask = Ask {
            registry: &registry,
            allow: prompt.tools.iter().map(String::as_str).collect(),
            family,
            thinking,
            namespace,
            routing,
            starts: &starts,
        };
        if let Err(e) = steps(larch, &ask, &prompt, &mut row) {
            row.error = Some(e.to_string());
        }
        row.wall_ms = ms(started);
        progress(&row);
        rows.push(row);
    }
    rows
}

/// One multi-step row. A valid call is scored against the next step, or the first after the
/// `optional` steps it skips, or on a parallel row against the first step not yet made whose tool
/// and required arguments it has, so independent reads may come in any order; one that fits none
/// ends the row, as there is no result to send it. A parse or validate error gets the loop's one
/// retry, which a valid call renews.
fn steps(larch: &Larch, ask: &Ask, prompt: &StepPrompt, row: &mut StepRow) -> anyhow::Result<()> {
    let mut session = larch.session(SessionOptions {
        system: Some(ask.routing.system.clone()),
        ..SessionOptions::default()
    });
    let started = Instant::now();
    let (mut step, masked) = ask.first(larch, &mut session, &prompt.user, &mut row.routed)?;
    let allow: &[&str] = if masked { &[] } else { &ask.allow };
    let tools = !allow.is_empty();
    let params = |kind| ask.params(kind, false, masked);
    let (mut retried, mut calls) = (false, 0);
    let parallel = prompt.id.starts_with(PARALLEL);
    loop {
        if row.steps_used == 0 {
            row.ttft_ms = step.timings.ttft_ms;
        }
        row.steps_used += 1;
        row.prompt_tokens += step.usage.prompt_tokens;
        row.generated_tokens += step.usage.completion_tokens;
        row.cut.extend(step.cut);
        if row.steps_used == 1 && parallel {
            let several = !step.parsed.later.is_empty();
            row.calls_in_one_step =
                Some(several || step.parsed.error == Some(ParseError::MultipleCalls));
        }
        let checked = run::check(&step.parsed, ask.registry, allow);
        tally(&checked, &mut row.unknown_tools, &mut row.schema_fails);
        row.visible_ms = row.visible_ms.or_else(|| step.seen(&checked, started));
        let ran = as_run(&step.parsed, &checked);
        row.replies.push(reply_of(&ran));
        let mode = run::retry_mode(&checked);
        let done = row.steps.iter().all(|&s| s);
        let stream = match checked {
            Checked::Answer => {
                row.answered =
                    prompt.r#final.tool.is_none() && !step.parsed.text_before.trim().is_empty();
                return Ok(());
            }
            Checked::Call(..) if done => {
                let last = Expect {
                    tool: prompt.r#final.tool.clone(),
                    args_required: prompt.r#final.args_required.clone(),
                    ..Expect::default()
                };
                row.answered = fits(&last, &ran);
                row.wrong_but_valid = !row.answered;
                return Ok(());
            }
            Checked::Call(call, permission, more) => {
                // The next step, or past the optional ones in front of it; on a parallel row any
                // step not yet made.
                let open = |steps: &[bool], ran: &Parsed| {
                    for i in (0..steps.len()).filter(|&i| !steps[i]) {
                        if fits(&prompt.steps[i].expect, ran) {
                            return Some(i);
                        }
                        if !parallel && !prompt.steps[i].optional {
                            break;
                        }
                    }
                    None
                };
                let Some(i) = open(&row.steps, &ran) else {
                    row.wrong_but_valid = true;
                    return Ok(());
                };
                // A call past optional steps counts them as made.
                row.steps[if parallel { i } else { 0 }..=i].fill(true);
                // The reply's later reads that fit open steps run with it, as the chat runs a
                // reply's reads (`Agent::reads`).
                let plain = |p, c: &ToolCall| {
                    p == crate::tool::Permission::Read && !run::SKILLS.contains(&c.name.as_str())
                };
                let several = plain(permission, &call) && session.takes_several_calls();
                let mut batch = vec![(call, i)];
                if several {
                    for later in &step.parsed.later {
                        let one = Parsed {
                            call: Some(later.clone()),
                            ..Parsed::default()
                        };
                        let Checked::Call(c, p, _) = run::check(&one, ask.registry, allow) else {
                            break;
                        };
                        let ran = Parsed {
                            call: Some(c.clone()),
                            ..Parsed::default()
                        };
                        let Some(i) = open(&row.steps, &ran).filter(|_| plain(p, &c)) else {
                            break;
                        };
                        row.steps[if parallel { i } else { 0 }..=i].fill(true);
                        batch.push((c, i));
                    }
                }
                // Calls the reply held past the batch: the model hears they did not run.
                let more =
                    more && (step.parsed.error.is_some() || batch.len() <= step.parsed.later.len());
                retried = false;
                let mut model = Vec::new();
                let mut results = Vec::new();
                for (call, i) in batch {
                    calls += 1;
                    let id = format!("{calls:09}");
                    model.push(run::model_call(&call, &id));
                    results.push((id, prompt.steps[i].result.clone()));
                }
                if more && let Some((_, last)) = results.last_mut() {
                    *last = format!("{last}\n{}", ParseError::MultipleCalls);
                }
                session.set_several_tool_calls(&step.parsed.text_before, model)?;
                session.send_tool_results(results, params(Kind::Answer))
            }
            // The loop's one retry is spent.
            _ if retried => return Ok(()),
            Checked::ParseError(e) => {
                (retried, row.retries) = (true, row.retries + 1);
                session.send(&run::retry_text(&e, tools), ask.retry(mode, masked))
            }
            Checked::Invalid(call, e) => {
                (retried, row.retries, calls) = (true, row.retries + 1, calls + 1);
                let id = format!("{calls:09}");
                session.set_tool_calls(&step.parsed.text_before, run::model_call(&call, &id))?;
                session.send_tool_result(&id, &e.to_string(), ask.retry(mode, masked))
            }
        };
        step = read_step(stream, tools.then_some(ask.family))?;
    }
}

/// `parsed` with the call the loop checked and runs (`run::check`): its numeric strings coerced
/// where the schema takes an integer. The raw reply stays in `parsed`.
fn as_run(parsed: &Parsed, checked: &Checked) -> Parsed {
    match checked {
        Checked::Call(call, ..) | Checked::Invalid(call, _) => Parsed {
            call: Some(call.clone()),
            ..parsed.clone()
        },
        Checked::Answer | Checked::ParseError(_) => parsed.clone(),
    }
}

/// A reply for the report: its call as `name {args}`, else its parse error, else its text, cut
/// at `SAID_BYTES`.
fn reply_of(parsed: &Parsed) -> String {
    let text = match (&parsed.call, &parsed.error) {
        (Some(call), _) => format!("{} {}", call.name, call.args),
        (None, Some(e)) => format!("parse error: {e}"),
        (None, None) => parsed.text_before.trim().to_owned(),
    };
    text[..text.floor_char_boundary(SAID_BYTES)].to_owned()
}

/// The most of a reply a report keeps.
const SAID_BYTES: usize = 400;

/// Whether a multi-step call fits a step: it scores right as a tool row's call does, and names
/// the step's path itself, as the file tools take it, not a path that holds it (`a/x.txt` for
/// `x.txt`); case and a leading `./` aside, and a trailing `/` only on a folder (`list_dir`): a
/// file's path with one fails in the tool (`notes/todo.txt/` is not a folder).
fn fits(expect: &Expect, parsed: &Parsed) -> bool {
    // The step names the stand-in under `--rename-tools`; the rule is the real tool's.
    let folder = expect.tool.as_deref().map(real_name) == Some("list_dir");
    let path = |v: Option<&Value>| {
        v.and_then(Value::as_str).map(|p| {
            let p = p.trim_start_matches("./");
            let p = if folder { p.trim_end_matches('/') } else { p };
            p.to_lowercase()
        })
    };
    let want = path(expect.args_required.get("path"));
    score_toolcall(expect, parsed).args_ok
        && (want.is_none() || path(parsed.call.as_ref().and_then(|c| c.args.get("path"))) == want)
}

/// The multi-step numbers over its rows: the counters over all of them, the rest over those that
/// did not error.
pub fn step_summary(rows: &[StepRow]) -> StepSummary {
    let ok: Vec<&StepRow> = rows.iter().filter(|r| r.error.is_none()).collect();
    let mean =
        |f: &dyn Fn(&StepRow) -> f64| ok.iter().map(|r| f(r)).sum::<f64>() / ok.len().max(1) as f64;
    let mut counters = Counters::default();
    for r in rows {
        counters.add(&r.cut, r.wrong_but_valid.into(), r.retries);
        counters.unknown_tools += r.unknown_tools;
        counters.schema_fails += r.schema_fails;
    }
    let parallel: Vec<bool> = ok.iter().filter_map(|r| r.calls_in_one_step).collect();
    StepSummary {
        rows: rows.len(),
        errors: rows.len() - ok.len(),
        right_rate: mean(&|r| f64::from(u8::from(r.right()))),
        steps: mean(&|r| r.steps_used.into()),
        generated_tokens: mean(&|r| r.generated_tokens.into()),
        calls_in_one_step: (!parallel.is_empty())
            .then(|| parallel.iter().filter(|&&p| p).count() as f64 / parallel.len() as f64),
        ttft_ms: percentiles(ok.iter().map(|r| r.ttft_ms).collect()),
        visible_ms: some_percentiles(ok.iter().map(|r| r.visible_ms)),
        counters,
        wall_s: rows.iter().map(|r| r.wall_ms).sum::<f64>() / 1e3,
    }
}

/// One question of the document set's `questions.json`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Question {
    /// `q01` to `q24`.
    pub id: String,
    /// The question, as a user would ask it.
    pub question: String,
    /// The facts that answer it; empty when no document does.
    pub facts: Vec<Planted>,
    /// Calls that run first, in the question's chat folder, in order: "read_papers the folder,
    /// then ask about ResNet". One that fails ends the question with its error.
    #[serde(default)]
    pub before: Vec<Before>,
    /// The outcome a right run comes to (`outcome` of ask_docs's first line, then its answer):
    /// `unreadable` for a scanned PDF, `no_files` for a folder of other types; when set, it alone
    /// says whether the run is right.
    #[serde(default)]
    pub outcome: Option<Outcome>,
}

/// A call a document question runs before it is asked.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Before {
    /// The tool: a built-in one, or ask_docs or read_papers on the bench's engine.
    pub tool: String,
    /// Its arguments.
    pub args: Value,
}

/// A fact planted in the document set.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Planted {
    /// The file, relative to the set's folder.
    pub file: PathBuf,
    /// Text that occurs exactly once in that file.
    pub quote: String,
    /// The line the quote starts on, counted from 1.
    pub line: u32,
}

/// A set's `questions.json`; an ID twice is an error, as the report keeps each question's error
/// under its ID.
fn questions<T: DeserializeOwned>(
    dir: &Path,
    id: impl Fn(&T) -> &String,
) -> anyhow::Result<Vec<T>> {
    let path = dir.join("questions.json");
    let text = fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let questions: Vec<T> =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let mut ids = HashSet::new();
    if let Some(q) = questions.iter().find(|&q| !ids.insert(id(q))) {
        anyhow::bail!("{} has question {} twice", path.display(), id(q));
    }
    Ok(questions)
}

/// A document set, ready for `run_docs`.
#[derive(Debug)]
pub struct DocSet {
    /// Its folder, absolute, as the skill takes it.
    pub folder: String,
    /// Its `questions.json`.
    pub questions: Vec<Question>,
    /// Chunks each question gives its workers (`--docs-workers`).
    pub workers: usize,
    /// Tokens of a chunk's file a worker reads on each side of it (`--docs-window`).
    pub window: u32,
    /// A throwaway data folder for the runs' audit lines and notes, so the bench adds nothing to
    /// the user's memory.
    data: tempfile::TempDir,
}

/// The document set in `dir`: its folder and its questions. All that can be wrong with a set
/// shows here, so a bench fails on it before its long run.
pub fn doc_set(dir: &Path) -> anyhow::Result<DocSet> {
    let questions = questions(dir, |q: &Question| &q.id)?;
    let folder = std::path::absolute(dir)?
        .to_str()
        .context("the docs folder's path is not UTF-8")?
        .to_owned();
    let data = tempfile::tempdir().context("making a data folder for the docs runs")?;
    let skill = AskDocs::default();
    Ok(DocSet {
        folder,
        questions,
        workers: skill.docs_workers,
        window: skill.docs_window,
        data,
    })
}

/// Asks ask-docs each question of `set` about its folder, the only folder it may read, with the
/// set's workers and window, and scores the answers; `progress` sees each answer, or the error
/// that ended its run, as it comes. The model's `family` says whether its requests get a thinking
/// field, and the answers think on `thinking`; the workers sample under the facts grammar when
/// `held` (the run's `tool_grammar`).
#[allow(clippy::too_many_arguments)]
pub fn run_docs(
    larch: &Larch,
    family: Option<Family>,
    thinking: Thinking,
    held: bool,
    set: &DocSet,
    workers: Workers,
    mut progress: impl FnMut(&Question, &Result<SkillRun, SkillError>),
) -> anyhow::Result<Docs> {
    let ctx = ToolCtx {
        allowed: vec![PathBuf::from(&set.folder)],
        data_dir: set.data.path().into(),
        source: String::new(),
    };
    let (mut wall, mut said) = (BTreeMap::new(), BTreeMap::new());
    let skill = AskDocs {
        family,
        thinking,
        docs_workers: set.workers,
        docs_window: set.window,
        facts_grammar: held,
    };
    let runs: Vec<_> = set
        .questions
        .iter()
        .map(|q| {
            let input = SkillInput {
                folder: set.folder.clone(),
                question: q.question.clone(),
                search: String::new(),
            };
            let started = Instant::now();
            let before: Result<Vec<String>, SkillError> = (q.before.iter())
                .map(|b| before(larch, skill, workers, &ctx, b))
                .collect();
            let run = before.and_then(|lines| {
                said.insert(&q.id, lines);
                skill.run(larch, &ctx, workers, &input)
            });
            wall.insert(&q.id, ms(started));
            progress(q, &run);
            run
        })
        .collect();
    let mut docs = docs_metrics(&runs, &set.questions)?;
    for q in &mut docs.questions {
        q.wall_ms = wall.get(&q.id).copied().unwrap_or_default();
        q.before = said.remove(&q.id).unwrap_or_default();
    }
    Ok(docs)
}

/// Runs a question's `before` call in `ctx`, as a chat runs it and logs it: read_papers and
/// ask_docs on `larch` with `skill`'s settings and `workers`, any other a built-in read; its
/// result's first line, or its error.
fn before(
    larch: &Larch,
    skill: AskDocs,
    workers: Workers,
    ctx: &ToolCtx,
    call: &Before,
) -> Result<String, SkillError> {
    let stop = std::sync::Arc::new(AtomicBool::default());
    let builtin = Registry::builtin();
    let tool: Box<dyn crate::tool::Tool> = match call.tool.as_str() {
        crate::skills::read_papers::TOOL => Box::new(crate::skills::read_papers::PapersTool {
            engine: Some((larch.clone(), Default::default(), workers)),
            seen: None,
            stop,
            show: None,
            followup: None,
        }),
        ask_docs::TOOL => Box::new(ask_docs::DocsTool {
            engine: Some((larch.clone(), skill, workers)),
            stop,
        }),
        name => {
            let found = (builtin.get(name)).filter(|t| t.spec().permission == Permission::Read);
            let tool = found.ok_or_else(|| {
                SkillError::Tool(crate::tool::ToolError::Invalid(format!(
                    "before: {name} is no built-in read, ask_docs or read_papers"
                )))
            })?;
            return before_run(tool, call, ctx);
        }
    };
    before_run(&*tool, call, ctx)
}

/// `tool` on `call`'s arguments in `ctx`, logged as a chat logs a call; its first line.
fn before_run(
    tool: &dyn crate::tool::Tool,
    call: &Before,
    ctx: &ToolCtx,
) -> Result<String, SkillError> {
    let audit = Audit::new(&ctx.data_dir);
    let result = crate::tool::run(tool, &call.args, ctx, &audit, AuditEvent::default());
    let text = result.map_err(SkillError::Tool)?;
    Ok(text.lines().next().unwrap_or_default().to_owned())
}

/// Scores the runs against the questions they answered, `runs[i]` answering `questions[i]`. A
/// planted fact is recalled when the answer cites a fact that `recalls` it. A question no
/// document answers is abstained when the answer is `NOT_FOUND` alone, so a near miss told after
/// it is a leak; the notes under it are Rust's, not the model's. A run that failed recalls nothing
/// and does not abstain. Fails when the counts differ or a run's facts are not ask-docs facts.
pub fn docs_metrics(
    runs: &[Result<SkillRun, SkillError>],
    questions: &[Question],
) -> anyhow::Result<Docs> {
    anyhow::ensure!(
        runs.len() == questions.len(),
        "{} runs for {} questions",
        runs.len(),
        questions.len()
    );
    let (mut cited, mut emitted, mut recalled, mut abstained) = (0, 0, 0, 0);
    let (mut errors, mut each) = (BTreeMap::new(), Vec::new());
    let planted = questions.iter().map(|q| q.facts.len()).sum();
    let empty = questions.iter().filter(|q| q.facts.is_empty()).count();
    // A model may leave off the full stop.
    let abstains = NOT_FOUND.trim_end_matches('.');
    for (run, q) in runs.iter().zip(questions) {
        let run = match run {
            Ok(run) => run,
            Err(e) => {
                errors.insert(q.id.clone(), e.to_string());
                continue;
            }
        };
        let facts: Facts = serde_json::from_value(run.facts.clone())?;
        let verified = facts.cited.len();
        cited += verified;
        emitted += verified + facts.dropped;
        let shown: Vec<&Cited> = facts
            .cited
            .iter()
            .filter(|c| cites(&run.answer, &c.source()))
            .collect();
        let found: Vec<usize> = (q.facts.iter().enumerate())
            .filter(|(_, p)| shown.iter().any(|c| recalls(c, p)))
            .map(|(i, _)| i)
            .collect();
        recalled += found.len();
        let abstains = run.answer.trim_end_matches('.') == abstains;
        let right = match q.facts.is_empty() {
            true => abstains,
            false => found.len() == q.facts.len(),
        };
        // ask_docs's first line in a chat (#400), from its gather step, above the answer.
        let head = (run.steps.iter())
            .find(|e| e.step.as_deref() == Some("gather"))
            .and_then(|e| e.detail.as_ref()?["head"].as_str());
        let said = match head {
            Some(head) => format!("{head}\n{}", run.answer),
            None => run.answer.clone(),
        };
        let got = outcome(Some(&said), false, right, q.facts.is_empty());
        let right = q.outcome.map_or(right, |want| got == want);
        if q.facts.is_empty() {
            abstained += usize::from(abstains);
        }
        each.push(DocsQuestion {
            id: q.id.clone(),
            answer: run.answer.clone(),
            cited: (shown.iter())
                .map(|c| CitedQuote {
                    file: c.file.clone(),
                    line: c.line,
                    quote: c.quote.clone(),
                })
                .collect(),
            facts_emitted: verified + facts.dropped,
            facts_verified: verified,
            outcome: Some(got),
            right,
            stages: Some(stages(&run.steps, None)),
            before: Vec::new(),
            recalled: found,
            abstained: abstains,
            wall_ms: 0.0,
        });
    }
    let share = |n: usize, of: usize| (of > 0).then(|| n as f64 / of as f64);
    Ok(Docs {
        precision: share(cited, emitted),
        recall: share(recalled, planted),
        abstain_rate: share(abstained, empty),
        errors,
        questions: each,
    })
}

/// Whether the cited fact is the planted one: of its file, with quotes that overlap, at its place.
/// A cited quote that holds the whole planted quote is at its place, which occurs once in the
/// file; any other must start on one of the planted quote's lines, so a short quote ("month")
/// found earlier in the file does not count.
fn recalls(c: &Cited, p: &Planted) -> bool {
    let lines = p.line..=p.line + p.quote.matches('\n').count() as u32;
    c.file == p.file
        && overlaps(&c.quote, &p.quote)
        && (lines.contains(&c.line) || squash(&c.quote).0.contains(&squash(&p.quote).0))
}

/// Whether two quotes overlap, with every run of whitespace one space, as the skill copies a line: one holds the
/// other, or they share a run of 20 characters.
fn overlaps(a: &str, b: &str) -> bool {
    let (a, b) = (squash(a).0, squash(b).0);
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if a.contains(&b) || b.contains(&a) {
        return true;
    }
    let bounds: Vec<_> = a.char_indices().map(|(i, _)| i).chain([a.len()]).collect();
    bounds.windows(21).any(|w| b.contains(&a[w[0]..w[20]]))
}

/// One question of the memory set's `questions.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryQuestion {
    /// `m01` to `m20`.
    pub id: String,
    /// The question, as a user would ask it.
    pub question: String,
    /// The file that answers it, relative to the set's folder: `daily/<date>.md` or
    /// `facts/<slug>.md`.
    pub file: String,
    /// The line of that file, counted from 1.
    pub line: u32,
    /// Text the answer must contain, case aside.
    pub value: String,
    /// The days a date-range question asks about.
    pub range: Option<DateRange>,
}

/// Days from `from` to `to`, both included, as `YYYY-MM-DD`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DateRange {
    /// The first day.
    pub from: String,
    /// The last day.
    pub to: String,
}

/// A memory set, ready for `run_memory`.
#[derive(Debug)]
pub struct MemorySet {
    /// Its `questions.json`.
    pub questions: Vec<MemoryQuestion>,
    /// Whether `run_all` gives each note line its key phrases (`memory_keys`) before the
    /// questions.
    pub keys: bool,
    /// A throwaway data folder holding the set's notes and facts as `agent/memory/`.
    data: tempfile::TempDir,
}

impl MemorySet {
    /// The data folder `memory::recall` reads.
    pub fn data_dir(&self) -> &Path {
        self.data.path()
    }
}

/// The memory set in `dir`: its questions, and its `daily/` and `facts/` copied into a throwaway
/// data folder, so the bench reads none of the user's memory and adds nothing to it.
pub fn memory_set(dir: &Path) -> anyhow::Result<MemorySet> {
    let questions = questions(dir, |q: &MemoryQuestion| &q.id)?;
    let data = tempfile::tempdir().context("making a data folder for the memory set")?;
    let memory = data.path().join("agent").join("memory");
    for folder in ["daily", "facts"] {
        let (from, to) = (dir.join(folder), memory.join(folder));
        fs::create_dir_all(&to)?;
        let entries = fs::read_dir(&from).with_context(|| format!("reading {}", from.display()))?;
        for entry in entries {
            let entry = entry?;
            if entry.path().extension().is_some_and(|e| e == "md") {
                fs::copy(entry.path(), to.join(entry.file_name()))?;
            }
        }
    }
    Ok(MemorySet {
        questions,
        keys: false,
        data,
    })
}

/// The keys pass of a memory run: note lines asked about, keys saved, and each line's failure.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyPass {
    /// Daily note lines the model was asked for keys.
    pub lines: usize,
    /// Key phrases saved, over all lines.
    pub keys: usize,
    /// `daily/<day>.md:<line>: <error>` for each line that got none.
    pub errors: Vec<String>,
}

/// Gives every daily note line of `set` its key phrases, a model turn per line as a chat's note
/// gets, `workers` at a time.
pub fn memory_keys(
    larch: &Larch,
    family: Family,
    namespace: u64,
    workers: &Workers,
    set: &MemorySet,
) -> KeyPass {
    let mut pass = KeyPass::default();
    let notes = match memory::notes(set.data_dir()) {
        Ok(notes) => notes,
        Err(e) => {
            pass.errors.push(format!("daily notes: {e}"));
            return pass;
        }
    };
    let prompts: Vec<String> = notes.iter().map(|n| n.text.clone()).collect();
    let answers = workers.run(
        larch,
        memory::KEY_SYSTEM,
        &prompts,
        &memory::key_params(Some(family), namespace),
        |_| None,
    );
    for (noted, (answer, error)) in notes.iter().zip(answers) {
        pass.lines += 1;
        let saved = match (answer.first(), error) {
            (Some(a), _) => memory::parse_keys(&a.text, a.finish == Finish::Length)
                .and_then(|keys| memory::save_keys(set.data_dir(), noted, &keys).map(|()| keys))
                .map_err(|e| e.to_string()),
            (None, e) => Err(e.map_or("no answer".into(), |e| e.to_string())),
        };
        match saved {
            Ok(keys) => pass.keys += keys.len(),
            Err(e) => pass
                .errors
                .push(format!("daily/{}.md:{}: {e}", noted.day, noted.line)),
        }
    }
    pass
}

/// One memory question's result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryRow {
    /// The question's ID.
    pub id: String,
    /// Recall's hits, best first, as `file:line` relative to the set's folder.
    pub hits: Vec<String>,
    /// The question's line is one of them.
    pub found: bool,
    /// For a date-range question, whether the best hit is dated inside the range (a daily note
    /// by its name, a fact by its `created`).
    pub in_range: Option<bool>,
    /// The model's answer given the question and the hits; `None` without a model.
    pub answer: Option<String>,
    /// The answer holds the question's value, case aside.
    pub holds_value: Option<bool>,
    /// Where the answer was cut, when it ended at the token limit.
    pub cut: Option<Cut>,
    /// Recall and the answer, end to end.
    pub wall_ms: f64,
    /// The recall or engine error that ended the question.
    pub error: Option<String>,
}

/// The memory section of a report; each share is `None` when there was nothing to count. A
/// question whose recall failed is not found and not in range; one whose model step failed has
/// no tier 2 score.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Memory {
    /// Tier 1: questions whose line is in recall's 5 hits.
    pub recall_at_5: Option<f64>,
    /// Tier 1: date-range questions whose best hit is inside the range.
    pub in_range: Option<f64>,
    /// Tier 2: answers that hold the value.
    pub holds_value: Option<f64>,
    /// The keys pass, run before the questions; `None` with keys off.
    #[serde(default)]
    pub keys: Option<KeyPass>,
    /// One per question, in the set's order.
    pub questions: Vec<MemoryRow>,
}

/// Asks `memory::recall` each question of `set` (tier 1) and, with a model and its tool-call
/// family, asks the model the question with the hits, no tools shown (tier 2), as a chat step
/// thinks on `thinking`; `progress` sees each row as it is done.
pub fn run_memory(
    model: Option<(&Larch, Family, u64)>,
    thinking: Thinking,
    set: &MemorySet,
    mut progress: impl FnMut(&MemoryRow),
) -> Memory {
    let (family, namespace) = model.map_or((None, 0), |(_, f, n)| (Some(f), n));
    let params = step_params(
        &Registry::builtin(),
        &[],
        family,
        Kind::Chat,
        thinking,
        namespace,
    );
    let larch = model.map(|(larch, ..)| larch);
    let mut rows = Vec::with_capacity(set.questions.len());
    for q in &set.questions {
        let started = Instant::now();
        let mut row = MemoryRow {
            id: q.id.clone(),
            ..MemoryRow::default()
        };
        if let Err(e) = remembered(larch, set, q, &params, &mut row) {
            row.error = Some(e);
        }
        row.wall_ms = ms(started);
        progress(&row);
        rows.push(row);
    }
    let share = |f: &dyn Fn(&MemoryRow) -> Option<bool>| {
        let values: Vec<bool> = rows.iter().filter_map(f).collect();
        let yes = values.iter().filter(|&&v| v).count();
        (!values.is_empty()).then(|| yes as f64 / values.len() as f64)
    };
    Memory {
        recall_at_5: share(&|r| Some(r.found)),
        in_range: share(&|r| r.in_range),
        holds_value: share(&|r| r.holds_value),
        keys: None,
        questions: rows,
    }
}

/// One memory question, both tiers.
fn remembered(
    larch: Option<&Larch>,
    set: &MemorySet,
    q: &MemoryQuestion,
    params: &GenParams,
    row: &mut MemoryRow,
) -> Result<(), String> {
    row.in_range = q.range.as_ref().map(|_| false);
    // The hits go to the model unasked, as automatic recall's do, so they come from the same
    // sources, dated from the bench's day.
    let how = memory::Recall {
        sources: &memory::AUTO,
        hits: memory::RECALL_HITS,
        today: memory::day(DATE).unwrap_or_else(memory::today),
    };
    let hits = memory::recall_with(set.data_dir(), &q.question, how).map_err(|e| e.to_string())?;
    let place = |h: &Hit| {
        let file = h.path.strip_prefix("agent/memory/").unwrap_or(&h.path);
        format!("{file}:{}", h.line)
    };
    row.hits = hits.iter().map(place).collect();
    row.found = row.hits.contains(&format!("{}:{}", q.file, q.line));
    row.in_range = q.range.as_ref().map(|r| {
        let best = hits.first().and_then(|h| dated(set.data_dir(), &h.path));
        best.is_some_and(|day| r.from <= day && day <= r.to)
    });
    let Some(larch) = larch else {
        return Ok(());
    };
    let mut session = larch.session(SessionOptions {
        system: Some(run::system_prompt("", Some(""), DATE, None, "")),
        ..SessionOptions::default()
    });
    let user = format!(
        "{}\n\n[agent] Your memory search for this found, best first:\n{}",
        q.question,
        serde_json::json!(hits)
    );
    let step = read_step(session.send(&user, params.clone()), None).map_err(|e| e.to_string())?;
    row.cut = step.cut;
    let answer = step.parsed.text_before;
    row.holds_value = Some(answer.to_lowercase().contains(&q.value.to_lowercase()));
    row.answer = Some(answer);
    Ok(())
}

/// The day a memory file is from, `YYYY-MM-DD`: a daily note's name, a fact's `created`; `None`
/// for a transcript or a fact without one.
fn dated(data_dir: &Path, path: &str) -> Option<String> {
    if path.starts_with("agent/memory/daily/") {
        return Path::new(path).file_stem()?.to_str().map(String::from);
    }
    let text = fs::read_to_string(data_dir.join(path)).ok()?;
    let created = text.lines().find_map(|l| l.strip_prefix("created:"))?;
    created.trim().get(..10).map(String::from)
}

/// One question of the replay set's questions.jsonl (bench/agent/replay); an unknown key, such as
/// a misspelt `not`, fails the set rather than drop its check.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayQuestion {
    /// The question's ID.
    pub id: String,
    /// What it asks about: sports, prices, news, specs, recipes, summaries, abstain, files or
    /// reports.
    pub kind: String,
    /// The question, as a user would ask it.
    pub question: String,
    /// The day it was asked, `YYYY-MM-DD`, which the readers are told.
    pub today: String,
    /// Regexes a right answer matches, case aside: any one, or each one with `all`.
    pub answer: Vec<String>,
    /// Each `answer` regex must match.
    #[serde(default)]
    pub all: bool,
    /// Regexes a right answer does not match, case aside.
    #[serde(default)]
    pub not: Vec<String>,
    /// Its pages do not answer it, so a right answer says so.
    #[serde(default)]
    pub abstain: bool,
    /// Its saved search and pages, `web/<id>`, relative to the set.
    pub pages: Option<String>,
    /// The folder a question about files or saved reports asks about, relative to the set.
    pub folder: Option<String>,
    /// Why the answer is right, or why the pages do not answer, for a reader of the set.
    #[serde(default)]
    pub why: String,
    /// Pages the agent fetched when the set was saved.
    pub fetched: Option<u32>,
    /// The addresses whose fetch failed when the set was saved.
    #[serde(default)]
    pub failed: Vec<String>,
    /// Its page was written for the set, not saved from the web.
    #[serde(default)]
    pub synthetic: bool,
    /// For a row whose answer changes (the latest game, a price), the day `answer` was last
    /// checked, `YYYY-MM-DD`: a `--live` run more than `AS_OF_DAYS` later shows its answer for a
    /// human check instead of scoring it.
    #[serde(default)]
    pub answer_as_of: Option<String>,
    /// Every `--live` run shows its answer for a human check, whatever its age: its answer
    /// changes within days (the latest game, a spot price), or it abstains only because its
    /// saved pages lack the answer, which the live web may hold.
    #[serde(default)]
    pub live_check: bool,
}

/// Days after a changing row's `answer_as_of` that a live run still scores it.
pub const AS_OF_DAYS: i64 = 7;

/// The replay set's questions, every one, and its questions.jsonl; a regex that does not
/// compile, a day that does not parse or a row with no `answer` that is not `abstain` fails here.
pub fn replay_questions(dir: &Path) -> anyhow::Result<(Vec<ReplayQuestion>, PromptsFile)> {
    let (questions, file) = lines::<ReplayQuestion>(&dir.join("questions.jsonl"))?;
    for q in &questions {
        // No regex to miss would pass every answer.
        anyhow::ensure!(
            q.abstain || !q.answer.is_empty(),
            "{} has no answer regex and is not an abstain row",
            q.id
        );
        for p in q.answer.iter().chain(&q.not) {
            ignoring_case(p).with_context(|| format!("{}'s regex {p}", q.id))?;
        }
        dates::iso(&q.today).with_context(|| format!("{}'s today {:?}", q.id, q.today))?;
        if let Some(day) = &q.answer_as_of {
            dates::iso(day).with_context(|| format!("{}'s answer_as_of {day:?}", q.id))?;
        }
    }
    Ok((questions, file))
}

/// A saved search's hits and its pages in the order the agent fetched them: `hits.json` and
/// `1.json`, `2.json`, ... in `dir`.
pub fn saved(dir: &Path) -> anyhow::Result<(Vec<search::Hit>, Vec<Page>)> {
    let read = |name: String| {
        let path = dir.join(name);
        fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))
    };
    /// A saved hits.json; its other fields say how the search ran.
    #[derive(Deserialize)]
    struct Searched {
        hits: Vec<search::Hit>,
    }
    let searched: Searched = serde_json::from_str(&read("hits.json".into())?)?;
    let pages = (1..)
        .take_while(|n| dir.join(format!("{n}.json")).is_file())
        .map(|n| Ok(serde_json::from_str(&read(format!("{n}.json"))?)?))
        .collect::<anyhow::Result<_>>()?;
    Ok((searched.hits, pages))
}

/// `pattern`, case aside.
fn ignoring_case(pattern: &str) -> Result<regex::Regex, regex::Error> {
    regex::RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
}

/// The `answer` regexes of `q` that `text` misses, case aside: all of them unless one matches,
/// or, with `all`, each one that does not.
pub fn missed<'a>(q: &'a ReplayQuestion, text: &str) -> Vec<&'a str> {
    let fails: Vec<&str> = (q.answer.iter())
        .filter(|a| {
            !ignoring_case(a)
                .expect("replay_questions checked it")
                .is_match(text)
        })
        .map(String::as_str)
        .collect();
    match q.all || fails.len() == q.answer.len() {
        true => fails,
        false => Vec::new(),
    }
}

/// What an answer that says its pages do not answer holds besides `REFUSALS`, case aside: the
/// forms a reader writes in its own words, about the pages, what they do not cover, or an event
/// still to come or a thing that does not exist yet.
const ABSTAINED: &str = "pages? (?:don't|doesn't|do not|does not) \
(?:say|answer|mention|give|state|list)|(?:has|have)n't (?:been )?\\w+ yet|(?:has|have) not \
(?:yet )?been (?:played|held|released|announced|decided)|not (?:yet )?been \
(?:played|held|released|announced)|does not say|no information|not (?:in|on) the pages|cannot \
find|can't find|(?:has|have)(?: not|n't) (?:yet )?(?:happened|taken place|met|announced|decided)|\
not (?:happened|taken place) yet|\\bnot until\\b|\\bstill to come\\b|\\byet to (?:be|come|happen|\
take place|meet|announce|decide)\\b|\\bno (?:\\w+ ){1,3}yet\\b|(?:no one|nobody) (?:has|have) \
[^.;!?]*\\byet\\b|(?:do|does)(?: not|n't) cover|none of the pages";

/// Whether `answer` is right for `q`: for a question its pages do not answer, it says so
/// (`REFUSALS`, `ABSTAINED`); else it passes `missed`; and it matches none of `not`, so a refusal
/// that still states a planted answer is wrong. It is scored with its citations dropped, its
/// whitespace squashed and curly apostrophes straightened.
pub fn replay_right(q: &ReplayQuestion, answer: &str) -> bool {
    // A citation inside a sentence would break a regex's gap: "Villa [1] beat".
    static CITED: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"\[(?:[\d,\s-]|could not verify)+\]").expect("a valid pattern")
    });
    // A model writes the curly apostrophe as often as the straight one.
    let text = squash(&CITED.replace_all(answer, ""))
        .0
        .replace('\u{2019}', "'");
    let matches =
        |p: &str| (ignoring_case(p).expect("replay_questions checked it")).is_match(&text);
    let said = match q.abstain {
        true => {
            let lower = text.to_lowercase();
            REFUSALS.iter().any(|r| holds(&lower, r)) || matches(ABSTAINED)
        }
        false => missed(q, &text).is_empty(),
    };
    said && !q.not.iter().any(|p| matches(p))
}

/// The replay set's web questions with their saved searches and pages, ready for `run_replay`.
#[derive(Debug)]
pub struct ReplaySet {
    /// Its questions.jsonl.
    pub file: PromptsFile,
    /// The questions with saved pages, in file order, each with its hits and pages.
    pub questions: Vec<(ReplayQuestion, Vec<search::Hit>, Vec<Page>)>,
    /// The questions `run_replay` does not run: about files and saved reports, or web ones
    /// without saved pages (no `pages`).
    pub skipped: usize,
    /// The file `run_replay` copies the runs' audit log to after a row's error and at the end,
    /// as the data folder goes when the set does; `None` keeps no copy.
    pub audit: Option<PathBuf>,
    /// The set's folder, where a live run saves what it read (`web/<id>/live-<day>/`).
    pub dir: PathBuf,
    /// Runs of every question (`--repeat`), each kept; 1 by default.
    pub repeat: u32,
    /// A short question reads its snippets first, as in a chat (`WebAnswer::snippets`); off by
    /// default, the CLI's `web_answer::snippet_tier()`.
    pub snippets: bool,
    /// Snippets that do not answer have one page read first, as in a chat
    /// (`WebAnswer::one_page`); off by default, the CLI's `web_answer::one_page_tier()`.
    pub one_page: bool,
    /// A live run's plain reader prefills its pages as they come in, in a parked session, as in
    /// a chat (`WebAnswer::park`); the CLI's `web_answer::park()`, arrival order unless it says 0 or 1.
    pub park: Park,
    /// The plain reader writes the short answer (`WebAnswer::short`); off by default, the CLI's
    /// `web_answer::short_answer()`.
    pub short: bool,
    /// A throwaway data folder for the pages' saved copies, which the re-check reads, and the
    /// runs' audit lines, so the bench adds nothing to the user's data.
    pub data: tempfile::TempDir,
}

impl ReplaySet {
    /// The report's replay section before any question ran: `readers` on the chunks reader's
    /// `workers` and the chunks writer's `thinking`. The plain reader reads as a small model
    /// (`web_answer::READ_THINKING`, no pages' days) until `run_rows` sets the loaded model's.
    pub fn section(&self, readers: &[Reader], workers: Workers, thinking: Thinking) -> Replay {
        Replay {
            questions: self.file.clone(),
            readers: readers.to_vec(),
            skipped: self.skipped,
            workers_prefix_cache: workers.prefix_cache,
            plain_thinking_budget: web_answer::READ_THINKING,
            page_dates: false,
            short_answer: self.short,
            chunks_thinking_budget: thinking.budget,
            rows: Vec::new(),
            repeated: None,
        }
    }

    /// The web questions' day as a chat template prints today's date (`%d %b %Y`, "28 Oct
    /// 2026"), so a template that prints it agrees with the day the readers are told; `None`
    /// without web questions.
    pub fn date(&self) -> Option<String> {
        let (q, ..) = self.questions.first()?;
        template_date(&q.today)
    }

    /// The throwaway data folder the runs read and write: a live run's search reads its
    /// config.toml's keys.
    pub fn data_dir(&self) -> &Path {
        self.data.path()
    }

    /// Copies the runs' audit and search logs beside the report (`keep_logs`).
    fn keep_audit(&self) -> anyhow::Result<()> {
        self.audit
            .as_deref()
            .map_or(Ok(()), |to| keep_logs(self.data.path(), to))
    }
}

/// Copies `data`'s audit log to `to` as one file, over an earlier copy: its days' files in name
/// order, so one name matches the agent lane's `bench/results/agent-*`. A live run's searches
/// (provider, status, why others failed) go beside it as `<name>-search.jsonl`.
pub fn keep_logs(data: &Path, to: &Path) -> anyhow::Result<()> {
    copy_logs(data, to, false)
}

/// The search log `keep_logs` writes beside the audit log `to`.
fn search_log(to: &Path) -> PathBuf {
    to.to_string_lossy()
        .replace("-audit.jsonl", "-search.jsonl")
        .into()
}

/// `keep_logs`, or with `append` adding to `to` and its search log instead of replacing them:
/// a run whose rows each have a data folder of their own.
fn copy_logs(data: &Path, to: &Path, append: bool) -> anyhow::Result<()> {
    let put = |path: &Path, bytes: &[u8]| -> std::io::Result<()> {
        use std::io::Write;
        match append {
            true => (fs::OpenOptions::new().create(true).append(true).open(path))?.write_all(bytes),
            false => fs::write(path, bytes),
        }
    };
    let searches = data.join("agent").join("search").join("log.jsonl");
    if searches.is_file() {
        let name = search_log(to);
        put(&name, &fs::read(&searches)?)
            .with_context(|| format!("copying the search log to {}", name.display()))?;
    }
    let log = Audit::new(data);
    // No row has logged a line yet.
    if !log.dir().is_dir() {
        return Ok(());
    }
    let mut days: Vec<PathBuf> = (fs::read_dir(log.dir())?)
        .map(|f| Ok(f?.path()))
        .collect::<std::io::Result<_>>()?;
    // YYYY-MM-DD names sort by day.
    days.sort();
    let mut lines = Vec::new();
    for day in days {
        lines.extend(fs::read(day)?);
    }
    put(to, &lines).with_context(|| format!("copying the audit log to {}", to.display()))
}

/// Starts `to` and its search log empty, for a run that adds each row's logs (`append_logs`).
fn clear_logs(to: &Path) -> anyhow::Result<()> {
    for path in [to.to_path_buf(), search_log(to)] {
        match fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
            _ => {}
        }
    }
    Ok(())
}

/// Adds the audit and search logs of the data folder `data` to `to` and its search log.
fn append_logs(data: &Path, to: &Path) -> anyhow::Result<()> {
    copy_logs(data, to, true)
}

/// `day` (`YYYY-MM-DD`) as a chat template prints today's date (`%d %b %Y`, "28 Oct 2026").
pub fn template_date(day: &str) -> Option<String> {
    let written = dates::written(dates::iso(day)?);
    let [month, day, year] = written.split(' ').collect::<Vec<_>>()[..] else {
        return None;
    };
    Some(format!("{day:0>2} {} {year}", &month[..3]))
}

/// The replay set in `dir`: its web questions with their saved searches and pages, each page's
/// copy saved where the re-check reads it, and the count of the others. All that can be wrong
/// with the set shows here, so a bench fails on it before the model loads.
pub fn replay_set(dir: &Path) -> anyhow::Result<ReplaySet> {
    let (all, file) = replay_questions(dir)?;
    let mut set = ReplaySet {
        file,
        questions: Vec::new(),
        skipped: 0,
        audit: None,
        dir: dir.into(),
        repeat: 1,
        snippets: false,
        one_page: false,
        park: Park::Off,
        short: false,
        data: tempfile::tempdir().context("making a data folder for the replay runs")?,
    };
    for q in all {
        let Some(at) = &q.pages else {
            set.skipped += 1;
            continue;
        };
        let (hits, pages) = saved(&dir.join(at))?;
        for p in &pages {
            fetch::save(p, set.data.path())?;
        }
        set.questions.push((q, hits, pages));
    }
    // ponytail: the chat template prints one date a run (`date`); a set asked on several days
    // needs a template per question
    let days: BTreeSet<&str> = set
        .questions
        .iter()
        .map(|(q, ..)| q.today.as_str())
        .collect();
    anyhow::ensure!(
        days.len() <= 1,
        "the replay set's web questions must be asked on one day, as the chat template prints \
         one date a run: {days:?}"
    );
    Ok(set)
}

/// One replay question's answer from one reader.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayRow {
    /// The question's ID.
    pub id: String,
    /// The question's kind.
    pub kind: String,
    /// The reader that answered.
    pub reader: Reader,
    /// The question's pages do not answer it.
    pub abstain: bool,
    /// The answer is right (`replay_right`); never for an error.
    pub right: bool,
    /// The answer, whole, as the user reads it above its sources.
    pub answer: String,
    /// The notes under the answer: Rust's checks on it.
    pub notes: Vec<String>,
    /// The reader's run, end to end.
    pub wall_ms: f64,
    /// Prompt tokens of every model turn of the run.
    pub prompt_tokens: u64,
    /// Thinking tokens of the answer turns.
    pub thinking_tokens: u64,
    /// The engine error, or the panic, that ended the run.
    pub error: Option<String>,
    /// Why the plain reader's prompt did not fit the context, so the chunks reader answered and
    /// was timed (`fallback` in the read event); `None` when it read the pages itself.
    pub fallback: Option<String>,
    /// What the run came to (`outcome`); `None` in a report from before it.
    #[serde(default)]
    pub outcome: Option<Outcome>,
    /// Where its time went; `None` in a report from before it.
    #[serde(default)]
    pub stages: Option<Stages>,
    /// Which run of the question it is, from 1 (`--repeat`); 0 in a report from before it.
    #[serde(default)]
    pub run: u32,
    /// A live run's row marked `live_check`, or a changing one whose `answer_as_of` is over
    /// `AS_OF_DAYS` old: its answer is for a human to check, and it is not scored (`right` is
    /// false and counts nowhere).
    #[serde(default)]
    pub check: bool,
    /// The addresses its sources list names, in order (`urls`).
    #[serde(default)]
    pub sources: Vec<String>,
    /// Which tier answered ("snippets", "one page" or "pages", the tier event's); `None` when
    /// the snippet tier did not run.
    #[serde(default)]
    pub tier: Option<String>,
    /// Tokens the read turns generated past their thinking: the answer's own; 0 in a report from
    /// before it.
    #[serde(default)]
    pub answer_tokens: u64,
}

/// A `--replay` run: each web question of the replay set answered by each reader.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Replay {
    /// The set's questions.jsonl.
    pub questions: PromptsFile,
    /// The readers each question ran on, in order (`--reader`).
    pub readers: Vec<Reader>,
    /// The questions about files and saved reports, not run.
    pub skipped: usize,
    /// The chunks reader's workers anchored their system prompt in the prefix cache; off, as the
    /// desktop app opens its chats.
    pub workers_prefix_cache: bool,
    /// The plain reader's thinking budget, its own (`web_answer::read_thinking`).
    pub plain_thinking_budget: u32,
    /// The plain reader's prompt ended with the pages' days (`web_answer::page_dates`); false in a
    /// report from before it.
    #[serde(default)]
    pub page_dates: bool,
    /// The plain reader wrote the short answer (`web_answer::SHORT_VAR`); false in a report from
    /// before it.
    #[serde(default)]
    pub short_answer: bool,
    /// The chunks writer's thinking budget (`--thinking-budget`).
    pub chunks_thinking_budget: u32,
    /// One per question and reader: the questions in file order, each one's readers in order,
    /// whichever ran first, run after run.
    pub rows: Vec<ReplayRow>,
    /// The rows over their runs, with `--repeat` above 1; `None` otherwise.
    #[serde(default)]
    pub repeated: Option<Repeated>,
}

/// Answers each web question of `set` with each of `readers` on its saved search and pages, as a
/// chat's web_answer reads fetched pages (`WebAnswer::read`) with config.toml's defaults but
/// `thinking`, on the question's own day; nothing is searched or fetched. One untimed warm-up
/// turn goes first, and odd-numbered questions (from 0) run the readers in reverse; the set's
/// `repeat` runs go one after another, each kept. Each answer is scored (`replay_right`) into
/// `report.replay`, which `save` writes after each, so a run that stops keeps what it measured.
/// An engine error or a panic ends only its row, and the audit log is copied to `set.audit` after
/// it and at the end; `progress` gets a line for each row.
#[allow(clippy::too_many_arguments)]
pub fn run_replay(
    larch: &Larch,
    family: Family,
    thinking: Thinking,
    workers: Workers,
    set: &ReplaySet,
    readers: &[Reader],
    report: &mut Report,
    save: impl FnMut(&Report) -> anyhow::Result<()>,
    progress: impl FnMut(String),
) -> anyhow::Result<()> {
    let at = Section::Replay;
    run_rows(
        larch, family, thinking, workers, set, readers, at, report, save, progress,
    )
}

/// `run_replay` on the live web (`--live`): each question searched with `tools`' `web_search` and
/// its pages fetched once a run, on `today` (`YYYY-MM-DD`), then read by each reader. The hits and
/// pages are saved as the set saves them, in `web/<id>/live-<today>/` of the set's folder, so a
/// live run can refresh the set. A `live_check` row, or a changing one (`answer_as_of`) over
/// `AS_OF_DAYS` old, is shown for a human check, not scored. The rows go in `report.live`, never in `report.replay`.
#[allow(clippy::too_many_arguments)]
pub fn run_live(
    larch: &Larch,
    family: Family,
    thinking: Thinking,
    workers: Workers,
    set: &ReplaySet,
    readers: &[Reader],
    tools: &Registry,
    today: &str,
    report: &mut Report,
    save: impl FnMut(&Report) -> anyhow::Result<()>,
    progress: impl FnMut(String),
) -> anyhow::Result<()> {
    let at = Section::Live(tools, today);
    run_rows(
        larch, family, thinking, workers, set, readers, at, report, save, progress,
    )
}

/// Which section a replay run fills: the saved pages, or the live web with its tools and day.
#[derive(Clone, Copy)]
enum Section<'a> {
    Replay,
    Live(&'a Registry, &'a str),
}

/// `run_replay` and `run_live`.
#[allow(clippy::too_many_arguments)]
fn run_rows(
    larch: &Larch,
    family: Family,
    thinking: Thinking,
    workers: Workers,
    set: &ReplaySet,
    readers: &[Reader],
    at: Section,
    report: &mut Report,
    mut save: impl FnMut(&Report) -> anyhow::Result<()>,
    mut progress: impl FnMut(String),
) -> anyhow::Result<()> {
    let ctx = ToolCtx {
        allowed: Vec::new(),
        data_dir: set.data.path().into(),
        source: String::new(),
    };
    let docs = AskDocs {
        family: Some(family),
        thinking,
        ..AskDocs::default()
    };
    fn section<'r>(report: &'r mut Report, at: Section) -> &'r mut Option<Replay> {
        match at {
            Section::Replay => &mut report.replay,
            Section::Live(..) => &mut report.live,
        }
    }
    let params_b = larch.deployment().params_b;
    let (plain_thinking, dates) = (
        web_answer::read_thinking(params_b),
        web_answer::page_dates(params_b),
    );
    *section(report, at) = Some(Replay {
        plain_thinking_budget: plain_thinking,
        page_dates: dates,
        ..set.section(readers, workers, thinking)
    });
    // So the first row does not pay the engine's cold start: weights paged in, GPU pipelines
    // built. Namespace 0 is no reader's, so it leaves their prefix cache as it was.
    if let Err(e) = run::anchor(larch, "", &GenParams::default()) {
        progress(format!("the warm-up turn failed: {e}"));
    }
    for run in 1..=set.repeat.max(1) {
        for (i, (q, hits, pages)) in set.questions.iter().enumerate() {
            // Odd questions run the readers in reverse, so neither always runs on a GPU the other
            // warmed.
            let mut order = readers.to_vec();
            if i % 2 == 1 {
                order.reverse();
            }
            // A live question is searched and fetched once a run, as the chunks reader fetches
            // (the more pages) unless the plain one runs alone; each reader reads those pages.
            // The plain one alone parks its session as a chat's does, and the pages it fetches
            // are prefilled in it as they come in.
            let mut parked = None;
            let read = match at {
                Section::Replay => Ok((hits.clone(), pages.clone(), Vec::new(), 0.0)),
                Section::Live(tools, today) => {
                    let reader = match readers {
                        [one] => *one,
                        _ => Reader::Chunks,
                    };
                    let mut skill = WebAnswer {
                        docs,
                        reader,
                        snippets: set.snippets,
                        one_page: set.one_page,
                        short: set.short,
                        race: web_answer::race(),
                        read_thinking: plain_thinking,
                        dates,
                        ..WebAnswer::default()
                    };
                    let one_page = skill.tiered(&q.question) && skill.one_page;
                    if reader == Reader::Plain && set.park != Park::Off && !one_page {
                        parked = web_answer::parking(larch);
                    }
                    // Only the fetch that feeds a parked read takes the arrival order.
                    if parked.is_some() {
                        skill.park = set.park;
                    }
                    let saved_at = (set.dir.join("web").join(&q.id)).join(format!("live-{today}"));
                    let fetching = (larch, parked.as_ref().map(|p| &p.0));
                    live_pages(&ctx, &skill, tools, q, today, &saved_at, fetching)?
                }
            };
            let start = section(report, at).as_ref().map_or(0, |r| r.rows.len());
            for reader in order {
                let skill = WebAnswer {
                    docs,
                    reader,
                    snippets: set.snippets,
                    one_page: set.one_page,
                    park: set.park,
                    short: set.short,
                    read_thinking: plain_thinking,
                    dates,
                    ..WebAnswer::default()
                };
                let parked = (parked.as_ref())
                    .filter(|_| reader == Reader::Plain)
                    .map(|p| &p.0);
                let mut row = match &read {
                    Ok((hits, pages, before, spent)) => {
                        let today = match at {
                            Section::Replay => q.today.as_str(),
                            Section::Live(_, today) => today,
                        };
                        let read = (hits.as_slice(), pages.as_slice(), before.as_slice(), parked);
                        let mut row = replay(larch, &ctx, &skill, workers, q, today, read);
                        row.wall_ms += spent;
                        row
                    }
                    // The search or the fetches failed: the answer is the line that says so, and
                    // the row is an error, never a wrong answer.
                    Err((line, before, spent)) => ReplayRow {
                        id: q.id.clone(),
                        kind: q.kind.clone(),
                        reader,
                        abstain: q.abstain,
                        answer: line.clone(),
                        error: Some(line.clone()),
                        wall_ms: *spent,
                        outcome: Some(outcome(Some(line), true, false, q.abstain)),
                        stages: Some(stages(before, None)),
                        ..ReplayRow::default()
                    },
                };
                row.run = run;
                if let Section::Live(_, today) = at {
                    row.check = stale(q, today);
                    row.right &= !row.check;
                }
                let line = format!(
                    "{} {} {} {:.1} s",
                    row.id,
                    named(reader),
                    verdict(&row),
                    row.wall_ms / 1e3
                );
                let errored = row.error.is_some();
                if let Some(r) = section(report, at) {
                    r.rows.push(row);
                    r.rows[start..].sort_by_key(|r| readers.iter().position(|&x| x == r.reader));
                }
                save(report)?;
                if errored {
                    set.keep_audit()?;
                }
                progress(line);
            }
        }
    }
    if let Some(r) = section(report, at).as_mut().filter(|_| set.repeat > 1) {
        let scored = r.rows.iter().filter(|r| !r.check);
        r.repeated = Some(repeated(
            scored.map(|r| (format!("{} {}", r.id, named(r.reader)), r.right)),
        ));
        save(report)?;
    }
    set.keep_audit()
}

/// Whether a live run on `today` shows `q`'s answer for a human check: it says so
/// (`live_check`), or its answer changes and was last checked more than `AS_OF_DAYS` before.
fn stale(q: &ReplayQuestion, today: &str) -> bool {
    let day = |d: &str| dates::iso(d).expect("replay_questions and the caller checked it");
    q.live_check
        || (q.answer_as_of.as_deref()).is_some_and(|as_of| day(today) - day(as_of) > AS_OF_DAYS)
}

/// What `replay` reads: the hits, the pages, the steps that found them and the parked session.
type ReadPages<'a> = (
    &'a [search::Hit],
    &'a [Page],
    &'a [AuditEvent],
    Option<&'a Mutex<Session>>,
);

/// What a live question read: its hits, its pages, the search and fetch steps and their wall time;
/// or the line a chat would show when the search or every fetch failed, with those steps.
type LivePages =
    Result<(Vec<search::Hit>, Vec<Page>, Vec<AuditEvent>, f64), (String, Vec<AuditEvent>, f64)>;

/// Searches and fetches `q` as web_answer does on `today`, with `skill`'s page count, and saves
/// the hits and pages in `saved_at` as `saved` reads them, over an earlier run's. With a parked
/// session, the pages are prefilled in it on `larch` as they come in (`fetch_parked`).
fn live_pages(
    ctx: &ToolCtx,
    skill: &WebAnswer,
    tools: &Registry,
    q: &ReplayQuestion,
    today: &str,
    saved_at: &Path,
    (larch, parked): (&Larch, Option<&Mutex<Session>>),
) -> anyhow::Result<LivePages> {
    let mut log = Log::new(ctx, web_answer::NAME);
    let day = dates::iso(today).context("the live run's day is YYYY-MM-DD")?;
    let asked = dates::relative(&q.question, day);
    let started = Instant::now();
    let stop = AtomicBool::default();
    let fetched = match parked {
        Some(parked) => skill.fetch_parked(
            larch,
            ctx,
            &q.question,
            (asked, day),
            tools,
            Seen::default(),
            &mut log,
            &stop,
            parked,
            None,
        ),
        // No prefill: the replay reads the pages later, on its own engine run.
        None => skill.fetch(
            ctx,
            &q.question,
            (asked, day),
            tools,
            Seen::default(),
            &mut log,
            &stop,
            &mut |_, _| None,
            None,
        ),
    };
    let spent = ms(started);
    let (hits, pages, _) = match fetched {
        Ok(Ok(read)) => read,
        Ok(Err(line)) => return Ok(Err((line, log.events, spent))),
        Err(e) => return Ok(Err((e.to_string(), log.events, spent))),
    };
    let pages: Vec<Page> = pages.into_iter().map(|(_, p)| p).collect();
    if saved_at.exists() {
        fs::remove_dir_all(saved_at)?;
    }
    fs::create_dir_all(saved_at)?;
    let searched = serde_json::json!({
        "query": q.question,
        "searched_at": larch::bench::utc_now(),
        "hits": hits,
    });
    fs::write(
        saved_at.join("hits.json"),
        serde_json::to_string_pretty(&searched)?,
    )?;
    for (n, p) in (1..).zip(&pages) {
        fs::write(
            saved_at.join(format!("{n}.json")),
            serde_json::to_string_pretty(p)?,
        )?;
    }
    Ok(Ok((hits, pages, log.events, spent)))
}

/// `skill`'s answer to `q` on `hits` and `pages` on `today`, scored: the pages a fetch would have
/// read (the first `skill.pages`), each keyed as a fetch keys it. `before` are the steps that
/// found them (a live search and fetch), which its stages count; the plain reader's turn is
/// `parked`'s next request when given, as the live fetch prefilled the pages in it.
fn replay(
    larch: &Larch,
    ctx: &ToolCtx,
    skill: &WebAnswer,
    workers: Workers,
    q: &ReplayQuestion,
    today: &str,
    (hits, pages, before, parked): ReadPages,
) -> ReplayRow {
    let mut fetched: Vec<(PathBuf, Page)> = (pages.iter().take(skill.pages))
        .map(|p| (web_answer::key(&p.final_url), p.clone()))
        .collect();
    let mut log = Log::new(ctx, web_answer::NAME);
    let started = Instant::now();
    let read = panic::catch_unwind(AssertUnwindSafe(|| {
        let stop = AtomicBool::default();
        let question = &q.question;
        // The snippet tier first, as a chat's short question runs it.
        let day = dates::iso(today).expect("a replay day is YYYY-MM-DD");
        let mut snippet = None;
        if skill.tiered(question) {
            let (sourced, chosen, event) = skill.snippets(larch, ctx, question, day, hits, &stop);
            log.write(event)?;
            // The named page past the first ones is fetched too, as a chat's fetch takes it.
            let named = (chosen.filter(|_| skill.one_page && sourced.is_none()))
                .and_then(|n| hits.get(n - 1))
                .and_then(|h| {
                    pages
                        .iter()
                        .find(|p| p.url == h.url || p.final_url == h.url)
                })
                .filter(|p| !fetched.iter().any(|(_, f)| f.final_url == p.final_url));
            if let Some(p) = named {
                fetched.push((web_answer::key(&p.final_url), p.clone()));
            }
            snippet = Some((sourced, chosen));
        }
        skill.tiers(
            larch, ctx, workers, question, today, hits, fetched, snippet, &mut log, &stop, parked,
        )
    }));
    let mut row = ReplayRow {
        id: q.id.clone(),
        kind: q.kind.clone(),
        reader: skill.reader,
        abstain: q.abstain,
        wall_ms: ms(started),
        prompt_tokens: (log.events.iter())
            .filter_map(|e| e.prompt_tokens.map(u64::from))
            .sum(),
        thinking_tokens: (log.events.iter())
            .filter_map(|e| e.detail.as_ref()?["thinking_tokens"].as_u64())
            .sum(),
        answer_tokens: (log.events.iter())
            .filter(|e| e.kind == EventKind::Turn && e.step.as_deref() == Some("read"))
            .filter_map(|e| {
                let thought = e.detail.as_ref()?["thinking_tokens"].as_u64()?;
                Some(u64::from(e.generated_tokens?).saturating_sub(thought))
            })
            .sum(),
        // The read event's: the note under the answer is left out over "not found".
        fallback: (log.events.iter())
            .filter(|e| e.step.as_deref() == Some("read"))
            .find_map(|e| Some(e.detail.as_ref()?["fallback"].as_str()?.to_owned())),
        tier: (log.events.iter())
            .filter(|e| e.step.as_deref() == Some("tier"))
            .find_map(|e| Some(e.detail.as_ref()?["answered"].as_str()?.to_owned())),
        ..ReplayRow::default()
    };
    row.stages = Some(stages(&[before, &log.events].concat(), None));
    match read {
        Ok(Ok((answer, markdown, ..))) => {
            // The paragraphs under the answer but its sources.
            let under = markdown.strip_prefix(answer.as_str()).unwrap_or_default();
            row.notes = (under.split("\n\n"))
                .filter(|p| !p.is_empty() && !p.starts_with("Sources:"))
                .map(String::from)
                .collect();
            row.right = replay_right(q, &answer);
            row.sources = urls(under);
            row.answer = answer;
        }
        Ok(Err(e)) => row.error = Some(e.to_string()),
        Err(panic) => {
            let why = (panic.downcast_ref::<&str>().copied())
                .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                .unwrap_or_default();
            row.error = Some(format!("panicked: {why}"));
        }
    }
    row.outcome = Some(outcome(
        Some(&row.answer),
        row.error.is_some(),
        row.right,
        q.abstain,
    ));
    row
}

/// The reader's name, as `--reader` takes it.
fn named(reader: Reader) -> String {
    format!("{reader:?}").to_lowercase()
}

/// A replay row's outcome: pass, fail, error, or check (for a human, unscored).
fn verdict(row: &ReplayRow) -> &'static str {
    match (&row.error, row.right) {
        (Some(_), _) => "error",
        (None, _) if row.check => "check",
        (None, true) => "pass",
        (None, false) => "fail",
    }
}

/// The share of answerable rows, in percent, plain must answer right to become the default, with
/// every abstain row right (the 28 Oct ruling: 31 of the set's 36).
const BAR_RIGHT: usize = 85;

/// The most plain's paired median time may be over chunks' for plain to become the default.
const BAR_TIME: f64 = 1.2;

/// The replay run as the bench prints it: the questions skipped and why; each reader's right,
/// wrong and errors by kind; each reader's right answers of answerable and of abstain questions,
/// time p50 and p90 over the `n` rows that did not error, their mean prompt and thinking tokens,
/// and the rows the plain reader fell back to chunks on; then, with two readers, the median of
/// plain's time over chunks' on each question neither errored on and plain did not fall back on,
/// the decision against `BAR_RIGHT` and `BAR_TIME`, and each answerable question they disagree
/// on, with the start of each answer; last, each abstain question with each reader's answer whole.
pub fn replay_table(replay: &Replay) -> String {
    let s = if replay.skipped == 1 { "" } else { "s" };
    let mut out = format!(
        "replay: {} rows; {} file and report question{s} skipped: plain reader for files is a \
         later task\n{:<8}{:<12}{:>6}{:>6}{:>7}\n",
        replay.rows.len(),
        replay.skipped,
        "reader",
        "kind",
        "right",
        "wrong",
        "errors"
    );
    // `reader`'s rows the chunks reader answered for the plain one.
    let fell_back = |reader: Reader| {
        (replay.rows.iter())
            .filter(|r| r.reader == reader && r.fallback.is_some())
            .count()
    };
    // `reader`'s right answers of its answerable or abstain rows, and those rows.
    let right = |reader: Reader, abstain: bool| {
        let of =
            (replay.rows.iter()).filter(|r| r.reader == reader && r.abstain == abstain && !r.check);
        (of.clone().filter(|r| r.right).count(), of.count())
    };
    for &reader in &replay.readers {
        let rows: Vec<&ReplayRow> = (replay.rows.iter())
            .filter(|r| r.reader == reader)
            .collect();
        let mut kinds: BTreeMap<&str, [usize; 3]> = BTreeMap::new();
        for r in &rows {
            let at = match verdict(r) {
                "pass" => 0,
                "fail" => 1,
                "check" => continue,
                _ => 2,
            };
            kinds.entry(&r.kind).or_default()[at] += 1;
        }
        for (kind, [right, wrong, errors]) in kinds {
            out += &format!(
                "{:<8}{kind:<12}{right:>6}{wrong:>6}{errors:>7}\n",
                named(reader)
            );
        }
        let ((a, an), (b, bn)) = (right(reader, false), right(reader, true));
        let ran: Vec<&&ReplayRow> = rows.iter().filter(|r| r.error.is_none()).collect();
        let mut wall: Vec<f64> = ran.iter().map(|r| r.wall_ms / 1e3).collect();
        wall.sort_by(f64::total_cmp);
        let mean = |f: fn(&ReplayRow) -> u64| {
            ran.iter().map(|r| f(r)).sum::<u64>() as f64 / ran.len().max(1) as f64
        };
        out += &format!(
            "{}: answerable {a}/{an} right, abstain {b}/{bn} right; time p50 {:.1} s, p90 {:.1} s \
             (n {}); mean {:.0} prompt tokens, {:.0} thinking tokens; fell back {}\n",
            named(reader),
            nearest(&wall, 50.0),
            nearest(&wall, 90.0),
            ran.len(),
            mean(|r| r.prompt_tokens),
            mean(|r| r.thinking_tokens),
            fell_back(reader)
        );
    }
    // A question's line: each reader's verdict, then the first `chars` characters of each one's
    // answer, on one line.
    let line = |rows: &[ReplayRow], chars: usize| {
        let verdicts: Vec<String> = (rows.iter())
            .map(|r| format!("{} {}", named(r.reader), verdict(r)))
            .collect();
        let answers: Vec<String> = (rows.iter())
            .map(|r| {
                let text = (r.error.as_ref()).map_or(r.answer.clone(), |e| format!("error: {e}"));
                let head: String = squash(&text).0.chars().take(chars).collect();
                format!("{}: {head}", named(r.reader))
            })
            .collect();
        format!(
            "{} {} | {}\n",
            rows[0].id,
            verdicts.join(", "),
            answers.join(" | ")
        )
    };
    if replay.readers.len() == 2 {
        // Plain's time over chunks' on each question neither errored on, so a reader's errors on
        // heavy questions do not make its times look short, and plain read itself, as a
        // fallback's time is the chunks reader's.
        let mut ratios: Vec<f64> = (replay.rows.chunks_exact(2))
            .filter_map(|pair| match pair {
                [p, c] | [c, p] if p.reader == Reader::Plain && c.reader == Reader::Chunks => {
                    (p.error.is_none() && c.error.is_none() && p.fallback.is_none())
                        .then(|| p.wall_ms / c.wall_ms)
                }
                _ => None,
            })
            .collect();
        ratios.sort_by(f64::total_cmp);
        // Between the middle two of an even count, not the lower one, which would favour plain.
        let n = ratios.len();
        let ratio = (n > 0).then(|| (ratios[(n - 1) / 2] + ratios[n / 2]) / 2.0);
        let times = ratio.map_or("none".into(), |r| format!("{r:.2}x"));
        let fell = fell_back(Reader::Plain);
        let (s, them) = if fell == 1 { ("", "it") } else { ("s", "them") };
        out += &format!(
            "paired time plain/chunks: median {times} (n {}), without the {fell} question{s} plain \
             fell back on, as the chunks reader answered {them}\n",
            ratios.len()
        );
        let ((a, an), (b, bn)) = (right(Reader::Plain, false), right(Reader::Plain, true));
        let need = (an * BAR_RIGHT).div_ceil(100);
        let pass = [a >= need, b == bn, ratio.is_some_and(|r| r <= BAR_TIME)];
        let yes = |ok: bool| if ok { "yes" } else { "no" };
        out += &format!(
            "decision, plain as the default: answerable {a}/{an} right, bar {need} ({BAR_RIGHT}%) \
             {}; abstain {b}/{bn} right, bar {bn} {}; paired time {times}, bar {BAR_TIME}x {}: \
             plain {}, fell back {fell}\n",
            yes(pass[0]),
            yes(pass[1]),
            yes(pass[2]),
            match pass.iter().all(|&ok| ok) {
                true => "passes",
                false => "does not pass",
            }
        );
        // The abstain rows are all listed below.
        out += "disagree:\n";
        for pair in replay.rows.chunks_exact(2) {
            if let [a, b] = pair
                && a.right != b.right
                && !a.abstain
            {
                out += &line(pair, 160);
            }
        }
    }
    // Each one whole, so the owner checks by eye the abstain bar a `not` regex can miss.
    out += "abstain:\n";
    for rows in replay.rows.chunks(replay.readers.len().max(1)) {
        if rows[0].abstain {
            out += &line(rows, usize::MAX);
        }
    }
    // A live run's changing rows checked too long ago, whole, for the owner to judge.
    let checks = replay.rows.chunks(replay.readers.len().max(1));
    let checks: Vec<_> = checks.filter(|rows| rows[0].check).collect();
    if !checks.is_empty() {
        out += "human check (answer_as_of over 7 days old, not scored):\n";
        for rows in checks {
            out += &line(rows, usize::MAX);
        }
    }
    if let Some(r) = &replay.repeated {
        out += &format!(
            "repeat: {} runs, mean right {:.3}, all_pass {:.3} of {} rows\n",
            r.runs,
            r.mean,
            r.all_pass,
            r.rows.len()
        );
        for row in r.rows.iter().filter(|r| !r.all_pass) {
            out += &format!("  {} {}/{} right\n", row.id, row.passes, row.runs);
        }
    }
    out
}

/// What a scored row's run came to: `answered`, `refused`, `no_files`, `unreadable`, `wrong` or
/// `error`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// It answered, and the answer scored right (or was not scored).
    #[default]
    Answered,
    /// The tool, or the answer, said it could not answer.
    Refused,
    /// The tool found nothing to read: no files, no pages.
    NoFiles,
    /// The tool found something it could not read: a scanned PDF, pages that would not fetch.
    Unreadable,
    /// It answered, and the answer scored wrong.
    Wrong,
    /// The run failed: an engine error, a panic, a search that failed.
    Error,
}

/// The starts of the first lines of a tool result that found nothing to read, matched case aside:
/// ask_docs's (#400) and the chat's fixed answer with no folder attached, and web_answer's.
const NO_FILES: [&str; 4] = [
    "No readable files",
    "No files match",
    "I can't read your files in this chat yet",
    web_answer::NO_PAGES,
];

/// What a tool result's first line holds when it found something it could not read, matched case
/// aside.
const UNREADABLE: [&str; 4] = [
    "No text in",
    "could not be read",
    "has no text",
    web_answer::FETCH_FAILED,
];

/// A row's outcome from `result`, the tool's result where a tool ran (else the answer): `Error`
/// when the run failed or the search did; `NoFiles` or `Unreadable` by its first line; after
/// ask_docs's "Read N files (...)" line (#400), what the rest says, `Unreadable` when it refused
/// and the line names files with no text; `Answered` when it is `right` on a row that is not an
/// `abstain` one ("I'm not sure of the exact date, but it was 3 Oct"); a refusal (`NOT_FOUND`,
/// `REFUSALS`); else `Answered` or `Wrong` by `right`. The one place the mapping lives: a new
/// first line goes here.
pub fn outcome(result: Option<&str>, error: bool, right: bool, abstain: bool) -> Outcome {
    let text = (result.unwrap_or_default().to_lowercase()).replace('\u{2019}', "'");
    let (line, rest) = text.split_once('\n').unwrap_or((&text, ""));
    let refuses = |t: &str| {
        let first = t.trim_start().lines().next().unwrap_or_default();
        [NOT_FOUND, web_answer::NOT_FOUND, ask_docs::WITHHELD]
            .iter()
            .any(|s| first.starts_with(&s.to_lowercase()))
            || REFUSALS.iter().any(|r| holds(first, r))
    };
    static READ: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"^read \d+ files?\b").expect("a valid pattern"));
    let found = |list: &[&str]| list.iter().any(|s| line.contains(&s.to_lowercase()));
    match () {
        _ if error || line.starts_with(&web_answer::SEARCH_FAILED.to_lowercase()) => Outcome::Error,
        _ if READ.is_match(line) => match refuses(rest) {
            true if line.contains("; no text in") => Outcome::Unreadable,
            true => Outcome::Refused,
            false if right => Outcome::Answered,
            false => Outcome::Wrong,
        },
        _ if found(&NO_FILES) => Outcome::NoFiles,
        _ if found(&UNREADABLE) => Outcome::Unreadable,
        _ if right && !abstain => Outcome::Answered,
        _ if refuses(line) => Outcome::Refused,
        _ if right => Outcome::Answered,
        _ => Outcome::Wrong,
    }
}

/// Where a row's time went, from its audit lines; a stage that did not run is `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stages {
    /// The web searches (`search` steps).
    #[serde(default)]
    pub search_ms: Option<f64>,
    /// The page fetches (`fetch` steps).
    #[serde(default)]
    pub fetch_ms: Option<f64>,
    /// The model turns' prompt reading: the sum of their times to first token. Workers run
    /// their turns at once, so it can be more than the row's wall time; the audit lines have no
    /// start times to merge the turns' spans by.
    #[serde(default, alias = "prefill_ms")]
    pub prefill_ms_sum: Option<f64>,
    /// The model turns' writing: the sum of their tokens over their speed, as `prefill_ms_sum`.
    #[serde(default, alias = "generation_ms")]
    pub generation_ms_sum: Option<f64>,
    /// The quote re-check (`recheck` steps).
    #[serde(default)]
    pub recheck_ms: Option<f64>,
    /// From the start of the turn to the first text or progress line the user would see; `None`
    /// where no chat ran (a skill run alone).
    #[serde(default)]
    pub first_visible_ms: Option<f64>,
}

/// The stages of a run whose audit lines are `events`, with `first_visible_ms` as the caller
/// timed it. Each turn's prefill is its time to first token, its generation its tokens over its
/// speed, summed: wall time where the turns run one after another (a chat), more than it where
/// workers run them at once (the chunks reader, ask_docs).
pub fn stages(events: &[AuditEvent], first_visible_ms: Option<f64>) -> Stages {
    let steps = |name: &str| {
        let ms = (events.iter())
            .filter(|e| e.kind == EventKind::Step && e.step.as_deref() == Some(name))
            .filter_map(|e| e.wall_ms);
        ms.reduce(|a, b| a + b)
    };
    let turns = events.iter().filter(|e| e.kind == EventKind::Turn);
    let prefill = turns.clone().filter_map(|e| e.ttft_ms).reduce(|a, b| a + b);
    let generation = (turns.filter_map(|e| Some((e.generated_tokens?, e.tok_s?))))
        .map(|(n, speed)| {
            if speed > 0.0 {
                f64::from(n) / speed * 1e3
            } else {
                0.0
            }
        })
        .reduce(|a, b| a + b);
    Stages {
        search_ms: steps("search"),
        fetch_ms: steps("fetch"),
        prefill_ms_sum: prefill,
        generation_ms_sum: generation,
        recheck_ms: steps("recheck"),
        first_visible_ms,
    }
}

/// One row over a `--repeat` run's runs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepeatedRow {
    /// The row's ID (with its reader in a replay run).
    pub id: String,
    /// Its runs.
    pub runs: u32,
    /// The runs it passed.
    pub passes: u32,
    /// It passed every run.
    pub all_pass: bool,
}

/// A section over a `--repeat` run's runs: the mean beside the rows that passed every run.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Repeated {
    /// The most runs any row had.
    pub runs: u32,
    /// Passes over all runs of all rows.
    pub mean: f64,
    /// The share of rows that passed every run.
    pub all_pass: f64,
    /// Each row, in the order it first ran.
    pub rows: Vec<RepeatedRow>,
}

/// `runs`, one `(row ID, passed)` per run of a row, gathered by row.
pub fn repeated(runs: impl IntoIterator<Item = (String, bool)>) -> Repeated {
    let mut rows: Vec<RepeatedRow> = Vec::new();
    let (mut passes, mut total) = (0, 0);
    for (id, pass) in runs {
        let at = match rows.iter().position(|r| r.id == id) {
            Some(at) => at,
            None => {
                rows.push(RepeatedRow {
                    id,
                    all_pass: true,
                    ..RepeatedRow::default()
                });
                rows.len() - 1
            }
        };
        let r = &mut rows[at];
        r.runs += 1;
        r.passes += u32::from(pass);
        r.all_pass &= pass;
        (passes, total) = (passes + u32::from(pass), total + 1);
    }
    let share = |n: usize, of: usize| if of > 0 { n as f64 / of as f64 } else { 0.0 };
    Repeated {
        runs: rows.iter().map(|r| r.runs).max().unwrap_or_default(),
        mean: share(passes as usize, total),
        all_pass: share(rows.iter().filter(|r| r.all_pass).count(), rows.len()),
        rows,
    }
}

/// The everyday set's categories, as a row's `category` and its ID name them.
pub const CATEGORIES: [&str; 13] = [
    "homework",
    "papers",
    "life",
    "health",
    "philosophy",
    "writing",
    "travel",
    "money",
    "cooking",
    "tech",
    "smalltalk",
    "followup",
    "myfiles",
];

/// The tools that read or write the user's files, read_papers too (a chat with nothing attached
/// still shows it, for links): with nothing attached, a call to any is a miss.
pub const FILE_READS: [&str; 7] = [
    "list_dir",
    "read_file",
    "find_files",
    "search_files",
    "write_file",
    "ask_docs",
    "read_papers",
];

/// What a reply says when it claims an ability the agent lacks, lower case with a straight
/// apostrophe: reminders, mail, messages, bookings.
pub const INVENTED: [&str; 12] = [
    "i've set a reminder",
    "i have set a reminder",
    "i'll remind you",
    "i will remind you",
    "i sent the email",
    "i've sent the email",
    "i'll email you",
    "i will email you",
    "i'll text you",
    "i've booked",
    "i've scheduled",
    "added it to your calendar",
];

/// What a reply says when it claims to have read the user's files, lower case with a straight
/// apostrophe, each in the first person and past tense: one counts only where a sentence opens
/// with it, so "If I read your question right" and "attach your notes and I can tell you what
/// your notes say" are no claims.
pub const CLAIMED_FILES: [&str; 7] = [
    "i read your",
    "i've read your",
    "i have read your",
    "i looked at your",
    "i've looked at your",
    "i checked your",
    "i've checked your",
];

/// What a reply's first sentence says when it refuses a plain request whole, lower case with a
/// straight apostrophe (`MustNot::Deflect`, with `OFFLINE`): "I can't tell you what to choose,
/// but" and "I don't have feelings like people do, but" answer, so `REFUSALS` would be too wide.
pub const DEFLECTIONS: [&str; 5] = [
    "i can't help with",
    "i'm not able to help with",
    "i cannot help with",
    "i can't assist",
    "i'm unable to help",
];

/// The fewest words a statement before a reply's first `?` needs for the reply to have answered
/// before it asks ("Sure. Which city?" has none).
pub const ASK_BACK_WORDS: usize = 3;

/// `text`'s sentences, split at `.`, `!`, `?` and line ends, trimmed, the empty ones left out.
fn sentences(text: &str) -> impl Iterator<Item = &str> {
    (text.split(['.', '!', '?', '\n']).map(str::trim)).filter(|s| !s.is_empty())
}

/// Whether `said` asks before it answers: it holds a `?`, no statement before its first holds
/// `ASK_BACK_WORDS` words, as when it opens with the question, and it ends on a question, so a
/// joke's set-up and punchline is an answer. "World War I started in 1914 and ended in 1918.
/// Want to know what caused it?" is an answer with an offer after it; `score_everyday` counts a
/// question back only where the row's answer checks fail too.
pub fn asks_back(said: &str) -> bool {
    // A report's headings are often questions; they are titles, not a question back.
    let said: String = (said.lines())
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    // An emoji or a closing bracket after the last mark does not end a sentence.
    let last = said
        .chars()
        .rev()
        .find(|c| c.is_alphanumeric() || ".!?".contains(*c));
    let Some((before, _)) = said.split_once('?').filter(|_| last == Some('?')) else {
        return false;
    };
    // The last piece is the question's own words.
    let mut pieces: Vec<&str> = before.split(['.', '!', '\n']).collect();
    pieces.pop();
    !(pieces.iter()).any(|s| s.split_whitespace().count() >= ASK_BACK_WORDS)
}

/// Where an everyday row's question belongs: the model alone, the web, or research.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Route {
    /// No tool: the model answers (`"none"`).
    #[default]
    #[serde(rename = "none")]
    Model,
    /// web_answer.
    Web,
    /// research.
    Research,
}

/// What an everyday reply must not do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MustNot {
    /// Refuse or deflect a plain request whole (`DEFLECTIONS`, `OFFLINE`), in its first sentence.
    Deflect,
    /// Claim an ability the agent lacks (`INVENTED`).
    InventCapability,
    /// Answer a plain request with a question back (`asks_back`) and no answer that passes.
    AskBack,
    /// Say it read or looked at the user's files (`CLAIMED_FILES`).
    ClaimFiles,
}

/// Who said an earlier message of an everyday row's chat.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Speaker {
    /// The user.
    User,
    /// The agent.
    Assistant,
}

/// An earlier message of an everyday row's chat.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Said {
    /// Who said it.
    pub role: Speaker,
    /// What was said.
    pub content: String,
}

/// One row of the everyday set: what an ordinary person asks a chat app, and what a right reply
/// does; an unknown key fails the set.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EverydayQuestion {
    /// `everyday-<category>-<nnn>`.
    pub id: String,
    /// One of `CATEGORIES`.
    pub category: String,
    /// Earlier messages of the chat, before `user`.
    #[serde(default)]
    pub history: Vec<Said>,
    /// The user's message.
    pub user: String,
    /// Where the question belongs: `none` is no tool call at all.
    pub route: Route,
    /// The right call or answer, as a tool row's.
    pub expect: Expect,
    /// What the reply must not do.
    #[serde(default)]
    pub must_not: Vec<MustNot>,
}

/// The everyday set, ready for `run_everyday`.
#[derive(Debug)]
pub struct EverydaySet {
    /// Its file.
    pub file: PromptsFile,
    /// Its rows, in file order.
    pub questions: Vec<EverydayQuestion>,
    /// Runs of every row (`--repeat`), each kept; 1 by default.
    pub repeat: u32,
    /// Where the runs' audit log is copied (`keep_logs`); `None` keeps none.
    pub audit: Option<PathBuf>,
}

/// The everyday set at `path`. A row whose ID is not `everyday-<its category>-`, whose category
/// is not one of `CATEGORIES`, an ID twice, or a `none` row that expects a tool fails here.
pub fn everyday_set(path: &Path) -> anyhow::Result<EverydaySet> {
    let (questions, file) = lines::<EverydayQuestion>(path)?;
    let mut ids = HashSet::new();
    for q in &questions {
        anyhow::ensure!(ids.insert(&q.id), "{} is in the set twice", q.id);
        anyhow::ensure!(
            CATEGORIES.contains(&q.category.as_str()),
            "{}'s category {} is not one of {CATEGORIES:?}",
            q.id,
            q.category
        );
        anyhow::ensure!(
            q.id.starts_with(&format!("everyday-{}-", q.category)),
            "{} is not named everyday-{}-<nnn>",
            q.id,
            q.category
        );
        anyhow::ensure!(
            // `tool_any: [null]` is still no tool: it only lifts the refusal check, for the
            // rows whose right reply says no files are attached.
            q.route != Route::Model
                || (q.expect.tool.is_none() && q.expect.tool_any.iter().all(Option::is_none)),
            "{} is a route none row that expects a tool",
            q.id
        );
    }
    Ok(EverydaySet {
        file,
        questions,
        repeat: 1,
        audit: None,
    })
}

/// One everyday row's run and how it scored.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EverydayRow {
    /// The row's ID.
    pub id: String,
    /// Its category.
    pub category: String,
    /// Which run of the row it is, from 1.
    pub run: u32,
    /// The tools the turn called, in order: its first alone without the web.
    pub tools: Vec<String>,
    /// The reply as it showed.
    pub answer: String,
    /// No miss and no error.
    pub right: bool,
    /// Why it failed, each.
    pub misses: Vec<String>,
    /// The answer was scored: a `none` row, or a run with the web; else only the tool choice was.
    pub answer_scored: bool,
    /// It called a tool where none, or another, was right.
    pub wrong_tool: bool,
    /// It called no tool where one was right.
    pub no_tool: bool,
    /// It refused or deflected (`MustNot::Deflect`).
    pub deflected: bool,
    /// It claimed an ability it lacks (`MustNot::InventCapability`).
    pub invented: bool,
    /// It asked a question back (`MustNot::AskBack`).
    pub asked_back: bool,
    /// It said it read the user's files (`MustNot::ClaimFiles`).
    pub claimed_files: bool,
    /// Its calls to `FILE_READS`, with nothing attached.
    pub file_tool_calls: u32,
    /// The addresses the reply names, in order (`urls`).
    pub sources: Vec<String>,
    /// What the run came to (`outcome`).
    pub outcome: Option<Outcome>,
    /// Where its time went.
    pub stages: Option<Stages>,
    /// The engine error, or the failed turn's notice.
    pub error: Option<String>,
    /// The turn, end to end.
    pub wall_ms: f64,
}

/// Scores a turn of `q` that called `calls` and showed `said`, all in Rust: the tool choice
/// against `expect` (a `none` row calls nothing at all), a call to `FILE_READS` with nothing
/// attached, the answer checks (`score_answer`, without its refusal check) where the answer counts
/// (a `none` row, or `web`: the run had the web), and each of `must_not`.
pub fn score_everyday(
    q: &EverydayQuestion,
    calls: &[ToolCall],
    said: &str,
    web: bool,
) -> EverydayRow {
    let mut row = EverydayRow {
        id: q.id.clone(),
        category: q.category.clone(),
        tools: calls.iter().map(|c| c.name.clone()).collect(),
        answer: said.into(),
        sources: urls(said),
        answer_scored: q.route == Route::Model || web,
        ..EverydayRow::default()
    };
    let mut misses = Vec::new();
    match (q.route, calls.first()) {
        (Route::Model, Some(c)) => {
            row.wrong_tool = true;
            misses.push(format!("called {} where no tool is right", c.name));
        }
        (Route::Model, None) => {}
        (_, None) => {
            row.no_tool = true;
            misses.push("called no tool where one is right".into());
        }
        (_, Some(c)) => {
            let first = Parsed {
                call: Some(c.clone()),
                ..Parsed::default()
            };
            if !score_toolcall(&q.expect, &first).args_ok {
                row.wrong_tool = true;
                misses.push(format!("called {} {}", c.name, c.args));
            }
        }
    }
    row.file_tool_calls = (calls.iter())
        .filter(|c| FILE_READS.contains(&c.name.as_str()))
        .count() as u32;
    if row.file_tool_calls > 0 {
        misses.push(format!(
            "{} file tool calls with nothing attached",
            row.file_tool_calls
        ));
    }
    let answer = Parsed {
        text_before: said.into(),
        ..Parsed::default()
    };
    // Without the refusal check (`tool_any: [null]` is still no tool), which is for docs and
    // routing answers: "I don't have feelings like people do, but" is a right reply here, and a
    // whole refusal is `MustNot::Deflect`'s alone.
    let expect = Expect {
        tool_any: vec![None],
        ..q.expect.clone()
    };
    let missed = score_answer(&expect, &answer);
    let checked =
        !q.expect.answer_must_contain.is_empty() || !q.expect.answer_must_contain_any.is_empty();
    let answered = checked && missed.is_none();
    if row.answer_scored {
        misses.extend(missed);
    }
    let text = said.to_lowercase().replace('\u{2019}', "'");
    let first = sentences(&text).next().unwrap_or_default();
    for &m in &q.must_not {
        let (hit, flag) = match m {
            MustNot::Deflect => (
                DEFLECTIONS.iter().chain(&OFFLINE).any(|d| holds(first, d)),
                &mut row.deflected,
            ),
            MustNot::InventCapability => {
                (INVENTED.iter().any(|s| text.contains(s)), &mut row.invented)
            }
            MustNot::AskBack => (!answered && asks_back(said), &mut row.asked_back),
            MustNot::ClaimFiles => (
                sentences(&text).any(|s| {
                    (CLAIMED_FILES.iter()).any(|c| {
                        s.strip_prefix(c)
                            .is_some_and(|rest| !rest.starts_with(char::is_alphanumeric))
                    })
                }),
                &mut row.claimed_files,
            ),
        };
        if hit {
            *flag = true;
            misses.push(format!("{m:?}"));
        }
    }
    row.right = misses.is_empty();
    row.misses = misses;
    row
}

/// One category's rows over an everyday run, with the v3 counters.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoryCount {
    /// Its rows' runs.
    pub rows: usize,
    /// Of those, the right ones.
    pub passed: usize,
    /// `passed` over `rows`.
    pub pass_rate: f64,
    /// Runs an error ended.
    pub errors: usize,
    /// Deflections (`EverydayRow::deflected`).
    pub deflections: usize,
    /// Invented capabilities.
    pub invented_capabilities: usize,
    /// Questions back.
    pub ask_backs: usize,
    /// Claims to have read the user's files.
    pub claimed_files: usize,
    /// A tool where none, or another, was right.
    pub wrong_tool: usize,
    /// No tool where one was needed.
    pub no_tool: usize,
    /// Calls to `FILE_READS` with nothing attached.
    pub file_tool_calls: u32,
}

/// A `--everyday` run.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Everyday {
    /// The set's file.
    pub questions: PromptsFile,
    /// The run had the web (`--live`): web and research rows ran and their answers were scored.
    pub live: bool,
    /// Each row's run, rows in file order, run after run.
    pub rows: Vec<EverydayRow>,
    /// Each category's counts.
    pub by_category: BTreeMap<String, CategoryCount>,
    /// The rows over their runs, with `--repeat` above 1; `None` otherwise.
    pub repeated: Option<Repeated>,
}

/// `rows` by category.
fn by_category(rows: &[EverydayRow]) -> BTreeMap<String, CategoryCount> {
    let mut out: BTreeMap<String, CategoryCount> = BTreeMap::new();
    for r in rows {
        let c = out.entry(r.category.clone()).or_default();
        c.rows += 1;
        c.passed += usize::from(r.right);
        c.errors += usize::from(r.error.is_some());
        c.deflections += usize::from(r.deflected);
        c.invented_capabilities += usize::from(r.invented);
        c.ask_backs += usize::from(r.asked_back);
        c.claimed_files += usize::from(r.claimed_files);
        c.wrong_tool += usize::from(r.wrong_tool);
        c.no_tool += usize::from(r.no_tool);
        c.file_tool_calls += r.file_tool_calls;
        c.pass_rate = c.passed as f64 / c.rows as f64;
    }
    out
}

/// Runs each row of `set` as a new chat's turn after its history, with the chat's tools and
/// nothing attached (no allowed folder; with `routing.app` the desktop app's options instead, its
/// differences and the user's settings that differ from the flags in `report.app`), in a
/// throwaway data folder whose config.toml is `routing.config`'s with the run's settings
/// (`write_config`). Without `live` (no web) the turn stops at its first call, and only the tool choice
/// of a web or research row is scored; with it the turn runs to its end, a research plan is
/// allowed and a write declined, and the answer is scored. `save` writes the report after each
/// row; `progress` gets a line for each.
#[allow(clippy::too_many_arguments)]
pub fn run_everyday(
    larch: &Larch,
    thinking: Thinking,
    routing: &Routing,
    prefix_cache: bool,
    set: &EverydaySet,
    live: bool,
    report: &mut Report,
    mut save: impl FnMut(&Report) -> anyhow::Result<()>,
    mut progress: impl FnMut(String),
) -> anyhow::Result<()> {
    let data = tempfile::tempdir().context("making a data folder for the everyday rows")?;
    write_config(data.path(), routing, thinking, routing.config.as_deref())?;
    copy_probe(data.path(), routing)?;
    let (opts, mut differs) = chat_options(data.path(), Vec::new(), routing, prefix_cache)?;
    differs.extend(settings_differ(routing, thinking)?);
    if routing.app {
        for line in &differs {
            progress(format!("--app: {line}"));
        }
        report.app = Some(differs);
    }
    report.everyday = Some(Everyday {
        questions: set.file.clone(),
        live,
        ..Everyday::default()
    });
    let log = Audit::new(data.path());
    for run in 1..=set.repeat.max(1) {
        for q in &set.questions {
            let started = Instant::now();
            let before = logged(log.dir())?.len();
            let mut heard = Heard::new();
            let ran = everyday_turn(
                larch,
                &opts,
                routing.xml_body,
                routing.model_sha256.clone(),
                q,
                live,
                routing.flow,
                &mut heard,
            );
            let mut row = score_everyday(q, &heard.calls, &heard.said, live);
            if let Err(e) = ran {
                row.error = Some(format!("{e:#}"));
                row.right = false;
            }
            row.run = run;
            row.wall_ms = ms(started);
            let events = logged(log.dir())?;
            row.stages = Some(stages(
                &events[before.min(events.len())..],
                heard.visible_ms,
            ));
            let first = heard.results.first().or(Some(&heard.said));
            row.outcome = Some(outcome(
                first.map(String::as_str),
                row.error.is_some(),
                row.right,
                false,
            ));
            let line = format!(
                "{} {} {:?} {:.1} s{}",
                row.id,
                if row.right { "pass" } else { "fail" },
                row.tools,
                row.wall_ms / 1e3,
                row.misses
                    .iter()
                    .map(|m| format!("; {m}"))
                    .collect::<String>()
            );
            let errored = row.error.is_some();
            if let Some(e) = &mut report.everyday {
                e.rows.push(row);
                e.by_category = by_category(&e.rows);
            }
            save(report)?;
            if let Some(to) = set.audit.as_deref().filter(|_| errored) {
                keep_logs(data.path(), to)?;
            }
            progress(line);
        }
    }
    if let Some(e) = report.everyday.as_mut().filter(|_| set.repeat > 1) {
        e.repeated = Some(repeated(e.rows.iter().map(|r| (r.id.clone(), r.right))));
        save(report)?;
    }
    set.audit
        .as_deref()
        .map_or(Ok(()), |to| keep_logs(data.path(), to))
}

/// The keys `write_config` sets from the bench's flags, and `allowed`, which it leaves out: a
/// bench chat has attached what its row says.
const BENCH_KEYS: [&str; 5] = [
    "tool_grammar",
    "thinking_budget",
    "answer_max_tokens",
    "router_margin",
    "allowed",
];

/// Writes `data`'s config.toml, which a chat there reads: `from`'s (a data folder's, so the
/// chats search with its keys) without `BENCH_KEYS`, then the run's settings; with
/// `routing.app`, `from`'s whole, as the app reads it (`settings_differ` says where it differs).
/// A key the file holds twice would not parse, so each is written once.
pub fn write_config(
    data: &Path,
    routing: &Routing,
    thinking: Thinking,
    from: Option<&Path>,
) -> anyhow::Result<()> {
    let text = match from.map(|d| fs::read_to_string(crate::config::path(d))) {
        Some(Ok(text)) => text,
        Some(Err(e)) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
        _ => String::new(),
    };
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let mut out = String::new();
    for line in text.lines() {
        let key = line.split_once('=').map(|(k, _)| k.trim());
        if routing.app || !key.is_some_and(|k| BENCH_KEYS.contains(&k)) {
            out += &format!("{line}\n");
        }
    }
    if !(routing.app && from.is_some()) {
        out += &format!(
            "tool_grammar = {}\nthinking_budget = {}\nanswer_max_tokens = {}\nrouter_margin = {}\n",
            routing.tool_grammar, thinking.budget, thinking.answer_max_tokens, routing.margin
        );
    }
    let config = crate::config::path(data);
    fs::create_dir_all(config.parent().unwrap_or(data))?;
    fs::write(&config, out)?;
    Ok(())
}

/// With `routing.app`, each of `routing.config`'s settings the chat reads that differs from the
/// bench's flags, one line each (`tool_grammar`, `thinking_budget`, `answer_max_tokens`); empty
/// otherwise.
pub fn settings_differ(routing: &Routing, thinking: Thinking) -> anyhow::Result<Vec<String>> {
    let Some(dir) = routing.config.as_deref().filter(|_| routing.app) else {
        return Ok(Vec::new());
    };
    let app = crate::config::load(dir)?;
    let mut lines = Vec::new();
    let mut differ = |key: &str, app: String, bench: String| {
        if app != bench {
            lines.push(format!("{key}: {app} (bench: {bench})"));
        }
    };
    differ(
        "tool_grammar",
        app.tool_grammar.to_string(),
        routing.tool_grammar.to_string(),
    );
    differ(
        "thinking_budget",
        app.thinking.budget.to_string(),
        thinking.budget.to_string(),
    );
    differ(
        "answer_max_tokens",
        app.thinking.answer_max_tokens.to_string(),
        thinking.answer_max_tokens.to_string(),
    );
    Ok(lines)
}

/// One everyday row as a new chat's turn, heard into `heard`: up to its first call, or with
/// `whole` to its end, a research plan allowed and a write declined.
#[allow(clippy::too_many_arguments)]
fn everyday_turn(
    larch: &Larch,
    opts: &AgentOptions,
    xml_body: bool,
    model_sha256: Option<String>,
    q: &EverydayQuestion,
    whole: bool,
    flow: bool,
    heard: &mut Heard,
) -> anyhow::Result<()> {
    let mut agent = Agent::new(larch, opts.clone())?;
    agent.set_xml_body(xml_body);
    // As a turn row: config.toml's `router = "probe"` reaches the everyday rows too.
    agent.set_model_sha256(model_sha256);
    let history = (q.history.iter()).map(|m| ChatMessage {
        role: match m.role {
            Speaker::User => Role::User,
            Speaker::Assistant => Role::Assistant,
        },
        content: m.content.clone(),
        ..ChatMessage::default()
    });
    agent.plant(history.collect())?;
    let sent = Instant::now();
    let turn = match flow {
        true => agent.turn(&q.user)?,
        false => agent.turn_with(larch, &q.user)?,
    };
    *heard = Heard::new();
    if flow {
        heard.started = sent;
    }
    let mut parked = listen(turn, whole, heard);
    while let Some(call) = parked {
        let yes = call.name == crate::skills::research::TOOL;
        parked = listen(agent.confirm(yes)?, whole, heard);
    }
    if heard.calls.is_empty() && !heard.ended {
        anyhow::bail!(
            "{}",
            (heard.notice.clone()).unwrap_or_else(|| "the turn ended without an answer".into())
        );
    }
    Ok(())
}

/// The everyday run as the bench prints it: each category's pass rate and counters, the run's
/// file tool calls, then each failing row with why.
pub fn everyday_table(e: &Everyday) -> String {
    let web = match e.live {
        true => "with the web: answers scored",
        false => "without the web: web and research rows score their tool choice alone",
    };
    let mut out = format!(
        "everyday: {} rows, {web}\n{:<12}{:>6}{:>7}{:>8}{:>9}{:>9}{:>8}{:>7}{:>8}{:>8}{:>7}\n",
        e.rows.len(),
        "category",
        "rows",
        "pass",
        "deflect",
        "invented",
        "ask back",
        "claims",
        "wrong",
        "no tool",
        "files",
        "errors"
    );
    for (name, c) in &e.by_category {
        out += &format!(
            "{name:<12}{:>6}{:>7.3}{:>8}{:>9}{:>9}{:>8}{:>7}{:>8}{:>8}{:>7}\n",
            c.rows,
            c.pass_rate,
            c.deflections,
            c.invented_capabilities,
            c.ask_backs,
            c.claimed_files,
            c.wrong_tool,
            c.no_tool,
            c.file_tool_calls,
            c.errors
        );
    }
    let files: u32 = e.by_category.values().map(|c| c.file_tool_calls).sum();
    let passed = e.rows.iter().filter(|r| r.right).count();
    out += &format!(
        "all: {passed}/{} right; file tool calls {files} (must be 0 with nothing attached)\n",
        e.rows.len()
    );
    if let Some(r) = &e.repeated {
        out += &format!(
            "repeat: {} runs, mean right {:.3}, all_pass {:.3} of {} rows\n",
            r.runs,
            r.mean,
            r.all_pass,
            r.rows.len()
        );
    }
    for r in e.rows.iter().filter(|r| !r.right) {
        let why = match &r.error {
            Some(e) => format!("error: {e}"),
            None => r.misses.join("; "),
        };
        let head: String = squash(&r.answer).0.chars().take(120).collect();
        out += &format!("{} run {}: {why} | {head}\n", r.id, r.run);
    }
    out
}

/// The http and https addresses written in `text`, in order, each once, without the brackets and
/// punctuation around them.
fn urls(text: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for word in text.split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | '<' | '>')) {
        let word =
            word.trim_matches(|c: char| matches!(c, '[' | ']' | ',' | '.' | ';' | '"' | '\''));
        if (word.starts_with("http://") || word.starts_with("https://"))
            && !found.iter().any(|f| f == word)
        {
            found.push(word.into());
        }
    }
    found
}

/// One labelled example for continuous learning: a scored bench row as a line of
/// `<data_dir>/agent/evidence.jsonl`.
// ponytail: the plan's fields; #364's store keeps one camelCase file per user-verified answer
// (id, chatId, index, kind, messages, answer, modelSha, epoch, createdMs), a different shape for
// a different source, so these lines are additive until its owner names the bench's fields
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRecord {
    /// When the bench wrote it, UTC.
    pub at: String,
    /// The user's question.
    pub question: String,
    /// The tools the answer came through, in order.
    pub tools: Vec<String>,
    /// The answer as the user read it.
    pub answer: String,
    /// What the answer rests on: page addresses, or `file:line`.
    pub sources: Vec<String>,
    /// `right` or `wrong`.
    pub verdict: String,
    /// Who judged it: `bench`.
    pub source_of_verdict: String,
}

/// Each scored row of `report`'s replay, live, everyday and docs sections as evidence written
/// `at`, its question from `asked` (row ID to question); a row an error ended, one for a human
/// check, an everyday row whose answer was not scored (no web) and one whose ID `asked` lacks are
/// not scored examples and give none, and nor does a `--quick` run, a smoke test.
pub fn evidence(
    report: &Report,
    asked: &BTreeMap<String, String>,
    at: &str,
) -> Vec<EvidenceRecord> {
    let mut out = Vec::new();
    if report.quick {
        return out;
    }
    let mut add = |id: &str, tools: Vec<String>, answer: &str, sources: Vec<String>, right| {
        if let Some(question) = asked.get(id) {
            out.push(EvidenceRecord {
                at: at.into(),
                question: question.clone(),
                tools,
                answer: answer.into(),
                sources,
                verdict: if right { "right" } else { "wrong" }.into(),
                source_of_verdict: "bench".into(),
            });
        }
    };
    let web = || vec![web_answer::TOOL.to_string()];
    for section in [&report.replay, &report.live].into_iter().flatten() {
        for r in section
            .rows
            .iter()
            .filter(|r| r.error.is_none() && !r.check)
        {
            add(&r.id, web(), &r.answer, r.sources.clone(), r.right);
        }
    }
    for r in report.everyday.iter().flat_map(|e| &e.rows) {
        if r.error.is_none() && r.answer_scored {
            add(
                &r.id,
                r.tools.clone(),
                &r.answer,
                r.sources.clone(),
                r.right,
            );
        }
    }
    for q in report.docs.iter().flat_map(|d| &d.questions) {
        let cited = (q.cited.iter())
            .map(|c| format!("{}:{}", c.file.display(), c.line))
            .collect();
        add(
            &q.id,
            vec![ask_docs::TOOL.into()],
            &q.answer,
            cited,
            q.right,
        );
    }
    out
}

/// Appends `records` to `<data_dir>/agent/evidence.jsonl`, one JSON line each.
pub fn append_evidence(data_dir: &Path, records: &[EvidenceRecord]) -> anyhow::Result<()> {
    use std::io::Write;
    let path = data_dir.join("agent").join("evidence.jsonl");
    fs::create_dir_all(path.parent().unwrap_or(data_dir))?;
    let mut text = String::new();
    for r in records {
        text += &serde_json::to_string(r)?;
        text.push('\n');
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    file.write_all(text.as_bytes())
        .with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{Ask, Routing, step_params};
    use crate::{
        CHAT_TOOLS, config::Thinking, parse::Family, router, run::Kind, skills::ask_docs,
        tool::Registry,
    };

    #[test]
    fn the_second_turn_row_gets_the_tool_result_the_product_sends() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/agent/source-routing-holdout.jsonl");
        let (rows, _) = super::load(&path).unwrap();
        let row = rows
            .iter()
            .find(|r| r.id == "holdout-second-turn-002")
            .unwrap();
        let sent = format!("{}\n{}", ask_docs::NOT_FOUND, crate::run::NOT_IN_FILES);
        assert_eq!(row.history.last().unwrap().content, sent);
    }

    #[test]
    fn a_cut_is_in_a_call_only_where_the_text_ends_inside_one() {
        use super::{Cut, cut};
        use crate::parse::{Family, parse};
        let read = r#"{"name":"read_file","arguments":{"path":"notes/todo.txt"}}"#;
        let tag_in_text = r#"<tool_call>{"name":"write_file","arguments":{"path":"notes/tag.txt","text":"</tool_call>"}}"#;
        let open_text = r#"{"name":"write_file","arguments":{"path":"a.txt","text":"a } or ] here"#;
        let cases = [
            // Qwen3: a tag, its JSON, its closing tag; brackets in the text after a call and a
            // closing tag inside a string are not structure.
            (
                Family::Qwen3,
                format!("<tool_call>{read}</tool_call>\nI will explain [the contents"),
                Cut::Text,
            ),
            (Family::Qwen3, tag_in_text.to_owned(), Cut::Call),
            (Family::Qwen3, format!("<tool_call>{read}"), Cut::Call),
            (
                Family::Qwen3,
                format!("<tool_call>{read}</tool_"),
                Cut::Call,
            ),
            (
                Family::Qwen3,
                format!("<tool_call>{}", &read[..40]),
                Cut::Call,
            ),
            (
                Family::Qwen3,
                format!("<tool_call>{read}</tool_call>\n<tool_call>{read}</tool_call>\nBoth read."),
                Cut::Text,
            ),
            (
                Family::Qwen3,
                format!("<tool_call>{read}</tool_call>\n<tool_call>"),
                Cut::Call,
            ),
            (
                Family::Qwen3,
                format!("Let me check.\n<tool_call>{read}</tool_call> then {{ more"),
                Cut::Text,
            ),
            // Llama 3: bare JSON, `;` between calls.
            (
                Family::Llama3,
                format!("{read}; {read} and [the rest"),
                Cut::Text,
            ),
            (
                Family::Llama3,
                format!("{read}; {}", &read[..40]),
                Cut::Call,
            ),
            (Family::Llama3, open_text.to_owned(), Cut::Call),
        ];
        for (family, text, want) in cases {
            let parsed = parse(family, &text, false);
            assert!(
                parsed.call.is_some() || parsed.error.is_some(),
                "{text}: {parsed:?}"
            );
            assert_eq!(cut(&text, &parsed), want, "{text}: {parsed:?}");
        }
    }

    #[test]
    fn every_request_kind_is_greedy_and_seeded() {
        let registry = Registry::builtin();
        // Tool rows and multi-step steps show the chat's tools, memory tier 2 none. The chat
        // steps and the docs answer think on the budget, retries and the docs workers never;
        // Llama 3 gets no thinking field.
        let (qwen3, llama3) = (Some(Family::Qwen3), Some(Family::Llama3));
        let on = Thinking::default();
        let step =
            |allow: &[&str], family, kind| step_params(&registry, allow, family, kind, on, 1);
        let routing = Routing::default();
        let ask = Ask {
            registry: &registry,
            allow: CHAT_TOOLS.to_vec(),
            family: Family::Qwen3,
            thinking: on,
            namespace: 1,
            routing: &routing,
            starts: &[],
        };
        // The router's steps: the gate's one token thinks not and asks for the top two; the step
        // it decided thinks not and asks for no logprobs, so no token after the first carries
        // any; the step it sends again thinks as a chat step does; a masked one shows no tools.
        let decided = ask.params(Kind::Chat, true, false);
        let gated = crate::run::gate_params(&decided);
        assert_eq!(
            (gated.max_tokens, gated.logprobs, gated.tools.len()),
            (1, router::TOP, CHAT_TOOLS.len())
        );
        assert_eq!(decided.logprobs, 0);
        assert_eq!(ask.params(Kind::Chat, false, false).logprobs, 0);
        let masked = ask.params(Kind::Chat, true, true);
        assert!(masked.tools.is_empty());
        let kinds = [
            ("router: the gate's one token", gated, (Some(false), None)),
            ("router: the step it decided", decided, (Some(false), None)),
            (
                "router: the step with thinking",
                ask.params(Kind::Chat, false, false),
                (Some(true), Some(100)),
            ),
            ("router: a masked step", masked, (Some(false), None)),
            (
                "tool row, multi-step first step",
                step(&CHAT_TOOLS, qwen3, Kind::Chat),
                (Some(true), Some(100)),
            ),
            (
                "multi-step step after a result",
                step(&CHAT_TOOLS, qwen3, Kind::Answer),
                (Some(true), Some(100)),
            ),
            (
                "tool row and multi-step retry",
                step(&CHAT_TOOLS, qwen3, Kind::Retry),
                (Some(false), None),
            ),
            (
                "memory tier 2",
                step(&[], qwen3, Kind::Chat),
                (Some(true), Some(100)),
            ),
            (
                "Llama 3 tool row",
                step(&CHAT_TOOLS, llama3, Kind::Chat),
                (None, None),
            ),
            (
                "docs worker and its retry",
                ask_docs::params(qwen3, 1),
                (Some(false), None),
            ),
            (
                "Llama 3 docs worker",
                ask_docs::params(llama3, 1),
                (None, None),
            ),
            // The orchestrator thinks on the chat's budget; Rust, not the think block, abstains.
            (
                "docs orchestrator",
                ask_docs::answer_params(qwen3, Thinking::default(), 1),
                (Some(true), Some(100)),
            ),
            (
                "docs orchestrator, thinking off",
                ask_docs::answer_params(
                    qwen3,
                    Thinking {
                        budget: 0,
                        ..Thinking::default()
                    },
                    1,
                ),
                (Some(false), None),
            ),
            // The cap after the think block is `answer_max_tokens` when it sets one.
            (
                "docs orchestrator, capped",
                ask_docs::answer_params(
                    qwen3,
                    Thinking {
                        answer_max_tokens: 200,
                        ..Thinking::default()
                    },
                    1,
                ),
                (Some(true), Some(100)),
            ),
            (
                "Llama 3 docs orchestrator",
                ask_docs::answer_params(llama3, Thinking::default(), 1),
                (None, None),
            ),
            // An anchor renders as the step it warms (the same `enable_thinking`), and one token
            // ends it, with no budget to continue after.
            (
                "anchor",
                crate::run::anchor_params(&step(&CHAT_TOOLS, qwen3, Kind::Chat)),
                (Some(true), None),
            ),
            (
                "worker anchor",
                crate::run::anchor_params(&ask_docs::params(qwen3, 1)),
                (Some(false), None),
            ),
            (
                "memory keys",
                crate::memory::key_params(qwen3, 1),
                (Some(false), None),
            ),
            (
                "Llama 3 memory keys",
                crate::memory::key_params(llama3, 1),
                (None, None),
            ),
        ];
        let max = |t| ask_docs::answer_params(qwen3, t, 1).max_tokens;
        let cap = |answer_max_tokens| Thinking {
            answer_max_tokens,
            ..Thinking::default()
        };
        assert_eq!((max(cap(0)), max(cap(200))), (1124, 300));
        for (kind, p, thinking) in kinds {
            assert_eq!((p.temperature, p.seed), (Some(0.0), Some(0)), "{kind}");
            assert_eq!((p.thinking, p.thinking_budget), thinking, "{kind}");
            // In the data folder's namespace, so two folders never share the prefix cache.
            assert_eq!(p.cache_namespace, 1, "{kind}");
        }
    }
}
