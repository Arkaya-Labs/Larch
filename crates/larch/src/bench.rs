//! `larch bench` (architecture 16): closed-loop replay of the frozen workloads on larch or over
//! HTTP (Ollama, larch-http, an OpenAI-compatible server), the fixed-shape mirror of
//! `llama-batched-bench`, and the `larch-bench/1` result that `scripts/baseline.py check|compare`
//! reads (bench/README.md).

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpStream,
    path::{Path, PathBuf},
    sync::{
        OnceLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use larch_engine::{EngineEvent, EngineHandle, EngineOptions, OnFault, Request, TelemetryTier};
use larch_model::{ChatTemplate, Message, Tokenizer, gguf::Gguf};
use larch_records::{
    Authority, Cause, EngineSample, KnobReply, KnobRequest, Phase, ReadFrom, Record, knob_id,
    knob_spec,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{GenParams, Larch, Overflow, SessionOptions, TextEvent, knob};

/// The render date of every engine (bench/README.md), so prompts and their hashes match llama.cpp's.
pub const DATE: &str = "26 Jul 2024";
/// The `knob_timeline` reply of a `--set k=v@S` whose run ended before S.
pub const NOT_REACHED: &str = "NotReached";
/// Engine session ids of scripted conversations, from far above the facade's counter so the two
/// never meet. Never reused in a process: a released id never parks again (10.1.1).
static SCRIPTED_SESSIONS: AtomicU64 = AtomicU64::new(1 << 62);

/// `larch bench` options (architecture 10.2).
#[derive(Args, Clone, Debug)]
pub struct Options {
    /// GGUF under test; its tokenizer and template render every prompt, for the HTTP engines too.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// Engine under test.
    #[arg(long, value_enum, default_value_t = EngineKind::Larch)]
    pub engine: EngineKind,
    /// HTTP engine's base URL, plain HTTP, e.g. http://127.0.0.1:11434.
    #[arg(long)]
    pub url: Option<String>,
    /// OpenAI-compatible server's prefix cache, where its `/metrics` does not say (openai only).
    #[arg(long)]
    pub cache_prompt: Option<bool>,
    /// Workload (architecture 16).
    #[arg(long, value_enum, conflicts_with = "mode")]
    pub workload: Option<Workload>,
    /// `serve` is the batch workload (W4), `batched` the fixed shape (W6).
    #[arg(long, value_enum)]
    pub mode: Option<Mode>,
    /// Conversations continue with the scripted answers (comparable) or larch's own (larch only).
    #[arg(long, value_enum, default_value_t = Replay::Scripted)]
    pub replay: Replay,
    /// Closed-loop clients; several values run one after another, each into its own result (ollama
    /// and larch-http take one: their server's parallelism is set per N; openai takes one when its
    /// server caches prefixes, which it keeps between runs).
    #[arg(long, value_delimiter = ',', default_value = "1")]
    pub concurrency: Vec<usize>,
    /// Workload file; default the workload's own under bench/prompts.
    #[arg(long)]
    pub prompts: Option<PathBuf>,
    /// Fixed shape: prompt tokens per sequence.
    #[arg(long, default_value_t = 512)]
    pub pp: u32,
    /// Fixed shape: generated tokens fed per sequence.
    #[arg(long, default_value_t = 128)]
    pub tg: u32,
    /// Fixed shape: sequences in flight.
    #[arg(long, value_delimiter = ',', default_value = "1,4,8,16")]
    pub npl: Vec<usize>,
    /// Seconds: clients wrap around the workload until then (R2, R12); unset runs it once.
    #[arg(long)]
    pub duration: Option<f64>,
    /// `K/N`: only the K-th of N contiguous parts of W4 or W5, in file order, so a long run fits a bounded
    /// GPU hold; `scripts/bench-join` merges the parts into the whole row.
    #[arg(long, value_parser = slice_arg)]
    pub slice: Option<(usize, usize)>,
    /// Fraction of requests cancelled after their first token (R3).
    #[arg(long, default_value_t = 0.0)]
    pub cancel_rate: f64,
    /// Let end-of-generation tokens terminate answers (historical default suppresses them).
    #[arg(long)]
    pub natural_eos: bool,
    /// Engine telemetry tier; the fixed shape checks every step, so it takes t1.
    #[arg(long, value_enum, default_value_t = Telemetry::T1)]
    pub telemetry: Telemetry,
    /// `knob=value` at startup, or `knob=value@seconds` live into each run; repeatable.
    #[arg(long = "set")]
    pub set: Vec<String>,
    /// Run on the CPU reference.
    #[arg(long, conflicts_with = "device")]
    pub cpu: bool,
    /// GPU ordinal (default zero).
    #[arg(long)]
    pub device: Option<i32>,
    /// Machine name recorded in the result (win5070, wsl5070, win4060, sm86, mac, cpu).
    #[arg(long, default_value = "local")]
    pub machine: String,
    /// Result file; with several concurrencies, `<stem>-c<N>.json` each.
    #[arg(long)]
    pub json: Option<PathBuf>,
}

/// The engine a run measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EngineKind {
    /// In process, through the facade.
    Larch,
    /// Over its HTTP API (D15).
    Ollama,
    /// Benchmark-only Larch adapter over the same localhost streaming HTTP boundary.
    LarchHttp,
    /// An OpenAI-compatible server's streamed `/v1/completions` (vLLM, llama-server, larch serve).
    #[value(name = "openai")]
    OpenAi,
}

/// The workloads of architecture 16.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Workload {
    /// W1: one 12-turn conversation.
    Chat,
    /// W2: concurrent conversations.
    Chats,
    /// W3: agent loops on a 4096-token prefix.
    Agent,
    /// W4: 200 independent prompts.
    Batch,
    /// W5: 2048- and 4096-token prompts.
    Longprompt,
    /// W6: the llama-batched-bench shape.
    Fixed,
}

impl Workload {
    fn file(self) -> &'static str {
        match self {
            Self::Chat => "w1-chat.jsonl",
            Self::Chats => "w2-chats.jsonl",
            Self::Agent => "w3-agent.jsonl",
            Self::Batch => "chat.jsonl",
            Self::Longprompt => "w5-longprompt.jsonl",
            Self::Fixed => "",
        }
    }

    /// Prefix reuse happens only in W1 to W3, on every engine.
    fn reuse(self) -> bool {
        matches!(self, Self::Chat | Self::Chats | Self::Agent)
    }
}

/// `--mode`, the serve-mode spelling of W4 and W6.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// W4.
    Serve,
    /// W6.
    Batched,
}

/// How a conversation continues after a timed answer (D14).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Replay {
    /// With the scripted answer: byte-identical prompts on every engine.
    Scripted,
    /// With larch's own answer, through `Session`.
    Generated,
}

/// `--telemetry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Telemetry {
    /// No engine samples.
    Off,
    /// Decimated samples.
    T0,
    /// Every step.
    T1,
}

/// A workload file: conversations whose assistant messages are the timed turns.
pub struct WorkloadFile {
    /// File name, as the driver records it.
    pub name: String,
    /// sha256 of the file bytes.
    pub sha256: String,
    /// Conversations in file order.
    pub convs: Vec<Conv>,
}

/// One conversation of a workload file.
pub struct Conv {
    /// Its `id`.
    pub id: String,
    /// Every message; a timed turn's own content is its scripted answer.
    pub messages: Vec<Message>,
    /// The timed turns, in order.
    pub turns: Vec<Turn>,
}

/// A timed turn: its prompt is `messages[..n_msgs]`, its answer exactly `max_tokens` long.
pub struct Turn {
    /// Messages before this assistant message.
    pub n_msgs: usize,
    /// Answer length.
    pub max_tokens: u32,
}

#[derive(Deserialize)]
struct Line {
    id: String,
    messages: Vec<LineMessage>,
}

#[derive(Deserialize)]
struct LineMessage {
    role: String,
    content: Option<String>,
    max_tokens: Option<u32>,
}

/// Reads a workload with the llama.cpp driver's rules, so both engines refuse the same files.
pub fn load_workload(path: &Path) -> Result<WorkloadFile> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = std::str::from_utf8(&bytes).with_context(|| path.display().to_string())?;
    let mut convs = Vec::new();
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.is_empty()) {
        let at = format!("{}:{}", path.display(), i + 1);
        let line: Line = serde_json::from_str(line).with_context(|| at.clone())?;
        let n = line.messages.len();
        let mut c = Conv {
            id: line.id,
            messages: Vec::with_capacity(n),
            turns: Vec::new(),
        };
        for (k, m) in line.messages.into_iter().enumerate() {
            let last = k + 1 == n;
            match m.role.as_str() {
                "assistant" => {
                    let after_user = c.messages.last().is_some_and(|p| p.role == "user");
                    let Some(max_tokens) = m.max_tokens.filter(|&t| t >= 1 && after_user) else {
                        bail!(
                            "{at}: an assistant turn needs max_tokens >= 1 and a user message before it"
                        );
                    };
                    ensure!(
                        last || m.content.is_some(),
                        "{at}: an assistant message the conversation continues after needs its scripted content"
                    );
                    c.turns.push(Turn {
                        n_msgs: k,
                        max_tokens,
                    });
                }
                "system" | "user" => ensure!(
                    !last,
                    "{at}: a conversation ends with the assistant turn that answers it"
                ),
                _ => bail!("{at}: role must be system, user or assistant"),
            }
            c.messages.push(Message {
                role: m.role,
                content: m.content.unwrap_or_default(),
            });
        }
        ensure!(
            !c.turns.is_empty(),
            "{at}: a conversation needs at least one assistant turn"
        );
        convs.push(c);
    }
    ensure!(!convs.is_empty(), "{}: no conversations", path.display());
    Ok(WorkloadFile {
        name: file_name(path),
        sha256: hex(&sha256(&bytes)),
        convs,
    })
}

/// The GGUF's tokenizer and template, the template rendering with the benchmark date.
pub fn vocab_of(g: &Gguf) -> Result<(Tokenizer, ChatTemplate)> {
    let tok = Tokenizer::from_gguf(g)?;
    let mut tmpl = ChatTemplate::from_gguf(g, &tok)?;
    tmpl.date = DATE.into();
    Ok((tok, tmpl))
}

/// `messages` rendered with the generation prompt, and that text tokenised whole, as llama-server's
/// chat path.
pub fn render(
    tok: &Tokenizer,
    tmpl: &ChatTemplate,
    messages: &[Message],
) -> Result<(String, Vec<u32>)> {
    let text = tmpl.render(messages, true).context("chat template")?;
    let tokens = tok.encode(&text, true);
    Ok((text, tokens))
}

/// `prompt_sha256`: sha256 of the token IDs as consecutive little-endian int32 (bench/README.md).
pub fn prompt_sha256(tokens: &[u32]) -> String {
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    hex(&sha256(&bytes))
}

/// The running binary's sha256, so `bench-join` can refuse parts of one row from two builds; null
/// when the binary cannot be read.
fn build() -> Value {
    static ID: OnceLock<Option<String>> = OnceLock::new();
    json!(ID.get_or_init(|| {
        let exe = std::env::current_exe().and_then(fs::read).ok()?;
        Some(hex(&sha256(&exe)))
    }))
}

/// SHA-256 (FIPS 180-4): the crate has no hashing dependency, and the prompt hash needs no more.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&(data.len() as u64 * 8).to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (wi, c) in w.iter_mut().zip(block.chunks_exact(4)) {
            *wi = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for (k, wi) in K.iter().zip(w) {
            let [a, b, c, d, e, f, g, hh] = v;
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(*k)
                .wrapping_add(wi);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let t2 = s0.wrapping_add((a & b) ^ (a & c) ^ (b & c));
            v = [t1.wrapping_add(t2), a, b, c, d.wrapping_add(t1), e, f, g];
        }
        for (x, y) in h.iter_mut().zip(v) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 32];
    for (o, x) in out.chunks_exact_mut(4).zip(h) {
        o.copy_from_slice(&x.to_be_bytes());
    }
    out
}

/// `bytes` as lower-case hex, two digits each; the agent's audit log hashes arguments with it.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// One timed turn in the driver's request schema; [`Req::finish`] derives the latencies.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Req {
    /// Closed-loop client.
    pub client: usize,
    /// Conversation id.
    pub conv: String,
    /// Turn index within the conversation.
    pub turn: usize,
    /// Benchmark adapter releases this conversation after its final scripted turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_end: Option<bool>,
    /// Sequence slot; one per client.
    pub slot: usize,
    /// Prompt length.
    pub prompt_tokens: u32,
    /// Prompt tokens reused from the conversation's cached prefix; null when the engine does not say.
    pub cached_tokens: Option<u32>,
    /// Prompt tokens evaluated; null on error.
    pub evaluated_tokens: Option<u32>,
    /// [`prompt_sha256`] of the prompt.
    pub prompt_sha256: String,
    /// Target answer length.
    pub max_tokens: u32,
    /// Tokens generated.
    pub gen_tokens: u32,
    /// HTTP output counts are unknown until the terminal usage arrives; partial chunks are separate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_tokens_known: Option<bool>,
    /// Server-reported model-load time, outside steady serving after independent warmup.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_load_ms: Option<f64>,
    /// Submission, from the run's start.
    pub t_submit_ms: f64,
    /// First token received.
    pub t_first_ms: Option<f64>,
    /// Terminal event received.
    pub t_done_ms: f64,
    /// First - submit; null on error.
    pub ttft_ms: Option<f64>,
    /// (done - first) / (generated - 1); null on error or a one-token answer.
    pub tpot_ms: Option<f64>,
    /// Individual token receive gaps for the raw in-process engine stream; absent on text streams.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_gaps_ms: Option<Vec<f64>>,
    /// Done - submit; null on error.
    pub e2e_ms: Option<f64>,
    /// The engine's error, `Cancelled` for a cancellation.
    pub error: Option<String>,
    /// Generated replay: the session's parked KV before the turn, which `cached_tokens` must equal.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_tokens: Option<u32>,
    /// Ollama: client e2e minus the server's own `total_duration`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_overhead_ms: Option<f64>,
    /// Actual terminal reason reported by the engine.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Receive gaps between nonempty HTTP text chunks; these are not individual token gaps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_chunk_gaps_ms: Option<Vec<f64>>,
    /// Nonempty HTTP chunks received, including partial work on cancelled answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_chunks: Option<u32>,
    /// Cancelled on purpose (`--cancel-rate`), so its `Cancelled` is not a failure.
    #[serde(skip)]
    pub cancel: bool,
}

impl Req {
    /// Fills the derived fields as the driver does: latencies only for a clean request.
    pub fn finish(mut self) -> Req {
        let ok = self.error.is_none();
        self.evaluated_tokens = self
            .cached_tokens
            .filter(|_| ok)
            .map(|c| self.prompt_tokens.saturating_sub(c));
        self.ttft_ms = self.t_first_ms.filter(|_| ok).map(|f| f - self.t_submit_ms);
        self.tpot_ms = self
            .t_first_ms
            .filter(|_| ok && self.gen_tokens > 1)
            .map(|f| (self.t_done_ms - f) / f64::from(self.gen_tokens - 1));
        self.e2e_ms = ok.then_some(self.t_done_ms - self.t_submit_ms);
        self
    }

    fn cancelled(&self) -> bool {
        self.cancel && self.error.as_deref() == Some("Cancelled")
    }
}

/// The driver's summary over `reqs`, plus `cancelled` and, for Ollama, `http_overhead_ms`.
pub fn summary(reqs: &[Req]) -> Value {
    let ok: Vec<&Req> = reqs.iter().filter(|r| r.error.is_none()).collect();
    let first = reqs
        .iter()
        .map(|r| r.t_submit_ms)
        .fold(f64::INFINITY, f64::min);
    let last = reqs.iter().map(|r| r.t_done_ms).fold(0.0, f64::max);
    let wall_s = if reqs.is_empty() {
        0.0
    } else {
        (last - first) / 1000.0
    };
    let sum = |f: fn(&Req) -> u32| ok.iter().map(|r| u64::from(f(r))).sum::<u64>();
    let (prompt, generated) = (sum(|r| r.prompt_tokens), sum(|r| r.gen_tokens));
    // Unknown for the run when unknown for any of its requests.
    let cached: Option<u64> = ok.iter().map(|r| r.cached_tokens.map(u64::from)).sum();
    let of = |f: fn(&Req) -> Option<f64>| stats(ok.iter().filter_map(|r| f(r)).collect());
    let cancelled = reqs.iter().filter(|r| r.cancelled()).count();
    let mut s = json!({
        "requests": reqs.len(),
        "failures": reqs.len() - ok.len() - cancelled,
        "cancelled": cancelled,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "evaluated_tokens": cached.map(|c| prompt - c),
        "gen_tokens": generated,
        "wall_s": wall_s,
        "agg_tok_s": if wall_s > 0.0 { generated as f64 / wall_s } else { 0.0 },
        "ttft_ms": of(|r| r.ttft_ms),
        "tpot_ms": of(|r| r.tpot_ms),
        "e2e_ms": of(|r| r.e2e_ms),
    });
    if !ok.is_empty() && ok.iter().all(|r| r.token_gaps_ms.is_some()) {
        s["token_gap_ms"] = stats(
            ok.iter()
                .flat_map(|r| r.token_gaps_ms.as_ref().unwrap().iter().copied())
                .collect(),
        );
    }
    if !ok.is_empty() && ok.iter().all(|r| r.http_chunk_gaps_ms.is_some()) {
        s["http_chunk_gap_ms"] = stats(
            ok.iter()
                .flat_map(|r| r.http_chunk_gaps_ms.as_ref().unwrap().iter().copied())
                .collect(),
        );
    }
    if reqs.iter().any(|r| r.http_overhead_ms.is_some()) {
        s["http_overhead_ms"] = of(|r| r.http_overhead_ms);
    }
    s
}

/// p50, p99, mean and max; percentiles are nearest-rank, every engine's definition.
fn stats(mut v: Vec<f64>) -> Value {
    if v.is_empty() {
        return Value::Null;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    let pct = |q: f64| v[((q * n as f64).ceil() as usize).clamp(1, n) - 1];
    json!({"p50": pct(0.50), "p99": pct(0.99), "mean": v.iter().sum::<f64>() / n as f64, "max": v[n - 1]})
}

/// One `--set`: a startup value, or with `@seconds` a live change that long into each run.
struct KnobSet {
    at_s: Option<f64>,
    at: Duration,
    knob: u16,
    value: f64,
}

/// Every `--set` by registered name, startup values first, then live changes in time order.
fn sets(opts: &Options) -> Result<Vec<KnobSet>> {
    let mut out = Vec::new();
    for raw in &opts.set {
        let (assignment, at_s) = match raw.rsplit_once('@') {
            Some((assignment, at)) => {
                let at_s: f64 = at
                    .parse()
                    .with_context(|| format!("--set {raw}: seconds"))?;
                (assignment, Some(at_s))
            }
            None => (raw.as_str(), None),
        };
        let at = secs(at_s.unwrap_or_default()).with_context(|| format!("--set {raw}"))?;
        let (knob, value) = knob::assignment(assignment)?;
        out.push(KnobSet {
            at_s,
            at,
            knob,
            value,
        });
    }
    out.sort_by(|a, b| a.at_s.unwrap_or(-1.0).total_cmp(&b.at_s.unwrap_or(-1.0)));
    Ok(out)
}

/// `s` seconds as an offset any instant of this process can take: finite, nonnegative, in range.
fn secs(s: f64) -> Result<Duration> {
    let d = Duration::try_from_secs_f64(s).with_context(|| format!("{s} seconds"))?;
    ensure!(
        Instant::now().checked_add(d).is_some(),
        "{s} seconds: out of range"
    );
    Ok(d)
}

fn startup(opts: &Options) -> Result<Vec<(u16, f64)>> {
    Ok(sets(opts)?
        .into_iter()
        .filter(|s| s.at_s.is_none())
        .map(|s| (s.knob, s.value))
        .collect())
}

fn knob_name(id: u16) -> String {
    knob_spec(id).map_or_else(|| id.to_string(), |k| k.name.as_str().into())
}

fn slice_arg(s: &str) -> Result<(usize, usize), String> {
    let (k, n) = s.split_once('/').ok_or("K/N")?;
    Ok((
        k.parse().map_err(|e| format!("K: {e}"))?,
        n.parse().map_err(|e| format!("N: {e}"))?,
    ))
}

fn resolve(opts: &Options) -> Result<Workload> {
    let w = match (opts.workload, opts.mode) {
        (Some(w), None) => w,
        (None, Some(Mode::Serve)) => Workload::Batch,
        (None, Some(Mode::Batched)) => Workload::Fixed,
        _ => bail!("give one of --workload or --mode"),
    };
    Ok(w)
}

/// Per-sequence context: `--set max_context`, else 16384 for W3 and 8192 otherwise (architecture 16).
fn n_ctx(opts: &Options, w: Workload) -> Result<u32> {
    Ok(startup(opts)?
        .iter()
        .rfind(|(k, _)| *k == knob_id::MAX_CONTEXT)
        .map_or(if w == Workload::Agent { 16384 } else { 8192 }, |(_, v)| {
            *v as u32
        }))
}

/// The engine's load options for `opts`: device, context, startup knob values and telemetry tier.
pub fn engine_options(opts: &Options) -> Result<EngineOptions> {
    let device = if opts.cpu {
        larch_kernels::CPU
    } else {
        opts.device.unwrap_or(0)
    };
    let mut e = EngineOptions::new(opts.model.clone(), device);
    e.max_context = n_ctx(opts, resolve(opts)?)?;
    e.knob_overrides = startup(opts)?;
    e.telemetry = match opts.telemetry {
        Telemetry::Off => TelemetryTier::Off,
        Telemetry::T0 => TelemetryTier::T0,
        Telemetry::T1 => TelemetryTier::T1,
    };
    Ok(e)
}

/// Every check that needs no model, so a bad invocation fails before the load: the options, and
/// the workload file (none for the fixed shape) read and validated.
fn check(opts: &Options) -> Result<(Workload, Option<WorkloadFile>)> {
    let workload = resolve(opts)?;
    ensure!(
        (0.0..=1.0).contains(&opts.cancel_rate),
        "--cancel-rate is a fraction"
    );
    ensure!(
        !opts.concurrency.is_empty() && !opts.concurrency.contains(&0),
        "--concurrency takes values >= 1"
    );
    if let Some(d) = opts.duration {
        ensure!(d > 0.0, "--duration {d}: seconds > 0");
        secs(d).context("--duration")?;
    }
    if let Some((k, n)) = opts.slice {
        ensure!(k < n, "--slice {k}/{n}: K < N");
        // A conversation's turns never straddle two runs, and each part runs once.
        ensure!(
            matches!(workload, Workload::Batch | Workload::Longprompt) && opts.duration.is_none(),
            "--slice: W4 or W5 only, run once (no --duration)"
        );
    }
    ensure!(
        opts.cache_prompt.is_none() || opts.engine == EngineKind::OpenAi,
        "--cache-prompt states an OpenAI-compatible server's prefix cache: --engine openai only"
    );
    ensure!(
        opts.engine != EngineKind::OpenAi || opts.set.is_empty(),
        "--engine openai takes no --set: a server's knobs are its startup flags"
    );
    let sets = sets(opts)?;
    let live = sets.iter().any(|s| s.at_s.is_some());
    // A live change is never undone, so a second run would start from it, not from the startup values.
    ensure!(
        !live || opts.concurrency.len() == 1,
        "--set k=v@S takes one --concurrency value per invocation"
    );
    // Each part would replay the change from its own start, so a joined row would not be one run's.
    ensure!(
        !live || opts.slice.is_none(),
        "--slice takes startup --set values only, no --set k=v@S"
    );
    if workload == Workload::Fixed {
        ensure!(
            !opts.natural_eos,
            "--natural-eos applies to serving workloads, not fixed shapes"
        );
        ensure!(
            !live,
            "the fixed-shape mirror takes startup values only, no --set k=v@S"
        );
        ensure!(
            opts.pp > 0 && opts.tg > 0 && !opts.npl.is_empty() && !opts.npl.contains(&0),
            "--pp, --tg and --npl take values >= 1"
        );
        ensure!(
            opts.telemetry == Telemetry::T1
                && sets
                    .iter()
                    .all(|s| s.knob != knob_id::TELEMETRY_TIER || s.value == 1.0),
            "the fixed-shape mirror checks every engine step, so it needs --telemetry t1"
        );
    }
    if opts.engine == EngineKind::LarchHttp {
        ensure!(
            opts.duration.is_none(),
            "larch-http adapter runs each scripted conversation once; --duration is unsupported"
        );
        ensure!(
            opts.natural_eos,
            "--engine larch-http requires --natural-eos"
        );
    }
    if opts.engine != EngineKind::Larch {
        ensure!(
            opts.replay == Replay::Scripted,
            "--replay generated validates larch's sessions; it runs on larch only"
        );
        ensure!(
            workload != Workload::Fixed,
            "the fixed-shape mirror runs on larch only (architecture 16)"
        );
        ensure!(
            sets.iter()
                .all(|s| s.knob == knob_id::MAX_CONTEXT && s.at_s.is_none()),
            "ollama and larch-http take only --set max_context (their num_ctx); their other settings are the server's environment"
        );
        // Architecture 16 configures Ollama's parallelism per N: one server, one N. An
        // OpenAI-compatible server serves any N.
        ensure!(
            opts.engine == EngineKind::OpenAi || opts.concurrency.len() == 1,
            "--engine ollama and larch-http take one --concurrency value per invocation, against a server started for that N"
        );
    }
    if workload == Workload::Fixed {
        return Ok((workload, None));
    }
    let path = opts.prompts.clone().unwrap_or_else(|| {
        // The frozen files of the checkout this binary was built from, wherever it runs.
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/prompts")
            .join(workload.file())
    });
    let file = load_workload(&path)?;
    if let Some((_, n)) = opts.slice {
        // More parts than conversations: a part would run nothing.
        ensure!(
            n <= file.convs.len(),
            "--slice K/{n}: N <= the file's {} conversations",
            file.convs.len()
        );
    }
    if opts.replay == Replay::Generated {
        for c in &file.convs {
            let first = usize::from(c.messages[0].role == "system");
            ensure!(
                c.messages[first..]
                    .iter()
                    .enumerate()
                    .all(|(i, m)| m.role == ["user", "assistant"][i % 2]),
                "{}: --replay generated needs [system] then user/assistant pairs",
                c.id
            );
        }
    }
    Ok((workload, Some(file)))
}

/// `larch bench`: loads the model (larch) or only its vocabulary (an HTTP engine), then runs `opts`.
pub fn run(opts: Options, out: &mut impl Write) -> Result<()> {
    check(&opts)?;
    let file = File::open(&opts.model).with_context(|| opts.model.display().to_string())?;
    let len = file.metadata()?.len();
    let g = Gguf::read(BufReader::new(file), len)?;
    let engine = match opts.engine {
        EngineKind::Larch => {
            let mut e = engine_options(&opts)?;
            e.on_fault = OnFault::Exit; // `larch supervise` restarts it (R12)
            Some(larch_engine::start(e)?)
        }
        EngineKind::Ollama | EngineKind::LarchHttp | EngineKind::OpenAi => None,
    };
    run_with(&opts, &g, engine, out)
}

/// Runs `opts` against `engine` (larch) or the HTTP server at `--url`, rendering with `vocab`'s
/// tokenizer and template. Writes each result, prints one line per run, and fails if any
/// request failed. The engine is shut down on return.
pub fn run_with(
    opts: &Options,
    vocab: &Gguf,
    engine: Option<EngineHandle>,
    out: &mut impl Write,
) -> Result<()> {
    let (workload, file) = check(opts)?;
    build(); // hashed once here, before any clock starts
    let live: Vec<KnobSet> = sets(opts)?
        .into_iter()
        .filter(|s| s.at_s.is_some())
        .collect();
    let (tok, tmpl) = vocab_of(vocab)?;
    let n_ctx = n_ctx(opts, workload)?;
    let target = match opts.engine {
        EngineKind::Ollama | EngineKind::LarchHttp => {
            let alias = opts.model.file_stem().unwrap_or_default().to_string_lossy();
            let mut o = Ollama {
                host: host(opts)?,
                // `scripts/baseline.py ollama` creates the 8192 and 16384 models; num_ctx sets the row's.
                model: format!(
                    "larch-{alias}-{}",
                    if workload == Workload::Agent {
                        16384
                    } else {
                        8192
                    }
                ),
                n_ctx,
                larch: opts.engine == EngineKind::LarchHttp,
                warmup: Value::Null,
            };
            o.warmup = ollama_warm_up(&o)?;
            Target::Ollama(o)
        }
        EngineKind::OpenAi => {
            let host = host(opts)?;
            let r = http_ok(&host, "GET /v1/models HTTP/1.1", "").context("GET /v1/models")?;
            let models: Value = serde_json::from_reader(r).context("GET /v1/models")?;
            let m = &models["data"][0];
            let model = m["id"].as_str().context("GET /v1/models lists no model")?;
            // Informational only: a probe that does not answer 200 with JSON does not say.
            let probe = |path: &str| -> Result<Value> {
                Ok(get(&host, path)?
                    .and_then(|b| serde_json::from_str(&b).ok())
                    .unwrap_or_default())
            };
            let mut version = probe("/version")?["version"].clone();
            if version.is_null() {
                version = probe("/props")?["build_info"].clone();
            }
            // vLLM's `vllm:cache_config_info` gauge carries its prefix-cache setting as a label.
            let said = get(&host, "/metrics")?.and_then(|m| {
                let line = m
                    .lines()
                    .find(|l| l.starts_with("vllm:cache_config_info{"))?;
                let label = line
                    .split(['{', ','])
                    .find_map(|l| l.strip_prefix("enable_prefix_caching=\""))?;
                label.split('"').next()?.to_ascii_lowercase().parse().ok()
            });
            let cache_prompt = match (said, opts.cache_prompt) {
                (Some(s), Some(o)) if s != o => {
                    bail!(
                        "--cache-prompt {o}: the server's /metrics says enable_prefix_caching {s}"
                    )
                }
                (Some(s), _) | (None, Some(s)) => s,
                (None, None) => bail!(
                    "the server's /metrics does not say whether it caches prompt prefixes: state it with --cache-prompt true|false"
                ),
            };
            // A later run, or a --duration wrap, would replay prompts an earlier one left cached.
            ensure!(
                !cache_prompt || (opts.concurrency.len() == 1 && opts.duration.is_none()),
                "the server keeps its prefix cache between runs: one --concurrency and no --duration per server start"
            );
            let mut o = OpenAi {
                server: json!({"model": model, "root": m["root"], "max_model_len": m["max_model_len"],
                    "owned_by": m["owned_by"], "version": version}),
                model: model.into(),
                host,
                cache_prompt,
                ignore_eos: !opts.natural_eos,
                warmup: Value::Null,
            };
            o.warmup = openai_warm_up(&o, &tok.encode("warm up", true))?;
            Target::OpenAi(o)
        }
        EngineKind::Larch => {
            let engine = engine.context("--engine larch needs a loaded engine")?;
            let (t, m) = vocab_of(vocab)?;
            // Takes shutdown ownership: the engine stops when this facade drops at return.
            let facade = Larch::from_engine(engine.clone(), t, m);
            let rejected: Vec<_> = facade
                .rejected_overrides()
                .iter()
                .map(|(k, v, why)| format!("--set {}={v} rejected: {why:?}", knob_name(*k)))
                .collect();
            ensure!(rejected.is_empty(), "{}", rejected.join("; "));
            // --pp is the fixed shape's; a workload's prompts vary, so it keeps the short request.
            warm_up(
                &engine,
                if workload == Workload::Fixed {
                    opts.pp
                } else {
                    16
                },
            )?;
            Target::Larch { engine, facade }
        }
    };
    let bench = Bench {
        opts,
        tok: &tok,
        tmpl: &tmpl,
        workload,
        target,
        live,
    };
    let Some(file) = file else {
        let result = bench.fixed(out)?;
        return write_result(opts.json.as_deref(), &result);
    };
    let prompts = bench.prompts(&file)?;
    let (mut failures, mut unapplied) = (0, 0);
    for &c in &opts.concurrency {
        let result = bench.workload_run(&file, &prompts, c, out)?;
        failures += result["summary"]["failures"].as_u64().unwrap_or(0);
        unapplied += result["knob_timeline"]
            .as_array()
            .map_or(0, |t| t.iter().filter(|e| e["applied"] == false).count());
        let json = opts.json.as_deref().map(|p| {
            if opts.concurrency.len() > 1 {
                let stem = p.file_stem().unwrap_or_default().to_string_lossy();
                let ext = p
                    .extension()
                    .map_or(String::new(), |e| format!(".{}", e.to_string_lossy()));
                p.with_file_name(format!("{stem}-c{c}{ext}"))
            } else {
                p.to_path_buf()
            }
        });
        write_result(json.as_deref(), &result)?;
    }
    ensure!(failures == 0, "{failures} requests failed");
    ensure!(
        unapplied == 0,
        "{unapplied} --set k=v@S not applied (knob_timeline: refused, indeterminate or after the run ended): the run did not have them"
    );
    Ok(())
}

/// `--url`'s `host:port`, port 80 when it names none.
fn host(opts: &Options) -> Result<String> {
    let url = opts.url.as_deref().context("an HTTP engine needs --url")?;
    let host = url
        .strip_prefix("http://")
        .context("--url: plain http://host:port")?
        .trim_end_matches('/');
    ensure!(!host.contains('/'), "--url: no path, http://host:port");
    Ok(if host.contains(':') {
        host.into()
    } else {
        format!("{host}:80")
    })
}

fn write_result(path: Option<&Path>, v: &Value) -> Result<()> {
    if let Some(p) = path {
        fs::write(p, serde_json::to_string_pretty(v)? + "\n")
            .with_context(|| p.display().to_string())?;
    }
    Ok(())
}

/// One `n`-token request before the clock, as the llama.cpp driver warms its context, so a path
/// taken only above some row count (HIP: rocBLAS loads its code objects on its first call) is not
/// first run in a timed pass. Clamped to what a request may hold.
fn warm_up(e: &EngineHandle, n: u32) -> Result<()> {
    let d = e.deployment();
    let n = n.min(d.ctx_max.saturating_sub(2));
    let h = e.submit(Request::new((0..n).map(|t| t % d.vocab).collect(), 2))?;
    for ev in h.events.iter() {
        match ev {
            EngineEvent::Token { .. } | EngineEvent::Prefilled { .. } => {}
            EngineEvent::Done { .. } => return Ok(()),
            EngineEvent::Error(err) => bail!("warm-up: {err:?}"),
        }
    }
    bail!("warm-up: the engine closed the stream")
}

enum Target {
    Larch { engine: EngineHandle, facade: Larch },
    Ollama(Ollama),
    OpenAi(OpenAi),
}

struct Ollama {
    host: String,
    model: String,
    n_ctx: u32,
    larch: bool,
    warmup: Value,
}

struct OpenAi {
    host: String,
    /// The first id `/v1/models` lists.
    model: String,
    /// What the server says of itself, null where it is silent.
    server: Value,
    /// Its prefix cache, from `/metrics` or `--cache-prompt`.
    cache_prompt: bool,
    ignore_eos: bool,
    warmup: Value,
}

type Prompts = Vec<Vec<(String, Vec<u32>)>>;

struct Bench<'a> {
    opts: &'a Options,
    tok: &'a Tokenizer,
    tmpl: &'a ChatTemplate,
    workload: Workload,
    target: Target,
    live: Vec<KnobSet>,
}

/// One concurrency's run: the clock, the clients' shared cursors and the pre-rendered prompts.
struct Clock<'a> {
    file: &'a WorkloadFile,
    /// Conversations in play: one per client with reuse (the driver's `--conversations N`), else all;
    /// with `--slice`, those from `first` up to `n_conv`.
    first: usize,
    n_conv: usize,
    /// Scripted prompts per conversation and turn, rendered before the clock starts.
    prompts: &'a Prompts,
    t0: Instant,
    deadline: Option<Instant>,
    next_conv: AtomicUsize,
    next_req: AtomicUsize,
}

impl Clock<'_> {
    fn ms(&self) -> f64 {
        self.t0.elapsed().as_secs_f64() * 1e3
    }

    fn expired(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// Spreads `rate` evenly over the requests in submission order: exactly `floor(n * rate)` of n.
    fn pick_cancel(&self, rate: f64) -> bool {
        let k = self.next_req.fetch_add(1, Ordering::Relaxed) as f64;
        (k * rate).floor() != ((k + 1.0) * rate).floor()
    }
}

#[derive(Default)]
struct RingStats {
    engine_samples: u64,
    last_engine_step: Option<u64>,
    incomplete_engine_samples: bool,
    engine_total_preemptions: u64,
    peak_vram_used_bytes: Option<u64>,
    gbps_sum: f64,
    gbps_n: u64,
    revisions: Vec<Value>,
    guardrail_trips: u64,
    shapes: BTreeMap<(u8, u32, u32, u32), StepTimes>,
    active_sequences: BTreeMap<u32, u64>,
    max_waiting: u32,
    peak_kv_blocks_used: u32,
    peak_kv_tokens_used: u32,
    kv_blocks_total: u32,
}

#[derive(Default)]
struct StepTimes {
    steps: u64,
    dur_ns: u64,
    gpu_ns: u64,
    measured_gpu_steps: u64,
    host_ns: u64,
    graph_replays: u64,
    capture_marker_samples: u64,
    graph_captures: u64,
    capture_step_dur_ns: u64,
}

impl RingStats {
    fn step_profile(&self, dropped: u64) -> Value {
        let sum = |f: fn(&StepTimes) -> u64| self.shapes.values().map(f).sum::<u64>();
        let captures_complete = self.engine_samples > 0
            && !self.incomplete_engine_samples
            && dropped == 0
            && sum(|t| t.capture_marker_samples) == self.engine_samples;
        let shapes: Vec<_> = self
            .shapes
            .iter()
            .map(|(&(phase, decode, prefill, tokens), t)| {
                json!({
                    "phase": format!("{:?}", Phase::from_raw(phase)),
                    "decode_sequences": decode, "prefill_sequences": prefill,
                    "prefill_tokens": tokens, "steps": t.steps, "dur_ns": t.dur_ns,
                    "gpu_ns": (t.measured_gpu_steps == t.steps).then_some(t.gpu_ns),
                    "host_ns": t.host_ns, "graph_replays": t.graph_replays,
                    "graph_captures": captures_complete.then_some(t.graph_captures),
                    "capture_step_dur_ns": captures_complete.then_some(t.capture_step_dur_ns),
                })
            })
            .collect();
        json!({
            "steps": sum(|t| t.steps), "dur_ns": sum(|t| t.dur_ns),
            "gpu_ns": (sum(|t| t.measured_gpu_steps) == self.engine_samples && self.engine_samples > 0).then(|| sum(|t| t.gpu_ns)),
            // host_ns is the engine's residual step duration, including download/sync gaps.
            "host_ns": sum(|t| t.host_ns), "graph_replays": sum(|t| t.graph_replays),
            "capture_counts_complete": captures_complete,
            "graph_captures": captures_complete.then(|| sum(|t| t.graph_captures)),
            // Entire steps that successfully captured, including upload, replay and sampling.
            // This is not isolated CUDA capture API latency.
            "capture_step_dur_ns": captures_complete.then(|| sum(|t| t.capture_step_dur_ns)),
            "shapes": shapes,
            "active_sequences": self.active_sequences.iter().map(|(sequences, dur_ns)| json!({"sequences": sequences, "dur_ns": dur_ns})).collect::<Vec<_>>(),
            "max_waiting": self.max_waiting, "kv_blocks_total": self.kv_blocks_total,
            "peak_kv_blocks_used": self.peak_kv_blocks_used,
            "peak_kv_tokens_used": self.peak_kv_tokens_used,
        })
    }

    fn take(&mut self, r: Record) {
        match r {
            Record::EngineSample(h, s) => {
                self.engine_samples += 1;
                self.incomplete_engine_samples |= h.tier != 1
                    || self
                        .last_engine_step
                        .is_some_and(|last| last.checked_add(1) != Some(s.step));
                self.last_engine_step = Some(s.step);
                self.engine_total_preemptions =
                    self.engine_total_preemptions.max(s.tot_preemptions);
                let t = self
                    .shapes
                    .entry((
                        s.phase,
                        s.n_decode_seqs,
                        s.n_prefill_seqs,
                        s.n_prefill_tokens,
                    ))
                    .or_default();
                t.steps += 1;
                t.dur_ns += s.dur_ns;
                t.host_ns += s.host_ns;
                t.graph_replays += u64::from(s.graph_replayed != 0);
                if let Some(captured) = s.graph_captured() {
                    t.capture_marker_samples += 1;
                    if captured {
                        t.graph_captures += 1;
                        t.capture_step_dur_ns += s.dur_ns;
                    }
                }
                if let Some(ns) = s.gpu_ns.get() {
                    t.gpu_ns += ns;
                    t.measured_gpu_steps += 1;
                }
                *self
                    .active_sequences
                    .entry(s.n_decode_seqs + s.n_prefill_seqs)
                    .or_default() += s.dur_ns;
                self.max_waiting = self.max_waiting.max(s.n_waiting);
                self.peak_kv_blocks_used = self.peak_kv_blocks_used.max(s.kv_blocks_used);
                self.peak_kv_tokens_used = self.peak_kv_tokens_used.max(s.kv_tokens_used);
                self.kv_blocks_total = self.kv_blocks_total.max(s.kv_blocks_total);
                if let Some(g) = s.achieved_gbps.get() {
                    self.gbps_sum += f64::from(g);
                    self.gbps_n += 1;
                }
            }
            Record::DeviceSample(_, s) => {
                self.peak_vram_used_bytes = self.peak_vram_used_bytes.max(s.vram_used_bytes.get());
            }
            Record::ConfigurationRevision(_, c) => {
                self.guardrail_trips += u64::from(c.cause() == Cause::Guardrail);
                self.revisions.push(json!({
                    "rev": c.rev, "knob": knob_name(c.knob), "cause": format!("{:?}", c.cause()),
                    "requested": c.requested, "effective": c.effective, "previous": c.previous,
                    "step": c.step_applied, "trial_id": c.trial_id,
                }));
            }
            Record::HealthEvent(_, h) if h.fault() != larch_records::FaultCode::None => {
                // A failed step may have captured before returning without an EngineSample.
                self.incomplete_engine_samples = true;
            }
            _ => {}
        }
    }
}

impl Bench<'_> {
    /// Scripted prompts per conversation and turn as (text, tokens), rendered before any clock.
    fn prompts(&self, file: &WorkloadFile) -> Result<Prompts> {
        let scripted = self.opts.replay == Replay::Scripted;
        file.convs
            .iter()
            .map(|conv| {
                conv.turns
                    .iter()
                    .filter(|_| scripted)
                    .map(|t| render(self.tok, self.tmpl, &conv.messages[..t.n_msgs]))
                    .collect::<Result<_>>()
            })
            .collect()
    }

    fn workload_run(
        &self,
        file: &WorkloadFile,
        prompts: &Prompts,
        c: usize,
        out: &mut impl Write,
    ) -> Result<Value> {
        let reuse = self.workload.reuse();
        let n_conv = if reuse {
            c.min(file.convs.len())
        } else {
            file.convs.len()
        };
        let (first, n_conv) = match self.opts.slice {
            Some((k, n)) => (k * n_conv / n, (k + 1) * n_conv / n),
            None => (0, n_conv),
        };
        let ring = match &self.target {
            Target::Larch { facade, .. } => Some(facade.telemetry().reader(ReadFrom::Latest)),
            Target::Ollama(_) | Target::OpenAi(_) => None,
        };
        // A later switch to T1 cannot recover captures from an initially silent/decimated tier.
        let all_steps_t1 = match &self.target {
            Target::Larch { facade, .. } => {
                self.opts.telemetry != Telemetry::Off
                    && facade.knobs().effective().values[knob_id::TELEMETRY_TIER as usize] == 1.0
                    && self
                        .live
                        .iter()
                        .all(|l| l.knob != knob_id::TELEMETRY_TIER || l.value == 1.0)
            }
            Target::Ollama(_) | Target::OpenAi(_) => false,
        };
        let t0 = Instant::now();
        let clock = Clock {
            file,
            first,
            n_conv,
            prompts,
            t0,
            deadline: match self.opts.duration {
                Some(s) => Some(t0.checked_add(secs(s)?).context("--duration")?),
                None => None,
            },
            // Client k starts on conversation k, as every slot of the driver gets a request in its
            // first iteration; later conversations go to whichever client frees up first.
            next_conv: AtomicUsize::new(first + c),
            next_req: AtomicUsize::new(0),
        };
        let done = AtomicBool::new(false);
        let (stop_live, live_rx) = mpsc::channel::<()>();
        // Dropped once every change due at 0 s is in force; the clients start then.
        let (at_zero, live_started) = mpsc::channel::<()>();
        let (header, mut reqs, ring, timeline) = thread::scope(|s| {
            let ring = ring.map(|mut reader| {
                let done = &done;
                s.spawn(move || {
                    let (mut st, mut buf) = (
                        RingStats {
                            incomplete_engine_samples: !all_steps_t1,
                            ..RingStats::default()
                        },
                        Vec::new(),
                    );
                    loop {
                        let last = done.load(Ordering::Acquire);
                        reader.drain(&mut buf, usize::MAX);
                        buf.drain(..).for_each(|r| st.take(r));
                        if last {
                            return (st, reader.dropped_total());
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                })
            });
            let timeline = match &self.target {
                Target::Larch { facade, .. } => {
                    let control = facade.knobs();
                    Some(s.spawn(move || {
                        let mut timeline = Vec::new();
                        let mut ended = false;
                        let mut at_zero = Some(at_zero);
                        for l in &self.live {
                            if !l.at.is_zero() {
                                drop(at_zero.take());
                            }
                            let wait = l.at.saturating_sub(t0.elapsed());
                            ended = ended
                                || live_rx.recv_timeout(wait)
                                    == Err(mpsc::RecvTimeoutError::Disconnected);
                            // A change the run never reached or the engine did not apply is
                            // recorded, and fails the invocation.
                            let (reply, applied) = if ended {
                                (NOT_REACHED.into(), false)
                            } else {
                                let r = KnobRequest::new(l.knob, l.value, Authority::Human);
                                let reply = control.set(r);
                                let applied = matches!(reply, KnobReply::Applied { .. });
                                (format!("{reply:?}"), applied)
                            };
                            timeline.push(json!({
                                "at_s": l.at_s, "t_ms": t0.elapsed().as_secs_f64() * 1e3,
                                "knob": knob_name(l.knob), "value": l.value, "reply": reply,
                                "applied": applied,
                            }));
                        }
                        timeline
                    }))
                }
                Target::Ollama(_) | Target::OpenAi(_) => {
                    drop(at_zero);
                    None
                }
            };
            // An error is the sender dropped: nothing more is due at 0 s.
            let _ = live_started.recv();
            // After the changes due at 0 s, so its params are the ones the run starts with.
            let header = self.header("run");
            let clients: Vec<_> = (0..c)
                .map(|client| {
                    let clock = &clock;
                    s.spawn(move || self.client(clock, client))
                })
                .collect();
            let joined: Vec<_> = clients.into_iter().map(|h| h.join()).collect();
            // A round trip through the engine thread: it answers only after the last step wrote its
            // sample and ran the guardrail, so the ring reader sees them.
            if let Target::Larch { engine, .. } = &self.target {
                engine.parked_tokens(0);
            }
            // Stop the ring and live threads before a client's panic propagates, or the scope waits forever.
            drop(stop_live);
            done.store(true, Ordering::Release);
            let reqs: Vec<Req> = joined
                .into_iter()
                .flat_map(|j| j.unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
                .collect();
            (
                header,
                reqs,
                ring.map(|h| h.join().expect("ring reader")),
                timeline.map(|h| h.join().expect("live --set")),
            )
        });
        let mut v = header?;
        reqs.sort_by(|a, b| a.t_submit_ms.total_cmp(&b.t_submit_ms));
        let s = summary(&reqs);
        v["replay"] = json!(self.opts.replay);
        v["workload"] =
            json!({"file": file.name, "sha256": file.sha256, "conversations": n_conv - first});
        if let Some((k, n)) = self.opts.slice {
            v["slice"] = json!({"k": k, "n": n, "first": first, "end": n_conv});
        }
        v["concurrency"] = json!(c);
        v["cache_prompt"] = json!(match &self.target {
            Target::Larch { .. } => reuse,
            // Ollama always reuses prompt prefixes (its runner sends cache_prompt), so `compare`
            // refuses its W4 and W5 against llama.cpp's rows, which reuse nothing.
            Target::Ollama(_) => true,
            Target::OpenAi(o) => o.cache_prompt,
        });
        if self.opts.engine == EngineKind::LarchHttp {
            v["cache_policy"] = json!(
                "exact scripted prompts; reuse only within corpus conversation identity; release after final turn"
            );
        }
        match &self.target {
            Target::Ollama(o) => v["warmup"] = o.warmup.clone(),
            Target::OpenAi(o) => {
                v["warmup"] = o.warmup.clone();
                v["server"] = o.server.clone();
            }
            Target::Larch { .. } => {}
        }
        v["ignore_eos"] = json!(
            matches!(self.opts.engine, EngineKind::Larch | EngineKind::OpenAi)
                && self.opts.replay == Replay::Scripted
                && !self.opts.natural_eos
        );
        v["stream_boundary"] = json!(if self.opts.engine != EngineKind::Larch {
            "http-text-chunks"
        } else if self.opts.replay == Replay::Generated {
            "in-process-text-chunks"
        } else {
            "in-process-tokens"
        });
        v["cancel_rate"] = json!(self.opts.cancel_rate);
        v["duration"] = json!(self.opts.duration);
        if let (Some((st, dropped)), Some(live)) = (ring, timeline) {
            let mut t: Vec<Value> = startup(self.opts)?
                .iter()
                .map(|(k, val)| json!({"at_s": null, "knob": knob_name(*k), "value": val, "reply": "startup"}))
                .collect();
            t.extend(live);
            v["knob_timeline"] = json!(t);
            v["telemetry"] = json!({
                "engine_samples": st.engine_samples,
                "engine_total_preemptions": (st.engine_samples > 0).then_some(st.engine_total_preemptions),
                "step_profile": st.step_profile(dropped),
                "peak_vram_used_bytes": st.peak_vram_used_bytes,
                "achieved_gbps": (st.gbps_n > 0).then(|| st.gbps_sum / st.gbps_n as f64),
                "revisions": st.revisions,
                "guardrail_trips": st.guardrail_trips,
                "dropped": dropped,
            });
        }
        let q = |k: &str, p: &str| s[k][p].as_f64().unwrap_or(f64::NAN);
        write!(
            out,
            "{} c={c}: {} requests, {} failed, {} cancelled, {:.2} tok/s, TTFT p50 {:.1} p99 {:.1} ms, TPOT p50 {:.2} p99 {:.2} ms",
            file.name,
            s["requests"],
            s["failures"],
            s["cancelled"],
            s["agg_tok_s"].as_f64().unwrap_or(0.0),
            q("ttft_ms", "p50"),
            q("ttft_ms", "p99"),
            q("tpot_ms", "p50"),
            q("tpot_ms", "p99"),
        )?;
        if s.get("http_overhead_ms").is_some() {
            write!(
                out,
                ", HTTP overhead p50 {:.1} ms",
                q("http_overhead_ms", "p50")
            )?;
        }
        writeln!(out)?;
        v["summary"] = s;
        v["requests"] = serde_json::to_value(&reqs)?;
        Ok(v)
    }

    /// A closed-loop client: takes the next conversation and runs its turns back to back.
    fn client(&self, clock: &Clock, client: usize) -> Vec<Req> {
        let mut out = Vec::new();
        let mut i = clock.first + client;
        loop {
            if clock.expired() || (clock.deadline.is_none() && i >= clock.n_conv) {
                return out;
            }
            let ci = i % clock.n_conv;
            let conv = &clock.file.convs[ci];
            let reuse = self.workload.reuse();
            let mut session = match &self.target {
                Target::Larch { facade, .. } if self.opts.replay == Replay::Generated => {
                    Some(
                        facade.session(SessionOptions {
                            system: (conv.messages[0].role == "system")
                                .then(|| conv.messages[0].content.clone()),
                            keep_kv: reuse,
                            // Overflow is tested on its own, never inside a benchmark row (10.1.1).
                            on_overflow: Overflow::Fail,
                        }),
                    )
                }
                _ => None,
            };
            let id = SCRIPTED_SESSIONS.fetch_add(1, Ordering::Relaxed);
            for (k, turn) in conv.turns.iter().enumerate() {
                if clock.expired() {
                    break;
                }
                let mut r = Req {
                    client,
                    slot: client,
                    conv: conv.id.clone(),
                    turn: k,
                    max_tokens: turn.max_tokens,
                    cancel: clock.pick_cancel(self.opts.cancel_rate),
                    ..Req::default()
                };
                match (&self.target, session.as_mut()) {
                    (_, Some(s)) => {
                        self.session_turn(clock, s, &mut r, &conv.messages[turn.n_msgs - 1].content)
                    }
                    (Target::Larch { engine, .. }, None) => {
                        let prompt = &clock.prompts[ci][k].1;
                        engine_turn(
                            clock,
                            engine,
                            self.tok,
                            &mut r,
                            prompt,
                            reuse.then_some(id),
                            !self.opts.natural_eos,
                        )
                    }
                    (Target::Ollama(o), None) => {
                        let (text, prompt) = &clock.prompts[ci][k];
                        if o.larch {
                            r.session_end = Some(k + 1 == conv.turns.len());
                        }
                        ollama_turn(clock, o, &mut r, text, prompt)
                    }
                    (Target::OpenAi(o), None) => {
                        openai_turn(clock, o, &mut r, &clock.prompts[ci][k].1)
                    }
                }
                out.push(r.finish());
            }
            if let (Target::Larch { engine, .. }, true) = (&self.target, reuse) {
                engine.release_session(id);
            }
            i = clock.next_conv.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Generated replay: the facade's `Session` continues with larch's own answers, and every turn
    /// must reuse exactly the KV the previous one parked (architecture 16, D14). Its TTFT is the
    /// first delivered text, held back for UTF-8: this mode checks bookkeeping, scripted is timed.
    fn session_turn(&self, clock: &Clock, s: &mut crate::Session, r: &mut Req, user: &str) {
        let kv = s.tokens_in_kv();
        r.kv_tokens = Some(kv);
        let mut history: Vec<Message> = s
            .history()
            .into_iter()
            .map(|m| Message {
                role: m.role.as_str().into(),
                content: m.content,
            })
            .collect();
        history.push(Message {
            role: "user".into(),
            content: user.into(),
        });
        match render(self.tok, self.tmpl, &history) {
            Ok((_, p)) => {
                r.prompt_tokens = p.len() as u32;
                r.prompt_sha256 = prompt_sha256(&p);
            }
            Err(e) => r.error = Some(format!("{e:#}")),
        }
        r.t_submit_ms = clock.ms();
        let params = GenParams {
            max_tokens: r.max_tokens,
            // Greedy as the scripted path: no knob default may change the argmax.
            temperature: Some(0.0),
            top_k: Some(0),
            top_p: Some(1.0),
            repeat_penalty: Some(1.0),
            ..GenParams::default()
        };
        match s.send(user, params) {
            Err(e) => r.error = Some(format!("{e:?}")),
            Ok(mut stream) => {
                while let Some(ev) = stream.next() {
                    match ev {
                        TextEvent::Prefilled { .. } | TextEvent::Thinking { .. } => {}
                        TextEvent::Text { tokens, .. } => {
                            r.gen_tokens += tokens;
                            if tokens > 0 && r.t_first_ms.is_none() {
                                r.t_first_ms = Some(clock.ms());
                                if r.cancel {
                                    stream.cancel();
                                }
                            }
                        }
                        TextEvent::Done { usage, timings, .. } => {
                            r.gen_tokens = usage.completion_tokens;
                            r.cached_tokens = Some(timings.cached_tokens);
                            if usage.prompt_tokens != r.prompt_tokens {
                                r.error = Some(format!(
                                    "the session sent {} prompt tokens, the bench render has {}",
                                    usage.prompt_tokens, r.prompt_tokens
                                ));
                            } else if timings.cached_tokens != kv {
                                r.error = Some(format!(
                                    "cached_tokens {} != tokens_in_kv {kv}",
                                    timings.cached_tokens
                                ));
                            }
                        }
                        TextEvent::Error(e) => r.error = Some(format!("{e:?}")),
                    }
                }
            }
        }
        r.t_done_ms = clock.ms();
    }

    fn header(&self, mode: &str) -> Result<Value> {
        let mut params = json!({});
        for k in [
            "n_ctx",
            "n_ctx_seq",
            "n_seq_max",
            "n_batch",
            "n_ubatch",
            "n_gpu_layers",
            "flash_attn",
            "type_k",
            "type_v",
            "n_threads",
            "n_threads_batch",
            "kv_unified",
        ] {
            params[k] = Value::Null;
        }
        let (engine, system_info) = match &self.target {
            Target::Larch { engine, facade } => {
                let d = engine.deployment();
                params["n_ctx_seq"] = json!(d.ctx_max);
                let knobs = facade.knobs().effective();
                params["n_seq_max"] = json!(knobs.values[knob_id::MAX_BATCH_SEQS as usize]);
                // The tier in force: --telemetry, or a startup or 0 s telemetry_tier over it.
                params["telemetry"] = match self.opts.telemetry {
                    Telemetry::Off => json!(Telemetry::Off),
                    _ if knobs.values[knob_id::TELEMETRY_TIER as usize] == 1.0 => {
                        json!(Telemetry::T1)
                    }
                    _ => json!(Telemetry::T0),
                };
                (
                    "larch",
                    format!("{:?} {}", d.backend(), d.device_name.as_str()),
                )
            }
            Target::Ollama(o) => {
                params["n_ctx_seq"] = json!(o.n_ctx);
                params["ollama_model"] = json!(o.model);
                (if o.larch { "larch-http" } else { "ollama" }, String::new())
            }
            Target::OpenAi(o) => {
                params["n_ctx_seq"] = o.server["max_model_len"].clone();
                ("openai", String::new())
            }
        };
        Ok(json!({
            "schema": "larch-bench/1",
            "engine": engine,
            "driver": "larch bench",
            "mode": mode,
            "llama_cpp": null,
            "system_info": system_info,
            "machine": self.opts.machine,
            "started_at": utc_now(),
            "build": build(),
            "model": file_name(&self.opts.model),
            "date_string": DATE,
            "params": params,
            // The opt-in route switches this process read; an HTTP engine's are its own.
            "routes": matches!(self.target, Target::Larch { .. }).then(|| {
                let r = larch_kernels::routes();
                json!({"quant_i8": r.quant_i8, "prefill_tc": r.prefill_tc, "mixed_graphs": r.mixed_graphs})
            }),
        }))
    }

    /// W6: `llama-batched-bench`'s shape. Per npl, npl pp-token random prompts are prefilled and
    /// parked (one token sampled each), then all npl decode together from the parked KV, feeding tg
    /// tokens each; S = npl * (pp + tg) / (t_pp + t_tg). A row is refused unless every prompt stayed
    /// parked whole and the engine ran the decode phase as the driver's decode loop ([`decode_shape`]).
    fn fixed(&self, out: &mut impl Write) -> Result<Value> {
        let Target::Larch { engine, .. } = &self.target else {
            bail!("the fixed-shape mirror runs on larch");
        };
        let (pp, tg) = (self.opts.pp as usize, self.opts.tg as usize);
        let d = engine.deployment();
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        // The driver's bound: tg + 1 are sampled, so pp + tg must stay under the context.
        ensure!(
            pp + tg < d.ctx_max as usize,
            "pp={pp} tg={tg}: context {} is too small",
            d.ctx_max
        );
        let mut v = self.header("fixed")?;
        let mut rows = Vec::new();
        for &pl in &self.opts.npl {
            let prompts: Vec<Vec<u32>> = (0..pl)
                .map(|_| {
                    (0..pp)
                        .map(|_| {
                            // xorshift64: reproducible prompts; llama.cpp's are rand() % n_vocab.
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            (seed % u64::from(d.vocab)) as u32
                        })
                        .collect()
                })
                .collect();
            let ids: Vec<u64> = (0..pl)
                .map(|_| SCRIPTED_SESSIONS.fetch_add(1, Ordering::Relaxed))
                .collect();
            let parked = |prompt: Vec<u32>, id: u64, max_tokens: u32| Request {
                // The end-of-generation ids ignore_eos must never sample (#61), as on the scripted path.
                stop_tokens: self.tok.eog().to_vec(),
                ignore_eos: true,
                session: id,
                keep_kv: true,
                ..Request::new(prompt, max_tokens)
            };
            let phases = || -> Result<(f64, f64, u32, usize, u64)> {
                let first = run_all(
                    engine,
                    prompts
                        .iter()
                        .zip(&ids)
                        .map(|(p, &id)| parked(p.clone(), id, 1)),
                )?;
                let t_pp = first.iter().map(|f| f.1).fold(0.0, f64::max);
                for &id in &ids {
                    let n = engine.parked_tokens(id) as usize;
                    ensure!(
                        n == pp,
                        "{n} of a prompt's {pp} tokens parked after the prefill phase: the KV does not hold {pl} prompts, so decode would recompute them"
                    );
                }
                // Each parked_tokens is a round trip through the engine thread, which answers it only
                // after the prefill phase's last step wrote its sample: the decode phase's start here.
                let mut ring = engine.ring().reader(ReadFrom::Latest);
                let eager = engine.eager_decode_steps();
                let next = prompts.iter().zip(&ids).zip(&first).map(|((p, &id), f)| {
                    let mut p = p.clone();
                    p.push(f.0);
                    parked(p, id, self.opts.tg)
                });
                let decoded = run_all(engine, next)?;
                let t_tg = decoded.iter().map(|f| f.1).fold(0.0, f64::max);
                engine.parked_tokens(ids[0]); // the round trip: the last decode step's sample is in
                let mut records = Vec::new();
                ring.drain(&mut records, usize::MAX);
                ensure!(
                    ring.dropped_total() == 0,
                    "the ring dropped decode-phase samples"
                );
                let steps: Vec<EngineSample> = records
                    .into_iter()
                    .filter_map(|r| match r {
                        Record::EngineSample(_, s) => Some(s),
                        _ => None,
                    })
                    .collect();
                decode_shape(&steps, pl, tg)?;
                let replays = steps.iter().filter(|s| s.graph_replayed != 0).count();
                let eager = engine.eager_decode_steps() - eager;
                let cached: Vec<u32> = decoded.iter().map(|f| f.2).collect();
                ensure!(
                    cached.iter().all(|&c| c as usize == pp),
                    "decode reused {cached:?} of each prompt's {pp} parked tokens"
                );
                Ok((t_pp, t_tg, cached.iter().sum(), replays, eager))
            };
            let phases = phases();
            ids.iter().for_each(|&id| engine.release_session(id));
            let (t_pp, t_tg, cached, replays, eager) =
                phases.with_context(|| format!("fixed pp={pp} tg={tg} pl={pl}"))?;
            // The driver's T: its two phases, without bench's checks between them.
            let t = t_pp + t_tg;
            let s = (pl * (pp + tg)) as f64 / t;
            writeln!(
                out,
                "fixed pp={pp} tg={tg} pl={pl}: T {t:.3} s, S {s:.2} t/s"
            )?;
            rows.push(json!({
                "pp": pp, "tg": tg, "pl": pl, "n_kv": pl * (pp + tg),
                "t_pp": t_pp, "speed_pp": (pl * pp) as f64 / t_pp,
                "t_tg": t_tg, "speed_tg": (pl * tg) as f64 / t_tg,
                "t": t, "speed": s, "cached_tokens": cached,
                // Decode-phase steps that replayed a graph, and that ran eager on a refused capture.
                "graph_replays": replays, "eager_decode_steps": eager,
            }));
        }
        v["fixed"] = json!(rows);
        Ok(v)
    }
}

/// Refuses a W6 decode phase unless the engine ran it as llama-batched-bench's decode loop: `tg`
/// consecutive steps, each exactly `pl` decode rows and no prefill. `steps` are the phase's t1
/// engine samples, in order.
pub fn decode_shape(steps: &[EngineSample], pl: usize, tg: usize) -> Result<()> {
    ensure!(
        !steps.is_empty() && steps.windows(2).all(|w| w[1].step == w[0].step + 1),
        "the decode phase's engine samples skip steps: the fixed shape needs every one (telemetry_tier t1)"
    );
    for (i, s) in steps.iter().enumerate() {
        ensure!(
            s.phase() == Phase::Decode && s.n_decode_seqs as usize == pl && s.n_prefill_seqs == 0,
            "decode step {i} of {tg} was {:?} with {} decode and {} prefill rows; the shape decodes all {pl} together (seats, chunk budget and KV must hold them)",
            s.phase(),
            s.n_decode_seqs,
            s.n_prefill_seqs
        );
    }
    ensure!(
        steps.len() == tg,
        "the decode phase took {} engine steps; the shape takes {tg}",
        steps.len()
    );
    Ok(())
}

/// Submits `reqs` as one group, admitted in one engine turn (#64), and waits for all: per request
/// its first sampled token, the time of its Done in seconds from the submission, and its cached
/// prompt tokens.
fn run_all(e: &EngineHandle, reqs: impl Iterator<Item = Request>) -> Result<Vec<(u32, f64, u32)>> {
    let reqs = reqs.collect(); // built before the clock, as the driver's batch
    let t0 = Instant::now();
    let handles = e.submit_group(reqs)?;
    thread::scope(|s| {
        let joins: Vec<_> = handles
            .into_iter()
            .map(|h| {
                s.spawn(move || {
                    let mut first = None;
                    for ev in h.events.iter() {
                        match ev {
                            EngineEvent::Prefilled { .. } => {}
                            EngineEvent::Token { id, .. } => {
                                first.get_or_insert(id);
                            }
                            EngineEvent::Done { usage, .. } => {
                                let t = t0.elapsed().as_secs_f64();
                                let first = first.context("Done without a token")?;
                                return Ok((first, t, usage.cached_tokens));
                            }
                            EngineEvent::Error(err) => bail!("{err:?}"),
                        }
                    }
                    bail!("the engine closed the stream")
                })
            })
            .collect();
        joins
            .into_iter()
            .map(|j| j.join().expect("fixed client"))
            .collect()
    })
}

/// Scripted replay and W4 to W5 on larch: the prompt the other engines get, greedy, EOS ignored
/// (`Request::ignore_eos` is bench's; `GenParams` has none, 10.1). With reuse, the conversation
/// parks under `session` and its next turn keeps the common prefix.
fn engine_turn(
    clock: &Clock,
    e: &EngineHandle,
    tok: &Tokenizer,
    r: &mut Req,
    prompt: &[u32],
    session: Option<u64>,
    ignore_eos: bool,
) {
    r.prompt_tokens = prompt.len() as u32;
    r.prompt_sha256 = prompt_sha256(prompt);
    r.token_gaps_ms = Some(Vec::new());
    let mut last_token_ms = None;
    let req = Request {
        stop_tokens: tok.eog().to_vec(),
        ignore_eos,
        session: session.unwrap_or(0),
        keep_kv: session.is_some(),
        ..Request::new(prompt.to_vec(), r.max_tokens)
    };
    r.t_submit_ms = clock.ms();
    match e.submit(req) {
        Err(err) => r.error = Some(format!("{err:?}")),
        Ok(h) => {
            r.error = Some("the engine closed the stream".into());
            for ev in h.events.iter() {
                match ev {
                    EngineEvent::Prefilled { .. } => {}
                    EngineEvent::Token { .. } => {
                        let now = clock.ms();
                        if let Some(last) = last_token_ms.replace(now) {
                            r.token_gaps_ms.as_mut().unwrap().push(now - last);
                        }
                        r.gen_tokens += 1;
                        if r.t_first_ms.is_none() {
                            r.t_first_ms = Some(now);
                            if r.cancel {
                                h.cancel();
                            }
                        }
                    }
                    EngineEvent::Done { usage, finish, .. } => {
                        r.finish_reason = Some(format!("{finish:?}"));
                        r.gen_tokens = usage.completion_tokens;
                        r.cached_tokens = Some(usage.cached_tokens);
                        r.error = None;
                        break;
                    }
                    EngineEvent::Error(err) => {
                        r.error = Some(format!("{err:?}"));
                        break;
                    }
                }
            }
        }
    }
    r.t_done_ms = clock.ms();
}

/// Ollama (D15): the same rendered prompt as `raw` text on `/api/generate`, so its prompt is the
/// one the hash names (`/api/chat` would apply Ollama's own template), and a turn fails unless
/// Ollama tokenised exactly as many tokens. Greedy; Ollama has no ignore_eos, so an answer can
/// end early and `compare` then refuses the run. TTFT is the first streamed token.
fn ollama_turn(clock: &Clock, o: &Ollama, r: &mut Req, text: &str, prompt: &[u32]) {
    r.prompt_tokens = prompt.len() as u32;
    r.prompt_sha256 = prompt_sha256(prompt);
    r.t_submit_ms = clock.ms();
    let done = ollama_stream(o, text, r, || clock.ms());
    r.t_done_ms = clock.ms();
    match done {
        Ok(Some((ms, n))) if n == r.prompt_tokens => {
            r.http_overhead_ms = Some(r.t_done_ms - r.t_submit_ms - ms)
        }
        Ok(Some((_, n))) => {
            r.error = Some(format!(
                "Ollama tokenised {n} prompt tokens; the hash names {}",
                r.prompt_tokens
            ))
        }
        Ok(None) => r.error = Some("Cancelled".into()),
        Err(e) => r.error = Some(format!("{e:#}")),
    }
}

/// Loads the model at the run's context and runs one short answer before the clock, as larch
/// and the llama.cpp driver are warm when their clocks start.
fn ollama_warm_up(o: &Ollama) -> Result<Value> {
    let mut r = Req {
        max_tokens: 2,
        ..Req::default()
    };
    let start = Instant::now();
    let completed = ollama_stream(o, "warm up", &mut r, || 0.0).context("Ollama warm-up")?;
    Ok(json!({"wall_ms":start.elapsed().as_secs_f64()*1000.0,
        "server_total_ms":completed.map(|(ms, _)| ms), "model_load_ms":r.http_load_ms,
        "prompt_tokens":completed.map(|(_, n)| n), "cached_tokens":r.cached_tokens,
        "output_tokens":r.gen_tokens, "finish_reason":r.finish_reason}))
}

/// One streamed greedy generate over HTTP/1.0, so the body is plain NDJSON up to the server's close.
/// Returns the server's `total_duration` in ms and its `prompt_eval_count` (cached tokens included),
/// or None when cancelled (the client hangs up after the first token).
fn ollama_stream(
    o: &Ollama,
    text: &str,
    r: &mut Req,
    now: impl Fn() -> f64,
) -> Result<Option<(f64, u32)>> {
    let mut body = json!({
        "model": o.model, "prompt": text, "raw": true, "stream": true, "keep_alive": -1,
        "options": {
            // Greedy as larch and llama.cpp: Ollama's default repeat_penalty 1.1 would move the argmax.
            "temperature": 0.0, "repeat_penalty": 1.0, "num_predict": r.max_tokens, "num_ctx": o.n_ctx,
        },
    });
    if o.larch
        && let Some(last) = r.session_end
    {
        body["larch_session"] = json!({"conversation": r.conv, "turn": r.turn, "last": last});
    }
    let lines = http_ok(&o.host, "POST /api/generate HTTP/1.0", &body.to_string())?.lines();
    r.http_chunk_gaps_ms = Some(Vec::new());
    r.http_chunks = Some(0);
    r.gen_tokens_known = Some(false);
    let mut last_chunk = None;
    for line in lines {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line).context("Ollama stream")?;
        if let Some(e) = v.get("error") {
            bail!("ollama: {e}");
        }
        if v["done"] == true {
            if o.larch && r.session_end.is_some() {
                ensure!(
                    v["conversation"].as_str() == Some(r.conv.as_str())
                        && v["turn"].as_u64() == Some(r.turn as u64),
                    "adapter terminal conversation identity differs from the scripted request"
                );
            }
            r.finish_reason = v["done_reason"].as_str().map(str::to_owned);
            r.http_load_ms = v["load_duration"].as_f64().map(|ns| ns / 1e6);
            let generated = count(&v, "/eval_count")?.context("no eval_count")?;
            ensure!(
                generated <= r.max_tokens,
                "eval_count {generated} exceeds requested cap {}",
                r.max_tokens
            );
            r.gen_tokens = generated;
            r.gen_tokens_known = Some(true);
            let prompt = count(&v, "/prompt_eval_count")?.context("no prompt_eval_count")?;
            // Ollama's own count, absent on older servers: unknown then, never inferred.
            if let Some(c) = v["prompt_eval_cached_count"]
                .as_u64()
                .filter(|&c| c > u64::from(prompt))
            {
                bail!("Ollama reported {c} cached of {prompt} prompt tokens");
            }
            r.cached_tokens = count(&v, "/prompt_eval_cached_count")?;
            let total = v["total_duration"].as_f64().context("no total_duration")?;
            return Ok(Some((total / 1e6, prompt)));
        }
        if v["response"].as_str().is_none_or(str::is_empty) {
            continue;
        }
        let received = now();
        chunk(r, &mut last_chunk, received);
        if r.t_first_ms.is_none() {
            r.t_first_ms = Some(received);
            if r.cancel {
                return Ok(None);
            }
        }
    }
    bail!("the stream ended before done")
}

/// OpenAI-compatible: the hashed token ids themselves on `/v1/completions`, so no server template
/// or tokenizer touches the prompt, greedy, `ignore_eos` unless `--natural-eos`.
fn openai_turn(clock: &Clock, o: &OpenAi, r: &mut Req, prompt: &[u32]) {
    r.prompt_tokens = prompt.len() as u32;
    r.prompt_sha256 = prompt_sha256(prompt);
    let body = openai_body(o, prompt, r.max_tokens);
    r.t_submit_ms = clock.ms();
    let done = openai_stream(o, &body, r, || clock.ms());
    r.t_done_ms = clock.ms();
    match done {
        Ok(true) => {}
        Ok(false) => r.error = Some("Cancelled".into()),
        Err(e) => r.error = Some(format!("{e:#}")),
    }
}

/// One short answer before the clock, as larch and the llama.cpp driver are warm when theirs start.
fn openai_warm_up(o: &OpenAi, prompt: &[u32]) -> Result<Value> {
    let mut r = Req {
        prompt_tokens: prompt.len() as u32,
        max_tokens: 2,
        ..Req::default()
    };
    let start = Instant::now();
    openai_stream(o, &openai_body(o, prompt, 2), &mut r, || 0.0).context("OpenAI warm-up")?;
    Ok(json!({
        "wall_ms": start.elapsed().as_secs_f64() * 1000.0, "prompt_tokens": r.prompt_tokens,
        "cached_tokens": r.cached_tokens, "output_tokens": r.gen_tokens, "finish_reason": r.finish_reason,
    }))
}

/// The `/v1/completions` body for `prompt`, built before the clock as the in-process `Request` is.
fn openai_body(o: &OpenAi, prompt: &[u32], max_tokens: u32) -> String {
    json!({
        "model": o.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
        "ignore_eos": o.ignore_eos, "stream": true, "stream_options": {"include_usage": true},
    })
    .to_string()
}

/// One streamed greedy completion of `body` over SSE up to `data: [DONE]`; false when cancelled
/// (the client hangs up after the first event). Fails unless the usage counts exactly the hashed
/// prompt and, under `ignore_eos`, exactly `max_tokens`.
fn openai_stream(o: &OpenAi, body: &str, r: &mut Req, now: impl Fn() -> f64) -> Result<bool> {
    let lines = http_ok(&o.host, "POST /v1/completions HTTP/1.1", body)?.lines();
    r.http_chunk_gaps_ms = Some(Vec::new());
    r.http_chunks = Some(0);
    r.gen_tokens_known = Some(false);
    let (mut last_chunk, mut usage) = (None, Value::Null);
    for line in lines {
        let line = line?;
        // llama-server (cpp-httplib) sends a stream error as an `error:` line; SSE comments,
        // `event:` lines and the blank separators carry nothing.
        let data = match line.strip_prefix("data:") {
            Some(data) => data.trim_start(),
            None if line.starts_with("error:") => {
                bail!("{}", line.chars().take(200).collect::<String>())
            }
            None => continue,
        };
        if data == "[DONE]" {
            // Without one the turn has no t_first, so its tokens would count with no latency.
            ensure!(
                r.t_first_ms.is_some(),
                "the stream carried no choices[0] event"
            );
            ensure!(!usage.is_null(), "the stream ended without usage");
            let prompt = count(&usage, "/prompt_tokens")?.context("no usage.prompt_tokens")?;
            ensure!(
                prompt == r.prompt_tokens,
                "the server counted {prompt} prompt tokens; the hash names {}",
                r.prompt_tokens
            );
            let generated =
                count(&usage, "/completion_tokens")?.context("no usage.completion_tokens")?;
            ensure!(
                generated <= r.max_tokens,
                "completion_tokens {generated} exceeds requested cap {}",
                r.max_tokens
            );
            ensure!(
                !o.ignore_eos || generated == r.max_tokens,
                "completion_tokens {generated} under ignore_eos, max_tokens {}",
                r.max_tokens
            );
            // A missing count is unknown, never inferred.
            let cached = count(&usage, "/prompt_tokens_details/cached_tokens")?;
            if let Some(c) = cached {
                ensure!(
                    c <= prompt,
                    "the server reported {c} cached of {prompt} prompt tokens"
                );
                ensure!(
                    c == 0 || o.cache_prompt,
                    "{c} cached prompt tokens under cache_prompt false"
                );
            }
            r.gen_tokens = generated;
            r.gen_tokens_known = Some(true);
            r.cached_tokens = cached;
            return Ok(true);
        }
        let v: Value = serde_json::from_str(data).context("OpenAI stream")?;
        if v.get("error").is_some() {
            bail!("{}", data.chars().take(200).collect::<String>());
        }
        if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
            usage = u.clone();
        }
        let Some(c) = v.pointer("/choices/0") else {
            continue;
        };
        let received = now();
        if c["text"].as_str().is_some_and(|t| !t.is_empty()) {
            chunk(r, &mut last_chunk, received);
        }
        if let Some(f) = c["finish_reason"].as_str() {
            r.finish_reason = Some(f.into());
        }
        // Servers hold back partial UTF-8, so the first token is the first event, text or not.
        if r.t_first_ms.is_none() {
            r.t_first_ms = Some(received);
            if r.cancel {
                return Ok(false);
            }
        }
    }
    bail!("the stream ended before data: [DONE]")
}

/// Counts a nonempty HTTP text chunk received at `t`, and its gap after the one before.
fn chunk(r: &mut Req, last: &mut Option<f64>, t: f64) {
    *r.http_chunks.as_mut().unwrap() += 1;
    if let Some(last) = last.replace(t) {
        r.http_chunk_gaps_ms.as_mut().unwrap().push(t - last);
    }
}

/// The count at `pointer` in `v`, None when absent; one beyond u32 is malformed, never narrowed.
fn count(v: &Value, pointer: &str) -> Result<Option<u32>> {
    v.pointer(pointer)
        .and_then(Value::as_u64)
        .map(u32::try_from)
        .transpose()
        .with_context(|| format!("{pointer} exceeds u32"))
}

/// One request on its own connection, closed after the response: the status line, and the body
/// chunk-decoded when the server sends it chunked, else up to the close.
fn http(host: &str, request_line: &str, body: &str) -> Result<(String, Box<dyn BufRead>)> {
    let mut s = TcpStream::connect(host).with_context(|| format!("connecting to {host}"))?;
    // A stalled server fails the request instead of hanging the run; 5 min covers a 16k prefill.
    s.set_read_timeout(Some(Duration::from_secs(300)))?;
    s.set_write_timeout(Some(Duration::from_secs(300)))?;
    let head = format!(
        "{request_line}\r\nHost: {host}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    s.write_all((head + body).as_bytes())?;
    let mut r = BufReader::new(s);
    let mut lines = r.by_ref().lines();
    let status = lines.next().context("no HTTP response")??;
    let mut chunked = false;
    for header in lines {
        let header = header?.to_ascii_lowercase();
        if header.is_empty() {
            break;
        }
        chunked |= header.starts_with("transfer-encoding:") && header.contains("chunked");
    }
    Ok((
        status,
        if chunked {
            Box::new(BufReader::new(Chunked { r, left: Some(0) }))
        } else {
            Box::new(r)
        },
    ))
}

/// [`http`] that fails unless the server answers 200, with the start of what it sent instead.
fn http_ok(host: &str, request_line: &str, body: &str) -> Result<Box<dyn BufRead>> {
    let (status, r) = http(host, request_line, body)?;
    if status.split(' ').nth(1) != Some("200") {
        let body: String = r
            .lines()
            .map_while(Result::ok)
            .collect::<Vec<_>>()
            .join(" ");
        bail!("{status}: {}", body.chars().take(200).collect::<String>());
    }
    Ok(r)
}

/// `path`'s body; None unless the server answers 200, as a server that does not say.
fn get(host: &str, path: &str) -> Result<Option<String>> {
    let (status, mut r) = http(host, &format!("GET {path} HTTP/1.1"), "")?;
    if status.split(' ').nth(1) != Some("200") {
        return Ok(None);
    }
    let mut body = String::new();
    r.read_to_string(&mut body)
        .with_context(|| format!("GET {path}"))?;
    Ok(Some(body))
}

/// An HTTP/1.1 chunked body (RFC 9112 7.1) as the bytes it carries.
struct Chunked<R> {
    r: R,
    /// Bytes left in the current chunk; None after the last.
    left: Option<usize>,
}

impl<R: BufRead> Read for Chunked<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.left == Some(0) {
            // The previous chunk's CRLF is skipped here, before the next size line, so its data
            // was returned without waiting for it.
            let mut line = String::new();
            while line.trim().is_empty() {
                line.clear();
                if self.r.read_line(&mut line)? == 0 {
                    return Err(io::ErrorKind::UnexpectedEof.into());
                }
            }
            let size = line.split(';').next().unwrap_or_default().trim();
            let n = usize::from_str_radix(size, 16).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("chunk size {size:?}: {e}"),
                )
            })?;
            self.left = (n > 0).then_some(n);
        }
        let Some(left) = self.left else {
            return Ok(0);
        };
        let want = buf.len().min(left);
        let n = self.r.read(&mut buf[..want])?;
        if n == 0 && want > 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        self.left = Some(left - n);
        Ok(n)
    }
}

/// Now as `%Y-%m-%dT%H:%M:%SZ`; its first ten bytes name the agent's daily and audit files.
pub fn utc_now() -> String {
    utc(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64)
}

/// Seconds since 1970 as `%Y-%m-%dT%H:%M:%SZ` (Howard Hinnant's `civil_from_days`).
// ponytail: the civil_from_days inside larch-model's private `chat::date_string`; call larch-model's
// once it exposes one (#72).
fn utc(s: i64) -> String {
    let (days, rem) = (s.div_euclid(86_400), s.rem_euclid(86_400));
    let z = days + 719_468;
    let (era, doe) = (z.div_euclid(146_097), z.rem_euclid(146_097));
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_totals_require_complete_known_samples() {
        let mut stats = RingStats {
            engine_samples: 3,
            ..RingStats::default()
        };
        stats.shapes.insert(
            (Phase::Decode.raw(), 8, 0, 0),
            StepTimes {
                steps: 3,
                capture_marker_samples: 3,
                graph_captures: 2,
                capture_step_dur_ns: 37,
                ..StepTimes::default()
            },
        );
        let known = stats.step_profile(0);
        assert_eq!(known["graph_captures"], 2);
        assert_eq!(known["capture_step_dur_ns"], 37);
        assert_eq!(known["shapes"][0]["graph_captures"], 2);
        for dropped in [1, 100] {
            let profile = stats.step_profile(dropped);
            assert!(profile["graph_captures"].is_null());
            assert!(profile["shapes"][0]["capture_step_dur_ns"].is_null());
        }
        stats.incomplete_engine_samples = true;
        assert!(stats.step_profile(0)["graph_captures"].is_null());
        stats.incomplete_engine_samples = false;
        stats
            .shapes
            .values_mut()
            .next()
            .unwrap()
            .capture_marker_samples = 2;
        assert!(stats.step_profile(0)["graph_captures"].is_null());
        assert!(RingStats::default().step_profile(0)["graph_captures"].is_null());
    }

    #[test]
    fn utc_dates_cross_leap_days_and_centuries() {
        assert_eq!(super::utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(super::utc(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(super::utc(1_791_892_800), "2026-10-13T12:00:00Z");
        assert_eq!(super::utc(4_107_542_399), "2100-02-28T23:59:59Z");
    }
}
