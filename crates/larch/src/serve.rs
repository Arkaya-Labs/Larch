//! `larch serve`: the OpenAI endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
//! and Anthropic's (`/v1/messages`) over the facade, `/health` for probes (plan v20-16), and
//! `/api/status` for monitors. One request per connection, one thread per connection, std
//! only: no HTTP or TLS dependency.
//!
//! Agent clients run against it as they would against a gateway or an OpenAI-compatible server
//! (tests/serve-*.json are real requests from Qwen Code and OpenCode, and Claude Code's shape: its
//! field and tool names with filler text). A forced tool
//! call (`tool_choice`) and a JSON answer (`response_format`, Anthropic's `output_config.format`)
//! are held to their schemas by a grammar, so every such reply parses. Claude Code
//! takes it as an LLM gateway (code.claude.com/docs/en/llm-gateway-protocol):
//!
//! ```text
//! export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
//! export ANTHROPIC_AUTH_TOKEN=dummy   # or LARCH_API_KEY's value, when the server has one
//! export ANTHROPIC_DEFAULT_OPUS_MODEL=larch ANTHROPIC_DEFAULT_SONNET_MODEL=larch \
//!        ANTHROPIC_DEFAULT_HAIKU_MODEL=larch   # any name: the server runs the model it loaded
//! export CLAUDE_CODE_ATTRIBUTION_HEADER=0   # keeps the attribution block out of the prompt
//! claude
//! ```
//!
//! Anthropic's own terms for that (code.claude.com/docs/en/llm-gateway): "Anthropic doesn't
//! endorse, maintain, or audit third-party gateway products, and doesn't support routing Claude
//! Code to non-Claude models through any gateway."

use std::{
    collections::VecDeque,
    env,
    fs::File,
    hash::{DefaultHasher, Hash, Hasher},
    io::{self, BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream, ToSocketAddrs},
    path::PathBuf,
    sync::{
        Arc, Mutex, PoisonError,
        mpsc::{self, RecvTimeoutError},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use clap::Args;
use larch_model::{ToolCall as HistoryCall, ToolMessage, gguf::Gguf};
use larch_records::{
    DeviceSample, EngineSample, HealthState, Outcome, ReadFrom, Record, RequestSpan, RingReader,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    EngineError, Finish, GenParams, Larch, LoadOptions, Prompt, Session, SessionOptions, TextEvent,
    TextStream, Usage,
    api::{Grammar, GrammarError, invalid, json_schema_regex},
    bench,
    toolcall::{self, Family, Mode},
};

/// Request heads (the request line and headers) above this size are refused.
const MAX_HEAD: u64 = 64 << 10;
/// Request bodies above this size are refused; an agent's long context with its tools is a few MB.
const MAX_BODY: usize = 16 << 20;
/// How long a read or a write may stall: a client that stops sending its request, or stops reading
/// its answer, would otherwise hold its thread (and its request, until `max_tokens`) forever.
const TIMEOUT: Duration = Duration::from_secs(30);
/// The longest a stream stays silent while its request waits, prefills or holds back a call: agent
/// clients abort a stream idle for long (Claude Code's watchdog), and a ping is bytes on the wire.
const PING: Duration = Duration::from_secs(if cfg!(test) { 1 } else { 5 });
/// `/v1/completions`' `max_tokens` when unset, as OpenAI and vLLM have it.
const COMPLETION_TOKENS: u32 = 16;
/// How far back `/api/status`'s throughput reaches, in buckets of [`BUCKET_MS`].
const HISTORY_MS: f64 = 30.0 * 60_000.0;
const BUCKET_MS: f64 = 10_000.0;
/// `/api/status`'s request totals and latencies cover the last five minutes.
const RECENT_MS: f64 = 5.0 * 60_000.0;
/// The latest requests `/api/status` lists one by one.
const RECENT_ROWS: usize = 50;
/// `larch serve` options.
#[derive(Args)]
pub struct Options {
    /// GGUF to load.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// Explicit reference/debug mode.
    #[arg(long, conflicts_with = "device")]
    pub cpu: bool,
    /// GPU ordinal (default zero).
    #[arg(long)]
    pub device: Option<i32>,
    /// Address to listen on; any but loopback needs LARCH_API_KEY, which every request then sends.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Port to listen on.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// The model name `/v1/models` lists; the GGUF's file name without `.gguf` when unset.
    #[arg(long)]
    pub name: Option<String>,
    /// The tool-call format, for a model whose architecture names none or the wrong one (Mistral
    /// GGUFs say `llama`).
    #[arg(long, value_parser = ["qwen3", "llama3", "mistral"])]
    pub tool_format: Option<String>,
    /// Hold tool calls under `tool_choice` auto to their tool's schema with the call grammar, as
    /// larch's agent does; without it only a forced call and an XML-body call are held. Qwen3's
    /// JSON calls are held from `<tool_call>` on, wherever it shows (a `<think>` block too); Llama 3
    /// and Mistral calls only after `<|python_tag|>` or `[TOOL_CALLS]`, so a bare one stays free. A
    /// tool list too large for the grammar runs without it.
    #[arg(long)]
    pub tool_grammar: bool,
    /// Startup knob assignment, repeatable.
    #[arg(long = "set")]
    pub set: Vec<String>,
}

/// Loads the model, then serves until the process is stopped.
pub fn run(opts: Options, err: &mut impl Write) -> Result<()> {
    let key = env::var("LARCH_API_KEY").ok().filter(|k| !k.is_empty());
    // Checked before the weights load, so a bad --host fails fast under the supervisor.
    let reachable = (opts.host.as_str(), opts.port)
        .to_socket_addrs()
        .with_context(|| format!("--host {}", opts.host))?
        .any(|a| !a.ip().is_loopback());
    ensure!(
        key.is_some() || !reachable,
        "--host {} is reachable from other machines: set LARCH_API_KEY first",
        opts.host
    );
    let name = opts.name.clone().unwrap_or_else(|| {
        let stem = opts.model.file_stem().map(|s| s.to_string_lossy());
        stem.map_or("larch".into(), String::from)
    });
    let device = if opts.cpu {
        larch_kernels::CPU
    } else {
        opts.device.unwrap_or(0)
    };
    // The facade keeps only the compiled template; the XML body is read off its source.
    let file = File::open(&opts.model).with_context(|| opts.model.display().to_string())?;
    let gguf = Gguf::read(BufReader::new(&file), file.metadata()?.len())?;
    let template = gguf.get("tokenizer.chat_template").and_then(|t| t.as_str());
    let xml = template.is_some_and(toolcall::xml_body);
    drop(gguf);
    let mut load = LoadOptions::new(opts.model, device);
    load.on_fault = larch_engine::OnFault::Exit;
    load.knob_overrides = opts
        .set
        .iter()
        .map(|s| crate::knob::assignment(s))
        .collect::<Result<_>>()?;
    // Stateless requests reuse a conversation's KV only through the prefix cache, so serve turns it
    // on at the engine's cap (a quarter of the pool); LARCH_PREFIX_CACHE_BLOCKS still sets it.
    if !env::var("LARCH_PREFIX_CACHE_BLOCKS").is_ok_and(|v| v.parse::<usize>().is_ok()) {
        load.prefix_cache_blocks = usize::MAX;
    }
    // Bound before the weights load, so a taken port fails at once: under the supervisor a bind
    // that failed after the load would reload the whole model on every restart.
    let listener = TcpListener::bind((opts.host.as_str(), opts.port))?;
    let larch = Larch::load(load)?;
    let rejected: Vec<_> = larch
        .rejected_overrides()
        .iter()
        .map(|(knob, value, reason)| {
            let name = larch_records::knob_spec(*knob).map_or("?", |k| k.name.as_str());
            format!("--set {name}={value} rejected: {reason:?}")
        })
        .collect();
    ensure!(rejected.is_empty(), "{}", rejected.join("; "));
    writeln!(err, "{}", crate::supervise::READY_LINE)?;
    writeln!(
        err,
        "serving {name} on http://{}/v1",
        listener.local_addr()?
    )?;
    let server = Server {
        xml,
        tool_grammar: opts.tool_grammar,
        ..Server::new(larch, name, key, opts.tool_format.as_deref())
    };
    serve(&server, listener);
    Ok(())
}

/// A request's tools as [`Server::tools`] settles them: the tools to render, the format their calls
/// are read in, the grammar that holds the answer, and whether that grammar is Auto's, which
/// [`Server::generate`] drops past the size limit.
type Tools = (Vec<Value>, Option<Family>, Option<Grammar>, bool);

/// What every connection reads: the engine, the name `/v1/models` lists and the key requests need.
struct Server {
    larch: Larch,
    name: String,
    key: Option<String>,
    /// The loaded model's tool-call format; `None` refuses requests with tools.
    family: Option<Family>,
    /// The chat template writes calls in the XML body ([`toolcall::xml_body`]).
    xml: bool,
    /// `--tool-grammar`: an Auto call is held to its schema in the JSON body too.
    tool_grammar: bool,
    /// The template thinks by default: it renders differently with thinking off. A forced call
    /// then arms after `</think>`, and a JSON answer renders with thinking off.
    thinks: bool,
    /// Start time in seconds: `created` in responses, and in ids so they differ across restarts.
    started: u64,
    watch: Mutex<Watch>,
    /// The conversations whose KV (and a hybrid model's state) the engine keeps between requests.
    convs: Mutex<Convs>,
}

impl Server {
    fn new(larch: Larch, name: String, key: Option<String>, tool_format: Option<&str>) -> Self {
        let family = toolcall::family_for(larch.deployment().arch.as_str(), tool_format);
        let probe = [ToolMessage {
            role: "user".into(),
            content: "hi".into(),
            ..ToolMessage::default()
        }];
        let render = |thinking| larch.runtime.encode_messages(&probe, &[], thinking).ok();
        let thinks = matches!((render(None), render(Some(false))), (Some(a), Some(b)) if a != b);
        let started = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let watch = Mutex::new(Watch {
            reader: larch.telemetry().reader(ReadFrom::Oldest),
            spans: VecDeque::new(),
            engine: None,
            device: None,
        });
        Self {
            larch,
            name,
            key,
            family,
            xml: false,
            tool_grammar: false,
            thinks,
            started,
            watch,
            convs: Mutex::default(),
        }
    }

    /// A response id for the engine's request `id`.
    fn id(&self, prefix: &str, id: u64) -> String {
        format!("{prefix}{:x}{id}", self.started)
    }

    /// The id of call `i` in request `id`: nine hex digits, the only length Mistral's template takes
    /// back, hashed so ids from other requests and restarts differ.
    fn call_id(&self, id: u64, i: usize) -> String {
        let mut h = DefaultHasher::new();
        (self.started, id, i).hash(&mut h);
        format!("{:09x}", h.finish() >> 28)
    }

    /// The tools to render, the format their calls are read in (none for `tool_choice` none, `mode`
    /// None, or a JSON answer) and the grammar: the JSON answer's `schema`, the call `mode` forces,
    /// or under Auto an XML-body call, which would otherwise come back malformed.
    fn tools(
        &self,
        tools: Vec<Value>,
        mode: Option<Mode>,
        schema: Option<&Value>,
    ) -> Result<Tools, Fail> {
        let forced = mode.as_ref().is_some_and(|m| *m != Mode::Auto);
        if forced && schema.is_some() {
            return Err(bad(
                "a forced tool call and a JSON answer cannot share a reply: one grammar per request",
            ));
        }
        if let Some(Mode::Named(n)) = &mode
            && !tools.iter().any(|t| t["function"]["name"] == n.as_str())
        {
            return Err(bad(format!("tool_choice names {n}, which is not in tools")));
        }
        let (tools, family) = match mode {
            None => (Vec::new(), None),
            Some(_) if forced && tools.is_empty() => {
                return Err(bad("tool_choice forces a call, but there are no tools"));
            }
            Some(_) if tools.is_empty() => (tools, None),
            Some(_) => match self.family {
                Some(family) => (tools, Some(family)),
                None => {
                    return Err(bad(format!(
                        "a {} model has no tool-call format larch reads",
                        self.larch.deployment().arch
                    )));
                }
            },
        };
        let grammar = match (schema, mode, family) {
            (Some(schema), ..) => Some(Grammar {
                trigger: None,
                pattern: json_schema_regex(schema)?,
            }),
            (None, Some(mode), Some(family)) if forced => {
                Some(toolcall::grammar(family, &tools, self.xml, mode)?)
            }
            // Auto keeps XML-body calls well-formed, and JSON-body ones under --tool-grammar, so a
            // schema without a regex leaves the request as it ran before.
            (None, Some(Mode::Auto), Some(family)) if self.xml || self.tool_grammar => {
                match toolcall::grammar(family, &tools, self.xml, Mode::Auto) {
                    Ok(g) => Some(g),
                    Err(e) => {
                        eprintln!("serve: tool calls run without the grammar: {e}");
                        None
                    }
                }
            }
            _ => None,
        };
        let auto = !forced && schema.is_none() && grammar.is_some();
        // An untriggered call would forbid the think block, so it follows it (IR-07b's trigger). A
        // JSON answer stays untriggered, as a triggered grammar re-arms to free text after it: its
        // prompt renders with thinking off instead.
        let grammar = grammar.map(|mut g| {
            if forced && self.thinks {
                g.trigger = Some("</think>".into());
            }
            g
        });
        Ok((tools, family.filter(|_| schema.is_none()), grammar, auto))
    }

    /// Starts `p` on `tokens`; on a model with recurrent state, in the session of the conversation
    /// they continue, so the engine resumes its kept state (see [`Convs`]). `history` is the length
    /// of the chat history the
    /// conversation's next request repeats (`Request::history_end`), 0 when that is not known. An
    /// Auto grammar (`auto`) past the DFA limit drops, as a schema without a regex does, so a client
    /// with a long tool list still gets answers.
    fn generate(
        &self,
        tokens: Vec<u32>,
        history: u32,
        p: GenParams,
        auto: bool,
    ) -> Result<TextStream, EngineError> {
        // ponytail: a pattern past the limit compiles up to it again on every request; remember
        // refused patterns when a long tool list shows in request times.
        let retry = |submit: &dyn Fn(Vec<u32>, GenParams) -> Result<TextStream, EngineError>,
                     tokens: Vec<u32>,
                     p: GenParams| {
            let again = auto.then(|| (tokens.clone(), p.clone()));
            match (submit(tokens, p), again) {
                (Err(EngineError::InvalidRequest(invalid::GRAMMAR_SIZE)), Some((tokens, p))) => {
                    eprintln!(
                        "serve: tool calls run without the grammar: {}",
                        GrammarError::TooLarge
                    );
                    submit(tokens, GenParams { grammar: None, ..p })
                }
                (r, _) => r,
            }
        };
        // A model without recurrent state shares KV across requests through the prefix cache,
        // which stateless requests feed and a kept session's release would withdraw.
        if !self.keeps() {
            return retry(
                &|tokens, p| self.larch.generate(Prompt::Tokens(tokens), p),
                tokens,
                p,
            );
        }
        // Reserved under the lock, submitted outside it: a grammar compile must not hold up
        // every other request's start.
        let session =
            (self.convs.lock().unwrap_or_else(PoisonError::into_inner)).pick(&self.larch, &tokens);
        let end = match history as usize {
            0 => tokens.len(),
            n => n.min(tokens.len()),
        };
        let kept = (session.id(), tokens[..end].to_vec());
        let r = retry(
            &|tokens, p| session.generate_tokens(tokens, history, p),
            tokens,
            p,
        );
        (self.convs.lock().unwrap_or_else(PoisonError::into_inner))
            .done(&session, r.is_ok().then_some(kept));
        r
    }

    /// Whether the model has recurrent state, which only a kept conversation session resumes.
    fn keeps(&self) -> bool {
        self.larch.runtime.engine.state_slots() > 0
    }
}

/// The conversations [`Convs`] keeps an engine session for, least recently used first out. Each
/// keeps its host-side history; its KV and a hybrid model's state stay in the engine's pool and
/// state slots, which the engine reclaims least recently parked first when it needs the room.
const CONVERSATIONS: usize = 16;

/// A conversation's engine session and the prompt history its kept KV holds.
struct Conv {
    session: Arc<Session>,
    /// The session ID `history` was submitted under: a cancelled or failed turn releases its KV
    /// and rotates the ID, and then the history no longer stands for anything kept.
    id: u64,
    /// Its last prompt up to the history end: what this conversation's next request starts with.
    history: Vec<u32>,
    used: u64,
    /// Picked by a request that has not submitted yet.
    reserved: bool,
}

/// Agent clients are stateless: each request resends the whole conversation. The engine shares
/// KV across requests through its prefix cache, but a hybrid model's recurrent state cannot be cut
/// back to a shared prefix, so on such a model it resumes only from a kept session parked where
/// the next prompt goes on (`Request::history_end`). Each request there runs in the session of
/// the conversation whose history its prompt repeats, token for token, and a prompt that repeats
/// none starts a new one. A busy conversation is never picked, so parallel agents keep their own.
#[derive(Default)]
struct Convs {
    list: Vec<Conv>,
    clock: u64,
}

impl Convs {
    /// The session of the conversation `prompt` continues, or of a new one, reserved until
    /// [`Server::generate`] has submitted on it.
    // ponytail: a side request that extends the main conversation's history takes its session, so
    // on a hybrid model the main line's next request re-prefills; fork sessions when a measured
    // agent shows it.
    fn pick(&mut self, larch: &Larch, prompt: &[u32]) -> Arc<Session> {
        let free = |c: &Conv| !c.reserved && !c.session.busy();
        // The longest such history: a shorter one is an older fork of the same conversation.
        let at = (0..self.list.len())
            .filter(|&i| {
                let c = &self.list[i];
                free(c)
                    && c.session.id() == c.id
                    && c.history.len() < prompt.len()
                    && prompt.starts_with(&c.history)
            })
            .max_by_key(|&i| self.list[i].history.len());
        let at = at.unwrap_or_else(|| {
            let shared = |c: &Conv| {
                c.history
                    .iter()
                    .zip(prompt)
                    .take_while(|(a, b)| a == b)
                    .count()
            };
            let (shared, of) = (self.list.iter().filter(|c| free(c)))
                .map(|c| (shared(c), c.history.len()))
                .max()
                .unwrap_or_default();
            eprintln!(
                "serve: a {}-token prompt starts a conversation; the closest kept one shares \
                 {shared} of its {of} tokens",
                prompt.len()
            );
            self.list.push(Conv {
                session: Arc::new(larch.session(SessionOptions::default())),
                id: 0,
                history: Vec::new(),
                used: 0,
                reserved: false,
            });
            self.list.len() - 1
        });
        self.list[at].reserved = true;
        self.list[at].session.clone()
    }

    /// Ends `session`'s reservation: `kept` is the session ID and history a successful submit
    /// left, `None` after a failed one.
    fn done(&mut self, session: &Arc<Session>, kept: Option<(u64, Vec<u32>)>) {
        self.clock += 1;
        let at = (self.list.iter())
            .position(|c| Arc::ptr_eq(&c.session, session))
            .expect("a reserved conversation stays listed");
        let c = &mut self.list[at];
        c.reserved = false;
        match kept {
            Some((id, history)) => (c.id, c.history, c.used) = (id, history, self.clock),
            // A failed submit leaves the KV as it was; a conversation it opened goes again, so a
            // rejected request never retires a kept one.
            None if c.used == 0 => drop(self.list.swap_remove(at)),
            None => {}
        }
        // Past the cap after a burst of parallel requests, the idle ones go, oldest first.
        while self.list.len() > CONVERSATIONS
            && let Some(old) = (0..self.list.len())
                .filter(|&i| !self.list[i].reserved && !self.list[i].session.busy())
                .min_by_key(|&i| self.list[i].used)
        {
            self.list.swap_remove(old);
        }
    }
}

/// Answers every connection on its own thread, for as long as the process runs.
fn serve(server: &Server, listener: TcpListener) {
    // ponytail: one thread per connection, uncapped, so a client trickling its request holds one
    // for up to TIMEOUT a byte; cap them when a deployment binds beyond a trusted network.
    thread::scope(|s| {
        // Read every second, so a busy engine's ring never laps the status's half hour, and every
        // trial gait finishes reaches the ledger (one lasts at least BASELINE's minute).
        s.spawn(|| {
            loop {
                server.poll(&mut server.watch());
                thread::sleep(Duration::from_secs(1));
            }
        });
        for conn in listener.incoming() {
            // A failed accept is that client's connection gone; the next one is unaffected.
            let Ok(conn) = conn else { continue };
            s.spawn(move || {
                // An I/O error is the client leaving: drive cancelled its request, and there is no
                // one left to answer.
                let _ = handle(server, conn);
            });
        }
    });
}

/// Which API's error shape a route answers in.
#[derive(Clone, Copy)]
enum Api {
    OpenAi,
    Anthropic,
}

/// A request that ends in an error response, or a client that left.
enum Fail {
    Status(u16, String),
    Io(io::Error),
}

impl From<io::Error> for Fail {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<GrammarError> for Fail {
    fn from(e: GrammarError) -> Self {
        bad(e.to_string())
    }
}

impl From<EngineError> for Fail {
    fn from(e: EngineError) -> Self {
        let (status, message) = engine_status(e);
        Self::Status(status, message)
    }
}

/// The HTTP status and the client's message for an engine error.
fn engine_status(e: EngineError) -> (u16, String) {
    let status = match e {
        EngineError::InvalidRequest(_) | EngineError::PromptTooLong { .. } => 400,
        EngineError::Overloaded { .. } => 429,
        EngineError::NotReady(_) => 503,
        _ => 500,
    };
    let message = match e {
        // Anthropic's words, which Claude Code compacts its conversation on.
        EngineError::PromptTooLong { tokens, limit } => {
            format!("prompt is too long: {tokens} tokens > {limit} maximum")
        }
        EngineError::InvalidRequest(invalid::CHAT_TEMPLATE) => {
            "the model's chat template refused these messages".into()
        }
        EngineError::InvalidRequest(invalid::EMPTY_PROMPT | invalid::NO_TOKENS) => {
            "the prompt is empty".into()
        }
        EngineError::InvalidRequest(invalid::TOKEN_OUT_OF_VOCAB) => {
            "a prompt token is outside the model's vocabulary".into()
        }
        EngineError::InvalidRequest(invalid::SAMPLING) => "a sampling value is out of range".into(),
        EngineError::InvalidRequest(invalid::GRAMMAR_SIZE) => {
            format!("{}: fewer or smaller schemas", GrammarError::TooLarge)
        }
        EngineError::InvalidRequest(invalid::GRAMMAR_PATTERN) => {
            "the grammar pattern is not a regex".into()
        }
        e => e.to_string(),
    };
    (status, message)
}

fn bad(message: impl Into<String>) -> Fail {
    Fail::Status(400, message.into())
}

/// Why an answer stopped before its end.
enum Cut {
    Engine(EngineError),
    Client(io::Error),
}

impl From<Cut> for Fail {
    fn from(cut: Cut) -> Self {
        match cut {
            Cut::Engine(e) => e.into(),
            Cut::Client(e) => e.into(),
        }
    }
}

/// The request line and the headers larch reads.
#[derive(Default)]
struct Head {
    method: String,
    path: String,
    length: usize,
    chunked: bool,
    expect: bool,
    /// `Authorization: Bearer` (OpenAI) or `x-api-key` (Anthropic).
    key: Option<String>,
    /// `Host` and `Origin`, which a keyless server checks against DNS rebinding and web pages.
    host: Option<String>,
    origin: Option<String>,
}

/// An authority (`name[:port]`, `[v6][:port]`) that names this machine: `localhost` or a loopback IP,
/// with a port of digits if any.
fn loopback_authority(authority: &str) -> bool {
    let (name, port) = match authority.strip_prefix('[') {
        Some(rest) => match rest.split_once(']') {
            Some((name, "")) => (name, None),
            Some((name, tail)) => match tail.strip_prefix(':') {
                Some(port) => (name, Some(port)),
                None => return false,
            },
            None => return false,
        },
        None => match authority.rsplit_once(':') {
            Some((name, port)) => (name, Some(port)),
            None => (authority, None),
        },
    };
    if port.is_some_and(|p| p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit())) {
        return false;
    }
    name.eq_ignore_ascii_case("localhost")
        || name
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Whether a keyless server takes this request. With no key the listener is loopback only
/// (startup refuses anything else), but a browser can still reach it: a web page can POST to
/// 127.0.0.1, and DNS rebinding can read the answer under a foreign host name. Both show in the
/// head, so a `Host` that is not this machine, or an `Origin` from any other site (`null`
/// included), is refused. curl and the SDKs send `Host: 127.0.0.1:<port>` and no `Origin`, so
/// they pass.
fn local_request(head: &Head) -> bool {
    let host_ok = head.host.as_deref().is_none_or(loopback_authority);
    let origin_ok = head.origin.as_deref().is_none_or(|o| {
        o.split_once("://")
            .is_some_and(|(_, authority)| loopback_authority(authority.trim_end_matches('/')))
    });
    host_ok && origin_ok
}

/// Reads the head; `None` when it is over [`MAX_HEAD`] or the client ended it early.
fn read_head(r: &mut impl BufRead) -> io::Result<Option<Head>> {
    let mut r = r.take(MAX_HEAD);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let mut head = Head {
        method: parts.next().unwrap_or_default().into(),
        path: parts
            .next()
            .unwrap_or_default()
            .split('?')
            .next()
            .unwrap_or_default()
            .into(),
        ..Head::default()
    };
    loop {
        let mut header = String::new();
        if r.read_line(&mut header)? == 0 {
            return Ok(None);
        }
        if header.trim().is_empty() {
            return Ok(Some(head));
        }
        let (name, value) = header.split_once(':').unwrap_or((&header, ""));
        let value = value.trim();
        match name.to_ascii_lowercase().as_str() {
            "content-length" => head.length = value.parse().unwrap_or(usize::MAX),
            "transfer-encoding" => head.chunked = value.to_ascii_lowercase().contains("chunked"),
            "expect" => head.expect = value.eq_ignore_ascii_case("100-continue"),
            "authorization" => {
                // The scheme's case is the client's choice (RFC 9110).
                if let Some((scheme, key)) = value.split_once(' ')
                    && scheme.eq_ignore_ascii_case("bearer")
                {
                    head.key = Some(key.trim().into());
                }
            }
            "x-api-key" => head.key = Some(value.into()),
            "host" => head.host = Some(value.into()),
            "origin" => head.origin = Some(value.into()),
            _ => {}
        }
    }
}

/// Equal keys, compared in time that depends only on their lengths.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0, |d, (x, y)| d | (x ^ y)) == 0
}

fn handle(server: &Server, conn: TcpStream) -> io::Result<()> {
    conn.set_read_timeout(Some(TIMEOUT))?;
    conn.set_write_timeout(Some(TIMEOUT))?;
    // Each event goes out as one write, and at once: Nagle would hold small ones back.
    conn.set_nodelay(true)?;
    // `&TcpStream` reads and writes, so no `try_clone` duplicates the socket.
    let mut out = &conn;
    let mut reader = BufReader::new(&conn);
    let Some(head) = read_head(&mut reader)? else {
        return error(
            &mut out,
            Api::OpenAi,
            431,
            "request head too large or incomplete",
        );
    };
    let api = if head.path.starts_with("/v1/messages") {
        Api::Anthropic
    } else {
        Api::OpenAi
    };
    if head.chunked {
        return error(&mut out, api, 411, "send the body with Content-Length");
    }
    if head.length > MAX_BODY {
        return error(&mut out, api, 413, "request body too large");
    }
    let route = (head.method.as_str(), head.path.as_str());
    // Probes hold no data and need no key. The key is checked before the body is read, so a
    // client without it never makes the server hold a body.
    let open = route == ("GET", "/health");
    let refused = match &server.key {
        Some(key) if !open && !head.key.as_deref().is_some_and(|k| same(k, key)) => {
            Some((401, "missing or wrong API key"))
        }
        None if !local_request(&head) => Some((
            403,
            "a keyless server answers this machine only: Host or Origin names another site \
             (set LARCH_API_KEY to serve other clients)",
        )),
        _ => None,
    };
    if let Some((status, message)) = refused {
        // A body already on its way is drained: closing on unread bytes resets the connection,
        // and the client would lose the refusal. One waiting for 100 Continue sends none.
        if !head.expect {
            io::copy(
                &mut reader.by_ref().take(head.length as u64),
                &mut io::sink(),
            )?;
        }
        return error(&mut out, api, status, message);
    }
    if head.expect {
        out.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    }
    // Read before any answer, even a refusal: closing on an unread body resets the connection.
    let mut body = vec![0; head.length];
    reader.read_exact(&mut body)?;
    let result = match route {
        ("GET", "/health") => health(server, &mut out).map_err(Fail::Io),
        ("GET", "/v1/models") => respond(&mut out, 200, &models(server)).map_err(Fail::Io),
        ("GET", "/api/status") => respond(&mut out, 200, &server.status()).map_err(Fail::Io),
        ("POST", "/v1/chat/completions") => chat(server, &body, &conn, &mut out),
        ("POST", "/v1/completions") => completions(server, &body, &conn, &mut out),
        ("POST", "/v1/messages") => messages(server, &body, &conn, &mut out),
        ("POST", "/v1/messages/count_tokens") => count_tokens(server, &body, &mut out),
        (method, path) => Err(Fail::Status(404, format!("no route for {method} {path}"))),
    };
    match result {
        Ok(()) => Ok(()),
        Err(Fail::Status(status, message)) => error(&mut out, api, status, &message),
        Err(Fail::Io(e)) => Err(e),
    }
}

fn health(server: &Server, out: &mut impl Write) -> io::Result<()> {
    let state = server.larch.health().state;
    let up = matches!(state, HealthState::Ready | HealthState::Degraded);
    let status = format!("{state:?}").to_ascii_lowercase();
    respond(out, if up { 200 } else { 503 }, &json!({"status": status}))
}

fn models(server: &Server) -> Value {
    json!({"object": "list", "data": [{
        "id": server.name, "object": "model", "created": server.started, "owned_by": "larch",
        "max_model_len": server.larch.deployment().ctx_max,
    }]})
}

/// The engine's telemetry as `/api/status` reports it: a ring reader of its own, and what it read.
struct Watch {
    reader: RingReader,
    /// Requests that ended in the last [`HISTORY_MS`], oldest first, with their end in Unix ms.
    // ponytail: every span of the half hour is kept; bucket them as they arrive past ~100 req/s.
    spans: VecDeque<(f64, RequestSpan)>,
    engine: Option<EngineSample>,
    device: Option<DeviceSample>,
}

impl Watch {
    /// Takes what the ring gained since the last call, and forgets requests past the history.
    fn drain(&mut self) {
        let mut records = Vec::new();
        self.reader.drain(&mut records, usize::MAX);
        for r in records {
            match r {
                Record::RequestSpan(h, s) => self.spans.push_back((h.t_wall_ns as f64 / 1e6, s)),
                Record::EngineSample(_, e) => self.engine = Some(e),
                Record::DeviceSample(_, d) => self.device = Some(d),
                _ => {}
            }
        }
        let since = now_ms() - HISTORY_MS;
        while self.spans.front().is_some_and(|(end, _)| *end < since) {
            self.spans.pop_front();
        }
    }
}

impl Server {
    fn watch(&self) -> std::sync::MutexGuard<'_, Watch> {
        self.watch.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Brings `w` up to date with the ring's new records.
    fn poll(&self, w: &mut Watch) {
        w.drain();
    }

    /// `GET /api/status`: the deployment, its health and traffic, and the knobs in force.
    fn status(&self) -> Value {
        let now = now_ms();
        let mut watch = self.watch();
        self.poll(&mut watch);
        let reqs: Vec<_> = watch.spans.iter().map(|(end, s)| req(*end, s)).collect();
        let recent: Vec<_> = reqs
            .iter()
            .filter(|r| r.t_done_ms >= now - RECENT_MS)
            .cloned()
            .collect();
        let d = self.larch.deployment();
        let health = self.larch.health();
        let knobs = self.larch.knobs();
        let config = knobs.effective();
        json!({
            "edition": "open source",
            "version": crate::VERSION,
            "name": self.name,
            "started": self.started,
            "health": format!("{:?}", health.state).to_ascii_lowercase(),
            "restarts": health.restarts,
            "auth": self.key.is_some(),
            "tool_format": self.family.map(|f| format!("{f:?}")),
            "model": {
                "file": d.model.as_str(), "arch": d.arch.as_str(), "quant": d.quant.as_str(),
                "params_b": d.params_b, "layers": d.n_layers, "heads": d.n_heads,
                "kv_heads": d.n_kv_heads, "context": d.ctx_max, "vocab": d.vocab,
                "weights_bytes": d.weights_bytes, "backend": format!("{:?}", d.backend()),
                "device": d.device_name.as_str(), "peak_gbps": d.peak_gbps.get(),
                "kv_blocks": d.kv_pool_blocks, "max_seqs": d.fit_max_seqs,
            },
            "load": watch.engine.map(|e| json!({
                "running": e.n_running, "waiting": e.n_waiting, "parked": e.n_parked,
                "kv_util": e.kv_util.get(), "preemptions": e.tot_preemptions,
            })),
            "gpu": watch.device.map(|g| json!({
                "util": g.gpu_util.get(), "temp_c": g.temp_c.get(), "power_w": g.power_w.get(),
                "vram_used": g.vram_used_bytes.get(), "vram_total": g.vram_total_bytes.get(),
            })),
            "traffic": bench::summary(&recent),
            "throughput": {
                "bucket_s": BUCKET_MS / 1000.0, "end_ms": now, "tok_s": throughput(&reqs, now),
            },
            "requests": reqs.iter().rev().take(RECENT_ROWS).map(|r| json!({
                "end_ms": r.t_done_ms, "prompt": r.prompt_tokens, "cached": r.cached_tokens,
                "generated": r.gen_tokens, "ttft_ms": r.ttft_ms, "tpot_ms": r.tpot_ms,
                "e2e_ms": r.e2e_ms, "finish": r.finish_reason, "error": r.error,
            })).collect::<Vec<_>>(),
            "config_rev": config.rev,
            "knobs": knobs.registry().iter().map(|k| json!({
                "id": k.id, "name": k.name.as_str(), "value": config.values[usize::from(k.id)],
                "default": k.default, "min": k.min, "max": k.max,
                "risk": format!("{:?}", k.risk()), "activation": format!("{:?}", k.activation()),
            })).collect::<Vec<_>>(),
        })
    }
}

/// A finished request as the bench's summary reads it, in Unix ms.
fn req(end: f64, s: &RequestSpan) -> bench::Req {
    let ms = |ns: u64| ns as f64 / 1e6;
    let submit = end - ms(s.t_done_ns.saturating_sub(s.t_submit_ns));
    bench::Req {
        prompt_tokens: s.prompt_tokens,
        cached_tokens: Some(s.cached_tokens),
        gen_tokens: s.gen_tokens,
        t_submit_ms: submit,
        t_first_ms: (s.t_first_ns != 0)
            .then(|| submit + ms(s.t_first_ns.saturating_sub(s.t_submit_ns))),
        t_done_ms: end,
        finish_reason: Some(format!("{:?}", s.finish()).to_ascii_lowercase()),
        error: (s.outcome() != Outcome::Ok).then(|| format!("{:?}", s.outcome())),
        // A client that hung up cancelled its request, which is no failure.
        cancel: true,
        ..bench::Req::default()
    }
    .finish()
}

/// Generated tokens per second in each [`BUCKET_MS`] of the history, oldest first; a request's
/// tokens are spread evenly from its first token to its end.
fn throughput(reqs: &[bench::Req], now: f64) -> Vec<f64> {
    let start = now - HISTORY_MS;
    let mut tokens = vec![0.0; (HISTORY_MS / BUCKET_MS) as usize];
    for r in reqs {
        let Some(first) = r.t_first_ms else { continue };
        let span = (r.t_done_ms - first).max(1.0);
        let rate = f64::from(r.gen_tokens) / span;
        let (from, to) = ((r.t_done_ms - span).max(start), r.t_done_ms.min(now));
        if from >= to {
            continue;
        }
        let first_bucket = ((from - start) / BUCKET_MS) as usize;
        let last_bucket = ((to - start) / BUCKET_MS) as usize;
        let buckets = tokens.iter_mut().enumerate();
        for (i, slot) in buckets.take(last_bucket + 1).skip(first_bucket) {
            let lo = (start + i as f64 * BUCKET_MS).max(from);
            let hi = (start + (i + 1) as f64 * BUCKET_MS).min(to);
            *slot += (hi - lo).max(0.0) * rate;
        }
    }
    tokens.iter().map(|t| t * 1000.0 / BUCKET_MS).collect()
}

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64() * 1000.0)
}

fn body<'a, T: Deserialize<'a>>(body: &'a [u8]) -> Result<T, Fail> {
    serde_json::from_slice(body).map_err(|e| bad(format!("invalid request: {e}")))
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    fn strings(stop: Option<Stop>) -> Vec<String> {
        match stop {
            None => Vec::new(),
            Some(Stop::One(s)) => vec![s],
            Some(Stop::Many(v)) => v,
        }
    }
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

/// OpenAI's chat request; unknown fields are ignored, known ones larch cannot honour refused.
#[derive(Deserialize)]
struct ChatIn {
    messages: Vec<MessageIn>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop: Option<Stop>,
    #[serde(default)]
    tools: Vec<Value>,
    tool_choice: Option<Value>,
    n: Option<u32>,
    #[serde(default)]
    logprobs: bool,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    response_format: Option<Value>,
}

#[derive(Deserialize)]
struct MessageIn {
    role: String,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    tool_calls: Vec<CallIn>,
    #[serde(default)]
    tool_call_id: String,
}

#[derive(Deserialize)]
struct CallIn {
    #[serde(default)]
    id: String,
    function: FunctionIn,
}

#[derive(Deserialize)]
struct FunctionIn {
    name: String,
    /// JSON text, as OpenAI sends it; the template parses it.
    #[serde(default)]
    arguments: Value,
}

/// A string, or text parts joined by newlines; null is empty. Any other part is refused rather
/// than dropped, so a request never silently loses an image.
fn text_of(content: &Value) -> Result<String, Fail> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => parts
            .iter()
            .map(|p| match (p["type"].as_str(), p["text"].as_str()) {
                (Some("text"), Some(text)) => Ok(text),
                _ => Err(bad(format!(
                    "only text content is supported, not {}",
                    p["type"]
                ))),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|texts| texts.join("\n")),
        _ => Err(bad("content must be a string or an array of parts")),
    }
}

fn openai_message(m: MessageIn) -> Result<ToolMessage, Fail> {
    let role = match m.role.as_str() {
        "developer" => "system",
        role @ ("system" | "user" | "assistant" | "tool") => role,
        role => return Err(bad(format!("unknown role {role}"))),
    };
    Ok(ToolMessage {
        role: role.into(),
        content: text_of(&m.content)?,
        tool_calls: m
            .tool_calls
            .into_iter()
            .map(|c| HistoryCall {
                name: c.function.name,
                arguments: c.function.arguments,
                id: c.id,
            })
            .collect(),
        tool_call_id: m.tool_call_id,
    })
}

/// Every system message as one leading system message, their texts in order: chat templates
/// take a system prompt first only, and Claude Code sends some mid-conversation.
// ponytail: a system entry mid-conversation moves into the leading text, so each new one changes
// the prompt from its start and misses the prefix cache; carry it in the next user turn instead
// when long agent sessions show full re-prefills.
fn fold_system(messages: Vec<ToolMessage>) -> Vec<ToolMessage> {
    let (system, rest): (Vec<_>, Vec<_>) = messages.into_iter().partition(|m| m.role == "system");
    // Kept even when empty: a template given none may add its own (Qwen2.5's names the model).
    let any = !system.is_empty();
    let texts: Vec<_> = system
        .into_iter()
        .map(|m| m.content)
        .filter(|t| !t.is_empty())
        .collect();
    let system = any.then(|| ToolMessage {
        role: "system".into(),
        content: texts.join("\n\n"),
        ..ToolMessage::default()
    });
    system.into_iter().chain(rest).collect()
}

/// A chat prompt's tokens, its system entries folded ([`fold_system`]), and, when `history`, the
/// length of its chat history, which the conversation's next request repeats
/// ([`Server::generate`]); 0 otherwise, sparing the two renders that length takes.
fn render(
    server: &Server,
    messages: Vec<ToolMessage>,
    tools: &[Value],
    thinking: Option<bool>,
    history: bool,
) -> Result<(Vec<u32>, u32), Fail> {
    let messages = fold_system(messages);
    let runtime = &server.larch.runtime;
    let tokens = runtime.encode_messages(&messages, tools, thinking)?;
    let history = match history {
        true => runtime.history_len(&messages, tools, thinking, &tokens),
        false => 0,
    };
    Ok((tokens, history))
}

/// `max_tokens` as GenParams takes it: 0, unset or past what the context holds after the
/// `prompt`, runs to the end of the context, where such a cap would stop the answer anyway. Agent
/// clients ask for 32000 whatever the model, and the engine refuses a cap the context cannot hold.
fn max_tokens(server: &Server, n: Option<u32>, prompt: usize) -> Result<u32, Fail> {
    let ctx = u64::from(server.larch.deployment().ctx_max);
    match n {
        Some(0) => Err(bad("max_tokens must be at least 1")),
        Some(n) if prompt as u64 + u64::from(n) <= ctx => Ok(n),
        _ => Ok(0),
    }
}

/// The sampling options both OpenAI endpoints refuse rather than ignore.
fn openai_refusals(n: Option<u32>, penalties: [Option<f32>; 2]) -> Result<(), Fail> {
    if n.is_some_and(|n| n != 1) {
        return Err(bad("n must be 1"));
    }
    if penalties.iter().any(|p| p.is_some_and(|p| p != 0.0)) {
        return Err(bad(
            "frequency and presence penalties are not supported; larch has repeat_penalty",
        ));
    }
    Ok(())
}

fn chat(server: &Server, raw: &[u8], conn: &TcpStream, out: &mut &TcpStream) -> Result<(), Fail> {
    let req: ChatIn = body(raw)?;
    openai_refusals(req.n, [req.frequency_penalty, req.presence_penalty])?;
    if req.logprobs {
        return Err(bad("logprobs are not supported yet"));
    }
    let schema = match &req.response_format {
        None => None,
        Some(f) => match f["type"].as_str() {
            Some("text") => None,
            // ponytail: values nest at most json_schema_regex's ANY_DEPTH; deeper objects need a
            // recursive grammar, when a client's JSON answers show grammar_forced.
            Some("json_object") => Some(json!({"type": "object"})),
            Some("json_schema") => Some(f["json_schema"]["schema"].clone()),
            _ => return Err(bad(format!("response_format {f} is not supported"))),
        },
    };
    let mode = match &req.tool_choice {
        None => Some(Mode::Auto),
        Some(c) => match (c.as_str(), c["function"]["name"].as_str()) {
            (Some("auto"), _) => Some(Mode::Auto),
            (Some("none"), _) => None,
            (Some("required"), _) => Some(Mode::Required),
            (None, Some(name)) if c["type"] == "function" => Some(Mode::Named(name.into())),
            _ => return Err(bad(format!("tool_choice {c} is not supported"))),
        },
    };
    let (tools, family, grammar, auto) = server.tools(req.tools, mode, schema.as_ref())?;
    let messages = req
        .messages
        .into_iter()
        .map(openai_message)
        .collect::<Result<Vec<_>, _>>()?;
    let thinking = (schema.is_some() && server.thinks).then_some(false);
    let (tokens, history) = render(server, messages, &tools, thinking, server.keeps())?;
    let p = GenParams {
        max_tokens: max_tokens(
            server,
            req.max_completion_tokens.or(req.max_tokens),
            tokens.len(),
        )?,
        temperature: req.temperature,
        top_p: req.top_p,
        seed: req.seed,
        stop: Stop::strings(req.stop),
        grammar,
        ..GenParams::default()
    };
    let stream = server.generate(tokens, history, p, auto)?;
    let request = stream.request_id();
    let id = server.id("chatcmpl-", request);
    let (model, created) = (&server.name, server.started);
    let finish = |a: &Answer| match a.finish {
        _ if !a.calls.is_empty() => "tool_calls",
        Finish::Length => "length",
        _ => "stop",
    };
    if !req.stream {
        let a = drive(stream, family, &tools, |_| alive(conn))?;
        let content = (a.calls.is_empty() || !a.text.trim().is_empty()).then_some(&a.text);
        let mut message = json!({"role": "assistant", "content": content});
        if !a.calls.is_empty() {
            message["tool_calls"] = openai_calls(server, request, &a.calls).into();
        }
        let reply = json!({
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish(&a)}],
            "usage": openai_usage(a.usage),
        });
        return Ok(respond(out, 200, &reply)?);
    }
    let chunk = |delta: Value, finish: Option<&str>| {
        json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })
    };
    start_events(out)?;
    data(
        out,
        &chunk(json!({"role": "assistant", "content": ""}), None),
    )?;
    let result = drive(stream, family, &tools, |piece| match piece {
        // A closed socket takes the first write after it, so the client is asked first.
        None => alive(conn).and_then(|()| ping(out)),
        // Text held back as a possible call sends nothing, so the client is asked directly.
        Some("") => alive(conn),
        Some(piece) => data(out, &chunk(json!({"content": piece}), None)),
    });
    match result {
        Ok(a) => {
            if !a.calls.is_empty() {
                let calls = openai_calls(server, request, &a.calls);
                data(out, &chunk(json!({"tool_calls": calls}), None))?;
            }
            data(out, &chunk(json!({}), Some(finish(&a))))?;
            if req.stream_options.is_some_and(|o| o.include_usage) {
                let mut last = chunk(Value::Null, None);
                last["choices"] = json!([]);
                last["usage"] = openai_usage(a.usage);
                data(out, &last)?;
            }
        }
        Err(Cut::Engine(e)) => data(out, &stream_error(Api::OpenAi, e))?,
        Err(Cut::Client(e)) => return Err(e.into()),
    }
    Ok(out.write_all(b"data: [DONE]\n\n")?)
}

/// OpenAI's legacy completion request: a raw prompt, no chat template.
#[derive(Deserialize)]
struct CompletionIn {
    prompt: Value,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop: Option<Stop>,
    n: Option<u32>,
    logprobs: Option<Value>,
    #[serde(default)]
    echo: bool,
    suffix: Option<Value>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
}

/// A string, one string in an array, or token ids.
fn prompt_of(prompt: Value) -> Result<Prompt, Fail> {
    match prompt {
        Value::String(text) => Ok(Prompt::Text(text)),
        Value::Array(mut items) if items.len() == 1 && items[0].is_string() => {
            prompt_of(items.remove(0))
        }
        Value::Array(items) => items
            .iter()
            .map(|t| t.as_u64().and_then(|t| u32::try_from(t).ok()))
            .collect::<Option<Vec<_>>>()
            .map(Prompt::Tokens)
            .ok_or_else(|| bad("prompt must be a string or an array of token ids")),
        _ => Err(bad("prompt must be a string or an array of token ids")),
    }
}

fn completions(
    server: &Server,
    raw: &[u8],
    conn: &TcpStream,
    out: &mut &TcpStream,
) -> Result<(), Fail> {
    let req: CompletionIn = body(raw)?;
    openai_refusals(req.n, [req.frequency_penalty, req.presence_penalty])?;
    if req
        .logprobs
        .as_ref()
        .is_some_and(|l| !l.is_null() && *l != 0)
    {
        return Err(bad("logprobs are not supported yet"));
    }
    if req.echo || req.suffix.as_ref().is_some_and(|s| !s.is_null()) {
        return Err(bad("echo and suffix are not supported"));
    }
    let tokens = server
        .larch
        .runtime
        .encode_prompt(prompt_of(req.prompt)?, &[], None)?;
    let n = Some(req.max_tokens.unwrap_or(COMPLETION_TOKENS));
    let p = GenParams {
        max_tokens: max_tokens(server, n, tokens.len())?,
        temperature: req.temperature,
        top_p: req.top_p,
        seed: req.seed,
        stop: Stop::strings(req.stop),
        ..GenParams::default()
    };
    let stream = server.larch.generate(Prompt::Tokens(tokens), p)?;
    let id = server.id("cmpl-", stream.request_id());
    let (model, created) = (&server.name, server.started);
    let choice = |text: &str, finish: Option<Finish>| {
        let finish = finish.map(|f| {
            if f == Finish::Length {
                "length"
            } else {
                "stop"
            }
        });
        json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": finish}],
        })
    };
    if !req.stream {
        let a = drive(stream, None, &[], |_| alive(conn))?;
        let mut reply = choice(&a.text, Some(a.finish));
        reply["usage"] = openai_usage(a.usage);
        return Ok(respond(out, 200, &reply)?);
    }
    start_events(out)?;
    let result = drive(stream, None, &[], |piece| match piece {
        None => alive(conn).and_then(|()| ping(out)),
        Some("") => alive(conn),
        Some(piece) => data(out, &choice(piece, None)),
    });
    match result {
        Ok(a) => {
            data(out, &choice("", Some(a.finish)))?;
            if req.stream_options.is_some_and(|o| o.include_usage) {
                let mut last = choice("", None);
                last["choices"] = json!([]);
                last["usage"] = openai_usage(a.usage);
                data(out, &last)?;
            }
        }
        Err(Cut::Engine(e)) => data(out, &stream_error(Api::OpenAi, e))?,
        Err(Cut::Client(e)) => return Err(e.into()),
    }
    Ok(out.write_all(b"data: [DONE]\n\n")?)
}

fn openai_calls(server: &Server, request: u64, calls: &[toolcall::ToolCall]) -> Vec<Value> {
    let calls = calls.iter().enumerate();
    calls
        .map(|(i, c)| {
            json!({
                "index": i, "id": server.call_id(request, i), "type": "function",
                "function": {"name": c.name, "arguments": c.args.to_string()},
            })
        })
        .collect()
}

fn openai_usage(u: Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": u.cached_tokens},
    })
}

/// An engine error as the event that ends a stream.
fn stream_error(api: Api, e: EngineError) -> Value {
    let (status, message) = engine_status(e);
    error_body(api, status, &message)
}

/// Anthropic's Messages request; unknown fields (`thinking`, `metadata`, `context_management`,
/// `cache_control`) are ignored, since none changes what the answer means.
#[derive(Deserialize)]
struct MessagesIn {
    #[serde(flatten)]
    prompt: PromptIn,
    max_tokens: u32,
    /// Its `effort` is ignored; its `format` (or the older `output_format`) asks for a JSON answer:
    /// `{"type": "json_schema", "schema": ...}`.
    output_config: Option<Value>,
    output_format: Option<Value>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    #[serde(default)]
    stream: bool,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
}

/// What a Messages request renders into its prompt, which is all `count_tokens` takes.
#[derive(Deserialize)]
struct PromptIn {
    messages: Vec<AnthropicIn>,
    system: Option<Value>,
    #[serde(default)]
    tools: Vec<ToolIn>,
    tool_choice: Option<Value>,
}

#[derive(Deserialize)]
struct AnthropicIn {
    role: String,
    #[serde(default)]
    content: Value,
}

#[derive(Deserialize)]
struct ToolIn {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    input_schema: Value,
    /// Absent or `custom` for a client tool; anything else is one of Anthropic's server tools.
    #[serde(rename = "type")]
    kind: Option<String>,
}

/// Appends one Anthropic message: a user message's tool results become tool messages ahead of its
/// text, since they answer the assistant message before it.
fn anthropic_message(m: AnthropicIn, into: &mut Vec<ToolMessage>) -> Result<(), Fail> {
    // Claude Code's mid-conversation reminders, which `fold_system` moves into the system text.
    if m.role == "system" {
        into.push(ToolMessage {
            role: m.role,
            content: text_of(&m.content)?,
            ..ToolMessage::default()
        });
        return Ok(());
    }
    let blocks = match m.content {
        Value::String(text) => vec![json!({"type": "text", "text": text})],
        Value::Array(blocks) => blocks,
        _ => return Err(bad("content must be a string or an array of blocks")),
    };
    let (mut text, mut calls) = (Vec::new(), Vec::new());
    let field = |b: &Value, key: &str| b[key].as_str().unwrap_or_default().to_owned();
    for b in &blocks {
        match (m.role.as_str(), b["type"].as_str()) {
            (_, Some("text")) => text.push(field(b, "text")),
            ("assistant", Some("tool_use")) => calls.push(HistoryCall {
                name: field(b, "name"),
                arguments: b["input"].clone(),
                id: field(b, "id"),
            }),
            ("user", Some("tool_result")) => into.push(ToolMessage {
                role: "tool".into(),
                content: text_of(&b["content"])?,
                tool_call_id: field(b, "tool_use_id"),
                ..ToolMessage::default()
            }),
            // The model's own reasoning; templates drop it from history.
            ("assistant", Some("thinking" | "redacted_thinking")) => {}
            (role, kind) => {
                return Err(bad(format!(
                    "{} blocks are not supported in {role} messages",
                    kind.unwrap_or("untyped")
                )));
            }
        }
    }
    let message = |role: &str| ToolMessage {
        role: role.into(),
        content: text.join("\n"),
        tool_calls: calls,
        ..ToolMessage::default()
    };
    match m.role.as_str() {
        "user" if text.is_empty() => {}
        role @ ("user" | "assistant") => into.push(message(role)),
        role => return Err(bad(format!("unknown role {role}"))),
    }
    Ok(())
}

/// The prompt's tokens through the model's chat template, the length of its chat history
/// ([`Server::generate`], 0 unless `keep`) and its [`Tools`], held to a JSON `schema` when there
/// is one.
fn anthropic_prompt(
    server: &Server,
    req: PromptIn,
    schema: Option<&Value>,
    keep: bool,
) -> Result<(Vec<u32>, u32, Tools), Fail> {
    let tools = req
        .tools
        .into_iter()
        .map(|t| match t.kind.as_deref() {
            None | Some("custom") => Ok(json!({"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.input_schema,
            }})),
            Some(kind) => Err(bad(format!("server tool {kind} is not supported"))),
        })
        .collect::<Result<_, _>>()?;
    let choice = req.tool_choice.unwrap_or_else(|| json!({"type": "auto"}));
    let mode = match (choice["type"].as_str(), choice["name"].as_str()) {
        (Some("auto") | None, _) => Some(Mode::Auto),
        (Some("none"), _) => None,
        (Some("any"), _) => Some(Mode::Required),
        (Some("tool"), Some(name)) => Some(Mode::Named(name.into())),
        _ => return Err(bad(format!("tool_choice {choice} is not supported"))),
    };
    let (tools, family, grammar, auto) = server.tools(tools, mode, schema)?;
    let mut history = Vec::new();
    if let Some(system) = &req.system {
        // Claude Code sends its billing header as a system block, which the Anthropic API reads
        // as a header. Its value differs between conversations, so as prompt text it would end
        // the prefix they share a few tokens in.
        // ponytail: one client's header, in the array form it sends; generalise when another
        // client's per-request system text shows in cache misses.
        // Only the header line goes: text after it in the same block is a real instruction.
        let unheader = |p: &Value| match p["text"].as_str() {
            Some(t) if t.starts_with("x-anthropic-billing-header:") => {
                let rest = t.split_once('\n').map_or("", |(_, rest)| rest);
                (!rest.is_empty()).then(|| {
                    let mut p = p.clone();
                    p["text"] = rest.into();
                    p
                })
            }
            _ => Some(p.clone()),
        };
        let system = match system {
            Value::Array(parts) => Value::Array(parts.iter().filter_map(unheader).collect()),
            system => system.clone(),
        };
        history.push(ToolMessage {
            role: "system".into(),
            content: text_of(&system)?,
            ..ToolMessage::default()
        });
    }
    for m in req.messages {
        anthropic_message(m, &mut history)?;
    }
    let thinking = (schema.is_some() && server.thinks).then_some(false);
    let (tokens, end) = render(server, history, &tools, thinking, keep)?;
    Ok((tokens, end, (tools, family, grammar, auto)))
}

/// `POST /v1/messages/count_tokens`: the loaded tokenizer's count, as `/v1/messages` prefills it.
fn count_tokens(server: &Server, raw: &[u8], out: &mut impl Write) -> Result<(), Fail> {
    let (tokens, ..) = anthropic_prompt(server, body(raw)?, None, false)?;
    Ok(respond(out, 200, &json!({"input_tokens": tokens.len()}))?)
}

fn messages(
    server: &Server,
    raw: &[u8],
    conn: &TcpStream,
    out: &mut &TcpStream,
) -> Result<(), Fail> {
    let req: MessagesIn = body(raw)?;
    let format = (req.output_config.as_ref().map(|c| &c["format"]))
        .filter(|f| !f.is_null())
        .or(req.output_format.as_ref().filter(|f| !f.is_null()));
    let schema = match format {
        None => None,
        Some(f) if f["type"] == "json_schema" => Some(&f["schema"]),
        Some(f) => return Err(bad(format!("output format {f} is not supported"))),
    };
    let (tokens, history, (tools, family, grammar, auto)) =
        anthropic_prompt(server, req.prompt, schema, server.keeps())?;
    let prompt_tokens = tokens.len();
    // 0 here is a cap the context could not hold: max_tokens refuses a 0 from the client.
    let cap = max_tokens(server, Some(req.max_tokens), prompt_tokens)?;
    // ponytail: `thinking` is ignored and answers carry no thinking blocks; map it to
    // GenParams::thinking_budget when a thinking model serves Anthropic clients.
    let p = GenParams {
        max_tokens: cap,
        temperature: req.temperature,
        top_p: req.top_p,
        // The engine samples from its 64 best candidates and refuses more; Anthropic's API takes
        // any top_k, and one above 64 samples exactly as 64 does.
        top_k: req.top_k.map(|k| k.min(64)),
        stop: req.stop_sequences.clone(),
        grammar,
        ..GenParams::default()
    };
    let stream = server.generate(tokens, history, p, auto)?;
    let request = stream.request_id();
    let id = server.id("msg_", request);
    let stop_reason = |a: &Answer| stop_reason(a, cap == 0);
    // ponytail: the engine reports a stop string matched, not which one, so it is named only when
    // there was one; carry the matched string in Finish when a client needs it with several.
    let stop_sequence = |a: &Answer| match req.stop_sequences.as_slice() {
        [one] if a.finish == Finish::StopSeq => Some(one.clone()),
        _ => None,
    };
    let tool_use = |i: usize, c: &toolcall::ToolCall, input: &Value| json!({"type": "tool_use", "id": server.call_id(request, i), "name": c.name, "input": input});
    if !req.stream {
        let a = drive(stream, family, &tools, |_| alive(conn))?;
        let text = (!a.text.trim().is_empty() || a.calls.is_empty())
            .then(|| json!({"type": "text", "text": a.text}));
        let calls = a
            .calls
            .iter()
            .enumerate()
            .map(|(i, c)| tool_use(i, c, &c.args));
        let reply = json!({
            "id": id, "type": "message", "role": "assistant", "model": server.name,
            "content": text.into_iter().chain(calls).collect::<Vec<_>>(),
            "stop_reason": stop_reason(&a), "stop_sequence": stop_sequence(&a),
            "usage": anthropic_usage(a.usage),
        });
        return Ok(respond(out, 200, &reply)?);
    }
    start_events(out)?;
    // The cache's share is known only at the end, so the start counts the whole prompt as input;
    // message_delta carries the final split.
    let start = json!({"type": "message_start", "message": {
        "id": id, "type": "message", "role": "assistant", "model": server.name, "content": [],
        "stop_reason": null, "stop_sequence": null,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
    }});
    event(out, &start)?;
    let mut open = false;
    let result = drive(stream, family, &tools, |piece| {
        let piece = match piece {
            None => return alive(conn).and_then(|()| event(out, &json!({"type": "ping"}))),
            // Text held back as a possible call sends nothing, so the client is asked directly.
            Some("") => return alive(conn),
            Some(piece) => piece,
        };
        if !open {
            let block = json!({"type": "text", "text": ""});
            event(
                out,
                &json!({"type": "content_block_start", "index": 0, "content_block": block}),
            )?;
            open = true;
        }
        let delta = json!({"type": "text_delta", "text": piece});
        event(
            out,
            &json!({"type": "content_block_delta", "index": 0, "delta": delta}),
        )
    });
    match result {
        Ok(a) => {
            let first = usize::from(open);
            if open {
                event(out, &json!({"type": "content_block_stop", "index": 0}))?;
            }
            for (i, c) in a.calls.iter().enumerate() {
                let index = first + i;
                let block = tool_use(i, c, &json!({}));
                event(
                    out,
                    &json!({"type": "content_block_start", "index": index, "content_block": block}),
                )?;
                let delta = json!({"type": "input_json_delta", "partial_json": c.args.to_string()});
                event(
                    out,
                    &json!({"type": "content_block_delta", "index": index, "delta": delta}),
                )?;
                event(out, &json!({"type": "content_block_stop", "index": index}))?;
            }
            let delta = json!({"stop_reason": stop_reason(&a), "stop_sequence": stop_sequence(&a)});
            let usage = anthropic_usage(a.usage);
            event(
                out,
                &json!({"type": "message_delta", "delta": delta, "usage": usage}),
            )?;
            event(out, &json!({"type": "message_stop"}))?;
        }
        Err(Cut::Engine(e)) => event(out, &stream_error(Api::Anthropic, e))?,
        Err(Cut::Client(e)) => return Err(e.into()),
    }
    Ok(())
}

/// Why an Anthropic answer stopped. With `clamped` (the client's cap was past the context) the
/// context ended a long answer: told its cap was reached, Claude Code would ask for more.
fn stop_reason(a: &Answer, clamped: bool) -> &'static str {
    match a.finish {
        _ if !a.calls.is_empty() => "tool_use",
        Finish::Length if clamped => "model_context_window_exceeded",
        Finish::Length => "max_tokens",
        Finish::StopSeq => "stop_sequence",
        _ => "end_turn",
    }
}

/// Anthropic counts cache reads apart from `input_tokens`.
fn anthropic_usage(u: Usage) -> Value {
    json!({
        "input_tokens": u.prompt_tokens.saturating_sub(u.cached_tokens),
        "output_tokens": u.completion_tokens,
        "cache_read_input_tokens": u.cached_tokens,
        "cache_creation_input_tokens": 0,
    })
}

/// A finished answer: its text up to its calls, the calls the parser read, and the engine's finish
/// and counts.
struct Answer {
    text: String,
    calls: Vec<toolcall::ToolCall>,
    finish: Finish,
    usage: Usage,
}

/// Runs `stream` to its end and returns the answer, handing `piece` each run of text as it becomes
/// safe to send: with `family` set, the text before anything that may be a call, each call typed to
/// its spec in `tools` ([`toolcall::coerce`]). A piece may be
/// empty, so a caller that streams nothing still notices the client leave, and is `None` after
/// [`PING`] with nothing sent, so a stream can ping; an error from `piece` cancels the request.
fn drive(
    stream: TextStream,
    family: Option<Family>,
    tools: &[Value],
    mut piece: impl FnMut(Option<&str>) -> io::Result<()>,
) -> Result<Answer, Cut> {
    let (cancel, request) = (stream.handle.canceller(), stream.request_id());
    // The stream blocks until the engine's next token, so it is read on a thread of its own and
    // every write stays on this one. It is never joined, so a client that leaves frees its
    // connection at once while the reader waits out the engine's step.
    // ponytail: a thread per answer, since TextStream has no recv_timeout (text.rs, F4's); read
    // the stream here once it gains one.
    let (tx, events) = mpsc::channel();
    thread::spawn(move || {
        for event in stream {
            if tx.send(event).is_err() {
                break;
            }
        }
    });
    let answer = answer(&events, request, family, tools, &mut piece);
    // At once: the reader holds the stream, and would drop it only at the engine's next event,
    // after a whole prefill.
    if answer.is_err() {
        cancel.cancel();
    }
    answer
}

/// [`drive`]'s loop over the events its reader thread forwards.
fn answer(
    events: &mpsc::Receiver<TextEvent>,
    request: u64,
    family: Option<Family>,
    tools: &[Value],
    piece: &mut impl FnMut(Option<&str>) -> io::Result<()>,
) -> Result<Answer, Cut> {
    let (mut text, mut sent, mut quiet) = (String::new(), 0, Instant::now());
    loop {
        // Checked per event too: text held back as a possible call arrives but sends nothing.
        if quiet.elapsed() >= PING {
            piece(None).map_err(Cut::Client)?;
            quiet = Instant::now();
        }
        let event = match events.recv_timeout(PING.saturating_sub(quiet.elapsed())) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => continue,
            // The engine thread is gone.
            Err(RecvTimeoutError::Disconnected) => {
                return Err(Cut::Engine(EngineError::NotReady(HealthState::Failed)));
            }
        };
        match event {
            TextEvent::Prefilled { .. } | TextEvent::Thinking { .. } => {}
            TextEvent::Text { delta, .. } => {
                text.push_str(&delta);
                let safe = family.map_or(text.len(), |f| hold(f, &text, sent));
                piece(Some(&text[sent..safe])).map_err(Cut::Client)?;
                if safe > sent {
                    quiet = Instant::now();
                }
                sent = safe;
            }
            TextEvent::Done {
                finish,
                usage,
                timings,
                ..
            } => {
                // Kept out of the API body: picks forced past the model's 64 candidates, the number
                // a grammar's operator watches.
                if timings.grammar_forced > 0 {
                    eprintln!(
                        "serve: request {request}: grammar forced {} tokens",
                        timings.grammar_forced
                    );
                }
                let (end, mut calls) = family.map_or((text.len(), Vec::new()), |f| split(f, &text));
                for c in &mut calls {
                    toolcall::coerce(c, tools);
                }
                let end = end.max(sent);
                piece(Some(&text[sent..end])).map_err(Cut::Client)?;
                text.truncate(end);
                return Ok(Answer {
                    text,
                    calls,
                    finish,
                    usage,
                });
            }
            TextEvent::Error(e) => return Err(Cut::Engine(e)),
        }
    }
}

/// Where text that may start a call begins: a marker, a marker cut off by the end of the text so
/// far, or for Llama and Mistral a line that starts like JSON (their calls may come bare,
/// `toolcall::parse`). Nothing from there streams until the parser reads the whole answer. The
/// text before `sent` already streamed, so neither starts there and only the rest is scanned.
fn hold(family: Family, text: &str, sent: usize) -> usize {
    let rest = &text[sent..];
    // Every opening the parser reads as a call: Qwen3's `<function-call>` streamed as text, then
    // came again as the call.
    let mut at = rest.len();
    for (open, _) in family.wrappers() {
        let cut = (1..open.len()).rev().find(|&k| rest.ends_with(&open[..k]));
        at = at.min(rest.find(open).unwrap_or(rest.len()));
        at = at.min(cut.map_or(rest.len(), |k| rest.len() - k));
    }
    if family != Family::Qwen3 {
        // The first line start in `rest` is its own start only when a line ended right before it.
        let skip = usize::from(sent > 0 && !text[..sent].ends_with('\n'));
        if let Some(line) = toolcall::line_starts(rest).skip(skip).find(|&i| {
            let line = rest[i..].trim_start();
            line.is_empty() || line.starts_with(['{', '['])
        }) {
            at = at.min(line);
        }
    }
    sent + at
}

/// The byte where the answer's text ends, and its calls: none, and all of it text, when the
/// parser finds no call or a broken one. Only a Qwen model takes several calls in one turn
/// (`Session::set_several_tool_calls`); the others keep their first.
fn split(family: Family, text: &str) -> (usize, Vec<toolcall::ToolCall>) {
    let parsed = toolcall::parse(family, text, false);
    let Some(first) = parsed.call else {
        return (text.len(), Vec::new());
    };
    // The parser trims; its text before the call starts at the answer's first visible byte (the
    // stream withholds think blocks, so none comes first).
    let end = text.len() - text.trim_start().len() + parsed.text_before.len();
    (end, [first].into_iter().chain(parsed.later).collect())
}

/// Err once the client has hung up, so a non-streamed answer stops when nobody waits for it.
fn alive(conn: &TcpStream) -> io::Result<()> {
    let mut byte = [0];
    conn.set_nonblocking(true)?;
    let peek = conn.peek(&mut byte);
    conn.set_nonblocking(false)?;
    match peek {
        Ok(0) => Err(io::ErrorKind::ConnectionAborted.into()),
        Err(e) if e.kind() != io::ErrorKind::WouldBlock => Err(e),
        _ => Ok(()),
    }
}

fn error_body(api: Api, status: u16, message: &str) -> Value {
    match api {
        Api::OpenAi => {
            let kind = if status < 500 {
                "invalid_request_error"
            } else {
                "server_error"
            };
            json!({"error": {"message": message, "type": kind, "param": null, "code": null}})
        }
        Api::Anthropic => {
            let kind = match status {
                400 | 411 | 431 => "invalid_request_error",
                401 => "authentication_error",
                403 => "permission_error",
                404 => "not_found_error",
                413 => "request_too_large",
                429 => "rate_limit_error",
                503 => "overloaded_error",
                _ => "api_error",
            };
            json!({"type": "error", "error": {"type": kind, "message": message}})
        }
    }
}

fn error(out: &mut impl Write, api: Api, status: u16, message: &str) -> io::Result<()> {
    respond(out, status, &error_body(api, status, message))
}

fn respond(out: &mut impl Write, status: u16, body: &Value) -> io::Result<()> {
    send(out, status, "application/json", body.to_string().as_bytes())
}

/// One whole response in one write; every connection carries one request.
fn send(out: &mut impl Write, status: u16, kind: &str, body: &[u8]) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    out.write_all(&[head.as_bytes(), body].concat())
}

fn start_events(out: &mut impl Write) -> io::Result<()> {
    out.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
          Connection: close\r\n\r\n",
    )
}

/// One OpenAI server-sent event.
fn data(out: &mut impl Write, v: &Value) -> io::Result<()> {
    out.write_all(format!("data: {v}\n\n").as_bytes())
}

/// An OpenAI stream's keepalive: a comment, which SSE parsers skip.
fn ping(out: &mut impl Write) -> io::Result<()> {
    out.write_all(b": ping\n\n")
}

/// One Anthropic server-sent event, named by its `type`.
fn event(out: &mut impl Write, v: &Value) -> io::Result<()> {
    let name = v["type"].as_str().unwrap_or_default();
    out.write_all(format!("event: {name}\ndata: {v}\n\n").as_bytes())
}

#[cfg(test)]
mod tests {
    use std::{
        net::Shutdown,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        time::Instant,
    };

    use clap::Parser;
    use larch_engine::{EngineOptions, start_with};
    use larch_kernels::{CPU, KError, KvPool, Stream, Tensor};
    use larch_model::{
        ChatTemplate, FakeModel, Model, ModelConfig, StepBatch, Tokenizer, gguf::Gguf, tiny,
    };

    use super::*;

    /// Says the scripted tokens one per logits row, then the end of turn, taking `pace` per pass;
    /// `steps` counts the passes, so a test sees whether the engine still runs a request.
    struct Script {
        cfg: ModelConfig,
        said: Arc<Mutex<VecDeque<u32>>>,
        eos: u32,
        steps: Arc<AtomicUsize>,
        pace: Duration,
        /// Unscripted rows get random logits instead of the end of turn, so sampling at temperature
        /// 1 walks a grammar freely.
        noise: Arc<AtomicBool>,
        rng: u64,
    }

    impl Model for Script {
        fn config(&self) -> &ModelConfig {
            &self.cfg
        }

        fn forward(
            &mut self,
            s: &Stream,
            b: &StepBatch,
            _kv: &KvPool,
            logits: &Tensor,
        ) -> Result<(), KError> {
            self.steps.fetch_add(1, Ordering::SeqCst);
            thread::sleep(self.pace);
            let vocab = self.cfg.vocab as usize;
            let mut l = vec![0f32; b.logits_rows.shape()[0] as usize * vocab];
            let mut said = self.said.lock().unwrap();
            for row in l.chunks_mut(vocab) {
                match said.pop_front() {
                    Some(id) => row[id as usize] = 100.0,
                    None if self.noise.load(Ordering::SeqCst) => row.iter_mut().for_each(|v| {
                        self.rng = (self.rng.wrapping_mul(6364136223846793005))
                            .wrapping_add(1442695040888963407);
                        *v = (self.rng >> 40) as f32 / (1u64 << 21) as f32;
                    }),
                    None => row[self.eos as usize] = 100.0,
                }
            }
            s.upload(logits.buffer(), logits.offset(), &l)
        }
    }

    /// A served Llama 3.1 vocabulary (so the Llama 3 call format) whose model says what `say` scripts.
    struct Fixture {
        server: &'static Server,
        port: u16,
        tokenizer: Tokenizer,
        said: Arc<Mutex<VecDeque<u32>>>,
        steps: Arc<AtomicUsize>,
        noise: Arc<AtomicBool>,
    }

    impl Fixture {
        fn new(key: Option<&str>, pace: Duration) -> Self {
            Self::with(key, pace, |_| {})
        }

        /// [`Fixture::new`] with `edit` applied to the server before it serves.
        fn with(key: Option<&str>, pace: Duration, edit: impl FnOnce(&mut Server)) -> Self {
            let g = Gguf::parse(&std::fs::read(tiny::TOKENIZER_GGUF).unwrap()).unwrap();
            let tokenizer = Tokenizer::from_gguf(&g).unwrap();
            let vocab = g
                .get("tokenizer.ggml.tokens")
                .unwrap()
                .as_arr()
                .unwrap()
                .len();
            let mut cfg = FakeModel::new(vocab as u32).config().clone();
            // The tiny vocabulary spends many tokens on a tool list.
            cfg.ctx_train = 1 << 16;
            let (said, steps, noise) = (Arc::default(), Arc::default(), Arc::default());
            let model = Script {
                cfg,
                said: Arc::clone(&said),
                eos: tokenizer.eos().unwrap(),
                steps: Arc::clone(&steps),
                pace,
                noise: Arc::clone(&noise),
                rng: 1,
            };
            let mut opts = EngineOptions::new("fake".into(), CPU);
            // The agent clients' recorded prompts run to 45k tokens of the tiny vocabulary.
            opts.max_context = 1 << 16;
            let engine = start_with(opts, Box::new(move |_, _, _| Ok(Box::new(model)))).unwrap();
            let template = ChatTemplate::from_gguf(&g, &tokenizer).unwrap();
            let larch = Larch::from_engine(engine, Tokenizer::from_gguf(&g).unwrap(), template);
            let mut server = Server::new(larch, "fake".into(), key.map(String::from), None);
            edit(&mut server);
            let server: &'static Server = Box::leak(Box::new(server));
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            thread::spawn(move || serve(server, listener));
            Self {
                server,
                port,
                tokenizer,
                said,
                steps,
                noise,
            }
        }

        /// Serves `larch` (a loaded model) as `larch serve` does.
        fn serving(larch: Larch, tokenizer: Tokenizer) -> Self {
            let server = Server::new(larch, "tiny".into(), None, None);
            let server: &'static Server = Box::leak(Box::new(server));
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            thread::spawn(move || serve(server, listener));
            Self {
                server,
                port,
                tokenizer,
                said: Arc::default(),
                steps: Arc::default(),
                noise: Arc::default(),
            }
        }

        fn say(&self, text: &str) {
            let ids = self.tokenizer.encode(text, false);
            self.said.lock().unwrap().extend(ids);
        }

        fn send(&self, method: &str, path: &str, headers: &str, body: &str) -> TcpStream {
            let mut conn = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
            let len = body.len();
            let request = format!(
                "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{headers}Content-Length: {len}\r\n\r\n{body}"
            );
            conn.write_all(request.as_bytes()).unwrap();
            conn
        }

        /// One request; the status and the body.
        fn call(&self, method: &str, path: &str, headers: &str, body: &str) -> (u16, String) {
            let mut response = String::new();
            let mut conn = self.send(method, path, headers, body);
            let read = conn.read_to_string(&mut response);
            read.unwrap_or_else(|e| panic!("{method} {path} {body}: {e}"));
            let (head, body) = response.split_once("\r\n\r\n").unwrap();
            (head[9..12].parse().unwrap(), body.into())
        }

        fn post(&self, path: &str, body: &Value) -> (u16, String) {
            self.call("POST", path, "", &body.to_string())
        }
    }

    /// The JSON of every server-sent event, `[DONE]` left out.
    fn events(body: &str) -> Vec<Value> {
        let lines = body.lines().filter_map(|l| l.strip_prefix("data: "));
        let json = lines.filter(|d| *d != "[DONE]");
        json.map(|d| serde_json::from_str(d).unwrap()).collect()
    }

    fn json(body: &str) -> Value {
        serde_json::from_str(body).unwrap()
    }

    fn weather() -> Value {
        json!({"type": "function", "function": {"name": "get_weather", "description": "Weather now",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}})
    }

    const CALL: &str = r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#;

    #[test]
    fn chat_answers_plain_and_streamed_alike() {
        let f = Fixture::new(None, Duration::ZERO);
        let (status, body) = f.call("GET", "/v1/models", "", "");
        assert_eq!(
            (status, json(&body)["data"][0]["id"].clone()),
            (200, json!("fake"))
        );
        let req = json!({"messages": [{"role": "user", "content": "hi"}], "temperature": 0});
        f.say("Hello there");
        let (status, body) = f.post("/v1/chat/completions", &req);
        assert_eq!(status, 200, "{body}");
        let v = json(&body);
        assert_eq!(v["choices"][0]["message"]["content"], "Hello there");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        let completion = v["usage"]["completion_tokens"].as_u64().unwrap();
        assert!(completion > 0);

        let mut req = req;
        req["stream"] = true.into();
        req["stream_options"] = json!({"include_usage": true});
        f.say("Hello there");
        let (status, body) = f.post("/v1/chat/completions", &req);
        assert_eq!(status, 200);
        assert!(body.ends_with("data: [DONE]\n\n"));
        let chunks = events(&body);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        let text: String = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, "Hello there");
        assert_eq!(
            chunks[chunks.len() - 2]["choices"][0]["finish_reason"],
            "stop"
        );
        assert_eq!(
            chunks.last().unwrap()["usage"]["completion_tokens"],
            completion
        );
    }

    #[test]
    fn chat_returns_tool_calls() {
        let f = Fixture::new(None, Duration::ZERO);
        let req =
            json!({"messages": [{"role": "user", "content": "weather?"}], "tools": [weather()]});
        f.say(CALL);
        let (status, body) = f.post("/v1/chat/completions", &req);
        assert_eq!(status, 200, "{body}");
        let v = json(&body);
        let message = &v["choices"][0]["message"];
        assert_eq!(message["content"], Value::Null);
        let call = &message["tool_calls"][0];
        assert_eq!(call["function"]["name"], "get_weather");
        // Nine characters, or a Mistral template refuses the call when it comes back.
        assert_eq!(call["id"].as_str().map(str::len), Some(9));
        let args: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"city": "Paris"}));
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");

        // The call goes back as history with its result, and the next answer streams; the call
        // markup itself never streams as text.
        let mut next = req.clone();
        next["messages"] = json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "18 C"},
        ]);
        next["stream"] = true.into();
        f.say("It is 18 C.");
        let (status, body) = f.post("/v1/chat/completions", &next);
        assert_eq!(status, 200, "{body}");
        let text: String = events(&body)
            .iter()
            .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(text, "It is 18 C.");
        next["messages"] = json!([{"role": "user", "content": "weather?"}]);
        f.say(CALL);
        let (_, body) = f.post("/v1/chat/completions", &next);
        let chunks = events(&body);
        assert!(chunks.iter().all(|c| {
            c["choices"][0]["delta"]["content"]
                .as_str()
                .is_none_or(str::is_empty)
        }));
        let calls = chunks
            .iter()
            .find_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array().cloned());
        assert_eq!(calls.unwrap()[0]["function"]["name"], "get_weather");
        assert_eq!(
            chunks.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    #[test]
    fn completions_take_a_raw_prompt() {
        let f = Fixture::new(None, Duration::ZERO);
        f.say("a b c d");
        let (status, body) = f.post("/v1/completions", &json!({"prompt": "x", "max_tokens": 2}));
        assert_eq!(status, 200, "{body}");
        let v = json(&body);
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(v["usage"]["completion_tokens"], 2);
        f.said.lock().unwrap().clear();
        f.say("ok");
        // A cap past what the context holds after the prompt runs to its end.
        let req = json!({"prompt": [1, 2, 3], "stream": true, "max_tokens": 1 << 16});
        let (_, body) = f.post("/v1/completions", &req);
        let text: String = events(&body)
            .iter()
            .filter_map(|c| c["choices"][0]["text"].as_str())
            .collect();
        assert_eq!(text, "ok");
    }

    #[test]
    fn messages_speak_anthropic() {
        let f = Fixture::new(None, Duration::ZERO);
        let tool = json!({"name": "get_weather", "description": "Weather now",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}});
        let req = json!({"max_tokens": 100, "system": [{"type": "text", "text": "Be brief."}],
            "messages": [{"role": "user", "content": "weather?"}], "tools": [tool]});
        f.say(CALL);
        let (status, body) = f.post("/v1/messages", &req);
        assert_eq!(status, 200, "{body}");
        let v = json(&body);
        assert_eq!(v["type"], "message");
        assert_eq!(v["stop_reason"], "tool_use");
        let block = &v["content"][0];
        assert_eq!(
            (&block["type"], &block["name"]),
            (&json!("tool_use"), &json!("get_weather"))
        );
        assert_eq!(block["input"], json!({"city": "Paris"}));
        assert_eq!(block["id"].as_str().map(str::len), Some(9));
        assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);

        let mut next = req.clone();
        next["messages"] = json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [block]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": block["id"], "content": "18 C"}]},
        ]);
        next["stream"] = true.into();
        f.say("It is 18 C.");
        let (status, body) = f.post("/v1/messages", &next);
        assert_eq!(status, 200, "{body}");
        let names: Vec<_> = body
            .lines()
            .filter_map(|l| l.strip_prefix("event: "))
            .collect();
        assert_eq!(names[0], "message_start");
        assert_eq!(names[names.len() - 2..], ["message_delta", "message_stop"]);
        let all = events(&body);
        let text: String = all
            .iter()
            .filter_map(|e| e["delta"]["text"].as_str())
            .collect();
        assert_eq!(text, "It is 18 C.");
        assert_eq!(all[all.len() - 2]["delta"]["stop_reason"], "end_turn");

        next["messages"] = json!([{"role": "user", "content": "weather?"}]);
        f.say(CALL);
        let (_, body) = f.post("/v1/messages", &next);
        let all = events(&body);
        let start = all
            .iter()
            .find(|e| e["type"] == "content_block_start")
            .unwrap();
        assert_eq!(start["content_block"]["type"], "tool_use");
        let input: String = all
            .iter()
            .filter_map(|e| e["delta"]["partial_json"].as_str())
            .collect();
        assert_eq!(json(&input), json!({"city": "Paris"}));
        assert_eq!(all[all.len() - 2]["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn keys_and_refusals_answer_in_each_api_shape() {
        let f = Fixture::new(Some("sesame"), Duration::ZERO);
        let chat = json!({"messages": [{"role": "user", "content": "hi"}]}).to_string();
        let (status, body) = f.call("POST", "/v1/chat/completions", "", &chat);
        assert_eq!(
            (status, json(&body)["error"]["type"].clone()),
            (401, json!("invalid_request_error"))
        );
        let wrong = "x-api-key: sesamf\r\n";
        let (status, body) = f.call("POST", "/v1/messages", wrong, "{}");
        assert_eq!(
            (status, json(&body)["error"]["type"].clone()),
            (401, json!("authentication_error"))
        );
        assert_eq!(f.call("GET", "/health", "", "").0, 200);
        // A client waiting for 100 Continue hears the refusal instead, and never sends its body.
        let mut conn = TcpStream::connect(("127.0.0.1", f.port)).unwrap();
        let head = "POST /v1/chat/completions HTTP/1.1\r\nExpect: 100-continue\r\n\
                    Content-Length: 9000000\r\n\r\n";
        conn.write_all(head.as_bytes()).unwrap();
        let mut response = String::new();
        conn.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 401"), "{response}");
        let lower = "authorization: bearer sesame\r\n";
        assert_eq!(f.call("GET", "/v1/models", lower, "").0, 200);
        let key = "Authorization: Bearer sesame\r\n";
        assert_eq!(f.call("GET", "/v1/models", key, "").0, 200);
        assert_eq!(f.call("GET", "/api/status", "", "").0, 401);
        assert_eq!(f.call("GET", "/api/status", key, "").0, 200);
        assert_eq!(f.call("GET", "/nope", key, "").0, 404);
        for refused in [
            json!({"messages": [{"role": "user", "content": "hi"}], "n": 2}),
            json!({"messages": [{"role": "user", "content": "hi"}], "logprobs": true}),
            json!({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}),
            json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}),
            // A cap past the context is clamped; a prompt past it is still refused.
            json!({"messages": [{"role": "user", "content": "a ".repeat(1 << 16)}],
                "max_tokens": 32000}),
        ] {
            let (status, body) = f.call("POST", "/v1/chat/completions", key, &refused.to_string());
            assert_eq!(status, 400, "{refused}: {body}");
            if refused["max_tokens"] == 32000 {
                // Anthropic's words, which Claude Code compacts its conversation on.
                let message = json(&body)["error"]["message"].clone();
                assert!(
                    message
                        .as_str()
                        .unwrap()
                        .starts_with("prompt is too long: ")
                );
            }
        }
        let (status, _) = f.call("POST", "/v1/chat/completions", key, "{");
        assert_eq!(status, 400);
        let image = json!({"max_tokens": 5, "messages": [{"role": "user", "content": [{"type": "image"}]}]});
        let (status, body) = f.call(
            "POST",
            "/v1/messages",
            "x-api-key: sesame\r\n",
            &image.to_string(),
        );
        assert_eq!((status, json(&body)["type"].clone()), (400, json!("error")));
        // Structured output is held to its schema; effort alone changes nothing.
        let n =
            json!({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]});
        let format = json!({"type": "json_schema", "schema": n});
        for structured in [
            json!({"output_config": {"format": format, "effort": "high"}}),
            json!({"output_format": format}),
        ] {
            let mut req =
                json!({"max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]});
            req.as_object_mut()
                .unwrap()
                .extend(structured.as_object().unwrap().clone());
            let (status, body) = f.call(
                "POST",
                "/v1/messages",
                "x-api-key: sesame\r\n",
                &req.to_string(),
            );
            assert_eq!(status, 200, "{req}: {body}");
            let text = json(&body)["content"][0]["text"].as_str().map(json);
            assert!(text.is_some_and(|v| v["n"].is_i64()), "{body}");
        }
        // A forced call is held to its tool.
        let forced = json!({"max_tokens": 100, "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "t", "input_schema": {}}], "tool_choice": {"type": "any"}});
        let (status, body) = f.call(
            "POST",
            "/v1/messages",
            "x-api-key: sesame\r\n",
            &forced.to_string(),
        );
        assert_eq!(status, 200, "{body}");
        let block = &json(&body)["content"][0];
        assert_eq!((&block["name"], &block["input"]), (&json!("t"), &json!({})));
        let chat = json!({"messages": [{"role": "user", "content": "hi"}], "tools": [weather()],
            "tool_choice": "required"});
        let (status, body) = f.call("POST", "/v1/chat/completions", key, &chat.to_string());
        assert_eq!(status, 200, "{body}");
        let call = &json(&body)["choices"][0]["message"]["tool_calls"][0]["function"];
        assert_eq!(call["name"], "get_weather");

        // One grammar per request; schemas without a regex, and unknown tools, are refused with
        // the reason, in each API's shape.
        let user = json!([{"role": "user", "content": "hi"}]);
        let all_of =
            json!({"type": "object", "properties": {"n": {"allOf": [{"type": "integer"}]}}});
        let openai_all_of =
            json!([{"type": "function", "function": {"name": "n", "parameters": all_of}}]);
        let schema = json!({"type": "json_schema", "json_schema": {"schema": {"type": "object"}}});
        for (refused, says) in [
            (
                json!({"messages": user, "tools": [weather()], "tool_choice": "required",
                "response_format": schema}),
                "one grammar",
            ),
            (
                json!({"messages": user, "tools": [weather()],
                "tool_choice": {"type": "function", "function": {"name": "nope"}}}),
                "nope",
            ),
            (
                json!({"messages": user, "tools": openai_all_of, "tool_choice": "required"}),
                "allOf",
            ),
            (
                json!({"messages": user, "tool_choice": "required"}),
                "no tools",
            ),
        ] {
            let (status, body) = f.call("POST", "/v1/chat/completions", key, &refused.to_string());
            let error = &json(&body)["error"];
            assert_eq!(
                (status, &error["type"]),
                (400, &json!("invalid_request_error")),
                "{body}"
            );
            assert!(error["message"].as_str().unwrap().contains(says), "{body}");
        }
        let tool = json!([{"name": "t", "input_schema": {}}]);
        for (refused, says) in [
            (
                json!({"tools": tool, "tool_choice": {"type": "any"}, "output_format": format}),
                "one grammar",
            ),
            (
                json!({"tools": tool, "tool_choice": {"type": "tool", "name": "nope"}}),
                "nope",
            ),
            (
                json!({"tools": [{"name": "n", "input_schema": all_of}], "tool_choice": {"type": "any"}}),
                "allOf",
            ),
        ] {
            let mut req = json!({"max_tokens": 5, "messages": user});
            req.as_object_mut()
                .unwrap()
                .extend(refused.as_object().unwrap().clone());
            let (status, body) = f.call(
                "POST",
                "/v1/messages",
                "x-api-key: sesame\r\n",
                &req.to_string(),
            );
            let error = &json(&body)["error"];
            assert_eq!(
                (status, &error["type"]),
                (400, &json!("invalid_request_error")),
                "{body}"
            );
            assert!(error["message"].as_str().unwrap().contains(says), "{body}");
        }
        // The engine's grammar refusals say why too.
        for (code, says) in [
            (invalid::GRAMMAR_SIZE, "too large"),
            (invalid::GRAMMAR_PATTERN, "not a regex"),
        ] {
            let (status, message) = engine_status(EngineError::InvalidRequest(code));
            assert!(status == 400 && message.contains(says), "{message}");
        }
    }

    /// Tools without free strings, so a random walk through their grammar ends well inside
    /// `max_tokens`: under random logits a string closes about once per vocabulary size.
    fn forced_tools() -> Vec<Value> {
        let weather = json!({"type": "object", "properties": {
            "city": {"enum": ["Paris", "Oslo"]}, "days": {"type": "integer"},
            "metric": {"type": "boolean"}}, "required": ["city", "days"]});
        let search = json!({"type": "object", "properties": {
            "limit": {"type": ["integer", "null"]},
            "tags": {"type": "array", "items": {"type": "number"}}}, "required": ["tags"]});
        [("get_weather", weather), ("search", search)]
            .map(|(name, parameters)| {
                json!({"type": "function", "function": {"name": name, "parameters": parameters}})
            })
            .into()
    }

    /// OpenAI function specs as Anthropic tools.
    fn anthropic_tools(tools: &[Value]) -> Vec<Value> {
        let tool = |f: &Value| json!({"name": f["name"], "input_schema": f["parameters"]});
        tools.iter().map(|t| tool(&t["function"])).collect()
    }

    /// Whether `args` fits tool `name` of [`forced_tools`].
    fn fits(name: &str, args: &Value) -> bool {
        let Some(o) = args.as_object() else {
            return false;
        };
        let int = |v: &Value| v.is_i64() || v.is_u64();
        let only = |keys: &[&str]| o.keys().all(|k| keys.contains(&k.as_str()));
        match name {
            "get_weather" => {
                only(&["city", "days", "metric"])
                    && matches!(
                        o.get("city").and_then(Value::as_str),
                        Some("Paris" | "Oslo")
                    )
                    && o.get("days").is_some_and(int)
                    && o.get("metric").is_none_or(Value::is_boolean)
            }
            "search" => {
                only(&["limit", "tags"])
                    && o.get("limit").is_none_or(|l| l.is_null() || int(l))
                    && (o.get("tags").and_then(Value::as_array))
                        .is_some_and(|t| t.iter().all(Value::is_number))
            }
            _ => false,
        }
    }

    /// A reply's calls as (name, arguments), from either API's body, streamed or not.
    fn calls_of(path: &str, stream: bool, body: &str) -> Vec<(String, Value)> {
        let pair = |name: &Value, args: Value| (name.as_str().unwrap().to_owned(), args);
        let openai = |c: &Value| {
            let f = &c["function"];
            pair(&f["name"], json(f["arguments"].as_str().unwrap()))
        };
        match (path, stream) {
            ("/v1/chat/completions", false) => (json(body)["choices"][0]["message"]["tool_calls"])
                .as_array()
                .map_or(Vec::new(), |c| c.iter().map(openai).collect()),
            ("/v1/chat/completions", true) => events(body)
                .iter()
                .filter_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array().cloned())
                .flatten()
                .map(|c| openai(&c))
                .collect(),
            (_, false) => (json(body)["content"].as_array().unwrap().iter())
                .filter(|b| b["type"] == "tool_use")
                .map(|b| pair(&b["name"], b["input"].clone()))
                .collect(),
            (_, true) => {
                let all = events(body);
                let names = (all.iter()).filter(|e| e["content_block"]["type"] == "tool_use");
                let args = all
                    .iter()
                    .filter_map(|e| e["delta"]["partial_json"].as_str());
                (names.zip(args))
                    .map(|(e, a)| pair(&e["content_block"]["name"], json(a)))
                    .collect()
            }
        }
    }

    /// IR-07c: under random logits at temperature 1, every forced reply is one call that fits its
    /// tool's schema, in both API shapes, streamed (odd seeds) or not. Anthropic's request has no
    /// seed; the engine seeds it with the request id.
    #[test]
    fn forced_calls_always_validate() {
        let f = Fixture::new(None, Duration::ZERO);
        f.noise.store(true, Ordering::SeqCst);
        let tools = forced_tools();
        let anthropic = anthropic_tools(&tools);
        let user = json!([{"role": "user", "content": "go"}]);
        let (mut names, started) = (std::collections::BTreeSet::new(), Instant::now());
        for seed in 0..50u64 {
            let stream = seed % 2 == 1;
            for (path, req) in [
                (
                    "/v1/chat/completions",
                    json!({"messages": user, "tools": tools,
                    "tool_choice": "required", "temperature": 1.0, "seed": seed,
                    "max_tokens": 400, "stream": stream}),
                ),
                (
                    "/v1/messages",
                    json!({"messages": user, "tools": anthropic,
                    "tool_choice": {"type": "any"}, "temperature": 1.0, "max_tokens": 400,
                    "stream": stream}),
                ),
            ] {
                let (status, body) = f.post(path, &req);
                assert_eq!(status, 200, "{body}");
                let calls = calls_of(path, stream, &body);
                assert!(
                    calls.len() == 1 && fits(&calls[0].0, &calls[0].1),
                    "{path} seed {seed}: {body}"
                );
                names.insert(calls[0].0.clone());
            }
        }
        assert_eq!(names.len(), 2, "the walk is random: both tools come up");
        eprintln!("100 forced requests in {:?}", started.elapsed());
    }

    /// A named tool is the one called; a JSON answer fits its schema; a JSON object ends where the
    /// object does.
    #[test]
    fn named_calls_and_json_answers_hold() {
        let f = Fixture::new(None, Duration::ZERO);
        f.noise.store(true, Ordering::SeqCst);
        let tools = forced_tools();
        let anthropic = anthropic_tools(&tools);
        let user = json!([{"role": "user", "content": "go"}]);
        let named = json!({"type": "function", "function": {"name": "search"}});
        for (path, req) in [
            (
                "/v1/chat/completions",
                json!({"messages": user, "tools": tools,
                "tool_choice": named, "temperature": 1.0, "max_tokens": 400}),
            ),
            (
                "/v1/messages",
                json!({"messages": user, "tools": anthropic,
                "tool_choice": {"type": "tool", "name": "search"}, "temperature": 1.0,
                "max_tokens": 400}),
            ),
        ] {
            let (status, body) = f.post(path, &req);
            assert_eq!(status, 200, "{body}");
            let calls = calls_of(path, false, &body);
            assert!(
                calls.len() == 1 && calls[0].0 == "search" && fits("search", &calls[0].1),
                "{body}"
            );
        }
        let schema = &tools[0]["function"]["parameters"];
        let format = json!({"type": "json_schema", "schema": schema});
        let answers = [
            (
                "/v1/chat/completions",
                json!({"messages": user, "temperature": 1.0, "max_tokens": 400,
                "response_format": {"type": "json_schema",
                    "json_schema": {"name": "w", "schema": schema, "strict": true}}}),
            ),
            (
                "/v1/messages",
                json!({"messages": user, "temperature": 1.0, "max_tokens": 400,
                "output_config": {"format": format}}),
            ),
            (
                "/v1/messages",
                json!({"messages": user, "temperature": 1.0, "max_tokens": 400,
                "output_format": format, "tools": anthropic}),
            ),
        ];
        for (path, req) in answers {
            let (status, body) = f.post(path, &req);
            assert_eq!(status, 200, "{body}");
            let v = json(&body);
            let text = match path {
                "/v1/messages" => &v["content"][0]["text"],
                _ => &v["choices"][0]["message"]["content"],
            };
            assert!(fits("get_weather", &json(text.as_str().unwrap())), "{body}");
        }
        f.noise.store(false, Ordering::SeqCst);
        f.say(r#"{"a": 1} and then some"#);
        let req = json!({"messages": user, "response_format": {"type": "json_object"}});
        let (status, body) = f.post("/v1/chat/completions", &req);
        f.said.lock().unwrap().clear();
        assert_eq!(status, 200, "{body}");
        assert_eq!(
            json(&body)["choices"][0]["message"]["content"],
            r#"{"a": 1}"#
        );
    }

    /// `--tool-grammar`: under random logits at temperature 1, every Auto call in the JSON body
    /// fits its tool's schema; without it the same reply holds no call that fits.
    #[test]
    fn tool_grammar_holds_auto_calls_in_the_json_body() {
        for held in [true, false] {
            let f = Fixture::with(None, Duration::ZERO, |s| {
                (s.family, s.tool_grammar) = (Some(Family::Qwen3), held);
            });
            f.noise.store(true, Ordering::SeqCst);
            for seed in 0..20u64 {
                f.say("<tool_call>");
                let req = json!({"messages": [{"role": "user", "content": "go"}],
                    "tools": forced_tools(), "temperature": 1.0, "seed": seed, "max_tokens": 300});
                let (status, body) = f.post("/v1/chat/completions", &req);
                assert_eq!(status, 200, "{body}");
                let calls = calls_of("/v1/chat/completions", false, &body);
                // The grammar re-arms after a call, so random text may start another.
                let fit = !calls.is_empty() && calls.iter().all(|(name, args)| fits(name, args));
                assert_eq!(fit, held, "held={held} seed {seed}: {body}");
            }
        }
    }

    /// A keyless server answers this machine only: a foreign Host (DNS rebinding) or a web
    /// page's Origin is refused before the body is read; loopback names and no headers pass.
    #[test]
    fn a_keyless_server_refuses_a_foreign_host_or_origin() {
        let f = Fixture::new(None, Duration::ZERO);
        let req = json!({"model": "fake", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]})
            .to_string();
        for headers in [
            "Host: evil.example:8080\r\n",
            "Host: 127.0.0.1.evil.example\r\n",
            "Host: 127.0.0.1:8080\r\nOrigin: https://evil.example\r\n",
            "Origin: null\r\n",
        ] {
            let (status, body) = f.call("POST", "/v1/chat/completions", headers, &req);
            assert_eq!(status, 403, "{headers}: {body}");
        }
        for headers in [
            "",
            "Host: 127.0.0.1:8080\r\n",
            "Host: localhost\r\nOrigin: http://localhost:8080\r\n",
            "Host: [::1]:8080\r\nOrigin: http://[::1]:8080\r\n",
        ] {
            f.say("ok");
            let (status, body) = f.call("POST", "/v1/chat/completions", headers, &req);
            assert_eq!(status, 200, "{headers}: {body}");
        }
    }

    #[test]
    fn loopback_authorities() {
        for ok in [
            "localhost",
            "LOCALHOST:1",
            "127.0.0.1",
            "127.8.9.1:80",
            "[::1]",
            "[::1]:8080",
        ] {
            assert!(loopback_authority(ok), "{ok}");
        }
        for no in [
            "evil.example",
            "127.0.0.1.evil.example",
            "localhost.evil",
            "10.0.0.1:80",
            "[::2]",
            "",
            "127.0.0.1:junk",
            "localhost:",
            "localhost:80/x",
            "[::1]junk",
            "[::1",
        ] {
            assert!(!loopback_authority(no), "{no}");
        }
    }

    /// `--tool-grammar` with a tool list past the grammar's size limit (Claude Code's 12 tools and
    /// a renamed copy of each, as MCP servers add): the request runs without the grammar.
    #[test]
    fn a_tool_list_too_large_for_the_grammar_runs_without_it() {
        let f = Fixture::with(None, Duration::ZERO, |s| {
            (s.family, s.tool_grammar) = (Some(Family::Qwen3), true);
        });
        let recorded = json(include_str!("../tests/serve-claude-code.json"));
        let tools = recorded["tools"].as_array().unwrap();
        let copies = tools.iter().map(|t| {
            let mut t = t.clone();
            t["name"] = json!(format!("{}_2", t["name"].as_str().unwrap()));
            t
        });
        let tools: Vec<Value> = tools.iter().cloned().chain(copies).collect();
        let req = json!({"model": "fake", "max_tokens": 64, "tools": tools,
            "messages": [{"role": "user", "content": "hi"}]});
        f.say("Hello.");
        let (status, body) = f.call("POST", "/v1/messages", CLAUDE_CODE, &req.to_string());
        assert_eq!(status, 200, "{body}");
        assert_eq!(json(&body)["content"][0]["text"], "Hello.", "{body}");
    }

    /// The XML body: with thinking on, a forced call arms after `</think>`, and Auto keeps calls
    /// well-formed; both come back with their values typed. A JSON answer stays held to its end.
    #[test]
    fn xml_calls_come_back_typed() {
        let f = Fixture::with(None, Duration::ZERO, |s| {
            (s.family, s.xml, s.thinks) = (Some(Family::Qwen3), true, true);
        });
        let call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
                    <parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let user = json!([{"role": "user", "content": "go"}]);
        for choice in [json!("required"), json!("auto")] {
            f.say(&format!("<think>\nweather, then\n</think>\n\n{call}"));
            let req = json!({"messages": user, "tools": forced_tools(), "tool_choice": choice});
            let (status, body) = f.post("/v1/chat/completions", &req);
            assert_eq!(status, 200, "{body}");
            let calls = calls_of("/v1/chat/completions", false, &body);
            let typed = json!({"city": "Paris", "days": 3});
            assert_eq!(
                calls,
                [("get_weather".to_owned(), typed)],
                "{choice}: {body}"
            );
        }
        f.say(r#"{"a": 1} and then some"#);
        let req = json!({"messages": user, "response_format": {"type": "json_object"}});
        let (status, body) = f.post("/v1/chat/completions", &req);
        f.said.lock().unwrap().clear();
        assert_eq!(status, 200, "{body}");
        assert_eq!(
            json(&body)["choices"][0]["message"]["content"],
            r#"{"a": 1}"#
        );
    }

    #[test]
    fn status_reads_the_traffic_off_the_ring() {
        let f = Fixture::new(None, Duration::ZERO);
        f.say("Hello there");
        let req = json!({"messages": [{"role": "user", "content": "hi"}]});
        let (_, body) = f.post("/v1/chat/completions", &req);
        let generated = json(&body)["usage"]["completion_tokens"].as_f64().unwrap();
        // The engine publishes a request's span as it ends, which may trail the answer.
        let deadline = Instant::now() + Duration::from_secs(5);
        let v = loop {
            let (status, body) = f.call("GET", "/api/status", "", "");
            assert_eq!(status, 200, "{body}");
            let v = json(&body);
            if v["traffic"]["requests"] == 1 || Instant::now() > deadline {
                break v;
            }
            thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(v["traffic"]["requests"], 1, "{v}");
        assert_eq!(v["traffic"]["gen_tokens"].as_f64(), Some(generated));
        assert_eq!(v["requests"][0]["finish"], "eos");
        // The chart's buckets hold each generated token once.
        let tok_s = v["throughput"]["tok_s"].as_array().unwrap();
        let tokens: f64 = tok_s
            .iter()
            .map(|t| t.as_f64().unwrap() * BUCKET_MS / 1e3)
            .sum();
        assert!(
            (tokens - generated).abs() < 1e-6 * generated,
            "{tokens} != {generated}"
        );
        let names: Vec<_> = v["knobs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|k| &k["name"])
            .collect();
        assert!(names.contains(&&json!("max_context")), "{names:?}");
    }

    #[test]
    fn a_client_that_leaves_cancels_its_request() {
        let f = Fixture::new(None, Duration::from_millis(5));
        let still = |f: &Fixture| {
            let before = f.steps.load(Ordering::SeqCst);
            thread::sleep(Duration::from_millis(300));
            before == f.steps.load(Ordering::SeqCst)
        };
        let words = "word ".repeat(400);
        // The third answer looks like a call from its first token, so nothing of it streams.
        let call = format!(r#"{{"name": "get_weather", "parameters": {{"city": "{words}"#);
        for (stream, said) in [(true, &words), (false, &words), (true, &call)] {
            f.said.lock().unwrap().clear();
            f.say(said);
            let req = json!({"messages": [{"role": "user", "content": "go"}], "stream": stream,
                "tools": [weather()]});
            let mut conn = f.send("POST", "/v1/chat/completions", "", &req.to_string());
            if stream {
                // The answer has started: its first chunk, naming the role, is in.
                let (mut seen, mut b) = (String::new(), [0; 256]);
                while !seen.contains("assistant") {
                    let n = conn.read(&mut b).unwrap();
                    assert!(n > 0, "{seen}");
                    seen.push_str(&String::from_utf8_lossy(&b[..n]));
                }
            } else {
                thread::sleep(Duration::from_millis(50));
            }
            drop(conn);
            thread::sleep(Duration::from_millis(100));
            assert!(
                still(&f),
                "stream {stream}, {said:.12}: the engine still runs the request"
            );
            assert!(
                f.said.lock().unwrap().len() > 100,
                "stream {stream}, {said:.12}"
            );
        }
    }

    #[test]
    fn streams_ping_while_a_prompt_prefills() {
        // Every pass, the prefill's too, outlasts two pings (a second apart under test).
        let f = Fixture::new(None, Duration::from_secs(2));
        let messages = json!([{"role": "user", "content": "hi"}]);
        // One token, which the prefill's pass samples.
        let anthropic = json!({"max_tokens": 1, "messages": messages, "stream": true});
        let openai = json!({"max_tokens": 1, "messages": messages, "stream": true});
        for (path, req, ping, text) in [
            ("/v1/messages", anthropic, "event: ping\n", "text_delta"),
            (
                "/v1/chat/completions",
                openai,
                ": ping\n",
                r#""delta":{"content":"#,
            ),
        ] {
            f.say("ok");
            let (status, body) = f.post(path, &req);
            assert_eq!(status, 200, "{body}");
            let at = |s: &str| body.find(s).unwrap_or_else(|| panic!("{s:?} in {body}"));
            assert!(at(ping) < at(text), "{body}");
        }
    }

    #[test]
    fn system_entries_fold_into_one_leading_system_message() {
        let m = |role: &str, content: &str| ToolMessage {
            role: role.into(),
            content: content.into(),
            ..ToolMessage::default()
        };
        let folded = fold_system(vec![
            m("system", "A"),
            m("user", "hi"),
            // Claude Code's per-turn effort entry carries no text.
            m("system", ""),
            m("system", "B"),
            m("assistant", "yo"),
        ]);
        let got: Vec<_> = folded
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str()))
            .collect();
        assert_eq!(
            got,
            [("system", "A\n\nB"), ("user", "hi"), ("assistant", "yo")]
        );
        assert_eq!(fold_system(vec![m("user", "hi")]).len(), 1);
        // An empty system prompt stays one.
        let empty = fold_system(vec![m("system", ""), m("user", "hi")]);
        assert_eq!((empty[0].role.as_str(), empty.len()), ("system", 2));
    }

    /// The headers Claude Code sends besides its key, with a beta larch has never heard of.
    const CLAUDE_CODE: &str = "anthropic-version: 2023-06-01\r\nanthropic-beta: \
        claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27,\
        advanced-tool-use-2025-11-20,effort-2025-11-24,not-yet-invented-2099-01-01\r\n";

    #[test]
    fn agent_clients_get_well_formed_answers() {
        // One real request from each client (tests/serve-*.json): a system prompt, its tools and a
        // tool call round trip, streamed with max_tokens 32000, more than the context holds.
        let f = Fixture::new(None, Duration::ZERO);
        let file = "/home/user/project/notes.txt";
        let clients = [
            (
                include_str!("../tests/serve-claude-code.json"),
                "Read",
                json!({"file_path": file}),
            ),
            (
                include_str!("../tests/serve-qwen-code.json"),
                "read_file",
                json!({"file_path": file}),
            ),
            (
                include_str!("../tests/serve-opencode.json"),
                "read",
                json!({"filePath": file}),
            ),
        ];
        for (recorded, tool, args) in clients {
            let mut req = json(recorded);
            let anthropic = req.get("system").is_some();
            let (path, headers) = if anthropic {
                ("/v1/messages?beta=true", CLAUDE_CODE)
            } else {
                ("/v1/chat/completions", "")
            };
            let post = |req: &Value, say: &str| {
                f.say(say);
                let (status, body) = f.call("POST", path, headers, &req.to_string());
                assert_eq!(status, 200, "{tool}: {body}");
                body
            };
            let call = json!({"name": tool, "parameters": args}).to_string();
            let body = post(&req, &call);
            let all = events(&body);
            let messages = req["messages"].as_array_mut().unwrap();
            if anthropic {
                let names: Vec<_> = body
                    .lines()
                    .filter_map(|l| l.strip_prefix("event: "))
                    .collect();
                assert_eq!(names[0], "message_start", "{body}");
                assert_eq!(names[names.len() - 2..], ["message_delta", "message_stop"]);
                let start = all.iter().find(|e| e["type"] == "content_block_start");
                let block = start.unwrap()["content_block"].clone();
                assert_eq!(block["name"], tool, "{body}");
                let input: String = all
                    .iter()
                    .filter_map(|e| e["delta"]["partial_json"].as_str())
                    .collect();
                assert_eq!(json(&input), args);
                assert_eq!(all[all.len() - 2]["delta"]["stop_reason"], "tool_use");
                let id = &block["id"];
                messages.push(json!({"role": "assistant", "content": [
                    {"type": "tool_use", "id": id, "name": tool, "input": args}]}));
                messages.push(json!({"role": "user", "content": [{"type": "tool_result",
                    "tool_use_id": id, "content": "1\tThe answer is 42.\n"}]}));
            } else {
                assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
                let calls = all
                    .iter()
                    .find_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array().cloned());
                let call = calls.unwrap()[0].clone();
                assert_eq!(call["function"]["name"], tool, "{body}");
                let arguments = call["function"]["arguments"].as_str().unwrap();
                assert_eq!(json(arguments), args);
                assert_eq!(
                    all[all.len() - 2]["choices"][0]["finish_reason"],
                    "tool_calls"
                );
                assert!(all.last().unwrap()["usage"]["prompt_tokens"].as_u64() > Some(0));
                messages.push(json!({"role": "assistant", "content": null, "tool_calls": [call]}));
                messages.push(json!({"role": "tool", "tool_call_id": call["id"],
                    "content": "The answer is 42."}));
            }
            let body = post(&req, "It says 42.");
            let all = events(&body);
            let text: String = all
                .iter()
                .filter_map(|e| {
                    e["delta"]["text"]
                        .as_str()
                        .or(e["choices"][0]["delta"]["content"].as_str())
                })
                .collect();
            assert_eq!(text, "It says 42.", "{tool}: {body}");
            let stop = if anthropic {
                &all[all.len() - 2]["delta"]["stop_reason"]
            } else {
                &all[all.len() - 2]["choices"][0]["finish_reason"]
            };
            assert_eq!(stop, if anthropic { "end_turn" } else { "stop" });
        }
    }

    #[test]
    fn count_tokens_counts_what_messages_prefills() {
        let f = Fixture::new(None, Duration::ZERO);
        let mut req = json(include_str!("../tests/serve-claude-code.json"));
        let full = req.clone();
        // A count carries no max_tokens, and runs no model.
        let object = req.as_object_mut().unwrap();
        object.remove("max_tokens");
        object.remove("stream");
        let steps = f.steps.load(Ordering::SeqCst);
        let path = "/v1/messages/count_tokens?beta=true";
        let (status, body) = f.call("POST", path, CLAUDE_CODE, &req.to_string());
        assert_eq!(status, 200, "{body}");
        assert_eq!(f.steps.load(Ordering::SeqCst), steps);
        f.say("ok");
        let (_, streamed) = f.post("/v1/messages", &full);
        // The engine's own count of the prompt it ran, cached or not.
        let all = events(&streamed);
        let usage = &all[all.len() - 2]["usage"];
        let ran = usage["input_tokens"].as_u64().unwrap()
            + usage["cache_read_input_tokens"].as_u64().unwrap();
        assert_eq!(json(&body)["input_tokens"], ran);
    }

    #[test]
    fn system_entries_reach_the_template_folded() {
        let f = Fixture::new(None, Duration::ZERO);
        let user = |text: &str| json!({"role": "user", "content": text});
        let system = |text: &str| json!({"role": "system", "content": text});
        // Anthropic: a mid-conversation entry counts as if the top-level system said it too.
        let count = |system: &str, messages: Value| {
            let req = json!({"system": system, "messages": messages});
            let (status, body) = f.post("/v1/messages/count_tokens", &req);
            assert_eq!(status, 200, "{body}");
            json(&body)["input_tokens"].clone()
        };
        assert_eq!(
            count("S1", json!([user("hi"), system("S2"), user("go")])),
            count("S1\n\nS2", json!([user("hi"), user("go")]))
        );
        // OpenAI: the same, by the prompt its answer reports.
        let prompt = |messages: Value| {
            f.say("ok");
            let (status, body) = f.post("/v1/chat/completions", &json!({"messages": messages}));
            assert_eq!(status, 200, "{body}");
            json(&body)["usage"]["prompt_tokens"].clone()
        };
        assert_eq!(
            prompt(json!([system("S1"), user("hi"), system("S2"), user("go")])),
            prompt(json!([system("S1\n\nS2"), user("hi"), user("go")]))
        );
    }

    #[test]
    fn a_stream_ends_when_its_client_leaves_before_the_first_token() {
        // The first ping to a closed socket still succeeds, so the client is asked first.
        let f = Fixture::new(None, Duration::from_secs(2));
        let messages = json!([{"role": "user", "content": "hi"}]);
        for (path, text) in [
            ("/v1/messages", "text_delta"),
            ("/v1/chat/completions", r#""delta":{"content":"#),
        ] {
            f.said.lock().unwrap().clear();
            f.say("a b c d e f");
            let req = json!({"max_tokens": 6, "messages": messages, "stream": true});
            let mut conn = f.send("POST", path, "", &req.to_string());
            // Gone as the server sees it: nothing more will ever come from this client.
            conn.shutdown(Shutdown::Write).unwrap();
            let mut body = String::new();
            conn.read_to_string(&mut body).unwrap();
            // Cancelled at the first ping's turn, inside the prefill's pass.
            assert!(!body.contains(text), "{path}: {body}");
        }
    }

    #[test]
    fn the_context_running_out_is_not_the_clients_cap() {
        let a = |finish| Answer {
            text: String::new(),
            calls: Vec::new(),
            finish,
            usage: Usage::default(),
        };
        assert_eq!(stop_reason(&a(Finish::Length), false), "max_tokens");
        // A cap the context could not hold was dropped, so the context ended the answer.
        assert_eq!(
            stop_reason(&a(Finish::Length), true),
            "model_context_window_exceeded"
        );
        assert_eq!(stop_reason(&a(Finish::StopSeq), true), "stop_sequence");
    }

    #[test]
    fn calls_hold_back_and_split() {
        // A Qwen model's several calls, and the text before them.
        let qwen = "Two.\n<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>\n\
                    <tool_call>{\"name\": \"b\", \"arguments\": {\"x\": 1}}</tool_call>";
        let (end, calls) = split(Family::Qwen3, qwen);
        assert_eq!(&qwen[..end], "Two.");
        let names: Vec<_> = calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
        let broken = "<tool_call>{broken";
        assert_eq!(split(Family::Qwen3, broken), (broken.len(), Vec::new()));
        // A marker, or the start of one, holds the text from where it begins.
        assert_eq!(hold(Family::Qwen3, "Sure <tool", 0), 5);
        assert_eq!(hold(Family::Qwen3, "Sure <function-call>{\"na", 0), 5);
        assert_eq!(hold(Family::Qwen3, "Sure <func", 0), 5);
        assert_eq!(hold(Family::Qwen3, "Sure <b>", 0), 8);
        assert_eq!(hold(Family::Qwen3, "Sure <b> <tool_call>", 5), 9);
        // A Llama line that may be a bare call holds; prose before it streams.
        assert_eq!(hold(Family::Llama3, "Let me check.\n{\"na", 0), 14);
        assert_eq!(hold(Family::Llama3, "Let me check.\n", 0), 14);
        assert_eq!(hold(Family::Llama3, "Let me\ncheck", 0), 12);
        assert_eq!(hold(Family::Llama3, "  {", 0), 0);
        // Scanning resumes where streaming stopped: mid-line JSON is prose, a new line may not be.
        assert_eq!(hold(Family::Llama3, "Set x = {a}", 8), 11);
        assert_eq!(hold(Family::Llama3, "Let me check.\n{", 14), 14);
        assert_eq!(hold(Family::Llama3, "Let me check.\n\n{", 14), 14);
    }

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        serve: Options,
    }

    #[test]
    fn a_fresh_start_listens_only_on_this_machine() {
        let start = |host: &str| {
            let cli = Cli::parse_from(["serve", "-m", "missing.gguf", "--cpu", "--host", host]);
            format!("{:#}", run(cli.serve, &mut Vec::new()).unwrap_err())
        };
        // Unless LARCH_API_KEY is set, which this test leaves alone.
        if env::var_os("LARCH_API_KEY").is_none() {
            assert!(start("0.0.0.0").contains("LARCH_API_KEY"));
        }
        assert!(!start("127.0.0.1").contains("LARCH_API_KEY"));
        assert!(!start("localhost").contains("LARCH_API_KEY"));
        let format = |name: &str| Cli::try_parse_from(["serve", "-m", "x", "--tool-format", name]);
        assert!(format("mistral").is_ok() && format("gpt").is_err());
    }

    #[test]
    fn a_restart_rebinds_the_port() {
        // A supervised restart binds the port its last run served, while that run's closed
        // connections still sit in TIME_WAIT.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = TcpStream::connect(addr).unwrap();
        drop(listener.accept().unwrap());
        drop((listener, client));
        TcpListener::bind(addr).unwrap();
    }

    /// The 35B's chat template, as its GGUF carries it (llama.cpp's Qwen3.5-4B.jinja is the same).
    const QWEN35: &str = include_str!("../../larch-model/tests/data/no-roundtrip/qwen35.jinja");

    /// A tiny random `arch` model on the CPU with byte tokens, Qwen 3.5's chat template and room for
    /// an agent's conversation, loaded as `larch serve` loads it (the prefix cache at its cap).
    fn tiny_model(arch: larch_model::Arch) -> (Larch, Tokenizer) {
        use larch_model::gguf::Value as V;
        let bytes = tiny::gguf(arch, 3);
        let g = Gguf::parse(&bytes).unwrap();
        let tensors: Vec<_> = (g.tensors.iter())
            .map(|t| {
                let at = (g.data_offset + t.offset) as usize;
                let data = bytes[at..at + t.bytes as usize].to_vec();
                (t.name.clone(), t.dims.clone(), t.ggml_type, data)
            })
            .collect();
        let replaced = |k: &str| k.starts_with("tokenizer.") || k.ends_with(".context_length");
        let name = g
            .get("general.architecture")
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        let mut kv: Vec<_> = g.kv.into_iter().filter(|(k, _)| !replaced(k)).collect();
        let bytes_ = (0..256).map(|b| V::Str(format!("<0x{b:02X}>"))).collect();
        kv.extend([
            (format!("{name}.context_length"), V::U32(16384)),
            ("tokenizer.ggml.model".into(), V::Str("llama".into())),
            ("tokenizer.ggml.tokens".into(), V::Arr(bytes_)),
            (
                "tokenizer.ggml.token_type".into(),
                V::Arr(vec![V::I32(6); 256]),
            ),
            ("tokenizer.ggml.bos_token_id".into(), V::U32(97)),
            ("tokenizer.ggml.eos_token_id".into(), V::U32(255)),
            ("tokenizer.ggml.add_bos_token".into(), V::Bool(false)),
            ("tokenizer.ggml.add_space_prefix".into(), V::Bool(false)),
            ("tokenizer.chat_template".into(), V::Str(QWEN35.into())),
        ]);
        let file = larch_model::gguf::write(&kv, &tensors);
        static FILES: AtomicUsize = AtomicUsize::new(0);
        let n = FILES.fetch_add(1, Ordering::SeqCst);
        let dir = env::temp_dir().join(format!("larch-serve-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.gguf");
        std::fs::write(&path, &file).unwrap();
        let tokenizer = Tokenizer::from_gguf(&Gguf::parse(&file).unwrap()).unwrap();
        let mut load = LoadOptions::new(path, CPU);
        load.prefix_cache_blocks = usize::MAX;
        (Larch::load(load).unwrap(), tokenizer)
    }

    /// An agent's tool, in OpenAI's shape.
    fn bash() -> Value {
        json!({"type": "function", "function": {"name": "bash", "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
            "required": ["command"]}}})
    }

    /// A Claude Code conversation as it grows, one request body a step: its system prompt (billing
    /// header first) and tools, a mid-conversation system entry, tool round trips, reminders
    /// beside tool results, and at the end a new question.
    fn claude_code_steps() -> Vec<Value> {
        let call = |i: usize| {
            json!({"role": "assistant", "content": [
            {"type": "text", "text": format!("Step {i}.")},
            {"type": "tool_use", "id": format!("toolu_{i}"), "name": "bash",
                "input": {"command": format!("cat f{i}")}}]})
        };
        let result = |i: usize, reminder: bool| {
            let mut blocks = vec![
                json!({"type": "tool_result", "tool_use_id": format!("toolu_{i}"),
                "content": format!("contents of f{i}")}),
            ];
            if reminder {
                blocks.push(json!({"type": "text",
                    "text": "<system-reminder>The task list is empty.</system-reminder>"}));
            }
            json!({"role": "user", "content": blocks})
        };
        let mut messages = vec![
            json!({"role": "user", "content": "What do f0 and f1 say?"}),
            json!({"role": "system", "content": "# Environment\nPlatform: linux"}),
        ];
        let mut steps = Vec::new();
        let mut step = |messages: &Vec<Value>, header: &str| {
            steps.push(json!({
                "max_tokens": 4, "temperature": 0,
                "system": [{"type": "text", "text": format!("x-anthropic-billing-header: {header}")},
                    {"type": "text", "text": "You are a coding agent."}],
                "tools": [{"name": "bash", "description": "Run a command",
                    "input_schema": bash()["function"]["parameters"].clone()}],
                "messages": messages,
            }));
        };
        step(&messages, "cch=1;");
        for i in 0..3 {
            messages.push(call(i));
            messages.push(result(i, i == 1));
            step(&messages, &format!("cch={};", i + 2));
        }
        messages.push(json!({"role": "assistant", "content": "f0 says a, f1 says b."}));
        messages.push(json!({"role": "user", "content": "Now f2?"}));
        step(&messages, "cch=9;");
        steps
    }

    /// A mini-swe-agent conversation as it grows: commands in the reply text, outputs as user
    /// messages, no tools.
    fn mini_swe_steps() -> Vec<Value> {
        let mut messages = vec![
            json!({"role": "system", "content": "You solve tasks with one bash command a reply."}),
            json!({"role": "user", "content": "<pr_description>Fix the paginator.</pr_description>"}),
        ];
        let mut steps = Vec::new();
        for i in 0..4 {
            steps.push(json!({"max_tokens": 4, "temperature": 0, "messages": messages.clone()}));
            messages.push(json!({"role": "assistant",
                "content": format!("THOUGHT: look.\n\n```bash\nls dir{i}\n```")}));
            messages.push(json!({"role": "user",
                "content": format!("<returncode>0</returncode>\n<output>file{i}</output>")}));
        }
        steps
    }

    /// An OpenAI tool-calling conversation (OpenCode, Qwen Code) as it grows.
    fn openai_tool_steps() -> Vec<Value> {
        let mut messages = vec![
            json!({"role": "system", "content": "You are a coding agent."}),
            json!({"role": "user", "content": "Read f0 and f1."}),
        ];
        let mut steps = Vec::new();
        for i in 0..3 {
            steps.push(json!({"max_tokens": 4, "temperature": 0, "tools": [bash()],
                "messages": messages.clone()}));
            messages.push(json!({"role": "assistant", "content": null, "tool_calls": [{
                "id": format!("call_{i}"), "type": "function",
                "function": {"name": "bash", "arguments": format!("{{\"command\":\"cat f{i}\"}}")}}]}));
            messages.push(json!({"role": "tool", "tool_call_id": format!("call_{i}"),
                "content": format!("contents of f{i}")}));
        }
        steps
    }

    /// What serve prefills for an OpenAI or Anthropic request body, and its history's length.
    fn rendered(server: &Server, body: &Value) -> (Vec<u32>, usize) {
        if body.get("system").is_some() {
            let req: PromptIn = serde_json::from_value(body.clone()).unwrap();
            let Ok((tokens, end, _)) = anthropic_prompt(server, req, None, true) else {
                panic!("{body} renders")
            };
            return (tokens, end as usize);
        }
        let messages: Vec<MessageIn> = serde_json::from_value(body["messages"].clone()).unwrap();
        let Ok(messages) = messages.into_iter().map(openai_message).collect() else {
            panic!("{body} reads")
        };
        let tools = body["tools"].as_array().cloned().unwrap_or_default();
        let Ok((tokens, end)) = render(server, messages, &tools, None, true) else {
            panic!("{body} renders")
        };
        (tokens, end as usize)
    }

    /// Each step of an agent's conversation prefills the last step's history again, token for
    /// token, so the engine can run on from where the last request's prompt ended: through tool
    /// round trips, Claude Code's environment entry and a changing billing header. New user text
    /// (a question, or a reminder beside a tool result) renders the replies since the last one
    /// anew (the template drops their empty think blocks), and its prompt still repeats everything
    /// before them. A system entry that first shows mid-way joins the leading system text, so it
    /// changes the prompt from its start: a miss serve accepts rather than move an instruction.
    #[test]
    fn agent_prompts_repeat_the_last_history() {
        let (larch, tokenizer) = tiny_model(larch_model::Arch::Llama);
        let f = Fixture::serving(larch, tokenizer);
        for (client, steps) in [
            ("claude code", claude_code_steps()),
            ("mini-swe-agent", mini_swe_steps()),
            ("openai tools", openai_tool_steps()),
        ] {
            let prompts: Vec<_> = steps.iter().map(|s| rendered(f.server, s)).collect();
            for (i, pair) in prompts.windows(2).enumerate() {
                let ((last, end), (next, _)) = (&pair[0], &pair[1]);
                assert!(
                    *end > 0 && *end < last.len(),
                    "{client} step {i}: history {end}"
                );
                let shared = last.iter().zip(next).take_while(|(a, b)| a == b).count();
                // A reminder beside a tool result (step 2) and a new question (the last step) are
                // new user text: the template renders the replies since the last one anew.
                let question = client == "claude code" && (i + 1 == 2 || i + 2 == prompts.len());
                if question {
                    // Up to the first reply after the opening question, past the system prompt.
                    let first = text(&f.tokenizer, &last[..shared]);
                    assert!(first.ends_with("<|im_start|>assistant\n"), "{first}");
                    assert!(first.contains("What do f0 and f1 say?"), "{first}");
                } else {
                    let around = shared.saturating_sub(40)..(shared + 40).min(next.len());
                    let at = text(&f.tokenizer, &next[around]);
                    assert!(shared >= *end, "{client} step {} differs at: {at:?}", i + 1);
                }
            }
        }
    }

    fn text(tokenizer: &Tokenizer, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| tokenizer.decode_piece(id).to_vec())
            .collect();
        String::from_utf8_lossy(&bytes).replace('▁', " ")
    }

    /// The usage of one non-streamed request: prompt and cached tokens, and the answer's text.
    fn ask(f: &Fixture, body: &Value) -> (u64, u64, String) {
        let anthropic = body.get("system").is_some();
        let path = if anthropic {
            "/v1/messages"
        } else {
            "/v1/chat/completions"
        };
        let (status, out) = f.post(path, body);
        assert_eq!(status, 200, "{out}");
        let v = json(&out);
        if anthropic {
            let u = &v["usage"];
            let cached = u["cache_read_input_tokens"].as_u64().unwrap();
            let prompt = u["input_tokens"].as_u64().unwrap() + cached;
            (prompt, cached, v["content"].to_string())
        } else {
            let u = &v["usage"];
            let cached = u["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap();
            (
                u["prompt_tokens"].as_u64().unwrap(),
                cached,
                v["choices"][0]["message"].to_string(),
            )
        }
    }

    /// Three agents' conversations, interleaved a step each on one server, each read back its own
    /// last prompt: every step after a conversation's first reports its last history as cached
    /// (all of it on a hybrid model, which resumes from the recurrent state parked there), and
    /// answers at temperature 0 as a server that never saw the conversation does.
    fn conversations_reuse_their_own_prompts(arch: larch_model::Arch) {
        let hybrid = arch == larch_model::Arch::Qwen35;
        let (larch, tokenizer) = tiny_model(arch);
        let f = Fixture::serving(larch, tokenizer);
        let all = [claude_code_steps(), mini_swe_steps(), openai_tool_steps()];
        let rounds = all.iter().map(Vec::len).max().unwrap();
        let mut last: Vec<Option<usize>> = vec![None; all.len()];
        for round in 0..rounds {
            for (c, steps) in all.iter().enumerate() {
                let Some(body) = steps.get(round) else {
                    continue;
                };
                let (tokens, end) = rendered(f.server, body);
                let (prompt, cached, _) = ask(&f, body);
                assert_eq!(prompt as usize, tokens.len());
                let question = c == 0 && (round == 2 || round + 1 == steps.len());
                match last[c] {
                    // The system prompt's blocks are shared; a hybrid resumes only at a
                    // checkpoint, a block's end. The first request finds nothing.
                    None => assert!(
                        cached < end as u64
                            && (!hybrid || cached.is_multiple_of(16))
                            && (round > 0 || c > 0 || cached == 0),
                        "conversation {c}'s first step: {cached} cached"
                    ),
                    // A new question re-renders past replies: no claim but that nothing breaks.
                    Some(_) if question => {}
                    // The prefix cache keeps whole blocks; a kept session, everything.
                    Some(history) => assert!(
                        cached as usize >= if hybrid { history } else { history / 16 * 16 }
                            && cached < prompt,
                        "conversation {c} step {round}: {cached} cached of {prompt}, history {history}"
                    ),
                }
                last[c] = Some(end);
            }
        }
        // The last step of each, cached and fresh, answers alike.
        for steps in &all {
            let (larch, tokenizer) = tiny_model(arch);
            let fresh = Fixture::serving(larch, tokenizer);
            let body = steps.last().unwrap();
            let (_, cached, warm) = ask(&f, body);
            let (_, none, cold) = ask(&fresh, body);
            assert!(cached > 0);
            assert_eq!(none, 0);
            assert_eq!(warm, cold);
        }
    }

    /// A client that leaves mid-answer leaves its conversation's cached history in place: the next
    /// request of the conversation still reads the last answered one's, to the whole blocks it
    /// fills (the engine withdraws only the left request's own blocks).
    #[test]
    fn a_cancelled_answer_keeps_its_conversation_cached() {
        let (larch, tokenizer) = tiny_model(larch_model::Arch::Llama);
        let f = Fixture::serving(larch, tokenizer);
        let mut messages = vec![
            json!({"role": "system", "content": "You are terse. ".repeat(20)}),
            json!({"role": "user", "content": "Count to a thousand. ".repeat(10)}),
        ];
        let first = json!({"messages": messages, "max_tokens": 2});
        let (_, history) = rendered(f.server, &first);
        ask(&f, &first);
        messages.extend([
            json!({"role": "assistant", "content": "1, 2"}),
            json!({"role": "user", "content": "Go on."}),
        ]);
        let left = json!({"messages": messages, "stream": true, "max_tokens": 8000,
            "temperature": 0});
        let mut conn = f.send("POST", "/v1/chat/completions", "", &left.to_string());
        // Past the role chunk: the prefill is done and the answer streams.
        let (mut seen, mut b) = (String::new(), [0; 256]);
        while seen.matches("data: ").count() < 3 {
            let n = conn.read(&mut b).unwrap();
            assert!(n > 0, "{seen}");
            seen.push_str(&String::from_utf8_lossy(&b[..n]));
        }
        drop(conn);
        thread::sleep(Duration::from_millis(300));
        messages.extend([
            json!({"role": "assistant", "content": "3, 4"}),
            json!({"role": "user", "content": "Go on."}),
        ]);
        let (_, cached, _) = ask(&f, &json!({"messages": messages, "max_tokens": 1}));
        assert!(history >= 16, "{history}");
        let blocks = history / 16 * 16;
        assert!(
            cached as usize >= blocks,
            "{cached} cached, history {history}"
        );
    }

    #[test]
    fn dense_conversations_reuse_their_own_prompts() {
        conversations_reuse_their_own_prompts(larch_model::Arch::Llama);
    }

    #[test]
    fn hybrid_conversations_reuse_their_own_prompts() {
        conversations_reuse_their_own_prompts(larch_model::Arch::Qwen35);
    }

    /// After a burst of parallel conversations past the cap, the next request trims the idle
    /// ones back to it, oldest first, and keeps the one it continues.
    #[test]
    fn idle_conversations_trim_back_to_the_cap() {
        let (larch, _) = tiny_model(larch_model::Arch::Qwen35);
        let mut convs = Convs::default();
        let burst: Vec<_> = (0..CONVERSATIONS as u32 + 3)
            .map(|i| convs.pick(&larch, &[i, i]))
            .collect();
        assert_eq!(convs.list.len(), CONVERSATIONS + 3);
        for (i, session) in burst.iter().enumerate() {
            convs.done(session, Some((session.id(), vec![i as u32])));
        }
        assert_eq!(convs.list.len(), CONVERSATIONS);
        assert!(
            convs.list.iter().all(|c| c.history[0] >= 3),
            "the oldest went"
        );
    }

    #[test]
    fn a_rejected_request_keeps_every_conversation() {
        let (larch, _) = tiny_model(larch_model::Arch::Qwen35);
        let mut convs = Convs::default();
        for i in 0..CONVERSATIONS as u32 {
            let session = convs.pick(&larch, &[i, i]);
            convs.done(&session, Some((session.id(), vec![i])));
        }
        let before: Vec<_> = convs.list.iter().map(|c| c.session.clone()).collect();
        let rejected = convs.pick(&larch, &[99, 99]);
        convs.done(&rejected, None);
        assert_eq!(convs.list.len(), CONVERSATIONS);
        assert!(
            (before.iter()).all(|s| convs.list.iter().any(|c| Arc::ptr_eq(&c.session, s))),
            "a kept conversation went for a rejected one"
        );
    }

    #[test]
    fn a_released_history_gives_way_to_a_kept_one() {
        let (larch, _) = tiny_model(larch_model::Arch::Qwen35);
        let mut convs = Convs::default();
        let (a, b) = (convs.pick(&larch, &[1, 2]), convs.pick(&larch, &[1, 2]));
        convs.done(&a, Some((a.id(), vec![1])));
        // B went on to [1, 2], and that turn was cancelled: its ID moved on.
        convs.done(&b, Some((b.id() + 1, vec![1, 2])));
        assert!(Arc::ptr_eq(&convs.pick(&larch, &[1, 2, 3]), &a));
    }

    #[test]
    fn the_billing_header_goes_and_the_text_after_it_stays() {
        let (larch, tokenizer) = tiny_model(larch_model::Arch::Llama);
        let f = Fixture::serving(larch, tokenizer);
        let system = |parts: Value| {
            let body = json!({
                "system": parts,
                "messages": [{"role": "user", "content": "Read the repository."}],
                "max_tokens": 4,
            });
            text(&f.tokenizer, &rendered(f.server, &body).0)
        };
        let joined = system(json!([{"type": "text",
            "text": "x-anthropic-billing-header: cch=1;\nNever transmit repository files."}]));
        assert!(
            joined.contains("Never transmit repository files."),
            "{joined}"
        );
        assert!(!joined.contains("billing"), "{joined}");
        let alone = system(json!([
            {"type": "text", "text": "x-anthropic-billing-header: cch=1;"},
            {"type": "text", "text": "Base instructions."},
        ]));
        assert!(
            !alone.contains("billing") && alone.contains("Base instructions."),
            "{alone}"
        );
    }
}
