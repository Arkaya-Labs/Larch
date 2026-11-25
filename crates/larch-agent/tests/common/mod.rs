//! A CPU engine whose model says what the test scripts: the tiny tokenizer (Llama 3.1 template,
//! so the Llama 3 call format) with `<think>` and `</think>` added as single tokens, and the prefix
//! cache on. Every request samples one scripted token per step, except an anchor (a prompt that
//! ends in an empty user message, as `run::anchor` sends it), which ends at once and takes nothing
//! from the script: a chat anchors at its first turn, after the test scripted that turn's reply. A
//! step can give its token a runner-up (`close_call`), for the first token's logprobs, and a
//! tapped step reads each row's last token.
// Each test crate uses part of it.
#![allow(dead_code)]

use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use larch::Larch;
use larch_agent::{Agent, AgentEvent, AgentOptions};
use larch_engine::{EngineEvent, EngineHandle, EngineOptions, Request, start_with};
use larch_kernels::{CPU, KError, KvPool, Stream, Tensor};
use larch_model::{
    ChatTemplate, FakeModel, Model, ModelConfig, StepBatch, Taps, Tokenizer, ToolMessage,
    gguf::{Gguf, Value},
    tiny,
};

/// Samples the next scripted token for every logits row, whatever the prompt, and keeps the
/// tokens it was fed. Fed `</think>` inside a scripted think block, as the facade feeds it after a
/// thinking budget's cut, it stops thinking: the rest of the block is never said.
struct Script {
    cfg: ModelConfig,
    tokens: Arc<Mutex<VecDeque<u32>>>,
    eos: u32,
    /// `<think>` and `</think>`.
    tags: [u32; 2],
    fed: Arc<Mutex<Vec<u32>>>,
    stall: Arc<Mutex<Option<Duration>>>,
    pace: Arc<Mutex<Duration>>,
    /// The next time the model says the first token, the second as its runner-up, the third
    /// nats below it (`Fake::close_call`).
    runner: Arc<Mutex<Option<(u32, u32, f32)>>>,
    /// Once at most that many scripted tokens are left, `pace` becomes the duration
    /// (`Fake::pace_last`).
    slow: Arc<Mutex<Option<(usize, Duration)>>>,
    /// The bytes an anchor's prompt ends with: the end of the user header, the empty message and
    /// the generation prompt (`anchor_tail`); `None` when the template does not render them.
    anchor_tail: Option<Vec<u8>>,
    /// Each token's bytes, to read a prompt's end.
    pieces: Vec<Vec<u8>>,
    /// The token each KV slot was last written with, so a row's earlier tokens, cached ones too,
    /// can be read back through its block table.
    written: HashMap<i32, u32>,
}

/// Tokens of a KV block, as the engine's scheduler cuts them.
const BLOCK_TOKENS: i32 = 16;

impl Script {
    /// Whether the batch row `r` ends an anchor's prompt: its sequence's tokens up to its position,
    /// read back from the KV slots its block table names, end with `anchor_tail`.
    fn is_anchor(&self, b: &Batch, r: usize) -> bool {
        let Some(tail) = &self.anchor_tail else {
            return false;
        };
        let decode = b.decode_rows;
        let seq = if r < decode {
            r
        } else {
            let mut start = decode;
            let mut seq = None;
            for (j, &len) in b.q_len.iter().enumerate() {
                if r < start + len as usize {
                    seq = Some(decode + j);
                    break;
                }
                start += len as usize;
            }
            match seq {
                Some(seq) => seq,
                None => return false,
            }
        };
        let blocks = &b.block_table[seq * b.max_blocks..][..b.max_blocks];
        let mut bytes = Vec::new();
        let last = b.positions[r];
        // Enough tokens for the tail: each token is at least one byte.
        for pos in (last + 1 - tail.len() as i32).max(0)..=last {
            let block = blocks[(pos / BLOCK_TOKENS) as usize];
            let slot = block * BLOCK_TOKENS + pos % BLOCK_TOKENS;
            let Some(&token) = self.written.get(&slot) else {
                return false;
            };
            bytes.extend(&self.pieces[token as usize]);
        }
        bytes.ends_with(tail)
    }
}

/// What `Script::is_anchor` reads of a step batch.
struct Batch {
    positions: Vec<i32>,
    decode_rows: usize,
    q_len: Vec<i32>,
    block_table: Vec<i32>,
    max_blocks: usize,
}

/// The bytes an anchor's prompt ends with on `template`: the rendering of an empty user message
/// from where a message's text would start, less the two bytes before it, then the generation
/// prompt. A message with text has text there instead; the Llama 3.1 template trims it, and the
/// tool list it puts in the first user message ends with the same two bytes.
fn anchor_tail(template: &ChatTemplate) -> Option<Vec<u8>> {
    let render = |user: &str| {
        let messages = [("system", "rules"), ("user", user)].map(|(role, content)| ToolMessage {
            role: role.into(),
            content: content.into(),
            ..ToolMessage::default()
        });
        template.render_tools(&messages, &[], true).ok()
    };
    let (empty, said) = (render("")?, render("q")?);
    let common = empty
        .bytes()
        .zip(said.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    Some(empty.as_bytes()[common.checked_sub(2)?..].to_vec())
}

fn download(s: &Stream, t: &Tensor) -> Result<Vec<i32>, KError> {
    let mut out = vec![0i32; t.shape().iter().product::<i64>() as usize];
    s.download(t.buffer(), t.offset(), &mut out)?;
    Ok(out)
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
        // The script as it stands when the pass starts: a reply scripted during the pass, such as
        // the one after a dropped turn whose cancelled step is still running, is for later passes.
        let mut tokens = self.tokens.lock().unwrap();
        // Taken after the script, so `wait_stalled` sees the pass holding it.
        let stall = self.stall.lock().unwrap().take();
        if let Some(stall) = stall {
            std::thread::sleep(stall);
        }
        std::thread::sleep(*self.pace.lock().unwrap());
        let mut fed = vec![0i32; b.token_ids.shape()[0] as usize];
        s.download(b.token_ids.buffer(), b.token_ids.offset(), &mut fed)?;
        let fed: Vec<u32> = fed.iter().map(|&t| t as u32).collect();
        let [open, close] = self.tags;
        if fed.contains(&close) {
            let opens = tokens.iter().position(|&t| t == open);
            let closes = tokens.iter().position(|&t| t == close);
            if let Some(end) = closes.filter(|&c| opens.is_none_or(|o| c < o)) {
                tokens.drain(..=end);
            }
        }
        self.fed.lock().unwrap().extend(&fed);
        for (slot, &token) in download(s, &b.slot_ids)?.into_iter().zip(&fed) {
            if slot >= 0 {
                self.written.insert(slot, token);
            }
        }
        let batch = Batch {
            positions: download(s, &b.positions)?,
            decode_rows: b.decode_rows as usize,
            q_len: download(s, &b.prefill_q_len)?,
            block_table: download(s, &b.block_table)?,
            max_blocks: b.block_table.shape()[1] as usize,
        };
        let wanted = download(s, &b.logits_rows)?;
        let rows = b.logits_rows.shape()[0] as usize;
        let vocab = self.cfg.vocab as usize;
        let mut l = vec![0f32; rows * vocab];
        for (row, &r) in l.chunks_mut(vocab).zip(&wanted) {
            if self.is_anchor(&batch, r as usize) {
                row[self.eos as usize] = 100.0;
                continue;
            }
            // An unscripted step ends at once, so a test that scripts too little fails on text.
            let next = tokens.pop_front().unwrap_or(self.eos);
            row[next as usize] = 100.0;
            let mut runner = self.runner.lock().unwrap();
            if let Some((_, id, margin)) = runner.take_if(|(on, ..)| *on == next) {
                row[id as usize] = 100.0 - margin;
            }
            if let Some((_, d)) = self
                .slow
                .lock()
                .unwrap()
                .take_if(|(n, _)| tokens.len() <= *n)
            {
                *self.pace.lock().unwrap() = d;
            }
        }
        s.upload(logits.buffer(), logits.offset(), &l)
    }

    /// A tapped step is the probe's activations request, which these tests run alone before
    /// the turn's step: it takes nothing from the script, and each tapped row reads
    /// `[its last token, the layer, 0, ...]`, as the engine's FakeModel's rows start.
    fn forward_taps(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
        taps: Option<&Taps>,
    ) -> Result<(), KError> {
        let Some(t) = taps else {
            return self.forward(s, b, kv, logits);
        };
        let rows = b.logits_rows.shape()[0] as usize;
        let vocab = self.cfg.vocab as usize;
        let mut l = vec![0f32; rows * vocab];
        for row in l.chunks_mut(vocab) {
            row[self.eos as usize] = 100.0;
        }
        s.upload(logits.buffer(), logits.offset(), &l)?;
        let mut fed = vec![0i32; b.token_ids.shape()[0] as usize];
        s.download(b.token_ids.buffer(), b.token_ids.offset(), &mut fed)?;
        let mut tapped = vec![0i32; t.rows.shape()[0] as usize];
        s.download(t.rows.buffer(), t.rows.offset(), &mut tapped)?;
        let d = self.cfg.d_model as usize;
        let labels: Vec<f32> = (t.layers.iter().map(|&l| f32::from(l)))
            .chain(t.final_state.then_some(self.cfg.n_layers as f32))
            .collect();
        let mut out = vec![0f32; labels.len() * tapped.len() * d];
        for (k, label) in labels.iter().enumerate() {
            for (j, &r) in tapped.iter().enumerate() {
                let row = &mut out[(k * tapped.len() + j) * d..][..d];
                row[0] = fed[r as usize] as f32;
                row[1] = *label;
            }
        }
        s.upload(t.out.buffer(), t.out.offset(), &out)
    }
}

/// The tiny tokenizer with the think tags as user-defined tokens.
fn vocabulary() -> Gguf {
    let mut g = Gguf::parse(&std::fs::read(tiny::TOKENIZER_GGUF).unwrap()).unwrap();
    for (key, add) in [
        (
            "tokenizer.ggml.tokens",
            ["<think>", "</think>"].map(|t| Value::Str(t.into())),
        ),
        ("tokenizer.ggml.token_type", [Value::I32(4), Value::I32(4)]),
    ] {
        let Some(Value::Arr(values)) = g.kv.get_mut(key) else {
            panic!("{key}")
        };
        values.extend(add);
    }
    g
}

/// Blocks the harness's prefix cache may keep: the tiny vocabulary spends about 8,600 tokens on
/// the system prompt with the tool list, so room for both prefixes of two data folders.
pub const CACHE_BLOCKS: usize = 1280;

/// A scripted engine, a data folder and an allowed folder.
pub struct Fake {
    pub larch: Larch,
    /// The engine, for its prefix cache counters.
    pub engine: EngineHandle,
    pub tokenizer: Tokenizer,
    /// The template the engine renders with.
    pub template: ChatTemplate,
    tokens: Arc<Mutex<VecDeque<u32>>>,
    fed: Arc<Mutex<Vec<u32>>>,
    stall: Arc<Mutex<Option<Duration>>>,
    pace: Arc<Mutex<Duration>>,
    runner: Arc<Mutex<Option<(u32, u32, f32)>>>,
    slow: Arc<Mutex<Option<(usize, Duration)>>>,
    pub dir: tempfile::TempDir,
}

impl Fake {
    pub fn new(max_context: u32) -> Fake {
        Fake::with(max_context, vocabulary(), CACHE_BLOCKS, &[])
    }

    /// `new` with the prefix cache off.
    pub fn uncached(max_context: u32) -> Fake {
        Fake::with(max_context, vocabulary(), 0, &[])
    }

    /// `new` with no chat template in the GGUF, so ChatML stands in.
    pub fn chatml(max_context: u32) -> Fake {
        let mut g = vocabulary();
        g.kv.remove("tokenizer.chat_template").unwrap();
        Fake::with(max_context, g, CACHE_BLOCKS, &[])
    }

    /// A Fake with the Qwen3 template and `<tool_call>` and `</tool_call>` as single tokens, as a
    /// Qwen3 vocabulary has them, for the logprob gate on a family that thinks; with the prefix
    /// cache off, as the gate is not about it, so a step sent again feeds its whole prompt.
    pub fn qwen3(max_context: u32) -> Fake {
        let template = include_str!("../../../larch-model/tests/data/qwen3.jinja");
        Fake::qwen3_with(max_context, template)
    }

    /// `qwen3` whose KV pool holds one context and one sequence, as a small Mac's does.
    pub fn qwen3_one_context(max_context: u32) -> Fake {
        let template = include_str!("../../../larch-model/tests/data/qwen3.jinja");
        let one = [(larch_records::knob_id::MAX_BATCH_SEQS, 1.0)];
        Fake::with(max_context, qwen3_gguf(template), 0, &one)
    }

    /// `qwen3` rendering its prompts with `template`.
    pub fn qwen3_with(max_context: u32, template: &str) -> Fake {
        Fake::with(max_context, qwen3_gguf(template), 0, &[])
    }

    /// A Fake that renders its prompts with the chat template `template`.
    pub fn with_template(max_context: u32, template: &str) -> Fake {
        let mut g = vocabulary();
        g.kv.insert(
            "tokenizer.chat_template".into(),
            Value::Str(template.into()),
        );
        Fake::with(max_context, g, CACHE_BLOCKS, &[])
    }

    fn with(max_context: u32, g: Gguf, cache_blocks: usize, knobs: &[(u16, f64)]) -> Fake {
        let tokenizer = Tokenizer::from_gguf(&g).unwrap();
        let tokens = Arc::new(Mutex::new(VecDeque::new()));
        let (fed, stall, pace) = (Arc::default(), Arc::default(), Arc::default());
        let runner: Arc<Mutex<Option<(u32, u32, f32)>>> = Arc::default();
        let slow: Arc<Mutex<Option<(usize, Duration)>>> = Arc::default();
        let template = ChatTemplate::from_gguf(&g, &tokenizer).unwrap();
        let pieces = (0..tokenizer_size(&g))
            .map(|id| tokenizer.decode_piece(id).to_vec())
            .collect();
        let mut cfg = FakeModel::new(tokenizer_size(&g)).config().clone();
        // The session takes several calls in one turn by the architecture.
        if matches!(g.kv.get("general.architecture"), Some(Value::Str(a)) if a == "qwen3") {
            cfg.arch = larch_model::Arch::Qwen3;
        }
        // The tiny vocabulary spends many tokens on the tool list; the engine caps max_context here.
        cfg.ctx_train = 1 << 16;
        let model = Script {
            cfg,
            tokens: tokens.clone(),
            eos: tokenizer.eos().unwrap(),
            tags: ["<think>", "</think>"].map(|t| tokenizer.encode(t, false)[0]),
            fed: Arc::clone(&fed),
            stall: Arc::clone(&stall),
            pace: Arc::clone(&pace),
            runner: Arc::clone(&runner),
            slow: Arc::clone(&slow),
            anchor_tail: anchor_tail(&template),
            pieces,
            written: HashMap::new(),
        };
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = max_context;
        opts.knob_overrides = knobs.to_vec();
        // Without the attest the engine keeps the cache off for a custom model.
        opts.prefix_cache_blocks = cache_blocks;
        opts.prefix_cache_custom_compatible = true;
        let engine = start_with(opts, Box::new(move |_, _, _| Ok(Box::new(model)))).unwrap();
        let larch = Larch::from_engine(
            engine.clone(),
            Tokenizer::from_gguf(&g).unwrap(),
            ChatTemplate::from_gguf(&g, &tokenizer).unwrap(),
        );
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("files")).unwrap();
        Fake {
            larch,
            engine,
            tokenizer,
            template,
            tokens,
            fed,
            stall,
            pace,
            runner,
            slow,
            dir,
        }
    }

    /// Scripts the model's next reply: `text`, then end of turn.
    pub fn say(&self, text: &str) {
        let mut ids = self.tokenizer.encode(text, false);
        ids.push(self.tokenizer.eos().unwrap());
        self.say_tokens(&ids);
    }

    /// Scripts tokens as they are.
    pub fn say_tokens(&self, ids: &[u32]) {
        self.tokens.lock().unwrap().extend(ids);
    }

    /// The next time the model says `on`, gives it the runner-up `id`, `margin` nats below it, so
    /// that token's logprobs show a close call.
    pub fn close_call(&self, on: u32, id: u32, margin: f32) {
        *self.runner.lock().unwrap() = Some((on, id, margin));
    }

    /// The token of `text`, which must be one.
    pub fn token(&self, text: &str) -> u32 {
        let ids = self.tokenizer.encode(text, false);
        assert_eq!(ids.len(), 1, "{text}");
        ids[0]
    }

    /// The token that is the one byte `b`.
    pub fn byte(&self, b: u8) -> u32 {
        (0..)
            .find(|&id| self.tokenizer.decode_piece(id) == [b])
            .unwrap()
    }

    /// Scripted tokens the model has not said yet.
    pub fn unsaid(&self) -> usize {
        self.tokens.lock().unwrap().len()
    }

    /// The scripted tokens the model never said, as text.
    pub fn unsaid_text(&self) -> String {
        let left: Vec<u32> = self.tokens.lock().unwrap().iter().copied().collect();
        self.text_of(&left)
    }

    /// How many tokens the model was fed so far, prompts and its own: a chat that reuses its KV
    /// feeds each token once, so after a turn with no overflow, about the turn's final prompt.
    pub fn fed_tokens(&self) -> usize {
        self.fed.lock().unwrap().len()
    }

    /// Everything the model was fed so far, prompts and its own tokens, as text.
    pub fn fed(&self) -> String {
        self.fed_from(0)
    }

    /// What the model was fed after its first `start` tokens, as text.
    pub fn fed_from(&self, start: usize) -> String {
        let fed = self.fed.lock().unwrap();
        self.text_of(&fed[start..])
    }

    /// `tokens` as text.
    pub fn text_of(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|&t| self.tokenizer.decode_piece(t).to_vec())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Makes the model's next forward pass take at least `d`.
    pub fn stall(&self, d: Duration) {
        *self.stall.lock().unwrap() = Some(d);
    }

    /// Waits until a forward pass has taken the stall `stall` set, so that pass is running now.
    pub fn wait_stalled(&self) {
        while self.stall.lock().unwrap().is_some() {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Makes every forward pass from now on take at least `d`.
    pub fn pace(&self, d: Duration) {
        *self.pace.lock().unwrap() = d;
    }

    /// Makes every forward pass take at least `d` once at most `n` scripted tokens are left: the
    /// engine decodes ahead of a string stop until the stream acknowledges it, so a test that
    /// checks what a stop left unsaid gives the stream that time on a busy machine.
    pub fn pace_last(&self, n: usize, d: Duration) {
        *self.slow.lock().unwrap() = Some((n, d));
    }

    /// Waits until the model has said everything scripted.
    pub fn wait_said(&self) {
        while self.unsaid() > 0 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Waits until every forward pass the engine planned so far has run, with nothing scripted.
    /// A dropped turn cancels its step without waiting for the engine, which sees the cancel
    /// only when it plans a pass, so a pass planned before it can still run after the drop and
    /// take the first token of the reply scripted next. A one-token request is planned after
    /// that pass, so its end means the pass is done. It takes the empty script's end-of-text
    /// and feeds one token, which `fed` keeps.
    pub fn settle(&self) {
        assert_eq!(self.unsaid(), 0, "settle before the next reply is scripted");
        let eos = self.tokenizer.eos().unwrap();
        let fence = Request {
            prefix_cache: false,
            ..Request::new(vec![eos], 1)
        };
        let handle = self.engine.submit(fence).unwrap();
        while let Ok(EngineEvent::Token { .. } | EngineEvent::Prefilled { .. }) =
            handle.events.recv()
        {}
    }

    pub fn data(&self) -> PathBuf {
        self.dir.path().join("data")
    }

    pub fn files(&self) -> PathBuf {
        self.dir.path().join("files")
    }

    pub fn opts(&self) -> AgentOptions {
        AgentOptions {
            data_dir: self.data(),
            allowed: vec![self.files()],
            arch: "llama".into(),
            router: larch_agent::router::Router::Off,
            prefix_cache: self.engine.prefix_cache_stats().unwrap().capacity_blocks > 0,
            memory_capture: false,
        }
    }

    pub fn agent(&self) -> Agent {
        Agent::new(&self.larch, self.opts()).unwrap()
    }
}

impl Fake {
    /// The tokens the engine gets for `messages` (role, content) and `tools`, as the facade renders
    /// a request: with the generation prompt.
    pub fn tokens(&self, messages: &[(&str, &str)], tools: &[serde_json::Value]) -> Vec<u32> {
        let messages: Vec<_> = messages
            .iter()
            .map(|(role, content)| ToolMessage {
                role: (*role).into(),
                content: (*content).into(),
                ..ToolMessage::default()
            })
            .collect();
        let text = self.template.render_tools(&messages, tools, true).unwrap();
        self.tokenizer.encode(&text, true)
    }

    /// The prompt tokens a chat's first turn `user` on `system` with `tools` takes from the
    /// anchor: the tokens it shares with the anchor's prompt, in whole 16-token blocks.
    pub fn anchored(&self, system: &str, tools: &[serde_json::Value], user: &str) -> u32 {
        let anchor = self.tokens(&[("system", system), ("user", "")], tools);
        let turn = self.tokens(&[("system", system), ("user", user)], tools);
        let common = anchor.iter().zip(&turn).take_while(|(a, b)| a == b).count();
        (common / 16 * 16) as u32
    }
}

/// The test vocabulary with the Qwen3 template `template` and its call tags as single tokens.
fn qwen3_gguf(template: &str) -> Gguf {
    let mut g = vocabulary();
    for (key, add) in [
        (
            "tokenizer.ggml.tokens",
            ["<tool_call>", "</tool_call>"].map(|t| Value::Str(t.into())),
        ),
        ("tokenizer.ggml.token_type", [Value::I32(4), Value::I32(4)]),
    ] {
        let Some(Value::Arr(values)) = g.kv.get_mut(key) else {
            panic!("{key}")
        };
        values.extend(add);
    }
    g.kv.insert(
        "tokenizer.chat_template".into(),
        Value::Str(template.into()),
    );
    g.kv.insert("general.architecture".into(), Value::Str("qwen3".into()));
    g
}

/// The tool list of a chat with the default tools, as the model is shown it.
pub fn chat_tools() -> Vec<serde_json::Value> {
    larch_agent::tool::Registry::builtin()
        .specs(&larch_agent::CHAT_TOOLS)
        .iter()
        .map(larch_agent::tool::ToolSpec::function)
        .collect()
}

fn tokenizer_size(g: &Gguf) -> u32 {
    g.get("tokenizer.ggml.tokens")
        .and_then(Value::as_arr)
        .unwrap()
        .len() as u32
}

/// `name` in `folder` as the agent shows a resolved path: the folder's real path, without the
/// `\\?\` Windows puts in front of it.
pub fn shown(folder: &std::path::Path, name: &str) -> String {
    let real = std::fs::canonicalize(folder).unwrap().join(name);
    let real = real.to_str().unwrap();
    real.strip_prefix(r"\\?\").unwrap_or(real).to_owned()
}

/// The URL of a local server that serves `body` as Markdown at every path, as a site would.
pub fn serve(body: &'static str) -> String {
    serve_as("text/markdown", body)
}

/// `serve`, as `kind`.
pub fn serve_as(kind: &'static str, body: impl AsRef<[u8]> + Send + 'static) -> String {
    use std::io::{BufRead, BufReader, Write};
    larch_agent::web::fetch::allow_this_machine();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/essay", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            // The request line and the headers, up to the blank line.
            while reader.read_line(&mut line).unwrap_or(0) > 2 {
                line.clear();
            }
            let body = body.as_ref();
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(&[head.as_bytes(), body].concat());
        }
    });
    url
}

/// A Llama 3 call, as the model writes it.
pub fn call(name: &str, args: serde_json::Value) -> String {
    serde_json::json!({"name": name, "parameters": args}).to_string()
}

/// A Llama 3 call spaced as the template writes it, which the call grammar takes.
pub fn spaced(name: &str, args: serde_json::Value) -> String {
    format!(r#"{{"name": "{name}", "parameters": {args}}}"#)
}

/// Every event of a turn, or of the confirm that continues it, less the empty texts a running
/// ask_docs hands out as time passes.
pub fn events(turn: impl IntoIterator<Item = AgentEvent>) -> Vec<AgentEvent> {
    let ticks = |e: &AgentEvent| matches!(e, AgentEvent::Text(t) if t.is_empty());
    turn.into_iter().filter(|e| !ticks(e)).collect()
}

/// The text events of `events`, joined.
pub fn text(events: &[AgentEvent]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

/// Scripts a Qwen3 call of `tool` as the grammar spells it, for a `Fake::qwen3`; `cut` leaves off
/// its last brace and close tag, which only the grammar then writes (IR-07d), so nothing may be
/// scripted after it.
pub fn qwen3_call(fake: &Fake, tool: &str, args: serde_json::Value, cut: bool) {
    fake.say_tokens(&[fake.token("<tool_call>")]);
    let body = format!("\n{{\"name\": \"{tool}\", \"arguments\": {args}");
    if cut {
        return fake.say(&body);
    }
    let mut ids = fake.tokenizer.encode(&format!("{body}}}\n"), false);
    ids.extend([fake.token("</tool_call>"), fake.tokenizer.eos().unwrap()]);
    fake.say_tokens(&ids);
}

/// Scripts one plan call whose arguments are `args`, its keys in the schema's order, for a
/// `Fake::qwen3`.
pub fn qwen3_plan(fake: &Fake, args: &str) {
    fake.say_tokens(&[fake.token("<tool_call>")]);
    let body = format!("\n{{\"name\": \"plan\", \"arguments\": {args}}}\n");
    let mut ids = fake.tokenizer.encode(&body, false);
    ids.extend([fake.token("</tool_call>"), fake.tokenizer.eos().unwrap()]);
    fake.say_tokens(&ids);
}

/// Code mode's free mode on a `Fake::qwen3`; one worker, so the attempts' replies come one after
/// the other, attempt 0 first. The fake writes no think block, so the steps run as with a budget
/// of 0, and a step's cap is the room of the largest call, 1024 tokens.
pub fn qwen3_code() -> larch_agent::code::r#loop::Code {
    larch_agent::code::r#loop::Code {
        family: larch_agent::parse::Family::Qwen3,
        xml_body: false,
        thinking: larch_agent::config::Thinking {
            budget: 0,
            answer_max_tokens: 0,
        },
        workers: larch_agent::workers::Workers {
            max_workers: 1,
            prefix_cache: false,
        },
        mode: larch_agent::code::r#loop::Mode::Free,
        experts: false,
        verify: larch_agent::code::task::Verify::Full,
    }
}

/// The real model in the GGUF at `path` on `device`, with the desktop app's prefix cache, and
/// code mode for it as `larch-agent code` runs it.
pub fn real(path: &str, device: i32) -> (Larch, larch_agent::code::r#loop::Code) {
    let file = std::fs::File::open(path).unwrap();
    let len = file.metadata().unwrap().len();
    let g = Gguf::read(std::io::BufReader::new(file), len).unwrap();
    let arch = (g.get("general.architecture").and_then(Value::as_str))
        .unwrap()
        .to_owned();
    let xml = (g.get("tokenizer.chat_template").and_then(Value::as_str))
        .is_some_and(larch::toolcall::xml_body);
    drop(g);
    let mut opts = larch::LoadOptions::new(path.into(), device);
    // larch-app's PREFIX_CACHE_BLOCKS.
    opts.prefix_cache_blocks = 256;
    let larch = Larch::load(opts).unwrap();
    let config = larch_agent::config::Config::default();
    let mode = larch_agent::code::r#loop::Mode::Free;
    let code = larch_agent::code::r#loop::Code::new(&arch, xml, &config, true, mode).unwrap();
    (larch, code)
}

/// Makes `dir` a repository whose one commit holds its files, an empty commit when it has none,
/// with git isolated from the host's identity and signing.
pub fn commit(dir: &std::path::Path) {
    for args in [
        &["init", "-q"][..],
        &["add", "."],
        &["commit", "-q", "--allow-empty", "-m", "start"],
    ] {
        let out = std::process::Command::new("git")
            .args([
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
            ])
            .args([
                "-c",
                "commit.gpgsign=false",
                "-c",
                "init.defaultBranch=main",
            ])
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}: {out:?}");
    }
}
