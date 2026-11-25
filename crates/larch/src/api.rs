//! Synchronous public facade over the merged engine and model tokenizer.

use std::{
    collections::HashMap,
    fs::File,
    io::BufReader,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use larch_engine::grammar::{Compiled, Constraint};
use larch_engine::{
    DOC_CONTEXT_STEP, DOC_POOL_FRACTION, EngineEvent, EngineHandle, ImageSpan, ReloadChange,
    Request, RequestHandle, SamplingParams,
};
use larch_model::projector::{ImagePixels, ProjectorConfig};
use larch_model::vision::{Image, image_tokens};
use larch_model::{ChatTemplate, Detokenizer, Message, ModelConfig, MoeCfg, Tokenizer, gguf::Gguf};
use larch_records::{
    DeploymentSnapshot, Health, HealthState, KnobControl, RejectReason, RingHandle, knob_id,
};

use crate::{Pin, Session, SessionOptions, TextStream};

pub use larch_engine::EngineOptions as LoadOptions;
pub use larch_engine::grammar::{self, Grammar, GrammarError, json_schema_regex};
pub use larch_engine::{
    Class, DocFit, DocumentsPlan, ModelClass, RouterDeltas, TURN_ROOM, invalid,
};
pub use larch_model::ToolMessage;
pub use larch_records::{EngineError, LoadError};

/// A prompt supplied as text, a complete conversation, or already encoded tokens.
#[derive(Clone, Debug)]
pub enum Prompt {
    /// Plain text, with the tokenizer's configured special tokens.
    Text(String),
    /// Whole conversation, rendered with an assistant generation header.
    Messages(Vec<Message>),
    /// Token IDs, passed to the engine unchanged.
    Tokens(Vec<u32>),
}

/// Request options; unset sampling values come from the current effective configuration, except a
/// score request's repetition penalty (`score_ids`).
#[derive(Clone, Debug)]
pub struct GenParams {
    /// Maximum generated tokens, including a terminal token; 0 runs until the model ends its
    /// answer or the context fills.
    pub max_tokens: u32,
    /// Temperature; zero selects greedy sampling.
    pub temperature: Option<f32>,
    /// Nucleus probability.
    pub top_p: Option<f32>,
    /// Candidate count; zero includes all engine candidates.
    pub top_k: Option<u32>,
    /// Repetition penalty.
    pub repeat_penalty: Option<f32>,
    /// Random seed; unset lets the engine use the request ID.
    pub seed: Option<u64>,
    /// Caller-side string stops, matched by the model's Detokenizer.
    pub stop: Vec<String>,
    /// Top logprobs per token, at most twenty.
    pub logprobs: u8,
    /// Allow automatic sharing if LoadOptions explicitly enables global retention.
    pub prefix_cache: bool,
    /// Requests in different trust namespaces never share automatic prefixes.
    pub cache_namespace: u64,
    /// Report prefill progress as [`TextEvent::Prefilled`]: one per chunk of the prompt read.
    pub progress: bool,
    /// Tool specs in the OpenAI function shape, rendered by the chat template on every render of
    /// this request (a session's turn, or `Prompt::Messages`); empty is none.
    // ponytail: on GenParams until #138 lands; then move to SessionOptions as the task block names.
    pub tools: Vec<serde_json::Value>,
    /// Withhold `<think>` blocks, and the blank lines right after one, from text events and
    /// history; string stops never match inside.
    // ponytail: on GenParams until #138 lands; then move to SessionOptions as the task block names.
    pub hide_thinking: bool,
    /// With `hide_thinking`, also yield the withheld blocks' text as it comes, as
    /// [`TextEvent::Thinking`], for a caller that shows the reasoning apart from the answer.
    pub stream_thinking: bool,
    /// The chat template's `enable_thinking` on every render of this request; None is the
    /// template's default (on). `Some(false)` renders Qwen3's empty think block after the
    /// generation prompt, so the model answers at once.
    pub thinking: Option<bool>,
    /// The most tokens a think block may take: the first request stops after this many, and when
    /// it stops inside `<think>` the stream feeds [`THINKING_CLOSE`](crate::text::THINKING_CLOSE)
    /// and continues in the same session with the rest of `max_tokens` (`Timings::thinking_cut`).
    /// A cap at or above `max_tokens` changes nothing; None is no budget; 0 is refused.
    pub thinking_budget: Option<u32>,
    /// Token ids to score, at most 256: each token's logprobs `top` holds them with their exact
    /// full-vocabulary logprobs, in this order, in place of the most likely ones, whatever
    /// `logprobs` asks. An unset `repeat_penalty` is then 1, not the configured default, since the
    /// scores are of the raw logits, and any other is refused. Empty is none.
    ///
    /// With `thinking` unset the template renders with thinking on, so on a thinking model the
    /// first token of a `Prompt::Messages` or session turn opens a think block instead of
    /// answering. Score such a model with `thinking: Some(false)`, which closes the block in the
    /// prompt.
    pub score_ids: Vec<u32>,
    /// Foreground for a person's turn, Background for work nobody watches token by token; the
    /// engine starts foreground prompts first ([`Class`]).
    pub class: Class,
    /// Sample only output the grammar allows ([`Grammar`]), e.g. a tool call or a JSON answer
    /// ([`json_schema_regex`]); None leaves sampling as it is. An untriggered grammar constrains
    /// the first token, so a thinking model needs `thinking: Some(false)`. Refused with
    /// `score_ids`; a bad or empty pattern is `InvalidRequest(GRAMMAR_PATTERN)`, one past the
    /// DFA size limit `InvalidRequest(GRAMMAR_SIZE)`.
    pub grammar: Option<Grammar>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_tokens: 0,
            temperature: None,
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            seed: None,
            stop: Vec::new(),
            logprobs: 0,
            prefix_cache: true,
            cache_namespace: 0,
            progress: false,
            tools: Vec::new(),
            hide_thinking: true,
            stream_thinking: false,
            thinking: None,
            thinking_budget: None,
            score_ids: Vec::new(),
            class: Class::Foreground,
            grammar: None,
        }
    }
}

/// The character a message's content holds where an image goes, for [`Runtime::encode_images`]:
/// Unicode's own placeholder for an embedded object.
pub(crate) const IMAGE_MARK: char = '\u{FFFC}';

/// What [`describe`] asks for: a note an older image shows as once it leaves the pixel budget, so a
/// later question can still be answered from it (and a text-only model can see it).
pub const DESCRIBE_PROMPT: &str = "Describe this image so that someone who cannot see it could \
answer questions about it. Copy any readable text exactly, and mention the layout, names, numbers \
and code it shows. Use at most 250 words.";
/// [`describe`]'s answer cap when the caller sets none: 250 words and some room. A text-dense
/// screenshot needs it, and it is still a fraction of the image's 1,792 rows.
pub const DESCRIBE_TOKENS: u32 = 400;

/// Asks for a short description of `image` (see [`DESCRIBE_PROMPT`]) as a background request outside
/// any history: it reuses the engine's encoded tiles when the image was just read, and yields to
/// foreground turns. The image is cut here, on the caller's thread.
pub(crate) fn describe(
    rt: &Arc<Runtime>,
    image: &Arc<Image>,
    mut p: GenParams,
) -> Result<TextStream, EngineError> {
    let v = (rt.engine.vision()).ok_or(EngineError::InvalidRequest(invalid::NO_VISION))?;
    let rgb = (image.decode()).map_err(|_| EngineError::InvalidRequest(invalid::IMAGE))?;
    let pixels = Arc::new(v.cut(&rgb));
    let messages = [ToolMessage {
        role: "user".into(),
        content: format!("{IMAGE_MARK}\n{DESCRIBE_PROMPT}"),
        tool_calls: Vec::new(),
        tool_call_id: String::new(),
    }];
    let (tokens, spans) = rt.encode_images(
        &messages,
        &[],
        Some(false),
        &[(image.clone(), Some(pixels))],
    )?;
    if p.max_tokens == 0 {
        p.max_tokens = DESCRIBE_TOKENS;
    }
    p.tools = Vec::new();
    let mut request = rt.request(tokens, &p);
    request.images = spans;
    request.class = Class::Background;
    let opens = rt.opens_thinking(&request.prompt);
    let (handle, next) = rt.submit(request, &p)?;
    Ok(TextStream::new(
        rt.clone(),
        handle,
        &p,
        None,
        0,
        next,
        opens,
    ))
}

pub(crate) struct Runtime {
    pub engine: EngineHandle,
    pub knobs: Arc<dyn KnobControl>,
    pub tokenizer: Arc<Tokenizer>,
    pub template: Arc<ChatTemplate>,
    pub tuner: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// Each token's bytes as the output text shows them (none for a control token), built on
    /// the first grammar.
    pub vocab: OnceLock<Arc<[Vec<u8>]>>,
    /// Compiled grammars by pattern: an agent sends the same tools every turn.
    pub grammars: Mutex<HashMap<String, Arc<Compiled>>>,
    /// The GGUF `load` read; empty for [`Larch::from_engine`].
    pub model_path: PathBuf,
    /// (layers, experts) of an MoE GGUF that `load` read; None otherwise.
    pub moe: Option<(u16, u32)>,
    /// Images cut for the encoder ahead of their turn ([`Larch::prepare_image`]), oldest first, by
    /// digest: the turn that sends one takes its pixels here instead of decoding it again.
    pub prepared: Mutex<Vec<([u8; 32], Arc<ImagePixels>)>>,
    /// Milliseconds the last model load or reload took; 0 before one was measured.
    pub load_ms: AtomicU64,
}

/// The images [`Runtime::prepared`] keeps: a message carries at most 8, and one is about 8 MB.
const PREPARED_IMAGES: usize = 8;

impl Drop for Runtime {
    fn drop(&mut self) {
        // The tuner first, so a trial gait closes within its stop's grace is undone on a live
        // engine; a longer one is cut off with the engine, which gait's journal recovery covers
        // (12.6). A panicking stop still shuts the engine down, then propagates unless already
        // unwinding.
        let slot = self.tuner.get_mut().unwrap_or_else(PoisonError::into_inner);
        let stopped = slot.take().map(|stop| catch_unwind(AssertUnwindSafe(stop)));
        self.engine.clone().shutdown();
        if let Some(Err(panic)) = stopped
            && !std::thread::panicking()
        {
            resume_unwind(panic);
        }
    }
}

/// `messages` as the chat template renders them, without tool calls.
pub(crate) fn tool_messages(messages: Vec<Message>) -> Vec<ToolMessage> {
    (messages.into_iter())
        .map(|m| ToolMessage {
            role: m.role,
            content: m.content,
            ..ToolMessage::default()
        })
        .collect()
}

impl Runtime {
    /// `image` cut for the encoder: as [`Larch::prepare_image`] cut it (for this projector), or
    /// now, on the caller's thread. `IMAGE` when it does not decode.
    pub fn cut(&self, image: &Image, v: &ProjectorConfig) -> Result<Arc<ImagePixels>, EngineError> {
        let rows = v.tokens_for(image.width(), image.height());
        let held = (self
            .prepared
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter())
        .find(|(d, px)| d == image.digest() && px.fits(v, rows))
        .map(|(_, px)| px.clone());
        if let Some(px) = held {
            return Ok(px);
        }
        let rgb = image
            .decode()
            .map_err(|_| EngineError::InvalidRequest(invalid::IMAGE))?;
        Ok(Arc::new(v.cut(&rgb)))
    }

    /// Tokenizes a prompt as [`Larch::generate`] does; messages render with `tools` and
    /// `thinking` (`GenParams::thinking`).
    pub fn encode_prompt(
        &self,
        prompt: Prompt,
        tools: &[serde_json::Value],
        thinking: Option<bool>,
    ) -> Result<Vec<u32>, EngineError> {
        Ok(match prompt {
            Prompt::Text(text) => self.tokenizer.encode(&text, true),
            Prompt::Messages(messages) => {
                self.encode_messages(&tool_messages(messages), tools, thinking)?
            }
            Prompt::Tokens(tokens) => tokens,
        })
    }

    pub fn encode_messages(
        &self,
        messages: &[ToolMessage],
        tools: &[serde_json::Value],
        thinking: Option<bool>,
    ) -> Result<Vec<u32>, EngineError> {
        let rendered = self
            .template
            .render_tools_thinking(messages, tools, true, thinking)
            .map_err(|_| EngineError::InvalidRequest(invalid::CHAT_TEMPLATE))?;
        Ok(self.tokenizer.encode(&rendered, true))
    }

    /// [`Runtime::encode_messages`] for messages holding [`IMAGE_MARK`]s, one per image of `images`,
    /// in order: each mark becomes the projector's wrapper around the image's rows (llama.cpp's mtmd
    /// split, the text either side tokenised on its own), whose ids and place the spans name.
    pub fn encode_images(
        &self,
        messages: &[ToolMessage],
        tools: &[serde_json::Value],
        thinking: Option<bool>,
        images: &[(Arc<Image>, Option<Arc<ImagePixels>>)],
    ) -> Result<(Vec<u32>, Vec<ImageSpan>), EngineError> {
        let v = (self.engine.vision()).ok_or(EngineError::InvalidRequest(invalid::NO_VISION))?;
        let rendered = self
            .template
            .render_tools_thinking(messages, tools, true, thinking)
            .map_err(|_| EngineError::InvalidRequest(invalid::CHAT_TEMPLATE))?;
        let parts: Vec<&str> = rendered.split(IMAGE_MARK).collect();
        // A template that dropped or repeated a message's content cannot place its images.
        if parts.len() != images.len() + 1 {
            return Err(EngineError::InvalidRequest(invalid::CHAT_TEMPLATE));
        }
        let (mut tokens, mut spans) = (Vec::new(), Vec::new());
        let (begin, end) = v.markers();
        for (i, part) in parts.iter().enumerate() {
            let before = if i == 0 { "" } else { end };
            let after = if i < images.len() { begin } else { "" };
            tokens.extend(
                self.tokenizer
                    .encode(&format!("{before}{part}{after}"), i == 0),
            );
            if let Some((image, pixels)) = images.get(i) {
                let rows = v.tokens_for(image.width(), image.height());
                spans.push(ImageSpan {
                    at: tokens.len() as u32,
                    rows,
                    tile_rows: v.unit_rows(image.width(), image.height()),
                    image: image.clone(),
                    pixels: pixels.clone(),
                });
                tokens.extend(image_tokens(image.digest(), rows));
            }
        }
        Ok((tokens, spans))
    }

    /// The length of `prompt`, rendered from `messages`, without its generation prompt: the chat
    /// history a session's next prompt repeats (`Request::history_end`). 0 when the template does
    /// not end `prompt` with the generation prompt alone.
    pub fn history_len(
        &self,
        messages: &[ToolMessage],
        tools: &[serde_json::Value],
        thinking: Option<bool>,
        prompt: &[u32],
    ) -> u32 {
        let render = |generation| {
            self.template
                .render_tools_thinking(messages, tools, generation, thinking)
        };
        let (Ok(full), Ok(history)) = (render(true), render(false)) else {
            return 0;
        };
        let Some(suffix) = full.strip_prefix(history.as_str()) else {
            return 0;
        };
        let suffix = self.tokenizer.encode(suffix, false);
        prompt
            .strip_suffix(&suffix[..])
            .map_or(0, |h| h.len() as u32)
    }

    /// Whether `prompt` ends inside a `<think>` block it opened, as Qwen 3.5's generation prompt
    /// does: the reply then starts in the block.
    pub fn opens_thinking(&self, prompt: &[u32]) -> bool {
        (prompt.iter().rev())
            .map(|&id| self.tokenizer.decode_piece(id))
            .find(|piece| !piece.iter().all(|&b| b == b'\n'))
            .is_some_and(|piece| piece == b"<think>")
    }

    /// Submits `request`, capped at `p.thinking_budget` when that is below its max_tokens, and
    /// returns the request a budget cut continues (`TextStream`).
    // ponytail: the continuation is a new request, so a seeded sampler restarts its stream there,
    // even when the block closed before the budget. Upgrade: carry the RNG state across when a
    // caller samples above temperature 0 with a seed and a budget.
    pub fn submit(
        &self,
        mut request: Request,
        p: &GenParams,
    ) -> Result<(RequestHandle, Option<Request>), EngineError> {
        // Compiled here, the first fallible step every path takes, so `request` stays infallible.
        if let Some(g) = &p.grammar {
            request.grammar = Some(Constraint {
                trigger: g.trigger.clone(),
                compiled: self.compiled(&g.pattern)?,
                replay: Vec::new(),
            });
        }
        // ponytail: on a hybrid model a continuation reruns the thinking from the history end,
        // where the first request parks its state; park the turn's end too if a Qwen 3.5
        // thinking-budget bench shows the rerun.
        let next = match p.thinking_budget {
            Some(budget) if budget < request.max_tokens => {
                let next = request.clone();
                request.max_tokens = budget;
                Some(next)
            }
            _ => None,
        };
        Ok((self.engine.submit(request)?, next))
    }

    fn compiled(&self, pattern: &str) -> Result<Arc<Compiled>, EngineError> {
        let vocab = self.vocab.get_or_init(|| {
            (0..self.engine.deployment().vocab)
                .map(|id| {
                    // ponytail: a Detokenizer per token tells shown from hidden (larch-model keeps
                    // `hidden` private); a public visible-piece accessor there replaces this.
                    let mut d = Detokenizer::new(Vec::new());
                    let shown = d.push(&self.tokenizer, id).0 + &d.finish();
                    match shown.is_empty() {
                        true => Vec::new(),
                        false => self.tokenizer.decode_piece(id).to_vec(),
                    }
                })
                .collect::<Vec<_>>()
                .into()
        });
        let cache = || self.grammars.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(c) = cache().get(pattern) {
            return Ok(c.clone());
        }
        // Compiled unlocked: a large pattern takes a while, and other requests' hits must not wait.
        let c = Arc::new(Compiled::new(pattern, vocab.clone()).map_err(|e| {
            EngineError::InvalidRequest(match e {
                GrammarError::TooLarge => invalid::GRAMMAR_SIZE,
                _ => invalid::GRAMMAR_PATTERN,
            })
        })?);
        // ponytail: dropped whole past 64 patterns; an LRU when a server's callers rotate more.
        let mut cache = cache();
        if cache.len() >= 64 {
            cache.clear();
        }
        cache.insert(pattern.into(), c.clone());
        Ok(c)
    }

    pub fn request(&self, prompt: Vec<u32>, p: &GenParams) -> Request {
        // An uncapped answer gets the rest of the context; a prompt that fills it still gets 1,
        // so the engine reports PromptTooLong rather than NO_TOKENS.
        let max_tokens = match p.max_tokens {
            0 => (u64::from(self.engine.deployment().ctx_max))
                .saturating_sub(prompt.len() as u64)
                .max(1) as u32,
            n => n,
        };
        let cfg = self.knobs.effective();
        let value = |id: u16| cfg.values[id as usize] as f32;
        Request {
            prompt,
            images: Vec::new(),
            sampling: SamplingParams {
                temperature: p
                    .temperature
                    .unwrap_or_else(|| value(knob_id::TEMPERATURE_DEFAULT)),
                top_p: p.top_p.unwrap_or_else(|| value(knob_id::TOP_P_DEFAULT)),
                top_k: p
                    .top_k
                    .unwrap_or(cfg.values[knob_id::TOP_K_DEFAULT as usize] as u32),
                // Scores are of the raw logits, so a score request's unset penalty is none.
                repeat_penalty: p.repeat_penalty.unwrap_or_else(|| {
                    if p.score_ids.is_empty() {
                        value(knob_id::REPEAT_PENALTY_DEFAULT)
                    } else {
                        1.0
                    }
                }),
                seed: p.seed,
            },
            max_tokens,
            stop_tokens: self.tokenizer.eog().to_vec(),
            logprobs: p.logprobs,
            ignore_eos: false,
            session: 0,
            keep_kv: false,
            prefix_cache: p.prefix_cache,
            cache_namespace: p.cache_namespace,
            history_end: 0,
            score_ids: p.score_ids.clone(),
            activations: Vec::new(),
            activation_rows: Vec::new(),
            class: p.class,
            progress: p.progress,
            grammar: None,
        }
    }
}

/// A loaded engine, shared by its sessions and streams until the last owner drops. A clone is
/// one more owner of the same engine, as a `Session` is.
#[derive(Clone)]
pub struct Larch {
    pub(crate) runtime: Arc<Runtime>,
}

impl Larch {
    /// Loads the model and tokenizer and returns only after the engine reaches Ready.
    /// The tokenizer and template are read first, so a bad header fails before the weights load.
    pub fn load(opts: LoadOptions) -> Result<Self, LoadError> {
        let (tokenizer, template, moe) = {
            let file = File::open(&opts.model).map_err(|_| LoadError::Io)?;
            let len = file.metadata().map_err(|_| LoadError::Io)?.len();
            let g = Gguf::read(BufReader::new(file), len)?; // dropped before the weights load
            let tokenizer = Tokenizer::from_gguf(&g)?;
            let template = ChatTemplate::from_gguf(&g, &tokenizer)?;
            let cfg = ModelConfig::from_gguf(&g)?;
            let moe = MoeCfg::from_gguf(&g, cfg.arch)?.map(|m| (cfg.n_layers as u16, m.experts));
            (tokenizer, template, moe)
        };
        let path = opts.model.clone();
        let began = Instant::now();
        let mut v = Self::from_engine(larch_engine::start(opts)?, tokenizer, template);
        let rt = Arc::get_mut(&mut v.runtime).expect("not yet shared");
        (rt.model_path, rt.moe) = (path, moe);
        *rt.load_ms.get_mut() = began.elapsed().as_millis() as u64;
        Ok(v)
    }

    /// Takes shutdown ownership: the last facade/session/stream drops the engine, even with external clones.
    pub fn from_engine(engine: EngineHandle, tokenizer: Tokenizer, template: ChatTemplate) -> Self {
        Self {
            runtime: Arc::new(Runtime {
                knobs: engine.knobs(),
                engine,
                tokenizer: Arc::new(tokenizer),
                template: Arc::new(template),
                tuner: Mutex::new(None),
                vocab: OnceLock::new(),
                grammars: Mutex::new(HashMap::new()),
                model_path: PathBuf::new(),
                moe: None,
                prepared: Mutex::new(Vec::new()),
                load_ms: AtomicU64::new(0),
            }),
        }
    }

    /// Whether this engine reads images: a vision projector was loaded beside the model.
    pub fn sees_images(&self) -> bool {
        self.runtime.engine.vision().is_some()
    }

    /// The prompt rows `image` takes on this engine (InternVL: up to 1,792 at 6 tiles; Qwen: up to
    /// 1,600); None without a projector.
    pub fn image_rows(&self, image: &Image) -> Option<u32> {
        (self.runtime.engine.vision()).map(|v| v.tokens_for(image.width(), image.height()))
    }

    /// Encodes `image` while the engine has nothing else to run (S5, encode on paste), so the turn
    /// that sends it reads its rows from the engine's tile cache instead of encoding them then:
    /// call it as the image is attached. It is cut here, on the caller's thread, and the cut is
    /// kept for that turn. `NO_VISION` without a projector, `IMAGE` when it does not decode.
    pub fn prepare_image(&self, image: &Image) -> Result<(), EngineError> {
        let Some(v) = self.runtime.engine.vision() else {
            return Err(EngineError::InvalidRequest(invalid::NO_VISION));
        };
        let px = self.runtime.cut(image, &v)?;
        self.runtime.engine.prepare_image(image, px.clone())?;
        let mut held = (self.runtime.prepared.lock()).unwrap_or_else(PoisonError::into_inner);
        held.retain(|(d, _)| d != image.digest());
        held.push((*image.digest(), px));
        if held.len() > PREPARED_IMAGES {
            held.remove(0);
        }
        Ok(())
    }

    /// Image tiles the engine has encoded since it started, for a turn or ahead of one; a tile its
    /// cache held is not counted.
    pub fn image_tiles_encoded(&self) -> u64 {
        self.runtime.engine.tiles_encoded()
    }

    /// Creates an independent conversation with a fresh session ID.
    pub fn session(&self, opts: SessionOptions) -> Session {
        Session::new(self.runtime.clone(), opts)
    }

    /// Generates without a session; the engine never parks this request's KV.
    pub fn generate(&self, prompt: Prompt, p: GenParams) -> Result<TextStream, EngineError> {
        let tokens = self.runtime.encode_prompt(prompt, &p.tools, p.thinking)?;
        let request = self.runtime.request(tokens, &p);
        let opens = self.runtime.opens_thinking(&request.prompt);
        let (handle, next) = self.runtime.submit(request, &p)?;
        Ok(TextStream::new(
            self.runtime.clone(),
            handle,
            &p,
            None,
            0,
            next,
            opens,
        ))
    }

    /// What the documents plan holds for a document of `tokens` on this device, without loading
    /// ([`DocFit`]); `max_tokens` is the longest a pin can be, a turn's room
    /// ([`TURN_ROOM`]) taken from the largest context the device holds. An engine that cannot
    /// say (not ready, a model not from a file) holds nothing: `fits` false, 0 tokens.
    pub fn document_fit(&self, tokens: u32) -> DocFit {
        self.runtime.engine.document_fit(tokens).unwrap_or(DocFit {
            fits: false,
            max_tokens: 0,
            host_layers: 0,
            seats: 0,
        })
    }

    /// Whether [`Self::pin`] (or [`Self::import_pin`]) of a document of `tokens` would reload the
    /// engine for the documents plan: it fits that plan, but not the loaded context with a turn's
    /// room ([`TURN_ROOM`]). A document too long for the plan is refused, not reloaded: false.
    /// Unstable: may change before 1.0.
    pub fn pin_reloads(&self, tokens: u32) -> bool {
        self.document_fit(tokens).fits
            && tokens.saturating_add(TURN_ROOM) > self.runtime.engine.deployment().ctx_max
    }

    /// Seconds, rounded up, the last model load or reload took (a reload for a document is one);
    /// 0 before one was measured. Unstable: may change before 1.0.
    pub fn load_secs(&self) -> u32 {
        let ms = self.runtime.load_ms.load(Ordering::Relaxed);
        u32::try_from(ms.div_ceil(1000)).unwrap_or(u32::MAX)
    }

    /// Keeps `tokens`, a document's prompt, in the prefix cache of `namespace` until the returned
    /// [`Pin`] drops, through a same-model reload, and reads it in as a background request: the
    /// stream (with [`GenParams::progress`], so it reports [`TextEvent::Prefilled`]) ends when the document is cached, and a
    /// stream dropped before then withdraws the part read. A document the plan cannot hold is
    /// `InvalidRequest(DOC_TOO_LONG)`, whose `document_fit(..).max_tokens` says how long one may
    /// be (the longest pin is that, not the context's 131,072: a turn's room is kept out);
    /// an engine without a prefix cache is `InvalidRequest(NO_PREFIX_CACHE)`. The pins held with
    /// [`TURN_ROOM`] must fit the KV pool: one that would pass that is
    /// `InvalidRequest(PINS_FULL)`; drop (after exporting) the pins the active chat no longer
    /// needs. A pin never fails or holds up another request: one that needs the blocks takes the
    /// oldest pins cold first ([`Self::pin_live`]), and a question on that document reads it
    /// again. Unstable: may change before 1.0.
    ///
    /// When the loaded context is too small for the document and a turn, this reloads once
    /// for the documents plan: the context rounded up to a step, the pool at 95% of the memory
    /// left, the seats `document_fit` picks (two at least, more where the document still fits)
    /// and no drafter, with the other pins and parked sessions carried.
    pub fn pin(&self, tokens: Vec<u32>, namespace: u64) -> Result<(Pin, TextStream), EngineError> {
        let engine = &self.runtime.engine;
        self.require_prefix_cache()?;
        let n = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        self.prepare_document(n)?;
        // After the reload, which may have made the pool this pin must fit.
        let hold = match engine.pin(tokens.clone(), namespace) {
            Ok(hold) if !self.pin_reloads(n) => hold,
            Err(e) if !self.pin_reloads(n) => return Err(e),
            // The documents plan went back between the check and the pin (an idle engine, or a
            // request that needed its seat): the plan it left is too small, so reload once more.
            _ => {
                self.prepare_document(n)?;
                engine.pin(tokens.clone(), namespace)?
            }
        };
        let p = GenParams {
            max_tokens: 1,
            class: Class::Background,
            prefix_cache: true,
            cache_namespace: namespace,
            progress: true,
            ..GenParams::default()
        };
        let read = self.generate(Prompt::Tokens(tokens.clone()), p)?;
        Ok((Pin::new(hold, tokens), read))
    }

    /// Nothing could keep a document without the shared prefix cache: refused before any reload.
    pub(crate) fn require_prefix_cache(&self) -> Result<(), EngineError> {
        match self.runtime.engine.prefix_cache_stats() {
            Some(s) if s.capacity_blocks > 0 => Ok(()),
            _ => Err(EngineError::InvalidRequest(invalid::NO_PREFIX_CACHE)),
        }
    }

    /// When the loaded context is too small for a document of `n` tokens and a turn, checks that
    /// it fits the documents plan and reloads once for the plan ([`Self::pin`]). A context that
    /// already holds it needs no plan: the pin's own pool check (`PINS_FULL`) is the one that
    /// counts there, so a re-pin inside the plan is not refused by a second fit.
    pub(crate) fn prepare_document(&self, n: u32) -> Result<(), EngineError> {
        let engine = &self.runtime.engine;
        let wanted = n.saturating_add(TURN_ROOM);
        if wanted <= engine.deployment().ctx_max {
            return Ok(());
        }
        let fit = self.document_fit(n);
        if !fit.fits {
            return Err(EngineError::InvalidRequest(invalid::DOC_TOO_LONG));
        }
        let began = Instant::now();
        let most = larch_records::knob_spec(knob_id::MAX_CONTEXT)
            .expect("registered")
            .max;
        engine.reload(ReloadChange {
            max_context: Some(wanted.next_multiple_of(DOC_CONTEXT_STEP).min(most as u32)),
            kv_pool_fraction: Some(DOC_POOL_FRACTION),
            max_batch_seqs: Some(fit.seats),
            // ponytail: drafter memory is not in the fit, so the plan runs without one; keep
            // it on when `document_fit` counts the drafter's caches at `fit.seats`.
            drafter_off: true,
            documents: true,
            ..ReloadChange::default()
        })?;
        let ms = began.elapsed().as_millis() as u64;
        self.runtime.load_ms.store(ms, Ordering::Relaxed);
        Ok(())
    }

    /// `messages` rendered by the chat template with `tools` and the generation prompt, and
    /// tokenised as a request would be: the ids the engine sees for a turn, so a caller can find
    /// the prefix two renders share and pin it ([`Self::pin`]). `thinking` is the template's
    /// `enable_thinking` (None: its default).
    pub fn encode_messages(
        &self,
        messages: &[ToolMessage],
        tools: &[serde_json::Value],
        thinking: Option<bool>,
    ) -> Result<Vec<u32>, EngineError> {
        self.runtime.encode_messages(messages, tools, thinking)
    }

    /// The residual stream of the prompt's last token after each of `layers` (0-based), in that
    /// order, as one f32 row of `d_model` per layer: one prefill, nothing sampled, nothing parked.
    /// The prompt renders as in [`Self::generate`] with no tools and the template's thinking
    /// default. A layer past the model's depth, or more layers than it has, is
    /// `InvalidRequest(ACTIVATION_LAYER)`; no layers is `InvalidRequest(NO_TOKENS)`.
    pub fn activations(
        &self,
        prompt: Prompt,
        layers: &[u16],
    ) -> Result<Vec<Vec<f32>>, EngineError> {
        let tokens = self.runtime.encode_prompt(prompt, &[], None)?;
        let request = Request {
            activations: layers.to_vec(),
            ..Request::new(tokens, 0)
        };
        let handle = self.runtime.engine.submit(request)?;
        for event in handle.events.iter() {
            match event {
                EngineEvent::Token { .. } | EngineEvent::Prefilled { .. } => {}
                EngineEvent::Done { activations, .. } => return Ok(activations),
                EngineEvent::Error(e) => return Err(e),
            }
        }
        Err(EngineError::NotReady(HealthState::Failed)) // the engine thread is gone
    }

    /// The tokens `prompt` takes as [`Larch::generate`] renders it, with `tools` and `thinking`
    /// (`GenParams::tools`, `GenParams::thinking`): what the engine counts against its context.
    pub fn prompt_len(
        &self,
        prompt: Prompt,
        tools: &[serde_json::Value],
        thinking: Option<bool>,
    ) -> Result<usize, EngineError> {
        Ok(self.runtime.encode_prompt(prompt, tools, thinking)?.len())
    }

    /// The residual stream entering `layer` (the output of layer `layer - 1`) at each of `rows`
    /// of one prefill of `tokens`, in order, under the adaptation version in force: nothing
    /// sampled or parked. Layer 0 (the embeddings) and a layer past the depth are
    /// `InvalidRequest(ACTIVATION_LAYER)`; a row past the tokens is `InvalidRequest(NO_TOKENS)`.
    /// One tapped prefill serves all rows (`Request::activation_rows`).
    pub fn hidden_at(
        &self,
        tokens: Vec<u32>,
        layer: u16,
        rows: Vec<u32>,
    ) -> Result<Vec<Vec<f32>>, EngineError> {
        let Some(tap) = layer.checked_sub(1) else {
            return Err(EngineError::InvalidRequest(invalid::ACTIVATION_LAYER));
        };
        if rows.iter().any(|&r| r as usize >= tokens.len()) {
            return Err(EngineError::InvalidRequest(invalid::NO_TOKENS));
        }
        // The engine taps strictly increasing positions; ask for the sorted set, answer in `rows`' order.
        let mut sorted = rows.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.is_empty() {
            return Ok(Vec::new());
        }
        let request = Request {
            activations: vec![tap],
            activation_rows: sorted.clone(),
            prefix_cache: false,
            ..Request::new(tokens, 0)
        };
        let handle = self.runtime.engine.submit(request)?;
        let got = match handle.events.iter().last() {
            Some(EngineEvent::Done { activations, .. }) => activations,
            Some(EngineEvent::Error(e)) => return Err(e),
            _ => return Err(EngineError::NotReady(HealthState::Failed)), // the engine thread is gone
        };
        Ok(rows
            .iter()
            .map(|r| got[sorted.binary_search(r).expect("asked row")].clone())
            .collect())
    }

    /// Serves MoE router deltas as adaptation `version` (`None` with 0: the base model), from the
    /// first step boundary where no started request remains; returns once applied
    /// ([`EngineHandle::set_router_deltas`]).
    pub fn set_router_deltas(
        &self,
        deltas: Option<RouterDeltas>,
        version: u64,
    ) -> Result<(), EngineError> {
        self.runtime.engine.set_router_deltas(deltas, version)
    }

    /// Runs `f` on the resident transformer and the engine's stream between steps, with no request
    /// running ([`EngineHandle::with_model`]).
    pub fn with_model<T: Send + 'static>(
        &self,
        f: impl FnOnce(Option<&larch_model::Transformer>, &larch_kernels::Stream) -> T + Send + 'static,
    ) -> Result<T, EngineError> {
        self.runtime.engine.with_model(f)
    }

    /// The adaptation version in force; 0 is the base model.
    pub fn adapter_version(&self) -> u64 {
        self.runtime.engine.adapter_version()
    }

    /// Mean next-token cross-entropy (nats) over the positions `mask` marks, under the version in
    /// force ([`EngineHandle::token_loss`]).
    pub fn token_loss(&self, tokens: Vec<u32>, mask: Vec<bool>) -> Result<f64, EngineError> {
        self.runtime.engine.token_loss(tokens, mask)
    }

    /// The positions [`Self::token_loss`] scores, each with its top 64 logprobs
    /// ([`EngineHandle::token_positions`]).
    pub fn token_positions(
        &self,
        tokens: Vec<u32>,
        mask: Vec<bool>,
    ) -> Result<Vec<larch_engine::Position>, EngineError> {
        self.runtime.engine.token_positions(tokens, mask)
    }

    /// The tokens [`Self::generate`] would prefill for `prompt` with no tools and `thinking`
    /// ([`GenParams::thinking`]).
    pub fn encode_chat(
        &self,
        prompt: Prompt,
        thinking: Option<bool>,
    ) -> Result<Vec<u32>, EngineError> {
        self.runtime.encode_prompt(prompt, &[], thinking)
    }

    /// The routed expert layers the load kept in host memory: 0 when the whole model is on the
    /// device ([`larch_engine::EngineHandle::host_expert_layers`]).
    pub fn host_expert_layers(&self) -> u32 {
        self.runtime.engine.host_expert_layers()
    }

    /// (layers, experts) of the MoE GGUF `load` read; None for a dense model or `from_engine`.
    pub fn moe_shape(&self) -> Option<(u16, u32)> {
        self.runtime.moe
    }

    /// The GGUF `load` read (the trainer reads weights from it); empty for `from_engine`.
    pub fn model_path(&self) -> &Path {
        &self.runtime.model_path
    }

    /// The ids of `text` tokenized on its own, as llama.cpp does with no BOS or EOS added; a special
    /// token's text in it (an EOS or a chat marker) still becomes that token's id. A label for
    /// `GenParams::score_ids` must come out as one id: " A" for byte-level BPE, "A" where
    /// SentencePiece adds its space prefix.
    pub fn token_ids(&self, text: &str) -> Vec<u32> {
        self.runtime.tokenizer.encode(text, false)
    }

    /// The vocabulary's EOS token (an instruct model's end of turn), if it has one.
    pub fn eos(&self) -> Option<u32> {
        self.runtime.tokenizer.eos()
    }

    /// Returns the engine's real telemetry ring for observers.
    pub fn telemetry(&self) -> RingHandle {
        self.runtime.engine.ring()
    }

    /// Returns the engine's controls for humans and the external tuning consumer.
    pub fn knobs(&self) -> Arc<dyn KnobControl> {
        self.runtime.knobs.clone()
    }

    /// Returns the engine's current health.
    pub fn health(&self) -> Health {
        self.runtime.engine.health()
    }

    /// The `knob_overrides` the engine rejected at load, so a strict caller can fail by name.
    /// A clamp, such as `max_context` to the model's training context, is not a rejection.
    pub fn rejected_overrides(&self) -> &[(u16, f64, RejectReason)] {
        self.runtime.engine.rejected_overrides()
    }

    /// The loaded model and device (bounds and identity for strided and gait), updated by reload.
    pub fn deployment(&self) -> DeploymentSnapshot {
        self.runtime.engine.deployment()
    }

    /// The prefix cache's capacity and use; `None` when the engine has none.
    pub fn prefix_cache_stats(&self) -> Option<larch_engine::PrefixCacheStats> {
        self.runtime.engine.prefix_cache_stats()
    }

    /// Speculative decoding's counters since the engine started.
    pub fn guess_stats(&self) -> larch_engine::GuessStats {
        self.runtime.engine.guess_stats()
    }

    /// What kind of model is loaded, for a caller that sizes its concurrency by it.
    pub fn model_class(&self) -> ModelClass {
        self.runtime.engine.model_class()
    }

    /// This engine's identity, as every ring record's header carries it; gait's journal recovery
    /// compares it (12.6). Telemetry Off leaves the tier knob at 1 but the ring empty, so PREFLIGHT
    /// must abstain on an empty ring rather than trust the knob.
    pub fn engine_id(&self) -> u64 {
        self.runtime.engine.engine_id()
    }

    /// Installs the hook that runs after a fault, with the final records in the ring and a
    /// deadline for gait's journal fsync (9.11); it replaces any earlier hook.
    pub fn set_fault_flush(
        &self,
        hook: impl FnOnce(RingHandle, Instant) + Send + 'static,
    ) -> Result<(), EngineError> {
        self.runtime.engine.set_fault_flush(hook)
    }

    /// Attaches the running tuner's stop, then stops any tuner already attached. The last owner's
    /// drop runs it before the engine shuts down, so `stop` must not own a `Larch`, `Session` or
    /// `TextStream` (it would keep the engine alive) nor call back into this API. `stop` must:
    /// - return within a bounded grace: until it does, the drop can't shut the engine down and
    ///   every later `attach_tuner`/`stop_tuner` waits;
    /// - catch its own panics (9.11: callers see nothing); one that escapes reaches this caller
    ///   or whoever drops the last owner.
    ///
    /// This orders the calls, not the tuners: call `stop_tuner` before starting a new tuner, or
    /// both run against the engine until this returns.
    pub fn attach_tuner(&self, stop: impl FnOnce() + Send + 'static) {
        // The whole transition holds the slot, so no attach or stop completes while an earlier
        // tuner is still undoing its trial. The new stop goes in first, so a panicking old stop
        // can't leave the slot empty.
        let mut slot = self.stop_lock();
        if let Some(previous) = slot.replace(Box::new(stop)) {
            previous();
        }
    }

    /// Stops and detaches the attached tuner, returning once it has stopped; false when none
    /// was attached.
    pub fn stop_tuner(&self) -> bool {
        let mut slot = self.stop_lock();
        slot.take().map(|stop| stop()).is_some()
    }

    fn stop_lock(&self) -> std::sync::MutexGuard<'_, Option<Box<dyn FnOnce() + Send>>> {
        self.runtime
            .tuner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}
