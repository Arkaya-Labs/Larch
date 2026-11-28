//! The engine (architecture 9): one engine thread owns the device, the model, the KV pool and the
//! scheduler. Callers read per-request event channels. The engine and host-only watchdog serialize
//! ring publication and terminal notifications without holding those locks across device work.

mod engine;
#[cfg(any(test, debug_assertions, feature = "fault-inject"))]
mod fault;
mod fit;
pub use fit::DOC_POOL_FRACTION;
pub mod grammar;
mod guard;
mod guess;
mod knobs;
mod lifecycle;
mod pinfile;
mod prefix;
mod sample;
mod sched;
mod suffix;

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use larch_kernels::{DType, Device, DeviceProps, Stream};
pub use larch_model::RouterDeltas;
use larch_model::gguf::{self, Gguf};
use larch_model::projector::{ImagePixels, ProjectorConfig};
use larch_model::vision::{IMAGE_TOKEN, Image, image_tokens};
use larch_model::{ExpertPlan, Histogram, Model, ModelConfig, MoeCfg, Placement, Transformer};
use larch_records::{
    AbCommand, AbReply, Authority, Backend, DeploymentSnapshot, EffectiveConfig, EngineError,
    Finish as SpanFinish, Health, HealthState, KnobControl, KnobReply, KnobRequest, LoadError,
    MState, Outcome, RejectReason, RequestSpan, RevertTarget, RingHandle, knob_id,
};

pub use larch_records::Finish;

/// `EngineError::InvalidRequest` codes.
pub use pinfile::PinError;

pub mod invalid {
    pub const EMPTY_PROMPT: u16 = 1;
    pub const NO_TOKENS: u16 = 2;
    pub const TOKEN_OUT_OF_VOCAB: u16 = 3;
    pub const LOGPROBS: u16 = 4;
    /// A `Session` turn is already active (10.1.1); set by the facade, kept here so the codes stay one
    /// table.
    pub const SESSION_BUSY: u16 = 5;
    pub const SAMPLING: u16 = 6;
    /// The model's chat template refused the conversation.
    pub const CHAT_TEMPLATE: u16 = 7;
    /// Restored history is not system?, then whole user/assistant exchanges.
    pub const HISTORY: u16 = 8;
    /// A `RequestHandle` from another engine.
    pub const OTHER_ENGINE: u16 = 9;
    /// `finish_stop` after the request's terminal event was already taken.
    pub const TERMINAL_TAKEN: u16 = 10;
    /// A reload option is outside its registry bounds.
    pub const RELOAD_VALUE: u16 = 11;
    /// `ignore_eos` with as many stop tokens as sampling candidates: none might stay drawable.
    pub const IGNORED_STOPS: u16 = 12;
    /// `score_ids` with a repetition penalty: the scores would mix penalised and raw logits.
    pub const SCORE_PENALTY: u16 = 13;
    /// More `score_ids` than [`MAX_SCORE_IDS`](crate::MAX_SCORE_IDS).
    pub const SCORE_IDS: u16 = 14;
    /// `activations` with `max_tokens` other than 0 or with `score_ids` (it samples nothing), or
    /// with a `session` (it neither reads nor parks a session's sequence).
    pub const ACTIVATIONS: u16 = 15;
    /// An `activations` layer past the model's depth, or more layers than the model has.
    pub const ACTIVATION_LAYER: u16 = 16;
    /// The model refused an activations request's taps (`Model::forward_taps`' default); the
    /// rest of its step ran without them.
    pub const NO_TAPS: u16 = 17;
    /// A `grammar` with `score_ids`: both use the step's gathered ids.
    pub const GRAMMAR_SCORE_IDS: u16 = 18;
    /// A `grammar` with `ignore_eos`: its stops are allowed where the pattern matches.
    pub const GRAMMAR_IGNORE_EOS: u16 = 19;
    /// An empty grammar pattern or trigger, or a pattern that is not a regex.
    pub const GRAMMAR_PATTERN: u16 = 20;
    /// A grammar pattern whose DFA passes the size limit.
    pub const GRAMMAR_SIZE: u16 = 21;
    /// A `grammar` with no `stop_tokens`: nothing could end it where its pattern matches.
    pub const GRAMMAR_STOPS: u16 = 22;
    /// An image span that does not fit its prompt: outside it, overlapping another, of another
    /// length than its image's rows, or over ids other than its image's; or an image id outside
    /// every span.
    pub const IMAGE: u16 = 25;
    /// Images for an engine loaded without a vision projector.
    pub const NO_VISION: u16 = 26;
    /// A rejected GGUF header: combined with `LoadError::code()` to identify the cause.
    pub const RELOAD_MODEL: u16 = 0x100;
    /// Router deltas the model refuses (shape, values, or a model without routers), or deltas
    /// with version 0 / version 0 without them.
    pub const ROUTER_DELTAS: u16 = 23;
    /// `token_loss`: a mask of another length than the tokens, one that marks position 0 (nothing
    /// predicts it), or one that marks none.
    pub const LOSS_MASK: u16 = 24;
    /// A document the documents plan cannot hold on this device (`Larch::document_fit`).
    pub const DOC_TOO_LONG: u16 = 27;
    /// A pin on an engine whose prefix cache is off (`prefix_cache_blocks` 0): nothing could keep it.
    pub const NO_PREFIX_CACHE: u16 = 28;
    /// A pin's chain or an import's blocks are not what a persistence call needs: the engine
    /// reloaded or evicted them since, the call's sizes are not the loaded model's, or the model
    /// did not load from a file (nothing then identifies its KV).
    pub const PIN_CHANGED: u16 = 29;
    /// The pins, with [`crate::TURN_ROOM`], would not fit the KV pool: unpin one first.
    pub const PINS_FULL: u16 = 30;
    /// A pin file's import that a reload broke off (another chat's documents reload, a knob's):
    /// its blocks went with the pool, but the file is good, so import it again; not a bad file,
    /// as [`PIN_CHANGED`] is. Told once, to the import's next step.
    pub const IMPORT_TORN: u16 = 31;
}

/// Tokens of room the documents plan leaves beside a pinned document for the whole next turn: the
/// agent's own system text (tools, memory, date: 4 to 6k), the chat history, the question and the
/// answer.
pub const TURN_ROOM: u32 = 16_384;

/// The documents plan's context steps: a pinned document's `max_context` is rounded up to one.
pub const DOC_CONTEXT_STEP: u32 = 16_384;

/// What the documents plan can hold on this device, from the fit a load runs ([`fit::document_fit`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DocFit {
    /// The document fits the plan, with [`TURN_ROOM`] left for a turn, and the plan reads it no
    /// slower than the load outside it would (no more host layers where those prefill on the CPU).
    pub fits: bool,
    /// The longest document the plan holds so, in tokens; 0 when not even its smallest context fits.
    pub max_tokens: u32,
    /// Expert layers the plan keeps in host memory for the document (or, past the fit, for the
    /// longest that fits).
    pub host_layers: u32,
    /// The sequences the plan runs: the most up to the load's own `max_batch_seqs` that hold the
    /// document's context with no more host layers than two would; meaningful when `fits`.
    pub seats: u32,
}

/// Checks `req`'s images against its prompt and the projector: each span inside the prompt, in order,
/// apart, as long as its image's rows and holding its ids, with its pixels cut for the projector
/// (the engine never decodes: a bad image fails its request here, not the engine in a step),
/// and no image id outside a span. The number
/// of image rows, 0 for a prompt without images (its ids are then checked against the vocabulary).
fn check_images(
    req: &Request,
    vision: Option<&ProjectorConfig>,
    forced: bool,
) -> Result<usize, EngineError> {
    if req.images.is_empty() {
        return Ok(0);
    }
    let bad = EngineError::InvalidRequest(invalid::IMAGE);
    let Some(v) = vision else {
        return Err(EngineError::InvalidRequest(invalid::NO_VISION));
    };
    if forced || !req.score_ids.is_empty() || !req.activations.is_empty() {
        return Err(bad);
    }
    let (mut end, mut rows) = (0usize, 0usize);
    for span in &req.images {
        let (w, h) = (span.image.width(), span.image.height());
        let n = v.tokens_for(w, h) as usize;
        let at = span.at as usize;
        let ids = req
            .prompt
            .get(at..at.saturating_add(n))
            .filter(|_| at >= end)
            .ok_or(bad)?;
        if span.rows as usize != n
            || span.tile_rows != v.unit_rows(w, h)
            || !ids
                .iter()
                .copied()
                .eq(image_tokens(span.image.digest(), n as u32))
            || span.pixels.as_ref().is_none_or(|p| !p.fits(v, n as u32))
        {
            return Err(bad);
        }
        // Image ids between the spans are outside every span.
        if req.prompt[end..at].iter().any(|&t| t & IMAGE_TOKEN != 0) {
            return Err(bad);
        }
        (end, rows) = (at + n, rows + n);
    }
    if req.prompt[end..].iter().any(|&t| t & IMAGE_TOKEN != 0) {
        return Err(bad);
    }
    Ok(rows)
}

/// The most `score_ids` one request names: each sampled sequence's scores share the step's one
/// download (9.5), which is sized for this many.
pub const MAX_SCORE_IDS: usize = 256;

/// The least `max_context` for an engine with a vision projector: an image is up to 1,792 rows, so
/// 8,192 holds about three of them with their chat, 16,384 about seven.
pub const VISION_MAX_CONTEXT: u32 = 16_384;

/// The same with a Qwen3-VL projector (Qwen 3.5), whose model keeps 12 KiB of KV a token where
/// InternVL3.5's keeps 112: about 40 screenshots of 1,280 x 720 (920 rows each) with their chat. The
/// fit halves it down to [`VISION_MIN_CONTEXT`] where the KV pool cannot hold it.
pub const QWEN_VISION_MAX_CONTEXT: u32 = 65_536;

/// The least context an engine with projector `v` asks for.
pub(crate) fn vision_context(v: &ProjectorConfig) -> u32 {
    match v {
        ProjectorConfig::InternVl(_) => VISION_MAX_CONTEXT,
        ProjectorConfig::Qwen(_) => QWEN_VISION_MAX_CONTEXT,
    }
}

/// The least context a load with a projector drops to when the KV pool cannot hold more (one
/// 1,792-row image and its question).
pub const VISION_MIN_CONTEXT: u32 = 2048;

/// Resolved values: the facade fills what its caller left unset from the RequestScoped `*_default`
/// knobs in `effective()` before it submits (9.6), so a change applies to requests submitted after it.
/// One exception: a score request's unset `repeat_penalty` is 1, not `repeat_penalty_default`,
/// since its scores are of the raw logits (`invalid::SCORE_PENALTY`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    /// 0 = greedy.
    pub temperature: f32,
    pub top_p: f32,
    /// 0 = all 64 candidates.
    pub top_k: u32,
    pub repeat_penalty: f32,
    /// None: seeded from the request id.
    pub seed: Option<u64>,
}

impl SamplingParams {
    pub const GREEDY: SamplingParams = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repeat_penalty: 1.0,
        seed: None,
    };

    fn valid(&self) -> bool {
        self.temperature.is_finite()
            && self.temperature >= 0.0
            && self.top_p.is_finite()
            && self.top_p >= 0.0 // 0 keeps the top candidate, as the registry allows
            && self.top_p <= 1.0
            && self.top_k <= 64
            && self.repeat_penalty.is_finite()
            && self.repeat_penalty > 0.0
    }
}

/// Who waits on a request. The scheduler starts foreground prompts before background ones and,
/// while a foreground row decodes, decodes at most `BACKGROUND_ROWS` background rows (sched.rs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Class {
    /// A person waits on it: a chat turn.
    #[default]
    Foreground,
    /// Nobody waits on it token by token: workers, indexing, summaries, prefetch.
    Background,
}

/// An image's rows in a prompt: `at..at + rows`, holding `vision::image_tokens(image.digest(), rows)`.
#[derive(Clone, Debug)]
pub struct ImageSpan {
    pub at: u32,
    /// The projector's rows for the image ([`ProjectorConfig::tokens_for`]).
    pub rows: u32,
    /// One encode unit's rows ([`ProjectorConfig::unit_rows`]): an InternVL tile's, so a prefill
    /// chunk ends where a tile does, a step encodes at most one tile of a request's images and the
    /// other chats wait for one tile (about 175 ms on a 3050), not a whole image; a Qwen image's
    /// all, encoded whole in the step that reaches its first row.
    // ponytail: a whole Qwen image encodes in one step (an estimated 0.3-0.7 s at 1600 rows);
    // slice the encode by layers across steps if other chats' pauses show.
    pub tile_rows: u32,
    pub image: Arc<Image>,
    /// The image cut for the encoder ([`ProjectorConfig::cut`]). Admission refuses a span without
    /// it (the engine never decodes); it is an `Option` only so a caller can fill it once the prompt
    /// fits.
    pub pixels: Option<Arc<ImagePixels>>,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: Vec<u32>,
    /// The prompt's images, in prompt order; their rows are fed embeddings, not tokens. Empty is
    /// today's path, bit for bit.
    pub images: Vec<ImageSpan>,
    pub sampling: SamplingParams,
    pub max_tokens: u32,
    /// Generation stops after any of these (the caller passes the model's EOS/EOT here); `Finish::Eos`.
    pub stop_tokens: Vec<u32>,
    /// Top-n logprobs per token, at most 20.
    pub logprobs: u8,
    /// Never sample `stop_tokens` (a -inf logit) nor stop on them (`larch bench` only), as
    /// llama-server's `ignore_eos`, so every request generates exactly `max_tokens`.
    pub ignore_eos: bool,
    /// 0 = none.
    pub session: u64,
    /// Park the sequence at Done under `session` (D14). Parking is reclaimable; automatic sharing
    /// additionally bounds parked host histories to one pool's worth of block-table references.
    pub keep_kv: bool,
    /// Eligible for automatic prefix reuse/retention when the engine explicitly enables it.
    /// Session keep_kv=false still forbids global retention. False opts this request out.
    pub prefix_cache: bool,
    /// Trust boundary for automatic sharing. Distinct values never share KV.
    pub cache_namespace: u64,
    /// With `keep_kv` on a model with recurrent layers (Qwen 3.5): the prompt's chat history, which
    /// the session's next prompt repeats (the prompt without its generation prompt). The sequence
    /// parks with its state at this position, so the next turn runs on from there. 0, or a value
    /// past the prompt, parks the state at the turn's end. Other models ignore it.
    pub history_end: u32,
    /// When set, each token's `top` holds these ids with their exact full-vocabulary logprobs, in
    /// this order, in place of the most likely ones: System One's label scores. At most
    /// [`MAX_SCORE_IDS`], with no repetition penalty; `logprobs` is then unused (still at most 20).
    pub score_ids: Vec<u32>,
    /// When set, the request prefills only (`max_tokens` 0, nothing sampled, no Token events,
    /// `session` 0, so nothing parked; the automatic prefix cache may serve its prefix) and its
    /// Done carries the residual stream of the last prompt token after each of these layers
    /// (0-based), in this order; its `ttft_ms` times the rows. At most four run at once (the tap
    /// scratch reserved at load); the next waits in the queue. A model without taps ends it with
    /// `invalid::NO_TAPS`.
    pub activations: Vec<u16>,
    /// With exactly one `activations` layer: the prompt positions (strictly increasing, below the
    /// prompt's length) whose rows Done carries, one per position in this order, instead of the last
    /// token's. Empty is the last token. Sets `prefix_cache` aside: every position is computed.
    pub activation_rows: Vec<u32>,
    /// Foreground unless set: see [`Class`].
    pub class: Class,
    /// Send [`EngineEvent::Prefilled`] as the prompt is read; a request that did not ask gets
    /// none, and no per-chunk event costs it a send.
    pub progress: bool,
    /// Sample only what this allows ([`grammar::Grammar`]); None is today's path, bit for bit.
    /// Refused with `score_ids` or `ignore_eos`, with no `stop_tokens`, or with a vocabulary or
    /// a stop past the model's.
    pub grammar: Option<grammar::Constraint>,
}

impl Request {
    /// Greedy, no stops, no session.
    pub fn new(prompt: Vec<u32>, max_tokens: u32) -> Request {
        Request {
            prompt,
            images: Vec::new(),
            sampling: SamplingParams::GREEDY,
            max_tokens,
            stop_tokens: Vec::new(),
            logprobs: 0,
            ignore_eos: false,
            session: 0,
            keep_kv: false,
            prefix_cache: true,
            cache_namespace: 0,
            history_end: 0,
            score_ids: Vec::new(),
            activations: Vec::new(),
            activation_rows: Vec::new(),
            class: Class::Foreground,
            progress: false,
            grammar: None,
        }
    }
}

/// Teacher forcing for the oracle (architecture 15): the reference tokens run through the production
/// scheduler, executor and paged KV, and every computed position reports its full-vocabulary top-64.
/// The modes are this one call: `prompt = tokens.len()` prefills (one batch when
/// `prefill_chunk_tokens` and the logits rows that fit in device memory hold it, else chunked),
/// `prompt = 1` decodes token by token, several in flight run in parallel, and a `session` turn
/// reuses its parked prefix (the cached mode). For decode rows in the chunk steps (the chunked mode,
/// D16), start a decode-mode run first and the chunked one while it still decodes: admission follows
/// submission, and under `decode_first` a chunked run admitted first takes the whole chunk budget
/// each step, so a decode run admitted after it never shares a step with a chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Forced {
    /// The reference tokens.
    pub tokens: Vec<u32>,
    /// `tokens[..prompt]` are the prompt; each later token is fed by its own decode step in place of
    /// the sampled one. 1 ≤ prompt ≤ tokens.len().
    pub prompt: u32,
    /// As in [`Request`].
    pub session: u64,
    /// As in [`Request`]: parks `tokens` under `session` at Done.
    pub keep_kv: bool,
}

/// The next-token distribution after `tokens[..=pos]`: the 64 most likely tokens (fewer if the
/// vocabulary is smaller), most likely first, as (id, logprob over the full vocabulary).
#[derive(Clone, Debug, PartialEq)]
pub struct Position {
    /// The last fed position.
    pub pos: u32,
    /// (id, logprob), most likely first.
    pub top: Vec<(u32, f32)>,
    /// The exact logprob of the token fed at `pos + 1`, for a [`EngineHandle::token_loss`] run;
    /// None otherwise.
    pub next: Option<f32>,
}

/// A forced run in flight.
pub struct ForcedHandle {
    /// Ends with exactly one event, Done (`finish` Length; `usage.cached_tokens` = the positions
    /// reused from the session, which are not computed) or Error; no Token events. Dropping it
    /// cancels the run.
    pub request: RequestHandle,
    /// Each computed position once, ascending, from `cached_tokens` to `tokens.len() − 1`; all of
    /// them have arrived when `request` delivers Done.
    pub positions: Receiver<Position>,
}

/// What this engine does on its device, for `larch info`, gait's preflight and the parity command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// The device's backend.
    pub backend: Backend,
    /// Captured decode graphs are replayed (9.5 step 2). While false, `decode_graphs` is 0, setting
    /// it to 1 is Rejected `Unsupported`, and every `EngineSample.graph_replayed` is 0.
    pub graph_replay: bool,
    /// The largest live `prefill_chunk_tokens`: its startup value, and at least 1024. The longest
    /// one-batch prefill; `DeploymentSnapshot.prefill_chunk_max` carries it.
    pub prefill_chunk_max: u32,
    /// How `EngineSample.achieved_gbps` arrives here: Estimated (from the GGUF's weight bytes and
    /// the KV the step read, over `gpu_ns`), Missing (no weight bytes: a test model) or Unsupported.
    /// When Estimated, a step of more than 16 rows is still Missing, and so is its `hbm_util`: the
    /// model dequantizes each weight and runs a gemm there, which the weight bytes undercount.
    pub achieved_gbps: MState,
    /// `EngineSample.hbm_util` (achieved / the deployment's peak): Unsupported on the CPU reference.
    pub hbm_util: MState,
    /// `EngineSample.sm_occupancy`: Missing on CUDA/HIP until the occupancy query is wired.
    pub sm_occupancy: MState,
    /// `DeviceSample`: Measured where the engine opened NVML on its device (CUDA): one a second
    /// (one in 64 at T0) while telemetry is on, besides the one each load and reload writes.
    /// Otherwise only the load records exist: Missing on CUDA, whose VRAM fields come from the
    /// device props; Unsupported elsewhere, every field Unsupported.
    pub device_sample: MState,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampledToken {
    pub id: u32,
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Prompt tokens reused from session parking or the explicitly enabled automatic cache.
    pub cached_tokens: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    pub ttft_ms: f64,
    pub tpot_ms: f64,
    pub tok_s: f64,
    /// Grammar picks made with no allowed token among the 64 candidates (`Request::grammar`).
    pub grammar_forced: u32,
}

impl Timings {
    fn of(s: &RequestSpan) -> Timings {
        let ms = |ns: u64| ns as f64 / 1e6;
        let tokens = f64::from(s.gen_tokens);
        let decode = s.t_done_ns.saturating_sub(s.t_first_ns);
        Timings {
            ttft_ms: ms(s.t_first_ns.saturating_sub(s.t_submit_ns)),
            tpot_ms: if s.gen_tokens > 1 {
                ms(decode) / (tokens - 1.0)
            } else {
                0.0
            },
            tok_s: if decode > 0 {
                (tokens - 1.0) / (decode as f64 / 1e9)
            } else {
                0.0
            },
            grammar_forced: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum EngineEvent {
    Token {
        id: u32,
        logprob: f32,
        top: Vec<(u32, f32)>,
    },
    Done {
        finish: Finish,
        usage: Usage,
        timings: Timings,
        /// One f32 row of `d_model` per `Request.activations` layer, in that order; empty otherwise.
        activations: Vec<Vec<f32>>,
        /// The adaptation version (router deltas) that produced it; 0 is the base model.
        adapter_version: u64,
    },
    Error(EngineError),
    /// A prompt chunk was read: `done` of the prompt's `total` tokens have KV, those reused
    /// included. One per chunk, before the first Token; only for a request with
    /// [`Request::progress`].
    Prefilled {
        done: u32,
        total: u32,
    },
}

/// What a `.pin` file must match to load on this engine: the model file, the KV layout and the
/// build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PinInfo {
    /// The GGUF's identity at this load.
    pub file: u64,
    /// KV layers.
    pub layers: u32,
    /// Bytes of one block of K (or V) in one layer.
    pub block_bytes: u64,
    /// The pool's element type, as `DType`'s ordinal.
    pub dtype: u32,
    /// One sequence's recurrent state; 0 for a model without any.
    pub state_bytes: u64,
    /// Tokens per block.
    pub block_tokens: u32,
    /// Tokens per file record: the pin read's state-checkpoint mark.
    pub segment: u32,
    /// The kernel ABI version.
    pub abi: u32,
    /// The engine crate's version.
    pub version: &'static str,
    /// The source commit it was built from ([`larch_model::card::BUILD`]): a kernel or RoPE fix
    /// changes KV without a version bump.
    pub build: &'static str,
}

/// One step of a pin's persistence, run on the engine thread ([`pinfile`]).
pub(crate) enum PinOp {
    Info,
    /// The tokens pin `id` can export now (its chain's whole blocks, on a hybrid to its last state
    /// checkpoint, and while its read is under way to the last mark), the ids from `from` on, and
    /// the state after them if asked: together, since the next mark replaces the checkpoint.
    Chain {
        id: u64,
        from: usize,
        state: bool,
    },
    /// Pin `id`'s tokens `from..to` (whole blocks) of KV in one layer, each block's K then V.
    Layer {
        id: u64,
        from: usize,
        to: usize,
        layer: u32,
    },
    /// Takes blocks for an import of `tokens` in `namespace` under `id` (a new one when None).
    Begin {
        id: Option<u64>,
        namespace: u64,
        tokens: Vec<u32>,
    },
    /// Fills `layer`'s blocks from block `from_block` with `bytes`.
    Upload {
        id: u64,
        layer: u32,
        from_block: usize,
        bytes: Vec<u8>,
    },
    /// Publishes and pins the first `tokens` tokens of the import, with the state after them.
    Finish(u64, Vec<u8>, usize),
    Abort(u64),
}

impl PinOp {
    /// The step's name, for the log.
    fn name(&self) -> &'static str {
        match self {
            PinOp::Info => "info",
            PinOp::Chain { .. } => "chain",
            PinOp::Layer { .. } => "layer",
            PinOp::Begin { .. } => "begin",
            PinOp::Upload { .. } => "upload",
            PinOp::Finish(..) => "finish",
            PinOp::Abort(_) => "abort",
        }
    }
}

pub(crate) enum PinReply {
    Info(PinInfo),
    Chain {
        count: usize,
        ids: Vec<u32>,
        state: Vec<u8>,
    },
    Bytes(Vec<u8>),
    Id(u64),
    Done,
}

/// A pinned chain's hold on the prefix cache ([`EngineHandle::pin`]); dropping it unpins.
pub struct PinHandle {
    id: u64,
    tx: Sender<Msg>,
}

impl Drop for PinHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Unpin(self.id));
    }
}

/// A submitted request's stream. Dropping it cancels the request.
pub struct RequestHandle {
    pub id: u64,
    pub events: Receiver<EngineEvent>,
    cancel: Arc<AtomicBool>,
    /// The submitting engine, for `finish_stop`; ids are per engine.
    engine: Weak<Shared>,
}

/// Cloneable cancellation access without sharing a request's event receiver.
#[derive(Clone)]
pub struct RequestCanceller(Arc<AtomicBool>);

impl RequestCanceller {
    /// Cancels the request; observable immediately, including after Done was queued.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl RequestHandle {
    /// Cancels this request.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation was requested, here or through a canceller.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Shares only cancellation access, leaving this handle as the stream's sole owner.
    pub fn canceller(&self) -> RequestCanceller {
        RequestCanceller(self.cancel.clone())
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TelemetryTier {
    Off,
    T0,
    T1,
}

/// What a fault does to the process (9.11, D2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnFault {
    /// Exit with code 3 for `larch supervise` to restart (the CLI).
    Exit,
    /// Stay up with health Failed; every call then fails (library callers).
    Fail,
}

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub model: PathBuf,
    /// The vision projector loaded beside the model (an mmproj GGUF), so requests may carry images.
    /// With one, the load raises `max_context` to at least [`VISION_MAX_CONTEXT`] and halves it to
    /// what fits; a reload to another model drops it and asks for `max_context` as given.
    pub mmproj: Option<PathBuf>,
    /// [`larch_kernels::CPU`] or a GPU ordinal.
    pub device: i32,
    pub max_context: u32,
    pub kv_pool_fraction: f32,
    /// The KV pool's pages: `DType::F16` (the default), or `DType::Q8_0`, 8 bits a value and an f16
    /// scale a 32 (larch.h `larch_kv_pool`), 17 / 32 of the bytes, so the same pool holds about 1.9x
    /// the tokens. A backend without Q8_0 pages (CUDA and HIP today), or a model whose head_dim is
    /// not a multiple of 32, loads F16 (`engine::kv_dtype`). LARCH_KV_DTYPE (`f16` or `q8_0`)
    /// initializes it.
    pub kv_dtype: DType,
    /// A power of two.
    pub ring_capacity: usize,
    /// Startup knob values (`--set k=v`).
    pub knob_overrides: Vec<(u16, f64)>,
    pub telemetry: TelemetryTier,
    pub on_fault: OnFault,
    /// Explicit consent to global KV retention, including eligible stateless requests. Zero (the
    /// default) disables it. Bounded to pool/4 and 8192 index entries; session parking is separate.
    /// LARCH_PREFIX_CACHE_BLOCKS initializes this option for CLI/controlled comparisons.
    pub prefix_cache_blocks: usize,
    /// Custom loaders must explicitly attest deterministic, prefix-independent KV semantics.
    /// Ordinary Transformer loads are compatible; this is false by default for start_with/reload_with.
    pub prefix_cache_custom_compatible: bool,
    /// The least time a step may take before the watchdog faults the engine (9.12). `None` picks
    /// it from the device the engine opens: 5 s on a GPU, 60 s on the CPU reference, where a long
    /// prefill chunk in a debug build or on a slow machine takes seconds and is not a lost device.
    /// A/B and revert commands wait up to the same limit (9.8); a knob `set` keeps its own
    /// `timeout_ms`. Zero removes the floor and leaves only 10 × p99 of measured steps.
    // ponytail: a real CPU hang now takes 60 s to detect; scale the floor by the step's prefill
    // tokens if CPU serving needs faster hang detection.
    pub step_timeout_floor: Option<Duration>,
    /// MoE models: the layers, from the first, whose routed experts stay in host memory and run on
    /// the CPU ([`Placement`]). `None` (the default) plans it at load from the device's free memory:
    /// none when the model fits. LARCH_CPU_EXPERT_LAYERS initializes it, for measuring.
    pub cpu_expert_layers: Option<u32>,
    /// Tests only: engine time in ns, read in place of the wall clock, so a fake model that
    /// advances it sets every step's duration exactly; `None` uses the wall clock. The watchdog
    /// reads it too, so it times steps in test time; `gpu_ns` still comes from the device timers,
    /// so a sample's `host_ns` mixes the two clocks.
    #[doc(hidden)]
    pub clock: Option<Arc<AtomicU64>>,
    /// Speculative decoding: the most tokens a decoding sequence guesses (by suffix drafts: its own
    /// tokens, and the outputs finished before it, up to 64k tokens of ids per cache namespace, kept in host
    /// memory) and checks in one step, at most [`MAX_GUESSES`]; 0 turns it off. Output
    /// keeps the model's: sampled in distribution; greedy bit for bit against a run without guesses
    /// at the same nonzero `attn_decode_splits`, where the check's rows run decode's kernels. With
    /// guesses on, an `attn_decode_splits` of 0 (backend picks) is pinned to 16 while checks can run,
    /// so a check row and the decode row it replaces split attention alike; against a plain run
    /// at the backend's pick that moves attention by f32 rounding, which can flip a near-tie token,
    /// so compare runs with the knob at 16 on both sides.
    /// LARCH_GUESSES initializes it (8 is the tested setting); default 0.
    // ponytail: an option, not a knob, since the knob registry is the frozen larch-records
    // contract; becomes a knob with the registry's next revision. Off by default until a Mac run
    // shows exact greedy output and no prose slowdown, and until the agent and serve test models
    // (scripted one token per logits row, not a function of their input) are pure or opt out.
    pub guesses: u32,
    /// A DFlash drafter GGUF (`scripts/dflash_convert.py`) for this model: loading one turns
    /// guessing on, its block length bounding the guesses (15 for a block of 16), whatever
    /// `guesses` says; suffix drafts then serve only sequences the drafter has no context for.
    /// The drafter's weights, its caches for four sequences at `max_context` and its
    /// taps count in the fit. `mtp`, or the model's own path, picks the model's built-in
    /// multi-token-prediction block instead (Qwen 3.5's nextn; a model without one refuses the
    /// load). LARCH_DRAFTER initializes it.
    pub drafter: Option<PathBuf>,
    /// A drafter the engine picks up on its own, as `drafter` names one, kept only where it costs
    /// no chat: the load plans with it and without it and keeps it at no more host expert layers
    /// and no fewer state slots (`larch_drafter_auto`), else loads as without one. A `drafter`
    /// naming another file wins. LARCH_DRAFTER_AUTO initializes it.
    pub drafter_auto: Option<PathBuf>,
    /// With a drafter: the most guesses a check tries, 0 for the drafter's block length less one
    /// (a block of 16 guesses 15; a recurrent model's spare state bounds either to 7), or 3 for the
    /// MTP block, llama.cpp's default.
    /// LARCH_DRAFTER_GUESSES initializes it; default 4, the MTP block's too (0 asks for its 3):
    /// on the M5 a full block kept ~2.4 guesses a round and ran at 0.5x, a block of 4 ran at ~1.1-1.45x.
    pub drafter_guesses: u32,
    /// Tests only: the host memory a reload's carry may copy, in place of what the host has free.
    #[doc(hidden)]
    pub carry_bytes: Option<usize>,
    /// Tests only: tokens between a pin read's state checkpoints, in place of 8,192 (a multiple of
    /// the block).
    #[doc(hidden)]
    pub pin_mark: Option<usize>,
    /// Load straight into a documents plan, as a documents reload ([`ReloadChange::documents`])
    /// would set it: this context and seats, [`DOC_POOL_FRACTION`] and no drafter. The plan the
    /// other options name is the one the engine goes back to once no pin is held, as after that
    /// reload. A restart that reopens a pinned chat loads once instead of twice.
    pub documents: Option<DocumentsPlan>,
}

/// The most guesses one check step tries: with the last token, 16 rows, the most a Metal step
/// runs through the one-row decode's mmvq kernels.
pub const MAX_GUESSES: u32 = 15;

/// Speculative decoding's counters since start: check rounds (one per checked sequence a step),
/// the guesses they tried and those kept. Each round also emits one token of the model's own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GuessStats {
    pub rounds: u64,
    pub tried: u64,
    pub kept: u64,
    /// The guesses picked from the measured step cost, by count, for each sequence of a step
    /// that could check: `k[0]` decoded plainly.
    pub k: [u64; MAX_GUESSES as usize + 1],
}

impl GuessStats {
    /// The counters since `before`, an earlier reading.
    pub fn since(self, before: GuessStats) -> GuessStats {
        GuessStats {
            rounds: self.rounds.saturating_sub(before.rounds),
            tried: self.tried.saturating_sub(before.tried),
            kept: self.kept.saturating_sub(before.kept),
            k: std::array::from_fn(|i| self.k[i].saturating_sub(before.k[i])),
        }
    }
}

impl std::fmt::Display for GuessStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let per = self.kept as f64 / self.rounds.max(1) as f64;
        write!(
            f,
            "guesses: {} rounds · {} tried · {} kept · {per:.2} kept/round · k",
            self.rounds, self.tried, self.kept
        )?;
        for (k, n) in self.k.iter().enumerate().filter(|p| *p.1 > 0) {
            write!(f, " {k}:{n}")?;
        }
        Ok(())
    }
}

impl EngineOptions {
    /// Whether a load with these options may check guesses.
    pub(crate) fn guessing(&self) -> bool {
        self.guesses > 0 || self.drafter.is_some()
    }

    /// Whether the drafter is the model's own MTP block (see `drafter`).
    pub(crate) fn mtp(&self) -> bool {
        let canon = |p: &Path| std::fs::canonicalize(p);
        let same = |p: &Path| matches!((canon(p), canon(&self.model)), (Ok(a), Ok(b)) if a == b);
        (self.drafter.as_deref())
            .is_some_and(|p| p == Path::new("mtp") || p == self.model || same(p))
    }

    /// The automatic drafter a load plans (`drafter_auto`). A `drafter` naming the same file is
    /// its last load's choice, so a reload plans it again.
    pub(crate) fn auto_drafter(&self) -> Option<&PathBuf> {
        (self.drafter_auto.as_ref()).filter(|&a| self.drafter.as_ref().is_none_or(|d| d == a))
    }

    /// These options reloading onto `model`: a drafter named by the old model's own path still
    /// means the model's own block, and that path now names another file.
    pub(crate) fn onto(mut self, model: PathBuf) -> EngineOptions {
        // A projector and an automatic drafter belong to the model they came with: another model
        // loads without them.
        if model != self.model {
            if self.auto_drafter().is_some() {
                self.drafter = None;
            }
            (self.mmproj, self.drafter_auto) = (None, None);
        }
        if self.mtp() {
            self.drafter = Some(PathBuf::from("mtp"));
        }
        self.model = model;
        self
    }

    pub fn new(model: PathBuf, device: i32) -> EngineOptions {
        EngineOptions {
            model,
            mmproj: None,
            device,
            max_context: 8192,
            kv_pool_fraction: 0.85,
            kv_dtype: match std::env::var("LARCH_KV_DTYPE").as_deref() {
                Ok("q8_0") => DType::Q8_0,
                _ => DType::F16,
            },
            ring_capacity: larch_records::DEFAULT_CAPACITY,
            knob_overrides: Vec::new(),
            telemetry: TelemetryTier::T1,
            on_fault: OnFault::Fail,
            prefix_cache_blocks: std::env::var("LARCH_PREFIX_CACHE_BLOCKS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            prefix_cache_custom_compatible: false,
            step_timeout_floor: None,
            carry_bytes: None,
            pin_mark: None,
            documents: None,
            cpu_expert_layers: std::env::var("LARCH_CPU_EXPERT_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok()),
            clock: None,
            guesses: std::env::var("LARCH_GUESSES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            drafter: std::env::var_os("LARCH_DRAFTER")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
            drafter_auto: std::env::var_os("LARCH_DRAFTER_AUTO")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
            drafter_guesses: std::env::var("LARCH_DRAFTER_GUESSES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
        }
    }
}

/// A documents plan to load into ([`EngineOptions::documents`]): what the facade's documents
/// reload passes, the context and the seats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DocumentsPlan {
    /// `max_context` of the plan.
    pub max_context: u32,
    /// `max_batch_seqs` of the plan.
    pub seats: u32,
}

/// Model and fit options changed by a drain, rebuild and resume (9.1).
#[derive(Clone, Debug, Default)]
pub struct ReloadChange {
    /// Replacement GGUF; absent means reload the current path. Another model reloads without the
    /// vision projector (`EngineOptions::mmproj` was the old model's).
    pub model: Option<PathBuf>,
    /// New context limit, clamped to the model's training context after loading.
    pub max_context: Option<u32>,
    /// Fraction of available memory reserved for the KV pool.
    pub kv_pool_fraction: Option<f32>,
    /// New `max_batch_seqs`, which a hybrid's plan also pays its state slots for.
    pub max_batch_seqs: Option<u32>,
    /// Load without a drafter, whether named or automatic (a document's KV is worth more).
    pub drafter_off: bool,
    /// A documents plan's reload (the facade's `Larch::pin`): the engine remembers the plan it
    /// replaces and goes back to it when the last pin drops, at its next idle moment.
    pub documents: bool,
}

/// What kind of model is loaded, so a caller sizes its concurrency by it, not by the model's name:
/// a dense step reads every weight once whatever the batch, while each row of a mixture of experts
/// reads its own experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelClass {
    /// No experts: a step's weight reads do not grow with its rows.
    pub dense: bool,
    /// The weight bytes one token reads: every tensor of the GGUF except the embedding rows a step
    /// gathers (`token_embd`, unless the output projection reuses it). For a custom loader's model,
    /// the device memory its load took; 0 is unknown (a CPU device, a test model).
    pub active_bytes: u64,
    /// The experts are read from system RAM, not device memory.
    pub experts_in_ram: bool,
}

/// What the GGUF header says about the weights; [`start`] reads it. A clone shares the load's
/// readings (an automatic drafter's two loaders).
#[derive(Clone)]
pub(crate) struct Weights {
    pub(crate) bytes: u64,
    /// What one step reads: every tensor except the token embeddings, whose rows are gathered,
    /// unless the output projection reuses them.
    pub(crate) read_per_step: u64,
    pub(crate) params: u64,
    /// The type holding the most bytes, by its GGML name.
    pub(crate) quant: String,
    /// Each layer's routed expert bytes; empty for a dense model.
    pub(crate) experts: Vec<u64>,
    /// The layers whose experts the load kept in host memory, as the loader planned it.
    pub(crate) host_layers: Arc<AtomicU32>,
    /// The max_context and kv_pool_fraction the loader plans the placement for: `Loaded::load`
    /// sets the effective ones (the options after the startup overrides) before it calls the loader.
    pub(crate) budget: Arc<Mutex<(u32, f32)>>,
    /// The most sequences a hybrid model's load gives recurrent state (`max_batch_seqs` up to the
    /// cap), which `Loaded::load` sets before it calls the loader; the plan gives what fits.
    pub(crate) seqs: Arc<AtomicU32>,
    /// The state slots a hybrid's plan pays host layers for, set by `Loaded::load` with `seqs`:
    /// 1 by default (the plan for one chat, the rest from what it leaves), `seqs` when
    /// `max_batch_seqs` was set by hand. Other models ignore it.
    pub(crate) asked: Arc<AtomicU32>,
    /// The seats fit() reserves decode graphs for (`max_batch_seqs`), 0 when decode_graphs is off
    /// at load: `Loaded::load` sets it before it calls the loader.
    pub(crate) graph_seats: Arc<AtomicU32>,
    /// What the loader planned, read back by `Loaded::load`.
    pub(crate) ledger: Arc<Mutex<Ledger>>,
    /// The routed experts' shape; `None` for a dense model.
    pub(crate) moe: Option<MoeCfg>,
}

/// The loader's plan terms: the reserve beside the weights (LARCH_FIT_DIAGNOSTICS' ledger) and
/// whether the plan paid for the decode graphs.
#[derive(Clone, Copy, Default)]
pub(crate) struct Ledger {
    pub(crate) free_before: u64,
    pub(crate) weights_on_device: u64,
    pub(crate) activation: u64,
    pub(crate) routers: u64,
    pub(crate) hybrid: u64,
    pub(crate) scratch: u64,
    pub(crate) graph_slots: u64,
    pub(crate) graph_arena: u64,
    pub(crate) graphs_paid: bool,
    /// The rows the load planned for (0: no plan ran), which `Loaded::load` then serves.
    pub(crate) rows: u32,
    /// The draft caches the load planned for, which `Loaded::load` then allocates.
    pub(crate) draft_seats: u32,
    /// An automatic drafter's chats with it and without it, and whether the load kept it.
    pub(crate) auto: Option<(fit::Chats, fit::Chats, bool)>,
}

/// What a load of `rows` rows and `draft_seats` draft caches plans with: the weight bytes it
/// places on the device, the check spares, the serving scratch, a hybrid's own scratch and each
/// state slot's, and the activations.
#[derive(Clone, Copy, PartialEq)]
struct RowTerms {
    rows: u32,
    draft_seats: u32,
    bytes: u64,
    host_embed: bool,
    checks: u32,
    scratch: u64,
    hybrid_base: u64,
    slot_bytes: u64,
    activation: u64,
}

impl Weights {
    fn of(g: &Gguf) -> Weights {
        let gathered = match (g.tensor("token_embd.weight"), g.tensor("output.weight")) {
            (Some(e), Some(_)) => e.bytes,
            _ => 0,
        };
        let mut by_type = std::collections::BTreeMap::new();
        for t in &g.tensors {
            let n: &mut u64 = by_type.entry(t.ggml_type).or_insert(0);
            *n = n.saturating_add(t.bytes);
        }
        let main = by_type.into_iter().max_by_key(|&(_, b)| b);
        let (layers, moe) = ModelConfig::from_gguf(g).ok().map_or((0, None), |c| {
            (c.n_layers, MoeCfg::from_gguf(g, c.arch).ok().flatten())
        });
        // The hybrid MoE target's built-in drafter (blk.40, nextn) is past the config's layers, and the load
        // leaves it out of the GPU map. Saturating: a crafted header's unused tensors may claim
        // any size (types it cannot load have no byte bound).
        let bytes = g
            .tensors
            .iter()
            .filter(|t| {
                let i = t.name.strip_prefix("blk.").and_then(|r| r.split_once('.'));
                i.and_then(|(i, _)| i.parse::<u32>().ok())
                    .is_none_or(|i| i < layers)
            })
            .fold(0u64, |a, t| a.saturating_add(t.bytes));
        // One entry per layer the config declares: a crafted name (blk.1000000000.ffn_up_exps)
        // outside them is ignored as other unused tensors are, never an allocation. A mixed layer's
        // cold experts count with its hot ones.
        let mut experts = vec![0u64; if moe.is_some() { layers as usize } else { 0 }];
        for t in &g.tensors {
            let layer = t.name.strip_prefix("blk.").and_then(|r| {
                let (i, rest) = r.split_once('.')?;
                matches!(
                    rest,
                    "ffn_gate_exps.weight"
                        | "ffn_up_exps.weight"
                        | "ffn_down_exps.weight"
                        | "ffn_gate_cold_exps.weight"
                        | "ffn_up_cold_exps.weight"
                        | "ffn_down_cold_exps.weight"
                )
                .then(|| i.parse::<usize>().ok())?
            });
            if let Some(e) = layer.and_then(|i| experts.get_mut(i)) {
                *e = e.saturating_add(t.bytes);
            }
        }
        // A token reads `used` of each layer's experts.
        let unread = moe.map_or(0, |m| {
            let all = experts.iter().fold(0u64, |a, &e| a.saturating_add(e));
            (all as f64 * (1.0 - f64::from(m.used) / f64::from(m.experts))) as u64
        });
        Weights {
            bytes,
            read_per_step: bytes.saturating_sub(gathered).saturating_sub(unread),
            params: g.tensors.iter().fold(0u64, |a, t| {
                a.saturating_add(t.dims.iter().fold(1u64, |n, &d| n.saturating_mul(d)))
            }),
            quant: main
                .and_then(|(t, _)| gguf::dtype(t).ok())
                .map_or(String::new(), |d| ggml_name(d).to_owned()),
            experts,
            host_layers: Arc::default(),
            budget: Arc::default(),
            seqs: Arc::default(),
            asked: Arc::default(),
            graph_seats: Arc::default(),
            ledger: Arc::default(),
            moe,
        }
    }
}

/// GGML's name for a type, spelled as in 5 and 6.1: strided reads it as its `dtype` (11.2), so it
/// must not follow a rename of a `DType` variant.
fn ggml_name(d: DType) -> &'static str {
    match d {
        DType::F32 => "F32",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::I32 => "I32",
        DType::Q4K => "Q4_K",
        DType::Q5K => "Q5_K",
        DType::Q6K => "Q6_K",
        DType::Q8_0 => "Q8_0",
        DType::Q4_0 => "Q4_0",
        DType::Q5_0 => "Q5_0",
        DType::Q3K => "Q3_K",
        DType::Q2K => "Q2_K",
    }
}

/// Rejected spans waiting for the engine at most, so a flood of refusals cannot load the engine
/// thread (9.2).
const REJECTS_QUEUED: usize = 64;

/// Builds the model on the engine thread: `(device, stream, max rows per step)`.
pub type Loader =
    Box<dyn FnOnce(&Device, &Stream, u32) -> Result<Box<dyn Model>, LoadError> + Send>;

/// Teacher forcing at admission: the tokens fed in place of sampled ones, where the positions go,
/// and whether each position carries the next token's exact logprob (`Position::next`).
type Forcing = (Vec<u32>, Sender<Position>, bool);

/// Messages to the engine thread.
enum Msg {
    /// Admitted together: queued in one engine turn (#64).
    Submit(Vec<sched::Req>),
    Knob(KnobRequest, u64, SyncSender<KnobReply>),
    Revert(RevertTarget, Authority, u64, SyncSender<KnobReply>),
    /// Taken by whichever of the engine and the caller swaps the flag first (see `EngineKnobs::ab`).
    Ab(AbCommand, u64, Arc<AtomicBool>, SyncSender<AbReply>),
    /// A request refused at admission, for the ring (9.2).
    Rejected(RequestSpan),
    Release(u64),
    /// Pins (pin id, cache namespace, prompt tokens) a chain in the prefix cache, until `Unpin`.
    Pin(u64, u64, Vec<u32>, SyncSender<Result<(), EngineError>>),
    Unpin(u64),
    /// A step of a pin's export or import, answered on the engine thread.
    PinOp(PinOp, SyncSender<Result<PinReply, EngineError>>),
    /// The documents plan for this many tokens, on the engine thread's device and options.
    DocFit(u32, SyncSender<Result<DocFit, EngineError>>),
    Parked(u64, SyncSender<u32>),
    PrefixStats(SyncSender<PrefixCacheStats>),
    FinishStop(u64, SyncSender<()>),
    /// Router deltas and their version, applied at the first boundary with no started request.
    RouterDeltas(
        Option<RouterDeltas>,
        u64,
        SyncSender<Result<(), EngineError>>,
    ),
    /// Work on the resident model, run at the first boundary with no started request.
    Job(engine::Job),
    /// An image's units, cut by the caller, to encode while the engine is idle (S5).
    Prepare([u8; 32], Arc<ImagePixels>),
    Reload(
        ReloadChange,
        Option<Loader>,
        SyncSender<Result<(), EngineError>>,
    ),
    Shutdown,
}

/// Model-specific information swapped before Ready, also used for admission.
#[derive(Clone, Copy)]
struct Published {
    deployment: DeploymentSnapshot,
    caps: Capabilities,
    class: ModelClass,
    /// The vision projector's geometry, when one is loaded.
    vision: Option<ProjectorConfig>,
    /// Sequences of recurrent state the model holds (`Model::state_slots`); 0 without any.
    state_slots: u32,
    /// Device memory free when this load began ([`EngineHandle::free_before`]).
    free_before: u64,
    /// Expert layers this load keeps in host memory ([`EngineHandle::host_expert_layers`]).
    host_layers: u32,
}

/// State the engine publishes to every handle.
struct Shared {
    /// Held across `submit`'s send, so the engine's drain after it leaves Ready sees every request.
    health: Mutex<Health>,
    // ponytail: an RwLock'd Arc stands in for arc-swap; readers hold it for a clone only.
    effective: RwLock<Arc<EffectiveConfig>>,
    info: RwLock<Published>,
    ring: RingHandle,
    host: Mutex<lifecycle::Host>,
    inbox: Mutex<Option<Receiver<Msg>>>,
    watch: lifecycle::Watch,
    epoch: Instant,
    /// `EngineOptions::clock`.
    clock: Option<Arc<AtomicU64>>,
    /// Shutdown closes admission immediately, even while a reload is building.
    shutdown: AtomicBool,
    /// Set with Failed under `health`, so the engine's early-outs never sleep on a caller's send.
    failed: AtomicBool,
    next_id: AtomicU64,
    /// Submitted but not yet seen by the engine thread, and waiting in its queue.
    in_flight: AtomicUsize,
    sched_waiting: AtomicUsize,
    /// Pins that went cold to give way to a request ([`EngineHandle::pin_live`]).
    cold_pins: Mutex<std::collections::HashSet<u64>>,
    /// Image tiles encoded into the tile cache since start ([`Engine::tiles_encoded`]).
    tiles_encoded: AtomicU64,
    /// Steps that ran a slice of an image's encode ([`Engine::encode_slices`]).
    encode_slices: AtomicU64,
    /// A demand switch reloads: requests queue through its Draining and Loading (same model).
    switching: AtomicBool,
    /// Knob, revert and A/B messages not yet handled (bounded, 9.8).
    pending_knobs: AtomicUsize,
    /// Admission refusals, every one (`EngineSample.tot_rejected`).
    rejected: AtomicU64,
    /// Decode steps run eager on a refused capture ([`EngineHandle::eager_decode_steps`]).
    eager_decode: AtomicU64,
    /// Rejected spans not yet written; past [`REJECTS_QUEUED`] a refusal is counted but has no span.
    pending_rejects: AtomicUsize,
    /// The engine thread, joined by `shutdown`.
    thread: Mutex<Option<JoinHandle<()>>>,
    /// Startup overrides the engine did not apply: (knob, value, reason).
    rejected_overrides: Vec<(u16, f64, RejectReason)>,
    /// `EngineOptions.on_fault`, for `shutdown` after a fault.
    on_fault: OnFault,
    /// Every ring record's header carries it, telemetry on or off.
    engine_id: u64,
    /// NVML on the engine's device, for the sampler and each load's DeviceSample.
    device: Option<larch_sys::nvml::Nvml>,
    /// The adaptation version in force; 0 is the base model.
    adapter_version: AtomicU64,
    /// Set after each step that checked guesses.
    guesses: Mutex<GuessStats>,
}

/// Engine time: the test clock when one is set, else ns since `epoch`.
fn mono_ns(epoch: Instant, clock: Option<&AtomicU64>) -> u64 {
    clock.map_or_else(
        || epoch.elapsed().as_nanos() as u64,
        |c| c.load(Ordering::Acquire),
    )
}

impl Shared {
    fn now(&self) -> u64 {
        mono_ns(self.epoch, self.clock.as_deref())
    }
}

/// Host diagnostics, separate from the protected telemetry record contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    pub capacity_blocks: usize,
    /// Blocks in the KV pool.
    pub total_blocks: usize,
    pub indexed_blocks: usize,
    /// Entries on a pinned chain, which count against the pool, not `capacity_blocks`.
    pub pinned_blocks: usize,
    pub unique_blocks_used: usize,
    pub unique_tokens_used: usize,
    /// Conservative upper allowance for both bounded hash tables, excluding per-pool refcounts.
    pub index_metadata_bytes: usize,
    pub evictions: u64,
    pub lookups: u64,
    pub hits: u64,
    pub reused_prompt_tokens: u64,
    /// Actual prompt positions fed, including recomputation after preemption.
    pub computed_prompt_tokens: u64,
}

/// The engine's handle: `Clone + Send + Sync`.
#[derive(Clone)]
pub struct EngineHandle {
    shared: Arc<Shared>,
    tx: Sender<Msg>,
}

/// Loads the GGUF at `opts.model` and starts the engine; returns once it is Ready or failed.
pub fn start(mut opts: EngineOptions) -> Result<EngineHandle, LoadError> {
    // An explicit drafter wins, the same file too (`auto_drafter` reads that as a last choice).
    if opts.drafter.is_some() {
        opts.drafter_auto = None;
    }
    // A documents plan loads without a drafter, so the loader plans for none; the engine keeps the
    // person's drafter for the plan it goes back to.
    let drafter =
        (opts.documents.is_some()).then(|| (opts.drafter.take(), opts.drafter_auto.take()));
    let (load, weights) = model_loader(&opts)?;
    if let Some(d) = drafter {
        (opts.drafter, opts.drafter_auto) = d;
    }
    // An MoE's grouped experts reuse each weight tile across more tokens in a bigger chunk, and the
    // prompts of concurrent requests share one: a 30B MoE on an M5 (25 Oct), 1,955 tokens to first
    // token 6.8 s at 512, 5.3 s at 2048. Measured only on Metal, so only a macOS GPU gets it. A
    // `--set` of the knob still wins.
    // ponytail: macOS GPU stands in for Metal; extend to CUDA/HIP once their MoE prefill is timed
    if cfg!(target_os = "macos")
        && opts.device >= 0
        && weights.as_ref().is_some_and(|w| !w.experts.is_empty())
    {
        opts.knob_overrides
            .insert(0, (knob_id::PREFILL_CHUNK_TOKENS, MOE_PREFILL_CHUNK));
    }
    spawn(opts, load, weights)
}

/// `prefill_chunk_tokens` for a mixture-of-experts model (see [`start`]).
const MOE_PREFILL_CHUNK: f64 = 2048.0;

#[cfg(test)]
type FakeProps = std::rc::Rc<dyn Fn(DeviceProps) -> DeviceProps>;

#[cfg(test)]
thread_local! {
    /// Stands a fake device's properties in for the real ones in the load's readings.
    static FAKE_PROPS: std::cell::RefCell<Option<FakeProps>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// Stands a pool size in for the file's: the tiny fixture's (34 MB) is past all its experts.
    static FAKE_POOL: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// The prefill streaming pool a load of `header` asks for, as the plan reserves it.
fn pool_bytes(header: &Gguf) -> u64 {
    #[cfg(test)]
    if let Some(bytes) = FAKE_POOL.get() {
        return bytes;
    }
    Transformer::stream_pool_bytes(header)
}

/// The device's properties as the load plans with them.
fn device_props(dev: &Device) -> Result<DeviceProps, larch_kernels::KError> {
    let props = dev.props()?;
    #[cfg(test)]
    if let Some(fake) = FAKE_PROPS.with_borrow(Clone::clone) {
        return Ok(fake(props));
    }
    Ok(props)
}

fn model_loader(opts: &EngineOptions) -> Result<(Loader, Option<Weights>), LoadError> {
    // An automatic drafter: this loader plans with it, and the plain one (no drafter, the same
    // readings) loads wherever the drafter would cost a chat.
    let with;
    let mut plain = None;
    let opts = match opts.auto_drafter() {
        Some(auto) => {
            let p = EngineOptions {
                drafter: None,
                drafter_auto: None,
                ..opts.clone()
            };
            let (load, weights) = model_loader(&p)?;
            let checks = engine::check_spares(&p, None);
            plain = Some((load, weights.expect("a model's weights"), checks));
            with = EngineOptions {
                drafter: Some(auto.clone()),
                ..p
            };
            &with
        }
        None => opts,
    };
    // ponytail: a second header parse (the loader parses it too); take the bytes from the model
    // once larch-model exposes them, or sooner if the parse shows in the start-to-Ready time.
    let path = opts.model.clone();
    let file = File::open(&path).map_err(|_| LoadError::Io)?;
    let len = file.metadata().map_err(|_| LoadError::Io)?.len();
    // The streaming pool reads host experts through a map of this file: where the platform or the
    // filesystem cannot map it, the model builds no pool, so the plan pays nothing for one.
    let maps = larch_kernels::Mmap::map(&file).is_ok();
    let header = Gguf::read(BufReader::new(file), len)?;
    let c = ModelConfig::from_gguf(&header)?;
    let weights = match &plain {
        Some((_, w, _)) => w.clone(),
        None => Weights::of(&header),
    };
    *weights.budget.lock().expect("load budget") = (opts.max_context, opts.kv_pool_fraction);
    let (bytes, experts, host, budget, seqs, graph_seats, ledger) = (
        weights.bytes,
        weights.experts.clone(),
        weights.host_layers.clone(),
        weights.budget.clone(),
        weights.seqs.clone(),
        weights.graph_seats.clone(),
        weights.ledger.clone(),
    );
    let asked = weights.asked.clone();
    let forced = opts.cpu_expert_layers;
    // `start` reads the header before it sets an MoE's startup chunk, so only a chunk set by hand
    // is here; a reload's options carry the startup one, and its load keeps that.
    let chunk_set = (opts.knob_overrides.iter()).any(|&(k, _)| k == knob_id::PREFILL_CHUNK_TOKENS);
    let seqs_cap = larch_records::knob_spec(knob_id::MAX_BATCH_SEQS)
        .expect("registered")
        .max as u32;
    let mtp = opts.mtp();
    let drafter = (opts.drafter.as_deref())
        .filter(|_| !mtp)
        .map(larch_model::DraftShape::read)
        .transpose()?;
    // What `Loaded::load` takes after this loader, which the plan must leave room for, and the
    // check-step graph slots beside the decode buckets.
    let checks = engine::check_spares(opts, drafter.as_ref().map(larch_model::DraftShape::block));
    let spares_of = |n| match n {
        0 => Ok(0),
        n => Transformer::check_bytes(&header, &c, n),
    };
    let spares = spares_of(checks)?;
    // The plan without the automatic drafter: its check spares and their bytes.
    let plain_checks = match &plain {
        Some((_, _, n)) => Some((*n, spares_of(*n)?)),
        None => None,
    };
    // Bounded by the file: a crafted header's unused tensors may claim any size, and the plan's
    // sums below are plain additions.
    let mtp_weights = mtp.then(|| Transformer::nextn_bytes(&header, &c).min(len));
    let block_experts = mtp_weights.map_or(0, |w| {
        let n = c.n_layers;
        (["gate", "up", "down"].iter())
            .filter_map(|p| header.tensor(&format!("blk.{n}.ffn_{p}_exps.weight")))
            .fold(0u64, |a, t| a.saturating_add(t.bytes))
            .min(w)
    });
    let host_plan = plan_experts(&experts, block_experts);
    let moe = MoeCfg::from_gguf(&header, c.arch)?;
    // The expert cache (per-expert residency): 0 keeps whole host layers, 1 places experts
    // statically, 2 adds dynamic residency. Only LARCH_EXPERT_CACHE takes it: on the 5070 Ti at
    // W4@8 (the hybrid MoE target, 27 Nov) the cache from a recorded histogram ran 188.5 tok/s
    // against 237.1 for whole layers and the prefill pool, so neither seats nor a histogram beside
    // the model pick it.
    // ponytail: only a load that takes the cache records routes. Pick it by seats again, and
    // record on whole-layer loads, if a measured per-expert plan beats whole layers with the pool.
    let cache = std::env::var("LARCH_EXPERT_CACHE")
        .ok()
        .and_then(|v| expert_cache(&v));
    let kv_want = opts.kv_dtype;
    let load = Box::new(move |dev: &Device, s: &Stream, rows| {
        let before = device_props(dev).map_err(|e| LoadError::Backend(e.code()))?;
        // The pool's pages as `Loaded::load` picks them after this loader, so the plan prices them.
        let kv = engine::kv_dtype(s, kv_want, &c).map_err(|e| LoadError::Backend(e.code()))?;
        // The effective max_context and fraction, as the fit after the load sees them.
        let (max_context, fraction) = *budget.lock().expect("load budget");
        // The most state slots a hybrid gets (`Loaded::load` sets it); the plan gives what fits.
        let cap = seqs.load(Ordering::Relaxed);
        // What a load of `rows` rows and `draft_seats` draft caches plans with; without
        // `drafting`, an automatic drafter's plan without it.
        let terms = |rows: u32, draft_seats: usize, drafting: bool| {
            let (drafter, mtp_weights, (checks, spares)) = match plain_checks {
                Some(plain) if !drafting => (None, None, plain),
                _ => (drafter.as_ref(), mtp_weights, (checks, spares)),
            };
            let guess = (drafter, mtp_weights, checks, spares);
            row_terms(
                &header,
                &c,
                moe,
                &before,
                bytes,
                max_context,
                rows,
                draft_seats,
                guess,
            )
        };
        // `cap` is the startup max_batch_seqs, capped at 8 (above DRAFT_SEATS): the engine's seats.
        // A cell, so a drafter's smaller attempts below re-plan through the same closures.
        let t = std::cell::Cell::new(terms(rows, guess::draft_seats(cap), true)?);
        let graph_seats = graph_seats.load(Ordering::Relaxed);
        // The fewest state slots the plan keeps: more host layers before fewer of them. Only a
        // hybrid's seats cost state; another model's KV floor stays at one seat's.
        let floor = match c.arch.hybrid() {
            true => asked.load(Ordering::Relaxed).clamp(1, cap.max(1)),
            false => 1,
        };
        // A hybrid's seats stop at its state slots, so the buckets past them never capture.
        let seats_of = |slots: u32| {
            if c.arch.hybrid() {
                graph_seats.min(slots)
            } else {
                graph_seats
            }
        };
        let held = s.capture_arena_held();
        let diagnostics = std::env::var("LARCH_FIT_DIAGNOSTICS").is_ok_and(|v| v == "1");
        let hybrid_moe = c.arch == larch_model::Arch::Qwen35Moe;
        let routers = routers_bytes(&c, moe);
        // The prefill streaming pool, which only host layers use: a plan that moves any pays for
        // it as for the state slots, ~2.4x prefill (#489, pp512) for ~0.6 ms a token at one more
        // host layer (its want, as `Loaded::load`'s fit admits it).
        // `stream_pool_bytes` is the file's largest expert tensor, an upper bound on the host
        // layers' (`stream_pool_want`), so the plan never reserves less than the model asks.
        let pool = match c.arch.hybrid() && cache.is_none() && maps && stream_rows_for(&before) > 0
        {
            true => pool_bytes(&header),
            false => 0,
        };
        // The serving buffers, the f16 copy of the routers the model keeps beside the weights,
        // and a hybrid's own scratch and `slots` slots of recurrent state.
        let reserve =
            |slots: u32| plan_reserve(&before, &t.get(), routers, (seats_of(slots), held), slots);
        // The fewest host layers for `slots` state slots, with the check's `swing` as the check
        // after the load counts it, so a plan the check refuses is not loaded first.
        let layers_for = |slots: u32, swing: u64| {
            let (reserve, bytes) = (reserve(slots) + swing, t.get().bytes);
            let (ctx, f) = (max_context, fraction);
            fit::host_expert_layers_with_pool(
                &before, bytes, &host_plan, &c, ctx, f, reserve, slots, pool, kv,
            )
        };
        // Whether `n` host layers at the floor's slots leave the pool its room, whether or not a
        // layer moved for it: the slots past the floor then keep that room, so a second slot never
        // spends it, and the check after the load keeps it ([`pool_checked`]).
        let covers = |n: u32, swing: u64| {
            let (ctx, f, bytes) = (max_context, fraction, t.get().bytes);
            let with = reserve(floor) + swing + pool;
            let fits = |k| k <= n;
            pool > 0
                && n > 0
                && fit::host_expert_layers_that_fit(
                    &before, bytes, &host_plan, &c, ctx, f, with, floor, kv,
                )
                .is_some_and(fits)
        };
        // The most state slots up to the cap that `n` host layers leave room for; past the pool's
        // room, a count that no number of layers fits counts as more than every layer, so the
        // slots never take it (with no pool the check after the load gives slots up instead).
        let slots_for = |n: u32, swing: u64| {
            let (ctx, f, bytes) = (max_context, fraction, t.get().bytes);
            let covers = covers(n, swing);
            fit::most_state_slots(cap, n, |s| match covers {
                true => {
                    let reserve = reserve(s) + swing + pool;
                    fit::host_expert_layers_that_fit(
                        &before, bytes, &host_plan, &c, ctx, f, reserve, s, kv,
                    )
                    .unwrap_or(u32::MAX)
                }
                false => {
                    let reserve = reserve(s) + swing;
                    fit::host_expert_layers(&before, bytes, &host_plan, &c, ctx, f, reserve, s, kv)
                }
            })
        };
        // What one placement keeps on the device, its plan's terms and whether it paid for the graphs.
        let placed = |host_bytes: u64, count: u32, planned: bool, slots: u32| {
            let (seats, t) = (seats_of(slots), t.get());
            let (graph_slots, graph_arena) =
                fit::graph_need(&before, seats, seats != 0, held, t.checks);
            let mut l = ledger.lock().expect("load ledger");
            (l.free_before, l.activation, l.routers) = (before.vram_free, t.activation, routers);
            l.hybrid = t.hybrid_base + u64::from(slots) * t.slot_bytes;
            (l.scratch, l.graph_slots, l.graph_arena) = (t.scratch, graph_slots, graph_arena);
            (l.rows, l.draft_seats) = (t.rows, t.draft_seats);
            l.weights_on_device = t.bytes.saturating_sub(host_bytes);
            l.graphs_paid = graphs_paid(planned, seats, count, experts.len());
        };
        // A loaded model's recurrent state (before the free figure, so the fit counts it as the
        // model's), then the check after the load, with CUDA's measured free swing: the fit reads
        // free memory again, and a low reading there would cost the graphs. A load short of it
        // gives up state slots, down to one, before it asks for more host layers.
        // `count` is the host (or split) layers; with all of them there the graphs give way.
        let settle = |mut m: Transformer,
                      tag: &str,
                      count: u32,
                      mapped: bool,
                      planned: bool,
                      tried: bool,
                      mut slots: u32|
         -> Result<Transformer, Refusal> {
            let (last, t) = (count as usize == experts.len(), t.get());
            let left = scratch_after_load(t.scratch, mtp_weights);
            let covers = covers(count, load_check_need(before.backend, hybrid_moe, tried, 0));
            let stream = pool_checked(covers, last, m.stream_pool_want().0, pool);
            let need = |slots: u32| {
                let beside = kept_beside(&before, left, seats_of(slots), held, last, t.checks);
                let (ctx, f) = (max_context, fraction);
                let need =
                    fit::serving_need(&before, &c, ctx, f, beside, mapped, slots, kv) + stream;
                load_check_need(before.backend, hybrid_moe, tried, need)
            };
            loop {
                if c.arch.hybrid() {
                    m.alloc_state(s, slots).map_err(|e| (e, 0))?;
                    ledger.lock().expect("load ledger").hybrid =
                        t.hybrid_base + u64::from(slots) * t.slot_bytes;
                }
                let free = device_props(dev)
                    .map_err(|e| (LoadError::Backend(e.code()), 0))?
                    .vram_free;
                if diagnostics {
                    eprintln!("{tag},{count},{free},{},{slots}", need(slots));
                }
                // A slot costs 64 MB on the hybrid MoE target, a layer 381 MB: the most slots
                // down to `floor` that the freed state covers, then measured again.
                if free >= need(slots) || !c.arch.hybrid() {
                    return fits_after_load(m, planned, before.vram_free, free, need(slots));
                }
                // Only a planned GPU load with a layer left to move can trade one for a slot.
                let keep = planned && !last && before.backend == larch_kernels::Backend::Cuda;
                let low = settle_low(slots, floor, keep);
                match settle_slots(free, slots, low, t.slot_bytes, need) {
                    Ok(k) => slots = k,
                    // Short even at the floor: the host layers cover what it leaves.
                    Err(freed) if planned || slots == low => {
                        return fits_after_load(m, planned, before.vram_free, freed, need(low));
                    }
                    Err(_) => slots = low,
                }
            }
        };
        // The CPU reference keeps every expert where it is (fit::host_expert_layers), so only a GPU
        // plan moves experts.
        let planned = forced.is_none()
            && !experts.is_empty()
            && before.backend != larch_kernels::Backend::Cpu;
        // A drafter's smaller attempts at the startup terms (below).
        let first = t.get();
        let drafting = drafter.is_some() || mtp;
        let small = (!chunk_set).then_some(seqs_cap + engine::PREFILL_CHUNK_SCRATCH as u32);
        let tries =
            || fit::draft_fits(drafting, first.rows, small, first.draft_seats).filter(|_| planned);
        // An automatic drafter: the whole-layer plans with it (its first attempt at no host layer,
        // as the load below takes it) and without it, before any load. One that would cost a chat
        // refuses, and the plain loader loads (`model_loader`).
        if plain_checks.is_some() {
            let swing = load_check_need(before.backend, hybrid_moe, false, 0);
            // What `x`'s plan leaves the chats; its KV blocks as the fit after the load counts.
            let chats = |x: RowTerms| {
                t.set(x);
                let n = layers_for(floor, swing);
                let slots = if c.arch.hybrid() {
                    slots_for(n, swing)
                } else {
                    0
                };
                let on_device = x.bytes.saturating_sub(experts[..n as usize].iter().sum());
                let free = before.vram_free.saturating_sub(on_device + reserve(slots));
                let (ctx, mapped) = (max_context.min(c.ctx_train), n == 0);
                let plan = fit::fit(
                    &before,
                    free,
                    &c,
                    ctx,
                    fraction,
                    0,
                    slots.max(1),
                    false,
                    held,
                    mapped,
                    0,
                    0,
                    kv,
                );
                fit::Chats {
                    host_layers: n,
                    state_slots: slots,
                    kv_blocks: plan.map_or(0, |p| p.kv_blocks),
                }
            };
            let mut with = first;
            for (rows, draft_seats) in tries() {
                t.set(terms(rows, draft_seats as usize, true)?);
                if layers_for(floor, swing) == 0 {
                    with = t.get();
                    break;
                }
            }
            let without = terms(first.rows, first.draft_seats as usize, false)?;
            let (with, without) = (chats(with), chats(without));
            t.set(first);
            ledger.lock().expect("load ledger").auto = Some((with, without, false));
            if !fit::drafter_keeps(with, without) {
                return Err(LoadError::DoesNotFit { need: 0, have: 0 });
            }
        }
        // A crafted expert_count has no histogram: the whole-layer load below refuses that file.
        let hist_path = Histogram::path(&path);
        // What the load records from, usable or not; only counts make it a per-expert plan.
        let start = moe
            .filter(|_| planned && cache.is_some())
            .and_then(|m| Histogram::for_load(&header, &c, m, &hist_path, len));
        // A per-expert attempt moved memory, so a fallback's check allows CUDA's free swing too.
        let mut tried = false;
        if let (Some(m), Some(hist)) = (moe, plan_from(start.as_ref())) {
            let dynamic = cache == Some(true);
            let log = Transformer::route_log_bytes(&c, m, rows);
            let tensors = header.tensors.len() as u64;
            // Its check allows the swing; its slots are what whole layers would leave room for.
            let swing = post_load_need(before.backend, 0);
            let slots = slots_for(layers_for(floor, swing), swing);
            let reserve = reserve(slots);
            let (bytes, host_embed) = (t.get().bytes, t.get().host_embed);
            let charge = fit::expert_cache_bytes(&before, log, tensors, dynamic);
            let largest = experts.iter().map(|&b| b / u64::from(m.experts)).max();
            let margin = replan_margin(before.backend, largest.unwrap_or(0));
            let loaded = replan_on_shortfall(charge, margin, |charge| {
                let resident = fit::expert_residency(
                    &before,
                    bytes,
                    &experts,
                    &hist.counts,
                    m.experts,
                    &c,
                    max_context,
                    fraction,
                    reserve,
                    charge,
                    slots,
                    kv,
                );
                let Some(resident) = resident else {
                    return Err((LoadError::DoesNotFit { need: 0, have: 0 }, 0));
                };
                let split = resident
                    .iter()
                    .filter(|r| r.len() < m.experts as usize)
                    .count();
                if diagnostics {
                    let on: usize = resident.iter().map(Vec::len).sum();
                    eprintln!(
                        "larch_expert_cache_plan,{split},{on},{},{},{bytes},{},{max_context},{fraction},{rows}",
                        before.vram_free,
                        before.vram_total,
                        reserve + charge
                    );
                }
                host.store(split as u32, Ordering::Relaxed);
                tried = true;
                let on_host = resident.iter().zip(&experts).map(|(r, &e)| {
                    (e / u64::from(m.experts)) * (u64::from(m.experts) - r.len() as u64)
                });
                placed(on_host.sum(), split as u32, true, slots);
                let plan = ExpertPlan { resident, dynamic };
                let place = Placement::default();
                match mtp {
                    true => Transformer::load_cached_mtp(&path, dev, s, rows, &plan),
                    false => Transformer::load_planned(
                        &path,
                        dev,
                        s,
                        rows,
                        place,
                        Some(&plan),
                        host_embed,
                    ),
                }
                .and_then(|mut t| {
                    t.record_routes(s, hist_path.clone(), hist.clone())
                        .map_err(|e| LoadError::Backend(e.code()))?;
                    Ok(t)
                })
                .map_err(|e| (e, 0))
                .and_then(|t| {
                    let tag = "larch_expert_cache_load";
                    settle(t, tag, split as u32, split == 0, true, true, slots)
                })
            });
            // Per-expert placement that does not fit falls back to whole layers.
            let oom = LoadError::Backend(larch_kernels::KError::Oom.code());
            match loaded {
                Err(LoadError::DoesNotFit { .. }) => {}
                Err(e) if e == oom => {}
                r => return r.map(|m| Box::new(m) as Box<dyn Model>),
            }
        }
        let swing = load_check_need(before.backend, hybrid_moe, tried, 0);
        let layers = match forced {
            Some(n) => n.min(experts.len() as u32),
            None if experts.is_empty() => 0,
            None => {
                // The fewest layers at the floor's state slots; the slots then take what they
                // leave.
                let n = layers_for(floor, swing);
                if diagnostics {
                    let (reserve, bytes) = (reserve(slots_for(n, swing)) + swing, t.get().bytes);
                    let covers = covers(n, swing);
                    eprintln!(
                        "larch_host_experts_plan,{n},{},{},{bytes},{reserve},{max_context},{fraction},{rows},{pool},{covers}",
                        before.vram_free, before.vram_total
                    );
                }
                n
            }
        };
        let stream_rows = stream_rows_for(&before);
        let whole = |layers| {
            host.store(layers, Ordering::Relaxed);
            let slots = slots_for(layers, swing);
            placed(
                experts[..layers as usize].iter().sum(),
                layers,
                planned,
                slots,
            );
            let place = Placement {
                cpu_expert_layers: layers,
                stream_rows,
            };
            // A refusal of the load itself names one allocation, not what more layers must free.
            let unknown = |e| (e, 0);
            let (rows, host_embed) = (t.get().rows, t.get().host_embed);
            let m = match mtp {
                true => Transformer::load_mtp(&path, dev, s, rows, place),
                false => Transformer::load_planned(&path, dev, s, rows, place, None, host_embed),
            }
            .map_err(unknown)?;
            let mut m = settle(
                m,
                "larch_host_experts_load",
                layers,
                layers == 0,
                planned,
                tried,
                slots,
            )?;
            // After the check, so plan and check stay whole-layer ones. A placement hint: losing
            // it costs the next load its plan, never this load.
            if let Some(h) = &start
                && let Err(e) = m.record_routes(s, hist_path.clone(), h.clone())
            {
                eprintln!("larch_expert_histogram_record,{e:?}");
            }
            Ok(m)
        };
        // A host layer costs a drafter its speed-up, so its smaller attempts come first, each at no
        // host layer and through the check after its load. The startup terms' refusal stands for
        // their first load in the fallback, which plans with them again.
        let mut refused = None;
        for (rows, draft_seats) in tries() {
            t.set(terms(rows, draft_seats as usize, true)?);
            if layers_for(floor, swing) > 0 {
                continue;
            }
            match whole(0) {
                Err(r @ (LoadError::DoesNotFit { .. }, _)) if t.get() == first => refused = Some(r),
                Err((LoadError::DoesNotFit { .. }, _)) => {}
                Ok(m) if t.get() != first => {
                    let chunk = rows.saturating_sub(seqs_cap);
                    eprintln!(
                        "larch_drafter_fit,chunk={chunk},draft_seats={draft_seats},host_layers=0"
                    );
                    return Ok(Box::new(m));
                }
                r => return r.map(|m| Box::new(m) as Box<dyn Model>).map_err(|r| r.0),
            }
        }
        t.set(first);
        more_host_layers_on_refusal(layers, &host_plan, planned, |n| match refused.take() {
            Some(r) => Err(r),
            None => whole(n),
        })
        .map(|m| Box::new(m) as Box<dyn Model>)
    });
    let Some((plain, ..)) = plain else {
        return Ok((load, Some(weights)));
    };
    let (ledger, host) = (weights.ledger.clone(), weights.host_layers.clone());
    let auto: Loader = Box::new(move |dev: &Device, s: &Stream, rows| {
        let loaded = load(dev, s, rows);
        let mut l = ledger.lock().expect("load ledger");
        let Some((mut with, without, _)) = l.auto else {
            return loaded;
        };
        // Past the plan, what the load got decides; one short of memory may fit without the
        // drafter. The KV blocks stay the plan's: the fit after the load counts them.
        let oom = LoadError::Backend(larch_kernels::KError::Oom.code());
        match &loaded {
            Ok(m) => {
                with.host_layers = host.load(Ordering::Relaxed);
                with.state_slots = m.state_slots();
            }
            Err(LoadError::DoesNotFit { .. }) => {}
            Err(e) if *e == oom => {}
            Err(_) => return loaded,
        }
        let kept = loaded.is_ok() && fit::drafter_keeps(with, without);
        l.auto = Some((with, without, kept));
        drop(l);
        eprintln!(
            "larch_drafter_auto,kept={},host_layers={}/{},state_slots={}/{},kv_blocks={}/{}",
            u8::from(kept),
            with.host_layers,
            without.host_layers,
            with.state_slots,
            without.state_slots,
            with.kv_blocks,
            without.kv_blocks
        );
        match kept {
            true => loaded,
            false => {
                drop(loaded);
                plain(dev, s, rows)
            }
        }
    });
    Ok((auto, Some(weights)))
}

/// [`Placement::stream_rows`] for a load onto `before`.
fn stream_rows_for(before: &DeviceProps) -> u32 {
    // CUDA is the one backend measured; unified memory reads the host experts in place anyway.
    if before.backend == larch_kernels::Backend::Cuda
        && before.flags & DeviceProps::CAP_UNIFIED == 0
    {
        larch_kernels::routes()
            .prefill_stream_rows
            .unwrap_or(larch_model::PREFILL_STREAM_ROWS)
    } else {
        0
    }
}

/// LARCH_EXPERT_CACHE: `Some(dynamic)` for per-expert residency (1 static, 2 dynamic), `None`
/// (unset, 0) for whole host layers.
fn expert_cache(v: &str) -> Option<bool> {
    match v {
        "1" => Some(false),
        "2" => Some(true),
        _ => None,
    }
}

/// The histogram a per-expert plan starts from; `None` keeps whole layers. `Histogram::for_load`
/// reads a missing or corrupt file as uniform, and Drop writes all zeros after a load that never
/// stepped: without counts a per-expert plan only spreads the leading layers across the device.
fn plan_from(start: Option<&Histogram>) -> Option<&Histogram> {
    start.filter(|h| h.counts.iter().any(|&n| n > 0))
}

/// `load(reserve)` and, when the check after that load found it short, `load` once more with the
/// shortfall and `margin` reserved. A per-expert plan leaves under one expert's bytes unused, so
/// the shortfall alone fits only when the second plan's slack is at least the first's; `margin`,
/// the largest expert, makes it fit by construction. What the plan did not count (the MoE target on a
/// 5070 Ti: about 118 MB) costs experts, not the placement.
fn replan_on_shortfall<T>(
    reserve: u64,
    margin: u64,
    mut load: impl FnMut(u64) -> Result<T, Refusal>,
) -> Result<T, LoadError> {
    match load(reserve) {
        Err((LoadError::DoesNotFit { .. }, short)) if short > 0 => load(reserve + short + margin),
        r => r,
    }
    .map_err(|r| r.0)
}

/// [`RowTerms`] for a load of `rows` rows and `draft_seats` draft caches at `max_context`, with the
/// drafter's shape, the MTP block's weights, and the check spares and their bytes (`guess`).
#[allow(clippy::too_many_arguments)] // what the loader's plan and the documents plan both read
fn row_terms(
    header: &Gguf,
    c: &ModelConfig,
    moe: Option<MoeCfg>,
    before: &DeviceProps,
    bytes: u64,
    max_context: u32,
    rows: u32,
    draft_seats: usize,
    guess: (Option<&larch_model::DraftShape>, Option<u64>, u32, u64),
) -> Result<RowTerms, LoadError> {
    let (drafter, mtp_weights, checks, spares) = guess;
    // A drafter reads the target's token embedding on the device (`Model::embed_and_head`).
    let host_embed = mtp_weights.is_none()
        && drafter.is_none()
        && Transformer::embeds_on_host(before, c, moe.is_some());
    // The plan places what the device holds: the weights but those the load keeps off it.
    let off = Transformer::off_device_bytes(header, c, before, host_embed);
    let scratch = fit::serving_scratch(before, c, moe, rows, max_context)
        + fit::guess_bytes(
            spares,
            drafter,
            mtp_weights,
            c,
            rows,
            max_context,
            draft_seats,
        );
    // A hybrid's own scratch, and the recurrent state each slot adds to it.
    let hybrid_base = Transformer::hybrid_bytes(header, c, rows, 0)?;
    let slot_bytes = Transformer::hybrid_bytes(header, c, rows, 1)? - hybrid_base;
    let activation = fit::activation_bytes(c, moe, rows);
    Ok(RowTerms {
        rows,
        draft_seats: draft_seats as u32,
        bytes: bytes.saturating_sub(off),
        host_embed,
        checks,
        scratch,
        hybrid_base,
        slot_bytes,
        activation,
    })
}

/// The f16 copy of the routers the model keeps beside the weights.
fn routers_bytes(c: &ModelConfig, moe: Option<MoeCfg>) -> u64 {
    moe.map_or(0, |m| {
        2 * u64::from(m.experts) * u64::from(c.d_model) * u64::from(c.n_layers)
    })
}

/// What a plan with `slots` state slots keeps free beside the weights: the serving buffers and
/// graph reserve (`graphs`: the seats they cover and whether the arena is held), the activations,
/// the routers' copy and the hybrid's state.
fn plan_reserve(
    before: &DeviceProps,
    t: &RowTerms,
    routers: u64,
    graphs: (u32, bool),
    slots: u32,
) -> u64 {
    let hybrid = t.hybrid_base + u64::from(slots) * t.slot_bytes;
    let beside = kept_beside(before, t.scratch, graphs.0, graphs.1, false, t.checks);
    t.activation + beside + routers + hybrid
}

/// What the documents plan reads from the loaded model's file, which only a reload changes.
// ponytail: the whole parsed header is kept (the plan's helpers read it); keep only the terms they
// use if its memory shows beside a small card's load.
pub(crate) struct DocTerms {
    header: Gguf,
    c: ModelConfig,
    moe: Option<MoeCfg>,
    weights: Weights,
    /// The load checks' spares, and their bytes, as a load with no drafter plans them.
    checks: (u32, u64),
    /// Whether the platform maps the file (the streaming pool's precondition).
    maps: bool,
}

impl DocTerms {
    /// Reads `opts.model`'s header, once per load ([`document_plan`] runs from the result).
    pub(crate) fn read(opts: &EngineOptions) -> Result<Self, LoadError> {
        let file = File::open(&opts.model).map_err(|_| LoadError::Io)?;
        let len = file.metadata().map_err(|_| LoadError::Io)?.len();
        let maps = larch_kernels::Mmap::map(&file).is_ok();
        let header = Gguf::read(BufReader::new(file), len)?;
        let c = ModelConfig::from_gguf(&header)?;
        let moe = MoeCfg::from_gguf(&header, c.arch)?;
        let weights = Weights::of(&header);
        let plain = EngineOptions {
            drafter: None,
            drafter_auto: None,
            ..opts.clone()
        };
        let checks = engine::check_spares(&plain, None);
        let spares = match checks {
            0 => 0,
            n => Transformer::check_bytes(&header, &c, n)?,
        };
        Ok(Self {
            header,
            c,
            moe,
            weights,
            checks: (checks, spares),
            maps,
        })
    }
}

/// [`fit::document_fit`] for a document of `tokens` on `before`, the device as the loaded model's
/// load saw it: the header's terms at the documents plan's slots and no drafter (the plan's
/// reload runs without one: its memory is not in the fit), from the same helpers the loader's plan
/// uses. `rows`, `graphs` and `held` are the engine's own; `max_seats` is the load's
/// `max_batch_seqs`; `kv` the loaded pool's page dtype. Where host experts prefill on the CPU (no streaming pool: unified memory,
/// HIP, an MoE that is not hybrid), the plan keeps no more host layers than `base_host_layers`,
/// the load outside the plan's: each layer more reads the whole document's experts on the CPU (a
/// 77-page paper took 40 minutes to read on an M5, 21 Nov), so a longer one is read by section.
fn document_plan(
    t: &DocTerms,
    before: &DeviceProps,
    (rows, graphs, held): (u32, bool, bool),
    max_seats: u32,
    base_host_layers: u32,
    tokens: u32,
    kv: DType,
) -> Result<DocFit, LoadError> {
    let DocTerms {
        header,
        c,
        moe,
        weights,
        checks: (checks, spares),
        maps,
    } = t;
    let (moe, maps) = (*moe, *maps);
    let guess = (None, None, *checks, *spares);
    let terms = |ctx, slots| {
        let draft_seats = guess::draft_seats(slots);
        row_terms(
            header,
            c,
            moe,
            before,
            weights.bytes,
            ctx,
            rows,
            draft_seats,
            guess,
        )
    };
    terms(0, fit::DOC_SLOTS)?;
    let routers = routers_bytes(c, moe);
    let swing = load_check_need(
        before.backend,
        c.arch == larch_model::Arch::Qwen35Moe,
        false,
        0,
    );
    // The terms only fail on the header, which the call above read; a failure there is no fit.
    // A hybrid's graph seats stop at its state slots, as the loader's `seats_of` has it.
    let reserve = |ctx, slots| {
        terms(ctx, slots).map_or(u64::MAX, |t| {
            plan_reserve(
                before,
                &t,
                routers,
                (if graphs { slots } else { 0 }, held),
                slots,
            ) + swing
        })
    };
    let pool = match c.arch.hybrid() && maps && stream_rows_for(before) > 0 {
        true => pool_bytes(header),
        false => 0,
    };
    let host_plan = plan_experts(&weights.experts, 0);
    let bytes = terms(0, fit::DOC_SLOTS)?.bytes;
    // The check after the load keeps the pool's room ([`pool_checked`]), moving a layer more where
    // the device reads less free than planned. ponytail: a load with every layer on the host may
    // still lose its pool; cap the plan too if a CUDA documents reload logs that (`streams=0`).
    let most = (pool == 0).then_some(base_host_layers);
    Ok(fit::document_fit(
        before, bytes, &host_plan, c, reserve, pool, max_seats, most, tokens, kv,
    ))
}

/// What a load onto `before` keeps free beside the model for fit(): the serving `scratch` and, with
/// decode graphs on at load for `seats` (0: off), fit()'s whole graph reserve, so the host-expert
/// plan and its check move the layers that leave every capture its room. With every expert layer
/// on the host (`last`) the graphs give way: fit() then runs them eager rather than refuse the load.
/// `checks`: a guessing load's check-step slots ([`engine::check_spares`]).
fn kept_beside(
    before: &DeviceProps,
    scratch: u64,
    seats: u32,
    held: bool,
    last: bool,
    checks: u32,
) -> u64 {
    let (slots, arena) = fit::graph_need(before, seats, seats != 0 && !last, held, checks);
    scratch + slots + arena
}

/// The part of the plan's `scratch` still to come after the load: `mtp`, the target's MTP block's
/// weights, is in it for the plan, but the load has taken those, so they are out of `free`.
fn scratch_after_load(scratch: u64, mtp: Option<u64>) -> u64 {
    scratch.saturating_sub(mtp.unwrap_or(0))
}

/// Each layer's routed expert bytes for the whole-layer plan, the MTP block's (`block`) on the
/// first: [`Transformer::load_mtp`] keeps them in host memory with any host layer.
fn plan_experts(experts: &[u64], block: u64) -> Vec<u64> {
    let mut e = experts.to_vec();
    if let Some(f) = e.first_mut() {
        *f = f.saturating_add(block);
    }
    e
}

/// Whether a load's plan paid for its graphs: it chose `layers` of `total` host layers with the
/// graphs' reserve in the room (`seats` != 0). With every layer on the host they gave way.
fn graphs_paid(planned: bool, seats: u32, layers: u32, total: usize) -> bool {
    planned && seats != 0 && layers > 0 && (layers as usize) < total
}

/// The fewest state slots the check after a load gives up to: the plan's `floor`, but where the
/// load can move another layer (`keep`: planned, on CUDA, a layer left) never below
/// [`engine::MANY_SEATS`] when the plan gave that many. Four chats on three seats queue one (W4@4
/// on the hybrid MoE target, 5070 Ti: 112.7 against 174 tok/s, its graphs eager), so such a load
/// short of its fourth slot refuses with the shortfall and moves one more host layer instead, at
/// one chat's cost of that layer. A setup of fewer than MANY_SEATS seats (max_batch_seqs 1 to 3)
/// never plans that many slots (they stop at [`engine::hybrid_slots`] of it), so it gives the slot
/// up and keeps the layer.
fn settle_low(slots: u32, floor: u32, keep: bool) -> u32 {
    match keep && slots >= engine::MANY_SEATS {
        true => floor.max(engine::MANY_SEATS),
        false => floor,
    }
    .min(slots)
}

/// The most state slots, from `slots` down to `low`, whose freed state (`slot_bytes` each) leaves
/// `free` at least `need` of them; else what `low` slots would leave free.
fn settle_slots(
    free: u64,
    slots: u32,
    low: u32,
    slot_bytes: u64,
    need: impl Fn(u32) -> u64,
) -> Result<u32, u64> {
    let freed = |k: u32| free + u64::from(slots - k) * slot_bytes;
    (low..slots)
        .rev()
        .find(|&k| freed(k) >= need(k))
        .ok_or(freed(low))
}

/// A refused load: its error, and the bytes more host layers must free when a load fitted but left
/// too little free; 0 when that is unknown.
type Refusal = (LoadError, u64);

/// `m`, or DoesNotFit with the shortfall when a planned load left less than `need` free, so
/// [`more_host_layers_on_refusal`] moves the layers that cover it. Dropping `m` frees its buffers
/// first.
fn fits_after_load<T>(
    m: T,
    planned: bool,
    before: u64,
    free: u64,
    need: u64,
) -> Result<T, Refusal> {
    if planned && free < need {
        let refused = LoadError::DoesNotFit {
            need: before.saturating_sub(free) + need,
            have: before,
        };
        return Err((refused, need - free));
    }
    Ok(m)
}

/// The streaming pool the check after a load keeps room for: the model's `want`, at most the
/// `pool` its plan reserved, when its host layers' placement `covers` the pool (a layer moved for
/// it or the layers moved anyway leave its room), else none. fit() admits the
/// pool only past every other reserve, and a load short of it prefills its host layers on the CPU
/// (Windows 5070 Ti, 24 Nov: a documents reload at 10 host layers cleared the check by 24 MB and
/// left the 149 MB pool out, so a 121-page read ran at 90 % CPU). With every layer on the host
/// (`last`) no more can move: that load keeps the old check, the pool optional.
fn pool_checked(covers: bool, last: bool, want: u64, pool: u64) -> u64 {
    match covers && !last {
        true => want.min(pool),
        false => 0,
    }
}

/// What the check after a load needs free for the fit's `need`: on CUDA also [`FREE_SWING`], so the
/// engine's fit, which reads free memory again after the loader returns, still finds `need`.
fn post_load_need(backend: larch_kernels::Backend, need: u64) -> u64 {
    match backend {
        larch_kernels::Backend::Cuda => need + FREE_SWING,
        _ => need,
    }
}

/// What the check after a load needs free: [`post_load_need`] where the swing was measured (a
/// discrete CUDA hybrid MoE, `hybrid_moe`) or a per-expert attempt moved memory first (`tried`),
/// else the fit's `need` alone, as before the swing.
fn load_check_need(
    backend: larch_kernels::Backend,
    hybrid_moe: bool,
    tried: bool,
    need: u64,
) -> u64 {
    if hybrid_moe || tried {
        post_load_need(backend, need)
    } else {
        need
    }
}

/// The re-plan's margin past the shortfall on `backend`, for `largest`-byte experts: one expert,
/// and on CUDA [`FREE_SWING`], since the re-plan's own reading can drop by it.
fn replan_margin(backend: larch_kernels::Backend, largest: u64) -> u64 {
    largest + post_load_need(backend, 0)
}

/// How far CUDA's free figure moved between two readings with nothing allocated between them, on
/// the 16 GB 5070 Ti under Windows: 10 fresh loads of one plan read 9 high and 1 low by exactly
/// 67,391,488 B (64.27 MiB, one quantum), the same figure as the 7 Nov per-expert loads and
/// re-plans (others there moved 50.3 MB and 66.8 MB). A plan that leaves less than this spare
/// refused at the fit, past the fallback to whole layers.
// ponytail: the measured quantum rounded up to 68 MiB, an allowance and not a bound. If a larger
// swing shows (the fit refuses or drops graphs after a passed check), raise this to that figure,
// or find what moves the reading (in-flight load work or WDDM residency; try a stream sync before
// it) and drop the allowance.
const FREE_SWING: u64 = 68 << 20;

/// `load(first)`, and while it refuses with `DoesNotFit` and the count is `planned`, `load` with the
/// fewest more host-expert layers whose experts (`experts[i]` bytes) cover the refusal's shortfall,
/// at least one, up to every layer. The plan sees memory, not the device's own limits (Metal's
/// maxBufferLength caps the one weights buffer), and that refusal comes before any upload.
fn more_host_layers_on_refusal<T>(
    first: u32,
    experts: &[u64],
    planned: bool,
    mut load: impl FnMut(u32) -> Result<T, Refusal>,
) -> Result<T, LoadError> {
    let mut n = first as usize;
    loop {
        match load(n as u32) {
            Err((LoadError::DoesNotFit { .. }, mut short)) if planned && n < experts.len() => {
                loop {
                    short = short.saturating_sub(experts[n]);
                    n += 1;
                    if short == 0 || n == experts.len() {
                        break;
                    }
                }
            }
            r => return r.map_err(|(e, _)| e),
        }
    }
}

/// [`start`] with another model (tests use [`larch_model::FakeModel`]).
/// Custom loaders default to eager decode; graph-safe models can explicitly enable decode graphs.
pub fn start_with(mut opts: EngineOptions, load: Loader) -> Result<EngineHandle, LoadError> {
    opts.knob_overrides.insert(0, (knob_id::DECODE_GRAPHS, 0.0));
    spawn(opts, load, None)
}

fn spawn(
    opts: EngineOptions,
    load: Loader,
    weights: Option<Weights>,
) -> Result<EngineHandle, LoadError> {
    let (ready_tx, ready_rx) = sync_channel(1);
    let thread = std::thread::Builder::new()
        .name("larch-engine".into())
        .spawn(move || engine::run(opts, load, weights, ready_tx))
        .map_err(|_| LoadError::Io)?;
    match ready_rx.recv() {
        Ok(h) => {
            let h = h?;
            *h.shared.thread.lock().expect("thread") = Some(thread);
            Ok(h)
        }
        // The loader panicked: a bug, so it surfaces as that panic.
        Err(_) => match thread.join() {
            Err(panic) => std::panic::resume_unwind(panic),
            Ok(()) => unreachable!("the engine thread replies before it returns"),
        },
    }
}

impl EngineHandle {
    /// Installs the final-record hook; its deadline bounds journal fsync after a fault.
    /// The ring is visible before invocation. A hung or panicking hook cannot delay fault exit.
    pub fn set_fault_flush(
        &self,
        hook: impl FnOnce(RingHandle, Instant) + Send + 'static,
    ) -> Result<(), EngineError> {
        let health = self.shared.health.lock().unwrap_or_else(|e| e.into_inner());
        if health.state == HealthState::Failed {
            return Err(EngineError::NotReady(health.state));
        }
        let previous = self
            .shared
            .host
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .flush
            .replace(Box::new(hook));
        drop(health);
        // Captures can have arbitrary destructors; dispose only outside the fault boundary.
        drop(previous);
        Ok(())
    }

    /// Admission (9.2): health, length, queue depth; the request then goes to the engine thread.
    /// A refusal is also written to the ring as a `Rejected` span.
    pub fn submit(&self, req: Request) -> Result<RequestHandle, EngineError> {
        self.admit(vec![(req, None)])
            .map(|mut h| h.pop().expect("one request"))
    }

    /// Admits `reqs` all or none, in order, in one engine turn: the scheduler starts them in the same
    /// step as far as seats, the chunk budget and the KV allow (#64). The first refusal refuses the
    /// group, and every request of it gets a Rejected span with that error.
    pub fn submit_group(&self, reqs: Vec<Request>) -> Result<Vec<RequestHandle>, EngineError> {
        self.admit(reqs.into_iter().map(|r| (r, None)).collect())
    }

    /// Teacher forcing ([`Forced`]): admitted like [`Self::submit`], as a greedy request whose prompt
    /// is `tokens[..prompt]` and whose `max_tokens` covers the rest plus one, with the same errors
    /// over every token (`prompt` 0, or no tokens, is `EMPTY_PROMPT`; `prompt` past `tokens.len()`
    /// is `NO_TOKENS`). The length limit is `tokens.len() ≤ max_context`: the last sampled token is
    /// never fed. The first forced run grows the logits scratch to every row a step can hold
    /// ((`max_batch_seqs`' max + `prefill_chunk_max`) × vocab × 4 bytes); if that allocation fails,
    /// backend allocation errors fail the engine; a fit-budget refusal ends only this run.
    pub fn force(&self, f: Forced) -> Result<ForcedHandle, EngineError> {
        let (n, p) = (f.tokens.len(), f.prompt as usize);
        // p == 0: an empty prompt; p > n: no tokens. admit() refuses both with a Rejected span.
        let (prompt, forced) = f.tokens.split_at(p.min(n));
        let max_tokens = if p > n {
            0
        } else {
            u32::try_from(n - p + 1).unwrap_or(u32::MAX)
        };
        let (tx, positions) = std::sync::mpsc::channel();
        let req = Request {
            session: f.session,
            keep_kv: f.keep_kv,
            ..Request::new(prompt.to_vec(), max_tokens)
        };
        let request = self
            .admit(vec![(req, Some((forced.to_vec(), tx, false)))])?
            .pop()
            .expect("one request");
        Ok(ForcedHandle { request, positions })
    }

    /// Admission of `reqs`, each with its teacher forcing if any, all or none ([`Self::submit_group`]).
    fn admit(
        &self,
        reqs: Vec<(Request, Option<Forcing>)>,
    ) -> Result<Vec<RequestHandle>, EngineError> {
        let sh = &self.shared;
        let health = sh.health.lock().unwrap_or_else(|e| e.into_inner());
        let t = sh.now().max(1);
        for (reserved, (req, forced)) in reqs.iter().enumerate() {
            let Err(e) = self.reserve(req, forced.as_ref().map(|f| &f.0[..]), health.state) else {
                continue;
            };
            sh.in_flight.fetch_sub(reserved, Ordering::Relaxed);
            for (req, _) in &reqs {
                let span = RequestSpan {
                    req_id: sh.next_id.fetch_add(1, Ordering::Relaxed),
                    t_submit_ns: t,
                    t_done_ns: t,
                    prompt_tokens: req.prompt.len().min(u32::MAX as usize) as u32,
                    session_id: req.session,
                    outcome: Outcome::Rejected.raw(),
                    finish: SpanFinish::None.raw(),
                    error: e.code(),
                    ..bytemuck::Zeroable::zeroed()
                };
                // Not Ready: the span is written only if the engine's last drain still finds it.
                sh.rejected.fetch_add(1, Ordering::Relaxed);
                if sh.pending_rejects.fetch_add(1, Ordering::Relaxed) < REJECTS_QUEUED {
                    let _ = self.tx.send(Msg::Rejected(span));
                } else {
                    sh.pending_rejects.fetch_sub(1, Ordering::Relaxed);
                }
            }
            return Err(e);
        }
        let (rs, handles): (Vec<_>, Vec<_>) = reqs
            .into_iter()
            .map(|(req, forced)| {
                let id = sh.next_id.fetch_add(1, Ordering::Relaxed);
                let (events_tx, events) = std::sync::mpsc::channel();
                let cancel = Arc::new(AtomicBool::new(false));
                let mut r = sched::Req::new(id, req, events_tx, cancel.clone(), t);
                if let Some((tokens, tx, next)) = forced {
                    r.forced = tokens;
                    r.observe = true;
                    r.score_next = next;
                    r.completion
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .positions = Some(tx);
                }
                let h = RequestHandle {
                    id,
                    events,
                    cancel,
                    engine: Arc::downgrade(sh),
                };
                (r, h)
            })
            .unzip();
        sh.host
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .requests
            .extend(rs.iter().map(|r| (r.id, r.completion.clone())));
        if self.tx.send(Msg::Submit(rs)).is_err() {
            let mut host = sh.host.lock().unwrap_or_else(|e| e.into_inner());
            for h in &handles {
                host.requests.remove(&h.id);
            }
            sh.in_flight.fetch_sub(handles.len(), Ordering::Relaxed);
            return Err(EngineError::NotReady(health.state));
        }
        Ok(handles)
    }

    /// The 9.2 checks in order, over the forced tokens too; on success the request counts as in
    /// flight, which the caller undoes if it cannot send the request.
    fn reserve(
        &self,
        req: &Request,
        forced: Option<&[u32]>,
        state: HealthState,
    ) -> Result<(), EngineError> {
        let sh = &self.shared;
        let switching = sh.switching.load(Ordering::Acquire)
            && matches!(state, HealthState::Draining | HealthState::Loading)
            && !sh.shutdown.load(Ordering::Acquire);
        if state != HealthState::Ready && !switching {
            return Err(EngineError::NotReady(state));
        }
        if req.prompt.is_empty() {
            return Err(EngineError::InvalidRequest(invalid::EMPTY_PROMPT));
        }
        if req.max_tokens == 0 && req.activations.is_empty() {
            return Err(EngineError::InvalidRequest(invalid::NO_TOKENS));
        }
        if !req.activations.is_empty()
            && (req.max_tokens != 0 || !req.score_ids.is_empty() || req.session != 0)
        {
            return Err(EngineError::InvalidRequest(invalid::ACTIVATIONS));
        }
        if req.score_ids.len() > MAX_SCORE_IDS {
            return Err(EngineError::InvalidRequest(invalid::SCORE_IDS));
        }
        let all = req.prompt.iter().chain(forced.unwrap_or_default());
        let ids = all.clone().chain(&req.score_ids);
        let info = sh.info.read().expect("model info");
        // The count bounds the rows Done carries; each layer must exist.
        let depth = info.deployment.n_layers;
        if req.activations.len() > usize::from(depth) || req.activations.iter().any(|&l| l >= depth)
        {
            return Err(EngineError::InvalidRequest(invalid::ACTIVATION_LAYER));
        }
        let rows = &req.activation_rows;
        if !rows.is_empty() && req.activations.len() != 1 {
            return Err(EngineError::InvalidRequest(invalid::ACTIVATION_LAYER));
        }
        if rows.windows(2).any(|w| w[0] >= w[1])
            || rows.last().is_some_and(|&r| r as usize >= req.prompt.len())
        {
            return Err(EngineError::InvalidRequest(invalid::NO_TOKENS));
        }
        let images = check_images(req, info.vision.as_ref(), forced.is_some())?;
        if ids
            .into_iter()
            .any(|&t| t >= info.deployment.vocab && (images == 0 || t & IMAGE_TOKEN == 0))
        {
            return Err(EngineError::InvalidRequest(invalid::TOKEN_OUT_OF_VOCAB));
        }
        if req.logprobs > 20 {
            return Err(EngineError::InvalidRequest(invalid::LOGPROBS));
        }
        if !req.sampling.valid() {
            return Err(EngineError::InvalidRequest(invalid::SAMPLING));
        }
        if !req.score_ids.is_empty() && req.sampling.repeat_penalty != 1.0 {
            return Err(EngineError::InvalidRequest(invalid::SCORE_PENALTY));
        }
        // A banned id is dropped from sample_prep's candidates; at least one must stay drawable.
        let candidates = sample::CANDIDATES.min(info.deployment.vocab as usize);
        if req.ignore_eos && req.stop_tokens.len() >= candidates {
            return Err(EngineError::InvalidRequest(invalid::IGNORED_STOPS));
        }
        if let Some(g) = &req.grammar {
            let code = if !req.score_ids.is_empty() {
                invalid::GRAMMAR_SCORE_IDS
            } else if req.ignore_eos {
                invalid::GRAMMAR_IGNORE_EOS
            } else if g.trigger.as_deref() == Some("") {
                invalid::GRAMMAR_PATTERN
            } else if req.stop_tokens.is_empty() {
                invalid::GRAMMAR_STOPS
            } else if g.compiled.vocab_len() > info.deployment.vocab as usize
                || req.stop_tokens.iter().any(|&t| t >= info.deployment.vocab)
            {
                // Its allowed ids and the stops are gathered on the device.
                invalid::TOKEN_OUT_OF_VOCAB
            } else {
                0
            };
            if code != 0 {
                return Err(EngineError::InvalidRequest(code));
            }
        }
        let tokens = match forced {
            Some(_) => all.count() as u64,
            None => req.prompt.len() as u64 + u64::from(req.max_tokens),
        };
        if tokens > u64::from(info.deployment.ctx_max) {
            return Err(EngineError::PromptTooLong {
                tokens: tokens.min(u64::from(u32::MAX)) as u32,
                limit: info.deployment.ctx_max,
            });
        }
        let max_waiting =
            sh.effective.read().expect("config").values[knob_id::MAX_WAITING as usize] as usize;
        let waiting =
            sh.in_flight.fetch_add(1, Ordering::AcqRel) + sh.sched_waiting.load(Ordering::Acquire);
        if waiting >= max_waiting {
            sh.in_flight.fetch_sub(1, Ordering::Relaxed);
            return Err(EngineError::Overloaded {
                waiting: waiting as u32,
                limit: max_waiting as u32,
            });
        }
        Ok(())
    }

    /// Acknowledges the caller's string stop on `r` at a step boundary, parking only fed KV (10.1.1),
    /// and takes `r`'s terminal event from its channel, skipping the tokens sampled before the
    /// acknowledgement. Ok carries the measurements of a request that ends as StopSeq, or that had
    /// already finished, with its adaptation version. A cancelled request, or one stopped before any token, is `Cancelled`.
    /// A handle from another engine, or a call after the terminal event was read, is rejected.
    pub fn finish_stop(&self, r: &RequestHandle) -> Result<(Usage, Timings, u64), EngineError> {
        if !std::ptr::eq(r.engine.as_ptr(), Arc::as_ptr(&self.shared)) {
            return Err(EngineError::InvalidRequest(invalid::OTHER_ENGINE));
        }
        let (tx, rx) = sync_channel(1);
        // A stopping engine drops the reply instead.
        let acked = self.tx.send(Msg::FinishStop(r.id, tx)).is_ok()
            && self.shared.receive(&rx, None).is_ok();
        // The terminal is queued before the reply (a stopping engine ends every request before it
        // drops the acknowledgements), so the channel holds exactly what the engine decided.
        for event in r.events.try_iter() {
            match event {
                EngineEvent::Token { .. } | EngineEvent::Prefilled { .. } => {}
                EngineEvent::Done {
                    usage,
                    timings,
                    adapter_version,
                    ..
                } => return Ok((usage, timings, adapter_version)),
                EngineEvent::Error(e) => return Err(e),
            }
        }
        Err(if acked {
            EngineError::InvalidRequest(invalid::TERMINAL_TAKEN)
        } else {
            EngineError::NotReady(HealthState::Failed) // the engine thread is gone
        })
    }

    pub fn ring(&self) -> RingHandle {
        self.shared.ring.clone()
    }

    pub fn knobs(&self) -> Arc<dyn KnobControl> {
        Arc::new(knobs::EngineKnobs {
            shared: self.shared.clone(),
            tx: self.tx.clone(),
        })
    }

    pub fn health(&self) -> Health {
        *self.shared.health.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn deployment(&self) -> DeploymentSnapshot {
        self.shared.info.read().expect("model info").deployment
    }

    /// Device memory free, in bytes, when the loaded model's load began, before it took any: what
    /// its fit planned from (the `larch_fit_ledger` line's `free_before`). A bench compares it
    /// across reloads, which each start on a new stream and so should find the first load's.
    /// Unstable: may change before 1.0.
    pub fn free_before(&self) -> u64 {
        self.shared.info.read().expect("model info").free_before
    }

    /// The routed expert layers the loaded model keeps in host memory, as its load placed them: 0
    /// for a model on the device whole, a dense model or one not loaded from a file.
    pub fn host_expert_layers(&self) -> u32 {
        self.shared.info.read().expect("model info").host_layers
    }

    /// Sequences of recurrent state the loaded model holds, 0 for a model without recurrent
    /// layers: such a model reuses KV across requests only through a kept session, never the
    /// prefix cache.
    pub fn state_slots(&self) -> u32 {
        self.shared.info.read().expect("model info").state_slots
    }

    /// The loaded model's class, updated by reload.
    pub fn model_class(&self) -> ModelClass {
        self.shared.info.read().expect("model info").class
    }

    /// This engine's identity, as every ring record's header carries it (gait's recovery, 12.6).
    pub fn engine_id(&self) -> u64 {
        self.shared.engine_id
    }

    /// The `knob_overrides` load rejected (also in the ring), so a strict caller can fail by name.
    /// A clamp, such as `max_context` to the model's training context, is not a rejection:
    /// `effective()` reports the value in force.
    pub fn rejected_overrides(&self) -> &[(u16, f64, RejectReason)] {
        &self.shared.rejected_overrides
    }

    /// The vision projector's geometry, when the engine loaded one ([`EngineOptions::mmproj`]): what a
    /// caller needs to lay out an image's rows.
    pub fn vision(&self) -> Option<ProjectorConfig> {
        self.shared.info.read().expect("model info").vision
    }

    /// Encodes `image`'s units (`pixels`, which the caller cut from it with
    /// [`ProjectorConfig::cut`]) into the unit cache while the engine has nothing else to run, one
    /// unit a turn, so a request that later shows the image reads its rows without encoding them (S5,
    /// encode on paste). The newest image goes first; the cache keeps the most recent 16 InternVL
    /// tiles or 4 Qwen images. `NO_VISION` without a projector, `IMAGE` for pixels not cut from the
    /// image for this projector.
    pub fn prepare_image(
        &self,
        image: &Image,
        pixels: Arc<ImagePixels>,
    ) -> Result<(), EngineError> {
        let Some(v) = self.vision() else {
            return Err(EngineError::InvalidRequest(invalid::NO_VISION));
        };
        if !pixels.fits(&v, v.tokens_for(image.width(), image.height())) {
            return Err(EngineError::InvalidRequest(invalid::IMAGE));
        }
        self.tx
            .send(Msg::Prepare(*image.digest(), pixels))
            .map_err(|_| EngineError::NotReady(self.health().state))
    }

    /// Image units (InternVL tiles, Qwen images) the engine has encoded into its cache since it
    /// started, for a request or ahead of one ([`Self::prepare_image`]); one found in the cache is
    /// not counted.
    pub fn tiles_encoded(&self) -> u64 {
        self.shared.tiles_encoded.load(Ordering::Relaxed)
    }

    /// Steps with no prompt rows (plain decode, and speculative check steps, whose shapes capture
    /// alike) that a load with decode graphs in force ran eager because their shape's capture was
    /// refused (quota short, measured over it, or the capture failed), since the engine started.
    pub fn eager_decode_steps(&self) -> u64 {
        self.shared.eager_decode.load(Ordering::Relaxed)
    }

    /// Steps that ran some layers of an image's encode beside other requests (a Qwen image, which
    /// encodes in slices while other chats run; [`Self::tiles_encoded`] counts the whole unit once).
    pub fn encode_slices(&self) -> u64 {
        self.shared.encode_slices.load(Ordering::Relaxed)
    }

    /// What this engine does on its device ([`Capabilities`]).
    pub fn capabilities(&self) -> Capabilities {
        self.shared.info.read().expect("model info").caps
    }

    /// Drains accepted requests, reloads the GGUF and resumes with updated fit options.
    pub fn reload(&self, change: ReloadChange) -> Result<(), EngineError> {
        self.enqueue_reload(change, None)
    }

    /// [`Self::reload`] with a replacement model loader, as in [`start_with`].
    /// Resumes with eager decode. For a capture-safe loader, enable `decode_graphs` after reload.
    pub fn reload_with(&self, change: ReloadChange, load: Loader) -> Result<(), EngineError> {
        self.enqueue_reload(change, Some(load))
    }

    fn enqueue_reload(
        &self,
        change: ReloadChange,
        load: Option<Loader>,
    ) -> Result<(), EngineError> {
        let (tx, rx) = sync_channel(1);
        {
            let mut health = self.shared.health.lock().unwrap_or_else(|e| e.into_inner());
            if health.state != HealthState::Ready {
                return Err(EngineError::NotReady(health.state));
            }
            let context = larch_records::knob_spec(knob_id::MAX_CONTEXT).expect("registered");
            let fraction = larch_records::knob_spec(knob_id::KV_POOL_FRACTION).expect("registered");
            if change
                .max_context
                .is_some_and(|v| !(context.min..=context.max).contains(&f64::from(v)))
                || change
                    .kv_pool_fraction
                    .is_some_and(|v| !(fraction.min as f32..=fraction.max as f32).contains(&v))
            {
                return Err(EngineError::InvalidRequest(invalid::RELOAD_VALUE));
            }
            // The send and state transition share submit's lock: every accepted request precedes
            // this message, and a second reload cannot race it.
            self.tx
                .send(Msg::Reload(change, load, tx))
                .map_err(|_| EngineError::NotReady(HealthState::Failed))?;
            health.state = HealthState::Draining;
            health.since_mono_ns = self.shared.now();
        }
        self.shared.receive(&rx, None).unwrap_or_else(|_| {
            let state = self.health().state;
            Err(self
                .shared
                .host
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .error
                .unwrap_or(EngineError::NotReady(state)))
        })
    }

    /// Serves router deltas `d` as adaptation `version` (0 with `None`: the base model). They take
    /// effect at the first step boundary where no started request remains: requests that started
    /// before keep the old version to their end, and every request admitted after this returns
    /// runs on the new one, so no answer mixes versions. Returns once they are applied. The
    /// model's refusal is `InvalidRequest(ROUTER_DELTAS)` and keeps the version in force.
    pub fn set_router_deltas(
        &self,
        d: Option<RouterDeltas>,
        version: u64,
    ) -> Result<(), EngineError> {
        if d.is_some() != (version != 0) {
            return Err(EngineError::InvalidRequest(invalid::ROUTER_DELTAS));
        }
        let (tx, rx) = sync_channel(1);
        self.tx
            .send(Msg::RouterDeltas(d, version, tx))
            .map_err(|_| EngineError::NotReady(HealthState::Failed))?;
        self.shared
            .receive(&rx, None)
            .unwrap_or_else(|_| Err(EngineError::NotReady(self.health().state)))
    }

    /// Runs `f` on the resident [`Transformer`] (`None` for another model) and the engine's
    /// stream at the first step boundary with no started request, as [`Self::set_router_deltas`]
    /// waits: no request runs meanwhile, and requests admitted after this call wait for it.
    pub fn with_model<T: Send + 'static>(
        &self,
        f: impl FnOnce(Option<&Transformer>, &Stream) -> T + Send + 'static,
    ) -> Result<T, EngineError> {
        let (tx, rx) = sync_channel(1);
        self.tx
            .send(Msg::Job(Box::new(move |m, s| {
                let _ = tx.send(f(m, s));
            })))
            .map_err(|_| EngineError::NotReady(HealthState::Failed))?;
        self.shared
            .receive(&rx, None)
            .map_err(|_| EngineError::NotReady(self.health().state))
    }

    /// The adaptation version in force; 0 is the base model.
    pub fn adapter_version(&self) -> u64 {
        self.shared.adapter_version.load(Ordering::Acquire)
    }

    /// Mean next-token cross-entropy (nats) of `tokens` under the version in force, over the
    /// positions `i` where `mask[i]`, each predicted from `tokens[..i]`: one observed prefill,
    /// nothing parked. Admission as [`Self::force`]; a bad mask is `InvalidRequest(LOSS_MASK)`.
    pub fn token_loss(&self, tokens: Vec<u32>, mask: Vec<bool>) -> Result<f64, EngineError> {
        let scored = self.token_positions(tokens, mask)?;
        let mut sum = 0f64;
        for p in &scored {
            sum -= f64::from(p.next.ok_or(EngineError::Internal(0))?);
        }
        Ok(sum / scored.len() as f64)
    }

    /// The positions [`Self::token_loss`] scores, ascending, each with its top 64 and the exact
    /// logprob of the token `mask` marks after it; the same run and errors.
    pub fn token_positions(
        &self,
        tokens: Vec<u32>,
        mask: Vec<bool>,
    ) -> Result<Vec<Position>, EngineError> {
        if mask.len() != tokens.len() || mask.first() == Some(&true) || !mask.contains(&true) {
            return Err(EngineError::InvalidRequest(invalid::LOSS_MASK));
        }
        let (tx, positions) = std::sync::mpsc::channel();
        let req = Request::new(tokens, 1);
        let h = self
            .admit(vec![(req, Some((Vec::new(), tx, true)))])?
            .pop()
            .expect("one request");
        for event in h.events.iter() {
            match event {
                EngineEvent::Token { .. } | EngineEvent::Prefilled { .. } => {}
                EngineEvent::Done { .. } => {
                    return Ok(positions
                        .try_iter()
                        .filter(|p| mask.get(p.pos as usize + 1) == Some(&true))
                        .collect());
                }
                EngineEvent::Error(e) => return Err(e),
            }
        }
        Err(EngineError::NotReady(HealthState::Failed)) // the engine thread is gone
    }

    /// Pins the chain of `tokens` in the prefix cache of `namespace`: it is exempt from eviction,
    /// passes the half-pool cap while the pool keeps an [`TURN_ROOM`] free, and survives a
    /// same-model reload, until the handle drops or a request that is not its read needs its
    /// blocks and the pool has no other ([`Self::pin_live`]). It holds only what a request of those tokens
    /// publishes: submit one (`Class::Background`) after this. A prefix cache that is off
    /// (`prefix_cache_blocks` 0) refuses with `NO_PREFIX_CACHE`.
    pub fn pin(&self, tokens: Vec<u32>, namespace: u64) -> Result<PinHandle, EngineError> {
        if self
            .prefix_cache_stats()
            .is_none_or(|s| s.capacity_blocks == 0)
        {
            return Err(EngineError::InvalidRequest(invalid::NO_PREFIX_CACHE));
        }
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = sync_channel(1);
        self.tx
            .send(Msg::Pin(id, namespace, tokens, tx))
            .map_err(|_| EngineError::NotReady(HealthState::Failed))?;
        self.shared
            .receive(&rx, None)
            .unwrap_or(Err(EngineError::NotReady(self.health().state)))?;
        Ok(PinHandle {
            id,
            tx: self.tx.clone(),
        })
    }

    /// Whether `pin` still holds its chain: false once it went cold to give way to a request that
    /// needed its blocks (a pin is a cache and never fails a request), and a question on the
    /// document then reads it again. Unstable: may change before 1.0.
    pub fn pin_live(&self, pin: &PinHandle) -> bool {
        let cold = self
            .shared
            .cold_pins
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        !cold.contains(&pin.id)
    }

    fn pin_op(&self, op: PinOp) -> Result<PinReply, EngineError> {
        let (tx, rx) = sync_channel(1);
        self.tx
            .send(Msg::PinOp(op, tx))
            .map_err(|_| EngineError::NotReady(HealthState::Failed))?;
        self.shared
            .receive(&rx, None)
            .unwrap_or(Err(EngineError::NotReady(self.health().state)))
    }

    /// Appends `pin`'s segments from token `from` (0, or what an earlier call returned) up to its
    /// last exportable token to `w`, with the header first when `from` is 0; returns where it
    /// ended. A hybrid's last exportable token is its last state checkpoint, and while the pin's
    /// read is under way a mark ([`EngineOptions::pin_mark`]'s multiple). Only the new segments' KV
    /// leave the device. Nothing at all to export is `Miss`. Unstable: may change before 1.0.
    pub fn export_pin(
        &self,
        pin: &PinHandle,
        w: &mut dyn std::io::Write,
        from: u32,
    ) -> Result<u32, PinError> {
        pinfile::export(&mut |op| self.pin_op(op), pin.id, from, w)
    }

    /// Writes `pin`'s state sidecar and returns the tokens it covers; see [`pinfile`]. Unstable.
    pub fn export_pin_state(
        &self,
        pin: &PinHandle,
        w: &mut dyn std::io::Write,
    ) -> Result<u32, PinError> {
        pinfile::export_state(&mut |op| self.pin_op(op), pin.id, w)
    }

    /// Loads a pin file and its sidecar and pins `tokens` up to the tokens the sidecar covers
    /// (returned beside the hold); see [`pinfile`]. `prepare` runs after the files pass their
    /// header checks and before any block is taken. Unstable: may change before 1.0.
    pub fn import_pin(
        &self,
        tokens: &[u32],
        namespace: u64,
        pin: &mut dyn std::io::Read,
        state: &mut dyn std::io::Read,
        prepare: &mut dyn FnMut() -> Result<(), EngineError>,
    ) -> Result<(PinHandle, usize), PinError> {
        let (id, n) = pinfile::import(
            &mut |op| self.pin_op(op),
            None,
            tokens,
            namespace,
            pin,
            state,
            prepare,
        )?;
        let hold = PinHandle {
            id,
            tx: self.tx.clone(),
        };
        Ok((hold, n))
    }

    /// What the documents plan holds for a document of `tokens`, from the fit a load runs and
    /// without loading ([`DocFit`]).
    pub fn document_fit(&self, tokens: u32) -> Result<DocFit, EngineError> {
        let (tx, rx) = sync_channel(1);
        self.tx
            .send(Msg::DocFit(tokens, tx))
            .map_err(|_| EngineError::NotReady(HealthState::Failed))?;
        self.shared
            .receive(&rx, None)
            .unwrap_or(Err(EngineError::NotReady(self.health().state)))
    }

    /// Frees the session's parked sequence; later parks under the id free instead (D14, 10.1.1).
    pub fn release_session(&self, session: u64) {
        let _ = self.tx.send(Msg::Release(session));
    }

    /// Speculative decoding's counters, as of the last step that checked guesses.
    pub fn guess_stats(&self) -> GuessStats {
        *self
            .shared
            .guesses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Snapshot after preceding steps/messages; failed engines return None.
    pub fn prefix_cache_stats(&self) -> Option<PrefixCacheStats> {
        let (tx, rx) = sync_channel(1);
        self.tx.send(Msg::PrefixStats(tx)).ok()?;
        self.shared.receive(&rx, None).ok()
    }

    /// `n_kv` of the session's parked sequence, 0 if none (D14).
    pub fn parked_tokens(&self, session: u64) -> u32 {
        let (tx, rx) = sync_channel(1);
        if self.tx.send(Msg::Parked(session, tx)).is_err() {
            return 0;
        }
        self.shared.receive(&rx, None).unwrap_or(0)
    }

    /// Stops the engine after the current step and waits for its thread: health goes to Draining,
    /// every request ends with `NotReady(Draining)`, and the ring holds every record on return.
    /// After a fault this returns without joining the poisoned executor; its resources are retained.
    /// With `OnFault::Exit` it never returns after a fault: the process exits with 3.
    pub fn shutdown(self) {
        {
            let mut health = self.shared.health.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.shutdown.store(true, Ordering::Release);
            if health.state != HealthState::Failed {
                let from = health.state;
                health.state = HealthState::Draining;
                health.since_mono_ns = self.shared.now();
                if from != HealthState::Draining {
                    let mut host = self.shared.host.lock().unwrap_or_else(|e| e.into_inner());
                    let cfg = **self
                        .shared
                        .effective
                        .read()
                        .unwrap_or_else(|e| e.into_inner());
                    host.draining(&cfg, from, health.since_mono_ns);
                }
            }
            let _ = self.tx.send(Msg::Shutdown);
        }
        let thread = self.shared.thread.lock().expect("thread").take();
        if let Some(t) = thread {
            while !t.is_finished() {
                if self.health().state == HealthState::Failed {
                    if self.shared.on_fault == OnFault::Exit {
                        // The fault path exits with 3 within 100 ms (9.11); returning would let
                        // the caller's own exit code win.
                        loop {
                            std::thread::park();
                        }
                    }
                    return;
                }
                std::thread::sleep(lifecycle::POLL);
            }
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use larch_kernels::DType;
    use larch_model::gguf::{Gguf, TensorInfo};

    use super::{Histogram, Weights};

    /// An MoE header's experts per layer, of which a token reads `used` of `experts`; a dense one
    /// has none.
    #[test]
    fn weights_of_an_moe_header() {
        use larch_model::{Arch, tiny};
        let parse = |arch| Gguf::parse(&tiny::gguf(arch, 1)).unwrap();
        let (g, d) = (parse(Arch::Qwen3Moe), parse(Arch::Qwen3));
        let (moe, dense) = (Weights::of(&g), Weights::of(&d));
        let embd = |g: &Gguf| g.tensor("token_embd.weight").unwrap().bytes;
        assert!(dense.experts.is_empty() && dense.read_per_step == dense.bytes - embd(&d));
        assert_eq!(moe.experts.len(), tiny::LAYERS as usize);
        let all: u64 = moe.experts.iter().sum();
        assert!(moe.experts.iter().all(|&e| e > 0));
        let unread = all - all * u64::from(tiny::EXPERTS_USED) / u64::from(tiny::EXPERTS);
        let read = moe.bytes - embd(&g) - unread;
        assert!(
            moe.read_per_step.abs_diff(read) <= 1,
            "{}",
            moe.read_per_step
        );
    }

    /// An expert name past the config's layers (or near usize::MAX) is an unused tensor: no
    /// allocation follows its index, and it adds to no layer; a layer past the config's adds no bytes.
    #[test]
    fn weights_of_crafted_expert_names_allocate_nothing() {
        use larch_model::{Arch, tiny};
        let mut g = Gguf::parse(&tiny::gguf(Arch::Qwen3Moe, 1)).unwrap();
        let honest = Weights::of(&g).experts;
        for name in [
            "blk.1000000000.ffn_gate_exps.weight",
            "blk.18446744073709551615.ffn_up_exps.weight",
            "blk.0.ffn_gate_exps.weight.extra",
        ] {
            g.tensors.push(TensorInfo {
                name: name.into(),
                ggml_type: 999,
                dims: vec![1],
                offset: 0,
                bytes: 0,
            });
        }
        let w = Weights::of(&g);
        assert_eq!(w.experts, honest);
        assert_eq!(w.experts.len(), tiny::LAYERS as usize);
        // A built-in drafter's layer past the config's (the hybrid MoE target's blk.40) is not loaded.
        let trunk = Weights::of(&Gguf::parse(&tiny::gguf(Arch::Qwen35Moe, 1)).unwrap());
        let mtp = Weights::of(&Gguf::parse(&tiny::gguf_mtp(Arch::Qwen35Moe, 1)).unwrap());
        assert_eq!((mtp.bytes, mtp.experts), (trunk.bytes, trunk.experts));
    }

    /// A mixed layer (`larch repack --cold-fraction`) counts at its real bytes, its hot and cold
    /// tensors together, which the fit's host_expert_layers plans from: less than the whole layer's.
    #[test]
    fn a_mixed_layer_counts_its_hot_and_cold_experts() {
        use larch_model::{Arch, Histogram, tiny};
        let dir = std::env::temp_dir().join(format!("larch-fit-cold-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (src, dst) = (dir.join("src.gguf"), dir.join("dst.gguf"));
        let a = tiny::gguf(Arch::Qwen3Moe, 1);
        std::fs::write(&src, &a).unwrap();
        let hist = Histogram::uniform(a.len() as u64, tiny::LAYERS, tiny::EXPERTS);
        larch_model::q2k::repack_cold(&src, &dst, &[1], 0.5, &hist).unwrap();
        let g = Gguf::parse(&std::fs::read(&dst).unwrap()).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let (whole, mixed) = (Weights::of(&Gguf::parse(&a).unwrap()), Weights::of(&g));
        let real: u64 = (g.tensors.iter())
            .filter(|t| t.name.starts_with("blk.1.ffn_") && t.name.ends_with("exps.weight"))
            .map(|t| t.bytes)
            .sum();
        assert_eq!(mixed.experts[0], whole.experts[0]);
        assert_eq!(mixed.experts[1], real);
        assert!(real < whole.experts[1]);
    }

    /// A reload onto another model keeps "the model's own block" whichever way the drafter was
    /// named, and leaves a DFlash drafter's path alone.
    #[test]
    fn a_reload_keeps_the_choice_of_the_models_own_block() {
        use super::{EngineOptions, PathBuf};
        let mut o = EngineOptions::new(PathBuf::from("/m/a.gguf"), 0);
        o.drafter = Some(PathBuf::from("/m/a.gguf"));
        assert!(o.clone().onto(PathBuf::from("/m/b.gguf")).mtp());
        o.drafter = Some(PathBuf::from("/m/dflash.gguf"));
        let o = o.onto(PathBuf::from("/m/b.gguf"));
        assert_eq!(o.drafter, Some(PathBuf::from("/m/dflash.gguf")));
        assert!(!o.mtp());
    }

    /// A mistyped drafter path beside a model path that does not resolve either (a custom loader's
    /// placeholder) is a DFlash drafter's path, whose read reports the I/O error.
    #[test]
    fn two_unresolved_paths_are_not_the_same_model() {
        use super::{EngineOptions, PathBuf};
        let mut o = EngineOptions::new(PathBuf::from("/nonexistent/placeholder"), 0);
        o.drafter = Some(PathBuf::from("/nonexistent/typo.gguf"));
        assert!(!o.mtp());
    }

    /// The loader plans host experts for the context the fit will use: the options after the
    /// startup overrides, not the ones `start` was given.
    #[test]
    fn the_loader_plans_for_the_overridden_context_and_fraction() {
        use super::{Device, LoadError, PathBuf, Stream, knob_id};
        use larch_model::{Arch, tiny};
        use std::sync::{Arc, Mutex};
        let w = Weights::of(&Gguf::parse(&tiny::gguf(Arch::Qwen3Moe, 1)).unwrap());
        let budget = w.budget.clone();
        *budget.lock().unwrap() = (8192, 0.5);
        let mut opts = super::EngineOptions::new(PathBuf::from("unused"), -1);
        opts.knob_overrides = vec![
            (knob_id::MAX_CONTEXT, 32768.0),
            (knob_id::KV_POOL_FRACTION, 0.75),
        ];
        let seen = Arc::new(Mutex::new(None));
        let at_load = seen.clone();
        let load: super::Loader = Box::new(move |_: &Device, _: &Stream, _| {
            *at_load.lock().unwrap() = Some(*budget.lock().unwrap());
            Err(LoadError::Io)
        });
        assert!(super::spawn(opts, load, Some(w)).is_err());
        assert_eq!(*seen.lock().unwrap(), Some((32768, 0.75)));
    }

    /// The per-expert residency the loader plans from `start` (`None`: the cache off or a
    /// dishonest file); `None` is the whole-layer path.
    fn plan_of(start: Option<&Histogram>) -> Option<Vec<Vec<u32>>> {
        use super::{Device, fit};
        use larch_kernels::{Backend, CPU};
        use larch_model::Model;
        let mut before = Device::open(CPU).unwrap().props().unwrap();
        before.backend = Backend::Cuda;
        (before.vram_total, before.vram_free) = (17_066_033_152, 15_631_122_432);
        let mut c = larch_model::FakeModel::new(248_320).config().clone();
        (c.n_layers, c.ctx_train) = (40, 262_144);
        let (weights, reserve) = (16_209_005_056, 578_451_072);
        let experts = vec![381_681_664; 40];
        super::plan_from(start).and_then(|h| {
            fit::expert_residency(
                &before,
                weights,
                &experts,
                &h.counts,
                128,
                &c,
                1795,
                0.85,
                reserve,
                0,
                1,
                DType::F16,
            )
        })
    }

    /// No histogram plans as the cache off does, compared as the plan itself; recorded counts do not.
    #[test]
    fn expert_cache_is_off_unless_asked() {
        assert_eq!(super::expert_cache(""), None);
        assert_eq!(super::expert_cache("0"), None);
        assert_eq!(super::expert_cache("1"), Some(false));
        assert_eq!(super::expert_cache("2"), Some(true));
    }

    #[test]
    fn no_histogram_plans_what_the_cache_off_plans() {
        let zeros = Histogram::uniform(1, 40, 128);
        assert!(plan_of(Some(&zeros)) == plan_of(None), "no histogram");
        let mut hot = zeros.clone();
        hot.counts[0] = 1;
        assert!(plan_of(Some(&hot)).is_some(), "control: a real plan");
        assert!(plan_of(Some(&hot)) != plan_of(None), "control");
    }

    /// A missing, corrupt or all-zero histogram (Drop writes one after a load that never stepped)
    /// is no histogram: the load keeps whole layers.
    #[test]
    fn no_usable_histogram_plans_whole_layers() {
        use larch_model::{Arch, ModelConfig, MoeCfg, tiny};
        let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
        let g = Gguf::parse(&bytes).unwrap();
        let c = ModelConfig::from_gguf(&g).unwrap();
        let m = MoeCfg::from_gguf(&g, c.arch).unwrap().unwrap();
        let len = bytes.len() as u64;
        let dir = std::env::temp_dir().join(format!("larch-usable-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = Histogram::path(&dir.join("tiny.gguf"));
        let usable = || {
            let h = Histogram::for_load(&g, &c, m, &path, len).expect("an honest file");
            super::plan_from(Some(&h)).is_some()
        };
        assert!(!usable(), "missing");
        std::fs::write(&path, b"not a histogram").unwrap();
        assert!(!usable(), "corrupt");
        let mut h = Histogram::uniform(len, c.n_layers, m.experts);
        h.write(&path).unwrap();
        assert!(!usable(), "all zero");
        h.counts[1] = 1;
        h.write(&path).unwrap();
        assert!(usable(), "recorded");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Only a plan that moved layers leaving the graphs' reserve paid for them: not a forced
    /// count, not a load with graphs off, no layer moved, or every layer (the graphs gave way).
    #[test]
    fn only_a_plan_that_moved_layers_with_the_reserve_pays_for_graphs() {
        assert!(super::graphs_paid(true, 16, 5, 40));
        assert!(!super::graphs_paid(false, 16, 5, 40));
        assert!(!super::graphs_paid(true, 0, 5, 40));
        assert!(!super::graphs_paid(true, 16, 0, 40));
        assert!(!super::graphs_paid(true, 16, 40, 40));
    }

    /// The GPU ledger's 5-layer load of the hybrid MoE target on the 5070 Ti, 503,660,544 B free against a fit
    /// need of 413,444,622 B: the check passes it at the high reading and refuses the low one,
    /// 67,391,488 B lower, which then moves a layer.
    #[test]
    fn the_swing_passes_the_high_reading_and_refuses_the_low() {
        use larch_kernels::Backend::Cuda;
        let (before, free, drop) = (15_631_122_432, 503_660_544, 67_391_488);
        let need = super::post_load_need(Cuda, 413_444_622);
        assert!(super::fits_after_load((), true, before, free, need).is_ok());
        assert!(super::fits_after_load((), true, before, free - drop, need).is_err());
        assert!(super::FREE_SWING >= drop);
    }

    /// The non-hybrid MoE on the 5070 Ti at 2,304 tokens: 11 host layers left 998,588,416 free
    /// against a need of 968,817,883. The swing was measured on the hybrid MoE only, so here the
    /// check passes the 11 (12 layers, 4.6 % slower W4@1, once needed 1,040,121,051); a hybrid, or a
    /// fallback after a per-expert attempt, still wants the swing spare.
    #[test]
    fn the_swing_is_asked_only_where_it_was_measured() {
        use larch_kernels::Backend::Cuda;
        let (before, free, need) = (15_631_122_432, 998_588_416, 968_817_883);
        let fits = |hybrid, tried| {
            let need = super::load_check_need(Cuda, hybrid, tried, need);
            super::fits_after_load((), true, before, free, need).is_ok()
        };
        assert!(fits(false, false));
        assert!(!fits(true, false));
        assert!(!fits(false, true));
    }

    /// A planned count moves at least one more layer per refusal until a load fits; a forced one
    /// never moves.
    #[test]
    fn a_refused_planned_load_moves_more_host_layers() {
        let fits = |n: u32| {
            if n >= 12 {
                Ok(n)
            } else {
                Err((super::LoadError::DoesNotFit { need: 2, have: 1 }, 0))
            }
        };
        let experts = [1; 48];
        assert_eq!(
            super::more_host_layers_on_refusal(10, &experts, true, fits),
            Ok(12)
        );
        assert!(super::more_host_layers_on_refusal(10, &experts, false, fits).is_err());
        assert!(super::more_host_layers_on_refusal(10, &experts[..11], true, fits).is_err());
        let io = |_| Err::<u32, _>((super::LoadError::Io, 0));
        assert_eq!(
            super::more_host_layers_on_refusal(0, &experts, true, io),
            Err(super::LoadError::Io)
        );
        // A refusal of the load itself names one allocation and the device's figure (0 when its
        // props fail), not a shortfall: it moves one layer, however large the allocation.
        let refused = |n: u32| match n {
            0 => Err((
                super::LoadError::DoesNotFit {
                    need: 1 << 40,
                    have: 0,
                },
                0,
            )),
            n => Ok(n),
        };
        assert_eq!(
            super::more_host_layers_on_refusal(0, &experts, true, refused),
            Ok(1)
        );
    }

    /// A per-expert load short of serving memory plans once more with the shortfall reserved; a
    /// second refusal, one with no shortfall and any other error go to the caller.
    #[test]
    fn a_short_per_expert_load_replans_once_with_the_shortfall() {
        let no = || super::LoadError::DoesNotFit { need: 2, have: 1 };
        let mut seen = Vec::new();
        let got = super::replan_on_shortfall(100, 0, |r| {
            seen.push(r);
            if r >= 150 {
                Ok(r)
            } else {
                Err((no(), 150 - r))
            }
        });
        assert_eq!((got, seen), (Ok(150), vec![100, 150]));
        for (refusal, tries) in [
            ((no(), 50), 2),
            ((no(), 0), 1),
            ((super::LoadError::Io, 0), 1),
        ] {
            let mut n = 0;
            let got = super::replan_on_shortfall(100, 0, |_| {
                n += 1;
                Err::<u64, _>(refusal)
            });
            assert_eq!((got, n), (Err(refusal.0), tries));
        }
    }

    /// The re-plan fits by construction, whatever slack either plan leaves: a greedy plan of `e`-byte
    /// experts in a room of `room - reserve` leaves under `e` unused, the load takes `x` the plan
    /// did not count (the MoE target on a 5070 Ti: about 118 MB), so a load fits when the room it left plus
    /// the reserve it charged past the real one covers `x`. Reserving only the shortfall fits when
    /// the second slack is at least the first, about half the time.
    #[test]
    fn a_replanned_per_expert_load_fits_whatever_its_slack() {
        let no = || super::LoadError::DoesNotFit { need: 2, have: 1 };
        let (e, x) = (1_600_000u64, 118_000_000u64);
        for room in (10_000_000_000..10_000_000_000 + 2 * e).step_by(99_991) {
            let load = |reserve: u64| {
                let held = (room - reserve) / e * e;
                match (room - held).checked_sub(x) {
                    Some(_) => Ok(reserve),
                    None => Err((no(), x - (room - held))),
                }
            };
            assert!(super::replan_on_shortfall(0, e, load).is_ok(), "{room}");
        }
    }

    /// A CUDA re-plan survives the measured swing: on the 5070 Ti the re-plan of the MoE target read
    /// 67391488 bytes less than the same re-plan in two other runs (842776576 against 910168064)
    /// and fell back to whole layers, as the fit had read that much less after a passed check. The
    /// check needs the swing spare, and a re-plan whose reading drops by it still passes.
    #[test]
    fn a_cuda_replan_survives_the_measured_swing() {
        use larch_kernels::Backend::Cuda;
        let (e, x, drop) = (1_490_944u64, 118_000_000u64, 67_391_488u64);
        let need = super::post_load_need(Cuda, x);
        for room in (16_000_000_000..16_000_000_000 + 2 * e).step_by(99_991) {
            let mut reads = 0;
            let load = |reserve: u64| {
                let held = (room - reserve) / e * e;
                // The second reading, the re-plan's, is the one that dropped.
                reads += 1;
                let free = (room - held).saturating_sub(if reads == 2 { drop } else { 0 });
                super::fits_after_load(reserve, true, room, free, need)
            };
            let got = super::replan_on_shortfall(0, super::replan_margin(Cuda, e), load);
            assert!(got.is_ok(), "{room}");
        }
    }

    /// A planned load that succeeds but leaves less than the serving need is dropped and retried
    /// once with the layers that cover the shortfall, such as a hybrid's recurrent state, which the
    /// plan does not count; a forced load is kept as it is; a shortfall past the last layer fails.
    /// m13 on the 5070 Ti (26 Nov): a load planned for 4 state slots read 630,738,944 B free against
    /// 637,157,298 needed, and 608,244,358 at 3 slots. It gave a slot up, so 4 chats ran on 3 seats
    /// (112.7 against 174 tok/s at W4@4, graphs eager). Short of the 4th slot, it now keeps MANY_SEATS
    /// and refuses with the shortfall, so the load moves one more host layer for the slot.
    #[test]
    fn a_load_planned_for_four_seats_keeps_them_and_moves_a_layer() {
        let slot = 64_389_120u64;
        let need = |k: u32| match k {
            4 => 637_157_298,
            3 => 608_244_358,
            k => 608_244_358 - u64::from(3 - k) * 28_912_940,
        };
        let free = 630_738_944;
        let low = |slots, floor| super::settle_low(slots, floor, true);
        // Planned 4, floor 1: the check may not drop below the 4 seats the plan gave.
        assert_eq!(
            super::settle_slots(free, 4, low(4, 1), slot, need),
            Err(free)
        );
        // Planned 7 (the faster copy): dropping to 5 still seats 4 chats.
        let roomy = |k: u32| 500_000_000 + u64::from(k) * 30_000_000;
        assert_eq!(
            super::settle_slots(600_000_000, 7, low(7, 1), slot, roomy),
            Ok(5)
        );
        // A plan of fewer slots than MANY_SEATS gives them up to its floor, as before.
        assert_eq!(
            super::settle_slots(free - slot, 3, low(3, 1), slot, need),
            Ok(2)
        );
        // A load that cannot move a layer (not planned, every layer on the host, not CUDA) gives
        // slots up to its floor as before.
        assert_eq!(super::settle_low(4, 1, false), 1);
        assert_eq!(
            super::settle_slots(free, 4, super::settle_low(4, 1, false), slot, need),
            Ok(3)
        );
        // A setup that serves fewer than MANY_SEATS chats at once (max_batch_seqs 1 to 3) plans at
        // most hybrid_slots of it, so every slot count it can have gives slots up to its floor and
        // keeps the layer.
        for seats in 1..crate::engine::MANY_SEATS {
            let cap = crate::engine::hybrid_slots(seats);
            assert!(cap < crate::engine::MANY_SEATS, "max_batch_seqs {seats}");
            for slots in 1..=cap {
                assert_eq!(
                    super::settle_low(slots, 1, true),
                    1,
                    "{seats}: {slots} slots"
                );
            }
        }
        // The floor itself is never dropped below; short there, the load refuses.
        assert_eq!(super::settle_slots(100, 2, low(2, 2), slot, need), Err(100));
    }

    #[test]
    fn a_planned_load_short_of_serving_memory_moves_more_host_layers() {
        use std::cell::Cell;
        struct M<'a>(&'a Cell<u32>, u32);
        impl Drop for M<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let dropped = Cell::new(0);
        let d = &dropped;
        // Each host layer frees 10 bytes; 100 are needed.
        let experts = [10; 48];
        let load = |planned: bool| {
            move |n: u32| super::fits_after_load(M(d, n), planned, 200, 60 + 10 * u64::from(n), 100)
        };
        let m = super::more_host_layers_on_refusal(0, &experts, true, load(true));
        assert_eq!(
            m.as_ref().map(|m| m.1).ok(),
            Some(4),
            "layers 0 to 3 cover 40 short"
        );
        assert_eq!(
            dropped.get(),
            1,
            "one load short of serving memory, then one that fits"
        );
        drop(m);
        assert!(super::more_host_layers_on_refusal(0, &experts, false, load(false)).is_ok());
        assert_eq!(dropped.get(), 3);
        assert_eq!(
            super::more_host_layers_on_refusal(0, &experts[..3], true, load(true)).err(),
            Some(super::LoadError::DoesNotFit {
                need: 210,
                have: 200
            })
        );
        assert_eq!(
            dropped.get(),
            5,
            "two loads, at 0 and at 3 host layers, both short"
        );
    }

    /// The MoE target's per-expert load on a 16 GB 5070 Ti (frozen-1): the check after the load passed at
    /// 809504768 free for a need of 807063231, and the engine's fit, reading free memory again with
    /// nothing allocated in between, found 67391488 bytes less and refused outside the fallback. A
    /// CUDA load now passes only with that swing to spare, so it re-plans or falls back instead;
    /// the CPU reference, whose free figure is fixed, checks the need alone.
    #[test]
    fn a_cuda_load_keeps_the_measured_swing_beside_the_fit() {
        use larch_kernels::Backend;
        let (before, free, need) = (15_631_122_432, 809_504_768, 807_063_231);
        let check =
            |b| super::fits_after_load((), true, before, free, super::post_load_need(b, need));
        assert!(check(Backend::Cuda).is_err());
        assert!(check(Backend::Cpu).is_ok());
        let fits = |free| {
            super::fits_after_load(
                (),
                true,
                before,
                free,
                super::post_load_need(Backend::Cuda, need),
            )
        };
        let passed = (free..)
            .step_by(1 << 20)
            .find(|&f| fits(f).is_ok())
            .unwrap();
        assert!(passed - 67_391_488 >= need, "{passed}");
    }

    /// The hybrid MoE target at 1,795 tokens and 4 seats on the 5070 Ti, as LARCH_FIT_DIAGNOSTICS
    /// logged it: the plan's reserve 578,451,072 gave 5 host layers, whose load left 696,832,000
    /// free against a need of 653,144,244, so fit() admitted no graph slot; a layer's experts are
    /// 381,681,664 (the 7 to 8 layer step of the 2,304-token loads). With decode graphs on at load
    /// the plan and its check leave fit() its whole graph reserve; off, the plan keeps its 5
    /// layers; with every layer on the host the graphs give way to the load.
    #[test]
    fn graphs_on_at_load_leave_the_fit_its_graph_reserve() {
        use super::{DeviceProps, Model, fit};
        use larch_kernels::{Backend, CPU, Device};
        let mut before = Device::open(CPU).unwrap().props().unwrap();
        before.backend = Backend::Cuda;
        before.flags = DeviceProps::CAP_GRAPHS;
        (before.vram_total, before.vram_free) = (17_066_033_152, 15_631_122_432);
        let mut c = larch_model::FakeModel::new(248_320).config().clone();
        (
            c.n_layers,
            c.kv_layers,
            c.n_kv_heads,
            c.head_dim,
            c.ctx_train,
        ) = (40, 10, 2, 256, 262_144);
        let (weights, ctx, e) = (16_209_005_056, 1795, 381_681_664);
        let scratch =
            653_144_244 - fit::serving_need(&before, &c, ctx, 0.85, 0, false, 4, DType::F16);
        // The model's own allocations the plan counts beside the weights and the serving scratch.
        let own = 578_451_072 - scratch;
        let free = |n: u32| 696_832_000 + u64::from(n) * e - 5 * e;
        let load = |experts: &[u64], graphs: bool| {
            let seats = if graphs { 4 } else { 0 };
            let reserve = own + super::kept_beside(&before, scratch, seats, false, false, 0);
            let planned = fit::host_expert_layers(
                &before,
                weights,
                experts,
                &c,
                ctx,
                0.85,
                reserve,
                4,
                DType::F16,
            );
            let n = super::more_host_layers_on_refusal(planned, experts, true, |n| {
                let last = n as usize == experts.len();
                let beside = super::kept_beside(&before, scratch, seats, false, last, 0);
                let need = fit::serving_need(&before, &c, ctx, 0.85, beside, n == 0, 4, DType::F16);
                super::fits_after_load(n, true, before.vram_free, free(n), need)
            })
            .unwrap();
            let plan = fit::fit(
                &before,
                free(n),
                &c,
                ctx,
                0.85,
                scratch,
                4,
                graphs,
                false,
                false,
                0,
                0,
                DType::F16,
            );
            (planned, n, plan.unwrap())
        };
        let experts = vec![e; 40];
        let (planned, n, off) = load(&experts, false);
        assert_eq!((planned, n, off.graph_bytes), (5, 5, 0));
        let (slots, arena) = fit::graph_need(&before, 4, true, false, 0);
        let (_, n, on) = load(&experts, true);
        assert!(slots > 0);
        assert_eq!(
            (on.graph_bytes, on.graph_arena_bytes),
            (slots, arena),
            "{n} host layers"
        );
        let (_, n, all) = load(&experts[..5], true);
        assert_eq!((n, all.graph_bytes), (5, 0));
    }

    /// The plan reserves the MTP block in `scratch` and the load takes it: the check after that
    /// load counts it once, so it keeps the planned host layers, and fit() admits the result.
    #[test]
    fn the_check_after_a_load_counts_the_mtp_block_once() {
        use super::{DeviceProps, Model, fit};
        use larch_kernels::{Backend, CPU, Device};
        let mut before = Device::open(CPU).unwrap().props().unwrap();
        before.backend = Backend::Cuda;
        before.flags = DeviceProps::CAP_GRAPHS;
        (before.vram_total, before.vram_free) = (17_066_033_152, 15_631_122_432);
        let mut c = larch_model::FakeModel::new(248_320).config().clone();
        (
            c.n_layers,
            c.kv_layers,
            c.n_kv_heads,
            c.head_dim,
            c.ctx_train,
        ) = (40, 10, 2, 256, 262_144);
        let (weights, ctx, e, block) = (16_209_005_056u64, 1795, 381_681_664u64, 856u64 << 20);
        let serving =
            653_144_244 - fit::serving_need(&before, &c, ctx, 0.85, 0, false, 1, DType::F16);
        let own = 578_451_072 - serving;
        let scratch = serving + block;
        let experts = vec![e; 40];
        let reserve = own + super::kept_beside(&before, scratch, 0, false, false, 0);
        let planned = fit::host_expert_layers(
            &before,
            weights,
            &experts,
            &c,
            ctx,
            0.85,
            reserve,
            1,
            DType::F16,
        );
        assert!(planned > 0 && planned < 40, "{planned}");
        // What the load leaves: the weights the plan keeps on the device, its own, and the block.
        let free = |n: u32| before.vram_free - (weights - u64::from(n) * e) - own - block;
        let n = super::more_host_layers_on_refusal(planned, &experts, true, |n| {
            let left = super::scratch_after_load(scratch, Some(block));
            let beside = super::kept_beside(&before, left, 0, false, false, 0);
            let need = fit::serving_need(&before, &c, ctx, 0.85, beside, n == 0, 1, DType::F16);
            super::fits_after_load(n, true, before.vram_free, free(n), need)
        })
        .unwrap();
        assert_eq!(n, planned);
        let plan = fit::fit(
            &before,
            free(n),
            &c,
            ctx,
            0.85,
            serving,
            4,
            false,
            false,
            false,
            0,
            0,
            DType::F16,
        );
        assert!(plan.is_ok());
    }

    /// The MTP block's routed experts leave the device with the first host layer: the plan that
    /// counts them there takes fewer host layers than one that keeps them on the device, and the
    /// check after its load keeps it.
    #[test]
    fn the_mtp_blocks_experts_leave_with_the_first_host_layer() {
        use super::{DeviceProps, Model, fit};
        use larch_kernels::{Backend, CPU, Device};
        let mut before = Device::open(CPU).unwrap().props().unwrap();
        before.backend = Backend::Cuda;
        before.flags = DeviceProps::CAP_GRAPHS;
        (before.vram_total, before.vram_free) = (17_066_033_152, 15_631_122_432);
        let mut c = larch_model::FakeModel::new(248_320).config().clone();
        (
            c.n_layers,
            c.kv_layers,
            c.n_kv_heads,
            c.head_dim,
            c.ctx_train,
        ) = (40, 10, 2, 256, 262_144);
        // The hybrid MoE target's blk.40: 897,955,840 B, 855,638,016 of them routed experts.
        let (weights, ctx, e) = (16_209_005_056u64, 1795, 381_681_664u64);
        let (block, bx) = (897_955_840u64, 855_638_016u64);
        let serving =
            653_144_244 - fit::serving_need(&before, &c, ctx, 0.85, 0, false, 1, DType::F16);
        let own = 578_451_072 - serving;
        let scratch = serving + block;
        let experts = vec![e; 40];
        let reserve = own + super::kept_beside(&before, scratch, 0, false, false, 0);
        let plan = |ex: &[u64]| {
            fit::host_expert_layers(&before, weights, ex, &c, ctx, 0.85, reserve, 1, DType::F16)
        };
        let ex = super::plan_experts(&experts, bx);
        let (device, host) = (plan(&experts), plan(&ex));
        assert_eq!(ex[1..], experts[1..]);
        // The block's experts are more than two layers' worth.
        assert!(host > 0 && host + 2 <= device, "{host} {device}");
        // What the load leaves: the trunk's device weights, its own, and the block's device part.
        let on = |n: u32| if n > 0 { block - bx } else { block };
        let free = |n: u32| before.vram_free - (weights - u64::from(n) * e) - own - on(n);
        let n = super::more_host_layers_on_refusal(host, &ex, true, |n| {
            let left = super::scratch_after_load(scratch, Some(block));
            let beside = super::kept_beside(&before, left, 0, false, false, 0);
            let need = fit::serving_need(&before, &c, ctx, 0.85, beside, n == 0, 1, DType::F16);
            super::fits_after_load(n, true, before.vram_free, free(n), need)
        })
        .unwrap();
        assert_eq!(n, host);
    }

    /// Prefill streaming runs on the one backend it was measured on, and never on unified memory.
    #[test]
    fn only_discrete_cuda_streams() {
        use larch_kernels::{Backend, CPU, Device};
        let mut props = Device::open(CPU).unwrap().props().unwrap();
        let on = larch_kernels::routes()
            .prefill_stream_rows
            .unwrap_or(larch_model::PREFILL_STREAM_ROWS);
        for (backend, flags, rows) in [
            (Backend::Cuda, 0, on),
            (Backend::Cuda, super::DeviceProps::CAP_UNIFIED, 0),
            (Backend::Hip, 0, 0),
            (Backend::Metal, 0, 0),
            (Backend::Cpu, 0, 0),
        ] {
            (props.backend, props.flags) = (backend, flags);
            assert_eq!(super::stream_rows_for(&props), rows, "{backend:?} {flags}");
        }
    }

    /// A crafted header's unused tensors of unknown type may claim any size; the sums saturate.
    #[test]
    fn weights_of_a_crafted_header_saturate() {
        let huge = TensorInfo {
            name: "unused".into(),
            ggml_type: 999,
            dims: vec![1 << 63, 4],
            offset: 0,
            bytes: u64::MAX,
        };
        let g = Gguf {
            kv: Default::default(),
            tensors: vec![huge.clone(), huge],
            data_offset: 0,
        };
        let w = Weights::of(&g);
        assert_eq!(
            (w.bytes, w.params, w.quant.as_str()),
            (u64::MAX, u64::MAX, "")
        );
    }
}
