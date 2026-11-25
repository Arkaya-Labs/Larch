//! The engine thread (architecture 9.5, 9.8, 9.11, 9.13): handle messages (knob changes apply there,
//! at the step boundary), plan, run the step with one host sync, sample, stream, record.

use std::collections::{HashMap, VecDeque};
use std::io::{Seek, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{RecvTimeoutError, SyncSender, TryRecvError, channel, sync_channel};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use larch_kernels::{
    Backend as KBackend, Buffer, DType, Device, DeviceProps, Event, Graph, KError, KvPool,
    SampleParams, Stream, Tensor, ops,
};
use larch_model::card::Route;
use larch_model::projector::{ImagePixels, Projector, ProjectorConfig};
use larch_model::qwen_vision::EncodeJob;
use larch_model::vision::IMAGE_TOKEN;
use larch_model::{
    DflashDrafter, EmbedRun, Model, ModelConfig, MtpDrafter, StateBatch, StepBatch, Taps,
    Transformer,
};
use larch_records::{
    AbArm, AbCommand, AbEnd, AbReply, ActivationClass, ActivationEvent, ActivationResult,
    Authority, Backend, Cause, ConfigurationRevision, DeploymentSnapshot, DeviceSample,
    EffectiveConfig, EngineError, EngineSample, FaultCode, FixedStr, Health, HealthEvent,
    HealthState, KnobReply, KnobRequest, LoadError, MF32, MState, MU64, Phase, REGISTRY,
    RecordBody, RecordHeader, RejectReason, RevertTarget, Tier, knob_id, knob_spec, validate,
};
use larch_sys::nvml::Nvml;

use crate::fit::{
    FitPlan, GRAPH_BYTES, drafter_bytes, fit, graph_arena_reserve, graph_quota, graph_reserve,
    mixed_reserve, mtp_bytes, serving_scratch,
};
use crate::guard::{Guard, Target};
use crate::guess::{Draft, DraftStep, drafted};
use crate::sample::{CANDIDATES, Row, check, top};
use crate::sched::{
    BLOCK_TOKENS, Limits, Logit, PinnedChain, Policy, Sampled, Scheduler, Seq, StepPlan, TAP_SEATS,
};
use crate::suffix::SuffixDrafter;
use crate::{
    Capabilities, EngineHandle, EngineOptions, GuessStats, Loader, MAX_GUESSES, MAX_SCORE_IDS,
    ModelClass, Msg, Position, Published, ReloadChange, RouterDeltas, Shared, TelemetryTier,
    Weights, invalid, model_loader,
};
use crate::{PinInfo, PinOp, PinReply};
use sha2::{Digest, Sha256};

/// `attn_splits` of every step while checks can run and the knob leaves it to the backend.
const GUESS_SPLITS: u32 = 16;

/// The most guesses a check of a load with `opts` tries: `drafted` (the drafter's, see
/// [`crate::guess::drafted`]), else `guesses`. A `recurrent` model's check snapshots a row a guess
/// into its spares, one more taken while its sequence lives in a spare, so it allocates a spare a
/// guess and one ([`Model::alloc_checks`]), at most [`Transformer::CHECK_SLOTS`].
pub(crate) fn guess_count(opts: &EngineOptions, drafted: Option<u32>, recurrent: bool) -> u32 {
    let n = match drafted {
        Some(d) => d,
        None => opts.guesses,
    };
    let spares = match recurrent {
        false => MAX_GUESSES,
        true => Transformer::CHECK_SLOTS - 1,
    };
    n.min(MAX_GUESSES).min(spares)
}

/// The check spares a load with `opts` allocates ([`guess_count`] and one) and the check-step
/// graph slots it reserves, its drafter DFlash with `block` rows or else the MTP block or plain
/// guesses; 0 when it does not guess.
pub(crate) fn check_spares(opts: &EngineOptions, block: Option<u32>) -> u32 {
    if !opts.guessing() {
        return 0;
    }
    let drafted = (block.is_some() || opts.mtp()).then(|| drafted(opts.drafter_guesses, block));
    (guess_count(opts, drafted, true) + 1).min(Transformer::CHECK_SLOTS)
}

/// [`guess_count`] for a loaded model.
fn guess_max(opts: &EngineOptions, loaded: &Loaded) -> usize {
    let drafted = loaded
        .draft
        .as_ref()
        .map(|d| d.guesses(opts.drafter_guesses));
    guess_count(opts, drafted, loaded.model.state_slots() > 0) as usize
}

/// Recent tokens the repetition penalty looks at (llama.cpp's `penalty_last_n`).
const PENALTY_WINDOW: usize = 64;

/// The smallest prefill chunk the scratch is sized for, so a live change or A/B can raise
/// `prefill_chunk_tokens` from its 512 default to 1024 without a reload. An MoE model sizes for
/// its startup chunk instead: on a host-expert plan those rows' scratch (~59 MB on the hybrid MoE
/// target, 11 Nov) is the streaming pool's room, so a larger chunk is set at startup.
pub(crate) const PREFILL_CHUNK_SCRATCH: usize = 1024;

/// Whether a pure decode step runs as a check step (`StepBatch::check`): on Metal, while a guessing
/// load may still check. Its rows then take the check kernels at every row count, one row
/// included, so a row's bits never depend on whether a check runs beside it, and the output with
/// guesses is the output of the same load without them (exact within the mode). Other backends'
/// check kernels already equal their plain decode, and a load without guesses keeps plain decode.
fn decodes_as_check(
    metal: bool,
    guess_max: usize,
    checkable: Option<bool>,
    plan: &StepPlan,
) -> bool {
    metal
        && guess_max > 0
        && checkable != Some(false)
        && plan.prefill.is_empty()
        && !plan.decode.is_empty()
}

/// LARCH_VERIFY_TIME=1: each pure decode or check step prints its rows, guessed rows, whether it ran
/// the check kernels, its wall and GPU time and the last one-row plain step's wall time
/// (`larch_verify_time,...` on stderr), so a run reads a check step's cost in plain steps; and
/// each step that may guess, the guess counters so far and its draft forward's time
/// (`larch_guesses,...`).
fn verify_time_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LARCH_VERIFY_TIME").is_ok_and(|v| v == "1"))
}

fn verify_time(plan: &StepPlan, check: bool, wall_ns: u64, gpu_ns: Option<u64>) {
    static PLAIN1: AtomicU64 = AtomicU64::new(0);
    if !verify_time_on() || !(plan.prefill.is_empty() || plan.guessed > 0) {
        return;
    }
    let rows = plan.decode.len() + plan.prefill.iter().map(|p| p.2).sum::<usize>();
    if plan.guessed == 0 && rows == 1 {
        PLAIN1.store(wall_ns, Ordering::Relaxed);
    }
    eprintln!(
        "larch_verify_time,rows={rows},chats={},guessed={},check={},wall_us={},gpu_us={},plain1_us={}",
        plan.decode.len() + plan.prefill.len(),
        plan.guessed,
        u8::from(check),
        wall_ns / 1000,
        gpu_ns.map_or(-1, |n| (n / 1000) as i64),
        PLAIN1.load(Ordering::Relaxed) / 1000
    );
}

/// Whether a step's time prices the next checks: a plain decode step or a check step (which
/// feeds no prompt), but not a graph capture, whose time is the capture's.
fn prices_checks(plan: &StepPlan, captured: bool) -> bool {
    !captured && (plan.prefill.is_empty() || plan.guessed > 0)
}

/// The decode graph a pure decode step of `rows` replays: on CUDA exactly 9 to 32 rows, else the
/// next power of two (on sm_120, 15 Oct 2026, padding beat eager decode by 34 % at 12 rows and 4 %
/// at 24, so 33 to 63 rows pad to 64).
fn graph_decode_bucket(backend: Backend, rows: usize) -> usize {
    if backend == Backend::Cuda && (9..=32).contains(&rows) {
        rows
    } else {
        rows.next_power_of_two()
    }
}

/// A captured step's shape: decode rows (padding included), prefill rows and logits rows, plus the
/// attention splits it froze. Decode buckets and mixed full-chunk shapes share one cache and quota.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct GraphKey {
    decode: usize,
    prefill: usize,
    logits: usize,
    splits: u32,
    /// Replays the drafter's taps, whose buffers go with the drafter.
    taps: bool,
    /// A check step's chunks, 0 for any other shape: its key can't collide with a mixed one.
    checks: usize,
}

/// The bounded mixed shapes: one full prefill chunk beside a full or half admission's decode rows,
/// with or without the chunk's final logits row.
fn mixed_key(
    backend: Backend,
    (chunk, max_seqs): (usize, usize),
    nd: usize,
    prefill: Option<usize>,
    logits: usize,
    splits: u32,
) -> Option<GraphKey> {
    let bounded = backend == Backend::Cuda
        && nd != 0
        && prefill == Some(chunk)
        && (nd..=nd + 1).contains(&logits)
        && [max_seqs, max_seqs.div_ceil(2)].contains(&(nd + 1));
    bounded.then_some(GraphKey {
        decode: nd,
        prefill: chunk,
        logits,
        splits,
        taps: false,
        checks: 0,
    })
}

/// A check step's shape, exact: decode rows lead its check rows and padding must trail the live rows.
// ponytail: exact check shapes; bucket np with a trailing pad chunk when a bench shows check steps
// refused or missing the second hit.
fn check_key(
    backend: Backend,
    nd: usize,
    np: usize,
    chunks: usize,
    logits: usize,
    splits: u32,
    taps: bool,
) -> Option<GraphKey> {
    (backend == Backend::Cuda).then_some(GraphKey {
        decode: nd,
        prefill: np,
        logits,
        splits,
        taps,
        checks: chunks,
    })
}

/// The check captures to drop, least recently replayed first, from `held` (last replayed step, key,
/// bytes) so that `free` quota bytes reach `need`, and the bytes free after.
fn evict_lru(
    mut held: Vec<(u64, GraphKey, u64)>,
    mut free: u64,
    need: u64,
) -> (Vec<GraphKey>, u64) {
    held.sort_unstable_by_key(|h| h.0);
    let mut gone = Vec::new();
    for (_, k, bytes) in held {
        if free >= need {
            break;
        }
        gone.push(k);
        free += bytes;
    }
    (gone, free)
}

/// Frees quota for a plain decode key: check captures go, least recently replayed first, until
/// `free` reaches `need` or none is left. The quota bytes free after.
fn evict_checks(
    graphs: &mut HashMap<GraphKey, Capture>,
    used: &mut HashMap<GraphKey, u64>,
    free: u64,
    need: u64,
) -> u64 {
    if free >= need {
        return free;
    }
    let held = graphs.iter().filter_map(|(k, c)| match c {
        Capture::Graph(.., bytes) if k.checks != 0 => {
            Some((used.get(k).copied().unwrap_or(0), *k, *bytes))
        }
        _ => None,
    });
    let (gone, free) = evict_lru(held.collect(), free, need);
    for k in &gone {
        graphs.remove(k);
        used.remove(k);
    }
    free
}

/// What a refused capture saw. It is retried once either changes for the better: more free quota
/// bytes, or the stream's arena allocated; a refusal over a measured charge also once more,
/// [`RETRY_STEPS`] later. Knob changes that alter a shape give it a new key; a reload starts with
/// an empty cache. An A/B flip alone never retries one.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Refusal {
    /// Quota bytes left beside the live captures' measured sizes.
    free: u64,
    arena: bool,
    /// The engine step this describes: now, or when the capture was refused.
    step: u64,
    /// One retry is owed whatever else changes: a capture charged past the quota by the drop in
    /// free memory, which the next reading may not repeat.
    again: bool,
}

/// Whether a capture of a shape whose cache entry is `prev` runs the one retry its refusal owed;
/// a capture that does is owed none if refused again.
fn owed_retry(prev: Option<&Capture>, room: &Refusal) -> bool {
    matches!(prev, Some(Capture::Refused(r)) if r.again && r.retry(room))
}

/// Steps a capture refused over its measured charge waits before its one owed retry.
// ponytail: WSL's free figure swung 40-70 MB low across one capture window in 1 of 5 fresh loads
// (the hybrid MoE on the 5070 Ti; before and after both low, still low after a sync), so the
// retry waits past that window: 256 one-row steps is about 3 s eager. Re-derive from
// larch_graph_retry lines if retries are refused again.
const RETRY_STEPS: u64 = 256;

impl Refusal {
    fn retry(&self, now: &Refusal) -> bool {
        now.free > self.free
            || now.arena != self.arena
            || (self.again && now.step >= self.step + RETRY_STEPS)
    }

    /// The refusal of a capture that measured `bytes` past the quota left: its shape retries once
    /// the quota has room for that size.
    fn over(&self, bytes: u64) -> Option<Refusal> {
        (bytes > self.free).then_some(Refusal {
            free: bytes - 1,
            ..*self
        })
    }
}

/// The one line (LARCH_FIT_DIAGNOSTICS) a process prints at its first decode step that runs eager
/// on refusal `r`: the bucket, why, and the quota bytes it is short beside `free` left of `quota`.
/// A quota refusal needs a capture's floor or the size it measured; a `failed` capture needs none.
fn eager_line(decode: usize, r: &Refusal, free: u64, quota: u64, failed: Option<&str>) -> String {
    let (why, short) = match failed {
        Some(e) => (format!("capture failed ({e})"), 0),
        None => {
            let need = r.free.saturating_add(1).max(GRAPH_BYTES);
            ("quota".into(), need.saturating_sub(free))
        }
    };
    format!(
        "larch_graph_eager,decode={decode},reason={why},short_bytes={short},free_bytes={free},quota_bytes={quota}"
    )
}

/// Whether the eager step counted after `before` others prints [`eager_line`]: only the first, and
/// only with `diagnostics` (read only then) "1".
fn says_eager(before: u64, diagnostics: impl FnOnce() -> Option<String>) -> bool {
    before == 0 && diagnostics().as_deref() == Some("1")
}

/// What a capture is charged: the device bytes free memory lost from before it (`before`) to after
/// its first replay (`after`), which uploads its executables, less the capture `arena` it
/// allocated; never less than the fit's estimate of one capture (`GRAPH_BYTES`). Free memory moves
/// in driver granules, so a reading can show nothing for a capture that took memory (a non-hybrid
/// MoE's 14-piece capture read 0); the floor keeps the quota from admitting more captures than the
/// fit reserved.
fn capture_charge(before: u64, after: u64, arena: u64) -> u64 {
    before
        .saturating_sub(after)
        .saturating_sub(arena)
        .max(GRAPH_BYTES)
}

/// [`capture_charge`] for a first reading of free memory `after`; one that exceeds the quota
/// `room` left is taken again through `reread` (stream synced, free read), and the second charge
/// stands, fits or not. A failed second reading keeps the first.
fn remeasured_charge<E>(
    (room, before, arena): (u64, u64, u64),
    after: u64,
    reread: impl FnOnce() -> Result<u64, E>,
) -> u64 {
    let first = capture_charge(before, after, arena);
    if first <= room {
        return first;
    }
    reread().map_or(first, |again| capture_charge(before, again, arena))
}

/// Whether the device holds a capture measured past the quota left `room` anyway: free memory
/// after it (`free`, re-read after a sync) still covers the fit's `headroom` and all of `room`.
/// Free memory is device-wide and WSL's reading swings by tens of MB, so the charge over-counted;
/// the capture is then charged the floor, and free memory still holds what the plan promises.
/// Otherwise the shortage is real (or the reading too low to tell), and it is refused.
fn device_holds(room: u64, free: u64, headroom: u64) -> bool {
    free >= headroom.saturating_add(room)
}

/// One shape's capture state.
enum Capture {
    /// A mixed or check shape seen once: one-off shapes stay eager and capture on the second hit.
    Seen,
    /// The pieces in order, each with the model's host step after it ([`Model::run_host`]), then
    /// the last piece and the device bytes the capture took, as measured.
    Graph(Vec<(Graph, u64)>, Graph, u64),
    /// Runs eager at its exact rows, without padding rows, until retried (its tables still go
    /// out at the full width a graph-capable backend sends).
    Refused(Refusal),
}

/// One applied revision, for reverts (9.8).
struct Rev {
    rev: u64,
    knob: u16,
    previous: f64,
    trial_id: u64,
    cause: Cause,
}

/// A running A/B (9.9).
#[derive(Clone)]
pub(crate) struct Ab {
    pub(crate) trial: u64,
    /// (knob, arm A value, arm B value).
    pub(crate) knobs: Vec<(u16, f64, f64)>,
    pub(crate) period: u64,
    /// 0 = until stopped.
    pub(crate) max_steps: u64,
    /// Its first step.
    pub(crate) start: u64,
}

/// Sizes of the per-step buffers: inputs (i32), logits (f32) and `sample_prep`'s outputs.
#[derive(Clone, Copy)]
pub(crate) struct Dims {
    /// Batch rows: `max_batch_seqs`' registry max plus `Capabilities::prefill_chunk_max`.
    rows: usize,
    seqs: usize,
    max_blocks: usize,
    vocab: usize,
    /// Activations requests one step taps ([`TAP_SEATS`]): their batch rows go up after the score
    /// ids, and their rows come back after the scores, in the step's one upload and download.
    tap_seats: usize,
    /// f32 a step's taps can fill: `tap_seats` requests of every layer, or one layer of every batch row.
    tap_floats: usize,
}

impl Dims {
    /// A model's buffers at `rows` batch rows (the registry's largest batch plus the prefill chunk)
    /// and `max_context` clamped to its training context.
    pub(crate) fn of(c: &ModelConfig, rows: usize, max_context: u32) -> Dims {
        Dims {
            rows,
            seqs: knob_spec(knob_id::MAX_BATCH_SEQS).expect("registered").max as usize,
            max_blocks: (max_context.min(c.ctx_train) as usize).div_ceil(BLOCK_TOKENS),
            vocab: c.vocab as usize,
            tap_seats: rows.max(TAP_SEATS),
            tap_floats: (TAP_SEATS * c.n_layers as usize).max(rows) * c.d_model as usize,
        }
    }

    /// Initial serving buffers; forced-run growth is checked only when requested.
    pub(crate) fn fit_bytes(&self) -> u64 {
        self.bytes(self.seqs).into_iter().sum()
    }

    /// Growing to `rows` logits rows: old buffers fund net growth only after replacement, so the
    /// allocation peak needs fresh free bytes.
    fn growth_fits(&self, free: u64, current_rows: usize, rows: usize, reserve: u64) -> bool {
        let fresh = self.bytes(rows).into_iter().sum::<u64>();
        free >= fresh && free + self.bytes(current_rows).into_iter().sum::<u64>() >= fresh + reserve
    }

    /// The most logits rows, up to a step's `rows`, that [`Self::growth_fits`]; the current ones
    /// at least, which need no growth.
    fn rows_fitting(&self, free: u64, current_rows: usize, reserve: u64) -> usize {
        let (mut lo, mut hi) = (current_rows, self.rows);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if self.growth_fits(free, current_rows, mid, reserve) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    /// Bytes of (input, logits, samp) for steps of up to `logit_rows` logits rows. A step samples
    /// each of its sequences at most once, so its score ids go up and their logits come back in
    /// `seqs × MAX_SCORE_IDS` slots; its taps (one step's worth, at their full width) follow, then a
    /// hybrid model's two ints a sequence.
    fn bytes(&self, logit_rows: usize) -> [u64; 3] {
        let scores = self.seqs * MAX_SCORE_IDS;
        let ints = self.rows * 3
            + self.seqs * (self.max_blocks + 5)
            + logit_rows * (1 + PENALTY_WINDOW)
            + scores
            + self.tap_seats;
        [
            ints,
            logit_rows * self.vocab,
            logit_rows * (CANDIDATES * 2 + 1) + scores + self.tap_floats,
        ]
        .map(|n| (n * 4).max(4) as u64)
    }
}

/// Router deltas, their version and the caller waiting for them.
type Swap = (
    Option<RouterDeltas>,
    u64,
    SyncSender<Result<(), EngineError>>,
);

/// Work on the resident model and the engine's stream, run at a boundary with no started request.
pub(crate) type Job = Box<dyn FnOnce(Option<&Transformer>, &Stream) + Send>;

pub(crate) struct Engine {
    sh: Arc<Shared>,
    opts: EngineOptions,
    s: Stream,
    loaded: Option<Loaded>,
    partial: Partial,
    reload: Option<PendingReload>,
    sched: Scheduler,
    cfg: EffectiveConfig,
    // ponytail: one entry per applied change, never pruned (a revert of an old revision must still
    // answer Superseded); bound it if gait's change rate ever makes it matter.
    history: Vec<Rev>,
    step: u64,
    engine_id: u64,
    ab: Option<Ab>,
    guard: Guard,
    /// `decode_graphs` just before a custom reload forced it off, which the next ordinary reload restores.
    graphs_before_custom: f64,
    /// A custom reload turned decode graphs off; the next ordinary reload restores `graphs_before_custom`.
    graphs_forced_off: bool,
    /// The highest adaptation version applied, which a version at or under it may reuse with
    /// other deltas.
    top_version: u64,
    /// Router deltas waiting for a boundary with no started request, their version and the caller.
    swap: Option<Swap>,
    /// A job waiting for such a boundary (learning's device trainer).
    job: Option<Job>,
    /// Images whose tiles the engine encodes while it has nothing else to run (S5, encode on
    /// paste), newest first.
    prepare: VecDeque<Prepared>,
    /// The step before which no live state grow is tried again ([`Engine::grow_seats`]).
    grow_at: u64,
    /// `max_batch_seqs` was set by hand since the load: [`Engine::replan_seats`] once idle.
    replan: bool,
    /// The plan a documents reload replaced, to go back to once no pin is held and a request needs
    /// a seat the documents plan lacks, or the engine has idled [`DOCS_RESTORE_IDLE`].
    docs_before: Option<DocsBefore>,
    /// A request waits for a seat the documents plan lacks and no pin is held: new requests wait
    /// for the started ones to end, then the plan goes back ([`Engine::restore_ready`]).
    restoring: bool,
    /// When a request last ran or waited, the last pin dropped, a pin step ran or a documents
    /// reload ended.
    idle_since: Instant,
    /// What the documents plan reads from the model's file, read on the first
    /// [`Engine::doc_fit`] of a load.
    doc_terms: Option<crate::DocTerms>,
    /// Host expert layers of the last load outside a documents plan: where host experts prefill on
    /// the CPU, the most a documents plan may keep ([`crate::document_plan`]).
    base_host_layers: u32,
    /// The device memory free before that load: every documents fit reads the device as that load
    /// found it, so a fit inside the plan answers as the one that reloaded into it. Each load
    /// measures on a stream of its own (`Engine::renew_stream`), so a reload finds what the first
    /// load did, not less by what the last plan's stream grew.
    base_free: u64,
    /// Imports in progress ([`PinOp::Begin`]): pin id to its blocks, taken but not yet published.
    imports: HashMap<u64, ImportJob>,
    /// Imports a reload broke off, oldest first: each one's next step is told `IMPORT_TORN`, not
    /// `PIN_CHANGED`, so its caller keeps the good file and imports it again. An id goes once
    /// told; at most [`TORN_KEPT`] are kept.
    torn: VecDeque<u64>,
    /// Demand picks a hybrid's plan (fast or many chats): nobody set `max_batch_seqs` or
    /// `LARCH_EXPERT_CACHE` by hand.
    auto: bool,
    demand: Demand,
    /// A demand switch waits for the started requests ([`Engine::switch_seats`]).
    switch: Option<Switch>,
}

/// What a documents reload replaced ([`ReloadChange::documents`]).
struct DocsBefore {
    max_context: u32,
    kv_pool_fraction: f32,
    /// `max_batch_seqs` as it stood, and whether a person had set it.
    seats: (f64, bool),
    drafter: (Option<std::path::PathBuf>, Option<std::path::PathBuf>),
    /// What the documents reloads set, which a field goes back from only while it still holds:
    /// a setting changed in between is the person's.
    set: DocsSet,
}

/// The fields documents reloads set (None: left alone).
#[derive(Default)]
struct DocsSet {
    max_context: Option<u32>,
    kv_pool_fraction: Option<f32>,
    seats: Option<f64>,
    drafter_off: bool,
    /// The load into the plan (`EngineOptions::documents`) turned the demand switch off, which
    /// goes back on with the seats.
    auto_off: bool,
}

/// An import's blocks and what they will hold.
struct ImportJob {
    namespace: u64,
    tokens: Vec<u32>,
    blocks: Vec<u32>,
}

/// An image whose units wait to be encoded into the unit cache while the engine is idle: `next` is
/// the next unit. A warm-up (`keep` false) leaves no slot behind: it only runs the encoder's
/// kernels once at load, so the first image does not pay for their first launch.
struct Prepared {
    digest: [u8; 32],
    pixels: Arc<ImagePixels>,
    next: u32,
    keep: bool,
}

/// Resources whose lifetimes end before refitting, so old allocations cannot consume the new fit.
struct Loaded {
    /// Captures must die before the model and buffers whose addresses they retain.
    graphs: HashMap<GraphKey, Capture>,
    /// The step that last replayed each check capture, for [`evict_checks`].
    check_used: HashMap<GraphKey, u64>,
    /// [`larch_kernels::Routes::mixed_graphs`]: bounded mixed shapes capture too.
    mixed_graphs: bool,
    headroom_bytes: u64,
    /// Successful-live executable quota; forced observation growth may only decrease it.
    graph_bytes: u64,
    /// The capture arena still to be allocated, kept from observation growth even when the quota
    /// reaches zero; 0 once the stream holds it.
    graph_arena_bytes: u64,
    model: Box<dyn Model>,
    k: Buffer,
    v: Buffer,
    input: Buffer,
    logits: Buffer,
    samp: Buffer,
    staging: Staging,
    dims: Dims,
    /// Logits rows `logits` and `samp` hold, grown to the largest observed step so far.
    logit_rows: usize,
    /// Weight bytes one step reads, when the GGUF header was read.
    weights_read: Option<u64>,
    /// K and V bytes of one token, every layer.
    kv_token_bytes: u64,
    info: Published,
    /// The drafter, with `EngineOptions::drafter`.
    draft: Option<Draft>,
    /// The state slots a planned load paid host layers for (`Weights::asked`); None when the
    /// loader did not plan (a custom loader, forced host layers).
    seats_planned: Option<u32>,
    /// The vision projector, when the engine loaded one.
    vision: Option<VisionRt>,
    /// The GGUF's identity at this load ([`file_id`]), which a reload's carry compares; None for
    /// a custom loader or a file that could not be read.
    file: Option<u64>,
    /// Device memory free before this load took any, which a reload would find again: the
    /// documents plan's `before` ([`Engine::doc_fit`]).
    free_before: u64,
    /// Expert layers this load keeps in host memory (`Weights::host_layers`); 0 without any.
    host_layers: u32,
    /// Its host experts prefill through the stream pool, not on the CPU.
    streams: bool,
}

/// Encoded units the engine keeps, InternVL tiles: a step's image rows come from them, and a later
/// turn whose prefix lost its KV finds a recent image's tiles here.
const TILE_SLOTS: usize = 16;
/// The same for Qwen images, each a unit of up to `max_tokens` rows (13 MB a slot at 1600 rows of
/// 2048).
const IMAGE_SLOTS: usize = 4;

/// The projector and everything image rows need, allocated at load, so a step allocates nothing:
/// encoding a large image cannot run the device out of memory mid-serve.
struct VisionRt {
    model: Projector,
    config: ProjectorConfig,
    /// f32 `[slots * max_unit_rows, d_model]`: slot i at rows `i * max_unit_rows..`.
    tiles: Buffer,
    /// Each slot's (image digest, unit, last use).
    slots: Vec<Option<([u8; 32], u32, u64)>>,
    /// f32 `[rows, d_model]`: a step's image rows at their batch rows, copied from the tiles before
    /// a later encode in the step may take a slot.
    rows: Buffer,
    clock: u64,
    /// An image unit encoding a slice of layers a step (a sliced projector's), its activations in
    /// the projector's scratch: (digest, unit, rows, job). Any other encode settles it first.
    job: Option<([u8; 32], u32, usize, EncodeJob)>,
    /// GPU ns per encoder layer per image row, averaged over the slices run so far; None before the
    /// first.
    layer_row_ns: Option<f64>,
    /// The last slice's events around its layers, its layer count and rows, read at the next slice.
    timing: Option<(Event, Event, u32, usize)>,
}

/// GPU time a step spends on an image encode beside other requests: a pause under about 50 ms is hard
/// to see in streamed text (#424's pacing). Measured on the M5: 2B at 920 rows 16 ms a layer, 35B 31.
// ponytail: one constant; a per-chip value if M1 to M4's pauses stay visible in vision_pause.
const ENCODE_BUDGET_NS: f64 = 30e6;

/// Layers of a `rows`-row encode that fit [`ENCODE_BUDGET_NS`] at `layer_row_ns`, at least 1 and at
/// most `max`; 1 until a slice has been timed.
fn budget_layers(layer_row_ns: Option<f64>, rows: usize, max: u32) -> u32 {
    layer_row_ns.map_or(1, |ns| {
        (ENCODE_BUDGET_NS / (ns * rows.max(1) as f64)).clamp(1.0, f64::from(max.max(1))) as u32
    })
}

impl VisionRt {
    /// Finishes the encode in flight, if any, into the least recently used slot: every layer it has
    /// left, the merger, the slot. Whether one finished.
    fn settle(&mut self, s: &Stream, d_model: usize) -> Result<bool, KError> {
        let Some((digest, unit, rows, mut job)) = self.job.take() else {
            return Ok(false);
        };
        self.model.advance(s, &mut job, u32::MAX)?;
        let i = self.lru();
        self.slots[i] = None;
        self.model
            .finish(s, job, &self.slot_rows(i, rows, d_model)?)?;
        self.clock += 1;
        self.slots[i] = Some((digest, unit, self.clock));
        Ok(true)
    }

    /// One slice of the encode of unit `unit` (`rows` rows) of image `digest` (`pixels`): begun if
    /// none is in flight, else a slice of the unit in flight, which goes first; finished into a slot
    /// when its layers are done. Whether a unit finished.
    fn slice(
        &mut self,
        s: &Stream,
        digest: [u8; 32],
        unit: u32,
        rows: usize,
        pixels: &ImagePixels,
        d_model: usize,
    ) -> Result<bool, KError> {
        // Another unit's encode in flight goes on first, under the same budget: settling it whole
        // here would pause the other chats for all its layers. This unit begins once it is done.
        if self.job.is_none() {
            let job = self.model.begin(s, pixels, unit)?;
            self.job = Some((digest, unit, rows, job));
        }
        let rows = self.job.as_ref().expect("a job").2;
        // The last slice ran in an earlier step, which has synced: its events are done.
        if let Some((t0, t1, layers, rows)) = self.timing.take() {
            let ns = t0.elapsed_ns(&t1)? as f64 / (f64::from(layers) * rows.max(1) as f64);
            self.layer_row_ns = Some(self.layer_row_ns.map_or(ns, |avg| (avg + ns) / 2.0));
        }
        let layers = budget_layers(self.layer_row_ns, rows, self.model.layers());
        let (.., job) = self.job.as_mut().expect("a job");
        let before = job.layers_done();
        let t0 = Event::record(s)?;
        let done = self.model.advance(s, job, layers)?;
        let ran = job.layers_done() - before;
        if ran > 0 {
            self.timing = Some((t0, Event::record(s)?, ran, rows));
        }
        if done {
            return self.settle(s, d_model);
        }
        Ok(false)
    }

    /// The least recently used slot (an empty one first).
    fn lru(&self) -> usize {
        (0..self.slots.len())
            .min_by_key(|&i| self.slots[i].map_or(0, |e| e.2))
            .expect("slots")
    }

    /// Slot `i`'s first `rows` rows, f32 `[rows, d_model]`.
    fn slot_rows(&self, i: usize, rows: usize, d_model: usize) -> Result<Tensor<'_>, KError> {
        let unit = self.config.max_unit_rows() as usize;
        if rows > unit {
            return Err(KError::InvalidArg);
        }
        Tensor::new(
            &self.tiles,
            (i * unit * d_model * 4) as u64,
            DType::F32,
            &[rows as i64, d_model as i64],
        )
    }

    /// The slot holding unit `tile` (`rows` rows) of the image `digest`, after encoding it into the
    /// least recently used slot when none does (`pixels` gives the image's units then), and whether
    /// it encoded.
    fn tile_slot(
        &mut self,
        s: &Stream,
        digest: [u8; 32],
        tile: u32,
        rows: usize,
        pixels: impl FnOnce() -> Result<Arc<ImagePixels>, KError>,
        d_model: usize,
    ) -> Result<(usize, bool), KError> {
        self.clock += 1;
        let held =
            (self.slots.iter()).position(|e| e.is_some_and(|e| e.0 == digest && e.1 == tile));
        if let Some(i) = held {
            self.slots[i] = Some((digest, tile, self.clock));
            return Ok((i, false));
        }
        // An encode in flight holds the scratch this one would use: finish it first (into its own
        // slot), and this unit may be the one it was.
        if self.job.is_some() {
            self.settle(s, d_model)?;
            if let Some(i) =
                (self.slots.iter()).position(|e| e.is_some_and(|e| e.0 == digest && e.1 == tile))
            {
                return Ok((i, true));
            }
        }
        let i = self.lru();
        let px = pixels()?;
        // Empty until the encode is in: a failed one never leaves a slot naming rows it lacks.
        self.slots[i] = None;
        let out = self.slot_rows(i, rows, d_model)?;
        self.model.encode_unit(s, &px, tile, &out)?;
        self.slots[i] = Some((digest, tile, self.clock));
        Ok((i, true))
    }
}

/// Host buffers retain capacity across steps; their contents never outlive a synchronous download.
#[derive(Default)]
struct Staging {
    rows: Vec<Logit>,
    packed: Vec<i32>,
    params: Vec<SampleParams>,
    out: Vec<u32>,
    sampled: Sampled,
    /// The batch rows of the step's taps, padding rows included: block `k` starts at `k * tap_rows`.
    tap_rows: usize,
    /// The step's tapped requests with the batch row of each one's last prompt token, set only
    /// while one is admitted ([`Scheduler::tapping`]).
    taps: Vec<(u64, i32)>,
    /// Every layer they tap, once each, ascending: the tap blocks' order.
    layers: Vec<u16>,
    /// Where their rows start in `out`.
    tap_out: usize,
    /// The model refused the taps: the tapped requests end with `invalid::NO_TAPS`.
    untapped: bool,
    /// The step ran with the drafter's taps of every row.
    drafted: bool,
}

/// Replacements stay owned by the executor until successful installation or terminal fault.
#[derive(Default)]
struct Partial {
    model: Option<Box<dyn Model>>,
    buffers: Vec<Buffer>,
    startup: bool,
    /// Graphs are on by the arch default, not set by hand, so the load may drop them (`graphs_fit`).
    graphs_fit_gated: bool,
    /// `max_batch_seqs` was set by hand: a hybrid's plan pays host layers for that many state
    /// slots (more chats at once), else for one (the fastest single chat), the rest from what is
    /// left (`Weights::asked`).
    seats_asked: bool,
    /// The state slots demand asks a plan for when nobody set them ([`Demand`]); 0 is one.
    seats_auto: u32,
}

/// A parked sequence a same-model reload carries ([`Engine::carry_out`]): K then V of each of its
/// blocks, layer by layer, and its recurrent state.
struct Carry {
    session: u64,
    seq: Seq,
    kv: Vec<u8>,
    state: Vec<u8>,
    /// A pinned chain's, not a session's: `session` and `state` are then unused.
    pin: Option<CarriedPin>,
}

/// What a carried pinned chain brings back besides its blocks ([`Scheduler::carry_pinned`]).
struct CarriedPin {
    namespace: u64,
    checkpoint: Option<Arc<Vec<u8>>>,
    /// The pin went to unnamed files instead of `Carry::kv` ([`Engine::spill_pin`]): its pin file
    /// and its state sidecar, which the OS deletes when they close, even after a crash.
    disk: Option<(std::fs::File, std::fs::File)>,
}

/// Says a pin went cold through a reload: it stays registered, and its next question reads the
/// document again.
fn pin_cold(id: u64, reason: impl std::fmt::Debug) {
    eprintln!("larch_pin_cold,{id},{reason:?}");
}

/// What a carry's bytes must match on the new load: the GGUF file ([`file_id`]), the KV pool's
/// layers and block bytes ([`kv_layout`]) and one state slot's bytes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct CarryKey {
    file: u64,
    kv: (usize, usize),
    state: usize,
}

/// The KV pool's element type ([`KvPool::dtype`]).
const KV_DTYPE: DType = DType::F16;

/// `blocks` split into runs of consecutive ids, each with the index of its first block: a run's
/// K (and its V) is one range of a layer, so it takes one copy, not one a block.
fn block_runs(blocks: &[u32]) -> impl Iterator<Item = (usize, &[u32])> {
    let mut at = 0;
    blocks
        .chunk_by(|&a, &b| a.checked_add(1) == Some(b))
        .map(move |run| {
            at += run.len();
            (at - run.len(), run)
        })
}

/// One block's K (or V) bytes in a layer copy of `len` bytes over `blocks`: 0 when both are empty
/// (nothing to copy), else `InvalidArg` unless `len` is a whole K and V for every block.
fn block_bytes(blocks: &[u32], len: usize) -> Result<usize, KError> {
    let block = len / (2 * blocks.len().max(1));
    match blocks.is_empty() && len == 0 {
        true => Ok(0),
        false if block == 0 || len != 2 * blocks.len() * block => Err(KError::InvalidArg),
        false => Ok(block),
    }
}

/// A host buffer of `len` zero bytes for a layer copy's staging, or `Oom`.
fn staging(buf: &mut Vec<u8>, len: usize) -> Result<(), KError> {
    buf.clear();
    buf.try_reserve_exact(len).map_err(|_| KError::Oom)?;
    buf.resize(len, 0);
    Ok(())
}

/// One layer of `blocks`' KV (pool buffers `k` and `v`) in a carry's order, each block's K then
/// its V, into `out`: one download of K and one of V a run of consecutive blocks (each download
/// syncs the stream), then interleaved on the host. On the 128k Folio run, a block at a time took
/// about 85 ms a layer of an 8,192-token segment. Staging: a run's K and V, at most `out`'s size.
fn layer_download(
    s: &larch_kernels::Stream,
    (k_pool, v_pool): (&Buffer, &Buffer),
    n_blocks: u64,
    layer: u64,
    blocks: &[u32],
    out: &mut [u8],
) -> Result<(), KError> {
    let block = block_bytes(blocks, out.len())?;
    let (mut k, mut v) = (Vec::<u8>::new(), Vec::<u8>::new());
    for (i, run) in block_runs(blocks) {
        let at = (layer * n_blocks + u64::from(run[0])) * block as u64;
        staging(&mut k, run.len() * block)?;
        staging(&mut v, run.len() * block)?;
        s.download(k_pool, at, &mut k)?;
        s.download(v_pool, at, &mut v)?;
        let dst = &mut out[2 * i * block..2 * (i + run.len()) * block];
        for ((pair, k), v) in (dst.chunks_exact_mut(2 * block))
            .zip(k.chunks_exact(block))
            .zip(v.chunks_exact(block))
        {
            pair[..block].copy_from_slice(k);
            pair[block..].copy_from_slice(v);
        }
    }
    Ok(())
}

/// [`layer_download`]'s inverse: `bytes` into `blocks` of one layer, one upload of K and one of V
/// a run of consecutive blocks. Staging: a run's K (then its V), at most half of `bytes`.
fn layer_upload(
    s: &larch_kernels::Stream,
    (k_pool, v_pool): (&Buffer, &Buffer),
    n_blocks: u64,
    layer: u64,
    blocks: &[u32],
    bytes: &[u8],
) -> Result<(), KError> {
    let block = block_bytes(blocks, bytes.len())?;
    let mut half = Vec::<u8>::new();
    for (i, run) in block_runs(blocks) {
        let at = (layer * n_blocks + u64::from(run[0])) * block as u64;
        let src = &bytes[2 * i * block..2 * (i + run.len()) * block];
        for (skip, pool) in [(0, k_pool), (1, v_pool)] {
            half.clear();
            half.try_reserve_exact(run.len() * block)
                .map_err(|_| KError::Oom)?;
            half.extend(src.chunks_exact(block).skip(skip).step_by(2).flatten());
            s.upload(pool, at, &half)?;
        }
    }
    Ok(())
}

/// How long the engine must sit with no request running or queued, no pin held and no import
/// open, before the documents plan gives way to the plan before it (a reload the user would see).
/// The plan costs plain chats their drafter and seats (about a fifth of their decode speed in a
/// Mac bench), and the app lets an idle chat's pin go after 5 minutes (`PIN_IDLE`), so the plan
/// goes back at the next quiet minute after the last pin goes. A minute, not less: a restart that
/// loads straight into a chat's plan (`EngineOptions::documents`) leaves the chat's first
/// question that long to take its pin, and plain use pauses that long between turns now and
/// then. A pin step and a documents reload start the clock again.
// ponytail: fixed at a minute; revisit if a pinned chat's first question after a restart pays
// for a reload, or plain chats on a busy engine stay on the documents plan.
const DOCS_RESTORE_IDLE: Duration = Duration::from_secs(60);

/// Torn imports remembered ([`Engine::torn`]): one an import's caller never steps again (it was
/// gone, or another reload tore more) is forgotten after this many more.
// ponytail: an import per chat reading at once, so 16 is room to spare; raise it if a run of
// reloads tears more imports than that at once.
const TORN_KEPT: usize = 16;

/// The error for an import step on `id` that is not open: `IMPORT_TORN` once, when a reload broke
/// it off ([`Engine::torn`]), else `PIN_CHANGED`.
fn torn_or_changed(torn: &mut VecDeque<u64>, id: u64) -> EngineError {
    match torn.iter().position(|&t| t == id) {
        Some(i) => {
            torn.remove(i);
            EngineError::InvalidRequest(invalid::IMPORT_TORN)
        }
        None => EngineError::InvalidRequest(invalid::PIN_CHANGED),
    }
}

/// A pin step that holds the engine thread longer than this is logged (`larch_pin_stall`).
const PIN_STALL_MS: u128 = 50;

/// The least host memory a reload's carry copies (the 35B's 8 state slots plus 32k tokens of KV).
const CARRY_BYTES: usize = 2 << 30;

/// The most host memory a reload's carry copies: half what the host has available, at least
/// [`CARRY_BYTES`] (a 128k pin of the hybrid MoE target is 2.75 GB). Pinned chains go first, then
/// sessions, least recently used first; those past it are dropped.
fn carry_budget(available: Option<u64>) -> usize {
    let half = available.map_or(0, |a| usize::try_from(a / 2).unwrap_or(usize::MAX));
    half.max(CARRY_BYTES)
}

/// Whether host memory is the device's own (the CPU reference, Apple): a carry copy in RAM would
/// stack on the pool it copies from.
fn host_available_is_device(p: &larch_kernels::DeviceProps) -> bool {
    p.backend == larch_kernels::Backend::Cpu
        || p.flags & larch_kernels::DeviceProps::CAP_UNIFIED != 0
}

/// Host memory free now: a unified device's is its own free memory; elsewhere Linux's
/// MemAvailable. None where it is not read, which leaves the carry at [`CARRY_BYTES`].
// ponytail: Windows and macOS discrete cards keep the 2 GiB floor; read their free RAM when a
// reload there should carry a pin past it.
fn host_available(p: &larch_kernels::DeviceProps) -> Option<u64> {
    if host_available_is_device(p) {
        return Some(p.vram_free);
    }
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = info.lines().find_map(|l| l.strip_prefix("MemAvailable:"))?;
    let kib: u64 = line.trim().strip_suffix("kB")?.trim().parse().ok()?;
    kib.checked_mul(1024)
}

/// The KV pool's layers and one block's bytes in one layer of K (or V), as [`KvPool`] lays it out:
/// `[layer][block][token][head][dim]`.
fn kv_layout(mc: &ModelConfig) -> (usize, usize) {
    let per_block = BLOCK_TOKENS as u64 * u64::from(mc.n_kv_heads) * u64::from(mc.head_dim);
    (mc.kv_layers as usize, KV_DTYPE.bytes(per_block) as usize)
}

/// Bytes [`file_id`] hashes at each of its spread offsets.
const ID_SAMPLE: u64 = 64 << 10;

/// The GGUF at `path` as a carry tells it apart: a hash of its length, write time, on Unix its
/// device, inode and change time (which no tool sets back), and 16 samples of its bytes spread
/// from first to last, so a same-shaped rewrite that keeps every timestamp (`cp -p`, a Windows
/// rename that tunnels the creation time) still differs.
// ponytail: 1 MiB sampled, not every byte; hash the whole file if an edit of a few tensors in
// place, timestamps kept, ever needs to be told apart on Windows.
fn file_id(path: &std::path::Path) -> Option<u64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).ok()?;
    let m = f.metadata().ok()?;
    // Sha-256, not `DefaultHasher`, whose output may change with the toolchain: a `.pin` file
    // keeps this id across builds.
    let mut h = Sha256::new();
    let written = m
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    h.update(m.len().to_le_bytes());
    h.update(written.as_secs().to_le_bytes());
    h.update(written.subsec_nanos().to_le_bytes());
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        h.update(m.dev().to_le_bytes());
        h.update(m.ino().to_le_bytes());
        h.update(m.ctime().to_le_bytes());
        h.update(m.ctime_nsec().to_le_bytes());
    }
    let mut sample = Vec::with_capacity(ID_SAMPLE as usize);
    for i in 0..16 {
        f.seek(SeekFrom::Start(m.len().saturating_sub(ID_SAMPLE) * i / 15))
            .ok()?;
        sample.clear();
        (&mut f).take(ID_SAMPLE).read_to_end(&mut sample).ok()?;
        h.update(&sample);
    }
    Some(u64::from_le_bytes(h.finalize()[..8].try_into().expect("8")))
}

/// A zeroed buffer of `n` bytes, or None where the host cannot give it (a carry drops instead).
pub(crate) fn host_bytes(n: usize) -> Option<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, 0);
    Some(v)
}

/// What `l`'s parked bytes are tied to ([`CarryKey`]); None for a model not loaded from a file.
fn carry_key(l: &Loaded) -> Option<CarryKey> {
    Some(CarryKey {
        file: l.file?,
        kv: kv_layout(l.model.config()),
        state: l.model.state_size(),
    })
}

struct PendingReload {
    change: ReloadChange,
    load: Option<Loader>,
    reply: SyncSender<Result<(), EngineError>>,
}

fn wall_ns() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as i64)
}

/// The watchdog's floor: a step that runs on the CPU, or computes experts kept in RAM there (a 512-row
/// prefill chunk takes seconds), runs at the CPU's pace, not a GPU's. Apple GPUs span about 10x in
/// speed, and a 2048-row MoE prefill chunk takes 6.4 s on the M5 without the tensor API, so 60 s holds
/// a GPU about 9x slower. macOS aborts a stuck command buffer, which larch reports as device lost, so
/// on Metal the watchdog is only the backstop.
fn step_floor(opts: &EngineOptions, backend: Backend, class: ModelClass) -> Duration {
    opts.step_timeout_floor.unwrap_or(Duration::from_secs(
        if matches!(backend, Backend::Cpu | Backend::Metal) || class.experts_in_ram {
            60
        } else {
            5
        },
    ))
}

fn backend(backend: KBackend) -> Backend {
    match backend {
        KBackend::Cpu => Backend::Cpu,
        KBackend::Cuda => Backend::Cuda,
        KBackend::Hip => Backend::Hip,
        KBackend::Metal => Backend::Metal,
    }
}

fn backend_err(e: KError) -> LoadError {
    match e {
        KError::Driver { have, need } if have > 0 => LoadError::DriverTooOld { have, need },
        e => LoadError::Backend(e.code()),
    }
}

/// Execution and rebuild carry the same backend status into the common fault path.
fn backend_fault(code: u16) -> (FaultCode, EngineError) {
    if code == KError::DeviceLost.code() || code == KError::Timeout.code() {
        (FaultCode::DeviceLost, EngineError::DeviceLost)
    } else if code == KError::Oom.code() {
        (FaultCode::Oom, EngineError::Internal(code))
    } else {
        (FaultCode::LaunchFailure, EngineError::Internal(code))
    }
}

fn gpu_time(elapsed: Result<u64, KError>) -> Result<Option<u64>, KError> {
    match elapsed {
        Ok(ns) => Ok(Some(ns)),
        Err(KError::Unsupported) => Ok(None),
        Err(error) => Err(error),
    }
}

fn allocate(dev: &Device, bytes: u64) -> Result<Buffer, KError> {
    #[cfg(test)]
    tests::ALLOCATIONS.with(|count| count.set(count.get() + 1));
    #[cfg(any(test, debug_assertions, feature = "fault-inject"))]
    crate::fault::allocation()?;
    Buffer::alloc(dev, bytes)
}

/// A step's one download of `samp` (`Stream::download` syncs); tests count them.
fn download(s: &Stream, samp: &Buffer, out: &mut [u32]) -> Result<(), KError> {
    #[cfg(test)]
    tests::DOWNLOADS.with(|count| count.set(count.get() + 1));
    s.download(samp, 0, out)
}

/// Whether a load into the documents plan (`EngineOptions::documents`) stands: it skips the cap a
/// documents reload has ([`crate::document_plan`]), so it may keep no expert layers the CPU
/// prefills (`host_layers` on the host, not `streams`); a memory that shrank since the plan was
/// recorded would otherwise read the whole document's experts there.
fn direct_documents_fit(host_layers: u32, streams: bool) -> bool {
    host_layers == 0 || streams
}

fn alloc(dev: &Device, bytes: u64, startup: bool) -> Result<Buffer, LoadError> {
    allocate(dev, bytes.max(4)).map_err(|e| match e {
        KError::Oom if startup => LoadError::DoesNotFit {
            need: bytes,
            have: dev.props().map_or(0, |p| p.vram_free),
        },
        e => backend_err(e),
    })
}

/// An estimate, or the state it has when it cannot be one.
fn estimate(state: MState, v: Option<f32>) -> MF32 {
    match (state, v) {
        (MState::Unsupported, _) => MF32::unsupported(),
        (_, Some(v)) => MF32::estimated(v),
        (_, None) => MF32::missing(),
    }
}

pub(crate) fn run(
    opts: EngineOptions,
    load: Loader,
    weights: Option<Weights>,
    ready: SyncSender<Result<EngineHandle, LoadError>>,
) {
    match Engine::init(opts, load, weights) {
        Ok((mut e, handle)) => {
            let watchdog = match crate::lifecycle::watchdog(&e.sh, e.opts.on_fault) {
                Ok(thread) => thread,
                Err(_) => {
                    let _ = ready.send(Err(LoadError::Io));
                    return;
                }
            };
            let sampler = match crate::lifecycle::sampler(&e.sh, e.opts.telemetry) {
                Ok(thread) => thread,
                Err(_) => {
                    let _ = ready.send(Err(LoadError::Io));
                    return;
                }
            };
            let _ = ready.send(Ok(handle));
            e.main_loop();
            e.sh.watch.stopped.store(true, Ordering::Release);
            let failed = e.failed();
            if failed {
                // A failed device's destructors can wait forever. Retain its resources until exit.
                std::mem::forget(e);
            }
            watchdog.thread().unpark();
            let _ = watchdog.join();
            if let Some(sampler) = sampler {
                sampler.thread().unpark();
                // A dead GPU may never answer the NVML call the sampler is in: after a failure it
                // stays detached, and sees `stopped` if the call returns.
                if !failed {
                    let _ = sampler.join();
                }
            }
        }
        Err(err) => {
            let _ = ready.send(Err(err));
        }
    }
}

/// The CUDA archs whose graphs-on parity passed (`larch parity --mode decode --graphs on`, also
/// `--parallel 8`); `decode_graphs` defaults on only there, and only while it fits (`graphs_fit`).
// ponytail: sm_120 (5070 Ti), Oct 2026; sm_86 on the 3050 (#239). sm_89 by judgment, not measured: graph capture is
// arch-generic and parity passed on both neighbours; confirm with `larch parity --graphs on --parallel 8` on the 4060.
const GRAPH_DEFAULT_ARCHES: [u32; 3] = [86, 89, 120];

/// Steps between live state grows that found no room: each try reads the device's free memory.
// ponytail: fixed; key it to a request finishing if a 5th chat's wait shows in W4@5.
const GROW_RETRY_STEPS: u64 = 256;

/// The most recurrent state slots a hybrid model gets for `seats` seats, on every device; the load
/// gives as many as fit up to it, a live grow adds more up to it.
// ponytail: 8 is the agent's widest fan-out, also on a 24/32 GB card with no host layers, which is
// held to 8 seats. Size it from the fit when a bench on such a card shows the ceiling.
pub(crate) fn hybrid_slots(seats: u32) -> u32 {
    seats.min(8)
}

/// Whether a hand-set `max_batch_seqs` (`want` slots, [`hybrid_slots`]) re-plans a load of a
/// hybrid (`slots` state slots) whose experts partly sit in host memory (`in_ram`), planned for
/// `planned` slots: when it asks for more slots than the load holds, or for fewer than its host
/// layers pay for. Elsewhere the slots grow live from free memory ([`Engine::grow_seats`]) or cost
/// no host layer.
/// Where the loader picks the expert cache by seats (`flips`: CUDA, `LARCH_EXPERT_CACHE`
/// unset), many chats ([`MANY_SEATS`] or more) also flip it, a plan of its own.
fn replan_due(want: u32, slots: u32, in_ram: bool, planned: Option<u32>, flips: bool) -> bool {
    let many = |n| n >= MANY_SEATS;
    slots > 0
        && in_ram
        && planned.is_some_and(|p| want > slots || want < p || (flips && many(want) != many(p)))
}

/// The state slots a hybrid's many-chats plan pays for, with the expert cache (the loader's
/// `cache`); fast mode plans for one. 5070 Ti, 35B, 11 Nov, 3 alternating rounds: 4 chats 169
/// against 126 tok/s (TTFT p50 264 against 811 ms), 1 chat 66 against 98.
pub(crate) const MANY_SEATS: u32 = 4;

/// Requests waiting on full seats this long switch a fast plan to many chats.
const DEMAND_UP_NS: u64 = 1_000_000_000;

/// At most one chat this long switches many chats back to fast: a switch is a reload, so rare.
/// A cancelled switch waits as long before demand asks again.
const DEMAND_DOWN_NS: u64 = 120_000_000_000;

/// The longest a switch holds the queue for started requests to end; past it the switch is off.
const SWITCH_HOLD_NS: u64 = 5_000_000_000;

/// A demand switch waiting for the started requests: the seats it replaced, and its deadline.
#[derive(Clone, Copy, Debug)]
struct Switch {
    from: u32,
    by: u64,
}

/// Picks a hybrid's planned seats from demand, with hysteresis.
// ponytail: fixed thresholds from one card; learn them from switch cost (reload ms, re-read
// tokens) if a second card's bench shows switches that do not pay.
#[derive(Default)]
struct Demand {
    many: bool,
    since: Option<u64>,
    /// A cancelled switch: no new one before this (ns).
    quiet_until: u64,
}

impl Demand {
    /// The seats to re-plan for once demand at `now` (ns) held past its threshold: `running`
    /// requests on `slots` seats and `waiting` behind them. It flips only once a switch loads.
    fn seats(&mut self, now: u64, running: usize, waiting: usize, slots: u32) -> Option<u32> {
        let (pressed, hold) = match self.many {
            // Two waiting on full seats, or as many chats as MANY_SEATS on fewer slots (a load the check
            // left at 3): there, 4 chats never queue two.
            false => {
                let full = running >= slots as usize;
                let many = slots < MANY_SEATS && running + waiting >= MANY_SEATS as usize;
                (full && (waiting >= 2 || many), DEMAND_UP_NS)
            }
            true => (running + waiting <= 1, DEMAND_DOWN_NS),
        };
        if !pressed || now < self.quiet_until {
            self.since = None;
            return None;
        }
        let since = *self.since.get_or_insert(now);
        (now.saturating_sub(since) >= hold).then_some(if self.many { 1 } else { MANY_SEATS })
    }

    /// A switch to `seats` loaded: demand now weighs the other way.
    fn switched(&mut self, seats: u32) {
        (self.many, self.since) = (seats >= MANY_SEATS, None);
    }

    /// A switch did not run at `now`: demand waits [`DEMAND_DOWN_NS`] before it asks again.
    fn cancelled(&mut self, now: u64) {
        (self.since, self.quiet_until) = (None, now.saturating_add(DEMAND_DOWN_NS));
    }
}

/// The seats decode graphs can serve: `seats`, at most a recurrent model's `slots` (0: no state).
fn servable_seats(slots: u32, seats: u32) -> u32 {
    if slots == 0 { seats } else { seats.min(slots) }
}

/// Share of the graphs-off KV pool the arch default must keep for graphs to stay on.
// ponytail: from two cards. 4 GB 3050 (#239): m3 1 chat +64 % with graphs, m6 c8 -10.8 % as the
// pool lost a third of its seats (0.67): off. 5070 Ti, the hybrid MoE with 8 host-expert layers
// (Nov 2026): graphs 2.2x at 1 chat, +24 % at 4 chats (max_context 2304, pp 512 tg 128), its pool
// keeping 0.686-0.697 of its blocks with 4 MiB graph slots: on. Re-derive when
// LARCH_GRAPH_DIAGNOSTICS shows a card near the line.
const GRAPH_FIT_MIN_KV: f64 = 0.68;

/// `decode_graphs` before overrides: on where the device captures graphs and its arch is measured
/// or its card profile proved them (`profiled`, `larch routes-check`).
fn graphs_default(backend: Backend, arch_code: u32, graph_replay: bool, profiled: bool) -> f64 {
    let measured = GRAPH_DEFAULT_ARCHES.contains(&arch_code) || profiled;
    f64::from(u8::from(
        backend == Backend::Cuda && graph_replay && measured,
    ))
}

/// The default is fit-gated: graphs are on by `graphs_default`, not set by hand.
fn graphs_fit_gated(cfg: &EffectiveConfig, explicit: bool) -> bool {
    !explicit && cfg.values[knob_id::DECODE_GRAPHS as usize] != 0.0
}

/// `LARCH_GRAPH_DIAGNOSTICS=1`: print the load's graph fit and each capture's measured size.
fn graph_diagnostics() -> bool {
    std::env::var("LARCH_GRAPH_DIAGNOSTICS").is_ok_and(|v| v == "1")
}

/// Whether a context the engine raised for a projector (above the caller's `asked`) should be
/// halved because it fits only without decode graphs (`decode_graphs` on, and the plan kept none:
/// `with_graphs` false when the gate turned them off or none were reserved). One full context comes
/// before the graphs in the fit, so a raised context can take their room: Qwen3.5-2B on a 4 GB card
/// fit 32k without graphs and decoded at 38 tok/s, where 16k with them gave 68.
fn trade_context_for_graphs(
    graphs_on: bool,
    with_graphs: bool,
    projector: bool,
    context: u32,
    asked: u32,
) -> bool {
    graphs_on && !with_graphs && projector && context > asked.max(crate::VISION_MIN_CONTEXT)
}

/// Keep fit-gated graphs only if the pool keeps `GRAPH_FIT_MIN_KV` of its blocks without them.
/// `set_aside`: device bytes the load gave something beside the text model (a vision projector). They
/// count as pool on both sides, so the rule weighs graphs against the pool the text model alone would
/// have, and loading a projector never turns graphs off (on a 4 GB 3050 it cost InternVL3-1B its
/// graphs, 151 → 55 tok/s, with 81k tokens of pool left).
fn graphs_fit(with: &FitPlan, without: &FitPlan, set_aside: u64) -> bool {
    let extra = (set_aside / without.kv_block_bytes.max(1)) as f64;
    f64::from(with.kv_blocks) + extra >= GRAPH_FIT_MIN_KV * (f64::from(without.kv_blocks) + extra)
}

/// The plan for a load and whether decode graphs stay: a gated default keeps them only if
/// [`keep_graphs`] without the streaming pool, which comes after the graphs and must not turn them
/// off; the pool then joins the plan if it too leaves the KV pool that floor, else stays out.
/// `waive_floor` (an operator's `LARCH_PREFILL_STREAM_ROWS`, or the hybrid MoE target, whose KV
/// floor already holds each seat's measured peak) admits the pool wherever [`fit`] does.
/// `set_aside`: the projector's bytes ([`graphs_fit`]).
fn gated_plan(
    plan_for: impl Fn(bool, u64) -> Result<FitPlan, LoadError>,
    mut graphs: bool,
    gated: bool,
    paid: bool,
    stream_want: u64,
    waive_floor: bool,
    set_aside: u64,
) -> Result<(FitPlan, bool), LoadError> {
    let mut plan = plan_for(graphs, 0);
    if gated {
        let without = plan_for(false, 0)?;
        let kept = keep_graphs(paid, &plan, &without, set_aside);
        if graph_diagnostics() {
            let with = plan.as_ref().map_or(0, |p| p.kv_blocks);
            eprintln!("larch_graph_fit,{with},{},{kept}", without.kv_blocks);
        }
        if !kept {
            graphs = false;
            plan = Ok(without);
        }
    }
    plan?;
    let with = plan_for(graphs, stream_want)?;
    if waive_floor || graphs_fit(&with, &plan_for(false, 0)?, set_aside) {
        return Ok((with, graphs));
    }
    Ok((plan_for(graphs, 0)?, graphs))
}

/// Whether fit-gated graphs stay on: `with` fitted and either the load's plan already moved layers
/// to leave their reserve (`paid`: dropping them would keep that cost and lose the graphs) or the
/// pool keeps its share (`set_aside` as in [`graphs_fit`]).
fn keep_graphs(
    paid: bool,
    with: &Result<FitPlan, LoadError>,
    without: &FitPlan,
    set_aside: u64,
) -> bool {
    with.as_ref()
        .is_ok_and(|with| paid || graphs_fit(with, without, set_aside))
}

fn startup_config(
    opts: &mut EngineOptions,
    backend: Backend,
    graph_replay: bool,
    arch_code: u32,
    profiled_graphs: bool,
) -> (EffectiveConfig, Vec<(u16, f64, RejectReason)>) {
    // Startup values: the defaults and the options, then the `--set` overrides, each checked as a
    // live change would be. The ModelReload knobs go to the options; a rejected override is
    // recorded in the ring below and kept for `EngineHandle::rejected_overrides`.
    let mut cfg = EffectiveConfig::defaults(REGISTRY);
    if opts.telemetry == TelemetryTier::T0 {
        cfg.values[knob_id::TELEMETRY_TIER as usize] = 0.0;
    }
    // Explicit overrides follow and may enable graphs on any arch that captures them.
    cfg.values[knob_id::DECODE_GRAPHS as usize] =
        graphs_default(backend, arch_code, graph_replay, profiled_graphs);
    let mut rejected = Vec::new();
    for (k, v) in std::mem::take(&mut opts.knob_overrides) {
        let req = KnobRequest::new(k, v, Authority::Human);
        match validate(&req, backend, HealthState::Ready) {
            Ok(_) if k == knob_id::DECODE_GRAPHS && v != 0.0 && !graph_replay => {
                rejected.push((k, v, RejectReason::Unsupported));
            }
            Ok(_) => cfg.values[k as usize] = v,
            Err(RejectReason::NotLive { .. }) if k == knob_id::MAX_CONTEXT => {
                opts.max_context = v as u32;
            }
            Err(RejectReason::NotLive { .. }) if k == knob_id::KV_POOL_FRACTION => {
                opts.kv_pool_fraction = v as f32;
            }
            Err(reason) => rejected.push((k, v, reason)),
        }
    }
    (cfg, rejected)
}

impl Loaded {
    fn load(
        opts: &mut EngineOptions,
        cfg: &mut EffectiveConfig,
        load: Loader,
        weights: Option<Weights>,
        s: &Stream,
        chunk_cap: usize,
        partial: &mut Partial,
    ) -> Result<Self, LoadError> {
        let dev = s.device();
        let props = crate::device_props(dev).map_err(backend_err)?;
        // Scratch for the registry's largest batch, so max_batch_seqs can rise live; prefill rows
        // cost activations per token, so their ceiling is `chunk_cap`.
        let seqs_cap = knob_spec(knob_id::MAX_BATCH_SEQS).expect("registered").max as usize;
        let rows = seqs_cap + chunk_cap;
        let startup_seqs = cfg.values[knob_id::MAX_BATCH_SEQS as usize] as u32;
        let draft_seats = crate::guess::draft_seats(startup_seqs);
        if let Some(w) = &weights {
            *w.budget.lock().expect("load budget") = (opts.max_context, opts.kv_pool_fraction);
            // A hybrid's state slot costs 64 MB on the hybrid MoE target; 16 of them keep it off a 24 GB Mac's
            // GPU and cost a 16 GB card more host layers. The loader gives what fits up to this cap;
            // requests past the slots wait (`limits`) or grow them live (`grow_seats`).
            w.seqs.store(hybrid_slots(startup_seqs), Ordering::Relaxed);
            let asked = match partial.seats_asked {
                true => startup_seqs,
                false => partial.seats_auto.max(1),
            };
            w.asked.store(hybrid_slots(asked), Ordering::Relaxed);
            // The fit-gated default reserves too: its gate compares this load's fits with and
            // without graphs, which shows their real KV cost only when the load left them their
            // reserve (else graphs pass the gate on 0 bytes and never capture); `without` gets
            // that room back as KV blocks.
            let graphs = cfg.values[knob_id::DECODE_GRAPHS as usize] != 0.0;
            w.graph_seats
                .store(if graphs { startup_seqs } else { 0 }, Ordering::Relaxed);
        }
        let stat = || weights.as_ref().and_then(|_| file_id(&opts.model));
        let file = stat();
        // The vision ops (layernorm, gelu, attn_full, rope_multi) exist on the CPU reference, CUDA and
        // Metal: a projector on HIP is refused here, not at the first image's step.
        // ponytail: lift for HIP when it has the four ops.
        if opts.mmproj.is_some()
            && !matches!(
                props.backend,
                KBackend::Cpu | KBackend::Cuda | KBackend::Metal
            )
        {
            return Err(LoadError::Backend(KError::Unsupported.code()));
        }
        // Before the model, so the model's own fit (which experts stay on the GPU) leaves the projector
        // its room, and before the pool's, so its memory is not the KV's. Its width is checked
        // against the model's below.
        let before_vision = crate::device_props(dev).map_err(backend_err)?.vram_free;
        let vision = match &opts.mmproj {
            None => None,
            Some(path) => {
                let width = Projector::width(path)?;
                let model = Projector::load(path, dev, s, width)?;
                let config = model.config();
                // CUDA's attn_full is 64 wide only: refused here, so the app loads the model alone,
                // not at the warm-up's encode, which would fault the engine.
                if let ProjectorConfig::Qwen(c) = config
                    && props.backend == KBackend::Cuda
                    && c.n_embd / c.n_head != 64
                {
                    return Err(LoadError::Backend(KError::Unsupported.code()));
                }
                let row = u64::from(width) * 4;
                let unit = u64::from(config.max_unit_rows()) * row;
                let slots = match config {
                    ProjectorConfig::InternVl(_) => TILE_SLOTS,
                    ProjectorConfig::Qwen(_) => IMAGE_SLOTS,
                };
                Some(VisionRt {
                    tiles: alloc(dev, slots as u64 * unit, partial.startup)?,
                    slots: vec![None; slots],
                    rows: alloc(dev, rows as u64 * row, partial.startup)?,
                    model,
                    config,
                    clock: 0,
                    job: None,
                    layer_row_ns: None,
                    timing: None,
                })
            }
        };
        let vision_bytes =
            before_vision.saturating_sub(crate::device_props(dev).map_err(backend_err)?.vram_free);
        partial.model = Some(load(dev, s, rows as u32)?);
        // A drafter's plan may have taken fewer rows and draft seats (`larch_drafter_fit`).
        let planned = weights
            .as_ref()
            .map(|w| *w.ledger.lock().expect("load ledger"));
        let (rows, draft_seats) = match planned {
            Some(l) if l.rows != 0 => (l.rows as usize, l.draft_seats as usize),
            _ => (rows, draft_seats),
        };
        // An automatic drafter serves only where the load kept it (`larch_drafter_auto`).
        if let (Some(auto), Some(l)) = (opts.auto_drafter().cloned(), planned) {
            opts.drafter = l.auto.is_some_and(|a| a.2).then_some(auto);
        }
        let chunk_cap = rows - seqs_cap;
        let chunk = &mut cfg.values[knob_id::PREFILL_CHUNK_TOKENS as usize];
        *chunk = chunk.min(chunk_cap as f64);
        // A file rewritten while it loaded ties nothing: the bytes read may be either version.
        let file = file.filter(|&f| stat() == Some(f));
        let mc = partial
            .model
            .as_ref()
            .expect("loaded model")
            .config()
            .clone();
        // The drafter's weights count as the model's: in what the load took.
        let drafter = match &opts.drafter {
            Some(path) if !opts.mtp() => {
                let d = DflashDrafter::load(path, dev, s)?;
                let lends = partial
                    .model
                    .as_ref()
                    .is_some_and(|m| m.embed_and_head().is_some());
                if !d.fits(&mc) || !lends {
                    return Err(LoadError::UnsupportedArch);
                }
                Some(d)
            }
            _ => None,
        };
        let checks = check_spares(opts, drafter.as_ref().map(DflashDrafter::block));
        if checks > 0 {
            let model = partial.model.as_mut().expect("loaded model");
            model.alloc_checks(s, checks)?;
        }
        if vision
            .as_ref()
            .is_some_and(|v| v.config.proj_dim() != mc.d_model)
        {
            return Err(LoadError::BadGguf(larch_model::gguf::code::MISMATCH, 0));
        }
        let free = crate::device_props(dev).map_err(backend_err)?.vram_free; // what the model left
        let seats = servable_seats(
            partial.model.as_ref().expect("loaded model").state_slots(),
            startup_seqs,
        );
        let moe = weights.as_ref().and_then(|w| w.moe);
        // The load maps the file when no expert layer went to host memory.
        let mapped = weights
            .as_ref()
            .is_some_and(|w| w.host_layers.load(Ordering::Relaxed) == 0);
        let (stream_want, stream_rows) = partial
            .model
            .as_ref()
            .map_or((0, 0), |m| m.stream_pool_want());
        let graphs_on = cfg.values[knob_id::DECODE_GRAPHS as usize] != 0.0;
        let paid = weights
            .as_ref()
            .is_some_and(|w| w.ledger.lock().expect("load ledger").graphs_paid);
        let waive_floor = matches!(mc.arch, larch_model::Arch::Qwen35Moe)
            || larch_kernels::routes()
                .prefill_stream_rows
                .is_some_and(|rows| rows != 0);
        // With a projector the context is a ceiling, at least VISION_MAX_CONTEXT: halved here, down
        // to VISION_MIN_CONTEXT, until the KV pool holds one, instead of a refusal the caller can only
        // meet by reading the weights again at half (InternVL3.5's 112 KiB a token fit a 4 GB card at
        // 8k). `opts` keeps the caller's context, so a reload that drops the projector asks for it.
        // Qwen 3.5's small KV asks for more (QWEN_VISION_MAX_CONTEXT).
        let mut max_context = match (&opts.mmproj, &vision) {
            (Some(_), Some(v)) => opts.max_context.max(crate::vision_context(&v.config)),
            (Some(_), None) => opts.max_context.max(crate::VISION_MAX_CONTEXT),
            (None, _) => opts.max_context,
        }
        .min(mc.ctx_train);
        // The largest context that fit only without decode graphs, kept while smaller ones are tried.
        let mut first_fit = None;
        let (dims, [ib, lb, sb], scratch, plan, graphs) = loop {
            let dims = Dims::of(&mc, rows, max_context);
            let bytes = dims.bytes(seqs_cap);
            // A check's rows are logits rows, within the `seqs_cap` rows the step buffers start with.
            let scratch = serving_scratch(&props, &mc, moe, rows as u32, max_context)
                + (drafter.as_ref()).map_or(0, |d| {
                    drafter_bytes(d.shape(), mc.d_model, rows, max_context, draft_seats)
                })
                + if opts.mtp() {
                    mtp_bytes(&mc, rows, max_context, draft_seats)
                } else {
                    0
                };
            let plan_for = |graphs, stream_want| {
                fit(
                    &props,
                    free,
                    &mc,
                    max_context,
                    opts.kv_pool_fraction,
                    scratch,
                    seats,
                    graphs,
                    s.capture_arena_held(),
                    mapped,
                    stream_want,
                    checks,
                )
            };
            let planned = gated_plan(
                plan_for,
                graphs_on,
                partial.graphs_fit_gated,
                paid,
                stream_want,
                waive_floor,
                vision_bytes,
            );
            if std::env::var("LARCH_FIT_DIAGNOSTICS").is_ok_and(|v| v == "1") {
                // Each context the fit tries: what was free after the model, the serving scratch,
                // the projector's bytes, and the KV blocks it gave or the bytes it lacked.
                let outcome = match &planned {
                    Ok((p, g)) => format!(
                        "kv_blocks={},graph_bytes={},graphs={g}",
                        p.kv_blocks, p.graph_bytes
                    ),
                    Err(LoadError::DoesNotFit { need, have }) => format!("need={need},have={have}"),
                    Err(e) => format!("{e:?}"),
                };
                eprintln!(
                    "larch_context_fit,{max_context},free={free},scratch={scratch},vision={vision_bytes},{outcome}"
                );
            }
            let projector = opts.mmproj.is_some();
            match planned {
                Err(LoadError::DoesNotFit { .. })
                    if first_fit.is_none()
                        && projector
                        && max_context > crate::VISION_MIN_CONTEXT =>
                {
                    max_context = (max_context / 2).max(crate::VISION_MIN_CONTEXT);
                }
                Ok((p, kept))
                    if trade_context_for_graphs(
                        graphs_on,
                        kept && p.graph_bytes > 0,
                        projector,
                        max_context,
                        opts.max_context,
                    ) =>
                {
                    first_fit.get_or_insert((max_context, (dims, bytes, scratch, p, kept)));
                    max_context = (max_context / 2).max(crate::VISION_MIN_CONTEXT);
                }
                Ok((p, kept)) if !graphs_on || (kept && p.graph_bytes > 0) => {
                    break (dims, bytes, scratch, p, kept);
                }
                // No smaller context got the graphs: the largest that fit, as before.
                planned => match first_fit.take() {
                    Some((context, chosen)) => {
                        max_context = context;
                        break chosen;
                    }
                    None => {
                        let (p, kept) = planned?;
                        break (dims, bytes, scratch, p, kept);
                    }
                },
            }
        };
        if !graphs {
            cfg.values[knob_id::DECODE_GRAPHS as usize] = 0.0;
        }
        if std::env::var("LARCH_FIT_DIAGNOSTICS").is_ok_and(|v| v == "1")
            && let Some(w) = &weights
        {
            let l = *w.ledger.lock().expect("load ledger");
            eprintln!(
                "larch_fit_ledger,free_before={},host_layers={},weights_on_device={},activation={},routers={},hybrid={},reserve_scratch={},reserve_graph_slots={},reserve_graph_arena={},graphs_paid={},fit_scratch={},headroom={},free_after_load={},model_bytes={},kv_blocks={},kv_block_bytes={},graph_quota={},graph_arena={}",
                l.free_before,
                w.host_layers.load(Ordering::Relaxed),
                l.weights_on_device,
                l.activation,
                l.routers,
                l.hybrid,
                l.scratch,
                l.graph_slots,
                l.graph_arena,
                l.graphs_paid,
                scratch,
                plan.headroom_bytes,
                free,
                plan.model_bytes,
                plan.kv_blocks,
                plan.kv_block_bytes,
                plan.graph_bytes,
                plan.graph_arena_bytes,
            );
        }
        cfg.values[knob_id::MAX_CONTEXT as usize] = f64::from(max_context);
        let fraction = &mut cfg.values[knob_id::KV_POOL_FRACTION as usize];
        if *fraction as f32 != opts.kv_pool_fraction {
            *fraction = f64::from(opts.kv_pool_fraction); // else keep the registry's exact f64
        }
        cfg.values[knob_id::DEVICE as usize] = f64::from(opts.device);
        let half = u64::from(plan.kv_blocks) * plan.kv_block_bytes / 2;
        for bytes in [half, half, ib, lb, sb] {
            partial.buffers.push(alloc(dev, bytes, partial.startup)?);
        }
        let oom = |e| match e {
            KError::Oom => LoadError::DoesNotFit {
                need: plan.model_bytes + scratch,
                have: props.vram_free,
            },
            e => backend_err(e),
        };
        let model = partial.model.as_deref().expect("loaded model");
        let draft = match drafter {
            Some(d) => Some(
                Draft::dflash(d, dev, s, mc.d_model, rows, max_context, draft_seats)
                    .map_err(oom)?,
            ),
            // The model loads its MTP block with this option; one without the block refuses.
            None if opts.mtp() => {
                let t = model.transformer().ok_or(LoadError::UnsupportedArch)?;
                let d = MtpDrafter::new(t, s, rows, max_context).map_err(oom)?;
                let d = d.ok_or(LoadError::UnsupportedArch)?;
                Some(Draft::mtp(d, t, s, draft_seats).map_err(oom)?)
            }
            None => None,
        };
        // After the mandatory KV and serving buffers, so the optional pool can never refuse a load.
        // ponytail: a pool that fails here leaves its reserved bytes out of the KV pool; fold them
        // back when a load shows pool failures.
        let diagnostics = std::env::var("LARCH_FIT_DIAGNOSTICS").is_ok_and(|v| v == "1");
        let stream_pool = plan.stream_pool_bytes != 0
            && match partial
                .model
                .as_mut()
                .expect("loaded model")
                .alloc_stream_pool(s)
            {
                Ok(made) => made,
                // Out of memory only: no pool, the CPU path. Anything else is the device's.
                Err(KError::Oom) => {
                    if diagnostics {
                        eprintln!("larch_stream_pool_failed,oom");
                    }
                    false
                }
                Err(e) => return Err(backend_err(e)),
            };
        if diagnostics {
            eprintln!(
                "larch_stream_pool,{stream_want},{},{stream_pool},{stream_rows}",
                plan.stream_pool_bytes
            );
        }

        let backend = backend(props.backend);
        let cpu = backend == Backend::Cpu;
        let caps = Capabilities {
            backend,
            graph_replay: props.flags & DeviceProps::CAP_GRAPHS != 0,
            prefill_chunk_max: chunk_cap as u32,
            achieved_gbps: if weights.is_some() {
                MState::Estimated
            } else {
                MState::Missing
            },
            // The CPU reference has no device memory to be a fraction of.
            hbm_util: match (cpu, weights.is_some() && props.peak_gbps > 0.0) {
                (true, _) => MState::Unsupported,
                (false, true) => MState::Estimated,
                (false, false) => MState::Missing,
            },
            // ponytail: the occupancy query is not wired; Missing where the platform has it,
            // until it is.
            sm_occupancy: match backend {
                Backend::Cuda | Backend::Hip => MState::Missing,
                _ => MState::Unsupported,
            },
            // Measured once the engine has opened NVML on the device, which outlives a reload;
            // CUDA without it (no driver NVML, or two cards of one name) is Missing.
            device_sample: match backend {
                Backend::Cuda => MState::Missing,
                _ => MState::Unsupported,
            },
        };

        let deployment = DeploymentSnapshot {
            model: FixedStr::from_str_truncating(
                &opts
                    .model
                    .file_stem()
                    .map_or(String::new(), |s| s.to_string_lossy().into()),
            ),
            arch: FixedStr::from_str_truncating(&format!("{:?}", mc.arch).to_lowercase()),
            quant: FixedStr::from_str_truncating(weights.as_ref().map_or("", |w| &w.quant)),
            params_b: weights.as_ref().map_or(0.0, |w| w.params as f32 / 1e9),
            n_layers: mc.n_layers as u16,
            n_heads: mc.n_heads as u16,
            n_kv_heads: mc.n_kv_heads as u16,
            head_dim: mc.head_dim as u16,
            ctx_max: max_context,
            vocab: mc.vocab,
            backend: backend.raw(),
            device_name: FixedStr::from_str_truncating(&props.name),
            arch_code: props.arch_code,
            peak_gbps: if cpu {
                MF32::unsupported()
            } else if props.peak_gbps > 0.0 {
                MF32::estimated(props.peak_gbps)
            } else {
                MF32::missing()
            },
            // Without the header (test models): memory not free before the pool.
            weights_bytes: weights.as_ref().map_or(plan.model_bytes, |w| w.bytes),
            kv_pool_blocks: plan.kv_blocks,
            fit_max_seqs: plan.max_seqs_fit,
            prefill_chunk_max: chunk_cap as u32,
            ..bytemuck::Zeroable::zeroed()
        };
        let class = ModelClass {
            dense: weights.as_ref().is_none_or(|w| w.experts.is_empty()),
            active_bytes: weights
                .as_ref()
                .map_or(plan.model_bytes, |w| w.read_per_step),
            experts_in_ram: weights
                .as_ref()
                .is_some_and(|w| w.host_layers.load(Ordering::Relaxed) > 0),
        };
        let state_slots = partial.model.as_ref().map_or(0, |m| m.state_slots());
        let [k, v, input, logits, samp] = std::mem::take(&mut partial.buffers)
            .try_into()
            .ok()
            .expect("five serving buffers");
        let seats_planned = (weights.as_ref())
            .filter(|_| opts.cpu_expert_layers.is_none())
            .map(|w| w.asked.load(Ordering::Relaxed));
        let free_before = before_vision.saturating_sub(vision_bytes);
        Ok(Self {
            seats_planned,
            file,
            free_before,
            host_layers: (weights.as_ref()).map_or(0, |w| w.host_layers.load(Ordering::Relaxed)),
            streams: stream_pool,
            graphs: HashMap::new(),
            check_used: HashMap::new(),
            mixed_graphs: larch_kernels::routes().mixed_graphs,
            headroom_bytes: plan.headroom_bytes,
            graph_bytes: plan.graph_bytes,
            graph_arena_bytes: plan.graph_arena_bytes,
            model: partial.model.take().expect("loaded model"),
            k,
            v,
            input,
            logits,
            samp,
            staging: Staging::default(),
            dims,
            logit_rows: seqs_cap,
            weights_read: weights.map(|w| w.read_per_step),
            kv_token_bytes: plan.kv_block_bytes / BLOCK_TOKENS as u64,
            info: Published {
                deployment,
                caps,
                class,
                vision: vision.as_ref().map(|v| v.config),
                state_slots,
                free_before,
            },
            draft,
            vision,
        })
    }
}

impl Engine {
    pub(crate) fn init(
        mut opts: EngineOptions,
        load: Loader,
        weights: Option<Weights>,
    ) -> Result<(Engine, EngineHandle), LoadError> {
        // A documents plan at load: no drafter now; the context, fraction and seats after the
        // startup knobs (below), as its reload sets them over the person's. What the options named
        // is the plan to go back to (`docs_before`).
        let docs = opts.documents.take();
        let drafter = docs.map(|_| (opts.drafter.take(), opts.drafter_auto.take()));
        // Bad options fail here, not as a panic in ring(); fit rejects max_context 0. NaN is not in
        // the registry range. The person's fraction too, which a documents plan's restore brings
        // back; and the plan's own values, which a reload would check as knobs.
        let fraction = knob_spec(knob_id::KV_POOL_FRACTION).expect("registered");
        let in_range = |id, v: f64| knob_spec(id).is_some_and(|k| (k.min..=k.max).contains(&v));
        if !(fraction.min as f32..=fraction.max as f32).contains(&opts.kv_pool_fraction)
            || opts.ring_capacity < 2
            || !opts.ring_capacity.is_power_of_two()
            || docs.is_some_and(|p| {
                !in_range(knob_id::MAX_CONTEXT, f64::from(p.max_context))
                    || !in_range(knob_id::MAX_BATCH_SEQS, f64::from(p.seats))
            })
        {
            return Err(LoadError::Backend(KError::InvalidArg.code()));
        }
        let epoch = Instant::now();
        let dev = Device::open(opts.device).map_err(backend_err)?;
        let s = Stream::new(&dev).map_err(backend_err)?;
        let props = dev.props().map_err(backend_err)?;
        let backend = backend(props.backend);
        let graph_replay = props.flags & DeviceProps::CAP_GRAPHS != 0;
        let explicit = |knob| opts.knob_overrides.iter().any(|&(k, _)| k == knob);
        let (explicit_graphs, seats_asked) = (
            explicit(knob_id::DECODE_GRAPHS),
            explicit(knob_id::MAX_BATCH_SEQS),
        );
        let profiled =
            larch_model::card::routes(&props, &opts.model).contains(&Route::DecodeGraphs);
        let (mut cfg, rejected) =
            startup_config(&mut opts, backend, graph_replay, props.arch_code, profiled);
        let seats_knob = knob_id::MAX_BATCH_SEQS as usize;
        let documents = docs.is_some();
        let docs_before = docs.zip(drafter).map(|(plan, drafter)| {
            let before = DocsBefore {
                max_context: opts.max_context,
                kv_pool_fraction: opts.kv_pool_fraction,
                seats: (cfg.values[seats_knob], seats_asked),
                drafter,
                set: DocsSet {
                    max_context: Some(plan.max_context),
                    kv_pool_fraction: Some(crate::DOC_POOL_FRACTION),
                    seats: Some(f64::from(plan.seats)),
                    drafter_off: true,
                    auto_off: true,
                },
            };
            opts.max_context = plan.max_context;
            opts.kv_pool_fraction = crate::DOC_POOL_FRACTION;
            // As a hand-set knob, as the reload sets it: the plan pays its state slots.
            cfg.values[seats_knob] = f64::from(plan.seats);
            before
        });

        let chunk = cfg.values[knob_id::PREFILL_CHUNK_TOKENS as usize] as usize;
        let chunk_cap = match weights.as_ref().is_some_and(|w| w.moe.is_some()) {
            true => chunk,
            false => chunk.max(PREFILL_CHUNK_SCRATCH),
        };
        let device = if backend == Backend::Cuda {
            Nvml::open(&props.name)
        } else {
            None
        };
        let mut partial = Partial {
            startup: true,
            graphs_fit_gated: graphs_fit_gated(&cfg, explicit_graphs),
            seats_asked: seats_asked || documents,
            ..Partial::default()
        };
        let cache_compatible = weights.is_some() || opts.prefix_cache_custom_compatible;
        let mut loaded = Loaded::load(
            &mut opts,
            &mut cfg,
            load,
            weights,
            &s,
            chunk_cap,
            &mut partial,
        )?;
        if documents && !direct_documents_fit(loaded.host_layers, loaded.streams) {
            // As `free_before` against itself and one byte more: the plan does not fit as a
            // documents reload would plan it, which the caller's own load and the pin then make.
            let have = loaded.free_before;
            return Err(LoadError::DoesNotFit {
                need: have + 1,
                have,
            });
        }
        if device.is_some() {
            loaded.info.caps.device_sample = MState::Measured;
        }
        let Published {
            deployment, caps, ..
        } = loaded.info;
        let (ring, ring_handle) = larch_records::ring(opts.ring_capacity);
        let health = Health {
            state: HealthState::Ready,
            since_mono_ns: crate::mono_ns(epoch, opts.clock.as_deref()),
            last_fault: FaultCode::None,
            restarts: 0,
        };
        let engine_id = wall_ns() as u64 ^ u64::from(std::process::id()) << 32;
        let (tx, rx) = channel();
        let sh = Arc::new(Shared {
            health: Mutex::new(health),
            effective: RwLock::new(Arc::new(cfg)),
            info: RwLock::new(loaded.info),
            ring: ring_handle,
            host: Mutex::new(crate::lifecycle::Host::new(
                ring,
                RecordHeader {
                    engine_id,
                    device: opts.device as i16,
                    ..bytemuck::Zeroable::zeroed()
                },
                opts.telemetry != TelemetryTier::Off,
            )),
            watch: crate::lifecycle::Watch::new(step_floor(&opts, backend, loaded.info.class)),
            inbox: Mutex::new(Some(rx)),
            epoch,
            clock: opts.clock.clone(),
            shutdown: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            next_id: AtomicU64::new(1),
            in_flight: AtomicUsize::new(0),
            sched_waiting: AtomicUsize::new(0),
            cold_pins: Mutex::new(Default::default()),
            tiles_encoded: AtomicU64::new(0),
            encode_slices: AtomicU64::new(0),
            switching: AtomicBool::new(false),
            pending_knobs: AtomicUsize::new(0),
            rejected: AtomicU64::new(0),
            eager_decode: AtomicU64::new(0),
            pending_rejects: AtomicUsize::new(0),
            thread: Mutex::new(None),
            rejected_overrides: rejected.clone(),
            on_fault: opts.on_fault,
            engine_id,
            device,
            adapter_version: AtomicU64::new(0),
            guesses: Mutex::new(GuessStats::default()),
        });
        let state_slots = loaded.model.state_slots();
        let mut e = Engine {
            sh: sh.clone(),
            s,
            base_host_layers: loaded.host_layers,
            base_free: loaded.free_before,
            loaded: Some(loaded),
            // A documents plan's seats stay asked through later rebuilds, as after its reload.
            partial: Partial {
                seats_asked: documents,
                ..Partial::default()
            },
            reload: None,
            sched: Scheduler::new(deployment.kv_pool_blocks),
            cfg,
            history: Vec::new(),
            step: 0,
            engine_id,
            ab: None,
            guard: Guard::default(),
            graphs_before_custom: cfg.values[knob_id::DECODE_GRAPHS as usize],
            graphs_forced_off: false,
            top_version: 0,
            swap: None,
            job: None,
            prepare: VecDeque::new(),
            grow_at: 0,
            replan: false,
            docs_before,
            restoring: false,
            idle_since: Instant::now(),
            imports: HashMap::new(),
            torn: VecDeque::new(),
            doc_terms: None,
            auto: !seats_asked && !documents && std::env::var_os("LARCH_EXPERT_CACHE").is_none(),
            demand: Demand::default(),
            switch: None,
            opts,
        };
        e.warm_up();
        e.sched.set_state_slots(state_slots);
        e.sched.pace_decode(backend == Backend::Metal);
        e.sched.pin_mark = e.opts.pin_mark.unwrap_or(crate::sched::PIN_MARK);
        e.sched.guess_max = guess_max(&e.opts, e.loaded.as_ref().expect("loaded engine"));
        e.sched.configure_prefix_cache(if cache_compatible {
            e.opts.prefix_cache_blocks
        } else {
            0
        });
        e.push(Tier::T0, &deployment);
        e.push_device(&caps);
        let up = HealthEvent {
            from: HealthState::Loading.raw(),
            to: HealthState::Ready.raw(),
            recovery_ms: crate::lifecycle::recovery_ms(
                std::env::var("LARCH_RESTARTED_AT").ok().as_deref(),
                crate::lifecycle::wall_ns(),
            ),
            ..bytemuck::Zeroable::zeroed()
        };
        e.push(Tier::T0, &up);
        for (knob, value, reason) in rejected {
            e.activation(knob, value, 0, 0, &KnobReply::Rejected { reason });
        }
        Ok((e, EngineHandle { shared: sh, tx }))
    }

    /// The DeviceSample after each DeploymentSnapshot, with `sample_tick` 0: it is not one of the
    /// sampler's (`lifecycle::sampler`). Read through NVML where the engine opened it; else, on
    /// CUDA, the VRAM fields from the device props (both Missing if that read fails) and the rest
    /// Missing, since the device has them; elsewhere every field Unsupported, at T0, so a T0 ring
    /// knows.
    fn push_device(&mut self, caps: &Capabilities) {
        let tier =
            if caps.backend == Backend::Cuda && self.value(knob_id::TELEMETRY_TIER) as u32 == 1 {
                Tier::T1
            } else {
                Tier::T0
            };
        let d = match &self.sh.device {
            Some(nvml) => crate::lifecycle::device_sample(nvml),
            None => {
                let (u, f) = if caps.backend == Backend::Cuda {
                    (MU64::missing(), MF32::missing())
                } else {
                    (MU64::unsupported(), MF32::unsupported())
                };
                // CUDA's props trim the memory pool, so without NVML VRAM is read at load only.
                let (used, total) =
                    match caps.backend {
                        Backend::Cuda => self.s.device().props().map_or(
                            (MU64::missing(), MU64::missing()),
                            |p| {
                                (
                                    MU64::measured(p.vram_total.saturating_sub(p.vram_free)),
                                    MU64::measured(p.vram_total),
                                )
                            },
                        ),
                        _ => (u, u),
                    };
                DeviceSample {
                    sm_clock_mhz: f,
                    mem_clock_mhz: f,
                    temp_c: f,
                    power_w: f,
                    gpu_util: f,
                    vram_used_bytes: used,
                    vram_total_bytes: total,
                    power_limit_w: f,
                    spill: f,
                    clocks_event_reasons: u,
                    other_procs: u,
                    sample_tick: 0,
                }
            }
        };
        self.push(tier, &d);
    }

    /// Writes a record under the A/B arm in force.
    fn push<R: RecordBody>(&mut self, tier: Tier, body: &R) {
        self.record(tier, self.cfg.ab_arm, body);
    }

    fn record<R: RecordBody>(&mut self, tier: Tier, ab_arm: u8, body: &R) {
        if self.opts.telemetry == TelemetryTier::Off {
            return;
        }
        let header = self.header(tier, ab_arm);
        self.sh
            .host
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .record(header, body);
    }

    fn header(&self, tier: Tier, ab_arm: u8) -> RecordHeader {
        RecordHeader {
            kind: 0,
            layout: 0,
            tier: tier.raw(),
            ab_arm,
            device: self.opts.device as i16,
            seq: 0,
            t_mono_ns: self.sh.now(),
            t_wall_ns: wall_ns(),
            engine_id: self.engine_id,
            config_rev: self.cfg.rev,
        }
    }

    /// Runs turns until a Shutdown, a dropped last handle or a fault; a panic anywhere in a turn is a
    /// fault (9.11).
    fn main_loop(&mut self) {
        let fault = loop {
            match catch_unwind(AssertUnwindSafe(|| self.turn())) {
                Ok(Ok(true)) => {}
                Ok(Ok(false)) => break None,
                Ok(Err(e)) => break Some(backend_fault(e.code())),
                Err(_) => break Some((FaultCode::Panic, EngineError::Internal(0))),
            }
        };
        if !self.failed() {
            self.stop(fault);
        }
    }

    fn failed(&self) -> bool {
        self.sh.failed.load(Ordering::Acquire)
    }

    /// Waits for work while idle, handles queued messages, then runs one step.
    fn turn(&mut self) -> Result<bool, KError> {
        if self.failed() {
            return Ok(false);
        }
        if !self.sched.is_idle() {
            self.idle_since = Instant::now();
        }
        if self.replan && self.sched.is_idle() && self.reload.is_none() {
            self.replan = false;
            self.replan_seats();
        }
        self.watch_demand(self.sh.now());
        if self.sched.is_idle()
            && self.reload.is_none()
            && self.swap.is_none()
            && self.job.is_none()
            && self.prepare.is_empty()
            && self.switch.is_none()
        {
            self.sh.watch.idle();
            let message = self
                .sh
                .inbox
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .map_or(Err(RecvTimeoutError::Disconnected), |rx| {
                    rx.recv_timeout(crate::lifecycle::IDLE_WAIT)
                });
            match message {
                Ok(m) => {
                    self.sh.watch.begin(self.sh.now());
                    if !self.handle(m)? {
                        return Ok(false);
                    }
                }
                // A restore that is due runs below, after any pin that arrived has been seen.
                Err(RecvTimeoutError::Timeout) if !self.restore_ready() => return Ok(true),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Ok(false),
            }
        }
        loop {
            let message = self
                .sh
                .inbox
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .map_or(Err(TryRecvError::Disconnected), |rx| rx.try_recv());
            let Ok(m) = message else {
                break;
            };
            if !self.handle(m)? {
                return Ok(false);
            }
        }
        if self.failed() {
            return Ok(false);
        }
        if self.swap.is_some() {
            self.swap()?;
        }
        if self.job.is_some() {
            self.run_job();
        }
        self.watch_restore();
        let resume = if self.switch_ready() {
            self.switch_seats()
        } else if self.restore_ready() {
            self.restore_now()
        } else if self.reload.is_some() && self.sched.is_idle() {
            // Loading has the same unbounded host/device setup as startup, not a step deadline.
            self.sh.watch.idle();
            self.rebuild()
        } else {
            self.step()?;
            // Nothing to run: one tile of a prepared image, so a request that arrives meanwhile
            // waits for one tile at most.
            if self.sched.is_idle() && !self.prepare.is_empty() && !self.failed() {
                self.sh.watch.begin(self.sh.now());
                self.prepare_tile()?;
            }
            true
        };
        self.sh.watch.complete(self.sh.now(), None);
        Ok(resume)
    }

    /// One slice of the encode of the first image unit a request has yet to read and the cache
    /// lacks (`Scheduler::wanted_unit`), or all of it with `whole`; nothing without a sliced
    /// projector or a wanted unit.
    fn encode_slice(&mut self, whole: bool) -> Result<(), KError> {
        if !self.sched.slice_images {
            return Ok(());
        }
        // A request's unit first; else, beside other requests, the newest pasted image (encode on
        // paste), so it is ready by the time its message is sent.
        let (digest, unit, px) = match self.sched.wanted_unit() {
            Some((id, k, unit)) => {
                let span = &self.sched.req(id).expect("wanted request").images[k];
                // Admission refused a span without pixels (`check_images`).
                let Some(px) = span.pixels.clone() else {
                    return Ok(());
                };
                (*span.image.digest(), unit, px)
            }
            // The paste whose encode is in flight first, so a newer paste does not settle it whole.
            None => {
                let flight = (self.loaded.as_ref())
                    .and_then(|l| l.vision.as_ref())
                    .and_then(|v| v.job.as_ref())
                    .map(|j| (j.0, j.1));
                let p = (self.prepare.iter())
                    .filter(|p| p.keep)
                    .find(|p| flight == Some((p.digest, p.next)))
                    .or_else(|| self.prepare.front().filter(|p| p.keep));
                match p {
                    Some(p) if !whole => (p.digest, p.next, p.pixels.clone()),
                    _ => return Ok(()),
                }
            }
        };
        let Some(loaded) = self.loaded.as_mut() else {
            return Ok(());
        };
        let d_model = loaded.model.config().d_model as usize;
        let Some(v) = loaded.vision.as_mut() else {
            return Ok(());
        };
        let rows = px.unit_rows(&v.config) as usize;
        let held = (v.slots.iter()).any(|e| e.is_some_and(|e| (e.0, e.1) == (digest, unit)));
        let done = held
            || match whole {
                true => {
                    v.slice(&self.s, digest, unit, rows, &px, d_model)?
                        || v.settle(&self.s, d_model)?
                }
                false => v.slice(&self.s, digest, unit, rows, &px, d_model)?,
            };
        if !held {
            self.sh.encode_slices.fetch_add(1, Ordering::Relaxed);
        }
        if done && !held {
            self.sh.tiles_encoded.fetch_add(1, Ordering::Relaxed);
        }
        // The unit that finished may be another one in flight (`VisionRt::slice`): every pasted image
        // whose next unit a slot now holds moves past it.
        if done {
            for p in self.prepare.iter_mut().filter(|p| p.keep) {
                while p.next < p.pixels.units()
                    && (v.slots.iter()).any(|e| e.is_some_and(|e| (e.0, e.1) == (p.digest, p.next)))
                {
                    p.next += 1;
                }
            }
            self.prepare.retain(|p| p.next < p.pixels.units());
        }
        Ok(())
    }

    /// Queues `pixels`, the image `digest` names cut for the encoder, to encode while idle, newest
    /// first. Past the cache's slots of units waiting, the oldest images go: the cache would not
    /// keep them. Nothing without a projector, or for pixels that are not its units.
    fn queue_prepare(&mut self, digest: [u8; 32], pixels: Arc<ImagePixels>) {
        // `prepare_image` checked the pixels; a reload since may have dropped or changed the
        // projector.
        let Some(v) = self.loaded.as_ref().and_then(|l| l.vision.as_ref()) else {
            return;
        };
        let slots = v.slots.len();
        if !pixels.fits(&v.config, pixels.unit_rows(&v.config) * pixels.units()) {
            return;
        }
        self.prepare.retain(|p| p.digest != digest || !p.keep);
        self.prepare.push_front(Prepared {
            digest,
            pixels,
            next: 0,
            keep: true,
        });
        let mut units = 0;
        self.prepare.retain(|p| {
            units += (p.pixels.units() - p.next) as usize;
            !p.keep || units <= slots
        });
    }

    /// Queues the warm-up: a blank input of a small screenshot's size through the projector
    /// (`ProjectorConfig::warm_up_input`), its slot freed after.
    fn warm_up(&mut self) {
        let Some(v) = self.loaded.as_ref().and_then(|l| l.vision.as_ref()) else {
            return;
        };
        self.prepare.push_back(Prepared {
            digest: [0; 32],
            pixels: Arc::new(v.config.warm_up_input()),
            next: 0,
            keep: false,
        });
    }

    /// Encodes the next unit of the first queued image into the unit cache (S5), unless a slot
    /// holds it already. A failure is a fault, as it is in a step.
    fn prepare_tile(&mut self) -> Result<(), KError> {
        let Some(p) = self.prepare.front_mut() else {
            return Ok(());
        };
        let (digest, tile, keep, pixels) = (p.digest, p.next, p.keep, p.pixels.clone());
        p.next += 1;
        if p.next >= p.pixels.units() {
            self.prepare.pop_front();
        }
        let Some(loaded) = self.loaded.as_mut() else {
            return Ok(());
        };
        let d_model = loaded.model.config().d_model as usize;
        let Some(v) = loaded.vision.as_mut() else {
            self.prepare.clear();
            return Ok(());
        };
        let rows = pixels.unit_rows(&v.config) as usize;
        let (slot, encoded) = v.tile_slot(&self.s, digest, tile, rows, || Ok(pixels), d_model)?;
        if !keep {
            v.slots[slot] = None;
        } else if encoded {
            self.sh.tiles_encoded.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Whether `tokens` could be pinned beside the pins held: they and a turn's room
    /// ([`crate::TURN_ROOM`]) must fit the pool, else `PINS_FULL`. A pin that later goes cold through a reload stays
    /// registered and is logged (`larch_pin_cold`); its next question reads the document again.
    fn pins_fit(&self, namespace: u64, tokens: &[u32]) -> Result<(), EngineError> {
        let pinned = self
            .sched
            .prefix
            .pinned_blocks_with(Some((namespace, tokens)));
        if pinned + self.sched.prefix.turn_blocks > self.sched.blocks.total() {
            return Err(EngineError::InvalidRequest(invalid::PINS_FULL));
        }
        Ok(())
    }

    /// Registers pin `id` ([`EngineHandle::pin`]) when the pool has room for it.
    fn pin_chain_of(
        &mut self,
        id: u64,
        namespace: u64,
        tokens: Vec<u32>,
    ) -> Result<(), EngineError> {
        self.pins_fit(namespace, &tokens)?;
        self.sched.prefix.pin(id, namespace, tokens);
        Ok(())
    }

    /// One step of a pin's export or import ([`PinOp`]); the pin's chain is looked up afresh each
    /// time, so a reload or an eviction between steps shows as `PIN_CHANGED`, never as other bytes.
    fn pin_op(&mut self, op: PinOp) -> Result<PinReply, EngineError> {
        let changed = || EngineError::InvalidRequest(invalid::PIN_CHANGED);
        let l = self
            .loaded
            .as_ref()
            .ok_or(EngineError::NotReady(HealthState::Loading))?;
        let key = carry_key(l).ok_or_else(changed)?;
        let ((layers, block), n_blocks) = (key.kv, self.sched.blocks.total() as u64);
        let internal = |e: KError| EngineError::Internal(e.code());
        let pinned = |e: &Self, id| {
            (e.sched.prefix.pins())
                .find(|p| p.0 == id)
                .map(|(_, ns, tokens)| (ns, tokens.to_vec()))
        };
        Ok(match op {
            PinOp::Info => PinReply::Info(PinInfo {
                file: key.file,
                layers: layers as u32,
                block_bytes: block as u64,
                dtype: KV_DTYPE as u32,
                state_bytes: key.state as u64,
                block_tokens: BLOCK_TOKENS as u32,
                segment: self.sched.pin_mark as u32,
                abi: larch_sys::LARCH_ABI_VERSION,
                version: env!("CARGO_PKG_VERSION"),
                build: larch_model::card::BUILD,
            }),
            PinOp::Chain { id, from, state } => {
                let (ns, tokens) = pinned(self, id).ok_or_else(changed)?;
                let (blocks, checkpoint) = self.sched.pin_chain(ns, &tokens);
                let count = blocks.len() * BLOCK_TOKENS;
                let state = match checkpoint {
                    Some(a) if state => (*a).clone(),
                    _ => Vec::new(),
                };
                let ids = tokens.get(from..count).unwrap_or_default().to_vec();
                PinReply::Chain { count, ids, state }
            }
            PinOp::Layer {
                id,
                from,
                to,
                layer,
            } => {
                let (ns, tokens) = pinned(self, id).ok_or_else(changed)?;
                // Version 0 only, as in `Scheduler::pin_chain`.
                let blocks = self.sched.prefix.chain((ns, 0), &tokens);
                let ok = from % BLOCK_TOKENS == 0
                    && to % BLOCK_TOKENS == 0
                    && from < to
                    && to / BLOCK_TOKENS <= blocks.len()
                    && layer < layers as u32;
                if !ok {
                    return Err(changed());
                }
                let blocks = &blocks[from / BLOCK_TOKENS..to / BLOCK_TOKENS];
                let mut out = host_bytes(2 * blocks.len() * block)
                    .ok_or(EngineError::Internal(KError::Oom.code()))?;
                layer_download(
                    &self.s,
                    (&l.k, &l.v),
                    n_blocks,
                    u64::from(layer),
                    blocks,
                    &mut out,
                )
                .map_err(internal)?;
                PinReply::Bytes(out)
            }
            PinOp::Begin {
                id,
                namespace,
                tokens,
            } => {
                if tokens.is_empty() || tokens.len() % BLOCK_TOKENS != 0 {
                    return Err(changed());
                }
                // A carried pin (its id given) was counted when it was made.
                if id.is_none() {
                    self.pins_fit(namespace, &tokens)?;
                }
                let id = id.unwrap_or_else(|| self.sh.next_id.fetch_add(1, Ordering::Relaxed));
                let blocks = (self.sched.alloc_pin(tokens.len() / BLOCK_TOKENS))
                    .ok_or(EngineError::InvalidRequest(invalid::DOC_TOO_LONG))?;
                self.imports.insert(
                    id,
                    ImportJob {
                        namespace,
                        tokens,
                        blocks,
                    },
                );
                PinReply::Id(id)
            }
            PinOp::Upload {
                id,
                layer,
                from_block,
                bytes,
            } => {
                let Some(job) = self.imports.get(&id) else {
                    return Err(torn_or_changed(&mut self.torn, id));
                };
                let n = bytes.len() / (2 * block).max(1);
                if layer >= layers as u32
                    || bytes.len() != 2 * n * block
                    || from_block + n > job.blocks.len()
                {
                    return Err(changed());
                }
                let to = &job.blocks[from_block..from_block + n];
                layer_upload(
                    &self.s,
                    (&l.k, &l.v),
                    n_blocks,
                    u64::from(layer),
                    to,
                    &bytes,
                )
                .map_err(internal)?;
                PinReply::Done
            }
            PinOp::Finish(id, state, tokens_done) => {
                let Some(job) = self.imports.remove(&id) else {
                    return Err(torn_or_changed(&mut self.torn, id));
                };
                let ImportJob {
                    namespace,
                    mut tokens,
                    mut blocks,
                } = job;
                let keep = tokens_done / BLOCK_TOKENS;
                if state.len() != key.state || keep == 0 || keep > blocks.len() {
                    self.sched.free_pin(&blocks);
                    return Err(changed());
                }
                // A torn import's later blocks were never filled.
                self.sched.free_pin(&blocks.split_off(keep));
                tokens.truncate(keep * BLOCK_TOKENS);
                // A carried pin comes back under the registration that survived the reload.
                let carried = (self.sched.prefix.pins()).any(|p| p.0 == id);
                if !carried {
                    self.sched.prefix.pin(id, namespace, tokens.clone());
                }
                let checkpoint = (key.state > 0).then_some(state);
                let got = (self
                    .sched
                    .publish_pin(id, namespace, &tokens, blocks, checkpoint))
                .len();
                if got != keep {
                    // No room for the whole chain is not a bad file. A new import has no handle
                    // yet, so its registration goes; a carried pin's stays, as its caller's
                    // handle names it, and its next question reads the document again.
                    if carried {
                        pin_cold(id, ("no room", got, keep));
                    } else {
                        self.sched.unpin(id);
                    }
                    return Err(EngineError::InvalidRequest(invalid::PINS_FULL));
                }
                PinReply::Done
            }
            PinOp::Abort(id) => {
                if let Some(job) = self.imports.remove(&id) {
                    self.sched.free_pin(&job.blocks);
                }
                // An import given up needs no word that it was torn.
                self.torn.retain(|&t| t != id);
                PinReply::Done
            }
        })
    }

    /// The documents plan for `tokens` ([`crate::document_plan`]) on the device as the last load
    /// outside a documents plan found it, so a fit inside the plan gives the answer that reloaded
    /// into it.
    fn doc_fit(&mut self, tokens: u32) -> Result<crate::DocFit, EngineError> {
        let refused =
            |e: LoadError| EngineError::InvalidRequest(crate::invalid::RELOAD_MODEL | e.code());
        if self.doc_terms.is_none() {
            self.doc_terms = Some(crate::DocTerms::read(&self.opts).map_err(refused)?);
        }
        let l = self
            .loaded
            .as_ref()
            .ok_or(EngineError::NotReady(HealthState::Loading))?;
        let mut before =
            crate::device_props(self.s.device()).map_err(|e| EngineError::Internal(e.code()))?;
        let seqs = knob_spec(knob_id::MAX_BATCH_SEQS).expect("registered").max as u32;
        let max_seats = self.cfg.values[knob_id::MAX_BATCH_SEQS as usize] as u32;
        let rows = seqs + l.info.caps.prefill_chunk_max;
        let graphs = self.cfg.values[knob_id::DECODE_GRAPHS as usize] != 0.0;
        // The reload starts on a new stream (`renew_stream`), which holds no capture arena yet.
        let state = (rows, graphs, false);
        // The plan loaded now, outside a documents plan: on a hybrid, the demand switch's fast or
        // many-chats plan, whichever it holds. A load into the documents plan keeps no layers the
        // CPU prefills (`direct_documents_fit`), so its 0 is no more than the plain load's.
        if self.docs_before.is_none() {
            self.base_host_layers = l.host_layers;
            self.base_free = l.free_before;
        }
        before.vram_free = self.base_free;
        let terms = self.doc_terms.as_ref().expect("read above");
        let base = self.base_host_layers;
        crate::document_plan(terms, &before, state, max_seats, base, tokens).map_err(refused)
    }

    /// Returns false on Shutdown. Knob changes apply here, between steps (9.8).
    fn handle(&mut self, m: Msg) -> Result<bool, KError> {
        // Publication after a fault is stopped by `terminal` and `host.failed`, not by this check.
        if self.failed() {
            return Ok(false);
        }
        match m {
            Msg::Submit(rs) => {
                let n = rs.len();
                for r in rs {
                    if r.observe {
                        r.completion
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .running = true;
                    }
                    self.sched.admit(r);
                }
                // Publish the scheduler's count before releasing the in-flight count. A concurrent
                // submit then sees the request in at least one of the two counters.
                self.sh
                    .sched_waiting
                    .store(self.sched.n_waiting(), Ordering::Relaxed);
                self.sh.in_flight.fetch_sub(n, Ordering::Release);
            }
            Msg::Knob(req, t, reply) => {
                self.sh.pending_knobs.fetch_sub(1, Ordering::Relaxed);
                let _ = reply.send(self.apply_set(&req, t)?);
            }
            Msg::Revert(target, who, t, reply) => {
                self.sh.pending_knobs.fetch_sub(1, Ordering::Relaxed);
                let _ = reply.send(self.apply_revert(target, who, t)?);
            }
            Msg::Ab(cmd, t, taken, reply) => {
                self.sh.pending_knobs.fetch_sub(1, Ordering::Relaxed);
                // The caller gave up first: it answered Busy, so the command must not run.
                if !taken.swap(true, Ordering::AcqRel) {
                    let _ = reply.send(self.apply_ab(cmd, t)?);
                }
            }
            Msg::Rejected(span) => {
                self.sh.pending_rejects.fetch_sub(1, Ordering::Relaxed);
                self.push(Tier::T0, &span);
            }
            Msg::Release(session) => self.sched.release_session(session),
            Msg::Pin(id, namespace, tokens, reply) => {
                let _ = reply.send(self.pin_chain_of(id, namespace, tokens));
            }
            Msg::Unpin(id) => {
                // A no-op for a pin that gave way; its mark goes so the set stays bounded.
                self.sched.unpin(id);
                (self.sh.cold_pins.lock().unwrap_or_else(|e| e.into_inner())).remove(&id);
                // The documents plan stays while any pin does ([`Self::restore_ready`]).
                self.idle_since = Instant::now();
            }
            Msg::PinOp(op, reply) => {
                // The device copies run here, between steps: what a chat streaming beside a read
                // waits through. The GPU run reads these to size them.
                let (name, began) = (op.name(), Instant::now());
                let done = self.pin_op(op);
                let ms = began.elapsed().as_millis();
                if ms > PIN_STALL_MS {
                    eprintln!("larch_pin_stall,{name},{ms}");
                }
                // A pin's export or import uses the documents plan: the idle clock starts after.
                self.idle_since = Instant::now();
                let _ = reply.send(done);
            }
            Msg::DocFit(tokens, reply) => {
                let _ = reply.send(self.doc_fit(tokens));
            }
            Msg::PrefixStats(reply) => {
                let _ = reply.send(self.sched.prefix_cache_stats());
            }
            Msg::Parked(session, reply) => {
                let _ = reply.send(self.sched.parked_tokens(session) as u32);
            }
            Msg::FinishStop(id, reply) => {
                self.sched.finish_stop(id, self.sh.now());
                self.flush_spans();
                let _ = reply.send(());
            }
            Msg::Reload(change, load, reply) => {
                // enqueue_reload already closed admission under the same lock as submit.
                self.push(
                    Tier::T0,
                    &HealthEvent {
                        from: HealthState::Ready.raw(),
                        to: HealthState::Draining.raw(),
                        ..bytemuck::Zeroable::zeroed()
                    },
                );
                self.reload = Some(PendingReload {
                    change,
                    load,
                    reply,
                });
            }
            Msg::RouterDeltas(d, version, reply) => {
                // A newer set supersedes one still waiting; its caller sees it was never applied.
                if let Some((.., old)) = self.swap.replace((d, version, reply)) {
                    let _ = old.send(Err(EngineError::Cancelled));
                }
                self.sched.hold = true;
            }
            Msg::Job(job) => {
                // A job still waiting is dropped: its caller sees the channel close.
                self.job = Some(job);
                self.sched.hold = true;
            }
            Msg::Prepare(digest, pixels) => self.queue_prepare(digest, pixels),
            Msg::Shutdown => return Ok(false),
        }
        Ok(true)
    }

    /// Applies waiting router deltas once no started request remains. They change each MoE layer's
    /// output and so every later layer's K/V, so KV, the prefix cache and parked sessions are keyed
    /// by version ([`crate::sched::Namespace`]): nothing computed under the old version is reused.
    fn swap(&mut self) -> Result<(), KError> {
        if self.sched.started() {
            return Ok(());
        }
        let Some((d, version, reply)) = self.swap.take() else {
            return Ok(());
        };
        self.sched.hold = self.job.is_some() || self.switch.is_some();
        let model = &mut self.loaded.as_mut().expect("loaded engine").model;
        match model.set_router_deltas(&self.s, d.as_ref()) {
            Ok(()) => {
                // A version number that comes back may bring other deltas: nothing computed
                // before is reused then, a hybrid's checkpoints included. 0 is the base model.
                if version != 0 && version <= self.top_version {
                    self.sched.new_generation();
                }
                self.top_version = self.top_version.max(version);
                self.sched.version = version;
                self.sh.adapter_version.store(version, Ordering::Release);
                let _ = reply.send(Ok(()));
            }
            Err(KError::InvalidArg | KError::Unsupported) => {
                let _ = reply.send(Err(EngineError::InvalidRequest(invalid::ROUTER_DELTAS)));
            }
            Err(e) => {
                let _ = reply.send(Err(EngineError::Internal(e.code())));
                return Err(e);
            }
        }
        Ok(())
    }

    /// Runs the waiting job once no started request remains, on the engine's stream, so nothing
    /// serves while it does; the model's version is the one in force.
    fn run_job(&mut self) {
        if self.sched.started() {
            return;
        }
        let Some(job) = self.job.take() else {
            return;
        };
        self.sched.hold = self.swap.is_some() || self.switch.is_some();
        // A job may run for seconds: not a step the watchdog times.
        self.sh.watch.idle();
        let model = &self.loaded.as_ref().expect("loaded engine").model;
        job(model.transformer(), &self.s);
    }

    /// Ends the engine (9.11). Health leaves Ready first, under the lock `submit` sends under, so the
    /// drain below sees every accepted request. On a fault, running requests get its error and queued
    /// ones `NotReady(Failed)`, and with `OnFault::Exit` the process exits with code 3; on shutdown
    /// every request gets `NotReady(Draining)`. The KV accounting is not touched: it may be what failed.
    fn stop(&mut self, fault: Option<(FaultCode, EngineError)>) {
        if let Some(fault) = fault {
            crate::lifecycle::fault(&self.sh, fault, self.opts.on_fault);
            return;
        }
        let now = self.sh.now();
        let to = HealthState::Draining;
        let code = FaultCode::None;
        let mut health = self.sh.health.lock().unwrap_or_else(|e| e.into_inner());
        if health.state == HealthState::Failed {
            return;
        }
        let from = health.state;
        *health = Health {
            state: to,
            since_mono_ns: now,
            last_fault: code,
            restarts: 0,
        };
        drop(health);
        let down = HealthEvent {
            from: from.raw(),
            to: to.raw(),
            fault: code.raw(),
            ..bytemuck::Zeroable::zeroed()
        };
        if from != to {
            self.push(Tier::T0, &down);
        }
        // The A/B ends on arm A with health, before any request hears of it (9.9). A request with no
        // token yet still carries the arm in force when it ended, as on every other end.
        let arm = self.cfg.ab_arm;
        self.end_ab(AbEnd::Stop);
        // A change its caller stopped waiting for still gets its ActivationEvent (9.8).
        let reason = RejectReason::NotReady;
        // Stop acknowledgements are answered (their senders dropped) only after fail_all has
        // written every request's completion.
        let mut acks = Vec::new();
        let messages: Vec<_> = self
            .sh
            .inbox
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .flat_map(|rx| rx.try_iter())
            .collect();
        for m in messages {
            match m {
                Msg::Submit(rs) => rs.into_iter().for_each(|r| self.sched.admit(r)),
                Msg::Knob(req, t, reply) => {
                    self.sh.pending_knobs.fetch_sub(1, Ordering::Relaxed);
                    let r = KnobReply::Rejected { reason };
                    self.activation(req.knob, req.value, req.trial_id, t, &r);
                    let _ = reply.send(r);
                }
                Msg::Revert(.., reply) => {
                    self.sh.pending_knobs.fetch_sub(1, Ordering::Relaxed);
                    let _ = reply.send(KnobReply::Rejected { reason });
                }
                Msg::Ab(.., reply) => {
                    self.sh.pending_knobs.fetch_sub(1, Ordering::Relaxed);
                    let _ = reply.send(AbReply::Rejected { reason });
                }
                m @ Msg::Rejected(_) => {
                    let _ = self.handle(m);
                }
                Msg::PrefixStats(reply) => drop(reply),
                Msg::DocFit(_, reply) => {
                    let _ = reply.send(Err(EngineError::NotReady(to)));
                }
                Msg::PinOp(_, reply) => {
                    let _ = reply.send(Err(EngineError::NotReady(to)));
                }
                Msg::Pin(.., reply) => {
                    let _ = reply.send(Err(EngineError::NotReady(to)));
                }
                Msg::Unpin(_) => {}
                Msg::Parked(_, reply) => {
                    let _ = reply.send(0);
                }
                Msg::FinishStop(_, reply) => acks.push(reply),
                Msg::RouterDeltas(.., reply) => drop(reply),
                Msg::Job(job) => drop(job),
                Msg::Prepare(..) => {}
                Msg::Reload(_, _, reply) => {
                    let _ = reply.send(Err(EngineError::NotReady(to)));
                }
                Msg::Release(_) | Msg::Shutdown => {}
            }
        }
        if std::env::var("LARCH_PREFIX_DIAGNOSTICS").is_ok_and(|v| v == "1") {
            eprintln!("larch_prefix_stats,{:?}", self.sched.prefix_cache_stats());
        }
        let queued = EngineError::NotReady(to);
        self.sched.fail_all(queued, queued, arm, now);
        drop(acks);
        if let Some(reload) = self.reload.take() {
            let _ = reload.reply.send(Err(queued));
        }
        self.flush_spans();
        // Close the original intake after final records/replies, including sends racing the drain.
        let inbox = self
            .sh
            .inbox
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        drop(inbox);
    }

    /// The host memory a reload's carry may copy.
    fn carry_left(&self, props: &DeviceProps) -> usize {
        (self.opts.carry_bytes).unwrap_or_else(|| carry_budget(host_available(props)))
    }

    /// Takes the pins a same-model reload carries through disk out of `pins`, written to files
    /// ([`Self::spill_pin`]), and returns the rest beside their carries. A pin is a whole
    /// document's read, a session one chat's, so pins come first; one goes to disk where host
    /// memory is the device's own, or it is past what is left of the budget. One the disk cannot
    /// take goes cold and is logged (`larch_pin_cold`): it stays registered and its next question
    /// reads the document again. Before the checkpoints go.
    fn spill_pins(
        &mut self,
        pins: Vec<PinnedChain>,
    ) -> Result<(Vec<PinnedChain>, Vec<Carry>), KError> {
        let Some(key) = carry_key(self.loaded.as_ref().expect("loaded engine")) else {
            return Ok((pins, Vec::new()));
        };
        let props = self.s.device().props()?;
        let (mut left, unified) = (self.carry_left(&props), host_available_is_device(&props));
        let (mut ram, mut out) = (Vec::new(), Vec::new());
        for pin in pins {
            let kv_n = 2 * key.kv.0 * pin.blocks.len() * key.kv.1;
            if !unified && kv_n <= left {
                left -= kv_n;
                ram.push(pin);
            } else {
                match self.spill_pin(pin.id) {
                    Ok(files) => {
                        let carried = CarriedPin {
                            namespace: pin.namespace,
                            checkpoint: None,
                            disk: Some(files),
                        };
                        out.push(Carry {
                            session: pin.id,
                            seq: Seq::of_pin(pin.tokens),
                            kv: Vec::new(),
                            state: Vec::new(),
                            pin: Some(carried),
                        });
                    }
                    Err(e) => pin_cold(pin.id, ("spill", e)),
                }
            }
        }
        Ok((ram, out))
    }

    /// A same-model reload's parked sequences with copies of their KV and state, taken before
    /// the teardown ([`Scheduler::carry_out`]), most recently used first, with what the new load
    /// must match for them to carry. A session past [`CARRY_BYTES`], or whose bytes the host or the
    /// model cannot copy, goes as a reload's reclaim; a device error drops them all.
    fn carry_out(
        &mut self,
        ram: Vec<PinnedChain>,
    ) -> Result<(Option<CarryKey>, Vec<Carry>), KError> {
        // Not from a file (a custom loader's model): nothing tells its next load is the same.
        let Some(key) = carry_key(self.loaded.as_ref().expect("loaded engine")) else {
            return Ok((None, Vec::new()));
        };
        let ((layers, block), n_blocks) = (key.kv, self.sched.blocks.total() as u64);
        let props = self.s.device().props()?;
        let mut left = self.carry_left(&props);
        let mut out = Vec::with_capacity(ram.len());
        // Pins first (`spill_pins` took those for disk), the budget what they left.
        for pin in &ram {
            left = left.saturating_sub(2 * layers * pin.blocks.len() * block);
        }
        let l = self.loaded.as_ref().expect("loaded engine");
        for pin in ram {
            let kv_n = 2 * layers * pin.blocks.len() * block;
            let Some(mut kv) = host_bytes(kv_n) else {
                pin_cold(pin.id, "host memory");
                continue;
            };
            let per = 2 * pin.blocks.len() * block;
            for (layer, out) in kv.chunks_exact_mut(per.max(1)).enumerate() {
                layer_download(
                    &self.s,
                    (&l.k, &l.v),
                    n_blocks,
                    layer as u64,
                    &pin.blocks,
                    out,
                )?;
            }
            let seq = Seq::of_pin(pin.tokens);
            let carried = CarriedPin {
                namespace: pin.namespace,
                checkpoint: pin.checkpoint,
                disk: None,
            };
            out.push(Carry {
                session: pin.id,
                seq,
                kv,
                state: Vec::new(),
                pin: Some(carried),
            });
        }
        let mut parked = self.sched.carry_out().into_iter();
        while let Some((session, seq)) = parked.next() {
            let kv_n = 2 * layers * seq.blocks.len() * block;
            let state_n = seq.state.map_or(0, |_| key.state);
            let bufs = (kv_n + state_n <= left)
                .then(|| Some((host_bytes(kv_n)?, host_bytes(state_n)?)))
                .flatten();
            let Some((mut kv, mut state)) = bufs else {
                self.sched.drop_carried(session, seq);
                continue;
            };
            let copy = (|| {
                if let Some(slot) = seq.state {
                    l.model.save_state(&self.s, slot, &mut state)?;
                }
                let per = 2 * seq.blocks.len() * block;
                for (layer, out) in kv.chunks_exact_mut(per.max(1)).enumerate() {
                    layer_download(
                        &self.s,
                        (&l.k, &l.v),
                        n_blocks,
                        layer as u64,
                        &seq.blocks,
                        out,
                    )?;
                }
                Ok(())
            })();
            match copy {
                Ok(()) => {
                    left -= kv_n + state_n;
                    out.push(Carry {
                        session,
                        seq,
                        kv,
                        state,
                        pin: None,
                    });
                }
                Err(KError::Unsupported) => self.sched.drop_carried(session, seq),
                Err(e) => {
                    self.sched.drop_carried(session, seq);
                    parked.for_each(|(session, seq)| self.sched.drop_carried(session, seq));
                    self.drop_carry(out);
                    return Err(e);
                }
            }
        }
        Ok((Some(key), out))
    }

    /// Saves the states the step ended on at a checkpoint ([`Scheduler::saves`]) to host memory,
    /// before the next step moves them. One the budget or the host has no room for, or the model
    /// cannot copy, is not saved; only a device error fails the step.
    fn save_checkpoints(&mut self) -> Result<(), KError> {
        let model = &self.loaded.as_ref().expect("loaded engine").model;
        for (slot, block) in std::mem::take(&mut self.sched.saves) {
            let Some(mut state) = self.sched.prefix.reserve(model.state_size()) else {
                continue;
            };
            match model.save_state(&self.s, slot, &mut state) {
                Ok(()) => self.sched.prefix.keep(block, state),
                Err(KError::Unsupported | KError::InvalidArg) => {}
                Err(e) => return Err(e),
            }
        }
        // A pin read's last mark goes once its next one is saved.
        for (new, old) in std::mem::take(&mut self.sched.retire) {
            self.sched.prefix.retire_checkpoint(new, old);
        }
        Ok(())
    }

    /// Drops carried sequences as reload reclaims, as [`Scheduler::reset_pool`] would have; a
    /// pin's files close with it.
    fn drop_carry(&mut self, carried: impl IntoIterator<Item = Carry>) {
        for c in carried {
            // A pin's chain is a cache entry, not a session: nothing was parked, so none is reclaimed.
            if c.pin.is_none() {
                self.sched.drop_carried(c.session, c.seq);
            }
        }
    }

    /// Puts the engine on a new stream for a reload, once the old model is gone: a stream keeps
    /// what it grew to serve a plan (Metal's 64 MiB staging copy and its scratch, split-K
    /// attention's growing with the context; CUDA's capture arena), which counts as used when the
    /// load measures the device, so each reload found less free than the first load (19.064 GB,
    /// then 18.996 and 18.920 on the M5) and planned less (#591 Mac check). Nothing needs the old
    /// stream by now: the captures, the stream pool and every buffer went with the model, the
    /// carried pins and sessions are host copies (`download` syncs), and `sync` ends its queued
    /// work. A new stream that cannot be made leaves the old one; a failed sync is the device's.
    fn renew_stream(&mut self) -> Result<(), KError> {
        self.s.sync()?;
        let free = |s: &Stream| crate::device_props(s.device()).map_or(0, |p| p.vram_free);
        let kept = free(&self.s);
        match Stream::new(self.s.device()) {
            // The old stream drops here, and its buffers with it.
            Ok(s) => self.s = s,
            Err(e) => eprintln!("larch_reload_stream,kept_old={e:?}"),
        }
        eprintln!(
            "larch_reload_stream,free_old_stream={kept},free_new_stream={}",
            free(&self.s)
        );
        Ok(())
    }

    /// Writes pin `id` to two unnamed files for a reload to carry, as an export: the pin file
    /// and its sidecar, rewound for reading. They live only as long as the handles.
    fn spill_pin(&mut self, id: u64) -> Result<(std::fs::File, std::fs::File), crate::PinError> {
        let (mut pin, mut state) = (tempfile::tempfile()?, tempfile::tempfile()?);
        let mut w = std::io::BufWriter::new(&mut pin);
        let end = crate::pinfile::export(&mut |op| self.pin_op(op), id, 0, &mut w)?;
        w.flush()?;
        drop(w);
        let mut w = std::io::BufWriter::new(&mut state);
        let covered = crate::pinfile::export_state(&mut |op| self.pin_op(op), id, &mut w)?;
        w.flush()?;
        drop(w);
        // The engine thread ran both, so nothing moved between them.
        if covered > end {
            return Err(crate::PinError::Miss);
        }
        for f in [&mut pin, &mut state] {
            f.seek(std::io::SeekFrom::Start(0))?;
        }
        Ok((pin, state))
    }

    /// Reads a spilled pin back under its own id (its registration survived the reload). One that
    /// cannot load (another file, past `max_context`, a pool without room, a disk error) is cold
    /// and logged: it stays registered and its next question reads the document again.
    fn unspill_pin(&mut self, c: Carry, same: bool) {
        let Some(CarriedPin {
            namespace,
            disk: Some((pin, state)),
            ..
        }) = c.pin
        else {
            return;
        };
        if !same {
            return pin_cold(c.session, "another model");
        }
        if c.seq.n_kv >= self.opts.max_context as usize {
            return pin_cold(c.session, "past max_context");
        }
        let (mut pin, mut state) = (std::io::BufReader::new(pin), std::io::BufReader::new(state));
        let loaded = crate::pinfile::import(
            &mut |op| self.pin_op(op),
            Some(c.session),
            &c.seq.tokens,
            namespace,
            &mut pin,
            &mut state,
            &mut || Ok(()),
        );
        if let Err(e) = loaded {
            pin_cold(c.session, ("import", e));
        }
    }

    /// Parks each carried sequence on the new load, most recently used first. All go as reload
    /// reclaims unless the new load matches `key` (the same file, KV layout and state size); one
    /// past the new `max_context`, or that the new pool, state slots or model have no room for,
    /// goes alone. Only a device error fails the reload.
    fn carry_in(&mut self, key: Option<CarryKey>, carried: Vec<Carry>) -> Result<(), KError> {
        let same = key.is_some() && key == carry_key(self.loaded.as_ref().expect("rebuilt engine"));
        let (disk, carried): (Vec<_>, Vec<_>) =
            (carried.into_iter()).partition(|c| c.pin.as_ref().is_some_and(|p| p.disk.is_some()));
        for c in disk {
            self.unspill_pin(c, same);
        }
        let l = self.loaded.as_mut().expect("rebuilt engine");
        let (layers, block) = kv_layout(l.model.config());
        let (n_blocks, state_n) = (self.sched.blocks.total() as u64, l.model.state_size());
        let mut carried = carried.into_iter();
        while let Some(c) = carried.next() {
            if let Some(pin) = c.pin {
                let n = c.seq.tokens.len() / BLOCK_TOKENS;
                if !same
                    || c.kv.len() != 2 * layers * n * block
                    || c.seq.n_kv >= self.opts.max_context as usize
                {
                    pin_cold(c.session, "another model, layout or past max_context");
                    continue;
                }
                let state =
                    (pin.checkpoint).map(|a| Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone()));
                let blocks =
                    self.sched
                        .carry_in_pin(c.session, pin.namespace, &c.seq.tokens, state);
                if blocks.len() < n {
                    pin_cold(c.session, ("no room", blocks.len(), n));
                }
                let copy = (|| {
                    // The chain may be shorter than the copy: the start of each layer's part.
                    for layer in 0..layers {
                        let from = layer * n * 2 * block;
                        let part = &c.kv[from..from + blocks.len() * 2 * block];
                        layer_upload(&self.s, (&l.k, &l.v), n_blocks, layer as u64, &blocks, part)?;
                    }
                    Ok(())
                })();
                match copy {
                    Ok(()) => {}
                    Err(e @ (KError::InvalidArg | KError::Unsupported)) => {
                        pin_cold(c.session, ("upload", e));
                        self.sched.uncarry_pin(c.session);
                    }
                    Err(e) => {
                        self.sched.uncarry_pin(c.session);
                        self.drop_carry(carried);
                        return Err(e);
                    }
                }
                continue;
            }
            let sized = c.kv.len() == 2 * layers * c.seq.blocks.len() * block
                && c.state.len() == c.seq.state.map_or(0, |_| state_n);
            if !same || !sized || c.seq.n_kv >= self.opts.max_context as usize {
                self.sched.drop_carried(c.session, c.seq);
                continue;
            }
            if !self.sched.carry_in(c.session, c.seq) {
                continue;
            }
            let (blocks, slot) = self.sched.carried(c.session);
            let copy = (|| {
                let per = 2 * blocks.len() * block;
                for (layer, part) in c.kv.chunks_exact(per.max(1)).enumerate() {
                    layer_upload(&self.s, (&l.k, &l.v), n_blocks, layer as u64, blocks, part)?;
                }
                match slot {
                    Some(slot) => l.model.load_state(&self.s, slot, &c.state),
                    None => Ok(()),
                }
            })();
            match copy {
                Ok(()) => {}
                Err(KError::InvalidArg | KError::Unsupported) => self.sched.uncarry(c.session),
                Err(e) => {
                    self.sched.uncarry(c.session);
                    self.drop_carry(carried);
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    fn rebuild(&mut self) -> bool {
        let PendingReload {
            change,
            load,
            reply,
        } = self.reload.take().expect("reload pending");
        let mut torn_down = false;
        // Keep the reply outside the unwind boundary: teardown faults must publish Failed before
        // the caller hears their error, just like faults from the replacement loader.
        let result = catch_unwind(AssertUnwindSafe(|| -> Result<bool, LoadError> {
            if self.sh.shutdown.load(Ordering::Acquire) {
                return Ok(false);
            }
            let mut opts = self.opts.clone();
            let before = DocsBefore {
                max_context: opts.max_context,
                kv_pool_fraction: opts.kv_pool_fraction,
                seats: (
                    self.cfg.values[knob_id::MAX_BATCH_SEQS as usize],
                    self.partial.seats_asked,
                ),
                drafter: (opts.drafter.clone(), opts.drafter_auto.clone()),
                set: DocsSet::default(),
            };
            let documents = change.documents;
            let same_model = change.model.is_none();
            if let Some(path) = change.model {
                opts = opts.onto(path);
            }
            if let Some(context) = change.max_context {
                opts.max_context = context;
            }
            if let Some(fraction) = change.kv_pool_fraction {
                opts.kv_pool_fraction = fraction;
            }
            if change.drafter_off {
                (opts.drafter, opts.drafter_auto) = (None, None);
            }
            // Header errors leave the current model, parked KV, knobs and A/B trial intact.
            let custom_loader = load.is_some();
            let (load, weights) = match load {
                Some(load) => (load, None),
                None => model_loader(&opts)?,
            };
            self.end_ab(AbEnd::Stop);
            self.guard = Guard::default();
            if !self.transition(HealthState::Loading) {
                return Ok(false);
            }
            let old_caps = self.sh.info.read().expect("model info").caps;
            let mut cfg = self.cfg;
            if let Some(seats) = change.max_batch_seqs {
                // As a hand-set knob: the plan pays its state slots (`Partial::seats_asked`).
                cfg.values[knob_id::MAX_BATCH_SEQS as usize] = f64::from(seats);
                self.partial.seats_asked = true;
            }
            if custom_loader {
                if !self.graphs_forced_off {
                    self.graphs_before_custom = cfg.values[knob_id::DECODE_GRAPHS as usize];
                }
                // As at start_with, a replacement loader may copy host memory during forward.
                cfg.values[knob_id::DECODE_GRAPHS as usize] = 0.0;
            } else if self.graphs_forced_off {
                cfg.values[knob_id::DECODE_GRAPHS as usize] = self.graphs_before_custom;
            }
            torn_down = true;
            // The pinned chains and their checkpoints are taken first: the checkpoints go before the
            // carry takes host memory, which they would stack on.
            let pins = match same_model && !custom_loader {
                true => self.sched.carry_pinned(),
                false => {
                    self.sched.prefix.unpin_all();
                    self.docs_before = None;
                    Vec::new()
                }
            };
            let (pins, spilled) = match same_model && !custom_loader {
                true => self
                    .spill_pins(pins)
                    .map_err(|e| LoadError::Backend(e.code()))?,
                false => (pins, Vec::new()),
            };
            // An import's blocks went with the pool; its next step hears that a reload tore it.
            for (id, _) in self.imports.drain() {
                if self.torn.len() == TORN_KEPT {
                    self.torn.pop_front();
                }
                self.torn.push_back(id);
            }
            self.sched.prefix.drop_checkpoints();
            // The same model's pinned chains and parked sessions keep their KV and state through
            // host copies.
            let carried = match same_model && !custom_loader {
                true => self.carry_out(pins),
                false => Ok((None, Vec::new())),
            };
            self.sched.reset_pool(0);
            let (key, mut carried) = match carried {
                Ok(c) => c,
                Err(e) => {
                    self.drop_carry(spilled);
                    return Err(LoadError::Backend(e.code()));
                }
            };
            carried.splice(0..0, spilled);
            // The model goes, and its deltas with it, and the suffix drafts of its token ids.
            self.sched.version = 0;
            self.sched.drafter = Box::<SuffixDrafter>::default();
            self.sh.adapter_version.store(0, Ordering::Release);
            drop(self.loaded.take());
            if self.failed() {
                // Teardown may outlive the watchdog. Unused captures can own failed resources.
                std::mem::forget(load);
                self.drop_carry(carried);
                return Err(LoadError::Backend(KError::DeviceLost.code()));
            }
            if let Err(e) = self.renew_stream() {
                self.drop_carry(carried);
                return Err(LoadError::Backend(e.code()));
            }
            let shared = self.sh.clone();
            let load: Loader = Box::new(move |dev, stream, rows| {
                if shared.failed.load(Ordering::Acquire) {
                    // The pre-load device query or header read may also outlive the watchdog.
                    std::mem::forget(load);
                    return Err(LoadError::Backend(KError::DeviceLost.code()));
                }
                let model = load(dev, stream, rows)?;
                if shared.failed.load(Ordering::Acquire) {
                    // A held loader may return after fault. Retain its device resources and stop
                    // before Loaded::load can start a new fit or allocate replacement buffers.
                    std::mem::forget(model);
                    return Err(LoadError::Backend(KError::DeviceLost.code()));
                }
                Ok(model)
            });
            let cache_compatible = weights.is_some() || opts.prefix_cache_custom_compatible;
            let loaded = Loaded::load(
                &mut opts,
                &mut cfg,
                load,
                weights,
                &self.s,
                old_caps.prefill_chunk_max as usize,
                &mut self.partial,
            );
            let mut loaded = match loaded {
                Ok(loaded) => loaded,
                Err(e) => {
                    self.drop_carry(carried);
                    return Err(e);
                }
            };
            loaded.info.caps.device_sample = old_caps.device_sample;
            self.sched.reset_pool(loaded.info.deployment.kv_pool_blocks);
            self.sched.set_state_slots(loaded.model.state_slots());
            self.sched.guess_max = guess_max(&opts, &loaded);
            self.sched.configure_prefix_cache(if cache_compatible {
                opts.prefix_cache_blocks
            } else {
                0
            });
            self.opts = opts;
            self.sched.pin_mark = self.opts.pin_mark.unwrap_or(crate::sched::PIN_MARK);
            if documents {
                // One line to set beside the fit's: a plan that keeps more host layers than the
                // cap (`base_host_layers`, where they prefill on the CPU) prefills them there.
                // ponytail: logged, not refused (the old model is gone by now); pass the fit's
                // layers to the planner if a Mac run shows `host_layers` past `cap` with streams=0
                eprintln!(
                    "larch_doc_reload,host_layers={},streams={},cap={},seats={},free_before={},fit_free={}",
                    loaded.host_layers,
                    u8::from(loaded.streams),
                    self.base_host_layers,
                    change.max_batch_seqs.unwrap_or(0),
                    loaded.free_before,
                    self.base_free,
                );
                let docs = self.docs_before.get_or_insert(before);
                let set = &mut docs.set;
                set.max_context = change.max_context.or(set.max_context);
                set.kv_pool_fraction = change.kv_pool_fraction.or(set.kv_pool_fraction);
                set.seats = change.max_batch_seqs.map(f64::from).or(set.seats);
                set.drafter_off |= change.drafter_off;
                // The plan was loaded for a pin about to come: its idle clock starts now, not
                // before the load.
                self.idle_since = Instant::now();
            }
            // Set, not revised: a revision of the seats would drop the cache and carry it just made.
            if let Some(seats) = change.max_batch_seqs {
                self.cfg.values[knob_id::MAX_BATCH_SEQS as usize] = f64::from(seats);
            }
            self.loaded = Some(loaded);
            self.doc_terms = None;
            self.carry_in(key, carried)
                .map_err(|e| LoadError::Backend(e.code()))?;
            // A new projector's cache is empty, and the queued pixels were cut for the old one.
            self.prepare.clear();
            self.warm_up();
            let sh = self.sh.clone();
            let health = sh.health.lock().unwrap_or_else(|e| e.into_inner());
            if sh.shutdown.load(Ordering::Acquire) || health.state == HealthState::Failed {
                drop(health);
                return Ok(false);
            }
            let info = self.loaded.as_ref().expect("rebuilt engine").info;
            for knob in [knob_id::MAX_CONTEXT, knob_id::KV_POOL_FRACTION] {
                if cfg.values[knob as usize] != self.value(knob) {
                    self.revise(
                        knob,
                        cfg.values[knob as usize],
                        Cause::Human,
                        0,
                        self.sh.now(),
                    );
                }
            }
            let graphs = cfg.values[knob_id::DECODE_GRAPHS as usize];
            if graphs != self.value(knob_id::DECODE_GRAPHS) {
                // The loader's kind decides it, not a person: a revert never undoes it (9.8).
                self.change(
                    knob_id::DECODE_GRAPHS,
                    graphs,
                    Cause::Reload,
                    0,
                    self.sh.now(),
                );
            }
            self.graphs_forced_off = custom_loader;
            sh.watch
                .set_floor(step_floor(&self.opts, info.caps.backend, info.class));
            *sh.info.write().expect("model info") = info;
            self.publish();
            // The pool reset's Reload reclaims, ahead of the deployment they preceded.
            self.flush_spans();
            self.push(Tier::T0, &info.deployment);
            // Hold admission closed through all publication. transition takes the same lock.
            drop(health);
            // Still before Ready, outside the lock: a device call should not hold up health readers.
            self.push_device(&info.caps);
            if !self.transition(HealthState::Ready) {
                return Ok(false);
            }
            Ok(true)
        }));
        let (resume, response) = match result {
            Ok(Ok(true)) => (true, Ok(())),
            Ok(Ok(false)) => (false, Err(EngineError::NotReady(HealthState::Draining))),
            Ok(Err(error)) if !torn_down => (
                self.transition(HealthState::Ready),
                Err(EngineError::InvalidRequest(
                    crate::invalid::RELOAD_MODEL | error.code(),
                )),
            ),
            result => {
                let (fault, error) = match result {
                    Ok(Err(LoadError::Backend(code))) => backend_fault(code),
                    Ok(Err(e)) => (
                        FaultCode::LoadError,
                        EngineError::Internal(crate::invalid::RELOAD_MODEL | e.code()),
                    ),
                    Err(_) => (FaultCode::Panic, EngineError::Internal(0)),
                    Ok(Ok(_)) => unreachable!(),
                };
                // The pool reset's and dropped carries' Reload reclaims, while the ring is open.
                self.flush_spans();
                self.stop(Some((fault, error)));
                (false, Err(error))
            }
        };
        let _ = reply.send(response);
        resume
    }

    fn transition(&mut self, state: HealthState) -> bool {
        let sh = self.sh.clone();
        let mut health = sh.health.lock().unwrap_or_else(|e| e.into_inner());
        if sh.shutdown.load(Ordering::Acquire) || health.state == HealthState::Failed {
            return false;
        }
        let from = health.state;
        health.state = state;
        health.since_mono_ns = sh.now();
        self.push(
            Tier::T0,
            &HealthEvent {
                from: from.raw(),
                to: state.raw(),
                ..bytemuck::Zeroable::zeroed()
            },
        );
        true
    }

    /// Writes the step's reclaims, then its ended spans, which may name them.
    fn flush_spans(&mut self) {
        for reclaim in std::mem::take(&mut self.sched.reclaims) {
            self.push(Tier::T0, &reclaim);
        }
        let mut ended = std::mem::take(&mut self.sched.ended);
        for (span, arm) in ended.drain(..) {
            let header = self.header(Tier::T0, arm);
            self.sh
                .host
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .span(header, &span);
        }
        self.sched.ended = ended;
    }

    /// The logits rows a step may plan: what the buffers can grow to from fresh free bytes beside
    /// the headroom and the capture arena. Captures still held count against it, so growth stays
    /// within it; growth pays with graph slots instead (see [`Self::observe_rows`]).
    fn observe_budget(&self) -> Result<usize, KError> {
        let loaded = self.loaded.as_ref().expect("loaded engine");
        let reserve = loaded.headroom_bytes + loaded.graph_arena_bytes;
        let free = self.s.device().props()?.vram_free;
        Ok(loaded.dims.rows_fitting(free, loaded.logit_rows, reserve))
    }

    /// Grows logits scratch to `rows`, which [`Self::observe_budget`] admitted, and shrinks the graph
    /// quota to the whole slots left beside it. Never shrinks the buffers.
    fn observe_rows(&mut self, rows: usize) -> Result<(), KError> {
        let loaded = self.loaded.as_mut().expect("loaded engine");
        // Captures hold the buffers about to be replaced.
        loaded.graphs.clear();
        let dev = self.s.device();
        let [ib, lb, sb] = loaded.dims.bytes(rows);
        let free = dev.props()?.vram_free;
        let reserve = loaded.headroom_bytes + loaded.graph_arena_bytes;
        let old = loaded
            .dims
            .bytes(loaded.logit_rows)
            .into_iter()
            .sum::<u64>();
        // Observation work trades future executable slots for scratch, never headroom or the
        // possibly allocated arena.
        let graph_bytes = loaded
            .graph_bytes
            .min((free + old).saturating_sub(ib + lb + sb + reserve) / GRAPH_BYTES * GRAPH_BYTES);
        // Allocate before swapping: a physical OOM at the temporary peak leaves serving buffers intact.
        for bytes in [ib, lb, sb] {
            self.partial.buffers.push(allocate(dev, bytes)?);
        }
        let [input, logits, samp] = std::mem::take(&mut self.partial.buffers)
            .try_into()
            .ok()
            .expect("three observation buffers");
        (loaded.input, loaded.logits, loaded.samp) = (input, logits, samp);
        loaded.logit_rows = rows;
        // Publish the monotonic quota only once replacement succeeds; live knobs cannot restore it.
        loaded.graph_bytes = graph_bytes;
        Ok(())
    }

    /// Stamps `arm` on what follows; the caller publishes once the values match it.
    fn set_arm(&mut self, arm: AbArm) {
        self.cfg.ab_arm = arm.raw();
        self.sched.set_arm(arm.raw());
    }

    fn publish(&self) {
        let mut host = self.sh.host.lock().unwrap_or_else(|e| e.into_inner());
        if host.failed {
            return;
        }
        host.ab = self.ab.clone();
        host.step = self.step;
        *self.sh.effective.write().expect("config") = Arc::new(self.cfg);
    }

    fn value(&self, knob: u16) -> f64 {
        self.cfg.values[knob as usize]
    }

    /// The tier step `step` writes its EngineSample at: every step at t1, one in 64 at t0.
    fn sample_tier(&self, step: u64) -> Option<Tier> {
        match self.value(knob_id::TELEMETRY_TIER) as u32 {
            _ if self.opts.telemetry == TelemetryTier::Off => None,
            1 => Some(Tier::T1),
            _ if step.is_multiple_of(64) => Some(Tier::T0),
            _ => None,
        }
    }

    /// A hybrid whose every state slot runs while a request waits grows its state toward
    /// `max_batch_seqs` (at most [`hybrid_slots`]) from what is free beside the headroom, the
    /// capture arena and the graph quota not yet captured; the request waits while there is no
    /// room, and the next try is [`GROW_RETRY_STEPS`] later.
    fn grow_seats(&mut self) -> Result<(), KError> {
        let loaded = self.loaded.as_mut().expect("loaded engine");
        let slots = loaded.model.state_slots();
        let want = hybrid_slots(self.cfg.values[knob_id::MAX_BATCH_SEQS as usize] as u32);
        if slots == 0
            || slots >= want
            || self.sched.n_waiting() == 0
            || self.sched.n_running() < slots as usize
            || self.step < self.grow_at
        {
            return Ok(());
        }
        let used: u64 = (loaded.graphs.values())
            .map(|c| match c {
                Capture::Graph(.., bytes) => *bytes,
                _ => 0,
            })
            .sum();
        let kept = loaded.headroom_bytes
            + loaded.graph_arena_bytes
            + loaded.graph_bytes.saturating_sub(used);
        let room = self.s.device().props()?.vram_free.saturating_sub(kept);
        let grown = match loaded.model.grow_state(&self.s, want, room) {
            Err(KError::Oom) => slots,
            r => r?,
        };
        if grown > slots {
            // Captures froze the old state's addresses and slot count.
            loaded.graphs.clear();
            self.sched.grow_state_slots(grown);
        } else {
            self.grow_at = self.step + GROW_RETRY_STEPS;
        }
        Ok(())
    }

    /// Reloads for a hand-set `max_batch_seqs` once idle, where the loaded plan differs from what
    /// it asks ([`replan_due`]). A reload drains like [`EngineHandle::reload`]: requests submitted
    /// while it loads are refused NotReady, and parked sessions carry across ([`Self::carry_out`]).
    fn replan_seats(&mut self) {
        let want = hybrid_slots(self.value(knob_id::MAX_BATCH_SEQS) as u32);
        if self.replan_due(want) {
            // A person's seats: the reload's own events report it.
            self.queue_replan();
        }
    }

    /// [`replan_due`] for this load.
    fn replan_due(&self, want: u32) -> bool {
        let l = self.loaded.as_ref().expect("loaded engine");
        let in_ram = l.info.class.experts_in_ram;
        let flips = l.info.caps.backend == Backend::Cuda
            && std::env::var_os("LARCH_EXPERT_CACHE").is_none();
        replan_due(want, l.model.state_slots(), in_ram, l.seats_planned, flips)
    }

    /// Queues a reload with the plan's seats as `partial` asks and returns its reply; None when
    /// health is not Ready.
    fn queue_replan(&mut self) -> Option<std::sync::mpsc::Receiver<Result<(), EngineError>>> {
        self.queue_reload(ReloadChange::default())
    }

    /// Queues a reload of `change`, as [`Self::queue_replan`] does.
    fn queue_reload(
        &mut self,
        change: ReloadChange,
    ) -> Option<std::sync::mpsc::Receiver<Result<(), EngineError>>> {
        let sh = self.sh.clone();
        let mut health = sh.health.lock().unwrap_or_else(|e| e.into_inner());
        // Under submit's lock, as `enqueue_reload`: a caller's reload in flight wins.
        if health.state != HealthState::Ready {
            return None;
        }
        (health.state, health.since_mono_ns) = (HealthState::Draining, sh.now());
        drop(health);
        self.push(
            Tier::T0,
            &HealthEvent {
                from: HealthState::Ready.raw(),
                to: HealthState::Draining.raw(),
                ..bytemuck::Zeroable::zeroed()
            },
        );
        let (reply, rx) = sync_channel(1);
        self.reload = Some(PendingReload {
            change,
            load: None,
            reply,
        });
        Some(rx)
    }

    /// Goes back to the plan a documents reload replaced: each field the documents reloads set
    /// returns to what it was, if it still holds what they set (the rest is the person's now),
    /// then an ordinary reload, which carries the parked sessions. None when nothing went back.
    fn restore_plan(&mut self) -> Option<std::sync::mpsc::Receiver<Result<(), EngineError>>> {
        let before = self.docs_before.take()?;
        let set = &before.set;
        let seats = &mut self.cfg.values[knob_id::MAX_BATCH_SEQS as usize];
        let mut back = false;
        if set.max_context == Some(self.opts.max_context) {
            self.opts.max_context = before.max_context;
            back = true;
        }
        if set.kv_pool_fraction == Some(self.opts.kv_pool_fraction) {
            self.opts.kv_pool_fraction = before.kv_pool_fraction;
            back = true;
        }
        if set.seats == Some(*seats) {
            (*seats, self.partial.seats_asked) = before.seats;
            // As the options' own load would have it: nobody set the seats or the expert cache.
            if set.auto_off {
                self.auto =
                    !self.partial.seats_asked && std::env::var_os("LARCH_EXPERT_CACHE").is_none();
            }
            back = true;
        }
        if set.drafter_off && self.opts.drafter.is_none() && self.opts.drafter_auto.is_none() {
            (self.opts.drafter, self.opts.drafter_auto) = before.drafter;
            back = true;
        }
        back.then(|| self.queue_replan()).flatten()
    }

    /// No pin is held or being imported, so a documents plan has no one to serve. An open import
    /// ([`Self::imports`]) is a pin on its way: a restore would drop its blocks, and its next
    /// step's `PIN_CHANGED` would read as a bad file, which the caller deletes (#590 Mac review).
    fn docs_unpinned(&self) -> bool {
        self.docs_before.is_some()
            && self.sched.prefix.pins().next().is_none()
            && self.imports.is_empty()
    }

    /// Latches [`Self::restoring`] when a request waits for a seat the documents plan lacks and
    /// no pin is held, and holds new requests back until the started ones end; a pin that
    /// returns first keeps the plan.
    fn watch_restore(&mut self) {
        if !self.docs_unpinned() {
            if std::mem::take(&mut self.restoring) {
                self.sched.hold =
                    self.swap.is_some() || self.job.is_some() || self.switch.is_some();
            }
            return;
        }
        let seats = self.limits().max_batch_seqs;
        if self.sched.n_waiting() > 0 && self.sched.n_running() >= seats {
            self.restoring = true;
        }
        if self.restoring {
            self.sched.hold = true;
        }
    }

    /// Whether the documents plan goes back now, never while a request is mid-flight or a pin
    /// file's import is open ([`Self::docs_unpinned`]): a request held for a seat it lacks, once
    /// the started ones ended; or the engine idle with no pin for [`DOCS_RESTORE_IDLE`].
    fn restore_ready(&self) -> bool {
        self.docs_unpinned()
            && self.reload.is_none()
            && ((self.restoring && !self.sched.started())
                || (self.sched.is_idle() && self.idle_since.elapsed() >= DOCS_RESTORE_IDLE))
    }

    /// Reloads for [`Self::restore_ready`] at once, as a demand switch does ([`Self::switch_seats`]):
    /// no message is handled between its decision and the load, so a pin cannot find the reload
    /// queued; the queue and requests submitted meanwhile wait out the load, then run on the plan.
    fn restore_now(&mut self) -> bool {
        self.restoring = false;
        let queued = self.sched.take_unstarted();
        self.sh.switching.store(true, Ordering::Release);
        self.sh.watch.idle();
        let reply = match self.sched.is_idle() {
            true => self.restore_plan(),
            false => None,
        };
        let resume = reply.is_none() || self.rebuild();
        self.sh.switching.store(false, Ordering::Release);
        queued.into_iter().for_each(|r| self.sched.admit(r));
        self.sched.hold = self.swap.is_some() || self.job.is_some();
        resume
    }

    /// Starts a demand switch ([`Demand`]) on a planned hybrid with host experts on CUDA, where
    /// the plans were measured: new requests wait until no started one remains, at most
    /// [`SWITCH_HOLD_NS`].
    fn watch_demand(&mut self, now: u64) {
        if self.switch.is_some() || self.demand_blocked() {
            // Time blocked is not time at one chat (or under load): the clock starts over after.
            self.demand.since = None;
            return;
        }
        let l = self.loaded.as_ref().expect("loaded engine");
        let slots = l.model.state_slots();
        if l.info.caps.backend != Backend::Cuda
            || slots == 0
            || !l.info.class.experts_in_ram
            || l.seats_planned.is_none()
        {
            return;
        }
        let (running, waiting) = (self.sched.n_running(), self.sched.n_waiting());
        let Some(want) = self.demand.seats(now, running, waiting, slots) else {
            return;
        };
        if self.replan_due(want) {
            let (from, by) = (self.partial.seats_auto, now + SWITCH_HOLD_NS);
            self.switch = Some(Switch { from, by });
            (self.partial.seats_auto, self.sched.hold) = (want, true);
        }
    }

    /// What a switch's reload would undo or stop: router deltas (applied or waiting), an open
    /// A/B or a guardrail watch, `max_batch_seqs` held under many chats (by gait or an arm), a
    /// person's reload or seats.
    fn demand_blocked(&self) -> bool {
        !self.auto
            || self.reload.is_some()
            || self.sched.version != 0
            || self.swap.is_some()
            || self.ab.is_some()
            || self.guard.watching()
            || self.value(knob_id::MAX_BATCH_SEQS) < f64::from(MANY_SEATS)
    }

    /// Whether the pending switch runs now: once no started request remains. One whose reload is
    /// blocked, that held the queue past its deadline, or a switch down with two or more waiting
    /// (demand turned while it held), is cancelled.
    fn switch_ready(&mut self) -> bool {
        let Some(sw) = self.switch else {
            return false;
        };
        let now = self.sh.now();
        let down_stale = self.partial.seats_auto < MANY_SEATS && self.sched.n_waiting() >= 2;
        let late = self.sched.started() && now > sw.by;
        if self.demand_blocked() || late || (!self.sched.started() && down_stale) {
            self.switch = None;
            self.partial.seats_auto = sw.from;
            self.sched.hold = self.swap.is_some() || self.job.is_some();
            self.demand.cancelled(now);
            return false;
        }
        !self.sched.started()
    }

    /// Reloads for the demand switch once no started request remains. Unlike a person's reload,
    /// the queue and requests submitted meanwhile wait out the load, then run on the new plan;
    /// parked sessions keep their KV and state where the reload can carry them (else their next
    /// turns read their history again).
    /// A load that fails before the teardown keeps the old plan and ends the demand switch; one
    /// that fails after it faults the engine as a person's reload does.
    fn switch_seats(&mut self) -> bool {
        let sw = self.switch.take().expect("switch pending");
        let (t, seats) = (Instant::now(), self.partial.seats_auto);
        let queued = self.sched.take_unstarted();
        self.sh.switching.store(true, Ordering::Release);
        // Loading has the same unbounded host/device setup as startup, not a step deadline.
        self.sh.watch.idle();
        let reply = match self.sched.is_idle() {
            true => self.queue_replan(),
            false => None,
        };
        let resume = reply.is_none() || self.rebuild();
        self.sh.switching.store(false, Ordering::Release);
        queued.into_iter().for_each(|r| self.sched.admit(r));
        self.sched.hold = self.swap.is_some() || self.job.is_some();
        match reply.map(|rx| rx.try_recv()) {
            Some(Ok(Ok(()))) => self.demand.switched(seats),
            Some(Ok(Err(e))) => {
                // ponytail: stderr, as no record type holds a failed switch; add one to
                // larch-records if the app needs to show it.
                eprintln!("larch_seat_switch_failed,{seats},{e:?}");
                (self.partial.seats_auto, self.auto) = (sw.from, false);
            }
            _ => self.partial.seats_auto = sw.from,
        }
        if std::env::var("LARCH_FIT_DIAGNOSTICS").is_ok_and(|v| v == "1") {
            let ms = t.elapsed().as_millis();
            eprintln!("larch_seat_switch,{seats},{ms}");
        }
        resume
    }

    fn limits(&self) -> Limits {
        let loaded = self.loaded.as_ref().expect("loaded engine");
        // Past the state slots, requests wait until a slot frees or `grow_seats` adds one.
        let seats = match loaded.model.state_slots() {
            0 => usize::MAX,
            n => n as usize,
        };
        Limits {
            max_batch_seqs: (self.value(knob_id::MAX_BATCH_SEQS) as usize).min(seats),
            prefill_chunk_tokens: self.value(knob_id::PREFILL_CHUNK_TOKENS) as usize,
            policy: match self.value(knob_id::SCHEDULE_POLICY) as u32 {
                1 => Policy::Balanced,
                2 => Policy::PrefillFirst,
                _ => Policy::DecodeFirst,
            },
            kv_preempt_watermark: self.value(knob_id::KV_PREEMPT_WATERMARK),
            logit_rows: loaded.logit_rows,
        }
    }

    fn step(&mut self) -> Result<(), KError> {
        if self.failed() {
            return Ok(());
        }
        // A job or a seat switch disarms the watchdog, and with work still queued no idle wait
        // comes to re-arm it: a step with work is always a watched one.
        if !self.sched.is_idle() {
            self.sh.watch.begin(self.sh.now());
        }
        self.ab_boundary();
        let start = self.sh.now();
        // What the unit cache holds, for the plan's image rules (`Scheduler::image_chunk`).
        match self.loaded.as_ref().and_then(|l| l.vision.as_ref()) {
            Some(v) => {
                self.sched.slice_images = v.model.sliced();
                self.sched.units_ready = v.slots.iter().flatten().map(|e| (e.0, e.1)).collect();
            }
            None => self.sched.slice_images = false,
        }
        self.grow_seats()?;
        let mut lim = self.limits();
        if self.sched.observing() {
            // Budgeted before scheduling: observed prefill runs in the row chunks the budget
            // holds, so no refusal can come after a request took over its session's parked KV.
            lim.logit_rows = self.observe_budget()?;
        }
        let loaded = self.loaded.as_mut().expect("loaded engine");
        // The model refused its first check, so no step guesses again: the drafter goes, so no
        // step taps or feeds it. Its graphs write its freed tap buffers: their quota comes back.
        // ponytail: the MTP block's weights stay with the model for the load (only Metal before
        // Apple GPU family 10 has no mmvq_verify); free them with the drafter or refuse MTP there
        // when Metal runs it.
        if self.sched.checkable == Some(false) {
            loaded.graphs.retain(|k, _| !k.taps);
            loaded.draft = None;
        }
        let mut plan = match &loaded.draft {
            Some(draft) => {
                let mut d = DraftStep {
                    draft,
                    s: &self.s,
                    model: &*loaded.model,
                    err: None,
                };
                let plan = self.sched.schedule_with(&lim, start, Some(&mut d));
                if let Some(e) = d.err {
                    return Err(e);
                }
                plan
            }
            None => self.sched.schedule_with(&lim, start, None),
        };
        // The plan's reclaims reach the ring before anything below can fail, so no span ever names
        // a ReclaimEvent that was not written.
        self.flush_spans();
        // Pins that gave way to this plan's requests are cold before any of them can finish.
        for id in self.sched.take_cold() {
            pin_cold(id, "gave way");
            (self.sh.cold_pins.lock().unwrap_or_else(|e| e.into_inner())).insert(id);
        }
        // A saturated step pays for the probe only when it writes its sample.
        if self.sample_tier(self.step + 1).is_some() {
            plan.cap_blocked = self.sched.cap_blocked(&lim);
        }
        let loaded = self.loaded.as_mut().expect("loaded engine");
        // A rewound slot that sits this step out (parked, freed or waiting) goes home, so only the
        // last checker can still hold a spare and guesses + 1 of them always suffice.
        if loaded.model.state_slots() > 0 && self.opts.guessing() {
            let seqs = plan.decode.iter().chain(plan.prefill.iter().map(|p| &p.0));
            let stepped: Vec<u32> = seqs
                .filter_map(|&id| self.sched.req(id)?.state_slot)
                .collect();
            for slot in (0..loaded.model.state_slots()).filter(|s| !stepped.contains(s)) {
                loaded.model.copy_state(&self.s, slot, slot)?;
            }
        }
        // A recurrent model rewinds a check's state to its kept rows, or refuses checks.
        let mut checks = Vec::new();
        if loaded.model.state_slots() > 0 && plan.guessed > 0 {
            for (id, _, len) in plan.prefill.clone() {
                let r = self.sched.req(id).expect("planned request");
                if r.guesses == 0 {
                    continue;
                }
                let slot = r.state_slot.expect("a running sequence");
                match loaded.model.begin_check(&self.s, slot, len as u32) {
                    Ok(()) => {
                        self.sched.checkable = Some(true);
                        checks.push((id, slot));
                    }
                    Err(KError::Unsupported) => self.sched.uncheck(&mut plan, id),
                    Err(e) => return Err(e),
                }
            }
        }
        self.sched.logits(&plan, &mut loaded.staging.rows);
        let rows = loaded.staging.rows.len();
        if rows > loaded.logit_rows {
            self.observe_rows(rows)?;
            if self.failed() {
                return Ok(());
            }
        }
        self.sh
            .sched_waiting
            .store(self.sched.n_waiting(), Ordering::Relaxed);
        // A sliced projector encodes the next image a request will read, some layers a step, while
        // the chunk that reads it waits; a step with nothing else to run finishes it.
        self.encode_slice(plan.is_empty())?;
        if plan.is_empty() {
            // Only an idle scheduler plans nothing (the last requests were cancelled), or every
            // request waited on the image encoded above.
            return Ok(());
        }
        self.step += 1;
        self.sh.host.lock().unwrap_or_else(|e| e.into_inner()).step = self.step;
        // Before the step: each load and copy is a state where its row starts, a checkpoint's
        // or its snapshot.
        let model = &mut self.loaded.as_mut().expect("loaded engine").model;
        for (slot, state) in &plan.loads {
            model.load_state(&self.s, *slot, state)?;
        }
        for &(from, to) in &plan.copies {
            model.copy_state(&self.s, from, to)?;
        }
        let before = (self.sched.guess_max > 0).then(|| self.sh.now());
        let as_check = self.decodes_as_check(&plan);
        let wall = Instant::now();
        let ran = match self.execute(&plan) {
            // A backend without the check kernels: the model refused the step before any op, so
            // each check runs as its decode row and no step guesses again.
            Err(KError::Unsupported) if plan.guessed > 0 || as_check => {
                let loaded = self.loaded.as_mut().expect("loaded engine");
                for (_, slot) in checks.drain(..) {
                    // A check whose step did not run is dropped (InvalidArg); one that ran in
                    // part may have moved its state, which only a fault leaves untouched.
                    match loaded.model.end_check(&self.s, slot, 1) {
                        Err(KError::InvalidArg) => {}
                        Ok(()) => return Err(KError::Unsupported),
                        Err(e) => return Err(e),
                    }
                }
                let guessed = |id: &u64| self.sched.req(*id).is_some_and(|r| r.guesses > 0);
                let ids: Vec<u64> = plan.prefill.iter().map(|p| p.0).filter(guessed).collect();
                for id in ids {
                    self.sched.uncheck(&mut plan, id);
                }
                self.sched.checkable = Some(false);
                self.sched.logits(&plan, &mut loaded.staging.rows);
                self.execute(&plan)
            }
            r => r,
        };
        let (gpu_ns, kv_tokens, replayed, captured) = ran?;
        if self.failed() {
            return Ok(());
        }
        let check = plan.guessed > 0 || self.decodes_as_check(&plan);
        verify_time(&plan, check, wall.elapsed().as_nanos() as u64, gpu_ns);
        // A plain decode step or a check step (which feeds no prompt) prices the next checks; a
        // backend without GPU timing gives the step's own time.
        if let Some(before) = before
            && prices_checks(&plan, captured)
        {
            let ns = gpu_ns.unwrap_or_else(|| self.sh.now().saturating_sub(before));
            // A check's draft forward ran during scheduling, outside `before` and `gpu_ns`.
            let draft = if plan.guessed > 0 { plan.draft_ns } else { 0 };
            let rows = plan.decode.len() + plan.prefill.len();
            self.sched.costs.record(rows, plan.guessed, ns + draft);
        }
        if self.sched.tapping() {
            self.publish_taps();
        }
        self.observe_drafts(&plan)?;
        let loaded = self.loaded.as_mut().expect("loaded engine");
        for (id, slot) in checks {
            let fed = loaded.staging.sampled.iter().filter(|s| s.0 == id).count();
            loaded.model.end_check(&self.s, slot, fed as u32)?;
        }
        self.sched.complete(
            &plan,
            self.loaded
                .as_mut()
                .expect("loaded engine")
                .staging
                .sampled
                .drain(..),
            self.sh.now(),
        );
        self.save_checkpoints()?;
        let now = self.sh.now();
        if let Some(d) = &mut self.loaded.as_mut().expect("loaded engine").draft {
            d.finished(|id| self.sched.req(id).is_none(), now);
        }
        if plan.guess_room > 0 {
            let t = &self.sched.tot;
            *self.sh.guesses.lock().unwrap_or_else(|e| e.into_inner()) = GuessStats {
                rounds: t.guess_rounds,
                tried: t.guesses_tried,
                kept: t.guesses_kept,
                k: t.guess_k,
            };
            if verify_time_on() {
                eprintln!(
                    "larch_guesses,rounds={},tried={},kept={},draft_us={}",
                    t.guess_rounds,
                    t.guesses_tried,
                    t.guesses_kept,
                    plan.draft_ns / 1000
                );
            }
        }
        self.flush_spans();
        let dur = self.sh.now().saturating_sub(start);
        self.sh.watch.complete(self.sh.now(), Some(dur));
        if let Some(tier) = self.sample_tier(self.step) {
            let sample = self.sample_record(&plan, dur, gpu_ns, kv_tokens, replayed, captured);
            self.push(tier, &sample);
        }
        // The guardrail acts at this boundary, so its revert is in force for the next step.
        let n_prefill_tokens = plan.prefill.iter().map(|p| p.2 as u32).sum();
        let arm_b = self.cfg.ab_arm == AbArm::B.raw();
        let ratio = self.value(knob_id::GUARDRAIL_RATIO);
        // A check step's cost is its rows', not a decode row's: it stays out of the history.
        let n_decode = if plan.guessed > 0 {
            0
        } else {
            plan.decode.len() as u32
        };
        let trips = self
            .guard
            .step(n_decode, n_prefill_tokens, dur, captured, arm_b, ratio);
        for target in trips {
            match target {
                Target::Rev { rev, .. } => self.guard_revert(rev),
                Target::Ab => {
                    self.end_ab(AbEnd::Guardrail);
                }
            }
        }
        // Here, not at the next boundary: an idle engine would otherwise wait on arm B.
        if let Some(ab) = &self.ab
            && ab.max_steps > 0
            && self.step + 1 - ab.start >= ab.max_steps
        {
            self.end_ab(AbEnd::MaxSteps);
        }
        Ok(())
    }

    /// [`decodes_as_check`] for this load.
    fn decodes_as_check(&self, plan: &StepPlan) -> bool {
        let metal = (self.loaded.as_ref()).is_some_and(|l| l.info.caps.backend == Backend::Metal);
        decodes_as_check(metal, self.sched.guess_max, self.sched.checkable, plan)
    }

    /// The next step's `attn_splits`. Checks are exact only at the decode rows' own nonzero splits.
    /// The Metal backend's pick depends on the step's rows and table width, which a check changes,
    /// so guessing pins the pick while checks can run: 16 is what it takes for up to 10 decode rows
    /// on an M5 past 256 tokens. After a refused check decode is plain again.
    fn attn_splits(&self) -> u32 {
        match self.value(knob_id::ATTN_DECODE_SPLITS) as u32 {
            0 if self.sched.guess_max > 0 && self.sched.checkable != Some(false) => GUESS_SPLITS,
            splits => splits,
        }
    }

    /// Feeds the drafter every position the step committed: each sequence's decode row, prompt
    /// chunk, or a check's rows up to the last token it kept (the step's sampled tokens, before
    /// [`Scheduler::complete`] drops the rest), from the step's taps of every row.
    fn observe_drafts(&mut self, plan: &StepPlan) -> Result<(), KError> {
        let loaded = self.loaded.as_mut().expect("loaded engine");
        let Some(draft) = &mut loaded.draft else {
            return Ok(());
        };
        let st = &loaded.staging;
        let chunks = plan.decode.iter().map(|&id| (id, None, 1));
        let chunks = chunks.chain(
            plan.prefill
                .iter()
                .map(|&(id, start, len)| (id, Some(start), len)),
        );
        let n = st.tap_rows;
        let mut row = 0u32;
        for (id, start, len) in chunks {
            let first = row;
            row += len as u32;
            let r = self.sched.req(id).expect("planned request");
            if !r.may_draft() {
                continue;
            }
            let start = start.unwrap_or(r.seq.n_kv);
            let kept = match r.guesses {
                0 => len,
                _ => st.sampled.iter().filter(|s| s.0 == id).count(),
            } as u32;
            let (session, untapped) = (r.parks_as(), !st.drafted);
            let tokens = &r.seq.tokens[start..start + kept as usize];
            draft.observe(
                &self.s,
                &*loaded.model,
                id,
                session,
                start,
                n,
                first..first + kept,
                tokens,
                untapped,
            )?;
        }
        Ok(())
    }

    /// Hands each tapped request its rows from the step's download, at the step boundary: they
    /// leave with its Done, from [`Scheduler::complete`].
    fn publish_taps(&mut self) {
        let loaded = self.loaded.as_ref().expect("loaded engine");
        let st = &loaded.staging;
        let (d, n) = (loaded.model.config().d_model as usize, st.taps.len());
        let out: &[f32] = bytemuck::cast_slice(&st.out[st.tap_out.min(st.out.len())..]);
        for (j, &(id, _)) in st.taps.iter().enumerate() {
            let r = self.sched.req_mut(id).expect("tapped request");
            if st.untapped {
                r.untapped = true;
                continue;
            }
            let block = |l| st.layers.binary_search(l).expect("a tapped layer");
            let row = |l| out[(block(l) * n + j) * d..][..d].to_vec();
            r.rows.extend(r.activations.iter().map(row));
        }
    }

    /// Packs the step's inputs into one upload, runs the model (eligible decode buckets or bounded
    /// full-chunk mixed shapes can replay graphs), `sample_prep` and the score requests' gathers,
    /// downloads the candidates and scores (the step's one host sync), reports observed rows and
    /// samples on the host. `gpu_ns` covers the upload to the gathers; the download's sync
    /// completes both events, so reading them waits for nothing.
    /// Also returns the KV tokens the step's attention read, whether a graph replayed, and whether
    /// this step created a graph (capture warm-up).
    fn execute(&mut self, plan: &StepPlan) -> Result<(Option<u64>, u64, bool, bool), KError> {
        #[cfg(any(test, debug_assertions, feature = "fault-inject"))]
        crate::fault::step(self.step)?;
        // Separate profiling runs only: CPU wall intervals include API blocking, not GPU kernel
        // duration. The ordinary timing path does not read clocks or print this diagnostic.
        static PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let profile =
            *PROFILE.get_or_init(|| std::env::var("LARCH_PROFILE_STEPS").is_ok_and(|v| v == "1"));
        let profile_start = profile.then(Instant::now);
        let mut profile_marks = [0u64; 6];
        let mark = |marks: &mut [u64; 6], i: usize| {
            if let Some(start) = profile_start {
                marks[i] = start.elapsed().as_nanos() as u64;
            }
        };
        let splits = self.attn_splits();
        let as_check = plan.guessed > 0 || self.decodes_as_check(plan);
        let replay = self.value(knob_id::DECODE_GRAPHS) != 0.0;
        let mixed = (
            self.value(knob_id::PREFILL_CHUNK_TOKENS) as usize,
            self.value(knob_id::MAX_BATCH_SEQS) as usize,
        );
        let arena = self.s.capture_arena_held();
        let loaded = self.loaded.as_mut().expect("loaded engine");
        let backend = loaded.info.caps.backend;
        let hybrid = loaded.model.state_slots() > 0;
        // Stale splits, and mixed shapes the knobs no longer produce, free their quota bytes.
        loaded.graphs.retain(|k, _| {
            k.splits == splits
                && (k.prefill == 0
                    || k.checks != 0
                    || mixed_key(backend, mixed, k.decode, Some(k.prefill), k.logits, splits)
                        == Some(*k))
        });
        let Staging {
            rows,
            packed,
            params,
            out,
            sampled,
            taps,
            layers,
            tap_out,
            untapped,
            drafted,
            tap_rows: tapped_rows,
        } = &mut loaded.staging;
        assert!(
            rows.len() <= loaded.logit_rows,
            "logits rows past the scratch"
        );
        let nd = plan.decode.len();
        // Only while an activations request is admitted: the batch row of each one's last prompt
        // token, when its last chunk runs in this step (prefill rows follow the decode rows: a
        // tapped step is never padded). The vectors keep their capacity across steps.
        let n_taps = if self.sched.tapping() {
            taps.clear();
            layers.clear();
            *untapped = false;
            let mut end = nd;
            for &(id, start, len) in &plan.prefill {
                end += len;
                let r = self.sched.req(id).expect("planned request");
                let before = taps.len();
                if r.activations.is_empty() {
                } else if r.tap_rows.is_empty() {
                    if start + len == r.seq.tokens.len() {
                        taps.push((id, end as i32 - 1));
                    }
                } else {
                    let (lo, hi) = (start as u32, (start + len) as u32);
                    let at = |p: &u32| {
                        (lo..hi)
                            .contains(p)
                            .then(|| (id, (end - len) as i32 + (p - lo) as i32))
                    };
                    taps.extend(r.tap_rows.iter().filter_map(at));
                }
                if taps.len() > before {
                    layers.extend_from_slice(&r.activations);
                }
            }
            layers.sort_unstable();
            layers.dedup();
            taps.len()
        } else {
            0
        };
        // An activations tap writes rows the step's batch picks, so its step runs eager; so does
        // a step under router deltas. A hybrid decode step pads only onto unheld state slots, one
        // per padding row, so padding never touches a live sequence's state. A guessing hybrid's
        // captures read their state slots from a table the model writes before every replay
        // (Model::route_table), so a rewind's slot moves are data; only CUDA replays that path.
        // ponytail: on HIP and Metal every step of a guessing hybrid runs eager, k=0 too, so a load
        // whose picks are nearly all k=0 loses replay for no speculation and `Costs` cannot see it
        // (eager against eager); key plain steps on no live slot being off home if a bench there
        // shows that gap.
        // ponytail: a hybrid's mixed shapes run eager (only decode was lifted); capture them when
        // a hybrid's mixed steps show in a prefill profile.
        // ponytail: deltas and adapters run eager (their buffers are the version's, with no host
        // sync); capture them with the version in the graph key when a fully resident MoE shows the
        // eager cost (a host-expert layer refuses them too: the target MoE on 16 GB measured no gap).
        // A check step's chunks are all checks (the scheduler keeps prompts out of it), which the
        // model computes as decode rows at decode's `splits`. It captures at its exact shape on the
        // second hit; per-chunk lengths and row values are data the replay reads.
        // A drafter taps every row of every step but one that taps activations, whose sequences
        // then lose their draft context (see `Draft::observe`). Its taps land in fixed buffers
        // whatever it taps, so a pure decode step captures them; its graphs get their own keys.
        // ponytail: one Taps a forward; merge the two when activations requests meet a drafter.
        *drafted = loaded.draft.is_some() && n_taps == 0;
        // A step feeding image rows runs eager: its rows come from the projector, not the inputs a
        // capture froze. A prompt's last pending row runs as a decode row, so an image-only
        // request's last image row is one.
        let decoded = plan.decode.iter().map(|&id| {
            let n_kv = self.sched.req(id).expect("planned request").seq.n_kv;
            (id, n_kv, 1)
        });
        let images = plan
            .prefill
            .iter()
            .copied()
            .chain(decoded)
            .any(|(id, start, len)| {
                let r = self.sched.req(id).expect("planned request");
                !r.images.is_empty()
                    && r.seq.tokens[start..start + len]
                        .iter()
                        .any(|&t| t & IMAGE_TOKEN != 0)
            });
        let key = if !replay
            || n_taps > 0
            || (hybrid && self.opts.guessing() && backend != Backend::Cuda)
            || self.sched.version != 0
            || (*drafted && backend != Backend::Cuda)
            || images
        {
            None
        } else if plan.guessed > 0 {
            let np = plan.prefill.iter().map(|p| p.2).sum();
            let chunks = plan.prefill.len();
            check_key(backend, nd, np, chunks, rows.len(), splits, *drafted)
        } else if plan.prefill.is_empty() {
            let bucket = graph_decode_bucket(backend, nd).min(loaded.dims.seqs);
            (!hybrid || self.sched.free_state_slots().len() >= bucket - nd).then_some(GraphKey {
                decode: bucket,
                prefill: 0,
                logits: bucket,
                splits,
                taps: *drafted,
                checks: 0,
            })
        } else if loaded.mixed_graphs && loaded.info.caps.graph_replay && !hybrid && !*drafted {
            let prefill = (plan.prefill.len() == 1).then(|| plan.prefill[0].2);
            mixed_key(backend, mixed, nd, prefill, rows.len(), splits)
        } else {
            None
        };
        let (mut used, mut check_held) = (0, 0);
        for (k, c) in &loaded.graphs {
            if let Capture::Graph(.., bytes) = c {
                used += *bytes;
                check_held += if k.checks != 0 { *bytes } else { 0 };
            }
        }
        let mut now = Refusal {
            free: loaded.graph_bytes.saturating_sub(used),
            arena,
            step: self.step,
            again: false,
        };
        // A plain decode key counts what check keys hold as free: they give it back (eviction
        // below), so it is never refused because of them.
        // ponytail: check keys never evict each other, so the first shapes keep their slots; LRU
        // among them too when a bench shows late check shapes refused while early ones go cold.
        let plain = key.is_some_and(|k| k.prefill == 0);
        let room = Refusal {
            free: now.free + if plain { check_held } else { 0 },
            ..now
        };
        // Replay a captured shape or capture an admitted one; anything else runs eager at its
        // exact rows. A refusal is checked here, so a quota-less step never pads.
        let (mut graph, mut capture) = (false, false);
        // A capture that runs an owed retry owes none if it is refused again.
        let mut retrying = false;
        if let Some(k) = key {
            retrying = owed_retry(loaded.graphs.get(&k), &room);
            if retrying && graph_diagnostics() {
                eprintln!(
                    "larch_graph_retry,{},{},{},{}",
                    k.decode, k.prefill, k.logits, self.step
                );
            }
            match loaded.graphs.get(&k) {
                Some(Capture::Graph(..)) => graph = true,
                Some(Capture::Refused(r)) if !r.retry(&room) => {}
                None if k.prefill != 0 => {
                    loaded.graphs.insert(k, Capture::Seen);
                }
                // A capture is charged at least the floor, so a quota short of it never keeps one.
                _ if room.free < GRAPH_BYTES => {
                    loaded.graphs.insert(k, Capture::Refused(room));
                }
                _ => (graph, capture) = (true, true),
            }
            if capture && plain {
                let (graphs, used) = (&mut loaded.graphs, &mut loaded.check_used);
                now.free = evict_checks(graphs, used, now.free, GRAPH_BYTES);
            }
            if graph && k.checks != 0 {
                loaded.check_used.insert(k, self.step);
            }
        }
        // 9.5 step 2: a pure decode step replays its bucket's graph, padding where the bucket needs it
        // with the table at full width, and every replay of a bucket reads its inputs at the offsets
        // and shapes its capture froze. Padding rows write no KV (slot -1) and attend to nothing
        // (length 0); their logits are computed and never sampled.
        let pad = key
            .filter(|k| graph && k.prefill == 0)
            .map_or(0, |k| k.decode - nd);
        let seqs = || {
            plan.decode
                .iter()
                .copied()
                .chain(plan.prefill.iter().map(|p| p.0))
        };
        let req = |id: u64| self.sched.req(id).expect("planned request");
        // Decode attention picks its splits from the table width, so where graphs can replay every
        // step sends the replay's full width: eager and replayed steps then compute the same bits.
        let width = if graph || loaded.info.caps.graph_replay {
            loaded.dims.max_blocks
        } else {
            seqs()
                .map(|id| req(id).seq.blocks.len())
                .max()
                .unwrap_or(1)
                .max(1)
        };
        let token_rows = nd + pad + plan.prefill.iter().map(|p| p.2).sum::<usize>();
        *tapped_rows = token_rows;
        let seq_rows = nd + pad + plan.prefill.len();
        let penalty_on = rows
            .iter()
            .any(|l| req(l.id).sampling.repeat_penalty != 1.0);
        // The ids whose exact logits each row gathers, in row order: the next fed token of a
        // `token_loss` row, then a sampled row's score ids.
        let next = |l: &Logit| {
            let r = req(l.id);
            (l.observe && r.score_next)
                .then(|| r.seq.tokens.get(l.pos + 1).copied())
                .flatten()
        };
        let gathers = |l: &Logit| {
            usize::from(next(l).is_some())
                + if l.sample {
                    req(l.id).gathered().len()
                } else {
                    0
                }
        };
        let scored: usize = rows.iter().map(gathers).sum();
        // Qwen 3.5 (M-RoPE, a projector with grids): each row's rope positions from its KV index
        // and the images before it ([`mrope_at`]). A step feeding image rows passes four a row; any
        // other the text position alone, which ropes the same, so decode graphs keep their shapes.
        let grids = (loaded.model.config().rope.sections)
            .and(loaded.vision.as_ref().map(|v| v.config))
            .filter(|c| matches!(c, ProjectorConfig::Qwen(_)));
        let multi = images && grids.is_some();
        let lengths = [
            token_rows,
            if multi { 4 * token_rows } else { token_rows },
            token_rows,
            seq_rows * width,
            nd + pad,
            plan.prefill.len(),
            plan.prefill.len(),
            rows.len() + pad,
            if penalty_on {
                rows.len() * PENALTY_WINDOW
            } else {
                0
            },
            scored,
        ];
        // A hybrid model's state slots and rows per sequence lead, so a bucket's replay finds them
        // where its capture did whatever the penalty and score rows; the tap rows follow the scores.
        let mut at = [if hybrid { 2 * seq_rows } else { 0 }; 10];
        for i in 1..at.len() {
            at[i] = at[i - 1] + lengths[i - 1];
        }
        let tap_rows = at[9] + lengths[9];
        packed.resize(tap_rows + n_taps, 0);
        packed.fill(0);
        packed[at[2]..at[2] + lengths[2]].fill(-1);
        packed[at[8]..at[9]].fill(-1);
        let mut row = 0;
        // (batch row, request, position) of each image row; its token is the image's, not the vocabulary's.
        let mut image_rows: Vec<(usize, u64, usize)> = Vec::new();
        let mut feed = |id: u64, p: usize| {
            let s = &req(id).seq;
            let t = s.tokens[p];
            packed[at[0] + row] = if t & IMAGE_TOKEN != 0 {
                image_rows.push((row, id, p));
                0
            } else {
                t as i32
            };
            match grids {
                None => packed[at[1] + row] = p as i32,
                Some(c) => {
                    let grid =
                        |sp: &crate::ImageSpan| c.grid_for(sp.image.width(), sp.image.height());
                    let four = mrope_at(&req(id).images, grid, p);
                    if multi {
                        packed[at[1] + 4 * row..][..4].copy_from_slice(&four);
                    } else {
                        packed[at[1] + row] = four[0];
                    }
                }
            }
            packed[at[2] + row] = s.slot(p);
            row += 1;
        };
        for &id in &plan.decode {
            feed(id, req(id).seq.n_kv);
        }
        for &(id, start, len) in &plan.prefill {
            for p in start..start + len {
                feed(id, p);
            }
        }
        // ponytail: the whole block table is re-sent each step (at most 64 rows × max_blocks on a
        // graph-capable backend); upload deltas to a resident table (9.4) if the upload shows in a
        // decode profile.
        for (i, id) in seqs().enumerate() {
            for (j, &block) in req(id).seq.blocks.iter().enumerate() {
                packed[at[3] + i * width + j] = block as i32;
            }
        }
        let mut kv_tokens = 0;
        for (i, &id) in plan.decode.iter().enumerate() {
            let len = req(id).seq.n_kv + 1;
            packed[at[4] + i] = len as i32;
            kv_tokens += len as u64;
        }
        for (i, &(_, start, len)) in plan.prefill.iter().enumerate() {
            packed[at[5] + i] = start as i32;
            packed[at[6] + i] = len as i32;
            kv_tokens += (start + len) as u64;
        }
        params.clear();
        let mut score_at = at[9];
        for (i, l) in rows.iter().enumerate() {
            packed[at[7] + i] = l.row as i32;
            let r = req(l.id);
            if let Some(id) = next(l) {
                packed[score_at] = id as i32;
                score_at += 1;
            }
            if l.sample {
                for (slot, &id) in packed[score_at..].iter_mut().zip(r.gathered()) {
                    *slot = id as i32;
                }
                score_at += r.gathered().len();
            }
            let on = r.sampling.repeat_penalty != 1.0;
            let t = &r.seq.tokens[..=l.pos];
            let recent = &t[t.len().saturating_sub(PENALTY_WINDOW)..];
            if on {
                // An image row's id (`IMAGE_TOKEN`, bit 31) packs negative, and both sample_prep
                // kernels skip negative ids, so image rows are never penalised as tokens.
                for (j, &token) in recent.iter().enumerate() {
                    packed[at[8] + i * PENALTY_WINDOW + j] = token as i32;
                }
            }
            params.push(SampleParams {
                temperature: r.sampling.temperature,
                // Every banned candidate may outrank the greedy pick; top logprobs still include
                // them, and observations, stochastic sampling and grammar filters retain all 64
                // candidates.
                top_k: if !r.observe
                    && r.sampling.temperature <= 0.0
                    && !r.grammar.as_ref().is_some_and(|g| g.constrained())
                {
                    let banned = if r.ignore_eos { r.stop_tokens.len() } else { 0 };
                    usize::from(r.top_n())
                        .max(banned.saturating_add(1))
                        .min(CANDIDATES) as u32
                } else {
                    0
                },
                repeat_penalty: r.sampling.repeat_penalty,
                penalty_window: if on { recent.len() as u32 } else { 0 },
            });
        }
        for i in 0..pad {
            packed[at[7] + rows.len() + i] = (nd + i) as i32;
        }
        for (slot, t) in packed[tap_rows..].iter_mut().zip(&taps[..n_taps]) {
            *slot = t.1;
        }
        if hybrid {
            let lens = plan
                .decode
                .iter()
                .map(|_| 1)
                .chain(plan.prefill.iter().map(|p| p.2));
            for (i, (id, len)) in seqs().zip(lens).enumerate() {
                packed[i] = req(id).state_slot.expect("a running sequence") as i32;
                packed[seq_rows + i] = len as i32;
            }
            // The kernels take distinct slots and one row a slot, so each padding row gets its own.
            let free = self.sched.free_state_slots();
            for i in nd..nd + pad {
                packed[i] = free[i - nd] as i32;
                packed[seq_rows + i] = 1;
            }
        }
        // A graph freezes offsets and shapes, so its table keeps the full context width.
        let shapes = [
            [token_rows as i64, 0],
            [token_rows as i64, if multi { 4 } else { 0 }],
            [token_rows as i64, 0],
            [seq_rows as i64, width as i64],
            [(nd + pad) as i64, 0],
            [plan.prefill.len() as i64, 0],
            [plan.prefill.len() as i64, 0],
            [(rows.len() + pad) as i64, 0],
            [rows.len() as i64, PENALTY_WINDOW as i64],
        ];
        mark(&mut profile_marks, 0);
        let before = Event::record(&self.s)?;
        self.s.upload(&loaded.input, 0, packed)?;
        // A sequence's state starts zeroed at position 0, a first start or a recompute.
        for &(id, ..) in plan.prefill.iter().filter(|p| hybrid && p.1 == 0) {
            let slot = req(id).state_slot.expect("a running sequence");
            loaded.model.reset_state(&self.s, slot)?;
        }
        mark(&mut profile_marks, 1);
        let t = |i: usize| {
            let ndim = if i == 3 || i == 8 || (i == 1 && multi) {
                2
            } else {
                1
            };
            Tensor::new(
                &loaded.input,
                (at[i] * 4) as u64,
                DType::I32,
                &shapes[i][..ndim],
            )
        };
        let state = if hybrid {
            let at = |i: usize| (i * seq_rows * 4) as u64;
            let n = [seq_rows as i64];
            Some(StateBatch {
                slots: Tensor::new(&loaded.input, at(0), DType::I32, &n)?,
                q_lens: Tensor::new(&loaded.input, at(1), DType::I32, &n)?,
            })
        } else {
            None
        };
        let batch = StepBatch {
            token_ids: t(0)?,
            positions: t(1)?,
            slot_ids: t(2)?,
            decode_rows: (nd + pad) as u32,
            prefill_rows: (token_rows - nd - pad) as u32,
            block_table: t(3)?,
            seq_lens: t(4)?,
            prefill_q_start: t(5)?,
            prefill_q_len: t(6)?,
            logits_rows: t(7)?,
            attn_splits: splits,
            check: as_check,
        };
        let mc = loaded.model.config();
        let (vocab, d_model) = (mc.vocab, mc.d_model as usize);
        let pool = KvPool {
            k: &loaded.k,
            v: &loaded.v,
            n_blocks: self.sched.blocks.total() as u32,
            block_tokens: BLOCK_TOKENS as u32,
            n_layers: mc.kv_layers,
            n_kv_heads: mc.n_kv_heads,
            head_dim: mc.head_dim,
            dtype: KV_DTYPE,
        };
        let m = (rows.len() + pad) as i64;
        let logits = Tensor::new(&loaded.logits, 0, DType::F32, &[m, i64::from(vocab)])?;
        // The taps' rows land in `samp` after the candidates and scores, where the step's one
        // download reads them; at most `TAP_SEATS` requests of at most every layer, as loaded.
        let candidates = rows.len() * (2 * CANDIDATES + 1);
        let tap_len = layers.len() * n_taps * d_model;
        // ponytail: a step whose taps overflow the scratch (a many-row request beside a multi-layer
        // one) runs untapped and ends them NO_TAPS; split the scratch per request if that shows.
        let tap = if n_taps > 0 && tap_len <= loaded.dims.tap_floats {
            *tap_out = candidates + scored;
            assert!(n_taps <= loaded.dims.tap_seats, "taps past the scratch");
            Some(Taps {
                layers: &layers[..],
                final_state: false,
                rows: Tensor::new(
                    &loaded.input,
                    (tap_rows * 4) as u64,
                    DType::I32,
                    &[n_taps as i64],
                )?,
                out: Tensor::new(
                    &loaded.samp,
                    (*tap_out * 4) as u64,
                    DType::F32,
                    &[(layers.len() * n_taps) as i64, d_model as i64],
                )?,
            })
        } else if let Some(d) = loaded.draft.as_ref().filter(|_| *drafted) {
            Some(d.taps(token_rows)?)
        } else {
            *untapped = n_taps > 0;
            None
        };
        let mut captured = false;
        // A capture to charge once its first replay below has run: its key, the device bytes free
        // before it, the arena it allocated, and what it saw.
        let mut measure = None;
        if capture {
            let k = key.expect("a capture has a shape");
            let free = self.s.device().props()?.vram_free;
            let model = &mut loaded.model;
            let t = tap.as_ref();
            let g = Graph::capture_pieces(&self.s, |s| match &state {
                Some(st) => model.forward_state(s, &batch, st, &pool, &logits, t),
                None => model.forward_taps(s, &batch, &pool, &logits, t),
            })?;
            captured = g.is_some();
            let arena = self.s.capture_arena_held();
            // Allocated now (0 if held before): outside free memory, so observation no longer
            // reserves it, and reserved apart from the quota, so the capture is not charged it.
            let arena_bytes = if arena {
                std::mem::take(&mut loaded.graph_arena_bytes)
            } else {
                0
            };
            // A failed capture refuses with what it saw; this step runs eagerly below.
            if captured {
                let seen = Refusal {
                    arena,
                    again: !retrying,
                    ..now
                };
                measure = Some((k, free, arena_bytes, seen));
            }
            if !captured && graph_diagnostics() {
                let why = larch_kernels::last_error();
                eprintln!(
                    "larch_graph_refused,{},{},{},{},{why}",
                    k.decode, k.prefill, k.logits, k.checks
                );
            }
            let refusal = Refusal { arena, ..room };
            loaded.graphs.insert(
                k,
                // Charged the estimate until measured below, should the step fail before that.
                g.map_or(Capture::Refused(refusal), |(cut, last)| {
                    Capture::Graph(cut, last, GRAPH_BYTES)
                }),
            );
        }
        let cached = key
            .filter(|_| graph)
            .and_then(|k| match loaded.graphs.get(&k) {
                Some(Capture::Graph(cut, last, _)) => Some((cut, last)),
                _ => None,
            });
        // A plain decode key without a graph to replay was refused: this step runs eager, counted,
        // and the process's first such step says why. A capture that failed did so this step.
        let graph_capable = loaded.info.caps.graph_replay;
        if let Some(k) = key.filter(|k| k.prefill == 0 && cached.is_none() && graph_capable) {
            let before = self.sh.eager_decode.fetch_add(1, Ordering::Relaxed);
            if let Some(Capture::Refused(r)) = loaded.graphs.get(&k)
                && says_eager(before, || std::env::var("LARCH_FIT_DIAGNOSTICS").ok())
            {
                let failed = (capture && !captured).then(larch_kernels::last_error);
                let (free, quota) = (room.free, loaded.graph_bytes);
                eprintln!(
                    "{}",
                    eager_line(k.decode, r, free, quota, failed.as_deref())
                );
            }
        }
        mark(&mut profile_marks, 2);
        let runs = match &mut loaded.vision {
            Some(v) if !image_rows.is_empty() => image_runs(
                v,
                &self.s,
                &image_rows,
                |id| self.sched.req(id).expect("planned request"),
                d_model,
                &self.sh.tiles_encoded,
            )?,
            _ => Vec::new(),
        };
        // Every row's token, as `token_ids` holds them (an image row's as 0): a host embedding's
        // rows, gathered here so neither a replay nor an eager pass reads them back from the device.
        let ids = &packed[at[0]..at[0] + token_rows];
        loaded.model.embed_tokens(&self.s, ids)?;
        let replayed = if !runs.is_empty() {
            let v = loaded
                .vision
                .as_ref()
                .expect("image rows need the projector");
            let runs = runs
                .iter()
                .map(|&(row, n)| {
                    Ok(EmbedRun {
                        row: row as u32,
                        table: Tensor::new(
                            &v.rows,
                            (row * d_model * 4) as u64,
                            DType::F32,
                            &[n as i64, d_model as i64],
                        )?,
                        ids: v.model.ids(n as u32)?,
                    })
                })
                .collect::<Result<Vec<_>, KError>>()?;
            (loaded.model).forward_embeds(
                &self.s,
                &batch,
                state.as_ref(),
                &pool,
                &logits,
                tap.as_ref(),
                &runs,
            )?;
            false
        } else {
            match (cached, &state) {
                (Some((cut, last)), _) => {
                    // ponytail: route_table's two uploads per replay; fold the physical slots into
                    // the step's upload when a decode profile shows them.
                    if hybrid && self.opts.guessing() {
                        let (slots, lens) = (&packed[..seq_rows], &packed[seq_rows..2 * seq_rows]);
                        loaded.model.route_table(&self.s, slots, lens)?;
                    }
                    for (g, tag) in cut {
                        g.launch(&self.s)?;
                        // Padding rows trail the live ones; their experts' output is never read.
                        loaded
                            .model
                            .run_host(&self.s, *tag, (token_rows - pad) as u32)?;
                    }
                    last.launch(&self.s).map(|()| true)?
                }
                (None, Some(st)) => {
                    let t = tap.as_ref();
                    (loaded.model).forward_state(&self.s, &batch, st, &pool, &logits, t)?;
                    false
                }
                (None, None) if tap.is_none() => loaded
                    .model
                    .forward(&self.s, &batch, &pool, &logits)
                    .map(|()| false)?,
                (None, None) => {
                    let ran =
                        (loaded.model).forward_taps(&self.s, &batch, &pool, &logits, tap.as_ref());
                    // The trait's default refuses taps before any work: the step runs without them,
                    // and its tapped requests end with NO_TAPS.
                    if ran == Err(KError::Unsupported) {
                        loaded.model.forward(&self.s, &batch, &pool, &logits)?;
                        (*untapped, *drafted) = (true, false);
                    } else {
                        ran?;
                    }
                    false
                }
            }
        };

        // ponytail: free memory is device-wide, so a lazy module load, another process or WSL's
        // swinging reading inside the window over-charges a capture. A charge past the quota left
        // is read once more after a sync (what a sync settles, the driver's deferred frees and the
        // mempool, it settles in full, and a permanent allocation reads the same again). A second
        // charge past the quota is held if free memory still covers the headroom and the quota
        // left ([`device_holds`]), else refused, retried once [`RETRY_STEPS`] later, then eager
        // until the quota grows. Count the graph's own bytes when a larch_graph_capture line
        // shows a refused shape charged past the estimate on an otherwise idle device.
        if let Some((k, free, arena_bytes, seen)) = measure {
            // The replay ran: a failed reading charges the floor and keeps the step's result.
            let after = self.s.device().props().map_or(free, |p| p.vram_free);
            let bytes = remeasured_charge((seen.free, free, arena_bytes), after, || {
                self.s.sync()?;
                self.s.device().props().map(|p| p.vram_free)
            });
            let mut seen = seen;
            if k.prefill == 0 {
                let (graphs, used) = (&mut loaded.graphs, &mut loaded.check_used);
                seen.free = evict_checks(graphs, used, seen.free, bytes);
            }
            let Some(Capture::Graph(cut, _, charged)) = loaded.graphs.get_mut(&k) else {
                unreachable!("a capture this step stored")
            };
            let pieces = cut.len() + 1;
            // Free memory after the re-read, which a charge past the floor was measured from.
            let after_sync = free.saturating_sub(arena_bytes).saturating_sub(bytes);
            let over = seen.over(bytes);
            let refusal =
                over.filter(|_| !device_holds(seen.free, after_sync, loaded.headroom_bytes));
            if graph_diagnostics() {
                eprintln!(
                    "larch_graph_capture,{},{},{},{pieces},{bytes},{},{},{free},{after},{arena_bytes}",
                    k.decode,
                    k.prefill,
                    k.logits,
                    seen.free,
                    refusal.is_none()
                );
            }
            match refusal {
                // Over the quota left: dropped once the replay above has run (HIP documents no
                // deferred destroy of a launched graph), and the shape runs eager until the quota
                // has room for it.
                Some(r) => {
                    self.s.sync()?;
                    loaded.graphs.insert(k, Capture::Refused(r));
                }
                None if over.is_some() => *charged = GRAPH_BYTES,
                None => *charged = bytes,
            }
        }
        mark(&mut profile_marks, 3);
        let n = rows.len() as i64;
        let c = CANDIDATES as i64;
        let vals = Tensor::new(&loaded.samp, 0, DType::F32, &[n, c])?;
        let ids = Tensor::new(&loaded.samp, (n * c * 4) as u64, DType::I32, &[n, c])?;
        let lse = Tensor::new(&loaded.samp, (n * c * 8) as u64, DType::F32, &[n])?;
        let real = logits.rows(0, n)?;
        let penalty = if penalty_on { Some(t(8)?) } else { None };
        ops::sample_prep(&self.s, &real, params, penalty.as_ref(), &vals, &ids, &lse)?;
        // A score id may sit outside the top 64, so each scored row's logits, viewed as a
        // [vocab, 1] table, are gathered into `samp` after the LSE, where the one download reads
        // them.
        // ponytail: one gather launch per scored row, as an index over every row's logits could
        // pass i32; one launch per step from row-relative indices if score batches show in the
        // step profile.
        let mut k = 0;
        for (i, l) in rows.iter().enumerate() {
            if k == scored {
                break;
            }
            let len = gathers(l);
            if len == 0 {
                continue;
            }
            let at_row = (i * vocab as usize * 4) as u64;
            let table = Tensor::new(&loaded.logits, at_row, DType::F32, &[i64::from(vocab), 1])?;
            let named = Tensor::new(
                &loaded.input,
                ((at[9] + k) * 4) as u64,
                DType::I32,
                &[len as i64],
            )?;
            let gathered = Tensor::new(
                &loaded.samp,
                ((candidates + k) * 4) as u64,
                DType::F32,
                &[len as i64, 1],
            )?;
            ops::embed(&self.s, &table, &named, &gathered)?;
            k += len;
        }
        let after = Event::record(&self.s)?;
        out.resize(candidates + scored + tap_len, 0);
        download(&self.s, &loaded.samp, out)?;
        let gpu_ns = gpu_time(before.elapsed_ns(&after))?;
        mark(&mut profile_marks, 4);

        let c = CANDIDATES;
        let (vals, rest) = out.split_at(rows.len() * c);
        let (ids, rest) = rest.split_at(rows.len() * c);
        let (lse, rest) = rest.split_at(rows.len());
        let scores = &rest[..scored];
        let (vals, ids, lse, scores): (&[f32], &[i32], &[f32], &[f32]) = (
            bytemuck::cast_slice(vals),
            bytemuck::cast_slice(ids),
            bytemuck::cast_slice(lse),
            bytemuck::cast_slice(scores),
        );
        let mut scores = scores.iter();
        sampled.clear();
        let mut stopped = None;
        for (i, l) in rows.iter().enumerate() {
            let row = Row {
                vals: &vals[i * c..][..c],
                ids: &ids[i * c..][..c],
                lse: lse[i],
            };
            let r = self.sched.req_mut(l.id).expect("planned request");
            let has_next = l.observe && r.score_next && l.pos + 1 < r.seq.tokens.len();
            if l.observe {
                let next = has_next
                    .then(|| scores.next().map(|s| s - row.lse))
                    .flatten();
                if r.observe {
                    let pos = l.pos as u32;
                    r.completion
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .position(Position {
                            pos,
                            top: top(&row, c),
                            next,
                        });
                }
                r.next_obs = l.pos + 1;
            }
            // A check's rows are its sequence's, in order: row j checks guess j, and the first row
            // that does not keep its guess (or keeps a stop token) emits the sequence's last token
            // this step. Grammar sequences never guess.
            if l.sample && stopped != Some(l.id) {
                let guess = (r.guesses > 0)
                    .then(|| r.seq.tokens.get(l.pos + 1).copied())
                    .flatten();
                let top_n = r.top_n();
                let t = match r.grammar.as_mut().filter(|g| g.constrained()) {
                    Some(g) => {
                        let (gathered, rest) = scores.as_slice().split_at(g.gather().len());
                        scores = rest.iter();
                        let t = &r.seq.tokens[..=l.pos];
                        let recent = &t[t.len().saturating_sub(PENALTY_WINDOW)..];
                        g.pick(
                            &row,
                            gathered,
                            &r.sampling,
                            top_n,
                            recent,
                            &r.stop_tokens,
                            &mut r.rng,
                        )
                    }
                    None => {
                        let ban: &[u32] = if r.ignore_eos { &r.stop_tokens } else { &[] };
                        let mut t = check(&row, &r.sampling, top_n, ban, guess, &mut r.rng);
                        if !r.score_ids.is_empty() {
                            t.top = (r.score_ids.iter())
                                .zip(scores.by_ref())
                                .map(|(&id, &logit)| (id, logit - row.lse))
                                .collect();
                        }
                        t
                    }
                };
                if let Some(g) = &mut r.grammar {
                    g.advance(t.id, &r.stop_tokens);
                }
                let stops = !r.ignore_eos && r.stop_tokens.contains(&t.id);
                if r.guesses > 0 && (guess != Some(t.id) || stops) {
                    stopped = Some(l.id);
                }
                sampled.push((l.id, t));
            }
        }
        mark(&mut profile_marks, 5);
        if profile {
            let intervals: Vec<_> = std::iter::once(profile_marks[0])
                .chain(profile_marks.windows(2).map(|w| w[1] - w[0]))
                .collect();
            eprintln!(
                "larch_step_profile,{},{},{},{},{},{},{},{},{},{},{},{}",
                self.step,
                plan.decode.len(),
                plan.prefill.len(),
                plan.prefill.iter().map(|p| p.2).sum::<usize>(),
                intervals[0],
                intervals[1],
                intervals[2],
                intervals[3],
                intervals[4],
                intervals[5],
                u8::from(replayed),
                u8::from(captured)
            );
        }
        Ok((gpu_ns, kv_tokens, replayed, captured))
    }

    fn sample_record(
        &self,
        plan: &StepPlan,
        dur: u64,
        gpu_ns: Option<u64>,
        kv_tokens: u64,
        replayed: bool,
        captured: bool,
    ) -> EngineSample {
        let loaded = self.loaded.as_ref().expect("loaded engine");
        // Past 16 rows the model dequantizes each weight to f16 and runs a gemm over it (9.5), so the
        // quantized bytes undercount the traffic: Missing there.
        // ponytail: until the model reports the bytes it moved (9.5 StepOutput.bytes_moved).
        let rows = plan.decode.len() + plan.prefill.iter().map(|p| p.2).sum::<usize>();
        // Bytes per ns is GB/s.
        let bytes = loaded
            .weights_read
            .filter(|_| rows <= 16)
            .map(|w| w + kv_tokens * loaded.kv_token_bytes);
        let gbps = bytes
            .zip(gpu_ns.filter(|&ns| ns > 0))
            .map(|(b, ns)| b as f32 / ns as f32);
        let peak = loaded.info.deployment.peak_gbps.get().filter(|&p| p > 0.0);
        let caps = &loaded.info.caps;
        let (total, used) = (self.sched.blocks.total(), self.sched.blocks.used());
        let used_tokens = self.sched.used_tokens();
        let (nd, np, prefill_tokens) = plan.counts();
        let tot = self.sched.tot;
        let mut sample = EngineSample {
            step: self.step,
            dur_ns: dur,
            gpu_ns: gpu_ns.map_or(MU64::missing(), MU64::measured),
            host_ns: dur.saturating_sub(gpu_ns.unwrap_or(0)),
            phase: match (nd == 0, np == 0) {
                (false, true) => Phase::Decode,
                (true, false) => Phase::Prefill,
                _ => Phase::Mixed,
            }
            .raw(),
            n_decode_seqs: nd,
            n_prefill_seqs: np,
            n_prefill_tokens: prefill_tokens,
            n_running: self.sched.n_running() as u32,
            n_waiting: self.sched.n_waiting() as u32,
            n_preempted: plan.n_preempted,
            n_parked: self.sched.n_parked() as u32,
            parked_blocks: self.sched.parked_blocks() as u32,
            parked_reclaimed: plan.n_reclaimed,
            kv_blocks_total: total as u32,
            kv_blocks_used: used as u32,
            kv_block_tokens: BLOCK_TOKENS as u32,
            kv_tokens_used: used_tokens as u32,
            kv_frag: if used > 0 {
                MF32::measured(1.0 - used_tokens as f32 / (used * BLOCK_TOKENS) as f32)
            } else {
                MF32::measured(0.0)
            },
            kv_util: MF32::measured(used as f32 / total.max(1) as f32),
            tot_preemptions: tot.preemptions,
            tot_finished: tot.finished,
            tot_prompt_tokens: tot.prompt_tokens,
            tot_gen_tokens: tot.gen_tokens,
            tot_queue_ns: tot.queue_ns,
            tot_queued: tot.queued,
            tot_rejected: self.sh.rejected.load(Ordering::Relaxed),
            achieved_gbps: estimate(caps.achieved_gbps, gbps),
            hbm_util: estimate(caps.hbm_util, gbps.zip(peak).map(|(a, p)| a / p)),
            sm_occupancy: estimate(caps.sm_occupancy, None),
            graph_replayed: u8::from(replayed),
            tot_recomputed_tokens: tot.recomputed_tokens,
            cap_blocked_ready: plan.cap_blocked,
            ..bytemuck::Zeroable::zeroed()
        };
        sample.set_graph_captured(captured);
        sample
    }

    /// The preconditions only the engine can check: the scratch and the backend.
    fn dynamic_check(&mut self, knob: u16, value: f64) -> Result<Option<RejectReason>, KError> {
        let caps = self.sh.info.read().expect("model info").caps;
        Ok(match knob {
            knob_id::PREFILL_CHUNK_TOKENS if value > f64::from(caps.prefill_chunk_max) => {
                Some(RejectReason::ExceedsFit {
                    limit: f64::from(caps.prefill_chunk_max),
                })
            }
            knob_id::DECODE_GRAPHS if value != 0.0 && !caps.graph_replay => {
                Some(RejectReason::Unsupported)
            }
            knob_id::DECODE_GRAPHS if value != 0.0 => self.reserve_graphs()?,
            _ => None,
        })
    }

    /// A load with graphs off reserved no graph memory: switching them on takes their reserve from
    /// what is free beside the headroom, or is refused, never over-committed.
    fn reserve_graphs(&mut self) -> Result<Option<RejectReason>, KError> {
        let props = self.s.device().props()?;
        let held = self.s.capture_arena_held();
        let seqs = self.value(knob_id::MAX_BATCH_SEQS) as u32;
        let loaded = self.loaded.as_mut().expect("loaded engine");
        let seqs = servable_seats(loaded.model.state_slots(), seqs);
        if loaded.graph_bytes != 0 {
            return Ok(None);
        }
        // As fit sizes it at load: the current seats' buckets, and the arena unless it is held.
        let block = loaded.draft.as_ref().and_then(Draft::block);
        let need = graph_reserve(props.backend, seqs, check_spares(&self.opts, block))
            + mixed_reserve(&props, larch_kernels::routes().mixed_graphs);
        let arena = if held { 0 } else { graph_arena_reserve(&props) };
        let room = props
            .vram_free
            .saturating_sub(loaded.headroom_bytes + arena);
        let class = loaded.info.class;
        let quota = graph_quota(need, room, !class.dense && !class.experts_in_ram);
        if quota < need {
            return Ok(Some(RejectReason::ExceedsFit { limit: 0.0 }));
        }
        (loaded.graph_bytes, loaded.graph_arena_bytes) = (quota, arena);
        Ok(None)
    }

    /// A set on a knob of the running A/B ends it on arm A first, so the new value is not flipped
    /// away and its `previous` is arm A's value (the knob's value when the A/B started). A human or
    /// gait change to a live knob other than `guardrail_ratio` is then guarded (9.9).
    fn apply_set(&mut self, req: &KnobRequest, t: u64) -> Result<KnobReply, KError> {
        if let Some(reason) = self.dynamic_check(req.knob, req.value)? {
            let reply = KnobReply::Rejected { reason };
            self.activation(req.knob, req.value, req.trial_id, t, &reply);
            return Ok(reply);
        }
        if self.in_ab(req.knob) {
            self.end_ab(AbEnd::Stop);
        }
        let cause = match req.authority {
            Authority::Human => Cause::Human,
            Authority::Gait => Cause::Gait,
            Authority::Guardrail => Cause::Guardrail,
        };
        let reply = self.change(req.knob, req.value, cause, req.trial_id, t);
        let live = knob_spec(req.knob).is_some_and(|k| k.activation() == ActivationClass::Live);
        if let KnobReply::Applied { rev, .. } = reply
            && matches!(cause, Cause::Human | Cause::Gait)
            && live
            && req.knob != knob_id::GUARDRAIL_RATIO
        {
            self.guard.watch(Target::Rev {
                rev,
                knob: req.knob,
            });
        }
        Ok(reply)
    }

    /// Writes the ActivationEvent of a knob request's outcome.
    fn activation(&mut self, knob: u16, requested: f64, trial_id: u64, t: u64, reply: &KnobReply) {
        let current = self.cfg.values.get(knob as usize).copied().unwrap_or(0.0);
        let (result, reason, effective, step) = match *reply {
            KnobReply::Applied {
                effective, step, ..
            } => (ActivationResult::Applied, 0, effective, step),
            KnobReply::Rejected { reason } => (
                ActivationResult::Rejected,
                reason.code(),
                current,
                self.step,
            ),
            KnobReply::Indeterminate { .. } => return,
        };
        let event = ActivationEvent {
            knob,
            result: result.raw(),
            reason,
            requested,
            effective,
            t_request_ns: t,
            t_applied_ns: self.sh.now(),
            step,
            trial_id,
            ..bytemuck::Zeroable::zeroed()
        };
        self.push(Tier::T0, &event);
    }

    /// [`Self::revise`], kept for reverts. An undo (inverse, guardrail) closes any trial's run of
    /// revisions, so a trial's revert sees only the changes it made itself. The knob's older watches
    /// end: only its latest revision can be reverted.
    fn change(&mut self, knob: u16, value: f64, cause: Cause, trial_id: u64, t: u64) -> KnobReply {
        let reply = self.revise(knob, value, cause, trial_id, t);
        self.publish();
        if knob == knob_id::MAX_BATCH_SEQS && cause == Cause::Human {
            // A person's seats, not a tuner's: only they are worth a reload, and they end the
            // demand switch (one already waiting is cancelled).
            (self.partial.seats_asked, self.replan, self.auto) = (true, true, false);
        }
        if let KnobReply::Applied { rev, previous, .. } = reply {
            self.guard.unwatch(knob);
            let undo = matches!(cause, Cause::Inverse | Cause::Guardrail);
            self.history.push(Rev {
                rev,
                knob,
                previous,
                trial_id: if undo { 0 } else { trial_id },
                cause,
            });
        }
        reply
    }

    /// Records a new revision; it takes effect on the next step, whose EngineSample carries it. The
    /// caller publishes. An A/B's own revisions (flips, its end) come here directly and stay out of
    /// the history: the reverts and the guardrail see the values outside the A/B.
    fn revise(&mut self, knob: u16, value: f64, cause: Cause, trial_id: u64, t: u64) -> KnobReply {
        let previous = self.value(knob);
        if previous != value
            && matches!(
                knob,
                knob_id::ATTN_DECODE_SPLITS
                    | knob_id::DECODE_GRAPHS
                    | knob_id::PREFILL_CHUNK_TOKENS
                    | knob_id::MAX_BATCH_SEQS
                    | knob_id::SCHEDULE_POLICY
            )
        {
            self.sched.invalidate_prefix_cache();
        }
        let prev_rev = self.cfg.rev;
        self.cfg.values[knob as usize] = value;
        self.cfg.rev += 1;
        let rev = self.cfg.rev;
        let step = self.step + 1;
        let revision = ConfigurationRevision {
            rev,
            prev_rev,
            knob,
            cause: cause.raw(),
            requested: value,
            effective: value,
            previous,
            step_applied: step,
            trial_id,
            ..bytemuck::Zeroable::zeroed()
        };
        self.push(Tier::T0, &revision);
        let reply = KnobReply::Applied {
            rev,
            effective: value,
            previous,
            step,
            latency_us: (self.sh.now().saturating_sub(t) / 1000) as u32,
        };
        // A flip answers no request.
        if cause != Cause::AbFlip {
            self.activation(knob, value, trial_id, t, &reply);
        }
        reply
    }

    /// The declared inverse: restore the value before the revision, only while it is the knob's
    /// latest; a trial restores each knob it changed last to its value before the trial, and ends
    /// its running A/B on arm A with an inverse revision per knob. Gait may undo only revisions it
    /// made (D7: it cannot undo a human's quality change). A single revision of a knob in a running
    /// A/B is Busy: stop the A/B, or revert its trial.
    fn apply_revert(
        &mut self,
        target: RevertTarget,
        who: Authority,
        t: u64,
    ) -> Result<KnobReply, KError> {
        let latest = |h: &[Rev], knob: u16| h.iter().rposition(|r| r.knob == knob);
        // (knob, value to restore, trial, every undone revision was gait's)
        let undo: Vec<(u16, f64, u64, bool)> = match target {
            RevertTarget::Rev(rev) => {
                let Some(i) = self.history.iter().position(|r| r.rev == rev) else {
                    return Ok(KnobReply::Rejected {
                        reason: RejectReason::NothingToRevert,
                    });
                };
                let r = &self.history[i];
                if r.cause == Cause::Reload {
                    return Ok(KnobReply::Rejected {
                        reason: RejectReason::NothingToRevert,
                    });
                }
                let last = latest(&self.history, r.knob).expect("this one at least");
                if last != i {
                    return Ok(KnobReply::Rejected {
                        reason: RejectReason::Superseded {
                            by_rev: self.history[last].rev,
                        },
                    });
                }
                vec![(r.knob, r.previous, r.trial_id, r.cause == Cause::Gait)]
            }
            RevertTarget::Trial(0) => Vec::new(),
            RevertTarget::Trial(trial) => {
                let mut knobs: Vec<u16> = self
                    .history
                    .iter()
                    .filter(|r| r.trial_id == trial)
                    .map(|r| r.knob)
                    .collect();
                knobs.sort_unstable();
                knobs.dedup();
                knobs
                    .into_iter()
                    .filter_map(|knob| {
                        let ours: Vec<&Rev> =
                            self.history.iter().filter(|r| r.knob == knob).collect();
                        let run = ours
                            .iter()
                            .rev()
                            .take_while(|r| r.trial_id == trial)
                            .count();
                        let first = ours.len() - run;
                        let gait = ours[first..].iter().all(|r| r.cause == Cause::Gait);
                        (run > 0).then(|| (knob, ours[first].previous, trial, gait))
                    })
                    .collect()
            }
        };
        // Another trial's running A/B would flip the restored value away.
        let own = matches!(target, RevertTarget::Trial(trial) if self.ab.as_ref().is_some_and(|a| a.trial == trial));
        if !own && undo.iter().any(|u| self.in_ab(u.0)) {
            return Ok(KnobReply::Rejected {
                reason: RejectReason::Busy,
            });
        }
        if who == Authority::Gait
            && let Some(&(knob, ..)) = undo.iter().find(|u| !u.3)
        {
            let risk = knob_spec(knob).expect("registered").risk();
            return Ok(KnobReply::Rejected {
                reason: RejectReason::Unauthorized { risk },
            });
        }
        for &(knob, value, ..) in &undo {
            if let Some(reason) = self.dynamic_check(knob, value)? {
                return Ok(KnobReply::Rejected { reason });
            }
        }
        let ab = if own {
            self.end_ab(AbEnd::Revert)
        } else {
            None
        };
        Ok(undo
            .into_iter()
            .map(|(knob, value, trial, _)| self.change(knob, value, Cause::Inverse, trial, t))
            .last()
            .or(ab)
            .unwrap_or(KnobReply::Rejected {
                reason: RejectReason::NothingToRevert,
            }))
    }

    fn in_ab(&self, knob: u16) -> bool {
        self.ab
            .as_ref()
            .is_some_and(|ab| ab.knobs.iter().any(|k| k.0 == knob))
    }

    /// Starts an A/B on arm A, or stops one (9.9). The caller validated the command (see
    /// `EngineKnobs::ab`). The start writes an AbFlip revision per knob on arm A, so the A/B's first
    /// edge is in the ring and `arm_a_rev` is a revision of arm A.
    fn apply_ab(&mut self, cmd: AbCommand, t: u64) -> Result<AbReply, KError> {
        Ok(match cmd {
            AbCommand::Start {
                b,
                period_steps,
                max_steps,
                trial_id,
                ..
            } => {
                if self.ab.is_some() {
                    return Ok(AbReply::Rejected {
                        reason: RejectReason::Busy,
                    });
                }
                let mut knobs = Vec::new();
                for d in b.iter().filter(|d| d.knob != 0) {
                    if let Some(reason) = self.dynamic_check(d.knob, d.value)? {
                        let reply = KnobReply::Rejected { reason };
                        self.activation(d.knob, d.value, trial_id, t, &reply);
                        return Ok(AbReply::Rejected { reason });
                    }
                    knobs.push((d.knob, self.value(d.knob), d.value));
                }
                let start = self.step + 1;
                self.sh
                    .host
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .ab_done = None;
                self.set_arm(AbArm::A);
                for &(knob, a, _) in &knobs {
                    self.revise(knob, a, Cause::AbFlip, trial_id, t);
                }
                self.ab = Some(Ab {
                    trial: trial_id,
                    knobs,
                    period: u64::from(period_steps),
                    max_steps,
                    start,
                });
                self.publish();
                self.guard.watch(Target::Ab);
                AbReply::Started {
                    arm_a_rev: self.cfg.rev,
                    step: start,
                }
            }
            AbCommand::Stop { trial_id } => {
                if self.ab.as_ref().is_some_and(|ab| ab.trial == trial_id) {
                    self.end_ab(AbEnd::Stop);
                }
                match self
                    .sh
                    .host
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .ab_done
                {
                    Some((trial, steps, end)) if trial == trial_id => {
                        AbReply::Stopped { steps, end }
                    }
                    _ => AbReply::Rejected {
                        reason: RejectReason::NothingToRevert,
                    },
                }
            }
        })
    }

    /// At the boundary before each step: puts the step's arm in force (A for the first `period`
    /// steps, then B, alternating).
    fn ab_boundary(&mut self) {
        let Some(ab) = &self.ab else {
            return;
        };
        let n = self.step + 1 - ab.start;
        let arm = if (n / ab.period).is_multiple_of(2) {
            AbArm::A
        } else {
            AbArm::B
        };
        if arm.raw() == self.cfg.ab_arm {
            return;
        }
        let (trial, knobs) = (ab.trial, ab.knobs.clone());
        self.set_arm(arm);
        let now = self.sh.now();
        for (knob, a, b) in knobs {
            let value = if arm == AbArm::B { b } else { a };
            self.revise(knob, value, Cause::AbFlip, trial, now);
        }
        self.publish();
    }

    /// Ends the running A/B on arm A with a revision per knob (cause Guardrail on a trip, Inverse on
    /// a revert of its trial, else AbFlip), on arm A too, so the end and its trial are in the ring
    /// (9.8). The revisions carry no arm, like what follows them. Returns the last one's reply.
    fn end_ab(&mut self, end: AbEnd) -> Option<KnobReply> {
        let ab = self.ab.take()?;
        let cause = match end {
            AbEnd::Guardrail => Cause::Guardrail,
            AbEnd::Revert => Cause::Inverse,
            _ => Cause::AbFlip,
        };
        self.set_arm(AbArm::None);
        let now = self.sh.now();
        let mut last = None;
        for &(knob, a, _) in &ab.knobs {
            last = Some(self.revise(knob, a, cause, ab.trial, now));
        }
        self.publish();
        self.guard.unwatch_ab();
        let steps = (self.step + 1).saturating_sub(ab.start);
        let mut host = self.sh.host.lock().unwrap_or_else(|e| e.into_inner());
        if !host.failed {
            host.ab_done = Some((ab.trial, steps, end));
        }
        last
    }

    /// The guardrail's revert of a watched change, the knob's latest revision: its declared inverse,
    /// with cause Guardrail and the change's trial. An A/B running on the knob ends first, so it
    /// cannot flip the value back: cause Guardrail if it is the change's own trial, else Stop, as
    /// when a `set` supersedes it (its arm A ran the change, so the regression is not its own).
    fn guard_revert(&mut self, rev: u64) {
        let r = self
            .history
            .iter()
            .rfind(|r| r.rev == rev)
            .expect("a watched change is in the history");
        let (knob, previous, trial) = (r.knob, r.previous, r.trial_id);
        if self.in_ab(knob) {
            let own = self.ab.as_ref().is_some_and(|ab| ab.trial == trial);
            self.end_ab(if own { AbEnd::Guardrail } else { AbEnd::Stop });
        }
        let now = self.sh.now();
        self.change(knob, previous, Cause::Guardrail, trial, now);
    }
}

/// Row `p`'s M-RoPE positions (t, h, w, e) in a sequence whose prompt holds `images`, as llama.cpp's
/// mtmd helpers lay them out: row i of an image with a `grid` (nx, ny) at (p0, p0 + i / nx,
/// p0 + i % nx, 0) from the image's first position p0, and a text row at (q, q, q, 0), where q is
/// `p` less the positions the images before it saved (rows - max(nx, ny) each). An image without a
/// grid takes a position a row.
fn mrope_at(
    images: &[crate::ImageSpan],
    grid: impl Fn(&crate::ImageSpan) -> Option<(u32, u32)>,
    p: usize,
) -> [i32; 4] {
    let mut saved = 0i64;
    for sp in images.iter().take_while(|sp| sp.at as usize <= p) {
        let (at, rows) = (sp.at as usize, sp.rows as usize);
        let Some((nx, ny)) = grid(sp).filter(|&(x, y)| x > 0 && x as usize * y as usize == rows)
        else {
            continue;
        };
        let p0 = at as i64 - saved;
        if p < at + rows {
            let i = (p - at) as i64;
            let nx = i64::from(nx);
            return [p0, p0 + i / nx, p0 + i % nx, 0].map(|v| v as i32);
        }
        saved += rows as i64 - i64::from(nx.max(ny));
    }
    let q = (p as i64 - saved) as i32;
    [q, q, q, 0]
}

/// Puts a step's image rows (`(batch row, request, position)`, in row order) into `v.rows` at their
/// batch rows, encoding each tile it lacks into the least recently used slot, and returns the runs of
/// consecutive rows of one image `(first batch row, rows)`.
fn image_runs<'r>(
    v: &mut VisionRt,
    s: &Stream,
    image_rows: &[(usize, u64, usize)],
    req: impl Fn(u64) -> &'r crate::sched::Req,
    d_model: usize,
    encodes: &AtomicU64,
) -> Result<Vec<(usize, usize)>, KError> {
    let unit = v.config.max_unit_rows() as usize;
    // Group rows of one unit of one image that follow each other: (row, n, request, span, unit,
    // first). A span's units are its `tile_rows` (checked at admission).
    let mut groups: Vec<(usize, usize, u64, usize, u32, usize)> = Vec::new();
    for &(row, id, p) in image_rows {
        let images = &req(id).images;
        let k = images.partition_point(|sp| sp.at as usize <= p) - 1;
        let tt = images[k].tile_rows.max(1) as usize;
        let (tile, within) = (
            (p - images[k].at as usize) / tt,
            (p - images[k].at as usize) % tt,
        );
        match groups.last_mut() {
            Some(g)
                if g.2 == id
                    && g.3 == k
                    && g.4 == tile as u32
                    && g.0 + g.1 == row
                    && g.5 + g.1 == within =>
            {
                g.1 += 1
            }
            _ => groups.push((row, 1, id, k, tile as u32, within)),
        }
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut last = None;
    for &(row, n, id, k, tile, within) in &groups {
        let span = &req(id).images[k];
        let digest = *span.image.digest();
        // Admission refused a span without pixels (`check_images`): never None here.
        let pixels = || span.pixels.clone().ok_or(KError::InvalidArg);
        let tt = span.tile_rows as usize;
        let (slot, encoded) = v.tile_slot(s, digest, tile, tt, pixels, d_model)?;
        if encoded {
            encodes.fetch_add(1, Ordering::Relaxed);
        }
        let table = Tensor::new(
            &v.tiles,
            (slot * unit * d_model * 4) as u64,
            DType::F32,
            &[tt as i64, d_model as i64],
        )?;
        let ids = v
            .model
            .ids((within + n) as u32)?
            .rows(within as i64, n as i64)?;
        let out = Tensor::new(
            &v.rows,
            (row * d_model * 4) as u64,
            DType::F32,
            &[n as i64, d_model as i64],
        )?;
        ops::embed(s, &table, &ids, &out)?;
        // A run's ids are sized for one image (`Projector::ids`): rows of the next span start a
        // new run even where they follow on.
        match runs.last_mut() {
            Some(r) if last == Some((id, k)) && r.0 + r.1 == row => r.1 += n,
            _ => runs.push((row, n)),
        }
        last = Some((id, k));
    }
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    use larch_kernels::{CPU, KError, KvPool, Stream, Tensor};
    use larch_model::{Arch, FakeModel, Model, ModelConfig, StepBatch, Transformer, tiny};
    use larch_records::{Backend, EngineError, HealthState, LoadError, RejectReason, knob_id};

    use super::{
        Capture, Dims, EffectiveConfig, Engine, FitPlan, GRAPH_BYTES, GraphKey, Loaded, Partial,
        Refusal, block_runs, budget_layers, capture_charge, eager_line, fit, gated_plan,
        graph_decode_bucket, graph_reserve, graphs_fit, graphs_fit_gated, layer_download,
        layer_upload, mixed_key, remeasured_charge, says_eager, startup_config, step_floor,
    };
    use crate::{
        EngineEvent, EngineOptions, Finish, Forced, ModelClass, ReloadChange, Request, start_with,
    };

    /// A layer's bytes (each block's K then V) land at each named block's place in the K and V
    /// pools, over runs and lone blocks alike, and come back in the same order; real blocks with no
    /// bytes are refused, nothing with nothing is a no-op.
    #[test]
    fn a_layer_copies_through_runs_to_the_blocks_it_names() {
        let dev = larch_kernels::Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let (block, n_blocks, layers) = (8usize, 12u64, 2u64);
        let size = layers * n_blocks * block as u64;
        let pool = || larch_kernels::Buffer::alloc(&dev, size).unwrap();
        let (k, v) = (pool(), pool());
        let blocks = [4u32, 5, 6, 9, 2];
        let bytes: Vec<u8> = (0..2 * blocks.len() * block).map(|i| i as u8).collect();
        layer_upload(&s, (&k, &v), n_blocks, 1, &blocks, &bytes).unwrap();
        let (mut ks, mut vs) = (vec![0u8; size as usize], vec![0u8; size as usize]);
        s.download(&k, 0, &mut ks).unwrap();
        s.download(&v, 0, &mut vs).unwrap();
        for (j, &b) in blocks.iter().enumerate() {
            let at = (n_blocks as usize + b as usize) * block;
            assert_eq!(
                ks[at..at + block],
                bytes[2 * j * block..(2 * j + 1) * block]
            );
            assert_eq!(
                vs[at..at + block],
                bytes[(2 * j + 1) * block..(2 * j + 2) * block]
            );
        }
        let mut back = vec![0u8; bytes.len()];
        layer_download(&s, (&k, &v), n_blocks, 1, &blocks, &mut back).unwrap();
        assert_eq!(back, bytes);
        let refused = layer_upload(&s, (&k, &v), n_blocks, 1, &blocks, &[]);
        assert!(matches!(refused, Err(KError::InvalidArg)));
        assert!(layer_upload(&s, (&k, &v), n_blocks, 1, &[], &[]).is_ok());
    }

    /// A pin's or carry's blocks copy a run of consecutive ids at a time, each run with the index
    /// of its first block in the list; anything else, a wrap at `u32::MAX` included, starts a run.
    #[test]
    fn blocks_copy_in_runs_of_consecutive_ids() {
        let runs = |b: &[u32]| {
            block_runs(b)
                .map(|(i, r)| (i, r.to_vec()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            runs(&[4, 5, 6, 9, 10, 2]),
            [(0, vec![4, 5, 6]), (3, vec![9, 10]), (5, vec![2])]
        );
        assert_eq!(runs(&[7]), [(0, vec![7])]);
        assert_eq!(runs(&[u32::MAX, 0]), [(0, vec![u32::MAX]), (1, vec![0])]);
        assert!(runs(&[]).is_empty());
    }

    /// An encode slice is as many layers as the measured time fits in the step budget, at least one.
    #[test]
    fn an_encode_slice_fits_the_step_budget() {
        // 2B on the M5: about 16 ms a layer at 920 rows, so one layer per 30 ms step; 300 rows fit 5.
        let ns = 16e6 / 920.0;
        assert_eq!(budget_layers(None, 920, 24), 1);
        assert_eq!(budget_layers(Some(ns), 920, 24), 1);
        assert_eq!(budget_layers(Some(ns), 300, 24), 5);
        assert_eq!(budget_layers(Some(1.0), 1, 24), 24);
        assert_eq!(budget_layers(Some(1e9), 1590, 24), 1);
    }

    /// A Qwen projector asks for a context four times InternVL's: its model's KV is about a ninth.
    #[test]
    fn a_qwen_projector_asks_for_more_context() {
        use larch_model::gguf::Gguf;
        use larch_model::projector::ProjectorConfig;
        let c = |b: Vec<u8>| ProjectorConfig::from_gguf(&Gguf::parse(&b).unwrap()).unwrap();
        let (q, i) = (
            c(tiny::Projector::qwen(32, 1).gguf()),
            c(tiny::Projector::new(32, 1).gguf()),
        );
        assert_eq!(crate::vision_context(&q), crate::QWEN_VISION_MAX_CONTEXT);
        assert_eq!(crate::vision_context(&i), crate::VISION_MAX_CONTEXT);
        assert_eq!(
            crate::QWEN_VISION_MAX_CONTEXT,
            4 * crate::VISION_MAX_CONTEXT
        );
    }

    /// llama.cpp's M-RoPE layout over two images and the text around them: an image's rows on its
    /// grid from its first position, the text after it max(nx, ny) on, and an image without a grid
    /// one position a row.
    #[test]
    fn mrope_positions_follow_each_images_grid() {
        use larch_model::vision::Image;
        use std::sync::Arc;
        let png = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../larch-model/tests/data/vision/palette.png"
        ))
        .unwrap();
        let image = Arc::new(Image::new(png.into()).unwrap());
        let span = |at: u32, rows: u32| crate::ImageSpan {
            at,
            rows,
            tile_rows: rows,
            image: image.clone(),
            pixels: None,
        };
        // Rows 3..9 an image of 3 x 2, rows 11..23 one of 2 x 6; text between and after.
        let spans = [span(3, 6), span(11, 12)];
        let grid = |sp: &crate::ImageSpan| Some(if sp.rows == 6 { (3, 2) } else { (2, 6) });
        let at = |p| super::mrope_at(&spans, grid, p);
        assert_eq!(at(2), [2, 2, 2, 0]);
        assert_eq!(
            (at(3), at(5), at(6), at(8)),
            ([3, 3, 3, 0], [3, 3, 5, 0], [3, 4, 3, 0], [3, 4, 5, 0])
        );
        // The first image saved 6 - 3 rows of positions.
        assert_eq!((at(9), at(10)), ([6, 6, 6, 0], [7, 7, 7, 0]));
        assert_eq!(
            (at(11), at(12), at(13), at(22)),
            ([8, 8, 8, 0], [8, 8, 9, 0], [8, 9, 8, 0], [8, 13, 9, 0])
        );
        // The second saved 12 - 6 more.
        assert_eq!(at(23), [14, 14, 14, 0]);
        // Without grids (InternVL), every row is its own position.
        let none = |p| super::mrope_at(&spans, |_| None, p);
        assert_eq!((none(5), none(23)), ([5, 5, 5, 0], [23, 23, 23, 0]));
        // A grid that does not hold the span's rows counts as none.
        let odd = |p| super::mrope_at(&spans, |_| Some((5, 5)), p);
        assert_eq!(odd(23), [23, 23, 23, 0]);
    }

    std::thread_local! {
        pub(super) static ALLOCATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        pub(super) static DOWNLOADS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    #[test]
    fn cuda_graph_buckets_keep_changing_decode_rows_exact() {
        for rows in 1usize..=64 {
            let cuda = if (9..=32).contains(&rows) {
                rows
            } else {
                rows.next_power_of_two()
            };
            assert_eq!(graph_decode_bucket(Backend::Cuda, rows), cuda, "{rows}");
            for backend in [Backend::Cpu, Backend::Metal, Backend::Hip] {
                assert_eq!(
                    graph_decode_bucket(backend, rows),
                    rows.next_power_of_two(),
                    "{backend:?} {rows}"
                );
            }
        }
    }

    /// A pure decode bucket's cache key.
    fn decode_key(rows: usize, splits: u32) -> GraphKey {
        GraphKey {
            decode: rows,
            prefill: 0,
            logits: rows,
            splits,
            taps: false,
            checks: 0,
        }
    }

    /// A refusal waits for what could change its outcome: more free quota bytes or the arena
    /// appearing. Fewer bytes, or nothing new, keep the shape eager.
    #[test]
    fn refusals_retry_once_their_quota_or_arena_change() {
        let r = Refusal {
            free: 2,
            arena: false,
            step: 0,
            again: false,
        };
        assert!(!r.retry(&r));
        assert!(!r.retry(&Refusal { free: 1, ..r }));
        assert!(r.retry(&Refusal { free: 3, ..r }));
        assert!(r.retry(&Refusal { arena: true, ..r }));
    }

    /// A capture charged past the quota left is held when free memory after it, re-read, still
    /// covers the headroom and the whole quota left (a reading that swung low by tens of MB, as on
    /// WSL), and refused when it does not (a real shortage).
    #[test]
    fn an_over_charged_capture_is_held_only_where_the_device_has_room() {
        let (room, headroom) = (16 << 20, 112 << 20);
        // Charged 24.9 MB against 16 MiB, with 160 MiB free after it: the reading over-counted.
        assert!(super::device_holds(room, 160 << 20, headroom));
        assert!(super::device_holds(room, headroom + room, headroom));
        // Free memory short of the headroom and quota: refused as before.
        assert!(!super::device_holds(room, headroom + room - 1, headroom));
        assert!(!super::device_holds(room, 96 << 20, headroom));
        assert!(!super::device_holds(u64::MAX, u64::MAX - 1, headroom));
    }

    /// A refusal over a measured charge owes one retry, [`RETRY_STEPS`] after it, whatever else
    /// changed; the retried capture owes none, so a second refusal waits for quota or the arena as
    /// any other. A refusal that measured nothing (no quota, a failed capture) owes none either.
    #[test]
    fn a_refusal_over_its_measured_charge_retries_once() {
        use super::{RETRY_STEPS, owed_retry};
        let seen = Refusal {
            free: 16 << 20,
            arena: true,
            step: 100,
            again: true,
        };
        let first = seen.over(24_907_776).expect("over the quota left");
        let at = |step| Refusal { step, ..seen };
        let refused = Capture::Refused(first);
        assert!(!first.retry(&at(100 + RETRY_STEPS - 1)));
        assert!(!owed_retry(Some(&refused), &at(100 + RETRY_STEPS - 1)));
        assert!(owed_retry(Some(&refused), &at(100 + RETRY_STEPS)));
        // The retry, refused again, is owed nothing more.
        let retried = Refusal {
            step: 100 + RETRY_STEPS,
            again: !owed_retry(Some(&refused), &at(100 + RETRY_STEPS)),
            ..seen
        };
        let second = retried.over(24_907_776).expect("over again");
        for step in [100 + 2 * RETRY_STEPS, u64::MAX / 2] {
            assert!(!second.retry(&at(step)));
            assert!(!owed_retry(Some(&Capture::Refused(second)), &at(step)));
        }
        // It still retries once the quota holds its size.
        assert!(second.retry(&Refusal {
            free: 24_907_776,
            ..at(u64::MAX / 2)
        }));
        let unmeasured = Refusal {
            again: false,
            ..seen
        };
        assert!(!owed_retry(
            Some(&Capture::Refused(unmeasured)),
            &at(u64::MAX / 2)
        ));
        assert!(!owed_retry(Some(&Capture::Seen), &at(u64::MAX / 2)));
        assert!(!owed_retry(None, &at(u64::MAX / 2)));
    }

    #[test]
    fn a_capture_step_is_no_cost_sample() {
        let plain = crate::sched::StepPlan::default();
        assert!(crate::engine::prices_checks(&plain, false));
        assert!(!crate::engine::prices_checks(&plain, true));
    }

    /// A capture is charged what free memory lost across it and its first replay, less the arena it
    /// allocated, and never less than the fit's estimate: a 2 MiB granule reading can show nothing.
    #[test]
    fn a_capture_is_charged_at_least_the_estimate() {
        let (before, arena) = (8u64 << 30, 64 << 20);
        assert_eq!(capture_charge(before, before, 0), GRAPH_BYTES);
        assert_eq!(capture_charge(before, before - arena, arena), GRAPH_BYTES);
        assert_eq!(capture_charge(before, before - (2 << 20), 0), GRAPH_BYTES);
        let big = GRAPH_BYTES + (2 << 20);
        assert_eq!(capture_charge(before, before - arena - big, arena), big);
        // Free memory that grew across the window reads as nothing taken.
        assert_eq!(
            capture_charge(before, before + (2 << 20), arena),
            GRAPH_BYTES
        );
    }

    /// A charge past the quota left is read again once: the second reading stands if it fits or
    /// not, a failed one keeps the first, and a charge within the quota never reads again.
    #[test]
    fn an_over_quota_capture_is_remeasured_once() {
        let (before, room) = (8u64 << 30, 10 << 20);
        let ctx = (room, before, 0);
        let calls = std::cell::Cell::new(0);
        let read = |free: Result<u64, ()>| {
            calls.set(calls.get() + 1);
            free
        };
        assert_eq!(
            remeasured_charge(ctx, before - (6 << 20), || read(Ok(0))),
            6 << 20
        );
        assert_eq!(calls.get(), 0);
        let over = before - (40 << 20);
        assert_eq!(
            remeasured_charge(ctx, over, || read(Ok(before - (8 << 20)))),
            8 << 20
        );
        assert_eq!(remeasured_charge(ctx, over, || read(Ok(over))), 40 << 20);
        assert_eq!(remeasured_charge(ctx, over, || read(Err(()))), 40 << 20);
        assert_eq!(calls.get(), 3);
    }

    /// The quota charges a capture the bytes it measured: one over what is left is refused, and its
    /// shape retries only once freed bytes leave room for that size; one that fits is kept.
    #[test]
    fn a_capture_over_the_quota_left_waits_for_room() {
        let now = Refusal {
            free: 10 << 20,
            arena: true,
            step: 0,
            again: false,
        };
        assert_eq!(now.over(10 << 20), None);
        let r = now.over((10 << 20) + 1).expect("over the quota left");
        assert!(!r.retry(&now));
        assert!(r.retry(&Refusal {
            free: (10 << 20) + 1,
            ..now
        }));
    }

    /// The report of a load's first eager decode step on a refusal names the bucket, why, and the
    /// quota bytes the shape is short now: a capture's floor beside a quota short of it, its
    /// measured size beside one it measured over, none for a capture the backend refused.
    #[test]
    fn eager_line_names_the_reason_and_the_bytes_short() {
        let line = |r: &Refusal, free, failed| eager_line(4, r, free, 16 << 20, failed);
        let quota = Refusal {
            free: 1 << 20,
            arena: false,
            step: 0,
            again: false,
        };
        assert_eq!(
            line(&quota, 1 << 20, None),
            "larch_graph_eager,decode=4,reason=quota,short_bytes=3145728,free_bytes=1048576,quota_bytes=16777216"
        );
        let over = Refusal {
            free: 10 << 20,
            arena: true,
            step: 0,
            again: false,
        }
        .over(12 << 20)
        .unwrap();
        assert!(line(&over, 10 << 20, None).contains(",reason=quota,short_bytes=2097152,"));
        let failed = line(&quota, 8 << 20, Some("graph_begin: arena does not fit"));
        assert!(
            failed.contains(
                ",reason=capture failed (graph_begin: arena does not fit),short_bytes=0,"
            ),
            "{failed}"
        );
    }

    /// The eager line prints once a process, at its first counted step, and only under
    /// LARCH_FIT_DIAGNOSTICS=1, which is read only for that step.
    #[test]
    fn the_eager_line_prints_once_and_only_under_diagnostics() {
        let on = || Some("1".to_owned());
        assert!(says_eager(0, on));
        assert!(!says_eager(1, || panic!("read past the first step")));
        assert!(!says_eager(0, || None));
        assert!(!says_eager(0, || Some("0".to_owned())));
    }

    #[test]
    fn the_watchdog_floor_follows_the_device_opened() {
        // Built for GPU 0, then moved to the CPU, as a caller falling back does.
        let mut opts = EngineOptions::new("fake".into(), 0);
        opts.device = CPU;
        let (_e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        assert_eq!(
            h.shared.watch.step_limit(),
            std::time::Duration::from_secs(60)
        );
    }

    #[test]
    fn a_metal_step_gets_the_slow_device_floor() {
        let opts = EngineOptions::new("fake".into(), 0);
        let class = ModelClass {
            dense: false,
            active_bytes: 0,
            experts_in_ram: false,
        };
        assert_eq!(
            step_floor(&opts, Backend::Metal, class),
            std::time::Duration::from_secs(60)
        );
        assert_eq!(
            step_floor(&opts, Backend::Cuda, class),
            std::time::Duration::from_secs(5)
        );
    }

    #[test]
    fn a_job_run_over_queued_work_leaves_the_watchdog_armed() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        let _request = h.submit(Request::new(vec![1], 16)).unwrap();
        e.job = Some(Box::new(|_, _| {}));
        assert!(e.turn().unwrap());
        assert!(e.job.is_none() && !e.sched.is_idle());
        assert!(e.sh.watch.pending.load(Ordering::Acquire));
    }

    #[test]
    fn eager_only_fit_keeps_live_graph_and_batch_changes_within_budget() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 8.0)];
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        let loaded = e.loaded.as_mut().unwrap();
        loaded.info.caps.backend = Backend::Cuda;
        loaded.info.caps.graph_replay = true;
        loaded.graph_bytes = 0;
        loaded.graph_arena_bytes = 0;
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 1.0;
        let mut requests: Vec<_> = (0..8)
            .map(|_| h.submit(Request::new(vec![1], 16)).unwrap())
            .collect();
        assert!(e.turn().unwrap());
        assert!(e.turn().unwrap());
        assert!(matches!(
            e.loaded.as_ref().unwrap().graphs.get(&decode_key(8, 0)),
            Some(Capture::Refused(_))
        ));
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 0.0;
        assert!(e.turn().unwrap());
        e.cfg.values[knob_id::MAX_BATCH_SEQS as usize] = 16.0;
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 1.0;
        e.cfg.values[knob_id::ATTN_DECODE_SPLITS as usize] = 3.0;
        requests.extend((0..8).map(|_| h.submit(Request::new(vec![1], 16)).unwrap()));
        assert!(e.turn().unwrap());
        assert!(e.turn().unwrap());
        let loaded = e.loaded.as_ref().unwrap();
        assert!(matches!(
            loaded.graphs.get(&decode_key(16, 3)),
            Some(Capture::Refused(_))
        ));
        assert_eq!(loaded.graph_bytes + loaded.graph_arena_bytes, 0);
        while !e.sched.is_idle() {
            assert!(e.turn().unwrap());
        }
        for request in requests {
            let mut tokens = Vec::new();
            for event in request.events.try_iter() {
                match event {
                    EngineEvent::Token { id, .. } => tokens.push(id),
                    EngineEvent::Done { usage, .. } => assert_eq!(usage.completion_tokens, 16),
                    EngineEvent::Prefilled { .. } => {}
                    event => panic!("{event:?}"),
                }
            }
            assert_eq!(tokens, (2..18).collect::<Vec<_>>());
        }
    }

    #[test]
    fn mixed_graph_keys_are_bounded_exact_and_configuration_derived() {
        for (nd, logits) in [(15, 15), (15, 16), (7, 7), (7, 8)] {
            assert_eq!(
                mixed_key(Backend::Cuda, (512, 16), nd, Some(512), logits, 3),
                Some(GraphKey {
                    decode: nd,
                    prefill: 512,
                    logits,
                    splits: 3,
                    taps: false,
                    checks: 0,
                })
            );
        }
        for (nd, prefill, logits) in [
            (14, Some(512), 14),
            (15, Some(511), 15),
            (15, None, 15),
            (15, Some(512), 17),
            (0, Some(512), 1),
        ] {
            assert_eq!(
                mixed_key(Backend::Cuda, (512, 16), nd, prefill, logits, 3),
                None
            );
        }
        for backend in [Backend::Cpu, Backend::Hip, Backend::Metal] {
            assert_eq!(mixed_key(backend, (512, 16), 15, Some(512), 15, 3), None);
        }
        // An odd seat count: its full (15) and half (8) admissions.
        assert!(mixed_key(Backend::Cuda, (512, 15), 14, Some(512), 14, 3).is_some());
        assert!(mixed_key(Backend::Cuda, (512, 15), 7, Some(512), 7, 3).is_some());
    }

    /// A mixed shape runs eager on its first hit, attempts a capture on its second, and after a
    /// refusal runs eager again, all with the full-width tables a graph-capable backend sends;
    /// cancelling the prefill reuses its rows and KV, and a chunk change drops the shapes it can no
    /// longer produce.
    #[test]
    fn mixed_shapes_capture_on_the_second_hit_and_refusals_run_exact() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        opts.knob_overrides = vec![
            (knob_id::MAX_BATCH_SEQS, 8.0),
            (knob_id::PREFILL_CHUNK_TOKENS, 128.0),
        ];
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        // CUDA capability over the CPU stream: every capture attempt comes back empty.
        let loaded = e.loaded.as_mut().unwrap();
        loaded.info.caps.backend = Backend::Cuda;
        loaded.info.caps.graph_replay = true;
        loaded.mixed_graphs = true;
        loaded.graph_bytes = 4 * GRAPH_BYTES;
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 1.0;
        e.cfg.values[knob_id::PREFILL_CHUNK_TOKENS as usize] = 64.0;
        let survivors: Vec<_> = (0..7)
            .map(|i| h.submit(Request::new(vec![i + 1], 32)).unwrap())
            .collect();
        assert!(e.turn().unwrap());
        // 7 decode + 64 prefill tokens, eight tables, seven decode lengths, one prefill start/length
        // and seven logits indices; no padding or penalty rows.
        let full = |e: &Engine| 71 * 3 + 8 * e.loaded.as_ref().unwrap().dims.max_blocks + 16;
        let key = GraphKey {
            decode: 7,
            prefill: 64,
            logits: 7,
            splits: 0,
            taps: false,
            checks: 0,
        };
        let cancelled = h.submit(Request::new(vec![11; 192], 1)).unwrap();
        assert!(e.turn().unwrap());
        let loaded = e.loaded.as_ref().unwrap();
        assert!(matches!(loaded.graphs.get(&key), Some(Capture::Seen)));
        assert_eq!(loaded.staging.packed.len(), full(&e));
        assert!(e.turn().unwrap());
        let loaded = e.loaded.as_ref().unwrap();
        assert!(matches!(loaded.graphs.get(&key), Some(Capture::Refused(_))));
        assert_eq!(loaded.staging.packed.len(), full(&e));
        cancelled.cancel();
        let replacement = h.submit(Request::new(vec![12; 192], 1)).unwrap();
        assert!(e.turn().unwrap());
        assert!(matches!(
            cancelled
                .events
                .try_iter()
                .find(|e| !matches!(e, EngineEvent::Prefilled { .. })),
            Some(EngineEvent::Error(EngineError::Cancelled))
        ));
        let replacement_req = e.sched.req(replacement.id).unwrap();
        assert_eq!(replacement_req.seq.n_kv, 64);
        assert_eq!(replacement_req.seq.tokens[0], 12);
        // The refused shape ran eager, at the full width too.
        assert_eq!(e.loaded.as_ref().unwrap().staging.packed.len(), full(&e));
        // A live chunk change drops the 64-row shapes; the next one starts over as seen.
        e.cfg.values[knob_id::PREFILL_CHUNK_TOKENS as usize] = 128.0;
        assert!(e.turn().unwrap());
        let loaded = e.loaded.as_ref().unwrap();
        assert!(loaded.graphs.keys().all(|k| k.prefill != 64));
        assert!(
            loaded
                .graphs
                .iter()
                .any(|(k, c)| k.prefill == 128 && matches!(c, Capture::Seen))
        );
        drop(survivors);
    }

    /// Seven requests decoding on a graph-capable (faked) stream with mixed graphs on: a 64-token
    /// activations prompt joins them in the mixed shape (7, 64, 7).
    fn tapped_mixed() -> (Engine, crate::EngineHandle, Vec<crate::RequestHandle>) {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 8.0)];
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::with_layers(64, 2)))),
            None,
        )
        .unwrap();
        let loaded = e.loaded.as_mut().unwrap();
        loaded.info.caps.backend = Backend::Cuda;
        loaded.info.caps.graph_replay = true;
        loaded.mixed_graphs = true;
        loaded.graph_bytes = 4 * GRAPH_BYTES;
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 1.0;
        e.cfg.values[knob_id::PREFILL_CHUNK_TOKENS as usize] = 64.0;
        let decoding: Vec<_> = (0..7)
            .map(|i| h.submit(Request::new(vec![i + 1], 32)).unwrap())
            .collect();
        assert!(e.turn().unwrap());
        (e, h, decoding)
    }

    /// The activations request of [`tapped_mixed`]: 63 tokens, then `last`, read after layer 1.
    fn tap_prompt(h: &crate::EngineHandle, last: u32) -> crate::RequestHandle {
        let mut prompt = vec![11; 63];
        prompt.push(last);
        let tapped = Request {
            activations: vec![1],
            ..Request::new(prompt, 0)
        };
        h.submit(tapped).unwrap()
    }

    /// Its Done's rows; FakeModel's row after layer 1 is `[last, 1, ..]`.
    fn tapped_rows(tapped: &crate::RequestHandle, last: u32) {
        let end = tapped
            .events
            .try_iter()
            .find(|e| !matches!(e, EngineEvent::Prefilled { .. }));
        let Some(EngineEvent::Done { activations, .. }) = end else {
            panic!("{last}: {end:?}");
        };
        assert_eq!(activations.len(), 1, "{last}");
        assert_eq!(activations[0][..2], [last as f32, 1.0]);
    }

    /// A graph would replay without the taps: the mixed shape above, with an activations request's
    /// last chunk in it, takes no graph key on its first hit (no Seen) or its second (no capture),
    /// and the rows still come back.
    #[test]
    fn a_tapped_step_takes_no_graph() {
        let (mut e, h, decoding) = tapped_mixed();
        let key = GraphKey {
            decode: 7,
            prefill: 64,
            logits: 7,
            splits: 0,
            taps: false,
            checks: 0,
        };
        for last in [20, 30] {
            let tapped = tap_prompt(&h, last);
            assert!(e.turn().unwrap());
            assert!(
                !e.loaded.as_ref().unwrap().graphs.contains_key(&key),
                "{last}"
            );
            tapped_rows(&tapped, last);
        }
        drop(decoding);
    }

    /// A tapped mixed step serves its decoders with one download (its one host sync: the taps'
    /// rows ride in it) and, once a first one has run, allocates nothing: no device buffer, and no
    /// host staging grows. Every decoder still gets its token in that step.
    #[test]
    fn a_tapped_mixed_step_downloads_once_and_allocates_nothing() {
        let (mut e, h, decoding) = tapped_mixed();
        let capacity = |e: &Engine| {
            let st = &e.loaded.as_ref().unwrap().staging;
            let caps = [
                st.rows.capacity(),
                st.packed.capacity(),
                st.params.capacity(),
            ];
            let more = [st.out.capacity(), st.sampled.capacity(), st.taps.capacity()];
            (caps, more, st.layers.capacity())
        };
        let warm = tap_prompt(&h, 20);
        assert!(e.turn().unwrap());
        tapped_rows(&warm, 20);
        let before = capacity(&e);
        for d in &decoding {
            d.events.try_iter().for_each(drop);
        }
        let tapped = tap_prompt(&h, 30);
        ALLOCATIONS.with(|count| count.set(0));
        DOWNLOADS.with(|count| count.set(0));
        assert!(e.turn().unwrap());
        assert_eq!(DOWNLOADS.with(|count| count.get()), 1, "host syncs");
        assert_eq!(
            ALLOCATIONS.with(|count| count.get()),
            0,
            "device allocations"
        );
        assert_eq!(capacity(&e), before, "host staging grew");
        tapped_rows(&tapped, 30);
        for d in &decoding {
            let got = d.events.try_iter().count();
            assert!(got > 0, "a decoder missed the tapped step");
        }
    }

    /// The tap scratch is in the fit budget: one step's taps (`tap_seats` requests, `tap_width`
    /// f32 each) add their rows to the input and their output to `samp`, whatever the logits rows;
    /// a loaded engine reserves every layer for `TAP_SEATS` requests.
    #[test]
    fn the_tap_scratch_is_reserved_at_load() {
        let none = Dims {
            rows: 576,
            seqs: 64,
            max_blocks: 512,
            vocab: 128_256,
            tap_seats: 0,
            tap_floats: 0,
        };
        let taps = Dims {
            tap_seats: 4,
            tap_floats: 4 * 32 * 4096,
            ..none
        };
        for rows in [64, 576] {
            let [i0, l0, s0] = none.bytes(rows);
            assert_eq!(taps.bytes(rows), [i0 + 16, l0, s0 + 4 * 32 * 4096 * 4]);
        }
        assert_eq!(taps.fit_bytes(), none.fit_bytes() + 16 + 4 * 32 * 4096 * 4);
        let (e, _h, _decoding) = tapped_mixed();
        let loaded = e.loaded.as_ref().unwrap();
        let mc = loaded.model.config();
        assert!(loaded.dims.tap_seats >= crate::sched::TAP_SEATS);
        assert!(
            loaded.dims.tap_floats >= crate::sched::TAP_SEATS * (mc.n_layers * mc.d_model) as usize
        );
        let [_, _, samp] = loaded.dims.bytes(loaded.logit_rows);
        assert_eq!(loaded.samp.bytes(), samp);
    }

    #[test]
    fn unavailable_decode_capture_is_cached_until_split_or_buffer_change() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 1.0)];
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        // A graph-capable advertised model over the CPU stream makes capture return None.
        // The request must still progress eagerly and the failed bucket must not be recaptured.
        let loaded = e.loaded.as_mut().unwrap();
        loaded.info.caps.backend = Backend::Cuda;
        loaded.info.caps.graph_replay = true;
        loaded.graph_bytes = 4 * GRAPH_BYTES;
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 1.0;
        let request = h.submit(Request::new(vec![1], 8)).unwrap();
        assert!(e.turn().unwrap());
        assert!(e.turn().unwrap());
        assert!(
            e.loaded
                .as_ref()
                .unwrap()
                .graphs
                .contains_key(&decode_key(1, 0))
        );
        assert!(e.turn().unwrap());
        assert_eq!(e.loaded.as_ref().unwrap().graphs.len(), 1);
        e.cfg.values[knob_id::ATTN_DECODE_SPLITS as usize] = 3.0;
        assert!(e.turn().unwrap());
        let graphs = &e.loaded.as_ref().unwrap().graphs;
        assert_eq!(graphs.keys().collect::<Vec<_>>(), [&decode_key(1, 3)]);
        let rows = e.loaded.as_ref().unwrap().dims.rows;
        e.observe_rows(rows).unwrap();
        assert!(e.loaded.as_ref().unwrap().graphs.is_empty());
        assert!(e.turn().unwrap());
        assert!(
            e.loaded
                .as_ref()
                .unwrap()
                .graphs
                .contains_key(&decode_key(1, 3))
        );
        while !e.sched.is_idle() {
            assert!(e.turn().unwrap());
        }
        let mut tokens = Vec::new();
        for event in request.events.try_iter() {
            match event {
                EngineEvent::Token { id, .. } => tokens.push(id),
                EngineEvent::Done { usage, .. } => assert_eq!(usage.completion_tokens, 8),
                EngineEvent::Prefilled { .. } => {}
                event => panic!("{event:?}"),
            }
        }
        assert_eq!(tokens, (2..10).collect::<Vec<_>>());
    }

    /// Greedy prep keeps enough candidates for EOS masking and the requested top list; stochastic
    /// and parity rows still see the full top-64 distribution in the same mixed step.
    #[test]
    fn greedy_prep_preserves_banned_logprobs_and_observation() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        let mut greedy = Request::new(vec![1], 1);
        greedy.ignore_eos = true;
        greedy.stop_tokens = vec![2, 3];
        greedy.logprobs = 2;
        let greedy = h.submit(greedy).unwrap();
        let mut top = Request::new(vec![20], 1);
        top.logprobs = 20;
        let top = h.submit(top).unwrap();
        let mut stochastic = Request::new(vec![30], 1);
        stochastic.sampling.temperature = 1.0;
        let stochastic = h.submit(stochastic).unwrap();
        let forced = h
            .force(Forced {
                tokens: vec![10, 11, 12],
                prompt: 3,
                session: 0,
                keep_kv: false,
            })
            .unwrap();
        assert!(e.turn().unwrap());
        let params = &e.loaded.as_ref().unwrap().staging.params;
        assert_eq!(params[0].top_k, 3);
        assert_eq!(params[1].top_k, 20);
        assert!(params[2..].iter().all(|p| p.top_k == 0));
        let first = greedy
            .events
            .try_iter()
            .find(|e| !matches!(e, EngineEvent::Prefilled { .. }));
        match first.unwrap() {
            EngineEvent::Token { id, logprob, top } => {
                assert_eq!(id, 0);
                assert_eq!(logprob, -100.0);
                assert_eq!(top, [(2, 0.0), (0, -100.0)]);
            }
            event => panic!("{event:?}"),
        }
        let token = |h: &crate::RequestHandle| {
            h.events
                .try_iter()
                .find(|e| !matches!(e, EngineEvent::Prefilled { .. }))
                .unwrap()
        };
        assert!(matches!(token(&top), EngineEvent::Token { top, .. } if top.len() == 20));
        assert!(matches!(
            token(&stochastic),
            EngineEvent::Token { id: 31, .. }
        ));
        let positions: Vec<_> = forced.positions.try_iter().collect();
        assert_eq!(positions.len(), 3);
        assert!(positions.iter().all(|p| p.top.len() == 64));
        assert_eq!(
            positions.iter().map(|p| p.top[0].0).collect::<Vec<_>>(),
            [11, 12, 13]
        );
    }

    /// Shrinking a batch and disabling its last penalty row must reuse host storage without stale
    /// token, logits-row or penalty metadata reaching the next step.
    #[test]
    fn decode_reuses_host_staging_as_requests_finish() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        let mut penalized = Request::new(vec![1; 17], 3);
        penalized.sampling.repeat_penalty = 1.1;
        let handles = [
            h.submit(penalized).unwrap(),
            h.submit(Request::new(vec![10; 7], 6)).unwrap(),
        ];
        assert!(e.turn().unwrap());
        let pointers = |e: &Engine| {
            let b = &e.loaded.as_ref().unwrap().staging;
            (
                b.rows.as_ptr(),
                b.packed.as_ptr(),
                b.params.as_ptr(),
                b.out.as_ptr(),
                b.sampled.as_ptr(),
            )
        };
        let warm = pointers(&e);
        while !e.sched.is_idle() {
            assert!(e.turn().unwrap());
            assert_eq!(
                pointers(&e),
                warm,
                "host staging was reallocated during decode"
            );
        }
        for (handle, expected) in handles
            .iter()
            .zip([vec![2, 3, 4], vec![11, 12, 13, 14, 15, 16]])
        {
            let events: Vec<_> = handle.events.try_iter().collect();
            let tokens: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    EngineEvent::Token { id, .. } => Some(*id),
                    _ => None,
                })
                .collect();
            assert_eq!(tokens, expected);
            assert!(matches!(events.last(), Some(EngineEvent::Done { .. })));
        }
    }

    struct DropTracked {
        inner: FakeModel,
        dropped: Arc<std::sync::atomic::AtomicBool>,
        drop_wait: Option<(
            std::sync::mpsc::SyncSender<()>,
            std::sync::mpsc::Receiver<()>,
        )>,
    }
    impl Model for DropTracked {
        fn config(&self) -> &ModelConfig {
            self.inner.config()
        }
        fn forward(
            &mut self,
            s: &Stream,
            b: &StepBatch,
            kv: &KvPool,
            logits: &Tensor,
        ) -> Result<(), KError> {
            self.inner.forward(s, b, kv, logits)
        }
    }
    impl Drop for DropTracked {
        fn drop(&mut self) {
            if let Some((entered, release)) = self.drop_wait.take() {
                let _ = entered.send(());
                let _ = release.recv_timeout(std::time::Duration::from_secs(10));
            }
            self.dropped
                .store(true, std::sync::atomic::Ordering::Release);
        }
    }

    // Catches dropping the first replacement buffer when a later physical allocation fails.
    #[test]
    fn observation_oom_retains_partial_buffers_until_process_exit() {
        for (target, retained) in [(1, 0), (2, 1), (3, 2)] {
            let mut opts = EngineOptions::new("fake".into(), CPU);
            opts.max_context = 512;
            let (mut e, h) = Engine::init(
                opts,
                Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
                None,
            )
            .unwrap();
            let free = e.s.device().props().unwrap().vram_free;
            let loaded = e.loaded.as_mut().unwrap();
            let growth = loaded.dims.bytes(128).into_iter().sum::<u64>()
                - loaded
                    .dims
                    .bytes(loaded.logit_rows)
                    .into_iter()
                    .sum::<u64>();
            loaded.graph_bytes = 29 * GRAPH_BYTES;
            loaded.graph_arena_bytes = 64 << 20;
            loaded.headroom_bytes = free - growth - loaded.graph_arena_bytes - 3 * GRAPH_BYTES;
            let old_rows = loaded.logit_rows;
            crate::fault::tests::set(&format!("oom@alloc:{target}"));
            let result = e.observe_rows(128);
            crate::fault::tests::set("");
            assert_eq!(result, Err(KError::Oom));
            assert_eq!(e.loaded.as_ref().unwrap().graph_bytes, 29 * GRAPH_BYTES);
            assert_eq!(e.loaded.as_ref().unwrap().logit_rows, old_rows);
            assert_eq!(e.partial.buffers.len(), retained);
            for buffer in &e.partial.buffers {
                assert!(buffer.bytes() > 0);
            }
            e.stop(Some((
                larch_records::FaultCode::Oom,
                EngineError::Internal(2),
            )));
            assert_eq!(h.health().last_fault, larch_records::FaultCode::Oom);
            assert_eq!(e.partial.buffers.len(), retained);
            std::mem::forget(e);
        }
    }

    #[test]
    fn reload_oom_retains_model_and_each_completed_allocation() {
        use std::sync::atomic::{AtomicBool, Ordering};
        for (target, retained) in [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)] {
            let mut opts = EngineOptions::new("fake".into(), CPU);
            opts.max_context = 512;
            let (mut e, h) = Engine::init(
                opts,
                Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
                None,
            )
            .unwrap();
            let dropped = Arc::new(AtomicBool::new(false));
            let track_drop = dropped.clone();
            let (reply, response) = std::sync::mpsc::sync_channel(1);
            e.reload = Some(super::PendingReload {
                change: crate::ReloadChange::default(),
                reply,
                load: Some(Box::new(move |_, _, _| {
                    Ok(Box::new(DropTracked {
                        inner: FakeModel::new(64),
                        dropped: track_drop,
                        drop_wait: None,
                    }))
                })),
            });
            crate::fault::tests::set(&format!("oom@alloc:{target}"));
            let resumed = e.rebuild();
            crate::fault::tests::set("");
            assert!(!resumed);
            assert_eq!(response.try_recv().unwrap(), Err(EngineError::Internal(2)));
            assert_eq!(h.health().last_fault, larch_records::FaultCode::Oom);
            assert!(!dropped.load(Ordering::Acquire));
            assert!(e.partial.model.is_some());
            assert_eq!(e.partial.buffers.len(), retained);
            for buffer in &e.partial.buffers {
                assert!(buffer.bytes() > 0);
            }
            std::mem::forget(e);
        }
    }

    // A terminal watchdog fault during old-model Drop must prevent the next loader invocation.
    #[test]
    fn failed_reload_does_not_call_loader_after_teardown() {
        use larch_records::{FaultCode, Read, ReadFrom, Record};
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc::sync_channel;
        use std::time::{Duration, Instant};
        let (entered_tx, entered) = sync_channel(1);
        let (release, held) = sync_channel(1);
        let old_dropped = Arc::new(AtomicBool::new(false));
        let old_drop = old_dropped.clone();
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        // The test waits for the watchdog, so it keeps the GPU floor on the fast fake model.
        opts.step_timeout_floor = Some(Duration::from_secs(5));
        let (mut e, h) = Engine::init(
            opts,
            Box::new(move |_, _, _| {
                Ok(Box::new(DropTracked {
                    inner: FakeModel::new(64),
                    dropped: old_drop,
                    drop_wait: Some((entered_tx, held)),
                }))
            }),
            None,
        )
        .unwrap();
        let called = Arc::new(AtomicBool::new(false));
        let call = called.clone();
        let capture_dropped = Arc::new(AtomicBool::new(false));
        let capture = DropTracked {
            inner: FakeModel::new(16),
            dropped: capture_dropped.clone(),
            drop_wait: None,
        };
        let (reply, response) = sync_channel(1);
        e.reload = Some(super::PendingReload {
            change: crate::ReloadChange::default(),
            reply,
            load: Some(Box::new(move |_, _, _| {
                call.store(true, Ordering::Release);
                drop(capture);
                Ok(Box::new(FakeModel::new(16)))
            })),
        });
        e.sh.watch.begin(e.sh.now());
        let watchdog = crate::lifecycle::watchdog(&e.sh, crate::OnFault::Fail).unwrap();
        let rebuilding = std::thread::spawn(move || {
            let resumed = e.rebuild();
            std::mem::forget(e);
            resumed
        });
        let teardown_entered = entered.recv_timeout(Duration::from_secs(1));
        let until = Instant::now() + Duration::from_secs(7);
        while h.health().state != HealthState::Failed && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
        let before_release = h.health();
        // Release teardown and join both controlled threads before every assertion, including RED.
        let _ = release.send(());
        let resumed = rebuilding.join();
        h.shared.watch.stopped.store(true, Ordering::Release);
        let watched = watchdog.join();
        assert_eq!(teardown_entered, Ok(()));
        watched.unwrap();
        assert_eq!(
            (before_release.state, before_release.last_fault),
            (HealthState::Failed, FaultCode::Timeout)
        );
        assert!(!resumed.unwrap());
        assert_eq!(response.try_recv().unwrap(), Err(EngineError::DeviceLost));
        assert_eq!(
            (
                old_dropped.load(Ordering::Acquire),
                called.load(Ordering::Acquire),
                capture_dropped.load(Ordering::Acquire)
            ),
            (true, false, false),
            "completed teardown must not invoke or destroy the unused replacement loader"
        );
        assert_eq!(h.health().state, HealthState::Failed);
        assert_eq!(h.health().last_fault, FaultCode::Timeout);
        assert_eq!(h.deployment().vocab, 64);
        let mut reader = h.ring().reader(ReadFrom::Oldest);
        let mut states = Vec::new();
        while let Read::Record(record) = reader.next() {
            if let Record::HealthEvent(_, event) = record {
                states.push(event.to);
            }
        }
        assert_eq!(
            states,
            [
                HealthState::Ready.raw(),
                HealthState::Loading.raw(),
                HealthState::Failed.raw()
            ]
        );
    }

    #[test]
    fn failed_reload_does_not_allocate_after_loader_returns() {
        use larch_records::FaultCode;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc::sync_channel;
        use std::time::Duration;
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        let (entered_tx, entered) = sync_channel(1);
        let (release, held) = sync_channel(1);
        let (reply, response) = sync_channel(1);
        let dropped = Arc::new(AtomicBool::new(false));
        let track_drop = dropped.clone();
        e.reload = Some(super::PendingReload {
            change: crate::ReloadChange::default(),
            reply,
            load: Some(Box::new(move |_, _, _| {
                entered_tx.send(()).unwrap();
                held.recv_timeout(Duration::from_secs(10)).unwrap();
                Ok(Box::new(DropTracked {
                    inner: FakeModel::new(16),
                    dropped: track_drop,
                    drop_wait: None,
                }))
            })),
        });
        let rebuilding = std::thread::spawn(move || {
            ALLOCATIONS.with(|count| count.set(0));
            let resumed = e.rebuild();
            let allocations = ALLOCATIONS.with(|count| count.get());
            std::mem::forget(e);
            (resumed, allocations)
        });
        entered.recv_timeout(Duration::from_secs(1)).unwrap();
        crate::lifecycle::fault(
            &h.shared,
            (FaultCode::DeviceLost, EngineError::DeviceLost),
            crate::OnFault::Fail,
        );
        release.send(()).unwrap();
        let (resumed, allocations) = rebuilding.join().unwrap();
        assert_eq!(allocations, 0, "terminal fault forbids new fit allocations");
        assert_eq!(
            response.recv_timeout(Duration::from_secs(1)).unwrap(),
            Err(EngineError::DeviceLost)
        );
        assert!(!resumed);
        assert!(
            !dropped.load(Ordering::Acquire),
            "late model must be retained after fault"
        );
        assert_eq!(h.health().state, HealthState::Failed);
    }

    #[test]
    fn timing_query_faults_reach_the_fault_path() {
        for error in [
            KError::DeviceLost,
            KError::Timeout,
            KError::Oom,
            KError::Launch,
        ] {
            assert_eq!(super::gpu_time(Err(error)), Err(error));
        }
        assert_eq!(super::gpu_time(Ok(42)), Ok(Some(42)));
        assert_eq!(super::gpu_time(Err(KError::Unsupported)), Ok(None));
    }

    #[test]
    fn fault_retains_ab_end_reason_and_steps() {
        use larch_records::{AbCommand, AbEnd, Allow, Authority, FaultCode, KnobDelta};
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let (mut e, _) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        let mut deltas = [KnobDelta::default(); 4];
        deltas[0] = KnobDelta {
            knob: knob_id::MAX_BATCH_SEQS,
            value: 8.0,
        };
        e.apply_ab(
            AbCommand::Start {
                b: deltas,
                period_steps: 2,
                max_steps: 0,
                trial_id: 9,
                authority: Authority::Gait,
                allow: Allow::NONE,
            },
            0,
        )
        .unwrap();
        e.step = 3;
        e.publish();
        crate::lifecycle::fault(
            &e.sh,
            (FaultCode::Panic, EngineError::Internal(0)),
            crate::OnFault::Fail,
        );
        e.end_ab(AbEnd::Stop);
        assert_eq!(
            e.sh.host.lock().unwrap().ab_done,
            Some((9, 3, AbEnd::Fault))
        );
    }

    /// Only the listed archs default to replay: sm_75 and sm_121 stay eager.
    #[test]
    fn startup_graphs_default_only_on_measured_cuda_arches() {
        for backend in [Backend::Cuda, Backend::Hip, Backend::Metal, Backend::Cpu] {
            for arch in [0, 75, 86, 89, 120, 121] {
                for capable in [false, true] {
                    let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
                    let (cfg, rejected) = startup_config(&mut opts, backend, capable, arch, false);
                    let on = backend == Backend::Cuda && capable && matches!(arch, 86 | 89 | 120);
                    assert_eq!(
                        cfg.values[knob_id::DECODE_GRAPHS as usize],
                        f64::from(u8::from(on)),
                        "{backend:?} sm_{arch} capable {capable}"
                    );
                    assert!(rejected.is_empty());
                }
            }
        }
    }

    /// A card profile that proved graphs turns the default on where the device captures them, on
    /// any CUDA arch (no profile: the test above); an override still wins.
    #[test]
    fn a_card_profile_turns_graphs_on_off_the_measured_arches() {
        for backend in [Backend::Cuda, Backend::Hip, Backend::Metal, Backend::Cpu] {
            for arch in [0, 75, 86, 89, 120, 121] {
                for capable in [false, true] {
                    let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
                    let (cfg, _) = startup_config(&mut opts, backend, capable, arch, true);
                    assert_eq!(
                        cfg.values[knob_id::DECODE_GRAPHS as usize],
                        f64::from(u8::from(backend == Backend::Cuda && capable)),
                        "{backend:?} sm_{arch} capable {capable}"
                    );
                }
            }
        }
        let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
        opts.knob_overrides = vec![(knob_id::DECODE_GRAPHS, 0.0)];
        let (cfg, _) = startup_config(&mut opts, Backend::Cuda, true, 121, true);
        assert_eq!(cfg.values[knob_id::DECODE_GRAPHS as usize], 0.0);
    }

    /// The streaming pool never decides the graphs: a pool that would drop the plan under the gate's
    /// ratio leaves the graphs on, is still admitted after them, and a load the gate refuses without
    /// a pool is refused with one too.
    #[test]
    fn the_streaming_pool_does_not_move_the_graph_gate() {
        let mut before = larch_kernels::Device::open(CPU).unwrap().props().unwrap();
        before.backend = larch_kernels::Backend::Cuda;
        before.flags = larch_kernels::DeviceProps::CAP_GRAPHS;
        (before.vram_total, before.vram_free) = (17 << 30, 15 << 30);
        let mut c = FakeModel::new(248_320).config().clone();
        (
            c.n_layers,
            c.kv_layers,
            c.n_kv_heads,
            c.head_dim,
            c.ctx_train,
        ) = (40, 10, 2, 256, 262_144);
        let room = |usable: u64| usable + (512 << 20);
        let (before, c) = (&before, &c);
        let plan = |free: u64| {
            move |graphs, want| {
                fit(
                    before, free, c, 2048, 0.85, 0, 16, graphs, false, false, want, 0,
                )
            }
        };
        let want = 800_000_000;
        // Graphs stay; the pool would take the KV pool under the floor, so it stays out.
        let (p, graphs) =
            gated_plan(plan(room(1_000_000_000)), true, true, false, want, false, 0).unwrap();
        assert!(graphs && p.graph_bytes > 0, "{p:?}");
        assert_eq!(p.stream_pool_bytes, 0, "{p:?}");
        // Under the gate without a pool: the graphs go, whatever the want.
        for want in [0, 100_000_000] {
            let (p, graphs) =
                gated_plan(plan(room(300_000_000)), true, true, false, want, false, 0).unwrap();
            assert!(!graphs && p.graph_bytes == 0, "{p:?}");
        }
        // The floor, either side of 68 % with graphs on (112 MiB) ...
        let (p, graphs) =
            gated_plan(plan(room(3_000_000_000)), true, true, false, want, false, 0).unwrap();
        assert!(graphs && p.graph_bytes > 0, "{p:?}");
        assert_eq!(p.stream_pool_bytes, want, "69 %: {p:?}");
        let (p, graphs) =
            gated_plan(plan(room(2_700_000_000)), true, true, false, want, false, 0).unwrap();
        assert!(graphs && p.graph_bytes > 0, "{p:?}");
        assert_eq!(p.stream_pool_bytes, 0, "66 %: {p:?}");
        // An operator who set the crossover accepted the KV cost: past one context and the graphs
        // the floor does not apply, and a pool that does not fit there stays out all the same.
        let (p, graphs) =
            gated_plan(plan(room(1_000_000_000)), true, true, false, want, true, 0).unwrap();
        assert!(graphs && p.graph_bytes > 0, "{p:?}");
        assert_eq!(p.stream_pool_bytes, want, "opted in: {p:?}");
        let (p, _) = gated_plan(plan(room(900_000_000)), true, true, false, want, true, 0).unwrap();
        assert_eq!(p.stream_pool_bytes, 0, "opted in, no room: {p:?}");
        // ... and with graphs off and no gate, against the same plan without the pool.
        let (p, _) = gated_plan(
            plan(room(3_000_000_000)),
            false,
            false,
            false,
            want,
            false,
            0,
        )
        .unwrap();
        assert_eq!(p.stream_pool_bytes, want, "73 %: {p:?}");
        let (p, _) = gated_plan(
            plan(room(2_000_000_000)),
            false,
            false,
            false,
            want,
            false,
            0,
        )
        .unwrap();
        assert_eq!(p.stream_pool_bytes, 0, "60 %: {p:?}");
    }

    /// A model whose pool build fails with `error`, recording how many device buffers existed then.
    struct PoolFails(FakeModel, KError, Arc<AtomicU64>);

    impl Model for PoolFails {
        fn config(&self) -> &ModelConfig {
            self.0.config()
        }
        fn forward(
            &mut self,
            s: &Stream,
            b: &StepBatch,
            kv: &KvPool,
            logits: &Tensor,
        ) -> Result<(), KError> {
            self.0.forward(s, b, kv, logits)
        }
        fn stream_pool_want(&self) -> (u64, u32) {
            (1 << 20, 8)
        }
        fn alloc_stream_pool(&mut self, _s: &Stream) -> Result<bool, KError> {
            let made = ALLOCATIONS.with(|count| count.get());
            self.2.store(made + 1, Ordering::Relaxed);
            Err(self.1)
        }
    }

    /// `(load result, buffers allocated before the pool + 1; 0 when it was never tried)`.
    fn load_with_pool_failing(error: KError) -> (Option<LoadError>, u64) {
        let tried = Arc::new(AtomicU64::new(0));
        let t = tried.clone();
        ALLOCATIONS.with(|count| count.set(0));
        let err = Engine::init(
            EngineOptions::new(PathBuf::from("fake"), CPU),
            Box::new(move |_, _, _| Ok(Box::new(PoolFails(FakeModel::new(64), error, t.clone())))),
            None,
        )
        .err();
        (err, tried.load(Ordering::Relaxed))
    }

    /// Out of memory for the optional pool is no pool, the CPU path, and comes after the mandatory
    /// KV and serving buffers, so it can never refuse a load.
    #[test]
    fn a_failed_stream_pool_still_loads() {
        let (err, tried) = load_with_pool_failing(KError::Oom);
        assert!(err.is_none(), "{err:?}");
        assert!(
            tried >= 6,
            "the pool came before the serving buffers: {tried}"
        );
    }

    /// Anything else while building the pool is the device's: the load fails with it.
    #[test]
    fn a_device_error_building_the_stream_pool_fails_the_load() {
        let (err, tried) = load_with_pool_failing(KError::DeviceLost);
        assert!(tried > 0);
        assert!(
            matches!(err, Some(LoadError::Backend(c)) if c == KError::DeviceLost.code()),
            "{err:?}"
        );
    }

    /// Every arch default is fit-gated unless set by hand; an eager default has nothing to gate.
    /// Graphs stay only while the pool keeps 68 % of its blocks: a pool that loses a quarter keeps
    /// them; one that loses a third (the 3050's m6) or 40 % turns them off.
    #[test]
    fn default_graphs_fit_gated() {
        for (arch, explicit, gated) in [
            (86, false, true),
            (120, false, true),
            (120, true, false),
            (89, false, true),
            (75, false, false),
        ] {
            let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
            let (cfg, _) = startup_config(&mut opts, Backend::Cuda, true, arch, false);
            assert_eq!(graphs_fit_gated(&cfg, explicit), gated, "sm_{arch}");
        }
        let plan = |kv_blocks| FitPlan {
            model_bytes: 0,
            scratch_bytes: 0,
            graph_bytes: 0,
            graph_arena_bytes: 0,
            stream_pool_bytes: 0,
            headroom_bytes: 0,
            kv_block_bytes: 1,
            kv_blocks,
            max_seqs_fit: 1,
        };
        assert!(graphs_fit(&plan(75), &plan(100), 0));
        assert!(graphs_fit(&plan(68), &plan(100), 0));
        assert!(!graphs_fit(&plan(67), &plan(100), 0));
        assert!(!graphs_fit(&plan(60), &plan(100), 0));
        // A projector's bytes count as pool on both sides: a pool the graphs leave at 60 % of 1,000
        // blocks drops them alone, but beside a projector worth 500 blocks it keeps 73 % of the pool
        // the text model would have had, and keeps them.
        assert!(graphs_fit(&plan(600), &plan(1000), 500));
        assert!(!graphs_fit(&plan(600), &plan(1000), 0));
        // InternVL3-1B's load on a 4 GB 3050 (graphs on 5,061 blocks, off 6,222, the projector
        // 2,947 blocks' worth) keeps its graphs; a pool graphs would really squeeze still drops them.
        assert!(graphs_fit(&plan(5061), &plan(6222), 2947));
        assert!(!graphs_fit(&plan(500), &plan(6222), 2947));
    }

    /// The measured hybrid MoE target load (5070 Ti, W4, max_context 2304): a host-expert plan that moved layers
    /// for the graphs' reserve keeps them at any pool share; the same share on a load no plan
    /// paid for is refused (the 3050's case); an `Err` from the fit always falls back. The fit's
    /// seats are the 3 state slots the plan admits at 4 host layers
    /// (`fit::tests::the_target_plan_fills_its_slack_with_state_slots`), which size its KV floor.
    #[test]
    fn a_paid_host_expert_load_keeps_graphs_at_any_pool_share() {
        let mut before = larch_kernels::Device::open(CPU).unwrap().props().unwrap();
        before.backend = larch_kernels::Backend::Cuda;
        before.flags = larch_kernels::DeviceProps::CAP_GRAPHS;
        (before.vram_total, before.vram_free) = (17_066_033_152, 15_631_122_432);
        let mut c = FakeModel::new(248_320).config().clone();
        (
            c.arch,
            c.n_layers,
            c.kv_layers,
            c.n_kv_heads,
            c.head_dim,
            c.ctx_train,
        ) = (Arch::Qwen35Moe, 40, 10, 2, 256, 262_144);
        let scratch = 72_719_104;
        let plan = |free, graphs| {
            crate::fit::fit(
                &before, free, &c, 2304, 0.85, scratch, 3, graphs, false, false, 0, 0,
            )
        };
        let mut seen = 0;
        for free in (300_000_000..700_000_000).step_by(5_000_000) {
            let (with, without) = (plan(free, true), plan(free, false).unwrap());
            let share = f64::from(with.as_ref().unwrap().kv_blocks) / f64::from(without.kv_blocks);
            if !(0.45..0.68).contains(&share) {
                continue;
            }
            seen += 1;
            assert!(
                super::keep_graphs(true, &with, &without, 0),
                "{free}: {share}"
            );
            assert!(
                !super::keep_graphs(false, &with, &without, 0),
                "{free}: {share}"
            );
        }
        assert!(seen > 10, "{seen} loads in the measured share range");
        let without = plan(600_000_000, false).unwrap();
        let refused = Err(LoadError::DoesNotFit { need: 1, have: 0 });
        assert!(!super::keep_graphs(true, &refused, &without, 0));
    }

    /// The target's loads as LARCH_FIT_DIAGNOSTICS logged them on the 5070 Ti (a hybrid MoE with 8
    /// host-expert layers, 10 KV layers of 2 heads of 256, 16 seats) keep fit-gated graphs.
    #[test]
    fn the_target_host_expert_load_keeps_graphs() {
        let mut before = larch_kernels::Device::open(CPU).unwrap().props().unwrap();
        before.backend = larch_kernels::Backend::Cuda;
        before.flags = larch_kernels::DeviceProps::CAP_GRAPHS;
        (before.vram_total, before.vram_free) = (17_066_033_152, 15_631_122_432);
        let mut c = FakeModel::new(248_320).config().clone();
        (
            c.n_layers,
            c.kv_layers,
            c.n_kv_heads,
            c.head_dim,
            c.ctx_train,
        ) = (40, 10, 2, 256, 262_144);
        // (max_context, free after the load, the serving need)
        for (ctx, free, need) in [
            (8192, 984_403_968, 807_063_231),
            (8192, 996_724_736, 807_063_231),
            (2304, 996_724_736, 665_102_862),
        ] {
            let scratch = need - crate::fit::serving_need(&before, &c, ctx, 0.85, 0, false, 16);
            let plan = |graphs| {
                fit(
                    &before, free, &c, ctx, 0.85, scratch, 16, graphs, false, false, 0, 0,
                )
                .unwrap()
            };
            let (with, without) = (plan(true), plan(false));
            assert!(with.graph_bytes > 0, "{ctx}: {with:?}");
            assert!(
                graphs_fit(&with, &without, 0),
                "{ctx}: {} of {}",
                with.kv_blocks,
                without.kv_blocks
            );
        }
    }

    #[test]
    fn startup_graphs_explicit_opt_out_wins_over_cuda_default() {
        for overrides in [
            vec![(knob_id::DECODE_GRAPHS, 0.0)],
            vec![(knob_id::DECODE_GRAPHS, 1.0), (knob_id::DECODE_GRAPHS, 0.0)],
        ] {
            let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
            opts.knob_overrides = overrides;
            let (cfg, rejected) = startup_config(&mut opts, Backend::Cuda, true, 120, false);
            assert_eq!(cfg.values[knob_id::DECODE_GRAPHS as usize], 0.0);
            assert!(rejected.is_empty());
        }
    }

    #[test]
    fn startup_graphs_accept_explicit_opt_in_on_capable_backends() {
        for (backend, arch) in [
            (Backend::Cuda, 86),
            (Backend::Cuda, 89),
            (Backend::Cuda, 120),
            (Backend::Hip, 0),
        ] {
            let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
            opts.knob_overrides = vec![(knob_id::DECODE_GRAPHS, 1.0)];
            let (cfg, rejected) = startup_config(&mut opts, backend, true, arch, false);
            assert_eq!(cfg.values[knob_id::DECODE_GRAPHS as usize], 1.0);
            assert!(rejected.is_empty());
        }
    }

    #[test]
    fn startup_graphs_reject_opt_in_without_support() {
        for backend in [Backend::Cpu, Backend::Cuda, Backend::Hip, Backend::Metal] {
            let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
            opts.knob_overrides = vec![(knob_id::DECODE_GRAPHS, 1.0)];
            let (cfg, rejected) = startup_config(&mut opts, backend, false, 120, false);
            assert_eq!(cfg.values[knob_id::DECODE_GRAPHS as usize], 0.0);
            assert_eq!(
                rejected,
                vec![(knob_id::DECODE_GRAPHS, 1.0, RejectReason::Unsupported)]
            );
        }
    }

    /// FakeModel with another config: the fit sees the KV this config needs.
    struct Shaped(FakeModel, ModelConfig);

    impl Model for Shaped {
        fn config(&self) -> &ModelConfig {
            &self.1
        }

        fn forward(
            &mut self,
            s: &Stream,
            b: &StepBatch,
            kv: &KvPool,
            logits: &Tensor,
        ) -> Result<(), KError> {
            self.0.forward(s, b, kv, logits)
        }
    }

    /// With a projector the context is a ceiling: one the device cannot hold (16 GiB of KV against
    /// the CPU reference's 2 GiB) is halved at load until the pool holds it, without loading again.
    #[test]
    fn with_a_projector_the_context_halves_to_what_fits() {
        let d = FakeModel::new(64).config().d_model;
        let dir = std::env::temp_dir().join(format!("larch-ctx-clamp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mmproj = dir.join("mmproj.gguf");
        std::fs::write(&mmproj, tiny::Projector::new(d, 3).gguf()).unwrap();
        let loads = Arc::new(Mutex::new(0));
        let counted = loads.clone();
        let load = Box::new(move |_: &_, _: &_, _| {
            *counted.lock().unwrap() += 1;
            let f = FakeModel::new(64);
            let mut c = f.config().clone();
            (
                c.n_layers,
                c.kv_layers,
                c.n_kv_heads,
                c.head_dim,
                c.ctx_train,
            ) = (32, 32, 8, 128, 131_072);
            Ok(Box::new(Shaped(f, c)) as Box<dyn Model>)
        });
        let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
        opts.max_context = 131_072;
        opts.mmproj = Some(mmproj);
        let engine = start_with(opts, load).expect("a smaller context fits");
        let ctx = engine.deployment().ctx_max;
        assert!(
            (crate::VISION_MIN_CONTEXT..131_072).contains(&ctx) && ctx.is_power_of_two(),
            "{ctx}"
        );
        assert_eq!(*loads.lock().unwrap(), 1, "the weights were read once");
        engine.shutdown();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A context raised for a projector is halved rather than kept without decode graphs; the
    /// caller's own context, a load without a projector, a plan with graphs, and graphs off are not.
    #[test]
    fn a_raised_vision_context_gives_way_to_the_decode_graphs() {
        let (none, some) = (false, true);
        use crate::engine::trade_context_for_graphs;
        assert!(trade_context_for_graphs(true, none, true, 32_768, 8_192));
        assert!(
            !trade_context_for_graphs(true, some, true, 32_768, 8_192),
            "graphs reserved"
        );
        assert!(
            !trade_context_for_graphs(false, none, true, 32_768, 8_192),
            "graphs off"
        );
        assert!(
            !trade_context_for_graphs(true, none, false, 32_768, 8_192),
            "no projector"
        );
        assert!(
            !trade_context_for_graphs(true, none, true, 32_768, 32_768),
            "the caller's own"
        );
        assert!(
            !trade_context_for_graphs(true, none, true, crate::VISION_MIN_CONTEXT, 0),
            "the floor"
        );
    }

    /// A projector raises the context to `VISION_MAX_CONTEXT`; a reload to another model drops the
    /// projector and asks for the caller's context again, not the raised one.
    #[test]
    fn a_reload_without_the_projector_asks_for_the_callers_context() {
        let d = FakeModel::new(64).config().d_model;
        let dir = std::env::temp_dir().join(format!("larch-ctx-reload-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mmproj = dir.join("mmproj.gguf");
        std::fs::write(&mmproj, tiny::Projector::new(d, 3).gguf()).unwrap();
        let load = || {
            Box::new(|_: &_, _: &_, _| {
                let f = FakeModel::new(64);
                let mut c = f.config().clone();
                c.ctx_train = 131_072;
                Ok(Box::new(Shaped(f, c)) as Box<dyn Model>)
            })
        };
        let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
        opts.max_context = 4096;
        opts.mmproj = Some(mmproj);
        let engine = start_with(opts, load()).expect("a small model fits");
        assert_eq!(engine.deployment().ctx_max, crate::VISION_MAX_CONTEXT);
        let change = ReloadChange {
            model: Some(PathBuf::from("other")),
            ..ReloadChange::default()
        };
        engine.reload_with(change, load()).unwrap();
        assert!(
            engine.vision().is_none(),
            "another model loads without the projector"
        );
        assert_eq!(engine.deployment().ctx_max, 4096);
        engine.shutdown();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A model file's id is the same on every read and differs for other bytes (sha-256 of fixed
    /// fields, so a `.pin` file keeps it across builds).
    #[test]
    fn a_file_id_tells_files_apart_and_repeats() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        std::fs::write(&a, vec![1u8; 200_000]).unwrap();
        std::fs::write(&b, vec![2u8; 200_000]).unwrap();
        assert!(super::file_id(&a).is_some());
        assert_eq!(super::file_id(&a), super::file_id(&a));
        assert_ne!(super::file_id(&a), super::file_id(&b));
    }

    /// A full context the device cannot hold is refused at load with what it needs: 16 GiB of KV
    /// (128 KiB a token × 131072) against the CPU reference's 2 GiB.
    #[test]
    fn a_context_past_the_budget_is_refused() {
        let load = Box::new(|_: &_, _: &_, _| {
            let f = FakeModel::new(64);
            let mut c = f.config().clone();
            (
                c.n_layers,
                c.kv_layers,
                c.n_kv_heads,
                c.head_dim,
                c.ctx_train,
            ) = (32, 32, 8, 128, 131_072);
            Ok(Box::new(Shaped(f, c)) as Box<dyn Model>)
        });
        let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
        opts.max_context = 131_072;
        match start_with(opts, load) {
            Err(LoadError::DoesNotFit { need, have }) => {
                assert!(need > have && have == 2 << 30, "{need} {have}")
            }
            Ok(_) => panic!("loaded"),
            Err(e) => panic!("{e:?}"),
        }
    }

    /// Serving fit keeps one full context and the startup graph reserve at every chunk size, on the
    /// real 7.6 GB free of an 8 GB card. Graphs off keep 733 blocks; the 12 slots of the default 16
    /// seats plus the 64 MiB capture arena cost 48 (#107 kept 672 with 9 slots of 16 MiB).
    #[test]
    fn serving_buffers_fit_beside_a_full_context() {
        let mut before = larch_kernels::Device::open(CPU).unwrap().props().unwrap();
        before.backend = larch_kernels::Backend::Cuda;
        before.flags = larch_kernels::DeviceProps::CAP_GRAPHS;
        before.vram_total = 8 << 30;
        before.vram_free = 7_600_000_000;
        let extra = if larch_kernels::routes().mixed_graphs {
            4 * GRAPH_BYTES
        } else {
            0
        };
        let free = before.vram_free - 4_920_739_232 - 300_000_000;
        let mut c = FakeModel::new(128_256).config().clone();
        (
            c.n_layers,
            c.kv_layers,
            c.n_kv_heads,
            c.head_dim,
            c.ctx_train,
        ) = (32, 32, 8, 128, 131_072);
        for chunk in [512, 1024, 2048] {
            let dims = Dims {
                rows: 64 + chunk,
                seqs: 64,
                max_blocks: 512,
                vocab: 128_256,
                tap_seats: 0,
                tap_floats: 0,
            };
            // Graphs off at load reserve no graph memory; on, the 12 slots of the default 16 seats
            // and the capture arena.
            let off = fit(
                &before,
                free,
                &c,
                8192,
                0.85,
                dims.fit_bytes(),
                16,
                false,
                false,
                false,
                0,
                0,
            )
            .unwrap();
            assert_eq!(
                (off.kv_blocks, off.graph_bytes, off.graph_arena_bytes),
                (733, 0, 0),
                "chunk {chunk}"
            );
            let p = fit(
                &before,
                free,
                &c,
                8192,
                0.85,
                dims.fit_bytes(),
                16,
                true,
                false,
                false,
                0,
                0,
            )
            .unwrap();
            assert!(p.kv_blocks >= 512 && p.max_seqs_fit == 1, "{p:?}");
            // 2 MiB blocks: 12 slots of GRAPH_BYTES (4 MiB) are 24 of them, the 64 MiB arena 32
            // more; 96 is a loose bound.
            assert!(p.kv_blocks >= 672 - 96, "chunk {chunk}: {p:?}");
            if extra == 0 {
                assert_eq!(p.kv_blocks, 685, "chunk {chunk}");
            }
            assert_eq!(p.graph_bytes, 12 * GRAPH_BYTES + extra);
            assert_eq!(p.graph_arena_bytes, 64 << 20);
            let larger = Dims {
                max_blocks: 1024,
                ..dims
            };
            match fit(
                &before,
                free,
                &c,
                16_384,
                0.85,
                larger.fit_bytes(),
                16,
                true,
                false,
                false,
                0,
                0,
            ) {
                Err(LoadError::DoesNotFit { need, have }) => {
                    assert!(need > have && have == before.vram_free, "{need} {have}");
                }
                r => panic!("{r:?}"),
            }
        }
    }

    /// Parity's prefill mode on an 8 GiB card with graphs (the AMD review's RX 6700S runs m4): where
    /// the longest prompt's logits (1,884 rows of a ~152k vocabulary, 1.14 GB of f32) do not fit
    /// beside the KV pool, headroom and arena at once, observation proceeds in the row chunks that do.
    #[test]
    fn eight_gib_parity_prefill_observes_in_row_chunks() {
        // (vocab, layers, kv heads, head dim, weights + activation scratch for 1,984 rows: estimates,
        // whether all 1,884 rows still need chunks once unused graph slots are spent)
        for (vocab, layers, kv_heads, head_dim, model, chunks) in [
            (151_936, 24, 2, 64, 220_000_000, false),
            (152_064, 28, 4, 128, 5_300_000_000, true),
        ] {
            for backend in [larch_kernels::Backend::Hip, larch_kernels::Backend::Cuda] {
                let mut before = larch_kernels::Device::open(CPU).unwrap().props().unwrap();
                before.backend = backend;
                before.flags = larch_kernels::DeviceProps::CAP_GRAPHS;
                (before.vram_total, before.vram_free) = (8 << 30, 8_200_000_000);
                let free = before.vram_free - model;
                let mut c = FakeModel::new(vocab).config().clone();
                (
                    c.n_layers,
                    c.kv_layers,
                    c.n_kv_heads,
                    c.head_dim,
                    c.ctx_train,
                ) = (layers, layers, kv_heads, head_dim, 32_768);
                // parity --mode prefill: max_context = the longest prompt, one 1,920-token chunk.
                let dims = Dims {
                    rows: 64 + 1920,
                    seqs: 64,
                    max_blocks: 1884usize.div_ceil(crate::sched::BLOCK_TOKENS),
                    vocab: vocab as usize,
                    tap_seats: 0,
                    tap_floats: 0,
                };
                let p = fit(
                    &before,
                    free,
                    &c,
                    1884,
                    0.85,
                    dims.fit_bytes(),
                    16,
                    true,
                    false,
                    false,
                    0,
                    0,
                )
                .unwrap();
                let left = free - u64::from(p.kv_blocks) * p.kv_block_bytes - dims.fit_bytes();
                // With every graph slot kept, no model fits all rows and each chunks.
                let full = p.headroom_bytes + p.graph_bytes + p.graph_arena_bytes;
                assert!(
                    !dims.growth_fits(left, 64, 1884, full),
                    "{backend:?} {vocab}"
                );
                let kept = dims.rows_fitting(left, 64, full);
                assert!((65..1884).contains(&kept), "{backend:?} {vocab}: {kept}");
                assert!(!dims.growth_fits(left, 64, kept + 1, full));
                // Unused graph slots are spent on rows; headroom and the arena never are.
                let reserve = p.headroom_bytes + p.graph_arena_bytes;
                let rows = dims.rows_fitting(left, 64, reserve);
                assert!(rows > kept, "{backend:?} {vocab}: {rows} vs {kept}");
                assert!(dims.growth_fits(left, 64, rows, reserve));
                assert_eq!(
                    !dims.growth_fits(left, 64, 1884, reserve),
                    chunks,
                    "{backend:?} {vocab}"
                );
                if chunks {
                    assert!((65..1884).contains(&rows), "{backend:?} {vocab}: {rows}");
                    assert!(!dims.growth_fits(left, 64, rows + 1, reserve));
                }
            }
        }
    }

    #[test]
    fn forced_growth_credits_old_buffers_and_preserves_selected_reserve() {
        let dims = Dims {
            rows: 576,
            seqs: 64,
            max_blocks: 512,
            vocab: 128_256,
            tap_seats: 0,
            tap_floats: 0,
        };
        // Independently counted: old input/logits/sample = 155392 + 32833536 + 33024;
        // new = 288512 + 295501824 + 297216. Net growth is 263065600 bytes.
        // Example free figures: 1222165344 and 909689696 bytes.
        assert!(dims.growth_fits(1_222_165_344, 64, 576, 536_870_912));
        assert!(!dims.growth_fits(1_222_165_344, 64, 576, 1_023_410_176));
        assert!(dims.growth_fits(909_689_696, 64, 576, 536_870_912));
        // The actual 512-row prefill (263204352 bytes) fits where unused maximum decode rows do not.
        assert!(!dims.growth_fits(909_689_696, 64, 576, 654_311_424));
        assert!(dims.growth_fits(909_689_696, 64, 512, 654_311_424));
        assert!(dims.growth_fits(884_493_824, 64, 512, 654_311_424));
        assert!(!dims.growth_fits(884_493_823, 64, 512, 654_311_424));
        // Reserving all 29 graphs keeps their future captures from competing with forced buffers.
        for (reserve, required) in [(536_870_912, 799_936_512), (1_023_410_176, 1_286_475_776)] {
            assert!(dims.growth_fits(required, 64, 576, reserve));
            assert!(!dims.growth_fits(required - 1, 64, 576, reserve));
        }
    }

    #[test]
    fn short_forced_prefill_does_not_reserve_unused_chunk_rows() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 64;
        opts.knob_overrides = vec![(knob_id::PREFILL_CHUNK_TOKENS, 4096.0)];
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(152_064)))),
            None,
        )
        .unwrap();
        let forced = h
            .force(Forced {
                tokens: vec![1, 2],
                prompt: 2,
                session: 0,
                keep_kv: false,
            })
            .unwrap();
        assert!(e.turn().unwrap());
        assert!(matches!(
            forced.request.events.try_recv().unwrap(),
            EngineEvent::Done { .. }
        ));
        assert_eq!(forced.positions.try_iter().count(), 2);
        assert_eq!(e.loaded.as_ref().unwrap().logit_rows, 64);
        assert_eq!(h.health().state, HealthState::Ready);
    }

    #[test]
    fn forced_prefill_grows_for_combined_scheduled_rows() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 128;
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        let handles: Vec<_> = [vec![1; 40], vec![2; 40]]
            .into_iter()
            .map(|tokens| {
                h.force(Forced {
                    tokens,
                    prompt: 40,
                    session: 0,
                    keep_kv: false,
                })
                .unwrap()
            })
            .collect();
        assert!(e.turn().unwrap());
        for handle in handles {
            assert!(matches!(
                handle.request.events.try_recv().unwrap(),
                EngineEvent::Done { .. }
            ));
            assert_eq!(handle.positions.try_iter().count(), 40);
        }
        assert_eq!(e.loaded.as_ref().unwrap().logit_rows, 80);
    }

    /// Graphs switched on live after a load that reserved none must still fit beside the headroom;
    /// otherwise the knob is refused, never over-committed, and the reserve counts once accepted.
    #[test]
    fn live_graphs_on_without_their_reserve_is_refused() {
        use larch_records::{Authority, KnobReply, KnobRequest};
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let (mut e, _h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        // The CPU reference has no graphs: pretend it captures them, after a load with graphs off.
        e.sh.info.write().unwrap().caps.graph_replay = true;
        let free = e.s.device().props().unwrap().vram_free;
        assert_eq!(e.loaded.as_ref().unwrap().graph_bytes, 0);
        let on = KnobRequest::new(knob_id::DECODE_GRAPHS, 1.0, Authority::Human);
        e.loaded.as_mut().unwrap().headroom_bytes = free - (16 << 20);
        assert_eq!(
            e.apply_set(&on, 0).unwrap(),
            KnobReply::Rejected {
                reason: RejectReason::ExceedsFit { limit: 0.0 }
            }
        );
        assert_eq!(e.value(knob_id::DECODE_GRAPHS), 0.0);
        e.loaded.as_mut().unwrap().headroom_bytes = free / 2;
        assert!(matches!(
            e.apply_set(&on, 0).unwrap(),
            KnobReply::Applied { .. }
        ));
        assert_eq!(e.value(knob_id::DECODE_GRAPHS), 1.0);
        assert!(e.loaded.as_ref().unwrap().graph_bytes > 0);
    }

    /// A revert across a load that reserved no graph memory goes through the same reserve check as
    /// a set; observation growth spends unused graph slots (the quota shrinks), never the headroom
    /// or the arena, whether graphs are on or off.
    #[test]
    fn graph_revert_is_checked_and_observation_spends_unused_slots() {
        use larch_records::{Authority, KnobReply, KnobRequest, RevertTarget};
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let (mut e, _h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        e.sh.info.write().unwrap().caps.graph_replay = true;
        let free = e.s.device().props().unwrap().vram_free;
        e.loaded.as_mut().unwrap().headroom_bytes = free / 2;
        let set = |e: &mut Engine, v| {
            e.apply_set(
                &KnobRequest::new(knob_id::DECODE_GRAPHS, v, Authority::Human),
                0,
            )
            .unwrap()
        };
        assert!(matches!(set(&mut e, 1.0), KnobReply::Applied { .. }));
        let KnobReply::Applied { rev, .. } = set(&mut e, 0.0) else {
            panic!("graphs off refused");
        };
        // Headroom and arena that leave 100 rows: the unused graph slots do not shorten them.
        let loaded = e.loaded.as_mut().unwrap();
        let bytes = |rows| loaded.dims.bytes(rows).into_iter().sum::<u64>();
        loaded.headroom_bytes =
            free + bytes(loaded.logit_rows) - bytes(100) - loaded.graph_arena_bytes;
        assert!(loaded.graph_bytes > 0);
        assert_eq!(e.observe_budget().unwrap(), 100);
        // A load with graphs off reserves nothing; reverting to graphs on must then fit.
        let loaded = e.loaded.as_mut().unwrap();
        (loaded.graph_bytes, loaded.headroom_bytes) = (0, free - (16 << 20));
        assert_eq!(
            e.apply_revert(RevertTarget::Rev(rev), Authority::Human, 0)
                .unwrap(),
            KnobReply::Rejected {
                reason: RejectReason::ExceedsFit { limit: 0.0 }
            }
        );
        assert_eq!(e.value(knob_id::DECODE_GRAPHS), 0.0);
    }

    /// An observed turn whose rows the budget cannot hold at once runs in row chunks, so the parked
    /// KV it took over from its session is never freed by a refusal: the next turn reuses it all.
    #[test]
    fn parked_session_survives_observation_budget() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        let done = |e: &mut Engine, events: &std::sync::mpsc::Receiver<EngineEvent>| loop {
            assert!(e.turn().unwrap());
            if let Some(event) = events.try_iter().find(|ev| {
                !matches!(
                    ev,
                    EngineEvent::Token { .. } | EngineEvent::Prefilled { .. }
                )
            }) {
                return event;
            }
        };
        let tokens: Vec<u32> = (0..355).map(|i| i % 60 + 1).collect();
        let first = h
            .submit(Request {
                session: 7,
                keep_kv: true,
                ..Request::new(tokens[..200].to_vec(), 1)
            })
            .unwrap();
        assert!(matches!(
            done(&mut e, &first.events),
            EngineEvent::Done { .. }
        ));
        // Headroom that leaves room for 100 logits rows: the next turn's 150 observed rows do not fit
        // (one growth: the CPU reference's fixed free figure budgets only the first as a GPU would).
        let free = e.s.device().props().unwrap().vram_free;
        let loaded = e.loaded.as_mut().unwrap();
        let bytes = |rows| loaded.dims.bytes(rows).into_iter().sum::<u64>();
        loaded.headroom_bytes = free + bytes(loaded.logit_rows) - bytes(100);
        let forced = h
            .force(Forced {
                tokens: tokens[..350].to_vec(),
                prompt: 350,
                session: 7,
                keep_kv: true,
            })
            .unwrap();
        match done(&mut e, &forced.request.events) {
            EngineEvent::Done { usage, .. } => assert_eq!(usage.cached_tokens, 200),
            event => panic!("{event:?}"),
        }
        let positions: Vec<_> = forced.positions.try_iter().map(|p| p.pos).collect();
        assert_eq!(positions, (200..350).collect::<Vec<_>>());
        assert!(e.loaded.as_ref().unwrap().logit_rows <= 100);
        let next = h
            .submit(Request {
                session: 7,
                ..Request::new(tokens.clone(), 1)
            })
            .unwrap();
        match done(&mut e, &next.events) {
            EngineEvent::Done { usage, .. } => assert_eq!(usage.cached_tokens, 350),
            event => panic!("{event:?}"),
        }
        assert_eq!(h.health().state, HealthState::Ready);
    }

    /// A forced prefill past the budget observes every position in the row chunks the budget holds,
    /// in the same steps as ordinary work, which completes unchanged.
    #[test]
    fn forced_growth_past_the_budget_runs_in_row_chunks() {
        let load = Box::new(|_: &_, _: &_, _| Ok(Box::new(FakeModel::new(64)) as Box<dyn Model>));
        let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
        opts.max_context = 512;
        let (mut e, h) = Engine::init(opts, load, None).unwrap();
        let free = e.s.device().props().unwrap().vram_free;
        let loaded = e.loaded.as_mut().unwrap();
        let bytes = |rows| loaded.dims.bytes(rows).into_iter().sum::<u64>();
        // The CPU reference reports a fixed free figure that growth does not lower, so only the first
        // growth is budgeted as on a GPU: 150 rows need exactly one.
        loaded.headroom_bytes = free + bytes(loaded.logit_rows) - bytes(100);
        assert!(
            !loaded
                .dims
                .growth_fits(free, 64, 150, loaded.headroom_bytes)
        );
        let forced = h
            .force(Forced {
                tokens: (0..150).map(|i| i % 60 + 1).collect(),
                prompt: 150,
                session: 0,
                keep_kv: false,
            })
            .unwrap();
        let ordinary = h.submit(Request::new(vec![5], 2)).unwrap();
        let thread = std::thread::spawn(move || {
            e.main_loop();
            e.loaded.as_ref().unwrap().logit_rows
        });
        // Always stop and join the engine, including when a request assertion fails.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let wait = std::time::Duration::from_secs(5);
            assert!(matches!(
                forced.request.events.recv_timeout(wait).unwrap(),
                EngineEvent::Done { .. }
            ));
            let positions: Vec<_> = forced.positions.try_iter().map(|p| p.pos).collect();
            assert_eq!(positions, (0..150).collect::<Vec<_>>());
            let mut tokens = Vec::new();
            loop {
                match ordinary.events.recv_timeout(wait).unwrap() {
                    EngineEvent::Token { id, .. } => tokens.push(id),
                    EngineEvent::Done { finish, usage, .. } => {
                        assert_eq!(finish, Finish::Length);
                        assert_eq!(usage.completion_tokens, 2);
                        break;
                    }
                    EngineEvent::Prefilled { .. } => {}
                    ev => panic!("{ev:?}"),
                }
            }
            assert_eq!(tokens, vec![6, 7]);
            assert_eq!(h.health().state, HealthState::Ready);
        }));
        h.shutdown();
        let rows = thread.join().unwrap();
        assert!((65..=100).contains(&rows), "{rows}");
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    #[test]
    fn forced_growth_refuses_an_unfunded_allocation_peak() {
        let dims = Dims {
            rows: 576,
            seqs: 64,
            max_blocks: 512,
            vocab: 128_256,
            tap_seats: 0,
            tap_floats: 0,
        };
        // The new buffers total 296219136 bytes, while net growth is only 263065600.
        // A free-memory figure just below the new total funds net growth but not the swap peak.
        assert!(!dims.growth_fits(296_219_135, 64, 576, 0));
        assert!(dims.growth_fits(296_219_136, 64, 576, 0));
    }

    /// #107's growth rules merged with this branch's: an observed prefill grows only to the rows
    /// its step schedules and pays with graph slots, never headroom or the arena; the budget never
    /// admits new buffers past the fresh free bytes, even when old buffers would fund net growth.
    #[test]
    fn observed_step_grows_to_scheduled_rows_from_fresh_bytes_and_shrinks_quota() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 4096;
        opts.knob_overrides = vec![(knob_id::PREFILL_CHUNK_TOKENS, 4096.0)];
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(152_064)))),
            None,
        )
        .unwrap();
        // The CPU reference reports a constant 2 GiB free, so these budgets are exact.
        let free = e.s.device().props().unwrap().vram_free;
        let loaded = e.loaded.as_mut().unwrap();
        let sum = |d: &Dims, rows| d.bytes(rows).into_iter().sum::<u64>();
        let (old_rows, dims) = (loaded.logit_rows, loaded.dims);
        let growth = sum(&dims, 80) - sum(&dims, old_rows);
        loaded.graph_bytes = 29 * GRAPH_BYTES;
        loaded.graph_arena_bytes = 64 << 20;
        // Two whole slots and one byte short of a third remain after growing to 80 rows.
        loaded.headroom_bytes = free - growth - loaded.graph_arena_bytes - 3 * GRAPH_BYTES + 1;
        let reserve = loaded.headroom_bytes + loaded.graph_arena_bytes;
        // Growing to every row a step can have, or keeping the whole quota, would refuse.
        assert!(!dims.growth_fits(free, old_rows, dims.rows, reserve));
        assert!(!dims.growth_fits(free, old_rows, 80, reserve + loaded.graph_bytes));
        let forced = |tokens: usize| {
            h.force(Forced {
                tokens: vec![1; tokens],
                prompt: tokens as u32,
                session: 0,
                keep_kv: false,
            })
            .unwrap()
        };
        let grown = forced(80);
        assert!(e.turn().unwrap());
        assert!(matches!(
            grown.request.events.try_recv().unwrap(),
            EngineEvent::Done { .. }
        ));
        assert_eq!(grown.positions.try_iter().count(), 80);
        let loaded = e.loaded.as_mut().unwrap();
        assert_eq!(loaded.logit_rows, 80);
        assert_eq!(loaded.graph_bytes, 2 * GRAPH_BYTES);
        // 3560 rows of new buffers exceed the 2 GiB free, while the 80-row buffers they replace
        // would fund the net growth with no reserve at all: the budget stops at the fresh bytes.
        assert!(sum(&dims, 3560) > free && free + sum(&dims, 80) >= sum(&dims, 3560));
        let rows = dims.rows_fitting(free, 80, 0);
        assert!(rows < 3560 && sum(&dims, rows) <= free && sum(&dims, rows + 1) > free);
    }

    /// A budget one byte short of 80 rows beside the headroom and arena runs the forced prompt in
    /// row chunks; neither is spent and ordinary work completes.
    #[test]
    fn forced_growth_runs_in_chunks_within_headroom_and_arena() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let (mut e, h) = Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        let free = e.s.device().props().unwrap().vram_free;
        let loaded = e.loaded.as_mut().unwrap();
        let old_rows = loaded.logit_rows;
        let old = loaded.dims.bytes(old_rows);
        // An 80-token observed prefill needs 80 logits rows, past the 64 serving rows.
        let growth = loaded.dims.bytes(80).into_iter().sum::<u64>() - old.into_iter().sum::<u64>();
        loaded.graph_bytes = 0;
        loaded.graph_arena_bytes = 64 << 20;
        loaded.headroom_bytes = free - growth - loaded.graph_arena_bytes + 1;
        let forced = h
            .force(Forced {
                tokens: vec![1; 80],
                prompt: 80,
                session: 0,
                keep_kv: false,
            })
            .unwrap();
        let ordinary = h.submit(Request::new(vec![5], 2)).unwrap();
        assert!(e.turn().unwrap());
        while !e.sched.is_idle() {
            assert!(e.turn().unwrap());
        }
        assert!(matches!(
            forced.request.events.try_recv().unwrap(),
            EngineEvent::Done { .. }
        ));
        assert_eq!(forced.positions.try_iter().count(), 80);
        assert_eq!(h.health().state, HealthState::Ready);
        let loaded = e.loaded.as_ref().unwrap();
        assert!((old_rows..80).contains(&loaded.logit_rows));
        assert_eq!(loaded.graph_bytes, 0);
        assert_eq!(loaded.graph_arena_bytes, 64 << 20);
        assert!(
            matches!(ordinary.events.try_iter().last(), Some(EngineEvent::Done { usage, .. }) if usage.completion_tokens == 2)
        );
    }

    #[test]
    fn forced_growth_keeps_future_graph_reserve_while_graphs_are_off() {
        for (slots, remainder) in [(0, 0), (3, 0), (3, GRAPH_BYTES - 1)] {
            let mut opts = EngineOptions::new("fake".into(), CPU);
            opts.max_context = 512;
            let (mut e, _) = Engine::init(
                opts,
                Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
                None,
            )
            .unwrap();
            let free = e.s.device().props().unwrap().vram_free;
            let loaded = e.loaded.as_mut().unwrap();
            let old_rows = loaded.logit_rows;
            let old = loaded.dims.bytes(old_rows);
            let growth = loaded.dims.bytes(loaded.dims.rows).into_iter().sum::<u64>()
                - old.into_iter().sum::<u64>();
            // Force growth to leave only a declared number of whole future executable slots.
            loaded.graph_bytes = 29 * GRAPH_BYTES;
            loaded.graph_arena_bytes = 64 << 20;
            loaded.headroom_bytes =
                free - growth - loaded.graph_arena_bytes - slots * GRAPH_BYTES - remainder;
            let headroom = loaded.headroom_bytes;
            let expected = loaded.dims.bytes(loaded.dims.rows);
            let rows = loaded.dims.rows;
            assert!(!loaded.dims.growth_fits(
                free,
                old_rows,
                rows,
                headroom + loaded.graph_bytes + loaded.graph_arena_bytes
            ));
            e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 0.0;
            e.observe_rows(rows).unwrap();
            for enabled in [1.0, 0.0, 1.0] {
                e.cfg.values[knob_id::DECODE_GRAPHS as usize] = enabled;
                e.observe_rows(rows).unwrap();
                let loaded = e.loaded.as_ref().unwrap();
                assert_eq!(loaded.graph_bytes, slots * GRAPH_BYTES);
                assert_eq!(loaded.graph_arena_bytes, 64 << 20);
                assert_eq!(loaded.headroom_bytes, headroom);
                assert_eq!(loaded.logit_rows, loaded.dims.rows);
                assert_eq!(
                    [
                        loaded.input.bytes(),
                        loaded.logits.bytes(),
                        loaded.samp.bytes()
                    ],
                    expected
                );
                // Decode and mixed captures share these whole slots.
                assert_eq!(loaded.graph_bytes / GRAPH_BYTES, slots);
            }
        }
    }

    /// Each decode forward's (rows, block-table width), as [`Shapes`] records them.
    type Seen = Arc<Mutex<Vec<(u32, i64)>>>;

    /// The tiny model, recording each decode forward's rows and block-table width.
    struct Shapes {
        model: Transformer,
        seen: Seen,
    }

    impl Model for Shapes {
        fn config(&self) -> &ModelConfig {
            self.model.config()
        }

        fn forward(
            &mut self,
            s: &Stream,
            b: &StepBatch,
            kv: &KvPool,
            logits: &Tensor,
        ) -> Result<(), KError> {
            if b.prefill_rows == 0 {
                let width = b.block_table.shape()[1];
                self.seen.lock().unwrap().push((b.decode_rows, width));
            }
            self.model.forward(s, b, kv, logits)
        }
    }

    /// An engine on `device` over the tiny llama (`tag` names its file), loaded with `knobs`, whose
    /// decode forwards [`Shapes`] records.
    fn shapes_engine(
        tag: &str,
        device: i32,
        knobs: Vec<(u16, f64)>,
    ) -> (Engine, crate::EngineHandle, Seen) {
        let path = std::env::temp_dir().join(format!("larch-{tag}-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Llama, 3)).unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (spy, p) = (seen.clone(), path.clone());
        let load = Box::new(move |dev: &_, s: &_, rows| {
            Ok(Box::new(Shapes {
                model: Transformer::load(&p, dev, s, rows)?,
                seen: spy,
            }) as Box<dyn Model>)
        });
        let mut opts = EngineOptions::new(path.clone(), device);
        opts.knob_overrides = knobs;
        let (e, h) = Engine::init(opts, load, None).unwrap();
        std::fs::remove_file(&path).unwrap();
        (e, h, seen)
    }

    /// Moves each request's new tokens onto its row of `out`; the number of them that finished.
    fn drain(hs: &[crate::RequestHandle], out: &mut [Vec<u32>]) -> usize {
        let mut done = 0;
        for (h, o) in hs.iter().zip(out) {
            for ev in h.events.try_iter() {
                match ev {
                    EngineEvent::Token { id, .. } => o.push(id),
                    EngineEvent::Done { .. } => done += 1,
                    EngineEvent::Prefilled { .. } => {}
                    ev => panic!("{ev:?}"),
                }
            }
        }
        done
    }

    /// A refused bucket's steps run eager at the exact rows, without padding rows (at the full
    /// width every step of a graph-capable backend sends), until its quota grows; then it
    /// captures (on the CPU reference, whose capture comes back empty, the attempt runs the padded
    /// shape a replay would). A quota short of one capture's floor (`GRAPH_BYTES`) refuses before
    /// any capture, as an empty one does. `LARCH_TEST_DEVICE` picks the device, as in
    /// tests/graphs.rs.
    #[test]
    fn refused_bucket_runs_exact_rows_until_the_quota_grows() {
        let device = std::env::var("LARCH_TEST_DEVICE").map_or(CPU, |v| v.parse().unwrap());
        for quota in [0, GRAPH_BYTES - 1] {
            let (mut e, h, seen) =
                shapes_engine("refused", device, vec![(knob_id::MAX_BATCH_SEQS, 8.0)]);
            let captures = e.loaded.as_ref().unwrap().info.caps.graph_replay;
            let loaded = e.loaded.as_mut().unwrap();
            (loaded.info.caps.backend, loaded.info.caps.graph_replay) = (Backend::Cuda, true);
            loaded.graph_bytes = quota;
            let max_blocks = loaded.dims.max_blocks as i64;
            e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 1.0;
            let _requests: Vec<_> = (0..3u32)
                .map(|i| h.submit(Request::new(vec![i + 1], 8)).unwrap())
                .collect();
            assert!(e.turn().unwrap()); // the prompts
            for _ in 0..2 {
                assert!(e.turn().unwrap());
                assert_eq!(
                    seen.lock().unwrap().last(),
                    Some(&(3, max_blocks)),
                    "{quota}"
                );
                // A revision that changes nothing a capture needs (an A/B flip) does not retry it.
                e.cfg.rev += 1;
            }
            assert!(matches!(
                e.loaded.as_ref().unwrap().graphs.get(&decode_key(4, 0)),
                Some(Capture::Refused(Refusal { free, .. })) if *free == quota
            ));
            e.loaded.as_mut().unwrap().graph_bytes = 16 * GRAPH_BYTES; // a refit's larger quota
            assert!(e.turn().unwrap());
            assert_eq!(seen.lock().unwrap().last(), Some(&(4, max_blocks)));
            let graphs = &e.loaded.as_ref().unwrap().graphs;
            if captures {
                assert!(matches!(
                    graphs.get(&decode_key(4, 0)),
                    Some(Capture::Graph(..))
                ));
                assert_eq!(e.loaded.as_ref().unwrap().graph_arena_bytes, 0);
            } else {
                assert!(matches!(
                    graphs.get(&decode_key(4, 0)),
                    Some(Capture::Refused(Refusal { free, .. })) if *free == 16 * GRAPH_BYTES
                ));
                assert!(e.turn().unwrap());
                assert_eq!(seen.lock().unwrap().last(), Some(&(3, max_blocks)));
            }
        }
    }

    /// A graph-capable load's decode step that runs eager on a refusal is counted, every one: a
    /// quota short of a capture, then a retry whose capture fails (the CPU stream refuses every
    /// one). A prompt step and an eager step with graphs off count nothing.
    #[test]
    fn refused_decode_steps_are_counted() {
        let (mut e, h, _) = shapes_engine("eager-count", CPU, Vec::new());
        let loaded = e.loaded.as_mut().unwrap();
        (loaded.info.caps.backend, loaded.info.caps.graph_replay) = (Backend::Cuda, true);
        loaded.graph_bytes = 0;
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 1.0;
        let _request = h.submit(Request::new(vec![1, 2, 3], 8)).unwrap();
        assert!(e.turn().unwrap()); // the prompt
        assert_eq!(h.eager_decode_steps(), 0);
        for n in 1..=2 {
            assert!(e.turn().unwrap());
            assert_eq!(h.eager_decode_steps(), n);
        }
        e.loaded.as_mut().unwrap().graph_bytes = 16 * GRAPH_BYTES;
        assert!(e.turn().unwrap());
        assert_eq!(h.eager_decode_steps(), 3);
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 0.0;
        assert!(e.turn().unwrap());
        assert_eq!(h.eager_decode_steps(), 3);
    }

    /// With decode graphs off on a backend that replays them, an eager 1-row decode step still
    /// sends the full-width table a replay sends, so the attention split layout derived from the
    /// width (and with it the bits) is the same in both modes.
    #[test]
    fn eager_decode_sends_the_replay_table_width() {
        let (mut e, h, seen) = shapes_engine("width", CPU, Vec::new());
        let loaded = e.loaded.as_mut().unwrap();
        (loaded.info.caps.backend, loaded.info.caps.graph_replay) = (Backend::Cuda, true);
        let max_blocks = loaded.dims.max_blocks as i64;
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 0.0;
        let _request = h.submit(Request::new(vec![1], 4)).unwrap();
        assert!(e.turn().unwrap()); // the prompt
        assert!(e.turn().unwrap());
        assert_eq!(seen.lock().unwrap().last(), Some(&(1, max_blocks)));
    }

    /// A model with a host-expert layer admits its decode keys. Where the device captures, the
    /// bucket holds two pieces cut at that layer, charged the bytes the capture measured, within
    /// the quota; on the CPU reference the attempt comes back empty and refuses. As above,
    /// `LARCH_TEST_DEVICE` picks the device.
    #[test]
    fn host_expert_decode_keys_capture_charged_what_they_measure() {
        let device = std::env::var("LARCH_TEST_DEVICE").map_or(CPU, |v| v.parse().unwrap());
        let name = format!("larch-host-graphs-{}.gguf", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let p = path.clone();
        let load = Box::new(move |dev: &_, s: &_, rows| {
            let place = larch_model::Placement {
                cpu_expert_layers: 1,
                ..Default::default()
            };
            Ok(Box::new(Transformer::load_placed(&p, dev, s, rows, place)?) as Box<dyn Model>)
        });
        let mut opts = EngineOptions::new(path.clone(), device);
        opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 8.0)];
        let (mut e, h) = Engine::init(opts, load, None).unwrap();
        std::fs::remove_file(&path).unwrap();
        let captures = e.loaded.as_ref().unwrap().info.caps.graph_replay;
        let loaded = e.loaded.as_mut().unwrap();
        (loaded.info.caps.backend, loaded.info.caps.graph_replay) = (Backend::Cuda, true);
        loaded.graph_bytes = 4 * GRAPH_BYTES;
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 1.0;
        let _requests: Vec<_> = (0..2u32)
            .map(|i| h.submit(Request::new(vec![i + 1], 8)).unwrap())
            .collect();
        for _ in 0..3 {
            assert!(e.turn().unwrap());
        }
        match e.loaded.as_ref().unwrap().graphs.get(&decode_key(2, 0)) {
            Some(Capture::Graph(cut, _, bytes)) if captures => {
                assert_eq!(cut.len(), 1);
                assert!((GRAPH_BYTES..=4 * GRAPH_BYTES).contains(bytes), "{bytes}");
            }
            Some(Capture::Refused(_)) if !captures => {}
            _ => panic!("the decode key was not admitted"),
        }
    }

    /// Greedy tokens of 4 requests decoding together in the 4-row bucket (no padding) with
    /// `graphs` on a quota of one capture's floor and no arena reserve, raised to 64 slots after the
    /// fourth decode step: each step's (decode rows, table width) as the model saw it, and whether
    /// the device captures.
    fn over_quota_decode(graphs: bool) -> (Vec<Vec<u32>>, Vec<(u32, i64)>, bool) {
        let device = std::env::var("LARCH_TEST_DEVICE").map_or(CPU, |v| v.parse().unwrap());
        let tag = format!("over-quota-{graphs}");
        let (mut e, h, seen) = shapes_engine(&tag, device, vec![(knob_id::MAX_BATCH_SEQS, 8.0)]);
        let captures = e.loaded.as_ref().unwrap().info.caps.graph_replay;
        let loaded = e.loaded.as_mut().unwrap();
        (loaded.info.caps.backend, loaded.info.caps.graph_replay) = (Backend::Cuda, true);
        // The floor admits the capture; the arena it allocates, reserved for nothing, is charged
        // to it, which puts it over. A headroom past any device's memory makes that a real
        // shortage: free memory never covers it, so the device does not hold the capture.
        (loaded.graph_bytes, loaded.graph_arena_bytes) = (GRAPH_BYTES, 0);
        loaded.headroom_bytes = 1 << 50;
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = f64::from(u8::from(graphs));
        let hs: Vec<_> = (0..4u32)
            .map(|i| h.submit(Request::new(vec![i + 1], 8)).unwrap())
            .collect();
        let mut turns = 0;
        let mut out = vec![Vec::new(); hs.len()];
        let mut open = hs.len();
        while open > 0 {
            if turns == 5 {
                e.loaded.as_mut().unwrap().graph_bytes = 64 * GRAPH_BYTES; // freed quota
            }
            assert!(e.turn().unwrap());
            turns += 1;
            open -= drain(&hs, &mut out);
            if !graphs {
                continue;
            }
            let loaded = e.loaded.as_ref().unwrap();
            let c = loaded.graphs.get(&decode_key(4, 0));
            match (turns, c) {
                (1, None) => {} // the prompts
                // The capture, with its arena, is over the quota: refused with the bytes it took,
                // so it retries once the quota holds them. An attempt that comes back empty is
                // refused with the quota it saw.
                (2..=5, Some(Capture::Refused(r))) if captures => {
                    assert!(r.arena && r.free >= GRAPH_BYTES, "{r:?}")
                }
                (2..=5, Some(Capture::Refused(r))) => assert_eq!(r.free, GRAPH_BYTES),
                (6.., Some(Capture::Graph(_, _, bytes))) if captures => {
                    assert!((GRAPH_BYTES..=64 * GRAPH_BYTES).contains(bytes), "{bytes}")
                }
                (6.., Some(Capture::Refused(r))) if !captures => {
                    assert_eq!(r.free, 64 * GRAPH_BYTES)
                }
                _ => panic!("turn {turns}: unexpected capture state"),
            }
        }
        let seen = std::mem::take(&mut *seen.lock().unwrap());
        (out, seen, captures)
    }

    /// A capture over the quota left is dropped after its step (which its replay served), refused
    /// with its measured size, and its shape runs eager (at the full width, as every step of a
    /// graph-capable backend), sampling what an eager run does, until the quota frees room; then
    /// it captures. On the CPU reference every attempt comes back empty and only the refusals and
    /// their retry show.
    #[test]
    fn an_over_quota_capture_is_dropped_and_retried_once_the_quota_frees() {
        let ((on, shapes, captures), (off, _, _)) =
            (over_quota_decode(true), over_quota_decode(false));
        assert_eq!(on, off);
        let full = i64::from(tiny::CTX.div_ceil(crate::sched::BLOCK_TOKENS as u32));
        // Decode forwards: the capture, eager steps 3-5, the retry's capture, all at full width;
        // a device then replays, the CPU reference runs eager.
        assert!(shapes.iter().all(|s| s.1 == full), "{shapes:?}");
        assert!(shapes.len() >= 5, "{shapes:?}");
        assert_eq!(shapes.len() == 5, captures, "{shapes:?}");
    }

    /// The tiny model, recording each forward's (decode rows, prefill rows).
    struct Spy {
        model: Transformer,
        seen: Arc<Mutex<Vec<(u32, u32)>>>,
        next_decode_rows: u32,
        padded: bool,
        backend: Backend,
    }

    impl Model for Spy {
        fn config(&self) -> &ModelConfig {
            self.model.config()
        }

        fn forward(
            &mut self,
            s: &Stream,
            b: &StepBatch,
            kv: &KvPool,
            logits: &Tensor,
        ) -> Result<(), KError> {
            if b.prefill_rows == 0 {
                // The fixture finishes one of its 32 requests per decode step. Do not infer the
                // real row count from the metadata being checked.
                let real = self.next_decode_rows;
                let graph = self.padded;
                let exact = self.backend == Backend::Cuda && (9..=32).contains(&real);
                let total = if graph && !exact {
                    real.next_power_of_two()
                } else {
                    real
                };
                assert_eq!(b.decode_rows, total);
                assert_eq!(b.slot_ids.shape(), &[i64::from(total)]);
                assert_eq!(b.seq_lens.shape(), &[i64::from(total)]);
                let mut slots = vec![0i32; total as usize];
                let mut lens = vec![0i32; total as usize];
                s.download(b.slot_ids.buffer(), b.slot_ids.offset(), &mut slots)?;
                s.download(b.seq_lens.buffer(), b.seq_lens.offset(), &mut lens)?;
                assert!(slots[..real as usize].iter().all(|&slot| slot >= 0));
                assert!(lens[..real as usize].iter().all(|&len| len > 0));
                let width = if graph {
                    tiny::CTX.div_ceil(crate::sched::BLOCK_TOKENS as u32)
                } else {
                    (*lens[..real as usize].iter().max().unwrap() as u32)
                        .div_ceil(crate::sched::BLOCK_TOKENS as u32)
                };
                assert_eq!(b.block_table.shape(), &[i64::from(total), i64::from(width)]);
                assert!(
                    slots[real as usize..].iter().all(|&slot| slot == -1),
                    "padded slot metadata: {slots:?}"
                );
                assert!(
                    lens[real as usize..].iter().all(|&len| len == 0),
                    "padded length metadata: {lens:?}"
                );
                self.next_decode_rows -= 1;
            }
            self.seen
                .lock()
                .unwrap()
                .push((b.decode_rows, b.prefill_rows));
            self.model.forward(s, b, kv, logits)
        }
    }

    /// Greedy tokens of 32 requests decoding together and finishing one a step (buckets 32 down to
    /// 1), and the rows each forward saw. The CPU reference has no graphs, so `graphs` set directly
    /// drives the padded step, whose capture comes back empty: what a replay runs, run directly.
    fn decode(graphs: bool, backend: Backend) -> (Vec<Vec<u32>>, Vec<(u32, u32)>) {
        let path = std::env::temp_dir().join(format!(
            "larch-pad-{backend:?}-{graphs}-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, tiny::gguf(Arch::Llama, 3)).unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (spy, p) = (seen.clone(), path.clone());
        let load = Box::new(move |dev: &_, s: &_, rows| {
            Ok(Box::new(Spy {
                model: Transformer::load(&p, dev, s, rows)?,
                seen: spy,
                next_decode_rows: 32,
                padded: graphs,
                backend,
            }) as Box<dyn Model>)
        });
        let mut opts = EngineOptions::new(path.clone(), CPU);
        opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 32.0)];
        let (mut e, h) = Engine::init(opts, load, None).unwrap();
        std::fs::remove_file(&path).unwrap();
        // Exercise backend graph selection on the CPU oracle; capture still returns no graph.
        e.loaded.as_mut().unwrap().info.caps.backend = backend;
        e.loaded.as_mut().unwrap().graph_bytes = 64 * GRAPH_BYTES;
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = f64::from(u8::from(graphs));
        let hs: Vec<_> = (0..32u32)
            .map(|i| {
                h.submit(Request::new(vec![i + 1; i as usize % 5 + 1], i + 2))
                    .unwrap()
            })
            .collect();
        let mut out = vec![Vec::new(); hs.len()];
        let mut open = hs.len();
        let mut refused = std::collections::HashSet::new();
        while open > 0 {
            // Forgetting refusals makes each step attempt a capture with the padded shape a replay
            // would run, instead of the exact rows a refused bucket otherwise runs.
            let graphs = &mut e.loaded.as_mut().unwrap().graphs;
            assert!(
                graphs.values().all(|c| matches!(c, Capture::Refused(_))),
                "the CPU reference captured a graph"
            );
            refused.extend(graphs.drain().map(|(k, _)| k));
            assert!(e.turn().unwrap());
            open -= drain(&hs, &mut out);
        }
        refused.extend(e.loaded.as_mut().unwrap().graphs.drain().map(|(k, _)| k));
        assert_eq!(
            refused.len(),
            if !graphs {
                0
            } else if backend == Backend::Cuda {
                28
            } else {
                6
            },
            "unavailable buckets must be remembered without retaining graphs"
        );
        (out, std::mem::take(&mut *seen.lock().unwrap()))
    }

    /// A padded decode step samples what the unpadded one does: padding rows write no KV and attend
    /// to nothing, past 16 rows (gemm) as below.
    #[test]
    fn padded_decode_steps_sample_the_same_tokens() {
        let ((on, padded), (off, plain)) =
            (decode(true, Backend::Cpu), decode(false, Backend::Cpu));
        assert_eq!(on, off);
        let decode_rows =
            |v: &[(u32, u32)]| -> Vec<u32> { v.iter().filter(|r| r.1 == 0).map(|r| r.0).collect() };
        let (padded, plain) = (decode_rows(&padded), decode_rows(&plain));
        assert_eq!(plain, (1..=32).rev().collect::<Vec<_>>());
        assert_eq!(padded.len(), 32);
        assert!(padded.iter().all(|n| n.is_power_of_two()), "{padded:?}");
        assert!(padded.contains(&32), "{padded:?}");
        assert!(plain.iter().any(|n| !n.is_power_of_two()), "{plain:?}");
    }

    #[test]
    fn exact_cuda_graph_buckets_keep_decode_metadata_and_tokens() {
        let ((on, padded), (off, _)) = (decode(true, Backend::Cuda), decode(false, Backend::Cuda));
        assert_eq!(on, off);
        let rows: Vec<_> = padded.iter().filter(|r| r.1 == 0).map(|r| r.0).collect();
        assert_eq!(
            rows,
            [
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                11, 10, 9, 8, 8, 8, 8, 4, 4, 2, 1
            ]
        );
    }

    /// One `forward_state`: (decode rows, prefill rows, state slots, rows per sequence, the
    /// slots' and rows' input offsets).
    type StateStep = (u32, u32, Vec<i32>, Vec<i32>, [u64; 2]);

    /// The tiny hybrid model, recording each step's state slots and rows per sequence.
    struct StateSpy(Transformer, Arc<Mutex<Vec<StateStep>>>);

    impl Model for StateSpy {
        fn config(&self) -> &ModelConfig {
            self.0.config()
        }

        fn forward(
            &mut self,
            s: &Stream,
            b: &StepBatch,
            kv: &KvPool,
            logits: &Tensor,
        ) -> Result<(), KError> {
            self.0.forward(s, b, kv, logits)
        }

        fn state_slots(&self) -> u32 {
            self.0.state_slots()
        }

        fn reset_state(&mut self, s: &Stream, slot: u32) -> Result<(), KError> {
            self.0.reset_state(s, slot)
        }

        fn copy_state(&mut self, s: &Stream, from: u32, to: u32) -> Result<(), KError> {
            self.0.copy_state(s, from, to)
        }

        fn forward_state(
            &mut self,
            s: &Stream,
            b: &StepBatch,
            state: &larch_model::StateBatch,
            kv: &KvPool,
            logits: &Tensor,
            taps: Option<&larch_model::Taps>,
        ) -> Result<(), KError> {
            let n = state.slots.shape()[0] as usize;
            let (mut slots, mut lens) = (vec![0i32; n], vec![0i32; n]);
            s.download(state.slots.buffer(), state.slots.offset(), &mut slots)?;
            s.download(state.q_lens.buffer(), state.q_lens.offset(), &mut lens)?;
            let at = [state.slots.offset(), state.q_lens.offset()];
            let step = (b.decode_rows, b.prefill_rows, slots, lens, at);
            self.1.lock().unwrap().push(step);
            self.0.forward_state(s, b, state, kv, logits, taps)
        }
    }

    /// `Loaded::load` of the tiny hybrid MoE through its real loader, on a fake discrete CUDA card
    /// with `before` free until the load places its layers and `after(host layers, state bytes)`
    /// from then on. (The loaded engine, its config, host layers, and the load's ledger.)
    fn fake_cuda_load(
        before: u64,
        after: impl Fn(u32, u64) -> u64 + 'static,
        forced: Option<u32>,
    ) -> (Loaded, EffectiveConfig, u32, crate::Ledger) {
        fake_cuda_seats(before, after, forced, None, None)
    }

    /// [`fake_cuda_load`] with `max_batch_seqs` set by hand to `asked` and the automatic drafter
    /// `auto`, if some.
    fn fake_cuda_seats(
        before: u64,
        after: impl Fn(u32, u64) -> u64 + 'static,
        forced: Option<u32>,
        asked: Option<u32>,
        auto: Option<PathBuf>,
    ) -> (Loaded, EffectiveConfig, u32, crate::Ledger) {
        let path = std::env::temp_dir().join(format!(
            "larch-fake-cuda-{before}-{forced:?}-{:?}-{}.gguf",
            std::thread::current().id(),
            std::process::id()
        ));
        std::fs::write(&path, tiny::gguf(Arch::Qwen35Moe, 3)).unwrap();
        let mut opts = EngineOptions::new(path.clone(), CPU);
        opts.max_context = tiny::CTX;
        (opts.cpu_expert_layers, opts.drafter_auto) = (forced, auto);
        let (load, weights) = crate::model_loader(&opts).unwrap();
        let w = weights.as_ref().unwrap();
        let (host, ledger) = (w.host_layers.clone(), w.ledger.clone());
        let (h, l) = (host.clone(), ledger.clone());
        crate::FAKE_PROPS.set(Some(std::rc::Rc::new(move |mut p| {
            p.backend = larch_kernels::Backend::Cuda;
            p.flags = larch_kernels::DeviceProps::CAP_GRAPHS;
            p.vram_total = 17_066_033_152;
            let l = *l.lock().unwrap();
            p.vram_free = if l.weights_on_device != 0 {
                after(h.load(Ordering::Relaxed), l.hybrid)
            } else {
                before
            };
            p
        })));
        let dev = larch_kernels::Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let mut cfg = larch_records::EffectiveConfig::defaults(larch_records::REGISTRY);
        cfg.values[knob_id::DECODE_GRAPHS as usize] = 1.0;
        if let Some(n) = asked {
            cfg.values[knob_id::MAX_BATCH_SEQS as usize] = f64::from(n);
        }
        let mut partial = Partial {
            startup: true,
            graphs_fit_gated: true,
            seats_asked: asked.is_some(),
            ..Partial::default()
        };
        let loaded = Loaded::load(&mut opts, &mut cfg, load, weights, &s, 512, &mut partial);
        crate::FAKE_PROPS.set(None);
        std::fs::remove_file(&path).unwrap();
        let (n, l) = (host.load(Ordering::Relaxed), *ledger.lock().unwrap());
        (loaded.expect("the load fits"), cfg, n, l)
    }

    /// (fake props at `before`, the tiny hybrid's config, the weight bytes its plan places on the
    /// device, the ledger of one roomy load (8 state slots), and one slot's state bytes). Those
    /// bytes leave out what a discrete CUDA load keeps off the device: the token embedding (host
    /// memory) and the f32 originals of the routers, shared-expert gates and float alpha (read as
    /// f16 side copies).
    fn fake_cuda_terms() -> (
        larch_kernels::DeviceProps,
        ModelConfig,
        u64,
        crate::Ledger,
        u64,
    ) {
        let (loaded, _, n, l) = fake_cuda_load(1 << 30, |_, _| 512 << 20, None);
        assert_eq!((n, loaded.model.state_slots()), (0, 8));
        let g = larch_model::gguf::Gguf::parse(&tiny::gguf(Arch::Qwen35Moe, 3)).unwrap();
        let c = ModelConfig::from_gguf(&g).unwrap();
        let mut props = larch_kernels::Device::open(CPU).unwrap().props().unwrap();
        props.backend = larch_kernels::Backend::Cuda;
        props.flags = larch_kernels::DeviceProps::CAP_GRAPHS;
        let slot = Transformer::hybrid_bytes(&g, &c, 1, 1).unwrap()
            - Transformer::hybrid_bytes(&g, &c, 1, 0).unwrap();
        let off: u64 = (g.tensors.iter())
            .filter(|t| {
                t.name == "token_embd.weight"
                    || t.name.ends_with(".ffn_gate_inp.weight")
                    || t.name.ends_with(".ffn_gate_inp_shexp.weight")
                    || t.name.ends_with(".ssm_alpha.weight") // tiny's alpha is f32, its beta not
            })
            .map(|t| t.bytes)
            .sum();
        let bytes = crate::Weights::of(&g).bytes - off;
        assert_eq!(
            l.weights_on_device, bytes,
            "the plan places what the device holds"
        );
        (props, c, bytes, l, slot)
    }

    /// The plan's reserve beside the weights at `slots` state slots, from the roomy load's ledger.
    fn reserve_at(l: &crate::Ledger, slot: u64, slots: u32) -> u64 {
        l.activation
            + l.routers
            + l.scratch
            + l.graph_arena
            + (l.hybrid - u64::from(8 - slots) * slot)
            + graph_reserve(larch_kernels::Backend::Cuda, slots, 0)
    }

    /// Through the real loader on a fake CUDA card, with a pool of one layer's experts (the tiny
    /// fixture's own is past all of them): a card a layer short of the weights and one slot moves
    /// one layer without the pool and two with it; resident stays resident. Tiny's two expert
    /// layers make the pool's second layer every layer, so the slots' share of the pool's room and
    /// the expert cache's no-pool plan are not told apart here (a fixture of three or more would).
    #[test]
    fn the_loader_pays_a_layer_for_the_stream_pool() {
        let (props, c, bytes, l, slot) = fake_cuda_terms();
        let need = |beside, seats| {
            crate::fit::serving_need(&props, &c, tiny::CTX, 0.85, beside, false, seats)
        };
        let one = need(reserve_at(&l, slot, 1), 1) + crate::FREE_SWING;
        let g = larch_model::gguf::Gguf::parse(&tiny::gguf(Arch::Qwen35Moe, 3)).unwrap();
        let layer: u64 = (g.tensors.iter())
            .filter(|t| t.name.starts_with("blk.0.") && t.name.ends_with("_exps.weight"))
            .map(|t| t.bytes)
            .sum();
        let plan = |before, pool, asked| {
            crate::FAKE_POOL.set(pool);
            let (loaded, _, n, _) = fake_cuda_seats(before, |_, _| 1 << 30, None, asked, None);
            crate::FAKE_POOL.set(None);
            (n, loaded.model.state_slots())
        };
        let short = bytes - layer + one;
        assert_eq!(plan(short, Some(0), None), (1, 1), "no pool");
        assert_eq!(plan(short, Some(layer), None), (2, 1), "the pool's layer");
        assert_eq!(plan(bytes + one, Some(layer), None).0, 0, "resident");
    }

    /// Through the real loader on a fake CUDA card: a host layer moved without the pool that
    /// already leaves its room keeps it, so the check after the load wants the pool too (Windows
    /// 5070 Ti, 24 Nov: a documents reload at 10 host layers cleared the check by 24 MB and left
    /// the 149 MB pool out, so a 121-page read ran at 90 % CPU). A reading short of it moves the
    /// last layer, where the pool is optional.
    #[test]
    fn the_check_after_a_load_keeps_the_pools_room_its_layers_leave() {
        let (props, c, bytes, l, slot) = fake_cuda_terms();
        let need = |beside, seats| {
            crate::fit::serving_need(&props, &c, tiny::CTX, 0.85, beside, false, seats)
        };
        let one = need(reserve_at(&l, slot, 1), 1) + crate::FREE_SWING;
        let g = larch_model::gguf::Gguf::parse(&tiny::gguf(Arch::Qwen35Moe, 3)).unwrap();
        let layer: u64 = (g.tensors.iter())
            .filter(|t| t.name.starts_with("blk.0.") && t.name.ends_with("_exps.weight"))
            .map(|t| t.bytes)
            .sum();
        let pool = layer / 2;
        // One layer with or without the pool: none moves for it.
        let before = bytes - layer + one + pool;
        let graphs = graph_reserve(larch_kernels::Backend::Cuda, 1, 0);
        let after = need(l.scratch + graphs + l.graph_arena, 1) + crate::FREE_SWING;
        let load = |free: u64| {
            crate::FAKE_POOL.set(Some(pool));
            let (loaded, _, n, _) = fake_cuda_load(before, move |_, _| free, None);
            crate::FAKE_POOL.set(None);
            (n, loaded.model.state_slots())
        };
        assert_eq!(load(after + pool), (1, 1), "room for the pool");
        assert_eq!(load(after).0, 2, "short of the pool: the last layer");
    }

    /// An automatic drafter on a card that holds the plain load at no host layer and one state
    /// slot would cost a host layer: the load serves without it, as the plain load does. A roomy
    /// card keeps it.
    #[test]
    fn an_automatic_drafter_loads_only_where_it_costs_no_chat() {
        let (props, c, bytes, l, slot) = fake_cuda_terms();
        let need = |beside, seats| {
            crate::fit::serving_need(&props, &c, tiny::CTX, 0.85, beside, false, seats)
        };
        let short = bytes + need(reserve_at(&l, slot, 1), 1) + crate::FREE_SWING;
        let dflash = std::env::temp_dir().join(format!("larch-auto-{}.gguf", std::process::id()));
        std::fs::write(&dflash, tiny::dflash_gguf(7)).unwrap();
        let roomy = |_, _| 1 << 30;
        let load = |before, auto| fake_cuda_seats(before, roomy, None, None, auto);
        let (plain, _, n, _) = load(short, None);
        let (auto, _, n_auto, l) = load(short, Some(dflash.clone()));
        let (big, _, _, l_big) = load(1 << 30, Some(dflash.clone()));
        std::fs::remove_file(&dflash).unwrap();
        assert!(auto.draft.is_none(), "dropped");
        assert_eq!(
            (n_auto, auto.model.state_slots()),
            (n, plain.model.state_slots())
        );
        let (with, without, kept) = l.auto.expect("planned");
        assert!(
            !kept && with.host_layers > without.host_layers,
            "{with:?} {without:?}"
        );
        assert!(
            big.draft.is_some() && l_big.auto.is_some_and(|a| a.2),
            "roomy"
        );
    }

    /// A hand-set seat count re-plans only a planned hybrid with host experts, and only when the
    /// plan differs: more slots than it holds, or fewer than it paid for.
    #[test]
    fn only_a_differing_plan_replans() {
        use super::replan_due;
        assert!(replan_due(4, 2, true, Some(1), true));
        assert!(replan_due(1, 4, true, Some(4), true));
        assert!(
            !replan_due(2, 2, true, Some(1), true),
            "the plan for one chat already holds 2"
        );
        assert!(
            !replan_due(4, 2, false, Some(1), true),
            "no host experts: the slots grow live"
        );
        assert!(!replan_due(4, 0, true, Some(1), true), "no recurrent state");
        assert!(
            !replan_due(4, 2, true, None, true),
            "a forced or custom load"
        );
        assert!(
            replan_due(4, 4, true, Some(1), true),
            "many chats take the expert cache"
        );
        assert!(!replan_due(8, 8, true, Some(4), true), "already many chats");
        assert!(
            !replan_due(4, 8, true, Some(1), false),
            "off CUDA or with a forced cache, crossing 4 seats changes no plan"
        );
    }

    /// Demand switches to many chats only after two requests waited a second on full seats, and
    /// back only after two minutes of at most one chat; relief in between restarts the clock.
    #[test]
    fn demand_switches_with_hysteresis() {
        let s = 1_000_000_000;
        let mut d = super::Demand::default();
        assert_eq!(d.seats(0, 2, 2, 2), None);
        assert_eq!(d.seats(s - 1, 2, 2, 2), None, "under a second");
        assert_eq!(d.seats(s, 2, 1, 2), None, "one waiting is not demand");
        assert_eq!(d.seats(s + 1, 1, 3, 2), None, "a free seat is not demand");
        assert_eq!(d.seats(2 * s, 2, 2, 2), None, "relief restarted the clock");
        assert_eq!(d.seats(3 * s, 2, 2, 2), Some(super::MANY_SEATS));
        assert_eq!(
            d.seats(3 * s + 1, 2, 2, 2),
            Some(super::MANY_SEATS),
            "unflipped until a switch loads"
        );
        d.switched(super::MANY_SEATS);
        assert_eq!(d.seats(4 * s, 1, 0, 4), None, "many chats hold");
        assert_eq!(
            d.seats(100 * s, 2, 0, 4),
            None,
            "two chats restart the clock"
        );
        assert_eq!(d.seats(101 * s, 1, 0, 4), None);
        assert_eq!(d.seats(220 * s, 0, 0, 4), None, "under two minutes");
        assert_eq!(d.seats(221 * s, 0, 0, 4), Some(1));
        d.switched(1);
        assert_eq!(
            d.seats(222 * s, 2, 2, 2),
            None,
            "fast again: the up clock starts over"
        );
        // A cancelled switch keeps demand quiet for two minutes.
        d.cancelled(222 * s);
        assert_eq!(d.seats(224 * s, 2, 2, 2), None);
        assert_eq!(d.seats(341 * s, 2, 2, 2), None, "still quiet");
        assert_eq!(d.seats(342 * s, 2, 2, 2), None, "the up clock starts");
        assert_eq!(d.seats(343 * s, 2, 2, 2), Some(super::MANY_SEATS));
    }

    /// A load that kept fewer than MANY_SEATS slots (3: the check after it was short of the 4th) still
    /// switches when as many chats as MANY_SEATS want seats, though only one of them waits: 4 chats on
    /// 3 slots never put two in the queue, and ran on 3 slots without graphs (5070 Ti, 35B, 25 Nov:
    /// 113 tok/s, against 174 on 4 slots).
    #[test]
    fn many_chats_on_fewer_slots_switch_with_one_waiting() {
        let s = 1_000_000_000;
        let mut d = super::Demand::default();
        assert_eq!(d.seats(0, 3, 1, 3), None);
        assert_eq!(d.seats(s, 3, 1, 3), Some(super::MANY_SEATS));
        let mut d = super::Demand::default();
        assert_eq!(d.seats(0, 2, 1, 2), None);
        assert_eq!(
            d.seats(s, 2, 1, 2),
            None,
            "3 chats on 2 slots: one waiting is not demand"
        );
        let mut d = super::Demand::default();
        assert_eq!(d.seats(0, 4, 1, 4), None);
        assert_eq!(
            d.seats(s, 4, 1, 4),
            None,
            "4 slots: the fast plan already seats MANY_SEATS"
        );
    }

    /// A hand-set `max_batch_seqs` plans its state slots and moves the host layers they need: on a
    /// card where the plan for one chat keeps fewer, the asked load holds all 4 slots.
    #[test]
    fn asked_seats_take_host_layers() {
        let (props, c, bytes, l, slot) = fake_cuda_terms();
        let need = |beside, seats| {
            crate::fit::serving_need(&props, &c, tiny::CTX, 0.85, beside, false, seats)
        };
        let before = bytes - 1 + need(reserve_at(&l, slot, 1), 1) + crate::FREE_SWING;
        let roomy = |_, _| 1 << 30;
        let (fast, _, n_fast, _) = fake_cuda_seats(before, roomy, None, None, None);
        let (asked, _, n_asked, _) = fake_cuda_seats(before, roomy, None, Some(4), None);
        assert!(fast.model.state_slots() < 4, "{}", fast.model.state_slots());
        assert_eq!(asked.model.state_slots(), 4);
        assert!(n_asked > n_fast, "{n_asked} {n_fast}");
    }

    /// Through the real loader and `Loaded::load` on a fake CUDA card: a plan that moved a layer
    /// marks the load paid and keeps graphs at a pool share under the gate, its quota sized for
    /// the one state slot its room leaves; the same share on a forced count refuses them.
    #[test]
    fn a_planned_host_expert_load_is_paid_through_the_loader() {
        let (props, c, bytes, l, slot) = fake_cuda_terms();
        let one = graph_reserve(larch_kernels::Backend::Cuda, 1, 0);
        let need = |beside, seats| {
            crate::fit::serving_need(&props, &c, tiny::CTX, 0.85, beside, false, seats)
        };
        let before = bytes - 1 + need(reserve_at(&l, slot, 1), 1) + crate::FREE_SWING;
        let after = need(l.scratch + one + l.graph_arena, 1) + crate::FREE_SWING;
        let (loaded, cfg, n, l) = fake_cuda_load(before, move |_, _| after, None);
        assert_eq!((n, loaded.model.state_slots()), (1, 1));
        assert!(l.graphs_paid);
        assert_ne!(cfg.values[knob_id::DECODE_GRAPHS as usize], 0.0, "paid: on");
        assert_eq!(loaded.graph_bytes, one);
        let (loaded, cfg, n, l) = fake_cuda_load(before, move |_, _| after, Some(1));
        assert_eq!(n, 1);
        assert!(!l.graphs_paid);
        assert_eq!(
            cfg.values[knob_id::DECODE_GRAPHS as usize],
            0.0,
            "forced: gated"
        );
        assert_eq!(loaded.graph_bytes, 0);
    }

    /// The check after a default whole-layer load allows CUDA's free swing: a load that left the
    /// fit's need and no more, at one state slot, moves a layer, so a low reading at the fit cannot
    /// drop the graphs.
    #[test]
    fn the_whole_layer_check_allows_the_free_swing() {
        let (props, c, bytes, l, slot) = fake_cuda_terms();
        let need = |beside, seats| {
            crate::fit::serving_need(&props, &c, tiny::CTX, 0.85, beside, false, seats)
        };
        let before = bytes - 1 + need(reserve_at(&l, slot, 1), 1) + crate::FREE_SWING;
        let one = graph_reserve(larch_kernels::Backend::Cuda, 1, 0);
        let exact = need(l.scratch + one + l.graph_arena, 1);
        let (_, _, n, _) = fake_cuda_load(
            before,
            move |n, _| exact + u64::from(n - 1) * (1 << 30),
            None,
        );
        assert_eq!(n, 2);
    }

    /// A hybrid load plans the fewest host layers at one state slot, then gives the slots what
    /// those layers leave, up to the cap: room for 4 slots' reserve gets 4 with no layer moved (a
    /// resident load reserves no streaming pool), a byte less gets 3.
    #[test]
    fn a_hybrid_load_gives_the_state_slots_its_room_allows() {
        let (props, c, bytes, l, slot) = fake_cuda_terms();
        let need = |beside, seats| {
            crate::fit::serving_need(&props, &c, tiny::CTX, 0.85, beside, false, seats)
        };
        let before = bytes + need(reserve_at(&l, slot, 4), 4) + crate::FREE_SWING;
        for (before, slots) in [(before, 4), (before - 1, 3)] {
            let (loaded, _, n, got) = fake_cuda_load(before, |_, _| 1 << 30, None);
            assert_eq!((n, loaded.model.state_slots()), (0, slots));
            assert_eq!(got.hybrid, l.hybrid - u64::from(8 - slots) * slot);
            let quota = graph_reserve(larch_kernels::Backend::Cuda, slots, 0);
            assert_eq!(loaded.graph_bytes, quota);
        }
    }

    /// A load the check finds short gives up state slots before it moves a layer: the most slots
    /// whose freed state covers the need (2 of 8 here), measured again.
    #[test]
    fn a_short_hybrid_load_gives_up_state_slots_before_a_layer() {
        let (props, c, _, l, slot) = fake_cuda_terms();
        let need = |beside, seats| {
            crate::fit::serving_need(&props, &c, tiny::CTX, 0.85, beside, false, seats)
        };
        let two = graph_reserve(larch_kernels::Backend::Cuda, 2, 0);
        let state = |slots: u64| l.hybrid - (8 - slots) * slot;
        let room = need(l.scratch + two + l.graph_arena, 2) + crate::FREE_SWING + state(2);
        let (loaded, _, n, _) = fake_cuda_load(1 << 30, move |_, held| room - held, None);
        assert_eq!((n, loaded.model.state_slots()), (0, 2));
    }

    /// A hybrid's real loader gives `max_batch_seqs` 16 only 8 state slots on any device; the
    /// requests past the slots wait for a seat and all finish, none refused.
    #[test]
    fn a_hybrid_serves_sixteen_seats_on_eight_state_slots() {
        let path = std::env::temp_dir().join(format!("larch-eight-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen35, 3)).unwrap();
        let mut opts = EngineOptions::new(path.clone(), CPU);
        opts.max_context = 512;
        opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 16.0)];
        let (load, weights) = crate::model_loader(&opts).unwrap();
        let (mut e, h) = Engine::init(opts, load, weights).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(e.loaded.as_ref().unwrap().model.state_slots(), 8);
        assert_eq!(e.limits().max_batch_seqs, 8);
        let n = 12u32;
        let hs: Vec<_> = (0..n)
            .map(|i| h.submit(Request::new(vec![i + 1; 3], 4)).unwrap())
            .collect();
        let mut out = vec![Vec::new(); hs.len()];
        let (mut open, mut queued) = (hs.len(), false);
        while open > 0 {
            assert!(e.turn().unwrap());
            assert!(e.sched.n_running() <= 8);
            queued |= e.sched.n_running() == 8 && e.sched.n_waiting() > 0;
            open -= drain(&hs, &mut out);
        }
        assert!(queued, "requests past the slots waited");
        assert!(out.iter().all(|o| o.len() == 4), "{out:?}");
    }

    /// Greedy tokens of 6 hybrid requests on `slots` state slots at `max_batch_seqs` `seats`, and
    /// the slots the model holds once they finished.
    fn hybrid_on_slots(slots: u32, seats: u32) -> (Vec<Vec<u32>>, u32) {
        let path = std::env::temp_dir().join(format!(
            "larch-slots-{slots}-{seats}-{:?}-{}.gguf",
            std::thread::current().id(),
            std::process::id()
        ));
        std::fs::write(&path, tiny::gguf(Arch::Qwen35, 3)).unwrap();
        let p = path.clone();
        let load = Box::new(move |dev: &_, s: &_, rows| {
            let mut m = Transformer::load(&p, dev, s, rows)?;
            m.alloc_state(s, slots)?;
            Ok(Box::new(m) as Box<dyn Model>)
        });
        let mut opts = EngineOptions::new(path.clone(), CPU);
        opts.max_context = 512;
        opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, f64::from(seats))];
        let (mut e, h) = Engine::init(opts, load, None).unwrap();
        std::fs::remove_file(&path).unwrap();
        let hs: Vec<_> = (0..6u32)
            .map(|i| {
                h.submit(Request::new(vec![i + 1; i as usize + 2], 6))
                    .unwrap()
            })
            .collect();
        let mut out = vec![Vec::new(); hs.len()];
        let mut open = hs.len();
        while open > 0 {
            assert!(e.turn().unwrap());
            open -= drain(&hs, &mut out);
        }
        (out, e.loaded.as_ref().unwrap().model.state_slots())
    }

    /// What a hybrid answers does not depend on its state slots: one at a time, 3 at once, 8 at
    /// once, or 2 grown live to 8 while the first two decode, the tokens are the same.
    #[test]
    fn results_do_not_depend_on_the_state_slots() {
        let (one, _) = hybrid_on_slots(1, 1);
        for (slots, seats, held) in [(3, 3, 3), (8, 8, 8), (2, 8, 8)] {
            let got = hybrid_on_slots(slots, seats);
            assert_eq!(got, (one.clone(), held), "{slots} slots, {seats} seats");
        }
    }

    /// Greedy tokens of 5 hybrid requests decoding together on 8 state slots and finishing one a
    /// step (decode rows 5 to 1), and each step's state batch. As [`decode`], `graphs` runs the
    /// padded shape a replay runs. The longest request has a repeat penalty, whose window rows
    /// change with the real rows of a bucket.
    fn hybrid_decode(graphs: bool) -> (Vec<Vec<u32>>, Vec<StateStep>) {
        let path = std::env::temp_dir().join(format!(
            "larch-pad-hybrid-{graphs}-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, tiny::gguf(Arch::Qwen35, 3)).unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (spy, p) = (seen.clone(), path.clone());
        let load = Box::new(move |dev: &_, s: &_, rows| {
            let mut m = Transformer::load(&p, dev, s, rows)?;
            m.alloc_state(s, 8)?;
            Ok(Box::new(StateSpy(m, spy)) as Box<dyn Model>)
        });
        let mut opts = EngineOptions::new(path.clone(), CPU);
        opts.max_context = 512;
        opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 8.0)];
        let (mut e, h) = Engine::init(opts, load, None).unwrap();
        std::fs::remove_file(&path).unwrap();
        e.loaded.as_mut().unwrap().graph_bytes = 64 * GRAPH_BYTES;
        e.cfg.values[knob_id::DECODE_GRAPHS as usize] = f64::from(u8::from(graphs));
        let hs: Vec<_> = (0..5u32)
            .map(|i| {
                let mut r = Request::new(vec![i + 1; i as usize + 1], i + 2);
                if i == 4 {
                    r.sampling.repeat_penalty = 1.1;
                }
                h.submit(r).unwrap()
            })
            .collect();
        let mut out = vec![Vec::new(); hs.len()];
        let mut open = hs.len();
        while open > 0 {
            // Forgetting refusals makes every step pad, as in `decode`.
            e.loaded.as_mut().unwrap().graphs.clear();
            assert!(e.turn().unwrap());
            open -= drain(&hs, &mut out);
        }
        (out, std::mem::take(&mut *seen.lock().unwrap()))
    }

    /// A padded hybrid decode step samples what the exact one does, and its padding rows run one
    /// row each on distinct state slots no sequence holds, so no live state is read or written. A
    /// bucket's steps read the state batch at the offsets its capture froze.
    #[test]
    fn padded_hybrid_decode_keeps_held_states() {
        let ((on, padded), (off, exact)) = (hybrid_decode(true), hybrid_decode(false));
        assert_eq!(on, off);
        assert_eq!(padded.len(), exact.len());
        let (mut buckets, mut frozen) = (Vec::new(), std::collections::HashMap::new());
        for ((rows, prefill, slots, lens, at), (nd, _, real, ..)) in padded.iter().zip(&exact) {
            let (rows, nd) = (*rows as usize, *nd as usize);
            if *prefill == 0 {
                assert_eq!(rows, nd.next_power_of_two());
                buckets.push((nd, rows));
                assert_eq!(frozen.entry(rows).or_insert(*at), at, "bucket {rows}");
            }
            assert_eq!(slots[..nd], real[..nd]);
            let pad = &slots[nd..rows];
            let held: Vec<_> = slots[..nd].iter().chain(&slots[rows..]).collect();
            assert!(pad.iter().all(|s| !held.contains(&s)), "{slots:?}");
            let distinct: std::collections::HashSet<_> = pad.iter().collect();
            assert_eq!(distinct.len(), pad.len(), "{slots:?}");
            assert!(lens[nd..rows].iter().all(|&l| l == 1), "{lens:?}");
        }
        assert_eq!(buckets, [(5, 8), (4, 4), (3, 4), (2, 2), (1, 1)]);
    }

    /// Speculative decoding (guess.rs): checked guesses never change greedy output, and sampled
    /// output keeps the model's distribution.
    mod guesses {
        use std::collections::{HashMap, HashSet};
        use std::sync::atomic::{AtomicU64, Ordering};

        use super::*;
        use crate::engine::{MANY_SEATS, check_key, evict_lru};
        use crate::guess::{Costs, Drafter, PromptLookup};
        use crate::sched::Namespace;
        use crate::suffix::SuffixDrafter;
        use crate::{EngineHandle, GuessStats, RequestHandle, SamplingParams, Usage};
        use larch_records::Cause;

        fn mix(mut z: u64) -> u64 {
            z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        /// Proposes the reference continuation of the sequence `tokens` extends, with one token
        /// wrong at a pseudo-random place (or none): every kept length, and misses, happen.
        struct Oracle(Vec<Vec<u32>>, u64);

        impl Drafter for Oracle {
            fn propose(
                &mut self,
                _: u64,
                _: Namespace,
                tokens: &[u32],
                max: usize,
                out: &mut Vec<u32>,
            ) {
                let mut refs = self.0.iter();
                let Some(r) = refs.find(|r| r.len() > tokens.len() && r.starts_with(tokens)) else {
                    return;
                };
                self.1 = mix(self.1);
                let wrong = (self.1 % (max as u64 + 2)) as usize;
                let next = r[tokens.len()..].iter().take(max).enumerate();
                out.extend(next.map(|(i, &t)| if i == wrong { (t + 1) % tiny::VOCAB } else { t }));
            }
        }

        /// Proposes the reference continuation: every guess kept.
        struct Right(Vec<Vec<u32>>);

        impl Drafter for Right {
            fn propose(
                &mut self,
                _: u64,
                _: Namespace,
                tokens: &[u32],
                max: usize,
                out: &mut Vec<u32>,
            ) {
                let mut refs = self.0.iter();
                if let Some(r) = refs.find(|r| r.len() > tokens.len() && r.starts_with(tokens)) {
                    out.extend(r[tokens.len()..].iter().take(max));
                }
            }
        }

        /// Random guesses: nearly all misses.
        struct Random(u64);

        impl Drafter for Random {
            fn propose(&mut self, _: u64, _: Namespace, _: &[u32], max: usize, out: &mut Vec<u32>) {
                for _ in 0..max {
                    self.0 = mix(self.0);
                    out.push((self.0 % u64::from(tiny::VOCAB)) as u32);
                }
            }
        }

        /// Always guesses `.0`.
        struct Constant(u32);

        impl Drafter for Constant {
            fn propose(&mut self, _: u64, _: Namespace, _: &[u32], _: usize, out: &mut Vec<u32>) {
                out.push(self.0);
            }
        }

        static FILE: AtomicU64 = AtomicU64::new(0);

        /// A model on a backend without the check kernels: it begins checks but refuses every
        /// check step before any op, as the Transformer's probe does.
        struct Refuses(Transformer);

        impl Model for Refuses {
            fn config(&self) -> &ModelConfig {
                self.0.config()
            }

            fn forward(
                &mut self,
                s: &Stream,
                b: &StepBatch,
                kv: &KvPool,
                logits: &Tensor,
            ) -> Result<(), KError> {
                if b.check {
                    return Err(KError::Unsupported);
                }
                self.0.forward(s, b, kv, logits)
            }

            fn forward_taps(
                &mut self,
                s: &Stream,
                b: &StepBatch,
                kv: &KvPool,
                logits: &Tensor,
                taps: Option<&larch_model::Taps>,
            ) -> Result<(), KError> {
                if b.check {
                    return Err(KError::Unsupported);
                }
                self.0.forward_taps(s, b, kv, logits, taps)
            }

            fn embed_and_head(&self) -> Option<(Tensor<'_>, Tensor<'_>)> {
                self.0.embed_and_head()
            }

            fn state_slots(&self) -> u32 {
                self.0.state_slots()
            }

            fn alloc_checks(&mut self, s: &Stream, spares: u32) -> Result<(), LoadError> {
                self.0.alloc_checks(s, spares)
            }

            fn begin_check(&mut self, s: &Stream, slot: u32, rows: u32) -> Result<(), KError> {
                self.0.begin_check(s, slot, rows)
            }

            fn end_check(&mut self, s: &Stream, slot: u32, kept: u32) -> Result<(), KError> {
                self.0.end_check(s, slot, kept)
            }

            fn reset_state(&mut self, s: &Stream, slot: u32) -> Result<(), KError> {
                self.0.reset_state(s, slot)
            }

            fn copy_state(&mut self, s: &Stream, from: u32, to: u32) -> Result<(), KError> {
                self.0.copy_state(s, from, to)
            }

            fn forward_state(
                &mut self,
                s: &Stream,
                b: &StepBatch,
                state: &larch_model::StateBatch,
                kv: &KvPool,
                logits: &Tensor,
                taps: Option<&larch_model::Taps>,
            ) -> Result<(), KError> {
                if b.check {
                    return Err(KError::Unsupported);
                }
                self.0.forward_state(s, b, state, kv, logits, taps)
            }
        }

        /// The tiny model on the CPU, guessing up to `guesses` with `drafter`; a recurrent one
        /// with state for 8 sequences, refusing check steps if `refuse` ([`Refuses`]).
        fn engine(
            arch: Arch,
            guesses: u32,
            drafter: Box<dyn Drafter + Send>,
            prefix_blocks: usize,
            refuse: bool,
        ) -> (Engine, EngineHandle) {
            try_engine(arch, guesses, drafter, prefix_blocks, refuse, None).unwrap()
        }

        /// A drafter of the engine tests, guessing up to its cap (0: its own).
        #[derive(Clone, Copy)]
        enum Drafted {
            /// A tiny DFlash drafter.
            Dflash(u32),
            /// A tiny DFlash2 drafter ([`tiny::dflash2_gguf`]).
            Dflash2(u32),
            /// The model's own MTP block, the model built with one ([`tiny::gguf_mtp`]).
            Mtp(u32),
        }

        /// [`engine`] with a drafter guessing, if some.
        fn engine_with(
            arch: Arch,
            guesses: u32,
            drafter: Box<dyn Drafter + Send>,
            prefix_blocks: usize,
            refuse: bool,
            draft: Option<Drafted>,
        ) -> (Engine, EngineHandle) {
            try_engine(arch, guesses, drafter, prefix_blocks, refuse, draft).unwrap()
        }

        /// [`engine_with`], or the load's refusal.
        fn try_engine(
            arch: Arch,
            guesses: u32,
            drafter: Box<dyn Drafter + Send>,
            prefix_blocks: usize,
            refuse: bool,
            draft: Option<Drafted>,
        ) -> Result<(Engine, EngineHandle), LoadError> {
            engine_on(CPU, arch, guesses, drafter, prefix_blocks, refuse, draft)
        }

        /// [`try_engine`] on `device`.
        fn engine_on(
            device: i32,
            arch: Arch,
            guesses: u32,
            drafter: Box<dyn Drafter + Send>,
            prefix_blocks: usize,
            refuse: bool,
            draft: Option<Drafted>,
        ) -> Result<(Engine, EngineHandle), LoadError> {
            let n = FILE.fetch_add(1, Ordering::Relaxed);
            let name = format!("larch-guess-{}-{n}.gguf", std::process::id());
            let path = std::env::temp_dir().join(name);
            let mtp = matches!(draft, Some(Drafted::Mtp(_)));
            let file = match mtp && arch.hybrid() {
                true => tiny::gguf_mtp(arch, 3),
                false => tiny::gguf(arch, 3),
            };
            std::fs::write(&path, file).unwrap();
            let p = path.clone();
            let load = Box::new(move |dev: &_, s: &_, rows| {
                let mut m = match mtp {
                    true => Transformer::load_mtp(&p, dev, s, rows, Default::default())?,
                    false => Transformer::load(&p, dev, s, rows)?,
                };
                if arch.hybrid() {
                    m.alloc_state(s, 8)?;
                }
                Ok(match refuse {
                    true => Box::new(Refuses(m)) as Box<dyn Model>,
                    false => Box::new(m),
                })
            });
            let mut opts = EngineOptions::new(path.clone(), device);
            opts.max_context = 512;
            opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 8.0)];
            opts.guesses = guesses;
            opts.prefix_cache_blocks = prefix_blocks;
            opts.prefix_cache_custom_compatible = true;
            let dflash =
                std::env::temp_dir().join(format!("larch-dflash-{}-{n}.gguf", std::process::id()));
            match draft {
                Some(Drafted::Dflash(cap)) => {
                    std::fs::write(&dflash, tiny::dflash_gguf(7)).unwrap();
                    (opts.drafter, opts.drafter_guesses) = (Some(dflash.clone()), cap);
                }
                Some(Drafted::Dflash2(cap)) => {
                    std::fs::write(&dflash, tiny::dflash2_gguf(7)).unwrap();
                    (opts.drafter, opts.drafter_guesses) = (Some(dflash.clone()), cap);
                }
                Some(Drafted::Mtp(cap)) => {
                    (opts.drafter, opts.drafter_guesses) = (Some("mtp".into()), cap);
                }
                None => {}
            }
            let init = Engine::init(opts, load, None);
            std::fs::remove_file(&path).unwrap();
            if let Some(Drafted::Dflash(_) | Drafted::Dflash2(_)) = draft {
                std::fs::remove_file(&dflash).unwrap();
            }
            let (mut e, h) = init?;
            e.sched.drafter = drafter;
            // Checks as cheap as a decode row, as the CPU's real step times would not be: every
            // check takes all the guesses its sequence may, as before measured draft lengths.
            e.sched.costs = Costs::fixed(|_, _| 1.0);
            Ok((e, h))
        }

        #[derive(Clone, Debug, PartialEq)]
        struct Out {
            ids: Vec<u32>,
            logprobs: Vec<f32>,
            finish: Finish,
            usage: Usage,
        }

        /// Steps the engine until every request ends.
        fn drive(e: &mut Engine, hs: &[RequestHandle]) -> Vec<Out> {
            let mut outs: Vec<Option<Out>> = vec![None; hs.len()];
            let mut ids = vec![Vec::new(); hs.len()];
            let mut logprobs = vec![Vec::new(); hs.len()];
            while outs.iter().any(Option::is_none) {
                assert!(e.turn().unwrap());
                for (i, h) in hs.iter().enumerate() {
                    for ev in h.events.try_iter() {
                        match ev {
                            EngineEvent::Token { id, logprob, .. } => {
                                ids[i].push(id);
                                logprobs[i].push(logprob);
                            }
                            EngineEvent::Done { finish, usage, .. } => {
                                outs[i] = Some(Out {
                                    ids: std::mem::take(&mut ids[i]),
                                    logprobs: std::mem::take(&mut logprobs[i]),
                                    finish,
                                    usage,
                                });
                            }
                            EngineEvent::Prefilled { .. } => {}
                            ev => panic!("{ev:?}"),
                        }
                    }
                }
            }
            outs.into_iter().map(Option::unwrap).collect()
        }

        /// Arms a demand switch to `seats` with deadline `by` (ns), as [`Engine::watch_demand`]
        /// does, on an engine left to demand.
        fn arm(e: &mut Engine, seats: u32, by: u64) {
            let from = e.partial.seats_auto;
            (e.auto, e.switch) = (true, Some(crate::engine::Switch { from, by }));
            (e.partial.seats_auto, e.sched.hold) = (seats, true);
        }

        /// A demand switch reloads under its queue: the requests that had not started run on the
        /// new load, a parked session keeps its KV, and a submit while it loads is queued, not
        /// refused, where a person's reload refuses it.
        #[test]
        fn a_demand_switch_serves_its_queue() {
            let (mut e, h) = engine(Arch::Qwen35, 0, Box::<SuffixDrafter>::default(), 0, false);
            // The test engine's file goes after its load; the switch's reload reads it again, and
            // a plain reload first records the file a carry checks.
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Qwen35, 3)).unwrap();
            plain_reload(&mut e);
            let mut parked = Request::new(vec![1, 2, 3], 2);
            (parked.session, parked.keep_kv) = (7, true);
            drive(&mut e, &[h.submit(parked).unwrap()]);
            assert_eq!(e.sched.n_parked(), 1);
            let queued: Vec<_> = (0..3)
                .map(|i| h.submit(Request::new(vec![4, 5, 6 + i], 3)).unwrap())
                .collect();
            arm(&mut e, MANY_SEATS, u64::MAX);
            assert!(e.turn().unwrap());
            assert_eq!(e.sched.n_parked(), 1, "the reload carried the parked KV");
            assert!(e.switch.is_none() && !e.sched.hold);
            assert!(e.demand.many, "demand flips once the switch loaded");
            assert_eq!(h.health().state, HealthState::Ready);
            let outs = drive(&mut e, &queued);
            assert!(outs.iter().all(|o| o.ids.len() == 3), "{outs:?}");
            // While a switch loads, health is Draining or Loading and submits still queue.
            let set = |state| e.sh.health.lock().unwrap().state = state;
            set(HealthState::Loading);
            let refused = h.submit(Request::new(vec![1], 1));
            assert!(
                matches!(refused, Err(EngineError::NotReady(_))),
                "a person's reload"
            );
            e.sh.switching.store(true, Ordering::Release);
            let during = h.submit(Request::new(vec![1], 1)).expect("queued");
            e.sh.switching.store(false, Ordering::Release);
            set(HealthState::Ready);
            assert_eq!(drive(&mut e, &[during])[0].ids.len(), 1);
            // An idle engine runs a pending switch at once, not after waiting for a message.
            let mut parked = Request::new(vec![1, 2, 3], 2);
            (parked.session, parked.keep_kv) = (8, true);
            drive(&mut e, &[h.submit(parked).unwrap()]);
            arm(&mut e, 1, u64::MAX);
            assert!(e.turn().unwrap());
            assert_eq!(
                e.sched.n_parked(),
                2,
                "switched while idle, in one turn, carrying both"
            );
            assert!(!e.demand.many);
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// A switch holds the queue only while a started request runs and its deadline lasts; a
        /// cancel while it holds survives the reload; a switch down that more requests now wait
        /// on, or one a reload would undo something for, is cancelled; a load that fails before
        /// the teardown keeps the old plan and ends the demand switch.
        #[test]
        fn a_demand_switch_waits_and_cancels() {
            let (mut e, h) = engine(Arch::Qwen35, 0, Box::<SuffixDrafter>::default(), 0, false);
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Qwen35, 3)).unwrap();
            let cancelled = |e: &Engine, from: u32| {
                e.switch.is_none()
                    && !e.sched.hold
                    && e.partial.seats_auto == from
                    && e.demand.quiet_until > e.sh.now()
            };
            // A started request holds the switch until the deadline, which cancels it.
            let long = h.submit(Request::new(vec![1, 2, 3], 40)).unwrap();
            assert!(e.turn().unwrap());
            assert!(e.sched.started());
            let from = e.partial.seats_auto;
            arm(&mut e, MANY_SEATS, u64::MAX);
            let held = h.submit(Request::new(vec![4, 5], 2)).unwrap();
            assert!(e.turn().unwrap());
            assert!(e.switch.is_some() && e.sched.hold, "held while one runs");
            e.switch.as_mut().unwrap().by = 0;
            assert!(e.turn().unwrap());
            assert!(cancelled(&e, from), "past the deadline");
            drive(&mut e, &[long, held]);
            // A cancel while the switch holds the queue ends that request after the reload.
            e.demand.quiet_until = 0;
            let long = h.submit(Request::new(vec![1, 2, 3], 4)).unwrap();
            assert!(e.turn().unwrap());
            arm(&mut e, MANY_SEATS, u64::MAX);
            let gone = h.submit(Request::new(vec![4, 5], 2)).unwrap();
            gone.cancel();
            drive(&mut e, &[long]);
            while e.switch.is_some() {
                assert!(e.turn().unwrap());
            }
            assert!(e.demand.many, "switched");
            let ended = loop {
                assert!(e.turn().unwrap());
                if let Some(ev) = gone.events.try_iter().last() {
                    break ev;
                }
            };
            assert!(
                matches!(ended, EngineEvent::Error(EngineError::Cancelled)),
                "{ended:?}"
            );
            // A switch down with two waiting: demand turned while it held.
            let from = e.partial.seats_auto;
            arm(&mut e, 1, u64::MAX);
            let waiting: Vec<_> = (0..2)
                .map(|i| h.submit(Request::new(vec![4, 5, 6 + i], 2)).unwrap())
                .collect();
            assert!(e.turn().unwrap());
            assert!(cancelled(&e, from), "stale switch down");
            drive(&mut e, &waiting);
            // Each thing a reload would undo or stop blocks demand and cancels a pending switch.
            e.demand.quiet_until = 0;
            let blocks: [fn(&mut Engine, bool); 5] = [
                |e, on| e.auto = !on,
                |e, on| e.sched.version = u64::from(on),
                |e, on| match on {
                    true => e.guard.watch(crate::guard::Target::Ab),
                    false => e.guard.unwatch_ab(),
                },
                |e, on| {
                    let seats = if on { 2.0 } else { 8.0 };
                    e.cfg.values[knob_id::MAX_BATCH_SEQS as usize] = seats;
                },
                |e, on| {
                    e.reload = on.then(|| {
                        let (reply, _) = std::sync::mpsc::sync_channel(1);
                        crate::engine::PendingReload {
                            change: crate::ReloadChange::default(),
                            load: None,
                            reply,
                        }
                    })
                },
            ];
            for (i, block) in blocks.iter().enumerate() {
                arm(&mut e, MANY_SEATS, u64::MAX);
                block(&mut e, true);
                assert!(e.demand_blocked(), "{i}");
                assert!(!e.switch_ready(), "{i}");
                assert!(cancelled(&e, from), "{i}");
                block(&mut e, false);
                e.demand.quiet_until = 0;
                assert!(!e.demand_blocked(), "{i}");
            }
            // A person's seats end the demand switch.
            let now = e.sh.now();
            e.change(knob_id::MAX_BATCH_SEQS, 8.0, Cause::Human, 0, now);
            assert!(!e.auto);
            // A load that fails before the teardown (no file) keeps the plan and the queue.
            std::fs::remove_file(&e.opts.model).unwrap();
            let (many, from) = (e.demand.many, e.partial.seats_auto);
            arm(&mut e, 1, u64::MAX);
            let queued = h.submit(Request::new(vec![4, 5], 2)).unwrap();
            assert!(e.turn().unwrap());
            assert_eq!(h.health().state, HealthState::Ready);
            assert!(!e.auto && e.switch.is_none(), "the demand switch ends");
            assert_eq!((e.demand.many, e.partial.seats_auto), (many, from));
            assert_eq!(drive(&mut e, &[queued])[0].ids.len(), 2);
        }

        /// Demand's clock stops while a gate blocks it: one chat before and after a blocked
        /// stretch (an A/B's watch, busy or not) is not two minutes at one chat.
        #[test]
        fn a_blocked_stretch_restarts_the_demand_clock() {
            let (mut e, _h) = engine(Arch::Qwen35, 0, Box::<SuffixDrafter>::default(), 0, false);
            let s = 1_000_000_000;
            // The plan gates as on the 35B on CUDA, planned for many chats.
            let l = e.loaded.as_mut().unwrap();
            (l.info.caps.backend, l.info.class.experts_in_ram) = (Backend::Cuda, true);
            l.seats_planned = Some(MANY_SEATS);
            (e.auto, e.demand.many) = (true, true);
            e.watch_demand(0);
            assert_eq!(e.demand.since, Some(0), "one chat starts the down clock");
            e.guard.watch(crate::guard::Target::Ab);
            for t in 10..150 {
                e.watch_demand(t * s);
            }
            e.guard.unwatch_ab();
            e.watch_demand(150 * s);
            assert!(e.switch.is_none(), "no two minutes at one chat yet");
            e.watch_demand(270 * s);
            assert!(e.switch.is_some(), "two minutes after the block");
        }

        /// A same-model reload carries a parked session's KV and state: its next turn reuses as
        /// many tokens and writes what it writes with no reload, on a hybrid and a dense model.
        #[test]
        fn a_reload_carries_parked_sessions() {
            for arch in [Arch::Qwen35, Arch::Llama] {
                let turn = |reload: bool| {
                    let (mut e, h) = engine(arch, 0, Box::<SuffixDrafter>::default(), 0, false);
                    // The test engine's file goes after its custom load; a plain reload reads it
                    // again and ties the parks to it.
                    std::fs::write(&e.opts.model, tiny::gguf(arch, 3)).unwrap();
                    plain_reload(&mut e);
                    let a = drive(&mut e, &[h.submit(session7(vec![1, 2, 3, 4])).unwrap()]);
                    if reload {
                        plain_reload(&mut e);
                        assert_eq!(e.sched.n_parked(), 1, "{arch:?}: carried");
                    }
                    let b = drive(&mut e, &[h.submit(session7(next_turn(&a[0]))).unwrap()]);
                    std::fs::remove_file(&e.opts.model).unwrap();
                    b.into_iter().next().unwrap()
                };
                let (kept, carried) = (turn(false), turn(true));
                assert_eq!(
                    kept.usage.cached_tokens, 8,
                    "{arch:?}: control, the park is reused"
                );
                assert_eq!(carried.usage.cached_tokens, 8, "{arch:?}");
                assert_eq!(carried.ids, kept.ids, "{arch:?}");
            }
        }

        /// A file rewritten at the same path (new weights, the same length and write time), a
        /// reload that fails after the carry is taken, or a new pool with no room: each parked
        /// session goes as a Reload reclaim, published, and a changed file's next turn reuses
        /// nothing.
        #[test]
        fn a_reload_drops_what_it_cannot_carry() {
            for arch in [Arch::Qwen35, Arch::Llama] {
                let (mut e, h) = engine(arch, 0, Box::<SuffixDrafter>::default(), 0, false);
                std::fs::write(&e.opts.model, tiny::gguf(arch, 3)).unwrap();
                plain_reload(&mut e);
                let a = drive(&mut e, &[h.submit(session7(vec![1, 2, 3, 4])).unwrap()]);
                let other = tiny::gguf(arch, 4);
                assert_eq!(other.len(), tiny::gguf(arch, 3).len(), "{arch:?}");
                let kept = std::fs::metadata(&e.opts.model)
                    .unwrap()
                    .modified()
                    .unwrap();
                std::fs::write(&e.opts.model, other).unwrap();
                // As `cp -p` would: the write time put back.
                let file = std::fs::File::options().write(true).open(&e.opts.model);
                file.unwrap().set_modified(kept).unwrap();
                let reclaims = ring_reclaims(&h).len();
                plain_reload(&mut e);
                assert_eq!(e.sched.n_parked(), 0, "{arch:?}: another file's KV");
                assert_eq!(reload_reclaims(&h, reclaims), [7], "{arch:?}");
                let b = drive(&mut e, &[h.submit(session7(next_turn(&a[0]))).unwrap()]);
                assert_eq!(b[0].usage.cached_tokens, 0, "{arch:?}");
                // A fault after the carry is out still records the parked session's reclaim.
                let reclaims = ring_reclaims(&h).len();
                e.sh.failed.store(true, Ordering::Release);
                plain_reload(&mut e);
                assert_eq!(reload_reclaims(&h, reclaims), [7], "{arch:?}: failed");
                std::fs::remove_file(&e.opts.model).unwrap();
            }
            // A new pool or state slots with no room drop the carried sequence as a reclaim.
            let (mut e, h) = engine(Arch::Qwen35, 0, Box::<SuffixDrafter>::default(), 0, false);
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Qwen35, 3)).unwrap();
            plain_reload(&mut e);
            drive(&mut e, &[h.submit(session7(vec![1, 2, 3, 4])).unwrap()]);
            let mut carried = e.sched.carry_out();
            e.sched.reset_pool(0);
            let (session, seq) = carried.pop().unwrap();
            assert!(!e.sched.carry_in(session, seq));
            let event = e.sched.reclaims.pop().unwrap();
            assert_eq!(
                (event.session_id, event.reason()),
                (7, larch_records::ReclaimReason::Reload)
            );
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// A pinned document's chain goes through a same-model reload, on a hybrid and a dense
        /// model, and the next question reuses the whole document. Host memory here is the
        /// device's, and a budget short of its KV is the same case on a discrete card: the pin
        /// goes through unnamed files, gone once it is back.
        #[test]
        fn a_pin_survives_a_reload_through_disk() {
            let doc: Vec<u32> = (1..=48).collect();
            let question = |tail: u32| -> Request {
                let mut p = doc.clone();
                p.extend([tail; 3]);
                Request::new(p, 2)
            };
            for arch in [Arch::Qwen35, Arch::Llama] {
                for budget in [usize::MAX, 0] {
                    let (mut e, h) = engine(arch, 0, Box::<SuffixDrafter>::default(), 64, false);
                    std::fs::write(&e.opts.model, tiny::gguf(arch, 3)).unwrap();
                    plain_reload(&mut e);
                    // Pinned as the pool's size allows; a turn's room is a sixth of it here.
                    e.sched.prefix.turn_blocks = e.sched.blocks.total() / 6;
                    e.sched.prefix.pin(1, 0, doc.clone());
                    let read = Request::new(doc.clone(), 1);
                    drive(&mut e, &[h.submit(read).unwrap()]);
                    assert_eq!(e.sched.prefix.pinned(), 3, "{arch:?}: read");
                    e.opts.carry_bytes = Some(budget);
                    plain_reload(&mut e);
                    e.sched.prefix.turn_blocks = e.sched.blocks.total() / 6;
                    assert_eq!(e.sched.prefix.chain((0, 0), &doc).len(), 3, "{arch:?}");
                    let asked = drive(&mut e, &[h.submit(question(7)).unwrap()]);
                    assert_eq!(asked[0].usage.cached_tokens, 48, "{arch:?} {budget}");
                    // Another model drops every pin.
                    e.sched.prefix.unpin_all();
                    assert_eq!(e.sched.prefix.pinned(), 0);
                    std::fs::remove_file(&e.opts.model).unwrap();
                }
            }
        }

        /// A spilled pin whose files do not load is cold and logged, not lost: it stays registered
        /// and the next question over the same tokens still runs.
        #[test]
        fn a_failed_reimport_leaves_the_pin_registered() {
            let doc: Vec<u32> = (1..=48).collect();
            let (mut e, h) = engine(Arch::Qwen35, 0, Box::<SuffixDrafter>::default(), 64, false);
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Qwen35, 3)).unwrap();
            plain_reload(&mut e);
            e.sched.prefix.turn_blocks = e.sched.blocks.total() / 6;
            e.sched.prefix.pin(1, 0, doc.clone());
            drive(&mut e, &[h.submit(Request::new(doc.clone(), 1)).unwrap()]);
            let (pin, mut state) = e.spill_pin(1).unwrap();
            // The sidecar's magic is wrong now.
            std::io::Write::write_all(&mut state, b"garbage!").unwrap();
            std::io::Seek::rewind(&mut state).unwrap();
            let carried = crate::engine::Carry {
                session: 1,
                seq: crate::sched::Seq::of_pin(doc.clone()),
                kv: Vec::new(),
                state: Vec::new(),
                pin: Some(crate::engine::CarriedPin {
                    namespace: 0,
                    checkpoint: None,
                    disk: Some((pin, state)),
                }),
            };
            e.unspill_pin(carried, true);
            assert_eq!(e.sched.prefix.pins().count(), 1);
            let mut ask = doc.clone();
            ask.extend([7; 3]);
            let out = drive(&mut e, &[h.submit(Request::new(ask, 2)).unwrap()]);
            assert_eq!(out[0].usage.prompt_tokens, 51);
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// A carried pin's import the cache has no room for (a reload onto a smaller pool) is
        /// `NoRoom`, not a `Miss` (which the agent answers by deleting the file); the registration
        /// stays, as the caller's handle names it. A hybrid's chain without its state is withdrawn
        /// whole, a dense one's start stays as plain cache.
        #[test]
        fn an_import_without_room_is_no_room_not_a_miss() {
            let doc: Vec<u32> = (1..=48).collect();
            for arch in [Arch::Qwen35, Arch::Llama] {
                let (mut e, h) = engine(arch, 0, Box::<SuffixDrafter>::default(), 64, false);
                std::fs::write(&e.opts.model, tiny::gguf(arch, 3)).unwrap();
                plain_reload(&mut e);
                e.sched.prefix.turn_blocks = e.sched.blocks.total() / 6;
                e.sched.prefix.pin(1, 0, doc.clone());
                drive(&mut e, &[h.submit(Request::new(doc.clone(), 1)).unwrap()]);
                let (mut pin, mut state) = e.spill_pin(1).unwrap();
                // A cache of one block, and a pool that leaves the pin no turn's room.
                e.sched.configure_prefix_cache(1);
                e.sched.prefix.turn_blocks = e.sched.blocks.total();
                let got = crate::pinfile::import(
                    &mut |op| e.pin_op(op),
                    Some(1),
                    &doc,
                    0,
                    &mut pin,
                    &mut state,
                    &mut || Ok(()),
                );
                assert!(matches!(got, Err(crate::PinError::NoRoom)), "{arch:?}");
                assert_eq!(e.sched.prefix.pins().count(), 1, "{arch:?}");
                let held = e.sched.prefix.chain((0, 0), &doc).len();
                assert_eq!(held, usize::from(arch == Arch::Llama), "{arch:?}");
                std::fs::remove_file(&e.opts.model).unwrap();
            }
        }

        /// The documents plan reads the model file's header once per load, not on every call: the
        /// answer stays after the file is gone, and a reload reads it again.
        #[test]
        fn the_documents_plan_reads_the_header_once_per_load() {
            let (mut e, _h) = engine(Arch::Llama, 0, Box::<SuffixDrafter>::default(), 64, false);
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Llama, 3)).unwrap();
            plain_reload(&mut e);
            let first = e.doc_fit(1000).unwrap();
            std::fs::remove_file(&e.opts.model).unwrap();
            assert_eq!(e.doc_fit(1000), Ok(first));
            assert!(e.doc_terms.is_some());
            e.doc_terms = None;
            assert!(
                e.doc_fit(1000).is_err(),
                "a file that is gone is read again"
            );
        }

        /// A pin is a cache: a request that fits beside it leaves it live, one too big for what is
        /// left runs anyway and takes it cold (`pin_live` false), and a short prompt with a long
        /// answer allowed, as an uncapped chat is, completes instead of being refused.
        #[test]
        fn a_request_past_the_turn_room_beside_a_pin_runs_and_the_pin_goes_cold() {
            let doc: Vec<u32> = (1..=48).collect();
            let (mut e, h) = engine(Arch::Llama, 0, Box::<SuffixDrafter>::default(), 64, false);
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Llama, 3)).unwrap();
            plain_reload(&mut e);
            // A pool smaller than the context, as a document's plan leaves beside a big pin.
            e.sched.reset_pool(24);
            e.sched.configure_prefix_cache(8);
            pin(&mut e, 1, 0, doc.clone());
            let hold = crate::PinHandle {
                id: 1,
                tx: h.tx.clone(),
            };
            drive(&mut e, &[h.submit(Request::new(doc.clone(), 1)).unwrap()]);
            assert_eq!(e.sched.prefix.pinned(), 3);
            let small = drive(&mut e, &[h.submit(Request::new(vec![6; 40], 2)).unwrap()]);
            assert_eq!(small[0].usage.completion_tokens, 2);
            assert!(h.pin_live(&hold), "it fitted beside the pin");
            // 20 prompt tokens, an answer allowed to fill the context: admitted, and it completes.
            let long = drive(&mut e, &[h.submit(Request::new(vec![7; 20], 352)).unwrap()]);
            assert!(long[0].usage.completion_tokens > 0);
            let room = (24 - 3) * crate::sched::BLOCK_TOKENS;
            let big = drive(&mut e, &[h.submit(Request::new(vec![5; room], 1)).unwrap()]);
            assert_eq!(big[0].usage.completion_tokens, 1);
            assert!(!h.pin_live(&hold), "its blocks were the pin's");
            assert_eq!(e.sched.prefix.pinned(), 0);
            // The hold's drop is a no-op for the pin that is gone, and forgets its mark.
            drop(hold);
            assert!(e.turn().unwrap());
            assert!(e.sh.cold_pins.lock().unwrap().is_empty());
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// A request waiting while a pin grows past what it needs is not refused: the pin goes cold
        /// and it runs.
        #[test]
        fn a_waiting_request_a_growing_pin_leaves_no_room_for_takes_it_cold() {
            let doc: Vec<u32> = (1..=48).collect();
            let (mut e, h) = engine(Arch::Llama, 0, Box::<SuffixDrafter>::default(), 64, false);
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Llama, 3)).unwrap();
            plain_reload(&mut e);
            e.sched.reset_pool(24);
            e.sched.configure_prefix_cache(8);
            pin(&mut e, 1, 0, doc.clone());
            // 22 blocks of 24: no pin yet, so it is admitted, and held in the queue.
            e.sched.hold = true;
            let waiting = h.submit(Request::new(vec![5; 337], 1)).unwrap();
            assert!(e.turn().unwrap());
            assert_eq!(e.sched.n_waiting(), 1);
            // The pin's chain is published: 3 blocks held, 21 left.
            let blocks = e.sched.alloc_pin(3).unwrap();
            assert_eq!(e.sched.publish_pin(1, 0, &doc, blocks, None).len(), 3);
            e.sched.hold = false;
            let out = drive(&mut e, &[waiting]);
            assert_eq!(out[0].usage.completion_tokens, 1);
            assert_eq!(e.sched.prefix.pinned(), 0);
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// An import takes the blocks of an unpinned chat's cached prefix when the pool is short of
        /// free ones, as a request's allocation would: no room only when pins and a turn's room
        /// leave none.
        #[test]
        fn an_import_reclaims_unpinned_cache_blocks() {
            let doc: Vec<u32> = (1..=320).map(|i| i % 200 + 1).collect();
            let (mut e, h) = engine(Arch::Llama, 0, Box::<SuffixDrafter>::default(), 64, false);
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Llama, 3)).unwrap();
            plain_reload(&mut e);
            e.sched.prefix.turn_blocks = 1;
            e.sched.prefix.pin(1, 0, doc.clone());
            drive(&mut e, &[h.submit(Request::new(doc.clone(), 1)).unwrap()]);
            let (mut pin, mut state) = e.spill_pin(1).unwrap();
            e.sched.unpin(1);
            // A pool of 24 blocks, eight of them an unpinned chat's cached prefix: 16 free, and
            // the 20-block pin needs the others.
            e.sched.reset_pool(24);
            e.sched.configure_prefix_cache(8);
            e.sched.prefix.turn_blocks = 1;
            let chat: Vec<u32> = (0..129).map(|i| 201 + i % 50).collect();
            drive(&mut e, &[h.submit(Request::new(chat, 1)).unwrap()]);
            assert_eq!(e.sched.prefix.len(), 8);
            let got = crate::pinfile::import(
                &mut |op| e.pin_op(op),
                None,
                &doc,
                0,
                &mut pin,
                &mut state,
                &mut || Ok(()),
            );
            assert!(got.is_ok(), "{:?}", got.err());
            assert_eq!(e.sched.prefix.chain((0, 0), &doc).len(), 20);
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// Pins and a turn's room must fit the pool: a pin past that is `PINS_FULL`, never a silently
        /// cold one; a prompt that shares a pinned prefix costs only its own blocks.
        #[test]
        fn pins_past_the_pool_are_refused() {
            let (mut e, _h) = engine(Arch::Llama, 0, Box::<SuffixDrafter>::default(), 64, false);
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Llama, 3)).unwrap();
            plain_reload(&mut e);
            let try_pin = |e: &mut Engine, id: u64, tokens: Vec<u32>| {
                let (reply, rx) = std::sync::mpsc::sync_channel(1);
                e.handle(crate::Msg::Pin(id, 0, tokens, reply)).unwrap();
                rx.recv().unwrap()
            };
            // Room for two blocks of pins.
            e.sched.prefix.turn_blocks = e.sched.blocks.total() - 2;
            let full = Err(EngineError::InvalidRequest(crate::invalid::PINS_FULL));
            let doc: Vec<u32> = (1..=48).collect();
            assert_eq!(try_pin(&mut e, 1, doc.clone()), full);
            assert_eq!(try_pin(&mut e, 2, doc[..32].to_vec()), Ok(()));
            assert_eq!(try_pin(&mut e, 3, doc[..16].to_vec()), Ok(()), "shared");
            assert_eq!(try_pin(&mut e, 4, vec![9; 16]), full);
            // An import's blocks are bounded the same way.
            e.handle(crate::Msg::Unpin(2)).unwrap();
            e.handle(crate::Msg::Unpin(3)).unwrap();
            assert_eq!(try_pin(&mut e, 5, doc[..32].to_vec()), Ok(()));
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// Runs a reload of `change` from the engine's file.
        fn reload_to(e: &mut Engine, change: crate::ReloadChange) {
            let (reply, _) = std::sync::mpsc::sync_channel(1);
            e.reload = Some(crate::engine::PendingReload {
                change,
                load: None,
                reply,
            });
            assert!(e.turn().unwrap());
        }

        /// Registers a pin as `Msg::Pin` does, with a turn's room that fits this small pool.
        fn pin(e: &mut Engine, id: u64, namespace: u64, tokens: Vec<u32>) {
            e.sched.prefix.turn_blocks = 1;
            let (reply, rx) = std::sync::mpsc::sync_channel(1);
            e.handle(crate::Msg::Pin(id, namespace, tokens, reply))
                .unwrap();
            rx.recv().unwrap().unwrap();
        }

        /// A documents reload for a context of `ctx` tokens.
        fn documents(ctx: u32) -> crate::ReloadChange {
            crate::ReloadChange {
                max_context: Some(ctx),
                kv_pool_fraction: Some(0.9),
                max_batch_seqs: Some(2),
                drafter_off: true,
                documents: true,
                ..crate::ReloadChange::default()
            }
        }

        /// A second, bigger document's reload keeps the first pin: both chains go through it
        /// (their own carry, not a session's), and each document's question reuses all of its own.
        #[test]
        fn a_second_bigger_documents_reload_keeps_the_first_pin() {
            let small: Vec<u32> = (1..=48).collect();
            let big: Vec<u32> = (0..160).map(|i| i % 97 + 1).collect();
            let question = |doc: &[u32]| {
                let mut p = doc.to_vec();
                p.extend([7; 3]);
                Request::new(p, 2)
            };
            for arch in [Arch::Qwen35, Arch::Llama] {
                let (mut e, h) = engine(arch, 0, Box::<SuffixDrafter>::default(), 64, false);
                std::fs::write(&e.opts.model, tiny::gguf(arch, 3)).unwrap();
                plain_reload(&mut e);
                e.sched.prefix.pin(1, 0, small.clone());
                drive(&mut e, &[h.submit(Request::new(small.clone(), 1)).unwrap()]);
                reload_to(&mut e, documents(1024));
                assert_eq!(
                    e.sched.prefix.chain((0, 0), &small).len(),
                    3,
                    "{arch:?}: first"
                );
                e.sched.prefix.pin(2, 0, big.clone());
                drive(&mut e, &[h.submit(Request::new(big.clone(), 1)).unwrap()]);
                assert_eq!(e.sched.prefix.chain((0, 0), &big).len(), 10, "{arch:?}");
                // The bigger plan: both carried, both still pinned, both read again as needed.
                reload_to(&mut e, documents(2048));
                assert_eq!(e.opts.max_context, 2048);
                assert_eq!(
                    e.sched.prefix.chain((0, 0), &small).len(),
                    3,
                    "{arch:?}: kept"
                );
                assert_eq!(
                    e.sched.prefix.chain((0, 0), &big).len(),
                    10,
                    "{arch:?}: carried"
                );
                for (doc, cached) in [(&small, 48), (&big, 160)] {
                    let out = drive(&mut e, &[h.submit(question(doc)).unwrap()]);
                    assert_eq!(out[0].usage.cached_tokens, cached, "{arch:?}");
                }
                std::fs::remove_file(&e.opts.model).unwrap();
            }
        }

        /// An engine on the documents plan, with the plan before it noted: (context, pool fraction,
        /// seats) of the plan before.
        fn on_documents_plan(arch: Arch) -> (Engine, EngineHandle, (u32, f32, f64)) {
            let (mut e, h) = engine(arch, 0, Box::<SuffixDrafter>::default(), 64, false);
            std::fs::write(&e.opts.model, tiny::gguf(arch, 3)).unwrap();
            plain_reload(&mut e);
            let before = (
                e.opts.max_context,
                e.opts.kv_pool_fraction,
                e.cfg.values[knob_id::MAX_BATCH_SEQS as usize],
            );
            e.opts.drafter = Some("drafter".into());
            e.opts.drafter_auto = Some("auto".into());
            reload_to(&mut e, documents(2048));
            assert_eq!((e.opts.max_context, e.opts.kv_pool_fraction), (2048, 0.9));
            (e, h, before)
        }

        /// A pin that is dropped and taken again (its chat let it go, then asked again) within
        /// the idle time keeps the documents plan; with none held and the engine idle for a
        /// minute, and not before, the plan before comes back, drafter and all.
        #[test]
        fn the_documents_plan_returns_after_an_idle_minute_with_no_pin() {
            let (mut e, h, (ctx, fraction, seats)) = on_documents_plan(Arch::Llama);
            let doc: Vec<u32> = (1..=48).collect();
            let secs = |s: u64| std::time::Instant::now() - std::time::Duration::from_secs(s);
            let turns = |e: &mut Engine| (0..4).for_each(|_| assert!(e.turn().unwrap()));
            pin(&mut e, 1, 0, doc.clone());
            // A pin held outlasts any idle time.
            e.idle_since = secs(3600);
            turns(&mut e);
            assert_eq!(e.opts.max_context, 2048, "a pin is held");
            // Dropped and taken again: the clock starts at the drop, and the pin returns before it
            // ends.
            e.handle(crate::Msg::Unpin(1)).unwrap();
            e.idle_since = secs(30);
            turns(&mut e);
            pin(&mut e, 2, 0, doc.clone());
            e.handle(crate::Msg::Unpin(2)).unwrap();
            turns(&mut e);
            assert_eq!(e.opts.max_context, 2048, "a round trip within a minute");
            // A request running never sees it, however long the engine idled before.
            e.idle_since = secs(61);
            let long = h.submit(Request::new(vec![1, 2, 3], 200)).unwrap();
            assert!(e.turn().unwrap());
            assert_eq!(e.opts.max_context, 2048, "a request runs");
            drive(&mut e, &[long]);
            e.idle_since = secs(30);
            turns(&mut e);
            assert_eq!(e.opts.max_context, 2048, "idle for less than a minute");
            e.idle_since = secs(61);
            turns(&mut e);
            assert_eq!(
                (e.opts.max_context, e.opts.kv_pool_fraction),
                (ctx, fraction)
            );
            assert_eq!(e.cfg.values[knob_id::MAX_BATCH_SEQS as usize], seats);
            assert_eq!(
                e.opts.drafter.as_deref(),
                Some(std::path::Path::new("drafter"))
            );
            assert_eq!(
                e.opts.drafter_auto.as_deref(),
                Some(std::path::Path::new("auto"))
            );
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// A fit asked inside the documents plan reads the device as the load before the plan found
        /// it: a reload that found less free (Metal's stream keeps buffers across one) does not turn
        /// a document the plan was loaded for into one too long for it (#590 F1).
        #[test]
        fn a_fit_inside_the_documents_plan_answers_as_the_one_that_reloaded_into_it() {
            let (mut e, _h) = engine(Arch::Llama, 0, Box::<SuffixDrafter>::default(), 64, false);
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Llama, 3)).unwrap();
            plain_reload(&mut e);
            let outside = e.doc_fit(1000).unwrap();
            let seats = e.cfg.values[knob_id::MAX_BATCH_SEQS as usize];
            reload_to(&mut e, documents(2048));
            assert!(e.docs_before.is_some());
            // The plan's own load found almost nothing free: with the seats it may pick as they
            // were, the fit inside it is still the one outside (the tiny model's seats follow the
            // memory, so a fit on the plan's own reading picks fewer).
            e.loaded.as_mut().unwrap().free_before = 1 << 20;
            e.cfg.values[knob_id::MAX_BATCH_SEQS as usize] = seats;
            assert_eq!(e.doc_fit(1000), Ok(outside));
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// A load into the documents plan stands with no host expert layers, or with ones that
        /// stream; one the CPU would prefill does not fit.
        #[test]
        fn a_load_into_the_documents_plan_keeps_no_layers_the_cpu_prefills() {
            use crate::engine::direct_documents_fit as fits;
            assert!(fits(0, false) && fits(0, true) && fits(11, true));
            assert!(!fits(3, false));
        }

        /// A load straight into the documents plan turns the demand switch off; the plan going
        /// back turns it on again, as the options' own load has it.
        #[test]
        fn the_demand_switch_comes_back_with_the_plan_a_load_turned_it_off_for() {
            let (mut e, _h, _) = on_documents_plan(Arch::Llama);
            // As `Engine::new` leaves a load into the plan (`EngineOptions::documents`) whose
            // options nobody set the seats of.
            e.auto = false;
            let before = e.docs_before.as_mut().unwrap();
            (before.set.auto_off, before.seats.1) = (true, false);
            e.idle_since = std::time::Instant::now() - std::time::Duration::from_secs(61);
            (0..4).for_each(|_| assert!(e.turn().unwrap()));
            assert!(e.docs_before.is_none() && e.auto);
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// A request that must wait for a seat the documents plan lacks, with no pin held, has the
        /// plan go back before it runs: the started requests end first (none is cut off), the
        /// held one then runs on the plan before.
        #[test]
        fn a_request_without_a_seat_restores_the_documents_plan_before_it_runs() {
            let (mut e, h, (ctx, ..)) = on_documents_plan(Arch::Llama);
            let doc: Vec<u32> = (1..=48).collect();
            pin(&mut e, 1, 0, doc.clone());
            e.handle(crate::Msg::Unpin(1)).unwrap();
            let seats = e.cfg.values[knob_id::MAX_BATCH_SEQS as usize] as usize;
            let hs: Vec<_> = (0..=seats)
                .map(|i| {
                    h.submit(Request::new(vec![1, 2, 3 + i as u32], 100))
                        .unwrap()
                })
                .collect();
            let mut done = vec![false; hs.len()];
            let mut restored_at = None;
            while !done.iter().all(|d| *d) {
                assert!(e.turn().unwrap());
                if e.opts.max_context == ctx && restored_at.is_none() {
                    restored_at = Some(done.iter().filter(|d| **d).count());
                }
                for (d, h) in done.iter_mut().zip(&hs) {
                    for ev in h.events.try_iter() {
                        match ev {
                            EngineEvent::Done { .. } => *d = true,
                            EngineEvent::Error(err) => panic!("{err:?}"),
                            _ => {}
                        }
                    }
                }
            }
            // The plan went back only after the two seated requests had finished, before the third.
            assert_eq!(restored_at, Some(seats));
            assert!(e.docs_before.is_none() && !e.restoring);
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// A pin that arrives with a restore due is seen first: the plan stays, nothing is queued
        /// for the pin to find.
        #[test]
        fn a_pin_arriving_with_a_restore_due_keeps_the_documents_plan() {
            let (mut e, h, _) = on_documents_plan(Arch::Llama);
            e.sched.prefix.turn_blocks = 1;
            e.idle_since = std::time::Instant::now() - std::time::Duration::from_secs(61);
            let (reply, rx) = std::sync::mpsc::sync_channel(1);
            h.tx.send(crate::Msg::Pin(9, 0, (1..=48).collect(), reply))
                .unwrap();
            assert!(e.turn().unwrap());
            assert_eq!(rx.recv().unwrap(), Ok(()));
            assert!(e.reload.is_none());
            assert_eq!(e.opts.max_context, 2048);
            assert_eq!(e.sched.prefix.pins().count(), 1);
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// A pin file's import that is open keeps the documents plan however long the engine idled
        /// with no pin held: a restore would drop the import's blocks, and its next step's
        /// `PIN_CHANGED` reads as a bad file, which the caller deletes. A documents reload and
        /// each pin step start the idle clock again; with the import ended and no pin, the plan
        /// goes back (#590 Mac review).
        #[test]
        fn an_open_import_keeps_the_documents_plan() {
            let (mut e, _h, (ctx, ..)) = on_documents_plan(Arch::Llama);
            let secs = |s: u64| std::time::Instant::now() - std::time::Duration::from_secs(s);
            let turns = |e: &mut Engine| (0..4).for_each(|_| assert!(e.turn().unwrap()));
            let op = |e: &mut Engine, op| {
                let (reply, rx) = std::sync::mpsc::sync_channel(1);
                e.handle(crate::Msg::PinOp(op, reply)).unwrap();
                rx.recv().unwrap()
            };
            e.idle_since = secs(61);
            reload_to(&mut e, documents(2048));
            assert!(!e.restore_ready(), "a documents reload starts the clock");
            e.sched.prefix.turn_blocks = 1;
            let begin = crate::PinOp::Begin {
                id: None,
                namespace: 0,
                tokens: (1..=48).collect(),
            };
            let Ok(crate::PinReply::Id(id)) = op(&mut e, begin) else {
                panic!("begin");
            };
            assert_eq!(e.sched.prefix.pins().count(), 0, "nothing published yet");
            e.idle_since = secs(61);
            assert!(!e.restore_ready(), "an import is open");
            turns(&mut e);
            assert_eq!(e.opts.max_context, 2048, "the plan stays for the import");
            assert!(e.imports.contains_key(&id), "the import's blocks stay");
            e.idle_since = secs(61);
            let abort = op(&mut e, crate::PinOp::Abort(id));
            assert!(matches!(abort, Ok(crate::PinReply::Done)));
            assert!(!e.restore_ready(), "a pin step starts the clock");
            e.idle_since = secs(61);
            assert!(e.restore_ready(), "the import ended and no pin is held");
            turns(&mut e);
            assert_eq!(e.opts.max_context, ctx);
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// An import a reload broke off (another chat's documents reload) hears so at its next
        /// step: `IMPORT_TORN`, not `PIN_CHANGED`, which reads as a bad file that the caller
        /// deletes. Each torn import is told once, an abort forgets it, and at most `TORN_KEPT`
        /// are kept (#591 review).
        #[test]
        fn an_import_a_reload_tore_is_told_so_once() {
            let (mut e, _h, _) = on_documents_plan(Arch::Llama);
            let op = |e: &mut Engine, op| {
                let (reply, rx) = std::sync::mpsc::sync_channel(1);
                e.handle(crate::Msg::PinOp(op, reply)).unwrap();
                match rx.recv().unwrap() {
                    Ok(crate::PinReply::Id(id)) => Ok(id),
                    Ok(_) => Ok(0),
                    Err(EngineError::InvalidRequest(code)) => Err(code),
                    Err(e) => panic!("{e:?}"),
                }
            };
            let begin = |e: &mut Engine| {
                // A turn's room that fits this small pool; a reload sets it again.
                e.sched.prefix.turn_blocks = 1;
                let tokens = (1..=16).collect();
                (op(
                    e,
                    crate::PinOp::Begin {
                        id: None,
                        namespace: 0,
                        tokens,
                    },
                ))
                .unwrap()
            };
            let upload = |id| crate::PinOp::Upload {
                id,
                layer: 0,
                from_block: 0,
                bytes: Vec::new(),
            };
            let finish = |id| crate::PinOp::Finish(id, Vec::new(), 16);
            let id = begin(&mut e);
            reload_to(&mut e, documents(4096));
            assert!(
                e.imports.is_empty(),
                "the import's blocks went with the pool"
            );
            assert_eq!(op(&mut e, upload(id)), Err(crate::invalid::IMPORT_TORN));
            assert_eq!(
                op(&mut e, finish(id)),
                Err(crate::invalid::PIN_CHANGED),
                "told once"
            );
            let ids: Vec<u64> = (0..=super::super::TORN_KEPT)
                .map(|_| begin(&mut e))
                .collect();
            plain_reload(&mut e);
            assert_eq!(e.torn.len(), super::super::TORN_KEPT, "bounded");
            let given_up = e.torn[0];
            assert_eq!(op(&mut e, crate::PinOp::Abort(given_up)), Ok(0));
            assert_eq!(
                op(&mut e, upload(given_up)),
                Err(crate::invalid::PIN_CHANGED)
            );
            let told = (ids.iter())
                .filter(|&&id| op(&mut e, finish(id)) == Err(crate::invalid::IMPORT_TORN))
                .count();
            assert_eq!(told, super::super::TORN_KEPT - 1);
            assert!(e.torn.is_empty());
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// A reload measures the device on a new stream, once the old model is gone: a stream
        /// keeps what it grew to serve a plan (Metal's staging copy and scratch), which counted as
        /// used at the next load, so each reload found less free than the first and planned less
        /// (#591 Mac check). The pin the reload carries comes back on the new stream, and the
        /// handle tells what the load found free.
        #[test]
        fn a_reload_runs_on_a_new_stream() {
            let (mut e, h) = engine(Arch::Llama, 0, Box::<SuffixDrafter>::default(), 64, false);
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Llama, 3)).unwrap();
            plain_reload(&mut e);
            let doc: Vec<u32> = (1..=48).collect();
            e.sched.prefix.pin(1, 0, doc.clone());
            drive(&mut e, &[h.submit(Request::new(doc.clone(), 1)).unwrap()]);
            let old = e.s.id();
            plain_reload(&mut e);
            assert_ne!(e.s.id(), old, "a new stream");
            assert_eq!(h.free_before(), e.loaded.as_ref().unwrap().free_before);
            let mut question = doc.clone();
            question.extend([7; 3]);
            let out = drive(&mut e, &[h.submit(Request::new(question, 2)).unwrap()]);
            assert_eq!(out[0].usage.cached_tokens, 48, "the carried pin");
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// Going back undoes only what the documents reload changed and the person has not
        /// changed since: seats set in between stay, the rest returns.
        #[test]
        fn restoring_the_plan_keeps_what_the_person_changed() {
            let (mut e, _h, (ctx, fraction, _)) = on_documents_plan(Arch::Llama);
            e.cfg.values[knob_id::MAX_BATCH_SEQS as usize] = 3.0;
            e.opts.kv_pool_fraction = 0.5;
            e.idle_since = std::time::Instant::now() - std::time::Duration::from_secs(61);
            for _ in 0..4 {
                assert!(e.turn().unwrap());
            }
            assert_eq!(
                e.opts.max_context, ctx,
                "the documents plan's context goes back"
            );
            assert_eq!(
                e.opts.kv_pool_fraction, 0.5,
                "a pool fraction set since stays"
            );
            assert_ne!(e.opts.kv_pool_fraction, fraction);
            assert_eq!(e.cfg.values[knob_id::MAX_BATCH_SEQS as usize], 3.0);
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// The documents plan's reload sets the seats and drops the drafter without costing the
        /// cache or a parked session: the new seats are the knob's value, as a hand-set one, and
        /// the session is carried.
        #[test]
        fn a_documents_reload_sets_seats_and_drafter_and_keeps_the_cache() {
            let arch = Arch::Qwen35;
            let (mut e, h) = engine(arch, 0, Box::<SuffixDrafter>::default(), 64, false);
            std::fs::write(&e.opts.model, tiny::gguf(arch, 3)).unwrap();
            plain_reload(&mut e);
            e.opts.drafter = Some("drafter".into());
            let a = drive(&mut e, &[h.submit(session7(vec![1, 2, 3, 4])).unwrap()]);
            let (reply, _) = std::sync::mpsc::sync_channel(1);
            e.reload = Some(crate::engine::PendingReload {
                change: crate::ReloadChange {
                    max_batch_seqs: Some(2),
                    drafter_off: true,
                    ..crate::ReloadChange::default()
                },
                load: None,
                reply,
            });
            assert!(e.turn().unwrap());
            assert!(e.opts.drafter.is_none() && e.opts.drafter_auto.is_none());
            assert_eq!(e.cfg.values[knob_id::MAX_BATCH_SEQS as usize], 2.0);
            assert_eq!(e.sched.n_parked(), 1, "carried");
            let b = drive(&mut e, &[h.submit(session7(next_turn(&a[0]))).unwrap()]);
            assert_eq!(b[0].usage.cached_tokens, 8);
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// A 40-token system prompt and then `tail`, as `session`'s turn (0: stateless).
        fn shared(session: u64, tail: &[u32]) -> Request {
            let prompt = (1..=40).chain(tail.iter().copied()).collect();
            Request {
                session,
                keep_kv: session != 0,
                ..Request::new(prompt, 5)
            }
        }

        /// A request built from the outputs of the turns before it.
        type Turn = fn(&[Out]) -> Request;

        /// The hybrid tiny model running `turns` one after another, with checkpoints in a budget
        /// of `budget(state bytes)` or with no prefix cache (`None`, today's behaviour), in prompt
        /// chunks of at most `chunk` tokens: each turn's output.
        fn hybrid_turns(
            budget: Option<fn(usize) -> usize>,
            chunk: usize,
            turns: &[Turn],
        ) -> Vec<Out> {
            let prefix = if budget.is_some() { 64 } else { 0 };
            let (mut e, h) = engine(
                Arch::Qwen35,
                0,
                Box::<SuffixDrafter>::default(),
                prefix,
                false,
            );
            e.cfg.values[knob_id::PREFILL_CHUNK_TOKENS as usize] = chunk as f64;
            if let Some(budget) = budget {
                let state = e.loaded.as_ref().unwrap().model.state_size();
                e.sched.prefix.state_budget = budget(state);
            }
            let mut outs = Vec::new();
            for turn in turns {
                let r = turn(&outs);
                outs.extend(drive(&mut e, &[h.submit(r).unwrap()]));
            }
            outs
        }

        /// Turns that reuse a checkpoint: a session's second turn (its parked state), the same
        /// history sent stateless twice (the first turn's prompt end where a chunk ended there,
        /// else the first stateless one's, which it cuts a chunk for), a second and a third session
        /// sharing the system prompt (the second saves where it diverges, the third loads it), and
        /// a prompt diverging inside a block of the first turn's (back to the last checkpoint).
        fn checkpoint_turns() -> [Turn; 7] {
            [
                |_| shared(7, &[50; 9]),
                |o| shared(7, &second(&o[0])),
                |o| shared(0, &second(&o[0])),
                |o| shared(0, &second(&o[0])),
                |_| shared(8, &[60; 9]),
                |_| shared(9, &[70; 9]),
                |_| shared(10, &[50, 50, 50, 50, 50, 90, 90, 90, 90]),
            ]
        }

        /// The tail of the turn after [`shared`]`(_, &[50; 9])` answered `first`.
        fn second(first: &Out) -> Vec<u32> {
            [&[50; 9][..], &first.ids[..4], &[5, 6]].concat()
        }

        /// A hybrid answers alike with checkpoints and without, token for token, and with them
        /// computes only each prompt's new tokens: with room for every checkpoint, for one (each
        /// save evicts the one before: in 16-token chunks a kept session's own prompt end at 48,
        /// where its chunk ends anyway, evicts its divergence at 32 before the next session reads
        /// it), or for none. A kept session's 49-token prompt is one 512-token chunk, so it saves
        /// no prompt end there. In 16-token chunks, where both cut
        /// a prompt at the same positions, the logprobs are equal too: a checkpoint is the state
        /// exactly. (Elsewhere the CPU's numerics move with where a chunk ends, by about 1e-4
        /// nats here: so does a parked turn's against the same prompt read cold.)
        #[test]
        fn checkpoints_answer_as_without_them() {
            let turns = checkpoint_turns();
            let same = |on: &[Out], off: &[Out], logprobs: bool| {
                for (a, b) in on.iter().zip(off) {
                    assert_eq!(a.ids, b.ids);
                    assert!(!logprobs || a.logprobs == b.logprobs, "{a:?} {b:?}");
                }
            };
            let off = hybrid_turns(None, 512, &turns);
            let off16 = hybrid_turns(None, 16, &turns);
            let cached =
                |outs: &[Out]| -> Vec<u32> { outs.iter().map(|o| o.usage.cached_tokens).collect() };
            // Without them only the parked session reuses: 49 prompt tokens and 4 fed answers.
            assert_eq!(cached(&off), [0, 53, 0, 0, 0, 0, 0]);
            let every: fn(usize) -> usize = |_| usize::MAX;
            let one: fn(usize) -> usize = |n| n;
            let none: fn(usize) -> usize = |n| n - 1;
            for (budget, want, want16) in [
                (every, [0, 53, 0, 48, 0, 32, 32], [0, 53, 48, 48, 0, 32, 32]),
                (one, [0, 53, 0, 48, 0, 32, 32], [0, 53, 48, 48, 0, 0, 0]),
                (none, [0, 53, 0, 0, 0, 0, 0], [0, 53, 0, 0, 0, 0, 0]),
            ] {
                let on = hybrid_turns(Some(budget), 512, &turns);
                assert_eq!(cached(&on), want);
                same(&on, &off, false);
                let on16 = hybrid_turns(Some(budget), 16, &turns);
                assert_eq!(cached(&on16), want16);
                same(&on16, &off16, true);
            }
        }

        /// A reload, a plan switch and a knob that drops the prefix cache drop its checkpoints;
        /// the reload still carries the parked session, whose next turn reuses it whole.
        #[test]
        fn checkpoints_go_with_the_prefix_cache() {
            let (mut e, h) = engine(Arch::Qwen35, 0, Box::<SuffixDrafter>::default(), 64, false);
            std::fs::write(&e.opts.model, tiny::gguf(Arch::Qwen35, 3)).unwrap();
            plain_reload(&mut e);
            let run = |e: &mut Engine, r| drive(e, &[h.submit(r).unwrap()]).remove(0);
            let first = run(&mut e, shared(7, &[50; 9]));
            // The knob goes last: it also ends parked reuse (a new KV generation), as on dev.
            let drops: [&dyn Fn(&mut Engine); 3] = [
                &plain_reload,
                &|e| {
                    arm(e, MANY_SEATS, u64::MAX);
                    assert!(e.turn().unwrap());
                    assert!(e.switch.is_none());
                },
                &|e| {
                    let now = e.sh.now();
                    e.change(knob_id::PREFILL_CHUNK_TOKENS, 64.0, Cause::Human, 0, now);
                },
            ];
            for (i, drop) in drops.iter().enumerate() {
                let tail = [60 + i as u32; 9];
                assert_eq!(run(&mut e, shared(0, &tail)).usage.cached_tokens, 0, "{i}");
                assert_eq!(run(&mut e, shared(0, &tail)).usage.cached_tokens, 48, "{i}");
                drop(&mut e);
                assert_eq!(run(&mut e, shared(0, &tail)).usage.cached_tokens, 0, "{i}");
                if i == 1 {
                    assert_eq!(e.sched.n_parked(), 1, "carried");
                    let next = run(&mut e, shared(7, &second(&first)));
                    assert_eq!(next.usage.cached_tokens, 53);
                }
            }
            std::fs::remove_file(&e.opts.model).unwrap();
        }

        /// Router deltas refuse a hybrid's checkpoints as they refuse its KV, and a version number
        /// that comes back with other deltas reuses nothing computed under the old ones.
        #[test]
        fn router_deltas_refuse_hybrid_checkpoints() {
            let (mut e, h) = engine(
                Arch::Qwen35Moe,
                0,
                Box::<SuffixDrafter>::default(),
                64,
                false,
            );
            let swap = |e: &mut Engine, sign: f32, version| {
                let d = crate::RouterDeltas {
                    layers: vec![0, 1],
                    experts: tiny::EXPERTS,
                    deltas: vec![vec![4.0 * sign, -4.0 * sign, 4.0 * sign, -4.0 * sign]; 2],
                    ..Default::default()
                };
                let h = h.clone();
                let t = std::thread::spawn(move || h.set_router_deltas(Some(d), version));
                while !t.is_finished() {
                    assert!(e.turn().unwrap());
                }
                t.join().unwrap().unwrap();
            };
            let run = |e: &mut Engine| {
                let r = Request::new((1..=40).collect(), 2);
                drive(e, &[h.submit(r).unwrap()])
                    .remove(0)
                    .usage
                    .cached_tokens
            };
            assert_eq!((run(&mut e), run(&mut e)), (0, 32), "a stateless repeat");
            swap(&mut e, 1.0, 1);
            assert_eq!((run(&mut e), run(&mut e)), (0, 32), "new deltas");
            swap(&mut e, -1.0, 1);
            assert_eq!(run(&mut e), 0, "the same version with other deltas");
        }

        /// Session 7's request, its KV kept.
        fn session7(prompt: Vec<u32>) -> Request {
            Request {
                session: 7,
                keep_kv: true,
                ..Request::new(prompt, 5)
            }
        }

        /// The turn after `a` (a [`session7`] turn on 1..=4): the parked history is the prompt and
        /// every answer token but the last, then two new tokens.
        fn next_turn(a: &Out) -> Vec<u32> {
            let mut next = vec![1, 2, 3, 4];
            next.extend(&a.ids[..4]);
            next.extend([5, 6]);
            next
        }

        /// Runs a reload of the same options from the engine's file.
        fn plain_reload(e: &mut Engine) {
            let (reply, _) = std::sync::mpsc::sync_channel(1);
            e.reload = Some(crate::engine::PendingReload {
                change: crate::ReloadChange::default(),
                load: None,
                reply,
            });
            if e.failed() {
                e.rebuild();
            } else {
                assert!(e.turn().unwrap());
            }
        }

        /// The ring's reclaims, oldest first.
        fn ring_reclaims(h: &EngineHandle) -> Vec<larch_records::ReclaimEvent> {
            let mut reader = h.ring().reader(larch_records::ReadFrom::Oldest);
            let mut out = Vec::new();
            while let larch_records::Read::Record(r) = reader.next() {
                if let larch_records::Record::ReclaimEvent(_, c) = r {
                    out.push(c);
                }
            }
            out
        }

        /// The sessions of the Reload reclaims written past the ring's first `skip`.
        fn reload_reclaims(h: &EngineHandle, skip: usize) -> Vec<u64> {
            (ring_reclaims(h).into_iter().skip(skip))
                .filter(|c| c.reason() == larch_records::ReclaimReason::Reload)
                .map(|c| c.session_id)
                .collect()
        }

        fn run(
            arch: Arch,
            refuse: bool,
            guesses: u32,
            drafter: Box<dyn Drafter + Send>,
            reqs: &[Request],
        ) -> (Vec<Out>, GuessStats) {
            run_costs(arch, refuse, guesses, drafter, None, reqs)
        }

        /// [`run`] with step costs `costs` in place of [`engine_with`]'s, if some.
        fn run_costs(
            arch: Arch,
            refuse: bool,
            guesses: u32,
            drafter: Box<dyn Drafter + Send>,
            costs: Option<Costs>,
            reqs: &[Request],
        ) -> (Vec<Out>, GuessStats) {
            let (mut e, h) = engine(arch, guesses, drafter, 0, refuse);
            if let Some(c) = costs {
                e.sched.costs = c;
            }
            let hs: Vec<_> = reqs.iter().map(|r| h.submit(r.clone()).unwrap()).collect();
            let outs = drive(&mut e, &hs);
            (outs, h.guess_stats())
        }

        /// No check ran: no round, nothing tried or kept.
        fn unchecked(s: GuessStats) -> bool {
            (s.rounds, s.tried, s.kept) == (0, 0, 0)
        }

        /// Bit for bit: the CPU backend computes each row on its own, so a check row equals the
        /// decode row it replaces (Metal needs `StepBatch::check`'s model path for the same).
        fn same(a: &[Out], b: &[Out]) {
            assert_eq!(a, b);
        }

        /// Each request's prompt and reference output.
        fn refs(reqs: &[Request], outs: &[Out]) -> Vec<Vec<u32>> {
            (reqs.iter().zip(outs))
                .map(|(r, o)| [&r.prompt[..], &o.ids[..]].concat())
                .collect()
        }

        /// Repetitive prompts (prompt lookup hits) and random ones (misses), five chats at once,
        /// `max_tokens` 1, 2 and odd lengths.
        fn prompts(seed: u64) -> Vec<Request> {
            let mut z = seed;
            let mut random = |n: usize| -> Vec<u32> {
                (0..n)
                    .map(|_| {
                        z = mix(z);
                        (z % u64::from(tiny::VOCAB)) as u32
                    })
                    .collect()
            };
            let cycle: Vec<u32> = random(7).into_iter().cycle().take(40).collect();
            let mut quoted = random(30);
            quoted.extend_from_within(5..25);
            let reqs = [
                (cycle, 60),
                (quoted, 41),
                (random(23), 37),
                (random(9), 2),
                (random(5), 1),
            ];
            reqs.into_iter().map(|(p, n)| Request::new(p, n)).collect()
        }

        /// Greedy: with guesses, the same tokens, finishes and counts as without, for prompt
        /// lookup, a drafter right to every length and a random one, including stop tokens that
        /// land inside a check.
        #[test]
        fn greedy_output_is_unchanged() {
            for seed in 0..3 {
                let reqs = prompts(seed);
                let (plain, none) = run(Arch::Llama, false, 0, Box::new(PromptLookup), &reqs);
                assert_eq!(none, GuessStats::default());
                let refs = refs(&reqs, &plain);
                let drafters: [Box<dyn Drafter + Send>; 4] = [
                    Box::new(PromptLookup),
                    Box::<SuffixDrafter>::default(),
                    Box::new(Oracle(refs.clone(), seed)),
                    Box::new(Random(seed)),
                ];
                for d in drafters {
                    let (out, stats) = run(Arch::Llama, false, 8, d, &reqs);
                    same(&out, &plain);
                    assert!(stats.rounds > 0, "{stats:?}");
                }
                let (_, stats) = run(
                    Arch::Llama,
                    false,
                    8,
                    Box::new(Oracle(refs.clone(), seed)),
                    &reqs,
                );
                assert!(stats.kept > 0 && stats.kept < stats.tried, "{stats:?}");
                // Stops at tokens the outputs reach mid-way: never a token past them.
                let stopped: Vec<Request> = (reqs.iter().zip(&plain))
                    .map(|(r, o)| {
                        let mut r = r.clone();
                        r.stop_tokens = o.ids.get(o.ids.len() / 2).copied().into_iter().collect();
                        r
                    })
                    .collect();
                let (plain, _) = run(Arch::Llama, false, 0, Box::new(PromptLookup), &stopped);
                assert!(plain.iter().any(|o| o.finish == Finish::Eos));
                let drafters: [Box<dyn Drafter + Send>; 3] = [
                    Box::new(PromptLookup),
                    Box::<SuffixDrafter>::default(),
                    Box::new(Oracle(refs, seed + 9)),
                ];
                for d in drafters {
                    let (out, _) = run(Arch::Llama, false, 8, d, &stopped);
                    same(&out, &plain);
                }
            }
        }

        /// Only a Metal load that may still check, on a pure decode step, takes the check kernels.
        #[test]
        fn only_metal_guessing_decode_runs_as_check() {
            use crate::engine::decodes_as_check;
            use crate::sched::StepPlan;
            let mut decode = StepPlan::default();
            decode.decode.push(1);
            let mut mixed = StepPlan::default();
            (mixed.decode, mixed.prefill) = (vec![1], vec![(2, 0, 4)]);
            assert!(decodes_as_check(true, 8, None, &decode));
            assert!(decodes_as_check(true, 8, Some(true), &decode));
            assert!(!decodes_as_check(false, 8, None, &decode));
            assert!(!decodes_as_check(true, 0, None, &decode));
            assert!(!decodes_as_check(true, 8, Some(false), &decode));
            assert!(!decodes_as_check(true, 8, None, &mixed));
            assert!(!decodes_as_check(true, 8, None, &StepPlan::default()));
        }

        /// On a Metal device, dense, MoE and hybrid MoE: a guessing load's output equals the same
        /// load's with no guess proposed, since every pure decode step takes the check kernels.
        /// LARCH_TEST_DEVICE=0 cargo test --release -p larch-engine --lib -- --ignored
        /// metal_check_steps
        #[test]
        #[ignore = "needs a Metal device: LARCH_TEST_DEVICE"]
        fn metal_check_steps_keep_greedy_output() {
            let device = std::env::var("LARCH_TEST_DEVICE").unwrap().parse().unwrap();
            for arch in [Arch::Llama, Arch::Qwen3Moe, Arch::Qwen35Moe] {
                let reqs = prompts(0);
                let run = |d: Box<dyn Drafter + Send>| {
                    let (mut e, h) = engine_on(device, arch, 8, d, 0, false, None).unwrap();
                    let hs: Vec<_> = reqs.iter().map(|r| h.submit(r.clone()).unwrap()).collect();
                    let out = drive(&mut e, &hs);
                    (out, h.guess_stats(), e.sched.checkable)
                };
                let (plain, none, _) = run(Box::new(Oracle(Vec::new(), 0)));
                assert_eq!(none.tried, 0, "{none:?}");
                let refs = refs(&reqs, &plain);
                let drafters: [Box<dyn Drafter + Send>; 2] =
                    [Box::<SuffixDrafter>::default(), Box::new(Oracle(refs, 1))];
                for d in drafters {
                    let (out, stats, checkable) = run(d);
                    same(&out, &plain);
                    assert!(stats.rounds > 0, "{arch:?} {stats:?}");
                    assert_eq!(checkable, Some(true), "{arch:?}");
                }
            }
        }

        /// An agent turn: a tool result, a reply quoting part of it, then the prompt quoting it
        /// again up to where the model goes on.
        fn transcript(seed: u64) -> Request {
            let mut z = seed + 100;
            let mut random = |n: usize| -> Vec<u32> {
                (0..n)
                    .map(|_| {
                        z = mix(z);
                        (z % u64::from(tiny::VOCAB)) as u32
                    })
                    .collect()
            };
            let mut t = random(12);
            let tool = random(40);
            t.extend(&tool);
            t.extend(random(5));
            t.extend(&tool[8..30]);
            t.extend(random(6));
            t.extend(&tool[..20]);
            Request::new(t, 48)
        }

        /// An agent loop with suffix drafts: the prompts and a transcript quoting a tool result,
        /// then the same requests again as a retried step: greedy output as without guesses, and
        /// more guesses kept than prompt lookup keeps, more a check step.
        #[test]
        fn suffix_drafts_keep_output_and_beat_prompt_lookup() {
            for seed in 0..3 {
                let mut reqs = prompts(seed);
                reqs.push(transcript(seed));
                let agent = |guesses: u32, d: Box<dyn Drafter + Send>| {
                    let (mut e, h) = engine(Arch::Llama, guesses, d, 64, false);
                    let mut outs = Vec::new();
                    for _ in 0..2 {
                        let hs: Vec<_> =
                            reqs.iter().map(|r| h.submit(r.clone()).unwrap()).collect();
                        outs.extend(drive(&mut e, &hs));
                    }
                    (outs, h.guess_stats())
                };
                let (plain, _) = agent(0, Box::new(PromptLookup));
                let (lookup, l) = agent(8, Box::new(PromptLookup));
                let (suffix, s) = agent(8, Box::<SuffixDrafter>::default());
                same(&lookup, &plain);
                same(&suffix, &plain);
                // More kept, and more a check step (seeds 0-2: 2.1-2.4x a step).
                assert!(s.kept > l.kept, "{l:?} {s:?}");
                assert!(2 * s.kept * l.rounds > 3 * l.kept * s.rounds, "{l:?} {s:?}");
            }
        }

        /// A guessed turn parks and caches only the tokens it kept: the next turn of its session,
        /// and a stateless request the global cache serves, reuse the same prefix and produce the
        /// same tokens as without guesses.
        #[test]
        fn caches_reuse_a_guessed_turn() {
            let reqs = prompts(5);
            let outs = |guesses: u32| {
                let (mut e, h) = engine(Arch::Llama, guesses, Box::new(PromptLookup), 64, false);
                let mut first = reqs[0].clone();
                (first.session, first.keep_kv) = (7, true);
                let a = drive(&mut e, &[h.submit(first.clone()).unwrap()]).remove(0);
                let mut next = first.clone();
                next.prompt.extend(&a.ids);
                next.prompt.extend([3, 4, 5]);
                let b = drive(&mut e, &[h.submit(next).unwrap()]).remove(0);
                let c = drive(&mut e, &[h.submit(reqs[1].clone()).unwrap()]).remove(0);
                let d = drive(&mut e, &[h.submit(reqs[1].clone()).unwrap()]).remove(0);
                (vec![a, b, c, d], h.guess_stats())
            };
            let ((plain, _), (guessed, stats)) = (outs(0), outs(8));
            same(&guessed, &plain);
            assert!(stats.kept > 0, "{stats:?}");
            assert!(guessed[1].usage.cached_tokens > reqs[0].prompt.len() as u32);
            assert!(guessed[3].usage.cached_tokens > 0);
        }

        /// A recurrent model checks with its state rewound to the kept rows (one check a step),
        /// and a backend that refuses check steps decodes plainly, recurrent or not: either way
        /// the output is unchanged.
        #[test]
        fn recurrent_models_rewind_and_refused_checks_decode() {
            for seed in 0..2 {
                let reqs = prompts(seed);
                let (plain, _) = run(Arch::Qwen35, false, 0, Box::new(PromptLookup), &reqs);
                let refs = refs(&reqs, &plain);
                let (out, stats) = run(
                    Arch::Qwen35,
                    false,
                    8,
                    Box::new(Oracle(refs.clone(), seed)),
                    &reqs,
                );
                same(&out, &plain);
                assert!(stats.kept > 0 && stats.kept < stats.tried, "{stats:?}");
                let (out, stats) = run(Arch::Qwen35, true, 8, Box::new(Oracle(refs, seed)), &reqs);
                same(&out, &plain);
                assert!(unchecked(stats), "{stats:?}");
                let (plain, _) = run(Arch::Llama, false, 0, Box::new(PromptLookup), &reqs);
                let (out, stats) = run(Arch::Llama, true, 8, Box::new(PromptLookup), &reqs);
                same(&out, &plain);
                assert!(unchecked(stats), "{stats:?}");
            }
        }

        /// A slow drafter: `Right`'s guess of token 7, after a device forward's wait.
        struct Slow;

        const DRAFT_NS: u64 = 50_000_000;

        impl Drafter for Slow {
            fn propose(&mut self, _: u64, _: Namespace, _: &[u32], max: usize, out: &mut Vec<u32>) {
                std::thread::sleep(std::time::Duration::from_nanos(DRAFT_NS));
                out.extend(std::iter::repeat_n(7, max));
            }
        }

        /// A DFlash check runs its draft forward during scheduling, outside the step's own time:
        /// the check's bucket holds it, the plain decode bucket does not.
        #[test]
        fn a_checks_draft_time_is_in_its_cost() {
            let (mut e, h) = engine(Arch::Llama, 8, Box::new(Slow), 0, false);
            e.sched.costs = Costs::default();
            let hs = vec![h.submit(Request::new(vec![1, 2, 3], 24)).unwrap()];
            drive(&mut e, &hs);
            let (check, checks) = e.sched.costs.sample(1, 8);
            let (plain, plains) = e.sched.costs.sample(1, 0);
            assert!(checks > 0 && plains > 0, "{checks} {plains}");
            assert!(check >= DRAFT_NS as f64, "{check}");
            assert!(plain < DRAFT_NS as f64, "{plain}");
        }

        /// Draft length from the step cost, the output unchanged throughout. Where a check costs
        /// its rows (as on the CPU), every sequence decodes plainly: nothing tried. Where rows are nearly free, a drafter that is always right checks the most guesses
        /// allowed and one that misses checks few; a recurrent model's checks stay within its
        /// spare state slots.
        #[test]
        fn draft_length_follows_the_step_cost() {
            let reqs = prompts(0);
            for arch in [Arch::Llama, Arch::Qwen35] {
                let (plain, _) = run(arch, false, 0, Box::new(PromptLookup), &reqs);
                let refs = refs(&reqs, &plain);
                let rows = Some(Costs::fixed(|d, g| (d + g) as f64));
                let right = || Box::new(Right(refs.clone()));
                let (out, s) = run_costs(arch, false, 8, right(), rows, &reqs);
                same(&out, &plain);
                assert!(
                    unchecked(s) && s.k[0] > 0 && s.k[0] == s.k.iter().sum::<u64>(),
                    "{s:?}"
                );
                let cheap = || Some(Costs::fixed(|d, g| 100.0 + (d + g) as f64));
                let (out, hit) = run_costs(arch, false, 8, right(), cheap(), &reqs);
                same(&out, &plain);
                let (out, miss) = run_costs(arch, false, 8, Box::new(Random(1)), cheap(), &reqs);
                same(&out, &plain);
                let top = if arch.hybrid() { 7 } else { 8 };
                let most = (1..=top).max_by_key(|&k| (hit.k[k], k)).unwrap();
                assert_eq!(most, top, "{arch:?} {hit:?}");
                assert!(hit.k[top + 1..].iter().all(|&n| n == 0), "{arch:?} {hit:?}");
                let mean = |s: GuessStats| {
                    let n: u64 = s.k.iter().sum();
                    (s.k.iter().enumerate())
                        .map(|(k, &c)| k as u64 * c)
                        .sum::<u64>() as f64
                        / n as f64
                };
                assert!(
                    miss.k[top] == 0 && mean(miss) * 2.0 < mean(hit),
                    "{miss:?} {hit:?}"
                );
                assert!(miss.rounds > 0 && hit.kept == hit.tried, "{miss:?} {hit:?}");
            }
        }

        /// A session that parks on a rewound check gives its spare back: another sequence then
        /// checks as often as with nothing parked, on guesses + 1 spares.
        #[test]
        fn a_session_parked_on_a_rewind_leaves_the_spares() {
            let a = Request::new(vec![1, 2, 3, 4, 5], 8);
            let (plain, _) = run(
                Arch::Qwen35,
                false,
                0,
                Box::new(PromptLookup),
                std::slice::from_ref(&a),
            );
            let ids = &plain[0].ids;
            assert_ne!(ids[0], ids[1], "the stop must land on the first check");
            let rounds = |parked: bool| {
                let (mut e, h) = engine(Arch::Qwen35, 2, Box::new(Random(3)), 0, false);
                if parked {
                    // Its first check samples the stop at row 0 and rewinds both guesses: it
                    // parks with its state in a spare.
                    let mut first = a.clone();
                    (first.session, first.keep_kv) = (7, true);
                    first.stop_tokens = std::iter::once(ids[1]).collect();
                    let out = drive(&mut e, &[h.submit(first).unwrap()]).remove(0);
                    assert_eq!(out.finish, Finish::Eos);
                    assert_eq!(e.sched.n_parked(), 1);
                    let stats = h.guess_stats();
                    assert_eq!((stats.rounds, stats.tried, stats.kept), (1, 2, 0));
                }
                let before = h.guess_stats();
                drive(
                    &mut e,
                    &[h.submit(Request::new(vec![6, 7, 8], 12)).unwrap()],
                );
                let after = h.guess_stats();
                assert!(
                    after.tried - before.tried > after.kept - before.kept,
                    "rewinds"
                );
                after.rounds - before.rounds
            };
            let alone = rounds(false);
            assert!(alone > 2, "{alone}");
            assert_eq!(rounds(true), alone);
        }

        /// Sampled: the joint distribution of the two tokens after the first is the model's own
        /// when a check guesses the second, under a strong repeat penalty, so row 1's penalty
        /// window must hold the guess. Exact seeds make it deterministic; 3000 draws a side put
        /// the total variation's noise near 0.03.
        #[test]
        fn sampled_output_keeps_the_distribution() {
            let n = 3000u64;
            let sampling = SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 4,
                repeat_penalty: 2.0,
                seed: None,
            };
            let joint = |guesses: u32, guess: u32| {
                let (mut e, h) = engine(Arch::Llama, guesses, Box::new(Constant(guess)), 0, false);
                let mut counts = HashMap::new();
                for batch in (0..n).collect::<Vec<u64>>().chunks(8) {
                    let hs: Vec<_> = batch
                        .iter()
                        .map(|&i| {
                            let mut r = Request::new(vec![11, 12, 13], 3);
                            r.sampling = SamplingParams {
                                seed: Some(i),
                                ..sampling
                            };
                            h.submit(r).unwrap()
                        })
                        .collect();
                    for o in drive(&mut e, &hs) {
                        *counts.entry((o.ids[1], o.ids[2])).or_insert(0u32) += 1;
                    }
                }
                (counts, h.guess_stats())
            };
            let (plain, _) = joint(0, 0);
            // Guess the likeliest second token, so most checks keep it and sample row 1.
            let mut second = HashMap::new();
            for (&(a, _), &c) in &plain {
                *second.entry(a).or_insert(0) += c;
            }
            let likeliest = *second.iter().max_by_key(|e| e.1).unwrap().0;
            for guess in [likeliest, (likeliest + 1) % tiny::VOCAB] {
                let (checked, stats) = joint(8, guess);
                assert_eq!(stats.rounds, n, "{stats:?}");
                let keys: HashSet<_> = plain.keys().chain(checked.keys()).collect();
                let freq = |m: &HashMap<_, u32>, k| f64::from(*m.get(k).unwrap_or(&0)) / n as f64;
                let tv: f64 = keys
                    .into_iter()
                    .map(|k| (freq(&plain, k) - freq(&checked, k)).abs())
                    .sum::<f64>()
                    / 2.0;
                assert!(tv < 0.07, "total variation {tv} for guess {guess}");
            }
        }
        fn run_dflash(arch: Arch, cap: u32, reqs: &[Request]) -> (Vec<Out>, GuessStats) {
            run_draft(arch, Drafted::Dflash(cap), reqs)
        }

        fn run_draft(arch: Arch, draft: Drafted, reqs: &[Request]) -> (Vec<Out>, GuessStats) {
            let (mut e, h) = engine_with(arch, 0, Box::new(PromptLookup), 0, false, Some(draft));
            let hs: Vec<_> = reqs.iter().map(|r| h.submit(r.clone()).unwrap()).collect();
            let outs = drive(&mut e, &hs);
            (outs, h.guess_stats())
        }

        /// Every seated sequence's draft context holds exactly its KV positions.
        fn no_drift(e: &Engine) -> usize {
            let draft = e.loaded.as_ref().unwrap().draft.as_ref().unwrap();
            for (id, len) in draft.live() {
                let r = e.sched.req(id).expect("a live seat's request runs");
                assert_eq!(len as usize, r.seq.n_kv, "request {id}");
            }
            draft.live().len()
        }

        /// A DFlash drafter, at its full block and capped at 4 guesses, on a dense model and the
        /// hybrid targets (dense and MoE; their taps include Gated DeltaNet layers),
        /// five chats at once with stop tokens: the same output as without guesses. The tiny
        /// random drafter mostly misses, so the low hit rate switch turns it off and on.
        #[test]
        fn dflash_greedy_output_is_unchanged() {
            let cases = [
                (Arch::Llama, 0),
                (Arch::Llama, 1),
                (Arch::Qwen35, 2),
                (Arch::Qwen35Moe, 3),
            ];
            for (arch, seed) in cases {
                let mut reqs = prompts(seed);
                reqs.push(Request::new(prompts(seed + 7)[2].prompt.clone(), 200));
                let (plain, _) = run(arch, false, 0, Box::new(PromptLookup), &reqs);
                // The hybrid model's 8 spare state slots bound its checks to 7 guesses.
                for cap in if arch.hybrid() { [4, 8] } else { [0, 4] } {
                    let (out, stats) = run_dflash(arch, cap, &reqs);
                    same(&out, &plain);
                    assert!(
                        stats.rounds > 0 && stats.kept < stats.tried,
                        "{arch:?} {seed} {cap} {stats:?}"
                    );
                    // The switch: the 200-token chat ran fewer rounds than tokens past its first.
                    assert!(stats.rounds < 199, "{stats:?}");
                }
                let stopped: Vec<Request> = (reqs.iter().zip(&plain))
                    .map(|(r, o)| {
                        let mut r = r.clone();
                        r.stop_tokens = o.ids.get(o.ids.len() / 2).copied().into_iter().collect();
                        r
                    })
                    .collect();
                let (plain, _) = run(arch, false, 0, Box::new(PromptLookup), &stopped);
                let (out, _) = run_dflash(arch, 4, &stopped);
                same(&out, &plain);
            }
        }

        /// A DFlash2 drafter (its convs, the ring of its window wrapped by the 200-token chat, its
        /// selector) at its whole block on the dense and the MoE hybrid targets: the same output as
        /// without guesses, with rounds checked.
        #[test]
        fn dflash2_greedy_output_is_unchanged() {
            for (arch, seed) in [(Arch::Llama, 0), (Arch::Qwen35Moe, 3)] {
                let mut reqs = prompts(seed);
                reqs.push(Request::new(prompts(seed + 7)[2].prompt.clone(), 200));
                let (plain, _) = run(arch, false, 0, Box::new(PromptLookup), &reqs);
                let (out, stats) = run_draft(arch, Drafted::Dflash2(7), &reqs);
                same(&out, &plain);
                assert!(stats.rounds > 0, "{arch:?} {stats:?}");
            }
        }

        /// [`engine_with`]'s tiny Llama on a faked graph-capable `backend` that never guesses, so
        /// every decode step is a plain one; the CPU stream refuses each capture and runs it eager.
        fn graph_engine(backend: Backend, dflash: Option<u32>, on: bool) -> (Engine, EngineHandle) {
            let draft = dflash.map(Drafted::Dflash);
            let (mut e, h) = engine_with(Arch::Llama, 0, Box::new(PromptLookup), 0, false, draft);
            e.sched.guess_max = 0;
            let loaded = e.loaded.as_mut().unwrap();
            (loaded.info.caps.backend, loaded.info.caps.graph_replay) = (backend, true);
            loaded.graph_bytes = 64 * GRAPH_BYTES;
            e.cfg.values[knob_id::DECODE_GRAPHS as usize] = f64::from(u8::from(on));
            (e, h)
        }

        /// A drafted decode step keys its bucket with the taps flag, an undrafted one without it;
        /// a drafted step takes no key off CUDA, and neither shape shares a key with the other.
        #[test]
        fn drafted_decode_steps_key_with_the_taps_flag() {
            for (backend, dflash, taps) in [
                (Backend::Cuda, Some(0), Some(true)),
                (Backend::Cuda, None, Some(false)),
                (Backend::Hip, Some(0), None),
                (Backend::Hip, None, Some(false)),
            ] {
                let (mut e, h) = graph_engine(backend, dflash, true);
                let _r = h.submit(Request::new(vec![5; 4], 8)).unwrap();
                for _ in 0..3 {
                    assert!(e.turn().unwrap());
                }
                let graphs = &e.loaded.as_ref().unwrap().graphs;
                let keys: Vec<_> = graphs
                    .keys()
                    .map(|k| (k.decode, k.prefill, k.taps))
                    .collect();
                let want: Vec<_> = taps.into_iter().map(|t| (1, 0, t)).collect();
                assert_eq!(keys, want, "{backend:?} {dflash:?}");
            }
        }

        /// A mixed shape stays eager under a drafter: no second-hit `Seen` entry for its key.
        #[test]
        fn a_drafted_mixed_step_takes_no_key() {
            for dflash in [None, Some(0)] {
                let (mut e, h) = graph_engine(Backend::Cuda, dflash, true);
                let loaded = e.loaded.as_mut().unwrap();
                loaded.mixed_graphs = true;
                e.cfg.values[knob_id::PREFILL_CHUNK_TOKENS as usize] = 64.0;
                let _decoding: Vec<_> = (0..7)
                    .map(|i| h.submit(Request::new(vec![i + 1], 32)).unwrap())
                    .collect();
                assert!(e.turn().unwrap());
                let _joins = h.submit(Request::new(vec![11; 64], 1)).unwrap();
                assert!(e.turn().unwrap());
                let graphs = &e.loaded.as_ref().unwrap().graphs;
                let mixed = graphs.keys().any(|k| k.prefill != 0);
                assert_eq!(mixed, dflash.is_none(), "{dflash:?}");
            }
        }

        /// Dropping the drafter frees the taps' buffers, so its graphs could never launch: their
        /// keys give back their quota, plain ones stay, and later keys carry no taps.
        #[test]
        fn a_dropped_drafter_purges_its_taps_keys() {
            let (mut e, h) = graph_engine(Backend::Cuda, Some(0), true);
            e.sched.checkable = Some(false);
            let splits = e.attn_splits();
            let tapped = GraphKey {
                taps: true,
                ..decode_key(4, splits)
            };
            let plain = decode_key(4, splits);
            let refused = Capture::Refused(Refusal {
                free: u64::MAX,
                arena: false,
                step: 0,
                again: false,
            });
            let loaded = e.loaded.as_mut().unwrap();
            loaded.graphs.insert(tapped, refused);
            loaded.graphs.insert(plain, Capture::Seen);
            let _r = h.submit(Request::new(vec![5; 4], 8)).unwrap();
            for _ in 0..3 {
                assert!(e.turn().unwrap());
            }
            let loaded = e.loaded.as_ref().unwrap();
            assert!(loaded.draft.is_none());
            assert!(loaded.graphs.contains_key(&plain));
            assert!(
                !loaded.graphs.keys().any(|k| k.taps),
                "{:?}",
                loaded.graphs.keys()
            );
        }

        /// [`graph_engine`] guessing one token each step: a check of 2 rows whenever a sequence
        /// decodes alone.
        fn check_engine(on: bool, dflash: Option<u32>) -> (Engine, EngineHandle) {
            let (mut e, h) = graph_engine(Backend::Cuda, dflash, on);
            e.sched.guess_max = 4;
            if dflash.is_none() {
                e.sched.drafter = Box::new(Constant(1));
            }
            (e, h)
        }

        /// A plain bucket takes its quota from a check capture: a device run, since only a real
        /// capture holds quota. LARCH_TEST_DEVICE=<ordinal> cargo test --release -p larch-engine
        /// --lib -- --ignored a_plain_bucket_evicts
        #[test]
        #[ignore = "needs a device that captures graphs: LARCH_TEST_DEVICE"]
        fn a_plain_bucket_evicts_a_check_capture() {
            let device = std::env::var("LARCH_TEST_DEVICE").unwrap().parse().unwrap();
            let c = Box::new(Constant(1));
            let (mut e, h) = engine_on(device, Arch::Llama, 0, c, 0, false, None).unwrap();
            e.sched.guess_max = 4;
            let loaded = e.loaded.as_mut().unwrap();
            loaded.graph_bytes = 4 * GRAPH_BYTES;
            e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 1.0;
            let _r = h.submit(Request::new(vec![5; 4], 64)).unwrap();
            let held = (0..8).find_map(|_| {
                assert!(e.turn().unwrap());
                let graphs = &e.loaded.as_ref().unwrap().graphs;
                graphs.iter().find_map(|(k, c)| match c {
                    Capture::Graph(.., bytes) if k.checks != 0 => Some(*bytes),
                    _ => None,
                })
            });
            let held = held.expect("no check capture in 8 steps");
            // The quota is exactly the check capture: the next plain bucket must evict it.
            let splits = e.attn_splits();
            let loaded = e.loaded.as_mut().unwrap();
            loaded.graphs.retain(|k, _| k.checks != 0);
            loaded.graph_bytes = held;
            e.sched.drafter = Box::new(Oracle(Vec::new(), 0));
            for _ in 0..4 {
                assert!(e.turn().unwrap());
            }
            let graphs = &e.loaded.as_ref().unwrap().graphs;
            assert!(
                matches!(graphs.get(&decode_key(1, splits)), Some(Capture::Graph(..))),
                "the plain bucket did not capture"
            );
            assert!(
                !graphs
                    .iter()
                    .any(|(k, c)| k.checks != 0 && matches!(c, Capture::Graph(..))),
                "the check capture stayed"
            );
        }

        /// Check captures go least recently replayed first, only as many as the need takes.
        #[test]
        fn check_evictions_go_oldest_first_and_stop_when_the_key_fits() {
            let k = |decode| GraphKey {
                decode,
                prefill: 2,
                logits: 2,
                splits: 16,
                taps: false,
                checks: 1,
            };
            let g = GRAPH_BYTES;
            let held = vec![(9, k(1), g), (3, k(2), g), (5, k(3), g)];
            assert_eq!(evict_lru(held.clone(), 0, g), (vec![k(2)], g));
            assert_eq!(
                evict_lru(held.clone(), g / 2, 2 * g),
                (vec![k(2), k(3)], g / 2 + 2 * g)
            );
            assert_eq!(evict_lru(held.clone(), g, g), (vec![], g));
            assert_eq!(
                evict_lru(held, 0, 9 * g).0.len(),
                3,
                "all of them, if that is short"
            );
        }

        /// A check key is its exact shape, on CUDA only.
        #[test]
        fn a_check_key_is_its_exact_shape_on_cuda_only() {
            let k = GraphKey {
                decode: 2,
                prefill: 7,
                logits: 9,
                splits: 16,
                taps: true,
                checks: 3,
            };
            assert_eq!(check_key(Backend::Cuda, 2, 7, 3, 9, 16, true), Some(k));
            for b in [Backend::Hip, Backend::Metal, Backend::Cpu] {
                assert_eq!(check_key(b, 2, 7, 3, 9, 16, true), None);
            }
        }

        /// A check shape is `Seen` on its first step, attempted on its second (refused on this
        /// stream, as a failed capture is) and eager after that, surviving the step's stale-key
        /// retain; a drafter's check keys carry the taps flag.
        #[test]
        fn a_check_shape_admits_on_its_second_hit_and_stays_refused() {
            for dflash in [None, Some(4)] {
                let (mut e, h) = check_engine(true, dflash);
                let _r = h.submit(Request::new(vec![5; 4], 16)).unwrap();
                let mut seen = Vec::new();
                for _ in 0..8 {
                    assert!(e.turn().unwrap());
                    let graphs = &e.loaded.as_ref().unwrap().graphs;
                    let mut checks: Vec<_> = graphs.iter().filter(|(k, _)| k.checks != 0).collect();
                    assert!(checks.len() <= 1, "one shape");
                    let Some((k, c)) = checks.pop() else { continue };
                    assert_eq!(
                        (k.decode, k.checks, k.taps),
                        (0, 1, dflash.is_some()),
                        "{k:?}"
                    );
                    assert_eq!(k.prefill, k.logits, "{k:?}");
                    seen.push(matches!(c, Capture::Seen));
                }
                assert!(seen.len() >= 3, "{dflash:?}: {seen:?}");
                assert!(seen[0], "first hit: {seen:?}");
                assert!(!seen[1..].iter().any(|&s| s), "then refused: {seen:?}");
            }
        }

        /// A refused check capture runs eager with no error: output and guess stats as graphs off.
        #[test]
        fn refused_check_captures_fall_back_to_eager() {
            let reqs = prompts(3);
            let out = |on| {
                let (mut e, h) = check_engine(on, None);
                let hs: Vec<_> = reqs.iter().map(|r| h.submit(r.clone()).unwrap()).collect();
                (drive(&mut e, &hs), h.guess_stats())
            };
            let ((on, on_stats), (off, off_stats)) = (out(true), out(false));
            same(&on, &off);
            assert_eq!(on_stats, off_stats);
            assert!(on_stats.rounds > 0, "{on_stats:?}");
        }

        /// A guessing hybrid keys its decode and check shapes on CUDA, not elsewhere, and a capture
        /// this stream refuses leaves the step eager with the output of graphs off.
        #[test]
        fn a_guessing_hybrid_keys_on_cuda_only() {
            let engine = |backend, on| {
                let (mut e, h) = engine(Arch::Qwen35, 4, Box::new(Constant(1)), 0, false);
                let loaded = e.loaded.as_mut().unwrap();
                (loaded.info.caps.backend, loaded.info.caps.graph_replay) = (backend, true);
                loaded.graph_bytes = 64 * GRAPH_BYTES;
                e.cfg.values[knob_id::DECODE_GRAPHS as usize] = f64::from(u8::from(on));
                (e, h)
            };
            let reqs = prompts(3);
            let out = |backend, on| {
                let (mut e, h) = engine(backend, on);
                let hs: Vec<_> = reqs.iter().map(|r| h.submit(r.clone()).unwrap()).collect();
                let outs = drive(&mut e, &hs);
                let keys: Vec<GraphKey> =
                    e.loaded.as_ref().unwrap().graphs.keys().copied().collect();
                (outs, h.guess_stats(), keys)
            };
            let (on, on_stats, keys) = out(Backend::Cuda, true);
            let (off, off_stats, none) = out(Backend::Cuda, false);
            same(&on, &off);
            assert_eq!(on_stats, off_stats);
            assert!(on_stats.rounds > 0, "{on_stats:?}");
            assert!(none.is_empty(), "{none:?}");
            assert!(keys.iter().any(|k| k.checks != 0), "{keys:?}");
            assert!(keys.iter().any(|k| k.prefill == 0), "{keys:?}");
            let (_, _, hip) = out(Backend::Hip, true);
            assert!(hip.is_empty(), "{hip:?}");
        }

        /// [`crate::guess::Draft::guess_logits`]'s seats.
        type SeatLogits = Vec<(u64, u32, Vec<f32>)>;

        /// Three sequences' drafted decode steps, each attempted (and on this stream refused) at
        /// the padded bucket of 4 with graphs `on`, at their exact 3 rows without. The batch rows
        /// of the taps at each decode step, and each drafted sequence's guess logits at the end.
        fn drafted_rows(on: bool) -> (Vec<usize>, SeatLogits) {
            let (mut e, h) = graph_engine(Backend::Cuda, Some(0), on);
            let _rs: Vec<_> = (1..4)
                .map(|i| h.submit(Request::new(vec![i; 4], 8)).unwrap())
                .collect();
            let mut rows = Vec::new();
            for _ in 0..5 {
                // Forget refusals so that each step attempts the shape a replay runs.
                e.loaded.as_mut().unwrap().graphs.clear();
                assert!(e.turn().unwrap());
                rows.push(e.loaded.as_ref().unwrap().staging.tap_rows);
            }
            let loaded = e.loaded.as_ref().unwrap();
            let draft = loaded.draft.as_ref().unwrap();
            (rows, draft.guess_logits(&e.s, &*loaded.model))
        }

        /// A padded step's tap blocks start at `k * (live + pad)`, so the drafter reads each live
        /// row's own taps: its context is the unpadded step's, bit for bit.
        #[test]
        fn a_padded_step_feeds_the_drafter_its_own_rows() {
            let (padded, on) = drafted_rows(true);
            let (exact, off) = drafted_rows(false);
            assert_eq!(padded[1..], [4, 4, 4, 4], "{padded:?}");
            assert_eq!(exact[1..], [3, 3, 3, 3], "{exact:?}");
            assert_eq!(on.len(), 3);
            assert!(on.iter().all(|c| c.1 > 4), "{on:?}");
            assert_eq!(on, off);
        }

        /// A DFlash drafter on a model that refuses check steps, dense and hybrid: the step after
        /// the first refusal drops the drafter,
        /// so no later step taps or feeds it and none keeps decode from graphs, and the output is
        /// plain decode's.
        #[test]
        fn a_refused_check_drops_the_drafter() {
            for arch in [Arch::Llama, Arch::Qwen35, Arch::Qwen35Moe] {
                let reqs = prompts(4);
                let (plain, _) = run(arch, false, 0, Box::new(PromptLookup), &reqs);
                let (mut e, h) = engine_with(
                    arch,
                    0,
                    Box::new(PromptLookup),
                    0,
                    true,
                    Some(Drafted::Dflash(0)),
                );
                assert!(e.loaded.as_ref().unwrap().draft.is_some());
                let hs: Vec<_> = reqs.iter().map(|r| h.submit(r.clone()).unwrap()).collect();
                let out = drive(&mut e, &hs);
                same(&out, &plain);
                assert!(unchecked(h.guess_stats()), "{:?}", h.guess_stats());
                assert_eq!(
                    e.sched.checkable,
                    Some(false),
                    "{arch:?}: no check was refused"
                );
                let loaded = e.loaded.as_ref().unwrap();
                assert!(
                    loaded.draft.is_none() && !loaded.staging.drafted,
                    "{arch:?}"
                );
            }
        }

        /// After every step, rejected rows included, each seated sequence's draft context holds
        /// its committed positions, no more and no fewer: dense, and hybrid, where a check's
        /// rewound rows are taps of the recurrent layers that the drafter must not keep.
        #[test]
        fn dflash_observes_exactly_the_committed_positions() {
            for arch in [Arch::Llama, Arch::Qwen35, Arch::Qwen35Moe] {
                let reqs = prompts(3);
                let draft = Some(Drafted::Dflash(0));
                let (mut e, h) = engine_with(arch, 0, Box::new(PromptLookup), 0, false, draft);
                let _hs: Vec<_> = reqs.iter().map(|r| h.submit(r.clone()).unwrap()).collect();
                let mut seated = 0;
                loop {
                    assert!(e.turn().unwrap());
                    seated = seated.max(no_drift(&e));
                    if e.sched.is_idle() {
                        break;
                    }
                }
                let stats = h.guess_stats();
                assert!(
                    seated >= 3 && stats.kept < stats.tried,
                    "{arch:?} {seated} {stats:?}"
                );
            }
        }

        /// A session's next turn takes over its parked draft context, cut back to the KV it
        /// reuses, and decodes the same tokens as without guesses.
        #[test]
        fn dflash_reuses_a_parked_session() {
            let reqs = prompts(5);
            let outs = |dflash: Option<u32>| {
                let (mut e, h) = engine_with(
                    Arch::Llama,
                    0,
                    Box::new(PromptLookup),
                    64,
                    false,
                    dflash.map(Drafted::Dflash),
                );
                let mut first = reqs[0].clone();
                (first.session, first.keep_kv) = (7, true);
                let a = drive(&mut e, &[h.submit(first.clone()).unwrap()]).remove(0);
                let mut next = first.clone();
                next.prompt.extend(&a.ids);
                next.prompt.extend([3, 4, 5]);
                let hb = h.submit(next).unwrap();
                if dflash.is_some() {
                    let draft = e.loaded.as_ref().unwrap().draft.as_ref().unwrap();
                    assert_eq!(draft.parked(), [7]);
                    // The first step feeds the turn past its reused KV, onto the parked context.
                    assert!(e.turn().unwrap());
                    let draft = e.loaded.as_ref().unwrap().draft.as_ref().unwrap();
                    let live = draft.live();
                    assert!(draft.parked().is_empty());
                    assert!(
                        live.len() == 1 && live[0].1 as usize > reqs[0].prompt.len(),
                        "{live:?}"
                    );
                    no_drift(&e);
                }
                let b = drive(&mut e, &[hb]).remove(0);
                (vec![a, b], h.guess_stats())
            };
            let ((plain, _), (guessed, stats)) = (outs(None), outs(Some(0)));
            same(&guessed, &plain);
            assert!(stats.rounds > 0, "{stats:?}");
        }

        /// Spare state slots cost device memory, so only a load that will guess has them: without
        /// guessing a check has none to snapshot into.
        #[test]
        fn spare_state_slots_come_with_guessing() {
            for (guesses, spares) in [(0, false), (8, true)] {
                let (mut e, _h) = engine(Arch::Qwen35, guesses, Box::new(PromptLookup), 0, false);
                let model = &mut e.loaded.as_mut().unwrap().model;
                assert_eq!(model.begin_check(&e.s, 0, 2).is_ok(), spares, "{guesses}");
            }
        }

        /// A guessing hybrid holds a spare state slot for each guess its checks try and one for
        /// the sequence living in a spare, never more than [`Transformer::CHECK_SLOTS`]: 3 guesses
        /// take 4 rows' snapshots (a check of 5 rows), not 5.
        #[test]
        fn a_guessing_hybrid_holds_a_spare_a_guess_and_one() {
            for (guesses, spares) in [(3, 4), (8, Transformer::CHECK_SLOTS)] {
                let (mut e, _h) = engine(Arch::Qwen35, guesses, Box::new(PromptLookup), 0, false);
                let model = &mut e.loaded.as_mut().unwrap().model;
                assert_eq!(model.begin_check(&e.s, 0, spares + 1), Ok(()), "{guesses}");
                model.end_check(&e.s, 0, 1).unwrap_err();
                let more = model.begin_check(&e.s, 0, spares + 2);
                assert_eq!(more, Err(KError::Unsupported), "{guesses}");
            }
        }

        /// Guessing pins the attention splits only while checks can run: once a check was refused,
        /// decode is back on the backend's pick.
        #[test]
        fn a_refused_check_unpins_the_splits() {
            let (mut e, h) = engine(Arch::Llama, 8, Box::new(PromptLookup), 0, true);
            assert_eq!(e.attn_splits(), 16);
            let hs: Vec<_> = (prompts(4).into_iter())
                .map(|r| h.submit(r).unwrap())
                .collect();
            drive(&mut e, &hs);
            assert_eq!(e.sched.checkable, Some(false));
            assert_eq!(e.attn_splits(), 0);
        }

        /// A drafter's full block on a recurrent model stays within its spares, so the first check
        /// is not refused for room and guessing goes on, with the output unchanged.
        #[test]
        fn a_full_block_on_a_recurrent_model_keeps_guessing() {
            let reqs = prompts(4);
            let (plain, _) = run(Arch::Qwen35, false, 0, Box::new(PromptLookup), &reqs);
            let (out, stats) = run_dflash(Arch::Qwen35, 0, &reqs);
            same(&out, &plain);
            assert!(stats.rounds > 0, "{stats:?}");
        }

        /// The model's own MTP block as the drafter, on the dense and the MoE Qwen 3.5, at its
        /// default chain and capped at 4 and 7 guesses, five chats at once, then with stop tokens:
        /// the same output as the model without the block decodes without guesses.
        #[test]
        fn mtp_greedy_output_is_unchanged() {
            for (arch, seed) in [(Arch::Qwen35, 0), (Arch::Qwen35Moe, 1)] {
                let mut reqs = prompts(seed);
                reqs.push(Request::new(prompts(seed + 7)[2].prompt.clone(), 200));
                let (plain, _) = run(arch, false, 0, Box::new(PromptLookup), &reqs);
                for cap in [0, 4, 7] {
                    let (out, stats) = run_draft(arch, Drafted::Mtp(cap), &reqs);
                    same(&out, &plain);
                    assert!(stats.rounds > 0, "{arch:?} {cap} {stats:?}");
                }
                let stopped: Vec<Request> = (reqs.iter().zip(&plain))
                    .map(|(r, o)| {
                        let mut r = r.clone();
                        r.stop_tokens = o.ids.get(o.ids.len() / 2).copied().into_iter().collect();
                        r
                    })
                    .collect();
                let (plain, _) = run(arch, false, 0, Box::new(PromptLookup), &stopped);
                let (out, _) = run_draft(arch, Drafted::Mtp(4), &stopped);
                same(&out, &plain);
            }
        }

        /// After every step, rejected rows included, each seated sequence's MTP context holds its
        /// committed positions, no more and no fewer.
        #[test]
        fn mtp_observes_exactly_the_committed_positions() {
            let reqs = prompts(3);
            let draft = Some(Drafted::Mtp(4));
            let (mut e, h) =
                engine_with(Arch::Qwen35Moe, 0, Box::new(PromptLookup), 0, false, draft);
            let _hs: Vec<_> = reqs.iter().map(|r| h.submit(r.clone()).unwrap()).collect();
            let mut seated = 0;
            loop {
                assert!(e.turn().unwrap());
                seated = seated.max(no_drift(&e));
                if e.sched.is_idle() {
                    break;
                }
            }
            let stats = h.guess_stats();
            assert!(seated >= 3 && stats.rounds > 0, "{seated} {stats:?}");
        }

        /// A model without an MTP block refuses the MTP drafter at load.
        #[test]
        fn mtp_needs_the_block() {
            let draft = Some(Drafted::Mtp(0));
            let got = try_engine(Arch::Llama, 0, Box::new(PromptLookup), 0, false, draft);
            assert!(matches!(got, Err(LoadError::UnsupportedArch)));
        }

        /// A recurrent model that guesses takes no graph key off CUDA: only the CUDA path replays
        /// its routing as data ([`Model::route_table`]), and this CPU stream has no captures.
        #[test]
        fn a_guessing_hybrid_takes_no_graph() {
            let (mut e, h) = engine(Arch::Qwen35, 8, Box::new(PromptLookup), 0, false);
            e.loaded.as_mut().unwrap().graph_bytes = 64 * GRAPH_BYTES;
            e.cfg.values[knob_id::DECODE_GRAPHS as usize] = 1.0;
            let hs: Vec<_> = (prompts(4).into_iter())
                .map(|r| h.submit(r).unwrap())
                .collect();
            drive(&mut e, &hs);
            assert!(e.loaded.as_ref().unwrap().graphs.is_empty());
        }
    }
}
