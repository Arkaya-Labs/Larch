//! Scheduler and KV bookkeeping (architecture 9.3, 9.4, D14): continuous batching, chunked prefill,
//! preemption by recompute, 16-token blocks and parked sessions. Pure host code: the engine thread
//! runs each [`StepPlan`] on the device and hands the sampled tokens to [`Scheduler::complete`].

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use larch_model::vision::IMAGE_TOKEN;
use larch_records::{
    EngineError, Finish, Outcome, ReclaimEvent, ReclaimReason, RequestClass, RequestSpan, ReuseMiss,
};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

use crate::grammar::Live;
use crate::guess::{Costs, Drafter, EXPLORE, Hits, PROBE_TOP, chained, expected};
use crate::lifecycle::Completion;
use crate::prefix::PrefixCache;
use crate::suffix::SuffixDrafter;
use crate::{Class, EngineEvent, MAX_GUESSES, Request, SampledToken, SamplingParams};

pub const BLOCK_TOKENS: usize = 16;

/// The most rows a step of decoding sequences reaches with their checks (each sequence's last
/// token plus its guesses): up to 16 rows a Metal step runs the matmuls through the same mmvq
/// kernels as a one-row decode, so a check keeps decode's numerics.
const CHECK_ROWS: usize = 16;

/// The tokens a step sampled, by request, in [`Scheduler::logits`] order.
pub(crate) type Sampled = Vec<(u64, SampledToken)>;

/// One logits row of a step.
pub(crate) struct Logit {
    pub(crate) id: u64,
    /// Its row in the step's batch.
    pub(crate) row: usize,
    /// The position it computes.
    pub(crate) pos: usize,
    /// Reported on the request's observation channel.
    pub(crate) observe: bool,
    /// Samples the request's next token.
    pub(crate) sample: bool,
}

/// Released session ids remembered so a late completion cannot park under them (10.1.1).
const RELEASED_RING: usize = 4096;

/// An observed request's prefill chunk at most. Its boundaries then do not move with the logits
/// budget, which free memory sizes from step to step; the numerics depend on them (a 450-token
/// passage on the target MoE scored 0.497 to 0.504 nats/token at chunks of 64, 128 and 512), so
/// without the cap scoring one version twice could disagree by up to 2 %.
pub(crate) const OBSERVE_CHUNK: usize = 128;

/// Activations requests running at once: a step taps at most this many, so the tap rows and
/// their output fit the scratch reserved for them at load (`Dims::tap_seats`).
pub(crate) const TAP_SEATS: usize = 4;

/// Background rows a step decodes beside a foreground row. A dense model reads every weight once a
/// step whatever the batch, so the whole batch runs.
// ponytail: a constant, since the knob registry is the frozen larch-records contract; becomes a
// knob set from the model class when the first MoE arch loads (an MoE in RAM wants 0). A budget
// below the whole batch fills with the oldest background rows (no rotation), and a foreground
// start or chunk takes a seat or KV from background rows only as any row does (a free seat, an
// empty plan); add both when a bench shows a chat waiting on a full set of workers.
pub(crate) const BACKGROUND_ROWS: usize = usize::MAX;

/// The KV block free list. Freeing a free block panics: a double free is a scheduler bug.
pub(crate) struct Blocks {
    free: Vec<u32>,
    is_free: Vec<bool>,
    pub(crate) refs: Vec<u32>,
    computed: Vec<u8>,
    tokens_used: usize,
}

impl Blocks {
    fn new(n: u32) -> Blocks {
        Blocks {
            free: (0..n).rev().collect(),
            is_free: vec![true; n as usize],
            refs: vec![0; n as usize],
            computed: vec![0; n as usize],
            tokens_used: 0,
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.is_free.len()
    }

    pub(crate) fn used(&self) -> usize {
        self.total() - self.free.len()
    }

    fn alloc(&mut self) -> Option<u32> {
        let b = self.free.pop()?;
        self.is_free[b as usize] = false;
        self.refs[b as usize] = 1;
        Some(b)
    }

    fn set_computed(&mut self, block: u32, n: usize) {
        let old = &mut self.computed[block as usize];
        self.tokens_used = self.tokens_used - usize::from(*old) + n;
        *old = n as u8;
    }

    fn completed(&mut self, seq: &Seq, start: usize, end: usize) {
        for i in start / BLOCK_TOKENS..blocks_for(end) {
            self.set_computed(seq.blocks[i], (end - i * BLOCK_TOKENS).min(BLOCK_TOKENS));
        }
    }

    pub(crate) fn retain(&mut self, b: u32) {
        assert!(self.refs[b as usize] > 0, "pin free block");
        self.refs[b as usize] = self.refs[b as usize]
            .checked_add(1)
            .expect("block references");
    }

    pub(crate) fn release(&mut self, b: u32) {
        assert!(!self.is_free[b as usize], "KV block {b} freed twice");
        self.refs[b as usize] -= 1;
        if self.refs[b as usize] != 0 {
            return;
        }
        self.set_computed(b, 0);
        self.is_free[b as usize] = true;
        self.free.push(b);
    }
}

/// A token history and its KV: `tokens[..n_kv]` have KV entries in `blocks`.
#[derive(Default)]
pub(crate) struct Seq {
    pub(crate) tokens: Vec<u32>,
    pub(crate) n_kv: usize,
    pub(crate) blocks: Vec<u32>,
    /// Engine clock when it parked, for least-recently-used reclaim.
    last_used: u64,
    generation: u64,
    namespace: Namespace,
    note: Note,
    /// A parked sequence's recurrent state slot, holding the state after `tokens[..n_kv]`.
    pub(crate) state: Option<u32>,
}

impl Seq {
    /// A carried pin's history: `tokens` with KV, no blocks yet.
    pub(crate) fn of_pin(tokens: Vec<u32>) -> Self {
        Self {
            n_kv: tokens.len(),
            tokens,
            ..Self::default()
        }
    }
}

/// Who may share KV: the cache namespace and the adaptation version the KV was computed under.
/// Router deltas change every later layer's K/V, so KV of one version never serves another.
pub(crate) type Namespace = (u64, u64);

/// A pinned chain a reload carries ([`Scheduler::carry_pinned`]).
pub(crate) struct PinnedChain {
    pub(crate) id: u64,
    pub(crate) namespace: u64,
    /// The pin's tokens through its last carried block.
    pub(crate) tokens: Vec<u32>,
    /// Its blocks in the old pool.
    pub(crate) blocks: Vec<u32>,
    /// The recurrent state after `tokens`, on a hybrid.
    pub(crate) checkpoint: Option<Arc<Vec<u8>>>,
}

/// A session's turn record for S03 and S04: on its sequence, then in `Scheduler::gone` once the
/// parked sequence went.
#[derive(Default)]
struct Note {
    /// Turns the session has started.
    turn: u32,
    /// Why its parked KV shrank or went since its last turn ended; None while intact.
    miss: Option<ReuseMiss>,
    /// The last reclaim that took its blocks since then, 0 if none.
    reclaim: u64,
    /// Fed tokens past `Seq::tokens` that lost their KV, so a prompt still matches its history.
    tail: Vec<u32>,
}

/// What a session turn's first start matched (S03, S04); [`NO_REUSE`] outside a session.
#[derive(Clone, Copy, Debug)]
struct Reuse {
    turn: u32,
    lcp: usize,
    parked: usize,
    available: usize,
    prior: bool,
    miss: ReuseMiss,
    reclaim: u64,
}

const NO_REUSE: Reuse = Reuse {
    turn: 0,
    lcp: 0,
    parked: 0,
    available: 0,
    prior: false,
    miss: ReuseMiss::None,
    reclaim: 0,
};

/// Tokens between the state checkpoints a hybrid's pin read keeps (one at a time), which a partial
/// export stops at: a quit at page 90 of 120 resumes from the last.
pub(crate) const PIN_MARK: usize = 8192;

/// The next multiple of `mark` past `from`, 0 when `mark` is 0 (marks off).
fn next_mark(mark: usize, from: usize) -> usize {
    from.checked_div(mark).map_or(0, |q| (q + 1) * mark)
}

/// What [`Scheduler::start`] would commit for a queued request: the reuse it found and its chunk.
struct Fit {
    global: bool,
    reuse: usize,
    retained: Vec<u32>,
    len: usize,
    /// `Req::save_at` and `Req::save_by` from this start.
    save_at: usize,
    save_by: usize,
    /// A hybrid's pin read: it cuts and saves at every this many tokens, else 0.
    marks: usize,
    /// The tokens of the prompt's cached chain, checkpointed or not (S03's `available`).
    indexed: usize,
}

impl Seq {
    fn pending(&self) -> usize {
        self.tokens.len() - self.n_kv
    }

    /// KV slot of position `p`.
    pub(crate) fn slot(&self, p: usize) -> i32 {
        (self.blocks[p / BLOCK_TOKENS] as usize * BLOCK_TOKENS + p % BLOCK_TOKENS) as i32
    }
}

/// LARCH_GUESS_OFF=1: a guessing load that checks no guesses, with the fit, attention splits and
/// kernels of one that does: the side without guesses of a greedy identity run.
fn guess_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("LARCH_GUESS_OFF").is_ok_and(|v| v == "1"))
}

/// LARCH_DRAFT_HYBRID=n: a sequence the load's drafter serves takes the suffix drafts first when
/// they guess at least n tokens (or all a check wants), a long confident match such as copied text
/// or JSON, and skips the draft model's forward; unset or 0, the load's drafter goes first.
fn hybrid() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("LARCH_DRAFT_HYBRID")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// LARCH_DRAFT_CONF=0: a DFlash drafter's checks are sized by the sequence's acceptance alone, not
/// by the draft's confidence in each guess.
fn by_confidence() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("LARCH_DRAFT_CONF").is_ok_and(|v| v == "0"))
}

fn blocks_for(tokens: usize) -> usize {
    tokens.div_ceil(BLOCK_TOKENS)
}

/// Keeps the first `n` tokens' KV: frees the blocks past them (9.3 step 2, reclaim).
fn truncate(blocks: &mut Blocks, s: &mut Seq, n: usize) {
    let n = private_reuse(blocks, s, n);
    for b in s.blocks.drain(blocks_for(n)..) {
        blocks.release(b);
    }
    s.n_kv = s.n_kv.min(n);
    if !s.n_kv.is_multiple_of(BLOCK_TOKENS) {
        let block = s.blocks[s.n_kv / BLOCK_TOKENS];
        assert_eq!(
            blocks.refs[block as usize], 1,
            "truncated tail must be private"
        );
        blocks.set_computed(block, s.n_kv % BLOCK_TOKENS);
    }
}

/// A shared full block may be read, but a truncation into it must recompute a private tail.
fn private_reuse(blocks: &Blocks, s: &Seq, n: usize) -> usize {
    if !n.is_multiple_of(BLOCK_TOKENS)
        && s.blocks
            .get(n / BLOCK_TOKENS)
            .is_some_and(|&b| blocks.refs[b as usize] > 1)
    {
        n / BLOCK_TOKENS * BLOCK_TOKENS
    } else {
        n
    }
}

/// Tokens from position `from` to the first of `stops` past it; all when none is.
fn to_stop(stops: impl IntoIterator<Item = usize>, from: usize) -> usize {
    (stops.into_iter().filter(|&at| at > from))
        .map(|at| at - from)
        .min()
        .unwrap_or(usize::MAX)
}

fn free(blocks: &mut Blocks, s: &mut Seq) {
    truncate(blocks, s, 0);
}

/// One admitted request.
pub(crate) struct Req {
    pub(crate) id: u64,
    session: u64,
    keep_kv: bool,
    prefix_cache: bool,
    namespace: u64,
    /// The adaptation version it first started under, which it keeps to its end (preemptions
    /// included); None until then.
    pub(crate) version: Option<u64>,
    pub(crate) seq: Seq,
    /// `Request::images`: the prompt rows fed an image's embedding rows, in prompt order.
    pub(crate) images: Vec<crate::ImageSpan>,
    /// Its recurrent state slot while it runs, on a model with recurrent layers.
    pub(crate) state_slot: Option<u32>,
    /// `Request::history_end`, 0 when it parks the state at its turn's end.
    history_end: usize,
    /// The slot holding its state at `history_end` (its snapshot), once a step started there.
    snap: Option<u32>,
    /// The block boundary where a hybrid's steps end to save its state as a checkpoint, 0 for
    /// none: where the chain it found outran the checkpoints it found, which a later prompt sharing
    /// that prefix then loads.
    save_at: usize,
    /// The last block boundary a prompt's own repeat can look up to (`PrefixCache::lookup` stops
    /// before its last token), 0 for none. A request that parks nothing (a stateless repeat's
    /// only reuse) cuts a chunk there to save; one that parks its state, which its next turn
    /// reuses, saves only where a chunk already ends on a block boundary within its last chunk
    /// before it, so its steps are the ones it runs without checkpoints.
    // ponytail: a kept session's one-chunk prompt keeps no prompt-end checkpoint (a pinned
    // document's read does: `fit`); cut for it too if the agent bench shows other sessions
    // prefilling its history again.
    save_by: usize,
    /// `Request::progress`: report each prefill chunk.
    progress: bool,
    /// A background request whose prompt is a pin's: a pinned document's read, which the pacing
    /// holds back for a chat ([`Scheduler::pace_decode`]).
    reads_pin: bool,
    /// A hybrid's pin read: the tokens between its checkpoints ([`PIN_MARK`]), else 0.
    pin_mark: usize,
    /// The block of its last mark, whose checkpoint the next one replaces.
    mark: Option<u32>,
    prompt_len: usize,
    max_tokens: u32,
    generated: u32,
    pub(crate) stop_tokens: Vec<u32>,
    pub(crate) ignore_eos: bool,
    pub(crate) logprobs: u8,
    pub(crate) score_ids: Vec<u32>,
    /// `Request.activations`: non-empty makes this a prefill-only request.
    pub(crate) activations: Vec<u16>,
    /// `Request.activation_rows`.
    pub(crate) tap_rows: Vec<u32>,
    /// The model refused this request's taps: it ends with `invalid::NO_TAPS`.
    pub(crate) untapped: bool,
    /// Its rows, set at the step boundary after its last chunk ran; its Done carries them.
    pub(crate) rows: Vec<Vec<f32>>,
    pub(crate) sampling: SamplingParams,
    class: Class,
    pub(crate) rng: ChaCha8Rng,
    pub(crate) grammar: Option<Live>,
    cancel: Arc<AtomicBool>,
    pub(crate) completion: Arc<Mutex<Completion>>,
    /// Prompt tokens reused from the parked sequence, set when it first runs.
    cached: usize,
    /// Teacher forcing: fed in place of the sampled tokens, in order.
    pub(crate) forced: Vec<u32>,
    /// Teacher forcing: every computed position's top-64 goes here.
    pub(crate) observe: bool,
    /// Each observed position also carries the next fed token's exact logprob (`token_loss`).
    pub(crate) score_next: bool,
    /// Positions below it were reported (a preempted request recomputes them unreported).
    pub(crate) next_obs: usize,
    /// Guessed tokens at the end of `seq.tokens` that this step checks; 0 between steps.
    pub(crate) guesses: usize,
    /// The draft's chance of each of its `guesses`; empty when its drafter gives none.
    conf: Vec<f32>,
    hits: Hits,
    /// The A/B arm at its first token (9.9).
    arm: Option<u8>,
    preemptions: u16,
    /// Its KV positions below this were computed once before a preemption discarded them (S05).
    redo: usize,
    reuse: Reuse,
    /// Nanoseconds on the engine clock.
    t_submit: u64,
    t_admit: u64,
    t_first: u64,
    t_last: u64,
    max_gap: u64,
}

impl Req {
    /// Tokens from position `from` to its snapshot or its next checkpoint, which a chunk does not
    /// cross; all when none is ahead.
    fn to_snapshot(&self, from: usize) -> usize {
        to_stop(
            [
                self.save_at,
                self.cut_by(),
                self.history_end,
                next_mark(self.pin_mark, from),
            ],
            from,
        )
    }

    /// `save_by` where a chunk is cut for it: a request that parks nothing; else 0.
    fn cut_by(&self) -> usize {
        if self.parks_as() == 0 {
            self.save_by
        } else {
            0
        }
    }

    /// The prompt tokens whose KV it may publish: all, except on a hybrid that snapshots inside a
    /// block, that block and on: its park cuts back into it, which a shared block refuses (its KV
    /// would end before its state).
    fn publishable(&self) -> usize {
        match self.history_end % BLOCK_TOKENS {
            0 => self.prompt_len,
            _ => self.history_end / BLOCK_TOKENS * BLOCK_TOKENS,
        }
    }

    /// The session its sequence parks under at its end, 0 if none.
    pub(crate) fn parks_as(&self) -> u64 {
        if self.keep_kv { self.session } else { 0 }
    }

    /// Whether it may guess: it samples its own tokens (no grammar, scores, teacher forcing or
    /// taps), more than one.
    pub(crate) fn may_draft(&self) -> bool {
        self.grammar.is_none()
            && self.score_ids.is_empty()
            && self.forced.is_empty()
            && !self.observe
            && self.activations.is_empty()
            && self.max_tokens > 1
    }

    /// Drops this step's guesses from its tokens.
    fn unguess(&mut self) {
        let n = self.seq.tokens.len() - self.guesses;
        self.seq.tokens.truncate(n);
        self.guesses = 0;
    }

    /// Tokens from position `from` through the end of the first image tile at or after it: while
    /// other requests run, a chunk stops there, so one step encodes at most one of this request's
    /// tiles and the other rows of the step wait for one tile (about 175 ms on a 3050), never a
    /// whole image. All when no image row is ahead.
    fn to_tile_end(&self, from: usize) -> usize {
        for sp in &self.images {
            let (at, end) = (sp.at as usize, (sp.at + sp.rows) as usize);
            if end <= from {
                continue;
            }
            let tile = sp.tile_rows.max(1) as usize;
            let first = from.max(at);
            let tile_end = (at + ((first - at) / tile + 1) * tile).min(end);
            return tile_end - from;
        }
        usize::MAX
    }

    /// Tokens from position `from` to the first image row at or after it (0 inside an image); all
    /// when no image row is ahead.
    fn to_image(&self, from: usize) -> usize {
        (self.images.iter())
            .find(|sp| (sp.at + sp.rows) as usize > from)
            .map_or(usize::MAX, |sp| (sp.at as usize).saturating_sub(from))
    }

    /// `len`, or less, so a chunk from `from` does not end inside an image tile it reaches: it ends
    /// where that tile starts. A tile the chunk starts inside, or one longer than the chunk, splits.
    fn tile_aligned(&self, from: usize, len: usize) -> usize {
        let end = from.saturating_add(len);
        for sp in &self.images {
            let (at, stop) = (sp.at as usize, (sp.at + sp.rows) as usize);
            if end <= at {
                break;
            }
            if end >= stop {
                continue;
            }
            let tile = sp.tile_rows.max(1) as usize;
            let start = at + (end - at) / tile * tile;
            return if start > from { start - from } else { len };
        }
        len
    }

    /// Whether positions `from..from + len` hold an image row.
    fn reads_image(&self, from: usize, len: usize) -> bool {
        (self.images.iter())
            .any(|sp| (sp.at as usize) < from + len && from < (sp.at + sp.rows) as usize)
    }

    /// The ids this request's sampled row gathers exact logits for: its score ids, or a narrow
    /// grammar state's allowed tokens.
    pub(crate) fn gathered(&self) -> &[u32] {
        self.grammar
            .as_ref()
            .map_or(&self.score_ids, |g| g.gather())
    }

    pub(crate) fn new(
        id: u64,
        r: Request,
        events: Sender<EngineEvent>,
        cancel: Arc<AtomicBool>,
        t_submit: u64,
    ) -> Req {
        let span = RequestSpan {
            req_id: id,
            t_submit_ns: t_submit,
            prompt_tokens: r.prompt.len() as u32,
            session_id: r.session,
            ..bytemuck::Zeroable::zeroed()
        };
        let grammar = r.grammar.map(|g| Live::new(g, &r.stop_tokens));
        Req {
            id,
            session: r.session,
            keep_kv: r.keep_kv,
            prefix_cache: r.prefix_cache,
            namespace: r.cache_namespace,
            version: None,
            prompt_len: r.prompt.len(),
            seq: Seq {
                tokens: r.prompt,
                ..Default::default()
            },
            images: r.images,
            state_slot: None,
            history_end: r.history_end as usize,
            snap: None,
            save_at: 0,
            save_by: 0,
            progress: r.progress,
            reads_pin: false,
            pin_mark: 0,
            mark: None,
            max_tokens: r.max_tokens,
            generated: 0,
            stop_tokens: r.stop_tokens,
            ignore_eos: r.ignore_eos,
            logprobs: r.logprobs,
            score_ids: r.score_ids,
            activations: r.activations,
            tap_rows: r.activation_rows,
            untapped: false,
            rows: Vec::new(),
            sampling: r.sampling,
            class: r.class,
            rng: ChaCha8Rng::seed_from_u64(r.sampling.seed.unwrap_or(id)),
            grammar,
            cancel,
            completion: Arc::new(Mutex::new(Completion::new(events, span))),
            cached: 0,
            forced: Vec::new(),
            observe: false,
            score_next: false,
            next_obs: 0,
            guesses: 0,
            conf: Vec::new(),
            hits: Hits::default(),
            arm: None,
            preemptions: 0,
            redo: 0,
            reuse: NO_REUSE,
            t_submit,
            t_admit: 0,
            t_first: 0,
            t_last: 0,
            max_gap: 0,
        }
    }
    fn span(&self, end: Result<Finish, EngineError>, now: u64) -> RequestSpan {
        RequestSpan {
            req_id: self.id,
            t_submit_ns: self.t_submit,
            t_admit_ns: self.t_admit,
            t_first_ns: self.t_first,
            t_done_ns: now,
            prompt_tokens: self.prompt_len as u32,
            gen_tokens: self.generated,
            cached_tokens: self.cached as u32,
            session_id: self.session,
            preemptions: self.preemptions,
            outcome: match end {
                Ok(_) => Outcome::Ok,
                Err(EngineError::Cancelled) => Outcome::Cancelled,
                Err(EngineError::DeviceLost) => Outcome::DeviceLost,
                Err(_) => Outcome::Error,
            }
            .raw(),
            finish: end.map_or(Finish::None, |f| f).raw(),
            error: end.err().map_or(0, |e| e.code()),
            class: match self.class {
                Class::Foreground => RequestClass::Foreground,
                Class::Background => RequestClass::Background,
            }
            .raw(),
            turn_id: self.reuse.turn,
            reuse_miss: self.reuse.miss.raw(),
            max_gap_ns: self.max_gap,
            lcp_tokens: self.reuse.lcp as u32,
            parked_kv_at_match: self.reuse.parked as u32,
            reuse_available_tokens: self.reuse.available as u32,
            prior_kv_applies: u8::from(self.reuse.prior),
            reclaim_id: self.reuse.reclaim,
            ..bytemuck::Zeroable::zeroed()
        }
    }

    fn snapshot(&self, arm: u8, running: bool) {
        self.completion
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .snapshot(
                self.span(Ok(Finish::None), 0),
                self.arm.unwrap_or(arm),
                running,
            );
    }

    /// The most likely candidates each token reports: none for a score request, whose `top` holds
    /// its scores instead.
    pub(crate) fn top_n(&self) -> u8 {
        if self.score_ids.is_empty() {
            self.logprobs
        } else {
            0
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Policy {
    DecodeFirst,
    Balanced,
    PrefillFirst,
}

/// The live limits, from `EffectiveConfig`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub(crate) max_batch_seqs: usize,
    pub(crate) prefill_chunk_tokens: usize,
    pub(crate) policy: Policy,
    pub(crate) kv_preempt_watermark: f64,
    /// Logits rows a step may have: the observation buffers, or what they can still grow to.
    pub(crate) logit_rows: usize,
}

/// What one step runs. Rows are the decode sequences (one token each) then the prefill chunks.
#[derive(Debug, Default)]
pub(crate) struct StepPlan {
    pub(crate) decode: Vec<u64>,
    /// (request, first position, tokens).
    pub(crate) prefill: Vec<(u64, usize, usize)>,
    pub(crate) n_preempted: u32,
    pub(crate) n_reclaimed: u32,
    /// `EngineSample.cap_blocked_ready`.
    pub(crate) cap_blocked: u32,
    /// (from, to) state slots the engine copies before the step: rows that start at their
    /// snapshot, into the slot the sequence parks with.
    pub(crate) copies: Vec<(u32, u32)>,
    /// Guessed rows the step checks, beyond one row a checked sequence; prefill chunks get the
    /// rest of the token budget and the logits rows.
    pub(crate) guessed: usize,
    /// (slot, checkpoint) the engine loads before the step, before the copies: rows that start
    /// where a checkpoint holds their state.
    pub(crate) loads: Vec<(u32, Arc<Vec<u8>>)>,
    /// The most `guessed` may reach.
    pub(crate) guess_room: usize,
    /// While guesses may be checked: the sequences decoding, the expected tokens the guesses
    /// planned so far emit beyond one a sequence, and whether the step explores ([`Costs::pick`]).
    decoding: usize,
    gain: f64,
    explore: bool,
    /// The sequence whose turn it is to retry a guess if the pick leaves it plain.
    probe: Option<u64>,
    /// Time the drafter took, which runs before the step and outside its GPU time.
    pub(crate) draft_ns: u64,
}

impl StepPlan {
    /// Decode sequences, prompt sequences and prompt tokens, as samples count them: a check's
    /// sequence decodes and feeds no prompt.
    pub(crate) fn counts(&self) -> (u32, u32, u32) {
        let (nd, np) = (self.decode.len() as u32, self.prefill.len() as u32);
        if self.guessed > 0 {
            return (nd + np, 0, 0);
        }
        (nd, np, self.prefill.iter().map(|p| p.2 as u32).sum())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.decode.is_empty() && self.prefill.is_empty()
    }

    fn has(&self, id: u64) -> bool {
        self.decode.contains(&id) || self.prefill.iter().any(|p| p.0 == id)
    }

    /// Takes a row back out of the step; the budget it used stays spent.
    fn unplan(&mut self, id: u64) {
        self.decode.retain(|&d| d != id);
        self.prefill.retain(|p| p.0 != id);
    }
}

/// The `EngineSample.tot_*` counters, each counted when it happens.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Totals {
    pub(crate) preemptions: u64,
    /// Requests that ended with Done.
    pub(crate) finished: u64,
    /// Prompt tokens of requests that started running.
    pub(crate) prompt_tokens: u64,
    /// Tokens sampled.
    pub(crate) gen_tokens: u64,
    /// Submit to first run, summed over `queued`.
    pub(crate) queue_ns: u64,
    /// Requests that started running.
    pub(crate) queued: u64,
    /// KV positions computed again after a preemption.
    pub(crate) recomputed_tokens: u64,
    /// Check steps' rounds (one a checked sequence), the guesses they tried and those kept.
    pub(crate) guess_rounds: u64,
    pub(crate) guesses_tried: u64,
    pub(crate) guesses_kept: u64,
    /// The guesses picked for each sequence that could check, by count (0: it decoded plainly).
    pub(crate) guess_k: [u64; MAX_GUESSES as usize + 1],
}

pub(crate) struct Scheduler {
    pub(crate) blocks: Blocks,
    pub(crate) prefix: PrefixCache,
    /// (slot, block): states the last step ended on at a checkpoint of a block that holds none,
    /// for the engine to save ([`PrefixCache::reserve`], [`PrefixCache::keep`]) before the next.
    pub(crate) saves: Vec<(u32, u32)>,
    /// (new mark block, the mark block before it): the old checkpoint goes once the new one is
    /// saved, so a partial export never finds neither.
    pub(crate) retire: Vec<(u32, u32)>,
    /// Tokens between a pin read's checkpoints ([`PIN_MARK`]; tests set fewer).
    pub(crate) pin_mark: usize,
    /// Pins that went cold to make room, for the engine to log and publish ([`Self::take_cold`]).
    gave_way: Vec<u64>,
    generation: u64,
    cache_lookups: u64,
    cache_hits: u64,
    reused_prompt_tokens: u64,
    computed_prompt_tokens: u64,
    reqs: HashMap<u64, Req>,
    waiting: VecDeque<u64>,
    /// Oldest first; preemption takes from the back.
    running: Vec<u64>,
    parked: HashMap<u64, Seq>,
    released: HashSet<u64>,
    released_order: VecDeque<u64>,
    /// Sessions whose parked sequence went, for their next turn: at most `RELEASED_RING` of them,
    /// oldest dropped first, holding at most one pool's worth of tail tokens.
    gone: HashMap<u64, Note>,
    gone_order: VecDeque<u64>,
    gone_tokens: usize,
    /// Reclaims for the engine to write to the ring, each with its id (from 1).
    pub(crate) reclaims: Vec<ReclaimEvent>,
    last_reclaim: u64,
    pub(crate) tot: Totals,
    /// Spans of the requests that ended, each with its A/B arm, for the engine to write to the ring.
    pub(crate) ended: Vec<(RequestSpan, u8)>,
    /// The A/B arm in force; the engine sets it.
    pub(crate) arm: u8,
    /// Admitted activations requests. At 0, every step runs exactly the code it ran before
    /// activations existed: no tap lookups here or in the executor.
    tapping: usize,
    /// Admitted background requests. At 0, `decode` and `prefill` run one pass in queue order,
    /// exactly as before classes existed.
    background: usize,
    /// [`BACKGROUND_ROWS`]; a test sets it lower.
    background_rows: usize,
    /// The model's recurrent state slots ([`Self::set_state_slots`]); 0 without recurrent layers.
    state_slots: u32,
    /// The slots no running request holds.
    free_slots: Vec<u32>,
    /// Proposes each decoding sequence's guesses.
    pub(crate) drafter: Box<dyn Drafter + Send>,
    draft: Vec<u32>,
    /// The draft's chance of each of `draft`; empty when its drafter gives none.
    draft_conf: Vec<f32>,
    /// The most guesses a sequence's check step tries; 0 turns guessing off.
    pub(crate) guess_max: usize,
    /// The most decoding sequences a step checks beside ([`crate::EngineOptions::guess_chats`]):
    /// past them a step decodes plainly.
    pub(crate) guess_chats: usize,
    /// Every sequence's check rounds of this load, whose acceptance a new request starts from.
    accept: Hits,
    /// Whether the model takes checks (`Model::begin_check`): None until it first answers. A
    /// model that refuses its first one never guesses again; one that took a check before may
    /// refuse one for want of room (its spare state slots) and guesses on.
    pub(crate) checkable: Option<bool>,
    /// A recurrent model's spare state slots free before this step's checks
    /// ([`larch_model::Model::check_spares_free`], which the engine sets before each schedule):
    /// a check snapshots its first row into one at most, so a step checks as many sequences. None:
    /// no count, and the model refuses a check it cannot hold.
    pub(crate) check_spares: Option<usize>,
    /// Step time by decode and guessed rows, which sizes each check.
    pub(crate) costs: Costs,
    /// Steps planned that could check, which set the exploring ones.
    may_check: u64,
    /// Each state slot's fed tokens, as the engine's copies and resets leave them (tests' `step`).
    #[cfg(test)]
    shadow: HashMap<u32, Vec<u32>>,
    /// The most sequences one planned step checked.
    #[cfg(test)]
    pub(crate) most_checks: usize,
    /// When the last step that ran a prompt chunk beside decode rows was planned, and the share
    /// of its time those rows then run alone.
    chunk_at: Option<(u64, f64)>,
    /// No prompt chunk joins running decode rows before this time ([`Self::schedule`]).
    prefill_due: u64,
    /// Set by [`Self::pace_decode`]; 0 runs every chunk at once.
    decode_share: f64,
    /// [`Self::decode_share`] for a pin read's chunks beside a chat's decode rows; 0 runs them
    /// at once.
    pin_share: f64,
    /// The most tokens a pin read's chunk takes in the step being planned: 0 holds it back, as
    /// pacing does for a chat's sake; not paced is `usize::MAX`.
    pin_piece: usize,
    /// The adaptation version new requests start under.
    pub(crate) version: u64,
    /// A version swap waits: no request that has not started yet starts.
    pub(crate) hold: bool,
    /// This step's plan holds a capped chunk that reads image rows: another capped chunk stops
    /// before its own (one tile a step, however many requests carry images).
    step_tile: bool,
    /// The projector encodes a unit some layers a step (a Qwen image): beside other requests, a
    /// chunk stops before an image whose unit is not in [`Self::units_ready`] until the engine has
    /// encoded it ([`Self::wanted_unit`]).
    pub(crate) slice_images: bool,
    /// (image digest, unit) the engine's unit cache holds, as it stood before this step's plan.
    pub(crate) units_ready: HashSet<([u8; 32], u32)>,
}

/// Decode-only time after each prompt chunk that ran beside decode rows, as a share of that
/// chunk's step. An MoE on Metal reads a prompt in one 2048-token chunk (2.2 s on the 35B on an
/// M5), during which the chats already replying got no token (1 Nov Mac run). ponytail: fixed
/// share, Mac-measured; make it adaptive (the newest chat's wait against the decoders') if agent
/// benches disagree.
const DECODE_SHARE: f64 = 0.25;

/// The largest prompt chunk beside decode rows while pacing is on: 512 tokens is 0.63 s on the
/// 35B, against 2.2 s for 2048 (the 4 pieces cost 15 % more than one).
const BUSY_CHUNK: usize = 512;

/// A chunk step shorter than this (a few decode steps) costs the decode rows too little to pace.
const PACE_MIN_NS: u64 = 20_000_000;

/// The longest chunk step pacing counts (a 2048-token chunk of the 35B on an M5 is 2.2 s).
const PACE_MAX_NS: u64 = 5_000_000_000;

impl Scheduler {
    /// Paces prompt chunks beside decode rows by [`DECODE_SHARE`] on Metal, where it was measured
    /// (CUDA's lane decides its own), and a pin read's chunks beside a chat's decode rows on every
    /// backend (a pinned document's read: nobody waits on it, a chat does; no other background
    /// request is paced); `LARCH_DECODE_SHARE` overrides both.
    pub(crate) fn pace_decode(&mut self, metal: bool) {
        let set = std::env::var("LARCH_DECODE_SHARE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &f64| v.is_finite() && *v >= 0.0);
        self.decode_share = set.unwrap_or(if metal { DECODE_SHARE } else { 0.0 });
        self.pin_share = set.unwrap_or(DECODE_SHARE);
    }

    pub(crate) fn new(n_blocks: u32) -> Scheduler {
        Scheduler {
            step_tile: false,
            slice_images: false,
            units_ready: HashSet::new(),
            blocks: Blocks::new(n_blocks),
            prefix: PrefixCache::default(),
            saves: Vec::new(),
            retire: Vec::new(),
            pin_mark: PIN_MARK,
            gave_way: Vec::new(),
            generation: 0,
            cache_lookups: 0,
            cache_hits: 0,
            reused_prompt_tokens: 0,
            computed_prompt_tokens: 0,
            reqs: HashMap::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            parked: HashMap::new(),
            released: HashSet::new(),
            released_order: VecDeque::new(),
            gone: HashMap::new(),
            gone_order: VecDeque::new(),
            gone_tokens: 0,
            reclaims: Vec::new(),
            last_reclaim: 0,
            tot: Totals::default(),
            ended: Vec::new(),
            arm: 0,
            tapping: 0,
            background: 0,
            background_rows: BACKGROUND_ROWS,
            chunk_at: None,
            prefill_due: 0,
            decode_share: 0.0,
            pin_share: 0.0,
            pin_piece: usize::MAX,
            state_slots: 0,
            free_slots: Vec::new(),
            drafter: Box::<SuffixDrafter>::default(),
            draft: Vec::new(),
            draft_conf: Vec::new(),
            guess_max: 0,
            guess_chats: CHECK_ROWS,
            accept: Hits::default(),
            checkable: None,
            check_spares: None,
            costs: Costs::default(),
            may_check: 0,
            #[cfg(test)]
            shadow: HashMap::new(),
            #[cfg(test)]
            most_checks: 0,
            version: 0,
            hold: false,
        }
    }

    /// A model with `n` recurrent state slots (0: none) serves at most `n` sequences at once, each
    /// in a slot of its own from its first chunk until it stops running. Its state cannot be cut
    /// back to a shorter prefix as KV can: a parked sequence keeps a slot with the state at its
    /// end (its snapshot, see `Request::history_end`), and the session's next turn reuses all of
    /// it or none. Slots left over hold the parked states and the snapshots; a start or a snapshot
    /// short of one takes the least recently parked. Across sessions a start reads the prefix
    /// cache up to the last block with a checkpoint (the state there, in host memory), which it
    /// loads; checkpoints are saved where `Req::save_at` cuts its steps.
    pub(crate) fn set_state_slots(&mut self, n: u32) {
        assert!(self.is_idle(), "state slots change with drained requests");
        self.state_slots = n;
        self.free_slots = (0..n).rev().collect();
        self.checkable = None;
        self.costs = Costs::default();
        self.accept = Hits::default();
    }

    /// The model grew its state to `n` slots live: the new ones are free, the held ones unchanged.
    pub(crate) fn grow_state_slots(&mut self, n: u32) {
        self.free_slots.extend((self.state_slots..n).rev());
        self.state_slots = self.state_slots.max(n);
    }

    /// The state slots no request holds, running or parked: a padded graph step's scratch.
    pub(crate) fn free_state_slots(&self) -> &[u32] {
        &self.free_slots
    }

    /// Replaces a drained pool, dropping parked KV but retaining counters and sticky releases.
    /// Each dropped session gets a Reload reclaim, in session order.
    pub(crate) fn reset_pool(&mut self, n_blocks: u32) {
        assert!(self.is_idle(), "pool reset requires drained requests");
        self.invalidate_prefix_cache();
        let mut parked: Vec<_> = self.parked.drain().collect();
        parked.sort_unstable_by_key(|&(session, _)| session);
        for (session, mut p) in parked {
            self.free_slots.extend(p.state);
            let id = self.reclaim_event(session, p.blocks.len(), p.n_kv, 0, ReclaimReason::Reload);
            p.tokens.append(&mut p.note.tail);
            self.forget(session, p.note.turn, ReuseMiss::Reloaded, id, p.tokens);
        }
        self.blocks = Blocks::new(n_blocks);
    }

    /// Takes the parked sequences a same-model reload may carry across ([`Self::carry_in`]), most
    /// recently used first: those a next turn could still reuse (the current generation) and
    /// computed with no router deltas, which the reload drops. The rest stay for
    /// [`Self::reset_pool`] to drop.
    pub(crate) fn carry_out(&mut self) -> Vec<(u64, Seq)> {
        assert!(self.is_idle(), "a carry requires drained requests");
        let sessions: Vec<u64> = (self.parked.iter())
            .filter(|(_, p)| p.namespace.1 == 0 && p.generation == self.generation)
            .map(|(&session, _)| session)
            .collect();
        let mut out: Vec<_> = (sessions.into_iter())
            .map(|session| (session, self.parked.remove(&session).expect("parked")))
            .collect();
        out.sort_unstable_by_key(|(session, p)| (std::cmp::Reverse(p.last_used), *session));
        out
    }

    /// Parks a carried sequence (`p`, with its old homes) on new blocks and a new state slot, which
    /// the engine fills from its copy; false, dropping it as [`Self::drop_carried`], when the pool or
    /// the slots lack room.
    pub(crate) fn carry_in(&mut self, session: u64, mut p: Seq) -> bool {
        let state = p.state.and_then(|_| self.free_slots.pop());
        if p.state.is_some() != state.is_some() || self.blocks.free.len() < p.blocks.len() {
            self.free_slots.extend(state);
            self.drop_carried(session, p);
            return false;
        }
        for b in &mut p.blocks {
            *b = self.blocks.alloc().expect("counted free");
        }
        self.blocks.completed(&p, 0, p.n_kv);
        (p.state, p.generation) = (state, self.generation);
        self.parked.insert(session, p);
        true
    }

    /// Ends pin `id`: its chain is evictable, and the cache comes down to its size at once.
    pub(crate) fn unpin(&mut self, id: u64) {
        self.prefix.unpin(id);
        self.prefix.trim(&mut self.blocks);
    }

    /// The chain of `tokens` in `namespace` that a carry or an export may take: its whole blocks
    /// under the base version, on a hybrid only up to its last checkpoint (KV past it cannot be
    /// read without the state), and for a read still under way only to a [`PIN_MARK`]; with the
    /// checkpoint of its last block on a hybrid. Empty when nothing qualifies.
    pub(crate) fn pin_chain(
        &mut self,
        namespace: u64,
        tokens: &[u32],
    ) -> (Vec<u32>, Option<Arc<Vec<u8>>>) {
        // Adapter version 0 only: router deltas, which make another version, are off; revisit when
        // a chat can run them beside a pinned document.
        let mut blocks = self.prefix.chain((namespace, 0), tokens);
        if blocks.len() < tokens.len() / BLOCK_TOKENS {
            let step = self.pin_mark / BLOCK_TOKENS;
            blocks.truncate(blocks.len() / step * step);
        }
        if self.state_slots > 0 {
            blocks.truncate(self.prefix.checkpointed(&blocks));
        }
        let checkpoint = (blocks.last().copied())
            .and_then(|b| self.prefix.checkpoint(b))
            .filter(|_| self.state_slots > 0);
        (blocks, checkpoint)
    }

    /// The pinned chains a same-model reload may carry ([`Self::pin_chain`]), taken before the
    /// checkpoints go.
    pub(crate) fn carry_pinned(&mut self) -> Vec<PinnedChain> {
        let pins: Vec<_> = (self.prefix.pins())
            .map(|(id, namespace, tokens)| (id, namespace, tokens.to_vec()))
            .collect();
        let mut out = Vec::new();
        for (id, namespace, mut tokens) in pins {
            let (blocks, checkpoint) = self.pin_chain(namespace, &tokens);
            if blocks.is_empty() {
                continue;
            }
            tokens.truncate(blocks.len() * BLOCK_TOKENS);
            out.push(PinnedChain {
                id,
                namespace,
                tokens,
                blocks,
                checkpoint,
            });
        }
        out
    }

    /// Takes `n` pool blocks for a chain to be filled and published ([`Self::publish_pin`]),
    /// reclaiming unpinned cache and parked blocks if short, or none when even that leaves fewer.
    pub(crate) fn alloc_pin(&mut self, n: usize) -> Option<Vec<u32>> {
        // Short of free blocks, the unpinned cache and parked tails go as a request's would.
        if self.blocks.free.len() < n && !self.reclaim(n, 0, &mut 0, true) {
            return None;
        }
        let blocks: Vec<u32> = (0..n)
            .map(|_| self.blocks.alloc().expect("counted free"))
            .collect();
        for &b in &blocks {
            self.blocks.set_computed(b, BLOCK_TOKENS);
        }
        Some(blocks)
    }

    /// Frees blocks [`Self::alloc_pin`] took that were never published.
    pub(crate) fn free_pin(&mut self, blocks: &[u32]) {
        for &b in blocks {
            self.blocks.release(b);
        }
    }

    /// Publishes `blocks` as the chain of `tokens` (whole blocks), pinned by its registered pin
    /// `id` (which is also the chains' producer, no request's id), and takes the cache's references
    /// (this one is released); returns the blocks that took it, all of them or fewer where the
    /// cache had no room (the chain's start is kept). On a hybrid a chain without its checkpoint is
    /// KV no one can read, which still counts against the pool: it is withdrawn, and none returned.
    pub(crate) fn publish_pin(
        &mut self,
        id: u64,
        namespace: u64,
        tokens: &[u32],
        mut blocks: Vec<u32>,
        checkpoint: Option<Vec<u8>>,
    ) -> Vec<u32> {
        let n = blocks.len();
        let ns = (namespace, 0);
        self.prefix
            .publish(&mut self.blocks, ns, 0, id, tokens, &mut blocks);
        // The cache holds the blocks it took; this reference was only for the publish.
        for &b in &blocks {
            self.blocks.release(b);
        }
        blocks.truncate(self.prefix.chain(ns, tokens).len());
        if let (Some(state), Some(&last)) = (checkpoint, blocks.last())
            && blocks.len() == n
        {
            self.prefix.keep(last, state);
        }
        if self.state_slots > 0
            && !blocks
                .last()
                .is_some_and(|&b| self.prefix.has_checkpoint(b))
        {
            self.uncarry_pin(id);
            blocks.clear();
        }
        blocks
    }

    /// Publishes a carried pinned chain (`tokens`, whole blocks) on new blocks, pinned again by its
    /// registered pin `id`; returns the blocks that took it, which the engine fills from its copy:
    /// all of them, or fewer where the pool or the cache had no room (the chain's start is kept;
    /// none for a hybrid's without its checkpoint, [`Self::publish_pin`]).
    pub(crate) fn carry_in_pin(
        &mut self,
        id: u64,
        namespace: u64,
        tokens: &[u32],
        checkpoint: Option<Vec<u8>>,
    ) -> Vec<u32> {
        match self.alloc_pin(tokens.len() / BLOCK_TOKENS) {
            Some(blocks) => self.publish_pin(id, namespace, tokens, blocks, checkpoint),
            None => Vec::new(),
        }
    }

    /// Withdraws what pin `id`'s publish made, whose copy would not load or has no state.
    pub(crate) fn uncarry_pin(&mut self, id: u64) {
        self.prefix.cancel(id, &mut self.blocks);
    }

    /// The new homes [`Self::carry_in`] gave `session`'s parked sequence: its blocks and state slot.
    pub(crate) fn carried(&self, session: u64) -> (&[u32], Option<u32>) {
        let p = &self.parked[&session];
        (&p.blocks, p.state)
    }

    /// Drops a carried sequence as a reload's reclaim, as [`Self::reset_pool`] drops one.
    pub(crate) fn drop_carried(&mut self, session: u64, mut p: Seq) {
        let id = self.reclaim_event(session, p.blocks.len(), p.n_kv, 0, ReclaimReason::Reload);
        p.tokens.append(&mut p.note.tail);
        self.forget(session, p.note.turn, ReuseMiss::Reloaded, id, p.tokens);
    }

    /// Drops a sequence [`Self::carry_in`] parked whose copy would not load, freeing its new homes.
    pub(crate) fn uncarry(&mut self, session: u64) {
        let p = self.parked.remove(&session).expect("carried");
        for &b in &p.blocks {
            self.blocks.release(b);
        }
        self.free_slots.extend(p.state);
        self.drop_carried(session, p);
    }

    /// Records a reclaim for the ring and returns its id.
    fn reclaim_event(
        &mut self,
        session: u64,
        blocks: usize,
        tokens: usize,
        work: usize,
        reason: ReclaimReason,
    ) -> u64 {
        self.last_reclaim += 1;
        self.reclaims.push(ReclaimEvent {
            reclaim_id: self.last_reclaim,
            session_id: session,
            blocks: blocks as u32,
            tokens: tokens as u32,
            admitted_work_tokens: work as u32,
            reason: reason.raw(),
            ..bytemuck::Zeroable::zeroed()
        });
        self.last_reclaim
    }

    /// Notes for `session`'s next turn that its parked KV went, and why; `history` is what that
    /// turn's prompt is matched against.
    fn forget(
        &mut self,
        session: u64,
        turn: u32,
        miss: ReuseMiss,
        reclaim: u64,
        history: Vec<u32>,
    ) {
        self.unforget(session);
        self.gone_tokens += history.len();
        self.gone.insert(
            session,
            Note {
                turn,
                miss: Some(miss),
                reclaim,
                tail: history,
            },
        );
        self.gone_order.push_back(session);
        let pool = self.blocks.total() * BLOCK_TOKENS;
        while self.gone.len() > RELEASED_RING || self.gone_tokens > pool {
            let old = self.gone_order.pop_front().expect("gone order");
            self.gone_tokens -= self.gone.remove(&old).expect("gone").tail.len();
        }
    }

    fn unforget(&mut self, session: u64) {
        if let Some(note) = self.gone.remove(&session) {
            self.gone_tokens -= note.tail.len();
            self.gone_order.retain(|&s| s != session);
        }
    }

    /// The highest turn `session` has started, from its notes and its live turns: two turns of one
    /// session may run at once, and each holds only its own.
    fn last_turn(&self, session: u64) -> u32 {
        let live = self.reqs.values().filter(|r| r.session == session);
        live.map(|r| r.reuse.turn)
            .chain(self.parked.get(&session).map(|p| p.note.turn))
            .chain(self.gone.get(&session).map(|n| n.turn))
            .max()
            .unwrap_or(0)
    }

    /// Ends every reuse of KV and states computed before: a parked session's too.
    pub(crate) fn new_generation(&mut self) {
        self.prefix.clear(&mut self.blocks);
        self.generation = self.generation.checked_add(1).expect("KV generation");
    }

    /// Explicit startup retention budget, bounded physically and on the CPU.
    pub(crate) fn configure_prefix_cache(&mut self, capacity: usize) {
        self.prefix.configure(&mut self.blocks, capacity);
    }

    pub(crate) fn invalidate_prefix_cache(&mut self) {
        self.prefix.clear(&mut self.blocks);
        // Performance-only settings retain legacy private session reuse when automatic sharing
        // is disabled. Pool/model replacement still drops every parked sequence in reset_pool.
        if self.prefix.capacity() > 0 {
            self.generation = self.generation.checked_add(1).expect("KV generation");
        }
    }

    /// Whether request `id`'s chunk ends where an image tile does: when any other request runs or
    /// waits to start (it would share the step), and always for a background request (a note's
    /// image must not hold a chat that starts mid-step). Alone, a request reads its image in whole
    /// chunks: fewer, larger steps.
    fn tile_capped(&self, id: u64) -> bool {
        self.reqs
            .get(&id)
            .is_some_and(|r| r.class == crate::Class::Background)
            || self.running.iter().any(|&x| x != id)
            || self.waiting.iter().any(|&x| x != id)
    }

    /// Request `id`'s chunk from `from`, at most `len`, under the image rules. Capped
    /// ([`Self::tile_capped`]), it ends at a tile's end, and once another capped chunk of the step
    /// reads image rows it stops before its own: the step waits for one tile, however many requests
    /// carry images (a background note's and a chat's). Alone, a chunk that would end inside a tile
    /// ends where the tile starts, so each tile is encoded within one step.
    fn image_chunk(&self, id: u64, from: usize, len: usize) -> usize {
        let r = &self.reqs[&id];
        if r.images.is_empty() {
            return len;
        }
        // Beside other requests, an image the projector encodes in slices waits, its first row
        // and on, until its unit is encoded: the step pays for a slice, not the whole image.
        if self.slice_images && self.tile_capped(id) {
            let reached = (r.images.iter())
                .find(|sp| (sp.at + sp.rows) as usize > from && (sp.at as usize) < from + len);
            if let Some(sp) = reached {
                let at = sp.at as usize;
                let unit = (from.max(at) - at) / sp.tile_rows.max(1) as usize;
                if !self
                    .units_ready
                    .contains(&(*sp.image.digest(), unit as u32))
                {
                    return len.min(at.saturating_sub(from));
                }
            }
        }
        if !self.tile_capped(id) {
            r.tile_aligned(from, len)
        } else if self.step_tile {
            len.min(r.to_image(from))
        } else {
            len.min(r.to_tile_end(from))
        }
    }

    /// The first image unit a request beside others has yet to read that the unit cache lacks,
    /// running requests first, then waiting ones: (request, span, unit). The engine encodes it some
    /// layers a step when the projector slices ([`Self::slice_images`]), ahead of the chunk that
    /// will read it; a request alone reads its image in one step, encoding it whole.
    pub(crate) fn wanted_unit(&self) -> Option<(u64, usize, u32)> {
        self.running.iter().chain(&self.waiting).find_map(|&id| {
            if !self.tile_capped(id) {
                return None;
            }
            let r = self.reqs.get(&id)?;
            let at = r.seq.n_kv;
            let k = (r.images.iter()).position(|sp| (sp.at + sp.rows) as usize > at)?;
            let sp = &r.images[k];
            let unit = (at.saturating_sub(sp.at as usize) / sp.tile_rows.max(1) as usize) as u32;
            (!self.units_ready.contains(&(*sp.image.digest(), unit))).then_some((id, k, unit))
        })
    }

    /// Records that request `id`'s chunk `from..from + len` joined the plan (see `step_tile`).
    fn planned_chunk(&mut self, id: u64, from: usize, len: usize) {
        if self.tile_capped(id) && self.reqs[&id].reads_image(from, len) {
            self.step_tile = true;
        }
    }

    fn eligible(&self, r: &Req) -> bool {
        self.prefix.capacity() > 0
            && r.prefix_cache
            && r.tap_rows.is_empty()
            && !r.observe
            && (r.session == 0 || (r.keep_kv && !self.released.contains(&r.session)))
    }

    pub(crate) fn prefix_cache_stats(&self) -> crate::PrefixCacheStats {
        crate::PrefixCacheStats {
            capacity_blocks: self.prefix.capacity(),
            total_blocks: self.blocks.total(),
            indexed_blocks: self.prefix.len(),
            pinned_blocks: self.prefix.pinned(),
            unique_blocks_used: self.blocks.used(),
            unique_tokens_used: self.used_tokens(),
            // Include the minimum hash-table allocation/headers even for tiny budgets.
            index_metadata_bytes: self.prefix.capacity().max(4) * 512,
            evictions: self.prefix.evictions,
            lookups: self.cache_lookups,
            hits: self.cache_hits,
            reused_prompt_tokens: self.reused_prompt_tokens,
            computed_prompt_tokens: self.computed_prompt_tokens,
        }
    }

    pub(crate) fn n_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub(crate) fn n_running(&self) -> usize {
        self.running.len()
    }

    pub(crate) fn n_parked(&self) -> usize {
        self.parked.len()
    }

    pub(crate) fn parked_blocks(&self) -> usize {
        if self.prefix.capacity() == 0 {
            self.parked.values().map(|s| s.blocks.len()).sum()
        } else {
            self.parked
                .values()
                .flat_map(|s| s.blocks.iter().copied())
                .collect::<HashSet<_>>()
                .len()
        }
    }

    /// Unique physical token occupancy: shared references never multiply utilization.
    pub(crate) fn used_tokens(&self) -> usize {
        self.blocks.tokens_used
    }

    /// `n_kv` of the session's parked sequence, 0 if none (D14).
    pub(crate) fn parked_tokens(&self, session: u64) -> usize {
        self.parked.get(&session).map_or(0, |s| s.n_kv)
    }

    pub(crate) fn req(&self, id: u64) -> Option<&Req> {
        self.reqs.get(&id)
    }

    pub(crate) fn req_mut(&mut self, id: u64) -> Option<&mut Req> {
        self.reqs.get_mut(&id)
    }

    pub(crate) fn is_idle(&self) -> bool {
        self.reqs.is_empty()
    }

    /// An observed request still has prompt rows to feed, which may need more logits rows.
    pub(crate) fn observing(&self) -> bool {
        self.reqs.values().any(|r| r.observe && r.seq.pending() > 1)
    }

    /// An activations request is admitted: the executor looks for taps only then.
    pub(crate) fn tapping(&self) -> bool {
        self.tapping > 0
    }

    /// A started request still runs (or waits, preempted) under the version it started with.
    pub(crate) fn started(&self) -> bool {
        self.reqs.values().any(|r| r.version.is_some())
    }

    /// Where `r`'s KV belongs: its own version once started, else the one it would start under.
    fn ns(&self, r: &Req) -> Namespace {
        (r.namespace, r.version.unwrap_or(self.version))
    }

    /// Takes the queue, oldest first, once no request has started (a demand switch resets the
    /// pool under it); [`Self::admit`] queues each again.
    pub(crate) fn take_unstarted(&mut self) -> Vec<Req> {
        assert!(!self.started(), "a switch waits for the started requests");
        let mut out = Vec::with_capacity(self.waiting.len());
        for id in std::mem::take(&mut self.waiting) {
            let Some(mut r) = self.reqs.remove(&id) else {
                continue;
            };
            self.tapping -= usize::from(!r.activations.is_empty());
            self.background -= usize::from(r.class == Class::Background);
            self.free_slots.extend(r.snap.take());
            out.push(r);
        }
        out
    }

    /// Queues a request. It holds no KV until it first runs (see [`Self::start`]).
    pub(crate) fn admit(&mut self, mut r: Req) {
        if self.state_slots == 0 || r.session == 0 || !r.keep_kv || r.history_end > r.prompt_len {
            r.history_end = 0;
        }
        self.tapping += usize::from(!r.activations.is_empty());
        self.background += usize::from(r.class == Class::Background);
        r.reads_pin =
            r.class == Class::Background && self.prefix.is_pin(r.namespace, &r.seq.tokens);
        r.completion.lock().unwrap_or_else(|e| e.into_inner()).arm = self.arm;
        r.hits = Hits::from_prior(self.accept.acceptance());
        self.waiting.push_back(r.id);
        self.reqs.insert(r.id, r);
    }

    /// Makes the oldest pin cold, as a request that is not a pin's read needs its blocks: a pin
    /// is a cache, so it gives way instead of failing or holding up the request. False with none left.
    fn unpin_oldest(&mut self) -> bool {
        let Some(id) = self.prefix.pins().next().map(|p| p.0) else {
            return false;
        };
        self.unpin(id);
        self.gave_way.push(id);
        true
    }

    /// The pins that went cold since the last call.
    pub(crate) fn take_cold(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.gave_way)
    }

    /// Requests with no first token yet follow an arm transition; the others retain their arm.
    pub(crate) fn set_arm(&mut self, arm: u8) {
        if self.arm != arm {
            self.arm = arm;
            for r in self.reqs.values().filter(|r| r.arm.is_none()) {
                r.completion.lock().unwrap_or_else(|e| e.into_inner()).arm = arm;
            }
        }
    }

    /// Frees the session's parked sequence; later parks under it free instead (sticky, 10.1.1).
    pub(crate) fn release_session(&mut self, session: u64) {
        // Keep releases sticky for accepted requests even after the bounded ID ring wraps.
        // This state is bounded by admission, and survives preemption/resume with the request.
        for r in self.reqs.values_mut().filter(|r| r.session == session) {
            r.keep_kv = false;
            r.prefix_cache = false;
            // It parks nothing now, so its snapshot would only evict another session's state.
            r.history_end = 0;
            self.free_slots.extend(r.snap.take());
        }
        if self.released.insert(session) {
            self.released_order.push_back(session);
            if self.released_order.len() > RELEASED_RING
                && let Some(old) = self.released_order.pop_front()
            {
                self.released.remove(&old);
            }
        }
        self.prefix.release_session(session, &mut self.blocks);
        if let Some(mut p) = self.parked.remove(&session) {
            free(&mut self.blocks, &mut p);
            self.free_slots.extend(p.state);
            self.forget(
                session,
                p.note.turn,
                ReuseMiss::Reset,
                p.note.reclaim,
                Vec::new(),
            );
        } else if let Some(&Note { turn, reclaim, .. }) = self.gone.get(&session) {
            self.forget(session, turn, ReuseMiss::Reset, reclaim, Vec::new());
        }
    }

    /// Blocks reclaimable without touching active readers or a proposed acquisition. Counting
    /// physical IDs (not references) makes feasibility checks transactional under sharing.
    fn reclaimable(&self, excluded: &HashSet<u32>) -> usize {
        self.reclaimable_with(excluded, false)
    }

    /// [`Self::reclaimable`], counting the pinned chains too when `pins`: what the pool could give
    /// once every pin went cold.
    fn reclaimable_with(&self, excluded: &HashSet<u32>, pins: bool) -> usize {
        let mut live: HashSet<_> = self
            .reqs
            .values()
            .flat_map(|r| r.seq.blocks.iter().copied())
            .collect();
        live.extend(excluded);
        self.prefix.protect_ancestors(&mut live);
        self.parked
            .values()
            .flat_map(|s| s.blocks.iter().copied())
            .chain(self.prefix.unpinned())
            .chain(self.prefix.blocks().filter(|_| pins))
            .filter(|b| !live.contains(b) && !excluded.contains(b))
            .collect::<HashSet<_>>()
            .len()
    }

    /// Idle global entries and parked tails are reclaimed before any live request is preempted.
    /// Short of that, pins go cold oldest first, unless `pin_read` (a pin's own read, which pins
    /// do not give way to). Failed feasibility leaves both forms of retention unchanged. Each
    /// parked tail taken is a Demand reclaim for `work` tokens; its lost tokens move to the
    /// sequence's note.
    fn reclaim(&mut self, need: usize, work: usize, reclaimed: &mut u32, pin_read: bool) -> bool {
        let excluded = HashSet::new();
        // A pin goes cold only when that makes the room: blocks live requests hold are waited
        // for (or preempted) as without pins.
        let short =
            |s: &Self, pins| s.blocks.free.len() + s.reclaimable_with(&excluded, pins) < need;
        if pin_read && short(self, false) || short(self, true) {
            return false;
        }
        while short(self, false) {
            if !self.unpin_oldest() {
                return false;
            }
        }
        while self.blocks.free.len() < need {
            if self.prefix.evict(&mut self.blocks, &excluded) {
                continue;
            }
            let Some((&session, _)) = self
                .parked
                .iter()
                .min_by_key(|&(&session, s)| (s.last_used, session))
            else {
                return false;
            };
            let p = self.parked.get_mut(&session).expect("parked");
            // A recurrent state stands at the sequence's end: such a sequence goes whole.
            let take = match p.state {
                Some(_) => p.blocks.len(),
                None => (need - self.blocks.free.len()).min(p.blocks.len()),
            };
            let n_kv = p.n_kv;
            truncate(&mut self.blocks, p, (p.blocks.len() - take) * BLOCK_TOKENS);
            let mut lost = p.tokens.split_off(p.n_kv);
            lost.append(&mut p.note.tail);
            p.note.tail = lost;
            let lost = n_kv - p.n_kv;
            *reclaimed += take as u32;
            let id = self.reclaim_event(session, take, lost, work, ReclaimReason::Demand);
            let p = self.parked.get_mut(&session).expect("parked");
            (p.note.miss, p.note.reclaim) = (Some(ReuseMiss::Reclaimed), id);
            if p.blocks.is_empty() {
                let p = self.parked.remove(&session).expect("parked");
                self.free_slots.extend(p.state);
                self.forget(session, p.note.turn, ReuseMiss::Reclaimed, id, p.note.tail);
            }
        }
        true
    }

    /// Gives request `id` blocks for its first `tokens` positions, reclaiming parked blocks if short.
    fn grow(&mut self, id: u64, tokens: usize, reclaimed: &mut u32) -> bool {
        let seq = &self.reqs[&id].seq;
        let (need, work) = (
            blocks_for(tokens).saturating_sub(seq.blocks.len()),
            tokens.saturating_sub(seq.n_kv),
        );
        let pin_read = self.reqs[&id].reads_pin;
        if need > self.blocks.free.len() && !self.reclaim(need, work, reclaimed, pin_read) {
            return false;
        }
        for _ in 0..need {
            let b = self.blocks.alloc().expect("reclaimed");
            self.reqs.get_mut(&id).expect("request").seq.blocks.push(b);
        }
        true
    }

    /// Back to the front of the waiting queue, to recompute its prompt and generated tokens.
    fn preempt(&mut self, id: u64, plan: &mut StepPlan) {
        let r = self.reqs.get_mut(&id).expect("request");
        r.unguess();
        r.redo = r.redo.max(r.seq.n_kv);
        free(&mut self.blocks, &mut r.seq);
        self.free_slots.extend(r.state_slot.take());
        self.free_slots.extend(r.snap.take());
        r.preemptions += 1;
        r.snapshot(self.arm, false);
        self.running.retain(|&x| x != id);
        self.waiting.push_front(id);
        self.tot.preemptions += 1;
        plan.n_preempted += 1;
    }

    /// Plans one step (9.3). Every policy plans every running foreground decode row, and every
    /// background one up to `background_rows` beside a foreground row (see [`Self::decode`]);
    /// `prefill_first` lets the prompts claim the token budget, the seats and the free blocks
    /// first, except that a start leaves the decode rows their next blocks (see [`Self::start`])
    /// and a decode row short of a block takes a newer or a background row's (see [`Self::victim`]).
    #[cfg(test)]
    pub(crate) fn schedule(&mut self, lim: &Limits, now: u64) -> StepPlan {
        self.schedule_with(lim, now, None)
    }

    /// [`Self::schedule`] with `drafter` proposing the guesses first, [`Self::drafter`] for the
    /// sequences it proposes none for.
    pub(crate) fn schedule_with(
        &mut self,
        lim: &Limits,
        now: u64,
        d: Option<&mut (dyn Drafter + '_)>,
    ) -> StepPlan {
        let mut plan = StepPlan::default();
        self.step_tile = false;
        let cancelled: Vec<u64> = self
            .reqs
            .values()
            .filter(|r| r.cancel.load(Ordering::Relaxed))
            .map(|r| r.id)
            .collect();
        for id in cancelled {
            self.finish(id, Err(EngineError::Cancelled), now);
        }
        // A live cut of max_batch_seqs takes effect on this step: the newest go back to the queue,
        // background rows first, so a cut never takes a person's seat while a worker holds one.
        while self.running.len() > lim.max_batch_seqs {
            let mut seated = self.running.iter().rev().copied();
            let background = (self.background > 0)
                .then(|| seated.find(|&v| self.reqs[&v].class == Class::Background))
                .flatten();
            let newest = background.unwrap_or(*self.running.last().expect("running"));
            self.preempt(newest, &mut plan);
        }
        self.pin_piece = usize::MAX;
        if lim.policy == Policy::PrefillFirst {
            self.prefill(lim, &mut plan, now);
            let used = plan.prefill.iter().map(|p| p.2).sum();
            self.room(&mut plan, lim, used);
            self.decode(&mut plan, d);
        } else {
            // The step after a chunk beside decode rows tells how long that chunk took; the
            // decode rows then run alone for its share before the next chunk.
            if let Some((at, share)) = self.chunk_at.take() {
                // An idle engine between the two plans would stretch the wait: a chunk step is
                // never longer than PACE_MAX_NS.
                let took = now.saturating_sub(at).min(PACE_MAX_NS);
                if took >= PACE_MIN_NS {
                    let wait = took as f64 * share;
                    self.prefill_due = now.saturating_add(wait as u64);
                }
            }
            self.room(&mut plan, lim, 0);
            self.decode(&mut plan, d);
            let chat = (plan.decode.iter()).any(|id| self.reqs[id].class == Class::Foreground);
            if plan.decode.is_empty() {
                self.prefill(lim, &mut plan, now);
            } else if self.decode_share > 0.0 {
                if now >= self.prefill_due {
                    // Beside replying chats a prompt goes in pieces they can run between; alone it
                    // keeps the full chunk (2048 for an MoE on Metal), the faster read.
                    let mut busy = *lim;
                    busy.prefill_chunk_tokens = busy.prefill_chunk_tokens.min(BUSY_CHUNK);
                    self.prefill(&busy, &mut plan, now);
                }
            } else {
                // Where only pin reads are paced, every other prompt, a chat's own included, goes
                // as it does unpaced.
                if chat {
                    self.pin_piece = if now >= self.prefill_due {
                        BUSY_CHUNK
                    } else {
                        0
                    };
                }
                self.prefill(lim, &mut plan, now);
            }
            if !plan.decode.is_empty() && !plan.prefill.is_empty() && plan.guessed == 0 {
                let paced = (plan.prefill.iter()).any(|p| self.reqs[&p.0].reads_pin);
                let share = if self.decode_share > 0.0 {
                    self.decode_share
                } else if paced && chat {
                    self.pin_share
                } else {
                    0.0
                };
                if share > 0.0 {
                    self.chunk_at = Some((now, share));
                }
            }
        }
        self.snapshots(&mut plan);
        plan
    }

    /// Plans a copy of the state of each row that starts at its snapshot into a slot of its own,
    /// which its sequence parks with; a row that finds no slot parks the state at its turn's end.
    fn snapshots(&mut self, plan: &mut StepPlan) {
        let starts = plan.decode.iter().map(|&id| (id, self.reqs[&id].seq.n_kv));
        let due: Vec<u64> = starts
            .chain(plan.prefill.iter().map(|&(id, start, _)| (id, start)))
            .filter(|&(id, start)| {
                let r = &self.reqs[&id];
                r.history_end == start && r.history_end > 0 && r.snap.is_none()
            })
            .map(|(id, _)| id)
            .collect();
        for id in due {
            let slot = self.take_slot(&mut plan.n_reclaimed);
            let r = self.reqs.get_mut(&id).expect("request");
            match slot {
                Some(slot) => {
                    r.snap = Some(slot);
                    plan.copies
                        .push((r.state_slot.expect("a running sequence"), slot));
                }
                None => r.history_end = 0,
            }
        }
    }

    /// A free state slot, else the least recently parked sequence's, which goes whole (a Demand
    /// reclaim, counted in `reclaimed`): its KV is no use without its state.
    fn take_slot(&mut self, reclaimed: &mut u32) -> Option<u32> {
        if let Some(slot) = self.free_slots.pop() {
            return Some(slot);
        }
        let (&session, _) = self
            .parked
            .iter()
            .min_by_key(|&(&session, p)| (p.last_used, session))?;
        let mut p = self.parked.remove(&session).expect("parked");
        *reclaimed += p.blocks.len() as u32;
        let id = self.reclaim_event(session, p.blocks.len(), p.n_kv, 0, ReclaimReason::Demand);
        free(&mut self.blocks, &mut p);
        p.tokens.append(&mut p.note.tail);
        self.forget(session, p.note.turn, ReuseMiss::Reclaimed, id, p.tokens);
        p.state
    }

    /// The slot of the newest running snapshot, for a start that finds every other slot a running
    /// sequence's: a seat is free, so one is a snapshot's. Its sequence parks the state at its
    /// turn's end instead.
    fn steal_snapshot(&mut self) -> u32 {
        let id = *self
            .running
            .iter()
            .rev()
            .find(|id| self.reqs[id].snap.is_some())
            .expect("max_batch_seqs is at most the state slots");
        let r = self.reqs.get_mut(&id).expect("request");
        r.history_end = 0;
        r.snap.take().expect("a snapshot")
    }

    /// The waiting requests when every seat is taken and the request that starts next would fit
    /// a free seat's chunk; 0 when KV, not `max_batch_seqs`, holds the queue back.
    // ponytail: probes only the queue's head, the one that starts next; count those behind it
    // that would fit too if S01 needs more than "the cap blocks ready work".
    pub(crate) fn cap_blocked(&self, lim: &Limits) -> u32 {
        if self.running.len() < lim.max_batch_seqs {
            return 0;
        }
        let fg = |id: &&u64| self.reqs[*id].class == Class::Foreground;
        let head = (self.background > 0)
            .then(|| self.waiting.iter().find(fg))
            .flatten()
            .or(self.waiting.front());
        match head {
            Some(&id) if self.fit(id, lim.prefill_chunk_tokens, lim).is_ok() => {
                self.waiting.len() as u32
            }
            _ => 0,
        }
    }

    /// Every running sequence with one token to feed, oldest first, except an activations request,
    /// whose taps run only in an eager prefill row (never a captured decode graph). With a
    /// background request admitted, the foreground rows go first, so a foreground row short of a
    /// block preempts a background one, and beside a foreground row at most `background_rows`
    /// background rows decode; the others keep their KV and wait.
    fn decode(&mut self, plan: &mut StepPlan, mut d: Option<&mut (dyn Drafter + '_)>) {
        if self.background == 0 {
            return self.decode_rows(plan, None, usize::MAX, d);
        }
        self.decode_rows(plan, Some(Class::Foreground), usize::MAX, d.as_deref_mut());
        let beside = plan
            .decode
            .iter()
            .any(|id| self.reqs[id].class == Class::Foreground);
        let room = if beside {
            self.background_rows
        } else {
            usize::MAX
        };
        self.decode_rows(plan, Some(Class::Background), room, d);
    }

    /// The row a row of `class` short of blocks preempts: the newest one of `rows` of its class,
    /// planned this step or not, never one of a higher class. A foreground row takes the newest
    /// background row first, older than it or not. A decode row passes every running row, so its
    /// victim is itself or newer within its class: the oldest row of the highest class always goes
    /// on and every admitted request ends. Skipping planned rows instead let a decode row preempt
    /// itself beside a newer planned chunk under `prefill_first`, whose restart did the same to that
    /// row, forever (#196). A row that started this step is never the victim (see [`Self::start`]).
    fn victim(&self, class: Option<Class>, rows: &[u64]) -> Option<u64> {
        let mut seated = self.running.iter().rev().copied();
        let newest = |rows: &[u64], only: Option<Class>| {
            let mut seated = rows.iter().rev().copied();
            seated.find(|&v| only.is_none_or(|c| self.reqs[&v].class == c))
        };
        match class {
            Some(Class::Foreground) => seated
                .find(|&v| self.reqs[&v].class == Class::Background)
                .or_else(|| newest(rows, None)),
            _ => newest(rows, class),
        }
    }

    /// Plans up to `room` decode rows of `class` (None: any). Short of a block, a row preempts the
    /// one [`Self::victim`] names, which may be itself, and drops its plan for this step; the budget
    /// it used stays spent.
    fn decode_rows(
        &mut self,
        plan: &mut StepPlan,
        class: Option<Class>,
        mut room: usize,
        mut d: Option<&mut (dyn Drafter + '_)>,
    ) {
        let mut i = 0;
        while let Some(&id) = self.running.get(i) {
            let r = &self.reqs[&id];
            if plan.has(id)
                || r.seq.pending() != 1
                || !r.activations.is_empty()
                || class.is_some_and(|c| c != r.class)
                || room == 0
            {
                i += 1;
                continue;
            }
            let n_kv = self.reqs[&id].seq.n_kv;
            // A recurrent model's checks share its free spares: each takes one at most.
            let mut left = plan.guess_room - plan.guessed;
            if self.state_slots > 0
                && let Some(free) = self.check_spares
                && plan.prefill.len() >= free
            {
                left = 0;
            }
            let k = self.guess(id, left, plan, d.as_deref_mut());
            loop {
                if self.grow(id, n_kv + 1 + k, &mut plan.n_reclaimed) {
                    if k > 0 {
                        // A check: the last token and its guesses, a logit for every row.
                        plan.prefill.push((id, n_kv, k + 1));
                        plan.guessed += k;
                        // A step that checks feeds no prompt: its prefill rows are its checks.
                        #[cfg(test)]
                        {
                            self.most_checks = self.most_checks.max(plan.prefill.len());
                        }
                    } else {
                        plan.decode.push(id);
                    }
                    room -= 1;
                    break;
                }
                let victim = self.victim(class, &self.running).expect("id is running");
                // Each older victim moves this row, and the ones after it, one seat down.
                i -= usize::from(self.running[..i].contains(&victim));
                // ponytail: a dropped chunk's row loses all its KV even where waiting a step would
                // have left the block free; reserve for chunks as `start` does if a bench shows
                // preemption churn under prefill_first.
                plan.unplan(victim);
                self.preempt(victim, plan);
                if victim == id {
                    break;
                }
            }
            if self.running.get(i) == Some(&id) {
                i += 1;
            }
        }
    }

    /// The guessed rows a step may check: none for a model that refused a check, while a
    /// teacher-forced run observes (its rows take the spare logits rows), or while a prompt has
    /// rows to feed (`used` tokens planned, a chunk to continue, a request that could start): a
    /// check step holds only decode rows and checks, which the model computes as decode rows
    /// (`StepBatch::check`), so prompts never wait on guesses, or past [`Self::guess_chats`]
    /// decoding sequences. Else the room left up to [`CHECK_ROWS`] beside one row a decoding
    /// sequence, within the logits rows past one a seat.
    fn guess_room(&self, lim: &Limits, used: usize, decoding: usize) -> usize {
        let prompts = used > 0
            || (!self.waiting.is_empty() && self.running.len() < lim.max_batch_seqs)
            || self.running.iter().any(|id| {
                let r = &self.reqs[id];
                r.seq.pending() > 1 || (r.seq.pending() == 1 && !r.activations.is_empty())
            });
        if prompts
            || self.guess_max == 0
            || decoding > self.guess_chats
            || self.checkable == Some(false)
            || self.reqs.values().any(|r| r.observe)
            || guess_off()
        {
            return 0;
        }
        CHECK_ROWS
            .saturating_sub(decoding)
            .min(lim.logit_rows.saturating_sub(lim.max_batch_seqs))
            .min(lim.prefill_chunk_tokens.saturating_sub(used))
    }

    /// Sets the plan's [`Self::guess_room`] beside `used` prompt tokens, and what sizes its checks.
    fn room(&mut self, plan: &mut StepPlan, lim: &Limits, used: usize) {
        let decoding = |id: &&u64| self.reqs[*id].seq.pending() == 1;
        plan.decoding = self.running.iter().filter(decoding).count();
        plan.guess_room = self.guess_room(lim, used, plan.decoding);
        if plan.guess_room == 0 {
            return;
        }
        plan.explore = self.may_check.is_multiple_of(EXPLORE);
        if plan.explore {
            // Bounded per step, not per chat: a check step costs every row of the batch.
            let turn = (self.may_check / EXPLORE) as usize % plan.decoding.max(1);
            plan.probe = self.running.iter().filter(decoding).nth(turn).copied();
        }
        self.may_check += 1;
    }

    /// Appends the drafter's guesses for decoding request `id` to its tokens, at most `room` and
    /// `guess_max`, and returns how many. None past `max_tokens` (a round emits one token more
    /// than it guesses), on a prompt's or a recomputed token (the counters and the prefix publish
    /// read a decode row), for a request whose rows are its own (grammar, scores, teacher forcing),
    /// while its hit rate keeps it off, or when the guesses' blocks would leave fewer free blocks
    /// than the running sequences may need: guessing never reclaims or preempts. Within those, the
    /// measured step cost and its acceptance size the check ([`Costs::pick`]), none when nothing
    /// beats a plain decode row, but for one guess now and then where good guesses would pay. `d`
    /// proposes first, [`Self::drafter`] if it proposes none (DFlash proposes for every seated
    /// sequence, or fails the step). A draft that scores its guesses then trims them by its
    /// calibrated confidence in each ([`chained`]) where acceptance alone would size the check.
    fn guess(
        &mut self,
        id: u64,
        room: usize,
        plan: &mut StepPlan,
        d: Option<&mut (dyn Drafter + '_)>,
    ) -> usize {
        let r = self.reqs.get_mut(&id).expect("request");
        let left = r.max_tokens.saturating_sub(r.generated + 1) as usize;
        let max = self.guess_max.min(room).min(left);
        if max == 0
            || r.seq.n_kv < r.prompt_len.max(r.redo)
            || !r.may_draft()
            || !r.hits.may_guess()
        {
            return 0;
        }
        let a = r.hits.acceptance();
        let pick = |a, max, explore| {
            (self.costs).pick(plan.decoding, plan.guessed, plan.gain, a, max, explore)
        };
        let mut k = pick(a, max, plan.explore);
        // Acceptance only moves by checking, so a sequence left plain could never show it guesses
        // well. One guess, by turns, on an exploring step, only where what it can show would flip
        // the pick; the rows round up to their bucket's, so a check is not credited with rows the
        // bucket's cost holds.
        let rows = plan.decoding.next_power_of_two();
        let probe = k == 0
            && plan.probe == Some(id)
            && (self.costs).pick(rows, plan.guessed, plan.gain, PROBE_TOP, max, false) > 0;
        if k > 0 || probe {
            let want = k.max(1);
            self.draft.clear();
            self.draft_conf.clear();
            let ns = r.seq.namespace;
            let started = std::time::Instant::now();
            // A seeded sampled request replays by seed: guesses decide how the RNG is drawn, so
            // they must not come from other requests' outputs.
            // ponytail: it loses its own-text drafts too; split the drafter's own and global
            // lookups if seeded sampling with guesses shows up in a bench.
            let replayable = r.sampling.seed.is_some() && r.sampling.temperature > 0.0;
            let suffix_ok = !(replayable && self.drafter.uses_history());
            if let Some(d) = d {
                let h = hybrid();
                if h > 0 && suffix_ok {
                    (self.drafter).propose(id, ns, &r.seq.tokens, want, &mut self.draft);
                    if self.draft.len() < h.min(want) {
                        self.draft.clear();
                    }
                }
                // A scoring drafter is asked for all `max`: its confidences trim them after.
                let scored = self.draft.is_empty()
                    && by_confidence()
                    && d.propose_scored(
                        id,
                        ns,
                        &r.seq.tokens,
                        max,
                        &mut self.draft,
                        &mut self.draft_conf,
                    );
                if self.draft.is_empty() && !scored {
                    d.propose(id, ns, &r.seq.tokens, want, &mut self.draft);
                }
            }
            if self.draft.is_empty() && suffix_ok {
                (self.drafter).propose(id, ns, &r.seq.tokens, want, &mut self.draft);
            }
            plan.draft_ns += started.elapsed().as_nanos() as u64;
            // A drafter copies what followed a match, an image's rows included; a guess is text,
            // and the check step would read an image row past every span.
            let text = (self.draft.iter()).take_while(|&&t| t & IMAGE_TOKEN == 0);
            let got = text.count();
            k = match (probe, self.draft_conf.is_empty()) {
                (true, _) => got.min(want),
                (false, true) => pick(a, got.min(want), plan.explore),
                (false, false) => {
                    let (f, c) = (r.hits.calibration(), &self.draft_conf);
                    let e = |k| chained(c, f, k);
                    (self.costs).pick_by(
                        plan.decoding,
                        plan.guessed,
                        plan.gain,
                        e,
                        got,
                        plan.explore,
                    )
                }
            };
        }
        let need = blocks_for(r.seq.n_kv + 1 + k).saturating_sub(r.seq.blocks.len());
        if k > 0 && need + self.running.len() > self.blocks.free.len() {
            k = 0;
        }
        self.tot.guess_k[k] += 1;
        if k == 0 {
            return 0;
        }
        r.seq.tokens.extend_from_slice(&self.draft[..k]);
        r.guesses = k;
        r.conf.clear();
        r.conf.extend(self.draft_conf.iter().take(k));
        plan.gain += match r.conf.is_empty() {
            true => expected(a, k),
            false => chained(&r.conf, r.hits.calibration(), k),
        } - 1.0;
        k
    }

    /// A model refused request `id`'s check: it runs as the decode row it would have been, and the
    /// round kept nothing, so a sequence that keeps finding no room backs off like one that misses.
    pub(crate) fn uncheck(&mut self, plan: &mut StepPlan, id: u64) {
        let r = self.reqs.get_mut(&id).expect("request");
        plan.guessed -= r.guesses;
        r.hits.refused(r.guesses as u32, 0);
        r.unguess();
        for b in r.seq.blocks.drain(blocks_for(r.seq.n_kv + 1)..) {
            self.blocks.release(b);
        }
        plan.prefill.retain(|p| p.0 != id);
        // In seat order, where `decode_rows` would have put it.
        let seat = |x: &u64| self.running.iter().position(|r| r == x);
        let at = plan
            .decode
            .iter()
            .take_while(|d| seat(d) < seat(&id))
            .count();
        plan.decode.insert(at, id);
        self.checkable.get_or_insert(false);
    }

    /// Continues chunked prompts, then starts waiting requests, oldest first, within the token budget.
    /// A chunk short of blocks waits, and nothing behind it jumps the queue; if nothing else would
    /// run, running sequences newer than it are preempted so the oldest prompt goes on. With a
    /// background request admitted, the foreground chunks and starts go first, then the background
    /// ones; a foreground start that waits holds back the background starts, not their chunks.
    fn prefill(&mut self, lim: &Limits, plan: &mut StepPlan, now: u64) {
        // A check step feeds no prompt (see `guess_room`, which plans none while one waits).
        if plan.guessed > 0 {
            return;
        }
        let mut budget = lim.prefill_chunk_tokens;
        if lim.policy == Policy::Balanced {
            budget = budget.saturating_sub(plan.decode.len());
        }
        // Every seat has at most one logits row unless observed; an observed chunk of `len` has up
        // to `len`, so its extra rows come out of what `logit_rows` holds beyond one a seat.
        let mut extra = lim.logit_rows.saturating_sub(lim.max_batch_seqs);
        if self.background == 0 {
            if self.chunks(plan, None, (&mut budget, &mut extra)) {
                self.starts(lim, plan, now, None, (&mut budget, &mut extra));
            }
            return;
        }
        let (fg, bg) = (Some(Class::Foreground), Some(Class::Background));
        if !self.chunks(plan, fg, (&mut budget, &mut extra)) {
            return;
        }
        let started = self.starts(lim, plan, now, fg, (&mut budget, &mut extra));
        if self.chunks(plan, bg, (&mut budget, &mut extra)) && started {
            self.starts(lim, plan, now, bg, (&mut budget, &mut extra));
        }
    }

    /// The most tokens of a chunk `id` takes this step: a pin read's is cut or held while pacing
    /// ([`Self::pin_piece`]); every other request's is not.
    fn piece(&self, id: u64) -> usize {
        if self.reqs[&id].reads_pin {
            self.pin_piece
        } else {
            usize::MAX
        }
    }

    /// Continues the chunked prompts of `class` (None: any) within the token budget and the extra
    /// logits rows; false when one waits for blocks.
    fn chunks(
        &mut self,
        plan: &mut StepPlan,
        class: Option<Class>,
        (budget, extra): (&mut usize, &mut usize),
    ) -> bool {
        let mut i = 0;
        while let Some(&id) = self.running.get(i) {
            if *budget == 0 {
                return true;
            }
            i += 1;
            let (n_kv, pending) = (self.reqs[&id].seq.n_kv, self.reqs[&id].seq.pending());
            // An activations request prefills its one-token remainder too (see `decode`). A
            // check is planned already.
            if pending == 0
                || self.reqs[&id].guesses > 0
                || (pending == 1 && (self.tapping == 0 || self.reqs[&id].activations.is_empty()))
                || class.is_some_and(|c| c != self.reqs[&id].class)
                || self.piece(id) == 0
            {
                continue;
            }
            let observe = self.reqs[&id].observe;
            let len = pending
                .min(*budget)
                .min(self.piece(id))
                .min(if observe {
                    (*extra + 1).min(OBSERVE_CHUNK)
                } else {
                    usize::MAX
                })
                .min(self.reqs[&id].to_snapshot(n_kv));
            // Another request shares (or is about to join) the step: it waits for one tile, not
            // an image. Zero: this one's image waits for the next step.
            let len = self.image_chunk(id, n_kv, len);
            if len == 0 {
                continue;
            }
            let mut fits = self.grow(id, n_kv + len, &mut plan.n_reclaimed);
            while !fits && plan.is_empty() {
                let at = self.running.iter().position(|&x| x == id).expect("running");
                let Some(victim) = self.victim(class, &self.running[at + 1..]) else {
                    break;
                };
                // An older victim moves this row, and the ones after it, one seat down.
                i -= usize::from(self.running[..at].contains(&victim));
                self.preempt(victim, plan);
                fits = self.grow(id, n_kv + len, &mut plan.n_reclaimed);
            }
            if !fits {
                return false;
            }
            plan.prefill.push((id, n_kv, len));
            self.planned_chunk(id, n_kv, len);
            *budget -= len;
            if observe {
                *extra -= len - 1;
            }
        }
        true
    }

    /// Starts waiting requests of `class` (None: any), in queue order, on free seats of the running
    /// set (4.5: max_batch_seqs bounds it); false when the first of them waits.
    fn starts(
        &mut self,
        lim: &Limits,
        plan: &mut StepPlan,
        now: u64,
        class: Option<Class>,
        (budget, extra): (&mut usize, &mut usize),
    ) -> bool {
        while *budget > 0 && self.running.len() < lim.max_batch_seqs {
            let reqs = &self.reqs;
            let Some(at) = self.waiting.iter().position(|id| {
                class.is_none_or(|c| c == reqs[id].class)
                    && !(self.hold && reqs[id].version.is_none())
                    && self.piece(*id) > 0
            }) else {
                return true;
            };
            let id = self.waiting[at];
            // An activations request waits for a tap seat, and nothing behind it jumps the queue.
            if self.tapping > TAP_SEATS && !self.reqs[&id].activations.is_empty() {
                let seated = self.running.iter();
                if seated
                    .filter(|r| !self.reqs[r].activations.is_empty())
                    .count()
                    >= TAP_SEATS
                {
                    return false;
                }
            }
            let observe = self.reqs[&id].observe;
            let most = (*budget)
                .min(if observe {
                    (*extra + 1).min(OBSERVE_CHUNK)
                } else {
                    usize::MAX
                })
                .min(self.piece(id));
            let Some((start, len)) = self.start(id, most, lim, &mut plan.n_reclaimed) else {
                return false;
            };
            if observe {
                *extra -= len - 1;
            }
            self.waiting.remove(at);
            self.running.push(id);
            if self.state_slots > 0 {
                // The parked state it took over, which a start at 0 resets.
                let held = self.reqs.get_mut(&id).expect("request").seq.state.take();
                let slot = (held.or_else(|| self.take_slot(&mut plan.n_reclaimed)))
                    .unwrap_or_else(|| self.steal_snapshot());
                let r = self.reqs.get_mut(&id).expect("request");
                r.state_slot = Some(slot);
                // A start past 0 with no parked state starts at a checkpoint.
                if held.is_none() && start > 0 {
                    let block = r.seq.blocks[start / BLOCK_TOKENS - 1];
                    let state = self.prefix.checkpoint(block).expect("fit found it");
                    plan.loads.push((slot, state));
                }
            }
            let r = self.reqs.get_mut(&id).expect("request");
            if r.t_admit == 0 {
                r.t_admit = now.max(1);
                self.tot.queue_ns += r.t_admit.saturating_sub(r.t_submit);
                self.tot.queued += 1;
                self.tot.prompt_tokens += r.prompt_len as u64;
            }
            r.snapshot(self.arm, true);
            // A parked session's one-token remainder (a resumed prefix, one token left to feed) joins
            // the decode batch instead of a one-token prefill (#64). A fresh one-token prompt still
            // prefills, and so does every remainder while a background budget below the whole batch
            // applies, so that `decode` alone counts the rows beside a foreground row.
            if start > 0
                && r.seq.pending() == 1
                && r.activations.is_empty()
                && (self.background == 0 || self.background_rows == BACKGROUND_ROWS)
            {
                plan.decode.push(id);
            } else {
                plan.prefill.push((id, start, len));
                self.planned_chunk(id, start, len);
            }
            *budget -= len;
        }
        true
    }

    /// Gives a queued request blocks for its first chunk, or leaves everything as it was: the head of
    /// the queue, or with a background request admitted the first of its class (see `starts`).
    ///
    /// The request takes over its session's parked sequence and keeps the longest common token prefix,
    /// bounded by the KV that exists and leaving one token to compute (9.3 step 2). The match is
    /// computed first and committed only once the chunk fits, so a request that waits leaves the
    /// parked sequence whole and reclaimable. A new start may not push the live (unparked) blocks over
    /// `kv_preempt_watermark`, which keeps headroom for the running sequences to grow; with nothing
    /// running it starts anyway. Under `prefill_first` it also leaves the blocks this step's decode
    /// rows still need.
    fn start(
        &mut self,
        id: u64,
        budget: usize,
        lim: &Limits,
        reclaimed: &mut u32,
    ) -> Option<(usize, usize)> {
        let Fit {
            global,
            reuse,
            retained,
            len,
            save_at,
            save_by,
            marks,
            indexed,
        } = loop {
            match self.fit(id, budget, lim) {
                Ok(fit) => break fit,
                // Short of blocks that pins hold: a pin gives way before the request waits for them.
                Err(true) if !self.reqs[&id].reads_pin && self.unpin_oldest() => {}
                Err(_) => return None,
            }
        };
        let r = &self.reqs[&id];
        let first = r.t_admit == 0;
        let turn = (first && r.session != 0).then(|| self.reuse(r, global, reuse, indexed));
        // Commit only after both capacity checks succeed. Pin lookup blocks before reclaiming.
        let r = self.reqs.get_mut(&id).expect("request");
        if global {
            for &block in &retained {
                self.blocks.retain(block);
            }
            self.prefix.touch(&retained);
            r.seq.blocks = retained;
            r.seq.n_kv = reuse;
            if let Some(mut p) = self.parked.remove(&r.session) {
                free(&mut self.blocks, &mut p);
                self.free_slots.extend(p.state);
            }
        } else if let Some(mut p) = self.parked.remove(&r.session) {
            truncate(&mut self.blocks, &mut p, reuse);
            p.tokens = std::mem::take(&mut r.seq.tokens);
            r.seq = p;
        }
        r.seq.generation = self.generation;
        r.version = Some(r.version.unwrap_or(self.version));
        r.seq.namespace = (r.namespace, r.version.unwrap_or(self.version));
        if let Some(turn) = turn {
            r.reuse = turn;
        }
        r.seq.note = Note {
            turn: r.reuse.turn,
            ..Note::default()
        };
        if r.t_admit == 0 {
            r.cached = reuse;
            self.reused_prompt_tokens += reuse as u64;
            if self.prefix.capacity() > 0
                && r.prefix_cache
                && !r.observe
                && (r.session == 0 || r.keep_kv)
            {
                self.cache_lookups += 1;
            }
            if global {
                self.cache_hits += 1;
            }
        }
        r.save_at = save_at;
        r.save_by = save_by;
        r.pin_mark = marks;
        let session = r.session;
        if turn.is_some() {
            self.unforget(session);
        }
        assert!(
            self.grow(id, reuse + len, reclaimed),
            "checked before acquisition"
        );
        Some((reuse, len))
    }

    /// The match [`Self::start`] would commit for request `id`, and the chunk of at most `budget`
    /// tokens it would get, or `Err` when that chunk does not fit (see `start`): true when only
    /// blocks are short. Changes nothing.
    fn fit(&self, id: u64, budget: usize, lim: &Limits) -> Result<Fit, bool> {
        let r = &self.reqs[&id];
        let prompt = &r.seq.tokens;
        let parked = self.parked.get(&r.session);
        let session_reuse = parked.map_or(0, |p| {
            if p.generation != self.generation || p.namespace != self.ns(r) {
                return 0;
            }
            let lcp = p
                .tokens
                .iter()
                .zip(prompt)
                .take_while(|(a, b)| a == b)
                .count();
            private_reuse(&self.blocks, p, lcp.min(p.n_kv).min(prompt.len() - 1))
        });
        // Initial same-session observation deliberately observes only the residual positions;
        // global adoption is disabled for observation, which must never skip required positions.
        let session_reuse = if r.observe && r.t_admit != 0 {
            session_reuse.min(r.next_obs)
        } else {
            session_reuse
        };
        // A recurrent state stands at the parked sequence's end: all of it or none.
        let session_reuse = match parked {
            Some(p) if p.state.is_some() && session_reuse < p.n_kv => 0,
            _ => session_reuse,
        };
        let mut chain = if self.eligible(r) {
            self.prefix.lookup(self.ns(r), prompt)
        } else {
            Vec::new()
        };
        let indexed = chain.len() * BLOCK_TOKENS;
        // A recurrent state stands where a checkpoint holds it, and a whole parked one goes first.
        if self.state_slots > 0 {
            chain.truncate(match session_reuse {
                0 => self.prefix.checkpointed(&chain),
                _ => 0,
            });
        }
        let global = chain.len() * BLOCK_TOKENS > session_reuse;
        let reuse = if global {
            chain.len() * BLOCK_TOKENS
        } else {
            session_reuse
        };
        let retained: Vec<_> = if global {
            chain
        } else {
            self.parked
                .get(&r.session)
                .map_or(Vec::new(), |p| p.blocks[..blocks_for(reuse)].to_vec())
        };
        // A hybrid saves where the chain it found outran its checkpoints, and by where its own
        // repeat can look up to.
        let publishable = r.publishable();
        // A pinned document's read saves at its last whole block, even one that ends the prompt: a
        // question after it repeats the whole chain, which `lookup` bounds by its own last token.
        let pin = self.prefix.is_pin(r.namespace, prompt);
        let last = match pin {
            true => prompt.len(),
            false => prompt.len() - 1,
        };
        // On a block boundary it saves one block earlier too: its own repeat looks up only that far,
        // and would find no checkpoint and read the document again.
        let before_end = match pin && prompt.len().is_multiple_of(BLOCK_TOKENS) {
            true => prompt.len() - BLOCK_TOKENS,
            false => 0,
        };
        let marks = if pin && self.state_slots > 0 && self.eligible(r) {
            self.pin_mark
        } else {
            0
        };
        let [save_at, save_by] = match self.state_slots > 0 && self.eligible(r) {
            true => [
                indexed.max(before_end),
                publishable.min(last) / BLOCK_TOKENS * BLOCK_TOKENS,
            ]
            .map(|at| {
                if at > reuse && at <= publishable {
                    at
                } else {
                    0
                }
            }),
            false => [0; 2],
        };
        let cut_by = if r.parks_as() == 0 { save_by } else { 0 };
        // Alone, a request reads its images in whole chunks (fewer, larger steps); beside running
        // requests, a chunk ends where a tile does, so they wait for one tile (`image_chunk`).
        let len = (prompt.len() - reuse).min(budget).min(to_stop(
            [save_at, cut_by, r.history_end, next_mark(marks, reuse)],
            reuse,
        ));
        let len = self.image_chunk(id, reuse, len);
        if len == 0 {
            return Err(false);
        }
        let need = blocks_for(reuse + len).saturating_sub(retained.len());
        let live: HashSet<_> = self
            .reqs
            .values()
            .flat_map(|r| r.seq.blocks.iter().copied())
            .collect();
        let reserved: HashSet<_> = retained.iter().copied().collect();
        let newly_live = retained.iter().filter(|b| !live.contains(b)).count() + need;
        let high = (lim.kv_preempt_watermark * self.blocks.total() as f64) as usize;
        let room = self.blocks.free.len() + self.reclaimable(&reserved);
        // Under prefill_first the decode rows grow after the starts: a start leaves each of them its
        // next block, so `decode` never takes back a start (#196). The scan runs only past a
        // conservative bound (every running row needing a block), so it also runs on some steps
        // where every row fits; `scheduler_pressure_workload` measures it.
        // A foreground start leaves blocks to foreground rows only: background rows yield to it.
        let decoding = if lim.policy == Policy::PrefillFirst && need + self.running.len() > room {
            let fg = self.background > 0 && r.class == Class::Foreground;
            let rows = self.running.iter().map(|q| &self.reqs[q]);
            rows.filter(|q| !fg || q.class == Class::Foreground)
                .filter(|q| q.seq.pending() == 1 && blocks_for(q.seq.n_kv + 1) > q.seq.blocks.len())
                .count()
        } else {
            0
        };
        if need > room {
            // True when pins going cold would make the room ([`Self::start`]).
            let with_pins = self.blocks.free.len() + self.reclaimable_with(&reserved, true);
            return Err(need <= with_pins);
        }
        if need + decoding > room || (!self.running.is_empty() && live.len() + newly_live > high) {
            return Err(false);
        }
        Ok(Fit {
            global,
            reuse,
            retained,
            len,
            save_at,
            save_by,
            marks,
            indexed,
        })
    }

    /// What session turn `r` matches before its first start commits `cached` tokens of reuse,
    /// from the shared cache when `global`: S03's counts, with `cached <= available <= lcp`, and
    /// `available <= parked` when the parked KV bounds it.
    fn reuse(&self, r: &Req, global: bool, cached: usize, indexed: usize) -> Reuse {
        let prompt = &r.seq.tokens;
        let p = self.parked.get(&r.session);
        let note = p.map(|p| &p.note).or_else(|| self.gone.get(&r.session));
        let fed = p.map_or(&[][..], |p| &p.tokens);
        let tail = note.map_or(&[][..], |n| &n.tail);
        let lcp = fed
            .iter()
            .chain(tail)
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        let parked = p.map_or(0, |p| p.n_kv);
        let valid = p.is_some_and(|p| p.generation == self.generation && p.namespace == self.ns(r));
        let available = match (global, valid) {
            // A hybrid's chain past its last checkpoint was there, but not reusable.
            (true, _) => indexed.max(cached),
            (false, true) => lcp.min(parked).min(prompt.len() - 1),
            (false, false) => 0,
        };
        let miss = match (p, note) {
            (None, None) => ReuseMiss::FirstTurn,
            // The lost KV would have reused nothing (submit refuses an empty prompt).
            (None, Some(n)) if lcp == 0 && !n.tail.is_empty() => ReuseMiss::PrefixDiverged,
            (None, Some(n)) => n.miss.unwrap_or(ReuseMiss::Reset),
            (Some(p), _) if p.generation != self.generation => ReuseMiss::Reloaded,
            // A new cache namespace is the caller's choice, like a reset; so is a new version.
            (Some(p), _) if p.namespace != self.ns(r) => ReuseMiss::Reset,
            // Only a reclaim leaves history past the parked KV.
            (Some(p), _) if lcp > parked => p.note.miss.unwrap_or(ReuseMiss::Reclaimed),
            _ if lcp < (fed.len() + tail.len()).min(prompt.len()) => ReuseMiss::PrefixDiverged,
            _ if cached < available => ReuseMiss::MatchFailure,
            _ => ReuseMiss::None,
        };
        Reuse {
            turn: self.last_turn(r.session) + 1,
            lcp: if global { lcp.max(cached) } else { lcp },
            parked,
            available,
            prior: !global && p.is_some(),
            miss,
            reclaim: note.map_or(0, |n| n.reclaim),
        }
    }

    /// The step's logits rows, in batch order: every decode row, the last row of each prefill chunk
    /// that reaches the end of its tokens (unless an activations request's, which samples nothing),
    /// and every unreported row of an observed request.
    pub(crate) fn logits(&self, plan: &StepPlan, out: &mut Vec<Logit>) {
        let observed = |r: &Req, pos: usize| r.observe && pos >= r.next_obs;
        out.clear();
        out.extend(plan.decode.iter().enumerate().map(|(row, &id)| {
            let pos = self.reqs[&id].seq.n_kv;
            let observe = observed(&self.reqs[&id], pos);
            Logit {
                id,
                row,
                pos,
                observe,
                sample: true,
            }
        }));
        let mut row = plan.decode.len();
        for &(id, start, len) in &plan.prefill {
            let r = &self.reqs[&id];
            let last = start + len - 1;
            let ends = last + 1 == r.seq.tokens.len();
            // Only an observed request or a check has rows before `last`; a check samples them all.
            let first = if r.observe {
                start.max(r.next_obs).min(last)
            } else if r.guesses > 0 {
                start
            } else {
                last
            };
            for pos in first..=last {
                let sample = ends && (pos == last || r.guesses > 0) && r.activations.is_empty();
                let observe = observed(r, pos);
                if observe || sample {
                    out.push(Logit {
                        id,
                        row: row + pos - start,
                        pos,
                        observe,
                        sample,
                    });
                }
            }
            row += len;
        }
    }

    /// Records what the step fed and appends the sampled tokens (one per sampling [`Self::logits`] row,
    /// except that a check emits only its kept guesses and the token after them), streaming each and
    /// ending requests on a stop token or `max_tokens`. A check keeps the KV of its rows up to its
    /// last kept guess; the rejected rows' KV is past `n_kv`, as if never written.
    pub(crate) fn complete(
        &mut self,
        plan: &StepPlan,
        sampled: impl IntoIterator<Item = (u64, SampledToken)>,
        now: u64,
    ) {
        let sampled: Vec<_> = sampled.into_iter().collect();
        let mut checked = Vec::new();
        for &(id, start, _) in &plan.prefill {
            let r = self.reqs.get_mut(&id).expect("request");
            if r.guesses == 0 {
                continue;
            }
            checked.push(id);
            let emitted = sampled.iter().filter(|s| s.0 == id).map(|s| s.1.id);
            let fed = emitted.clone().count();
            let kept = (emitted.zip(&r.seq.tokens[start + 1..]))
                .take_while(|(t, g)| t == *g)
                .count();
            r.hits.round(r.guesses as u32, kept as u32);
            r.hits.scored(&r.conf[..r.conf.len().min(kept + 1)]);
            self.accept.round(r.guesses as u32, kept as u32);
            self.tot.guess_rounds += 1;
            self.tot.guesses_tried += r.guesses as u64;
            self.tot.guesses_kept += kept as u64;
            r.unguess();
            // Rows 0..fed hold the last token and the kept guesses (the last emitted token is
            // never fed, as after a decode row).
            r.seq.n_kv = start + fed;
            self.blocks.completed(&r.seq, start, r.seq.n_kv);
            for b in r.seq.blocks.drain(blocks_for(r.seq.n_kv)..) {
                self.blocks.release(b);
            }
        }
        for &id in &plan.decode {
            let r = &self.reqs[&id];
            self.computed_prompt_tokens += u64::from(r.seq.n_kv < r.prompt_len);
            self.tot.recomputed_tokens += u64::from(r.seq.n_kv < r.redo);
            self.blocks.completed(&r.seq, r.seq.n_kv, r.seq.n_kv + 1);
            self.reqs.get_mut(&id).expect("request").seq.n_kv += 1;
        }
        for &(id, start, len) in plan.prefill.iter().filter(|p| !checked.contains(&p.0)) {
            self.computed_prompt_tokens += ((start + len).min(self.reqs[&id].prompt_len)
                - start.min(self.reqs[&id].prompt_len))
                as u64;
            self.tot.recomputed_tokens +=
                (start + len).min(self.reqs[&id].redo).saturating_sub(start) as u64;
            self.blocks
                .completed(&self.reqs[&id].seq, start, start + len);
            let r = self.reqs.get_mut(&id).expect("request");
            r.seq.n_kv = start + len;
            // One per chunk, cumulative; a recomputed chunk after a preemption reports nothing.
            let done = (start + len).min(r.prompt_len);
            if r.progress && !r.observe && done > r.redo {
                let total = r.prompt_len as u32;
                let event = EngineEvent::Prefilled {
                    done: done as u32,
                    total,
                };
                r.completion
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .token(event);
            }
        }
        for id in plan
            .decode
            .iter()
            .copied()
            .chain(plan.prefill.iter().map(|p| p.0))
        {
            let r = &self.reqs[&id];
            if r.seq.n_kv <= r.prompt_len
                && self.eligible(r)
                && r.seq.generation == self.generation
                && !r.cancel.load(Ordering::Relaxed)
            {
                let r = self.reqs.get_mut(&id).expect("request");
                let n = r.seq.n_kv.min(r.publishable());
                self.prefix.publish(
                    &mut self.blocks,
                    r.seq.namespace,
                    r.session,
                    r.id,
                    &r.seq.tokens[..n],
                    &mut r.seq.blocks,
                );
                let n = r.seq.n_kv;
                // The chunk that just ended here; a decode row's is one token.
                let len = (plan.prefill.iter().find(|p| p.0 == id)).map_or(1, |p| p.2);
                let by = n.is_multiple_of(BLOCK_TOKENS) && n <= r.save_by && r.save_by - n < len;
                if let Some(slot) = r.state_slot
                    && n > 0
                    && (n == r.save_at || by)
                {
                    let block = r.seq.blocks[n / BLOCK_TOKENS - 1];
                    // Identical prompts in one step publish onto one block: one save.
                    if self.prefix.wants_checkpoint(block)
                        && !self.saves.iter().any(|&(_, b)| b == block)
                    {
                        self.saves.push((slot, block));
                    }
                }
                // A pin read's mark: its checkpoint stands where the last one did.
                if let Some(slot) = r.state_slot
                    && r.pin_mark > 0
                    && n.is_multiple_of(r.pin_mark)
                    && n < r.prompt_len
                {
                    let block = r.seq.blocks[n / BLOCK_TOKENS - 1];
                    if self.prefix.wants_checkpoint(block)
                        && !self.saves.iter().any(|&(_, b)| b == block)
                    {
                        self.saves.push((slot, block));
                    }
                    if let Some(old) = r.mark.replace(block) {
                        self.retire.push((block, old));
                    }
                }
            }
        }
        // An activations request ends once its prompt is fed; the engine has set its rows, its
        // first output, which its span times as a first token (no Ok span lacks one).
        let tapped = if self.tapping > 0 {
            &plan.prefill[..]
        } else {
            &[]
        };
        for &(id, ..) in tapped {
            let r = self.reqs.get_mut(&id).expect("request");
            if !r.activations.is_empty() && r.seq.pending() == 0 {
                let end = if r.untapped {
                    Err(EngineError::InvalidRequest(crate::invalid::NO_TAPS))
                } else {
                    r.t_first = now.max(1);
                    r.arm = Some(self.arm);
                    Ok(Finish::Length)
                };
                self.finish(id, end, now);
            }
        }
        for (id, t) in sampled {
            let r = self.reqs.get_mut(&id).expect("request");
            let fed = r.forced.get(r.generated as usize).copied().unwrap_or(t.id);
            r.seq.tokens.push(fed);
            r.generated += 1;
            self.tot.gen_tokens += 1;
            if r.t_first == 0 {
                r.t_first = now.max(1);
                r.arm = Some(self.arm);
            } else {
                r.max_gap = r.max_gap.max(now.saturating_sub(r.t_last));
            }
            r.t_last = now;
            let finish = if !r.ignore_eos && r.stop_tokens.contains(&t.id) {
                Some(Finish::Eos)
            } else if r.generated >= r.max_tokens {
                Some(Finish::Length)
            } else {
                None
            };
            {
                let mut completion = r.completion.lock().unwrap_or_else(|e| e.into_inner());
                completion.snapshot(r.span(Ok(Finish::None), 0), r.arm.unwrap_or(self.arm), true);
                if !r.observe {
                    completion.token(EngineEvent::Token {
                        id: t.id,
                        logprob: t.logprob,
                        top: t.top,
                    });
                }
            }
            if let Some(f) = finish {
                self.finish(id, Ok(f), now);
            }
        }
    }

    /// Stops only the matching active request; a late acknowledgement cannot touch newer KV.
    pub(crate) fn finish_stop(&mut self, id: u64, now: u64) {
        if let Some(r) = self.reqs.get(&id) {
            // A stop before any sampled token is a cancellation: there is no first token to time.
            let end = if r.cancel.load(Ordering::Relaxed) || r.generated == 0 {
                Err(EngineError::Cancelled)
            } else {
                Ok(Finish::StopSeq)
            };
            self.finish(id, end, now);
        }
    }

    /// Ends a request: Done parks its sequence when it keeps KV under a live session and was not
    /// cancelled; otherwise (and on Error) the sequence is freed.
    pub(crate) fn finish(&mut self, id: u64, end: Result<Finish, EngineError>, now: u64) {
        let mut r = self.reqs.remove(&id).expect("request");
        // One completion boundary for both global withdrawal and session parking.
        let cancelled = r.cancel.load(Ordering::Relaxed);
        if self.guess_max > 0 && r.may_draft() {
            // Retained only where the prefix cache would (it needs prefix_cache_blocks); otherwise the drafter just frees
            // the request's own automaton.
            // ponytail: a released session's earlier outputs stay until evicted (64k tokens): the
            // drafter keeps no session id per output, and forgetting needs a tag and a rebuild of
            // the namespace automaton; add both if release must withdraw drafts, not only KV.
            let keep = end.is_ok() && !cancelled && self.eligible(&r);
            let output = &r.seq.tokens[r.prompt_len.min(r.seq.tokens.len())..];
            let output = if keep { output } else { &[] };
            self.drafter.ended(id, r.seq.namespace, output);
        }
        self.running.retain(|&x| x != id);
        self.waiting.retain(|&x| x != id);
        if end.is_err() || cancelled {
            self.prefix.cancel(id, &mut self.blocks);
        } else if r.session == 0 && self.eligible(&r) && r.seq.generation == self.generation {
            // Only the blocks this request holds, so it adopts no chain it did not read.
            let prompt = &r.seq.tokens[..r.prompt_len.min(r.seq.tokens.len())];
            self.prefix.adopt(r.seq.namespace, prompt, &r.seq.blocks);
        }
        // n_kv 0 (preempted for recompute): parking that would free the session's parked sequence.
        let park = end.is_ok()
            && r.seq.n_kv > 0
            && r.keep_kv
            && r.session != 0
            && !cancelled
            && !self.released.contains(&r.session);
        if park {
            // A recurrent state parks with its sequence: the snapshot, the sequence cut back to it,
            // else the state at the turn's end.
            if let Some(slot) = r.snap.take() {
                truncate(&mut self.blocks, &mut r.seq, r.history_end);
                r.seq.state = Some(slot);
            } else {
                r.seq.state = r.state_slot.take();
            }
            r.seq.tokens.truncate(r.seq.n_kv); // the last sampled token was never fed
            r.seq.last_used = now;
            // A later turn of the session may have started (and parked) while this one ran.
            r.seq.note.turn = r.seq.note.turn.max(self.last_turn(r.session));
            if let Some(mut old) = self.parked.remove(&r.session) {
                free(&mut self.blocks, &mut old);
                self.free_slots.extend(old.state);
            }
            if self.prefix.capacity() > 0 {
                // Sharing removes the old implicit bound on parked CPU token histories. Bound
                // their block-table references separately from unique physical KV occupancy:
                // at most one pool's worth of references and therefore <= pool * 16 tokens.
                let mut held: usize = self.parked.values().map(|p| p.blocks.len()).sum();
                while held + r.seq.blocks.len() > self.blocks.total() {
                    let session = *self
                        .parked
                        .iter()
                        .min_by_key(|(session, p)| (p.last_used, **session))
                        .expect("one sequence fits the pool")
                        .0;
                    let mut old = self.parked.remove(&session).expect("parked");
                    held -= old.blocks.len();
                    free(&mut self.blocks, &mut old);
                    self.free_slots.extend(old.state);
                    old.tokens.append(&mut old.note.tail);
                    self.forget(
                        session,
                        old.note.turn,
                        ReuseMiss::OverflowDrop,
                        old.note.reclaim,
                        old.tokens,
                    );
                }
            }
            self.parked.insert(r.session, std::mem::take(&mut r.seq));
        } else {
            free(&mut self.blocks, &mut r.seq);
            // A turn that never started took nothing over: its session's record stays as it was.
            if r.session != 0 && r.t_admit != 0 {
                let miss = match end {
                    _ if cancelled => ReuseMiss::Cancelled,
                    Err(EngineError::Cancelled) => ReuseMiss::Cancelled,
                    Err(_) => ReuseMiss::EngineError,
                    Ok(_) => ReuseMiss::Reset,
                };
                let turn = r.reuse.turn.max(self.last_turn(r.session));
                self.forget(r.session, turn, miss, 0, Vec::new());
            }
        }
        self.free_slots.extend(r.state_slot.take());
        self.free_slots.extend(r.snap.take());
        self.end(r, end, now);
    }

    /// Sends the terminal event (an activations request's rows in its Done) and records the span;
    /// touches no KV.
    fn end(&mut self, mut r: Req, end: Result<Finish, EngineError>, now: u64) {
        let span = r.span(end, now);
        self.tot.finished += u64::from(end.is_ok());
        self.background -= usize::from(r.class == Class::Background);
        let mut completion = r.completion.lock().unwrap_or_else(|e| e.into_inner());
        if !r.activations.is_empty() {
            self.tapping -= 1;
            completion.activations = std::mem::take(&mut r.rows);
        }
        completion.grammar_forced = r.grammar.as_ref().map_or(0, |g| g.forced);
        completion.adapter_version = r.version.unwrap_or(self.version);
        let fresh = completion.finish(span, r.arm.unwrap_or(self.arm), end);
        drop(completion);
        if fresh {
            self.ended.push((span, r.arm.unwrap_or(self.arm)));
        }
    }

    /// Ends every request without touching the KV accounting, which may be what just faulted (9.11):
    /// running ones with `running`, queued ones with `queued`. One with no token yet carries `arm`.
    pub(crate) fn fail_all(
        &mut self,
        running: EngineError,
        queued: EngineError,
        arm: u8,
        now: u64,
    ) {
        for (id, mut r) in std::mem::take(&mut self.reqs) {
            r.arm = r.arm.or(Some(arm));
            let e = if self.running.contains(&id) {
                running
            } else {
                queued
            };
            self.end(r, Err(e), now);
        }
        self.running.clear();
        self.waiting.clear();
        let parked: HashSet<u32> = self.parked.values().filter_map(|p| p.state).collect();
        self.free_slots = (0..self.state_slots)
            .rev()
            .filter(|s| !parked.contains(s))
            .collect();
    }
}

#[cfg(test)]
mod tests {
    //! The 9.3 invariants and the six D14 ones, after every operation of random admits, steps,
    //! cancels and releases, plus scenario tests (reclaim_keeps_prefix at full scale).

    use std::sync::mpsc::{Receiver, channel};

    use proptest::prelude::*;

    use super::*;
    use crate::Usage;

    const STOP: u32 = 7;

    impl Scheduler {
        /// Checks the 9.3 invariants: every block is owned once, and parked sequences are consistent.
        fn check(&self) {
            let owned = self
                .reqs
                .values()
                .map(|r| &r.seq)
                .chain(self.parked.values());
            let mut counts = vec![0u32; self.blocks.total()];
            for seq in owned {
                assert!(
                    seq.n_kv <= seq.tokens.len() && seq.n_kv <= seq.blocks.len() * BLOCK_TOKENS
                );
                let mut seen = HashSet::new();
                for (i, &block) in seq.blocks.iter().enumerate() {
                    assert!(
                        seen.insert(block),
                        "same physical block twice in one sequence"
                    );
                    assert!(!self.blocks.is_free[block as usize], "owned block free");
                    counts[block as usize] += 1;
                    if self.blocks.refs[block as usize] > 1 {
                        assert!(
                            seq.n_kv >= (i + 1) * BLOCK_TOKENS,
                            "shared block has a mutable tail"
                        );
                    }
                }
            }
            for block in self.prefix.blocks() {
                counts[block as usize] += 1;
            }
            assert_eq!(
                counts, self.blocks.refs,
                "all references must have exactly one owner"
            );
            assert_eq!(
                counts.iter().filter(|&&n| n > 0).count(),
                self.blocks.used()
            );
            assert!(self.prefix.len() - self.prefix.pinned() <= self.prefix.capacity());
            let tapping = self.reqs.values().filter(|r| !r.activations.is_empty());
            assert_eq!(tapping.count(), self.tapping, "live activations requests");
            let background = self.reqs.values().filter(|r| r.class == Class::Background);
            assert_eq!(
                background.count(),
                self.background,
                "live background requests"
            );
            let seated = self.running.iter();
            let seated = seated.filter(|id| !self.reqs[id].activations.is_empty());
            assert!(seated.count() <= TAP_SEATS, "activations past their seats");
            assert!(
                self.parked.values().map(|p| p.blocks.len()).sum::<usize>() <= self.blocks.total(),
                "parked host metadata must remain pool bounded"
            );
            assert!(self.used_tokens() <= self.blocks.used() * BLOCK_TOKENS);
            let mut physical_tokens = vec![0usize; self.blocks.total()];
            for seq in self
                .reqs
                .values()
                .map(|r| &r.seq)
                .chain(self.parked.values())
            {
                for (i, &block) in seq.blocks.iter().enumerate() {
                    physical_tokens[block as usize] = physical_tokens[block as usize]
                        .max(seq.n_kv.saturating_sub(i * BLOCK_TOKENS).min(BLOCK_TOKENS));
                }
            }
            for block in self.prefix.blocks() {
                physical_tokens[block as usize] = BLOCK_TOKENS;
            }
            assert_eq!(physical_tokens.iter().sum::<usize>(), self.used_tokens());
            for s in self.parked.values() {
                assert_eq!(
                    s.tokens.len(),
                    s.n_kv,
                    "a parked sequence holds exactly its fed tokens"
                );
                assert!(blocks_for(s.n_kv) <= s.blocks.len());
            }
            assert!(
                self.parked.keys().all(|s| !self.released.contains(s)),
                "parked under a released id"
            );
            assert!(
                self.waiting
                    .iter()
                    .all(|id| self.reqs[id].seq.blocks.is_empty()),
                "a waiting request holds KV"
            );
            // Each running request holds a state slot of its own, and only while it runs; a slot is
            // free, a running request's, its snapshot's or a parked sequence's, never two of them.
            let mut slots: Vec<u32> = self.free_slots.clone();
            for r in self.reqs.values() {
                let running = self.running.contains(&r.id);
                assert_eq!(r.state_slot.is_some(), running && self.state_slots > 0);
                assert!(
                    r.snap.is_none() || running,
                    "a snapshot off the running set"
                );
                assert!(r.seq.state.is_none(), "a request holds a parked state");
                slots.extend(r.state_slot);
                slots.extend(r.snap);
            }
            for p in self.parked.values() {
                assert_eq!(p.state.is_some(), self.state_slots > 0, "a parked state");
                slots.extend(p.state);
            }
            slots.sort_unstable();
            assert_eq!(
                slots,
                (0..self.state_slots).collect::<Vec<_>>(),
                "state slots"
            );
        }
    }

    fn req(
        id: u64,
        prompt: Vec<u32>,
        max_tokens: u32,
        session: u64,
    ) -> (Req, Receiver<EngineEvent>, Arc<AtomicBool>) {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let r = Request {
            stop_tokens: vec![STOP],
            session,
            keep_kv: session != 0,
            ..Request::new(prompt, max_tokens)
        };
        (Req::new(id, r, tx, cancel.clone(), 1), rx, cancel)
    }

    #[test]
    fn an_image_chunk_ends_where_its_tile_does() {
        // `to_tile_end` reads the span's place and sizes; the image only has to exist.
        let png = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../larch-model/tests/data/vision/palette.png"
        ))
        .unwrap();
        let image = std::sync::Arc::new(larch_model::vision::Image::new(png.into()).unwrap());
        let (mut r, _rx, _cancel) = req(1, vec![1; 40], 1, 0);
        r.images = vec![crate::ImageSpan {
            at: 10,
            rows: 20,
            tile_rows: 8,
            image,
            pixels: None,
        }];
        // The text before the image and its first tile; from inside a tile, to that tile's end; the
        // last tile is short; past the image, no limit.
        assert_eq!(r.to_tile_end(0), 18);
        assert_eq!(r.to_tile_end(20), 6);
        assert_eq!(r.to_tile_end(27), 3);
        assert_eq!(r.to_tile_end(30), usize::MAX);
        // To the image's first row; inside it, nothing; past it, no limit.
        assert_eq!(
            (r.to_image(0), r.to_image(12), r.to_image(30)),
            (10, 0, usize::MAX)
        );
        // Alone: a chunk ending inside a tile ends where the tile starts, unless it started there.
        assert_eq!(r.tile_aligned(0, 21), 18);
        assert_eq!(r.tile_aligned(0, 18), 18);
        assert_eq!(r.tile_aligned(0, 12), 10);
        assert_eq!(r.tile_aligned(19, 4), 4);
        assert_eq!(r.tile_aligned(0, 35), 35);
        assert!(r.reads_image(5, 6) && !r.reads_image(0, 10) && !r.reads_image(30, 10));
    }

    fn image_req(id: u64) -> (Req, Receiver<EngineEvent>, Arc<AtomicBool>) {
        let png = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../larch-model/tests/data/vision/palette.png"
        ))
        .unwrap();
        let image = Arc::new(larch_model::vision::Image::new(png.into()).unwrap());
        let (mut r, rx, cancel) = req(id, vec![1; 40], 1, 0);
        r.images = vec![crate::ImageSpan {
            at: 10,
            rows: 20,
            tile_rows: 8,
            image,
            pixels: None,
        }];
        (r, rx, cancel)
    }

    /// Two requests with images in one step: one encodes a tile, the other feeds only its text, so
    /// the step (and every row beside it) waits for one tile, not two. Alone, a chunk is not capped.
    #[test]
    fn one_image_tile_a_step_however_many_requests_carry_images() {
        let mut s = Scheduler::new(64);
        let limits = lim(Policy::DecodeFirst, 8, 512, 0.95);
        let mut held = Vec::new();
        for id in [1, 2] {
            let (r, rx, cancel) = image_req(id);
            held.push((rx, cancel));
            s.admit(r);
        }
        let mut t = 0;
        let plan = step(&mut s, &limits, &[90], &mut t);
        assert_eq!(plan.prefill, [(1, 0, 18), (2, 0, 10)]);
        let plan = step(&mut s, &limits, &[90], &mut t);
        assert_eq!(
            plan.prefill,
            [(1, 18, 8)],
            "the second waits while the first reads a tile"
        );
        let mut alone = Scheduler::new(64);
        let (r, rx, cancel) = image_req(3);
        held.push((rx, cancel));
        alone.admit(r);
        let plan = step(&mut alone, &limits, &[90], &mut t);
        assert_eq!(plan.prefill, [(3, 0, 40)]);
    }

    fn lim(policy: Policy, max_batch_seqs: usize, chunk: usize, watermark: f64) -> Limits {
        Limits {
            max_batch_seqs,
            prefill_chunk_tokens: chunk,
            policy,
            kv_preempt_watermark: watermark,
            logit_rows: 64,
        }
    }

    #[test]
    fn automatic_stateless_and_cross_session_prefix_hits() {
        let mut s = Scheduler::new(128);
        let limits = lim(Policy::DecodeFirst, 8, 512, 0.95);
        let mut clock = 0;
        // Explicit global-cache retention: unrelated request IDs, including session zero.
        s.configure_prefix_cache(24);
        for (id, session) in [(1, 0), (2, 42), (3, 0)] {
            let (r, events, _) = req(id, (0..65).collect(), 1, session);
            s.admit(r);
            run_all(&mut s, &limits, &[90], &mut clock);
            let cached = events
                .try_iter()
                .find_map(|e| match e {
                    EngineEvent::Done { usage, .. } => Some(usage.cached_tokens),
                    _ => None,
                })
                .unwrap();
            assert_eq!(cached, if id == 1 { 0 } else { 64 });
        }
    }

    /// A hybrid saves its state where its prompt's chain ends, and where the chain it found
    /// outran the checkpoints it found; a later prompt sharing either loads it and computes only
    /// the rest. A divergence inside a block reads back to the last checkpoint before it.
    #[test]
    fn hybrid_checkpoints_save_at_chain_ends_and_load() {
        let mut s = Scheduler::new(128);
        s.set_state_slots(4);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        let system: Vec<u32> = (0..40).collect();
        let ask = |s: &mut Scheduler, t: &mut u64, id, tail: &[u32]| {
            let (r, rx, _) = req(id, [&system[..], tail].concat(), 1, 0);
            s.admit(r);
            let mut loads = 0;
            while !s.is_idle() {
                loads += step(s, &limits, &[90], t).loads.len();
            }
            (cached(&rx), loads)
        };
        // 65 tokens: one checkpoint, at 64.
        assert_eq!(ask(&mut s, &mut t, 1, &[50; 25]), (0, 0));
        assert_eq!(
            ask(&mut s, &mut t, 2, &[50; 25]),
            (64, 1),
            "the same prompt"
        );
        // Its chain covers 32 tokens of the next, past every checkpoint: it saves there.
        assert_eq!(ask(&mut s, &mut t, 3, &[60; 25]), (0, 0));
        assert_eq!(
            ask(&mut s, &mut t, 4, &[70; 25]),
            (32, 1),
            "a shared system prompt"
        );
        // Diverging at 33, inside the third block: back to the checkpoint at 32.
        assert_eq!(ask(&mut s, &mut t, 5, &[]), (32, 1));
        // Without the prefix cache's consent, nothing is saved or loaded.
        s.configure_prefix_cache(0);
        assert_eq!(ask(&mut s, &mut t, 6, &[50; 25]), (0, 0));
    }

    /// A prompt's prompt-end checkpoint stands where its own repeat can look up to: a 64-token
    /// prompt's repeat loads the state at 48. A request that parks nothing cuts its chunk there;
    /// a kept session's prompt runs in the chunks it runs without checkpoints.
    #[test]
    fn prompt_end_checkpoints_are_reachable_and_cut_only_stateless_prompts() {
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        let first_chunk = |session: u64, n: u32| {
            let mut s = Scheduler::new(128);
            s.set_state_slots(4);
            s.configure_prefix_cache(24);
            let (r, _, _) = req(1, (0..n).collect(), 1, session);
            s.admit(r);
            step(&mut s, &limits, &[90], &mut 0).prefill[0].2
        };
        assert_eq!(first_chunk(0, 65), 64, "stateless: cut at its prompt end");
        assert_eq!(
            first_chunk(7, 65),
            65,
            "kept: one chunk, as without checkpoints"
        );
        let mut s = Scheduler::new(128);
        s.set_state_slots(4);
        s.configure_prefix_cache(24);
        let mut ask = |s: &mut Scheduler, id| {
            let (r, rx, _) = req(id, (0..64).collect(), 1, 0);
            s.admit(r);
            let mut loads = 0;
            while !s.is_idle() {
                loads += step(s, &limits, &[90], &mut t).loads.len();
            }
            (cached(&rx), loads)
        };
        assert_eq!(ask(&mut s, 1), (0, 0));
        assert_eq!(ask(&mut s, 2), (48, 1), "the first repeat");
    }

    /// Identical prompts prefilled in one step publish onto one block and save it once: the
    /// second save would evict another checkpoint for nothing.
    #[test]
    fn identical_prompts_in_one_step_save_once() {
        let mut s = Scheduler::new(128);
        s.set_state_slots(4);
        s.configure_prefix_cache(24);
        // Room for two checkpoints of 64 tokens (4 bytes a token in the shadow).
        s.prefix.state_budget = 2 * 64 * 4;
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        let prompt = |x: u32| -> Vec<u32> { (0..64).map(|i| i * 3 + x).chain([1]).collect() };
        for (id, x) in [(1, 0), (2, 1)] {
            let (r, _, _) = req(id, prompt(x), 1, 0);
            s.admit(r);
            run_all(&mut s, &limits, &[90], &mut t);
        }
        for id in [3, 4] {
            let (r, _, _) = req(id, prompt(2), 1, 0);
            s.admit(r);
        }
        let plan = step(&mut s, &limits, &[90], &mut t);
        assert_eq!(plan.prefill.len(), 2, "one step");
        run_all(&mut s, &limits, &[90], &mut t);
        let held = |s: &Scheduler, x| {
            let chain = s.prefix.lookup((0, 0), &prompt(x));
            s.prefix.checkpointed(&chain) == chain.len()
        };
        assert!(!held(&s, 0), "the least recently used went");
        assert!(held(&s, 1) && held(&s, 2));
    }

    /// Checkpoints stay under their byte budget, the least recently saved or loaded going first;
    /// one larger than the budget, or that the host cannot allocate, is not saved and takes
    /// nothing from the others. A checkpoint goes with its KV entry.
    #[test]
    fn checkpoints_hold_their_budget_least_recently_used_first() {
        let mut s = Scheduler::new(64);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        let (r, _, _) = req(1, (0..65).collect(), 1, 0);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        let chain = s.prefix.lookup((0, 0), &(0..65).collect::<Vec<_>>());
        assert_eq!(chain.len(), 4);
        s.prefix.state_budget = 30;
        let save = |s: &mut Scheduler, block: u32, n: usize| {
            let state = s.prefix.reserve(n);
            let saved = state.is_some();
            if let Some(state) = state {
                s.prefix.keep(block, state);
            }
            saved
        };
        assert!(save(&mut s, chain[0], 10));
        assert!(save(&mut s, chain[1], 10));
        assert!(save(&mut s, chain[2], 10));
        assert!(!s.prefix.wants_checkpoint(chain[2]));
        // Loading the first makes the second the least recently used.
        assert!(s.prefix.checkpoint(chain[0]).is_some());
        assert!(save(&mut s, chain[3], 10));
        let held: Vec<bool> = chain
            .iter()
            .map(|&b| !s.prefix.wants_checkpoint(b))
            .collect();
        assert_eq!(held, [true, false, true, true]);
        assert_eq!(s.prefix.checkpointed(&chain), 4);
        assert_eq!(s.prefix.checkpointed(&chain[..2]), 1);
        // Too large for the budget, or for the host: nothing saved, nothing evicted.
        assert!(!save(&mut s, chain[1], 31));
        s.prefix.state_budget = usize::MAX;
        assert!(!save(&mut s, chain[1], isize::MAX as usize + 1));
        let held: Vec<bool> = chain
            .iter()
            .map(|&b| !s.prefix.wants_checkpoint(b))
            .collect();
        assert_eq!(held, [true, false, true, true]);
        // Under budget pressure too: a failed allocation evicts nothing.
        s.prefix.state_budget = isize::MAX as usize + 10;
        assert!(!save(&mut s, chain[1], isize::MAX as usize + 1));
        let held: Vec<bool> = chain
            .iter()
            .map(|&b| !s.prefix.wants_checkpoint(b))
            .collect();
        assert_eq!(held, [true, false, true, true]);
        // A second save of one block keeps the first.
        let first = s.prefix.checkpoint(chain[3]).unwrap();
        s.prefix.keep(chain[3], vec![1; 10]);
        assert!(Arc::ptr_eq(&first, &s.prefix.checkpoint(chain[3]).unwrap()));
        // The leaf's eviction takes its checkpoint; an invalidation takes them all.
        assert!(s.prefix.evict(&mut s.blocks, &HashSet::new()));
        assert!(s.prefix.checkpoint(chain[3]).is_none());
        s.prefix.state_budget = 30;
        assert!(
            save(&mut s, chain[1], 10),
            "the evicted entry's bytes came back"
        );
        assert!(!s.prefix.wants_checkpoint(chain[0]) && !s.prefix.wants_checkpoint(chain[2]));
        s.invalidate_prefix_cache();
        assert_eq!(s.prefix.len(), 0);
        s.prefix.state_budget = 10;
        s.configure_prefix_cache(24);
        assert!(s.prefix.reserve(10).is_some(), "no bytes held");
    }

    #[test]
    fn automatic_exact_and_divergent_block_boundaries() {
        for n in [15, 16, 17, 31, 32, 33, 65] {
            let mut s = Scheduler::new(128);
            s.configure_prefix_cache(24);
            let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
            let mut t = 0;
            let prompt: Vec<_> = (0..n).collect();
            let (r, _, _) = req(1, prompt.clone(), 1, 0);
            s.admit(r);
            run_all(&mut s, &limits, &[90], &mut t);
            let (r, rx, _) = req(2, prompt.clone(), 1, 0);
            s.admit(r);
            run_all(&mut s, &limits, &[90], &mut t);
            assert_eq!(cached(&rx) as usize, (n as usize - 1) / 16 * 16);
            let mut divergent = prompt;
            divergent[(n as usize - 1).min(19)] = 100;
            let (r, rx, _) = req(3, divergent, 1, 0);
            s.admit(r);
            run_all(&mut s, &limits, &[90], &mut t);
            assert_eq!(cached(&rx), if n > 19 { 16 } else { (n - 1) / 16 * 16 });
        }
    }

    #[test]
    fn automatic_concurrent_readers_keep_private_suffixes() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 8, 512, 0.95);
        let mut t = 0;
        let (r, _, _) = req(1, (0..65).collect(), 1, 0);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        let shared = s.prefix.lookup((0, 0), &(0..65).collect::<Vec<_>>());
        let mut cancelled = None;
        for id in 2..=5 {
            let (r, _, c) = req(id, (0..65).collect(), 20, id);
            if id == 3 {
                cancelled = Some(c);
            }
            s.admit(r);
        }
        step(&mut s, &limits, &[90], &mut t);
        for id in 2..=5 {
            assert_eq!(&s.reqs[&id].seq.blocks[..4], &shared);
        }
        assert_eq!(s.blocks.used(), 8); // four common blocks plus four private tails
        assert_eq!(s.used_tokens(), 68); // shared 64 plus one position per private tail
        cancelled.unwrap().store(true, Ordering::Relaxed);
        s.release_session(2);
        run_all(&mut s, &limits, &[90], &mut t);
        s.check();
        assert_eq!(s.parked_tokens(2), 0);
        assert_eq!(s.parked_tokens(3), 0);
        assert!(s.parked_tokens(4) > 0);
    }

    #[test]
    fn automatic_shared_truncation_recomputes_tail_but_private_tail_reuses() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        let (r, _, _) = req(1, (0..69).collect(), 1, 9);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        let original = s.parked[&9].blocks[2];
        let mut p: Vec<_> = (0..40).collect();
        p.push(100);
        let (r, rx, _) = req(2, p, 1, 9);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        assert_eq!(cached(&rx), 32); // cuts into shared block, never writes it
        assert_ne!(s.parked[&9].blocks[2], original);
        let mut p: Vec<_> = (0..40).collect();
        p.extend([100, 101]);
        let (r, rx, _) = req(3, p, 1, 9);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        assert_eq!(cached(&rx), 41); // previous private tail retained
    }

    #[test]
    fn automatic_policy_namespace_release_and_reset() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        for (id, session, enabled, keep, namespace, want) in [
            (1, 9, true, true, 0, 0),
            (2, 0, false, false, 0, 0),
            (3, 10, true, false, 0, 0),
            (4, 0, true, false, 1, 0),
            (5, 0, true, false, 0, 64),
        ] {
            let (mut r, rx, _) = req(id, (0..65).collect(), 1, session);
            r.prefix_cache = enabled;
            r.keep_kv = keep;
            r.namespace = namespace;
            s.admit(r);
            run_all(&mut s, &limits, &[90], &mut t);
            assert_eq!(cached(&rx), want);
        }
        // Request 5, stateless, read session 9's chain, so the release leaves it.
        s.release_session(9);
        assert_eq!(
            s.prefix.lookup((0, 0), &(0..65).collect::<Vec<_>>()).len(),
            4
        );
        assert_eq!(
            s.prefix.lookup((1, 0), &(0..65).collect::<Vec<_>>()).len(),
            4
        );
        s.reset_pool(128);
        s.check();
        assert_eq!(s.prefix.len(), 0);
        assert_eq!(s.blocks.used(), 0);
    }

    #[test]
    fn automatic_stateless_hit_keeps_a_sessions_chain_past_its_release() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        // Chat A's rebuilt prompt after a compaction: six blocks only its session published.
        let (r, _, _) = req(1, (0..97).collect(), 1, 9);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        // Chat B's stateless anchor reads the first four.
        let anchor: Vec<_> = (0..65).collect();
        let (r, rx, _) = req(2, anchor.clone(), 1, 0);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        assert_eq!(cached(&rx), 64);
        s.release_session(9);
        // The release withdraws the two blocks no stateless request read, not the anchored four.
        assert_eq!(
            s.prefix.lookup((0, 0), &(0..97).collect::<Vec<_>>()).len(),
            4
        );
        // B's first turn still reuses the anchored prefix.
        let mut turn = anchor;
        turn.extend([200, 201]);
        let (r, rx, _) = req(3, turn, 1, 11);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        assert_eq!(cached(&rx), 64);
        s.check();
    }

    #[test]
    fn automatic_producer_cancel_after_a_stateless_adoption_still_withdraws() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        let prompt: Vec<u32> = (0..65).collect();
        // A session's producer publishes the prefix and keeps generating.
        let (r, _, cancel) = req(1, prompt.clone(), 20, 9);
        s.admit(r);
        step(&mut s, &limits, &[90], &mut t);
        assert_eq!(s.prefix.len(), 4);
        // A stateless reader completes on it, which adopts the chain.
        let (r, rx, _) = req(2, prompt.clone(), 1, 0);
        s.admit(r);
        while s.reqs.contains_key(&2) {
            step(&mut s, &limits, &[90], &mut t);
        }
        assert_eq!(cached(&rx), 64);
        assert!(s.reqs.contains_key(&1), "the producer still generates");
        // Another reader holds the chain when the producer is cancelled.
        let (r, _, _) = req(3, prompt, 20, 11);
        s.admit(r);
        step(&mut s, &limits, &[90], &mut t);
        assert_eq!(s.reqs[&3].cached, 64);
        let reader = s.reqs[&3].seq.blocks.clone();
        cancel.store(true, Ordering::Relaxed);
        let _plan = s.schedule(&limits, t + 1);
        assert!(!s.reqs.contains_key(&1));
        // The adoption moved the chain off session 9, not off its producer.
        assert_eq!(
            s.prefix.len(),
            0,
            "cancelled producer contributions must be withdrawn"
        );
        assert_eq!(&s.reqs[&3].seq.blocks[..reader.len()], &reader);
        s.check();
    }

    /// An activations request is a stateless reader that completes (Done, Length) once its prompt
    /// is fed, so it adopts the chain it read, as any other.
    #[test]
    fn automatic_an_activations_request_adopts_the_chain_it_read() {
        let whole = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut s = Scheduler::new(16);
        s.configure_prefix_cache(8);
        let mut t = 0;
        // Session 9 publishes one block.
        let (r, _rx, _c) = req(1, (1..=16).collect(), 1, 9);
        s.admit(r);
        run_all(&mut s, &whole, &[20], &mut t);
        let (tx, rx) = channel();
        let r = Request {
            activations: vec![0],
            ..Request::new((1..=17).collect(), 0)
        };
        s.admit(Req::new(2, r, tx, Arc::new(AtomicBool::new(false)), 1));
        t += 1;
        let plan = s.schedule(&whole, t);
        assert_eq!(plan.prefill, vec![(2, 16, 1)], "it reads the cached block");
        s.complete(&plan, Vec::new(), t);
        assert!(matches!(
            rx.try_iter().last(),
            Some(EngineEvent::Done { .. })
        ));
        s.release_session(9);
        assert_eq!(
            s.prefix.lookup((0, 0), &(1..=17).collect::<Vec<_>>()).len(),
            1
        );
        s.check();
    }

    #[test]
    fn automatic_adoption_grows_no_collection() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        let prompt: Vec<u32> = (0..65).collect();
        let (r, _, _) = req(1, prompt.clone(), 1, 9);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        let held = s.parked[&9].blocks.clone();
        let adopted = |s: &Scheduler| s.prefix.sources().filter(|&source| source == 0).count();
        let before = (s.prefix.heap(), s.prefix.len());
        // Another namespace's or history's blocks are never adopted.
        s.prefix.adopt((1, 0), &prompt, &held);
        let mut other = prompt.clone();
        other[0] = 100;
        s.prefix.adopt((0, 0), &other, &held);
        assert_eq!(adopted(&s), 0);
        // Nor the block that holds the prompt's final token, as lookup never reuses it.
        s.prefix.adopt((0, 0), &prompt[..64], &held);
        assert_eq!(adopted(&s), 3);
        for _ in 0..1000 {
            s.prefix.adopt((0, 0), &prompt, &held);
        }
        assert_eq!(adopted(&s), 4);
        // No table grew: adoption only rewrites entries in place.
        assert_eq!((s.prefix.heap(), s.prefix.len()), before);
        s.check();
    }

    #[test]
    fn automatic_reclaims_before_preemption_and_bounds_metadata() {
        let mut s = Scheduler::new(16);
        s.configure_prefix_cache(usize::MAX);
        // Half the pool (toolcall-opt item 4).
        assert_eq!(s.prefix.capacity(), 8);
        let limits = lim(Policy::DecodeFirst, 4, 128, 0.99);
        let mut t = 0;
        for id in 1..=8 {
            let (r, _, _) = req(id, vec![id as u32; 65], 1, 0);
            s.admit(r);
            run_all(&mut s, &limits, &[90], &mut t);
            assert!(s.prefix.len() <= 8);
        }
        assert!(s.prefix.evictions > 0);
        let (r, _, _) = req(9, vec![99; 239], 1, 0);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        assert_eq!(s.tot.preemptions, 0);
        s.check();
    }

    #[test]
    fn automatic_not_published_until_completion_and_errors_do_not_publish() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let (r, _, _) = req(1, (0..65).collect(), 1, 0);
        s.admit(r);
        let _planned = s.schedule(&limits, 1);
        assert_eq!(s.prefix.len(), 0);
        s.finish(1, Err(EngineError::Internal(1)), 2);
        assert_eq!(s.prefix.len(), 0);
        s.check();
    }

    #[test]
    fn observed_chunks_do_not_move_with_the_logits_budget() {
        // token_loss scores one text the same way whatever free memory sizes the logits budget at:
        // its chunk boundaries are fixed, so scoring one version twice gives one answer.
        let chunks = |rows: usize| {
            let mut s = Scheduler::new(256);
            let limits = Limits {
                logit_rows: rows,
                ..lim(Policy::DecodeFirst, 4, 512, 0.95)
            };
            let (mut r, _, _) = req(1, (0..300).collect(), 1, 0);
            r.observe = true;
            s.admit(r);
            let (mut t, mut seen) = (0, Vec::new());
            while !s.is_idle() {
                seen.extend(step(&mut s, &limits, &[90], &mut t).prefill);
            }
            seen
        };
        for rows in [OBSERVE_CHUNK + 4, 300, 4096] {
            assert_eq!(
                chunks(rows),
                [(1, 0, 128), (1, 128, 128), (1, 256, 44)],
                "{rows} rows"
            );
        }
    }

    #[test]
    fn automatic_observation_never_adopts_global_prefix() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        // Room to observe the whole 65-token prompt in one step: the 64-row logits budget would split it.
        let limits = Limits {
            logit_rows: 128,
            ..lim(Policy::DecodeFirst, 4, 512, 0.95)
        };
        let mut t = 0;
        let (r, _, _) = req(1, (0..65).collect(), 1, 0);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        let (mut r, _, _) = req(2, (0..65).collect(), 1, 0);
        r.observe = true;
        s.admit(r);
        let plan = s.schedule(&limits, 10);
        assert_eq!(plan.prefill, [(2, 0, 65)]);
        let mut rows = Vec::new();
        s.logits(&plan, &mut rows);
        assert_eq!(rows.iter().filter(|l| l.observe).count(), 65);
        s.finish(2, Err(EngineError::Cancelled), 11);
        s.check();
    }

    #[test]
    fn automatic_cancellation_withdraws_published_chunks_without_revoking_readers() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 4, 32, 0.95);
        let mut t = 0;
        let (r, _, cancel) = req(1, (0..65).collect(), 2, 0);
        s.admit(r);
        step(&mut s, &limits, &[90], &mut t);
        let limits = lim(Policy::DecodeFirst, 4, 64, 0.95);
        let (r, _, _) = req(2, (0..65).collect(), 3, 0);
        s.admit(r);
        step(&mut s, &limits, &[90], &mut t);
        assert!(s.reqs[&2].cached > 0);
        let reader_blocks = s.reqs[&2].seq.blocks.clone();
        cancel.store(true, Ordering::Relaxed);
        let _plan = s.schedule(&limits, t + 1);
        assert_eq!(
            s.prefix.len(),
            0,
            "cancelled producer contributions must be withdrawn"
        );
        assert_eq!(
            &s.reqs[&2].seq.blocks[..reader_blocks.len()],
            &reader_blocks
        );
        s.check();
    }

    #[test]
    fn automatic_invalidation_keeps_readers_but_prevents_late_publication() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 4, 32, 0.95);
        let mut t = 0;
        let (r, _, _) = req(1, (0..65).collect(), 2, 9);
        s.admit(r);
        step(&mut s, &limits, &[90], &mut t);
        assert_eq!(s.prefix.len(), 2);
        let owned = s.reqs[&1].seq.blocks.clone();
        s.invalidate_prefix_cache();
        assert_eq!(s.reqs[&1].seq.blocks, owned);
        run_all(&mut s, &limits, &[90], &mut t);
        assert_eq!(s.prefix.len(), 0, "old generation must not republish");
        // A reload's carry leaves the invalidated park for the pool reset to drop.
        let carried = s.carry_out();
        assert!(carried.is_empty());
        s.reset_pool(128);
        for (session, p) in carried {
            s.carry_in(session, p);
        }
        let (r, rx, _) = req(2, (0..65).collect(), 1, 9);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        assert_eq!(
            cached(&rx),
            0,
            "parked KV from incompatible generation cannot match"
        );
        s.check();
    }

    #[test]
    fn automatic_disabled_cache_preserves_legacy_session_reuse_on_performance_change() {
        let mut s = Scheduler::new(128);
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        let (r, _, _) = req(1, (0..37).collect(), 1, 9);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        s.invalidate_prefix_cache(); // performance-only setting transition with caching disabled
        let (r, rx, _) = req(2, (0..38).collect(), 1, 9);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        assert_eq!(cached(&rx), 37);
    }

    #[test]
    fn automatic_parked_blocks_count_unique_physical_ids() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 8, 512, 0.95);
        let mut t = 0;
        for id in 1..=4 {
            let (r, _, _) = req(id, (0..65).collect(), 1, id);
            s.admit(r);
            run_all(&mut s, &limits, &[90], &mut t);
        }
        assert_eq!(
            s.parked_blocks(),
            8,
            "four shared blocks plus four private tails"
        );
        assert_eq!(s.blocks.used(), 8);
        s.check();
    }

    #[test]
    fn automatic_release_stays_sticky_for_live_request_after_ring_wrap() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 4, 32, 0.95);
        let mut t = 0;
        let (r, _, _) = req(1, (0..65).collect(), 1, 9);
        s.admit(r);
        step(&mut s, &limits, &[90], &mut t);
        s.release_session(9);
        for session in 100..100 + RELEASED_RING as u64 {
            s.release_session(session);
        }
        run_all(&mut s, &limits, &[90], &mut t);
        assert_eq!(s.parked_tokens(9), 0);
        assert_eq!(
            s.prefix.len(),
            0,
            "live released request cannot resurrect global KV"
        );
    }

    #[test]
    fn automatic_cold_duplicate_prefixes_do_not_preempt_a_request_that_fits() {
        let mut s = Scheduler::new(32);
        s.configure_prefix_cache(8);
        let limits = lim(Policy::DecodeFirst, 2, 512, 0.95);
        let mut t = 0;
        let mut p1 = vec![1; 16];
        p1.extend([2; 17]);
        let mut p2 = vec![1; 16];
        p2.extend([3; 17]);
        let (r, _, _) = req(1, p1, 1, 0);
        s.admit(r);
        let (r, _, _) = req(2, p2, 479, 0);
        s.admit(r);
        // Both start cold in the same GPU step. The second must fit its full 512-token context
        // without retaining an unused duplicate canonical parent alongside its own root block.
        run_all(&mut s, &limits, &[90], &mut t);
        assert_eq!(
            s.tot.preemptions, 0,
            "idle canonical copies must not force recomputation"
        );
        assert_eq!(s.computed_prompt_tokens, 66);
        s.check();
    }

    #[test]
    fn automatic_shared_parking_keeps_host_metadata_bounded() {
        let mut s = Scheduler::new(32);
        s.configure_prefix_cache(8);
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        for id in 1..=40 {
            let (r, _, _) = req(id, (0..64).collect(), 1, id);
            s.admit(r);
            run_all(&mut s, &limits, &[90], &mut t);
            let refs: usize = s.parked.values().map(|p| p.blocks.len()).sum();
            assert!(
                refs <= s.blocks.total(),
                "parked host metadata exceeds pool-sized reference budget: {refs}"
            );
            s.check();
        }
        assert_eq!(
            s.parked_blocks(),
            4,
            "physical occupancy still counts shared blocks once"
        );
        assert_eq!(s.n_parked(), 8);
        assert_eq!(s.parked_tokens(1), 0);
        assert_eq!(s.parked_tokens(40), 64);
    }

    #[test]
    fn automatic_failed_start_acquires_nothing() {
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let limits = lim(Policy::DecodeFirst, 8, 512, 0.95);
        let mut t = 0;
        let (r, _, _) = req(1, (0..65).collect(), 1, 0);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        let (r, _, _) = req(2, vec![90], 100, 0);
        s.admit(r);
        step(&mut s, &limits, &[90], &mut t);
        let (r, _, _) = req(3, (0..65).collect(), 1, 0);
        s.admit(r);
        let before = s.blocks.refs.clone();
        let mut reclaimed = 0;
        let tight = lim(Policy::DecodeFirst, 8, 512, 0.01);
        assert_eq!(s.start(3, 512, &tight, &mut reclaimed), None);
        assert_eq!(s.blocks.refs, before);
        assert_eq!(reclaimed, 0);
        assert_eq!(s.reqs[&3].cached, 0);
        s.check();
    }

    proptest! {
        #[test]
        fn automatic_random_lifetimes(blocks in 4u32..32, lim in limits(), budget in background_rows(), hybrid in any::<bool>(), states in 0usize..4096, ops in prop::collection::vec(op(), 1..100)) {
            let mut s = Scheduler::new(blocks); s.configure_prefix_cache(8); s.background_rows = budget;
            // Checkpoints of 4 bytes a token: none, a few or many.
            s.prefix.state_budget = states;
            // A hybrid model: the engine caps max_batch_seqs at its state slots.
            if hybrid { s.set_state_slots(lim.max_batch_seqs as u32); }
            let mut t = 0; let mut next = 1;
            let mut handles = Vec::new();
            for op in ops {
                match op {
                    Op::Admit { observe, background, session, keep, new, max_tokens } => {
                        let mut prompt = vec![1; 17 + keep % 16];
                        prompt.extend(new.into_iter().take(16));
                        if prompt.len() + max_tokens as usize > blocks as usize * 16 { continue }
                        let snapshot = prompt.len() * (keep % 5) / 4;
                        let (mut r, rx, cancel) = req(next, prompt, max_tokens, session);
                        r.observe = observe;
                        r.class = class(background);
                        r.history_end = snapshot;
                        s.admit(r); handles.push((rx, cancel)); next += 1;
                    }
                    Op::Step { tokens } => { step(&mut s, &lim, &tokens, &mut t); }
                    Op::Cancel(i) => { if !handles.is_empty() { handles[i % handles.len()].1.store(true, Ordering::Relaxed); } }
                    Op::Release(session) => { s.release_session(session); }
                    Op::FinishStop(i) => { if !handles.is_empty() { s.finish_stop((i % handles.len()) as u64 + 1, t); } }
                }
                s.check();
            }
            for _ in 0..10_000 { if s.is_idle() { break } step(&mut s, &lim, &[2], &mut t); }
            prop_assert!(s.is_idle());
            for (rx, _) in handles {
                let events: Vec<_> = rx.try_iter().collect();
                prop_assert_eq!(events.iter().filter(|e| matches!(e, EngineEvent::Done { .. } | EngineEvent::Error(_))).count(), 1);
                prop_assert!(matches!(events.last(), Some(EngineEvent::Done { .. } | EngineEvent::Error(_))), "one terminal event last");
            }
        }
    }

    #[test]
    fn unchanged_waiting_completion_does_not_block_a_step() {
        use std::sync::mpsc::sync_channel;
        use std::time::Duration;
        let mut s = Scheduler::new(16);
        let limits = lim(Policy::DecodeFirst, 1, 512, 0.95);
        let (active, _events, _) = req(1, vec![1], 3, 0);
        s.admit(active);
        let plan = s.schedule(&limits, 2);
        s.complete(
            &plan,
            vec![(
                1,
                SampledToken {
                    id: 2,
                    logprob: 0.0,
                    top: Vec::new(),
                },
            )],
            3,
        );
        let (waiting, _waiting_events, _) = req(2, vec![3], 1, 0);
        let completion = waiting.completion.clone();
        s.admit(waiting);
        let held = completion.lock().unwrap();
        let (done_tx, done) = sync_channel(1);
        let worker = std::thread::spawn(move || {
            let plan = s.schedule(&limits, 4);
            s.complete(
                &plan,
                vec![(
                    1,
                    SampledToken {
                        id: 3,
                        logprob: 0.0,
                        top: Vec::new(),
                    },
                )],
                5,
            );
            done_tx.send(()).unwrap();
        });
        let progressed = done.recv_timeout(Duration::from_millis(100));
        drop(held);
        worker.join().unwrap();
        assert_eq!(
            progressed,
            Ok(()),
            "unchanged waiting metadata blocked a step"
        );
    }

    #[derive(Clone, Debug)]
    enum Op {
        /// Prompt as a prefix of a session's last history plus new tokens, or fresh.
        Admit {
            observe: bool,
            background: bool,
            session: u64,
            keep: usize,
            new: Vec<u32>,
            max_tokens: u32,
        },
        Step {
            tokens: Vec<u32>,
        },
        Cancel(usize),
        /// A caller's StopSeq acknowledgement, possibly for a request that already ended.
        FinishStop(usize),
        Release(u64),
    }

    fn op() -> impl Strategy<Value = Op> {
        prop_oneof![
            1 => (any::<bool>(), any::<bool>(), 0u64..4, 0usize..80, prop::collection::vec(0u32..12, 1..60), 1u32..12)
                .prop_map(|(observe, background, session, keep, new, max_tokens)| Op::Admit { observe, background, session, keep, new, max_tokens }),
            4 => prop::collection::vec(0u32..12, 64).prop_map(|tokens| Op::Step { tokens }),
            1 => any::<usize>().prop_map(Op::Cancel),
            1 => any::<usize>().prop_map(Op::FinishStop),
            1 => (1u64..4).prop_map(Op::Release),
        ]
    }

    fn class(background: bool) -> Class {
        if background {
            Class::Background
        } else {
            Class::Foreground
        }
    }

    /// The background budget: none, one row, or today's whole batch.
    fn background_rows() -> impl Strategy<Value = usize> {
        prop_oneof![Just(0), Just(1), Just(BACKGROUND_ROWS)]
    }

    fn limits() -> impl Strategy<Value = Limits> {
        (1usize..6, 1usize..48, 0u8..3, 0.80f64..0.99, 0usize..8).prop_map(
            |(seqs, chunk, p, w, spare)| {
                let policy =
                    [Policy::DecodeFirst, Policy::Balanced, Policy::PrefillFirst][p as usize];
                // An observation budget from one row a seat (the least the engine has) up.
                Limits {
                    logit_rows: seqs + spare,
                    ..lim(policy, seqs, chunk, w)
                }
            },
        )
    }

    /// One step, with `tokens` as what the model samples; checks every per-step invariant.
    fn step(s: &mut Scheduler, lim: &Limits, tokens: &[u32], t: &mut u64) -> StepPlan {
        *t += 1;
        let idle = s.is_idle();
        let parked_before = s.parked_blocks();
        let ready: Vec<(u64, u16)> = s
            .running
            .iter()
            .map(|id| &s.reqs[id])
            .filter(|r| r.seq.pending() == 1)
            .map(|r| (r.id, r.preemptions))
            .collect();
        let queued: Vec<(u64, u16)> = s
            .waiting
            .iter()
            .map(|id| (*id, s.reqs[id].preemptions))
            .collect();
        let plan = s.schedule(lim, *t);
        for (id, preemptions) in queued {
            assert!(
                s.reqs.get(&id).is_none_or(|r| r.preemptions == preemptions),
                "request {id} started and was taken back in one step"
            );
        }
        s.check();
        assert!(
            s.n_running() <= lim.max_batch_seqs,
            "running set over max_batch_seqs"
        );
        assert!(
            plan.decode.len() + plan.prefill.len() <= lim.max_batch_seqs,
            "too many sequences"
        );
        let budget = if lim.policy == Policy::Balanced {
            lim.prefill_chunk_tokens.saturating_sub(plan.decode.len())
        } else {
            lim.prefill_chunk_tokens
        };
        assert!(
            plan.prefill.iter().map(|p| p.2).sum::<usize>() <= budget,
            "over the token budget"
        );
        assert!(
            !plan.is_empty() || s.is_idle(),
            "work left but nothing planned"
        );
        if idle {
            assert_eq!(s.parked_blocks(), parked_before, "reclaim with no demand");
        }
        if plan.n_reclaimed > 0 {
            assert!(!plan.is_empty(), "parked blocks taken for nothing");
        }
        // Beside a foreground row, background rows decode only up to their budget.
        let rows = |c: Class| {
            plan.decode
                .iter()
                .filter(|id| s.reqs[id].class == c)
                .count()
        };
        let paused = rows(Class::Foreground) > 0 && rows(Class::Background) >= s.background_rows;
        assert!(
            rows(Class::Foreground) == 0 || rows(Class::Background) <= s.background_rows,
            "background rows past their budget beside a foreground row"
        );
        // Under every policy, a running sequence ready to decode decodes, unless cancelled or preempted
        // (it may restart at the head of the queue in the same step), or a background row the
        // budget paused.
        for (id, preemptions) in ready {
            assert!(
                s.reqs.get(&id).is_none_or(|r| plan.decode.contains(&id)
                    || r.preemptions > preemptions
                    || (r.class == Class::Background && paused)),
                "running sequence {id} starved"
            );
        }
        // Each row's state slot holds exactly the tokens before its first position, after the
        // loads and copies (which run before the step) and the resets at position 0. A
        // checkpoint's bytes are the tokens its state was fed.
        for (slot, state) in &plan.loads {
            let fed = state
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
            s.shadow.insert(*slot, fed.collect());
        }
        for &(from, to) in &plan.copies {
            let state = s.shadow.get(&from).cloned().unwrap_or_default();
            s.shadow.insert(to, state);
        }
        let decode = plan.decode.iter().map(|&id| (id, s.reqs[&id].seq.n_kv, 1));
        let fed: Vec<_> = decode.chain(plan.prefill.iter().copied()).collect();
        for (id, start, len) in fed {
            let r = &s.reqs[&id];
            // A row stops at its history end, and one that starts there takes its snapshot or
            // dropped it for want of a slot: else the next turn recomputes the whole chat.
            let end = r.history_end;
            assert!(
                !(start < end && end < start + len),
                "request {id} crosses its history end"
            );
            assert!(
                start != end || end == 0 || r.snap.is_some(),
                "request {id} took no snapshot"
            );
            if let Some(slot) = r.state_slot {
                let state = s.shadow.entry(slot).or_default();
                if start == 0 {
                    state.clear();
                }
                assert_eq!(state[..], r.seq.tokens[..start], "request {id}'s state");
                state.extend(&r.seq.tokens[start..start + len]);
            }
        }
        let mut rows = Vec::new();
        s.logits(&plan, &mut rows);
        assert!(
            rows.len() <= lim.logit_rows,
            "logits rows past the observation budget"
        );
        for l in rows.iter().filter(|l| l.observe) {
            s.reqs.get_mut(&l.id).expect("planned").next_obs = l.pos + 1; // as the engine reports it
        }
        let sampled: Sampled = rows
            .into_iter()
            .filter(|l| l.sample)
            .enumerate()
            .map(|(i, l)| {
                let id = l.id;
                let t = SampledToken {
                    id: tokens[i % tokens.len()],
                    logprob: 0.0,
                    top: Vec::new(),
                };
                (id, t)
            })
            .collect();
        s.complete(&plan, sampled, *t);
        s.check();
        for (slot, block) in std::mem::take(&mut s.saves) {
            let fed: Vec<u8> = s.shadow[&slot]
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect();
            if let Some(mut state) = s.prefix.reserve(fed.len()) {
                state.copy_from_slice(&fed);
                s.prefix.keep(block, state);
            }
        }
        for (new, old) in std::mem::take(&mut s.retire) {
            s.prefix.retire_checkpoint(new, old);
        }
        for p in s.parked.values() {
            if let Some(slot) = p.state {
                let state = s.shadow.get(&slot).map_or(&[][..], Vec::as_slice);
                assert_eq!(state, &p.tokens[..p.n_kv], "a parked state");
            }
        }
        plan
    }

    fn run_all(s: &mut Scheduler, lim: &Limits, tokens: &[u32], t: &mut u64) {
        while !s.is_idle() {
            step(s, lim, tokens, t);
        }
    }

    /// Host-side scheduler cost and outcome of a seeded workload under `prefill_first` (#196):
    /// 48 requests arriving over time, mixed prompt and decode lengths, a third background and a
    /// quarter observed. Pools: 10 blocks and 4 seats (pressure), 5 blocks and 2 seats with
    /// one-token observed chunks (the #196 shape), and 1024 blocks (everything fits).
    /// Run in release: `cargo test --release -p larch-engine --lib -- --ignored --nocapture
    /// sched::tests::scheduler_pressure_workload`.
    #[test]
    #[ignore]
    fn scheduler_pressure_workload() {
        const CAP: u64 = 20_000;
        const REPS: u32 = 200;
        println!("pool|seats|seed|finished|steps|tok/step|p50|p99|preempt|recomputed|ns/step");
        for (pool, seats, chunk, logit_rows) in
            [(10u32, 4, 32, 6), (5, 2, 512, 2), (1024, 4, 32, 6)]
        {
            let lim = Limits {
                logit_rows,
                ..lim(Policy::PrefillFirst, seats, chunk, 0.95)
            };
            let most = (pool as usize * BLOCK_TOKENS - 10).min(150);
            for seed in 1u64..=5 {
                let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                let mut rnd = |n: u64| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x % n
                };
                let mut arrive = 0;
                let work: Vec<_> = (1..=48u64)
                    .map(|id| {
                        arrive += rnd(6);
                        let prompt = 8 + rnd(65.min(most as u64 / 2)) as usize;
                        let max = (1 + rnd(48) as u32).min((most - prompt) as u32);
                        let bg = rnd(3) == 0;
                        (id, arrive, prompt, max, bg, rnd(4) == 0)
                    })
                    .collect();
                let (mut ns, mut all_steps) = (0u128, 0u128);
                let mut out = String::new();
                for rep in 0..REPS {
                    let mut s = Scheduler::new(pool);
                    let (mut t, mut next, mut steps) = (0u64, 0, 0u64);
                    let mut live: Vec<(u64, Receiver<EngineEvent>)> = Vec::new();
                    let mut latency = Vec::new();
                    let mut rows = Vec::new();
                    while t < CAP && (next < work.len() || !s.is_idle()) {
                        while next < work.len() && work[next].1 <= t {
                            let (id, at, prompt, max, bg, observe) = work[next];
                            let (mut r, rx, _) = req(id, vec![(id % 5 + 1) as u32; prompt], max, 0);
                            r.observe = observe;
                            if bg {
                                r.class = Class::Background;
                            }
                            s.admit(r);
                            live.push((at, rx));
                            next += 1;
                        }
                        t += 1;
                        let clock = std::time::Instant::now();
                        let plan = s.schedule(&lim, t);
                        s.logits(&plan, &mut rows);
                        let sampled: Sampled = rows
                            .iter()
                            .filter(|l| l.sample)
                            .map(|l| {
                                let tok = SampledToken {
                                    id: 3,
                                    logprob: 0.0,
                                    top: Vec::new(),
                                };
                                (l.id, tok)
                            })
                            .collect();
                        for l in rows.iter().filter(|l| l.observe) {
                            s.reqs.get_mut(&l.id).expect("planned").next_obs = l.pos + 1;
                        }
                        s.complete(&plan, sampled, t);
                        ns += clock.elapsed().as_nanos();
                        steps += 1;
                        live.retain(|(at, rx)| {
                            let done = rx.try_iter().any(|e| matches!(e, EngineEvent::Done { .. }));
                            if done {
                                latency.push(t - at);
                            }
                            !done
                        });
                    }
                    if rep == 0 {
                        latency.sort_unstable();
                        let pct = |p: usize| {
                            latency
                                .get((latency.len().saturating_sub(1)) * p / 100)
                                .copied()
                                .unwrap_or(0)
                        };
                        let prompts: u64 = work.iter().map(|w| w.2 as u64).sum();
                        out = format!(
                            "{pool}|{seats}|{seed}|{}/{}|{}{}|{:.2}|{}|{}|{}|{}",
                            latency.len(),
                            work.len(),
                            steps,
                            if t >= CAP { " (cap)" } else { "" },
                            s.tot.gen_tokens as f64 / steps as f64,
                            pct(50),
                            pct(99),
                            s.tot.preemptions,
                            s.computed_prompt_tokens.saturating_sub(prompts),
                        );
                    }
                    all_steps += u128::from(steps);
                }
                println!("{out}|{}", ns / all_steps);
            }
        }
    }

    // 256 cases by default; PROPTEST_CASES sets more.
    proptest! {
        #[test]
        fn invariants(blocks in 4u32..24, lim in limits(), budget in background_rows(), hybrid in any::<bool>(), ops in prop::collection::vec(op(), 1..120)) {
            let mut s = Scheduler::new(blocks);
            s.background_rows = budget;
            // A hybrid model: the engine caps max_batch_seqs at its state slots.
            if hybrid { s.set_state_slots(lim.max_batch_seqs as u32); }
            let pool = blocks as usize * BLOCK_TOKENS;
            let (mut t, mut next) = (0u64, 1u64);
            let mut handles: Vec<(Receiver<EngineEvent>, Arc<AtomicBool>)> = Vec::new();
            let mut history: HashMap<u64, Vec<u32>> = HashMap::new();
            for op in ops {
                match op {
                    Op::Admit { observe, background, session, keep, new, max_tokens } => {
                        let mut prompt: Vec<u32> = history.get(&session).map_or(vec![], |h| h[..keep.min(h.len())].to_vec());
                        prompt.extend(new);
                        if prompt.len() + max_tokens as usize > pool {
                            continue; // admission's PromptTooLong
                        }
                        let (mut r, rx, cancel) = req(next, prompt.clone(), max_tokens, session);
                        r.observe = observe;
                        r.class = class(background);
                        r.history_end = prompt.len() * (keep % 5) / 4;
                        s.admit(r);
                        if session != 0 {
                            history.insert(session, prompt);
                        }
                        handles.push((rx, cancel));
                        next += 1;
                    }
                    Op::Step { tokens } => {
                        // D14: a request matched as it starts reuses ≤ the common prefix, ≤ n_kv and
                        // ≤ prompt.len() − 1 of its session's parked sequence.
                        let parked: HashMap<u64, (Vec<u32>, usize)> =
                            s.parked.iter().map(|(&ss, p)| (ss, (p.tokens.clone(), p.n_kv))).collect();
                        let queued: Vec<u64> = s.reqs.values().filter(|r| r.t_admit == 0).map(|r| r.id).collect();
                        step(&mut s, &lim, &tokens, &mut t);
                        for r in s.reqs.values().filter(|r| queued.contains(&r.id) && r.t_admit != 0) {
                            let prompt = &r.seq.tokens[..r.prompt_len];
                            match parked.get(&r.session) {
                                Some((tokens, n_kv)) => {
                                    let lcp = tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
                                    prop_assert!(r.cached <= lcp && r.cached <= *n_kv && r.cached < prompt.len());
                                    // A recurrent state is reused whole or not at all.
                                    prop_assert!(!hybrid || r.cached == 0 || r.cached == *n_kv);
                                }
                                None => prop_assert_eq!(r.cached, 0),
                            }
                        }
                    }
                    Op::Cancel(i) => {
                        if !handles.is_empty() {
                            handles[i % handles.len()].1.store(true, Ordering::Relaxed);
                        }
                    }
                    Op::FinishStop(i) => {
                        if !handles.is_empty() {
                            s.finish_stop((i % handles.len()) as u64 + 1, t); // ids count from 1
                            s.check();
                        }
                    }
                    Op::Release(session) => {
                        s.release_session(session);
                        prop_assert!(!s.parked.contains_key(&session), "release frees exactly that session");
                        s.check();
                    }
                }
            }
            // Every admitted request ends with exactly one Done or Error, as its last event.
            for _ in 0..10_000 {
                if s.is_idle() {
                    break;
                }
                step(&mut s, &lim, &[1, 2, 3], &mut t);
            }
            prop_assert!(s.is_idle(), "requests never finished");
            for (rx, _) in handles {
                let events: Vec<EngineEvent> = rx.try_iter().collect();
                let ends = events.iter().filter(|e| matches!(e, EngineEvent::Done { .. } | EngineEvent::Error(_))).count();
                prop_assert_eq!(ends, 1);
                let ends_last = matches!(events.last(), Some(EngineEvent::Done { .. } | EngineEvent::Error(_)));
                prop_assert!(ends_last);
            }
            // Parked blocks are all that is left in use.
            prop_assert_eq!(s.blocks.used(), s.parked_blocks());
        }
    }

    /// `usage.cached_tokens` of the request's Done event.
    fn cached(rx: &Receiver<EngineEvent>) -> u32 {
        match rx.try_iter().last() {
            Some(EngineEvent::Done { usage, .. }) => usage.cached_tokens,
            e => panic!("no Done: {e:?}"),
        }
    }

    /// D14: an 8k-token parked conversation under a 256-token request loses 16 trailing blocks, and its
    /// next turn reuses 8192 − 256 tokens.
    #[test]
    fn reclaim_keeps_prefix() {
        let lim = lim(Policy::DecodeFirst, 4, 8192, 0.95);
        let mut s = Scheduler::new(512); // exactly the conversation
        let mut t = 0;
        let conv: Vec<u32> = (0..8192).map(|i| i % 11 + 1).collect();
        let (r, rx, _c) = req(1, conv[..8191].to_vec(), 1, 9);
        s.admit(r);
        run_all(&mut s, &lim, &[conv[8191]], &mut t);
        assert!(matches!(
            rx.try_iter().last(),
            Some(EngineEvent::Done { .. })
        ));
        assert_eq!(s.parked_tokens(9), 8191); // the sampled token was never fed
        assert_eq!(s.blocks.used(), 512);

        let (r2, rx2, _c2) = req(2, vec![3; 255], 1, 0);
        s.admit(r2);
        run_all(&mut s, &lim, &[1], &mut t);
        assert!(matches!(
            rx2.try_iter().last(),
            Some(EngineEvent::Done { .. })
        ));
        let p = &s.parked[&9];
        assert_eq!(p.blocks.len(), 512 - 16, "lost exactly 16 trailing blocks");
        assert_eq!(p.n_kv, 8192 - 256);

        // The next turn shares the whole conversation (and fits the pool beside it).
        let mut next = conv[..8190].to_vec();
        next.push(4);
        let (r3, rx3, _c3) = req(3, next, 1, 9);
        s.admit(r3);
        run_all(&mut s, &lim, &[1], &mut t);
        assert_eq!(cached(&rx3), 8192 - 256);
    }

    /// D14 partial_block: n_kv not a multiple of 16 through park, match and reclaim.
    #[test]
    fn partial_block() {
        let lim = lim(Policy::DecodeFirst, 4, 64, 0.99);
        let mut s = Scheduler::new(8);
        let mut t = 0;
        let (r, _rx, _c) = req(1, (1..=37).collect(), 1, 5);
        s.admit(r);
        run_all(&mut s, &lim, &[99], &mut t);
        let p = &s.parked[&5];
        assert_eq!((p.n_kv, p.blocks.len()), (37, 3));
        // Match: 37 common tokens, then a new one; reuse keeps the partial third block.
        let mut prompt: Vec<u32> = (1..=37).collect();
        prompt.push(50);
        let (r2, rx2, _c2) = req(2, prompt, 1, 5);
        s.admit(r2);
        run_all(&mut s, &lim, &[99], &mut t);
        assert_eq!(cached(&rx2), 37);
        assert_eq!(s.parked[&5].n_kv, 38);
        // Reclaim one block: 38 tokens in 3 blocks become 32 in 2.
        let (r3, _rx3, _c3) = req(3, vec![1; 16 * 6 - 1], 1, 0);
        s.admit(r3);
        run_all(&mut s, &lim, &[1], &mut t);
        let p = &s.parked[&5];
        assert_eq!((p.n_kv, p.blocks.len(), p.tokens.len()), (32, 2, 32));
        s.check();
    }

    /// D14: a parked conversation is never touched while nothing else needs the memory, even above
    /// the watermark.
    #[test]
    fn no_demand_no_reclaim() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut s = Scheduler::new(64);
        let mut t = 0;
        let (r, _rx, _c) = req(1, (0..1000).map(|i| i % 50 + 8).collect(), 1, 7);
        s.admit(r);
        run_all(&mut s, &lim, &[1], &mut t);
        assert_eq!(s.parked[&7].blocks.len(), 63); // 98 % of the pool
        let plan = step(&mut s, &lim, &[1], &mut t);
        assert_eq!((plan.n_reclaimed, s.parked[&7].blocks.len()), (0, 63));
    }

    /// A reclaim that cannot cover the request takes nothing; the request waits.
    #[test]
    fn failed_reclaim_keeps_parked() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.99);
        let mut s = Scheduler::new(10);
        let mut t = 0;
        let (r, _rx, _c) = req(1, (0..30).map(|i| i % 50 + 8).collect(), 1, 5);
        s.admit(r);
        run_all(&mut s, &lim, &[1], &mut t);
        assert_eq!(s.parked[&5].blocks.len(), 2);
        // A holds 7 blocks; B needs 4, and free (1) plus parked (2) cannot make that.
        let (a, _ra, _ca) = req(2, (0..100).map(|i| i % 50 + 8).collect(), 40, 0);
        s.admit(a);
        step(&mut s, &lim, &[1], &mut t);
        let (b, _rb, _cb) = req(3, (0..50).map(|i| i % 50 + 8).collect(), 1, 0);
        s.admit(b);
        let plan = step(&mut s, &lim, &[1], &mut t);
        assert!(!plan.prefill.iter().any(|p| p.0 == 3), "B cannot start");
        assert_eq!((plan.n_reclaimed, s.parked[&5].blocks.len()), (0, 2));
    }

    /// 4.5: max_batch_seqs bounds the running set under every policy, and a live cut preempts the
    /// newest sequences on the next step.
    #[test]
    fn max_batch_seqs_bounds_the_running_set() {
        for policy in [Policy::DecodeFirst, Policy::Balanced, Policy::PrefillFirst] {
            let mut s = Scheduler::new(64);
            let mut t = 0;
            let rxs: Vec<_> = (0..6)
                .map(|i| {
                    let (r, rx, _) = req(10 + i, vec![2, 3, 4], 20, 0);
                    s.admit(r);
                    rx
                })
                .collect();
            for _ in 0..3 {
                step(&mut s, &lim(policy, 4, 512, 0.95), &[1], &mut t);
            }
            assert_eq!(s.n_running(), 4);
            let plan = step(&mut s, &lim(policy, 2, 512, 0.95), &[1], &mut t);
            assert_eq!((plan.n_preempted, s.n_running()), (2, 2));
            assert_eq!(s.running, vec![10, 11], "the newest went back to the queue");
            run_all(&mut s, &lim(policy, 2, 512, 0.95), &[1], &mut t);
            for rx in rxs {
                assert!(matches!(
                    rx.try_iter().last(),
                    Some(EngineEvent::Done { .. })
                ));
            }
        }
    }

    /// 9.3 step 4 under `prefill_first`: a running sequence decodes every step while a short request
    /// arrives every step.
    #[test]
    fn prefill_first_never_starves_decode() {
        let lim = lim(Policy::PrefillFirst, 8, 512, 0.95);
        let mut s = Scheduler::new(1024);
        let mut t = 0;
        let (r, _rx, _c) = req(1, vec![2, 3, 4], 1000, 0);
        s.admit(r);
        step(&mut s, &lim, &[1], &mut t);
        let mut keep = Vec::new();
        for i in 0..50 {
            let (r, rx, _) = req(100 + i, vec![2, 3, 4], 1, 0);
            s.admit(r);
            keep.push(rx);
            let plan = step(&mut s, &lim, &[1], &mut t);
            assert!(plan.decode.contains(&1), "no decode at step {i}");
            assert!(plan.prefill.iter().any(|p| p.0 == 100 + i));
        }
    }

    /// A long prompt's chunk beside a running decode row is followed by decode-only steps for half
    /// that chunk's step time, then the next chunk; with no decode row, chunks run back to back.
    #[test]
    fn decode_rows_run_alone_between_prompt_chunks() {
        let lim = lim(Policy::DecodeFirst, 8, 16, 0.95);
        let mut s = Scheduler::new(1024);
        s.decode_share = 0.5;
        let mut t = 0;
        let (r, _rx, _c) = req(1, vec![2, 3, 4], 1000, 0);
        s.admit(r);
        step(&mut s, &lim, &[1], &mut t);
        let (r, _rx2, _c2) = req(2, vec![5; 64], 1, 0);
        s.admit(r);
        let chunk = |p: &StepPlan| p.prefill.iter().any(|c| c.0 == 2);
        let ms = 1_000_000;
        t = 1000 * ms - 1; // step() adds 1
        assert!(
            chunk(&step(&mut s, &lim, &[1], &mut t)),
            "the first chunk joins at once"
        );
        t = 1100 * ms - 1; // that chunk's step took 100 ms: the decode row runs alone for 50 more
        let plan = step(&mut s, &lim, &[1], &mut t);
        assert!(plan.decode.contains(&1) && !chunk(&plan));
        t = 1150 * ms - 2;
        assert!(!chunk(&step(&mut s, &lim, &[1], &mut t)));
        assert!(chunk(&step(&mut s, &lim, &[1], &mut t)), "due at 1150 ms");
        // That chunk's step took 1 ns, under PACE_MIN_NS: the next chunk joins at once.
        assert!(chunk(&step(&mut s, &lim, &[1], &mut t)));
        // An idle minute after a chunk counts as PACE_MAX_NS, not a 30 s wait.
        t += 60_000 * ms;
        assert!(!chunk(&step(&mut s, &lim, &[1], &mut t)));
        t += PACE_MAX_NS / 2;
        assert!(chunk(&step(&mut s, &lim, &[1], &mut t)));
    }

    /// While pacing, a prompt beside a replying chat goes in BUSY_CHUNK pieces; alone, whole.
    #[test]
    fn a_prompt_beside_replies_goes_in_busy_chunks() {
        let lim = lim(Policy::DecodeFirst, 8, 2048, 0.95);
        let mut s = Scheduler::new(4096);
        s.decode_share = 0.25;
        let mut t = 0;
        let (r, _rx, _c) = req(1, vec![5; 1500], 1000, 0);
        s.admit(r);
        let plan = step(&mut s, &lim, &[1], &mut t);
        assert_eq!(plan.prefill, [(1, 0, 1500)], "alone: the whole prompt");
        let (r, _rx2, _c2) = req(2, vec![6; 1500], 1, 0);
        s.admit(r);
        let plan = step(&mut s, &lim, &[1], &mut t);
        assert!(plan.decode.contains(&1));
        assert_eq!(
            plan.prefill,
            [(2, 0, BUSY_CHUNK)],
            "beside a reply: a piece"
        );
    }

    /// A background request, as a pinned document's read is.
    fn background_req(id: u64, prompt: Vec<u32>) -> (Req, Receiver<EngineEvent>) {
        let (tx, rx) = channel();
        let r = Request {
            class: Class::Background,
            ..Request::new(prompt, 1)
        };
        (Req::new(id, r, tx, Arc::new(AtomicBool::new(false)), 1), rx)
    }

    /// A pin read beside a replying chat goes in BUSY_CHUNK pieces on any backend (the decode
    /// share is the Metal lane's alone) and is what the pacing waits for; any other background
    /// prompt, and a foreground one, goes whole and unpaced, as on dev.
    #[test]
    fn only_a_pin_read_beside_a_chat_goes_in_busy_chunks_on_any_backend() {
        let lim = lim(Policy::DecodeFirst, 8, 2048, 0.95);
        for kind in ["pin read", "background", "foreground"] {
            let mut s = Scheduler::new(4096);
            s.pin_share = 0.25;
            let mut t = 0;
            let (r, _rx, _c) = req(1, vec![5; 20], 1000, 0);
            s.admit(r);
            step(&mut s, &lim, &[1], &mut t);
            let prompt = vec![6; 1500];
            if kind == "pin read" {
                s.prefix.pin(1, 0, prompt.clone());
            }
            let (r, _rx2) = match kind {
                "foreground" => {
                    let (r, rx, _) = req(2, prompt, 1, 0);
                    (r, rx)
                }
                _ => background_req(2, prompt),
            };
            s.admit(r);
            let plan = step(&mut s, &lim, &[1], &mut t);
            assert!(plan.decode.contains(&1));
            let len = if kind == "pin read" { BUSY_CHUNK } else { 1500 };
            assert_eq!(plan.prefill, [(2, 0, len)], "{kind}");
            assert_eq!(s.chunk_at.is_some(), kind == "pin read", "{kind}");
        }
    }

    /// One Prefilled per prefill chunk, cumulative and counting the tokens a cache served, for a
    /// request that asked; none for an observed one, nor one that did not ask.
    #[test]
    fn prefill_progress_is_cumulative_and_counts_reused_tokens() {
        let lim = lim(Policy::DecodeFirst, 4, 32, 0.95);
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(24);
        let mut t = 0;
        let progress = |rx: &Receiver<EngineEvent>| -> Vec<(u32, u32)> {
            rx.try_iter()
                .filter_map(|e| match e {
                    EngineEvent::Prefilled { done, total } => Some((done, total)),
                    _ => None,
                })
                .collect()
        };
        let prompt: Vec<u32> = (1..=100).collect();
        let (mut r, rx, _) = req(1, prompt.clone(), 1, 0);
        r.progress = true;
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        assert_eq!(progress(&rx), [(32, 100), (64, 100), (96, 100), (100, 100)]);
        // The repeat reuses 96 tokens and computes the last 4: its one chunk reports all 100.
        let (mut r, rx, _) = req(2, prompt.clone(), 1, 0);
        r.progress = true;
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        assert_eq!(progress(&rx), [(100, 100)]);
        let (mut r, rx, _) = req(3, prompt.clone(), 1, 0);
        (r.observe, r.progress) = (true, true);
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        assert_eq!(progress(&rx), []);
        // A request that did not ask is sent none.
        let (r, rx, _) = req(4, prompt, 1, 0);
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        assert_eq!(progress(&rx), []);
    }

    /// A pinned chain stays past the cache's capacity and demand eviction until it is unpinned,
    /// and a pool short of the answer's room stops pinning more.
    #[test]
    fn a_pinned_chain_outlives_eviction_and_ends_with_its_pin() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(8);
        s.prefix.turn_blocks = 8;
        let mut t = 0;
        let doc: Vec<u32> = (0..320).map(|i| i % 200 + 1).collect();
        s.prefix.pin(1, 0, doc.clone());
        let (r, _rx, _) = req(1, doc.clone(), 1, 0);
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        // Twenty blocks, though the cache holds eight.
        assert_eq!(s.prefix.pinned(), 20);
        assert_eq!(s.prefix.chain((0, 0), &doc).len(), 20);
        // Other prompts fill and churn the eight it may evict; the chain stays.
        for id in 2..8u64 {
            let other: Vec<u32> = (0..80).map(|i| 300 + id as u32 * 100 + i).collect();
            let (r, _rx, _) = req(id, other, 1, 0);
            s.admit(r);
            run_all(&mut s, &lim, &[90], &mut t);
            assert_eq!(s.prefix.pinned(), 20, "request {id}");
            assert_eq!(s.prefix.chain((0, 0), &doc).len(), 20, "request {id}");
            s.check();
        }
        // A question after the document reuses it whole.
        let mut ask = doc.clone();
        ask.extend([7; 5]);
        let (r, rx, _) = req(9, ask, 1, 0);
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        assert_eq!(cached(&rx), 320);
        // The freed chain is evictable at once: the cache is down to its size.
        s.unpin(1);
        assert_eq!((s.prefix.pinned(), s.prefix.len()), (0, 8));
        s.check();
    }

    /// Beside a pin, a request that fits the pool leaves it alone (a small one, a pin's own read, a
    /// question sharing the pinned prefix); one whose own blocks exceed what is left runs anyway,
    /// and the pin goes cold for it.
    #[test]
    fn a_request_past_the_turn_room_beside_a_pin_runs_and_the_pin_goes_cold() {
        let lim = lim(Policy::DecodeFirst, 4, 4096, 0.95);
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(8);
        s.prefix.turn_blocks = 8;
        let mut t = 0;
        let doc: Vec<u32> = (0..320).map(|i| i % 200 + 1).collect();
        s.prefix.pin(1, 0, doc.clone());
        let (r, _rx, _) = req(1, doc.clone(), 1, 0);
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        assert_eq!(s.prefix.pinned(), 20);
        let small: Vec<u32> = (0..100).map(|i| 300 + i).collect();
        let (r, rx, _) = req(3, small, 1, 0);
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        assert_eq!(cached(&rx), 0);
        // The document's own blocks are the pin's, not the question's.
        let mut ask = doc;
        ask.extend((0..1700).map(|i| 300 + i % 500));
        let (r, rx, _) = req(4, ask, 1, 0);
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        assert!(matches!(
            rx.try_iter().last(),
            Some(EngineEvent::Done { .. })
        ));
        assert!(s.take_cold().is_empty());
        // 128 - 20 = 108 blocks are left: 1,728 tokens. One more block takes the pin.
        let big: Vec<u32> = (0..1730).map(|i| 300 + i % 500).collect();
        let (r, rx, _) = req(2, big, 1, 0);
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        assert!(matches!(
            rx.try_iter().last(),
            Some(EngineEvent::Done { .. })
        ));
        assert_eq!((s.take_cold(), s.prefix.pinned()), (vec![1], 0));
        s.check();
    }

    /// A pin goes cold only when that makes the room: a request short of blocks that a live
    /// request holds waits for them, as without the pin, and the pin stays.
    #[test]
    fn a_pin_stays_when_live_requests_hold_the_room() {
        let lim = lim(Policy::DecodeFirst, 4, 4096, 0.95);
        let mut s = Scheduler::new(128);
        s.configure_prefix_cache(8);
        s.prefix.turn_blocks = 8;
        let mut t = 0;
        let doc: Vec<u32> = (0..320).map(|i| i % 200 + 1).collect();
        s.prefix.pin(1, 0, doc.clone());
        let (r, _rx, _) = req(1, doc, 1, 0);
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        // 108 blocks are left; the first request takes about 103 of them while it decodes, and
        // the second needs 30: the pin's 20 would not make that room, so it waits.
        let (a, rx_a, _) = req(2, (0..1600).map(|i| 300 + i % 500).collect(), 40, 0);
        let (b, rx_b, _) = req(3, (0..480).map(|i| 900 + i % 300).collect(), 1, 0);
        s.admit(a);
        s.admit(b);
        run_all(&mut s, &lim, &[90], &mut t);
        for rx in [rx_a, rx_b] {
            assert!(matches!(
                rx.try_iter().last(),
                Some(EngineEvent::Done { .. })
            ));
        }
        assert_eq!((s.take_cold(), s.prefix.pinned()), (vec![], 20));
        s.check();
    }

    /// A carried chain the cache takes only part of is kept on a dense model (its start is valid
    /// KV) and withdrawn on a hybrid, whose KV no one can read without the state after it; one
    /// the cache takes whole keeps its checkpoint, and a pool with no free blocks takes none.
    #[test]
    fn a_hybrid_pin_published_short_is_withdrawn() {
        let doc: Vec<u32> = (1..=48).collect();
        for (hybrid, capacity, kept) in [(true, 24, 3), (true, 1, 0), (false, 1, 1), (false, 24, 3)]
        {
            let mut s = Scheduler::new(64);
            if hybrid {
                s.set_state_slots(4);
            }
            s.configure_prefix_cache(capacity);
            // The pool passes no block as pinned, so only `capacity` bounds the cache.
            s.prefix.turn_blocks = 64;
            s.prefix.pin(1, 0, doc.clone());
            let got = s.carry_in_pin(1, 0, &doc, hybrid.then(|| vec![7; 4]));
            assert_eq!(got.len(), kept, "hybrid {hybrid}, capacity {capacity}");
            assert_eq!(s.prefix.chain((0, 0), &doc).len(), kept);
            assert_eq!(s.prefix.checkpointed(&got) > 0, hybrid && kept > 0);
            s.check();
        }
        let mut s = Scheduler::new(2);
        s.configure_prefix_cache(1);
        assert!(
            s.carry_in_pin(1, 0, &doc, None).is_empty(),
            "no free blocks"
        );
    }

    /// A hybrid's pin read keeps one state checkpoint at a time, at the last mark its chain has
    /// passed, so a partial export has the whole blocks to that mark and the state after them; the
    /// last one's goes only once the next is saved. A dense model's chain needs no state, and its
    /// partial export stops at the mark too.
    #[test]
    fn a_pin_read_keeps_one_checkpoint_at_its_last_mark() {
        let lim = lim(Policy::DecodeFirst, 4, 128, 0.95);
        let doc: Vec<u32> = (0..700).map(|i| i % 190 + 1).collect();
        for hybrid in [true, false] {
            let mut s = Scheduler::new(256);
            if hybrid {
                s.set_state_slots(4);
            }
            s.configure_prefix_cache(64);
            s.pin_mark = 256;
            s.prefix.turn_blocks = 8;
            s.prefix.pin(1, 0, doc.clone());
            let (r, _rx, _) = req(1, doc.clone(), 1, 0);
            s.admit(r);
            let mut t = 0;
            let mut marks = Vec::new();
            while !s.is_idle() {
                step(&mut s, &lim, &[90], &mut t);
                let (blocks, state) = s.pin_chain(0, &doc);
                if marks.last() != Some(&blocks.len()) && !blocks.is_empty() {
                    marks.push(blocks.len());
                    // Whole blocks to a multiple of the mark, with a state on a hybrid.
                    let done = s.prefix.chain((0, 0), &doc).len() < doc.len() / BLOCK_TOKENS;
                    if done {
                        assert_eq!(blocks.len() % 16, 0, "hybrid: {hybrid}");
                    }
                    assert_eq!(state.is_some(), hybrid, "hybrid: {hybrid}");
                    if hybrid {
                        // Of the marks, only the last holds a checkpoint.
                        let held = s.prefix.chain((0, 0), &doc);
                        let last = held.len() / 16 * 16;
                        let marks_held: Vec<_> = (0..held.len())
                            .filter(|&i| (i + 1) % 16 == 0 && s.prefix.has_checkpoint(held[i]))
                            .collect();
                        assert_eq!(marks_held, [last - 1], "{} blocks", held.len());
                    }
                }
            }
            // The chain is whole at the end, with the checkpoint at its last block on a hybrid.
            let (blocks, state) = s.pin_chain(0, &doc);
            assert_eq!(blocks.len(), 43, "hybrid: {hybrid}");
            assert_eq!(state.is_some(), hybrid);
            assert_eq!(&marks[..2], &[16, 32], "hybrid: {hybrid}: {marks:?}");
        }
    }

    /// A read's old mark checkpoint goes once the next is saved, unless another pin's chain ends on
    /// it; with that pin gone, it goes.
    #[test]
    fn retiring_a_mark_keeps_the_checkpoint_a_pin_ends_on() {
        let mut s = Scheduler::new(64);
        s.set_state_slots(2);
        s.configure_prefix_cache(32);
        s.prefix.turn_blocks = 4;
        let long: Vec<u32> = (0..64).collect();
        let (a_doc, b_doc) = (long[..33].to_vec(), long);
        s.prefix.pin(1, 0, a_doc.clone());
        s.prefix.pin(2, 0, b_doc.clone());
        let blocks = s.alloc_pin(4).unwrap();
        s.publish_pin(2, 0, &b_doc, blocks, Some(vec![2; 4]));
        let chain = s.prefix.chain((0, 0), &b_doc);
        assert_eq!(chain.len(), 4);
        // A ends on block 1, B's old mark; the new mark is block 3.
        s.prefix.keep(chain[1], vec![1; 4]);
        s.prefix.retire_checkpoint(chain[3], chain[1]);
        assert!(s.prefix.has_checkpoint(chain[1]), "A still ends on it");
        s.unpin(1);
        s.prefix.retire_checkpoint(chain[3], chain[1]);
        assert!(!s.prefix.has_checkpoint(chain[1]));
        assert!(s.prefix.has_checkpoint(chain[3]));
    }

    /// A pinned chain's checkpoint stays through checkpoint-budget pressure from other chats: a
    /// save that would need its bytes finds no room instead, and the others' go least recently
    /// used first.
    #[test]
    fn a_pinned_checkpoint_survives_budget_pressure() {
        let mut s = Scheduler::new(64);
        s.configure_prefix_cache(24);
        s.prefix.turn_blocks = 4;
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        let doc: Vec<u32> = (0..65).collect();
        s.prefix.pin(1, 0, doc.clone());
        let (r, _, _) = req(1, doc.clone(), 1, 0);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        let pinned = s.prefix.chain((0, 0), &doc);
        let other: Vec<u32> = (100..165).collect();
        let (r, _, _) = req(2, other.clone(), 1, 0);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        let chain = s.prefix.chain((0, 0), &other);
        assert_eq!((pinned.len(), chain.len()), (4, 4));
        s.prefix.keep(pinned[3], vec![1; 10]);
        s.prefix.state_budget = 25;
        let save = |s: &mut Scheduler, block: u32| match s.prefix.reserve(10) {
            Some(state) => {
                s.prefix.keep(block, state);
                true
            }
            None => false,
        };
        // Another chat's checkpoints fill the rest, then each new one replaces its oldest.
        assert!(save(&mut s, chain[0]) && save(&mut s, chain[1]));
        assert!(save(&mut s, chain[2]));
        assert!(!s.prefix.has_checkpoint(chain[0]) && s.prefix.has_checkpoint(chain[2]));
        assert!(save(&mut s, chain[3]));
        assert!(
            s.prefix.has_checkpoint(pinned[3]),
            "the pinned checkpoint stays"
        );
        // A reserve the pinned one leaves no room for evicts nothing on its way to None.
        s.prefix.state_budget = 25;
        let held: Vec<bool> = chain.iter().map(|&b| s.prefix.has_checkpoint(b)).collect();
        assert!(held.contains(&true));
        assert!(s.prefix.reserve(20).is_none());
        let after: Vec<bool> = chain.iter().map(|&b| s.prefix.has_checkpoint(b)).collect();
        assert_eq!(held, after);
        // A budget the pinned one fills leaves the others no room at all, and takes none of it.
        s.prefix.state_budget = 10;
        assert!(!save(&mut s, chain[0]));
        assert!(s.prefix.has_checkpoint(pinned[3]));
        // Unpinned, it is the others' to evict.
        s.unpin(1);
        s.prefix.state_budget = 10;
        s.prefix.checkpoint(chain[3]);
        assert!(save(&mut s, chain[0]));
        assert!(!s.prefix.has_checkpoint(pinned[3]));
    }

    /// While a pin holds the pool, other requests complete, whatever their number: the pin gives
    /// way when their blocks do not fit beside it.
    #[test]
    fn requests_beside_a_pin_complete_and_it_gives_way() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut s = Scheduler::new(64);
        s.configure_prefix_cache(32);
        s.prefix.turn_blocks = 16;
        let mut t = 0;
        let doc: Vec<u32> = (0..760).map(|i| i % 190 + 1).collect();
        s.prefix.pin(1, 0, doc.clone());
        let (r, _rx, _) = req(1, doc, 1, 0);
        s.admit(r);
        run_all(&mut s, &lim, &[90], &mut t);
        assert!(s.prefix.pinned() >= 40, "{}", s.prefix.pinned());
        // Eight chats of seven blocks each, for room of sixteen minus the pin's spill: four seats
        // want 28 blocks at once.
        let mut ends = Vec::new();
        for id in 2..10u64 {
            let prompt: Vec<u32> = (0..64).map(|i| 400 + id as u32 * 3 + i).collect();
            let (r, rx, _) = req(id, prompt, 40, 0);
            s.admit(r);
            ends.push(rx);
        }
        run_all(&mut s, &lim, &[90], &mut t);
        for rx in ends {
            let last = rx.try_iter().last();
            assert!(matches!(last, Some(EngineEvent::Done { .. })), "{last:?}");
        }
        assert_eq!((s.take_cold(), s.prefix.pinned()), (vec![1], 0));
        s.check();
    }

    /// A hybrid's pin read saves its state at the document's last whole block, even when the
    /// document ends on one: the question after it loads that state, not an earlier one.
    #[test]
    fn a_hybrid_pin_read_checkpoints_a_document_ending_on_a_block() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let doc: Vec<u32> = (1..=64).collect();
        let mut ask = doc.clone();
        ask.push(99);
        for (pinned, cached_tokens) in [(true, 64), (false, 48)] {
            let mut s = Scheduler::new(128);
            s.set_state_slots(4);
            s.configure_prefix_cache(24);
            let mut t = 0;
            if pinned {
                s.prefix.pin(1, 0, doc.clone());
            }
            let (r, _rx, _) = req(1, doc.clone(), 1, 0);
            s.admit(r);
            run_all(&mut s, &lim, &[90], &mut t);
            let (r, rx, _) = req(2, ask.clone(), 1, 0);
            s.admit(r);
            run_all(&mut s, &lim, &[90], &mut t);
            assert_eq!(cached(&rx), cached_tokens, "pinned: {pinned}");
        }
    }

    /// A hybrid's pin read of a document that ends on a block is read again (a reload, a second
    /// chat on it): the repeat looks up one token short, so the first read saved a state one
    /// block earlier too, and the repeat reuses all but that block.
    #[test]
    fn a_pin_read_ending_on_a_block_reads_again_from_the_block_before() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let doc: Vec<u32> = (1..=64).collect();
        let mut s = Scheduler::new(128);
        s.set_state_slots(4);
        s.configure_prefix_cache(24);
        s.prefix.pin(1, 0, doc.clone());
        let mut t = 0;
        for (id, reused) in [(1, 0), (2, 48), (3, 48)] {
            let (r, rx, _) = req(id, doc.clone(), 1, 0);
            s.admit(r);
            run_all(&mut s, &lim, &[90], &mut t);
            assert_eq!(cached(&rx), reused, "read {id}");
        }
    }

    /// #196: under `prefill_first` a decode row short of a block found the newer row's one-token
    /// observed chunk planned and preempted itself; its restart did the same to the other, forever.
    /// Now the older request is never preempted.
    #[test]
    fn two_observed_requests_never_preempt_each_other_forever() {
        let lim = Limits {
            logit_rows: 2,
            ..lim(Policy::PrefillFirst, 2, 512, 0.8)
        };
        let mut s = Scheduler::new(5);
        s.configure_prefix_cache(8);
        let mut t = 0;
        let (mut r, older, _) = req(1, vec![1; 40], 11, 0);
        r.observe = true;
        s.admit(r);
        for _ in 0..17 {
            step(&mut s, &lim, &[0; 64], &mut t);
        }
        let (mut r, newer, _) = req(2, vec![2; 40], 11, 0);
        r.observe = true;
        s.admit(r);
        for _ in 0..10_000 {
            if s.is_idle() {
                break;
            }
            step(&mut s, &lim, &[3], &mut t);
            let preemptions: Vec<_> = s.reqs.values().map(|r| (r.id, r.preemptions)).collect();
            assert!(
                s.reqs.get(&1).is_none_or(|r| r.preemptions == 0),
                "the older request was preempted: {preemptions:?}"
            );
        }
        assert!(s.is_idle(), "never idle");
        for rx in [older, newer] {
            assert!(matches!(
                rx.try_iter().last(),
                Some(EngineEvent::Done { .. })
            ));
        }
    }

    /// #196 without observation: under `prefill_first` a resumed turn's one-token remainder starts
    /// before the decode rows run, so it waits while it would take the block an older row needs.
    #[test]
    fn a_start_leaves_an_older_decode_row_its_next_block() {
        // A watermark of 1.0 lets the resumed turn take the pool's last free block.
        let lim = lim(Policy::PrefillFirst, 2, 512, 1.0);
        let mut s = Scheduler::new(4);
        let mut t = 0;
        let (r, _, _) = req(1, (0..32).collect(), 1, 9);
        s.admit(r);
        run_all(&mut s, &lim, &[1], &mut t);
        assert_eq!(s.parked[&9].blocks.len(), 2);
        let (r, older, _) = req(2, vec![2; 15], 20, 0);
        s.admit(r);
        step(&mut s, &lim, &[1], &mut t);
        step(&mut s, &lim, &[1], &mut t);
        let (r, newer, _) = req(3, (0..33).collect(), 1, 9);
        s.admit(r);
        // Request 2 needs its second block for position 16; request 3 needs the other two and its
        // session's parked two.
        let plan = step(&mut s, &lim, &[1], &mut t);
        assert_eq!((&plan.decode[..], plan.n_preempted), (&[2][..], 0));
        assert_eq!(s.waiting, [3]);
        run_all(&mut s, &lim, &[1], &mut t);
        for rx in [older, newer] {
            assert!(matches!(
                rx.try_iter().last(),
                Some(EngineEvent::Done { .. })
            ));
        }
    }

    /// 10.1.1 reset_race: a release while the turn runs is sticky; the late completion frees.
    #[test]
    fn release_during_a_turn_is_sticky() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut s = Scheduler::new(16);
        let mut t = 0;
        let (r, rx, _c) = req(1, (0..20).map(|i| i % 50 + 8).collect(), 6, 4);
        s.admit(r);
        step(&mut s, &lim, &[1], &mut t);
        s.release_session(4);
        run_all(&mut s, &lim, &[1], &mut t);
        assert!(matches!(
            rx.try_iter().last(),
            Some(EngineEvent::Done { .. })
        ));
        assert!(s.parked.is_empty());
        assert_eq!(s.blocks.used(), 0);
    }

    /// A request cancelled during its last step is not parked.
    #[test]
    fn cancelled_never_parks() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut s = Scheduler::new(16);
        let (r, rx, cancel) = req(1, vec![8, 9, 10], 1, 4);
        s.admit(r);
        let plan = s.schedule(&lim, 1);
        cancel.store(true, Ordering::Relaxed);
        let tok = SampledToken {
            id: 1,
            logprob: 0.0,
            top: Vec::new(),
        };
        s.complete(&plan, vec![(1, tok)], 1);
        assert!(matches!(
            rx.try_iter().last(),
            Some(EngineEvent::Done { .. })
        ));
        assert!(s.parked.is_empty() && s.blocks.used() == 0);
    }

    /// The span request `id` ended with.
    fn ended(s: &Scheduler, id: u64) -> RequestSpan {
        s.ended
            .iter()
            .find(|(sp, _)| sp.req_id == id)
            .expect("span")
            .0
    }

    /// A turn cancelled in the queue started nothing, so the next turn still reports the reload,
    /// as the session's second turn, with the Reload reclaim's id.
    #[test]
    fn a_queued_cancel_leaves_the_session_record() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut s = Scheduler::new(16);
        let mut t = 0;
        let (r, _rx, _c) = req(1, (1..=20).collect(), 2, 7);
        s.admit(r);
        run_all(&mut s, &lim, &[30], &mut t);
        s.reset_pool(16);
        let [reload] = s.reclaims[..] else {
            panic!("{:?}", s.reclaims)
        };
        let (r, _rx, cancel) = req(2, (1..=21).collect(), 2, 7);
        s.admit(r);
        cancel.store(true, Ordering::Relaxed);
        let (r, _rx, _c) = req(3, (1..=21).collect(), 2, 7);
        s.admit(r);
        run_all(&mut s, &lim, &[30], &mut t);
        assert_eq!(ended(&s, 2).turn_id, 0, "never started");
        let span = ended(&s, 3);
        assert_eq!(
            (span.turn_id, span.reuse_miss(), span.reclaim_id),
            (2, ReuseMiss::Reloaded, reload.reclaim_id)
        );
    }

    /// The next turn reports why the last one left no KV: a cancel, a release (which keeps the
    /// reload's reclaim id when the KV had already gone), and a prompt that leaves the lost history
    /// at its first token. Two turns of one session at once get two turn ids, and the next turn
    /// the one after both.
    #[test]
    fn each_lost_turn_reports_its_reason() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut s = Scheduler::new(16);
        let mut t = 0;
        let prompt: Vec<u32> = (1..=20).collect();
        let turn = |s: &mut Scheduler, t: &mut u64, id: u64, session: u64, prompt: &[u32]| {
            let (r, _rx, _c) = req(id, prompt.to_vec(), 2, session);
            s.admit(r);
            run_all(s, &lim, &[30], t);
            let span = ended(s, id);
            (span.turn_id, span.reuse_miss(), span.reclaim_id)
        };
        assert_eq!(turn(&mut s, &mut t, 1, 5, &prompt).0, 1);
        let (r, _rx, cancel) = req(2, prompt.clone(), 8, 5);
        s.admit(r);
        step(&mut s, &lim, &[30], &mut t);
        cancel.store(true, Ordering::Relaxed);
        run_all(&mut s, &lim, &[30], &mut t);
        let cancelled = (3, ReuseMiss::Cancelled, 0);
        assert_eq!(turn(&mut s, &mut t, 3, 5, &prompt), cancelled);
        s.release_session(5);
        let reset = (4, ReuseMiss::Reset, 0);
        assert_eq!(turn(&mut s, &mut t, 4, 5, &prompt), reset);

        turn(&mut s, &mut t, 5, 6, &prompt);
        s.reset_pool(16);
        let reload = s.reclaims.last().expect("reload").reclaim_id;
        s.release_session(6);
        let reset = (2, ReuseMiss::Reset, reload);
        assert_eq!(turn(&mut s, &mut t, 6, 6, &prompt), reset);

        turn(&mut s, &mut t, 7, 8, &prompt);
        s.reset_pool(16);
        let reload = s.reclaims.last().expect("reload").reclaim_id;
        let diverged = (2, ReuseMiss::PrefixDiverged, reload);
        assert_eq!(turn(&mut s, &mut t, 8, 8, &[99, 98]), diverged);

        for id in [9, 10] {
            let (r, _rx, _c) = req(id, prompt.clone(), 2, 8);
            s.admit(r);
        }
        run_all(&mut s, &lim, &[30], &mut t);
        assert_eq!((ended(&s, 9).turn_id, ended(&s, 10).turn_id), (3, 4));
        assert_eq!(turn(&mut s, &mut t, 11, 8, &prompt).0, 5);
    }

    /// A session whose parked KV went to bound parked metadata reports OverflowDrop.
    #[test]
    fn an_overflow_drop_reaches_the_next_turn() {
        let mut s = Scheduler::new(32);
        s.configure_prefix_cache(8);
        let limits = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut t = 0;
        for id in 1..=10 {
            let (r, _, _) = req(id, (0..64).collect(), 1, id);
            s.admit(r);
            run_all(&mut s, &limits, &[90], &mut t);
        }
        assert_eq!(s.parked_tokens(1), 0);
        let (r, _, _) = req(11, (0..65).collect(), 1, 1);
        s.admit(r);
        run_all(&mut s, &limits, &[90], &mut t);
        let span = ended(&s, 11);
        assert_eq!(
            (span.turn_id, span.reuse_miss()),
            (2, ReuseMiss::OverflowDrop)
        );
    }

    /// Terminal events, spans and reuse agree for every end and keep_kv setting (10.1.1).
    #[test]
    fn terminal_rows() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        for end in [
            Some(Finish::Eos),
            Some(Finish::Length),
            Some(Finish::StopSeq),
            None,
        ] {
            for keep in [true, false] {
                let mut s = Scheduler::new(16);
                let mut t = 0;
                // Turn 1 parks [1, 2, 3, 20]; 21 was sampled last, so it was never fed.
                let (r, _rx1, _c1) = req(1, vec![1, 2, 3], 2, 5);
                s.admit(r);
                step(&mut s, &lim, &[20], &mut t);
                step(&mut s, &lim, &[21], &mut t);
                assert_eq!(s.parked[&5].tokens, [1, 2, 3, 20], "{end:?} keep={keep}");
                // Turn 2 re-sends the history plus 30, reuses 4 tokens and samples 40, 41, then 42
                // (STOP for Eos).
                let max_tokens = if end == Some(Finish::Length) { 3 } else { 100 };
                let (mut r, rx, cancel) = req(2, vec![1, 2, 3, 20, 21, 30], max_tokens, 5);
                r.keep_kv = keep;
                s.admit(r);
                let last = if end == Some(Finish::Eos) { STOP } else { 42 };
                for tok in [40, 41, last] {
                    step(&mut s, &lim, &[tok], &mut t);
                }
                match end {
                    // The caller matched a stop string on 42: acknowledged before 42 is fed.
                    Some(Finish::StopSeq) => s.finish_stop(2, t),
                    None => {
                        cancel.store(true, Ordering::Relaxed);
                        step(&mut s, &lim, &[1], &mut t);
                    }
                    Some(_) => {}
                }
                s.check();
                let usage = Usage {
                    prompt_tokens: 6,
                    completion_tokens: 3,
                    cached_tokens: 4,
                };
                match (end, rx.try_iter().last()) {
                    (
                        Some(f),
                        Some(EngineEvent::Done {
                            finish, usage: u, ..
                        }),
                    ) => {
                        assert_eq!((finish, u), (f, usage), "{end:?} keep={keep}");
                    }
                    (None, Some(EngineEvent::Error(EngineError::Cancelled))) => {}
                    (_, e) => panic!("{end:?} keep={keep}: {e:?}"),
                }
                let (span, _) = *s
                    .ended
                    .last()
                    .unwrap_or_else(|| panic!("{end:?} keep={keep}: turn 2 did not end"));
                let outcome = end.map_or(Outcome::Cancelled, |_| Outcome::Ok);
                assert_eq!(
                    (span.req_id, span.outcome(), span.finish()),
                    (2, outcome, end.unwrap_or(Finish::None)),
                    "{end:?} keep={keep}"
                );
                assert_eq!(
                    (span.gen_tokens, span.cached_tokens),
                    (3, 4),
                    "{end:?} keep={keep}"
                );
                // Done under keep_kv parks the 8 fed tokens; every other end leaves the pool empty.
                let parked = keep && end.is_some();
                let fed = [1, 2, 3, 20, 21, 30, 40, 41];
                assert_eq!(
                    s.parked.get(&5).map(|p| &p.tokens[..]),
                    parked.then_some(&fed[..]),
                    "{end:?} keep={keep}"
                );
                assert_eq!(s.blocks.used(), s.parked_blocks(), "{end:?} keep={keep}");
                // Turn 3 renders the history: turn 2's delivered text, but only what came before a
                // stop string (here [41, 42]), so StopSeq's reuse ends one token early.
                let mut next = fed[..7].to_vec();
                if end != Some(Finish::StopSeq) {
                    next.extend([41, last]);
                }
                next.push(50);
                let (r, rx3, _c3) = req(3, next, 1, 5);
                s.admit(r);
                run_all(&mut s, &lim, &[60], &mut t);
                let want = match (parked, end) {
                    (false, _) => 0,
                    (true, Some(Finish::StopSeq)) => 7,
                    (true, _) => 8,
                };
                assert_eq!(cached(&rx3), want, "{end:?} keep={keep}");
            }
        }
    }

    /// Stop, cancel and release end a request once and never resurrect released KV (10.1.1).
    #[test]
    fn stop_release_and_cancel_in_every_order() {
        // Reset cancels then releases; Session drop can precede its stream cancelling.
        // Done means stop beat cancel.
        #[derive(Clone, Copy, Debug)]
        enum Race {
            Release,
            Cancel,
            Stop,
        }
        use Race::*;
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let cases: [(&[Race], bool); 8] = [
            (&[Release, Cancel, Stop], false),
            (&[Release, Stop, Cancel], true),
            (&[Stop, Release, Cancel], true),
            (&[Stop, Release], true),
            (&[Release, Stop], true),
            (&[Stop, Cancel, Release], true),
            (&[Cancel, Stop, Release], false),
            (&[Cancel, Release, Stop], false),
        ];
        for (order, done) in cases {
            let mut s = Scheduler::new(16);
            let mut t = 0;
            let (r, rx, cancel) = req(1, vec![1, 2, 3], 100, 5);
            s.admit(r);
            step(&mut s, &lim, &[20], &mut t);
            step(&mut s, &lim, &[21], &mut t);
            for op in order {
                match op {
                    Release => s.release_session(5),
                    Cancel => cancel.store(true, Ordering::Relaxed),
                    Stop => s.finish_stop(1, t),
                }
                s.check();
            }
            assert!(s.is_idle(), "{order:?}");
            let events: Vec<EngineEvent> = rx.try_iter().collect();
            let ends = events
                .iter()
                .filter(|e| matches!(e, EngineEvent::Done { .. } | EngineEvent::Error(_)))
                .count();
            assert_eq!(ends, 1, "{order:?}");
            let stopped = match events.last() {
                Some(EngineEvent::Done {
                    finish: Finish::StopSeq,
                    ..
                }) => true,
                Some(EngineEvent::Error(EngineError::Cancelled)) => false,
                e => panic!("{order:?}: {e:?}"),
            };
            assert_eq!(stopped, done, "{order:?}");
            assert!(s.parked.is_empty() && s.blocks.used() == 0, "{order:?}");
        }
    }

    /// A late stop preserves parked KV and cannot end the session's next request.
    #[test]
    fn a_late_stop_touches_only_its_request() {
        let lim = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let mut s = Scheduler::new(16);
        let mut t = 0;
        let (r, rx1, _c1) = req(1, vec![1, 2, 3], 2, 5);
        s.admit(r);
        run_all(&mut s, &lim, &[20], &mut t);
        s.finish_stop(1, t);
        s.check();
        assert_eq!(s.parked[&5].tokens, [1, 2, 3, 20]);
        let ends: Vec<EngineEvent> = rx1.try_iter().collect();
        assert!(matches!(
            ends.last(),
            Some(EngineEvent::Done {
                finish: Finish::Length,
                ..
            })
        ));
        // Turn 2 takes the parked prefix; turn 1's acknowledgement arriving now leaves it running.
        let (r, rx2, _c2) = req(2, vec![1, 2, 3, 20, 20, 30], 100, 5);
        s.admit(r);
        step(&mut s, &lim, &[40], &mut t);
        let n_kv = s.req(2).expect("running").seq.n_kv;
        s.finish_stop(1, t);
        s.check();
        assert_eq!(s.req(2).map(|r| r.seq.n_kv), Some(n_kv));
        assert!(rx1.try_iter().next().is_none(), "no second terminal");
        s.finish_stop(2, t);
        s.check();
        assert!(matches!(
            rx2.try_iter().last(),
            Some(EngineEvent::Done {
                finish: Finish::StopSeq,
                usage: Usage {
                    cached_tokens: 4,
                    ..
                },
                ..
            })
        ));
    }

    /// A forced run preempted after reporting positions 0 and 1 restarts onto the prefix another
    /// request of its session parked meanwhile, and still reports every position once, in order.
    #[test]
    fn a_restarted_forced_run_reports_every_position() {
        let mut s = Scheduler::new(64);
        let r: Vec<u32> = (0..40).map(|i| i % 11).collect();
        let (b, _b_rx, _) = req(1, r[..30].to_vec(), 3, 7);
        let (mut a, _a_rx, _) = req(2, r.clone(), 1, 7);
        let (positions, _) = channel();
        (a.keep_kv, a.observe) = (false, true);
        a.completion.lock().unwrap().positions = Some(positions);
        s.admit(b);
        s.admit(a);
        let mut reported = Vec::new();
        // B prefills 30 tokens in steps 1-4 (8 a step) and A starts in step 4's last 2; one seat
        // from step 5 preempts A (the newest) until B ends in step 6 and parks its sequence, whose
        // first 30 tokens are A's: a restart without the fix would resume A at position 30.
        for t in 1..=40 {
            let seats = if t >= 5 { 1 } else { 2 };
            let plan = s.schedule(&lim(Policy::DecodeFirst, seats, 8, 0.99), t);
            if s.is_idle() && plan.is_empty() {
                break;
            }
            let mut sampled = Vec::new();
            let mut rows = Vec::new();
            s.logits(&plan, &mut rows);
            for l in rows {
                let rq = s.reqs.get_mut(&l.id).expect("planned");
                if l.observe {
                    reported.push(l.pos);
                    rq.next_obs = l.pos + 1; // as the engine reports it
                }
                if l.sample {
                    let tok = SampledToken {
                        id: 11, // in no prompt, so A shares exactly r[..30] with B's sequence
                        logprob: 0.0,
                        top: Vec::new(),
                    };
                    sampled.push((l.id, tok));
                }
            }
            s.complete(&plan, sampled, t);
            s.check();
        }
        assert!(s.tot.preemptions > 0 && s.is_idle());
        assert_eq!(reported, (0..40).collect::<Vec<_>>());
    }

    /// An activations request never takes a decode row, even with one token left (its last chunk's
    /// remainder, or the one after a cached prefix), has no logits row, and ends Done (Length) once
    /// its prompt is fed, its span timed.
    #[test]
    fn activations_run_as_prefill_rows_and_finish_after_the_last() {
        let whole = lim(Policy::DecodeFirst, 4, 512, 0.95);
        let lim = lim(Policy::DecodeFirst, 4, 4, 0.95);
        let mut s = Scheduler::new(16);
        let mut t = 0;
        let tapped = |id, prompt: Vec<u32>| {
            let (tx, rx) = channel();
            let r = Request {
                activations: vec![0],
                ..Request::new(prompt, 0)
            };
            (Req::new(id, r, tx, Arc::new(AtomicBool::new(false)), 1), rx)
        };
        let go = |s: &mut Scheduler, t: &mut u64| {
            *t += 1;
            let plan = s.schedule(&lim, *t);
            let mut rows = Vec::new();
            s.logits(&plan, &mut rows);
            assert!(rows.is_empty(), "a logits row");
            s.complete(&plan, Vec::new(), *t);
            s.check();
            (plan.decode, plan.prefill)
        };
        let ended = |rx: &Receiver<EngineEvent>| match rx.try_iter().last() {
            Some(EngineEvent::Done { finish, usage, .. }) => (finish, usage.completion_tokens),
            e => panic!("{e:?}"),
        };
        // Five tokens in chunks of four: the remainder of one prefills.
        let (r, rx) = tapped(1, vec![1, 2, 3, 4, 5]);
        s.admit(r);
        assert_eq!(go(&mut s, &mut t), (vec![], vec![(1, 0, 4)]));
        assert_eq!(go(&mut s, &mut t), (vec![], vec![(1, 4, 1)]));
        assert_eq!(ended(&rx), (Finish::Length, 0));
        assert!(s.is_idle());
        // A cached block of 16; the request reuses it and prefills its one new token.
        s.configure_prefix_cache(8);
        let (r, _rx, _c) = req(2, (1..=16).collect(), 1, 0);
        s.admit(r);
        run_all(&mut s, &whole, &[20], &mut t);
        let (r, rx) = tapped(3, (1..=17).collect());
        s.admit(r);
        assert_eq!(go(&mut s, &mut t), (vec![], vec![(3, 16, 1)]));
        assert_eq!(ended(&rx), (Finish::Length, 0));
        assert!(s.is_idle() && s.parked.is_empty());
        let ok = |sp: &RequestSpan| sp.outcome == Outcome::Ok.raw() && sp.t_first_ns > 0;
        assert!(s.ended.iter().all(|(sp, _)| ok(sp)), "{:?}", s.ended);
    }

    /// At most `TAP_SEATS` activations requests run at once, so a step taps at most that many (the
    /// scratch reserved at load); the next waits at the head of the queue, and the request behind
    /// it waits too. The live count returns to 0 once they end.
    #[test]
    fn activations_wait_for_a_tap_seat() {
        let lim = lim(Policy::DecodeFirst, 8, 64, 0.95);
        let mut s = Scheduler::new(64);
        let seats = TAP_SEATS as u64;
        let mut ends = Vec::new();
        for id in 1..=seats + 1 {
            let (tx, rx) = channel();
            let r = Request {
                activations: vec![0],
                ..Request::new(vec![1, 2, 3], 0)
            };
            s.admit(Req::new(id, r, tx, Arc::new(AtomicBool::new(false)), 1));
            ends.push(rx);
        }
        let (r, _rx, _c) = req(9, vec![4, 5], 4, 0);
        s.admit(r);
        let ids = |plan: &StepPlan| plan.prefill.iter().map(|p| p.0).collect::<Vec<_>>();
        let plan = s.schedule(&lim, 1);
        assert_eq!(ids(&plan), (1..=seats).collect::<Vec<_>>());
        s.complete(&plan, Vec::new(), 1);
        s.check();
        let plan = s.schedule(&lim, 2);
        assert_eq!(ids(&plan), [seats + 1, 9]);
        s.complete(&plan, Vec::new(), 2);
        s.check();
        for rx in &ends {
            assert!(matches!(
                rx.try_iter().last(),
                Some(EngineEvent::Done { .. })
            ));
        }
        assert!(!s.tapping());
    }

    #[test]
    fn a_foreground_prompt_behind_seven_background_prefills_runs_next_step() {
        // Seven 1,600-token workers would hold the 512-token budget for 22 steps in queue order.
        let mut s = Scheduler::new(1024);
        let limits = lim(Policy::DecodeFirst, 8, 512, 0.95);
        let mut t = 0;
        for id in 1..=7 {
            let (mut r, _, _) = req(id, vec![1; 1600], 4, 0);
            r.class = Class::Background;
            s.admit(r);
        }
        step(&mut s, &limits, &[2], &mut t);
        let (r, _, _) = req(8, vec![3; 20], 4, 0);
        s.admit(r);
        let plan = step(&mut s, &limits, &[2], &mut t);
        assert_eq!(plan.prefill, [(8, 0, 20), (1, 512, 492)]);
        // A second chat turn waits behind the first only, not behind the workers.
        let (r, _, _) = req(9, vec![4; 600], 4, 0);
        s.admit(r);
        let plan = step(&mut s, &limits, &[2], &mut t);
        assert_eq!(plan.decode, [8]);
        assert_eq!(plan.prefill, [(9, 0, 512)]);
    }

    #[test]
    fn with_no_background_budget_a_foreground_row_never_waits_on_background_rows() {
        let mut s = Scheduler::new(256);
        s.background_rows = 0;
        let limits = lim(Policy::DecodeFirst, 8, 512, 0.95);
        let mut t = 0;
        for id in 1..=3 {
            let (mut r, _, _) = req(id, vec![1; 8], 40, 0);
            r.class = Class::Background;
            s.admit(r);
        }
        step(&mut s, &limits, &[2], &mut t);
        // No foreground row decodes: every background row does.
        assert_eq!(step(&mut s, &limits, &[2], &mut t).decode, [1, 2, 3]);
        let (r, chat, _) = req(4, vec![3; 8], 5, 0);
        s.admit(r);
        step(&mut s, &limits, &[2], &mut t); // the chat's prefill
        let mut alone = 0;
        while s.req(4).is_some() {
            assert_eq!(step(&mut s, &limits, &[2], &mut t).decode, [4]);
            alone += 1;
        }
        assert_eq!(alone, 4);
        assert!(matches!(
            chat.try_iter().last(),
            Some(EngineEvent::Done { .. })
        ));
        // The paused rows kept their KV and go on where they stopped.
        assert_eq!(step(&mut s, &limits, &[2], &mut t).decode, [1, 2, 3]);
        assert_eq!(s.reqs[&1].preemptions, 0);
    }

    #[test]
    fn a_resumed_turn_prefills_its_last_token_while_the_background_budget_binds() {
        let mut s = Scheduler::new(256);
        s.background_rows = 0;
        let limits = lim(Policy::DecodeFirst, 8, 512, 0.95);
        let mut t = 0;
        let (r, _, _) = req(1, vec![1; 8], 2, 5);
        s.admit(r);
        run_all(&mut s, &limits, &[2], &mut t);
        for id in 2..=3 {
            let (mut r, _, _) = req(id, vec![1; 8], 40, 0);
            r.class = Class::Background;
            s.admit(r);
        }
        step(&mut s, &limits, &[2], &mut t);
        // The next turn of session 5 has one token past its parked KV; the background rows decode,
        // so it prefills that token rather than join them as a foreground decode row.
        let mut prompt = s.parked[&5].tokens.clone();
        prompt.push(3);
        let (r, _, _) = req(4, prompt, 2, 5);
        s.admit(r);
        let plan = step(&mut s, &limits, &[2], &mut t);
        assert_eq!((plan.decode, plan.prefill), (vec![2, 3], vec![(4, 9, 1)]));
    }

    /// Admits `(id, prompt length, class)` in order, nothing parked.
    fn admit_all(s: &mut Scheduler, reqs: &[(u64, usize, Class)], max_tokens: u32) {
        for &(id, n, class) in reqs {
            let (mut r, _, _) = req(id, vec![1; n], max_tokens, 0);
            r.class = class;
            s.admit(r);
        }
    }

    #[test]
    fn a_background_decode_row_short_of_a_block_never_preempts_a_foreground_row() {
        // B decodes; F, newer, is mid-prompt and unplanned when B needs the pool's last block.
        let mut s = Scheduler::new(4);
        let limits = lim(Policy::DecodeFirst, 8, 32, 1.0);
        let mut t = 0;
        admit_all(&mut s, &[(1, 31, Class::Background)], 40);
        step(&mut s, &limits, &[2], &mut t);
        admit_all(&mut s, &[(2, 100, Class::Foreground)], 1);
        step(&mut s, &limits, &[2], &mut t);
        assert_eq!(s.blocks.free.len(), 0);
        step(&mut s, &limits, &[2], &mut t);
        assert_eq!((s.reqs[&1].preemptions, s.reqs[&2].preemptions), (1, 0));
    }

    #[test]
    fn a_background_chunk_short_of_blocks_never_preempts_a_foreground_row() {
        // B is mid-prompt; F, newer, decodes, and B's chunk runs first under prefill_first.
        let mut s = Scheduler::new(4);
        let limits = lim(Policy::PrefillFirst, 8, 16, 1.0);
        let mut t = 0;
        admit_all(&mut s, &[(1, 64, Class::Background)], 1);
        step(&mut s, &limits, &[2], &mut t);
        admit_all(&mut s, &[(2, 16, Class::Foreground)], 10);
        step(&mut s, &limits, &[2], &mut t);
        step(&mut s, &limits, &[2], &mut t);
        assert_eq!(s.blocks.free.len(), 0);
        let plan = step(&mut s, &limits, &[2], &mut t);
        assert_eq!(plan.decode, [2]);
        assert_eq!(s.reqs[&2].preemptions, 0);
    }

    #[test]
    fn a_foreground_chunk_that_preempts_an_older_row_skips_no_chunk_behind_it() {
        let mut s = Scheduler::new(4);
        let limits = lim(Policy::PrefillFirst, 8, 40, 1.0);
        let (fg, bg) = (Class::Foreground, Class::Background);
        admit_all(&mut s, &[(1, 48, bg), (2, 48, fg), (3, 48, fg)], 1);
        // Seat them by hand with 32, 16 and 8 tokens of KV in 2, 1 and 1 blocks: the pool is full.
        for (id, n) in [(1, 32), (2, 16), (3, 8)] {
            assert!(s.start(id, n, &limits, &mut 0).is_some());
            s.waiting.retain(|&w| w != id);
            s.running.push(id);
            // As `complete` records the chunk.
            s.reqs.get_mut(&id).unwrap().seq.n_kv = n;
            s.blocks.completed(&s.reqs[&id].seq, 0, n);
        }
        let plan = s.schedule(&limits, 1);
        assert_eq!(plan.prefill, [(2, 16, 32), (3, 8, 8)]);
        assert_eq!(s.reqs[&1].preemptions, 1);
        s.check();
    }

    /// Seats a queued request by hand with `n_kv` tokens of KV in `blocks`, as `start` and
    /// `complete` would leave it.
    fn seat(s: &mut Scheduler, id: u64, blocks: Vec<u32>, n_kv: usize) {
        s.waiting.retain(|&w| w != id);
        s.running.push(id);
        let seq = &mut s.reqs.get_mut(&id).unwrap().seq;
        (seq.blocks, seq.n_kv) = (blocks, n_kv);
        s.blocks.completed(&s.reqs[&id].seq, 0, n_kv);
    }

    #[test]
    fn a_foreground_and_a_background_request_never_preempt_each_other_forever() {
        // Review of #195, finding 1: under prefill_first a background chunk was planned when the
        // foreground decode row needed its block, so the foreground row preempted itself, and its
        // restart preempted the background row, forever.
        let limits = Limits {
            max_batch_seqs: 2,
            prefill_chunk_tokens: 512,
            policy: Policy::PrefillFirst,
            kv_preempt_watermark: 0.8,
            logit_rows: 2,
        };
        let mut s = Scheduler::new(5);
        s.configure_prefix_cache(8);
        let mut t = 0;
        let prompt = |ones, zeros| [vec![1; ones], vec![0; zeros]].concat();
        for id in 1..=2 {
            let (r, _, _) = req(id, prompt(17, 1), 1, 0);
            s.admit(r);
        }
        let (mut r, _, _) = req(3, prompt(32, 7), 11, 0);
        (r.observe, r.class) = (true, Class::Background);
        s.admit(r);
        for _ in 0..3 {
            step(&mut s, &limits, &[0; 64], &mut t);
        }
        let (mut r, _, _) = req(4, prompt(32, 7), 11, 0);
        r.observe = true;
        s.admit(r);
        for _ in 0..10_000 {
            if s.is_idle() {
                break;
            }
            step(&mut s, &limits, &[2], &mut t);
        }
        let preemptions: Vec<_> = s.reqs.values().map(|r| (r.id, r.preemptions)).collect();
        assert!(s.is_idle(), "never idle: preemptions {preemptions:?}");
    }

    #[test]
    fn a_foreground_decode_row_takes_a_background_row_planned_for_prefill() {
        let mut s = Scheduler::new(3);
        let limits = lim(Policy::PrefillFirst, 8, 16, 1.0);
        let mut t = 0;
        admit_all(&mut s, &[(1, 40, Class::Background)], 1);
        step(&mut s, &limits, &[2], &mut t);
        admit_all(&mut s, &[(2, 16, Class::Foreground)], 10);
        for _ in 0..100 {
            if s.is_idle() {
                break;
            }
            step(&mut s, &limits, &[2], &mut t);
            assert!(
                s.req(2).is_none_or(|r| r.preemptions == 0),
                "the chat lost its KV"
            );
        }
        assert!(s.is_idle());
    }

    #[test]
    fn a_live_seat_cut_preempts_the_newest_background_row_first() {
        let mut s = Scheduler::new(64);
        let mut t = 0;
        let three = lim(Policy::DecodeFirst, 3, 512, 0.95);
        for (id, class) in [
            (1, Class::Foreground),
            (2, Class::Background),
            (3, Class::Foreground),
        ] {
            admit_all(&mut s, &[(id, 8, class)], 20);
            step(&mut s, &three, &[2], &mut t);
        }
        assert_eq!(s.running, [1, 2, 3]);
        step(
            &mut s,
            &lim(Policy::DecodeFirst, 2, 512, 0.95),
            &[2],
            &mut t,
        );
        assert_eq!(
            (&s.running, &s.waiting),
            (&vec![1, 3], &VecDeque::from([2]))
        );
    }

    #[test]
    fn a_foreground_decode_row_that_preempts_two_older_rows_skips_no_row_behind_it() {
        // Review of #195, finding 3: B2's only block is shared with F4, so preempting B2 frees
        // nothing; F3 then preempts B1 too, and F4 moved two seats down.
        let mut s = Scheduler::new(4);
        let limits = lim(Policy::DecodeFirst, 8, 16, 1.0);
        let (fg, bg) = (Class::Foreground, Class::Background);
        admit_all(
            &mut s,
            &[(1, 40, bg), (2, 40, bg), (3, 17, fg), (4, 18, fg)],
            10,
        );
        let [a, shared, c, x] = [(); 4].map(|_| s.blocks.alloc().unwrap());
        s.blocks.retain(shared);
        seat(&mut s, 1, vec![a], 16);
        seat(&mut s, 2, vec![shared], 16);
        seat(&mut s, 3, vec![c], 16);
        seat(&mut s, 4, vec![shared, x], 17);
        s.check();
        let plan = s.schedule(&limits, 1);
        assert_eq!(plan.decode, [3, 4]);
        assert_eq!((s.reqs[&1].preemptions, s.reqs[&2].preemptions), (1, 1));
        s.check();
    }

    /// Guesses `max` copies of a token for request 1 alone.
    struct GuessOne;

    impl Drafter for GuessOne {
        fn propose(&mut self, id: u64, _: Namespace, _: &[u32], max: usize, out: &mut Vec<u32>) {
            if id == 1 {
                out.extend(std::iter::repeat_n(9, max));
            }
        }
    }

    fn token(id: u64) -> (u64, SampledToken) {
        let t = SampledToken {
            id: 1,
            logprob: 0.0,
            top: Vec::new(),
        };
        (id, t)
    }

    /// A scheduler of `chunk`-token prompts guessing 4 tokens for request 1, with `prompts`
    /// (request, length) admitted.
    fn guessing(
        chunk: usize,
        prompts: &[(u64, usize)],
    ) -> (Scheduler, Limits, Vec<Receiver<EngineEvent>>) {
        let mut s = Scheduler::new(64);
        (s.guess_max, s.drafter) = (4, Box::new(GuessOne));
        s.costs = Costs::fixed(|_, _| 1.0);
        let rx = (prompts.iter())
            .map(|&(id, n)| {
                let (r, rx, _) = req(id, vec![5; n], 1000, 0);
                s.admit(r);
                rx
            })
            .collect();
        (s, lim(Policy::DecodeFirst, 8, chunk, 0.95), rx)
    }

    /// An output the prefix cache would not keep (opted out, session kept off, released, cancelled,
    /// failed) is not retained for drafting either; one it would keep is.
    #[test]
    fn only_outputs_the_cache_would_keep_enter_the_drafter() {
        let drafts = |s: &mut Scheduler| {
            let mut out = Vec::new();
            s.drafter.propose(99, (0, 0), &[9, 20, 21], 8, &mut out);
            !out.is_empty()
        };
        // (opt-out, session, keep_kv, released, cancelled, fails, cache blocks) -> retained
        let cases = [
            ((true, 0, false, false, false, false, 24), true),
            ((true, 7, true, false, false, false, 24), true),
            ((false, 0, false, false, false, false, 24), false),
            ((true, 7, false, false, false, false, 24), false),
            // A turn admitted after its session was released.
            ((true, 7, true, true, false, false, 24), false),
            ((true, 0, false, false, true, false, 24), false),
            ((true, 0, false, false, false, true, 24), false),
            // No prefix_cache_blocks: no consent to retention.
            ((true, 0, false, false, false, false, 0), false),
        ];
        for ((cache, session, keep, released, cancel, fails, blocks), kept) in cases {
            let mut s = Scheduler::new(64);
            s.guess_max = 4;
            s.configure_prefix_cache(blocks);
            if released {
                s.release_session(session);
            }
            let (mut r, _rx, c) = req(1, vec![1, 2, 3], 100, session);
            (r.prefix_cache, r.keep_kv) = (cache, keep);
            s.admit(r);
            (s.reqs.get_mut(&1).unwrap().seq.tokens).extend([20, 21, 22, 23]);
            c.store(cancel, Ordering::Relaxed);
            let end = if fails {
                Err(EngineError::Cancelled)
            } else {
                Ok(Finish::Length)
            };
            s.finish(1, end, 1);
            assert_eq!(
                drafts(&mut s),
                kept,
                "{cache} {session} {keep} {released} {cancel} {fails} {blocks}"
            );
        }
    }

    /// A seeded sampled request is replayable: its guesses do not come from other requests'
    /// outputs, so the same seed gives the same tokens however much the server ran before.
    #[test]
    fn a_seeded_sampled_request_takes_no_history_drafts() {
        let mut s = Scheduler::new(64);
        s.guess_max = 4;
        s.configure_prefix_cache(24);
        let (r, _rx, _) = req(1, vec![1, 2, 3], 100, 0);
        s.admit(r);
        (s.reqs.get_mut(&1).unwrap().seq.tokens).extend([20, 21, 22, 23]);
        s.finish(1, Ok(Finish::Length), 1);
        for (seed, temperature, drafts) in [
            (None, 1.0, true),
            (Some(5), 0.0, true),
            (Some(5), 1.0, false),
        ] {
            let (mut r, _rx, _) = req(2, vec![9, 20, 21], 100, 0);
            r.sampling = SamplingParams {
                seed,
                temperature,
                ..SamplingParams::GREEDY
            };
            s.admit(r);
            let q = s.reqs.get_mut(&2).unwrap();
            q.seq.n_kv = q.prompt_len;
            // An exploring plan tries a guess wherever no cost is measured yet.
            let mut plan = StepPlan {
                decoding: 1,
                explore: true,
                ..StepPlan::default()
            };
            assert_eq!(
                s.guess(2, 4, &mut plan, None) > 0,
                drafts,
                "{seed:?} {temperature}"
            );
            s.finish(2, Err(EngineError::Cancelled), 2);
        }
    }

    /// A refused check leaves the plan a plain decode step: no guesses counted, so the engine's
    /// retry of a refused check step does not take this step's own refusal for one.
    #[test]
    fn a_refused_check_leaves_no_guesses_in_the_plan() {
        let (mut s, limits, _rx) = guessing(64, &[(1, 3)]);
        let plan = s.schedule(&limits, 1);
        s.complete(&plan, [token(1)], 1);
        let mut plan = s.schedule(&limits, 2);
        assert_eq!(plan.guessed, 4);
        s.uncheck(&mut plan, 1);
        assert_eq!(plan.guessed, 0);
        assert_eq!((plan.decode.as_slice(), plan.prefill.len()), (&[1][..], 0));
    }

    /// A check refused for want of room counts as a round that kept nothing, so a sequence that
    /// keeps finding no room stops asking the drafter, as one whose guesses keep missing does.
    #[test]
    fn refused_checks_back_off() {
        let (mut s, limits, _rx) = guessing(64, &[(1, 3)]);
        s.checkable = Some(true);
        let plan = s.schedule(&limits, 1);
        s.complete(&plan, [token(1)], 1);
        for t in 2..10 {
            let mut plan = s.schedule(&limits, t);
            assert!(!plan.prefill.is_empty(), "round {t} checks");
            s.uncheck(&mut plan, 1);
            s.complete(&plan, [token(1)], t);
        }
        let plan = s.schedule(&limits, 10);
        assert!(
            plan.prefill.is_empty(),
            "32 guesses refused: off for a while"
        );
        assert_eq!(s.checkable, Some(true));
    }

    /// A prompt's one-token remainder is a decode row, never a check: a check keeps no prompt
    /// counters and publishes no prefix. The token after it checks.
    #[test]
    fn a_prompts_last_token_is_not_checked() {
        let (mut s, limits, _rx) = guessing(4, &[(1, 5)]);
        let plan = s.schedule(&limits, 1);
        assert_eq!(plan.prefill, [(1, 0, 4)]);
        s.complete(&plan, [], 1);
        let plan = s.schedule(&limits, 2);
        assert_eq!(plan.decode, [1]);
        assert_eq!(plan.guessed, 0);
        s.complete(&plan, [token(1)], 2);
        assert_eq!(s.computed_prompt_tokens, 5);
        assert!(s.schedule(&limits, 3).guessed > 0);
    }

    /// A check beside a decode row is no prompt chunk, so it arms no pacing wait.
    #[test]
    fn a_check_beside_a_decode_row_does_not_pace() {
        let (mut s, limits, _rx) = guessing(64, &[(1, 3), (2, 3)]);
        s.decode_share = 0.5;
        let plan = s.schedule(&limits, 1);
        s.complete(&plan, [token(1), token(2)], 1);
        let plan = s.schedule(&limits, 2);
        assert_eq!((plan.guessed, plan.decode.as_slice()), (4, &[2][..]));
        assert_eq!(s.chunk_at, None);
    }

    /// With no step cost measured, a step checks only when it explores: the first that may check,
    /// then every [`EXPLORE`]-th, deterministically.
    #[test]
    fn unmeasured_costs_explore_every_nth_step() {
        let (mut s, limits, _rx) = guessing(64, &[(1, 3)]);
        s.costs = Costs::default();
        let plan = s.schedule(&limits, 1);
        s.complete(&plan, [token(1)], 1);
        let checked: Vec<u64> = (0..2 * EXPLORE)
            .filter(|&t| {
                let plan = s.schedule(&limits, t + 2);
                s.complete(&plan, [token(1)], t + 2);
                plan.guessed > 0
            })
            .collect();
        assert_eq!(checked, [0, EXPLORE]);
    }

    /// Proposes `max` guesses for every request.
    struct GuessAll;

    impl Drafter for GuessAll {
        fn propose(&mut self, _: u64, _: Namespace, _: &[u32], max: usize, out: &mut Vec<u32>) {
            out.extend(std::iter::repeat_n(9, max));
        }
    }

    /// Runs 8 chats of [`GuessAll`] on the table `ns` for `steps` decode steps after their
    /// prompts: the steps that held a check and the requests they checked.
    fn eight_chats(ns: impl Fn(usize, usize) -> f64, steps: u64) -> (Scheduler, u64, Vec<u64>) {
        // Blocks to spare: the guard on free blocks never takes a sequence's turn here.
        let mut s = Scheduler::new(256);
        (s.guess_max, s.drafter) = (4, Box::new(GuessAll));
        s.costs = Costs::fixed(ns);
        let _rx: Vec<_> = (1..=8)
            .map(|id| {
                let (r, rx, _) = req(id, vec![5; 3], 1000, 0);
                s.admit(r);
                rx
            })
            .collect();
        let limits = lim(Policy::DecodeFirst, 8, 64, 0.95);
        let (mut checks, mut seen) = (0, Vec::new());
        for t in 0..steps + 1 {
            let plan = s.schedule(&limits, t + 1);
            if plan.guessed > 0 {
                checks += 1;
                seen.extend(plan.prefill.iter().map(|p| p.0));
            }
            let ids = (plan.decode.iter().copied()).chain(plan.prefill.iter().map(|p| p.0));
            let sampled: Vec<_> = ids.map(token).collect();
            s.complete(&plan, sampled, t + 1);
        }
        seen.sort_unstable();
        seen.dedup();
        (s, checks, seen)
    }

    /// At 8 chats the measured pick says no check at a prior acceptance, so a sequence whose
    /// guesses would be good can only show it by checking. The retries are bounded per step, not
    /// per sequence: one guess in one step of every [`EXPLORE`], taking the sequences in turn.
    #[test]
    fn a_sequence_stuck_at_k0_retries() {
        let steps = 9 * EXPLORE;
        let (s, checks, seen) = eight_chats(|d, g| 1.0 + 0.25 * (d + g) as f64, steps);
        assert!(checks <= steps / EXPLORE + 1, "{checks} of {steps} steps");
        assert_eq!(seen, [1, 2, 3, 4, 5, 6, 7, 8]);
        let k = s.tot.guess_k;
        let tried: u64 = k[1..].iter().sum();
        assert!(s.tot.guess_rounds >= 8, "{k:?}");
        assert!(k[0] >= 10 * tried, "plain steps dominate: {k:?}");
    }

    /// One-guess probes can show at most an acceptance of 11/12: where only a perfect guesser
    /// would check, a probe could never flip the pick, so none runs.
    #[test]
    fn no_probe_where_it_could_never_flip_the_pick() {
        let (s, checks, _) = eight_chats(|_, g| if g == 0 { 1.0 } else { 1.45 }, 9 * EXPLORE);
        assert_eq!(checks, 0, "{:?}", s.tot.guess_k);
    }

    /// Past [`Scheduler::guess_chats`] decoding sequences a step checks nothing: two chats at a
    /// limit of one decode plainly, at a limit of two the step checks.
    #[test]
    fn no_check_past_the_guessing_chats() {
        for (chats, guessed) in [(1, 0), (2, 4)] {
            let (mut s, limits, _rx) = guessing(64, &[(1, 3), (2, 3)]);
            s.guess_chats = chats;
            let plan = s.schedule(&limits, 1);
            s.complete(&plan, [token(1), token(2)], 1);
            let plan = s.schedule(&limits, 2);
            assert_eq!((plan.guessed, plan.decoding), (guessed, 2), "{chats}");
        }
    }

    /// Proposes `max` guesses of the token [`token`] samples, for every request: a check keeps
    /// them all when each of its rows samples it.
    struct GuessRight;

    impl Drafter for GuessRight {
        fn propose(&mut self, _: u64, _: Namespace, _: &[u32], max: usize, out: &mut Vec<u32>) {
            out.extend(std::iter::repeat_n(1, max));
        }
    }

    /// Completes `plan` with a sampled token for each row it feeds a request: a check's last
    /// token and guesses, a prompt chunk's one, a decode row's one.
    fn complete_right(s: &mut Scheduler, plan: &StepPlan, t: u64) {
        let checked = |id: u64| s.reqs[&id].guesses + 1;
        let rows: Vec<_> = (plan.decode.iter().map(|&id| (id, 1)))
            .chain(plan.prefill.iter().map(|p| match s.reqs[&p.0].guesses {
                0 => (p.0, 1),
                _ => (p.0, checked(p.0)),
            }))
            .collect();
        let sampled: Vec<_> = (rows.into_iter())
            .flat_map(|(id, n)| std::iter::repeat_n(token(id), n))
            .collect();
        s.complete(plan, sampled, t);
    }

    /// A new request starts at the load's acceptance, not a fixed 1/2: on the fast Metal checks'
    /// one-chat costs (27 Nov, draft included), where a check pays only above about 0.6, a request
    /// after well-kept rounds checks at its first decode step instead of decoding plainly until
    /// a probe.
    #[test]
    fn a_new_request_starts_at_the_loads_acceptance() {
        let ns = |_, g| match g {
            0 => 18.5,
            1 => 29.9,
            2 => 34.2,
            4 => 42.3,
            _ => 58.3,
        };
        let mut s = Scheduler::new(256);
        (s.guess_max, s.drafter) = (4, Box::new(GuessRight));
        s.costs = Costs::fixed(ns);
        let limits = lim(Policy::DecodeFirst, 8, 64, 0.95);
        // A fresh load's first request starts at 1/2, which the costs leave plain.
        let (r, _rx1, _) = req(1, vec![5; 3], 40, 0);
        s.admit(r);
        assert_eq!(s.reqs[&1].hits.acceptance(), 0.5);
        assert_eq!(s.costs.pick(1, 0, 0.0, 0.5, 4, false), 0);
        let mut t = 1;
        while s.reqs.contains_key(&1) {
            let plan = s.schedule(&limits, t);
            complete_right(&mut s, &plan, t);
            t += 1;
        }
        assert!(s.tot.guess_rounds > 0 && s.tot.guesses_kept == s.tot.guesses_tried);
        let a = s.accept.acceptance();
        assert!(a > 0.8, "{a}");
        let (r, _rx2, _) = req(2, vec![5; 3], 40, 0);
        s.admit(r);
        assert_eq!(s.reqs[&2].hits.acceptance(), a);
        let plan = s.schedule(&limits, t);
        complete_right(&mut s, &plan, t);
        t += 1;
        // Its first decode step, not an exploring one, checks all four guesses.
        let plan = s.schedule(&limits, t);
        assert!(!plan.explore);
        assert_eq!(plan.guessed, 4);
    }

    /// Samples count a check's sequence as decoding and its rows as no prompt tokens.
    #[test]
    fn a_check_counts_as_decode_in_samples() {
        let (mut s, limits, _rx) = guessing(64, &[(1, 3), (2, 3)]);
        let plan = s.schedule(&limits, 1);
        assert_eq!(plan.counts(), (0, 2, 6));
        s.complete(&plan, [token(1), token(2)], 1);
        assert_eq!(s.schedule(&limits, 2).counts(), (2, 0, 0));
    }
}
