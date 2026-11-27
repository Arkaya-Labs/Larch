//! GGUF models over larch-kernels (architecture 8): the [`Model`] trait the engine drives, the
//! [`Transformer`] that implements it on any backend, [`FakeModel`] for engine tests, [`tiny`], a
//! generated test model, [`learn`], the native router-delta trainer, [`oracle`], the comparison with
//! llama.cpp (architecture 15), and [`vision`], image input.

pub mod card;
pub mod chat;
mod config;
mod dflash;
mod experts;
mod fake;
pub mod gguf;
pub mod learn;
pub mod oracle;
pub mod projector;
pub mod q2k;
pub mod qwen_vision;
pub mod tiny;
mod tokcheck;
mod tokenizer;
mod transformer;
pub mod vision;

pub use chat::{ChatTemplate, Message, ToolCall, ToolMessage};
pub use config::{Arch, ModelConfig, MoeCfg, RopeCfg};
pub use dflash::{DflashDrafter, DraftCache, DraftShape};
pub use experts::Histogram;
pub use fake::FakeModel;
pub use tokcheck::{TokcheckError, tokcheck};
pub use tokenizer::{Detokenizer, INKLING_TAGS, Tokenizer};
pub use transformer::{
    ExpertPlan, MMVQ_MAX_ROWS, MTP_GUESSES, MtpCache, MtpDrafter, PREFILL_STREAM_ROWS, Placement,
    RouteCase, Transformer,
};

use larch_kernels::{DType, KError, KvPool, Stream, Tensor};

/// One step's inputs, built by the executor; every tensor is i32 on the model's device. Rows are the
/// decode rows (one per decoding sequence) followed by the prefill rows (each prefilling sequence's
/// chunk, in sequence order).
pub struct StepBatch<'a> {
    /// `[rows]`
    pub token_ids: Tensor<'a>,
    /// `[rows]` absolute positions; or on a model with rope sections (Qwen 3.5's M-RoPE), `[rows, 4]`
    /// each row's (t, h, w, e): an image row's grid place, a text row's (p, p, p, 0), which ropes as
    /// `[rows]`'s p bit for bit.
    pub positions: Tensor<'a>,
    /// `[rows]` KV slot of each row (block id × block_tokens + offset); -1 skips the write (padding).
    pub slot_ids: Tensor<'a>,
    /// Decode rows, first in every tensor above.
    pub decode_rows: u32,
    /// Prefill rows, after the decode rows.
    pub prefill_rows: u32,
    /// `[decode seqs + prefill seqs, max_blocks]`, in row order.
    pub block_table: Tensor<'a>,
    /// `[decode_rows]` positions each decode row attends to, its own included; 0 = padding.
    pub seq_lens: Tensor<'a>,
    /// `[prefill seqs]` position of each chunk's first row.
    pub prefill_q_start: Tensor<'a>,
    /// `[prefill seqs]` rows in each chunk.
    pub prefill_q_len: Tensor<'a>,
    /// `[n]` rows whose logits are wanted (the last row of each sequence that samples).
    pub logits_rows: Tensor<'a>,
    /// `larch_attn_decode`'s split-K count; 0 lets the backend pick.
    pub attn_splits: u32,
    /// The prefill chunks are check chunks (a decoding sequence's last token plus guesses): the model
    /// must compute every row bit-identically to a one-row decode at the same `attn_splits`.
    pub check: bool,
}

/// The recurrent layers' view of one step (Qwen 3.5's Gated DeltaNet), whose state lives in slots
/// of the model, one per sequence, not in the KV pool. It covers every row of the [`StepBatch`] in
/// row order, padding rows included: `q_lens[s]` rows per sequence, sequence `s` in slot
/// `slots[s]`, its state before the first of those rows and after the last. The slots of one step
/// are distinct and below [`Model::state_slots`]; device values, as `StepBatch`'s.
pub struct StateBatch<'a> {
    /// `[seqs]` i32.
    pub slots: Tensor<'a>,
    /// `[seqs]` i32, summing to the batch's rows.
    pub q_lens: Tensor<'a>,
}

/// Activations a forward pass copies out beside the logits. A graph capture freezes `rows` and
/// `out`, so inside one both must sit in buffers every replay reuses (a drafter's taps; the
/// engine's activations requests stay eager).
pub struct Taps<'a> {
    /// Layers (0-based) whose output residual stream is copied, in `out`'s block order.
    pub layers: &'a [u16],
    /// Also copy the final hidden state: the output norm's f16 result widened to f32 (f16-rounded,
    /// as the head reads it).
    pub final_state: bool,
    /// `[n]` i32 batch rows to copy, such as each tapped sequence's last row. Each must be below the
    /// batch's rows: device values, which no GPU op checks (as `StepBatch::logits_rows`).
    pub rows: Tensor<'a>,
    /// f32 `[blocks * n, d_model]`, blocks = `layers.len()` (+ 1 for the final state): block `k`
    /// (rows `k * n .. (k + 1) * n`) holds `layers[k]` at `rows`, the final state the last block.
    pub out: Tensor<'a>,
}

impl<'a> Taps<'a> {
    /// `out`'s blocks, once the layers, dtypes and shapes are checked against `c`. The row values are
    /// not checked (see `rows`).
    pub(crate) fn blocks(&self, c: &ModelConfig) -> Result<Vec<Tensor<'a>>, KError> {
        let n = match self.rows.shape() {
            &[n] if n > 0 && self.rows.dtype() == DType::I32 => n,
            _ => return Err(KError::InvalidArg),
        };
        let blocks = self.layers.len() + usize::from(self.final_state);
        let shape = i64::try_from(blocks).ok().and_then(|b| b.checked_mul(n));
        let fits = self.layers.iter().all(|&l| u32::from(l) < c.n_layers)
            && blocks > 0
            && self.out.dtype() == DType::F32
            && shape.is_some_and(|r| self.out.shape() == [r, i64::from(c.d_model)]);
        if !fits {
            return Err(KError::InvalidArg);
        }
        (0..blocks as i64)
            .map(|k| self.out.rows(k * n, n))
            .collect()
    }
}

/// Per-layer f32 deltas added to MoE router logits before softmax/top-k (doc I4). Matches
/// larch-router-candidate/1: `layers` (unique MoE layer indices), `experts`, one vector per layer.
/// With `rank != 0` (doc I5) the vectors are expert adapters instead, as in `learn::RouterCandidate`:
/// layer `i` holds, for each expert of `ids[i]`, B `[d_model, rank]` then A `[rank, expert ffn]`, and
/// the expert's down projection becomes `W_down + B A`; the router is untouched.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouterDeltas {
    /// MoE layer indices, unique.
    pub layers: Vec<u16>,
    /// Experts per layer, the model's.
    pub experts: u32,
    /// `[experts]` per entry of `layers`, in its order.
    pub deltas: Vec<Vec<f32>>,
    /// 0 for router deltas; the adapters' rank otherwise.
    pub rank: u32,
    /// The adapted experts per layer, ascending, parallel to `layers`; empty for router deltas.
    pub ids: Vec<Vec<u32>>,
}

/// Rows of a step whose input is an embedding, not a token's: an image's rows, which the token
/// table's row is replaced by after the lookup. Each run copies `table` (f32 `[n, d_model]`) into
/// the batch rows `row..row + n`, gathering through `ids` (i32 `[n]`, the identity `0..n`).
pub struct EmbedRun<'a> {
    pub row: u32,
    pub table: Tensor<'a>,
    pub ids: Tensor<'a>,
}

/// What the engine drives: a config and one forward pass per step.
pub trait Model: Send {
    fn config(&self) -> &ModelConfig;

    /// One forward pass over a mixed batch: writes K/V of every row into `kv` and f32 logits
    /// `[logits_rows, vocab]` into `logits`.
    fn forward(
        &mut self,
        s: &Stream,
        batch: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError>;

    /// [`Model::forward`] that also writes the activations `taps` names into `taps.out`; `None` is
    /// exactly `forward`. A model without taps refuses `Some` with `Unsupported`.
    // ponytail: taps ride beside `forward` because 24 test models outside this crate implement it,
    // 22 of them in other lanes; fold them into `forward` when those lanes change their test models.
    fn forward_taps(
        &mut self,
        s: &Stream,
        batch: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
        taps: Option<&Taps>,
    ) -> Result<(), KError> {
        match taps {
            None => self.forward(s, batch, kv, logits),
            Some(_) => Err(KError::Unsupported),
        }
    }

    /// [`Model::forward_state`] (with `state` when the model has recurrent layers) whose `embeds`
    /// rows take their input from the runs' tables. A model without embedding input refuses with
    /// `Unsupported` before any work.
    #[allow(clippy::too_many_arguments)]
    fn forward_embeds(
        &mut self,
        _s: &Stream,
        _batch: &StepBatch,
        _state: Option<&StateBatch>,
        _kv: &KvPool,
        _logits: &Tensor,
        _taps: Option<&Taps>,
        _embeds: &[EmbedRun],
    ) -> Result<(), KError> {
        Err(KError::Unsupported)
    }

    /// Sequence slots of recurrent state the model holds; 0 for a model without recurrent layers,
    /// whose state is all in the KV pool.
    fn state_slots(&self) -> u32 {
        0
    }

    /// Zeroes `slot`: the state of a sequence before its position 0.
    fn reset_state(&mut self, _s: &Stream, _slot: u32) -> Result<(), KError> {
        Err(KError::Unsupported)
    }

    /// Copies slot `from` into slot `to`: `to` then runs on from where `from` stands.
    fn copy_state(&mut self, _s: &Stream, _from: u32, _to: u32) -> Result<(), KError> {
        Err(KError::Unsupported)
    }

    /// Bytes of one state slot as [`Model::save_state`] writes them; 0 without recurrent state.
    fn state_size(&self) -> usize {
        0
    }

    /// Writes `slot`'s state into `out` ([`Model::state_size`] bytes, else `InvalidArg`), for a
    /// reload to write back with [`Model::load_state`].
    fn save_state(&self, _s: &Stream, _slot: u32, _out: &mut [u8]) -> Result<(), KError> {
        Err(KError::Unsupported)
    }

    /// Writes [`Model::save_state`]'s bytes into `slot`.
    fn load_state(&mut self, _s: &Stream, _slot: u32, _bytes: &[u8]) -> Result<(), KError> {
        Err(KError::Unsupported)
    }

    /// The device bytes the host-expert streaming pool wants and the prefill rows it starts at
    /// ([`Placement::stream_rows`]); (0, 0) for a model with none to allocate.
    fn stream_pool_want(&self) -> (u64, u32) {
        (0, 0)
    }

    /// Allocates the streaming pool [`Model::stream_pool_want`] names; `Ok(false)` with none to
    /// allocate. On `Err(KError::Oom)` there is no pool and the steps it would serve stay on the
    /// CPU path; any other error is the device's.
    fn alloc_stream_pool(&mut self, _s: &Stream) -> Result<bool, KError> {
        Ok(false)
    }

    /// Between steps: grows the recurrent state to the most slots up to `want` whose buffers fit
    /// `room` bytes beside the current ones, keeping every slot's state; the slots it now holds.
    fn grow_state(&mut self, _s: &Stream, _want: u32, _room: u64) -> Result<u32, KError> {
        Err(KError::Unsupported)
    }

    /// Before any step of a load that will guess: `spares` slots of the spare state
    /// [`Model::begin_check`] takes, which costs device memory, so a load that never checks does
    /// without. Zeroes the state.
    fn alloc_checks(&mut self, _s: &Stream, _spares: u32) -> Result<(), larch_records::LoadError> {
        Ok(())
    }

    /// Before a check step of `rows` rows on recurrent-state `slot`: the next forward on it may be rewound.
    fn begin_check(&mut self, _s: &Stream, _slot: u32, _rows: u32) -> Result<(), KError> {
        Err(KError::Unsupported)
    }
    /// After that check step: keep the state as it stood after the first `kept` rows (1..=rows).
    fn end_check(&mut self, _s: &Stream, _slot: u32, _kept: u32) -> Result<(), KError> {
        Err(KError::Unsupported)
    }

    /// Before a graph replay of a guessing load, with the step's slots and rows per sequence: routes
    /// them as [`Model::forward_state`] would for an eager step (each sequence's physical slot, a
    /// begun check's row snapshots, a moved slot's return home) and writes that where the capture
    /// reads it. `Unsupported` for a model without recurrent state.
    fn route_table(&mut self, _s: &Stream, _slots: &[i32], _lens: &[i32]) -> Result<(), KError> {
        Err(KError::Unsupported)
    }

    /// [`Model::forward_taps`] with each row's recurrent state slot, which a model with recurrent
    /// layers needs (its `forward` refuses them `InvalidArg`); others ignore `state`.
    fn forward_state(
        &mut self,
        s: &Stream,
        batch: &StepBatch,
        _state: &StateBatch,
        kv: &KvPool,
        logits: &Tensor,
        taps: Option<&Taps>,
    ) -> Result<(), KError> {
        self.forward_taps(s, batch, kv, logits, taps)
    }

    /// The host step a [`larch_kernels::Graph::capture_pieces`] capture of a forward pass cut at
    /// `tag`, run between the replayed pieces around it, for the step's first `live` rows: the
    /// padding rows after them keep stale outputs, which nothing reads. `Unsupported` for a model
    /// that never cuts.
    fn run_host(&self, _s: &Stream, _tag: u64, _live: u32) -> Result<(), KError> {
        Err(KError::Unsupported)
    }

    /// The host step before each forward pass, eager or a replay's first piece: the token
    /// embedding of the step's `ids` (every row, in row order, as `StepBatch::token_ids`) into the
    /// pass's input, for a model that gathers it in host memory, which a capture leaves out. The
    /// next eager pass of as many rows takes those rows instead of reading its ids back from the
    /// device, so a caller that calls it once calls it before every pass. Nothing for any other
    /// model.
    fn embed_tokens(&self, _s: &Stream, _ids: &[i32]) -> Result<(), KError> {
        Ok(())
    }

    /// Adds `d` to the router logits of every later forward; `None` restores the base model exactly.
    /// A model without routers, or one that cannot take deltas, answers `Unsupported`.
    fn set_router_deltas(&mut self, _s: &Stream, _d: Option<&RouterDeltas>) -> Result<(), KError> {
        Err(KError::Unsupported)
    }

    /// The resident [`Transformer`], for learning between steps; `None` for any other model.
    fn transformer(&self) -> Option<&Transformer> {
        None
    }

    /// The token embedding and the output head, `[vocab, d_model]` each, which a [`DflashDrafter`]
    /// shares; `None` for a model that does not lend them.
    fn embed_and_head(&self) -> Option<(Tensor<'_>, Tensor<'_>)> {
        None
    }
}
