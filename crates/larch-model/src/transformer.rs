//! The forward pass (architecture 8): per layer rmsnorm → q/k/v (+bias) → qk_norm → rope → kv_write →
//! attention (decode rows, then prefill rows) → o → add → rmsnorm → gate/up → silu_mul → down → add;
//! then the final norm and the output projection on the logits rows only. An MoE layer's gate/up and
//! down are its routed experts: router → moe_route → gate/up per chosen expert → silu_mul → down per
//! chosen expert, weighted and summed in the same call. Qwen 3.5's hybrid layers are in
//! [`qwen35`]. Activations are f16 and the residual stream f32 (architecture 6.1). Taps gather
//! residual rows after their layers and the final norm's rows with `embed`, the row copy the logits
//! rows already use.

use std::cell::{Cell, RefCell};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Instant;

use larch_kernels::{
    Backend, Buffer, CPU_COMPUTE, DType, Device, DeviceProps, HostCopy, KError, KvPool, Mmap,
    Stream, Tensor, ops,
};
use larch_records::LoadError;

use crate::card::{self, Route};
use crate::experts::Histogram;
use crate::gguf::{self, Gguf, code};
use crate::{
    Arch, EmbedRun, Model, ModelConfig, MoeCfg, RouterDeltas, StateBatch, StepBatch, Taps,
};

#[path = "qwen35.rs"]
mod qwen35;
#[path = "qwen35_mtp.rs"]
mod qwen35_mtp;
pub use qwen35_mtp::{MTP_GUESSES, MtpCache, MtpDrafter};

/// CUDA step-row ceiling for Q4_K/Q6_K in models at least 3584 wide; otherwise 16.
// ponytail: CUDA R=32 from cold Q4_K/Q6_K layer totals on sm_120 (11 Oct 2026);
// re-measure on other CUDA architectures and when mmvq or the gemm path changes.
pub const MMVQ_MAX_ROWS: i64 = 32;
/// ABI 2 limits each mmvq call to 16 rows.
const MMVQ_SLICE_ROWS: i64 = 16;
/// larch.h's mmvq_verify limit.
const VERIFY_ROWS: i64 = 32;
/// Rows a check step's gemm takes at a time: Metal's GEMV_M, up to which gemm runs vv_gemv, whose rows
/// never mix, as one-row decode's router does; past it, tile kernels that round otherwise.
const GEMV_ROWS: i64 = 8;
/// Rows a check step's mmvq_id takes at a time: CPU_COMPUTE's and Metal's mmvq_id group a call's
/// (row, slot) pairs by expert from 4 pairs per expert, which rounds otherwise; 3 rows of k distinct
/// choices hold under 4 per expert, so they run per pair, as a one-row decode does.
// CUDA's mmvq_id groups pairs by expert from 33 rows, so on CUDA too the slices are what keeps check steps on the
// per-pair path (exact speculation). Metal's check-step mmvq_id runs per pair at any row count (drafter mode), so a
// Metal device's slices are [`METAL_ID_ROWS`].
const ID_ROWS: i64 = 3;
/// Rows a Metal check step's mmvq_id takes at a time: 32 rows a call, mmvq_verify's most.
const METAL_ID_ROWS: i64 = 32;
/// The staging chunk of the weight upload.
const CHUNK: u64 = 64 << 20;
/// Rows from which CUDA's mmvq_id may group its pairs by expert, which the fused MoE entries do not reproduce.
const FUSED_ID_ROWS: i64 = 33;

/// The fused entries on by default (`LARCH_FUSED` unset): each at least 1.3x its op sequence at one row in
/// kernels/tests/fused_bench.cu (CUDA graph replays, the hybrid MoE target's shapes), and those replacing a gemm only
/// once all-mode 30-prompt parity shows no drift from the unfused path (mean KL against it under 0.002). moe_in gives
/// the unfused bytes at up to 8 rows on CUDA (its gemms are gemm_rows_k's there): no drift by construction.
// ponytail: sm_120, 15 Nov 2026 (5070 Ti), us unfused/fused at 1 row: gate_up 31.4/21.3 (1.47x), down 23.5/15.6 (1.51x)
// on the Q3_K-down layers and 19.4/15.3 (1.27x) on the Q4_K-down ones, moe_in 17.7/13.6 and 20.0/13.8, attn_in
// 11.1/5.3 (2.08x); off at 1.0x or less: gdn_in 35.0/43.8, gdn 8.4/9.9, proj_add 7.6/10.7 (Q3_K) and 7.2/24.5 gated.
// The off ones repeat work in every block (gdn_in's norm, the gated proj_add's sigmoid) and decode a column a warp
// (mmvq_col) where mmvq_k shares x over several; re-measure when either changes, on another arch, or another backend
// (these defaults apply on CUDA only; Metal's are [`METAL_FUSED`]).
#[derive(Clone, Copy)]
struct Fused {
    /// `fused_moe_gate_up` and `fused_moe_down`.
    moe: bool,
    /// `fused_moe_in`, which replaces the router's and the shared gate input's gemms.
    moe_in: bool,
    /// `fused_gdn_in`, which replaces alpha's and beta's gemms where they are f16.
    gdn_in: bool,
    /// `fused_proj_add`: the state-space, attention and dense down projections with the residual add.
    proj_add: bool,
    /// `fused_attn_in`: the gated Q's split, the q and k norms, rope and the KV write.
    attn_in: bool,
    /// `fused_gdn`: the conv, delta rule, per-head norm and gate of a Gated DeltaNet layer.
    gdn: bool,
}
const FUSED: Fused = Fused {
    moe: true,
    moe_in: true,
    gdn_in: false,
    proj_add: false,
    attn_in: true,
    gdn: false,
};
/// Metal's fused entries on by default: attn_in alone, the only one Metal runs (the others answer UNSUPPORTED). It
/// gives the sequence's bytes (kernels/tests/fused.cpp) and drops the gated split's two selector gemms.
const METAL_FUSED: Fused = Fused {
    attn_in: true,
    ..NO_FUSED
};
const NO_FUSED: Fused = Fused {
    moe: false,
    moe_in: false,
    gdn_in: false,
    proj_add: false,
    attn_in: false,
    gdn: false,
};

/// What the MoE block's inputs leave for its experts: the route's ids and weights, and the shared expert's down
/// weight, activation and gate.
type MoeIn<'a> = (Tensor<'a>, Tensor<'a>, Option<(W, Tensor<'a>, Tensor<'a>)>);

fn mmvq_crossover(backend: Backend, d_model: u32) -> i64 {
    // ponytail: Metal/HIP/CPU ceiling 16; measure Metal/HIP before raising.
    // ponytail: unmeasured widths below 3584 retain 16; measure their crossover before raising.
    if backend == Backend::Cuda && d_model >= 3584 {
        MMVQ_MAX_ROWS
    } else {
        MMVQ_SLICE_ROWS
    }
}

fn use_mmvq(crossover: i64, w: W, rows: i64) -> bool {
    // ponytail: only Q4_K/Q6_K crossovers measured; measure other types before raising their 16.
    rows <= MMVQ_SLICE_ROWS || (rows <= crossover && matches!(w.dtype, DType::Q4K | DType::Q6K))
}

// ponytail: measured on sm_120; elsewhere only where the card's profile proved it (`larch routes-check`).
fn quant_gemm_device(backend: Backend, arch_code: u32, profiled: bool) -> bool {
    backend == Backend::Cuda && (arch_code == 120 || profiled)
}

/// The fused entries on by default with `LARCH_FUSED` unset: [`FUSED`] where they were measured, Metal's [`METAL_FUSED`].
// ponytail: sm_120 only, like quant_gemm_device: on sm_86/89 the H2 decoder refuses fused_moe_in and the shared
// fused_moe_down, so those MoE steps would run fallback placements no GPU run has covered. Add 86 and 89 when the
// 3050 and 4060 pass the fused ctest, `larch parity --mode all` and a fused-on vs off bench there.
fn fused_defaults(backend: Backend, arch_code: u32) -> Fused {
    match backend {
        Backend::Cuda if arch_code == 120 => FUSED,
        Backend::Metal => METAL_FUSED,
        _ => NO_FUSED,
    }
}

/// The CUDA archs where activation INT8 defaults on: its parity passed and serving gained there.
// ponytail: sm_86 (3050) and sm_120 (5070 Ti), Oct 2026: on the 3050 m3 W4 at 16 gains 18.6 % tok/s, W6 S_tg at
// npl 16 28.8 % and S_pp 10 %, m6 W4 at 16 7.2 % (#156; `LARCH_QUANT_I8=0` forces it off). The sm_86 numbers are a
// 16-SM 3050 Laptop's: re-measure on a large sm_86 part (e.g. a 3090, 82 SMs) when one enters a bench. sm_89 by
// judgment, not measured: Ada's integer mma.sync is Ampere's, and it keeps the full-grid Q4_K Plain route, the same
// exact-integer kernel as sm_86; drop 89 if `larch parity --mode all` fails or its serve rows lose on the 4060.
const QUANT_I8_DEFAULT_ARCHES: [u32; 3] = [86, 89, 120];

// Integer tensor cores are a capability requirement; `switch` (`LARCH_QUANT_I8`) overrides the
// measured-arch default. Unsupported shapes retain the existing projection path, and the one-row MMVQ
// remains the single-user route.
fn quant_i8_device(backend: Backend, arch_code: u32, switch: Option<bool>) -> bool {
    backend == Backend::Cuda
        && arch_code >= 80
        && switch.unwrap_or(QUANT_I8_DEFAULT_ARCHES.contains(&arch_code))
}

/// The int8 tile a projection takes when activation INT8 is on (the arch default or LARCH_QUANT_I8=1).
#[derive(Clone, Copy, Debug, PartialEq)]
enum I8 {
    /// One CTA per output tile and query tile.
    Plain,
    /// Stream-K over K, for a decode grid short of the SMs.
    Split,
}

/// Decides by the whole grid, output tiles times query rows, against the routes it would replace:
/// cold-weight crossovers on sm_120 (15 Oct 2026, m1/m5 Q4_K and Q6_K, m8 Q8_0 shapes). Other
/// architectures keep only the earlier full-grid Q4_K route, plus each rule `card` (the card's
/// profile) proved, minus [`card::EXCLUDED`]. The op still refuses what it cannot run.
fn quant_i8_route(
    w: W,
    rows: i64,
    compute_units: u32,
    arch_code: u32,
    card: &[Route],
) -> Option<I8> {
    let tiles = (w.shape[0] + 127) / 128;
    let units = i64::from(compute_units);
    if rows < 8 || w.shape[0] <= 0 || w.shape[1] % 128 != 0 || units == 0 {
        return None;
    }
    // ponytail: sm_86 measured the rules below on the 3050 (16 SMs): every routed shape won its microbenchmark (Q6_K
    // down 1.08-3.75x, head 1.86-4.78x, k/v Stream-K 1.13-2.94x), but m3 `larch parity --mode decode --parallel
    // 32` misses decisive top-1 (0.984) on the 512-wide k/v Stream-K route at 28-32 rows, which card::EXCLUDED holds
    // off sm_120 until it passes; a card takes the other rules as its profile proves them (`larch routes-check`).
    let on = |route| arch_code == 120 || card.contains(&route);
    let full_q4 = (w.dtype == DType::Q4K && tiles >= units).then_some(I8::Plain);
    // ponytail: Q8_0 stays off: m8's 896-wide K loses to mmvq and cuBLAS at 8-512 rows; measure a
    // Q8_0 model with k >= 4096 when one enters a bench.
    if !matches!(w.dtype, DType::Q4K | DType::Q6K) {
        return None;
    }
    if rows <= 32 {
        // ponytail: Stream-K beats mmvq from 112 tile-rows (1024-wide k/v from 14 rows, m5's 512-wide
        // k/v from 28: 12.3 vs 23.3 us at 32), and past 16 rows (two J16 query tiles) it beats the
        // plain J32 tile on every short grid (ffn_down 94 vs 117 us Q4_K, 104 vs 193 Q6_K);
        // re-measure when either changes.
        return if tiles >= units {
            if on(Route::I8FullGrid) {
                Some(I8::Plain)
            } else {
                full_q4
            }
        } else if tiles * rows >= 112
            && on(Route::I8StreamK)
            && card::excluded(Route::I8StreamK, w.shape[0], rows, arch_code).is_none()
        {
            Some(I8::Split)
        } else {
            None
        };
    }
    if !on(Route::I8Long) {
        return full_q4;
    }
    // ponytail: past 32 rows, outputs of 8 tiles or fewer (k/v) stay on dequant + cuBLAS (2-4x
    // faster; Q6_K v 1.4-4.7x on every tile), and so does Q6_K from 96 rows: its J64/J112 tiles
    // beat cuBLAS at 65-256 rows (down 0.59-0.96x) but tie or lose to use_quant_gemm's f16 tiles
    // at 96-256 (m5 1.00-1.19x at 96/160/192/256, m1 1.05x at 256) and to cuBLAS past 256 (m1
    // 1.07x at 320). Q4_K q/o's 9 % loss at 65-128 rows is accepted. Re-measure (quant_gemm
    // --bench-i8 and --bench) when a tile or use_quant_gemm's window changes.
    (tiles > 8 && !(w.dtype == DType::Q6K && rows >= 96)).then_some(I8::Plain)
}

/// The direct quantized GEMM's routes by shape class: output tiles (64 columns) against
/// `compute_units`, and k. The row windows are sm_120 measurements (70 SMs, cold weights,
/// `quant_gemm --bench`, 11 Oct 2026; m5 shapes and m1's Q6_K down 15 Oct 2026); outside them the
/// existing mmvq or dequant + gemm path won or was not measured.
fn use_quant_gemm(w: W, rows: i64, compute_units: u32) -> bool {
    let [n, k] = w.shape;
    let (tiles, units) = ((n + 63) / 64, i64::from(compute_units));
    if units == 0 {
        return false;
    }
    if tiles >= 3 * units {
        // Wide (FFN up, vocabulary heads): at least three waves of tiles win from 16 rows. Q6_K and
        // Q8_0 won only at vocabulary width (Q6_K lost at 16 rows on a 3.2-wave FFN up).
        return match w.dtype {
            DType::Q4K => (16..=1024).contains(&rows),
            DType::Q6K => tiles >= 8 * units && (16..=512).contains(&rows),
            DType::Q8_0 => tiles >= 8 * units && rows == 16,
            _ => false,
        };
    }
    // Narrow (attention output, FFN down): half a wave or more with k ≥ 4096 wins at 16 rows by
    // split-K and on M32 tiles; deep k (FFN down) also on M64 tiles and decoded panels.
    if 2 * tiles < units || tiles >= units || k < 4096 {
        return false;
    }
    let deep = k >= 14336;
    match w.dtype {
        DType::Q4K => {
            rows == 16
                || (24..=32).contains(&rows)
                || deep && ((33..=64).contains(&rows) || (96..=256).contains(&rows))
        }
        DType::Q6K => rows == 16 || rows == 32 || deep && (96..=256).contains(&rows),
        _ => false,
    }
}

/// A projection shape and row count where a gated route changes what runs
/// ([`Transformer::route_cases`]).
#[derive(Clone, Debug)]
pub struct RouteCase {
    pub route: Route,
    pub rows: i64,
    /// The weight the case ends on, which [`Transformer::route_reference`] checks.
    w: W,
    /// The model's other weights of its shape, run first.
    ws: Vec<W>,
}

impl RouteCase {
    /// `[n, k]`: output width, then input width.
    pub fn shape(&self) -> [i64; 2] {
        self.w.shape
    }

    pub fn dtype(&self) -> DType {
        self.w.dtype
    }
}

/// Where a weight sits in the weights buffer, or with `side` in [`Transformer::side`].
#[derive(Clone, Copy, Debug)]
struct W {
    off: u64,
    dtype: DType,
    shape: [i64; 2],
    ndim: usize,
    side: bool,
}

impl W {
    fn t<'a>(&self, b: &'a Buffer) -> Result<Tensor<'a>, KError> {
        Tensor::new(b, self.off, self.dtype, &self.shape[..self.ndim])
    }
}

struct Layer {
    attn_norm: W,
    mixer: Mixer,
    ffn_norm: W,
    ffn: Ffn,
}

impl Layer {
    /// The matrices `linear` runs.
    fn linears(&self) -> Vec<W> {
        let mut w = match &self.mixer {
            Mixer::Attn(a) => vec![a.wq, a.wk, a.wv, a.wo],
            Mixer::Gdn(g) => vec![g.qkv, g.z, g.alpha, g.beta, g.out],
        };
        match self.ffn {
            Ffn::Dense { gate, up, down } => w.extend([gate, up, down]),
            Ffn::Moe {
                shared: Some(sh), ..
            } => w.extend([sh.gate, sh.up, sh.down]),
            Ffn::Moe { .. } => {}
        }
        w
    }
}

/// What mixes positions in a layer.
enum Mixer {
    Attn(Attn),
    /// Qwen 3.5's Gated DeltaNet.
    Gdn(qwen35::Gdn),
}

/// An attention layer. `gated`: Qwen 3.5's, whose `wq` holds a [q, gate] pair per head.
struct Attn {
    wq: W,
    wk: W,
    wv: W,
    bq: Option<W>,
    bk: Option<W>,
    bv: Option<W>,
    q_norm: Option<W>,
    k_norm: Option<W>,
    wo: W,
    gated: bool,
    /// The layer in the KV pool: its index among the attention layers.
    kv_layer: u32,
}

/// A layer's feed-forward block.
// Copy and one per layer: the MoE variant's size costs nothing a box would save.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy)]
enum Ffn {
    Dense {
        gate: W,
        up: W,
        down: W,
    },
    /// Routed experts: `router` (f16 `[experts, d]`, in [`Transformer::side`]) picks
    /// [`MoeCfg::used`] of the stacked `[experts * N, k]` matrices for each row. `host`: the
    /// matrices are in [`HostExperts::weights`], else in the weights buffer. `shared`: qwen35moe's
    /// shared expert.
    Moe {
        router: W,
        /// The router as the GGUF stores it (f32 or f16), in the weights buffer: the trainer's,
        /// so qwen3moe's only ([`side_only`]).
        raw: Option<W>,
        gate: W,
        up: W,
        down: W,
        host: bool,
        shared: Option<qwen35::Shared>,
        /// A mixed layer's cold experts: `gate`, `up` and `down` then hold its hot ones.
        cold: Option<Cold>,
    },
}

/// A mixed layer's cold experts ([`crate::q2k::repack_cold`]): their gate, up and down slot
/// tensors, beside the layer's hot ones in the same buffer; [`Transformer::cold_tables`] holds the
/// layer's expert table.
#[derive(Clone, Copy)]
struct Cold {
    gate: W,
    up: W,
    down: W,
}

/// Activations for up to `max_rows` rows: `x` is the f32 residual stream, the rest f16.
struct Scratch {
    x: Buffer,
    h: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    /// A layer's output, f16; sized f32 because the f32 logits rows are gathered into it at the end.
    t: Buffer,
    /// f16 `[rows, ffn]`, or `[rows, used, expert ffn]` in an MoE model.
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    /// An MoE model's router logits (f32 `[rows, experts]`), chosen ids (i32 `[rows, used]`),
    /// their weights (f32 `[rows, used]`) and, at a host layer, `h` (f16 `[rows, d]`), in that order.
    route: Option<Buffer>,
    /// f16 copy of one weight for the dequant + gemm path; None when `max_rows` fits mmvq.
    dq: Option<Buffer>,
}

/// Where a model's weights go. The default puts every weight on the device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Placement {
    /// MoE models: the layers, from the first, whose routed experts stay in host memory and run on
    /// the CPU device ([`CPU_COMPUTE`]), as llama.cpp's `--n-cpu-moe` does. Past the layer count it
    /// is every layer; dense models have no routed experts and ignore it.
    pub cpu_expert_layers: u32,
    /// Prefill rows from which a prefill-only step runs the host layers' experts on the device, from
    /// the file through the pool [`Model::alloc_stream_pool`] allocates; 0 never.
    pub stream_rows: u32,
}

/// [`Placement::stream_rows`] when `LARCH_PREFILL_STREAM_ROWS` does not set it: streaming beats the
/// CPU prefill step from about 64 rows.
// ponytail: measured on one RTX 5070 Ti over PCIe at 12 host layers (streamed vs CPU prefill step: 32
// rows 0.98x, 64 rows 1.53x, 512 rows 2.62x); re-derive on another card or when a sweep moves the crossover.
pub const PREFILL_STREAM_ROWS: u32 = 64;

/// Per-expert residency, the expert cache (plan v2 E4), in place of [`Placement`]'s whole layers.
#[derive(Debug)]
pub struct ExpertPlan {
    /// Per layer, the routed experts the device keeps, each once. A layer past the list, or one
    /// that lists every expert, keeps them all in the weights buffer; any other layer splits: its
    /// other experts stay in host memory and run on the CPU device.
    pub resident: Vec<Vec<u32>>,
    /// Dynamic residency: host memory keeps every expert of a split layer, and hot host experts
    /// move into cold device slots between steps (see [`Transformer::load_cached`]).
    pub dynamic: bool,
}

/// A split MoE layer: gate, up and down of its resident experts in slots of [`ExpertCache::dev`]
/// and of its host experts in slots of [`HostExperts::weights`], `[slots * N, k]` each as
/// `mmvq_id` takes them, and which slot holds which expert.
struct Split {
    dev: [W; 3],
    host: [W; 3],
    /// Expert to device slot, -1 for none; dynamic residency moves experts in and out.
    slot: Vec<i32>,
    /// Expert to host slot, -1 for none (a resident expert of a static split).
    hslot: Vec<i32>,
    /// One expert's bytes in each of gate, up and down.
    slice: [u64; 3],
}

/// The expert cache's state: the split layers, routing counts (the histogram), and the uploads of
/// dynamic residency. A whole-layer load that records has one with no splits.
struct ExpertCache {
    /// Every split layer's device slots; `None` while only recording.
    dev: Option<Buffer>,
    /// Per layer.
    splits: Vec<Option<Split>>,
    /// The routing counts so far; where [`Transformer::record_routes`] saves them, once it is on.
    hist: Histogram,
    path: Option<PathBuf>,
    /// Dynamic residency: the stream that uploads experts beside the compute stream, and the
    /// uploads it has queued, (layer, expert, slot), which [`Transformer::pass`] maps once done.
    copy: Option<Stream>,
    pending: Vec<(usize, u32, u32)>,
    /// The passes so far.
    passes: u32,
    /// The save in flight ([`ExpertCache::save_behind`]).
    writer: Option<JoinHandle<()>>,
    /// The rows of the step whose resident routes the log holds, not yet counted.
    logged: Option<i64>,
    /// The split layers' (row, slot) pairs whose expert ran on the device, and on the host.
    hits: (u64, u64),
    /// Split layers' steps, replayed or eager, that routed no row to the host.
    no_host: u64,
    /// Split layers' steps, replayed or eager: the denominator of `no_host`.
    split_steps: u64,
    /// A recording whole-layer load's route log is still to allocate, at its first step: after
    /// the engine's fit, so its bytes never touch that plan.
    log_later: bool,
    /// A recording whole-layer load's host counts of the logged step, `[layers * experts]`: they
    /// reach `hist` with that step's resident ones or not at all, so neither kind of layer leads.
    staged: Vec<u64>,
}

impl ExpertCache {
    /// Dynamic residency at a split layer once the compute stream is idle: each of this step's
    /// host experts (`ids`) over twice as hot as the layer's coldest resident one takes its slot,
    /// so near-even counts do not swap back and forth. The cold
    /// one runs on the CPU from now on (host memory keeps every expert of a dynamic split), the
    /// hot one's upload is queued on the copy stream, and [`Transformer::pass`] maps it once done.
    fn upload_hot(&mut self, x: &HostExperts, layer: usize, ids: &[usize]) -> Result<(), KError> {
        let Some(cs) = &self.copy else {
            return Ok(());
        };
        let sp = self.splits[layer].as_mut().ok_or(KError::InvalidArg)?;
        let e = sp.slot.len();
        let counts = &self.hist.counts[layer * e..][..e];
        // ponytail: hotness is the cumulative count, the loaded histogram included; decay it if a
        // workload shift leaves the hit rate low for long.
        let mut hot: Vec<usize> = ids.iter().copied().filter(|&i| sp.slot[i] < 0).collect();
        hot.sort_unstable_by_key(|&i| (std::cmp::Reverse(counts[i]), i));
        hot.dedup();
        for i in hot {
            let queued = self
                .pending
                .iter()
                .any(|p| (p.0, p.1 as usize) == (layer, i));
            let cold = (0..e)
                .filter(|&j| sp.slot[j] >= 0)
                .min_by_key(|&j| (counts[j], j));
            let Some(cold) = cold.filter(|&j| counts[i] / 2 > counts[j]) else {
                break;
            };
            if self.pending.len() >= MAX_PENDING || queued {
                continue;
            }
            let slot = std::mem::replace(&mut sp.slot[cold], -1) as u64;
            for w in 0..3 {
                let mut b = vec![0u8; sp.slice[w] as usize];
                x.s.download(
                    &x.weights,
                    sp.host[w].off + sp.hslot[i] as u64 * sp.slice[w],
                    &mut b,
                )?;
                let dev = self.dev.as_ref().ok_or(KError::InvalidArg)?;
                cs.upload(dev, sp.dev[w].off + slot * sp.slice[w], &b)?;
            }
            self.pending.push((layer, i as u32, slot as u32));
        }
        Ok(())
    }

    /// Writes the hit rate to the fit's diagnostics and the counts to the recording's file, on a
    /// thread of its own: the write syncs, and a step must not wait for a placement hint. One save
    /// at a time, since they share a temporary file; a later one has any counts this one skips.
    fn save_behind(&mut self) {
        if self.dev.is_some() && std::env::var("LARCH_FIT_DIAGNOSTICS").is_ok_and(|v| v == "1") {
            eprintln!(
                "larch_expert_cache_hits,{},{},{},{}",
                self.hits.0, self.hits.1, self.no_host, self.split_steps
            );
        }
        if self.writer.as_ref().is_some_and(|w| !w.is_finished()) {
            return;
        }
        let Some(path) = self.path.clone() else {
            return;
        };
        let hist = self.hist.clone();
        let save = move || {
            if let Err(err) = hist.write(&path) {
                eprintln!("larch_expert_histogram_save,{err}");
            }
        };
        match std::thread::Builder::new().spawn(save) {
            Ok(w) => self.writer = Some(w),
            Err(err) => eprintln!("larch_expert_histogram_save,{err}"),
        }
    }

    /// Waits for the save in flight, if any.
    fn finish_save(&mut self) {
        if let Some(Err(_)) = self.writer.take().map(JoinHandle::join) {
            eprintln!("larch_expert_histogram_save,panicked");
        }
    }
}

/// (row, slot) pairs a split layer's host part runs at a time: CPU_COMPUTE's mmvq_id groups a call's
/// pairs from 4 per expert, which rounds otherwise, and a one-slot row is one pair, so 3 always run
/// per pair, as the row alone does.
// ponytail: a long prefill's misses run per pair, a pool dispatch per slice, without the grouped path's
// tile reuse; make the two paths' rounding agree and drop the slices if a split prefill's host part shows.
const HOST_ROWS: i64 = 3;
/// Why a model with an expert cache refuses expert adapters and suffix training: on any such
/// model, so a restart's new plan never takes them away from one that had them.
// ponytail: refused outright; remap adapter ids per split and keep the trained suffix resident when
// learning has to run on a model past the room with the cache on.
const NO_LEARNING: &str =
    "larch_expert_cache_no_learning,unsetting LARCH_EXPERT_CACHE keeps whole layers";
/// Recorded passes between two saves of the histogram.
const SAVE_EVERY: u32 = 256;
/// The same for a recording whole-layer load: a serve is killed, never dropped, and its replayed
/// decode steps make no passes, so at 256 a cold serve would write nothing and stay cold. A save
/// is off-thread and about 41 KB.
const SAVE_EVERY_COLD: u32 = 8;
/// Dynamic residency's uploads in flight; one expert is a few MB.
const MAX_PENDING: usize = 8;

/// The routed experts that [`Placement`] keeps in host memory, on the CPU device, with that
/// device's activations: `h` (f16 `[rows, d]`), `route` (ids, then weights, as in
/// [`Scratch::route`] without the logits), gate, up and act as in [`Scratch`], and `out` (f16
/// `[rows, d]`); `staging`, the host copy of the route and `h` (then `out`) for the largest step,
/// allocated at load so that a step allocates nothing.
struct HostExperts {
    s: Stream,
    staging: Mutex<Vec<u8>>,
    weights: Buffer,
    h: Buffer,
    route: Buffer,
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    out: Buffer,
}

/// What the copy thread's own stream takes on the device: CUDA's cuBLAS workspace (32 MiB from
/// Hopper on, 4 MiB before, kernels/src/cuda/backend.cu); the larger figure on every card.
const COPY_STREAM_BYTES: u64 = 32 << 20;

/// The streaming pool's slots: one unless LARCH_PREFILL_STREAM_SLOTS asks for two.
fn stream_slots() -> usize {
    match std::env::var("LARCH_PREFILL_STREAM_SLOTS").as_deref() {
        Ok("2") => 2,
        _ => 1,
    }
}

/// One host layer's gate, up and down experts as (file offset, bytes, the hot tensor's offset in that
/// span, a mixed layer's cold tensor's): [`expert_span`]'s, one slot each.
type StreamLayer = [(u64, u64, u64, Option<u64>); 3];

/// Where layer `i`'s routed experts of projection `p` lie in `g`'s file as one stream pool slot takes
/// them: (file offset, bytes, the hot tensor's offset in the span, the cold one's), a mixed layer's
/// two tensors in one span (repack writes them side by side).
fn expert_span(g: &Gguf, i: u32, p: &str) -> Option<(u64, u64, u64, Option<u64>)> {
    let hot = g.tensor(&format!("blk.{i}.ffn_{p}_exps.weight"))?;
    let cold = g.tensor(&crate::q2k::cold_name(i, p));
    let end = |t: &gguf::TensorInfo| t.offset + t.bytes;
    let (lo, hi) = cold.map_or((hot.offset, end(hot)), |c| {
        (hot.offset.min(c.offset), end(hot).max(end(c)))
    });
    let at = |t: &gguf::TensorInfo| t.offset - lo;
    Some((g.data_offset + lo, hi - lo, at(hot), cold.map(at)))
}

/// Where file bytes `start..start + bytes` lie in the host experts' weights: the host tensors
/// `held` (file start, bytes, offset there, in file order) that cover them back to back in both.
/// `None` where they do not, so the copy reads the file map.
fn held_at(held: &[(u64, u64, u64)], start: u64, bytes: u64) -> Option<u64> {
    let i = held.iter().position(|h| h.0 == start)?;
    let mut end = held[i];
    for h in &held[i + 1..] {
        if end.0 + end.1 >= start + bytes {
            break;
        }
        (h.0 == end.0 + end.1 && h.2 == end.2 + end.1).then_some(())?;
        end = *h;
    }
    (end.0 + end.1 >= start + bytes).then_some(held[i].2)
}

/// A slot on its way to the copy thread, to be filled with one expert tensor: from the file map's
/// bytes `piece`, or from the host experts' weights at `held` where they hold it.
struct CopyJob {
    slot: Buffer,
    piece: (u64, u64),
    held: Option<u64>,
}

/// A slot back from the copy thread; `ns` is the copy's wall time when profiling, else 0.
struct Copied {
    slot: Buffer,
    ns: u64,
    done: Result<(), KError>,
}

/// What [`Placement::stream_rows`] found at load: where each host layer's experts lie and a
/// slot's size, until the fit admits a pool and [`Model::alloc_stream_pool`] builds it. `src`: the
/// host experts' weights shared as a copy source, with each tensor's offset there ([`held_at`]).
struct StreamPlan {
    layers: Vec<StreamLayer>,
    slot_bytes: u64,
    slots: usize,
    map: Mmap,
    src: Option<(HostCopy, Vec<[Option<u64>; 3]>)>,
}

/// One or two tensor-sized slots passed between copy and compute threads in gate, up, down order.
/// The compute stream syncs before a slot is reused, so the next copy cannot overwrite a reader.
struct StreamPool {
    layers: Vec<StreamLayer>,
    /// Each tensor's offset in the copy thread's host source, where it has one.
    held: Vec<[Option<u64>; 3]>,
    idle: Vec<Buffer>,
    slots: usize,
    /// Slots at the copy thread. Jobs go and come back in layer order, and `begin` drains any a
    /// failed step left, so the next `take` is always the next tensor's.
    queued: usize,
    /// A failed step may have left readers of a slot queued on the compute stream.
    dirty: bool,
    /// A copy ran out of memory: the steps take the CPU path from then on.
    disabled: bool,
    /// Tests: after this many `take`s, the next copy fails with this error.
    #[cfg(test)]
    fail: Option<(u32, KError)>,
    steps: u64,
    tx: Option<Sender<CopyJob>>,
    rx: Receiver<Copied>,
    worker: Option<JoinHandle<()>>,
    profile: bool,
    /// This step's rows, copy ns and compute-thread wait ns, for the profile line.
    rows: i64,
    copy_ns: u64,
    wait_ns: u64,
}

impl StreamPool {
    fn spawn(
        s: &Stream,
        plan: StreamPlan,
        idle: Vec<Buffer>,
        profile: bool,
    ) -> Result<StreamPool, KError> {
        let StreamPlan {
            layers,
            map,
            slot_bytes: _,
            slots: _,
            src,
        } = plan;
        let (src, held) = src.map_or((None, Vec::new()), |(c, at)| (Some(c), at));
        let slots = idle.len();
        let copy = Stream::new(s.device())?;
        let (tx, jobs) = channel::<CopyJob>();
        let (done, rx) = channel();
        let worker = std::thread::Builder::new()
            .name("larch-prefill-copy".into())
            .spawn(move || {
                for j in jobs {
                    let t = profile.then(Instant::now);
                    let (file, bytes) = j.piece;
                    // ponytail: a GGUF truncated or rewritten while serving raises SIGBUS here.
                    let copied = match (&src, j.held) {
                        (Some(src), Some(at)) => copy.upload_from(&j.slot, 0, src, at..at + bytes),
                        _ => copy.upload_map(&j.slot, 0, &map, file..file + bytes),
                    }
                    .and_then(|()| copy.sync());
                    let ns = t.map_or(0, |t| t.elapsed().as_nanos() as u64);
                    let back = Copied {
                        slot: j.slot,
                        ns,
                        done: copied,
                    };
                    if done.send(back).is_err() {
                        return;
                    }
                }
            })
            // A thread the OS will not give is host resources run out: the same as no room.
            .map_err(|_| KError::Oom)?;
        Ok(StreamPool {
            layers,
            held,
            idle,
            slots,
            queued: 0,
            dirty: false,
            disabled: false,
            #[cfg(test)]
            fail: None,
            steps: 0,
            tx: Some(tx),
            rx,
            worker: Some(worker),
            profile,
            rows: 0,
            copy_ns: 0,
            wait_ns: 0,
        })
    }

    fn send(&mut self, tensor: usize) -> Result<(), KError> {
        let slot = self.idle.pop().ok_or(KError::InvalidArg)?;
        let job = CopyJob {
            slot,
            piece: {
                let p = self.layers[tensor / 3][tensor % 3];
                (p.0, p.1)
            },
            held: self.held.get(tensor / 3).and_then(|h| h[tensor % 3]),
        };
        let tx = self.tx.as_ref().ok_or(KError::Launch)?;
        tx.send(job).map_err(|_| KError::Launch)?;
        self.queued += 1;
        Ok(())
    }

    fn recv(&mut self) -> Result<Copied, KError> {
        let c = self.rx.recv().map_err(|_| KError::Launch)?;
        self.queued -= 1;
        self.copy_ns += c.ns;
        Ok(c)
    }

    /// Starts a streamed step of `rows` rows: fills the available tensor slots.
    fn begin(&mut self, s: &Stream, rows: i64) -> Result<(), KError> {
        // Only a failed step leaves queued copies or a slot with readers still queued.
        if self.dirty {
            s.sync()?;
            self.dirty = false;
        }
        while self.queued > 0 {
            let c = self.recv()?;
            self.idle.push(c.slot);
        }
        (self.rows, self.copy_ns, self.wait_ns) = (rows, 0, 0);
        self.steps += 1;
        (0..(self.layers.len() * 3).min(self.slots)).try_for_each(|i| self.send(i))
    }

    /// The next tensor's slot once its copy is done; `None` when the pool is off: a copy out
    /// of memory (the copy stream's lazy staging is pinned host memory) turns it off for good, and
    /// no device work is queued for the layer yet, so the caller runs it on the CPU path.
    fn take(&mut self) -> Result<Option<Buffer>, KError> {
        if self.disabled {
            return Ok(None);
        }
        let t = self.profile.then(Instant::now);
        let c = self.recv()?;
        #[cfg(test)]
        let c = self.injected(c);
        self.wait_ns += t.map_or(0, |t| t.elapsed().as_nanos() as u64);
        match c.done {
            Ok(()) => Ok(Some(c.slot)),
            Err(KError::Oom) => {
                self.idle.push(c.slot);
                self.disabled = true;
                if self.profile {
                    eprintln!("larch_stream_pool_disabled,oom");
                }
                Ok(None)
            }
            Err(e) => {
                self.idle.push(c.slot);
                self.dirty = true;
                Err(e)
            }
        }
    }

    #[cfg(test)]
    fn injected(&mut self, mut c: Copied) -> Copied {
        match self.fail {
            Some((0, e)) => (self.fail, c.done) = (None, Err(e)),
            Some((n, e)) => self.fail = Some((n - 1, e)),
            None => {}
        }
        c
    }

    /// Back from one tensor's compute, once its readers have finished.
    fn finish(
        &mut self,
        tensor: usize,
        slot: Buffer,
        ran: Result<(), KError>,
    ) -> Result<(), KError> {
        self.idle.push(slot);
        if ran.is_err() {
            self.dirty = true;
            return ran;
        }
        if tensor + self.slots < self.layers.len() * 3 {
            self.send(tensor + self.slots)?;
        }
        if self.profile && tensor + 1 == self.layers.len() * 3 {
            eprintln!(
                "larch_stream_step,{},{},{},{}",
                self.rows,
                self.layers.len(),
                self.copy_ns as f64 / 1e6,
                self.wait_ns as f64 / 1e6
            );
        }
        Ok(())
    }
}

impl Drop for StreamPool {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(worker) = self.worker.take() {
            // A panicked copy thread has nothing left to clean up.
            let _ = worker.join();
        }
    }
}

/// Whether a step streams its host experts: at least `from` prefill rows (beside any decode rows,
/// which keep the CPU path: [`Transformer::moe_ffn`]), never a capture, neither a check step nor one
/// where any layer has adapters (a host layer's run on the CPU device; a device layer's would not
/// stop it, but the check is per model), and only with a pool.
fn streams_step(
    from: u32,
    pool: bool,
    prefill: u32,
    check: bool,
    capturing: bool,
    adapters: bool,
) -> bool {
    from > 0 && pool && prefill >= from && !check && !capturing && !adapters
}

/// The CPU device's threads beside a GPU: every logical CPU but two, which the engine's thread (it
/// waits on the GPU between host layers) and the GPU driver's keep. All of them measured a host-expert
/// layer at one row 415 us against 334 us at 6 of 8 (the hybrid MoE target, 9800X3D, WSL2: 8 logical
/// CPUs), +5 % at W4@1, the same at 4 rows.
// ponytail: two, measured on one 8-CPU host; re-derive on a host with more cores, or where the
// engine's own work per step grows.
fn beside_gpu_threads(logical: usize) -> u32 {
    logical.saturating_sub(2).max(1) as u32
}

/// The token embedding in host memory, as llama.cpp keeps it beside a GPU's MoE: the table `w`
/// (`[vocab, d]` as the GGUF stores it) on the CPU device, which gathers a step's rows into `out`
/// (f32 `[rows, d]`) by `ids` (i32 `[rows]`), staged in `x` for one upload into the scratch x.
/// `primed`: the rows [`Model::embed_tokens`] put in x for the next pass, which it then keeps.
// ponytail: the table is copied into host memory (218 MB on the hybrid MoE target); gather from the file map
// when host RAM is what runs short.
struct HostEmbed {
    s: Stream,
    w: W,
    table: Buffer,
    ids: Buffer,
    out: Buffer,
    x: RefCell<Vec<f32>>,
    primed: Cell<Option<i64>>,
}

/// The float matrices of `g`'s layers a load reads only as f16 side copies, which the weights
/// buffer leaves out: the MoE routers but qwen3moe's (its trainer reads them, `Ffn::Moe::raw`),
/// the shared experts' gates, and Gated DeltaNet's alpha and beta where they are float.
fn side_only<'a>(g: &'a Gguf, c: &ModelConfig) -> impl Iterator<Item = &'a gguf::TensorInfo> {
    let (trainer, layers) = (c.arch == Arch::Qwen3Moe, c.n_layers);
    g.tensors.iter().filter(move |t| {
        let Some((i, rest)) = t.name.strip_prefix("blk.").and_then(|r| r.split_once('.')) else {
            return false;
        };
        let float = matches!(gguf::dtype(t.ggml_type), Ok(DType::F32 | DType::F16));
        i.parse::<u32>().is_ok_and(|i| i < layers)
            && match rest {
                "ffn_gate_inp.weight" => !trainer,
                "ffn_gate_inp_shexp.weight" => true,
                "ssm_alpha.weight" | "ssm_beta.weight" => float,
                _ => false,
            }
    })
}

/// The rank an adapted expert's Q8_0 tensors hold: one Q8_0 block, so a B row is one block.
const ADAPTER_RANK: usize = 32;

/// One layer's expert adapters as Q8_0 expert tensors for [`ops::mmvq_id`], on the device that runs
/// the layer's experts: A `[experts * 32, expert ffn]` and B `[experts * d_model, 32]`, the rank
/// padded to 32 and the experts without an adapter zero. A row's chosen experts then run A and the
/// route-weighted B on the device, with no host round trip.
struct Adapters {
    a: Buffer,
    b: Buffer,
}

/// `x` (whole blocks of 32) into `out` as Q8_0, as ggml's reference rounds: an f16 scale
/// max|x| / 127, then 32 i8.
fn q8_0(x: &[f32], out: &mut [u8]) {
    for (b, o) in x.chunks_exact(32).zip(out.chunks_exact_mut(34)) {
        let d = b.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        o[..2].copy_from_slice(&f16_bits(d).to_le_bytes());
        for (q, v) in o[2..].iter_mut().zip(b) {
            *q = (v * id).round() as i8 as u8;
        }
    }
}

/// A GGUF model loaded onto one device, with its activation scratch.
pub struct Transformer {
    cfg: ModelConfig,
    weights: Buffer,
    /// The file from byte `.1` on, when one wrap of it passes maxBufferLength; `weights` then ends
    /// where this starts (see [`wrap_halves`]).
    hi: Option<(Buffer, u64)>,
    /// Float matrices as f16, which gemm takes (GGUFs keep them f32): the MoE routers, the shared
    /// experts' gates and Gated DeltaNet's alpha and beta where they are not quantized.
    // ponytail: routers rounded to f16 (gemm's weight dtype); add an f32 router GEMV if qwen3moe
    // parity's top-1 misses at near-tied experts.
    side: Option<Buffer>,
    moe: Option<MoeCfg>,
    /// Qwen 3.5's layer kinds, their scratch and the sequences' recurrent state
    /// ([`Transformer::alloc_state`]).
    hybrid: Option<qwen35::Hybrid>,
    hy: Option<qwen35::Scratch>,
    states: Option<qwen35::States>,
    /// Per layer, the router-logit deltas (f32 `[experts]`) the router gemm adds as its bias; empty
    /// for the base model.
    router_deltas: Vec<Option<Buffer>>,
    /// Per layer, the expert adapters ([`RouterDeltas::rank`] != 0) on the device that runs its
    /// experts: `moe_ffn` adds them to the routed output; empty for the base model.
    adapters: Vec<Option<Adapters>>,
    host: Option<HostExperts>,
    stream_plan: Option<StreamPlan>,
    stream: Option<RefCell<StreamPool>>,
    stream_rows: u32,
    /// The running step streams its host experts ([`streams_step`]).
    streaming: bool,
    /// The last pass's decode rows, which lead its rows: a streamed pass keeps them on the CPU path.
    stream_decode: i64,
    /// Per trunk layer, a mixed layer's expert table (i32 `[experts]`, [`ops::mmvq_id_mixed`]'s) on
    /// the device that runs its experts; `None` for any other layer.
    cold_tables: Vec<Option<Buffer>>,
    /// Per host layer, a mixed one's expert table on the device as well, for the stream pool.
    stream_tables: Vec<Option<Buffer>>,
    /// The token embedding when it stays in host memory; `tok_embd` is then `None`.
    embd: Option<HostEmbed>,
    /// Per-expert residency ([`ExpertPlan`]); `None` under [`Placement`].
    cache: Option<RefCell<ExpertCache>>,
    /// While recording, the fully resident MoE layers' routed ids of a step (i32 `[layers, rows,
    /// used]`), read back once after it.
    route_log: Option<Buffer>,
    /// The token embedding in the weights buffer; `None` when it stays in host memory, so no
    /// device read can take another tensor's bytes for it ([`Transformer::embed_table`]).
    tok_embd: Option<W>,
    out_norm: W,
    output: W,
    rope_freqs: Option<W>,
    layers: Vec<Layer>,
    /// The MTP block, from [`Transformer::load_mtp`].
    mtp: Option<qwen35_mtp::Nextn>,
    sc: Scratch,
    max_rows: i64,
    mmvq_max_rows: i64,
    backend: Backend,
    /// The gated routes this card's profile turns on ([`card::routes`]).
    card: Vec<Route>,
    quant_i8: bool,
    compute_units: u32,
    arch_code: u32,
    /// The running step is a check step ([`StepBatch::check`]) on exact kernels: each row as a one-row
    /// decode. A cell so the MTP drafter, which shares the activations, resets it behind `&self`.
    check: std::cell::Cell<bool>,
    /// [`crate::exact_checks`] for this device at load: a check step takes the exact kernels.
    exact_checks: bool,
    /// Rows a check step's mmvq_id takes at a time on this device ([`ID_ROWS`], [`METAL_ID_ROWS`]).
    id_rows: i64,
    /// A check step's own kernels ran once on this backend ([`Transformer::probe_check`]).
    checks_ran: bool,
    /// `LARCH_FUSED` at load ([`larch_kernels::Routes::fused`]): which fused entries a step takes
    /// ([`Transformer::fusing`]).
    fused: Option<bool>,
    /// The fused entries on by default on this device ([`fused_defaults`]), where `fused` is `None`.
    fused_on: Fused,
    /// [`Transformer::refuse_fused`]: every fused entry answers UNSUPPORTED without running.
    refuse_fused: bool,
}

/// Apple silicon's 16 KiB pages; a multiple of every smaller page, which wrap_host checks.
const PAGE: u64 = 1 << 14;

/// Where `len` bytes holding the tensors `spans` (start, end) split into two buffers where one
/// passes the device's buffer limit: the lower ends at the page after the last tensor that starts
/// before the upper's first page, and the upper starts at the page of the first tensor whose end
/// passes the middle, so every tensor lies whole in one of them. `None` when no tensor passes it.
fn halves(mut spans: Vec<(u64, u64)>, len: u64) -> Option<(u64, u64)> {
    spans.sort_unstable();
    let upper = spans.iter().find(|t| t.1 > len / 2)?.0 / PAGE * PAGE;
    let lower = spans
        .iter()
        .filter(|t| t.0 < upper)
        .map(|t| t.1)
        .max()
        .map_or(PAGE, |end| end.div_ceil(PAGE) * PAGE)
        .min(len);
    Some((lower, upper))
}

/// The file map's first `len` bytes, holding the tensors `spans`, as two wraps split at [`halves`]:
/// a tensor at file offset `o` past the upper's start `u` sits at `o - u` in it. `None` when either wrap still fails with Oom, or the file has
/// no such tensor: the upload decides.
fn wrap_halves(
    dev: &Device,
    map: &Mmap,
    spans: Vec<(u64, u64)>,
    len: u64,
) -> Result<Option<(Buffer, Buffer, u64)>, LoadError> {
    let Some((lower, upper)) = halves(spans, len) else {
        return Ok(None);
    };
    let wrap = |r: std::ops::Range<u64>| match Buffer::wrap_host(dev, map, r) {
        Ok(b) => Ok(Some(b)),
        Err(KError::Oom | KError::Unsupported) => Ok(None),
        Err(e) => Err(backend(e)),
    };
    Ok(match (wrap(0..lower)?, wrap(upper..len)?) {
        (Some(lo), Some(hi)) => Some((lo, hi, upper)),
        _ => None,
    })
}

/// The file (start, end) of every tensor in `g` but those of layers past the first `layers`.
fn used_spans(g: &Gguf, layers: u32) -> Vec<(u64, u64)> {
    let past = |name: &str| {
        let i = name.strip_prefix("blk.").and_then(|r| r.split_once('.'));
        i.and_then(|(i, _)| i.parse::<u32>().ok())
            .is_some_and(|i| i >= layers)
    };
    g.tensors
        .iter()
        .filter(|t| !past(&t.name))
        .map(|t| (g.data_offset + t.offset, g.data_offset + t.offset + t.bytes))
        .collect()
}

/// A route's chosen ids (i32 `[rows, k]`) at byte `at` of `b`, then their weights (f32 `[rows, k]`).
fn route_of(b: &Buffer, at: u64, rows: i64, k: i64) -> Result<(Tensor<'_>, Tensor<'_>), KError> {
    let w = at + DType::I32.bytes((rows * k) as u64);
    Ok((
        Tensor::new(b, at, DType::I32, &[rows, k])?,
        Tensor::new(b, w, DType::F32, &[rows, k])?,
    ))
}

/// The [`larch_kernels::Stream::cut`] tag of a host-expert section: all its replay needs.
fn host_tag(layer: usize, rows: i64, check: bool) -> u64 {
    u64::from(check) << 63 | (layer as u64) << 32 | rows as u64
}

/// A [`host_tag`]'s layer, rows and check flag.
fn untag(tag: u64) -> (usize, i64, bool) {
    let layer = (tag >> 32) & 0x7fff_ffff;
    (layer as usize, i64::from(tag as u32), tag >> 63 != 0)
}

/// A routed projection's experts: one stacked tensor (`[experts * N, k]`), or a mixed layer's hot and
/// cold slot tensors and its expert table ([`ops::mmvq_id_mixed`]).
#[derive(Clone, Copy)]
enum Experts<'a> {
    One(Tensor<'a>),
    Mixed {
        hot: Tensor<'a>,
        cold: Tensor<'a>,
        table: Tensor<'a>,
    },
}

impl<'a> Experts<'a> {
    /// A layer's gate, up and down experts `w` as tensors `t` places them, a mixed layer's beside
    /// its `cold` ones and the expert `table` on the same device.
    fn of(
        w: [W; 3],
        cold: Option<Cold>,
        table: Option<&'a Buffer>,
        t: impl Fn(W) -> Result<Tensor<'a>, KError>,
    ) -> Result<[Experts<'a>; 3], KError> {
        let one = |i: usize| -> Result<Experts<'a>, KError> {
            let Some(c) = cold else {
                return Ok(Experts::One(t(w[i])?));
            };
            let table = table.ok_or(KError::InvalidArg)?;
            Ok(Experts::Mixed {
                hot: t(w[i])?,
                cold: t([c.gate, c.up, c.down][i])?,
                table: Tensor::new(table, 0, DType::I32, &[table.bytes() as i64 / 4])?,
            })
        };
        Ok([one(0)?, one(1)?, one(2)?])
    }

    /// `ops::mmvq_id` (no bias), or `ops::mmvq_id_mixed` of a mixed layer.
    fn mmvq_id(
        &self,
        s: &Stream,
        x: &Tensor,
        ids: &Tensor,
        weights: Option<&Tensor>,
        out: &Tensor,
    ) -> Result<(), KError> {
        match self {
            Experts::One(w) => ops::mmvq_id(s, w, x, ids, None, weights, out),
            Experts::Mixed { hot, cold, table } => {
                ops::mmvq_id_mixed(s, hot, cold, table, x, ids, weights, out)
            }
        }
    }
}

/// Each row's chosen experts (`ids`' columns) of `h` (gate, up, down in `wb`) into `out` on `s`:
/// gate and up per chosen expert into `sc`'s gate and up (`[rows, chosen, expert ffn]`), silu_mul
/// into its act, then down per chosen expert, weighted and summed. A check step's rows go `check`
/// at a time (0 outside a check).
#[allow(clippy::too_many_arguments)]
fn routed(
    s: &Stream,
    check: i64,
    [gate, up, down]: [Experts; 3],
    [gb, ub, ab]: [&Buffer; 3],
    m: MoeCfg,
    h: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    out: &Tensor,
) -> Result<(), KError> {
    let act = routed_act(s, check, [&gate, &up], [gb, ub, ab], m, h, ids)?;
    mmvq_id(s, check, &down, &act, ids, Some(weights), out)
}

/// The most rows [`host_routed`] runs as one fused gate and up: a decode step's, up to the hybrid's
/// most seats.
const FUSED_HOST_ROWS: i64 = 8;

/// [`routed`] on the CPU device of the host experts: outside a check step, one-tensor gate and up run
/// as its `fused_moe_gate_up`, the same bytes as their sequence in one pool job (the sequence where the
/// device refuses it).
#[allow(clippy::too_many_arguments)] // routed's
fn host_routed(
    s: &Stream,
    check: i64,
    [gate, up, down]: [Experts; 3],
    [gb, ub, ab]: [&Buffer; 3],
    m: MoeCfg,
    h: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    out: &Tensor,
) -> Result<(), KError> {
    let act = Tensor::new(
        ab,
        0,
        DType::F16,
        &[h.shape()[0], ids.shape()[1], i64::from(m.ffn)],
    )?;
    // Decode shapes: past a few rows the pairs group by expert, which the fused entry leaves to the
    // sequence through scratch of its own (routed_act reuses the host experts').
    let fused = match (&gate, &up, check) {
        (Experts::One(g), Experts::One(u), 0) if h.shape()[0] <= FUSED_HOST_ROWS => {
            ops::fused_moe_gate_up(s, g, u, h, ids, &act)
        }
        _ => Err(KError::Unsupported),
    };
    let act = match fused {
        Err(KError::Unsupported) => routed_act(s, check, [&gate, &up], [gb, ub, ab], m, h, ids)?,
        r => r.map(|()| act)?,
    };
    mmvq_id(s, check, &down, &act, ids, Some(weights), out)
}

/// [`routed`]'s gate and up per chosen expert into `gb` and `ub`, and their silu_mul into `ab`, returned
/// (`[rows, chosen, expert ffn]`).
fn routed_act<'a>(
    s: &Stream,
    check: i64,
    [gate, up]: [&Experts; 2],
    [gb, ub, ab]: [&'a Buffer; 3],
    m: MoeCfg,
    h: &Tensor,
    ids: &Tensor,
) -> Result<Tensor<'a>, KError> {
    let (rows, k, ef) = (h.shape()[0], ids.shape()[1], i64::from(m.ffn));
    let shape = [rows, k, ef];
    let g = Tensor::new(gb, 0, DType::F16, &shape)?;
    let u = Tensor::new(ub, 0, DType::F16, &shape)?;
    let act = Tensor::new(ab, 0, DType::F16, &shape)?;
    let flat = [rows * k, ef];
    mmvq_id(s, check, gate, h, ids, None, &g)?;
    mmvq_id(s, check, up, h, ids, None, &u)?;
    ops::silu_mul(
        s,
        &g.reshape(&flat)?,
        &u.reshape(&flat)?,
        &act.reshape(&flat)?,
    )?;
    Ok(act)
}

/// [`Experts::mmvq_id`] of `x`'s rows by `ids` (and `weights`) into `out`, a check step's `check`
/// at a time (0 outside a check).
fn mmvq_id(
    s: &Stream,
    check: i64,
    w: &Experts,
    x: &Tensor,
    ids: &Tensor,
    weights: Option<&Tensor>,
    out: &Tensor,
) -> Result<(), KError> {
    if check == 0 {
        return w.mmvq_id(s, x, ids, weights, out);
    }
    sliced(check, x, out, |at, x, o| {
        let n = x.shape()[0];
        let wt = weights.map(|t| t.rows(at, n)).transpose()?;
        w.mmvq_id(s, x, &ids.rows(at, n)?, wt.as_ref(), o)
    })
}

/// The layer's expert adapters on the routed rows: for each row's chosen experts, A on its slot of
/// `act` (`[rows, used, ffn]`) into `t`, then B weighted by the route into `y` (f16 `[rows, d]`,
/// returned), `sum_j w_j B_j A_j act_j`, which the caller adds to the residual beside the experts'
/// output (`W_down + B A` in all). `t` and `y` are the stream's gate and up scratch, free once
/// `routed` has run, which a check step's rows also leave `check` at a time.
#[allow(clippy::too_many_arguments)]
fn adapt<'a>(
    s: &Stream,
    check: i64,
    ad: &Adapters,
    [act, t, y]: [&'a Buffer; 3],
    m: MoeCfg,
    ids: &Tensor,
    weights: &Tensor,
    rows: i64,
    d: i64,
) -> Result<Tensor<'a>, KError> {
    // A graph would keep the version's buffers past the swap that frees them.
    if s.capturing() {
        return Err(KError::Unsupported);
    }
    let (e, k, ef) = (i64::from(m.experts), i64::from(m.used), i64::from(m.ffn));
    let r = ADAPTER_RANK as i64;
    let a = Tensor::new(&ad.a, 0, DType::Q8_0, &[e * r, ef])?;
    let b = Tensor::new(&ad.b, 0, DType::Q8_0, &[e * d, r])?;
    let x = Tensor::new(act, 0, DType::F16, &[rows, k, ef])?;
    let t = Tensor::new(t, 0, DType::F16, &[rows, k, r])?;
    let y = Tensor::new(y, 0, DType::F16, &[rows, d])?;
    mmvq_id(s, check, &Experts::One(a), &x, ids, None, &t)?;
    mmvq_id(s, check, &Experts::One(b), &t, ids, Some(weights), &y)?;
    Ok(y)
}

/// `[rows, heads, head_dim]` seen as `[rows, heads * head_dim]`.
pub(crate) fn flat<'a>(t: &Tensor<'a>) -> Result<Tensor<'a>, KError> {
    let s = t.shape();
    t.reshape(&[s[0], s[1] * s[2]])
}

pub(crate) fn backend(e: KError) -> LoadError {
    LoadError::Backend(e.code())
}

/// `dev` refusing `need` bytes: its `Oom` is its own measured limit, so the model does not fit.
/// Layers that keep attention K and V in the KV pool: every layer but a hybrid's recurrent ones, as
/// the loader numbers them.
pub(crate) fn kv_layers(g: &Gguf, c: &ModelConfig) -> Result<u32, LoadError> {
    let recurrent = qwen35::Hybrid::from_gguf(g, c)?.map_or(0, |h| h.gdn_layers());
    Ok(c.n_layers - recurrent)
}

pub(crate) fn fit_err(dev: &Device, need: u64, e: KError) -> LoadError {
    match e {
        KError::Oom => LoadError::DoesNotFit {
            need,
            have: dev.props().map_or(0, |p| p.recommended_working_set),
        },
        e => backend(e),
    }
}

/// f32 → f16 bits, rounded to nearest even as the GPU conversions round.
pub(crate) fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let (exp, man) = (((b >> 23) & 0xff) as i32, b & 0x7f_ffff);
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 112; // the f16 exponent field
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    // A normal f16 drops 13 mantissa bits; a subnormal drops 14 - e more, with the implicit 1.
    let (m, shift) = if e > 0 {
        (((e as u32) << 23) | man, 13)
    } else if e >= -10 {
        (man | 0x80_0000, (14 - e) as u32)
    } else {
        return sign;
    };
    let (h, rest, half) = (m >> shift, m & ((1 << shift) - 1), 1 << (shift - 1));
    // A carry out of the mantissa steps the exponent, up to infinity, as IEEE rounding does.
    sign | (h + u32::from(rest > half || (rest == half && h & 1 == 1))) as u16
}

impl Transformer {
    /// Loads a GGUF onto `dev`, with activations for steps of up to `max_rows` rows. Metal wraps the
    /// mapped file without a copy where it can (6.4); otherwise the weights stream through one
    /// staging chunk, so host memory stays at the header plus [`CHUNK`]. The token embedding stays
    /// on the device, as it does for [`Transformer::load_placed`] and [`Transformer::load_cached`]:
    /// only [`Transformer::load_planned`] moves it to the host, for a caller that runs
    /// [`Model::embed_tokens`] before every pass.
    pub fn load(
        path: &Path,
        dev: &Device,
        s: &Stream,
        max_rows: u32,
    ) -> Result<Transformer, LoadError> {
        Transformer::load_placed(path, dev, s, max_rows, Placement::default())
    }

    /// [`Transformer::load`] with the weights where `place` puts them.
    pub fn load_placed(
        path: &Path,
        dev: &Device,
        s: &Stream,
        max_rows: u32,
        place: Placement,
    ) -> Result<Transformer, LoadError> {
        Transformer::load_chunked(
            path,
            dev,
            s,
            max_rows,
            CHUNK,
            place,
            None,
            u64::MAX,
            false,
            false,
        )
    }

    /// [`Transformer::load_placed`] that also loads a Qwen 3.5 file's multi-token-prediction block
    /// past the trunk (the hybrid MoE target's blk.40) for [`MtpDrafter`], when it has one; the block's routed
    /// experts stay in host memory whenever any layer's do. The token embedding stays on the device.
    pub fn load_mtp(
        path: &Path,
        dev: &Device,
        s: &Stream,
        max_rows: u32,
        place: Placement,
    ) -> Result<Transformer, LoadError> {
        let (host_embed, mtp) = (false, true);
        Transformer::load_chunked(
            path,
            dev,
            s,
            max_rows,
            CHUNK,
            place,
            None,
            u64::MAX,
            host_embed,
            mtp,
        )
    }

    /// [`Transformer::load_placed`], or with `plan` [`Transformer::load_cached`], with the token
    /// embedding in host memory exactly when `host_embed` (no table to lend a drafter then,
    /// [`Model::embed_and_head`]); those two keep it on the device.
    pub fn load_planned(
        path: &Path,
        dev: &Device,
        s: &Stream,
        max_rows: u32,
        place: Placement,
        plan: Option<&ExpertPlan>,
        host_embed: bool,
    ) -> Result<Transformer, LoadError> {
        Transformer::load_chunked(
            path,
            dev,
            s,
            max_rows,
            CHUNK,
            place,
            plan,
            u64::MAX,
            host_embed,
            false,
        )
    }

    /// Whether the engine's load keeps the token embedding in host memory and gathers each step's
    /// rows there, as llama.cpp does: an MoE model on a discrete CUDA device, whose memory is worth
    /// more as experts than as a table a step reads a few rows of, unless the head reuses the table.
    pub fn embeds_on_host(props: &DeviceProps, c: &ModelConfig, moe: bool) -> bool {
        moe && !c.tie_embeddings
            && props.backend == Backend::Cuda
            && props.flags & DeviceProps::CAP_UNIFIED == 0
    }

    /// What the streaming pool asks of a load of `g` with host layers ([`Model::stream_pool_want`]):
    /// its slots, each the largest routed expert span ([`expert_span`]: a mixed layer's hot and
    /// cold tensors together), and the copy stream's workspace; 0 without routed experts.
    pub fn stream_pool_bytes(g: &Gguf) -> u64 {
        let most = (g.tensors.iter())
            .filter_map(|t| {
                let (i, rest) = t.name.strip_prefix("blk.")?.split_once('.')?;
                let p = rest.strip_prefix("ffn_")?.strip_suffix("_exps.weight")?;
                expert_span(g, i.parse().ok()?, p).map(|s| s.1)
            })
            .max();
        most.map_or(0, |b| stream_slots() as u64 * b + COPY_STREAM_BYTES)
    }

    /// The weight bytes of `g` a load on a device of `props` keeps out of device memory besides
    /// host experts: the side-only floats ([`side_only`]), and with `host_embed` the token
    /// embedding. Metal maps the file whole when it can, side-only floats included, so there none.
    pub fn off_device_bytes(
        g: &Gguf,
        c: &ModelConfig,
        props: &DeviceProps,
        host_embed: bool,
    ) -> u64 {
        let side = match props.backend {
            Backend::Metal => 0,
            _ => side_only(g, c).map(|t| t.bytes).sum(),
        };
        let embed = g
            .tensor("token_embd.weight")
            .filter(|_| host_embed && !c.tie_embeddings);
        side + embed.map_or(0, |t| t.bytes)
    }

    /// [`Transformer::load`] with each layer's routed experts where `plan` puts them. A split
    /// layer splits each row's top k: its resident experts run on the device (mmvq_id over the
    /// layer's slots, the ids remapped), its host experts on the CPU device, each weighted as
    /// usual and added to the f32 residual in a fixed order, the device's part then the host's,
    /// so a row's result depends on the residency and not on its batch, as far as the device's own
    /// mmvq_id holds that (the marker on `split_ffn`). Dynamic residency uploads hot host
    /// experts into cold slots on a second stream and maps them a step later; until then they run
    /// on the CPU, and the step that maps them waits for those uploads. Such a model refuses
    /// expert adapters and suffix training, whichever layers its plan splits.
    pub fn load_cached(
        path: &Path,
        dev: &Device,
        s: &Stream,
        max_rows: u32,
        plan: &ExpertPlan,
    ) -> Result<Transformer, LoadError> {
        let place = Placement::default();
        let plan = Some(plan);
        Transformer::load_chunked(
            path,
            dev,
            s,
            max_rows,
            CHUNK,
            place,
            plan,
            u64::MAX,
            false,
            false,
        )
    }

    /// [`Transformer::load_cached`] that also loads the MTP block, as [`Transformer::load_mtp`]
    /// does; its routed experts stay on the device, where the fit charges all of the block's bytes.
    /// The token embedding stays on the device.
    pub fn load_cached_mtp(
        path: &Path,
        dev: &Device,
        s: &Stream,
        max_rows: u32,
        plan: &ExpertPlan,
    ) -> Result<Transformer, LoadError> {
        let (place, plan) = (Placement::default(), Some(plan));
        Transformer::load_chunked(
            path,
            dev,
            s,
            max_rows,
            CHUNK,
            place,
            plan,
            u64::MAX,
            false,
            true,
        )
    }

    /// `max_buffer`: an upload past it splits as one past the device's limit does (tests force it).
    /// `host_embed`: the token embedding in host memory, unless the head reuses it.
    /// `mtp`: also the multi-token-prediction block past the trunk ([`Transformer::load_mtp`]).
    #[allow(clippy::too_many_arguments)] // load_cached's plan beside load_placed's inputs
    fn load_chunked(
        path: &Path,
        dev: &Device,
        s: &Stream,
        max_rows: u32,
        chunk: u64,
        place: Placement,
        plan: Option<&ExpertPlan>,
        max_buffer: u64,
        host_embed: bool,
        mtp: bool,
    ) -> Result<Transformer, LoadError> {
        let props = dev.props().map_err(backend)?;
        let device_backend = props.backend;
        let card = card::routes(&props, path);
        let quant_i8 = quant_i8_device(
            device_backend,
            props.arch_code,
            larch_kernels::routes().quant_i8,
        );
        let file = File::open(path).map_err(|_| LoadError::Io)?;
        let len = file.metadata().map_err(|_| LoadError::Io)?.len();
        let mut f = BufReader::new(file);
        let g = Gguf::read(&mut f, len)?;
        let cfg = ModelConfig::from_gguf(&g)?;
        let mmvq_max_rows = mmvq_crossover(device_backend, cfg.d_model);
        let (d, hd, ffn, vocab) = (
            cfg.d_model as i64,
            cfg.head_dim as i64,
            cfg.ffn as i64,
            cfg.vocab as i64,
        );
        let (qd, kvd) = (cfg.n_heads as i64 * hd, cfg.n_kv_heads as i64 * hd);
        let moe = MoeCfg::from_gguf(&g, cfg.arch)?;
        let n = cfg.n_layers;
        let nextn = mtp
            && cfg.arch.hybrid()
            && g.tensor(&format!("blk.{n}.nextn.eh_proj.weight")).is_some();
        // Whole host layers or split ones, never both: they share the host experts' buffer.
        let host_layers = match plan {
            None => moe.map_or(0, |_| place.cpu_expert_layers.min(cfg.n_layers)),
            Some(_) => 0,
        };
        // A split layer's resident experts; every list's experts in range and listed once.
        let splits: Vec<Option<Vec<u32>>> = match (moe, plan) {
            (Some(m), Some(p)) => (0..cfg.n_layers as usize)
                .map(|i| {
                    let Some(r) = p.resident.get(i) else {
                        return Ok(None);
                    };
                    let mut seen = vec![false; m.experts as usize];
                    let once = r
                        .iter()
                        .all(|&e| e < m.experts && !std::mem::replace(&mut seen[e as usize], true));
                    match (once, r.len() < m.experts as usize) {
                        (false, _) => Err(LoadError::Backend(KError::InvalidArg.code())),
                        (true, split) => Ok(split.then(|| r.clone())),
                    }
                })
                .collect::<Result<_, _>>()?,
            _ => Vec::new(),
        };
        // The MTP block's experts leave the device first: a whole-layer load with any host layer
        // keeps them in host memory (one CPU section a drafter step).
        let off_device = |i: u32| {
            i < host_layers
                || (i == n && host_layers > 0)
                || splits.get(i as usize).is_some_and(Option::is_some)
        };
        // Mixed layers' cold experts (`larch repack --cold-fraction`): each list ascending, under the
        // expert count and shorter than it, or the file is refused. The expert cache splits whole
        // tensors only.
        let cold: Vec<Option<Vec<u32>>> = (0..n)
            .map(|i| {
                let Some(v) = g.get(&crate::q2k::cold_key(i)) else {
                    return Ok(None);
                };
                let e = moe.map_or(0, |m| m.experts);
                let ids: Option<Vec<u32>> = v.as_arr().and_then(|a| {
                    a.iter()
                        .map(|x| x.as_u64().and_then(|x| u32::try_from(x).ok()))
                        .collect()
                });
                match ids {
                    Some(c)
                        if !c.is_empty()
                            && c.len() < e as usize
                            && c.windows(2).all(|w| w[0] < w[1])
                            && c.last().is_some_and(|&l| l < e) =>
                    {
                        Ok(Some(c))
                    }
                    _ => Err(LoadError::BadGguf(code::KEY, 0)),
                }
            })
            .collect::<Result<_, _>>()?;
        if plan.is_some() && cold.iter().any(Option::is_some) {
            return Err(LoadError::Backend(KError::InvalidArg.code()));
        }
        // The routed experts kept out of the weights buffer, by data-section offset: (offset,
        // bytes), a mixed layer's cold ones too. Missing ones fail below, as in any layer.
        let host_names: Vec<String> = (0..n + u32::from(nextn))
            .filter(|&i| off_device(i))
            .flat_map(|i| {
                let mixed = cold.get(i as usize).is_some_and(Option::is_some);
                ["gate", "up", "down"].into_iter().flat_map(move |p| {
                    std::iter::once(format!("blk.{i}.ffn_{p}_exps.weight"))
                        .chain(mixed.then(|| crate::q2k::cold_name(i, p)))
                })
            })
            .collect();
        let mut host: Vec<(u64, u64)> = host_names
            .iter()
            .filter_map(|n| g.tensor(n))
            .map(|t| (t.offset, t.bytes))
            .collect();
        host.sort_unstable();
        // Each split layer's maps and slot tables, each table 256-aligned in the device's or the
        // host's buffer; and its expert tensors, which hold their experts' slices in order.
        let dynamic = plan.is_some_and(|p| p.dynamic);
        let mut layout: Vec<Option<Split>> = (0..splits.len()).map(|_| None).collect();
        let mut tables = Vec::new();
        let (mut slots_len, mut held_len) = (0u64, 0u64);
        for (i, r) in splits.iter().enumerate() {
            let (Some(r), Some(m)) = (r, moe) else {
                continue;
            };
            let e = m.experts as usize;
            let mut slot = vec![-1i32; e];
            for (j, &x) in r.iter().enumerate() {
                slot[x as usize] = j as i32;
            }
            let mut hslot = vec![-1i32; e];
            let mut held = 0;
            for x in (0..e).filter(|&x| dynamic || slot[x] < 0) {
                (hslot[x], held) = (held, held + 1);
            }
            let mut ws = [[None; 3]; 2];
            let mut slices = [0; 3];
            for (which, p) in ["gate", "up", "down"].into_iter().enumerate() {
                let t = g
                    .tensor(&format!("blk.{i}.ffn_{p}_exps.weight"))
                    .ok_or(LoadError::BadGguf(code::MISSING_TENSOR, 0))?;
                let (shape, slice) = (t.shape(), t.bytes / e as u64);
                if shape.len() != 3 || t.bytes % e as u64 != 0 {
                    return Err(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset));
                }
                let dtype = gguf::dtype(t.ggml_type)?;
                let w = |off, slots: usize| W {
                    off,
                    dtype,
                    shape: [slots as i64 * shape[1], shape[2]],
                    ndim: 2,
                    side: false,
                };
                ws[0][which] = Some(w(slots_len, r.len()));
                ws[1][which] = Some(w(held_len, held as usize));
                slices[which] = slice;
                tables.push((t.offset, i, slots_len, held_len, slice));
                slots_len += (slice * r.len() as u64).next_multiple_of(256);
                held_len += (slice * held as u64).next_multiple_of(256);
            }
            let [dev, host] = ws.map(|w| w.map(|w| w.expect("gate, up and down")));
            layout[i] = Some(Split {
                dev,
                host,
                slot,
                hslot,
                slice: slices,
            });
        }
        let host_embed = host_embed && !cfg.tie_embeddings;

        // The wrap starts at the file's start, so a tensor sits at its file offset there. The map handle
        // drops here and the file at the end of `load`: the device holds the map until it closes.
        // Only a map of the length `g` was checked against, and only with the data section 32-aligned
        // (llama.cpp's default; a GGUF may pick less) so each tensor is as aligned as in the upload.
        // The map ends at the last page a used tensor needs: a built-in drafter's layers past the
        // config's (the hybrid MoE target's blk.40, 856 MiB at the file's end) stay out of the GPU's working set.
        let spans = used_spans(&g, n + u32::from(nextn));
        let end = spans
            .iter()
            .map(|t| t.1.div_ceil(PAGE) * PAGE)
            .max()
            .map_or(len, |e| e.min(len));
        let mapped = match Mmap::map(f.get_ref()) {
            Ok(map)
                if map.bytes() == len
                    && g.data_offset % 32 == 0
                    && host.is_empty()
                    && !host_embed =>
            {
                match Buffer::wrap_host(dev, &map, 0..end) {
                    Ok(b) => Some((b, None)),
                    // Past maxBufferLength (14.3 GB on a 24 GB M5): two wraps that meet at a tensor.
                    // Where those fail too, or there is no wrap, the upload loads or, past the limit,
                    // reports DoesNotFit after the tensors are validated.
                    Err(KError::Oom) => {
                        wrap_halves(dev, &map, spans, end)?.map(|(lo, hi, at)| (lo, Some((hi, at))))
                    }
                    Err(KError::Unsupported) => None,
                    Err(e) => return Err(backend(e)),
                }
            }
            // The map is only Metal's shortcut; the upload reads the same file and reports a real fault.
            _ => None,
        };
        let base = if mapped.is_some() { g.data_offset } else { 0 };
        // What the weights buffer leaves out, by data-section offset: the host experts and, on an
        // upload, the side-only floats (the side buffer reads them from the file) and a host
        // embedding's table.
        let embd = g
            .tensor("token_embd.weight")
            .filter(|_| host_embed)
            .map(|t| (t.offset, t.bytes));
        let mut gone = host.clone();
        if mapped.is_none() {
            gone.extend(side_only(&g, &cfg).map(|t| (t.offset, t.bytes)));
            gone.extend(embd);
        }
        gone.sort_unstable();
        if let Some(w) = gone.windows(2).find(|w| w[0].0 + w[0].1 > w[1].0) {
            return Err(LoadError::BadGguf(code::TENSOR, g.data_offset + w[1].0));
        }
        // A tensor's offset in its buffer: a host expert's among the host experts, the host
        // embedding's at the start of its table, any other's in the data section without what is
        // gone. Any other tensor that meets gone bytes (an alias at its start, or inside it, or a
        // read of a side-only float) would sit in none: BadGguf.
        let sum_before = |spans: &[(u64, u64)], off: u64| -> u64 {
            spans.iter().take_while(|r| r.0 < off).map(|r| r.1).sum()
        };
        let place_of = |t: &gguf::TensorInfo| -> Result<u64, LoadError> {
            if host_names.contains(&t.name) {
                return Ok(sum_before(&host, t.offset));
            }
            if t.name == "token_embd.weight" && embd.is_some() {
                return Ok(0);
            }
            let end = t.offset.saturating_add(t.bytes.max(1));
            if gone.iter().any(|&(o, b)| t.offset < o + b && o < end) {
                return Err(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset));
            }
            Ok(t.offset - sum_before(&gone, t.offset))
        };

        let find = |name: &str, shape: &[i64]| -> Result<Option<W>, LoadError> {
            let Some(t) = g.tensor(name) else {
                return Ok(None);
            };
            if t.shape() != shape {
                return Err(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset));
            }
            // 6.1: vectors (norms, biases, rope_freqs) are f32; matrices go through mmvq or dequant,
            // which take quants only, except the embedding table when it is not also the lm_head.
            let dtype = gguf::dtype(t.ggml_type)?;
            let ok = match shape.len() {
                1 => dtype == DType::F32,
                _ => dtype.block() > 1 || (name == "token_embd.weight" && !cfg.tie_embeddings),
            };
            if !ok {
                return Err(LoadError::UnsupportedQuant(t.ggml_type));
            }
            // Stacked experts [experts, n, k] are one [experts * n, k] matrix, as mmvq_id takes them.
            let (s, ndim) = match *shape {
                [e, n, k] => ([e * n, k], 2),
                _ => {
                    let mut s = [0; 2];
                    s[..shape.len()].copy_from_slice(shape);
                    (s, shape.len())
                }
            };
            Ok(Some(W {
                off: base + place_of(t)?,
                dtype,
                shape: s,
                ndim,
                side: false,
            }))
        };
        let need = |name: &str, shape: &[i64]| {
            find(name, shape)?.ok_or(LoadError::BadGguf(code::MISSING_TENSOR, 0))
        };
        // Required in every layer when the arch has them, as llama.cpp loads them.
        let need_if = |on: bool, name: &str, shape: &[i64]| {
            if on {
                need(name, shape).map(Some)
            } else {
                find(name, shape)
            }
        };
        let norm = |name: &str| cfg.qk_norm.then(|| need(name, &[hd])).transpose();
        // An f32 matrix the ops read as f32 (Gated DeltaNet's conv weight).
        let need_f32 = |name: &str, shape: &[i64]| -> Result<W, LoadError> {
            let t = g
                .tensor(name)
                .ok_or(LoadError::BadGguf(code::MISSING_TENSOR, 0))?;
            if t.shape() != shape {
                return Err(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset));
            }
            if gguf::dtype(t.ggml_type)? != DType::F32 {
                return Err(LoadError::UnsupportedQuant(t.ggml_type));
            }
            Ok(W {
                off: base + place_of(t)?,
                dtype: DType::F32,
                shape: [shape[0], shape[1]],
                ndim: 2,
                side: false,
            })
        };

        // Float matrices as llama.cpp writes them (f32, f16 or bf16), seen as `[rows, d]`: file
        // offset, float type and elements, in [`Transformer::side`] order; packed as f16 there
        // after the upload.
        let sides = std::cell::RefCell::new(Vec::<(u64, DType, u64)>::new());
        // The hybrid MoE target's MTP block keeps its router and shared-expert gate in bf16 (GGML type 30); the
        // trunk's trainer reads no bf16, so only that block may.
        let mtp_blk = format!("blk.{n}.");
        let bf16 = |name: &str, ty: u32| ty == 30 && nextn && name.starts_with(&mtp_blk);
        let side = |name: &str, shape: &[i64]| -> Result<W, LoadError> {
            let t = g
                .tensor(name)
                .ok_or(LoadError::BadGguf(code::MISSING_TENSOR, 0))?;
            if t.shape() != shape {
                return Err(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset));
            }
            let ty = match t.ggml_type {
                ty if bf16(name, ty) => DType::BF16,
                ty => match gguf::dtype(ty)? {
                    t @ (DType::F32 | DType::F16) => t,
                    _ => return Err(LoadError::UnsupportedQuant(ty)),
                },
            };
            let n: i64 = shape.iter().product();
            let mut sides = sides.borrow_mut();
            let off = sides.iter().map(|s| DType::F16.bytes(s.2)).sum();
            sides.push((g.data_offset + t.offset, ty, n as u64));
            Ok(W {
                off,
                dtype: DType::F16,
                shape: [n / shape[shape.len() - 1], shape[shape.len() - 1]],
                ndim: 2,
                side: true,
            })
        };
        // A float matrix `side` has checked, where the GGUF stores it: the trainer reads the router so.
        let raw = |name: &str, shape: &[i64]| -> Result<W, LoadError> {
            let t = g
                .tensor(name)
                .ok_or(LoadError::BadGguf(code::MISSING_TENSOR, 0))?;
            let dtype = match t.ggml_type {
                ty if bf16(name, ty) => DType::BF16,
                ty => gguf::dtype(ty)?,
            };
            Ok(W {
                off: base + place_of(t)?,
                dtype,
                shape: [shape[0], shape[1]],
                ndim: 2,
                side: false,
            })
        };
        // A matrix quantized, or float as an f16 side copy.
        let proj = |name: &str, shape: &[i64]| -> Result<W, LoadError> {
            let float = g
                .tensor(name)
                .and_then(|t| gguf::dtype(t.ggml_type).ok())
                .is_some_and(|t| matches!(t, DType::F32 | DType::F16));
            if float {
                side(name, shape)
            } else {
                need(name, shape)
            }
        };

        let hybrid = qwen35::Hybrid::from_gguf(&g, &cfg)?;
        let tok_embd = need("token_embd.weight", &[vocab, d])?;
        let (mut kv_layer, mut gdn_layer) = (0, 0);
        // Layer `i`: the trunk's, or past it the MTP block's, a full attention layer.
        let mut layer = |i: u32| {
            let n = |s: &str| format!("blk.{i}.{s}");
            let recurrent =
                (hybrid.as_ref()).is_some_and(|h| h.recurrent.get(i as usize) == Some(&true));
            let mixer = match &hybrid {
                Some(h) if recurrent => {
                    gdn_layer += 1;
                    Mixer::Gdn(qwen35::Gdn::load(
                        h,
                        d,
                        i,
                        gdn_layer - 1,
                        &need,
                        &need_f32,
                        &proj,
                    )?)
                }
                _ => {
                    kv_layer += 1;
                    // Qwen 3.5's Q projection holds a gate beside each head.
                    let wq = (1 + i64::from(hybrid.is_some())) * qd;
                    Mixer::Attn(Attn {
                        wq: need(&n("attn_q.weight"), &[wq, d])?,
                        wk: need(&n("attn_k.weight"), &[kvd, d])?,
                        wv: need(&n("attn_v.weight"), &[kvd, d])?,
                        bq: need_if(cfg.qkv_bias, &n("attn_q.bias"), &[qd])?,
                        bk: need_if(cfg.qkv_bias, &n("attn_k.bias"), &[kvd])?,
                        bv: need_if(cfg.qkv_bias, &n("attn_v.bias"), &[kvd])?,
                        // Only qwen3 and 3.5 have them; llama.cpp refuses them elsewhere, so they
                        // never run there.
                        q_norm: norm(&n("attn_q_norm.weight"))?,
                        k_norm: norm(&n("attn_k_norm.weight"))?,
                        wo: need(&n("attn_output.weight"), &[d, qd])?,
                        gated: hybrid.is_some(),
                        kv_layer: kv_layer - 1,
                    })
                }
            };
            let ffn_norm = match hybrid {
                Some(_) => "post_attention_norm.weight",
                None => "ffn_norm.weight",
            };
            Ok(Layer {
                attn_norm: need(&n("attn_norm.weight"), &[d])?,
                mixer,
                ffn_norm: need(&n(ffn_norm), &[d])?,
                ffn: match moe {
                    None => Ffn::Dense {
                        gate: need(&n("ffn_gate.weight"), &[ffn, d])?,
                        up: need(&n("ffn_up.weight"), &[ffn, d])?,
                        down: need(&n("ffn_down.weight"), &[d, ffn])?,
                    },
                    Some(m) => {
                        let (e, ef) = (i64::from(m.experts), i64::from(m.ffn));
                        // A mixed layer's tensors: its hot experts, and its cold ones apart.
                        let c = cold.get(i as usize).and_then(Option::as_ref);
                        let ce = c.map_or(0, |c| c.len() as i64);
                        let (he, cold) = (e - ce, c.map(|_| [ce, ef, d]));
                        let shared = match hybrid.as_ref().map_or(0, |h| h.shared_ffn) {
                            0 => None,
                            sf => {
                                let sf = i64::from(sf);
                                Some(qwen35::Shared {
                                    gate_inp: side(&n("ffn_gate_inp_shexp.weight"), &[d])?,
                                    gate: need(&n("ffn_gate_shexp.weight"), &[sf, d])?,
                                    up: need(&n("ffn_up_shexp.weight"), &[sf, d])?,
                                    down: need(&n("ffn_down_shexp.weight"), &[d, sf])?,
                                })
                            }
                        };
                        Ffn::Moe {
                            router: side(&n("ffn_gate_inp.weight"), &[e, d])?,
                            raw: (cfg.arch == Arch::Qwen3Moe)
                                .then(|| raw(&n("ffn_gate_inp.weight"), &[e, d]))
                                .transpose()?,
                            gate: need(&n("ffn_gate_exps.weight"), &[he, ef, d])?,
                            up: need(&n("ffn_up_exps.weight"), &[he, ef, d])?,
                            down: need(&n("ffn_down_exps.weight"), &[he, d, ef])?,
                            host: off_device(i),
                            shared,
                            cold: cold
                                .map(|[ce, ef, d]| {
                                    let cn = |p| crate::q2k::cold_name(i, p);
                                    Ok::<_, LoadError>(Cold {
                                        gate: need(&cn("gate"), &[ce, ef, d])?,
                                        up: need(&cn("up"), &[ce, ef, d])?,
                                        down: need(&cn("down"), &[ce, d, ef])?,
                                    })
                                })
                                .transpose()?,
                        }
                    }
                },
            })
        };
        let layers = (0..n)
            .map(&mut layer)
            .collect::<Result<Vec<_>, LoadError>>()?;
        let mtp = match nextn {
            true => {
                let mut l = layer(n)?;
                // Its K and V live in each sequence's MtpCache, a pool of one layer.
                if let Mixer::Attn(a) = &mut l.mixer {
                    a.kv_layer = 0;
                }
                let t = |s: &str| format!("blk.{n}.nextn.{s}.weight");
                Some(qwen35_mtp::Nextn {
                    eh_proj: need(&t("eh_proj"), &[d, 2 * d])?,
                    enorm: need(&t("enorm"), &[d])?,
                    hnorm: need(&t("hnorm"), &[d])?,
                    // ponytail: llama.cpp also prefers nextn.embed_tokens and nextn.shared_head_head
                    // when present; absent from the hybrid MoE target's header, so refuse or load them when a
                    // GGUF carries either.
                    // llama.cpp: without its own head norm, the block takes the trunk's.
                    head_norm: match find(&t("shared_head_norm"), &[d])? {
                        Some(w) => w,
                        None => need("output_norm.weight", &[d])?,
                    },
                    layer: l,
                })
            }
            false => None,
        };
        let output = if cfg.tie_embeddings {
            "token_embd.weight"
        } else {
            "output.weight"
        };
        let output = need(output, &[vocab, d])?;
        let out_norm = need("output_norm.weight", &[d])?;
        let rope_freqs = find("rope_freqs.weight", &[cfg.rope.n_dims as i64 / 2])?;

        let alloc =
            |bytes: u64| Buffer::alloc(dev, bytes.max(1)).map_err(|e| fit_err(dev, bytes, e));
        let r = max_rows as i64;
        let mut host_experts = None;
        let mut host_embd = None;
        // Each host tensor's (file start, bytes, offset in the host experts' weights).
        let mut host_at: Vec<(u64, u64, u64)> = Vec::new();
        let mut cache_dev = None;
        let (weights, hi) = match mapped {
            Some(b) => b,
            None => {
                // As the map: a built-in drafter's layers at the file's end are not uploaded.
                let data_len = end.saturating_sub(g.data_offset);
                let host_bytes: u64 = host.iter().map(|r| r.1).sum();
                if gone.last().is_some_and(|l| l.0 + l.1 > data_len) {
                    return Err(LoadError::BadGguf(code::TENSOR, len));
                }
                // The device's tensors without what is gone; past maxBufferLength (14.3 GB on a
                // 24 GB M5) in two buffers split as the wraps are.
                let dev_len = data_len - gone.iter().map(|r| r.1).sum::<u64>();
                let whole = match dev_len <= max_buffer {
                    true => alloc(dev_len),
                    false => Err(LoadError::DoesNotFit {
                        need: dev_len,
                        have: max_buffer,
                    }),
                };
                let (weights, hi) = match whole {
                    Ok(w) => (w, None),
                    Err(e @ LoadError::DoesNotFit { .. }) => {
                        let spans = g
                            .tensors
                            .iter()
                            .filter(|t| !gone.contains(&(t.offset, t.bytes)))
                            .filter_map(|t| place_of(t).ok().map(|o| (o, o + t.bytes)))
                            .collect();
                        let Some((lower, upper)) = halves(spans, dev_len) else {
                            return Err(e);
                        };
                        (alloc(lower)?, Some((alloc(dev_len - upper)?, upper)))
                    }
                    Err(e) => return Err(e),
                };
                let gpu: Vec<(&Buffer, u64)> = [(&weights, 0)]
                    .into_iter()
                    .chain(hi.as_ref().map(|(b, at)| (b, *at)))
                    .collect();
                let mut staging = vec![0u8; chunk.min(data_len) as usize];
                // File bytes `from..from + n` to `to..` of the layout `dst` holds: each buffer
                // from its start, the bytes where two overlap into both.
                let mut copy = |from: u64, n: u64, s: &Stream, dst: &[(&Buffer, u64)], to: u64| {
                    f.seek(SeekFrom::Start(g.data_offset + from))
                        .map_err(|_| LoadError::Io)?;
                    for off in (0..n).step_by(chunk as usize) {
                        let c = &mut staging[..chunk.min(n - off) as usize];
                        f.read_exact(c).map_err(|_| LoadError::Io)?;
                        let at = to + off;
                        for &(b, start) in dst {
                            let lo = at.max(start);
                            let hi = (at + c.len() as u64).min(start + b.bytes());
                            if lo < hi {
                                let piece = &c[(lo - at) as usize..(hi - at) as usize];
                                s.upload(b, lo - start, piece).map_err(backend)?;
                            }
                        }
                    }
                    Ok::<(), LoadError>(())
                };
                let cpu = match host.is_empty() && embd.is_none() {
                    true => None,
                    false => {
                        let cpu = Device::open(CPU_COMPUTE).map_err(backend)?;
                        // Beside a GPU only: a CPU backend's host layers have no driver to spare.
                        if s.device().props().map_err(backend)?.backend != Backend::Cpu {
                            let logical =
                                std::thread::available_parallelism().map_or(1, |n| n.get());
                            cpu.set_threads(beside_gpu_threads(logical))
                                .map_err(backend)?;
                        }
                        Some(cpu)
                    }
                };
                let on_cpu = || cpu.as_ref().ok_or(backend(KError::InvalidArg));
                let calloc = |bytes: u64| {
                    let cpu = on_cpu()?;
                    Buffer::alloc(cpu, bytes.max(1)).map_err(|e| fit_err(cpu, bytes, e))
                };
                let cstream = || Stream::new(on_cpu()?).map_err(backend);
                let he = match moe.filter(|_| !host.is_empty()) {
                    None => None,
                    Some(m) => {
                        let (k, cols) = (i64::from(m.used), i64::from(m.used) * i64::from(m.ffn));
                        let cbuf = |t: DType, cols: i64| calloc(t.bytes((r * cols) as u64));
                        // A split layer's host experts take each (row, slot) they get as a row,
                        // and return its weighted f32 output for the host to sum.
                        let split = !tables.is_empty();
                        let (held, out) = match split {
                            true => (held_len, cbuf(DType::F32, k * d)?),
                            false => (host_bytes, cbuf(DType::F16, d)?),
                        };
                        // Where the pool may stream them, in host memory the GPU copies from at
                        // full speed (page-locked: 57 against 22 GB/s through the file map and
                        // the staging on a 5070 Ti); plain memory where the device has none.
                        // As the pool below asks: no expert-cache plan, no split layer.
                        let pinned = match place.stream_rows > 0 && plan.is_none() && !split {
                            false => None,
                            true => match Buffer::alloc_host(s.device(), held) {
                                Ok(b) => Some(b),
                                Err(KError::Unsupported | KError::Oom) => None,
                                Err(e) => return Err(backend(e)),
                            },
                        };
                        Some(HostExperts {
                            weights: pinned.map_or_else(|| calloc(held), Ok)?,
                            h: cbuf(DType::F16, if split { k * d } else { d })?,
                            route: cbuf(DType::F32, 2 * k)?,
                            gate: cbuf(DType::F16, cols)?,
                            up: cbuf(DType::F16, cols)?,
                            act: cbuf(DType::F16, cols)?,
                            out,
                            staging: Mutex::new(vec![
                                0;
                                (DType::F32.bytes((2 * r * k) as u64)
                                    + DType::F16.bytes((r * d) as u64))
                                    as usize
                            ]),
                            s: cstream()?,
                        })
                    }
                };
                let he_embd = embd
                    .map(|(_, bytes)| {
                        Ok::<_, LoadError>(HostEmbed {
                            s: cstream()?,
                            w: tok_embd,
                            table: calloc(bytes)?,
                            ids: calloc(DType::I32.bytes(r as u64))?,
                            out: calloc(DType::F32.bytes((r * d) as u64))?,
                            x: RefCell::new(vec![0f32; (r * d) as usize]),
                            primed: Cell::new(None),
                        })
                    })
                    .transpose()?;
                // The data section in file order, each piece to the end of its buffer.
                // ponytail: one buffer for every split's slots, which max_buffer refuses and the
                // load falls back to whole layers (a 24 GB M5 past 14.3 GB of resident experts);
                // per-layer buffers when that shows.
                let slots = (!tables.is_empty()).then(|| alloc(slots_len)).transpose()?;
                let (mut at, mut gpu_at, mut cpu_at) = (0, 0, 0);
                for &(start, bytes) in &gone {
                    copy(at, start - at, s, &gpu, gpu_at)?;
                    let he = he.as_ref().filter(|_| host.contains(&(start, bytes)));
                    match (he, tables.iter().find(|t| t.0 == start), &slots, &he_embd) {
                        (Some(he), Some(&(_, i, dat, hat, slice)), Some(dev), _) => {
                            let sp = layout[i].as_ref().expect("a split's layout");
                            for (x, (&ds, &hs)) in sp.slot.iter().zip(&sp.hslot).enumerate() {
                                let from = start + x as u64 * slice;
                                if ds >= 0 {
                                    let to = dat + ds as u64 * slice;
                                    copy(from, slice, s, &[(dev, 0)], to)?;
                                }
                                if hs >= 0 {
                                    let to = hat + hs as u64 * slice;
                                    copy(from, slice, &he.s, &[(&he.weights, 0)], to)?;
                                }
                            }
                        }
                        (Some(he), ..) => {
                            copy(start, bytes, &he.s, &[(&he.weights, 0)], cpu_at)?;
                            host_at.push((start, bytes, cpu_at));
                            cpu_at += bytes;
                        }
                        (None, _, _, Some(e)) if embd == Some((start, bytes)) => {
                            copy(start, bytes, &e.s, &[(&e.table, 0)], 0)?;
                        }
                        // A side-only float: the side buffer reads it from the file below.
                        _ => {}
                    }
                    (gpu_at, at) = (gpu_at + start - at, start + bytes);
                }
                copy(at, data_len - at, s, &gpu, gpu_at)?;
                cache_dev = slots;
                host_experts = he;
                host_embd = he_embd;
                (weights, hi)
            }
        };
        // ponytail: a split layer stays on the CPU path: the pool would come out of the resident set the
        // plan just filled, and one mmvq_id reads one buffer, so a row's experts would need two passes
        // and the split path's f32 sum. Stream its host experts if the expert cache becomes the default.
        let stream_map = (place.stream_rows > 0 && plan.is_none() && host_experts.is_some())
            // Every error counts as no pool: mmap can only refuse this file here (ENOMEM, map count,
            // a filesystem without mapping, Unsupported off unix), the upload read the same file.
            .then(|| Mmap::map(f.get_ref()).ok().filter(|m| m.bytes() == len))
            .flatten();
        let stream_plan = match stream_map {
            Some(map) => {
                let (mut layers, mut slot_bytes) = (Vec::new(), 0);
                for i in 0..host_layers {
                    let mut pieces = [(0, 0, 0, None); 3];
                    for (p, c) in ["gate", "up", "down"].iter().zip(&mut pieces) {
                        // `need` above checked the tensors exist.
                        *c = expert_span(&g, i, p)
                            .ok_or(LoadError::BadGguf(code::TENSOR, g.data_offset))?;
                        slot_bytes = slot_bytes.max(c.1);
                    }
                    layers.push(pieces);
                }
                // From here the weights are read-only: the pool's copy thread reads them too.
                let src = host_experts.as_mut().and_then(|h| h.weights.share_host());
                let at = (layers.iter())
                    .map(|l| l.map(|p| held_at(&host_at, p.0 - g.data_offset, p.1)))
                    .collect();
                Some(StreamPlan {
                    layers,
                    slot_bytes,
                    slots: stream_slots(),
                    map,
                    src: src.map(|src| (src, at)),
                })
            }
            None => None,
        };
        let sides = sides.into_inner();
        let side = match sides.iter().map(|s| s.2).max() {
            None => None,
            Some(most) => {
                let side = alloc(sides.iter().map(|s| DType::F16.bytes(s.2)).sum())?;
                let mut raw = vec![0u8; DType::F32.bytes(most) as usize];
                let mut at = 0;
                for &(off, ty, n) in &sides {
                    let raw = &mut raw[..ty.bytes(n) as usize];
                    f.seek(SeekFrom::Start(off)).map_err(|_| LoadError::Io)?;
                    f.read_exact(raw).map_err(|_| LoadError::Io)?;
                    let half: Vec<u16> = match ty {
                        DType::F32 => raw
                            .chunks_exact(4)
                            .map(|b| f16_bits(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                            .collect(),
                        // bf16 is an f32's top half.
                        DType::BF16 => raw
                            .chunks_exact(2)
                            .map(|b| f16_bits(f32::from_le_bytes([0, 0, b[0], b[1]])))
                            .collect(),
                        _ => raw
                            .chunks_exact(2)
                            .map(|b| u16::from_le_bytes([b[0], b[1]]))
                            .collect(),
                    };
                    s.upload(&side, at, &half).map_err(backend)?;
                    at += DType::F16.bytes(n);
                }
                Some(side)
            }
        };

        let cache = match (cache_dev, moe) {
            (Some(slots), Some(m)) => Some(RefCell::new(ExpertCache {
                copy: dynamic
                    .then(|| Stream::new(dev))
                    .transpose()
                    .map_err(backend)?,
                dev: Some(slots),
                splits: layout,
                hist: Histogram::uniform(0, cfg.n_layers, m.experts),
                path: None,
                pending: Vec::new(),
                passes: 0,
                writer: None,
                logged: None,
                hits: (0, 0),
                no_host: 0,
                split_steps: 0,
                log_later: false,
                staged: Vec::new(),
            })),
            _ => None,
        };
        // Each mixed layer's expert table (ops::mmvq_id_mixed's), on the device that runs its experts,
        // and a host layer's on the device too, for the stream pool: a hot expert's slot among the hot
        // ones, a cold one's -1 - its slot among the cold ones.
        let mut stream_tables = Vec::new();
        let cold_tables = (cold.iter().enumerate())
            .map(|(i, c)| {
                let Some(c) = c else {
                    return Ok(None);
                };
                let e = moe.map_or(0, |m| m.experts);
                let (mut hot, mut cs) = (0i32, 0i32);
                let table: Vec<i32> = (0..e)
                    .map(|x| match c.binary_search(&x) {
                        Ok(_) => {
                            cs += 1;
                            -cs
                        }
                        Err(_) => {
                            hot += 1;
                            hot - 1
                        }
                    })
                    .collect();
                let bytes = DType::I32.bytes(u64::from(e));
                let on_device = || -> Result<Buffer, LoadError> {
                    let b = alloc(bytes)?;
                    s.upload(&b, 0, &table).map_err(backend)?;
                    Ok(b)
                };
                let Some(x) = host_experts.as_ref().filter(|_| off_device(i as u32)) else {
                    return Ok(Some(on_device()?));
                };
                stream_tables.resize_with(i + 1, || None);
                stream_tables[i] = stream_plan.as_ref().map(|_| on_device()).transpose()?;
                let b = Buffer::alloc(x.s.device(), bytes).map_err(backend)?;
                x.s.upload(&b, 0, &table).map_err(backend)?;
                Ok(Some(b))
            })
            .collect::<Result<Vec<_>, LoadError>>()?;
        let buf = |t: DType, cols: i64| alloc(t.bytes((r * cols) as u64));
        let largest = (layers.iter().chain(mtp.as_ref().map(|m| &m.layer)))
            .flat_map(Layer::linears)
            .chain(mtp.as_ref().map(|m| m.eh_proj))
            .filter(|w| !w.side && !use_mmvq(mmvq_max_rows, *w, r))
            .map(|w| w.shape[0] * w.shape[1])
            .max()
            .unwrap_or(0);
        // An MoE layer's gate, up and act hold every chosen expert's rows, or the shared expert's.
        let shared = hybrid.as_ref().map_or(0, |h| i64::from(h.shared_ffn));
        let ffn_cols = moe.map_or(ffn, |m| (i64::from(m.used) * i64::from(m.ffn)).max(shared));
        let sc = Scratch {
            x: buf(DType::F32, d)?,
            h: buf(DType::F16, d)?,
            q: buf(DType::F16, qd)?,
            k: buf(DType::F16, kvd)?,
            v: buf(DType::F16, kvd)?,
            attn: buf(DType::F16, qd)?,
            t: buf(DType::F32, d)?,
            gate: buf(DType::F16, ffn_cols)?,
            up: buf(DType::F16, ffn_cols)?,
            act: buf(DType::F16, ffn_cols)?,
            route: moe
                .map(|m| {
                    buf(
                        DType::F32,
                        i64::from(m.experts) + 2 * i64::from(m.used) + (d + 1) / 2,
                    )
                })
                .transpose()?,
            dq: if largest > 0 {
                Some(alloc(DType::F16.bytes(largest as u64))?)
            } else {
                None
            },
        };
        let hy = hybrid
            .as_ref()
            .map(|h| qwen35::Scratch::alloc(h, &cfg, r, s, &alloc))
            .transpose()?;
        let t = Transformer {
            cfg,
            weights,
            hi,
            cold_tables,
            stream_tables,
            side,
            moe,
            hybrid,
            hy,
            states: None,
            router_deltas: Vec::new(),
            adapters: Vec::new(),
            host: host_experts,
            stream_plan,
            stream: None,
            stream_rows: place.stream_rows,
            streaming: false,
            stream_decode: 0,
            tok_embd: host_embd.is_none().then_some(tok_embd),
            embd: host_embd,
            cache,
            route_log: None,
            out_norm,
            output,
            rope_freqs,
            layers,
            mtp,
            sc,
            max_rows: r,
            mmvq_max_rows,
            backend: device_backend,
            card,
            quant_i8,
            compute_units: props.compute_units,
            arch_code: props.arch_code,
            check: std::cell::Cell::new(false),
            exact_checks: crate::exact_checks(props.backend),
            id_rows: match props.backend {
                Backend::Metal => METAL_ID_ROWS,
                _ => ID_ROWS,
            },
            checks_ran: false,
            fused: larch_kernels::routes().fused,
            fused_on: fused_defaults(device_backend, props.arch_code),
            refuse_fused: false,
        };
        t.refuse_unsupported(dev, s)?;
        Ok(t)
    }

    /// A load whose device lacks a call its Q2_K weights need is refused here with UnsupportedQuant
    /// (Q2_K's GGML type), not at its first step: mmvq for a Q2_K matrix that `linear` runs,
    /// mmvq_id for Q2_K experts (a host layer's may stream to the device) and the mixed call for a
    /// mixed layer, each a one-row call on zeros in a buffer of its own (a zero-row call may return
    /// before a backend looks at the dtype).
    fn refuse_unsupported(&self, dev: &Device, s: &Stream) -> Result<(), LoadError> {
        let layers: Vec<&Layer> = self
            .layers
            .iter()
            .chain(self.mtp.as_ref().map(|m| &m.layer))
            .collect();
        let q2 = |w: &W| w.dtype == DType::Q2K && !w.side;
        let dense = q2(&self.output) || layers.iter().any(|l| l.linears().iter().any(q2));
        let moe = |f: &dyn Fn(&[W; 3], Option<Cold>) -> bool| {
            layers.iter().any(|l| match l.ffn {
                Ffn::Moe {
                    gate,
                    up,
                    down,
                    cold,
                    ..
                } => f(&[gate, up, down], cold),
                Ffn::Dense { .. } => false,
            })
        };
        let routed = moe(&|w, _| w.iter().any(q2));
        let mixed = moe(&|_, c| c.is_some());
        if !(dense || routed || mixed) {
            return Ok(());
        }
        // Zeros: one Q2_K block at 0, the table (expert 0 at hot slot 0) at 128, one f16 row of x at
        // 256, its id (expert 0) at 768, the output at 1024.
        let ran = (|| {
            let b = Buffer::alloc(dev, 2048)?;
            s.upload(&b, 0, &[0u8; 2048])?;
            let w = Tensor::new(&b, 0, DType::Q2K, &[1, 256])?;
            let table = Tensor::new(&b, 128, DType::I32, &[1])?;
            let x = Tensor::new(&b, 256, DType::F16, &[1, 256])?;
            let ids = Tensor::new(&b, 768, DType::I32, &[1, 1])?;
            let out = Tensor::new(&b, 1024, DType::F16, &[1, 1, 1])?;
            if dense {
                ops::mmvq(
                    s,
                    &w,
                    &x,
                    None,
                    &Tensor::new(&b, 1024, DType::F16, &[1, 1])?,
                )?;
            }
            if routed {
                ops::mmvq_id(s, &w, &x, &ids, None, None, &out)?;
            }
            if mixed {
                ops::mmvq_id_mixed(s, &w, &w, &table, &x, &ids, None, &out)?;
            }
            s.sync()
        })();
        match ran {
            Err(KError::Unsupported) => Err(LoadError::UnsupportedQuant(
                gguf::ggml_type(DType::Q2K).unwrap_or(0),
            )),
            r => r.map_err(backend),
        }
    }

    /// Sets this model's choice of fused entries in place of `LARCH_FUSED`'s ([`larch_kernels::Routes::fused`]):
    /// `Some(false)` none, `None` those on by default, `Some(true)` all. For A/B runs in one process.
    pub fn set_fused(&mut self, fused: Option<bool>) {
        self.fused = fused;
    }

    /// Makes every fused entry answer UNSUPPORTED without calling the backend, so a test reaches each entry's
    /// fallback on the CPU devices, which never refuse one (the CUDA H2 route on sm_86/89 does).
    #[doc(hidden)]
    pub fn refuse_fused(&mut self, refuse: bool) {
        self.refuse_fused = refuse;
    }

    /// A fused entry's `call`, or UNSUPPORTED without it under [`Transformer::refuse_fused`].
    fn fused_call(&self, call: impl FnOnce() -> Result<(), KError>) -> Result<(), KError> {
        if self.refuse_fused {
            Err(KError::Unsupported)
        } else {
            call()
        }
    }

    /// Steps whose prefill chunk streamed its host experts, mixed steps too ([`Placement::stream_rows`]);
    /// 0 without a pool.
    pub fn streamed_steps(&self) -> u64 {
        self.stream.as_ref().map_or(0, |p| p.borrow().steps)
    }

    /// `layer`'s experts on the device, from the pool: each projection waits for its own tensor's
    /// copy, then syncs its readers before that slot is filled again. `Ok(false)` when the
    /// pool is off, and the layer takes the CPU path.
    #[allow(clippy::too_many_arguments)]
    fn stream_experts(
        &self,
        s: &Stream,
        layer: usize,
        experts: [W; 3],
        cold: Option<Cold>,
        m: MoeCfg,
        h: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
        out: &Tensor,
    ) -> Result<bool, KError> {
        let mut pool = self.stream.as_ref().ok_or(KError::InvalidArg)?.borrow_mut();
        let table = self.stream_tables.get(layer).and_then(Option::as_ref);
        let sc = [&self.sc.gate, &self.sc.up, &self.sc.act];
        let (rows, used, ef) = (h.shape()[0], i64::from(m.used), i64::from(m.ffn));
        let shape = [rows, used, ef];
        let g = Tensor::new(sc[0], 0, DType::F16, &shape)?;
        let u = Tensor::new(sc[1], 0, DType::F16, &shape)?;
        let act = Tensor::new(sc[2], 0, DType::F16, &shape)?;
        for (i, expert) in experts.into_iter().enumerate() {
            let Some(slot) = pool.take()? else {
                return Ok(false);
            };
            let (_, _, hot_at, cold_at) = pool.layers[layer][i];
            let ran = (|| {
                // The slot holds the span: the hot tensor at hot_at, a mixed layer's cold one at cold_at.
                let hot = W {
                    off: hot_at,
                    ..expert
                }
                .t(&slot)?;
                let w = match (cold, cold_at, table) {
                    (None, _, _) => Experts::One(hot),
                    (Some(c), Some(at), Some(table)) => Experts::Mixed {
                        hot,
                        cold: W {
                            off: at,
                            ..[c.gate, c.up, c.down][i]
                        }
                        .t(&slot)?,
                        table: Tensor::new(table, 0, DType::I32, &[table.bytes() as i64 / 4])?,
                    },
                    _ => return Err(KError::InvalidArg),
                };
                match i {
                    0 => mmvq_id(s, 0, &w, h, ids, None, &g)?,
                    1 => {
                        mmvq_id(s, 0, &w, h, ids, None, &u)?;
                        ops::silu_mul(
                            s,
                            &g.reshape(&[rows * used, ef])?,
                            &u.reshape(&[rows * used, ef])?,
                            &act.reshape(&[rows * used, ef])?,
                        )?;
                    }
                    _ => mmvq_id(s, 0, &w, &act, ids, Some(weights), out)?,
                }
                s.sync()
            })();
            pool.finish(layer * 3 + i, slot, ran)?;
        }
        if self.cache.is_some() {
            let mut v = vec![0u8; DType::I32.bytes((rows * used) as u64) as usize];
            s.download(ids.buffer(), ids.offset(), &mut v)?;
            self.count_host_routes(layer, m.experts as usize, &v);
        }
        Ok(true)
    }

    /// The cases of the projection routes at `rows` (the head's only up to 64, its most logits
    /// rows): each weight shape and row count where one route, on alone over the arch defaults,
    /// changes the first route that runs. A case runs every layer weight of its shape in turn, so
    /// each is read cold, as a forward pass reads it.
    pub fn route_cases(&self, rows: &[i64]) -> Vec<RouteCase> {
        let first = |w, rows, card: &[Route]| match self.picks(w, rows, card) {
            (Some(i8), _) => Some(Some(i8)),
            (None, quant_gemm) => quant_gemm.then_some(None),
        };
        let weights: Vec<W> = self.layers.iter().flat_map(Layer::linears).collect();
        let mut cases: Vec<RouteCase> = Vec::new();
        for w in weights.iter().chain([&self.output]).filter(|w| !w.side) {
            let shaped = |v: &&W| (v.dtype, v.shape) == (w.dtype, w.shape) && v.off != w.off;
            let ws: Vec<W> = weights.iter().filter(shaped).copied().collect();
            for &r in rows
                .iter()
                .filter(|&&r| r <= 64 || w.off != self.output.off)
            {
                for route in Route::ALL.into_iter().filter(|&r| r != Route::DecodeGraphs) {
                    let seen = cases.iter().any(|c| {
                        (c.route, c.rows, c.w.dtype, c.w.shape) == (route, r, w.dtype, w.shape)
                    });
                    if !seen && first(*w, r, &[]) != first(*w, r, &[route]) {
                        let ws = ws.clone();
                        cases.push(RouteCase {
                            route,
                            rows: r,
                            w: *w,
                            ws,
                        });
                    }
                }
            }
        }
        cases
    }

    /// `out = x · wᵀ` over `case`'s weights in turn, by the projection the model runs with the
    /// case's route on (over the arch defaults) or off; `out` ends with the last weight's.
    pub fn route_case(
        &mut self,
        s: &Stream,
        case: &RouteCase,
        on: bool,
        x: &Tensor,
        out: &Tensor,
    ) -> Result<(), KError> {
        let card = if on { vec![case.route] } else { Vec::new() };
        let card = std::mem::replace(&mut self.card, card);
        let ran = case
            .ws
            .iter()
            .chain([&case.w])
            .try_for_each(|&w| self.linear(s, w, None, x, out));
        self.card = card;
        ran
    }

    /// The CPU reference of `case`'s last weight on its output columns `cols`, for `x` (f16 bits,
    /// `[rows, k]`): f32 `[rows, cols]`. Dequant + gemm for the f16 tiles, the CPU's activation-int8
    /// op for the int8 routes: the oracles kernels/tests/quant_gemm.cpp checks them against.
    pub fn route_reference(
        &self,
        s: &Stream,
        case: &RouteCase,
        x: &[u16],
        cols: std::ops::Range<i64>,
    ) -> Result<Vec<f32>, KError> {
        let ([_, k], dtype) = (case.w.shape, case.w.dtype);
        let (n, rows) = (cols.end - cols.start, case.rows);
        let row = dtype.bytes(k as u64);
        let wt = self.t(case.w)?;
        let mut w = vec![0u8; (row * n as u64) as usize];
        s.download(wt.buffer(), wt.offset() + row * cols.start as u64, &mut w)?;
        let cpu = Device::open(larch_kernels::CPU)?;
        let cs = Stream::new(&cpu)?;
        let buf = |bytes: u64| Buffer::alloc(&cpu, bytes);
        let (wb, xb, ob) = (
            buf(w.len() as u64)?,
            buf(x.len() as u64 * 2)?,
            buf((rows * n * 4) as u64)?,
        );
        cs.upload(&wb, 0, &w)?;
        cs.upload(&xb, 0, x)?;
        let wt = Tensor::new(&wb, 0, dtype, &[n, k])?;
        let xt = Tensor::new(&xb, 0, DType::F16, &[rows, k])?;
        let ot = Tensor::new(&ob, 0, DType::F32, &[rows, n])?;
        if case.route == Route::QuantGemm {
            let db = buf((n * k * 2) as u64)?;
            let dq = Tensor::new(&db, 0, DType::F16, &[n, k])?;
            ops::dequant(&cs, &wt, &dq)?;
            ops::gemm(&cs, &xt, &dq, None, &ot, 1.0, 0.0)?;
        } else {
            ops::quant_gemm_i8(&cs, &wt, &xt, None, &ot)?;
        }
        let mut out = vec![0f32; (rows * n) as usize];
        cs.download(&ob, 0, &mut out)?;
        Ok(out)
    }

    /// The int8 tile and whether the f16 direct tiles take `w` at `rows` with `card`'s routes.
    fn picks(&self, w: W, rows: i64, card: &[Route]) -> (Option<I8>, bool) {
        let units = self.compute_units;
        let i8 = self
            .quant_i8
            .then(|| quant_i8_route(w, rows, units, self.arch_code, card))
            .flatten();
        let profiled = card.contains(&Route::QuantGemm);
        let quant_gemm = quant_gemm_device(self.backend, self.arch_code, profiled)
            && use_quant_gemm(w, rows, units);
        (i8, quant_gemm)
    }

    /// Runs the first direct quantized route that takes `w` at these rows: the int8 tile
    /// [`quant_i8_route`] picks, then f16 tiles. `Ok(false)` when none ran (each refuses what it
    /// cannot), so the caller falls back.
    fn direct(
        &self,
        s: &Stream,
        w: W,
        wt: &Tensor,
        x: &Tensor,
        b: Option<&Tensor>,
        out: &Tensor,
    ) -> Result<bool, KError> {
        type Op = fn(&Stream, &Tensor, &Tensor, Option<&Tensor>, &Tensor) -> Result<(), KError>;
        let rows = x.shape()[0];
        let (i8, quant_gemm) = self.picks(w, rows, &self.card);
        let routes: [(bool, Op); 3] = [
            (i8 == Some(I8::Split), ops::quant_gemm_i8_split),
            (i8 == Some(I8::Plain), ops::quant_gemm_i8),
            (quant_gemm, ops::quant_gemm),
        ];
        for (_, op) in routes.into_iter().filter(|r| r.0) {
            match op(s, wt, x, b, out) {
                Ok(()) => return Ok(true),
                Err(KError::Unsupported) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(false)
    }

    /// The dense feed-forward block of `h` into `out`, or onto the residual `resid` when given and
    /// [`Transformer::proj_add`] takes the down projection: then true.
    #[allow(clippy::too_many_arguments)]
    fn dense_ffn(
        &self,
        s: &Stream,
        gate: W,
        up: W,
        down: W,
        h: &Tensor,
        out: &Tensor,
        resid: Option<&Tensor>,
    ) -> Result<bool, KError> {
        let shape = [h.shape()[0], gate.shape[0]];
        let g = Tensor::new(&self.sc.gate, 0, DType::F16, &shape)?;
        let u = Tensor::new(&self.sc.up, 0, DType::F16, &shape)?;
        let act = Tensor::new(&self.sc.act, 0, DType::F16, &shape)?;
        self.linear(s, gate, None, h, &g)?;
        self.linear(s, up, None, h, &u)?;
        ops::silu_mul(s, &g, &u, &act)?;
        if let Some(x) = resid
            && self.proj_add(s, down, &act, None, x)?
        {
            return Ok(true);
        }
        self.linear(s, down, None, &act, out)?;
        Ok(false)
    }

    /// `w` of `input` (times sigmoid of `gate`, when given) added onto the residual `x` through `fused_proj_add`,
    /// where it is on and `linear` would take mmvq: false where it does not run, and the caller's ops then do.
    fn proj_add(
        &self,
        s: &Stream,
        w: W,
        input: &Tensor,
        gate: Option<&Tensor>,
        x: &Tensor,
    ) -> Result<bool, KError> {
        if !self.fusing(self.fused_on.proj_add) || !self.takes_mmvq(w, input.shape()[0]) {
            return Ok(false);
        }
        let w = self.t(w)?;
        match self.fused_call(|| ops::fused_proj_add(s, &w, input, gate, x)) {
            Err(KError::Unsupported) => Ok(false),
            r => r.map(|()| true),
        }
    }

    /// The routed experts of `h` into `out`, llama.cpp's qwen3moe block: softmax over the router's
    /// logits, the top k renormalised, and each row's chosen experts summed by those weights. Host
    /// experts run on the CPU device ([`Transformer::host_experts`], from the layer's own weights
    /// into the scratch t and x, so `out` and `resid` must be those at their start), or, while
    /// capturing, end the graph's piece there. The layer's expert adapters, if any, are added to
    /// the f32 residual `resid` directly. `routed_by`: the route `fused_moe_in` already made.
    // ponytail: every row runs its experts through mmvq_id; group prefill rows by expert into one
    // quant GEMM per expert when a long prompt's prefill shows this block.
    #[allow(clippy::too_many_arguments)]
    fn moe_ffn(
        &self,
        s: &Stream,
        layer: usize,
        router: W,
        experts: [W; 3],
        cold: Option<Cold>,
        host: bool,
        h: &Tensor,
        out: &Tensor,
        resid: &Tensor,
        routed_by: Option<(Tensor, Tensor)>,
    ) -> Result<(), KError> {
        let m = match (self.moe, &self.sc.route) {
            (Some(m), Some(_)) => m,
            _ => return Err(KError::InvalidArg),
        };
        let (rows, d) = (h.shape()[0], h.shape()[1]);
        let (e, k) = (i64::from(m.experts), i64::from(m.used));
        let ad = self.adapters.get(layer).and_then(Option::as_ref);
        let at = DType::F32.bytes((rows * e) as u64);
        let (ids, weights) = match routed_by {
            Some(r) => r,
            None => self.route(s, layer, router, host, h)?,
        };
        let delta = self.router_deltas.get(layer).and_then(Option::as_ref);
        if !host {
            let sc = [&self.sc.gate, &self.sc.up, &self.sc.act];
            let table = self.cold_tables.get(layer).and_then(Option::as_ref);
            let experts = Experts::of(experts, cold, table, |w| self.t(w))?;
            routed(s, self.check_rows(), experts, sc, m, h, &ids, &weights, out)?;
            if let Some(ad) = ad {
                let act = [sc[2], sc[0], sc[1]];
                let y = adapt(s, self.check_rows(), ad, act, m, &ids, &weights, rows, d)?;
                ops::add(s, resid, &y, resid)?;
            }
            return Ok(());
        }
        // `streaming` is the last pass's: the MTP block (layer `layers.len()`) runs outside any pass
        // and has no plan in the pool, where it would wait on copies nothing queued. Beside decode
        // rows only the prefill rows stream; the decode rows run on the CPU path below, the bits a
        // decode step gives them (m13 W4@4 on the 5070 Ti: 578 ms such a step all on the CPU path).
        let mut live = rows;
        if self.streaming && layer < self.layers.len() {
            let nd = self.stream_decode;
            let n = rows - nd;
            let (h, i) = (h.rows(nd, n)?, ids.rows(nd, n)?);
            let (w, o) = (weights.rows(nd, n)?, out.rows(nd, n)?);
            if self.stream_experts(s, layer, experts, cold, m, &h, &i, &w, &o)? {
                if nd == 0 {
                    return Ok(());
                }
                live = nd;
            }
        }
        // Dynamic residency's uploads map in the next pass, which a replay never makes.
        if s.capturing()
            && self
                .split_cache()
                .is_some_and(|c| c.borrow().copy.is_some())
        {
            return Err(KError::Unsupported);
        }
        // forward put h right after the ids and weights, so one download, and one device sync per
        // host layer, reads all three: each sync costs a device round trip on top of the work.
        let start = |t: &Tensor, b: &Buffer| std::ptr::eq(t.buffer(), b) && t.offset() == 0;
        if h.offset() != at + DType::F32.bytes((2 * rows * k) as u64)
            || !start(out, &self.sc.t)
            || !start(resid, &self.sc.x)
        {
            return Err(KError::InvalidArg);
        }
        if !s.capturing() {
            return self.host_experts(s, layer, rows, live, self.check.get(), true);
        }
        // A graph would keep the version's buffers past the swap that frees them.
        if ad.is_some() || delta.is_some() {
            return Err(KError::Unsupported);
        }
        // The section ends this piece; the replay runs it by tag before the next.
        s.cut(host_tag(layer, rows, self.check.get()))
    }

    /// The router's logits of `h` (plus the layer's deltas) and their top k into the route scratch: the chosen ids
    /// (a resident layer's into its rows of the route log while recording) and their weights.
    fn route(
        &self,
        s: &Stream,
        layer: usize,
        router: W,
        host: bool,
        h: &Tensor,
    ) -> Result<(Tensor<'_>, Tensor<'_>), KError> {
        let [logits, ids, weights] = self.route_tensors(s, layer, host, h.shape()[0])?;
        let e = logits.shape()[1];
        let delta = self.router_deltas.get(layer).and_then(Option::as_ref);
        let delta = delta
            .map(|b| Tensor::new(b, 0, DType::F32, &[e]))
            .transpose()?;
        self.gemm(s, h, &self.t(router)?, delta.as_ref(), &logits)?;
        ops::moe_route(s, &logits, 0.0, &ids, &weights)?;
        Ok((ids, weights))
    }

    /// Where a route of `rows` rows goes in the route scratch: the router's logits, the chosen ids (a resident
    /// layer's into its rows of the route log while recording) and their weights.
    fn route_tensors(
        &self,
        s: &Stream,
        layer: usize,
        host: bool,
        rows: i64,
    ) -> Result<[Tensor<'_>; 3], KError> {
        let (m, route) = match (self.moe, &self.sc.route) {
            (Some(m), Some(route)) => (m, route),
            _ => return Err(KError::InvalidArg),
        };
        let (e, k) = (i64::from(m.experts), i64::from(m.used));
        let logits = Tensor::new(route, 0, DType::F32, &[rows, e])?;
        let at = DType::F32.bytes((rows * e) as u64);
        let (ids, weights) = route_of(route, at, rows, k)?;
        // While recording, a resident layer's ids go to its rows of the log, read after the step.
        // The MTP block past the trunk has no rows in the log.
        let trunk = layer < self.layers.len();
        let ids = match (self.route_log.as_ref()).filter(|_| !host && trunk && !s.capturing()) {
            Some(b) => {
                let at = DType::I32.bytes((layer as i64 * rows * k) as u64);
                Tensor::new(b, at, DType::I32, &[rows, k])?
            }
            None => ids,
        };
        Ok([logits, ids, weights])
    }

    /// The MoE block's inputs through `fused_moe_in`: the norm of the residual `x` into `h`, the route, and the
    /// shared expert's activation (in t) and gate ([`Transformer::shared_in`]) with its down weight; refused, the
    /// same through their ops.
    #[allow(clippy::too_many_arguments)]
    fn fused_moe_in(
        &self,
        s: &Stream,
        layer: usize,
        x: &Tensor,
        norm: W,
        h: &Tensor,
        router: W,
        host: bool,
        shared: Option<qwen35::Shared>,
    ) -> Result<MoeIn<'_>, KError> {
        let rows = h.shape()[0];
        let [logits, ids, weights] = self.route_tensors(s, layer, host, rows)?;
        let (norm, router_t) = (self.t(norm)?, self.t(router)?);
        let sh = shared
            .map(|sh| {
                let (act, sg) = self.shared_outs(&sh, rows)?;
                let w = [self.t(sh.gate)?, self.t(sh.up)?, self.t(sh.gate_inp)?];
                Ok::<_, KError>((sh, act, sg, w))
            })
            .transpose()?;
        let five = sh
            .as_ref()
            .map(|(_, act, sg, [g, u, i])| [g, u, act, i, sg]);
        let out = [&logits, &ids, &weights];
        let eps = self.cfg.rms_eps;
        match self.fused_call(|| ops::fused_moe_in(s, x, &norm, eps, h, &router_t, out, five)) {
            Err(KError::Unsupported) => {
                ops::rmsnorm(s, x, &norm, self.cfg.rms_eps, h)?;
                self.route(s, layer, router, host, h)?;
                if let Some((sh, ..)) = &sh {
                    self.shared_in(s, sh, h)?;
                }
            }
            r => r?,
        }
        let sh = sh.map(|(sh, act, sg, _)| (sh.down, act, sg));
        Ok((ids, weights, sh))
    }

    /// Whether layer `layer`'s MoE inputs at `rows` rows take [`Transformer::fused_moe_in`]: outside check steps,
    /// without router deltas, with the shared expert's gate and up on mmvq and its activation fitting t (and its down
    /// projection's output gate, where `fused_moe_down` does not run).
    fn moe_in_fusable(&self, layer: usize, shared: Option<&qwen35::Shared>, rows: i64) -> bool {
        self.fusing(self.fused_on.moe_in)
            && !self.check.get()
            && self.router_deltas.get(layer).is_none_or(Option::is_none)
            && shared.is_none_or(|sh| self.shared_fits(sh, rows))
    }

    /// The shared expert's gate and up run on mmvq at `rows` rows, its activation fits t and its down projection's
    /// output fits gate ([`Transformer::gate_holds_d`]), as the fused MoE path places them.
    fn shared_fits(&self, sh: &qwen35::Shared, rows: i64) -> bool {
        let d = i64::from(self.cfg.d_model);
        self.takes_mmvq(sh.gate, rows)
            && self.takes_mmvq(sh.up, rows)
            && sh.gate.shape[0] <= 2 * d
            && self.gate_holds_d()
    }

    /// The scratch gate (`[rows, max(used * expert ffn, shared ffn)]`) holds a `[rows, d]` output: where the fused
    /// MoE path's refused down projections write.
    fn gate_holds_d(&self) -> bool {
        let shared = self.hybrid.as_ref().map_or(0, |h| i64::from(h.shared_ffn));
        self.moe.is_some_and(|m| {
            (i64::from(m.used) * i64::from(m.ffn)).max(shared) >= i64::from(self.cfg.d_model)
        })
    }

    /// Whether a fused entry runs: never under `LARCH_FUSED=0`, always under `1`, else `default`
    /// (each entry's default on this device, see [`fused_defaults`]).
    fn fusing(&self, default: bool) -> bool {
        self.fused.unwrap_or(default)
    }

    /// `linear` of `w` at `rows` rows takes `ops::mmvq` without trying a direct route (`linear` asks this first), so
    /// the fused entries, which give mmvq's bytes, stand in for it: a quantized weight outside a check step and every
    /// direct route.
    fn takes_mmvq(&self, w: W, rows: i64) -> bool {
        let (i8, quant_gemm) = self.picks(w, rows, &self.card);
        !w.side && !self.check.get() && rows <= MMVQ_SLICE_ROWS && i8.is_none() && !quant_gemm
    }

    /// The MoE block of a resident layer on the fused entries, onto the residual `x`: the shared expert's gate,
    /// up and gate input ([`Transformer::shared_in`]) and the route, or those `made` by `fused_moe_in`, then
    /// `fused_moe_gate_up` and `fused_moe_down`.
    /// A fused call the backend refuses runs as its ops, whose scratch (gate, up, act and t) this path shares:
    /// the shared expert's activation waits in t, so a refused down writes gate. `None` where the block
    /// cannot take this path ([`Transformer::moe_fusable`]).
    #[allow(clippy::too_many_arguments)]
    fn fused_moe(
        &self,
        s: &Stream,
        layer: usize,
        router: W,
        experts: [W; 3],
        shared: Option<qwen35::Shared>,
        h: &Tensor,
        x: &Tensor,
        made: Option<MoeIn>,
    ) -> Result<(), KError> {
        let m = self.moe.ok_or(KError::InvalidArg)?;
        let (rows, d) = (h.shape()[0], h.shape()[1]);
        let (k, ef) = (i64::from(m.used), i64::from(m.ffn));
        let (ids, weights, sh) = match made {
            Some(made) => made,
            None => {
                let sh = shared.map(|sh| self.shared_in(s, &sh, h)).transpose()?;
                let (ids, weights) = self.route(s, layer, router, false, h)?;
                (ids, weights, sh)
            }
        };
        let [gate, up, down] = experts.map(|w| self.t(w));
        let (gate, up, down) = (gate?, up?, down?);
        let act = Tensor::new(&self.sc.act, 0, DType::F16, &[rows, k, ef])?;
        match self.fused_call(|| ops::fused_moe_gate_up(s, &gate, &up, h, &ids, &act)) {
            Err(KError::Unsupported) => {
                let sc = [&self.sc.gate, &self.sc.up, &self.sc.act];
                routed_act(
                    s,
                    0,
                    [&Experts::One(gate), &Experts::One(up)],
                    sc,
                    m,
                    h,
                    &ids,
                )?;
            }
            r => r?,
        }
        let sd = sh.as_ref().map(|(w, a, g)| (self.t(*w), a, g));
        let sd = match sd {
            Some((w, a, g)) => Some((w?, a, g)),
            None => None,
        };
        let three = sd.as_ref().map(|(w, a, g)| [w, *a, *g]);
        match self.fused_call(|| ops::fused_moe_down(s, &down, &act, &ids, &weights, three, x)) {
            Err(KError::Unsupported) => {
                if let Some(sh) = &sh {
                    self.shared_down_add(s, sh, x)?;
                }
                let t = Tensor::new(&self.sc.gate, 0, DType::F16, &[rows, d])?;
                mmvq_id(s, 0, &Experts::One(down), &act, &ids, Some(&weights), &t)?;
                ops::add(s, x, &t, x)
            }
            r => r,
        }
    }

    /// The shared expert's down projection of its activation (in t) through the scratch gate, times its gate, added
    /// onto `x`: [`Transformer::shared_ffn`]'s end and the residual add, on the fused path's placement.
    fn shared_down_add(
        &self,
        s: &Stream,
        (w, act, sg): &(W, Tensor, Tensor),
        x: &Tensor,
    ) -> Result<(), KError> {
        let y = Tensor::new(&self.sc.gate, 0, DType::F16, &[x.shape()[0], x.shape()[1]])?;
        self.linear(s, *w, None, act, &y)?;
        ops::sigmoid_mul(s, &y, sg, &y)?;
        ops::add(s, x, &y, x)
    }

    /// Whether layer `layer`'s MoE block at `rows` rows takes [`Transformer::fused_moe`]: a resident layer, no
    /// adapters or router deltas, every projection on mmvq, and room for the shared expert's activation in t (and
    /// for its refused down's output in gate).
    fn moe_fusable(
        &self,
        layer: usize,
        host: bool,
        shared: Option<&qwen35::Shared>,
        rows: i64,
    ) -> bool {
        let shared_ok =
            shared.is_none_or(|sh| self.shared_fits(sh, rows) && self.takes_mmvq(sh.down, rows));
        self.fusing(self.fused_on.moe)
            && !host
            && !self.check.get()
            && rows < FUSED_ID_ROWS
            && self.gate_holds_d()
            && shared_ok
            && self.adapters.get(layer).is_none_or(Option::is_none)
            && self.router_deltas.get(layer).is_none_or(Option::is_none)
    }

    /// The host experts of `layer` in a step of `rows` rows, the section between moe_route and the
    /// add after it, run eagerly and by a replay's host step alike, on the first `live` rows: the
    /// route and `h` that forward left in the route scratch go to the CPU device, its experts'
    /// output into the scratch t (`out`), and the layer's adapters, if any, into the residual x.
    /// `count`: the recording counts this section's routes, on eager passes and an all-split
    /// load's replays. Any other replay would count host layers but not the resident ones, whose
    /// log no graph writes.
    // ponytail: the device idles while the CPU runs; overlap them, or on unified memory read the
    // device's buffers in place, when a profile shows it.
    // ponytail: replayed decode steps go uncounted on both kinds of layer unless all are split, so a cold histogram is
    // prefill and eager steps; add a captured log copy if its plan's hit rate trails a replay-free one.
    fn host_experts(
        &self,
        s: &Stream,
        layer: usize,
        rows: i64,
        live: i64,
        check: bool,
        count: bool,
    ) -> Result<(), KError> {
        let (Some(m), Some(route), Some(x)) = (self.moe, &self.sc.route, &self.host) else {
            return Err(KError::InvalidArg);
        };
        // The host experts run on the CPU device, which groups from 4 pairs an expert.
        let host_rows = if check { ID_ROWS } else { 0 };
        let Some(Ffn::Moe {
            gate,
            up,
            down,
            host: true,
            cold,
            ..
        }) = (self.layers.get(layer))
            .or_else(|| {
                (self.mtp.as_ref())
                    .filter(|_| layer == self.layers.len())
                    .map(|m| &m.layer)
            })
            .map(|l| l.ffn)
        else {
            return Err(KError::InvalidArg);
        };
        if rows > self.max_rows || !(1..=rows).contains(&live) {
            return Err(KError::InvalidArg);
        }
        // The MTP block past the trunk has no row in the histogram.
        let count = count && layer < self.layers.len();
        if let Some(c) = self.split_cache() {
            let c = &mut *c.borrow_mut();
            return self.split_ffn(s, c, x, layer, m, rows, live, check, count);
        }
        let (d, e, k) = (
            i64::from(self.cfg.d_model),
            i64::from(m.experts),
            i64::from(m.used),
        );
        let ad = self.adapters.get(layer).and_then(Option::as_ref);
        let at = DType::F32.bytes((rows * e) as u64);
        let rk = DType::F32.bytes((2 * rows * k) as u64);
        let mut staging = x
            .staging
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let v = staging
            .get_mut(..(rk + DType::F16.bytes((rows * d) as u64)) as usize)
            .ok_or(KError::InvalidArg)?;
        s.download(route, at, v)?;
        // The live rows' ids, weights and h lead their sections of the step's route.
        let (rv, hv) = v.split_at_mut(rk as usize);
        let (ids, weights) = rv.split_at(rk as usize / 2);
        let lk = DType::F32.bytes((live * k) as u64);
        if count {
            self.count_host_routes(layer, e as usize, &ids[..lk as usize]);
        }
        let hv = &mut hv[..DType::F16.bytes((live * d) as u64) as usize];
        x.s.upload(&x.route, 0, &ids[..lk as usize])?;
        x.s.upload(&x.route, lk, &weights[..lk as usize])?;
        x.s.upload(&x.h, 0, hv)?;
        let xh = Tensor::new(&x.h, 0, DType::F16, &[live, d])?;
        let (xi, xw) = route_of(&x.route, 0, live, k)?;
        let xo = Tensor::new(&x.out, 0, DType::F16, &[live, d])?;
        let sc = [&x.gate, &x.up, &x.act];
        let table = self.cold_tables.get(layer).and_then(Option::as_ref);
        let experts = Experts::of([gate, up, down], cold, table, |w| w.t(&x.weights))?;
        host_routed(&x.s, host_rows, experts, sc, m, &xh, &xi, &xw, &xo)?;
        let act = [&x.act, &x.gate, &x.up];
        let y = ad
            .map(|ad| adapt(&x.s, host_rows, ad, act, m, &xi, &xw, live, d))
            .transpose()?;
        x.s.download(&x.out, 0, hv)?;
        s.upload(&self.sc.t, 0, hv)?;
        if let Some(y) = y {
            // `y` crosses like the experts' output, into the device's free up scratch.
            x.s.download(y.buffer(), 0, hv)?;
            s.upload(&self.sc.up, 0, hv)?;
            let y = Tensor::new(&self.sc.up, 0, DType::F16, &[live, d])?;
            let resid = Tensor::new(&self.sc.x, 0, DType::F32, &[live, d])?;
            ops::add(s, &resid, &y, &resid)?;
        }
        Ok(())
    }

    /// While recording, `layer`'s routed `ids` (little-endian i32) from a host section's download.
    fn count_host_routes(&self, layer: usize, e: usize, ids: &[u8]) {
        let Some(c) = &self.cache else {
            return;
        };
        let c = &mut *c.borrow_mut();
        // With no resident layer to wait for, nothing arrives later than these.
        let counts = match self.route_log {
            Some(_) => &mut c.staged,
            None => &mut c.hist.counts,
        };
        let counts = &mut counts[layer * e..][..e];
        for b in ids.chunks_exact(4) {
            let i = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
            if let Some(n) = counts.get_mut(i) {
                *n = n.saturating_add(1);
            }
        }
    }

    /// A split layer's routed experts of a step of `rows` rows, the first `live` of them, after
    /// [`moe_ffn`](Self::moe_ffn)'s route, run eagerly and by a replay's host step alike: its
    /// device part has a shape per step, so no graph holds it. One download reads the ids, weights
    /// and `h` from the route scratch; each (row, slot) runs where its expert is. The device's
    /// part (mmvq_id over the layer's slots, the ids remapped and the host experts' weights 0)
    /// joins the residual x here when a host part follows; the host's part (one row per (row,
    /// slot), its f32 outputs summed per row in slot order and rounded to f16 once) is then the
    /// output, the scratch t. A row's bytes depend on the residency only: a part that has none of
    /// its experts adds an exact 0, and the host part runs per pair at any batch ([`HOST_ROWS`]).
    /// `count` is [`host_experts`](Self::host_experts)'.
    // ponytail: Metal's mmvq_id groups from 4 pairs per slot of the compact table, so its device
    // part can round otherwise than the row alone (and a check step's 3-row slices stop holding
    // for 3 to 6 slots); one row at a time there when a Metal check step reports a mismatch.
    #[allow(clippy::too_many_arguments)]
    fn split_ffn(
        &self,
        s: &Stream,
        c: &mut ExpertCache,
        x: &HostExperts,
        layer: usize,
        m: MoeCfg,
        rows: i64,
        live: i64,
        check: bool,
        count: bool,
    ) -> Result<(), KError> {
        let route = self.sc.route.as_ref().ok_or(KError::InvalidArg)?;
        let d = i64::from(self.cfg.d_model);
        // The host experts run on the CPU device, which groups from 4 pairs an expert.
        let host_rows = if check { ID_ROWS } else { 0 };
        let (e, k) = (m.experts as usize, m.used as usize);
        let at = DType::F32.bytes((rows * i64::from(m.experts)) as u64);
        let (all_pairs, n) = (rows as usize * k, live as usize * k);
        let h = Tensor::new(route, at + 8 * all_pairs as u64, DType::F16, &[rows, d])?;
        let out = Tensor::new(&self.sc.t, 0, DType::F16, &[live, d])?;
        let resid = Tensor::new(&self.sc.x, 0, DType::F32, &[live, d])?;
        let mut staging = x
            .staging
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let v = staging
            .get_mut(..8 * all_pairs + 2 * rows as usize * d as usize)
            .ok_or(KError::InvalidArg)?;
        s.download(route, at, v)?;
        let (ids, rest) = v.split_at(4 * all_pairs);
        let (ws, hv) = rest.split_at(4 * all_pairs);
        let ids: Vec<usize> = ids[..4 * n]
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
            .collect();
        let ws: Vec<f32> = ws[..4 * n]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if ids.iter().any(|&i| i >= e) {
            return Err(KError::InvalidArg);
        }
        if count {
            let counts = &mut c.hist.counts[layer * e..][..e];
            for &i in &ids {
                counts[i] = counts[i].saturating_add(1);
            }
        }
        // The compute stream is idle up to here, so no queued op reads this layer's slots.
        c.upload_hot(x, layer, &ids)?;
        let sp = c.splits[layer].as_ref().ok_or(KError::InvalidArg)?;
        // The device runs only the rows with a resident expert, gathered to the front of `h`;
        // a host slot of such a row runs as a weight-0 copy of the row's first resident slot. So
        // no row runs an expert it did not route to (whose overflow would make 0 times it NaN),
        // and no id names a slot an upload is filling.
        // ponytail: k mmvq_id slots per device row whatever its hits; compact them when a
        // profile shows the weight-0 slots.
        let (mut dids, mut dws) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut miss, mut on) = (Vec::new(), Vec::new());
        for (r, (ids, ws)) in ids.chunks_exact(k).zip(ws.chunks_exact(k)).enumerate() {
            let first = ids.iter().map(|&i| sp.slot[i]).find(|&ds| ds >= 0);
            for (&i, &w) in ids.iter().zip(ws) {
                if sp.slot[i] >= 0 {
                    dids.push(sp.slot[i]);
                    dws.push(w);
                } else {
                    miss.push((r, sp.hslot[i], w));
                    if let Some(f) = first {
                        dids.push(f);
                        dws.push(0f32);
                    }
                }
            }
            on.extend(first.map(|_| r));
        }
        c.hits.0 += (n - miss.len()) as u64;
        c.hits.1 += miss.len() as u64;
        c.split_steps += 1;
        let row = 2 * d as usize;
        let gather = |rows: &mut dyn Iterator<Item = usize>| -> Vec<u8> {
            rows.flat_map(|r| &hv[r * row..][..row]).copied().collect()
        };
        if !on.is_empty() {
            let (nd, all) = (on.len() as i64, on.len() == live as usize);
            if !all {
                let hg = gather(&mut on.iter().copied());
                s.upload(h.buffer(), h.offset(), &hg)?;
            }
            let (ids, weights) = route_of(route, at, nd, k as i64)?;
            s.upload(route, ids.offset(), &dids)?;
            s.upload(route, weights.offset(), &dws)?;
            let dev = c.dev.as_ref().ok_or(KError::InvalidArg)?;
            let [g, u, dn] = sp.dev.map(|w| w.t(dev));
            let sc = [&self.sc.gate, &self.sc.up, &self.sc.act];
            let (hd, od) = (h.rows(0, nd)?, out.rows(0, nd)?);
            let id_rows = if check { self.id_rows } else { 0 };
            let experts = [Experts::One(g?), Experts::One(u?), Experts::One(dn?)];
            routed(s, id_rows, experts, sc, m, &hd, &ids, &weights, &od)?;
            if !all {
                // Back to their rows; a row without a resident expert gets an exact 0.
                let dw = d as usize;
                let mut part = vec![0u16; on.len() * dw];
                s.download(out.buffer(), out.offset(), &mut part)?;
                let mut full = vec![0u16; live as usize * dw];
                for (&r, p) in on.iter().zip(part.chunks_exact(dw)) {
                    full[r * dw..][..dw].copy_from_slice(p);
                }
                s.upload(out.buffer(), out.offset(), &full)?;
            }
            if miss.is_empty() {
                c.no_host += 1;
                return Ok(());
            }
            ops::add(s, &resid, &out, &resid)?;
        }
        let nm = miss.len() as i64;
        x.s.upload(&x.h, 0, &gather(&mut miss.iter().map(|m| m.0)))?;
        x.s.upload(&x.route, 0, &miss.iter().map(|m| m.1).collect::<Vec<_>>())?;
        let mw: Vec<f32> = miss.iter().map(|m| m.2).collect();
        x.s.upload(&x.route, DType::I32.bytes(nm as u64), &mw)?;
        let xh = Tensor::new(&x.h, 0, DType::F16, &[nm, d])?;
        let (xi, xw) = route_of(&x.route, 0, nm, 1)?;
        let xo = Tensor::new(&x.out, 0, DType::F32, &[nm, d])?;
        let [g, u, dn] = sp.host.map(|w| w.t(&x.weights));
        let (g, u, dn) = (g?, u?, dn?);
        let sc = [&x.gate, &x.up, &x.act];
        for at in (0..nm).step_by(HOST_ROWS as usize) {
            let n = HOST_ROWS.min(nm - at);
            let (xh, xi, xw, xo) = (
                xh.rows(at, n)?,
                xi.rows(at, n)?,
                xw.rows(at, n)?,
                xo.rows(at, n)?,
            );
            let experts = [Experts::One(g), Experts::One(u), Experts::One(dn)];
            host_routed(&x.s, host_rows, experts, sc, m, &xh, &xi, &xw, &xo)?;
        }
        let mut y = vec![0f32; miss.len() * d as usize];
        x.s.download(&x.out, 0, &mut y)?;
        let mut sum = vec![0f32; live as usize * d as usize];
        for (&(r, ..), y) in miss.iter().zip(y.chunks_exact(d as usize)) {
            for (a, b) in sum[r * d as usize..].iter_mut().zip(y) {
                *a += b;
            }
        }
        let half: Vec<u16> = sum.into_iter().map(f16_bits).collect();
        s.upload(out.buffer(), out.offset(), &half)
    }

    /// The host embedding's rows of `ids` (a step's token ids, in row order) into the scratch x,
    /// gathered on the CPU device and uploaded at once; nothing for a device embedding. An id past
    /// the table is `InvalidArg` (the CPU device checks it).
    fn host_embed(&self, s: &Stream, ids: &[i32]) -> Result<(), KError> {
        let Some(e) = &self.embd else {
            return Ok(());
        };
        let (n, d) = (ids.len() as i64, i64::from(self.cfg.d_model));
        if n > self.max_rows {
            return Err(KError::InvalidArg);
        }
        e.s.upload(&e.ids, 0, ids)?;
        let it = Tensor::new(&e.ids, 0, DType::I32, &[n])?;
        let out = Tensor::new(&e.out, 0, DType::F32, &[n, d])?;
        ops::embed(&e.s, &e.w.t(&e.table)?, &it, &out)?;
        let mut x = e.x.borrow_mut();
        let x = &mut x[..(n * d) as usize];
        e.s.download(&e.out, 0, x)?;
        s.upload(&self.sc.x, 0, x)
    }

    /// The token embedding on the device; `InvalidArg` when it stays in host memory.
    fn embed_table(&self) -> Result<Tensor<'_>, KError> {
        self.t(self.tok_embd.ok_or(KError::InvalidArg)?)
    }

    /// Weight `w` in the buffer that holds it.
    fn t(&self, w: W) -> Result<Tensor<'_>, KError> {
        if w.side {
            return w.t(self.side.as_ref().ok_or(KError::InvalidArg)?);
        }
        match &self.hi {
            Some((b, at)) if w.off >= *at => W {
                off: w.off - at,
                ..w
            }
            .t(b),
            _ => w.t(&self.weights),
        }
    }

    /// `out = x · wᵀ (+ b)`: mmvq through the backend/weight ceiling, else dequant to f16 and gemm;
    /// gemm alone for an f16 side weight.
    fn linear(
        &self,
        s: &Stream,
        w: W,
        b: Option<W>,
        x: &Tensor,
        out: &Tensor,
    ) -> Result<(), KError> {
        let wt = self.t(w)?;
        let bt = b.map(|b| self.t(b)).transpose()?;
        if w.side {
            return self.gemm(s, x, &wt, bt.as_ref(), out);
        }
        if self.check.get() {
            return sliced(VERIFY_ROWS, x, out, |_, x, o| {
                ops::mmvq_verify(s, &wt, x, bt.as_ref(), o)
            });
        }
        let rows = x.shape()[0];
        // takes_mmvq decides the fused entries' route too, so both reproduce the same op.
        if !self.takes_mmvq(w, rows) && self.direct(s, w, &wt, x, bt.as_ref(), out)? {
            return Ok(());
        }
        if rows <= MMVQ_SLICE_ROWS {
            return ops::mmvq(s, &wt, x, bt.as_ref(), out);
        }
        if use_mmvq(self.mmvq_max_rows, w, rows) {
            // Each slice passes its own shape check; also check the complete output before slicing.
            if rows != out.shape()[0] {
                return Err(KError::InvalidArg);
            }
            for at in (0..rows).step_by(MMVQ_SLICE_ROWS as usize) {
                let n = (rows - at).min(MMVQ_SLICE_ROWS);
                ops::mmvq(s, &wt, &x.rows(at, n)?, bt.as_ref(), &out.rows(at, n)?)?;
            }
            return Ok(());
        }
        let dq = Tensor::new(
            self.sc.dq.as_ref().ok_or(KError::InvalidArg)?,
            0,
            DType::F16,
            &w.shape,
        )?;
        ops::dequant(s, &wt, &dq)?;
        ops::gemm(s, x, &dq, bt.as_ref(), out, 1.0, 0.0)
    }

    /// Rows a check step's mmvq_id takes at a time on the device, 0 outside a check step.
    fn check_rows(&self) -> i64 {
        if self.check.get() { self.id_rows } else { 0 }
    }

    /// `out = x · wᵀ (+ b)` for an f16 `w`; a check step's in slices of [`GEMV_ROWS`].
    fn gemm(
        &self,
        s: &Stream,
        x: &Tensor,
        w: &Tensor,
        b: Option<&Tensor>,
        out: &Tensor,
    ) -> Result<(), KError> {
        if self.check.get() {
            return sliced(GEMV_ROWS, x, out, |_, x, o| {
                ops::gemm(s, x, w, b, o, 1.0, 0.0)
            });
        }
        ops::gemm(s, x, w, b, out, 1.0, 0.0)
    }
}

/// `op` over `x` and `out` in slices of at most `n` rows, given each slice's first row.
fn sliced(
    n: i64,
    x: &Tensor,
    out: &Tensor,
    op: impl Fn(i64, &Tensor, &Tensor) -> Result<(), KError>,
) -> Result<(), KError> {
    let rows = x.shape()[0];
    if rows != out.shape()[0] {
        return Err(KError::InvalidArg);
    }
    for at in (0..rows).step_by(n as usize) {
        let m = (rows - at).min(n);
        op(at, &x.rows(at, m)?, &out.rows(at, m)?)?;
    }
    Ok(())
}

/// A qwen3moe layer's weights as the router trainer ([`crate::learn::DeviceSuffix`]) reads them.
pub(crate) struct TrainLayer<'a> {
    pub attn_norm: Tensor<'a>,
    pub wq: Tensor<'a>,
    pub wk: Tensor<'a>,
    pub wv: Tensor<'a>,
    pub q_norm: Tensor<'a>,
    pub k_norm: Tensor<'a>,
    pub wo: Tensor<'a>,
    pub ffn_norm: Tensor<'a>,
    /// f32 or f16 `[experts, d]`, as the GGUF stores it.
    pub router: Tensor<'a>,
    /// `[experts * N, k]` each.
    pub gate: Tensor<'a>,
    pub up: Tensor<'a>,
    pub down: Tensor<'a>,
}

impl Transformer {
    /// ggml_rope_ext's parameters for this model, as llama.cpp sets them.
    pub(crate) fn rope_params(&self) -> Result<ops::RopeParams<'_>, KError> {
        let r = &self.cfg.rope;
        Ok(ops::RopeParams {
            n_dims: r.n_dims,
            neox: r.neox,
            n_ctx_orig: r.n_ctx_orig,
            freq_base: r.freq_base,
            freq_scale: r.freq_scale,
            ext_factor: if r.yarn { 1.0 } else { 0.0 },
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            freq_factors: self.rope_freqs.map(|w| self.t(w)).transpose()?,
        })
    }

    /// Layers `first..` of a qwen3moe model (None otherwise, under the expert cache, or when one of
    /// them keeps its experts in host memory or is mixed), the final norm and the head.
    #[allow(clippy::type_complexity)]
    pub(crate) fn train_view(
        &self,
        first: u16,
    ) -> Result<Option<(MoeCfg, Vec<TrainLayer<'_>>, Tensor<'_>, Tensor<'_>)>, KError> {
        let Some(moe) = self.moe.filter(|_| self.cfg.arch == Arch::Qwen3Moe) else {
            return Ok(None);
        };
        if self.split_cache().is_some() {
            eprintln!("{NO_LEARNING}");
            return Ok(None);
        }
        let mut out = Vec::new();
        for l in self.layers.iter().skip(usize::from(first)) {
            let (
                Ffn::Moe {
                    raw: Some(raw),
                    gate,
                    up,
                    down,
                    host: false,
                    cold: None,
                    ..
                },
                Mixer::Attn(Attn {
                    wq,
                    wk,
                    wv,
                    q_norm: Some(qn),
                    k_norm: Some(kn),
                    wo,
                    ..
                }),
            ) = (l.ffn, &l.mixer)
            else {
                return Ok(None);
            };
            out.push(TrainLayer {
                attn_norm: self.t(l.attn_norm)?,
                wq: self.t(*wq)?,
                wk: self.t(*wk)?,
                wv: self.t(*wv)?,
                q_norm: self.t(*qn)?,
                k_norm: self.t(*kn)?,
                wo: self.t(*wo)?,
                ffn_norm: self.t(l.ffn_norm)?,
                router: self.t(raw)?,
                gate: self.t(gate)?,
                up: self.t(up)?,
                down: self.t(down)?,
            });
        }
        Ok(Some((
            moe,
            out,
            self.t(self.out_norm)?,
            self.t(self.output)?,
        )))
    }
}

impl Model for Transformer {
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    fn transformer(&self) -> Option<&Transformer> {
        Some(self)
    }

    fn forward(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        self.pass(s, b, None, kv, logits, None, &[])
    }

    fn forward_taps(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
        taps: Option<&Taps>,
    ) -> Result<(), KError> {
        self.pass(s, b, None, kv, logits, taps, &[])
    }

    fn state_slots(&self) -> u32 {
        self.states.as_ref().map_or(0, qwen35::States::slots)
    }

    fn reset_state(&mut self, s: &Stream, slot: u32) -> Result<(), KError> {
        self.zero_slot(s, slot)
    }

    fn copy_state(&mut self, s: &Stream, from: u32, to: u32) -> Result<(), KError> {
        self.copy_slot(s, from, to)
    }

    fn state_size(&self) -> usize {
        Transformer::state_size(self)
    }

    fn save_state(&self, s: &Stream, slot: u32, out: &mut [u8]) -> Result<(), KError> {
        self.read_state(s, slot, out)
    }

    fn load_state(&mut self, s: &Stream, slot: u32, bytes: &[u8]) -> Result<(), KError> {
        self.set_state_bytes(s, slot, bytes)
    }

    fn grow_state(&mut self, s: &Stream, want: u32, room: u64) -> Result<u32, KError> {
        self.grow_slots(s, want, room)
    }

    fn alloc_checks(&mut self, s: &Stream, spares: u32) -> Result<(), LoadError> {
        self.alloc_check_slots(s, spares)
    }

    fn check_spares_free(&self) -> Option<u32> {
        self.states.as_ref().map(qwen35::States::spares_free)
    }

    fn begin_check(&mut self, _s: &Stream, slot: u32, rows: u32) -> Result<(), KError> {
        self.check_begin(slot, rows)
    }

    fn end_check(&mut self, s: &Stream, slot: u32, kept: u32) -> Result<(), KError> {
        self.check_end(s, slot, kept)
    }

    fn route_table(&mut self, s: &Stream, slots: &[i32], lens: &[i32]) -> Result<(), KError> {
        self.route_rows(s, slots.to_vec(), lens, true)
    }

    fn forward_state(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        state: &StateBatch,
        kv: &KvPool,
        logits: &Tensor,
        taps: Option<&Taps>,
    ) -> Result<(), KError> {
        self.pass(s, b, Some(state), kv, logits, taps, &[])
    }

    fn forward_embeds(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        state: Option<&StateBatch>,
        kv: &KvPool,
        logits: &Tensor,
        taps: Option<&Taps>,
        embeds: &[EmbedRun],
    ) -> Result<(), KError> {
        self.pass(s, b, state, kv, logits, taps, embeds)
    }

    fn stream_pool_want(&self) -> (u64, u32) {
        self.stream_plan.as_ref().map_or((0, 0), |p| {
            (
                p.slots as u64 * p.slot_bytes + COPY_STREAM_BYTES,
                self.stream_rows,
            )
        })
    }

    fn alloc_stream_pool(&mut self, s: &Stream) -> Result<bool, KError> {
        let Some(plan) = self.stream_plan.take() else {
            return Ok(false);
        };
        let idle = (0..plan.slots)
            .map(|_| Buffer::alloc(s.device(), plan.slot_bytes))
            .collect::<Result<Vec<_>, KError>>()?;
        let profile = std::env::var("LARCH_PROFILE_STEPS").is_ok_and(|v| v == "1");
        self.stream = Some(RefCell::new(StreamPool::spawn(s, plan, idle, profile)?));
        Ok(true)
    }

    fn run_host(&self, s: &Stream, tag: u64, live: u32) -> Result<(), KError> {
        let (layer, rows, check) = untag(tag);
        // On an all-split load no layer waits on the route log, so a replay's routes skew nothing.
        let count = self.split_cache().is_some() && self.route_log.is_none();
        self.host_experts(s, layer, rows, i64::from(live), check, count)
    }

    fn set_router_deltas(&mut self, s: &Stream, d: Option<&RouterDeltas>) -> Result<(), KError> {
        let (Some(m), false) = (self.moe, s.capturing()) else {
            return Err(KError::Unsupported);
        };
        let Some(d) = d else {
            self.router_deltas = Vec::new();
            self.adapters = Vec::new();
            return Ok(());
        };
        let layers = self.cfg.n_layers as usize;
        let (dm, rank) = (self.cfg.d_model as usize, d.rank as usize);
        let mut seen = vec![false; layers];
        let fits = d.experts == m.experts
            && d.deltas.len() == d.layers.len()
            && d.layers.iter().all(|&l| {
                let l = usize::from(l);
                l < layers && !std::mem::replace(&mut seen[l], true)
            })
            && if rank == 0 {
                d.ids.is_empty()
                    && d.deltas
                        .iter()
                        .all(|v| v.len() == m.experts as usize && v.iter().all(|x| x.is_finite()))
            } else {
                d.ids.len() == d.layers.len()
                    && d.ids.iter().zip(&d.deltas).all(|(ids, v)| {
                        ids.windows(2).all(|w| w[0] < w[1])
                            && ids.last().is_none_or(|&e| e < m.experts)
                            && v.len() == ids.len() * rank * (dm + m.ffn as usize)
                            && v.iter().all(|x| x.is_finite())
                    })
            };
        if !fits {
            return Err(KError::InvalidArg);
        }
        if rank != 0 {
            // Q8_0 blocks along the expert ffn, and the gate and up scratch hold `t` and `y`.
            let (ef, used, r) = (m.ffn as usize, m.used as usize, ADAPTER_RANK);
            if self.split_cache().is_some() {
                eprintln!("{NO_LEARNING}");
                return Err(KError::Unsupported);
            }
            if rank > r || ef % 32 != 0 || used * ef < dm.max(used * r) {
                return Err(KError::Unsupported);
            }
            // Built whole before it replaces the old set, so a failed upload leaves the model as it was.
            let mut ads: Vec<Option<Adapters>> = (0..layers).map(|_| None).collect();
            // Zero bytes are zero Q8_0 blocks: only the adapted experts' rows are written.
            let (arow, brow, ne) = (ef / 32 * 34, r / 32 * 34, m.experts as usize);
            for ((&l, v), ids) in d.layers.iter().zip(&d.deltas).zip(&d.ids) {
                let (mut a, mut b) = (vec![0u8; ne * r * arow], vec![0u8; ne * dm * brow]);
                let mut pad = [0f32; ADAPTER_RANK];
                for (p, &e) in ids.iter().enumerate() {
                    let (bv, av) =
                        v[p * rank * (dm + ef)..][..rank * (dm + ef)].split_at(dm * rank);
                    let e = e as usize;
                    for (j, row) in av.chunks_exact(ef).enumerate() {
                        q8_0(row, &mut a[(e * r + j) * arow..][..arow]);
                    }
                    for (i, row) in bv.chunks_exact(rank).enumerate() {
                        pad[..rank].copy_from_slice(row);
                        q8_0(&pad, &mut b[(e * dm + i) * brow..][..brow]);
                    }
                }
                let host = matches!(self.layers[usize::from(l)].ffn, Ffn::Moe { host: true, .. });
                let ls = match (&self.host, host) {
                    (Some(x), true) => &x.s,
                    _ => s,
                };
                let up = |q: &[u8]| -> Result<Buffer, KError> {
                    let buf = Buffer::alloc(ls.device(), q.len() as u64)?;
                    ls.upload(&buf, 0, q)?;
                    Ok(buf)
                };
                ads[usize::from(l)] = Some(Adapters {
                    a: up(&a)?,
                    b: up(&b)?,
                });
            }
            self.router_deltas = Vec::new();
            self.adapters = ads;
            return Ok(());
        }
        // Built whole before it replaces the old set, so a failed upload leaves the model as it was.
        let mut bufs: Vec<Option<Buffer>> = (0..layers).map(|_| None).collect();
        for (&l, v) in d.layers.iter().zip(&d.deltas) {
            let b = Buffer::alloc(s.device(), DType::F32.bytes(v.len() as u64))?;
            s.upload(&b, 0, v)?;
            bufs[usize::from(l)] = Some(b);
        }
        self.router_deltas = bufs;
        self.adapters = Vec::new();
        Ok(())
    }

    fn embed_and_head(&self) -> Option<(Tensor<'_>, Tensor<'_>)> {
        Some((self.embed_table().ok()?, self.t(self.output).ok()?))
    }

    fn embed_tokens(&self, s: &Stream, ids: &[i32]) -> Result<(), KError> {
        self.host_embed(s, ids)?;
        if let Some(e) = &self.embd {
            e.primed.set(Some(ids.len() as i64));
        }
        Ok(())
    }
}

impl Drop for Transformer {
    fn drop(&mut self) {
        let Some(c) = self.cache.as_mut().map(RefCell::get_mut) else {
            return;
        };
        // Uploads still queued write into the slots, which drop with the cache.
        if let Some(Err(err)) = c.copy.as_ref().map(Stream::sync) {
            eprintln!("larch_expert_cache_copy,{err:?}");
        }
        c.finish_save();
        c.save_behind();
        c.finish_save();
    }
}

/// One step's attention inputs and outputs, the same for every attention layer.
struct AttnIo<'a> {
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    attn: Tensor<'a>,
    rope: ops::RopeParams<'a>,
    scale: f32,
}

impl Transformer {
    /// The attention scratch of a step of `rows` rows.
    fn attn_io(&self, rows: i64) -> Result<AttnIo<'_>, KError> {
        let (c, sc) = (&self.cfg, &self.sc);
        let (nh, nkv, hd) = (c.n_heads as i64, c.n_kv_heads as i64, c.head_dim as i64);
        Ok(AttnIo {
            q: Tensor::new(&sc.q, 0, DType::F16, &[rows, nh, hd])?,
            k: Tensor::new(&sc.k, 0, DType::F16, &[rows, nkv, hd])?,
            v: Tensor::new(&sc.v, 0, DType::F16, &[rows, nkv, hd])?,
            attn: Tensor::new(&sc.attn, 0, DType::F16, &[rows, nh, hd])?,
            rope: self.rope_params()?,
            scale: 1.0 / (hd as f32).sqrt(),
        })
    }

    /// Layer `l` (the `i`-th: its router deltas and adapters) of the step's residual stream in the
    /// scratch's `x`, in place.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        s: &Stream,
        i: usize,
        l: &Layer,
        b: &StepBatch,
        state: Option<&StateBatch>,
        kv: &KvPool,
        io: &AttnIo,
    ) -> Result<(), KError> {
        let (sc, rows) = (&self.sc, io.q.shape()[0]);
        let (d, eps) = (self.cfg.d_model as i64, self.cfg.rms_eps);
        let x = Tensor::new(&sc.x, 0, DType::F32, &[rows, d])?;
        let h = Tensor::new(&sc.h, 0, DType::F16, &[rows, d])?;
        let t = Tensor::new(&sc.t, 0, DType::F16, &[rows, d])?;
        let w = |w: W| self.t(w);
        let gdn_in = matches!(&l.mixer, Mixer::Gdn(g) if self.gdn_in_fusable(g, rows));
        if !gdn_in {
            ops::rmsnorm(s, &x, &w(l.attn_norm)?, eps, &h)?;
        }
        // Each mixer adds its output onto x itself where fused_proj_add takes its output projection.
        let added = match &l.mixer {
            Mixer::Attn(a) => self.attention(s, b, kv, a, io, &h, &t, &x)?,
            Mixer::Gdn(g) => {
                let norm = gdn_in.then_some((&x, l.attn_norm));
                self.gdn(s, g, &h, state, &t, norm, &x)?
            }
        };
        if !added {
            ops::add(s, &x, &t, &x)?;
        }
        let host_h = match (&l.ffn, &sc.route, self.moe) {
            (Ffn::Moe { host: true, .. }, Some(route), Some(m)) => {
                let used = i64::from(m.used);
                let at = DType::F32.bytes((rows * (i64::from(m.experts) + 2 * used)) as u64);
                Some(Tensor::new(route, at, DType::F16, &[rows, d])?)
            }
            _ => None,
        };
        let h = host_h.as_ref().unwrap_or(&h);
        let made = match l.ffn {
            Ffn::Moe {
                router,
                host,
                shared,
                ..
            } if self.moe_in_fusable(i, shared.as_ref(), rows) => {
                Some(self.fused_moe_in(s, i, &x, l.ffn_norm, h, router, host, shared)?)
            }
            _ => {
                ops::rmsnorm(s, &x, &w(l.ffn_norm)?, eps, h)?;
                None
            }
        };
        match l.ffn {
            Ffn::Dense { gate, up, down } => {
                if self.dense_ffn(s, gate, up, down, h, &t, Some(&x))? {
                    return Ok(());
                }
            }
            Ffn::Moe {
                router,
                gate,
                up,
                down,
                host,
                shared,
                // A mixed layer's experts need mmvq_id_mixed, which no fused entry runs.
                cold: None,
                ..
            } if self.moe_fusable(i, host, shared.as_ref(), rows) => {
                return self.fused_moe(s, i, router, [gate, up, down], shared, h, &x, made);
            }
            Ffn::Moe {
                router,
                gate,
                up,
                down,
                host,
                cold,
                ..
            } if made.is_some() => {
                let (ids, weights, sh) = made.ok_or(KError::InvalidArg)?;
                if let Some(sh) = &sh {
                    self.shared_down_add(s, sh, &x)?;
                }
                let route = Some((ids, weights));
                self.moe_ffn(s, i, router, [gate, up, down], cold, host, h, &t, &x, route)?
            }
            Ffn::Moe {
                router,
                gate,
                up,
                down,
                host,
                shared,
                cold,
                ..
            } => {
                // llama.cpp adds the shared and routed outputs in f32 before the residual;
                // here each joins the f32 residual stream on its own.
                if let Some(sh) = shared {
                    self.shared_ffn(s, &sh, h, &t)?;
                    ops::add(s, &x, &t, &x)?;
                }
                self.moe_ffn(s, i, router, [gate, up, down], cold, host, h, &t, &x, None)?
            }
        }
        ops::add(s, &x, &t, &x)
    }

    /// An attention layer of `h` into `out`: q, k, v (+ norms), rope, the KV write, attention over
    /// the decode rows then the prefill rows, Qwen 3.5's output gate, and o; or the gate and o onto the
    /// residual `x` through [`Transformer::proj_add`], then true.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        a: &Attn,
        io: &AttnIo,
        h: &Tensor,
        out: &Tensor,
        x: &Tensor,
    ) -> Result<bool, KError> {
        let (nd, np) = (i64::from(b.decode_rows), i64::from(b.prefill_rows));
        let AttnIo {
            q,
            k,
            v,
            attn,
            rope,
            scale,
        } = io;
        let (eps, layer) = (self.cfg.rms_eps, a.kv_layer);
        let pairs = a.gated.then(|| self.gated_pairs(q)).transpose()?;
        match &pairs {
            Some(qfull) => self.linear(s, a.wq, a.bq, h, &flat(qfull)?)?,
            None => self.linear(s, a.wq, a.bq, h, &flat(q)?)?,
        }
        self.linear(s, a.wk, a.bk, h, &flat(k)?)?;
        self.linear(s, a.wv, a.bv, h, &flat(v)?)?;
        let norms = match (a.q_norm, a.k_norm) {
            (Some(qn), Some(kn)) => Some([self.t(qn)?, self.t(kn)?]),
            _ => None,
        };
        // Four positions a row (an image's grid rows among them) on an M-RoPE model; else one.
        let multi = self
            .cfg
            .rope
            .sections
            .filter(|_| b.positions.shape().len() == 2);
        // fused_attn_in takes the split, the norms, rope and the KV write in one call where it runs.
        let fused = multi.is_none() && self.fusing(self.fused_on.attn_in) && {
            let gate = pairs
                .as_ref()
                .map(|_| self.attn_gate_of(attn))
                .transpose()?;
            let gate = gate.map(|g| g.reshape(q.shape())).transpose()?;
            let n = norms.as_ref().map(|[a, b]| [a, b]);
            let qi = pairs.as_ref().unwrap_or(q);
            let pool = (kv, layer);
            match self.fused_call(|| {
                ops::fused_attn_in(
                    s,
                    qi,
                    [k, v],
                    n,
                    eps,
                    &b.positions,
                    rope,
                    &b.slot_ids,
                    pool,
                    q,
                    gate.as_ref(),
                )
            }) {
                Err(KError::Unsupported) => false,
                r => r.map(|()| true)?,
            }
        };
        if !fused {
            if a.gated {
                self.gated_split(s, q)?;
            }
            if let Some([qn, kn]) = &norms {
                ops::qk_norm(s, q, qn, eps)?;
                ops::qk_norm(s, k, kn, eps)?;
            }
            match multi {
                Some(sec) => {
                    ops::rope_multi(s, q, &b.positions, rope, sec, ops::RopeMode::Imrope)?;
                    ops::rope_multi(s, k, &b.positions, rope, sec, ops::RopeMode::Imrope)?;
                }
                None => {
                    ops::rope(s, q, &b.positions, rope)?;
                    ops::rope(s, k, &b.positions, rope)?;
                }
            }
            ops::kv_write(s, k, v, &b.slot_ids, kv, layer)?;
        }
        if nd > 0 {
            let table = b.block_table.rows(0, nd)?;
            let (qr, ar) = (q.rows(0, nd)?, attn.rows(0, nd)?);
            ops::attn_decode(
                s,
                &qr,
                kv,
                layer,
                &table,
                &b.seq_lens,
                b.attn_splits,
                *scale,
                &ar,
            )?;
        }
        if np > 0 && b.check {
            // Causal spans at decode's splits: each row as attn_decode of that row alone.
            let table = b.block_table.rows(nd, b.prefill_q_len.shape()[0])?;
            let (qr, ar) = (q.rows(nd, np)?, attn.rows(nd, np)?);
            ops::attn(
                s,
                &qr,
                kv,
                layer,
                &table,
                &b.prefill_q_start,
                &b.prefill_q_len,
                b.attn_splits,
                *scale,
                &ar,
            )?;
        } else if np > 0 {
            let table = b.block_table.rows(nd, b.prefill_q_len.shape()[0])?;
            let (qr, ar) = (q.rows(nd, np)?, attn.rows(nd, np)?);
            ops::attn_prefill(
                s,
                &qr,
                kv,
                layer,
                &table,
                &b.prefill_q_start,
                &b.prefill_q_len,
                *scale,
                &ar,
            )?;
        }
        // fused_proj_add takes the gate and o onto x; else they run as ops into out.
        let gate = a.gated.then(|| self.attn_gate_of(attn)).transpose()?;
        if self.proj_add(s, a.wo, &flat(attn)?, gate.as_ref(), x)? {
            return Ok(true);
        }
        if a.gated {
            self.attn_gate(s, attn)?;
        }
        self.linear(s, a.wo, None, &flat(attn)?, out)?;
        Ok(false)
    }

    /// A check step's own kernels, masked attention and the verify GEMV, on this step's scratch
    /// before any op writes KV or state: a backend without them refuses the whole step here, which
    /// the engine then runs as plain decode.
    fn probe_check(
        &self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        let c = &self.cfg;
        let (nd, np) = (i64::from(b.decode_rows), i64::from(b.prefill_rows));
        let qs = [nd + np, c.n_heads as i64, c.head_dim as i64];
        let attn = self.layers.iter().find_map(|l| match &l.mixer {
            Mixer::Attn(a) => Some(a.kv_layer),
            Mixer::Gdn(_) => None,
        });
        if let Some(layer) = attn.filter(|_| np > 0) {
            let q = Tensor::new(&self.sc.q, 0, DType::F16, &qs)?;
            let out = Tensor::new(&self.sc.attn, 0, DType::F16, &qs)?;
            let table = b.block_table.rows(nd, b.prefill_q_len.shape()[0])?;
            ops::attn(
                s,
                &q.rows(nd, np)?,
                kv,
                layer,
                &table,
                &b.prefill_q_start,
                &b.prefill_q_len,
                b.attn_splits,
                1.0,
                &out.rows(nd, np)?,
            )?;
        }
        let h = Tensor::new(&self.sc.h, 0, DType::F16, &[1, c.d_model as i64])?;
        self.linear(s, self.output, None, &h, &logits.rows(0, 1)?)
    }

    /// The forward pass every [`Model`] method runs, with or without taps; `state` maps the rows to
    /// recurrent state slots, which a hybrid model needs. With an expert cache, the uploads of
    /// earlier steps are mapped first and, while recording, the step's routes counted after.
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        state: Option<&StateBatch>,
        kv: &KvPool,
        logits: &Tensor,
        taps: Option<&Taps>,
        embeds: &[EmbedRun],
    ) -> Result<(), KError> {
        if let Some(c) = self.cache.as_mut().map(RefCell::get_mut) {
            // ponytail: syncs the copy stream a step after its uploads (a few MB each); a
            // non-blocking event query in larch.h would drop even that rare wait.
            if let (Some(cs), false) = (&c.copy, c.pending.is_empty()) {
                cs.sync()?;
                for (l, i, slot) in std::mem::take(&mut c.pending) {
                    c.splits[l].as_mut().ok_or(KError::InvalidArg)?.slot[i as usize] = slot as i32;
                }
            }
            if let (true, Some(m)) = (std::mem::take(&mut c.log_later), self.moe) {
                let log = Transformer::route_log_bytes(&self.cfg, m, self.max_rows as u32);
                // A placement hint: the step runs, and with host counts alone a plan would favour
                // those layers on no resident evidence, so this load saves nothing.
                match Buffer::alloc(s.device(), log) {
                    Ok(b) => self.route_log = Some(b),
                    Err(e) => {
                        eprintln!("larch_expert_histogram_log,{e:?}");
                        c.path = None;
                    }
                }
            }
        }
        self.count_routes(s)?;
        self.run(s, b, state, kv, logits, taps, embeds)?;
        if let Some(c) = self.cache.as_mut().map(RefCell::get_mut) {
            let logged = self.route_log.is_some() && !s.capturing();
            c.logged = logged.then(|| b.token_ids.shape()[0]);
            c.passes = c.passes.wrapping_add(1);
            let every = if c.dev.is_some() {
                SAVE_EVERY
            } else {
                SAVE_EVERY_COLD
            };
            if c.passes % every == 0 {
                c.save_behind();
            }
        }
        Ok(())
    }

    /// While recording, counts the fully resident layers' routes of the step before (the split
    /// layers counted theirs), read here, after the engine has waited for that step.
    fn count_routes(&mut self, s: &Stream) -> Result<(), KError> {
        let (Some(log), Some(c), Some(m)) = (&self.route_log, &mut self.cache, self.moe) else {
            return Ok(());
        };
        let c = c.get_mut();
        let Some(rows) = c.logged.take().filter(|_| !s.capturing()) else {
            c.staged.fill(0);
            return Ok(());
        };
        let (e, n) = (m.experts as usize, rows as usize * m.used as usize);
        let mut ids = vec![0i32; self.layers.len() * n];
        s.download(log, 0, &mut ids)?;
        for (l, (layer, ids)) in self.layers.iter().zip(ids.chunks_exact(n)).enumerate() {
            if matches!(layer.ffn, Ffn::Moe { host: false, .. }) {
                for &i in ids.iter().filter(|&&i| (i as usize) < e) {
                    let n = &mut c.hist.counts[l * e + i as usize];
                    *n = n.saturating_add(1);
                }
            }
        }
        for (n, staged) in c.hist.counts.iter_mut().zip(&mut c.staged) {
            *n = n.saturating_add(std::mem::take(staged));
        }
        Ok(())
    }

    /// The device bytes the route log of [`record_routes`](Self::record_routes) takes for `rows`
    /// rows; the plan charges them.
    pub fn route_log_bytes(c: &ModelConfig, m: MoeCfg, rows: u32) -> u64 {
        DType::I32.bytes(u64::from(c.n_layers) * u64::from(rows) * u64::from(m.used))
    }

    /// Records each MoE layer's routed experts from here on, starting from `h`'s counts, and
    /// saves them to `path` every [`SAVE_EVERY`] steps and when the model drops: the histogram
    /// the next load plans its residency from. Only a model with experts on the host records: a
    /// model with every expert on the device needs no plan. Only a resident layer needs the log; a
    /// split or whole host layer counts from its own download. `InvalidArg` when `h` is another
    /// model's. A whole-layer load's log waits for its first step (see `log_later`).
    pub fn record_routes(&mut self, s: &Stream, path: PathBuf, h: Histogram) -> Result<(), KError> {
        let Some(m) = self
            .moe
            .filter(|_| self.cache.is_some() || self.host.is_some())
        else {
            return Ok(());
        };
        let n = self.cfg.n_layers as usize * m.experts as usize;
        if (h.layers, h.experts, h.counts.len()) != (self.cfg.n_layers, m.experts, n) {
            return Err(KError::InvalidArg);
        }
        let split = self.cache.is_some();
        let c = self
            .cache
            .get_or_insert_with(|| {
                RefCell::new(ExpertCache {
                    dev: None,
                    splits: Vec::new(),
                    hist: Histogram::uniform(0, 0, 0),
                    path: None,
                    copy: None,
                    pending: Vec::new(),
                    passes: 0,
                    writer: None,
                    logged: None,
                    hits: (0, 0),
                    no_host: 0,
                    split_steps: 0,
                    log_later: false,
                    staged: vec![0; n],
                })
            })
            .get_mut();
        let resident = |l: &Layer| matches!(l.ffn, Ffn::Moe { host: false, .. });
        if self.layers.iter().any(resident) {
            if split {
                let log = Transformer::route_log_bytes(&self.cfg, m, self.max_rows as u32);
                self.route_log = Some(Buffer::alloc(s.device(), log)?);
            } else {
                c.log_later = true;
            }
        }
        c.hist = h;
        c.path = Some(path);
        Ok(())
    }

    /// The split layers' routed (row, slot) pairs so far whose expert ran on the device, and on
    /// the host: the expert cache's hit rate. `None` without a split layer.
    pub fn expert_hits(&self) -> Option<(u64, u64)> {
        self.split_cache().map(|c| c.borrow().hits)
    }

    /// The expert cache when a layer is split: a recording whole-layer load takes dev's paths.
    fn split_cache(&self) -> Option<&RefCell<ExpertCache>> {
        self.cache.as_ref().filter(|c| c.borrow().dev.is_some())
    }

    /// The forward pass proper, see [`pass`](Self::pass).
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        state: Option<&StateBatch>,
        kv: &KvPool,
        logits: &Tensor,
        taps: Option<&Taps>,
        embeds: &[EmbedRun],
    ) -> Result<(), KError> {
        let c = &self.cfg;
        let rows = b.token_ids.shape()[0];
        let (nd, np) = (i64::from(b.decode_rows), i64::from(b.prefill_rows));
        // Four positions a row only on an M-RoPE model, refused here before any state moves.
        let four = matches!(b.positions.shape(), &[r, 4] if r == rows) && c.rope.sections.is_some();
        if rows != nd + np || rows > self.max_rows || b.positions.shape().len() > 1 && !four {
            return Err(KError::InvalidArg);
        }
        // The taps' layers, dtypes and shapes are checked before the first op; row values are not.
        let taps = taps.map(|t| t.blocks(c).map(|out| (t, out))).transpose()?;
        self.check.set(b.check && self.exact_checks);
        // A capture would keep the probe's ops; second-hit admission runs it eagerly first.
        if b.check && !self.checks_ran && !s.capturing() {
            self.probe_check(s, b, kv, logits)?;
            self.checks_ran = true;
        }
        self.stream_decode = i64::from(b.decode_rows);
        self.streaming = streams_step(
            self.stream_rows,
            self.stream.as_ref().is_some_and(|p| !p.borrow().disabled),
            b.prefill_rows,
            b.check,
            s.capturing(),
            self.adapters.iter().any(Option::is_some),
        );
        if self.streaming
            && let Some(p) = &self.stream
        {
            p.borrow_mut().begin(s, i64::from(b.prefill_rows))?;
        }
        if let Some(b) = state {
            self.route_state(s, b)?;
        }
        let c = &self.cfg;
        let (d, eps) = (c.d_model as i64, c.rms_eps);
        let sc = &self.sc;
        let x = Tensor::new(&sc.x, 0, DType::F32, &[rows, d])?;
        let h = Tensor::new(&sc.h, 0, DType::F16, &[rows, d])?;
        let io = self.attn_io(rows)?;
        let w = |w: W| self.t(w);

        // Taken by every pass, so rows primed for one never stand in for a later one's.
        let primed = self.embd.as_ref().and_then(|e| e.primed.take());
        match &self.embd {
            None => ops::embed(s, &self.embed_table()?, &b.token_ids, &x)?,
            // A capture leaves x to the host step before its replay, and the engine's eager step
            // primed it ([`Model::embed_tokens`]).
            Some(_) if s.capturing() || primed == Some(rows) => {}
            // Any other caller's ids are read back from the device.
            Some(_) => {
                let mut ids = vec![0i32; rows as usize];
                s.download(b.token_ids.buffer(), b.token_ids.offset(), &mut ids)?;
                self.host_embed(s, &ids)?;
            }
        }
        // Image rows replace their placeholder ids' rows, from the projector's output.
        for e in embeds {
            let n = e.table.shape()[0];
            ops::embed(s, &e.table, &e.ids, &x.rows(i64::from(e.row), n)?)?;
        }
        for (i, l) in self.layers.iter().enumerate() {
            self.layer(s, i, l, b, state, kv, &io)?;
            if let Some((tp, out)) = &taps {
                for (_, o) in tp.layers.iter().zip(out).filter(|e| usize::from(*e.0) == i) {
                    ops::embed(s, &x, &tp.rows, o)?;
                }
            }
        }
        if let Some((tp, out)) = taps.as_ref().filter(|t| t.0.final_state) {
            // Every row's norm, then the tapped rows: per row the same norm the logits rows get.
            // ponytail: norms all rows to need no identity-ids buffer; gather first when a prefill
            // profile shows this norm.
            ops::rmsnorm(s, &x, &w(self.out_norm)?, eps, &h)?;
            ops::embed(s, &h, &tp.rows, &out[tp.layers.len()])?;
        }

        // Gather the logits rows (embed is a row gather), norm them, project.
        let n = b.logits_rows.shape()[0];
        let xs = Tensor::new(&sc.t, 0, DType::F32, &[n, d])?;
        let hs = h.rows(0, n)?;
        ops::embed(s, &x, &b.logits_rows, &xs)?;
        ops::rmsnorm(s, &xs, &w(self.out_norm)?, eps, &hs)?;
        // Direct tiles avoid a whole FP16 vocabulary matrix. Unmeasured shapes retain chunked mmvq.
        let output = w(self.output)?;
        if self.check.get() {
            return self.linear(s, self.output, None, &hs, logits);
        }
        if self.direct(s, self.output, &output, &hs, None, logits)? {
            return Ok(());
        }
        for at in (0..n).step_by(MMVQ_SLICE_ROWS as usize) {
            let m = (n - at).min(MMVQ_SLICE_ROWS);
            ops::mmvq(s, &output, &hs.rows(at, m)?, None, &logits.rows(at, m)?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Layer {
        fn attn(&self) -> &Attn {
            match &self.mixer {
                Mixer::Attn(a) => a,
                Mixer::Gdn(_) => panic!("a Gated DeltaNet layer"),
            }
        }
    }

    impl Ffn {
        /// gate, up and down of a dense block.
        fn dense(self) -> Option<[W; 3]> {
            match self {
                Ffn::Dense { gate, up, down } => Some([gate, up, down]),
                Ffn::Moe { .. } => None,
            }
        }
    }
    use crate::{Arch, tiny};
    use larch_kernels::CPU;

    /// Two logical CPUs stay with the engine's thread and the GPU driver's; never fewer than one.
    #[test]
    fn host_experts_leave_two_cpus_to_the_engine_and_the_driver() {
        assert_eq!(beside_gpu_threads(8), 6);
        assert_eq!(beside_gpu_threads(16), 14);
        assert_eq!(beside_gpu_threads(3), 1);
        assert_eq!(beside_gpu_threads(2), 1);
        assert_eq!(beside_gpu_threads(1), 1);
    }

    fn linear_fixture(rows: u32, name: &str) -> (Device, Stream, Transformer) {
        let path = std::env::temp_dir().join(format!("larch-{name}-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen2, 1)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let model = Transformer::load(&path, &dev, &s, rows).unwrap();
        std::fs::remove_file(path).unwrap();
        (dev, s, model)
    }

    /// A plan that splits every MoE layer records from the split layers' own downloads, so it has
    /// no route log; a resident layer's log is as large as the plan charged.
    #[test]
    fn only_a_resident_layer_needs_the_route_log() {
        let path = std::env::temp_dir().join(format!("larch-log-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 1)).unwrap();
        let hist = Histogram::path(&path);
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        for (resident, log) in [(vec![vec![0], vec![1]], false), (vec![vec![0]], true)] {
            let plan = ExpertPlan {
                resident,
                dynamic: false,
            };
            let mut m = Transformer::load_cached(&path, &dev, &s, 3, &plan).unwrap();
            let h = Histogram::uniform(0, tiny::LAYERS, tiny::EXPERTS);
            m.record_routes(&s, hist.clone(), h).unwrap();
            let want = Transformer::route_log_bytes(&m.cfg, m.moe.unwrap(), 3);
            assert_eq!(m.route_log.as_ref().map(Buffer::bytes), log.then_some(want));
        }
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&hist).unwrap();
    }

    /// The GPU map leaves out a built-in drafter's layer past the config's (the hybrid MoE target's blk.40).
    #[test]
    fn used_spans_skip_layers_past_the_config() {
        let trunk = Gguf::parse(&tiny::gguf(Arch::Qwen35Moe, 1)).unwrap();
        let g = Gguf::parse(&tiny::gguf_mtp(Arch::Qwen35Moe, 1)).unwrap();
        // Data-section offsets: the trunk's tensors sit where they do without the block, which
        // follows them.
        let data = |g: &Gguf, layers| {
            let at = g.data_offset;
            (used_spans(g, layers).into_iter()).map(move |(a, b)| (a - at, b - at))
        };
        let used: Vec<_> = data(&trunk, tiny::LAYERS).collect();
        assert_eq!(used.len(), trunk.tensors.len());
        assert_eq!(data(&g, tiny::LAYERS).collect::<Vec<_>>(), used);
        assert!(data(&g, tiny::LAYERS + 1).count() > used.len());
    }
    #[test]
    fn f16_bits_round_to_nearest_even() {
        let one = 0x3c00;
        let ulp = 2f32.powi(-10);
        for (x, want) in [
            (1.0, one),
            (-2.0, 0xc000),
            (1.0 + ulp / 2.0, one),      // a tie rounds to the even 1.0
            (1.0 + 1.5 * ulp, one + 2),  // a tie rounds up to even
            (1.0 + 0.75 * ulp, one + 1), // past half
            (65504.0, 0x7bff),           // the largest finite
            (65520.0, 0x7c00),           // rounds up to infinity
            (2f32.powi(-24), 1),         // the smallest subnormal
            (2f32.powi(-25), 0),         // half of it, a tie to even 0
            (1.5 * 2f32.powi(-24), 2),   // a subnormal tie to even
            (6.0e-5, 0x03ef),            // a subnormal near the normal range
            (2f32.powi(-14) * (1.0 - ulp / 4.0), 0x0400), // carries into the smallest normal
            (f32::INFINITY, 0x7c00),
            (1e-30, 0),
        ] {
            assert_eq!(f16_bits(x), want, "{x:e}");
        }
        assert_eq!(f16_bits(f32::NAN) & 0x7e00, 0x7e00);
    }

    #[test]
    fn backend_and_weight_routing_boundaries() {
        let w = W {
            off: 0,
            dtype: DType::Q4K,
            shape: [256, 256],
            ndim: 2,
            side: false,
        };
        for backend in [Backend::Cuda, Backend::Metal, Backend::Cpu, Backend::Hip] {
            for d_model in [256, 896, 2048, 3072, 3583, 3584, 4096] {
                for dtype in [
                    DType::Q4K,
                    DType::Q5K,
                    DType::Q6K,
                    DType::Q8_0,
                    DType::Q4_0,
                    DType::Q5_0,
                ] {
                    let w = W { dtype, ..w };
                    for (rows, cuda, other) in [
                        (16, true, true),
                        (17, true, false),
                        (32, true, false),
                        (33, false, false),
                    ] {
                        let expected = if backend == Backend::Cuda
                            && d_model >= 3584
                            && matches!(dtype, DType::Q4K | DType::Q6K)
                        {
                            cuda
                        } else {
                            other
                        };
                        assert_eq!(
                            use_mmvq(mmvq_crossover(backend, d_model), w, rows),
                            expected,
                            "{backend:?} d_model {d_model} {dtype:?} rows {rows}"
                        );
                    }
                }
            }
        }
    }

    /// The routing before card profiles, verbatim: the oracle for "no profile changes nothing".
    fn routes_before_profiles(w: W, rows: i64, units: u32, arch: u32) -> Option<I8> {
        let tiles = (w.shape[0] + 127) / 128;
        let units = i64::from(units);
        if rows < 8 || w.shape[0] <= 0 || w.shape[1] % 128 != 0 || units == 0 {
            None
        } else if arch != 120 {
            (w.dtype == DType::Q4K && tiles >= units).then_some(I8::Plain)
        } else if !matches!(w.dtype, DType::Q4K | DType::Q6K) {
            None
        } else if rows <= 32 {
            if tiles >= units {
                Some(I8::Plain)
            } else if tiles * rows >= 112 {
                Some(I8::Split)
            } else {
                None
            }
        } else {
            (tiles > 8 && !(w.dtype == DType::Q6K && rows >= 96)).then_some(I8::Plain)
        }
    }

    /// Projection shapes of m1, m3, m5 and m8, a full-grid Q6_K and a k the int8 pack refuses.
    const ROUTE_SHAPES: [[i64; 2]; 12] = [
        [4096, 4096],
        [1024, 4096],
        [14336, 4096],
        [4096, 14336],
        [128256, 4096],
        [512, 3584],
        [3584, 3584],
        [18944, 3584],
        [3584, 18944],
        [151936, 896],
        [896, 896],
        [14336, 4160],
    ];

    fn route_weight(dtype: DType, shape: [i64; 2]) -> W {
        W {
            off: 0,
            dtype,
            shape,
            ndim: 2,
            side: false,
        }
    }

    #[test]
    fn no_profile_keeps_todays_routes_on_sm86_sm89_sm120() {
        for arch in [86, 89, 120] {
            assert_eq!(quant_gemm_device(Backend::Cuda, arch, false), arch == 120);
            for units in [16, 24, 70] {
                for dtype in [DType::Q4K, DType::Q6K, DType::Q8_0] {
                    for shape in ROUTE_SHAPES {
                        let w = route_weight(dtype, shape);
                        for rows in 1..=600 {
                            assert_eq!(
                                quant_i8_route(w, rows, units, arch, &[]),
                                routes_before_profiles(w, rows, units, arch),
                                "sm_{arch} {units} SMs {dtype:?} {shape:?} rows {rows}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A profile with every route takes sm_120's rules off sm_120, except where the exclusion
    /// table says no; each route alone changes only its own rule; nothing changes off CUDA.
    #[test]
    fn a_profile_adds_sm120_rules_and_the_exclusion_table_wins() {
        let all = Route::ALL;
        for backend in [Backend::Cpu, Backend::Metal, Backend::Hip] {
            assert!(!quant_gemm_device(backend, 120, true));
        }
        for arch in [86, 89] {
            assert!(quant_gemm_device(Backend::Cuda, arch, true));
            for units in [16, 24, 70] {
                for dtype in [DType::Q4K, DType::Q6K, DType::Q8_0] {
                    for shape in ROUTE_SHAPES {
                        let w = route_weight(dtype, shape);
                        for rows in 1..=600 {
                            let sm120 = routes_before_profiles(w, rows, units, 120);
                            let held = shape[0] == 512 && (28..=32).contains(&rows);
                            assert_eq!(
                                quant_i8_route(w, rows, units, arch, &all),
                                if held { None } else { sm120 },
                                "sm_{arch} {units} SMs {dtype:?} {shape:?} rows {rows}"
                            );
                        }
                    }
                }
            }
        }
        let route = |route, shape, rows, units| {
            quant_i8_route(route_weight(DType::Q4K, shape), rows, units, 86, &[route])
        };
        // Stream-K alone: short grids split, full grids keep the Q4_K default, long rows stay off.
        assert_eq!(
            route(Route::I8StreamK, [1024, 4096], 14, 70),
            Some(I8::Split)
        );
        assert_eq!(route(Route::I8StreamK, [512, 3584], 28, 70), None);
        assert_eq!(
            route(Route::I8StreamK, [14336, 4096], 8, 70),
            Some(I8::Plain)
        );
        assert_eq!(route(Route::I8StreamK, [4096, 4096], 64, 70), None);
        assert_eq!(route(Route::I8Long, [4096, 4096], 64, 70), Some(I8::Plain));
        assert_eq!(route(Route::I8Long, [4096, 4096], 16, 70), None);
        let q6 = route_weight(DType::Q6K, [128256, 4096]);
        assert_eq!(quant_i8_route(q6, 16, 20, 86, &[]), None);
        assert_eq!(
            quant_i8_route(q6, 16, 20, 86, &[Route::I8FullGrid]),
            Some(I8::Plain)
        );
    }

    /// On a model posing as an sm_86 card: every case is one route changing the first route that
    /// runs at its rows, once per shape; the exclusion table leaves no case; and the CPU reference
    /// of a column slice matches the same op over the whole weight.
    #[test]
    fn route_cases_change_the_route_and_their_reference_slices_match() {
        let (dev, s, mut model) = linear_fixture(64, "route-cases");
        (model.backend, model.arch_code, model.quant_i8) = (Backend::Cuda, 86, true);
        // 8 SMs: the 512-wide Q4_K gate would split from 28 rows, which only sm_120 may.
        model.compute_units = 8;
        let held = |c: &RouteCase| c.route == Route::I8StreamK && c.shape()[0] == 512;
        assert!(!model.route_cases(&[8, 16, 28, 32, 64]).iter().any(held));
        // 2 SMs: the Q6_K 256-wide projections are full grids, which only the profile's rule takes.
        model.compute_units = 2;
        let cases = model.route_cases(&[8, 16, 28, 32, 64]);
        let first =
            |m: &Transformer, c: &RouteCase, card: &[Route]| match m.picks(c.w, c.rows, card) {
                (Some(i8), _) => Some(Some(i8)),
                (None, qg) => qg.then_some(None),
            };
        for c in &cases {
            assert_ne!(first(&model, c, &[]), first(&model, c, &[c.route]), "{c:?}");
            assert!(
                c.ws.iter()
                    .all(|w| (w.dtype, w.shape) == (c.w.dtype, c.w.shape))
            );
            let key = |d: &RouteCase| (d.route, d.rows, d.w.dtype, d.w.shape);
            assert_eq!(cases.iter().filter(|d| key(d) == key(c)).count(), 1);
        }
        let find = |rows, shape| {
            cases.iter().find(|c| {
                (c.route, c.rows, c.dtype(), c.shape())
                    == (Route::I8FullGrid, rows, DType::Q6K, shape)
            })
        };
        assert!(find(64, [256, 512]).is_none());
        let c = find(32, [256, 512])
            .expect("the Q6_K down projection at 32 rows")
            .clone();

        let [n, k] = c.shape();
        let x: Vec<u16> = (0..32 * k)
            .map(|i| 0x3000 + (i * 7919 % 0x0c00) as u16)
            .collect();
        let xb = Buffer::alloc(&dev, 32 * k as u64 * 2).unwrap();
        s.upload(&xb, 0, &x).unwrap();
        let ob = Buffer::alloc(&dev, 32 * n as u64 * 4).unwrap();
        let (xt, ot) = (
            Tensor::new(&xb, 0, DType::F16, &[32, k]).unwrap(),
            Tensor::new(&ob, 0, DType::F32, &[32, n]).unwrap(),
        );
        model.route_case(&s, &c, true, &xt, &ot).unwrap();
        let mut whole = vec![0f32; (32 * n) as usize];
        s.download(&ob, 0, &mut whole).unwrap();
        let slice = model.route_reference(&s, &c, &x, n - 128..n).unwrap();
        for r in 0..32 {
            let row = &whole[(r * n + n - 128) as usize..((r + 1) * n) as usize];
            assert_eq!(&slice[(r * 128) as usize..((r + 1) * 128) as usize], row);
        }
    }

    #[test]
    fn i8_routes_follow_the_whole_grid_and_the_measured_crossovers() {
        for backend in [Backend::Cuda, Backend::Cpu, Backend::Metal, Backend::Hip] {
            for arch in [0, 70, 75, 80, 86, 89, 120, 121] {
                for switch in [None, Some(false), Some(true)] {
                    assert_eq!(
                        quant_i8_device(backend, arch, switch),
                        backend == Backend::Cuda
                            && arch >= 80
                            && switch.unwrap_or(matches!(arch, 86 | 89 | 120)),
                        "{backend:?} sm_{arch} {switch:?}"
                    );
                }
            }
        }
        let (q4, q6, q8) = (DType::Q4K, DType::Q6K, DType::Q8_0);
        let (p, sp) = (Some(I8::Plain), Some(I8::Split));
        for (dtype, shape, rows, units, arch, route) in [
            // q/o: 32 tiles, Stream-K through 32 rows, then the plain tile.
            (q4, [4096, 4096], 7, 70, 120, None),
            (q4, [4096, 4096], 8, 70, 120, sp),
            (q4, [4096, 4096], 32, 70, 120, sp),
            (q4, [4096, 4096], 33, 70, 120, p),
            (q4, [4096, 4096], 512, 70, 120, p),
            // k/v: 8 tiles take Stream-K only from 14 rows, and nothing past 32.
            (q4, [1024, 4096], 13, 70, 120, None),
            (q4, [1024, 4096], 14, 70, 120, sp),
            (q6, [1024, 4096], 32, 70, 120, sp),
            (q4, [1024, 4096], 33, 70, 120, None),
            (q4, [1024, 4096], 512, 70, 120, None),
            (q4, [512, 3584], 24, 70, 120, None),
            (q4, [512, 3584], 28, 70, 120, sp),
            // gate/up and the lm_head fill the device on their own.
            (q4, [14336, 4096], 8, 70, 120, p),
            (q4, [14336, 4096], 527, 70, 120, p),
            (q6, [128256, 4096], 16, 70, 120, p),
            (q6, [128256, 4096], 512, 70, 120, None),
            // Q6_K's plain tile stops at 95 rows on both down shapes; its k/v stay off.
            (q6, [4096, 14336], 8, 70, 120, sp),
            (q6, [4096, 14336], 17, 70, 120, sp),
            (q6, [4096, 14336], 64, 70, 120, p),
            (q6, [4096, 14336], 65, 70, 120, p),
            (q6, [4096, 14336], 95, 70, 120, p),
            (q6, [4096, 14336], 96, 70, 120, None),
            (q6, [3584, 18944], 65, 70, 120, p),
            (q6, [3584, 18944], 95, 70, 120, p),
            (q6, [3584, 18944], 96, 70, 120, None),
            (q6, [512, 3584], 80, 70, 120, None),
            (q8, [151936, 896], 16, 70, 120, None),
            // sm_86 keeps full Q4_K grids (its sm_120-rule run missed m3 parity at --parallel 32), and so do unmeasured archs.
            (q4, [14336, 4096], 8, 16, 86, p),
            (q4, [1024, 4096], 16, 16, 86, None),
            (q4, [4096, 4096], 16, 70, 89, None),
            (q6, [128256, 4096], 16, 20, 89, None),
            // Shapes the pack cannot take, and no device.
            (q4, [14336, 4096], 8, 0, 120, None),
            (q4, [0, 4096], 8, 70, 120, None),
            (q4, [14336, 4160], 8, 70, 120, None),
        ] {
            let w = W {
                off: 0,
                dtype,
                shape,
                ndim: 2,
                side: false,
            };
            assert_eq!(
                quant_i8_route(w, rows, units, arch, &[]),
                route,
                "{dtype:?} {shape:?} rows={rows} units={units} arch={arch}"
            );
        }
    }

    /// Every measured sm_120 point of m1, m5 and m8 (70 SMs): the rows each projection routes and
    /// the measured rows it leaves on mmvq or dequant + gemm; and the model shapes never measured
    /// stay off the direct path at any row.
    #[test]
    fn direct_quant_routes_follow_measured_sm120_points() {
        for backend in [Backend::Cuda, Backend::Cpu, Backend::Metal, Backend::Hip] {
            for arch in [0, 86, 89, 120, 121] {
                let on = backend == Backend::Cuda && arch == 120;
                assert_eq!(quant_gemm_device(backend, arch, false), on);
            }
        }
        let w = |dtype, shape| W {
            off: 0,
            dtype,
            shape,
            ndim: 2,
            side: false,
        };
        let wide_q6 = &[16, 17, 24, 32, 48, 64, 96, 128, 256, 512][..];
        for (weight, routed, kept) in [
            // m1: attention q/o, FFN down (Q4_K and Q6_K halves), FFN up, head.
            (
                w(DType::Q4K, [4096, 4096]),
                &[16, 24, 32][..],
                &[1, 2, 4, 8, 17, 48, 64][..],
            ),
            (
                w(DType::Q4K, [4096, 14336]),
                &[16, 24, 32, 48, 64, 96, 128, 256][..],
                &[1, 2, 4, 17, 512][..],
            ),
            (
                w(DType::Q6K, [4096, 14336]),
                &[16, 32, 96, 128, 256][..],
                &[1, 2, 4, 8, 17, 24, 48, 64, 512][..],
            ),
            (
                w(DType::Q4K, [14336, 4096]),
                &[16, 17, 24, 32, 48, 64, 96, 128, 256, 512, 1024][..],
                &[1, 2, 4, 8, 2048][..],
            ),
            (w(DType::Q6K, [128256, 4096]), wide_q6, &[1, 2, 4, 8][..]),
            // m5: FFN up, head, FFN down (both halves).
            (
                w(DType::Q4K, [18944, 3584]),
                &[16, 17, 24, 32, 48, 64, 96, 128, 256, 512, 1024][..],
                &[1, 2, 4, 8][..],
            ),
            (w(DType::Q6K, [152064, 3584]), wide_q6, &[1, 2, 4, 8][..]),
            (
                w(DType::Q6K, [3584, 18944]),
                &[16, 32, 96, 128, 256][..],
                &[1, 2, 4, 8, 17, 24, 48, 64, 512][..],
            ),
            (
                w(DType::Q4K, [3584, 18944]),
                &[16, 24, 32, 48, 64, 96, 128, 256][..],
                &[1, 2, 4, 8, 17, 512][..],
            ),
            // m8: the tied Q8_0 head.
            (w(DType::Q8_0, [151936, 896]), &[16][..], &[8][..]),
        ] {
            for &rows in routed {
                assert!(
                    use_quant_gemm(weight, rows, 70),
                    "{:?} {:?} {rows}",
                    weight.dtype,
                    weight.shape
                );
            }
            for &rows in kept {
                assert!(
                    !use_quant_gemm(weight, rows, 70),
                    "{:?} {:?} {rows}",
                    weight.dtype,
                    weight.shape
                );
            }
        }
        for weight in [
            w(DType::Q4K, [3584, 3584]),
            w(DType::Q4K, [512, 3584]),
            w(DType::Q6K, [512, 3584]),
            w(DType::Q4K, [1024, 4096]),
            w(DType::Q6K, [1024, 4096]),
            w(DType::Q8_0, [896, 896]),
            w(DType::Q8_0, [128, 896]),
            w(DType::Q8_0, [4864, 896]),
            w(DType::Q8_0, [896, 4864]),
        ] {
            assert!(
                (1..=4096).all(|rows| !use_quant_gemm(weight, rows, 70)),
                "{:?}",
                weight.shape
            );
        }
    }

    #[test]
    fn linear_slices_match_cpu_mmvq_without_dq() {
        let (dev, s, mut model) = linear_fixture(32, "slices");
        model.sc.dq = None;
        // Exercise the CUDA slice policy against CPU reference arithmetic.
        model.mmvq_max_rows = mmvq_crossover(Backend::Cuda, 3584);
        // Both layers include Q4_K/Q6_K, including the second layer's down projection.
        let weights = model.layers.iter().flat_map(|l| {
            let Mixer::Attn(a) = &l.mixer else {
                unreachable!("qwen2 is all attention")
            };
            let biases = [a.bq, a.bk, a.bv];
            l.linears()
                .into_iter()
                .enumerate()
                .map(move |(i, w)| (w, biases.get(i).copied().flatten()))
        });
        let mut counts = vec![0, 1, 16, 17, 32, MMVQ_MAX_ROWS];
        counts.sort_unstable();
        counts.dedup();
        for (w, bias) in weights {
            for &rows in &counts {
                let input = Buffer::alloc(&dev, DType::F16.bytes(((rows + 2) * w.shape[1]) as u64))
                    .unwrap();
                let data: Vec<u8> = (0..(rows + 2) * w.shape[1])
                    .flat_map(|i| {
                        (0x2800u16 + ((i / w.shape[1]) * 64 + i % 64) as u16).to_le_bytes()
                    })
                    .collect();
                s.upload(&input, 0, &data).unwrap();
                let x = Tensor::new(&input, 0, DType::F16, &[rows + 2, w.shape[1]])
                    .unwrap()
                    .rows(1, rows)
                    .unwrap();
                for dtype in [DType::F16, DType::F32] {
                    let size = dtype.bytes(((rows + 2) * w.shape[0]) as u64);
                    let actual = Buffer::alloc(&dev, size).unwrap();
                    let expected = Buffer::alloc(&dev, size).unwrap();
                    let sentinel = vec![0xa5u8; size as usize];
                    s.upload(&actual, 0, &sentinel).unwrap();
                    s.upload(&expected, 0, &sentinel).unwrap();
                    let out = Tensor::new(&actual, 0, dtype, &[rows + 2, w.shape[0]])
                        .unwrap()
                        .rows(1, rows)
                        .unwrap();
                    let reference = Tensor::new(&expected, 0, dtype, &[rows + 2, w.shape[0]])
                        .unwrap()
                        .rows(1, rows)
                        .unwrap();
                    if !matches!(w.dtype, DType::Q4K | DType::Q6K) && rows > 16 {
                        assert_eq!(model.linear(&s, w, bias, &x, &out), Err(KError::InvalidArg));
                        continue;
                    }
                    model.linear(&s, w, bias, &x, &out).unwrap();
                    // ABI 2 caps a call at 16 rows; the CPU reference computes each row independently.
                    for row in 0..rows {
                        ops::mmvq(
                            &s,
                            &model.t(w).unwrap(),
                            &x.rows(row, 1).unwrap(),
                            bias.map(|b| model.t(b).unwrap()).as_ref(),
                            &reference.rows(row, 1).unwrap(),
                        )
                        .unwrap();
                    }
                    let mut got = vec![0u8; size as usize];
                    let mut want = vec![0u8; size as usize];
                    s.download(&actual, 0, &mut got).unwrap();
                    s.download(&expected, 0, &mut want).unwrap();
                    assert_eq!(
                        got, want,
                        "rows {rows}, {:?}, {dtype:?}: CPU reference or guard rows differ",
                        w.dtype
                    );
                }
            }
        }
    }

    #[test]
    fn linear_above_crossover_requires_gemm_scratch() {
        let rows = MMVQ_MAX_ROWS + 1;
        let (dev, s, mut model) = linear_fixture(rows as u32, "gemm-route");
        let w = model.layers[0].attn().wq;
        let bias = model.layers[0].attn().bq.unwrap();
        let input = Buffer::alloc(&dev, DType::F16.bytes((rows * w.shape[1]) as u64)).unwrap();
        let data: Vec<u8> = (0..rows * w.shape[1])
            .flat_map(|i| (0x3000u16 + (i % 1024) as u16).to_le_bytes())
            .collect();
        s.upload(&input, 0, &data).unwrap();
        let x = Tensor::new(&input, 0, DType::F16, &[rows, w.shape[1]]).unwrap();
        let size = DType::F32.bytes((rows * w.shape[0]) as u64);
        let actual = Buffer::alloc(&dev, size).unwrap();
        let expected = Buffer::alloc(&dev, size).unwrap();
        let out = Tensor::new(&actual, 0, DType::F32, &[rows, w.shape[0]]).unwrap();
        let reference = Tensor::new(&expected, 0, DType::F32, &[rows, w.shape[0]]).unwrap();
        let scratch =
            Buffer::alloc(&dev, DType::F16.bytes((w.shape[0] * w.shape[1]) as u64)).unwrap();
        let dq = Tensor::new(&scratch, 0, DType::F16, &w.shape).unwrap();
        ops::dequant(&s, &model.t(w).unwrap(), &dq).unwrap();
        ops::gemm(
            &s,
            &x,
            &dq,
            Some(&model.t(bias).unwrap()),
            &reference,
            1.0,
            0.0,
        )
        .unwrap();
        model.linear(&s, w, Some(bias), &x, &out).unwrap();
        let mut got = vec![0u8; size as usize];
        let mut want = vec![0u8; size as usize];
        s.download(&actual, 0, &mut got).unwrap();
        s.download(&expected, 0, &mut want).unwrap();
        assert_eq!(got, want, "R+1 must use dequant + gemm");
        model.sc.dq = None;
        assert_eq!(
            model.linear(&s, w, Some(bias), &x, &out),
            Err(KError::InvalidArg)
        );
    }

    #[test]
    fn dq_scratch_is_allocated_only_above_crossover() {
        for rows in [16, 17, 32, 33] {
            let (_, _, model) = linear_fixture(rows as u32, "dq-boundary");
            assert_eq!(model.sc.dq.is_some(), rows > 16);
        }
    }

    #[test]
    fn linear_rejects_extra_output_rows() {
        let (dev, s, mut model) = linear_fixture(32, "linear-shape");
        model.mmvq_max_rows = mmvq_crossover(Backend::Cuda, 3584);
        let w = model.layers[0].attn().wk;
        let input = Buffer::alloc(&dev, DType::F16.bytes((32 * w.shape[1]) as u64)).unwrap();
        let output = Buffer::alloc(&dev, DType::F16.bytes((33 * w.shape[0]) as u64)).unwrap();
        let x = Tensor::new(&input, 0, DType::F16, &[32, w.shape[1]]).unwrap();
        let out = Tensor::new(&output, 0, DType::F16, &[33, w.shape[0]]).unwrap();
        assert_eq!(model.linear(&s, w, None, &x, &out), Err(KError::InvalidArg));
    }

    /// tiny's data section is about 1 MB, so only a small chunk takes the loop past its first pass.
    #[test]
    fn weights_stream_in_chunks() {
        let bytes = tiny::gguf(Arch::Qwen2, 1);
        let path = std::env::temp_dir().join(format!("larch-chunks-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let m = Transformer::load_chunked(
            &path,
            &dev,
            &s,
            1,
            4096,
            Placement::default(),
            None,
            u64::MAX,
            false,
            false,
        );
        std::fs::remove_file(&path).unwrap();
        let m = m.unwrap();
        let mut got = vec![0u8; m.weights.bytes() as usize];
        s.download(&m.weights, 0, &mut got).unwrap();
        let data = Gguf::parse(&bytes).unwrap().data_offset as usize;
        assert!(
            got == bytes[data..],
            "weights differ from the file's data section"
        );
    }

    /// A device's `Oom` (an allocation past `maxBufferLength` or past memory) is `DoesNotFit`
    /// against its working set; other errors stay `Backend`.
    #[test]
    fn a_fit_failure_is_does_not_fit() {
        let dev = Device::open(CPU).unwrap();
        let have = dev.props().unwrap().recommended_working_set;
        assert_eq!(
            fit_err(&dev, 7, KError::Oom),
            LoadError::DoesNotFit { need: 7, have }
        );
        assert_eq!(
            fit_err(&dev, 7, KError::Launch),
            LoadError::Backend(KError::Launch.code())
        );
    }

    /// Every weight of every arch sits at its `off`, read after `load` dropped the file and its map;
    /// the weights are the mapped file exactly when the device is Metal.
    #[test]
    fn every_arch_loads_its_tensors() {
        let gpu = larch_kernels::device_count().unwrap() > 0;
        let dev = Device::open(if gpu { 0 } else { CPU }).unwrap();
        let s = Stream::new(&dev).unwrap();
        let metal = dev.props().unwrap().backend == larch_kernels::Backend::Metal;
        for arch in [Arch::Llama, Arch::Qwen2, Arch::Qwen3] {
            let bytes = tiny::gguf(arch, 3);
            let path = std::env::temp_dir()
                .join(format!("larch-arch-{arch:?}-{}.gguf", std::process::id()));
            std::fs::write(&path, &bytes).unwrap();
            let m = Transformer::load(&path, &dev, &s, 32);
            std::fs::remove_file(&path).unwrap();
            let m = m.unwrap();
            assert_eq!(
                m.mmvq_max_rows,
                if dev.props().unwrap().backend == Backend::Cuda && m.cfg.d_model >= 3584 {
                    32
                } else {
                    16
                }
            );
            // Small models and unmeasured weight types must have gemm scratch even on CUDA at 32.
            assert!(m.sc.dq.is_some());
            let g = Gguf::parse(&bytes).unwrap();
            let mapped = m.weights.bytes() == bytes.len() as u64;
            assert_eq!(mapped, metal, "{arch:?}: mapped exactly on Metal");

            let mut ws = vec![
                ("token_embd.weight".to_string(), m.tok_embd),
                ("output_norm.weight".into(), Some(m.out_norm)),
                ("output.weight".into(), Some(m.output)),
                ("rope_freqs.weight".into(), m.rope_freqs),
            ];
            for (i, l) in m.layers.iter().enumerate() {
                let n = |s: &str| format!("blk.{i}.{s}");
                let [gate, up, down] = l.ffn.dense().unwrap();
                ws.extend([
                    (n("attn_norm.weight"), Some(l.attn_norm)),
                    (n("attn_q.weight"), Some(l.attn().wq)),
                    (n("attn_k.weight"), Some(l.attn().wk)),
                    (n("attn_v.weight"), Some(l.attn().wv)),
                    (n("attn_q.bias"), l.attn().bq),
                    (n("attn_k.bias"), l.attn().bk),
                    (n("attn_v.bias"), l.attn().bv),
                    (n("attn_q_norm.weight"), l.attn().q_norm),
                    (n("attn_k_norm.weight"), l.attn().k_norm),
                    (n("attn_output.weight"), Some(l.attn().wo)),
                    (n("ffn_norm.weight"), Some(l.ffn_norm)),
                    (n("ffn_gate.weight"), Some(gate)),
                    (n("ffn_up.weight"), Some(up)),
                    (n("ffn_down.weight"), Some(down)),
                ]);
            }
            for (name, w) in ws {
                let t = g.tensor(&name);
                assert_eq!(
                    w.is_some(),
                    t.is_some(),
                    "{arch:?} {name}: loaded iff in the file"
                );
                let (Some(w), Some(t)) = (w, t) else { continue };
                let mut got = vec![0u8; t.bytes as usize];
                s.download(&m.weights, w.off, &mut got).unwrap();
                let at = (g.data_offset + t.offset) as usize;
                assert!(
                    got == bytes[at..at + t.bytes as usize],
                    "{arch:?} {name}: bytes differ from the file's"
                );
            }
        }
    }

    /// The token embedding a step gathers on the host (CPU_COMPUTE, then one upload into x) is the
    /// device's gather of the same rows, byte for byte, and its table is not in the weights buffer;
    /// a default load keeps the table on the device (on a discrete CUDA device too).
    fn host_embedding_is_the_device_gather(ordinal: i32) {
        let bytes = tiny::gguf(Arch::Qwen35Moe, 3);
        let path = std::env::temp_dir().join(format!("larch-embd-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let dev = Device::open(ordinal).unwrap();
        let s = Stream::new(&dev).unwrap();
        let place = Placement::default();
        let load = |host| {
            Transformer::load_chunked(
                &path,
                &dev,
                &s,
                8,
                CHUNK,
                place,
                None,
                u64::MAX,
                host,
                false,
            )
        };
        let (on_host, on_dev) = (load(true).unwrap(), load(false).unwrap());
        // A default load keeps the device table, even where the engine's loader would not: a
        // caller that never runs embed_tokens reads its input from it.
        let default = Transformer::load(&path, &dev, &s, 8).unwrap();
        assert!(default.embed_and_head().is_some() && default.tok_embd.is_some());
        std::fs::remove_file(&path).unwrap();
        let table = Gguf::parse(&bytes)
            .unwrap()
            .tensor("token_embd.weight")
            .unwrap()
            .bytes;
        assert_eq!(on_dev.weights.bytes() - on_host.weights.bytes(), table);
        assert!(
            on_host.embed_and_head().is_none(),
            "lends a table it does not hold"
        );
        assert!(on_dev.embed_and_head().is_some());
        // No device lookup reaches the host table: it is an error, never an offset into the weights.
        assert!(on_host.tok_embd.is_none());
        assert_eq!(on_host.embed_table().err(), Some(KError::InvalidArg));
        let ids = [0i32, tiny::VOCAB as i32 - 1, 17, 17, 200, 3];
        let (n, d) = (ids.len() as i64, tiny::D as i64);
        let ib = Buffer::alloc(&dev, 4 * n as u64).unwrap();
        s.upload(&ib, 0, &ids).unwrap();
        let out = Buffer::alloc(&dev, DType::F32.bytes((n * d) as u64)).unwrap();
        let (it, ot) = (
            Tensor::new(&ib, 0, DType::I32, &[n]).unwrap(),
            Tensor::new(&out, 0, DType::F32, &[n, d]).unwrap(),
        );
        ops::embed(&s, &on_dev.embed_table().unwrap(), &it, &ot).unwrap();
        on_host.host_embed(&s, &ids).unwrap();
        let (mut want, mut got) = (
            vec![0u8; (4 * n * d) as usize],
            vec![0u8; (4 * n * d) as usize],
        );
        s.download(&out, 0, &mut want).unwrap();
        s.download(&on_host.sc.x, 0, &mut got).unwrap();
        assert!(want.iter().any(|&b| b != 0), "no rows gathered");
        assert!(got == want, "host rows differ from the device's gather");
        // A row past the table is refused, not read.
        assert_eq!(
            on_host.host_embed(&s, &[tiny::VOCAB as i32]),
            Err(KError::InvalidArg)
        );
    }

    #[test]
    fn a_host_embedding_is_the_device_gather_on_cpu() {
        host_embedding_is_the_device_gather(CPU);
    }

    /// [`host_embedding_is_the_device_gather`] on GPU 0 (discrete CUDA in practice):
    ///   cargo test -p larch-model --lib -- --ignored a_host_embedding_is_the_device_gather_on_a_gpu
    #[test]
    #[ignore = "needs a GPU"]
    fn a_host_embedding_is_the_device_gather_on_a_gpu() {
        host_embedding_is_the_device_gather(0);
    }

    /// The float matrices the model reads only as f16 side copies (qwen35moe's routers, the shared
    /// experts' gates, float alpha and beta) stay out of the weights buffer, and nothing reads them
    /// there: every other weight holds its file bytes at its moved offset. qwen3moe's routers stay,
    /// since the trainer reads them.
    #[test]
    fn side_only_floats_stay_off_the_device() {
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        for (arch, gone) in [
            (
                Arch::Qwen35Moe,
                &[
                    "blk.0.ssm_alpha.weight",
                    "blk.0.ffn_gate_inp.weight",
                    "blk.0.ffn_gate_inp_shexp.weight",
                    "blk.1.ffn_gate_inp.weight",
                    "blk.1.ffn_gate_inp_shexp.weight",
                ][..],
            ),
            (Arch::Qwen3Moe, &[][..]),
        ] {
            let bytes = tiny::gguf(arch, 3);
            let path = std::env::temp_dir()
                .join(format!("larch-side-{arch:?}-{}.gguf", std::process::id()));
            std::fs::write(&path, &bytes).unwrap();
            let place = Placement::default();
            let m = Transformer::load_chunked(
                &path,
                &dev,
                &s,
                8,
                CHUNK,
                place,
                None,
                u64::MAX,
                false,
                false,
            );
            std::fs::remove_file(&path).unwrap();
            let m = m.unwrap();
            let g = Gguf::parse(&bytes).unwrap();
            let c = ModelConfig::from_gguf(&g).unwrap();
            let names: Vec<&str> = side_only(&g, &c).map(|t| t.name.as_str()).collect();
            assert_eq!(names, gone, "{arch:?}");
            let off: u64 = side_only(&g, &c).map(|t| t.bytes).sum();
            let data = bytes.len() as u64 - g.data_offset;
            assert_eq!(m.weights.bytes(), data - off, "{arch:?}");
            let mut ws = vec![
                ("token_embd.weight".to_string(), m.tok_embd.unwrap()),
                ("output.weight".into(), m.output),
                ("output_norm.weight".into(), m.out_norm),
            ];
            for (i, l) in m.layers.iter().enumerate() {
                let n = |s: &str| format!("blk.{i}.{s}");
                let norm = if arch.hybrid() {
                    "post_attention_norm"
                } else {
                    "ffn_norm"
                };
                ws.extend([
                    (n("attn_norm.weight"), l.attn_norm),
                    (n(&format!("{norm}.weight")), l.ffn_norm),
                ]);
                match &l.mixer {
                    Mixer::Attn(a) => ws.extend(
                        [
                            ("attn_q.weight", Some(a.wq)),
                            ("attn_k.weight", Some(a.wk)),
                            ("attn_v.weight", Some(a.wv)),
                            ("attn_output.weight", Some(a.wo)),
                            ("attn_q_norm.weight", a.q_norm),
                            ("attn_k_norm.weight", a.k_norm),
                        ]
                        .into_iter()
                        .filter_map(|(s, w)| Some((n(s), w?))),
                    ),
                    Mixer::Gdn(q) => ws.extend([
                        (n("attn_qkv.weight"), q.qkv),
                        (n("attn_gate.weight"), q.z),
                        (n("ssm_alpha.weight"), q.alpha),
                        (n("ssm_beta.weight"), q.beta),
                        (n("ssm_conv1d.weight"), q.conv),
                        (n("ssm_dt.bias"), q.dt),
                        (n("ssm_a"), q.a),
                        (n("ssm_norm.weight"), q.norm),
                        (n("ssm_out.weight"), q.out),
                    ]),
                }
                let Ffn::Moe {
                    router,
                    raw,
                    gate,
                    up,
                    down,
                    shared,
                    ..
                } = l.ffn
                else {
                    panic!("{arch:?} layer {i} is dense");
                };
                assert!(router.side);
                assert_eq!(
                    raw.is_some(),
                    arch == Arch::Qwen3Moe,
                    "{arch:?}: raw router"
                );
                ws.extend(raw.map(|r| (n("ffn_gate_inp.weight"), r)));
                ws.extend([
                    (n("ffn_gate_exps.weight"), gate),
                    (n("ffn_up_exps.weight"), up),
                    (n("ffn_down_exps.weight"), down),
                ]);
                if let Some(sh) = shared {
                    assert!(sh.gate_inp.side);
                    ws.extend([
                        (n("ffn_gate_shexp.weight"), sh.gate),
                        (n("ffn_up_shexp.weight"), sh.up),
                        (n("ffn_down_shexp.weight"), sh.down),
                    ]);
                }
            }
            for (name, w) in ws {
                let t = g.tensor(&name).unwrap();
                assert_eq!(
                    w.side,
                    gone.contains(&name.as_str()),
                    "{arch:?} {name}: side"
                );
                if w.side {
                    continue;
                }
                let mut got = vec![0u8; t.bytes as usize];
                s.download(&m.weights, w.off, &mut got).unwrap();
                let at = (g.data_offset + t.offset) as usize;
                assert!(
                    got == bytes[at..at + t.bytes as usize],
                    "{arch:?} {name}: bytes differ from the file's"
                );
            }
            assert_eq!(m.train_view(0).unwrap().is_some(), arch == Arch::Qwen3Moe);
        }
    }

    /// Past the buffer limit the upload splits at a tensor as the wraps do, the bytes where the
    /// halves overlap in both: every device weight reads what one buffer holds, host experts or not.
    #[test]
    fn a_split_upload_holds_every_weight() {
        let path = std::env::temp_dir().join(format!("larch-split-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        for layers in [0, 1] {
            let place = Placement {
                cpu_expert_layers: layers,
                ..Default::default()
            };
            let load = |cap| {
                Transformer::load_chunked(&path, &dev, &s, 1, 4096, place, None, cap, false, false)
            };
            let (one, two) = (load(u64::MAX).unwrap(), load(1).unwrap());
            assert!(one.hi.is_none());
            let at = two.hi.as_ref().unwrap().1;
            assert!(
                at > 0 && at < one.weights.bytes(),
                "{layers}: split at {at}"
            );
            let device = |m: &Transformer| -> Vec<W> {
                let mut ws = vec![m.tok_embd.unwrap(), m.out_norm, m.output];
                for l in &m.layers {
                    ws.extend(l.linears());
                    if let Ffn::Moe {
                        gate,
                        up,
                        down,
                        host: false,
                        ..
                    } = l.ffn
                    {
                        ws.extend([gate, up, down]);
                    }
                }
                ws.retain(|w| !w.side);
                ws
            };
            let read = |m: &Transformer, w: W| {
                let t = m.t(w).unwrap();
                let n = t.shape().iter().product::<i64>() as u64;
                let mut got = vec![0u8; t.dtype().bytes(n) as usize];
                s.download(t.buffer(), t.offset(), &mut got).unwrap();
                got
            };
            let (a, b) = (device(&one), device(&two));
            assert_eq!(a.len(), b.len());
            assert!(
                b.iter().any(|w| w.off >= at),
                "{layers}: a weight in the upper half"
            );
            for (i, (&wa, &wb)) in a.iter().zip(&b).enumerate() {
                assert!(
                    read(&one, wa) == read(&two, wb),
                    "{layers} host layers: weight {i}"
                );
            }
        }
        std::fs::remove_file(&path).unwrap();
    }

    /// A step streams its prefill chunk with a pool, past the crossover, and neither a check step, a
    /// capture nor adapters on the CPU; a crossover of 0 never does.
    #[test]
    fn only_eligible_steps_stream() {
        let go = |from, pool, prefill, check, capturing, adapters| {
            streams_step(from, pool, prefill, check, capturing, adapters)
        };
        assert!(go(128, true, 128, false, false, false));
        assert!(go(128, true, 512, false, false, false));
        assert!(!go(128, true, 127, false, false, false), "below R");
        assert!(!go(0, true, 512, false, false, false), "R 0 never");
        assert!(!go(128, false, 512, false, false, false), "no pool");
        assert!(!go(128, true, 512, true, false, false), "check step");
        assert!(!go(128, true, 512, false, true, false), "capturing");
        assert!(!go(128, true, 512, false, false, true), "adapters");
        // Decode rows beside the chunk keep the CPU path (`moe_ffn`); the chunk still streams:
        // a_mixed_step_streams_its_prefill_rows_and_keeps_the_decode_rows_bits.
    }

    /// The host tensors' places in the weights: one tensor, two back to back, and none where a
    /// span starts off a tensor or crosses a gap in either.
    #[test]
    fn a_span_lies_in_the_host_weights_only_back_to_back() {
        let held = [(100, 40, 0), (140, 60, 40), (300, 50, 100), (350, 10, 200)];
        assert_eq!(held_at(&held, 100, 40), Some(0));
        assert_eq!(
            held_at(&held, 100, 100),
            Some(0),
            "two tensors back to back"
        );
        assert_eq!(held_at(&held, 140, 60), Some(40));
        assert_eq!(held_at(&held, 120, 20), None, "inside a tensor");
        assert_eq!(held_at(&held, 140, 210), None, "a gap in the file");
        assert_eq!(held_at(&held, 300, 60), None, "a gap in the weights");
        assert_eq!(held_at(&held, 350, 20), None, "past the last");
    }

    /// Five layers through one and two tensor slots, twice; failure leaves the pool usable. With
    /// a host source, a tensor with a place there comes from it, the others from the map.
    #[test]
    fn the_pool_delivers_every_tensor_in_order() {
        let path = std::env::temp_dir().join(format!("larch-pool-{}.gguf", std::process::id()));
        let bytes: Vec<u8> = (0..2048u32).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let map = Mmap::map(&File::open(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let slot = |b: &mut Vec<Buffer>| b.push(Buffer::alloc(&dev, 72).unwrap());
        // upload_map checks what it reads and where.
        let b = Buffer::alloc(&dev, 16).unwrap();
        for (off, range) in [(0, 2040..2056), (8, 0..16), (0, 8..8)] {
            assert_eq!(
                s.upload_map(&b, off, &map, range),
                Err(KError::InvalidArg),
                "{off}"
            );
        }
        let layers: Vec<StreamLayer> = (0..5)
            .map(|i| {
                [
                    (i * 300, 40, 0, None),
                    (i * 300 + 40, 72, 0, None),
                    (i * 300 + 112, 16, 0, None),
                ]
            })
            .collect();
        // The host source holds each byte's complement, so a slot shows where it came from;
        // layers 0 to 2 have a place there (at file offset + 1), 3 and 4 none.
        let flipped: Vec<u8> = bytes.iter().map(|b| !b).collect();
        let held: Vec<[Option<u64>; 3]> = (layers.iter().enumerate())
            .map(|(i, l)| l.map(|p| (i < 3).then_some(p.0 + 1)))
            .collect();
        for (slots, host) in [(1, false), (2, false), (1, true), (2, true)] {
            let mut idle = Vec::new();
            for _ in 0..slots {
                slot(&mut idle);
            }
            let src = host.then(|| {
                let mut h = Buffer::alloc_host(&dev, 2049).unwrap();
                s.upload(&h, 1, &flipped).unwrap();
                (h.share_host().unwrap(), held.clone())
            });
            let plan = StreamPlan {
                layers: layers.clone(),
                slot_bytes: 72,
                slots,
                map: map.clone(),
                src,
            };
            let mut pool = StreamPool::spawn(&s, plan, idle, false).unwrap();
            let check = |i: usize, slot: &Buffer| {
                let (file, n, _, _) = layers[i / 3][i % 3];
                let mut got = vec![0u8; n as usize];
                s.download(slot, 0, &mut got).unwrap();
                let from = match host && i / 3 < 3 {
                    true => &flipped,
                    false => &bytes,
                };
                assert_eq!(got, from[file as usize..(file + n) as usize], "tensor {i}");
            };
            for _ in 0..2 {
                pool.begin(&s, 1).unwrap();
                for l in 0..layers.len() * 3 {
                    let slot = pool.take().unwrap().unwrap();
                    check(l, &slot);
                    pool.finish(l, slot, Ok(())).unwrap();
                }
                assert_eq!((pool.queued, pool.idle.len()), (0, slots));
            }
            pool.begin(&s, 1).unwrap();
            let slot = pool.take().unwrap().unwrap();
            assert_eq!(
                pool.finish(0, slot, Err(KError::Launch)),
                Err(KError::Launch)
            );
            assert!(pool.dirty);
            pool.begin(&s, 1).unwrap();
            let slot = pool.take().unwrap().unwrap();
            check(0, &slot);
            pool.finish(0, slot, Ok(())).unwrap();
            assert_eq!(pool.steps, 4);
        }
    }

    /// A forward pass of three prefill rows of one sequence, blocks 0..4 in order: its one logits row.
    fn prefill3(m: &mut Transformer, dev: &Device, s: &Stream) -> Result<Vec<f32>, KError> {
        let rows = 3i64;
        let ints: [(&[i32], Vec<i64>); 7] = [
            (&[5, 9, 2], vec![rows]),
            (&[0, 1, 2], vec![rows]),
            (&[0, 1, 2, 3], vec![1, 4]),
            (&[], vec![0]),
            (&[0], vec![1]),
            (&[3], vec![1]),
            (&[2], vec![1]),
        ];
        let bufs: Vec<Buffer> = ints
            .iter()
            .map(|(v, _)| {
                let b = Buffer::alloc(dev, (v.len() as u64 * 4).max(4)).unwrap();
                s.upload(&b, 0, v).unwrap();
                b
            })
            .collect();
        let t = |i: usize| Tensor::new(&bufs[i], 0, DType::I32, &ints[i].1).unwrap();
        let batch = StepBatch {
            token_ids: t(0),
            positions: t(1),
            slot_ids: t(1),
            decode_rows: 0,
            prefill_rows: rows as u32,
            block_table: t(2),
            seq_lens: t(3),
            prefill_q_start: t(4),
            prefill_q_len: t(5),
            logits_rows: t(6),
            attn_splits: 0,
            check: false,
        };
        let kv_bytes = u64::from(tiny::LAYERS) * 4 * 16 * tiny::KV_HEADS * tiny::HEAD_DIM * 2;
        let (k, v) = (
            Buffer::alloc(dev, kv_bytes).unwrap(),
            Buffer::alloc(dev, kv_bytes).unwrap(),
        );
        let kv = KvPool {
            k: &k,
            v: &v,
            n_blocks: 4,
            block_tokens: 16,
            n_layers: tiny::LAYERS,
            n_kv_heads: tiny::KV_HEADS as u32,
            head_dim: tiny::HEAD_DIM as u32,
            dtype: DType::F16,
        };
        let out = Buffer::alloc(dev, u64::from(tiny::VOCAB) * 4).unwrap();
        let logits = Tensor::new(&out, 0, DType::F32, &[1, i64::from(tiny::VOCAB)]).unwrap();
        m.forward(s, &batch, &kv, &logits)?;
        let mut got = vec![0f32; tiny::VOCAB as usize];
        s.download(&out, 0, &mut got)?;
        Ok(got)
    }

    /// One step on `kv` from its integer inputs (token ids, positions, KV slots, block table, decode
    /// lengths, prefill starts and lengths, logits rows) with `nd` decode rows: its logits rows.
    fn run_step(
        m: &mut Transformer,
        dev: &Device,
        s: &Stream,
        kv: &KvPool,
        ints: &[(Vec<i32>, Vec<i64>); 8],
        nd: u32,
    ) -> Result<Vec<f32>, KError> {
        let bufs: Vec<Buffer> = ints
            .iter()
            .map(|(v, _)| {
                let b = Buffer::alloc(dev, (v.len() as u64 * 4).max(4)).unwrap();
                s.upload(&b, 0, v).unwrap();
                b
            })
            .collect();
        let t = |i: usize| Tensor::new(&bufs[i], 0, DType::I32, &ints[i].1).unwrap();
        let rows = ints[0].0.len() as u32;
        let batch = StepBatch {
            token_ids: t(0),
            positions: t(1),
            slot_ids: t(2),
            decode_rows: nd,
            prefill_rows: rows - nd,
            block_table: t(3),
            seq_lens: t(4),
            prefill_q_start: t(5),
            prefill_q_len: t(6),
            logits_rows: t(7),
            attn_splits: 0,
            check: false,
        };
        let n = ints[7].0.len();
        let out = Buffer::alloc(dev, (n * tiny::VOCAB as usize * 4) as u64)?;
        let logits = Tensor::new(&out, 0, DType::F32, &[n as i64, i64::from(tiny::VOCAB)])?;
        m.forward(s, &batch, kv, &logits)?;
        let mut got = vec![0f32; n * tiny::VOCAB as usize];
        s.download(&out, 0, &mut got)?;
        Ok(got)
    }

    /// Sequence A's 3-token prefill (block 0), then A's decode row (position 3) and sequence B's
    /// 3-token prefill (block 1): in one mixed step (`mixed`), else B's prefill alone and then A's
    /// decode alone. The logits rows, A's then B's last.
    fn after_prefill(
        m: &mut Transformer,
        dev: &Device,
        s: &Stream,
        mixed: bool,
    ) -> Result<Vec<f32>, KError> {
        let kv_bytes = u64::from(tiny::LAYERS) * 4 * 16 * tiny::KV_HEADS * tiny::HEAD_DIM * 2;
        let (k, v) = (Buffer::alloc(dev, kv_bytes)?, Buffer::alloc(dev, kv_bytes)?);
        let kv = KvPool {
            k: &k,
            v: &v,
            n_blocks: 4,
            block_tokens: 16,
            n_layers: tiny::LAYERS,
            n_kv_heads: tiny::KV_HEADS as u32,
            head_dim: tiny::HEAD_DIM as u32,
            dtype: DType::F16,
        };
        let i = |v: &[i32], shape: &[i64]| (v.to_vec(), shape.to_vec());
        let a = [
            i(&[5, 9, 2], &[3]),
            i(&[0, 1, 2], &[3]),
            i(&[0, 1, 2], &[3]),
            i(&[0, 1, 2, 3], &[1, 4]),
            i(&[], &[0]),
            i(&[0], &[1]),
            i(&[3], &[1]),
            i(&[2], &[1]),
        ];
        run_step(m, dev, s, &kv, &a, 0)?;
        if mixed {
            let both = [
                i(&[7, 4, 8, 1], &[4]),
                i(&[3, 0, 1, 2], &[4]),
                i(&[3, 16, 17, 18], &[4]),
                i(&[0, 2, 3, 0, 1, 2, 3, 0], &[2, 4]),
                i(&[4], &[1]),
                i(&[0], &[1]),
                i(&[3], &[1]),
                i(&[0, 3], &[2]),
            ];
            return run_step(m, dev, s, &kv, &both, 1);
        }
        let b = [
            i(&[4, 8, 1], &[3]),
            i(&[0, 1, 2], &[3]),
            i(&[16, 17, 18], &[3]),
            i(&[1, 2, 3, 0], &[1, 4]),
            i(&[], &[0]),
            i(&[0], &[1]),
            i(&[3], &[1]),
            i(&[2], &[1]),
        ];
        let b = run_step(m, dev, s, &kv, &b, 0)?;
        let a = [
            i(&[7], &[1]),
            i(&[3], &[1]),
            i(&[3], &[1]),
            i(&[0, 2, 3, 0], &[1, 4]),
            i(&[4], &[1]),
            i(&[], &[0]),
            i(&[], &[0]),
            i(&[0], &[1]),
        ];
        Ok([run_step(m, dev, s, &kv, &a, 1)?, b].concat())
    }

    /// A step of decode rows beside a prefill chunk past the crossover streams the chunk's host
    /// experts and keeps the decode rows on the CPU path: the same bits as the CPU path for every
    /// row (m13 W4@4 on the 5070 Ti: such a step took 578 ms on the CPU path, 187 ms streamed alone).
    #[test]
    fn a_mixed_step_streams_its_prefill_rows_and_keeps_the_decode_rows_bits() {
        let path = std::env::temp_dir().join(format!("larch-mixed-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let load = |stream_rows| {
            let place = Placement {
                cpu_expert_layers: tiny::LAYERS,
                stream_rows,
            };
            Transformer::load_placed(&path, &dev, &s, 8, place).unwrap()
        };
        let pooled = || {
            let mut m = load(3);
            assert!(Model::alloc_stream_pool(&mut m, &s).unwrap());
            m
        };
        // Apart: B's chunk streams alone, then A's decode row runs on the CPU path alone.
        let want = after_prefill(&mut pooled(), &dev, &s, false).unwrap();
        assert!(want.iter().all(|x| x.is_finite()) && want.iter().any(|&x| x != 0.0));
        let mut m = pooled();
        assert_eq!(
            after_prefill(&mut m, &dev, &s, true).unwrap(),
            want,
            "each row's bits"
        );
        // The decode row's are the CPU path's, as with no pool at all.
        let cpu = after_prefill(&mut load(0), &dev, &s, true).unwrap();
        let v = tiny::VOCAB as usize;
        assert_eq!(cpu[..v], want[..v]);
        // A's 3-row prefill and the mixed step's 3 prefill rows both reach the crossover.
        assert_eq!(
            m.streamed_steps(),
            2,
            "the mixed step streamed its prefill rows"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// Two prompts packed into one streamed chunk give the logits of each prefilled alone, one
    /// after the other: per-sequence KV and positions stay exact, and the packed chunk streams each
    /// host layer's experts once instead of once a prompt.
    #[test]
    fn two_packed_prompts_give_the_logits_of_one_after_the_other() {
        let path = std::env::temp_dir().join(format!("larch-packed-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let run = |packed: bool| {
            let place = Placement {
                cpu_expert_layers: tiny::LAYERS,
                stream_rows: 3,
            };
            let mut m = Transformer::load_placed(&path, &dev, &s, 8, place).unwrap();
            assert!(Model::alloc_stream_pool(&mut m, &s).unwrap());
            let kv_bytes = u64::from(tiny::LAYERS) * 4 * 16 * tiny::KV_HEADS * tiny::HEAD_DIM * 2;
            let (k, v) = (
                Buffer::alloc(&dev, kv_bytes).unwrap(),
                Buffer::alloc(&dev, kv_bytes).unwrap(),
            );
            let kv = KvPool {
                k: &k,
                v: &v,
                n_blocks: 4,
                block_tokens: 16,
                n_layers: tiny::LAYERS,
                n_kv_heads: tiny::KV_HEADS as u32,
                head_dim: tiny::HEAD_DIM as u32,
                dtype: DType::F16,
            };
            let i = |v: &[i32], shape: &[i64]| (v.to_vec(), shape.to_vec());
            let one = |ids: &[i32], block: i32| {
                let slots: Vec<i32> = (0..3).map(|p| block * 16 + p).collect();
                [
                    i(ids, &[3]),
                    i(&[0, 1, 2], &[3]),
                    i(&slots, &[3]),
                    i(&[block, 2, 3, 0], &[1, 4]),
                    i(&[], &[0]),
                    i(&[0], &[1]),
                    i(&[3], &[1]),
                    i(&[2], &[1]),
                ]
            };
            let got = if packed {
                let both = [
                    i(&[5, 9, 2, 4, 8, 1], &[6]),
                    i(&[0, 1, 2, 0, 1, 2], &[6]),
                    i(&[0, 1, 2, 16, 17, 18], &[6]),
                    i(&[0, 2, 3, 0, 1, 2, 3, 0], &[2, 4]),
                    i(&[], &[0]),
                    i(&[0, 0], &[2]),
                    i(&[3, 3], &[2]),
                    i(&[2, 5], &[2]),
                ];
                run_step(&mut m, &dev, &s, &kv, &both, 0).unwrap()
            } else {
                let a = run_step(&mut m, &dev, &s, &kv, &one(&[5, 9, 2], 0), 0).unwrap();
                let b = run_step(&mut m, &dev, &s, &kv, &one(&[4, 8, 1], 1), 0).unwrap();
                [a, b].concat()
            };
            (got, m.streamed_steps())
        };
        let (apart, steps) = run(false);
        let (packed, packed_steps) = run(true);
        assert_eq!(packed, apart);
        assert_eq!(
            (steps, packed_steps),
            (2, 1),
            "one streamed pass for both prompts"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// A copy that runs out of memory costs the pool, not the request: the layer and every later
    /// one run on the CPU device, and so do later steps. A lost device still fails the step.
    #[test]
    fn a_copy_out_of_memory_moves_the_step_to_the_cpu_path() {
        let path = std::env::temp_dir().join(format!("larch-poolfail-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let load = |stream_rows| {
            let place = Placement {
                cpu_expert_layers: tiny::LAYERS,
                stream_rows,
            };
            Transformer::load_placed(&path, &dev, &s, 8, place).unwrap()
        };
        let pooled = |fail| {
            let mut m = load(1);
            assert!(Model::alloc_stream_pool(&mut m, &s).unwrap());
            m.stream.as_ref().unwrap().borrow_mut().fail = Some(fail);
            m
        };
        let disabled = |m: &Transformer| m.stream.as_ref().unwrap().borrow().disabled;
        let want = prefill3(&mut load(0), &dev, &s).unwrap();
        assert!(want.iter().all(|x| x.is_finite()) && want.iter().any(|&x| x != 0.0));
        // The first layer's copy fails: no layer ran on the device, so the bits are the CPU path's.
        let mut m = pooled((0, KError::Oom));
        assert_eq!(prefill3(&mut m, &dev, &s).unwrap(), want);
        assert!(disabled(&m));
        assert_eq!(m.streamed_steps(), 1);
        assert_eq!(prefill3(&mut m, &dev, &s).unwrap(), want, "the next step");
        assert_eq!(m.streamed_steps(), 1, "a disabled pool does not stream");
        // A failure after gate or up recomputes that layer on the CPU path.
        for tensor in [1, 2] {
            let mut m = pooled((tensor, KError::Oom));
            assert_eq!(prefill3(&mut m, &dev, &s).unwrap(), want);
            assert!(disabled(&m));
        }
        // Anything else is the device's: it propagates and the pool stays.
        let mut m = pooled((0, KError::DeviceLost));
        assert_eq!(prefill3(&mut m, &dev, &s), Err(KError::DeviceLost));
        assert!(!disabled(&m));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_stream_slot_holds_one_expert_tensor() {
        let path = std::env::temp_dir().join(format!("larch-slot-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let m = Transformer::load_placed(
            &path,
            &dev,
            &s,
            64,
            Placement {
                cpu_expert_layers: tiny::LAYERS,
                stream_rows: 1,
            },
        )
        .unwrap();
        let p = m.stream_plan.as_ref().unwrap();
        let largest = p
            .layers
            .iter()
            .flat_map(|layer| layer.iter().map(|piece| piece.1))
            .max()
            .unwrap();
        assert_eq!(p.slot_bytes, largest);
        assert_eq!(Model::stream_pool_want(&m).0, largest + COPY_STREAM_BYTES);
        std::fs::remove_file(path).unwrap();
    }

    /// A real streamed pass leaves `streaming` set; a replay's host piece after it still runs on the
    /// CPU device, the bytes of a model that never streamed.
    #[test]
    fn run_host_stays_on_the_cpu_path_after_a_streamed_step() {
        let path = std::env::temp_dir().join(format!("larch-runhost-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let load = |stream_rows| {
            let place = Placement {
                cpu_expert_layers: tiny::LAYERS,
                stream_rows,
            };
            Transformer::load_placed(&path, &dev, &s, 8, place).unwrap()
        };
        let (mut streamed, cpu) = (load(1), load(0));
        std::fs::remove_file(&path).unwrap();
        assert!(Model::alloc_stream_pool(&mut streamed, &s).unwrap());

        let (rows, d) = (3i64, tiny::D as i64);
        prefill3(&mut streamed, &dev, &s).unwrap();
        assert!(streamed.streaming, "the pass must leave the flag set");
        assert_eq!(streamed.streamed_steps(), 1);

        // The route and h forward leaves, the same on both models.
        let m = streamed.moe.unwrap();
        let at = DType::F32.bytes((rows * i64::from(m.experts)) as u64);
        let ids: [i32; 6] = [0, 1, 2, 3, 1, 3];
        let hv: Vec<u16> = (0..rows * d)
            .map(|i| f16_bits((i * 7919 % 199) as f32 / 99.0 - 1.0))
            .collect();
        let piece = |model: &Transformer| {
            let route = model.sc.route.as_ref().unwrap();
            s.upload(route, at, &ids).unwrap();
            s.upload(route, at + 24, &[0.5f32; 6]).unwrap();
            s.upload(route, at + 48, &hv).unwrap();
            Model::run_host(model, &s, host_tag(1, rows, false), rows as u32).unwrap();
            let mut got = vec![0u8; DType::F16.bytes((rows * d) as u64) as usize];
            s.download(&model.sc.t, 0, &mut got).unwrap();
            got
        };
        let got = piece(&streamed);
        assert!(got.iter().any(|&b| b != 0), "no output");
        assert!(got == piece(&cpu), "run_host left the CPU path");
        assert_eq!(streamed.streamed_steps(), 1);
    }

    /// A load with an expert plan has no pool, whatever its [`Placement`] asks: its split layers
    /// run their host experts on the CPU device in a step that would stream.
    #[test]
    fn a_split_layer_stays_on_the_cpu_path() {
        let path = std::env::temp_dir().join(format!("larch-splitcpu-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let place = Placement {
            stream_rows: 1,
            ..Default::default()
        };
        let plan = ExpertPlan {
            resident: vec![vec![0, 1], vec![3]],
            dynamic: false,
        };
        let mut m = Transformer::load_chunked(
            &path,
            &dev,
            &s,
            8,
            CHUNK,
            place,
            Some(&plan),
            u64::MAX,
            false,
            false,
        )
        .unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(Model::stream_pool_want(&m), (0, 0));
        assert!(!Model::alloc_stream_pool(&mut m, &s).unwrap());
        prefill3(&mut m, &dev, &s).unwrap();
        assert_eq!(m.streamed_steps(), 0);
        let (hits, misses) = m.expert_hits().unwrap();
        assert!(hits > 0 && misses > 0, "{hits} hits, {misses} misses");
    }

    /// Replays skip the resident layers' log, so counting their host pieces would skew host layers
    /// against resident ones.
    #[test]
    fn a_replayed_host_piece_is_not_counted() {
        let path = std::env::temp_dir().join(format!("larch-replay-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let (dev, hist) = (Device::open(CPU).unwrap(), Histogram::path(&path));
        let s = Stream::new(&dev).unwrap();
        let place = Placement {
            cpu_expert_layers: tiny::LAYERS,
            ..Default::default()
        };
        let mut m = Transformer::load_placed(&path, &dev, &s, 8, place).unwrap();
        let h = Histogram::uniform(0, tiny::LAYERS, tiny::EXPERTS);
        m.record_routes(&s, hist.clone(), h).unwrap();
        let route = m.sc.route.as_ref().unwrap();
        s.upload(route, 0, &vec![0u8; route.bytes() as usize])
            .unwrap();
        Model::run_host(&m, &s, host_tag(0, 2, false), 2).unwrap();
        let counted = m
            .cache
            .as_ref()
            .unwrap()
            .borrow()
            .hist
            .counts
            .iter()
            .sum::<u64>();
        assert_eq!(counted, 0);
        drop(m);
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&hist).unwrap();
    }

    /// [`Transformer::host_experts`] by tag against the eager section, on every host layer's
    /// bytes: the output t, and the residual x a split layer's device part joins.
    fn host_pieces_are_eager(m: &Transformer, s: &Stream) {
        let (moe, route) = (m.moe.unwrap(), m.sc.route.as_ref().unwrap());
        let d = i64::from(m.cfg.d_model);
        let (e, k) = (i64::from(moe.experts), i64::from(moe.used));
        let read = |rows: i64| {
            let mut t = vec![0u8; DType::F16.bytes((rows * d) as u64) as usize];
            let mut x = vec![0u8; DType::F32.bytes((rows * d) as u64) as usize];
            s.download(&m.sc.t, 0, &mut t).unwrap();
            s.download(&m.sc.x, 0, &mut x).unwrap();
            (t, x)
        };
        let zero = |rows: i64| {
            s.upload(
                &m.sc.t,
                0,
                &vec![0u8; DType::F16.bytes((rows * d) as u64) as usize],
            )
            .unwrap();
            s.upload(
                &m.sc.x,
                0,
                &vec![0u8; DType::F32.bytes((rows * d) as u64) as usize],
            )
            .unwrap();
        };
        for (layer, rows, live) in [(1, 3i64, 1i64), (0, 8, 6)] {
            let Ffn::Moe {
                router,
                gate,
                up,
                down,
                host: true,
                ..
            } = m.layers[layer].ffn
            else {
                panic!("layer {layer} keeps its experts on the device");
            };
            let hv: Vec<u16> = (0..rows * d)
                .map(|i| f16_bits(((i * 7919 + layer as i64) % 199) as f32 / 99.0 - 1.0))
                .collect();
            // h where forward puts it, for `n` rows.
            let h_of = |n: i64| {
                let at = DType::F32.bytes((n * (e + 2 * k)) as u64);
                s.upload(route, at, &hv[..(n * d) as usize]).unwrap();
                Tensor::new(route, at, DType::F16, &[n, d]).unwrap()
            };
            // An eager section over `n` rows: the experts' output and the residual.
            let eager = |n: i64| {
                zero(rows);
                let h = h_of(n);
                let t = Tensor::new(&m.sc.t, 0, DType::F16, &[n, d]).unwrap();
                let x = Tensor::new(&m.sc.x, 0, DType::F32, &[n, d]).unwrap();
                m.moe_ffn(
                    s,
                    layer,
                    router,
                    [gate, up, down],
                    None,
                    true,
                    &h,
                    &t,
                    &x,
                    None,
                )
                .unwrap();
                read(n)
            };
            // What a replay's graph piece leaves: `rows` rows' route, then h.
            let piece = |live: i64| {
                zero(rows);
                let h = h_of(rows);
                let logits = Tensor::new(route, 0, DType::F32, &[rows, e]).unwrap();
                let at = DType::F32.bytes((rows * e) as u64);
                let (ids, weights) = route_of(route, at, rows, k).unwrap();
                m.gemm(s, &h, &m.t(router).unwrap(), None, &logits).unwrap();
                ops::moe_route(s, &logits, 0.0, &ids, &weights).unwrap();
                Model::run_host(m, s, host_tag(layer, rows, false), live as u32).unwrap();
                read(rows)
            };
            let want = eager(live);
            let all = eager(rows);
            assert!(
                all.0.iter().chain(&all.1).any(|&b| b != 0),
                "layer {layer}: no output"
            );
            assert!(piece(rows) == all, "layer {layer}, {rows} rows");
            let (t, x) = piece(live);
            let (t, t_pad) = t.split_at(want.0.len());
            let (x, x_pad) = x.split_at(want.1.len());
            assert!(t == want.0, "layer {layer}, {live} of {rows} rows: t");
            assert!(x == want.1, "layer {layer}, {live} of {rows} rows: x");
            assert!(
                t_pad.iter().chain(x_pad).all(|&b| b == 0),
                "layer {layer}: padding computed"
            );
        }
    }

    /// What a replay's graph piece leaves for a one-row step of two routed experts.
    fn one_row_route(m: &Transformer, s: &Stream, ids: [i32; 2]) {
        let (moe, route) = (m.moe.unwrap(), m.sc.route.as_ref().unwrap());
        assert_eq!(moe.used, 2, "two routes per row");
        let at = DType::F32.bytes(u64::from(moe.experts));
        s.upload(route, at, &ids).unwrap();
        s.upload(route, at + 8, &[0.75f32, 0.25]).unwrap();
        s.upload(route, at + 16, &vec![f16_bits(0.5); m.cfg.d_model as usize])
            .unwrap();
    }

    /// A split layer's step with every expert resident counts as one with no host row, a replayed
    /// one too; one host expert in the step does not.
    #[test]
    fn a_step_with_no_host_row_is_counted() {
        let path = std::env::temp_dir().join(format!("larch-nohost-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let plan = ExpertPlan {
            resident: vec![vec![0, 1], vec![3]],
            dynamic: false,
        };
        let m = Transformer::load_cached(&path, &dev, &s, 8, &plan).unwrap();
        std::fs::remove_file(&path).unwrap();
        let step = |ids: [i32; 2]| {
            one_row_route(&m, &s, ids);
            Model::run_host(&m, &s, host_tag(0, 1, false), 1).unwrap();
            let c = m.split_cache().unwrap().borrow();
            (c.no_host, c.split_steps)
        };
        assert_eq!(step([0, 1]), (1, 1));
        assert_eq!(step([1, 0]), (2, 2));
        assert_eq!(step([0, 2]), (2, 3), "expert 2 is on the host");
        assert_eq!(step([2, 3]), (2, 4), "both on the host");
    }

    #[test]
    fn a_host_tag_round_trips_layer_rows_and_check() {
        for (layer, rows, check) in [(0, 1, false), (7, 64, true), (0x7fff_ffff, 5, true)] {
            assert_eq!(untag(host_tag(layer, rows, check)), (layer, rows, check));
        }
    }

    /// A replay teaches the histogram only when no layer waits on the route log: on an all-split
    /// load every layer counts from its own download, while in a mixed plan the resident layers'
    /// routes go unlogged on a replay, so the split ones must not lead them.
    #[test]
    fn a_replay_counts_routes_only_on_an_all_split_load() {
        let path = std::env::temp_dir().join(format!("larch-rcount-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let (dev, hist) = (Device::open(CPU).unwrap(), Histogram::path(&path));
        let s = Stream::new(&dev).unwrap();
        for (resident, want) in [
            (vec![vec![0, 1], vec![3]], 2u64),
            (vec![vec![0, 1], vec![0, 1, 2, 3]], 0),
        ] {
            let plan = ExpertPlan {
                resident,
                dynamic: false,
            };
            let mut m = Transformer::load_cached(&path, &dev, &s, 8, &plan).unwrap();
            let h = Histogram::uniform(0, tiny::LAYERS, tiny::EXPERTS);
            m.record_routes(&s, hist.clone(), h).unwrap();
            one_row_route(&m, &s, [0, 2]);
            Model::run_host(&m, &s, host_tag(0, 1, false), 1).unwrap();
            let counted = m
                .split_cache()
                .unwrap()
                .borrow()
                .hist
                .counts
                .iter()
                .sum::<u64>();
            assert_eq!(counted, want, "{plan:?}");
        }
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&hist).unwrap();
    }

    /// A replay's host piece, run by its tag after an eager host section on CPU_COMPUTE experts,
    /// writes the experts' output the eager section wrote, byte for byte, from the device's route
    /// and `h` alone. With fewer live rows than the tag's (a padded bucket) it writes the live
    /// rows' output of an eager step of just those rows, as the engine runs one, and nothing else:
    /// CPU_COMPUTE groups 8 rows' pairs by expert and 6 rows' not, which round otherwise. A split
    /// layer's piece is that too, its device part included.
    #[test]
    fn a_host_piece_by_tag_is_the_eager_host_section() {
        let path = std::env::temp_dir().join(format!("larch-hostpc-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let place = Placement {
            cpu_expert_layers: tiny::LAYERS,
            ..Default::default()
        };
        let m = Transformer::load_placed(&path, &dev, &s, 8, place).unwrap();
        let plan = ExpertPlan {
            resident: vec![vec![0, 1], vec![3]],
            dynamic: false,
        };
        let split = Transformer::load_cached(&path, &dev, &s, 8, &plan).unwrap();
        std::fs::remove_file(&path).unwrap();
        host_pieces_are_eager(&m, &s);
        host_pieces_are_eager(&split, &s);
        let (hits, misses) = split.expert_hits().unwrap();
        assert!(hits > 0 && misses > 0, "{hits} hits, {misses} misses");
        let (moe, route) = (m.moe.unwrap(), m.sc.route.as_ref().unwrap());
        let d = i64::from(m.cfg.d_model);
        // The host section writes the scratch t and x at their start itself: any other `out` or
        // `resid` is refused, not silently left unwritten; those two run.
        let Ffn::Moe {
            router,
            gate,
            up,
            down,
            ..
        } = m.layers[1].ffn
        else {
            panic!("layer 1 is not MoE");
        };
        let at = DType::F32.bytes(u64::from(moe.experts + 2 * moe.used));
        let h = Tensor::new(route, at, DType::F16, &[1, d]).unwrap();
        s.upload(route, at, &vec![f16_bits(0.5); d as usize])
            .unwrap();
        let t = Tensor::new(&m.sc.t, 0, DType::F16, &[1, d]).unwrap();
        let x = Tensor::new(&m.sc.x, 0, DType::F32, &[1, d]).unwrap();
        let t1 = Tensor::new(&m.sc.t, t.row_bytes(), DType::F16, &[1, d]).unwrap();
        let x1 = Tensor::new(&m.sc.x, x.row_bytes(), DType::F32, &[1, d]).unwrap();
        let u = Tensor::new(&m.sc.up, 0, DType::F16, &[1, d]).unwrap();
        let cases = [
            (&t, &x, true),
            (&t1, &x, false),
            (&u, &x, false),
            (&t, &x1, false),
        ];
        for (i, (out, resid, ok)) in cases.into_iter().enumerate() {
            let ran = m.moe_ffn(
                &s,
                1,
                router,
                [gate, up, down],
                None,
                true,
                &h,
                out,
                resid,
                None,
            );
            assert_eq!(ran, ok.then_some(()).ok_or(KError::InvalidArg), "case {i}");
        }
    }
}
