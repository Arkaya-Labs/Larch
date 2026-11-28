//! Model fit (architecture 9.10), once the model is on the device. Everything shares what the device
//! reported free before the load: the model's weights and activation scratch (sized for every row the
//! live `max_batch_seqs` and `prefill_chunk_tokens` ranges allow), measured as the memory the load
//! took; initial serving buffers and the decode graphs' reserve, counted from their sizes;
//! headroom; and the KV pool, which gets `kv_pool_fraction` of what is left and must hold one full
//! `max_context`, else loading refuses with `DoesNotFit`. Forced-run buffers grow on request using
//! current free memory and credit for the old buffers. Weights or scratch that do not fit at all
//! fail before this, in the loader, with the same error.

use larch_kernels::{Backend, DeviceProps, KError};
use larch_model::{Arch, DraftShape, ModelConfig, MoeCfg, MtpDrafter};
use larch_records::LoadError;

use crate::sched::BLOCK_TOKENS;

/// 512 MiB on discrete GPUs; 20 % of the total on unified memory (Apple) and the CPU reference.
const DISCRETE_HEADROOM: u64 = 512 << 20;

/// What a discrete CUDA card keeps free beside the hybrid MoE, in place of [`DISCRETE_HEADROOM`].
// ponytail: measured on the hybrid MoE target, 5070 Ti, W4@1: 117 MB peak over the plan (64 MiB arena and
// ~50 MB of lazy kernels and workspaces), rounded up to 112 MiB (117,440,512). The 67.4 MB
// free-reading swing is the check's (lib.rs FREE_SWING), not this term's: 192 MiB less 80 MiB.
// Raise it on an OOM on any gate, or when another card shows more; re-derive per model.
const HOST_PLAN_HEADROOM: u64 = 112 << 20;

// ponytail: the fit's estimate of one capture's device memory (a decode bucket or a mixed slot),
// needed before any capture exists: the largest measured, 4 MiB (`larch_graph_capture`, 5070 Ti
// sm_120, the hybrid MoE's bucket 4 in 9 pieces; bucket 1 took 2 MiB). The engine's quota charges
// each capture what it measures and refuses one over what is left, so an under-guess runs those
// shapes eager, never out of memory. Re-derive when a target card's capture line shows more.
pub(crate) const GRAPH_BYTES: u64 = 4 << 20;
// ponytail: the least quota a plan with every expert on the device grants when it grants one: its
// one-row decode capture, one piece over every layer, read 6,291,456 bytes (`larch_graph_capture`
// at bucket 1, 5070 Ti sm_120, the hybrid MoE's cold 0.3 and 0.4 files in 10 fresh loads), rounded
// up to whole slots; a one-seat load's single slot was short of it, so its one chat ran eager.
// Plans with host layers keep their slots: the standard file's 4 took 4 MiB in 5 pieces, 0.3's 1
// took 6 MiB in 2 within a 4-seat quota. Re-derive when a bucket-1 capture shows more, or a
// one-seat plan with host layers refuses its capture.
const ONE_ROW_GRAPH_BYTES: u64 = 2 * GRAPH_BYTES;
/// Matches the CUDA/HIP stream's lazy, reusable capture scratch arena; charged once per fit.
const GRAPH_ARENA_BYTES: u64 = 64 << 20;

/// The capture arena a CUDA or HIP stream allocates at its first capture.
pub(crate) fn graph_arena_reserve(props: &DeviceProps) -> u64 {
    if matches!(props.backend, Backend::Cuda | Backend::Hip)
        && props.flags & DeviceProps::CAP_GRAPHS != 0
    {
        GRAPH_ARENA_BYTES
    } else {
        0
    }
}

/// Four bounded mixed-shape slots on CUDA with graphs, when mixed graphs are enabled.
pub(crate) fn mixed_reserve(props: &DeviceProps, enabled: bool) -> u64 {
    if enabled && props.backend == Backend::Cuda && props.flags & DeviceProps::CAP_GRAPHS != 0 {
        // Four bounded slots; estimated exec metadata, separate from the shared scratch arena.
        4 * GRAPH_BYTES
    } else {
        0
    }
}

/// Graph memory for every decode bucket up to `max_seqs` rows: CUDA's exact 9 to 32 beside the
/// powers of two the other rows pad to, and on CUDA a guessing load's `checks` check-step slots
/// (0 for a plain load).
// ponytail: a check slot a check spare is a guess until `larch_graph_capture` lines show the check
// keys' real sizes; resize from them. An under-guess runs check shapes eager or evicts them for
// plain buckets, never out of memory.
pub(crate) fn graph_reserve(backend: Backend, max_seqs: u32, checks: u32) -> u64 {
    let buckets = if backend == Backend::Cuda {
        max_seqs.min(8).next_power_of_two().ilog2()
            + 1
            + max_seqs.min(32).saturating_sub(8)
            + max_seqs.next_power_of_two().ilog2().saturating_sub(5)
    } else {
        max_seqs.next_power_of_two().ilog2() + 1
    };
    let checks = if backend == Backend::Cuda { checks } else { 0 };
    u64::from(buckets + checks) * GRAPH_BYTES
}

/// The graph quota granted from `room` for a `desired` reserve, in whole slots: for an expert model
/// with no host layer (`whole`) raised to its one-row capture ([`ONE_ROW_GRAPH_BYTES`]) when the
/// room holds it; 0 stays 0.
pub(crate) fn graph_quota(desired: u64, room: u64, whole: bool) -> u64 {
    let floor = if desired != 0 && whole {
        ONE_ROW_GRAPH_BYTES
    } else {
        0
    };
    desired.max(floor).min(room / GRAPH_BYTES * GRAPH_BYTES)
}

/// What [`fit`] reserves for decode graphs when they all fit, as (slots, arena): the decode and
/// mixed slots for `max_seqs` seats and the capture arena unless the stream holds it; (0, 0) when
/// `graphs` is off at load or the device does not capture.
pub(crate) fn graph_need(
    before: &DeviceProps,
    max_seqs: u32,
    graphs: bool,
    arena_held: bool,
    checks: u32,
) -> (u64, u64) {
    if !graphs || before.flags & DeviceProps::CAP_GRAPHS == 0 {
        return (0, 0);
    }
    let slots = graph_reserve(before.backend, max_seqs, checks)
        + mixed_reserve(before, larch_kernels::routes().mixed_graphs);
    let arena = if arena_held {
        0
    } else {
        graph_arena_reserve(before)
    };
    (slots, arena)
}

/// Where the device's memory goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FitPlan {
    /// Weights and the model's activation scratch, as the load measured them; 0 on the CPU
    /// reference, whose free memory is a fixed figure.
    pub(crate) model_bytes: u64,
    /// Initial serving buffers; forced-run growth is checked separately.
    pub(crate) scratch_bytes: u64,
    /// Estimated reserve for the shared quota of decode and mixed executables, when decode_graphs
    /// is on at load.
    pub(crate) graph_bytes: u64,
    /// The capture arena the first capture allocates; 0 once the stream holds it (a reload).
    pub(crate) graph_arena_bytes: u64,
    /// The admitted streaming pool ([`larch_model::Model::stream_pool_want`]): 0 when it wants none
    /// or does not fit beside one full context and the graphs, and those steps stay on the CPU path.
    pub(crate) stream_pool_bytes: u64,
    /// Device headroom preserved beside the model, engine buffers, graphs and KV pool.
    pub(crate) headroom_bytes: u64,
    pub(crate) kv_block_bytes: u64,
    pub(crate) kv_blocks: u32,
    /// Full `max_context` sequences the pool holds.
    pub(crate) max_seqs_fit: u32,
    /// The graphs are the gate's floor: one seat's capture beside a pool for more seats, whose
    /// larger buckets run eager (`engine::gated_plan`).
    pub(crate) graph_floor: bool,
}

/// Bytes of one 16-token block, K and V, every layer that keeps KV, f16.
fn kv_block_bytes(c: &ModelConfig) -> u64 {
    2 * u64::from(c.kv_layers)
        * BLOCK_TOKENS as u64
        * u64::from(c.n_kv_heads)
        * u64::from(c.head_dim)
        * 2
}

/// Memory left to the OS and other apps: a fifth of a unified device's, less for any MoE on a
/// unified GPU, whose experts take the same RAM on the CPU as on the GPU: at 15 % a 30B MoE at 3 bits
/// runs all 48 layers on a 24 GB M5 with the KV pool it had with 3 on the CPU (measured 25 Oct 2026).
/// `mapped`: the weights are Metal's wrap of the file (no expert layer in host memory). Those pages
/// are clean file pages the OS can drop and read again, so they need no 15 % margin of their own:
/// 5 %, about what llama.cpp leaves when it maps the hybrid MoE target (run 11's erratic 5 % margin had host
/// layers, so copied, unpageable weights).
fn headroom(before: &DeviceProps, c: &ModelConfig, mapped: bool) -> u64 {
    let unified = before.backend == Backend::Cpu || before.flags & DeviceProps::CAP_UNIFIED != 0;
    let moe = matches!(c.arch, Arch::Qwen3Moe | Arch::Qwen35Moe);
    if unified && before.backend == Backend::Metal && moe && mapped {
        before.vram_total / 20
    } else if unified && before.backend != Backend::Cpu && moe {
        before.vram_total / 20 * 3
    } else if unified {
        before.vram_total / 5
    } else if before.backend == Backend::Cuda && matches!(c.arch, Arch::Qwen35Moe) {
        HOST_PLAN_HEADROOM
    } else {
        DISCRETE_HEADROOM
    }
}

/// KV blocks a discrete CUDA hybrid keeps for each seat the plan admits, whole host layers or not.
// ponytail: 75 blocks (1,200 tokens) from the hybrid MoE target, 5070 Ti, W4@4 at 6 layers: peak_kv_blocks_used
// 297 of 1275 for 4 chats, 74.25 a chat, rounded up (4 seats: 300). The 67.4 MB free-reading swing
// (205 blocks of the old 512-block floor) is the check's (lib.rs FREE_SWING). At 5 layers the same
// plan got 203 blocks, the pool filled and W4@4 fell from 96.0 to 84.9 tok/s. Re-derive per model,
// or when a workload with longer chats fills the pool.
const SEAT_PEAK_BLOCKS: u64 = 75;

/// The smallest pool of which `kv_pool_fraction` holds one full `max_context` (a discrete CUDA
/// hybrid: at least [`SEAT_PEAK_BLOCKS`] for each of `seats`). The division can round down onto a
/// byte that the pool's own rounding still rejects, so that byte is credited.
fn pool_min(
    before: &DeviceProps,
    c: &ModelConfig,
    max_context: u32,
    kv_pool_fraction: f32,
    seats: u32,
) -> u64 {
    let blocks = u64::from(max_context).div_ceil(BLOCK_TOKENS as u64);
    let floor = if before.backend == Backend::Cuda && matches!(c.arch, Arch::Qwen35Moe) {
        SEAT_PEAK_BLOCKS * u64::from(seats)
    } else {
        0
    };
    let full = blocks.max(floor) * kv_block_bytes(c);
    let mut pool_min = (full as f64 / f64::from(kv_pool_fraction)).ceil() as u64;
    pool_min += u64::from(((pool_min as f64 * f64::from(kv_pool_fraction)) as u64) < full);
    pool_min
}

/// What the fit after a load needs free beside the model: the headroom, `scratch` and the KV
/// floor for `seats` (one full `max_context` at least). A planned host-expert load checks it, so a
/// plan short of the model's real allocations moves one more layer instead of failing that fit.
pub(crate) fn serving_need(
    before: &DeviceProps,
    c: &ModelConfig,
    max_context: u32,
    kv_pool_fraction: f32,
    scratch: u64,
    mapped: bool,
    seats: u32,
) -> u64 {
    headroom(before, c, mapped)
        + scratch
        + pool_min(
            before,
            c,
            max_context.min(c.ctx_train),
            kv_pool_fraction,
            seats,
        )
}

/// An estimate of the Transformer's activation scratch for `rows` rows, before it loads: its
/// per-row buffers, and one f16 copy of the widest attention matrix for the dequant path.
pub(crate) fn activation_bytes(c: &ModelConfig, moe: Option<MoeCfg>, rows: u32) -> u64 {
    let (d, ffn) = (u64::from(c.d_model), u64::from(c.ffn));
    let qd = u64::from(c.n_heads) * u64::from(c.head_dim);
    let kvd = u64::from(c.n_kv_heads) * u64::from(c.head_dim);
    let (ffn, route) = moe.map_or((ffn, 0), |m| {
        let (e, k) = (u64::from(m.experts), u64::from(m.used));
        (k * u64::from(m.ffn), 4 * (e + 2 * k + d.div_ceil(2)))
    });
    let row = 4 * d + 2 * d + 4 * qd + 4 * kvd + 4 * d + 6 * ffn + route;
    u64::from(rows) * row + 2 * d * qd.max(d)
}

/// Metal's cap on the weighted down projection's f32 partials, kernels/src/metal/args.h's
/// `MOE_PARTIAL_BYTES`; a test keeps the two equal.
const MOE_PARTIAL_BYTES: u64 = 64 << 20;
/// What Metal's tensor path pads each expert-ordered x row to, args.h's `MOE_MM_K`.
const MOE_MM_K: u64 = 64;
/// MTLGPUFamilyApple10 (the M5), Metal's `arch_code` from which backend.mm can take the tensor path.
const METAL_TENSOR_FAMILY: u32 = 1010;

/// Metal's grouped mmvq_id scratch for `rows`, which the stream allocates in a step, after the load
/// measured free memory; sized as kernels/src/metal/backend.mm sizes it, with the overlap when it
/// grows. 0 on other backends.
pub(crate) fn grouped_scratch(props: &DeviceProps, moe: Option<MoeCfg>, d: u64, rows: u32) -> u64 {
    let (Backend::Metal, Some(m)) = (props.backend, moe) else {
        return 0;
    };
    let (e, k, rows) = (u64::from(m.experts), u64::from(m.used), u64::from(rows));
    let lists = |r: u64| (4 * (e + 1 + r * k)).next_multiple_of(16);
    // The tensor path's x rows in expert order; a GPU without the tensor API has none.
    let tensor = props.arch_code >= METAL_TENSOR_FAMILY;
    let xs = |r: u64, width: u64| u64::from(tensor) * 2 * r * k * width.next_multiple_of(MOE_MM_K);
    // The weighted down projection runs in blocks of rows whose f32 partials fit the cap, but at
    // least one row, which may pass it. Gate and up take every row at once.
    let r = rows.min((MOE_PARTIAL_BYTES / (4 * k * d)).max(1));
    let gate = lists(rows) + xs(rows, d);
    let down = lists(r) + (4 * r * k * d).next_multiple_of(16) + xs(r, u64::from(m.ffn));
    // Gate and up size the scratch first in a step. A larger down allocates a new buffer while
    // their queued work still holds the old one.
    if gate < down { gate + down } else { gate }
}

/// The serving buffers the fit after a load of `rows` rows charges: the engine's step buffers and
/// Metal's grouped expert scratch. The host-expert plan and the check after each load count the same.
pub(crate) fn serving_scratch(
    props: &DeviceProps,
    c: &ModelConfig,
    moe: Option<MoeCfg>,
    rows: u32,
    max_context: u32,
) -> u64 {
    crate::engine::Dims::of(c, rows as usize, max_context).fit_bytes()
        + grouped_scratch(props, moe, u64::from(c.d_model), rows)
}

/// What a load onto `before` may give the weights: free memory less the headroom (the smaller
/// one when the load maps the file), the KV floor for `seats` and `reserve`.
fn weights_room(
    before: &DeviceProps,
    c: &ModelConfig,
    (max_context, kv_pool_fraction, seats): (u32, f32, u32),
    reserve: u64,
    mapped: bool,
) -> u64 {
    before
        .vram_free
        .saturating_sub(headroom(before, c, mapped))
        .saturating_sub(pool_min(before, c, max_context, kv_pool_fraction, seats))
        .saturating_sub(reserve)
}

/// A DFlash drafter's device bytes past its weights (which the load measures with the model's):
/// `seats` caches of `max_context` positions and a block (a ring's: its window), every
/// batch row's taps of `rows` and their row ids. Its check steps' logits rows are within the step
/// buffers' first ones.
pub(crate) fn drafter_bytes(
    d: &DraftShape,
    d_model: u32,
    rows: usize,
    max_context: u32,
    seats: usize,
) -> u64 {
    let positions = u64::from(d.cache_positions(max_context));
    let taps = d.target_layers().len() as u64 * rows as u64 * u64::from(d_model);
    // Each seat's cache also holds its [16, pages] i32 table: 4 bytes a position.
    seats as u64 * (d.cache_bytes_per_position() + 4) * positions + 4 * taps + 4 * rows as u64
}

/// The target's MTP drafter's device bytes past the block's weights: `seats` caches of
/// `max_context` positions and its scratch, every batch row's taps of `rows` included.
pub(crate) fn mtp_bytes(c: &ModelConfig, rows: usize, max_context: u32, seats: usize) -> u64 {
    seats as u64 * MtpDrafter::cache_bytes(c, max_context)
        + MtpDrafter::scratch_bytes(c, rows, max_context)
}

/// What guessing takes on the device beside a load of the target, for the placement before it: the
/// check `spares`, a drafter's load and its [`drafter_bytes`], or the target's MTP block (`mtp`,
/// its weights' bytes) and its [`mtp_bytes`]. The block's routed experts leave the device with
/// the first whole host layer, which the plan counts in that layer's bytes.
pub(crate) fn guess_bytes(
    spares: u64,
    drafter: Option<&DraftShape>,
    mtp: Option<u64>,
    c: &ModelConfig,
    rows: u32,
    max_context: u32,
    seats: usize,
) -> u64 {
    let ctx = max_context.min(c.ctx_train);
    spares
        + drafter.map_or(0, |d| {
            d.load_bytes() + drafter_bytes(d, c.d_model, rows as usize, ctx, seats)
        })
        + mtp.map_or(0, |w| w + mtp_bytes(c, rows as usize, ctx, seats))
}

/// The (rows, draft seats) a load plans in turn before it moves expert layers: its own `rows` and
/// `seats`, then for a drafter (`drafting`) `small` rows (the 1024-token chunk's, when given and
/// fewer), then fewer draft seats down to one. A host layer turns off the file's map and its 15 %
/// headroom moves 5 to 7 layers: about 5 ms a step, the drafter's speed-up (24 GB M5, 15 Nov).
pub(crate) fn draft_fits(
    drafting: bool,
    rows: u32,
    small: Option<u32>,
    seats: u32,
) -> impl Iterator<Item = (u32, u32)> {
    let chunk = small.filter(|&r| r < rows).unwrap_or(rows);
    let fewer = (1..=seats).rev().map(move |k| (chunk, k));
    let fewer = fewer.filter(move |&t| drafting && t != (rows, seats));
    std::iter::once((rows, seats)).chain(fewer)
}

/// The (rows, draft seats) attempts of [`draft_fits`], those whose plan keeps the most state slots
/// (`slots` of each, 0 where it moves a host layer) first, in their own order on a tie. A chat that
/// waits for a slot costs more than a smaller prompt chunk or fewer draft seats (24 GB M5, 27 Nov:
/// the hybrid MoE target's drafter load took the 2048-row chunk and 1 state slot, and 8 chats ran
/// at 40.4 tok/s against 77.2 on the 8 slots of the load without it).
pub(crate) fn most_slots_first<E>(
    tries: impl Iterator<Item = (u32, u32)>,
    mut slots: impl FnMut(u32, u32) -> Result<u32, E>,
) -> Result<Vec<(u32, u32)>, E> {
    let mut ranked = Vec::new();
    for (rows, seats) in tries {
        ranked.push((std::cmp::Reverse(slots(rows, seats)?), rows, seats));
    }
    // Stable: a tie keeps draft_fits' order.
    ranked.sort_by_key(|r| r.0);
    Ok(ranked
        .into_iter()
        .map(|(_, rows, seats)| (rows, seats))
        .collect())
}

/// What a plan leaves the chats: host expert layers, a hybrid's state slots (0 for another
/// model) and KV blocks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Chats {
    pub(crate) host_layers: u32,
    pub(crate) state_slots: u32,
    pub(crate) kv_blocks: u32,
}

/// Whether an automatic drafter stays: its load (`with`) serves the chats the load `without` it
/// does, at no more host layers and no fewer state slots. Its KV blocks are its cost, so they
/// do not decide (24 GB M5, 15 Nov: 8 chats ran one after another on its 1 slot, 123.5 to 73.0
/// tok/s).
pub(crate) fn drafter_keeps(with: Chats, without: Chats) -> bool {
    with.host_layers <= without.host_layers && with.state_slots >= without.state_slots
}

/// [`larch_model::Placement::cpu_expert_layers`] for a load onto `before`: 0 when the `weights`
/// fit beside one full `max_context`, the mapped headroom and `reserve`; else the fewest leading layers whose
/// routed experts (`experts[i]` bytes) must stay in host memory so the rest fits; every layer
/// when even that is not enough, and the fit then refuses with `DoesNotFit`, as it would without
/// them. `seats` sizes the KV floor. On unified memory this is the GPU's working set; the CPU
/// reference keeps everything.
#[allow(clippy::too_many_arguments)] // the plan's inputs and the seats its KV floor holds
pub(crate) fn host_expert_layers(
    before: &DeviceProps,
    weights: u64,
    experts: &[u64],
    c: &ModelConfig,
    max_context: u32,
    kv_pool_fraction: f32,
    reserve: u64,
    seats: u32,
) -> u32 {
    // The fit clamps the context to the model's; so does the plan.
    let max_context = max_context.min(c.ctx_train);
    if before.backend == Backend::Cpu || max_context == 0 {
        return 0;
    }
    let budget = (max_context, kv_pool_fraction, seats);
    let room = |mapped| weights_room(before, c, budget, reserve, mapped);
    // With no host layer the load maps the file, which earns the smaller headroom.
    if weights <= room(true) {
        return 0;
    }
    let room = room(false);
    let mut need = weights;
    for (i, &e) in experts.iter().enumerate() {
        if need <= room {
            return i as u32;
        }
        need = need.saturating_sub(e);
    }
    experts.len() as u32
}

/// [`host_expert_layers`], or `None` where moving every layer still leaves the rest past the room
/// (`host_expert_layers` returns a count under `experts.len()` only where it fits, so the check
/// decides the all-layers case alone).
#[allow(clippy::too_many_arguments)] // host_expert_layers' inputs
pub(crate) fn host_expert_layers_that_fit(
    before: &DeviceProps,
    weights: u64,
    experts: &[u64],
    c: &ModelConfig,
    max_context: u32,
    kv_pool_fraction: f32,
    reserve: u64,
    seats: u32,
) -> Option<u32> {
    let (ctx, f) = (max_context, kv_pool_fraction);
    let n = host_expert_layers(before, weights, experts, c, ctx, f, reserve, seats);
    let left = weights.saturating_sub(experts[..n as usize].iter().sum());
    let budget = (max_context.min(c.ctx_train), kv_pool_fraction, seats);
    let fits =
        (n as usize) < experts.len() || left <= weights_room(before, c, budget, reserve, false);
    fits.then_some(n)
}

/// [`host_expert_layers`] for a plan that streams its host layers' experts through a `pool`-byte
/// prefill pool: a load that fits resident streams nothing and keeps the room; one that moves
/// layers moves those that leave room for the pool too, and keeps the plan without the pool when
/// moving every layer cannot cover it.
#[allow(clippy::too_many_arguments)] // host_expert_layers' inputs and the pool
pub(crate) fn host_expert_layers_with_pool(
    before: &DeviceProps,
    weights: u64,
    experts: &[u64],
    c: &ModelConfig,
    max_context: u32,
    kv_pool_fraction: f32,
    reserve: u64,
    seats: u32,
    pool: u64,
) -> u32 {
    let (ctx, f) = (max_context, kv_pool_fraction);
    let n = host_expert_layers(before, weights, experts, c, ctx, f, reserve, seats);
    if n == 0 || pool == 0 {
        return n;
    }
    let with = reserve + pool;
    host_expert_layers_that_fit(before, weights, experts, c, ctx, f, with, seats).unwrap_or(n)
}

/// A hybrid's recurrent state slots for a placement of `n` host layers: the most up to `cap` for
/// which the plan (`layers(slots)`, as [`host_expert_layers`]) still keeps `n`, at least one (the
/// check after the load gives up more when the device is short).
pub(crate) fn most_state_slots(cap: u32, n: u32, layers: impl Fn(u32) -> u32) -> u32 {
    (2..=cap)
        .rev()
        .find(|&s| layers(s) <= n)
        .unwrap_or(cap.min(1))
}

/// [`larch_model::ExpertPlan::resident`] for a load onto `before`, under the room
/// [`host_expert_layers`] plans with: `None` when the weights fit whole or when not even the
/// weights without their routed experts fit (whole layers then decide, and refuse); else the
/// (layer, expert) pairs `counts` routed most (`[layers * experts]`, layer-major) whose bytes
/// (`experts[i] / experts_per_layer` each) fit. Ties go to the later layer, then the lower
/// expert, so a uniform histogram keeps the leading layers' experts on the host as whole layers do.
/// `charge` is what a split costs past `reserve` ([`expert_cache_bytes`]): a model that fits whole
/// under `reserve` alone stays whole, as [`host_expert_layers`] keeps it.
#[allow(clippy::too_many_arguments)] // host_expert_layers' inputs and the histogram
pub(crate) fn expert_residency(
    before: &DeviceProps,
    weights: u64,
    experts: &[u64],
    counts: &[u64],
    per_layer: u32,
    c: &ModelConfig,
    max_context: u32,
    kv_pool_fraction: f32,
    reserve: u64,
    charge: u64,
    seats: u32,
) -> Option<Vec<Vec<u32>>> {
    let max_context = max_context.min(c.ctx_train);
    if before.backend == Backend::Cpu || max_context == 0 {
        return None;
    }
    let room =
        |mapped, r| weights_room(before, c, (max_context, kv_pool_fraction, seats), r, mapped);
    if weights <= room(true, reserve) {
        return None;
    }
    // The load aligns each layer's three slot tables to 256 bytes.
    let fixed = weights.saturating_sub(experts.iter().sum()) + 3 * 256 * experts.len() as u64;
    let mut left = room(false, reserve + charge).checked_sub(fixed)?;
    let e = per_layer as usize;
    let mut order: Vec<(usize, usize)> = (0..experts.len())
        .flat_map(|l| (0..e).map(move |x| (l, x)))
        .collect();
    order.sort_unstable_by_key(|&(l, x)| (std::cmp::Reverse((counts[l * e + x], l)), x));
    let mut resident = vec![Vec::new(); experts.len()];
    // ponytail: greedy by count; a knapsack over the layers' mixed quant sizes if it ever leaves
    // more than one expert's bytes unused.
    // ponytail: the plan does not charge a split layer's device round trip, so a skewed histogram can
    // cost a sync per layer (48 on a 48-layer fixture, against one per whole host layer); prefer whole
    // layers or cap the splits if the hit rate rises while tok/s falls.
    for (l, x) in order {
        let bytes = experts[l] / u64::from(per_layer);
        if bytes <= left {
            left -= bytes;
            resident[l].push(x as u32);
        }
    }
    Some(resident)
}

/// What a per-expert load allocates on the device past `reserve` and the `weights`' own bytes: the
/// `route_log` (charged even when a plan splits every layer and has none: the plan needs the figure
/// before it knows), the upload's alignment gaps at GGUF's usual 32 between `tensors` and its 16 KiB
/// page-rounded end, and for `dynamic` the copy stream's cuBLAS workspace, sized as
/// kernels/src/cuda/backend.cu's stream_create sizes it. Anything else shows as a shortfall, which
/// the load plans once more for.
pub(crate) fn expert_cache_bytes(
    props: &DeviceProps,
    route_log: u64,
    tensors: u64,
    dynamic: bool,
) -> u64 {
    let stream = match (dynamic, props.backend) {
        (true, Backend::Cuda | Backend::Hip) if props.arch_code >= 90 => 32 << 20,
        (true, Backend::Cuda | Backend::Hip) => 4 << 20,
        _ => 0,
    };
    route_log + tensors * 31 + (16 << 10) + stream
}

/// The state slots the documents plan asks for: the document's chat and one other.
pub(crate) const DOC_SLOTS: u32 = 2;

/// The fraction of memory the documents plan gives the KV pool (the knob's upper end); a reload
/// for the plan uses it too, so the reload and the fit cannot disagree.
pub const DOC_POOL_FRACTION: f32 = 0.95;

/// The largest `max_context` a load takes: the knob's maximum, or the model's trained context.
fn context_cap(c: &ModelConfig) -> u32 {
    let most = larch_records::knob_spec(larch_records::knob_id::MAX_CONTEXT)
        .expect("registered")
        .max as u32;
    most.min(c.ctx_train)
}

/// The documents plan's `max_context` for a document of `tokens`: with a turn's room rounded up
/// to a context step, at most [`context_cap`].
pub(crate) fn document_context(tokens: u32, c: &ModelConfig) -> u32 {
    let wanted = u64::from(tokens) + u64::from(crate::TURN_ROOM);
    let step = u64::from(crate::DOC_CONTEXT_STEP);
    wanted.next_multiple_of(step).min(u64::from(context_cap(c))) as u32
}

/// What the documents plan holds on `before`: the same host-expert plan a load makes
/// ([`host_expert_layers_with_pool`], [`host_expert_layers_that_fit`]) at [`DOC_SLOTS`] slots and
/// [`DOC_POOL_FRACTION`], for each context step; `reserve(context, seats)` is the loader's reserve
/// there. The longest document is the largest step that fits less a turn's room, and a longer
/// request than that does not fit, whatever the knob would let a load try. The plan's seats are
/// the most up to `max_seats` that still fit the document's context with no more host layers than
/// [`DOC_SLOTS`] needs, and below the many-chats plan's seats when the plan keeps experts on the
/// host and counts on the streaming `pool` for them. `most_host_layers`, when set, is the most host layers a step may take to
/// count as fitting.
#[allow(clippy::too_many_arguments)] // the fit's device, model and terms, and the question
pub(crate) fn document_fit(
    before: &DeviceProps,
    weights: u64,
    experts: &[u64],
    c: &ModelConfig,
    reserve: impl Fn(u32, u32) -> u64,
    pool: u64,
    max_seats: u32,
    most_host_layers: Option<u32>,
    tokens: u32,
) -> crate::DocFit {
    let f = DOC_POOL_FRACTION;
    let fits = |ctx, s| {
        host_expert_layers_that_fit(before, weights, experts, c, ctx, f, reserve(ctx, s), s)
            .is_some_and(|n| most_host_layers.is_none_or(|most| n <= most))
    };
    // The steps up to the largest context a load could take, which the last of them is.
    let top = context_cap(c);
    let steps =
        (1..=top.div_ceil(crate::DOC_CONTEXT_STEP)).map(|k| (k * crate::DOC_CONTEXT_STEP).min(top));
    let max_tokens = (steps.rev().find(|&ctx| fits(ctx, DOC_SLOTS)))
        .map_or(0, |ctx| ctx.saturating_sub(crate::TURN_ROOM));
    let ctx = document_context(tokens.min(max_tokens), c);
    let layers = |s| {
        host_expert_layers_with_pool(
            before,
            weights,
            experts,
            c,
            ctx,
            f,
            reserve(ctx, s),
            s,
            pool,
        )
    };
    let base = layers(DOC_SLOTS);
    // A load at MANY_SEATS seats or more takes the many-chats plan, whose expert cache has no
    // streaming pool: its host experts then prefill on the CPU (Windows 5070 Ti, 23 Nov: a
    // 140-page read at 90 % CPU and 10 % GPU, ~10x the bench's time). A plan that counts on the
    // pool stops short of it. With every expert on the GPU there is nothing to stream: it keeps
    // its seats.
    let most = match (pool, base) {
        (0, _) | (_, 0) => max_seats,
        _ => max_seats.min(crate::engine::MANY_SEATS - 1),
    };
    let seats = (DOC_SLOTS..=most)
        .rev()
        .find(|&s| fits(ctx, s) && layers(s) <= base)
        .unwrap_or(DOC_SLOTS);
    crate::DocFit {
        fits: tokens <= max_tokens,
        max_tokens,
        host_layers: layers(seats),
        seats,
    }
}

/// The plan for a model just loaded: `before` is the device as it was before the load, `free` what
/// it reports free now, `scratch_bytes` the initial serving buffers, `max_seqs` the startup
/// `max_batch_seqs`: the decode buckets the graph quota covers and, on the CPU reference, whose
/// memory is host RAM, the full contexts the pool holds at most. `graphs`: decode_graphs is on at
/// load. `arena_held`: the stream already allocated its capture arena (a reload after a capture),
/// so it is outside `free` and not charged.
#[allow(clippy::too_many_arguments)] // 9.10's inputs, plus the stream's graph state, the map and pool
pub(crate) fn fit(
    before: &DeviceProps,
    free: u64,
    c: &ModelConfig,
    max_context: u32,
    kv_pool_fraction: f32,
    scratch_bytes: u64,
    max_seqs: u32,
    graphs: bool,
    arena_held: bool,
    mapped: bool,
    stream_want: u64,
    checks: u32,
) -> Result<FitPlan, LoadError> {
    let (ctx, f, s) = (max_context, kv_pool_fraction, scratch_bytes);
    let (held, w) = (arena_held, stream_want);
    fit_seats(
        before, free, c, ctx, f, s, max_seqs, max_seqs, graphs, held, mapped, w, checks,
    )
}

/// [`fit`] with decode graphs for `graph_seats` seats beside the KV pool's `max_seqs`: the gate's
/// floor ([`crate::engine`]'s `gated_plan`) plans one seat's capture where all of them do not fit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_seats(
    before: &DeviceProps,
    free: u64,
    c: &ModelConfig,
    max_context: u32,
    kv_pool_fraction: f32,
    scratch_bytes: u64,
    max_seqs: u32,
    graph_seats: u32,
    graphs: bool,
    arena_held: bool,
    mapped: bool,
    stream_want: u64,
    checks: u32,
) -> Result<FitPlan, LoadError> {
    if max_context == 0 {
        return Err(LoadError::Backend(KError::InvalidArg.code()));
    }
    let headroom = headroom(before, c, mapped);
    let block = kv_block_bytes(c);
    let per_ctx = u64::from(max_context).div_ceil(BLOCK_TOKENS as u64);
    // Preserve one full context and headroom before admitting optional graphs.
    let pool_min = pool_min(before, c, max_context, kv_pool_fraction, max_seqs);
    // When decode_graphs is on at load; switching it on live reserves them from what is then free.
    // Buckets past the startup seats run eager if max_batch_seqs rises live.
    // ponytail: graph slots sized at load; graph_bucket_max knob if users raise max_batch_seqs live
    let (desired_graph_bytes, arena) = graph_need(before, graph_seats, graphs, arena_held, checks);
    let graph_room = free
        .saturating_sub(headroom)
        .saturating_sub(scratch_bytes)
        .saturating_sub(pool_min);
    let whole = mapped && matches!(c.arch, Arch::Qwen3Moe | Arch::Qwen35Moe);
    // The floor's one seat is charged at least the measured one-row capture, host layers or not.
    let floor = graphs && graph_seats < max_seqs;
    let graph_bytes = graph_quota(
        desired_graph_bytes,
        graph_room.saturating_sub(arena),
        whole || floor,
    );
    // No admitted slot means captures are prohibited, so the lazy arena need not be allocated.
    let graph_arena_bytes = if graph_bytes != 0 { arena } else { 0 };
    // The streaming pool is optional as the graphs are, after them: refused, those steps stay on
    // the CPU path, where a hard reserve would move about two more layers' experts to the host.
    let stream_room = graph_room
        .saturating_sub(graph_arena_bytes)
        .saturating_sub(graph_bytes);
    let stream_pool_bytes = if stream_want <= stream_room {
        stream_want
    } else {
        0
    };
    let model_bytes = before.vram_free.saturating_sub(free);
    let fixed = headroom + scratch_bytes + graph_bytes + graph_arena_bytes + stream_pool_bytes;
    let usable = (free.saturating_sub(fixed) as f64 * f64::from(kv_pool_fraction)) as u64;
    let mut blocks = (usable / block).min(u64::from(u32::MAX));
    if before.backend == Backend::Cpu {
        blocks = blocks.min(per_ctx * u64::from(max_seqs));
    }
    if blocks < per_ctx {
        return Err(LoadError::DoesNotFit {
            need: model_bytes + fixed + pool_min,
            have: before.vram_free,
        });
    }
    Ok(FitPlan {
        model_bytes,
        scratch_bytes,
        graph_bytes,
        graph_arena_bytes,
        stream_pool_bytes,
        headroom_bytes: headroom,
        kv_block_bytes: block,
        kv_blocks: blocks as u32,
        max_seqs_fit: (blocks / per_ctx) as u32,
        graph_floor: floor && graph_bytes != 0,
    })
}

#[cfg(test)]
mod tests {
    use larch_kernels::{Backend, DeviceProps};
    use larch_model::{DraftShape, FakeModel, Model, ModelConfig, MoeCfg};
    use larch_records::LoadError;
    use proptest::prelude::*;

    use crate::guess::{DRAFT_SEATS, draft_seats};

    use super::{
        GRAPH_BYTES, draft_fits, expert_residency, fit, graph_arena_reserve, graph_reserve,
        grouped_scratch, host_expert_layers, host_expert_layers_with_pool, mixed_reserve,
    };

    /// The 3050's 8B row: 32 layers of 8 KV heads of 128, so 128 KiB of KV a token.
    fn m8b() -> ModelConfig {
        let mut c = FakeModel::new(128_256).config().clone();
        (
            c.n_layers,
            c.kv_layers,
            c.n_kv_heads,
            c.head_dim,
            c.ctx_train,
        ) = (32, 32, 8, 128, 131_072);
        c
    }

    fn card(backend: Backend, free: u64, flags: u32) -> DeviceProps {
        DeviceProps {
            name: String::new(),
            backend,
            arch_code: 0,
            compute_units: 0,
            warp_size: 32,
            max_threads_per_group: 1024,
            shared_mem_per_group: 0,
            l2_bytes: 0,
            vram_total: 8 << 30,
            vram_free: free,
            recommended_working_set: free,
            peak_gbps: 0.0,
            driver_version: 0,
            runtime_version: 0,
            flags,
        }
    }

    /// Metal's grouped scratch, against the bytes backend.mm allocates: 16-byte aligned lists, a
    /// one-row block past the 64 MiB cap, the cap's blocks, and nothing off Metal or for a dense model.
    #[test]
    fn moe_partial_bytes_matches_the_metal_header() {
        let header = include_str!("../../../kernels/src/metal/args.h");
        assert!(header.contains("#define MOE_PARTIAL_BYTES (64ull << 20)"));
        assert_eq!(super::MOE_PARTIAL_BYTES, 64 << 20);
        assert!(
            header
                .lines()
                .any(|l| l.trim_end() == "#define MOE_MM_K 64")
        );
        assert_eq!(super::MOE_MM_K, 64);
    }

    #[test]
    fn grouped_scratch_matches_the_metal_allocation() {
        let moe = |experts, used, ffn| Some(MoeCfg { experts, used, ffn });
        let (mut m5, mut m4) = (card(Backend::Metal, 0, 0), card(Backend::Metal, 0, 0));
        (m5.arch_code, m4.arch_code) = (1010, 1009);
        let g = |e, k, d, rows| grouped_scratch(&m5, moe(e, k, 768), d, rows);
        // Gate and up's lists and x rows of 2048, then down's lists, partials and x rows of 768,
        // both live while down grows the scratch.
        assert_eq!(
            g(128, 8, 2048, 64),
            2 * 2576 + 2 * 64 * 8 * 2048 + 4 * 64 * 8 * 2048 + 2 * 64 * 8 * 768
        );
        assert_eq!(
            g(128, 8, 2048, 1024),
            2 * 33296 + (32 << 20) + (64 << 20) + 2 * 1024 * 8 * 768
        );
        // Gate and up's x rows past down's (a one-row block past the 64 MiB cap; 1024-row blocks
        // under it): down fits in their scratch.
        assert_eq!(g(512, 512, 65536, 4), 10256 + (256 << 20));
        assert_eq!(g(128, 8, 2048, 4096), 131600 + (128 << 20));
        // x rows pad to MOE_MM_K: a d of 2000 takes 2048, an ffn of 712 takes 768.
        assert_eq!(
            grouped_scratch(&m5, moe(128, 8, 712), 2000, 64),
            2 * 2576 + 2 * 64 * 8 * 2048 + 4 * 64 * 8 * 2000 + 2 * 64 * 8 * 768
        );
        // Without the tensor API, lists and partials only.
        assert_eq!(
            grouped_scratch(&m4, moe(128, 8, 768), 2048, 64),
            2 * 2576 + 4 * 64 * 8 * 2048
        );
        let cuda = card(Backend::Cuda, 0, 0);
        assert_eq!(grouped_scratch(&cuda, moe(128, 8, 768), 2048, 64), 0);
        assert_eq!(grouped_scratch(&m5, None, 2048, 64), 0);
    }

    /// A drafter's bytes reach the placement: weights that fit alone move an expert layer once the
    /// drafter counts, so the fit after the load does not refuse a load more host layers could serve.
    #[test]
    fn a_drafter_moves_expert_layers() {
        let path = std::env::temp_dir().join(format!("larch-fit-{}.gguf", std::process::id()));
        std::fs::write(&path, larch_model::tiny::dflash_gguf(7)).unwrap();
        let shape = DraftShape::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let c = m8b();
        let (ctx, fraction, weights) = (8192, 0.85, 8u64 << 30);
        let mut before = card(Backend::Cuda, 0, 0);
        let room = super::serving_need(&before, &c, ctx, fraction, 0, true, 1) + weights;
        before.vram_free = room;
        let experts = [1u64 << 30; 4];
        let layers =
            |reserve| host_expert_layers(&before, weights, &experts, &c, ctx, fraction, reserve, 1);
        let n = DRAFT_SEATS;
        assert_eq!(layers(super::guess_bytes(0, None, None, &c, 64, ctx, n)), 0);
        assert!(layers(super::guess_bytes(0, Some(&shape), None, &c, 64, ctx, n)) > 0);
        assert!(layers(super::guess_bytes(1, None, None, &c, 64, ctx, n)) > 0);
        // The MTP block's caches count with its weights.
        assert!(layers(super::guess_bytes(0, None, Some(1), &c, 64, ctx, n)) > 0);
    }

    /// A drafter's load a little short at its startup chunk takes the 1024-token chunk at its four
    /// draft seats and no host layer; fewer seats come after, and a load without a drafter plans
    /// its own rows and seats alone.
    #[test]
    fn a_drafter_takes_the_smaller_chunk_before_host_layers() {
        let path = std::env::temp_dir().join(format!("larch-chunk-{}.gguf", std::process::id()));
        std::fs::write(&path, larch_model::tiny::dflash_gguf(7)).unwrap();
        let shape = DraftShape::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let c = m8b();
        let (ctx, fraction, weights) = (8192, 0.85, 8u64 << 30);
        let (big, small) = (64 + 2048, 64 + 1024);
        let reserve = |rows, seats| {
            super::activation_bytes(&c, None, rows)
                + super::guess_bytes(0, Some(&shape), None, &c, rows, ctx, seats as usize)
        };
        let mut before = card(Backend::Cuda, 0, 0);
        let need = super::serving_need(&before, &c, ctx, fraction, 0, true, 1) + weights;
        // Short by half of what the smaller chunk saves.
        let short = (reserve(big, 4) - reserve(small, 4)) / 2;
        before.vram_free = need + reserve(big, 4) - short;
        let experts = [1u64 << 30; 4];
        let layers = |(rows, seats)| {
            let reserve = reserve(rows, seats);
            host_expert_layers(&before, weights, &experts, &c, ctx, fraction, reserve, 1)
        };
        let pick = |drafting| draft_fits(drafting, big, Some(small), 4).find(|&t| layers(t) == 0);
        assert!(layers((big, 4)) > 0);
        assert_eq!(pick(true), Some((small, 4)));
        assert_eq!(pick(false), None);
        let tries = |drafting, rows, seats| draft_fits(drafting, rows, Some(small), seats);
        assert_eq!(tries(false, big, 4).collect::<Vec<_>>(), [(big, 4)]);
        assert_eq!(
            tries(true, big, 3).collect::<Vec<_>>(),
            [(big, 3), (small, 3), (small, 2), (small, 1)]
        );
        // A chunk set by hand (no smaller rows) or already as small gives up seats only.
        assert_eq!(
            draft_fits(true, big, None, 2).collect::<Vec<_>>(),
            [(big, 2), (big, 1)]
        );
        assert_eq!(tries(true, small, 1).collect::<Vec<_>>(), [(small, 1)]);
    }

    /// A drafter's attempts go most state slots first, draft_fits' order on a tie, and the first
    /// refusal of the plan stops the ranking.
    #[test]
    fn a_drafter_tries_the_most_state_slots_first() {
        let (big, small) = (64 + 2048, 64 + 1024);
        let slots = |t: (u32, u32)| match t {
            (r, _) if r == big => 1,
            (_, 4) => 2,
            (_, 3 | 2) => 3,
            _ => 0,
        };
        let ranked = super::most_slots_first(draft_fits(true, big, Some(small), 4), |r, s| {
            Ok::<_, ()>(slots((r, s)))
        });
        assert_eq!(
            ranked.unwrap(),
            [(small, 3), (small, 2), (small, 4), (big, 4), (small, 1)]
        );
        let refused = super::most_slots_first(draft_fits(true, big, Some(small), 4), |r, _| {
            if r == small { Err("refused") } else { Ok(1) }
        });
        assert_eq!(refused, Err("refused"));
    }

    /// An automatic drafter stays at the plain load's host layers and state slots, whatever its
    /// KV blocks; one state slot fewer or one host layer more drops it.
    #[test]
    fn an_automatic_drafter_stays_only_where_it_costs_no_chat() {
        let chats = |host_layers, state_slots, kv_blocks| super::Chats {
            host_layers,
            state_slots,
            kv_blocks,
        };
        let plain = chats(0, 8, 1762);
        assert!(
            super::drafter_keeps(chats(0, 8, 581), plain),
            "same slots and layers"
        );
        assert!(
            !super::drafter_keeps(chats(0, 1, 1762), plain),
            "fewer slots"
        );
        assert!(
            !super::drafter_keeps(chats(1, 8, 1762), plain),
            "more host layers"
        );
    }

    /// One chat reserves one draft cache: three fewer than the cap's four, the bytes its fit frees.
    #[test]
    fn one_chat_reserves_one_draft_cache() {
        let path = std::env::temp_dir().join(format!("larch-seats-{}.gguf", std::process::id()));
        std::fs::write(&path, larch_model::tiny::dflash_gguf(7)).unwrap();
        let shape = DraftShape::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let (d_model, rows, ctx) = (m8b().d_model, 64, 8192);
        let cache = (shape.cache_bytes_per_position() + 4) * u64::from(shape.cache_positions(ctx));
        let bytes = |n| super::drafter_bytes(&shape, d_model, rows, ctx, draft_seats(n));
        assert!(cache > 0);
        assert_eq!(bytes(4) - bytes(1), 3 * cache);
        assert_eq!(draft_seats(0), 1);
        assert_eq!(draft_seats(16), DRAFT_SEATS);
    }

    /// serving_need is exactly what the fit after a load needs free: one byte less refuses.
    #[test]
    fn serving_need_is_the_fits_boundary() {
        let c = m8b();
        for flags in [0, DeviceProps::CAP_UNIFIED] {
            let before = card(Backend::Metal, 8 << 30, flags);
            let need = super::serving_need(&before, &c, 8192, 0.85, 1 << 20, false, 1);
            let at = |free| {
                fit(
                    &before,
                    free,
                    &c,
                    8192,
                    0.85,
                    1 << 20,
                    1,
                    false,
                    true,
                    false,
                    0,
                    0,
                )
            };
            assert!(at(need).is_ok(), "{flags}");
            assert!(
                matches!(at(need - 1), Err(LoadError::DoesNotFit { .. })),
                "{flags}"
            );
        }
    }

    /// A hybrid's KV block holds only its attention layers: a quarter of the layers, a quarter of
    /// the bytes, and four times the full contexts in the same memory.
    #[test]
    fn a_hybrid_block_holds_only_its_attention_layers() {
        let (dense, mut hybrid) = (m8b(), m8b());
        hybrid.kv_layers = dense.n_layers / 4;
        let card = card(Backend::Cuda, 8 << 30, 0);
        let plan = |c: &ModelConfig| {
            fit(
                &card,
                4 << 30,
                c,
                8192,
                0.85,
                0,
                1,
                false,
                false,
                false,
                0,
                0,
            )
        };
        let (d, h) = (plan(&dense).unwrap(), plan(&hybrid).unwrap());
        assert_eq!(h.kv_block_bytes * 4, d.kv_block_bytes);
        assert!(h.max_seqs_fit >= 4 * d.max_seqs_fit, "{h:?} against {d:?}");
    }

    /// Any MoE on a unified GPU (Qwen 3 or Qwen 3.5) leaves 15 % to the OS, 5 % on Metal when its
    /// weights are the mapped file; a dense model or the CPU reference a fifth.
    #[test]
    fn a_unified_moe_leaves_less_headroom() {
        let (mut moe, dense) = (m8b(), m8b());
        moe.arch = larch_model::Arch::Qwen3Moe;
        let metal = card(Backend::Metal, 8 << 30, DeviceProps::CAP_UNIFIED);
        assert_eq!(super::headroom(&metal, &moe, false), (8 << 30) / 20 * 3);
        assert_eq!(super::headroom(&metal, &dense, false), (8 << 30) / 5);
        assert_eq!(super::headroom(&metal, &moe, true), (8 << 30) / 20);
        assert_eq!(super::headroom(&metal, &dense, true), (8 << 30) / 5);
        let mut hybrid = moe.clone();
        hybrid.arch = larch_model::Arch::Qwen35Moe;
        assert_eq!(super::headroom(&metal, &hybrid, false), (8 << 30) / 20 * 3);
        hybrid.arch = larch_model::Arch::Qwen35;
        assert_eq!(super::headroom(&metal, &hybrid, false), (8 << 30) / 5);
        assert_eq!(
            super::headroom(&card(Backend::Cpu, 8 << 30, 0), &moe, false),
            (8 << 30) / 5
        );
        assert_eq!(
            super::headroom(&card(Backend::Cuda, 8 << 30, 0), &moe, true),
            512 << 20
        );
        hybrid.arch = larch_model::Arch::Qwen35Moe;
        assert_eq!(
            super::headroom(&card(Backend::Cuda, 8 << 30, 0), &hybrid, false),
            112 << 20
        );
        assert_eq!(
            super::headroom(&card(Backend::Hip, 8 << 30, 0), &hybrid, false),
            512 << 20
        );
        assert_eq!(
            super::headroom(&card(Backend::Cuda, 8 << 30, 0), &dense, false),
            512 << 20
        );
    }

    /// The hybrid MoE target's layout: 40 layers, 10 of them attention with 2 KV heads of 256.
    fn m35b() -> ModelConfig {
        let mut c = m8b();
        (
            c.arch,
            c.n_layers,
            c.kv_layers,
            c.n_kv_heads,
            c.head_dim,
            c.ctx_train,
        ) = (larch_model::Arch::Qwen35Moe, 40, 10, 2, 256, 262_144);
        c
    }

    /// The hybrid MoE target on the 5070 Ti before its load: the free memory, the weights, one
    /// layer's experts, and the loader's measured reserve at `slots` state slots (1,468,568,960 at
    /// 16 slots with 16 seats' graphs; a slot is 64,389,120).
    fn target(slots: u32) -> (DeviceProps, u64, Vec<u64>, u64) {
        let mut before = card(Backend::Cuda, 15_631_122_432, 0);
        before.vram_total = 17_066_033_152;
        let reserve = 1_468_568_960
            - u64::from(16 - slots) * 64_389_120
            - graph_reserve(Backend::Cuda, 16, 0)
            + graph_reserve(Backend::Cuda, slots, 0);
        (before, 16_209_005_056, vec![381_681_664; 40], reserve)
    }

    /// The target plan at max_context 2304, with the check's swing and the KV floor for its slots:
    /// the fewest host layers at one state slot (4, where 8 slots took 6), then the most slots
    /// those layers leave room for.
    #[test]
    fn the_target_plan_fills_its_slack_with_state_slots() {
        let c = m35b();
        let layers = |slots| {
            let (before, weights, experts, reserve) = target(slots);
            let reserve = reserve + crate::FREE_SWING;
            host_expert_layers(&before, weights, &experts, &c, 2304, 0.85, reserve, slots)
        };
        assert_eq!((layers(1), layers(8)), (4, 6));
        assert_eq!(super::most_state_slots(8, 4, layers), 3);
        assert_eq!(super::most_state_slots(8, 5, layers), 7);
        assert_eq!(super::most_state_slots(8, 6, layers), 8);
        assert_eq!(super::most_state_slots(1, 4, layers), 1);
    }

    /// The hybrid MoE target's KV floor at 2304 on the 5070 Ti: the measured per-chat peak for
    /// each seat the plan admits, one full context at least (the swing is the check's), where
    /// other arches and backends keep one full context and a longer context is not raised.
    #[test]
    fn the_hybrid_pool_floor_holds_the_measured_peak_a_seat() {
        let c = m35b();
        let (before, ..) = target(1);
        let blocks = |b: &DeviceProps, c: &ModelConfig, ctx, seats| {
            (super::pool_min(b, c, ctx, 0.85, seats) as f64 * f64::from(0.85f32)) as u64
                / super::kv_block_bytes(c)
        };
        assert_eq!(blocks(&before, &c, 2304, 1), 144);
        assert_eq!(blocks(&before, &c, 2304, 4), 300);
        assert_eq!(blocks(&before, &c, 2304, 8), 600);
        let mut dense = c.clone();
        dense.arch = larch_model::Arch::Qwen35;
        assert_eq!(blocks(&before, &dense, 2304, 8), 144);
        let metal = card(Backend::Metal, 8 << 30, DeviceProps::CAP_UNIFIED);
        assert_eq!(blocks(&metal, &c, 2304, 8), 144);
        assert_eq!(blocks(&before, &c, 16_384, 8), 1024);
    }

    #[test]
    fn expert_residency_keeps_the_hottest_experts_within_the_budget() {
        let mut c = m8b();
        (c.n_layers, c.kv_layers, c.n_kv_heads) = (48, 48, 4);
        let (e, each) = (256usize, 360_000_000u64);
        let experts = vec![each; 48];
        let (weights, reserve, free) = (18_600_000_000u64, 1 << 30, 15_500_000_000u64);
        let plan = |b: Backend, free: u64, counts: &[u64]| {
            let d = card(b, free, 0);
            expert_residency(
                &d, weights, &experts, counts, e as u32, &c, 8192, 0.85, reserve, 0, 1,
            )
        };
        let room = free
            - (512 << 20)
            - super::pool_min(&card(Backend::Cuda, 0, 0), &c, 8192, 0.85, 1)
            - reserve;
        let fixed = weights - 48 * each + 3 * 256 * 48;
        let held = |r: &[Vec<u32>]| r.iter().map(|l| l.len() as u64).sum::<u64>() * (each / 256);
        // Skewed: expert x of every layer routed 256 - x times. Each layer keeps a prefix of its
        // hottest experts, the budget holds and less than one expert's bytes stay unused.
        let skew: Vec<u64> = (0..48 * e).map(|i| (e - i % e) as u64).collect();
        let r = plan(Backend::Cuda, free, &skew).unwrap();
        assert!(fixed + held(&r) <= room);
        assert!(room - fixed - held(&r) < each / 256);
        for l in &r {
            let mut l = l.clone();
            l.sort_unstable();
            assert_eq!(l, (0..l.len() as u32).collect::<Vec<_>>());
        }
        // Uniform (no histogram yet): whole trailing layers, at most one split, never fewer
        // experts on the device than whole-layer placement keeps.
        let r = plan(Backend::Cuda, free, &vec![0; 48 * e]).unwrap();
        let split = r.iter().filter(|l| !l.is_empty() && l.len() < e).count();
        let first = r.iter().position(|l| l.len() == e).unwrap();
        assert!(
            split <= 1 && r[first..].iter().all(|l| l.len() == e),
            "{split}"
        );
        let whole = host_expert_layers(
            &card(Backend::Cuda, free, 0),
            weights,
            &experts,
            &c,
            8192,
            0.85,
            reserve,
            1,
        );
        assert!(held(&r) >= (48 - u64::from(whole)) * each);
        // A model that fits whole, one whose non-expert weights do not, and the CPU reference
        // place nothing per expert.
        assert_eq!(plan(Backend::Cuda, 40_000_000_000, &skew), None);
        assert_eq!(plan(Backend::Cuda, 1 << 30, &skew), None);
        assert_eq!(plan(Backend::Cpu, free, &skew), None);
    }

    /// What a per-expert load allocates past the plan's reserve is charged, so the check after the
    /// load passes its own plan, whatever slack (under one expert's bytes) the plan leaves.
    #[test]
    fn the_check_after_a_per_expert_load_passes_its_own_plan() {
        let mut c = m8b();
        (c.n_layers, c.kv_layers, c.n_kv_heads) = (48, 48, 4);
        let (e, each) = (256u64, 360_000_000u64);
        let experts = vec![each; 48];
        let (weights, reserve) = (18_600_000_000u64, 1u64 << 30);
        // CUDA's 64 + 1024 rows of 8 experts a layer, in i32.
        let (log, tensors) = (4 * 48 * 1088 * 8, 48 * 12 + 2);
        let skew: Vec<u64> = (0..48 * e).map(|i| e - i % e).collect();
        // The copy stream's cuBLAS workspace on sm_120.
        for (dynamic, stream) in [(false, 0), (true, 32u64 << 20)] {
            for step in 0..8 {
                let mut d = card(Backend::Cuda, 15_500_000_000 + step * (each / e) / 8, 0);
                d.arch_code = 120;
                let charge = super::expert_cache_bytes(&d, log, tensors, dynamic);
                let r = expert_residency(
                    &d, weights, &experts, &skew, e as u32, &c, 8192, 0.85, reserve, charge, 1,
                )
                .unwrap();
                let held = r.iter().map(|l| l.len() as u64).sum::<u64>() * (each / e);
                // The load: the weights without their experts with the tensors' alignment gaps and
                // the upload's page-rounded end, the slot tables, then the reserve, log and stream.
                let gaps = tensors * 31 + (16 << 10);
                let taken = weights - 48 * each + gaps + 3 * 256 * 48 + held;
                let free = d.vram_free - taken - reserve - log - stream;
                let need = super::serving_need(&d, &c, 8192, 0.85, 0, false, 1);
                assert!(free >= need, "{dynamic} {step}: {free} < {need}");
            }
        }
    }

    /// The cache's own costs count only once a split is planned: a model that fits whole by less
    /// than them stays whole, as whole-layer placement keeps it.
    #[test]
    fn a_model_that_fits_whole_by_a_little_is_not_split() {
        let mut c = m8b();
        (c.n_layers, c.kv_layers, c.n_kv_heads) = (48, 48, 4);
        let (weights, reserve, charge) = (18_600_000_000u64, 1u64 << 30, 32u64 << 20);
        let experts = vec![360_000_000u64; 48];
        let counts = vec![0; 48 * 256];
        // Exactly the free memory whole placement needs, mapped.
        let headroom = super::headroom(&card(Backend::Cuda, 0, 0), &c, true);
        let free = weights
            + headroom
            + super::pool_min(&card(Backend::Cuda, 0, 0), &c, 8192, 0.85, 1)
            + reserve;
        let d = card(Backend::Cuda, free, 0);
        assert_eq!(
            host_expert_layers(&d, weights, &experts, &c, 8192, 0.85, reserve, 1),
            0
        );
        let plan = expert_residency(
            &d, weights, &experts, &counts, 256, &c, 8192, 0.85, reserve, charge, 1,
        );
        assert_eq!(plan, None);
    }

    /// The streaming pool costs host layers only where the plan already moves some: resident
    /// stays resident, a plan short by a byte moves the layer that covers the pool too, and a pool
    /// no count of layers covers leaves the plan as it was.
    #[test]
    fn the_stream_pool_costs_layers_only_past_resident() {
        const LAYER: u64 = 360_000_000;
        const POOL: u64 = 200_000_000;
        let mut c = m8b();
        (c.n_layers, c.kv_layers, c.n_kv_heads) = (48, 48, 4);
        let experts = vec![LAYER; 48];
        let weights = 18_600_000_000;
        let reserve = 1 << 30;
        let plan = |free: u64, pool: u64| {
            let d = card(Backend::Cuda, free, 0);
            host_expert_layers_with_pool(&d, weights, &experts, &c, 8192, 0.85, reserve, 1, pool)
        };
        let resident = weights
            + super::DISCRETE_HEADROOM
            + super::pool_min(&card(Backend::Cuda, 0, 0), &c, 8192, 0.85, 1)
            + reserve;
        assert_eq!(plan(resident, 0), 0);
        assert_eq!(plan(resident, POOL), 0, "resident: no pool");
        // A layer short of resident moves one layer, and a pool under a layer takes a second;
        // with room for the pool beside the one layer, it stays at one.
        let short = resident - LAYER;
        assert_eq!(plan(short, 0), 1);
        assert_eq!(plan(short, POOL), 2, "the pool's layer");
        assert_eq!(plan(short + POOL, POOL), 1, "room for the pool");
        // A pool past every layer's experts reserves nothing.
        assert_eq!(plan(short, LAYER * 50), 1);
    }

    /// The documents plan on the large MoE fixture: a roomy card holds the model's whole context less
    /// the answer's room, a tight one a shorter document, and one with no room none; the context a
    /// document asks for rounds up to a step and stops at the model's.
    #[test]
    fn the_documents_plan_holds_the_context_less_the_answers_room() {
        use super::{document_context, document_fit};
        let mut c = m8b();
        (c.n_layers, c.kv_layers, c.n_kv_heads) = (48, 48, 4);
        let experts = vec![360_000_000u64; 48];
        let weights = 18_600_000_000;
        let plan = |free: u64, tokens: u32| {
            let d = card(Backend::Cuda, free, 0);
            document_fit(
                &d,
                weights,
                &experts,
                &c,
                |_, _| 1 << 30,
                0,
                8,
                None,
                tokens,
            )
        };
        // 131,072 tokens of context, 16,384 of them a turn's: the longest document.
        let roomy = plan(80 << 30, 100_000);
        assert_eq!(
            (roomy.fits, roomy.max_tokens),
            (true, 131_072 - crate::TURN_ROOM)
        );
        assert!(plan(80 << 30, roomy.max_tokens).fits);
        assert!(!plan(80 << 30, roomy.max_tokens + 1).fits);
        // 15.5 GB free holds the experts' rest and a shorter pool: fewer tokens, and every
        // document up to them fits while the next does not.
        let tight = plan(15_500_000_000, 1);
        assert!(tight.fits && tight.max_tokens > 0 && tight.max_tokens < roomy.max_tokens);
        assert!(plan(15_500_000_000, tight.max_tokens).fits);
        assert!(!plan(15_500_000_000, tight.max_tokens + 1).fits);
        assert!(tight.host_layers >= roomy.host_layers);
        let none = plan(1 << 30, 1);
        assert_eq!((none.fits, none.max_tokens), (false, 0));
        // Rounded up to a step with a turn's room; the model's context is the top.
        assert_eq!(document_context(1, &c), 32_768);
        assert_eq!(document_context(16_384, &c), 32_768);
        assert_eq!(document_context(16_385, &c), 49_152);
        assert_eq!(document_context(126_976, &c), 131_072);
        assert_eq!(document_context(u32::MAX - 16_384, &c), 131_072);
        c.ctx_train = 40_960;
        assert_eq!(document_context(126_976, &c), 40_960);
    }

    /// Where host experts prefill on the CPU, a document the plan could only hold with more host
    /// layers than the load outside it does not fit: the longest is what those layers hold, and a
    /// card that needs host layers for any document holds none at a cap of 0. A roomy card is not
    /// changed by the cap.
    #[test]
    fn a_cpu_prefill_plan_moves_no_more_experts_than_the_load_outside_it() {
        use super::document_fit;
        let mut c = m8b();
        (c.n_layers, c.kv_layers, c.n_kv_heads) = (48, 48, 4);
        let experts = vec![360_000_000u64; 48];
        let plan = |free: u64, most: Option<u32>, tokens: u32| {
            let d = card(Backend::Cuda, free, 0);
            document_fit(
                &d,
                18_600_000_000,
                &experts,
                &c,
                |_, _| 1 << 30,
                0,
                8,
                most,
                tokens,
            )
        };
        let roomy = plan(80 << 30, None, 100_000);
        assert_eq!(plan(80 << 30, Some(0), 100_000), roomy);
        let tight = plan(15_500_000_000, None, 1);
        assert!(tight.fits && tight.host_layers > 0);
        let none = plan(15_500_000_000, Some(0), 1);
        assert_eq!((none.fits, none.max_tokens), (false, 0));
        // The shortest document's layers hold a shorter longest document than no cap does.
        let held = plan(15_500_000_000, Some(tight.host_layers), 1);
        assert!(held.fits && held.host_layers <= tight.host_layers);
        assert!(held.max_tokens < tight.max_tokens);
        assert!(!plan(15_500_000_000, Some(tight.host_layers), tight.max_tokens).fits);
    }

    /// The documents plan seats as many as the device holds beside the document without moving more
    /// experts to the host than two seats do, up to the load's own cap; a tight card keeps two.
    #[test]
    fn the_documents_plan_seats_more_where_the_device_has_room() {
        let mut c = m8b();
        (c.n_layers, c.kv_layers, c.n_kv_heads) = (48, 48, 4);
        let experts = vec![360_000_000u64; 48];
        let plan = |free: u64, cap: u32| {
            let d = card(Backend::Cuda, free, 0);
            // Each seat's buffers cost a quarter of a gigabyte.
            let reserve = |_, seats| (1u64 << 30) + u64::from(seats) * (1 << 28);
            super::document_fit(
                &d,
                18_600_000_000,
                &experts,
                &c,
                reserve,
                0,
                cap,
                None,
                100_000,
            )
        };
        let roomy = plan(80 << 30, 8);
        assert_eq!((roomy.fits, roomy.seats), (true, 8));
        assert_eq!(plan(80 << 30, 1).seats, 2, "never under the plan's two");
        let tight = plan(15_500_000_000, 8);
        assert!(!tight.fits || tight.seats < 8);
        assert!(tight.seats >= 2);
    }

    /// A documents plan that keeps experts on the host and counts on the streaming pool seats
    /// fewer than the many-chats plan, whose expert cache has no pool and would prefill the host
    /// experts on the CPU. One with every expert on the GPU keeps its seats.
    #[test]
    fn a_documents_plan_with_the_pool_stays_under_the_many_chats_seats() {
        let mut c = m8b();
        (c.n_layers, c.kv_layers, c.n_kv_heads) = (48, 48, 4);
        let experts = vec![360_000_000u64; 48];
        let plan = |gib: u64, pool: u64| {
            let d = card(Backend::Cuda, gib << 30, 0);
            let reserve = |_, _| 1u64 << 30;
            super::document_fit(
                &d,
                18_600_000_000,
                &experts,
                &c,
                reserve,
                pool,
                8,
                None,
                100_000,
            )
        };
        let (offloaded, resident) = (plan(16, 1 << 30), plan(80, 1 << 30));
        assert!(offloaded.host_layers > 0 && resident.host_layers == 0);
        assert_eq!(plan(16, 0).seats, 8);
        assert_eq!(offloaded.seats, crate::engine::MANY_SEATS - 1);
        assert_eq!(resident.seats, 8);
    }

    /// A large MoE fixture: 48 layers of 4 KV heads of 128, about 18.6 GB, 17.3 GB of it routed
    /// experts.
    #[test]
    fn host_experts_are_the_fewest_leading_layers_that_fit() {
        let mut c = m8b();
        (c.n_layers, c.kv_layers, c.n_kv_heads) = (48, 48, 4);
        let experts = vec![360_000_000u64; 48];
        let weights = 18_600_000_000;
        let reserve = 1 << 30;
        let plan = |b: Backend, free: u64| {
            host_expert_layers(
                &card(b, free, 0),
                weights,
                &experts,
                &c,
                8192,
                0.85,
                reserve,
                1,
            )
        };
        // A 16 GiB card with 15.5 GB free: the rest of the model fits once n layers move, not n - 1.
        let free = 15_500_000_000;
        let n = plan(Backend::Cuda, free) as usize;
        let room = free
            - (512 << 20)
            - super::pool_min(&card(Backend::Cuda, 0, 0), &c, 8192, 0.85, 1)
            - reserve;
        let left = |n: usize| weights - experts[..n].iter().sum::<u64>();
        assert!(n > 0 && left(n) <= room && left(n - 1) > room, "{n}");
        // Room for everything moves nothing; no room at all moves every layer.
        assert_eq!(plan(Backend::Cuda, 40_000_000_000), 0);
        assert_eq!(plan(Backend::Cuda, 1 << 30), 48);
        // On Metal the whole model maps the file and needs only the 5 % headroom; a byte past
        // that leaves the mapped plan, and its 15 % moves layers.
        let mut moe = c.clone();
        moe.arch = larch_model::Arch::Qwen35Moe;
        let m5 = |free| {
            let mut d = card(Backend::Metal, free, DeviceProps::CAP_UNIFIED);
            d.vram_total = 20_000_000_000;
            host_expert_layers(&d, weights, &experts, &moe, 8192, 0.85, reserve, 1)
        };
        let fits = weights
            + 1_000_000_000
            + super::pool_min(
                &card(Backend::Metal, 0, DeviceProps::CAP_UNIFIED),
                &moe,
                8192,
                0.85,
                1,
            )
            + reserve;
        assert_eq!(m5(fits), 0);
        assert!(m5(fits - 1) > 0, "{}", m5(fits - 1));
        // The CPU reference holds it all in host memory already.
        assert_eq!(plan(Backend::Cpu, 1 << 30), 0);
        // A longer context's pool leaves less room, so more layers move; past the model's own
        // context, the plan clamps as the fit does.
        let at = |ctx: u32| {
            host_expert_layers(
                &card(Backend::Cuda, free, 0),
                weights,
                &experts,
                &c,
                ctx,
                0.85,
                reserve,
                1,
            )
        };
        assert!(
            at(c.ctx_train) > at(8192),
            "{} {}",
            at(c.ctx_train),
            at(8192)
        );
        assert_eq!(at(u32::MAX), at(c.ctx_train));
        // A dense model has no experts to move.
        assert_eq!(
            host_expert_layers(
                &card(Backend::Cuda, 1 << 30, 0),
                weights,
                &[],
                &c,
                8192,
                0.85,
                0,
                1
            ),
            0
        );
    }

    /// A guessing load reserves its check slots beside the buckets on CUDA; a plain load, or
    /// another backend, keeps its reserve.
    #[test]
    fn a_guessing_load_reserves_check_slots_on_cuda_only() {
        assert_eq!(
            graph_reserve(Backend::Cuda, 8, 5),
            graph_reserve(Backend::Cuda, 8, 0) + 5 * GRAPH_BYTES
        );
        for b in [Backend::Hip, Backend::Metal] {
            assert_eq!(graph_reserve(b, 8, 5), graph_reserve(b, 8, 0));
        }
    }

    /// A one-seat load's quota is its one bucket's slot, short of the one-row capture an expert
    /// model with no host layer measured (6,291,456 bytes, one piece): that plan's quota holds it
    /// whenever the room does; a room short of it, or a plan with host layers, keeps its slots.
    #[test]
    fn a_one_seat_quota_holds_the_measured_one_row_capture() {
        let before = card(Backend::Cuda, 7_600_000_000, DeviceProps::CAP_GRAPHS);
        let eager = (512 << 20) + (34 << 20) + 1_263_225_640;
        let mut moe = m8b();
        moe.arch = larch_model::Arch::Qwen3Moe;
        let quota = |room, mapped| {
            let free = eager + (64 << 20) + room;
            let p = fit(
                &before,
                free,
                &moe,
                8192,
                0.85,
                34 << 20,
                1,
                true,
                false,
                mapped,
                0,
                0,
            );
            p.unwrap().graph_bytes
        };
        assert!(quota(64 << 20, true) >= 6_291_456);
        assert!(quota(super::ONE_ROW_GRAPH_BYTES, true) >= 6_291_456);
        assert_eq!(quota(GRAPH_BYTES, true), GRAPH_BYTES);
        assert_eq!(quota(64 << 20, false), GRAPH_BYTES);
    }

    #[test]
    fn exact_cuda_graph_reserve_counts_only_reachable_buckets() {
        for (max_seqs, powers, cuda) in [
            (1, 1, 1),
            (8, 4, 4),
            (9, 5, 5),
            (15, 5, 11),
            (16, 5, 12),
            (17, 6, 13),
            (31, 6, 27),
            (32, 6, 28),
            // 33 to 63 rows pad to the 64 bucket.
            (33, 7, 29),
            (63, 7, 29),
            (64, 7, 29),
        ] {
            assert_eq!(
                graph_reserve(Backend::Cuda, max_seqs, 0),
                cuda * GRAPH_BYTES
            );
            for backend in [Backend::Cpu, Backend::Metal, Backend::Hip] {
                assert_eq!(graph_reserve(backend, max_seqs, 0), powers * GRAPH_BYTES);
            }
        }
    }

    #[test]
    fn graph_reserve_changes_only_cuda_with_graph_support() {
        let c = m8b();
        for backend in [Backend::Cuda, Backend::Hip, Backend::Metal, Backend::Cpu] {
            for capable in [false, true] {
                let d = card(
                    backend,
                    7_600_000_000,
                    if capable { DeviceProps::CAP_GRAPHS } else { 0 },
                );
                let p = fit(
                    &d,
                    d.vram_free,
                    &c,
                    8192,
                    0.85,
                    34 << 20,
                    16,
                    true,
                    false,
                    false,
                    0,
                    0,
                )
                .unwrap();
                let expected = if !capable {
                    0
                } else if backend == Backend::Cuda {
                    12
                } else {
                    5
                };
                assert_eq!(
                    p.graph_bytes,
                    expected * GRAPH_BYTES
                        + mixed_reserve(&d, larch_kernels::routes().mixed_graphs)
                );
            }
        }
    }

    #[test]
    fn mixed_reserve_is_four_estimated_slots_only_for_opted_in_cuda_graphs() {
        for backend in [Backend::Cpu, Backend::Hip, Backend::Metal, Backend::Cuda] {
            for capable in [false, true] {
                let d = card(
                    backend,
                    7_600_000_000,
                    if capable { DeviceProps::CAP_GRAPHS } else { 0 },
                );
                assert_eq!(super::mixed_reserve(&d, false), 0);
                assert_eq!(
                    super::mixed_reserve(&d, true),
                    if backend == Backend::Cuda && capable {
                        4 * GRAPH_BYTES
                    } else {
                        0
                    }
                );
            }
        }
    }

    #[test]
    fn arena_reserve_requires_cuda_or_hip_graph_capability() {
        for backend in [Backend::Cpu, Backend::Hip, Backend::Metal, Backend::Cuda] {
            for capable in [false, true] {
                let d = card(
                    backend,
                    7_600_000_000,
                    if capable { DeviceProps::CAP_GRAPHS } else { 0 },
                );
                assert_eq!(
                    graph_arena_reserve(&d),
                    if capable && matches!(backend, Backend::Cuda | Backend::Hip) {
                        64 << 20
                    } else {
                        0
                    }
                );
            }
        }
    }

    #[test]
    fn lazy_graph_arena_is_budgeted_once_beside_metadata() {
        let before = card(Backend::Cuda, 7_600_000_000, DeviceProps::CAP_GRAPHS);
        let p = fit(
            &before,
            before.vram_free,
            &m8b(),
            8192,
            0.85,
            34 << 20,
            16,
            true,
            false,
            false,
            0,
            0,
        )
        .unwrap();
        let metadata = graph_reserve(Backend::Cuda, 16, 0)
            + mixed_reserve(&before, larch_kernels::routes().mixed_graphs);
        let available = before.vram_free - (512 << 20) - (34 << 20) - metadata - (64 << 20);
        let expected = (available as f64 * f64::from(0.85_f32)) as u64 / p.kv_block_bytes;
        assert_eq!(u64::from(p.kv_blocks), expected);
    }

    /// A held arena is outside `free` already (a reload after a capture): the refit keeps the same
    /// graph quota and gives the pool its share of the 64 MiB back instead of charging it twice.
    #[test]
    fn a_held_arena_is_not_charged_again() {
        let before = card(Backend::Cuda, 7_600_000_000, DeviceProps::CAP_GRAPHS);
        let free = before.vram_free - 4_920_739_232 - 300_000_000;
        let new = fit(
            &before,
            free,
            &m8b(),
            8192,
            0.85,
            34 << 20,
            16,
            true,
            false,
            false,
            0,
            0,
        )
        .unwrap();
        let held = fit(
            &before,
            free,
            &m8b(),
            8192,
            0.85,
            34 << 20,
            16,
            true,
            true,
            false,
            0,
            0,
        )
        .unwrap();
        assert_eq!(
            (new.graph_arena_bytes, held.graph_arena_bytes),
            (64 << 20, 0)
        );
        assert_eq!(held.graph_bytes, new.graph_bytes);
        let blocks = |arena: u64| {
            let rest = free - (512 << 20) - (34 << 20) - new.graph_bytes - arena;
            ((rest as f64 * f64::from(0.85_f32)) as u64 / new.kv_block_bytes) as u32
        };
        assert_eq!(
            (new.kv_blocks, held.kv_blocks),
            (blocks(64 << 20), blocks(0))
        );
        assert!(held.kv_blocks > new.kv_blocks);
    }

    #[test]
    fn a_borderline_fit_limits_graphs_instead_of_rejecting_context() {
        let before = card(Backend::Cuda, 7_600_000_000, DeviceProps::CAP_GRAPHS);
        // The eager budget (one full context, headroom, scratch) plus the arena and 5.5 slots.
        let free = (512 << 20) + (34 << 20) + 1_263_225_640 + (64 << 20) + 11 * GRAPH_BYTES / 2;
        let p = fit(
            &before,
            free,
            &m8b(),
            8192,
            0.85,
            34 << 20,
            16,
            true,
            false,
            false,
            0,
            0,
        )
        .unwrap();
        assert_eq!(p.graph_bytes, 5 * GRAPH_BYTES);
        assert_eq!(p.graph_arena_bytes, 64 << 20);
        assert!(p.kv_blocks >= 512 && p.max_seqs_fit == 1, "{p:?}");
        assert!(
            p.model_bytes
                + p.scratch_bytes
                + p.graph_bytes
                + p.graph_arena_bytes
                + p.headroom_bytes
                + u64::from(p.kv_blocks) * p.kv_block_bytes
                <= before.vram_free
        );
    }

    #[test]
    fn a_full_context_can_load_eagerly_when_no_graph_slot_fits() {
        let before = card(Backend::Cuda, 7_600_000_000, DeviceProps::CAP_GRAPHS);
        // Exact minimum pool budget for the unchanged 8k/F16 fixture at fraction 0.85.
        let eager = (512 << 20) + (34 << 20) + 1_263_225_640;
        let p = fit(
            &before,
            eager + (64 << 20) + GRAPH_BYTES - 1,
            &m8b(),
            8192,
            0.85,
            34 << 20,
            16,
            true,
            false,
            false,
            0,
            0,
        )
        .unwrap();
        assert_eq!(p.graph_bytes, 0);
        assert_eq!(p.graph_arena_bytes, 0);
        assert!(p.kv_blocks >= 512);
        let p = fit(
            &before,
            eager + (64 << 20) + GRAPH_BYTES,
            &m8b(),
            8192,
            0.85,
            34 << 20,
            16,
            true,
            false,
            false,
            0,
            0,
        )
        .unwrap();
        assert_eq!(p.graph_bytes, GRAPH_BYTES);
        assert_eq!(p.graph_arena_bytes, 64 << 20);
        assert_eq!(p.kv_blocks, 512);
        match fit(
            &before,
            eager - 1,
            &m8b(),
            8192,
            0.85,
            34 << 20,
            16,
            true,
            false,
            false,
            0,
            0,
        ) {
            Err(LoadError::DoesNotFit { need, have }) => assert_eq!(need, have + 1),
            other => panic!("even the eager context must fit its real budget: {other:?}"),
        }
    }

    /// The streaming pool is counted beside the graphs and admitted only past one full context and
    /// them; refused, the plan is the plan without it.
    #[test]
    fn a_streaming_pool_is_counted_after_the_graphs_or_dropped() {
        let c = m8b();
        let before = card(Backend::Cuda, 15_000_000_000, DeviceProps::CAP_GRAPHS);
        let plan = |free, want| {
            fit(
                &before,
                free,
                &c,
                8192,
                0.85,
                34 << 20,
                16,
                true,
                false,
                false,
                want,
                0,
            )
            .unwrap()
        };
        let free = 9_000_000_000;
        let pool = 800_000_000;
        let (none, with) = (plan(free, 0), plan(free, pool));
        assert_eq!((none.stream_pool_bytes, with.stream_pool_bytes), (0, pool));
        assert_eq!(with.graph_bytes, none.graph_bytes);
        let usable = free - (512 << 20) - (34 << 20) - with.graph_bytes - with.graph_arena_bytes;
        let blocks = |b: u64| (b as f64 * f64::from(0.85_f32)) as u64 / with.kv_block_bytes;
        assert_eq!(u64::from(with.kv_blocks), blocks(usable - pool));
        assert_eq!(u64::from(none.kv_blocks), blocks(usable));
        // Tight: the room past the graphs and one context is one byte short of the pool.
        let eager = (512 << 20) + (34 << 20) + 1_263_225_640 + none.graph_bytes + (64 << 20);
        let tight = plan(eager + pool - 1, pool);
        assert_eq!(tight, plan(eager + pool - 1, 0));
        assert_eq!(tight.stream_pool_bytes, 0);
        // Exactly the room admits it.
        assert_eq!(plan(eager + pool, pool).stream_pool_bytes, pool);
    }

    /// An 8 GB card with the 8B model on it holds one full 8k context beside the graphs, and refuses
    /// 16k with what it would need.
    #[test]
    fn eight_gb_card_holds_one_context_or_refuses() {
        let before = card(Backend::Cuda, 7_600_000_000, DeviceProps::CAP_GRAPHS);
        let free = before.vram_free - 4_920_739_232 - 300_000_000; // weights and the model's scratch
        let p = fit(
            &before,
            free,
            &m8b(),
            8192,
            0.85,
            34 << 20,
            16,
            true,
            false,
            false,
            0,
            0,
        )
        .unwrap();
        assert_eq!(p.graph_bytes, 12 * GRAPH_BYTES);
        assert!(p.kv_blocks >= 512 && p.max_seqs_fit == 1, "{p:?}");
        let kv = u64::from(p.kv_blocks) * p.kv_block_bytes;
        assert!(
            p.model_bytes
                + p.scratch_bytes
                + p.graph_bytes
                + p.graph_arena_bytes
                + p.headroom_bytes
                + kv
                <= before.vram_free
        );
        match fit(
            &before,
            free,
            &m8b(),
            16_384,
            0.85,
            34 << 20,
            16,
            true,
            false,
            false,
            0,
            0,
        ) {
            Err(LoadError::DoesNotFit { need, have }) => {
                assert!(need > have && have == before.vram_free, "{need} {have}")
            }
            r => panic!("{r:?}"),
        }
    }

    /// Where the division for the refusal's pool rounds down onto an integer, that many bytes past
    /// the fixed costs still miss one block by the pool's own rounding: the refusal needs one more.
    #[test]
    fn a_refusal_at_the_rounding_edge_needs_more() {
        // Found by a search over every fraction from 0.30 to 0.95: ceil(P / f) here is 54_571_415_383,
        // where P is the full context KV bytes.
        let f = f32::from_bits(0x3ea1_2a52); // 0.314776
        let free = 54_571_415_383 + (512 << 20);
        let d = card(Backend::Cuda, free, 0);
        match fit(
            &d,
            free,
            &m8b(),
            8191 * 16,
            f,
            0,
            16,
            true,
            false,
            false,
            0,
            0,
        ) {
            Err(LoadError::DoesNotFit { need, have }) => assert!(need > have, "{need} {have}"),
            r => panic!("{r:?}"),
        }
    }

    proptest! {
        /// Whatever the device and the load: a plan fits its budget with one full context in the
        /// pool, and a refusal needs more than the device had.
        #[test]
        fn plans_fit_and_refusals_need_more(
            before in 0u64..(64 << 30),
            took in 0u64..(32 << 30),
            scratch in 0u64..(1 << 30),
            ctx in 1u32..=131_072,
            fraction in 0.30f32..=0.95,
            flags in 0u32..16,
            cpu in any::<bool>(),
            pool in 0u64..(2 << 30),
        ) {
            let backend = if cpu { Backend::Cpu } else { Backend::Cuda };
            let d = card(backend, before, flags);
            match fit(&d, before.saturating_sub(took), &m8b(), ctx, fraction, scratch, 16, true, false, false, pool, 0) {
                Ok(p) => {
                    if flags & DeviceProps::CAP_GRAPHS == 0 {
                        prop_assert_eq!(p.graph_bytes, 0);
                        prop_assert_eq!(p.graph_arena_bytes, 0);
                    }
                    let kv = u64::from(p.kv_blocks) * p.kv_block_bytes;
                    let sum = p.model_bytes + p.scratch_bytes + p.graph_bytes + p.graph_arena_bytes + p.stream_pool_bytes + p.headroom_bytes + kv;
                    prop_assert!(sum <= before, "{p:?}");
                    prop_assert!(u64::from(p.kv_blocks) >= u64::from(ctx).div_ceil(16), "{p:?}");
                }
                Err(LoadError::DoesNotFit { need, have }) => prop_assert!(need > have, "{need} {have}"),
                Err(e) => prop_assert!(false, "{e:?}"),
            }
        }
    }
}
