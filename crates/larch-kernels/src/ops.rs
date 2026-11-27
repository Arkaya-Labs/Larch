//! One function per larch.h op; semantics in `kernels/src/cpu/ops.cpp`. Who checks what:
//! - here: each tensor lies inside its live buffer ([`Tensor::new`]) on the stream's device (`on`),
//!   no output is a read-only wrapped buffer (`written`), and every host pointer is valid for what the
//!   op reads; inside a capture, `on` also hands each buffer to the graph ([`crate::Graph`]);
//! - the backend: each op's dtypes, shapes and host parameters, on the host before anything is
//!   enqueued (the op contract, architecture 6.1; the CPU reference does it, a GPU op must too);
//! - nobody: values in device memory (token ids, positions, slots, block tables, lengths), and an
//!   output that overlaps an input other than the in-place ones larch.h allows (rope, qk_norm, add
//!   with `out == a`, sigmoid_mul and gelu with `out == x`): the overlap makes the values garbage. One value
//!   out of range is undefined behaviour on the device, the one gap in this safe API, accepted under
//!   D2 (a fault is DeviceLost and the process exits).
//!
//! SAFETY (every call below): the first two checks hold, so the backend reads and writes only the
//! tensors' ranges, apart from the device values above.

use crate::{ConvState, GdnState, KError, KvPool, SampleParams, Stream, Tensor, check};
use larch_sys as sys;

fn on(s: &Stream, t: &Tensor) -> Result<sys::larch_tensor, KError> {
    s.uses(t.buf)?;
    Ok(t.raw())
}

/// [`on`] for a tensor the op writes.
fn written(s: &Stream, t: &Tensor) -> Result<sys::larch_tensor, KError> {
    if t.buf.inner.readonly {
        return Err(KError::InvalidArg);
    }
    on(s, t)
}

fn on_opt(s: &Stream, t: Option<&Tensor>) -> Result<Option<sys::larch_tensor>, KError> {
    t.map(|t| on(s, t)).transpose()
}

fn ptr(t: &Option<sys::larch_tensor>) -> *const sys::larch_tensor {
    t.as_ref().map_or(std::ptr::null(), |t| t as *const _)
}

/// `out[r] = table[ids[r]]`, dequantized.
pub fn embed(s: &Stream, table: &Tensor, ids: &Tensor, out: &Tensor) -> Result<(), KError> {
    let (t, i, o) = (on(s, table)?, on(s, ids)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_embed(s.raw, &t, &i, &o) })
}

/// `out = x / rms(x) * w` per row.
pub fn rmsnorm(s: &Stream, x: &Tensor, w: &Tensor, eps: f32, out: &Tensor) -> Result<(), KError> {
    let (x, w, o) = (on(s, x)?, on(s, w)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_rmsnorm(s.raw, &x, &w, eps, &o) })
}

/// `out = (x - mean) / sqrt(var + eps) * w + b` per row (a ViT's LayerNorm); `b` optional.
pub fn layernorm(
    s: &Stream,
    x: &Tensor,
    w: &Tensor,
    b: Option<&Tensor>,
    eps: f32,
    out: &Tensor,
) -> Result<(), KError> {
    let (x, w, b, o) = (on(s, x)?, on(s, w)?, on_opt(s, b)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_layernorm(s.raw, &x, &w, ptr(&b), eps, &o) })
}

/// The parameters of ggml_rope_ext.
#[derive(Clone, Copy)]
pub struct RopeParams<'a> {
    pub n_dims: u32,
    pub neox: bool,
    pub n_ctx_orig: u32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// f32 `[n_dims / 2]` dividing each frequency (Llama 3's rope_freqs).
    pub freq_factors: Option<Tensor<'a>>,
}

/// Rotates `qk` `[n, heads, head_dim]` in place, row `r` at position `pos[r]`.
pub fn rope(s: &Stream, qk: &Tensor, pos: &Tensor, p: &RopeParams) -> Result<(), KError> {
    rope_with(s, qk, pos, p, sys::larch_rope)
}

/// [`rope`] on f32 `qk`; position `-p` undoes position `p` (the backward pass).
pub fn rope_f32(s: &Stream, qk: &Tensor, pos: &Tensor, p: &RopeParams) -> Result<(), KError> {
    rope_with(s, qk, pos, p, sys::larch_rope_f32)
}

type RopeFn = unsafe extern "C" fn(
    *mut sys::larch_stream,
    *const sys::larch_tensor,
    *const sys::larch_tensor,
    *const sys::larch_rope_params,
) -> sys::larch_status;

fn rope_with(
    s: &Stream,
    qk: &Tensor,
    pos: &Tensor,
    p: &RopeParams,
    f: RopeFn,
) -> Result<(), KError> {
    let (q, ps, ff) = (
        written(s, qk)?,
        on(s, pos)?,
        on_opt(s, p.freq_factors.as_ref())?,
    );
    let raw = rope_raw(p, &ff);
    // SAFETY: see the module docs; `raw` and `ff` live across the call.
    check(unsafe { f(s.raw, &q, &ps, &raw) })
}

/// `p` as larch.h's params, its freq_factors `ff` (which must outlive their use).
fn rope_raw(p: &RopeParams, ff: &Option<sys::larch_tensor>) -> sys::larch_rope_params {
    sys::larch_rope_params {
        n_dims: p.n_dims,
        neox: p.neox.into(),
        n_ctx_orig: p.n_ctx_orig,
        freq_base: p.freq_base,
        freq_scale: p.freq_scale,
        ext_factor: p.ext_factor,
        attn_factor: p.attn_factor,
        beta_fast: p.beta_fast,
        beta_slow: p.beta_slow,
        freq_factors: ptr(ff),
    }
}

/// A Gated DeltaNet layer's recurrence in one call (`larch_fused_gdn`): [`gdn_conv`] of `qkv` into `conv`, [`gdn`]
/// of it, then the gated norm: [`qk_norm`] of each v head by `norm` and [`silu_mul`] by `z` into `out`. `[alpha,
/// beta, dt_bias, a]` as [`gdn`] takes them. `Unsupported` where the backend lacks it.
#[allow(clippy::too_many_arguments)]
pub fn fused_gdn(
    s: &Stream,
    qkv: &Tensor,
    conv_w: &Tensor,
    [alpha, beta, dt_bias, a]: [&Tensor; 4],
    eps: f32,
    (state, layer): (&GdnState, u32),
    slots: &Tensor,
    q_lens: &Tensor,
    snaps: Option<&Tensor>,
    [norm, z]: [&Tensor; 2],
    conv: &Tensor,
    out: &Tensor,
) -> Result<(), KError> {
    let (q, w, al, be) = (on(s, qkv)?, on(s, conv_w)?, on(s, alpha)?, on(s, beta)?);
    let (dt, a, st) = (on(s, dt_bias)?, on(s, a)?, state.raw(s, layer)?);
    let (sl, ql, sn) = (on(s, slots)?, on(s, q_lens)?, on_opt(s, snaps)?);
    let (n, z, c, o) = (on(s, norm)?, on(s, z)?, written(s, conv)?, written(s, out)?);
    // SAFETY: see the module docs; the state buffers hold every layer (GdnState::raw).
    check(unsafe {
        sys::larch_fused_gdn(
            s.raw,
            &q,
            &w,
            &al,
            &be,
            &dt,
            &a,
            eps,
            &st,
            &sl,
            &ql,
            ptr(&sn),
            &n,
            &z,
            &c,
            &o,
        )
    })
}

/// Attention's inputs after its projections in one call (`larch_fused_attn_in`): with `gate`, the [q, gate] halves
/// of `q_in` copied into `q` and `gate` (else `q` is `q_in`, done in place); [`qk_norm`] of q and k by `norms`, [`rope`]
/// of both, and [`kv_write`] of the roped k (k itself left as it was) and `v`. `Unsupported` where the backend
/// lacks it.
#[allow(clippy::too_many_arguments)]
pub fn fused_attn_in(
    s: &Stream,
    q_in: &Tensor,
    [k, v]: [&Tensor; 2],
    norms: Option<[&Tensor; 2]>,
    eps: f32,
    pos: &Tensor,
    p: &RopeParams,
    slots: &Tensor,
    (pool, layer): (&KvPool, u32),
    q: &Tensor,
    gate: Option<&Tensor>,
) -> Result<(), KError> {
    let (qi, k, v, ps, sl) = (
        on(s, q_in)?,
        on(s, k)?,
        on(s, v)?,
        on(s, pos)?,
        on(s, slots)?,
    );
    let n = norms
        .map(|[a, b]| Ok::<_, KError>([on(s, a)?, on(s, b)?]))
        .transpose()?;
    let np = |i: usize| n.as_ref().map_or(std::ptr::null(), |t| &t[i] as *const _);
    let ff = on_opt(s, p.freq_factors.as_ref())?;
    let raw = rope_raw(p, &ff);
    let (pl, q, g) = (
        pool.raw(s, layer)?,
        written(s, q)?,
        gate.map(|t| written(s, t)).transpose()?,
    );
    // SAFETY: see the module docs; `raw` and `ff` live across the call, and the pool buffers hold every layer
    // (KvPool::raw).
    check(unsafe {
        sys::larch_fused_attn_in(
            s.raw,
            &qi,
            &k,
            &v,
            np(0),
            np(1),
            eps,
            &ps,
            &raw,
            &sl,
            &pl,
            &q,
            ptr(&g),
        )
    })
}

/// How [`rope_multi`] gives each rotated pair its position (`larch_rope_mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeMode {
    /// Qwen2-VL: contiguous sections t, h, w, e.
    Mrope,
    /// Qwen3-VL and Qwen 3.5 text: the sections interleaved.
    Imrope,
    /// Qwen-VL encoders: contiguous, each section's frequency starting over; `n_dims` is half the head.
    Vision,
}

/// Rotates `qk` `[n, heads, head_dim]` in place with ggml_rope_multi: row `r`'s positions are
/// `pos[r]` (i32 `[n, 4]`: t, h, w, e), and `sections` pairs take each.
pub fn rope_multi(
    s: &Stream,
    qk: &Tensor,
    pos: &Tensor,
    p: &RopeParams,
    sections: [u32; 4],
    mode: RopeMode,
) -> Result<(), KError> {
    let (q, ps, ff) = (
        written(s, qk)?,
        on(s, pos)?,
        on_opt(s, p.freq_factors.as_ref())?,
    );
    let mode = match mode {
        RopeMode::Mrope => sys::larch_rope_mode::LARCH_ROPE_MROPE,
        RopeMode::Imrope => sys::larch_rope_mode::LARCH_ROPE_IMROPE,
        RopeMode::Vision => sys::larch_rope_mode::LARCH_ROPE_VISION,
    };
    // bindgen backs C enums with u32 on Linux/macOS but i32 on MSVC; the C field is uint32_t.
    #[allow(clippy::unnecessary_cast)]
    let mode = mode.0 as u32;
    let raw = sys::larch_rope_multi_params {
        rope: sys::larch_rope_params {
            n_dims: p.n_dims,
            neox: p.neox.into(),
            n_ctx_orig: p.n_ctx_orig,
            freq_base: p.freq_base,
            freq_scale: p.freq_scale,
            ext_factor: p.ext_factor,
            attn_factor: p.attn_factor,
            beta_fast: p.beta_fast,
            beta_slow: p.beta_slow,
            freq_factors: ptr(&ff),
        },
        sections,
        mode,
    };
    // SAFETY: see the module docs; `raw` and `ff` live across the call.
    check(unsafe { sys::larch_rope_multi(s.raw, &q, &ps, &raw) })
}

/// `out = x · wᵀ (+ bias)` for at most 16 rows of `x`: the decode GEMV.
pub fn mmvq(
    s: &Stream,
    w: &Tensor,
    x: &Tensor,
    bias: Option<&Tensor>,
    out: &Tensor,
) -> Result<(), KError> {
    let (w, x, b, o) = (on(s, w)?, on(s, x)?, on_opt(s, bias)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_mmvq(s.raw, &w, &x, ptr(&b), &o) })
}

/// [`mmvq`] for at most 32 rows, each row bit-identical to [`mmvq`] of that row alone: the GEMV that
/// checks guessed tokens.
pub fn mmvq_verify(
    s: &Stream,
    w: &Tensor,
    x: &Tensor,
    bias: Option<&Tensor>,
    out: &Tensor,
) -> Result<(), KError> {
    let (w, x, b, o) = (on(s, w)?, on(s, x)?, on_opt(s, bias)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_mmvq_verify(s.raw, &w, &x, ptr(&b), &o) })
}

/// MoE expert GEMV: row `r`'s slot `j` multiplies by expert `ids[r, j]` of `w` `[experts * N, k]`.
/// `x` is `[rows, k]` (every slot reads row `r`) or `[rows, slots, k]`; `out` is `[rows, slots, N]`,
/// or `[rows, N]` with `weights` `[rows, slots]`, the weighted sum over slots (the experts' combine).
pub fn mmvq_id(
    s: &Stream,
    w: &Tensor,
    x: &Tensor,
    ids: &Tensor,
    bias: Option<&Tensor>,
    weights: Option<&Tensor>,
    out: &Tensor,
) -> Result<(), KError> {
    let (w, x, i) = (on(s, w)?, on(s, x)?, on(s, ids)?);
    let (b, wt, o) = (on_opt(s, bias)?, on_opt(s, weights)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_mmvq_id(s.raw, &w, &x, &i, ptr(&b), ptr(&wt), &o) })
}

/// [`mmvq_id`] over a mixed layer (ABI 5): expert `e`'s rows are slot `table[e]` of `hot` when it is
/// at least 0, else slot `-1 - table[e]` of `cold` (`table` i32 `[experts]`); no bias. Each (row,
/// slot) reads only its expert's rows. A backend without it answers [`KError::Unsupported`].
#[allow(clippy::too_many_arguments)]
pub fn mmvq_id_mixed(
    s: &Stream,
    hot: &Tensor,
    cold: &Tensor,
    table: &Tensor,
    x: &Tensor,
    ids: &Tensor,
    weights: Option<&Tensor>,
    out: &Tensor,
) -> Result<(), KError> {
    let (h, c, t) = (on(s, hot)?, on(s, cold)?, on(s, table)?);
    let (x, i, wt, o) = (
        on(s, x)?,
        on(s, ids)?,
        on_opt(s, weights)?,
        written(s, out)?,
    );
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_mmvq_id_mixed(s.raw, &h, &c, &t, &x, &i, ptr(&wt), &o) })
}

/// [`mmvq_id`] of `gate` and of `up` on `x` by `ids`, then [`silu_mul`], into `out` (f16 `[rows, slots, N]`), in
/// one call: the routed experts' activation (`larch_fused_moe_gate_up`). `Unsupported` where the backend lacks it;
/// the caller then runs the three ops.
pub fn fused_moe_gate_up(
    s: &Stream,
    gate: &Tensor,
    up: &Tensor,
    x: &Tensor,
    ids: &Tensor,
    out: &Tensor,
) -> Result<(), KError> {
    let (g, u, x, i, o) = (
        on(s, gate)?,
        on(s, up)?,
        on(s, x)?,
        on(s, ids)?,
        written(s, out)?,
    );
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_fused_moe_gate_up(s.raw, &g, &u, &x, &i, &o) })
}

/// The shared expert's down projection of `shared` = (weight, activation, f16 `[rows, 1]` gate), when given, gated
/// and added to the f32 residual `x`, then the routed experts' weighted down projection of `act` added to it, in
/// one call (`larch_fused_moe_down`): [`mmvq`], [`sigmoid_mul`] and [`add`], then [`mmvq_id`] and [`add`].
/// `Unsupported` where the backend lacks it.
pub fn fused_moe_down(
    s: &Stream,
    down: &Tensor,
    act: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    shared: Option<[&Tensor; 3]>,
    x: &Tensor,
) -> Result<(), KError> {
    let (d, a, i, w) = (on(s, down)?, on(s, act)?, on(s, ids)?, on(s, weights)?);
    let sh = shared
        .map(|[w, a, g]| Ok::<_, KError>([on(s, w)?, on(s, a)?, on(s, g)?]))
        .transpose()?;
    let p = |k: usize| sh.as_ref().map_or(std::ptr::null(), |t| &t[k] as *const _);
    let x = written(s, x)?;
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_fused_moe_down(s.raw, &d, &a, &i, &w, p(0), p(1), p(2), &x) })
}

/// The MoE block's inputs in one call (`larch_fused_moe_in`): [`rmsnorm`] of `x` into `h`, the router's logits of
/// `h` ([`gemm`]) and [`moe_route`] into `ids` and `weights`; with `shared` = (gate, up, activation, gate input,
/// gate), the shared expert's [`mmvq`] gate and up, their [`silu_mul`] and its gate input's [`gemm`].
/// `Unsupported` where the backend lacks it.
#[allow(clippy::too_many_arguments)]
pub fn fused_moe_in(
    s: &Stream,
    x: &Tensor,
    norm: &Tensor,
    eps: f32,
    h: &Tensor,
    router: &Tensor,
    [logits, ids, weights]: [&Tensor; 3],
    shared: Option<[&Tensor; 5]>,
) -> Result<(), KError> {
    let (x, n, r) = (on(s, x)?, on(s, norm)?, on(s, router)?);
    let (h, l, i, w) = (
        written(s, h)?,
        written(s, logits)?,
        written(s, ids)?,
        written(s, weights)?,
    );
    let sh = shared
        .map(|[g, u, a, gi, go]| {
            Ok::<_, KError>([
                on(s, g)?,
                on(s, u)?,
                written(s, a)?,
                on(s, gi)?,
                written(s, go)?,
            ])
        })
        .transpose()?;
    let p = |k: usize| sh.as_ref().map_or(std::ptr::null(), |t| &t[k] as *const _);
    // SAFETY: see the module docs.
    check(unsafe {
        sys::larch_fused_moe_in(
            s.raw,
            &x,
            &n,
            eps,
            &h,
            &r,
            &l,
            &i,
            &w,
            p(0),
            p(1),
            p(2),
            p(3),
            p(4),
        )
    })
}

/// A Gated DeltaNet layer's inputs in one call (`larch_fused_gdn_in`): [`rmsnorm`] of `x` into `h`, then `h`'s
/// projections by `w` = (qkv, z, alpha, beta) into `out`, each [`mmvq`] for a quant weight and [`gemm`] for an f16
/// one. `Unsupported` where the backend lacks it.
pub fn fused_gdn_in(
    s: &Stream,
    x: &Tensor,
    norm: &Tensor,
    eps: f32,
    h: &Tensor,
    w: [&Tensor; 4],
    out: [&Tensor; 4],
) -> Result<(), KError> {
    let (x, n, h) = (on(s, x)?, on(s, norm)?, written(s, h)?);
    let [wq, wz, wa, wb] = w;
    let [oq, oz, oa, ob] = out;
    let (wq, wz, wa, wb) = (on(s, wq)?, on(s, wz)?, on(s, wa)?, on(s, wb)?);
    let (oq, oz, oa, ob) = (
        written(s, oq)?,
        written(s, oz)?,
        written(s, oa)?,
        written(s, ob)?,
    );
    // SAFETY: see the module docs.
    check(unsafe {
        sys::larch_fused_gdn_in(
            s.raw, &x, &n, eps, &h, &wq, &wz, &wa, &wb, &oq, &oz, &oa, &ob,
        )
    })
}

/// A projection onto the f32 residual `x` in one call (`larch_fused_proj_add`): [`sigmoid_mul`] of `input` by
/// `gate` when given, then [`mmvq`] by `w` and [`add`] onto `x`. `Unsupported` where the backend lacks it.
pub fn fused_proj_add(
    s: &Stream,
    w: &Tensor,
    input: &Tensor,
    gate: Option<&Tensor>,
    x: &Tensor,
) -> Result<(), KError> {
    let (w, i, g, x) = (on(s, w)?, on(s, input)?, on_opt(s, gate)?, written(s, x)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_fused_proj_add(s.raw, &w, &i, ptr(&g), &x) })
}

/// MoE routing (llama.cpp's softmax gating): per row of `logits` f32 `[rows, experts]`, the top `k`
/// ids, largest first, and their softmax weights over the top `k`'s sum, times `scale` (0 means 1).
pub fn moe_route(
    s: &Stream,
    logits: &Tensor,
    scale: f32,
    ids: &Tensor,
    weights: &Tensor,
) -> Result<(), KError> {
    let (l, i, w) = (on(s, logits)?, written(s, ids)?, written(s, weights)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_moe_route(s.raw, &l, scale, &i, &w) })
}

/// A quantized matrix as f16.
///
/// The direct tiled path avoids this whole-weight scratch on supported CUDA shapes.
pub fn dequant(s: &Stream, w: &Tensor, out: &Tensor) -> Result<(), KError> {
    let (w, o) = (on(s, w)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_dequant(s.raw, &w, &o) })
}

/// `out = x · wᵀ (+ bias)` with tile-local f16 weight conversion and f32 accumulation.
/// Returns [`KError::Unsupported`] when callers should use their existing linear fallback.
pub fn quant_gemm(
    s: &Stream,
    w: &Tensor,
    x: &Tensor,
    bias: Option<&Tensor>,
    out: &Tensor,
) -> Result<(), KError> {
    let (w, x, b, o) = (on(s, w)?, on(s, x)?, on_opt(s, bias)?, written(s, out)?);
    // SAFETY: see the module docs; all tensors remain live across the call.
    check(unsafe { sys::larch_quant_gemm(s.raw, &w, &x, ptr(&b), &o) })
}

/// Explicit activation-i8 Q4_K projection with f32 accumulation.
/// Activations use nearest-even signed quantization per group of 32; minimum correction retains
/// the original activation sum. This has a different numerical policy from [`quant_gemm`].
/// Returns [`KError::Unsupported`] without changing output when the direct path is unavailable.
pub fn quant_gemm_i8(
    s: &Stream,
    w: &Tensor,
    x: &Tensor,
    bias: Option<&Tensor>,
    out: &Tensor,
) -> Result<(), KError> {
    let (w, x, b, o) = (on(s, w)?, on(s, x)?, on_opt(s, bias)?, written(s, out)?);
    // SAFETY: see the module docs; all tensors remain live across the call.
    check(unsafe { sys::larch_quant_gemm_i8(s.raw, &w, &x, ptr(&b), &o) })
}

/// Bounded CUDA split-K variant of [`quant_gemm_i8`]; unsupported resources retain the caller's route.
pub fn quant_gemm_i8_split(
    s: &Stream,
    w: &Tensor,
    x: &Tensor,
    bias: Option<&Tensor>,
    out: &Tensor,
) -> Result<(), KError> {
    let (w, x, b, o) = (on(s, w)?, on(s, x)?, on_opt(s, bias)?, written(s, out)?);
    // SAFETY: see the module docs; all tensors remain live across the call.
    check(unsafe { sys::larch_quant_gemm_i8_split(s.raw, &w, &x, ptr(&b), &o) })
}

/// `c = alpha · a · b_tᵀ + beta · c (+ bias)`; `c` is not read when `beta == 0`.
pub fn gemm(
    s: &Stream,
    a: &Tensor,
    b_t: &Tensor,
    bias: Option<&Tensor>,
    c: &Tensor,
    alpha: f32,
    beta: f32,
) -> Result<(), KError> {
    let (a, bt, b, c) = (on(s, a)?, on(s, b_t)?, on_opt(s, bias)?, written(s, c)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_gemm(s.raw, &a, &bt, ptr(&b), &c, alpha, beta) })
}

/// `out = silu(gate) * up`.
pub fn silu_mul(s: &Stream, gate: &Tensor, up: &Tensor, out: &Tensor) -> Result<(), KError> {
    let (g, u, o) = (on(s, gate)?, on(s, up)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_silu_mul(s.raw, &g, &u, &o) })
}

/// `out = x * sigmoid(gate)`, `gate` `[n, d]` or `[n, 1]` (one value per row); `out` may be `x`.
pub fn sigmoid_mul(s: &Stream, x: &Tensor, gate: &Tensor, out: &Tensor) -> Result<(), KError> {
    let (x, g, o) = (on(s, x)?, on(s, gate)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_sigmoid_mul(s.raw, &x, &g, &o) })
}

/// Non-causal attention over dense `q`, `k`, `v` `[rows, heads, head_dim]`, per segment of `seg_len`
/// rows (an image tile's): `out = softmax(scale q k^T) v` per head.
pub fn attn_full(
    s: &Stream,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seg_len: u32,
    scale: f32,
    out: &Tensor,
) -> Result<(), KError> {
    let (q, k, v, o) = (on(s, q)?, on(s, k)?, on(s, v)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_attn_full(s.raw, &q, &k, &v, seg_len, scale, &o) })
}

/// `out = gelu(x)`, ggml_gelu's tanh form; `out` may be `x`.
pub fn gelu(s: &Stream, x: &Tensor, out: &Tensor) -> Result<(), KError> {
    let (x, o) = (on(s, x)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_gelu(s.raw, &x, &o) })
}

/// Gated DeltaNet step 1, the causal conv and SiLU of `x` `[n, channels]` with `w` f32
/// `[channels, kernel]` over `state`'s `layer`: `q_lens[s]` rows per sequence, sequence `s` in slot
/// `slots[s]`; a row whose `snaps` entry is >= 0 also leaves its state in that slot.
#[allow(clippy::too_many_arguments)]
pub fn gdn_conv(
    s: &Stream,
    x: &Tensor,
    w: &Tensor,
    state: &GdnState,
    layer: u32,
    slots: &Tensor,
    q_lens: &Tensor,
    snaps: Option<&Tensor>,
    out: &Tensor,
) -> Result<(), KError> {
    let (x, w, st) = (on(s, x)?, on(s, w)?, state.raw(s, layer)?);
    let (sl, ql, sn, o) = (
        on(s, slots)?,
        on(s, q_lens)?,
        on_opt(s, snaps)?,
        written(s, out)?,
    );
    // SAFETY: see the module docs; the state buffers hold every layer (GdnState::raw).
    check(unsafe { sys::larch_gdn_conv(s.raw, &x, &w, &st, &sl, &ql, ptr(&sn), &o) })
}

/// Gated DeltaNet step 2, the delta rule over `gdn_conv`'s `qkv`, with `alpha` and `beta` f32
/// `[n, v_heads]` and the GGUF's `dt_bias` and `a` f32 `[v_heads]`; sequences and snapshots as
/// [`gdn_conv`]. `out` is f16 `[n, v_heads * head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn gdn(
    s: &Stream,
    qkv: &Tensor,
    alpha: &Tensor,
    beta: &Tensor,
    dt_bias: &Tensor,
    a: &Tensor,
    eps: f32,
    state: &GdnState,
    layer: u32,
    slots: &Tensor,
    q_lens: &Tensor,
    snaps: Option<&Tensor>,
    out: &Tensor,
) -> Result<(), KError> {
    let (q, al, be) = (on(s, qkv)?, on(s, alpha)?, on(s, beta)?);
    let (dt, a, st) = (on(s, dt_bias)?, on(s, a)?, state.raw(s, layer)?);
    let (sl, ql, sn, o) = (
        on(s, slots)?,
        on(s, q_lens)?,
        on_opt(s, snaps)?,
        written(s, out)?,
    );
    // SAFETY: see the module docs; the state buffers hold every layer (GdnState::raw).
    check(unsafe {
        sys::larch_gdn(
            s.raw,
            &q,
            &al,
            &be,
            &dt,
            &a,
            eps,
            &st,
            &sl,
            &ql,
            ptr(&sn),
            &o,
        )
    })
}

/// [`silu_mul`] on f32 tensors.
pub fn silu_mul_f32(s: &Stream, gate: &Tensor, up: &Tensor, out: &Tensor) -> Result<(), KError> {
    let (g, u, o) = (on(s, gate)?, on(s, up)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_silu_mul_f32(s.raw, &g, &u, &o) })
}

/// The gradient of [`silu_mul_f32`]: `dgate` and `dup` from `dout`.
pub fn silu_mul_backward(
    s: &Stream,
    gate: &Tensor,
    up: &Tensor,
    dout: &Tensor,
    dgate: &Tensor,
    dup: &Tensor,
) -> Result<(), KError> {
    let (g, u, d) = (on(s, gate)?, on(s, up)?, on(s, dout)?);
    let (dg, du) = (written(s, dgate)?, written(s, dup)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_silu_mul_backward(s.raw, &g, &u, &d, &dg, &du) })
}

/// `c = alpha op(a) op(b) + beta c`, all f32, `op(x)` being `xᵀ` where its flag is set.
#[allow(clippy::too_many_arguments)]
pub fn gemm_ex(
    s: &Stream,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) -> Result<(), KError> {
    let (a, b, c) = (on(s, a)?, on(s, b)?, written(s, c)?);
    // SAFETY: see the module docs.
    check(unsafe {
        sys::larch_gemm_ex(
            s.raw,
            &a,
            &b,
            &c,
            trans_a.into(),
            trans_b.into(),
            alpha,
            beta,
        )
    })
}

/// [`rmsnorm`] with an f32 `out`.
pub fn rmsnorm_f32(
    s: &Stream,
    x: &Tensor,
    w: &Tensor,
    eps: f32,
    out: &Tensor,
) -> Result<(), KError> {
    let (x, w, o) = (on(s, x)?, on(s, w)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_rmsnorm_f32(s.raw, &x, &w, eps, &o) })
}

/// The gradient of [`rmsnorm_f32`] at `x` for `dy`, onto `dx` when `accumulate`, else into it.
pub fn rmsnorm_backward(
    s: &Stream,
    x: &Tensor,
    w: &Tensor,
    dy: &Tensor,
    eps: f32,
    dx: &Tensor,
    accumulate: bool,
) -> Result<(), KError> {
    let (x, w, g, o) = (on(s, x)?, on(s, w)?, on(s, dy)?, written(s, dx)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_rmsnorm_backward(s.raw, &x, &w, &g, eps, &o, accumulate.into()) })
}

/// `out = softmax(scale x)` per row; `causal` keeps columns `c <= r mod n` of row `r`, zeros the rest.
/// `out` may be `x`.
pub fn softmax(
    s: &Stream,
    x: &Tensor,
    scale: f32,
    causal: bool,
    out: &Tensor,
) -> Result<(), KError> {
    let (x, o) = (on(s, x)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_softmax(s.raw, &x, scale, causal.into(), &o) })
}

/// The gradient through [`softmax`]: `ds = scale p (dp - Σ p dp)` per row; `ds` may be `dp`.
pub fn softmax_backward(
    s: &Stream,
    p: &Tensor,
    dp: &Tensor,
    scale: f32,
    ds: &Tensor,
) -> Result<(), KError> {
    let (p, g, o) = (on(s, p)?, on(s, dp)?, written(s, ds)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_softmax_backward(s.raw, &p, &g, scale, &o) })
}

/// `out = a + b`; `out` may be `a`.
pub fn add(s: &Stream, a: &Tensor, b: &Tensor, out: &Tensor) -> Result<(), KError> {
    let (a, b, o) = (on(s, a)?, on(s, b)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_add(s.raw, &a, &b, &o) })
}

/// DFlash2's dynamic conv of `x` f16 `[n, d]` by `dyn` f32 `[n, 4, groups]` and `base` f32 `[4, d]`, stage
/// 0 or 1: `out` f16 is written, `out` f32 added into (larch.h).
pub fn draft_conv(
    s: &Stream,
    x: &Tensor,
    dyn_w: &Tensor,
    base: &Tensor,
    stage: u32,
    out: &Tensor,
) -> Result<(), KError> {
    let (x, d, b, o) = (on(s, x)?, on(s, dyn_w)?, on(s, base)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_draft_conv(s.raw, &x, &d, &b, stage, &o) })
}

/// Stores `k` and `v` `[n, n_kv_heads, head_dim]` at `slots[n]` of `layer`; slot -1 skips a row.
pub fn kv_write(
    s: &Stream,
    k: &Tensor,
    v: &Tensor,
    slots: &Tensor,
    pool: &KvPool,
    layer: u32,
) -> Result<(), KError> {
    let (k, v, sl, p) = (on(s, k)?, on(s, v)?, on(s, slots)?, pool.raw(s, layer)?);
    // SAFETY: see the module docs; the pool buffers hold every layer (KvPool::raw).
    check(unsafe { sys::larch_kv_write(s.raw, &k, &v, &sl, &p) })
}

/// One query row per sequence, attending to `seq_lens[r]` positions; 0 marks a padding row.
#[allow(clippy::too_many_arguments)]
pub fn attn_decode(
    s: &Stream,
    q: &Tensor,
    pool: &KvPool,
    layer: u32,
    block_table: &Tensor,
    seq_lens: &Tensor,
    splits: u32,
    scale: f32,
    out: &Tensor,
) -> Result<(), KError> {
    let (q, p, bt, sl, o) = (
        on(s, q)?,
        pool.raw(s, layer)?,
        on(s, block_table)?,
        on(s, seq_lens)?,
        written(s, out)?,
    );
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_attn_decode(s.raw, &q, &p, &bt, &sl, splits, scale, &o) })
}

/// Causal attention over prompt chunks: `q_lens[s]` rows per sequence from `q_start_pos[s]`.
#[allow(clippy::too_many_arguments)]
pub fn attn_prefill(
    s: &Stream,
    q: &Tensor,
    pool: &KvPool,
    layer: u32,
    block_table: &Tensor,
    q_start_pos: &Tensor,
    q_lens: &Tensor,
    scale: f32,
    out: &Tensor,
) -> Result<(), KError> {
    let (q, p, bt) = (on(s, q)?, pool.raw(s, layer)?, on(s, block_table)?);
    let (st, ql, o) = (on(s, q_start_pos)?, on(s, q_lens)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_attn_prefill(s.raw, &q, &p, &bt, &st, &ql, scale, &o) })
}

/// The general attention entry, causal: prefill's spans at decode's `splits`, each row attending to
/// every position before it.
#[allow(clippy::too_many_arguments)]
pub fn attn(
    s: &Stream,
    q: &Tensor,
    pool: &KvPool,
    layer: u32,
    block_table: &Tensor,
    q_start_pos: &Tensor,
    q_lens: &Tensor,
    splits: u32,
    scale: f32,
    out: &Tensor,
) -> Result<(), KError> {
    let (q, p, bt) = (on(s, q)?, pool.raw(s, layer)?, on(s, block_table)?);
    let (st, ql, o) = (on(s, q_start_pos)?, on(s, q_lens)?, written(s, out)?);
    // SAFETY: see the module docs; null mask and sinks and window 0 are the causal entry.
    check(unsafe {
        sys::larch_attn(
            s.raw,
            &q,
            &p,
            &bt,
            &st,
            &ql,
            std::ptr::null(),
            std::ptr::null(),
            0,
            splits,
            scale,
            &o,
        )
    })
}

/// Qwen3's per-head RMS norm of `qk` `[n, heads, head_dim]` in place.
pub fn qk_norm(s: &Stream, qk: &Tensor, w: &Tensor, eps: f32) -> Result<(), KError> {
    let (q, w) = (written(s, qk)?, on(s, w)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_qk_norm(s.raw, &q, &w, eps) })
}

/// Repetition penalty, then the top-64 logits and ids and the full-vocabulary log-sum-exp per row of
/// `logits` (one [`SampleParams`] per row).
pub fn sample_prep(
    s: &Stream,
    logits: &Tensor,
    per_row: &[SampleParams],
    penalty_ids: Option<&Tensor>,
    topk_vals: &Tensor,
    topk_ids: &Tensor,
    lse: &Tensor,
) -> Result<(), KError> {
    // The backend reads one entry per logits row.
    if per_row.len() as i64 != logits.shape()[0] {
        return Err(KError::InvalidArg);
    }
    let (l, pen) = (on(s, logits)?, on_opt(s, penalty_ids)?);
    let (v, i, e) = (
        written(s, topk_vals)?,
        written(s, topk_ids)?,
        written(s, lse)?,
    );
    // SAFETY: see the module docs; `per_row` holds one entry per logits row (above).
    check(unsafe { sys::larch_sample_prep(s.raw, &l, per_row.as_ptr(), ptr(&pen), &v, &i, &e) })
}

/// Inkling 2's causal short convolution with a residual over channels `offset..` of `state`'s `layer`:
/// `x` f16 `[n, C]`, `w` f32 `[C, kernel]`, optional per-head taps `dyn_taps` `[n, heads, kernel]` and
/// residual f32 `[n, C]`; sequences and snapshots as [`gdn_conv`]. `out` f16 or f32 `[n, C]` may be the
/// residual.
#[allow(clippy::too_many_arguments)]
pub fn short_conv(
    s: &Stream,
    x: &Tensor,
    w: &Tensor,
    dyn_taps: Option<&Tensor>,
    residual: Option<&Tensor>,
    state: &ConvState,
    layer: u32,
    offset: u32,
    slots: &Tensor,
    q_lens: &Tensor,
    snaps: Option<&Tensor>,
    out: &Tensor,
) -> Result<(), KError> {
    let (x, w, d, r) = (
        on(s, x)?,
        on(s, w)?,
        on_opt(s, dyn_taps)?,
        on_opt(s, residual)?,
    );
    let st = state.raw(s, layer)?;
    let (sl, ql, sn, o) = (
        on(s, slots)?,
        on(s, q_lens)?,
        on_opt(s, snaps)?,
        written(s, out)?,
    );
    // SAFETY: see the module docs; the state buffer holds every layer (ConvState::raw).
    check(unsafe {
        sys::larch_short_conv(
            s.raw,
            &x,
            &w,
            ptr(&d),
            ptr(&r),
            &st,
            offset,
            &sl,
            &ql,
            ptr(&sn),
            &o,
        )
    })
}

/// Inkling 2's clip of `x` `[n, heads, dim]` to an L2 norm of at most `t` per head, after the optional
/// per-channel `scale` f32 `[heads, dim]`; with `pos` (i32 `[n]`, and `floor`, `alpha`) times the
/// log-scaled position. `out` may be `x`.
#[allow(clippy::too_many_arguments)]
pub fn clip_norm(
    s: &Stream,
    x: &Tensor,
    scale: Option<&Tensor>,
    t: f32,
    pos: Option<&Tensor>,
    floor: u32,
    alpha: f32,
    out: &Tensor,
) -> Result<(), KError> {
    let (x, sc, p, o) = (
        on(s, x)?,
        on_opt(s, scale)?,
        on_opt(s, pos)?,
        written(s, out)?,
    );
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_clip_norm(s.raw, &x, ptr(&sc), t, ptr(&p), floor, alpha, &o) })
}

/// Inkling 2's attention over the depth: `states` f32 `[states, n, d]` normalised and mixed per row by
/// the softmax of their scores against each `query` f32 `[q, d]`, into `out` f32 `[n, q, d]`.
pub fn depth_mix(s: &Stream, states: &Tensor, query: &Tensor, out: &Tensor) -> Result<(), KError> {
    let (st, q, o) = (on(s, states)?, on(s, query)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_depth_mix(s.raw, &st, &q, &o) })
}

/// Inkling 2's loop controller: `out = (x - tanh(forget) x) + tanh(gain) msg`, `x`, `msg` and `out`
/// f32 `[n, d]`, `gain` and `forget` f16 or f32 `[n, d]`; `out` may be `x`.
pub fn loop_gate(
    s: &Stream,
    x: &Tensor,
    msg: &Tensor,
    gain: &Tensor,
    forget: &Tensor,
    out: &Tensor,
) -> Result<(), KError> {
    let (x, m, g, f, o) = (
        on(s, x)?,
        on(s, msg)?,
        on(s, gain)?,
        on(s, forget)?,
        written(s, out)?,
    );
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_loop_gate(s.raw, &x, &m, &g, &f, &o) })
}

/// Inkling 2's expert activation (SITU) on the fused gate-up output `gate_up` `[n, slots, 2 ff]`, each
/// slot's expert `ids` i32 `[n, slots]` scaling it by `scale[expert]` when given; `out` f16
/// `[n, slots, ff]`.
pub fn situ_mul(
    s: &Stream,
    gate_up: &Tensor,
    ids: &Tensor,
    scale: Option<&Tensor>,
    beta: f32,
    lin_beta: f32,
    out: &Tensor,
) -> Result<(), KError> {
    let (gu, i, sc, o) = (
        on(s, gate_up)?,
        on(s, ids)?,
        on_opt(s, scale)?,
        written(s, out)?,
    );
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_situ_mul(s.raw, &gu, &i, ptr(&sc), beta, lin_beta, &o) })
}

/// MoE routing with sigmoid gating: the `k` largest `sigmoid(logits) + bias` per row of `logits` f32
/// `[n, experts]`, their unbiased sigmoids divided by their sum plus `eps`, times `scale` (0 means 1)
/// and each expert's `down_scale`.
#[allow(clippy::too_many_arguments)]
pub fn moe_route_sigmoid(
    s: &Stream,
    logits: &Tensor,
    bias: Option<&Tensor>,
    eps: f32,
    scale: f32,
    down_scale: Option<&Tensor>,
    ids: &Tensor,
    weights: &Tensor,
) -> Result<(), KError> {
    let (l, b, d) = (on(s, logits)?, on_opt(s, bias)?, on_opt(s, down_scale)?);
    let (i, w) = (written(s, ids)?, written(s, weights)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_moe_route_sigmoid(s.raw, &l, ptr(&b), eps, scale, ptr(&d), &i, &w) })
}

/// Inkling 2's attention: [`attn`]'s spans (sliding `window` when > 0) with a relative-position bias
/// (`rel_q` `[n, heads, n_rel]` against `rel_w` f32 `[extent, n_rel]`) and each row's sink logits
/// `sink` `[n, heads]`. `q_lens` `None` is [`attn_decode`]'s form: `q_start_pos` holds each row's
/// seq_len (0: a padding row, whose output is zero).
#[allow(clippy::too_many_arguments)]
pub fn attn_rel(
    s: &Stream,
    q: &Tensor,
    pool: &KvPool,
    layer: u32,
    block_table: &Tensor,
    q_start_pos: &Tensor,
    q_lens: Option<&Tensor>,
    rel_q: &Tensor,
    rel_w: &Tensor,
    sink: &Tensor,
    window: u32,
    splits: u32,
    scale: f32,
    out: &Tensor,
) -> Result<(), KError> {
    let (q, p, bt) = (on(s, q)?, pool.raw(s, layer)?, on(s, block_table)?);
    let (st, ql) = (on(s, q_start_pos)?, on_opt(s, q_lens)?);
    let (rq, rw, sk, o) = (on(s, rel_q)?, on(s, rel_w)?, on(s, sink)?, written(s, out)?);
    // SAFETY: see the module docs; a null mask is the causal span.
    check(unsafe {
        sys::larch_attn_rel(
            s.raw,
            &q,
            &p,
            &bt,
            &st,
            ptr(&ql),
            std::ptr::null(),
            &rq,
            &rw,
            &sk,
            window,
            splits,
            scale,
            &o,
        )
    })
}

/// The logits' soft cap, `out = cap tanh(x / cap)`, `x` and `out` f32 `[n, vocab]`; `out` may be `x`.
pub fn softcap(s: &Stream, x: &Tensor, cap: f32, out: &Tensor) -> Result<(), KError> {
    let (x, o) = (on(s, x)?, written(s, out)?);
    // SAFETY: see the module docs.
    check(unsafe { sys::larch_softcap(s.raw, &x, cap, &o) })
}
