// CUDA ops other than the quant readers (quant.cu). Each checks what the CPU reference checks, in the
// same order, then launches; f32 math in the reference's order where it is cheap to keep.
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <numbers>
#include <type_traits>

#include "common/arch.h"
#include "common/warp.cuh"
#include "cuda/cuda.cuh"
#if !defined(__HIP_PLATFORM_AMD__)
#include <cuda_pipeline.h>
#endif

namespace cuda {

// V consecutive elements at p as f32: V = 4 is one float4 or two half2 (the host proved the
// 4 * sizeof(I)-byte alignment), V = 1 one element.
template <int V, class I>
__device__ __forceinline__ void vload(float (&v)[V], const I* p) {
  if constexpr (V == 1) {
    v[0] = float(*p);
  } else if constexpr (std::is_same_v<I, float>) {
    const float4 q = *reinterpret_cast<const float4*>(p);
    v[0] = q.x, v[1] = q.y, v[2] = q.z, v[3] = q.w;
  } else {
    const __half2* h = reinterpret_cast<const __half2*>(p);
    const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
    v[0] = a.x, v[1] = a.y, v[2] = b.x, v[3] = b.y;
  }
}
// v as f32 at p.
template <int V>
__device__ __forceinline__ void vstore(float* p, const float (&v)[V]) {
  for (int k = 0; k < V; ++k) p[k] = v[k];
}
// v as f16 at p, each rounded to nearest even like the CPU reference.
template <int V>
__device__ __forceinline__ void vstore(__half* p, const float (&v)[V]) {
  if constexpr (V == 1) {
    *p = __float2half_rn(v[0]);
  } else {
    __half2* h = reinterpret_cast<__half2*>(p);
    h[0] = __floats2half2_rn(v[0], v[1]), h[1] = __floats2half2_rn(v[2], v[3]);
  }
}

// out[g] = x[g] / sqrt(mean(x[g]^2) + eps) * w for each group g of n elements: block b normalises
// group b % per_row of row b / per_row, a whole row for rmsnorm or one head for qk_norm (in place).
// Thread t owns elements [V t, V t + V) of every chunk of V blockDim.x; the first chunk stays in
// registers between the sum of squares and the store (one pass up to d = 4096 at V = 4), later chunks
// are read again. V = 4 loads vectors (the host proved alignment), V = 1 is the scalar path.
// ponytail: x is re-read past d = 4096; two chunks in registers when a model with d > 4096 shows rms in
// the profile.
template <class I, class O, int V>
__global__ void __launch_bounds__(V == 4 ? 1024 : 256) rms_k(const I* x, int64_t x_ld, O* out, int64_t out_ld, int per_row,
                                              int64_t n, const float* w, int64_t w_ld, float eps) {
  __shared__ float part[32];
  const int64_t r = blockIdx.x / per_row, g = blockIdx.x % per_row;
  const I* xg = x + r * x_ld + g * n;
  O* og = out + r * out_ld + g * n;
  const int64_t i0 = int64_t(threadIdx.x) * V, step = int64_t(blockDim.x) * V;
  float v[V] = {}, ss = 0;
  if (i0 < n) vload(v, xg + i0);
  for (int k = 0; k < V; ++k) ss += v[k] * v[k];
  for (int64_t i = i0 + step; i < n; i += step) {
    float u[V];
    vload(u, xg + i);
    for (int k = 0; k < V; ++k) ss += u[k] * u[k];
  }
  // The reduction's syncs also order every read above before the writes below, so out may be x.
  const float scale = 1.0f / sqrtf(vv_block_reduce_sum(ss, part) / float(n) + eps);
  for (int64_t i = i0; i < n; i += step) {
    if (i != i0) vload(v, xg + i);
    for (int k = 0; k < V; ++k) v[k] = v[k] * scale * w[(i + k) * w_ld];
    vstore(og + i, v);
  }
}

// True when every group of n elements of v starts on a 4 * sizeof(T)-byte boundary (the base, the row
// stride and the group width all are), so rms_k<.., 4> may load it as vectors.
template <class T>
static bool vec4(const view& v, int64_t n) {
  constexpr uint64_t b = 4 * sizeof(T);
  return reinterpret_cast<uintptr_t>(v.p) % b == 0 && uint64_t(v.ld) * sizeof(T) % b == 0 &&
         uint64_t(n) * sizeof(T) % b == 0;
}

template <class I, class O>
rms_shape rms_shape_of(const stream& s, const view& x, const view& out, int64_t n) {
  // 1024 threads: one pass for d = 4096 (m1 decode, avg us per launch: 6.83 before; 2.07 here, 2.28 at 512,
  // 3.11 at 256); the scalar path measured 3.38 at 1024 against 3.11 at 256, so it keeps 256.
  const int vec = vec4<I>(x, n) && vec4<O>(out, n) ? 4 : 1;
  return {vec, int(std::min<int64_t>(vec == 4 ? 1024 : 256, ((n + vec - 1) / vec + s.warp - 1) / s.warp * s.warp))};
}
template rms_shape rms_shape_of<float, __half>(const stream&, const view&, const view&, int64_t);
template rms_shape rms_shape_of<__half, __half>(const stream&, const view&, const view&, int64_t);

template <class I, class O = __half>
static void rms(const stream& s, const view& x, const view& w, float eps, const view& out, int per_row) {
  const int64_t n = x.cols / per_row, blocks = x.rows * per_row;
  if (!blocks) return;
  if (blocks > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 rows"};
  const auto [vec, threads] = rms_shape_of<I, O>(s, x, out, n);
  launch(vec == 4 ? rms_k<I, O, 4> : rms_k<I, O, 1>, unsigned(blocks), unsigned(threads), s.s, x.as<I>(), x.ld,
         out.as<O>(), out.ld, per_row, n, w.as<float>(), w.ld, eps);
}

larch_status rmsnorm(larch_stream* vs, const larch_tensor* x, const larch_tensor* w, float eps,
                      const larch_tensor* out) {
  return run("rmsnorm", vs, [&](const stream& s) {
    const cpu::norm_checked a = cpu::check_rmsnorm(sizer(s), x, w, out);
    const view xv = a.x, wv = a.w, o = a.out;
    aligned({xv, wv, o});
    rms<float>(s, xv, wv, eps, o, 1);
  });
}

larch_status rmsnorm_f32(larch_stream* vs, const larch_tensor* x, const larch_tensor* w, float eps,
                          const larch_tensor* out) {
  return run("rmsnorm_f32", vs, [&](const stream& s) {
    const cpu::norm_checked a = cpu::check_rmsnorm(sizer(s), x, w, out, LARCH_F32);
    const view xv = a.x, wv = a.w, o = a.out;
    aligned({xv, wv, o});
    rms<float, float>(s, xv, wv, eps, o, 1);
  });
}

larch_status qk_norm(larch_stream* vs, const larch_tensor* qk, const larch_tensor* w, float eps) {
  return run("qk_norm", vs, [&](const stream& s) {
    const cpu::norm_checked a = cpu::check_qk_norm(sizer(s), qk, w);
    const view q = a.x, wv = a.w;
    aligned({q, wv});
    if (q.shape[1] > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 heads"};
    rms<__half>(s, q, wv, eps, q, int(q.shape[1]));
  });
}

// out = (x - mean) / sqrt(var + eps) * w + b per row, block r the row; as rms_k, the first chunk stays in
// registers across the two reductions and later chunks are read again.
template <int V>
__global__ void __launch_bounds__(V == 4 ? 1024 : 256) layernorm_k(const float* x, int64_t x_ld, __half* out,
                                                                   int64_t out_ld, int64_t n, const float* w,
                                                                   int64_t w_ld, const float* b, int64_t b_ld,
                                                                   float eps) {
  __shared__ float part[32];
  const float* xr = x + int64_t(blockIdx.x) * x_ld;
  __half* o = out + int64_t(blockIdx.x) * out_ld;
  const int64_t i0 = int64_t(threadIdx.x) * V, step = int64_t(blockDim.x) * V;
  float v[V] = {}, sum = 0;
  if (i0 < n) vload(v, xr + i0);
  for (int k = 0; k < V; ++k) sum += v[k];
  for (int64_t i = i0 + step; i < n; i += step) {
    float u[V];
    vload(u, xr + i);
    for (int k = 0; k < V; ++k) sum += u[k];
  }
  const float mean = vv_block_reduce_sum(sum, part) / float(n);
  float ss = 0;
  for (int64_t i = i0; i < n; i += step) {
    float u[V];
    if (i == i0) {
      for (int k = 0; k < V; ++k) u[k] = v[k];
    } else {
      vload(u, xr + i);
    }
    for (int k = 0; k < V; ++k) ss += (u[k] - mean) * (u[k] - mean);
  }
  const float scale = 1.0f / sqrtf(vv_block_reduce_sum(ss, part) / float(n) + eps);
  for (int64_t i = i0; i < n; i += step) {
    if (i != i0) vload(v, xr + i);
    for (int k = 0; k < V; ++k) v[k] = (v[k] - mean) * scale * w[(i + k) * w_ld] + (b ? b[(i + k) * b_ld] : 0.0f);
    vstore(o + i, v);
  }
}

larch_status layernorm(larch_stream* vs, const larch_tensor* x, const larch_tensor* w,
                        const larch_tensor* b_or_null, float eps, const larch_tensor* out) {
  return run("layernorm", vs, [&](const stream& s) {
    const cpu::layernorm_checked a = cpu::check_layernorm(sizer(s), x, w, b_or_null, out);
    const view xv = a.x, wv = a.w, bv = a.b, o = a.out;
    aligned({xv, wv, bv, o});
    const int64_t n = xv.cols;
    if (!xv.rows) return;
    if (xv.rows > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 rows"};
    const int vec = vec4<float>(xv, n) && vec4<__half>(o, n) ? 4 : 1;
    const int threads =
        int(std::min<int64_t>(vec == 4 ? 1024 : 256, ((n + vec - 1) / vec + s.warp - 1) / s.warp * s.warp));
    launch(vec == 4 ? layernorm_k<4> : layernorm_k<1>, unsigned(xv.rows), unsigned(threads), s.s, xv.as<float>(),
           xv.ld, o.as<__half>(), o.ld, n, wv.as<float>(), wv.ld, bv.as<float>(), bv.ld, eps);
  });
}

// ggml_gelu's tanh form, as ggml-cuda computes it; out may be x.
__global__ void gelu_k(const __half* x, int64_t x_ld, __half* out, int64_t o_ld, int64_t cols) {
  const int64_t r = blockIdx.x;
  for (int64_t c = blockIdx.y * int64_t(blockDim.x) + threadIdx.x; c < cols; c += int64_t(gridDim.y) * blockDim.x) {
    const float v = __half2float(x[r * x_ld + c]);
    out[r * o_ld + c] = __float2half_rn(0.5f * v * (1.0f + tanhf(cpu::GELU_K * v * (1.0f + cpu::GELU_A * v * v))));
  }
}

larch_status gelu(larch_stream* vs, const larch_tensor* x, const larch_tensor* out) {
  return run("gelu", vs, [&](const stream& s) {
    const cpu::elementwise_checked c = cpu::check_gelu(sizer(s), x, out);
    const view xv = c.a, o = c.out;
    aligned({xv, o});
    if (!xv.rows) return;
    launch(gelu_k, rows_grid(xv.rows, xv.cols, 256), 256, s.s, xv.as<__half>(), xv.ld, o.as<__half>(), o.ld,
           xv.cols);
  });
}

// attn_full: block (x, h, s) takes query rows [64 x, 64 x + 64) of segment s, head h, a thread a row;
// K and V of the segment pass through shared memory 64 rows at a time, each row's scores folded in
// with an online softmax (f32), as flash attention does. The route where the tensor-core kernel
// (attn_full_tc_k) cannot run: HIP, an arch below sm_80, rows not 16-byte aligned, or
// LARCH_ATTN_FULL_TC=0. ponytail: head_dim 64 (a ViT's); other widths when an encoder needs one.
constexpr int AF_ROWS = 64, AF_HD = 64;
__global__ void __launch_bounds__(AF_ROWS) attn_full_k(const __half* q, int64_t q_ld, const __half* k, int64_t k_ld,
                                                       const __half* v, int64_t v_ld, __half* out, int64_t o_ld,
                                                       int seg, float scale) {
  __shared__ __half2 ks[AF_ROWS][AF_HD / 2], vs[AF_ROWS][AF_HD / 2];
  const int64_t base = int64_t(blockIdx.z) * seg, col = int64_t(blockIdx.y) * AF_HD;
  const int r = blockIdx.x * AF_ROWS + threadIdx.x;
  const bool live = r < seg;
  float qv[AF_HD], acc[AF_HD] = {};
  if (live) {
    const __half2* qr = reinterpret_cast<const __half2*>(q + (base + r) * q_ld + col);
    for (int c = 0; c < AF_HD / 2; ++c) {
      const float2 f = __half22float2(qr[c]);
      qv[2 * c] = f.x * scale, qv[2 * c + 1] = f.y * scale;
    }
  }
  float m = -INFINITY, l = 0;
  for (int t0 = 0; t0 < seg; t0 += AF_ROWS) {
    __syncthreads();
    const int key = t0 + threadIdx.x;
    if (key < seg) {
      const __half2* kr = reinterpret_cast<const __half2*>(k + (base + key) * k_ld + col);
      const __half2* vr = reinterpret_cast<const __half2*>(v + (base + key) * v_ld + col);
      for (int c = 0; c < AF_HD / 2; ++c) ks[threadIdx.x][c] = kr[c], vs[threadIdx.x][c] = vr[c];
    }
    __syncthreads();
    const int nk = min(AF_ROWS, seg - t0);
    if (!live) continue;
    for (int j = 0; j < nk; ++j) {
      float s = 0;
      for (int c = 0; c < AF_HD / 2; ++c) {
        const float2 kf = __half22float2(ks[j][c]);
        s += qv[2 * c] * kf.x + qv[2 * c + 1] * kf.y;
      }
      if (s > m) {
        const float corr = expf(m - s);
        l *= corr;
        for (int c = 0; c < AF_HD; ++c) acc[c] *= corr;
        m = s;
      }
      // A -Inf score weighs 0, as the reference has it, even while m is still -Inf (exp(-Inf + Inf) is
      // NaN); a NaN score stays NaN.
      const float p = s == -INFINITY ? 0.f : expf(s - m);
      l += p;
      for (int c = 0; c < AF_HD / 2; ++c) {
        const float2 vf = __half22float2(vs[j][c]);
        acc[2 * c] += p * vf.x, acc[2 * c + 1] += p * vf.y;
      }
    }
  }
  if (live) {
    __half2* orow = reinterpret_cast<__half2*>(out + (base + r) * o_ld + col);
    for (int c = 0; c < AF_HD / 2; ++c) orow[c] = __floats2half2_rn(acc[2 * c] / l, acc[2 * c + 1] / l);
  }
}

// The tensor-core attn_full (after prefill.cuh, whose fragments it shares); false where it does not run.
static bool attn_full_tc(const stream& s, const view& q, const view& k, const view& v, const view& o, int seg,
                         float scale);

larch_status attn_full(larch_stream* vs, const larch_tensor* q, const larch_tensor* k, const larch_tensor* v,
                        uint32_t seg_len, float scale, const larch_tensor* out) {
  return run("attn_full", vs, [&](const stream& s) {
    const cpu::attn_full_checked a = cpu::check_attn_full(sizer(s), q, k, v, seg_len, out);
    const view qv = a.q, kv = a.k, vv = a.v, o = a.out;
    aligned({qv, kv, vv, o});
    if (qv.shape[2] != AF_HD) throw failure{LARCH_ERR_UNSUPPORTED, "attn_full: head_dim other than 64"};
    // half2 rows: every row and head starts on 4 bytes.
    for (const view* t : {&qv, &kv, &vv, &o})
      if (reinterpret_cast<uintptr_t>(t->p) % 4 != 0 || t->ld % 2 != 0)
        throw failure{LARCH_ERR_UNSUPPORTED, "attn_full: rows not 4-byte aligned"};
    if (!qv.rows) return;
    const int64_t segs = qv.rows / seg_len;
    if (segs > 65535 || qv.shape[1] > 65535 || seg_len > uint32_t(INT_MAX))
      throw failure{LARCH_ERR_UNSUPPORTED, "attn_full: grid too large"};
    if (attn_full_tc(s, qv, kv, vv, o, int(seg_len), scale)) return;
    const dim3 grid(unsigned((seg_len + AF_ROWS - 1) / AF_ROWS), unsigned(qv.shape[1]), unsigned(segs));
    launch(attn_full_k, grid, AF_ROWS, s.s, qv.as<__half>(), qv.ld, kv.as<__half>(), kv.ld, vv.as<__half>(), vv.ld,
           o.as<__half>(), o.ld, int(seg_len), scale);
  });
}

// ggml_rope_ext's per-call constants, computed on the host exactly as the CPU reference does.
struct rope_args {
  int64_t heads, hd, pairs;
  int neox;
  float theta_scale, freq_scale, ext_factor, mscale, lo, hi;
  const float* ff;  // freq_factors or null
  int64_t ff_ld;
};

// The angle table's size: pairs past it go in further chunks; one chunk covers n_dims <= 2048, every
// GGUF head_dim (576 for MLA) included.
constexpr int ROPE_CHUNK = 1024;

// Block (row r, blockDim.y heads): the angle of every pair once into shared memory, then thread (x, y)
// rotates pairs x, x + blockDim.x, .. of head y, in place. Products rounded one by one (no FMA), as the
// reference.
// ponytail: every block recomputes its row's angle table (the O(pairs) theta loop plus cos and sin);
// cache angles per position when prefill rope shows in a profile.
// Pair p's cos and sin (times mscale) at position pos: ggml's iterated theta, not powf, the reference's angle to the
// last bit before cos and sin, then rope_yarn.
__device__ __forceinline__ float2 rope_cs(const rope_args& a, int32_t pos, int64_t p) {
  float theta = float(pos);
  for (int64_t t = 0; t < p; ++t) theta *= a.theta_scale;
  const float extrap = theta / (a.ff ? a.ff[p * a.ff_ld] : 1.0f), interp = a.freq_scale * extrap;
  float t = interp;
  if (a.ext_factor != 0.0f) {
    const float y = (float(p) - a.lo) / fmaxf(0.001f, a.hi - a.lo);
    const float mix = (1 - fminf(1.0f, fmaxf(0.0f, y))) * a.ext_factor;
    t = __fadd_rn(__fmul_rn(interp, 1 - mix), __fmul_rn(extrap, mix));
  }
  return make_float2(cosf(t) * a.mscale, sinf(t) * a.mscale);
}

template <class T>
__global__ void rope_k(T* q, int64_t ld, const int32_t* pos, int64_t pos_ld, rope_args a) {
  __shared__ float2 cs[ROPE_CHUNK];
  const int64_t r = blockIdx.x;
  for (int64_t base = 0; base < a.pairs; base += ROPE_CHUNK) {
    const int end = int(min(a.pairs - base, int64_t(ROPE_CHUNK)));
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < end; i += blockDim.x * blockDim.y)
      cs[i] = rope_cs(a, pos[r * pos_ld], base + i);
    __syncthreads();
    for (int64_t h = blockIdx.y * int64_t(blockDim.y) + threadIdx.y; h < a.heads; h += int64_t(gridDim.y) * blockDim.y)
      for (int i = threadIdx.x; i < end; i += blockDim.x) {
        const int64_t p = base + i;
        T* x = q + r * ld + h * a.hd + (a.neox ? p : 2 * p);  // normal: adjacent pairs; neox: p, p + n_dims / 2
        T* y = x + (a.neox ? a.pairs : 1);
        const float x0 = float(*x), x1 = float(*y), c = cs[i].x, sn = cs[i].y;
        store(x, __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, sn)));
        store(y, __fadd_rn(__fmul_rn(x0, sn), __fmul_rn(x1, c)));
      }
    __syncthreads();  // the next chunk overwrites cs
  }
}

// rope's constants for q (checked by check_rope, freq_factors ff) under p.
static rope_args rope_arguments(const view& q, const view& ff, const larch_rope_params* p) {
  const int64_t nd = p->n_dims;
  const auto corr_dim = [&](float n_rot) {
    return float(nd) * logf(float(p->n_ctx_orig) / (n_rot * 2 * std::numbers::pi_v<float>)) / (2 * logf(p->freq_base));
  };
  rope_args a = {q.shape[1], q.shape[2], nd / 2, p->neox ? 1 : 0, powf(p->freq_base, -2.0f / float(nd)),
                 p->freq_scale, p->ext_factor, p->attn_factor,
                 std::max(0.0f, floorf(corr_dim(p->beta_fast))), std::min(float(nd - 1), ceilf(corr_dim(p->beta_slow))),
                 ff.as<float>(), ff.ld};
  if (p->ext_factor != 0.0f) a.mscale *= 1.0f + 0.1f * logf(1.0f / p->freq_scale);
  return a;
}

template <class T>
static larch_status rope_as(const char* op, larch_dtype t, larch_stream* vs, const larch_tensor* qk,
                             const larch_tensor* pos_i32, const larch_rope_params* p) {
  return run(op, vs, [&](const stream& s) {
    const cpu::rope_checked c = cpu::check_rope(sizer(s), qk, pos_i32, p, t);
    const view q = c.qk, pos = c.pos, ff = c.freq_factors;
    const rope_args a = rope_arguments(q, ff, p);
    aligned({q, pos, ff});
    if (!q.rows) return;
    // 256 threads as (pairs, heads). A decode row spreads its heads over several SMs; once the rows alone
    // fill the GPU (prefill), one block per row computes each angle table once instead of per head block.
    int sms = 0;
    ck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, s.ordinal), "cudaDeviceGetAttribute");
    const int px = int(std::min<int64_t>(256, (a.pairs + s.warp - 1) / s.warp * s.warp)), py = 256 / px;
    launch(rope_k<T>, rows_grid(q.rows, q.rows >= sms ? 1 : a.heads, py), dim3(unsigned(px), unsigned(py)), s.s,
           q.as<T>(), q.ld, pos.as<int32_t>(), pos.ld, a);
  });
}

// larch_fused_attn_in's device side: the shared checks' views as pointers and leading dimensions.
struct attn_in_args {
  const __half *q_in, *k, *v;
  __half *q, *gate, *kp, *vp;  // kp, vp: the pool layer, [slots, kv_heads * head_dim]
  int64_t q_in_ld, k_ld, v_ld, q_ld, gate_ld, pos_ld, slots_ld, qn_ld, kn_ld, heads, kv_heads;
  const float *qn, *kn;  // the norms' weights, or null
  const int32_t *pos, *slots;
  float eps;
  int q_vec, q_threads, k_vec, k_threads;  // rms_k's shapes of the q and k norms
  rope_args rope;
};

// Block (r, y): row r's angle table, its v to the pool (y 0), then a warp a head (q's, then k's, y's share): the copy out of q's pair (the
// gate half to gate), qk_norm's bytes (warp_rms), rope_k's rotation, then q's head to q and k's to the pool. Every
// value staged is an f16's, as each op stores f16 for the next.
__global__ void attn_in_k(attn_in_args a) {
  extern __shared__ float sm[];
  const int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32, warps = int(blockDim.x) / 32;
  const int64_t r = blockIdx.x, hd = a.rope.hd, pairs = a.rope.pairs, cols = a.kv_heads * hd;
  if (blockIdx.y * int64_t(warps) >= a.heads + a.kv_heads) return;
  float2* cs = reinterpret_cast<float2*>(sm);
  float* x = sm + 2 * pairs + warp * hd;
  for (int64_t i = threadIdx.x; i < pairs; i += blockDim.x) cs[i] = rope_cs(a.rope, a.pos[r * a.pos_ld], i);
  const int64_t slot = a.slots[r * a.slots_ld];
  if (slot >= 0 && !blockIdx.y)
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) a.vp[slot * cols + c] = a.v[r * a.v_ld + c];
  __syncthreads();
  for (int64_t u = blockIdx.y * int64_t(warps) + warp; u < a.heads + a.kv_heads; u += int64_t(gridDim.y) * warps) {
    const bool isq = u < a.heads;
    const int64_t h = isq ? u : u - a.heads;
    for (int64_t i = lane; i < hd; i += 32) {
      if (!isq) {
        x[i] = __half2float(a.k[r * a.k_ld + h * hd + i]);
      } else if (a.gate) {  // a 0/1 gemm's sum: + 0 makes a zero +0
        const __half* pr = a.q_in + r * a.q_in_ld + h * 2 * hd;
        x[i] = __half2float(pr[i]) + 0.0f;
        a.gate[r * a.gate_ld + h * hd + i] = __float2half_rn(__half2float(pr[hd + i]) + 0.0f);
      } else {
        x[i] = __half2float(a.q_in[r * a.q_in_ld + h * hd + i]);
      }
    }
    __syncwarp();
    if (a.qn) {
      if (isq) warp_rms(x, a.qn, a.qn_ld, a.eps, hd, a.q_vec, a.q_threads, lane);
      else warp_rms(x, a.kn, a.kn_ld, a.eps, hd, a.k_vec, a.k_threads, lane);
    }
    for (int64_t p = lane; p < pairs; p += 32) {
      const int64_t i0 = a.rope.neox ? p : 2 * p, i1 = i0 + (a.rope.neox ? pairs : 1);
      const float x0 = x[i0], x1 = x[i1], c = cs[p].x, sn = cs[p].y;
      x[i0] = __half2float(__float2half_rn(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, sn))));
      x[i1] = __half2float(__float2half_rn(__fadd_rn(__fmul_rn(x0, sn), __fmul_rn(x1, c))));
    }
    __syncwarp();
    for (int64_t i = lane; i < hd; i += 32) {
      if (isq) a.q[r * a.q_ld + h * hd + i] = __float2half_rn(x[i]);
      else if (slot >= 0) a.kp[slot * cols + h * hd + i] = __float2half_rn(x[i]);
    }
    __syncwarp();
  }
}

larch_status fused_attn_in(larch_stream* vs, const larch_tensor* q_in, const larch_tensor* k, const larch_tensor* v,
                            const larch_tensor* q_norm, const larch_tensor* k_norm, float eps,
                            const larch_tensor* pos_i32, const larch_rope_params* p, const larch_tensor* slots_i32,
                            const larch_kv_pool* pool, const larch_tensor* q_f16, const larch_tensor* gate) {
  return run("fused_attn_in", vs, [&](const stream& s) {
    const cpu::fused_attn_in_checked c = cpu::check_fused_attn_in(sizer(s), q_in, k, v, q_norm, k_norm, pos_i32, p,
                                                                  slots_i32, pool, q_f16, gate);
    const view qi = c.q_in, g = c.gate, q = c.q_rope.qk, kv = c.k_rope.qk, pos = c.q_rope.pos, ff = c.q_rope.freq_factors,
               vv = c.kv.v, sl = c.kv.slots, kp = c.kv.pool_k, vp = c.kv.pool_v, qn = c.q_norm.w, kn = c.k_norm.w;
    aligned({qi, g, q, kv, pos, ff, vv, sl, qn, kn});
    if (!q.rows) return;
    attn_in_args a{qi.as<__half>(), kv.as<__half>(), vv.as<__half>(), q.as<__half>(), g.as<__half>(), kp.as<__half>(),
                   vp.as<__half>(), qi.ld, kv.ld, vv.ld, q.ld, g.ld, pos.ld, sl.ld, qn.ld, kn.ld, q.shape[1],
                   kv.shape[1], qn.as<float>(), kn.as<float>(), pos.as<int32_t>(), sl.as<int32_t>(), eps, 0, 0, 0, 0,
                   rope_arguments(q, ff, p)};
    if (qn.p) {
      const rms_shape qs = rms_shape_of<__half, __half>(s, q, q, q.shape[2]),
                      ks = rms_shape_of<__half, __half>(s, kv, kv, kv.shape[2]);
      a.q_vec = qs.vec, a.q_threads = qs.threads, a.k_vec = ks.vec, a.k_threads = ks.threads;
    }
    const uint64_t smem = uint64_t(a.rope.pairs) * 8 + 8 * uint64_t(a.rope.hd) * 4;
    if (s.warp != 32 || smem > 48 * 1024 || q.rows > 0x7fffffff)
      throw failure{LARCH_ERR_UNSUPPORTED, "fused_attn_in: heads past a block's shared memory"};
    // A warp a head: a decode row's heads spread over blocks (and SMs), as rope_k spreads them.
    attn_in_k<<<dim3(unsigned(q.rows), unsigned((a.heads + a.kv_heads + 7) / 8)), 256, smem, s.s>>>(a);
    ck(cudaGetLastError(), "kernel launch");
  });
}

larch_status rope(larch_stream* vs, const larch_tensor* qk, const larch_tensor* pos_i32, const larch_rope_params* p) {
  return rope_as<__half>("rope", LARCH_F16, vs, qk, pos_i32, p);
}

larch_status rope_f32(larch_stream* vs, const larch_tensor* qk, const larch_tensor* pos_i32,
                       const larch_rope_params* p) {
  return rope_as<float>("rope_f32", LARCH_F32, vs, qk, pos_i32, p);
}

// ggml_rope_multi's per-call constants: rope_args's, the sections and the mode. Pair i rotates element i
// with element i + pairs (neox over n_dims, or the whole head under VISION).
struct rope_multi_args {
  rope_args r;
  int s0, s1, s2, s3, mode;
};

// As rope_k, a block per row: each pair's angle from the position its sector picks (ggml_mrope_cache_init),
// then thread (x, y) rotates pairs x, x + blockDim.x, .. of head y.
__global__ void rope_multi_k(__half* q, int64_t ld, const int32_t* pos, int64_t pos_ld, rope_multi_args m) {
  __shared__ float2 cs[ROPE_CHUNK];
  const rope_args& a = m.r;
  const int64_t r = blockIdx.x;
  const int sum = m.s0 + m.s1 + m.s2 + m.s3;
  for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < a.pairs; i += blockDim.x * blockDim.y) {
    const int j = i % sum;
    int k;
    if (m.mode == LARCH_ROPE_IMROPE)  // interleaved t, h, w within three times each section
      k = (j % 3 == 1 && j < 3 * m.s1) ? 1 : (j % 3 == 2 && j < 3 * m.s2) ? 2 : (j % 3 == 0 && j < 3 * m.s0) ? 0 : 3;
    else
      k = j < m.s0 ? 0 : j < m.s0 + m.s1 ? 1 : j < m.s0 + m.s1 + m.s2 ? 2 : 3;
    // VISION starts each section's frequency over at its first pair; the others run on from pair 0.
    const int first = k == 0 ? 0 : k == 1 ? m.s0 : k == 2 ? m.s0 + m.s1 : m.s0 + m.s1 + m.s2;
    const int steps = m.mode == LARCH_ROPE_VISION ? j - first : i;
    float theta = float(pos[r * pos_ld + k]);
    for (int t = 0; t < steps; ++t) theta *= a.theta_scale;
    const float extrap = theta / (a.ff ? a.ff[i * a.ff_ld] : 1.0f), interp = a.freq_scale * extrap;
    float t = interp;
    if (a.ext_factor != 0.0f) {
      const float y = (float(i) - a.lo) / fmaxf(0.001f, a.hi - a.lo);
      const float mix = (1 - fminf(1.0f, fmaxf(0.0f, y))) * a.ext_factor;
      t = __fadd_rn(__fmul_rn(interp, 1 - mix), __fmul_rn(extrap, mix));
    }
    cs[i] = make_float2(cosf(t) * a.mscale, sinf(t) * a.mscale);
  }
  __syncthreads();
  for (int64_t h = blockIdx.y * int64_t(blockDim.y) + threadIdx.y; h < a.heads; h += int64_t(gridDim.y) * blockDim.y)
    for (int i = threadIdx.x; i < a.pairs; i += blockDim.x) {
      __half* x = q + r * ld + h * a.hd + i;
      __half* y = x + a.pairs;
      const float x0 = float(*x), x1 = float(*y), c = cs[i].x, sn = cs[i].y;
      store(x, __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, sn)));
      store(y, __fadd_rn(__fmul_rn(x0, sn), __fmul_rn(x1, c)));
    }
}

larch_status rope_multi(larch_stream* vs, const larch_tensor* qk, const larch_tensor* pos_i32,
                         const larch_rope_multi_params* p) {
  return run("rope_multi", vs, [&](const stream& s) {
    const cpu::rope_checked c = cpu::check_rope_multi(sizer(s), qk, pos_i32, p);
    const view q = c.qk, pos = c.pos, ff = c.freq_factors;
    const larch_rope_params* rp = &p->rope;
    const int64_t nd = rp->n_dims, pairs = p->mode == LARCH_ROPE_VISION ? nd : nd / 2;
    if (pairs > ROPE_CHUNK) throw failure{LARCH_ERR_UNSUPPORTED, "rope_multi: more than 1024 pairs a head"};
    const auto corr_dim = [&](float n_rot) {
      return float(nd) * logf(float(rp->n_ctx_orig) / (n_rot * 2 * std::numbers::pi_v<float>)) / (2 * logf(rp->freq_base));
    };
    rope_multi_args m = {{q.shape[1], q.shape[2], pairs, 1, powf(rp->freq_base, -2.0f / float(nd)), rp->freq_scale,
                          rp->ext_factor, rp->attn_factor, std::max(0.0f, floorf(corr_dim(rp->beta_fast))),
                          std::min(float(nd - 1), ceilf(corr_dim(rp->beta_slow))), ff.as<float>(), ff.ld},
                         int(p->sections[0]), int(p->sections[1]), int(p->sections[2]), int(p->sections[3]),
                         int(p->mode)};
    if (rp->ext_factor != 0.0f) m.r.mscale *= 1.0f + 0.1f * logf(1.0f / rp->freq_scale);
    aligned({q, pos, ff});
    if (!q.rows) return;
    const int px = int(std::min<int64_t>(256, (pairs + s.warp - 1) / s.warp * s.warp)), py = 256 / px;
    launch(rope_multi_k, rows_grid(q.rows, 1, py), dim3(unsigned(px), unsigned(py)), s.s, q.as<__half>(), q.ld,
           pos.as<int32_t>(), pos.ld, m);
  });
}

// out = silu(gate) * up, silu(x) = x / (1 + e^-x).
template <class T>
__global__ void silu_mul_k(const T* g, int64_t g_ld, const T* u, int64_t u_ld, T* out, int64_t o_ld, int64_t cols) {
  const int64_t r = blockIdx.x;
  for (int64_t c = blockIdx.y * int64_t(blockDim.x) + threadIdx.x; c < cols; c += int64_t(gridDim.y) * blockDim.x) {
    const float x = float(g[r * g_ld + c]);
    store(out + r * o_ld + c, x / (1.0f + expf(-x)) * float(u[r * u_ld + c]));
  }
}

template <class T>
static larch_status silu_mul_as(const char* op, larch_dtype t, larch_stream* vs, const larch_tensor* gate,
                                 const larch_tensor* up, const larch_tensor* out) {
  return run(op, vs, [&](const stream& s) {
    const cpu::elementwise_checked c = cpu::check_silu_mul(sizer(s), gate, up, out, t);
    const view g = c.a, u = c.b, o = c.out;
    aligned({g, u, o});
    if (!g.rows) return;
    launch(silu_mul_k<T>, rows_grid(g.rows, g.cols, 256), 256, s.s, g.as<T>(), g.ld, u.as<T>(), u.ld, o.as<T>(), o.ld,
           g.cols);
  });
}

larch_status silu_mul(larch_stream* vs, const larch_tensor* gate, const larch_tensor* up, const larch_tensor* out) {
  return silu_mul_as<__half>("silu_mul", LARCH_F16, vs, gate, up, out);
}

larch_status silu_mul_f32(larch_stream* vs, const larch_tensor* gate, const larch_tensor* up,
                           const larch_tensor* out) {
  return silu_mul_as<float>("silu_mul_f32", LARCH_F32, vs, gate, up, out);
}

// The training ops (larch.h), f32, f32 sums in any order where the reference uses double.

// c = alpha op(a) op(b) + beta c, row-major, as cuBLAS's column-major c^T = op(b)^T op(a)^T. The stream's
// math mode keeps f32 (no TF32).
larch_status gemm_ex(larch_stream* vs, const larch_tensor* a, const larch_tensor* b, const larch_tensor* c,
                      uint32_t trans_a, uint32_t trans_b, float alpha, float beta) {
  return run("gemm_ex", vs, [&](const stream& s) {
    const cpu::train_checked g = cpu::check_gemm_ex(sizer(s), a, b, c, trans_a, trans_b);
    const view av = g.a, bv = g.b, cv = g.c;
    aligned({av, bv, cv});
    const int64_t k = trans_a ? av.rows : av.cols;
    for (const int64_t d : {cv.rows, cv.cols, k, av.ld, bv.ld, cv.ld})
      if (d > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "a dim or row stride past cuBLAS's 32-bit int"};
    if (!cv.rows || !cv.cols) return;
    ck(cublasSgemm(s.blas, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, int(cv.cols),
                   int(cv.rows), int(k), &alpha, bv.as<float>(), int(bv.ld), av.as<float>(), int(av.ld), &beta,
                   cv.as<float>(), int(cv.ld)),
       "cublasSgemm");
  });
}

// Block r: row r of the rmsnorm gradient (larch.h), onto dx when acc.
__global__ void rms_back_k(const float* x, int64_t x_ld, const float* w, int64_t w_ld, const float* dy, int64_t dy_ld,
                           float* dx, int64_t dx_ld, int64_t d, float eps, int acc) {
  __shared__ float part[32];
  const int64_t r = blockIdx.x;
  const float *xr = x + r * x_ld, *gr = dy + r * dy_ld;
  float* o = dx + r * dx_ld;
  float ss = 0, c = 0;
  for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
    const float v = xr[i];
    ss += v * v;
    c += gr[i] * w[i * w_ld] * v;
  }
  const float sc = 1.0f / sqrtf(vv_block_reduce_sum(ss, part) / float(d) + eps);
  c = vv_block_reduce_sum(c, part) * sc * sc * sc / float(d);
  for (int64_t i = threadIdx.x; i < d; i += blockDim.x) {
    const float v = sc * w[i * w_ld] * gr[i] - xr[i] * c;
    o[i] = acc ? o[i] + v : v;
  }
}

larch_status rmsnorm_backward(larch_stream* vs, const larch_tensor* x, const larch_tensor* w,
                               const larch_tensor* dy, float eps, const larch_tensor* dx, uint32_t accumulate) {
  return run("rmsnorm_backward", vs, [&](const stream& s) {
    const cpu::train_checked a = cpu::check_rmsnorm_backward(sizer(s), x, w, dy, dx);
    const view xv = a.a, wv = a.b, g = a.c, o = a.d;
    aligned({xv, wv, g, o});
    if (!xv.rows) return;
    launch(rms_back_k, rows_grid(xv.rows, 1, 256), 256, s.s, xv.as<float>(), xv.ld, wv.as<float>(), wv.ld,
           g.as<float>(), g.ld, o.as<float>(), o.ld, xv.cols, eps, int(accumulate != 0));
  });
}

__global__ void silu_back_k(const float* g, int64_t g_ld, const float* u, int64_t u_ld, const float* d, int64_t d_ld,
                            float* dg, int64_t dg_ld, float* du, int64_t du_ld, int64_t cols) {
  const int64_t r = blockIdx.x;
  for (int64_t c = blockIdx.y * int64_t(blockDim.x) + threadIdx.x; c < cols; c += int64_t(gridDim.y) * blockDim.x) {
    const float x = g[r * g_ld + c], sg = 1.0f / (1.0f + expf(-x)), dy = d[r * d_ld + c];
    dg[r * dg_ld + c] = dy * u[r * u_ld + c] * sg * (1.0f + x * (1.0f - sg));
    du[r * du_ld + c] = dy * x * sg;
  }
}

larch_status silu_mul_backward(larch_stream* vs, const larch_tensor* gate, const larch_tensor* up,
                                const larch_tensor* dout, const larch_tensor* dgate, const larch_tensor* dup) {
  return run("silu_mul_backward", vs, [&](const stream& s) {
    const cpu::train_checked a = cpu::check_silu_mul_backward(sizer(s), gate, up, dout, dgate, dup);
    const view g = a.a, u = a.b, d = a.c, dg = a.d, du = a.e;
    aligned({g, u, d, dg, du});
    if (!g.rows) return;
    launch(silu_back_k, rows_grid(g.rows, g.cols, 256), 256, s.s, g.as<float>(), g.ld, u.as<float>(), u.ld,
           d.as<float>(), d.ld, dg.as<float>(), dg.ld, du.as<float>(), du.ld, g.cols);
  });
}

// Block r: row r's softmax over its first `end` columns, zeros past them.
__global__ void softmax_k(const float* x, int64_t x_ld, float* out, int64_t o_ld, int64_t n, float scale, int causal) {
  __shared__ float part[32];
  const int64_t r = blockIdx.x, end = causal ? r % n + 1 : n;
  const float* xr = x + r * x_ld;
  float* o = out + r * o_ld;
  float mx = -INFINITY, z = 0;
  for (int64_t c = threadIdx.x; c < end; c += blockDim.x) mx = fmaxf(mx, xr[c] * scale);
  mx = vv_block_reduce_max(mx, part);
  for (int64_t c = threadIdx.x; c < end; c += blockDim.x) z += expf(xr[c] * scale - mx);
  z = vv_block_reduce_sum(z, part);
  for (int64_t c = threadIdx.x; c < n; c += blockDim.x) o[c] = c < end ? expf(xr[c] * scale - mx) / z : 0.0f;
}

larch_status softmax(larch_stream* vs, const larch_tensor* x, float scale, uint32_t causal, const larch_tensor* out) {
  return run("softmax", vs, [&](const stream& s) {
    const cpu::train_checked a = cpu::check_softmax(sizer(s), x, out);
    const view xv = a.a, o = a.b;
    aligned({xv, o});
    if (!xv.rows || !xv.cols) return;
    launch(softmax_k, rows_grid(xv.rows, 1, 256), 256, s.s, xv.as<float>(), xv.ld, o.as<float>(), o.ld, xv.cols, scale,
           int(causal != 0));
  });
}

__global__ void softmax_back_k(const float* p, int64_t p_ld, const float* dp, int64_t dp_ld, float* ds, int64_t ds_ld,
                               int64_t n, float scale) {
  __shared__ float part[32];
  const int64_t r = blockIdx.x;
  const float *pr = p + r * p_ld, *gr = dp + r * dp_ld;
  float sum = 0;
  for (int64_t c = threadIdx.x; c < n; c += blockDim.x) sum += pr[c] * gr[c];
  sum = vv_block_reduce_sum(sum, part);
  for (int64_t c = threadIdx.x; c < n; c += blockDim.x) ds[r * ds_ld + c] = scale * pr[c] * (gr[c] - sum);
}

larch_status softmax_backward(larch_stream* vs, const larch_tensor* p, const larch_tensor* dp, float scale,
                               const larch_tensor* ds) {
  return run("softmax_backward", vs, [&](const stream& s) {
    const cpu::train_checked a = cpu::check_softmax_backward(sizer(s), p, dp, ds);
    const view pv = a.a, g = a.b, o = a.c;
    aligned({pv, g, o});
    if (!pv.rows || !pv.cols) return;
    launch(softmax_back_k, rows_grid(pv.rows, 1, 256), 256, s.s, pv.as<float>(), pv.ld, g.as<float>(), g.ld,
           o.as<float>(), o.ld, pv.cols, scale);
  });
}

// out = a + b; out may be a.
__global__ void add_k(const float* a, int64_t a_ld, const __half* b, int64_t b_ld, float* out, int64_t o_ld,
                      int64_t cols) {
  const int64_t r = blockIdx.x;
  for (int64_t c = blockIdx.y * int64_t(blockDim.x) + threadIdx.x; c < cols; c += int64_t(gridDim.y) * blockDim.x)
    out[r * o_ld + c] = a[r * a_ld + c] + __half2float(b[r * b_ld + c]);
}

larch_status add(larch_stream* vs, const larch_tensor* a_f32, const larch_tensor* b_f16, const larch_tensor* out) {
  return run("add", vs, [&](const stream& s) {
    const cpu::elementwise_checked c = cpu::check_add(sizer(s), a_f32, b_f16, out);
    const view a = c.a, b = c.b, o = c.out;
    aligned({a, b, o});
    if (!a.rows) return;
    launch(add_k, rows_grid(a.rows, a.cols, 256), 256, s.s, a.as<float>(), a.ld, b.as<__half>(), b.ld, o.as<float>(),
           o.ld, a.cols);
  });
}

// DFlash2's dynamic conv (larch.h) of block row r: t0 is the stage's first tap row of base and dyn, and an f32 out
// is added into.
template <class T>
__global__ void draft_conv_k(const __half* x, int64_t x_ld, const float* dyn, int64_t dyn_ld, const float* base,
                             int64_t base_ld, T* out, int64_t o_ld, int64_t d, int64_t groups, int64_t t0) {
  const int64_t r = blockIdx.x, group = d / groups;
  const float* dr = dyn + r * dyn_ld;
  for (int64_t c = blockIdx.y * int64_t(blockDim.x) + threadIdx.x; c < d; c += int64_t(gridDim.y) * blockDim.x) {
    const int64_t g = c / group;
    float v = 0.0f;
    if constexpr (std::is_same_v<T, float>) v = out[r * o_ld + c];
    v += __half2float(x[r * x_ld + c]) * (base[t0 * base_ld + c] + dr[t0 * groups + g]);
    if (r > 0)
      v += __half2float(x[(r - 1) * x_ld + c]) * (base[(t0 + 1) * base_ld + c] + dr[(t0 + 1) * groups + g]);
    store(out + r * o_ld + c, v);
  }
}

larch_status draft_conv(larch_stream* vs, const larch_tensor* x, const larch_tensor* dyn, const larch_tensor* base,
                         uint32_t stage, const larch_tensor* out) {
  return run("draft_conv", vs, [&](const stream& s) {
    const cpu::draft_conv_checked c = cpu::check_draft_conv(sizer(s), x, dyn, base, stage, out);
    const view xv = c.x, dv = c.dyn, bv = c.base, o = c.out;
    aligned({xv, dv, bv, o});
    if (!xv.rows) return;
    const dim3 grid = rows_grid(xv.rows, xv.cols, 256);
    const int64_t groups = dv.shape[2], t0 = 2 * int64_t(stage);
    if (o.dtype == LARCH_F32)
      launch(draft_conv_k<float>, grid, 256, s.s, xv.as<__half>(), xv.ld, dv.as<float>(), dv.ld, bv.as<float>(), bv.ld,
             o.as<float>(), o.ld, xv.cols, groups, t0);
    else
      launch(draft_conv_k<__half>, grid, 256, s.s, xv.as<__half>(), xv.ld, dv.as<float>(), dv.ld, bv.as<float>(),
             bv.ld, o.as<__half>(), o.ld, xv.cols, groups, t0);
  });
}

// Row r of k and v to slot slots[r] of the pool layer ([slots, cols] each); a negative slot skips it.
__global__ void kv_write_k(const __half* k, int64_t k_ld, const __half* v, int64_t v_ld, const int32_t* slots,
                           int64_t slots_ld, __half* kp, __half* vp, int64_t cols) {
  const int64_t r = blockIdx.x, slot = slots[r * slots_ld];
  if (slot < 0) return;
  for (int64_t c = blockIdx.y * int64_t(blockDim.x) + threadIdx.x; c < cols; c += int64_t(gridDim.y) * blockDim.x) {
    kp[slot * cols + c] = k[r * k_ld + c];
    vp[slot * cols + c] = v[r * v_ld + c];
  }
}

larch_status kv_write(larch_stream* vs, const larch_tensor* k, const larch_tensor* v, const larch_tensor* slots_i32,
                       const larch_kv_pool* pool) {
  return run("kv_write", vs, [&](const stream& s) {
    const cpu::kv_write_checked c = cpu::check_kv_write(sizer(s), k, v, slots_i32, pool);
    const view kp = c.pool_k, vp = c.pool_v, kv = c.k, vv = c.v, slots = c.slots;
    aligned({kv, vv, slots});
    if (!kv.rows) return;
    launch(kv_write_k, rows_grid(kv.rows, kv.cols, 256), 256, s.s, kv.as<__half>(), kv.ld, vv.as<__half>(), vv.ld,
           slots.as<int32_t>(), slots.ld, kp.as<__half>(), vp.as<__half>(), kv.cols);
  });
}

constexpr int ATTN_WARPS = 4;
constexpr int ATTN_CHUNK = 256;  // output dims per block: wider heads take more blocks, each redoing the scores

struct attn_args {
  const __half *q, *k, *v;
  __half* out;
  int64_t q_ld, out_ld;
  const int32_t* table;
  int64_t table_ld;
  const int32_t* lens;   // decode: seq_lens [rows]; prefill: q_lens [seqs]
  const int32_t* start;  // prefill: q_start_pos [seqs]; decode: null
  int64_t lens_ld, start_ld, seqs, heads, group, hd, kv_heads, block_tokens, chunks;
  float scale;
};

#if !defined(__HIP_PLATFORM_AMD__)
#include "cuda/prefill.cuh"

static const int prefill_tc = tc_switch("LARCH_PREFILL_TC"), decode_tc = tc_switch("LARCH_DECODE_TC");

// attn_full on tensor cores: a CTA is (64 query rows, one head, one segment); each of its four warps owns 16 rows and
// keeps Q, O and the softmax state in registers, with attn_prefill_tc_k's fragments and its two-panel P (prefill.cuh).
// K and V of the segment pass through shared memory 16 rows at a time, chunk-major as there; head_dim 64, every
// position of the segment seen (a ViT's attention), so the only mask is the segment's end. 12 KB of shared memory.
// Non-finite inputs follow the reference as attn_prefill_tc_k does: a -Inf score weighs 0, and a tile whose V holds an
// Inf or NaN meets O in FP32 rather than in the two panels (whose low half would add 0 x Inf).
// ponytail: one stage of K/V; double-buffer it with cp.async as attn_prefill_tc_k does if an nsys run shows the loads.
namespace full_tc {
constexpr int ROWS = 64, KT = 16, HD = 64;
constexpr int Q_BYTES = ROWS * HD * 2, KV_BYTES = KT * HD * 2, SMEM = Q_BYTES + 2 * KV_BYTES;
}  // namespace full_tc

__global__ void __launch_bounds__(128) attn_full_tc_k(const __half* q, int64_t q_ld, const __half* k, int64_t k_ld,
                                                      const __half* v, int64_t v_ld, __half* out, int64_t o_ld, int seg,
                                                      float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int ROWS = full_tc::ROWS, KT = full_tc::KT, HD = full_tc::HD;
  __shared__ __align__(128) uint8_t tiles[full_tc::SMEM];
  const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int64_t first = int64_t(blockIdx.z) * seg;  // the segment's first row
  const int h = int(blockIdx.y), row0 = int(blockIdx.x) * ROWS, end = min(seg, row0 + ROWS);
  const uint32_t base = uint32_t(__cvta_generic_to_shared(tiles));
  // Q chunk-major: the 16-byte chunk c (dims 8c..) of row r at (c * ROWS + r) * 16; rows past the segment are 0.
  for (int i = tid; i < ROWS * HD / 8; i += 128) {
    const int r = row0 + i % ROWS, c = i / ROWS;
    uint4 x = make_uint4(0, 0, 0, 0);
    if (r < end) x = *reinterpret_cast<const uint4*>(q + (first + r) * q_ld + h * HD + c * 8);
    *reinterpret_cast<uint4*>(tiles + i * 16) = x;
  }
  __syncthreads();
  uint32_t qf[HD / 16][4];
#pragma unroll
  for (int k = 0; k < HD / 16; ++k)
    prefill::ldmatrix<false>(qf[k], base + (lane / 16 * ROWS + warp * 16 + lane % 16) * 16 + k * 2 * ROWS * 16);
  // Rows g (x = 0) and g + 8 (x = 1) of this warp; o[n] holds their columns 8n + 2t and 8n + 2t + 1.
  float o[HD / 8][4] = {}, m[2] = {-INFINITY, -INFINITY}, l[2] = {};
  const uint32_t k_at = full_tc::Q_BYTES, v_at = k_at + full_tc::KV_BYTES;
  const uint32_t k_lane = (lane / 8 * KT + lane % 8) * 16, v_lane = (lane / 16 * KT + lane % 16) * 16;
  // This thread stages position tid % KT, chunk tid / KT, of each K and V tile (128 threads, 16 x 8 chunks).
  const int copy_pos = tid % KT, copy_chunk = tid / KT;
  // The FP32 P of this warp's 16 rows for a tile with a non-finite V, in the warp's part of the Q area (Q is in
  // registers by the first tile's barrier): 16 x 17 floats of 2 KB.
  constexpr int LD = KT + 1;
  float* const fp32_p = reinterpret_cast<float*>(tiles + warp * 16 * HD * 2);
  for (int p0 = 0; p0 < seg; p0 += KT) {
    __syncthreads();  // the last tile is consumed
    bool bad = false;  // a non-finite V in this thread's copy
    {
      const int p = p0 + copy_pos;
      uint4 kx = make_uint4(0, 0, 0, 0), vx = kx;
      if (p < seg) {
        kx = *reinterpret_cast<const uint4*>(k + (first + p) * k_ld + h * HD + copy_chunk * 8);
        vx = *reinterpret_cast<const uint4*>(v + (first + p) * v_ld + h * HD + copy_chunk * 8);
      }
      *reinterpret_cast<uint4*>(tiles + k_at + (copy_chunk * KT + copy_pos) * 16) = kx;
      *reinterpret_cast<uint4*>(tiles + v_at + (copy_chunk * KT + copy_pos) * 16) = vx;
      prefill::scan_nonfinite(vx, bad);
    }
    const bool nonfinite = __syncthreads_or(bad);
    const int count = min(KT, seg - p0);
    float s[2][4] = {};
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int k2 = 0; k2 < HD / 32; ++k2) {
        uint32_t kf[4];
        prefill::ldmatrix<false>(kf, base + k_at + k_lane + (k2 * 4 * KT + j * 8) * 16);
        prefill::mma(s[j], qf[2 * k2], kf[0], kf[1]);
        prefill::mma(s[j], qf[2 * k2 + 1], kf[2], kf[3]);
      }
    float down[2];
    uint32_t ph[4], pl[4];
#pragma unroll
    for (int x = 0; x < 2; ++x) {
      float top = -INFINITY;
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e)
          if (8 * j + 2 * t + e < count) top = fmaxf(top, __fmul_rn(s[j][2 * x + e], scale));
      top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
      top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
      const float next = fmaxf(m[x], top), corr = l[x] == 0 ? 0.0f : expf(m[x] - next);
      float largest = 0;
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float& p = s[j][2 * x + e];
          const float score = __fmul_rn(p, scale), weight = expf(__fsub_rn(score, next));
          // -Inf weighs 0 also while next is -Inf (exp(-Inf + Inf) = NaN), as in attn_prefill_tc_k.
          p = 8 * j + 2 * t + e < count && score != -INFINITY ? weight : 0.0f;
          largest = fmaxf(largest, p);
        }
      largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, 1));
      largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, 2));
      float up;
      prefill::scales(largest, up, down[x]);
      float total = 0;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        __half high[2], low[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          total += s[j][2 * x + e];
          prefill::hi_lo(s[j][2 * x + e], up, high[e], low[e]);
        }
        ph[j * 2 + x] = prefill::pack(high[0], high[1]);
        pl[j * 2 + x] = prefill::pack(low[0], low[1]);
      }
      total += __shfl_xor_sync(0xffffffffu, total, 1);
      total += __shfl_xor_sync(0xffffffffu, total, 2);
      l[x] = l[x] * corr + total;
      m[x] = next;
#pragma unroll
      for (int n = 0; n < HD / 8; ++n) o[n][2 * x] *= corr, o[n][2 * x + 1] *= corr;
    }
    if (!nonfinite) {
#pragma unroll
      for (int n = 0; n < HD / 8; n += 2) {
        uint32_t vf[4];  // V fragments of dims 8n..8n+7, then 8n+8..8n+15
        prefill::ldmatrix<true>(vf, base + v_at + v_lane + n * KT * 16);
        float high[2][4] = {}, low[2][4] = {};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          prefill::mma(high[i], ph, vf[2 * i], vf[2 * i + 1]);
          prefill::mma(low[i], pl, vf[2 * i], vf[2 * i + 1]);
#pragma unroll
          for (int c = 0; c < 4; ++c) o[n + i][c] += (high[i][c] + low[i][c] * 0x1p-11f) * down[c / 2];
        }
      }
    } else {
      // The segment's positions of this tile (every row sees them all) meet V in FP32, which needs no panels.
#pragma unroll
      for (int x = 0; x < 2; ++x)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int e = 0; e < 2; ++e) fp32_p[(g + 8 * x) * LD + 8 * j + 2 * t + e] = s[j][2 * x + e];
      __syncwarp();
#pragma unroll
      for (int x = 0; x < 2; ++x) {
        const float* weight = fp32_p + (g + 8 * x) * LD;
#pragma unroll
        for (int c = 0; c < HD / 8; ++c) {
          float2 add{};
          for (int p = 0; p < count; ++p) {
            const uint8_t* at = tiles + v_at + (c * KT + p) * 16 + 4 * t;
            const float2 vv = __half22float2(*reinterpret_cast<const __half2*>(at));
            add.x += weight[p] * vv.x;
            add.y += weight[p] * vv.y;
          }
          o[c][2 * x] += add.x;
          o[c][2 * x + 1] += add.y;
        }
      }
    }
  }
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const int r = row0 + warp * 16 + g + 8 * x;
    if (r >= end) continue;
    const float inv = 1.0f / l[x];
    __half* row = out + (first + r) * o_ld + h * HD + 2 * t;
#pragma unroll
    for (int c = 0; c < HD / 8; ++c)
      *reinterpret_cast<__half2*>(row + 8 * c) = __floats2half2_rn(o[c][2 * x] * inv, o[c][2 * x + 1] * inv);
  }
#endif
}

static const int attn_full_tc_switch = tc_switch("LARCH_ATTN_FULL_TC");

static bool attn_full_tc(const stream& s, const view& q, const view& k, const view& v, const view& o, int seg,
                         float scale) {
  // sm_80 code; 16-byte rows for the uint4 copies (head offsets are 128 bytes); LARCH_ATTN_FULL_TC=0 turns it off.
  const bool rows16 = ((uintptr_t(q.p) | uintptr_t(k.p) | uintptr_t(v.p) | uintptr_t(o.p)) & 15) == 0 &&
                      (q.ld | k.ld | v.ld) % 8 == 0 && o.ld % 2 == 0;
  if (attn_full_tc_switch == 0 || s.image_arch < 800 || q.shape[2] != full_tc::HD || !rows16) return false;
  const int64_t segs = q.rows / seg;
  const dim3 grid(unsigned((seg + full_tc::ROWS - 1) / full_tc::ROWS), unsigned(q.shape[1]), unsigned(segs));
  launch(attn_full_tc_k, grid, 128, s.s, q.as<__half>(), q.ld, k.as<__half>(), k.ld, v.as<__half>(), v.ld,
         o.as<__half>(), o.ld, seg, scale);
  return true;
}
#else
static bool attn_full_tc(const stream&, const view&, const view&, const view&, const view&, int, float) {
  return false;
}
#endif

// softmax(scale q k^T) v for one (row, head, chunk of head_dim) per block, online in f32: each warp
// takes every ATTN_WARPS-th position, then the warps merge their maxima, sums and accumulators.
// General fallback for unusual head sizes and HIP wave32/wave64.
// s += x, Kahan-compensated by c: a warp adds ~n/4 small weights onto one large running sum, and plain f32 rounding
// drifts one way (check_probability_underflow's lost residual). Non-finite sums keep c at 0 so Inf stays Inf, not NaN.
// CUDA keeps the plain sum (it passes there, and attn_k stays unchanged on the CUDA hot path); c then stays 0.
__device__ __forceinline__ void attn_kahan(float& s, float& c, float x) {
#if defined(__HIP_PLATFORM_AMD__)
  const float y = x - c, t = s + y;
  c = isfinite(t) ? (t - s) - y : 0.0f;
  s = t;
#else
  s += x;
#endif
}

__global__ void attn_k(attn_args a) {
  __shared__ float m_w[ATTN_WARPS], l_w[ATTN_WARPS], acc_w[ATTN_WARPS][ATTN_CHUNK];
  const int64_t chunk = blockIdx.x % a.chunks, h = blockIdx.x / a.chunks % a.heads, row = blockIdx.x / a.chunks / a.heads;
  int64_t seq = row, n = 0;
  if (a.start) {  // prefill: row j of sequence s sits at position start[s] + j
    int64_t first = 0;
    seq = 0;
    while (seq + 1 < a.seqs && first + a.lens[seq * a.lens_ld] <= row) first += a.lens[seq++ * a.lens_ld];
    n = a.start[seq * a.start_ld] + (row - first) + 1;
  } else {
    n = a.lens[row * a.lens_ld];
  }
  const int warp = int(threadIdx.x) / warpSize, lane = int(threadIdx.x) % warpSize;
  const int64_t kvh = h / a.group, d0 = chunk * ATTN_CHUNK, d1 = min(a.hd, d0 + ATTN_CHUNK);
  const __half* q = a.q + row * a.q_ld + h * a.hd;
  const int32_t* blocks = a.table + seq * a.table_ld;
  float m = -INFINITY, l = 0, lc = 0, acc[ATTN_CHUNK / 32] = {}, accc[ATTN_CHUNK / 32] = {};
  for (int64_t p = warp; p < n; p += ATTN_WARPS) {
    const int64_t at = ((int64_t(blocks[p / a.block_tokens]) * a.block_tokens + p % a.block_tokens) * a.kv_heads + kvh) * a.hd;
    float dot = 0;
    for (int64_t d = lane; d < a.hd; d += warpSize) dot += __half2float(q[d]) * __half2float(a.k[at + d]);
    // While every score so far is -Inf, so is m_new, and exp(-Inf + Inf) is NaN: a -Inf score weighs 0 and an unmoved
    // max rescales by 1, as under the reference's global max.
    float sc;
    {  // An FMA of sum * scale - m_new gives the product's rounding error, not 0: at scores near 5e10 (high finite q, k)
       // the max position weighs exp(-2000) = 0. HIP inlines __fmul_rn as a contractable multiply, so the pragma it is.
#if defined(__HIP_PLATFORM_AMD__)
#pragma clang fp contract(off)
#endif
      sc = vv_warp_reduce_sum(dot) * a.scale;
    }
    const float m_new = fmaxf(m, sc), corr = m == m_new ? 1.0f : expf(m - m_new),
                e = sc == -INFINITY ? 0.0f : expf(sc - m_new);
    l *= corr;
    lc *= corr;
    attn_kahan(l, lc, e);
#pragma unroll
    for (int i = 0; i < ATTN_CHUNK / 32; ++i)
      if (const int64_t d = d0 + lane + i * warpSize; d < d1) {
        acc[i] *= corr;
        accc[i] *= corr;
        attn_kahan(acc[i], accc[i], e * __half2float(a.v[at + d]));
      }
    m = m_new;
  }
  if (lane == 0) {
    m_w[warp] = m;
    l_w[warp] = l - lc;
  }
#pragma unroll
  for (int i = 0; i < ATTN_CHUNK / 32; ++i)
    if (const int64_t d = lane + i * warpSize; d0 + d < d1) acc_w[warp][d] = acc[i] - accc[i];
  __syncthreads();
  float mx = -INFINITY, f[ATTN_WARPS], total = 0;
  for (int w = 0; w < ATTN_WARPS; ++w) mx = fmaxf(mx, m_w[w]);
  for (int w = 0; w < ATTN_WARPS; ++w) total += l_w[w] * (f[w] = n > 0 ? expf(m_w[w] - mx) : 0.0f);
  for (int64_t d = d0 + threadIdx.x; d < d1; d += blockDim.x) {
    float o = 0;  // seq_len 0 is a padding row: zeros
    for (int w = 0; w < ATTN_WARPS; ++w) o += acc_w[w][d - d0] * f[w];
    a.out[row * a.out_ld + h * a.hd + d] = __float2half_rn(n > 0 ? o / total : 0.0f);
  }
}

// Positions per split of a row of n: its share of `share` splits capped at `span` (auto decode), else of `splits`.
__device__ __forceinline__ int64_t split_len(int64_t n, int splits, int share, int64_t span) {
  return span ? min((n + share - 1) / share, span) : (n + splits - 1) / splits;
}

// Query heads and token warps share K/V tiles; prefill keeps one warp per query.
// The fixed 32-lane specialization uses no architecture-specific instructions; HIP wave32 takes it, wave64 attn_k.
template <int HD, int QROWS, int QHEADS, int TOKEN_WARPS = 1, bool VECTOR = false>
__global__ void attn_tiled_k(attn_args a, int64_t rows, int splits, int share, int64_t span, float* partial) {
  constexpr int TILE = 16, DIMS = HD / 32;
  __shared__ __align__(16) __half keys[TILE][HD], values[TILE][HD];
  __shared__ int64_t seq_row[QROWS], len_row[QROWS];
  const int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32;
  const int query_warp = warp / TOKEN_WARPS, token_warp = warp % TOKEN_WARPS;
  const int qr = query_warp / QHEADS, gh = query_warp % QHEADS;
  const int64_t head_tiles = (a.group + QHEADS - 1) / QHEADS;
  int64_t block = blockIdx.x;
  const int split = int(block % splits);
  block /= splits;
  const int64_t htile = block % head_tiles, kvh = block / head_tiles % a.kv_heads;
  const int64_t row0 = block / head_tiles / a.kv_heads * QROWS, row = row0 + qr;
  const int64_t h = kvh * a.group + htile * QHEADS + gh;
  const bool valid = row < rows && h < (kvh + 1) * a.group;
  // A capped decode split past its row's length has no work, and the merge never reads it.
  if (QROWS == 1 && span) {
    const int64_t n = row < rows ? a.lens[row * a.lens_ld] : 0;
    if (split * split_len(n, splits, share, span) >= n) return;
  }
  if (gh == 0 && token_warp == 0 && lane == 0) {
    int64_t seq = row, n = 0;
    if (row < rows) {
      if (a.start) {
        int64_t first = 0;
        seq = 0;
        while (seq + 1 < a.seqs && first + a.lens[seq * a.lens_ld] <= row)
          first += a.lens[seq++ * a.lens_ld];
        n = a.start[seq * a.start_ld] + row - first + 1;
      } else {
        n = a.lens[row * a.lens_ld];
      }
    }
    seq_row[qr] = row < rows ? seq : -1;
    len_row[qr] = n;
  }
  float query[DIMS] = {}, acc[DIMS] = {}, m = -INFINITY, l = 0;
#pragma unroll
  for (int i = 0; i < DIMS; ++i)
    if (valid) query[i] = __half2float(a.q[row * a.q_ld + h * HD + lane + i * 32]);
  __syncthreads();
  // A tile may straddle a ragged sequence boundary; each sequence stages its own pages.
  for (int segment = 0; segment < QROWS; ++segment) {
    const int64_t seq = seq_row[segment];
    if (seq < 0 || (segment && seq == seq_row[segment - 1])) continue;
    int64_t n = 0;
    for (int r = segment; r < QROWS; ++r)
      if (seq_row[r] == seq) n = max(n, len_row[r]);
    const int64_t per_split = split_len(n, splits, share, span);
    const int64_t begin = split * per_split, end = min(n, begin + per_split);
    const bool active = valid && seq_row[qr] == seq;
    const int64_t query_end = active ? min(end, len_row[qr]) : begin;
    const int32_t* blocks = a.table + seq * a.table_ld;
    for (int64_t p0 = begin; p0 < end; p0 += TILE) {
      const int count = int(min(int64_t(TILE), end - p0));
      if constexpr (VECTOR) {
        // HD128 and checked 16-byte pool bases make every eight-half chunk aligned, including page tails.
        for (int i = int(threadIdx.x); i < count * (HD / 8); i += int(blockDim.x)) {
          const int p = i / (HD / 8), d = i % (HD / 8) * 8;
          const int64_t pos = p0 + p;
          const int64_t at = ((int64_t(blocks[pos / a.block_tokens]) * a.block_tokens +
                                pos % a.block_tokens) * a.kv_heads + kvh) * HD + d;
          *reinterpret_cast<uint4*>(&keys[p][d]) = *reinterpret_cast<const uint4*>(a.k + at);
          *reinterpret_cast<uint4*>(&values[p][d]) = *reinterpret_cast<const uint4*>(a.v + at);
        }
      } else {
        for (int i = int(threadIdx.x); i < count * HD; i += int(blockDim.x)) {
          const int p = i / HD, d = i % HD;
          const int64_t pos = p0 + p;
          const int64_t at = ((int64_t(blocks[pos / a.block_tokens]) * a.block_tokens +
                                pos % a.block_tokens) * a.kv_heads + kvh) * HD + d;
          keys[p][d] = a.k[at];
          values[p][d] = a.v[at];
        }
      }
      __syncthreads();
      if (active) {
        // Blocked sums: thousands of tiny weights would each round against the running total, so
        // a tile sums its own (rescaled like the total) and adds once. corr_tile defers the total's.
        float tile_acc[DIMS] = {}, tile_l = 0, corr_tile = 1;
        for (int p = token_warp; p < count && p0 + p < query_end; p += TOKEN_WARPS) {
          float dot = 0;
#pragma unroll
          for (int i = 0; i < DIMS; ++i) dot += query[i] * __half2float(keys[p][lane + i * 32]);
          float score;
          {  // As in attn_k: no FMA of the score's multiply into score - next.
#if defined(__HIP_PLATFORM_AMD__)
#pragma clang fp contract(off)
#endif
            score = vv_warp_reduce_sum(dot) * a.scale;
          }
          // As in attn_k: a -Inf score weighs 0 and an unmoved max rescales by 1, also while the max is -Inf.
          const float next = fmaxf(m, score), corr = m == next ? 1.0f : expf(m - next),
                      weight = score == -INFINITY ? 0.0f : expf(score - next);
          tile_l = tile_l * corr + weight;
          corr_tile *= corr;
#pragma unroll
          for (int i = 0; i < DIMS; ++i)
            tile_acc[i] = tile_acc[i] * corr + weight * __half2float(values[p][lane + i * 32]);
          m = next;
        }
        l = l * corr_tile + tile_l;
#pragma unroll
        for (int i = 0; i < DIMS; ++i) acc[i] = acc[i] * corr_tile + tile_acc[i];
      }
      __syncthreads();
    }
  }
  if constexpr (TOKEN_WARPS > 1) {
    // Token warps have independent online-softmax maxima. Merge in f32 before the external split.
    __shared__ float warp_partial[QROWS * QHEADS][TOKEN_WARPS][HD + 2];
    float* own = warp_partial[query_warp][token_warp];
    if (lane == 0) own[HD] = m, own[HD + 1] = l;
#pragma unroll
    for (int i = 0; i < DIMS; ++i) own[lane + i * 32] = acc[i];
    __syncthreads();
    if (token_warp != 0) return;
    m = -INFINITY;
#pragma unroll
    for (int w = 0; w < TOKEN_WARPS; ++w)
      if (warp_partial[query_warp][w][HD + 1] != 0)
        m = fmaxf(m, warp_partial[query_warp][w][HD]);
    l = 0;
    float factor[TOKEN_WARPS];
#pragma unroll
    for (int w = 0; w < TOKEN_WARPS; ++w) {
      const float sum = warp_partial[query_warp][w][HD + 1];
      factor[w] = sum != 0 ? expf(warp_partial[query_warp][w][HD] - m) : 0.0f;
      l += sum * factor[w];
    }
#pragma unroll
    for (int i = 0; i < DIMS; ++i) {
      acc[i] = 0;
#pragma unroll
      for (int w = 0; w < TOKEN_WARPS; ++w)
        acc[i] += warp_partial[query_warp][w][lane + i * 32] * factor[w];
    }
  }
  if (valid) {
    float* part = partial ? partial + ((row * a.heads + h) * splits + split) * (HD + 2) : nullptr;
    if (part && lane == 0) part[HD] = m, part[HD + 1] = l;
#pragma unroll
    for (int i = 0; i < DIMS; ++i) {
      const int d = lane + i * 32;
      if (part) part[d] = acc[i];
      // l is also 0 after -Inf scores alone: 0 / 0 is the reference's NaN, and only a padding row writes zeros.
      else a.out[row * a.out_ld + h * HD + d] = __float2half_rn(len_row[qr] ? acc[i] / l : 0.0f);
    }
  }
}

// Partial accumulators have independent softmax maxima; normalize only after rescaling them.
__global__ void attn_merge_k(attn_args a, int splits, int share, int64_t span, const float* partial) {
  const int64_t row = blockIdx.x / a.heads, h = blockIdx.x % a.heads;
  const float* base = partial + (row * a.heads + h) * splits * (a.hd + 2);
  const int64_t n = a.lens[row * a.lens_ld];
  // Fixed-span splits past the row's length exited without writing.
  if (span && n) {
    const int64_t len = split_len(n, splits, share, span);
    splits = int(min(int64_t(splits), (n + len - 1) / len));
  } else if (span) {
    splits = 0;
  }
  float mx = -INFINITY;
  for (int s = 0; s < splits; ++s)
    if (base[s * (a.hd + 2) + a.hd + 1] != 0) mx = fmaxf(mx, base[s * (a.hd + 2) + a.hd]);
  float total = 0;
  for (int s = 0; s < splits; ++s) {
    const float* p = base + s * (a.hd + 2);
    if (p[a.hd + 1] != 0) total += p[a.hd + 1] * expf(p[a.hd] - mx);
  }
  for (int64_t d = threadIdx.x; d < a.hd; d += blockDim.x) {
    float out = 0;
    for (int s = 0; s < splits; ++s) {
      const float* p = base + s * (a.hd + 2);
      // A split of -Inf scores alone has sum 0 but keeps its 0 x Inf V = NaN, as the reference does.
      out += p[d] * (p[a.hd + 1] != 0 ? expf(p[a.hd] - mx) : 0.0f);
    }
    // total is also 0 after -Inf scores alone: 0 / 0 is the reference's NaN.
    a.out[row * a.out_ld + h * a.hd + d] = __float2half_rn(n ? out / total : 0.0f);
  }
}

// Launches a decode/prefill kernel of attn_tiled_k's signature over `blocks` CTAs; split, its partials (one layout
// for every kernel, so the capture arena's budget holds) go through scratch to attn_merge_k.
template <class K>
static void attend_split(const stream& s, const attn_args& a, K kernel, int64_t blocks, int threads, int64_t rows,
                         int splits, int share, int64_t span) {
  if (blocks > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 attention tiles"};
  // No allocation or extra launch for the common unsplit path.
  if (splits > 1) {
    scratch part(uint64_t(rows * a.heads) * uint64_t(a.hd + 2) * unsigned(splits) * sizeof(float), s);
    launch(kernel, unsigned(blocks), threads, s.s, a, rows, splits, share, span, static_cast<float*>(part.p));
    launch(attn_merge_k, unsigned(rows * a.heads), 128, s.s, a, splits, share, span, static_cast<const float*>(part.p));
  } else {
    launch(kernel, unsigned(blocks), threads, s.s, a, rows, 1, 0, int64_t(0), static_cast<float*>(nullptr));
  }
}

template <int HD, int QROWS, int QHEADS, int TOKEN_WARPS = 1, bool VECTOR = false>
static void attend_tiled(const stream& s, const attn_args& a, int64_t rows, int splits, int share = 0, int64_t span = 0) {
  attend_split(s, a, attn_tiled_k<HD, QROWS, QHEADS, TOKEN_WARPS, VECTOR>,
               ((rows + QROWS - 1) / QROWS) * a.kv_heads * ((a.group + QHEADS - 1) / QHEADS) * splits,
               QROWS * QHEADS * TOKEN_WARPS * 32, rows, splits, share, span);
}

#if !defined(__HIP_PLATFORM_AMD__)
#include "cuda/decode.cuh"
#include "cuda/attn_hd256.cuh"
#endif

// Launches attention over the checked tensors (start is null for decode), after this backend's refusals.
static void attend(const stream& s, const cpu::attn_checked& c, const larch_kv_pool* pool, float scale, uint32_t requested_splits = 0) {
  const view q = c.q, out = c.out, table = c.table, k = c.k, v = c.v, start = c.start, lens = c.lens;
  aligned({q, out, table, start, lens});
  const int64_t heads = q.shape[1], hd = pool->head_dim, kv_heads = pool->n_kv_heads;
  const attn_args a{q.as<__half>(), k.as<__half>(), v.as<__half>(), out.as<__half>(), q.ld, out.ld,
                    table.as<int32_t>(), table.ld, lens.as<int32_t>(), start.as<int32_t>(), lens.ld, start.ld,
                    table.rows, heads, heads / kv_heads, hd, kv_heads, pool->block_tokens,
                    (hd + ATTN_CHUNK - 1) / ATTN_CHUNK, scale};
  const int64_t blocks = q.rows * a.heads * a.chunks;
  if (!blocks) return;
  if (blocks > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 (row, head) blocks"};

#if !defined(__HIP_PLATFORM_AMD__)
  // What the tensor-core kernels need of K/V: HD128 or HD256 pages of 16, sm_80 code, and 16-byte pool bases for
  // cp.async's 16-byte chunks. pool_layer has refused every pool but f16, and their static shared memory fits every
  // sm_80 device.
  const bool tc_kv = (hd == 128 || hd == 256) && a.block_tokens == 16 && s.image_arch >= 800 &&
                     ((uintptr_t(a.k) | uintptr_t(a.v)) & 15) == 0;
  // HD256 takes HD128's arch list but not its crowding rule. sm_120, 16q/2kv (Nov 2026), the GQA-packed tensor-core
  // kernel against the CUDA-core tiles, ms: one sequence of 512 rows after 8k/32k/64k/128k history 2.06/7.66/14.5/29.1
  // against 29.3/126/253/505; after 4k, rows x sequences 64 x 8 (crowded at HD128) 0.79 against 3.21, 64 x 2 0.85
  // against 3.18, 256 x 64 0.98 against 9.89, 16 x 16 3.00 against 11.27.
  // HD256 packs a GQA group into its 64-row tile, so it takes groups to 64 (and rows * 64 under 2^31).
  if (tc_kv && a.start && (hd == 128 || (a.group <= hd256::M && q.rows <= INT_MAX / hd256::M)) &&
      arch::prefill_tc(prefill_tc, s.arch_code, q.rows, hd == 256 ? 1 : a.seqs)) {
    if (hd == 256) {
      // The GQA-packed kernel (a CTA is one kv head x TM / group query rows) where its full or half tile fills the
      // SMs; under that, each of its CTAs walks the history with too few warps, and the per-head kernel's eight
      // warps a CTA win (128 rows after 110k history: 11.7 against 20.9 ms; 256 rows: 23.3 against 21.1).
      const auto ctas = [&](int64_t tm) { return (q.rows + tm / a.group - 1) / (tm / a.group) * kv_heads; };
      constexpr int full = hd256::M, half = full / 2;
      if (ctas(full) >= s.compute_units || a.group > half)
        launch(attn_prefill_hd256_k<full>, unsigned(ctas(full)), 2 * full, s.s, a, q.rows);
      else if (ctas(half) >= s.compute_units)
        launch(attn_prefill_hd256_k<half>, unsigned(ctas(half)), 2 * half, s.s, a, q.rows);
      else
        launch(attn_prefill_hd256_head_k, unsigned((q.rows + hd256::HEAD_ROWS - 1) / hd256::HEAD_ROWS * heads), 256,
               s.s, a, q.rows);
    }
    else attend_prefill_tc(s, a, q.rows);
    return;
  }
#endif
  // Staged K/V tiles beat attn_k's one-position-at-a-time loads on cold decode at every context
  // (sm_120, 15 Oct 2026, m1 heads: seq 128 rows 1 6.7 vs 40.0 us, seq 2048 rows 16 244 vs 643 us).
  if (s.warp == 32 && (hd == 64 || hd == 128 || hd == 256)) {
    // The tensor-core decode takes a whole GQA group per CTA.
    // ponytail: hd 128 and 256 and groups to 8 (a group's heads fill rows 0-7 of the mma's 16), m1's, m5's and the
    // HD256 16q/2kv shapes; other head dims and groups keep the CUDA-core decode. Fill rows 8-15 for groups to 16 when a
    // catalog model with such a shape shows decode attention in a W2/W4 profile.
#if defined(__HIP_PLATFORM_AMD__)
    const bool tc = false;  // the tensor-core kernels are CUDA's (mma, cp.async); HIP wave32 takes the CUDA-core tiles
#else
    const bool tc = tc_kv && !a.start && a.group <= 8 && arch::decode_tc(decode_tc, s.arch_code);
#endif
    int splits = 1, share = 0;
    int64_t span = 0;
    if (!a.start) {
      const int64_t capacity = std::min(table.cols, int64_t(pool->n_blocks)) * pool->block_tokens;
      // CTAs per split: one per (row, kv head) on the tensor-core decode, per 4-head tile on the CUDA-core one.
      const int64_t groups = q.rows * kv_heads * (tc ? 1 : (a.group + 3) / 4);
      const int64_t occupancy = (int64_t(s.compute_units) * 2 + groups - 1) / groups;
      // The HD256 tensor-core decode takes a share of at most 16 splits, capped at 128 positions each: a long row
      // reaches 64 splits by 8k (as Metal's decode since #423) while short rows keep 16 or fewer.
      // ponytail: the default on sm_120 since Nov 2026, timed only as a whole against the CUDA-core decode (arch.h
      // decode_tc); 128 against 64/256 positions and the share of 16 are untimed. The HD128 and CUDA-core grids keep
      // 512 and 64, measured below. Time 64/128/256 positions at 2k/8k (attention --bench decode hd256) on the 5070 Ti
      // if HD256 decode attention shows in a W2/W4 profile.
      const bool hd256 = tc && hd == 256;
      const int64_t least = hd256 ? 128 : 512;
      const int64_t useful = std::max(int64_t(1), (capacity + least - 1) / least);
      if (requested_splits) {
        splits = int(std::min(int64_t(64), std::max(int64_t(1), std::min(int64_t(requested_splits), capacity))));
      } else {
        // Rows keep their share of the splits, but no CTA takes more than `span` positions: a long row in a
        // ragged batch gets more CTAs instead of holding the whole step (graphs fix the grid, so the cap is
        // in tokens of table capacity; splits past a row's length exit at once and are never merged).
        // ponytail: span 512 and 2 CTAs an SM. CUDA-core decode (sm_120 m1, 16 Oct 2026): against 384, fixed
        // 640-token decode on a 512-block table +0.3 to +2.2 % at 1-32 rows; 384 was -15 to -20 % on ragged
        // 160-2560-token batches against uncapped. Tensor-core decode (sm_120 m1/m5, 22 Oct 2026): span 1024 won
        // up to 4 % on uniform 8-32 rows but lost up to 13 % on ragged batches, 256 lost up to 31.6 % at 1 row,
        // 4 CTAs an SM up to 17 % at 8-16 rows x 1k-4k. W4's 2560-token table leaves m1's 1-4 rows (m5's 1-8) 5
        // splits: a share of 8 was 1-28 % faster at 256-2560 tokens but 2-17 % slower at 160, and a grid past the
        // share (span 160) up to 14 % slower at 160-512. Re-measure when split_len takes a minimum split length
        // (short rows would then keep few splits under a larger share), or W4's context mix changes.
        share = int(std::min({occupancy, useful, int64_t(hd256 ? 16 : 64)}));
        // At most 64 splits, and their partials within 16 MiB so a capture's arena holds them at any batch;
        // the span then widens until the splits cover the capacity.
        const int64_t bytes = q.rows * heads * (hd + 2) * int64_t(sizeof(float));
        const int64_t most = std::max(int64_t(share), std::min(int64_t(64), (int64_t(16) << 20) / bytes));
        splits = int(std::min(most, std::max(int64_t(share), useful)));
        span = std::max(least, (capacity + splits - 1) / splits);
      }
    }
#if !defined(__HIP_PLATFORM_AMD__)
    if (tc) {
      attend_split(s, a, hd == 256 ? attn_decode_hd256_k : attn_decode_tc_k, q.rows * kv_heads * splits, 128, q.rows,
                   splits, share, span);
      return;
    }
#endif
#define VV_ATTN_TILE(HD) \
    if (a.start) attend_tiled<HD, 4, 2>(s, a, q.rows, 1); \
    else attend_tiled<HD, 1, 4, 4>(s, a, q.rows, splits, share, span)
    if (hd == 64) { VV_ATTN_TILE(64); }
    else if (hd == 128) {
      // ponytail: vector K/V loads and 8-row prefill tiles measured on sm_120 and sm_86 (m6 heads, CUDA-core prefill
      // 1.27-1.45x from 64 rows on sm_86); sm_89 by judgment (16-byte loads and 8-row tiles are arch-generic and won on
      // both neighbours), confirm with the 4060's attention timings.
      // The sm_86 numbers are a 16-SM 3050 Laptop's: re-measure on a large sm_86 part (e.g. a 3090, 82 SMs) when one enters a bench.
#if defined(__HIP_PLATFORM_AMD__)
      const bool tuned = false;  // measured on NVIDIA only
#else
      const bool tuned = s.arch_code == 86 || s.arch_code == 89 || s.arch_code == 120;
#endif
      const bool vector = tuned && ((uintptr_t(a.k) | uintptr_t(a.v)) & 15) == 0;
      if (a.start && q.rows >= 64 && tuned) {
        if (vector) attend_tiled<128, 8, 2, 1, true>(s, a, q.rows, 1);
        else attend_tiled<128, 8, 2>(s, a, q.rows, 1);
      } else if (!a.start && vector) {
        attend_tiled<128, 1, 4, 4, true>(s, a, q.rows, splits, share, span);
      } else { VV_ATTN_TILE(128); }
    }
    // ponytail: since the leading -Inf guards, attn_tiled_k<256,1,4,4> takes 67 registers on sm_120 (was 64): 1 CTA an
    // SM, not 2 (sm_86/89 keep 64, 2 CTAs). It serves HD256 off the tensor-core decode (LARCH_DECODE_TC=0, sm_86/89
    // and groups past 8); try __launch_bounds__ with 2 blocks or a register trim, keeping STACK:0, if HD256 decode on
    // those shows it in a profile.
    else { VV_ATTN_TILE(256); }
#undef VV_ATTN_TILE
    return;
  }
  launch(attn_k, unsigned(blocks), unsigned(ATTN_WARPS * s.warp), s.s, a);
}

larch_status attn_decode(larch_stream* vs, const larch_tensor* q, const larch_kv_pool* pool,
                          const larch_tensor* block_table_i32, const larch_tensor* seq_lens_i32, uint32_t splits,
                          float scale, const larch_tensor* out) {
  return run("attn_decode", vs, [&](const stream& s) {
    attend(s, cpu::check_attn_decode(sizer(s), q, pool, block_table_i32, seq_lens_i32, out), pool, scale, splits);
  });
}

larch_status attn_prefill(larch_stream* vs, const larch_tensor* q, const larch_kv_pool* pool,
                           const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                           const larch_tensor* q_lens_i32, float scale, const larch_tensor* out) {
  return run("attn_prefill", vs, [&](const stream& s) {
    attend(s, cpu::check_attn_prefill(sizer(s), q, pool, block_table_i32, q_start_pos_i32, q_lens_i32, out), pool,
           scale);
  });
}

// Row r of a span as decode sees it: its sequence's block table row, and seq_len q_start + j + 1 for its row j.
__global__ void span_rows_k(const int32_t* start, int64_t start_ld, const int32_t* q_lens, int64_t lens_ld,
                            int64_t seqs, const int32_t* table, int64_t table_ld, int64_t cols, int32_t* row_table,
                            int32_t* row_lens) {
  const int64_t row = blockIdx.x;
  int64_t seq = 0, first = 0;
  while (seq + 1 < seqs && first + q_lens[seq * lens_ld] <= row) first += q_lens[seq++ * lens_ld];
  if (threadIdx.x == 0) row_lens[row] = int32_t(start[seq * start_ld] + (row - first) + 1);
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) row_table[row * cols + c] = table[seq * table_ld + c];
}

// The verify contract by construction: each span row runs as a decode row (attend: decode's paths and split rule)
// over its own copy of its sequence's table row, so at nonzero splits it is decode of that row alone; splits 0 is
// decode's auto rule over the span's rows, a speed hint as larch.h says.
// ponytail: rows x table-width i32 of scratch for the row copies, and the decode kernels for every row; fine for
// check spans (tens of rows), not for prefill-sized spans. Trigger: an engine path sending prefill through
// larch_attn (then a row-to-sequence index in the decode kernels).
larch_status attn(larch_stream* vs, const larch_tensor* q, const larch_kv_pool* pool,
                   const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                   const larch_tensor* q_lens_i32, const larch_tensor* mask_or_null,
                   const larch_tensor* sinks_or_null, uint32_t window, uint32_t splits, float scale,
                   const larch_tensor* out) {
  return run("attn", vs, [&](const stream& s) {
    cpu::attn_checked c = cpu::check_attn(sizer(s), q, pool, block_table_i32, q_start_pos_i32, q_lens_i32,
                                          mask_or_null, sinks_or_null, out);
    // ponytail: causal spans only. A mask's bits are device data, so not even a causal chain's can be told from a
    // tree's without a sync, and a tree keeps decode's bytes only over compacted positions; no catalog model uses a
    // window or sinks. Trigger: plan v2 task T1 (tree verify).
    if (c.mask.buf || c.sinks.buf || window)
      throw failure{LARCH_ERR_UNSUPPORTED, "a mask, window or sinks (causal spans only)"};
    const view start = c.start, lens = c.lens, table = c.table;
    aligned({c.q, c.out, start, lens, table});  // attend's own check, before anything is enqueued
    const int64_t rows = c.q.rows, cols = table.cols;
    if (!rows) return;
    const dim3 grid = rows_grid(rows, 1, 128);
    // attend's refusals at their worst case, so a refused call leaves no span_rows_k behind: 64 splits at most.
    // ponytail: a capture that would auto-split to fewer is refused too, from 1 MiB of partials a split (about 2000
    // row-heads at hd 128), far above a check step's 16 rows. Pass attend's split count here if a capture needs that.
    const int64_t heads = c.q.shape[1], hd = pool->head_dim, most = splits ? std::min<int64_t>(splits, 64) : 64;
    if (rows * heads * std::max<int64_t>(most, (hd + ATTN_CHUNK - 1) / ATTN_CHUNK) > INT_MAX)
      throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 attention blocks"};
    const scratch per_row(uint64_t(rows) * uint64_t(cols + 1) * sizeof(int32_t), s);
    if (s.capturing && most > 1) {
      const scratch partials(uint64_t(rows * heads) * uint64_t(hd + 2) * uint64_t(most) * sizeof(float), s);
    }
    buffer b{};  // per_row as a buffer, so the decode views below read it
    b.ptr = static_cast<uint8_t*>(per_row.p);
    int32_t* const row_table = reinterpret_cast<int32_t*>(b.ptr);
    launch(span_rows_k, grid, 128, s.s, start.as<int32_t>(), start.ld, lens.as<int32_t>(), lens.ld, table.rows,
           table.as<int32_t>(), table.ld, cols, row_table, row_table + rows * cols);
    c.table = {&b, 0, LARCH_I32, 2, {rows, cols}, rows, cols, cols, cols * 4};
    c.lens = {&b, uint64_t(rows * cols * 4), LARCH_I32, 1, {rows}, rows, 1, 1, 4};
    c.start = {};
    attend(s, c, pool, scale, splits);
  });
}

// Rows the row-stable gemm takes: Transformer::gemm slices a check step's side weights at 8 rows, so 8 is all the
// row contract needs and plain decodes of 9 or more rows stay on cuBLAS. Below 17 rows a quantized layer takes the
// direct quant routes or mmvq, never gemm. The one unsliced caller is a gated layer's 0/1 selector (rows x heads
// rows, on either side of the cap): it copies each element exactly, so its rows agree across both routes, but for
// the sign of a zero, which no later score sees.
// ponytail: raise this with the engine's slice (GEMV_ROWS) if that ever passes 8. (GEMM_ROWS, cuda.cuh)

// c = alpha a b_t^T + beta c + bias for up to GEMM_ROWS rows, one warp per output column: lane l sums the 8-wide k
// chunks l, l + 32, ... in k order and the warp's butterfly adds the lanes, so a row's sum order depends on k alone,
// never on how many rows share the call (cuBLAS promises no such thing across m). The epilogue is quant.cu's bias_k.
// ponytail: one warp per column and no k split, so n of 128 fills 16 CTAs; split k at a fixed partition (k alone
// picks it) when gemm_rows_k shows in a decode profile.
template <class O>
__global__ void __launch_bounds__(256) gemm_rows_k(const __half* a, int64_t a_ld, const __half* b, int64_t b_ld,
                                                   const float* bias, int64_t bias_ld, O* c, int64_t c_ld, int rows,
                                                   int64_t n, int64_t k, float alpha, float beta) {
  const int64_t col = int64_t(blockIdx.x) * 8 + threadIdx.x / 32;
  const int lane = int(threadIdx.x) % 32;
  if (col >= n) return;
  float acc[GEMM_ROWS] = {};
  for (int64_t c0 = int64_t(lane) * 8; c0 < k; c0 += 256) {
    const int count = int(min(int64_t(8), k - c0));
    const __half* const w = b + col * b_ld + c0;
#pragma unroll
    for (int r = 0; r < GEMM_ROWS; ++r) {
      if (r >= rows) break;
      const __half* const x = a + r * a_ld + c0;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (i < count) acc[r] = fmaf(__half2float(x[i]), __half2float(w[i]), acc[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < GEMM_ROWS; ++r) {
    if (r >= rows) break;
    const float sum = vv_warp_reduce_sum(acc[r]);
    if (lane) continue;
    float v = alpha * sum;
    if (beta != 0.0f) v += beta * float(c[r * c_ld + col]);
    if (bias) v += bias[col * bias_ld];
    store(c + r * c_ld + col, v);
  }
}

// larch_gemm: up to GEMM_ROWS rows on gemm_rows_k, so check rows equal decode's; more rows, and wave64, on
// quant.cu's cuBLAS gemm, which checks the call itself.
larch_status gemm_rows(larch_stream* vs, const larch_tensor* a_f16, const larch_tensor* b_f16_t,
                        const larch_tensor* bias_or_null, const larch_tensor* c, float alpha, float beta) {
  if (!a_f16 || a_f16->shape[0] > GEMM_ROWS || static_cast<const stream*>(vs)->warp != 32)
    return gemm(vs, a_f16, b_f16_t, bias_or_null, c, alpha, beta);
  return run("gemm", vs, [&](const stream& s) {
    const cpu::matmul_checked m = cpu::check_gemm(sizer(s), a_f16, b_f16_t, bias_or_null, c);
    const view a = m.a, bt = m.b_t, bias = m.bias, cv = m.c;
    aligned({a, bt, bias, cv});
    if (!a.rows) return;
    if ((bt.rows + 7) / 8 > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^34 output columns"};
    const unsigned blocks = unsigned((bt.rows + 7) / 8);
    const float* bp = bias.as<float>();
    if (cv.dtype == LARCH_F32)
      launch(gemm_rows_k<float>, blocks, 256, s.s, a.as<__half>(), a.ld, bt.as<__half>(), bt.ld, bp, bias.ld,
             cv.as<float>(), cv.ld, int(a.rows), bt.rows, a.cols, alpha, beta);
    else
      launch(gemm_rows_k<__half>, blocks, 256, s.s, a.as<__half>(), a.ld, bt.as<__half>(), bt.ld, bp, bias.ld,
             cv.as<__half>(), cv.ld, int(a.rows), bt.rows, a.cols, alpha, beta);
  });
}

constexpr int TOPK = 64;           // larch.h's candidates per row
// Slots one select block holds in shared memory. m1 decode select us/step at rows 1 / 16: 10.3 / 20.6 here
// (2 launches), 12.0 / 27.9 at 2048 (3 launches).
constexpr int SEL_N = 4096;
constexpr int SEL_THREADS = 1024;  // a multiple of every warp size
constexpr int SEL_PER = SEL_N / SEL_THREADS;

struct cand {
  float v;
  int32_t id;
};

// The top-k order as one unsigned key: larger value first, ties to the lower id, -0.0 equal to +0.0
// (the sign bit survives in `cand`, so a -0.0 logit still comes out as -0.0). Padding {-inf, INT_MAX}
// keys as 0 in the id half, below every real -inf, and only padding keys can be equal.
__device__ __forceinline__ uint64_t key_of(cand c) {
  const uint32_t b = __float_as_uint(c.v), z = b << 1 ? b : 0u;
  return uint64_t(z & 0x80000000u ? ~z : z | 0x80000000u) << 32 | uint32_t(INT_MAX - c.id);
}

// per_row of up to TOPK rows, passed by value: it is host memory the caller may reuse on return.
struct row_params {
  float penalty[TOPK];
  uint32_t window[TOPK];  // 0 without penalty ids
  uint32_t k[TOPK];
};

// One select pass over n entries per row: logits (in == null, the first pass) or the previous pass's
// candidates. The last pass (one chunk per row) writes the outputs, earlier ones `out`.
struct sel_args {
  const float* logits;
  int64_t l_ld;
  const cand* in;
  int64_t n;
  const int32_t* pen;
  int64_t pen_ld;
  cand* out;
  float2* part;  // (max, sum e^(x - max)) per first-pass chunk, row-major [rows, chunks of vocab]
  int64_t vocab;
  float* vals;
  int64_t vals_ld;
  int32_t* ids;
  int64_t ids_ld;
  float* lse;
  int64_t lse_ld;
};

// Block (c, r) selects the TOPK largest keys of chunk c of row r by radix, 8 bits a digit from the top,
// and writes them in order: to out[r][c], or, in the last pass, to vals / ids (-inf, -1 from k on) with
// lse from the first pass's part. The first pass also applies the repetition penalty and writes part.
template <int SELECT_K>
__global__ void __launch_bounds__(SEL_THREADS) select_k(sel_args a, row_params p) {
  __shared__ cand s[SEL_N], sel[SELECT_K];
  __shared__ uint64_t selk[SELECT_K];
  __shared__ unsigned hist[2][256], pick[3], cnt[2];
  __shared__ float red[32];
  const int64_t r = blockIdx.y, c0 = int64_t(blockIdx.x) * SEL_N, chunk = r * gridDim.x + blockIdx.x;
  for (int i = threadIdx.x; i < SEL_N; i += SEL_THREADS) {
    const int64_t g = c0 + i;
    s[i] = g >= a.n ? cand{-INFINITY, INT_MAX} : a.in ? a.in[r * a.n + g] : cand{a.logits[r * a.l_ld + g], int32_t(g)};
  }
  for (int i = threadIdx.x; i < 256; i += SEL_THREADS) hist[0][i] = 0;
  if (threadIdx.x < 2) cnt[threadIdx.x] = 0;
  __syncthreads();
  if (!a.in) {
    // Each distinct id once: the window entry that is its first occurrence applies it.
    for (uint32_t i = threadIdx.x; a.pen && i < p.window[r]; i += SEL_THREADS) {
      const int32_t id = a.pen[r * a.pen_ld + i];
      if (id < c0 || id >= c0 + SEL_N || id >= a.n) continue;
      bool first = true;
      for (uint32_t j = 0; j < i && first; ++j) first = a.pen[r * a.pen_ld + j] != id;
      if (!first) continue;
      float& x = s[id - c0].v;
      x = x <= 0 ? x * p.penalty[r] : x / p.penalty[r];
    }
    __syncthreads();
    float mx = -INFINITY, sum = 0;
    for (int i = threadIdx.x; i < SEL_N; i += SEL_THREADS) mx = fmaxf(mx, s[i].v);
    mx = vv_block_reduce_max(mx, red);
    for (int i = threadIdx.x; mx > -INFINITY && i < SEL_N; i += SEL_THREADS) sum += expf(s[i].v - mx);
    sum = vv_block_reduce_sum(sum, red);
    if (threadIdx.x == 0) a.part[chunk] = make_float2(mx, sum);
  }
  uint64_t key[SEL_PER];
  for (int k = 0; k < SEL_PER; ++k) key[k] = key_of(s[threadIdx.x + k * SEL_THREADS]);
  // Digit by digit: histogram the keys still in the running (top bits == prefix), one warp finds the
  // bucket holding the need-th largest, the search narrows to it. Done when that bucket is exactly
  // what is still needed, or the key is exhausted (equal keys left are padding).
  uint64_t prefix = 0, mask = 0;
  unsigned need = SELECT_K;
  for (int shift = 56, it = 0;; shift -= 8, ++it) {
    unsigned (&h)[256] = hist[it & 1];
    for (int i = threadIdx.x; i < 256; i += SEL_THREADS) hist[(it + 1) & 1][i] = 0;  // the next digit's, read last two syncs ago
    for (int k = 0; k < SEL_PER; ++k)
      if ((key[k] & mask) == prefix) atomicAdd(&h[unsigned(key[k] >> shift) & 255], 1u);
    __syncthreads();
    if (threadIdx.x < warpSize) {
      // Lane l owns a run of bins below lane l - 1's; an inclusive scan over lanes counts from the top.
      const int lane = int(threadIdx.x), per = 256 / warpSize, top = 255 - lane * per;
      unsigned own = 0;
      for (int b = 0; b < per; ++b) own += h[top - b];
      unsigned incl = own;
      for (int o = 1; o < warpSize; o <<= 1) {
        const unsigned u = __shfl_up_sync(VV_FULL_MASK, incl, o);
        if (lane >= o) incl += u;
      }
      if (const unsigned excl = incl - own; excl < need && need <= incl) {
        unsigned run = excl;
        int b = top;
        while (run + h[b] < need) run += h[b--];
        pick[0] = unsigned(b), pick[1] = run, pick[2] = h[b];
      }
    }
    __syncthreads();
    need -= pick[1];
    prefix |= uint64_t(pick[0]) << shift;
    mask |= uint64_t(0xFF) << shift;
    if (pick[2] == need || shift == 0) break;
  }
  // Keys above the prefix are all in; of the keys at it, `need` (past that only identical padding).
  for (int k = 0; k < SEL_PER; ++k) {
    const uint64_t m = key[k] & mask;
    int j = -1;
    if (m > prefix) {
      j = int(atomicAdd(&cnt[0], 1u));
    } else if (m == prefix) {
      if (const unsigned q = atomicAdd(&cnt[1], 1u); q < need) j = SELECT_K - int(need - q);
    }
    if (j >= 0) sel[j] = s[threadIdx.x + k * SEL_THREADS], selk[j] = key[k];
  }
  __syncthreads();
  if (gridDim.x == 1) {
    for (int i = SELECT_K + int(threadIdx.x); i < TOPK; i += SEL_THREADS) {
      a.vals[r * a.vals_ld + i] = -INFINITY;
      a.ids[r * a.ids_ld + i] = -1;
    }
  }
  if (threadIdx.x >= SELECT_K) return;
  const uint64_t mine = selk[threadIdx.x];
  int rank = 0;
  for (int j = 0; j < SELECT_K; ++j) rank += selk[j] > mine || (selk[j] == mine && j < int(threadIdx.x));
  if (gridDim.x > 1) {
    a.out[chunk * SELECT_K + rank] = sel[threadIdx.x];
    return;
  }
  const int64_t k = min(int64_t(p.k[r] ? p.k[r] : SELECT_K), a.vocab);
  a.vals[r * a.vals_ld + rank] = rank < k ? sel[threadIdx.x].v : -INFINITY;
  a.ids[r * a.ids_ld + rank] = rank < k ? sel[threadIdx.x].id : -1;
  if (threadIdx.x) return;
  const int64_t chunks1 = (a.vocab + SEL_N - 1) / SEL_N;
  float mx = -INFINITY, sum = 0;
  for (int64_t c = 0; c < chunks1; ++c) mx = fmaxf(mx, a.part[r * chunks1 + c].x);
  for (int64_t c = 0; c < chunks1; ++c) sum += a.part[r * chunks1 + c].y * expf(a.part[r * chunks1 + c].x - mx);
  a.lse[r * a.lse_ld] = mx + logf(sum);
}

larch_status sample_prep(larch_stream* vs, const larch_tensor* logits, const larch_sample_params* per_row,
                          const larch_tensor* penalty_ids_i32, const larch_tensor* out_topk_vals_f32,
                          const larch_tensor* out_topk_ids_i32, const larch_tensor* out_lse_f32) {
  return run("sample_prep", vs, [&](const stream& s) {
    const cpu::sample_checked c = cpu::check_sample_prep(sizer(s), logits, per_row, penalty_ids_i32, out_topk_vals_f32,
                                                     out_topk_ids_i32, out_lse_f32);
    const view lg = c.logits, vals = c.vals, ids = c.ids, lse = c.lse, pen = c.penalty_ids;
    const int64_t rows = lg.rows, vocab = lg.cols;
    if (vocab > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "a vocabulary past i32 ids"};
    aligned({lg, vals, ids, lse, pen});
    // per_row goes by value into the kernels, so a captured sample_prep would replay this call's
    // sampling settings for every later request: the engine runs it outside its graphs (9.5).
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    ck(cudaStreamIsCapturing(s.s, &cap), "cudaStreamIsCapturing");
    if (cap != cudaStreamCaptureStatusNone)
      throw failure{LARCH_ERR_UNSUPPORTED, "per_row is host memory: a graph would freeze it"};
    if (!rows) return;

    // Rows go TOPK at a time (row_params' capacity), each group through the passes in stream order,
    // reusing one scratch: A holds the first pass, B the second, A the third if any.
    // Greedy callers request only the candidates needed for bans and returned logprobs. The
    // observer and stochastic path still request all 64. Output shape and full-vocabulary LSE stay
    // identical to the CPU contract.
    int selected = 1;
    for (int64_t r = 0; r < rows; ++r) {
      const int k = int(per_row[r].top_k);
      selected = std::max(selected, k ? k : TOPK);
    }
    // 4 or 8 candidates measured within 0.02 % of a decode step of 64 (sm_120, 15 Oct 2026).
    selected = selected == 1 ? 1 : TOPK;
    const int64_t group = std::min<int64_t>(rows, TOPK), chunks1 = (vocab + SEL_N - 1) / SEL_N,
                  chunks2 = (chunks1 * selected + SEL_N - 1) / SEL_N;
    const uint64_t a_bytes = uint64_t(group * chunks1 * selected) * sizeof(cand),
                   b_bytes = uint64_t(group * chunks2 * selected) * sizeof(cand);
    const scratch mem(a_bytes + b_bytes + uint64_t(group * chunks1) * sizeof(float2), s);
    cand* const buf_a = static_cast<cand*>(mem.p);
    cand* const buf_b = reinterpret_cast<cand*>(static_cast<uint8_t*>(mem.p) + a_bytes);
    float2* const part = reinterpret_cast<float2*>(static_cast<uint8_t*>(mem.p) + a_bytes + b_bytes);
    for (int64_t g0 = 0; g0 < rows; g0 += TOPK) {
      const int64_t gn = std::min<int64_t>(TOPK, rows - g0);
      row_params p = {};
      for (int64_t r = 0; r < gn; ++r) {
        p.penalty[r] = per_row[g0 + r].repeat_penalty;
        p.window[r] = penalty_ids_i32 ? per_row[g0 + r].penalty_window : 0;
        p.k[r] = per_row[g0 + r].top_k;
      }
      sel_args a = {};
      a.logits = lg.as<float>() + g0 * lg.ld, a.l_ld = lg.ld, a.n = vocab, a.out = buf_a;
      a.pen = pen.p ? pen.as<int32_t>() + g0 * pen.ld : nullptr, a.pen_ld = pen.ld;
      a.part = part, a.vocab = vocab;
      a.vals = vals.as<float>() + g0 * vals.ld, a.vals_ld = vals.ld;
      a.ids = ids.as<int32_t>() + g0 * ids.ld, a.ids_ld = ids.ld;
      a.lse = lse.as<float>() + g0 * lse.ld, a.lse_ld = lse.ld;
      const auto select = [&](dim3 grid) {
        if (selected == 1) launch(select_k<1>, grid, unsigned(SEL_THREADS), s.s, a, p);
        else launch(select_k<TOPK>, grid, unsigned(SEL_THREADS), s.s, a, p);
      };
      select(dim3(unsigned(chunks1), unsigned(gn)));
      cand *cur = buf_a, *next = buf_b;
      for (int64_t chunks = chunks1; chunks > 1; std::swap(cur, next)) {
        a.in = cur, a.n = chunks * selected, a.out = next;
        chunks = (a.n + SEL_N - 1) / SEL_N;
        select(dim3(unsigned(chunks), unsigned(gn)));
      }
    }
  });
}

}  // namespace cuda
