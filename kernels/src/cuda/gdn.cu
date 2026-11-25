// CUDA Gated DeltaNet (Qwen 3.5's linear attention) and its output gate: larch.h's gdn_conv, gdn and sigmoid_mul,
// checked by the shared checks, f32 math in the CPU reference's order where it is cheap to keep. Every row count
// takes the same per-row code, so a check span of k rows equals k one-row steps bit for bit (draft rollback).
#include <algorithm>

#include "common/warp.cuh"
#include "cuda/cuda.cuh"

namespace cuda {

// out = x * sigmoid(gate), gate per element or (per_row) one per row. Each thread reads the x it writes, so out may
// be x.
__global__ void sigmoid_mul_k(const __half* x, int64_t x_ld, const __half* g, int64_t g_ld, bool per_row, __half* out,
                              int64_t o_ld, int64_t cols) {
  const int64_t r = blockIdx.x;
  for (int64_t c = blockIdx.y * int64_t(blockDim.x) + threadIdx.x; c < cols; c += int64_t(gridDim.y) * blockDim.x) {
    const float z = __half2float(g[r * g_ld + (per_row ? 0 : c)]);
    store(out + r * o_ld + c, __half2float(x[r * x_ld + c]) * (1.0f / (1.0f + expf(-z))));
  }
}

larch_status sigmoid_mul(larch_stream* vs, const larch_tensor* x, const larch_tensor* gate,
                          const larch_tensor* out) {
  return run("sigmoid_mul", vs, [&](const stream& s) {
    const cpu::elementwise_checked c = cpu::check_sigmoid_mul(sizer(s), x, gate, out);
    const view xv = c.a, g = c.b, o = c.out;
    aligned({xv, g, o});
    if (!xv.rows) return;
    launch(sigmoid_mul_k, rows_grid(xv.rows, xv.cols, 256), 256, s.s, xv.as<__half>(), xv.ld, g.as<__half>(), g.ld,
           g.cols == 1, o.as<__half>(), o.ld, xv.cols);
  });
}

// The device side of one gdn_conv or gdn call: the shared checks' views as pointers and leading dimensions. A
// vector's element i is at p[i * ld]; st is the layer's [n_slots, one slot's elements].
struct gdn_args {
  const __half* in;
  int64_t in_ld;
  const float *w, *alpha, *beta, *dt, *a;
  int64_t w_ld, alpha_ld, beta_ld, dt_ld, a_ld;
  void* st;
  int64_t st_ld;
  const int32_t *slots, *lens, *snaps;  // snaps null: none
  int64_t slots_ld, lens_ld, snaps_ld;
  __half* out;
  int64_t out_ld, rows;
  int channels, taps, k_heads, hd;
  float eps, scale;
};

static gdn_args gdn_arguments(const cpu::gdn_checked& c, const larch_gdn_state* state, float eps) {
  const view in = c.in, w = c.w, al = c.alpha, be = c.beta, dt = c.dt_bias, av = c.a, st = c.state, slots = c.slots,
             lens = c.lens, snaps = c.snaps, o = c.out;
  aligned({in, w, al, be, dt, av, st, slots, lens, snaps, o});
  if (in.cols > INT32_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 - 1 channels"};
  return {in.as<__half>(), in.ld, w.as<float>(), al.as<float>(), be.as<float>(), dt.as<float>(), av.as<float>(),
          w.ld, al.ld, be.ld, dt.ld, av.ld, st.p, st.ld, slots.as<int32_t>(), lens.as<int32_t>(),
          snaps.as<int32_t>(), slots.ld, lens.ld, snaps.ld, o.as<__half>(), o.ld, in.rows, int(in.cols),
          int(state->kernel), int(state->k_heads), int(state->head_dim), eps, 1.0f / sqrtf(float(state->head_dim))};
}

// Sequence s's first row (the q_lens before it) and its length. larch.h makes them >= 0 and sum to the rows; the
// device cannot refuse a call that breaks that, so a negative length counts as 0 (as on Metal) and none reaches past
// the call's rows, where it would fault the device.
// ponytail: every thread sums the q_lens before its sequence; a scan when decode batches pass 64 sequences.
__device__ __forceinline__ void gdn_rows(const gdn_args& a, int s, int64_t& r0, int64_t& len) {
  r0 = 0;
  for (int i = 0; i < s; ++i) r0 += max(a.lens[i * a.lens_ld], 0);
  r0 = min(r0, a.rows);
  len = min(int64_t(max(a.lens[s * a.lens_ld], 0)), a.rows - r0);
}

constexpr int CONV_CH = 32, CONV_ROWS = 8, CONV_MAX_TAPS = 8;

// gdn_conv: block (channel chunk, sequence s) of CONV_CH x blockDim.y threads (CONV_ROWS, or 1 at decode). The conv
// has no recurrence, so its rows go in parallel: input t of a channel is the slot's conv row t (t < taps - 1), else
// the sequence's row t - (taps - 1). Snapshots copy the inputs as of their row; the live slot, read by this block's
// channels only, is overwritten last. Four blocks an SM: without a floor sm_120's ptxas spills the taps.
__global__ void __launch_bounds__(CONV_CH* CONV_ROWS, 4) gdn_conv_k(gdn_args a) {
  const int c = int(blockIdx.x) * CONV_CH + int(threadIdx.x), s = int(blockIdx.y), hist = a.taps - 1;
  int64_t r0, len;
  gdn_rows(a, s, r0, len);
  const bool on = c < a.channels;
  __half* live = static_cast<__half*>(a.st) + a.slots[s * a.slots_ld] * a.st_ld;
  const auto input = [&](int64_t t) {
    return t < hist ? live[t * a.channels + c] : a.in[(r0 + t - hist) * a.in_ld + c];
  };
  float w[CONV_MAX_TAPS];  // registers: every loop over taps is unrolled to the bound
#pragma unroll
  for (int i = 0; i < CONV_MAX_TAPS; ++i) w[i] = on && i < a.taps ? a.w[c * a.w_ld + i] : 0.0f;
  for (int64_t t = threadIdx.y; on && t < len; t += blockDim.y) {
    float sum = 0;
#pragma unroll
    for (int i = 0; i < CONV_MAX_TAPS; ++i)
      if (i < a.taps) sum = fmaf(w[i], __half2float(input(t + i)), sum);
    store(a.out + (r0 + t) * a.out_ld + c, sum / (1.0f + expf(-sum)));
    const int32_t snap = a.snaps ? a.snaps[(r0 + t) * a.snaps_ld] : -1;
    if (snap >= 0) {
      __half* d = static_cast<__half*>(a.st) + snap * a.st_ld;
      for (int i = 0; i < hist; ++i) d[i * a.channels + c] = input(t + 1 + i);
    }
  }
  // Every thread of the channel reads the old live rows above; only then does one of them shift in the new ones.
  __syncthreads();
  if (!on || threadIdx.y) return;
  // Ascending, so row len + i >= i is read before it is overwritten.
  for (int i = 0; i < hist; ++i) live[i * a.channels + c] = input(len + i);
}

larch_status gdn_conv(larch_stream* vs, const larch_tensor* x, const larch_tensor* w, const larch_gdn_state* state,
                       const larch_tensor* slots_i32, const larch_tensor* q_lens_i32,
                       const larch_tensor* snap_slots_or_null, const larch_tensor* out) {
  return run("gdn_conv", vs, [&](const stream& s) {
    const cpu::gdn_checked c =
        cpu::check_gdn_conv(sizer(s), x, w, state, slots_i32, q_lens_i32, snap_slots_or_null, out);
    const gdn_args a = gdn_arguments(c, state, 0.0f);
    if (state->kernel > CONV_MAX_TAPS) throw failure{LARCH_ERR_UNSUPPORTED, "a conv kernel wider than 8 taps"};
    if (c.slots.rows > 65535) throw failure{LARCH_ERR_UNSUPPORTED, "more than 65535 sequences"};
    if (!c.in.rows) return;
    launch(gdn_conv_k, dim3(unsigned((a.channels + CONV_CH - 1) / CONV_CH), unsigned(c.slots.rows)),
           dim3(CONV_CH, c.in.rows > c.slots.rows ? CONV_ROWS : 1), s.s, a);
  });
}

// gdn's launch shape: GDN_WARPS warps a block, GDN_R rows of a head's matrix M a warp, GDN_T rows of the input
// staged in shared memory at a time; head_dim up to GDN_MAX_HD. Two blocks an SM cap it at 128 registers, which
// also keeps head_dim 32 from spilling on sm_120.
constexpr int GDN_WARPS = 8, GDN_R = 4, GDN_T = 16, GDN_MAX_HD = 256, GDN_LANES = 32;

// gdn: block (row block, v head h, sequence s) holds rows jc .. jc + 32 of h's M in registers for the whole
// sequence, GDN_R rows a warp, lane l holding entries l + 32 e of each (E = head_dim / 32 rounded up to a power of
// two). Per GDN_T rows of the sequence, the block first stages what does not depend on M: q and k (head h % k_heads)
// l2-normalised, the decay g, b and its v entries; then each warp walks the rows in order: M = e^g M, d = b (v - M k),
// M += d k^T, out = M q / sqrt(head_dim). A row's math, and so its bits, do not depend on how many rows the call
// holds. The rows of M never meet, so blocks and warps share nothing but the staged inputs.
// ponytail: the row walk serves prefill too (M's update is serial per row); the WY chunk form (Metal's
// vv_gdn_chunk) when the 35B's TTFT on CUDA shows gdn.
template <int E>
__global__ void __launch_bounds__(GDN_WARPS* GDN_LANES, 2) gdn_k(gdn_args a) {
  constexpr int ROWS = GDN_WARPS * GDN_R, COLS = E * GDN_LANES;
  __shared__ float qs[GDN_T][COLS], ks[GDN_T][COLS], vv[GDN_T][ROWS], gs[GDN_T], bs[GDN_T];
  __shared__ int32_t sn[GDN_T];
  const int lane = int(threadIdx.x) % GDN_LANES, warp = int(threadIdx.x) / GDN_LANES, h = int(blockIdx.y),
            s = int(blockIdx.z), hd = a.hd, kq = h % a.k_heads, jc = int(blockIdx.x) * ROWS, j0 = jc + warp * GDN_R;
  int64_t r0, len;
  gdn_rows(a, s, r0, len);
  const int64_t head = int64_t(h) * hd * hd;  // floats from a slot's start to h's matrix
  float* live = static_cast<float*>(a.st) + a.slots[s * a.slots_ld] * a.st_ld + head;
  const auto col = [&](int e) { return lane + GDN_LANES * e; };
  float m[GDN_R][E];
#pragma unroll
  for (int j = 0; j < GDN_R; ++j)
#pragma unroll
    for (int e = 0; e < E; ++e) m[j][e] = j0 + j < hd && col(e) < hd ? live[(j0 + j) * hd + col(e)] : 0.0f;
  const auto save = [&](float* p) {  // this thread's M entries into a slot's matrix at p
#pragma unroll
    for (int j = 0; j < GDN_R; ++j)
#pragma unroll
      for (int e = 0; e < E; ++e)
        if (j0 + j < hd && col(e) < hd) p[(j0 + j) * hd + col(e)] = m[j][e];
  };
  const float dt = a.dt[h * a.dt_ld], av = a.a[h * a.a_ld];
  for (int64_t t0 = 0; t0 < len; t0 += GDN_T) {
    const int n = int(min(int64_t(GDN_T), len - t0));
    __syncthreads();  // the last chunk's reads of the staging are done
    for (int t = warp; t < n; t += GDN_WARPS) {
      const int64_t r = r0 + t0 + t;
      const __half* row = a.in + r * a.in_ld;
      float q[E], k[E], qq = 0, kk = 0;
#pragma unroll
      for (int e = 0; e < E; ++e) {
        q[e] = col(e) < hd ? __half2float(row[kq * hd + col(e)]) : 0.0f;
        k[e] = col(e) < hd ? __half2float(row[(a.k_heads + kq) * hd + col(e)]) : 0.0f;
        qq = fmaf(q[e], q[e], qq);
        kk = fmaf(k[e], k[e], kk);
      }
      // ponytail: each row's q/k norms recur for every v head and row block sharing the k head (128 times for 16 pairs
      // at the 35B); hoist them into a pre-pass when decode shows the norms in the profile.
      const float qi = 1.0f / sqrtf(vv_warp_reduce_sum(qq) + a.eps), ki = 1.0f / sqrtf(vv_warp_reduce_sum(kk) + a.eps);
#pragma unroll
      for (int e = 0; e < E; ++e) qs[t][col(e)] = q[e] * qi, ks[t][col(e)] = k[e] * ki;
      if (lane == 0) {
        const float x = a.alpha[r * a.alpha_ld + h] + dt;
        gs[t] = expf((x > 20.0f ? x : logf(1.0f + expf(x))) * av);
        bs[t] = 1.0f / (1.0f + expf(-a.beta[r * a.beta_ld + h]));
        sn[t] = a.snaps ? a.snaps[r * a.snaps_ld] : -1;
      }
    }
    for (int i = int(threadIdx.x); i < n * ROWS; i += int(blockDim.x)) {
      const int t = i / ROWS, j = i % ROWS;
      vv[t][j] = jc + j < hd ? __half2float(a.in[(r0 + t0 + t) * a.in_ld + (2 * a.k_heads + h) * hd + jc + j]) : 0.0f;
    }
    __syncthreads();
    for (int t = 0; t < n; ++t) {
      const float g = gs[t], b = bs[t];
      float q[E], k[E], mk[GDN_R], mq[GDN_R];
#pragma unroll
      for (int e = 0; e < E; ++e) q[e] = qs[t][col(e)], k[e] = ks[t][col(e)];
#pragma unroll
      for (int j = 0; j < GDN_R; ++j) {
        mk[j] = 0;
#pragma unroll
        for (int e = 0; e < E; ++e) {
          m[j][e] *= g;
          mk[j] = fmaf(m[j][e], k[e], mk[j]);
        }
      }
#pragma unroll
      for (int j = 0; j < GDN_R; ++j) {
        const float d = (vv[t][warp * GDN_R + j] - vv_warp_reduce_sum(mk[j])) * b;
        mq[j] = 0;
#pragma unroll
        for (int e = 0; e < E; ++e) {
          m[j][e] = fmaf(d, k[e], m[j][e]);
          mq[j] = fmaf(m[j][e], q[e], mq[j]);
        }
      }
      float o = 0;
#pragma unroll
      for (int j = 0; j < GDN_R; ++j) {
        const float v = vv_warp_reduce_sum(mq[j]) * a.scale;
        o = lane == j ? v : o;
      }
      const int64_t r = r0 + t0 + t;
      if (lane < GDN_R && j0 + lane < hd) store(a.out + r * a.out_ld + h * hd + j0 + lane, o);
      if (sn[t] >= 0) save(static_cast<float*>(a.st) + sn[t] * a.st_ld + head);
    }
  }
  save(live);
}

// larch_fused_gdn's walk: gdn_k's math with a whole head a block (GDN_WARPS * 2 warps, 2 E rows of M a warp; a row's
// math is gdn_k's whatever the rows a warp holds), so after each staged chunk the block holds every o of its head
// (rounded to f16 as gdn stores it) for qk_norm's bytes (warp_rms, one warp a row) and silu_mul_k's product with z.
constexpr int NORM_WARPS = 2 * GDN_WARPS;

template <int E>
__global__ void __launch_bounds__(NORM_WARPS* GDN_LANES) gdn_norm_k(gdn_args a, const float* nw, int64_t nw_ld,
                                                                     int norm_vec, int norm_threads, const __half* z,
                                                                     int64_t z_ld) {
  constexpr int R = 2 * E, COLS = E * GDN_LANES;
  __shared__ float qs[GDN_T][COLS], ks[GDN_T][COLS], vv[GDN_T][COLS], os[GDN_T][COLS], gs[GDN_T], bs[GDN_T];
  __shared__ int32_t sn[GDN_T];
  const int lane = int(threadIdx.x) % GDN_LANES, warp = int(threadIdx.x) / GDN_LANES, h = int(blockIdx.y),
            s = int(blockIdx.z), hd = a.hd, kq = h % a.k_heads, j0 = warp * R;
  int64_t r0, len;
  gdn_rows(a, s, r0, len);
  const int64_t head = int64_t(h) * hd * hd;
  float* live = static_cast<float*>(a.st) + a.slots[s * a.slots_ld] * a.st_ld + head;
  const auto col = [&](int e) { return lane + GDN_LANES * e; };
  float m[R][E];
#pragma unroll
  for (int j = 0; j < R; ++j)
#pragma unroll
    for (int e = 0; e < E; ++e) m[j][e] = j0 + j < hd && col(e) < hd ? live[(j0 + j) * hd + col(e)] : 0.0f;
  const auto save = [&](float* p) {
#pragma unroll
    for (int j = 0; j < R; ++j)
#pragma unroll
      for (int e = 0; e < E; ++e)
        if (j0 + j < hd && col(e) < hd) p[(j0 + j) * hd + col(e)] = m[j][e];
  };
  const float dt = a.dt[h * a.dt_ld], av = a.a[h * a.a_ld];
  for (int64_t t0 = 0; t0 < len; t0 += GDN_T) {
    const int n = int(min(int64_t(GDN_T), len - t0));
    __syncthreads();  // the last chunk's reads of the staging are done
    for (int t = warp; t < n; t += NORM_WARPS) {
      const int64_t r = r0 + t0 + t;
      const __half* row = a.in + r * a.in_ld;
      float q[E], k[E], qq = 0, kk = 0;
#pragma unroll
      for (int e = 0; e < E; ++e) {
        q[e] = col(e) < hd ? __half2float(row[kq * hd + col(e)]) : 0.0f;
        k[e] = col(e) < hd ? __half2float(row[(a.k_heads + kq) * hd + col(e)]) : 0.0f;
        qq = fmaf(q[e], q[e], qq);
        kk = fmaf(k[e], k[e], kk);
      }
      const float qi = 1.0f / sqrtf(vv_warp_reduce_sum(qq) + a.eps), ki = 1.0f / sqrtf(vv_warp_reduce_sum(kk) + a.eps);
#pragma unroll
      for (int e = 0; e < E; ++e) qs[t][col(e)] = q[e] * qi, ks[t][col(e)] = k[e] * ki;
      if (lane == 0) {
        const float x = a.alpha[r * a.alpha_ld + h] + dt;
        gs[t] = expf((x > 20.0f ? x : logf(1.0f + expf(x))) * av);
        bs[t] = 1.0f / (1.0f + expf(-a.beta[r * a.beta_ld + h]));
        sn[t] = a.snaps ? a.snaps[r * a.snaps_ld] : -1;
      }
    }
    for (int i = int(threadIdx.x); i < n * COLS; i += int(blockDim.x)) {
      const int t = i / COLS, j = i % COLS;
      vv[t][j] = j < hd ? __half2float(a.in[(r0 + t0 + t) * a.in_ld + (2 * a.k_heads + h) * hd + j]) : 0.0f;
    }
    __syncthreads();
    for (int t = 0; t < n; ++t) {
      const float g = gs[t], b = bs[t];
      float q[E], k[E], mk[R], mq[R];
#pragma unroll
      for (int e = 0; e < E; ++e) q[e] = qs[t][col(e)], k[e] = ks[t][col(e)];
#pragma unroll
      for (int j = 0; j < R; ++j) {
        mk[j] = 0;
#pragma unroll
        for (int e = 0; e < E; ++e) {
          m[j][e] *= g;
          mk[j] = fmaf(m[j][e], k[e], mk[j]);
        }
      }
#pragma unroll
      for (int j = 0; j < R; ++j) {
        const float d = (vv[t][j0 + j] - vv_warp_reduce_sum(mk[j])) * b;
        mq[j] = 0;
#pragma unroll
        for (int e = 0; e < E; ++e) {
          m[j][e] = fmaf(d, k[e], m[j][e]);
          mq[j] = fmaf(m[j][e], q[e], mq[j]);
        }
      }
      float o = 0;
#pragma unroll
      for (int j = 0; j < R; ++j) {
        const float v = vv_warp_reduce_sum(mq[j]) * a.scale;
        o = lane == j ? v : o;
      }
      if (lane < R && j0 + lane < hd) os[t][j0 + lane] = __half2float(__float2half_rn(o));
      if (sn[t] >= 0) save(static_cast<float*>(a.st) + sn[t] * a.st_ld + head);
    }
    __syncthreads();
    for (int t = warp; t < n; t += NORM_WARPS) {
      const int64_t r = r0 + t0 + t;
      warp_rms(os[t], nw, nw_ld, a.eps, hd, norm_vec, norm_threads, lane);
      for (int i = lane; i < hd; i += GDN_LANES) {
        const float x = __half2float(z[r * z_ld + h * hd + i]);
        store(a.out + r * a.out_ld + h * hd + i, x / (1.0f + expf(-x)) * os[t][i]);
      }
    }
  }
  save(live);
}

larch_status fused_gdn(larch_stream* vs, const larch_tensor* qkv, const larch_tensor* conv_w,
                        const larch_tensor* alpha, const larch_tensor* beta, const larch_tensor* dt_bias,
                        const larch_tensor* a_log, float eps, const larch_gdn_state* state,
                        const larch_tensor* slots_i32, const larch_tensor* q_lens_i32,
                        const larch_tensor* snap_slots_or_null, const larch_tensor* norm_w, const larch_tensor* z,
                        const larch_tensor* conv, const larch_tensor* out) {
  return run("fused_gdn", vs, [&](const stream& s) {
    const cpu::fused_gdn_checked c =
        cpu::check_fused_gdn(sizer(s), qkv, conv_w, alpha, beta, dt_bias, a_log, eps, state, slots_i32, q_lens_i32,
                             snap_slots_or_null, norm_w, z, conv, out);
    const gdn_args ca = gdn_arguments(c.conv, state, 0.0f), ga = gdn_arguments(c.gdn, state, eps);
    const view nw = c.norm_w, zv = c.z.a;
    aligned({nw, zv});
    if (state->kernel > CONV_MAX_TAPS) throw failure{LARCH_ERR_UNSUPPORTED, "a conv kernel wider than 8 taps"};
    if (ga.hd > 4 * GDN_LANES) throw failure{LARCH_ERR_UNSUPPORTED, "fused_gdn: head_dim over 128"};
    if (c.conv.slots.rows > 65535 || state->v_heads > 65535)
      throw failure{LARCH_ERR_UNSUPPORTED, "more than 65535 sequences or v heads"};
    if (!c.conv.in.rows) return;
    launch(gdn_conv_k, dim3(unsigned((ca.channels + CONV_CH - 1) / CONV_CH), unsigned(c.conv.slots.rows)),
           dim3(CONV_CH, c.conv.in.rows > c.conv.slots.rows ? CONV_ROWS : 1), s.s, ca);
    // qk_norm's launch shape for o, as on rows of v_heads head_dim halves 16-byte aligned (larch.h)
    const int hd = ga.hd, vec = hd % 4 == 0 ? 4 : 1;
    const int threads = std::min(vec == 4 ? 1024 : 256, ((hd + vec - 1) / vec + s.warp - 1) / s.warp * s.warp);
    const dim3 grid(1, state->v_heads, unsigned(c.conv.slots.rows));
    const unsigned block = NORM_WARPS * GDN_LANES;
    if (hd <= 32) launch(gdn_norm_k<1>, grid, block, s.s, ga, nw.as<float>(), nw.ld, vec, threads, zv.as<__half>(), zv.ld);
    else if (hd <= 64) launch(gdn_norm_k<2>, grid, block, s.s, ga, nw.as<float>(), nw.ld, vec, threads, zv.as<__half>(), zv.ld);
    else launch(gdn_norm_k<4>, grid, block, s.s, ga, nw.as<float>(), nw.ld, vec, threads, zv.as<__half>(), zv.ld);
  });
}

larch_status gdn(larch_stream* vs, const larch_tensor* qkv, const larch_tensor* alpha, const larch_tensor* beta,
                  const larch_tensor* dt_bias, const larch_tensor* a_log, float eps, const larch_gdn_state* state,
                  const larch_tensor* slots_i32, const larch_tensor* q_lens_i32,
                  const larch_tensor* snap_slots_or_null, const larch_tensor* out) {
  return run("gdn", vs, [&](const stream& s) {
    const cpu::gdn_checked c = cpu::check_gdn(sizer(s), qkv, alpha, beta, dt_bias, a_log, eps, state, slots_i32,
                                              q_lens_i32, snap_slots_or_null, out);
    const gdn_args a = gdn_arguments(c, state, eps);
    if (a.hd > GDN_MAX_HD) throw failure{LARCH_ERR_UNSUPPORTED, "head_dim over 256"};
    if (c.slots.rows > 65535 || state->v_heads > 65535)
      throw failure{LARCH_ERR_UNSUPPORTED, "more than 65535 sequences or v heads"};
    if (!c.in.rows) return;
    const dim3 grid(unsigned((a.hd + GDN_WARPS * GDN_R - 1) / (GDN_WARPS * GDN_R)), state->v_heads,
                    unsigned(c.slots.rows));
    const unsigned threads = GDN_WARPS * GDN_LANES;
    if (a.hd <= 32) launch(gdn_k<1>, grid, threads, s.s, a);
    else if (a.hd <= 64) launch(gdn_k<2>, grid, threads, s.s, a);
    else if (a.hd <= 128) launch(gdn_k<4>, grid, threads, s.s, a);
    else launch(gdn_k<8>, grid, threads, s.s, a);
  });
}

}  // namespace cuda
