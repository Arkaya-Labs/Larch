// CPU reference ops. The contract they implement (dtypes, shapes, the KV pool layout, padding, gemm and
// sample_prep semantics) is larch.h's, from architecture 6.1; each op below rejects everything else
// with INVALID_ARG before it writes. Values in device memory (ids, slots, block tables, lengths) are
// checked too, as a debugging aid only: on a GPU they are undefined behaviour.
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <vector>

#include "cpu/cpu.h"

namespace cpu {

static bool same_shape(const meta& a, const meta& b) {
  return a.ndim == b.ndim && std::equal(a.shape, a.shape + a.ndim, b.shape);
}

meta layer_view(const larch_buffer* b, uint64_t size, larch_dtype t, int64_t rows, int64_t cols, uint32_t layer,
                uint32_t n_layers, const char* too_small) {
  const type_info ti = info(t);
  meta v;
  v.buf = b;
  v.dtype = t;
  v.ndim = 2;
  v.rows = v.shape[0] = rows;
  v.cols = v.shape[1] = cols;
  v.ld = cols / ti.block;
  v.row_bytes = mul(v.ld, ti.bytes);
  const int64_t bytes = mul(rows, v.row_bytes);
  need(uint64_t(mul(n_layers, bytes)) <= size, too_small);
  v.offset = layer * uint64_t(bytes);
  return v;
}

// One layer of the pool buffer b as [slots, n_kv_heads * head_dim] of the pool's dtype.
static meta pool_layer(const sizer& bytes, const larch_kv_pool* p, const larch_buffer* b, const char* what) {
  const uint64_t size = bytes(b, what);
  if (!b) throw bad_arg(std::string(what) + " is null");  // as check_tensor: never an absent pool buffer
  need(p->dtype == LARCH_F16 || p->dtype == LARCH_Q8_0 || p->dtype == LARCH_FP8_E4M3,
       "pool dtype must be f16, q8_0 or fp8_e4m3");
  need(p->block_tokens && p->n_kv_heads && p->head_dim && p->layer < p->n_layers, "bad pool geometry");
  need(p->dtype != LARCH_Q8_0 || p->head_dim % 32 == 0, "a q8_0 pool's head_dim must be a multiple of 32");
  return layer_view(b, size, p->dtype, mul(p->n_blocks, p->block_tokens), mul(p->n_kv_heads, p->head_dim), p->layer,
                    p->n_layers, "pool buffer too small");
}

embed_checked check_embed(const sizer& bytes, const larch_tensor* table, const larch_tensor* ids_i32,
                          const larch_tensor* out_f32) {
  const embed_checked a{check_tensor(bytes, table, "table"), check_tensor(bytes, ids_i32, "ids"),
                        check_tensor(bytes, out_f32, "out")};
  need(a.table.ndim == 2 && a.table.dtype != LARCH_BF16 && a.table.dtype != LARCH_I32,
       "table must be [vocab, d], quant, f16 or f32");
  need(is(a.ids, LARCH_I32, 1), "ids must be i32 [rows]");
  need(is(a.out, LARCH_F32, 2) && a.out.rows == a.ids.rows && a.out.cols == a.table.cols, "out must be f32 [rows, d]");
  return a;
}

// out[r] = table[ids[r]], dequantized.
larch_status embed(larch_stream*, const larch_tensor* table, const larch_tensor* ids_i32,
                    const larch_tensor* out_f32) {
  return run("embed", [&] {
    const embed_checked a = check_embed(buffer_bytes, table, ids_i32, out_f32);
    const view t = a.table, ids = a.ids, o = a.out;
    std::vector<float> row(static_cast<size_t>(t.cols));
    for (int64_t r = 0; r < ids.rows; ++r) {
      const int32_t id = ids.i32(r);
      need(id >= 0 && id < t.rows, "token id out of range");
      t.row_f32(id, row.data());
      for (int64_t c = 0; c < t.cols; ++c) o.set(r, c, row[size_t(c)]);
    }
  });
}

// o[r, c0 + i] = x[r, c0 + i] / sqrt(mean of x[r, c0 .. c0 + n)^2 + eps) * w[i], for i < n.
static void rms(const view& x, const view& w, float eps, const view& o, int64_t r, int64_t c0, int64_t n) {
  double ss = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double v = x.get(r, c0 + i);
    ss += v * v;
  }
  const float scale = float(1.0 / std::sqrt(ss / double(n) + eps));
  for (int64_t i = 0; i < n; ++i) o.set(r, c0 + i, x.get(r, c0 + i) * scale * w.get(i, 0));
}

norm_checked check_rmsnorm(const sizer& bytes, const larch_tensor* x, const larch_tensor* w,
                           const larch_tensor* out, larch_dtype out_dtype) {
  const norm_checked a{check_tensor(bytes, x, "x"), check_tensor(bytes, w, "w"), check_tensor(bytes, out, "out")};
  need(is(a.x, LARCH_F32, 2) && is(a.w, LARCH_F32, 1) && a.w.rows == a.x.cols,
       "x must be f32 [rows, d] and w f32 [d]");
  need(is(a.out, out_dtype, 2) && same_shape(a.x, a.out),
       out_dtype == LARCH_F16 ? "out must be f16 [rows, d]" : "out must be f32 [rows, d]");
  return a;
}

static larch_status rmsnorm_as(const char* op, larch_dtype t, const larch_tensor* x, const larch_tensor* w,
                                float eps, const larch_tensor* out) {
  return run(op, [&] {
    const norm_checked a = check_rmsnorm(buffer_bytes, x, w, out, t);
    const view xv = a.x, wv = a.w, o = a.out;
    for (int64_t r = 0; r < xv.rows; ++r) rms(xv, wv, eps, o, r, 0, xv.cols);
  });
}

larch_status rmsnorm(larch_stream*, const larch_tensor* x, const larch_tensor* w, float eps,
                      const larch_tensor* out) {
  return rmsnorm_as("rmsnorm", LARCH_F16, x, w, eps, out);
}

larch_status rmsnorm_f32(larch_stream*, const larch_tensor* x, const larch_tensor* w, float eps,
                          const larch_tensor* out) {
  return rmsnorm_as("rmsnorm_f32", LARCH_F32, x, w, eps, out);
}

layernorm_checked check_layernorm(const sizer& bytes, const larch_tensor* x, const larch_tensor* w,
                                  const larch_tensor* b_or_null, const larch_tensor* out) {
  layernorm_checked a{check_tensor(bytes, x, "x"), check_tensor(bytes, w, "w"), {}, check_tensor(bytes, out, "out")};
  need(is(a.x, LARCH_F32, 2) && is(a.w, LARCH_F32, 1) && a.w.rows == a.x.cols,
       "x must be f32 [rows, d] and w f32 [d]");
  if (b_or_null) {
    a.b = check_tensor(bytes, b_or_null, "b");
    need(is(a.b, LARCH_F32, 1) && a.b.rows == a.x.cols, "b must be f32 [d]");
  }
  need(is(a.out, LARCH_F16, 2) && same_shape(a.x, a.out), "out must be f16 [rows, d]");
  return a;
}

// ggml_norm then ggml_mul and ggml_add: the mean, then the mean square of x - mean, in double.
larch_status layernorm(larch_stream*, const larch_tensor* x, const larch_tensor* w, const larch_tensor* b_or_null,
                        float eps, const larch_tensor* out) {
  return run("layernorm", [&] {
    const layernorm_checked a = check_layernorm(buffer_bytes, x, w, b_or_null, out);
    const view xv = a.x, wv = a.w, bv = a.b, o = a.out;
    const int64_t n = xv.cols;
    for (int64_t r = 0; r < xv.rows; ++r) {
      double sum = 0;
      for (int64_t i = 0; i < n; ++i) sum += xv.get(r, i);
      const float mean = float(sum / double(n));
      double ss = 0;
      for (int64_t i = 0; i < n; ++i) {
        const double d = xv.get(r, i) - mean;
        ss += d * d;
      }
      const float scale = 1.0f / std::sqrt(float(ss / double(n)) + eps);
      for (int64_t i = 0; i < n; ++i)
        o.set(r, i, (xv.get(r, i) - mean) * scale * wv.get(i, 0) + (bv.buf ? bv.get(i, 0) : 0.0f));
    }
  });
}

rope_checked check_rope(const sizer& bytes, const larch_tensor* qk, const larch_tensor* pos_i32,
                        const larch_rope_params* p, larch_dtype qk_dtype) {
  rope_checked a{check_tensor(bytes, qk, "qk"), check_tensor(bytes, pos_i32, "pos"), {}};
  need(p != nullptr, "params are null");
  need(is(a.qk, qk_dtype, 3) && is(a.pos, LARCH_I32, 1) && a.pos.rows == a.qk.rows,
       qk_dtype == LARCH_F16 ? "qk must be f16 [rows, heads, head_dim] and pos i32 [rows]"
                              : "qk must be f32 [rows, heads, head_dim] and pos i32 [rows]");
  const int64_t nd = p->n_dims;
  need(nd > 0 && nd % 2 == 0 && nd <= a.qk.shape[2], "n_dims must be even and <= head_dim");
  if (p->freq_factors) {
    a.freq_factors = check_tensor(bytes, p->freq_factors, "freq_factors");
    need(is(a.freq_factors, LARCH_F32, 1) && a.freq_factors.rows == nd / 2, "freq_factors must be f32 [n_dims / 2]");
  }
  return a;
}

// ggml's rope_yarn for one pair: the YaRN ramp bounds of p (ggml_rope_yarn_corr_dims), then pair i's cos
// and sin at angle theta (before its freq_factor), f32 in ggml's order.
struct yarn {
  const larch_rope_params* p;
  view ff;
  float lo, hi;
  yarn(const larch_rope_params* rp, view f) : p(rp), ff(f) {
    const int64_t nd = p->n_dims;
    const auto corr_dim = [&](float n_rot) {
      return float(nd) * logf(float(p->n_ctx_orig) / (n_rot * 2 * std::numbers::pi_v<float>)) / (2 * logf(p->freq_base));
    };
    lo = std::max(0.0f, floorf(corr_dim(p->beta_fast)));
    hi = std::min(float(nd - 1), ceilf(corr_dim(p->beta_slow)));
  }
  void operator()(float theta, int64_t i, float* cs) const {
    const float extrap = theta / (p->freq_factors ? ff.get(i, 0) : 1.0f);
    const float interp = p->freq_scale * extrap;
    float ang = interp, mscale = p->attn_factor;
    if (p->ext_factor != 0.0f) {
      const float y = (float(i) - lo) / std::max(0.001f, hi - lo);
      const float mix = (1 - std::min(1.0f, std::max(0.0f, y))) * p->ext_factor;
      ang = interp * (1 - mix) + extrap * mix;
      mscale *= 1.0f + 0.1f * logf(1.0f / p->freq_scale);
    }
    cs[0] = cosf(ang) * mscale;
    cs[1] = sinf(ang) * mscale;
  }
};

// ggml_rope_ext in place, f32 math in ggml's order (rope_yarn, iterated theta).
static larch_status rope_as(const char* op, larch_dtype t, const larch_tensor* qk, const larch_tensor* pos_i32,
                             const larch_rope_params* p) {
  return run(op, [&] {
    const rope_checked args = check_rope(buffer_bytes, qk, pos_i32, p, t);
    const view q = args.qk, pos = args.pos;
    const int64_t heads = q.shape[1], hd = q.shape[2], nd = p->n_dims;
    const yarn angle(p, args.freq_factors);
    const float theta_scale = powf(p->freq_base, -2.0f / float(nd));
    std::vector<float> cs(static_cast<size_t>(nd));  // cos, sin per pair
    for (int64_t r = 0; r < q.rows; ++r) {
      float theta = float(pos.i32(r));
      for (int64_t i0 = 0; i0 < nd; i0 += 2) {
        angle(theta, i0 / 2, &cs[size_t(i0)]);
        theta *= theta_scale;
      }
      for (int64_t h = 0; h < heads; ++h) {
        const int64_t base = h * hd;
        for (int64_t i0 = 0; i0 < nd; i0 += 2) {
          // normal: adjacent pairs; neox: element i with element i + n_dims / 2
          const int64_t a = base + (p->neox ? i0 / 2 : i0), b = p->neox ? a + nd / 2 : a + 1;
          const float x0 = q.get(r, a), x1 = q.get(r, b), c = cs[size_t(i0)], s = cs[size_t(i0) + 1];
          q.set(r, a, x0 * c - x1 * s);
          q.set(r, b, x0 * s + x1 * c);
        }
      }
    }
  });
}

larch_status rope(larch_stream*, const larch_tensor* qk, const larch_tensor* pos_i32, const larch_rope_params* p) {
  return rope_as("rope", LARCH_F16, qk, pos_i32, p);
}

rope_checked check_rope_multi(const sizer& bytes, const larch_tensor* qk, const larch_tensor* pos_i32,
                              const larch_rope_multi_params* p) {
  need(p != nullptr, "params are null");
  rope_checked a{check_tensor(bytes, qk, "qk"), check_tensor(bytes, pos_i32, "pos"), {}};
  need(is(a.qk, LARCH_F16, 3) && is(a.pos, LARCH_I32, 2) && a.pos.rows == a.qk.rows && a.pos.cols == 4,
       "qk must be f16 [rows, heads, head_dim] and pos i32 [rows, 4]");
  const larch_rope_params* r = &p->rope;
  const int64_t nd = r->n_dims, hd = a.qk.shape[2];
  need(p->mode <= LARCH_ROPE_VISION, "mode must be a larch_rope_mode");
  need(p->mode == LARCH_ROPE_VISION ? nd > 0 && 2 * nd == hd : nd > 0 && nd % 2 == 0 && nd <= hd,
       "n_dims must be even and <= head_dim (VISION: head_dim / 2)");
  const uint64_t s0 = p->sections[0], s1 = p->sections[1], s2 = p->sections[2], s3 = p->sections[3];
  need(s0 + s1 + s2 > 0 && s0 + s1 + s2 + s3 <= uint64_t(hd), "sections must hold a pair and fit head_dim");
  if (r->freq_factors) {
    a.freq_factors = check_tensor(bytes, r->freq_factors, "freq_factors");
    const int64_t pairs = p->mode == LARCH_ROPE_VISION ? nd : nd / 2;
    need(is(a.freq_factors, LARCH_F32, 1) && a.freq_factors.rows == pairs, "freq_factors must be f32 [pairs]");
  }
  return a;
}

static int rope_multi_position(const uint32_t s[4], uint32_t mode, int64_t pair, int64_t* step) {
  const int64_t sum = int64_t(s[0]) + s[1] + s[2] + s[3], j = pair % sum;
  const int64_t w = int64_t(s[0]) + s[1], e = w + s[2];
  int k = j < int64_t(s[0]) ? 0 : j < w ? 1 : j < e ? 2 : 3;
  if (mode == LARCH_ROPE_IMROPE)
    k = j % 3 == 1 && j < 3 * int64_t(s[1])   ? 1
        : j % 3 == 2 && j < 3 * int64_t(s[2]) ? 2
        : j % 3 == 0 && j < 3 * int64_t(s[0]) ? 0
                                              : 3;
  // ggml_mrope_cache_init's theta: every position's runs on from pair 0, unless VISION starts it over
  // at its section's first pair (each time the sectors come round).
  const int64_t start[4] = {0, int64_t(s[0]), w, e};
  *step = mode == LARCH_ROPE_VISION ? j - start[k] : pair;
  return k;
}

// ggml_rope_multi in place, f32 math in ggml's order (rope_yarn, iterated theta per position).
larch_status rope_multi(larch_stream*, const larch_tensor* qk, const larch_tensor* pos_i32,
                         const larch_rope_multi_params* p) {
  return run("rope_multi", [&] {
    const rope_checked args = check_rope_multi(buffer_bytes, qk, pos_i32, p);
    const view q = args.qk, pos = args.pos;
    const larch_rope_params* r = &p->rope;
    const int64_t heads = q.shape[1], hd = q.shape[2], nd = r->n_dims;
    const bool vision = p->mode == LARCH_ROPE_VISION;
    const int64_t pairs = vision ? nd : nd / 2;  // pair i is element i with element i + pairs
    const yarn angle(r, args.freq_factors);
    const float theta_scale = powf(r->freq_base, -2.0f / float(nd));
    std::vector<float> cs(static_cast<size_t>(2 * pairs));
    for (int64_t row = 0; row < q.rows; ++row) {
      for (int64_t i = 0; i < pairs; ++i) {
        int64_t step = 0;
        const int k = rope_multi_position(p->sections, p->mode, i, &step);
        float theta = float(pos.i32(row, k));
        for (int64_t n = 0; n < step; ++n) theta *= theta_scale;
        angle(theta, i, &cs[size_t(2 * i)]);
      }
      for (int64_t h = 0; h < heads; ++h)
        for (int64_t i = 0; i < pairs; ++i) {
          const int64_t a = h * hd + i, b = a + pairs;
          const float x0 = q.get(row, a), x1 = q.get(row, b), c = cs[size_t(2 * i)], s = cs[size_t(2 * i) + 1];
          q.set(row, a, x0 * c - x1 * s);
          q.set(row, b, x0 * s + x1 * c);
        }
    }
  });
}

larch_status rope_f32(larch_stream*, const larch_tensor* qk, const larch_tensor* pos_i32,
                       const larch_rope_params* p) {
  return rope_as("rope_f32", LARCH_F32, qk, pos_i32, p);
}

// The checks mmvq and gemm share, after their own.
static void check_matmul(const sizer& bytes, matmul_checked& m, const larch_tensor* bias) {
  need(m.a.ndim == 2 && m.b_t.ndim == 2 && m.a.cols == m.b_t.cols, "inputs must be [m, k] and [n, k]");
  need((m.c.dtype == LARCH_F16 || m.c.dtype == LARCH_F32) && m.c.ndim == 2 && m.c.rows == m.a.rows &&
           m.c.cols == m.b_t.rows,
       "out must be f16 or f32 [m, n]");
  if (bias) {
    m.bias = check_tensor(bytes, bias, "bias");
    need(is(m.bias, LARCH_F32, 1) && m.bias.rows == m.b_t.rows, "bias must be f32 [n]");
  }
}

matmul_checked check_gemm(const sizer& bytes, const larch_tensor* a_f16, const larch_tensor* b_f16_t,
                          const larch_tensor* bias_or_null, const larch_tensor* c) {
  matmul_checked m;
  m.a = check_tensor(bytes, a_f16, "a");
  m.b_t = check_tensor(bytes, b_f16_t, "b_t");
  m.c = check_tensor(bytes, c, "c");
  need(m.a.dtype == LARCH_F16 && m.b_t.dtype == LARCH_F16, "a and b_t must be f16");
  check_matmul(bytes, m, bias_or_null);
  return m;
}

matmul_checked check_quant_gemm(const sizer& bytes, const larch_tensor* w_quant, const larch_tensor* x_f16,
                                const larch_tensor* bias_or_null, const larch_tensor* out) {
  matmul_checked m;
  m.b_t = check_tensor(bytes, w_quant, "w");
  m.a = check_tensor(bytes, x_f16, "x");
  m.c = check_tensor(bytes, out, "out");
  need(is_quant(m.b_t.dtype) && m.a.dtype == LARCH_F16, "w must be quant and x f16");
  check_matmul(bytes, m, bias_or_null);
  return m;
}

matmul_checked check_mmvq(const sizer& bytes, const larch_tensor* w_quant, const larch_tensor* x_f16,
                          const larch_tensor* bias_or_null, const larch_tensor* out) {
  const matmul_checked m = check_quant_gemm(bytes, w_quant, x_f16, bias_or_null, out);
  need(m.a.rows <= 16, "x has more than 16 rows: use dequant + gemm");
  return m;
}

matmul_checked check_mmvq_verify(const sizer& bytes, const larch_tensor* w_quant, const larch_tensor* x_f16,
                                 const larch_tensor* bias_or_null, const larch_tensor* out) {
  const matmul_checked m = check_quant_gemm(bytes, w_quant, x_f16, bias_or_null, out);
  need(m.a.rows <= 32, "x has more than 32 rows");
  return m;
}

mmvq_id_checked check_mmvq_id(const sizer& bytes, const larch_tensor* w_quant, const larch_tensor* x_f16,
                              const larch_tensor* ids_i32, const larch_tensor* bias_or_null,
                              const larch_tensor* weights_or_null, const larch_tensor* out) {
  mmvq_id_checked m;
  m.w = check_tensor(bytes, w_quant, "w");
  m.x = check_tensor(bytes, x_f16, "x");
  m.ids = check_tensor(bytes, ids_i32, "ids");
  m.out = check_tensor(bytes, out, "out");
  need(is_quant(m.w.dtype) && m.x.dtype == LARCH_F16, "w must be quant and x f16");
  need(is(m.ids, LARCH_I32, 2) && m.ids.rows == m.x.rows, "ids must be i32 [rows, slots]");
  const int64_t slots = m.ids.cols, k = m.w.cols;
  need((m.x.ndim == 2 && m.x.cols == k) || (m.x.ndim == 3 && m.x.shape[1] == slots && m.x.shape[2] == k),
       "x must be f16 [rows, k] or [rows, slots, k]");
  need(m.out.dtype == LARCH_F16 || m.out.dtype == LARCH_F32, "out must be f16 or f32");
  if (weights_or_null) {
    m.weights = check_tensor(bytes, weights_or_null, "weights");
    need(is(m.weights, LARCH_F32, 2) && m.weights.rows == m.x.rows && m.weights.cols == slots,
         "weights must be f32 [rows, slots]");
    need(m.out.ndim == 2 && m.out.rows == m.x.rows, "out must be [rows, N] with weights");
  } else {
    need(m.out.ndim == 3 && m.out.rows == m.x.rows && m.out.shape[1] == slots, "out must be [rows, slots, N]");
  }
  const int64_t n = m.out.shape[m.out.ndim - 1];
  need(m.w.rows >= n && m.w.rows % n == 0, "w must be [experts * N, k]");
  if (bias_or_null) {
    m.bias = check_tensor(bytes, bias_or_null, "bias");
    need(is(m.bias, LARCH_F32, 1) && m.bias.rows == m.w.rows, "bias must be f32 [experts * N]");
  }
  return m;
}

// c[m, n] = alpha * sum_k a[m, k] b_t[n, k] + beta * c[m, n] + bias[n]; c is not read when beta == 0.
// The body of both mmvq (alpha 1, beta 0) and gemm. Each row of b_t is decoded once.
static void matmul(const matmul_checked& args, float alpha, float beta) {
  const view a = args.a, bt = args.b_t, b = args.bias, c = args.c;
  const bool bias = b.buf;
  const size_t k = size_t(a.cols);
  std::vector<float> as(size_t(a.rows) * k), br(k);
  for (int64_t m = 0; m < a.rows; ++m) a.row_f32(m, as.data() + size_t(m) * k);
  for (int64_t n = 0; n < bt.rows; ++n) {
    bt.row_f32(n, br.data());
    for (int64_t m = 0; m < a.rows; ++m) {
      const float* am = as.data() + size_t(m) * k;
      double acc = 0;
      for (size_t i = 0; i < k; ++i) acc += double(am[i]) * br[i];
      double v = double(alpha) * acc;
      if (beta != 0.0f) v += double(beta) * c.get(m, n);
      if (bias) v += b.get(n, 0);
      c.set(m, n, float(v));
    }
  }
}

// The decode GEMV: out[r, n] = sum_k w[n, k] x[r, k] + bias[n], rows <= 16.
larch_status mmvq(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                   const larch_tensor* bias_or_null, const larch_tensor* out) {
  return run("mmvq", [&] { matmul(check_mmvq(buffer_bytes, w_quant, x_f16, bias_or_null, out), 1.0f, 0.0f); });
}

// mmvq's rows, each on its own: matmul never mixes rows, so each equals the one-row mmvq bit for bit.
larch_status mmvq_verify(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                          const larch_tensor* bias_or_null, const larch_tensor* out) {
  return run("mmvq_verify",
             [&] { matmul(check_mmvq_verify(buffer_bytes, w_quant, x_f16, bias_or_null, out), 1.0f, 0.0f); });
}

mmvq_id_mixed_checked check_mmvq_id_mixed(const sizer& bytes, const larch_tensor* w_hot, const larch_tensor* w_cold,
                                          const larch_tensor* table_i32, const larch_tensor* x_f16,
                                          const larch_tensor* ids_i32, const larch_tensor* weights_or_null,
                                          const larch_tensor* out) {
  mmvq_id_mixed_checked m;
  m.id = check_mmvq_id(bytes, w_hot, x_f16, ids_i32, nullptr, weights_or_null, out);
  m.cold = check_tensor(bytes, w_cold, "w_cold");
  m.table = check_tensor(bytes, table_i32, "table");
  const int64_t n = m.id.out.shape[m.id.out.ndim - 1];
  need(is_quant(m.cold.dtype) && m.cold.ndim == 2 && m.cold.cols == m.id.w.cols && m.cold.rows % n == 0,
       "w_cold must be quant [cold * N, k]");
  need(is(m.table, LARCH_I32, 1) && m.table.rows > 0, "table must be i32 [experts]");
  return m;
}

// out from each (row, slot)'s expert ids[r, j] in [0, experts), whose row col `row(e, col, wr)` decodes into wr (and
// whose bias, if any, it returns): y[r, j] = that row's GEMV of x's row r (or row (r, j)); out is y, or the weights' sum
// of it over j ascending, rounded once. mmvq_id's sum, and mmvq_id_mixed's, which so match bit for bit. Each (r, col)
// output is computed on its own.
template <class Row>
static void id_sums(const view& x, const view& ids, const view& wt, const view& o, int64_t k, int64_t experts,
                    const Row& row) {
  const int64_t slots = ids.cols, n = o.shape[o.ndim - 1], xs = x.ndim == 3 ? slots : 1;
  for (int64_t r = 0; r < x.rows; ++r)
    for (int64_t j = 0; j < slots; ++j) need(ids.i32(r, j) >= 0 && ids.i32(r, j) < experts, "expert id out of range");
  std::vector<float> xf(static_cast<size_t>(x.rows * xs * k));
  for (int64_t r = 0; r < x.rows; ++r)
    for (int64_t i = 0; i < xs * k; ++i) xf[size_t(r * xs * k + i)] = x.get(r, i);
  {
    std::vector<float> wr(static_cast<size_t>(k));
    for (int64_t item = 0; item < x.rows * n; ++item) {
      const int64_t r = item / n, col = item % n;
      double sum = 0;
      for (int64_t j = 0; j < slots; ++j) {
        const float* xr = xf.data() + (r * xs + (xs > 1 ? j : 0)) * k;
        const std::optional<double> bias = row(ids.i32(r, j), col, wr.data());
        double y = 0;
        for (size_t i = 0; i < size_t(k); ++i) y += double(xr[i]) * wr[i];
        if (bias) y += *bias;
        if (wt.buf) sum += double(wt.get(r, j)) * y;
        else o.set(r, j * n + col, float(y));
      }
      if (wt.buf) o.set(r, col, float(sum));
    }
  }
}

// y[r, j] = expert ids[r, j]'s GEMV of x's row r (or row (r, j)) plus its bias; out is y, or the
// weights' sum of it over j, rounded once.
larch_status mmvq_id(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                      const larch_tensor* ids_i32, const larch_tensor* bias_or_null,
                      const larch_tensor* weights_or_null, const larch_tensor* out) {
  return run("mmvq_id", [&] {
    const mmvq_id_checked c = check_mmvq_id(buffer_bytes, w_quant, x_f16, ids_i32, bias_or_null, weights_or_null, out);
    const view w = c.w, b = c.bias;
    const int64_t n = c.out.shape[c.out.ndim - 1];
    id_sums(c.x, c.ids, c.weights, c.out, w.cols, w.rows / n, [&](int32_t e, int64_t col, float* wr) {
      w.row_f32(e * n + col, wr);
      return b.buf ? std::optional<double>(b.get(e * n + col, 0)) : std::nullopt;
    });
  });
}

// mmvq_id's sums, each expert's rows read from the tensor and slot its table entry names.
larch_status mmvq_id_mixed(larch_stream*, const larch_tensor* w_hot, const larch_tensor* w_cold,
                            const larch_tensor* table_i32, const larch_tensor* x_f16, const larch_tensor* ids_i32,
                            const larch_tensor* weights_or_null, const larch_tensor* out) {
  return run("mmvq_id_mixed", [&] {
    const mmvq_id_mixed_checked c = check_mmvq_id_mixed(buffer_bytes, w_hot, w_cold, table_i32, x_f16, ids_i32,
                                                        weights_or_null, out);
    const view hot = c.id.w, cold = c.cold, table = c.table;
    const int64_t n = c.id.out.shape[c.id.out.ndim - 1];
    id_sums(c.id.x, c.id.ids, c.id.weights, c.id.out, hot.cols, table.rows, [&](int32_t e, int64_t col, float* wr) {
      const int32_t s = table.i32(e, 0);
      const view& w = s >= 0 ? hot : cold;
      const int64_t slot = s >= 0 ? s : -1 - int64_t(s);
      need(slot < w.rows / n, "table slot out of range");
      w.row_f32(slot * n + col, wr);
      return std::optional<double>();
    });
  });
}

// Independent scalar oracle for the explicit activation-INT8 contract. Integer dots are bounded
// per 32 values; double sums isolate CUDA's f32 accumulation error from activation approximation.
larch_status quant_gemm_i8(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                            const larch_tensor* bias_or_null, const larch_tensor* out) {
  return run("quant_gemm_i8", [&] {
    const matmul_checked checked = check_quant_gemm(buffer_bytes, w_quant, x_f16, bias_or_null, out);
    const view w = checked.b_t, x = checked.a, b = checked.bias, o = checked.c;
    // Refuse before the empty-row return, so the answer does not depend on the row count.
    need(w.dtype != LARCH_Q3_K && w.dtype != LARCH_Q2_K, "no int8 policy for this dtype");
    if (!x.rows) return;
    const int64_t groups = w.cols / 32;
    struct activation {
      float d;
      double sum;
      int8_t q[32];
    };
    std::vector<activation> packed(size_t(x.rows * groups));
    for (int64_t r = 0; r < x.rows; ++r)
      for (int64_t g = 0; g < groups; ++g) {
        activation& a = packed[size_t(r * groups + g)];
        float amax = 0.0f;
        bool finite = true;
        a.sum = 0.0;
        for (int j = 0; j < 32; ++j) {
          const float v = x.get(r, g * 32 + j);
          finite = finite && std::isfinite(v);
          amax = std::max(amax, std::abs(v));
          a.sum += v;
        }
        if (!finite) {
          a.d = std::numeric_limits<float>::quiet_NaN();
          a.sum = std::numeric_limits<double>::quiet_NaN();
          std::fill(std::begin(a.q), std::end(a.q), int8_t(0));
          continue;
        }
        a.d = amax / 127.0f;
        for (int j = 0; j < 32; ++j) {
          const float v = x.get(r, g * 32 + j);
          // Round ties to even explicitly; nearbyint depends on the caller's ambient rounding mode.
          const double scaled = amax == 0.0f ? 0.0 : double(v * (127.0f / amax));
          const double magnitude = std::abs(scaled), lower = std::floor(magnitude);
          const double fraction = magnitude - lower;
          int q = int(lower) + int(fraction > 0.5 || (fraction == 0.5 && int(lower) % 2 != 0));
          if (scaled < 0) q = -q;
          a.q[j] = int8_t(std::clamp(q, -127, 127));
        }
      }
    std::vector<int8_t> codes(static_cast<size_t>(groups * 32));
    std::vector<float> scales(static_cast<size_t>(groups * 2)), mins(static_cast<size_t>(groups));
    for (int64_t col = 0; col < w.rows; ++col) {
      for (int64_t g = 0; g < groups; ++g)
        i8_group(w.dtype, w.row(col), g, &codes[size_t(g * 32)], &scales[size_t(g * 2)], &mins[size_t(g)]);
      for (int64_t r = 0; r < x.rows; ++r) {
        double sum = 0.0;
        for (int64_t g = 0; g < groups; ++g) {
          const activation& a = packed[size_t(r * groups + g)];
          int dot[2] = {};
          for (int j = 0; j < 32; ++j) dot[j / 16] += int(codes[size_t(g * 32 + j)]) * int(a.q[j]);
          sum += double(scales[size_t(g * 2)]) * a.d * dot[0] + double(scales[size_t(g * 2 + 1)]) * a.d * dot[1] -
                 double(mins[size_t(g)]) * a.sum;
        }
        if (b.buf) sum += b.get(col, 0);
        o.set(r, col, float(sum));
      }
    }
  });
}

dequant_checked check_dequant(const sizer& bytes, const larch_tensor* w_quant, const larch_tensor* out_f16) {
  const dequant_checked a{check_tensor(bytes, w_quant, "w"), check_tensor(bytes, out_f16, "out")};
  need(is_quant(a.w.dtype) && is(a.out, LARCH_F16, 2) && a.out.rows == a.w.rows && a.out.cols == a.w.cols,
       "w must be quant and out f16 of w's shape");
  return a;
}

// out = w, dequantized.
larch_status dequant(larch_stream*, const larch_tensor* w_quant, const larch_tensor* out_f16) {
  return run("dequant", [&] {
    const dequant_checked a = check_dequant(buffer_bytes, w_quant, out_f16);
    const view w = a.w, o = a.out;
    std::vector<float> row(static_cast<size_t>(w.cols));
    for (int64_t r = 0; r < w.rows; ++r) {
      w.row_f32(r, row.data());
      for (int64_t c = 0; c < w.cols; ++c) o.set(r, c, row[size_t(c)]);
    }
  });
}

larch_status gemm(larch_stream*, const larch_tensor* a_f16, const larch_tensor* b_f16_t,
                   const larch_tensor* bias_or_null, const larch_tensor* c, float alpha, float beta) {
  return run("gemm", [&] { matmul(check_gemm(buffer_bytes, a_f16, b_f16_t, bias_or_null, c), alpha, beta); });
}

elementwise_checked check_silu_mul(const sizer& bytes, const larch_tensor* gate, const larch_tensor* up,
                                   const larch_tensor* out, larch_dtype t) {
  const elementwise_checked a{check_tensor(bytes, gate, "gate"), check_tensor(bytes, up, "up"),
                              check_tensor(bytes, out, "out")};
  need(is(a.a, t, 2) && is(a.b, t, 2) && is(a.out, t, 2) && same_shape(a.a, a.b) && same_shape(a.a, a.out),
       t == LARCH_F16 ? "gate, up and out must be f16 [rows, ff]" : "gate, up and out must be f32 [rows, ff]");
  return a;
}

// out = silu(gate) * up, silu(x) = x / (1 + e^-x).
static larch_status silu_mul_as(const char* op, larch_dtype t, const larch_tensor* gate, const larch_tensor* up,
                                 const larch_tensor* out) {
  return run(op, [&] {
    const elementwise_checked a = check_silu_mul(buffer_bytes, gate, up, out, t);
    const view g = a.a, u = a.b, o = a.out;
    for (int64_t r = 0; r < g.rows; ++r)
      for (int64_t c = 0; c < g.cols; ++c) {
        const float x = g.get(r, c);
        o.set(r, c, x / (1.0f + expf(-x)) * u.get(r, c));
      }
  });
}

larch_status silu_mul(larch_stream*, const larch_tensor* gate, const larch_tensor* up, const larch_tensor* out) {
  return silu_mul_as("silu_mul", LARCH_F16, gate, up, out);
}

larch_status silu_mul_f32(larch_stream*, const larch_tensor* gate, const larch_tensor* up,
                           const larch_tensor* out) {
  return silu_mul_as("silu_mul_f32", LARCH_F32, gate, up, out);
}

elementwise_checked check_add(const sizer& bytes, const larch_tensor* a_f32, const larch_tensor* b_f16,
                              const larch_tensor* out) {
  const elementwise_checked a{check_tensor(bytes, a_f32, "a"), check_tensor(bytes, b_f16, "b"),
                              check_tensor(bytes, out, "out")};
  need(is(a.a, LARCH_F32, 2) && is(a.b, LARCH_F16, 2) && is(a.out, LARCH_F32, 2) && same_shape(a.a, a.b) &&
           same_shape(a.a, a.out),
       "a must be f32, b f16 and out f32, all [rows, d]");
  return a;
}

elementwise_checked check_sigmoid_mul(const sizer& bytes, const larch_tensor* x, const larch_tensor* gate,
                                      const larch_tensor* out) {
  const elementwise_checked a{check_tensor(bytes, x, "x"), check_tensor(bytes, gate, "gate"),
                              check_tensor(bytes, out, "out")};
  need(is(a.a, LARCH_F16, 2) && is(a.out, LARCH_F16, 2) && same_shape(a.a, a.out), "x and out must be f16 [rows, d]");
  need(is(a.b, LARCH_F16, 2) && a.b.rows == a.a.rows && (a.b.cols == a.a.cols || a.b.cols == 1),
       "gate must be f16 [rows, d] or [rows, 1]");
  return a;
}

// out = x * sigmoid(gate), gate per element or per row; out may be x.
larch_status sigmoid_mul(larch_stream*, const larch_tensor* x, const larch_tensor* gate, const larch_tensor* out) {
  return run("sigmoid_mul", [&] {
    const elementwise_checked a = check_sigmoid_mul(buffer_bytes, x, gate, out);
    const view xv = a.a, g = a.b, o = a.out;
    for (int64_t r = 0; r < xv.rows; ++r)
      for (int64_t c = 0; c < xv.cols; ++c)
        o.set(r, c, xv.get(r, c) * (1.0f / (1.0f + expf(-g.get(r, g.cols == 1 ? 0 : c)))));
  });
}

elementwise_checked check_gelu(const sizer& bytes, const larch_tensor* x, const larch_tensor* out) {
  const elementwise_checked a{check_tensor(bytes, x, "x"), {}, check_tensor(bytes, out, "out")};
  need(is(a.a, LARCH_F16, 2) && is(a.out, LARCH_F16, 2) && same_shape(a.a, a.out), "x and out must be f16 [rows, d]");
  return a;
}

// ggml_gelu's tanh form in f32, as ggml-cuda computes it; out may be x.
larch_status gelu(larch_stream*, const larch_tensor* x, const larch_tensor* out) {
  return run("gelu", [&] {
    const elementwise_checked a = check_gelu(buffer_bytes, x, out);
    const view xv = a.a, o = a.out;
    for (int64_t r = 0; r < xv.rows; ++r)
      for (int64_t c = 0; c < xv.cols; ++c) {
        const float v = xv.get(r, c);
        o.set(r, c, 0.5f * v * (1.0f + tanhf(GELU_K * v * (1.0f + GELU_A * v * v))));
      }
  });
}

attn_full_checked check_attn_full(const sizer& bytes, const larch_tensor* q, const larch_tensor* k,
                                  const larch_tensor* v, uint32_t seg_len, const larch_tensor* out) {
  const attn_full_checked a{check_tensor(bytes, q, "q"), check_tensor(bytes, k, "k"), check_tensor(bytes, v, "v"),
                            check_tensor(bytes, out, "out")};
  need(is(a.q, LARCH_F16, 3) && is(a.k, LARCH_F16, 3) && is(a.v, LARCH_F16, 3) && is(a.out, LARCH_F16, 3) &&
           same_shape(a.q, a.k) && same_shape(a.q, a.v) && same_shape(a.q, a.out),
       "q, k, v and out must be f16 [rows, heads, head_dim] of one shape");
  need(seg_len > 0 && a.q.rows % seg_len == 0, "rows must be whole segments of seg_len > 0");
  return a;
}

// Per segment and head: softmax(scale q k^T) v over the segment's rows, in double.
larch_status attn_full(larch_stream*, const larch_tensor* q, const larch_tensor* k, const larch_tensor* v,
                        uint32_t seg_len, float scale, const larch_tensor* out) {
  return run("attn_full", [&] {
    const attn_full_checked a = check_attn_full(buffer_bytes, q, k, v, seg_len, out);
    const view qv = a.q, kv = a.k, vv = a.v, o = a.out;
    const int64_t heads = qv.shape[1], hd = qv.shape[2], n = seg_len;
    std::vector<double> s(size_t(n), 0.0), acc(size_t(hd), 0.0);
    for (int64_t r = 0; r < qv.rows; ++r) {
      const int64_t base = r / n * n;
      for (int64_t h = 0; h < heads; ++h) {
        double mx = -std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < n; ++j) {
          double d = 0;
          for (int64_t c = 0; c < hd; ++c) d += double(qv.get(r, h * hd + c)) * kv.get(base + j, h * hd + c);
          s[size_t(j)] = d * scale;
          mx = std::max(mx, s[size_t(j)]);
        }
        double sum = 0;
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int64_t j = 0; j < n; ++j) {
          const double p = std::exp(s[size_t(j)] - mx);
          sum += p;
          for (int64_t c = 0; c < hd; ++c) acc[size_t(c)] += p * vv.get(base + j, h * hd + c);
        }
        for (int64_t c = 0; c < hd; ++c) o.set(r, h * hd + c, float(acc[size_t(c)] / sum));
      }
    }
  });
}

// The residual update out = a + b; out may be a.
larch_status add(larch_stream*, const larch_tensor* a_f32, const larch_tensor* b_f16, const larch_tensor* out) {
  return run("add", [&] {
    const elementwise_checked a = check_add(buffer_bytes, a_f32, b_f16, out);
    const view av = a.a, bv = a.b, o = a.out;
    for (int64_t r = 0; r < av.rows; ++r)
      for (int64_t c = 0; c < av.cols; ++c) o.set(r, c, av.get(r, c) + bv.get(r, c));
  });
}

draft_conv_checked check_draft_conv(const sizer& bytes, const larch_tensor* x, const larch_tensor* dyn,
                                    const larch_tensor* base, uint32_t stage, const larch_tensor* out) {
  const draft_conv_checked a{check_tensor(bytes, x, "x"), check_tensor(bytes, dyn, "dyn"),
                             check_tensor(bytes, base, "base"), check_tensor(bytes, out, "out")};
  need(is(a.x, LARCH_F16, 2) && (is(a.out, LARCH_F16, 2) || is(a.out, LARCH_F32, 2)) && same_shape(a.x, a.out),
       "x must be f16 [rows, d] and out f16 or f32 of its shape");
  need(is(a.dyn, LARCH_F32, 3) && a.dyn.rows == a.x.rows && a.dyn.shape[1] == 4 && a.x.cols % a.dyn.shape[2] == 0,
       "dyn must be f32 [rows, 4, groups], groups dividing d");
  need(is(a.base, LARCH_F32, 2) && a.base.rows == 4 && a.base.cols == a.x.cols, "base must be f32 [4, d]");
  need(stage <= 1, "stage must be 0 or 1");
  return a;
}

// DFlash2's dynamic conv, in the f32 order larch.h gives (the GPU kernels keep it).
larch_status draft_conv(larch_stream*, const larch_tensor* x, const larch_tensor* dyn, const larch_tensor* base,
                         uint32_t stage, const larch_tensor* out) {
  return run("draft_conv", [&] {
    const draft_conv_checked a = check_draft_conv(buffer_bytes, x, dyn, base, stage, out);
    const view xv = a.x, dv = a.dyn, bv = a.base, o = a.out;
    const int64_t groups = dv.shape[2], group = xv.cols / groups, t0 = 2 * stage, t1 = t0 + 1;
    for (int64_t r = 0; r < xv.rows; ++r)
      for (int64_t c = 0; c < xv.cols; ++c) {
        float v = o.dtype == LARCH_F32 ? o.get(r, c) : 0.0f;
        v += xv.get(r, c) * (bv.get(t0, c) + dv.get(r, t0 * groups + c / group));
        if (r > 0) v += xv.get(r - 1, c) * (bv.get(t1, c) + dv.get(r, t1 * groups + c / group));
        o.set(r, c, v);
      }
  });
}

kv_write_checked check_kv_write(const sizer& bytes, const larch_tensor* k, const larch_tensor* v,
                                const larch_tensor* slots_i32, const larch_kv_pool* pool) {
  need(pool != nullptr, "pool is null");
  kv_write_checked a;
  a.pool_k = pool_layer(bytes, pool, pool->k, "pool k");
  a.pool_v = pool_layer(bytes, pool, pool->v, "pool v");
  a.k = check_tensor(bytes, k, "k");
  a.v = check_tensor(bytes, v, "v");
  a.slots = check_tensor(bytes, slots_i32, "slots");
  need(is(a.k, LARCH_F16, 3) && a.k.shape[1] == pool->n_kv_heads && a.k.shape[2] == pool->head_dim &&
           is(a.v, LARCH_F16, 3) && same_shape(a.k, a.v),
       "k and v must be f16 [rows, n_kv_heads, head_dim]");
  need(is(a.slots, LARCH_I32, 1) && a.slots.rows == a.k.rows, "slots must be i32 [rows]");
  return a;
}

// Stores row r of k and v at slot slots[r] of the pool's layer, rounded into its dtype; a negative slot
// skips the row.
larch_status kv_write(larch_stream*, const larch_tensor* k, const larch_tensor* v, const larch_tensor* slots_i32,
                       const larch_kv_pool* pool) {
  return run("kv_write", [&] {
    const kv_write_checked a = check_kv_write(buffer_bytes, k, v, slots_i32, pool);
    const view kp = a.pool_k, vp = a.pool_v, kv = a.k, vv = a.v, slots = a.slots;
    std::vector<float> row(static_cast<size_t>(kv.cols));
    for (int64_t r = 0; r < kv.rows; ++r) {
      const int32_t s = slots.i32(r);
      if (s < 0) continue;
      need(s < kp.rows, "slot out of range");
      kv.row_f32(r, row.data());
      kv_from_f32(kp.dtype, row.data(), kv.cols, kp.row(s));
      vv.row_f32(r, row.data());
      kv_from_f32(vp.dtype, row.data(), vv.cols, vp.row(s));
    }
  });
}

// The checks all three attention ops share: the pool layer, q, out, the block table and the GQA mapping.
static attn_checked check_attn_layer(const sizer& bytes, const larch_tensor* qt, const larch_kv_pool* pool,
                                     const larch_tensor* bt, const larch_tensor* ot) {
  need(pool != nullptr, "pool is null");
  attn_checked a;
  a.k = pool_layer(bytes, pool, pool->k, "pool k");
  a.v = pool_layer(bytes, pool, pool->v, "pool v");
  a.q = check_tensor(bytes, qt, "q");
  a.out = check_tensor(bytes, ot, "out");
  a.table = check_tensor(bytes, bt, "block_table");
  need(is(a.q, LARCH_F16, 3), "q must be f16 [rows, heads, head_dim]");
  need(a.q.shape[2] == pool->head_dim && a.q.shape[1] % pool->n_kv_heads == 0,
       "q's head_dim must be the pool's and heads a multiple of n_kv_heads");
  need(a.out.dtype == LARCH_F16 && same_shape(a.q, a.out), "out must be f16 of q's shape");
  need(is(a.table, LARCH_I32, 2), "block_table must be i32 [seqs, max_blocks]");
  return a;
}

attn_checked check_attn_decode(const sizer& bytes, const larch_tensor* q, const larch_kv_pool* pool,
                               const larch_tensor* block_table_i32, const larch_tensor* seq_lens_i32,
                               const larch_tensor* out) {
  attn_checked a = check_attn_layer(bytes, q, pool, block_table_i32, out);
  a.lens = check_tensor(bytes, seq_lens_i32, "seq_lens");
  need(a.table.rows == a.q.rows && is(a.lens, LARCH_I32, 1) && a.lens.rows == a.q.rows,
       "one block table row and seq_len per q row");
  return a;
}

attn_checked check_attn_prefill(const sizer& bytes, const larch_tensor* q, const larch_kv_pool* pool,
                                const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                                const larch_tensor* q_lens_i32, const larch_tensor* out) {
  attn_checked a = check_attn_layer(bytes, q, pool, block_table_i32, out);
  a.start = check_tensor(bytes, q_start_pos_i32, "q_start_pos");
  a.lens = check_tensor(bytes, q_lens_i32, "q_lens");
  const int64_t seqs = a.table.rows;
  need(is(a.start, LARCH_I32, 1) && a.start.rows == seqs && is(a.lens, LARCH_I32, 1) && a.lens.rows == seqs,
       "q_start_pos and q_lens must be i32 [seqs]");
  // The sum itself is in device memory; no sequences sum to 0 whatever it holds.
  need(seqs || !a.q.rows, "q rows must equal the sum of q_lens");
  return a;
}

attn_checked check_attn(const sizer& bytes, const larch_tensor* q, const larch_kv_pool* pool,
                        const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                        const larch_tensor* q_lens_i32, const larch_tensor* mask_or_null,
                        const larch_tensor* sinks_or_null, const larch_tensor* out) {
  attn_checked a = check_attn_prefill(bytes, q, pool, block_table_i32, q_start_pos_i32, q_lens_i32, out);
  if (mask_or_null) {
    a.mask = check_tensor(bytes, mask_or_null, "mask");
    need(is(a.mask, LARCH_I32, 1) && a.mask.rows == a.q.rows, "mask must be i32 [rows]");
  }
  if (sinks_or_null) {
    a.sinks = check_tensor(bytes, sinks_or_null, "sinks");
    need(is(a.sinks, LARCH_F32, 1) && a.sinks.rows == a.q.shape[1], "sinks must be f32 [n_heads]");
  }
  return a;
}

// The attention state: the checked tensors and the GQA mapping.
struct attn {
  view q, out, k, v, table, sinks;
  int64_t heads, kv_heads, hd, block_tokens, n_blocks;
  float scale;

  attn(const attn_checked& a, const larch_kv_pool* pool, float sc)
      : q(a.q),
        out(a.out),
        k(a.k),
        v(a.v),
        table(a.table),
        sinks(a.sinks),
        heads(a.q.shape[1]),
        kv_heads(pool->n_kv_heads),
        hd(pool->head_dim),
        block_tokens(pool->block_tokens),
        n_blocks(pool->n_blocks),
        scale(sc) {}

  // The slot of position p of sequence seq.
  int64_t slot(int64_t seq, int64_t p) const {
    need(p < table.cols * block_tokens, "sequence longer than its block table");
    const int32_t b = table.i32(seq, p / block_tokens);
    need(b >= 0 && b < n_blocks, "block id out of range");
    return int64_t(b) * block_tokens + p % block_tokens;
  }

  // Query row `row` attends to `slots` in order, all heads; softmax in double, with head h's sink as one
  // more logit when there are sinks. The pages are decoded once per row, so every dtype reads alike.
  void attend(int64_t row, const std::vector<int64_t>& slots) const {
    const size_t n = slots.size(), kc = size_t(k.cols);
    std::vector<float> ks(n * kc), vs(n * kc);
    for (size_t p = 0; p < n; ++p) {
      k.row_f32(slots[p], ks.data() + p * kc);
      v.row_f32(slots[p], vs.data() + p * kc);
    }
    std::vector<double> w(n);
    for (int64_t h = 0; h < heads; ++h) {
      const int64_t qo = h * hd, ko = h / (heads / kv_heads) * hd;
      const double sink = sinks.buf ? sinks.get(h, 0) : -std::numeric_limits<double>::infinity();
      double mx = sink;
      for (size_t p = 0; p < n; ++p) {
        double dot = 0;
        for (int64_t d = 0; d < hd; ++d) dot += double(q.get(row, qo + d)) * ks[p * kc + size_t(ko + d)];
        w[p] = dot * scale;
        mx = std::max(mx, w[p]);
      }
      double sum = sinks.buf ? std::exp(sink - mx) : 0;
      for (double& x : w) sum += x = std::exp(x - mx);
      for (int64_t d = 0; d < hd; ++d) {
        double acc = 0;
        for (size_t p = 0; p < n; ++p) acc += w[p] * vs[p * kc + size_t(ko + d)];
        out.set(row, qo + d, n ? float(acc / sum) : 0.0f);
      }
    }
  }
};

// Row r attends to positions [0, seq_lens[r]) through row r of the block table; seq_lens 0 is padding
// and gives zeros. splits is a speed hint with no effect here.
larch_status attn_decode(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool,
                          const larch_tensor* block_table_i32, const larch_tensor* seq_lens_i32, uint32_t,
                          float scale, const larch_tensor* out) {
  return run("attn_decode", [&] {
    const attn_checked c = check_attn_decode(buffer_bytes, q, pool, block_table_i32, seq_lens_i32, out);
    const attn a(c, pool, scale);
    const view lens = c.lens;
    std::vector<int64_t> slots;
    for (int64_t r = 0; r < a.q.rows; ++r) {
      need(lens.i32(r) >= 0, "negative seq_len");
      slots.clear();
      for (int64_t p = 0; p < lens.i32(r); ++p) slots.push_back(a.slot(r, p));
      a.attend(r, slots);
    }
  });
}

// Row j of sequence s, at p = q_start_pos[s] + j, attends to the context before the span, then to the
// span's positions its mask bits (or causality) keep; window > 0 drops positions p' with p - p' >= window.
// Each row's positions go to attend in order, which is what makes it decode's row over those positions.
static void attend_spans(const attn_checked& c, const larch_kv_pool* pool, float scale, uint32_t window) {
  const attn a(c, pool, scale);
  const view start = c.start, lens = c.lens, mask = c.mask;
  const int64_t seqs = a.table.rows;
  int64_t total = 0;
  for (int64_t s = 0; s < seqs; ++s) {
    need(start.i32(s) >= 0 && lens.i32(s) >= 0, "negative q_start_pos or q_len");
    need(!mask.buf || lens.i32(s) <= 32, "q_len over 32 with a mask");
    total += lens.i32(s);
  }
  need(total == a.q.rows, "q rows must equal the sum of q_lens");
  std::vector<int64_t> slots;
  for (int64_t s = 0, row = 0; s < seqs; ++s) {
    const int64_t p0 = start.i32(s), len = lens.i32(s);
    for (int64_t j = 0; j < len; ++j, ++row) {
      const int64_t lo = window ? std::max<int64_t>(0, p0 + j - window + 1) : 0;
      const uint64_t bits = mask.buf ? uint32_t(mask.i32(row)) : 0;
      need(!mask.buf || bits >> len == 0, "mask bit past the span");
      slots.clear();
      for (int64_t p = lo; p < p0; ++p) slots.push_back(a.slot(s, p));
      for (int64_t i = 0; i < len; ++i)
        if ((mask.buf ? (bits >> i & 1) != 0 : i <= j) && p0 + i >= lo) slots.push_back(a.slot(s, p0 + i));
      a.attend(row, slots);
    }
  }
}

// Causal and chunk-aware: row j of sequence s sits at position q_start_pos[s] + j and attends to
// positions [0, that position]; earlier positions come from the pool (cached or earlier chunks).
larch_status attn_prefill(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool,
                           const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                           const larch_tensor* q_lens_i32, float scale, const larch_tensor* out) {
  return run("attn_prefill", [&] {
    attend_spans(check_attn_prefill(buffer_bytes, q, pool, block_table_i32, q_start_pos_i32, q_lens_i32, out), pool,
                 scale, 0);
  });
}

// larch.h's general entry; splits has no effect here.
larch_status attn(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool,
                   const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                   const larch_tensor* q_lens_i32, const larch_tensor* mask_or_null,
                   const larch_tensor* sinks_or_null, uint32_t window, uint32_t, float scale,
                   const larch_tensor* out) {
  return run("attn", [&] {
    attend_spans(check_attn(buffer_bytes, q, pool, block_table_i32, q_start_pos_i32, q_lens_i32, mask_or_null,
                            sinks_or_null, out),
                 pool, scale, window);
  });
}

norm_checked check_qk_norm(const sizer& bytes, const larch_tensor* qk, const larch_tensor* w) {
  norm_checked a{check_tensor(bytes, qk, "qk"), check_tensor(bytes, w, "w"), {}};
  need(is(a.x, LARCH_F16, 3) && is(a.w, LARCH_F32, 1) && a.w.rows == a.x.shape[2],
       "qk must be f16 [rows, heads, head_dim] and w f32 [head_dim]");
  a.out = a.x;
  return a;
}

// Qwen3: rmsnorm of every head of qk in place, times w [head_dim].
larch_status qk_norm(larch_stream*, const larch_tensor* qk, const larch_tensor* w, float eps) {
  return run("qk_norm", [&] {
    const norm_checked a = check_qk_norm(buffer_bytes, qk, w);
    const view q = a.x, wv = a.w;
    const int64_t hd = q.shape[2];
    for (int64_t r = 0; r < q.rows; ++r)
      for (int64_t h = 0; h < q.shape[1]; ++h) rms(q, wv, eps, q, r, h * hd, hd);
  });
}

sample_checked check_sample_prep(const sizer& bytes, const larch_tensor* logits, const larch_sample_params* per_row,
                                 const larch_tensor* penalty_ids_i32, const larch_tensor* out_topk_vals_f32,
                                 const larch_tensor* out_topk_ids_i32, const larch_tensor* out_lse_f32) {
  sample_checked a;
  a.logits = check_tensor(bytes, logits, "logits");
  a.vals = check_tensor(bytes, out_topk_vals_f32, "out_topk_vals");
  a.ids = check_tensor(bytes, out_topk_ids_i32, "out_topk_ids");
  a.lse = check_tensor(bytes, out_lse_f32, "out_lse");
  need(per_row != nullptr, "per_row is null");
  need(is(a.logits, LARCH_F32, 2), "logits must be f32 [rows, vocab]");
  const int64_t n = a.logits.rows;
  need(is(a.vals, LARCH_F32, 2) && a.vals.rows == n && a.vals.cols == 64, "out_topk_vals must be f32 [rows, 64]");
  need(is(a.ids, LARCH_I32, 2) && a.ids.rows == n && a.ids.cols == 64, "out_topk_ids must be i32 [rows, 64]");
  need(is(a.lse, LARCH_F32, 1) && a.lse.rows == n, "out_lse must be f32 [rows]");
  if (penalty_ids_i32) {
    a.penalty_ids = check_tensor(bytes, penalty_ids_i32, "penalty_ids");
    need(is(a.penalty_ids, LARCH_I32, 2) && a.penalty_ids.rows == n, "penalty_ids must be i32 [rows, width]");
  }
  for (int64_t r = 0; r < n; ++r)
    need(per_row[r].top_k <= 64 && per_row[r].repeat_penalty > 0.0f &&
             (!penalty_ids_i32 || per_row[r].penalty_window <= a.penalty_ids.cols),
         "per_row: top_k > 64, repeat_penalty <= 0 or penalty_window wider than penalty_ids");
  return a;
}

// Per row: llama.cpp's repetition penalty (once per distinct id in the window; a logit <= 0 is
// multiplied, a positive one divided), the log-sum-exp of the penalised row, and its top k
// (k = top_k ? top_k : 64), largest first, ties to the lower id, -inf and id -1 past k. Temperature is
// not read (the host applies it to the candidates, architecture 9.6).
larch_status sample_prep(larch_stream*, const larch_tensor* logits, const larch_sample_params* per_row,
                          const larch_tensor* penalty_ids_i32, const larch_tensor* out_topk_vals_f32,
                          const larch_tensor* out_topk_ids_i32, const larch_tensor* out_lse_f32) {
  return run("sample_prep", [&] {
    const sample_checked args = check_sample_prep(buffer_bytes, logits, per_row, penalty_ids_i32, out_topk_vals_f32,
                                                  out_topk_ids_i32, out_lse_f32);
    const view lg = args.logits, vals = args.vals, ids = args.ids, lse = args.lse, pen = args.penalty_ids;
    const int64_t n = lg.rows, vocab = lg.cols;
    std::vector<float> l(static_cast<size_t>(vocab));
    std::vector<int32_t> order(static_cast<size_t>(vocab)), window;
    for (int64_t r = 0; r < n; ++r) {
      const larch_sample_params& sp = per_row[r];
      lg.row_f32(r, l.data());
      need(std::none_of(l.begin(), l.end(), [](float x) { return std::isnan(x); }), "NaN logit");
      if (penalty_ids_i32 && sp.repeat_penalty != 1.0f) {
        window.clear();
        for (int64_t i = 0; i < sp.penalty_window; ++i)
          if (const int32_t id = pen.i32(r, i); id >= 0) {
            need(id < vocab, "penalty id out of range");
            window.push_back(id);
          }
        std::sort(window.begin(), window.end());
        window.erase(std::unique(window.begin(), window.end()), window.end());
        for (const int32_t id : window) {
          float& x = l[size_t(id)];
          x = x <= 0 ? x * sp.repeat_penalty : x / sp.repeat_penalty;
        }
      }
      const float mx = *std::max_element(l.begin(), l.end());
      double sum = 0;
      for (const float x : l) sum += std::exp(double(x) - mx);
      lse.set(r, 0, float(double(mx) + std::log(sum)));

      const int64_t k = std::min<int64_t>(sp.top_k ? sp.top_k : 64, vocab);
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int32_t a, int32_t b) {
        return l[size_t(a)] > l[size_t(b)] || (l[size_t(a)] == l[size_t(b)] && a < b);
      });
      for (int64_t j = 0; j < 64; ++j) {
        vals.set(r, j, j < k ? l[size_t(order[size_t(j)])] : -std::numeric_limits<float>::infinity());
        ids.set_i32(r, j, j < k ? order[size_t(j)] : -1);
      }
    }
  });
}

draw_checked check_sample(const sizer& bytes, const larch_tensor* topk_vals_f32, const larch_tensor* topk_ids_i32,
                          const larch_tensor* lse_f32, const larch_draw_params* per_row, const larch_tensor* pos_i32,
                          const larch_tensor* ban_ids_or_null, const larch_tensor* out_ids_i32,
                          const larch_tensor* out_logprobs_f32) {
  draw_checked a;
  a.vals = check_tensor(bytes, topk_vals_f32, "topk_vals");
  a.ids = check_tensor(bytes, topk_ids_i32, "topk_ids");
  a.lse = check_tensor(bytes, lse_f32, "lse");
  a.pos = check_tensor(bytes, pos_i32, "pos");
  a.out_ids = check_tensor(bytes, out_ids_i32, "out_ids");
  a.out_logprobs = check_tensor(bytes, out_logprobs_f32, "out_logprobs");
  need(per_row != nullptr, "per_row is null");
  const int64_t n = a.vals.rows;
  need(is(a.vals, LARCH_F32, 2) && a.vals.cols == 64, "topk_vals must be f32 [rows, 64]");
  need(is(a.ids, LARCH_I32, 2) && a.ids.rows == n && a.ids.cols == 64, "topk_ids must be i32 [rows, 64]");
  need(is(a.lse, LARCH_F32, 1) && a.lse.rows == n, "lse must be f32 [rows]");
  need(is(a.pos, LARCH_I32, 1) && a.pos.rows == n, "pos must be i32 [rows]");
  need(is(a.out_ids, LARCH_I32, 1) && a.out_ids.rows == n, "out_ids must be i32 [rows]");
  need(is(a.out_logprobs, LARCH_F32, 1) && a.out_logprobs.rows == n, "out_logprobs must be f32 [rows]");
  if (ban_ids_or_null) {
    a.ban = check_tensor(bytes, ban_ids_or_null, "ban_ids");
    need(is(a.ban, LARCH_I32, 2) && a.ban.rows == n, "ban_ids must be i32 [rows, width]");
  }
  for (int64_t r = 0; r < n; ++r)
    need(per_row[r].top_k <= 64 && per_row[r].top_p > 0.0f && per_row[r].top_p <= 1.0f &&
             !std::isnan(per_row[r].temperature),
         "per_row: top_k > 64, top_p outside (0, 1] or a NaN temperature");
  return a;
}

// The row's random word: splitmix64's output at index pos + 1 of the stream seeded with seed.
static uint64_t draw_word(uint64_t seed, int32_t pos) {
  uint64_t z = seed + (uint64_t(pos) + 1) * 0x9E3779B97F4A7C15ull;
  z = (z ^ z >> 30) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ z >> 27) * 0x94D049BB133111EBull;
  return z ^ z >> 31;
}

// larch.h's draw: the host sampler's math (crates/larch-engine/src/sample.rs), with the random word
// addressed by position instead of drawn from the host's ChaCha8 stream.
larch_status sample(larch_stream*, const larch_tensor* topk_vals_f32, const larch_tensor* topk_ids_i32,
                     const larch_tensor* lse_f32, const larch_draw_params* per_row, const larch_tensor* pos_i32,
                     const larch_tensor* ban_ids_or_null, const larch_tensor* out_ids_i32,
                     const larch_tensor* out_logprobs_f32) {
  return run("sample", [&] {
    const draw_checked a = check_sample(buffer_bytes, topk_vals_f32, topk_ids_i32, lse_f32, per_row, pos_i32,
                                        ban_ids_or_null, out_ids_i32, out_logprobs_f32);
    const view vals = a.vals, ids = a.ids, lse = a.lse, pos = a.pos, ban = a.ban, oi = a.out_ids, ol = a.out_logprobs;
    std::vector<float> v;
    std::vector<int32_t> id;
    for (int64_t r = 0; r < vals.rows; ++r) {
      const larch_draw_params& dp = per_row[r];
      need(pos.i32(r) >= 0, "negative pos");
      v.clear();
      id.clear();
      for (int64_t j = 0; j < 64; ++j) {
        bool banned = false;
        for (int64_t i = 0; ban.buf && i < ban.cols && !banned; ++i)
          banned = ban.i32(r, i) >= 0 && ban.i32(r, i) == ids.i32(r, j);
        if (!banned) {
          v.push_back(vals.get(r, j));
          id.push_back(ids.i32(r, j));
        }
      }
      const size_t n0 = size_t(std::find_if(id.begin(), id.end(), [](int32_t x) { return x < 0; }) - id.begin());
      size_t pick = 0;
      if (dp.temperature <= 0.0f) {
        pick = size_t(std::find_if(id.begin(), id.end(), [](int32_t x) { return x >= 0; }) - id.begin());
      } else if (n0) {
        const size_t k = dp.top_k ? std::min<size_t>(dp.top_k, n0) : n0;
        const auto l = [&](size_t i) { return double(v[i] - v[0]); };
        double t = 0, m = 0, sum = 0;
        for (size_t i = 0; i < k; ++i) t += std::exp(l(i));
        size_t n = 0;
        while (n < k && m < dp.top_p) m += std::exp(l(n++)) / t;
        const auto w = [&](size_t i) { return std::exp(l(i) / double(dp.temperature)); };
        for (size_t i = 0; i < n; ++i) sum += w(i);
        const double u = double(draw_word(dp.seed, pos.i32(r)) >> 11) * 0x1p-53 * sum;
        double acc = 0;
        pick = n - 1;
        for (size_t i = 0; i < n; ++i)
          if (u < (acc += w(i))) {
            pick = i;
            break;
          }
      } else {
        pick = id.size();
      }
      const bool none = pick >= id.size() || id[pick] < 0;
      oi.set_i32(r, 0, none ? -1 : id[pick]);
      ol.set(r, 0, none ? -std::numeric_limits<float>::infinity() : v[pick] - lse.get(r, 0));
    }
  });
}

route_checked check_moe_route(const sizer& bytes, const larch_tensor* logits_f32, float,
                              const larch_tensor* out_ids_i32, const larch_tensor* out_weights_f32) {
  const route_checked a{check_tensor(bytes, logits_f32, "logits"), check_tensor(bytes, out_ids_i32, "out_ids"),
                        check_tensor(bytes, out_weights_f32, "out_weights")};
  need(is(a.logits, LARCH_F32, 2), "logits must be f32 [rows, experts]");
  need(is(a.ids, LARCH_I32, 2) && a.ids.rows == a.logits.rows && a.ids.cols <= a.logits.cols,
       "out_ids must be i32 [rows, k], k <= experts");
  need(is(a.weights, LARCH_F32, 2) && same_shape(a.ids, a.weights), "out_weights must be f32 [rows, k]");
  return a;
}

// llama.cpp's softmax gating (build_moe_ffn): softmax, top k, the k's weights over their sum clamped at
// 2^-14, times scale; in double, rounded once.
larch_status moe_route(larch_stream*, const larch_tensor* logits_f32, float scale, const larch_tensor* out_ids_i32,
                        const larch_tensor* out_weights_f32) {
  return run("moe_route", [&] {
    const route_checked a = check_moe_route(buffer_bytes, logits_f32, scale, out_ids_i32, out_weights_f32);
    const view lg = a.logits, ids = a.ids, wt = a.weights;
    const int64_t experts = lg.cols, k = ids.cols;
    std::vector<float> l(static_cast<size_t>(experts));
    std::vector<int32_t> order(static_cast<size_t>(experts));
    for (int64_t r = 0; r < lg.rows; ++r) {
      lg.row_f32(r, l.data());
      need(std::none_of(l.begin(), l.end(), [](float x) { return std::isnan(x); }), "NaN logit");
      const double mx = *std::max_element(l.begin(), l.end());
      double z = 0, top = 0;
      for (const float x : l) z += std::exp(x - mx);
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int32_t x, int32_t y) {
        return l[size_t(x)] > l[size_t(y)] || (l[size_t(x)] == l[size_t(y)] && x < y);
      });
      for (int64_t j = 0; j < k; ++j) top += std::exp(l[size_t(order[size_t(j)])] - mx) / z;
      const double norm = (scale == 0.0f ? 1.0 : double(scale)) / std::max(top, 0x1p-14);
      for (int64_t j = 0; j < k; ++j) {
        ids.set_i32(r, j, order[size_t(j)]);
        wt.set(r, j, float(std::exp(l[size_t(order[size_t(j)])] - mx) / z * norm));
      }
    }
  });
}

// The training ops (larch.h): f32 in and out, double accumulators.
train_checked check_gemm_ex(const sizer& bytes, const larch_tensor* a, const larch_tensor* b, const larch_tensor* c,
                            uint32_t trans_a, uint32_t trans_b) {
  const train_checked g{check_tensor(bytes, a, "a"), check_tensor(bytes, b, "b"), check_tensor(bytes, c, "c"), {}, {}};
  need(is(g.a, LARCH_F32, 2) && is(g.b, LARCH_F32, 2) && is(g.c, LARCH_F32, 2), "a, b and c must be f32 [rows, cols]");
  const int64_t m = trans_a ? g.a.cols : g.a.rows, k = trans_a ? g.a.rows : g.a.cols;
  const int64_t kb = trans_b ? g.b.cols : g.b.rows, n = trans_b ? g.b.rows : g.b.cols;
  need(k == kb && g.c.rows == m && g.c.cols == n, "op(a) must be [m, k], op(b) [k, n] and c [m, n]");
  return g;
}

// c = alpha op(a) op(b) + beta c.
larch_status gemm_ex(larch_stream*, const larch_tensor* a, const larch_tensor* b, const larch_tensor* c,
                      uint32_t trans_a, uint32_t trans_b, float alpha, float beta) {
  return run("gemm_ex", [&] {
    const train_checked g = check_gemm_ex(buffer_bytes, a, b, c, trans_a, trans_b);
    const view av = g.a, bv = g.b, cv = g.c;
    const int64_t m = cv.rows, n = cv.cols, k = trans_a ? av.rows : av.cols;
    // op(a)'s rows and op(b)'s columns, each k dense.
    std::vector<double> ar(size_t(m * k)), bc(size_t(n * k));
    for (int64_t i = 0; i < m; ++i)
      for (int64_t j = 0; j < k; ++j) ar[size_t(i * k + j)] = trans_a ? av.get(j, i) : av.get(i, j);
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < k; ++j) bc[size_t(i * k + j)] = trans_b ? bv.get(i, j) : bv.get(j, i);
    for (int64_t i = 0; i < m; ++i)
      for (int64_t j = 0; j < n; ++j) {
        double acc = 0;
        for (int64_t l = 0; l < k; ++l) acc += ar[size_t(i * k + l)] * bc[size_t(j * k + l)];
        double v = double(alpha) * acc;
        if (beta != 0.0f) v += double(beta) * cv.get(i, j);
        cv.set(i, j, float(v));
      }
  });
}

// Every tensor f32 [rows, cols] of a's shape.
static void same_f32(std::initializer_list<const meta*> ts, const char* what) {
  for (const meta* t : ts) need(is(*t, LARCH_F32, 2) && same_shape(**ts.begin(), *t), what);
}

train_checked check_rmsnorm_backward(const sizer& bytes, const larch_tensor* x, const larch_tensor* w,
                                     const larch_tensor* dy, const larch_tensor* dx) {
  const train_checked a{check_tensor(bytes, x, "x"), check_tensor(bytes, w, "w"), check_tensor(bytes, dy, "dy"),
                        check_tensor(bytes, dx, "dx"), {}};
  same_f32({&a.a, &a.c, &a.d}, "x, dy and dx must be f32 [rows, d]");
  need(is(a.b, LARCH_F32, 1) && a.b.rows == a.a.cols && a.a.cols > 0, "w must be f32 [d], d > 0");
  return a;
}

larch_status rmsnorm_backward(larch_stream*, const larch_tensor* x, const larch_tensor* w, const larch_tensor* dy,
                               float eps, const larch_tensor* dx, uint32_t accumulate) {
  return run("rmsnorm_backward", [&] {
    const train_checked a = check_rmsnorm_backward(buffer_bytes, x, w, dy, dx);
    const view xv = a.a, wv = a.b, g = a.c, o = a.d;
    const int64_t d = xv.cols;
    for (int64_t r = 0; r < xv.rows; ++r) {
      double ss = 0, c = 0;
      for (int64_t i = 0; i < d; ++i) {
        const double v = xv.get(r, i);
        ss += v * v;
        c += double(g.get(r, i)) * wv.get(i, 0) * v;
      }
      // The scale as rmsnorm rounds it.
      const double s = float(1.0 / std::sqrt(ss / double(d) + eps));
      c *= s * s * s / double(d);
      for (int64_t i = 0; i < d; ++i) {
        const double v = s * wv.get(i, 0) * g.get(r, i) - xv.get(r, i) * c;
        o.set(r, i, float((accumulate ? o.get(r, i) : 0.0) + v));
      }
    }
  });
}

train_checked check_silu_mul_backward(const sizer& bytes, const larch_tensor* gate, const larch_tensor* up,
                                      const larch_tensor* dout, const larch_tensor* dgate, const larch_tensor* dup) {
  const train_checked a{check_tensor(bytes, gate, "gate"), check_tensor(bytes, up, "up"),
                        check_tensor(bytes, dout, "dout"), check_tensor(bytes, dgate, "dgate"),
                        check_tensor(bytes, dup, "dup")};
  same_f32({&a.a, &a.b, &a.c, &a.d, &a.e}, "gate, up, dout, dgate and dup must be f32 [rows, ff]");
  return a;
}

larch_status silu_mul_backward(larch_stream*, const larch_tensor* gate, const larch_tensor* up,
                                const larch_tensor* dout, const larch_tensor* dgate, const larch_tensor* dup) {
  return run("silu_mul_backward", [&] {
    const train_checked a = check_silu_mul_backward(buffer_bytes, gate, up, dout, dgate, dup);
    const view g = a.a, u = a.b, d = a.c, dg = a.d, du = a.e;
    for (int64_t r = 0; r < g.rows; ++r)
      for (int64_t c = 0; c < g.cols; ++c) {
        const float x = g.get(r, c), sg = 1.0f / (1.0f + expf(-x)), dy = d.get(r, c);
        dg.set(r, c, dy * u.get(r, c) * sg * (1.0f + x * (1.0f - sg)));
        du.set(r, c, dy * x * sg);
      }
  });
}

train_checked check_softmax(const sizer& bytes, const larch_tensor* x, const larch_tensor* out) {
  const train_checked a{check_tensor(bytes, x, "x"), check_tensor(bytes, out, "out"), {}, {}, {}};
  same_f32({&a.a, &a.b}, "x and out must be f32 [rows, n]");
  return a;
}

larch_status softmax(larch_stream*, const larch_tensor* x, float scale, uint32_t causal, const larch_tensor* out) {
  return run("softmax", [&] {
    const train_checked a = check_softmax(buffer_bytes, x, out);
    const view xv = a.a, o = a.b;
    const int64_t n = xv.cols;
    for (int64_t r = 0; r < xv.rows; ++r) {
      const int64_t end = causal ? r % n + 1 : n;
      double mx = -INFINITY, z = 0;
      for (int64_t c = 0; c < end; ++c) mx = std::max(mx, double(xv.get(r, c) * scale));
      for (int64_t c = 0; c < end; ++c) z += std::exp(double(xv.get(r, c) * scale) - mx);
      for (int64_t c = 0; c < n; ++c) o.set(r, c, c < end ? float(std::exp(double(xv.get(r, c) * scale) - mx) / z) : 0.0f);
    }
  });
}

train_checked check_softmax_backward(const sizer& bytes, const larch_tensor* p, const larch_tensor* dp,
                                     const larch_tensor* ds) {
  const train_checked a{check_tensor(bytes, p, "p"), check_tensor(bytes, dp, "dp"), check_tensor(bytes, ds, "ds"), {},
                        {}};
  same_f32({&a.a, &a.b, &a.c}, "p, dp and ds must be f32 [rows, n]");
  return a;
}

larch_status softmax_backward(larch_stream*, const larch_tensor* p, const larch_tensor* dp, float scale,
                               const larch_tensor* ds) {
  return run("softmax_backward", [&] {
    const train_checked a = check_softmax_backward(buffer_bytes, p, dp, ds);
    const view pv = a.a, g = a.b, o = a.c;
    for (int64_t r = 0; r < pv.rows; ++r) {
      double s = 0;
      for (int64_t c = 0; c < pv.cols; ++c) s += double(pv.get(r, c)) * g.get(r, c);
      for (int64_t c = 0; c < pv.cols; ++c) o.set(r, c, float(double(scale) * pv.get(r, c) * (g.get(r, c) - s)));
    }
  });
}

}  // namespace cpu
