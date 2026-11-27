// CPU reference of larch.h's Inkling 2 entries (ABI 6), in llama.cpp's math (src/models/inkling2.cpp,
// inkling2-loop.cpp and build_moe_ffn's sigmoid gating and SITU): the f32 steps in ggml's order, sums and dot
// products in double. attn_rel is in ops.cpp, beside attn.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include "cpu/cpu.h"

namespace cpu {

static bool f16_or_f32(const meta& m) { return m.dtype == LARCH_F16 || m.dtype == LARCH_F32; }

short_conv_checked check_short_conv(const sizer& bytes, const larch_tensor* x, const larch_tensor* w,
                                    const larch_tensor* dyn, const larch_tensor* residual,
                                    const larch_conv_state* st, uint32_t offset, const larch_tensor* slots,
                                    const larch_tensor* q_lens, const larch_tensor* snaps, const larch_tensor* out) {
  need(st != nullptr && st->conv, "state or its buffer is null");
  need(st->n_slots && st->n_layers && st->layer < st->n_layers && st->kernel >= 2 && st->channels,
       "bad state geometry");
  short_conv_checked a;
  a.state = layer_view(st->conv, bytes(st->conv, "conv state"), LARCH_F16, st->n_slots,
                       mul(st->kernel - 1, st->channels), st->layer, st->n_layers, "state buffer too small");
  a.in = check_tensor(bytes, x, "x");
  a.w = check_tensor(bytes, w, "w");
  a.out = check_tensor(bytes, out, "out");
  need(is(a.in, LARCH_F16, 2), "x must be f16 [rows, C]");
  const int64_t ch = a.in.cols;
  need(uint64_t(offset) + uint64_t(ch) <= st->channels, "offset + C must be at most the state's channels");
  need(is(a.w, LARCH_F32, 2) && a.w.rows == ch && a.w.cols == st->kernel, "w must be f32 [C, kernel]");
  if (dyn) {
    a.dyn = check_tensor(bytes, dyn, "dyn");
    need(f16_or_f32(a.dyn) && a.dyn.ndim == 3 && a.dyn.rows == a.in.rows && a.dyn.shape[1] > 0 &&
             ch % a.dyn.shape[1] == 0 && a.dyn.shape[2] == st->kernel,
         "dyn must be f16 or f32 [rows, heads, kernel] with heads dividing C");
  }
  if (residual) {
    a.residual = check_tensor(bytes, residual, "residual");
    need(is(a.residual, LARCH_F32, 2) && a.residual.rows == a.in.rows && a.residual.cols == ch,
         "residual must be f32 [rows, C]");
  }
  need(f16_or_f32(a.out) && a.out.ndim == 2 && a.out.rows == a.in.rows && a.out.cols == ch,
       "out must be f16 or f32 [rows, C]");
  check_sequences(bytes, a.in.rows, slots, q_lens, snaps, a.slots, a.lens, a.snaps);
  return a;
}

larch_status short_conv(larch_stream*, const larch_tensor* x, const larch_tensor* w, const larch_tensor* dyn,
                         const larch_tensor* residual, const larch_conv_state* state, uint32_t offset,
                         const larch_tensor* slots_i32, const larch_tensor* q_lens_i32,
                         const larch_tensor* snap_slots_or_null, const larch_tensor* out) {
  return run("short_conv", [&] {
    const short_conv_checked c = check_short_conv(buffer_bytes, x, w, dyn, residual, state, offset, slots_i32,
                                                  q_lens_i32, snap_slots_or_null, out);
    const view xv = c.in, wv = c.w, dv = c.dyn, rv = c.residual, o = c.out, st = c.state, slots = c.slots,
               snaps = c.snaps;
    const int64_t ch = xv.cols, kern = wv.cols, hist = kern - 1, all = state->channels;
    const int64_t heads = dv.buf ? dv.shape[1] : 1, per_head = ch / heads;
    const float scale = 1.0f / std::sqrt(float((dv.buf ? 2 : 1) * kern + 1));
    const std::vector<int64_t> first = sequences(c.slots, c.lens, c.snaps, xv.rows, st.rows);
    std::vector<float> in, dyn_row(static_cast<size_t>(heads * kern));  // in: [hist + len][ch], oldest first
    // The state as of input `t` (hist + row) into `slot`: inputs t - hist + 1 .. t, these channels only.
    const auto keep = [&](int64_t t, int32_t slot) {
      for (int64_t i = 0; i < hist; ++i)
        for (int64_t k = 0; k < ch; ++k)
          st.set(slot, i * all + offset + k, in[size_t((t - hist + 1 + i) * ch + k)]);
    };
    for (int64_t s = 0; s < slots.rows; ++s) {
      const int64_t r0 = first[size_t(s)], len = first[size_t(s) + 1] - r0;
      in.resize(size_t((hist + len) * ch));
      for (int64_t i = 0; i < hist; ++i)
        for (int64_t k = 0; k < ch; ++k) in[size_t(i * ch + k)] = st.get(slots.i32(s), i * all + offset + k);
      for (int64_t t = 0; t < len; ++t) xv.row_f32(r0 + t, in.data() + (hist + t) * ch);
      for (int64_t t = 0; t < len; ++t) {
        const int64_t r = r0 + t;
        if (dv.buf) dv.row_f32(r, dyn_row.data());
        for (int64_t k = 0; k < ch; ++k) {
          double sum = 0;
          for (int64_t i = 0; i < kern; ++i) sum += double(wv.get(k, i)) * in[size_t((t + i) * ch + k)];
          float y = float(sum);
          for (int64_t lag = 0; dv.buf && lag < kern; ++lag)
            y += in[size_t((t + hist - lag) * ch + k)] * dyn_row[size_t((k / per_head) * kern + lag)];
          y = (in[size_t((hist + t) * ch + k)] + y) * scale;
          o.set(r, k, rv.buf ? rv.get(r, k) + y : y);
        }
        if (snaps.buf && snaps.i32(r) >= 0) keep(hist + t, snaps.i32(r));
      }
      keep(hist + len - 1, slots.i32(s));
    }
  });
}

clip_norm_checked check_clip_norm(const sizer& bytes, const larch_tensor* x, const larch_tensor* scale, float t,
                                  const larch_tensor* pos, uint32_t floor, float alpha, const larch_tensor* out) {
  clip_norm_checked a;
  a.x = check_tensor(bytes, x, "x");
  a.out = check_tensor(bytes, out, "out");
  need(f16_or_f32(a.x) && a.x.ndim == 3, "x must be f16 or f32 [rows, heads, dim]");
  need(f16_or_f32(a.out) && a.out.ndim == 3 && std::equal(a.x.shape, a.x.shape + 3, a.out.shape),
       "out must be f16 or f32 of x's shape");
  need(std::isfinite(t) && t > 0.0f, "t must be finite and > 0");
  if (scale) {
    a.scale = check_tensor(bytes, scale, "scale");
    need(is(a.scale, LARCH_F32, 2) && a.scale.shape[0] == a.x.shape[1] && a.scale.shape[1] == a.x.shape[2],
         "scale must be f32 [heads, dim]");
  }
  if (pos) {
    a.pos = check_tensor(bytes, pos, "pos");
    need(is(a.pos, LARCH_I32, 1) && a.pos.rows == a.x.rows, "pos must be i32 [rows]");
    need(floor > 0 && std::isfinite(alpha) && alpha >= 0.0f, "floor must be > 0 and alpha finite and >= 0");
  }
  return a;
}

larch_status clip_norm(larch_stream*, const larch_tensor* x, const larch_tensor* scale, float t,
                        const larch_tensor* pos, uint32_t floor, float alpha, const larch_tensor* out) {
  return run("clip_norm", [&] {
    const clip_norm_checked c = check_clip_norm(buffer_bytes, x, scale, t, pos, floor, alpha, out);
    const view xv = c.x, sv = c.scale, pv = c.pos, o = c.out;
    const int64_t heads = c.x.shape[1], dim = c.x.shape[2];
    std::vector<float> y(static_cast<size_t>(dim));
    for (int64_t r = 0; r < xv.rows; ++r) {
      float tau = 1.0f;
      if (pv.buf) {
        float p = float(pv.i32(r)) + 1.0f;
        p = std::max(p * (1.0f / float(floor)), 1.0f);
        tau = std::log(p) * alpha + 1.0f;
      }
      for (int64_t h = 0; h < heads; ++h) {
        double sum = 0;
        for (int64_t i = 0; i < dim; ++i) {
          float v = xv.get(r, h * dim + i);
          if (sv.buf) v *= sv.get(h, i);
          y[size_t(i)] = v;
          sum += double(v * v);
        }
        const float inv = 1.0f / std::max(std::sqrt(float(sum)), t);
        for (int64_t i = 0; i < dim; ++i) {
          float v = y[size_t(i)] * inv * t;
          if (pv.buf) v *= tau;
          o.set(r, h * dim + i, v);
        }
      }
    }
  });
}

elementwise_checked check_depth_mix(const sizer& bytes, const larch_tensor* states, const larch_tensor* query,
                                    const larch_tensor* out) {
  const elementwise_checked a{check_tensor(bytes, states, "states"), check_tensor(bytes, query, "query"),
                              check_tensor(bytes, out, "out")};
  need(is(a.a, LARCH_F32, 3) && a.a.shape[0] > 0, "states must be f32 [n, rows, d] with n > 0");
  need(is(a.b, LARCH_F32, 2) && a.b.shape[1] == a.a.shape[2], "query must be f32 [q, d]");
  need(is(a.out, LARCH_F32, 3) && a.out.shape[0] == a.a.shape[1] && a.out.shape[1] == a.b.shape[0] &&
           a.out.shape[2] == a.a.shape[2],
       "out must be f32 [rows, q, d]");
  return a;
}

larch_status depth_mix(larch_stream*, const larch_tensor* states, const larch_tensor* query, const larch_tensor* out) {
  return run("depth_mix", [&] {
    const elementwise_checked c = check_depth_mix(buffer_bytes, states, query, out);
    const view sv = c.a, qv = c.b, o = c.out;
    const int64_t n = c.a.shape[0], rows = c.a.shape[1], d = c.a.shape[2], nq = c.b.shape[0];
    std::vector<float> s(static_cast<size_t>(n * d));
    std::vector<double> p(static_cast<size_t>(n));
    for (int64_t r = 0; r < rows; ++r) {
      // ggml_rms_norm with eps 1e-6: the mean in f32, then x * 1 / sqrt(mean + eps).
      for (int64_t j = 0; j < n; ++j) {
        double sum = 0;
        for (int64_t k = 0; k < d; ++k) {
          const float v = sv.get(j, r * d + k);
          sum += double(v * v);
        }
        const float mean = float(sum / double(d)), inv = 1.0f / std::sqrt(mean + 1e-6f);
        for (int64_t k = 0; k < d; ++k) s[size_t(j * d + k)] = sv.get(j, r * d + k) * inv;
      }
      for (int64_t i = 0; i < nq; ++i) {
        double mx = -INFINITY, total = 0;
        for (int64_t j = 0; j < n; ++j) {
          double dot = 0;
          for (int64_t k = 0; k < d; ++k) dot += double(s[size_t(j * d + k)]) * qv.get(i, k);
          p[size_t(j)] = dot / std::sqrt(double(d));
          mx = std::max(mx, p[size_t(j)]);
        }
        for (double& v : p) total += v = std::exp(v - mx);
        for (int64_t k = 0; k < d; ++k) {
          double acc = 0;
          for (int64_t j = 0; j < n; ++j) acc += p[size_t(j)] * s[size_t(j * d + k)];
          o.set(r, i * d + k, float(acc / total));
        }
      }
    }
  });
}

loop_gate_checked check_loop_gate(const sizer& bytes, const larch_tensor* x, const larch_tensor* msg,
                                  const larch_tensor* gain, const larch_tensor* forget, const larch_tensor* out) {
  const loop_gate_checked a{check_tensor(bytes, x, "x"), check_tensor(bytes, msg, "msg"),
                            check_tensor(bytes, gain, "gain"), check_tensor(bytes, forget, "forget"),
                            check_tensor(bytes, out, "out")};
  const auto like = [&](const meta& m, larch_dtype t) { return is(m, t, 2) && m.rows == a.x.rows && m.cols == a.x.cols; };
  need(is(a.x, LARCH_F32, 2) && like(a.msg, LARCH_F32) && like(a.out, LARCH_F32), "x, msg and out must be f32 [rows, d]");
  need(like(a.gain, LARCH_F16) && like(a.forget, LARCH_F16), "gain and forget must be f16 [rows, d]");
  return a;
}

larch_status loop_gate(larch_stream*, const larch_tensor* x, const larch_tensor* msg, const larch_tensor* gain,
                        const larch_tensor* forget, const larch_tensor* out) {
  return run("loop_gate", [&] {
    const loop_gate_checked c = check_loop_gate(buffer_bytes, x, msg, gain, forget, out);
    const view xv = c.x, mv = c.msg, gv = c.gain, fv = c.forget, o = c.out;
    for (int64_t r = 0; r < xv.rows; ++r)
      for (int64_t k = 0; k < xv.cols; ++k) {
        const float xi = xv.get(r, k), kept = xi - std::tanh(fv.get(r, k)) * xi;
        o.set(r, k, kept + std::tanh(gv.get(r, k)) * mv.get(r, k));
      }
  });
}

situ_checked check_situ_mul(const sizer& bytes, const larch_tensor* gate_up, const larch_tensor* ids,
                            const larch_tensor* scale, float beta, float lin_beta, const larch_tensor* out) {
  situ_checked a;
  a.gate_up = check_tensor(bytes, gate_up, "gate_up");
  a.out = check_tensor(bytes, out, "out");
  need(f16_or_f32(a.gate_up) && a.gate_up.ndim == 3 && a.gate_up.shape[2] % 2 == 0,
       "gate_up must be f16 or f32 [rows, slots, 2 ff]");
  need(is(a.out, LARCH_F16, 3) && a.out.shape[0] == a.gate_up.shape[0] && a.out.shape[1] == a.gate_up.shape[1] &&
           a.out.shape[2] * 2 == a.gate_up.shape[2],
       "out must be f16 [rows, slots, ff]");
  need(std::isfinite(beta) && beta > 0.0f && std::isfinite(lin_beta), "beta must be finite and > 0");
  a.ids = check_tensor(bytes, ids, "ids");
  need(is(a.ids, LARCH_I32, 2) && a.ids.shape[0] == a.gate_up.shape[0] && a.ids.shape[1] == a.gate_up.shape[1],
       "ids must be i32 [rows, slots]");
  if (scale) {
    a.scale = check_tensor(bytes, scale, "scale");
    need(is(a.scale, LARCH_F32, 1), "scale must be f32 [experts]");
  }
  return a;
}

larch_status situ_mul(larch_stream*, const larch_tensor* gate_up, const larch_tensor* ids, const larch_tensor* scale,
                       float beta, float lin_beta, const larch_tensor* out) {
  return run("situ_mul", [&] {
    const situ_checked c = check_situ_mul(buffer_bytes, gate_up, ids, scale, beta, lin_beta, out);
    const view gv = c.gate_up, iv = c.ids, sv = c.scale, o = c.out;
    const int64_t rows = c.gate_up.shape[0], slots = c.gate_up.shape[1], ff = c.out.shape[2];
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t j = 0; j < slots; ++j) {
        float s = 1.0f;
        if (sv.buf) {
          const int32_t e = iv.i32(r, j);
          need(e >= 0 && e < sv.rows, "expert id out of range");
          s = sv.get(e, 0);
        }
        for (int64_t k = 0; k < ff; ++k) {
          float g = gv.get(r, j * 2 * ff + k), u = gv.get(r, j * 2 * ff + ff + k);
          if (sv.buf) {
            g *= s;
            u *= s;
          }
          const float act = beta * std::tanh(g * (1.0f / beta)) * (1.0f / (1.0f + std::exp(-g)));
          const float up = lin_beta > 0.0f ? lin_beta * std::tanh(u * (1.0f / lin_beta)) : u;
          o.set(r, j * ff + k, act * up);
        }
      }
  });
}

route_sigmoid_checked check_moe_route_sigmoid(const sizer& bytes, const larch_tensor* logits, const larch_tensor* bias,
                                              float eps, float scale, const larch_tensor* down_scale,
                                              const larch_tensor* ids, const larch_tensor* weights) {
  route_sigmoid_checked a;
  a.logits = check_tensor(bytes, logits, "logits");
  a.ids = check_tensor(bytes, ids, "ids");
  a.weights = check_tensor(bytes, weights, "weights");
  need(is(a.logits, LARCH_F32, 2) && a.logits.cols > 0, "logits must be f32 [rows, experts]");
  need(is(a.ids, LARCH_I32, 2) && a.ids.rows == a.logits.rows && a.ids.cols > 0 && a.ids.cols <= a.logits.cols,
       "ids must be i32 [rows, k], 0 < k <= experts");
  need(is(a.weights, LARCH_F32, 2) && a.weights.rows == a.ids.rows && a.weights.cols == a.ids.cols,
       "weights must be f32 [rows, k]");
  need(std::isfinite(eps) && eps >= 0.0f && std::isfinite(scale), "eps must be finite and >= 0, scale finite");
  for (auto [t, m, what] : {std::tuple{bias, &a.bias, "bias"}, std::tuple{down_scale, &a.down_scale, "down_scale"}})
    if (t) {
      *m = check_tensor(bytes, t, what);
      need(is(*m, LARCH_F32, 1) && m->rows == a.logits.cols, "bias and down_scale must be f32 [experts]");
    }
  return a;
}

larch_status moe_route_sigmoid(larch_stream*, const larch_tensor* logits, const larch_tensor* bias, float eps,
                                float scale, const larch_tensor* down_scale, const larch_tensor* ids,
                                const larch_tensor* weights) {
  return run("moe_route_sigmoid", [&] {
    const route_sigmoid_checked c =
        check_moe_route_sigmoid(buffer_bytes, logits, bias, eps, scale, down_scale, ids, weights);
    const view lv = c.logits, bv = c.bias, dv = c.down_scale, iv = c.ids, wv = c.weights;
    const int64_t n = lv.cols, k = iv.cols;
    std::vector<float> p(static_cast<size_t>(n)), sel(static_cast<size_t>(n));
    std::vector<int32_t> order(static_cast<size_t>(n));
    for (int64_t r = 0; r < lv.rows; ++r) {
      for (int64_t e = 0; e < n; ++e) {
        p[size_t(e)] = 1.0f / (1.0f + std::exp(-lv.get(r, e)));
        sel[size_t(e)] = bv.buf ? p[size_t(e)] + bv.get(e, 0) : p[size_t(e)];
      }
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int32_t a, int32_t b) {
        return sel[size_t(a)] > sel[size_t(b)] || (sel[size_t(a)] == sel[size_t(b)] && a < b);
      });
      double sum = 0;
      for (int64_t j = 0; j < k; ++j) sum += p[size_t(order[size_t(j)])];
      const float denom = float(sum) + eps, mult = scale == 0.0f ? 1.0f : scale;
      for (int64_t j = 0; j < k; ++j) {
        const int32_t e = order[size_t(j)];
        float w = p[size_t(e)] / denom * mult;
        if (dv.buf) w *= dv.get(e, 0);
        iv.set_i32(r, j, e);
        wv.set(r, j, w);
      }
    }
  });
}

elementwise_checked check_softcap(const sizer& bytes, const larch_tensor* x, float cap, const larch_tensor* out) {
  const elementwise_checked a{check_tensor(bytes, x, "x"), {}, check_tensor(bytes, out, "out")};
  need(is(a.a, LARCH_F32, 2) && is(a.out, LARCH_F32, 2) && a.out.rows == a.a.rows && a.out.cols == a.a.cols,
       "x and out must be f32 [rows, n]");
  need(std::isfinite(cap) && cap > 0.0f, "cap must be finite and > 0");
  return a;
}

larch_status softcap(larch_stream*, const larch_tensor* x, float cap, const larch_tensor* out) {
  return run("softcap", [&] {
    const elementwise_checked c = check_softcap(buffer_bytes, x, cap, out);
    const view xv = c.a, o = c.out;
    for (int64_t r = 0; r < xv.rows; ++r)
      for (int64_t k = 0; k < xv.cols; ++k) o.set(r, k, cap * std::tanh(xv.get(r, k) * (1.0f / cap)));
  });
}

}  // namespace cpu
