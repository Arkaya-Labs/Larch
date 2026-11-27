// CPU reference of larch.h's Gated DeltaNet ops (Qwen 3.5's linear attention), in llama.cpp's math:
// ggml_ssm_conv + silu for gdn_conv; build_gdn_l2_norm, softplus, sigmoid and the autoregressive delta
// rule (ggml_compute_forward_gated_delta_net) for gdn. Per-token math in f32 where llama.cpp's order
// decides the value, sums in double, and the state rounded to its own dtype after every token.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "cpu/cpu.h"

namespace cpu {

// The layer's part of state buffer b as [n_slots, per_slot] of dtype t.
static meta state_layer(const sizer& bytes, const larch_gdn_state* st, const larch_buffer* b, larch_dtype t,
                        int64_t per_slot, const char* what) {
  return layer_view(b, bytes(b, what), t, st->n_slots, per_slot, st->layer, st->n_layers, "state buffer too small");
}

// The geometry and the channel count (2 k_heads + v_heads) head_dim.
static int64_t check_geometry(const larch_gdn_state* st) {
  need(st != nullptr && st->conv && st->ssm, "state or one of its buffers is null");
  need(st->n_slots && st->n_layers && st->layer < st->n_layers && st->kernel >= 2 && st->k_heads && st->v_heads &&
           st->head_dim && st->v_heads % st->k_heads == 0,
       "bad state geometry");
  return mul(2 * int64_t(st->k_heads) + st->v_heads, st->head_dim);
}

void check_sequences(const sizer& bytes, int64_t rows, const larch_tensor* slots, const larch_tensor* q_lens,
                     const larch_tensor* snaps, meta& slots_m, meta& lens_m, meta& snaps_m) {
  slots_m = check_tensor(bytes, slots, "slots");
  lens_m = check_tensor(bytes, q_lens, "q_lens");
  need(is(slots_m, LARCH_I32, 1) && is(lens_m, LARCH_I32, 1) && lens_m.rows == slots_m.rows,
       "slots and q_lens must be i32 [seqs]");
  // The sum itself is in device memory; no sequences sum to 0 whatever it holds.
  need(slots_m.rows || !rows, "rows must equal the sum of q_lens");
  if (snaps) {
    snaps_m = check_tensor(bytes, snaps, "snap_slots");
    need(is(snaps_m, LARCH_I32, 1) && snaps_m.rows == rows, "snap_slots must be i32 [rows]");
  }
}

// The sequence tensors both ops share, for their input's rows.
static void check_sequences(const sizer& bytes, gdn_checked& a, const larch_tensor* slots, const larch_tensor* q_lens,
                            const larch_tensor* snaps) {
  check_sequences(bytes, a.in.rows, slots, q_lens, snaps, a.slots, a.lens, a.snaps);
}

gdn_checked check_gdn_conv(const sizer& bytes, const larch_tensor* x, const larch_tensor* w,
                           const larch_gdn_state* state, const larch_tensor* slots, const larch_tensor* q_lens,
                           const larch_tensor* snaps, const larch_tensor* out) {
  const int64_t ch = check_geometry(state);
  gdn_checked a;
  a.state = state_layer(bytes, state, state->conv, LARCH_F16, mul(state->kernel - 1, ch), "conv state");
  a.in = check_tensor(bytes, x, "x");
  a.w = check_tensor(bytes, w, "w");
  a.out = check_tensor(bytes, out, "out");
  need(is(a.in, LARCH_F16, 2) && a.in.cols == ch, "x must be f16 [rows, channels]");
  need(is(a.w, LARCH_F32, 2) && a.w.rows == ch && a.w.cols == state->kernel, "w must be f32 [channels, kernel]");
  need(is(a.out, LARCH_F16, 2) && a.out.rows == a.in.rows && a.out.cols == ch, "out must be f16 [rows, channels]");
  check_sequences(bytes, a, slots, q_lens, snaps);
  return a;
}

gdn_checked check_gdn(const sizer& bytes, const larch_tensor* qkv, const larch_tensor* alpha,
                      const larch_tensor* beta, const larch_tensor* dt_bias, const larch_tensor* a_log, float eps,
                      const larch_gdn_state* state, const larch_tensor* slots, const larch_tensor* q_lens,
                      const larch_tensor* snaps, const larch_tensor* out) {
  const int64_t ch = check_geometry(state), vh = state->v_heads, hd = state->head_dim;
  gdn_checked a;
  a.state = state_layer(bytes, state, state->ssm, LARCH_F32, mul(vh, mul(hd, hd)), "ssm state");
  a.in = check_tensor(bytes, qkv, "qkv");
  a.alpha = check_tensor(bytes, alpha, "alpha");
  a.beta = check_tensor(bytes, beta, "beta");
  a.dt_bias = check_tensor(bytes, dt_bias, "dt_bias");
  a.a = check_tensor(bytes, a_log, "a");
  a.out = check_tensor(bytes, out, "out");
  need(eps >= 0.0f, "eps must be >= 0");
  need(is(a.in, LARCH_F16, 2) && a.in.cols == ch, "qkv must be f16 [rows, channels]");
  for (const meta* m : {&a.alpha, &a.beta})
    need(is(*m, LARCH_F32, 2) && m->rows == a.in.rows && m->cols == vh, "alpha and beta must be f32 [rows, v_heads]");
  for (const meta* m : {&a.dt_bias, &a.a})
    need(is(*m, LARCH_F32, 1) && m->rows == vh, "dt_bias and a must be f32 [v_heads]");
  need(is(a.out, LARCH_F16, 2) && a.out.rows == a.in.rows && a.out.cols == vh * hd,
       "out must be f16 [rows, v_heads * head_dim]");
  check_sequences(bytes, a, slots, q_lens, snaps);
  return a;
}

std::vector<int64_t> sequences(const meta& slots_m, const meta& lens_m, const meta& snaps_m, int64_t rows,
                               int64_t n_slots) {
  const view slots = slots_m, lens = lens_m, snaps = snaps_m;
  std::vector<bool> named(static_cast<size_t>(n_slots));
  const auto name = [&](int32_t s) {
    need(s >= 0 && s < n_slots, "slot out of range");
    need(!named[size_t(s)], "a slot named twice in one call");
    named[size_t(s)] = true;
  };
  std::vector<int64_t> first{0};
  for (int64_t s = 0; s < slots.rows; ++s) {
    need(lens.i32(s) >= 0, "negative q_len");
    name(slots.i32(s));
    first.push_back(first.back() + lens.i32(s));
  }
  need(first.back() == rows, "rows must equal the sum of q_lens");
  for (int64_t r = 0; snaps.buf && r < snaps.rows; ++r)
    if (snaps.i32(r) >= 0) name(snaps.i32(r));
  return first;
}

// Step 1: per channel, the conv inputs are the slot's last kernel - 1 inputs then the sequence's rows.
larch_status gdn_conv(larch_stream*, const larch_tensor* x, const larch_tensor* w, const larch_gdn_state* state,
                       const larch_tensor* slots_i32, const larch_tensor* q_lens_i32,
                       const larch_tensor* snap_slots_or_null, const larch_tensor* out) {
  return run("gdn_conv", [&] {
    const gdn_checked c = check_gdn_conv(buffer_bytes, x, w, state, slots_i32, q_lens_i32, snap_slots_or_null, out);
    const view xv = c.in, wv = c.w, o = c.out, st = c.state, slots = c.slots, snaps = c.snaps;
    const int64_t ch = xv.cols, kern = wv.cols, hist = kern - 1;
    const std::vector<int64_t> first = sequences(c.slots, c.lens, c.snaps, c.in.rows, st.rows);
    std::vector<float> in;  // [hist + len][ch], oldest first
    for (int64_t s = 0; s < slots.rows; ++s) {
      const int64_t r0 = first[size_t(s)], len = first[size_t(s) + 1] - r0;
      in.resize(size_t((hist + len) * ch));
      st.row_f32(slots.i32(s), in.data());
      for (int64_t t = 0; t < len; ++t) xv.row_f32(r0 + t, in.data() + (hist + t) * ch);
      // The state as of row t: inputs t + 1 .. t + hist (the sequence's row t being input hist + t).
      const auto keep = [&](int64_t t, int32_t slot) {
        kv_from_f32(LARCH_F16, in.data() + (t + 1) * ch, hist * ch, st.row(slot));
      };
      for (int64_t t = 0; t < len; ++t) {
        for (int64_t k = 0; k < ch; ++k) {
          double sum = 0;
          for (int64_t i = 0; i < kern; ++i) sum += double(wv.get(k, i)) * in[size_t((t + i) * ch + k)];
          const float v = float(sum);
          o.set(r0 + t, k, v / (1.0f + expf(-v)));
        }
        if (snaps.buf && snaps.i32(r0 + t) >= 0) keep(t, snaps.i32(r0 + t));
      }
      keep(len - 1, slots.i32(s));
    }
  });
}

// Step 2: the delta rule per v head, over M as larch_gdn_state stores it (row j gives output j).
larch_status gdn(larch_stream*, const larch_tensor* qkv, const larch_tensor* alpha, const larch_tensor* beta,
                  const larch_tensor* dt_bias, const larch_tensor* a_log, float eps, const larch_gdn_state* state,
                  const larch_tensor* slots_i32, const larch_tensor* q_lens_i32,
                  const larch_tensor* snap_slots_or_null, const larch_tensor* out) {
  return run("gdn", [&] {
    const gdn_checked c = check_gdn(buffer_bytes, qkv, alpha, beta, dt_bias, a_log, eps, state, slots_i32, q_lens_i32,
                                    snap_slots_or_null, out);
    const view in = c.in, al = c.alpha, be = c.beta, dt = c.dt_bias, av = c.a, o = c.out, st = c.state,
               slots = c.slots, snaps = c.snaps;
    const int64_t kh = state->k_heads, vh = state->v_heads, hd = state->head_dim, mm = hd * hd;
    const std::vector<int64_t> first = sequences(c.slots, c.lens, c.snaps, c.in.rows, st.rows);
    const float scale = 1.0f / sqrtf(float(hd));
    std::vector<float> m(static_cast<size_t>(vh * mm)), row(static_cast<size_t>(in.cols)), q(static_cast<size_t>(hd)),
        k(static_cast<size_t>(hd));
    std::vector<double> d(static_cast<size_t>(hd));
    // q or k head h of row, times 1 / sqrt(its sum of squares + eps) (build_gdn_l2_norm).
    const auto l2 = [&](int64_t at, std::vector<float>& dst) {
      double ss = 0;
      for (int64_t i = 0; i < hd; ++i) ss += double(row[size_t(at + i)]) * row[size_t(at + i)];
      const float inv = float(1.0 / std::sqrt(ss + eps));
      for (int64_t i = 0; i < hd; ++i) dst[size_t(i)] = row[size_t(at + i)] * inv;
    };
    for (int64_t s = 0; s < slots.rows; ++s) {
      std::memcpy(m.data(), st.row(slots.i32(s)), m.size() * 4);
      for (int64_t r = first[size_t(s)]; r < first[size_t(s) + 1]; ++r) {
        in.row_f32(r, row.data());
        for (int64_t h = 0; h < vh; ++h) {
          l2(h % kh * hd, q);
          l2((kh + h % kh) * hd, k);
          const float* v = row.data() + (2 * kh + h) * hd;
          const float x = al.get(r, h) + dt.get(h, 0);
          const float g = expf((x > 20.0f ? x : logf(1.0f + expf(x))) * av.get(h, 0));
          const float b = 1.0f / (1.0f + expf(-be.get(r, h)));
          float* mh = m.data() + h * mm;
          for (int64_t i = 0; i < mm; ++i) mh[i] *= g;
          for (int64_t j = 0; j < hd; ++j) {
            double mk = 0;
            for (int64_t i = 0; i < hd; ++i) mk += double(mh[j * hd + i]) * k[size_t(i)];
            d[size_t(j)] = (v[j] - mk) * b;
          }
          for (int64_t j = 0; j < hd; ++j) {
            double mq = 0;
            for (int64_t i = 0; i < hd; ++i) {
              mh[j * hd + i] = float(mh[j * hd + i] + d[size_t(j)] * k[size_t(i)]);
              mq += double(mh[j * hd + i]) * q[size_t(i)];
            }
            o.set(r, h * hd + j, float(mq) * scale);
          }
        }
        if (snaps.buf && snaps.i32(r) >= 0) std::memcpy(st.row(snaps.i32(r)), m.data(), m.size() * 4);
      }
      std::memcpy(st.row(slots.i32(s)), m.data(), m.size() * 4);
    }
  });
}

}  // namespace cpu
