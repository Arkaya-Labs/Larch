// The fused entries on the CPU devices (larch.h, ABI 4): each runs its sequence of ops through the stream's own
// table, so a CPU device's fused bytes are its unfused bytes by construction; intermediates live in host memory
// that table's ops take as their own buffers. The checks are every backend's (dispatch.cpp runs them for a
// backend without the entry).
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <memory>

#include "cpu/cpu.h"

namespace cpu {

fused_gate_up_checked check_fused_moe_gate_up(const sizer& bytes, const larch_tensor* gate_w,
                                              const larch_tensor* up_w, const larch_tensor* x,
                                              const larch_tensor* ids, const larch_tensor* out) {
  fused_gate_up_checked c{check_mmvq_id(bytes, gate_w, x, ids, nullptr, nullptr, out),
                          check_mmvq_id(bytes, up_w, x, ids, nullptr, nullptr, out)};
  need(c.gate.out.dtype == LARCH_F16, "out must be f16 [rows, slots, N]");
  return c;
}

fused_down_checked check_fused_moe_down(const sizer& bytes, const larch_tensor* down_w, const larch_tensor* act,
                                        const larch_tensor* ids, const larch_tensor* weights,
                                        const larch_tensor* shared_w, const larch_tensor* shared_act,
                                        const larch_tensor* shared_gate, const larch_tensor* x) {
  fused_down_checked c;
  c.x = check_tensor(bytes, x, "x");
  need(is(c.x, LARCH_F32, 2), "x must be f32 [rows, d]");
  need(!shared_w == !shared_act && !shared_w == !shared_gate, "shared_w, shared_act and shared_gate go together");
  if (shared_w) {
    c.shared = check_mmvq(bytes, shared_w, shared_act, nullptr, x);
    c.shared_gate = check_tensor(bytes, shared_gate, "shared_gate");
    need(is(c.shared_gate, LARCH_F16, 2) && c.shared_gate.rows == c.x.rows && c.shared_gate.cols == 1,
         "shared_gate must be f16 [rows, 1]");
  }
  need(weights != nullptr, "weights is null");
  c.down = check_mmvq_id(bytes, down_w, act, ids, nullptr, weights, x);
  return c;
}

fused_moe_in_checked check_fused_moe_in(const sizer& bytes, const larch_tensor* x, const larch_tensor* norm_w,
                                        const larch_tensor* h, const larch_tensor* router,
                                        const larch_tensor* logits, const larch_tensor* ids,
                                        const larch_tensor* weights, const larch_tensor* shared_gate_w,
                                        const larch_tensor* shared_up_w, const larch_tensor* shared_act,
                                        const larch_tensor* gate_inp, const larch_tensor* shared_gate) {
  fused_moe_in_checked c;
  c.norm = check_rmsnorm(bytes, x, norm_w, h);
  c.router = check_gemm(bytes, h, router, nullptr, logits);
  need(c.router.c.dtype == LARCH_F32, "logits must be f32 [rows, experts]");
  c.route = check_moe_route(bytes, logits, 0.0f, ids, weights);
  const bool sh = shared_gate_w;
  need(sh == bool(shared_up_w) && sh == bool(shared_act) && sh == bool(gate_inp) && sh == bool(shared_gate),
       "the shared expert's five arguments go together");
  if (sh) {
    c.shared_gate = check_mmvq(bytes, shared_gate_w, h, nullptr, shared_act);
    c.shared_up = check_mmvq(bytes, shared_up_w, h, nullptr, shared_act);
    need(c.shared_gate.c.dtype == LARCH_F16, "shared_act must be f16 [rows, shared ffn]");
    c.gate_inp = check_gemm(bytes, h, gate_inp, nullptr, shared_gate);
    need(c.gate_inp.c.dtype == LARCH_F16 && c.gate_inp.c.cols == 1, "shared_gate must be f16 [rows, 1]");
  }
  return c;
}

fused_gdn_in_checked check_fused_gdn_in(const sizer& bytes, const larch_tensor* x, const larch_tensor* norm_w,
                                        const larch_tensor* h, const larch_tensor* qkv_w, const larch_tensor* z_w,
                                        const larch_tensor* alpha_w, const larch_tensor* beta_w,
                                        const larch_tensor* qkv, const larch_tensor* z, const larch_tensor* alpha,
                                        const larch_tensor* beta) {
  fused_gdn_in_checked c;
  c.norm = check_rmsnorm(bytes, x, norm_w, h);
  c.proj[0] = check_mmvq(bytes, qkv_w, h, nullptr, qkv);
  c.proj[1] = check_mmvq(bytes, z_w, h, nullptr, z);
  need(c.proj[0].c.dtype == LARCH_F16 && c.proj[1].c.dtype == LARCH_F16, "qkv and z must be f16");
  const larch_tensor* ws[2] = {alpha_w, beta_w};
  const larch_tensor* os[2] = {alpha, beta};
  for (int i = 0; i < 2; ++i) {
    // gemm's checks for an f16 weight, mmvq's for any other (they refuse a null one alike)
    c.proj[2 + i] = ws[i] && ws[i]->dtype == LARCH_F16 ? check_gemm(bytes, h, ws[i], nullptr, os[i])
                                                        : check_mmvq(bytes, ws[i], h, nullptr, os[i]);
    need(c.proj[2 + i].c.dtype == LARCH_F32, "alpha and beta must be f32 [rows, v_heads]");
  }
  return c;
}

fused_proj_add_checked check_fused_proj_add(const sizer& bytes, const larch_tensor* w, const larch_tensor* in,
                                            const larch_tensor* gate, const larch_tensor* x) {
  fused_proj_add_checked c;
  if (gate) c.gate = check_sigmoid_mul(bytes, in, gate, in);
  c.proj = check_mmvq(bytes, w, in, nullptr, x);
  need(c.proj.c.dtype == LARCH_F32, "x must be f32 [rows, n]");
  return c;
}

fused_attn_in_checked check_fused_attn_in(const sizer& bytes, const larch_tensor* q_in, const larch_tensor* k,
                                          const larch_tensor* v, const larch_tensor* q_norm,
                                          const larch_tensor* k_norm, const larch_tensor* pos,
                                          const larch_rope_params* p, const larch_tensor* slots,
                                          const larch_kv_pool* pool, const larch_tensor* q,
                                          const larch_tensor* gate) {
  fused_attn_in_checked c;
  c.q_in = check_tensor(bytes, q_in, "q_in");
  const meta qo = check_tensor(bytes, q, "q");
  need(is(qo, LARCH_F16, 3), "q must be f16 [rows, heads, head_dim]");
  if (gate) {
    c.gate = check_tensor(bytes, gate, "gate");
    need(is(c.q_in, LARCH_F16, 3) && c.q_in.rows == qo.rows && c.q_in.shape[1] == qo.shape[1] &&
             c.q_in.shape[2] == 2 * qo.shape[2],
         "q_in must be f16 [rows, heads, 2 head_dim] with a gate");
    need(is(c.gate, LARCH_F16, 3) && std::equal(qo.shape, qo.shape + 3, c.gate.shape), "gate must be q's shape");
  } else {
    need(q_in->buf == q->buf && q_in->offset == q->offset && std::equal(q->shape, q->shape + 3, q_in->shape) &&
             std::equal(q->stride, q->stride + 3, q_in->stride),
         "q must be q_in itself without a gate");
  }
  const meta kt = check_tensor(bytes, k, "k");
  need(kt.ndim == 3 && qo.shape[2] == kt.shape[2], "q and k heads must be one head_dim wide");
  need(!q_norm == !k_norm, "q_norm and k_norm go together");
  if (q_norm) {
    c.q_norm = check_qk_norm(bytes, q, q_norm);
    c.k_norm = check_qk_norm(bytes, k, k_norm);
  }
  c.q_rope = check_rope(bytes, q, pos, p);
  c.k_rope = check_rope(bytes, k, pos, p);
  c.kv = check_kv_write(bytes, k, v, slots, pool);
  return c;
}

fused_gdn_checked check_fused_gdn(const sizer& bytes, const larch_tensor* qkv, const larch_tensor* conv_w,
                                  const larch_tensor* alpha, const larch_tensor* beta, const larch_tensor* dt_bias,
                                  const larch_tensor* a, float eps, const larch_gdn_state* state,
                                  const larch_tensor* slots, const larch_tensor* q_lens, const larch_tensor* snaps,
                                  const larch_tensor* norm_w, const larch_tensor* z, const larch_tensor* conv,
                                  const larch_tensor* out) {
  fused_gdn_checked c;
  c.conv = check_gdn_conv(bytes, qkv, conv_w, state, slots, q_lens, snaps, conv);
  c.gdn = check_gdn(bytes, conv, alpha, beta, dt_bias, a, eps, state, slots, q_lens, snaps, out);
  c.norm_w = check_tensor(bytes, norm_w, "norm_w");
  need(is(c.norm_w, LARCH_F32, 1) && c.norm_w.rows == state->head_dim, "norm_w must be f32 [head_dim]");
  c.z = check_silu_mul(bytes, z, out, out);
  return c;
}

// Host memory of n bytes the stream's own ops take as one of their buffers.
struct temp : larch_buffer {
  std::unique_ptr<uint8_t[]> mem;
  temp(const larch_stream* s, uint64_t n) : mem(new uint8_t[n]()) {
    ops = s->ops;
    bytes = n;
    host = mem.get();
  }
  // A dense tensor of t over the whole buffer.
  larch_tensor at(larch_dtype t, std::initializer_list<int64_t> shape) {
    larch_tensor v = {this, 0, t, uint32_t(shape.size()), {}, {}};
    std::copy(shape.begin(), shape.end(), v.shape);
    for (int64_t d = int64_t(shape.size()) - 1, stride = 1; d >= 0; stride *= v.shape[d], --d) v.stride[d] = stride;
    return v;
  }
};

// Returns a step's status unless it is OK.
#define VV_STEP(call)                       \
  do {                                      \
    const larch_status st_ = (call);       \
    if (st_ != LARCH_OK) return st_;       \
  } while (0)

larch_status fused_moe_in(larch_stream* s, const larch_tensor* x, const larch_tensor* norm_w, float eps,
                           const larch_tensor* h, const larch_tensor* router, const larch_tensor* logits,
                           const larch_tensor* ids, const larch_tensor* weights, const larch_tensor* shared_gate_w,
                           const larch_tensor* shared_up_w, const larch_tensor* shared_act,
                           const larch_tensor* gate_inp, const larch_tensor* shared_gate) {
  fused_moe_in_checked c;
  VV_STEP(run("fused_moe_in", [&] {
    c = check_fused_moe_in(buffer_bytes, x, norm_w, h, router, logits, ids, weights, shared_gate_w, shared_up_w,
                           shared_act, gate_inp, shared_gate);
  }));
  VV_STEP(s->ops->rmsnorm(s, x, norm_w, eps, h));
  VV_STEP(s->ops->gemm(s, h, router, nullptr, logits, 1.0f, 0.0f));
  VV_STEP(s->ops->moe_route(s, logits, 0.0f, ids, weights));
  if (!shared_gate_w) return LARCH_OK;
  const int64_t rows = c.shared_gate.c.rows, sf = c.shared_gate.c.cols;
  temp g(s, uint64_t(rows * sf) * 2), u(s, uint64_t(rows * sf) * 2);
  const larch_tensor gt = g.at(LARCH_F16, {rows, sf}), ut = u.at(LARCH_F16, {rows, sf});
  VV_STEP(s->ops->mmvq(s, shared_gate_w, h, nullptr, &gt));
  VV_STEP(s->ops->mmvq(s, shared_up_w, h, nullptr, &ut));
  VV_STEP(s->ops->silu_mul(s, &gt, &ut, shared_act));
  return s->ops->gemm(s, h, gate_inp, nullptr, shared_gate, 1.0f, 0.0f);
}

larch_status fused_gdn_in(larch_stream* s, const larch_tensor* x, const larch_tensor* norm_w, float eps,
                           const larch_tensor* h, const larch_tensor* qkv_w, const larch_tensor* z_w,
                           const larch_tensor* alpha_w, const larch_tensor* beta_w, const larch_tensor* qkv,
                           const larch_tensor* z, const larch_tensor* alpha, const larch_tensor* beta) {
  VV_STEP(run("fused_gdn_in", [&] {
    check_fused_gdn_in(buffer_bytes, x, norm_w, h, qkv_w, z_w, alpha_w, beta_w, qkv, z, alpha, beta);
  }));
  VV_STEP(s->ops->rmsnorm(s, x, norm_w, eps, h));
  const larch_tensor* ws[4] = {qkv_w, z_w, alpha_w, beta_w};
  const larch_tensor* os[4] = {qkv, z, alpha, beta};
  for (int i = 0; i < 4; ++i)
    VV_STEP(ws[i]->dtype == LARCH_F16 ? s->ops->gemm(s, h, ws[i], nullptr, os[i], 1.0f, 0.0f)
                                       : s->ops->mmvq(s, ws[i], h, nullptr, os[i]));
  return LARCH_OK;
}

larch_status fused_proj_add(larch_stream* s, const larch_tensor* w, const larch_tensor* in,
                             const larch_tensor* gate, const larch_tensor* x) {
  fused_proj_add_checked c;
  VV_STEP(run("fused_proj_add", [&] { c = check_fused_proj_add(buffer_bytes, w, in, gate, x); }));
  const int64_t rows = c.proj.a.rows, k = c.proj.a.cols, n = c.proj.c.cols;
  temp g(s, gate ? uint64_t(rows * k) * 2 : 1), t(s, uint64_t(rows * n) * 2);
  const larch_tensor gt = g.at(LARCH_F16, {rows, k}), tt = t.at(LARCH_F16, {rows, n});
  if (gate) VV_STEP(s->ops->sigmoid_mul(s, in, gate, &gt));
  VV_STEP(s->ops->mmvq(s, w, gate ? &gt : in, nullptr, &tt));
  return s->ops->add(s, x, &tt, x);
}

larch_status fused_gdn(larch_stream* s, const larch_tensor* qkv, const larch_tensor* conv_w, const larch_tensor* alpha,
                        const larch_tensor* beta, const larch_tensor* dt_bias, const larch_tensor* a, float eps,
                        const larch_gdn_state* state, const larch_tensor* slots, const larch_tensor* q_lens,
                        const larch_tensor* snaps, const larch_tensor* norm_w, const larch_tensor* z,
                        const larch_tensor* conv, const larch_tensor* out) {
  fused_gdn_checked c;
  VV_STEP(run("fused_gdn", [&] {
    c = check_fused_gdn(buffer_bytes, qkv, conv_w, alpha, beta, dt_bias, a, eps, state, slots, q_lens, snaps, norm_w, z,
                        conv, out);
  }));
  const int64_t rows = c.z.out.rows, vh = state->v_heads, hd = state->head_dim;
  temp o(s, uint64_t(rows * vh * hd) * 2);
  const larch_tensor o2 = o.at(LARCH_F16, {rows, vh * hd}), o3 = o.at(LARCH_F16, {rows, vh, hd});
  VV_STEP(s->ops->gdn_conv(s, qkv, conv_w, state, slots, q_lens, snaps, conv));
  VV_STEP(s->ops->gdn(s, conv, alpha, beta, dt_bias, a, eps, state, slots, q_lens, snaps, &o2));
  VV_STEP(s->ops->qk_norm(s, &o3, norm_w, eps));
  return s->ops->silu_mul(s, z, &o2, out);
}

larch_status fused_attn_in(larch_stream* s, const larch_tensor* q_in, const larch_tensor* k, const larch_tensor* v,
                            const larch_tensor* q_norm, const larch_tensor* k_norm, float eps,
                            const larch_tensor* pos, const larch_rope_params* p, const larch_tensor* slots,
                            const larch_kv_pool* pool, const larch_tensor* q, const larch_tensor* gate) {
  fused_attn_in_checked c;
  VV_STEP(run("fused_attn_in", [&] {
    c = check_fused_attn_in(buffer_bytes, q_in, k, v, q_norm, k_norm, pos, p, slots, pool, q, gate);
  }));
  const int64_t rows = c.kv.k.rows, heads = c.q_rope.qk.shape[1], hd = c.q_rope.qk.shape[2];
  // Only with rows: a zero-row tensor's trailing dims are bounded by no buffer, so they must size no selector.
  if (gate && rows > 0) {  // the model's gemms by 0/1 selectors (qwen35.rs gated_split): the first half of each pair, then the second
    temp sel(s, uint64_t(2 * hd * 2 * hd) * 2);
    for (int64_t half = 0; half < 2; ++half)
      for (int64_t i = 0; i < hd; ++i) {
        const uint16_t one = 0x3c00;
        std::memcpy(sel.host + 2 * ((half * hd + i) * 2 * hd + half * hd + i), &one, 2);
      }
    // Each row's [heads, 2 head_dim] by each selector: the gemm's elements are its rows' alone.
    for (int64_t r = 0; r < rows; ++r)
      for (int i = 0; i < 2; ++i) {
        const larch_tensor* to = i ? gate : q;
        const larch_tensor pr = {q_in->buf, q_in->offset + uint64_t(r * q_in->stride[0]) * 2, LARCH_F16, 2,
                                  {heads, 2 * hd}, {2 * hd, 1}};
        const larch_tensor o = {to->buf, to->offset + uint64_t(r * to->stride[0]) * 2, LARCH_F16, 2, {heads, hd},
                                 {hd, 1}};
        larch_tensor st = sel.at(LARCH_F16, {hd, 2 * hd});
        st.offset = uint64_t(i * hd * 2 * hd) * 2;
        VV_STEP(s->ops->gemm(s, &pr, &st, nullptr, &o, 1.0f, 0.0f));
      }
  }
  temp kt(s, uint64_t(rows * c.kv.k.cols) * 2);  // k's normed and roped rows
  const larch_tensor kc = kt.at(LARCH_F16, {rows, c.kv.k.shape[1], c.kv.k.shape[2]});
  for (int64_t r = 0; r < rows; ++r) std::memcpy(kt.host + r * c.kv.k.cols * 2, view(c.kv.k).row(r), c.kv.k.cols * 2);
  if (q_norm) {
    VV_STEP(s->ops->qk_norm(s, q, q_norm, eps));
    VV_STEP(s->ops->qk_norm(s, &kc, k_norm, eps));
  }
  VV_STEP(s->ops->rope(s, q, pos, p));
  VV_STEP(s->ops->rope(s, &kc, pos, p));
  return s->ops->kv_write(s, &kc, v, slots, pool);
}

larch_status fused_moe_gate_up(larch_stream* s, const larch_tensor* gate_w, const larch_tensor* up_w,
                                const larch_tensor* x, const larch_tensor* ids, const larch_tensor* out) {
  fused_gate_up_checked c;
  VV_STEP(run("fused_moe_gate_up", [&] { c = check_fused_moe_gate_up(buffer_bytes, gate_w, up_w, x, ids, out); }));
  const int64_t rows = c.gate.out.rows, slots = c.gate.ids.cols, n = c.gate.out.shape[2];
  temp g(s, uint64_t(rows * slots * n) * 2), u(s, uint64_t(rows * slots * n) * 2);
  const larch_tensor gt = g.at(LARCH_F16, {rows, slots, n}), ut = u.at(LARCH_F16, {rows, slots, n});
  VV_STEP(s->ops->mmvq_id(s, gate_w, x, ids, nullptr, nullptr, &gt));
  VV_STEP(s->ops->mmvq_id(s, up_w, x, ids, nullptr, nullptr, &ut));
  // silu_mul is elementwise: [rows, slots, N] as [rows, slots * N], out keeping its row stride.
  larch_tensor o = *out;
  o.ndim = 2;
  o.shape[1] = slots * n;
  o.stride[1] = 1;
  const larch_tensor g2 = g.at(LARCH_F16, {rows, slots * n}), u2 = u.at(LARCH_F16, {rows, slots * n});
  return s->ops->silu_mul(s, &g2, &u2, &o);
}

larch_status fused_moe_down(larch_stream* s, const larch_tensor* down_w, const larch_tensor* act,
                             const larch_tensor* ids, const larch_tensor* weights, const larch_tensor* shared_w,
                             const larch_tensor* shared_act, const larch_tensor* shared_gate,
                             const larch_tensor* x) {
  fused_down_checked c;
  VV_STEP(run("fused_moe_down", [&] {
    c = check_fused_moe_down(buffer_bytes, down_w, act, ids, weights, shared_w, shared_act, shared_gate, x);
  }));
  temp t(s, uint64_t(c.x.rows * c.x.cols) * 2);
  const larch_tensor tt = t.at(LARCH_F16, {c.x.rows, c.x.cols});
  if (shared_w) {
    VV_STEP(s->ops->mmvq(s, shared_w, shared_act, nullptr, &tt));
    VV_STEP(s->ops->sigmoid_mul(s, &tt, shared_gate, &tt));
    VV_STEP(s->ops->add(s, x, &tt, x));
  }
  VV_STEP(s->ops->mmvq_id(s, down_w, act, ids, nullptr, weights, &tt));
  return s->ops->add(s, x, &tt, x);
}

}  // namespace cpu
