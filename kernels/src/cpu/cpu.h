// CPU reference internals (architecture 6.5): plain loops, single-threaded, synchronous. It is the
// oracle every GPU op is tested against, so it favours exactness: f32 math in llama.cpp's order where
// parity depends on it, double accumulators for dot products and softmax. Its argument checks (meta,
// sizer, check_*) are every backend's, so that all of them refuse the same calls.
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/backend.hpp"

namespace cpu {

// Its memory is the base's `host`, as a host buffer of another backend's is.
struct buffer : larch_buffer {
  std::unique_ptr<uint8_t[]> data;
};

// A table of the CPU devices: the reference (-1) or the CPU device (-2).
inline bool cpu_ops(const vv_ops* o) { return o == vv_cpu_ops() || o == vv_cpu_device_ops(); }

// The threads a CPU stream's ops may use: its device's larch_device_set_threads, else one per logical CPU.
uint32_t stream_threads(const larch_stream* s);

// b is host memory a CPU op may use: a CPU device's buffer or a host buffer (a GPU buffer can reach a
// CPU stream in a mixed process).
inline bool ours(const larch_buffer* b) { return b && b->host; }

// Thrown by argument checks; the op wrapper turns it into LARCH_ERR_INVALID_ARG.
struct bad_arg : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// a * b for sizes (both >= 0), or bad_arg where it would wrap.
inline int64_t mul(int64_t a, int64_t b) {
  if (a && b > INT64_MAX / a) throw bad_arg("tensor size overflows");
  return a * b;
}

inline float f16_to_f32(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
  if (exp == 0) {  // zero or subnormal: man * 2^-24, exact in f32
    const float f = float(man) * 0x1p-24f;
    return sign ? -f : f;
  }
  if (exp == 0x1f) return std::bit_cast<float>(sign | 0x7f800000 | (man << 13));
  return std::bit_cast<float>(sign | ((exp + 112) << 23) | (man << 13));
}

// Round to nearest even, like the hardware conversion.
inline uint16_t f32_to_f16(float f) {
  const uint32_t x = std::bit_cast<uint32_t>(f), ax = x & 0x7fffffff;
  const uint16_t sign = uint16_t((x >> 16) & 0x8000);
  if (ax > 0x7f800000) return sign | 0x7e00;            // NaN
  if (ax >= 0x477ff000) return sign | 0x7c00;           // >= 65520 rounds to inf
  if (ax < 0x38800000)                                  // below 2^-14: subnormal or zero
    return sign | uint16_t(std::nearbyint(std::bit_cast<float>(ax) * 0x1p24f));
  return sign | uint16_t((ax + 0xfff + ((ax >> 13) & 1) - 0x38000000) >> 13);
}

inline float bf16_to_f32(uint16_t h) { return std::bit_cast<float>(uint32_t(h) << 16); }
// fp8 e4m3 KV pages (OCP E4M3FN): the value of a byte, and a float rounded to the nearest byte, ties to
// even, saturating at +-448 (NaN stays NaN).
float fp8_to_f32(uint8_t b);
uint8_t f32_to_fp8(float f);
inline uint16_t u16(const uint8_t* p) { return uint16_t(p[0] | p[1] << 8); }  // little-endian, like GGUF
inline float half(const uint8_t* p) { return f16_to_f32(u16(p)); }

struct type_info {
  int64_t block;  // elements per block (1 for plain types)
  int64_t bytes;  // bytes per block
};
type_info info(larch_dtype t);  // throws bad_arg for an unknown dtype
inline bool is_quant(larch_dtype t) { return info(t).block > 1; }

// Dequantizes n elements (a multiple of the block size) of any dtype starting at src.
void to_f32(larch_dtype t, const uint8_t* src, int64_t n, float* dst);
// Rounds n elements into a KV page row of dtype f16, q8_0 (n a multiple of 32, as ggml's
// quantize_row_q8_0_ref) or fp8_e4m3.
void kv_from_f32(larch_dtype t, const float* src, int64_t n, uint8_t* dst);
// Group g (32 weights) of a quant row as quant_gemm_i8 reads it: weight j is s[j / 16] q[j] - m, with GGML's own
// scales (Q6_K: one per 16 weights) and minimum (0 for the symmetric formats).
void i8_group(larch_dtype t, const uint8_t* row, int64_t g, int8_t* q, float* s, float* m);
larch_status quant_gemm_i8(larch_stream*, const larch_tensor*, const larch_tensor*,
                            const larch_tensor*, const larch_tensor*);

// The argument checks every backend shares. They read metadata only (never device memory), in
// the order and wording of the CPU reference, which runs them too, so every backend answers a malformed
// call the same way. A backend passes its `sizer` and adds only its own refusals after them.

// A checked tensor seen as rows x cols: rows = shape[0], cols = the product of the other dims (1 for a
// vector, so element i of a vector is row i). It starts `offset` bytes into `buf` (null: an absent
// optional tensor), rows `ld` elements (blocks for quant) or `row_bytes` bytes apart.
struct meta {
  const larch_buffer* buf = nullptr;
  uint64_t offset = 0;
  larch_dtype dtype{};
  uint32_t ndim = 0;
  int64_t shape[4] = {};
  int64_t rows = 0, cols = 0, ld = 0, row_bytes = 0;
  bool aligned = true;  // offset a multiple of the element size (2 bytes for a quant block)
};

// A backend's buffer lookup: b's size in bytes, or throws bad_arg naming `what` when b is null, another
// backend's or on another device than the stream.
using sizer = std::function<uint64_t(const larch_buffer* b, const std::string& what)>;

// t checked against larch.h's layout rules and its buffer's size; `what` names it in errors.
meta check_tensor(const sizer& bytes, const larch_tensor* t, const char* what);

// Each op's checks, in the reference's order: the checked tensors, or bad_arg. An absent optional
// tensor has a null buf.
struct embed_checked {
  meta table, ids, out;
};
embed_checked check_embed(const sizer&, const larch_tensor* table, const larch_tensor* ids, const larch_tensor* out);
// rmsnorm, and qk_norm (in place, so out is qk).
struct norm_checked {
  meta x, w, out;
};
norm_checked check_rmsnorm(const sizer&, const larch_tensor* x, const larch_tensor* w, const larch_tensor* out,
                           larch_dtype out_dtype = LARCH_F16);  // f32: rmsnorm_f32
norm_checked check_qk_norm(const sizer&, const larch_tensor* qk, const larch_tensor* w);
// layernorm; b is optional.
struct layernorm_checked {
  meta x, w, b, out;
};
layernorm_checked check_layernorm(const sizer&, const larch_tensor* x, const larch_tensor* w, const larch_tensor* b,
                                  const larch_tensor* out);
// rope; freq_factors is optional.
struct rope_checked {
  meta qk, pos, freq_factors;
};
rope_checked check_rope(const sizer&, const larch_tensor* qk, const larch_tensor* pos, const larch_rope_params* p,
                        larch_dtype qk_dtype = LARCH_F16);  // f32: rope_f32
// rope_multi: pos is i32 [rows, 4].
rope_checked check_rope_multi(const sizer&, const larch_tensor* qk, const larch_tensor* pos,
                              const larch_rope_multi_params* p);
// mmvq (a = x, b_t = w, c = out) and gemm; bias is optional.
struct matmul_checked {
  meta a, b_t, bias, c;
};
matmul_checked check_mmvq(const sizer&, const larch_tensor* w, const larch_tensor* x, const larch_tensor* bias,
                          const larch_tensor* out);
matmul_checked check_gemm(const sizer&, const larch_tensor* a, const larch_tensor* b_t, const larch_tensor* bias,
                          const larch_tensor* c);
matmul_checked check_quant_gemm(const sizer&, const larch_tensor* w, const larch_tensor* x,
                                const larch_tensor* bias, const larch_tensor* out);
matmul_checked check_mmvq_verify(const sizer&, const larch_tensor* w, const larch_tensor* x,
                                 const larch_tensor* bias, const larch_tensor* out);
// mmvq_id; bias and weights are optional.
struct mmvq_id_checked {
  meta w, x, ids, bias, weights, out;
};
mmvq_id_checked check_mmvq_id(const sizer&, const larch_tensor* w, const larch_tensor* x, const larch_tensor* ids,
                              const larch_tensor* bias, const larch_tensor* weights, const larch_tensor* out);
// mmvq_id_mixed: mmvq_id's checks with w the hot tensor and no bias; the cold tensor quant, of the hot one's k and a
// whole number of N-row slots; the table i32 [experts].
struct mmvq_id_mixed_checked {
  mmvq_id_checked id;
  meta cold, table;
};
mmvq_id_mixed_checked check_mmvq_id_mixed(const sizer&, const larch_tensor* w_hot, const larch_tensor* w_cold,
                                          const larch_tensor* table, const larch_tensor* x, const larch_tensor* ids,
                                          const larch_tensor* weights, const larch_tensor* out);
// dequant (out is f16 of w's shape).
struct dequant_checked {
  meta w, out;
};
dequant_checked check_dequant(const sizer&, const larch_tensor* w, const larch_tensor* out);
// silu_mul (a = gate, b = up) and add.
struct elementwise_checked {
  meta a, b, out;
};
elementwise_checked check_silu_mul(const sizer&, const larch_tensor* gate, const larch_tensor* up,
                                   const larch_tensor* out, larch_dtype dtype = LARCH_F16);  // f32: silu_mul_f32
elementwise_checked check_add(const sizer&, const larch_tensor* a, const larch_tensor* b, const larch_tensor* out);
// sigmoid_mul (a = x, b = gate).
elementwise_checked check_sigmoid_mul(const sizer&, const larch_tensor* x, const larch_tensor* gate,
                                      const larch_tensor* out);
// attn_full: q, k, v and out of one shape.
struct attn_full_checked {
  meta q, k, v, out;
};
attn_full_checked check_attn_full(const sizer&, const larch_tensor* q, const larch_tensor* k, const larch_tensor* v,
                                  uint32_t seg_len, const larch_tensor* out);
// gelu (a = x, no b).
elementwise_checked check_gelu(const sizer&, const larch_tensor* x, const larch_tensor* out);
// draft_conv; stage is 0 or 1.
struct draft_conv_checked {
  meta x, dyn, base, out;
};
draft_conv_checked check_draft_conv(const sizer&, const larch_tensor* x, const larch_tensor* dyn,
                                    const larch_tensor* base, uint32_t stage, const larch_tensor* out);
// ggml_gelu's tanh form, 0.5 x (1 + tanh(GELU_K (x + GELU_A x^3))): ggml's SQRT_2_OVER_PI and
// GELU_COEF_A, for the reference and the CUDA kernel alike.
inline constexpr float GELU_K = 0.79788456080286535587989211986876f, GELU_A = 0.044715f;
// kv_write; pool_k and pool_v are the layer's [slots, n_kv_heads * head_dim] of the pool's dtype.
struct kv_write_checked {
  meta k, v, slots, pool_k, pool_v;
};
kv_write_checked check_kv_write(const sizer&, const larch_tensor* k, const larch_tensor* v,
                                const larch_tensor* slots, const larch_kv_pool* pool);
// attn_decode (lens = seq_lens, no start), attn_prefill and attn (mask and sinks optional); k and v as
// kv_write's pool_k and pool_v.
struct attn_checked {
  meta q, out, table, k, v, start, lens, mask, sinks, rel_q, rel_w;
};
attn_checked check_attn_decode(const sizer&, const larch_tensor* q, const larch_kv_pool* pool,
                               const larch_tensor* block_table, const larch_tensor* seq_lens,
                               const larch_tensor* out);
attn_checked check_attn_prefill(const sizer&, const larch_tensor* q, const larch_kv_pool* pool,
                                const larch_tensor* block_table, const larch_tensor* q_start_pos,
                                const larch_tensor* q_lens, const larch_tensor* out);
attn_checked check_attn(const sizer&, const larch_tensor* q, const larch_kv_pool* pool,
                        const larch_tensor* block_table, const larch_tensor* q_start_pos, const larch_tensor* q_lens,
                        const larch_tensor* mask, const larch_tensor* sinks, const larch_tensor* out);
// sample_prep; penalty_ids is optional.
struct sample_checked {
  meta logits, penalty_ids, vals, ids, lse;
};
sample_checked check_sample_prep(const sizer&, const larch_tensor* logits, const larch_sample_params* per_row,
                                 const larch_tensor* penalty_ids, const larch_tensor* vals, const larch_tensor* ids,
                                 const larch_tensor* lse);
// sample; ban is optional.
struct draw_checked {
  meta vals, ids, lse, pos, ban, out_ids, out_logprobs;
};
draw_checked check_sample(const sizer&, const larch_tensor* vals, const larch_tensor* ids, const larch_tensor* lse,
                          const larch_draw_params* per_row, const larch_tensor* pos, const larch_tensor* ban,
                          const larch_tensor* out_ids, const larch_tensor* out_logprobs);
// moe_route.
struct route_checked {
  meta logits, ids, weights;
};
route_checked check_moe_route(const sizer&, const larch_tensor* logits, float scale, const larch_tensor* ids,
                              const larch_tensor* weights);
// The training ops, all f32: gemm_ex (a, b, c), rmsnorm_backward (a = x, b = w, c = dy, d = dx),
// silu_mul_backward (gate, up, dout, dgate, dup), softmax (x, out) and softmax_backward (p, dp, ds).
struct train_checked {
  meta a, b, c, d, e;
};
train_checked check_gemm_ex(const sizer&, const larch_tensor* a, const larch_tensor* b, const larch_tensor* c,
                            uint32_t trans_a, uint32_t trans_b);
train_checked check_rmsnorm_backward(const sizer&, const larch_tensor* x, const larch_tensor* w,
                                     const larch_tensor* dy, const larch_tensor* dx);
train_checked check_silu_mul_backward(const sizer&, const larch_tensor* gate, const larch_tensor* up,
                                      const larch_tensor* dout, const larch_tensor* dgate, const larch_tensor* dup);
train_checked check_softmax(const sizer&, const larch_tensor* x, const larch_tensor* out);
train_checked check_softmax_backward(const sizer&, const larch_tensor* p, const larch_tensor* dp,
                                     const larch_tensor* ds);
// gdn_conv and gdn (gdn.cpp): `state` is the layer's part of the larch_gdn_state, as [n_slots, one
// slot's elements] (conv: f16, gdn: f32); in is x (conv) or qkv; snaps is optional, and alpha, beta,
// dt_bias and a are gdn's.
struct gdn_checked {
  meta in, w, slots, lens, snaps, out, state, alpha, beta, dt_bias, a;
};
gdn_checked check_gdn_conv(const sizer&, const larch_tensor* x, const larch_tensor* w, const larch_gdn_state* state,
                           const larch_tensor* slots, const larch_tensor* q_lens, const larch_tensor* snaps,
                           const larch_tensor* out);
gdn_checked check_gdn(const sizer&, const larch_tensor* qkv, const larch_tensor* alpha, const larch_tensor* beta,
                      const larch_tensor* dt_bias, const larch_tensor* a, float eps, const larch_gdn_state* state,
                      const larch_tensor* slots, const larch_tensor* q_lens, const larch_tensor* snaps,
                      const larch_tensor* out);

// The sequence tensors of gdn_conv, gdn and short_conv, for `rows` input rows: slots and q_lens i32 [seqs], snaps
// optional i32 [rows].
void check_sequences(const sizer&, int64_t rows, const larch_tensor* slots, const larch_tensor* q_lens,
                     const larch_tensor* snaps, meta& slots_m, meta& lens_m, meta& snaps_m);
// Each sequence's first row (and the end, rows), after the debug checks on the device values: lengths >= 0
// summing to the rows, slots (and snapshot slots) in [0, n_slots) and none named twice.
std::vector<int64_t> sequences(const meta& slots, const meta& lens, const meta& snaps, int64_t rows, int64_t n_slots);

// Inkling 2's entries (inkling.cpp; attn_rel in ops.cpp beside attn): `state` is the layer's part of the
// larch_conv_state as [n_slots, (kernel - 1) * channels]; dyn, residual, snaps, scale, pos, bias and down_scale are
// optional.
struct short_conv_checked {
  meta in, w, dyn, residual, slots, lens, snaps, out, state;
};
short_conv_checked check_short_conv(const sizer&, const larch_tensor* x, const larch_tensor* w,
                                    const larch_tensor* dyn, const larch_tensor* residual,
                                    const larch_conv_state* state, uint32_t offset, const larch_tensor* slots,
                                    const larch_tensor* q_lens, const larch_tensor* snaps, const larch_tensor* out);
struct clip_norm_checked {
  meta x, scale, pos, out;
};
clip_norm_checked check_clip_norm(const sizer&, const larch_tensor* x, const larch_tensor* scale, float t,
                                  const larch_tensor* pos, uint32_t floor, float alpha, const larch_tensor* out);
// depth_mix (a = states, b = query) and softcap (a = x, no b).
elementwise_checked check_depth_mix(const sizer&, const larch_tensor* states, const larch_tensor* query,
                                    const larch_tensor* out);
elementwise_checked check_softcap(const sizer&, const larch_tensor* x, float cap, const larch_tensor* out);
struct loop_gate_checked {
  meta x, msg, gain, forget, out;
};
loop_gate_checked check_loop_gate(const sizer&, const larch_tensor* x, const larch_tensor* msg,
                                  const larch_tensor* gain, const larch_tensor* forget, const larch_tensor* out);
struct situ_checked {
  meta gate_up, ids, scale, out;
};
situ_checked check_situ_mul(const sizer&, const larch_tensor* gate_up, const larch_tensor* ids,
                            const larch_tensor* scale, float beta, float lin_beta, const larch_tensor* out);
struct route_sigmoid_checked {
  meta logits, bias, down_scale, ids, weights;
};
route_sigmoid_checked check_moe_route_sigmoid(const sizer&, const larch_tensor* logits, const larch_tensor* bias,
                                              float eps, float scale, const larch_tensor* down_scale,
                                              const larch_tensor* ids, const larch_tensor* weights);
// attn_rel: attn's checks with no sinks (attn_checked's sinks hold the per-row sink), plus rel_q and rel_w.
attn_checked check_attn_rel(const sizer&, const larch_tensor* q, const larch_kv_pool* pool,
                            const larch_tensor* block_table, const larch_tensor* q_start_pos,
                            const larch_tensor* q_lens, const larch_tensor* mask, const larch_tensor* rel_q,
                            const larch_tensor* rel_w, const larch_tensor* sink, const larch_tensor* out);

// The fused entries (fused.cpp): each runs its sequence's own checks, in sequence order, plus the ones
// that tie the steps together. x is a residual stream updated in place.
struct fused_gate_up_checked {
  mmvq_id_checked gate, up;
};
fused_gate_up_checked check_fused_moe_gate_up(const sizer&, const larch_tensor* gate_w, const larch_tensor* up_w,
                                              const larch_tensor* x, const larch_tensor* ids,
                                              const larch_tensor* out);
// shared.b_t is null without a shared expert; the down and shared steps are checked against x as their out.
struct fused_down_checked {
  matmul_checked shared;
  meta shared_gate;
  mmvq_id_checked down;
  meta x;
};
fused_down_checked check_fused_moe_down(const sizer&, const larch_tensor* down_w, const larch_tensor* act,
                                        const larch_tensor* ids, const larch_tensor* weights,
                                        const larch_tensor* shared_w, const larch_tensor* shared_act,
                                        const larch_tensor* shared_gate, const larch_tensor* x);
// shared_* are null without a shared expert.
struct fused_moe_in_checked {
  norm_checked norm;
  matmul_checked router;
  route_checked route;
  matmul_checked shared_gate, shared_up, gate_inp;
};
fused_moe_in_checked check_fused_moe_in(const sizer&, const larch_tensor* x, const larch_tensor* norm_w,
                                        const larch_tensor* h, const larch_tensor* router,
                                        const larch_tensor* logits, const larch_tensor* ids,
                                        const larch_tensor* weights, const larch_tensor* shared_gate_w,
                                        const larch_tensor* shared_up_w, const larch_tensor* shared_act,
                                        const larch_tensor* gate_inp, const larch_tensor* shared_gate);
larch_status fused_moe_in(larch_stream*, const larch_tensor*, const larch_tensor*, float, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                           const larch_tensor*);
struct fused_gdn_in_checked {
  norm_checked norm;
  matmul_checked proj[4];  // qkv, z, alpha, beta
};
fused_gdn_in_checked check_fused_gdn_in(const sizer&, const larch_tensor* x, const larch_tensor* norm_w,
                                        const larch_tensor* h, const larch_tensor* qkv_w, const larch_tensor* z_w,
                                        const larch_tensor* alpha_w, const larch_tensor* beta_w,
                                        const larch_tensor* qkv, const larch_tensor* z, const larch_tensor* alpha,
                                        const larch_tensor* beta);
larch_status fused_gdn_in(larch_stream*, const larch_tensor*, const larch_tensor*, float, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
// gate.a is null without a gate.
struct fused_proj_add_checked {
  elementwise_checked gate;
  matmul_checked proj;
};
fused_proj_add_checked check_fused_proj_add(const sizer&, const larch_tensor* w, const larch_tensor* in,
                                            const larch_tensor* gate, const larch_tensor* x);
larch_status fused_proj_add(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                             const larch_tensor*);
// The q and k norms' meta are null without them; gate's without a gate.
struct fused_attn_in_checked {
  meta q_in, gate;
  norm_checked q_norm, k_norm;
  rope_checked q_rope, k_rope;
  kv_write_checked kv;
};
fused_attn_in_checked check_fused_attn_in(const sizer&, const larch_tensor* q_in, const larch_tensor* k,
                                          const larch_tensor* v, const larch_tensor* q_norm,
                                          const larch_tensor* k_norm, const larch_tensor* pos,
                                          const larch_rope_params* p, const larch_tensor* slots,
                                          const larch_kv_pool* pool, const larch_tensor* q,
                                          const larch_tensor* gate);
larch_status fused_attn_in(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                            const larch_tensor*, const larch_tensor*, float, const larch_tensor*,
                            const larch_rope_params*, const larch_tensor*, const larch_kv_pool*,
                            const larch_tensor*, const larch_tensor*);
// conv: gdn_conv's checks; gdn: gdn's on conv, with out standing in for o's shape; z: silu_mul's (a = z).
struct fused_gdn_checked {
  gdn_checked conv, gdn;
  meta norm_w;
  elementwise_checked z;
};
fused_gdn_checked check_fused_gdn(const sizer&, const larch_tensor* qkv, const larch_tensor* conv_w,
                                  const larch_tensor* alpha, const larch_tensor* beta, const larch_tensor* dt_bias,
                                  const larch_tensor* a, float eps, const larch_gdn_state* state,
                                  const larch_tensor* slots, const larch_tensor* q_lens, const larch_tensor* snaps,
                                  const larch_tensor* norm_w, const larch_tensor* z, const larch_tensor* conv,
                                  const larch_tensor* out);
larch_status fused_gdn(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                        const larch_tensor*, const larch_tensor*, const larch_tensor*, float,
                        const larch_gdn_state*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                        const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status fused_moe_gate_up(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                                const larch_tensor*, const larch_tensor*);
larch_status fused_moe_down(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                             const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                             const larch_tensor*);

// A checked CPU tensor with its memory. Plain dtypes only for get/set; quant tensors are read a row at a
// time.
struct view : meta {
  uint8_t* base = nullptr;
  int64_t elem_bytes = 0;

  view(const meta& m);  // m.buf is a CPU buffer, or null

  uint8_t* row(int64_t r) const { return base + r * row_bytes; }
  float get(int64_t r, int64_t c) const;
  void set(int64_t r, int64_t c, float v) const;
  int32_t i32(int64_t r, int64_t c = 0) const;
  void set_i32(int64_t r, int64_t c, int32_t v) const;
  void row_f32(int64_t r, float* dst) const { to_f32(dtype, row(r), cols, dst); }
};

// The CPU reference's sizer.
uint64_t buffer_bytes(const larch_buffer* b, const std::string& what);

// Runs f, turning bad_arg into INVALID_ARG and nothing else unwinding into the C caller.
template <class F>
larch_status run(const char* op, F&& f) {
  try {
    f();
    return LARCH_OK;
  } catch (const bad_arg& e) {
    return vv_fail(LARCH_ERR_INVALID_ARG, std::string(op) + ": " + e.what());
  } catch (const std::bad_alloc&) {
    return vv_fail(LARCH_ERR_OOM, std::string(op) + ": out of host memory");
  } catch (...) {  // nothing may unwind into the C caller
    return vv_fail(LARCH_ERR_LAUNCH, std::string(op) + ": unexpected exception");
  }
}

// Throws bad_arg(what) unless ok.
inline void need(bool ok, const char* what) {
  if (!ok) throw bad_arg(what);
}

// v is `ndim`-D of dtype t.
inline bool is(const meta& v, larch_dtype t, uint32_t ndim) { return v.dtype == t && v.ndim == ndim; }

// Layer `layer` of a buffer b of `size` bytes that holds n_layers of [rows, cols] of dtype t, or
// bad_arg(too_small) when it holds fewer.
meta layer_view(const larch_buffer* b, uint64_t size, larch_dtype t, int64_t rows, int64_t cols, uint32_t layer,
                uint32_t n_layers, const char* too_small);

}  // namespace cpu
