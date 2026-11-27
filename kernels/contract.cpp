// The op contract of larch.h on the CPU reference and, when one is built, GPU 0. Two passes per op:
// - dtypes: every dtype (and NULL where the op takes NULL) on every tensor argument and the pool. The
//   combinations the header lists must return LARCH_OK and every other one LARCH_ERR_INVALID_ARG.
//   Shapes and data are valid for every allowed dtype (rows of 256 where a quant is taken), so the
//   dtype alone decides.
// - metadata: from the first allowed combination, one argument at a time is NULL, of a forbidden dtype,
//   one dim more or fewer, one tied dim longer, overlapping or not dense, past its buffer or absurdly
//   long. Each call must be INVALID_ARG and leave every buffer as it was; so must every `bad` op, whose
//   host parameters, pool or Gated DeltaNet state are out of range (every pool rule runs on every op that
//   takes a pool, every state rule on both GDN ops).
// An ABI 3 entry a GPU backend lacks answers a valid call UNSUPPORTED (larch.h), and so does every
// 8-bit pool on a GPU, since none has those pages yet; the CPU devices have them all. Then the verify
// contract: rows 1-32 of mmvq_verify, and attn's rows over a chain and a tree of draft tokens, bit-equal
// to the one-row mmvq (Metal: mmvq_verify, larch.h) and attn_decode on the same device.
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "tests/testdata.h"

using dtypes = uint32_t;  // a set: bit t for larch_dtype t, bit NUL for a NULL tensor
constexpr int NUL = LARCH_NVFP4 + 1;
constexpr dtypes F32 = 1u << LARCH_F32, F16 = 1u << LARCH_F16, I32 = 1u << LARCH_I32, OPT = 1u << NUL;
constexpr dtypes QUANT = 1u << LARCH_NVFP4 | 1u << LARCH_Q2_K | 1u << LARCH_Q3_K | 1u << LARCH_Q4_K | 1u << LARCH_Q5_K | 1u << LARCH_Q6_K | 1u << LARCH_Q8_0 |
                         1u << LARCH_Q4_0 | 1u << LARCH_Q5_0;
constexpr dtypes KV = F16 | 1u << LARCH_Q8_0 | 1u << LARCH_FP8_E4M3;  // the pool dtypes
const char* const NAMES[] = {"f32",  "f16",  "bf16", "i32",  "q4_k",     "q5_k",
                             "q6_k", "q8_0", "q4_0", "q5_0", "fp8_e4m3", "q3_k", "q2_k", "nvfp4", "NULL"};
constexpr uint32_t DIM0 = 1, DIM1 = 2, DIM2 = 4, POOL = ~0u;

struct arg {
  std::vector<int64_t> shape;
  dtypes allowed;
  int32_t fill = 0;  // i32 arguments: every element, the ids, slots and lengths of the accepted calls
  uint32_t free = 0;  // dims no other argument fixes; POOL: the pool, of which only the dtype is used
};

struct op {
  std::string name;
  std::vector<arg> args;
  larch_status (*run)(larch_stream*, const larch_tensor* const* t);
  bool bad = false;  // host parameters or pool out of range: the first allowed combination is INVALID_ARG
  void (*pool_rule)(larch_kv_pool&) = nullptr;  // breaks (or reshapes) the pool check() sets up
  void (*state_rule)(larch_gdn_state&) = nullptr;  // breaks the GDN state check() sets up
};

// One layer of 16 slots for 1 kv head of 32 over two 2048-byte buffers, which fit any pool dtype, or 2
// heads of f16. check() sets it up per op; the pool argument of an op only picks its dtype.
constexpr uint64_t POOL_BYTES = 2048;
static larch_buffer *pool_k, *pool_v;
static larch_kv_pool pool_now;
static const larch_kv_pool* pool(const larch_tensor* t) {
  if (t) pool_now.dtype = t->dtype;
  return t ? &pool_now : nullptr;
}
// The Gated DeltaNet state, set up by check() like the pool: 2 slots of a layer with kernel 4, 1 k head
// and 2 v heads of 8 (32 channels), over two 2048-byte buffers.
static larch_buffer *gdn_conv_buf, *gdn_ssm_buf;
static larch_gdn_state gdn_now;

// A GPU backend may lack an ABI 3 entry and answer a valid call UNSUPPORTED; a CPU device never does.
static bool on_gpu = false;
// CUDA has Q2_K's embed, dequant and mmvq_id calls (larch.h), so it may not refuse those.
static bool on_cuda = false;
static larch_status lacking_weights(larch_status got, bool may_lack) {
  return on_gpu && may_lack && got == LARCH_ERR_UNSUPPORTED ? LARCH_OK : got;
}
static larch_status lacking(larch_status got) { return on_gpu && got == LARCH_ERR_UNSUPPORTED ? LARCH_OK : got; }
// lacking, on an f16 pool only: an 8-bit pool's UNSUPPORTED on a GPU is the contract's own answer.
static larch_status lacking_on(const larch_tensor* pool_arg, larch_status got) {
  return pool_arg && pool_arg->dtype == LARCH_F16 ? lacking(got) : got;
}

// kv_write and the two attentions predate ABI 3: every backend runs them on f16 pages.
static larch_status kv_write(larch_stream* s, const larch_tensor* const* t) {
  return larch_kv_write(s, t[0], t[1], t[2], pool(t[3]));
}
static larch_status attn_decode(larch_stream* s, const larch_tensor* const* t) {
  return larch_attn_decode(s, t[0], pool(t[1]), t[2], t[3], 0, 0.35f, t[4]);
}
static larch_status attn_prefill(larch_stream* s, const larch_tensor* const* t) {
  return larch_attn_prefill(s, t[0], pool(t[1]), t[2], t[3], t[4], 0.35f, t[5]);
}
// On an 8-bit pool attn must be refused even where it exists (check()), so only its f16 calls may be lacking.
static larch_status attn(larch_stream* s, const larch_tensor* const* t) {
  const larch_status got = larch_attn(s, t[0], pool(t[1]), t[2], t[3], t[4], t[5], t[6], 2, 0, 0.35f, t[7]);
  return pool_now.dtype == LARCH_F16 ? lacking(got) : got;
}
static larch_status gdn_conv(larch_stream* s, const larch_tensor* const* t) {
  return lacking(larch_gdn_conv(s, t[0], t[1], &gdn_now, t[2], t[3], t[4], t[5]));
}
static larch_status gdn(larch_stream* s, const larch_tensor* const* t) {
  return lacking(larch_gdn(s, t[0], t[1], t[2], t[3], t[4], 1e-6f, &gdn_now, t[5], t[6], t[7], t[8]));
}
// Inkling 2's short convolutions' state (ABI 6), set up by check() over the GDN conv buffer: 2 slots of one layer,
// kernel 4, 64 channels, of which the rows below take 32 at offset 32.
static larch_conv_state conv_now;
static larch_status short_conv(larch_stream* s, const larch_tensor* const* t, const larch_conv_state* st,
                               uint32_t offset) {
  return lacking(larch_short_conv(s, t[0], t[1], t[2], t[3], st, offset, t[4], t[5], t[6], t[7]));
}
// attn_rel on an 8-bit pool must be refused on a GPU even where it exists, as attn.
static larch_status attn_rel(larch_stream* s, const larch_tensor* const* t) {
  const larch_status got =
      larch_attn_rel(s, t[0], pool(t[1]), t[2], t[3], t[4], t[5], t[6], t[7], t[8], 2, 0, 0.35f, t[9]);
  return pool_now.dtype == LARCH_F16 ? lacking(got) : got;
}

static const larch_sample_params SAMPLE = {1.0f, 0, 1.0f, 1};
static const larch_draw_params DRAW = {0.7f, 0.9f, 0, 0, 42};
static const larch_rope_params ROPE = {8, 0, 4096, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, nullptr};

// The valid arguments of the `bad` ops below (position 1, so a rope that wrongly writes changes qk).
static const std::vector<arg> ROPE_ARGS = {{{1, 2, 8}, F16}, {{1}, I32, 1}};
static const std::vector<arg> SAMPLE_ARGS = {{{1, 64}, F32}, {{1, 1}, I32}, {{1, 64}, F32}, {{1, 64}, I32}, {{1}, F32}};
static const std::vector<arg> DRAW_ARGS = {{{1, 64}, F32}, {{1, 64}, I32}, {{1}, F32},
                                           {{1}, I32},     {{1}, I32, 5},  {{1}, F32}};
static const std::vector<arg> CONV_ARGS = {{{1, 32}, F16}, {{32, 4}, F32}, {{1}, I32}, {{1}, I32, 1}, {{1}, I32, 1},
                                           {{1, 32}, F16}};
// short_conv: x, w, dyn, residual, slots, q_lens, snaps, out (the valid arguments of its `bad` rows).
static const std::vector<arg> SHORT_CONV_ARGS = {{{2, 32}, F16},           {{32, 4}, F32},
                                                 {{2, 2, 4}, F16 | F32 | OPT}, {{2, 32}, F32 | OPT},
                                                 {{1}, I32},               {{1}, I32, 2},
                                                 {{2}, I32 | OPT, -1},     {{2, 32}, F16 | F32}};
static const std::vector<arg> CLIP_ARGS = {{{2, 2, 32}, F16 | F32}, {{2, 32}, F32 | OPT}, {{2}, I32 | OPT},
                                           {{2, 2, 32}, F16 | F32}};
static const std::vector<arg> GDN_ARGS = {{{1, 32}, F16}, {{1, 2}, F32}, {{1, 2}, F32}, {{2}, F32}, {{2}, F32},
                                          {{1}, I32}, {{1}, I32, 1}, {{1}, I32, 1}, {{1, 16}, F16}};
static larch_status draw(larch_stream* s, const larch_tensor* const* t, const larch_draw_params* p) {
  return lacking(larch_sample(s, t[0], t[1], t[2], p, t[3], nullptr, t[4], t[5]));
}

// Adds every pool rule of larch.h, as a `bad` op, to every op that takes a pool.
static std::vector<op> with_bad_pools(std::vector<op> ops) {
  const std::pair<const char*, void (*)(larch_kv_pool&)> rules[] = {
      {"pool k NULL", [](larch_kv_pool& p) { p.k = nullptr; }},
      {"pool v NULL", [](larch_kv_pool& p) { p.v = nullptr; }},
      {"layer >= n_layers", [](larch_kv_pool& p) { p.layer = p.n_layers; }},
      {"pool past its buffers", [](larch_kv_pool& p) { p.n_blocks = 3; }},
      {"block_tokens 0", [](larch_kv_pool& p) { p.block_tokens = 0; }},
      {"n_kv_heads 0", [](larch_kv_pool& p) { p.n_kv_heads = 0; }},
      {"head_dim 0", [](larch_kv_pool& p) { p.head_dim = 0; }},
      // More kv heads than q heads (in 1536 bytes): a group size of 0 for attention, k's heads for kv_write.
      {"n_kv_heads 3",
       [](larch_kv_pool& p) {
         p.n_kv_heads = 3;
         p.block_tokens = 8;
       }},
  };
  for (size_t i = 0, n = ops.size(); i < n; ++i) {
    const bool pooled = std::any_of(ops[i].args.begin(), ops[i].args.end(), [](const arg& a) { return a.free == POOL; });
    if (ops[i].bad || !pooled) continue;
    for (const auto& [what, rule] : rules) {
      op o = ops[i];
      o.name += std::string(", ") + what;
      o.bad = true;
      o.pool_rule = rule;
      ops.push_back(std::move(o));
    }
  }
  return ops;
}

// Adds every larch_gdn_state rule of larch.h, as a `bad` op, to each op.
static std::vector<op> with_bad_states(std::vector<op> ops) {
  const std::pair<const char*, void (*)(larch_gdn_state&)> rules[] = {
      {"conv NULL", [](larch_gdn_state& g) { g.conv = nullptr; }},
      {"ssm NULL", [](larch_gdn_state& g) { g.ssm = nullptr; }},
      {"layer >= n_layers", [](larch_gdn_state& g) { g.layer = g.n_layers; }},
      {"state past its buffers", [](larch_gdn_state& g) { g.n_slots = 64; }},
      {"n_slots 0", [](larch_gdn_state& g) { g.n_slots = 0; }},
      {"kernel 1", [](larch_gdn_state& g) { g.kernel = 1; }},
      {"head_dim 0", [](larch_gdn_state& g) { g.head_dim = 0; }},
      // 32 channels still, so for gdn_conv the heads alone are wrong.
      {"v_heads not a multiple of k_heads",
       [](larch_gdn_state& g) {
         g.k_heads = 3;
         g.v_heads = 2;
         g.head_dim = 4;
       }},
  };
  for (size_t i = 0, n = ops.size(); i < n; ++i) {
    if (ops[i].bad) continue;
    for (const auto& [what, rule] : rules) {
      op o = ops[i];
      o.name += std::string(", ") + what;
      o.bad = true;
      o.state_rule = rule;
      ops.push_back(std::move(o));
    }
  }
  return ops;
}

static const std::vector<op> OPS = with_bad_pools({
    {"embed", {{{2, 256}, QUANT | F16 | F32, 0, DIM0}, {{1}, I32}, {{1, 256}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_embed(s, t[0], t[1], t[2]); }},
    {"rmsnorm", {{{1, 256}, F32}, {{256}, F32}, {{1, 256}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_rmsnorm(s, t[0], t[1], 1e-6f, t[2]); }},
    {"rope", {{{1, 2, 8}, F16, 0, DIM1 | DIM2}, {{1}, I32, 1}, {{4}, F32 | OPT}},
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_rope_params p = ROPE;
       p.freq_factors = t[2];
       return larch_rope(s, t[0], t[1], &p);
     }},
    // ggml_rope_multi (ABI 3; a GPU backend may lack it): Qwen 3.5's interleaved text rope, a Qwen-VL
    // encoder's vision rope (n_dims head_dim / 2), and the malformed calls.
    {"rope_multi", {{{1, 2, 16}, F16, 0, DIM1 | DIM2}, {{1, 4}, I32, 1}, {{4}, F32 | OPT}},
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_rope_multi_params p = {ROPE, {2, 1, 1, 0}, LARCH_ROPE_IMROPE};
       p.rope.freq_factors = t[2];
       return lacking(larch_rope_multi(s, t[0], t[1], &p));
     }},
    {"rope_multi, vision", {{{1, 2, 16}, F16, 0, DIM1}, {{1, 4}, I32, 1}},
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_rope_multi_params p = {ROPE, {4, 4, 4, 4}, LARCH_ROPE_VISION};
       return lacking(larch_rope_multi(s, t[0], t[1], &p));
     }},
    {"rope_multi, params NULL", {{{1, 2, 16}, F16, 0, DIM1}, {{1, 4}, I32, 1}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_rope_multi(s, t[0], t[1], nullptr); }, true},
    {"rope_multi, vision n_dims not head_dim / 2", {{{1, 2, 16}, F16, 0, DIM1}, {{1, 4}, I32, 1}},
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_rope_multi_params p = {ROPE, {4, 4, 4, 4}, LARCH_ROPE_VISION};
       p.rope.n_dims = 16;
       return larch_rope_multi(s, t[0], t[1], &p);
     },
     true},
    {"rope, params NULL", ROPE_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) { return larch_rope(s, t[0], t[1], nullptr); }, true},
    {"rope, odd n_dims", ROPE_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_rope_params p = ROPE;
       p.n_dims = 7;
       return larch_rope(s, t[0], t[1], &p);
     },
     true},
    {"rope, n_dims 0", ROPE_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_rope_params p = ROPE;
       p.n_dims = 0;
       return larch_rope(s, t[0], t[1], &p);
     },
     true},
    {"rope, n_dims > head_dim", ROPE_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_rope_params p = ROPE;
       p.n_dims = 10;
       return larch_rope(s, t[0], t[1], &p);
     },
     true},
    {"qk_norm", {{{1, 2, 8}, F16, 0, DIM0 | DIM1}, {{8}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_qk_norm(s, t[0], t[1], 1e-6f); }},
    {"mmvq", {{{2, 256}, QUANT}, {{1, 256}, F16}, {{2}, F32 | OPT}, {{1, 2}, F16 | F32}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_mmvq(s, t[0], t[1], t[2], t[3]); }},
    {"mmvq, 17 rows", {{{2, 256}, QUANT}, {{17, 256}, F16}, {{17, 2}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_mmvq(s, t[0], t[1], nullptr, t[2]); },
     true},
    {"mmvq_verify", {{{2, 256}, QUANT}, {{1, 256}, F16}, {{2}, F32 | OPT}, {{1, 2}, F16 | F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_mmvq_verify(s, t[0], t[1], t[2], t[3]));
     }},
    {"mmvq_verify, 33 rows", {{{2, 256}, QUANT}, {{33, 256}, F16}, {{33, 2}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_mmvq_verify(s, t[0], t[1], nullptr, t[2]));
     },
     true},
    // Two experts of N = 2 over w's 4 rows; both slots of the row pick expert 1.
    {"mmvq_id", {{{4, 256}, QUANT}, {{1, 256}, F16}, {{1, 2}, I32, 1}, {{4}, F32 | OPT}, {{1, 2, 2}, F16 | F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_mmvq_id(s, t[0], t[1], t[2], t[3], nullptr, t[4]));
     }},
    {"mmvq_id, weighted",
     {{{4, 256}, QUANT}, {{1, 2, 256}, F16}, {{1, 2}, I32, 1}, {{4}, F32 | OPT}, {{1, 2}, F32}, {{1, 2}, F16 | F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_mmvq_id(s, t[0], t[1], t[2], t[3], t[4], t[5]));
     }},
    // Two experts of N = 2, expert 0 cold and expert 1 hot (the table's fill puts every expert at hot slot 0, so the
    // slot is in range), both slots of the row picking expert 1.
    {"mmvq_id_mixed",
     {{{2, 256}, QUANT}, {{2, 256}, QUANT}, {{2}, I32, 0, DIM0}, {{1, 256}, F16}, {{1, 2}, I32, 1}, {{1, 2, 2}, F16 | F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_mmvq_id_mixed(s, t[0], t[1], t[2], t[3], t[4], nullptr, t[5]));
     }},
    {"mmvq_id_mixed, weighted",
     {{{2, 256}, QUANT},
      {{2, 256}, QUANT},
      {{2}, I32, 0, DIM0},
      {{1, 2, 256}, F16},
      {{1, 2}, I32, 1},
      {{1, 2}, F32},
      {{1, 2}, F16 | F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_mmvq_id_mixed(s, t[0], t[1], t[2], t[3], t[4], t[5], t[6]));
     }},
    // The fused entries (ABI 4), on mmvq_id's shapes: a backend may lack one, a dtype pair or a route.
    {"fused_moe_gate_up", {{{4, 256}, QUANT}, {{4, 256}, QUANT}, {{1, 256}, F16}, {{1, 2}, I32, 1}, {{1, 2, 2}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_fused_moe_gate_up(s, t[0], t[1], t[2], t[3], t[4]));
     }},
    {"fused_moe_down", {{{4, 256}, QUANT}, {{1, 2, 256}, F16}, {{1, 2}, I32, 1}, {{1, 2}, F32}, {{1, 2}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_fused_moe_down(s, t[0], t[1], t[2], t[3], nullptr, nullptr, nullptr, t[4]));
     }},
    {"fused_moe_down, shared",
     {{{4, 256}, QUANT}, {{1, 2, 256}, F16}, {{1, 2}, I32, 1}, {{1, 2}, F32}, {{2, 256}, QUANT}, {{1, 256}, F16},
      {{1, 1}, F16}, {{1, 2}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_fused_moe_down(s, t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]));
     }},
    {"fused_moe_down, shared without its gate",
     {{{4, 256}, QUANT}, {{1, 2, 256}, F16}, {{1, 2}, I32, 1}, {{1, 2}, F32}, {{2, 256}, QUANT}, {{1, 256}, F16},
      {{1, 2}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_fused_moe_down(s, t[0], t[1], t[2], t[3], t[4], t[5], nullptr, t[6]));
     },
     true},
    // x [1, 256], two experts of k 1 (ids [1, 1]), a shared expert of 2.
    {"fused_moe_in",
     {{{1, 256}, F32}, {{256}, F32}, {{1, 256}, F16}, {{2, 256}, F16}, {{1, 2}, F32}, {{1, 1}, I32}, {{1, 1}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_fused_moe_in(s, t[0], t[1], 1e-6f, t[2], t[3], t[4], t[5], t[6], nullptr, nullptr,
                                          nullptr, nullptr, nullptr));
     }},
    {"fused_moe_in, shared",
     {{{1, 256}, F32}, {{256}, F32}, {{1, 256}, F16}, {{2, 256}, F16}, {{1, 2}, F32}, {{1, 1}, I32}, {{1, 1}, F32},
      {{2, 256}, QUANT}, {{2, 256}, QUANT}, {{1, 2}, F16}, {{1, 256}, F16}, {{1, 1}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_fused_moe_in(s, t[0], t[1], 1e-6f, t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9],
                                          t[10], t[11]));
     }},
    // x [1, 256]: channels 2, z 2, 1 v head; alpha and beta quantized or f16.
    {"fused_gdn_in",
     {{{1, 256}, F32}, {{256}, F32}, {{1, 256}, F16}, {{2, 256}, QUANT}, {{2, 256}, QUANT}, {{1, 256}, QUANT | F16},
      {{1, 256}, QUANT | F16}, {{1, 2}, F16}, {{1, 2}, F16}, {{1, 1}, F32}, {{1, 1}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_fused_gdn_in(s, t[0], t[1], 1e-6f, t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9],
                                          t[10]));
     }},
    {"fused_proj_add", {{{2, 256}, QUANT}, {{1, 256}, F16}, {{1, 256}, F16 | OPT}, {{1, 2}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_fused_proj_add(s, t[0], t[1], t[2], t[3]));
     }},
    // Attention's inputs on kv_write's pool (1 kv head of 32): plain q in place (its heads free), with and without the
    // q and k norms, then a gated q.
    {"fused_attn_in", {{{2, 1, 32}, F16, 0, DIM1}, {{2, 1, 32}, F16}, {{2, 1, 32}, F16}, {{2}, I32}, {{2}, I32, 3},
                       {{1}, KV, 0, POOL}},
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_rope_params p = {};
       p.n_dims = 32, p.n_ctx_orig = 4096, p.freq_base = 10000, p.freq_scale = 1, p.attn_factor = 1;
       return lacking_on(t[5], larch_fused_attn_in(s, t[0], t[1], t[2], nullptr, nullptr, 1e-6f, t[3], &p, t[4],
                                                    pool(t[5]), t[0], nullptr));
     }},
    {"fused_attn_in, normed",
     {{{2, 1, 32}, F16, 0, DIM1}, {{2, 1, 32}, F16}, {{2, 1, 32}, F16}, {{32}, F32}, {{32}, F32}, {{2}, I32},
      {{2}, I32, 3}, {{1}, KV, 0, POOL}},
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_rope_params p = {};
       p.n_dims = 32, p.n_ctx_orig = 4096, p.freq_base = 10000, p.freq_scale = 1, p.attn_factor = 1;
       return lacking_on(t[7], larch_fused_attn_in(s, t[0], t[1], t[2], t[3], t[4], 1e-6f, t[5], &p, t[6], pool(t[7]),
                                                    t[0], nullptr));
     }},
    {"fused_attn_in, gated",
     {{{2, 1, 64}, F16}, {{2, 1, 32}, F16}, {{2, 1, 32}, F16}, {{2}, I32}, {{2}, I32, 3}, {{1}, KV, 0, POOL},
      {{2, 1, 32}, F16}, {{2, 1, 32}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_rope_params p = {};
       p.n_dims = 32, p.n_ctx_orig = 4096, p.freq_base = 10000, p.freq_scale = 1, p.attn_factor = 1;
       return lacking_on(t[5], larch_fused_attn_in(s, t[0], t[1], t[2], nullptr, nullptr, 1e-6f, t[3], &p, t[4],
                                                    pool(t[5]), t[6], t[7]));
     }},
    // UNSUPPORTED (no direct path on this backend or for this weight type) answers a valid call
    // only: a malformed one is INVALID_ARG on every backend.
    {"quant_gemm", {{{2, 256}, QUANT}, {{1, 256}, F16}, {{2}, F32 | OPT}, {{1, 2}, F16 | F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       const larch_status got = larch_quant_gemm(s, t[0], t[1], t[2], t[3]);
       return got == LARCH_ERR_UNSUPPORTED ? LARCH_OK : got;
     }},
    {"dequant", {{{2, 256}, QUANT}, {{2, 256}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_dequant(s, t[0], t[1]); }},
    {"gemm", {{{1, 256}, F16}, {{2, 256}, F16}, {{2}, F32 | OPT}, {{1, 2}, F16 | F32}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_gemm(s, t[0], t[1], t[2], t[3], 1.0f, 0.0f); }},
    {"silu_mul", {{{1, 256}, F16}, {{1, 256}, F16}, {{1, 256}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_silu_mul(s, t[0], t[1], t[2]); }},
    {"sigmoid_mul", {{{1, 256}, F16}, {{1, 256}, F16}, {{1, 256}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) { return lacking(larch_sigmoid_mul(s, t[0], t[1], t[2])); }},
    {"sigmoid_mul, a gate per row", {{{1, 256}, F16}, {{1, 1}, F16}, {{1, 256}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) { return lacking(larch_sigmoid_mul(s, t[0], t[1], t[2])); }},
    {"sigmoid_mul, out == x", {{{1, 256}, F16}, {{1, 256}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) { return lacking(larch_sigmoid_mul(s, t[0], t[1], t[0])); }},
    // A vision encoder's three (ABI 3): Metal lacks them, and CUDA's attn_full takes head_dim 64 only,
    // so its row of 32 is UNSUPPORTED there; a malformed call is INVALID_ARG everywhere. (gelu in place
    // has one tensor, which no metadata change can set against another: ops.cpp runs it.)
    {"layernorm", {{{1, 256}, F32}, {{256}, F32}, {{256}, F32 | OPT}, {{1, 256}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_layernorm(s, t[0], t[1], t[2], 1e-6f, t[3]));
     }},
    {"gelu", {{{1, 256}, F16}, {{1, 256}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) { return lacking(larch_gelu(s, t[0], t[1])); }},
    {"attn_full", {{{2, 1, 64}, F16}, {{2, 1, 64}, F16}, {{2, 1, 64}, F16}, {{2, 1, 64}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_attn_full(s, t[0], t[1], t[2], 2, 0.125f, t[3]));
     }},
    {"attn_full, head_dim 32", {{{2, 1, 32}, F16}, {{2, 1, 32}, F16}, {{2, 1, 32}, F16}, {{2, 1, 32}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_attn_full(s, t[0], t[1], t[2], 2, 0.125f, t[3]));
     }},
    {"attn_full, rows not whole segments", {{{2, 1, 64}, F16}, {{2, 1, 64}, F16}, {{2, 1, 64}, F16}, {{2, 1, 64}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return larch_attn_full(s, t[0], t[1], t[2], 3, 0.125f, t[3]);
     },
     true},
    // DFlash2's dynamic conv (a GPU backend may lack it): 8 groups of 32 channels, then stage 2.
    {"draft_conv", {{{2, 256}, F16}, {{2, 4, 8}, F32}, {{4, 256}, F32}, {{2, 256}, F16 | F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_draft_conv(s, t[0], t[1], t[2], 1, t[3]));
     }},
    {"draft_conv, stage 2", {{{2, 256}, F16}, {{2, 4, 8}, F32}, {{4, 256}, F32}, {{2, 256}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_draft_conv(s, t[0], t[1], t[2], 2, t[3]); },
     true},
    {"add", {{{1, 256}, F32}, {{1, 256}, F16}, {{1, 256}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_add(s, t[0], t[1], t[2]); }},
    {"add, out == a", {{{1, 256}, F32}, {{1, 256}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) { return larch_add(s, t[0], t[1], t[0]); }},
    {"kv_write", {{{1, 1, 32}, F16}, {{1, 1, 32}, F16}, {{1}, I32}, {{1}, KV, 0, POOL}}, kv_write},
    {"kv_write, q8_0 pool of head_dim 16", {{{1, 1, 16}, F16}, {{1, 1, 16}, F16}, {{1}, I32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_kv_pool p = pool_now;
       p.dtype = LARCH_Q8_0;
       p.head_dim = 16;
       return larch_kv_write(s, t[0], t[1], t[2], &p);
     },
     true},
    {"attn_decode",
     {{{1, 2, 32}, F16}, {{1}, KV, 0, POOL}, {{1, 1}, I32, 0, DIM1}, {{1}, I32, 1}, {{1, 2, 32}, F16}},
     attn_decode},
    {"attn_decode, 3 heads over 2 kv heads",
     {{{1, 3, 32}, F16}, {{1}, KV, 0, POOL}, {{1, 1}, I32}, {{1}, I32, 1}, {{1, 3, 32}, F16}}, attn_decode, true,
     [](larch_kv_pool& p) { p.n_kv_heads = 2; }},
    {"attn_prefill",
     {{{1, 2, 32}, F16}, {{1}, KV, 0, POOL}, {{1, 1}, I32, 0, DIM1}, {{1}, I32}, {{1}, I32, 1}, {{1, 2, 32}, F16}},
     attn_prefill},
    {"attn_prefill, 3 heads over 2 kv heads",
     {{{1, 3, 32}, F16}, {{1}, KV, 0, POOL}, {{1, 1}, I32}, {{1}, I32}, {{1}, I32, 1}, {{1, 3, 32}, F16}},
     attn_prefill, true, [](larch_kv_pool& p) { p.n_kv_heads = 2; }},
    // Row 0 attends to itself through mask bit 0, over a window of 2.
    {"attn",
     {{{1, 2, 32}, F16}, {{1}, KV, 0, POOL}, {{1, 1}, I32, 0, DIM1}, {{1}, I32}, {{1}, I32, 1}, {{1}, I32 | OPT, 1},
      {{2}, F32 | OPT}, {{1, 2, 32}, F16}},
     attn},
    {"sample_prep",
     {{{1, 64}, F32, 0, DIM1}, {{1, 1}, I32 | OPT, 0, DIM1}, {{1, 64}, F32}, {{1, 64}, I32}, {{1}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return larch_sample_prep(s, t[0], &SAMPLE, t[1], t[2], t[3], t[4]);
     }},
    {"sample_prep, per_row NULL", SAMPLE_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       return larch_sample_prep(s, t[0], nullptr, t[1], t[2], t[3], t[4]);
     },
     true},
    {"sample_prep, top_k 65", SAMPLE_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_sample_params p = SAMPLE;
       p.top_k = 65;
       return larch_sample_prep(s, t[0], &p, t[1], t[2], t[3], t[4]);
     },
     true},
    {"sample_prep, repeat_penalty 0", SAMPLE_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_sample_params p = SAMPLE;
       p.repeat_penalty = 0.0f;
       return larch_sample_prep(s, t[0], &p, t[1], t[2], t[3], t[4]);
     },
     true},
    {"sample_prep, penalty_window > width", SAMPLE_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_sample_params p = SAMPLE;
       p.penalty_window = 2;
       return larch_sample_prep(s, t[0], &p, t[1], t[2], t[3], t[4]);
     },
     true},
    // Every candidate is id 0, drawn at position 0; out_ids starts as 5.
    {"sample",
     {{{1, 64}, F32}, {{1, 64}, I32}, {{1}, F32}, {{1}, I32}, {{1, 1}, I32 | OPT, -1, DIM1}, {{1}, I32, 5}, {{1}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_sample(s, t[0], t[1], t[2], &DRAW, t[3], t[4], t[5], t[6]));
     }},
    {"sample, per_row NULL", DRAW_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) { return draw(s, t, nullptr); }, true},
    {"sample, top_k 65", DRAW_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_draw_params p = DRAW;
       p.top_k = 65;
       return draw(s, t, &p);
     },
     true},
    {"sample, top_p 0", DRAW_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_draw_params p = DRAW;
       p.top_p = 0.0f;
       return draw(s, t, &p);
     },
     true},
    {"sample, top_p 1.5", DRAW_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_draw_params p = DRAW;
       p.top_p = 1.5f;
       return draw(s, t, &p);
     },
     true},
    {"sample, NaN temperature", DRAW_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_draw_params p = DRAW;
       p.temperature = std::bit_cast<float>(0x7fc00000u);
       return draw(s, t, &p);
     },
     true},
    // All four logits tie: ids 0 and 1 (out_ids starts as 9).
    {"moe_route", {{{1, 4}, F32, 0, DIM1}, {{1, 2}, I32, 9}, {{1, 2}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_moe_route(s, t[0], 2.5f, t[1], t[2]));
     }},
    {"moe_route, k > experts", {{{1, 2}, F32}, {{1, 3}, I32, 9}, {{1, 3}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_moe_route(s, t[0], 0.0f, t[1], t[2]));
     },
     true},
    // Inkling 2's entries (ABI 6): a GPU backend may lack each.
    {"short_conv", SHORT_CONV_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) { return short_conv(s, t, &conv_now, 32); }},
    {"short_conv, out == residual", {{{2, 32}, F16}, {{32, 4}, F32}, {{2, 2, 4}, F16 | F32 | OPT}, {{2, 32}, F32},
                                     {{1}, I32}, {{1}, I32, 2}, {{2}, I32 | OPT, -1}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_short_conv(s, t[0], t[1], t[2], t[3], &conv_now, 0, t[4], t[5], t[6], t[3]));
     }},
    {"short_conv, offset + C past the channels", SHORT_CONV_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) { return short_conv(s, t, &conv_now, 33); }, true},
    {"short_conv, state NULL", SHORT_CONV_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) { return short_conv(s, t, nullptr, 0); }, true},
    {"short_conv, conv NULL", SHORT_CONV_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_conv_state st = conv_now;
       st.conv = nullptr;
       return short_conv(s, t, &st, 0);
     },
     true},
    {"short_conv, layer >= n_layers", SHORT_CONV_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_conv_state st = conv_now;
       st.layer = 1;
       return short_conv(s, t, &st, 0);
     },
     true},
    {"short_conv, state past its buffer", SHORT_CONV_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_conv_state st = conv_now;
       st.n_slots = 8;
       return short_conv(s, t, &st, 0);
     },
     true},
    {"short_conv, kernel 1", SHORT_CONV_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_conv_state st = conv_now;
       st.kernel = 1;
       return short_conv(s, t, &st, 0);
     },
     true},
    {"short_conv, n_slots 0", SHORT_CONV_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       larch_conv_state st = conv_now;
       st.n_slots = 0;
       return short_conv(s, t, &st, 0);
     },
     true},
    {"clip_norm", CLIP_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_clip_norm(s, t[0], t[1], 2.0f, t[2], 8, 0.1f, t[3]));
     }},
    {"clip_norm, out == x", {{{2, 2, 32}, F16 | F32}, {{2, 32}, F32 | OPT}, {{2}, I32 | OPT}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_clip_norm(s, t[0], t[1], 2.0f, t[2], 8, 0.1f, t[0]));
     }},
    {"clip_norm, t 0", CLIP_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_clip_norm(s, t[0], t[1], 0.0f, t[2], 8, 0.1f, t[3]));
     },
     true},
    {"clip_norm, floor 0", CLIP_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_clip_norm(s, t[0], t[1], 2.0f, t[2], 0, 0.1f, t[3]));
     },
     true},
    {"clip_norm, alpha < 0", CLIP_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_clip_norm(s, t[0], t[1], 2.0f, t[2], 8, -0.1f, t[3]));
     },
     true},
    {"depth_mix", {{{3, 2, 32}, F32, 0, DIM0}, {{2, 32}, F32}, {{2, 2, 32}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) { return lacking(larch_depth_mix(s, t[0], t[1], t[2])); }},
    {"loop_gate", {{{2, 32}, F32}, {{2, 32}, F32}, {{2, 32}, F16 | F32}, {{2, 32}, F16 | F32}, {{2, 32}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_loop_gate(s, t[0], t[1], t[2], t[3], t[4]));
     }},
    {"loop_gate, out == x", {{{2, 32}, F32}, {{2, 32}, F32}, {{2, 32}, F16 | F32}, {{2, 32}, F16 | F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_loop_gate(s, t[0], t[1], t[2], t[3], t[0]));
     }},
    {"situ_mul", {{{2, 2, 64}, F16 | F32}, {{2, 2}, I32}, {{8}, F32 | OPT, 0, DIM0}, {{2, 2, 32}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_situ_mul(s, t[0], t[1], t[2], 4.0f, 25.0f, t[3]));
     }},
    {"situ_mul, beta 0", {{{2, 2, 64}, F16}, {{2, 2}, I32}, {{8}, F32 | OPT}, {{2, 2, 32}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_situ_mul(s, t[0], t[1], t[2], 0.0f, 25.0f, t[3]));
     },
     true},
    {"moe_route_sigmoid", {{{2, 8}, F32}, {{8}, F32 | OPT}, {{8}, F32 | OPT}, {{2, 2}, I32, 9}, {{2, 2}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_moe_route_sigmoid(s, t[0], t[1], 1e-20f, 8.0f, t[2], t[3], t[4]));
     }},
    {"moe_route_sigmoid, eps < 0", {{{2, 8}, F32}, {{8}, F32 | OPT}, {{8}, F32 | OPT}, {{2, 2}, I32, 9}, {{2, 2}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_moe_route_sigmoid(s, t[0], t[1], -1.0f, 8.0f, t[2], t[3], t[4]));
     },
     true},
    {"moe_route_sigmoid, k > experts", {{{2, 2}, F32}, {{2}, F32 | OPT}, {{2}, F32 | OPT}, {{2, 3}, I32, 9}, {{2, 3}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_moe_route_sigmoid(s, t[0], t[1], 0.0f, 0.0f, t[2], t[3], t[4]));
     },
     true},
    {"attn_rel",
     {{{1, 2, 32}, F16}, {{1}, KV, 0, POOL}, {{1, 1}, I32, 0, DIM1}, {{1}, I32}, {{1}, I32, 1}, {{1}, I32 | OPT, 1},
      {{1, 2, 16}, F16 | F32}, {{8, 16}, F32, 0, DIM0}, {{1, 2}, F16 | F32}, {{1, 2, 32}, F16}},
     attn_rel},
    // Decode's form: q_lens NULL, q_start_pos the seq_lens (0: a padding row).
    {"attn_rel, decode",
     {{{1, 2, 32}, F16}, {{1}, KV, 0, POOL}, {{1, 1}, I32, 0, DIM1}, {{1}, I32}, {{1, 2, 16}, F16 | F32},
      {{8, 16}, F32, 0, DIM0}, {{1, 2}, F16 | F32}, {{1, 2, 32}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       const larch_status got =
           larch_attn_rel(s, t[0], pool(t[1]), t[2], t[3], nullptr, nullptr, t[4], t[5], t[6], 2, 0, 0.35f, t[7]);
       return pool_now.dtype == LARCH_F16 ? lacking(got) : got;
     }},
    {"softcap", {{{2, 32}, F32}, {{2, 32}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) { return lacking(larch_softcap(s, t[0], 30.0f, t[1])); }},
    {"softcap, cap 0", {{{2, 32}, F32}, {{2, 32}, F32}},
     [](larch_stream* s, const larch_tensor* const* t) { return lacking(larch_softcap(s, t[0], 0.0f, t[1])); }, true},
});

static const std::vector<op> GDN_OPS = with_bad_states({
    {"gdn_conv", {{{1, 32}, F16}, {{32, 4}, F32}, {{1}, I32}, {{1}, I32, 1}, {{1}, I32 | OPT, 1}, {{1, 32}, F16}},
     gdn_conv},
    {"gdn_conv, state NULL", CONV_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_gdn_conv(s, t[0], t[1], nullptr, t[2], t[3], t[4], t[5]));
     },
     true},
    {"gdn",
     {{{1, 32}, F16}, {{1, 2}, F32}, {{1, 2}, F32}, {{2}, F32}, {{2}, F32}, {{1}, I32}, {{1}, I32, 1},
      {{1}, I32 | OPT, 1}, {{1, 16}, F16}},
     gdn},
    {"fused_gdn",
     {{{1, 32}, F16}, {{32, 4}, F32}, {{1, 2}, F32}, {{1, 2}, F32}, {{2}, F32}, {{2}, F32}, {{1}, I32}, {{1}, I32, 1},
      {{1}, I32 | OPT, 1}, {{8}, F32}, {{1, 16}, F16}, {{1, 32}, F16}, {{1, 16}, F16}},
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_fused_gdn(s, t[0], t[1], t[2], t[3], t[4], t[5], 1e-6f, &gdn_now, t[6], t[7], t[8], t[9],
                                       t[10], t[11], t[12]));
     }},
    {"gdn, state NULL", GDN_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_gdn(s, t[0], t[1], t[2], t[3], t[4], 1e-6f, nullptr, t[5], t[6], t[7], t[8]));
     },
     true},
    {"gdn, eps < 0", GDN_ARGS,
     [](larch_stream* s, const larch_tensor* const* t) {
       return lacking(larch_gdn(s, t[0], t[1], t[2], t[3], t[4], -1e-6f, &gdn_now, t[5], t[6], t[7], t[8]));
     },
     true},
});

// A dense tensor over b; the last dim counts blocks for quant.
static larch_tensor tensor(larch_buffer* b, int dtype, const std::vector<int64_t>& shape) {
  const int64_t block = dtype == LARCH_NVFP4                                                ? 64
                        : (dtype >= LARCH_Q4_K && dtype <= LARCH_Q6_K) || dtype >= LARCH_Q3_K ? 256
                        : dtype >= LARCH_Q8_0 && dtype <= LARCH_FP8_E4M3              ? 32
                                                                                        : 1;
  larch_tensor t = {b, 0, larch_dtype(dtype), uint32_t(shape.size()), {}, {}};
  for (int64_t d = int64_t(shape.size()) - 1, stride = 1; d >= 0; --d) {
    t.shape[d] = shape[size_t(d)];
    t.stride[d] = stride;
    stride *= d == int64_t(shape.size()) - 1 ? shape[size_t(d)] / block : shape[size_t(d)];
  }
  return t;
}

// Every byte of the buffers a call can reach, or nothing when a copy fails. Downloads are asynchronous,
// so `all` is sized once, before the first is queued, and outlives the sync on every path.
static std::vector<uint8_t> contents(larch_stream* s, const std::vector<std::pair<larch_buffer*, uint64_t>>& bufs) {
  uint64_t total = 0;
  for (const auto& [b, bytes] : bufs) total += bytes;
  std::vector<uint8_t> all(static_cast<size_t>(total));
  bool ok = true;
  uint64_t at = 0;
  for (const auto& [b, bytes] : bufs) {
    ok = ok && larch_download(s, b, 0, all.data() + at, bytes) == LARCH_OK;
    at += bytes;
  }
  const bool synced = larch_stream_sync(s) == LARCH_OK;
  if (!ok || !synced) return {};
  return all;
}

// Both passes for o (the metadata pass alone without `dtypes`); returns the number of wrong answers.
static int check(larch_device* dev, larch_stream* s, const op& o, bool dtypes) {
  const size_t n = o.args.size();
  std::vector<std::pair<larch_buffer*, uint64_t>> bufs(n);
  // Every exit syncs before it frees: queued ops and copies may still use the buffers.
  const auto done = [&](int wrong) {
    if (larch_stream_sync(s) != LARCH_OK) {
      std::fprintf(stderr, "%s: stream_sync: %s\n", o.name.c_str(), larch_last_error(dev));
      ++wrong;
    }
    if (wrong) std::fprintf(stderr, "%s: %d wrong answers\n", o.name.c_str(), wrong);
    for (const auto& [b, bytes] : bufs) larch_buffer_free(b);
    return wrong;
  };
  pool_now = {pool_k, pool_v, 1, 16, 1, 1, 32, LARCH_F16, 0};
  if (o.pool_rule) o.pool_rule(pool_now);
  gdn_now = {gdn_conv_buf, gdn_ssm_buf, 2, 1, 0, 4, 1, 2, 8};
  conv_now = {gdn_conv_buf, 2, 1, 0, 4, 64};
  if (o.state_rule) o.state_rule(gdn_now);
  for (size_t i = 0; i < n; ++i) {
    int64_t elems = 1;
    for (const int64_t d : o.args[i].shape) elems *= d;
    bufs[i].second = uint64_t(elems) * 8;  // 4 bytes an element fit every dtype, twice for one dim longer
    if (larch_buffer_alloc(dev, bufs[i].second, &bufs[i].first) != LARCH_OK) {
      bufs[i].first = nullptr;
      std::fprintf(stderr, "%s: setup failed: %s\n", o.name.c_str(), larch_last_error(dev));
      return done(1);
    }
  }
  // Each float argument holds its own f16 value (2, 3, 4, 6, 8, 12: finite in every float dtype) and
  // each i32 argument its fill; the pools and states are zero. So whatever an op writes differs from what
  // was there.
  const auto reset = [&] {
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
      const int32_t v = o.args[i].allowed & I32 ? o.args[i].fill : 0x40004000 + int32_t(i) * 0x02000200;
      const std::vector<int32_t> fill(size_t(bufs[i].second / 4), v);
      ok = ok && larch_upload(s, bufs[i].first, 0, fill.data(), bufs[i].second) == LARCH_OK;
    }
    const std::vector<uint8_t> zeros(POOL_BYTES);
    for (larch_buffer* b : {pool_k, pool_v, gdn_conv_buf, gdn_ssm_buf})
      ok = ok && larch_upload(s, b, 0, zeros.data(), POOL_BYTES) == LARCH_OK;
    return ok;
  };
  if (!reset()) {
    std::fprintf(stderr, "%s: setup failed: %s\n", o.name.c_str(), larch_last_error(dev));
    return done(1);
  }
  std::vector<larch_tensor> t(n);
  std::vector<const larch_tensor*> p(n);
  int wrong = 0;
  const auto call = [&](const std::vector<int>& pick) {
    bool allowed = true, eight_bit = false, may_lack = false;
    std::string combo;
    for (size_t i = 0; i < n; ++i) {
      allowed = allowed && (o.args[i].allowed >> pick[i] & 1);
      eight_bit = eight_bit || (o.args[i].free == POOL && pick[i] != LARCH_F16);
      const bool q2_kernel = on_cuda && (o.name.starts_with("embed") || o.name.starts_with("dequant") ||
                                         o.name.starts_with("mmvq_id"));
      may_lack = may_lack || pick[i] == LARCH_Q3_K || pick[i] == LARCH_NVFP4 || (pick[i] == LARCH_Q2_K && !q2_kernel);
      if (pick[i] != NUL) t[i] = tensor(bufs[i].first, pick[i], o.args[i].shape);
      p[i] = pick[i] == NUL ? nullptr : &t[i];
      combo += (i ? ", " : "") + std::string(NAMES[pick[i]]);
    }
    // No GPU backend has 8-bit pages, and every CUDA paged kernel reads its pages as f16: there a valid call on
    // an 8-bit pool must be refused, never run.
    // ponytail: GPU pages are f16 only; when a GPU backend gains 8-bit pages (v20-21), check its output against
    // the CPU reference's here instead.
    const larch_status want =
        !allowed ? LARCH_ERR_INVALID_ARG : on_gpu && eight_bit ? LARCH_ERR_UNSUPPORTED : LARCH_OK;
    // A GPU backend without Q3_K kernels refuses a valid Q3_K call as it does a lacking entry (vv_ops::no_weights), and
    // one without a Q2_K kernel for the op a valid Q2_K call.
    const larch_status got = lacking_weights(o.run(s, p.data()), allowed && may_lack);
    if (got != want && ++wrong <= 5)
      std::fprintf(stderr, "%s(%s): got %d, want %d (%s)\n", o.name.c_str(), combo.c_str(), got, want,
                   got == LARCH_OK ? "accepted" : larch_last_error(dev));
  };
  // Every combination; for an op with more than attn_prefill's 11^6, the most any op had before ABI 3,
  // every allowed one, then each forbidden dtype of one argument with the first allowed dtypes of the others.
  // ponytail: an exception per rejected call costs 4 s per device at 11^6; the full product for the wider
  // ops once the checks stop throwing.
  std::vector<std::vector<int>> all(n), choice(n);
  uint64_t combos = 1;
  for (size_t i = 0; i < n; ++i) {
    for (int d = 0; d <= (o.args[i].allowed & OPT ? NUL : NUL - 1); ++d) all[i].push_back(d);
    combos *= all[i].size();
  }
  const bool wide = combos > 1771561;
  for (size_t i = 0; i < n; ++i)
    for (const int d : all[i])
      if (!wide || o.args[i].allowed >> d & 1) choice[i].push_back(d);
  std::vector<size_t> at(n, 0);
  std::vector<int> pick(n);
  for (bool more = dtypes && !o.bad; more;) {
    for (size_t i = 0; i < n; ++i) pick[i] = choice[i][at[i]];
    call(pick);
    more = false;  // next combination, odometer order
    for (size_t i = 0; i < n && !more; ++i) {
      more = at[i] + 1 < choice[i].size();
      at[i] = more ? at[i] + 1 : 0;
    }
  }
  for (size_t i = 0; i < n && dtypes && wide && !o.bad; ++i)
    for (const int d : all[i])
      if (!(o.args[i].allowed >> d & 1)) {
        for (size_t j = 0; j < n; ++j) pick[j] = std::countr_zero(o.args[j].allowed);
        pick[i] = d;
        call(pick);
      }

  // The metadata pass, from the first allowed dtype of every argument.
  std::vector<std::pair<larch_buffer*, uint64_t>> reach = bufs;
  reach.insert(reach.end(),
               {{pool_k, POOL_BYTES}, {pool_v, POOL_BYTES}, {gdn_conv_buf, POOL_BYTES}, {gdn_ssm_buf, POOL_BYTES}});
  const std::vector<uint8_t> before = reset() ? contents(s, reach) : std::vector<uint8_t>{};
  const auto rejects = [&](const std::string& what) {
    const larch_status got = o.run(s, p.data());
    const bool kept = !before.empty() && contents(s, reach) == before;
    if (got != LARCH_ERR_INVALID_ARG || !kept) {
      std::fprintf(stderr, "%s, %s: got %d%s\n", o.name.c_str(), what.c_str(), got, kept ? "" : ", and a buffer changed");
      ++wrong;
    }
  };
  for (size_t i = 0; i < n; ++i) {
    t[i] = tensor(bufs[i].first, std::countr_zero(o.args[i].allowed), o.args[i].shape);
    p[i] = &t[i];
  }
  if (o.bad) rejects("the first allowed dtypes");
  for (size_t i = 0; i < n && !o.bad; ++i) {
    const arg& a = o.args[i];
    const larch_tensor base = t[i];
    const std::string at = "argument " + std::to_string(i);
    if (!(a.allowed & OPT)) {
      p[i] = nullptr;
      rejects(at + " NULL");
      p[i] = &t[i];
    }
    t[i].dtype = larch_dtype(std::countr_zero(~a.allowed));
    rejects(at + " " + NAMES[t[i].dtype]);
    t[i] = base;
    if (a.free == POOL) continue;
    std::vector<std::pair<std::string, std::vector<int64_t>>> shapes = {{" one dim more", a.shape},
                                                                       {" one dim fewer", a.shape}};
    shapes[0].second.push_back(1);
    shapes[1].second.pop_back();
    for (size_t d = 0; d < a.shape.size(); ++d)
      if (!(a.free >> d & 1)) {
        shapes.push_back({" dim " + std::to_string(d) + " longer", a.shape});
        ++shapes.back().second[d];
      }
    for (const auto& [what, shape] : shapes) {
      t[i] = tensor(bufs[i].first, base.dtype, shape);
      rejects(at + what);
    }
    const auto mutated = [&](const char* what, auto change) {
      t[i] = base;
      change(t[i]);
      rejects(at + what);
    };
    mutated(" rows overlapping", [](larch_tensor& x) { --x.stride[0]; });
    if (base.ndim > 1) mutated(" last dim not dense", [](larch_tensor& x) { x.stride[x.ndim - 1] = 2; });
    mutated(" past its buffer", [&](larch_tensor& x) { x.offset = bufs[i].second; });
    mutated(" offset wrapping", [](larch_tensor& x) { x.offset = UINT64_MAX - 3; });
    mutated(" 2^62 rows", [](larch_tensor& x) { x.shape[0] = int64_t(1) << 62; });
    mutated(" -1 rows", [](larch_tensor& x) { x.shape[0] = -1; });
    t[i] = base;
  }
  return done(wrong);
}

// The verify contract on dev: rows 1-32 of mmvq_verify and attn's chain and tree rows, each bit-equal to the
// one-row mmvq and attn_decode. Returns the number of unequal rows; an entry the backend lacks is NOT RUN.
static int verify(larch_device* dev, larch_stream* s) {
  rng g(5);
  int wrong = 0;
  std::vector<larch_buffer*> owned;
  const auto buffer = [&](uint64_t bytes, const void* src) {
    larch_buffer* b = nullptr;
    if (larch_buffer_alloc(dev, bytes, &b) != LARCH_OK || (src && larch_upload(s, b, 0, src, bytes) != LARCH_OK)) {
      std::fprintf(stderr, "verify: setup failed: %s\n", larch_last_error(dev));
      ++wrong;
    }
    owned.push_back(b);
    return b;
  };
  const auto fetch = [&](larch_buffer* b, uint64_t bytes) {
    std::vector<uint8_t> v(static_cast<size_t>(bytes));
    if (larch_download(s, b, 0, v.data(), bytes) != LARCH_OK || larch_stream_sync(s) != LARCH_OK) ++wrong;
    return v;
  };
  const auto lacks = [&](larch_status got, const char* what) {
    if (got == LARCH_ERR_UNSUPPORTED && on_gpu)
      return std::printf("verify: %s not on this backend: NOT RUN\n", what), true;
    if (got != LARCH_OK) std::fprintf(stderr, "verify: %s: %s\n", what, larch_last_error(dev)), ++wrong;
    return got != LARCH_OK;
  };

  // mmvq_verify, every quant dtype and both output dtypes: 8 outputs over k = 256. Metal's rows match its own
  // one-row mmvq_verify (larch.h).
  larch_device_props props{};
  if (larch_device_props_get(dev, &props) != LARCH_OK) ++wrong;
  const auto one_row = props.backend == LARCH_BACKEND_METAL ? larch_mmvq_verify : larch_mmvq;
  const bytes x = halves(32 * 256, g);
  larch_buffer* xb = buffer(x.size(), x.data());
  for (const larch_dtype t : QUANTS)
    for (const larch_dtype od : {LARCH_F32, LARCH_F16}) {
      const bytes w = quant(t, 8, 256, g);
      larch_buffer* wb = buffer(w.size(), w.data());
      const uint64_t ob = od == LARCH_F32 ? 4 : 2;
      larch_buffer *one = buffer(32 * 8 * ob, nullptr), *many = buffer(32 * 8 * ob, nullptr);
      const larch_tensor wt = tensor(wb, t, {8, 256});
      bool lacked = false;
      for (int64_t r = 0; r < 32 && !lacked; ++r) {
        larch_tensor xr = tensor(xb, LARCH_F16, {1, 256}), o = tensor(one, od, {1, 8});
        xr.offset = uint64_t(r) * 512;
        o.offset = uint64_t(r) * 8 * ob;
        lacked = lacks(one_row(s, &wt, &xr, nullptr, &o), "one-row mmvq");
      }
      const std::vector<uint8_t> want = fetch(one, 32 * 8 * ob);
      for (int64_t n = 1; n <= 32 && !lacked; ++n) {
        const larch_tensor xn = tensor(xb, LARCH_F16, {n, 256}), o = tensor(many, od, {n, 8});
        if ((lacked = lacks(larch_mmvq_verify(s, &wt, &xn, nullptr, &o), "mmvq_verify"))) break;
        const std::vector<uint8_t> got = fetch(many, uint64_t(n) * 8 * ob);
        for (int64_t r = 0; r < n; ++r)
          if (std::memcmp(&got[size_t(r * 8 * int64_t(ob))], &want[size_t(r * 8 * int64_t(ob))], size_t(8 * ob))) {
            std::fprintf(stderr, "verify: mmvq_verify %s, %s out, %lld rows: row %lld differs from one-row mmvq\n",
                         NAMES[t], NAMES[od], (long long)n, (long long)r);
            ++wrong;
          }
      }
    }

  // attn over 37 context positions and a span of 5 drafts at 37..41, 8 heads over 2 kv heads of 32, blocks
  // of 16 shuffled over 8: a chain (causal, no mask) and a tree (parents 0, 0, 1, 2 for rows 1-4).
  constexpr int64_t hd = 32, kvh = 2, heads = 8, bt = 16, ctx = 37, span = 5, row = kvh * hd;
  const uint64_t pool_bytes = 8 * bt * row * 2;
  const bytes kv = halves(size_t(2 * 8 * bt * row), g), q = halves(size_t(span * heads * hd), g);
  const std::vector<int32_t> blocks = {5, 2, 7}, table = {5, 2, 7, 0, 1, 3, 4, 6};
  larch_buffer *kb = buffer(pool_bytes, kv.data()), *vb = buffer(pool_bytes, kv.data() + kv.size() / 2);
  larch_buffer *qb = buffer(q.size(), q.data()), *tb = buffer(8 * 4, table.data());
  const larch_kv_pool pool = {kb, vb, 8, uint32_t(bt), 1, uint32_t(kvh), uint32_t(hd), LARCH_F16, 0};
  const int32_t start_v = int32_t(ctx), len_v = int32_t(span), masks_v[span] = {1, 3, 5, 11, 21};
  larch_buffer *sb = buffer(4, &start_v), *lb = buffer(4, &len_v), *mb = buffer(sizeof masks_v, masks_v);
  larch_buffer *out = buffer(uint64_t(span * heads * hd * 2), nullptr),
                *one = buffer(uint64_t(heads * hd * 2), nullptr);
  const larch_tensor qt = tensor(qb, LARCH_F16, {span, heads, hd}), ot = tensor(out, LARCH_F16, {span, heads, hd});
  const larch_tensor tt = tensor(tb, LARCH_I32, {1, 8}), st = tensor(sb, LARCH_I32, {1}),
                      lt = tensor(lb, LARCH_I32, {1});
  const larch_tensor mt = tensor(mb, LARCH_I32, {span}), o1 = tensor(one, LARCH_F16, {1, heads, hd});
  // A decode pool per row: the positions the row attends to, in order, in blocks 0.. of a fresh pool.
  larch_buffer *ck = buffer(pool_bytes, nullptr), *cv = buffer(pool_bytes, nullptr);
  const std::vector<int32_t> identity = {0, 1, 2, 3, 4, 5, 6, 7};
  larch_buffer* ib = buffer(8 * 4, identity.data());
  const larch_tensor it = tensor(ib, LARCH_I32, {1, 8});
  const larch_kv_pool compact = {ck, cv, 8, uint32_t(bt), 1, uint32_t(kvh), uint32_t(hd), LARCH_F16, 0};
  const auto slot = [&](int64_t p) { return size_t(blocks[size_t(p / bt)] * bt + p % bt); };
  for (const bool tree : {false, true})
    for (const uint32_t splits : {1u, 2u, 5u}) {
      if (wrong) break;
      if (lacks(larch_attn(s, &qt, &pool, &tt, &st, &lt, tree ? &mt : nullptr, nullptr, 0, splits, 0.17f, &ot),
                "attn"))
        break;
      const std::vector<uint8_t> got = fetch(out, uint64_t(span * heads * hd * 2));
      for (int64_t j = 0; j < span; ++j) {
        std::vector<int64_t> at;
        for (int64_t p = 0; p < ctx; ++p) at.push_back(p);
        for (int64_t i = 0; i < span; ++i)
          if (tree ? (masks_v[j] >> i & 1) : i <= j) at.push_back(ctx + i);
        bytes k2(at.size() * size_t(row) * 2), v2(k2.size());
        for (size_t p = 0; p < at.size(); ++p) {
          std::memcpy(&k2[p * row * 2], &kv[slot(at[p]) * row * 2], size_t(row) * 2);
          std::memcpy(&v2[p * row * 2], &kv[kv.size() / 2 + slot(at[p]) * row * 2], size_t(row) * 2);
        }
        larch_tensor qj = tensor(qb, LARCH_F16, {1, heads, hd});
        qj.offset = uint64_t(j * heads * hd * 2);
        const int32_t n = int32_t(at.size());
        if (larch_upload(s, ck, 0, k2.data(), k2.size()) != LARCH_OK ||
            larch_upload(s, cv, 0, v2.data(), v2.size()) != LARCH_OK || larch_upload(s, lb, 0, &n, 4) != LARCH_OK ||
            lacks(larch_attn_decode(s, &qj, &compact, &it, &lt, splits, 0.17f, &o1), "attn_decode"))
          break;
        const std::vector<uint8_t> want = fetch(one, uint64_t(heads * hd * 2));
        if (larch_upload(s, lb, 0, &len_v, 4) != LARCH_OK || larch_stream_sync(s) != LARCH_OK) ++wrong;
        if (std::memcmp(&got[size_t(j * heads * hd * 2)], want.data(), want.size())) {
          std::fprintf(stderr, "verify: attn %s, splits %u: row %lld differs from attn_decode over its positions\n",
                       tree ? "tree" : "chain", splits, (long long)j);
          ++wrong;
        }
      }
    }
  if (larch_stream_sync(s) != LARCH_OK) ++wrong;
  for (larch_buffer* b : owned) larch_buffer_free(b);
  return wrong;
}

int main() {
  if (larch_abi_version() != LARCH_ABI_VERSION) {
    std::fprintf(stderr, "library ABI %u, header ABI %u\n", larch_abi_version(), unsigned(LARCH_ABI_VERSION));
    return 1;
  }
  int32_t gpus = 0;
  if (larch_device_count(&gpus) != LARCH_OK) return 1;
  int wrong = 0;
  for (int32_t ordinal = -2; ordinal < std::min(gpus, 1); ++ordinal) {  // both CPU devices, then GPU 0
    larch_device* dev = nullptr;
    larch_stream* s = nullptr;
    on_gpu = ordinal >= 0;
    bool ok = larch_device_open(ordinal, &dev) == LARCH_OK && larch_stream_create(dev, &s) == LARCH_OK;
    larch_device_props props{};
    on_cuda = ok && larch_device_props_get(dev, &props) == LARCH_OK && props.backend == LARCH_BACKEND_CUDA;
    for (larch_buffer** b : {&pool_k, &pool_v, &gdn_conv_buf, &gdn_ssm_buf})
      ok = ok && larch_buffer_alloc(dev, POOL_BYTES, b) == LARCH_OK;
    if (!ok) {
      std::fprintf(stderr, "device %d: setup failed: %s\n", ordinal, larch_last_error(dev));
      return 1;
    }
    // ponytail: -2 runs the reference's table until the fast CPU backend (c1) fills its slots, so its dtype
    // pass would repeat -1's; run it there once c1 lands.
    for (const std::vector<op>* ops : {&OPS, &GDN_OPS})
      for (const op& o : *ops) wrong += check(dev, s, o, ordinal != -2);
    wrong += verify(dev, s);
    for (larch_buffer* b : {pool_k, pool_v, gdn_conv_buf, gdn_ssm_buf}) larch_buffer_free(b);
    larch_stream_destroy(s);
    larch_device_close(dev);
  }
  std::printf("contract: %d wrong answers\n", wrong);
  return wrong != 0;
}
