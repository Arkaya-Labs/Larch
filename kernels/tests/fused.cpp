// The fused entries (larch.h, ABI 4) against the sequence of ops each stands for, on the same device: the outputs
// must be the same bytes, over rows 1, 2, 3, 4, 8, 16 and 32 (16 where the sequence holds an mmvq) and every weight
// dtype the sequence's ops take. LARCH_TEST_DEVICE picks the device as in ops.cpp: unset, GPU 0 when there is one,
// else the CPU reference; -2 is the CPU device. A GPU may answer a fused call UNSUPPORTED (larch.h) only where it
// cannot reproduce the sequence's bytes, so a refusal is counted and printed for a case the caller marks `may_refuse`
// (CUDA's by-design refusals: gate and up of different dtypes or strides, more than 8 rows where a gemm is replaced)
// and on a GPU whose backend refuses fused calls wholesale (HIP and Metal have none; CUDA's H2 decoder on sm_86/89
// refuses the mmvq-based ones), and fails anywhere else. No bias: no fused entry takes one.
#define _CRT_SECURE_NO_WARNINGS  // std::getenv on MSVC
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

#include "testdata.h"

static int failures = 0, unsupported = 0, passed = 0;
static larch_device* dev;
static larch_stream* st;
static bool gpu = false;
static bool any_refusal = false;  // a GPU that may refuse any fused call (see the header comment)
static std::vector<larch_buffer*> owned;
static rng g(7);

#define CHECK(cond, ...)                                       \
  do {                                                         \
    if (!(cond)) {                                             \
      std::fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);     \
      std::fprintf(stderr, __VA_ARGS__);                       \
      std::fprintf(stderr, "\n");                              \
      ++failures;                                              \
    }                                                          \
  } while (0)

static larch_buffer* buffer(const void* src, size_t n) {
  larch_buffer* b = nullptr;
  if (larch_buffer_alloc(dev, n ? n : 1, &b) != LARCH_OK) {
    std::fprintf(stderr, "alloc of %zu bytes failed: %s\n", n, larch_last_error(dev));
    std::exit(1);
  }
  if (src && n) larch_upload(st, b, 0, src, n);
  owned.push_back(b);
  return b;
}
static void release() {
  larch_stream_sync(st);
  for (larch_buffer* b : owned) larch_buffer_free(b);
  owned.clear();
}
// A dense tensor of t, holding src (n bytes) when given.
static larch_tensor tensor(larch_dtype t, std::vector<int64_t> shape, const void* src = nullptr) {
  const size_t n = nbytes(t, shape);
  larch_tensor x = {buffer(src, n), 0, t, uint32_t(shape.size()), {}, {}};
  for (int64_t i = int64_t(shape.size()) - 1, stride = 1; i >= 0; --i) {
    x.shape[i] = shape[size_t(i)];
    x.stride[i] = stride;
    stride *= i == int64_t(shape.size()) - 1 ? shape[size_t(i)] / block_of(t) : shape[size_t(i)];
  }
  return x;
}
static larch_tensor random(larch_dtype t, std::vector<int64_t> shape) {
  int64_t n = 1;
  for (const int64_t d : shape) n *= d;
  const bytes b = t == LARCH_F16  ? halves(size_t(n), g)
                  : t == LARCH_F32 ? [&] {
                      bytes f(size_t(n) * 4);
                      std::uniform_real_distribution<float> u(-2, 2);
                      for (int64_t i = 0; i < n; ++i) {
                        const float v = u(g);
                        std::memcpy(&f[size_t(i) * 4], &v, 4);
                      }
                      return f;
                    }()
                                    : quant(t, shape[0], shape[1], g);
  return tensor(t, shape, b.data());
}
static bytes fetch(const larch_tensor& t) {
  int64_t n = 1;
  for (uint32_t i = 0; i < t.ndim; ++i) n *= t.shape[i];
  bytes v(size_t(n / block_of(t.dtype) * block_bytes(t.dtype)));
  larch_download(st, t.buf, t.offset, v.data(), v.size());
  larch_stream_sync(st);
  return v;
}
// A copy of t's bytes in a new buffer.
static larch_tensor copy(const larch_tensor& t) {
  const bytes b = fetch(t);
  larch_tensor c = t;
  c.buf = buffer(b.data(), b.size());
  c.offset = 0;
  return c;
}
// Row r's `slots` distinct expert ids in [0, experts).
static larch_tensor ids(int64_t rows, int64_t slots, int64_t experts) {
  std::vector<int32_t> v;
  for (int64_t r = 0; r < rows; ++r) {
    std::vector<int32_t> all(static_cast<size_t>(experts));
    for (int64_t e = 0; e < experts; ++e) all[size_t(e)] = int32_t(e);
    std::shuffle(all.begin(), all.end(), g);
    v.insert(v.end(), all.begin(), all.begin() + slots);
  }
  return tensor(LARCH_I32, {rows, slots}, v.data());
}
static larch_tensor weights(int64_t rows, int64_t slots) {
  std::vector<float> v(size_t(rows * slots));
  std::uniform_real_distribution<float> u(0, 1);
  for (float& x : v) x = u(g);
  return tensor(LARCH_F32, {rows, slots}, v.data());
}

// The outcome of a fused call against the sequence's bytes.
static void judge(const std::string& name, larch_status fused, larch_status unfused, const bytes& want,
                  const bytes& got, bool may_refuse = false) {
  if (unfused != LARCH_OK) {
    CHECK(false, "%s: the sequence failed (%d): %s", name.c_str(), int(unfused), larch_last_error(dev));
  } else if (fused == LARCH_ERR_UNSUPPORTED && gpu && (may_refuse || any_refusal)) {
    ++unsupported;
    std::printf("unsupported: %s\n", name.c_str());
  } else if (fused != LARCH_OK) {
    CHECK(false, "%s: fused failed (%d): %s", name.c_str(), int(fused), larch_last_error(dev));
  } else {
    size_t first = 0;
    while (first < want.size() && first < got.size() && want[first] == got[first]) ++first;
    CHECK(want == got, "%s: bytes differ from byte %zu of %zu", name.c_str(), first, want.size());
    passed += want == got;
  }
}

static const int64_t ROWS[] = {1, 2, 3, 4, 8, 16, 32};
// The model dims the entries run at: the defaults, then main's edge passes (k = 288 on the 32-weight-block quants,
// 64 experts top-8, an odd N).
static int64_t D = 512, EF = 256, SF = 256, EXPERTS = 8, SLOTS = 4;
static const larch_dtype QUANTS32[] = {LARCH_Q8_0, LARCH_Q4_0, LARCH_Q5_0};  // 32-weight blocks: k = 288 fits

// Entry 1: gate and up of the chosen experts, then silu_mul.
static void moe_gate_up(larch_dtype gt, larch_dtype ut, int64_t rows) {
  const std::string name = std::string("moe_gate_up ") + DTYPE[gt] + "/" + DTYPE[ut] + " rows " + std::to_string(rows);
  const larch_tensor gw = random(gt, {EXPERTS * EF, D}), uw = random(ut, {EXPERTS * EF, D});
  const larch_tensor x = random(LARCH_F16, {rows, D}), id = ids(rows, SLOTS, EXPERTS);
  const larch_tensor gs = tensor(LARCH_F16, {rows, SLOTS, EF}), us = tensor(LARCH_F16, {rows, SLOTS, EF});
  const larch_tensor want = tensor(LARCH_F16, {rows, SLOTS, EF}), got = tensor(LARCH_F16, {rows, SLOTS, EF});
  larch_status u = larch_mmvq_id(st, &gw, &x, &id, nullptr, nullptr, &gs);
  if (u == LARCH_OK) u = larch_mmvq_id(st, &uw, &x, &id, nullptr, nullptr, &us);
  const larch_tensor g2 = {gs.buf, 0, LARCH_F16, 2, {rows * SLOTS, EF}, {EF, 1}},
                      u2 = {us.buf, 0, LARCH_F16, 2, {rows * SLOTS, EF}, {EF, 1}},
                      w2 = {want.buf, 0, LARCH_F16, 2, {rows * SLOTS, EF}, {EF, 1}};
  if (u == LARCH_OK) u = larch_silu_mul(st, &g2, &u2, &w2);
  const larch_status f = larch_fused_moe_gate_up(st, &gw, &uw, &x, &id, &got);
  judge(name, f, u, fetch(want), fetch(got), gt != ut);
  release();
}

// Entry 1 with up's rows a block further apart than gate's (a padded stride): each weight keeps its own rows, as
// mmvq_id reads them, or the backend refuses the pair.
static void moe_gate_up_padded(int64_t rows) {
  const std::string name = "moe_gate_up q4_k, up rows padded, rows " + std::to_string(rows);
  const int64_t nb = D / 256 + 1;  // blocks a padded row of up holds
  const larch_tensor gw = random(LARCH_Q4_K, {EXPERTS * EF, D});
  const bytes ub = quant(LARCH_Q4_K, EXPERTS * EF, nb * 256, g);
  const larch_tensor uw = {buffer(ub.data(), ub.size()), 0, LARCH_Q4_K, 2, {EXPERTS * EF, D}, {nb, 1}};
  const larch_tensor x = random(LARCH_F16, {rows, D}), id = ids(rows, SLOTS, EXPERTS);
  const larch_tensor gs = tensor(LARCH_F16, {rows, SLOTS, EF}), us = tensor(LARCH_F16, {rows, SLOTS, EF});
  const larch_tensor want = tensor(LARCH_F16, {rows, SLOTS, EF}), got = tensor(LARCH_F16, {rows, SLOTS, EF});
  larch_status u = larch_mmvq_id(st, &gw, &x, &id, nullptr, nullptr, &gs);
  if (u == LARCH_OK) u = larch_mmvq_id(st, &uw, &x, &id, nullptr, nullptr, &us);
  const larch_tensor g2 = {gs.buf, 0, LARCH_F16, 2, {rows * SLOTS, EF}, {EF, 1}},
                      u2 = {us.buf, 0, LARCH_F16, 2, {rows * SLOTS, EF}, {EF, 1}},
                      w2 = {want.buf, 0, LARCH_F16, 2, {rows * SLOTS, EF}, {EF, 1}};
  if (u == LARCH_OK) u = larch_silu_mul(st, &g2, &u2, &w2);
  const larch_status f = larch_fused_moe_gate_up(st, &gw, &uw, &x, &id, &got);
  judge(name, f, u, fetch(want), fetch(got), true);
  release();
}

// Entry 2: the shared expert's down (when st_ is a dtype), its gate and add, then the routed down and add.
static void moe_down(larch_dtype dt, int shared, int64_t rows) {
  const bool sh = shared >= 0;
  const std::string name = std::string("moe_down ") + DTYPE[dt] + " shared " + (sh ? DTYPE[shared] : "none") +
                           " rows " + std::to_string(rows);
  if (rows > 16 && sh) {  // the sequence's mmvq refuses it, and so must the entry
    const larch_tensor dw = random(dt, {EXPERTS * D, EF}), act = random(LARCH_F16, {rows, SLOTS, EF});
    const larch_tensor sw = random(larch_dtype(shared), {D, SF}), sa = random(LARCH_F16, {rows, SF});
    const larch_tensor sg = random(LARCH_F16, {rows, 1}), x = random(LARCH_F32, {rows, D});
    const larch_tensor id = ids(rows, SLOTS, EXPERTS), wt = weights(rows, SLOTS);
    CHECK(larch_fused_moe_down(st, &dw, &act, &id, &wt, &sw, &sa, &sg, &x) == LARCH_ERR_INVALID_ARG,
          "%s: more than 16 rows with a shared expert must be INVALID_ARG", name.c_str());
    release();
    return;
  }
  const larch_tensor dw = random(dt, {EXPERTS * D, EF}), act = random(LARCH_F16, {rows, SLOTS, EF});
  const larch_tensor id = ids(rows, SLOTS, EXPERTS), wt = weights(rows, SLOTS);
  const larch_tensor x = random(LARCH_F32, {rows, D}), want = copy(x), t = tensor(LARCH_F16, {rows, D});
  larch_tensor sw{}, sa{}, sg{};
  larch_status u = LARCH_OK;
  if (sh) {
    sw = random(larch_dtype(shared), {D, SF});
    sa = random(LARCH_F16, {rows, SF});
    sg = random(LARCH_F16, {rows, 1});
    u = larch_mmvq(st, &sw, &sa, nullptr, &t);
    if (u == LARCH_OK) u = larch_sigmoid_mul(st, &t, &sg, &t);
    if (u == LARCH_OK) u = larch_add(st, &want, &t, &want);
  }
  if (u == LARCH_OK) u = larch_mmvq_id(st, &dw, &act, &id, nullptr, &wt, &t);
  if (u == LARCH_OK) u = larch_add(st, &want, &t, &want);
  const larch_status f =
      larch_fused_moe_down(st, &dw, &act, &id, &wt, sh ? &sw : nullptr, sh ? &sa : nullptr, sh ? &sg : nullptr, &x);
  judge(name, f, u, fetch(want), fetch(x));
  release();
}

// The bytes of several outputs, one after the other.
static bytes concat(std::initializer_list<const larch_tensor*> ts) {
  bytes all;
  for (const larch_tensor* t : ts) {
    const bytes b = fetch(*t);
    all.insert(all.end(), b.begin(), b.end());
  }
  return all;
}

// Entry 3: rmsnorm, the router's gemm and moe_route, then (shared: a dtype) the shared expert's gate, up, silu_mul and
// its gate input's gemm.
static void moe_in(int shared, int64_t rows) {
  const bool sh = shared >= 0;
  const std::string name =
      std::string("moe_in shared ") + (sh ? DTYPE[shared] : "none") + " rows " + std::to_string(rows);
  const larch_tensor x = random(LARCH_F32, {rows, D}), nw = random(LARCH_F32, {D});
  const larch_tensor router = random(LARCH_F16, {EXPERTS, D});
  const auto outs = [&] {
    return std::vector<larch_tensor>{tensor(LARCH_F16, {rows, D}), tensor(LARCH_F32, {rows, EXPERTS}),
                                      tensor(LARCH_I32, {rows, SLOTS}), tensor(LARCH_F32, {rows, SLOTS}),
                                      tensor(LARCH_F16, {rows, SF}), tensor(LARCH_F16, {rows, 1})};
  };
  const std::vector<larch_tensor> want = outs(), got = outs();
  larch_tensor sg{}, su{}, gi{};
  if (sh) {
    sg = random(larch_dtype(shared), {SF, D});
    su = random(larch_dtype(shared), {SF, D});
    gi = random(LARCH_F16, {1, D});
  }
  if (rows > 16 && sh) {  // the sequence's mmvq refuses it, and so must the entry
    CHECK(larch_fused_moe_in(st, &x, &nw, 1e-6f, &got[0], &router, &got[1], &got[2], &got[3], &sg, &su, &got[4], &gi,
                              &got[5]) == LARCH_ERR_INVALID_ARG,
          "%s: more than 16 rows with a shared expert must be INVALID_ARG", name.c_str());
    release();
    return;
  }
  const larch_tensor g = tensor(LARCH_F16, {rows, SF}), u = tensor(LARCH_F16, {rows, SF});
  larch_status us = larch_rmsnorm(st, &x, &nw, 1e-6f, &want[0]);
  if (us == LARCH_OK) us = larch_gemm(st, &want[0], &router, nullptr, &want[1], 1, 0);
  if (us == LARCH_OK) us = larch_moe_route(st, &want[1], 0, &want[2], &want[3]);
  if (sh && us == LARCH_OK) us = larch_mmvq(st, &sg, &want[0], nullptr, &g);
  if (sh && us == LARCH_OK) us = larch_mmvq(st, &su, &want[0], nullptr, &u);
  if (sh && us == LARCH_OK) us = larch_silu_mul(st, &g, &u, &want[4]);
  if (sh && us == LARCH_OK) us = larch_gemm(st, &want[0], &gi, nullptr, &want[5], 1, 0);
  const larch_status f = sh ? larch_fused_moe_in(st, &x, &nw, 1e-6f, &got[0], &router, &got[1], &got[2], &got[3],
                                                   &sg, &su, &got[4], &gi, &got[5])
                             : larch_fused_moe_in(st, &x, &nw, 1e-6f, &got[0], &router, &got[1], &got[2], &got[3],
                                                   nullptr, nullptr, nullptr, nullptr, nullptr);
  const auto all = [&](const std::vector<larch_tensor>& t) {
    return sh ? concat({&t[0], &t[1], &t[2], &t[3], &t[4], &t[5]}) : concat({&t[0], &t[1], &t[2], &t[3]});
  };
  judge(name, f, us, all(want), all(got), rows > 8);  // the router's gemm
  release();
}

// Entry 4: rmsnorm, then qkv and z by mmvq and alpha and beta by mmvq (wq: alpha's dtype, beta's too unless f16) or
// gemm (f16).
static void gdn_in(larch_dtype wq, larch_dtype wa, int64_t rows) {
  const std::string name = std::string("gdn_in ") + DTYPE[wq] + "/" + DTYPE[wa] + " rows " + std::to_string(rows);
  constexpr int64_t CH = 768, VD = 256, VH = 8;
  const larch_tensor x = random(LARCH_F32, {rows, D}), nw = random(LARCH_F32, {D});
  const larch_tensor wqkv = random(wq, {CH, D}), wz = random(wq, {VD, D}), wal = random(wa, {VH, D}),
                      wbe = random(wa == LARCH_F16 ? wq : wa, {VH, D});
  const auto outs = [&] {
    return std::vector<larch_tensor>{tensor(LARCH_F16, {rows, D}), tensor(LARCH_F16, {rows, CH}),
                                      tensor(LARCH_F16, {rows, VD}), tensor(LARCH_F32, {rows, VH}),
                                      tensor(LARCH_F32, {rows, VH})};
  };
  const std::vector<larch_tensor> want = outs(), got = outs();
  const auto fused = [&] {
    return larch_fused_gdn_in(st, &x, &nw, 1e-6f, &got[0], &wqkv, &wz, &wal, &wbe, &got[1], &got[2], &got[3],
                               &got[4]);
  };
  if (rows > 16) {
    CHECK(fused() == LARCH_ERR_INVALID_ARG, "%s: more than 16 rows must be INVALID_ARG", name.c_str());
    release();
    return;
  }
  larch_status us = larch_rmsnorm(st, &x, &nw, 1e-6f, &want[0]);
  const larch_tensor* ws[4] = {&wqkv, &wz, &wal, &wbe};
  for (int i = 0; i < 4 && us == LARCH_OK; ++i)
    us = ws[i]->dtype == LARCH_F16 ? larch_gemm(st, &want[0], ws[i], nullptr, &want[1 + i], 1, 0)
                                    : larch_mmvq(st, ws[i], &want[0], nullptr, &want[1 + i]);
  const larch_status f = fused();
  judge(name, f, us, concat({&want[0], &want[1], &want[2], &want[3], &want[4]}),
        concat({&got[0], &got[1], &got[2], &got[3], &got[4]}), wa == LARCH_F16 && rows > 8);  // alpha's gemm
  release();
}

// Entries 6 and 8: (gate: sigmoid_mul,) mmvq and add onto the residual. gate 0 none, 1 per element, 2 per row.
static void proj_add(larch_dtype t, int gate, int64_t rows) {
  const std::string name =
      std::string("proj_add ") + DTYPE[t] + " gate " + std::to_string(gate) + " rows " + std::to_string(rows);
  const larch_tensor w = random(t, {D, EF}), in = random(LARCH_F16, {rows, EF});
  const larch_tensor gt = random(LARCH_F16, {rows, gate == 2 ? 1 : EF}), x = random(LARCH_F32, {rows, D});
  const larch_tensor want = copy(x), g = tensor(LARCH_F16, {rows, EF}), tt = tensor(LARCH_F16, {rows, D});
  const larch_tensor* gp = gate ? &gt : nullptr;
  if (rows > 16) {
    CHECK(larch_fused_proj_add(st, &w, &in, gp, &x) == LARCH_ERR_INVALID_ARG, "%s: more than 16 rows must be "
          "INVALID_ARG", name.c_str());
    release();
    return;
  }
  larch_status us = gate ? larch_sigmoid_mul(st, &in, gp, &g) : LARCH_OK;
  if (us == LARCH_OK) us = larch_mmvq(st, &w, gate ? &g : &in, nullptr, &tt);
  if (us == LARCH_OK) us = larch_add(st, &want, &tt, &want);
  const larch_status f = larch_fused_proj_add(st, &w, &in, gp, &x);
  judge(name, f, us, fetch(want), fetch(x));
  release();
}

// Entry 7: (gated: the 0/1 selector gemms,) qk_norm of q and k (norm), rope of both and kv_write; partial neox or full
// normal rope; row 1 is bucket padding (slot -1) when there are two rows or more.
static void attn_in(bool gated, bool norm, bool neox, int64_t hd, int64_t rows) {
  const std::string name = std::string("attn_in") + (gated ? " gated" : "") + (norm ? " normed" : "") +
                           (neox ? " neox" : " normal") + " hd " + std::to_string(hd) + " rows " + std::to_string(rows);
  constexpr int64_t H = 4, KVH = 2, BLOCKS = 4, BT = 16;
  const larch_tensor q_in = random(LARCH_F16, {rows, H, gated ? 2 * hd : hd}), k = random(LARCH_F16, {rows, KVH, hd});
  const larch_tensor v = random(LARCH_F16, {rows, KVH, hd}), qn = random(LARCH_F32, {hd}), kn = random(LARCH_F32, {hd});
  std::vector<int32_t> pos(static_cast<size_t>(rows)), slots(static_cast<size_t>(rows));
  for (int64_t r = 0; r < rows; ++r) {
    pos[size_t(r)] = int32_t(g() % 5000);
    slots[size_t(r)] = r == 1 ? -1 : int32_t(r * 2 + 1) % int32_t(BLOCKS * BT);
  }
  const larch_tensor pt = tensor(LARCH_I32, {rows}, pos.data()), sl = tensor(LARCH_I32, {rows}, slots.data());
  const auto pool_buf = [&] { return tensor(LARCH_F16, {2 * BLOCKS * BT, KVH, hd}, halves(size_t(2 * BLOCKS * BT * KVH * hd), g).data()); };
  const larch_tensor pk = pool_buf(), pv = pool_buf(), wk = copy(pk), wv = copy(pv);
  const larch_kv_pool got_pool = {pk.buf, pv.buf, BLOCKS, BT, 2, KVH, uint32_t(hd), LARCH_F16, 1},
                       want_pool = {wk.buf, wv.buf, BLOCKS, BT, 2, KVH, uint32_t(hd), LARCH_F16, 1};
  larch_rope_params p = {};
  p.n_dims = uint32_t(neox ? hd / 4 : hd);
  p.neox = neox;
  p.n_ctx_orig = 4096;
  p.freq_base = neox ? 1e7f : 1e4f;
  p.freq_scale = 1;
  p.attn_factor = 1;
  p.beta_fast = 32;
  p.beta_slow = 1;
  // want: the sequence on copies
  const larch_tensor wq = gated ? tensor(LARCH_F16, {rows, H, hd}) : copy(q_in), wg = tensor(LARCH_F16, {rows, H, hd});
  const larch_tensor wkc = copy(k), k0 = copy(k);
  larch_status us = LARCH_OK;
  if (gated) {
    bytes sel(size_t(2 * hd * 2 * hd) * 2, 0);
    for (int64_t half = 0; half < 2; ++half)
      for (int64_t i = 0; i < hd; ++i) sel[size_t((half * hd + i) * 2 * hd + half * hd + i) * 2 + 1] = 0x3c;
    const larch_tensor selt = tensor(LARCH_F16, {2 * hd, 2 * hd}, sel.data());
    const larch_tensor pr = {q_in.buf, 0, LARCH_F16, 2, {rows * H, 2 * hd}, {2 * hd, 1}};
    for (int i = 0; i < 2 && us == LARCH_OK; ++i) {
      const larch_tensor s1 = {selt.buf, uint64_t(i * hd * 2 * hd) * 2, LARCH_F16, 2, {hd, 2 * hd}, {2 * hd, 1}};
      const larch_tensor o = {(i ? wg : wq).buf, 0, LARCH_F16, 2, {rows * H, hd}, {hd, 1}};
      us = larch_gemm(st, &pr, &s1, nullptr, &o, 1, 0);
    }
  }
  if (norm && us == LARCH_OK) us = larch_qk_norm(st, &wq, &qn, 1e-6f);
  if (norm && us == LARCH_OK) us = larch_qk_norm(st, &wkc, &kn, 1e-6f);
  if (us == LARCH_OK) us = larch_rope(st, &wq, &pt, &p);
  if (us == LARCH_OK) us = larch_rope(st, &wkc, &pt, &p);
  if (us == LARCH_OK) us = larch_kv_write(st, &wkc, &v, &sl, &want_pool);
  // got
  const larch_tensor gq = gated ? tensor(LARCH_F16, {rows, H, hd}) : copy(q_in), gg = tensor(LARCH_F16, {rows, H, hd});
  const larch_tensor* qi = gated ? &q_in : &gq;
  const larch_status f = larch_fused_attn_in(st, qi, &k, &v, norm ? &qn : nullptr, norm ? &kn : nullptr, 1e-6f, &pt,
                                               &p, &sl, &got_pool, &gq, gated ? &gg : nullptr);
  const bytes kw = fetch(k0), kg = fetch(k);
  CHECK(kw == kg, "%s: k changed", name.c_str());
  judge(name, f, us, gated ? concat({&wq, &wg, &wk, &wv}) : concat({&wq, &wk, &wv}),
        gated ? concat({&gq, &gg, &pk, &pv}) : concat({&gq, &pk, &pv}), gated && rows > 8);  // the selector gemms
  release();
}

// Entry 5: gdn_conv, gdn, the per-head qk_norm and silu_mul by z, on one layer (of 2) of a state of `slots` slots,
// either `rows` decode sequences of a row each or one sequence of `rows` rows snapshotting every row but its last.
static void gdn(int64_t hd, int64_t rows, bool span) {
  const std::string name =
      std::string("gdn hd ") + std::to_string(hd) + (span ? " span" : " decode") + " rows " + std::to_string(rows);
  constexpr int64_t KH = 2, VH = 4, KERN = 4;
  const int64_t ch = (2 * KH + VH) * hd, vd = VH * hd, seqs = span ? 1 : rows, nslots = 2 * rows + 2;
  const int64_t conv_slot = (KERN - 1) * ch, ssm_slot = VH * hd * hd;
  const larch_tensor qkv = random(LARCH_F16, {rows, ch}), z = random(LARCH_F16, {rows, vd});
  std::vector<float> cw(size_t(ch * KERN)), al(size_t(rows * VH)), be(size_t(rows * VH)), dt(VH), av(VH), nwv(static_cast<size_t>(hd));
  std::uniform_real_distribution<float> u(-1, 1);
  for (float& x : cw) x = 0.25f + 0.2f * u(g);
  for (float& x : al) x = u(g);
  for (float& x : be) x = u(g);
  for (float& x : dt) x = 0.5f + 0.1f * u(g);
  for (float& x : av) x = -0.5f + 0.1f * u(g);
  for (float& x : nwv) x = 1 + 0.1f * u(g);
  const larch_tensor cwt = tensor(LARCH_F32, {ch, KERN}, cw.data()), alt = tensor(LARCH_F32, {rows, VH}, al.data()),
                      bet = tensor(LARCH_F32, {rows, VH}, be.data()), dtt = tensor(LARCH_F32, {VH}, dt.data()),
                      avt = tensor(LARCH_F32, {VH}, av.data()), nwt = tensor(LARCH_F32, {hd}, nwv.data());
  std::vector<int32_t> sl(static_cast<size_t>(seqs)), lens(static_cast<size_t>(seqs), span ? int32_t(rows) : 1),
      snaps(static_cast<size_t>(rows), -1);
  for (int64_t i = 0; i < seqs; ++i) sl[size_t(i)] = int32_t(nslots - 1 - i);
  if (span)
    for (int64_t r = 0; r + 1 < rows; ++r) snaps[size_t(r)] = int32_t(r);
  const larch_tensor slt = tensor(LARCH_I32, {seqs}, sl.data()), lt = tensor(LARCH_I32, {seqs}, lens.data()),
                      snt = tensor(LARCH_I32, {rows}, snaps.data());
  const larch_tensor conv0 = random(LARCH_F16, {2 * nslots, conv_slot});
  std::vector<float> ssm(size_t(2 * nslots * ssm_slot));
  for (float& x : ssm) x = 0.1f * u(g);
  const larch_tensor ssm0 = tensor(LARCH_F32, {2 * nslots, ssm_slot}, ssm.data());
  const larch_tensor wc = copy(conv0), ws = copy(ssm0), gc = copy(conv0), gs = copy(ssm0);
  const auto state = [&](const larch_tensor& c, const larch_tensor& s) {
    return larch_gdn_state{c.buf, s.buf, uint32_t(nslots), 2, 1, uint32_t(KERN), uint32_t(KH), uint32_t(VH), uint32_t(hd)};
  };
  const larch_gdn_state wst = state(wc, ws), gst = state(gc, gs);
  const larch_tensor* sn = span ? &snt : nullptr;
  const larch_tensor wconv = tensor(LARCH_F16, {rows, ch}), o2 = tensor(LARCH_F16, {rows, vd}),
                      wout = tensor(LARCH_F16, {rows, vd});
  const larch_tensor o3 = {o2.buf, 0, LARCH_F16, 3, {rows, VH, hd}, {VH * hd, hd, 1}};
  larch_status us = larch_gdn_conv(st, &qkv, &cwt, &wst, &slt, &lt, sn, &wconv);
  if (us == LARCH_OK) us = larch_gdn(st, &wconv, &alt, &bet, &dtt, &avt, 1e-6f, &wst, &slt, &lt, sn, &o2);
  if (us == LARCH_OK) us = larch_qk_norm(st, &o3, &nwt, 1e-6f);
  if (us == LARCH_OK) us = larch_silu_mul(st, &z, &o2, &wout);
  const larch_tensor gconv = tensor(LARCH_F16, {rows, ch}), gout = tensor(LARCH_F16, {rows, vd});
  const larch_status f = larch_fused_gdn(st, &qkv, &cwt, &alt, &bet, &dtt, &avt, 1e-6f, &gst, &slt, &lt, sn, &nwt, &z,
                                           &gconv, &gout);
  judge(name, f, us, concat({&wout, &wconv, &wc, &ws}), concat({&gout, &gconv, &gc, &gs}));
  release();
}

static bool pick(int32_t gpus, int32_t* ordinal) {
  const char* e = std::getenv("LARCH_TEST_DEVICE");
  if (!e || !*e) return *ordinal = gpus > 0 ? 0 : -1, true;
  char* end = nullptr;
  const long v = std::strtol(e, &end, 10);
  if (*end || v < -2 || v >= gpus) {
    std::fprintf(stderr, "LARCH_TEST_DEVICE=%s: want -1, -2 or a GPU ordinal below %d\n", e, gpus);
    return false;
  }
  return *ordinal = int32_t(v), true;
}

int main() {
  int32_t gpus = 0, ordinal = 0;
  if (larch_device_count(&gpus) != LARCH_OK || !pick(gpus, &ordinal)) return 1;
  if (larch_device_open(ordinal, &dev) != LARCH_OK || larch_stream_create(dev, &st) != LARCH_OK) {
    std::fprintf(stderr, "device %d: open failed: %s\n", ordinal, larch_last_error(dev));
    return 1;
  }
  gpu = ordinal >= 0;
  if (gpu) {
    larch_device_props props{};
    if (larch_device_props_get(dev, &props) != LARCH_OK) {
      std::fprintf(stderr, "device %d: props failed: %s\n", ordinal, larch_last_error(dev));
      return 1;
    }
    any_refusal = props.backend != LARCH_BACKEND_CUDA || props.arch_code == 86 || props.arch_code == 89;
  }
  for (const larch_dtype t : QUANTS)
    for (const int64_t r : ROWS) moe_gate_up(t, t, r);
  for (const int64_t r : {int64_t(1), int64_t(3)}) moe_gate_up(LARCH_Q3_K, LARCH_Q4_K, r);
  for (const int64_t r : {int64_t(1), int64_t(3)}) moe_gate_up_padded(r);
  for (const larch_dtype t : QUANTS)
    for (const int64_t r : ROWS) {
      moe_down(t, -1, r);
      for (const larch_dtype s : QUANTS) moe_down(t, s, r);
    }
  for (const larch_dtype t : QUANTS)
    for (const int64_t r : ROWS) {
      gdn_in(t, LARCH_F16, r);
      gdn_in(t, t, r);
    }
  // Quant-only gdn_in takes no gemm, so every row count up to 16 runs, through mmvq's per-row routes and splits.
  for (const larch_dtype t : QUANTS)
    for (int64_t r = 9; r < 16; ++r) gdn_in(t, t, r);
  for (const larch_dtype t : QUANTS)
    for (const int64_t r : ROWS)
      for (const int gate : {0, 1, 2}) proj_add(t, gate, r);
  for (const int64_t r : ROWS)
    for (const bool gated : {false, true})
      for (const bool norm : {false, true}) {
        attn_in(gated, norm, true, 256, r);
        attn_in(gated, norm, false, 64, r);
      }
  for (const int64_t hd : {32, 64, 128})
    for (const int64_t r : ROWS) {
      gdn(hd, r, false);
      gdn(hd, r, true);
    }
  for (const int64_t r : ROWS) {
    moe_in(-1, r);
    for (const larch_dtype s : QUANTS) moe_in(s, r);
  }
  // Edge shapes: k = 288 (a whole number of 32-weight blocks, not of 256) with 64 experts and top-8 routing ...
  D = 288, EF = 96, SF = 96, EXPERTS = 64, SLOTS = 8;
  for (const larch_dtype t : QUANTS32)
    for (const int64_t r : {int64_t(1), int64_t(3), int64_t(8)}) {
      moe_gate_up(t, t, r);
      moe_down(t, -1, r);
      moe_down(t, t, r);
      moe_in(t, r);
      for (const int gate : {0, 2}) proj_add(t, gate, r);
    }
  // ... an odd N: 255 expert columns for gate and up, 255 output rows for proj_add ...
  D = 512, EF = 255, SF = 256, EXPERTS = 8, SLOTS = 4;
  for (const larch_dtype t : QUANTS)
    for (const int64_t r : {int64_t(1), int64_t(3)}) moe_gate_up(t, t, r);
  D = 255, EF = 256;
  for (const larch_dtype t : QUANTS)
    for (const int64_t r : {int64_t(1), int64_t(3)}) proj_add(t, 1, r);
  // ... and head dims that are not powers of two.
  for (const int64_t hd : {80, 96})
    for (const int64_t r : {int64_t(1), int64_t(3), int64_t(8)})
      for (const bool gated : {false, true}) {
        attn_in(gated, true, true, hd, r);
        attn_in(gated, true, false, hd, r);
      }
  larch_stream_destroy(st);
  larch_device_close(dev);
  std::printf("fused on device %d: %d byte-equal, %d unsupported, %d failures\n", ordinal, passed, unsupported,
              failures);
  return failures ? 1 : 0;
}
