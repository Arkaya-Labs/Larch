// Inkling 2's entries (ABI 6) on GPU 0 against the CPU reference (ordinal -1) at Inkling 2 Mini's shapes: d 2,304,
// 24 q and 8 kv heads of 128, 128 experts of 1,152, a 1,024 window and extent, the 201,024-wide logits. Both sides get
// the same bytes; each output must be within its tolerance of the reference's, and the conv state bit for bit (it
// holds f16 inputs). tests/inkling.cpp checks the entries against larch.h's definitions at small shapes; this file
// checks the GPU kernels at the sizes, head counts and lengths the model runs. 77 without a GPU or where GPU 0 lacks
// an entry.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "testdata.h"

static int failures = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                            \
    }                                                                        \
  } while (0)
#define OK(call) CHECK((call) == LARCH_OK)

struct side {
  larch_device* dev = nullptr;
  larch_stream* st = nullptr;
  std::vector<larch_buffer*> owned;
};
static side sides[2];  // the CPU reference, then GPU 0
static rng g(29);
static bool lacking = false;

// One tensor on both sides, the same bytes (src) or zeros.
struct pair {
  larch_tensor t[2];
};
static pair tensor(larch_dtype dt, std::vector<int64_t> shape, const void* src, size_t n) {
  pair p;
  for (int i = 0; i < 2; ++i) {
    larch_buffer* b = nullptr;
    OK(larch_buffer_alloc(sides[i].dev, n, &b));
    const bytes zero(src ? 0 : n);
    OK(larch_upload(sides[i].st, b, 0, src ? src : zero.data(), n));
    sides[i].owned.push_back(b);
    larch_tensor& x = p.t[i];
    x = {b, 0, dt, uint32_t(shape.size()), {}, {}};
    for (int64_t d = int64_t(shape.size()) - 1, stride = 1; d >= 0; stride *= shape[size_t(d)], --d) {
      x.shape[d] = shape[size_t(d)];
      x.stride[d] = stride;
    }
  }
  return p;
}
static size_t elems(const larch_tensor& t) {
  size_t n = 1;
  for (uint32_t i = 0; i < t.ndim; ++i) n *= size_t(t.shape[i]);
  return n;
}
static size_t width(larch_dtype t) { return t == LARCH_F16 ? 2 : 4; }
static pair f32(std::vector<int64_t> shape, const std::vector<float>& v) {
  return tensor(LARCH_F32, shape, v.data(), v.size() * 4);
}
static pair f16(std::vector<int64_t> shape, const bytes& v) { return tensor(LARCH_F16, shape, v.data(), v.size()); }
static pair i32(std::vector<int64_t> shape, const std::vector<int32_t>& v) {
  return tensor(LARCH_I32, shape, v.data(), v.size() * 4);
}
static pair out(larch_dtype t, std::vector<int64_t> shape) {
  size_t n = 1;
  for (int64_t d : shape) n *= size_t(d);
  return tensor(t, shape, nullptr, n * width(t));
}
static bytes fetch(int i, const larch_buffer* b, size_t n) {
  bytes v(n);
  OK(larch_download(sides[i].st, b, 0, v.data(), n));
  OK(larch_stream_sync(sides[i].st));
  return v;
}
static float half_at(const uint8_t* p) {
  uint16_t h;
  std::memcpy(&h, p, 2);
  const int e = h >> 10 & 31, m = h & 1023;
  const float v = e == 31 ? (m ? NAN : INFINITY) : e ? std::ldexp(float(m + 1024), e - 25) : std::ldexp(float(m), -24);
  return h & 0x8000 ? -v : v;
}
static std::vector<float> values(int i, const larch_tensor& t) {
  const size_t n = elems(t), w = width(t.dtype);
  const bytes b = fetch(i, t.buf, n * w);
  std::vector<float> v(n);
  for (size_t k = 0; k < n; ++k)
    if (w == 2)
      v[k] = half_at(&b[2 * k]);
    else
      std::memcpy(&v[k], &b[4 * k], 4);
  return v;
}
static std::vector<float> uniform(size_t n, float lo, float hi) {
  std::uniform_real_distribution<float> u(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) x = u(g);
  return v;
}
// GPU 0's call status: UNSUPPORTED marks the entry missing (NOT RUN), anything else must be OK.
static bool ran(larch_status s) {
  if (s == LARCH_ERR_UNSUPPORTED) {
    std::fprintf(stderr, "inkling_shapes: GPU 0: %s\n", larch_last_error(sides[1].dev));
    lacking = true;
    return false;
  }
  OK(s);
  return s == LARCH_OK;
}
// Runs `call` on both sides (side i's tensors), the reference first; false when GPU 0 lacks it.
template <class F>
static bool both(F&& call) {
  OK(call(0));
  return ran(call(1));
}
// GPU 0's output within tol of the reference's, relative to 1 + |reference|; prints the worst.
static void near_ref(const char* what, const pair& o, double tol) {
  const std::vector<float> want = values(0, o.t[0]), got = values(1, o.t[1]);
  double worst = 0;
  size_t at = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double err = std::fabs(double(got[i]) - want[i]) / (1 + std::fabs(double(want[i])));
    if (!(err <= worst)) {  // NaN counts as the worst
      worst = err;
      at = i;
    }
  }
  std::printf("%-28s worst %.3g at %zu (tolerance %.3g)\n", what, worst, at, tol);
  CHECK(worst <= tol);
}
static void same_bytes(const char* what, const larch_buffer* const b[2], size_t n) {
  const bool eq = fetch(0, b[0], n) == fetch(1, b[1], n);
  std::printf("%-28s %s\n", what, eq ? "bit for bit" : "DIFFERS");
  CHECK(eq);
}

// The K and V convolutions of one layer: C 1,024 of a 2,048-channel state (V at offset 1,024), kernel 4, 8 dynamic
// heads, a 37-row prefill in slot 1 snapshotting after row 20 into slot 5, a decode row in slot 3.
static void short_conv(bool dyn, larch_dtype out_t) {
  constexpr int64_t C = 1024, ALL = 2048, OFF = 1024, K = 4, SLOTS = 6, HEADS = 8, ROWS = 38;
  const bytes state = halves(size_t(SLOTS * (K - 1) * ALL), g), x = halves(ROWS * C, g);
  std::vector<int32_t> snaps(ROWS, -1);
  snaps[20] = 5;
  const pair xt = f16({ROWS, C}, x), wt = f32({C, K}, uniform(C * K, -1, 1)),
             dt = f32({ROWS, HEADS, K}, uniform(ROWS * HEADS * K, -1, 1)), rt = f32({ROWS, C}, uniform(ROWS * C, -4, 4)),
             sl = i32({2}, {1, 3}), ln = i32({2}, {37, 1}), sn = i32({ROWS}, snaps), ot = out(out_t, {ROWS, C});
  larch_conv_state cs[2];
  for (int i = 0; i < 2; ++i) {
    larch_buffer* b = nullptr;
    OK(larch_buffer_alloc(sides[i].dev, state.size(), &b));
    OK(larch_upload(sides[i].st, b, 0, state.data(), state.size()));
    sides[i].owned.push_back(b);
    cs[i] = {b, uint32_t(SLOTS), 1, 0, uint32_t(K), uint32_t(ALL)};
  }
  if (!both([&](int i) {
        return larch_short_conv(sides[i].st, &xt.t[i], &wt.t[i], dyn ? &dt.t[i] : nullptr, &rt.t[i], &cs[i],
                                uint32_t(OFF), &sl.t[i], &ln.t[i], &sn.t[i], &ot.t[i]);
      }))
    return;
  near_ref(dyn ? "short_conv dyn" : "short_conv", ot, out_t == LARCH_F16 ? 2e-3 : 2e-5);
  const larch_buffer* const bufs[2] = {cs[0].conv, cs[1].conv};
  same_bytes("short_conv state", bufs, state.size());
}

// Q (24 heads, scaled, log-scaled positions to 40,000) and K (8 heads) clips, threshold 1.
static void clip_norm() {
  constexpr int64_t ROWS = 9;
  const std::vector<int32_t> pos = {0, 1, 511, 1023, 1024, 4095, 8191, 20000, 40000};
  const pair q = f16({ROWS, 24, 128}, halves(ROWS * 24 * 128, g, 2)), sc = f32({24, 128}, uniform(24 * 128, 0.25f, 4)),
             pt = i32({ROWS}, pos), qo = out(LARCH_F16, {ROWS, 24, 128});
  if (both([&](int i) { return larch_clip_norm(sides[i].st, &q.t[i], &sc.t[i], 1.0f, &pt.t[i], 1024, 0.5f, &qo.t[i]); }))
    near_ref("clip_norm q", qo, 2e-3);
  const pair k = f32({ROWS, 8, 128}, uniform(ROWS * 8 * 128, -0.2f, 0.2f)), ko = out(LARCH_F32, {ROWS, 8, 128});
  if (both([&](int i) { return larch_clip_norm(sides[i].st, &k.t[i], nullptr, 1.0f, nullptr, 0, 0, &ko.t[i]); }))
    near_ref("clip_norm k", ko, 2e-5);
}

// A full layer's depth mix: 7 states of d 2,304, 3 queries.
static void depth_mix() {
  constexpr int64_t N = 7, ROWS = 5, D = 2304, Q = 3;
  std::vector<float> s = uniform(N * ROWS * D, -1, 1);
  for (int64_t i = 0; i < ROWS * D; ++i) s[size_t(i)] *= 30;  // the embedding's state, far larger than the others
  const pair st = f32({N, ROWS, D}, s), qt = f32({Q, D}, uniform(Q * D, -0.1f, 0.1f)), ot = out(LARCH_F32, {ROWS, Q, D});
  if (both([&](int i) { return larch_depth_mix(sides[i].st, &st.t[i], &qt.t[i], &ot.t[i]); }))
    near_ref("depth_mix", ot, 1e-5);
}

// The gain and forget projections' outputs in f16 and in f32 (larch.h takes either).
static void loop_gate(larch_dtype gf) {
  constexpr int64_t ROWS = 5, D = 2304;
  const auto proj = [&] {
    return gf == LARCH_F16 ? f16({ROWS, D}, halves(ROWS * D, g, 2)) : f32({ROWS, D}, uniform(ROWS * D, -4, 4));
  };
  const pair x = f32({ROWS, D}, uniform(ROWS * D, -8, 8)), m = f32({ROWS, D}, uniform(ROWS * D, -8, 8)), gn = proj(),
             fg = proj(), ot = out(LARCH_F32, {ROWS, D});
  if (both([&](int i) { return larch_loop_gate(sides[i].st, &x.t[i], &m.t[i], &gn.t[i], &fg.t[i], &ot.t[i]); }))
    near_ref(gf == LARCH_F16 ? "loop_gate f16" : "loop_gate f32", ot, 1e-6);
}

// 5 rows of 8 experts of 1,152, f16 gate-up as the grouped expert matmul writes it, with the experts' scales.
static void situ_mul() {
  constexpr int64_t ROWS = 5, SLOTS = 8, FF = 1152, E = 128;
  std::vector<int32_t> ids(ROWS * SLOTS);
  for (int64_t r = 0; r < ROWS; ++r) {
    std::vector<int32_t> all(E);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), g);
    std::copy(all.begin(), all.begin() + SLOTS, ids.begin() + r * SLOTS);
  }
  const pair gu = f16({ROWS, SLOTS, 2 * FF}, halves(ROWS * SLOTS * 2 * FF, g, 5)), it = i32({ROWS, SLOTS}, ids),
             sc = f32({E}, uniform(E, 0.5f, 2)), ot = out(LARCH_F16, {ROWS, SLOTS, FF});
  if (both([&](int i) { return larch_situ_mul(sides[i].st, &gu.t[i], &it.t[i], &sc.t[i], 4.0f, 32.0f, &ot.t[i]); }))
    near_ref("situ_mul", ot, 2e-3);
}

// 128 experts, top 8, eps 1e-20, scale 8, with the selection bias and the down scales; ids must match exactly.
static void route() {
  constexpr int64_t ROWS = 33, E = 128, K = 8;
  std::normal_distribution<float> n(0, 3);
  std::vector<float> logits(ROWS * E);
  for (float& v : logits) v = n(g);
  const pair lt = f32({ROWS, E}, logits), bt = f32({E}, uniform(E, -0.2f, 0.2f)), dt = f32({E}, uniform(E, 0.5f, 2)),
             ids = out(LARCH_I32, {ROWS, K}), w = out(LARCH_F32, {ROWS, K});
  if (!both([&](int i) {
        return larch_moe_route_sigmoid(sides[i].st, &lt.t[i], &bt.t[i], 1e-20f, 8.0f, &dt.t[i], &ids.t[i], &w.t[i]);
      }))
    return;
  const larch_buffer* const bufs[2] = {ids.t[0].buf, ids.t[1].buf};
  same_bytes("moe_route_sigmoid ids", bufs, ROWS * K * 4);
  near_ref("moe_route_sigmoid weights", w, 1e-6);
}

// attn_rel over a pool of 16-token blocks, 8 kv heads of 128: a decode row at position 1,500, a 20-row prefill
// span at 30 to 49, and a 4-row draft tree at 900 (rows 1 and 2 children of row 0, row 3 of row 2), on a window
// layer (window and extent 1,024) and a full layer (no window, extent 1,024), at automatic and 64 splits.
static void attention(uint32_t window, uint32_t splits, bool tree) {
  constexpr int64_t BT = 16, KVH = 8, H = 24, HD = 128, NREL = 32, EXT = 1024, MAXB = 96, SEQS = 3, NB = SEQS * MAXB;
  const std::vector<int32_t> start = {1500, 30, 900}, lens = {1, 20, 4};
  const int64_t rows = 25;
  std::vector<int32_t> table(SEQS * MAXB);
  std::iota(table.begin(), table.end(), 0);
  std::shuffle(table.begin(), table.end(), g);
  // Every position each sequence's rows can see.
  std::vector<int32_t> slots;
  for (int64_t s = 0; s < SEQS; ++s)
    for (int64_t p = 0; p < start[size_t(s)] + lens[size_t(s)]; ++p)
      slots.push_back(table[size_t(s * MAXB + p / BT)] * int32_t(BT) + int32_t(p % BT));
  const int64_t n = int64_t(slots.size());
  const pair kt = f16({n, KVH, HD}, halves(size_t(n * KVH * HD), g)), vt = f16({n, KVH, HD}, halves(size_t(n * KVH * HD), g)),
             st = i32({n}, slots);
  larch_kv_pool pool[2];
  for (int i = 0; i < 2; ++i) {
    larch_buffer *k = nullptr, *v = nullptr;
    OK(larch_buffer_alloc(sides[i].dev, NB * BT * KVH * HD * 2, &k));
    OK(larch_buffer_alloc(sides[i].dev, NB * BT * KVH * HD * 2, &v));
    sides[i].owned.push_back(k);
    sides[i].owned.push_back(v);
    pool[i] = {k, v, uint32_t(NB), uint32_t(BT), 1, uint32_t(KVH), uint32_t(HD), LARCH_F16, 0};
    OK(larch_kv_write(sides[i].st, &kt.t[i], &vt.t[i], &st.t[i], &pool[i]));
  }
  std::vector<int32_t> mask(rows, 0);
  // The decode row and the prefill span causal, as bits; then the tree (0 -> {1, 2}, 2 -> 3), each row its
  // ancestors and itself.
  mask[0] = 1;
  for (int64_t r = 1; r < 21; ++r) mask[size_t(r)] = int32_t((1u << r) - 1);
  mask[21] = 0b0001;
  mask[22] = 0b0011;
  mask[23] = 0b0101;
  mask[24] = 0b1101;
  const bytes qb = halves(size_t(rows * H * HD), g);
  const std::vector<float> rqv = uniform(rows * H * NREL, -1, 1), skv = uniform(rows * H, -2, 2);
  const pair q = f16({rows, H, HD}, qb), bt = i32({SEQS, MAXB}, table), sp = i32({SEQS}, start),
             lp = i32({SEQS}, lens), mp = i32({rows}, mask), rq = f32({rows, H, NREL}, rqv),
             rw = f32({EXT, NREL}, uniform(EXT * NREL, -1, 1)), sk = f32({rows, H}, skv),
             ot = out(LARCH_F16, {rows, H, HD});
  if (!both([&](int i) {
        return larch_attn_rel(sides[i].st, &q.t[i], &pool[i], &bt.t[i], &sp.t[i], &lp.t[i], tree ? &mp.t[i] : nullptr,
                              &rq.t[i], &rw.t[i], &sk.t[i], window, splits, 1 / std::sqrt(128.0f), &ot.t[i]);
      }))
    return;
  const std::string what = std::string("attn_rel") + (window ? " window" : " full") + (tree ? " mask" : "") +
                           (splits ? " 64 splits" : "");
  near_ref(what.c_str(), ot, 2e-3);
  if (tree) return;
  // larch.h's decode form (q_lens NULL): each sequence's last row as a decode row of seq_len start + len, then a
  // padding row (seq_len 0). Each must be its span row's bytes on GPU 0, and the padding row zeros.
  const int64_t last[SEQS] = {0, 20, 24}, drows = SEQS + 1, row_q = H * HD * 2;
  bytes dq(size_t(drows * row_q), 0);
  std::vector<float> drq(size_t(drows * H * NREL), 0.5f), dsk(size_t(drows * H), 0.5f);
  std::vector<int32_t> dtable(size_t(drows * MAXB)), dlens(size_t(drows), 0);
  for (int64_t s = 0; s < SEQS; ++s) {
    const int64_t r = last[s];
    std::copy_n(qb.begin() + r * row_q, row_q, dq.begin() + s * row_q);
    std::copy_n(rqv.begin() + r * H * NREL, H * NREL, drq.begin() + s * H * NREL);
    std::copy_n(skv.begin() + r * H, H, dsk.begin() + s * H);
    std::copy_n(table.begin() + s * MAXB, MAXB, dtable.begin() + s * MAXB);
    dlens[size_t(s)] = start[size_t(s)] + lens[size_t(s)];
  }
  std::copy_n(table.begin(), MAXB, dtable.begin() + SEQS * MAXB);
  const pair dqt = f16({drows, H, HD}, dq), dbt = i32({drows, MAXB}, dtable), dl = i32({drows}, dlens),
             drqt = f32({drows, H, NREL}, drq), dskt = f32({drows, H}, dsk), dot = out(LARCH_F16, {drows, H, HD});
  if (!both([&](int i) {
        return larch_attn_rel(sides[i].st, &dqt.t[i], &pool[i], &dbt.t[i], &dl.t[i], nullptr, nullptr, &drqt.t[i],
                              &rw.t[i], &dskt.t[i], window, splits, 1 / std::sqrt(128.0f), &dot.t[i]);
      }))
    return;
  near_ref((what + " decode rows").c_str(), dot, 2e-3);
  const bytes span = fetch(1, ot.t[1].buf, size_t(rows * row_q)), dec = fetch(1, dot.t[1].buf, size_t(drows * row_q));
  bool eq = std::all_of(dec.begin() + SEQS * row_q, dec.end(), [](uint8_t b) { return b == 0; });
  for (int64_t s = 0; s < SEQS; ++s)
    eq = eq && std::equal(dec.begin() + s * row_q, dec.begin() + (s + 1) * row_q, span.begin() + last[s] * row_q);
  std::printf("%-28s %s\n", (what + " decode = span").c_str(), eq ? "bit for bit" : "DIFFERS");
  CHECK(eq);
}

static void softcap() {
  constexpr int64_t ROWS = 3, V = 201024;
  const pair x = f32({ROWS, V}, uniform(ROWS * V, -200, 200)), o = out(LARCH_F32, {ROWS, V});
  if (both([&](int i) { return larch_softcap(sides[i].st, &x.t[i], 30.0f, &o.t[i]); })) near_ref("softcap", o, 1e-6);
}

int main() {
  int32_t gpus = 0;
  if (larch_device_count(&gpus) != LARCH_OK || gpus < 1) {
    std::printf("inkling_shapes: no GPU: NOT RUN\n");
    return 77;
  }
  for (int i = 0; i < 2; ++i) {
    OK(larch_device_open(i ? 0 : -1, &sides[i].dev));
    OK(larch_stream_create(sides[i].dev, &sides[i].st));
  }
  if (failures) return 1;
  for (bool dyn : {false, true})
    for (larch_dtype t : {LARCH_F16, LARCH_F32}) short_conv(dyn, t);
  clip_norm();
  depth_mix();
  loop_gate(LARCH_F16);
  loop_gate(LARCH_F32);
  situ_mul();
  route();
  for (uint32_t window : {1024u, 0u})
    for (uint32_t splits : {0u, 64u}) attention(window, splits, false);
  attention(1024, 0, true);
  softcap();
  for (side& s : sides) {
    for (larch_buffer* b : s.owned) larch_buffer_free(b);
    larch_stream_destroy(s.st);
    larch_device_close(s.dev);
  }
  if (failures) std::fprintf(stderr, "inkling_shapes: %d failures\n", failures);
  return failures ? 1 : lacking ? 77 : 0;
}
