// Inkling 2's entries (ABI 6) on the CPU reference, then (inkling_gpu) on GPU 0 where it has them: each against
// larch.h's definition in double, then what chunked prefill, draft rollback and decode rely on, bit for bit: a
// sequence split over short_conv calls gives the outputs and state of one call, a snapshot is the live state after
// its row, other channels and slots are left alone, and attn_rel's rows of a span equal each row as its own span.
// The model against llama.cpp is crates/larch-model's tests/inkling.rs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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

static larch_device* dev;
static larch_stream* st;
static std::vector<larch_buffer*> owned;
static rng g(23);
// A GPU run may lack an entry: each test then stops at its first call, NOT RUN.
static bool lacking = false;
static bool run(larch_status s) {
  if (s == LARCH_ERR_UNSUPPORTED) {
    lacking = true;
    return false;
  }
  OK(s);
  return s == LARCH_OK;
}

static larch_buffer* buffer(const void* src, size_t n) {
  larch_buffer* b = nullptr;
  OK(larch_buffer_alloc(dev, n, &b));
  if (src) OK(larch_upload(st, b, 0, src, n));
  owned.push_back(b);
  return b;
}
static larch_tensor tensor(larch_dtype t, std::vector<int64_t> shape, const void* src, size_t n) {
  larch_tensor x = {buffer(src, n), 0, t, uint32_t(shape.size()), {}, {}};
  for (int64_t i = int64_t(shape.size()) - 1, stride = 1; i >= 0; stride *= shape[size_t(i)], --i) {
    x.shape[i] = shape[size_t(i)];
    x.stride[i] = stride;
  }
  return x;
}
static size_t elems(const larch_tensor& t) {
  size_t n = 1;
  for (uint32_t i = 0; i < t.ndim; ++i) n *= size_t(t.shape[i]);
  return n;
}
static larch_tensor f32(std::vector<int64_t> shape, const std::vector<float>& v) {
  return tensor(LARCH_F32, shape, v.data(), v.size() * 4);
}
static larch_tensor i32(std::vector<int64_t> shape, const std::vector<int32_t>& v) {
  return tensor(LARCH_I32, shape, v.data(), v.size() * 4);
}
static larch_tensor out(larch_dtype t, std::vector<int64_t> shape) {
  size_t n = 1;
  for (int64_t d : shape) n *= size_t(d);
  return tensor(t, shape, nullptr, n * (t == LARCH_F16 ? 2 : 4));
}
static bytes fetch(larch_buffer* b, size_t n) {
  bytes v(n);
  OK(larch_download(st, b, 0, v.data(), n));
  OK(larch_stream_sync(st));
  return v;
}
static float half(const uint8_t* p) {
  uint16_t h;
  std::memcpy(&h, p, 2);
  const int e = h >> 10 & 31, m = h & 1023;
  const float v = e ? std::ldexp(float(m + 1024), e - 25) : std::ldexp(float(m), -24);  // finite halves only
  return h & 0x8000 ? -v : v;
}
// A tensor's values as floats, f16 or f32.
static std::vector<float> values(const larch_tensor& t) {
  const size_t n = elems(t), w = t.dtype == LARCH_F16 ? 2 : 4;
  const bytes b = fetch(t.buf, n * w);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i)
    if (w == 2)
      v[i] = half(&b[2 * i]);
    else
      std::memcpy(&v[i], &b[4 * i], 4);
  return v;
}
static std::vector<float> uniform(size_t n, float lo, float hi) {
  std::uniform_real_distribution<float> u(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) x = u(g);
  return v;
}
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * (1 + std::fabs(b)); }
// The tolerance of an output dtype against a definition in double.
static double tol(larch_dtype t) { return t == LARCH_F16 ? 2e-3 : 2e-5; }

// short_conv: two sequences, A 3 rows in slot 2 snapshotting after its row 1 into slot 3, B 2 rows in slot 0, over
// channels [OFF, OFF + C) of a state of ALL channels, 4 slots, 2 layers (layer 1 used).
namespace conv {
constexpr int64_t C = 16, ALL = 40, OFF = 8, K = 4, SLOTS = 4, HEADS = 4, ROWS = 5;
constexpr int64_t SLOT_ELEMS = (K - 1) * ALL;
static const std::vector<int32_t> SLOT = {2, 0}, LEN = {3, 2}, SNAP = {-1, 3, -1, -1, -1};

static larch_conv_state state(const bytes& c) {
  return {buffer(c.data(), c.size()), SLOTS, 2, 1, uint32_t(K), uint32_t(ALL)};
}
// Element i of slot s's row in layer 1 of state bytes c.
static float at(const bytes& c, int64_t s, int64_t i) { return half(&c[size_t(((SLOTS + s) * SLOT_ELEMS + i) * 2)]); }

static void by_definition(bool dyn, bool residual, larch_dtype out_t) {
  const bytes c0 = halves(size_t(2 * SLOTS * SLOT_ELEMS), g), x = halves(ROWS * C, g);
  const std::vector<float> w = uniform(C * K, -1, 1), d = uniform(ROWS * HEADS * K, -1, 1),
                           res = uniform(ROWS * C, -2, 2);
  larch_conv_state s = state(c0);
  const larch_tensor xt = tensor(LARCH_F16, {ROWS, C}, x.data(), x.size()), wt = f32({C, K}, w),
                     dt = f32({ROWS, HEADS, K}, d), rt = f32({ROWS, C}, res);
  const larch_tensor sl = i32({2}, SLOT), ln = i32({2}, LEN), sn = i32({ROWS}, SNAP), ot = out(out_t, {ROWS, C});
  if (!run(larch_short_conv(st, &xt, &wt, dyn ? &dt : nullptr, residual ? &rt : nullptr, &s, uint32_t(OFF), &sl, &ln,
                            &sn, &ot)))
    return;
  const std::vector<float> o = values(ot);
  const bytes c1 = fetch(s.conv, c0.size());
  for (int seq = 0, r0 = 0; seq < 2; r0 += LEN[size_t(seq)], ++seq) {
    // in[t][c]: the slot's 3 inputs, then the sequence's rows.
    std::vector<std::vector<double>> in;
    for (int64_t t = 0; t < K - 1; ++t) {
      in.emplace_back();
      for (int64_t c = 0; c < C; ++c) in.back().push_back(at(c0, SLOT[size_t(seq)], t * ALL + OFF + c));
    }
    for (int t = 0; t < LEN[size_t(seq)]; ++t) {
      in.emplace_back();
      for (int64_t c = 0; c < C; ++c) in.back().push_back(half(&x[size_t(((r0 + t) * C + c) * 2)]));
    }
    for (int t = 0; t < LEN[size_t(seq)]; ++t) {
      const int64_t r = r0 + t;
      for (int64_t c = 0; c < C; ++c) {
        double y = 0;
        for (int64_t i = 0; i < K; ++i) y += w[size_t(c * K + i)] * in[size_t(t + i)][size_t(c)];
        for (int64_t lag = 0; dyn && lag < K; ++lag)
          y += in[size_t(t + K - 1 - lag)][size_t(c)] * d[size_t((r * HEADS + c / (C / HEADS)) * K + lag)];
        y = (in[size_t(K - 1 + t)][size_t(c)] + y) / std::sqrt(double((dyn ? 2 : 1) * K + 1));
        if (residual) y += res[size_t(r * C + c)];
        CHECK(near(o[size_t(r * C + c)], y, tol(out_t)));
      }
    }
    // The live slot holds the last 3 inputs at these channels; A's snapshot the 3 before its row 2.
    for (int64_t t = 0; t < K - 1; ++t)
      for (int64_t c = 0; c < C; ++c) {
        CHECK(at(c1, SLOT[size_t(seq)], t * ALL + OFF + c) == float(in[size_t(LEN[size_t(seq)] + t)][size_t(c)]));
        if (seq == 0) CHECK(at(c1, 3, t * ALL + OFF + c) == float(in[size_t(2 + t)][size_t(c)]));
      }
  }
  // Every other channel of every slot, the slot nobody named and layer 0 stay as they were.
  for (int64_t sl2 = 0; sl2 < SLOTS; ++sl2)
    for (int64_t t = 0; t < K - 1; ++t)
      for (int64_t c = 0; c < ALL; ++c)
        if (sl2 == 1 || c < OFF || c >= OFF + C) CHECK(at(c1, sl2, t * ALL + c) == at(c0, sl2, t * ALL + c));
  CHECK(std::memcmp(c1.data(), c0.data(), size_t(SLOTS * SLOT_ELEMS * 2)) == 0);
}

// A in one call, and in two (row 0, then rows 1 and 2), from the same state: the same bytes.
static void split() {
  const bytes c0 = halves(size_t(2 * SLOTS * SLOT_ELEMS), g), x = halves(3 * C, g);
  const std::vector<float> w = uniform(C * K, -1, 1), d = uniform(3 * HEADS * K, -1, 1), res = uniform(3 * C, -2, 2);
  larch_conv_state one = state(c0), two = state(c0);
  const larch_tensor a = tensor(LARCH_F16, {3, C}, x.data(), x.size()), wt = f32({C, K}, w),
                     dt = f32({3, HEADS, K}, d), rt = f32({3, C}, res), s2 = i32({1}, {2});
  const auto rows = [](larch_tensor t, int64_t r0, int64_t n) {
    t.offset += uint64_t(r0 * t.stride[0] * (t.dtype == LARCH_F16 ? 2 : 4));
    t.shape[0] = n;
    return t;
  };
  const larch_tensor l3 = i32({1}, {3}), l1 = i32({1}, {1}), l2 = i32({1}, {2});
  const larch_tensor o3 = out(LARCH_F32, {3, C}), q3 = out(LARCH_F32, {3, C});
  if (!run(larch_short_conv(st, &a, &wt, &dt, &rt, &one, uint32_t(OFF), &s2, &l3, nullptr, &o3))) return;
  const larch_tensor a0 = rows(a, 0, 1), a12 = rows(a, 1, 2), d0 = rows(dt, 0, 1), d12 = rows(dt, 1, 2),
                     r0 = rows(rt, 0, 1), r12 = rows(rt, 1, 2), p0 = rows(q3, 0, 1), p12 = rows(q3, 1, 2);
  OK(larch_short_conv(st, &a0, &wt, &d0, &r0, &two, uint32_t(OFF), &s2, &l1, nullptr, &p0));
  OK(larch_short_conv(st, &a12, &wt, &d12, &r12, &two, uint32_t(OFF), &s2, &l2, nullptr, &p12));
  CHECK(fetch(o3.buf, 3 * C * 4) == fetch(q3.buf, 3 * C * 4));
  CHECK(fetch(one.conv, c0.size()) == fetch(two.conv, c0.size()));
  // In place onto the residual: the bytes of a separate out.
  const larch_tensor in_place = f32({3, C}, res);
  OK(larch_short_conv(st, &a, &wt, &dt, &in_place, &two, uint32_t(OFF), &s2, &l3, nullptr, &in_place));
  OK(larch_short_conv(st, &a, &wt, &dt, &rt, &one, uint32_t(OFF), &s2, &l3, nullptr, &o3));
  CHECK(fetch(o3.buf, 3 * C * 4) == fetch(in_place.buf, 3 * C * 4));
}
}  // namespace conv

// clip_norm: rows whose heads clip and rows whose heads do not, at three positions, with and without the scale.
static void clip(bool scaled, bool positions, larch_dtype out_t) {
  constexpr int64_t ROWS = 4, H = 2, D = 8;
  constexpr float T = 1.5f, ALPHA = 0.25f;
  constexpr uint32_t FLOOR = 8;
  std::vector<float> x = uniform(ROWS * H * D, -1, 1);
  for (int64_t i = 0; i < H * D; ++i) x[size_t(i)] *= 0.1f;  // row 0 well inside the threshold
  const std::vector<float> sc = uniform(H * D, 0.5f, 2.0f);
  const std::vector<int32_t> pos = {0, 7, 100, 9000};
  const larch_tensor xt = f32({ROWS, H, D}, x), stn = f32({H, D}, sc), pt = i32({ROWS}, pos),
                     ot = out(out_t, {ROWS, H, D});
  if (!run(larch_clip_norm(st, &xt, scaled ? &stn : nullptr, T, positions ? &pt : nullptr, FLOOR, ALPHA, &ot))) return;
  const std::vector<float> o = values(ot);
  for (int64_t r = 0; r < ROWS; ++r) {
    const double tau = positions ? std::log(std::max((pos[size_t(r)] + 1.0) / FLOOR, 1.0)) * ALPHA + 1 : 1;
    for (int64_t h = 0; h < H; ++h) {
      double n = 0;
      std::vector<double> y(D);
      for (int64_t i = 0; i < D; ++i) {
        y[size_t(i)] = x[size_t((r * H + h) * D + i)] * (scaled ? sc[size_t(h * D + i)] : 1.0f);
        n += y[size_t(i)] * y[size_t(i)];
      }
      n = std::sqrt(n);
      double norm = 0;
      for (int64_t i = 0; i < D; ++i) {
        const double want = y[size_t(i)] / std::max(n, double(T)) * T * tau, got = o[size_t((r * H + h) * D + i)];
        CHECK(near(got, want, tol(out_t)));
        norm += got * got;
      }
      CHECK(std::sqrt(norm) <= T * tau * (1 + tol(out_t)));
    }
  }
}

// depth_mix: three states of different scales, two queries.
static void mix() {
  constexpr int64_t N = 3, ROWS = 2, D = 16, Q = 2;
  std::vector<float> s = uniform(N * ROWS * D, -1, 1);
  for (int64_t i = 0; i < ROWS * D; ++i) s[size_t(i)] *= 40;  // a large state, which the norm takes out
  const std::vector<float> q = uniform(Q * D, -3, 3);
  const larch_tensor stt = f32({N, ROWS, D}, s), qt = f32({Q, D}, q), ot = out(LARCH_F32, {ROWS, Q, D});
  if (!run(larch_depth_mix(st, &stt, &qt, &ot))) return;
  const std::vector<float> o = values(ot);
  for (int64_t r = 0; r < ROWS; ++r) {
    std::vector<std::vector<double>> n(N, std::vector<double>(D));
    for (int64_t j = 0; j < N; ++j) {
      double ss = 0;
      for (int64_t k = 0; k < D; ++k) ss += double(s[size_t((j * ROWS + r) * D + k)]) * s[size_t((j * ROWS + r) * D + k)];
      for (int64_t k = 0; k < D; ++k) n[size_t(j)][size_t(k)] = s[size_t((j * ROWS + r) * D + k)] / std::sqrt(ss / D + 1e-6);
    }
    for (int64_t i = 0; i < Q; ++i) {
      std::vector<double> p(N);
      for (int64_t j = 0; j < N; ++j) {
        for (int64_t k = 0; k < D; ++k) p[size_t(j)] += n[size_t(j)][size_t(k)] * q[size_t(i * D + k)];
        p[size_t(j)] /= std::sqrt(double(D));
      }
      const double mx = *std::max_element(p.begin(), p.end());
      double total = 0;
      for (double& v : p) total += v = std::exp(v - mx);
      for (int64_t k = 0; k < D; ++k) {
        double want = 0;
        for (int64_t j = 0; j < N; ++j) want += p[size_t(j)] / total * n[size_t(j)][size_t(k)];
        CHECK(near(o[size_t((r * Q + i) * D + k)], want, 2e-5));
      }
    }
  }
}

static void gate(larch_dtype gf) {
  constexpr int64_t ROWS = 3, D = 16;
  const std::vector<float> x = uniform(ROWS * D, -2, 2), m = uniform(ROWS * D, -2, 2);
  const bytes gn = halves(ROWS * D, g), fg = halves(ROWS * D, g);
  std::vector<float> gnf(ROWS * D), fgf(ROWS * D);
  for (int64_t i = 0; i < ROWS * D; ++i) {
    gnf[size_t(i)] = half(&gn[size_t(2 * i)]);
    fgf[size_t(i)] = half(&fg[size_t(2 * i)]);
  }
  const larch_tensor gt = gf == LARCH_F16 ? tensor(LARCH_F16, {ROWS, D}, gn.data(), gn.size()) : f32({ROWS, D}, gnf),
                     ft = gf == LARCH_F16 ? tensor(LARCH_F16, {ROWS, D}, fg.data(), fg.size()) : f32({ROWS, D}, fgf);
  const larch_tensor xt = f32({ROWS, D}, x), mt = f32({ROWS, D}, m), ot = out(LARCH_F32, {ROWS, D});
  if (!run(larch_loop_gate(st, &xt, &mt, &gt, &ft, &ot))) return;
  const std::vector<float> o = values(ot);
  for (int64_t i = 0; i < ROWS * D; ++i) {
    const double want = x[size_t(i)] * (1 - std::tanh(double(fgf[size_t(i)]))) + std::tanh(double(gnf[size_t(i)])) * m[size_t(i)];
    CHECK(near(o[size_t(i)], want, 2e-5));
  }
  OK(larch_loop_gate(st, &xt, &mt, &gt, &ft, &xt));  // in place: the bytes of a separate out
  CHECK(fetch(xt.buf, ROWS * D * 4) == fetch(ot.buf, ROWS * D * 4));
}

// situ_mul on saturating inputs, with and without the experts' scales.
static void situ(bool scaled) {
  constexpr int64_t ROWS = 2, SLOTS = 3, FF = 4, E = 5;
  constexpr float BETA = 4.0f, LIN = 25.0f;
  const std::vector<float> gu = uniform(ROWS * SLOTS * 2 * FF, -30, 30), sc = uniform(E, 0.25f, 2.0f);
  const std::vector<int32_t> ids = {4, 0, 2, 1, 4, 3};
  const larch_tensor gt = f32({ROWS, SLOTS, 2 * FF}, gu), it = i32({ROWS, SLOTS}, ids), stn = f32({E}, sc),
                     ot = out(LARCH_F16, {ROWS, SLOTS, FF});
  if (!run(larch_situ_mul(st, &gt, &it, scaled ? &stn : nullptr, BETA, LIN, &ot))) return;
  const std::vector<float> o = values(ot);
  for (int64_t r = 0; r < ROWS * SLOTS; ++r) {
    const double s = scaled ? sc[size_t(ids[size_t(r)])] : 1;
    for (int64_t k = 0; k < FF; ++k) {
      const double gg = gu[size_t(r * 2 * FF + k)] * s, u = gu[size_t(r * 2 * FF + FF + k)] * s;
      const double want = BETA * std::tanh(gg / BETA) / (1 + std::exp(-gg)) * LIN * std::tanh(u / LIN);
      CHECK(near(o[size_t(r * FF + k)], want, 2e-3));
    }
  }
}

// moe_route_sigmoid: the bias decides the choice, the unbiased sigmoid the weights; equal biased scores go to the
// lower id.
static void route() {
  constexpr int64_t E = 6, K = 3;
  const std::vector<float> logits = {0.5f, -1.0f, 2.0f, 0.0f, 0.5f, -3.0f, 1.0f, 1.0f, 1.0f, 1.0f, -2.0f, 3.0f};
  const std::vector<float> bias = {0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f}, down = {1, 2, 3, 4, 5, 6};
  const larch_tensor lt = f32({2, E}, logits), bt = f32({E}, bias), dt = f32({E}, down), ids = out(LARCH_I32, {2, K}),
                     w = out(LARCH_F32, {2, K});
  if (!run(larch_moe_route_sigmoid(st, &lt, &bt, 1e-20f, 8.0f, &dt, &ids, &w))) return;
  const bytes ib = fetch(ids.buf, 2 * K * 4);
  const std::vector<float> wv = values(w);
  std::vector<int32_t> got(2 * K);
  std::memcpy(got.data(), ib.data(), ib.size());
  // Row 0: sigmoid + bias puts 1 (-1 + 2: 2.27), then 2, then 0 and 4 tie: the lower id, 0.
  // Row 1: 1 (1 + 2), then 5, then 0, 2 and 3 tie: 0.
  CHECK((got == std::vector<int32_t>{1, 2, 0, 1, 5, 0}));
  const auto sig = [](double x) { return 1 / (1 + std::exp(-x)); };
  for (int r = 0; r < 2; ++r) {
    double sum = 0;
    for (int j = 0; j < K; ++j) sum += sig(logits[size_t(r * E + got[size_t(r * K + j)])]);
    for (int j = 0; j < K; ++j) {
      const int e = got[size_t(r * K + j)];
      CHECK(near(wv[size_t(r * K + j)], sig(logits[size_t(r * E + e)]) / (sum + 1e-20) * 8 * down[size_t(e)], 2e-6));
    }
  }
}

// attn_rel: one sequence of 10 positions over scattered blocks, 2 q heads over 1 kv head; a span of rows 6 to 9,
// whole and windowed, against the definition, then each row alone as its own span, bit for bit.
static void attention(uint32_t window) {
  constexpr int64_t BT = 4, NB = 8, HD = 8, H = 2, NREL = 4, EXT = 5, LEN = 10, P0 = 6, ROWS = LEN - P0;
  constexpr float SCALE = 0.35f;
  const std::vector<int32_t> table = {3, 1, 0, 2};
  const bytes k = halves(LEN * HD, g), v = halves(LEN * HD, g), q = halves(ROWS * H * HD, g);
  const std::vector<float> rq = uniform(ROWS * H * NREL, -1, 1), rw = uniform(EXT * NREL, -1, 1),
                           sink = uniform(ROWS * H, -1, 1);
  larch_kv_pool pool = {buffer(nullptr, NB * BT * HD * 2), buffer(nullptr, NB * BT * HD * 2), uint32_t(NB), uint32_t(BT),
                        1, 1, uint32_t(HD), LARCH_F16, 0};
  std::vector<int32_t> slots(LEN);
  for (int64_t p = 0; p < LEN; ++p) slots[size_t(p)] = int32_t(table[size_t(p / BT)] * BT + p % BT);
  const larch_tensor kt = tensor(LARCH_F16, {LEN, 1, HD}, k.data(), k.size()),
                     vt = tensor(LARCH_F16, {LEN, 1, HD}, v.data(), v.size()), st_ = i32({LEN}, slots);
  OK(larch_kv_write(st, &kt, &vt, &st_, &pool));
  const larch_tensor qt = tensor(LARCH_F16, {ROWS, H, HD}, q.data(), q.size()), bt = i32({1, 4}, table),
                     start = i32({1}, {int32_t(P0)}), lens = i32({1}, {int32_t(ROWS)}), rqt = f32({ROWS, H, NREL}, rq),
                     rwt = f32({EXT, NREL}, rw), skt = f32({ROWS, H}, sink), ot = out(LARCH_F16, {ROWS, H, HD});
  if (!run(larch_attn_rel(st, &qt, &pool, &bt, &start, &lens, nullptr, &rqt, &rwt, &skt, window, 0, SCALE, &ot)))
    return;
  const std::vector<float> o = values(ot);
  for (int64_t j = 0; j < ROWS; ++j) {
    const int64_t p = P0 + j;
    for (int64_t h = 0; h < H; ++h) {
      std::vector<double> l;
      for (int64_t pp = window ? std::max<int64_t>(0, p - window + 1) : 0; pp <= p; ++pp) {
        double dot = 0, b = 0;
        for (int64_t d = 0; d < HD; ++d) dot += double(half(&q[size_t(((j * H + h) * HD + d) * 2)])) * half(&k[size_t((pp * HD + d) * 2)]);
        if (p - pp < EXT)
          for (int64_t i = 0; i < NREL; ++i) b += double(rq[size_t((j * H + h) * NREL + i)]) * rw[size_t((p - pp) * NREL + i)];
        l.push_back(SCALE * dot + (b - sink[size_t(j * H + h)]));
      }
      const double mx = std::max(0.0, *std::max_element(l.begin(), l.end()));
      double total = std::exp(-mx);  // the sink's logit 0
      for (double& x : l) total += x = std::exp(x - mx);
      for (int64_t d = 0; d < HD; ++d) {
        double want = 0;
        const int64_t first = window ? std::max<int64_t>(0, p - window + 1) : 0;
        for (size_t i = 0; i < l.size(); ++i) want += l[i] / total * half(&v[size_t(((first + int64_t(i)) * HD + d) * 2)]);
        CHECK(near(o[size_t((j * H + h) * HD + d)], want, 2e-3));
      }
    }
  }
  // Each row as its own span (a decode step's form): the span's bytes.
  const bytes whole = fetch(ot.buf, ROWS * H * HD * 2);
  for (int64_t j = 0; j < ROWS; ++j) {
    larch_tensor qj = qt, rqj = rqt, skj = skt;
    qj.offset = uint64_t(j * H * HD * 2);
    rqj.offset = uint64_t(j * H * NREL * 4);
    skj.offset = uint64_t(j * H * 4);
    qj.shape[0] = rqj.shape[0] = skj.shape[0] = 1;
    const larch_tensor sj = i32({1}, {int32_t(P0 + j)}), one = i32({1}, {1}), oj = out(LARCH_F16, {1, H, HD});
    OK(larch_attn_rel(st, &qj, &pool, &bt, &sj, &one, nullptr, &rqj, &rwt, &skj, window, 0, SCALE, &oj));
    CHECK(std::memcmp(fetch(oj.buf, H * HD * 2).data(), &whole[size_t(j * H * HD * 2)], size_t(H * HD * 2)) == 0);
  }
  // Decode's form: the rows as decode rows of 2 sequences each (seq_len p + 1), and a padding row (seq_len 0) of
  // zeros; the span's bytes again.
  const std::vector<int32_t> tables = {3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2};
  const larch_tensor bt5 = i32({ROWS + 1, 4}, tables), od = out(LARCH_F16, {ROWS + 1, H, HD});
  const larch_tensor sl = i32({ROWS + 1}, {int32_t(P0 + 1), int32_t(P0 + 2), int32_t(P0 + 3), int32_t(P0 + 4), 0});
  const bytes q5 = [&] {
    bytes b(q);
    b.resize(size_t((ROWS + 1) * H * HD * 2), 0x3c);
    return b;
  }();
  std::vector<float> rq5(rq), sink5(sink);
  rq5.resize(size_t((ROWS + 1) * H * NREL), 0.5f);
  sink5.resize(size_t((ROWS + 1) * H), 0.5f);
  const larch_tensor q5t = tensor(LARCH_F16, {ROWS + 1, H, HD}, q5.data(), q5.size()),
                     rq5t = f32({ROWS + 1, H, NREL}, rq5), sk5t = f32({ROWS + 1, H}, sink5);
  OK(larch_attn_rel(st, &q5t, &pool, &bt5, &sl, nullptr, nullptr, &rq5t, &rwt, &sk5t, window, 0, SCALE, &od));
  const bytes dec = fetch(od.buf, size_t((ROWS + 1) * H * HD * 2));
  CHECK(std::memcmp(dec.data(), whole.data(), whole.size()) == 0);
  CHECK(std::all_of(dec.begin() + std::ptrdiff_t(whole.size()), dec.end(), [](uint8_t x) { return x == 0; }));
}

static void cap() {
  const std::vector<float> x = uniform(64, -200, 200);
  const larch_tensor xt = f32({2, 32}, x), ot = out(LARCH_F32, {2, 32});
  if (!run(larch_softcap(st, &xt, 30.0f, &ot))) return;
  const std::vector<float> o = values(ot);
  for (size_t i = 0; i < x.size(); ++i) CHECK(near(o[i], 30 * std::tanh(x[i] / 30.0), 1e-6));
}

static void all() {
  for (bool dyn : {false, true})
    for (bool residual : {false, true})
      for (larch_dtype t : {LARCH_F16, LARCH_F32}) conv::by_definition(dyn, residual, t);
  conv::split();
  for (bool scaled : {false, true})
    for (bool positions : {false, true})
      for (larch_dtype t : {LARCH_F16, LARCH_F32}) clip(scaled, positions, t);
  mix();
  gate(LARCH_F16);
  gate(LARCH_F32);
  situ(false);
  situ(true);
  route();
  attention(0);
  attention(3);
  cap();
}

int main() {
  OK(larch_device_open(-1, &dev));
  OK(larch_stream_create(dev, &st));
  larch_device* const cpu = dev;
  all();
  CHECK(!lacking);  // the CPU reference has every entry
  int rc = 0;
  larch_device* gpu = nullptr;
  larch_stream* ref_st = st;
  if (const char* e = std::getenv("LARCH_TEST_INKLING_GPU"); e && e[0] == '1') {
    int32_t gpus = 0;
    larch_status err = larch_device_count(&gpus);
    if (!err && gpus > 0) err = larch_device_open(0, &gpu);
    if (!err && gpu) {
      dev = gpu;
      err = larch_stream_create(dev, &st);
      if (!err) {
        lacking = false;
        all();
        std::printf(lacking ? "inkling: GPU 0 lacks entries: NOT RUN\n" : "inkling: GPU 0\n");
        rc = lacking ? 77 : 0;
      }
    } else if (!err) {
      std::printf("inkling: no GPU: NOT RUN\n");
      rc = 77;
    }
    if (err) {
      std::fprintf(stderr, "inkling: GPU 0 setup failed: %s\n", larch_last_error(gpu));
      ++failures;
    }
  }
  for (larch_buffer* b : owned) larch_buffer_free(b);
  if (st != ref_st) larch_stream_destroy(st);
  larch_stream_destroy(ref_st);
  if (gpu) larch_device_close(gpu);
  larch_device_close(cpu);
  if (failures) std::fprintf(stderr, "inkling: %d failures\n", failures);
  return failures ? 1 : rc;
}
