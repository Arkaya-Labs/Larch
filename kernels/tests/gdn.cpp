// Gated DeltaNet on the CPU reference, then (gdn_gpu) on GPU 0 where it is CUDA: both ops against larch.h's
// definitions in double, then what draft rollback and chunked prefill rely on, bit for bit: a sequence split over
// calls gives the outputs and state of one call, and a snapshot is the live state after its row. Then the checks the
// ops add (the ones on device values on the reference only: a GPU never reads them). On CUDA, the same rules at the
// 35B's layer shape.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <tuple>
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
#define BAD(call) CHECK((call) == LARCH_ERR_INVALID_ARG)

// One layer group of 4 slots in layer 1 of 2: kernel 4, 2 k heads and 4 v heads of 4, so 32 channels; v head
// h reads q and k head h % 2 (tiled), not h / 2.
constexpr int64_t KH = 2, VH = 4, HD = 4, CH = (2 * KH + VH) * HD, KERN = 4, SLOTS = 4;
constexpr int64_t CONV = (KERN - 1) * CH, SSM = VH * HD * HD;  // elements per slot

static larch_device* dev;
static larch_stream* st;
static bool reference = true;  // dev is the CPU reference, which also checks the device values
static std::vector<larch_buffer*> owned;
static rng g(11);

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
static larch_tensor f32(std::vector<int64_t> shape, const std::vector<float>& v) {
  return tensor(LARCH_F32, shape, v.data(), v.size() * 4);
}
static larch_tensor i32(std::vector<int64_t> shape, const std::vector<int32_t>& v) {
  return tensor(LARCH_I32, shape, v.data(), v.size() * 4);
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
static std::vector<float> uniform(size_t n, float lo, float hi) {
  std::uniform_real_distribution<float> u(lo, hi);
  std::vector<float> v(n);
  for (float& x : v) x = u(g);
  return v;
}
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * (1 + std::fabs(b)); }

// A state over new buffers holding conv and ssm, both layers.
static larch_gdn_state state(const bytes& conv, const std::vector<float>& ssm) {
  return {buffer(conv.data(), conv.size()), buffer(ssm.data(), ssm.size() * 4), SLOTS, 2, 1, KERN, KH, VH, HD};
}

// Two sequences: A, 3 rows in slot 2, snapshotting after its row 1 into slot 3; B, 2 rows in slot 0.
static const std::vector<int32_t> SLOT = {2, 0}, LEN = {3, 2}, SNAP = {-1, 3, -1, -1, -1};

static void conv() {
  const bytes c0 = halves(size_t(2 * SLOTS * CONV), g), x = halves(5 * CH, g);
  const std::vector<float> w = uniform(CH * KERN, -1, 1);
  larch_gdn_state s = state(c0, std::vector<float>(2 * SLOTS * SSM, 0));
  const larch_tensor xt = tensor(LARCH_F16, {5, CH}, x.data(), x.size()), wt = f32({CH, KERN}, w);
  const larch_tensor sl = i32({2}, SLOT), ln = i32({2}, LEN), sn = i32({5}, SNAP);
  const larch_tensor out = tensor(LARCH_F16, {5, CH}, nullptr, 5 * CH * 2);
  OK(larch_gdn_conv(st, &xt, &wt, &s, &sl, &ln, &sn, &out));
  const bytes o = fetch(out.buf, 5 * CH * 2), c1 = fetch(s.conv, c0.size());
  // inputs[t] of a sequence: its slot's 3 conv rows, then its rows of x (as bytes of halves).
  const auto layer1 = [&](const bytes& c, int32_t slot) { return &c[size_t(((SLOTS + slot) * CONV) * 2)]; };
  for (int seq = 0, r0 = 0; seq < 2; r0 += LEN[size_t(seq)], ++seq) {
    std::vector<const uint8_t*> in;
    for (int64_t t = 0; t < KERN - 1; ++t) in.push_back(layer1(c0, SLOT[size_t(seq)]) + t * CH * 2);
    for (int t = 0; t < LEN[size_t(seq)]; ++t) in.push_back(&x[size_t((r0 + t) * CH * 2)]);
    for (int t = 0; t < LEN[size_t(seq)]; ++t)
      for (int64_t c = 0; c < CH; ++c) {
        double sum = 0;
        for (int64_t i = 0; i < KERN; ++i) sum += double(w[size_t(c * KERN + i)]) * half(in[size_t(t + i)] + c * 2);
        CHECK(near(half(&o[size_t(((r0 + t) * CH + c) * 2)]), sum / (1 + std::exp(-sum)), 2e-3));
      }
    // The live slot holds the last 3 inputs; A's snapshot the 3 before its row 2.
    for (int64_t t = 0; t < KERN - 1; ++t) {
      CHECK(std::memcmp(layer1(c1, SLOT[size_t(seq)]) + t * CH * 2, in[size_t(LEN[size_t(seq)] + t)], CH * 2) == 0);
      if (seq == 0) CHECK(std::memcmp(layer1(c1, 3) + t * CH * 2, in[size_t(2 + t)], CH * 2) == 0);
    }
  }
  CHECK(std::memcmp(layer1(c1, 1), layer1(c0, 1), CONV * 2) == 0);  // a slot nobody named
  CHECK(std::memcmp(c1.data(), c0.data(), SLOTS * CONV * 2) == 0);   // layer 0

  // A in one call, and in two (row 0, then rows 1 and 2), from the same state.
  larch_gdn_state one = state(c0, std::vector<float>(2 * SLOTS * SSM, 0)),
                   two = state(c0, std::vector<float>(2 * SLOTS * SSM, 0));
  const larch_tensor a = tensor(LARCH_F16, {3, CH}, x.data(), 3 * CH * 2), s2 = i32({1}, {2});
  larch_tensor a0 = a, a12 = a;
  a0.shape[0] = 1;
  a12.shape[0] = 2;
  a12.offset = CH * 2;
  const larch_tensor l3 = i32({1}, {3}), l1 = i32({1}, {1}), l2 = i32({1}, {2});
  const larch_tensor o3 = tensor(LARCH_F16, {3, CH}, nullptr, 3 * CH * 2),
                      q3 = tensor(LARCH_F16, {3, CH}, nullptr, 3 * CH * 2);
  larch_tensor p0 = q3, p12 = q3;
  p0.shape[0] = 1;
  p12.shape[0] = 2;
  p12.offset = CH * 2;
  OK(larch_gdn_conv(st, &a, &wt, &one, &s2, &l3, nullptr, &o3));
  OK(larch_gdn_conv(st, &a0, &wt, &two, &s2, &l1, nullptr, &p0));
  OK(larch_gdn_conv(st, &a12, &wt, &two, &s2, &l2, nullptr, &p12));
  CHECK(fetch(o3.buf, 3 * CH * 2) == fetch(q3.buf, 3 * CH * 2));
  CHECK(fetch(one.conv, c0.size()) == fetch(two.conv, c0.size()));
}

// out and the final M of each sequence by larch.h's definition, in double; M is [slot][head][j][i].
static void delta_rule_by_definition(const bytes& qkv, const std::vector<float>& alpha, const std::vector<float>& beta,
                                     const std::vector<float>& dt, const std::vector<float>& a, double eps,
                                     std::vector<double>& m, std::vector<double>& out, std::vector<double>& snap) {
  for (int seq = 0, r = 0; seq < 2; ++seq)
    for (int t = 0; t < LEN[size_t(seq)]; ++t, ++r) {
      for (int64_t h = 0; h < VH; ++h) {
        const auto unit = [&](int64_t head) {  // q or k head, times 1 / sqrt(its sum of squares + eps)
          std::vector<double> v(HD);
          double ss = 0;
          for (int64_t i = 0; i < HD; ++i)
            ss += (v[size_t(i)] = half(&qkv[size_t((r * CH + head * HD + i) * 2)])) * v[size_t(i)];
          for (double& e : v) e /= std::sqrt(ss + eps);
          return v;
        };
        const std::vector<double> q = unit(h % KH), k = unit(KH + h % KH);
        const double x = double(alpha[size_t(r * VH + h)]) + dt[size_t(h)];
        const double gate = std::exp((x > 20 ? x : std::log1p(std::exp(x))) * a[size_t(h)]);
        const double b = 1 / (1 + std::exp(-double(beta[size_t(r * VH + h)])));
        double* mh = &m[size_t(((SLOT[size_t(seq)] * VH) + h) * HD * HD)];
        for (int64_t i = 0; i < HD * HD; ++i) mh[i] *= gate;
        for (int64_t j = 0; j < HD; ++j) {
          double mk = 0;
          for (int64_t i = 0; i < HD; ++i) mk += mh[j * HD + i] * k[size_t(i)];
          const double d = b * (half(&qkv[size_t((r * CH + (2 * KH + h) * HD + j) * 2)]) - mk);
          for (int64_t i = 0; i < HD; ++i) mh[j * HD + i] += d * k[size_t(i)];
        }
        for (int64_t j = 0; j < HD; ++j) {
          double mq = 0;
          for (int64_t i = 0; i < HD; ++i) mq += mh[j * HD + i] * q[size_t(i)];
          out[size_t(r * VH * HD + h * HD + j)] = mq / std::sqrt(double(HD));
        }
      }
      if (SNAP[size_t(r)] >= 0) std::copy_n(&m[size_t(SLOT[size_t(seq)] * SSM)], SSM, snap.begin());
    }
}

static void delta_rule() {
  const bytes qkv = halves(5 * CH, g);
  const std::vector<float> m0 = uniform(size_t(2 * SLOTS * SSM), -0.5f, 0.5f), alpha = uniform(5 * VH, -4, 8);
  const std::vector<float> beta = uniform(5 * VH, -2, 2), dt = uniform(VH, -1, 1);
  std::vector<float> a = uniform(VH, -1, 1);
  for (float& e : a) e = -std::exp(e);
  const larch_tensor qt = tensor(LARCH_F16, {5, CH}, qkv.data(), qkv.size()), at = f32({5, VH}, alpha);
  const larch_tensor bt = f32({5, VH}, beta), dtt = f32({VH}, dt), aa = f32({VH}, a);
  const larch_tensor sl = i32({2}, SLOT), ln = i32({2}, LEN), sn = i32({5}, SNAP);
  const larch_tensor out = tensor(LARCH_F16, {5, VH * HD}, nullptr, 5 * VH * HD * 2);
  larch_gdn_state s = state(bytes(size_t(2 * SLOTS * CONV * 2), 0), m0);
  OK(larch_gdn(st, &qt, &at, &bt, &dtt, &aa, 1e-6f, &s, &sl, &ln, &sn, &out));
  const bytes o = fetch(out.buf, 5 * VH * HD * 2), m1b = fetch(s.ssm, m0.size() * 4);
  std::vector<float> m1(m0.size());
  std::memcpy(m1.data(), m1b.data(), m1b.size());

  std::vector<double> m(&m0[SLOTS * SSM], &m0[SLOTS * SSM] + SLOTS * SSM), want(5 * VH * HD), snap(SSM);
  delta_rule_by_definition(qkv, alpha, beta, dt, a, 1e-6, m, want, snap);
  for (int64_t i = 0; i < 5 * VH * HD; ++i) CHECK(near(half(&o[size_t(i * 2)]), want[size_t(i)], 2e-3));
  for (const int32_t slot : SLOT)
    for (int64_t i = 0; i < SSM; ++i)
      CHECK(near(m1[size_t((SLOTS + slot) * SSM + i)], m[size_t(slot * SSM + i)], 1e-5));
  for (int64_t i = 0; i < SSM; ++i) CHECK(near(m1[size_t((SLOTS + 3) * SSM + i)], snap[size_t(i)], 1e-5));
  for (int64_t i = 0; i < SSM; ++i) CHECK(m1[size_t((SLOTS + 1) * SSM + i)] == m0[size_t((SLOTS + 1) * SSM + i)]);
  CHECK(std::memcmp(m1.data(), m0.data(), SLOTS * SSM * 4) == 0);  // layer 0

  // A in one call with its snapshot after row 0, and row by row from the same state: the snapshot is the
  // state after row 0 alone, and both runs end alike.
  larch_gdn_state one = state(bytes(size_t(2 * SLOTS * CONV * 2), 0), m0),
                   two = state(bytes(size_t(2 * SLOTS * CONV * 2), 0), m0);
  const larch_tensor s2 = i32({1}, {2}), l3 = i32({1}, {3}), l1 = i32({1}, {1}), snap0 = i32({3}, {3, -1, -1});
  const larch_tensor o3 = tensor(LARCH_F16, {3, VH * HD}, nullptr, 3 * VH * HD * 2);
  const larch_tensor o1 = tensor(LARCH_F16, {3, VH * HD}, nullptr, 3 * VH * HD * 2);
  larch_tensor q03 = qt, a03 = at, b03 = bt;  // A's rows
  for (larch_tensor* x : {&q03, &a03, &b03}) x->shape[0] = 3;
  OK(larch_gdn(st, &q03, &a03, &b03, &dtt, &aa, 1e-6f, &one, &s2, &l3, &snap0, &o3));
  for (int64_t r = 0; r < 3; ++r) {
    larch_tensor qr = qt, ar = at, br = bt, orow = o1;
    for (larch_tensor* x : {&qr, &ar, &br, &orow}) x->shape[0] = 1;
    qr.offset = uint64_t(r * CH * 2);
    ar.offset = br.offset = uint64_t(r * VH * 4);
    orow.offset = uint64_t(r * VH * HD * 2);
    OK(larch_gdn(st, &qr, &ar, &br, &dtt, &aa, 1e-6f, &two, &s2, &l1, nullptr, &orow));
    if (r == 0) {
      const bytes live = fetch(two.ssm, m0.size() * 4), whole = fetch(one.ssm, m0.size() * 4);
      CHECK(std::memcmp(&live[size_t((SLOTS + 2) * SSM * 4)], &whole[size_t((SLOTS + 3) * SSM * 4)], SSM * 4) == 0);
    }
  }
  CHECK(fetch(o3.buf, 3 * VH * HD * 2) == fetch(o1.buf, 3 * VH * HD * 2));
  const bytes e1 = fetch(one.ssm, m0.size() * 4), e2 = fetch(two.ssm, m0.size() * 4);
  CHECK(std::memcmp(&e1[size_t((SLOTS + 2) * SSM * 4)], &e2[size_t((SLOTS + 2) * SSM * 4)], SSM * 4) == 0);

  // One thing wrong per call: a slot named twice (live and live, live and snapshot), a slot out of range, q_lens
  // not summing to the rows, a negative q_len, a negative eps, and x of the wrong width.
  if (reference)  // the first five are wrong only in device values
    for (const auto& [slots, lens, snaps] :
         {std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<int32_t>>{{2, 2}, LEN, SNAP},
          {SLOT, LEN, {-1, 2, -1, -1, -1}},
          {{2, 4}, LEN, SNAP},
          {SLOT, {3, 1}, SNAP},
          {SLOT, {-1, 6}, SNAP}}) {
      const larch_tensor bs = i32({2}, slots), bl = i32({2}, lens), bn = i32({5}, snaps);
      BAD(larch_gdn(st, &qt, &at, &bt, &dtt, &aa, 1e-6f, &s, &bs, &bl, &bn, &out));
      const larch_tensor xw = f32({CH, KERN}, uniform(CH * KERN, -1, 1)),
                          co = tensor(LARCH_F16, {5, CH}, nullptr, 5 * CH * 2);
      BAD(larch_gdn_conv(st, &qt, &xw, &s, &bs, &bl, &bn, &co));
    }
  BAD(larch_gdn(st, &qt, &at, &bt, &dtt, &aa, -1e-6f, &s, &sl, &ln, &sn, &out));
  const larch_tensor narrow = tensor(LARCH_F16, {5, CH - 4}, qkv.data(), size_t(5 * (CH - 4) * 2));
  BAD(larch_gdn(st, &narrow, &at, &bt, &dtt, &aa, 1e-6f, &s, &sl, &ln, &sn, &out));
}

// A layer of KH k heads and VH v heads of HD (4 taps) on one sequence of R rows in slot 0, snapshotting after rows
// 0, 15, 16 and 32 (the staging edges of CUDA's gdn) into slots 1-4. The 35B's is 16, 32, 128; head_dim 64 and 256
// are the other two instantiations of gdn_k (E = 2 and 8), HD 4 above the first. ops.cpp checks these shapes
// against the reference; here, the bits draft rollback relies on.
namespace big {
constexpr int64_t KERN = 4, SLOTS = 5, R = 33;
static const int32_t SNAP_AT[] = {0, 15, 16, 32};

// The layer on dev: the same inputs and starting state for every instance, the outputs of every row.
struct layer {
  larch_tensor x, w, qkv, alpha, beta, dt, a, conv_out, out;
  larch_gdn_state s;
  int64_t kh, vh, hd, ch, conv, ssm;
  layer(int64_t kh_, int64_t vh_, int64_t hd_)
      : kh(kh_), vh(vh_), hd(hd_), ch((2 * kh_ + vh_) * hd_), conv((KERN - 1) * ch), ssm(vh_ * hd_ * hd_) {
    g.seed(35);  // the same inputs for every instance
    const size_t CH = size_t(ch), VH = size_t(vh);
    const bytes xs = halves(R * CH, g), qs = halves(R * CH, g), cs = halves(size_t(2 * SLOTS * conv), g);
    x = tensor(LARCH_F16, {R, ch}, xs.data(), xs.size());
    w = f32({ch, KERN}, uniform(CH * KERN, -1, 1));
    qkv = tensor(LARCH_F16, {R, ch}, qs.data(), qs.size());
    alpha = f32({R, vh}, uniform(R * VH, -3, 3));
    beta = f32({R, vh}, uniform(R * VH, -3, 3));
    dt = f32({vh}, uniform(VH, -1, 1));
    a = f32({vh}, uniform(VH, -2, -0.05f));  // as ops.cpp
    const std::vector<float> ms = uniform(size_t(2 * SLOTS * ssm), -0.5f, 0.5f);
    conv_out = tensor(LARCH_F16, {R, ch}, nullptr, size_t(R * ch * 2));
    out = tensor(LARCH_F16, {R, vh * hd}, nullptr, size_t(R * vh * hd * 2));
    s = {buffer(cs.data(), cs.size()), buffer(ms.data(), ms.size() * 4), SLOTS, 2, 1, KERN, uint32_t(kh),
         uint32_t(vh), uint32_t(hd)};
  }
  // Rows [r0, r0 + n) as one call of each op, with the snapshots that fall in them.
  void call(int64_t r0, int64_t n) {
    std::vector<int32_t> snaps(size_t(n), -1);
    for (int i = 0; i < 4; ++i)
      if (SNAP_AT[i] >= r0 && SNAP_AT[i] < r0 + n) snaps[size_t(SNAP_AT[i] - r0)] = 1 + i;
    const auto rows = [&](larch_tensor t) {
      t.offset += uint64_t(r0 * t.stride[0]) * (t.dtype == LARCH_F16 ? 2 : 4);
      t.shape[0] = n;
      return t;
    };
    const larch_tensor xr = rows(x), cr = rows(conv_out), qr = rows(qkv), ar = rows(alpha), br = rows(beta),
                        orow = rows(out), sl = i32({1}, {0}), ln = i32({1}, {int32_t(n)}), sn = i32({n}, snaps);
    OK(larch_gdn_conv(st, &xr, &w, &s, &sl, &ln, &sn, &cr));
    OK(larch_gdn(st, &qr, &ar, &br, &dt, &a, 1e-6f, &s, &sl, &ln, &sn, &orow));
  }
  // Slot `slot` of layer 1: its conv rows, then its delta-rule matrices, as bytes.
  bytes slot(int32_t slot) {
    const bytes c = fetch(s.conv, size_t(2 * SLOTS * conv * 2)), m = fetch(s.ssm, size_t(2 * SLOTS * ssm * 4));
    const uint8_t *cp = c.data() + (SLOTS + slot) * conv * 2, *mp = m.data() + (SLOTS + slot) * ssm * 4;
    bytes v(cp, cp + conv * 2);
    v.insert(v.end(), mp, mp + ssm * 4);
    return v;
  }
  bytes outputs() {
    bytes v = fetch(conv_out.buf, size_t(R * ch * 2)), o = fetch(out.buf, size_t(R * vh * hd * 2));
    v.insert(v.end(), o.begin(), o.end());
    return v;
  }
};

// One call, row by row (each snapshot equal to the live state after its row), and a 5-row check span then the
// rest: the same outputs and state, bit for bit.
static void rollback_bits(int64_t kh, int64_t vh, int64_t hd) {
  layer one(kh, vh, hd), steps(kh, vh, hd), span(kh, vh, hd);
  one.call(0, R);
  span.call(0, 5);
  span.call(5, R - 5);
  for (int64_t r = 0; r < R; ++r) {
    steps.call(r, 1);
    for (int i = 0; i < 4; ++i)
      if (SNAP_AT[i] == r) CHECK(steps.slot(0) == one.slot(1 + i));
  }
  const bytes outs = one.outputs(), live = one.slot(0);
  CHECK(steps.outputs() == outs);
  CHECK(steps.slot(0) == live);
  CHECK(span.outputs() == outs);
  CHECK(span.slot(0) == live);
  for (int i = 1; i <= 4; ++i) CHECK(span.slot(i) == one.slot(i));
}
}  // namespace big

// Snapshot slots of all -1 and no snapshot tensor at all: the same computation, bit for bit, for both ops (a guessing
// hybrid's captured step always passes the tensor, and a step that takes no snapshot writes it all -1).
static void minus_one_snaps_are_no_snaps() {
  const bytes c0 = halves(size_t(2 * SLOTS * CONV), g), x = halves(5 * CH, g), qkv = halves(5 * CH, g);
  const std::vector<float> w = uniform(CH * KERN, -1, 1), m0 = uniform(2 * SLOTS * SSM, -1, 1);
  const std::vector<float> alpha = uniform(5 * VH, -2, 2), beta = uniform(5 * VH, -2, 2), dt = uniform(VH, -1, 1),
                           a = uniform(VH, -1, 1);
  const larch_tensor xt = tensor(LARCH_F16, {5, CH}, x.data(), x.size()), wt = f32({CH, KERN}, w);
  const larch_tensor qt = tensor(LARCH_F16, {5, CH}, qkv.data(), qkv.size()), at = f32({5, VH}, alpha),
                      bt = f32({5, VH}, beta), dtt = f32({VH}, dt), aa = f32({VH}, a);
  const larch_tensor sl = i32({2}, SLOT), ln = i32({2}, LEN), none = i32({5}, {-1, -1, -1, -1, -1});
  const auto run = [&](const larch_tensor* snaps) {
    larch_gdn_state s = state(c0, m0);
    const larch_tensor co = tensor(LARCH_F16, {5, CH}, nullptr, 5 * CH * 2),
                        out = tensor(LARCH_F16, {5, VH * HD}, nullptr, 5 * VH * HD * 2);
    OK(larch_gdn_conv(st, &xt, &wt, &s, &sl, &ln, snaps, &co));
    OK(larch_gdn(st, &qt, &at, &bt, &dtt, &aa, 1e-6f, &s, &sl, &ln, snaps, &out));
    bytes r = fetch(co.buf, 5 * CH * 2), o = fetch(out.buf, 5 * VH * HD * 2), c = fetch(s.conv, c0.size()),
          m = fetch(s.ssm, m0.size() * 4);
    r.insert(r.end(), o.begin(), o.end());
    r.insert(r.end(), c.begin(), c.end());
    r.insert(r.end(), m.begin(), m.end());
    return r;
  };
  CHECK(run(nullptr) == run(&none));
}

// The reference, then with LARCH_TEST_GDN_GPU=1 (ctest's gdn_gpu) the same on GPU 0, which must be CUDA: else 77.
int main() {
  OK(larch_device_open(-1, &dev));
  OK(larch_stream_create(dev, &st));
  conv();
  delta_rule();
  minus_one_snaps_are_no_snaps();
  larch_device *ref = dev, *gpu = nullptr;
  larch_stream* ref_st = st;
  int rc = 0;
  if (const char* e = std::getenv("LARCH_TEST_GDN_GPU"); e && e[0] == '1') {
    // Only a machine without GPUs, or whose GPU 0 is not CUDA, skips: a call that fails on the way is a failure.
    int32_t gpus = 0;
    larch_device_props props{};
    larch_status err = larch_device_count(&gpus);
    if (!err && gpus > 0) err = larch_device_open(0, &gpu);
    if (!err && gpu) err = larch_device_props_get(gpu, &props);
    if (!err && gpu && props.backend == LARCH_BACKEND_CUDA) {
      dev = gpu;
      err = larch_stream_create(dev, &st);  // st stays the CPU stream if this fails, so nothing below may run
      if (!err) {
        reference = false;
        conv();
        delta_rule();
        minus_one_snaps_are_no_snaps();
        big::rollback_bits(16, 32, 128);
        big::rollback_bits(2, 4, 64);
        big::rollback_bits(2, 4, 256);
        std::printf("gdn: GPU 0 (%s)\n", props.name);
      }
    } else if (!err) {
      std::printf("gdn: GPU 0 absent or not CUDA: NOT RUN\n");
      rc = 77;
    }
    if (err) {
      std::fprintf(stderr, "gdn: GPU 0 setup failed: %s\n", larch_last_error(gpu));
      ++failures;
    }
  }
  for (larch_buffer* b : owned) larch_buffer_free(b);
  if (st != ref_st) larch_stream_destroy(st);
  if (gpu) larch_device_close(gpu);
  larch_stream_destroy(ref_st);
  larch_device_close(ref);
  std::printf("%s (%d failures)\n", failures ? "FAIL" : "ok", failures);
  return failures ? 1 : rc;
}
