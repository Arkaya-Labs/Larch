// The CPU reference against answers that do not come from it: every f16 value both ways, every f16 value
// into an fp8 KV page, the quant decoders against gguf-py (quant_answers.py), YaRN rope against ggml, the
// device sampler's random words against splitmix64's published outputs, and every other op against its
// definition by hand or as a naive loop, in the dtypes larch.h names. Then the CPU device (-2), and
// larch.h's argument checks.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <vector>

#include "testdata.h"

static int failures = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                         \
    }                                                                     \
  } while (0)
#define OK(call) CHECK((call) == LARCH_OK)
#define BAD(call) CHECK((call) == LARCH_ERR_INVALID_ARG)

static larch_device* dev;
static larch_stream* st;
static std::vector<larch_buffer*> owned;
static std::mt19937 gen(7);
using h16 = std::vector<uint16_t>;

// A dense tensor over a new buffer holding `bytes` from src (left as the backend fills it when src is null).
static larch_tensor tensor(larch_dtype t, std::vector<int64_t> shape, const void* src, uint64_t bytes) {
  larch_buffer* b = nullptr;
  OK(larch_buffer_alloc(dev, bytes, &b));
  owned.push_back(b);
  if (src) OK(larch_upload(st, b, 0, src, bytes));
  larch_tensor x = {b, 0, t, uint32_t(shape.size()), {}, {}};
  int64_t stride = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    x.shape[i] = shape[i];
    x.stride[i] = stride;
    stride *= i + 1 == shape.size() ? shape[i] / block_of(t) : shape[i];
  }
  return x;
}
static larch_tensor out(larch_dtype t, std::vector<int64_t> shape) {
  const int64_t n = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<>());
  return tensor(t, shape, nullptr, uint64_t(n) * (t == LARCH_F16 ? 2 : 4));
}
static larch_tensor f32(std::vector<int64_t> shape, const std::vector<float>& v) {
  return tensor(LARCH_F32, shape, v.data(), v.size() * 4);
}
static larch_tensor f16(std::vector<int64_t> shape, const h16& v) { return tensor(LARCH_F16, shape, v.data(), v.size() * 2); }
static larch_tensor i32(std::vector<int64_t> shape, const std::vector<int32_t>& v) {
  return tensor(LARCH_I32, shape, v.data(), v.size() * 4);
}
template <class T>
static std::vector<T> read(const larch_tensor& t, size_t n) {
  std::vector<T> v(n);
  OK(larch_download(st, t.buf, t.offset, v.data(), n * sizeof(T)));
  return v;
}
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * (1 + std::fabs(b)); }

// Exact f16 decoding, written apart from the reference's.
static float from_h(uint16_t h) {
  const int e = h >> 10 & 31, m = h & 1023;
  const float v = e == 31 ? (m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
                  : e     ? std::ldexp(float(m + 1024), e - 25)
                          : std::ldexp(float(m), -24);
  return h & 0x8000 ? -v : v;
}
static std::vector<float> from_h(const h16& h) {
  std::vector<float> v(h.size());
  for (size_t i = 0; i < h.size(); ++i) v[i] = from_h(h[i]);
  return v;
}
// n random halves with magnitude in [2^-6, 2), either sign.
static h16 random_h(size_t n) {
  h16 v(n);
  for (uint16_t& h : v) h = uint16_t((gen() & 0x8000) | (9 + gen() % 7) << 10 | (gen() & 0x3ff));
  return v;
}
static std::vector<float> random_f32(size_t n) {
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<float> v(n);
  for (float& x : v) x = u(gen);
  return v;
}

// All 65536 halves both ways: embed decodes an f16 table to f32, and gemm with a = 0 and alpha = -0
// writes -0 + bias = bias, so it encodes f32 into c's dtype, f16.
static void f16_conversion() {
  h16 all(65536);
  std::iota(all.begin(), all.end(), uint16_t(0));
  std::vector<int32_t> rows(256);
  std::iota(rows.begin(), rows.end(), 0);
  const larch_tensor table = f16({256, 256}, all), ids = i32({256}, rows), dec_t = out(LARCH_F32, {256, 256});
  OK(larch_embed(st, &table, &ids, &dec_t));
  const std::vector<float> dec = read<float>(dec_t, 65536);
  for (uint32_t i = 0; i < 65536; ++i) {
    const float want = from_h(all[i]);
    CHECK(std::isnan(want) ? std::isnan(dec[i]) : std::memcmp(&dec[i], &want, 4) == 0);
  }
  const larch_tensor zero = f16({1, 1}, {0}), zeros = f16({65536, 1}, h16(65536, 0));
  const larch_tensor bias = f32({65536}, dec), enc_t = out(LARCH_F16, {1, 65536});
  OK(larch_gemm(st, &zero, &zeros, &bias, &enc_t, -0.0f, 0.0f));
  const h16 enc = read<uint16_t>(enc_t, 65536);
  for (uint32_t i = 0; i < 65536; ++i) CHECK(std::isnan(dec[i]) ? (enc[i] & 0x7fff) > 0x7c00 : enc[i] == all[i]);
#ifdef __FLT16_MANT_DIG__
  // Rounding between halves, subnormals and overflow, against the compiler.
  std::vector<float> xs = random_f32(65536);
  for (size_t i = 0; i < xs.size(); ++i) xs[i] = std::ldexp(xs[i], int(i % 48) - 30);
  const larch_tensor xb = f32({65536}, xs), o = out(LARCH_F16, {1, 65536});
  OK(larch_gemm(st, &zero, &zeros, &xb, &o, -0.0f, 0.0f));
  const h16 got = read<uint16_t>(o, 65536);
  for (size_t i = 0; i < xs.size(); ++i) {
    const _Float16 want = _Float16(xs[i]);
    uint16_t w;
    std::memcpy(&w, &want, 2);
    CHECK(got[i] == w);
  }
#endif
}

// Known answers for every quant type, from quant_answers.py: 2 rows of 512 elements of LCG bytes with
// the f16 scale fields masked to [0, 1), decoded by gguf-py. Then mmvq on the same weights against a
// naive dot product over those decoded values.
static void quant_known_answers() {
  struct answer {
    larch_dtype t;
    int64_t bytes;
    std::vector<int> scales;
    uint64_t f32_hash, f16_hash;
    float y0, y777;
  };
  const answer answers[] = {
      {LARCH_Q2_K, 84, {80, 82}, 0xa1511273757dadf1u, 0xadf29f126bdbc549u, 0.688804626f, -0.0219011307f},
      {LARCH_Q3_K, 110, {108}, 0x5b095715ae0af69du, 0x3b3b595de70ee424u, -2.61035156f, 0.000753521919f},
      {LARCH_Q4_K, 144, {0, 2}, 0x1234b6272add2f76u, 0xa13f4774a6e62f22u, 52.1602936f, -0.0626525879f},
      {LARCH_Q5_K, 176, {0, 2}, 0xe84f5a1b747890b5u, 0x8f1c9d73ff58b8efu, 16.4233475f, -0.0417079926f},
      {LARCH_Q6_K, 210, {208}, 0xcf23910ef2fafa0bu, 0x172afd07078df727u, 170.276367f, 0.00141352415f},
      {LARCH_Q8_0, 34, {0}, 0x018f1636440f15edu, 0x1b27dd5ef720afadu, -3.36914062f, -0.310546875f},
      {LARCH_Q4_0, 18, {0}, 0x248f8e34b93ae30au, 0x572d3032e27fcabdu, -0.0142478943f, 0.000392198563f},
      {LARCH_Q5_0, 22, {0}, 0x8d6c2dec59caa3d5u, 0xe719fcba43eb9a39u, -0.000108003616f, -0.00174188614f},
  };
  for (const answer& a : answers) {
    std::vector<uint8_t> raw(size_t(2 * 512 / block_of(a.t) * a.bytes));
    uint64_t s = uint64_t(a.t);
    for (uint8_t& b : raw) b = uint8_t((s = s * 6364136223846793005u + 1442695040888963407u) >> 56);
    for (size_t blk = 0; blk < raw.size(); blk += size_t(a.bytes))
      for (const int o : a.scales) raw[blk + size_t(o) + 1] &= 0x3b;
    const larch_tensor w = tensor(a.t, {2, 512}, raw.data(), raw.size()), ids = i32({2}, {0, 1});
    const larch_tensor y = out(LARCH_F32, {2, 512}), h = out(LARCH_F16, {2, 512});
    OK(larch_embed(st, &w, &ids, &y));
    OK(larch_dequant(st, &w, &h));
    const std::vector<float> yv = read<float>(y, 1024);
    const h16 hv = read<uint16_t>(h, 1024);
    CHECK(fnv(yv.data(), 4096) == a.f32_hash);
    CHECK(fnv(hv.data(), 2048) == a.f16_hash);
    if (yv[0] != a.y0 || yv[777] != a.y777)
      std::fprintf(stderr, "dtype %d: y[0] = %.9g (want %.9g), y[777] = %.9g (want %.9g)\n", int(a.t), yv[0], a.y0,
                   yv[777], a.y777);

    const h16 x = random_h(3 * 512);
    const std::vector<float> xf = from_h(x), bias = random_f32(2);
    const larch_tensor xt = f16({3, 512}, x), bt = f32({2}, bias);
    const larch_tensor o32 = out(LARCH_F32, {3, 2}), o16 = out(LARCH_F16, {3, 2});
    OK(larch_mmvq(st, &w, &xt, &bt, &o32));
    OK(larch_mmvq(st, &w, &xt, &bt, &o16));
    const std::vector<float> g32 = read<float>(o32, 6), g16 = from_h(read<uint16_t>(o16, 6));
    for (int r = 0; r < 3; ++r)
      for (int n = 0; n < 2; ++n) {
        double want = bias[size_t(n)];
        for (int k = 0; k < 512; ++k) want += double(yv[size_t(n * 512 + k)]) * xf[size_t(r * 512 + k)];
        CHECK(near(g32[size_t(r * 2 + n)], want, 1e-6));
        CHECK(near(g16[size_t(r * 2 + n)], want, 1e-3));
      }
  }
  // 17 rows is the only thing wrong here.
  const std::vector<uint8_t> q8(2 * 16 * 34, 0);
  const larch_tensor w = tensor(LARCH_Q8_0, {2, 512}, q8.data(), q8.size());
  const larch_tensor x17 = f16({17, 512}, random_h(17 * 512)), o17 = out(LARCH_F32, {17, 2});
  BAD(larch_mmvq(st, &w, &x17, nullptr, &o17));
  CHECK(std::strstr(larch_last_error(dev), "16 rows") != nullptr);
}

// By hand: a = [[1, 2], [3, 4]], b_t = [[1, 0], [1, -1], [0.5, 2]], bias = [0.25, 0, -1] and alpha = 2
// give c = [[2.25, -2, 8], [6.25, -2, 18]] over a c that starts as NaN (beta = 0 must not read it);
// then beta = 0.5 over that c gives [[3.375, -3, 12], [9.375, -3, 27]].
static void gemm_by_hand() {
  const larch_tensor a = f16({2, 2}, {0x3c00, 0x4000, 0x4200, 0x4400});
  const larch_tensor bt = f16({3, 2}, {0x3c00, 0x0000, 0x3c00, 0xbc00, 0x3800, 0x4000});
  const larch_tensor bias = f32({3}, {0.25f, 0.0f, -1.0f}), c16 = out(LARCH_F16, {2, 3});
  OK(larch_gemm(st, &a, &bt, &bias, &c16, 2.0f, 0.0f));
  CHECK((from_h(read<uint16_t>(c16, 6)) == std::vector<float>{2.25f, -2, 8, 6.25f, -2, 18}));
  const larch_tensor c32 = f32({2, 3}, {2.25f, -2, 8, 6.25f, -2, 18});
  OK(larch_gemm(st, &a, &bt, &bias, &c32, 2.0f, 0.5f));
  CHECK((read<float>(c32, 6) == std::vector<float>{3.375f, -3, 12, 9.375f, -3, 27}));
}

// (i + 1) / 16 for i < 16, exact in f16.
static h16 sixteenths() {
  h16 xh(16);
  for (int i = 0; i < 16; ++i) {
    const float v = float(i + 1) / 16;
    const int e = std::ilogb(v);
    xh[size_t(i)] = uint16_t((e + 15) << 10 | int(std::ldexp(v, 10 - e)) - 1024);
  }
  return xh;
}

// Closed forms with ext_factor 0: position 0 is the identity, position 1 turns pair i by
// base^(-2i / n_dims) / ff[i], and dims past n_dims pass through. Then YaRN with the ramp strictly
// inside (0, 1) for pairs 2..4, against ggml_rope_ext's f32 output, dumped once.
static void rope_positions() {
  const int64_t hd = 16, nd = 12;
  const float base = 500000.0f;
  const std::vector<float> ffv = {1, 2, 4, 8, 16, 32};
  const larch_tensor ff = f32({nd / 2}, ffv);
  for (uint32_t neox : {0u, 1u})
    for (bool use_ff : {false, true}) {
      const h16 xh = random_h(2 * 2 * hd);
      const std::vector<float> x = from_h(xh);
      const larch_tensor qk = f16({2, 2, hd}, xh), pos = i32({2}, {0, 1});
      const larch_rope_params p = {uint32_t(nd), neox, 4096, base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, use_ff ? &ff : nullptr};
      OK(larch_rope(st, &qk, &pos, &p));
      const std::vector<float> y = from_h(read<uint16_t>(qk, x.size()));
      for (size_t i = 0; i < 2 * hd; ++i) CHECK(y[i] == x[i]);
      for (int64_t h = 0; h < 2; ++h) {
        for (int64_t pair = 0; pair < nd / 2; ++pair) {
          const size_t a = size_t(2 * hd + h * hd + (neox ? pair : 2 * pair)), b = a + (neox ? nd / 2 : 1);
          const double th = std::pow(double(base), -2.0 * double(pair) / double(nd)) / (use_ff ? ffv[size_t(pair)] : 1);
          CHECK(near(y[a], x[a] * std::cos(th) - x[b] * std::sin(th), 1e-3));
          CHECK(near(y[b], x[a] * std::sin(th) + x[b] * std::cos(th), 1e-3));
        }
        for (int64_t i = nd; i < hd; ++i) CHECK(y[size_t(2 * hd + h * hd + i)] == x[size_t(2 * hd + h * hd + i)]);
      }
    }
  // ggml_rope_ext(x[i] = (i + 1) / 16, pos 1000, n_dims 12, n_ctx_orig 4096, base 1e4, freq_scale 0.25,
  // ext_factor 1, attn_factor 1, beta_fast 32, beta_slow 1), normal then neox.
  const float ggml[2][16] = {
      {-0.0776673406f, 0.138887003f, -0.32785368f, 0.138278872f, 0.34989816f, 0.431853443f, 0.516765773f, 0.55247277f,
       -0.19934921f, 0.936434567f, 0.678670824f, 0.938862085f, 0.8125f, 0.875f, 0.9375f, 1},
      {-0.371889055f, -0.58683604f, 0.204638183f, 0.308112741f, -0.424223334f, 0.325242102f, 0.338993669f,
       0.000357523561f, 0.643362939f, 0.701806724f, 0.747951627f, 0.897665262f, 0.8125f, 0.875f, 0.9375f, 1}};
  for (uint32_t neox : {0u, 1u}) {
    const larch_tensor qk = f16({1, 1, 16}, sixteenths()), pos = i32({1}, {1000});
    const larch_rope_params p = {12, neox, 4096, 10000.0f, 0.25f, 1.0f, 1.0f, 32.0f, 1.0f, nullptr};
    OK(larch_rope(st, &qk, &pos, &p));
    const std::vector<float> y = from_h(read<uint16_t>(qk, 16));
    for (int i = 0; i < 16; ++i) CHECK(near(y[size_t(i)], ggml[neox][i], 1e-3));
  }
}

// ggml_rope_multi: Qwen 3.5's text rope with t = h = w is larch_rope's neox, bit for bit; with t alone
// set, only its interleaved pairs (j % 3 == 0, j < 33) turn; then each mode against ggml_rope_multi's f32 output, dumped once
// (x[i] = (i + 1) / 16, one head of 16, base 1e4, n_ctx_orig 4096, beta 32 / 1).
static void rope_multi_positions() {
  const int64_t hd = 256, nd = 64;
  for (int32_t at : {0, 1, 123, 40000}) {
    const h16 xh = random_h(2 * hd);
    const larch_tensor a = f16({1, 2, hd}, xh), b = f16({1, 2, hd}, xh);
    const larch_tensor pos = i32({1}, {at}), pos4 = i32({1, 4}, {at, at, at, 0});
    const larch_rope_params r = {uint32_t(nd), 1, 4096, 1e7f, 1, 0, 1, 32, 1, nullptr};
    const larch_rope_multi_params m = {r, {11, 11, 10, 0}, LARCH_ROPE_IMROPE};
    OK(larch_rope(st, &a, &pos, &r));
    OK(larch_rope_multi(st, &b, &pos4, &m));
    CHECK(read<uint16_t>(a, size_t(2 * hd)) == read<uint16_t>(b, size_t(2 * hd)));
  }
  {
    const h16 xh = random_h(hd);
    const std::vector<float> x = from_h(xh);
    // Base 1e4 at position 1000: every pair of t turns by at least 0.1 rad, so f16 sees it.
    const larch_tensor qk = f16({1, 1, hd}, xh), pos = i32({1, 4}, {1000, 0, 0, 0});
    const larch_rope_multi_params m = {{uint32_t(nd), 0, 4096, 1e4f, 1, 0, 1, 32, 1, nullptr}, {11, 11, 10, 0},
                                        LARCH_ROPE_IMROPE};
    OK(larch_rope_multi(st, &qk, &pos, &m));
    const std::vector<float> y = from_h(read<uint16_t>(qk, size_t(hd)));
    for (int64_t j = 0; j < nd / 2; ++j) {
      const bool turns = j % 3 == 0 && j < 33;
      CHECK(turns ? y[size_t(j)] != x[size_t(j)] || y[size_t(j + nd / 2)] != x[size_t(j + nd / 2)]
                  : y[size_t(j)] == x[size_t(j)] && y[size_t(j + nd / 2)] == x[size_t(j + nd / 2)]);
    }
  }
  struct dumped {
    uint32_t nd, s[4], mode;
    int32_t pos[4];
    float freq_scale, ext;
    float y[16];
  };
  const dumped cases[] = {
      {16, {3, 3, 2, 0}, LARCH_ROPE_IMROPE, {7, 3, 11, 0}, 1, 0,
       {-0.322436064f, -0.435061336f, -0.52765584f, 0.0792328864f, 0.287988037f, 0.344342381f, 0.43092683f,
        0.499051064f, 0.4651317f, 0.465802133f, 0.478948683f, 0.786588967f, 0.82150799f, 0.887512505f, 0.940539539f,
        1.00047386f}},
      {16, {3, 3, 2, 0}, LARCH_ROPE_IMROPE, {1000, 900, 800, 0}, 0.25f, 1,
       {-0.48957774f, -0.722570062f, 0.754456878f, -0.213108227f, 0.847381294f, -0.699873686f, 0.21856904f,
        0.486927688f, 0.419036329f, -0.0677212626f, -0.298600286f, 0.874575913f, 0.51422888f, 0.827713251f,
        1.1575247f, 1.17622173f}},
      {16, {2, 3, 3, 0}, LARCH_ROPE_MROPE, {4, 7, 13, 0}, 1, 0,
       {0.384848684f, -0.558345795f, -0.299491793f, 0.0792328864f, 0.254906118f, 0.338722378f, 0.425275862f,
        0.495884806f, -0.414974689f, 0.307408512f, 0.646619797f, 0.786588967f, 0.832367361f, 0.889672458f,
        0.943108141f, 1.00204706f}},
      {8, {4, 4, 4, 4}, LARCH_ROPE_VISION, {5, 9, 5, 9}, 1, 0,
       {0.55712378f, -0.189943135f, 0.152905002f, 0.246246889f, -0.619574487f, -0.452307314f, 0.351468176f,
        0.49097988f, 0.0996272266f, 0.608417273f, 0.696011901f, 0.751240551f, -0.611506283f, 0.837656319f,
        0.973027527f, 1.0044595f}},
  };
  for (const dumped& c : cases) {
    const larch_tensor qk = f16({1, 1, 16}, sixteenths()),
                        pos = i32({1, 4}, {c.pos[0], c.pos[1], c.pos[2], c.pos[3]});
    const larch_rope_multi_params m = {{c.nd, 0, 4096, 1e4f, c.freq_scale, c.ext, 1, 32, 1, nullptr},
                                        {c.s[0], c.s[1], c.s[2], c.s[3]},
                                        c.mode};
    OK(larch_rope_multi(st, &qk, &pos, &m));
    const std::vector<float> y = from_h(read<uint16_t>(qk, 16));
    for (int i = 0; i < 16; ++i) CHECK(near(y[size_t(i)], c.y[i], 1e-3));
  }
}

// Softmax attention for one query row over positions [lo, n) of dense K/V ([pos][kv_head][d]), in double,
// with head h's sinks[h] as one more logit (no value) when sinks is given.
static std::vector<double> naive_attention(const float* q, const std::vector<float>& k, const std::vector<float>& v,
                                           int64_t n, int64_t heads, int64_t kvh, int64_t hd, double scale,
                                           int64_t lo = 0, const float* sinks = nullptr) {
  std::vector<double> o(size_t(heads * hd), 0.0), w(static_cast<size_t>(n));
  for (int64_t h = 0; h < heads; ++h) {
    const int64_t kv = h / (heads / kvh);
    double mx = sinks ? sinks[h] : -1e300, sum = 0;
    for (int64_t p = lo; p < n; ++p) {
      double dot = 0;
      for (int64_t d = 0; d < hd; ++d) dot += double(q[h * hd + d]) * k[size_t((p * kvh + kv) * hd + d)];
      mx = std::max(mx, w[size_t(p)] = scale * dot);
    }
    if (sinks) sum = std::exp(sinks[h] - mx);
    for (int64_t p = lo; p < n; ++p) sum += w[size_t(p)] = std::exp(w[size_t(p)] - mx);
    for (int64_t p = lo; p < n; ++p)
      for (int64_t d = 0; d < hd; ++d) o[size_t(h * hd + d)] += w[size_t(p)] / sum * v[size_t((p * kvh + kv) * hd + d)];
  }
  return o;
}

// Both attention ops against naive_attention. Pool: layer 1 of 2, 6 blocks of 16 slots, 2 kv heads of
// 8; q has 4 heads. Sequence A has 37 positions in blocks {3, 0, 2}, B 20 in {5, 1}. kv_write also gets
// two padding rows (slot -1) of NaN; prefill runs all of A plus B's chunk at positions 12..19; decode
// runs A, a padding row (seq_len 0, output zero) and B.
static void attention_vs_naive() {
  // The padding-row and layer-0 checks below rely on new buffers being 0xFF.
  for (const uint32_t w : read<uint32_t>(out(LARCH_F32, {4}), 4)) CHECK(w == 0xffffffffu);
  const int64_t bt = 16, kvh = 2, hd = 8, heads = 4, row = kvh * hd, qrow = heads * hd;
  const float scale = 0.35f;
  const uint64_t pool_bytes = 2 * 6 * bt * row * 2;
  const larch_tensor kb = tensor(LARCH_F16, {1}, nullptr, pool_bytes), vb = tensor(LARCH_F16, {1}, nullptr, pool_bytes);
  const larch_kv_pool pool = {kb.buf, vb.buf, 6, uint32_t(bt), 2, uint32_t(kvh), uint32_t(hd), LARCH_F16, 1};
  const std::vector<std::vector<int32_t>> tables = {{3, 0, 2}, {5, 1, -1}};
  const int64_t lens[2] = {37, 20};

  h16 kh, vh;  // kv_write rows: A, a padding row, B, a padding row
  std::vector<int32_t> slots;
  std::vector<float> k[2], v[2];  // dense per sequence
  for (int s = 0; s < 2; ++s) {
    const h16 ks = random_h(size_t(lens[s] * row)), vs = random_h(size_t(lens[s] * row));
    kh.insert(kh.end(), ks.begin(), ks.end());
    vh.insert(vh.end(), vs.begin(), vs.end());
    k[s] = from_h(ks);
    v[s] = from_h(vs);
    for (int64_t p = 0; p < lens[s]; ++p) slots.push_back(tables[size_t(s)][size_t(p / bt)] * int32_t(bt) + int32_t(p % bt));
    kh.insert(kh.end(), size_t(row), 0x7e00);
    vh.insert(vh.end(), size_t(row), 0x7e00);
    slots.push_back(-1);
  }
  const int64_t n_kv = int64_t(slots.size());
  const larch_tensor kt = f16({n_kv, kvh, hd}, kh), vt = f16({n_kv, kvh, hd}, vh), st_t = i32({n_kv}, slots);
  OK(larch_kv_write(st, &kt, &vt, &st_t, &pool));
  // larch.h's layout, [layer][block * block_tokens + i][kv_head][head_dim]: layer 0 stays poisoned.
  const h16 pk = read<uint16_t>(kb, size_t(pool_bytes / 2));
  for (int64_t i = 0; i < 6 * bt * row; ++i) CHECK(pk[size_t(i)] == 0xffff);
  for (size_t r = 0; r < slots.size(); ++r)
    if (slots[r] >= 0) CHECK(std::memcmp(&pk[size_t((6 * bt + slots[r]) * row)], &kh[r * size_t(row)], size_t(row) * 2) == 0);

  const h16 qh = random_h(size_t((37 + 8) * qrow));
  const std::vector<float> q = from_h(qh);
  const larch_tensor qt = f16({45, heads, hd}, qh), table = i32({2, 3}, {3, 0, 2, 5, 1, -1});
  const larch_tensor start = i32({2}, {0, 12}), qlens = i32({2}, {37, 8}), pre = out(LARCH_F16, {45, heads, hd});
  OK(larch_attn_prefill(st, &qt, &pool, &table, &start, &qlens, scale, &pre));
  const std::vector<float> p = from_h(read<uint16_t>(pre, size_t(45 * qrow)));
  for (int64_t r = 0; r < 45; ++r) {
    const int s = r < 37 ? 0 : 1;
    const int64_t n = s == 0 ? r + 1 : 12 + (r - 37) + 1;
    const std::vector<double> want = naive_attention(&q[size_t(r * qrow)], k[s], v[s], n, heads, kvh, hd, scale);
    for (int64_t i = 0; i < qrow; ++i) CHECK(near(p[size_t(r * qrow + i)], want[size_t(i)], 2e-3));
  }

  const h16 qdh = random_h(size_t(3 * qrow));
  const std::vector<float> qd = from_h(qdh);
  const larch_tensor qdt = f16({3, heads, hd}, qdh), dtable = i32({3, 3}, {3, 0, 2, 0, 0, 0, 5, 1, -1});
  const larch_tensor seq_lens = i32({3}, {37, 0, 20}), dec = out(LARCH_F16, {3, heads, hd});
  OK(larch_attn_decode(st, &qdt, &pool, &dtable, &seq_lens, 0, scale, &dec));
  const std::vector<float> d = from_h(read<uint16_t>(dec, size_t(3 * qrow)));
  const std::vector<double> wa = naive_attention(&qd[0], k[0], v[0], 37, heads, kvh, hd, scale);
  const std::vector<double> wb = naive_attention(&qd[size_t(2 * qrow)], k[1], v[1], 20, heads, kvh, hd, scale);
  for (int64_t i = 0; i < qrow; ++i) {
    CHECK(near(d[size_t(i)], wa[size_t(i)], 2e-3));
    CHECK(d[size_t(qrow + i)] == 0.0f);
    CHECK(near(d[size_t(2 * qrow + i)], wb[size_t(i)], 2e-3));
  }
}

// rmsnorm, qk_norm, silu_mul, sigmoid_mul and add against their definitions, in double.
static void elementwise_by_definition() {
  const std::vector<float> x = {3, -4, 1, 0.5f, 0.25f, 2, -1, 8}, w = {1, 0.5f, 2, -1};
  const larch_tensor xt = f32({2, 4}, x), wt = f32({4}, w), o = out(LARCH_F16, {2, 4});
  OK(larch_rmsnorm(st, &xt, &wt, 0.5f, &o));
  const std::vector<float> y = from_h(read<uint16_t>(o, 8));
  for (size_t r = 0; r < 2; ++r) {
    double ss = 0;
    for (size_t c = 0; c < 4; ++c) ss += double(x[r * 4 + c]) * x[r * 4 + c];
    for (size_t c = 0; c < 4; ++c) CHECK(near(y[r * 4 + c], x[r * 4 + c] / std::sqrt(ss / 4 + 0.5) * w[c], 1e-3));
  }

  const h16 qh = random_h(8);
  const std::vector<float> q = from_h(qh);
  const larch_tensor qk = f16({1, 2, 4}, qh);
  OK(larch_qk_norm(st, &qk, &wt, 0.25f));
  const std::vector<float> qn = from_h(read<uint16_t>(qk, 8));
  for (size_t h = 0; h < 2; ++h) {
    double ss = 0;
    for (size_t c = 0; c < 4; ++c) ss += double(q[h * 4 + c]) * q[h * 4 + c];
    for (size_t c = 0; c < 4; ++c) CHECK(near(qn[h * 4 + c], q[h * 4 + c] / std::sqrt(ss / 4 + 0.25) * w[c], 1e-3));
  }

  const h16 gh = random_h(16), uh = random_h(16);
  const std::vector<float> g = from_h(gh), u = from_h(uh);
  const larch_tensor gt = f16({2, 8}, gh), ut = f16({2, 8}, uh), so = out(LARCH_F16, {2, 8});
  OK(larch_silu_mul(st, &gt, &ut, &so));
  const std::vector<float> sy = from_h(read<uint16_t>(so, 16));
  for (size_t i = 0; i < 16; ++i) CHECK(near(sy[i], g[i] / (1 + std::exp(-double(g[i]))) * u[i], 1e-3));
  // sigmoid_mul: a gate per element, a gate per row, and in place.
  const h16 rh = random_h(2);
  const larch_tensor gr = f16({2, 1}, rh), go = out(LARCH_F16, {2, 8}), gro = out(LARCH_F16, {2, 8});
  OK(larch_sigmoid_mul(st, &ut, &gt, &go));
  OK(larch_sigmoid_mul(st, &ut, &gr, &gro));
  OK(larch_sigmoid_mul(st, &ut, &gt, &ut));
  const std::vector<float> ge = from_h(read<uint16_t>(go, 16)), gw = from_h(read<uint16_t>(gro, 16));
  const std::vector<float> gi = from_h(read<uint16_t>(ut, 16));
  for (size_t i = 0; i < 16; ++i) {
    CHECK(near(ge[i], u[i] / (1 + std::exp(-double(g[i]))), 1e-3) && gi[i] == ge[i]);
    CHECK(near(gw[i], u[i] / (1 + std::exp(-double(from_h(rh[i / 8])))), 1e-3));
  }

  const std::vector<float> a = random_f32(6);
  const h16 bh = random_h(6);
  const larch_tensor at = f32({2, 3}, a), bt = f16({2, 3}, bh), ao = out(LARCH_F32, {2, 3});
  OK(larch_add(st, &at, &bt, &ao));
  OK(larch_add(st, &at, &bt, &at));  // in place: out == a
  const std::vector<float> s1 = read<float>(ao, 6), s2 = read<float>(at, 6);
  for (size_t i = 0; i < 6; ++i) CHECK(s1[i] == a[i] + from_h(bh[i]) && s2[i] == s1[i]);
}

// The f16 bits of a value f16 holds exactly (small integers here).
static uint16_t exact_h(float v) {
  if (v == 0) return 0;
  const int e = int(std::floor(std::log2(std::fabs(v))));
  const uint16_t m = uint16_t(std::lround((std::fabs(v) / std::ldexp(1.0f, e) - 1) * 1024));
  return uint16_t((v < 0 ? 0x8000 : 0) | (e + 15) << 10 | m);
}

// layernorm, gelu and attn_full against their definitions, in double: a ViT's norm (with and without a bias) and
// ggml_gelu's tanh form, in place too.
static void vit_ops_by_definition() {
  const std::vector<float> x = {3, -4, 1, 0.5f, 100.25f, 102, 99, 108}, w = {1, 0.5f, 2, -1}, b = {0.5f, -1, 0, 2};
  const larch_tensor xt = f32({2, 4}, x), wt = f32({4}, w), bt = f32({4}, b), o = out(LARCH_F16, {2, 4}),
                      on = out(LARCH_F16, {2, 4});
  OK(larch_layernorm(st, &xt, &wt, &bt, 1e-5f, &o));
  OK(larch_layernorm(st, &xt, &wt, nullptr, 1e-5f, &on));
  const std::vector<float> y = from_h(read<uint16_t>(o, 8)), yn = from_h(read<uint16_t>(on, 8));
  for (size_t r = 0; r < 2; ++r) {
    double mean = 0, var = 0;
    for (size_t c = 0; c < 4; ++c) mean += x[r * 4 + c] / 4.0;
    for (size_t c = 0; c < 4; ++c) var += (x[r * 4 + c] - mean) * (x[r * 4 + c] - mean) / 4.0;
    for (size_t c = 0; c < 4; ++c) {
      const double n = (x[r * 4 + c] - mean) / std::sqrt(var + 1e-5) * w[c];
      CHECK(near(y[r * 4 + c], n + b[c], 1e-3) && near(yn[r * 4 + c], n, 1e-3));
    }
  }
  const h16 gh = random_h(16);
  const std::vector<float> g = from_h(gh);
  const larch_tensor gt = f16({2, 8}, gh), go = out(LARCH_F16, {2, 8});
  OK(larch_gelu(st, &gt, &go));
  OK(larch_gelu(st, &gt, &gt));  // in place: out == x
  const std::vector<float> gy = from_h(read<uint16_t>(go, 16)), gi = from_h(read<uint16_t>(gt, 16));
  for (size_t i = 0; i < 16; ++i) {
    const double v = g[i], want = 0.5 * v * (1 + std::tanh(std::sqrt(2 / std::numbers::pi) * (v + 0.044715 * v * v * v)));
    CHECK(near(gy[i], want, 1e-3) && gi[i] == gy[i]);
  }
  // attn_full: two segments of two rows, one head of 2; each row's softmax over its own segment only.
  const std::vector<float> qf = {1, 0, 0, 1, 2, 0, 0, 0}, kf = {1, 0, 0, 1, 1, 1, 0, 0}, vf = {1, 2, 3, 4, 5, 6, 7, 8};
  h16 qh, kh, vh;
  for (size_t i = 0; i < 8; ++i) qh.push_back(exact_h(qf[i])), kh.push_back(exact_h(kf[i])), vh.push_back(exact_h(vf[i]));
  const larch_tensor qt = f16({4, 1, 2}, qh), kt = f16({4, 1, 2}, kh), vt = f16({4, 1, 2}, vh), at = out(LARCH_F16, {4, 1, 2});
  OK(larch_attn_full(st, &qt, &kt, &vt, 2, 0.5f, &at));
  const std::vector<float> ay = from_h(read<uint16_t>(at, 8));
  for (size_t r = 0; r < 4; ++r) {
    const size_t b = r / 2 * 2;
    double s[2], z = 0, o[2] = {0, 0};
    for (size_t j = 0; j < 2; ++j) {
      s[j] = std::exp(0.5 * (qf[r * 2] * kf[(b + j) * 2] + qf[r * 2 + 1] * kf[(b + j) * 2 + 1]));
      z += s[j];
    }
    for (size_t j = 0; j < 2; ++j)
      for (size_t c = 0; c < 2; ++c) o[c] += s[j] / z * vf[(b + j) * 2 + c];
    CHECK(near(ay[r * 2], o[0], 1e-3) && near(ay[r * 2 + 1], o[1], 1e-3));
  }
  BAD(larch_attn_full(st, &qt, &kt, &vt, 3, 0.5f, &at));  // 4 rows are not whole segments of 3
  const larch_tensor b5 = f32({5}, {0, 0, 0, 0, 0}), g9 = out(LARCH_F16, {2, 9});
  BAD(larch_layernorm(st, &xt, &wt, &b5, 1e-5f, &o));  // b wider than a row
  BAD(larch_layernorm(st, &xt, &wt, &bt, 1e-5f, &xt));  // f32 out
  BAD(larch_gelu(st, &gt, &g9));                      // out wider than x
}

static void sample_prep_order_and_penalty() {
  // vocab 6; ids 1 and 4 tie at 2.0; id 5 (3.0) and id 0 (-1.0) are penalised by 2.
  const larch_tensor lg = f32({1, 6}, {-1.0f, 2.0f, 0.5f, 1.0f, 2.0f, 3.0f});
  larch_sample_params sp = {0.0f, 4, 2.0f, 3};
  const larch_tensor pen = i32({1, 4}, {5, 0, 5, 1});  // window 3: ids 5 and 0 (5 once)
  const larch_tensor vals = out(LARCH_F32, {1, 64}), ids = out(LARCH_I32, {1, 64}), lse = out(LARCH_F32, {1});
  OK(larch_sample_prep(st, &lg, &sp, &pen, &vals, &ids, &lse));
  const std::vector<int32_t> top = read<int32_t>(ids, 64);
  const std::vector<float> tv = read<float>(vals, 64);
  CHECK(top[0] == 1 && top[1] == 4 && top[2] == 5 && top[3] == 3 && top[4] == -1);
  CHECK(tv[2] == 1.5f && std::isinf(tv[4]));
  const double want = std::log(std::exp(-2.0) + 2 * std::exp(2.0) + std::exp(0.5) + std::exp(1.0) + std::exp(1.5));
  CHECK(near(read<float>(lse, 1)[0], want, 1e-6));
  // Host parameters out of range, and a NaN logit (the reference's debugging aid).
  for (const larch_sample_params bad : {larch_sample_params{0, 65, 1, 0}, {0, 4, 0, 0}, {0, 4, 1.5f, 5}})
    BAD(larch_sample_prep(st, &lg, &bad, &pen, &vals, &ids, &lse));
  const larch_tensor nan = f32({1, 6}, {0, 1, std::numeric_limits<float>::quiet_NaN(), 0, 0, 0});
  BAD(larch_sample_prep(st, &nan, &sp, nullptr, &vals, &ids, &lse));
  // top_k 0 is 64, capped at the vocabulary; a negative penalty id is padding.
  sp = {0.0f, 0, 2.0f, 4};
  const larch_tensor pen2 = i32({1, 4}, {5, -1, 0, -1});
  OK(larch_sample_prep(st, &lg, &sp, &pen2, &vals, &ids, &lse));
  const std::vector<int32_t> all = read<int32_t>(ids, 64);
  CHECK(all[0] == 1 && all[1] == 4 && all[2] == 5 && all[3] == 3 && all[4] == 2 && all[5] == 0 && all[6] == -1);
}

// fp8 e4m3 written apart from the reference: the value of code c < 127, and the code nearest to |x| by
// distance, ties to the even code (448 and infinity land on 126, the largest), NaN on 127; with x's sign.
static double fp8_value(int c) { return c >> 3 ? std::ldexp(8 + (c & 7), (c >> 3) - 10) : std::ldexp(c & 7, -9); }
static uint8_t fp8_nearest(float x) {
  int best = 0;
  for (int c = 1; c < 127; ++c) {
    const double dc = std::fabs(fp8_value(c) - std::fabs(x)), db = std::fabs(fp8_value(best) - std::fabs(x));
    if (dc < db || (dc == db && c % 2 == 0)) best = c;
  }
  return uint8_t((std::signbit(x) ? 0x80 : 0) | (std::isnan(x) ? 127 : best));
}

// The 8-bit KV pages. fp8: every f16 value rounded by kv_write, then every byte read back by attention (one
// position, so the output is V itself). q8_0: one row by ggml's quantize_row_q8_0_ref, by hand. Then
// attention over pages of both dtypes against naive_attention over the values the pages hold.
static void kv_pages() {
  h16 all(65536);
  std::iota(all.begin(), all.end(), uint16_t(0));
  std::vector<int32_t> slots(2048);
  std::iota(slots.begin(), slots.end(), 0);
  const larch_tensor kt = f16({2048, 1, 32}, all), sl = i32({2048}, slots);
  const larch_tensor kb = tensor(LARCH_F16, {1}, nullptr, 65536), vb = tensor(LARCH_F16, {1}, nullptr, 65536);
  const larch_kv_pool fp8 = {kb.buf, vb.buf, 128, 16, 1, 1, 32, LARCH_FP8_E4M3, 0};
  OK(larch_kv_write(st, &kt, &kt, &sl, &fp8));
  const std::vector<uint8_t> codes = read<uint8_t>(kb, 65536);
  for (size_t i = 0; i < 65536; ++i) CHECK(codes[i] == fp8_nearest(from_h(all[i])));

  std::vector<uint8_t> every(4096, 0);  // byte c at slot (c / 32) * 16, column c % 32
  for (int c = 0; c < 256; ++c) every[size_t(c / 32 * 16 * 32 + c % 32)] = uint8_t(c);
  const larch_tensor zk = tensor(LARCH_F16, {1}, nullptr, 4096), ev = tensor(LARCH_F16, {1}, every.data(), 4096);
  OK(larch_upload(st, zk.buf, 0, std::vector<uint8_t>(4096, 0).data(), 4096));
  const larch_kv_pool small = {zk.buf, ev.buf, 8, 16, 1, 1, 32, LARCH_FP8_E4M3, 0};
  const larch_tensor q8 = f16({8, 1, 32}, h16(256, 0)), table = i32({8, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
  const larch_tensor ones = i32({8}, std::vector<int32_t>(8, 1)), o8 = out(LARCH_F16, {8, 1, 32});
  OK(larch_attn_decode(st, &q8, &small, &table, &ones, 0, 1.0f, &o8));
  const std::vector<float> dec = from_h(read<uint16_t>(o8, 256));
  for (int c = 0; c < 256; ++c) {
    const double want = (c & 127) == 127 ? std::nan("") : (c & 128 ? -1 : 1) * fp8_value(c & 127);
    CHECK(std::isnan(want) ? std::isnan(dec[size_t(c)]) : dec[size_t(c)] == want);
  }

  // q8_0 by hand: x_j = (j - 16) / 4, so amax = 4, d = 4 / 127 and each code roundf(x_j / d) in f32.
  h16 xh(32);
  for (int j = 0; j < 32; ++j) {  // (j - 16) / 4 as a half: exact
    const float v = float(j - 16) / 4;
    xh[size_t(j)] = v == 0 ? 0 : uint16_t((v < 0 ? 0x8000 : 0) | (std::ilogb(v) + 15) << 10 |
                                          (int(std::ldexp(std::fabs(v), 10 - std::ilogb(v))) - 1024));
  }
  const larch_tensor xq = f16({1, 1, 32}, xh), s0 = i32({1}, {0});
  const larch_tensor qk = tensor(LARCH_F16, {1}, nullptr, 34), qv = tensor(LARCH_F16, {1}, nullptr, 34);
  const larch_kv_pool q8_0 = {qk.buf, qv.buf, 1, 1, 1, 1, 32, LARCH_Q8_0, 0};
  OK(larch_kv_write(st, &xq, &xq, &s0, &q8_0));
  const std::vector<uint8_t> blk = read<uint8_t>(qk, 34);
  const float id = 1.0f / (4.0f / 127);
  CHECK(blk[0] == 0x08 && blk[1] == 0x28);  // f16(4 / 127) = 0x2808
  for (int j = 0; j < 32; ++j) CHECK(int8_t(blk[size_t(2 + j)]) == int(std::round(from_h(xh[size_t(j)]) * id)));

  // Attention over 40 positions in blocks {2, 0, 3} of 16, 2 kv heads of 32, 4 q heads.
  const int64_t kvh = 2, hd = 32, heads = 4, row = kvh * hd, n = 40;
  const h16 kh = random_h(size_t(n * row)), vh = random_h(size_t(n * row)), qh = random_h(size_t(heads * hd));
  std::vector<int32_t> at(n);
  for (int32_t p = 0; p < n; ++p) at[size_t(p)] = std::vector<int32_t>{2, 0, 3}[size_t(p / 16)] * 16 + p % 16;
  const larch_tensor k40 = f16({n, kvh, hd}, kh), v40 = f16({n, kvh, hd}, vh), s40 = i32({n}, at);
  const larch_tensor qt = f16({1, heads, hd}, qh), t3 = i32({1, 3}, {2, 0, 3}), len = i32({1}, {int32_t(n)});
  const std::vector<float> q = from_h(qh);
  for (const larch_dtype t : {LARCH_Q8_0, LARCH_FP8_E4M3}) {
    const uint64_t page = t == LARCH_Q8_0 ? uint64_t(row / 32 * 34) : uint64_t(row);
    const larch_tensor pk = tensor(LARCH_F16, {1}, nullptr, 64 * page),
                        pv = tensor(LARCH_F16, {1}, nullptr, 64 * page);
    const larch_kv_pool pool = {pk.buf, pv.buf, 4, 16, 1, uint32_t(kvh), uint32_t(hd), t, 0};
    OK(larch_kv_write(st, &k40, &v40, &s40, &pool));
    const std::vector<uint8_t> kp = read<uint8_t>(pk, size_t(64 * page)), vp = read<uint8_t>(pv, size_t(64 * page));
    std::vector<float> k(size_t(n * row)), v(k.size());  // what the pages hold, decoded here
    for (int64_t p = 0; p < n; ++p)
      for (int64_t c = 0; c < row; ++c) {
        const size_t i = size_t(p * row + c), base = size_t(at[size_t(p)]) * size_t(page);
        if (t == LARCH_FP8_E4M3) {
          const auto val = [&](uint8_t b) { return float((b & 128 ? -1 : 1) * fp8_value(b & 127)); };
          k[i] = val(kp[base + size_t(c)]);
          v[i] = val(vp[base + size_t(c)]);
        } else {
          const size_t b = base + size_t(c / 32 * 34);
          k[i] = from_h(uint16_t(kp[b] | kp[b + 1] << 8)) * float(int8_t(kp[b + 2 + size_t(c % 32)]));
          v[i] = from_h(uint16_t(vp[b] | vp[b + 1] << 8)) * float(int8_t(vp[b + 2 + size_t(c % 32)]));
        }
      }
    const larch_tensor o = out(LARCH_F16, {1, heads, hd});
    OK(larch_attn_decode(st, &qt, &pool, &t3, &len, 0, 0.17f, &o));
    const std::vector<float> got = from_h(read<uint16_t>(o, size_t(heads * hd)));
    const std::vector<double> want = naive_attention(q.data(), k, v, n, heads, kvh, hd, 0.17);
    for (int64_t i = 0; i < heads * hd; ++i) CHECK(near(got[size_t(i)], want[size_t(i)], 2e-3));
  }
  // A q8_0 pool's head_dim is whole q8_0 blocks.
  const larch_tensor k16 = f16({1, 1, 16}, random_h(16));
  const larch_kv_pool q8_16 = {qk.buf, qv.buf, 1, 1, 1, 1, 16, LARCH_Q8_0, 0};
  BAD(larch_kv_write(st, &k16, &k16, &s0, &q8_16));
}

// mmvq_id against a naive dot over the decoded experts: 3 experts of N = 4 over k = 64 in q8_0, each slot
// on its own (x per row, f32 and f16 out) and combined by the weights (x per slot).
static void experts() {
  std::vector<uint8_t> raw(12 * 2 * 34);
  for (uint8_t& b : raw) b = uint8_t(gen());
  for (size_t blk = 0; blk < raw.size(); blk += 34) raw[blk + 1] &= 0x3b;
  std::vector<int32_t> rows(12);
  std::iota(rows.begin(), rows.end(), 0);
  const larch_tensor w = tensor(LARCH_Q8_0, {12, 64}, raw.data(), raw.size()), wi = i32({12}, rows);
  const larch_tensor wd_t = out(LARCH_F32, {12, 64});
  OK(larch_embed(st, &w, &wi, &wd_t));
  const std::vector<float> wd = read<float>(wd_t, 12 * 64), bias = random_f32(12);
  const h16 x2h = random_h(2 * 64), x3h = random_h(2 * 2 * 64);
  const std::vector<float> x2 = from_h(x2h), x3 = from_h(x3h);
  const std::vector<int32_t> ids = {2, 0, 1, 1};
  const std::vector<float> wts = {0.25f, 0.75f, -1.0f, 2.0f};
  const larch_tensor x2t = f16({2, 64}, x2h), x3t = f16({2, 2, 64}, x3h), it = i32({2, 2}, ids);
  const larch_tensor bt = f32({12}, bias), wt = f32({2, 2}, wts);
  const larch_tensor y32 = out(LARCH_F32, {2, 2, 4}), y16 = out(LARCH_F16, {2, 2, 4}), yw = out(LARCH_F32, {2, 4});
  OK(larch_mmvq_id(st, &w, &x2t, &it, &bt, nullptr, &y32));
  OK(larch_mmvq_id(st, &w, &x2t, &it, &bt, nullptr, &y16));
  OK(larch_mmvq_id(st, &w, &x3t, &it, nullptr, &wt, &yw));
  const std::vector<float> g32 = read<float>(y32, 16), g16 = from_h(read<uint16_t>(y16, 16)), gw = read<float>(yw, 8);
  const auto dot = [&](int e, int n, const float* x) {
    double acc = 0;
    for (int k = 0; k < 64; ++k) acc += double(wd[size_t((e * 4 + n) * 64 + k)]) * x[k];
    return acc;
  };
  for (int r = 0; r < 2; ++r)
    for (int n = 0; n < 4; ++n) {
      double sum = 0;
      for (int j = 0; j < 2; ++j) {
        const int e = ids[size_t(r * 2 + j)];
        const double y = dot(e, n, &x2[size_t(r * 64)]) + bias[size_t(e * 4 + n)];
        CHECK(near(g32[size_t((r * 2 + j) * 4 + n)], y, 1e-6) && near(g16[size_t((r * 2 + j) * 4 + n)], y, 1e-3));
        sum += wts[size_t(r * 2 + j)] * dot(e, n, &x3[size_t((r * 2 + j) * 64)]);
      }
      CHECK(near(gw[size_t(r * 4 + n)], sum, 1e-6));
    }
  const larch_tensor bad_ids = i32({2, 2}, {2, 3, 1, 1});  // 3 experts: id 3 is out of range
  BAD(larch_mmvq_id(st, &w, &x2t, &bad_ids, nullptr, nullptr, &y32));
}

// larch_attn over one sequence of 40 positions in blocks {2, 0, 3} of 16, 2 kv heads of 8 and 4 q heads,
// a span of 6 rows at positions 30..35: with no mask, sinks or window it is attn_prefill bit for bit; with a
// window of 7 and sinks it is naive_attention over each row's last 7 positions plus the sinks. Then the
// mask's limits.
static void attention_window_and_sinks() {
  const int64_t kvh = 2, hd = 8, heads = 4, row = kvh * hd, qrow = heads * hd, n = 36;
  const larch_tensor kb = tensor(LARCH_F16, {1}, nullptr, 4 * 16 * row * 2),
                      vb = tensor(LARCH_F16, {1}, nullptr, 4 * 16 * row * 2);
  const larch_kv_pool pool = {kb.buf, vb.buf, 4, 16, 1, uint32_t(kvh), uint32_t(hd), LARCH_F16, 0};
  const h16 kh = random_h(size_t(n * row)), vh = random_h(size_t(n * row)), qh = random_h(size_t(6 * qrow));
  std::vector<int32_t> at(n);
  for (int32_t p = 0; p < n; ++p) at[size_t(p)] = std::vector<int32_t>{2, 0, 3}[size_t(p / 16)] * 16 + p % 16;
  const larch_tensor kt = f16({n, kvh, hd}, kh), vt = f16({n, kvh, hd}, vh), sl = i32({n}, at);
  OK(larch_kv_write(st, &kt, &vt, &sl, &pool));
  const larch_tensor qt = f16({6, heads, hd}, qh), table = i32({1, 3}, {2, 0, 3}), start = i32({1}, {30});
  const larch_tensor lens = i32({1}, {6}), a = out(LARCH_F16, {6, heads, hd}), pre = out(LARCH_F16, {6, heads, hd});
  OK(larch_attn(st, &qt, &pool, &table, &start, &lens, nullptr, nullptr, 0, 0, 0.35f, &a));
  OK(larch_attn_prefill(st, &qt, &pool, &table, &start, &lens, 0.35f, &pre));
  CHECK(read<uint16_t>(a, size_t(6 * qrow)) == read<uint16_t>(pre, size_t(6 * qrow)));

  const std::vector<float> sinks = {0.5f, -1.0f, 2.0f, 0.0f}, q = from_h(qh), k = from_h(kh), v = from_h(vh);
  const larch_tensor sk = f32({heads}, sinks), ws = out(LARCH_F16, {6, heads, hd});
  OK(larch_attn(st, &qt, &pool, &table, &start, &lens, nullptr, &sk, 7, 0, 0.35f, &ws));
  const std::vector<float> got = from_h(read<uint16_t>(ws, size_t(6 * qrow)));
  for (int64_t j = 0; j < 6; ++j) {
    const std::vector<double> want =
        naive_attention(&q[size_t(j * qrow)], k, v, 31 + j, heads, kvh, hd, 0.35, 31 + j - 7, sinks.data());
    for (int64_t i = 0; i < qrow; ++i) CHECK(near(got[size_t(j * qrow + i)], want[size_t(i)], 2e-3));
  }
  // A mask bit past the span, and a span over 32 rows with a mask (both device values: debugging aids).
  const larch_tensor past = i32({6}, {1, 3, 7, 15, 31, 64});
  BAD(larch_attn(st, &qt, &pool, &table, &start, &lens, &past, nullptr, 0, 0, 0.35f, &ws));
  const larch_tensor q33 = f16({33, heads, hd}, random_h(size_t(33 * qrow))),
                      m33 = i32({33}, std::vector<int32_t>(33, 1));
  const larch_tensor s33 = i32({1}, {0}), l33 = i32({1}, {33}), o33 = out(LARCH_F16, {33, heads, hd});
  BAD(larch_attn(st, &q33, &pool, &table, &s33, &l33, &m33, nullptr, 0, 0, 0.35f, &o33));
}

// splitmix64 by stepping its state, written apart from the reference.
static uint64_t splitmix(uint64_t seed, uint64_t index) {
  uint64_t z = seed;
  for (uint64_t i = 0; i < index; ++i) z += 0x9E3779B97F4A7C15u;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9u;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBu;
  return z ^ (z >> 31);
}

// larch_sample: the host sampler's test row by hand (greedy, bans, top_k, top_p, nothing left), then draws
// whose pick follows from the row's random word alone, checked against splitmix64 at pos + 1.
static void device_sampler() {
  CHECK(splitmix(0, 1) == 0xe220a8397b1dcdafu && splitmix(0, 2) == 0x6e789e6aa1b965f4u &&
        splitmix(0, 3) == 0x06c45d188009454fu);  // Vigna's splitmix64.c, seed 0
  const float inf = std::numeric_limits<float>::infinity();
  // Draws per row: candidates (padded with -inf and id -1), lse, settings, position and up to 4 banned ids.
  struct row {
    std::vector<float> v;
    std::vector<int32_t> id;
    float lse;
    larch_draw_params p;
    int32_t pos;
    std::vector<int32_t> ban;
  };
  const auto draw = [&](const std::vector<row>& rs) {
    const int64_t n = int64_t(rs.size());
    std::vector<float> v(size_t(n * 64), -inf), lse;
    std::vector<int32_t> id(size_t(n * 64), -1), pos, ban(size_t(n * 4), -1);
    std::vector<larch_draw_params> p;
    for (int64_t r = 0; r < n; ++r) {
      std::copy(rs[size_t(r)].v.begin(), rs[size_t(r)].v.end(), v.begin() + r * 64);
      std::copy(rs[size_t(r)].id.begin(), rs[size_t(r)].id.end(), id.begin() + r * 64);
      std::copy(rs[size_t(r)].ban.begin(), rs[size_t(r)].ban.end(), ban.begin() + r * 4);
      lse.push_back(rs[size_t(r)].lse);
      p.push_back(rs[size_t(r)].p);
      pos.push_back(rs[size_t(r)].pos);
    }
    const larch_tensor vt = f32({n, 64}, v), it = i32({n, 64}, id), lt = f32({n}, lse), pt = i32({n}, pos);
    const larch_tensor bt = i32({n, 4}, ban), oi = out(LARCH_I32, {n}), ol = out(LARCH_F32, {n});
    OK(larch_sample(st, &vt, &it, &lt, p.data(), &pt, &bt, &oi, &ol));
    return std::make_pair(read<int32_t>(oi, size_t(n)), read<float>(ol, size_t(n)));
  };
  const std::vector<float> v4 = {3, 2, 1, -1};
  const std::vector<int32_t> id4 = {7, 3, 5, 9};
  const float lse4 = float(std::log(std::exp(3.0) + std::exp(2.0) + std::exp(1.0) + std::exp(-1.0)));
  const auto [ids, lps] = draw({
      {v4, id4, lse4, {0, 1, 0, 0, 1}, 0, {}},              // greedy
      {v4, id4, lse4, {0, 1, 0, 0, 1}, 0, {7}},             // greedy past a banned id
      {v4, id4, lse4, {1, 1, 1, 0, 1}, 5, {}},              // top_k 1
      {v4, id4, lse4, {1, 0.5f, 0, 0, 1}, 6, {}},           // top_p 0.5: the first holds 66 %
      {v4, id4, lse4, {1, 1, 0, 0, 1}, 7, {7, 3, 5, 9}},    // every candidate banned
      {{}, {}, 0, {1, 1, 0, 0, 1}, 8, {}},                  // no candidate at all
  });
  CHECK((ids == std::vector<int32_t>{7, 3, 7, 7, -1, -1}));
  CHECK(lps[0] == 3 - lse4 && lps[1] == 2 - lse4 && lps[2] == 3 - lse4 && lps[4] == -inf && lps[5] == -inf);

  // Two candidates at temperature 1, seed 0: equal values draw the second iff the word's top bit is set, so
  // positions 0, 1 and 2 give ids 9, 7, 7. Then values 0 and -ln 3 at temperatures 1 and 0.5 over 64
  // positions: the first iff u (W the weights' sum) is below 1, skipping words within 2^-40 of it.
  const auto [eq, eql] = draw({{{1, 1}, {7, 9}, 1 + float(std::log(2.0)), {1, 1, 0, 0, 0}, 0, {}},
                               {{1, 1}, {7, 9}, 0, {1, 1, 0, 0, 0}, 1, {}},
                               {{1, 1}, {7, 9}, 0, {1, 1, 0, 0, 0}, 2, {}}});
  CHECK((eq == std::vector<int32_t>{9, 7, 7}) && eql[0] == 1 - (1 + float(std::log(2.0))));
  for (const float temp : {1.0f, 0.5f}) {
    std::vector<row> rs;
    for (int32_t p = 0; p < 64; ++p)
      rs.push_back({{0, -float(std::log(3.0))}, {4, 6}, 0, {temp, 1, 0, 0, 12345}, p, {}});
    const std::vector<int32_t> got = draw(rs).first;
    const double total = 1 + std::exp(double(-float(std::log(3.0))) / temp);
    for (int32_t p = 0; p < 64; ++p) {
      const double u = double(splitmix(12345, uint64_t(p) + 1) >> 11) * 0x1p-53 * total;
      if (std::fabs(u - 1) > 0x1p-40) CHECK(got[size_t(p)] == (u < 1 ? 4 : 6));
    }
  }
  // top_k 2 of four never draws past the second; top_k 0 stops at the first id -1.
  std::vector<row> rs;
  for (int32_t p = 0; p < 32; ++p) {
    rs.push_back({{1, 1, 1, 1}, {7, 3, 5, 9}, 0, {1, 1, 2, 0, 5}, p, {}});
    rs.push_back({{1, 1, 1, 1}, {7, 3, -1, 9}, 0, {1, 1, 0, 0, 5}, p, {}});
  }
  for (const int32_t id : draw(rs).first) CHECK(id == 7 || id == 3);
}

// moe_route by hand: [1, 3, 2, 3] keeps experts 1 and 3 (the tie to the lower id first) at 0.5 each, and
// [0, -1, 4, 1] keeps 2 and 3 at e^4 / (e^4 + e) and e / (e^4 + e); scale 0 is 1. All four experts of a row
// weigh 1 in all. A NaN logit is refused (a debugging aid).
static void moe_routing() {
  const larch_tensor lg = f32({2, 4}, {1, 3, 2, 3, 0, -1, 4, 1}), ids = out(LARCH_I32, {2, 2}),
                      w = out(LARCH_F32, {2, 2});
  const double hi = std::exp(4.0) / (std::exp(4.0) + std::exp(1.0)), lo = 1 - hi;
  for (const float scale : {0.0f, 2.5f}) {
    OK(larch_moe_route(st, &lg, scale, &ids, &w));
    const double s = scale == 0 ? 1 : scale;
    const std::vector<float> wv = read<float>(w, 4);
    CHECK((read<int32_t>(ids, 4) == std::vector<int32_t>{1, 3, 2, 3}));
    CHECK(near(wv[0], 0.5 * s, 1e-6) && near(wv[1], 0.5 * s, 1e-6) && near(wv[2], hi * s, 1e-6) &&
          near(wv[3], lo * s, 1e-6));
  }
  const larch_tensor i4 = out(LARCH_I32, {2, 4}), w4 = out(LARCH_F32, {2, 4});
  OK(larch_moe_route(st, &lg, 1.0f, &i4, &w4));
  const std::vector<float> all = read<float>(w4, 8);
  CHECK((read<int32_t>(i4, 8) == std::vector<int32_t>{1, 3, 2, 0, 2, 3, 0, 1}));
  CHECK(near(all[0] + all[1] + all[2] + all[3], 1, 1e-6) && near(all[4] + all[5] + all[6] + all[7], 1, 1e-6));
  const larch_tensor nan = f32({1, 4}, {0, std::numeric_limits<float>::quiet_NaN(), 0, 0});
  const larch_tensor i1 = out(LARCH_I32, {1, 2}), w1 = out(LARCH_F32, {1, 2});
  BAD(larch_moe_route(st, &nan, 1.0f, &i1, &w1));
}

// The CPU device (-2) beside the reference: its threads, a host buffer that ops on both read and write, and
// an event one waits on for the other.
static void cpu_device() {
  larch_device* cpu = nullptr;
  larch_stream* cs = nullptr;
  OK(larch_device_open(-2, &cpu));
  OK(larch_stream_create(cpu, &cs));
  larch_device_props p{};
  OK(larch_device_props_get(cpu, &p));
  CHECK(std::strcmp(p.name, "CPU") == 0 && p.backend == LARCH_BACKEND_CPU && p.compute_units >= 1);
  OK(larch_device_set_threads(cpu, 3));
  OK(larch_device_props_get(cpu, &p));
  CHECK(p.compute_units == 3);
  OK(larch_device_set_threads(dev, 3));  // the reference keeps it and stays single-threaded
  OK(larch_device_props_get(dev, &p));
  CHECK(p.compute_units == 1 && std::strcmp(p.name, "CPU reference") == 0);

  larch_buffer* hb = nullptr;
  void* host = nullptr;
  OK(larch_buffer_alloc_host(cpu, 16, &hb, &host));
  const float a[4] = {1, 2, 3, 4};
  std::memcpy(host, a, 16);
  const larch_tensor at = {hb, 0, LARCH_F32, 2, {1, 4}, {4, 1}}, bt = f16({1, 4}, {0x3c00, 0x3c00, 0x3c00, 0x3c00});
  const larch_tensor o = out(LARCH_F32, {1, 4});
  OK(larch_add(st, &at, &bt, &o));  // the reference's stream reads the CPU device's host buffer
  CHECK((read<float>(o, 4) == std::vector<float>{2, 3, 4, 5}));
  OK(larch_add(cs, &at, &bt, &at));  // and the CPU device's stream writes it, with the reference's b
  larch_event* e = nullptr;
  OK(larch_event_record(cs, &e));
  OK(larch_event_sync(e));
  OK(larch_stream_wait(st, e));  // another device's event: the host waits
  OK(larch_stream_wait(cs, e));  // its own: the CPU slot
  float back[4];
  std::memcpy(back, host, 16);
  CHECK(back[0] == 2 && back[3] == 5);
  BAD(larch_stream_wait(st, nullptr));
  BAD(larch_stream_wait(nullptr, e));
  BAD(larch_event_sync(nullptr));
  BAD(larch_device_set_threads(nullptr, 1));
  BAD(larch_buffer_alloc_host(nullptr, 16, &hb, &host));
  larch_event_destroy(e);
  larch_buffer_free(hb);
  int32_t gpus = 0;
  OK(larch_device_count(&gpus));
  if (gpus > 0) {
    larch_device* gpu = nullptr;
    OK(larch_device_open(0, &gpu));
    CHECK(larch_device_set_threads(gpu, 2) == LARCH_ERR_UNSUPPORTED);
    larch_device_close(gpu);
  }
  larch_stream_destroy(cs);
  larch_device_close(cpu);
}

static void rejects_bad_arguments() {
  larch_device* d = nullptr;
  int32_t n = -1;
  OK(larch_device_count(&n));
  if (n == 0) BAD(larch_device_open(0, &d));
  CHECK(larch_last_error(nullptr) != nullptr);
  // Each call below has exactly one thing wrong.
  const larch_tensor h = out(LARCH_F16, {4, 8}), h2 = out(LARCH_F16, {1, 2});
  larch_tensor x = f32({4, 8}, random_f32(32));
  x.offset = 4;  // one float past the end
  BAD(larch_add(st, &x, &h, &x));
  CHECK(std::strstr(larch_last_error(dev), "past the end") != nullptr);
  // Two views that used to wrap the bounds check: an offset near 2^64, and 2^62 + 1 rows.
  larch_tensor y = f32({1, 2}, {1, 2});
  y.offset = UINT64_MAX - 3;
  BAD(larch_add(st, &y, &h2, &y));
  y = f32({1, 2}, {1, 2});
  y.shape[0] = (int64_t(1) << 62) + 1;
  y.shape[1] = y.stride[0] = 1;
  larch_tensor yh = y;
  yh.dtype = LARCH_F16;
  BAD(larch_add(st, &y, &yh, &y));
  // Overlapping rows, a 3-D quant tensor, and dtypes larch.h does not list.
  larch_tensor ov = f32({4, 8}, random_f32(32));
  ov.stride[0] = 4;
  BAD(larch_add(st, &ov, &h, &ov));
  const std::vector<uint8_t> q8(34, 0);
  const larch_tensor q2 = tensor(LARCH_Q8_0, {1, 32}, q8.data(), q8.size());
  const larch_tensor q3 = {q2.buf, 0, LARCH_Q8_0, 3, {1, 1, 32}, {1, 1, 1}}, h32 = out(LARCH_F16, {1, 32});
  OK(larch_dequant(st, &q2, &h32));
  BAD(larch_dequant(st, &q3, &h32));
  const larch_tensor a32 = f32({4, 8}, random_f32(32)), c44 = out(LARCH_F32, {4, 4});
  BAD(larch_add(st, &a32, &a32, &a32));
  BAD(larch_gemm(st, &a32, &h, nullptr, &c44, 1, 0));
  // One dim wrong per call: these checks keep each op's loops inside its views.
  const larch_tensor x28 = f32({2, 8}, random_f32(16)), w4 = f32({4}, {1, 1, 1, 1}), o28 = out(LARCH_F16, {2, 8});
  const larch_tensor g28 = f16({2, 8}, random_h(16)), g24 = f16({2, 4}, random_h(8)), ids2 = i32({2}, {0, 1});
  const larch_tensor e24 = out(LARCH_F32, {2, 4}), c22 = out(LARCH_F16, {2, 2}), c23 = out(LARCH_F16, {2, 3});
  BAD(larch_embed(st, &g28, &ids2, &e24));               // out narrower than the table
  BAD(larch_rmsnorm(st, &x28, &w4, 1e-6f, &o28));        // w shorter than d
  BAD(larch_silu_mul(st, &g28, &g24, &o28));             // up narrower than gate
  BAD(larch_add(st, &x28, &g24, &x28));                  // b narrower than a
  BAD(larch_gemm(st, &g28, &g24, nullptr, &c22, 1, 0));  // k of a and b_t differ
  BAD(larch_gemm(st, &g28, &g28, nullptr, &c23, 1, 0));  // c is not [m, n]
  const larch_tensor h16o = out(LARCH_F16, {1, 16});
  BAD(larch_dequant(st, &q2, &h16o));                    // out narrower than w
  const larch_tensor q18 = f16({2, 1, 8}, random_h(16)), p1 = i32({1}, {0}), p2 = i32({2}, {0, 1});
  const larch_tensor ff3 = f32({3}, {1, 1, 1});
  BAD(larch_qk_norm(st, &q18, &w4, 1e-6f));              // w shorter than head_dim
  const larch_rope_params r8 = {8, 0, 4096, 1e4f, 1, 0, 1, 32, 1, nullptr};
  const larch_rope_params r10 = {10, 0, 4096, 1e4f, 1, 0, 1, 32, 1, nullptr};
  const larch_rope_params rff = {8, 0, 4096, 1e4f, 1, 0, 1, 32, 1, &ff3};
  BAD(larch_rope(st, &q18, &p1, &r8));                   // one position for two rows
  BAD(larch_rope(st, &q18, &p2, &r10));                  // n_dims > head_dim
  BAD(larch_rope(st, &q18, &p2, &rff));                  // freq_factors not [n_dims / 2]
  // A pool of 2 layers, 1 block of 16 slots, 1 kv head of 8.
  const larch_tensor pkb = out(LARCH_F16, {2 * 16 * 8}), pvb = out(LARCH_F16, {2 * 16 * 8});
  const larch_kv_pool kp = {pkb.buf, pvb.buf, 1, 16, 2, 1, 8, LARCH_F16, 0};
  larch_kv_pool kp3 = kp;
  kp3.n_layers = 3;
  const larch_tensor k14 = f16({1, 1, 4}, random_h(4)), k18 = f16({1, 1, 8}, random_h(8)), s0 = i32({1}, {0});
  const larch_tensor t11 = i32({1, 1}, {0}), t21 = i32({2, 1}, {0, 0}), l1 = i32({1}, {1}), l2 = i32({1}, {2});
  const larch_tensor o114 = out(LARCH_F16, {1, 1, 4}), o118 = out(LARCH_F16, {1, 1, 8});
  BAD(larch_kv_write(st, &k14, &k14, &s0, &kp));                     // head_dim 4 into a pool of 8
  BAD(larch_kv_write(st, &k18, &k18, &s0, &kp3));                    // buffers too small for 3 layers
  BAD(larch_attn_decode(st, &k14, &kp, &t11, &l1, 0, 1, &o114));     // q's head_dim is not the pool's
  BAD(larch_attn_decode(st, &k18, &kp, &t11, &l1, 0, 1, &o114));     // out is not q's shape
  BAD(larch_attn_decode(st, &k18, &kp, &t21, &l1, 0, 1, &o118));     // two table rows for one q row
  BAD(larch_attn_prefill(st, &k18, &kp, &t11, &s0, &l2, 1, &o118));  // q_lens sum to 2, q has 1 row
  const larch_tensor lg64 = f32({1, 64}, random_f32(64)), v32 = out(LARCH_F32, {1, 32});
  const larch_tensor i64 = out(LARCH_I32, {1, 64}), lse1 = out(LARCH_F32, {1});
  const larch_sample_params sp1 = {1, 0, 1, 0};
  BAD(larch_sample_prep(st, &lg64, &sp1, nullptr, &v32, &i64, &lse1));  // vals not [rows, 64]
  // Copies need our buffer and a host pointer; a GPU buffer never reaches a CPU stream.
  float f = 0;
  BAD(larch_upload(st, nullptr, 0, &f, 4));
  BAD(larch_download(st, h.buf, 0, nullptr, 4));
  if (n > 0) {
    larch_device* gpu = nullptr;
    larch_buffer* gb = nullptr;
    OK(larch_device_open(0, &gpu));
    OK(larch_buffer_alloc(gpu, 4, &gb));
    BAD(larch_upload(st, gb, 0, &f, 4));
    larch_buffer_free(gb);
    larch_device_close(gpu);
  }
  CHECK(larch_abi_version() == LARCH_ABI_VERSION);
}

int main() {
  OK(larch_device_open(-1, &dev));
  OK(larch_stream_create(dev, &st));
  f16_conversion();
  quant_known_answers();
  gemm_by_hand();
  rope_positions();
  rope_multi_positions();
  attention_vs_naive();
  elementwise_by_definition();
  vit_ops_by_definition();
  sample_prep_order_and_penalty();
  kv_pages();
  experts();
  attention_window_and_sinks();
  device_sampler();
  moe_routing();
  cpu_device();
  rejects_bad_arguments();
  for (larch_buffer* b : owned) larch_buffer_free(b);
  larch_stream_destroy(st);
  larch_device_close(dev);
  std::printf("%s (%d failures)\n", failures ? "FAIL" : "ok", failures);
  return failures ? 1 : 0;
}
