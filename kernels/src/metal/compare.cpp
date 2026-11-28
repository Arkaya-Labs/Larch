// The Metal-only cases, on GPU 0 against the CPU reference (ordinal -1), fed the same bytes. The per-op
// comparison is A2's kernels/tests/ops.cpp; this file holds what only Metal has: misaligned offsets and
// head_dim > 256 (UNSUPPORTED, after every INVALID_ARG check), weights read from a wrapped (zero-copy)
// buffer and every op's rejection of a read-only output, plus the one op combination ops.cpp does not run
// yet and a chain of ops with no sync between them. Tolerances are ops.cpp's (TOLERANCES), in its form.
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "args.h"
#include "larch.h"

struct side {
  larch_device* dev = nullptr;
  larch_stream* s = nullptr;
  std::vector<larch_buffer*> owned;
};
static side sides[2];  // the CPU reference, then Metal
static std::mt19937 rng(1);
static int failures = 0;

#define CHECK(cond, ...)                         \
  do {                                           \
    if (!(cond)) {                               \
      std::fprintf(stderr, "FAIL " __VA_ARGS__); \
      std::fprintf(stderr, "\n");                \
      ++failures;                                \
    }                                            \
  } while (0)

static uint16_t to_h(float f) {
  const _Float16 x = _Float16(f);
  uint16_t u;
  std::memcpy(&u, &x, 2);
  return u;
}
static float from_h(uint16_t u) {
  _Float16 x;
  std::memcpy(&x, &u, 2);
  return float(x);
}
static float uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }

static int64_t block_of(larch_dtype t) {
  return t == LARCH_NQ3 ? 512
         : t == LARCH_Q3_K || t == LARCH_Q4_K || t == LARCH_Q5_K || t == LARCH_Q6_K ? 256
         : t >= LARCH_Q8_0 && t <= LARCH_Q5_0                                         ? 32
                                                                                        : 1;
}
static int64_t bytes_of(larch_dtype t) {  // per block, or per element for plain types
  static const int64_t b[] = {4, 2, 2, 4, 144, 176, 210, 34, 18, 22, 1, 110, 84, 220};
  return b[t];
}

// A tensor's layout and bytes: dense after the first dim, rows `pad` elements (blocks) longer than dense.
struct host {
  larch_dtype t;
  std::vector<int64_t> shape;
  int64_t stride0;
  std::vector<uint8_t> data;
};

// Random contents: normal floats times `scale`, random quant blocks with sane scales, zero i32.
static host make(larch_dtype t, std::vector<int64_t> shape, int64_t pad = 0, float scale = 1) {
  int64_t row = 1;
  for (size_t d = 1; d < shape.size(); ++d) row *= d + 1 == shape.size() ? shape[d] / block_of(t) : shape[d];
  host h{t, shape, row + pad, std::vector<uint8_t>(size_t(shape[0] * (row + pad) * bytes_of(t)))};
  std::normal_distribution<float> normal(0, scale);
  uint8_t* p = h.data.data();
  const size_t n = h.data.size() / size_t(bytes_of(t));
  const auto put = [](uint8_t* at, float v) {
    const uint16_t x = to_h(v);
    std::memcpy(at, &x, 2);
  };
  for (size_t i = 0; i < n; ++i) {
    uint8_t* e = p + i * size_t(bytes_of(t));
    switch (t) {
      case LARCH_F32: {
        const float v = normal(rng);
        std::memcpy(e, &v, 4);
        break;
      }
      case LARCH_F16: put(e, normal(rng)); break;
      case LARCH_I32: std::memset(e, 0, 4); break;
      default:
        for (int64_t j = 0; j < bytes_of(t); ++j) e[j] = uint8_t(rng());
        if (t == LARCH_Q6_K) put(e + 208, uniform(-0.01f, 0.01f));
        else if (t == LARCH_Q3_K) put(e + 108, uniform(-0.01f, 0.01f));  // d follows hmask, qs and scales
        else if (t == LARCH_NQ3) put(e + 216, uniform(-0.01f, 0.01f)), put(e + 218, uniform(-0.01f, 0.01f));
        else if (t == LARCH_Q4_K || t == LARCH_Q5_K) put(e, uniform(0, 0.01f)), put(e + 2, uniform(0, 0.01f));
        else put(e, uniform(-0.05f, 0.05f));
    }
  }
  return h;
}

// i32 values v, row-major in `shape`, with rows `pad` elements longer than dense.
static host ints(std::vector<int64_t> shape, const std::vector<int32_t>& v, int64_t pad = 0) {
  host h = make(LARCH_I32, shape, pad);
  const int64_t row = h.stride0 - pad;
  for (size_t i = 0; i < v.size(); ++i)
    std::memcpy(&h.data[size_t(int64_t(i) / row * h.stride0 + int64_t(i) % row) * 4], &v[i], 4);
  return h;
}

// The same host tensor on both sides, `off` bytes into a buffer of its own.
struct both {
  larch_tensor t[2];
  host h;
  uint64_t off;
};

static larch_tensor tensor_on(larch_buffer* b, const host& h, uint64_t off) {
  larch_tensor x = {b, off, h.t, uint32_t(h.shape.size()), {}, {}};
  int64_t stride = 1;
  for (size_t i = h.shape.size(); i-- > 0;) {
    x.shape[i] = h.shape[i];
    x.stride[i] = i ? stride : h.stride0;
    stride *= i + 1 == h.shape.size() ? h.shape[i] / block_of(h.t) : h.shape[i];
  }
  return x;
}

static both up(const host& h, uint64_t off = 256) {
  both b{{}, h, off};
  for (int i = 0; i < 2; ++i) {
    larch_buffer* buf = nullptr;
    if (larch_buffer_alloc(sides[i].dev, off + h.data.size(), &buf) != LARCH_OK ||
        larch_upload(sides[i].s, buf, off, h.data.data(), h.data.size()) != LARCH_OK) {
      std::fprintf(stderr, "setup failed: %s\n", larch_last_error(sides[i].dev));
      std::exit(1);
    }
    sides[i].owned.push_back(buf);
    b.t[i] = tensor_on(buf, h, off);
  }
  return b;
}

static std::vector<uint8_t> down(int i, const both& b) {
  std::vector<uint8_t> v(b.h.data.size());
  if (larch_download(sides[i].s, b.t[i].buf, b.off, v.data(), v.size()) != LARCH_OK ||
      larch_stream_sync(sides[i].s) != LARCH_OK) {
    std::fprintf(stderr, "download failed: %s\n", larch_last_error(sides[i].dev));
    std::exit(1);
  }
  return v;
}

// Runs op on the CPU reference (0) and Metal (1); both must return LARCH_OK.
static void run(const std::string& what, const std::function<larch_status(int)>& op) {
  for (int i = 0; i < 2; ++i) {
    const larch_status st = op(i);
    CHECK(st == LARCH_OK, "%s on %s: status %d (%s)", what.c_str(), i ? "Metal" : "CPU", st, larch_last_error(sides[i].dev));
  }
}

static double val(larch_dtype t, const std::vector<uint8_t>& v, size_t i) {
  if (t == LARCH_F16) return from_h(uint16_t(v[2 * i] | v[2 * i + 1] << 8));
  int32_t x;
  float f;
  std::memcpy(&x, &v[4 * i], 4);
  std::memcpy(&f, &v[4 * i], 4);
  return t == LARCH_I32 ? double(x) : double(f);
}

// The elements of g that differ from c, the first in *first. An element agrees bit-equal, or finite on both
// sides with |g - c| <= rtol |c| + atol max|c| (max over c's finite elements); i32 only bit-equal.
static size_t differ(larch_dtype t, const std::vector<uint8_t>& c, const std::vector<uint8_t>& g, double rtol,
                     double atol, size_t* first) {
  const size_t eb = t == LARCH_F16 ? 2 : 4, n = c.size() / eb;
  double mx = 0;
  for (size_t i = 0; i < n; ++i)
    if (std::isfinite(val(t, c, i))) mx = std::max(mx, std::fabs(val(t, c, i)));
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!std::memcmp(&c[i * eb], &g[i * eb], eb)) continue;
    const double a = val(t, c, i), b = val(t, g, i);
    if (t != LARCH_I32 && std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= rtol * std::fabs(a) + atol * mx)
      continue;
    if (!bad++) *first = i;
  }
  return bad;
}

// out on both sides agrees (differ) over its whole extent, row padding included.
static void same(const std::string& what, const both& out, double rtol, double atol) {
  const std::vector<uint8_t> c = down(0, out), g = down(1, out);
  size_t first = 0;
  const size_t bad = differ(out.h.t, c, g, rtol, atol, &first);
  CHECK(!bad, "%s: %zu elements differ, first [%zu] cpu %.9g metal %.9g", what.c_str(), bad, first,
        val(out.h.t, c, first), val(out.h.t, g, first));
  if (!bad) std::printf("ok   %s\n", what.c_str());
}

// Exercise this test's comparison helper with corrupted sentinels, in both output precisions.
static void nonfinite() {
  for (const larch_dtype t : {LARCH_F32, LARCH_F16}) {
    const size_t eb = t == LARCH_F16 ? 2 : 4;
    for (const auto [ref, got] : {std::pair{-INFINITY, 0.0f}, std::pair{-INFINITY, INFINITY},
                                 std::pair{INFINITY, 0.0f}, std::pair{INFINITY, -INFINITY},
                                 std::pair{1.0f, -INFINITY}, std::pair{NAN, 0.0f}, std::pair{1.0f, NAN},
                                 std::pair{1.0f, 2.0f}}) {
      std::vector<uint8_t> c(eb), g(eb);
      const uint16_t rh = to_h(ref), gh = to_h(got);
      std::memcpy(c.data(), t == LARCH_F16 ? static_cast<const void*>(&rh) : &ref, eb);
      std::memcpy(g.data(), t == LARCH_F16 ? static_cast<const void*>(&gh) : &got, eb);
      size_t first = 9;
      CHECK(differ(t, c, g, 1e-6, 0, &first) == 1 && first == 0, "corrupted %d output passes the comparison", t);
      CHECK(differ(t, c, c, 1e-6, 0, &first) == 0, "bit-equal %d output fails the comparison", t);
    }
  }
  std::printf("ok   f16/f32 infinity and NaN corruption fails the comparison\n");
}

// f16 c with bias and beta 0 at an edge prefill shape: not in ops.cpp's gemm grid yet (#42).
static void gemm_bias_edge() {
  const both a = up(make(LARCH_F16, {40, 100}, 3)), bt = up(make(LARCH_F16, {70, 100}));
  const both b = up(make(LARCH_F32, {70})), c = up(make(LARCH_F16, {40, 70}, 2));
  run("gemm", [&](int i) { return larch_gemm(sides[i].s, &a.t[i], &bt.t[i], &b.t[i], &c.t[i], 1, 0); });
  same("gemm 40x70x100 bias f16", c, 2e-3, 1e-4);
}

// The tensor prefill shape crosses the 64-wide column tile edge, with padded rows and aligned nonzero offsets.
static void gemm_tensor_contract() {
  const both a = up(make(LARCH_F16, {65, 97}, 3), 256);
  const both bt = up(make(LARCH_F16, {67, 97}, 5), 512);
  const both bias = up(make(LARCH_F32, {67}, 2), 768);
  for (const larch_dtype out_type : {LARCH_F16, LARCH_F32}) {
    for (const bool with_bias : {false, true}) {
      const both c = up(make(out_type, {65, 67}, 7), 1024);
      const std::string name = "gemm tensor 65x67x97 " + std::string(out_type == LARCH_F16 ? "f16" : "f32") +
                               (with_bias ? " bias" : " no bias");
      run(name, [&](int i) {
        return larch_gemm(sides[i].s, &a.t[i], &bt.t[i], with_bias ? &bias.t[i] : nullptr, &c.t[i], 0.75f, -0.25f);
      });
      same(name, c, out_type == LARCH_F16 ? 2e-3 : 1e-4, 1e-4);
    }
  }
}

// k <= n puts c's row tiles on the grid's x (backend.mm gemm): several ragged tiles each way (128 x 64 for
// the tensor path, 32 x 32 for simdgroup), beta and bias.
static void gemm_rows_first() {
  const both a = up(make(LARCH_F16, {161, 96}, 3), 256);
  const both bt = up(make(LARCH_F16, {130, 96}, 5), 512);
  const both bias = up(make(LARCH_F32, {130}, 2), 768);
  for (const larch_dtype out_type : {LARCH_F16, LARCH_F32}) {
    const both c = up(make(out_type, {161, 130}, 7), 1024);
    const std::string name = std::string("gemm rows first 161x130x96 ") + (out_type == LARCH_F16 ? "f16" : "f32");
    run(name, [&](int i) { return larch_gemm(sides[i].s, &a.t[i], &bt.t[i], &bias.t[i], &c.t[i], 0.75f, -0.25f); });
    same(name, c, out_type == LARCH_F16 ? 2e-3 : 1e-4, 1e-4);
  }
}

// k > n at 256+ rows takes the deep tensor tile (args.h GEMM_TENSOR_DEEP_M): two ragged row tiles, beta and bias.
static void gemm_deep() {
  const both a = up(make(LARCH_F16, {300, 96}, 3), 256);
  const both bt = up(make(LARCH_F16, {70, 96}, 5), 512);
  const both bias = up(make(LARCH_F32, {70}, 2), 768);
  for (const larch_dtype out_type : {LARCH_F16, LARCH_F32}) {
    const both c = up(make(out_type, {300, 70}, 7), 1024);
    const std::string name = std::string("gemm deep 300x70x96 ") + (out_type == LARCH_F16 ? "f16" : "f32");
    run(name, [&](int i) { return larch_gemm(sides[i].s, &a.t[i], &bt.t[i], &bias.t[i], &c.t[i], 0.75f, -0.25f); });
    same(name, c, out_type == LARCH_F16 ? 2e-3 : 1e-4, 1e-4);
  }
}

// More tile paths (backend.mm gemm): 128-row tensor tiles with x on columns across two row tiles; rows first at
// an odd k; the deep tile at a long odd k. Bias and beta on each.
static void gemm_tile_paths() {
  struct shape {
    int64_t m, n, k;
  };
  for (const shape c : {shape{200, 70, 96}, shape{161, 1100, 1031}, shape{300, 70, 4099}}) {
    const both a = up(make(LARCH_F16, {c.m, c.k}, 3), 256);
    const both bt = up(make(LARCH_F16, {c.n, c.k}, 5), 512);
    const both bias = up(make(LARCH_F32, {c.n}, 2), 768);
    const both out = up(make(LARCH_F32, {c.m, c.n}, 7), 1024);
    const std::string name = "gemm tiles " + std::to_string(c.m) + "x" + std::to_string(c.n) + "x" + std::to_string(c.k);
    run(name, [&](int i) { return larch_gemm(sides[i].s, &a.t[i], &bt.t[i], &bias.t[i], &out.t[i], 0.75f, -0.25f); });
    same(name, out, 1e-4, 1e-4);
  }
  // vv_gemv (up to GEMV_M rows): a 32-column and a 1-column matrix, k past the 4-wide loads (2051, 97 tails),
  // a and b_t only 2-byte aligned.
  for (const shape c : {shape{1, 32, 2051}, shape{8, 1, 97}, shape{3, 130, 4096}}) {
    const both a = up(make(LARCH_F16, {c.m, c.k}, 3), 258);
    const both bt = up(make(LARCH_F16, {c.n, c.k}, 5), 514);
    const both bias = up(make(LARCH_F32, {c.n}, 2), 768);
    const both out = up(make(LARCH_F32, {c.m, c.n}, 7), 1024);
    const std::string name = "gemv " + std::to_string(c.m) + "x" + std::to_string(c.n) + "x" + std::to_string(c.k);
    run(name, [&](int i) { return larch_gemm(sides[i].s, &a.t[i], &bt.t[i], &bias.t[i], &out.t[i], 0.75f, -0.25f); });
    same(name, out, 1e-4, 1e-4);
  }
}

static void gemm_tensor_fallbacks() {
  constexpr int64_t n = 67, k = 97;
  const both a = up(make(LARCH_F16, {16, k}, 3), 256);
  const both bt = up(make(LARCH_F16, {n, k}, 5), 512);
  const both bias = up(make(LARCH_F32, {n}, 2), 768);
  const both c = up(make(LARCH_F32, {16, n}, 7), 1024);
  run("gemm tensor fallback 16 rows", [&](int i) {
    return larch_gemm(sides[i].s, &a.t[i], &bt.t[i], &bias.t[i], &c.t[i], 0.75f, -0.25f);
  });
  same("gemm tensor fallback 16 rows", c, 1e-4, 1e-4);

  host untouched = make(LARCH_F16, {0, n}, 7);
  untouched.data.assign(64, 0xA5);
  const both zero_a = up(make(LARCH_F16, {0, k}, 3), 256), zero_c = up(untouched, 1024);
  const std::vector<uint8_t> before = zero_c.h.data;
  run("gemm tensor fallback 0 rows", [&](int i) {
    return larch_gemm(sides[i].s, &zero_a.t[i], &bt.t[i], nullptr, &zero_c.t[i], 0.75f, -0.25f);
  });
  CHECK(down(0, zero_c) == before && down(1, zero_c) == before, "gemm tensor fallback 0 rows wrote output");

  for (int i = 0; i < 2; ++i) {
    larch_tensor wide_a = zero_a.t[i];
    wide_a.stride[0] = int64_t(std::numeric_limits<int>::max()) + 1;
    CHECK(larch_gemm(sides[i].s, &wide_a, &bt.t[i], nullptr, &zero_c.t[i], 0.75f, -0.25f) == LARCH_OK,
          "gemm tensor fallback oversized stride on %s", i ? "Metal" : "CPU");
  }
  CHECK(down(0, zero_c) == before && down(1, zero_c) == before,
        "gemm tensor fallback oversized stride wrote output");
}

// The deferred dequant (backend.mm dequant). Its gemm reads w through gemm_q: every format, both tile heights
// (up to 64 rows and more) by both widths (k > n and k <= n), ragged n, beta and bias, f16 and f32 out; vv_gemm_qs
// without tensor GEMM. A k that is no multiple of GEMM_Q_K (tensor path only) and a gemm on part of the scratch
// take the plain path. Its output is right wherever anything else
// sees it: a download after the gemm, another op reading it, a write into its buffer, a write to w before the
// gemm, a dequant that covers it, one that covers its first rows, one that ends inside a row, one with gaps, one
// that reads it, one that writes over its w, the gemm reading it as a too, the gemm writing over w, another
// stream after a sync.
static void deferred_dequant() {
  const auto dequant_gemm = [](const both& w, const larch_tensor* dq, const both& a, const both* bias, const both& c) {
    return [&w, dq, &a, bias, &c](int i) {
      const larch_status st = larch_dequant(sides[i].s, &w.t[i], &dq[i]);
      return st ? st : larch_gemm(sides[i].s, &a.t[i], &dq[i], bias ? &bias->t[i] : nullptr, &c.t[i], 0.75f, -0.25f);
    };
  };
  for (const larch_dtype q :
       {LARCH_Q3_K, LARCH_Q4_K, LARCH_Q5_K, LARCH_Q6_K, LARCH_Q8_0, LARCH_Q4_0, LARCH_Q5_0, LARCH_NQ3})
    for (const auto& [m, n, k] : {std::tuple<int64_t, int64_t, int64_t>{17, 67, 512}, {130, 67, 512}, {40, 300, 256},
                                  {130, 300, 256}}) {
      // 256 or 128 for the 32-element formats; nq3 512 and 1024
      const int64_t kq = q == LARCH_NQ3 ? 2 * k : block_of(q) == 256 ? k : k / 2;
      const larch_dtype ot = m == 17 ? LARCH_F16 : LARCH_F32;
      const both w = up(make(q, {n, kq}, 1)), dq = up(make(LARCH_F16, {n, kq}, 3));
      const both a = up(make(LARCH_F16, {m, kq}, 5)), bias = up(make(LARCH_F32, {n})), c = up(make(ot, {m, n}, 2));
      const std::string name = "deferred dequant " + std::to_string(int(q)) + ", gemm " + std::to_string(m) + "x" +
                               std::to_string(n) + "x" + std::to_string(kq);
      run(name, dequant_gemm(w, dq.t, a, &bias, c));
      same(name, c, ot == LARCH_F16 ? 2e-3 : 1e-4, 1e-4);
      same(name + ", then its scratch", dq, 1e-3, 1e-6);
    }
  {
    const both w = up(make(LARCH_Q8_0, {40, 96})), dq = up(make(LARCH_F16, {40, 96}));
    const both a = up(make(LARCH_F16, {33, 96})), c = up(make(LARCH_F16, {33, 40}));
    const both c2 = up(make(LARCH_F16, {33, 32}));
    run("deferred dequant, gemm at k 96", dequant_gemm(w, dq.t, a, nullptr, c));
    same("deferred dequant, gemm at k 96", c, 2e-3, 1e-4);
    larch_tensor part[2] = {dq.t[0], dq.t[1]};
    for (larch_tensor& p : part) p.offset += 8 * 96 * 2, p.shape[0] = 32;
    run("deferred dequant, gemm on rows 8-39", [&](int i) {
      const larch_status st = larch_dequant(sides[i].s, &w.t[i], &dq.t[i]);
      return st ? st : larch_gemm(sides[i].s, &a.t[i], &part[i], nullptr, &c2.t[i], 1, 0);
    });
    same("deferred dequant, gemm on rows 8-39", c2, 2e-3, 1e-4);
  }

  constexpr int64_t n = 64, k = 256;
  const both w1 = up(make(LARCH_Q4_K, {n, k})), w2 = up(make(LARCH_Q6_K, {n, k}));
  {
    const both dq = up(make(LARCH_F16, {n, k})), r = up(make(LARCH_F32, {n, k})), out = up(make(LARCH_F32, {n, k}));
    run("deferred dequant, then add reads it", [&](int i) {
      const larch_status st = larch_dequant(sides[i].s, &w1.t[i], &dq.t[i]);
      return st ? st : larch_add(sides[i].s, &r.t[i], &dq.t[i], &out.t[i]);
    });
    same("deferred dequant, then add reads it", out, 1e-3, 1e-3);
  }
  {
    const both dq = up(make(LARCH_F16, {n, k}));
    const host patch = make(LARCH_F16, {5, k});
    run("deferred dequant, then an upload into rows 5-9", [&](int i) {
      const larch_status st = larch_dequant(sides[i].s, &w1.t[i], &dq.t[i]);
      return st ? st : larch_upload(sides[i].s, dq.t[i].buf, dq.off + 5 * k * 2, patch.data.data(), patch.data.size());
    });
    same("deferred dequant, then an upload into rows 5-9", dq, 1e-3, 1e-6);
  }
  {
    const both w = up(w1.h), dq = up(make(LARCH_F16, {n, k}));
    const both a = up(make(LARCH_F16, {20, k})), c = up(make(LARCH_F16, {20, n}));
    run("deferred dequant, w overwritten before its gemm", [&](int i) {
      larch_status st = larch_dequant(sides[i].s, &w.t[i], &dq.t[i]);
      if (!st) st = larch_upload(sides[i].s, w.t[i].buf, w.off, w2.h.data.data(), w.h.data.size());
      return st ? st : larch_gemm(sides[i].s, &a.t[i], &dq.t[i], nullptr, &c.t[i], 1, 0);
    });
    same("deferred dequant, w overwritten before its gemm", c, 2e-3, 1e-4);
  }
  for (const bool covering : {false, true}) {
    const both dq = up(make(LARCH_F16, {n, k})), w3 = up(make(LARCH_Q4_K, {24, k}));
    larch_tensor head[2] = {dq.t[0], dq.t[1]};
    for (larch_tensor& h : head) h.shape[0] = 24;
    const std::string name = covering ? "deferred dequant into rows 0-23, then one into all" :
                                        "deferred dequant into all rows, then one into rows 0-23";
    run(name, [&](int i) {
      const auto all = [&] { return larch_dequant(sides[i].s, &w1.t[i], &dq.t[i]); };
      const auto rows = [&] { return larch_dequant(sides[i].s, &w3.t[i], &head[i]); };
      const larch_status st = covering ? rows() : all();
      return st ? st : covering ? all() : rows();
    });
    same(name, dq, 1e-3, 1e-6);
  }
  {  // gemm reading the scratch as a too, and writing c over w's bytes (w is no gemm input): both see w1
    constexpr int64_t m = 20;
    const both w = up(w1.h), dq = up(make(LARCH_F16, {n, k})), c = up(make(LARCH_F16, {m, n}));
    const both a = up(make(LARCH_F16, {m, k})), bias = up(make(LARCH_F32, {n}));
    larch_tensor rows[2] = {dq.t[0], dq.t[1]};
    for (larch_tensor& r : rows) r.shape[0] = m;
    run("deferred dequant, gemm reads it as a too", [&](int i) {
      const larch_status st = larch_dequant(sides[i].s, &w.t[i], &dq.t[i]);
      return st ? st : larch_gemm(sides[i].s, &rows[i], &dq.t[i], &bias.t[i], &c.t[i], 1, 0);
    });
    same("deferred dequant, gemm reads it as a too", c, 2e-3, 1e-4);
    both over{{}, make(LARCH_F16, {m, n}), w.off};  // 2560 of w's 9216 bytes
    for (int i = 0; i < 2; ++i) over.t[i] = tensor_on(w.t[i].buf, over.h, over.off);
    run("deferred dequant, gemm writes over w", [&](int i) {
      const larch_status st = larch_dequant(sides[i].s, &w.t[i], &dq.t[i]);
      return st ? st : larch_gemm(sides[i].s, &a.t[i], &dq.t[i], nullptr, &over.t[i], 1, 0);
    });
    same("deferred dequant, gemm writes over w", over, 2e-3, 1e-4);
    same("deferred dequant, gemm writes over w, then the scratch", dq, 1e-3, 1e-6);
  }
  // A dequant over the first 4.5 rows of a pending one with 1024-byte rows: its row 4 crosses the end and runs,
  // rows 5 on stay pending. Then one at bytes 512-1023 runs the 4.5-row one alone, before the rows that stay.
  {
    const both w = up(make(LARCH_Q4_K, {16, 512})), dq = up(make(LARCH_F16, {16, 512}));
    const both w9 = up(make(LARCH_Q4_K, {9, 256})), w1r = up(make(LARCH_Q4_K, {1, 256}));
    larch_tensor head[2], mid[2];
    for (int i = 0; i < 2; ++i) {
      head[i] = tensor_on(dq.t[i].buf, w9.h, dq.off), mid[i] = tensor_on(dq.t[i].buf, w1r.h, dq.off + 512);
      head[i].dtype = mid[i].dtype = LARCH_F16, head[i].stride[0] = mid[i].stride[0] = 256;
    }
    run("deferred dequant, then one ending inside its row 4", [&](int i) {
      larch_status st = larch_dequant(sides[i].s, &w.t[i], &dq.t[i]);
      if (!st) st = larch_dequant(sides[i].s, &w9.t[i], &head[i]);
      return st ? st : larch_dequant(sides[i].s, &w1r.t[i], &mid[i]);
    });
    same("deferred dequant, then one ending inside its row 4", dq, 1e-3, 1e-6);
  }
  {  // a dequant with gaps in its output (128 of every 256 halves) over a pending one: the gaps keep its values
    const both dq = up(make(LARCH_F16, {n, k})), w8 = up(make(LARCH_Q8_0, {n, 128}));
    larch_tensor half[2] = {dq.t[0], dq.t[1]};
    for (larch_tensor& h : half) h.shape[1] = 128;
    run("deferred dequant, then one with gaps over it", [&](int i) {
      const larch_status st = larch_dequant(sides[i].s, &w1.t[i], &dq.t[i]);
      return st ? st : larch_dequant(sides[i].s, &w8.t[i], &half[i]);
    });
    same("deferred dequant, then one with gaps over it", dq, 1e-3, 1e-6);
  }
  {  // a dequant reading a pending one's output as its w. Q8_0 with d = 1 gives f16 integers, bit-equal on both sides.
    host ones = make(LARCH_Q8_0, {n, k});
    for (size_t b = 0; b < ones.data.size(); b += 34) ones.data[b] = 0x00, ones.data[b + 1] = 0x3C;
    const both w = up(ones), dq = up(make(LARCH_F16, {n, k})), y = up(make(LARCH_F16, {8, 128}));
    const both a = up(make(LARCH_F16, {17, 128})), c = up(make(LARCH_F16, {17, 8}));
    const host as_q8 = make(LARCH_Q8_0, {8, 128});  // 8 rows of 4 blocks: the scratch's first 1088 bytes
    larch_tensor q8[2] = {tensor_on(dq.t[0].buf, as_q8, dq.off), tensor_on(dq.t[1].buf, as_q8, dq.off)};
    run("deferred dequant, then one reading its output, then its gemm", [&](int i) {
      larch_status st = larch_dequant(sides[i].s, &w.t[i], &dq.t[i]);
      if (!st) st = larch_dequant(sides[i].s, &q8[i], &y.t[i]);
      return st ? st : larch_gemm(sides[i].s, &a.t[i], &y.t[i], nullptr, &c.t[i], 1, 0);
    });
    same("deferred dequant, then one reading its output, then its gemm", c, 2e-3, 1e-4);
    same("deferred dequant, then one reading its output", y, 1e-3, 1e-6);
  }
  // A dequant writing over a pending one's w (writable): the pending one reads w first. A third one at bytes
  // 256-511 of the second runs the second alone.
  {
    const both w = up(w1.h), dq = up(make(LARCH_F16, {n, k})), w8 = up(make(LARCH_Q8_0, {8, 128}));
    const both w8r = up(make(LARCH_Q8_0, {1, 128}));
    both over{{}, make(LARCH_F16, {8, 128}), w.off};  // 2048 of w's 9216 bytes
    larch_tensor inner[2];
    for (int i = 0; i < 2; ++i)
      over.t[i] = tensor_on(w.t[i].buf, over.h, over.off), inner[i] = over.t[i], inner[i].offset += 256, inner[i].shape[0] = 1;
    run("deferred dequant, then one writing over its w", [&](int i) {
      larch_status st = larch_dequant(sides[i].s, &w.t[i], &dq.t[i]);
      if (!st) st = larch_dequant(sides[i].s, &w8.t[i], &over.t[i]);
      return st ? st : larch_dequant(sides[i].s, &w8r.t[i], &inner[i]);
    });
    same("deferred dequant, then one writing over its w", dq, 1e-3, 1e-6);
    same("deferred dequant, then one writing over its w: the new one", over, 1e-3, 1e-6);
  }
  {
    const both dq = up(make(LARCH_F16, {n, k}));
    run("deferred dequant, sync", [&](int i) {
      const larch_status st = larch_dequant(sides[i].s, &w2.t[i], &dq.t[i]);
      return st ? st : larch_stream_sync(sides[i].s);
    });
    std::vector<uint8_t> got[2];
    for (int i = 0; i < 2; ++i) {
      larch_stream* other = nullptr;
      got[i].resize(dq.h.data.size());
      CHECK(larch_stream_create(sides[i].dev, &other) == LARCH_OK &&
                larch_download(other, dq.t[i].buf, dq.off, got[i].data(), got[i].size()) == LARCH_OK,
            "deferred dequant, another stream's download: %s", larch_last_error(sides[i].dev));
      if (other) larch_stream_destroy(other);
    }
    size_t first = 0;
    const size_t bad = differ(LARCH_F16, got[0], got[1], 1e-3, 1e-6, &first);
    CHECK(!bad, "deferred dequant, read on another stream after a sync: %zu elements differ", bad);
    if (!bad) std::printf("ok   deferred dequant, read on another stream after a sync\n");
  }
}

// k 256 leaves 4 of mmvq_mma's 8 simdgroups without a k step.
static void mmvq_rows() {
  constexpr int64_t n = 67;
  for (const int64_t k : {256, 768}) {
    for (const larch_dtype q : {LARCH_Q3_K, LARCH_Q4_K, LARCH_Q5_K, LARCH_Q6_K}) {
      const both w = up(make(q, {n, k}, 2)), bias = up(make(LARCH_F32, {n}, 3));
      for (int64_t rows = 1; rows <= 16; ++rows) {
        const both x = up(make(LARCH_F16, {rows, k}, 5));
        for (const larch_dtype out_type : {LARCH_F16, LARCH_F32}) {
          for (const bool with_bias : {false, true}) {
            const both out = up(make(out_type, {rows, n}, 7));
            const std::string name = "mmvq " + std::string(q == LARCH_Q3_K ? "q3_k" : q == LARCH_Q4_K ? "q4_k" : q == LARCH_Q5_K ? "q5_k" : "q6_k") + " k " +
                                     std::to_string(k) + ", " + std::to_string(rows) + " rows " +
                                     (out_type == LARCH_F16 ? "f16" : "f32") +
                                     (with_bias ? " bias" : " no bias");
            run(name, [&](int i) {
              return larch_mmvq(sides[i].s, &w.t[i], &x.t[i], with_bias ? &bias.t[i] : nullptr, &out.t[i]);
            });
            same(name, out, out_type == LARCH_F16 ? 2e-3 : 1e-3, 1e-3);
          }
        }
      }
    }
  }
}

// mmvq's tile kernel: every format, a ragged last threadgroup, and k = 96 for the 32-element formats (a
// partial 128-deep step). backend.mm takes it from 5 rows at 4 32-column threadgroups per core; n gives 8 per
// core, twice that, so a moved threshold still lands here (cores: as backend.mm counts them, at most 32 when
// the chip does not report them). The same w at 2-4 rows takes the shared-x kernel.
static void mmvq_tiles() {
  larch_device_props props{};
  CHECK(larch_device_props_get(sides[1].dev, &props) == LARCH_OK, "mmvq tiles: device props");
  const int64_t n = 32 * 8 * int64_t(props.compute_units ? props.compute_units : 32) + 5;
  const std::pair<larch_dtype, const char*> formats[] = {{LARCH_Q3_K, "q3_k"}, {LARCH_Q4_K, "q4_k"}, {LARCH_Q5_K, "q5_k"},
                                                          {LARCH_Q6_K, "q6_k"}, {LARCH_Q8_0, "q8_0"},
                                                          {LARCH_Q4_0, "q4_0"}, {LARCH_Q5_0, "q5_0"}};
  for (const auto& [q, format] : formats) {
    const int64_t k = block_of(q) == 256 ? 768 : 96;
    const both w = up(make(q, {n, k}, 2)), bias = up(make(LARCH_F32, {n}, 3));
    // The K-quants from 3 rows (Q3_K from 2): mmvq_mma. The others at 2-4: the shared-x kernel (4 its widest one-tile case);
    // 5 and 8: the 1-tile kernel (8 fills it); 9 and 16: 2
    for (const int64_t rows : {2, 3, 4, 5, 8, 9, 16}) {
      const both x = up(make(LARCH_F16, {rows, k}, 5));
      const larch_dtype out_type = rows == 3 || rows == 8 || rows == 9 ? LARCH_F32 : LARCH_F16;
      const both out = up(make(out_type, {rows, n}, 7));
      const std::string name = "mmvq tiles " + std::string(format) + " " + std::to_string(rows) + " rows x " +
                               std::to_string(n) + (rows == 4 ? "" : " bias");
      run(name, [&](int i) {
        return larch_mmvq(sides[i].s, &w.t[i], &x.t[i], rows == 4 ? nullptr : &bias.t[i], &out.t[i]);
      });
      same(name, out, out_type == LARCH_F16 ? 2e-3 : 1e-3, 1e-3);
    }
  }
}

// mmvq's shared-x kernel at 5 and 7 rows: a w too narrow for the tile kernel (3 of its threadgroups per core,
// 4 needed) with 6 shared-x threadgroups per core (4 needed), every format, a ragged last threadgroup.
static void mmvq_shared_x() {
  larch_device_props props{};
  CHECK(larch_device_props_get(sides[1].dev, &props) == LARCH_OK, "mmvq shared x: device props");
  const int64_t n = 16 * 6 * int64_t(props.compute_units ? props.compute_units : 32) + 3;
  for (const larch_dtype q : {LARCH_Q3_K, LARCH_Q4_K, LARCH_Q5_K, LARCH_Q6_K, LARCH_Q8_0, LARCH_Q4_0, LARCH_Q5_0}) {
    const int64_t k = block_of(q) == 256 ? 768 : 96;
    const both w = up(make(q, {n, k}, 2)), bias = up(make(LARCH_F32, {n}, 3));
    for (const int64_t rows : {5, 7}) {
      const both x = up(make(LARCH_F16, {rows, k}, 5)), out = up(make(LARCH_F16, {rows, n}, 7));
      const std::string name = "mmvq shared x " + std::to_string(int(q)) + " " + std::to_string(rows) + " rows x " +
                               std::to_string(n) + " bias";
      run(name, [&](int i) { return larch_mmvq(sides[i].s, &w.t[i], &x.t[i], &bias.t[i], &out.t[i]); });
      same(name, out, 2e-3, 1e-3);
    }
  }
}

static void mmvq_zero_rows() {
  constexpr int64_t n = 3, k = 256;
  host untouched = make(LARCH_F16, {0, n}, 7);
  untouched.data.assign(64, 0xA5);
  const both w = up(make(LARCH_Q6_K, {n, k}, 2)), x = up(make(LARCH_F16, {0, k}, 5)), out = up(untouched);
  const std::vector<uint8_t> before = out.h.data;
  run("mmvq q6_k 0 rows", [&](int i) { return larch_mmvq(sides[i].s, &w.t[i], &x.t[i], nullptr, &out.t[i]); });
  CHECK(down(0, out) == before && down(1, out) == before, "mmvq q6_k 0 rows wrote output");
}

// A pool of n_blocks blocks of 16 slots over 2 layers, used at layer 1, filled at random.
struct pool2 {
  both k, v;
  larch_kv_pool p[2];
};
static pool2 make_pool(uint32_t n_blocks, uint32_t kv_heads, uint32_t hd) {
  const int64_t n = 2 * int64_t(n_blocks) * 16 * kv_heads * hd;
  pool2 pl{up(make(LARCH_F16, {n}), 0), up(make(LARCH_F16, {n}), 0), {}};
  for (int i = 0; i < 2; ++i) pl.p[i] = {pl.k.t[i].buf, pl.v.t[i].buf, n_blocks, 16, 2, kv_heads, hd, LARCH_F16, 1};
  return pl;
}

static void attn_decode_splits() {
  constexpr uint32_t blocks = 12, kv_heads = 2, head_dim = 128;
  constexpr int64_t rows = 4, heads = 4, table_width = 12;
  const pool2 pool = make_pool(blocks, kv_heads, head_dim);
  const both query = up(make(LARCH_F16, {rows, heads, head_dim}, 3));
  const both block_table = up(ints({rows, table_width},
                                   {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    7, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    2, 5, 0, 11, 4, 8, 3, 10, 6, 9, 7, 1},
                                   2));
  const both sequence_lengths = up(ints({rows}, {0, 1, 17, 183}, 3));
  const float scale = 1.0f / std::sqrt(float(head_dim));
  // splits is a speed hint (larch.h): above 16 runs as 16, and still matches the CPU.
  std::vector<uint32_t> all_splits{17, std::numeric_limits<uint32_t>::max()};
  for (uint32_t splits = 0; splits <= 16; ++splits) all_splits.push_back(splits);
  for (const uint32_t splits : all_splits) {
    const both output = up(make(LARCH_F16, {rows, heads, head_dim}, 5));
    const std::string name = "attn_decode splits " + std::to_string(splits);
    run(name, [&](int i) {
      return larch_attn_decode(sides[i].s, &query.t[i], &pool.p[i], &block_table.t[i], &sequence_lengths.t[i],
                                 splits, scale, &output.t[i]);
    });
    same(name, output, 2e-3, 5e-4);
  }

  // head_dim 6: attend_range's 4-wide k loads end in a scalar tail; 37 positions span 3 blocks and two tiles.
  const pool2 odd_pool = make_pool(3, 1, 6);
  const both odd_query = up(make(LARCH_F16, {1, 2, 6}, 3));
  const both odd_table = up(ints({1, 3}, {2, 0, 1}, 2));
  const both odd_lengths = up(ints({1}, {37}, 3));
  for (const uint32_t splits : {0u, 1u, 2u}) {
    const both odd_output = up(make(LARCH_F16, {1, 2, 6}, 5));
    const std::string odd_name = "attn_decode head_dim 6 over 37 positions, splits " + std::to_string(splits);
    run(odd_name, [&](int i) {
      return larch_attn_decode(sides[i].s, &odd_query.t[i], &odd_pool.p[i], &odd_table.t[i], &odd_lengths.t[i],
                                 splits, 1.0f / std::sqrt(6.0f), &odd_output.t[i]);
    });
    same(odd_name, odd_output, 2e-3, 5e-4);
  }

  // head_dim 160 and 256: the accumulator's second 128-dim chunk (160 ends inside it), 4 q heads per kv head.
  for (const uint32_t hd : {160u, 256u}) {
    const pool2 wide_pool = make_pool(3, 1, hd);
    const both wide_query = up(make(LARCH_F16, {1, 4, hd}, 3));
    const both wide_table = up(ints({1, 3}, {1, 2, 0}, 2));
    const both wide_lengths = up(ints({1}, {45}, 3));
    for (const uint32_t splits : {0u, 1u, 2u}) {
      const both wide_output = up(make(LARCH_F16, {1, 4, hd}, 5));
      const std::string wide_name =
          "attn_decode head_dim " + std::to_string(hd) + " over 45 positions, splits " + std::to_string(splits);
      run(wide_name, [&](int i) {
        return larch_attn_decode(sides[i].s, &wide_query.t[i], &wide_pool.p[i], &wide_table.t[i],
                                   &wide_lengths.t[i], splits, 1.0f / std::sqrt(float(hd)), &wide_output.t[i]);
      });
      same(wide_name, wide_output, 2e-3, 5e-4);
    }
  }

  const pool2 narrow_pool = make_pool(1, 1, 8);
  const both narrow_query = up(make(LARCH_F16, {2, 1, 8}, 3));
  const both narrow_table = up(ints({2, 1}, {-1, 0}, 2));
  const both narrow_lengths = up(ints({2}, {0, 1}, 3));
  const both narrow_output = up(make(LARCH_F16, {2, 1, 8}, 5));
  const std::string narrow_name = "attn_decode splits 16 head_dim 8";
  run(narrow_name, [&](int i) {
    return larch_attn_decode(sides[i].s, &narrow_query.t[i], &narrow_pool.p[i], &narrow_table.t[i],
                               &narrow_lengths.t[i], 16, 1.0f / std::sqrt(8.0f), &narrow_output.t[i]);
  });
  same(narrow_name, narrow_output, 2e-3, 5e-4);

  // A few-head full-attention shape (16 q heads, 2 kv heads, head_dim 256) with automatic splits: 8200 positions
  // run 64 splits (vv_attn_split's past-16 splits, the reduce's second split a lane), 300 run 16.
  constexpr int32_t long_blocks = 520;
  const pool2 long_pool = make_pool(long_blocks, 2, 256);
  std::vector<int32_t> long_ids(2 * long_blocks);
  for (int32_t b = 0; b < long_blocks; ++b) long_ids[size_t(b)] = (b * 7 + 3) % long_blocks;  // 7 is coprime
  for (int32_t b = 0; b < long_blocks; ++b) long_ids[size_t(long_blocks + b)] = long_blocks - 1 - b;
  const both long_query = up(make(LARCH_F16, {2, 16, 256}, 3));
  const both long_table = up(ints({2, long_blocks}, long_ids, 2));
  const both long_lengths = up(ints({2}, {8200, 300}, 3));
  const both long_output = up(make(LARCH_F16, {2, 16, 256}, 5));
  const std::string long_name = "attn_decode 16 q / 2 kv heads of 256, 8200 and 300 positions, automatic splits";
  run(long_name, [&](int i) {
    return larch_attn_decode(sides[i].s, &long_query.t[i], &long_pool.p[i], &long_table.t[i], &long_lengths.t[i], 0,
                               1.0f / 16.0f, &long_output.t[i]);
  });
  same(long_name, long_output, 2e-3, 5e-4);
}

// Flash prefill (backend.mm attend): 32-row tiles across sequences, cached prefixes, mid-block starts, an empty
// chunk, GQA 7, head_dim 64, 96 (inside the 128 tile), 128 and 256, q rows 2-byte aligned, 2-3 rows per sequence,
// and a few hundred positions; head_dim 6 keeps vv_attn.
static void attn_prefill_flash() {
  struct shape {
    uint32_t kv_heads, hd;
    int64_t heads, pad;
    std::vector<std::pair<int32_t, int32_t>> chunks;  // (start, len) per sequence
  };
  const shape shapes[] = {
      {2, 64, 14, 8, {{0, 70}, {37, 45}, {5, 0}, {130, 33}}}, {2, 128, 8, 8, {{0, 70}, {37, 45}, {5, 0}, {130, 33}}},
      {1, 96, 4, 0, {{3, 40}}},  {1, 256, 2, 0, {{0, 40}}}, {1, 6, 2, 0, {{17, 40}}}, {2, 64, 4, 3, {{0, 40}}},
      {2, 64, 4, 8, {{9, 2}, {0, 3}, {40, 2}}}, {1, 80, 4, 0, {{3, 40}}}, {1, 40, 4, 0, {{3, 40}}},
      // vv_attn_paged_nax: several 64-row (32 at 256) tiles over many 32-position steps, from a mid-block start
      {2, 128, 8, 0, {{90, 300}, {0, 17}}}, {2, 256, 16, 0, {{7, 260}}},
  };
  constexpr uint32_t blocks = 32;
  for (const shape& c : shapes) {
    std::vector<int32_t> start, len, table;
    int64_t rows = 0, width = 0;
    std::string name = "attn_prefill " + std::to_string(c.heads) + " heads over " + std::to_string(c.kv_heads) +
                       " x " + std::to_string(c.hd) + ", q pad " + std::to_string(c.pad) + ", chunks";
    for (const auto& [st, n] : c.chunks) {
      start.push_back(st);
      len.push_back(n);
      rows += n;
      width = std::max<int64_t>(width, (st + n + 15) / 16);
      name += " " + std::to_string(st) + "+" + std::to_string(n);
    }
    for (size_t seq = 0; seq < c.chunks.size(); ++seq) {  // each sequence's blocks, shuffled
      std::vector<int32_t> ids(blocks);
      std::iota(ids.begin(), ids.end(), 0);
      std::shuffle(ids.begin(), ids.end(), rng);
      table.insert(table.end(), ids.begin(), ids.begin() + width);
    }
    const int64_t seqs = int64_t(c.chunks.size());
    const pool2 pool = make_pool(blocks, c.kv_heads, c.hd);
    const both q = up(make(LARCH_F16, {rows, c.heads, c.hd}, c.pad));
    const both out = up(make(LARCH_F16, {rows, c.heads, c.hd}, c.pad));
    const both tb = up(ints({seqs, width}, table)), st = up(ints({seqs}, start)), ln = up(ints({seqs}, len));
    run(name, [&](int i) {
      return larch_attn_prefill(sides[i].s, &q.t[i], &pool.p[i], &tb.t[i], &st.t[i], &ln.t[i],
                                 1.0f / std::sqrt(float(c.hd)), &out.t[i]);
    });
    same(name, out, 2e-3, 5e-4);
  }

  // q_lens summing past q's rows: a caller bug larch.h forbids and the host cannot see. Nothing past out moves.
  const pool2 pool = make_pool(blocks, 1, 64);
  const both q = up(make(LARCH_F16, {40, 4, 64})), out = up(make(LARCH_F16, {80, 4, 64}));
  const both tb = up(ints({2, 3}, {0, 1, 2, 3, 4, 5})), st = up(ints({2}, {0, 0})), ln = up(ints({2}, {40, 40}));
  larch_tensor head = out.t[1];
  head.shape[0] = 40;
  const auto before = down(1, out);
  CHECK(larch_attn_prefill(sides[1].s, &q.t[1], &pool.p[1], &tb.t[1], &st.t[1], &ln.t[1], 0.125f, &head) ==
            LARCH_OK, "attn_prefill with q_lens past q's rows: status");
  const auto after = down(1, out);
  CHECK(std::equal(before.begin() + 40 * 4 * 64 * 2, before.end(), after.begin() + 40 * 4 * 64 * 2),
        "attn_prefill with q_lens past q's rows writes past out");
}

// Mandatory metadata errors take priority over Metal's head-dimension limit; rejected calls write nothing.
static void wide_heads() {
  const pool2 pl = make_pool(1, 1, 512);
  const both q = up(make(LARCH_F16, {1, 1, 512})), out = up(make(LARCH_F16, {1, 1, 512}));
  const both table = up(ints({1, 1}, {0})), lens = up(ints({1}, {1})), start = up(ints({1}, {0}));
  const both wrong_dtype = up(make(LARCH_F32, {1})), wrong_shape = up(ints({2}, {1, 1}));
  const auto before = down(1, out);
  for (int i = 0; i < 2; ++i) {
    for (const larch_tensor* bad : {static_cast<const larch_tensor*>(nullptr), &wrong_dtype.t[i], &wrong_shape.t[i]}) {
      CHECK(larch_attn_decode(sides[i].s, &q.t[i], &pl.p[i], &table.t[i], bad, 0, 1, &out.t[i]) ==
                LARCH_ERR_INVALID_ARG, "wide decode must validate seq_lens first");
      CHECK(larch_attn_prefill(sides[i].s, &q.t[i], &pl.p[i], &table.t[i], bad, &lens.t[i], 1, &out.t[i]) ==
                LARCH_ERR_INVALID_ARG, "wide prefill must validate q_start_pos first");
      CHECK(larch_attn_prefill(sides[i].s, &q.t[i], &pl.p[i], &table.t[i], &start.t[i], bad, 1, &out.t[i]) ==
                LARCH_ERR_INVALID_ARG, "wide prefill must validate q_lens first");
    }
    const larch_status expected = i ? LARCH_ERR_UNSUPPORTED : LARCH_OK;
    CHECK(larch_attn_decode(sides[i].s, &q.t[i], &pl.p[i], &table.t[i], &lens.t[i], 0, 1, &out.t[i]) == expected,
          "valid wide decode status");
    CHECK(larch_attn_prefill(sides[i].s, &q.t[i], &pl.p[i], &table.t[i], &start.t[i], &lens.t[i], 1, &out.t[i]) == expected,
          "valid wide prefill status");
  }
  CHECK(down(1, out) == before, "rejected wide attention wrote to output");
}

// gdn's prefill kernels (over GDN_WIDE_ROWS rows a sequence): vv_gdn_chunk at head_dim 128 and 64 (full and ragged
// 8-row chunks, chunks cut short at snapshot rows, an empty sequence), and vv_gdn's GDN_ROWS rows a simdgroup at
// the other head_dims. Tolerances are ops.cpp's gdn rows.
static void gdn_chunks() {
  struct gcase {
    uint32_t k_heads, v_heads, hd;
    std::vector<int32_t> lens, live, snaps;  // snaps: (row, slot) pairs
    bool strong = false;  // alpha 1e8 on every 11th row: a decay of 0 that must not swallow the next rows' decays
  };
  const gcase cases[] = {{2, 4, 128, {45}, {0}, {3, 1, 17, 2, 44, 3}},
                         {1, 2, 64, {70, 0, 30}, {2, 0, 1}, {69, 3, 70, 4}},
                         {2, 4, 96, {40}, {0}, {5, 1}},
                         {1, 2, 256, {40}, {1}, {}},
                         {1, 2, 64, {45}, {0}, {14, 1, 44, 2}, true},
                         {2, 4, 128, {40}, {1}, {24, 3}, true}};
  for (const gcase& c : cases) {
    const int64_t rows = std::accumulate(c.lens.begin(), c.lens.end(), int64_t(0)), vh = c.v_heads, hd = c.hd;
    const int64_t ch = (2 * int64_t(c.k_heads) + vh) * hd, slots = 5, seqs = int64_t(c.lens.size());
    CHECK(rows > GDN_WIDE_ROWS * seqs, "gdn prefill case of %lld rows takes the decode kernel", (long long)rows);
    std::vector<int32_t> snap(size_t(rows), -1);
    for (size_t i = 0; i < c.snaps.size(); i += 2) snap[size_t(c.snaps[i])] = c.snaps[i + 1];
    host a = make(LARCH_F32, {vh});
    for (int64_t h = 0; h < vh; ++h) {
      const float v = -uniform(0.05f, 1.0f);
      std::memcpy(&a.data[size_t(h) * 4], &v, 4);
    }
    host al = make(LARCH_F32, {rows, vh});
    for (int64_t r = 2; c.strong && r < rows; r += 11)
      for (int64_t h = 0; h < vh; ++h) {
        const float big = 1e8f;
        std::memcpy(&al.data[size_t(r * vh + h) * 4], &big, 4);
      }
    const both qkv = up(make(LARCH_F16, {rows, ch})), alpha = up(al),
               beta = up(make(LARCH_F32, {rows, vh})), dt = up(make(LARCH_F32, {vh})), av = up(a),
               lens = up(ints({seqs}, c.lens)), live = up(ints({seqs}, c.live)), snaps = up(ints({rows}, snap)),
               out = up(make(LARCH_F16, {rows, vh * hd})), conv = up(make(LARCH_F16, {slots * 3, ch}), 0),
               ssm = up(make(LARCH_F32, {slots * vh * hd, hd}, 0, 0.1f), 0);
    const std::string name = "gdn prefill, head_dim " + std::to_string(hd) + ", " + std::to_string(rows) + " rows" +
                             (c.strong ? ", strong decays" : "");
    run(name, [&](int i) {
      const larch_gdn_state st{conv.t[i].buf, ssm.t[i].buf, uint32_t(slots), 1, 0, 4, c.k_heads, c.v_heads, c.hd};
      return larch_gdn(sides[i].s, &qkv.t[i], &alpha.t[i], &beta.t[i], &dt.t[i], &av.t[i], 1e-6f, &st, &live.t[i],
                        &lens.t[i], &snaps.t[i], &out.t[i]);
    });
    same(name, out, 2e-3, 2e-4);
    same(name + " state", ssm, 1e-3, 1e-5);
  }
}

// The barrier tracker across ops (backend.mm encode), which every other case syncs between: a dropped read-after-
// write barrier or an output marked input() fails here on every run. 256 adds in place, each read by an rmsnorm
// into its own slice, then kv_write and attn_decode on layer 1 of a pool and then on layer 0, fed the last
// slices; one sync at the end. Over 500 dispatches, so early commits split the chain.
static void feed_chain() {
  constexpr int64_t rows = 4, dim = 64, steps = 256;
  const both x = up(make(LARCH_F32, {rows, dim})), b = up(make(LARCH_F16, {rows, dim}, 0, 0.1f));
  const both w = up(make(LARCH_F32, {dim})), norms = up(make(LARCH_F16, {steps * rows, dim}));
  const auto slice = [&](int i, int64_t step, bool heads) {  // norms' rows of one step; heads: as [rows, 1, dim]
    larch_tensor t = norms.t[i];
    t.offset += uint64_t(step * rows * dim) * 2;
    t.shape[0] = rows;
    if (heads) t.ndim = 3, t.shape[1] = 1, t.shape[2] = dim, t.stride[1] = dim, t.stride[2] = 1;
    return t;
  };
  for (int64_t step = 0; step < steps; ++step) {
    run("feed chain add", [&](int i) { return larch_add(sides[i].s, &x.t[i], &b.t[i], &x.t[i]); });
    run("feed chain rmsnorm", [&](int i) {
      const larch_tensor out = slice(i, step, false);
      return larch_rmsnorm(sides[i].s, &x.t[i], &w.t[i], 1e-6f, &out);
    });
  }
  const pool2 pool = make_pool(rows, 1, dim);  // sequence r: block r, 16 positions, the last one written here
  const both slots = up(ints({rows}, {15, 31, 47, 63})), table = up(ints({rows, 1}, {0, 1, 2, 3}));
  const both lens = up(ints({rows}, {16, 16, 16, 16}));
  const both outs[2] = {up(make(LARCH_F16, {rows, 1, dim})), up(make(LARCH_F16, {rows, 1, dim}))};
  for (const uint32_t layer : {1u, 0u}) {
    const int64_t first = steps - 3 - 3 * layer;  // k, v and q: the last six slices
    run("feed chain kv_write", [&](int i) {
      larch_kv_pool p = pool.p[i];
      p.layer = layer;
      const larch_tensor k = slice(i, first, true), v = slice(i, first + 1, true);
      return larch_kv_write(sides[i].s, &k, &v, &slots.t[i], &p);
    });
    run("feed chain attn_decode", [&](int i) {
      larch_kv_pool p = pool.p[i];
      p.layer = layer;
      const larch_tensor q = slice(i, first + 2, true);
      return larch_attn_decode(sides[i].s, &q, &p, &table.t[i], &lens.t[i], 0, 0.125f, &outs[layer].t[i]);
    });
  }
  same("feed chain: 256 adds in place", x, 0, 0);
  same("feed chain: an rmsnorm after each add", norms, 2e-3, 1e-5);
  same("feed chain: kv_write k, layers 1 and 0", pool.k, 2e-3, 1e-5);
  same("feed chain: kv_write v, layers 1 and 0", pool.v, 2e-3, 1e-5);
  same("feed chain: attn_decode layer 1", outs[1], 2e-3, 5e-4);
  same("feed chain: attn_decode layer 0", outs[0], 2e-3, 5e-4);
}

// sample_prep in parts of SP_CHUNK logits. Row 0, k 64: values that tie across parts, so ids decide between
// parts. Row 1, k 7: a part of -inf only (its sum is 0, not NaN). Row 2, k 1: the top value in the last id.
// Row 3, k 0 (64): the top value once in every part, a last part shorter than k.
static void sample_parts() {
  constexpr int64_t c = SP_CHUNK;
  for (const int64_t vocab : {c - 3, 2 * c + 1, 3 * c, int64_t(128256), int64_t(SP_MERGE_THREADS) * c}) {
    const std::vector<larch_sample_params> per_row = {{1, 64, 1, 0}, {1, 7, 1, 0}, {1, 1, 1, 0}, {1, 0, 1, 0}};
    host h = make(LARCH_F32, {4, vocab});
    std::vector<float> x(size_t(4 * vocab));
    for (float& v : x) v = uniform(-8, -1);
    for (int64_t i = 0; i < vocab; i += 97) x[size_t(i)] = i % 2 ? 2.0f : 3.0f;  // row 0: ties in every part
    for (int64_t i = c; i < std::min(2 * c, vocab); ++i) x[size_t(vocab + i)] = -INFINITY;
    x[size_t(3 * vocab - 1)] = 9;
    for (int64_t i = 0; i < vocab; i += c) x[size_t(3 * vocab + std::min(i + i / c % 5, vocab - 1))] = 5;
    std::memcpy(h.data.data(), x.data(), h.data.size());
    const both lg = up(h), vals = up(make(LARCH_F32, {4, 64})), ids = up(make(LARCH_I32, {4, 64})),
               lse = up(make(LARCH_F32, {4}));
    const std::string name = "sample_prep parts, vocab " + std::to_string(vocab);
    run(name, [&](int i) {
      return larch_sample_prep(sides[i].s, &lg.t[i], per_row.data(), nullptr, &vals.t[i], &ids.t[i], &lse.t[i]);
    });
    same(name + " vals", vals, 0, 0);
    same(name + " ids", ids, 0, 0);
    same(name + " lse", lse, 1e-5, 1e-5);
  }
  // One part past what vv_sp_merge takes: refused, not wrong.
  const int64_t vocab = int64_t(SP_MERGE_THREADS) * c + 1;
  const both lg = up(make(LARCH_F32, {1, vocab})), vals = up(make(LARCH_F32, {1, 64})),
             ids = up(make(LARCH_I32, {1, 64})), lse = up(make(LARCH_F32, {1}));
  const larch_sample_params p{1, 0, 1, 0};
  CHECK(larch_sample_prep(sides[1].s, &lg.t[1], &p, nullptr, &vals.t[1], &ids.t[1], &lse.t[1]) == LARCH_ERR_UNSUPPORTED,
        "sample_prep, vocab %lld: not refused", (long long)vocab);
}

// Data the CPU reference refuses, because it reads the logits and the penalty ids; Metal reads no data on the
// host, so its kernels keep every write and every id inside the vocabulary. A penalty id past the vocabulary is
// skipped like padding (on row 0 it would land in row 1 of the penalised copy). A NaN logit in a vocabulary
// shorter than a part brings out no id of the part's padding.
static void sample_bad_data() {
  constexpr int64_t vocab = 2 * SP_CHUNK - 48;
  host h = make(LARCH_F32, {2, vocab});
  const float top = 10, nan = NAN;
  std::memcpy(&h.data[size_t(vocab + 3) * 4], &top, 4);  // row 1, id 3: where row 0's id vocab + 3 lands
  const both lg = up(h), past = up(ints({2, 2}, {int32_t(vocab + 3), 7, -1, -1})),
             padded = up(ints({2, 2}, {-1, 7, -1, -1})), vals = up(make(LARCH_F32, {2, 64})),
             ids = up(make(LARCH_I32, {2, 64})), lse = up(make(LARCH_F32, {2}));
  const larch_sample_params pen[2] = {{1, 0, 2, 2}, {1, 0, 2, 2}}, plain[2] = {{1, 0, 1, 0}, {1, 0, 1, 0}};
  const std::string name = "sample_prep, penalty id past the vocabulary";
  run(name, [&](int i) {
    return larch_sample_prep(sides[i].s, &lg.t[i], pen, i ? &past.t[1] : &padded.t[0], &vals.t[i], &ids.t[i],
                              &lse.t[i]);
  });
  same(name + " vals", vals, 0, 0);
  same(name + " ids", ids, 0, 0);
  same(name + " lse", lse, 1e-5, 1e-5);

  host hn = make(LARCH_F32, {2, 5});
  for (const size_t i : {0, 1, 2, 3, 4, 9}) std::memcpy(&hn.data[i * 4], &nan, 4);  // all of row 0, the end of row 1
  const both lgn = up(hn);
  CHECK(larch_sample_prep(sides[1].s, &lgn.t[1], plain, nullptr, &vals.t[1], &ids.t[1], &lse.t[1]) == LARCH_OK,
        "sample_prep, NaN logits: %s", larch_last_error(sides[1].dev));
  const std::vector<uint8_t> got = down(1, ids);
  int past_vocab = 0;
  for (size_t i = 0; i < got.size(); i += 4) {
    int32_t id;
    std::memcpy(&id, &got[i], 4);
    past_vocab += id < -1 || id >= 5;
  }
  CHECK(!past_vocab, "sample_prep, NaN logits: %d ids outside the vocabulary", past_vocab);
  if (!past_vocab) std::printf("ok   sample_prep, NaN logits: ids inside the vocabulary\n");
}

// A tensor offset that is not a multiple of its element size. larch.h does not forbid it yet and the CPU
// reference runs it; Metal's typed kernels cannot, so Metal refuses it with UNSUPPORTED and writes nothing.
static void misaligned() {
  const both a = up(make(LARCH_F32, {2, 64})), b = up(make(LARCH_F16, {2, 64})), out = up(make(LARCH_F32, {2, 65}));
  const both w = up(make(LARCH_Q8_0, {9, 512})), x = up(make(LARCH_F16, {1, 512})), y = up(make(LARCH_F16, {1, 8}));
  larch_status st[2];
  for (int i = 0; i < 2; ++i) {
    const larch_tensor o = tensor_on(out.t[i].buf, make(LARCH_F32, {2, 64}), out.off + 2);
    st[i] = larch_add(sides[i].s, &a.t[i], &b.t[i], &o);
  }
  CHECK(st[0] == LARCH_OK && st[1] == LARCH_ERR_UNSUPPORTED, "add, out 2 bytes off: CPU %d, Metal %d", st[0], st[1]);
  CHECK(down(1, out) == out.h.data, "add, out 2 bytes off: Metal wrote");
  const larch_tensor wm = tensor_on(w.t[1].buf, make(LARCH_Q8_0, {8, 512}), w.off + 1);
  const larch_status q = larch_mmvq(sides[1].s, &wm, &x.t[1], nullptr, &y.t[1]);
  CHECK(q == LARCH_ERR_UNSUPPORTED && down(1, y) == y.h.data, "mmvq, weight 1 byte off: Metal %d", q);
  std::printf("ok   misaligned offsets refused by Metal, nothing written\n");
}

// Weights in a wrapped buffer (the GGUF mmap path) are read like any other; an output there is refused.
static void wrapped() {
  const size_t page = size_t(getpagesize()), bytes = 4 * page;
  void* mem = nullptr;
  if (posix_memalign(&mem, page, bytes)) std::exit(1);
  const host w = make(LARCH_Q4_K, {16, 512});  // 16 x 2 blocks of 144 bytes
  std::memset(mem, 0x5A, bytes);
  std::memcpy(mem, w.data.data(), w.data.size());
  larch_buffer* ro = nullptr;
  CHECK(larch_buffer_wrap_host(sides[1].dev, mem, bytes, &ro) == LARCH_OK, "wrap_host: %s", larch_last_error(sides[1].dev));
  const both wc = up(w), x = up(make(LARCH_F16, {2, 512})), out = up(make(LARCH_F16, {2, 16}));
  const both ids = up(ints({2}, {3, 15})), emb = up(make(LARCH_F32, {2, 512}));
  both wm = wc;
  wm.t[1] = tensor_on(ro, w, 0);
  run("mmvq from a wrapped buffer", [&](int i) { return larch_mmvq(sides[i].s, &wm.t[i], &x.t[i], nullptr, &out.t[i]); });
  same("mmvq q4_k from a wrapped buffer", out, 2e-3, 1e-3);
  run("embed from a wrapped buffer", [&](int i) { return larch_embed(sides[i].s, &wm.t[i], &ids.t[i], &emb.t[i]); });
  same("embed q4_k from a wrapped buffer", emb, 1e-6, 1e-6);

  // Every op with its one output (or in-place tensor, or pool) in the wrapped buffer: INVALID_ARG, where
  // the same call with a writable output passes, and the mapping keeps its bytes.
  larch_stream* s = sides[1].s;
  const auto g = [](const both& b) { return &b.t[1]; };
  const auto dst = [&](bool r, larch_dtype t, std::vector<int64_t> shape) {
    return r ? tensor_on(ro, make(t, shape), 0) : up(make(t, shape)).t[1];
  };
  const both f32v = up(make(LARCH_F32, {64})), w8 = up(make(LARCH_F32, {8})), f32m = up(make(LARCH_F32, {2, 64}));
  const both f16m = up(make(LARCH_F16, {2, 64})), f16q = up(make(LARCH_F16, {2, 1, 8})), i32v = up(ints({2}, {0, 1}));
  const both table = up(ints({2, 1}, {0, 0})), ones = up(ints({2}, {1, 1})), i32m = up(make(LARCH_I32, {2, 64}));
  const both lse = up(make(LARCH_F32, {2})), f32m2 = up(make(LARCH_F32, {2, 64}));
  const pool2 pl = make_pool(1, 1, 8);
  const larch_rope_params rp = {8, 0, 4096, 10000, 1, 0, 1, 32, 1, nullptr};
  const larch_sample_params sp[2] = {{1, 0, 1, 0}, {1, 0, 1, 0}};
  const std::pair<const char*, std::function<larch_status(bool)>> calls[] = {
      {"embed", [&](bool r) { const larch_tensor o = dst(r, LARCH_F32, {2, 512}); return larch_embed(s, &wm.t[1], g(i32v), &o); }},
      {"rmsnorm", [&](bool r) { const larch_tensor o = dst(r, LARCH_F16, {2, 64}); return larch_rmsnorm(s, g(f32m), g(f32v), 1e-6f, &o); }},
      {"rope", [&](bool r) { const larch_tensor o = dst(r, LARCH_F16, {2, 2, 8}); return larch_rope(s, &o, g(i32v), &rp); }},
      {"mmvq", [&](bool r) { const larch_tensor o = dst(r, LARCH_F16, {2, 16}); return larch_mmvq(s, &wm.t[1], g(x), nullptr, &o); }},
      {"dequant", [&](bool r) { const larch_tensor o = dst(r, LARCH_F16, {16, 512}); return larch_dequant(s, &wm.t[1], &o); }},
      {"gemm", [&](bool r) { const larch_tensor o = dst(r, LARCH_F32, {2, 2}); return larch_gemm(s, g(f16m), g(f16m), nullptr, &o, 1, 0); }},
      {"silu_mul", [&](bool r) { const larch_tensor o = dst(r, LARCH_F16, {2, 64}); return larch_silu_mul(s, g(f16m), g(f16m), &o); }},
      {"add", [&](bool r) { const larch_tensor o = dst(r, LARCH_F32, {2, 64}); return larch_add(s, g(f32m), g(f16m), &o); }},
      {"kv_write",
       [&](bool r) {
         larch_kv_pool p = pl.p[1];
         if (r) p.k = ro;
         return larch_kv_write(s, g(f16q), g(f16q), g(i32v), &p);
       }},
      {"attn_decode",
       [&](bool r) {
         const larch_tensor o = dst(r, LARCH_F16, {2, 1, 8});
         return larch_attn_decode(s, g(f16q), &pl.p[1], g(table), g(ones), 0, 1, &o);
       }},
      {"attn_prefill",
       [&](bool r) {
         const larch_tensor o = dst(r, LARCH_F16, {2, 1, 8});
         return larch_attn_prefill(s, g(f16q), &pl.p[1], g(table), g(i32v), g(ones), 1, &o);
       }},
      {"qk_norm", [&](bool r) { const larch_tensor o = dst(r, LARCH_F16, {2, 2, 8}); return larch_qk_norm(s, &o, g(w8), 1e-6f); }},
      {"sample_prep vals",
       [&](bool r) {
         const larch_tensor o = dst(r, LARCH_F32, {2, 64});
         return larch_sample_prep(s, g(f32m), sp, nullptr, &o, g(i32m), g(lse));
       }},
      {"sample_prep ids",
       [&](bool r) {
         const larch_tensor o = dst(r, LARCH_I32, {2, 64});
         return larch_sample_prep(s, g(f32m), sp, nullptr, g(f32m2), &o, g(lse));
       }},
      {"sample_prep lse",
       [&](bool r) {
         const larch_tensor o = dst(r, LARCH_F32, {2});
         return larch_sample_prep(s, g(f32m), sp, nullptr, g(f32m2), g(i32m), &o);
       }},
  };
  for (const auto& [op, call] : calls) {
    const larch_status ok = call(false), refused = call(true);
    CHECK(ok == LARCH_OK && refused == LARCH_ERR_INVALID_ARG, "%s: writable output %d, read-only output %d (%s)", op,
          ok, refused, larch_last_error(sides[1].dev));
  }

  // Shared metadata checks precede Metal's read-only refusal, even when both faults are present.
  const larch_tensor readonly_out = dst(true, LARCH_F16, {2, 1, 8});
  CHECK(larch_attn_decode(s, g(f16q), &pl.p[1], g(table), nullptr, 0, 1, &readonly_out) == LARCH_ERR_INVALID_ARG &&
            std::strstr(larch_last_error(sides[1].dev), "seq_lens"),
        "decode must check seq_lens before a read-only output");
  CHECK(larch_attn_prefill(s, g(f16q), &pl.p[1], g(table), g(i32v), nullptr, 1, &readonly_out) == LARCH_ERR_INVALID_ARG &&
            std::strstr(larch_last_error(sides[1].dev), "q_lens"),
        "prefill must check q_lens before a read-only output");

  // The checker finishes pool k before looking up pool v, including with two malformed buffers.
  larch_kv_pool bad_pool = pl.p[1];
  bad_pool.k = pl.p[0].k;
  bad_pool.v = nullptr;
  CHECK(larch_kv_write(s, g(f16q), g(f16q), g(i32v), &bad_pool) == LARCH_ERR_INVALID_ARG &&
            std::strstr(larch_last_error(sides[1].dev), "pool k"),
        "pool k ownership must be checked before pool v");
  bad_pool = pl.p[1];
  bad_pool.n_blocks = 1000;
  bad_pool.v = pl.p[0].v;
  CHECK(larch_kv_write(s, g(f16q), g(f16q), g(i32v), &bad_pool) == LARCH_ERR_INVALID_ARG &&
            std::strstr(larch_last_error(sides[1].dev), "pool buffer too small"),
        "pool k bounds must be checked before pool v ownership");
  CHECK(larch_stream_sync(s) == LARCH_OK, "sync after the read-only calls");
  CHECK(!std::memcmp(mem, w.data.data(), w.data.size()), "the wrapped mapping changed");
  std::printf("ok   read-only outputs refused by all %zu calls (13 ops)\n", std::size(calls));

  // A CPU reference buffer reaching a Metal op is refused too.
  CHECK(larch_add(s, &f32m.t[0], g(f16m), g(f32m)) == LARCH_ERR_INVALID_ARG, "add with a CPU input on Metal");
  larch_buffer_free(ro);
  std::free(mem);
}

static void close_sides() {
  for (side& sd : sides) {
    if (sd.s) (void)larch_stream_sync(sd.s);
    for (larch_buffer* b : sd.owned) larch_buffer_free(b);
    sd.owned.clear();
    if (sd.s) larch_stream_destroy(sd.s);
    if (sd.dev) larch_device_close(sd.dev);
    sd.s = nullptr;
    sd.dev = nullptr;
  }
}

static larch_status open_sides() {
  int32_t gpus = 0;
  if (larch_device_count(&gpus) != LARCH_OK || gpus < 1) {
    std::fprintf(stderr, "no Metal device\n");
    return LARCH_ERR_DRIVER;
  }
  for (int i = 0; i < 2; ++i) {
    const larch_status open = larch_device_open(i - 1, &sides[i].dev);
    if (open != LARCH_OK || larch_stream_create(sides[i].dev, &sides[i].s) != LARCH_OK) {
      std::fprintf(stderr, "device %d: %s\n", i - 1, larch_last_error(sides[i].dev));
      close_sides();
      return open != LARCH_OK ? open : LARCH_ERR_DRIVER;
    }
  }
  return LARCH_OK;
}

static bool skipped = false;

static void run_compare() {
  const larch_status open = open_sides();
  // Forced tensor GEMM on a GPU without it (before Apple family 10) is a skip, not a pass. Any other
  // UNSUPPORTED open also fails the unforced and LARCH_METAL_TENSOR=0 registrations.
  const char* const forced = std::getenv("LARCH_METAL_TENSOR");
  if (open == LARCH_ERR_UNSUPPORTED && forced && !std::strcmp(forced, "1")) {
    std::printf("skip forced tensor GEMM: this GPU has no Metal tensor API\n");
    skipped = true;
    return;
  }
  if (open != LARCH_OK) {
    ++failures;
    return;
  }
  gemm_bias_edge();
  gemm_tensor_contract();
  gemm_tensor_fallbacks();
  gemm_rows_first();
  gemm_deep();
  gemm_tile_paths();
  deferred_dequant();
  mmvq_rows();
  mmvq_tiles();
  mmvq_shared_x();
  mmvq_zero_rows();
  attn_decode_splits();
  attn_prefill_flash();
  wide_heads();
  gdn_chunks();
  feed_chain();
  sample_parts();
  sample_bad_data();
  misaligned();
  wrapped();
  close_sides();
}

int main() {
  nonfinite();
  run_compare();
  // Each switch rejects a value other than 0 or 1 at open.
  for (const char* name : {"LARCH_METAL_TENSOR", "LARCH_METAL_CONCURRENT", "LARCH_METAL_COMMIT",
                           "LARCH_METAL_PROFILE", "LARCH_METAL_SIMD_KERNELS", "LARCH_METAL_MOE_PREFETCH",
                           "LARCH_METAL_GDN_WIDE", "LARCH_METAL_MOE_NARROW",
                           "LARCH_METAL_GDN_CHUNK"}) {
    const char* const saved = std::getenv(name);
    const std::string saved_value = saved ? saved : "";
    const bool had_saved = saved;
    CHECK(setenv(name, "invalid", 1) == 0, "set invalid %s", name);
    larch_device* invalid = nullptr;
    const larch_status invalid_status = larch_device_open(0, &invalid);
    CHECK(invalid_status == LARCH_ERR_INVALID_ARG, "invalid %s: status %d (%s)", name, invalid_status,
          larch_last_error(invalid));
    if (invalid) larch_device_close(invalid);
    CHECK(had_saved ? setenv(name, saved_value.c_str(), 1) == 0 : unsetenv(name) == 0, "restore %s", name);
  }
  std::printf("metal_compare: %d failures%s\n", failures, skipped ? ", skipped" : "");
  return failures ? 1 : skipped ? 77 : 0;  // 77: SKIP_RETURN_CODE in src/metal/CMakeLists.txt
}
