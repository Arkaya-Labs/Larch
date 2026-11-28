// The op comparison of architecture 15: every larch.h op on the device under test against the CPU
// reference, over a shape grid, with a tolerance per op and dtype (TOLERANCES). LARCH_TEST_DEVICE is
// the device's ordinal, -1 for the CPU reference itself; unset, it is GPU 0 when the build has one and
// the CPU reference otherwise, which is what GPU-less CI runs. Strict: any status but the expected one
// fails, UNSUPPORTED included. Every argument gets its own buffer, at its own offset between guard
// bytes; after each call every input and guard byte must be as it was, and on a rejected call
// (INVALID_ARG on both devices) every byte. The grid holds bucket padding (slot -1, seq_len 0,
// penalty id -1), paged KV through shuffled block tables at a nonzero layer, absolute positions to
// 131071, quant rows of one block, odd block counts and edge blocks, and one wrong shape, dtype or host
// parameter per invalid case.
#define _CRT_SECURE_NO_WARNINGS  // std::getenv on MSVC
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "testdata.h"
#include "larch.h"

// |dut - ref| <= rtol * |ref| + (atol + per_pos * the case's largest position) * max|ref|, per element
// (max over the output); all 0 is bitwise.
struct tol {
  double rtol, atol, per_pos;
};

// The tolerance table. dtype is the output's, except embed's, which is the table's (-1 = any other).
// Every contract op accumulates in f32 (larch.h), so these bound f32 reordering and one or two f16
// roundings, never a lower-precision algorithm; the one exception, mmvq's and mmvq_verify's f16 run sums on
// CUDA sm_86/89 (one row and the routed rows, larch.h), meets the same mmvq rows below.
static const struct {
  const char* op;
  int dtype;
  tol t;
} TOLERANCES[] = {
    {"embed", LARCH_F32, {0, 0, 0}},             // a copy
    {"embed", LARCH_F16, {0, 0, 0}},             // f16 to f32 is exact
    {"embed", -1, {1e-6, 1e-6, 0}},               // ggml's f32 decode, bitwise on CUDA; room for one FMA
    {"dequant", LARCH_F16, {1e-3, 1e-6, 0}},     // that decode, rounded once to f16
    {"rmsnorm", LARCH_F16, {2e-3, 1e-5, 0}},     // f32 statistics in any order, then f16
    {"qk_norm", LARCH_F16, {2e-3, 1e-5, 0}},     //
    {"rope", LARCH_F16, {2e-3, 1e-4, 0x1p-21}},  // + 8 f32 roundoffs of the angle, which grows with pos
    {"rope_multi", LARCH_F16, {2e-3, 1e-4, 0x1p-21}},  // rope's, per position
    // x is used as f16 (rounding w or the products to f16 fits); quantizing x to 8 bits, as llama.cpp's
    // q8_1 path does, is ~1e-2 off and a contract change, not a tolerance change.
    {"mmvq", LARCH_F32, {1e-3, 1e-3, 0}},
    {"mmvq", LARCH_F16, {2e-3, 1e-3, 0}},
    {"mmvq_id", LARCH_F32, {1e-3, 1e-3, 0}},  // mmvq's sums, then (weighted) an f32 sum over the slots
    {"mmvq_id", LARCH_F16, {2e-3, 1e-3, 0}},  //
    {"gemm", LARCH_F32, {1e-4, 1e-4, 0}},             // f16 products are exact in f32; only the sums move
    {"gemm", LARCH_F16, {2e-3, 1e-4, 0}},             //
    {"silu_mul", LARCH_F16, {2e-3, 1e-5, 0}},         //
    {"add", LARCH_F32, {0, 0, 0}},                    // one f32 add, correctly rounded everywhere
    {"kv_write", LARCH_F16, {0, 0, 0}},               // a copy
    {"attn_decode", LARCH_F16, {2e-3, 5e-4, 0}},      // f32 softmax and sums (p may be stored f16)
    {"attn_prefill", LARCH_F16, {2e-3, 5e-4, 0}},     //
    {"sample_prep", LARCH_F32, {0, 0, 0}},            // vals: one f32 multiply or divide per logit
    {"sample_prep", LARCH_I32, {0, 0, 0}},            // ids: ties to the lower id
    {"sample_prep lse", LARCH_F32, {1e-5, 1e-5, 0}},  // an f32 sum over the vocabulary
    {"moe_route", LARCH_F32, {1e-5, 1e-6, 0}},        // weights: f32 softmax where the reference uses double
    {"moe_route", LARCH_I32, {0, 0, 0}},              // ids: ties to the lower id
    {"sigmoid_mul", LARCH_F16, {2e-3, 1e-5, 0}},      // one f32 sigmoid, then f16
    {"layernorm", LARCH_F16, {2e-3, 1e-5, 0}},        // f32 statistics in any order, then f16 (as rmsnorm)
    {"gelu", LARCH_F16, {2e-3, 1e-5, 0}},             // one f32 tanh, then f16
    {"attn_full", LARCH_F16, {2e-3, 5e-4, 0}},        // f32 softmax and sums, as attn_decode
    {"gdn_conv", LARCH_F16, {2e-3, 1e-4, 0}},         // an f32 sum where the reference sums in double, silu, f16
    {"gdn_conv state", LARCH_F16, {0, 0, 0}},         // the last inputs, copied
    {"gdn", LARCH_F16, {2e-3, 2e-4, 0}},              // f32 sums where the reference sums in double, then f16
    {"gdn state", LARCH_F32, {1e-3, 1e-5, 0}},        // the same matrix after every row, kept in f32
    {"draft_conv", LARCH_F16, {2e-3, 1e-5, 0}},       // two f32 products and adds (an FMA may fuse them), then f16
    {"draft_conv", LARCH_F32, {1e-6, 1e-6, 0}},       // the same, added into the residual
    // The training ops: f32 throughout, against double sums in the reference.
    {"gemm_ex", LARCH_F32, {1e-4, 1e-5, 0}},              // f32 sums in cuBLAS's order
    {"rmsnorm_f32", LARCH_F32, {1e-5, 1e-6, 0}},          //
    {"rmsnorm_backward", LARCH_F32, {1e-4, 1e-5, 0}},     // two f32 row sums, then a difference
    {"rope_f32", LARCH_F32, {1e-5, 1e-5, 0x1p-21}},       // rope's angle, unrounded
    {"silu_mul_f32", LARCH_F32, {1e-5, 1e-6, 0}},         // expf to an ulp or two
    {"silu_mul_backward", LARCH_F32, {1e-5, 1e-6, 0}},    //
    {"softmax", LARCH_F32, {1e-5, 1e-6, 0}},              // an f32 row sum
    {"softmax_backward", LARCH_F32, {1e-4, 1e-5, 0}},     // an f32 row sum, then a difference
};

static tol T(const std::string& op, int dtype) {
  for (const auto& e : TOLERANCES)
    if (op == e.op && (e.dtype == dtype || e.dtype == -1)) return e.t;
  std::fprintf(stderr, "no tolerance for %s, dtype %d\n", op.c_str(), dtype);
  std::exit(2);
}

// One argument: a tensor (none for a KV pool buffer, which the call reaches through its pool).
struct arg {
  larch_dtype t;
  std::vector<int64_t> shape;
  std::shared_ptr<const bytes> data;  // the contents before the call; cases reading the same weight share it
  bool out = false;
  tol tl{};
  bool pool = false;
  uint64_t skew = 0;  // bytes added to its offset: 2 puts a 2-byte dtype off 16-byte alignment
  int64_t pad = 0;    // elements (blocks for quant) after each row, which data holds: row stride = dense + pad
};

struct test {
  std::string name;
  std::vector<arg> args;
  std::function<larch_status(larch_stream*, const larch_tensor*)> run;  // tensor i views args[i]
  larch_status want = LARCH_OK;
  double pos_max = 0;
};

static arg in(larch_dtype t, std::vector<int64_t> shape, bytes data) {
  return {t, std::move(shape), std::make_shared<const bytes>(std::move(data))};
}
// 0xFF bytes: NaN in every float dtype, -1 in i32, so an element an op never writes shows.
static arg poison(larch_dtype t, std::vector<int64_t> shape) {
  const size_t n = size_t(nbytes(t, shape));
  return in(t, std::move(shape), bytes(n, 0xFF));
}
// An output, compared under op's tolerance for its dtype; it starts as poison, or as `init`.
static arg out(const std::string& op, larch_dtype t, std::vector<int64_t> shape, bytes init = {}) {
  arg a = init.empty() ? poison(t, std::move(shape)) : in(t, std::move(shape), std::move(init));
  a.out = true;
  a.tl = T(op, t);
  return a;
}
// A buffer the call reaches through a struct (a KV pool, a GDN state): compared as a whole, under op's tolerance
// for t when written, byte for byte otherwise.
static arg pool(bytes data, bool written, larch_dtype t = LARCH_F16, const std::string& op = "kv_write") {
  return {t, {}, std::make_shared<const bytes>(std::move(data)), written, T(op, t), true};
}
// Gives a p elements after each row (a 1-D tensor's rows are its elements), its data re-laid with NaN in the gaps
// of a float dtype and 0 in i32's: a kernel that strides by the dense width reads them, and its result shows it.
static void pad_rows(arg& a, int64_t p) {
  const std::vector<int64_t> tail(a.shape.begin() + 1, a.shape.end());
  const size_t row = size_t(nbytes(a.t, tail)), gap = size_t(nbytes(a.t, {p}));
  bytes b;
  for (size_t at = 0; at < a.data->size(); at += row) {
    b.insert(b.end(), a.data->begin() + ptrdiff_t(at), a.data->begin() + ptrdiff_t(at + row));
    b.insert(b.end(), gap, a.t == LARCH_I32 ? 0 : 0xFF);
  }
  a.data = std::make_shared<const bytes>(std::move(b));
  a.pad = p;
}

// Data, beside testdata.h's halves and quant blocks.
static bytes floats(size_t n, rng& g, float lo, float hi) {
  std::uniform_real_distribution<float> u(lo, hi);
  bytes b(4 * n);
  for (size_t i = 0; i < n; ++i) {
    const float f = u(g);
    std::memcpy(&b[4 * i], &f, 4);
  }
  return b;
}
template <class V>
static bytes raw(const std::vector<V>& v) {
  bytes b(v.size() * sizeof(V));
  if (!v.empty()) std::memcpy(b.data(), v.data(), b.size());
  return b;
}
static rng seeded(const std::string& name) {
  uint32_t h = 2166136261u;
  for (const char c : name) h = (h ^ uint8_t(c)) * 16777619u;
  return rng(h);
}

// Each device: the CPU reference, and the device under test.
struct side {
  larch_device* dev;
  larch_stream* s;
};

static uint64_t lead(size_t i, const arg& a) { return a.pool ? 0 : 32 * (i + 1) + a.skew; }
static const uint64_t TAIL = 64;
static const uint8_t GUARD = 0xA5;

// The bytes of argument i's buffer before the call: guard, data, guard.
static bytes image(size_t i, const arg& a) {
  bytes b(size_t(lead(i, a)), GUARD);
  b.insert(b.end(), a.data->begin(), a.data->end());
  b.insert(b.end(), TAIL, GUARD);
  return b;
}

// Runs c on x and fills `after` with every buffer's bytes. The op's status, or -1 when setup failed.
static int run_on(const side& x, const test& c, std::vector<bytes>& after) {
  const size_t n = c.args.size();
  std::vector<larch_buffer*> bufs(n, nullptr);
  std::vector<larch_tensor> t(n);
  int status = -1;
  bool ok = true;
  for (size_t i = 0; i < n && ok; ++i) {
    const arg& a = c.args[i];
    const bytes img = image(i, a);
    ok = larch_buffer_alloc(x.dev, img.size(), &bufs[i]) == LARCH_OK &&
         larch_upload(x.s, bufs[i], 0, img.data(), img.size()) == LARCH_OK;
    t[i] = {bufs[i], lead(i, a), a.t, uint32_t(a.shape.size()), {}, {}};
    for (int64_t d = int64_t(a.shape.size()) - 1, stride = 1; d >= 0; --d) {
      t[i].shape[d] = a.shape[size_t(d)];
      t[i].stride[d] = stride;
      stride *= d + 1 == int64_t(a.shape.size()) ? a.shape[size_t(d)] / block_of(a.t) : a.shape[size_t(d)];
    }
    t[i].stride[0] += a.pad;
  }
  if (ok && larch_stream_sync(x.s) == LARCH_OK) {
    status = c.run(x.s, t.data());
    after.assign(n, {});
    // Downloads are asynchronous: every `after` is sized before the first is queued.
    for (size_t i = 0; i < n; ++i) after[i].resize(c.args[i].data->size() + size_t(lead(i, c.args[i]) + TAIL));
    ok = larch_stream_sync(x.s) == LARCH_OK;
    for (size_t i = 0; i < n && ok; ++i)
      ok = larch_download(x.s, bufs[i], 0, after[i].data(), after[i].size()) == LARCH_OK;
    ok = larch_stream_sync(x.s) == LARCH_OK && ok;
  }
  for (larch_buffer* b : bufs) larch_buffer_free(b);
  if (!ok) {
    std::fprintf(stderr, "%s: setup or copy failed: %s\n", c.name.c_str(), larch_last_error(x.dev));
    return -1;
  }
  return status;
}

static float from_h(uint16_t h) {
  const int e = h >> 10 & 31, m = h & 1023;
  const float v = e == 31 ? (m ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
                  : e     ? std::ldexp(float(m + 1024), e - 25)
                          : std::ldexp(float(m), -24);
  return h & 0x8000 ? -v : v;
}
static double elem(const uint8_t* p, larch_dtype t) {
  if (t == LARCH_F16) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return from_h(h);
  }
  if (t == LARCH_I32) {
    int32_t v;
    std::memcpy(&v, p, 4);
    return v;
  }
  float f;
  std::memcpy(&f, p, 4);
  return f;
}

// Output i of c: the dut's data against the reference's under its tolerance. Returns 1 if wrong.
static int compare(const test& c, size_t i, const bytes& ref, const bytes& dut) {
  const arg& a = c.args[i];
  const size_t at = size_t(lead(i, a)), size = size_t(block_bytes(a.t)), n = a.data->size() / size;
  const bool exact = a.tl.rtol == 0 && a.tl.atol == 0 && a.tl.per_pos == 0;
  double m = 0;
  for (size_t j = 0; j < n; ++j)
    if (const double r = elem(&ref[at + j * size], a.t); std::isfinite(r)) m = std::max(m, std::fabs(r));
  const double slack = (a.tl.atol + a.tl.per_pos * c.pos_max) * m;
  size_t bad = 0, first = 0;
  double worst = 0;
  for (size_t j = 0; j < n; ++j) {
    const uint8_t *rp = &ref[at + j * size], *dp = &dut[at + j * size];
    const double r = elem(rp, a.t), d = elem(dp, a.t);
    bool good;
    if (exact)
      good = std::memcmp(rp, dp, size) == 0;
    else if (std::isnan(r) || std::isnan(d))
      good = std::isnan(r) && std::isnan(d);
    else if (std::isinf(r) || std::isinf(d))
      good = r == d;
    else {
      const double allowed = a.tl.rtol * std::fabs(r) + slack;
      worst = std::max(worst, allowed > 0 ? std::fabs(d - r) / allowed : INFINITY);
      good = std::fabs(d - r) <= allowed;
    }
    if (!good && bad++ == 0) first = j;
  }
  if (!bad) return 0;
  std::fprintf(stderr, "%s: argument %zu (%s): %zu of %zu elements off, first [%zu] ref %.9g dut %.9g", c.name.c_str(),
               i, DTYPE[a.t], bad, n, first, elem(&ref[at + first * size], a.t), elem(&dut[at + first * size], a.t));
  if (!exact) std::fprintf(stderr, ", worst %.3g x the tolerance", worst);
  std::fprintf(stderr, "\n");
  return 1;
}

// c on both devices: statuses, then outputs; inputs, guards and rejected calls unchanged. 1 if wrong.
static int check(const side& ref, const side& dut, const test& c) {
  std::vector<bytes> ra, da;
  const int rs = run_on(ref, c, ra);
  if (rs != c.want) {
    std::fprintf(stderr, "%s: the CPU reference returned %d, want %d: %s\n", c.name.c_str(), rs, c.want,
                 larch_last_error(ref.dev));
    return 1;
  }
  const int ds = run_on(dut, c, da);
  if (ds != c.want) {
    std::fprintf(stderr, "%s: returned %d%s, want %d: %s\n", c.name.c_str(), ds,
                 ds == LARCH_ERR_UNSUPPORTED ? " (UNSUPPORTED, a failure)" : "", c.want, larch_last_error(dut.dev));
    return 1;
  }
  int wrong = 0;
  for (size_t i = 0; i < c.args.size(); ++i) {
    const arg& a = c.args[i];
    const bytes before = image(i, a);
    const size_t at = size_t(lead(i, a)), end = at + a.data->size();
    if (!a.out || c.want != LARCH_OK) {
      for (const bytes* got : {&ra[i], &da[i]})
        if (*got != before) {
          std::fprintf(stderr, "%s: argument %zu changed on the %s\n", c.name.c_str(), i,
                       got == &ra[i] ? "CPU reference" : "device under test");
          wrong = 1;
        }
      continue;
    }
    for (const bytes* got : {&ra[i], &da[i]})
      if (!std::equal(before.begin(), before.begin() + at, got->begin()) ||
          !std::equal(before.begin() + end, before.end(), got->begin() + end)) {
        std::fprintf(stderr, "%s: argument %zu written outside its tensor on the %s\n", c.name.c_str(), i,
                     got == &ra[i] ? "CPU reference" : "device under test");
        wrong = 1;
      }
    wrong |= compare(c, i, ra[i], da[i]);
  }
  return wrong;
}

// --- The grid, one function per op. ---

static std::vector<test> tests;
static void add_test(test c) { tests.push_back(std::move(c)); }
// One wrong shape, dtype or host parameter. Never a device value out of range: that is undefined
// behaviour (larch.h), which only the CPU reference reports, as a debugging aid.
static void invalid(std::string name, std::vector<arg> args,
                    std::function<larch_status(larch_stream*, const larch_tensor*)> run) {
  for (arg& a : args) a.out = false;
  tests.push_back({std::move(name), std::move(args), std::move(run), LARCH_ERR_INVALID_ARG});
}

static larch_status embed_run(larch_stream* s, const larch_tensor* t) {
  return larch_embed(s, &t[0], &t[1], &t[2]);
}

// One embed case; out_skew moves out off 16-byte alignment. padded: an element (a block for quant) after each
// table row and 3 poison elements after each out row, which must stay poison.
static void embed_case(larch_dtype tt, int64_t vocab, int64_t rows, int64_t w, uint64_t out_skew = 0,
                       bool padded = false) {
  std::string name = std::string("embed ") + DTYPE[tt] + " [" + std::to_string(vocab) + ", " + std::to_string(w) +
                     "], " + std::to_string(rows) + " ids";
  if (out_skew) name += ", out at +" + std::to_string(out_skew);
  if (padded) name += ", padded rows";
  rng g = seeded(name);
  const int64_t p = padded, wt = w + p * block_of(tt);
  const bytes table = tt == LARCH_F32   ? floats(size_t(vocab * wt), g, -4, 4)
                      : tt == LARCH_F16 ? halves(size_t(vocab * wt), g)
                                         : quant(tt, vocab, wt, g);
  std::vector<int32_t> ids(static_cast<size_t>(rows));
  for (int32_t& id : ids) id = int32_t(g() % uint32_t(vocab));
  if (rows > 2) ids[0] = 0, ids[1] = int32_t(vocab - 1), ids[2] = ids[1];
  std::vector<arg> args = {
      in(tt, {vocab, w}, table), in(LARCH_I32, {rows}, raw(ids)),
      out("embed", LARCH_F32, {rows, w}, bytes(size_t(nbytes(LARCH_F32, {rows, w + 3 * p})), 0xFF))};
  args[0].pad = p;
  args[2].pad = 3 * p;
  args[2].skew = out_skew;
  args[2].tl = T("embed", tt);
  add_test({name, args, embed_run});
}

// QUANTS and Q2_K: the weight formats embed, dequant and mmvq_id take on every backend that has Q2_K (the other quant
// ops may refuse Q2_K on a GPU backend, so they test QUANTS alone).
static const std::vector<larch_dtype> ID_QUANTS = [] {
  std::vector<larch_dtype> q(std::begin(QUANTS), std::end(QUANTS));
  q.push_back(LARCH_Q2_K);
  return q;
}();

static void embed() {
  std::vector<larch_dtype> types = {LARCH_F32, LARCH_F16};
  types.insert(types.end(), ID_QUANTS.begin(), ID_QUANTS.end());
  for (const larch_dtype tt : types) {
    for (const auto& [vocab, rows, i] : {std::tuple<int64_t, int64_t, int>{37, 19, 0}, {5, 1, 1}, {9, 0, 2}}) {
      const int64_t widths[3][3] = {{896, 7, 1}, {896, 32, 64}, {768, 256, 512}};  // plain, 32-blocks, 256-blocks
      embed_case(tt, vocab, rows, widths[block_of(tt) == 1 ? 0 : block_of(tt) == 32 ? 1 : 2][i]);
    }
    // The element-at-a-time stores: out at +4, then padded rows. Plain tables are 900 wide, whole 8-element
    // pieces and a short last one, also with out aligned.
    const int64_t w = block_of(tt) == 1 ? 900 : 768;
    if (block_of(tt) == 1) embed_case(tt, 37, 19, w);
    embed_case(tt, 37, 19, w, 4);
    embed_case(tt, 37, 19, w, 0, true);
  }
  rng g = seeded("embed invalid");
  const bytes q8 = quant(LARCH_Q8_0, 4, 64, g), ids = raw(std::vector<int32_t>{1, 3});
  invalid("embed, ids f32",
          {in(LARCH_Q8_0, {4, 64}, q8), in(LARCH_F32, {2}, floats(2, g, 0, 1)), poison(LARCH_F32, {2, 64})},
          embed_run);
  invalid("embed, out f16", {in(LARCH_Q8_0, {4, 64}, q8), in(LARCH_I32, {2}, ids), poison(LARCH_F16, {2, 64})},
          embed_run);
  invalid("embed, bf16 table",
          {in(LARCH_BF16, {4, 64}, halves(256, g)), in(LARCH_I32, {2}, ids), poison(LARCH_F32, {2, 64})}, embed_run);
  invalid("embed, out narrower than the table",
          {in(LARCH_Q8_0, {4, 64}, q8), in(LARCH_I32, {2}, ids), poison(LARCH_F32, {2, 32})}, embed_run);
}

static void rmsnorm() {
  const auto run = [](float eps) {
    return [eps](larch_stream* s, const larch_tensor* t) { return larch_rmsnorm(s, &t[0], &t[1], eps, &t[2]); };
  };
  for (const auto& [rows, d, eps] : {std::tuple<int64_t, int64_t, float>{1, 4096, 1e-5f},
                                     {7, 896, 1e-6f},
                                     {33, 2048, 1e-6f},
                                     {2, 4100, 1e-5f},  // d % 4 == 0 past one 1024 x 4 pass: the vector path's tail
                                     {3, 4098, 1e-6f},  // d % 4 != 0, rows not 16-byte aligned: the scalar path
                                     {3, 1, 1e-6f},
                                     {0, 64, 1e-6f}}) {
    const std::string name = "rmsnorm [" + std::to_string(rows) + ", " + std::to_string(d) + "]";
    rng g = seeded(name);
    bytes x = floats(size_t(rows * d), g, -4, 4);
    for (int64_t r = 0; r < rows; ++r)  // rows from 2^-6 to 2^6 in scale
      for (int64_t c = 0; c < d; ++c) {
        float f;
        std::memcpy(&f, &x[size_t(4 * (r * d + c))], 4);
        // 64x past one 1024 x 4 pass: the tail dominates the sum of squares, so a tail dropped from it shows
        f = std::ldexp(f, int(r % 13) - 6 + (c >= 4096 ? 6 : 0));
        std::memcpy(&x[size_t(4 * (r * d + c))], &f, 4);
      }
    add_test({name,
              {in(LARCH_F32, {rows, d}, x), in(LARCH_F32, {d}, floats(size_t(d), g, -2, 2)),
               out("rmsnorm", LARCH_F16, {rows, d})},
              run(eps)});
  }
  rng g = seeded("rmsnorm invalid");
  invalid("rmsnorm, w shorter than d",
          {in(LARCH_F32, {2, 64}, floats(128, g, -1, 1)), in(LARCH_F32, {32}, floats(32, g, -1, 1)),
           poison(LARCH_F16, {2, 64})},
          run(1e-6f));
  invalid("rmsnorm, d 0",
          {in(LARCH_F32, {3, 0}, {}), in(LARCH_F32, {0}, {}), poison(LARCH_F16, {3, 0})}, run(1e-6f));
  invalid("rmsnorm, out f32",
          {in(LARCH_F32, {2, 64}, floats(128, g, -1, 1)), in(LARCH_F32, {64}, floats(64, g, -1, 1)),
           poison(LARCH_F32, {2, 64})},
          run(1e-6f));
}

// ggml_rope_multi's three modes at the shapes their models use, positions differing per plane.
static void rope_multi() {
  struct grid {
    const char* what;
    int64_t rows, heads, hd;
    uint32_t nd, mode, s[4];
    float base, freq_scale, ext;
    bool ff;
  };
  const grid cases[] = {
      {"qwen3.5 text, imrope", 6, 8, 256, 64, LARCH_ROPE_IMROPE, {11, 11, 10, 0}, 1e7f, 1, 0, false},
      {"imrope, sections wrap", 3, 2, 128, 64, LARCH_ROPE_IMROPE, {4, 4, 4, 0}, 1e6f, 1, 0, false},
      {"imrope, yarn scale 4", 4, 2, 128, 64, LARCH_ROPE_IMROPE, {11, 11, 10, 0}, 1e6f, 0.25f, 1, false},
      {"qwen2-vl, mrope, rope_freqs", 3, 4, 128, 128, LARCH_ROPE_MROPE, {16, 24, 24, 0}, 1e6f, 1, 0, true},
      {"mrope, an extra section", 2, 2, 64, 64, LARCH_ROPE_MROPE, {8, 8, 8, 8}, 1e4f, 1, 0, false},
      {"qwen-vl encoder, vision", 7, 12, 64, 32, LARCH_ROPE_VISION, {16, 16, 16, 16}, 1e4f, 1, 0, false},
      {"vision, head 72", 5, 3, 72, 36, LARCH_ROPE_VISION, {18, 18, 18, 18}, 1e4f, 1, 0, false},
      {"vision, prefill, one block per row", 300, 16, 64, 32, LARCH_ROPE_VISION, {16, 16, 16, 16}, 1e4f, 1, 0, false},
      {"0 rows", 0, 8, 256, 64, LARCH_ROPE_IMROPE, {11, 11, 10, 0}, 1e7f, 1, 0, false},
  };
  for (const grid& c : cases) {
    const std::string name = std::string("rope_multi ") + c.what + " [" + std::to_string(c.rows) + ", " +
                             std::to_string(c.heads) + ", " + std::to_string(c.hd) + "]";
    rng g = seeded(name);
    std::vector<int32_t> pos(size_t(4 * c.rows));
    for (int32_t& v : pos) v = int32_t(g() % (c.mode == LARCH_ROPE_VISION ? 128 : 8192));
    const int64_t pairs = c.mode == LARCH_ROPE_VISION ? c.nd : c.nd / 2;
    std::vector<float> ff(static_cast<size_t>(pairs));
    for (size_t i = 0; i < ff.size(); ++i) ff[i] = i < ff.size() / 4 ? 1.0f : i < ff.size() / 2 ? 2.5f : 8.0f;
    const larch_rope_multi_params p = {{c.nd, 0, 4096, c.base, c.freq_scale, c.ext, 1.0f, 32.0f, 1.0f, nullptr},
                                        {c.s[0], c.s[1], c.s[2], c.s[3]},
                                        c.mode};
    std::vector<arg> args = {
        out("rope_multi", LARCH_F16, {c.rows, c.heads, c.hd}, halves(size_t(c.rows * c.heads * c.hd), g)),
        in(LARCH_I32, {c.rows, 4}, raw(pos))};
    if (c.ff) args.push_back(in(LARCH_F32, {pairs}, raw(ff)));
    test t = {name, args, [p, ff = c.ff](larch_stream* s, const larch_tensor* x) {
                larch_rope_multi_params q = p;
                q.rope.freq_factors = ff ? &x[2] : nullptr;
                return larch_rope_multi(s, &x[0], &x[1], &q);
              }};
    for (const int32_t v : pos) t.pos_max = std::max(t.pos_max, double(v));
    add_test(std::move(t));
  }
  rng g = seeded("rope_multi invalid");
  const auto run = [](uint32_t nd, uint32_t mode, uint32_t s0) {
    return [nd, mode, s0](larch_stream* s, const larch_tensor* x) {
      const larch_rope_multi_params p = {{nd, 0, 4096, 1e4f, 1, 0, 1, 32, 1, nullptr}, {s0, s0, s0, 0}, mode};
      return larch_rope_multi(s, &x[0], &x[1], &p);
    };
  };
  const auto four = raw(std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7});
  invalid("rope_multi, one position plane", {in(LARCH_F16, {2, 2, 64}, halves(256, g)),
                                              in(LARCH_I32, {2}, raw(std::vector<int32_t>{0, 1}))},
          run(32, LARCH_ROPE_IMROPE, 4));
  invalid("rope_multi, vision n_dims not head_dim / 2",
          {in(LARCH_F16, {2, 2, 64}, halves(256, g)), in(LARCH_I32, {2, 4}, four)}, run(64, LARCH_ROPE_VISION, 8));
  invalid("rope_multi, no sections", {in(LARCH_F16, {2, 2, 64}, halves(256, g)), in(LARCH_I32, {2, 4}, four)},
          run(32, LARCH_ROPE_IMROPE, 0));
  invalid("rope_multi, no such mode", {in(LARCH_F16, {2, 2, 64}, halves(256, g)), in(LARCH_I32, {2, 4}, four)},
          run(32, 3, 4));
}

static void rope() {
  struct grid {
    const char* what;
    int64_t rows, heads, hd;
    uint32_t nd, neox;
    float base, freq_scale, ext;
    bool ff;
    std::vector<int32_t> pos;  // empty: random in [0, 8192)
  };
  const grid cases[] = {
      {"llama3, normal, rope_freqs", 5, 8, 128, 128, 0, 500000, 1, 0, true, {0, 1, 4095, 777, 2048}},
      {"qwen2, neox", 3, 14, 64, 64, 1, 1e6f, 1, 0, false, {0, 100, 4000}},
      {"partial rotary", 2, 2, 64, 32, 1, 1e4f, 1, 0, false, {3, 1000}},
      {"yarn, scale 4", 4, 4, 64, 64, 0, 1e4f, 0.25f, 1, false, {0, 5000, 12000, 16383}},
      {"k, one head", 7, 1, 128, 128, 1, 1e6f, 1, 0, false, {}},
      {"absolute positions to 131071", 4, 8, 128, 128, 0, 500000, 1, 0, true, {131071, 65536, 100000, 32768}},
      {"wide head, more pairs than a block's threads", 2, 2, 1024, 1024, 0, 1e4f, 1, 0, false, {5, 77}},
      {"wider head, more pairs than the angle table", 1, 1, 2056, 2056, 1, 1e4f, 1, 0, false, {9}},
      {"prefill, one block per row", 256, 32, 128, 128, 1, 1e6f, 1, 0, false, {}},  // rows >= SMs, heads > a block's
      {"0 rows", 0, 8, 64, 64, 0, 1e4f, 1, 0, false, {}},
  };
  for (const grid& c : cases) {
    const std::string name = std::string("rope ") + c.what + " [" + std::to_string(c.rows) + ", " +
                             std::to_string(c.heads) + ", " + std::to_string(c.hd) + "]";
    rng g = seeded(name);
    std::vector<int32_t> pos = c.pos;
    for (int64_t r = int64_t(pos.size()); r < c.rows; ++r) pos.push_back(int32_t(g() % 8192));
    std::vector<float> ff(c.nd / 2);
    for (size_t i = 0; i < ff.size(); ++i)
      ff[i] = i < ff.size() / 4 ? 1.0f : i < ff.size() / 2 ? 2.5f : 8.0f;  // Llama 3's shape
    const larch_rope_params p = {c.nd, c.neox, 4096, c.base, c.freq_scale, c.ext, 1.0f, 32.0f, 1.0f, nullptr};
    std::vector<arg> args = {
        out("rope", LARCH_F16, {c.rows, c.heads, c.hd}, halves(size_t(c.rows * c.heads * c.hd), g)),
        in(LARCH_I32, {c.rows}, raw(pos))};
    if (c.ff) args.push_back(in(LARCH_F32, {int64_t(c.nd / 2)}, raw(ff)));
    test t = {name, args, [p, ff = c.ff](larch_stream* s, const larch_tensor* x) {
                larch_rope_params q = p;
                q.freq_factors = ff ? &x[2] : nullptr;
                return larch_rope(s, &x[0], &x[1], &q);
              }};
    for (const int32_t v : pos) t.pos_max = std::max(t.pos_max, double(v));
    add_test(std::move(t));
  }
  rng g = seeded("rope invalid");
  const auto run = [](uint32_t nd) {
    return [nd](larch_stream* s, const larch_tensor* x) {
      const larch_rope_params p = {nd, 0, 4096, 1e4f, 1, 0, 1, 32, 1, nullptr};
      return larch_rope(s, &x[0], &x[1], &p);
    };
  };
  invalid("rope, 0 heads", {in(LARCH_F16, {4, 0, 64}, {}), in(LARCH_I32, {4}, raw(std::vector<int32_t>{0, 1, 2, 3}))},
          run(64));
  invalid("rope, odd n_dims",
          {in(LARCH_F16, {2, 2, 8}, halves(32, g)), in(LARCH_I32, {2}, raw(std::vector<int32_t>{0, 1}))}, run(7));
  invalid("rope, one position for two rows",
          {in(LARCH_F16, {2, 2, 8}, halves(32, g)), in(LARCH_I32, {1}, raw(std::vector<int32_t>{0}))}, run(8));
}

static void qk_norm() {
  const auto run = [](larch_stream* s, const larch_tensor* t) { return larch_qk_norm(s, &t[0], &t[1], 1e-6f); };
  // head_dim 10 is 20 bytes, not a multiple of the f16 vector width: the scalar path. From 16 rows a GPU may put
  // several heads in one threadgroup, and 12 heads leave a part-filled last one.
  for (const auto& [rows, heads, hd] :
       {std::tuple<int64_t, int64_t, int64_t>{1, 16, 128}, {5, 2, 128}, {3, 14, 64}, {2, 3, 10}, {0, 4, 64},
        {16, 4, 128}, {17, 12, 128}, {32, 40, 64}}) {
    const std::string name =
        "qk_norm [" + std::to_string(rows) + ", " + std::to_string(heads) + ", " + std::to_string(hd) + "]";
    rng g = seeded(name);
    add_test({name,
              {out("qk_norm", LARCH_F16, {rows, heads, hd}, halves(size_t(rows * heads * hd), g)),
               in(LARCH_F32, {hd}, floats(size_t(hd), g, -2, 2))},
              run});
  }
  rng g = seeded("qk_norm invalid");
  invalid("qk_norm, 0 heads", {in(LARCH_F16, {2, 0, 128}, {}), in(LARCH_F32, {128}, floats(128, g, 0, 1))}, run);
  invalid("qk_norm, w not [head_dim]",
          {in(LARCH_F16, {2, 2, 8}, halves(32, g)), in(LARCH_F32, {4}, floats(4, g, 0, 1))}, run);
}

static larch_status mmvq_run(larch_stream* s, const larch_tensor* t) {
  return larch_mmvq(s, &t[0], &t[1], &t[2], &t[3]);
}
static larch_status mmvq_nobias(larch_stream* s, const larch_tensor* t) {
  return larch_mmvq(s, &t[0], &t[1], nullptr, &t[2]);
}

using mmvq_fn = decltype(&larch_mmvq);
// larch_mmvq_verify inside graph_begin/end/launch where the device captures, eagerly where it answers UNSUPPORTED
// (the CPU reference). A failed op still ends the capture: its graph is discarded.
static larch_status verify_captured(larch_stream* s, const larch_tensor* w, const larch_tensor* x,
                                     const larch_tensor* bias, const larch_tensor* out) {
  const larch_status begun = larch_graph_begin(s);
  if (begun != LARCH_OK) return begun == LARCH_ERR_UNSUPPORTED ? larch_mmvq_verify(s, w, x, bias, out) : begun;
  larch_graph* graph = nullptr;
  larch_status status = larch_mmvq_verify(s, w, x, bias, out);
  const larch_status ended = larch_graph_end(s, &graph);
  if (status == LARCH_OK) status = ended != LARCH_OK ? ended : larch_graph_launch(graph, s);
  if (status == LARCH_OK) status = larch_stream_sync(s);  // the exec must not be destroyed while it runs
  larch_graph_destroy(graph);
  return status;
}
// The ops mmvq_case runs, by the prefix of their cases' names.
struct mmvq_op {
  const char* name;
  mmvq_fn run;
};
static const mmvq_op MMVQ = {"mmvq", larch_mmvq}, VERIFY = {"mmvq_verify", larch_mmvq_verify},
                     CAPTURED = {"mmvq_verify captured", verify_captured}, LARGE_X = {"mmvq large x", larch_mmvq};

// x as halves() draws it, then in every 16-run along k element 5 is +-30000 and element 9 +-0.001: outliers
// near the f16 maximum beside values near its smallest, so a half2 decode that does not scale cannot hold.
static std::shared_ptr<const bytes> large_x(size_t n, rng& g) {
  bytes b = halves(n, g);
  for (size_t i = 0; i + 16 <= n; i += 16)
    for (const auto [at, h] : {std::pair{5, uint16_t(0x7753)}, std::pair{9, uint16_t(0x1419)}}) {
      const uint16_t v = h | uint16_t((g() & 1) << 15);  // f16 30000 and 0.001, sign from the rng
      std::memcpy(&b[2 * (i + at)], &v, 2);
    }
  return std::make_shared<const bytes>(std::move(b));
}

// One mmvq case; w_skew and x_skew move w and x off 16-byte alignment, which larch.h allows. padded: a
// block after each w row, an f16 after each x row (x.p aligned, its rows 2 bytes off 16), a bias every
// other f32 and 3 poison elements after each out row, which must stay poison. w: a dense [n, k] weight shared
// with other cases; null draws the case's own. x: likewise for the activations. op: what runs, always under
// mmvq's tolerances.
static void mmvq_case(larch_dtype q, int64_t rows, int64_t n, int64_t k, larch_dtype ot, bool bias,
                      uint64_t w_skew = 0, uint64_t x_skew = 0, bool padded = false,
                      std::shared_ptr<const bytes> w = nullptr, const mmvq_op& op = MMVQ,
                      std::shared_ptr<const bytes> x = nullptr) {
  std::string name = std::string(op.name) + " " + DTYPE[q] + " [" + std::to_string(n) + ", " + std::to_string(k) +
                     "] x " + std::to_string(rows) + " rows -> " + DTYPE[ot] + (bias ? " + bias" : "");
  if (w_skew) name += ", w at +" + std::to_string(w_skew);
  if (x_skew) name += ", x at +" + std::to_string(x_skew);
  if (padded) name += ", padded rows";
  rng g = seeded(name);
  const int64_t p = padded;
  std::vector<arg> args = {in(q, {n, k}, w ? bytes() : quant(q, n, k + p * block_of(q), g)),
                           in(LARCH_F16, {rows, k}, halves(size_t(rows * (k + p)), g))};
  if (w) args[0].data = std::move(w);
  if (x) args[1].data = std::move(x);
  args[0].skew = w_skew;
  args[1].skew = x_skew;
  if (bias) args.push_back(in(LARCH_F32, {n}, floats(size_t(n * (1 + p)), g, -1, 1)));
  args.push_back(out("mmvq", ot, {rows, n}, bytes(size_t(nbytes(ot, {rows, n + 3 * p})), 0xFF)));
  for (arg& a : args) a.pad = p * (a.out ? 3 : 1);
  add_test({name, args, [run = op.run, bias](larch_stream* s, const larch_tensor* t) {
              return run(s, &t[0], &t[1], bias ? &t[2] : nullptr, &t[bias ? 3 : 2]);
            }});
}

static void mmvq() {
  for (const larch_dtype q : QUANTS) {
    const bool k_type = block_of(q) == 256;
    const std::tuple<int64_t, int64_t, int64_t> shapes[] = {
        {1, 1, block_of(q)}, {16, 67, k_type ? 768 : 896}, {7, 33, k_type ? 2816 : 4864}};
    for (size_t si = 0; si < 3; ++si)
      for (const larch_dtype ot : {LARCH_F16, LARCH_F32}) {
        const auto [rows, n, k] = shapes[si];
        mmvq_case(q, rows, n, k, ot, (si + (ot == LARCH_F32)) % 2);
      }
    // Every row count the kernel is built for, at k = 2304: 72 lane units of 32 weights (Q8_0: 144 of 16), more
    // than KS 2's 64 lanes, so lanes take more than one step (the next unit's loads in flight) at every CW and the
    // table's KS, which n 6144 keeps (6144 / 16 columns per block >= 2 blocks per SM on GPUs up to 192 SMs); the
    // 16 share one weight per format, so the grid holds one copy. On a wave32 GPU with >= 18 SMs, n 67 leaves the
    // grid short of SMs and KS rises to 4 (128 lanes) instead: at k 2304, and at k 4608, where the 144 units
    // (Q8_0: 288) again give lanes more than one step.
    rng wg = seeded(std::string("mmvq ") + DTYPE[q] + " [6144, 2304] weight");
    const auto w = std::make_shared<const bytes>(quant(q, 6144, 2304, wg));
    for (int64_t rows = 1; rows <= 16; ++rows) mmvq_case(q, rows, 6144, 2304, LARCH_F16, rows % 2, 0, 0, false, w);
    for (int64_t rows = 1; rows <= 16; ++rows) mmvq_case(q, rows, 67, 2304, LARCH_F16, rows % 2);
    for (int64_t rows = 1; rows <= 16; ++rows) mmvq_case(q, rows, 67, 4608, LARCH_F16, rows % 2 == 0);
    // The 2-byte load paths, one argument off alignment at a time and both; then rows long enough for several
    // lane steps (the next step's loads in flight) and for every split-K slice to hold a share; then padded rows
    // at two column widths.
    for (const int64_t rows : {1, 9}) {
      mmvq_case(q, rows, 67, 768, LARCH_F16, false, 2, 0);
      mmvq_case(q, rows, 67, 768, LARCH_F16, false, 0, 2);
      mmvq_case(q, rows, 67, 768, LARCH_F16, false, 2, 2);
    }
    for (const int64_t rows : {1, 16}) mmvq_case(q, rows, 67, 5120, LARCH_F32, true);
    for (const int64_t rows : {2, 9})
      mmvq_case(q, rows, 67, 2304, rows == 2 ? LARCH_F16 : LARCH_F32, true, 0, 0, true);
  }
  // Activations at the f16 range (see large_x): the one-row and the routed kernel must not overflow. f32 out, so
  // the true sums stay in range and the case tests the decode, not the output store.
  for (const larch_dtype q : {LARCH_Q4_K, LARCH_Q6_K, LARCH_Q3_K})
    for (const int64_t rows : {1, 3}) {
      rng xg = seeded(std::string("mmvq large x ") + DTYPE[q] + " " + std::to_string(rows) + " x");
      mmvq_case(q, rows, 67, 4608, LARCH_F32, false, 0, 0, false, nullptr, LARGE_X, large_x(size_t(rows * 4608), xg));
    }
  // Q3_K (the 35B's Q3_K_M) at decode shapes of the targets whose k is whole 256-weight blocks (m1 q, m5 k, m8 down)
  // and the 35B's k 2048: one row and two routed widths (sm_120 routes every Q3_K row).
  for (const auto& [n, k] : {std::pair<int64_t, int64_t>{4096, 4096}, {512, 3584}, {896, 4864}, {2048, 2048}})
    for (const int64_t rows : {1, 4, 8})
      mmvq_case(LARCH_Q3_K, rows, n, k, rows == 4 ? LARCH_F32 : LARCH_F16, rows == 1);
  // Wide eight-row Q4 with four lane steps, an odd column tail, bias and all padded/unaligned load paths.
  mmvq_case(LARCH_Q4_K, 8, 8193, 4096, LARCH_F32, true, 2, 2, true);
  // On a 70-SM GPU, KS rises at n 128 from the table's to 4 (k 4608; Q8_0 at rows 1 needs >= 33 SMs), and at
  // k 1536 from 1 to 2 (lanes would get < 16 weights at 4): every format at rows 8 and 16, the K quants at rows 4
  // too (Q8_0, Q4_0 and Q5_0 run their table's KS 2 there).
  for (const larch_dtype q : {LARCH_Q4_K, LARCH_Q8_0})
    for (const int64_t rows : {1, 4, 16}) mmvq_case(q, rows, 128, 4608, rows == 4 ? LARCH_F32 : LARCH_F16, rows == 1);
  for (const larch_dtype q : QUANTS)
    for (const int64_t rows : {4, 8, 16}) mmvq_case(q, rows, 128, 1536, LARCH_F16, false);
  rng g = seeded("mmvq 0 rows");
  add_test({"mmvq q4_k, 0 rows",
            {in(LARCH_Q4_K, {3, 256}, quant(LARCH_Q4_K, 3, 256, g)), in(LARCH_F16, {0, 256}, {}),
             out("mmvq", LARCH_F16, {0, 3})},
            mmvq_nobias});
  const bytes w = quant(LARCH_Q8_0, 2, 64, g);
  invalid("mmvq, 17 rows",
          {in(LARCH_Q8_0, {2, 64}, w), in(LARCH_F16, {17, 64}, halves(17 * 64, g)), poison(LARCH_F16, {17, 2})},
          mmvq_nobias);
  invalid("mmvq, f16 weight",
          {in(LARCH_F16, {2, 64}, halves(128, g)), in(LARCH_F16, {1, 64}, halves(64, g)), poison(LARCH_F16, {1, 2})},
          mmvq_nobias);
  invalid("mmvq, bias not [n]",
          {in(LARCH_Q8_0, {2, 64}, w), in(LARCH_F16, {1, 64}, halves(64, g)), in(LARCH_F32, {3}, floats(3, g, 0, 1)),
           poison(LARCH_F16, {1, 2})},
          mmvq_run);
}

// larch_mmvq_verify against the CPU reference's own, under mmvq's tolerances (the byte-equality to one-row mmvq
// is the verify ctest's). Rows start at 2: one row is mmvq's own case.
static void mmvq_verify() {
  for (const larch_dtype q : QUANTS) {
    // As mmvq's sweep at k 2304 (n 6144 keeps the table's KS); then n 67, where a 70-SM GPU raises KS; then the
    // rows past 16, which take two launches.
    rng wg = seeded(std::string("mmvq_verify ") + DTYPE[q] + " [6144, 2304] weight");
    const auto w = std::make_shared<const bytes>(quant(q, 6144, 2304, wg));
    for (int64_t rows = 2; rows <= 16; ++rows) {
      mmvq_case(q, rows, 6144, 2304, LARCH_F16, rows % 2, 0, 0, false, w, VERIFY);
      mmvq_case(q, rows, 67, 2304, LARCH_F16, rows % 2 == 0, 0, 0, false, nullptr, VERIFY);
    }
    for (const int64_t rows : {17, 32}) mmvq_case(q, rows, 67, 2304, LARCH_F32, true, 0, 0, false, nullptr, VERIFY);
    for (const int64_t rows : {2, 9}) {
      mmvq_case(q, rows, 33, block_of(q), LARCH_F16, rows % 2, 0, 0, false, nullptr, VERIFY);
      mmvq_case(q, rows, 67, 768, LARCH_F16, false, 2, 0, false, nullptr, VERIFY);
      mmvq_case(q, rows, 67, 768, LARCH_F16, false, 0, 2, false, nullptr, VERIFY);
      mmvq_case(q, rows, 67, 768, LARCH_F16, false, 2, 2, false, nullptr, VERIFY);
    }
    for (const int64_t rows : {2, 9, 17, 32})  // 17 and 32: the second launch's x and out at padded strides
      mmvq_case(q, rows, 67, 2304, LARCH_F32, true, 0, 0, true, nullptr, VERIFY);
    for (const int64_t rows : {4, 24})
      mmvq_case(q, rows, 67, 2304, rows == 4 ? LARCH_F16 : LARCH_F32, true, 0, 0, false, nullptr, CAPTURED);
  }
  for (const int64_t rows : {4, 8})  // the 35B's k 2048 in Q3_K_M
    mmvq_case(LARCH_Q3_K, rows, 2048, 2048, LARCH_F16, rows == 4, 0, 0, false, nullptr, VERIFY);
  rng g = seeded("mmvq_verify 33 rows");
  invalid("mmvq_verify, 33 rows",
          {in(LARCH_Q8_0, {2, 64}, quant(LARCH_Q8_0, 2, 64, g)), in(LARCH_F16, {33, 64}, halves(33 * 64, g)),
           poison(LARCH_F16, {33, 2})},
          [](larch_stream* s, const larch_tensor* t) { return larch_mmvq_verify(s, &t[0], &t[1], nullptr, &t[2]); });
}

static larch_status dequant_run(larch_stream* s, const larch_tensor* t) { return larch_dequant(s, &t[0], &t[1]); }

// One dequant case; skew moves w and out 2 bytes off 16-byte alignment. padded: a block after each w row and
// 3 poison elements after each out row, which must stay poison.
static void dequant_case(larch_dtype q, int64_t n, int64_t k, uint64_t skew = 0, bool padded = false) {
  std::string name = std::string("dequant ") + DTYPE[q] + " [" + std::to_string(n) + ", " + std::to_string(k) + "]";
  if (skew) name += ", w and out at +" + std::to_string(skew);
  if (padded) name += ", padded rows";
  rng g = seeded(name);
  const int64_t p = padded;
  std::vector<arg> args = {
      in(q, {n, k}, quant(q, n, k + p * block_of(q), g)),
      out("dequant", LARCH_F16, {n, k}, bytes(size_t(nbytes(LARCH_F16, {n, k + 3 * p})), 0xFF))};
  args[0].pad = p;
  args[1].pad = 3 * p;
  for (arg& a : args) a.skew = skew;
  add_test({name, args, dequant_run});
}

static void dequant() {
  for (const larch_dtype q : ID_QUANTS) {
    const bool k_type = block_of(q) == 256;
    for (const auto& [n, k] :
         {std::pair<int64_t, int64_t>{1, block_of(q)}, {5, k_type ? 768 : 896}, {33, k_type ? 2816 : 4864}})
      dequant_case(q, n, k);
    // The 2-byte loads of Q4_K and Q5_K and the element-at-a-time stores: w and out off alignment, then padded
    // rows.
    dequant_case(q, 5, k_type ? 768 : 896, 2);
    dequant_case(q, 5, k_type ? 768 : 896, 0, true);
  }
  // The 35B's Q3_K experts, gate/up [512, 2048] and down [2048, 512], as prefill's dequant + gemm reads them, and
  // those experts repacked to Q2_K.
  for (const larch_dtype q : {LARCH_Q3_K, LARCH_Q2_K})
    for (const auto& [n, k] : {std::pair<int64_t, int64_t>{512, 2048}, {2048, 512}}) {
      dequant_case(q, n, k);
      dequant_case(q, n, k, 2);
    }
  rng g = seeded("dequant invalid");
  invalid("dequant, out f32", {in(LARCH_Q4_K, {1, 256}, quant(LARCH_Q4_K, 1, 256, g)), poison(LARCH_F32, {1, 256})},
          dequant_run);
}

static void gemm() {
  struct grid {
    int64_t m, n, k;
    larch_dtype c;
    bool bias;
    float alpha, beta;
  };
  const grid cases[] = {{17, 33, 256, LARCH_F16, false, 1, 0},  {33, 65, 896, LARCH_F32, true, 1, 0},
                        {64, 128, 512, LARCH_F16, true, 1, 0},  {5, 7, 3, LARCH_F16, false, 0.5f, 0.25f},
                        {40, 24, 2048, LARCH_F32, false, 2, 1}, {12, 20, 160, LARCH_F32, true, 0.5f, 2},
                        {1, 1, 1, LARCH_F32, true, 1, 0},       {0, 16, 64, LARCH_F16, false, 1, 0},
                        // A decode's MoE router, and the most rows the Metal GEMV takes.
                        {1, 128, 2048, LARCH_F32, false, 1, 0}, {8, 33, 896, LARCH_F16, true, 0.5f, 0.25f},
                        // A check span at a MoE router's shape and f32 beta on CUDA's row-stable gemm, then one
                        // row past it, and rows on cuBLAS up to and past the engine's mmvq slice.
                        {4, 256, 2048, LARCH_F32, true, 1, 0},  {6, 20, 160, LARCH_F32, true, 0.5f, 2},
                        {9, 512, 2048, LARCH_F16, false, 1, 0},
                        {15, 256, 2048, LARCH_F32, true, 1, 0}, {16, 128, 2048, LARCH_F16, true, 1, 0},
                        {17, 512, 2048, LARCH_F16, false, 1, 0}};
  for (const grid& c : cases) {
    const std::string name = "gemm [" + std::to_string(c.m) + ", " + std::to_string(c.k) + "] x [" +
                             std::to_string(c.n) + ", " + std::to_string(c.k) + "]^T -> " + DTYPE[c.c] +
                             (c.bias ? " + bias" : "") + ", alpha " + std::to_string(c.alpha) + ", beta " +
                             std::to_string(c.beta);
    rng g = seeded(name);
    std::vector<arg> args = {in(LARCH_F16, {c.m, c.k}, halves(size_t(c.m * c.k), g)),
                             in(LARCH_F16, {c.n, c.k}, halves(size_t(c.n * c.k), g))};
    if (c.bias) args.push_back(in(LARCH_F32, {c.n}, floats(size_t(c.n), g, -1, 1)));
    // beta 0 never reads c, so it starts as NaN; otherwise as values.
    const size_t mn = size_t(c.m * c.n);
    args.push_back(out("gemm", c.c, {c.m, c.n},
                       c.beta == 0         ? bytes{}
                       : c.c == LARCH_F16 ? halves(mn, g)
                                           : floats(mn, g, -4, 4)));
    add_test({name, args, [c](larch_stream* s, const larch_tensor* t) {
                return larch_gemm(s, &t[0], &t[1], c.bias ? &t[2] : nullptr, &t[c.bias ? 3 : 2], c.alpha, c.beta);
              }});
  }
  rng g = seeded("gemm invalid");
  const auto run = [](larch_stream* s, const larch_tensor* t) {
    return larch_gemm(s, &t[0], &t[1], nullptr, &t[2], 1, 0);
  };
  invalid("gemm, k differs",
          {in(LARCH_F16, {2, 8}, halves(16, g)), in(LARCH_F16, {3, 4}, halves(12, g)), poison(LARCH_F16, {2, 3})},
          run);
  invalid(
      "gemm, a f32",
      {in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)), in(LARCH_F16, {3, 8}, halves(24, g)), poison(LARCH_F16, {2, 3})},
      run);
}

// larch_gemm at rows 1 to 8 (a check step's slice of a side weight; CUDA's row-stable gemm) on the decode shapes of
// MoE routers, shared-expert gates and DeltaNet's alpha and beta, f16 side weights at k 2048 and n 1 to 512, then
// alpha, beta and a k that ends mid-chunk: every row byte for byte the one-row call of that row, so a check span's
// rows are decode's (exact speculation). Rows past 8 promise no such thing; gemm's grid checks them against the CPU
// reference. The device under test alone; returns the failures.
static int gemm_rows(const side& x) {
  constexpr int64_t most = 8;
  int wrong = 0;
  const auto fail = [&](const std::string& what) {
    std::fprintf(stderr, "gemm rows: %s: %s\n", what.c_str(), larch_last_error(x.dev));
    ++wrong;
  };
  struct grid {
    int64_t n, k;
    larch_dtype ot;
    bool bias;
    float alpha, beta;
  };
  std::vector<grid> cases;
  for (const int64_t n : {1, 32, 128, 256, 512})
    for (const larch_dtype ot : {LARCH_F32, LARCH_F16})
      for (const bool bias : {false, true}) cases.push_back({n, 2048, ot, bias, 1, 0});
  for (const larch_dtype ot : {LARCH_F32, LARCH_F16}) cases.push_back({33, 2051, ot, true, 0.5f, 0.25f});
  for (const grid& c : cases) {
    const std::string name = "[" + std::to_string(c.n) + ", " + std::to_string(c.k) + "] -> " + DTYPE[c.ot] +
                             (c.bias ? " + bias" : "") + ", alpha " + std::to_string(c.alpha) + ", beta " +
                             std::to_string(c.beta);
    rng g = seeded("gemm rows " + name);
    const int64_t n = c.n, k = c.k;
    const uint64_t ob = c.ot == LARCH_F32 ? 4 : 2, out_bytes = uint64_t(most * n) * ob;
    const bytes a = halves(size_t(most * k), g), b = halves(size_t(n * k), g), bb = floats(size_t(n), g, -1, 1),
                init = c.ot == LARCH_F32 ? floats(size_t(most * n), g, -4, 4) : halves(size_t(most * n), g);
    larch_buffer *ab = nullptr, *wb = nullptr, *biasb = nullptr, *one = nullptr, *many = nullptr;
    bool ok = larch_buffer_alloc(x.dev, a.size(), &ab) == LARCH_OK &&
              larch_buffer_alloc(x.dev, b.size(), &wb) == LARCH_OK &&
              larch_buffer_alloc(x.dev, bb.size(), &biasb) == LARCH_OK &&
              larch_buffer_alloc(x.dev, out_bytes, &one) == LARCH_OK &&
              larch_buffer_alloc(x.dev, out_bytes, &many) == LARCH_OK &&
              larch_upload(x.s, ab, 0, a.data(), a.size()) == LARCH_OK &&
              larch_upload(x.s, wb, 0, b.data(), b.size()) == LARCH_OK &&
              larch_upload(x.s, biasb, 0, bb.data(), bb.size()) == LARCH_OK &&
              larch_upload(x.s, one, 0, init.data(), out_bytes) == LARCH_OK;
    const larch_tensor wt = {wb, 0, LARCH_F16, 2, {n, k}, {k, 1}}, bt = {biasb, 0, LARCH_F32, 1, {n}, {1}};
    for (int64_t r = 0; r < most && ok; ++r) {
      const larch_tensor at = {ab, uint64_t(r * k * 2), LARCH_F16, 2, {1, k}, {k, 1}},
                          o = {one, uint64_t(r * n) * ob, c.ot, 2, {1, n}, {n, 1}};
      ok = larch_gemm(x.s, &at, &wt, c.bias ? &bt : nullptr, &o, c.alpha, c.beta) == LARCH_OK;
    }
    bytes want(out_bytes), got(out_bytes);
    ok = ok && larch_download(x.s, one, 0, want.data(), out_bytes) == LARCH_OK &&
         larch_stream_sync(x.s) == LARCH_OK;
    for (const int64_t rows : {2, 3, 4, 5, 6, 7, 8}) {
      if (!ok) break;
      const larch_tensor at = {ab, 0, LARCH_F16, 2, {rows, k}, {k, 1}}, o = {many, 0, c.ot, 2, {rows, n}, {n, 1}};
      ok = larch_upload(x.s, many, 0, init.data(), out_bytes) == LARCH_OK &&
           larch_gemm(x.s, &at, &wt, c.bias ? &bt : nullptr, &o, c.alpha, c.beta) == LARCH_OK &&
           larch_download(x.s, many, 0, got.data(), uint64_t(rows * n) * ob) == LARCH_OK &&
           larch_stream_sync(x.s) == LARCH_OK;
      for (int64_t r = 0; r < rows && ok; ++r)
        if (std::memcmp(&got[size_t(r * n) * ob], &want[size_t(r * n) * ob], size_t(n) * ob))
          fail(name + ", " + std::to_string(rows) + " rows: row " + std::to_string(r) + " differs from one row");
    }
    if (!ok) fail(name + ": a call failed");
    for (larch_buffer* buf : {ab, wb, biasb, one, many}) larch_buffer_free(buf);
  }
  return wrong;
}

static void silu_mul() {
  const auto run = [](larch_stream* s, const larch_tensor* t) { return larch_silu_mul(s, &t[0], &t[1], &t[2]); };
  for (const auto& [rows, ff] : {std::pair<int64_t, int64_t>{1, 8192}, {7, 4864}, {16, 3}, {0, 8}}) {
    const std::string name = "silu_mul [" + std::to_string(rows) + ", " + std::to_string(ff) + "]";
    rng g = seeded(name);
    const size_t n = size_t(rows * ff);
    add_test({name,
              {in(LARCH_F16, {rows, ff}, halves(n, g, 5)), in(LARCH_F16, {rows, ff}, halves(n, g)),
               out("silu_mul", LARCH_F16, {rows, ff})},
              run});
  }
  rng g = seeded("silu_mul invalid");
  invalid("silu_mul, up narrower",
          {in(LARCH_F16, {2, 8}, halves(16, g)), in(LARCH_F16, {2, 4}, halves(8, g)), poison(LARCH_F16, {2, 8})},
          run);
}

static void add() {
  const auto run = [](larch_stream* s, const larch_tensor* t) { return larch_add(s, &t[0], &t[1], &t[2]); };
  for (const auto& [rows, d] : {std::pair<int64_t, int64_t>{1, 4096}, {9, 896}, {33, 7}, {0, 16}}) {
    const std::string name = "add [" + std::to_string(rows) + ", " + std::to_string(d) + "]";
    rng g = seeded(name);
    const size_t n = size_t(rows * d);
    add_test({name,
              {in(LARCH_F32, {rows, d}, floats(n, g, -8, 8)), in(LARCH_F16, {rows, d}, halves(n, g)),
               out("add", LARCH_F32, {rows, d})},
              run});
  }
  rng g = seeded("add in place");
  add_test({"add [5, 64], out == a",
            {out("add", LARCH_F32, {5, 64}, floats(320, g, -8, 8)), in(LARCH_F16, {5, 64}, halves(320, g))},
            [](larch_stream* s, const larch_tensor* t) { return larch_add(s, &t[0], &t[1], &t[0]); }});
  invalid("add, b f32",
          {in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)), in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)),
           poison(LARCH_F32, {2, 8})},
          run);
  invalid(
      "add, out f16",
      {in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)), in(LARCH_F16, {2, 8}, halves(16, g)), poison(LARCH_F16, {2, 8})},
      run);
}

// A paged KV pool: n_layers of n_blocks * block_tokens slots, both buffers random f16.
struct geometry {
  uint32_t n_blocks, block_tokens, n_layers, kv_heads, hd, layer;
  size_t elems() const { return size_t(n_layers) * n_blocks * block_tokens * kv_heads * hd; }
  larch_kv_pool at(const larch_tensor* t, size_t k, larch_dtype dtype = LARCH_F16) const {
    return {t[k].buf, t[k + 1].buf, n_blocks, block_tokens, n_layers, kv_heads, hd, dtype, layer};
  }
};

static void kv_write() {
  const geometry cases[] = {{12, 16, 3, 2, 64, 2}, {12, 16, 3, 8, 128, 0}, {12, 16, 3, 1, 8, 1}};
  for (const geometry& p : cases)
    for (const int64_t rows : {1, 37}) {
      const std::string name = "kv_write " + std::to_string(rows) + " rows into layer " + std::to_string(p.layer) +
                               " of " + std::to_string(p.kv_heads) + " x " + std::to_string(p.hd);
      rng g = seeded(name);
      std::vector<int32_t> slots(static_cast<size_t>(p.n_blocks * p.block_tokens));
      std::iota(slots.begin(), slots.end(), 0);
      std::shuffle(slots.begin(), slots.end(), g);
      slots.resize(size_t(rows));
      for (size_t r = 3; r < slots.size(); r += 5) slots[r] = -1;  // bucket padding
      const size_t n = size_t(rows) * p.kv_heads * p.hd;
      add_test({name,
                {in(LARCH_F16, {rows, p.kv_heads, p.hd}, halves(n, g)),
                 in(LARCH_F16, {rows, p.kv_heads, p.hd}, halves(n, g)), in(LARCH_I32, {rows}, raw(slots)),
                 pool(halves(p.elems(), g), true), pool(halves(p.elems(), g), true)},
                [p](larch_stream* s, const larch_tensor* t) {
                  const larch_kv_pool kv = p.at(t, 3);
                  return larch_kv_write(s, &t[0], &t[1], &t[2], &kv);
                }});
    }
  const geometry p = {2, 16, 1, 1, 8, 0};
  rng g = seeded("kv_write invalid");
  const auto run = [p](larch_dtype dtype) {
    return [p, dtype](larch_stream* s, const larch_tensor* t) {
      const larch_kv_pool kv = p.at(t, 3, dtype);
      return larch_kv_write(s, &t[0], &t[1], &t[2], &kv);
    };
  };
  const bytes slot = raw(std::vector<int32_t>{5});
  invalid("kv_write, head_dim 4 into a pool of 8",
          {in(LARCH_F16, {1, 1, 4}, halves(4, g)), in(LARCH_F16, {1, 1, 4}, halves(4, g)), in(LARCH_I32, {1}, slot),
           pool(halves(p.elems(), g), true), pool(halves(p.elems(), g), true)},
          run(LARCH_F16));
  // Room for f32, so only the dtype is wrong.
  invalid("kv_write, an f32 pool",
          {in(LARCH_F16, {1, 1, 8}, halves(8, g)), in(LARCH_F16, {1, 1, 8}, halves(8, g)), in(LARCH_I32, {1}, slot),
           pool(halves(2 * p.elems(), g), true), pool(halves(2 * p.elems(), g), true)},
          run(LARCH_F32));
}

// Row r of a block table: ceil(len / block_tokens) distinct shuffled blocks, then -1 to `width`.
static std::vector<int32_t> table(const std::vector<int64_t>& lens, const geometry& p, int64_t width, rng& g) {
  std::vector<int32_t> t;
  for (const int64_t len : lens) {
    std::vector<int32_t> blocks(p.n_blocks);
    std::iota(blocks.begin(), blocks.end(), 0);
    std::shuffle(blocks.begin(), blocks.end(), g);
    const int64_t used = (len + p.block_tokens - 1) / p.block_tokens;
    if (used > int64_t(p.n_blocks) || used > width) {
      std::fprintf(stderr, "grid error: %lld positions need %lld blocks\n", (long long)len, (long long)used);
      std::exit(2);
    }
    for (int64_t b = 0; b < width; ++b) t.push_back(b < used ? blocks[size_t(b)] : -1);
  }
  return t;
}

static void attention() {
  struct decode {
    geometry p;
    int64_t heads;
    std::vector<int64_t> lens;
    uint32_t splits;
  };
  const decode dec[] = {
      {{24, 16, 2, 2, 64, 1}, 14, {37, 0, 300, 1}, 0},  // GQA 7, a padding row (seq_len 0, table all -1)
      {{24, 16, 2, 8, 128, 1}, 32, {17, 16, 129}, 0},   // GQA 4, block edges
      {{24, 16, 2, 8, 128, 1}, 32, {17, 16, 129}, 1},   // splits are a speed hint: the same answer
      {{24, 16, 2, 8, 128, 1}, 32, {17, 16, 129}, 3},
      {{24, 16, 2, 8, 128, 1}, 32, {17, 16, 129}, 8},
      {{24, 16, 2, 4, 128, 1}, 8, {384}, 0},  // every block of the pool
      {{24, 16, 2, 1, 8, 0}, 1, {1, 2}, 0},
      {{24, 16, 2, 2, 256, 1}, 16, {37, 300}, 0},  // Qwen 3.5's full attention: head_dim 256, GQA 8
      {{24, 16, 2, 2, 256, 1}, 16, {37, 300}, 1},  // and both of its paths: single pass, split-K
      {{24, 16, 2, 2, 256, 1}, 16, {37, 300}, 3},
  };
  for (const decode& c : dec) {
    std::string name = "attn_decode " + std::to_string(c.heads) + " heads over " + std::to_string(c.p.kv_heads) +
                       " x " + std::to_string(c.p.hd) + ", seq_lens";
    for (const int64_t l : c.lens) name += " " + std::to_string(l);
    name += ", splits " + std::to_string(c.splits);
    rng g = seeded(name);
    const int64_t rows = int64_t(c.lens.size()),
                  width =
                      (*std::max_element(c.lens.begin(), c.lens.end()) + c.p.block_tokens - 1) / c.p.block_tokens + 1;
    std::vector<int32_t> lens(c.lens.begin(), c.lens.end());
    const float scale = 1.0f / std::sqrt(float(c.p.hd));
    add_test({name,
              {in(LARCH_F16, {rows, c.heads, c.p.hd}, halves(size_t(rows * c.heads * c.p.hd), g)),
               pool(halves(c.p.elems(), g), false), pool(halves(c.p.elems(), g), false),
               in(LARCH_I32, {rows, width}, raw(table(c.lens, c.p, width, g))), in(LARCH_I32, {rows}, raw(lens)),
               out("attn_decode", LARCH_F16, {rows, c.heads, c.p.hd})},
              [c, scale](larch_stream* s, const larch_tensor* t) {
                const larch_kv_pool kv = c.p.at(t, 1);
                return larch_attn_decode(s, &t[0], &kv, &t[3], &t[4], c.splits, scale, &t[5]);
              }});
  }

  struct prefill {
    geometry p;
    int64_t heads;
    std::vector<std::pair<int32_t, int32_t>> chunks;  // (start, len) per sequence
  };
  const prefill pre[] = {
      {{24, 16, 2, 2, 64, 1}, 14, {{0, 37}, {12, 8}, {280, 20}, {5, 0}}},  // cached prefixes, an empty chunk
      {{24, 16, 2, 8, 128, 1}, 32, {{100, 33}}},                           // a chunk across blocks, mid-block start
      {{24, 16, 2, 1, 8, 0}, 1, {{0, 1}, {0, 16}, {15, 2}}},
      {{24, 16, 2, 2, 256, 1}, 16, {{0, 70}, {200, 33}}},  // Qwen 3.5's full attention: head_dim 256
  };
  for (const prefill& c : pre) {
    std::string name = "attn_prefill " + std::to_string(c.heads) + " heads over " + std::to_string(c.p.kv_heads) +
                       " x " + std::to_string(c.p.hd) + ", chunks";
    std::vector<int32_t> start, len;
    std::vector<int64_t> span;
    for (const auto& [st, n] : c.chunks) {
      name += " " + std::to_string(st) + "+" + std::to_string(n);
      start.push_back(st);
      len.push_back(n);
      span.push_back(st + n);
    }
    rng g = seeded(name);
    const int64_t rows = std::accumulate(len.begin(), len.end(), int64_t(0)), seqs = int64_t(c.chunks.size());
    const int64_t width = (*std::max_element(span.begin(), span.end()) + c.p.block_tokens - 1) / c.p.block_tokens + 1;
    const float scale = 1.0f / std::sqrt(float(c.p.hd));
    add_test({name,
              {in(LARCH_F16, {rows, c.heads, c.p.hd}, halves(size_t(rows * c.heads * c.p.hd), g)),
               pool(halves(c.p.elems(), g), false), pool(halves(c.p.elems(), g), false),
               in(LARCH_I32, {seqs, width}, raw(table(span, c.p, width, g))), in(LARCH_I32, {seqs}, raw(start)),
               in(LARCH_I32, {seqs}, raw(len)), out("attn_prefill", LARCH_F16, {rows, c.heads, c.p.hd})},
              [c, scale](larch_stream* s, const larch_tensor* t) {
                const larch_kv_pool kv = c.p.at(t, 1);
                return larch_attn_prefill(s, &t[0], &kv, &t[3], &t[4], &t[5], scale, &t[6]);
              }});
  }

  const geometry p = {4, 16, 1, 2, 8, 0};
  rng g = seeded("attention invalid");
  const auto dec_run = [p](larch_stream* s, const larch_tensor* t) {
    const larch_kv_pool kv = p.at(t, 1);
    return larch_attn_decode(s, &t[0], &kv, &t[3], &t[4], 0, 0.35f, &t[5]);
  };
  const bytes one = raw(std::vector<int32_t>{1}), zeros = raw(std::vector<int32_t>{0, 0});
  invalid(
      "attn_decode, 3 heads over 2 kv heads",
      {in(LARCH_F16, {1, 3, 8}, halves(24, g)), pool(halves(p.elems(), g), false), pool(halves(p.elems(), g), false),
       in(LARCH_I32, {1, 1}, raw(std::vector<int32_t>{0})), in(LARCH_I32, {1}, one), poison(LARCH_F16, {1, 3, 8})},
      dec_run);
  invalid(
      "attn_decode, two table rows for one q row",
      {in(LARCH_F16, {1, 2, 8}, halves(16, g)), pool(halves(p.elems(), g), false), pool(halves(p.elems(), g), false),
       in(LARCH_I32, {2, 1}, zeros), in(LARCH_I32, {1}, one), poison(LARCH_F16, {1, 2, 8})},
      dec_run);
  invalid("attn_prefill, q_start_pos not [seqs]",
          {in(LARCH_F16, {1, 2, 8}, halves(16, g)), pool(halves(p.elems(), g), false),
           pool(halves(p.elems(), g), false), in(LARCH_I32, {1, 1}, raw(std::vector<int32_t>{0})),
           in(LARCH_I32, {2}, zeros), in(LARCH_I32, {1}, one), poison(LARCH_F16, {1, 2, 8})},
          [p](larch_stream* s, const larch_tensor* t) {
            const larch_kv_pool kv = p.at(t, 1);
            return larch_attn_prefill(s, &t[0], &kv, &t[3], &t[4], &t[5], 0.35f, &t[6]);
          });
}

static void sample_prep() {
  struct grid {
    const char* what;
    int64_t vocab;
    std::vector<larch_sample_params> per_row;
    bool penalty, ties, masked, zeros;  // zeros: ~200 entries +0.0 or -0.0 lead a negative row
    int64_t peak = -1;                  // this logit alone above a row below -1: losing it changes ids[0]
  };
  const grid cases[] = {
      {"qwen vocab, penalties", 151936, {{1, 0, 1.0f, 0}, {1, 40, 1.1f, 64}, {1, 64, 0.7f, 10}}, true, false, false},
      {"llama3 vocab, padding row", 128256, {{1, 64, 1.15f, 64}, {1, 0, 1.15f, 64}}, true, false, false},
      {"vocab 64, top_k 0", 64, {{1, 0, 1.3f, 5}}, true, false, false},
      {"small candidates, penalty and ties", 151936, {{1, 1, 1.1f, 64}, {1, 3, 0.7f, 10}, {1, 8, 1, 0}}, true, true, false},
      {"one candidate, penalties and ties", 131072, {{1, 1, 1.1f, 64}, {1, 1, 0.7f, 64}}, true, true, false},
      {"four candidates, penalties and ties", 131072, {{1, 1, 1.1f, 64}, {1, 3, 0.7f, 64}, {1, 4, 1, 0}}, true, true, false},
      {"one candidate, +-0.0 ties", 4096, {{1, 1, 1, 0}}, false, false, false, true},
      {"four candidates, +-0.0 ties", 4096, {{1, 3, 1, 0}, {1, 4, 1, 0}}, false, false, false, true},
      {"one candidate, vocab tail", 1, {{1, 1, 1, 0}}, false, false, false},
      {"four candidates, chunk edge", 4097, {{1, 3, 1, 0}}, false, false, false, false, 4096},
      {"one candidate, 65 rows", 1000, std::vector<larch_sample_params>(65, {1, 1, 1, 0}), false, false, false},
      {"small candidates, +-0.0 ties", 4096, {{1, 1, 1, 0}, {1, 7, 1, 0}}, false, false, false, true},
      {"small candidates, chunk edge", 4097, {{1, 1, 1, 0}}, false, false, false, false, 4096},
      {"small candidates, masked logits", 32000, {{1, 1, 1, 0}, {1, 3, 1, 0}}, false, false, true},
      {"small candidates, vocab tail", 3, {{1, 1, 1, 0}, {1, 3, 1, 0}}, false, true, false},
      {"small candidates, 65 rows", 1000, std::vector<larch_sample_params>(65, {1, 3, 1, 0}), false, false, false},
      {"vocab 100, ties",
       100,
       {{1, 1, 1, 0},
        {1, 7, 1, 0},
        {1, 64, 1, 0},
        {1, 0, 1, 0},
        {1, 33, 1, 0},
        {1, 2, 1, 0},
        {1, 64, 1, 0},
        {1, 9, 1, 0}},
       false,
       true,
       false},
      {"vocab 32000, -inf logits", 32000, {{1, 64, 1, 0}, {1, 64, 1, 0}}, false, false, true},
      {"vocab 4096, +-0.0 ties", 4096, {{1, 64, 1, 0}}, false, false, false, true},
      {"vocab 1000, 65 rows", 1000, std::vector<larch_sample_params>(65, {1, 64, 1, 0}), false, false, false},
      {"vocab 4097, chunk edge", 4097, {{1, 64, 1, 0}}, false, false, false, false, 4096},
      {"vocab 262145, three passes", 262145, {{1, 64, 1, 0}}, false, false, false},  // scratch A -> B -> A
  };
  for (const grid& c : cases) {
    const int64_t rows = int64_t(c.per_row.size());
    const std::string name =
        std::string("sample_prep ") + c.what + " [" + std::to_string(rows) + ", " + std::to_string(c.vocab) + "]";
    rng g = seeded(name);
    std::uniform_real_distribution<float> u(-8, 8);
    std::vector<float> logits(static_cast<size_t>(rows * c.vocab));
    // A row below -1 also shows a slot read past it (the guard bytes are ~ -3e-16) in the ids.
    for (float& x : logits) x = c.ties ? std::round(u(g) * 2) / 4 : c.zeros || c.peak >= 0 ? -std::fabs(u(g)) - 1 : u(g);
    if (c.peak >= 0) logits[size_t(c.peak)] = 9;
    if (c.zeros)  // the zeros are the top values, so their order is pure id order, -0.0 bit-exact
      for (int i = 0; i < 200; ++i) logits[g() % uint32_t(c.vocab)] = g() & 1 ? -0.0f : 0.0f;
    if (c.masked)  // 20 per row
      for (int64_t r = 0; r < rows; ++r)
        for (int i = 0; i < 20; ++i)
          logits[size_t(r * c.vocab + g() % uint32_t(c.vocab))] = -std::numeric_limits<float>::infinity();
    std::vector<arg> args = {in(LARCH_F32, {rows, c.vocab}, raw(logits))};
    if (c.penalty) {
      // Every third id is one of the row's top 8, so the penalty moves the top 64; ids 0 and 1 are the
      // same (penalised once); every ninth is -1 padding; row 1 of the llama3 case is all padding.
      std::vector<int32_t> ids(static_cast<size_t>(rows * 64)), order(static_cast<size_t>(c.vocab));
      for (int64_t r = 0; r < rows; ++r) {
        const float* row = &logits[size_t(r * c.vocab)];
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + 8, order.end(),
                          [&](int32_t a, int32_t b) { return row[a] > row[b]; });
        for (int64_t i = 0; i < 64; ++i)
          ids[size_t(r * 64 + i)] = (r == 1 && c.vocab == 128256) || i % 9 == 8 ? -1
                                    : i % 3 == 0                                ? order[g() % 8]
                                                                                : int32_t(g() % uint32_t(c.vocab));
        ids[size_t(r * 64 + 1)] = ids[size_t(r * 64)];
      }
      args.push_back(in(LARCH_I32, {rows, 64}, raw(ids)));
    }
    args.push_back(out("sample_prep", LARCH_F32, {rows, 64}));
    args.push_back(out("sample_prep", LARCH_I32, {rows, 64}));
    args.push_back(out("sample_prep", LARCH_F32, {rows}));
    args.back().tl = T("sample_prep lse", LARCH_F32);
    add_test({name, args, [per_row = c.per_row, pen = c.penalty](larch_stream* s, const larch_tensor* t) {
                const size_t o = pen ? 2 : 1;
                return larch_sample_prep(s, &t[0], per_row.data(), pen ? &t[1] : nullptr, &t[o], &t[o + 1], &t[o + 2]);
              }});
  }
  rng g = seeded("sample_prep invalid");
  const auto run = [](uint32_t top_k) {
    return [top_k](larch_stream* s, const larch_tensor* t) {
      const larch_sample_params p = {1, top_k, 1, 0};
      return larch_sample_prep(s, &t[0], &p, nullptr, &t[1], &t[2], &t[3]);
    };
  };
  invalid("sample_prep, vals [rows, 32]",
          {in(LARCH_F32, {1, 100}, floats(100, g, -1, 1)), poison(LARCH_F32, {1, 32}), poison(LARCH_I32, {1, 64}),
           poison(LARCH_F32, {1})},
          run(0));
  invalid("sample_prep, top_k 65",
          {in(LARCH_F32, {1, 100}, floats(100, g, -1, 1)), poison(LARCH_F32, {1, 64}), poison(LARCH_I32, {1, 64}),
           poison(LARCH_F32, {1})},
          run(65));
}

// One mmvq_id case: `experts` stacked [n, k] experts, `slots` ids per row (repeats allowed), x one row per row
// (per_slot: one per (row, slot), the down projection's), out per slot or, weighted, their weighted sum.
// padded: as mmvq_case's, a block after each w row, an element after each x, ids and weights row, a bias every
// other f32 and 3 poison elements after each out row.
// w: a dense weight shared with other cases; null draws the case's own.
static void mmvq_id_case(larch_dtype q, int64_t experts, int64_t rows, int64_t slots, int64_t n, int64_t k,
                         bool per_slot, bool weighted, bool bias, larch_dtype ot, bool padded = false,
                         std::shared_ptr<const bytes> w = nullptr) {
  std::string name = std::string("mmvq_id ") + DTYPE[q] + " " + std::to_string(experts) + " x [" +
                     std::to_string(n) + ", " + std::to_string(k) + "], " + std::to_string(rows) + " rows of " +
                     std::to_string(slots) + (per_slot ? ", x per slot" : "") + (weighted ? ", weighted" : "") +
                     (bias ? " + bias" : "") + " -> " + DTYPE[ot] + (padded ? ", padded rows" : "");
  rng g = seeded(name);
  const int64_t p = padded, xk = (per_slot ? slots : 1) * k, on = weighted ? n : slots * n;
  std::uniform_int_distribution<int32_t> pick(0, int32_t(experts - 1));
  std::vector<int32_t> ids(size_t(rows * (slots + p)));
  for (int32_t& e : ids) e = pick(g);
  const std::vector<int64_t> xs = per_slot ? std::vector<int64_t>{rows, slots, k} : std::vector<int64_t>{rows, k};
  std::vector<arg> args = {in(q, {experts * n, k}, w ? bytes() : quant(q, experts * n, k + p * block_of(q), g)),
                           in(LARCH_F16, xs, halves(size_t(rows * (xk + p)), g)),
                           in(LARCH_I32, {rows, slots}, raw(ids))};
  if (w) args[0].data = std::move(w);
  if (bias) args.push_back(in(LARCH_F32, {experts * n}, floats(size_t(experts * n * (1 + p)), g, -1, 1)));
  if (weighted) args.push_back(in(LARCH_F32, {rows, slots}, floats(size_t(rows * (slots + p)), g, 0, 1)));
  const std::vector<int64_t> os = weighted ? std::vector<int64_t>{rows, n} : std::vector<int64_t>{rows, slots, n};
  args.push_back(out("mmvq_id", ot, os, bytes(size_t(nbytes(ot, {rows, on + 3 * p})), 0xFF)));
  for (arg& a : args) a.pad = p * (a.out ? 3 : 1);
  add_test({name, args, [bias, weighted](larch_stream* s, const larch_tensor* t) {
              const size_t o = 3 + bias + weighted;
              return larch_mmvq_id(s, &t[0], &t[1], &t[2], bias ? &t[3] : nullptr, weighted ? &t[3 + bias] : nullptr,
                                    &t[o]);
            }});
}

static void mmvq_id() {
  for (const larch_dtype q : ID_QUANTS) {
    const int64_t k = block_of(q) == 256 ? 768 : 896;
    mmvq_id_case(q, 4, 1, 2, 67, k, false, false, false, LARCH_F16);  // gate and up: one x row, out per slot
    mmvq_id_case(q, 4, 5, 2, 33, k, true, true, false, LARCH_F32);    // down: x per slot, the weighted sum
    mmvq_id_case(q, 8, 3, 8, 67, k, false, false, true, LARCH_F32);
    mmvq_id_case(q, 8, 2, 3, 33, k, true, true, true, LARCH_F16);
    mmvq_id_case(q, 4, 3, 2, 33, k, false, false, true, LARCH_F16, true);
    mmvq_id_case(q, 4, 2, 3, 67, k, true, true, true, LARCH_F32, true);
    mmvq_id_case(q, 2, 1, 1, 1, block_of(q), false, false, false, LARCH_F32);  // one block per row, one column
    // 72 pairs over 32 experts: Metal's per-pair kernels past 64 pairs (8 chats' decode rows and more).
    mmvq_id_case(q, 32, 9, 8, 67, k, false, false, true, LARCH_F16);
    mmvq_id_case(q, 32, 9, 8, 33, k, true, true, false, LARCH_F32);
    mmvq_id_case(q, 32, 9, 8, 33, 2 * k, true, true, true, LARCH_F16, true);
    // Prefills: enough pairs per expert that Metal groups them by expert.
    mmvq_id_case(q, 4, 19, 2, 67, k, false, false, true, LARCH_F32);
    mmvq_id_case(q, 4, 19, 3, 33, k, true, true, true, LARCH_F16);
    // From 33 rows CUDA groups the pairs by expert (quant.cu's MOE_GROUPED_ROWS): every format, padded.
    mmvq_id_case(q, 4, 40, 3, 67, k, false, false, true, LARCH_F16, true);
    mmvq_id_case(q, 4, 40, 3, 33, k, true, true, true, LARCH_F32, true);
  }
  // A weighted prefill whose f32 partials pass Metal's 64 MiB a call (300 rows x 4 x 16384 x 4 bytes): grouped in
  // blocks of rows.
  mmvq_id_case(LARCH_Q8_0, 4, 300, 4, 16384, 32, true, true, false, LARCH_F32);
  // qwen3moe's shapes: 128 experts of [768, 2048] (gate, up) and [2048, 768] (down), 8 used, at decode, a small
  // prefill and one Metal groups by expert.
  for (const int64_t rows : {1, 7, 64}) {
    mmvq_id_case(LARCH_Q4_K, 128, rows, 8, 768, 2048, false, false, false, LARCH_F16);
    mmvq_id_case(LARCH_Q6_K, 128, rows, 8, 2048, 768, true, true, false, LARCH_F32);
  }
  // The hybrid MoE target's experts (Q3_K) and the same shapes in Q4_K and in Q2_K (larch repack's): 256 of [512, 2048]
  // (gate, up) and [2048, 512] (down), 8 used, at decode, a verify width, either side of CUDA's grouped threshold
  // (MOE_GROUPED_ROWS 33), a prefill chunk and pp 512, each projection's weight shared by its cases (Q3_K 115 MB each,
  // Q4_K 151 MB, Q2_K 88 MB).
  for (const larch_dtype q : {LARCH_Q3_K, LARCH_Q4_K, LARCH_Q2_K}) {
    rng wg = seeded(std::string("mmvq_id ") + DTYPE[q] + " hybrid experts");
    const auto up = std::make_shared<const bytes>(quant(q, 256 * 512, 2048, wg));
    const auto down = std::make_shared<const bytes>(quant(q, 256 * 2048, 512, wg));
    for (const int64_t rows : {1, 8, 32, 33, 64, 512}) {
      mmvq_id_case(q, 256, rows, 8, 512, 2048, false, false, false, LARCH_F16, false, up);
      mmvq_id_case(q, 256, rows, 8, 2048, 512, true, true, false, LARCH_F32, false, down);
    }
  }
  // 64 or more pairs an expert: Metal's 64 x 32 Q3_K tile.
  mmvq_id_case(LARCH_Q3_K, 4, 130, 2, 67, 512, false, false, true, LARCH_F16);
  mmvq_id_case(LARCH_Q3_K, 4, 130, 2, 33, 512, true, true, true, LARCH_F32);
  // ... and its register prefetch: two pair tiles an expert, the second partial, over 12 k steps; Q6_K's 64 x 32.
  mmvq_id_case(LARCH_Q3_K, 4, 200, 2, 67, 768, false, false, false, LARCH_F16);
  mmvq_id_case(LARCH_Q6_K, 4, 130, 2, 67, 512, true, true, false, LARCH_F32);
  // More than 32 pairs an expert on Metal's 32 x 64 tile, which reads the x rows gathered in expert order: an
  // expert's pairs span tiles, and k 96 is not a whole number of its 64-element steps.
  mmvq_id_case(LARCH_Q4_K, 4, 130, 2, 67, 512, false, false, true, LARCH_F16, true);
  mmvq_id_case(LARCH_Q3_K, 4, 100, 2, 67, 512, false, false, true, LARCH_F16);
  mmvq_id_case(LARCH_Q8_0, 4, 130, 2, 33, 96, false, false, false, LARCH_F32);
  mmvq_id_case(LARCH_Q8_0, 4, 130, 2, 33, 96, true, true, true, LARCH_F16);
  rng g0 = seeded("mmvq_id 0 rows");
  add_test({"mmvq_id q4_k, 0 rows",
            {in(LARCH_Q4_K, {6, 256}, quant(LARCH_Q4_K, 6, 256, g0)), in(LARCH_F16, {0, 256}, {}),
             in(LARCH_I32, {0, 2}, {}), out("mmvq_id", LARCH_F16, {0, 2, 3})},
            [](larch_stream* s, const larch_tensor* t) {
              return larch_mmvq_id(s, &t[0], &t[1], &t[2], nullptr, nullptr, &t[3]);
            }});
  rng g = seeded("mmvq_id invalid");
  const bytes w = quant(LARCH_Q8_0, 4, 64, g);
  invalid("mmvq_id, out [rows, n] without weights",
          {in(LARCH_Q8_0, {4, 64}, w), in(LARCH_F16, {1, 64}, halves(64, g)),
           in(LARCH_I32, {1, 2}, raw(std::vector<int32_t>{0, 1})), poison(LARCH_F16, {1, 2})},
          [](larch_stream* s, const larch_tensor* t) {
            return larch_mmvq_id(s, &t[0], &t[1], &t[2], nullptr, nullptr, &t[3]);
          });
}

// One moe_route case: rows of `experts` logits, the top k. ties: every row holds equal logits, so the lower ids
// must win. masked: every other logit is -inf (a masked expert), so a k past the finite half picks -inf ties.
// equal: every logit 0.
static void moe_route_case(int64_t rows, int64_t experts, int64_t k, float scale, bool ties = false,
                           bool masked = false, bool equal = false) {
  const std::string name = "moe_route " + std::to_string(rows) + " rows of " + std::to_string(experts) + ", top " +
                           std::to_string(k) + ", scale " + std::to_string(scale) + (ties ? ", ties" : "") +
                           (masked ? ", masked" : "") + (equal ? ", all equal" : "");
  rng g = seeded(name);
  bytes lg = floats(size_t(rows * experts), g, -4, 4);
  if (ties)
    for (size_t i = 0; i < lg.size(); i += 8) std::memcpy(&lg[i + 4], &lg[i], 4);  // pairs of equal logits
  const float inf = -std::numeric_limits<float>::infinity(), zero = 0;
  for (size_t i = 0; i < lg.size(); i += 4) {
    if (masked && i / 4 % 2) std::memcpy(&lg[i], &inf, 4);
    if (equal) std::memcpy(&lg[i], &zero, 4);
  }
  add_test({name,
            {in(LARCH_F32, {rows, experts}, lg), out("moe_route", LARCH_I32, {rows, k}),
             out("moe_route", LARCH_F32, {rows, k})},
            [scale](larch_stream* s, const larch_tensor* t) {
              return larch_moe_route(s, &t[0], scale, &t[1], &t[2]);
            }});
}

static void sigmoid_mul() {
  const auto run = [](larch_stream* s, const larch_tensor* t) { return larch_sigmoid_mul(s, &t[0], &t[1], &t[2]); };
  // Qwen 3.5's attention output gate (per element) and shared-expert gate (one per row).
  for (const auto& [rows, d, per_row] : {std::tuple<int64_t, int64_t, bool>{1, 4096, false}, {7, 4096, false},
                                          {5, 2048, true}, {33, 7, true}, {0, 8, false}}) {
    const std::string name =
        "sigmoid_mul [" + std::to_string(rows) + ", " + std::to_string(d) + "]" + (per_row ? ", a gate per row" : "");
    rng g = seeded(name);
    const int64_t gc = per_row ? 1 : d;
    add_test({name,
              {in(LARCH_F16, {rows, d}, halves(size_t(rows * d), g)), in(LARCH_F16, {rows, gc}, halves(size_t(rows * gc), g, 5)),
               out("sigmoid_mul", LARCH_F16, {rows, d})},
              run});
  }
  rng g = seeded("sigmoid_mul in place");
  add_test({"sigmoid_mul [5, 64], out == x",
            {out("sigmoid_mul", LARCH_F16, {5, 64}, halves(320, g)), in(LARCH_F16, {5, 64}, halves(320, g, 5))},
            [](larch_stream* s, const larch_tensor* t) { return larch_sigmoid_mul(s, &t[0], &t[1], &t[0]); }});
  for (const bool per_row : {false, true}) {
    const std::string name = std::string("sigmoid_mul [7, 64], ") + (per_row ? "a gate per row, " : "") + "padded rows";
    rng gp = seeded(name);
    const int64_t gc = per_row ? 1 : 64;
    std::vector<arg> args = {in(LARCH_F16, {7, 64}, halves(448, gp)), in(LARCH_F16, {7, gc}, halves(size_t(7 * gc), gp, 5)),
                             out("sigmoid_mul", LARCH_F16, {7, 64})};
    for (arg& a : args) pad_rows(a, a.out ? 3 : 1);
    add_test({name, args, run});
  }
  invalid("sigmoid_mul, gate [rows, 2]",
          {in(LARCH_F16, {2, 8}, halves(16, g)), in(LARCH_F16, {2, 2}, halves(4, g)), poison(LARCH_F16, {2, 8})},
          run);
}

// DFlash2's dynamic conv at a drafter's width and group (16 channels): stage 0 into f16, stage 1 added into the f32
// residual, a row alone and a block of 8.
static void draft_conv() {
  const auto run = [](uint32_t stage) {
    return [stage](larch_stream* s, const larch_tensor* t) {
      return larch_draft_conv(s, &t[0], &t[1], &t[2], stage, &t[3]);
    };
  };
  const auto args = [](rng& g, int64_t rows, int64_t d, int64_t groups, larch_dtype ot) {
    const size_t n = size_t(rows * d);
    return std::vector<arg>{in(LARCH_F16, {rows, d}, halves(n, g)),
                            in(LARCH_F32, {rows, 4, groups}, floats(size_t(rows * 4 * groups), g, -0.5f, 0.5f)),
                            in(LARCH_F32, {4, d}, floats(size_t(4 * d), g, -1, 1)),
                            out("draft_conv", ot, {rows, d}, ot == LARCH_F32 ? floats(n, g, -8, 8) : bytes{})};
  };
  for (const int64_t rows : {1, 8})
    for (const uint32_t stage : {0u, 1u}) {
      const std::string name = "draft_conv [" + std::to_string(rows) + ", 2048], stage " + std::to_string(stage);
      rng g = seeded(name);
      add_test({name, args(g, rows, 2048, 128, stage ? LARCH_F32 : LARCH_F16), run(stage)});
    }
  rng g = seeded("draft_conv padded");
  std::vector<arg> padded = args(g, 8, 64, 4, LARCH_F32);
  for (arg& a : padded) pad_rows(a, a.out ? 3 : 1);
  add_test({"draft_conv [8, 64], stage 1, padded rows", padded, run(1)});
  invalid("draft_conv, stage 2", args(g, 2, 64, 4, LARCH_F16), run(2));
  invalid("draft_conv, groups not dividing d", args(g, 2, 64, 3, LARCH_F16), run(0));
}

static void layernorm() {
  // InternViT's widths: 1024 per layer, 4096 for the projector's norm over the pixel-shuffled rows.
  for (const auto& [rows, d, bias] : {std::tuple<int64_t, int64_t, bool>{1, 1024, true}, {7, 4096, true},
                                       {1025, 1024, true}, {3, 1000, false}, {5, 7, true}, {0, 8, true}}) {
    const std::string name = "layernorm [" + std::to_string(rows) + ", " + std::to_string(d) + "]" + (bias ? "" : ", no bias");
    rng g = seeded(name);
    // An offset mean (a ViT residual's channels sit far from 0), so the centering matters.
    add_test({name,
              {in(LARCH_F32, {rows, d}, floats(size_t(rows * d), g, 2.0f, 4.0f)), in(LARCH_F32, {d}, floats(size_t(d), g, -1, 1)),
               bias ? in(LARCH_F32, {d}, floats(size_t(d), g, -1, 1)) : poison(LARCH_F32, {d}), out("layernorm", LARCH_F16, {rows, d})},
              [bias](larch_stream* s, const larch_tensor* t) {
                return larch_layernorm(s, &t[0], &t[1], bias ? &t[2] : nullptr, 1e-6f, &t[3]);
              }});
  }
  rng g = seeded("layernorm padded rows");
  std::vector<arg> args = {in(LARCH_F32, {6, 1024}, floats(6 * 1024, g, -4, 4)), in(LARCH_F32, {1024}, floats(1024, g, -1, 1)),
                           in(LARCH_F32, {1024}, floats(1024, g, -1, 1)), out("layernorm", LARCH_F16, {6, 1024})};
  pad_rows(args[0], 4);
  pad_rows(args[3], 2);
  add_test({"layernorm [6, 1024], padded rows", args, [](larch_stream* s, const larch_tensor* t) {
              return larch_layernorm(s, &t[0], &t[1], &t[2], 1e-5f, &t[3]);
            }});
  const auto run = [](larch_stream* s, const larch_tensor* t) { return larch_layernorm(s, &t[0], &t[1], &t[2], 1e-6f, &t[3]); };
  invalid("layernorm, b [d + 1]",
          {in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)), in(LARCH_F32, {8}, floats(8, g, -1, 1)), in(LARCH_F32, {9}, floats(9, g, -1, 1)),
           poison(LARCH_F16, {2, 8})},
          run);
  invalid("layernorm, f32 out",
          {in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)), in(LARCH_F32, {8}, floats(8, g, -1, 1)), in(LARCH_F32, {8}, floats(8, g, -1, 1)),
           poison(LARCH_F32, {2, 8})},
          run);
}

static void attn_full(const larch_device_props& props) {
  const auto run = [](uint32_t seg) {
    return [seg](larch_stream* s, const larch_tensor* t) {
      return larch_attn_full(s, &t[0], &t[1], &t[2], seg, 0.125f, &t[3]);
    };
  };
  // InternViT's tile (1024 patches and CLS, 16 heads of 64), two segments that end inside a
  // 64-row block, and short ones; head_dim 72 (Qwen 3.5 35B's encoder) across a 32-key block.
  for (const auto& [rows, heads, hd, seg] :
       {std::tuple<int64_t, int64_t, int64_t, uint32_t>{1025, 16, 64, 1025}, {130, 2, 64, 65}, {3, 1, 64, 3},
        {64, 1, 64, 1}, {0, 1, 64, 1}, {100, 2, 72, 50}, {40, 3, 72, 40}}) {
    if (hd == 72 && (props.backend == LARCH_BACKEND_CUDA || props.backend == LARCH_BACKEND_HIP)) {
      std::printf("ops: attn_full head_dim 72 not on this backend (%s): NOT RUN\n", props.name);
      continue;
    }
    const std::string name = "attn_full [" + std::to_string(rows) + ", " + std::to_string(heads) + ", " +
                             std::to_string(hd) + "], segments of " + std::to_string(seg);
    rng g = seeded(name);
    const size_t n = size_t(rows * heads * hd);
    add_test({name,
              {in(LARCH_F16, {rows, heads, hd}, halves(n, g, 2)), in(LARCH_F16, {rows, heads, hd}, halves(n, g, 2)),
               in(LARCH_F16, {rows, heads, hd}, halves(n, g)), out("attn_full", LARCH_F16, {rows, heads, hd})},
              run(seg)});
  }
  // A first key block of only -Inf scores (every query's dimension 0 is 1, keys 0-31's is -Inf, so a running
  // max that starts at -Inf must not turn into exp(-inf - -inf) = NaN), and V all +Inf (+Inf out, not NaN).
  {
    rng g = seeded("attn_full -Inf block, +Inf V");
    bytes q = halves(33 * 64, g, 2), k = halves(33 * 64, g, 2), inf_v(64 * 2);
    const auto set = [](bytes& b, size_t i, uint16_t h) {
      b[2 * i] = uint8_t(h);
      b[2 * i + 1] = uint8_t(h >> 8);
    };
    for (size_t r = 0; r < 33; ++r) set(q, r * 64, 0x3c00);  // 1
    for (size_t r = 0; r < 32; ++r) set(k, r * 64, 0xfc00);  // -Inf
    add_test({"attn_full [33, 1, 64], keys 0-31 at -Inf",
              {in(LARCH_F16, {33, 1, 64}, q), in(LARCH_F16, {33, 1, 64}, k), in(LARCH_F16, {33, 1, 64}, halves(33 * 64, g)),
               out("attn_full", LARCH_F16, {33, 1, 64})},
              run(33)});
    for (size_t c = 0; c < 64; ++c) set(inf_v, c, 0x7c00);
    add_test({"attn_full [1, 1, 64], V all +Inf",
              {in(LARCH_F16, {1, 1, 64}, halves(64, g, 2)), in(LARCH_F16, {1, 1, 64}, halves(64, g, 2)),
               in(LARCH_F16, {1, 1, 64}, inf_v), out("attn_full", LARCH_F16, {1, 1, 64})},
              run(1)});
  }
  // Non-finite inputs: a NaN in a query makes its (row, head) NaN and nothing else; a +Inf key makes the
  // rows of its segment whose score is +Inf NaN (exp(inf - inf)), on the GPU routes as in the reference.
  {
    rng g = seeded("attn_full non-finite");
    bytes q = halves(512, g, 2), k = halves(512, g, 2);
    const auto set = [](bytes& b, size_t i, uint16_t h) {
      b[2 * i] = uint8_t(h);
      b[2 * i + 1] = uint8_t(h >> 8);
    };
    set(q, 0, 0x7e00);       // row 0's query: NaN
    set(k, 4 * 64, 0x7c00);  // segment 1's first key: +Inf in its first dimension
    add_test({"attn_full [8, 1, 64], NaN and Inf in",
              {in(LARCH_F16, {8, 1, 64}, q), in(LARCH_F16, {8, 1, 64}, k), in(LARCH_F16, {8, 1, 64}, halves(512, g)),
               out("attn_full", LARCH_F16, {8, 1, 64})},
              run(4)});
  }
  // A first tile of -Inf scores weighs 0, not NaN (exp(-Inf - -Inf)), across the tensor-core route's 16-position tile:
  // the 17th key's score is the only finite one, so every row is V's, 1. And a +Inf V stays +Inf through the
  // tensor-core route's two P panels (whose low half would add 0 x Inf).
  {
    const auto same = [](size_t n, uint16_t h) {
      bytes b(2 * n);
      for (size_t i = 0; i < n; ++i) b[2 * i] = uint8_t(h), b[2 * i + 1] = uint8_t(h >> 8);
      return b;
    };
    bytes k = same(17 * 64, 0xfc00);  // -Inf
    for (size_t i = 16 * 64; i < 17 * 64; ++i) k[2 * i] = k[2 * i + 1] = 0;
    add_test({"attn_full [17, 1, 64], a tile of -Inf scores",
              {in(LARCH_F16, {17, 1, 64}, same(17 * 64, 0x3c00)), in(LARCH_F16, {17, 1, 64}, k),
               in(LARCH_F16, {17, 1, 64}, same(17 * 64, 0x3c00)), out("attn_full", LARCH_F16, {17, 1, 64})},
              run(17)});
    add_test({"attn_full [1, 1, 64], +Inf V",
              {in(LARCH_F16, {1, 1, 64}, same(64, 0)), in(LARCH_F16, {1, 1, 64}, same(64, 0)),
               in(LARCH_F16, {1, 1, 64}, same(64, 0x7c00)), out("attn_full", LARCH_F16, {1, 1, 64})},
              run(1)});
  }
  rng g = seeded("attn_full padded rows");
  std::vector<arg> args = {in(LARCH_F16, {6, 2, 64}, halves(768, g, 2)), in(LARCH_F16, {6, 2, 64}, halves(768, g, 2)),
                           in(LARCH_F16, {6, 2, 64}, halves(768, g)), out("attn_full", LARCH_F16, {6, 2, 64})};
  for (arg& a : args) pad_rows(a, a.out ? 64 : 128);
  add_test({"attn_full [6, 2, 64], padded rows", args, run(3)});
  invalid("attn_full, rows not whole segments",
          {in(LARCH_F16, {5, 1, 64}, halves(320, g)), in(LARCH_F16, {5, 1, 64}, halves(320, g)),
           in(LARCH_F16, {5, 1, 64}, halves(320, g)), poison(LARCH_F16, {5, 1, 64})},
          run(2));
  invalid("attn_full, k of another shape",
          {in(LARCH_F16, {4, 1, 64}, halves(256, g)), in(LARCH_F16, {4, 2, 32}, halves(256, g)),
           in(LARCH_F16, {4, 1, 64}, halves(256, g)), poison(LARCH_F16, {4, 1, 64})},
          run(4));
}

static void gelu() {
  const auto run = [](larch_stream* s, const larch_tensor* t) { return larch_gelu(s, &t[0], &t[1]); };
  for (const auto& [rows, d] : {std::pair<int64_t, int64_t>{1, 4096}, {1025, 4096}, {256, 896}, {33, 7}, {0, 8}}) {
    const std::string name = "gelu [" + std::to_string(rows) + ", " + std::to_string(d) + "]";
    rng g = seeded(name);
    add_test({name, {in(LARCH_F16, {rows, d}, halves(size_t(rows * d), g, 3)), out("gelu", LARCH_F16, {rows, d})}, run});
  }
  rng g = seeded("gelu in place");
  add_test({"gelu [5, 64], out == x", {out("gelu", LARCH_F16, {5, 64}, halves(320, g, 3))},
            [](larch_stream* s, const larch_tensor* t) { return larch_gelu(s, &t[0], &t[0]); }});
  std::vector<arg> args = {in(LARCH_F16, {7, 64}, halves(448, g, 3)), out("gelu", LARCH_F16, {7, 64})};
  for (arg& a : args) pad_rows(a, a.out ? 3 : 1);
  add_test({"gelu [7, 64], padded rows", args, run});
  invalid("gelu, out [rows, d + 1]", {in(LARCH_F16, {2, 8}, halves(16, g)), poison(LARCH_F16, {2, 9})}, run);
}

// A Gated DeltaNet layer group: n_slots slots in layer 1 of 2, sequences of q_lens rows in slots `live`, and
// snaps[r] >= 0 leaving the state of row r in that slot. pad: elements after each row of every tensor (3 x pad for
// the output).
struct gdn_case {
  uint32_t k_heads, v_heads, hd, kernel, n_slots;
  std::vector<int32_t> q_lens, live, snaps;
  int64_t pad = 0;
  int64_t rows() const { return std::accumulate(q_lens.begin(), q_lens.end(), int64_t(0)); }
  int64_t channels() const { return (2 * int64_t(k_heads) + v_heads) * hd; }
  larch_gdn_state state(const larch_tensor* t, size_t at) const {
    return {t[at].buf, t[at + 1].buf, n_slots, 2, 1, kernel, k_heads, v_heads, hd};
  }
  std::string name() const {
    std::string n = std::to_string(k_heads) + " k heads, " + std::to_string(v_heads) + " v heads of " +
                    std::to_string(hd) + ", q_lens";
    for (const int32_t l : q_lens) n += " " + std::to_string(l);
    return n + (snaps.empty() ? "" : ", snapshots") + (pad ? ", padded rows" : "");
  }
};

static void gdn() {
  const gdn_case cases[] = {
      {2, 4, 32, 4, 8, {1}, {5}, {}},                    // a decode row
      {2, 4, 32, 4, 8, {1, 1, 1}, {2, 6, 0}, {}},        // a decode batch
      {2, 4, 32, 4, 8, {5, 0, 9}, {3, 1, 7}, {}},        // prefill chunks and an empty sequence
      {1, 2, 64, 2, 8, {6, 3}, {0, 4}, {-1, 6, -1, -1, -1, 2, 5, -1, -1}},  // snapshots
      // Snapshots at a sequence's rows 0 and 1 with 4 taps: a conv snapshot mixes old state rows and new inputs.
      {2, 4, 32, 4, 8, {5, 0, 9}, {3, 1, 7}, {0, 2, -1, -1, -1, 4, 5, -1, -1, -1, -1, -1, -1, 6}},
      {2, 4, 32, 4, 8, {5, 0, 9}, {3, 1, 7}, {}, 1},  // padded rows
      {16, 32, 128, 4, 2, {1}, {1}, {}},                 // Qwen3.5-35B-A3B's layer at decode
      {16, 32, 128, 4, 2, {40}, {0}, {}},                // and at a prefill chunk
      {1, 2, 256, 8, 4, {1, 5}, {2, 0}, {}},             // the limits: head_dim 256 and 8 taps
      {2, 4, 96, 4, 4, {5}, {1}, {}},                    // head_dim 96: 3 of a lane's 8 register columns
  };
  for (const gdn_case& c : cases) {
    const int64_t rows = c.rows(), ch = c.channels(), vh = c.v_heads, hd = c.hd, seqs = int64_t(c.q_lens.size());
    const size_t conv = size_t(2) * c.n_slots * (c.kernel - 1) * size_t(ch), ssm = size_t(2) * c.n_slots * vh * hd * hd;
    const bytes snaps = c.snaps.empty() ? bytes{} : raw(c.snaps);
    // The state buffers as the call reaches them: conv f16 [2][n_slots][kernel - 1][channels], ssm f32
    // [2][n_slots][v_heads][hd][hd]; each op writes its own and must leave the other unchanged.
    {
      const std::string name = "gdn_conv " + c.name();
      rng g = seeded(name);
      std::vector<arg> args = {in(LARCH_F16, {rows, ch}, halves(size_t(rows * ch), g)),
                               in(LARCH_F32, {ch, int64_t(c.kernel)}, floats(size_t(ch) * c.kernel, g, -1, 1)),
                               in(LARCH_I32, {seqs}, raw(c.live)), in(LARCH_I32, {seqs}, raw(c.q_lens)),
                               out("gdn_conv", LARCH_F16, {rows, ch}),
                               pool(halves(conv, g), true, LARCH_F16, "gdn_conv state"),
                               pool(floats(ssm, g, -1, 1), false, LARCH_F32, "gdn state")};
      if (!snaps.empty()) args.push_back(in(LARCH_I32, {rows}, snaps));
      for (arg& a : args)
        if (!a.pool) pad_rows(a, c.pad * (a.out ? 3 : 1));
      add_test({name, args, [c](larch_stream* s, const larch_tensor* t) {
                  const larch_gdn_state st = c.state(t, 5);
                  return larch_gdn_conv(s, &t[0], &t[1], &st, &t[2], &t[3], c.snaps.empty() ? nullptr : &t[7], &t[4]);
                }});
    }
    {
      const std::string name = "gdn " + c.name();
      rng g = seeded(name);
      std::vector<arg> args = {in(LARCH_F16, {rows, ch}, halves(size_t(rows * ch), g)),
                               in(LARCH_F32, {rows, vh}, floats(size_t(rows * vh), g, -3, 3)),
                               in(LARCH_F32, {rows, vh}, floats(size_t(rows * vh), g, -3, 3)),
                               in(LARCH_F32, {vh}, floats(size_t(vh), g, -1, 1)),
                               in(LARCH_F32, {vh}, floats(size_t(vh), g, -2, -0.05f)),
                               in(LARCH_I32, {seqs}, raw(c.live)), in(LARCH_I32, {seqs}, raw(c.q_lens)),
                               out("gdn", LARCH_F16, {rows, vh * hd}),
                               pool(halves(conv, g), false, LARCH_F16, "gdn_conv state"),
                               pool(floats(ssm, g, -0.5f, 0.5f), true, LARCH_F32, "gdn state")};
      if (!snaps.empty()) args.push_back(in(LARCH_I32, {rows}, snaps));
      for (arg& a : args)
        if (!a.pool) pad_rows(a, c.pad * (a.out ? 3 : 1));
      add_test({name, args, [c](larch_stream* s, const larch_tensor* t) {
                  const larch_gdn_state st = c.state(t, 8);
                  return larch_gdn(s, &t[0], &t[1], &t[2], &t[3], &t[4], 1e-6f, &st, &t[5], &t[6],
                                    c.snaps.empty() ? nullptr : &t[10], &t[7]);
                }});
    }
  }
}

static void moe_route() {
  moe_route_case(1, 4, 2, 0);
  moe_route_case(3, 128, 8, 0);  // qwen3moe
  moe_route_case(17, 128, 8, 2.5f);
  moe_route_case(5, 60, 4, 0);    // a partial last lane step
  moe_route_case(2, 512, 10, 0);  // the largest table
  moe_route_case(4, 64, 64, 0);  // every expert
  moe_route_case(6, 128, 8, 0, true);
  moe_route_case(3, 128, 1, 0);
  moe_route_case(2, 33, 33, 0);  // every expert, one past a lane step
  moe_route_case(4, 64, 8, 0, false, true);
  moe_route_case(2, 64, 40, 1.5f, false, true);  // past the finite half: -inf ties to the lower id
  // A large finite scale: scale / sum overflows f32 where weight / sum * scale does not (#252 review).
  moe_route_case(1, 2, 1, std::numeric_limits<float>::max(), false, false, true);
  moe_route_case(3, 128, 8, std::numeric_limits<float>::max());
  add_test({"moe_route 0 rows",
            {in(LARCH_F32, {0, 8}, {}), out("moe_route", LARCH_I32, {0, 2}), out("moe_route", LARCH_F32, {0, 2})},
            [](larch_stream* s, const larch_tensor* t) { return larch_moe_route(s, &t[0], 0, &t[1], &t[2]); }});
  rng g = seeded("moe_route invalid");
  invalid("moe_route, k > experts",
          {in(LARCH_F32, {1, 4}, floats(4, g, -1, 1)), poison(LARCH_I32, {1, 5}), poison(LARCH_F32, {1, 5})},
          [](larch_stream* s, const larch_tensor* t) { return larch_moe_route(s, &t[0], 0, &t[1], &t[2]); });
}

// LARCH_TEST_DEVICE, or GPU 0 when there is one, else -1 (the CPU reference against itself); -2 is the CPU
// device the engine computes on.
// The training entries (larch.h): each against the CPU reference, plus one malformed call each.
static void train() {
  struct mm {
    int64_t m, n, k;
    uint32_t ta, tb;
    float alpha, beta;
  };
  for (const mm& c : {mm{17, 33, 64, 0, 0, 1, 0}, mm{33, 65, 256, 1, 0, 1, 1}, mm{40, 24, 2048, 0, 1, 0.5f, 0.25f},
                      mm{5, 7, 3, 1, 1, 2, 1}, mm{128, 128, 128, 1, 1, 1, 0}, mm{0, 4, 8, 0, 1, 1, 0}}) {
    const std::string name = "gemm_ex [" + std::to_string(c.m) + ", " + std::to_string(c.k) + "] x [" +
                             std::to_string(c.k) + ", " + std::to_string(c.n) + "], trans " + std::to_string(c.ta) +
                             std::to_string(c.tb) + ", beta " + std::to_string(c.beta);
    rng g = seeded(name);
    const std::vector<int64_t> as = c.ta ? std::vector<int64_t>{c.k, c.m} : std::vector<int64_t>{c.m, c.k};
    const std::vector<int64_t> bs = c.tb ? std::vector<int64_t>{c.n, c.k} : std::vector<int64_t>{c.k, c.n};
    add_test({name,
              {in(LARCH_F32, as, floats(size_t(c.m * c.k), g, -2, 2)), in(LARCH_F32, bs, floats(size_t(c.n * c.k), g, -2, 2)),
               out("gemm_ex", LARCH_F32, {c.m, c.n}, c.beta == 0 ? bytes{} : floats(size_t(c.m * c.n), g, -4, 4))},
              [c](larch_stream* s, const larch_tensor* t) {
                return larch_gemm_ex(s, &t[0], &t[1], &t[2], c.ta, c.tb, c.alpha, c.beta);
              }});
  }
  for (const auto& [rows, d] : {std::pair<int64_t, int64_t>{7, 896}, {2, 4100}, {3, 4098}, {0, 64}}) {
    const std::string dims = " [" + std::to_string(rows) + ", " + std::to_string(d) + "]";
    rng g = seeded("rmsnorm_f32" + dims);
    const size_t n = size_t(rows * d);
    const bytes x = floats(n, g, -4, 4), w = floats(size_t(d), g, -2, 2);
    add_test({"rmsnorm_f32" + dims,
              {in(LARCH_F32, {rows, d}, x), in(LARCH_F32, {d}, w), out("rmsnorm_f32", LARCH_F32, {rows, d})},
              [](larch_stream* s, const larch_tensor* t) { return larch_rmsnorm_f32(s, &t[0], &t[1], 1e-6f, &t[2]); }});
    for (const uint32_t acc : {0u, 1u})
      add_test({"rmsnorm_backward" + dims + ", accumulate " + std::to_string(acc),
                {in(LARCH_F32, {rows, d}, x), in(LARCH_F32, {d}, w), in(LARCH_F32, {rows, d}, floats(n, g, -1, 1)),
                 out("rmsnorm_backward", LARCH_F32, {rows, d}, acc ? floats(n, g, -1, 1) : bytes{})},
                [acc](larch_stream* s, const larch_tensor* t) {
                  return larch_rmsnorm_backward(s, &t[0], &t[1], &t[2], 1e-6f, &t[3], acc);
                }});
  }
  // Negative positions are the inverse rotation, the backward pass.
  for (const auto& [neox, ext] : {std::pair<uint32_t, float>{0, 0}, {1, 0}, {0, 1}}) {
    const std::string name = "rope_f32, neox " + std::to_string(neox) + ", ext " + std::to_string(ext);
    rng g = seeded(name);
    const std::vector<int32_t> pos = {0, 5, -5, 4095, -4095, 131071, -131071};
    const larch_rope_params p = {64, neox, 4096, 1e6f, ext != 0 ? 0.25f : 1.0f, ext, 1.0f, 32.0f, 1.0f, nullptr};
    test t = {name,
              {out("rope_f32", LARCH_F32, {7, 4, 128}, floats(7 * 4 * 128, g, -2, 2)), in(LARCH_I32, {7}, raw(pos))},
              [p](larch_stream* s, const larch_tensor* x) { return larch_rope_f32(s, &x[0], &x[1], &p); }};
    t.pos_max = 131071;
    add_test(std::move(t));
  }
  for (const auto& [rows, ff] : {std::pair<int64_t, int64_t>{7, 768}, {16, 3}, {0, 8}}) {
    const std::string dims = " [" + std::to_string(rows) + ", " + std::to_string(ff) + "]";
    rng g = seeded("silu_mul" + dims);
    const size_t n = size_t(rows * ff);
    const bytes gate = floats(n, g, -8, 8), up = floats(n, g, -2, 2);
    add_test({"silu_mul_f32" + dims,
              {in(LARCH_F32, {rows, ff}, gate), in(LARCH_F32, {rows, ff}, up), out("silu_mul_f32", LARCH_F32, {rows, ff})},
              [](larch_stream* s, const larch_tensor* t) { return larch_silu_mul_f32(s, &t[0], &t[1], &t[2]); }});
    add_test({"silu_mul_backward" + dims,
              {in(LARCH_F32, {rows, ff}, gate), in(LARCH_F32, {rows, ff}, up), in(LARCH_F32, {rows, ff}, floats(n, g, -1, 1)),
               out("silu_mul_backward", LARCH_F32, {rows, ff}), out("silu_mul_backward", LARCH_F32, {rows, ff})},
              [](larch_stream* s, const larch_tensor* t) {
                return larch_silu_mul_backward(s, &t[0], &t[1], &t[2], &t[3], &t[4]);
              }});
  }
  // Causal: two heads of [5, 5] stacked, and one window of 300 tokens; plain rows wider than a block.
  for (const auto& [rows, n, causal] : {std::tuple<int64_t, int64_t, uint32_t>{10, 5, 1}, {300, 300, 1}, {3, 1000, 0}, {0, 4, 1}}) {
    const std::string dims = " [" + std::to_string(rows) + ", " + std::to_string(n) + "], causal " + std::to_string(causal);
    rng g = seeded("softmax" + dims);
    const size_t e = size_t(rows * n);
    add_test({"softmax" + dims,
              {in(LARCH_F32, {rows, n}, floats(e, g, -20, 20)), out("softmax", LARCH_F32, {rows, n})},
              [causal](larch_stream* s, const larch_tensor* t) { return larch_softmax(s, &t[0], 0.125f, causal, &t[1]); }});
    // p: any rows that sum to one (their zeros, the masked columns, included).
    std::vector<float> p(e);
    for (int64_t r = 0; r < rows; ++r) {
      float z = 0;
      for (int64_t c = 0; c < n; ++c) z += p[size_t(r * n + c)] = c <= (causal ? r % n : n) ? float(g() % 1000 + 1) : 0;
      for (int64_t c = 0; c < n; ++c) p[size_t(r * n + c)] /= z;
    }
    add_test({"softmax_backward" + dims,
              {in(LARCH_F32, {rows, n}, raw(p)), in(LARCH_F32, {rows, n}, floats(e, g, -1, 1)),
               out("softmax_backward", LARCH_F32, {rows, n})},
              [](larch_stream* s, const larch_tensor* t) { return larch_softmax_backward(s, &t[0], &t[1], 0.125f, &t[2]); }});
  }
  rng g = seeded("train invalid");
  invalid("gemm_ex, k differs",
          {in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)), in(LARCH_F32, {4, 3}, floats(12, g, -1, 1)),
           poison(LARCH_F32, {2, 3})},
          [](larch_stream* s, const larch_tensor* t) { return larch_gemm_ex(s, &t[0], &t[1], &t[2], 0, 0, 1, 0); });
  invalid("gemm_ex, a f16",
          {in(LARCH_F16, {2, 8}, halves(16, g)), in(LARCH_F32, {8, 3}, floats(24, g, -1, 1)), poison(LARCH_F32, {2, 3})},
          [](larch_stream* s, const larch_tensor* t) { return larch_gemm_ex(s, &t[0], &t[1], &t[2], 0, 0, 1, 0); });
  invalid("rmsnorm_f32, out f16",
          {in(LARCH_F32, {2, 64}, floats(128, g, -1, 1)), in(LARCH_F32, {64}, floats(64, g, -1, 1)), poison(LARCH_F16, {2, 64})},
          [](larch_stream* s, const larch_tensor* t) { return larch_rmsnorm_f32(s, &t[0], &t[1], 1e-6f, &t[2]); });
  invalid("rmsnorm_backward, w shorter than d",
          {in(LARCH_F32, {2, 64}, floats(128, g, -1, 1)), in(LARCH_F32, {32}, floats(32, g, -1, 1)),
           in(LARCH_F32, {2, 64}, floats(128, g, -1, 1)), poison(LARCH_F32, {2, 64})},
          [](larch_stream* s, const larch_tensor* t) {
            return larch_rmsnorm_backward(s, &t[0], &t[1], &t[2], 1e-6f, &t[3], 0);
          });
  invalid("rope_f32, qk f16",
          {in(LARCH_F16, {2, 2, 8}, halves(32, g)), in(LARCH_I32, {2}, raw(std::vector<int32_t>{0, -1}))},
          [](larch_stream* s, const larch_tensor* t) {
            const larch_rope_params p = {8, 0, 4096, 1e4f, 1, 0, 1, 32, 1, nullptr};
            return larch_rope_f32(s, &t[0], &t[1], &p);
          });
  invalid("silu_mul_f32, up narrower",
          {in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)), in(LARCH_F32, {2, 4}, floats(8, g, -1, 1)), poison(LARCH_F32, {2, 8})},
          [](larch_stream* s, const larch_tensor* t) { return larch_silu_mul_f32(s, &t[0], &t[1], &t[2]); });
  invalid("silu_mul_backward, dup f16",
          {in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)), in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)),
           in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)), poison(LARCH_F32, {2, 8}), poison(LARCH_F16, {2, 8})},
          [](larch_stream* s, const larch_tensor* t) {
            return larch_silu_mul_backward(s, &t[0], &t[1], &t[2], &t[3], &t[4]);
          });
  invalid("softmax, out narrower",
          {in(LARCH_F32, {2, 8}, floats(16, g, -1, 1)), poison(LARCH_F32, {2, 4})},
          [](larch_stream* s, const larch_tensor* t) { return larch_softmax(s, &t[0], 1, 1, &t[1]); });
  invalid("softmax_backward, dp f16",
          {in(LARCH_F32, {2, 8}, floats(16, g, 0, 1)), in(LARCH_F16, {2, 8}, halves(16, g)), poison(LARCH_F32, {2, 8})},
          [](larch_stream* s, const larch_tensor* t) { return larch_softmax_backward(s, &t[0], &t[1], 1, &t[2]); });
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

// A mixed layer of `experts` [n, k] experts, the ones `cold` lists in w_cold (Q2_K) and the others in w_hot (q):
// the slices of the whole tensors `full` (q) and `full_cold` (Q2_K, [experts * n, k]) at their experts, and the table.
struct mixed_layer {
  bytes hot, cold, table;
  int64_t n_hot, n_cold;
};
static mixed_layer split_layer(larch_dtype q, const bytes& full, const bytes& full_cold, int64_t experts, int64_t n,
                               int64_t k, const std::vector<int32_t>& cold) {
  mixed_layer m{};
  std::vector<int32_t> table(static_cast<size_t>(experts));
  const size_t hb = size_t(nbytes(q, {n, k})), cb = size_t(nbytes(LARCH_Q2_K, {n, k}));
  for (int32_t e = 0; e < experts; ++e)
    if (std::find(cold.begin(), cold.end(), e) != cold.end()) {
      table[size_t(e)] = -1 - int32_t(m.n_cold++);
      m.cold.insert(m.cold.end(), full_cold.begin() + ptrdiff_t(e * cb), full_cold.begin() + ptrdiff_t((e + 1) * cb));
    } else {
      table[size_t(e)] = int32_t(m.n_hot++);
      m.hot.insert(m.hot.end(), full.begin() + ptrdiff_t(e * hb), full.begin() + ptrdiff_t((e + 1) * hb));
    }
  m.table = raw(table);
  return m;
}

// mmvq_id_mixed against the reference: hot experts in q, cold ones in Q2_K, ids over both.
static void mmvq_id_mixed_case(larch_dtype q, int64_t experts, const std::vector<int32_t>& cold, int64_t rows,
                               int64_t slots, int64_t n, int64_t k, bool weighted, larch_dtype ot) {
  const std::string name = std::string("mmvq_id_mixed ") + DTYPE[q] + " + q2_k (" + std::to_string(cold.size()) +
                           " of " + std::to_string(experts) + " cold) x [" + std::to_string(n) + ", " +
                           std::to_string(k) + "], " + std::to_string(rows) + " rows of " + std::to_string(slots) +
                           (weighted ? ", x per slot, weighted" : "") + " -> " + DTYPE[ot];
  rng g = seeded(name);
  const mixed_layer m =
      split_layer(q, quant(q, experts * n, k, g), quant(LARCH_Q2_K, experts * n, k, g), experts, n, k, cold);
  std::uniform_int_distribution<int32_t> pick(0, int32_t(experts - 1));
  std::vector<int32_t> ids(size_t(rows * slots));
  for (int32_t& e : ids) e = pick(g);
  const std::vector<int64_t> xs = weighted ? std::vector<int64_t>{rows, slots, k} : std::vector<int64_t>{rows, k};
  std::vector<arg> args = {in(q, {m.n_hot * n, k}, m.hot), in(LARCH_Q2_K, {m.n_cold * n, k}, m.cold),
                           in(LARCH_I32, {experts}, m.table),
                           in(LARCH_F16, xs, halves(size_t(rows * (weighted ? slots : 1) * k), g)),
                           in(LARCH_I32, {rows, slots}, raw(ids))};
  if (weighted) args.push_back(in(LARCH_F32, {rows, slots}, floats(size_t(rows * slots), g, 0, 1)));
  const std::vector<int64_t> os = weighted ? std::vector<int64_t>{rows, n} : std::vector<int64_t>{rows, slots, n};
  args.push_back(out("mmvq_id", ot, os));
  add_test({name, args, [weighted](larch_stream* s, const larch_tensor* t) {
              return larch_mmvq_id_mixed(s, &t[0], &t[1], &t[2], &t[3], &t[4], weighted ? &t[5] : nullptr,
                                          &t[5 + weighted]);
            }});
}

static void mmvq_id_mixed() {
  // k whole Q2_K blocks, as the cold experts' rows must be.
  for (const larch_dtype q : {LARCH_Q3_K, LARCH_Q4_K, LARCH_Q2_K, LARCH_Q8_0})
    for (const int64_t rows : {1, 3, 40}) {  // one row, a check step's slice, and CUDA's grouped path
      mmvq_id_mixed_case(q, 8, {1, 4, 6}, rows, 3, 67, 768, false, LARCH_F16);
      mmvq_id_mixed_case(q, 8, {0, 7}, rows, 3, 33, 768, true, LARCH_F32);
    }
  // The hybrid MoE target's shapes, a third of 256 experts cold: decode and a prefill chunk.
  std::vector<int32_t> third;
  for (int32_t e = 2; e < 256; e += 3) third.push_back(e);
  for (const int64_t rows : {1, 64}) {
    mmvq_id_mixed_case(LARCH_Q3_K, 256, third, rows, 8, 512, 2048, false, LARCH_F16);
    mmvq_id_mixed_case(LARCH_Q4_K, 256, third, rows, 8, 2048, 512, true, LARCH_F32);
  }
}

// mmvq_id_mixed gives, per (row, slot), the bytes mmvq_id gives on one tensor holding every expert at the dtype
// mixed keeps it in: the hot experts' tensor `full` (q) for a hot pair, the cold ones' `full_cold` (Q2_K) for a cold
// one; weighted, on rows whose slots are all hot or all cold. Run on the CPU reference and on CUDA (larch.h's
// promise); the fast CPU device's mmvq_id has its own sums.
static int mixed_bytes(const side& x) {
  int wrong = 0;
  const int64_t experts = 8, n = 33, k = 512, slots = 3;
  const std::vector<int32_t> cold = {1, 4, 6};
  const auto is_cold = [&](int32_t e) { return std::find(cold.begin(), cold.end(), e) != cold.end(); };
  for (const larch_dtype q : {LARCH_Q3_K, LARCH_Q4_K})
    for (const int64_t rows : {1, 3, 40})
      for (const bool weighted : {false, true}) {
        const std::string name = std::string("mixed bytes ") + DTYPE[q] + ", " + std::to_string(rows) + " rows" +
                                 (weighted ? ", weighted" : "");
        rng g = seeded(name);
        const bytes full = quant(q, experts * n, k, g), full_cold = quant(LARCH_Q2_K, experts * n, k, g);
        const mixed_layer m = split_layer(q, full, full_cold, experts, n, k, cold);
        // Weighted rows alternate all hot and all cold; unweighted pairs mix both.
        std::vector<int32_t> ids(size_t(rows * slots));
        for (int64_t r = 0; r < rows; ++r)
          for (int64_t j = 0; j < slots; ++j) {
            const int32_t hot[] = {0, 2, 3, 5, 7};
            ids[size_t(r * slots + j)] =
                weighted ? (r % 2 ? cold[size_t(g() % 3)] : hot[g() % 5]) : int32_t(g() % uint32_t(experts));
          }
        const std::vector<int64_t> xs = weighted ? std::vector<int64_t>{rows, slots, k} : std::vector<int64_t>{rows, k};
        const arg xa = in(LARCH_F16, xs, halves(size_t(rows * (weighted ? slots : 1) * k), g)),
                  ia = in(LARCH_I32, {rows, slots}, raw(ids)),
                  wa = in(LARCH_F32, {rows, slots}, floats(size_t(rows * slots), g, 0, 1));
        const std::vector<int64_t> os = weighted ? std::vector<int64_t>{rows, n} : std::vector<int64_t>{rows, slots, n};
        const arg oa = out("mmvq_id", LARCH_F32, os);
        std::vector<arg> margs = {in(q, {m.n_hot * n, k}, m.hot), in(LARCH_Q2_K, {m.n_cold * n, k}, m.cold),
                                  in(LARCH_I32, {experts}, m.table), xa, ia};
        if (weighted) margs.push_back(wa);
        margs.push_back(oa);
        const test mixed{name, margs, [weighted](larch_stream* s, const larch_tensor* t) {
                           return larch_mmvq_id_mixed(s, &t[0], &t[1], &t[2], &t[3], &t[4], weighted ? &t[5] : nullptr,
                                                       &t[5 + weighted]);
                         }};
        const auto whole = [&](larch_dtype t, const bytes& w) {
          std::vector<arg> a = {in(t, {experts * n, k}, w), xa, ia};
          if (weighted) a.push_back(wa);
          a.push_back(oa);
          return test{name, a, [weighted](larch_stream* s, const larch_tensor* t) {
                        return larch_mmvq_id(s, &t[0], &t[1], &t[2], nullptr, weighted ? &t[3] : nullptr,
                                              &t[3 + weighted]);
                      }};
        };
        const test as_hot = whole(q, full), as_cold = whole(LARCH_Q2_K, full_cold);
        std::vector<bytes> got, hot, coldb;
        if (run_on(x, mixed, got) || run_on(x, as_hot, hot) || run_on(x, as_cold, coldb)) {
          std::fprintf(stderr, "%s: a call failed: %s\n", name.c_str(), larch_last_error(x.dev));
          ++wrong;
          continue;
        }
        // The output's data, past its guard bytes, in each run.
        const size_t mo = mixed.args.size() - 1, wo = as_hot.args.size() - 1, per = size_t(n) * 4;
        const auto at = [&](const std::vector<bytes>& b, size_t i, const arg& a, size_t unit) {
          return b[i].data() + lead(i, a) + unit * per;
        };
        for (int64_t r = 0; r < rows; ++r)
          for (int64_t j = 0; j < (weighted ? 1 : slots); ++j) {
            const size_t unit = size_t(weighted ? r : r * slots + j);
            const bool c = is_cold(ids[size_t(r * slots + j)]);
            const auto& want = c ? coldb : hot;
            if (std::memcmp(at(got, mo, mixed.args[mo], unit), at(want, wo, as_hot.args[wo], unit), per)) {
              std::fprintf(stderr, "%s: row %lld slot %lld (%s) differs from mmvq_id\n", name.c_str(), (long long)r,
                           (long long)j, c ? "cold" : "hot");
              ++wrong;
            }
          }
      }
  return wrong;
}

// A valid one-block Q2_K dequant answers UNSUPPORTED on this device: no Q2_K kernels (vv_ops::no_weights).
static bool lacks_q2(const side& dut) {
  larch_buffer *w = nullptr, *o = nullptr;
  bool lacks = false;
  if (larch_buffer_alloc(dut.dev, 84, &w) == LARCH_OK && larch_buffer_alloc(dut.dev, 512, &o) == LARCH_OK) {
    const larch_tensor wt = {w, 0, LARCH_Q2_K, 2, {1, 256}, {1, 1}}, ot = {o, 0, LARCH_F16, 2, {1, 256}, {256, 1}};
    lacks = larch_dequant(dut.s, &wt, &ot) == LARCH_ERR_UNSUPPORTED;
    larch_stream_sync(dut.s);
  }
  larch_buffer_free(w);
  larch_buffer_free(o);
  return lacks;
}

int main() {
  int32_t gpus = 0, ordinal = 0;
  if (larch_device_count(&gpus) != LARCH_OK || !pick(gpus, &ordinal)) return 1;
  side ref{}, dut{};
  if (larch_device_open(-1, &ref.dev) != LARCH_OK || larch_stream_create(ref.dev, &ref.s) != LARCH_OK ||
      larch_device_open(ordinal, &dut.dev) != LARCH_OK || larch_stream_create(dut.dev, &dut.s) != LARCH_OK) {
    std::fprintf(stderr, "device %d: open failed: %s\n", ordinal, larch_last_error(dut.dev));
    return 1;
  }
  larch_device_props props{};
  larch_device_props_get(dut.dev, &props);
  std::printf("ops: device %d (%s) against the CPU reference\n", ordinal, props.name);

  embed();
  rmsnorm();
  rope();
  rope_multi();
  qk_norm();
  mmvq();
  // HIP has no entry; Metal has it with the staged tile, Apple GPU family 10 (MTLGPUFamilyApple10, arch_code 1010).
  if (props.backend == LARCH_BACKEND_CUDA || props.backend == LARCH_BACKEND_CPU ||
      (props.backend == LARCH_BACKEND_METAL && props.arch_code >= 1010))
    mmvq_verify();
  else
    std::printf("ops: mmvq_verify not on this backend (%s): NOT RUN\n", props.name);
  dequant();
  gemm();
  silu_mul();
  add();
  kv_write();
  attention();
  sample_prep();
  mmvq_id();
  mmvq_id_mixed();
  moe_route();
  layernorm();
  gelu();
  attn_full(props);
  sigmoid_mul();
  draft_conv();
  if (props.backend != LARCH_BACKEND_HIP)  // HIP has no entry
    gdn();
  else
    std::printf("ops: gdn not on this backend (%s): NOT RUN\n", props.name);
  if (props.backend == LARCH_BACKEND_CUDA || props.backend == LARCH_BACKEND_CPU)  // HIP and Metal have no entry
    train();
  else
    std::printf("ops: training ops not on this backend (%s): NOT RUN\n", props.name);
  // A GPU backend may lack Q3_K kernels (vv_ops::no_weights) and refuse its valid calls UNSUPPORTED; those
  // cases are dropped and counted as not run. Malformed Q3_K calls still must be INVALID_ARG. CUDA has them, so a
  // refusal there is a failure, not a skip.
  if (lacks_kernels(dut.dev, dut.s, LARCH_Q3_K)) {
    if (props.backend == LARCH_BACKEND_CUDA) return std::fputs("ops: CUDA refused a valid q3_k mmvq\n", stderr), 1;
    const size_t all = tests.size();
    std::erase_if(tests, [](const test& c) {
      return c.want == LARCH_OK && std::any_of(c.args.begin(), c.args.end(), [](const arg& a) { return a.t == LARCH_Q3_K; });
    });
    std::printf("ops: no q3_k on this device: %zu cases NOT RUN\n", all - tests.size());
  }
  // Metal and HIP have no Q2_K kernels and refuse its valid calls UNSUPPORTED: dropped likewise. CUDA has the ops
  // these cases call (embed, dequant, mmvq_id), so a refusal there is a failure.
  if (lacks_q2(dut)) {
    if (props.backend == LARCH_BACKEND_CUDA) return std::fputs("ops: CUDA refused a valid q2_k dequant\n", stderr), 1;
    const size_t all = tests.size();
    std::erase_if(tests, [](const test& c) {
      return c.want == LARCH_OK && std::any_of(c.args.begin(), c.args.end(), [](const arg& a) { return a.t == LARCH_Q2_K; });
    });
    std::printf("ops: no q2_k on this device: %zu cases NOT RUN\n", all - tests.size());
  }
  int wrong = 0;
  for (const test& c : tests) wrong += check(ref, dut, c);
  wrong += mixed_bytes(ref);
  if (props.backend == LARCH_BACKEND_CUDA)
    wrong += mixed_bytes(dut);
  else
    std::printf("ops: mixed bytes against mmvq_id not run on this device (%s): NOT RUN\n", props.name);
  // Row equality is CUDA's (and wave32 HIP's) gemm_rows_k promise and the CPU's; wave64 HIP stays on hipBLAS. Metal
  // has its own promise, its GEMV up to the same 8 rows, not run here.
  if (props.backend == LARCH_BACKEND_CPU ||
      ((props.backend == LARCH_BACKEND_CUDA || props.backend == LARCH_BACKEND_HIP) && props.warp_size == 32))
    wrong += gemm_rows(dut);
  else
    std::printf("ops: gemm rows not run on this backend (%s): NOT RUN\n", props.name);

  larch_stream_destroy(dut.s);
  larch_device_close(dut.dev);
  larch_stream_destroy(ref.s);
  larch_device_close(ref.dev);
  std::printf("ops: %zu cases, %d wrong\n", tests.size(), wrong);
  return wrong != 0;
}
