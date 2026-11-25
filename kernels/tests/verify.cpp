// The verify gate (v20-35): on GPU 0, row r of larch_mmvq_verify is byte-equal to larch_mmvq of x's row r alone, for
// every quant format, the decode shapes of the three target models (m1, m5, m8) and contract.cpp's, every row count
// 1-32 and both output dtypes; then the same on shifted and padded x and w, and a captured call against the eager one.
// Where cuda::mmvq routes rows to mmvq_verify's kernel (sm_120 only), larch_mmvq must give larch_mmvq_verify's bytes.
// Exact speculation rests on it, so any unequal row, or a call that fails, fails the test. On Metal (Apple GPU family
// 10, the staged tile) the rows are byte-equal to larch_mmvq_verify of each row alone instead (larch.h), and Metal
// captures no graphs. GPU 0 absent, neither CUDA nor Metal, or a Metal GPU without the tile: NOT RUN, 77. A format
// it has no kernels for (UNSUPPORTED): NOT RUN. First, one-row larch_mmvq's bytes are pinned per format on sm_89; on
// Metal, last, mmvq_id's rows in drafter mode (drafter_experts).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "../src/common/arch.h"
#include "testdata.h"

static larch_device* dev;
static larch_stream* stream;
static rng g(35);
static std::string where;  // the case a failure message names
static int failed = 0;
static larch_device_props props{};

// st must be OK: else the failure is printed with the case and counted. Returns st == OK.
static bool ok(larch_status st, const std::string& what) {
  if (st == LARCH_OK) return true;
  std::fprintf(stderr, "verify: %s: %s returned %d: %s\n", where.c_str(), what.c_str(), int(st),
               larch_last_error(dev));
  ++failed;
  return false;
}
// A setup step that cannot be skipped: ends the test when it fails.
static void need(larch_status st, const char* what) {
  if (!ok(st, what)) std::exit(1);
}

// A device buffer of `size` bytes (none for 0) holding src, whose size - at bytes go to byte `at`; freed with it.
struct dbuf {
  larch_buffer* b = nullptr;
  dbuf(uint64_t size, const void* src = nullptr, uint64_t at = 0) {
    if (!size) return;
    need(larch_buffer_alloc(dev, size, &b), "buffer_alloc");
    if (src) need(larch_upload(stream, b, at, src, size - at), "upload");
  }
  dbuf(const dbuf&) = delete;
  dbuf& operator=(const dbuf&) = delete;
  ~dbuf() { larch_buffer_free(b); }
};

// A [rows, cols] view of b at byte offset off; stride0 elements (blocks for quant) between rows, dense when 0.
static larch_tensor view(larch_buffer* b, uint64_t off, larch_dtype t, int64_t rows, int64_t cols,
                          int64_t stride0 = 0) {
  return {b, off, t, 2, {rows, cols}, {stride0 ? stride0 : cols / block_of(t), 1}};
}
// The first `size` bytes of b, once the stream has caught up.
static bytes fetch(larch_buffer* b, uint64_t size) {
  bytes v(static_cast<size_t>(size));
  need(larch_download(stream, b, 0, v.data(), size), "download");
  need(larch_stream_sync(stream), "sync");
  return v;
}

// ALIGN: also run VARIANTS; LM_HEAD: f32 out only; ROUTE: larch_mmvq at ROUTED rows too
enum : unsigned { BIAS = 1, LM_HEAD = 2, ALIGN = 4, CAPTURE = 8, ROUTE = 16 };
struct shape {
  const char* name;
  int64_t n, k;
  unsigned flags;
};
static const shape SHAPES[] = {
    {"m1 q", 4096, 4096, ROUTE},
    {"m1 k", 1024, 4096, CAPTURE},
    {"m1 gate", 14336, 4096, 0},
    {"m1 down", 4096, 14336, 0},
    {"m1 lm_head", 128256, 4096, LM_HEAD},
    {"m5 q", 3584, 3584, BIAS},
    {"m5 k", 512, 3584, BIAS | ALIGN | ROUTE},
    {"m5 gate", 18944, 3584, 0},
    {"m5 down", 3584, 18944, 0},
    {"m5 lm_head", 152064, 3584, LM_HEAD},
    {"m8 q", 896, 896, BIAS},
    {"m8 k", 128, 896, BIAS},
    {"m8 gate", 4864, 896, 0},
    {"m8 down", 896, 4864, 0},
    {"m8 lm_head", 151936, 896, LM_HEAD},
    {"mix", 67, 2304, BIAS | ALIGN | CAPTURE | ROUTE},
    {"mix", 67, 4608, 0},
    {"mix", 8, 256, 0},  // contract.cpp's
    {"35B up", 512, 2048, 0},  // the 35B's shared-expert gate/up and down (Q3_K_M)
    {"35B down", 2048, 512, 0},
};

// What ALIGN shapes also run: x or w 2 bytes off 16-byte alignment (Q4_K's and Q5_K's aligned loads off), x rows
// padded by 8 elements, each with the bias on and off; bias -1 is the shape's own, and the first entry is the plain
// run.
struct variant {
  uint64_t xoff;
  int64_t xpad;
  uint64_t woff;
  int bias;
};
static const variant VARIANTS[] = {{0, 0, 0, -1}, {0, 0, 0, 0}, {2, 0, 0, 1}, {2, 0, 0, 0},
                                   {0, 8, 0, 1},  {0, 8, 0, 0}, {0, 0, 2, 1}, {0, 0, 2, 0}};

// The rows cuda::mmvq runs on mmvq_verify's kernel on sm_120 (quant.cu's MMVQ_ROUTE), per format from Q4_K on in
// dtype order, then Q3_K's (every row); elsewhere it keeps its own kernel, whose rows may differ.
static const std::vector<int64_t> ROUTED[] = {
    {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14},                // q4_k
    {2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16},            // q5_k
    {2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16},              // q6_k
    {2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16},       // q8_0
    {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},            // q4_0
    {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14},            // q5_0
    {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, // q3_k
};

static int64_t cap_eq = 0, cap_total = 0;      // captured outputs byte-equal to the eager call's
static int64_t route_eq = 0, route_total = 0;  // routed larch_mmvq outputs byte-equal to larch_mmvq_verify's
static int routed_up_to = 1;                   // the most rows GPU 0's arch routes (arch::mmvq_rows_up_to)
static auto one_row = larch_mmvq;             // what a row alone must give: Metal's own mmvq_verify (larch.h)

// One (format, shape, variant, out dtype): the 32 one-row mmvq rows against larch_mmvq_verify at every row count,
// larch_mmvq at the format's ROUTED rows against the verify's output when the shape has ROUTE, and, when capture is
// set, the verify at rows 4 and 24 (two launches) run in a graph against its eager output. w is the shape's first n
// rows of wb, a weight of wh's bytes; xh holds x's 32 rows with 8 elements of padding each. Adds the rows byte-equal
// to eq; false once a call failed.
static bool gate(larch_dtype q, const shape& c, const variant& v, larch_dtype od, bool capture, larch_buffer* wb,
                 const bytes& wh, const bytes& xh, larch_buffer* bias_buf, int64_t& eq) {
  const int64_t n = c.n, k = c.k, pad = v.xpad, row = n * (od == LARCH_F32 ? 4 : 2);
  const bool bias = v.bias < 0 ? (c.flags & BIAS) != 0 : v.bias != 0;
  where = std::string(DTYPE[q]) + " " + c.name + " " + std::to_string(n) + "x" + std::to_string(k) + " -> " +
          DTYPE[od] + (bias ? " + bias" : "") + (v.xoff ? ", x at +2" : "") + (pad ? ", x padded" : "") +
          (v.woff ? ", w at +2" : "");
  const dbuf w2(v.woff ? nbytes(q, {n, k}) + v.woff : 0, wh.data(), v.woff), xb(xh.size() + v.xoff, xh.data(), v.xoff);
  const bytes poison(size_t(32 * row), 0xFF);  // an output row nobody wrote shows
  const dbuf one(poison.size(), poison.data()), many(poison.size(), poison.data());
  const larch_tensor w = view(v.woff ? w2.b : wb, v.woff, q, n, k), b = {bias_buf, 0, LARCH_F32, 1, {n}, {1}};
  const larch_tensor* bp = bias ? &b : nullptr;
  const bool route = c.flags & ROUTE;
  const std::vector<int64_t>& route_rows = ROUTED[q == LARCH_Q3_K ? 6 : q - LARCH_Q4_K];
  const auto x = [&](int64_t first, int64_t rows) {
    return view(xb.b, v.xoff + 2 * uint64_t(first * (k + pad)), LARCH_F16, rows, k, k + pad);
  };
  const auto out = [&](larch_buffer* o, int64_t first, int64_t rows) {
    return view(o, uint64_t(first * row), od, rows, n);
  };
  for (int64_t r = 0; r < 32; ++r) {
    const larch_tensor xr = x(r, 1), o = out(one.b, r, 1);
    need(one_row(stream, &w, &xr, bp, &o), "one-row mmvq");
  }
  const bytes want = fetch(one.b, poison.size());
  for (int64_t rows = 1; rows <= 32; ++rows) {
    need(larch_upload(stream, many.b, 0, poison.data(), poison.size()), "upload");
    const larch_tensor xn = x(0, rows), o = out(many.b, 0, rows);
    if (!ok(larch_mmvq_verify(stream, &w, &xn, bp, &o), "mmvq_verify, " + std::to_string(rows) + " rows"))
      return false;
    const bytes got = fetch(many.b, uint64_t(rows * row));
    for (int64_t r = 0; r < rows; ++r)
      eq += std::memcmp(&got[size_t(r * row)], &want[size_t(r * row)], size_t(row)) == 0;
    if (route && rows <= routed_up_to && std::ranges::find(route_rows, rows) != route_rows.end()) {
      const larch_tensor routed = out(one.b, 0, rows);
      need(larch_upload(stream, one.b, 0, poison.data(), poison.size()), "upload");
      if (!ok(larch_mmvq(stream, &w, &xn, bp, &routed), "mmvq, " + std::to_string(rows) + " rows")) return false;
      const bool same = fetch(one.b, uint64_t(rows * row)) == got;
      if (!same) std::fprintf(stderr, "verify: %s: mmvq at %lld rows != mmvq_verify\n", where.c_str(), (long long)rows);
      ++route_total;
      route_eq += same;
    }
  }
  if (capture)
    for (const int64_t rows : {4, 24}) {
      const larch_tensor xn = x(0, rows), o = out(many.b, 0, rows);
      const std::string what = "captured mmvq_verify, " + std::to_string(rows) + " rows";
      need(larch_upload(stream, many.b, 0, poison.data(), poison.size()), "upload");
      if (!ok(larch_mmvq_verify(stream, &w, &xn, bp, &o), "eager mmvq_verify, " + std::to_string(rows) + " rows"))
        return false;
      const bytes eager = fetch(many.b, uint64_t(rows * row));
      need(larch_upload(stream, many.b, 0, poison.data(), poison.size()), "upload");
      larch_graph* graph = nullptr;
      larch_status st = larch_graph_begin(stream);
      if (st == LARCH_ERR_UNSUPPORTED && props.backend == LARCH_BACKEND_METAL) continue;  // no graphs
      if (st == LARCH_OK) {
        st = larch_mmvq_verify(stream, &w, &xn, bp, &o);
        const larch_status ended = larch_graph_end(stream, &graph);  // ends the capture even after a failed op
        if (st == LARCH_OK) st = ended;
      }
      if (st == LARCH_OK) st = larch_graph_launch(graph, stream);
      if (st == LARCH_OK) st = larch_stream_sync(stream);  // the exec must not be destroyed while it runs
      larch_graph_destroy(graph);
      if (!ok(st, what)) return false;
      ++cap_total;
      cap_eq += fetch(many.b, uint64_t(rows * row)) == eager;
    }
  return true;
}

// One-row mmvq bytes on sm_89 with 24 SMs, per CUDA format: FNV-1a of case A then B. hash: d605acb's f32 decoder
// (LARCH_DECODE_H2=0); h2: b5090142's half2 decoder, the default route (0: no half2 decoder). Expected hash 0: not
// pinned yet, the row prints what it got and fails nothing. New rows go last: every row draws from one rng.
// ponytail: Q3_K unpinned, so a rewrite that moves its bytes stays green; copy both hashes from the first sm_89
// 24-SM run's verify and verify_f32 `pinned q3_k` lines.
static const struct {
  larch_dtype q;
  uint64_t hash, h2;
} PINNED[] = {{LARCH_Q4_K, 0x332b237db21229cb, 0xa649e1b6b1b1a064}, {LARCH_Q5_K, 0x5ef84d428e9fd6f3, 0},
              {LARCH_Q6_K, 0x00bcaabfeec14d38, 0xa3a457a8e870e841}, {LARCH_Q8_0, 0x29d2da5daccdfbaa, 0},
              {LARCH_Q4_0, 0xeab42025f3dbff53, 0},                  {LARCH_Q5_0, 0x7ef2c12c95676d7c, 0},
              {LARCH_Q3_K, 0, 0}};

// larch_mmvq at one row must keep its bytes across kernel rewrites. A: 4096x4096, f32, no bias, x aligned.
// B: 128x4096, f16, bias, w and x 2 bytes off 16-byte alignment (the K quants' raised KS, and the unaligned loops).
// Pins hold on sm_89 with 24 SMs only (the k-split of B follows the SM count); elsewhere the hashes print for
// capturing.
static void pinned() {
  const bool sm89 = props.arch_code == 89 && props.compute_units == 24;
  const char* env = std::getenv("LARCH_DECODE_H2");
  const bool h2 = arch::mmvq_h2(int(props.arch_code), env && !std::strcmp(env, "0") ? 0 : -1);
  rng p(89);
  int64_t eq = 0;
  for (const auto& [q, f32, h2_hash] : PINNED) {
    const uint64_t want = h2 && h2_hash ? h2_hash : f32;
    const bytes wh = quant(q, 4096, 4096, p), xh = halves(2 * (4096 + 8), p);
    bytes bh(4 * 128);  // f32 in [-1, 1)
    for (int j = 0; j < 128; ++j) {
      const float f = float(int32_t(p())) * 0x1p-31f;
      std::memcpy(&bh[size_t(4 * j)], &f, 4);
    }
    where = std::string(DTYPE[q]) + " pinned";
    const uint64_t wb_bytes = nbytes(q, {128, 4096});
    const dbuf wa(wh.size(), wh.data()), wb(wb_bytes + 2, wh.data(), 2), xb(xh.size(), xh.data()),
        bias(bh.size(), bh.data()), oa(4096 * 4), ob(128 * 2);
    const larch_tensor w_a = view(wa.b, 0, q, 4096, 4096), x_a = view(xb.b, 0, LARCH_F16, 1, 4096),
                        o_a = view(oa.b, 0, LARCH_F32, 1, 4096), w_b = view(wb.b, 2, q, 128, 4096),
                        x_b = view(xb.b, 2 * (4096 + 8) + 2, LARCH_F16, 1, 4096),
                        o_b = view(ob.b, 0, LARCH_F16, 1, 128), b = {bias.b, 0, LARCH_F32, 1, {128}, {1}};
    need(larch_mmvq(stream, &w_a, &x_a, nullptr, &o_a), "mmvq A");
    need(larch_mmvq(stream, &w_b, &x_b, &b, &o_b), "mmvq B");
    bytes out = fetch(oa.b, 4096 * 4);
    const bytes out_b = fetch(ob.b, 128 * 2);
    out.insert(out.end(), out_b.begin(), out_b.end());
    const uint64_t got = fnv(out.data(), out.size());
    if (!sm89 || !want)
      std::printf("verify: pinned %s: 0x%016llx\n", DTYPE[q], (unsigned long long)got);
    else if (got == want)
      ++eq;
    else {
      std::fprintf(stderr, "verify: pinned %s: got 0x%016llx want 0x%016llx\n", DTYPE[q], (unsigned long long)got,
                   (unsigned long long)want);
      ++failed;
    }
  }
  if (sm89)
    std::printf("verify: one-row mmvq == pinned bytes (sm_89, 24 SMs, half2 route %s): %lld/%lld formats\n",
                h2 ? "on" : "off", (long long)eq,
                (long long)std::count_if(std::begin(PINNED), std::end(PINNED), [](auto& p) { return p.hash != 0; }));
  else
    std::printf("verify: one-row mmvq == pinned bytes (sm_89, 24 SMs): GPU 0 is sm_%u with %u SMs: NOT RUN\n",
                unsigned(props.arch_code), unsigned(props.compute_units));
}

// Metal's mmvq_id in drafter mode (a stream whose last dense GEMV was mmvq_verify, as in every step of a guessing
// engine: backend.mm runs it per pair at any row count): at the 35B's expert shapes (64 experts, top 8) and two ragged
// ones with bias, on random routes, every row on the same 8 experts, and each expert picked twice a row, row r of a 2-
// to 32-row call is byte-equal to row r alone; each row is within ops.cpp's mmvq_id tolerance of the CPU reference
// (random routes); and mmvq leaves the mode (the plain bytes come back). The unequal rows.
static int64_t drafter_experts() {
  larch_device* cpu = nullptr;
  larch_stream* cs = nullptr;
  need(larch_device_open(-1, &cpu), "CPU reference device_open");
  need(larch_stream_create(cpu, &cs), "CPU reference stream_create");
  const bytes tw = quant(LARCH_Q4_K, 1, 256, g), tx = halves(256, g);
  const dbuf twb(tw.size(), tw.data()), txb(tx.size(), tx.data()), tob(4);
  const larch_tensor tw_t = view(twb.b, 0, LARCH_Q4_K, 1, 256), tx_t = view(txb.b, 0, LARCH_F16, 1, 256),
                      to_t = view(tob.b, 0, LARCH_F32, 1, 1);
  const auto mode = [&](bool drafter) {
    need(drafter ? larch_mmvq_verify(stream, &tw_t, &tx_t, nullptr, &to_t)
                 : larch_mmvq(stream, &tw_t, &tx_t, nullptr, &to_t),
         "mode switch");
  };
  static const struct {
    const char* name;
    int64_t experts, n, k;
    bool weighted, bias;
    larch_dtype od;
  } CASES[] = {{"35B gate/up", 64, 512, 2048, false, false, LARCH_F16},
               {"35B down", 64, 2048, 512, true, false, LARCH_F32},
               {"ragged", 8, 67, 1024, false, true, LARCH_F32},
               {"ragged down", 8, 33, 512, true, true, LARCH_F16}};
  constexpr int64_t R = 32, S = 8;
  const auto floats = [](size_t n) {  // f32 in [-1, 1)
    bytes b(4 * n);
    for (size_t i = 0; i < n; ++i) {
      const float f = float(int32_t(g())) * 0x1p-31f;
      std::memcpy(&b[4 * i], &f, 4);
    }
    return b;
  };
  int64_t unequal = 0;
  for (const larch_dtype q : {LARCH_Q3_K, LARCH_Q4_K, LARCH_Q6_K})
    for (const auto& c : CASES) {
      const int64_t en = c.experts * c.n, x_rows = c.weighted ? R * S : R, per_row = (c.weighted ? 1 : S) * c.n;
      const uint64_t row = uint64_t(per_row) * (c.od == LARCH_F32 ? 4 : 2);
      const bytes wh = quant(q, en, c.k, g), xh = halves(size_t(x_rows * c.k), g), bh = floats(size_t(en)),
                  wth = floats(size_t(R * S)), poison(size_t(R * row), 0xFF);
      const dbuf wb(wh.size(), wh.data()), xb(xh.size(), xh.data()), bb(bh.size(), bh.data()),
          wtb(wth.size(), wth.data()), one(poison.size()), many(poison.size());
      for (int route = 0; route < 3; ++route) {  // random, all rows the same 8 experts, each of 4 experts twice
        std::vector<int32_t> ids;
        std::vector<int32_t> pick(size_t(c.experts));
        std::iota(pick.begin(), pick.end(), 0);
        std::shuffle(pick.begin(), pick.end(), g);
        for (int64_t r = 0; r < R; ++r) {
          if (route != 1) std::shuffle(pick.begin(), pick.end(), g);
          std::vector<int32_t> slots(pick.begin(), pick.begin() + S);
          if (route == 2) std::copy(slots.begin(), slots.begin() + S / 2, slots.begin() + S / 2);
          std::shuffle(slots.begin(), slots.end(), g);
          ids.insert(ids.end(), slots.begin(), slots.end());
        }
        const dbuf idb(ids.size() * 4, ids.data());
        where = std::string(DTYPE[q]) + " " + c.name + " drafter mmvq_id, route " + std::to_string(route);
        // larch_mmvq_id on rows r0 .. r0 + rows - 1 into o's rows r0 on, on stream sv: bufs w, x, ids, bias, weights.
        const auto call = [&](larch_stream* sv, larch_buffer* const* bufs, larch_buffer* o, int64_t r0,
                              int64_t rows) {
          const larch_tensor w = {bufs[0], 0, q, 2, {en, c.k}, {c.k / block_of(q), 1}},
                              x = c.weighted ? larch_tensor{bufs[1], uint64_t(r0 * S * c.k * 2), LARCH_F16, 3,
                                                             {rows, S, c.k}, {S * c.k, c.k, 1}}
                                             : larch_tensor{bufs[1], uint64_t(r0 * c.k * 2), LARCH_F16, 2,
                                                             {rows, c.k}, {c.k, 1}},
                              id = {bufs[2], uint64_t(r0 * S * 4), LARCH_I32, 2, {rows, S}, {S, 1}},
                              b = {bufs[3], 0, LARCH_F32, 1, {en}, {1}},
                              wt = {bufs[4], uint64_t(r0 * S * 4), LARCH_F32, 2, {rows, S}, {S, 1}},
                              out = c.weighted ? larch_tensor{o, uint64_t(r0) * row, c.od, 2, {rows, c.n}, {c.n, 1}}
                                               : larch_tensor{o, uint64_t(r0) * row, c.od, 3, {rows, S, c.n},
                                                               {S * c.n, c.n, 1}};
          return larch_mmvq_id(sv, &w, &x, &id, c.bias ? &b : nullptr, c.weighted ? &wt : nullptr, &out);
        };
        larch_buffer* const gpu_bufs[5] = {wb.b, xb.b, idb.b, bb.b, wtb.b};
        const auto run = [&](larch_buffer* o, int64_t r0, int64_t rows) {
          return ok(call(stream, gpu_bufs, o, r0, rows), std::to_string(rows) + "-row mmvq_id");
        };
        // Plain mode's 8 rows; only where it runs them per pair (it groups from 4 pairs an expert, args.h
        // MOE_GROUP_PAIRS), whose bytes repeat.
        const bool per_pair = 8 * S < 4 * c.experts;
        mode(false);
        need(larch_upload(stream, many.b, 0, poison.data(), poison.size()), "upload");
        if (!run(many.b, 0, 8)) continue;
        const bytes plain = fetch(many.b, 8 * row);
        mode(true);
        need(larch_upload(stream, one.b, 0, poison.data(), poison.size()), "upload");
        for (int64_t r = 0; r < R; ++r) run(one.b, r, 1);
        const bytes want = fetch(one.b, poison.size());
        int64_t eq = 0, planned = 0;
        for (const int64_t rows : {2, 3, 4, 8, 16, 32}) {
          need(larch_upload(stream, many.b, 0, poison.data(), poison.size()), "upload");
          if (!run(many.b, 0, rows)) continue;
          const bytes got = fetch(many.b, uint64_t(rows) * row);
          for (int64_t r = 0; r < rows; ++r) {
            const size_t at = size_t(r) * row;
            if (!std::memcmp(&got[at], &want[at], row)) {
              ++eq;
              continue;
            }
            // Which slot and column, how many elements, and whether either side is the poison (an unwritten output).
            const size_t es = c.od == LARCH_F32 ? 4 : 2;
            int64_t first = -1, diff = 0;
            for (size_t e = 0; e < row / es; ++e)
              if (std::memcmp(&got[at + e * es], &want[at + e * es], es)) {
                first = first < 0 ? int64_t(e) : first;
                ++diff;
              }
            uint32_t gv = 0, wv = 0;
            std::memcpy(&gv, &got[at + size_t(first) * es], es);
            std::memcpy(&wv, &want[at + size_t(first) * es], es);
            std::fprintf(stderr,
                         "verify: %s: %lld-row call, row %lld: %lld elements differ, first slot %lld column %lld: "
                         "got 0x%08x want 0x%08x\n",
                         where.c_str(), (long long)rows, (long long)r, (long long)diff, (long long)(first / c.n),
                         (long long)(first % c.n), gv, wv);
          }
          planned += rows;
        }
        // The rows alone again: a reference that changes between two passes is nondeterministic itself.
        need(larch_upload(stream, one.b, 0, poison.data(), poison.size()), "upload");
        for (int64_t r = 0; r < R; ++r) run(one.b, r, 1);
        if (fetch(one.b, poison.size()) != want)
          std::fprintf(stderr, "verify: %s: two passes of the rows alone gave other bytes\n", where.c_str());
        unequal += planned - eq;
        mode(false);
        need(larch_upload(stream, many.b, 0, poison.data(), poison.size()), "upload");
        if (run(many.b, 0, 8) && per_pair && fetch(many.b, 8 * row) != plain) {
          std::fprintf(stderr, "verify: %s: plain mode after mmvq gave other bytes\n", where.c_str());
          ++failed;
        }
        int64_t off = 0;  // elements off ops.cpp's mmvq_id tolerance of the CPU reference: rtol |ref| + atol max |ref|
        if (route == 0) {
          const bytes idh(reinterpret_cast<const uint8_t*>(ids.data()),
                          reinterpret_cast<const uint8_t*>(ids.data() + ids.size()));
          const bytes* src[6] = {&wh, &xh, &idh, &bh, &wth, &poison};
          larch_buffer* cb[6] = {};
          for (int i = 0; i < 6; ++i) {
            need(larch_buffer_alloc(cpu, src[i]->size(), &cb[i]), "CPU reference buffer_alloc");
            need(larch_upload(cs, cb[i], 0, src[i]->data(), src[i]->size()), "CPU reference upload");
          }
          need(call(cs, cb, cb[5], 0, R), "CPU reference mmvq_id");
          bytes ref(poison.size());
          need(larch_download(cs, cb[5], 0, ref.data(), ref.size()), "CPU reference download");
          need(larch_stream_sync(cs), "CPU reference sync");
          for (larch_buffer* b : cb) larch_buffer_free(b);
          const bool f32 = c.od == LARCH_F32;
          const double rtol = f32 ? 1e-3 : 2e-3, atol = 1e-3;
          const auto value = [&](const bytes& v, size_t i) -> double {
            if (f32) {
              float f;
              std::memcpy(&f, &v[4 * i], 4);
              return f;
            }
            uint16_t h;
            std::memcpy(&h, &v[2 * i], 2);
            const int e = h >> 10 & 31, m = h & 1023;
            const double mag =
                e == 31 ? (m ? NAN : INFINITY) : e ? std::ldexp(double(m + 1024), e - 25) : std::ldexp(double(m), -24);
            return h & 0x8000 ? -mag : mag;
          };
          const size_t elems = size_t(R * per_row);
          double peak = 0;
          for (size_t i = 0; i < elems; ++i)
            if (const double r = value(ref, i); std::isfinite(r)) peak = std::max(peak, std::fabs(r));
          for (size_t i = 0; i < elems; ++i) {  // as ops.cpp: an infinity or NaN only matches its own kind
            const double r = value(ref, i), d = value(want, i);
            off += !(d == r || (std::isnan(d) && std::isnan(r)) || std::fabs(d - r) <= rtol * std::fabs(r) + atol * peak);
          }
          if (off) {
            std::fprintf(stderr, "verify: %s: %lld elements off the CPU reference\n", where.c_str(), (long long)off);
            ++failed;
          }
        }
        std::printf("verify: %s %s drafter mmvq_id, route %d, rows 2-32: %lld/%lld rows byte-equal to the row alone, "
                    "%lld elements off the CPU reference\n",
                    DTYPE[q], c.name, route, (long long)eq, (long long)planned, (long long)off);
      }
    }
  larch_stream_destroy(cs);
  larch_device_close(cpu);
  return unequal;
}

int main() {
  int32_t gpus = 0;
  need(larch_device_count(&gpus), "device_count");
  if (gpus < 1) return std::puts("verify: no GPU: NOT RUN"), 77;
  need(larch_device_open(0, &dev), "device_open");
  need(larch_device_props_get(dev, &props), "device_props_get");
  if (props.backend != LARCH_BACKEND_CUDA && props.backend != LARCH_BACKEND_METAL) {
    std::printf("verify: GPU 0 (%s) is neither CUDA nor Metal: NOT RUN\n", props.name);
    return 77;
  }
  need(larch_stream_create(dev, &stream), "stream_create");
  if (props.backend == LARCH_BACKEND_METAL) {
    one_row = larch_mmvq_verify;
    const bytes w = quant(LARCH_Q4_K, 1, 256, g), x = halves(256, g);
    const dbuf wb(w.size(), w.data()), xb(x.size(), x.data()), ob(4);
    const larch_tensor wt = view(wb.b, 0, LARCH_Q4_K, 1, 256), xt = view(xb.b, 0, LARCH_F16, 1, 256),
                        ot = view(ob.b, 0, LARCH_F32, 1, 1);
    if (larch_mmvq_verify(stream, &wt, &xt, nullptr, &ot) == LARCH_ERR_UNSUPPORTED) {
      std::printf("verify: GPU 0 (%s) has no staged tile: NOT RUN\n", props.name);
      return 77;
    }
  } else {
    routed_up_to = arch::mmvq_rows_up_to(int(props.arch_code));
    pinned();
  }
  int64_t unequal = 0, n_max = 0;
  for (const shape& c : SHAPES) n_max = std::max(n_max, c.n);
  {
    bytes bias(size_t(4 * n_max));  // f32 in [-1, 1)
    for (int64_t i = 0; i < n_max; ++i) {
      const float f = float(int32_t(g())) * 0x1p-31f;
      std::memcpy(&bias[size_t(4 * i)], &f, 4);
    }
    const dbuf bias_buf(bias.size(), bias.data());
    // One weight per (format, k), as tall as the tallest shape of that k, so the shapes of a k view its first n rows.
    for (const larch_dtype q : QUANTS) {
      larch_status probe = LARCH_OK;
      where = std::string(DTYPE[q]) + " probe";
      if (lacks_kernels(dev, stream, q, &probe)) {
        std::printf("verify: %s: GPU 0 has no kernels for it: NOT RUN\n", DTYPE[q]);
        continue;
      }
      ok(probe, "mmvq probe");  // any other failure fails the test; the format is still gated
      for (size_t i = 0; i < std::size(SHAPES); ++i) {
        const int64_t k = SHAPES[i].k;
        const auto has_k = [&](const shape& c) { return c.k == k; };
        if (k % block_of(q) || std::any_of(SHAPES, SHAPES + i, has_k)) continue;
        int64_t n = 0;
        for (const shape& c : SHAPES) n = has_k(c) ? std::max(n, c.n) : n;
        const bytes wh = quant(q, n, k, g), xh = halves(size_t(32 * (k + 8)), g);
        const dbuf wb(wh.size(), wh.data());
        for (const shape& c : SHAPES) {
          if (!has_k(c)) continue;
          const size_t variants = c.flags & ALIGN ? std::size(VARIANTS) : 1;
          const int64_t planned = int64_t(variants) * (c.flags & LM_HEAD ? 1 : 2) * 528;  // 528 = 1 + 2 + ... + 32
          int64_t eq = 0;
          bool good = true;
          for (size_t vi = 0; vi < variants && good; ++vi)
            for (const larch_dtype od : {LARCH_F32, LARCH_F16})
              if (good && !(od == LARCH_F16 && c.flags & LM_HEAD))
                good = gate(q, c, VARIANTS[vi], od, vi == 0 && c.flags & CAPTURE, wb.b, wh, xh, bias_buf.b, eq);
          std::printf("verify: %s %s %lldx%lld%s rows 1-32: %lld/%lld rows byte-equal\n", DTYPE[q], c.name,
                      (long long)c.n, (long long)c.k, variants > 1 ? " +variants" : "", (long long)eq,
                      (long long)planned);
          unequal += planned - eq;
        }
      }
    }
  }
  if (props.backend == LARCH_BACKEND_METAL) unequal += drafter_experts();
  std::printf("verify: capture == eager: %lld/%lld outputs byte-equal\n", (long long)cap_eq, (long long)cap_total);
  if (routed_up_to > 1)
    std::printf("verify: routed mmvq == mmvq_verify: %lld/%lld outputs byte-equal\n", (long long)route_eq,
                (long long)route_total);
  else
    std::printf("verify: routed mmvq == mmvq_verify: GPU 0 is sm_%u, which routes no rows: NOT RUN\n",
                unsigned(props.arch_code));
  std::printf("verify: %lld rows unequal\n", (long long)unequal);
  larch_stream_destroy(stream);
  larch_device_close(dev);
  return unequal || failed || cap_eq != cap_total || route_eq != route_total;
}
