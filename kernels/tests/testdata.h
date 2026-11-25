// Test data shared by ops.cpp, attention.cpp, quant_gemm.cpp, mmvq_bench.cu, cpu_reference.cpp and verify.cpp: GGML
// block geometry, dtype names, random halves and blocks, and the graph-replay timer.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "larch.h"

using bytes = std::vector<uint8_t>;
using rng = std::mt19937;

// Elements per block of dtype t: 256 for the K quants, 32 for the other quants, 1 for plain dtypes.
static inline int64_t block_of(int t) {
  return t == LARCH_Q2_K || t == LARCH_Q3_K || t == LARCH_Q4_K || t == LARCH_Q5_K || t == LARCH_Q6_K ? 256
         : t == LARCH_Q8_0 || t == LARCH_Q4_0 || t == LARCH_Q5_0 ? 32
                                                                    : 1;
}
// Bytes per block of dtype t (per element for plain dtypes).
static inline int64_t block_bytes(int t) {
  switch (t) {
    case LARCH_F16:
    case LARCH_BF16:
      return 2;
    case LARCH_Q2_K:
      return 84;
    case LARCH_Q3_K:
      return 110;
    case LARCH_Q4_K:
      return 144;
    case LARCH_Q5_K:
      return 176;
    case LARCH_Q6_K:
      return 210;
    case LARCH_Q8_0:
      return 34;
    case LARCH_Q4_0:
      return 18;
    case LARCH_Q5_0:
      return 22;
    default:
      return 4;
  }
}
// Bytes of a dense tensor of dtype t and this shape.
static inline uint64_t nbytes(int t, const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (const int64_t d : shape) n *= d;
  return uint64_t(n / block_of(t) * block_bytes(t));
}

// The quant formats larch.h takes.
static const larch_dtype QUANTS[] = {LARCH_Q3_K, LARCH_Q4_K, LARCH_Q5_K, LARCH_Q6_K, LARCH_Q8_0, LARCH_Q4_0, LARCH_Q5_0};
// True when a valid one-row mmvq of q (one block, zero-filled inputs) on this device answers UNSUPPORTED, i.e. the
// backend has no kernels for q (vv_ops::no_weights; malformed calls answer INVALID_ARG instead). mmvq, not dequant:
// both are refused for the same dtypes, and mmvq is the call verify.cpp gates. Any other status (an alloc or upload
// failure included) is stored in *st, when given, for the caller to report.
static inline bool lacks_kernels(larch_device* dev, larch_stream* s, larch_dtype q, larch_status* st = nullptr) {
  const int64_t k = block_of(q);
  const uint64_t wn = nbytes(q, {1, k}), xn = 2 * uint64_t(k);
  const std::vector<uint8_t> zeros(xn);
  larch_buffer *w = nullptr, *x = nullptr, *o = nullptr;
  larch_status r = larch_buffer_alloc(dev, wn, &w);
  if (r == LARCH_OK) r = larch_buffer_alloc(dev, xn, &x);
  if (r == LARCH_OK) r = larch_buffer_alloc(dev, 4, &o);
  if (r == LARCH_OK) r = larch_upload(s, w, 0, zeros.data(), wn);
  if (r == LARCH_OK) r = larch_upload(s, x, 0, zeros.data(), xn);
  if (r == LARCH_OK) {
    const larch_tensor wt = {w, 0, q, 2, {1, k}, {1, 1}}, xt = {x, 0, LARCH_F16, 2, {1, k}, {k, 1}},
                        ot = {o, 0, LARCH_F32, 2, {1, 1}, {1, 1}};
    r = larch_mmvq(s, &wt, &xt, nullptr, &ot);
  }
  const larch_status sync = larch_stream_sync(s);  // the buffers must not be freed under a running call
  if (r == LARCH_OK) r = sync;
  larch_buffer_free(w);
  larch_buffer_free(x);
  larch_buffer_free(o);
  if (st) *st = r;
  return r == LARCH_ERR_UNSUPPORTED;
}
// Each dtype's name, indexed by larch_dtype.
static const char* const DTYPE[] = {"f32",  "f16",  "bf16", "i32",  "q4_k",     "q5_k",
                                    "q6_k", "q8_0", "q4_0", "q5_0", "fp8_e4m3", "q3_k", "q2_k"};

// n halves with magnitude in [2^-6, 2^hi), either sign.
static inline bytes halves(size_t n, rng& g, int hi = 1) {
  bytes b(2 * n);
  for (size_t i = 0; i < n; ++i) {
    const uint16_t h = uint16_t((g() & 1) << 15 | (9 + g() % uint32_t(6 + hi)) << 10 | (g() & 0x3ff));
    std::memcpy(&b[2 * i], &h, 2);
  }
  return b;
}
// Random blocks with f16 scales in [0, 1), and three edge blocks: the first all ones, the second all
// zeros, the last with zero scales.
static inline bytes quant(larch_dtype t, int64_t rows, int64_t k, rng& g) {
  const int64_t nb = rows * k / block_of(t), size = block_bytes(t);
  bytes b(size_t(nb * size));
  for (uint8_t& x : b) x = uint8_t(g());
  const std::vector<int> scales = t == LARCH_Q6_K     ? std::vector<int>{208}
                                  : t == LARCH_Q3_K   ? std::vector<int>{108}
                                  : t == LARCH_Q2_K   ? std::vector<int>{80, 82}
                                  : block_of(t) == 256 ? std::vector<int>{0, 2}
                                                       : std::vector<int>{0};
  uint8_t* blk = b.data();
  if (nb > 0) std::fill(blk, blk + size, uint8_t(0xFF));
  if (nb > 1) std::fill(blk + size, blk + 2 * size, uint8_t(0));
  for (int64_t i = 0; i < nb; ++i)
    for (const int o : scales) blk[i * size + o + 1] &= 0x3b;
  if (nb > 2)
    for (const int o : scales) blk[(nb - 1) * size + o] = blk[(nb - 1) * size + o + 1] = 0;
  return b;
}

// FNV-1a 64 of n bytes at p.
static inline uint64_t fnv(const void* p, size_t n) {
  uint64_t h = 0xcbf29ce484222325u;
  for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 0x100000001b3u;
  return h;
}

// Microseconds per call of issue(i), best of four graph replays after a warm-up and a probe that sizes the
// capture to ~10 ms (10 to 512 calls); callers rotate weights or pool layers by i for cold reads. -1 on failure.
template <class F>
static double replay_time(larch_stream* s, F&& issue) {
  int at = 0;
  const auto timed = [&](auto&& run, int count) {
    larch_event *start = nullptr, *end = nullptr;
    uint64_t ns = 0;
    const bool ok = larch_stream_sync(s) == LARCH_OK && larch_event_record(s, &start) == LARCH_OK && run() &&
                    larch_event_record(s, &end) == LARCH_OK && larch_stream_sync(s) == LARCH_OK &&
                    larch_event_elapsed_ns(start, end, &ns) == LARCH_OK;
    larch_event_destroy(start);  // null-safe: a failed record leaves its event null
    larch_event_destroy(end);
    return ok ? double(ns) / count / 1e3 : -1.0;
  };
  const auto direct = [&](int count) {
    return timed([&] {
      for (int i = 0; i < count; ++i)
        if (issue(at++) != LARCH_OK) return false;
      return true;
    }, count);
  };
  if (direct(3) < 0) return -1;
  const double probe = direct(10);
  if (probe <= 0) return -1;
  const int iters = std::clamp(int(std::ceil(10e3 / probe)), 10, 512);
  larch_graph* graph = nullptr;
  if (larch_graph_begin(s) != LARCH_OK) return -1;
  larch_status status = LARCH_OK;
  for (int i = 0; i < iters && status == LARCH_OK; ++i) status = issue(at++);
  const larch_status ended = larch_graph_end(s, &graph);
  if (status != LARCH_OK || ended != LARCH_OK) {
    if (graph) larch_graph_destroy(graph);
    return -1;
  }
  const auto replay = [&] { return timed([&] { return larch_graph_launch(graph, s) == LARCH_OK; }, iters); };
  double us = replay();
  if (us >= 0)
    for (int i = 0; i < 3; ++i) {
      const double next = replay();
      if (next < 0) { us = -1; break; }
      us = std::min(us, next);
    }
  larch_graph_destroy(graph);
  return us;
}
