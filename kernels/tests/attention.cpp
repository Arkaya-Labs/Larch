// Attention paths share the ops.cpp CPU oracle and its unchanged f16 tolerance.
#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <tuple>
#include <vector>

#include "../src/common/arch.h"
#include "testdata.h"

static void require(larch_status s, larch_device* d) {
  if (s != LARCH_OK) {
    std::fprintf(stderr, "attention: status %d: %s\n", int(s), larch_last_error(d));
    std::exit(1);
  }
}

struct side {
  larch_device* d = nullptr;
  larch_stream* s = nullptr;
  explicit side(int ordinal) {
    require(larch_device_open(ordinal, &d), d);
    require(larch_stream_create(d, &s), d);
  }
  ~side() {
    larch_stream_destroy(s);
    larch_device_close(d);
  }
};

// Guarded, row-padded and half-aligned tensors catch tails and indexing errors; `aligned` starts the data at a
// 16-byte offset instead, for the vector loads that need it.
struct tensor {
  side& x;
  larch_tensor t{};
  bytes before;
  uint64_t off;
  tensor(side& device, larch_dtype dtype, const std::vector<int64_t>& shape, const bytes& data,
         int64_t pad = 0, bool aligned = false)
      : x(device), before(data.size() + 100, 0xa5), off(aligned ? 32 : dtype == LARCH_F16 ? 34 : 36) {
    std::copy(data.begin(), data.end(), before.begin() + off);
    require(larch_buffer_alloc(x.d, before.size(), &t.buf), x.d);
    t.offset = off;
    t.dtype = dtype;
    t.ndim = uint32_t(shape.size());
    int64_t stride = 1;
    for (int i = int(shape.size()) - 1; i >= 0; --i) {
      t.shape[i] = shape[size_t(i)];
      t.stride[i] = stride;
      stride *= shape[size_t(i)];
    }
    t.stride[0] += pad;
    require(larch_upload(x.s, t.buf, 0, before.data(), before.size()), x.d);
  }
  ~tensor() { larch_buffer_free(t.buf); }
  bytes read(bool output = false) {
    bytes got(before.size());
    require(larch_download(x.s, t.buf, 0, got.data(), got.size()), x.d);
    require(larch_stream_sync(x.s), x.d);
    for (size_t i = 0; i < got.size(); ++i)
      if ((!output || i < off || i >= before.size() - 100 + off) && got[i] != before[i]) {
        std::fprintf(stderr, "attention: input or guard changed at %zu\n", i);
        std::exit(1);
      }
    return got;
  }
};

template <class T> static bytes raw(const std::vector<T>& v) {
  bytes b(v.size() * sizeof(T));
  if (!b.empty()) std::memcpy(b.data(), v.data(), b.size());
  return b;
}

static float from_half(const uint8_t* p) {
  uint16_t h;
  std::memcpy(&h, p, 2);
  const int e = h >> 10 & 31, m = h & 1023;
  const float v = e == 31 ? (m ? std::numeric_limits<float>::quiet_NaN() : INFINITY)
                  : e ? std::ldexp(float(m + 1024), e - 25) : std::ldexp(float(m), -24);
  return h & 0x8000 ? -v : v;
}

struct shape {
  int hd, kv, group;
  std::vector<int32_t> lens, start;
  uint32_t splits;
  bool nan_query = false;
  bool graph = false;
  std::vector<int32_t> replay_lens;  // decode graph: replay at these lengths, longer than captured
  int v_hi = 1;                      // V magnitudes up to 2^v_hi (16: up to 65504)
  // The prefill stages q with 16-byte loads only when both its offset and its rows (pad 8) are 16-byte aligned.
  bool q_aligned = false;  // q at a 16-byte offset (else 34 bytes)
  int q_pad = 3;           // q row padding in halves
  bool one_row = false;    // decode: each row run alone on the device must give the batch's bytes
  // larch_attn over causal spans, start holding each sequence's first position; a graph replays again at
  // replay_start and replay_lens. Mask, window and sinks are there for a GPU backend's refusal.
  bool span = false;
  bool mask = false;  // the causal chain as a mask: row j keeps bits 0..j
  uint32_t window = 0;
  bool sinks = false;
  std::vector<int32_t> replay_start;
  larch_status* status = nullptr;  // takes a refusal instead of failing, and then no byte may have changed
};

static void compare(const bytes& ref, const bytes& got, const shape& c);

static bytes run(side& x, const shape& c, const bytes& q, const bytes& k, const bytes& v,
                 const std::vector<int32_t>& table, int width, int rows) {
  const int heads = c.kv * c.group, row_width = heads * c.hd;
  tensor qt(x, LARCH_F16, {rows, heads, c.hd}, q, c.q_pad, c.q_aligned);
  tensor kt(x, LARCH_F16, {int64_t(k.size() / 2)}, k);
  tensor vt(x, LARCH_F16, {int64_t(v.size() / 2)}, v);
  // Pool storage starts at byte zero; tensor guards are only relevant to ordinary tensors.
  require(larch_upload(x.s, kt.t.buf, 0, k.data(), k.size()), x.d);
  require(larch_upload(x.s, vt.t.buf, 0, v.data(), v.size()), x.d);
  std::copy(k.begin(), k.end(), kt.before.begin());
  std::copy(v.begin(), v.end(), vt.before.begin());
  tensor bt(x, LARCH_I32, {int64_t(c.lens.size()), width}, raw(table), 1);
  tensor lt(x, LARCH_I32, {int64_t(c.lens.size())}, raw(c.lens));
  tensor st(x, LARCH_I32, {int64_t(c.start.size())}, raw(c.start));
  tensor ot(x, LARCH_F16, {rows, heads, c.hd}, bytes(size_t(rows * (row_width + 3) * 2), 0xff), 3);
  const larch_kv_pool pool{kt.t.buf, vt.t.buf, uint32_t(k.size() / 2 / (32 * c.kv * c.hd)), 16, 2,
                            uint32_t(c.kv), uint32_t(c.hd), LARCH_F16, 1};
  const float scale = 1.0f / std::sqrt(float(c.hd));
  std::optional<tensor> mt, sk;
  if (c.mask) {
    std::vector<int32_t> chain;
    for (const int32_t n : c.lens)
      for (int32_t j = 0; j < n; ++j) chain.push_back(int32_t((2u << j) - 1));
    mt.emplace(x, LARCH_I32, std::vector<int64_t>{rows}, raw(chain));
  }
  if (c.sinks) sk.emplace(x, LARCH_F32, std::vector<int64_t>{heads}, bytes(size_t(heads) * 4, 0));
  larch_status got = LARCH_OK;
  const auto invoke = [&] {
    got = c.span ? larch_attn(x.s, &qt.t, &pool, &bt.t, &st.t, &lt.t, mt ? &mt->t : nullptr, sk ? &sk->t : nullptr,
                               c.window, c.splits, scale, &ot.t)
          : c.start.empty() ? larch_attn_decode(x.s, &qt.t, &pool, &bt.t, &lt.t, c.splits, scale, &ot.t)
                            : larch_attn_prefill(x.s, &qt.t, &pool, &bt.t, &st.t, &lt.t, scale, &ot.t);
    if (c.status) *c.status = got;
    else require(got, x.d);
  };
  larch_graph* graph = nullptr;
  larch_device_props props{};
  require(larch_device_props_get(x.d, &props), x.d);
  if (c.graph && (props.flags & LARCH_CAP_GRAPHS)) {
    require(larch_stream_sync(x.s), x.d);
    require(larch_graph_begin(x.s), x.d);
    invoke();
    require(larch_graph_end(x.s, &graph), x.d);
    if (got == LARCH_OK) require(larch_graph_launch(graph, x.s), x.d);
  } else {
    // Without graphs, the replay lengths run directly; the stream orders the restore after the op.
    const auto upload = [&](const std::vector<int32_t>& n) {
      require(larch_upload(x.s, lt.t.buf, lt.off, n.data(), n.size() * sizeof(int32_t)), x.d);
    };
    if (!c.replay_lens.empty()) upload(c.replay_lens);
    invoke();
    if (!c.replay_lens.empty()) upload(c.lens);
  }
  require(larch_stream_sync(x.s), x.d);
  bytes out = ot.read(got == LARCH_OK), replayed;
  if (graph) {
    const std::vector<int32_t> zeros(c.lens.size());
    if (c.span) {
      // Positions and boundaries are device data: the capture must serve ones it never saw, then the originals return.
      if (got == LARCH_OK && !c.replay_lens.empty()) {
        require(larch_upload(x.s, st.t.buf, st.off, c.replay_start.data(), c.replay_start.size() * 4), x.d);
        require(larch_upload(x.s, lt.t.buf, lt.off, c.replay_lens.data(), c.replay_lens.size() * 4), x.d);
        require(larch_graph_launch(graph, x.s), x.d);
        replayed = ot.read(true);
        require(larch_upload(x.s, st.t.buf, st.off, c.start.data(), c.start.size() * 4), x.d);
      }
    } else if (c.start.empty()) {
      require(larch_upload(x.s, lt.t.buf, lt.off, zeros.data(), zeros.size() * sizeof(int32_t)), x.d);
      require(larch_graph_launch(graph, x.s), x.d);
      const bytes padded = ot.read(true);
      for (int r = 0; r < rows; ++r)
        for (int d = 0; d < row_width * 2; ++d)
          if (padded[ot.off + size_t(r * (row_width + 3) * 2 + d)] != 0) {
            std::fprintf(stderr, "attention: captured decode ignored updated padding lengths\n");
            std::exit(1);
          }
      if (!c.replay_lens.empty()) {
        // Lengths are device data: the capture's launch shape must serve contexts it never saw.
        require(larch_upload(x.s, lt.t.buf, lt.off, c.replay_lens.data(), c.replay_lens.size() * sizeof(int32_t)),
                x.d);
        require(larch_graph_launch(graph, x.s), x.d);
        replayed = ot.read(true);
      }
    } else {
      // Preserve total query rows while changing sequence boundaries and cached offsets.
      shape replay = c;
      replay.graph = false;
      replay.start = zeros;
      std::reverse(replay.lens.begin(), replay.lens.end());
      require(larch_upload(x.s, lt.t.buf, lt.off, replay.lens.data(), replay.lens.size() * sizeof(int32_t)), x.d);
      require(larch_upload(x.s, st.t.buf, st.off, zeros.data(), zeros.size() * sizeof(int32_t)), x.d);
      require(larch_graph_launch(graph, x.s), x.d);
      const bytes changed = ot.read(true);
      bytes dense(size_t(rows * row_width * 2));
      for (int r = 0; r < rows; ++r)
        std::copy_n(changed.begin() + int64_t(ot.off) + int64_t(r) * (row_width + 3) * 2, row_width * 2,
                    dense.begin() + int64_t(r) * row_width * 2);
      side reference(-1);
      compare(run(reference, replay, q, k, v, table, width, rows), dense, replay);
      require(larch_upload(x.s, st.t.buf, st.off, c.start.data(), c.start.size() * sizeof(int32_t)), x.d);
    }
    require(larch_upload(x.s, lt.t.buf, lt.off, c.lens.data(), c.lens.size() * sizeof(int32_t)), x.d);
    require(larch_stream_sync(x.s), x.d);
    larch_graph_destroy(graph);
  }
  qt.read(); kt.read(); vt.read(); bt.read(); lt.read(); st.read();
  if (mt) mt->read();
  if (sk) sk->read();
  if (!replayed.empty()) out = replayed;
  // Row padding remains poisoned, even in partially filled query tiles.
  for (int r = 0; r < rows; ++r)
    for (int d = row_width; d < row_width + 3; ++d)
      if (out[ot.off + size_t((r * (row_width + 3) + d) * 2)] != 0xff ||
          out[ot.off + size_t((r * (row_width + 3) + d) * 2 + 1)] != 0xff) std::exit(1);
  bytes dense(size_t(rows * row_width * 2));
  for (int r = 0; r < rows; ++r)
    std::copy_n(out.begin() + int64_t(ot.off) + int64_t(r) * (row_width + 3) * 2, row_width * 2,
                dense.begin() + int64_t(r) * row_width * 2);
  return dense;
}

static void compare(const bytes& ref, const bytes& got, const shape& c) {
  double max_ref = 0;
  for (size_t i = 0; i < ref.size(); i += 2) {
    const double r = from_half(&ref[i]);
    if (std::isfinite(r)) max_ref = std::max(max_ref, std::fabs(r));
  }
  for (size_t i = 0; i < ref.size(); i += 2) {
    const double r = from_half(&ref[i]), d = from_half(&got[i]);
    const bool equal = std::isnan(r) ? std::isnan(d) : std::isinf(r) ? d == r :
                       std::isfinite(d) && std::fabs(d - r) <= 2e-3 * std::fabs(r) + 5e-4 * max_ref;
    if (!equal) {
      std::fprintf(stderr, "attention hd%d gqa%d splits%u prefill%d element%zu ref%.9g got%.9g\n",
                   c.hd, c.group, c.splits, int(!c.start.empty()), i / 2, r, d);
      std::exit(1);
    }
  }
}

static void check(side& cpu, side& dut, const shape& c) {
  const int rows = c.start.empty() ? int(c.lens.size()) : std::accumulate(c.lens.begin(), c.lens.end(), 0);
  const std::vector<int32_t>& lens = c.replay_lens.empty() ? c.lens : c.replay_lens;
  int span = 0;
  for (size_t s = 0; s < c.lens.size(); ++s) span = std::max(span, lens[s] + (c.start.empty() ? 0 : c.start[s]));
  const int width = (span + 15) / 16 + 1, blocks = width + 3, row_width = c.kv * c.group * c.hd;
  rng g(1234 + uint32_t(c.hd));
  bytes q = halves(size_t(rows * (row_width + c.q_pad)), g);
  if (c.nan_query) q[0] = 0xff, q[1] = 0x7f;
  const bytes k = halves(size_t(blocks * 16 * 2 * c.kv * c.hd), g);
  const bytes v = halves(k.size() / 2, g, c.v_hi);
  std::vector<int32_t> table(c.lens.size() * size_t(width + 1), -1);
  for (size_t s = 0; s < c.lens.size(); ++s) {
    std::vector<int32_t> permutation(size_t(blocks), 0);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), g);
    const int n = lens[s] + (c.start.empty() ? 0 : c.start[s]);
    for (int b = 0; b < (n + 15) / 16; ++b) table[s * size_t(width + 1) + size_t(b)] = permutation[size_t(b)];
  }
  shape replay = c;
  if (!c.replay_lens.empty()) replay.lens = c.replay_lens, replay.graph = false, replay.replay_lens.clear();
  const bytes ref = run(cpu, replay, q, k, v, table, width, rows), got = run(dut, c, q, k, v, table, width, rows);
  compare(ref, got, c);
  // A decode row of a multi-row step (decode rows, check spans) is the one-row decode at the same splits, bit for bit.
  const size_t q_row = size_t(row_width + c.q_pad) * 2, out_row = size_t(row_width) * 2;  // run() returns dense rows
  for (size_t r = 0; c.one_row && r < c.lens.size(); ++r) {
    shape one = c;
    one.lens = {c.lens[r]};
    const bytes alone = run(dut, one, bytes(q.begin() + int64_t(r * q_row), q.begin() + int64_t((r + 1) * q_row)), k,
                            v, std::vector<int32_t>(table.begin() + int64_t(r * (width + 1)),
                                                    table.begin() + int64_t((r + 1) * (width + 1))),
                            width, 1);
    if (!std::equal(alone.begin(), alone.end(), got.begin() + int64_t(r * out_row))) {
      std::fprintf(stderr, "attention hd%d gqa%d splits%u: row %zu of %zu differs from its one-row decode\n", c.hd,
                   c.group, c.splits, r, c.lens.size());
      std::exit(1);
    }
  }
}

// Four token subsets have different maxima; omitting either rescale or the unnormalized sum fails.
static void check_warp_merge(side& cpu, side& dut) {
  const shape c{64, 1, 1, {17}, {}, 3, false, true};
  constexpr int width = 98, blocks = width + 3, layer_slots = blocks * 16;
  bytes q((64 + 3) * 2, 0), k(layer_slots * 2 * 64 * 2, 0), v(k.size(), 0);
  const uint16_t query = 0x4800, scores[4] = {0xc600, 0xc000, 0x4000, 0x4600};
  const uint16_t values[4] = {0, 0x3c00, 0x4000, 0x4200};
  std::memcpy(q.data(), &query, 2);
  for (int p = 0; p < 17; ++p) {
    std::memcpy(k.data() + (layer_slots + p) * 64 * 2, &scores[p % 4], 2);
    for (int d = 0; d < 64; ++d)
      std::memcpy(v.data() + ((layer_slots + p) * 64 + d) * 2, &values[p % 4], 2);
  }
  std::vector<int32_t> table(width + 1, -1);
  table[0] = 0;
  table[1] = 1;
  // Full graph-style table width selects the tiled path even with17 live tokens; splits3 leaves tails.
  compare(run(cpu, c, q, k, v, table, width, 1), run(dut, c, q, k, v, table, width, 1), c);
}

// Future nonfinite K/V stays masked, valid Inf V stays Inf, and adjacent sequences stay isolated. At head dim 256 V is
// random, so the FP32 fallback's dims can be told apart, and tiles of their own carry non-finite V in dims 128-255.
static void check_nonfinite_prefill(side& cpu, side& dut, int hd) {
  const shape c{hd, 1, 1, {16, 49}, {0, 0}, 0};
  constexpr int rows = 65, width = 5, blocks = 8;
  const int row_ld = hd + 3;
  bytes q(rows * row_ld * 2, 0xff), k(blocks * 16 * 2 * hd * 2, 0), v(k.size(), 0);
  const auto put = [](bytes& data, size_t element, uint16_t half) {
    data[element * 2] = uint8_t(half);
    data[element * 2 + 1] = uint8_t(half >> 8);
  };
  for (int r = 0; r < rows; ++r)
    for (int d = 0; d < hd; ++d) put(q, size_t(r * row_ld + d), 0x3c00);
  for (size_t i = 0; i < v.size() / 2; ++i) put(v, i, 0x3c00);
  if (hd == 256) {
    rng g(4245);
    v = halves(v.size() / 2, g);
  }
  const size_t layer = size_t(blocks * 16 * hd);
  put(k, layer + size_t(15 * hd + 2), 0x7e00);
  put(v, layer + size_t(15 * hd), 0x7e00);
  put(v, layer + size_t(16 * hd + 1), 0x7c00);
  // Sequence 1's position p is slot 16 + p. One non-finite V flags its whole tile, whichever copies the scan reads, so
  // these stay off the tiles above; only the scan of dims 128-255 sends them to the fallback, and rows 16-19 and 32-39
  // of the sequence, which mask them, would otherwise take 0 x V = NaN from the panels.
  if (hd == 256) {
    put(v, layer + size_t((16 + 20) * hd + 150), 0x7c00);
    put(v, layer + size_t((16 + 40) * hd + 255), 0x7e00);
  }
  std::vector<int32_t> table((width + 1) * 2, -1);
  table[0] = 0;
  for (int b = 0; b < 4; ++b) table[width + 1 + b] = b + 1;
  compare(run(cpu, c, q, k, v, table, width, rows), run(dut, c, q, k, v, table, width, rows), c);
}

// NaN, +Inf and -Inf in K inside rows' visible windows, across GQA heads, ragged sequences and a
// straddled tile: a -Inf score drops its position; NaN or +Inf makes every row that sees it NaN. At head dim 256 the
// keys sit in dims 128-130, the second half of the head, which the HD256 prefill's QK steps reach last.
static void check_nonfinite_keys(side& cpu, side& dut, int hd) {
  const shape c{hd, 1, 2, {24, 20, 21}, {16, 16, 16}, 0};
  constexpr int rows = 65, width = 3, blocks = 9, heads = 2;
  const int row_ld = heads * hd + 3, d0 = hd - 128;
  rng g(4242);
  bytes q = halves(size_t(rows * row_ld), g), k = halves(size_t(blocks * 16 * 2 * hd), g);
  const bytes v = halves(k.size() / 2, g);
  const auto put = [](bytes& data, size_t element, uint16_t half) {
    data[element * 2] = uint8_t(half);
    data[element * 2 + 1] = uint8_t(half >> 8);
  };
  // Positive q on dims 1 and 2 keeps -Inf keys at -Inf scores and +Inf keys at +Inf.
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < heads; ++h)
      for (int d : {1, 2}) put(q, size_t(r * row_ld + h * hd + d0 + d), 0x3c00);
  const auto key = [&](int seq, int pos, int d, uint16_t half) {
    put(k, size_t(blocks * 16 * hd + ((seq * width + pos / 16) * 16 + pos % 16) * hd + d0 + d), half);
  };
  key(0, 3, 0, 0x7e00);   // NaN every row of the first sequence sees
  key(1, 7, 1, 0xfc00);   // -Inf: weight zero, rows stay finite
  key(1, 21, 0, 0x7e00);  // NaN seen from the sixth row on
  key(2, 4, 1, 0xfc00);
  key(2, 33, 2, 0x7c00);  // +Inf seen only by the last four rows
  std::vector<int32_t> table(size_t(3 * (width + 1)), -1);
  for (int s = 0; s < 3; ++s)
    for (int b = 0; b < width; ++b) table[size_t(s * (width + 1) + b)] = s * width + b;
  compare(run(cpu, c, q, k, v, table, width, rows), run(dut, c, q, k, v, table, width, rows), c);
}

// Decode over non-finite K/V: NaN, +Inf and -Inf V reach every row that sees them, also through the split merge, a NaN
// K makes only its kv head's rows NaN, and the row of length 20 never sees the NaN V at 20 on its last page. Table
// entries past a row's pages hold INT32_MIN, which puts a K/V address below zero: following one faults (16 TB past the
// pool, where INT32_MAX lands, a WSL 5070 Ti reads zeros instead).
// ponytail: the zero-fill check holds only where the platform faults at pool - 16 TB; run this case under
// compute-sanitizer memcheck where that runs, which flags any read past the pool.
static void check_nonfinite_decode(side& cpu, side& dut, int hd, int group) {
  constexpr int kv = 2, width = 376;
  const int row_ld = kv * group * hd + 3;
  const std::vector<int32_t> lens{0, 20, 21, 701, 5000, 5001, 6001};
  const int rows = int(lens.size());
  rng g(4244);
  const bytes q = halves(size_t(rows * row_ld), g);
  bytes k = halves(size_t(width * 16 * 2 * kv * hd), g), v = halves(k.size() / 2, g);
  const auto put = [](bytes& data, size_t element, uint16_t half) {
    data[element * 2] = uint8_t(half);
    data[element * 2 + 1] = uint8_t(half >> 8);
  };
  // Every row reads pages 0, 1, 2, ..., so position p is slot p of the pool's layer 1.
  const auto at = [&](int pos, int head, int d) { return size_t(width * 16 * kv * hd + (pos * kv + head) * hd + d); };
  for (int h = 0; h < kv; ++h) {
    put(v, at(20, h, 3), 0x7e00);
    put(v, at(700, h, 9), 0x7c00);
    put(v, at(5000, h, 100), 0xfc00);
  }
  // Dims 128-255 at head dim 256, on pages of their own: one non-finite V flags its whole page, whichever copies the
  // scan reads. Infinite because a NaN V comes out NaN from the panels too; there, a negative low part makes
  // Inf - Inf = NaN.
  if (hd == 256)
    for (int h = 0; h < kv; ++h) {
      put(v, at(300, h, 155), 0x7c00);
      put(v, at(2000, h, 252), 0xfc00);
    }
  put(k, at(6000, 1, 0), 0x7e00);
  std::vector<int32_t> table(size_t(rows * (width + 1)), std::numeric_limits<int32_t>::min());
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < (lens[size_t(r)] + 15) / 16; ++b) table[size_t(r * (width + 1) + b)] = b;
  for (uint32_t splits : {0u, 1u, 3u, 8u}) {
    const shape c{hd, kv, group, lens, {}, splits};
    compare(run(cpu, c, q, k, v, table, width, rows), run(dut, c, q, k, v, table, width, rows), c);
  }
}

// The first 33 positions every row sees key -Inf, so they weigh zero as under the reference's global max, though they
// fill whole tensor-core tiles, decode splits and token warps' first positions. A row that sees only -Inf is NaN there,
// and so is the dim where an Inf V meets a zero weight, also in a split of -Inf scores alone.
static void check_leading_minus_inf(side& cpu, side& dut, int hd, int heads) {
  constexpr int minus = 33, width = 38, blocks = 41;
  const int row_ld = heads * hd + 3;
  rng g(4243);
  bytes k = halves(size_t(blocks * 16 * 2 * hd), g), v = halves(k.size() / 2, g);
  const auto put = [](bytes& data, size_t element, uint16_t half) {
    data[element * 2] = uint8_t(half);
    data[element * 2 + 1] = uint8_t(half >> 8);
  };
  const size_t layer = size_t(blocks * 16 * hd);
  for (int p = 0; p < minus; ++p) put(k, layer + size_t(p * hd + 1), 0xfc00);
  put(v, layer + size_t(5 * hd + 3), 0x7c00);
  const auto attend = [&](const shape& c, int rows) {
    // Positive q on dim 1 keeps the -Inf keys at -Inf scores. Every sequence reads pages 0, 1, 2, ...
    bytes q = halves(size_t(rows * row_ld), g);
    for (int r = 0; r < rows; ++r)
      for (int h = 0; h < heads; ++h) put(q, size_t(r * row_ld + h * hd + 1), 0x3c00);
    std::vector<int32_t> table(c.lens.size() * (width + 1), -1);
    for (size_t s = 0; s < c.lens.size(); ++s)
      for (int b = 0; b < width; ++b) table[s * (width + 1) + size_t(b)] = b;
    compare(run(cpu, c, q, k, v, table, width, rows), run(dut, c, q, k, v, table, width, rows), c);
  };
  // 40 cached positions before the first sequence's rows; the second starts at 0, so its first 33 rows see only -Inf.
  attend({hd, 1, heads, {70, 50}, {40, 0}, 0}, 120);
  for (uint32_t splits : {0u, 1u, 3u, 8u}) attend({hd, 1, heads, {40, 600, 17, 0, 1}, {}, splits}, 5);
}

// 128 heads as kv x group: 1 x 128 takes the CUDA-core decode, 16 x 8 the tensor-core one where it routes; the pool
// keeps one size (160 / kv pages).
static void check_capture_exhaustion(side& cpu, side& dut, int kv, int group) {
  larch_device_props props{};
  require(larch_device_props_get(dut.d, &props), dut.d);
  // HIP uses the general decode kernel without split scratch; this fixture targets CUDA's arena.
  if (props.backend != LARCH_BACKEND_CUDA || !(props.flags & LARCH_CAP_GRAPHS)) return;
  constexpr int rows = 16, heads = 128, hd = 128, row_width = heads * hd;
  const int width = 160 / kv;
  const shape c{hd, kv, group, std::vector<int32_t>(rows, 1), {}, 1};
  rng g(911);
  const bytes q = halves(size_t(rows * (row_width + 3)), g);
  const bytes k = halves(size_t(width * 16 * 2 * kv * hd), g), v = halves(k.size() / 2, g);
  std::vector<int32_t> table(rows * (width + 1), -1);
  for (int r = 0; r < rows; ++r) table[size_t(r * (width + 1))] = 0;
  const bytes reference = run(cpu, c, q, k, v, table, width, rows);
  tensor qt(dut, LARCH_F16, {rows, heads, hd}, q, 3);
  tensor kt(dut, LARCH_F16, {int64_t(k.size() / 2)}, k);
  tensor vt(dut, LARCH_F16, {int64_t(v.size() / 2)}, v);
  require(larch_upload(dut.s, kt.t.buf, 0, k.data(), k.size()), dut.d);
  require(larch_upload(dut.s, vt.t.buf, 0, v.data(), v.size()), dut.d);
  std::copy(k.begin(), k.end(), kt.before.begin());
  std::copy(v.begin(), v.end(), vt.before.begin());
  tensor bt(dut, LARCH_I32, {rows, width}, raw(table), 1);
  tensor lt(dut, LARCH_I32, {rows}, raw(c.lens));
  tensor ot(dut, LARCH_F16, {rows, heads, hd}, bytes(q.size(), 0xff), 3);
  const larch_kv_pool pool{kt.t.buf, vt.t.buf, uint32_t(width), 16, 2, uint32_t(kv), hd, LARCH_F16, 1};
  const auto invoke = [&](uint32_t splits) {
    return larch_attn_decode(dut.s, &qt.t, &pool, &bt.t, &lt.t, splits, 1.0f / std::sqrt(float(hd)), &ot.t);
  };
  const auto compare_output = [&] {
    const bytes out = ot.read(true);
    bytes dense(size_t(rows * row_width * 2));
    for (int r = 0; r < rows; ++r) {
      std::copy_n(out.begin() + int64_t(ot.off) + int64_t(r) * (row_width + 3) * 2, row_width * 2,
                  dense.begin() + int64_t(r) * row_width * 2);
      for (int d = row_width * 2; d < (row_width + 3) * 2; ++d)
        if (out[ot.off + size_t(r * (row_width + 3) * 2 + d)] != 0xff) std::exit(1);
    }
    compare(reference, dense, c);
  };
  require(larch_stream_sync(dut.s), dut.d);
  require(larch_graph_begin(dut.s), dut.d);
  require(invoke(1), dut.d);  // A partial graph exists before the rejected operation.
  // 16*128*64*130*4 = 68,157,440 scratch bytes, rejected before allocation or launch.
  // Actual Q/K/V/output buffers total less than4MiB; no oversized eager scratch is allocated.
  if (invoke(64) != LARCH_ERR_UNSUPPORTED) {
    std::fprintf(stderr, "attention: oversized captured scratch was not rejected\n");
    std::exit(1);
  }
  larch_graph* graph = nullptr;
  require(larch_graph_end(dut.s, &graph), dut.d);
  larch_graph_destroy(graph);  // The caller discards partial work rather than replaying it.
  if (ot.read(true) != ot.before) {
    std::fprintf(stderr, "attention: discarded partial capture executed output work\n");
    std::exit(1);
  }
  require(invoke(1), dut.d);
  compare_output();  // Eager continuation must remain usable after the host-side rejection.
  require(larch_graph_begin(dut.s), dut.d);
  require(invoke(3), dut.d);
  require(larch_graph_end(dut.s, &graph), dut.d);
  require(larch_graph_launch(graph, dut.s), dut.d);
  compare_output();
  larch_graph_destroy(graph);
  qt.read(); kt.read(); vt.read(); bt.read(); lt.read();
}

static void check_high_finite(side& cpu,side& dut,int rows=17,int hd=128){
  const shape c{hd,2,4,{rows},{31},0};
  constexpr int heads=8,kv=2;
  const int row_width=heads*hd;
  const int width=(rows+31+15)/16,blocks=width+3;
  bytes q(size_t(rows*(row_width+3)*2),0),k(size_t(blocks*16*2*kv*hd*2),0),v(k.size(),0);
  const auto fill=[](bytes& b,uint16_t value){for(size_t i=0;i<b.size();i+=2){b[i]=uint8_t(value);b[i+1]=uint8_t(value>>8);}};
  fill(q,0x7bff);fill(k,0x7bff);fill(v,0x3c00);
  std::vector<int32_t> table(width+1,-1);for(int b=0;b<width;++b)table[b]=b;
  compare(run(cpu,c,q,k,v,table,width,rows),run(dut,c,q,k,v,table,width,rows),c);
  std::puts("high finite QK dot and output PASS");
}


// Finite cancellation: half probability equality must not erase a small FP32 difference.
static void check_probability_cancellation(side& cpu, side& dut) {
  const shape c{128,2,4,{65},{2},0};
  constexpr int rows=65,width=6,blocks=9,heads=8,kv=2,hd=128,row_width=heads*hd;
  bytes q(size_t(rows*(row_width+3)*2),0),k(size_t(blocks*16*2*kv*hd*2),0),v(k.size(),0);
  const auto put=[](bytes& data,size_t i,uint16_t value){data[i*2]=uint8_t(value);data[i*2+1]=uint8_t(value>>8);};
  for(int r=0;r<rows;++r)for(int h=0;h<heads;++h)put(q,size_t(r*(row_width+3)+h*hd),0x3c00);
  const size_t layer=size_t(blocks*16*kv*hd);
  for(int h=0;h<kv;++h){
    put(k,layer+size_t(h*hd),0x1419);  // Nearest F16 to 0.001.
    for(int d=0;d<hd;++d){
      put(v,layer+size_t(h*hd+d),0x7bff);
      put(v,layer+size_t(kv*hd+h*hd+d),0xfbff);
    }
  }
  std::vector<int32_t> table(width+1,-1);for(int b=0;b<5;++b)table[b]=b;
  compare(run(cpu,c,q,k,v,table,width,rows),run(dut,c,q,k,v,table,width,rows),c);
  std::puts("finite probability cancellation CPU oracle PASS");
}


// Very small probabilities are nonzero in FP32 even when both half panels would round to zero.
// Every row count takes the tensor-core kernel where it runs (the sm_120 default, LARCH_PREFILL_TC=1 elsewhere);
// with LARCH_PREFILL_TC=0, fewer than 64 rows take attend_tiled<128,4,2> and 64 the sm_120-only 8-row variants.
// Positions past `keyed` get a key whose weight is exactly zero.
static void check_probability_underflow(side& cpu, side& dut, uint16_t key=0xbe5d, uint16_t dominant_v=0, int rows=64,
                                        int keyed=2559) {
  const shape c{128,1,4,{rows},{2560-rows},0};
  constexpr int width=161,blocks=164,heads=4,hd=128,row_width=heads*hd;
  bytes q(size_t(rows*(row_width+3)*2),0),k(size_t(blocks*16*2*hd*2),0),v(k.size(),0);
  const auto put=[](bytes& data,size_t i,uint16_t value){data[i*2]=uint8_t(value);data[i*2+1]=uint8_t(value>>8);};
  for(int r=0;r<rows;++r)for(int d=0;d<row_width;++d)put(q,size_t(r*(row_width+3)+d),0x3c00);
  const size_t layer=size_t(blocks*16*hd);
  for(int pos=1;pos<2560;++pos)for(int d=0;d<hd;++d){
    put(k,layer+size_t(pos*hd+d),pos<=keyed?key:0xfbff);
    put(v,layer+size_t(pos*hd+d),0x7bff);
  }
  for(int d=0;d<hd;++d)put(v,layer+size_t(d),dominant_v);
  std::vector<int32_t> table(width+1,-1);for(int b=0;b<160;++b)table[b]=b;
  compare(run(cpu,c,q,k,v,table,width,rows),run(dut,c,q,k,v,table,width,rows),c);
  // Decode: a row of all 2560 positions and one of the dominant alone. Split, the pages after the first sit far below
  // the max of the split holding position 0, and the other splits hold the small weights alone.
  std::vector<int32_t> decode_table(size_t(2*(width+1)),-1);for(int b=0;b<160;++b)decode_table[size_t(b)]=b;
  decode_table[size_t(width+1)]=0;
  for(uint32_t splits:{0u,3u}){
    const shape d{128,1,4,{2560,1},{},splits};
    compare(run(cpu,d,q,k,v,decode_table,width,2),run(dut,d,q,k,v,decode_table,width,2),d);
  }
}


// m1-shaped sequences (32 q / 8 kv heads, hd 128; at hd 256, 16 q / 2 kv; page 16, f16 pool): `rows` queries split
// equally over `seqs` sequences, each after `hist` cached tokens on its own pages. The pool spans >= 4x L2 and replayed calls cycle
// through its layers (a capture touches min(calls, layers) of them), so K/V mostly come from DRAM; attention at
// these shapes is compute-bound, so a few L2 hits would move the number little.
static double bench_prefill(side& x, const larch_device_props& props, int rows, int hist, int seqs, int hd) {
  const int heads = hd == 256 ? 16 : 32, kv = hd == 256 ? 2 : 8;
  const int width = (hist + rows / seqs + 15) / 16;
  const uint64_t layer_halves = uint64_t(seqs) * width * 16 * kv * hd;
  const int layers = int(std::max<uint64_t>(1, (4 * props.l2_bytes + layer_halves * 4 - 1) / (layer_halves * 4)));
  rng g(1234);
  // q and k up to 8 spread the scores like a real model's, so the bench takes serving's numeric paths.
  const bytes keys = halves(layer_halves, g, 3), values = halves(layer_halves, g);
  const bytes q = halves(size_t(rows) * heads * hd, g, 3);
  std::vector<larch_buffer*> buffers;
  const auto alloc = [&](uint64_t size, const void* data) {
    larch_buffer* b = nullptr;
    require(larch_buffer_alloc(x.d, size, &b), x.d);
    buffers.push_back(b);
    if (data) require(larch_upload(x.s, b, 0, data, size), x.d);
    return b;
  };
  larch_buffer *k = alloc(keys.size() * layers, nullptr), *v = alloc(values.size() * layers, nullptr);
  for (int l = 0; l < layers; ++l) {
    require(larch_upload(x.s, k, uint64_t(l) * keys.size(), keys.data(), keys.size()), x.d);
    require(larch_upload(x.s, v, uint64_t(l) * values.size(), values.data(), values.size()), x.d);
  }
  std::vector<int32_t> table(size_t(seqs) * width, 0);
  std::iota(table.begin(), table.end(), 0);
  const std::vector<int32_t> lengths(size_t(seqs), rows / seqs), firsts(size_t(seqs), hist);
  const larch_tensor qt{alloc(q.size(), q.data()), 0, LARCH_F16, 3, {rows, heads, hd}, {heads * hd, hd, 1}};
  const larch_tensor ot{alloc(q.size(), nullptr), 0, LARCH_F16, 3, {rows, heads, hd}, {heads * hd, hd, 1}};
  const larch_tensor bt{alloc(table.size() * 4, table.data()), 0, LARCH_I32, 2, {seqs, width}, {width, 1}};
  const larch_tensor lt{alloc(uint64_t(seqs) * 4, lengths.data()), 0, LARCH_I32, 1, {seqs}, {1}};
  const larch_tensor st{alloc(uint64_t(seqs) * 4, firsts.data()), 0, LARCH_I32, 1, {seqs}, {1}};
  const float scale = 1.0f / std::sqrt(float(hd));
  const double us = replay_time(x.s, [&](int i) {
    const larch_kv_pool pool{
        k, v, uint32_t(seqs * width), 16, uint32_t(layers), uint32_t(kv), uint32_t(hd), LARCH_F16, uint32_t(i % layers)};
    return larch_attn_prefill(x.s, &qt, &pool, &bt, &st, &lt, scale, &ot);
  });
  if (us < 0) {
    std::fprintf(stderr, "attention bench failed: %s\n", larch_last_error(x.d));
    std::exit(1);
  }
  for (larch_buffer* b : buffers) larch_buffer_free(b);
  return us;
}

// Layers of the decode bench's pool: enough that the K/V one call reads span >= 4x L2, so replays cycling through
// them mostly read DRAM.
static uint64_t decode_layers(const larch_device_props& props, int kv, int hd, const std::vector<int32_t>& lengths) {
  const uint64_t read = uint64_t(std::accumulate(lengths.begin(), lengths.end(), int64_t(0))) * kv * hd;
  return std::max<uint64_t>(1, (4 * props.l2_bytes + read * 4 - 1) / (read * 4));
}

// Decode at `heads`/`kv` and head dim `hd` (page 16, f16 pool): one sequence a row of lengths[r] positions, each on its own
// contiguous blocks of a pool with `width` pages of capacity, as in serving, decode_layers deep. Returns microseconds
// per call.
static double bench_decode(side& x, const larch_device_props& props, int heads, int kv, int hd,
                           const std::vector<int32_t>& lengths, int width) {
  const int rows = int(lengths.size()), context = *std::max_element(lengths.begin(), lengths.end());
  const uint64_t per_row = uint64_t(context + 15) / 16, used = uint64_t(rows) * per_row;
  const uint64_t blocks = std::max<uint64_t>(used, uint64_t(width)), touched = used * 16 * kv * hd;
  const uint64_t layers = decode_layers(props, kv, hd, lengths);
  const uint64_t layer_bytes = blocks * 16 * kv * hd * 2;
  rng g(1234);
  // q, K and V up to 8 spread the scores like a real model's, so the bench takes serving's numeric paths.
  const bytes keys = halves(touched, g, 3), values = halves(touched, g, 3);
  const bytes q = halves(size_t(rows) * heads * hd, g, 3);
  std::vector<larch_buffer*> buffers;
  const auto alloc = [&](uint64_t size, const void* data) {
    larch_buffer* b = nullptr;
    require(larch_buffer_alloc(x.d, size, &b), x.d);
    buffers.push_back(b);
    if (data) require(larch_upload(x.s, b, 0, data, size), x.d);
    return b;
  };
  larch_buffer *k = alloc(layer_bytes * layers, nullptr), *v = alloc(layer_bytes * layers, nullptr);
  for (uint64_t l = 0; l < layers; ++l) {
    require(larch_upload(x.s, k, l * layer_bytes, keys.data(), keys.size()), x.d);
    require(larch_upload(x.s, v, l * layer_bytes, values.data(), values.size()), x.d);
  }
  std::vector<int32_t> table(size_t(rows) * width, 0);
  for (int r = 0; r < rows; ++r)
    std::iota(table.begin() + int64_t(r) * width, table.begin() + int64_t(r) * width + int64_t(per_row),
              int32_t(uint64_t(r) * per_row));
  const larch_tensor qt{alloc(q.size(), q.data()), 0, LARCH_F16, 3, {rows, heads, hd}, {heads * hd, hd, 1}};
  const larch_tensor ot{alloc(q.size(), nullptr), 0, LARCH_F16, 3, {rows, heads, hd}, {heads * hd, hd, 1}};
  const larch_tensor bt{alloc(table.size() * 4, table.data()), 0, LARCH_I32, 2, {rows, width}, {width, 1}};
  const larch_tensor lt{alloc(uint64_t(rows) * 4, lengths.data()), 0, LARCH_I32, 1, {rows}, {1}};
  const float scale = 1.0f / std::sqrt(float(hd));
  const auto once = [&] {
    return replay_time(x.s, [&](int i) {
      const larch_kv_pool pool{k, v, uint32_t(blocks), 16, uint32_t(layers), uint32_t(kv), uint32_t(hd), LARCH_F16,
                                uint32_t(uint64_t(i) % layers)};
      return larch_attn_decode(x.s, &qt, &pool, &bt, &lt, 0, scale, &ot);
    });
  };
  // The host builds K/V for seconds with the GPU idle, and the driver then drops the memory clock (13801 to
  // 7001 MHz on the 5070 Ti) and takes 70 to 270 ms of load to raise it again, longer than replay_time's ~50 ms
  // window: points came out 16-42 % slow depending on the idle gap, not on allocation history. Serving never
  // idles, so time at full clocks.
  // ponytail: 1 s of discarded replays is ~4x the longest ramp seen; lengthen it if a point reads slow again.
  for (const auto t0 = std::chrono::steady_clock::now();
       std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1);)
    once();
  const double us = once();
  if (us < 0) {
    std::fprintf(stderr, "attention bench failed: %s\n", larch_last_error(x.d));
    std::exit(1);
  }
  for (larch_buffer* b : buffers) larch_buffer_free(b);
  return us;
}

// attention --bench decode [hd256] [heads kv rows context [width]]: the m1 (32/8) and m5 (28/4) heads (hd256: 16/2 at
// head dim 256) x rows 1..32 x context 1k..16k grid, or one point; context `lo-hi` is a ragged batch of two or more rows, row r of lo + (hi - lo)
// * r / (rows - 1) positions, and width defaults to serving's table (at least 8192 positions). GB/s count the summed
// lengths. Exit 2 on bad arguments, also where the host K and V or the device pool exceed 8 GiB (a small call or a
// wide table takes many or large layers: tiny shapes asked for terabytes). LARCH_DECODE_TC is read at library load,
// so the header echoes it; unset is arch.h's decode_tc default, at hd256 as at hd 128. Run only with the
// exclusive GPU lock.
static int bench_decode_cli(side& x, int argc, char** argv, int hd) {
  if (argc != 3 && argc != 7 && argc != 8) return 2;
  larch_device_props props{};
  require(larch_device_props_get(x.d, &props), x.d);
  struct point {
    int heads, kv, rows, lo, context, width;
  };
  const auto serving_width = [](int context) { return (std::max(context, 8192) + 15) / 16; };
  const auto lengths_of = [](int rows, int lo, int context) {
    std::vector<int32_t> lengths(size_t(rows), 0);
    for (int r = 0; r < rows; ++r) lengths[size_t(r)] = int32_t(lo + int64_t(context - lo) * r / std::max(1, rows - 1));
    return lengths;
  };
  std::vector<point> points;
  if (argc == 3) {
    for (const auto [heads, kv] : hd == 256 ? std::vector{std::pair{16, 2}} : std::vector{std::pair{32, 8}, std::pair{28, 4}})
      for (int rows : {1, 8, 16, 32})
        for (int context : {1024, 2048, 4096, 8192, 16384})
          points.push_back({heads, kv, rows, context, context, serving_width(context)});
  } else {
    long n[5] = {}, lo = 0;
    bool ragged = false;
    for (int i = 3; i < argc; ++i) {
      char* end = nullptr;
      n[i - 3] = std::strtol(argv[i], &end, 10);
      if (i == 6 && end != argv[i] && *end == '-') {
        ragged = true;
        lo = n[3];
        const char* hi = end + 1;
        n[3] = std::strtol(hi, &end, 10);
        if (end == hi) return 2;
      }
      if (end == argv[i] || *end || n[i - 3] < 1 || n[i - 3] > 131072) return 2;
    }
    if (!ragged) lo = n[3];
    if (argc == 7) n[4] = serving_width(int(n[3]));
    // One row would time lo alone.
    if (n[0] % n[1] || n[4] * 16 < n[3] || lo < 1 || lo > n[3] || (n[2] == 1 && lo != n[3])) return 2;
    // bench_decode's host K and V hold the rows' pages, its pool 2 x decode_layers x max(those pages, width) of them:
    // either over 8 GiB is K's share over 4 GiB.
    const int64_t page_bytes = int64_t(n[1]) * hd * 2 * 16, pages = int64_t(n[2]) * ((n[3] + 15) / 16);
    const int64_t half = int64_t(4) << 30;
    const int64_t layers = int64_t(decode_layers(props, int(n[1]), hd, lengths_of(int(n[2]), int(lo), int(n[3]))));
    if (pages * page_bytes > half || std::max(pages, int64_t(n[4])) * page_bytes > half / layers) return 2;
    points.push_back({int(n[0]), int(n[1]), int(n[2]), int(lo), int(n[3]), int(n[4])});
  }
  const char* tc = std::getenv("LARCH_DECODE_TC");
  std::printf("attention decode hd%d page16 f16, LARCH_DECODE_TC=%s, cold K/V, graph replay, microseconds\n"
              "| heads | rows | context | width | us | GB/s |\n|---|---:|---:|---:|---:|---:|\n", hd,
              tc ? tc : "unset");
  for (const auto& [heads, kv, rows, lo, context, width] : points) {
    const std::vector<int32_t> lengths = lengths_of(rows, lo, context);
    const double us = bench_decode(x, props, heads, kv, hd, lengths, width);
    std::printf("| %d/%d | %d | ", heads, kv, rows);
    if (lo != context) std::printf("%d-", lo);
    std::printf("%d | %d | %.2f | %.1f |\n", context, width, us,
                double(std::accumulate(lengths.begin(), lengths.end(), int64_t(0))) * kv * hd * 2 * 2 /
                    (us * 1e3));
    std::fflush(stdout);
  }
  return 0;
}

// attention --bench [decode] [hd256] ...: hd256 times head dim 256 at 16 q / 2 kv heads instead of hd 128.
// attention --bench [rows hist [seqs]]: the W1 grid (rows 256/512 x hist 1k..8k, one sequence) or one point, its
// rows split equally over seqs (default 1); exit 2 on bad arguments.
// The dispatch (LARCH_PREFILL_TC: 1 the tensor-core kernel, 0 the CUDA-core one, unset arch.h's sm_120 default
// by rows and sequences, at hd256 by arch) is fixed at library load, so the header echoes the
// environment. Run only with the exclusive GPU lock.
static int bench(side& x, int argc, char** argv) {
  if (std::strcmp(argv[1], "--bench")) return 2;
  const bool decode = argc > 2 && !std::strcmp(argv[2], "decode");
  int hd = 128;
  if (const int at = decode ? 3 : 2; argc > at && !std::strcmp(argv[at], "hd256")) {
    hd = 256;
    std::copy(argv + at + 1, argv + argc, argv + at);  // drop the token: the parsers below read fixed positions
    --argc;
  }
  if (decode) return bench_decode_cli(x, argc, argv, hd);
  if (argc != 2 && argc != 4 && argc != 5) return 2;
  int seqs = 1;
  std::vector<std::pair<int, int>> points;
  if (argc == 2) {
    for (int rows : {256, 512})
      for (int hist : {1024, 2048, 4096, 8192}) points.push_back({rows, hist});
  } else {
    char *end_rows = nullptr, *end_hist = nullptr, *end_seqs = nullptr;
    const long rows = std::strtol(argv[2], &end_rows, 10), hist = std::strtol(argv[3], &end_hist, 10);
    const long n = argc == 5 ? std::strtol(argv[4], &end_seqs, 10) : 1;
    if (end_rows == argv[2] || *end_rows || end_hist == argv[3] || *end_hist || rows < 1 || rows > 4096 ||
        hist < 0 || hist > 131072 || (argc == 5 && (end_seqs == argv[4] || *end_seqs)) || n < 1 || rows % n)
      return 2;
    points.push_back({int(rows), int(hist)});
    seqs = int(n);
  }
  larch_device_props props{};
  require(larch_device_props_get(x.d, &props), x.d);
  const char* tc = std::getenv("LARCH_PREFILL_TC");
  std::printf("attention prefill %s page16 f16, LARCH_PREFILL_TC=%s, %d sequence(s), cold K/V, graph "
              "replay, microseconds\n| rows | hist | us |\n|---:|---:|---:|\n", hd == 256 ? "16q/2kv hd256" : "32q/8kv hd128",
              tc ? tc : "unset", seqs);
  for (const auto& [rows, hist] : points) {
    std::printf("| %d | %d | %.2f |\n", rows, hist, bench_prefill(x, props, rows, hist, seqs, hd));
    std::fflush(stdout);
  }
  return 0;
}

// The verify contract (larch.h) on causal spans: row j of each sequence is byte for byte larch_attn_decode of that
// row alone at the same nonzero splits with seq_len start + j + 1, on whichever decode the shape and the
// registration's LARCH_DECODE_TC take, and (auto splits too) within the f16 tolerance of the CPU reference. Spans of
// 2 to 9 rows at page (15, 16) and split (511) boundaries and past 4k, head dims 8 (attn_k), 64, 128 and 256, one and
// three sequences, the three-sequence ones also captured and replayed at moved boundaries. The CPU reference runs it
// too, so the logic itself is checked without a GPU. A GPU backend answers a mask (even the causal chain's), a window
// or sinks UNSUPPORTED and writes nothing, and so does CUDA's capture for partials past its arena.
static void check_spans(side& cpu, side& dut) {
  larch_device_props props{};
  require(larch_device_props_get(dut.d, &props), dut.d);
  const int positions[] = {0, 15, 16, 511, 4000}, kv = 2;
  const auto causal = [&](int hd, int group, const std::vector<int32_t>& start, const std::vector<int32_t>& lens,
                          uint32_t splits, bool graph) {
    shape c{hd, kv, group, lens, start, splits};
    c.graph = graph;
    c.q_pad = 0;
    c.span = true;
    return c;
  };
  std::vector<std::pair<std::vector<int32_t>, std::vector<int32_t>>> spans;  // (start, lens)
  for (int32_t len = 2; len <= 9; ++len) spans.push_back({{positions[len % 5]}, {len}});
  spans.push_back({{0, 511, 4000}, {9, 2, 5}});
  spans.push_back({{15, 16, 4000}, {3, 8, 7}});
  for (int hd : {8, 64, 128, 256})  // 8: attn_k, the fallback for other head dims
    for (int group : {1, 4, 8})
      for (const auto& [start, lens] : spans) {
        const int seqs = int(start.size()), heads = kv * group, row_width = heads * hd;
        int span = 0;
        for (int s = 0; s < seqs; ++s) span = std::max(span, start[size_t(s)] + lens[size_t(s)]);
        const int width = (span + 15) / 16 + 1, blocks = width + 3, rows = std::accumulate(lens.begin(), lens.end(), 0);
        rng g(uint32_t(97 * hd + 13 * group + span));
        const bytes q = halves(size_t(rows * row_width), g), k = halves(size_t(blocks * 16 * 2 * kv * hd), g),
                    v = halves(k.size() / 2, g);
        std::vector<int32_t> table(size_t(seqs * (width + 1)), -1);
        for (int s = 0; s < seqs; ++s) {
          std::vector<int32_t> permutation(static_cast<size_t>(blocks));
          std::iota(permutation.begin(), permutation.end(), 0);
          std::shuffle(permutation.begin(), permutation.end(), g);
          std::copy_n(permutation.begin(), width, table.begin() + s * (width + 1));
        }
        const bool graph = seqs > 1;
        // The reference ignores splits, and so does a CPU device under test.
        const bytes ref = run(cpu, causal(hd, group, start, lens, 0, false), q, k, v, table, width, rows);
        for (uint32_t splits : {1u, 4u, 16u, 0u}) {
          if (props.backend == LARCH_BACKEND_CPU && splits != 1) break;
          const bytes got = run(dut, causal(hd, group, start, lens, splits, graph), q, k, v, table, width, rows);
          compare(ref, got, {hd, kv, group, lens, start, splits});
          for (int s = 0, r = 0; s < seqs && splits; ++s)
            for (int j = 0; j < lens[size_t(s)]; ++j, ++r) {
              const bytes qr(q.begin() + r * row_width * 2, q.begin() + (r + 1) * row_width * 2);
              const std::vector<int32_t> one(table.begin() + s * (width + 1), table.begin() + (s + 1) * (width + 1));
              const bytes want =
                  run(dut, {hd, kv, group, {start[size_t(s)] + j + 1}, {}, splits}, qr, k, v, one, width, 1);
              if (std::memcmp(want.data(), got.data() + r * row_width * 2, want.size())) {
                std::fprintf(stderr, "attention: attn hd%d gqa%d splits%u seq%d row%d (position %d) differs from "
                             "attn_decode of that row\n", hd, group, splits, s, j, start[size_t(s)] + j);
                std::exit(1);
              }
            }
          if (props.backend == LARCH_BACKEND_CPU || !graph) continue;
          // The capture replayed at other boundaries and positions (lens reversed, starts rotated, inside the table)
          // gives the eager call's bytes there.
          if (props.flags & LARCH_CAP_GRAPHS) {
            shape moved = causal(hd, group, start, lens, splits, true);
            moved.replay_lens.assign(lens.rbegin(), lens.rend());
            moved.replay_start = {start.back()};
            moved.replay_start.insert(moved.replay_start.end(), start.begin(), start.end() - 1);
            const bytes eager = run(dut, causal(hd, group, moved.replay_start, moved.replay_lens, splits, false), q, k,
                                    v, table, width, rows);
            if (run(dut, moved, q, k, v, table, width, rows) != eager) {
              std::fprintf(stderr, "attention: attn hd%d gqa%d splits%u replayed at new boundaries differs from "
                           "eager\n", hd, group, splits);
              std::exit(1);
            }
          }
          if (hd != 128 || group != 4 || splits != 4) continue;
          for (const auto& [mask, window, sinks] :
               {std::tuple{true, 0u, false}, std::tuple{false, 4u, false}, std::tuple{false, 0u, true}}) {
            larch_status st = LARCH_OK;
            shape refused = causal(hd, group, start, lens, splits, false);
            refused.mask = mask, refused.window = window, refused.sinks = sinks, refused.status = &st;
            run(dut, refused, q, k, v, table, width, rows);
            if (st != LARCH_ERR_UNSUPPORTED) {
              std::fprintf(stderr, "attention: attn with mask %d, window %u, sinks %d returned %d, want UNSUPPORTED\n",
                           int(mask), window, int(sinks), int(st));
              std::exit(1);
            }
          }
        }
      }
  if (props.backend != LARCH_BACKEND_CUDA || !(props.flags & LARCH_CAP_GRAPHS)) return;
  // Partials of 1024 x 16 x 130 x 16 x 4 = 130 MiB do not fit the capture's arena. An orphan span_rows_k left in the
  // capture is not visible through larch.h; this holds the refusal and its unchanged bytes.
  constexpr int hd = 128, group = 8, rows = 1024, width = 65, blocks = 68;
  rng g(1024);
  const bytes q = halves(size_t(rows * kv * group * hd), g), k = halves(size_t(blocks * 16 * 2 * kv * hd), g),
              v = halves(k.size() / 2, g);
  std::vector<int32_t> table(size_t(width + 1), -1);
  std::iota(table.begin(), table.end() - 1, 0);
  larch_status st = LARCH_OK;
  shape big = causal(hd, group, {0}, {rows}, 16, true);
  big.status = &st;
  run(dut, big, q, k, v, table, width, rows);
  if (st != LARCH_ERR_UNSUPPORTED) {
    std::fprintf(stderr, "attention: a captured span past the arena returned %d, want UNSUPPORTED\n", int(st));
    std::exit(1);
  }
}

// The default route of the tensor-core prefill (common/arch.h, the function ops.cu calls): both kernels are correct, so
// only this table catches an inverted arch test or crowding inequality. CPU-only: it runs on every backend and registration.
static void check_route() {
  const struct {
    int sw, arch;
    int64_t rows, seqs;
    bool on;
  } cases[] = {{-1, 120, 256, 1, true}, {-1, 120, 64, 8, false}, {-1, 120, 16, 16, true},
               {-1, 86, 256, 1, true},  {-1, 86, 64, 8, false}, {-1, 89, 256, 1, true},
               {-1, 75, 256, 1, false},
               {1, 86, 256, 1, true},   {0, 120, 256, 1, false}};
  for (const auto& c : cases)
    if (arch::prefill_tc(c.sw, c.arch, c.rows, c.seqs) != c.on) {
      std::fprintf(stderr, "attention: prefill_tc(switch %d, sm_%d, %lld rows, %lld seqs) should be %s\n", c.sw, c.arch,
                   static_cast<long long>(c.rows), static_cast<long long>(c.seqs), c.on ? "on" : "off");
      std::exit(1);
    }
  // The tensor-core decode's default (arch.h): both decode kernels are correct, so only this table catches a bad arch list.
  const struct {
    int sw, arch;
    bool on;
  } decode[] = {{-1, 120, true}, {-1, 86, false}, {-1, 89, false}, {1, 86, true}, {0, 120, false}};
  for (const auto& c : decode)
    if (arch::decode_tc(c.sw, c.arch) != c.on) {
      std::fprintf(stderr, "attention: decode_tc(switch %d, sm_%d) should be %s\n", c.sw, c.arch, c.on ? "on" : "off");
      std::exit(1);
    }
  // quant.cu's multi-row mmvq cap (arch.h), checked here because this table runs without a GPU.
  const struct {
    int arch, rows;
  } mmvq[] = {{120, 16}, {89, 8}, {86, 4}, {75, 1}, {121, 1}};
  for (const auto& c : mmvq)
    if (arch::mmvq_rows_up_to(c.arch) != c.rows) {
      std::fprintf(stderr, "attention: mmvq_rows_up_to(sm_%d) should be %d\n", c.arch, c.rows);
      std::exit(1);
    }
  // The tensor-core routes need sm_80 code, not only an sm_80 device: the device's own SASS, else the top entry's PTX.
  const struct {
    std::vector<int> built;
    int cc, image;
  } images[] = {{{860, 890, 1200}, 1200, 1200}, {{750}, 1200, 750}, {{860}, 890, 860}};
  for (const auto& c : images)
    if (arch::cuda_image(c.built.data(), c.built.size(), c.cc) != c.image) {
      std::fprintf(stderr, "attention: cuda_image(%zu entries, %d) should be %d\n", c.built.size(), c.cc, c.image);
      std::exit(1);
    }
}

int main(int argc, char** argv) {
  check_route();
  int32_t count = 0;
  require(larch_device_count(&count), nullptr);
  const char* ordinal = std::getenv("LARCH_TEST_DEVICE");
  side cpu(-1), dut(ordinal ? std::atoi(ordinal) : count ? 0 : -1);
  if (argc > 1) return bench(dut, argc, argv);
  check_warp_merge(cpu, dut);
  larch_device_props props{};
  require(larch_device_props_get(dut.d, &props), dut.d);
  // Without a GPU the tensor-core registrations would compare the CPU with itself: report Not Run.
  for (const char* name : {"LARCH_PREFILL_TC", "LARCH_DECODE_TC"})
    if (const char* tc = std::getenv(name); tc && std::strcmp(tc, "1") == 0 && props.backend == LARCH_BACKEND_CPU) {
      std::printf("%s=1 needs a GPU device: NOT RUN\n", name);
      return 77;
    }
  // Where the shipped defaults are both CUDA-core dispatches, the pinned-off `attention` run is this run.
  if (!std::getenv("LARCH_PREFILL_TC") && !std::getenv("LARCH_DECODE_TC") && props.backend == LARCH_BACKEND_CUDA &&
      !arch::prefill_tc(-1, int(props.arch_code), 256, 1) && !arch::decode_tc(-1, int(props.arch_code))) {
    std::printf("sm_%u defaults to the CUDA-core prefill and decode, which attention (LARCH_PREFILL_TC=0, "
                "LARCH_DECODE_TC=0) runs: NOT RUN\n", unsigned(props.arch_code));
    return 77;
  }
  for (int hd : {8, 64, 128, 256, 257}) {
    // 257 covers the CUDA/HIP fallback for unusual head sizes; Metal refuses head_dim > 256 by
    // design (its ponytail marker in backend.mm), which ops.cpp's rule would count as a failure.
    if (hd > 256 && props.backend == LARCH_BACKEND_METAL) continue;
    for (uint32_t splits : {0u, 1u, 3u, 8u, 64u, UINT32_MAX})
      check(cpu, dut, {hd, 2, 7, {0, 1, 17, 513}, {}, splits});
    check(cpu, dut, {hd, 2, 7, {0, 7, 1, 5, 0, 3}, {0, 13, 37, 254, 0, 509}, 0});
  }
  check(cpu, dut, {128, 2, 4, {2049, 31, 0}, {}, 0});
  // Auto splits cap at 64: a table past 64 x 512 tokens must still attend to every position.
  check(cpu, dut, {64, 1, 2, {33000, 7}, {}, 0});
  // The same table at the tensor-core decode's shapes (64 splits of 516 tokens, so most start mid-page), then a
  // 32-row x 32-head batch whose long row wants 40 splits of 512 where the 16 MiB partial cap allows 31 (span 646).
  for (int group : {1, 8}) check(cpu, dut, {128, 2, group, {33000, 7}, {}, 0});
  std::vector<int32_t> lens(32);
  for (int r = 0; r < 32; ++r) lens[size_t(r)] = r == 9 ? 20000 : r * 61 % 1000;
  check(cpu, dut, {128, 4, 8, lens, {}, 0});
  // Short rows in a long block table, captured, then replayed with one long row: the CTAs take the replay's
  // live splits, not the table's (on 70 SMs 87 of them for a 72-CTA grid, so CTAs take several in turn).
  check(cpu, dut, {128, 2, 4, {7, 1, 3, 17, 2, 5, 9, 4}, {}, 0, false, true, {9001, 17, 33, 64, 9, 12, 100, 40}});
  check(cpu, dut, {128, 2, 1, {17, 33, 0}, {0, 100, 0}, 0});
  check(cpu, dut, {128, 2, 4, {17}, {}, 3, true});
  check(cpu, dut, {128, 2, 4, {0, 0}, {0, 37}, 0});
  check(cpu, dut, {128, 2, 4, {513, 1, 0}, {}, 0, false, true});
  for (int hd : {64, 128, 256})
    for (uint32_t splits : {1u, 3u, 8u, 64u, UINT32_MAX})
      check(cpu, dut, {hd, 2, 7, {0, 1, 17, 1537}, {}, splits});
  check(cpu, dut, {128, 2, 4, {1537, 1, 0}, {}, 3, false, true});
  check(cpu, dut, {128, 2, 4, {1537}, {}, 3, true});
  // Long partial page, ragged rows and GQA tails exercise internal, split2 and automatic merges.
  for (uint32_t splits : {2u, 0u})
    check(cpu, dut, {128, 2, 7, {8193, 517, 1, 0}, {}, splits});
  // Eight-query tiles cross cached sequence boundaries and end with one valid row.
  check(cpu, dut, {128, 2, 7, {0, 9, 1, 17, 0, 38}, {0, 31, 255, 509, 0, 1021}, 0, true});
  // On the measured sm_120, 64 head groups exercise the automatic three-split decode choice.
  check(cpu, dut, {128, 8, 4, {1537, 517, 63, 17, 1, 0, 61, 0}, {}, 0});
  // Token-warp decode coverage includes external splits, GQA tails and graph padding replay.
  check(cpu, dut, {128, 8, 4, {1537, 517, 63, 17, 1, 0, 61, 0}, {}, 2, false, true});
  check(cpu, dut, {128, 8, 4, {1537, 517, 63, 17, 1, 0, 61, 0}, {}, 0, true});
  check(cpu, dut, {128, 4, 7, {1537, 517, 63, 17, 1, 0, 61, 0, 1}, {}, 0});
  check(cpu, dut, {128, 4, 7, {1537, 517, 63, 17, 1, 0, 61, 0, 1}, {}, 2, true, true});
  // Captured at short contexts, replayed at longer ones: the engine's graphs keep one capture per
  // bucket while every sequence grows.
  check(cpu, dut, {128, 8, 4, {17, 1, 63, 0}, {}, 0, false, true, {1537, 600, 1024, 9}});
  check(cpu, dut, {64, 2, 7, {5, 17}, {}, 3, false, true, {2049, 300}});
  // The tensor-core decode's groups (to 8) at every split count, rows past 8k, a split starting mid-page; then head
  // dims 64 and 256 (64 keeps the CUDA-core decode), and captures replayed past 8k.
  for (int group : {1, 4, 7, 8})
    for (uint32_t splits : {0u, 1u, 3u, 8u, 64u}) check(cpu, dut, {128, 2, group, {0, 1, 17, 513, 8209}, {}, splits});
  for (int hd : {64, 256})
    for (int group : {4, 8})
      for (uint32_t splits : {0u, 3u}) check(cpu, dut, {hd, 2, group, {0, 1, 17, 513, 8209}, {}, splits});
  for (int group : {7, 8}) check(cpu, dut, {128, 2, group, {17, 1, 63, 0}, {}, 0, false, true, {8209, 600, 1024, 9}});
  // Full table capacity covers neighboring short-live lengths, external splits and graph padding.
  for (uint32_t splits : {0u, 1u, 3u})
    check(cpu, dut, {128, 2, 7, {0, 511, 512, 513, 1023, 1024, 1025, 1537, 1}, {}, splits,
                    false, splits == 3});
  for (int group : {1, 4, 7})
    check(cpu, dut, {128, 2, group, {0, 15, 1, 17, 0, 48}, {0, 31, 255, 509, 0, 1021}, 0});
  for (const auto& c : {shape{128, 2, 4, {64}, {448}, 0},
                        shape{128, 2, 4, {511}, {513}, 0},
                        shape{128, 2, 4, {513}, {2047}, 0},
                        shape{128, 2, 7, {17, 48}, {512, 1021}, 0, false, true}})
    check(cpu, dut, c);
  // Exercise the serving context boundary and the neighboring longer context.
  check(cpu, dut, {128, 2, 4, {64}, {2496}, 0, false, true});
  check(cpu, dut, {128, 2, 4, {64}, {2497}, 0});
  // 64-row tiles holding several sequences past 4k history, partial last pages, at m1's and m5's GQA.
  check(cpu, dut, {128, 2, 4, {70, 300, 64}, {4111, 17, 4000}, 0});
  check(cpu, dut, {128, 1, 7, {70, 300, 64}, {4111, 17, 4000}, 0});
  // Causal-diagonal tiles split off a multiple of 16, rows that see none of a tile, a last page ending mid-page and
  // V up to +-65504: masked positions and other sequences' rows must add exactly zero.
  check(cpu, dut, {128, 2, 4, {37, 29, 23}, {0, 5, 1000}, 0, false, false, {}, 16});
  // The multi-sequence and diagonal cases above with q 16-byte aligned: the prefill's 16-byte Q staging, its zero-fill
  // branch for rows past a tile's end included. Then only the offset, or only the rows, aligned: the element loads.
  check(cpu, dut, {128, 2, 4, {70, 300, 64}, {4111, 17, 4000}, 0, false, false, {}, 1, true, 8});
  check(cpu, dut, {128, 2, 4, {37, 29, 23}, {0, 5, 1000}, 0, false, false, {}, 16, true, 8});
  check(cpu, dut, {128, 2, 4, {37, 29, 23}, {0, 5, 1000}, 0, false, false, {}, 16, true, 3});
  check(cpu, dut, {128, 2, 4, {37, 29, 23}, {0, 5, 1000}, 0, false, false, {}, 16, false, 8});
  // Head dim 256 at 16 q / 2 kv heads (GQA 8), on its tensor-core kernels under attention_prefill_tc and
  // attention_decode_tc. Prefill: one sequence across several tiles (8 rows x 8 heads, or 4 x 8), ragged sequences past
  // 4k history, causal diagonals with V to +-65504, q 16-byte aligned (the staged loads), and a capture replayed at
  // other lengths and starts.
  for (const auto& c : {shape{256, 2, 8, {300}, {0}, 0}, shape{256, 2, 8, {70, 300, 64}, {4111, 17, 4000}, 0},
                        shape{256, 2, 8, {37, 29, 23}, {0, 5, 1000}, 0, false, false, {}, 16},
                        shape{256, 2, 8, {70, 300, 64}, {4111, 17, 4000}, 0, false, false, {}, 1, true, 8},
                        shape{256, 2, 8, {17, 48}, {512, 1021}, 0, false, true}})
    check(cpu, dut, c);
  // Decode from one position to 64 splits of 129 (8200), auto and pinned, one sequence and three (short rows next to a
  // long one keep 16 or fewer splits); pinned, every row of the three equals its one-row decode. Then a capture at
  // short contexts replayed past 8k, and tables of 128 and 144 positions (capacity at and past one 128-position split).
  for (uint32_t splits : {0u, 1u, 16u, 64u}) {
    for (int32_t n : {1, 127, 128, 511, 2048, 8200}) check(cpu, dut, {256, 2, 8, {n}, {}, splits});
    for (const auto& lens : {std::vector<int32_t>{1, 511, 8200}, std::vector<int32_t>{127, 128, 2048}}) {
      shape c{256, 2, 8, lens, {}, splits};
      c.one_row = splits != 0;
      check(cpu, dut, c);
    }
  }
  check(cpu, dut, {256, 2, 8, {17, 1, 63, 0}, {}, 0, false, true, {8209, 600, 1024, 9}});
  for (int32_t n : {112, 113}) check(cpu, dut, {256, 2, 8, {n, 1}, {}, 0});
  // The HD256 prefill's three paths on any GPU: at 8 x SMs rows the packed kernel's full tile (8 rows a CTA x 2 kv
  // heads: twice the SMs), at 3 x SMs its half tile (the full one's grid is under the SMs, the half one's over), and
  // the small cases above, whose half-tile grid is under the SMs, the per-head kernel. Then GQA groups that leave
  // virtual rows unused: 3 (6 q / 2 kv: 21 rows of 3 heads, one padding row a tile) and 48 (48 q / 1 kv: one row a
  // tile).
  const int32_t sms = int32_t(std::max(props.compute_units, 1u));
  check(cpu, dut, {256, 2, 8, {8 * sms}, {64}, 0});
  check(cpu, dut, {256, 2, 8, {3 * sms}, {64}, 0});
  check(cpu, dut, {256, 2, 3, {37, 29}, {0, 900}, 0});
  check(cpu, dut, {256, 1, 48, {5, 3}, {0, 700}, 0});
  // GQA 4 at head dim 256 (8 q / 2 kv heads), which the arch defaults route too: ragged prefill past 4k with q aligned,
  // and decode auto and at 16 splits next to a long row.
  check(cpu, dut, {256, 2, 4, {70, 300, 64}, {4111, 17, 4000}, 0, false, false, {}, 1, true, 8});
  for (uint32_t splits : {0u, 16u}) check(cpu, dut, {256, 2, 4, {1, 511, 8200}, {}, splits});
  for (int hd : {128, 256}) check_nonfinite_keys(cpu, dut, hd);
  check_high_finite(cpu, dut);
  check_high_finite(cpu, dut, 65);  // 65 rows: two tensor-core tiles under ctest attention_prefill_tc.
  check_high_finite(cpu, dut, 65, 256);
  // ponytail: Metal's attn_flash rounds P to f16 and multiplies masked columns through the matmul, so f32-only
  // probabilities and future nonfinite V are its ceiling; run these on Metal when f1 stages P in f32 and masks V
  // per row (issue #157). Leading -Inf scores are unchecked on attn_flash too, f1's: an issue for them; run
  // check_leading_minus_inf on Metal when f1 confirms they weigh zero there.
  if (props.backend != LARCH_BACKEND_METAL) {
    for (int hd : {128, 256, 8})  // 8: attn_k
      for (int heads : {2, 8}) check_leading_minus_inf(cpu, dut, hd, heads);
    for (int hd : {128, 256}) check_nonfinite_prefill(cpu, dut, hd);
    check_nonfinite_decode(cpu, dut, 128, 7);
    check_nonfinite_decode(cpu, dut, 256, 8);
    check_probability_cancellation(cpu, dut);
    check_probability_underflow(cpu, dut);
    check_probability_underflow(cpu, dut, 0xbe20);  // Just above the half zero midpoint.
    check_probability_underflow(cpu, dut, 0xbe21);  // Just below it.
    check_probability_underflow(cpu, dut, 0xbc3e, 0xe407);  // Nonzero high, lost residual, cancellation.
    // 64 weights near 3.3e-3 leave subnormal half residuals that must still count (ref 5.067).
    check_probability_underflow(cpu, dut, 0xb80b, 0xf2ba, 64, 64);
    check_probability_underflow(cpu, dut, 0xbe5d, 0, 63);
    check_probability_underflow(cpu, dut, 0xbc3e, 0xe407, 63);
    check_probability_underflow(cpu, dut, 0xb80b, 0xf2ba, 63, 64);
    // Scores near -28 (P about 7e-13): both half panels of an unscaled P round to zero (output about 1.2e-4).
    check_probability_underflow(cpu, dut, 0xc0f3);
  }
  check_capture_exhaustion(cpu, dut, 1, 128);
  check_capture_exhaustion(cpu, dut, 16, 8);
  check_spans(cpu, dut);
  std::puts("attention CPU oracle grid passed");
  return 0;
}
