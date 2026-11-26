// The fast CPU backend's mmvq_id (cpu::fast_mmvq_id, called directly on the CPU device's stream, -2) against
// the CPU reference (-1): every format it has and one it lacks (the reference's then), gate/up and the
// weighted down at qwen3moe's shapes, bias, thread counts that change nothing, an out-of-range id, K-quant
// blocks whose weights decode to 0 against cancelling x, x spanning 2^21 in a block and an infinite x, and a bias
// the weighted sum must keep; its pool's exceptions; then its read bandwidth at a qwen3moe decode step and its
// time at a 512-row prefill chunk.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <new>
#include <string>
#include <thread>

#include "testdata.h"

namespace cpu {
larch_status fast_mmvq_id(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status fast_mmvq_id_no_vnni(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                                   const larch_tensor*, const larch_tensor*, const larch_tensor*);
bool fast_has_vnni();
void fast_pool_run(int64_t n, int64_t grain, uint32_t threads, const std::function<void(int64_t, int64_t)>& f);
bool fast_decode_row(int dtype, const uint8_t* w, int64_t k, float* out);
void to_f32(larch_dtype t, const uint8_t* src, int64_t n, float* dst);
}  // namespace cpu

namespace {

struct dev {
  larch_device* d = nullptr;
  larch_stream* s = nullptr;
  bool fast = false;  // the CPU device: cpu::fast_mmvq_id itself, whether or not the device's table holds it
  bool no_vnni = false;  // ... or cpu::fast_mmvq_id_no_vnni, its AVX2 (NEON) kernel even where VNNI runs
};

dev open(int32_t ordinal) {
  dev x;
  x.fast = ordinal == -2;
  if (larch_device_open(ordinal, &x.d) != LARCH_OK || larch_stream_create(x.d, &x.s) != LARCH_OK) std::exit(1);
  return x;
}

struct arg {
  larch_dtype t;
  std::vector<int64_t> shape;
  bytes data;
};

// Runs mmvq_id on x's device with args w, x, ids, [bias], [weights], out; out's bytes, or empty on an error.
bytes call(const dev& x, const std::vector<arg>& args, bool bias, bool weighted, larch_status* status) {
  std::vector<larch_buffer*> b(args.size());
  std::vector<larch_tensor> t(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    const arg& a = args[i];
    larch_buffer_alloc(x.d, a.data.size(), &b[i]);
    larch_upload(x.s, b[i], 0, a.data.data(), a.data.size());
    t[i] = {b[i], 0, a.t, uint32_t(a.shape.size()), {}, {}};
    for (int64_t d = int64_t(a.shape.size()) - 1, stride = 1; d >= 0; --d) {
      t[i].shape[d] = a.shape[size_t(d)];
      t[i].stride[d] = stride;
      stride *= d + 1 == int64_t(a.shape.size()) ? a.shape[size_t(d)] / block_of(a.t) : a.shape[size_t(d)];
    }
  }
  const size_t o = args.size() - 1;
  *status = (x.no_vnni ? cpu::fast_mmvq_id_no_vnni : x.fast ? cpu::fast_mmvq_id : larch_mmvq_id)(x.s, &t[0], &t[1], &t[2], bias ? &t[3] : nullptr,
                                                         weighted ? &t[3 + bias] : nullptr, &t[o]);
  bytes got(args[o].data.size());
  larch_stream_sync(x.s);
  larch_download(x.s, b[o], 0, got.data(), got.size());
  larch_stream_sync(x.s);
  for (larch_buffer* p : b) larch_buffer_free(p);
  return got;
}

float elem(const bytes& b, size_t i, larch_dtype t) {
  if (t == LARCH_F32) {
    float f;
    std::memcpy(&f, &b[4 * i], 4);
    return f;
  }
  const uint16_t h = uint16_t(b[2 * i] | b[2 * i + 1] << 8);
  const uint32_t e = h >> 10 & 0x1F, m = h & 0x3FF;
  const float v = e ? std::ldexp(float(m | 0x400), int(e) - 25) : std::ldexp(float(m), -24);
  return h & 0x8000 ? -v : v;
}

int wrong = 0;

// args on both devices, compared under mmvq's tolerance: |dut - ref| <= rtol |ref| + atol max|ref|.
void compare(const std::string& name, const dev& ref, const dev& dut, const std::vector<arg>& args, bool bias,
             bool weighted, int64_t count, larch_dtype ot) {
  larch_status rs, ds;
  const bytes want = call(ref, args, bias, weighted, &rs), got = call(dut, args, bias, weighted, &ds);
  const double rtol = ot == LARCH_F32 ? 1e-3 : 2e-3, atol = 1e-3;
  double peak = 0, worst = 0;
  for (size_t i = 0; i < size_t(count); ++i) peak = std::max(peak, double(std::fabs(elem(want, i, ot))));
  for (size_t i = 0; i < size_t(count); ++i) {
    const double r = elem(want, i, ot), d = elem(got, i, ot);
    const double over = std::fabs(d - r) - (rtol * std::fabs(r) + atol * peak);
    if (!(over <= 0)) worst = std::max(worst, std::isnan(over) ? INFINITY : over);
  }
  if (rs != LARCH_OK || ds != LARCH_OK || worst != 0) {
    std::printf("WRONG %s: status %d/%d, worst excess %g\n", name.c_str(), rs, ds, worst);
    ++wrong;
  }
}

bytes raw(const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  return bytes(b, b + n);
}

// One case of random data.
void check(const dev& ref, const dev& dut, larch_dtype q, int64_t experts, int64_t rows, int64_t slots, int64_t n,
           int64_t k, bool per_slot, bool weighted, bool bias, larch_dtype ot) {
  const std::string name = std::string(DTYPE[q]) + " " + std::to_string(experts) + " x [" + std::to_string(n) + ", " +
                           std::to_string(k) + "], " + std::to_string(rows) + " rows of " + std::to_string(slots) +
                           (per_slot ? ", x per slot" : "") + (weighted ? ", weighted" : "") + (bias ? " + bias" : "") +
                           " -> " + DTYPE[ot];
  rng g(uint32_t(std::hash<std::string>{}(name)));
  std::vector<int32_t> ids(size_t(rows * slots));
  for (int32_t& e : ids) e = int32_t(g() % uint32_t(experts));
  bytes idb(ids.size() * 4);
  std::memcpy(idb.data(), ids.data(), idb.size());
  const auto floats = [&](size_t count, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    bytes b(4 * count);
    for (size_t i = 0; i < count; ++i) {
      const float f = u(g);
      std::memcpy(&b[4 * i], &f, 4);
    }
    return b;
  };
  const int64_t xs = per_slot ? slots : 1;
  std::vector<arg> args = {{q, {experts * n, k}, quant(q, experts * n, k, g)},
                           {LARCH_F16, per_slot ? std::vector<int64_t>{rows, slots, k} : std::vector<int64_t>{rows, k},
                            halves(size_t(rows * xs * k), g)},
                           {LARCH_I32, {rows, slots}, idb}};
  if (bias) args.push_back({LARCH_F32, {experts * n}, floats(size_t(experts * n), -1, 1)});
  if (weighted) args.push_back({LARCH_F32, {rows, slots}, floats(size_t(rows * slots), 0, 1)});
  const std::vector<int64_t> os = weighted ? std::vector<int64_t>{rows, n} : std::vector<int64_t>{rows, slots, n};
  args.push_back({ot, os, bytes(size_t(nbytes(ot, os)), 0xFF)});
  compare(name, ref, dut, args, bias, weighted, rows * (weighted ? 1 : slots) * n, ot);
}

// A Q4_K or Q5_K block of d = dmin = 1024 whose sub-blocks 0-3 hold sc = m = 1 and nibbles 1 (1024 - 1024) and
// 4-7 sc = m = 0: every weight decodes to 0. x is 65504, 2^-10, -65504, then 0. The reference's output is 0;
// d sc sum(q x) - dmin m sum(x) gave 1.
// Grouped prefill's row decode against the reference's to_f32: the same weights, bit for bit where the reference
// rounds its product and difference apart (x86); where its compiler fuses them (clang on ARM64), within an ulp
// of the row's largest weight.
void decode_rows() {
  for (const larch_dtype q : {LARCH_Q3_K, LARCH_Q4_K, LARCH_Q5_K, LARCH_Q6_K, LARCH_Q8_0}) {
    rng g(5);
    const int64_t k = 2048;
    const bytes w = quant(q, 1, k, g);
    std::vector<float> want(static_cast<size_t>(k)), got(static_cast<size_t>(k));
    cpu::to_f32(q, w.data(), k, want.data());
    if (!cpu::fast_decode_row(q, w.data(), k, got.data())) {
      std::printf("fast_mmvq_id: no ISA here, %s decode not checked\n", DTYPE[q]);
      continue;
    }
    float top = 0;
    for (const float v : want) top = std::max(top, std::fabs(v));
    int off = 0;
    for (int64_t i = 0; i < k; ++i) {
#if defined(__aarch64__) || defined(_M_ARM64)
      off += std::fabs(got[size_t(i)] - want[size_t(i)]) > top * 0x1p-23f;
#else
      off += std::memcmp(&got[size_t(i)], &want[size_t(i)], 4) != 0;
#endif
    }
    if (off) {
      std::printf("WRONG: fast decode of %s differs from to_f32 at %d of %lld weights\n", DTYPE[q], off, (long long)k);
      ++wrong;
    }
  }
}

void zero_weights(const dev& ref, const dev& dut, larch_dtype q) {
  const bool q5 = q == LARCH_Q5_K;
  bytes w(q5 ? 176 : 144, 0);
  w[0] = w[2] = 0x00, w[1] = w[3] = 0x64;  // f16 1024
  for (int i = 4; i < 12; ++i) w[size_t(i)] = 1;
  for (size_t i = q5 ? 48 : 16; i < w.size(); ++i) w[i] = 0x11;
  std::vector<uint16_t> x(256, 0);
  x[0] = 0x7BFF, x[1] = 0x1400, x[2] = 0xFBFF;  // 65504, 2^-10, -65504
  const int32_t id = 0;
  compare(std::string("weights decoding to 0, ") + DTYPE[q], ref, dut,
          {{q, {1, 256}, w}, {LARCH_F16, {1, 256}, raw(x.data(), 512)}, {LARCH_I32, {1, 1}, raw(&id, 4)},
           {LARCH_F32, {1, 1, 1}, bytes(4, 0xFF)}},
          false, false, 1, LARCH_F32);
}

// Two Q8_0 experts of one column: projections 2^-24 (d = 2^-24, q 1, x 1) and 0, biases 1 and -1, weights
// 0.5 each. The reference's sum is 2^-25; adding the bias to an f32 projection first gave 0. rows 4 take the
// grouped prefill path, rows 1 the per-pair one.
void bias_in_the_sum(const dev& ref, const dev& dut, int64_t rows) {
  bytes w(68, 0);
  w[0] = 0x01;  // f16 2^-24
  w[2] = 1;
  std::vector<uint16_t> x(size_t(rows * 32), 0);
  std::vector<int32_t> ids(size_t(rows * 2));
  std::vector<float> wt(size_t(rows * 2), 0.5f);
  for (int64_t r = 0; r < rows; ++r) x[size_t(r * 32)] = 0x3C00, ids[size_t(2 * r)] = 0, ids[size_t(2 * r + 1)] = 1;
  const float bias[2] = {1, -1};
  compare("bias in the weighted sum, " + std::to_string(rows) + " rows", ref, dut,
          {{LARCH_Q8_0, {2, 32}, w},
           {LARCH_F16, {rows, 32}, raw(x.data(), x.size() * 2)},
           {LARCH_I32, {rows, 2}, raw(ids.data(), ids.size() * 4)},
           {LARCH_F32, {2}, raw(bias, 8)},
           {LARCH_F32, {rows, 2}, raw(wt.data(), wt.size() * 4)},
           {LARCH_F32, {rows, 1}, bytes(size_t(rows * 4), 0xFF)}},
          true, true, rows, LARCH_F32);
}

// One Q3_K or Q4_K block of d sc = 1 (no mins) whose weight 0 is 0 and the rest 1, against x 1 (on the 0) and
// values 2^14 to 2^21 below it: the sum is those small values alone, each in a different integer limb of the
// AVX-512 kernel's x, so a lost limb is off by far more than the tolerance. Then an infinite x, which that
// kernel cannot hold as an integer: the sum is +inf, as the reference's.
void wide_x(const dev& ref, const dev& dut, larch_dtype q) {
  const bool q3 = q == LARCH_Q3_K;
  bytes w(q3 ? 110 : 144, 0);
  if (q3) {  // q = lo + 4 hb - 4: hb 1 and lo 1, lo 0 for weight 0; every scale 33 - 32 (nibble 1, high bits 2)
    std::fill(w.begin(), w.begin() + 32, uint8_t(0xFF));
    std::fill(w.begin() + 32, w.begin() + 96, uint8_t(0x55));
    w[32] = 0x54;
    std::fill(w.begin() + 96, w.begin() + 104, uint8_t(0x11));
    std::fill(w.begin() + 104, w.begin() + 108, uint8_t(0xAA));
    w[109] = 0x3C;  // d = 1
  } else {  // nibbles 1, 0 for weight 0; sc 1 and m 0 in every sub-block; d = dmin = 1
    w[1] = w[3] = 0x3C;
    for (int i : {4, 5, 6, 7, 12, 13, 14, 15}) w[size_t(i)] = 1;
    std::fill(w.begin() + 16, w.end(), uint8_t(0x11));
    w[16] = 0x10;
  }
  std::vector<uint16_t> x(256, 0);
  x[0] = 0x3C00, x[1] = 0x0400, x[2] = 0x8008, x[3] = 0x9400, x[4] = 0x0010;  // 1, 2^-14, -2^-21, -2^-10, 2^-20
  const int32_t id = 0;
  const auto args = [&] {
    return std::vector<arg>{{q, {1, 256}, w},
                            {LARCH_F16, {1, 256}, raw(x.data(), 512)},
                            {LARCH_I32, {1, 1}, raw(&id, 4)},
                            {LARCH_F32, {1, 1, 1}, bytes(4, 0xFF)}};
  };
  compare(std::string("x across 2^21 in one block, ") + DTYPE[q], ref, dut, args(), false, false, 1, LARCH_F32);
  x[5] = 0x7C00;  // +inf
  larch_status st;
  const float got = elem(call(dut, args(), false, false, &st), 0, LARCH_F32);
  if (st != LARCH_OK || !(std::isinf(got) && got > 0)) {
    std::printf("WRONG: an infinite x gave %g (status %d), %s\n", double(got), st, DTYPE[q]);
    ++wrong;
  }
}

// The pool: a chunk that throws on a helper, on the caller, or on every thread ends the job; run returns only
// after every helper left the callback, and rethrows; the next job runs every chunk.
void pool_exceptions() {
  for (const int which : {0, 1, 2}) {  // a helper's chunk, the first chunk (the caller's, most often), all
    std::atomic<int> inside{0}, after{0};
    bool thrown = false;
    {
      const std::function<void(int64_t, int64_t)> f = [&](int64_t first, int64_t) {
        ++inside;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const bool fail = which == 2 || (which == 1 ? first == 0 : first == 37);
        --inside;
        if (fail) throw std::bad_alloc();
      };
      try {
        cpu::fast_pool_run(64, 1, 4, f);
      } catch (const std::bad_alloc&) {
        thrown = true;
      }
      after = inside.load();
    }
    if (!thrown || after != 0) {
      std::printf("WRONG pool exception %d: thrown %d, %d still inside\n", which, thrown, after.load());
      ++wrong;
    }
  }
  std::atomic<int64_t> chunks{0};
  cpu::fast_pool_run(64, 1, 4, [&](int64_t, int64_t) { ++chunks; });
  if (chunks != 64) {
    std::printf("WRONG pool after exceptions: %lld of 64 chunks\n", (long long)chunks.load());
    ++wrong;
  }
}

// Short jobs back to back, as a decode step's host experts run them: the helpers are still spinning when the
// next job comes, the helper count changes from job to job, and every 5th job throws, on every helper count in
// turn. Each job runs every chunk once (or throws), and no helper is lost or woken twice (that miscounts, or
// hangs into the ctest TIMEOUT).
void pool_back_to_back() {
  for (int job = 0; job < 20000; ++job) {
    const uint32_t threads = 2 + uint32_t(job % 7);
    const bool fail = job % 5 == 3;
    std::atomic<int64_t> chunks{0};
    bool thrown = false;
    try {
      cpu::fast_pool_run(24, 1, threads, [&](int64_t first, int64_t) {
        ++chunks;
        if (fail && first == 17) throw std::bad_alloc();
      });
    } catch (const std::bad_alloc&) {
      thrown = true;
    }
    if (thrown != fail || (!fail && chunks != 24)) {
      std::printf("WRONG pool job %d: thrown %d, %lld of 24 chunks\n", job, thrown, (long long)chunks.load());
      ++wrong;
      return;
    }
  }
}

// `fast_mmvq_id bench` (ctest runs it without arguments): one host-expert section of a decode step at 1-8 rows at
// the hybrid target's expert shapes above, gate and up Q3_K [512, 2048] then down [2048, 512] weighted, Q3_K and Q4_K as
// its layers mix them, 8 of 256 experts, other experts each rep so the weights come from RAM (0.5 GB, past any L3);
// then the same experts each rep (warm), on all threads and on one (threads 0 is all). Median and best per phase;
// the silu_mul between up and down is ops.cpp's, not timed here, but the fused gate and up (the device's
// fused_moe_gate_up, as the host experts run them) include theirs.
void bench_section(const dev& dut, bool warm, uint32_t threads) {
  const int64_t experts = 256, d = 2048, ef = 512, slots = 8, reps = 40;
  larch_device_set_threads(dut.d, threads);
  rng g(13);
  const auto upload = [&](const bytes& b) {
    larch_buffer* p;
    larch_buffer_alloc(dut.d, b.size(), &p);
    larch_upload(dut.s, p, 0, b.data(), b.size());
    return p;
  };
  larch_buffer* gate = upload(quant(LARCH_Q3_K, experts * ef, d, g));
  larch_buffer* up = upload(quant(LARCH_Q3_K, experts * ef, d, g));
  larch_buffer* down3 = upload(quant(LARCH_Q3_K, experts * d, ef, g));
  larch_buffer* down4 = upload(quant(LARCH_Q4_K, experts * d, ef, g));
  larch_buffer *xb = upload(halves(size_t(8 * d), g)), *ab = upload(halves(size_t(8 * slots * ef), g));
  larch_buffer *ib, *wb, *gb, *ob;
  larch_buffer_alloc(dut.d, size_t(4 * 8 * slots), &ib);
  larch_buffer_alloc(dut.d, size_t(4 * 8 * slots), &wb);
  larch_buffer_alloc(dut.d, size_t(2 * 8 * slots * ef), &gb);
  larch_buffer_alloc(dut.d, size_t(2 * 8 * d), &ob);
  const larch_tensor gt = {gate, 0, LARCH_Q3_K, 2, {experts * ef, d}, {d / 256, 1}};
  const larch_tensor ut = {up, 0, LARCH_Q3_K, 2, {experts * ef, d}, {d / 256, 1}};
  const larch_tensor dt[2] = {{down3, 0, LARCH_Q3_K, 2, {experts * d, ef}, {ef / 256, 1}},
                               {down4, 0, LARCH_Q4_K, 2, {experts * d, ef}, {ef / 256, 1}}};
  for (const int64_t rows : {1, 2, 4, 8}) {
    if (warm && rows > 1) break;
    const larch_tensor xt = {xb, 0, LARCH_F16, 2, {rows, d}, {d, 1}};
    const larch_tensor it = {ib, 0, LARCH_I32, 2, {rows, slots}, {slots, 1}};
    const larch_tensor wt = {wb, 0, LARCH_F32, 2, {rows, slots}, {slots, 1}};
    const larch_tensor gout = {gb, 0, LARCH_F16, 3, {rows, slots, ef}, {slots * ef, ef, 1}};
    const larch_tensor at = {ab, 0, LARCH_F16, 3, {rows, slots, ef}, {slots * ef, ef, 1}};
    const larch_tensor ot = {ob, 0, LARCH_F16, 2, {rows, d}, {d, 1}};
    std::vector<double> t[5];  // gate, up, down Q3_K, down Q4_K, gate and up fused (with their silu_mul)
    for (int64_t rep = 0; rep < reps; ++rep) {
      std::vector<int32_t> ids(size_t(rows * slots));
      if (warm) g.seed(13);
      for (int64_t r = 0; r < rows; ++r)  // distinct per row, as a top-k
        for (int64_t j = 0; j < slots; ++j) {
          const auto row = ids.begin() + r * slots;
          int32_t e;
          do e = int32_t(g() % experts);
          while (std::find(row, row + j, e) != row + j);
          ids[size_t(r * slots + j)] = e;
        }
      const std::vector<float> w(ids.size(), 0.125f);
      larch_upload(dut.s, ib, 0, ids.data(), 4 * ids.size());
      larch_upload(dut.s, wb, 0, w.data(), 4 * w.size());
      const auto time = [&](std::vector<double>& v, const larch_tensor& wq, const larch_tensor& x, bool weighted,
                            const larch_tensor& o) {
        const auto t0 = std::chrono::steady_clock::now();
        if (cpu::fast_mmvq_id(dut.s, &wq, &x, &it, nullptr, weighted ? &wt : nullptr, &o) != LARCH_OK) ++wrong;
        v.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6);
      };
      time(t[0], gt, xt, false, gout);
      time(t[1], ut, xt, false, gout);
      time(t[2], dt[0], at, true, ot);
      time(t[3], dt[1], at, true, ot);
      const auto t0 = std::chrono::steady_clock::now();
      if (larch_fused_moe_gate_up(dut.s, &gt, &ut, &xt, &it, &gout) != LARCH_OK) ++wrong;  // not down's x
      t[4].push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6);
    }
    double med[5], best[5];
    for (int i = 0; i < 5; ++i) {
      std::sort(t[i].begin(), t[i].end());
      med[i] = t[i][t[i].size() / 2], best[i] = t[i][0];
    }
    std::printf("bench %s %u threads rows %lld: gate %.0f (%.0f) up %.0f (%.0f) down q3_k %.0f (%.0f) q4_k %.0f (%.0f)"
                " us median (best); section q3_k %.0f us, q4_k %.0f us; gate and up fused %.0f (%.0f) us\n",
                warm ? "warm" : "cold", threads, (long long)rows, med[0], best[0], med[1], best[1], med[2], best[2],
                med[3], best[3], med[0] + med[1] + med[2], med[0] + med[1] + med[3], med[4], best[4]);
  }
  for (larch_buffer* b : {gate, up, down3, down4, xb, ab, ib, wb, gb, ob}) larch_buffer_free(b);
}

}  // namespace

int main(int argc, char** argv) {
  const dev ref = open(-1), dut = open(-2);
  if (argc > 1 && std::string(argv[1]) == "bench") {
    bench_section(dut, false, 0);
    bench_section(dut, true, 0);
    bench_section(dut, true, 1);
    return wrong != 0;
  }
  // The reference's cases on the kernel this CPU runs, then on the AVX2 (NEON) one where that is the VNNI kernel,
  // so ctest covers both wherever it runs.
  const auto cases = [&](const dev& dut) {
    const int before = wrong;
    for (const larch_dtype q : QUANTS) {
      const int64_t k = block_of(q) == 256 ? 768 : 896;
      check(ref, dut, q, 4, 1, 2, 67, k, false, false, false, LARCH_F16);
      check(ref, dut, q, 4, 5, 2, 33, k, true, true, false, LARCH_F32);
      check(ref, dut, q, 8, 3, 8, 67, k, false, false, true, LARCH_F32);
      check(ref, dut, q, 8, 2, 3, 33, k, true, true, true, LARCH_F16);
      // Prefills: enough pairs per expert to group them by expert, odd columns and short groups.
      check(ref, dut, q, 4, 19, 2, 67, k, false, false, true, LARCH_F32);
      check(ref, dut, q, 4, 19, 3, 33, k, true, true, true, LARCH_F16);
    }
    for (const int64_t rows : {1, 7, 100}) {
      check(ref, dut, LARCH_Q4_K, 128, rows, 8, 768, 2048, false, false, false, LARCH_F16);
      check(ref, dut, LARCH_Q6_K, 128, rows, 8, 2048, 768, true, true, false, LARCH_F32);
    }
    // The hybrid MoE target's experts: 256 of [512, 2048] (gate, up) and [2048, 512] (down, Q3_K or Q4_K by layer), 8 used.
    for (const int64_t rows : {1, 8}) {
      check(ref, dut, LARCH_Q3_K, 256, rows, 8, 512, 2048, false, false, false, LARCH_F16);
      check(ref, dut, LARCH_Q3_K, 256, rows, 8, 2048, 512, true, true, false, LARCH_F32);
      check(ref, dut, LARCH_Q4_K, 256, rows, 8, 2048, 512, true, true, false, LARCH_F16);
    }
    // One thread computes what many do: the work items never share a sum.
    larch_device_set_threads(dut.d, 1);
    check(ref, dut, LARCH_Q4_K, 16, 3, 8, 96, 512, false, false, true, LARCH_F32);
    larch_device_set_threads(dut.d, 0);
    zero_weights(ref, dut, LARCH_Q4_K);
    zero_weights(ref, dut, LARCH_Q5_K);
    wide_x(ref, dut, LARCH_Q3_K);
    wide_x(ref, dut, LARCH_Q4_K);
    for (const int64_t rows : {1, 4}) bias_in_the_sum(ref, dut, rows);
    std::printf("fast_mmvq_id: %d wrong against the reference on the %s kernel\n", wrong - before,
                dut.no_vnni || !cpu::fast_has_vnni() ? "AVX2/NEON" : "AVX-512 VNNI");
  };
  cases(dut);
  if (cpu::fast_has_vnni()) {
    dev avx2 = dut;
    avx2.no_vnni = true;
    cases(avx2);
  }
  decode_rows();
  pool_exceptions();
  pool_back_to_back();
  // An id past the experts is refused, as the reference refuses it.
  {
    rng g(7);
    std::vector<arg> args = {{LARCH_Q8_0, {4, 64}, quant(LARCH_Q8_0, 4, 64, g)},
                             {LARCH_F16, {1, 64}, halves(64, g)},
                             {LARCH_I32, {1, 1}, bytes{4, 0, 0, 0}},
                             {LARCH_F16, {1, 1, 2}, bytes(4, 0xFF)}};
    larch_status st;
    call(dut, args, false, false, &st);
    if (st != LARCH_ERR_INVALID_ARG) {
      std::printf("WRONG: id 4 of 2 experts gave status %d\n", st);
      ++wrong;
    }
  }

  // One qwen3moe gate, 8 of 128 experts of [768, 2048] in Q4_K and in Q3_K (Q3_K_M's experts): at a decode step
  // its read bandwidth (6.3 MB in Q4_K), at a prefill chunk its time.
  for (const larch_dtype q : {LARCH_Q4_K, LARCH_Q3_K})
  for (const int64_t rows : {1, 512}) {
    rng g(11);
    const int64_t experts = 128, n = 768, k = 2048, slots = 8;
    const bytes w = quant(q, experts * n, k, g), x = halves(size_t(rows * k), g);
    larch_buffer *wb, *xb, *ib, *ob;
    larch_buffer_alloc(dut.d, w.size(), &wb);
    larch_buffer_alloc(dut.d, x.size(), &xb);
    larch_buffer_alloc(dut.d, size_t(4 * rows * slots), &ib);
    larch_buffer_alloc(dut.d, size_t(2 * rows * slots * n), &ob);
    larch_upload(dut.s, wb, 0, w.data(), w.size());
    larch_upload(dut.s, xb, 0, x.data(), x.size());
    const larch_tensor wt = {wb, 0, q, 2, {experts * n, k}, {k / 256, 1}};
    const larch_tensor xt = {xb, 0, LARCH_F16, 2, {rows, k}, {k, 1}};
    const larch_tensor it = {ib, 0, LARCH_I32, 2, {rows, slots}, {slots, 1}};
    const larch_tensor ot = {ob, 0, LARCH_F16, 3, {rows, slots, n}, {slots * n, n, 1}};
    double best = 1e30;
    for (int rep = 0; rep < (rows == 1 ? 20 : 3); ++rep) {
      std::vector<int32_t> ids(size_t(rows * slots));
      for (int32_t& e : ids) e = int32_t(g() % experts);  // other experts each time: cold reads
      larch_upload(dut.s, ib, 0, ids.data(), 4 * ids.size());
      const auto t0 = std::chrono::steady_clock::now();
      if (cpu::fast_mmvq_id(dut.s, &wt, &xt, &it, nullptr, nullptr, &ot) != LARCH_OK) ++wrong;
      best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    larch_device_props p{};
    larch_device_props_get(dut.d, &p);
    const char* name = q == LARCH_Q4_K ? "q4_k" : "q3_k";
    if (rows == 1)
      std::printf("fast_mmvq_id: qwen3moe %s gate step %.0f us, %.1f GB/s on %u threads\n", name, best * 1e6,
                  double(slots * n * k) / 256 * double(block_bytes(q)) / best / 1e9, p.compute_units);
    else
      std::printf("fast_mmvq_id: qwen3moe %s gate, %lld-row prefill %.1f ms on %u threads\n", name, (long long)rows,
                  best * 1e3, p.compute_units);
    for (larch_buffer* b : {wb, xb, ib, ob}) larch_buffer_free(b);
  }
  std::printf("fast_mmvq_id: %d wrong\n", wrong);
  return wrong != 0;
}
