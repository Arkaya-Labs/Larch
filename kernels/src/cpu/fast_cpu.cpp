// The fast CPU backend (lane c1, architecture 4.4 of the v2 spec): the CPU device's (-2) ops that differ from
// the reference's, multi-threaded over a pool of the stream's threads, with the inner loops in the ISA file of
// the target (fast_*_avx2.cpp on x86-64 with AVX2 and FMA, fast_*_neon.cpp on ARM64). Without one, or for a
// dtype it lacks, an op runs the reference's code.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
#include <immintrin.h>
#include <intrin.h>
#endif

#include "cpu/cpu.h"
#include "cpu/fast_cpu.h"

namespace cpu {

larch_status mmvq_id(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                      const larch_tensor*, const larch_tensor*, const larch_tensor*);

namespace {

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER) && !defined(__clang__)
bool has_isa() {
  static const bool ok = [] {
    int r1[4], r7[4];
    __cpuid(r1, 1);
    __cpuidex(r7, 7, 0);
    const bool os_avx = (r1[2] >> 27 & 1) && (_xgetbv(0) & 6) == 6;  // OSXSAVE and the OS saves ymm
    return os_avx && (r1[2] >> 12 & 1) && (r1[2] >> 29 & 1) && (r7[1] >> 5 & 1);  // fma, f16c, avx2
  }();
  return ok;
}
// AVX-512 F, DQ, CD, BW, VL (fast_*_avx512.cpp's flags) and VNNI, with the OS saving opmask and zmm.
bool has_vnni() {
  static const bool ok = [] {
    int r7[4];
    __cpuidex(r7, 7, 0);
    const uint32_t b = uint32_t(r7[1]), need = 1u << 16 | 1u << 17 | 1u << 28 | 1u << 30 | 1u << 31;
    return has_isa() && (_xgetbv(0) & 0xE6) == 0xE6 && (b & need) == need && (r7[2] >> 11 & 1);
  }();
  return ok;
}
#else
bool has_isa() {
  static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
                         __builtin_cpu_supports("f16c");
  return ok;
}
// AVX-512 F, DQ, CD, BW, VL (fast_*_avx512.cpp's flags) and VNNI.
bool has_vnni() {
  static const bool ok = has_isa() && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
                         __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512bw") &&
                         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
  return ok;
}
#endif
#else
bool has_vnni() { return false; }
}  // namespace
bool fast_vnni_supports(int) { return false; }
bool fast_vnni_prep(const float*, int64_t, int8_t*, int32_t*, float*, float*) { return false; }
void fast_mmvq_id_vnni(const fast_id_args&, int64_t, int64_t) {}
namespace {
#if defined(__aarch64__) || defined(_M_ARM64)
bool has_isa() { return true; }  // NEON is ARM64's baseline
#else
bool has_isa() { return false; }
}  // namespace
bool fast_id_supports(int) { return false; }
void fast_mmvq_id_isa(const fast_id_args&, int64_t, int64_t) {}
void fast_dot_tile(const float* const*, const float* const*, int64_t, float*) {}
bool fast_decode_row_isa(int, const uint8_t*, int64_t, float*) { return false; }
namespace {
#endif
#endif

// Runs f over [0, n) in chunks of `grain` on up to `threads` threads, the caller one of them. One job of
// helpers at a time: the CPU device runs its ops at the call, and a second caller waits (one chunk, or one
// thread, runs on the caller with no lock). The helpers start at the first call that needs them and stay. A
// chunk that throws (bad_alloc) ends the job: the rest of the chunks are skipped, every helper finishes before
// run returns, and run rethrows the first exception. A helper waiting for a job, and the caller waiting for the
// helpers, spin for SPIN before they sleep on the condition variable: a decode step's host experts run jobs
// back to back, and a wake from the condition variable costs more than the gap between them.
class pool {
 public:
  void run(int64_t n, int64_t grain, uint32_t threads, const std::function<void(int64_t, int64_t)>& f) {
    const int64_t chunks = (n + grain - 1) / grain;
    if (chunks <= 1 || threads <= 1) return f(0, n);  // on the caller alone: no lock, no helper
    std::lock_guard<std::mutex> one(job_);
    const size_t helpers = size_t(std::min<int64_t>(threads, chunks)) - 1;
    {
      std::unique_lock<std::mutex> l(m_);
      while (workers_.size() < helpers) workers_.emplace_back([this, i = workers_.size()] { work(i); });
      f_ = &f, n_ = n, grain_ = grain, next_ = 0, active_ = helpers, busy_ = helpers, error_ = nullptr, ++gen_;
    }
    wake_.notify_all();
    take();
    spin([this] { return busy_ == 0; });
    std::unique_lock<std::mutex> l(m_);
    done_.wait(l, [this] { return busy_ == 0; });
    if (error_) std::rethrow_exception(std::exchange(error_, nullptr));
  }

  ~pool() {
    {
      std::lock_guard<std::mutex> l(m_);
      stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : workers_) t.join();
  }

 private:
  // ponytail: 50 us measured an A/B of -12 % a decode step on the hybrid MoE target (3 rounds
  // interleaved, WSL2, 8 threads, CUDA sm_120 host). It applies to every CPU pool job, so re-derive
  // it on another host, or when a CPU-device prefill shows the spin's cost.
  static constexpr std::chrono::microseconds SPIN{50};

  // Yields until ready() or SPIN ran out; the caller then waits on its condition variable under m_ as before,
  // so the lock, not this, publishes the job and its results.
  template <class F>
  static void spin(F ready) {
    const auto end = std::chrono::steady_clock::now() + SPIN;
    while (!ready() && std::chrono::steady_clock::now() < end) std::this_thread::yield();
  }

  void take() {
    try {
      for (int64_t at; (at = next_.fetch_add(grain_)) < n_;) (*f_)(at, std::min(n_, at + grain_));
    } catch (...) {
      std::lock_guard<std::mutex> l(m_);
      if (!error_) error_ = std::current_exception();
      next_ = n_;  // the others take no more
    }
  }

  void work(size_t i) {
    uint64_t seen = 0;
    for (;;) {
      spin([&] { return gen_ != seen; });
      {
        std::unique_lock<std::mutex> l(m_);
        wake_.wait(l, [&] { return stop_ || (gen_ != seen && i < active_); });
        if (stop_) return;
        seen = gen_;
      }
      take();
      std::lock_guard<std::mutex> l(m_);
      if (--busy_ == 0) done_.notify_one();
    }
  }

  std::mutex job_, m_;
  std::condition_variable wake_, done_;
  std::vector<std::thread> workers_;
  const std::function<void(int64_t, int64_t)>* f_ = nullptr;
  std::atomic<int64_t> next_{0};
  int64_t n_ = 0, grain_ = 1;
  std::exception_ptr error_;
  size_t active_ = 0;
  std::atomic<size_t> busy_{0};  // atomic, and gen_, for spin's reads outside m_; written under m_
  std::atomic<uint64_t> gen_{0};
  bool stop_ = false;
};

pool& threads_pool() {
  static pool p;
  return p;
}

// Pairs per distinct expert from which mmvq_id groups a call's pairs by expert (mmvq_id_on).
constexpr int64_t GROUPED = 4;

// A calling thread's mmvq_id scratch, kept between calls: once a step of the largest shape has run, a step
// allocates nothing.
struct id_scratch {
  std::vector<float> xf, y, bias, wv, yp, xm, xe;
  std::vector<int32_t> iv, xs;
  std::vector<int8_t> xl;
  std::vector<int64_t> start, at, order, used;
};

}  // namespace

// fast_decode_row_isa on a CPU that has the ISA; false without it, or for a dtype it lacks.
bool fast_decode_row(int dtype, const uint8_t* w, int64_t k, float* out) {
  return has_isa() && fast_decode_row_isa(dtype, w, k, out);
}

// The pool, for tests (fast_mmvq_id.cpp): its exceptions and its helpers' lifetimes.
void fast_pool_run(int64_t n, int64_t grain, uint32_t threads, const std::function<void(int64_t, int64_t)>& f) {
  threads_pool().run(n, grain, threads, f);
}

namespace {

// The reference's checks and results; x goes to f32 once, the GEMVs run in the ISA kernel (the VNNI one when
// vnni and the CPU has it), and out is written as the reference writes it.
larch_status mmvq_id_on(larch_stream* s, const larch_tensor* w_quant, const larch_tensor* x_f16,
                         const larch_tensor* ids_i32, const larch_tensor* bias_or_null,
                         const larch_tensor* weights_or_null, const larch_tensor* out, bool vnni) {
  if (!has_isa() || !w_quant || !fast_id_supports(w_quant->dtype))
    return mmvq_id(s, w_quant, x_f16, ids_i32, bias_or_null, weights_or_null, out);
  return run("mmvq_id", [&] {
    const mmvq_id_checked c = check_mmvq_id(buffer_bytes, w_quant, x_f16, ids_i32, bias_or_null, weights_or_null, out);
    const view w = c.w, x = c.x, ids = c.ids, b = c.bias, wt = c.weights, o = c.out;
    const int64_t rows = x.rows, slots = ids.cols, k = w.cols, n = o.shape[o.ndim - 1], experts = w.rows / n;
    const int64_t x_slots = x.ndim == 3 ? slots : 1;
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t j = 0; j < slots; ++j) need(ids.i32(r, j) >= 0 && ids.i32(r, j) < experts, "expert id out of range");
    thread_local id_scratch sc;
    std::vector<float>&xf = sc.xf, &y = sc.y, &bias = sc.bias, &wv = sc.wv;
    xf.resize(size_t(rows * x_slots * k));
    y.resize(size_t(rows * (wt.buf ? 1 : slots) * n));
    for (int64_t r = 0; r < rows; ++r) to_f32(LARCH_F16, x.row(r), x_slots * k, xf.data() + r * x_slots * k);
    std::vector<int32_t>& iv = sc.iv;
    iv.resize(size_t(rows * slots));
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t j = 0; j < slots; ++j) iv[size_t(r * slots + j)] = ids.i32(r, j);
    if (b.buf) {
      bias.resize(size_t(w.rows));
      for (int64_t i = 0; i < w.rows; ++i) bias[size_t(i)] = b.get(i, 0);
    }
    if (wt.buf) {
      wv.resize(size_t(rows * slots));
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t j = 0; j < slots; ++j) wv[size_t(r * slots + j)] = wt.get(r, j);
    }
    const fast_id_args a{w.dtype,   w.row(0), w.row_bytes, xf.data(), x_slots * k, x.ndim == 3 ? k : 0,
                         iv.data(), slots,    b.buf ? bias.data() : nullptr,
                         wt.buf ? wv.data() : nullptr, slots, y.data(),
                         n,         k,        slots};
    const int64_t n_pairs = rows * slots, blocks = (n + FAST_COLS - 1) / FAST_COLS;
    // The pairs by expert (a counting sort). Past GROUPED pairs per expert used (a prefill), each expert's rows
    // are decoded once and dotted with every x that chose it, where the per-pair GEMV decodes them for each.
    std::vector<int64_t>&start = sc.start, &order = sc.order, &used = sc.used;
    start.assign(size_t(experts + 1), 0);
    used.clear();
    for (const int32_t e : iv) ++start[size_t(e) + 1];
    for (int64_t e = 0; e < experts; ++e) {
      if (start[size_t(e) + 1]) used.push_back(e);
      start[size_t(e) + 1] += start[size_t(e)];
    }
    if (n_pairs < GROUPED * int64_t(used.size())) {
      const int64_t items = (wt.buf ? rows : n_pairs) * blocks;
      fast_id_args v = a;
      const size_t nx = xf.size();
      v.w_rows = w.rows;
      if (vnni && has_vnni() && fast_vnni_supports(w.dtype)) {
        sc.xl.resize(3 * nx), sc.xs.resize(nx / 4), sc.xm.resize(nx / 32), sc.xe.resize(nx / 256);
        if (fast_vnni_prep(xf.data(), int64_t(nx), sc.xl.data(), sc.xs.data(), sc.xm.data(), sc.xe.data()))
          v.xl = sc.xl.data(), v.xs = sc.xs.data(), v.xm = sc.xm.data(), v.xe = sc.xe.data();
      }
      threads_pool().run(items, 4, stream_threads(s), [&](int64_t first, int64_t last) {
        v.xl ? fast_mmvq_id_vnni(v, first, last) : fast_mmvq_id_isa(v, first, last);
      });
    } else {
      std::vector<int64_t>& at = sc.at;
      at.assign(start.begin(), start.end() - 1);
      order.resize(size_t(n_pairs));
      for (int64_t p = 0; p < n_pairs; ++p) order[size_t(at[size_t(iv[size_t(p)])]++)] = p;
      std::vector<float>& yp = sc.yp;  // per slot, unbiased, before the weighted sum
      yp.resize(wt.buf ? size_t(n_pairs * n) : 0);
      float* dst = wt.buf ? yp.data() : y.data();
      threads_pool().run(int64_t(used.size()) * blocks, 1, stream_threads(s), [&](int64_t first, int64_t last) {
        thread_local std::vector<float> wf;
        wf.resize(size_t(FAST_COLS * k));
        for (int64_t item = first; item < last; ++item) {
          const int64_t e = used[size_t(item / blocks)], c0 = item % blocks * FAST_COLS;
          const int64_t cols = std::min(FAST_COLS, n - c0);
          for (int64_t c = 0; c < cols; ++c)
            if (!fast_decode_row_isa(w.dtype, w.row(e * n + c0 + c), k, wf.data() + c * k))
              w.row_f32(e * n + c0 + c, wf.data() + c * k);
          const int64_t *first_p = order.data() + start[size_t(e)], *end = order.data() + start[size_t(e) + 1];
          // Column tiles outer: a tile's FAST_TILE_W decoded rows stay in L1 while every x that chose the expert
          // streams past them.
          for (int64_t c = 0; c < cols; c += FAST_TILE_W) {
            const float* wr[FAST_TILE_W];
            for (int64_t u = 0; u < FAST_TILE_W; ++u) wr[u] = wf.data() + std::min(c + u, cols - 1) * k;
            for (const int64_t* p = first_p; p < end; p += FAST_TILE_X) {
              const int64_t np = std::min<int64_t>(FAST_TILE_X, end - p);
              const float* xr[FAST_TILE_X];
              for (int64_t t = 0; t < FAST_TILE_X; ++t) {
                const int64_t q = p[std::min(t, np - 1)];  // a short group repeats its last x
                xr[t] = xf.data() + q / slots * x_slots * k + (x_slots > 1 ? q % slots * k : 0);
              }
              float d[FAST_TILE_W * FAST_TILE_X];
              fast_dot_tile(wr, xr, k, d);
              for (int64_t u = 0; u < std::min<int64_t>(FAST_TILE_W, cols - c); ++u)
                for (int64_t t = 0; t < np; ++t)  // weighted: the bias joins the sum in double
                  dst[p[t] * n + c0 + c + u] = wt.buf || !b.buf
                                                   ? d[FAST_TILE_X * u + t]
                                                   : float(double(d[FAST_TILE_X * u + t]) + bias[size_t(e * n + c0 + c + u)]);
            }
          }
        }
      });
      if (wt.buf)  // the biased, weighted slots in ascending order in double, rounded once, as the reference
        threads_pool().run(rows, 1, stream_threads(s), [&](int64_t first, int64_t last) {
          for (int64_t r = first; r < last; ++r)
            for (int64_t c = 0; c < n; ++c) {
              double sum = 0;
              for (int64_t j = 0; j < slots; ++j) {
                const int64_t row = int64_t(iv[size_t(r * slots + j)]) * n + c;
                const double v = double(yp[size_t((r * slots + j) * n + c)]) + (b.buf ? bias[size_t(row)] : 0.0);
                sum += double(wv[size_t(r * slots + j)]) * v;
              }
              y[size_t(r * n + c)] = float(sum);
            }
        });
    }
    const int64_t per_row = (wt.buf ? 1 : slots) * n;
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t i = 0; i < per_row; ++i) o.set(r, i, y[size_t(r * per_row + i)]);
  });
}

}  // namespace

larch_status fast_mmvq_id(larch_stream* s, const larch_tensor* w_quant, const larch_tensor* x_f16,
                           const larch_tensor* ids_i32, const larch_tensor* bias_or_null,
                           const larch_tensor* weights_or_null, const larch_tensor* out) {
  return mmvq_id_on(s, w_quant, x_f16, ids_i32, bias_or_null, weights_or_null, out, true);
}

// fused_moe_gate_up (larch.h) on the fast kernels, for the per-pair shapes of a decode step: the sequence's bytes
// (mmvq_id's f16 g and u, then ops.cpp's silu_mul) with x prepared once and one pool job, in which each chunk of
// (row, slot, column block) items runs its gate items, the same up items, then silu_mul on those columns. Shapes
// mmvq_id groups by expert (a prefill), an x per slot, and dtypes without a fast kernel run the sequence.
larch_status fast_fused_moe_gate_up(larch_stream* s, const larch_tensor* gate_w, const larch_tensor* up_w,
                                    const larch_tensor* x_f16, const larch_tensor* ids_i32, const larch_tensor* out) {
  if (!has_isa() || !gate_w || !up_w || !fast_id_supports(gate_w->dtype) || !fast_id_supports(up_w->dtype))
    return fused_moe_gate_up(s, gate_w, up_w, x_f16, ids_i32, out);
  bool ran = false;
  const larch_status st = run("fused_moe_gate_up", [&] {
    const fused_gate_up_checked c = check_fused_moe_gate_up(buffer_bytes, gate_w, up_w, x_f16, ids_i32, out);
    const view x = c.gate.x, ids = c.gate.ids, o = c.gate.out;
    const view w[2] = {c.gate.w, c.up.w};
    // The sequence checks each weight's ids against its own experts: gate and up may hold different counts.
    const int64_t rows = x.rows, slots = ids.cols, k = w[0].cols, n = o.shape[o.ndim - 1],
                  experts = std::min(w[0].rows, w[1].rows) / n;
    thread_local id_scratch sc;
    std::vector<int32_t>& iv = sc.iv;
    iv.resize(size_t(rows * slots));
    for (int64_t p = 0; p < rows * slots; ++p) {
      iv[size_t(p)] = ids.i32(p / slots, p % slots);
      need(iv[size_t(p)] >= 0 && iv[size_t(p)] < experts, "expert id out of range");
    }
    std::vector<int64_t>& used = sc.used;
    used.assign(iv.begin(), iv.end());
    std::sort(used.begin(), used.end());
    const int64_t distinct = std::unique(used.begin(), used.end()) - used.begin();
    if (x.ndim == 3 || rows * slots >= GROUPED * distinct) return;  // ran stays false: the sequence
    std::vector<float>&xf = sc.xf, &y = sc.y;
    xf.resize(size_t(rows * k));
    for (int64_t r = 0; r < rows; ++r) to_f32(LARCH_F16, x.row(r), k, xf.data() + r * k);
    const int64_t per = rows * slots * n;
    y.resize(size_t(2 * per));  // g then u, f32 as mmvq_id's y
    fast_id_args a[2];
    for (int i = 0; i < 2; ++i) {
      a[i] = {w[i].dtype, w[i].row(0), w[i].row_bytes, xf.data(), k, 0, iv.data(), slots, nullptr, nullptr, slots,
              y.data() + i * per, n, k, slots};
      a[i].w_rows = w[i].rows;
    }
    const size_t nx = xf.size();
    if (has_vnni() && (fast_vnni_supports(w[0].dtype) || fast_vnni_supports(w[1].dtype))) {
      sc.xl.resize(3 * nx), sc.xs.resize(nx / 4), sc.xm.resize(nx / 32), sc.xe.resize(nx / 256);
      if (fast_vnni_prep(xf.data(), int64_t(nx), sc.xl.data(), sc.xs.data(), sc.xm.data(), sc.xe.data()))
        for (fast_id_args& v : a)
          if (fast_vnni_supports(v.dtype)) v.xl = sc.xl.data(), v.xs = sc.xs.data(), v.xm = sc.xm.data(), v.xe = sc.xe.data();
    }
    const int64_t blocks = (n + FAST_COLS - 1) / FAST_COLS;
    threads_pool().run(rows * slots * blocks, 4, stream_threads(s), [&](int64_t first, int64_t last) {
      for (const fast_id_args& v : a) v.xl ? fast_mmvq_id_vnni(v, first, last) : fast_mmvq_id_isa(v, first, last);
      // ops.cpp's silu_mul on mmvq_id's f16 g and u: each read back as f32, the result rounded to f16 once.
      for (int64_t i = first; i < last; ++i) {
        const int64_t pair = i / blocks, c0 = i % blocks * FAST_COLS, c1 = std::min(n, c0 + FAST_COLS);
        for (int64_t col = pair * n + c0; col < pair * n + c1; ++col) {
          const float g = f16_to_f32(f32_to_f16(y[size_t(col)])), u = f16_to_f32(f32_to_f16(y[size_t(per + col)]));
          o.set(pair / slots, pair % slots * n + col % n, g / (1.0f + expf(-g)) * u);
        }
      }
    });
    ran = true;
  });
  return st != LARCH_OK || ran ? st : fused_moe_gate_up(s, gate_w, up_w, x_f16, ids_i32, out);
}

// The mixed call: the hot pairs through this device's own mmvq_id, one row a pair, so below its grouped threshold (as a
// check step's ID_ROWS slices keep it) a pair's sums never depend on the others; the cold ones (Q2_K, which the fast
// kernels lack, and the experts the routing histogram counts least) in double on the pool; weighted, each row's slots
// summed in double in ascending order and rounded once, as fast_mmvq_id's grouped path sums them.
larch_status fast_mmvq_id_mixed(larch_stream* s, const larch_tensor* w_hot, const larch_tensor* w_cold,
                                 const larch_tensor* table_i32, const larch_tensor* x_f16, const larch_tensor* ids_i32,
                                 const larch_tensor* weights_or_null, const larch_tensor* out) {
  return run("mmvq_id_mixed", [&] {
    const mmvq_id_mixed_checked c =
        check_mmvq_id_mixed(buffer_bytes, w_hot, w_cold, table_i32, x_f16, ids_i32, weights_or_null, out);
    const view hot = c.id.w, cold = c.cold, table = c.table, x = c.id.x, ids = c.id.ids, wt = c.id.weights,
               o = c.id.out;
    const int64_t rows = x.rows, slots = ids.cols, k = hot.cols, n = o.shape[o.ndim - 1], xs = x.ndim == 3 ? slots : 1;
    // Each pair r slots + j by tensor, with its slot there.
    std::vector<int64_t> hp, cp;
    std::vector<int32_t> hs, cs;
    for (int64_t p = 0; p < rows * slots; ++p) {
      const int32_t e = ids.i32(p / slots, p % slots);
      need(e >= 0 && e < table.rows, "expert id out of range");
      const int32_t t = table.i32(e, 0), slot = t >= 0 ? t : -1 - t;
      need(slot < (t >= 0 ? hot : cold).rows / n, "table slot out of range");
      (t >= 0 ? hp : cp).push_back(p);
      (t >= 0 ? hs : cs).push_back(slot);
    }
    std::vector<float> y(size_t(rows * slots * n));
    const auto x_of = [&](int64_t p) { return x.row(p / slots) + (xs > 1 ? p % slots * k * 2 : 0); };
    if (!hp.empty()) {
      const int64_t np = int64_t(hp.size());
      std::vector<uint8_t> xg(size_t(np * k * 2));
      for (int64_t i = 0; i < np; ++i) std::memcpy(&xg[size_t(i * k * 2)], x_of(hp[size_t(i)]), size_t(k * 2));
      std::vector<float> yh(size_t(np * n));
      const vv_ops* ops = w_hot->buf->ops;
      larch_buffer xb{ops, xg.size(), xg.data()}, ib{ops, uint64_t(np * 4), reinterpret_cast<uint8_t*>(hs.data())},
          yb{ops, uint64_t(np * n * 4), reinterpret_cast<uint8_t*>(yh.data())};
      const larch_tensor xt{&xb, 0, LARCH_F16, 2, {np, k}, {k, 1}}, it{&ib, 0, LARCH_I32, 2, {np, 1}, {1, 1}},
          yt{&yb, 0, LARCH_F32, 3, {np, 1, n}, {n, n, 1}};
      // The inner status as it is: an OOM or a launch failure read as INVALID_ARG would be taken for a soft refusal.
      const larch_status hot_status = mmvq_id_on(s, w_hot, &xt, &it, nullptr, nullptr, &yt, true);
      if (hot_status == LARCH_ERR_OOM) throw std::bad_alloc();
      if (hot_status != LARCH_OK && hot_status != LARCH_ERR_INVALID_ARG)
        throw std::runtime_error("the hot experts' mmvq_id failed");
      need(hot_status == LARCH_OK, "the hot experts' mmvq_id failed");
      for (int64_t i = 0; i < np; ++i)
        std::copy_n(&yh[size_t(i * n)], n, &y[size_t(hp[size_t(i)] * n)]);
    }
    if (!cp.empty())
      threads_pool().run(int64_t(cp.size()) * n, 16, stream_threads(s), [&](int64_t first, int64_t last) {
        std::vector<float> xr(static_cast<size_t>(k)), wr(static_cast<size_t>(k));
        for (int64_t item = first; item < last; ++item) {
          const int64_t i = item / n, col = item % n, p = cp[size_t(i)];
          to_f32(LARCH_F16, x_of(p), k, xr.data());
          cold.row_f32(cs[size_t(i)] * n + col, wr.data());
          double acc = 0;
          for (int64_t e = 0; e < k; ++e) acc += double(xr[size_t(e)]) * wr[size_t(e)];
          y[size_t(p * n + col)] = float(acc);
        }
      });
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t col = 0; col < n; ++col) {
        double sum = 0;
        for (int64_t j = 0; j < slots; ++j) {
          const float v = y[size_t((r * slots + j) * n + col)];
          if (wt.buf) sum += double(wt.get(r, j)) * v;
          else o.set(r, j * n + col, v);
        }
        if (wt.buf) o.set(r, col, float(sum));
      }
  });
}

// For tests (fast_mmvq_id.cpp): fast_mmvq_id on the AVX2 (NEON) kernel where the VNNI one would run, so ctest
// covers both on any CPU; and whether this CPU takes the VNNI kernel.
larch_status fast_mmvq_id_no_vnni(larch_stream* s, const larch_tensor* w_quant, const larch_tensor* x_f16,
                                   const larch_tensor* ids_i32, const larch_tensor* bias_or_null,
                                   const larch_tensor* weights_or_null, const larch_tensor* out) {
  return mmvq_id_on(s, w_quant, x_f16, ids_i32, bias_or_null, weights_or_null, out, false);
}
bool fast_has_vnni() { return has_vnni(); }

}  // namespace cpu
