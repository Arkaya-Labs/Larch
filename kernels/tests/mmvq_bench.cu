/* mmvq_bench: times larch_mmvq, larch_mmvq_verify, larch_dequant + larch_gemm and a plain copy kernel (the
   bandwidth ceiling) on GPU 0. Cold by default: each call reads the next of enough copies of the weights to miss L2,
   as a decode step does; --hot reuses one. Not a ctest (the CMake test glob is .cpp only): a timing run takes
   the exclusive GPU lock. Output is markdown.

     nvcc -O3 -std=c++20 -arch=sm_120 -ccbin g++-14 -Ikernels/include kernels/tests/mmvq_bench.cu \
       build/a2-cuda-mmvq/liblarch_kernels.a -lcublas -o build/a2-cuda-mmvq/mmvq_bench
     flock ~/src/gpu.lock build/a2-cuda-mmvq/mmvq_bench peak
     flock ~/src/gpu.lock build/a2-cuda-mmvq/mmvq_bench mmvq [--fmt q4_k,q6_k] [--rows 1,2,4,8,16] \
       [--model m1|m5|m8] [--hot] [--gate F]
     flock ~/src/gpu.lock build/a2-cuda-mmvq/mmvq_bench cross [--fmt q4_k,q6_k] [--model m1|m5|m8]

   mmvq prints, per tensor and rows, the GB/s of larch_mmvq / of larch_mmvq_verify (us per call of each), timed
   back to back on the same buffers. The decode-step row sums both over the model; the row under it is verify's
   step us over mmvq's at rows 1, "x1.00 of one row" when rows 2-16 cost what row 1 costs ("-" without rows 1).

   --gate F exits 1 when any selected (model, format, rows) decode-step aggregate GB/s is below F x peak: rows 1
   judges larch_mmvq, rows >= 2 judge larch_mmvq_verify. A (model, format) whose k is not whole blocks of that
   format (m8 for the K quants) is skipped with a note. An option the subcommand does not take, a malformed one,
   a cross selection with no tensor, or a gate with no pair to judge exits 2. */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ranges>
#include <string>
#include <vector>

#include "testdata.h"
#include "larch.h"

static larch_device* dev;
static larch_stream* str;
static larch_device_props props;

[[noreturn]] static void die(const char* what, const char* why) {
  std::fprintf(stderr, "%s: %s\n", what, why);
  std::exit(2);
}
#define CK(call)                                                                                \
  do {                                                                                          \
    if (const larch_status s_ = (call); s_ != LARCH_OK) die(#call, larch_last_error(dev));   \
  } while (0)
#define CKC(call)                                                                               \
  do {                                                                                          \
    if (const cudaError_t e_ = (call); e_ != cudaSuccess) die(#call, cudaGetErrorString(e_));   \
  } while (0)

// The quant format named n, or exit 2.
static larch_dtype quant_named(const std::string& n) {
  for (const larch_dtype t : QUANTS)
    if (n == DTYPE[t]) return t;
  die("not a quant format", n.c_str());
}

// Tensor shapes (alias, tensor, n, k, bias); `gguf` lists the formats the model's GGUF stores the tensor in,
// for `cross`. lm_head is read once per step, every other tensor once per layer.
struct shape {
  const char* name;
  int64_t n, k;
  bool bias;
  const char* gguf;
};
struct model {
  const char* name;
  int layers;
  std::vector<shape> t;
};
static const model MODELS[] = {
    {"m1",
     32,
     {{"q", 4096, 4096, false, "q4_k"},
      {"k", 1024, 4096, false, "q4_k"},
      {"v", 1024, 4096, false, "q4_k,q6_k"},
      {"o", 4096, 4096, false, "q4_k"},
      {"gate", 14336, 4096, false, "q4_k"},
      {"up", 14336, 4096, false, "q4_k"},
      {"down", 4096, 14336, false, "q4_k,q6_k"},
      {"lm_head", 128256, 4096, false, "q6_k"}}},
    {"m5",
     28,
     {{"q", 3584, 3584, true, "q4_k"},
      {"k", 512, 3584, true, "q4_k"},
      {"v", 512, 3584, true, "q4_k,q6_k"},
      {"o", 3584, 3584, false, "q4_k"},
      {"gate", 18944, 3584, false, "q4_k"},
      {"up", 18944, 3584, false, "q4_k"},
      {"down", 3584, 18944, false, "q4_k,q6_k"},
      {"lm_head", 152064, 3584, false, "q6_k"}}},
    {"m8",
     24,
     {{"q", 896, 896, true, "q8_0"},
      {"k", 128, 896, true, "q8_0"},
      {"v", 128, 896, true, "q8_0"},
      {"o", 896, 896, false, "q8_0"},
      {"gate", 4864, 896, false, "q8_0"},
      {"up", 4864, 896, false, "q8_0"},
      {"down", 896, 4864, false, "q8_0"},
      {"lm_head", 151936, 896, false, "q8_0"}}},
};
static bool is_head(const shape& s) { return !std::strcmp(s.name, "lm_head"); }

// The items of a comma-separated list.
static std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  for (const auto item : std::views::split(s, ',')) v.emplace_back(item.begin(), item.end());
  return v;
}

// Owns the device buffers of one measurement, so a model's tensors never pile up.
struct bufs {
  std::vector<larch_buffer*> v;
  ~bufs() {
    for (larch_buffer* b : v) larch_buffer_free(b);
  }
  larch_buffer* put(const void* src, uint64_t n) {
    larch_buffer* b;
    CK(larch_buffer_alloc(dev, n, &b));
    v.push_back(b);
    if (src) CK(larch_upload(str, b, 0, src, n));
    return b;
  }
};

// A dense row-major [rows, cols] view at offset 0 (a quant row's stride counts blocks).
static larch_tensor tensor(larch_buffer* b, larch_dtype t, int64_t rows, int64_t cols) {
  return {b, 0, t, 2, {rows, cols}, {cols / block_of(t), 1}};
}

// s's f32 [n] bias from x's first halves (biased n is under 16 k), uploaded; an empty tensor if s has none.
static larch_tensor bias_of(bufs& B, const shape& s, const bytes& xh) {
  if (!s.bias) return {};
  std::vector<float> bias(size_t(s.n));
  for (size_t i = 0; i < bias.size(); ++i) {
    __half h;
    std::memcpy(&h, &xh[2 * i], 2);
    bias[i] = __half2float(h);
  }
  return {B.put(bias.data(), 4 * bias.size()), 0, LARCH_F32, 1, {s.n}, {1}};
}

// The bias argument of a call on s: production passes a biased shape's bias to the gemm and to mmvq alike, so a
// call that drops it would time a different operation (gemm adds it in a second pass, mmvq in its kernel).
static const larch_tensor* carry(const shape& s, const larch_tensor* bias) {
  if (s.bias && !bias) die(s.name, "biased shape called without its bias");
  return bias;
}

// The copies of s's weights in format f, uploaded: the w each call reads in turn.
static std::vector<larch_tensor> weights(bufs& B, const shape& s, larch_dtype f, bool hot) {
  rng g(1234);
  const uint64_t wb = nbytes(f, {s.n, s.k});
  const bytes w = quant(f, s.n, s.k, g);
  // Copies of w to rotate through: 4 x L2 of weights, so every call reads DRAM.
  const int copies = hot ? 1 : int(std::max<uint64_t>(1, (4 * props.l2_bytes + wb - 1) / wb));
  std::vector<larch_tensor> W;
  for (int c = copies; c > 0; --c) W.push_back(tensor(B.put(w.data(), wb), f, s.n, s.k));
  return W;
}

// Microseconds per call(i): 3 warm-up calls, a 10-call direct probe that sizes iters to ~20 ms (at least 10),
// then iters calls captured into one graph and replayed between two events, as the engine's decode does
// (architecture 6.2), so host launch cost is out of the numbers. i counts calls, so w copies rotate through
// all of it.
template <class F>
static double time_us(F call) {
  int i = 0;
  const auto timed = [&](auto issue, int n) {
    larch_event *a, *b;
    CK(larch_stream_sync(str));
    CK(larch_event_record(str, &a));
    issue();
    CK(larch_event_record(str, &b));
    CK(larch_stream_sync(str));
    uint64_t ns;
    CK(larch_event_elapsed_ns(a, b, &ns));
    larch_event_destroy(a);
    larch_event_destroy(b);
    return double(ns) / n / 1e3;
  };
  const auto run = [&](int n) {
    return timed([&] { for (int j = 0; j < n; ++j) call(i++); }, n);
  };
  run(3);
  const double probe = run(10);
  const int iters = std::max(10, int(std::ceil(20e3 / probe)));
  CK(larch_stream_sync(str));
  larch_graph* g;
  CK(larch_graph_begin(str));
  for (int j = 0; j < iters; ++j) call(i++);
  CK(larch_graph_end(str, &g));
  const auto replay = [&] { return timed([&] { CK(larch_graph_launch(g, str)); }, iters); };
  replay();  // the first launch uploads the graph, which is not part of a decode step
  const double us = replay();
  larch_graph_destroy(g);
  return us;
}

__global__ void copy_k(const uint4* __restrict__ in, uint4* __restrict__ out, size_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    out[i] = in[i];
}

// The copy kernel's GB/s (2 x bytes / t, best of 20): what a memory-bound kernel can reach.
static double peak_gbps() {
  const size_t n = size_t(512) << 20;
  void *in, *out;
  CKC(cudaMalloc(&in, n));
  CKC(cudaMalloc(&out, n));
  CKC(cudaMemset(in, 1, n));
  cudaEvent_t a, b;
  CKC(cudaEventCreate(&a));
  CKC(cudaEventCreate(&b));
  const auto go = [&] {
    copy_k<<<props.compute_units * 8, 256>>>(static_cast<uint4*>(in), static_cast<uint4*>(out), n / 16);
  };
  go();
  float best = INFINITY;
  for (int i = 0; i < 20; ++i) {
    float ms;
    CKC(cudaEventRecord(a));
    go();
    CKC(cudaEventRecord(b));
    CKC(cudaEventSynchronize(b));
    CKC(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, ms);
  }
  CKC(cudaGetLastError());
  CKC(cudaEventDestroy(a));
  CKC(cudaEventDestroy(b));
  CKC(cudaFree(in));
  CKC(cudaFree(out));
  return 2.0 * double(n) / (double(best) * 1e-3) / 1e9;
}

struct opts {
  std::vector<std::string> fmt, model;
  std::vector<int> rows = {1, 2, 4, 8, 16};
  bool hot = false;
  double gate = -1;
};
static bool want(const std::vector<std::string>& sel, const std::string& name) {
  return sel.empty() || std::find(sel.begin(), sel.end(), name) != sel.end();
}

// One call's time for mmvq and for mmvq_verify, and the bytes either moves: w, x, bias and out.
struct point {
  double us, verify_us, bytes;
};
static std::vector<point> mmvq_tensor(const shape& s, larch_dtype f, const opts& o, int& copies) {
  bufs B;
  const std::vector<larch_tensor> W = weights(B, s, f, o.hot);
  copies = int(W.size());
  const bool head = is_head(s);
  const uint64_t ob = head ? 4 : 2;
  rng g(4321);
  const bytes xh = halves(size_t(16 * s.k), g);
  larch_tensor x = tensor(B.put(xh.data(), xh.size()), LARCH_F16, 16, s.k);
  const larch_tensor bt = bias_of(B, s, xh);
  larch_tensor out = tensor(B.put(nullptr, ob * 16 * uint64_t(s.n)), head ? LARCH_F32 : LARCH_F16, 16, s.n);
  std::vector<point> p;
  for (const int r : o.rows) {
    x.shape[0] = out.shape[0] = r;  // the row stride is the same for every r
    // Back to back, so clock drift hits both alike.
    const auto timed = [&](auto op) {
      return time_us([&](int i) { CK(op(str, &W[size_t(i % copies)], &x, s.bias ? &bt : nullptr, &out)); });
    };
    // At rows 1 larch_mmvq_verify runs larch_mmvq's kernel (mmvq_launch<T, 1>), so mmvq's time is verify's too.
    const double us = timed(larch_mmvq), verify_us = r == 1 ? us : timed(larch_mmvq_verify);
    p.push_back({us, verify_us,
                 double(nbytes(f, {s.n, s.k})) + 2.0 * r * s.k + (s.bias ? 4.0 * s.n : 0) + double(ob) * r * s.n});
  }
  return p;
}

static void print_header(const char* what, bool hot) {
  std::printf("# %s: %s, %u SMs, L2 %.0f MiB, theoretical %.0f GB/s, %s, graph replay\n\n", what, props.name,
              props.compute_units, double(props.l2_bytes) / (1 << 20), double(props.peak_gbps),
              hot ? "hot (one copy of w)" : "cold");
}

static int run_peak() {
  const double pk = peak_gbps();
  std::printf("## peak\n\n- copy kernel, 512 MiB to 512 MiB, best of 20, 2 x bytes / t: **%.1f GB/s**\n", pk);
  std::printf("- theoretical (props.peak_gbps): %.1f GB/s; copy / theoretical = %.3f\n", double(props.peak_gbps),
              pk / double(props.peak_gbps));
  return 0;
}

// Prints each (model, format) table as its tensors finish. The decode-step row is
// sum(count x bytes) / sum(count x time), for each kernel.
static int run_mmvq(const opts& o) {
  struct agg {
    std::string model, fmt;
    int rows;
    const char* kernel;
    double gbps;
  };
  std::vector<agg> judged;
  print_header("mmvq", o.hot);
  for (const model& m : MODELS) {
    if (!want(o.model, m.name)) continue;
    for (const larch_dtype f : QUANTS) {
      if (!want(o.fmt, DTYPE[f])) continue;
      const auto odd = std::ranges::find_if(m.t, [&](const shape& s) { return s.k % block_of(f) != 0; });
      if (odd != m.t.end()) {  // mmvq refuses it; the gate judges only the pairs that ran
        std::printf("%s %s skipped: %s's k %lld is not a whole number of %lld-weight blocks\n\n", m.name, DTYPE[f],
                    odd->name, (long long)odd->k, (long long)block_of(f));
        continue;
      }
      std::printf("## %s %s, cells: mmvq GB/s / verify GB/s (mmvq us / verify us)\n\n| tensor | n x k | copies |",
                  m.name, DTYPE[f]);
      for (const int r : o.rows) std::printf(" rows %d |", r);
      std::printf("\n|---|---|---|");
      for (size_t i = 0; i < o.rows.size(); ++i) std::printf("---|");
      std::printf("\n");
      std::vector<double> bytes_sum(o.rows.size()), us_sum(o.rows.size()), verify_sum(o.rows.size());
      for (const shape& s : m.t) {
        int copies;
        const std::vector<point> p = mmvq_tensor(s, f, o, copies);
        const int count = is_head(s) ? 1 : m.layers;
        std::printf("| %s | %lldx%lld | %d |", s.name, (long long)s.n, (long long)s.k, copies);
        for (size_t i = 0; i < p.size(); ++i) {
          std::printf(" %.1f / %.1f (%.2f / %.2f) |", p[i].bytes / p[i].us / 1e3, p[i].bytes / p[i].verify_us / 1e3,
                      p[i].us, p[i].verify_us);
          bytes_sum[i] += count * p[i].bytes;
          us_sum[i] += count * p[i].us;
          verify_sum[i] += count * p[i].verify_us;
        }
        std::printf("\n");
        std::fflush(stdout);
      }
      std::printf("| **decode step** (%d layers + lm_head), total us | | |", m.layers);
      for (size_t i = 0; i < o.rows.size(); ++i) {
        const double gbps = bytes_sum[i] / us_sum[i] / 1e3, verify_gbps = bytes_sum[i] / verify_sum[i] / 1e3;
        std::printf(" **%.1f / %.1f** (%.0f / %.0f) |", gbps, verify_gbps, us_sum[i], verify_sum[i]);
        const bool one = o.rows[i] == 1;
        judged.push_back({m.name, DTYPE[f], o.rows[i], one ? "mmvq" : "mmvq_verify", one ? gbps : verify_gbps});
      }
      const auto r1 = std::ranges::find(o.rows, 1);
      std::printf("\n| verify step us / mmvq step us at rows 1 | | |");
      for (const double v : verify_sum) {
        if (r1 == o.rows.end()) std::printf(" - |");
        else std::printf(" x%.2f of one row |", v / us_sum[r1 - o.rows.begin()]);
      }
      std::printf("\n\n");
    }
  }
  if (o.gate < 0) return 0;
  if (judged.empty()) die("--gate", "no (model, format) pair ran");
  const double pk = peak_gbps();
  int rc = 0;
  std::printf("## gate: decode-step aggregate >= %.2f x peak (%.1f GB/s); rows 1 judges mmvq, rows >= 2 verify\n\n",
              o.gate, pk);
  std::printf("| model | format | rows | kernel | GB/s | / peak | |\n|---|---|---|---|---|---|---|\n");
  for (const agg& a : judged) {
    const bool ok = a.gbps >= o.gate * pk;
    rc |= !ok;
    std::printf("| %s | %s | %d | %s | %.1f | %.3f | %s |\n", a.model.c_str(), a.fmt.c_str(), a.rows, a.kernel,
                a.gbps, a.gbps / pk, ok ? "pass" : "FAIL");
  }
  return rc;
}

// The tensors of --model (m1 by default) in their GGUF formats: mmvq against dequant + gemm over rows, and
// dequant alone.
static int run_cross(const opts& o) {
  static const int rows[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 24, 32, 48, 64, 96, 128};
  print_header("cross", false);
  for (const model& m : MODELS) {
    if (!want(o.model, m.name)) continue;
    for (const shape& s : m.t) {
      for (const std::string& name : split(s.gguf)) {
        if (!want(o.fmt, name)) continue;
        const larch_dtype f = quant_named(name);
        bufs B;
        const std::vector<larch_tensor> W = weights(B, s, f, false);
        const int copies = int(W.size());
        const bool head = is_head(s);
        const uint64_t wb = nbytes(f, {s.n, s.k});
        rng g(4321);
        const bytes xh = halves(size_t(128 * s.k), g);
        larch_tensor x = tensor(B.put(xh.data(), xh.size()), LARCH_F16, 128, s.k);
        const larch_tensor bt = bias_of(B, s, xh);
        const larch_tensor* bias = s.bias ? &bt : nullptr;
        larch_tensor out =
            tensor(B.put(nullptr, (head ? 4 : 2) * 128 * uint64_t(s.n)), head ? LARCH_F32 : LARCH_F16, 128, s.n);
        const larch_tensor scratch = tensor(B.put(nullptr, 2 * uint64_t(s.n * s.k)), LARCH_F16, s.n, s.k);
        const auto dq = [&](int i) { CK(larch_dequant(str, &W[size_t(i % copies)], &scratch)); };
        const double dq_us = time_us(dq);
        std::printf("## %s %s %s %lldx%lld (%d copies): dequant alone %.2f us, %.1f GB/s\n\n", m.name, s.name,
                    name.c_str(), (long long)s.n, (long long)s.k, copies, dq_us,
                    double(wb + 2 * uint64_t(s.n * s.k)) / dq_us / 1e3);
        std::puts("dequant + gemm: up to 8 rows that is gemm_rows_k, not cuBLAS.");
        std::printf("| rows | mmvq us | dequant + gemm us |\n|---|---|---|\n");
        for (const int r : rows) {
          x.shape[0] = out.shape[0] = r;
          const double pair_us = time_us([&](int i) {
            dq(i);
            CK(larch_gemm(str, &x, &scratch, carry(s, bias), &out, 1.0f, 0.0f));
          });
          if (r <= 16) {
            const double us =
                time_us([&](int i) { CK(larch_mmvq(str, &W[size_t(i % copies)], &x, carry(s, bias), &out)); });
            std::printf("| %d | %.2f | %.2f |\n", r, us, pair_us);
          } else {
            std::printf("| %d | | %.2f |\n", r, pair_us);
          }
          std::fflush(stdout);
        }
        std::printf("\n");
      }
    }
  }
  return 0;
}

[[noreturn]] static void usage() {
  std::fprintf(stderr,
               "usage: mmvq_bench peak | mmvq [--fmt a,b] [--rows 1,2] [--model m1,m5,m8] [--hot] [--gate F]"
               " | cross [--fmt a,b] [--model m1,m5,m8]\n");
  std::exit(2);
}

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd != "peak" && cmd != "mmvq" && cmd != "cross") usage();
  opts o;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const char* val = i + 1 < argc ? argv[i + 1] : nullptr;
    // Refused, never ignored, so a run is what its command line says: peak takes no option, cross only --fmt
    // and --model.
    if (cmd == "peak" || (cmd == "cross" && a != "--fmt" && a != "--model"))
      die(argv[i], ("not an option of " + cmd).c_str());
    if (a == "--hot") {
      o.hot = true;
    } else if (a == "--fmt" && val) {
      o.fmt = split(argv[++i]);
      for (const std::string& n : o.fmt) quant_named(n);
    } else if (a == "--model" && val) {
      o.model = split(argv[++i]);
      for (const std::string& n : o.model)
        if (std::none_of(std::begin(MODELS), std::end(MODELS), [&](const model& m) { return n == m.name; }))
          die("unknown model", n.c_str());
    } else if (a == "--rows" && val) {
      o.rows.clear();
      for (const std::string& n : split(argv[++i])) {
        char* end;
        const long r = std::strtol(n.c_str(), &end, 10);
        if (*end || r < 1 || r > 16) die("--rows", "each value must be 1..16");
        o.rows.push_back(int(r));
      }
      if (o.rows.empty()) die("--rows", "needs at least one value");
    } else if (a == "--gate" && val) {
      char* end;
      o.gate = std::strtod(argv[++i], &end);
      if (*end || !(o.gate > 0 && o.gate <= 1)) die("--gate", "needs a fraction in (0, 1]");
    } else {
      usage();
    }
  }
  if (cmd == "cross") {
    if (o.model.empty()) o.model = {"m1"};
    bool any = false;  // a selection that measures nothing is malformed too, refused before the GPU is touched
    for (const model& m : MODELS)
      for (const shape& s : m.t)
        for (const std::string& f : split(s.gguf)) any |= want(o.model, m.name) && want(o.fmt, f);
    if (!any) die("cross", "--model and --fmt select no tensor");
  }
  CK(larch_device_open(0, &dev));
  CK(larch_device_props_get(dev, &props));
  CK(larch_stream_create(dev, &str));
  const int rc = cmd == "peak" ? run_peak() : cmd == "mmvq" ? run_mmvq(o) : run_cross(o);
  larch_stream_destroy(str);
  larch_device_close(dev);
  return rc;
}
