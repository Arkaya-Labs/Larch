// Direct quantized GEMM against the existing CPU dequant + GEMM oracle and its unchanged tolerances.
// Reuse ops.cpp's data, guarded-buffer checks and comparison instead of a second test harness.
#include <array>
#include <cfenv>
#include <tuple>

#define main larch_ops_test_main
#include "ops.cpp"

// dispatch.cpp's debug hook, outside larch.h: its callers declare it.
extern "C" int vv_fault_quant_i8_probe;
#undef main

// The formats quant_gemm_i8 takes; the others must refuse.
static const larch_dtype I8_FORMATS[] = {LARCH_Q4_K, LARCH_Q5_K, LARCH_Q6_K, LARCH_Q8_0, LARCH_Q4_0, LARCH_Q5_0};

// Timing uses graph replay, cold weights by default, and the same operation and bias on every path.
// Run only with the exclusive GPU lock: quant_gemm --bench q4_k n k [f32] [bias] [hot] [rows=16];
// --bench-i8 times quant_gemm_i8 (its split-K entry with [split]).
static int quant_bench(const side& dut, const larch_device_props& props, int argc, char** argv, bool i8 = false) {
  if (argc < 5) return 2;
  larch_dtype fmt = LARCH_F32;
  for (const larch_dtype f : QUANTS)
    if (std::strcmp(argv[2], DTYPE[f]) == 0) fmt = f;
  char *end_n = nullptr, *end_k = nullptr;
  const int64_t n = std::strtoll(argv[3], &end_n, 10), k = std::strtoll(argv[4], &end_k, 10);
  if (fmt == LARCH_F32 || *end_n || *end_k || n <= 0 || k <= 0 || n > INT32_MAX || k > INT32_MAX ||
      k % block_of(fmt) != 0 || uint64_t(n) * uint64_t(k) > (uint64_t(1) << 31)) return 2;
  if (i8 && std::find(std::begin(I8_FORMATS), std::end(I8_FORMATS), fmt) == std::end(I8_FORMATS)) return 2;
  bool f32 = false, bias = false, hot = false, split = false;
  int64_t selected_rows = 0;
  for (int i = 5; i < argc; ++i) {
    if (!std::strcmp(argv[i], "f32")) f32 = true;
    else if (!std::strcmp(argv[i], "bias")) bias = true;
    else if (!std::strcmp(argv[i], "hot")) hot = true;
    else if (i8 && !std::strcmp(argv[i], "split")) split = true;
    else if (!std::strncmp(argv[i], "rows=", 5)) {
      char* end_rows = nullptr;
      selected_rows = std::strtoll(argv[i] + 5, &end_rows, 10);
      if (*end_rows || selected_rows < 1 || selected_rows > 4096) return 2;
    }
    else return 2;
  }
  rng g(4321);
  const int64_t max_rows = std::max(int64_t(512), selected_rows);
  const bytes weights = quant(fmt, n, k, g), input = halves(size_t(max_rows * k), g, 0);
  std::vector<larch_buffer*> buffers;
  const auto alloc = [&](uint64_t size, const void* data = nullptr) {
    larch_buffer* buffer = nullptr;
    if (larch_buffer_alloc(dut.dev, size, &buffer) != LARCH_OK) return static_cast<larch_buffer*>(nullptr);
    buffers.push_back(buffer);
    if (data && larch_upload(dut.s, buffer, 0, data, size) != LARCH_OK) return static_cast<larch_buffer*>(nullptr);
    return buffer;
  };
  const int copies = hot ? 1 : int(std::max(uint64_t(1), (4 * props.l2_bytes + weights.size() - 1) / weights.size()));
  std::vector<larch_tensor> w;
  for (int i = 0; i < copies; ++i)
    w.push_back({alloc(weights.size(), weights.data()), 0, fmt, 2, {n, k}, {k / block_of(fmt), 1}});
  larch_tensor x = {alloc(input.size(), input.data()), 0, LARCH_F16, 2, {max_rows, k}, {k, 1}};
  larch_tensor out = {alloc(uint64_t(max_rows * n) * (f32 ? 4 : 2)), 0, f32 ? LARCH_F32 : LARCH_F16, 2,
                         {max_rows, n}, {n, 1}};
  const larch_tensor dq = {alloc(uint64_t(n * k) * 2), 0, LARCH_F16, 2, {n, k}, {k, 1}};
  const bytes bdata = bias ? floats(size_t(n), g, -1, 1) : bytes();
  const larch_tensor b = {bias ? alloc(bdata.size(), bdata.data()) : nullptr, 0, LARCH_F32, 1, {n}, {1}};
  bool ok = x.buf && out.buf && dq.buf && (!bias || b.buf);
  for (const auto& weight : w) ok = ok && weight.buf;
  if (i8) std::puts("INT8 activation route; packing/scratch included; old routes retain their own numerical policies.");
  std::printf("%s %lldx%lld %s%s, %s weights (%d copies), graph replay, microseconds\n",
              DTYPE[fmt], (long long)n, (long long)k, f32 ? "f32" : "f16", bias ? " bias" : "",
              hot ? "hot" : "cold", copies);
  // Production never takes dequant + gemm at these small rows, so this column is no cuBLAS baseline there.
  std::puts("dequant+gemm: on CUDA up to 8 rows that is gemm_rows_k, not cuBLAS.");
  std::puts("| rows | direct quant | sliced mmvq | dequant+gemm | direct/slices | direct/dq+gemm |\n"
            "|---:|---:|---:|---:|---:|---:|");
  const std::vector<int64_t> bench_rows = selected_rows ? std::vector<int64_t>{selected_rows}
      : std::vector<int64_t>{1, 2, 4, 8, 16, 17, 24, 32, 48, 64, 96, 128, 256, 512};
  for (const int64_t rows : bench_rows) {
    if (!ok) break;
    x.shape[0] = out.shape[0] = rows;
    const auto weight = [&](int i) { return &w[size_t(i % copies)]; };
    const auto op = !i8 ? larch_quant_gemm : split ? larch_quant_gemm_i8_split : larch_quant_gemm_i8;
    const double direct = replay_time(dut.s, [&](int i) { return op(dut.s, weight(i), &x, bias ? &b : nullptr, &out); });
    const double slices = replay_time(dut.s, [&](int i) {
      for (int64_t at = 0; at < rows; at += 16) {
        larch_tensor xs = x, os = out;
        xs.offset = uint64_t(at * k) * 2;
        os.offset = uint64_t(at * n) * (f32 ? 4 : 2);
        xs.shape[0] = os.shape[0] = std::min(int64_t(16), rows - at);
        const larch_status status = larch_mmvq(dut.s, weight(i), &xs, bias ? &b : nullptr, &os);
        if (status != LARCH_OK) return status;
      }
      return LARCH_OK;
    });
    const double gemm = replay_time(dut.s, [&](int i) {
      const larch_status status = larch_dequant(dut.s, weight(i), &dq);
      return status == LARCH_OK ? larch_gemm(dut.s, &x, &dq, bias ? &b : nullptr, &out, 1, 0) : status;
    });
    std::printf("| %lld | %.2f | %.2f | %.2f | %.3f | %.3f |\n", (long long)rows, direct, slices, gemm,
                direct / slices, direct / gemm);
    std::fflush(stdout);
    ok = direct >= 0 && slices >= 0 && gemm >= 0;
  }
  if (!ok) std::fprintf(stderr, "quant_gemm bench failed: %s\n", larch_last_error(dut.dev));
  if (larch_stream_sync(dut.s) != LARCH_OK) ok = false;
  for (larch_buffer* buffer : buffers) larch_buffer_free(buffer);
  return ok ? 0 : 1;
}

static larch_status quant_reference(larch_stream* s, larch_device* dev, const larch_tensor* t, bool bias) {
  larch_buffer* scratch = nullptr;
  const int64_t n = t[0].shape[0], k = t[0].shape[1];
  larch_status status = larch_buffer_alloc(dev, uint64_t(n * k) * 2, &scratch);
  if (status != LARCH_OK) return status;
  const larch_tensor dq = {scratch, 0, LARCH_F16, 2, {n, k}, {k, 1}};
  status = larch_dequant(s, &t[0], &dq);
  if (status == LARCH_OK) status = larch_gemm(s, &t[1], &dq, bias ? &t[3] : nullptr, &t[2], 1.0f, 0.0f);
  larch_buffer_free(scratch);
  return status;
}

static test quant_case(const side& ref, larch_dtype fmt, int64_t m, int64_t n, int64_t k, larch_dtype output,
                       bool bias, bool skew, bool padded) {
  const std::string name = std::string("quant_gemm ") + DTYPE[fmt] + " " + std::to_string(m) + "x" +
                           std::to_string(n) + "x" + std::to_string(k) + " " + DTYPE[output] +
                           (bias ? " bias" : "") + (skew ? " offset +2" : "") + (padded ? " padded" : "");
  auto g = seeded(name);
  const int64_t wp = padded ? 1 : 0, xp = padded ? 3 : 0, op = padded ? 5 : 0;
  std::vector<arg> args = {in(fmt, {n, k}, quant(fmt, n, k + wp * block_of(fmt), g)),
                           in(LARCH_F16, {m, k}, halves(size_t(m * (k + xp)), g)),
                           out("gemm", output, {m, n}, bytes(size_t(m * (n + op) * block_bytes(output)), 0xff))};
  args[0].pad = wp;
  args[1].pad = xp;
  args[2].pad = op;
  if (skew) {
    args[0].skew = args[1].skew = 2;
    args[2].skew = output == LARCH_F16 ? 2 : 4;
  }
  if (bias) {
    args.push_back(in(LARCH_F32, {n}, floats(size_t(n * (padded ? 3 : 1)), g, -1, 1)));
    args.back().pad = padded ? 2 : 0;
  }
  return {name, std::move(args), [ref, bias](larch_stream* s, const larch_tensor* t) {
            return s == ref.s ? quant_reference(s, ref.dev, t, bias)
                              : larch_quant_gemm(s, &t[0], &t[1], bias ? &t[3] : nullptr, &t[2]);
          }};
}

// The independent CPU implementation quantizes activations by the new explicit contract.
// Reuse the unchanged GEMM comparison tolerances; activation approximation is not hidden in them.
static test quant_i8_case(int64_t m, int64_t n, int64_t k, larch_dtype output, bool bias,
                          larch_dtype fmt = LARCH_Q4_K) {
  test c = quant_case(side{}, fmt, m, n, k, output, bias, true, true);
  c.name = "quant_gemm_i8 " + c.name;
  c.run = [bias](larch_stream* s, const larch_tensor* t) {
    return larch_quant_gemm_i8(s, &t[0], &t[1], bias ? &t[3] : nullptr, &t[2]);
  };
  return c;
}

// The split route changes execution only; its independent numerical oracle is the existing CPU policy.
static test quant_i8_split_case(const side& ref, int64_t m, int64_t n, int64_t k, larch_dtype output,
                                larch_dtype fmt = LARCH_Q4_K) {
  test c = quant_i8_case(m, n, k, output, true, fmt);
  c.name = "quant_gemm_i8_split " + c.name;
  c.run = [ref](larch_stream* stream, const larch_tensor* t) {
    return (stream == ref.s ? larch_quant_gemm_i8 : larch_quant_gemm_i8_split)(
        stream, &t[0], &t[1], &t[3], &t[2]);
  };
  return c;
}

// Resource-optional fast paths report skips explicitly, while preserving a checked capture fallback.
static int check_i8_split(const side& ref, const side& dut, const test& c, int& skipped) {
  test probe = c;
  probe.run = [](larch_stream* stream, const larch_tensor* t) {
    return larch_quant_gemm_i8_split(stream, &t[0], &t[1], &t[3], &t[2]);
  };
  std::vector<bytes> after;
  const int status = run_on(dut, probe, after);
  if (status == LARCH_OK) return check(ref, dut, c);
  if (status != LARCH_ERR_UNSUPPORTED) {
    std::fprintf(stderr, "%s: fast-path probe failed with %d\n", c.name.c_str(), status);
    return 1;
  }
  bool unchanged = true;
  for (size_t i = 0; i < after.size(); ++i) unchanged = unchanged && after[i] == image(i, probe.args[i]);
  ++skipped;
  std::printf("SKIPPED fast path: %s; checking unchanged buffers and capture fallback\n", c.name.c_str());
  test fallback = c;
  fallback.name += " unavailable resources capture fallback";
  fallback.run = [ref](larch_stream* stream, const larch_tensor* t) {
    if (stream == ref.s) return larch_quant_gemm_i8(stream, &t[0], &t[1], &t[3], &t[2]);
    larch_status result = larch_graph_begin(stream);
    if (result != LARCH_OK) return result;
    const larch_status refused = larch_quant_gemm_i8_split(stream, &t[0], &t[1], &t[3], &t[2]);
    result = refused == LARCH_ERR_UNSUPPORTED
                 ? larch_quant_gemm_i8(stream, &t[0], &t[1], &t[3], &t[2]) : LARCH_ERR_INVALID_ARG;
    larch_graph* graph = nullptr;
    const larch_status ended = larch_graph_end(stream, &graph);
    if (result == LARCH_OK) result = ended;
    if (result == LARCH_OK) result = larch_graph_launch(graph, stream);
    if (result == LARCH_OK) result = larch_stream_sync(stream);
    if (graph) larch_graph_destroy(graph);
    return result;
  };
  return int(!unchanged) + check(ref, dut, fallback);
}

// Check the oracle itself with hand-computed ties and original-sum minimum corrections.
static test quant_i8_known() {
  bytes weights(3 * 144, 0), input(256 * 2, 0);
  for (int col = 0; col < 3; ++col) {
    uint8_t* block = weights.data() + col * 144;
    const uint16_t d = 0x3c00, min = 0x3800;
    std::memcpy(block, &d, 2);
    std::memcpy(block + 2, &min, 2);
    std::fill(block + 4, block + 12, uint8_t(1));
    std::fill(block + 12, block + 16, uint8_t(0x11));
    std::fill(block + 16, block + 144, uint8_t(col == 0 ? 0x11 : col == 1 ? 0 : 0xff));
  }
  // max=127, scale=1, codes=[127,0,2,2], original sum=131.5.
  const uint16_t values[] = {0x57f0, 0x3800, 0x3e00, 0x4100};
  std::memcpy(input.data(), values, sizeof(values));
  test c{"quant_gemm_i8 hand-computed oracle",
         {in(LARCH_Q4_K, {3, 256}, std::move(weights)), in(LARCH_F16, {1, 256}, std::move(input)),
          out("gemm", LARCH_F32, {1, 3}), in(LARCH_F32, {3}, raw(std::vector<float>(3, 0.25f)))},
         [](larch_stream* stream, const larch_tensor* t) {
           return larch_quant_gemm_i8(stream, &t[0], &t[1], &t[3], &t[2]);
         }};
  return c;
}

// Q6_K's two sub-block scales per 32-group and its -32 code offset, by hand: every code is +1 (row 0) or -1 (row 1),
// activations 127, 0.5, 1.5, 2.5 at elements 0-3 and 127 at 16 (scale 1, codes 127, 0, 2, 2 and 127), so row 0 is
// 1 * 131 + 2 * 127 and row 1 -(-3 * 131 + 5 * 127), plus the 0.25 bias.
static test quant_i8_known_q6() {
  bytes weights(2 * 210, 0), input(256 * 2, 0);
  for (int col = 0; col < 2; ++col) {
    uint8_t* block = weights.data() + col * 210;
    std::fill(block, block + 128, uint8_t(col ? 0xff : 0x11));
    std::fill(block + 128, block + 192, uint8_t(col ? 0x55 : 0xaa));
    block[192] = uint8_t(col ? -3 : 1);
    block[193] = uint8_t(col ? 5 : 2);
    const uint16_t d = 0x3c00;
    std::memcpy(block + 208, &d, 2);
  }
  const uint16_t values[] = {0x57f0, 0x3800, 0x3e00, 0x4100};
  std::memcpy(input.data(), values, sizeof(values));
  std::memcpy(input.data() + 32, values, 2);
  return {"quant_gemm_i8 q6_k hand-computed oracle",
          {in(LARCH_Q6_K, {2, 256}, std::move(weights)), in(LARCH_F16, {1, 256}, std::move(input)),
           out("gemm", LARCH_F32, {1, 2}), in(LARCH_F32, {2}, raw(std::vector<float>(2, 0.25f)))},
          [](larch_stream* stream, const larch_tensor* t) {
            return larch_quant_gemm_i8(stream, &t[0], &t[1], &t[3], &t[2]);
          }};
}

// One 256-wide row of fmt with every code 3 and scale 0.5: 0.5 * 3 * 131 + 0.25 bias with the Q4_K case's
// activations (codes 127, 0, 2, 2, scale 1).
static test quant_i8_known_code3(larch_dtype fmt) {
  const int64_t size = block_bytes(fmt);
  bytes weights(size_t(8 * size), 0), input(256 * 2, 0);
  for (int b = 0; b < 8; ++b) {
    uint8_t* block = weights.data() + b * size;
    const uint16_t d = 0x3800;
    std::memcpy(block, &d, 2);
    if (fmt == LARCH_Q8_0) std::fill(block + 2, block + 34, uint8_t(3));
    if (fmt == LARCH_Q4_0) std::fill(block + 2, block + 18, uint8_t(0xbb));  // 11 - 8
    if (fmt == LARCH_Q5_0) {  // (3 | 16) - 16
      std::fill(block + 2, block + 6, uint8_t(0xff));
      std::fill(block + 6, block + 22, uint8_t(0x33));
    }
  }
  const uint16_t values[] = {0x57f0, 0x3800, 0x3e00, 0x4100};
  std::memcpy(input.data(), values, sizeof(values));
  return {std::string("quant_gemm_i8 ") + DTYPE[fmt] + " hand-computed oracle",
          {in(fmt, {1, 256}, std::move(weights)), in(LARCH_F16, {1, 256}, std::move(input)),
           out("gemm", LARCH_F32, {1, 1}), in(LARCH_F32, {1}, raw(std::vector<float>(1, 0.25f)))},
          [](larch_stream* stream, const larch_tensor* t) {
            return larch_quant_gemm_i8(stream, &t[0], &t[1], &t[3], &t[2]);
          }};
}

// Q5_K: d 1, dmin 0.5, every scale and min 1, every code 16 + 1, so 17 * 131 - 0.5 * 131.5 + 0.25 bias.
static test quant_i8_known_q5k() {
  bytes weights(176, 0), input(256 * 2, 0);
  const uint16_t d = 0x3c00, min = 0x3800;
  std::memcpy(weights.data(), &d, 2);
  std::memcpy(weights.data() + 2, &min, 2);
  std::fill(weights.begin() + 4, weights.begin() + 12, uint8_t(1));
  std::fill(weights.begin() + 12, weights.begin() + 16, uint8_t(0x11));
  std::fill(weights.begin() + 16, weights.begin() + 48, uint8_t(0xff));
  std::fill(weights.begin() + 48, weights.end(), uint8_t(0x11));
  const uint16_t values[] = {0x57f0, 0x3800, 0x3e00, 0x4100};
  std::memcpy(input.data(), values, sizeof(values));
  return {"quant_gemm_i8 q5_k hand-computed oracle",
          {in(LARCH_Q5_K, {1, 256}, std::move(weights)), in(LARCH_F16, {1, 256}, std::move(input)),
           out("gemm", LARCH_F32, {1, 1}), in(LARCH_F32, {1}, raw(std::vector<float>(1, 0.25f)))},
          [](larch_stream* stream, const larch_tensor* t) {
            return larch_quant_gemm_i8(stream, &t[0], &t[1], &t[3], &t[2]);
          }};
}

// The same call captured in a graph and replayed once, where the backend has graphs (the engine
// replays decode projections); the CPU reference runs it directly.
static test captured(test c) {
  c.name += " graph";
  c.run = [run = c.run](larch_stream* s, const larch_tensor* t) {
    larch_status status = larch_graph_begin(s);
    if (status == LARCH_ERR_UNSUPPORTED) return run(s, t);
    if (status != LARCH_OK) return status;
    status = run(s, t);
    larch_graph* graph = nullptr;
    const larch_status ended = larch_graph_end(s, &graph);
    if (status == LARCH_OK) status = ended;
    if (status == LARCH_OK) status = larch_graph_launch(graph, s);
    if (status == LARCH_OK) status = larch_stream_sync(s);
    if (graph) larch_graph_destroy(graph);
    return status;
  };
  return c;
}

int main(int argc, char** argv) {
  int32_t gpus = 0, ordinal = 0;
  if (larch_device_count(&gpus) != LARCH_OK || !pick(gpus, &ordinal)) return 1;
  side ref{}, dut{};
  if (larch_device_open(-1, &ref.dev) != LARCH_OK || larch_stream_create(ref.dev, &ref.s) != LARCH_OK ||
      larch_device_open(ordinal, &dut.dev) != LARCH_OK || larch_stream_create(dut.dev, &dut.s) != LARCH_OK)
    return 1;
  larch_device_props props{};
  if (larch_device_props_get(dut.dev, &props) != LARCH_OK) return 1;
  const test known = quant_i8_known();
  const float expected[] = {65.5f, -65.5f, 1899.5f};
  const int previous_round = std::fegetround();
  bool known_good = true;
  for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    std::vector<bytes> known_after;
    known_good = std::fesetround(mode) == 0 && known_good;
    const int known_status = run_on(ref, known, known_after);
    known_good = known_status == LARCH_OK && known_good;
    if (known_status == LARCH_OK)
      for (size_t i = 0; i < 3; ++i)
        known_good = known_good && elem(known_after[2].data() + lead(2, known.args[2]) + 4 * i, LARCH_F32) == expected[i];
  }
  known_good = std::fesetround(previous_round) == 0 && known_good;
  const test known6 = quant_i8_known_q6();
  {
    std::vector<bytes> after;
    known_good = run_on(ref, known6, after) == LARCH_OK && known_good;
    const float expected6[] = {385.25f, -241.75f};
    for (size_t i = 0; known_good && i < 2; ++i)
      known_good = elem(after[2].data() + lead(2, known6.args[2]) + 4 * i, LARCH_F32) == expected6[i];
  }
  std::vector<test> known3;
  for (const larch_dtype fmt : {LARCH_Q8_0, LARCH_Q4_0, LARCH_Q5_0, LARCH_Q5_K}) {
    known3.push_back(fmt == LARCH_Q5_K ? quant_i8_known_q5k() : quant_i8_known_code3(fmt));
    std::vector<bytes> after;
    known_good = run_on(ref, known3.back(), after) == LARCH_OK && known_good;
    known_good = known_good && elem(after[2].data() + lead(2, known3.back().args[2]), LARCH_F32) ==
                                   (fmt == LARCH_Q5_K ? 2161.5f : 196.75f);
  }
  if (!known_good) {
    std::fputs("quant_gemm_i8 hand-computed CPU oracle or ambient rounding independence failed\n", stderr);
    return 1;
  }
  // Independent oracle assertions also run in the CPU-only build: all three projections are NaN,
  // including a zero-code weight row, whether the invalid activation is in the first or last group.
  for (const uint16_t value : {uint16_t(0x7e00), uint16_t(0x7c00), uint16_t(0xfc00)})
    for (const size_t column : {size_t(0), size_t(255)}) {
      test c = quant_i8_known();
      bytes input = *c.args[1].data;
      std::memcpy(input.data() + column * 2, &value, 2);
      c.args[1].data = std::make_shared<const bytes>(std::move(input));
      std::vector<bytes> after;
      const int status = run_on(ref, c, after);
      bool good = status == LARCH_OK;
      if (good)
        for (size_t i = 0; i < 3; ++i)
          good = good && std::isnan(elem(after[2].data() + lead(2, c.args[2]) + 4 * i, LARCH_F32));
      if (!good) {
        std::fputs("quant_gemm_i8 nonfinite CPU oracle failed\n", stderr);
        return 1;
      }
    }
  // Q3_K and Q2_K have no int8 policy: both the oracle and the device refuse them, with or without rows, and write
  // nothing.
  for (const larch_dtype q : {LARCH_Q3_K, LARCH_Q2_K}) {
    for (const int64_t rows : {0, 1}) {
      const test c = quant_i8_case(rows, 1, 256, LARCH_F16, false, q);
      for (const side* on : {&ref, &dut}) {
        std::vector<bytes> after;
        const int status = run_on(*on, c, after);
        bool good = status == (on == &ref ? LARCH_ERR_INVALID_ARG : LARCH_ERR_UNSUPPORTED) ||
                    (on == &dut && props.backend == LARCH_BACKEND_CPU && status == LARCH_ERR_INVALID_ARG);
        for (size_t i = 0; i < after.size(); ++i) good = good && after[i] == image(i, c.args[i]);
        if (!good) {
          std::fprintf(stderr, "%s: expected a refusal with unchanged buffers, got %d\n", c.name.c_str(), status);
          return 1;
        }
      }
    }
  }
  // Unsupported backends retain their existing projection path; this test exercises CUDA's new path.
  if (props.backend != LARCH_BACKEND_CUDA) {
    larch_stream_destroy(ref.s);
    larch_stream_destroy(dut.s);
    larch_device_close(ref.dev);
    larch_device_close(dut.dev);
    std::puts("quant_gemm: CUDA path skipped on this backend");
    return 0;
  }
  if (argc > 1) {
    const int result = std::strcmp(argv[1], "--bench") == 0 ? quant_bench(dut, props, argc, argv)
                       : std::strcmp(argv[1], "--bench-i8") == 0 ? quant_bench(dut, props, argc, argv, true) : 2;
    larch_stream_destroy(ref.s);
    larch_stream_destroy(dut.s);
    larch_device_close(ref.dev);
    larch_device_close(dut.dev);
    return result;
  }
  int wrong = check(ref, dut, known) + check(ref, dut, known6), count = 2, split_skipped = 0;
  for (const test& c : known3) {
    wrong += check(ref, dut, c);
    ++count;
  }
  for (const larch_dtype fmt : {LARCH_Q4_K, LARCH_Q6_K, LARCH_Q8_0}) {
    const int64_t block = block_of(fmt);
    for (const int64_t m : {0, 1, 2, 7, 16, 17, 31, 32, 33, 63, 64, 65, 97, 128}) {
      for (const larch_dtype output : {LARCH_F16, LARCH_F32}) {
        for (const auto& [n, blocks] : {std::pair<int64_t, int64_t>{1, 1}, {17, 3}, {67, 11}}) {
          const test c = quant_case(ref, fmt, m, n, block * blocks, output, m % 2 != 0, m % 3 != 0,
                                    m % 2 == 0 || m == 17 || m == 65);
          wrong += check(ref, dut, c);
          ++count;
        }
      }
    }
    // The second panel and its odd tail must preserve output row strides and the bias offset.
    for (const larch_dtype output : {LARCH_F16, LARCH_F32}) {
      wrong += check(ref, dut, quant_case(ref, fmt, 97, 2051, 256, output, true, true, true));
      ++count;
    }
  }
  // Exercise the wide M16 tile, including its odd output tail and unaligned padded rows. On a 16-SM
  // card (RTX 3050) its 129 tiles exceed one resident wave, so these run the M32 tile there, and the
  // split-K cases below are sm_120 only: a 3050 run covers neither variant.
  for (const int64_t rows : {8, 16})
    for (const larch_dtype output : {LARCH_F16, LARCH_F32}) {
      wrong += check(ref, dut, quant_case(ref, LARCH_Q4_K, rows, 8193, 256, output, true, true, true));
      ++count;
    }
  // Ragged query tiles and mixed prefill/decode row counts with padded, unaligned operands.
  for (const int64_t rows : {9, 15, 520, 528})
    for (const larch_dtype output : {LARCH_F16, LARCH_F32}) {
      wrong += check(ref, dut, quant_case(ref, LARCH_Q4_K, rows, 67, 768, output, true, true, true));
      ++count;
    }
  // Nine 256-weight blocks require two splits; eleven allow four with a 512-weight last partition.
  for (const larch_dtype fmt : {LARCH_Q4_K, LARCH_Q6_K})
    for (const int64_t rows : {8, 16})
      for (const int64_t blocks : {9, 11}) {
        const larch_dtype output = (rows == 8) == (blocks == 9) ? LARCH_F16 : LARCH_F32;
        wrong += check(ref, dut, quant_case(ref, fmt, rows, 1025, blocks * 256, output, true, true, true));
        ++count;
      }
  // An f16 output with bias past 2048 rows: the f32 result bounds each panel below 2048 columns.
  wrong += check(ref, dut, quant_case(ref, LARCH_Q4_K, 2049, 2051, 256, LARCH_F16, true, false, true));
  ++count;
  // Every direct sub-kernel replayed from a graph: M32, M64, split-K, M16 and cuBLAS panels.
  for (const auto& [fmt, m, n, k] : {std::tuple<larch_dtype, int64_t, int64_t, int64_t>{LARCH_Q4_K, 31, 67, 768},
                                     {LARCH_Q6_K, 64, 67, 768}, {LARCH_Q4_K, 16, 1025, 2304},
                                     {LARCH_Q6_K, 8, 1025, 2816}, {LARCH_Q4_K, 16, 8193, 256},
                                     {LARCH_Q8_0, 97, 2051, 256}}) {
    wrong += check(ref, dut, captured(quant_case(ref, fmt, m, n, k, LARCH_F32, true, true, true)));
    ++count;
  }
  for (const auto& [fmt, k] : {std::pair<larch_dtype, int64_t>{LARCH_Q4_K, 4096},
                               {LARCH_Q4_K, 14336}, {LARCH_Q6_K, 14336}}) {
    wrong += check(ref, dut, quant_case(ref, fmt, 16, 67, k, LARCH_F32, true, true, true));
    ++count;
  }
  // The R8/R16 MMVQ variants also need bias, odd columns and padded unaligned rows together.
  for (const larch_dtype fmt : {LARCH_Q4_K, LARCH_Q6_K})
    for (const int64_t rows : {8, 16}) {
      mmvq_case(fmt, rows, 67, 4608, LARCH_F32, true, 2, 2, true);
      wrong += check(ref, dut, tests.back());
      tests.pop_back();
      ++count;
    }
  for (const int invalid_case : {0, 1, 2, 3}) {
    test c = quant_case(ref, LARCH_Q4_K, 33, 17, 768, LARCH_F16, true, false, false);
    c.name += " invalid " + std::to_string(invalid_case);
    c.want = LARCH_ERR_INVALID_ARG;
    for (arg& a : c.args) a.out = false;
    if (invalid_case == 0) c.args[1].t = LARCH_BF16;
    if (invalid_case == 1) c.args[2].shape[0] = 32;
    if (invalid_case == 2) c.args[2].shape[1] = 16;
    if (invalid_case == 3) c.args[3].shape[0] = 16;
    wrong += check(ref, dut, c);
    ++count;
  }
  // Q3_K too: its prefill keeps dequant + gemm.
  for (const larch_dtype fmt : {LARCH_Q5_K, LARCH_Q4_0, LARCH_Q5_0, LARCH_Q3_K, LARCH_Q4_K}) {
    test c = quant_case(ref, fmt, 33, 17, block_of(fmt) * 3, LARCH_F16, true, false, false);
    if (fmt == LARCH_Q4_K) c.args[0].skew = 1;
    c.run = [](larch_stream* s, const larch_tensor* t) {
      return larch_quant_gemm(s, &t[0], &t[1], &t[3], &t[2]);
    };
    std::vector<bytes> after;
    const int status = run_on(dut, c, after);
    bool good = status == LARCH_ERR_UNSUPPORTED;
    for (size_t i = 0; i < after.size(); ++i) good = good && after[i] == image(i, c.args[i]);
    if (!good) {
      std::fprintf(stderr, "%s: expected UNSUPPORTED with all buffers unchanged, got %d\n", c.name.c_str(), status);
      ++wrong;
    }
    ++count;
  }
  for (const larch_dtype fmt : I8_FORMATS) {
    for (const int64_t rows : {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 97, 520})
      for (const larch_dtype output : {LARCH_F16, LARCH_F32})
        for (const int64_t cols : {1, 17, 67}) {
          wrong += check(ref, dut, quant_i8_case(rows, cols, 768, output, rows % 2 == 1, fmt));
          ++count;
        }
    // The non-power-of-two tile boundary and its neighboring/ragged batches retain isolation.
    for (const int64_t rows : {111, 112, 113, 224, 225})
      for (const larch_dtype output : {LARCH_F16, LARCH_F32}) {
        wrong += check(ref, dut, quant_i8_case(rows, 17, 768, output, true, fmt));
        ++count;
      }
    // On sm_120 Q6_K past 64 rows takes J112 only where J64 needs more waves: units / 2 + 1 output tiles.
    if (fmt == LARCH_Q6_K) {
      wrong += check(ref, dut, quant_i8_case(97, (props.compute_units / 2 + 1) * 128 - 3, 768, LARCH_F16, true, fmt));
      ++count;
    }
    for (const int64_t k : {4096, 14336}) {
      wrong += check(ref, dut, quant_i8_case(17, 67, k, LARCH_F32, true, fmt));
      ++count;
    }
    // 32-weight formats end on a K128 half step: m8's 896 and a lone half.
    if (block_of(fmt) == 32)
      for (const int64_t k : {896, 128})
        for (const int64_t rows : {1, 9, 16, 33, 520}) {
          wrong += check(ref, dut, quant_i8_case(rows, 67, k, LARCH_F16, rows % 2 == 1, fmt));
          ++count;
        }
    // Zero activation groups, nearest-even ties and the affine min-only correction need exact coverage.
    for (int pattern = 0; pattern < (fmt == LARCH_Q4_K ? 3 : 2); ++pattern) {
      test c = quant_i8_case(17, 67, 768, LARCH_F32, true, fmt);
      c.name += " activation pattern " + std::to_string(pattern);
      bytes input = *c.args[1].data;
      for (size_t i = 0; i < input.size() / 2; ++i) {
        static const uint16_t h[] = {0x3800, 0x3e00, 0xb800, 0xbe00, 0x57f0, 0xd7f0};
        const uint16_t value = pattern == 0 ? 0 : h[i % 6];
        std::memcpy(input.data() + 2 * i, &value, 2);
      }
      c.args[1].data = std::make_shared<const bytes>(std::move(input));
      if (pattern == 2) {
        bytes weights = *c.args[0].data;
        for (size_t i = 0; i < weights.size(); i += 144)
          std::fill(weights.begin() + i + 16, weights.begin() + i + 144, uint8_t(0));
        c.args[0].data = std::make_shared<const bytes>(std::move(weights));
      }
      wrong += check(ref, dut, c);
      ++count;
    }
    // Nonfinite metadata propagates only within its sequence, including the last live padded row.
    for (const uint16_t value : {uint16_t(0x7e00), uint16_t(0x7c00), uint16_t(0xfc00)})
      for (const int location : {0, 1})
        for (const larch_dtype output : {LARCH_F16, LARCH_F32}) {
          test c = quant_i8_case(113, 17, 768, output, true, fmt);
          c.name += " nonfinite " + std::to_string(value) + " location " + std::to_string(location);
          bytes input = *c.args[1].data;
          const int64_t row = location == 0 ? 0 : 112, column = location == 0 ? 31 : 767;
          std::memcpy(input.data() + 2 * (row * (768 + c.args[1].pad) + column), &value, 2);
          c.args[1].data = std::make_shared<const bytes>(std::move(input));
          wrong += check(ref, dut, c);
          ++count;
        }
  }
  for (int invalid_case = 0; invalid_case < 4; ++invalid_case) {
    test c = quant_i8_case(17, 17, 768, LARCH_F16, true);
    c.name += " invalid " + std::to_string(invalid_case);
    c.want = LARCH_ERR_INVALID_ARG;
    for (arg& a : c.args) a.out = false;
    if (invalid_case == 0) c.args[1].t = LARCH_BF16;
    if (invalid_case == 1) c.args[2].shape[0] = 16;
    if (invalid_case == 2) c.args[2].shape[1] = 16;
    if (invalid_case == 3) c.args[3].shape[0] = 16;
    wrong += check(ref, dut, c);
    // The split entry too, also where the backend has no slot for it (the CPU reference).
    c.name += " split";
    c.run = [](larch_stream* s, const larch_tensor* t) {
      return larch_quant_gemm_i8_split(s, &t[0], &t[1], &t[3], &t[2]);
    };
    wrong += check(ref, dut, c);
    count += 2;
  }
  // A misaligned weight, and k off the pack's K128 chunks, refuse before any work.
  for (const auto& [fmt, k] : {std::pair<larch_dtype, int64_t>{LARCH_Q4_K, 768}, {LARCH_Q8_0, 96}}) {
    test c = quant_case(ref, fmt, 17, 17, k, LARCH_F16, true, false, false);
    if (fmt == LARCH_Q4_K) c.args[0].skew = 1;
    c.run = [](larch_stream* s, const larch_tensor* t) {
      return larch_quant_gemm_i8(s, &t[0], &t[1], &t[3], &t[2]);
    };
    std::vector<bytes> after;
    const int status = run_on(dut, c, after);
    bool good = status == LARCH_ERR_UNSUPPORTED;
    for (size_t i = 0; i < after.size(); ++i) good = good && after[i] == image(i, c.args[i]);
    if (!good) {
      std::fprintf(stderr, "%s i8: expected UNSUPPORTED with unchanged buffers, got %d\n", c.name.c_str(), status);
      ++wrong;
    }
    ++count;
  }
  {
    // The 32 MiB activation scratch is a device limit: the device refuses before touching a
    // buffer, while the CPU oracle computes the shape. Backends without the op refuse it too.
    test c = quant_i8_case(95361, 1, 256, LARCH_F16, false);
    c.name = "quant_gemm_i8 scratch budget rejection";
    std::vector<bytes> after;
    const int cpu = run_on(ref, c, after);
    const int status = run_on(dut, c, after);
    bool good = cpu == LARCH_OK && status == (props.backend == LARCH_BACKEND_CPU ? LARCH_OK : LARCH_ERR_UNSUPPORTED);
    for (size_t i = 0; status != LARCH_OK && i < after.size(); ++i) good = good && after[i] == image(i, c.args[i]);
    if (!good) {
      std::fprintf(stderr, "%s: CPU %d, device %d (or a buffer changed)\n", c.name.c_str(), cpu, status);
      ++wrong;
    }
    ++count;
  }
  {
    // A failing INT8 tile probe (opt-in route) at stream creation means no INT8 tiles, never a
    // failed stream: the stream works and the op refuses so callers fall back.
    test c = quant_i8_case(17, 17, 768, LARCH_F16, true);
    c.name = "quant_gemm_i8 after a failed tile probe";
    vv_fault_quant_i8_probe = 999;  // cudaErrorUnknown
    side probed{dut.dev, nullptr};
    const larch_status created = larch_stream_create(dut.dev, &probed.s);
    vv_fault_quant_i8_probe = 0;
    std::vector<bytes> after;
    const int status = created == LARCH_OK ? run_on(probed, c, after) : -1;
    bool good = status == (props.backend == LARCH_BACKEND_CPU ? LARCH_OK : LARCH_ERR_UNSUPPORTED);
    for (size_t i = 0; status == LARCH_ERR_UNSUPPORTED && i < after.size(); ++i)
      good = good && after[i] == image(i, c.args[i]);
    if (!good) {
      std::fprintf(stderr, "%s: stream_create %d, op %d: %s\n", c.name.c_str(), int(created), status,
                   larch_last_error(dut.dev));
      ++wrong;
    }
    if (probed.s) larch_stream_destroy(probed.s);
    ++count;
  }
  for (const larch_dtype fmt : I8_FORMATS) {
    // Capture packs scratch once; each replay must consume the current input, including ragged tiles.
    for (const int64_t rows : {9, 33, 512, 520}) {
      test c = quant_i8_case(rows, 17, 768, LARCH_F32, true, fmt);
      c.name += " graph replay changed input";
      const auto final_input = c.args[1].data;
      c.run = [ref, final_input](larch_stream* stream, const larch_tensor* t) {
        if (stream == ref.s) return larch_quant_gemm_i8(stream, &t[0], &t[1], &t[3], &t[2]);
        const bytes zeros(final_input->size(), 0);
        larch_status status = larch_upload(stream, t[1].buf, t[1].offset, zeros.data(), zeros.size());
        if (status == LARCH_OK) status = larch_stream_sync(stream);
        if (status != LARCH_OK) return status;
        status = larch_graph_begin(stream);
        if (status != LARCH_OK) return status;
        status = larch_quant_gemm_i8(stream, &t[0], &t[1], &t[3], &t[2]);
        larch_graph* graph = nullptr;
        const larch_status ended = larch_graph_end(stream, &graph);
        if (status == LARCH_OK) status = ended;
        if (status == LARCH_OK) status = larch_graph_launch(graph, stream);
        if (status == LARCH_OK) status = larch_stream_sync(stream);
        if (status == LARCH_OK)
          status = larch_upload(stream, t[1].buf, t[1].offset, final_input->data(), final_input->size());
        if (status == LARCH_OK) status = larch_graph_launch(graph, stream);
        if (status == LARCH_OK) status = larch_stream_sync(stream);
        if (graph) larch_graph_destroy(graph);
        return status;
      };
      wrong += check(ref, dut, c);
      ++count;
    }
  }
  const int64_t fast_n_tiles = std::max(int64_t(1), int64_t(props.compute_units) / 2);
  const int64_t fast_n = fast_n_tiles * 128 - 3;
  const int64_t fast_k = ((16 * std::max(int64_t(2), int64_t(props.compute_units)) + fast_n_tiles - 1)
                          / fast_n_tiles) * 256;
  for (const larch_dtype fmt : I8_FORMATS) {
    // Actual serving down projections and a ragged neighboring long-K tile exercise the amortized route.
    for (const int64_t rows : {8, 16})
      for (const larch_dtype output : {LARCH_F16, LARCH_F32}) {
        wrong += check_i8_split(ref, dut, quant_i8_split_case(ref, rows, 4096, 14336, output, fmt), split_skipped);
        ++count;
      }
    for (const larch_dtype output : {LARCH_F16, LARCH_F32}) {
      wrong += check_i8_split(ref, dut, quant_i8_split_case(ref, 9, fast_n, fast_k, output, fmt), split_skipped);
      ++count;
    }
    // 17-32 rows split as two J16 query tiles, with a ragged second one.
    for (const int64_t rows : {17, 32})
      for (const larch_dtype output : {LARCH_F16, LARCH_F32}) {
        wrong += check_i8_split(ref, dut, quant_i8_split_case(ref, rows, 4093, 4096, output, fmt), split_skipped);
        ++count;
      }
    if (block_of(fmt) == 32)
      for (const int64_t rows : {8, 16}) {
        wrong += check_i8_split(ref, dut, quant_i8_split_case(ref, rows, 1021, 896, LARCH_F32, fmt), split_skipped);
        ++count;
      }
    for (const uint16_t value : {uint16_t(0x7e00), uint16_t(0x7c00), uint16_t(0xfc00)}) {
      test c = quant_i8_split_case(ref, 9, fast_n, fast_k, LARCH_F32, fmt);
      c.name += " split nonfinite " + std::to_string(value);
      bytes input = *c.args[1].data;
      const int64_t column = value == 0x7e00 ? 31 : fast_k - 1;
      std::memcpy(input.data() + 2 * (8 * (fast_k + c.args[1].pad) + column), &value, 2);
      c.args[1].data = std::make_shared<const bytes>(std::move(input));
      wrong += check_i8_split(ref, dut, c, split_skipped);
      ++count;
    }
    for (const auto& dims : {std::array<int64_t, 3>{9, fast_n, fast_k}, {15, fast_n, fast_k}}) {
      test c = quant_i8_split_case(ref, dims[0], dims[1], dims[2], LARCH_F32, fmt);
      c.name += " split graph replay changed input";
      const auto input = c.args[1].data;
      c.run = [ref, input](larch_stream* stream, const larch_tensor* t) {
        if (stream == ref.s) return larch_quant_gemm_i8(stream, &t[0], &t[1], &t[3], &t[2]);
        const bytes zeros(input->size(), 0);
        larch_status status = larch_upload(stream, t[1].buf, t[1].offset, zeros.data(), zeros.size());
        if (status == LARCH_OK) status = larch_stream_sync(stream);
        if (status == LARCH_OK) status = larch_graph_begin(stream);
        if (status != LARCH_OK) return status;
        status = larch_quant_gemm_i8_split(stream, &t[0], &t[1], &t[3], &t[2]);
        larch_graph* graph = nullptr;
        const larch_status ended = larch_graph_end(stream, &graph);
        if (status == LARCH_OK) status = ended;
        if (status == LARCH_OK) status = larch_graph_launch(graph, stream);
        if (status == LARCH_OK) status = larch_stream_sync(stream);
        if (status == LARCH_OK) status = larch_upload(stream, t[1].buf, t[1].offset, input->data(), input->size());
        if (status == LARCH_OK) status = larch_graph_launch(graph, stream);
        if (status == LARCH_OK) status = larch_stream_sync(stream);
        if (graph) larch_graph_destroy(graph);
        return status;
      };
      wrong += check_i8_split(ref, dut, c, split_skipped);
      ++count;
    }
    // Short K work is routing's call, not a refusal: the narrow 9-row q/o shape must split and match the oracle.
    for (const larch_dtype output : {LARCH_F16, LARCH_F32}) {
      wrong += check_i8_split(ref, dut, quant_i8_split_case(ref, 9, 4096, 4096, output, fmt), split_skipped);
      ++count;
    }
  }
  // Refusal occurs before any device work; all inputs and outputs remain unchanged.
  for (const auto& shape : {std::array<int64_t, 3>{33, 17, 14336}, {64, 17, 14336},
                            {8, 256 * std::max(int64_t(1), int64_t(props.compute_units)), 4096},
                            {519, 17, 256}, {527, 17, 256}, {16, 1, 1524992}}) {
    test c = quant_i8_case(shape[0], shape[1], shape[2], LARCH_F16, true);
    c.name += " split early rejection";
    c.run = [](larch_stream* stream, const larch_tensor* t) {
      return larch_quant_gemm_i8_split(stream, &t[0], &t[1], &t[3], &t[2]);
    };
    std::vector<bytes> after;
    const int status = run_on(dut, c, after);
    bool good = status == LARCH_ERR_UNSUPPORTED;
    for (size_t i = 0; i < after.size(); ++i) good = good && after[i] == image(i, c.args[i]);
    if (!good) {
      std::fprintf(stderr, "%s: expected UNSUPPORTED before packing with unchanged buffers, got %d\n",
                   c.name.c_str(), status);
      ++wrong;
    }
    ++count;
  }
  // An unavailable tile must permit the normal fallback within the same graph capture.
  for (const int64_t rows : {33, 64}) {
    test c = quant_i8_case(rows, 17, 768, LARCH_F32, true);
    c.name += " split graph refusal then fallback";
    c.run = [ref](larch_stream* stream, const larch_tensor* t) {
      if (stream == ref.s) return larch_quant_gemm_i8(stream, &t[0], &t[1], &t[3], &t[2]);
      larch_status status = larch_graph_begin(stream);
      if (status != LARCH_OK) return status;
      const larch_status refused = larch_quant_gemm_i8_split(stream, &t[0], &t[1], &t[3], &t[2]);
      status = refused == LARCH_ERR_UNSUPPORTED
                   ? larch_quant_gemm_i8(stream, &t[0], &t[1], &t[3], &t[2]) : LARCH_ERR_INVALID_ARG;
      larch_graph* graph = nullptr;
      const larch_status ended = larch_graph_end(stream, &graph);
      if (status == LARCH_OK) status = ended;
      if (status == LARCH_OK) status = larch_graph_launch(graph, stream);
      if (status == LARCH_OK) status = larch_stream_sync(stream);
      if (graph) larch_graph_destroy(graph);
      return status;
    };
    wrong += check(ref, dut, c);
    ++count;
  }
  larch_stream_destroy(ref.s);
  larch_stream_destroy(dut.s);
  larch_device_close(ref.dev);
  larch_device_close(dut.dev);
  std::printf("quant_gemm: %d cases, %d failed; %d split fast cases SKIPPED (fallback checked)\n",
              count, wrong, split_skipped);
  return wrong ? 1 : 0;
}
