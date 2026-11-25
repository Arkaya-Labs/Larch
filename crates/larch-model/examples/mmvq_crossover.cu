// A3S1: reuse A2C2a's shapes, cold weight rotation, bias and graph/event timing.
// Build with nvcc -O3 -std=c++20 -arch=sm_120 -ccbin g++-14 -Ikernels/include
//   crates/larch-model/examples/mmvq_crossover.cu build/a3-mmvq-routing/liblarch_kernels.a
//   -lcublas -o build/a3-mmvq-routing/mmvq_crossover
// Run each alias separately: timeout 1200 flock ~/src/gpu.lock <binary> m1|m5|m8.
#define main a2_bench_main
#include "../../../kernels/tests/mmvq_bench.cu"
#undef main

int main(int argc, char** argv) {
  if (argc != 2 || (std::strcmp(argv[1], "m1") && std::strcmp(argv[1], "m5") &&
                    std::strcmp(argv[1], "m8")))
    die("usage", "mmvq_crossover m1|m5|m8");
  CK(larch_device_open(0, &dev));
  CK(larch_device_props_get(dev, &props));
  CK(larch_stream_create(dev, &str));
  print_header("sliced crossover 17..128", false);
  std::puts("model,tensor,format,rows,sliced_us,gemm_us");
  for (const model& m : MODELS) {
    if (std::strcmp(argv[1], m.name)) continue;
    for (const shape& s : m.t) {
      if (is_head(s)) continue;  // The lm_head stays sliced at every row count.
      for (const std::string& fmt : split(s.gguf)) {
        bufs B;
        const auto W = weights(B, s, quant_named(fmt), false);
        rng g(4321);
        const bytes xh = halves(size_t(128 * s.k), g);
        larch_tensor x = tensor(B.put(xh.data(), xh.size()), LARCH_F16, 128, s.k);
        larch_tensor out = tensor(B.put(nullptr, 2 * 128 * uint64_t(s.n)), LARCH_F16, 128, s.n);
        const auto bt = bias_of(B, s, xh);
        const auto bias = carry(s, s.bias ? &bt : nullptr);
        const auto scratch = tensor(B.put(nullptr, 2 * uint64_t(s.n * s.k)), LARCH_F16, s.n, s.k);
        for (int r = 17; r <= 128; ++r) {
          x.shape[0] = out.shape[0] = r;
          const double pair_us = time_us([&](int i) {
            const auto& w = W[size_t(i) % W.size()];
            CK(larch_dequant(str, &w, &scratch));
            CK(larch_gemm(str, &x, &scratch, bias, &out, 1.0f, 0.0f));
          });
          const double slices_us = time_us([&](int i) {
            const auto& w = W[size_t(i) % W.size()];
            for (int at = 0; at < r; at += 16) {
              auto xs = x, ys = out;
              xs.shape[0] = ys.shape[0] = std::min(16, r - at);
              xs.offset += uint64_t(at) * 2 * s.k;
              ys.offset += uint64_t(at) * 2 * s.n;
              CK(larch_mmvq(str, &w, &xs, bias, &ys));
            }
          });
          std::printf("%s,%s,%s,%d,%.4f,%.4f\n", m.name, s.name, fmt.c_str(), r, slices_us, pair_us);
          std::fflush(stdout);
        }
      }
    }
  }
  larch_stream_destroy(str);
  larch_device_close(dev);
}
