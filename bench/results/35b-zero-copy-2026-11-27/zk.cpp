// Lane Z profile (not part of the repo): larch_mmvq_id on m13's expert shapes (gate and up Q3_K [256*512, 2048],
// down Q4_K [256*2048, 512]) over a wrapped page-locked host layer against the same call over device memory, at 1
// and 4 rows of 8 routed experts each; per host layer (gate + up + down), 200 reps, each rep other experts so L2
// holds none of them. GB/s counted by (row, slot) pairs' bytes and by unique experts' bytes.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#include <cuda_runtime.h>

#include "larch.h"

#define OK(c) do { larch_status st_ = (c); if (st_ != LARCH_OK) { std::printf("%s:%d: %s -> %d\n", __FILE__, __LINE__, #c, int(st_)); std::exit(1); } } while (0)

static larch_tensor t2(larch_buffer* b, uint64_t off, larch_dtype t, int64_t r, int64_t c, int64_t blk) {
  return {b, off, t, 2, {r, c}, {c / blk, 1}};
}
static larch_tensor t3(larch_buffer* b, uint64_t off, larch_dtype t, int64_t a, int64_t r, int64_t c) {
  return {b, off, t, 3, {a, r, c}, {r * c, c, 1}};
}

int main() {
  const int64_t E = 256, FF = 512, D = 2048, K = 8;
  const uint64_t gb = uint64_t(E * FF) * D / 256 * 110, db = uint64_t(E * D) * FF / 256 * 144;  // Q3_K, Q4_K
  const uint64_t layer = 2 * gb + db, per_expert = layer / E;
  larch_device* dev; larch_stream* s;
  OK(larch_device_open(0, &dev)); OK(larch_stream_create(dev, &s));
  larch_buffer* hb; void* host;
  OK(larch_buffer_alloc_host(dev, layer, &hb, &host));
  std::mt19937 g(1);
  auto* h = static_cast<uint8_t*>(host);
  for (uint64_t i = 0; i < layer; i++) h[i] = uint8_t(g());
  // Keep scales sane: zero every block's f16 d fields' top bits (Q3_K d at 108, Q4_K d/dmin at 0..3).
  for (uint64_t o = 0; o < 2 * gb; o += 110) h[o + 109] &= 0x3b;
  for (uint64_t o = 2 * gb; o < layer; o += 144) { h[o + 1] &= 0x3b; h[o + 3] &= 0x3b; }
  larch_buffer* w;
  OK(larch_buffer_wrap_host(dev, host, layer, &w));
  larch_buffer* dcopy;
  OK(larch_buffer_alloc(dev, layer, &dcopy));
  OK(larch_upload(s, dcopy, 0, host, layer));
  larch_buffer* io;
  const int64_t R = 4;
  OK(larch_buffer_alloc(dev, 64 << 20, &io));
  std::vector<uint16_t> x(R * D, 0x3c00);
  OK(larch_upload(s, io, 0, x.data(), x.size() * 2));
  OK(larch_stream_sync(s));
  const uint64_t ido = 1 << 20, go = 2 << 20, uo = 4 << 20, ao = 6 << 20, oo = 8 << 20, wo = 10 << 20;
  std::vector<float> wts(R * K, 0.125f);
  OK(larch_upload(s, io, wo, wts.data(), wts.size() * 4));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int64_t rows : {1, 4}) {
    for (int mapped = 1; mapped >= 0; --mapped) {
      larch_buffer* wb = mapped ? w : dcopy;
      const int reps = 200;
      double pair_bytes = 0, unique_bytes = 0;
      std::vector<std::vector<int32_t>> idsets;
      for (int r = 0; r < reps + 1; r++) {
        std::vector<int32_t> ids(rows * K);
        std::set<int32_t> u;
        for (int64_t i = 0; i < rows; i++) {
          std::set<int32_t> row;
          while ((int64_t)row.size() < K) row.insert(int32_t(g() % E));
          int j = 0;
          for (int32_t v : row) { ids[i * K + j++] = v; u.insert(v); }
        }
        if (r > 0) { pair_bytes += double(rows * K) * per_expert; unique_bytes += double(u.size()) * per_expert; }
        idsets.push_back(ids);
      }
      double ms_total = 0;
      for (int r = 0; r < reps + 1; r++) {
        OK(larch_upload(s, io, ido, idsets[r].data(), idsets[r].size() * 4));
        larch_tensor xt = t2(io, 0, LARCH_F16, rows, D, 1);
        larch_tensor it = t2(io, ido, LARCH_I32, rows, K, 1);
        larch_tensor gt = t2(wb, 0, LARCH_Q3_K, E * FF, D, 256), ut = t2(wb, gb, LARCH_Q3_K, E * FF, D, 256);
        larch_tensor dt = t2(wb, 2 * gb, LARCH_Q4_K, E * D, FF, 256);
        larch_tensor go_t = t3(io, go, LARCH_F16, rows, K, FF), uo_t = t3(io, uo, LARCH_F16, rows, K, FF);
        larch_tensor ao_t = t3(io, ao, LARCH_F16, rows, K, FF);
        larch_tensor wt = t2(io, wo, LARCH_F32, rows, K, 1);
        larch_tensor ot = t2(io, oo, LARCH_F16, rows, D, 1);
        OK(larch_stream_sync(s));
        cudaEventRecord(e0, 0);
        OK(larch_mmvq_id(s, &gt, &xt, &it, nullptr, nullptr, &go_t));
        OK(larch_mmvq_id(s, &ut, &xt, &it, nullptr, nullptr, &uo_t));
        OK(larch_mmvq_id(s, &dt, &ao_t, &it, nullptr, &wt, &ot));
        OK(larch_stream_sync(s));
        cudaEventRecord(e1, 0); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        if (r > 0) ms_total += ms;
      }
      // Wall time per layer from host-side syncs is coarse; also time it with the host clock.
      const double per = ms_total / reps;
      std::printf("%s rows=%lld: %.3f ms per host layer (gate+up+down); %.1f GB/s by pairs (%.1f MB), %.1f GB/s by unique experts (%.1f MB)\n",
                  mapped ? "mapped" : "device", (long long)rows, per, pair_bytes / reps / (per / 1e3) / 1e9,
                  pair_bytes / reps / 1e6, unique_bytes / reps / (per / 1e3) / 1e9, unique_bytes / reps / 1e6);
    }
  }
  return 0;
}
