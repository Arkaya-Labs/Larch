// Lane Z0 spike (not part of the repo): can the GPU read page-locked host memory in place fast enough to replace
// the CPU's host-expert decode? 1.6 GB of cudaHostAlloc memory (as larch_buffer_alloc_host takes it), a coalesced
// 16-byte-load read kernel over 12, 42 and 64 MB of it, 50 reps timed with cudaEvents, each rep a fresh window so no
// rep reads what L2 already holds. Beside it: the same kernel over device memory, cudaMemcpyAsync from the same
// pinned memory, and the CPU's multithreaded read at the CPU device's thread count (every logical CPU but two).
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void readk(const int4* __restrict__ p, size_t n, unsigned* out) {
  unsigned acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    int4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) *out = acc;  // keeps the loads
}

int main() {
  const size_t total = 1600ull << 20, mb = 1ull << 20;
  const size_t sizes[] = {12 * mb, 42 * mb, 64 * mb};
  const int reps = 50;
  int dev = 0, sms = 0;
  CK(cudaSetDevice(dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  void* h = nullptr;
  CK(cudaHostAlloc(&h, total, cudaHostAllocDefault));
  memset(h, 1, total);
  cudaPointerAttributes a{};
  CK(cudaPointerGetAttributes(&a, h));
  void* dp = nullptr;
  CK(cudaHostGetDevicePointer(&dp, h, 0));
  printf("pinned %zu MB: attr type %d, attr devicePointer %p, cudaHostGetDevicePointer %p (host %p)\n", total / mb,
         (int)a.type, a.devicePointer, dp, h);
  if (!a.devicePointer) { printf("no device pointer\n"); return 1; }
  void* d = nullptr;
  const size_t dtotal = 512 * mb;
  CK(cudaMalloc(&d, dtotal));
  CK(cudaMemset(d, 1, dtotal));
  unsigned* out;
  CK(cudaMalloc(&out, 4));
  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  const int blocks = sms * 8, threads = 512;
  auto kernel_gbs = [&](const char* base, size_t span, size_t bytes) -> double {
    size_t windows = span / bytes;
    readk<<<blocks, threads, 0, s>>>((const int4*)base, bytes / 16, out);  // warm-up
    cudaEventRecord(e0, s);
    for (int r = 0; r < reps; r++)
      readk<<<blocks, threads, 0, s>>>((const int4*)(base + ((r + 1) % windows) * bytes), bytes / 16, out);
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    return (double)bytes * reps / (ms / 1e3) / 1e9;
  };
  auto copy_gbs = [&](size_t bytes) -> double {
    size_t windows = total / bytes;
    cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s);
    cudaEventRecord(e0, s);
    for (int r = 0; r < reps; r++)
      cudaMemcpyAsync(d, (char*)h + ((r + 1) % windows) * bytes, bytes, cudaMemcpyHostToDevice, s);
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    return (double)bytes * reps / (ms / 1e3) / 1e9;
  };
  unsigned ncpu = std::thread::hardware_concurrency();
  unsigned nt = ncpu > 2 ? ncpu - 2 : 1;
  auto cpu_gbs = [&](size_t bytes) -> double {
    size_t windows = total / bytes;
    std::atomic<uint64_t> sink{0};
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
      const uint64_t* base = (const uint64_t*)((char*)h + ((r + 1) % windows) * bytes);
      size_t n = bytes / 8, per = n / nt;
      std::vector<std::thread> ts;
      for (unsigned t = 0; t < nt; t++)
        ts.emplace_back([&, t] {
          uint64_t acc = 0;
          for (size_t i = t * per; i < (t + 1) * per; i++) acc ^= base[i];
          sink ^= acc;
        });
      for (auto& t : ts) t.join();
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return (double)bytes * reps / sec / 1e9;
  };
  printf("%d SMs, kernel %d x %d, %d reps, CPU threads %u of %u\n", sms, blocks, threads, reps, nt, ncpu);
  printf("%-8s %16s %16s %18s %16s\n", "MB", "mapped GB/s", "device GB/s", "memcpyAsync GB/s", "CPU GB/s");
  for (size_t b : sizes) {
    double m = kernel_gbs((const char*)a.devicePointer, total, b);
    double dv = kernel_gbs((const char*)d, dtotal, b);
    double c = copy_gbs(b);
    double cp = cpu_gbs(b);
    printf("%-8zu %16.1f %16.1f %18.1f %16.1f\n", b / mb, m, dv, c, cp);
  }
  CK(cudaGetLastError());
  return 0;
}
