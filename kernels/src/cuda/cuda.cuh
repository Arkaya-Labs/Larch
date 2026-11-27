// CUDA backend internals (architecture 6.2), shared by backend.cu, quant.cu and ops.cu. Every op checks
// its arguments on the host, from metadata only, before it enqueues anything (larch.h), then launches
// on the stream's device; nothing here syncs.
#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>

#include "common/backend.hpp"
#include "common/warp.cuh"
#include "cpu/cpu.h"  // the GGML block table (cpu::info) and the argument-error type the reference uses

namespace cuda {

inline constexpr uint64_t capture_scratch_bytes = uint64_t(64) << 20;

/// A CUDA device handle: its ordinal and the props read at open.
struct device : larch_device {
  int ordinal;
  larch_device_props props;
};
/// Device memory of one device, or (wrapped) a page-locked host range it reads in place.
struct buffer : larch_buffer {
  int ordinal;
  uint8_t* ptr;
  uint64_t bytes;
  bool wrapped = false;  // buffer_wrap_host's: the host buffer owns the memory, so free never cudaFrees it
};
/// A cudaStream_t with its own cuBLAS handle and workspace, and the upload staging (backend.cu).
struct stream : larch_stream {
  int ordinal;
  uint64_t id;  // never reused in a process, unlike the pointer: a graph replays only on its stream
  int warp;     // the device's warp size, for launch shapes
  int compute_units;  // cached at stream creation, for occupancy-based dispatch
  int arch_code;  // cached compute capability, e.g. 86
  cudaStream_t s;
  cublasHandle_t blas;
  void* blas_workspace;
  cudaEvent_t staged_done[2] = {};  // recorded after the last copy out of each staging half
  uint8_t* staging = nullptr;  // two pinned halves, made on the first upload
  int half = 0;                // the half uploads fill
  uint64_t staged = 0;         // bytes of it in use
  int shared_mem_per_block = 0;  // device default, cached outside capture
  int image_arch = 0;  // __CUDA_ARCH__ of the code it runs (arch::cuda_image), e.g. 860: what sm_80 routes need
  uint32_t quant_i8_tiles[6] = {};  // per format from Q4_K: supported query tile sizes, configured once per stream
  uint8_t* capture_scratch = nullptr;  // shared by graphs on this uniquely identified stream
  mutable uint64_t capture_used = 0;
  bool capturing = false;
};

/// Route switch from the environment: 1 on, 0 off, -1 unset or anything else (the arch default). Cache it before any
/// stream or capture exists, so changing the environment later cannot alter a graph's route.
inline int tc_switch(const char* name) {
  const char* p = std::getenv(name);
  return p && (p[0] == '0' || p[0] == '1') && !p[1] ? p[0] - '0' : -1;
}

using cpu::bad_arg;  // thrown by argument checks: LARCH_ERR_INVALID_ARG

// Thrown by a failing CUDA or cuBLAS call: its status and detail.
struct failure {
  larch_status status;
  std::string detail;
};

/// larch's status for a CUDA error. Sticky errors are DEVICE_LOST: the process must restart (D2).
larch_status status_of(cudaError_t e);
/// Throws a failure for anything but cudaSuccess.
void ck(cudaError_t e, const char* what);
/// Throws a failure for anything but CUBLAS_STATUS_SUCCESS.
void ck(cublasStatus_t e, const char* what);

/// A tensor the shared checks passed (cpu::meta) with `p`, the device address of row 0; element i of a
/// vector sits at p[i * ld].
struct view : cpu::meta {
  uint8_t* p = nullptr;

  view() = default;
  view(const cpu::meta& m);  // m.buf is a CUDA buffer, or null

  template <class T>
  T* as() const {
    return reinterpret_cast<T*>(p);
  }
};

/// The shared checks' sizer on stream s: a CUDA buffer of s's device.
cpu::sizer sizer(const stream& s);

/// Throws UNSUPPORTED if a tensor's offset is not a multiple of its element size. larch.h allows such a
/// call, the kernels' typed loads would fault on it, and larch-kernels' Tensor never makes one. Called
/// after every INVALID_ARG check, so an invalid call still gets INVALID_ARG (as on Metal).
inline void aligned(std::initializer_list<view> vs) {
  for (const view& v : vs)
    if (!v.aligned) throw failure{LARCH_ERR_UNSUPPORTED, "a tensor offset is not a multiple of its element size"};
}

/// Runs an op's body on stream vs's device, turning bad_arg and failure into larch's status (dispatch
/// turns a std::bad_alloc into OOM).
template <class F>
larch_status run(const char* op, larch_stream* vs, F&& body) {
  const auto& s = *static_cast<stream*>(vs);
  try {
    ck(cudaSetDevice(s.ordinal), "cudaSetDevice");
    body(s);
    return LARCH_OK;
  } catch (const bad_arg& e) {
    return vv_fail(LARCH_ERR_INVALID_ARG, std::string(op) + ": " + e.what());
  } catch (const failure& f) {
    return vv_fail(f.status, std::string(op) + ": " + f.detail);
  }
}

/// Launches k on s. Kernels that index rows by blockIdx.y or .z keep those dims <= 65535 themselves.
template <class... P, class... A>
void launch(void (*k)(P...), dim3 grid, dim3 block, cudaStream_t s, A&&... args) {
  k<<<grid, block, 0, s>>>(std::forward<A>(args)...);
  ck(cudaGetLastError(), "kernel launch");
}

/// Blocks of `threads` covering n items, for grid-stride loops; capped so the grid stays launchable.
inline unsigned blocks_for(int64_t n, int threads) {
  const int64_t b = (n + threads - 1) / threads;
  return unsigned(b < 65535 ? b : 65535);
}

/// A grid of one block row per tensor row (x) by column chunks of `threads` (y, a grid-stride loop past
/// 65535 chunks), so element kernels index without a 64-bit divide.
inline dim3 rows_grid(int64_t rows, int64_t cols, int threads) {
  if (rows > 0x7fffffff) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 - 1 rows"};
  return dim3(unsigned(rows), blocks_for(cols, threads));
}

// rms_k's bytes (qk_norm of one head, n values in x, shared memory) from one warp: rms_k at `threads` threads and
// V-wide loads, its warps' butterflies taken in turn by this one, then vv_block_reduce's; x[i] becomes the f16 x[i]
// scale w[i], as a float.
__device__ inline void warp_rms(float* x, const float* w, int64_t w_ld, float eps, int64_t n, int V, int threads, int lane) {
  float mine = 0;  // rms_k's warp `lane`'s sum
  for (int vw = 0; vw < threads / 32; ++vw) {
    float ss = 0;
    for (int64_t i = int64_t(vw * 32 + lane) * V; i < n; i += int64_t(threads) * V)
      for (int k = 0; k < V; ++k) ss += x[i + k] * x[i + k];
    ss = vv_warp_reduce_sum(ss);
    if (lane == vw) mine = ss;
  }
  const float sum = vv_warp_reduce_sum(lane < threads / 32 ? mine : 0.0f);
  const float scale = 1.0f / sqrtf(sum / float(n) + eps);
  __syncwarp();
  for (int64_t i = lane; i < n; i += 32) x[i] = __half2float(__float2half_rn(x[i] * scale * w[i * w_ld]));
  __syncwarp();
}

/// Writes an op's f32 result as its output dtype, rounding to nearest even like the CPU reference.
__device__ __forceinline__ void store(float* p, float v) { *p = v; }
__device__ __forceinline__ void store(__half* p, float v) { *p = __float2half_rn(v); }

/// Eager temporaries are freed in stream order. Captured ops lease one bounded stream-owned arena:
/// all consumers precede the next lease on the same queue, including across graph replays. Nested
/// leases occupy distinct aligned regions until their enclosing op has issued every consumer.
struct scratch {
  void* p = nullptr;
  const stream& s;
  uint64_t before = 0;
  bool captured;
  scratch(uint64_t bytes, const stream& st) : s(st), captured(st.capturing) {
    if (!captured) {
      ck(cudaMallocAsync(&p, bytes, s.s), "scratch");
      return;
    }
    before = s.capture_used;
    constexpr uint64_t alignment = 256;
    const uint64_t at = (before + alignment - 1) & ~(alignment - 1);
    if (at > capture_scratch_bytes || bytes > capture_scratch_bytes - at)
      throw failure{LARCH_ERR_UNSUPPORTED, "captured scratch exceeds the 64 MiB stream arena"};
    p = s.capture_scratch + at;
    s.capture_used = at + bytes;
  }
  ~scratch() {
    if (captured) s.capture_used = before;
    else if (cudaFreeAsync(p, s.s) != cudaSuccess) cudaGetLastError();  // the next sync reports faults
  }
  scratch(const scratch&) = delete;
  scratch& operator=(const scratch&) = delete;
};

// The op slots of vv_cuda_ops(): quant.cu
larch_status embed(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status mmvq(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                   const larch_tensor*);
larch_status mmvq_verify(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                          const larch_tensor*);
larch_status quant_gemm(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                         const larch_tensor*);
void quant_i8_init(stream&);
larch_status quant_gemm_i8(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                            const larch_tensor*);
larch_status quant_gemm_i8_split(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                                  const larch_tensor*);
larch_status dequant(larch_stream*, const larch_tensor*, const larch_tensor*);
larch_status mmvq_id(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                      const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status mmvq_id_mixed(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                            const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status moe_route(larch_stream*, const larch_tensor*, float, const larch_tensor*, const larch_tensor*);
larch_status fused_moe_gate_up(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                                const larch_tensor*, const larch_tensor*);
larch_status fused_moe_down(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                             const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                             const larch_tensor*);
larch_status gemm(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                   const larch_tensor*, float, float);
// ops.cu
// rms_k's launch for groups of n elements of x into out: V-wide loads and its threads (rmsnorm, qk_norm).
struct rms_shape {
  int vec, threads;
};
template <class I, class O>
rms_shape rms_shape_of(const stream& s, const view& x, const view& out, int64_t n);
// Rows gemm_rows takes on gemm_rows_k (ops.cu), whose rows never mix; more go to cuBLAS.
constexpr int GEMM_ROWS = 8;
larch_status fused_moe_in(larch_stream*, const larch_tensor*, const larch_tensor*, float, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                           const larch_tensor*);
larch_status fused_gdn_in(larch_stream*, const larch_tensor*, const larch_tensor*, float, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status fused_proj_add(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                             const larch_tensor*);
larch_status fused_attn_in(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                            const larch_tensor*, const larch_tensor*, float, const larch_tensor*,
                            const larch_rope_params*, const larch_tensor*, const larch_kv_pool*, const larch_tensor*,
                            const larch_tensor*);
larch_status rmsnorm(larch_stream*, const larch_tensor*, const larch_tensor*, float, const larch_tensor*);
larch_status rope(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_rope_params*);
larch_status silu_mul(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status gemm_ex(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*, uint32_t,
                      uint32_t, float, float);
larch_status rmsnorm_f32(larch_stream*, const larch_tensor*, const larch_tensor*, float, const larch_tensor*);
larch_status rmsnorm_backward(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*, float,
                               const larch_tensor*, uint32_t);
larch_status rope_f32(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_rope_params*);
larch_status rope_multi(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_rope_multi_params*);
larch_status silu_mul_f32(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status silu_mul_backward(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                                const larch_tensor*, const larch_tensor*);
larch_status softmax(larch_stream*, const larch_tensor*, float, uint32_t, const larch_tensor*);
larch_status softmax_backward(larch_stream*, const larch_tensor*, const larch_tensor*, float, const larch_tensor*);
larch_status add(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status draft_conv(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*, uint32_t,
                         const larch_tensor*);
larch_status kv_write(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                       const larch_kv_pool*);
larch_status qk_norm(larch_stream*, const larch_tensor*, const larch_tensor*, float);
larch_status layernorm(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*, float,
                        const larch_tensor*);
larch_status gelu(larch_stream*, const larch_tensor*, const larch_tensor*);
larch_status attn_full(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*, uint32_t, float,
                        const larch_tensor*);
larch_status attn_decode(larch_stream*, const larch_tensor*, const larch_kv_pool*, const larch_tensor*,
                          const larch_tensor*, uint32_t, float, const larch_tensor*);
larch_status attn_prefill(larch_stream*, const larch_tensor*, const larch_kv_pool*, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, float, const larch_tensor*);
larch_status attn(larch_stream*, const larch_tensor*, const larch_kv_pool*, const larch_tensor*,
                   const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*, uint32_t,
                   uint32_t, float, const larch_tensor*);
larch_status gemm_rows(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                        const larch_tensor*, float, float);
larch_status sample_prep(larch_stream*, const larch_tensor*, const larch_sample_params*, const larch_tensor*,
                          const larch_tensor*, const larch_tensor*, const larch_tensor*);
// gdn.cu
larch_status sigmoid_mul(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status gdn_conv(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_gdn_state*,
                       const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status fused_gdn(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                        const larch_tensor*, const larch_tensor*, const larch_tensor*, float,
                        const larch_gdn_state*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                        const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status gdn(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                  const larch_tensor*, const larch_tensor*, float, const larch_gdn_state*, const larch_tensor*,
                  const larch_tensor*, const larch_tensor*, const larch_tensor*);

}  // namespace cuda
