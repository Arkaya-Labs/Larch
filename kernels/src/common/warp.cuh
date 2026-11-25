// Warp and block reductions shared by the CUDA sources and their hipified HIP build (architecture
// 6.3). Written against warpSize, so they hold for wave32 and wave64.
#pragma once

#include <cuda_runtime.h>

// Every lane of the warp: HIP's *_sync shuffles take a 64-bit mask.
#if defined(__HIP_PLATFORM_AMD__)
#define VV_FULL_MASK 0xffffffffffffffffull
#else
#define VV_FULL_MASK 0xffffffffu
#endif

/// The sum of v over the warp, in every lane.
__device__ __forceinline__ float vv_warp_reduce_sum(float v) {
  for (int o = warpSize / 2; o > 0; o /= 2) v += __shfl_xor_sync(VV_FULL_MASK, v, o);
  return v;
}

/// The max of v over the warp, in every lane.
__device__ __forceinline__ float vv_warp_reduce_max(float v) {
  for (int o = warpSize / 2; o > 0; o /= 2) v = fmaxf(v, __shfl_xor_sync(VV_FULL_MASK, v, o));
  return v;
}

// One partial per warp; blockDim.x <= 1024 is at most 32 warps of 32 (16 of 64), so the last warp
// reduces them. `part` is the caller's __shared__ float[32]; the leading sync lets a block reuse it.
template <class F>
__device__ __forceinline__ float vv_block_reduce(float v, float* part, float identity, F warp_reduce) {
  const int lane = int(threadIdx.x) % warpSize, warp = int(threadIdx.x) / warpSize;
  v = warp_reduce(v);
  __syncthreads();
  if (lane == 0) part[warp] = v;
  __syncthreads();
  return warp_reduce(lane < int(blockDim.x) / warpSize ? part[lane] : identity);
}

/// The sum of v over the block, in every thread; blockDim.x is a multiple of warpSize.
__device__ __forceinline__ float vv_block_reduce_sum(float v, float* part) {
  return vv_block_reduce(v, part, 0.0f, [](float x) { return vv_warp_reduce_sum(x); });
}

/// The max of v over the block, in every thread; blockDim.x is a multiple of warpSize.
__device__ __forceinline__ float vv_block_reduce_max(float v, float* part) {
  return vv_block_reduce(v, part, -INFINITY, [](float x) { return vv_warp_reduce_max(x); });
}
