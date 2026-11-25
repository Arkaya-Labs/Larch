// Packed-loader/fragment adaptation from llama.cpp's mmq-load-tiles.cuh, mmq-vec-dot.cuh and mma.cuh,
// pinned c479922ac520a08969b4c1dc154d7bbb3c386d85, for every GGUF quant format larch reads.
// F32 activation/weight metadata and original activation sums are Larch's explicit INT8 policy.
/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once
#include <cuda_pipeline.h>

namespace cuda::mmq_q4 {
constexpr int weight_rows = 128, weight_stride = 84, activation_stride = 44;
// K128-major packs: 32 code words, eight F32 metadata words and four bank-padding words.

__device__ __forceinline__ uint32_t word(const uint8_t* p, bool aligned4) {
  if (aligned4) return *reinterpret_cast<const uint32_t*>(p);
  const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
  return uint32_t(h[0]) | uint32_t(h[1]) << 16;
}
// Scalar registers avoid the local array required by a lane-dependent three-word index.
__device__ __forceinline__ uint32_t unpack_scales(uint32_t s, uint32_t high, int half, bool minimum) {
  return half ? ((high >> (minimum ? 4 : 0)) & 0x0f0f0f0f) | ((s >> 2) & 0x30303030)
              : s & 0x3f3f3f3f;
}
__device__ __forceinline__ void load_a(uint32_t (&a)[4], const uint32_t* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int lane = int(threadIdx.x) % 32;
  // A 32-bit shared-window offset: without .shared PTX reads it as a generic address (faults on sm_86).
  const uint32_t address = uint32_t(__cvta_generic_to_shared(p + (lane % 16) * weight_stride + (lane / 16) * 4));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(address));
#elif defined(__CUDA_ARCH__)
  asm volatile("trap;");  // The host routes here only from sm_80; a PTX-only build for an older arch must not.
#endif
}
__device__ __forceinline__ void mma(const uint32_t (&a)[4], const uint32_t (&b)[2], int (&c)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#elif defined(__CUDA_ARCH__)
  asm volatile("trap;");
#endif
}
// K16 halves of the K32 fragments: Q6_K's two 16-weight sub-blocks of a group carry their own scales.
__device__ __forceinline__ void mma16(uint32_t a0, uint32_t a1, uint32_t b, int (&c)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(b));
#endif
}

__global__ void pack(const __half* x, int64_t ld, int64_t rows, int64_t padded_rows,
                     int64_t k, uint32_t* out) {
  const int lane = int(threadIdx.x) % 32;
  const int64_t group = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int64_t groups = k / 32;
  if (group >= padded_rows * groups) return;
  const int64_t row = group / groups, g = group % groups;
  const float value = row < rows ? __half2float(x[row * ld + g * 32 + lane]) : 0;
  const bool finite = __ballot_sync(0xffffffff, isfinite(value)) == 0xffffffff;
  // A uniform invalid group never reaches float-to-integer conversion; its codes stay zero.
  const float v = finite ? value : 0;
  float amax = fabsf(v), sum = v;
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }
  const int q = amax == 0 ? 0 : __float2int_rn(v * (127.0f / amax));
  uint32_t* block = out + (g / 4 * padded_rows + row) * activation_stride;
  reinterpret_cast<int8_t*>(block)[g % 4 * 32 + lane] = int8_t(max(-127, min(127, q)));
  if (!lane) {
    const float nan = __int_as_float(0x7fc00000);
    reinterpret_cast<float2*>(block + 32)[g % 4] = finite ? make_float2(amax / 127.0f, sum) : make_float2(nan, nan);
  }
  if (g % 4 == 0 && lane < 4) block[40 + lane] = 0;
}


// One K256 step of 128 weight rows in shared memory: row r's words [0, 64) are its 256 int8 codes in K order and
// words [64, 80) one float2 per 32-group: (d sc, -dmin m) for Q4_K/Q5_K, (d, 0) for Q8_0/Q4_0/Q5_0 and, for Q6_K,
// the scales of the group's two 16-weight sub-blocks. Codes times the first scale (Q6_K: per half) plus the activation sum times the second
// give the GGML weight. tile<T> holds a step's bytes in registers, loaded ahead
// of the step that unpacks them (store). Codes: row 8 i + warp, lane l; metadata: row tid / 2, groups 4 (tid % 2)
// to +3. Rows past n and blocks past kend are zero.
template <int T> struct tile;
__device__ __forceinline__ float half_at(uint32_t v, int i) { return __half2float(__ushort_as_half(uint16_t(v >> 16 * i))); }
// Bits 0-3 of v as the low bit of four bytes.
__device__ __forceinline__ uint32_t spread4(uint32_t v) {
  return (v & 1) | (v & 2) << 7 | (v & 4) << 14 | (v & 8) << 21;
}
// Bytewise x - c for bytes of x below 128, wrapping like int8: no borrow crosses a byte. Plain integer
// ops, since hipify has no __vsub4.
__device__ __forceinline__ uint32_t sub4(uint32_t x, uint32_t c) { return ((x | 0x80808080u) - c) ^ 0x80808080u; }
// Where tid's metadata goes: four float2 of row tid / 2.
__device__ __forceinline__ float2* meta_at(uint32_t* wx) {
  return reinterpret_cast<float2*>(wx + int(threadIdx.x) / 2 * weight_stride + 64) + 4 * (threadIdx.x % 2);
}

// Q4_K/Q5_K: d, dmin, 12 scale bytes (get_scale_min_k4), (Q5_K: 32 fifth-bit bytes, bit 2 g (+1 for the high
// nibble) of byte j for element 64 g + j), then 128 code bytes whose low nibbles in bytes 32 g.. are elements
// 64 g.. and high ones 64 g + 32... A warp loads row r's 8 fifth-bit words once (lane 8 (r % 4) + j holds word j
// of rows 4 k + r % 4) and shuffles each to the four lanes that use it.
template <int T> struct tile_k {
  static constexpr bool q5 = T == LARCH_Q5_K;
  static constexpr int bytes = q5 ? 176 : 144;
  uint32_t q[16], qh[q5 ? 4 : 1], dm, sc[3];
  __device__ void load(const uint8_t* w, int64_t w_ld, int64_t n, int64_t n0, int64_t k0, int64_t, bool a4) {
    const int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int64_t row = n0 + 8 * i + warp;
      q[i] = row < n ? word(w + row * w_ld + k0 / 256 * bytes + bytes - 128 + 4 * lane, a4) : 0;
    }
    if constexpr (q5)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int64_t row = n0 + 8 * (4 * k + lane / 8) + warp;
        qh[k] = row < n ? word(w + row * w_ld + k0 / 256 * bytes + 16 + 4 * (lane % 8), a4) : 0;
      }
    const int64_t row = n0 + tid / 2;
    const uint8_t* b = w + row * w_ld + k0 / 256 * bytes;
    dm = row < n ? word(b, a4) : 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) sc[i] = row < n ? word(b + 4 + 4 * i, a4) : 0;
  }
  __device__ void store(uint32_t* wx) const {
    const int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32, g = lane / 8;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      uint32_t* r = wx + (8 * i + warp) * weight_stride + 16 * g + lane % 8;
      r[0] = q[i] & 0x0f0f0f0f;
      r[8] = (q[i] >> 4) & 0x0f0f0f0f;
      if constexpr (q5) {
        const uint32_t h = __shfl_sync(0xffffffff, qh[i / 4], 8 * (i % 4) + lane % 8);
        r[0] |= ((h >> (2 * g)) & 0x01010101) << 4;
        r[8] |= ((h >> (2 * g + 1)) & 0x01010101) << 4;
      }
    }
    const int h = tid % 2;
    const uint32_t s = unpack_scales(sc[0], sc[2], h, false), m = unpack_scales(sc[1], sc[2], h, true);
    float2* out = meta_at(wx);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      out[j] = make_float2(half_at(dm, 0) * float((s >> (8 * j)) & 255), -half_at(dm, 1) * float((m >> (8 * j)) & 255));
  }
};
template <> struct tile<LARCH_Q4_K> : tile_k<LARCH_Q4_K> {};
template <> struct tile<LARCH_Q5_K> : tile_k<LARCH_Q5_K> {};

// Lane l: half l / 16 of the block, ql bytes 32 t + 4 j (t = l % 16 / 8, j = l % 8) whose low nibbles are
// elements 32 t + 4 j.. and high ones 64 + 32 t + 4 j.., with qh bits 2 t and 4 + 2 t. 210-byte blocks are only
// ever 2-byte aligned.
template <> struct tile<LARCH_Q6_K> {
  uint32_t ql[16], qh[16], sc[2], d;
  __device__ void load(const uint8_t* w, int64_t w_ld, int64_t n, int64_t n0, int64_t k0, int64_t, bool) {
    const int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int part = lane / 16, t = lane % 16 / 8, j = lane % 8;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int64_t row = n0 + 8 * i + warp;
      const uint8_t* b = w + row * w_ld + k0 / 256 * 210;
      ql[i] = row < n ? word(b + 64 * part + 32 * t + 4 * j, false) : 0;
      qh[i] = row < n ? word(b + 128 + 32 * part + 4 * j, false) : 0;
    }
    // 16 int8 scales at byte 192, one per 16 weights, then d.
    const int64_t row = n0 + tid / 2;
    const uint8_t* b = w + row * w_ld + k0 / 256 * 210;
    sc[0] = row < n ? word(b + 192 + 8 * (tid % 2), false) : 0;
    sc[1] = row < n ? word(b + 196 + 8 * (tid % 2), false) : 0;
    d = row < n ? *reinterpret_cast<const uint16_t*>(b + 208) : 0;
  }
  __device__ void store(uint32_t* wx) const {
    const int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32;
    const int part = lane / 16, t = lane % 16 / 8, j = lane % 8;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      uint32_t* r = wx + (8 * i + warp) * weight_stride + 32 * part + 8 * t + j;
      const uint32_t hb = qh[i] >> (2 * t);
      r[0] = sub4((ql[i] & 0x0f0f0f0f) | (hb & 0x03030303) << 4, 0x20202020);
      r[16] = sub4(((ql[i] >> 4) & 0x0f0f0f0f) | ((hb >> 4) & 0x03030303) << 4, 0x20202020);
    }
    float2* out = meta_at(wx);
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const uint32_t v = sc[g / 2] >> (16 * (g % 2));
      out[g] = make_float2(half_at(d, 0) * float(int8_t(v)), half_at(d, 0) * float(int8_t(v >> 8)));
    }
  }
};

// The 32-weight formats, d first: Q8_0 then 32 int8 codes; Q4_0 then 16 bytes, low nibbles elements 0-15 and high
// ones 16-31, code q - 8; Q5_0 then 32 fifth bits (bit e for element e) before those 16 bytes, code q - 16. Lane l
// reads block l / 4 of the step (8 per K256), quarter l % 4. 34-, 18- and 22-byte blocks are only ever 2-byte
// aligned. A K128 tail (k % 256 == 128) leaves blocks 4-7 of the last step zero.
template <int T> struct tile32 {
  static constexpr int bytes = T == LARCH_Q8_0 ? 34 : T == LARCH_Q4_0 ? 18 : 22;
  uint32_t q[16][T == LARCH_Q4_0 ? 1 : 2], d[2];
  __device__ void load(const uint8_t* w, int64_t w_ld, int64_t n, int64_t n0, int64_t k0, int64_t kend, bool) {
    const int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32, blk = lane / 4, j = lane % 4;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int64_t row = n0 + 8 * i + warp;
      const bool valid = row < n && k0 + 32 * blk < kend;
      const uint8_t* b = w + row * w_ld + (k0 / 32 + blk) * bytes;
      if constexpr (T == LARCH_Q8_0) {
        q[i][0] = valid ? word(b + 2 + 8 * j, false) : 0;
        q[i][1] = valid ? word(b + 6 + 8 * j, false) : 0;
      } else if constexpr (T == LARCH_Q4_0) {
        q[i][0] = valid ? word(b + 2 + 4 * j, false) : 0;
      } else {
        q[i][0] = valid ? word(b + 6 + 4 * j, false) : 0;
        q[i][1] = valid ? word(b + 2, false) : 0;
      }
    }
    const int64_t row = n0 + tid / 2;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int64_t at = k0 + 32 * (4 * (tid % 2) + g);
      const uint32_t v = row < n && at < kend ? *reinterpret_cast<const uint16_t*>(w + row * w_ld + at / 32 * bytes) : 0;
      d[g / 2] = g % 2 ? d[g / 2] | v << 16 : v;
    }
  }
  __device__ void store(uint32_t* wx) const {
    const int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32, blk = lane / 4, j = lane % 4;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      uint32_t* r = wx + (8 * i + warp) * weight_stride;
      if constexpr (T == LARCH_Q8_0) {
        r[2 * lane] = q[i][0];
        r[2 * lane + 1] = q[i][1];
      } else if constexpr (T == LARCH_Q4_0) {
        r[8 * blk + j] = sub4(q[i][0] & 0x0f0f0f0f, 0x08080808);
        r[8 * blk + 4 + j] = sub4((q[i][0] >> 4) & 0x0f0f0f0f, 0x08080808);
      } else {
        r[8 * blk + j] = sub4((q[i][0] & 0x0f0f0f0f) | spread4(q[i][1] >> (4 * j)) << 4, 0x10101010);
        r[8 * blk + 4 + j] =
            sub4(((q[i][0] >> 4) & 0x0f0f0f0f) | spread4(q[i][1] >> (16 + 4 * j)) << 4, 0x10101010);
      }
    }
    float2* out = meta_at(wx);
#pragma unroll
    for (int g = 0; g < 4; ++g) out[g] = make_float2(half_at(d[g / 2], g % 2), 0.0f);
  }
};
template <> struct tile<LARCH_Q8_0> : tile32<LARCH_Q8_0> {};
template <> struct tile<LARCH_Q4_0> : tile32<LARCH_Q4_0> {};
template <> struct tile<LARCH_Q5_0> : tile32<LARCH_Q5_0> {};

// One K128 half of J packed activation rows into shared y, as one asynchronous copy group (cp.async on sm_80+).
template <int J>
__device__ __forceinline__ void fetch(uint32_t* y, const uint32_t* x, int64_t padded_rows, int64_t m0, int64_t chunk) {
  const uint4* src = reinterpret_cast<const uint4*>(x + (chunk * padded_rows + m0) * activation_stride);
  for (int i = int(threadIdx.x); i < J * activation_stride / 4; i += 256)
    __pipeline_memcpy_async(reinterpret_cast<uint4*>(y) + i, src + i, 16);
  __pipeline_commit();
}

// acc[l] of rows (l / 2) and query columns (l % 2) gains one 32-group: a, the weight fragment; b, the activation
// fragment; dm, the rows' metadata; ds, the columns' (scale, sum). Q4_K keeps its original f32 expression order.
template <int T>
__device__ __forceinline__ void dot(float (&acc)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                                    const float2 (&dm)[2], const float2 (&ds)[2]) {
  if constexpr (T == LARCH_Q6_K) {
    int lo[4] = {}, hi[4] = {};
    mma16(a[0], a[1], b[0], lo);
    mma16(a[2], a[3], b[1], hi);
#pragma unroll
    for (int l = 0; l < 4; ++l) acc[l] += ds[l % 2].x * (dm[l / 2].x * float(lo[l]) + dm[l / 2].y * float(hi[l]));
  } else {
    int c[4] = {};
    mma(a, b, c);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      acc[l] += dm[l / 2].x * ds[l % 2].x * float(c[l]);
      if constexpr (T == LARCH_Q4_K || T == LARCH_Q5_K) acc[l] += dm[l / 2].y * ds[l % 2].y;
    }
  }
}

template<int J> constexpr int shared_bytes = (weight_rows * weight_stride + 2 * J * activation_stride) * 4;
// One warp loads each packed weight row once; eight warps each own sixteen output rows.
// Integer accumulators reset at each scale group: |dot| <= 32*128*127 = 520192, exact in f32.
// Segment boundaries are K256 step boundaries; no scale group is split.
template<int T, int J, bool Ahead>
__device__ __forceinline__ void accumulate_segment(
    const uint8_t* w, int64_t w_ld, int64_t n, const uint32_t* x,
    int64_t padded_rows, bool aligned4, int64_t n0, int64_t m0, int64_t kbegin,
    int64_t kend, uint32_t* memory, float (&acc)[J / 8][4]) {
  uint32_t* wx = memory;
  uint32_t* y = wx + weight_rows * weight_stride;
  const int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32;
  // Two copy groups stay in flight: a half's activations land while the other half computes, and a step's
  // weights while the previous step computes. Group order: h0(k), h1(k), h0(k + 256), h1(k + 256), ...
  // Ahead: the codes of step k + 256 are in flight during step k (registers the paired schedule and two-CTA
  // Stream-K lack); otherwise they load in step and only the activations overlap.
  constexpr bool ahead = Ahead;
  tile<T> next;
  __syncthreads();  // a previous segment's fragments are read
  if constexpr (ahead) next.load(w, w_ld, n, n0, kbegin, kend, aligned4);
  fetch<J>(y, x, padded_rows, m0, kbegin / 128);
  for (int64_t k0 = kbegin; k0 < kend; k0 += 256) {
    __syncthreads();  // the previous step's fragments are read
    if (k0 + 128 < kend) fetch<J>(y + J * activation_stride, x, padded_rows, m0, k0 / 128 + 1);
    else __pipeline_commit();
    if constexpr (!ahead) next.load(w, w_ld, n, n0, k0, kend, aligned4);
    next.store(wx);
    __pipeline_wait_prior(1);
    __syncthreads();
    if (ahead && k0 + 256 < kend) next.load(w, w_ld, n, n0, k0 + 256, kend, aligned4);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      if (half) {
        if (k0 + 128 >= kend) break;
        __syncthreads();  // half 0's fragments are read
        if (k0 + 256 < kend) fetch<J>(y, x, padded_rows, m0, k0 / 128 + 2);
        else __pipeline_commit();
        __pipeline_wait_prior(1);
        __syncthreads();
      }
      const uint32_t* yh = y + half * J * activation_stride;
      if constexpr (J >= 48 && J <= 112 && J % 16 == 0) {
        // All operand fragments plus 64 accumulators spill at J128; retain its original schedule.
        // Pinned GGML keeps all scale-group operands live before its query-major loop.
        // Each activation fragment serves two weight fragments; paired warps split queries.
        uint32_t a[2][4][4];
        float2 dm[2][4][2];
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
          for (int g = 0; g < 4; ++g)
            load_a(a[tile][g], wx + (warp / 2 * 32 + tile * 16) * weight_stride + half * 32 + g * 8);
#pragma unroll
          for (int l = 0; l < 2; ++l) {
            const int wr = warp / 2 * 32 + tile * 16 + lane / 4 + l * 8;
#pragma unroll
            for (int g = 0; g < 4; ++g)
              dm[tile][g][l] = reinterpret_cast<const float2*>(wx + wr * weight_stride + 64)[half * 4 + g];
          }
        }
#pragma unroll
        for (int j = 0; j < J; j += 16) {
          const int q = j + warp % 2 * 8;
#pragma unroll
          for (int g = 0; g < 4; ++g) {
            const uint32_t b[2] = {yh[(q + lane / 4) * activation_stride + g * 8 + lane % 4],
                                   yh[(q + lane / 4) * activation_stride + g * 8 + lane % 4 + 4]};
            const float2 ds[2] = {
              reinterpret_cast<const float2*>(yh + (q + lane % 4 * 2) * activation_stride + 32)[g],
              reinterpret_cast<const float2*>(yh + (q + lane % 4 * 2 + 1) * activation_stride + 32)[g]};
#pragma unroll
            for (int tile = 0; tile < 2; ++tile) dot<T>(acc[j / 16 * 2 + tile], a[tile][g], b, dm[tile][g], ds);
          }
        }
      } else {
        // Group-outer keeps one weight fragment live; each accumulator still sums groups in order. Two-CTA
        // Stream-K (in-step loads) keeps the loop rolled to stay within its 128 registers.
#pragma unroll (Ahead ? 4 : 1)
        for (int g = 0; g < 4; ++g) {
          uint32_t a[4];
          load_a(a, wx + warp * 16 * weight_stride + half * 32 + g * 8);
          const float2 dm[2] = {
            reinterpret_cast<const float2*>(wx + (warp * 16 + lane / 4) * weight_stride + 64)[half * 4 + g],
            reinterpret_cast<const float2*>(wx + (warp * 16 + lane / 4 + 8) * weight_stride + 64)[half * 4 + g]};
#pragma unroll
          for (int j = 0; j < J; j += 8) {
            const uint32_t b[2] = {yh[(j + lane / 4) * activation_stride + g * 8 + lane % 4],
                                   yh[(j + lane / 4) * activation_stride + g * 8 + lane % 4 + 4]};
            const float2 ds[2] = {
              reinterpret_cast<const float2*>(yh + (j + lane % 4 * 2) * activation_stride + 32)[g],
              reinterpret_cast<const float2*>(yh + (j + lane % 4 * 2 + 1) * activation_stride + 32)[g]};
            dot<T>(acc[j / 8], a, b, dm, ds);
          }
        }
      }
    }
  }
}

// sm_86/sm_89 keep two CTAs of the J8/J16 tiles resident (their J32+ shared tiles allow one), except Q6_K and Q5_0,
// which would spill within 128 registers. sm_120 takes no bound: any bound changes its measured code (J128 spills).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
#define VV_I8_BOUNDS(T, J) __launch_bounds__(256, (J) <= 16 && (T) != LARCH_Q6_K && (T) != LARCH_Q5_0 ? 2 : 1)
#else
#define VV_I8_BOUNDS(T, J)
#endif
template<int T, int J>
__global__ void VV_I8_BOUNDS(T, J)
    gemm(const uint8_t* w, int64_t w_ld, int64_t n, int64_t k, const uint32_t* x,
                     int64_t m, int64_t padded_rows, bool aligned4, const float* bias, int64_t bias_ld, void* out, int64_t out_ld,
                     bool f32, int64_t n_tiles) {
  static_assert(J == 8 || J == 16 || J == 32 || J == 64 || J == 112 || J == 128);
  extern __shared__ __align__(16) uint32_t memory[];
  const int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32;
  const int64_t n0 = int64_t(blockIdx.x) % n_tiles * weight_rows;
  const int64_t m0 = int64_t(blockIdx.x) / n_tiles * J;
  float acc[J / 8][4] = {};
  // Q5_K's fifth bits leave its J128 tile no registers for codes in flight on sm_86/89 either.
  constexpr bool ahead = !(J >= 48 && J <= 112 && J % 16 == 0) && !(T == LARCH_Q5_K && J == 128);
  accumulate_segment<T, J, ahead>(w, w_ld, n, x, padded_rows, aligned4, n0, m0, 0, k, memory, acc);
  if constexpr (J >= 48 && J <= 112 && J % 16 == 0) {
#pragma unroll
    for (int j = 0; j < J; j += 16) {
#pragma unroll
      for (int tile = 0; tile < 2; ++tile) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
          const int64_t row = n0 + warp / 2 * 32 + tile * 16 + lane / 4 + l / 2 * 8;
          const int64_t col = m0 + j + warp % 2 * 8 + lane % 4 * 2 + l % 2;
          if (row < n && col < m) {
            const float v = acc[j / 16 * 2 + tile][l] + (bias ? bias[row * bias_ld] : 0);
            if (f32) static_cast<float*>(out)[col * out_ld + row] = v;
            else static_cast<__half*>(out)[col * out_ld + row] = __float2half_rn(v);
          }
        }
      }
    }
  } else {
#pragma unroll
  for (int j = 0; j < J; j += 8) {
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const int64_t row = n0 + warp * 16 + lane / 4 + l / 2 * 8;
      const int64_t col = m0 + j + lane % 4 * 2 + l % 2;
      if (row < n && col < m) {
        const float v = acc[j / 8][l] + (bias ? bias[row * bias_ld] : 0);
        if (f32) static_cast<float*>(out)[col * out_ld + row] = v;
        else static_cast<__half*>(out)[col * out_ld + row] = __float2half_rn(v);
      }
    }
  }
  }
}

}  // namespace cuda::mmq_q4
