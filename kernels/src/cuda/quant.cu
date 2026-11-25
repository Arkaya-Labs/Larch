// GGML blocks on the GPU and the ops that read them: embed, dequant, mmvq (and mmvq_verify) and gemm (cuBLAS).
// embed and dequant decode each element exactly as the CPU reference's dequantize_row_* does it (same f32
// expression order, no FMA contraction), so they match it bit for bit; mmvq's unit decoders may contract.
#include <algorithm>
#include <climits>
#include <cstring>
#include <optional>
#include <type_traits>

#if !defined(__HIP_PLATFORM_AMD__)
#include <mma.h>
#endif

#include "common/arch.h"
#include "common/blas.h"
#include "common/warp.cuh"
#include "cuda/cuda.cuh"
#if !defined(__HIP_PLATFORM_AMD__)
#include "cuda/mmq_q4.cuh"
#include "cuda/mmq_streamk.cuh"
#endif

namespace cuda {

__device__ __forceinline__ float ld_half(const uint8_t* p) { return __half2float(*reinterpret_cast<const __half*>(p)); }

// The 6-bit scale and min of sub-block j in a Q4_K/Q5_K super-block (ggml's get_scale_min_k4).
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Scale i of a Q3_K block (the CPU reference's scales_q3_k) from its scale bytes lo = s[i % 8] and hi = s[8 + i % 4].
__device__ __forceinline__ int q3_sc(int lo, int hi, int i) {
  return ((i < 8 ? lo & 0xF : lo >> 4) | (hi >> 2 * (i / 4) & 3) << 4) - 32;
}

// 8 bytes at p as two words: one 64-bit load when a8 says p is 8-byte aligned, else four 2-byte loads (every
// block is 2-byte aligned).
// ponytail: Q6_K, Q8_0, Q4_0 and Q5_0 always take the 2-byte loads and K-quant headers come in as scalar loads:
// 0.87-0.91 of the copy peak on dequants larger than L2 (sm_120). Stage a CUDA block's quant bytes in shared
// memory with 16-byte loads when such a dequant drops below 0.85 of the peak.
__device__ __forceinline__ uint2 ld8(const uint8_t* p, bool a8) {
  if (a8) return __ldg(reinterpret_cast<const uint2*>(p));
  const uint16_t* h = reinterpret_cast<const uint16_t*>(p);
  return {uint32_t(__ldg(h)) | uint32_t(__ldg(h + 1)) << 16, uint32_t(__ldg(h + 2)) | uint32_t(__ldg(h + 3)) << 16};
}

// Byte e of an 8-byte load: a register pick, as every caller's e is constant once its loop unrolls.
__device__ __forceinline__ int byte8(const uint2& v, int e) { return int((e < 4 ? v.x : v.y) >> 8 * (e % 4) & 0xFF); }

// Elements [c, c + n) of a row of dtype T at `row`, c a multiple of 8; n < 8 only at the end of a plain row,
// as quant rows are whole 32-element blocks. Each is the CPU reference's dequantize_row_* expression (bit for
// bit: no FMA contraction, so d q - m rounds twice). a8: the row is 8-byte aligned, which Q4_K's and Q5_K's
// blocks (144 and 176 bytes) keep; the other quants' blocks (Q3_K's 110) are only ever 2-byte aligned.
template <int T>
__device__ __forceinline__ void decode8(const uint8_t* row, int64_t c, int n, bool a8, float (&w)[8]) {
  if constexpr (T == LARCH_F32 || T == LARCH_F16) {  // embed's plain tables, a few rows: one load each
#pragma unroll
    for (int e = 0; e < 8; ++e)
      if (e < n)
        w[e] = T == LARCH_F32 ? reinterpret_cast<const float*>(row)[c + e]
                               : __half2float(reinterpret_cast<const __half*>(row)[c + e]);
  } else if constexpr (T == LARCH_Q8_0) {  // d, then 32 int8
    const uint8_t* b = row + c / 32 * 34;
    const float d = ld_half(b);
    const uint2 q = ld8(b + 2 + c % 32, false);
#pragma unroll
    for (int e = 0; e < 8; ++e) w[e] = float(int8_t(byte8(q, e))) * d;
  } else if constexpr (T == LARCH_Q4_0 || T == LARCH_Q5_0) {
    // d, (Q5_0: 32 high bits, byte j / 8 holding the 8 of this piece), then 16 bytes: low nibbles are 0..15, high
    // 16..31. Q4_0's h is 0, so its integer is the CPU's q - 8 and Q5_0's its (q | h << 4) - 16.
    constexpr bool q5 = T == LARCH_Q5_0;
    const uint8_t* b = row + c / 32 * (q5 ? 22 : 18);
    const int j = int(c % 32), h = q5 ? b[2 + j / 8] : 0;
    const float d = ld_half(b);
    const uint2 q = ld8(b + (q5 ? 6 : 2) + j % 16, false);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      w[e] = float(((j < 16 ? byte8(q, e) & 0xF : byte8(q, e) >> 4) | (h >> e & 1) << 4) - (q5 ? 16 : 8)) * d;
  } else if constexpr (T == LARCH_Q4_K || T == LARCH_Q5_K) {
    // d, dmin, 12 scale bytes, (Q5_K: 32 bytes whose bit 2 g + hi is sub-block 2 g + hi's fifth), then 4 groups g
    // of 32 bytes, low nibbles (hi 0) then high: a piece of 8 lies in one sub-block.
    constexpr bool q5 = T == LARCH_Q5_K;
    const uint8_t* b = row + c / 256 * (q5 ? 176 : 144);
    const int j = int(c % 256), g = j / 64, hi = j % 64 / 32, l = j % 32;
    int sc, m;
    scale_min_k4(2 * g + hi, b + 4, sc, m);
    const float d1 = ld_half(b) * float(sc), m1 = ld_half(b + 2) * float(m);
    const uint2 q = ld8(b + (q5 ? 48 : 16) + 32 * g + l, a8), qh = q5 ? ld8(b + 16 + l, a8) : uint2{};
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int v = (hi ? byte8(q, e) >> 4 : byte8(q, e) & 0xF) + (q5 ? 16 * (byte8(qh, e) >> (2 * g + hi) & 1) : 0);
      w[e] = __fsub_rn(d1 * float(v), m1);
    }
  } else if constexpr (T == LARCH_Q2_K) {
    // 16 scale bytes, qs 64, d, dmin. Element j: bits 2 u of qs[32 n + l] (n = j / 128, u = j % 128 / 32,
    // l = j % 32), times d (sc & 15), less dmin (sc >> 4), sc = scales[8 n + 2 u + l / 16]: a piece of 8 lies in one.
    const uint8_t* b = row + c / 256 * 84;
    const int j = int(c % 256), n = j / 128, u = j % 128 / 32, l = j % 32, sc = b[8 * n + 2 * u + l / 16];
    const float dl = ld_half(b + 80) * float(sc & 0xF), ml = ld_half(b + 82) * float(sc >> 4);  // the CPU's
    const uint2 q = ld8(b + 16 + 32 * n + l, false);
#pragma unroll
    for (int e = 0; e < 8; ++e) w[e] = __fsub_rn(dl * float(byte8(q, e) >> 2 * u & 3), ml);
  } else if constexpr (T == LARCH_Q3_K) {
    // hmask 32, qs 64, 12 scale bytes, d. Element j: bits 2 u of qs[32 n + l], less 4 unless bit 4 n + u of
    // hmask[l] is set (n = j / 128, u = j % 128 / 32, l = j % 32), times d sc[8 n + 2 u + l / 16].
    const uint8_t* b = row + c / 256 * 110;
    const int j = int(c % 256), n = j / 128, u = j % 128 / 32, l = j % 32, i = 8 * n + 2 * u + l / 16;
    const float dl = ld_half(b + 108) * float(q3_sc(b[96 + i % 8], b[104 + i % 4], i));  // the CPU's d sc
    const uint2 q = ld8(b + 32 + 32 * n + l, false), hm = ld8(b + l, false);
#pragma unroll
    for (int e = 0; e < 8; ++e)
      w[e] = dl * float((byte8(q, e) >> 2 * u & 3) - (byte8(hm, e) >> (4 * n + u) & 1 ? 0 : 4));
  } else {  // Q6_K: 128 low-nibble bytes, 64 high-bit bytes, 16 int8 scales, d; two halves of 128
    static_assert(T == LARCH_Q6_K);
    const uint8_t* b = row + c / 256 * 210;
    const int j = int(c % 256), part = j / 128, quarter = j % 128 / 32, l = j % 32;
    const uint2 ql = ld8(b + 64 * part + 32 * (quarter & 1) + l, false), qh = ld8(b + 128 + 32 * part + l, false);
    const float s = ld_half(b + 208) * float(int8_t(b[192 + 8 * part + l / 16 + 2 * quarter]));  // the CPU's d sc
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int q = quarter < 2 ? byte8(ql, e) & 0xF : byte8(ql, e) >> 4;
      w[e] = s * float((q | (byte8(qh, e) >> 2 * quarter & 3) << 4) - 32);
    }
  }
}

// out[0, 8) = w, as the CPU reference rounds them: one (f16) or two (f32) 16-byte stores, out 16-byte aligned.
__device__ __forceinline__ void store8(__half* out, const float (&w)[8]) {
  __half2 h[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(w[2 * i], w[2 * i + 1]);
  uint4 v;
  memcpy(&v, h, 16);
  *reinterpret_cast<uint4*>(out) = v;
}
__device__ __forceinline__ void store8(float* out, const float (&w)[8]) {
  reinterpret_cast<float4*>(out)[0] = make_float4(w[0], w[1], w[2], w[3]);
  reinterpret_cast<float4*>(out)[1] = make_float4(w[4], w[5], w[6], w[7]);
}

// Calls f with t as a compile-time dtype: the seven quant formats, f16 and f32, and Q2_K where Q2 (the ops with Q2_K
// kernels: embed, dequant and mmvq_id); another op answers a Q2_K call UNSUPPORTED.
template <bool Q2 = false, class F>
static void with_dtype(larch_dtype t, F&& f) {
  switch (t) {
#if !defined(__HIP_PLATFORM_AMD__)  // HIP refuses Q3_K and Q2_K (backend.cu's no_weights)
    case LARCH_Q3_K: return f(std::integral_constant<int, LARCH_Q3_K>());
    case LARCH_Q2_K:
      if constexpr (Q2) return f(std::integral_constant<int, LARCH_Q2_K>());
      throw failure{LARCH_ERR_UNSUPPORTED, "no Q2_K kernel for this op"};
#endif
    case LARCH_Q4_K: return f(std::integral_constant<int, LARCH_Q4_K>());
    case LARCH_Q5_K: return f(std::integral_constant<int, LARCH_Q5_K>());
    case LARCH_Q6_K: return f(std::integral_constant<int, LARCH_Q6_K>());
    case LARCH_Q8_0: return f(std::integral_constant<int, LARCH_Q8_0>());
    case LARCH_Q4_0: return f(std::integral_constant<int, LARCH_Q4_0>());
    case LARCH_Q5_0: return f(std::integral_constant<int, LARCH_Q5_0>());
    case LARCH_F16: return f(std::integral_constant<int, LARCH_F16>());
    case LARCH_F32: return f(std::integral_constant<int, LARCH_F32>());
    default: throw bad_arg("unsupported dtype");
  }
}

// out[r, c] = src[ids ? ids[r] : r, c], decoded: embed (f32 out, ids) and dequant (f16 out, no ids). A thread
// decodes 8 consecutive elements; src8: src.p is 8-byte aligned (Q4_K/Q5_K rows inherit it), out16: out's rows
// are 16-byte aligned.
template <int T, class O>
__global__ void rows_k(const uint8_t* src, int64_t src_ld, bool src8, const int32_t* ids, int64_t ids_ld, O* out,
                       int64_t out_ld, bool out16, int64_t cols) {
  const int64_t r = blockIdx.x;
  const uint8_t* row = src + (ids ? int64_t(ids[r * ids_ld]) : r) * src_ld;
  O* o = out + r * out_ld;
  for (int64_t c = 8 * (blockIdx.y * int64_t(blockDim.x) + threadIdx.x); c < cols;
       c += 8 * int64_t(gridDim.y) * blockDim.x) {
    constexpr bool plain = T == LARCH_F32 || T == LARCH_F16;  // quant rows are whole 32-element blocks
    const int n = plain && cols - c < 8 ? int(cols - c) : 8;
    float w[8];
    decode8<T>(row, c, n, src8, w);
    if (out16 && n == 8) {
      store8(o + c, w);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if (e < n) store(o + c + e, w[e]);
    }
  }
}

// ids: an i32 vector, or an empty view for dequant.
template <class O>
static void decode_rows(const stream& s, const view& src, const view& ids, const view& out) {
  aligned({src, ids, out});
  if (!out.rows) return;
  // Q4_K and Q5_K rows are whole 144- and 176-byte blocks, so every row keeps src.p's 8-byte alignment.
  const bool src8 = uintptr_t(src.p) % 8 == 0, out16 = (uintptr_t(out.p) | uint64_t(out.ld) * sizeof(O)) % 16 == 0;
  // A row under 256 pieces (896 wide: 112) gets a block its size in whole 64s, not 256 threads mostly idle.
  const int64_t pieces = (out.cols + 7) / 8;
  const int threads = pieces < 256 ? int((pieces + 63) / 64 * 64) : 256;
  with_dtype<true>(src.dtype, [&](auto t) {
    launch(rows_k<decltype(t)::value, O>, rows_grid(out.rows, pieces, threads), threads, s.s, src.p, src.row_bytes,
           src8, ids.as<int32_t>(), ids.ld, out.as<O>(), out.ld, out16, out.cols);
  });
}

larch_status embed(larch_stream* vs, const larch_tensor* table, const larch_tensor* ids_i32,
                    const larch_tensor* out_f32) {
  return run("embed", vs, [&](const stream& s) {
    const cpu::embed_checked c = cpu::check_embed(sizer(s), table, ids_i32, out_f32);
    decode_rows<float>(s, c.table, c.ids, c.out);
  });
}

larch_status dequant(larch_stream* vs, const larch_tensor* w_quant, const larch_tensor* out_f16) {
  return run("dequant", vs, [&](const stream& s) {
    const cpu::dequant_checked c = cpu::check_dequant(sizer(s), w_quant, out_f16);
    decode_rows<__half>(s, c.w, view(), c.out);
  });
}

constexpr int MMVQ_ROWS = 16;      // larch.h's row limit; more rows go through dequant + gemm
constexpr int MMVQ_THREADS = 128;  // mmvq_k's block: 4 warps of 32 (2 of 64)

// 16 bytes at p: one 128-bit load when a16 says p is 16-byte aligned, else eight 2-byte loads (every
// block and x row is 2-byte aligned).
__device__ __forceinline__ uint4 ld16(const uint8_t* p, bool a16) {
  if (a16) return __ldg(reinterpret_cast<const uint4*>(p));
  const uint2 lo = ld8(p, false), hi = ld8(p + 8, false);
  return {lo.x, lo.y, hi.x, hi.y};
}

// Word i of a 16-byte load: a register pick, as every caller's i is constant once its loop unrolls.
__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Two halves from their bits: HFMA2 operands, constants included, without a conversion.
__device__ __forceinline__ __half2 hw(uint32_t v) {
  __half2 h;
  memcpy(&h, &v, 4);
  return h;
}

// Two halves held in a word, as f32.
__device__ __forceinline__ float2 h2f(uint32_t v) { return __half22float2(hw(v)); }

// A lane's step along a weight row, P pieces of 8 weights: for the K quants 32 weights of one 256-weight
// block (8 lanes per block), two runs of 16 that share their quant bytes, at k(u) and k(u) + HI; for Q4_0 a
// whole 32-weight block (Q5_0 too), two such runs; for Q8_0 one run of 16. load() only issues the loads, so the next
// unit's bytes are in flight while this one is decoded; piece(p) decodes weights k(u) + 8 (p & 1) + HI (p >> 1)
// + [0, 8). Q6_K, Q8_0, Q4_0 and Q5_0 round as the CPU reference does; Q4_K's and Q5_K's d q - m is one FMA (one
// rounding where the CPU reference rounds twice), which mmvq's tolerance allows and the bitwise embed and
// dequant would not.
template <int T>
struct unit;

// float(byte e of v) without I2F (an ALU-pipe op on sm_89): PRMT builds the f32 2^23 + byte, whose FADD of -2^23
// is exact (Sterbenz), so float(byte) bit for bit; sbyte_f's sign flip makes it byte + 128 and takes 2^23 + 128.
#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float byte_f(uint32_t v, int e) { return float(v >> 8 * e & 0xFF); }
__device__ __forceinline__ float sbyte_f(uint32_t v, int e) { return float(int8_t(v >> 8 * e)); }
#else
__device__ __forceinline__ float byte_f(uint32_t v, int e) {
  return __uint_as_float(__byte_perm(v, 0x4B000000, 0x7540 | e)) - 8388608.0f;
}
__device__ __forceinline__ float sbyte_f(uint32_t v, int e) {
  return __uint_as_float(__byte_perm(v ^ 0x80808080, 0x4B000000, 0x7540 | e)) - 8388736.0f;
}

// Bytes 0-3 of v & mask as field 2^-24 in f32, byte order, exactly: mask selects within the low byte of each 16-bit
// half, so each half is an f16 subnormal field 2^-24 that widens bit for bit, two weights per LOP3 (v pairs bytes
// (0, 2), v >> 8 bytes (1, 3)). The caller's scale carries the 2^24.
__device__ __forceinline__ void nib4(uint32_t v, uint32_t mask, float (&f)[4]) {
  const float2 lo = h2f(v & mask), hi = h2f(v >> 8 & mask);
  f[0] = lo.x;
  f[1] = hi.x;
  f[2] = lo.y;
  f[3] = hi.y;
}
#endif

// d sc and dmin m of the sub-block run r of lane u decodes, from its Q4_K/Q5_K block's 16-byte header h:
// scale_min_k4 for 4 sub-blocks at once, bytes 0-3 serving sub-blocks 0-3 and the other form 4-7.
__device__ __forceinline__ float2 k4_dm(const uint4& h, int u, int r) {
  const bool lo = u % 8 < 4;
  const uint32_t sc = lo ? h.y & 0x3F3F3F3F : (h.w & 0x0F0F0F0F) | (h.y >> 2 & 0x30303030);
  const uint32_t mn = lo ? h.z & 0x3F3F3F3F : (h.w >> 4 & 0x0F0F0F0F) | (h.z >> 2 & 0x30303030);
  const int e = u % 4 / 2 * 2 + r;
  const float2 dm = h2f(h.x);
  return {dm.x * byte_f(sc, e), dm.y * byte_f(mn, e)};
}

// Lane i of a block: qs[16i, 16i + 16), whose low nibbles are sub-block 2 (i / 2) and high ones the next.
template <>
struct unit<LARCH_Q4_K> {
  static constexpr int HI = 32, P = 4;
  uint4 h, q;  // d, dmin and the 12 scale bytes (one L1 line for the block's 8 lanes); the lane's qs

  __device__ static int k(int u) { return u / 8 * 256 + u % 8 / 2 * 64 + u % 2 * 16; }
  __device__ void load(const uint8_t* row, int u, bool a16) {
    const uint8_t* b = row + int64_t(u / 8) * 144;
    h = ld16(b, a16);
    q = ld16(b + 16 + 16 * (u % 8), a16);
  }
  template <bool Exact = false>
  __device__ void piece(int p, int u, float (&w)[8]) const {
    const int r = p >> 1;
    const float2 dm = k4_dm(h, u, r);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t v = word(q, 2 * (p & 1) + i) >> 4 * r & 0x0F0F0F0F;
#pragma unroll
      for (int e = 0; e < 4; ++e)
        w[4 * i + e] = Exact ? __fsub_rn(dm.x * byte_f(v, e), dm.y) : fmaf(dm.x, byte_f(v, e), -dm.y);
    }
  }
};

// Q4_K's lanes plus qh[16 (i % 2), +16), whose bits 2 (i / 2) and the next are its two runs' fifth bits.
template <>
struct unit<LARCH_Q5_K> {
  static constexpr int HI = 32, P = 4;
  uint4 h, q, qh;

  __device__ static int k(int u) { return unit<LARCH_Q4_K>::k(u); }
  __device__ void load(const uint8_t* row, int u, bool a16) {
    const uint8_t* b = row + int64_t(u / 8) * 176;
    h = ld16(b, a16);
    qh = ld16(b + 16 + 16 * (u % 2), a16);
    q = ld16(b + 48 + 16 * (u % 8), a16);
  }
  __device__ void piece(int p, int u, float (&w)[8]) const {
    const int r = p >> 1, hb = u % 8 / 2 * 2 + r;
    const float2 dm = k4_dm(h, u, r);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int wi = 2 * (p & 1) + i;
      const uint32_t v = (word(q, wi) >> 4 * r & 0x0F0F0F0F) | (word(qh, wi) >> hb & 0x01010101) << 4;
#pragma unroll
      for (int e = 0; e < 4; ++e) w[4 * i + e] = fmaf(dm.x, byte_f(v, e), -dm.y);
    }
  }
};

// Run r's 4 q6 from a ql word and its qh word (shifted to the lane's bit pairs), one a byte: nibble plus 2 high bits.
__device__ __forceinline__ uint32_t q6_word(uint32_t ql, uint32_t qh, int r) {
  return (ql >> 4 * r & 0x0F0F0F0F) | (qh >> 4 * r & 0x03030303) << 4;
}

// Lane 4n + j of a block: ql[64n + 16j, +16) and the qh bytes beside it, scales 8n + j and 8n + j + 4.
template <>
struct unit<LARCH_Q6_K> {
  static constexpr int HI = 64, P = 4;
  uint4 ql, qh;
  int8_t sc[2];
  __half d;

  __device__ static int k(int u) { return u / 8 * 256 + u % 8 / 4 * 128 + u % 4 * 16; }
  __device__ void load(const uint8_t* row, int u, bool) {  // 210-byte blocks: only ever 2-byte aligned
    const uint8_t* b = row + int64_t(u / 8) * 210;
    const int n = u % 8 / 4, j = u % 4;
    ql = ld16(b + 64 * n + 16 * j, false);
    qh = ld16(b + 128 + 32 * n + 16 * (j % 2), false);
    sc[0] = int8_t(b[192 + 8 * n + j]);
    sc[1] = int8_t(b[196 + 8 * n + j]);
    d = *reinterpret_cast<const __half*>(b + 208);
  }
  __device__ void piece(int p, int u, float (&w)[8]) const {
    const int r = p >> 1, sh = u % 4 / 2 * 2;  // run 0 holds qh bits 0-1 (j < 2) or 2-3, run 1 4 higher
    const float s = __half2float(d) * float(sc[r]);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int wi = 2 * (p & 1) + i;
      const uint32_t v = q6_word(word(ql, wi), word(qh, wi) >> sh, r);
      // s (q - 32) rounded once, as the CPU's s * float(q - 32): 32 s is exact.
#pragma unroll
      for (int e = 0; e < 4; ++e) w[4 * i + e] = fmaf(s, byte_f(v, e), -32 * s);
    }
  }
};

// Run r's 4 q3 + 4 from a qs word shifted to the lane's bit pairs and an hmask word shifted to its bits, one a byte.
__device__ __forceinline__ uint32_t q3_word(uint32_t q, uint32_t hm, int r) {
  return (q >> 2 * r & 0x03030303) | (hm >> r & 0x01010101) << 2;
}

// Lane 4n + 2t + h of a block: qs[32n + 16h, +16) and hmask[16h, +16), whose bit pairs 2t and 2t + 1 (hmask bits
// 4n + 2t and the next) are its two runs, sub-blocks 2t and 2t + 1 of half n; scale bytes as q3_sc reads them.
template <>
struct unit<LARCH_Q3_K> {
  static constexpr int HI = 32, P = 4;
  uint4 q, hm;
  uint8_t lo[2], hi[2];
  __half d;

  // Scale i of run r: 8n + 4t + 2r + h.
  __device__ static int sc_i(int u, int r) { return u % 8 / 4 * 8 + u % 4 / 2 * 4 + 2 * r + u % 2; }
  __device__ static int k(int u) { return u / 8 * 256 + u % 8 / 4 * 128 + u % 4 / 2 * 64 + u % 2 * 16; }
  __device__ void load(const uint8_t* row, int u, bool) {  // 110-byte blocks: only ever 2-byte aligned
    const uint8_t* b = row + int64_t(u / 8) * 110;
    q = ld16(b + 32 + 32 * (u % 8 / 4) + 16 * (u % 2), false);
    hm = ld16(b + 16 * (u % 2), false);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      lo[r] = b[96 + sc_i(u, r) % 8];
      hi[r] = b[104 + sc_i(u, r) % 4];
    }
    d = *reinterpret_cast<const __half*>(b + 108);
  }
  // Run r's d sc, rounded as the CPU's.
  __device__ float scale(int u, int r) const { return __half2float(d) * float(q3_sc(lo[r], hi[r], sc_i(u, r))); }
  // The lane's qs and hmask words shifted down to run 0's bit pair and bit.
  __device__ static uint32_t q_at(uint32_t v, int u) { return v >> u % 4 / 2 * 4; }
  __device__ static uint32_t hm_at(uint32_t v, int u) { return v >> (u % 8 / 4 * 4 + u % 4 / 2 * 2); }
  __device__ void piece(int p, int u, float (&w)[8]) const {
    const int r = p >> 1;
    const float s = scale(u, r);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int wi = 2 * (p & 1) + i;
      const uint32_t v = q3_word(q_at(word(q, wi), u), hm_at(word(hm, wi), u), r);
      // s (q - 4) rounded once, as the CPU's dl * float(q - 4): 4 s is exact.
#pragma unroll
      for (int e = 0; e < 4; ++e) w[4 * i + e] = fmaf(s, byte_f(v, e), -4 * s);
    }
  }
};

// Lane 4n + 2t + h of a block: qs[32n + 16h, +16), whose bit pairs 2t and 2t + 1 are its two runs, sub-blocks 8n + 4t + h
// and 8n + 4t + 2 + h (Q3_K's lanes without the hmask). Only quant_gemm_k (the grouped mmvq_id) decodes with it, so
// piece rounds d sc q - dmin m twice, as the CPU reference and dequant do.
template <>
struct unit<LARCH_Q2_K> {
  static constexpr int HI = 32, P = 4;
  uint4 q;
  uint8_t sc[2];
  __half d, dmin;

  __device__ static int k(int u) { return unit<LARCH_Q3_K>::k(u); }
  __device__ void load(const uint8_t* row, int u, bool) {  // 84-byte blocks: only ever 4-byte aligned
    const uint8_t* b = row + int64_t(u / 8) * 84;
    q = ld16(b + 16 + 32 * (u % 8 / 4) + 16 * (u % 2), false);
#pragma unroll
    for (int r = 0; r < 2; ++r) sc[r] = b[u % 8 / 4 * 8 + u % 4 / 2 * 4 + 2 * r + u % 2];
    d = *reinterpret_cast<const __half*>(b + 80);
    dmin = *reinterpret_cast<const __half*>(b + 82);
  }
  __device__ void piece(int p, int u, float (&w)[8]) const {
    const int r = p >> 1;
    const float dl = __half2float(d) * float(sc[r] & 0xF), ml = __half2float(dmin) * float(sc[r] >> 4);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t v = word(q, 2 * (p & 1) + i) >> (u % 4 / 2 * 4 + 2 * r) & 0x03030303;
#pragma unroll
      for (int e = 0; e < 4; ++e) w[4 * i + e] = __fsub_rn(dl * byte_f(v, e), ml);
    }
  }
};

// Lane u: half u % 2 of block u / 2, qs[16 (u % 2), +16), one run (HI unused). Measured on sm_120 against
// a block per lane (two runs, as the K quants), both at CW 4 KS 2: 0-6 % faster at one row, 14-27 % at eight.
template <>
struct unit<LARCH_Q8_0> {
  static constexpr int HI = 0, P = 2;
  uint4 q;
  __half d;

  __device__ static int k(int u) { return u * 16; }
  __device__ void load(const uint8_t* row, int u, bool) {  // 34-byte blocks: only ever 2-byte aligned
    const uint8_t* b = row + int64_t(u / 2) * 34;
    d = *reinterpret_cast<const __half*>(b);
    q = ld16(b + 2 + 16 * (u % 2), false);
  }
  __device__ void piece(int p, int, float (&w)[8]) const {
    const float s = __half2float(d);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t v = word(q, 2 * p + i);
#pragma unroll
      for (int e = 0; e < 4; ++e) w[4 * i + e] = sbyte_f(v, e) * s;
    }
  }
};

// Lane u: block u, whose 16 qs bytes hold elements 0-15 in their low nibbles and 16-31 in the high ones.
template <>
struct unit<LARCH_Q4_0> {
  static constexpr int HI = 16, P = 4;
  uint4 q;
  __half d;

  __device__ static int k(int u) { return u * 32; }
  __device__ void load(const uint8_t* row, int u, bool) {  // 18-byte blocks: only ever 2-byte aligned
    const uint8_t* b = row + int64_t(u) * 18;
    d = *reinterpret_cast<const __half*>(b);
    q = ld16(b + 2, false);
  }
  __device__ void piece(int p, int, float (&w)[8]) const {
    const int r = p >> 1;
    const float s = __half2float(d);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t v = word(q, 2 * (p & 1) + i) >> 4 * r & 0x0F0F0F0F;
      // s (q - 8) rounded once, as the CPU's float(q - 8) * d: 8 s is exact.
#pragma unroll
      for (int e = 0; e < 4; ++e) w[4 * i + e] = fmaf(s, byte_f(v, e), -8 * s);
    }
  }
};

// Run r's 4 quants in word wi of a Q5_0 block, one a byte: nibbles plus fifth bits, bit e of the 4 spread to bit 4 of
// byte e by a multiply, as no two of its products overlap.
__device__ __forceinline__ uint32_t q5_0_word(const uint4& q, uint32_t qh, int r, int wi) {
  return (word(q, wi) >> 4 * r & 0x0F0F0F0F) | ((qh >> (16 * r + 4 * wi) & 0xF) * 0x02040810 & 0x10101010);
}

// Q4_0's lanes plus the block's 32 fifth bits, bit j for element j.
template <>
struct unit<LARCH_Q5_0> {
  static constexpr int HI = 16, P = 4;
  uint4 q;
  uint32_t qh;
  __half d;

  __device__ static int k(int u) { return u * 32; }
  __device__ void load(const uint8_t* row, int u, bool) {  // 22-byte blocks: only ever 2-byte aligned
    const uint8_t* b = row + int64_t(u) * 22;
    const uint16_t* h = reinterpret_cast<const uint16_t*>(b + 2);
    d = *reinterpret_cast<const __half*>(b);
    qh = uint32_t(__ldg(h)) | uint32_t(__ldg(h + 1)) << 16;
    q = ld16(b + 6, false);
  }
  __device__ void piece(int p, int, float (&w)[8]) const {
    const int r = p >> 1;
    const float s = __half2float(d);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t v = q5_0_word(q, qh, r, 2 * (p & 1) + i);
      // s (q - 16) rounded once, as the CPU's float(q - 16) * d: 16 s is exact.
#pragma unroll
      for (int e = 0; e < 4; ++e) w[4 * i + e] = fmaf(s, byte_f(v, e), -16 * s);
    }
  }
};

// One-row mmvq_k's and mmvq_rows_k's decoder: unit<T>::piece's f32 values, bit for bit, in fewer instructions a weight
// (HIP, multi-row mmvq_k and quant_gemm_k keep piece: quant_gemm_k's piece<true> Q4_K rounds twice, one fmaf once).
// set() computes a unit's scales once, where piece recomputes them per piece; quad(p, i) decodes weights 4 i + [0, 4)
// of piece p. A run of high nibbles stays high where its scale takes the 1 / 16 exactly: (a / 16) (16 q) is a q, the
// same product, so fmaf rounds it the same once; a / 16 is exact as every such a is an f16 d (times an integer), never
// an f32 subnormal. Q4_K's and Q6_K's quad read their fields as f16 subnormals q 2^-24 (Q4_K through nib4, Q6_K through
// h2f and one PRMT), as the ALU pipe is what one-row decode runs short of: one LOP3 or PRMT pairs two weights where
// byte_f spends a PRMT on each, and the widening HADD2.F32 replaces byte_f's FADD. Their a carries the 2^24 (exact,
// |a| < 2^23), so fmaf's exact product a q and its one rounding are unchanged.
template <int T>
struct fast;

// The H2 kernels' decoder (sm_86/89, arch::mmvq_h2): a run of 16 weights in half2 (h2_sum) against x' = x 2^-e, then
// its f32 scale. set() takes the f32 run scales once as fast<T> does; run(r, xw, meta) is run r's d sc 2^e S (+ -dmin m
// sum x for Q4_K): fmaf(a meta.x, S, b meta.y) for Q4_K and __fmul_rn(a meta.x, S) for Q6_K and Q3_K, which the
// caller's acc += cannot contract, so mmvq_k<T, 1, *, true> and mmvq_rows_k<T, R, true> round alike. a meta.x is
// exact (2^e).
// ponytail: Q5_K on the f32 decoder, add h2<Q5_K> when a measured model on sm_86/89 uses it.
template <int T>
struct h2;

#if !defined(__HIP_PLATFORM_AMD__)
template <>
struct fast<LARCH_Q4_K> {
  float a[2], b[2];  // run r's d sc 2^24 (run 1's over 16) and -dmin m
  uint4 q;
  __device__ void set(const unit<LARCH_Q4_K>& un, int u) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float2 dm = k4_dm(un.h, u, r);
      a[r] = dm.x * (r ? 0x1p20f : 0x1p24f);  // nib4's 2^24, over 16 for the high run
      b[r] = -dm.y;
    }
    q = un.q;
  }
  __device__ void quad(int p, int i, float* w) const {
    const int r = p >> 1;
    float f[4];
    nib4(word(q, 2 * (p & 1) + i), r ? 0x00F000F0 : 0x000F000F, f);
#pragma unroll
    for (int e = 0; e < 4; ++e) w[e] = fmaf(a[r], f[e], b[r]);
  }
};

template <>
struct fast<LARCH_Q5_K> {
  float a[2], b[2];
  uint4 q, qh;  // qh shifted down to this lane's bit pair
  __device__ void set(const unit<LARCH_Q5_K>& un, int u) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float2 dm = k4_dm(un.h, u, r);
      a[r] = dm.x;
      b[r] = -dm.y;
    }
    const int hb = u % 8 / 2 * 2;
    q = un.q;
    qh = {un.qh.x >> hb, un.qh.y >> hb, un.qh.z >> hb, un.qh.w >> hb};
  }
  __device__ void quad(int p, int i, float* w) const {
    const int r = p >> 1, wi = 2 * (p & 1) + i;
    const uint32_t v = (word(q, wi) >> 4 * r & 0x0F0F0F0F) | (word(qh, wi) >> r & 0x01010101) << 4;
#pragma unroll
    for (int e = 0; e < 4; ++e) w[e] = fmaf(a[r], byte_f(v, e), b[r]);
  }
};

template <>
struct fast<LARCH_Q6_K> {
  float a[2], b[2];  // run r's d sc 2^24 and -32 d sc
  uint4 ql, qh;      // qh shifted down to this lane's bit pairs
  __device__ void set(const unit<LARCH_Q6_K>& un, int u) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float s = __half2float(un.d) * float(un.sc[r]);
      a[r] = s * 0x1p24f;
      b[r] = -32 * s;
    }
    const int sh = u % 4 / 2 * 2;
    ql = un.ql;
    qh = {un.qh.x >> sh, un.qh.y >> sh, un.qh.z >> sh, un.qh.w >> sh};
  }
  __device__ void quad(int p, int i, float* w) const {
    const int r = p >> 1, wi = 2 * (p & 1) + i;
    const uint32_t v = q6_word(word(ql, wi), word(qh, wi), r);
    // v's bytes are whole q6, so one PRMT pairs bytes (1, 3) where nib4 shifts and masks
    const float2 lo = h2f(v & 0x00FF00FF), hi = h2f(__byte_perm(v, 0, 0x4341));
    const float f[4] = {lo.x, hi.x, lo.y, hi.y};
#pragma unroll
    for (int e = 0; e < 4; ++e) w[e] = fmaf(a[r], f[e], b[r]);
  }
};

// ponytail: not measured, byte_f per weight as unit::piece; take nib4's f16-subnormal pairing (two weights a LOP3)
// when Q3_K mmvq shows in the hybrid MoE target's decode profile on sm_120 below Q4_K's GB/s (mmvq_bench mmvq --rows 1).
template <>
struct fast<LARCH_Q3_K> {
  float a[2], b[2];  // run r's d sc and -4 d sc
  uint4 q, hm;       // shifted down to this lane's bit pairs and bits
  __device__ void set(const unit<LARCH_Q3_K>& un, int u) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      a[r] = un.scale(u, r);
      b[r] = -4 * a[r];
    }
    using U = unit<LARCH_Q3_K>;
    q = {U::q_at(un.q.x, u), U::q_at(un.q.y, u), U::q_at(un.q.z, u), U::q_at(un.q.w, u)};
    hm = {U::hm_at(un.hm.x, u), U::hm_at(un.hm.y, u), U::hm_at(un.hm.z, u), U::hm_at(un.hm.w, u)};
  }
  __device__ void quad(int p, int i, float* w) const {
    const int r = p >> 1, wi = 2 * (p & 1) + i;
    const uint32_t v = q3_word(word(q, wi), word(hm, wi), r);
#pragma unroll
    for (int e = 0; e < 4; ++e) w[e] = fmaf(a[r], byte_f(v, e), b[r]);
  }
};

// Q8_0's piece is already one PRMT, FADD and FMUL a weight: the same code.
template <>
struct fast<LARCH_Q8_0> {
  float s;
  uint4 q;
  __device__ void set(const unit<LARCH_Q8_0>& un, int) {
    s = __half2float(un.d);
    q = un.q;
  }
  __device__ void quad(int p, int i, float* w) const {
    const uint32_t v = word(q, 2 * p + i);
#pragma unroll
    for (int e = 0; e < 4; ++e) w[e] = sbyte_f(v, e) * s;
  }
};

template <>
struct fast<LARCH_Q4_0> {
  float a[2], b;  // s (run 1's over 16) and -8 s
  uint4 q;
  __device__ void set(const unit<LARCH_Q4_0>& un, int) {
    a[0] = __half2float(un.d);
    a[1] = a[0] * 0.0625f;
    b = -8 * a[0];
    q = un.q;
  }
  __device__ void quad(int p, int i, float* w) const {
    const int r = p >> 1;
    const uint32_t v = word(q, 2 * (p & 1) + i) & (r ? 0xF0F0F0F0 : 0x0F0F0F0F);
#pragma unroll
    for (int e = 0; e < 4; ++e) w[e] = fmaf(a[r], byte_f(v, e), b);
  }
};

template <>
struct fast<LARCH_Q5_0> {
  float a, b;  // s and -16 s
  uint4 q;
  uint32_t qh;
  __device__ void set(const unit<LARCH_Q5_0>& un, int) {
    a = __half2float(un.d);
    b = -16 * a;
    q = un.q;
    qh = un.qh;
  }
  __device__ void quad(int p, int i, float* w) const {
    const uint32_t v = q5_0_word(q, qh, p >> 1, 2 * (p & 1) + i);
#pragma unroll
    for (int e = 0; e < 4; ++e) w[e] = fmaf(a, byte_f(v, e), b);
  }
};

// S = the sum of a 16-weight run's q x' products: pairs (q of word wi's bytes 0 and 2, then 1 and 3, wi = 0..3, each
// against x' word 2 wi + (0, 1) of xw, mmvq_h2_prep_k's (x0, x2, x1, x3) order) in 8 HFMA2 into one half2 from 0, then
// widened and S = lo + hi in f32. qp(v, j) is pair j of word v as half2 q. |q x'| < 32 * 128 keeps the 8 a lane in
// the f16 range.
template <class Q>
__device__ __forceinline__ float h2_sum(const uint32_t (&v)[4], const uint4 (&xw)[2], Q qp) {
  __half2 s = hw(0);
#pragma unroll
  for (int wi = 0; wi < 4; ++wi)
#pragma unroll
    for (int j = 0; j < 2; ++j) s = __hfma2(qp(v[wi], j), hw(word(xw[wi / 2], 2 * wi % 4 + j)), s);
  const float2 f = __half22float2(s);
  return f.x + f.y;
}

template <>
struct h2<LARCH_Q4_K> {
  float a[2], b[2];  // run r's d sc and -dmin m
  uint4 q;
  __device__ void set(const unit<LARCH_Q4_K>& un, int u) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float2 dm = k4_dm(un.h, u, r);
      a[r] = dm.x;
      b[r] = -dm.y;
    }
    q = un.q;
  }
  __device__ float run(int r, const uint4 (&xw)[2], float2 meta) const {
    const uint32_t v[4] = {q.x, q.y, q.z, q.w};
    // A nibble field under 0x6400 is the f16 1024 + q (low) or 1024 + 16 q (high); bytes 1 and 3 shift down to 0 and 2.
    const float s = h2_sum(v, xw, [r](uint32_t w, int j) {
      const __half2 h = hw((w >> 8 * j & (r ? 0x00F000F0 : 0x000F000F)) | 0x64006400);
      return r ? __hfma2(h, hw(0x2C002C00), hw(0xD400D400)) : __hsub2(h, hw(0x64006400));  // h / 16 - 64; h - 1024
    });
    return fmaf(a[r] * meta.x, s, b[r] * meta.y);
  }
};

template <>
struct h2<LARCH_Q6_K> {
  float a[2];    // run r's d sc
  uint4 ql, qh;  // qh shifted down to this lane's bit pairs
  __device__ void set(const unit<LARCH_Q6_K>& un, int u) {
#pragma unroll
    for (int r = 0; r < 2; ++r) a[r] = __half2float(un.d) * float(un.sc[r]);
    const int sh = u % 4 / 2 * 2;
    ql = un.ql;
    qh = {un.qh.x >> sh, un.qh.y >> sh, un.qh.z >> sh, un.qh.w >> sh};
  }
  __device__ float run(int r, const uint4 (&xw)[2], float2 meta) const {
    uint32_t v[4];  // fast<Q6_K>::quad's whole q6 bytes
#pragma unroll
    for (int wi = 0; wi < 4; ++wi)
      v[wi] = q6_word(word(ql, wi), word(qh, wi), r);
    // One PRMT puts bytes (0, 2) or (1, 3) under 0x64: the f16 1024 + q6, less 1056 is q6 - 32.
    const float s = h2_sum(v, xw, [](uint32_t w, int j) {
      return __hsub2(hw(__byte_perm(w, 0x64, j ? 0x4341 : 0x4240)), hw(0x64206420));
    });
    return __fmul_rn(a[r] * meta.x, s);
  }
};

// fast<Q3_K>'s set and fields; run r as h2<Q6_K>'s, its q3 + 4 under 0x64 less 1028.
template <>
struct h2<LARCH_Q3_K> : fast<LARCH_Q3_K> {
  __device__ float run(int r, const uint4 (&xw)[2], float2 meta) const {
    uint32_t v[4];
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) v[wi] = q3_word(word(q, wi), word(hm, wi), r);
    const float s = h2_sum(v, xw, [](uint32_t w, int j) {
      return __hsub2(hw(__byte_perm(w, 0x64, j ? 0x4341 : 0x4240)), hw(0x64046404));
    });
    return __fmul_rn(a[r] * meta.x, s);
  }
};

// x' and meta for the H2 kernels, one thread per 16-weight run j of row r: e = max(0, E - 21), E max|x|'s biased f16
// exponent, so |x 2^-e| < 128; xp[r k + 16 j, +16) = x 2^-e (exact but where it goes subnormal), each 4 as (x0, x2, x1,
// x3), and meta[r k / 16 + j] = {2^e, the f32 sum of the 16 x in order}. inf and NaN stay inf and NaN, as in f32.
// x16: x and x_ld keep 16-byte alignment (else 2-byte loads).
__global__ void mmvq_h2_prep_k(const __half* x, int64_t x_ld, int64_t k, int rows, bool x16, __half* xp,
                               float2* meta) {
  const int64_t runs = k / 16, t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= rows * runs) return;
  const int64_t r = t / runs, j = t % runs;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(x + r * x_ld + 16 * j);
  const uint4 xw[2] = {ld16(p, x16), ld16(p + 16, x16)};
  __half v[16];
  memcpy(v, xw, 32);
  uint32_t mx = 0;  // max |x| bits: |f16| orders as its bits
  float sum = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    mx = max(mx, uint32_t(__half_as_ushort(v[i]) & 0x7FFF));
    sum += __half2float(v[i]);
  }
  const int e = max(0, int(mx >> 10) - 21);
  const __half2 sc = hw(uint32_t(15 - e) << 10 | uint32_t(15 - e) << 26);  // 2^-e
  uint32_t o[8];
#pragma unroll
  for (int g = 0; g < 4; ++g)
#pragma unroll
    for (int j2 = 0; j2 < 2; ++j2) {
      const __half2 h = __hmul2(__halves2half2(v[4 * g + j2], v[4 * g + j2 + 2]), sc);
      memcpy(&o[2 * g + j2], &h, 4);
    }
  uint4* out = reinterpret_cast<uint4*>(xp + r * k + 16 * j);
  out[0] = {o[0], o[1], o[2], o[3]};
  out[1] = {o[4], o[5], o[6], o[7]};
  meta[t] = {__int_as_float((127 + e) << 23), sum};
}
#endif

// mmvq_k<T, 1, *, true>'s and mmvq_rows_k<T, R, true>'s unit: run 0 then run 1, each row, each column, acc += run,
// one f32 add order in both kernels, so a verify row is one-row mmvq's bytes. xp: x', rows k halves apart (16-byte
// aligned); meta: rows k / 16 apart.
template <int T, int R, int CW>
__device__ __forceinline__ void h2_unit(const h2<T> (&f)[CW], int u, const __half* xp, int64_t k, const float2* meta,
                                        float (&acc)[R][CW]) {
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int64_t at = unit<T>::k(u) + unit<T>::HI * r;
#pragma unroll
    for (int row = 0; row < R; ++row) {
      const uint4* p = reinterpret_cast<const uint4*>(xp + row * k + at);
      const uint4 xw[2] = {__ldg(p), __ldg(p + 1)};
      const float2 m = __ldg(meta + row * (k / 16) + at / 16);
#pragma unroll
      for (int c = 0; c < CW; ++c) acc[row][c] += f[c].run(r, xw, m);
    }
  }
}

// The launch shape, measured on sm_120 (Q4_K and Q6_K 8 Oct, Q8_0 9 Oct; kernels/tests/mmvq_bench.cu): CW
// columns per warp, so more rows reuse each widened x over more columns until registers run out (Q6_K's unit
// sooner), and KS warps splitting k, which pays at one row (Q8_0: up to 6). Q8_0 at rows 1-2 is fastest at CW 1
// (more warps for m8's small tensors); it was measured at rows 1, 2, 4-8, 10, 12, 14 and 16, and rows 3, 9, 11,
// 13 and 15 take a neighbour's shape. The rare Q5_K takes Q4_K's shape and Q4_0 and Q5_0 Q8_0's, untuned. mmvq()
// raises KS where n leaves the grid short of SMs.
// ponytail: measured on sm_120 only, used as-is on sm_86, sm_89 and HIP (6.2 wants a table per arch falling back to
// derived shapes; only the KS raise is derived): a table per arch when another arch shows in the bench. KS 2 at rows
// 2-6 costs Q8_0 at k 896 (m8 q/o/gate/up/lm_head) 4-33 % per tensor: KS 1 where lanes get < 16 weights, when m8
// multi-row decode shows in the serving bench. Rows >= 8 are FMA/register-bound with spills (Q4_K R16 ~1/3 of
// peak): row-split warps when multi-row decode shows in the serving bench. Q5_K/Q4_0/Q5_0 shapes by analogy
// (Q4_0/Q5_0 36-45 % at rows 8; Q5_K spills 104-264 B at R13-16): tune (Q4_K's CW 4 at rows >= 4) when one of them
// shows multi-row decode in the serving bench. Q3_K takes Q6_K's CW (its unit holds as many words), untuned; on
// sm_120 every Q3_K row routes to mmvq_rows_k (mmvq_route), so its mmvq_k at 2-16 rows runs on sm_86/89 past
// arch::mmvq_rows_up_to only, spilling 12-470 B at rows 7-16 (nvcc 13.3): tune it with the other formats there.
template <int T>
constexpr bool MMVQ_B32 = T == LARCH_Q8_0 || T == LARCH_Q4_0 || T == LARCH_Q5_0;  // 32-weight blocks
template <int T, int R>
constexpr int MMVQ_CW = MMVQ_B32<T> ? (R <= 2 ? 1 : R <= 12 ? 2 : 4)
                                    : R == 1 || R >= (T == LARCH_Q6_K || T == LARCH_Q3_K ? 7 : 4) ? 4 : 2;
template <int T, int R>
constexpr int MMVQ_KS = R == 1 || (MMVQ_B32<T> && R <= 6) ? 2 : 1;

// Whether the image __CUDA_ARCH__ = IMAGE builds the H2 kernels: arch::mmvq_h2's list, read on the host side (nvcc
// rejects a constexpr host call in device code).
template <int IMAGE>
constexpr bool H2_IMAGE = arch::mmvq_h2_arch(IMAGE / 10);

// mmvq for T with R rows of x: a block of warps, each CW consecutive columns over a 1 / KS share of k (KS 0:
// split, the raised KS of a small-n launch), summed through shared memory. w16 and x16 say w's blocks and x's
// rows are 16-byte aligned (w16 only read for Q4_K and Q5_K: Q6_K's, Q8_0's, Q4_0's and Q5_0's blocks, 210, 34,
// 18 and 22 bytes, never are; Q3_K's 110 neither). H2 (Q4_K, Q6_K and Q3_K at R 1, sm_86/89): x is
// mmvq_h2_prep_k's x', meta its run sums.
template <int T, int R, int KS, bool H2 = false>
__global__ void __launch_bounds__(MMVQ_THREADS)
    mmvq_k(const uint8_t* __restrict__ w, int64_t w_ld, int64_t n, int units, const __half* __restrict__ x,
           int64_t x_ld, const float2* __restrict__ meta, const float* __restrict__ bias, int64_t bias_ld, void* out,
           int64_t out_ld, bool out_f32, bool w16, bool x16, int split) {
  constexpr int CW = MMVQ_CW<T, R>;
#if defined(__CUDA_ARCH__)
  if constexpr (H2 && !H2_IMAGE<__CUDA_ARCH__>) {  // h2<T> runs where arch::mmvq_h2 routes it
    asm volatile("trap;");  // not the trap intrinsic: hipify-perl cannot convert it (even named so in a comment), and the HIP build fails on that
    return;
  }
#endif
#if defined(__HIP_PLATFORM_AMD__)
  constexpr bool FAST = false;
#else
  constexpr bool FAST = R == 1;  // more rows spill with fast<T>'s registers (Q4_K R4-7, Q5_K R4-6, Q6_K R9)
#endif
  const int splits = KS ? KS : split;
  const int lane = int(threadIdx.x) % warpSize, warp = int(threadIdx.x) / warpSize, ks = warp % splits;
  const int64_t col0 = (int64_t(blockIdx.x) * (blockDim.x / warpSize / splits) + warp / splits) * CW;
  const uint8_t* row[CW];
#pragma unroll
  for (int c = 0; c < CW; ++c) row[c] = w + (col0 + c < n ? col0 + c : n - 1) * w_ld;  // past n: read, not stored
  float acc[R][CW] = {};
  // One copy of the loop per alignment, so neither load path is issued (predicated off) in the other; every
  // format but Q4_K and Q5_K builds one per x alignment only.
  const auto loop = [&](auto w_a16, auto x_a16) {
    const int step = warpSize * splits;
    int u = lane + warpSize * ks;
    unit<T> next[CW];
    if (u < units)
#pragma unroll
      for (int c = 0; c < CW; ++c) next[c].load(row[c], u, decltype(w_a16)::value);
    for (; u < units; u += step) {
      std::conditional_t<H2, h2<T>, std::conditional_t<FAST, fast<T>, unit<T>>> un[CW];
#pragma unroll
      for (int c = 0; c < CW; ++c)
        if constexpr (FAST || H2)
          un[c].set(next[c], u);  // copies what it decodes, so next reloads below
        else
          un[c] = next[c];
      if (u + step < units)  // in flight while this unit is decoded
#pragma unroll
        for (int c = 0; c < CW; ++c) next[c].load(row[c], u + step, decltype(w_a16)::value);
      if constexpr (H2) {
        h2_unit<T>(un, u, x, x_ld, meta, acc);
      } else {
#pragma unroll
        for (int p = 0; p < unit<T>::P; ++p) {
          float wv[CW][8];
#pragma unroll
          for (int c = 0; c < CW; ++c)
            if constexpr (FAST) {
              un[c].quad(p, 0, wv[c]);
              un[c].quad(p, 1, wv[c] + 4);
            } else {
              un[c].piece(p, u, wv[c]);
            }
          const __half* xp = x + unit<T>::k(u) + 8 * (p & 1) + unit<T>::HI * (p >> 1);
#pragma unroll
          for (int r = 0; r < R; ++r) {
            const uint4 xv = ld16(reinterpret_cast<const uint8_t*>(xp + r * x_ld), decltype(x_a16)::value);
#pragma unroll
            for (int e = 0; e < 8; e += 2) {
              const float2 xf = h2f(word(xv, e / 2));
#pragma unroll
              for (int c = 0; c < CW; ++c) acc[r][c] = fmaf(wv[c][e + 1], xf.y, fmaf(wv[c][e], xf.x, acc[r][c]));
            }
          }
        }
      }
    }
  };
  using yes = std::true_type;
  using no = std::false_type;
  if ((T == LARCH_Q4_K || T == LARCH_Q5_K) && w16)  // H2's x' is always aligned
    x16 || H2 ? loop(yes(), yes()) : loop(yes(), no());
  else
    x16 || H2 ? loop(no(), yes()) : loop(no(), no());
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int c = 0; c < CW; ++c) acc[r][c] = vv_warp_reduce_sum(acc[r][c]);
  if (splits > 1) {  // warp ks of a column group adds its share into the group's first warp, in ks order
    __shared__ float part[MMVQ_THREADS / 32][R * CW];
#pragma unroll
    for (int i = 0; i < R * CW; ++i)
      if (lane == i % warpSize) part[warp][i] = acc[i / CW][i % CW];
    __syncthreads();
    if (ks) return;
    for (int s = 1; s < splits; ++s)
#pragma unroll
      for (int i = 0; i < R * CW; ++i) acc[i / CW][i % CW] += part[warp + s][i];
  }
#pragma unroll
  for (int i = 0; i < R * CW; ++i) {
    const int r = i / CW;
    const int64_t col = col0 + i % CW;
    if (lane != i % warpSize || col >= n) continue;
    const float v = bias ? acc[r][i % CW] + bias[col * bias_ld] : acc[r][i % CW];
    if (out_f32)
      store(static_cast<float*>(out) + r * out_ld + col, v);
    else
      store(static_cast<__half*>(out) + r * out_ld + col, v);
  }
}

// Calls f with rows as a compile-time R, 1..MMVQ_ROWS.
template <int R = 1, class F>
static void with_rows(int rows, F&& f) {
  if constexpr (R < MMVQ_ROWS)
    if (rows != R) return with_rows<R + 1>(rows, f);
  f(std::integral_constant<int, R>());
}

// Blocks of MMVQ_THREADS covering n columns, CW a warp and ks warps a column group.
static int64_t mmvq_blocks(const stream& s, int64_t n, int cw, int ks) {
  const int64_t cols = int64_t(MMVQ_THREADS / s.warp / ks) * cw;
  return (n + cols - 1) / cols;
}

// mmvq_k<T, R, *>'s k-split for w, and so (R = 1) the split every row of mmvq_verify, and of mmvq's rows routed to
// mmvq_rows_k, sums in. The table's KS is a template parameter, so its kernels keep their code (a runtime KS changed it
// and cost m1/m5 decode steps up to 8 % at some row counts); one kernel per (T, R) takes a raised KS at run time
// (templating 2 and 4 too built quant.cu 3x slower). The table's KS is doubled while the grid leaves SMs short of two
// blocks each, k holds >= 16 weights per lane of the doubled split and the doubled grid still runs in one wave. The
// guard counts weights, not units: with 32-weight units (all but Q8_0's 16) up to half the doubled split's lanes may
// idle. Each threshold was chosen by per-tensor sweeps of KS 1/2/4 on sm_120 (m1, m5 k/v; m8 k/v, q/o, down): m5's k/v
// (k 3584) are fastest at KS 4 with 0.875 units per lane; without the one-wave cap, rows 8-16 lost up to 4.5 % of the
// decode step to a second wave; without the 16 weights, m8 q/o (k 896) lost 16-36 % per tensor at rows 8-16 to the
// per-warp sums.
// ponytail: one rule for every format; m1's k/v lose up to 14 % per tensor where the raised KS is slower (Q4_K at
// rows 1 and 15-16, Q6_K at rows 4-6; <= 0.8 % of the decode step): a per-format cap when that shows in the serving
// bench.
template <int T, int R = 1>
static int mmvq_split(const stream& s, const view& w, int sms) {
  constexpr int KS = MMVQ_KS<T, R>;
  const auto blocks = [&](int ks) { return mmvq_blocks(s, w.rows, MMVQ_CW<T, R>, ks); };
  const auto one_wave = [&](int ks) {
    int per_sm;
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ks == KS ? mmvq_k<T, R, KS> : mmvq_k<T, R, 0>,
                                                     MMVQ_THREADS, 0),
       "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    return blocks(ks) <= int64_t(per_sm) * sms;
  };
  int ks = KS;
  while (ks < MMVQ_THREADS / s.warp && blocks(ks) < 2 * sms && w.cols / (2 * ks * s.warp) >= 16 && one_wave(2 * ks))
    ks *= 2;
  return ks;
}

// Launches kernel (mmvq_k or mmvq_rows_k, which share an argument list) over w's columns, cw a warp and ks warps a
// column group; meta: the H2 kernels' (x is then x').
template <int T, class K>
static void mmvq_kernel_launch(const stream& s, K kernel, int cw, int ks, const view& w, const view& x, const view& b,
                               const view& o, const float2* meta = nullptr) {
  // Q4_K and Q5_K rows are whole 144- and 176-byte blocks, so every block keeps w.p's 16-byte alignment.
  const bool w16 = uintptr_t(w.p) % 16 == 0, x16 = (uintptr_t(x.p) | uint64_t(x.ld) * 2) % 16 == 0;
  const int64_t blocks = mmvq_blocks(s, w.rows, cw, ks);
  if (blocks > 0x7fffffff) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 - 1 blocks of columns"};
  launch(kernel, unsigned(blocks), MMVQ_THREADS, s.s, w.p, w.row_bytes, w.rows, int(w.cols / (8 * unit<T>::P)),
         x.as<__half>(), x.ld, meta, b.as<float>(), b.ld, o.p, o.ld, o.dtype == LARCH_F32, w16, x16, ks);
}

// The formats with an h2<T> decoder.
template <int T>
constexpr bool MMVQ_H2 = T == LARCH_Q4_K || T == LARCH_Q6_K || T == LARCH_Q3_K;

// mmvq_k over w's columns at its split; meta (one row): mmvq_k<T, 1, *, true> on x'. It, mmvq_route and
// mmvq_verify_launch are function templates, not lambdas: MSVC loses enclosing generic lambdas' constexpr locals in
// nested lambdas (#93), so with_rows' lambdas name T and R only as template arguments.
template <int T, int R>
static void mmvq_launch(const stream& s, const view& w, const view& x, const view& b, const view& o, int sms,
                        const float2* meta = nullptr) {
  const int ks = mmvq_split<T, R>(s, w, sms);
  constexpr int KS = MMVQ_KS<T, R>;
#if !defined(__HIP_PLATFORM_AMD__)  // HIP has no h2<T>
  if constexpr (R == 1 && MMVQ_H2<T>)
    if (meta)
      return mmvq_kernel_launch<T>(s, ks == KS ? mmvq_k<T, 1, KS, true> : mmvq_k<T, 1, 0, true>, MMVQ_CW<T, 1>, ks, w,
                                   x, b, o, meta);
#endif
  mmvq_kernel_launch<T>(s, ks == KS ? mmvq_k<T, R, KS> : mmvq_k<T, R, 0>, MMVQ_CW<T, R>, ks, w, x, b, o);
}

constexpr int ID_THREADS = 128;  // mmvq_id_k's block: 4 warps of 32 (2 of 64)

// mmvq_id_k's dot of expert row wr with x row xr: lane l takes pieces of 8 weights from 8 l on, warpSize pieces apart,
// then the warp's sum, which every lane holds. The fused entries' kernels call it too, for mmvq_id's bytes, with U
// pieces unrolled so their loads are in flight together (the sum's order is the same).
template <int T, int U = 1>
__device__ __forceinline__ float id_dot(const uint8_t* wr, bool w8, const __half* xr, int64_t k, int lane) {
  float acc = 0;
#pragma unroll U
  for (int64_t c = 8 * int64_t(lane); c < k; c += 8 * int64_t(warpSize)) {
    float v[8];
    decode8<T>(wr, c, 8, w8, v);
#pragma unroll
    for (int e = 0; e < 8; ++e) acc += v[e] * __half2float(xr[c + e]);
  }
  return vv_warp_reduce_sum(acc);
}

// mmvq_id: one warp per column of the chosen expert, for one (row, slot) pair (blockIdx.x) or, weighted, for one
// row over its slots in ascending order (the CPU reference's sum, in f32). Lanes take pieces of 8 weights.
// From MOE_GROUPED_ROWS rows mmvq_id_grouped takes the call instead.
// mmvq_id_mixed (table set): expert e's rows are slot table[e] of w (dtype T) when it is >= 0, else slot -1 - table[e]
// of wc (dtype TC), each pair decoding only its own expert's rows, with id_dot's arithmetic at its dtype.
// ponytail: each warp decodes its own expert row and there is no load pipelining; reuse mmvq_k's unit decoders when
// decode or verify (below MOE_GROUPED_ROWS) shows mmvq_id_k in the hybrid MoE target's decode profile.
template <int T, int TC = T>
__global__ void mmvq_id_k(const uint8_t* w, int64_t w_rb, bool w8, const uint8_t* wc, int64_t wc_rb,
                          const int32_t* table, const __half* x, int64_t x_ld, int64_t x_slot, const int32_t* ids,
                          int64_t ids_ld, const float* bias, int64_t bias_ld, const float* wt, int64_t wt_ld,
                          uint8_t* out, int64_t out_ld, int64_t out_slot, bool out32, int n, int64_t k, int slots,
                          bool weighted) {
  const int lane = int(threadIdx.x) % warpSize;
  const int col = int(blockIdx.y) * (int(blockDim.x) / warpSize) + int(threadIdx.x) / warpSize;
  if (col >= n) return;  // the whole warp
  const int64_t r = weighted ? blockIdx.x : blockIdx.x / slots;
  const int first = weighted ? 0 : int(blockIdx.x % slots), last = weighted ? slots : first + 1;
  float sum = 0;
  for (int j = first; j < last; ++j) {
    const int32_t e = ids[r * ids_ld + j];  // in range: the caller's moe_route ids
    const int64_t row = int64_t(e) * n + col;
    const __half* xr = x + r * x_ld + j * x_slot;
    const int32_t s = table ? table[e] : 0;
    float y = !table  ? id_dot<T>(w + row * w_rb, w8, xr, k, lane)
              : s >= 0 ? id_dot<T>(w + (int64_t(s) * n + col) * w_rb, w8, xr, k, lane)
                       : id_dot<TC>(wc + ((-1 - int64_t(s)) * n + col) * wc_rb, false, xr, k, lane);
    if (bias) y += bias[row * bias_ld];
    if (weighted) {
      sum += wt[r * wt_ld + j] * y;
    } else if (!lane) {
      uint8_t* o = out + (r * out_ld + j * out_slot) * (out32 ? 4 : 2);
      if (out32) reinterpret_cast<float*>(o)[col] = y;
      else reinterpret_cast<__half*>(o)[col] = __float2half(y);
    }
  }
  if (weighted && !lane) {
    uint8_t* o = out + r * out_ld * (out32 ? 4 : 2);
    if (out32) reinterpret_cast<float*>(o)[col] = sum;
    else reinterpret_cast<__half*>(o)[col] = __float2half(sum);
  }
}

#if !defined(__HIP_PLATFORM_AMD__)
template <int T, int TC>
static bool mmvq_id_grouped(const stream&, const view&, const view&, const view&, const view&, const view&,
                            const view&, const view&, const view&);
// From this many rows mmvq_id groups its pairs by expert (mmvq_id_grouped). The model slices check-step mmvq_id calls
// into ID_ROWS = 3 row slices (larch-model transformer.rs), which keeps check rows under this threshold, so checks keep
// decode's bytes (exact speculation); one-row decode is under it too. Measured on the small model (t/s, grouped vs
// mmvq_id_k): 33 rows ~336 vs ~250, 48 ~372 vs ~300, 64 ~410 vs ~320, 128 ~570 vs ~395: grouped wins from 33.
constexpr int64_t MOE_GROUPED_ROWS = 33;
#endif

// mmvq_id (wc and table empty) or mmvq_id_mixed (cold experts in wc, of dtype TC) at w's dtype T.
template <int T, int TC>
static void mmvq_id_issue(const stream& s, const view& w, const view& wc, const view& table, const view& x,
                          const view& ids, const view& b, const view& wt, const view& o) {
  const int64_t slots = ids.cols, n = o.shape[o.ndim - 1], k = w.cols;
  const bool weighted = wt.p != nullptr;
  const int64_t pairs = weighted ? x.rows : x.rows * slots, warps = ID_THREADS / s.warp;
  if (pairs > 0x7fffffff) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 - 1 (row, slot) pairs"};
  if ((n + warps - 1) / warps > 65535) throw failure{LARCH_ERR_UNSUPPORTED, "more than 65535 blocks of columns"};
#if !defined(__HIP_PLATFORM_AMD__)
  if (x.rows >= MOE_GROUPED_ROWS && slots && s.arch_code >= 70 &&
      mmvq_id_grouped<T, TC>(s, w, wc, table, x, ids, b, wt, o))
    return;
#endif
  const dim3 grid(unsigned(pairs), unsigned((n + warps - 1) / warps));
  launch(mmvq_id_k<T, TC>, grid, ID_THREADS, s.s, w.p, w.row_bytes, uintptr_t(w.p) % 8 == 0, wc.p, wc.row_bytes,
         table.as<int32_t>(), x.as<__half>(), x.ld, x.ndim == 3 ? k : 0, ids.as<int32_t>(), ids.ld, b.as<float>(),
         b.ld, wt.as<float>(), wt.ld, o.p, o.ld, weighted ? 0 : n, o.dtype == LARCH_F32, int(n), k, int(slots),
         weighted);
}

// The checked mmvq_id or mmvq_id_mixed call on s: a quant w, and for the second only Q2_K cold experts.
static void mmvq_id_run(const stream& s, const view& w, const view& wc, const view& table, const view& x,
                        const view& ids, const view& b, const view& wt, const view& o) {
  aligned({w, wc, table, x, ids, b, wt, o});
  if (table.p && wc.dtype != LARCH_Q2_K) throw failure{LARCH_ERR_UNSUPPORTED, "cold experts other than Q2_K"};
  if (!x.rows) return;
  with_dtype<true>(w.dtype, [&](auto t) {
    constexpr int T = decltype(t)::value;
    if constexpr (T != LARCH_F16 && T != LARCH_F32) {  // check_mmvq_id has refused f16 and f32 w
      if (table.p)
        mmvq_id_issue<T, LARCH_Q2_K>(s, w, wc, table, x, ids, b, wt, o);
      else
        mmvq_id_issue<T, T>(s, w, wc, table, x, ids, b, wt, o);
    } else {
      throw failure{LARCH_ERR_UNSUPPORTED, "mmvq_id weight must be a quant dtype"};
    }
  });
}

larch_status mmvq_id(larch_stream* vs, const larch_tensor* w_quant, const larch_tensor* x_f16,
                      const larch_tensor* ids_i32, const larch_tensor* bias_or_null,
                      const larch_tensor* weights_or_null, const larch_tensor* out) {
  return run("mmvq_id", vs, [&](const stream& s) {
    const cpu::mmvq_id_checked c =
        cpu::check_mmvq_id(sizer(s), w_quant, x_f16, ids_i32, bias_or_null, weights_or_null, out);
    mmvq_id_run(s, c.w, view(), view(), c.x, c.ids, c.bias, c.weights, c.out);
  });
}

larch_status mmvq_id_mixed(larch_stream* vs, const larch_tensor* w_hot, const larch_tensor* w_cold,
                            const larch_tensor* table_i32, const larch_tensor* x_f16, const larch_tensor* ids_i32,
                            const larch_tensor* weights_or_null, const larch_tensor* out) {
  return run("mmvq_id_mixed", vs, [&](const stream& s) {
    const cpu::mmvq_id_mixed_checked c =
        cpu::check_mmvq_id_mixed(sizer(s), w_hot, w_cold, table_i32, x_f16, ids_i32, weights_or_null, out);
    mmvq_id_run(s, c.id.w, c.cold, c.table, c.id.x, c.id.ids, c.id.bias, c.id.weights, c.id.out);
  });
}

constexpr int MOE_ROUTE_MAX = 512;  // experts per row: 16 logits per lane of a 32-wide warp

// moe_route: one warp per row. Lane l holds experts l, l + warpSize, ...; each of the k picks takes the largest
// logit not yet taken, the lower id on a tie, as the CPU reference's partial sort does. f32 where the CPU
// reference sums in double. A NaN logit, which the CPU reference refuses, gives NaN weights here.
__global__ void moe_route_k(const float* lg, int64_t lg_ld, int32_t* ids, int64_t ids_ld, float* wts, int64_t w_ld,
                            int experts, int k, float scale) {
  const int lane = int(threadIdx.x), per = (experts + warpSize - 1) / warpSize;
  const float* l = lg + int64_t(blockIdx.x) * lg_ld;
  float v[MOE_ROUTE_MAX / 32];
  float mx = -INFINITY;
#pragma unroll
  for (int i = 0; i < MOE_ROUTE_MAX / 32; ++i) {
    const int e = lane + warpSize * i;
    v[i] = i < per && e < experts ? l[e] : -INFINITY;
    mx = fmaxf(mx, v[i]);
  }
  mx = vv_warp_reduce_max(mx);
  float z = 0;
#pragma unroll
  for (int i = 0; i < MOE_ROUTE_MAX / 32; ++i)
    if (i < per && lane + warpSize * i < experts) z += expf(v[i] - mx);
  z = vv_warp_reduce_sum(z);
  uint32_t taken = 0;  // bit i: expert lane + warpSize i is picked
  float top = 0;
  int32_t* oi = ids + int64_t(blockIdx.x) * ids_ld;
  float* ow = wts + int64_t(blockIdx.x) * w_ld;
  for (int j = 0; j < k; ++j) {
    float best = -INFINITY;
    int id = INT_MAX;
#pragma unroll
    for (int i = 0; i < MOE_ROUTE_MAX / 32; ++i) {
      const int e = lane + warpSize * i;
      if (i < per && e < experts && !(taken >> i & 1) && (id == INT_MAX || v[i] > best)) {
        best = v[i];
        id = e;
      }
    }
    const float m = vv_warp_reduce_max(id == INT_MAX ? -INFINITY : best);
    int pick = id != INT_MAX && best == m ? id : INT_MAX;
    for (int o = warpSize / 2; o > 0; o /= 2) pick = min(pick, __shfl_xor_sync(VV_FULL_MASK, pick, o));
    if (lane == pick % warpSize) taken |= 1u << (pick / warpSize);
    const float p = expf(m - mx) / z;
    top += p;
    if (!lane) {
      oi[j] = pick;
      ow[j] = p;
    }
  }
  // Over the sum first, then the scale: scale / top overflows for a large finite scale (FLT_MAX, top 0.5).
  const float sum = fmaxf(top, 6.103515625e-05f), sc = scale == 0.0f ? 1.0f : scale;  // 2^-14, as llama.cpp
  if (!lane)
    for (int j = 0; j < k; ++j) ow[j] = ow[j] / sum * sc;
}

larch_status moe_route(larch_stream* vs, const larch_tensor* logits_f32, float scale,
                        const larch_tensor* out_ids_i32, const larch_tensor* out_weights_f32) {
  return run("moe_route", vs, [&](const stream& s) {
    const cpu::route_checked c = cpu::check_moe_route(sizer(s), logits_f32, scale, out_ids_i32, out_weights_f32);
    const view lg = c.logits, ids = c.ids, w = c.weights;
    aligned({lg, ids, w});
    if (!lg.rows || !ids.cols) return;
    if (lg.cols > MOE_ROUTE_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "more than 512 experts"};
    if (lg.rows > 0x7fffffff) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 - 1 rows"};
    launch(moe_route_k, unsigned(lg.rows), s.warp, s.s, lg.as<float>(), lg.ld, ids.as<int32_t>(), ids.ld,
           w.as<float>(), w.ld, int(lg.cols), int(ids.cols), scale);
  });
}

#if !defined(__HIP_PLATFORM_AMD__)
// The xor butterfly of vv_warp_reduce_sum (offsets 16, 8, 4, 2, 1) over M values at once, M a power of two, each lane
// keeping only part of the sums: at each offset a lane adds its partner's half of its values into its own half, so a
// pair of lanes shuffles each value once where the butterfly shuffled every value in both lanes (shuffles issue at a
// quarter of the FMA rate). Every sum is the butterfly's, operand for operand. Lane l ends with the sums of values
// [l M / 32, (l + 1) M / 32) in v[0, M / 32), or (M < 32) of value l M / 32 in v[0].
template <int O = 16, int N, int M>
__device__ __forceinline__ void warp_sums(float (&v)[M], int lane) {
  if constexpr (O > 0) {
    if constexpr (N > 1) {
      const bool hi = lane & O;
#pragma unroll
      for (int j = 0; j < N / 2; ++j) {
        const float keep = hi ? v[j + N / 2] : v[j], send = hi ? v[j] : v[j + N / 2];
        v[j] = keep + __shfl_xor_sync(VV_FULL_MASK, send, O);
      }
      warp_sums<O / 2, N / 2>(v, lane);
    } else {
      v[0] += __shfl_xor_sync(VV_FULL_MASK, v[0], O);
      warp_sums<O / 2, 1>(v, lane);
    }
  }
}

// The shape table's column for T: the formats from Q4_K in dtype order, then Q3_K.
template <int T>
constexpr int MMVQ_COL = T == LARCH_Q3_K ? 6 : T - LARCH_Q4_K;

__host__ __device__ constexpr int pow2_at_least(int n) { return n <= 1 ? 1 : 2 * pow2_at_least((n + 1) / 2); }

// mmvq_rows_k's shape per (rows, format), CW * 10 + B: CW columns a warp, B the blocks an SM __launch_bounds__ asks
// for (4: 128 registers, 3: 168, 1: 255). Measured on sm_120 (lever, cold, m1 and m5 every format, m8 Q8_0, Q4_0 and
// Q5_0; 25 Oct): the fastest shape whose kernel keeps STACK 0. Rows 9, 11, 13 and 15 take a neighbour's, except Q4_K
// rows 11 and 13 and Q5_0 row 13: CW 3, where the neighbours' CW 4 spills.
// ponytail: B 3 at rows 7-10 measured 5-12 % faster on the K quants but spills (8-88 B), as does CW 4 for K quants at
// rows 11-16 (up to 22 %, Q4_K rows 12): retune when ptxas or the kernel changes, or when verify at 7+ rows shows in
// the speculative-decoding bench.
// ponytail: Q3_K's column not measured, derived from Q4_K, except rows 7 (B 1), 12 and 14 (CW 3), where Q4_K's shape
// spilled 12-44 B under nvcc 13.4: tune it when the 35B's verify at 7+ rows shows in the speculative-decoding bench.
inline constexpr int8_t MMVQ_VSHAPE[MMVQ_ROWS - 1][7] = {
    // Q4_K Q5_K Q6_K Q8_0 Q4_0 Q5_0 Q3_K
    {44, 44, 24, 24, 44, 44, 44},  // 2 rows
    {44, 44, 24, 24, 44, 24, 44},  // 3
    {44, 34, 34, 44, 44, 44, 44},  // 4
    {43, 43, 43, 43, 43, 43, 43},  // 5
    {33, 33, 43, 43, 43, 43, 33},  // 6
    {43, 41, 31, 43, 43, 41, 41},  // 7
    {41, 41, 41, 43, 43, 41, 41},  // 8
    {41, 41, 41, 43, 41, 41, 41},  // 9
    {41, 41, 41, 43, 41, 41, 41},  // 10
    {31, 31, 31, 41, 41, 41, 31},  // 11
    {41, 31, 31, 41, 41, 41, 31},  // 12
    {31, 31, 31, 41, 41, 31, 31},  // 13
    {41, 31, 31, 41, 31, 41, 31},  // 14
    {31, 31, 31, 41, 31, 31, 31},  // 15
    {31, 31, 31, 41, 31, 31, 31},  // 16
};
// The table's CW, except that code built before sm_120 (PRE120: __CUDA_ARCH__ < 1200 in the kernel, s.image_arch at
// the launch, so both agree) takes CW 3 for every CW 4 at B 1 from 12 rows, five of which spill 8-56 B on sm_86/89.
// ponytail: pre-120 shape not measured: retune it on the 3050/4060 runs.
template <int T, int R, bool PRE120 = false>
constexpr int MMVQ_VCW =
    PRE120 && R >= 12 && MMVQ_VSHAPE[R - 2][MMVQ_COL<T>] == 41 ? 3 : MMVQ_VSHAPE[R - 2][MMVQ_COL<T>] / 10;
template <int T, int R>
constexpr int MMVQ_VB = MMVQ_VSHAPE[R - 2][MMVQ_COL<T>] % 10;

// mmvq's kernel at R rows (2-16) of a format: 1 mmvq_rows_k at the one-row split, whose rows are byte-equal to one-row
// mmvq, 0 the table's mmvq_k. Each is the faster by the decode step summed over m1 and m5 (and m8 for Q8_0, Q4_0 and
// Q5_0), the two timed interleaved (lever, cold, 3 reps, sm_120, 25 Oct); within 1 % (Q8_0 rows 2, 3, 14 and 15, Q6_K
// 9-12, Q4_0 14) the pick is noise.
// ponytail: measured on sm_120; arch::mmvq_rows_up_to caps it on sm_86 and sm_89 (by judgment, see there). Re-measure
// when either kernel changes. Q3_K is not in the table: mmvq_route routes its every row. Not timed: at rows 5-16 its
// mmvq_k spills 4-468 B on sm_120 (ptxas spill stores, nvcc 13.3 and 13.4) where mmvq_rows_k keeps STACK 0, and rows
// 2-4, which do not spill, go by analogy with the other formats, whose rows 2-4 all route but Q8_0's row 3. Time the
// two when Q3_K at 2+ rows shows in the 35B's serving bench.
inline constexpr bool MMVQ_ROUTE[MMVQ_ROWS - 1][6] = {
    // Q4_K Q5_K Q6_K Q8_0 Q4_0 Q5_0
    {1, 1, 1, 1, 1, 1},  // 2 rows
    {1, 1, 1, 0, 1, 1},  // 3
    {1, 1, 1, 1, 1, 1},  // 4
    {1, 1, 1, 1, 1, 1},  // 5
    {1, 1, 1, 1, 1, 1},  // 6
    {1, 1, 0, 1, 1, 1},  // 7
    {1, 1, 0, 1, 1, 1},  // 8
    {1, 1, 1, 1, 1, 1},  // 9
    {1, 1, 0, 1, 1, 1},  // 10
    {0, 0, 0, 1, 1, 1},  // 11
    {1, 0, 1, 1, 1, 1},  // 12
    {0, 0, 1, 1, 1, 0},  // 13
    {1, 1, 1, 0, 0, 1},  // 14
    {0, 1, 1, 1, 0, 0},  // 15
    {0, 1, 1, 1, 0, 0},  // 16
};

// mmvq_verify's R rows (2-16), each bit-identical to mmvq_k<T, 1, *> on that row alone at its split (the launch's ks1,
// mmvq_split<T>): lane l of split s sums units l + 32 s + 32 ks1 j in j order, pieces in order, two FMAs a weight pair
// on fast<T>'s values (unit<T>::piece's), then warp_sums (the butterfly's sums), the splits' sums in split order, bias,
// the f32 or f16 store. The block is mmvq_k's: CW columns a warp, ks1 warps a column group. No prefetch: the next
// unit's loads cost registers that more resident warps cover better.
// ponytail: rows 7-16 are FMA/issue-bound under the one-row split (R FMAs, R / CW x conversions and ~3.3 decode
// instructions a weight at ~55 % issue; m1 Q4_K 8 rows 0.61 of the copy peak): a grid of persistent tiles measured
// +4-7 % at rows 6-8 but spilled; take it when verify at 7+ rows shows in the speculative-decoding bench, or decode
// at 6-8 rows (larch_mmvq's routed rows on sm_120) in the serving bench.
// H2: as mmvq_k's, through the same h2_unit.
template <int T, int R, bool H2 = false>
__global__ void __launch_bounds__(MMVQ_THREADS, MMVQ_VB<T, R>)
    mmvq_rows_k(const uint8_t* __restrict__ w, int64_t w_ld, int64_t n, int units, const __half* __restrict__ x,
                int64_t x_ld, const float2* __restrict__ meta, const float* __restrict__ bias, int64_t bias_ld,
                void* out, int64_t out_ld, bool out_f32, bool w16, bool x16, int splits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
  constexpr int CW = MMVQ_VCW<T, R, true>;
#else
  constexpr int CW = MMVQ_VCW<T, R>;
#endif
#if defined(__CUDA_ARCH__)
  if constexpr (H2 && !H2_IMAGE<__CUDA_ARCH__>) {  // h2<T> runs where arch::mmvq_h2 routes it
    asm volatile("trap;");  // not the trap intrinsic: hipify-perl cannot convert it (even named so in a comment), and the HIP build fails on that
    return;
  }
#endif
  const int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32, ks = warp % splits;
  const int64_t col0 = (int64_t(blockIdx.x) * (MMVQ_THREADS / 32 / splits) + warp / splits) * CW;
  float acc[R][CW] = {};
  const auto loop = [&](auto w_a16, auto x_a16) {
    for (int u = lane + 32 * ks; u < units; u += 32 * splits) {
      std::conditional_t<H2, h2<T>, fast<T>> f[CW];
#pragma unroll
      for (int c = 0; c < CW; ++c) {
        unit<T> un;
        un.load(w + (col0 + c < n ? col0 + c : n - 1) * w_ld, u, decltype(w_a16)::value);  // past n: read, not stored
        f[c].set(un, u);
      }
      if constexpr (H2) {
        h2_unit<T>(f, u, x, x_ld, meta, acc);
      } else {
#pragma unroll
        for (int p = 0; p < unit<T>::P; ++p) {
          float wv[CW][8];
#pragma unroll
          for (int c = 0; c < CW; ++c) {
            f[c].quad(p, 0, wv[c]);
            f[c].quad(p, 1, wv[c] + 4);
          }
          const __half* xp = x + unit<T>::k(u) + 8 * (p & 1) + unit<T>::HI * (p >> 1);
#pragma unroll
          for (int r = 0; r < R; ++r) {
            const uint4 xv = ld16(reinterpret_cast<const uint8_t*>(xp + r * x_ld), decltype(x_a16)::value);
#pragma unroll
            for (int e = 0; e < 8; e += 2) {
              const float2 xf = h2f(word(xv, e / 2));
#pragma unroll
              for (int c = 0; c < CW; ++c) acc[r][c] = fmaf(wv[c][e + 1], xf.y, fmaf(wv[c][e], xf.x, acc[r][c]));
            }
          }
        }
      }
    }
  };
  using yes = std::true_type;
  using no = std::false_type;
  if ((T == LARCH_Q4_K || T == LARCH_Q5_K) && w16)  // H2's x' is always aligned
    x16 || H2 ? loop(yes(), yes()) : loop(yes(), no());
  else
    x16 || H2 ? loop(no(), yes()) : loop(no(), no());
  // warp_sums' values, those a lane ends with, and the lanes that end with the same one
  constexpr int M = pow2_at_least(R * CW), L = M < 32 ? 1 : M / 32, DUP = M < 32 ? 32 / M : 1;
  float v[M];
#pragma unroll
  for (int i = 0; i < M; ++i) v[i] = i < R * CW ? acc[i / CW][i % CW] : 0.0f;
  warp_sums<16, M>(v, lane);
  __shared__ float part[MMVQ_THREADS / 32][32 * L];
#pragma unroll
  for (int j = 0; j < L; ++j) part[warp][lane * L + j] = v[j];
  __syncthreads();
  if (ks || lane % DUP) return;  // split 0 stores, one lane a sum
  for (int s = 1; s < splits; ++s)
#pragma unroll
    for (int j = 0; j < L; ++j) v[j] += part[warp + s][lane * L + j];
#pragma unroll
  for (int j = 0; j < L; ++j) {
    const int i = lane / DUP * L + j;  // value i: row i / CW, column col0 + i % CW
    const int64_t col = col0 + i % CW;
    if (i >= R * CW || col >= n) continue;
    const float val = bias ? v[j] + bias[col * bias_ld] : v[j];
    if (out_f32)
      store(static_cast<float*>(out) + (i / CW) * out_ld + col, val);
    else
      store(static_cast<__half*>(out) + (i / CW) * out_ld + col, val);
  }
}

// mmvq_rows_k over w's columns at the one-row split ks1; meta: mmvq_rows_k<T, R, true> on x'.
template <int T, int R>
static void mmvq_rows_launch(const stream& s, const view& w, const view& x, const view& b, const view& o, int ks1,
                             const float2* meta = nullptr) {
  const int cw = s.image_arch < 1200 ? MMVQ_VCW<T, R, true> : MMVQ_VCW<T, R>;
  if constexpr (MMVQ_H2<T>)
    if (meta) return mmvq_kernel_launch<T>(s, mmvq_rows_k<T, R, true>, cw, ks1, w, x, b, o, meta);
  mmvq_kernel_launch<T>(s, mmvq_rows_k<T, R>, cw, ks1, w, x, b, o);
}

// x's rows as mmvq_h2_prep_k leaves them for the H2 kernels, in scratch: x' (k halves a row), then meta.
// ponytail: x prep per call, share it across q/k/v and gate/up when the prep shows in the step profile.
struct h2_x {
  scratch mem;
  view x;
  float2* meta;
  h2_x(const stream& s, const view& xs)
      : mem(uint64_t(xs.rows) * uint64_t(xs.cols) * 5 / 2, s), x(xs),
        meta(reinterpret_cast<float2*>(static_cast<__half*>(mem.p) + xs.rows * xs.cols)) {
    x.p = static_cast<uint8_t*>(mem.p);
    x.ld = xs.cols;
    x.row_bytes = 2 * xs.cols;
    const int64_t runs = xs.rows * (xs.cols / 16);
    const bool x16 = (uintptr_t(xs.p) | uint64_t(xs.ld) * 2) % 16 == 0;
    launch(mmvq_h2_prep_k, unsigned((runs + 255) / 256), 256, s.s, xs.as<__half>(), xs.ld, xs.cols, int(xs.rows), x16,
           x.as<__half>(), meta);
  }
};

// Cached at library load like ops.cu's attention switches: a changed environment cannot alter a captured graph's route.
// 0 keeps the f32 one-row decoder (arch::mmvq_h2); ctest verify_f32 pins it.
static const int decode_h2 = tc_switch("LARCH_DECODE_H2");

// Whether T's one-row kernel, and mmvq_rows_k, take the H2 route on s. Keyed on the running image, not the device:
// the H2 kernels trap on every other image (H2_IMAGE), and an sm_86/89 GPU may run another one.
template <int T>
static bool h2_route(const stream& s) {
  return MMVQ_H2<T> && arch::mmvq_h2(s.image_arch / 10, decode_h2);
}
#endif

// mmvq at R rows: mmvq_rows_k where MMVQ_ROUTE picks it within arch::mmvq_rows_up_to, else mmvq_k; one row and
// mmvq_rows_k on the H2 kernels where h2_route.
template <int T, int R>
static void mmvq_route(const stream& s, const view& w, const view& x, const view& b, const view& o, int sms) {
#if !defined(__HIP_PLATFORM_AMD__)  // HIP has no mmvq_rows_k
  if constexpr (R == 1) {
    if (h2_route<T>(s)) {
      const h2_x hx(s, x);
      return mmvq_launch<T, 1>(s, w, hx.x, b, o, sms, hx.meta);
    }
  } else if constexpr (T == LARCH_Q3_K || MMVQ_ROUTE[R - 2][T - LARCH_Q4_K]) {
    if (R <= arch::mmvq_rows_up_to(s.arch_code)) {
      if (h2_route<T>(s)) {
        const h2_x hx(s, x);
        return mmvq_rows_launch<T, R>(s, w, hx.x, b, o, mmvq_split<T>(s, w, sms), hx.meta);
      }
      mmvq_rows_launch<T, R>(s, w, x, b, o, mmvq_split<T>(s, w, sms));
      return;
    }
  }
#endif
  mmvq_launch<T, R>(s, w, x, b, o, sms);
}

larch_status mmvq(larch_stream* vs, const larch_tensor* w_quant, const larch_tensor* x_f16,
                   const larch_tensor* bias_or_null, const larch_tensor* out) {
  return run("mmvq", vs, [&](const stream& s) {
    const cpu::matmul_checked c = cpu::check_mmvq(sizer(s), w_quant, x_f16, bias_or_null, out);
    const view w = c.b_t, x = c.a, b = c.bias, o = c.c;
    aligned({w, x, b, o});
    if (!x.rows) return;
    if (w.cols > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "k past 2^31 - 1"};
    const int sms = s.compute_units;
    with_dtype(w.dtype, [&](auto t) {
      constexpr int T = decltype(t)::value;
      if constexpr (T != LARCH_F16 && T != LARCH_F32) {  // check_mmvq has refused f16 and f32 w
        with_rows(int(x.rows), [&](auto r) { mmvq_route<T, decltype(r)::value>(s, w, x, b, o, sms); });
      } else {
        throw failure{LARCH_ERR_UNSUPPORTED, "mmvq weight must be a quant dtype"};
      }
    });
  });
}

#if !defined(__HIP_PLATFORM_AMD__)
// mmvq_verify at R rows: one-row mmvq at one row, else mmvq_rows_k at the one-row split ks1.
template <int T, int R>
static void mmvq_verify_launch(const stream& s, const view& w, const view& x, const view& b, const view& o, int sms,
                               int ks1, const float2* meta) {
  if constexpr (R == 1)
    mmvq_launch<T, 1>(s, w, x, b, o, sms, meta);
  else
    mmvq_rows_launch<T, R>(s, w, x, b, o, ks1, meta);
}

larch_status mmvq_verify(larch_stream* vs, const larch_tensor* w_quant, const larch_tensor* x_f16,
                          const larch_tensor* bias_or_null, const larch_tensor* out) {
  return run("mmvq_verify", vs, [&](const stream& s) {
    const cpu::matmul_checked c = cpu::check_mmvq_verify(sizer(s), w_quant, x_f16, bias_or_null, out);
    const view w = c.b_t, b = c.bias;
    aligned({w, c.a, b, c.c});
    if (!c.a.rows) return;
    if (w.cols > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "k past 2^31 - 1"};
    const int sms = s.compute_units;
    with_dtype(w.dtype, [&](auto t) {
      constexpr int T = decltype(t)::value;
      if constexpr (T != LARCH_F16 && T != LARCH_F32) {  // check_mmvq_verify has refused f16 and f32 w
        const int ks1 = mmvq_split<T>(s, w, sms);
        std::optional<h2_x> hx;  // all rows' x', prepared once
        if (h2_route<T>(s)) hx.emplace(s, c.a);
        for (int64_t r0 = 0; r0 < c.a.rows; r0 += MMVQ_ROWS) {  // rows are independent: 17-32 as two launches
          view x = hx ? hx->x : view(c.a), o = c.c;  // view(): else ?: slices hx->x to its cpu::meta
          x.rows = o.rows = std::min<int64_t>(MMVQ_ROWS, c.a.rows - r0);
          x.p += r0 * x.row_bytes;
          o.p += r0 * o.row_bytes;
          const float2* meta = hx ? hx->meta + r0 * (w.cols / 16) : nullptr;
          with_rows(int(x.rows),
                    [&](auto r) { mmvq_verify_launch<T, decltype(r)::value>(s, w, x, b, o, sms, ks1, meta); });
        }
      } else {
        throw failure{LARCH_ERR_UNSUPPORTED, "mmvq weight must be a quant dtype"};
      }
    });
  });
}

// The grouped mmvq_id's pairs by expert (moe_place_k): expert e's at pairs[start[e], + count[e]), in ascending pair
// order, its M-row tiles from tile[e] (tile[experts]: all of them). Pair p = r slots + j reads x row r (+ j x_slot)
// and writes out row r (+ j out_slot).
struct moe_tiles {
  const int32_t *count, *start, *tile, *pairs;
  int experts, slots;
  int64_t x_slot, out_slot;
  // mmvq_id_mixed: the expert table, and whether this launch takes the cold experts' tiles (table < 0) or the hot
  // ones'; w is then that tensor, at the expert's slot.
  const int32_t* table;
  bool cold;
};

// A shared tile holds rounded weights, never a full dequantized matrix. Padding avoids shared-memory
// bank conflicts in the tensor-core loads; the existing decoder preserves dequant + GEMM rounding.
// Id (the grouped mmvq_id): a block's M rows are one tile of an expert's pairs (g), w that expert's n rows of w.
template <int T, int M, bool Split = false, bool Id = false>
__global__ void quant_gemm_k(const uint8_t* __restrict__ w, int64_t w_ld, int64_t n, int64_t k,
                            const __half* __restrict__ x, int64_t x_ld, int64_t m,
                            const float* __restrict__ bias, int64_t bias_ld, void* out, int64_t out_ld,
                            bool out_f32, bool w16, bool x16, int64_t n_tiles, int64_t k_chunk, moe_tiles g) {
  constexpr int N = 64, K = 128, LD = K + 8, THREADS = M == 16 ? 128 : 256;
  constexpr int NC = M < 32 ? 1 : M / 32, WN = N / (16 * NC);
  constexpr int U = K / (8 * unit<T>::P), UP = N * U / THREADS;
  extern __shared__ __align__(32) uint8_t shared[];
  __half* a = reinterpret_cast<__half*>(shared);
  __half* bt = a + M * LD;
  float* c = reinterpret_cast<float*>(shared);
  const int tid = int(threadIdx.x), warp = tid / 32;
  const int64_t m0 = int64_t(blockIdx.x) / n_tiles * M, n0 = int64_t(blockIdx.x) % n_tiles * N;
  int64_t rows = 0;  // Id: the tile's pairs, past M but in an expert's last tile, where the rest are padding
  int e = 0;
  const int32_t* pair = nullptr;
  if constexpr (Id) {
    const int t = int(m0 / M);
    if (t >= g.tile[g.experts]) return;  // past the tiles of this call: the grid is their upper bound
    int hi = g.experts - 1;  // e: the last expert whose tiles start at or before t, so a nonempty one
    while (e < hi) {
      const int mid = (e + hi + 1) / 2;
      if (g.tile[mid] <= t) e = mid;
      else hi = mid - 1;
    }
    int64_t slot = e;
    if (g.table) {
      const int32_t s = g.table[e];
      if ((s < 0) != g.cold) return;  // the other launch's tile
      slot = s < 0 ? -1 - int64_t(s) : s;
    }
    const int64_t first = int64_t(t - g.tile[e]) * M;
    rows = g.count[e] - first;
    pair = g.pairs + g.start[e] + first;
    w += slot * n * w_ld;
  }
  const int64_t k_begin = Split ? int64_t(blockIdx.y) * k_chunk : 0;
  const int64_t k_limit = k_begin + k_chunk;
  const int64_t k_end = Split && k_limit < k ? k_limit : k;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[NC];
#pragma unroll
  for (int j = 0; j < NC; ++j) nvcuda::wmma::fill_fragment(acc[j], 0.0f);
  unit<T> next[UP];
#pragma unroll
  for (int j = 0; j < UP; ++j) {
    const int i = tid + j * THREADS, r = i / U, u = int(k_begin / (8 * unit<T>::P)) + i % U;
    if (n0 + r < n && unit<T>::k(u) < k_end) next[j].load(w + (n0 + r) * w_ld, u, w16);
  }
  for (int64_t k0 = k_begin; k0 < k_end; k0 += K) {
#pragma unroll
    for (int j = 0; j < UP; ++j) {
      const int i = tid + j * THREADS, r = i / U, u = int(k0 / (8 * unit<T>::P)) + i % U;
      const bool valid = n0 + r < n && unit<T>::k(u) < k_end;
#pragma unroll
      for (int p = 0; p < unit<T>::P; ++p) {
        float values[8] = {};
        if (valid) {
          if constexpr (T == LARCH_Q4_K)
            next[j].template piece<true>(p, u, values);
          else
            next[j].piece(p, u, values);
        }
        const int col = unit<T>::k(u) - int(k0) + 8 * (p & 1) + unit<T>::HI * (p >> 1);
        store8(bt + r * LD + col, values);
      }
    }
    for (int i = tid; i < M * K / 8; i += THREADS) {
      const int r = i / (K / 8), col = i % (K / 8) * 8;
      uint4 values;
      if constexpr (Id)
        values = r < rows && k0 + col < k_end
                     ? ld16(reinterpret_cast<const uint8_t*>(x + pair[r] / g.slots * x_ld +
                                                             pair[r] % g.slots * g.x_slot + k0 + col),
                            x16)
                     : uint4{};
      else
        values = m0 + r < m && k0 + col < k_end
                     ? ld16(reinterpret_cast<const uint8_t*>(x + (m0 + r) * x_ld + k0 + col), x16)
                     : uint4{};
      *reinterpret_cast<uint4*>(a + r * LD + col) = values;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < UP; ++j) {
      const int i = tid + j * THREADS, r = i / U, u = int((k0 + K) / (8 * unit<T>::P)) + i % U;
      if (k0 + K < k_end && n0 + r < n && unit<T>::k(u) < k_end) next[j].load(w + (n0 + r) * w_ld, u, w16);
    }
#pragma unroll
    for (int col = 0; col < K; col += 16) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> av;
      nvcuda::wmma::load_matrix_sync(av, a + (warp / WN) * 16 * LD + col, LD);
#pragma unroll
      for (int j = 0; j < NC; ++j) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> bv;
        nvcuda::wmma::load_matrix_sync(bv, bt + ((warp % WN) * NC + j) * 16 * LD + col, LD);
        nvcuda::wmma::mma_sync(acc[j], av, bv, acc[j]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < NC; ++j)
    nvcuda::wmma::store_matrix_sync(c + (warp / WN) * 16 * N + ((warp % WN) * NC + j) * 16,
                                  acc[j], N, nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < M * N; i += THREADS) {
    const int64_t r = m0 + i / N, col = n0 + i % N;
    if constexpr (Id) {
      const int64_t t = i / N;
      if (t >= rows || col >= n) continue;
      const int64_t at = pair[t] / g.slots * out_ld + pair[t] % g.slots * g.out_slot + col;
      const float value = c[i] + (bias ? bias[(int64_t(e) * n + col) * bias_ld] : 0.0f);
      if (out_f32)
        store(static_cast<float*>(out) + at, value);
      else
        store(static_cast<__half*>(out) + at, value);
    } else if (r < m && col < n) {
      if constexpr (Split) {
        store(static_cast<float*>(out) + (int64_t(blockIdx.y) * m + r) * out_ld + col, c[i]);
      } else {
        const float value = c[i] + (bias ? bias[col * bias_ld] : 0.0f);
        if (out_f32)
          store(static_cast<float*>(out) + r * out_ld + col, value);
        else
          store(static_cast<__half*>(out) + r * out_ld + col, value);
      }
    }
  }
}

template <int T, int M>
static void quant_gemm_launch(const stream& s, const view& w, const view& x, const view& b, const view& o) {
  static_assert(M == 16 || M == 32 || M == 64);
  constexpr int threads = M == 16 ? 128 : 256;
  constexpr int bytes = (M + 64) * 136 * 2;
  const int64_t n_tiles = (w.rows + 63) / 64, m_tiles = (x.rows + M - 1) / M;
  if (n_tiles * m_tiles > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "direct quantized GEMM grid dimensions"};
  const bool w16 = uintptr_t(w.p) % 16 == 0, x16 = (uintptr_t(x.p) | uint64_t(x.ld) * 2) % 16 == 0;
  quant_gemm_k<T, M><<<unsigned(n_tiles * m_tiles), threads, bytes, s.s>>>(
      w.p, w.row_bytes, w.rows, w.cols, x.as<__half>(), x.ld, x.rows, b.as<float>(), b.ld, o.p, o.ld,
      o.dtype == LARCH_F32, w16, x16, n_tiles, 0, moe_tiles{});
  ck(cudaGetLastError(), "quant_gemm launch");
}

__global__ void quant_gemm_merge_k(const float* part, int64_t m, int64_t n, int splits, const float* bias,
                                  int64_t bias_ld, void* out, int64_t out_ld, bool out_f32) {
  const int64_t r = blockIdx.x;
  for (int64_t col = blockIdx.y * int64_t(blockDim.x) + threadIdx.x; col < n;
       col += int64_t(gridDim.y) * blockDim.x) {
    float value = part[r * n + col];
    for (int p = 1; p < splits; ++p) value += part[(int64_t(p) * m + r) * n + col];
    if (bias) value += bias[col * bias_ld];
    if (out_f32)
      store(static_cast<float*>(out) + r * out_ld + col, value);
    else
      store(static_cast<__half*>(out) + r * out_ld + col, value);
  }
}

template <int T>
static bool quant_gemm_split(const stream& s, const view& w, const view& x, const view& b, const view& o) {
  const int64_t n_tiles = (w.rows + 63) / 64;
  // ponytail: split-K measured on sm_120 only; drop the arch check when the 3050/4060 `larch bench`
  // fixed rows at 8-16 users gain with it dropped.
  if (s.arch_code != 120 || x.rows < 8 || x.rows > 16 || n_tiles >= s.compute_units || w.cols < 1024)
    return false;
  constexpr int bytes = 80 * 136 * 2;
  int per_sm = 0;
  ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, quant_gemm_k<T, 16, true>, 128, bytes),
     "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  for (const int splits : {4, 2}) {
    const int64_t k_chunk = ((w.cols / 256 + splits - 1) / splits) * 256;
    const uint64_t size = uint64_t(x.rows) * uint64_t(w.rows) * splits * 4;
    if (w.cols - (splits - 1) * k_chunk < 512 || n_tiles * splits > int64_t(s.compute_units) * per_sm ||
        size > (uint64_t(1) << 20)) continue;
    const scratch ws(size, s);
    const bool w16 = uintptr_t(w.p) % 16 == 0, x16 = (uintptr_t(x.p) | uint64_t(x.ld) * 2) % 16 == 0;
    quant_gemm_k<T, 16, true><<<dim3(unsigned(n_tiles), splits), 128, bytes, s.s>>>(
        w.p, w.row_bytes, w.rows, w.cols, x.as<__half>(), x.ld, x.rows, nullptr, 0, ws.p, w.rows,
        true, w16, x16, n_tiles, k_chunk, moe_tiles{});
    ck(cudaGetLastError(), "quant_gemm split launch");
    launch(quant_gemm_merge_k, rows_grid(x.rows, w.rows, 256), 256, s.s, static_cast<const float*>(ws.p),
           x.rows, w.rows, splits, b.as<float>(), b.ld, o.p, o.ld, o.dtype == LARCH_F32);
    return true;
  }
  return false;
}

static void quant_gemm_panels(const stream& s, const view& w, const view& x, const view& b, const view& o);

// Resource discovery and any opt-in happen at stream creation, never inside graph capture.
template<int T, int J, bool StreamK = false> static cudaError_t quant_i8_tile_init(stream& s, int& optin_limit) {
  cudaFuncAttributes attributes{};
  cudaError_t e;
  if constexpr (StreamK) e = cudaFuncGetAttributes(&attributes, mmq_streamk::gemm<T, J>);
  else e = cudaFuncGetAttributes(&attributes, mmq_q4::gemm<T, J>);
  if (e != cudaSuccess) return e;
  // Advertise only kernels that can launch this block without compiled local-memory traffic.
  if (attributes.maxThreadsPerBlock < 256 || attributes.localSizeBytes != 0) return cudaSuccess;
  constexpr int dynamic = mmq_q4::shared_bytes<J>;
  if (size_t(dynamic) + attributes.sharedSizeBytes > size_t(s.shared_mem_per_block)) {
    if (optin_limit < 0) {
      e = cudaDeviceGetAttribute(&optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, s.ordinal);
      if (e != cudaSuccess) return e;
    }
    if (size_t(dynamic) + attributes.sharedSizeBytes > size_t(optin_limit)) return cudaSuccess;
    if constexpr (StreamK)
      e = cudaFuncSetAttribute(mmq_streamk::gemm<T, J>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic);
    else e = cudaFuncSetAttribute(mmq_q4::gemm<T, J>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic);
    if (e != cudaSuccess) return e;
  }
  int resident = 0;
  if constexpr (StreamK)
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, mmq_streamk::gemm<T, J>, 256, dynamic);
  else e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, mmq_q4::gemm<T, J>, 256, dynamic);
  if (e != cudaSuccess) return e;
  // Shared-memory opt-in alone does not prove that the register/block footprint fits.
  if (resident < 1) return cudaSuccess;
  uint32_t& tiles = s.quant_i8_tiles[T - LARCH_Q4_K];
  tiles |= uint32_t(1) << (J / 8 - 1 + (StreamK ? 16 : 0));
  if constexpr (StreamK)
    if (resident >= 2) tiles |= uint32_t(1) << (J / 8 - 1 + 20);
  return cudaSuccess;
}
template<int T> static cudaError_t quant_i8_format_init(stream& s, int& optin_limit) {
  cudaError_t e = quant_i8_tile_init<T, 8>(s, optin_limit);
  if (!e) e = quant_i8_tile_init<T, 16>(s, optin_limit);
  if (!e) e = quant_i8_tile_init<T, 32>(s, optin_limit);
  if (!e) e = quant_i8_tile_init<T, 64>(s, optin_limit);
  if (!e) e = quant_i8_tile_init<T, 112>(s, optin_limit);
  if (!e) e = quant_i8_tile_init<T, 128>(s, optin_limit);
  if (!e) e = quant_i8_tile_init<T, 8, true>(s, optin_limit);
  if (!e) e = quant_i8_tile_init<T, 16, true>(s, optin_limit);
  return e;
}
// The INT8 route is opt-in, so its probe never fails a stream: any error means no INT8 tiles, is
// cleared, and callers keep their fallback. CUDA only, like the rest of this section.
void quant_i8_init(stream& s) {
  for (uint32_t& tiles : s.quant_i8_tiles) tiles = 0;
  if (s.image_arch < 800) return;
  int optin_limit = -1;
  cudaError_t e = cudaError_t(vv_fault_quant_i8_probe);
  if (!e) e = quant_i8_format_init<LARCH_Q4_K>(s, optin_limit);
  if (!e) e = quant_i8_format_init<LARCH_Q6_K>(s, optin_limit);
  if (!e) e = quant_i8_format_init<LARCH_Q8_0>(s, optin_limit);
  if (!e) e = quant_i8_format_init<LARCH_Q5_K>(s, optin_limit);
  if (!e) e = quant_i8_format_init<LARCH_Q4_0>(s, optin_limit);
  if (!e) e = quant_i8_format_init<LARCH_Q5_0>(s, optin_limit);
  if (e) {
    for (uint32_t& tiles : s.quant_i8_tiles) tiles = 0;
    (void)cudaGetLastError();
  }
}

// This explicit operation quantizes activations, unlike quant_gemm's FP16-rounded weights contract.
static larch_status quant_gemm_i8_impl(const char* name, larch_stream* vs, const larch_tensor* w_quant,
                                      const larch_tensor* x_f16, const larch_tensor* bias_or_null,
                                      const larch_tensor* out, bool split_required) {
  return run(name, vs, [&](const stream& s) {
    const cpu::matmul_checked checked = cpu::check_quant_gemm(sizer(s), w_quant, x_f16, bias_or_null, out);
    const view w = checked.b_t, x = checked.a, b = checked.bias, o = checked.c;
    aligned({w, x, b, o});
    // The pack's K128 chunks: every K quant, and the 32-weight formats with whole chunks.
    const uint32_t tiles = w.dtype >= LARCH_Q4_K && w.dtype <= LARCH_Q5_0 ? s.quant_i8_tiles[w.dtype - LARCH_Q4_K] : 0;
    if (!tiles || w.cols % 128)
      throw failure{LARCH_ERR_UNSUPPORTED, "INT8 quantized GEMM weight type"};
    if (!x.rows) return;
    if (x.rows > INT_MAX || w.rows > INT_MAX || w.cols > INT_MAX)
      throw failure{LARCH_ERR_UNSUPPORTED, "INT8 quantized GEMM capability or dimensions"};
    // The first supported tile attaining the fewest query tiles, unless sm_120's Q6_K rule below takes another past 64
    // rows; the chosen tile's count bounds padding and capture resources.
    int tile = 0;
    int64_t fewest = INT64_MAX;
    for (const int j : {8, 16, 32, 64, 112, 128}) {
      if (!(tiles & (uint32_t(1) << (j / 8 - 1)))) continue;
      const int64_t count = (x.rows + j - 1) / j;
      if (count < fewest) { tile = j; fewest = count; }
    }
    if (!tile) throw failure{LARCH_ERR_UNSUPPORTED, "INT8 quantized GEMM tile resources"};
    // ponytail: on sm_120, Q6_K past 64 rows takes J64 or J112 by fewer waves (one CTA per SM) times per-CTA
    // time, J64:J112 = 5:7; J128's unpaired schedule is 13 (one wave: m1 down 253/357/653 us, m5 328/465/860),
    // so it never wins. Other archs keep the fewest query tiles until quant_gemm --bench-i8 q6_k measures them;
    // re-measure there too when a tile's schedule or residency changes.
    if (s.arch_code == 120 && w.dtype == LARCH_Q6_K && x.rows > 64 && (tiles >> (64 / 8 - 1) & 1) &&
        (tiles >> (112 / 8 - 1) & 1)) {
      const int64_t units = std::max(s.compute_units, 1);
      const auto waves = [&](int64_t j) { return ((w.rows + 127) / 128 * ((x.rows + j - 1) / j) + units - 1) / units; };
      tile = 5 * waves(64) <= 7 * waves(112) ? 64 : 112;
      fewest = (x.rows + tile - 1) / tile;
    }
    // Stream-K takes 17-32 rows as two J16 query tiles: its two resident CTAs fit J16, not J32.
    if (split_required && x.rows > 16) tile = 16, fewest = (x.rows + 15) / 16;
    constexpr uint64_t budget = uint64_t(32) << 20;
    const uint64_t padded_rows = uint64_t(fewest) * tile;
    const uint64_t groups = uint64_t(w.cols / 128);
    constexpr uint64_t packed_bytes = mmq_q4::activation_stride * 4;
    if (padded_rows > budget / packed_bytes / groups)
      throw failure{LARCH_ERR_UNSUPPORTED, "INT8 quantized GEMM activation scratch budget"};
    const int64_t n_tiles = (w.rows + 127) / 128;
    const int64_t grid = n_tiles * ((x.rows + tile - 1) / tile);
    const uint64_t pack_grid = padded_rows * groups;
    if (grid > INT_MAX || pack_grid > INT_MAX)
      throw failure{LARCH_ERR_UNSUPPORTED, "INT8 quantized GEMM grid dimensions"};
    const uint64_t pack_size = padded_rows * groups * packed_bytes;
    const int64_t k_blocks = (w.cols + 255) / 256, work = grid * k_blocks;
    int stream_grid = 0;
    uint64_t partial_size = 0;
    if (split_required) {
      // Two resident CTAs hide the load latency; every CTA owns at least one K256 step. Whether the pack and
      // merge pay off for a shape is the caller's measured routing.
      const int64_t desired_grid = std::min(int64_t(s.compute_units) * 2, work);
      if (x.rows < 8 || x.rows > 32 || tile > 16 || s.compute_units <= 0 ||
          n_tiles >= s.compute_units ||
          !(tiles & (uint32_t(1) << (tile / 8 - 1 + 16))) ||
          !(tiles & (uint32_t(1) << (tile / 8 - 1 + 20))) ||
          work > INT_MAX)
        throw failure{LARCH_ERR_UNSUPPORTED, "INT8 split GEMM resources"};
      stream_grid = int(desired_grid);
      partial_size = uint64_t(grid + stream_grid) * tile * 128 * sizeof(float);
      if (partial_size > budget - pack_size)
        throw failure{LARCH_ERR_UNSUPPORTED, "INT8 split GEMM combined scratch budget"};
    }
    const scratch memory(pack_size + partial_size, s);
    auto* packed = static_cast<uint32_t*>(memory.p);
    auto* head = stream_grid ? reinterpret_cast<float*>(static_cast<uint8_t*>(memory.p) + pack_size) : nullptr;
    auto* tail = stream_grid ? head + grid * tile * 128 : nullptr;
    mmq_q4::pack<<<unsigned(pack_grid), 128, 0, s.s>>>(x.as<__half>(), x.ld, x.rows, int64_t(padded_rows), w.cols, packed);
    ck(cudaGetLastError(), "INT8 activation packing launch");
    const auto issue = [&](auto format, auto size) {
      constexpr int T = decltype(format)::value, J = decltype(size)::value;
      const bool aligned4 = (uintptr_t(w.p) | uint64_t(w.row_bytes)) % 4 == 0;
      if constexpr (J <= 16) {
        if (stream_grid) {
          mmq_streamk::gemm<T, J><<<stream_grid, 256, mmq_q4::shared_bytes<J>, s.s>>>(
              w.p, w.row_bytes, w.rows, w.cols, packed, int64_t(padded_rows), aligned4,
              n_tiles, grid, head, tail);
          ck(cudaGetLastError(), "INT8 StreamK launch");
          mmq_streamk::merge<J><<<unsigned(grid), 256, 0, s.s>>>(head, tail, grid, n_tiles, k_blocks,
              stream_grid, x.rows, w.rows, b.as<float>(), b.ld, o.p, o.ld, o.dtype == LARCH_F32);
          return;
        }
      }
      mmq_q4::gemm<T, J><<<unsigned(grid), 256, mmq_q4::shared_bytes<J>, s.s>>>(
          w.p, w.row_bytes, w.rows, w.cols, packed, x.rows, int64_t(padded_rows), aligned4,
          b.as<float>(), b.ld, o.p, o.ld, o.dtype == LARCH_F32, n_tiles);
    };
    const auto sized = [&](auto format) {
      switch (tile) {
        case 8: return issue(format, std::integral_constant<int, 8>());
        case 16: return issue(format, std::integral_constant<int, 16>());
        case 32: return issue(format, std::integral_constant<int, 32>());
        case 64: return issue(format, std::integral_constant<int, 64>());
        case 112: return issue(format, std::integral_constant<int, 112>());
        case 128: return issue(format, std::integral_constant<int, 128>());
      }
    };
    switch (w.dtype) {
      case LARCH_Q4_K: sized(std::integral_constant<int, LARCH_Q4_K>()); break;
      case LARCH_Q6_K: sized(std::integral_constant<int, LARCH_Q6_K>()); break;
      case LARCH_Q8_0: sized(std::integral_constant<int, LARCH_Q8_0>()); break;
      case LARCH_Q5_K: sized(std::integral_constant<int, LARCH_Q5_K>()); break;
      case LARCH_Q4_0: sized(std::integral_constant<int, LARCH_Q4_0>()); break;
      case LARCH_Q5_0: sized(std::integral_constant<int, LARCH_Q5_0>()); break;
      default: throw failure{LARCH_ERR_UNSUPPORTED, "INT8 quantized GEMM weight type"};
    }
    ck(cudaGetLastError(), "INT8 quantized GEMM launch");
  });
}

larch_status quant_gemm_i8(larch_stream* s, const larch_tensor* w, const larch_tensor* x,
                            const larch_tensor* bias, const larch_tensor* out) {
  return quant_gemm_i8_impl("quant_gemm_i8", s, w, x, bias, out, false);
}

larch_status quant_gemm_i8_split(larch_stream* s, const larch_tensor* w, const larch_tensor* x,
                                  const larch_tensor* bias, const larch_tensor* out) {
  return quant_gemm_i8_impl("quant_gemm_i8_split", s, w, x, bias, out, true);
}

/// Direct quantized projection with f16 tile dequantization and f32 accumulation on CUDA tensor cores.
/// CUDA only: HIP registers no slot, so dispatch.cpp answers it.
larch_status quant_gemm(larch_stream* vs, const larch_tensor* w_quant, const larch_tensor* x_f16,
                         const larch_tensor* bias_or_null, const larch_tensor* out) {
  return run("quant_gemm", vs, [&](const stream& s) {
    const cpu::matmul_checked checked = cpu::check_quant_gemm(sizer(s), w_quant, x_f16, bias_or_null, out);
    const view w = checked.b_t, x = checked.a, b = checked.bias, o = checked.c;
    aligned({w, x, b, o});
    if (w.dtype != LARCH_Q4_K && w.dtype != LARCH_Q6_K && w.dtype != LARCH_Q8_0)
      throw failure{LARCH_ERR_UNSUPPORTED, "direct quantized GEMM weight type"};
    if (!x.rows) return;
    if (x.rows > INT_MAX || w.rows > INT_MAX || w.cols > INT_MAX || s.arch_code < 70)
      throw failure{LARCH_ERR_UNSUPPORTED, "direct quantized GEMM launch dimensions"};
    if (x.rows > 64) return quant_gemm_panels(s, w, x, b, o);
    const auto issue = [&](auto dtype) {
      constexpr int T = decltype(dtype)::value;
      if constexpr (T == LARCH_Q4_K || T == LARCH_Q6_K)
        if (quant_gemm_split<T>(s, w, x, b, o)) return;
      // The smaller Q4 tile wins for wide outputs that fit a single resident CTA wave.
      if constexpr (T == LARCH_Q4_K) {
        if (x.rows <= 16 && w.rows >= 8192) {
          int per_sm = 0;
          ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, quant_gemm_k<T, 16>, 128, 80 * 136 * 2),
             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
          if ((w.rows + 63) / 64 <= int64_t(s.compute_units) * per_sm)
            return quant_gemm_launch<T, 16>(s, w, x, b, o);
        }
      }
      if (x.rows <= 32) return quant_gemm_launch<T, 32>(s, w, x, b, o);
      return quant_gemm_launch<T, 64>(s, w, x, b, o);
    };
    switch (w.dtype) {
      case LARCH_Q4_K: return issue(std::integral_constant<int, LARCH_Q4_K>());
      case LARCH_Q6_K: return issue(std::integral_constant<int, LARCH_Q6_K>());
      case LARCH_Q8_0: return issue(std::integral_constant<int, LARCH_Q8_0>());
      default: throw failure{LARCH_ERR_UNSUPPORTED, "direct quantized GEMM weight type"};
    }
  });
}

// The grouped mmvq_id's tile: quant_gemm_k's 16 rows, fixed so a pair's sums never depend on how many pairs share its
// expert. ponytail: 16 fits pp 512's ~16 pairs an expert (8 of 256); a 32-row tile, which decodes each weight half as
// often, when prefill chunks past 1024 rows show the grouped kernel decode-bound in the pp profile.
constexpr int MOE_TILE = 16;
// The weighted down projection's f32 partials per launch at most, in blocks of rows: quant_gemm_panels' 16 MiB.
constexpr uint64_t MOE_PARTIAL_BYTES = uint64_t(16) << 20;

// Lanes whose pair p (of `pairs`, pair r slots + j at ids[r ids_ld + j]) chose expert e.
__device__ __forceinline__ unsigned moe_hits(const int32_t* ids, int64_t ids_ld, int slots, int pairs, int e, int p) {
  return __ballot_sync(VV_FULL_MASK, p < pairs && ids[p / slots * ids_ld + p % slots] == e);
}

// count[e]: the pairs that chose expert e, one warp an expert. An id outside [0, experts) matches none.
__global__ void moe_count_k(const int32_t* ids, int64_t ids_ld, int slots, int pairs, int experts, int32_t* count) {
  const int lane = int(threadIdx.x) % warpSize, e = int(blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize);
  if (e >= experts) return;
  int c = 0;
  for (int p0 = 0; p0 < pairs; p0 += warpSize) c += __popc(moe_hits(ids, ids_ld, slots, pairs, e, p0 + lane));
  if (!lane) count[e] = c;
}

// moe_tiles' start, tile and pairs from count, one warp an expert: its pairs in ascending order, so the tiles, and
// which tile row a pair takes, are the same on every run.
__global__ void moe_place_k(const int32_t* ids, int64_t ids_ld, int slots, int pairs, int experts,
                            const int32_t* count, int32_t* start, int32_t* tile, int32_t* list) {
  const int lane = int(threadIdx.x) % warpSize, e = int(blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize);
  if (e >= experts) return;
  int before = 0, tiles = 0;
  for (int i = lane; i < e; i += warpSize) {
    before += count[i];
    tiles += (count[i] + MOE_TILE - 1) / MOE_TILE;
  }
  for (int o = warpSize / 2; o > 0; o /= 2) {
    before += __shfl_xor_sync(VV_FULL_MASK, before, o);
    tiles += __shfl_xor_sync(VV_FULL_MASK, tiles, o);
  }
  if (!lane) {
    start[e] = before;
    tile[e] = tiles;
    if (e == experts - 1) tile[experts] = tiles + (count[e] + MOE_TILE - 1) / MOE_TILE;
  }
  for (int p0 = 0; p0 < pairs; p0 += warpSize) {
    const unsigned hit = moe_hits(ids, ids_ld, slots, pairs, e, p0 + lane);
    if (hit >> lane & 1) list[before + __popc(hit & ((1u << lane) - 1))] = p0 + lane;
    before += __popc(hit);
  }
}

// out[r] = the sum over j ascending of wt[r, j] y[r slots + j], in f32 as mmvq_id_k sums the slots, rounded once.
template <class O>
__global__ void moe_combine_k(const float* y, int64_t n, int slots, const float* wt, int64_t wt_ld, O* out,
                              int64_t out_ld) {
  const int64_t r = blockIdx.x;
  for (int64_t col = blockIdx.y * int64_t(blockDim.x) + threadIdx.x; col < n; col += int64_t(gridDim.y) * blockDim.x) {
    float sum = 0;
    for (int j = 0; j < slots; ++j) sum += wt[r * wt_ld + j] * y[(r * slots + j) * n + col];
    store(out + r * out_ld + col, sum);
  }
}

// mmvq_id from MOE_GROUPED_ROWS rows: the (row, slot) pairs grouped by expert on the device (moe_count_k,
// moe_place_k), then quant_gemm_k over each expert's tiles, which decodes a weight tile once for its 16 pairs where
// mmvq_id_k decodes it per pair; weighted, the pairs' f32 results (bias added) to moe_combine_k, in blocks of rows
// within MOE_PARTIAL_BYTES. The rounding it changes: each weight is rounded to f16 (dequant's value) and the k sum is
// the tensor cores' f32 order, not mmvq_id_k's lane order. A pair's bytes depend only on its own x row, w and the
// tile size, never on the other rows, as below the threshold. No host sync and no allocation but the scratch, so a
// graph may capture it. False (nothing enqueued) where the launch would not fit the grid. mmvq_id_mixed (table set):
// the tiles of hot experts run from w at T, then those of cold ones from wc at TC, each launch skipping the other's
// tiles, so each expert's tiles are the ones a single tensor would give it, and no tile reads another's bytes.
template <int T, int TC>
static bool mmvq_id_grouped(const stream& s, const view& w, const view& wc, const view& table, const view& x,
                            const view& ids, const view& b, const view& wt, const view& o) {
  const int64_t slots = ids.cols, n = o.shape[o.ndim - 1], k = w.cols,
                experts = table.p ? table.rows : w.rows / n;
  const bool weighted = wt.p != nullptr;
  const int64_t block = weighted ? std::clamp<int64_t>(int64_t(MOE_PARTIAL_BYTES / uint64_t(4 * slots * n)), 1, x.rows)
                                 : x.rows;
  const int64_t pairs = block * slots, tiles = (pairs + MOE_TILE - 1) / MOE_TILE + std::min(experts, pairs),
                n_tiles = (n + 63) / 64;
  if (n_tiles * tiles > INT_MAX) return false;
  const uint64_t lists = (uint64_t(3 * experts + 1 + pairs) * 4 + 15) / 16 * 16;
  const scratch mem(lists + (weighted ? uint64_t(pairs * n) * 4 : 0), s);
  int32_t *const count = static_cast<int32_t*>(mem.p), *const start = count + experts, *const tile = start + experts,
                *const list = tile + experts + 1;
  const moe_tiles g{count, start, tile, list, int(experts), int(slots), x.ndim == 3 ? k : 0, n, table.as<int32_t>(),
                    false};
  float* const y = reinterpret_cast<float*>(static_cast<uint8_t*>(mem.p) + lists);
  const bool w16 = uintptr_t(w.p) % 16 == 0, x16 = (uintptr_t(x.p) | uint64_t(x.ld) * 2) % 16 == 0;
  const int per_block = ID_THREADS / s.warp;
  const unsigned expert_blocks = unsigned((experts + per_block - 1) / per_block);
  const bool f32 = o.dtype == LARCH_F32;
  for (int64_t r0 = 0; r0 < x.rows; r0 += block) {
    const int64_t rows = std::min(block, x.rows - r0);
    const int p = int(rows * slots);
    const int32_t* id = ids.as<int32_t>() + r0 * ids.ld;
    launch(moe_count_k, expert_blocks, ID_THREADS, s.s, id, ids.ld, int(slots), p, int(experts), count);
    launch(moe_place_k, expert_blocks, ID_THREADS, s.s, id, ids.ld, int(slots), p, int(experts), count, start, tile,
           list);
    uint8_t* const out = o.p + r0 * o.ld * (f32 ? 4 : 2);
    quant_gemm_k<T, MOE_TILE, false, true><<<unsigned(n_tiles * tiles), 128, (MOE_TILE + 64) * 136 * 2, s.s>>>(
        w.p, w.row_bytes, n, k, x.as<__half>() + r0 * x.ld, x.ld, p, b.as<float>(), b.ld,
        weighted ? static_cast<void*>(y) : out, weighted ? slots * n : o.ld, weighted || f32, w16, x16, n_tiles, 0,
        g);
    if (table.p) {
      moe_tiles gc = g;
      gc.cold = true;
      quant_gemm_k<TC, MOE_TILE, false, true><<<unsigned(n_tiles * tiles), 128, (MOE_TILE + 64) * 136 * 2, s.s>>>(
          wc.p, wc.row_bytes, n, k, x.as<__half>() + r0 * x.ld, x.ld, p, nullptr, 0,
          weighted ? static_cast<void*>(y) : out, weighted ? slots * n : o.ld, weighted || f32,
          uintptr_t(wc.p) % 16 == 0, x16, n_tiles, 0, gc);
    }
    ck(cudaGetLastError(), "mmvq_id grouped launch");
    if (!weighted) continue;
    const dim3 grid = rows_grid(rows, n, 256);
    const float* const wr = wt.as<float>() + r0 * wt.ld;
    if (f32)
      launch(moe_combine_k<float>, grid, 256, s.s, y, n, int(slots), wr, wt.ld, reinterpret_cast<float*>(out), o.ld);
    else
      launch(moe_combine_k<__half>, grid, 256, s.s, y, n, int(slots), wr, wt.ld, reinterpret_cast<__half*>(out),
             o.ld);
  }
  return true;
}
#endif

// c[r, j] = src[r, j] + beta c[r, j] + bias[j]; c is read only when beta != 0, and src may be c.
template <class O>
__global__ void bias_k(const float* src, int64_t src_ld, O* c, int64_t c_ld, int64_t n, float beta, const float* bias,
                       int64_t bias_ld) {
  const int64_t r = blockIdx.x;
  for (int64_t j = blockIdx.y * int64_t(blockDim.x) + threadIdx.x; j < n; j += int64_t(gridDim.y) * blockDim.x) {
    float v = src[r * src_ld + j];
    if (beta != 0.0f) v += beta * float(c[r * c_ld + j]);
    store(c + r * c_ld + j, v + bias[j * bias_ld]);
  }
}

#if !defined(__HIP_PLATFORM_AMD__)
// Output columns a panel decodes at once, in 128s, at most 2048, each costing its f16 weights and,
// when an f16 output takes a bias, its f32 result rows; 0 when 128 columns exceed 16 MiB.
constexpr int64_t panel_columns(int64_t n, int64_t k, int64_t rows, bool temporary) {
  const int64_t fit = (int64_t(16) << 20) / (2 * k + (temporary ? 4 * rows : 0)) / 128 * 128;
  return fit ? std::min(n, std::min(int64_t(2048), fit)) : 0;
}
// The panel's scratch: its f16 weights, plus the f32 result an f16 output with bias needs.
constexpr int64_t panel_bytes(int64_t n, int64_t k, int64_t rows, bool temporary) {
  const int64_t columns = panel_columns(n, k, rows, temporary);
  return columns * k * 2 + (temporary ? rows * columns * 4 : 0);
}
// The vocabulary heads (f32, no bias) and a biased f16 prefill projection at the largest chunk.
static_assert(panel_bytes(152064, 3584, 8192, false) <= 16 << 20 && panel_bytes(128256, 4096, 8192, false) <= 16 << 20 &&
              panel_bytes(151936, 896, 8192, false) <= 16 << 20 && panel_bytes(3584, 3584, 8192, true) <= 16 << 20);

// Reuse the existing decoder and BLAS for larger batches within 16 MiB of scratch (panel_bytes).
// The same scratch is reused in stream order, including during graph capture.
static void quant_gemm_panels(const stream& s, const view& w, const view& x, const view& b, const view& o) {
  if (x.ld > INT_MAX || o.ld > INT_MAX)
    throw failure{LARCH_ERR_UNSUPPORTED, "quantized GEMM panel stride past cuBLAS's 32-bit int"};
  const bool f32 = o.dtype == LARCH_F32, temporary = b.p && !f32;
  const int64_t columns = panel_columns(w.rows, w.cols, x.rows, temporary);
  if (!columns) throw failure{LARCH_ERR_UNSUPPORTED, "quantized GEMM minimum panel exceeds scratch budget"};
  const uint64_t weight_bytes = uint64_t(columns) * uint64_t(w.cols) * 2;
  const scratch memory(uint64_t(panel_bytes(w.rows, w.cols, x.rows, temporary)), s);
  void* result = static_cast<uint8_t*>(memory.p) + weight_bytes;
  view decoded;
  decoded.p = static_cast<uint8_t*>(memory.p);
  decoded.dtype = LARCH_F16;
  decoded.cols = decoded.ld = w.cols;
  decoded.row_bytes = w.cols * 2;
  for (int64_t at = 0; at < w.rows; at += columns) {
    const int64_t n = std::min(columns, w.rows - at);
    view source = w;
    source.p += at * w.row_bytes;
    source.rows = decoded.rows = n;
    decode_rows<__half>(s, source, view(), decoded);
    void* output = o.p + at * (f32 ? 4 : 2);
    ck(vv_blas_gemm_f16(s.blas, int(x.rows), int(n), int(w.cols), 1.0f, x.as<__half>(), int(x.ld),
                       decoded.as<__half>(), int(decoded.ld), 0.0f, temporary ? result : output,
                       temporary || f32, temporary ? int(n) : int(o.ld)), "quant_gemm panel cublasGemmEx");
    if (b.p) {
      const dim3 grid = rows_grid(x.rows, n, 256);
      if (f32)
        launch(bias_k<float>, grid, 256, s.s, static_cast<float*>(output), o.ld, static_cast<float*>(output),
               o.ld, n, 0.0f, b.as<float>() + at * b.ld, b.ld);
      else
        launch(bias_k<__half>, grid, 256, s.s, static_cast<float*>(result), n, static_cast<__half*>(output),
               o.ld, n, 0.0f, b.as<float>() + at * b.ld, b.ld);
    }
  }
}
#endif

larch_status gemm(larch_stream* vs, const larch_tensor* a_f16, const larch_tensor* b_f16_t,
                   const larch_tensor* bias_or_null, const larch_tensor* c, float alpha, float beta) {
  return run("gemm", vs, [&](const stream& s) {
    const cpu::matmul_checked checked = cpu::check_gemm(sizer(s), a_f16, b_f16_t, bias_or_null, c);
    const view a = checked.a, bt = checked.b_t, bias = checked.bias, cv = checked.c;
    aligned({a, bt, bias, cv});
    for (const int64_t d : {a.rows, bt.rows, a.cols, a.ld, bt.ld, cv.ld})
      if (d > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "a dim or row stride past cuBLAS's 32-bit int"};
    if (!a.rows) return;
    const int m = int(a.rows), n = int(bt.rows), k = int(a.cols);
    const bool f32 = cv.dtype == LARCH_F32;
    const auto mm = [&](float b, void* out, bool out_f32, int ld) {
      ck(vv_blas_gemm_f16(s.blas, m, n, k, alpha, a.as<__half>(), int(a.ld), bt.as<__half>(), int(bt.ld), b, out,
                          out_f32, ld),
         "cublasGemmEx");
    };
    if (!bias.p) return mm(beta, cv.p, f32, int(cv.ld));
    // The bias joins in f32 and f16 c is rounded once, as the cuBLASLt bias epilogue does.
    // ponytail: a second pass over c (and f32 scratch for f16 c); cuBLASLt's bias epilogue, one pass, when
    // biased prefill projections (Qwen2) show in the bench. It needs cublasLt and hipblasLt linked (A1P's
    // CMake and larch-sys).
    const dim3 grid = rows_grid(m, n, 256);
    if (f32) {
      mm(beta, cv.p, true, int(cv.ld));
      launch(bias_k<float>, grid, 256, s.s, cv.as<float>(), cv.ld, cv.as<float>(), cv.ld, bt.rows, 0.0f,
             bias.as<float>(), bias.ld);
    } else {
      const scratch ws(uint64_t(m) * uint64_t(n) * 4, s);
      mm(0.0f, ws.p, true, n);
      launch(bias_k<__half>, grid, 256, s.s, static_cast<const float*>(ws.p), int64_t(n), cv.as<__half>(), cv.ld,
             bt.rows, beta, bias.as<float>(), bias.ld);
    }
  });
}


#if !defined(__HIP_PLATFORM_AMD__)
// The fused entries (larch.h, ABI 4) over the decoders above. Each kernel gives its sequence's bytes on this device by
// running each step's own per-element arithmetic: mmvq_id's id_dot, mmvq's mmvq_col at the split larch_mmvq's route
// sums a row in, and the elementwise ops' expressions, rounding to f16 where the sequence stores f16. None syncs the
// host, so each can be captured in a graph.

// One column of larch_mmvq for one row x at `splits` (sum of a row on every non-H2 route; see mmvq_rows_k): lane l of
// split s sums units l + 32 s + 32 splits j in j order, pieces in order, two FMAs a weight pair on fast<T>'s values,
// then each split's warp sum (the butterfly's), the splits added in order. Every lane holds the result.
// The splits' units are interleaved here, and two units of each in turn, so their loads are in flight together; and
// each unit is decoded once for up to FR rows of x (x_ld apart; out[r] for r < rows), each row's sum in its own order.
// xs: x is in shared memory, 16-byte aligned, where ld16's read-only (__ldg) load is not allowed.
constexpr int FR = 4;
template <int T>
__device__ __forceinline__ void mmvq_col(const uint8_t* w, int units, int splits, bool w16, const __half* x,
                                         int64_t x_ld, int rows, bool x16, int lane, bool xs, float (&out)[FR]) {
  constexpr int MAX = MMVQ_THREADS / 32;  // mmvq_split's ceiling
  float acc[FR][MAX] = {};
#pragma unroll 2
  for (int u0 = lane; u0 < units; u0 += 32 * splits) {
#pragma unroll
    for (int sp = 0; sp < MAX; ++sp) {
      const int u = u0 + 32 * sp;
      if (sp >= splits || u >= units) continue;
      unit<T> un;
      un.load(w, u, (T == LARCH_Q4_K || T == LARCH_Q5_K) && w16);
      fast<T> f;
      f.set(un, u);
#pragma unroll
      for (int p = 0; p < unit<T>::P; ++p) {
        float wv[8];
        f.quad(p, 0, wv);
        f.quad(p, 1, wv + 4);
#pragma unroll
        for (int r = 0; r < FR; ++r) {
          if (r >= rows) break;
          const __half* xp = x + r * x_ld + unit<T>::k(u) + 8 * (p & 1) + unit<T>::HI * (p >> 1);
          const uint4 xv = xs ? *reinterpret_cast<const uint4*>(xp) : ld16(reinterpret_cast<const uint8_t*>(xp), x16);
#pragma unroll
          for (int e = 0; e < 8; e += 2) {
            const float2 xf = h2f(word(xv, e / 2));
            acc[r][sp] = fmaf(wv[e + 1], xf.y, fmaf(wv[e], xf.x, acc[r][sp]));
          }
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < FR; ++r) {
    if (r >= rows) break;
    out[r] = vv_warp_reduce_sum(acc[r][0]);
#pragma unroll
    for (int sp = 1; sp < MAX; ++sp)
      if (sp < splits) out[r] += vv_warp_reduce_sum(acc[r][sp]);
  }
}

// The split each row of an R-row larch_mmvq on w sums in, or 0 where its route takes the H2 decoder (sm_86/89), whose
// bytes mmvq_col does not give.
template <int T, int R>
static int mmvq_row_split(const stream& s, const view& w, int sms) {
  bool rows_k = false;  // mmvq_rows_k: the one-row split
  if constexpr (R > 1)
    if constexpr (T == LARCH_Q3_K || MMVQ_ROUTE[R - 2][T - LARCH_Q4_K]) rows_k = R <= arch::mmvq_rows_up_to(s.arch_code);
  if ((R == 1 || rows_k) && h2_route<T>(s)) return 0;
  return rows_k ? mmvq_split<T>(s, w, sms) : mmvq_split<T, R>(s, w, sms);
}

// Pieces a lane of the fused kernels' id_dot keeps in flight.
constexpr int FUSED_UNROLL = 8;

// larch_fused_moe_gate_up: mmvq_id_k's grid, a warp's column of gate and of up by id_dot, each rounded to f16 as
// mmvq_id stores it, then silu_mul_k's expression on the two.
template <int T>
__global__ void moe_gate_up_k(const uint8_t* gw, bool g8, const uint8_t* uw, bool u8, int64_t w_rb, const __half* x,
                              int64_t x_ld, int64_t x_slot, const int32_t* ids, int64_t ids_ld, __half* out,
                              int64_t out_ld, int n, int64_t k, int slots) {
  const int lane = int(threadIdx.x) % warpSize;
  const int col = int(blockIdx.y) * (int(blockDim.x) / warpSize) + int(threadIdx.x) / warpSize;
  if (col >= n) return;
  const int64_t r = blockIdx.x / slots;
  const int j = int(blockIdx.x % slots);
  const int64_t row = int64_t(ids[r * ids_ld + j]) * n + col;
  const __half* xr = x + r * x_ld + j * x_slot;
  const float g = __half2float(__float2half(id_dot<T, FUSED_UNROLL>(gw + row * w_rb, g8, xr, k, lane)));
  const float u = __half2float(__float2half(id_dot<T, FUSED_UNROLL>(uw + row * w_rb, u8, xr, k, lane)));
  if (!lane) store(out + r * out_ld + j * int64_t(n) + col, g / (1.0f + expf(-g)) * u);
}


larch_status fused_moe_gate_up(larch_stream* vs, const larch_tensor* gate_w, const larch_tensor* up_w,
                                const larch_tensor* x_f16, const larch_tensor* ids_i32, const larch_tensor* out) {
  return run("fused_moe_gate_up", vs, [&](const stream& s) {
    const cpu::fused_gate_up_checked c = cpu::check_fused_moe_gate_up(sizer(s), gate_w, up_w, x_f16, ids_i32, out);
    const view gw = c.gate.w, uw = c.up.w, x = c.gate.x, ids = c.gate.ids, o = c.gate.out;
    aligned({gw, uw, x, ids, o});
    if (gw.dtype != uw.dtype || gw.row_bytes != uw.row_bytes)
      throw failure{LARCH_ERR_UNSUPPORTED, "gate and up of different dtypes or row strides"};
    if (x.rows >= MOE_GROUPED_ROWS) throw failure{LARCH_ERR_UNSUPPORTED, "mmvq_id's grouped rows, which differ"};
    if (!x.rows) return;
    const int64_t slots = ids.cols, n = o.shape[2], k = gw.cols, warps = ID_THREADS / s.warp;
    if ((n + warps - 1) / warps > 65535) throw failure{LARCH_ERR_UNSUPPORTED, "more than 65535 blocks of columns"};
    const dim3 grid(unsigned(x.rows * slots), unsigned((n + warps - 1) / warps));
    with_dtype(gw.dtype, [&](auto t) {
      constexpr int T = decltype(t)::value;
      if constexpr (T != LARCH_F16 && T != LARCH_F32)
        launch(moe_gate_up_k<T>, grid, ID_THREADS, s.s, gw.p, uintptr_t(gw.p) % 8 == 0, uw.p,
               uintptr_t(uw.p) % 8 == 0, gw.row_bytes, x.as<__half>(), x.ld, x.ndim == 3 ? k : 0, ids.as<int32_t>(),
               ids.ld, o.as<__half>(), o.ld, int(n), k, int(slots));
    });
  });
}

// The shared down of larch_fused_moe_down, mmvq's step: its weight, row split and x (one row per block).
struct shared_down {
  const uint8_t* w;
  int64_t w_rb;
  int units, splits;
  bool w16, x16;
  const __half *x, *gate;
  int64_t x_ld, gate_ld;
};

// larch_fused_moe_down: block (row r, warps of columns). A warp's column: the shared down by mmvq_col (TS, unless it
// is F32: none), rounded to f16, times sigmoid_mul_k's sigmoid, rounded, onto x as add_k adds; then mmvq_id_k's
// weighted sum by id_dot, rounded, onto x.
template <int T, int TS>
__global__ void moe_down_k(const uint8_t* w, bool w8, int64_t w_rb, const __half* act, int64_t act_ld,
                           int64_t act_slot, const int32_t* ids, int64_t ids_ld, const float* wt, int64_t wt_ld,
                           shared_down sh, float* x, int64_t x_ld, int n, int64_t k, int slots) {
  const int lane = int(threadIdx.x) % warpSize;
  const int col = int(blockIdx.y) * (int(blockDim.x) / warpSize) + int(threadIdx.x) / warpSize;
  if (col >= n) return;
  const int64_t r = blockIdx.x;
  float v = x[r * x_ld + col];
  if constexpr (TS != LARCH_F32) {
    float y[FR];
    mmvq_col<TS>(sh.w + col * sh.w_rb, sh.units, sh.splits, sh.w16, sh.x + r * sh.x_ld, 0, 1, sh.x16, lane, false, y);
    const float t = __half2float(__float2half_rn(y[0]));
    const float z = __half2float(sh.gate[r * sh.gate_ld]);
    v = v + __half2float(__float2half_rn(t * (1.0f / (1.0f + expf(-z)))));
  }
  float sum = 0;
#pragma unroll FUSED_UNROLL
  for (int j = 0; j < slots; ++j) {
    const int64_t row = int64_t(ids[r * ids_ld + j]) * n + col;
    const float y = id_dot<T, FUSED_UNROLL>(w + row * w_rb, w8, act + r * act_ld + j * act_slot, k, lane);
    sum += wt[r * wt_ld + j] * y;
  }
  if (!lane) x[r * x_ld + col] = v + __half2float(__float2half(sum));
}

larch_status fused_moe_down(larch_stream* vs, const larch_tensor* down_w, const larch_tensor* act_f16,
                             const larch_tensor* ids_i32, const larch_tensor* weights_f32,
                             const larch_tensor* shared_w, const larch_tensor* shared_act,
                             const larch_tensor* shared_gate, const larch_tensor* x_f32) {
  return run("fused_moe_down", vs, [&](const stream& s) {
    const cpu::fused_down_checked c = cpu::check_fused_moe_down(sizer(s), down_w, act_f16, ids_i32, weights_f32,
                                                                shared_w, shared_act, shared_gate, x_f32);
    const view w = c.down.w, act = c.down.x, ids = c.down.ids, wt = c.down.weights, x = c.x;
    const view sw = c.shared.b_t, sa = c.shared.a, sg = c.shared_gate;
    aligned({w, act, ids, wt, x, sw, sa, sg});
    if (x.rows >= MOE_GROUPED_ROWS) throw failure{LARCH_ERR_UNSUPPORTED, "mmvq_id's grouped rows, which differ"};
    if (!x.rows) return;
    if (sw.p && sw.cols > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "k past 2^31 - 1"};
    const int64_t slots = ids.cols, n = x.cols, k = w.cols, warps = ID_THREADS / s.warp;
    if ((n + warps - 1) / warps > 65535) throw failure{LARCH_ERR_UNSUPPORTED, "more than 65535 blocks of columns"};
    const dim3 grid(unsigned(x.rows), unsigned((n + warps - 1) / warps));
    shared_down sh{sw.p, sw.row_bytes, 0, 0, uintptr_t(sw.p) % 16 == 0, (uintptr_t(sa.p) | uint64_t(sa.ld) * 2) % 16 == 0,
                   sa.as<__half>(), sg.as<__half>(), sa.ld, sg.ld};
    const auto go = [&](auto kernel) {
      launch(kernel, grid, ID_THREADS, s.s, w.p, uintptr_t(w.p) % 8 == 0, w.row_bytes, act.as<__half>(), act.ld,
             act.ndim == 3 ? k : 0, ids.as<int32_t>(), ids.ld, wt.as<float>(), wt.ld, sh, x.as<float>(), x.ld, int(n),
             k, int(slots));
    };
    with_dtype(w.dtype, [&](auto t) {
      constexpr int T = decltype(t)::value;
      if constexpr (T != LARCH_F16 && T != LARCH_F32) {
        if (!sw.p) return go(moe_down_k<T, LARCH_F32>);
        with_dtype(sw.dtype, [&](auto ts) {
          constexpr int TS = decltype(ts)::value;
          if constexpr (TS != LARCH_F16 && TS != LARCH_F32) {
            with_rows(int(x.rows), [&](auto r) { sh.splits = mmvq_row_split<TS, decltype(r)::value>(s, sw, s.compute_units); });
            if (!sh.splits) throw failure{LARCH_ERR_UNSUPPORTED, "the shared down takes mmvq's H2 decoder"};
            sh.units = int(sw.cols / (8 * unit<TS>::P));
            go(moe_down_k<T, TS>);
          }
        });
      }
    });
  });
}

// A projection of h (f16 [rows, k], shared memory) that norm_proj_k computes, a warp a column: a quant weight as
// larch_mmvq's rows (mmvq_col at `split`), an f16 one as gemm_rows_k's (rows_dot). pair: the next projection is its
// up, and out gets silu_mul_k of the two columns, each rounded to f16 first.
struct proj {
  const uint8_t* w;
  void* out;
  int64_t w_rb, n, out_ld;
  int dtype, units, split;
  bool w16, out_f32, pair;
};
constexpr int PROJS = 4;
struct projs {
  proj p[PROJS];
  int count;
};

// gemm_rows_k's column sum for one row (alpha 1, no bias): lane l's 8-wide chunks l, l + 32, ... in k order, fmaf of
// x and w, then the butterfly.
__device__ __forceinline__ float rows_dot(const __half* w, const __half* x, int64_t k, int lane) {
  float acc = 0;
  for (int64_t c0 = int64_t(lane) * 8; c0 < k; c0 += 256) {
    const int count = int(min(int64_t(8), k - c0));
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (i < count) acc = fmaf(__half2float(x[c0 + i]), __half2float(w[c0 + i]), acc);
  }
  return vv_warp_reduce_sum(acc);
}

// Column col of p on rows x (x_ld apart, out[r] for r < rows <= FR), its dtype switched per warp (one branch, so one
// kernel serves every weight). xs: x is in shared memory.
__device__ __forceinline__ void proj_rows(const proj& p, int64_t col, const __half* x, int64_t x_ld, int64_t k, int rows,
                                          bool x16, bool xs, int lane, float (&out)[FR]) {
  const uint8_t* w = p.w + col * p.w_rb;
  switch (p.dtype) {
    case LARCH_Q3_K: return mmvq_col<LARCH_Q3_K>(w, p.units, p.split, p.w16, x, x_ld, rows, x16, lane, xs, out);
    case LARCH_Q4_K: return mmvq_col<LARCH_Q4_K>(w, p.units, p.split, p.w16, x, x_ld, rows, x16, lane, xs, out);
    case LARCH_Q5_K: return mmvq_col<LARCH_Q5_K>(w, p.units, p.split, p.w16, x, x_ld, rows, x16, lane, xs, out);
    case LARCH_Q6_K: return mmvq_col<LARCH_Q6_K>(w, p.units, p.split, p.w16, x, x_ld, rows, x16, lane, xs, out);
    case LARCH_Q8_0: return mmvq_col<LARCH_Q8_0>(w, p.units, p.split, p.w16, x, x_ld, rows, x16, lane, xs, out);
    case LARCH_Q4_0: return mmvq_col<LARCH_Q4_0>(w, p.units, p.split, p.w16, x, x_ld, rows, x16, lane, xs, out);
    case LARCH_Q5_0: return mmvq_col<LARCH_Q5_0>(w, p.units, p.split, p.w16, x, x_ld, rows, x16, lane, xs, out);
    case LARCH_F16:
      for (int r = 0; r < rows; ++r) out[r] = rows_dot(reinterpret_cast<const __half*>(w), x + r * x_ld, k, lane);
      return;
    default:  // make_proj refuses every other dtype; NaN, never another dtype's arithmetic on these bytes
      for (int r = 0; r < rows; ++r) out[r] = __int_as_float(0x7fc00000);
  }
}

// rms_k's bytes for row x (n values) into out (shared memory) from a block of any size: rms_k at `threads` threads and
// V-wide loads has thread t sum the squares of its elements, chunk by chunk; each of its warps sums its threads by
// the butterfly, and vv_block_reduce the warps' sums by the butterfly. Here each warp takes rms_k's warps in turn.
__device__ void rms_row(const float* x, const float* w, int64_t w_ld, float eps, int64_t n, int V, int threads,
                        __half* out, float* part) {
  const int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32, warps = int(blockDim.x) / 32;
  const int64_t step = int64_t(threads) * V;
  for (int vw = warp; vw < threads / 32; vw += warps) {
    float ss = 0;
    for (int64_t i = int64_t(vw * 32 + lane) * V; i < n; i += step)
      for (int k = 0; k < V; ++k) ss += x[i + k] * x[i + k];
    ss = vv_warp_reduce_sum(ss);
    if (!lane) part[vw] = ss;
  }
  __syncthreads();
  const float sum = vv_warp_reduce_sum(lane < threads / 32 ? part[lane] : 0.0f);
  const float scale = 1.0f / sqrtf(sum / float(n) + eps);
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) out[i] = __float2half_rn(x[i] * scale * w[i * w_ld]);
  __syncthreads();
}

// rmsnorm of x's rows (rms_k's launch shape V, threads) into shared memory, block 0 also into h (when given), then
// each warp one column of the projections, every row (fused_moe_in, fused_gdn_in).
__global__ void __launch_bounds__(ID_THREADS) norm_proj_k(const float* x, int64_t x_ld, const float* nw, int64_t nw_ld,
                                                         float eps, int64_t k, int V, int threads, int rows, __half* h,
                                                         int64_t h_ld, projs ps) {
  extern __shared__ __half hs[];
  __shared__ float part[32];
  for (int r = 0; r < rows; ++r) rms_row(x + r * x_ld, nw, nw_ld, eps, k, V, threads, hs + r * k, part);
  if (h && !blockIdx.x)
    for (int64_t i = threadIdx.x; i < rows * k; i += blockDim.x) h[i / k * h_ld + i % k] = hs[i];
  const int lane = int(threadIdx.x) % 32;
  int64_t col = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  int p = 0;
  for (; p < ps.count && col >= ps.p[p].n; p += ps.p[p].pair ? 2 : 1) col -= ps.p[p].n;
  if (p >= ps.count) return;
  const proj& pj = ps.p[p];
  for (int r0 = 0; r0 < rows; r0 += FR) {
    const int n = min(FR, rows - r0);
    float v[FR], u[FR];
    proj_rows(pj, col, hs + r0 * k, k, k, n, true, true, lane, v);
    if (pj.pair) proj_rows(ps.p[p + 1], col, hs + r0 * k, k, k, n, true, true, lane, u);
    for (int i = 0; i < n; ++i) {
      if (pj.pair) {
        const float g = __half2float(__float2half_rn(v[i]));
        v[i] = g / (1.0f + expf(-g)) * __half2float(__float2half_rn(u[i]));
      }
      if (lane) continue;
      if (pj.out_f32)
        store(static_cast<float*>(pj.out) + (r0 + i) * pj.out_ld + col, v[i]);
      else
        store(static_cast<__half*>(pj.out) + (r0 + i) * pj.out_ld + col, v[i]);
    }
  }
}

// w as a projection of `rows` rows into o: its dtype's units and the split larch_mmvq's rows sum in (an f16 w:
// gemm_rows_k's, which needs nothing), or UNSUPPORTED where they take mmvq's H2 decoder.
static proj make_proj(const stream& s, const view& w, int rows, const view& o, bool pair = false) {
  proj p{w.p, o.p, w.row_bytes, w.rows, o.ld, int(w.dtype), 0, 0, uintptr_t(w.p) % 16 == 0, o.dtype == LARCH_F32,
         pair};
  if (w.dtype == LARCH_F16) return p;
  switch (w.dtype) {  // proj_rows's mmvq_col cases; any other weight (a new quant) is refused, not misread
    case LARCH_Q3_K: case LARCH_Q4_K: case LARCH_Q5_K: case LARCH_Q6_K: case LARCH_Q8_0: case LARCH_Q4_0:
    case LARCH_Q5_0: break;
    default: throw failure{LARCH_ERR_UNSUPPORTED, "a projection dtype without a fused decoder"};
  }
  if (w.cols > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "k past 2^31 - 1"};
  with_dtype(w.dtype, [&](auto t) {
    constexpr int T = decltype(t)::value;
    if constexpr (T != LARCH_F16 && T != LARCH_F32) {
      with_rows(rows, [&](auto r) { p.split = mmvq_row_split<T, decltype(r)::value>(s, w, s.compute_units); });
      p.units = int(w.cols / (8 * unit<T>::P));
    }
  });
  if (!p.split) throw failure{LARCH_ERR_UNSUPPORTED, "a projection takes mmvq's H2 decoder"};
  return p;
}

// norm_proj_k over ps, x's rows normalised by nw into h (or nowhere), where gemm_rows_k (and so gemm) and mmvq take
// every projection: a warp of 32, up to GEMM_ROWS rows where an f16 projection stands in for gemm, else mmvq's
// MMVQ_ROWS, whose h fits in a block's shared memory.
static void norm_proj(const stream& s, const view& x, const view& nw, float eps, const view& h, const projs& ps) {
  const int64_t rows = x.rows, k = x.cols, warps = ID_THREADS / 32;
  bool gemm = false;
  for (int p = 0; p < ps.count; ++p) gemm |= ps.p[p].dtype == LARCH_F16;
  if (s.warp != 32) throw failure{LARCH_ERR_UNSUPPORTED, "a warp other than 32"};
  if (rows > (gemm ? GEMM_ROWS : MMVQ_ROWS)) throw failure{LARCH_ERR_UNSUPPORTED, "rows past gemm_rows_k's or mmvq's"};
  if (k % 8) throw failure{LARCH_ERR_UNSUPPORTED, "h rows not 16-byte aligned in shared memory"};
  const uint64_t smem = uint64_t(rows * k) * 2;
  if (smem > 48 * 1024) throw failure{LARCH_ERR_UNSUPPORTED, "h past a block's shared memory"};
  int64_t cols = 0;
  for (int p = 0; p < ps.count; p += ps.p[p].pair ? 2 : 1) cols += ps.p[p].n;
  if ((cols + warps - 1) / warps > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "too many columns"};
  const rms_shape rs = rms_shape_of<float, __half>(s, x, h, k);
  norm_proj_k<<<unsigned((cols + warps - 1) / warps), ID_THREADS, smem, s.s>>>(
      x.as<float>(), x.ld, nw.as<float>(), nw.ld, eps, k, rs.vec, rs.threads, int(rows), h.as<__half>(), h.ld, ps);
  ck(cudaGetLastError(), "kernel launch");
}

larch_status fused_moe_in(larch_stream* vs, const larch_tensor* x_f32, const larch_tensor* norm_w, float eps,
                           const larch_tensor* h_f16, const larch_tensor* router_f16, const larch_tensor* logits_f32,
                           const larch_tensor* ids_i32, const larch_tensor* weights_f32,
                           const larch_tensor* shared_gate_w, const larch_tensor* shared_up_w,
                           const larch_tensor* shared_act, const larch_tensor* gate_inp,
                           const larch_tensor* shared_gate) {
  return run("fused_moe_in", vs, [&](const stream& s) {
    const cpu::fused_moe_in_checked c =
        cpu::check_fused_moe_in(sizer(s), x_f32, norm_w, h_f16, router_f16, logits_f32, ids_i32, weights_f32,
                                shared_gate_w, shared_up_w, shared_act, gate_inp, shared_gate);
    const view x = c.norm.x, nw = c.norm.w, h = c.norm.out, lg = c.router.c, ids = c.route.ids, wt = c.route.weights;
    const view sg = c.shared_gate.b_t, su = c.shared_up.b_t, act = c.shared_gate.c, gi = c.gate_inp.b_t,
               go = c.gate_inp.c;
    aligned({x, nw, h, c.router.b_t, lg, ids, wt, sg, su, act, gi, go});
    if (lg.cols > MOE_ROUTE_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "more than 512 experts"};
    if (!x.rows) return;
    projs ps{{make_proj(s, c.router.b_t, int(x.rows), lg)}, 1};
    if (sg.p) {
      ps.p[1] = make_proj(s, sg, int(x.rows), act, true);
      ps.p[2] = make_proj(s, su, int(x.rows), act);
      ps.p[3] = make_proj(s, gi, int(x.rows), go);
      ps.count = 4;
    }
    norm_proj(s, x, nw, eps, h, ps);
    if (ids.cols)
      launch(moe_route_k, unsigned(lg.rows), s.warp, s.s, lg.as<float>(), lg.ld, ids.as<int32_t>(), ids.ld,
             wt.as<float>(), wt.ld, int(lg.cols), int(ids.cols), 0.0f);
  });
}


larch_status fused_gdn_in(larch_stream* vs, const larch_tensor* x_f32, const larch_tensor* norm_w, float eps,
                           const larch_tensor* h_f16, const larch_tensor* qkv_w, const larch_tensor* z_w,
                           const larch_tensor* alpha_w, const larch_tensor* beta_w, const larch_tensor* qkv_f16,
                           const larch_tensor* z_f16, const larch_tensor* alpha_f32, const larch_tensor* beta_f32) {
  return run("fused_gdn_in", vs, [&](const stream& s) {
    const cpu::fused_gdn_in_checked c = cpu::check_fused_gdn_in(sizer(s), x_f32, norm_w, h_f16, qkv_w, z_w, alpha_w,
                                                                beta_w, qkv_f16, z_f16, alpha_f32, beta_f32);
    const view x = c.norm.x, nw = c.norm.w, h = c.norm.out;
    aligned({x, nw, h});
    projs ps{{}, 4};
    for (int i = 0; i < 4; ++i) {
      aligned({c.proj[i].b_t, c.proj[i].c});
      if (x.rows) ps.p[i] = make_proj(s, c.proj[i].b_t, int(x.rows), c.proj[i].c);
    }
    if (x.rows) norm_proj(s, x, nw, eps, h, ps);
  });
}


// larch_fused_proj_add: a warp a column of p, every row: mmvq's sum rounded to f16, added onto x as add_k adds. With
// a gate, each block first writes sigmoid_mul_k's g = in sigmoid(gate) to shared memory, which the columns then read.
__global__ void __launch_bounds__(ID_THREADS) proj_add_k(proj p, const __half* in, int64_t in_ld, const __half* gate,
                                                        int64_t gate_ld, bool per_row, int64_t k, int rows, float* x,
                                                        int64_t x_ld) {
  extern __shared__ __half gs[];
  if (gate) {
    for (int64_t i = threadIdx.x; i < rows * k; i += blockDim.x) {
      const int64_t r = i / k, c = i % k;
      const float z = __half2float(gate[r * gate_ld + (per_row ? 0 : c)]);
      gs[i] = __float2half_rn(__half2float(in[r * in_ld + c]) * (1.0f / (1.0f + expf(-z))));
    }
    __syncthreads();
  }
  const int lane = int(threadIdx.x) % 32;
  const int64_t col = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  if (col >= p.n) return;
  const bool x16 = gate || (uintptr_t(in) | uint64_t(in_ld) * 2) % 16 == 0;
  for (int r0 = 0; r0 < rows; r0 += FR) {
    const int n = min(FR, rows - r0);
    float v[FR];
    if (gate) proj_rows(p, col, gs + r0 * k, k, k, n, true, true, lane, v);
    else proj_rows(p, col, in + r0 * in_ld, in_ld, k, n, x16, false, lane, v);
    for (int i = 0; i < n && !lane; ++i)
      x[(r0 + i) * x_ld + col] = x[(r0 + i) * x_ld + col] + __half2float(__float2half_rn(v[i]));
  }
}

larch_status fused_proj_add(larch_stream* vs, const larch_tensor* w_quant, const larch_tensor* in_f16,
                             const larch_tensor* gate_or_null, const larch_tensor* x_f32) {
  return run("fused_proj_add", vs, [&](const stream& s) {
    const cpu::fused_proj_add_checked c = cpu::check_fused_proj_add(sizer(s), w_quant, in_f16, gate_or_null, x_f32);
    const view w = c.proj.b_t, in = c.proj.a, g = c.gate.b, x = c.proj.c;
    aligned({w, in, g, x});
    if (!in.rows) return;
    const uint64_t smem = g.p ? uint64_t(in.rows * in.cols) * 2 : 0;
    if (smem > 48 * 1024 || (g.p && in.cols % 8))
      throw failure{LARCH_ERR_UNSUPPORTED, "the gated input past a block's shared memory, or rows not 16-byte aligned"};
    const proj p = make_proj(s, w, int(in.rows), x);
    const int64_t warps = ID_THREADS / 32, blocks = (w.rows + warps - 1) / warps;
    if (s.warp != 32 || blocks > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "the column grid"};
    proj_add_k<<<unsigned(blocks), ID_THREADS, smem, s.s>>>(p, in.as<__half>(), in.ld, g.as<__half>(), g.ld,
                                                          g.cols == 1, in.cols, int(in.rows), x.as<float>(), x.ld);
    ck(cudaGetLastError(), "kernel launch");
  });
}

#endif

}  // namespace cuda
