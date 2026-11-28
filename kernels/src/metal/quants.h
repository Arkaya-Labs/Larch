// GGUF block decoders and the grouped expert kernels' shared pieces, for ops.metal and attn_full_nax.metal
// (the macOS 26.2 NAX library, built apart): one definition of each decode, so both libraries round alike.
#pragma once
#include <metal_stdlib>

#include "args.h"

namespace lq {
using namespace metal;

typedef device const uchar* bytes_in;

static inline float h16(bytes_in p) { return float(*(device const half*)p); }
template <typename T>
static inline T at(bytes_in p, u64 rs, uint i) {  // element i of a vector, or row i's first element
  return *(device const T*)(p + i * rs);
}

// GGML block decoders, in ggml-quants.c's f32 expression order (the CPU reference's): each writes the
// 32 elements of sub-block c of the block at b.
// Not static: Metal silently drops kernel instantiations whose template argument is a static function.
typedef void (*deq_fn)(bytes_in b, int c, thread float* y);

void deq_q8_0(bytes_in b, int, thread float* y) {
  const float d = h16(b);
  for (int j = 0; j < 32; ++j) y[j] = float(as_type<char>(b[2 + j])) * d;
}

void deq_q4_0(bytes_in b, int, thread float* y) {
  const float d = h16(b);
  for (int j = 0; j < 16; ++j) {
    y[j] = float(int(b[2 + j] & 0xF) - 8) * d;
    y[j + 16] = float(int(b[2 + j] >> 4) - 8) * d;
  }
}

void deq_q5_0(bytes_in b, int, thread float* y) {
  const float d = h16(b);
  const uint qh = uint(b[2]) | uint(b[3]) << 8 | uint(b[4]) << 16 | uint(b[5]) << 24;
  for (int j = 0; j < 16; ++j) {
    const int xh0 = int(((qh >> j) << 4) & 0x10), xh1 = int((qh >> (j + 12)) & 0x10);
    y[j] = float(int((b[6 + j] & 0xF) | xh0) - 16) * d;
    y[j + 16] = float(int((b[6 + j] >> 4) | xh1) - 16) * d;
  }
}

// The 6-bit scale and min of sub-block j of a Q4_K/Q5_K block (ggml's get_scale_min_k4).
static void scale_min_k4(int j, bytes_in q, thread float& sc, thread float& m) {
  if (j < 4) {
    sc = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

void deq_q4_k(bytes_in b, int c, thread float* y) {
  float sc, m;
  scale_min_k4(c, b + 4, sc, m);
  const float d = h16(b) * sc, mn = h16(b + 2) * m;
  bytes_in q = b + 16 + 32 * (c / 2);
  for (int l = 0; l < 32; ++l) y[l] = d * float(c & 1 ? q[l] >> 4 : q[l] & 0xF) - mn;
}

void deq_q5_k(bytes_in b, int c, thread float* y) {
  float sc, m;
  scale_min_k4(c, b + 4, sc, m);
  const float d = h16(b) * sc, mn = h16(b + 2) * m;
  bytes_in qh = b + 16, ql = b + 48 + 32 * (c / 2);
  const uchar u = uchar(1 << c);
  for (int l = 0; l < 32; ++l) y[l] = d * float((c & 1 ? ql[l] >> 4 : ql[l] & 0xF) + (qh[l] & u ? 16 : 0)) - mn;
}

// Sub-block c of a Q3_K block (hmask 32, qs 64, 12 scale bytes, d): bit pair u of half n, high bit 4 n + u of
// hmask (clear: minus 4), 16 elements per signed 6-bit scale (low nibble from byte i % 8, top 2 bits from
// byte 8 + i % 4 at bit 2 (i / 4), minus 32).
// Two elements per 2-byte load (larch.h keeps weights 2-byte aligned and the block is 110 bytes): byte l is the
// low byte of ushort l / 2.
void deq_q3_k(bytes_in b, int c, thread float* y) {
  const int n = c / 4, u = c % 4;
  device const ushort* q = (device const ushort*)(b + 32 + 32 * n);
  device const ushort* hm = (device const ushort*)b;
  bytes_in s = b + 96;
  const float d = h16(b + 108);
  for (int h = 0; h < 2; ++h) {
    const int i = 8 * n + 2 * u + h;
    const float dl = d * float(int((i < 8 ? s[i] & 0xF : s[i - 8] >> 4) | (s[8 + i % 4] >> (2 * (i / 4)) & 3) << 4) - 32);
    for (int l = 8 * h; l < 8 * h + 8; ++l) {
      const uint qq = q[l], hh = hm[l];
      y[2 * l] = dl * float(int((qq >> (2 * u)) & 3) - (hh >> (4 * n + u) & 1 ? 0 : 4));
      y[2 * l + 1] = dl * float(int((qq >> (8 + 2 * u)) & 3) - (hh >> (8 + 4 * n + u) & 1 ? 0 : 4));
    }
  }
}

// Half h (16 elements) of deq_q3_k's sub-block c, for kernels that give each thread 16 elements.
static void deq16_q3_k(bytes_in b, int c, int h, thread float* y) {
  const int n = c / 4, u = c % 4, i = 8 * n + 2 * u + h;
  device const ushort* q = (device const ushort*)(b + 32 + 32 * n);
  device const ushort* hm = (device const ushort*)b;
  bytes_in s = b + 96;
  const float dl = h16(b + 108) * float(int((i < 8 ? s[i] & 0xF : s[i - 8] >> 4) | (s[8 + i % 4] >> (2 * (i / 4)) & 3) << 4) - 32);
  for (int l = 0; l < 8; ++l) {
    const uint qq = q[8 * h + l], hh = hm[8 * h + l];
    y[2 * l] = dl * float(int((qq >> (2 * u)) & 3) - (hh >> (4 * n + u) & 1 ? 0 : 4));
    y[2 * l + 1] = dl * float(int((qq >> (8 + 2 * u)) & 3) - (hh >> (8 + 4 * n + u) & 1 ? 0 : 4));
  }
}

// deq16_q3_k in two halves, so a kernel can load step k + 1's bytes into registers while step k multiplies:
// the bytes deq16_q3_k reads, then the same math on them. A zeroed q3k16 (d 0) decodes to zeros.
struct q3k16 {
  ushort q[8], hm[8];
  uchar s0, s1;
  half d;
};
static q3k16 load16_q3_k(bytes_in b, int c, int h) {
  const int n = c / 4, u = c % 4, i = 8 * n + 2 * u + h;
  device const ushort* q = (device const ushort*)(b + 32 + 32 * n);
  device const ushort* hm = (device const ushort*)b;
  q3k16 v;
  for (int l = 0; l < 8; ++l) {
    v.q[l] = q[8 * h + l];
    v.hm[l] = hm[8 * h + l];
  }
  v.s0 = b[96 + (i < 8 ? i : i - 8)];
  v.s1 = b[96 + 8 + i % 4];
  v.d = *(device const half*)(b + 108);
  return v;
}
static void dec16_q3_k(thread const q3k16& v, int c, int h, thread float* y) {
  const int n = c / 4, u = c % 4, i = 8 * n + 2 * u + h;
  const float dl = float(v.d) * float(int((i < 8 ? v.s0 & 0xF : v.s0 >> 4) | (v.s1 >> (2 * (i / 4)) & 3) << 4) - 32);
  for (int l = 0; l < 8; ++l) {
    const uint qq = v.q[l], hh = v.hm[l];
    y[2 * l] = dl * float(int((qq >> (2 * u)) & 3) - (hh >> (4 * n + u) & 1 ? 0 : 4));
    y[2 * l + 1] = dl * float(int((qq >> (8 + 2 * u)) & 3) - (hh >> (8 + 4 * n + u) & 1 ? 0 : 4));
  }
}

void deq_q6_k(bytes_in b, int c, thread float* y) {
  const int n = c / 4, q4 = c % 4;  // 128-element half, then 32-element quarter within it
  bytes_in ql = b + 64 * n + 32 * (q4 & 1), qh = b + 128 + 32 * n;
  device const char* sc = (device const char*)(b + 192 + 8 * n + 2 * q4);
  const float d = h16(b + 208);
  for (int l = 0; l < 32; ++l) {
    const int q = int((q4 < 2 ? ql[l] & 0xF : ql[l] >> 4) | ((qh[l] >> (2 * q4)) & 3) << 4) - 32;
    y[l] = d * float(sc[l / 16]) * float(q);
  }
}

// Element l of sub-block c of a Q6_K block, as deq_q6_k decodes it, for kernels whose adjacent threads decode
// adjacent elements.
static float deq1_q6_k(bytes_in b, int c, int l) {
  const int n = c / 4, q4 = c % 4;
  const uchar ql = b[64 * n + 32 * (q4 & 1) + l], qh = b[128 + 32 * n + l];
  const int q = int((q4 < 2 ? ql & 0xF : ql >> 4) | ((qh >> (2 * q4)) & 3) << 4) - 32;
  return h16(b + 208) * float(((device const char*)b)[192 + 8 * n + 2 * q4 + l / 16]) * float(q);
}

// Sub-block c (32 elements) of a row of QK-element, BYTES-byte blocks.
template <int QK, int BYTES, deq_fn F>
static void deq32(bytes_in row, uint c, thread float* y) {
  F(row + c / (QK / 32) * BYTES, int(c % (QK / 32)), y);
}

// Expert e's w rows n0 .. n0 + TN - 1, elements k0 .. k0 + MOE_MM_K - 1, decoded to f16 (gemm_q's rounding) into
// ws4, rows L halves apart; columns past n and elements past k stage zeros. The w tile of both mmvq_id_mm kernels.
template <int QK, int BYTES, deq_fn F, int TN, int L>
static void stage_expert_w(bytes_in w, constant id_args& a, uint e, uint n0, uint k0, uint tid,
                           threadgroup half4* ws4) {
  constexpr int KC = MOE_MM_K, NT = MOE_MM_SG * 32, SUB = KC / 32;
  const uint blocks = a.k / 32;
  // Q6_K lane by element, so a simdgroup's byte loads are one block's adjacent bytes: M5, 512 rows, 2048 x 768,
  // 6.18 -> 5.43 ms. Q3_K and Q4_K lost that way (6.46 -> 6.92, 2.86 -> 7.52 ms) and decode by sub-block.
  if (BYTES == 210) {
    for (uint i = tid; i < uint(TN * KC); i += NT) {
      const uint r = i / KC, kk = k0 + i % KC;
      float v = 0;
      if (n0 + r < a.n && kk < a.k)
        v = deq1_q6_k(w + u64(e * a.n + n0 + r) * a.w_rs + kk / 256 * BYTES, int(kk / 32 % 8), int(kk % 32));
      ((threadgroup half*)ws4)[r * L + i % KC] = half(v);
    }
  } else if (BYTES == 110) {
    // Q3_K by half sub-block, so a 32-column tile keeps all NT threads decoding.
    for (uint i = tid; i < uint(TN * SUB * 2); i += NT) {
      const uint r = i / (SUB * 2), hs = k0 / 16 + i % (SUB * 2), c = hs / 2;
      float y[16] = {};
      if (n0 + r < a.n && c < blocks)
        deq16_q3_k(w + u64(e * a.n + n0 + r) * a.w_rs + c / 8 * BYTES, int(c % 8), int(hs % 2), y);
      threadgroup half4* o = ws4 + (r * L + i % (SUB * 2) * 16) / 4;
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j] = half4(y[4 * j], y[4 * j + 1], y[4 * j + 2], y[4 * j + 3]);
    }
  } else {
    for (uint i = tid; i < uint(TN * SUB); i += NT) {
      const uint r = i / SUB, c = k0 / 32 + i % SUB;
      float y[32] = {};
      if (n0 + r < a.n && c < blocks) deq32<QK, BYTES, F>(w + u64(e * a.n + n0 + r) * a.w_rs, c, y);
      threadgroup half4* o = ws4 + (r * L + i % SUB * 32) / 4;
#pragma unroll
      for (int j = 0; j < 8; ++j) o[j] = half4(y[4 * j], y[4 * j + 1], y[4 * j + 2], y[4 * j + 3]);
    }
  }
}

// Pair q's x row: row q / slots, slot q % slots.
static u64 pair_x_off(constant id_args& a, uint q) { return u64(q / a.slots) * a.x_rs + q % a.slots * a.x_slot_rs; }

// Pair order[m]'s product v at column n of expert e: to out (r, slot), or weighted to the f32 [pairs, n] scratch.
static void store_pair(device uchar* out, bytes_in bias, constant id_args& a, device const uint* order, uint m, uint e,
                       uint n, float v) {
  const uint q = order[m];
  if (a.has_bias) v += at<float>(bias, a.bias_rs, e * a.n + n);
  if (a.weighted) {
    ((device float*)out)[u64(q) * a.n + n] = v;
    return;
  }
  device uchar* o = out + q / a.slots * a.out_rs + q % a.slots * a.out_slot_rs;
  if (a.out_f32)
    ((device float*)o)[n] = v;
  else
    ((device half*)o)[n] = half(v);
}

}  // namespace lq
