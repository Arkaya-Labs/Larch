// The larch.h ops on Metal (architecture 6.1, 6.4). backend.mm checks every shape and dtype on the host;
// nothing here checks data. Dot products, norm statistics and softmax accumulate in f32. Built with
// -fmetal-math-mode=safe: the attention and top-k code relies on -inf comparing and exp'ing correctly.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#include "args.h"

using namespace metal;
using namespace mpp::tensor_ops;

typedef device const uchar* bytes_in;

static inline float h16(bytes_in p) { return float(*(device const half*)p); }
template <typename T>
static inline T at(bytes_in p, u64 rs, uint i) {  // element i of a vector, or row i's first element
  return *(device const T*)(p + i * rs);
}

/// metal_perf's peak-bandwidth probe: 16 bytes per thread, the tail byte by byte.
kernel void vv_perf_copy(device const uchar* source [[buffer(0)]], device uchar* destination [[buffer(1)]],
                         constant u64& bytes [[buffer(2)]], uint index [[thread_position_in_grid]]) {
  const u64 offset = u64(index) * 16;
  if (offset >= bytes) return;
  if (bytes - offset >= 16) {
    *(device uint4*)(destination + offset) = *(device const uint4*)(source + offset);
    return;
  }
  const uint tail = uint(bytes - offset);
  for (uint byte = 0; byte < tail; ++byte) destination[offset + byte] = source[offset + byte];
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

// Element (r, n) of a matmul's c: bias[n] added when there is one, then stored as c's f32 or f16.
void store_c(device uchar* c, constant mm_args& p, bytes_in bias, uint r, uint n, float v) {
  if (p.has_bias) v += at<float>(bias, p.bias_rs, n);
  device uchar* row = c + r * p.c_rs;
  if (p.c_f32)
    ((device float*)row)[n] = v;
  else
    ((device half*)row)[n] = half(v);
}

// embed, quant table: thread (c, r) decodes sub-block c of row ids[r]. rs0 table, rs1 ids, rs2 out.
template <int QK, int BYTES, deq_fn F>
kernel void embed_q(bytes_in table [[buffer(0)]], bytes_in ids [[buffer(1)]], device uchar* out [[buffer(2)]],
                    constant ew_args& a [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  float y[32];
  deq32<QK, BYTES, F>(table + u64(at<int>(ids, a.rs1, g.y)) * a.rs0, g.x, y);
  device float* o = (device float*)(out + g.y * a.rs2) + g.x * 32;
  for (int j = 0; j < 32; ++j) o[j] = y[j];
}

// embed, f16 or f32 table: one element per thread.
template <typename T>
kernel void embed_plain(bytes_in table [[buffer(0)]], bytes_in ids [[buffer(1)]], device uchar* out [[buffer(2)]],
                        constant ew_args& a [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  const T x = ((device const T*)(table + u64(at<int>(ids, a.rs1, g.y)) * a.rs0))[g.x];
  ((device float*)(out + g.y * a.rs2))[g.x] = float(x);
}

// dequant: thread (c, r) writes sub-block c of row r as f16. rs0 w, rs1 out.
template <int QK, int BYTES, deq_fn F>
kernel void dequant(bytes_in w [[buffer(0)]], device uchar* out [[buffer(1)]], constant ew_args& a [[buffer(2)]],
                    uint2 g [[thread_position_in_grid]]) {
  float y[32];
  deq32<QK, BYTES, F>(w + g.y * a.rs0, g.x, y);
  device half* o = (device half*)(out + g.y * a.rs1) + g.x * 32;
  for (int j = 0; j < 32; ++j) o[j] = half(y[j]);
}

// mmvq: one simdgroup per output column n. Lane l decodes sub-blocks l, l + 32, ... of w's row n once
// and multiplies each into every x row (rows <= 16). a = x, b = w, c = out.
// Each column reads all of x again, so backend.mm sends 1 row to vv_mmvq1_*, 2-16 rows of the K-quants to
// mmvq_mma (Q4_K, Q5_K and Q6_K from 3), and the other rows and formats to mmvq_ext and mmvq_mm below.
// ponytail: mmvq_mma decodes the K-quants only. Q8_0, Q4_0 and Q5_0 keep mmvq_ext and mmvq_mm (Q8_0
// 14336 x 4096 at 8 rows: 548 us, llama.cpp 541, M5, 10 Oct). Give mmvq_mma a 16-weight decoder for a format
// when a model with it shows mmvq in W2 (Q3_K and Q5_K: the 35B's 8-chat decode, 27 Nov).
template <int QK, int BYTES, deq_fn F>
kernel void mmvq(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], bytes_in bias [[buffer(2)]],
                 device uchar* out [[buffer(3)]], constant mm_args& a [[buffer(4)]],
                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                 uint sgs [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg * sgs + sg;
  if (n >= a.n) return;  // the whole simdgroup
  float acc[16] = {};
  float y[32];
  for (uint c = lane; c < a.k / 32; c += 32) {
    deq32<QK, BYTES, F>(w + n * a.b_rs, c, y);
    for (uint r = 0; r < a.m; ++r) {
      device const half* xr = (device const half*)(x + r * a.a_rs) + c * 32;
      float s = 0;
      for (int j = 0; j < 32; ++j) s += y[j] * float(xr[j]);
      acc[r] += s;
    }
  }
  for (uint r = 0; r < a.m; ++r) {
    const float v = simd_sum(acc[r]);
    if (!lane) store_c(out, a, bias, r, n, v);
  }
}

// mmvq at 1 row for Q4_K and Q6_K, after llama.cpp's kernel_mul_mv_q4_K_f32 and _q6_K_f32: a simdgroup owns
// MMVQ1_NR columns and walks several blocks at once, each lane reading a few quant bytes of each block in
// wide loads and the x values they pair with once for all its columns. mmvq's one-column kernel decodes 32
// weights per lane byte by byte: 90 against 62 us at 4096 x 4096 Q4_K (M5, 10 Oct).
// This lane's part of x . w[n0 + r] for NR columns of w (rows rs bytes apart, n of them, k wide), in sum.
template <short NR = MMVQ1_NR>
static void mmvq1_q4_k_sum(bytes_in w, u64 rs, uint n0, uint n, uint k, bytes_in x, ushort lane, thread float* sum) {
  // Lane group ix takes blocks ix, ix + 4, ...; in a block, lane (iq, ir) reads 8 quant bytes of each of
  // the two 64-weight halves (qs 32 iq + 8 ir and 64 on): 8 low and 8 high nibbles, 16 bits at a time.
  const short ix = lane / 8, iq = lane / 4 % 2, ir = lane % 4;
  device const half* y = (device const half*)x + 256 * ix + 64 * iq + 8 * ir;
  for (uint ib = ix; ib < k / 256; ib += 4, y += 4 * 256) {
    float yl[16], yh[16];
    float4 sumy = 0;
    for (short i = 0; i < 8; ++i) {
      yl[i] = y[i];
      yl[i + 8] = y[i + 32];
      yh[i] = y[i + 128];
      yh[i + 8] = y[i + 160];
      sumy += float4(yl[i], yl[i + 8], yh[i], yh[i + 8]);
    }
    for (short r = 0; r < NR; ++r) {
      bytes_in b = w + min(n0 + r, n - 1) * rs + ib * 144;  // a ragged tail recomputes column n - 1
      device const ushort* sc = (device const ushort*)(b + 4) + iq;
      device const ushort* q1 = (device const ushort*)(b + 16) + 16 * iq + 4 * ir;
      device const ushort* q2 = q1 + 32;
      // Scales and mins of sub-blocks 2 iq, 2 iq + 1 (bytes 0-3) and 2 iq + 4, 2 iq + 5 (bytes 4-7).
      const ushort s16[4] = {ushort(sc[0] & 0x3f3f), ushort(sc[2] & 0x3f3f),
                             ushort((sc[4] & 0x0f0f) | (sc[0] & 0xc0c0) >> 2),
                             ushort((sc[4] >> 4 & 0x0f0f) | (sc[2] & 0xc0c0) >> 2)};
      thread const uchar* s8 = (thread const uchar*)s16;
      float4 acc1 = 0, acc2 = 0;
      for (short i = 0; i < 4; ++i) {  // masked, not shifted: the scale products below undo the place value
        acc1 += float4(yl[2 * i] * (q1[i] & 0x000F), yl[2 * i + 1] * (q1[i] & 0x0F00),
                       yl[2 * i + 8] * (q1[i] & 0x00F0), yl[2 * i + 9] * (q1[i] & 0xF000));
        acc2 += float4(yh[2 * i] * (q2[i] & 0x000F), yh[2 * i + 1] * (q2[i] & 0x0F00),
                       yh[2 * i + 8] * (q2[i] & 0x00F0), yh[2 * i + 9] * (q2[i] & 0xF000));
      }
      sum[r] += h16(b) * ((acc1[0] + acc1[1] / 256) * s8[0] + (acc1[2] + acc1[3] / 256) * s8[1] / 16 +
                          (acc2[0] + acc2[1] / 256) * s8[4] + (acc2[2] + acc2[3] / 256) * s8[5] / 16) -
                h16(b + 2) * (sumy[0] * s8[2] + sumy[1] * s8[3] + sumy[2] * s8[6] + sumy[3] * s8[7]);
    }
  }
}

kernel void vv_mmvq1_q4_k(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], bytes_in bias [[buffer(2)]],
                          device uchar* out [[buffer(3)]], constant mm_args& a [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * MMVQ1_SG + sg) * MMVQ1_NR;
  if (n0 >= a.n) return;  // the whole simdgroup
  float sum[MMVQ1_NR] = {};
  mmvq1_q4_k_sum(w, a.b_rs, n0, a.n, a.k, x, lane, sum);
  for (short r = 0; r < MMVQ1_NR; ++r) {
    const float v = simd_sum(sum[r]);
    if (!lane && n0 + r < a.n) store_c(out, a, bias, 0, n0 + r, v);
  }
}

// mmvq1_q4_k_sum for Q6_K.
template <short NR = MMVQ1_NR>
static void mmvq1_q6_k_sum(bytes_in w, u64 rs, uint n0, uint n, uint k, bytes_in x, ushort lane, thread float* sum) {
  // Lane parity ix takes blocks ix, ix + 2, ...; in a block, lane (ip, il) takes elements l0 .. l0 + 3 of
  // each 32-element quarter of 128-element half ip: ql 64 ip + l0 and 32 on, qh 32 ip + l0.
  const short ix = lane % 2, ip = lane / 16, l0 = 4 * (lane / 2 % 8);
  device const half* y = (device const half*)x + 256 * ix + 128 * ip + l0;
  for (uint ib = ix; ib < k / 256; ib += 2, y += 2 * 256) {
    float yl[16];
    for (short l = 0; l < 4; ++l) {
      yl[4 * l] = y[l];
      yl[4 * l + 1] = y[l + 32];
      yl[4 * l + 2] = y[l + 64];
      yl[4 * l + 3] = y[l + 96];
    }
    for (short r = 0; r < NR; ++r) {
      bytes_in b = w + min(n0 + r, n - 1) * rs + ib * 210;  // a ragged tail recomputes column n - 1
      bytes_in q1 = b + 64 * ip + l0, q2 = q1 + 32, qh = b + 128 + 32 * ip + l0;
      device const char* sc = (device const char*)(b + 192 + 8 * ip + l0 / 16);
      float4 s = 0;
      for (short l = 0; l < 4; ++l)
        s += float4(yl[4 * l] * float(int((q1[l] & 0xF) | (qh[l] & 0x03) << 4) - 32),
                    yl[4 * l + 1] * float(int((q2[l] & 0xF) | (qh[l] & 0x0C) << 2) - 32),
                    yl[4 * l + 2] * float(int((q1[l] >> 4) | (qh[l] & 0x30)) - 32),
                    yl[4 * l + 3] * float(int((q2[l] >> 4) | (qh[l] & 0xC0) >> 2) - 32));
      sum[r] += h16(b + 208) * (s[0] * sc[0] + s[1] * sc[2] + s[2] * sc[4] + s[3] * sc[6]);
    }
  }
}

kernel void vv_mmvq1_q6_k(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], bytes_in bias [[buffer(2)]],
                          device uchar* out [[buffer(3)]], constant mm_args& a [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * MMVQ1_SG + sg) * MMVQ1_NR;
  if (n0 >= a.n) return;  // the whole simdgroup
  float sum[MMVQ1_NR] = {};
  mmvq1_q6_k_sum(w, a.b_rs, n0, a.n, a.k, x, lane, sum);
  for (short r = 0; r < MMVQ1_NR; ++r) {
    const float v = simd_sum(sum[r]);
    if (!lane && n0 + r < a.n) store_c(out, a, bias, 0, n0 + r, v);
  }
}

// mmvq1_q4_k_sum for Q5_K, after llama.cpp's kernel_mul_mv_q5_K_f32: lane % 4 takes blocks lane % 4, + 4, ...; in
// a block, lane (iq, ir) = lane / 4 reads 8 quant bytes of each 64-weight half (qs 32 iq + 8 ir and 64 on) and the
// matching 8 high-bit bytes, whose bits 2 iq, 2 iq + 1 (first half) and 2 iq + 4, 2 iq + 5 (second) it takes.
template <short NR = MMVQ1_NR>
static void mmvq1_q5_k_sum(bytes_in w, u64 rs, uint n0, uint n, uint k, bytes_in x, ushort lane, thread float* sum) {
  const short ix = lane % 4, iq = lane / 16, ir = lane / 4 % 4;
  const uchar hm1 = uchar(1 << (2 * iq)), hm2 = uchar(hm1 << 1), hm3 = uchar(hm1 << 4), hm4 = uchar(hm2 << 4);
  device const half* y = (device const half*)x + 256 * ix + 64 * iq + 8 * ir;
  for (uint ib = ix; ib < k / 256; ib += 4, y += 4 * 256) {
    float yl[16], yh[16];
    float4 sumy = 0;
    for (short l = 0; l < 8; ++l) {
      yl[l] = y[l];
      yl[l + 8] = y[l + 32];
      yh[l] = y[l + 128];
      yh[l + 8] = y[l + 160];
      sumy += float4(yl[l], yl[l + 8], yh[l], yh[l + 8]);
    }
    for (short r = 0; r < NR; ++r) {
      bytes_in b = w + min(n0 + r, n - 1) * rs + ib * 176;  // a ragged tail recomputes column n - 1
      device const ushort* a = (device const ushort*)(b + 4) + iq;
      bytes_in q1 = b + 48 + 32 * iq + 8 * ir, q2 = q1 + 64, qh = b + 16 + 8 * ir;
      const ushort s16[4] = {ushort(a[0] & 0x3f3f), ushort(a[2] & 0x3f3f),
                             ushort((a[4] & 0x0f0f) | (a[0] & 0xc0c0) >> 2),
                             ushort((a[4] >> 4 & 0x0f0f) | (a[2] & 0xc0c0) >> 2)};
      thread const uchar* s8 = (thread const uchar*)s16;
      float4 acc1 = 0, acc2 = 0;
      for (short l = 0; l < 8; ++l) {  // masked, not shifted: the scale products below undo the place value
        const uchar h = qh[l];
        acc1 += float4(yl[l] * (q1[l] & 0x0F), yl[l + 8] * (q1[l] & 0xF0), yh[l] * (q2[l] & 0x0F),
                       yh[l + 8] * (q2[l] & 0xF0));
        acc2 += float4(h & hm1 ? yl[l] : 0.0f, h & hm2 ? yl[l + 8] : 0.0f, h & hm3 ? yh[l] : 0.0f,
                       h & hm4 ? yh[l + 8] : 0.0f);
      }
      sum[r] += h16(b) * (s8[0] * (acc1[0] + 16 * acc2[0]) + s8[1] * (acc1[1] / 16 + 16 * acc2[1]) +
                          s8[4] * (acc1[2] + 16 * acc2[2]) + s8[5] * (acc1[3] / 16 + 16 * acc2[3])) -
                h16(b + 2) * (sumy[0] * s8[2] + sumy[1] * s8[3] + sumy[2] * s8[6] + sumy[3] * s8[7]);
    }
  }
}

kernel void vv_mmvq1_q5_k(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], bytes_in bias [[buffer(2)]],
                          device uchar* out [[buffer(3)]], constant mm_args& a [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * MMVQ1_SG + sg) * MMVQ1_NR;
  if (n0 >= a.n) return;  // the whole simdgroup
  float sum[MMVQ1_NR] = {};
  mmvq1_q5_k_sum(w, a.b_rs, n0, a.n, a.k, x, lane, sum);
  for (short r = 0; r < MMVQ1_NR; ++r) {
    const float v = simd_sum(sum[r]);
    if (!lane && n0 + r < a.n) store_c(out, a, bias, 0, n0 + r, v);
  }
}

// mmvq at 1 row for Q3_K: lane group ix takes blocks ix, ix + 4, ...; in a block, lane (hn, j) of the group takes
// bytes 8 j .. 8 j + 7 of qs half hn and of hmask, all four bit pairs u of them: elements 128 hn + 32 u + 8 j + l.
// 16 bits at a time, masked, not shifted (the place value divides out once a pair), so 8 loads cover 32 weights.
// Byte loads, two a weight, made mmvq_id1_q3_k 47.5% of the M5's MoE decode (Qwen3-30B-A3B Q3_K_M).
template <short NR = MMVQ1_NR>
static void mmvq1_q3_k_sum(bytes_in w, u64 rs, uint n0, uint n, uint k, bytes_in x, ushort lane, thread float* sum) {
  const short ix = lane / 8, hn = lane % 8 / 4, j = lane % 4;
  device const half* y = (device const half*)x + 256 * ix + 128 * hn + 8 * j;
  for (uint ib = ix; ib < k / 256; ib += 4, y += 4 * 256) {
    float yl[32];
    for (short u = 0; u < 4; ++u)
      for (short l = 0; l < 8; ++l) yl[8 * u + l] = y[32 * u + l];
    for (short r = 0; r < NR; ++r) {
      bytes_in b = w + min(n0 + r, n - 1) * rs + ib * 110;  // a ragged tail recomputes column n - 1
      device const ushort* q = (device const ushort*)(b + 32 + 32 * hn + 8 * j);
      device const ushort* hm = (device const ushort*)(b + 8 * j);
      bytes_in s = b + 96;
      float t = 0;
      for (short u = 0; u < 4; ++u) {
        const ushort qm = ushort(3 << (2 * u)), hb = ushort(1 << (4 * hn + u));
        float lo = 0, hi = 0, clear = 0;  // clear: x of the weights whose high bit is clear (minus 4)
        for (short i = 0; i < 4; ++i) {
          const ushort qq = q[i], hh = hm[i];
          const float y0 = yl[8 * u + 2 * i], y1 = yl[8 * u + 2 * i + 1];
          lo += y0 * float(qq & qm);
          hi += y1 * float(qq & ushort(qm << 8));
          clear += (hh & hb ? 0.0f : y0) + (hh & ushort(hb << 8) ? 0.0f : y1);
        }
        const short i = 8 * hn + 2 * u + j / 2;  // the scale of these 8 weights' 16
        const int sc = int((s[i % 8] >> (4 * (i / 8)) & 0xF) | (s[8 + i % 4] >> (2 * (i / 4)) & 3) << 4) - 32;
        t += float(sc) * ((lo + hi / 256) / float(1 << (2 * u)) - 4 * clear);
      }
      sum[r] += h16(b + 108) * t;
    }
  }
}

kernel void vv_mmvq1_q3_k(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], bytes_in bias [[buffer(2)]],
                          device uchar* out [[buffer(3)]], constant mm_args& a [[buffer(4)]],
                          uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg * MMVQ1_SG + sg) * MMVQ1_NR;
  if (n0 >= a.n) return;  // the whole simdgroup
  float sum[MMVQ1_NR] = {};
  mmvq1_q3_k_sum(w, a.b_rs, n0, a.n, a.k, x, lane, sum);
  for (short r = 0; r < MMVQ1_NR; ++r) {
    const float v = simd_sum(sum[r]);
    if (!lane && n0 + r < a.n) store_c(out, a, bias, 0, n0 + r, v);
  }
}

// mmvq for 5-16 rows, the way llama.cpp's mul_mm does it: a threadgroup of 4 simdgroups owns 32 columns of
// w. Per 128-deep step, each thread decodes one 32-element sub-block of one column into threadgroup memory
// (as f16: ops.cpp's mmvq tolerance allows rounding w to f16) and x's rows are staged beside it, zero past
// m and k; each simdgroup multiplies its 8 columns by every 8-row tile of x with simdgroup_matrix, f32
// accumulate. So w is read once per step for all rows, where mmvq reads all of x again for every column.
// RT is x's 8-row tiles: 1 for up to 8 rows, 2 for up to 16.
template <int QK, int BYTES, deq_fn F, int RT>
kernel void mmvq_mm(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], bytes_in bias [[buffer(2)]],
                    device uchar* out [[buffer(3)]], constant mm_args& a [[buffer(4)]],
                    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
  // k per step, columns and x rows per threadgroup. Tile rows are LD halves apart, 8 more than KS, so the
  // 8 rows an 8x8 load reads, and the 4 threads storing one column, are not on the same memory banks.
  constexpr uint KS = 128, LD = KS + 8, NC = 32, MR = 8 * RT;
  threadgroup half4 xs4[MR * LD / 4], ws4[NC * LD / 4];  // half4: 8-byte aligned for the 4-wide stores
  threadgroup half* xs = (threadgroup half*)xs4;
  threadgroup half* ws = (threadgroup half*)ws4;
  threadgroup float cs[MR * NC];
  const uint n0 = tg * NC, subs = a.k / 32;
  simdgroup_float8x8 acc[RT];
  for (uint i = 0; i < RT; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint i = a.m * LD / 4 + tid; i < MR * LD / 4; i += 128) xs4[i] = half4(0);  // rows past m stay zero
  for (uint k0 = 0; k0 < a.k; k0 += KS) {
    // k is whole 32-element blocks, so a 4-wide group is all in or all past k. Packed: larch.h promises x
    // 2-byte alignment only.
    for (uint i = tid; i < a.m * KS / 4; i += 128) {
      const uint r = i / (KS / 4), kk = k0 + i % (KS / 4) * 4;
      xs4[r * LD / 4 + i % (KS / 4)] =
          kk < a.k ? half4(*(device const packed_half4*)((device const half*)(x + r * a.a_rs) + kk)) : half4(0);
    }
    const uint col = tid / 4, sb = k0 / 32 + tid % 4;
    threadgroup half4* dst = (threadgroup half4*)(ws + col * LD + tid % 4 * 32);
    if (n0 + col < a.n && sb < subs) {
      float y[32];
      deq32<QK, BYTES, F>(w + u64(n0 + col) * a.b_rs, sb, y);
      for (int j = 0; j < 8; ++j) dst[j] = half4(y[4 * j], y[4 * j + 1], y[4 * j + 2], y[4 * j + 3]);
    } else {
      for (int j = 0; j < 8; ++j) dst[j] = half4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < KS; kk += 8) {
      simdgroup_half8x8 mb;
      simdgroup_load(mb, ws + sg * 8 * LD + kk, LD, ulong2(0, 0), true);
      for (uint i = 0; i < RT; ++i) {
        simdgroup_half8x8 ma;
        simdgroup_load(ma, xs + i * 8 * LD + kk, LD);
        simdgroup_multiply_accumulate(acc[i], ma, mb, acc[i]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < RT; ++i) simdgroup_store(acc[i], cs + i * 8 * NC + sg * 8, NC);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < MR * NC; i += 128) {
    const uint r = i / NC, n = n0 + i % NC;
    if (r < a.m && n < a.n) store_c(out, a, bias, r, n, cs[i]);
  }
}

// mmvq for 2-8 rows, the way llama.cpp's mul_mv_ext does it: lane (column, part) of a simdgroup takes
// sub-blocks part, part + NX, ... of one of the simdgroup's 32 / NX columns of w. Lanes with the same part
// read the same x, so one load serves 32 / NX columns. No threadgroup memory, no barriers. R is the most
// rows; rows past m are skipped.
template <int QK, int BYTES, deq_fn F, int R>
kernel void mmvq_ext(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], bytes_in bias [[buffer(2)]],
                     device uchar* out [[buffer(3)]], constant mm_args& a [[buffer(4)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint sgs [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NX = MMVQ_EXT_NX;
  const uint part = lane % NX, n = (tg * sgs + sg) * (32 / NX) + lane / NX;
  float acc[R] = {};
  if (n < a.n) {
    for (uint c = part; c < a.k / 32; c += NX) {
      float y[32];
      deq32<QK, BYTES, F>(w + n * a.b_rs, c, y);
#pragma clang loop unroll(full)
      for (uint r = 0; r < R; ++r) {
        if (r >= a.m) break;
        device const packed_half4* xr = (device const packed_half4*)((device const half*)(x + r * a.a_rs) + c * 32);
        float s = 0;
#pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
          const float4 v = float4(half4(xr[j]));
          s += y[4 * j] * v.x + y[4 * j + 1] * v.y + y[4 * j + 2] * v.z + y[4 * j + 3] * v.w;
        }
        acc[r] += s;
      }
    }
  }
#pragma clang loop unroll(full)
  for (uint r = 0; r < R; ++r) {
    float v = acc[r];
    for (ushort o = 1; o < NX; o *= 2) v += simd_shuffle_xor(v, o);
    if (part == 0 && n < a.n && r < a.m) store_c(out, a, bias, r, n, v);
  }
}

// 16 weights, piece p (0-15) of a 256-weight K-quant block (elements 16 p to 16 p + 15), for mmvq_mma.
typedef void (*deq16_fn)(bytes_in b, int p, thread float* y);
void deq16p_q3_k(bytes_in b, int p, thread float* y) { deq16_q3_k(b, p / 2, p % 2, y); }
void deq16_q5_k(bytes_in b, int p, thread float* y) {
  const int c = p / 2, h = p % 2;
  float sc, m;
  scale_min_k4(c, b + 4, sc, m);
  const float d = h16(b) * sc, mn = h16(b + 2) * m;
  bytes_in qh = b + 16 + 16 * h, ql = b + 48 + 32 * (c / 2) + 16 * h;
  const uchar u = uchar(1 << c);
#pragma clang loop unroll(full)
  for (int l = 0; l < 16; ++l) y[l] = d * float((c & 1 ? ql[l] >> 4 : ql[l] & 0xF) + (qh[l] & u ? 16 : 0)) - mn;
}
void deq16_q4_k(bytes_in b, int p, thread float* y) {
  const int c = p / 2;
  float sc, m;
  scale_min_k4(c, b + 4, sc, m);
  const float d = h16(b) * sc, mn = h16(b + 2) * m;
  bytes_in q = b + 16 + 32 * (c / 2) + 16 * (p % 2);
#pragma clang loop unroll(full)
  for (int l = 0; l < 16; ++l) y[l] = d * float(c & 1 ? q[l] >> 4 : q[l] & 0xF) - mn;
}
void deq16_q6_k(bytes_in b, int p, thread float* y) {
  const int c = p / 2, n = c / 4, q4 = c % 4, h = p % 2;
  bytes_in ql = b + 64 * n + 32 * (q4 & 1) + 16 * h, qh = b + 128 + 32 * n + 16 * h;
  const float d = h16(b + 208) * float(((device const char*)(b + 192 + 8 * n + 2 * q4))[h]);
#pragma clang loop unroll(full)
  for (int l = 0; l < 16; ++l) {
    const int q = int((q4 < 2 ? ql[l] & 0xF : ql[l] >> 4) | ((qh[l] >> (2 * q4)) & 3) << 4) - 32;
    y[l] = d * float(q);
  }
}

// mmvq for 2-16 rows of the K-quants on simdgroup matrices, after llama.cpp's few-row MMA kernel (mul_mv_mma):
// a threadgroup owns 8 NT columns of w and all 8 RT rows of x, and its MMVQ_MMA_SG simdgroups split k in 64-wide
// steps. A lane decodes the 16 weights its A fragments hold straight into registers and loads the x values of
// its B fragments, so threadgroup memory only sums the simdgroups' partial tiles at the end. The loops must be
// unrolled in full (plain #pragma unroll left them rolled: 2.5x slower, M5, 10 Oct).
// Lane l holds row fm and columns fn, fn + 1 of each 8x8 fragment. MMA s at k index j takes weight
// 16 (j / 2) + 8 (j % 2) + s of the step, so the A fragments of lane l hold piece fn / 2 of row fm.
inline ushort mma_fm(ushort l) { return ((l / 4) & 4) + ((l / 2) % 4); }
inline ushort mma_fn(ushort l) { return ((l / 4) & 2) * 2 + (l % 2) * 2; }

template <int BYTES, deq16_fn F, int RT>
kernel void mmvq_mma(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], bytes_in bias [[buffer(2)]],
                     device uchar* out [[buffer(3)]], constant mm_args& a [[buffer(4)]],
                     uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int NT = MMVQ_MMA_NT;
  threadgroup float red[MMVQ_MMA_SG * RT * NT * 64];
  const ushort fm = mma_fm(lane), fn = mma_fn(lane);
  const uint n0 = tg * 8 * NT;
  bytes_in wr[NT];
#pragma clang loop unroll(full)
  for (int t = 0; t < NT; ++t) wr[t] = w + u64(min(n0 + 8 * t + fm, a.n - 1)) * a.b_rs;  // past n: ignored
  device const half* xr[RT][2];
#pragma clang loop unroll(full)
  for (int rt = 0; rt < RT; ++rt)
#pragma clang loop unroll(full)
    for (int e = 0; e < 2; ++e)
      xr[rt][e] = (device const half*)(x + u64(min(uint(8 * rt + fn + e), a.m - 1)) * a.a_rs) + 16 * (fm / 2) +
                  8 * (fm % 2);
  simdgroup_float8x8 c[RT][NT];
#pragma clang loop unroll(full)
  for (int rt = 0; rt < RT; ++rt)
#pragma clang loop unroll(full)
    for (int t = 0; t < NT; ++t) c[rt][t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint g = sg; g < a.k / 64; g += MMVQ_MMA_SG) {
    simdgroup_float8x8 b[RT][8];
#pragma clang loop unroll(full)
    for (int rt = 0; rt < RT; ++rt)
#pragma clang loop unroll(full)
      for (int i = 0; i < 2; ++i) {  // packed: larch.h promises x 2-byte alignment only
        const float4 u = float4(half4(((device const packed_half4*)(xr[rt][0] + 64 * g))[i]));
        const float4 v = float4(half4(((device const packed_half4*)(xr[rt][1] + 64 * g))[i]));
#pragma clang loop unroll(full)
        for (int s = 0; s < 4; ++s) {
          b[rt][4 * i + s].thread_elements()[0] = u[s];
          b[rt][4 * i + s].thread_elements()[1] = v[s];
        }
      }
    const int p = 4 * g + fn / 2;
#pragma clang loop unroll(full)
    for (int t = 0; t < NT; ++t) {
      float y[16];
      F(wr[t] + p / 16 * BYTES, p % 16, y);
#pragma clang loop unroll(full)
      for (int s = 0; s < 8; ++s) {
        simdgroup_float8x8 am;
        am.thread_elements()[0] = y[s];
        am.thread_elements()[1] = y[8 + s];
#pragma clang loop unroll(full)
        for (int rt = 0; rt < RT; ++rt) simdgroup_multiply_accumulate(c[rt][t], am, b[rt][s], c[rt][t]);
      }
    }
  }
#pragma clang loop unroll(full)
  for (int rt = 0; rt < RT; ++rt)
#pragma clang loop unroll(full)
    for (int t = 0; t < NT; ++t)
#pragma clang loop unroll(full)
      for (int e = 0; e < 2; ++e) red[((sg * RT + rt) * NT + t) * 64 + 2 * lane + e] = c[rt][t].thread_elements()[e];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = sg * 32 + lane; i < RT * NT * 64; i += MMVQ_MMA_SG * 32) {
    float v = 0;
#pragma clang loop unroll(full)
    for (int s = 0; s < MMVQ_MMA_SG; ++s) v += red[s * RT * NT * 64 + i];
    const ushort l = i % 64 / 2;
    const uint n = n0 + 8 * (i / 64 % NT) + mma_fm(l), r = 8 * (i / (NT * 64)) + mma_fn(l) + i % 2;
    if (n < a.n && r < a.m) store_c(out, a, bias, r, n, v);
  }
}

// mmvq_id: one simdgroup per column n of w's experts, for one (row, slot) pair or, weighted, for one row
// over its slots in ascending order (the CPU reference's sum, in f32). Each lane decodes sub-blocks lane,
// lane + 32, ... of the chosen expert's row, as mmvq's one-column kernel does.
// Q4_K and Q6_K go to mmvq_id1 below, and from MOE_GROUP_PAIRS pairs per expert every format goes to
// mmvq_id_grouped.
template <int QK, int BYTES, deq_fn F>
kernel void mmvq_id(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], bytes_in ids [[buffer(2)]],
                    bytes_in bias [[buffer(3)]], bytes_in wt [[buffer(4)]], device uchar* out [[buffer(5)]],
                    constant id_args& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * sgs + sg;
  if (n >= a.n) return;  // the whole simdgroup
  const uint r = a.weighted ? tg.y : tg.y / a.slots, first = a.weighted ? 0 : tg.y % a.slots;
  const uint last = a.weighted ? a.slots : first + 1;
  float y[32], sum = 0;
  for (uint j = first; j < last; ++j) {
    const uint row = uint(at<int>(ids + r * a.ids_rs, 4, j)) * a.n + n;
    device const half* xr = (device const half*)(x + r * a.x_rs + j * a.x_slot_rs);
    float acc = 0;
    for (uint c = lane; c < a.k / 32; c += 32) {
      deq32<QK, BYTES, F>(w + u64(row) * a.w_rs, c, y);
      float s = 0;
      for (int i = 0; i < 32; ++i) s += y[i] * float(xr[c * 32 + i]);
      acc += s;
    }
    float v = simd_sum(acc);
    if (a.has_bias) v += at<float>(bias, a.bias_rs, row);
    if (a.weighted) {
      sum += at<float>(wt + r * a.wt_rs, 4, j) * v;
    } else if (!lane) {
      device uchar* o = out + r * a.out_rs + j * a.out_slot_rs;
      if (a.out_f32)
        ((device float*)o)[n] = v;
      else
        ((device half*)o)[n] = half(v);
    }
  }
  if (a.weighted && !lane) {
    device uchar* o = out + r * a.out_rs;
    if (a.out_f32)
      ((device float*)o)[n] = sum;
    else
      ((device half*)o)[n] = half(sum);
  }
}

// mmvq_id for Q3_K, Q4_K, Q5_K and Q6_K (BYTES 110, 144, 176 and 210) up to the grouped kernel's row counts: mmvq1's wide-load sums
// over MMVQ1_NR columns of the chosen expert per simdgroup, for one (row, slot) pair or, weighted, one row over
// its slots in ascending order. mmvq_id's one column per simdgroup decodes byte by byte, and at qwen3moe's
// down projection (k = 768) leaves 8 of 32 lanes idle.
template <int BYTES, short NR = MMVQ1_NR>
kernel void mmvq_id1(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], bytes_in ids [[buffer(2)]],
                     bytes_in bias [[buffer(3)]], bytes_in wt [[buffer(4)]], device uchar* out [[buffer(5)]],
                     constant id_args& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  const uint n0 = (tg.x * MMVQ1_SG + sg) * NR;
  if (n0 >= a.n) return;  // the whole simdgroup
  const uint r = a.weighted ? tg.y : tg.y / a.slots, first = a.weighted ? 0 : tg.y % a.slots;
  const uint last = a.weighted ? a.slots : first + 1;
  float acc[NR] = {};
  for (uint j = first; j < last; ++j) {
    const uint e = uint(at<int>(ids + r * a.ids_rs, 4, j));
    bytes_in we = w + u64(e) * a.n * a.w_rs, xr = x + r * a.x_rs + j * a.x_slot_rs;
    float sum[NR] = {};
    if (BYTES == 110)
      mmvq1_q3_k_sum<NR>(we, a.w_rs, n0, a.n, a.k, xr, lane, sum);
    else if (BYTES == 144)
      mmvq1_q4_k_sum<NR>(we, a.w_rs, n0, a.n, a.k, xr, lane, sum);
    else if (BYTES == 176)
      mmvq1_q5_k_sum<NR>(we, a.w_rs, n0, a.n, a.k, xr, lane, sum);
    else
      mmvq1_q6_k_sum<NR>(we, a.w_rs, n0, a.n, a.k, xr, lane, sum);
    for (short c = 0; c < NR; ++c) {
      const uint n = min(n0 + c, a.n - 1);
      float v = simd_sum(sum[c]);
      if (a.has_bias) v += at<float>(bias, a.bias_rs, e * a.n + n);
      if (a.weighted) {
        acc[c] += at<float>(wt + r * a.wt_rs, 4, j) * v;
      } else if (!lane && n0 + c < a.n) {
        device uchar* o = out + r * a.out_rs + j * a.out_slot_rs;
        if (a.out_f32)
          ((device float*)o)[n] = v;
        else
          ((device half*)o)[n] = half(v);
      }
    }
  }
  for (short c = 0; a.weighted && !lane && c < NR && n0 + c < a.n; ++c) {
    device uchar* o = out + r * a.out_rs;
    if (a.out_f32)
      ((device float*)o)[n0 + c] = acc[c];
    else
      ((device half*)o)[n0 + c] = half(acc[c]);
  }
}

// mmvq_id1 for experts under 4 blocks wide (k < 1024: MoE down projections of 512 and 768),
// where mmvq1's lane groups, which each walk every 4th block (every 2nd for Q6_K), leave groups idle (half of
// them at k = 512). Here a simdgroup owns MMVQ1N_C columns and lane group g takes the (column, block) units
// g, g + groups, ... of its columns' MMVQ1N_C x k / 256, so every group gets k / 256 units at any k. A unit runs
// mmvq1's sum on one block of one column, with the lane's place in its group as mmvq1 gives it. As in MLX's
// gather_qmv, all 32 lanes work at any expert width.
template <int BYTES, short C = MMVQ1N_C>
kernel void mmvq_id1n(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], bytes_in ids [[buffer(2)]],
                      bytes_in bias [[buffer(3)]], bytes_in wt [[buffer(4)]], device uchar* out [[buffer(5)]],
                      constant id_args& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  // Lane groups and a lane's place in its group, as each format's sum reads them (its block stride, then the
  // lane with that stride's bits cleared, so the sum starts at block 0).
  constexpr short G = BYTES == 210 ? 2 : 4;
  const short g = BYTES == 110 || BYTES == 144 ? lane / 8 : lane % G;
  const ushort vl = BYTES == 110 || BYTES == 144 ? lane % 8 : lane & ~ushort(G - 1);
  const uint n0 = (tg.x * MMVQ1_SG + sg) * C;
  if (n0 >= a.n) return;  // the whole simdgroup
  const uint r = a.weighted ? tg.y : tg.y / a.slots, first = a.weighted ? 0 : tg.y % a.slots;
  const uint last = a.weighted ? a.slots : first + 1, nb = a.k / 256;
  float acc[C] = {};
  for (uint j = first; j < last; ++j) {
    const uint e = uint(at<int>(ids + r * a.ids_rs, 4, j));
    bytes_in we = w + u64(e) * a.n * a.w_rs, xr = x + r * a.x_rs + j * a.x_slot_rs;
    float sum[C] = {};
    for (uint u = g; u < C * nb; u += G) {
      const uint c = u / nb, ib = u % nb, col = min(n0 + c, a.n - 1);  // a ragged tail recomputes column n - 1
      bytes_in wb = we + ib * BYTES, xb = xr + ib * 512;
      float s[1] = {};
      if (BYTES == 110)
        mmvq1_q3_k_sum<1>(wb, a.w_rs, col, a.n, 256, xb, vl, s);
      else if (BYTES == 144)
        mmvq1_q4_k_sum<1>(wb, a.w_rs, col, a.n, 256, xb, vl, s);
      else if (BYTES == 176)
        mmvq1_q5_k_sum<1>(wb, a.w_rs, col, a.n, 256, xb, vl, s);
      else
        mmvq1_q6_k_sum<1>(wb, a.w_rs, col, a.n, 256, xb, vl, s);
#pragma clang loop unroll(full)
      for (short cc = 0; cc < C; ++cc) sum[cc] += cc == short(c) ? s[0] : 0.0f;
    }
    for (short c = 0; c < C; ++c) {
      const uint n = min(n0 + c, a.n - 1);
      float v = simd_sum(sum[c]);
      if (a.has_bias) v += at<float>(bias, a.bias_rs, e * a.n + n);
      if (a.weighted) {
        acc[c] += at<float>(wt + r * a.wt_rs, 4, j) * v;
      } else if (!lane && n0 + c < a.n) {
        device uchar* o = out + r * a.out_rs + j * a.out_slot_rs;
        if (a.out_f32)
          ((device float*)o)[n] = v;
        else
          ((device half*)o)[n] = half(v);
      }
    }
  }
  for (short c = 0; a.weighted && !lane && c < C && n0 + c < a.n; ++c) {
    device uchar* o = out + r * a.out_rs;
    if (a.out_f32)
      ((device float*)o)[n0 + c] = acc[c];
    else
      ((device half*)o)[n0 + c] = half(acc[c]);
  }
}

// The (row, slot) pairs of mmvq_id by expert, for the grouped kernel: g[e] is expert e's first entry in order
// (g + experts + 1), g[experts] the pair count. One threadgroup; the order within an expert does not matter, as
// each pair's sum is its own.
kernel void vv_moe_group(bytes_in ids [[buffer(0)]], device uint* g [[buffer(1)]], constant id_args& a [[buffer(2)]],
                         uint t [[thread_position_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
  threadgroup atomic_uint cnt[MOE_ROUTE_MAX];
  const uint pairs = a.rows * a.slots;
  for (uint e = t; e < a.experts; e += nt) atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint p = t; p < pairs; p += nt)
    atomic_fetch_add_explicit(&cnt[uint(at<int>(ids + p / a.slots * a.ids_rs, 4, p % a.slots))], 1u,
                              memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    uint sum = 0;
    for (uint e = 0; e < a.experts; ++e) {
      const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
      g[e] = sum;
      atomic_store_explicit(&cnt[e], sum, memory_order_relaxed);
      sum += c;
    }
    g[a.experts] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  device uint* order = g + a.experts + 1;
  for (uint p = t; p < pairs; p += nt) {
    const uint e = uint(at<int>(ids + p / a.slots * a.ids_rs, 4, p % a.slots));
    order[atomic_fetch_add_explicit(&cnt[e], 1u, memory_order_relaxed)] = p;
  }
}

// mmvq_id grouped by expert: one simdgroup per column n of expert tg.y, over the pairs vv_moe_group listed for
// it. Each pair's sum runs in mmvq_id's order, so the results match its bits. Weighted, out is f32 [pairs, n]
// scratch that vv_moe_reduce sums.
template <int QK, int BYTES, deq_fn F>
kernel void mmvq_id_grouped(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], device const uint* g [[buffer(2)]],
                            bytes_in bias [[buffer(3)]], device uchar* out [[buffer(4)]],
                            constant id_args& a [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint sgs [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const uint n = tg.x * sgs + sg, e = tg.y;
  if (n >= a.n) return;  // the whole simdgroup
  const uint p0 = g[e], p1 = g[e + 1], blocks = a.k / 32, row = e * a.n + n;
  if (p0 == p1) return;
  device const uint* order = g + a.experts + 1;
  float y[MOE_GROUP_T][32];
  for (uint t = 0; t < MOE_GROUP_T; ++t)
    if (lane + 32 * t < blocks) deq32<QK, BYTES, F>(w + u64(row) * a.w_rs, lane + 32 * t, y[t]);
  for (uint i = p0; i < p1; ++i) {
    const uint q = order[i], r = q / a.slots, j = q % a.slots;
    device const half* xr = (device const half*)(x + r * a.x_rs + j * a.x_slot_rs);
    float acc = 0;
    for (uint t = 0; t < MOE_GROUP_T; ++t) {
      const uint c = lane + 32 * t;
      if (c >= blocks) break;
      float s = 0;
      for (int l = 0; l < 32; ++l) s += y[t][l] * float(xr[c * 32 + l]);
      acc += s;
    }
    float v = simd_sum(acc);
    if (a.has_bias) v += at<float>(bias, a.bias_rs, row);
    if (lane) continue;
    if (a.weighted) {
      ((device float*)out)[u64(q) * a.n + n] = v;
    } else {
      device uchar* o = out + r * a.out_rs + j * a.out_slot_rs;
      if (a.out_f32)
        ((device float*)o)[n] = v;
      else
        ((device half*)o)[n] = half(v);
    }
  }
}

// The grouped kernel's weighted rows: out (r, n) sums f32 [pairs, n] by wt over the slots in ascending order, as
// mmvq_id does.
kernel void vv_moe_reduce(bytes_in yp [[buffer(0)]], bytes_in wt [[buffer(1)]], device uchar* out [[buffer(2)]],
                          constant id_args& a [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  const uint n = p.x, r = p.y;
  float sum = 0;
  for (uint j = 0; j < a.slots; ++j)
    sum += at<float>(wt + r * a.wt_rs, 4, j) * ((device const float*)yp)[u64(r * a.slots + j) * a.n + n];
  device uchar* o = out + r * a.out_rs;
  if (a.out_f32)
    ((device float*)o)[n] = sum;
  else
    ((device half*)o)[n] = half(sum);
}

// moe_route: one simdgroup per row. Lane l holds experts l, l + 32, ...; each of the k picks takes the
// largest logit not yet taken, the lower id on a tie, as the CPU reference's partial sort does.
kernel void vv_moe_route(bytes_in lg [[buffer(0)]], device uchar* ids [[buffer(1)]], device uchar* wts [[buffer(2)]],
                         constant route_args& a [[buffer(3)]], uint r [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  device const float* l = (device const float*)(lg + r * a.lg_rs);
  device int* out_ids = (device int*)(ids + r * a.ids_rs);
  device float* out_w = (device float*)(wts + r * a.w_rs);
  const uint per = (a.experts + 31) / 32;
  float v[MOE_ROUTE_MAX / 32];
  float mx = -INFINITY;
  for (uint i = 0; i < per; ++i) {
    const uint e = lane + 32 * i;
    v[i] = e < a.experts ? l[e] : -INFINITY;
    mx = max(mx, v[i]);
  }
  mx = simd_max(mx);
  float z = 0;
  for (uint i = 0; i < per; ++i) z += lane + 32 * i < a.experts ? exp(v[i] - mx) : 0.0f;
  z = simd_sum(z);
  uint taken = 0;  // bit i: expert lane + 32 i is picked
  float top = 0;
  for (uint j = 0; j < a.k; ++j) {
    float best = -INFINITY;
    uint id = UINT_MAX;
    for (uint i = 0; i < per; ++i) {
      const uint e = lane + 32 * i;
      if (e < a.experts && !(taken >> i & 1) && (id == UINT_MAX || v[i] > best)) {
        best = v[i];
        id = e;
      }
    }
    const float m = simd_max(id == UINT_MAX ? -INFINITY : best);
    const uint pick = simd_min(id != UINT_MAX && best == m ? id : UINT_MAX);
    if (lane == pick % 32) taken |= 1u << (pick / 32);
    const float p = exp(m - mx) / z;
    top += p;
    if (!lane) {
      out_ids[j] = int(pick);
      out_w[j] = p;
    }
  }
  // Over the sum first, then the scale: scale / top overflows for a large finite scale (FLT_MAX, top 0.5).
  const float sum = max(top, 6.103515625e-05f), scale = a.scale == 0.0f ? 1.0f : a.scale;  // 2^-14, as llama.cpp
  if (!lane)
    for (uint j = 0; j < a.k; ++j) out_w[j] = out_w[j] / sum * scale;
}

typedef decltype(embed_q<32, 34, deq_q8_0>) embed_q_t;
typedef decltype(dequant<32, 34, deq_q8_0>) dequant_t;
typedef decltype(mmvq<32, 34, deq_q8_0>) mmvq_t;
typedef decltype(mmvq_mm<32, 34, deq_q8_0, 1>) mmvq_mm_t;
typedef decltype(mmvq_ext<32, 34, deq_q8_0, 4>) mmvq_ext_t;
typedef decltype(mmvq_id<32, 34, deq_q8_0>) mmvq_id_t;
typedef decltype(mmvq_id_grouped<32, 34, deq_q8_0>) mmvq_id_grouped_t;
#define QUANT(name, qk, bytes)                                                                        \
  template [[host_name("vv_embed_" #name)]] kernel embed_q_t embed_q<qk, bytes, deq_##name>;          \
  template [[host_name("vv_dequant_" #name)]] kernel dequant_t dequant<qk, bytes, deq_##name>;        \
  template [[host_name("vv_mmvq_" #name)]] kernel mmvq_t mmvq<qk, bytes, deq_##name>;                 \
  template [[host_name("vv_mmvq_mm8_" #name)]] kernel mmvq_mm_t mmvq_mm<qk, bytes, deq_##name, 1>;    \
  template [[host_name("vv_mmvq_mm16_" #name)]] kernel mmvq_mm_t mmvq_mm<qk, bytes, deq_##name, 2>;    \
  template [[host_name("vv_mmvq_ext4_" #name)]] kernel mmvq_ext_t mmvq_ext<qk, bytes, deq_##name, 4>;  \
  template [[host_name("vv_mmvq_ext8_" #name)]] kernel mmvq_ext_t mmvq_ext<qk, bytes, deq_##name, 8>;  \
  template [[host_name("vv_mmvq_id_" #name)]] kernel mmvq_id_t mmvq_id<qk, bytes, deq_##name>;          \
  template [[host_name("vv_mmvq_id_grouped_" #name)]] kernel mmvq_id_grouped_t mmvq_id_grouped<qk, bytes, deq_##name>;
QUANT(q3_k, 256, 110)
QUANT(q4_k, 256, 144)
QUANT(q5_k, 256, 176)
QUANT(q6_k, 256, 210)
QUANT(q8_0, 32, 34)
QUANT(q4_0, 32, 18)
QUANT(q5_0, 32, 22)
typedef decltype(mmvq_mma<144, deq16_q4_k, 1>) mmvq_mma_t;
template [[host_name("vv_mmvq_id1_q3_k")]] kernel decltype(mmvq_id1<110>) mmvq_id1<110>;
template [[host_name("vv_mmvq_id1_q4_k")]] kernel decltype(mmvq_id1<144>) mmvq_id1<144>;
template [[host_name("vv_mmvq_id1_q6_k")]] kernel decltype(mmvq_id1<210>) mmvq_id1<210>;
template [[host_name("vv_mmvq_id1_q5_k")]] kernel decltype(mmvq_id1<176>) mmvq_id1<176>;
template [[host_name("vv_mmvq_id1n_q3_k")]] kernel decltype(mmvq_id1n<110>) mmvq_id1n<110>;
template [[host_name("vv_mmvq_id1n_q4_k")]] kernel decltype(mmvq_id1n<144>) mmvq_id1n<144>;
template [[host_name("vv_mmvq_id1n_q5_k")]] kernel decltype(mmvq_id1n<176>) mmvq_id1n<176>;
template [[host_name("vv_mmvq_id1n_q6_k")]] kernel decltype(mmvq_id1n<210>) mmvq_id1n<210>;
template [[host_name("vv_mmvq_id1w_q3_k")]] kernel decltype(mmvq_id1<110, 2 * MMVQ1_NR>) mmvq_id1<110, 2 * MMVQ1_NR>;
template [[host_name("vv_mmvq_id1w_q4_k")]] kernel decltype(mmvq_id1<144, 2 * MMVQ1_NR>) mmvq_id1<144, 2 * MMVQ1_NR>;
template [[host_name("vv_mmvq_id1w_q5_k")]] kernel decltype(mmvq_id1<176, 2 * MMVQ1_NR>) mmvq_id1<176, 2 * MMVQ1_NR>;
template [[host_name("vv_mmvq_id1w_q6_k")]] kernel decltype(mmvq_id1<210, 2 * MMVQ1_NR>) mmvq_id1<210, 2 * MMVQ1_NR>;
template [[host_name("vv_mmvq_id1nw_q3_k")]] kernel decltype(mmvq_id1n<110, 2 * MMVQ1N_C>) mmvq_id1n<110, 2 * MMVQ1N_C>;
template [[host_name("vv_mmvq_id1nw_q4_k")]] kernel decltype(mmvq_id1n<144, 2 * MMVQ1N_C>) mmvq_id1n<144, 2 * MMVQ1N_C>;
template [[host_name("vv_mmvq_id1nw_q5_k")]] kernel decltype(mmvq_id1n<176, 2 * MMVQ1N_C>) mmvq_id1n<176, 2 * MMVQ1N_C>;
template [[host_name("vv_mmvq_id1nw_q6_k")]] kernel decltype(mmvq_id1n<210, 2 * MMVQ1N_C>) mmvq_id1n<210, 2 * MMVQ1N_C>;
template [[host_name("vv_mmvq_mma8_q3_k")]] kernel mmvq_mma_t mmvq_mma<110, deq16p_q3_k, 1>;
template [[host_name("vv_mmvq_mma16_q3_k")]] kernel mmvq_mma_t mmvq_mma<110, deq16p_q3_k, 2>;
template [[host_name("vv_mmvq_mma8_q5_k")]] kernel mmvq_mma_t mmvq_mma<176, deq16_q5_k, 1>;
template [[host_name("vv_mmvq_mma16_q5_k")]] kernel mmvq_mma_t mmvq_mma<176, deq16_q5_k, 2>;
template [[host_name("vv_mmvq_mma8_q4_k")]] kernel mmvq_mma_t mmvq_mma<144, deq16_q4_k, 1>;
template [[host_name("vv_mmvq_mma16_q4_k")]] kernel mmvq_mma_t mmvq_mma<144, deq16_q4_k, 2>;
template [[host_name("vv_mmvq_mma8_q6_k")]] kernel mmvq_mma_t mmvq_mma<210, deq16_q6_k, 1>;
template [[host_name("vv_mmvq_mma16_q6_k")]] kernel mmvq_mma_t mmvq_mma<210, deq16_q6_k, 2>;
template [[host_name("vv_embed_f32")]] kernel decltype(embed_plain<float>) embed_plain<float>;
template [[host_name("vv_embed_f16")]] kernel decltype(embed_plain<half>) embed_plain<half>;

// gemm: a 32 x 32 tile of c per threadgroup of 4 simdgroups, each 16 x 16 as 2 x 2 8x8 f32 accumulators.
// a and b_t tiles of k-width 32 are staged as f16 in threadgroup memory, zero past the edges.
// simdgroup_matrix fallback for tensor-ineligible prefill shapes and force-simd.
kernel void vv_gemm(bytes_in a [[buffer(0)]], bytes_in b [[buffer(1)]], bytes_in bias [[buffer(2)]],
                    device uchar* c [[buffer(3)]], constant mm_args& p [[buffer(4)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half as[32 * 32], bs[32 * 32];
  threadgroup float cs[32 * 32];
  const uint m0 = (p.m_fast ? tg.x : tg.y) * 32, n0 = (p.m_fast ? tg.y : tg.x) * 32;  // backend.mm gemm()
  const uint sm = sg / 2 * 16, sn = sg % 2 * 16;
  simdgroup_float8x8 acc[2][2];
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint k0 = 0; k0 < p.k; k0 += 32) {
    for (uint i = tid; i < 32 * 32; i += 128) {
      const uint r = i / 32, k = k0 + i % 32;
      as[i] = m0 + r < p.m && k < p.k ? ((device const half*)(a + (m0 + r) * p.a_rs))[k] : half(0);
      bs[i] = n0 + r < p.n && k < p.k ? ((device const half*)(b + (n0 + r) * p.b_rs))[k] : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 32; k += 8) {
      simdgroup_half8x8 ma[2], mb[2];
      for (uint i = 0; i < 2; ++i) {
        simdgroup_load(ma[i], as + (sm + 8 * i) * 32 + k, 32);
        simdgroup_load(mb[i], bs + (sn + 8 * i) * 32 + k, 32, ulong2(0, 0), true);  // b_t^T
      }
      for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], cs + (sm + 8 * i) * 32 + sn + 8 * j, 32);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 32 * 32; i += 128) {
    const uint m = m0 + i / 32, n = n0 + i % 32;
    if (m >= p.m || n >= p.n) continue;
    device uchar* row = c + m * p.c_rs;
    float v = p.alpha * cs[i];
    if (p.beta != 0.0f) v += p.beta * (p.c_f32 ? ((device float*)row)[n] : float(((device half*)row)[n]));
    store_c(c, p, bias, m, n, v);
  }
}

/// gemm at up to GEMV_M rows (a decode's MoE router, gated-delta ssm_alpha/beta): one threadgroup per column n of
/// c, its GEMV_SG x 32 threads striding k 4 elements at a time, every row of a at once, then vv_gemm's epilogue.
/// One simdgroup a column left 32-column matrices on 8 threadgroups (vv_gemv was 13 % of a hybrid MoE's decode
/// kernel time, M5, 1 Nov). The tiled kernels stage 32-row tiles of a, all but one row zero at
/// a decode: qwen3moe's router [1, 2048] x [128, 2048]^T took 102 us there (M5), ~4 us of weight reads.
kernel void vv_gemv(bytes_in a [[buffer(0)]], bytes_in b [[buffer(1)]], bytes_in bias [[buffer(2)]],
                    device uchar* c [[buffer(3)]], constant mm_args& p [[buffer(4)]],
                    uint n [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[GEMV_M][GEMV_SG];
  device const half* br = (device const half*)(b + n * p.b_rs);
  float acc[GEMV_M] = {};
  // 4 elements a step, packed: larch.h promises 2-byte alignment only. The tail past k / 4 * 4 one at a time.
  for (uint i = tid; i < p.k / 4; i += 32 * GEMV_SG) {
    const float4 w = float4(half4(((device const packed_half4*)br)[i]));
    for (uint r = 0; r < p.m; ++r) acc[r] += dot(w, float4(half4(((device const packed_half4*)(a + r * p.a_rs))[i])));
  }
  for (uint i = p.k / 4 * 4 + tid; i < p.k; i += 32 * GEMV_SG) {
    const float w = float(br[i]);
    for (uint r = 0; r < p.m; ++r) acc[r] += w * float(((device const half*)(a + r * p.a_rs))[i]);
  }
  for (uint r = 0; r < p.m; ++r) {
    const float s = simd_sum(acc[r]);
    if (!lane) part[r][sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = tid; r < p.m; r += 32 * GEMV_SG) {
    float s = 0;
    for (uint g = 0; g < GEMV_SG; ++g) s += part[r][g];
    device uchar* row = c + r * p.c_rs;
    float v = p.alpha * s;
    if (p.beta != 0.0f) v += p.beta * (p.c_f32 ? ((device float*)row)[n] : float(((device half*)row)[n]));
    store_c(c, p, bias, r, n, v);
  }
}

/// gemm through the Metal 4 tensor API (MPP matmul2d): a TM x TN tile of c per SG simdgroups (args.h's
/// GEMM_TENSOR_* sizes), f32 accumulate,
/// then vv_gemm's alpha / beta / bias epilogue.
template <int TM, int TN, int SG>
kernel void gemm_tensor(device half* a [[buffer(0)]], device half* b [[buffer(1)]], bytes_in bias [[buffer(2)]],
                        device uchar* c [[buffer(3)]], constant mm_args& p [[buffer(4)]],
                        uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr auto descriptor = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, false,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<SG>> op;
  using f16_tensor = tensor<device half, dextents<int, 2>, tensor_inline>;
  auto left = f16_tensor(a, dextents<int, 2>(int(p.k), int(p.m)), array<int, 2>({1, int(p.a_rs / sizeof(half))}));
  auto right = f16_tensor(b, dextents<int, 2>(int(p.k), int(p.n)), array<int, 2>({1, int(p.b_rs / sizeof(half))}));
  const int m0 = int(p.m_fast ? tg.x : tg.y) * TM, n0 = int(p.m_fast ? tg.y : tg.x) * TN;
  auto left_tile = left.slice(0, m0);
  auto right_tile = right.slice(0, n0);
  auto result = op.template get_destination_cooperative_tensor<decltype(left_tile), decltype(right_tile), float>();
  #pragma unroll
  for (uint16_t i = 0; i < result.get_capacity(); ++i)
    if (result.is_valid_element(i)) result[i] = 0.0f;
  op.run(left_tile, right_tile, result);
  #pragma unroll
  for (uint16_t i = 0; i < result.get_capacity(); ++i) {
    if (!result.is_valid_element(i)) continue;
    const auto index = result.get_multidimensional_index(i);
    const int m = m0 + index[1], n = n0 + index[0];
    if (m >= int(p.m) || n >= int(p.n)) continue;
    device uchar* row = c + u64(m) * p.c_rs;
    float v = p.alpha * result[i];
    if (p.beta != 0.0f) v += p.beta * (p.c_f32 ? ((device float*)row)[n] : float(((device half*)row)[n]));
    store_c(c, p, bias, uint(m), uint(n), v);
  }
}
template [[host_name("vv_gemm_tensor")]] kernel decltype(gemm_tensor<GEMM_TENSOR_M, GEMM_TENSOR_N, GEMM_TENSOR_SG>)
    gemm_tensor<GEMM_TENSOR_M, GEMM_TENSOR_N, GEMM_TENSOR_SG>;
template [[host_name("vv_gemm_tensor_deep")]] kernel decltype(gemm_tensor<GEMM_TENSOR_DEEP_M, GEMM_TENSOR_N,
                                                                           GEMM_TENSOR_DEEP_SG>)
    gemm_tensor<GEMM_TENSOR_DEEP_M, GEMM_TENSOR_N, GEMM_TENSOR_DEEP_SG>;

/// gemm with b_t still quantized, for a dequant that backend.mm deferred into the gemm reading its output:
/// c = alpha * a dequant(w)^T + beta * c + bias, as vv_gemm_tensor. A TM x TN tile of c per GEMM_Q_SG
/// simdgroups walks k GEMM_Q_K at a time: each thread decodes one 32-element sub-block of w's tile into
/// threadgroup memory as f16 (the rounding the deferred dequant's output has), then MPP matmul2d multiplies
/// a's tile in place by it, f32 accumulate. So w is read once per row tile of c, not dequant's f16 copy.
template <int QK, int BYTES, deq_fn F, int TM, int TN>
kernel void gemm_q(device half* a [[buffer(0)]], bytes_in w [[buffer(1)]], bytes_in bias [[buffer(2)]],
                   device uchar* c [[buffer(3)]], constant mm_args& p [[buffer(4)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  // LW: a w row's halves, 8 past KC where that fits (3-5 % on the M5). Shader validation counts twice the
  // static threadgroup memory against the 32 KB limit, which leaves the full-width tile no room for it.
  constexpr int KC = GEMM_Q_K, LW = KC + (TN * (KC + 8) * 2 * 2 <= 32768 ? 8 : 0), NT = GEMM_Q_SG * 32;
  threadgroup half4 ws4[TN * LW / 4];
  threadgroup half* ws = (threadgroup half*)ws4;
  using dev_t = tensor<device half, dextents<int, 2>, tensor_inline>;
  using tg_t = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
  constexpr auto descriptor = matmul2d_descriptor(TM, TN, KC, false, true, false,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<GEMM_Q_SG>> op;
  const int m0 = int(p.m_fast ? tg.x : tg.y) * TM, n0 = int(p.m_fast ? tg.y : tg.x) * TN;
  const int a_ld = int(p.a_rs / sizeof(half));
  tg_t wt(ws, dextents<int, 2>(KC, TN), array<int, 2>({1, LW}));
  auto result = op.template get_destination_cooperative_tensor<dev_t, tg_t, float>();
#pragma unroll
  for (uint16_t i = 0; i < result.get_capacity(); ++i)
    if (result.is_valid_element(i)) result[i] = 0.0f;
  for (uint k0 = 0; k0 < p.k; k0 += KC) {
    for (uint i = tid; i < TN * KC / 32; i += NT) {  // w rows past n stage zeros
      const uint r = i / (KC / 32), s = i % (KC / 32);
      float y[32] = {};
      if (n0 + r < p.n) deq32<QK, BYTES, F>(w + u64(n0 + r) * p.b_rs, k0 / 32 + s, y);
      threadgroup half4* o = ws4 + (r * LW + s * 32) / 4;
#pragma unroll
      for (int j = 0; j < 8; ++j) o[j] = half4(y[4 * j], y[4 * j + 1], y[4 * j + 2], y[4 * j + 3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dev_t left(a + u64(m0) * u64(a_ld) + k0, dextents<int, 2>(KC, int(p.m) - m0), array<int, 2>({1, a_ld}));
    op.run(left, wt, result);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#pragma unroll
  for (uint16_t i = 0; i < result.get_capacity(); ++i) {
    if (!result.is_valid_element(i)) continue;
    const auto index = result.get_multidimensional_index(i);
    const int m = m0 + index[1], n = n0 + index[0];
    if (m >= int(p.m) || n >= int(p.n)) continue;
    device uchar* row = c + u64(m) * p.c_rs;
    float v = p.alpha * result[i];
    if (p.beta != 0.0f) v += p.beta * (p.c_f32 ? ((device float*)row)[n] : float(((device half*)row)[n]));
    store_c(c, p, bias, uint(m), uint(n), v);
  }
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

// mmvq_id_mm PF's register loads for step k0: thread tid's XN x pieces (tid, tid + NT, ...; ~0 xoff or past k:
// zeros) and its Q3_K w half sub-block (row r's, tid % HS of the step's; past n or k: a zeroed q3k16).
template <int XN, int KC, int NT, int BYTES, int HS>
static void pf_load(bytes_in x, bytes_in w, constant id_args& a, threadgroup const u64* xoff, uint e, uint r, uint k0,
                    uint tid, uint blocks, thread half4* xr, thread q3k16& wq) {
#pragma unroll
  for (int j = 0; j < XN; ++j) {
    const uint i = tid + uint(j) * NT, m = i / (KC / 4), c = k0 + i % (KC / 4) * 4;
    const u64 off = xoff[m];
    // k is whole 32-element blocks, so a 4-wide group is all in or all past k. Packed: x is 2-byte aligned.
    xr[j] = off != ~u64(0) && c < a.k ? half4(*(device const packed_half4*)((device const half*)(x + off) + c))
                                      : half4(0);
  }
  const uint hs = k0 / 16 + tid % HS, c = hs / 2;
  if (r < a.n && c < blocks) {
    wq = load16_q3_k(w + u64(e * a.n + r) * a.w_rs + c / 8 * BYTES, int(c % 8), int(hs % 2));
  } else {
    for (int l = 0; l < 8; ++l) wq.q[l] = wq.hm[l] = 0;
    wq.s0 = wq.s1 = 0;
    wq.d = 0;
  }
}

/// mmvq_id grouped by expert through MPP matmul2d (a prefill on the tensor path): threadgroup (tg.x, e) owns
/// TN columns of expert e and walks the pairs vv_moe_group listed for it, TM at a time. Per MOE_MM_K step the
/// threads decode w's tile and gather the pairs' x rows, so w is read once per pair tile instead of once per pair
/// as in mmvq_id_grouped. The 64 x 32 Q3_K and Q6_K tile: there the x loads overlap the costly decode, and reading
/// x in place (mmvq_id_mm_xs) took a Q3_K 768 x 2048 call at 1,955 rows from 11.6 to 14.4 ms (M5 profile).
/// PF (Q3_K only, vv_mmvq_id_mm64x32p_q3_k, LARCH_METAL_MOE_PREFETCH): each thread loads the next step's x row
/// pieces and its w half sub-block's bytes into registers before the step's run, and stages them after it, so the
/// device reads overlap the multiply. Same math and rounding as PF false.
template <int QK, int BYTES, deq_fn F, int TM, int TN, bool PF>
kernel void mmvq_id_mm(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], device const uint* g [[buffer(2)]],
                       bytes_in bias [[buffer(3)]], device uchar* out [[buffer(4)]], constant id_args& a [[buffer(5)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr int KC = MOE_MM_K, L = KC + 8, NT = MOE_MM_SG * 32;
  const uint e = tg.y, n0 = tg.x * TN, p0 = g[e], p1 = g[e + 1];
  if (p0 == p1) return;
  device const uint* order = g + a.experts + 1;
  threadgroup half4 ws4[TN * L / 4], xs4[TM * L / 4];
  threadgroup u64 xoff[TM];  // byte offset of each pair's x row; ~0: past the expert's pairs, staged as zeros
  using tg_t = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
  constexpr auto descriptor = matmul2d_descriptor(TM, TN, KC, false, true, false,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<MOE_MM_SG>> op;
  tg_t wt((threadgroup half*)ws4, dextents<int, 2>(KC, TN), array<int, 2>({1, L}));
  tg_t xt((threadgroup half*)xs4, dextents<int, 2>(KC, TM), array<int, 2>({1, L}));
  for (uint m0 = p0; m0 < p1; m0 += TM) {
    for (uint m = tid; m < uint(TM); m += NT) {
      const uint q = m0 + m < p1 ? order[m0 + m] : 0;
      xoff[m] = m0 + m < p1 ? pair_x_off(a, q) : ~u64(0);
    }
    auto result = op.template get_destination_cooperative_tensor<tg_t, tg_t, float>();
#pragma unroll
    for (uint16_t i = 0; i < result.get_capacity(); ++i)
      if (result.is_valid_element(i)) result[i] = 0.0f;
    for (uint k0 = 0; !PF && k0 < a.k; k0 += KC) {
      threadgroup_barrier(mem_flags::mem_threadgroup);  // xoff written, or the last step's run done
      stage_expert_w<QK, BYTES, F, TN, L>(w, a, e, n0, k0, tid, ws4);
      // k is whole 32-element blocks, so a 4-wide group is all in or all past k. Packed: x is 2-byte aligned.
      for (uint i = tid; i < uint(TM * KC / 4); i += NT) {
        const uint m = i / (KC / 4), c = k0 + i % (KC / 4) * 4;
        const u64 off = xoff[m];
        xs4[(m * L) / 4 + i % (KC / 4)] =
            off != ~u64(0) && c < a.k ? half4(*(device const packed_half4*)((device const half*)(x + off) + c)) : half4(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      op.run(xt, wt, result);
    }
    if (PF) {
      // Thread tid's share of a step: x pieces tid, tid + NT, ...; for Q3_K, w half sub-block tid (TN rows of
      // KC / 16 halves is NT of them).
      constexpr int XN = TM * KC / 4 / NT, HS = KC / 16;
      static_assert(!PF || (BYTES == 110 && TM * KC / 4 % NT == 0 && TN * HS == NT), "PF: Q3_K, one half sub-block a thread");
      const uint wr = tid / HS, blocks = a.k / 32;
      half4 xr[XN];
      q3k16 wq;
      threadgroup_barrier(mem_flags::mem_threadgroup);  // xoff written
      pf_load<XN, KC, NT, BYTES, HS>(x, w, a, xoff, e, n0 + wr, 0, tid, blocks, xr, wq);
      for (uint k0 = 0; k0 < a.k; k0 += KC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);  // the last step's run done
        const uint hs = k0 / 16 + tid % HS;
        float y[16];
        dec16_q3_k(wq, int(hs / 2 % 8), int(hs % 2), y);
        threadgroup half4* o = ws4 + (wr * L + tid % HS * 16) / 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = half4(y[4 * j], y[4 * j + 1], y[4 * j + 2], y[4 * j + 3]);
#pragma unroll
        for (int j = 0; j < XN; ++j) {
          const uint i = tid + uint(j) * NT;
          xs4[(i / (KC / 4) * L) / 4 + i % (KC / 4)] = xr[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (k0 + KC < a.k) pf_load<XN, KC, NT, BYTES, HS>(x, w, a, xoff, e, n0 + wr, k0 + KC, tid, blocks, xr, wq);
        op.run(xt, wt, result);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // the next tile's xoff
#pragma unroll
    for (uint16_t i = 0; i < result.get_capacity(); ++i) {
      if (!result.is_valid_element(i)) continue;
      const auto index = result.get_multidimensional_index(i);
      const uint m = m0 + index[1], n = n0 + index[0];
      if (m < p1 && n < a.n) store_pair(out, bias, a, order, m, e, n, result[i]);
    }
  }
}

/// mmvq_id_mm's x: row i of xs is pair order[i]'s x row (vv_moe_group's order), zero-padded to a whole MOE_MM_K,
/// so a pair tile's rows sit at a stride for MPP to read in place. One thread per 4 elements.
kernel void vv_moe_gather(bytes_in x [[buffer(0)]], device const uint* g [[buffer(1)]], device half4* xs [[buffer(2)]],
                          constant id_args& a [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  const uint c = 4 * p.x, q = (g + a.experts + 1)[p.y], kp = (a.k + MOE_MM_K - 1) / MOE_MM_K * MOE_MM_K;
  device const half* xr = (device const half*)(x + pair_x_off(a, q));
  // k is whole 32-element blocks, so a 4-wide group is all in or all past it. Packed: x is 2-byte aligned.
  xs[u64(p.y) * (kp / 4) + p.x] = c < a.k ? half4(*(device const packed_half4*)(xr + c)) : half4(0);
}

/// mmvq_id_mm with x read in place from vv_moe_gather's xs, as gemm_q reads a: only w goes through threadgroup
/// memory. The 32 x 64 tile, which staged a pair tile's x once per 64 columns: 1,955 rows (M5 profile), Q4_K
/// 2048 x 768 weighted 6.01 -> 4.72 ms a call; metal_perf at 512 rows, Q4_K -9.9 %, Q3_K 768 x 2048 -1.8 %.
// ponytail: TM 32 at every row count. TM 64 took the Q4_K call at 1,955 rows a further 9.5 % down but lost 12 to
// 28 % at 512 rows; pick it by pairs an expert (as the tall tile) when down leads a long-prompt profile.
template <int QK, int BYTES, deq_fn F, int TM, int TN>
kernel void mmvq_id_mm_xs(bytes_in w [[buffer(0)]], device half* xs [[buffer(1)]], device const uint* g [[buffer(2)]],
                          bytes_in bias [[buffer(3)]], device uchar* out [[buffer(4)]],
                          constant id_args& a [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
  constexpr int KC = MOE_MM_K, L = KC + 8;
  const uint e = tg.y, n0 = tg.x * TN, p0 = g[e], p1 = g[e + 1], kp = (a.k + KC - 1) / KC * KC;
  if (p0 == p1) return;
  device const uint* order = g + a.experts + 1;
  threadgroup half4 ws4[TN * L / 4];
  using dev_t = tensor<device half, dextents<int, 2>, tensor_inline>;
  using tg_t = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
  constexpr auto descriptor = matmul2d_descriptor(TM, TN, KC, false, true, false,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<MOE_MM_SG>> op;
  tg_t wt((threadgroup half*)ws4, dextents<int, 2>(KC, TN), array<int, 2>({1, L}));
  for (uint m0 = p0; m0 < p1; m0 += TM) {
    auto result = op.template get_destination_cooperative_tensor<dev_t, tg_t, float>();
#pragma unroll
    for (uint16_t i = 0; i < result.get_capacity(); ++i)
      if (result.is_valid_element(i)) result[i] = 0.0f;
    for (uint k0 = 0; k0 < kp; k0 += KC) {
      threadgroup_barrier(mem_flags::mem_threadgroup);  // the last step's run done with ws
      stage_expert_w<QK, BYTES, F, TN, L>(w, a, e, n0, k0, tid, ws4);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // The expert's pairs from m0 on: MPP reads none past p1, which are the next expert's.
      dev_t xt(xs + u64(m0) * kp + k0, dextents<int, 2>(KC, int(p1 - m0)), array<int, 2>({1, int(kp)}));
      op.run(xt, wt, result);
    }
#pragma unroll
    for (uint16_t i = 0; i < result.get_capacity(); ++i) {
      if (!result.is_valid_element(i)) continue;
      const auto index = result.get_multidimensional_index(i);
      const uint m = m0 + index[1], n = n0 + index[0];
      if (m < p1 && n < a.n) store_pair(out, bias, a, order, m, e, n, result[i]);
    }
  }
}
typedef decltype(mmvq_id_mm<32, 34, deq_q8_0, 64, 32, false>) mmvq_id_mm_t;
typedef decltype(mmvq_id_mm_xs<32, 34, deq_q8_0, 32, 64>) mmvq_id_mm_xs_t;
typedef decltype(gemm_q<32, 34, deq_q8_0, GEMM_Q_M_SMALL, GEMM_Q_N>) gemm_q_t;
#define GEMM_Q1(prefix, name, qk, bytes, tm, tn) \
  template [[host_name(prefix #name)]] kernel gemm_q_t gemm_q<qk, bytes, deq_##name, tm, tn>;
#define GEMM_Q(name, qk, bytes)                                                       \
  template [[host_name("vv_mmvq_id_mm_" #name)]] kernel mmvq_id_mm_xs_t mmvq_id_mm_xs<qk, bytes, deq_##name, 32, 64>; \
  template [[host_name("vv_mmvq_id_mm64x32_" #name)]] kernel mmvq_id_mm_t mmvq_id_mm<qk, bytes, deq_##name, 64, 32, false>; \
  GEMM_Q1("vv_gemm_q_small_", name, qk, bytes, GEMM_Q_M_SMALL, GEMM_Q_N)              \
  GEMM_Q1("vv_gemm_q_", name, qk, bytes, GEMM_Q_M, GEMM_Q_N)                          \
  GEMM_Q1("vv_gemm_q_small_narrow_", name, qk, bytes, GEMM_Q_M_SMALL, GEMM_Q_N / 2) \
  GEMM_Q1("vv_gemm_q_narrow_", name, qk, bytes, GEMM_Q_M, GEMM_Q_N / 2)
GEMM_Q(q3_k, 256, 110)
GEMM_Q(q4_k, 256, 144)
GEMM_Q(q5_k, 256, 176)
GEMM_Q(q6_k, 256, 210)
template [[host_name("vv_mmvq_id_mm64x32p_q3_k")]] kernel mmvq_id_mm_t mmvq_id_mm<256, 110, deq_q3_k, 64, 32, true>;
GEMM_Q(q8_0, 32, 34)
GEMM_Q(q4_0, 32, 18)
GEMM_Q(q5_0, 32, 22)

// The simdgroup-matrix tile of vv_gemm_qs and vv_mmvq_id_mm_sg: c[m][n] += xs[m][:] . ws[n][:] over KC, both f16
// rows L halves apart in threadgroup memory. Simdgroup sg owns rows 16 (sg / 2) .. + 15 and columns 32 (sg % 2)
// .. + 31 of the 32 x 64 tile, as 2 x 4 8x8 f32 accumulators acc[4 i + j].
#define QS_MMA_STEP(acc, xs, ws, L, sg)                                                                  \
  for (uint k_ = 0; k_ < GEMM_QS_K; k_ += 8) {                                                            \
    simdgroup_half8x8 ma_[2], mb_[4];                                                                     \
    for (uint i_ = 0; i_ < 2; ++i_) simdgroup_load(ma_[i_], xs + (sg / 2 * 16 + 8 * i_) * L + k_, L);     \
    for (uint j_ = 0; j_ < 4; ++j_)                                                                       \
      simdgroup_load(mb_[j_], ws + (sg % 2 * 32 + 8 * j_) * L + k_, L, ulong2(0, 0), true);               \
    for (uint i_ = 0; i_ < 2; ++i_)                                                                       \
      for (uint j_ = 0; j_ < 4; ++j_)                                                                     \
        simdgroup_multiply_accumulate(acc[4 * i_ + j_], ma_[i_], mb_[j_], acc[4 * i_ + j_]);              \
  }

// Rows n0 .. n0 + GEMM_QS_N - 1 of w (row stride w_rs, rows past n zero), k0 .. k0 + GEMM_QS_K - 1, decoded to f16
// into ws (rows L halves apart; elements past k zero): thread t decodes sub-block t % 2 of row t / 2. The rounding
// is the deferred dequant's f16.
template <int QK, int BYTES, deq_fn F, int L>
static void stage_w(bytes_in w, u64 w_rs, uint n0, uint n, uint k0, uint k, uint tid, threadgroup half* ws) {
  static_assert(GEMM_QS_N * GEMM_QS_K / 32 == GEMM_QS_SG * 32, "one sub-block per thread per step");
  const uint r = tid / 2, sub = tid % 2, c = k0 / 32 + sub;
  float y[32] = {};
  if (n0 + r < n && c < k / 32) deq32<QK, BYTES, F>(w + u64(n0 + r) * w_rs, c, y);
  threadgroup half4* o = (threadgroup half4*)(ws + r * L + sub * 32);
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = half4(y[4 * j], y[4 * j + 1], y[4 * j + 2], y[4 * j + 3]);
}

/// gemm with b_t still quantized, on simdgroup matrices: gemm_q for the GPUs without the tensor API (M1 to M4, or
/// LARCH_METAL_TENSOR=0), where the deferred dequant otherwise writes w's f16 copy and vv_gemm reads it back.
/// c = alpha * a dequant(w)^T + beta * c + bias. A GEMM_QS_M x GEMM_QS_N tile of c per 4 simdgroups, k GEMM_QS_K
/// at a time; the f32 tile goes out through threadgroup memory (the k tiles' space) for vv_gemm's epilogue.
/// llama.cpp's kernel_mul_mm, and our CUDA quant_gemm, in shape. k is whole 32-element blocks (a quant row).
template <int QK, int BYTES, deq_fn F>
kernel void gemm_qs(bytes_in a [[buffer(0)]], bytes_in w [[buffer(1)]], bytes_in bias [[buffer(2)]],
                    device uchar* c [[buffer(3)]], constant mm_args& p [[buffer(4)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint TM = GEMM_QS_M, TN = GEMM_QS_N, KC = GEMM_QS_K, L = KC + 8;
  constexpr uint HALVES = (TN + TM) * L > 2 * TM * TN ? (TN + TM) * L : 2 * TM * TN;
  threadgroup half4 smem[HALVES / 4];
  threadgroup half* ws = (threadgroup half*)smem;
  threadgroup half* xs = ws + TN * L;
  threadgroup float* cs = (threadgroup float*)smem;
  const uint m0 = (p.m_fast ? tg.x : tg.y) * TM, n0 = (p.m_fast ? tg.y : tg.x) * TN;
  simdgroup_float8x8 acc[8];
  for (uint i = 0; i < 8; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const uint xr = tid / 4, xq = tid % 4;  // the row of a's tile and the 16-element quarter this thread stages
  for (uint k0 = 0; k0 < p.k; k0 += KC) {
    stage_w<QK, BYTES, F, L>(w, p.b_rs, n0, p.n, k0, p.k, tid, ws);
    // k is whole 32-element blocks, so a 16-element quarter is all in or all past it. Packed: a is 2-byte aligned.
    const uint kk = k0 + xq * 16;
    threadgroup half4* o = (threadgroup half4*)(xs + xr * L + xq * 16);
    if (m0 + xr < p.m && kk < p.k) {
      device const packed_half4* src = (device const packed_half4*)((device const half*)(a + u64(m0 + xr) * p.a_rs) + kk);
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j] = half4(src[j]);
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j] = half4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    QS_MMA_STEP(acc, xs, ws, L, sg)
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 2; ++i)
    for (uint j = 0; j < 4; ++j)
      simdgroup_store(acc[4 * i + j], cs + (sg / 2 * 16 + 8 * i) * TN + sg % 2 * 32 + 8 * j, TN);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < TM * TN; i += GEMM_QS_SG * 32) {
    const uint m = m0 + i / TN, n = n0 + i % TN;
    if (m >= p.m || n >= p.n) continue;
    device uchar* row = c + u64(m) * p.c_rs;
    float v = p.alpha * cs[i];
    if (p.beta != 0.0f) v += p.beta * (p.c_f32 ? ((device float*)row)[n] : float(((device half*)row)[n]));
    store_c(c, p, bias, m, n, v);
  }
}

/// mmvq_id grouped by expert on simdgroup matrices: mmvq_id_mm for the GPUs without the tensor API. Threadgroup
/// (tg.x, e) owns GEMM_QS_N columns of expert e and walks the pairs vv_moe_group listed for it, GEMM_QS_M at a time,
/// so the pairs' dot products run as 8x8 matrix multiplies. mmvq_id_grouped also decodes each column's row once,
/// but then runs one pair at a time, a simdgroup reduction each.
template <int QK, int BYTES, deq_fn F>
kernel void mmvq_id_mm_sg(bytes_in w [[buffer(0)]], bytes_in x [[buffer(1)]], device const uint* g [[buffer(2)]],
                          bytes_in bias [[buffer(3)]], device uchar* out [[buffer(4)]], constant id_args& a [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint TM = GEMM_QS_M, TN = GEMM_QS_N, KC = GEMM_QS_K, L = KC + 8, NT = GEMM_QS_SG * 32;
  constexpr uint HALVES = (TN + TM) * L > 2 * TM * TN ? (TN + TM) * L : 2 * TM * TN;
  const uint e = tg.y, n0 = tg.x * TN, p0 = g[e], p1 = g[e + 1];
  if (p0 == p1) return;
  device const uint* order = g + a.experts + 1;
  threadgroup half4 smem[HALVES / 4];
  threadgroup half* ws = (threadgroup half*)smem;
  threadgroup half* xs = ws + TN * L;
  threadgroup float* cs = (threadgroup float*)smem;
  threadgroup u64 xoff[TM];  // byte offset of each pair's x row; ~0: past the expert's pairs, staged as zeros
  bytes_in we = w + u64(e) * a.n * a.w_rs;  // expert e's rows
  for (uint m0 = p0; m0 < p1; m0 += TM) {
    for (uint m = tid; m < TM; m += NT) {
      const uint q = m0 + m < p1 ? order[m0 + m] : 0;
      xoff[m] = m0 + m < p1 ? pair_x_off(a, q) : ~u64(0);
    }
    simdgroup_float8x8 acc[8];
    for (uint i = 0; i < 8; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < a.k; k0 += KC) {
      threadgroup_barrier(mem_flags::mem_threadgroup);  // xoff written, or the last step's reads done
      stage_w<QK, BYTES, F, L>(we, a.w_rs, n0, a.n, k0, a.k, tid, ws);
      for (uint i = tid; i < TM * KC / 4; i += NT) {  // k is whole 32-element blocks: a 4-wide group is all in or out
        const uint m = i / (KC / 4), cc = k0 + i % (KC / 4) * 4;
        const u64 off = xoff[m];
        ((threadgroup half4*)(xs + m * L))[i % (KC / 4)] =
            off != ~u64(0) && cc < a.k ? half4(*(device const packed_half4*)((device const half*)(x + off) + cc)) : half4(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      QS_MMA_STEP(acc, xs, ws, L, sg)
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // every simdgroup's last reads of ws and xs
    for (uint i = 0; i < 2; ++i)
      for (uint j = 0; j < 4; ++j)
        simdgroup_store(acc[4 * i + j], cs + (sg / 2 * 16 + 8 * i) * TN + sg % 2 * 32 + 8 * j, TN);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < TM * TN; i += NT) {
      const uint m = m0 + i / TN, n = n0 + i % TN;
      if (m < p1 && n < a.n) store_pair(out, bias, a, order, m, e, n, cs[i]);
    }
  }
}
typedef decltype(gemm_qs<32, 34, deq_q8_0>) gemm_qs_t;
typedef decltype(mmvq_id_mm_sg<32, 34, deq_q8_0>) mmvq_id_mm_sg_t;
#define GEMM_QS(name, qk, bytes)                                                                    \
  template [[host_name("vv_gemm_qs_" #name)]] kernel gemm_qs_t gemm_qs<qk, bytes, deq_##name>;      \
  template [[host_name("vv_mmvq_id_mm_sg_" #name)]] kernel mmvq_id_mm_sg_t mmvq_id_mm_sg<qk, bytes, deq_##name>;
GEMM_QS(q3_k, 256, 110)
GEMM_QS(q4_k, 256, 144)
GEMM_QS(q5_k, 256, 176)
GEMM_QS(q6_k, 256, 210)
GEMM_QS(q8_0, 32, 34)
GEMM_QS(q4_0, 32, 18)
GEMM_QS(q5_0, 32, 22)

// rmsnorm: threadgroup r normalises row r: out = x / sqrt(mean(x^2) + eps) * w. Each thread writes only elements
// it read itself.
kernel void vv_rmsnorm(bytes_in x [[buffer(0)]], bytes_in w [[buffer(1)]], device uchar* out [[buffer(2)]],
                       constant rms_args& a [[buffer(3)]], uint r [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  device const float* xr = (device const float*)(x + r * a.x_rs);
  device half* o = (device half*)(out + r * a.out_rs);
  float ss = 0;
  for (uint i = tid; i < a.n; i += nt) ss += xr[i] * xr[i];
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ss = 0;
  for (uint i = 0; i < nt / 32; ++i) ss += part[i];
  const float scale = 1.0f / precise::sqrt(ss / float(a.n) + a.eps);
  for (uint i = tid; i < a.n; i += nt) o[i] = half(xr[i] * scale * at<float>(w, a.w_rs, i));
}

// qk_norm in place, as vv_rmsnorm a head: simdgroup s of threadgroup g takes head (g % per_row) * simdgroups + s of
// row g / per_row. A simdgroup a head needs no barrier; a prefill threadgroup keeps up to 8 in flight (a threadgroup a
// head took 1.7 ms a call at 2048 MoE prefill rows on the M5), decode's one.
kernel void vv_qk_norm(device uchar* qk [[buffer(0)]], bytes_in w [[buffer(1)]], constant rms_args& a [[buffer(2)]],
                       uint g [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort nsg [[simdgroups_per_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  const uint per_row = (a.heads + nsg - 1) / nsg, hh = g % per_row * nsg + sg;
  if (hh >= a.heads) return;  // the whole simdgroup
  device half* x = (device half*)(qk + g / per_row * a.x_rs) + hh * a.n;
  float ss = 0;
  for (uint i = lane; i < a.n; i += 32) ss += float(x[i]) * float(x[i]);
  const float scale = 1.0f / precise::sqrt(simd_sum(ss) / float(a.n) + a.eps);
  for (uint i = lane; i < a.n; i += 32) x[i] = half(float(x[i]) * scale * at<float>(w, a.w_rs, i));
}

// ggml_rope_ext in place: thread (pair i, head, row), with the CPU reference's f32 order (iterated theta).
kernel void vv_rope(device uchar* qk [[buffer(0)]], bytes_in pos [[buffer(1)]], bytes_in ff [[buffer(2)]],
                    constant rope_args& a [[buffer(3)]], uint3 g [[thread_position_in_grid]]) {
  const uint i = g.x;
  float theta = float(at<int>(pos, a.pos_rs, g.z));
  for (uint j = 0; j < i; ++j) theta *= a.theta_scale;
  const float extrap = theta / (a.has_ff ? at<float>(ff, a.ff_rs, i) : 1.0f), interp = a.freq_scale * extrap;
  float t = interp;
  if (a.ext_factor != 0.0f) {
    const float y = (float(i) - a.lo) / max(0.001f, a.hi - a.lo);
    const float mix = (1 - min(1.0f, max(0.0f, y))) * a.ext_factor;
    t = interp * (1 - mix) + extrap * mix;
  }
  const float c = precise::cos(t) * a.mscale, s = precise::sin(t) * a.mscale;
  device half* x = (device half*)(qk + g.z * a.rs) + g.y * a.hd;
  const uint p0 = a.neox ? i : 2 * i, p1 = a.neox ? i + a.n_dims / 2 : p0 + 1;
  const float x0 = x[p0], x1 = x[p1];
  x[p0] = half(x0 * c - x1 * s);
  x[p1] = half(x0 * s + x1 * c);
}

// layernorm: threadgroup r normalises row r as vv_rmsnorm, around its mean: out = (x - mean) / sqrt(var + eps) * w
// + b, the mean and var each a sum of its own pass (the reference's order of ggml_norm).
kernel void vv_layernorm(bytes_in x [[buffer(0)]], bytes_in w [[buffer(1)]], bytes_in b [[buffer(2)]],
                         device uchar* out [[buffer(3)]], constant ln_args& a [[buffer(4)]],
                         uint r [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  device const float* xr = (device const float*)(x + r * a.x_rs);
  device half* o = (device half*)(out + r * a.out_rs);
  float s = 0;
  for (uint i = tid; i < a.n; i += nt) s += xr[i];
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  s = 0;
  for (uint i = 0; i < nt / 32; ++i) s += part[i];
  const float mean = s / float(a.n);
  threadgroup_barrier(mem_flags::mem_threadgroup);  // every thread has read part before it is reused
  float ss = 0;
  for (uint i = tid; i < a.n; i += nt) {
    const float d = xr[i] - mean;
    ss += d * d;
  }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ss = 0;
  for (uint i = 0; i < nt / 32; ++i) ss += part[i];
  const float scale = 1.0f / precise::sqrt(ss / float(a.n) + a.eps);
  for (uint i = tid; i < a.n; i += nt)
    o[i] = half((xr[i] - mean) * scale * at<float>(w, a.w_rs, i) + (a.has_b ? at<float>(b, a.b_rs, i) : 0.0f));
}

// ggml_rope_multi in place: thread (pair i, head, row), each pair's position and theta steps as the CPU reference's
// rope_multi_position, then vv_rope's rope_yarn. Pair i turns element i with element i + pairs.
kernel void vv_rope_multi(device uchar* qk [[buffer(0)]], bytes_in pos [[buffer(1)]], bytes_in ff [[buffer(2)]],
                          constant rope_multi_args& a [[buffer(3)]], uint3 g [[thread_position_in_grid]]) {
  const uint i = g.x, s0 = a.s[0], s1 = a.s[1], w = s0 + s1, e = w + a.s[2], j = i % (e + a.s[3]);
  uint k = j < s0 ? 0 : j < w ? 1 : j < e ? 2 : 3;
  if (a.mode == 1)  // LARCH_ROPE_IMROPE: the sections interleaved
    k = j % 3 == 1 && j < 3 * s1 ? 1 : j % 3 == 2 && j < 3 * a.s[2] ? 2 : j % 3 == 0 && j < 3 * s0 ? 0 : 3;
  const uint start = k == 0 ? 0 : k == 1 ? s0 : k == 2 ? w : e;
  const uint steps = a.mode == 2 ? j - start : i;  // LARCH_ROPE_VISION: each section's theta starts over
  float theta = float(((device const int*)(pos + g.z * a.pos_rs))[k]);
  for (uint n = 0; n < steps; ++n) theta *= a.theta_scale;
  const float extrap = theta / (a.has_ff ? at<float>(ff, a.ff_rs, i) : 1.0f), interp = a.freq_scale * extrap;
  float t = interp;
  if (a.ext_factor != 0.0f) {
    const float y = (float(i) - a.lo) / max(0.001f, a.hi - a.lo);
    const float mix = (1 - min(1.0f, max(0.0f, y))) * a.ext_factor;
    t = interp * (1 - mix) + extrap * mix;
  }
  const float c = precise::cos(t) * a.mscale, sn = precise::sin(t) * a.mscale;
  device half* x = (device half*)(qk + g.z * a.rs) + g.y * a.hd;
  const float x0 = x[i], x1 = x[i + a.pairs];
  x[i] = half(x0 * c - x1 * sn);
  x[i + a.pairs] = half(x0 * sn + x1 * c);
}

// gelu, ggml_gelu's tanh form in f32: rs0 x, rs2 out. Each thread reads the x it writes, so out may be x.
kernel void vv_gelu(bytes_in x [[buffer(0)]], device uchar* out [[buffer(1)]], constant ew_args& a [[buffer(2)]],
                    uint2 g [[thread_position_in_grid]]) {
  const float v = ((device const half*)(x + g.y * a.rs0))[g.x];
  ((device half*)(out + g.y * a.rs2))[g.x] =
      half(0.5f * v * (1.0f + precise::tanh(0.79788456080286535588f * v * (1.0f + 0.044715f * v * v))));
}

// out = silu(gate) * up. rs0 gate, rs1 up, rs2 out.
kernel void vv_silu_mul(bytes_in gate [[buffer(0)]], bytes_in up [[buffer(1)]], device uchar* out [[buffer(2)]],
                        constant ew_args& a [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  const float x = ((device const half*)(gate + g.y * a.rs0))[g.x];
  const float u = ((device const half*)(up + g.y * a.rs1))[g.x];
  ((device half*)(out + g.y * a.rs2))[g.x] = half(x / (1.0f + exp(-x)) * u);
}

/// sigmoid_mul: out = x * sigmoid(gate), gate per element or, cols 1, per row. Each thread reads the x it writes,
/// so out may be x.
kernel void vv_sigmoid_mul(bytes_in x [[buffer(0)]], bytes_in gate [[buffer(1)]], device uchar* out [[buffer(2)]],
                           constant ew_args& a [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  const float v = ((device const half*)(x + g.y * a.rs0))[g.x];
  const float z = ((device const half*)(gate + g.y * a.rs1))[a.cols == 1 ? 0 : g.x];
  ((device half*)(out + g.y * a.rs2))[g.x] = half(v * (1.0f / (1.0f + exp(-z))));
}

// The first row of sequence s: the q_lens before it.
static uint gdn_first_row(bytes_in lens, constant gdn_args& a, uint s) {
  uint r0 = 0;
  for (uint i = 0; i < s; ++i) r0 += uint(max(at<int>(lens, a.lens_rs, i), 0));
  return r0;
}

/// gdn_conv, step 1 of Gated DeltaNet: thread (channel c, sequence s) walks the sequence's rows in order with
/// the channel's last taps - 1 inputs in registers (the slot's conv state first): out = silu(w[c] . window).
/// Snapshots and the live slot get the window as of their row, f16 as the state stores it (x is f16, so exact).
kernel void vv_gdn_conv(bytes_in x [[buffer(0)]], bytes_in w [[buffer(1)]], device uchar* st [[buffer(2)]],
                        bytes_in slots [[buffer(3)]], bytes_in lens [[buffer(4)]], bytes_in snaps [[buffer(5)]],
                        device uchar* out [[buffer(6)]], constant gdn_args& a [[buffer(7)]],
                        uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, s = g.y, hist = a.taps - 1, ch = a.channels;
  const uint r0 = gdn_first_row(lens, a, s), len = uint(max(at<int>(lens, a.lens_rs, s), 0));
  device half* live = (device half*)(st + u64(uint(at<int>(slots, a.slots_rs, s))) * a.state_rs);
  device const float* wc = (device const float*)(w + u64(c) * a.w_rs);
  float h[GDN_MAX_TAPS - 1], wk[GDN_MAX_TAPS];
  for (uint i = 0; i < hist; ++i) h[i] = float(live[i * ch + c]);
  for (uint i = 0; i < a.taps; ++i) wk[i] = wc[i];
  for (uint t = 0; t < len; ++t) {
    const uint r = r0 + t;
    const float in = float(((device const half*)(x + u64(r) * a.in_rs))[c]);
    float sum = 0;
    for (uint i = 0; i < hist; ++i) sum += wk[i] * h[i];
    sum += wk[hist] * in;
    ((device half*)(out + u64(r) * a.out_rs))[c] = half(sum / (1.0f + exp(-sum)));
    for (uint i = 0; i + 1 < hist; ++i) h[i] = h[i + 1];
    h[hist - 1] = in;
    if (a.has_snaps) {
      const int snap = at<int>(snaps, a.snaps_rs, r);
      if (snap >= 0) {
        device half* d = (device half*)(st + u64(uint(snap)) * a.state_rs);
        for (uint i = 0; i < hist; ++i) d[i * ch + c] = half(h[i]);
      }
    }
  }
  for (uint i = 0; i < hist; ++i) live[i * ch + c] = half(h[i]);
}

/// gdn, step 2: the delta rule. Threadgroup (row group, v head h, sequence s) of GDN_SG simdgroups; each holds
/// R rows of h's matrix M in registers for the whole sequence (lane: entries 32 e + lane of each row).
/// Per row r of the sequence: q and k (head h % k_heads) l2-normalised, M = e^g M, d = b (v - M k),
/// M += d k^T, out = M q / sqrt(head_dim). The rows of M are independent given q and k, so the row groups of a
/// head never meet, and a row's math is the same for every R: R = 1 (MLX's gated_delta_step layout, 4x the
/// simdgroups) for decode and check steps, R = GDN_ROWS for prefill, where the redone q and k norms add up.
template <uint R>
kernel void vv_gdn(bytes_in qkv [[buffer(0)]], bytes_in alpha [[buffer(1)]], bytes_in beta [[buffer(2)]],
                   bytes_in dt_bias [[buffer(3)]], bytes_in a_log [[buffer(4)]], device uchar* st [[buffer(5)]],
                   bytes_in slots [[buffer(6)]], bytes_in lens [[buffer(7)]], bytes_in snaps [[buffer(8)]],
                   device uchar* out [[buffer(9)]], constant gdn_args& a [[buffer(10)]],
                   uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint E = GDN_MAX_HD / 32;
  const uint h = tg.y, s = tg.z, hd = a.hd, ne = hd / 32, kq = h % a.k_heads, j0 = (tg.x * GDN_SG + sg) * R;
  if (j0 >= hd) return;  // the whole simdgroup
  const uint r0 = gdn_first_row(lens, a, s), len = uint(max(at<int>(lens, a.lens_rs, s), 0));
  const u64 head = (u64(h) * hd + j0) * hd;  // floats from a slot's start to row j0 of h's matrix
  device float* live = (device float*)(st + u64(uint(at<int>(slots, a.slots_rs, s))) * a.state_rs) + head;
  float m[R][E];
#pragma unroll
  for (uint j = 0; j < R; ++j)
#pragma unroll
    for (uint e = 0; e < E; ++e) m[j][e] = e < ne ? live[j * hd + 32 * e + lane] : 0.0f;
  const float dt = at<float>(dt_bias, a.dt_rs, h), av = at<float>(a_log, a.a_rs, h);
  for (uint t = 0; t < len; ++t) {
    const uint r = r0 + t;
    device const half* row = (device const half*)(qkv + u64(r) * a.in_rs);
    float q[E], k[E], qq = 0, kk = 0;
#pragma unroll
    for (uint e = 0; e < E; ++e) {
      q[e] = e < ne ? float(row[kq * hd + 32 * e + lane]) : 0.0f;
      k[e] = e < ne ? float(row[(a.k_heads + kq) * hd + 32 * e + lane]) : 0.0f;
      qq += q[e] * q[e];
      kk += k[e] * k[e];
    }
    const float qi = 1.0f / sqrt(simd_sum(qq) + a.eps), ki = 1.0f / sqrt(simd_sum(kk) + a.eps);
#pragma unroll
    for (uint e = 0; e < E; ++e) q[e] *= qi, k[e] *= ki;
    const float x = at<float>(alpha + u64(r) * a.alpha_rs, 4, h) + dt;
    const float gate = exp((x > 20.0f ? x : log(1.0f + exp(x))) * av);
    const float b = 1.0f / (1.0f + exp(-at<float>(beta + u64(r) * a.beta_rs, 4, h)));
    float o[R];
#pragma unroll
    for (uint j = 0; j < R; ++j) {
      float mk = 0;
#pragma unroll
      for (uint e = 0; e < E; ++e) {
        m[j][e] *= gate;
        mk += m[j][e] * k[e];
      }
      const float d = (float(row[(2 * a.k_heads + h) * hd + j0 + j]) - simd_sum(mk)) * b;
      float mq = 0;
#pragma unroll
      for (uint e = 0; e < E; ++e) {
        m[j][e] += d * k[e];
        mq += m[j][e] * q[e];
      }
      o[j] = simd_sum(mq) * a.scale;
    }
    if (lane < R) {
      float v = o[0];
#pragma unroll
      for (uint j = 1; j < R; ++j) v = lane == j ? o[j] : v;
      ((device half*)(out + u64(r) * a.out_rs))[h * hd + j0 + lane] = half(v);
    }
    if (a.has_snaps) {
      const int snap = at<int>(snaps, a.snaps_rs, r);
      if (snap >= 0) {
        device float* d = (device float*)(st + u64(uint(snap)) * a.state_rs) + head;
#pragma unroll
        for (uint j = 0; j < R; ++j)
#pragma unroll
          for (uint e = 0; e < E; ++e)
            if (e < ne) d[j * hd + 32 * e + lane] = m[j][e];
      }
    }
  }
#pragma unroll
  for (uint j = 0; j < R; ++j)
#pragma unroll
    for (uint e = 0; e < E; ++e)
      if (e < ne) live[j * hd + 32 * e + lane] = m[j][e];
}

// Element col of chunk row i (sequence row r + i) of qkv, 0 past the chunk's n rows.
static inline float gdn_in(bytes_in qkv, u64 rs, uint r, uint i, uint n, uint col) {
  return i < n ? float(((device const half*)(qkv + u64(r + i) * rs))[col]) : 0.0f;
}
#define GDN_AT(T, i) (T).thread_elements()[i]
// exp of the log decays of rows lo .. hi of a chunk (1 for none), summed apiece: a difference of prefix sums
// would let a huge earlier decay round a later one away (-1e8 - log 2 is -1e8 in f32).
static inline float gdn_decay(threadgroup const float* gl, uint lo, uint hi) {
  float s = 0;
  for (uint i = lo; i <= hi; ++i) s += gl[i];
  return exp(s);
}

// The vv_gdn_chunk adaptation below includes material from Apple MLX.
/*
MIT License

Copyright © 2023 Apple Inc.

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
/// gdn for prefill, 8 rows at a time: MLX's gated_delta_fused_chunk (mlx/backend/metal/kernels/
/// gated_delta_update.h, Copyright 2023 Apple Inc., MIT), on larch's layout. Threadgroup (row group, v head
/// h, sequence s) of GDN_SG simdgroups; each holds 8 rows j0 .. of h's matrix M as HD / 8 transposed 8x8
/// simdgroup tiles (k index by row j). Per chunk of n <= 8 rows (a chunk also ends after a snapshot row, so every
/// snapshot sees its own row's state), with q and k l2-normalised, g the rows' log decays and G their prefix sums:
/// T = (I + strictly-lower(diag(b) K K^T))^-1 by its Neumann series (nilpotent: 2 squarings reach the 7th power),
/// delta = (G-decayed T) (b V) - e^G T (b K) M0^T, out = (G-decayed lower(Q K^T)) delta + e^G Q M0^T, then
/// M = e^G_n M0 + delta^T (e^(G_n - G) K). The same recurrence as vv_gdn, summed in another order.
template <uint HD>
kernel void vv_gdn_chunk(bytes_in qkv [[buffer(0)]], bytes_in alpha [[buffer(1)]], bytes_in beta [[buffer(2)]],
                         bytes_in dt_bias [[buffer(3)]], bytes_in a_log [[buffer(4)]], device uchar* st [[buffer(5)]],
                         bytes_in slots [[buffer(6)]], bytes_in lens [[buffer(7)]], bytes_in snaps [[buffer(8)]],
                         device uchar* out [[buffer(9)]], constant gdn_args& a [[buffer(10)]],
                         uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint C = 8, KT = HD / 8;
  threadgroup float vals[GDN_SG][4][C];  // per simdgroup: log decay, 1 / |k|, scale / |q| and b of the chunk's rows
  threadgroup float* G = vals[sg][0];
  threadgroup float* KI = vals[sg][1];
  threadgroup float* QI = vals[sg][2];
  threadgroup float* B = vals[sg][3];
  const uint h = tg.y, s = tg.z, kq = h % a.k_heads, j0 = (tg.x * GDN_SG + sg) * 8;
  if (j0 >= HD) return;  // the whole simdgroup
  const ushort fm = mma_fm(ushort(lane)), fn = mma_fn(ushort(lane));
  const uint qoff = kq * HD, koff = (a.k_heads + kq) * HD, voff = (2 * a.k_heads + h) * HD + j0;
  const uint r0 = gdn_first_row(lens, a, s), len = uint(max(at<int>(lens, a.lens_rs, s), 0));
  const u64 head = (u64(h) * HD + j0) * HD;
  device float* live = (device float*)(st + u64(uint(at<int>(slots, a.slots_rs, s))) * a.state_rs) + head;
  simdgroup_float8x8 S[KT];
#pragma clang loop unroll(full)
  for (uint kk = 0; kk < KT; ++kk) simdgroup_load(S[kk], live + 8 * kk, HD, ulong2(0, 0), true);
  simdgroup_float8x8 I = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  GDN_AT(I, 0) = fm == fn ? 1.0f : 0.0f;
  GDN_AT(I, 1) = fm == fn + 1 ? 1.0f : 0.0f;
  const float dt = at<float>(dt_bias, a.dt_rs, h), av = at<float>(a_log, a.a_rs, h);
  for (uint t = 0; t < len;) {
    const uint r = r0 + t;
    uint n = min(C, len - t);
    if (a.has_snaps) {
      const int snap = lane < n ? at<int>(snaps, a.snaps_rs, r + lane) : -1;
      const uint first = simd_min(snap >= 0 ? lane : C);
      if (first < n) n = first + 1;
    }
    // The rows' norms (all lanes, a row at a time), then lane i < n's log decay and b.
    float qi = 0, ki = 0, gl = 0, bl = 0;
    for (uint i = 0; i < n; ++i) {
      device const half* row = (device const half*)(qkv + u64(r + i) * a.in_rs);
      float qq = 0, kk = 0;
      for (uint e = lane; e < HD; e += 32) {
        const float q = float(row[qoff + e]), k = float(row[koff + e]);
        qq += q * q;
        kk += k * k;
      }
      qq = simd_sum(qq);
      kk = simd_sum(kk);
      if (lane == i) {
        qi = a.scale / sqrt(qq + a.eps);
        ki = 1.0f / sqrt(kk + a.eps);
      }
    }
    if (lane < n) {
      const float x = at<float>(alpha + u64(r + lane) * a.alpha_rs, 4, h) + dt;
      gl = (x > 20.0f ? x : log(1.0f + exp(x))) * av;
      bl = 1.0f / (1.0f + exp(-at<float>(beta + u64(r + lane) * a.beta_rs, 4, h)));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);  // the last chunk's reads are done
    if (lane < C) {
      G[lane] = gl;
      KI[lane] = ki;
      QI[lane] = qi;
      B[lane] = bl;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const float g_fm = gdn_decay(G, 0, fm), g_fmfn = gdn_decay(G, fn + 1, fm), g_fmfn1 = gdn_decay(G, fn + 2, fm);
    const float g_cfn = gdn_decay(G, fn + 1, C - 1), g_cfn1 = gdn_decay(G, fn + 2, C - 1);
    const float g_c = gdn_decay(G, 0, C - 1);
    const float b_fm = B[fm];
    simdgroup_float8x8 K, KTt, Q, V, W, U, WS, D, QK, O, KD, tmp, T, P;
    // K: rows by row of the chunk; KTt: the same transposed.
#define GDN_LOAD_K(kk)                                                                   \
  GDN_AT(K, 0) = gdn_in(qkv, a.in_rs, r, fm, n, koff + 8 * (kk) + fn) * KI[fm];          \
  GDN_AT(K, 1) = gdn_in(qkv, a.in_rs, r, fm, n, koff + 8 * (kk) + fn + 1) * KI[fm];
#define GDN_LOAD_KT(kk)                                                                  \
  GDN_AT(KTt, 0) = gdn_in(qkv, a.in_rs, r, fn, n, koff + 8 * (kk) + fm) * KI[fn];        \
  GDN_AT(KTt, 1) = gdn_in(qkv, a.in_rs, r, fn + 1, n, koff + 8 * (kk) + fm) * KI[fn + 1];
    simdgroup_float8x8 KK = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma clang loop unroll(full)
    for (uint kk = 0; kk < KT; ++kk) {
      GDN_LOAD_K(kk)
      GDN_LOAD_KT(kk)
      simdgroup_multiply_accumulate(KK, K, KTt, KK);
    }
    // P = strictly-lower(diag(b) K K^T); T = (I - P)(I + P^2)(I + P^4) = (I + P)^-1.
    GDN_AT(P, 0) = fn < fm ? b_fm * GDN_AT(KK, 0) : 0.0f;
    GDN_AT(P, 1) = fn + 1 < fm ? b_fm * GDN_AT(KK, 1) : 0.0f;
    GDN_AT(T, 0) = GDN_AT(I, 0) - GDN_AT(P, 0);
    GDN_AT(T, 1) = GDN_AT(I, 1) - GDN_AT(P, 1);
#pragma clang loop unroll(full)
    for (uint step = 1; (1u << step) < C; ++step) {
      simdgroup_multiply(P, P, P);
      simdgroup_multiply_accumulate(T, T, P, T);
    }
    WS = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma clang loop unroll(full)
    for (uint kk = 0; kk < KT; ++kk) {
      GDN_LOAD_K(kk)
      GDN_AT(K, 0) *= b_fm;
      GDN_AT(K, 1) *= b_fm;
      simdgroup_multiply(W, T, K);
      GDN_AT(W, 0) *= g_fm;
      GDN_AT(W, 1) *= g_fm;
      simdgroup_multiply_accumulate(WS, W, S[kk], WS);
    }
    GDN_AT(T, 0) *= fn > fm ? 0.0f : g_fmfn;
    GDN_AT(T, 1) *= fn + 1 > fm ? 0.0f : g_fmfn1;
    GDN_AT(V, 0) = gdn_in(qkv, a.in_rs, r, fm, n, voff + fn) * b_fm;
    GDN_AT(V, 1) = gdn_in(qkv, a.in_rs, r, fm, n, voff + fn + 1) * b_fm;
    simdgroup_multiply(U, T, V);
    GDN_AT(D, 0) = GDN_AT(U, 0) - GDN_AT(WS, 0);
    GDN_AT(D, 1) = GDN_AT(U, 1) - GDN_AT(WS, 1);
    tmp = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    QK = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma clang loop unroll(full)
    for (uint kk = 0; kk < KT; ++kk) {
      GDN_AT(Q, 0) = gdn_in(qkv, a.in_rs, r, fm, n, qoff + 8 * kk + fn) * QI[fm];
      GDN_AT(Q, 1) = gdn_in(qkv, a.in_rs, r, fm, n, qoff + 8 * kk + fn + 1) * QI[fm];
      GDN_LOAD_KT(kk)
      simdgroup_multiply_accumulate(QK, Q, KTt, QK);
      GDN_AT(Q, 0) *= g_fm;
      GDN_AT(Q, 1) *= g_fm;
      simdgroup_multiply_accumulate(tmp, Q, S[kk], tmp);
    }
    GDN_AT(QK, 0) *= fn > fm ? 0.0f : g_fmfn;
    GDN_AT(QK, 1) *= fn + 1 > fm ? 0.0f : g_fmfn1;
    simdgroup_multiply_accumulate(O, QK, D, tmp);
    if (fm < n) {
      device half* o = (device half*)(out + u64(r + fm) * a.out_rs) + h * HD + j0 + fn;
      o[0] = half(GDN_AT(O, 0));
      o[1] = half(GDN_AT(O, 1));
    }
#pragma clang loop unroll(full)
    for (uint kk = 0; kk < KT; ++kk) {
      GDN_LOAD_KT(kk)
      GDN_AT(KTt, 0) *= g_cfn;
      GDN_AT(KTt, 1) *= g_cfn1;
      simdgroup_multiply(KD, KTt, D);
      GDN_AT(S[kk], 0) = g_c * GDN_AT(S[kk], 0) + GDN_AT(KD, 0);
      GDN_AT(S[kk], 1) = g_c * GDN_AT(S[kk], 1) + GDN_AT(KD, 1);
    }
#undef GDN_LOAD_K
#undef GDN_LOAD_KT
    if (a.has_snaps) {
      const int snap = at<int>(snaps, a.snaps_rs, r + n - 1);
      if (snap >= 0) {
        device float* d = (device float*)(st + u64(uint(snap)) * a.state_rs) + head;
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < KT; ++kk) simdgroup_store(S[kk], d + 8 * kk, HD, ulong2(0, 0), true);
      }
    }
    t += n;
  }
#pragma clang loop unroll(full)
  for (uint kk = 0; kk < KT; ++kk) simdgroup_store(S[kk], live + 8 * kk, HD, ulong2(0, 0), true);
}
#undef GDN_AT

template [[host_name("vv_gdn")]] kernel decltype(vv_gdn<GDN_ROWS>) vv_gdn<GDN_ROWS>;
template [[host_name("vv_gdn1")]] kernel decltype(vv_gdn<1>) vv_gdn<1>;
template [[host_name("vv_gdn_chunk64")]] kernel decltype(vv_gdn_chunk<64>) vv_gdn_chunk<64>;
template [[host_name("vv_gdn_chunk128")]] kernel decltype(vv_gdn_chunk<128>) vv_gdn_chunk<128>;

// out = a + b, out may be a. rs0 a, rs1 b, rs2 out.
kernel void vv_add(bytes_in a_ [[buffer(0)]], bytes_in b [[buffer(1)]], device uchar* out [[buffer(2)]],
                   constant ew_args& a [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  const float x = ((device const float*)(a_ + g.y * a.rs0))[g.x];
  ((device float*)(out + g.y * a.rs2))[g.x] = x + float(((device const half*)(b + g.y * a.rs1))[g.x]);
}

/// draft_conv, DFlash2's dynamic conv (larch.h): thread (channel c, block row r), in the reference's f32 order; an
/// f32 out (the residual) is added into, an f16 one written.
kernel void vv_draft_conv(bytes_in x [[buffer(0)]], bytes_in dyn [[buffer(1)]], bytes_in base [[buffer(2)]],
                          device uchar* out [[buffer(3)]], constant draft_conv_args& a [[buffer(4)]],
                          uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, r = g.y, k = c / a.group;
  device const float* dr = (device const float*)(dyn + u64(r) * a.dyn_rs);
  device const float* b0 = (device const float*)(base + u64(a.t0) * a.base_rs);
  device const float* b1 = (device const float*)(base + u64(a.t0 + 1) * a.base_rs);
  float v = a.f32_out ? ((device const float*)(out + u64(r) * a.out_rs))[c] : 0.0f;
  v += float(((device const half*)(x + u64(r) * a.x_rs))[c]) * (b0[c] + dr[a.t0 * a.groups + k]);
  if (r > 0) v += float(((device const half*)(x + u64(r - 1) * a.x_rs))[c]) * (b1[c] + dr[(a.t0 + 1) * a.groups + k]);
  if (a.f32_out)
    ((device float*)(out + u64(r) * a.out_rs))[c] = v;
  else
    ((device half*)(out + u64(r) * a.out_rs))[c] = half(v);
}

// Row r of k and v into slot slots[r] of the pool's layer (dense, cols = n_kv_heads * head_dim per slot);
// a negative slot skips the row. rs0 k, rs1 v, rs2 slots.
kernel void vv_kv_write(bytes_in k [[buffer(0)]], bytes_in v [[buffer(1)]], bytes_in slots [[buffer(2)]],
                        device half* kp [[buffer(3)]], device half* vp [[buffer(4)]], constant ew_args& a [[buffer(5)]],
                        uint2 g [[thread_position_in_grid]]) {
  const int s = at<int>(slots, a.rs2, g.y);
  if (s < 0) return;
  const u64 i = u64(s) * a.cols + g.x;
  kp[i] = ((device const half*)(k + g.y * a.rs0))[g.x];
  vp[i] = ((device const half*)(v + g.y * a.rs1))[g.x];
}

// Lane l's accumulator slot i (0..7) holds dim acc_dim(i, l) of its head: 2 chunks of 128 dims, 4
// contiguous dims per lane, so a position's v row is one 4-wide load per lane.
inline uint acc_dim(uint i, uint lane) { return 128 * (i / 4) + 4 * lane + i % 4; }

// The online softmax of up to ATTN_GROUP q heads that share kv head kvh (GQA) over positions [first, last)
// of one row's block table, 32 positions per step. Lane j scores position j for every head on its own (q
// from threadgroup memory, k summed in d order like the CPU reference), so each k and v row is read once
// for the group, and a step needs one simd_max and one simd_sum per head, not one per position. Head g's
// accumulator is acc[8 g .. 8 g + 7]. Loops over heads and slots have constant bounds, so every acc index
// is a constant and acc stays in registers.
// ponytail: always ATTN_GROUP heads (unused ones have q = 0): multi-head attention scores 4x and 3 or 7 q
// heads per kv head waste a quarter; give the group width a pipeline each when such a model's decode
// profile shows attention.
void attend_range(device const half* kp, device const half* vp, device const int* table, constant attn_args& a,
                  uint kvh, uint lane, threadgroup const half* qs, uint first, uint last, thread float* mx,
                  thread float* sum, thread float* acc) {
  for (uint p0 = first; p0 < last; p0 += 32) {
    const uint p = p0 + lane, n = min(32u, last - p0);
    int block = 0;
    float s[ATTN_GROUP];
#pragma unroll
    for (uint g = 0; g < ATTN_GROUP; ++g) s[g] = -INFINITY;
    if (lane < n) {
      block = table[p / a.block_tokens];
      device const half* kr =
          kp + ((u64(uint(block)) * a.block_tokens + p % a.block_tokens) * a.kv_heads + kvh) * a.hd;
      float dot[ATTN_GROUP] = {};
      uint d = 0;
#pragma unroll 4
      for (; d + 4 <= a.hd; d += 4) {  // packed: larch.h promises the pool 2-byte alignment only
        const float4 k = float4(half4(*(device const packed_half4*)(kr + d)));
#pragma unroll
        for (uint g = 0; g < ATTN_GROUP; ++g) {
          const float4 q = float4(*(threadgroup const half4*)(qs + ATTN_MAX_HEAD_DIM * g + d));
          dot[g] += q.x * k.x;
          dot[g] += q.y * k.y;
          dot[g] += q.z * k.z;
          dot[g] += q.w * k.w;
        }
      }
      for (; d < a.hd; ++d) {
        const float k = float(kr[d]);
#pragma unroll
        for (uint g = 0; g < ATTN_GROUP; ++g) dot[g] += float(qs[ATTN_MAX_HEAD_DIM * g + d]) * k;
      }
#pragma unroll
      for (uint g = 0; g < ATTN_GROUP; ++g) s[g] = dot[g] * a.scale;
    }
    float e[ATTN_GROUP];
#pragma unroll
    for (uint g = 0; g < ATTN_GROUP; ++g) {
      const float nmx = max(mx[g], simd_max(s[g])), corr = exp(mx[g] - nmx);
      e[g] = lane < n ? exp(s[g] - nmx) : 0.0f;
      sum[g] = sum[g] * corr + simd_sum(e[g]);
#pragma unroll
      for (uint i = 0; i < 8; ++i) acc[8 * g + i] *= corr;
      mx[g] = nmx;
    }
#pragma unroll 4
    for (uint j = 0; j < n; ++j) {  // unrolled so several positions' v loads are in flight at once
      float ej[ATTN_GROUP];
#pragma unroll
      for (uint g = 0; g < ATTN_GROUP; ++g) ej[g] = simd_shuffle(e[g], ushort(j));
      const u64 slot = u64(uint(simd_shuffle(block, ushort(j)))) * a.block_tokens + (p0 + j) % a.block_tokens;
      device const half* vr = vp + (slot * a.kv_heads + kvh) * a.hd;
#pragma unroll
      for (uint c = 0; c < 2; ++c) {
        const uint d = acc_dim(4 * c, lane);
        if (d + 4 <= a.hd) {
          const float4 v = float4(half4(*(device const packed_half4*)(vr + d)));
#pragma unroll
          for (uint g = 0; g < ATTN_GROUP; ++g) {
            acc[8 * g + 4 * c] += ej[g] * v.x;
            acc[8 * g + 4 * c + 1] += ej[g] * v.y;
            acc[8 * g + 4 * c + 2] += ej[g] * v.z;
            acc[8 * g + 4 * c + 3] += ej[g] * v.w;
          }
        } else {
#pragma unroll
          for (uint t = 0; t < 4; ++t) {
            if (d + t >= a.hd) break;
            const float v = float(vr[d + t]);
#pragma unroll
            for (uint g = 0; g < ATTN_GROUP; ++g) acc[8 * g + 4 * c + t] += ej[g] * v;
          }
        }
      }
    }
  }
}

// The simdgroup's q heads h0 .. h0 + ng - 1 of row r in threadgroup memory, ATTN_MAX_HEAD_DIM halves per
// head (q is f16, so nothing is rounded); the unused heads are zero.
void load_q(bytes_in q, constant attn_args& a, uint r, uint h0, uint ng, uint lane, threadgroup half* qs) {
  device const half* qr = (device const half*)(q + r * a.q_rs) + h0 * a.hd;
  for (uint g = 0; g < ATTN_GROUP; ++g)
    for (uint d = lane; d < a.hd; d += 32) qs[ATTN_MAX_HEAD_DIM * g + d] = g < ng ? qr[g * a.hd + d] : half(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Simdgroup unit's kv head and q heads: a kv head's q heads in groups of ATTN_GROUP.
struct head_group {
  uint kvh, h0, ng;
  head_group(constant attn_args& a, uint unit) {
    const uint per_kv = a.heads / a.kv_heads, groups = (per_kv + ATTN_GROUP - 1) / ATTN_GROUP;
    kvh = unit / groups;
    h0 = kvh * per_kv + unit % groups * ATTN_GROUP;
    ng = min(uint(ATTN_GROUP), kvh * per_kv + per_kv - h0);
  }
};

// Split-K's share of (row r, q head hh, split): max, sum, then head_dim f32 accumulators.
device float* share(device float* partial, constant attn_args& a, uint r, uint hh, uint split) {
  return partial + ((u64(r) * a.heads + hh) * a.splits + split) * a.partial_stride;
}

// Attention, one-split decode and the prefill attn_flash does not take (backend.mm attend): one simdgroup per
// (head group, row).
kernel void vv_attn(bytes_in q [[buffer(0)]], device const half* kp [[buffer(1)]], device const half* vp [[buffer(2)]],
                    bytes_in bt [[buffer(3)]], bytes_in lens [[buffer(4)]], bytes_in start [[buffer(5)]],
                    device uchar* out [[buffer(6)]], constant attn_args& a [[buffer(7)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const head_group h(a, tg.x);
  const uint r = tg.y;
  uint s = r;
  int n;
  if (a.prefill) {  // row r is row r - first of sequence s, at position start[s] + r - first
    uint first = 0;
    for (s = 0; s + 1 < a.seqs && r >= first + uint(at<int>(lens, a.lens_rs, s)); ++s) first += at<int>(lens, a.lens_rs, s);
    n = at<int>(start, a.start_rs, s) + int(r - first) + 1;
  } else {
    n = at<int>(lens, a.lens_rs, r);
  }
  threadgroup half4 qs[ATTN_GROUP * ATTN_MAX_HEAD_DIM / 4];  // half4: attend_range reads q 4-wide
  float acc[8 * ATTN_GROUP] = {}, mx[ATTN_GROUP], sum[ATTN_GROUP] = {};
  for (uint g = 0; g < ATTN_GROUP; ++g) mx[g] = -INFINITY;
  load_q(q, a, r, h.h0, h.ng, lane, (threadgroup half*)qs);
  attend_range(kp, vp, (device const int*)(bt + s * a.bt_rs), a, h.kvh, lane, (threadgroup const half*)qs, 0,
               uint(max(n, 0)), mx, sum, acc);
#pragma unroll
  for (uint g = 0; g < ATTN_GROUP; ++g) {
    if (g >= h.ng) break;
    device half* o = (device half*)(out + r * a.out_rs) + (h.h0 + g) * a.hd;
#pragma unroll
    for (uint i = 0; i < 8; ++i)
      if (const uint d = acc_dim(i, lane); d < a.hd) o[d] = half(n > 0 ? acc[8 * g + i] / sum[g] : 0.0f);
  }
}

// Prefill attention, flash style (FA2), on the tensor API. Threadgroup (tile, q head) takes ATTN_FLASH_ROWS
// consecutive q rows of one sequence (staged once as f16) and walks the positions they see one pool block
// (ATTN_FLASH_BLOCK slots, which for one kv head are kv_heads * hd halves apart) at a time, reading K and V in
// place: S = Q K^T and O += P V are MPP matmul2d with f32 accumulate, and the online softmax between them is f32
// through threadgroup memory (P is rounded to f16 for its multiply, which ops.cpp's attn_prefill tolerance allows).
// DK is head_dim rounded up to 64, 128 or 256 (in tiles of ATTN_FLASH_ROWS256 rows). Tile x counts the sequences'
// row tiles in order; the host launches an upper bound, and the tiles past the last return at once.
template <int DK>
kernel void attn_flash(bytes_in q [[buffer(0)]], device half* kp [[buffer(1)]], device half* vp [[buffer(2)]],
                       bytes_in bt [[buffer(3)]], bytes_in lens [[buffer(4)]], bytes_in start [[buffer(5)]],
                       device uchar* out [[buffer(6)]], constant attn_args& a [[buffer(7)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  // Q rows LQ halves apart, S rows LS floats, P rows LP halves; TPR threads share a softmax row.
  constexpr uint BR = DK > 128 ? ATTN_FLASH_ROWS256 : ATTN_FLASH_ROWS, BT = ATTN_FLASH_BLOCK, LQ = DK + 8;
  constexpr uint LS = BT + 1, LP = BT + 8, NT = ATTN_FLASH_SG * 32, TPR = NT / BR, CPT = BT / TPR;
  static_assert(NT % BR == 0 && BT % TPR == 0 && TPR <= 32, "a softmax row's threads are lanes of one simdgroup");
  uint t = tg.x, s = 0, first = 0, len = 0;
  for (; s < a.seqs; ++s, first += len, t -= (len + BR - 1) / BR) {
    len = uint(max(at<int>(lens, a.lens_rs, s), 0));
    if (t < (len + BR - 1) / BR) break;
  }
  // Past the last tile, or past q's rows: lens that sum past them are a caller bug the host cannot see.
  if (s == a.seqs || first + t * BR >= a.q_rows) return;  // the whole threadgroup
  const uint j0 = t * BR, rows = min(min(BR, len - j0), a.q_rows - first - j0);
  const uint h = tg.y, kvh = h / (a.heads / a.kv_heads);
  const uint pos0 = uint(at<int>(start, a.start_rs, s)) + j0, end = pos0 + rows;  // the tile sees [0, end)
  device const int* table = (device const int*)(bt + s * a.bt_rs);

  threadgroup half4 qs4[BR * LQ / 4];
  threadgroup half* qs = (threadgroup half*)qs4;
  threadgroup float ss[BR * LS], rescale[BR];
  threadgroup half ps[BR * LP];
  threadgroup uint moved[2];  // by step parity: some row's max moved, so O needs its rescale
  if (tid == 0) moved[0] = moved[1] = 0;
  for (uint i = tid; i < BR * DK / 4; i += NT) {  // zero past the tile's rows and head_dim (a multiple of 8)
    const uint r = i / (DK / 4), d = i % (DK / 4) * 4;
    qs4[r * LQ / 4 + d / 4] = r < rows && d < a.hd
        ? half4(*(device const packed_half4*)((device const half*)(q + u64(first + j0 + r) * a.q_rs) + h * a.hd + d))
        : half4(0);
  }

  using dev_t = tensor<device half, dextents<int, 2>, tensor_inline>;
  using tg_t = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
  constexpr auto qk_desc = matmul2d_descriptor(BR, BT, static_cast<int>(dynamic_extent), false, true, false,
                                               matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto pv_desc = matmul2d_descriptor(BR, DK, static_cast<int>(dynamic_extent), false, false, false,
                                               matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<ATTN_FLASH_SG>> qk;
  matmul2d<pv_desc, execution_simdgroups<ATTN_FLASH_SG>> pv;
  const tg_t qt(qs, dextents<int, 2>(int(a.hd), int(BR)), array<int, 2>({1, int(LQ)}));
  const int slot_rs = int(a.kv_heads * a.hd);  // halves between a kv head's consecutive slots
  auto o = pv.template get_destination_cooperative_tensor<tg_t, dev_t, float>();
#pragma unroll
  for (uint16_t i = 0; i < o.get_capacity(); ++i)
    if (o.is_valid_element(i)) o[i] = 0.0f;
  // Thread (r, part) runs the softmax of row r, columns CPT part ..; a dead row (past the tile's rows) scores
  // 0 everywhere, which keeps its softmax finite whatever S holds there.
  const uint r = tid / TPR, part = tid % TPR, pos = pos0 + r;
  const bool live = r < rows;
  float m = -INFINITY, l = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint p0 = 0, step = 0; p0 < end; p0 += BT, ++step) {
    const uint n = min(BT, end - p0);
    const u64 block = (u64(uint(table[p0 / BT])) * BT * a.kv_heads + kvh) * a.hd;
    const dev_t kt(kp + block, dextents<int, 2>(int(a.hd), int(n)), array<int, 2>({1, slot_rs}));
    auto sc = qk.template get_destination_cooperative_tensor<tg_t, dev_t, float>();
#pragma unroll
    for (uint16_t i = 0; i < sc.get_capacity(); ++i)
      if (sc.is_valid_element(i)) sc[i] = 0.0f;
    qk.run(qt, kt, sc);
#pragma unroll
    for (uint16_t i = 0; i < sc.get_capacity(); ++i)
      if (sc.is_valid_element(i)) {
        const auto at_i = sc.get_multidimensional_index(i);
        ss[at_i[1] * LS + at_i[0]] = sc[i];
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) moved[(step + 1) % 2] = 0;  // the last step's reads of it are done
    float x[CPT], mx = -INFINITY;
#pragma unroll
    for (uint c = 0; c < CPT; ++c) {  // columns past n are past end, so past every row's position
      const uint col = CPT * part + c;
      x[c] = !live ? 0.0f : p0 + col <= pos ? ss[r * LS + col] * a.scale : -INFINITY;
      mx = max(mx, x[c]);
    }
    for (ushort o_ = 1; o_ < TPR; o_ *= 2) mx = max(mx, simd_shuffle_xor(mx, o_));
    const float mn = max(m, mx), corr = exp(m - mn);  // a live row sees position 0 on the first step: mn is finite
    float sum = 0;
#pragma unroll
    for (uint c = 0; c < CPT; ++c) {
      const float e = exp(x[c] - mn);
      sum += e;
      ps[r * LP + CPT * part + c] = half(e);
    }
    for (ushort o_ = 1; o_ < TPR; o_ *= 2) sum += simd_shuffle_xor(sum, o_);
    l = l * corr + sum;
    m = mn;
    if (part == 0) rescale[r] = corr;
    if (corr != 1.0f) moved[step % 2] = 1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (moved[step % 2]) {  // the same for the whole threadgroup; rarer as the rows' maxima settle
#pragma unroll
      for (uint16_t i = 0; i < o.get_capacity(); ++i)
        if (o.is_valid_element(i)) o[i] *= rescale[o.get_multidimensional_index(i)[1]];
    }
    const tg_t pt(ps, dextents<int, 2>(int(n), int(BR)), array<int, 2>({1, int(LP)}));
    const dev_t vt(vp + block, dextents<int, 2>(int(a.hd), int(n)), array<int, 2>({1, slot_rs}));
    pv.run(pt, vt, o);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);  // every thread's last read of rescale is done before l
  if (part == 0) rescale[r] = l;
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (uint16_t i = 0; i < o.get_capacity(); ++i) {
    if (!o.is_valid_element(i)) continue;
    const auto at_i = o.get_multidimensional_index(i);
    const uint row = uint(at_i[1]), d = uint(at_i[0]);
    if (row < rows && d < a.hd)
      ((device half*)(out + u64(first + j0 + row) * a.out_rs))[h * a.hd + d] = half(o[i] / rescale[row]);
  }
}
template [[host_name("vv_attn_flash64")]] kernel decltype(attn_flash<64>) attn_flash<64>;
template [[host_name("vv_attn_flash128")]] kernel decltype(attn_flash<128>) attn_flash<128>;
template [[host_name("vv_attn_flash256")]] kernel decltype(attn_flash<256>) attn_flash<256>;

// Prefill attention, flash style, on simdgroup matrices: attn_flash for the GPUs without the tensor API (M1 to M4,
// or LARCH_METAL_TENSOR=0), where vv_attn re-reads every earlier K and V row for each q row. Same tiles and grid:
// threadgroup (tile, q head) takes ATTN_FLASH_ROWS q rows of one sequence. Simdgroup g owns rows 8 g .. 8 g + 7 and
// keeps their Q (f16) and O (f32) in 8x8 matrices. Per pool block the threads stage its K and V slots in
// threadgroup memory, zero past the last slot the tile sees and past head_dim (so P's zeros never meet garbage
// rows), then each simdgroup on its own: S = Q K^T, the online softmax of its rows (4 lanes a row), O = diag(corr) O
// + P V. P is rounded to f16 for its multiply, as attn_flash does. Our CUDA attn_prefill_tc_k's shape. At DK 256
// (Qwen 3.5's full attention) K and V take turns in one block's space and Q's tile loads in two halves, which
// keeps the threadgroup memory near 16 KB.
template <int DK>
kernel void attn_flash_sg(bytes_in q [[buffer(0)]], device const half* kp [[buffer(1)]], device const half* vp [[buffer(2)]],
                          bytes_in bt [[buffer(3)]], bytes_in lens [[buffer(4)]], bytes_in start [[buffer(5)]],
                          device uchar* out [[buffer(6)]], constant attn_args& a [[buffer(7)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint BR = ATTN_FLASH_ROWS, BT = ATTN_FLASH_BLOCK, L = DK + 8, LS = BT + 4, LP = BT + 8, D8 = DK / 8;
  constexpr uint NT = ATTN_FLASH_SG * 32;
  static_assert(BR == 8 * ATTN_FLASH_SG && BT == 16 && DK % 8 == 0, "a simdgroup's 8 rows, 4 lanes a row");
  constexpr bool SPLIT = DK > 128;                          // K, then V, through one block's space
  constexpr uint KVN = SPLIT ? BT : 2 * BT, QPASS = SPLIT ? 2 : 1, QR = BR / QPASS;  // Q's rows per pass
  uint t = tg.x, s = 0, first = 0, len = 0;
  for (; s < a.seqs; ++s, first += len, t -= (len + BR - 1) / BR) {
    len = uint(max(at<int>(lens, a.lens_rs, s), 0));
    if (t < (len + BR - 1) / BR) break;
  }
  if (s == a.seqs || first + t * BR >= a.q_rows) return;  // the whole threadgroup
  const uint j0 = t * BR, rows = min(min(BR, len - j0), a.q_rows - first - j0);
  const uint h = tg.y, kvh = h / (a.heads / a.kv_heads);
  const uint pos0 = uint(at<int>(start, a.start_rs, s)) + j0, end = pos0 + rows;  // the tile sees [0, end)
  device const int* table = (device const int*)(bt + s * a.bt_rs);

  threadgroup half4 kv4[KVN * L / 4];  // a block's K slots then (not SPLIT) its V slots; Q's tile before the first
  threadgroup half* kvs = (threadgroup half*)kv4;
  threadgroup float sb[ATTN_FLASH_SG * 8 * LS];  // a simdgroup's 8 rows of S, then of O on the way out
  threadgroup half pb[ATTN_FLASH_SG * 8 * LP];   // a simdgroup's 8 rows of P
  threadgroup float db[ATTN_FLASH_SG * 64];      // a simdgroup's diag(corr), then diag(1 / l)
  threadgroup float* ss = sb + sg * 8 * LS;
  threadgroup half* ps = pb + sg * 8 * LP;
  threadgroup float* dg = db + sg * 64;
  static_assert(QR <= KVN, "a pass of Q's tile fits the K and V space");
  simdgroup_half8x8 qm[D8];
  simdgroup_float8x8 om[D8];
  for (uint pass = 0; pass < QPASS; ++pass) {
    if (pass) threadgroup_barrier(mem_flags::mem_threadgroup);  // the last pass's simdgroups loaded their rows
    for (uint i = tid; i < QR * DK / 4; i += NT) {  // zero past the tile's rows and head_dim (a multiple of 8)
      const uint rp = i / (DK / 4), r = pass * QR + rp, d = i % (DK / 4) * 4;
      half4 v = half4(0);
      if (r < rows && d < a.hd)
        v = half4(*(device const packed_half4*)((device const half*)(q + u64(first + j0 + r) * a.q_rs) + h * a.hd + d));
      kv4[(rp * L + d) / 4] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg / (ATTN_FLASH_SG / QPASS) == pass)
      for (uint d = 0; d < D8; ++d) simdgroup_load(qm[d], kvs + (8 * sg - pass * QR) * L + 8 * d, L);
  }
  for (uint i = lane; i < 64; i += 32) dg[i] = 0.0f;
  for (uint d = 0; d < D8; ++d) om[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  // Lane (rr, part) runs the softmax of row 8 sg + rr, columns 4 part .. 4 part + 3; a dead row (past the tile's
  // rows) scores 0 everywhere, which keeps its softmax finite.
  const uint rr = lane / 4, part = lane % 4, pos = pos0 + 8 * sg + rr;
  const bool live = 8 * sg + rr < rows;
  const uint slot_rs = a.kv_heads * a.hd;  // halves between a kv head's consecutive slots
  float m = -INFINITY, l = 0;
  for (uint p0 = 0; p0 < end; p0 += BT) {
    const uint n = min(BT, end - p0);
    const u64 block = (u64(uint(table[p0 / BT])) * BT * a.kv_heads + kvh) * a.hd;
    threadgroup_barrier(mem_flags::mem_threadgroup);  // Q loaded, or every simdgroup done with the last block
    for (uint i = tid; i < BT * DK / 4; i += NT) {
      const uint j = i / (DK / 4), d = i % (DK / 4) * 4;
      half4 k = half4(0), v = half4(0);
      if (j < n && d < a.hd) {
        k = half4(*(device const packed_half4*)(kp + block + u64(j) * slot_rs + d));
        if (!SPLIT) v = half4(*(device const packed_half4*)(vp + block + u64(j) * slot_rs + d));
      }
      kv4[(j * L + d) / 4] = k;
      if (!SPLIT) kv4[((BT + j) * L + d) / 4] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), s1 = s0;
    for (uint d = 0; d < D8; ++d) {
      simdgroup_half8x8 k0, k1;
      simdgroup_load(k0, kvs + 8 * d, L, ulong2(0, 0), true);  // K^T: rows d, columns slots 0-7
      simdgroup_load(k1, kvs + 8 * L + 8 * d, L, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s0, qm[d], k0, s0);
      simdgroup_multiply_accumulate(s1, qm[d], k1, s1);
    }
    simdgroup_store(s0, ss, LS);
    simdgroup_store(s1, ss + 8, LS);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float x[4], mx = -INFINITY;
    for (uint c = 0; c < 4; ++c) {  // columns past n are past end, so past every row's position
      const uint col = 4 * part + c;
      x[c] = !live ? 0.0f : p0 + col <= pos ? ss[rr * LS + col] * a.scale : -INFINITY;
      mx = max(mx, x[c]);
    }
    mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
    const float mn = max(m, mx), corr = exp(m - mn);  // a live row sees position 0 on the first block: mn is finite
    float sum = 0;
    for (uint c = 0; c < 4; ++c) {
      const float e = exp(x[c] - mn);
      sum += e;
      ps[rr * LP + 4 * part + c] = half(e);
    }
    sum += simd_shuffle_xor(sum, ushort(1));
    sum += simd_shuffle_xor(sum, ushort(2));
    l = l * corr + sum;
    m = mn;
    if (part == 0) dg[rr * 9] = corr;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (!simd_all(corr == 1.0f)) {  // rarer as the rows' maxima settle
      simdgroup_float8x8 dm;
      simdgroup_load(dm, dg, 8);
      for (uint d = 0; d < D8; ++d) simdgroup_multiply(om[d], dm, om[d]);
    }
    if (SPLIT) {  // every simdgroup's S is done with K: V takes its place
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < BT * DK / 4; i += NT) {
        const uint j = i / (DK / 4), d = i % (DK / 4) * 4;
        half4 v = half4(0);
        if (j < n && d < a.hd) v = half4(*(device const packed_half4*)(vp + block + u64(j) * slot_rs + d));
        kv4[(j * L + d) / 4] = v;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup const half* vs = kvs + (SPLIT ? 0 : BT * L);
    simdgroup_half8x8 pm0, pm1;
    simdgroup_load(pm0, ps, LP);
    simdgroup_load(pm1, ps + 8, LP);
    for (uint d = 0; d < D8; ++d) {
      simdgroup_half8x8 v0, v1;
      simdgroup_load(v0, vs + 8 * d, L);
      simdgroup_load(v1, vs + 8 * L + 8 * d, L);
      simdgroup_multiply_accumulate(om[d], pm0, v0, om[d]);
      simdgroup_multiply_accumulate(om[d], pm1, v1, om[d]);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);  // every lane's last read of dg
  if (part == 0) dg[rr * 9] = 1.0f / l;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 dm;
  simdgroup_load(dm, dg, 8);
  for (uint d = 0; d < D8; ++d) {
    simdgroup_multiply(om[d], dm, om[d]);
    simdgroup_store(om[d], ss, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 64; i += 32) {
      const uint row = 8 * sg + i / 8, dd = 8 * d + i % 8;
      if (row < rows && dd < a.hd) ((device half*)(out + u64(first + j0 + row) * a.out_rs))[h * a.hd + dd] = half(ss[i]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
}
template [[host_name("vv_attn_flash_sg64")]] kernel decltype(attn_flash_sg<64>) attn_flash_sg<64>;
template [[host_name("vv_attn_flash_sg128")]] kernel decltype(attn_flash_sg<128>) attn_flash_sg<128>;
template [[host_name("vv_attn_flash_sg256")]] kernel decltype(attn_flash_sg<256>) attn_flash_sg<256>;

// Split-K decode, 1 of 2: simdgroup (split, head group, row) writes each head's share. With a.prefill, a causal span of
// larch_attn (a check step's rows): row r is row j of sequence s, attending to start[s] + j + 1 positions, as vv_attn's
// prefill finds it, so it is decode's row with that seq_len, bit for bit (larch.h's verify contract).
kernel void vv_attn_split(bytes_in q [[buffer(0)]], device const half* kp [[buffer(1)]], device const half* vp [[buffer(2)]],
                          bytes_in bt [[buffer(3)]], bytes_in lens [[buffer(4)]], bytes_in start [[buffer(5)]],
                          device float* partial [[buffer(6)]], constant attn_args& a [[buffer(7)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint2 tgs [[threadgroups_per_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  // Split-major order: the live splits' threadgroups come first and contiguous, as at 16 splits. Head-group-major
  // put 48 empty ones between each head group's 16 live ones at 2k positions (+23 %, M5, 1 Nov), which this
  // order is meant to remove.
  const uint groups = tgs.x / a.splits, split = tg.x / groups, r = tg.y;
  const head_group h(a, tg.x % groups);
  uint s = r, n = max(0, at<int>(lens, a.lens_rs, r));
  if (a.prefill) {
    uint first = 0;
    for (s = 0; s + 1 < a.seqs && r >= first + uint(at<int>(lens, a.lens_rs, s)); ++s) first += at<int>(lens, a.lens_rs, s);
    n = uint(at<int>(start, a.start_rs, s)) + r - first + 1;
  }
  const uint used = min(a.splits, max(16u, n / ATTN_SPLIT_POS));  // the rest stay empty
  const uint first = split < used ? uint(u64(n) * split / used) : n;
  const uint last = split < used ? uint(u64(n) * (split + 1) / used) : n;
  if (first == last) {  // the same for the whole threadgroup: an empty share is sum 0, which the reduce skips
    for (uint g = 0; g < h.ng && lane == 0; ++g) share(partial, a, r, h.h0 + g, split)[1] = 0.0f;
    return;
  }
  threadgroup half4 qs[ATTN_GROUP * ATTN_MAX_HEAD_DIM / 4];
  float acc[8 * ATTN_GROUP] = {}, mx[ATTN_GROUP], sum[ATTN_GROUP] = {};
  for (uint g = 0; g < ATTN_GROUP; ++g) mx[g] = -INFINITY;
  load_q(q, a, r, h.h0, h.ng, lane, (threadgroup half*)qs);
  attend_range(kp, vp, (device const int*)(bt + s * a.bt_rs), a, h.kvh, lane, (threadgroup const half*)qs, first,
               last, mx, sum, acc);
#pragma unroll
  for (uint g = 0; g < ATTN_GROUP; ++g) {
    if (g >= h.ng) break;
    device float* dst = share(partial, a, r, h.h0 + g, split);
    if (lane == 0) {
      dst[0] = mx[g];
      dst[1] = sum[g];
    }
#pragma unroll
    for (uint i = 0; i < 8; ++i)
      if (const uint d = acc_dim(i, lane); d < a.hd) dst[2 + d] = acc[8 * g + i];
  }
}

// Split-K decode, 2 of 2: merges a (q head, row)'s splits, skipping empty ones (sum 0). Lane l weighs splits l and
// l + 32 once (splits <= ATTN_MAX_SPLITS, which the host keeps); each output dim then sums the live splits'
// accumulators with those weights.
kernel void vv_attn_split_reduce(device const float* partial [[buffer(0)]], device uchar* out [[buffer(1)]],
                                 constant attn_args& a [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const uint hh = tg.x, r = tg.y;
  device const float* base = partial + (u64(r) * a.heads + hh) * a.splits * a.partial_stride;
  const bool live = lane < a.splits && base[lane * a.partial_stride + 1] != 0.0f;
  const bool live1 = lane + 32 < a.splits && base[(lane + 32) * a.partial_stride + 1] != 0.0f;
  const float m = live ? base[lane * a.partial_stride] : -INFINITY;
  const float z = live ? base[lane * a.partial_stride + 1] : 0.0f;
  const float m1 = live1 ? base[(lane + 32) * a.partial_stride] : -INFINITY;
  const float z1 = live1 ? base[(lane + 32) * a.partial_stride + 1] : 0.0f;
  const float mx = simd_max(max(m, m1)), w = live ? exp(m - mx) : 0.0f, w1 = live1 ? exp(m1 - mx) : 0.0f;
  const float total = simd_sum(w * z + w1 * z1);  // w1 z1 is 0 up to 32 splits: the same sum as one a lane
  float accum[ATTN_MAX_HEAD_DIM / 32] = {};
  for (uint split = 0; split < a.splits; ++split) {
    const float ws = simd_shuffle(split < 32 ? w : w1, ushort(split % 32));
    if (ws == 0.0f) continue;  // the same for every lane
    device const float* src = base + split * a.partial_stride + 2;
#pragma unroll
    for (uint i = 0; i < ATTN_MAX_HEAD_DIM / 32; ++i)
      if (lane + 32 * i < a.hd) accum[i] += ws * src[lane + 32 * i];
  }
  device half* output = (device half*)(out + r * a.out_rs) + hh * a.hd;
#pragma unroll
  for (uint i = 0; i < ATTN_MAX_HEAD_DIM / 32; ++i)
    if (lane + 32 * i < a.hd) output[lane + 32 * i] = half(total != 0.0f ? accum[i] / total : 0.0f);
}

// sample_prep with a penalty, 1 of 3: scratch row r = logits row r, which the penalty then edits.
kernel void vv_sp_copy(bytes_in lg [[buffer(0)]], device float* scratch [[buffer(1)]],
                       constant sample_args& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  scratch[u64(g.y) * a.vocab + g.x] = ((device const float*)(lg + g.y * a.lg_rs))[g.x];
}

// 2 of 3, thread (i, r): llama.cpp's repetition penalty for window entry i of row r, applied by the
// entry holding the id's first occurrence, so once per distinct id.
kernel void vv_sp_penalty(bytes_in lg [[buffer(0)]], bytes_in pen [[buffer(1)]],
                          device const sample_row* rows [[buffer(2)]], device float* scratch [[buffer(3)]],
                          constant sample_args& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  const sample_row p = rows[g.y];
  if (g.x >= p.penalty_window || p.repeat_penalty == 1.0f) return;
  device const int* ids = (device const int*)(pen + g.y * a.pen_rs);
  const int id = ids[g.x];
  if (uint(id) >= a.vocab) return;  // padding, or past the vocabulary (the CPU reference refuses that): no logit
  for (uint j = 0; j < g.x; ++j)
    if (ids[j] == id) return;
  const float x = ((device const float*)(lg + g.y * a.lg_rs))[id];
  scratch[u64(g.y) * a.vocab + uint(id)] = x <= 0 ? x * p.repeat_penalty : x / p.repeat_penalty;
}

// (v, i) comes before (bv, bi) in (value desc, id asc) order; bi < 0 is "none yet".
static bool before(float v, int i, float bv, int bi) { return i >= 0 && (bi < 0 || v > bv || (v == bv && i < bi)); }

// The threadgroup's max or sum of v in every thread; pv holds one value per simdgroup.
static float tg_max(float v, threadgroup float* pv, uint sg, uint lane, uint nsg) {
  v = simd_max(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);  // earlier reads of pv are done
  if (lane == 0) pv[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = -INFINITY;
  for (uint i = 0; i < nsg; ++i) v = max(v, pv[i]);
  return v;
}
static float tg_sum(float v, threadgroup float* pv, uint sg, uint lane, uint nsg) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) pv[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v = 0;
  for (uint i = 0; i < nsg; ++i) v += pv[i];
  return v;
}

// The simdgroup's first (bv, bi) in (value desc, id asc) order, in every lane.
static void simd_best(thread float& bv, thread int& bi) {
  for (ushort o = 16; o > 0; o /= 2) {
    const float v = simd_shuffle_xor(bv, o);
    const int i = simd_shuffle_xor(bi, o);
    if (before(v, i, bv, bi)) {
      bv = v;
      bi = i;
    }
  }
}

// Sorted lists of k entries for sp_merge: parts first, first + 1, ... of a row's partials in device memory, or
// lists in threadgroup memory.
struct sp_parts {
  device const float* p;
  uint first, count;
  float val(uint l, uint h) const { return p[(first + l) * SP_PART + 2 + h]; }
  int id(uint l, uint h) const { return as_type<int>(p[(first + l) * SP_PART + 2 + 64 + h]); }
};
struct sp_lists {
  threadgroup const float* v;
  threadgroup const int* i;
  uint count, k;
  float val(uint l, uint h) const { return v[l * k + h]; }
  int id(uint l, uint h) const { return i[l * k + h]; }
};

// One simdgroup merges up to 32 lists of k entries, each in (value desc, id asc) order, into the first k of their
// union: lane l keeps the head of list l. Ids are unique, so the lane whose
// head won advances. No barrier: each simdgroup of a threadgroup can run its own.
template <class L, class V, class I>
void sp_merge(L lists, uint k, V ov, I oi, uint lane) {
  uint h = 0;
  for (uint t = 0; t < k; ++t) {
    float bv = -INFINITY;
    int bi = -1;
    if (lane < lists.count && h < k) {
      bv = lists.val(lane, h);
      bi = lists.id(lane, h);
    }
    const int mine = bi;
    simd_best(bv, bi);
    if (mine == bi && bi >= 0) ++h;
    if (lane == 0) {
      ov[t] = bv;
      oi[t] = bi;
    }
  }
}

// 3 of 3 (the only step without a penalty), 1 of 2: threadgroup (p, r) takes chunk p of row r of lg (the
// logits, or the penalised scratch), rows a.lg_rs apart. Its partial gets the chunk's max m, the sum of
// exp(x - m), and its first k entries in (value desc, id asc) order, from a bitonic sort of the chunk in
// threadgroup memory. The last chunk's padding is -inf with chunk ids past the real ones: it sorts after every real
// entry, k <= vocab real entries come first, and it leaves with id -1.
kernel void vv_sp_part(bytes_in lg [[buffer(0)]], device const sample_row* rows [[buffer(1)]],
                       device float* part [[buffer(2)]], constant sample_args& a [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[SP_CHUNK], pv[32];
  threadgroup ushort xi[SP_CHUNK];  // the chunk's ids
  const uint v0 = tg.x * SP_CHUNK, n = min(uint(SP_CHUNK), a.vocab - v0), parts = (a.vocab + SP_CHUNK - 1) / SP_CHUNK;
  device const float* l = (device const float*)(lg + u64(tg.y) * a.lg_rs) + v0;
  device float* out = part + (u64(tg.y) * parts + tg.x) * SP_PART;
  float mx = -INFINITY;
  for (uint i = tid; i < SP_CHUNK; i += SP_THREADS) {
    const float x = i < n ? l[i] : -INFINITY;
    xs[i] = x;
    xi[i] = ushort(i);
    mx = max(mx, x);
  }
  mx = tg_max(mx, pv, sg, lane, SP_THREADS / 32);  // its barriers also make xs and xi visible to every thread
  float sum = 0;
  if (mx != -INFINITY)  // a chunk of -inf only adds 0 (not NaN) to the row's sum
    for (uint i = tid; i < n; i += SP_THREADS) sum += exp(xs[i] - mx);
  sum = tg_sum(sum, pv, sg, lane, SP_THREADS / 32);
  if (tid == 0) {
    out[0] = mx;
    out[1] = sum;
  }
  for (uint size = 2; size <= SP_CHUNK; size *= 2)
    for (uint stride = size / 2; stride > 0; stride /= 2) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint t = tid; t < SP_CHUNK / 2; t += SP_THREADS) {
        const uint i = 2 * t - (t & (stride - 1)), j = i + stride;
        const float x = xs[i], y = xs[j];
        const ushort xid = xi[i], yid = xi[j];
        if ((y > x || (y == x && yid < xid)) == ((i & size) == 0)) {  // in (value desc, id asc) order per block
          xs[i] = y;
          xs[j] = x;
          xi[i] = yid;
          xi[j] = xid;
        }
      }
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const sample_row p = rows[tg.y];
  const uint k = min(p.top_k ? p.top_k : 64u, a.vocab);
  device int* oi = (device int*)(out + 2 + 64);
  for (uint t = tid; t < k; t += SP_THREADS) {
    out[2 + t] = xs[t];
    oi[t] = xi[t] < n ? int(v0 + xi[t]) : -1;  // -1: the merge takes no padding, whatever a NaN did to the order
  }
}

// 2 of 2: one threadgroup per row merges the row's partials. The log-sum-exp rescales each part's sum to the row's
// max; a row of -inf only stays NaN, as in the CPU reference. Simdgroup s merges parts 32 s to 32 s + 31, and
// simdgroup 0 merges those lists.
kernel void vv_sp_merge(device const float* part [[buffer(0)]], device const sample_row* rows [[buffer(1)]],
                        device uchar* vals [[buffer(2)]], device uchar* ids [[buffer(3)]], device uchar* lse [[buffer(4)]],
                        constant sample_args& a [[buffer(5)]], uint r [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float pv[32], tv[SP_MERGE_THREADS / 32 * 64];
  threadgroup int ti[SP_MERGE_THREADS / 32 * 64];
  const uint parts = (a.vocab + SP_CHUNK - 1) / SP_CHUNK, nsg = nt / 32;  // nt <= SP_MERGE_THREADS
  device const float* pr = part + u64(r) * parts * SP_PART;
  float mx = -INFINITY;
  for (uint i = tid; i < parts; i += nt) mx = max(mx, pr[i * SP_PART]);
  mx = tg_max(mx, pv, sg, lane, nsg);
  float sum = 0;
  for (uint i = tid; i < parts; i += nt) sum += pr[i * SP_PART + 1] * exp(pr[i * SP_PART] - mx);
  sum = tg_sum(sum, pv, sg, lane, nsg);
  if (tid == 0) *(device float*)(lse + r * a.lse_rs) = mx + precise::log(sum);

  const sample_row p = rows[r];
  const uint k = min(p.top_k ? p.top_k : 64u, a.vocab), first = min(32 * sg, parts);
  sp_merge(sp_parts{pr, first, min(32u, parts - first)}, k, tv + sg * k, ti + sg * k, lane);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  device float* ov = (device float*)(vals + r * a.vals_rs);
  device int* oi = (device int*)(ids + r * a.ids_rs);
  sp_merge(sp_lists{tv, ti, nsg, k}, k, ov, oi, lane);
  for (uint t = k + lane; t < 64; t += 32) {
    ov[t] = -INFINITY;
    oi[t] = -1;
  }
}
