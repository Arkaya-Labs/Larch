// GGML block layouts, decoded exactly as ggml-quants.c's dequantize_row_* (same f32 expression
// order), so the reference sees the weights llama.cpp sees.
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "cpu/cpu.h"

namespace cpu {

type_info info(larch_dtype t) {
  switch (t) {
    case LARCH_F32: return {1, 4};
    case LARCH_F16: return {1, 2};
    case LARCH_BF16: return {1, 2};
    case LARCH_I32: return {1, 4};
    case LARCH_Q4_K: return {256, 144};
    case LARCH_Q5_K: return {256, 176};
    case LARCH_Q6_K: return {256, 210};
    case LARCH_Q8_0: return {32, 34};
    case LARCH_Q4_0: return {32, 18};
    case LARCH_Q5_0: return {32, 22};
    case LARCH_FP8_E4M3: return {1, 1};
    case LARCH_Q3_K: return {256, 110};
    case LARCH_Q2_K: return {256, 84};
    case LARCH_NVFP4: return {64, 36};
  }
  throw bad_arg("unknown dtype " + std::to_string(int(t)));
}

// The 6-bit scale and min of sub-block j in a Q4_K/Q5_K super-block (ggml's get_scale_min_k4).
static void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = uint8_t((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

// The 16 signed 6-bit scales of a Q3_K block (hmask 32, qs 64, then these 12 bytes): low nibbles from bytes
// i % 8 (high nibble for i >= 8), the top 2 bits from byte 8 + i % 4 at bit 2 (i / 4); minus 32.
static void scales_q3_k(const uint8_t* b, int* sc) {
  const uint8_t* s = b + 96;
  for (int i = 0; i < 16; ++i) sc[i] = ((i < 8 ? s[i] & 0xF : s[i - 8] >> 4) | (s[8 + i % 4] >> (2 * (i / 4)) & 3) << 4) - 32;
}

void i8_group(larch_dtype t, const uint8_t* row, int64_t g, int8_t* q, float* s, float* m) {
  *m = 0;
  switch (t) {
    case LARCH_Q4_K: {  // sub-block g % 8 of block g / 8: d sc, dmin m and nibbles of qs[32 (g % 8 / 2)..]
      const uint8_t* b = row + g / 8 * 144;
      uint8_t sc, mn;
      scale_min_k4(int(g % 8), b + 4, &sc, &mn);
      s[0] = s[1] = half(b) * float(sc);
      *m = half(b + 2) * float(mn);
      for (int j = 0; j < 32; ++j) q[j] = int8_t((b[16 + g % 8 / 2 * 32 + j] >> (g % 2 * 4)) & 15);
      return;
    }
    case LARCH_Q5_K: {  // Q4_K's layout after 32 bytes of fifth bits: bit g % 8 of qh[j]
      const uint8_t* b = row + g / 8 * 176;
      uint8_t sc, mn;
      scale_min_k4(int(g % 8), b + 4, &sc, &mn);
      s[0] = s[1] = half(b) * float(sc);
      *m = half(b + 2) * float(mn);
      for (int j = 0; j < 32; ++j)
        q[j] = int8_t(((b[48 + g % 8 / 2 * 32 + j] >> (g % 2 * 4)) & 15) | ((b[16 + j] >> (g % 8)) & 1) << 4);
      return;
    }
    case LARCH_Q6_K: {  // elements 32 (g % 8).. of block g / 8: quarter u of half n, scales 2 (g % 8) and + 1
      const uint8_t* b = row + g / 8 * 210;
      const int n = int(g % 8) / 4, u = int(g % 4);
      const uint8_t* ql = b + 64 * n + 32 * (u & 1);
      for (int l = 0; l < 32; ++l)
        q[l] = int8_t(((u < 2 ? ql[l] & 0xF : ql[l] >> 4) | ((b[128 + 32 * n + l] >> (2 * u)) & 3) << 4) - 32);
      s[0] = half(b + 208) * float(int8_t(b[192 + 2 * (g % 8)]));
      s[1] = half(b + 208) * float(int8_t(b[193 + 2 * (g % 8)]));
      return;
    }
    case LARCH_Q8_0: {  // block g: d, then 32 int8 codes
      const uint8_t* b = row + g * 34;
      s[0] = s[1] = half(b);
      for (int j = 0; j < 32; ++j) q[j] = int8_t(b[2 + j]);
      return;
    }
    case LARCH_Q4_0:  // block g: d, then low nibbles for elements 0-15 and high ones for 16-31, code q - 8
    case LARCH_Q5_0: {  // Q5_0: d, 32 fifth bits (bit j for element j), the same nibbles, code q - 16
      const bool q5 = t == LARCH_Q5_0;
      const uint8_t* b = row + g * (q5 ? 22 : 18);
      const uint32_t qh = q5 ? uint32_t(b[2]) | uint32_t(b[3]) << 8 | uint32_t(b[4]) << 16 | uint32_t(b[5]) << 24 : 0;
      s[0] = s[1] = half(b);
      for (int j = 0; j < 32; ++j) {
        const uint8_t byte = b[(q5 ? 6 : 2) + j % 16];
        q[j] = int8_t(((j < 16 ? byte & 15 : byte >> 4) | (qh >> j & 1) << 4) - (q5 ? 16 : 8));
      }
      return;
    }
    default:
      throw bad_arg("no int8 policy for this dtype");
  }
}

static void q4_0(const uint8_t* b, float* y) {
  const float d = half(b);
  const uint8_t* qs = b + 2;
  for (int j = 0; j < 16; ++j) {
    y[j] = float((qs[j] & 0x0F) - 8) * d;
    y[j + 16] = float((qs[j] >> 4) - 8) * d;
  }
}

static void q5_0(const uint8_t* b, float* y) {
  const float d = half(b);
  const uint32_t qh = uint32_t(b[2]) | uint32_t(b[3]) << 8 | uint32_t(b[4]) << 16 | uint32_t(b[5]) << 24;
  const uint8_t* qs = b + 6;
  for (int j = 0; j < 16; ++j) {
    const int xh0 = int(((qh >> j) << 4) & 0x10), xh1 = int((qh >> (j + 12)) & 0x10);
    y[j] = float(((qs[j] & 0x0F) | xh0) - 16) * d;
    y[j + 16] = float(((qs[j] >> 4) | xh1) - 16) * d;
  }
}

static void q8_0(const uint8_t* b, float* y) {
  const float d = half(b);
  for (int j = 0; j < 32; ++j) y[j] = float(int8_t(b[2 + j])) * d;
}

static void q4_k(const uint8_t* b, float* y) {
  const float d = half(b), min = half(b + 2);
  const uint8_t* scales = b + 4;
  const uint8_t* q = b + 16;
  uint8_t sc, m;
  for (int j = 0, is = 0; j < 256; j += 64, is += 2, q += 32) {
    scale_min_k4(is, scales, &sc, &m);
    const float d1 = d * sc, m1 = min * m;
    scale_min_k4(is + 1, scales, &sc, &m);
    const float d2 = d * sc, m2 = min * m;
    for (int l = 0; l < 32; ++l) *y++ = d1 * float(q[l] & 0xF) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * float(q[l] >> 4) - m2;
  }
}

static void q5_k(const uint8_t* b, float* y) {
  const float d = half(b), min = half(b + 2);
  const uint8_t* scales = b + 4;
  const uint8_t* qh = b + 16;
  const uint8_t* ql = b + 48;
  uint8_t sc, m, u1 = 1, u2 = 2;
  for (int j = 0, is = 0; j < 256; j += 64, is += 2, ql += 32, u1 = uint8_t(u1 << 2), u2 = uint8_t(u2 << 2)) {
    scale_min_k4(is, scales, &sc, &m);
    const float d1 = d * sc, m1 = min * m;
    scale_min_k4(is + 1, scales, &sc, &m);
    const float d2 = d * sc, m2 = min * m;
    for (int l = 0; l < 32; ++l) *y++ = d1 * float((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * float((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
  }
}

// 16 scale bytes (low nibble the sub-block's scale, high its min), 64 qs bytes, d, dmin. Two halves of 128: in
// half n, bit pair u of qs[32 n + l] is element 32 u + l, of sub-block 8 n + 2 u + l / 16.
static void q2_k(const uint8_t* b, float* y) {
  const float d = half(b + 80), min = half(b + 82);
  const uint8_t* q = b + 16;
  for (int n = 0, is = 0; n < 2; ++n, q += 32)
    for (int u = 0; u < 4; ++u)
      for (int half16 = 0; half16 < 2; ++half16) {
        const uint8_t sc = b[is++];
        const float dl = d * float(sc & 0xF), ml = min * float(sc >> 4);
        for (int l = 16 * half16; l < 16 * half16 + 16; ++l) *y++ = dl * float((q[l] >> (2 * u)) & 3) - ml;
      }
}

static void q3_k(const uint8_t* b, float* y) {
  const float d = half(b + 108);
  const uint8_t* hm = b;
  const uint8_t* q = b + 32;
  int sc[16];
  scales_q3_k(b, sc);
  for (int n = 0, is = 0; n < 256; n += 128, q += 32)
    for (int u = 0; u < 4; ++u) {
      const int m = 1 << (n / 32 + u);
      for (int half16 = 0; half16 < 2; ++half16) {
        const float dl = d * float(sc[is++]);
        for (int l = 16 * half16; l < 16 * half16 + 16; ++l)
          *y++ = dl * float(int((q[l] >> (2 * u)) & 3) - (hm[l] & m ? 0 : 4));
      }
    }
}

static void q6_k(const uint8_t* b, float* y) {
  const uint8_t* ql = b;
  const uint8_t* qh = b + 128;
  const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
  const float d = half(b + 208);
  for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      y[l] = d * float(sc[is]) * float(q1);
      y[l + 32] = d * float(sc[is + 2]) * float(q2);
      y[l + 64] = d * float(sc[is + 4]) * float(q3);
      y[l + 96] = d * float(sc[is + 6]) * float(q4);
    }
  }
}

// ggml_ue4m3_to_fp32: an unsigned E4M3 scale at half its value (kvalues_mxfp4 hold twice the E2M1 values).
static float ue4m3(uint8_t x) {
  if (x == 0 || x == 0x7F) return 0.0f;
  const int e = (x >> 3) & 0xF, m = x & 7;
  const float raw = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.0f + float(m) / 8.0f, e - 7);
  return raw * 0.5f;
}

static void nvfp4(const uint8_t* b, float* y) {
  static constexpr int8_t values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  for (int s = 0; s < 4; ++s) {
    const float d = ue4m3(b[s]);
    const uint8_t* q = b + 4 + 8 * s;
    for (int j = 0; j < 8; ++j) {
      y[16 * s + j] = float(values[q[j] & 0xF]) * d;
      y[16 * s + j + 8] = float(values[q[j] >> 4]) * d;
    }
  }
}

void to_f32(larch_dtype t, const uint8_t* src, int64_t n, float* dst) {
  void (*block)(const uint8_t*, float*) = nullptr;
  switch (t) {
    case LARCH_F32: std::memcpy(dst, src, size_t(n) * 4); return;
    case LARCH_F16:
      for (int64_t i = 0; i < n; ++i) dst[i] = half(src + 2 * i);
      return;
    case LARCH_BF16:
      for (int64_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(u16(src + 2 * i));
      return;
    case LARCH_I32: throw bad_arg("i32 is not a float type");
    case LARCH_FP8_E4M3:
      for (int64_t i = 0; i < n; ++i) dst[i] = fp8_to_f32(src[i]);
      return;
    case LARCH_Q4_0: block = q4_0; break;
    case LARCH_Q5_0: block = q5_0; break;
    case LARCH_Q8_0: block = q8_0; break;
    case LARCH_Q4_K: block = q4_k; break;
    case LARCH_Q5_K: block = q5_k; break;
    case LARCH_Q6_K: block = q6_k; break;
    case LARCH_Q3_K: block = q3_k; break;
    case LARCH_Q2_K: block = q2_k; break;
    case LARCH_NVFP4: block = nvfp4; break;
  }
  const type_info ti = info(t);
  for (int64_t i = 0; i < n / ti.block; ++i) block(src + i * ti.bytes, dst + i * ti.block);
}

// The 127 non-negative finite fp8 e4m3 values, by code: e = code >> 3, m = code & 7, m 2^-9 for e = 0
// (subnormal), else (8 + m) 2^(e - 10).
static const std::array<float, 127>& fp8_values() {
  static const std::array<float, 127> t = [] {
    std::array<float, 127> v{};
    for (int c = 0; c < 127; ++c)
      v[size_t(c)] = std::ldexp(float((c & 7) + (c >> 3 ? 8 : 0)), std::max(c >> 3, 1) - 10);
    return v;
  }();
  return t;
}

float fp8_to_f32(uint8_t b) {
  if ((b & 0x7F) == 0x7F) return std::numeric_limits<float>::quiet_NaN();
  const float v = fp8_values()[b & 0x7F];
  return b & 0x80 ? -v : v;
}

uint8_t f32_to_fp8(float f) {
  const uint8_t sign = std::signbit(f) ? 0x80 : 0;
  if (std::isnan(f)) return sign | 0x7F;
  const std::array<float, 127>& t = fp8_values();
  const float a = std::fabs(f);
  if (a >= t[126]) return sign | 126;  // saturate at 448, infinities included
  // t[i] <= a < t[i + 1]; both differences are exact in double, so a tie is a real one.
  const size_t i = size_t(std::upper_bound(t.begin(), t.end(), a) - t.begin()) - 1;
  const double lo = double(a) - t[i], hi = double(t[i + 1]) - a;
  return uint8_t(sign | (lo < hi || (lo == hi && i % 2 == 0) ? i : i + 1));
}

void kv_from_f32(larch_dtype t, const float* src, int64_t n, uint8_t* dst) {
  switch (t) {
    case LARCH_F16:
      for (int64_t i = 0; i < n; ++i) {
        const uint16_t h = f32_to_f16(src[i]);
        std::memcpy(dst + 2 * i, &h, 2);
      }
      return;
    case LARCH_FP8_E4M3:
      for (int64_t i = 0; i < n; ++i) dst[i] = f32_to_fp8(src[i]);
      return;
    case LARCH_Q8_0:  // ggml's quantize_row_q8_0_ref, its MAX macro's NaN behaviour included
      for (int64_t b = 0; b < n / 32; ++b, src += 32, dst += 34) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = amax > std::fabs(src[j]) ? amax : std::fabs(src[j]);
        const float d = amax / ((1 << 7) - 1), id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t h = f32_to_f16(d);
        std::memcpy(dst, &h, 2);
        for (int j = 0; j < 32; ++j) {
          const float q = roundf(src[j] * id);
          dst[2 + j] = uint8_t(int8_t(std::isnan(q) ? 0.0f : q));  // NaN: undefined in ggml, 0 here
        }
      }
      return;
    default:
      throw bad_arg("not a KV page dtype");
  }
}

}  // namespace cpu
