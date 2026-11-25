// The fast CPU backend's mmvq_id on ARM64, with NEON (its baseline).
#include <arm_neon.h>

#include "larch.h"
#include "cpu/fast_mmvq_id.inc"

namespace {

// 16 signed bytes q (as f32) times x[0, 16), into p.
float32x4_t dot16(int8x16_t q, const float* x, float32x4_t p) {
  const int16x8_t l = vmovl_s8(vget_low_s8(q)), h = vmovl_high_s8(q);
  p = vfmaq_f32(p, vcvtq_f32_s32(vmovl_s16(vget_low_s16(l))), vld1q_f32(x));
  p = vfmaq_f32(p, vcvtq_f32_s32(vmovl_high_s16(l)), vld1q_f32(x + 4));
  p = vfmaq_f32(p, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h))), vld1q_f32(x + 8));
  return vfmaq_f32(p, vcvtq_f32_s32(vmovl_high_s16(h)), vld1q_f32(x + 12));
}

// acc + 16 bytes q decoded as d q - m (the reference's weights, rounded as it rounds them) times x[0, 16).
float32x4_t dot16_dm(uint8x16_t q, const float* x, float d, float m, float32x4_t acc) {
  const uint16x8_t l = vmovl_u8(vget_low_u8(q)), h = vmovl_high_u8(q);
  const float32x4_t mv = vdupq_n_f32(m);
  const auto w = [&](uint32x4_t u) { return vsubq_f32(vmulq_n_f32(vcvtq_f32_u32(u), d), mv); };
  acc = vfmaq_f32(acc, w(vmovl_u16(vget_low_u16(l))), vld1q_f32(x));
  acc = vfmaq_f32(acc, w(vmovl_high_u16(l)), vld1q_f32(x + 4));
  acc = vfmaq_f32(acc, w(vmovl_u16(vget_low_u16(h))), vld1q_f32(x + 8));
  return vfmaq_f32(acc, w(vmovl_high_u16(h)), vld1q_f32(x + 12));
}

template <bool Q5>
float dot_k4(const uint8_t* w, const float* x, int64_t k) {
  constexpr int BYTES = Q5 ? 176 : 144;
  const uint8x16_t m4 = vdupq_n_u8(0x0F), one = vdupq_n_u8(1);
  // One accumulator per 16-weight quarter: a single chain serializes on FMA latency (2x slower on the M5).
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
  for (int64_t b = 0; b < k / 256; ++b, w += BYTES, x += 256) {
    const float d = h2f(w), min = h2f(w + 2);
    const uint8x16_t qh0 = vld1q_u8(w + 16), qh1 = vld1q_u8(w + 32);
    const uint8_t* q = w + (Q5 ? 48 : 16);
    for (int is = 0; is < 8; is += 2, q += 32) {
      uint8_t sc, m;
      scale_min_k4(is, w + 4, &sc, &m);
      const float d1 = d * sc, m1 = min * m;
      scale_min_k4(is + 1, w + 4, &sc, &m);
      const float d2 = d * sc, m2 = min * m;
      const uint8x16_t v0 = vld1q_u8(q), v1 = vld1q_u8(q + 16);
      uint8x16_t lo0 = vandq_u8(v0, m4), lo1 = vandq_u8(v1, m4), hi0 = vshrq_n_u8(v0, 4), hi1 = vshrq_n_u8(v1, 4);
      if (Q5) {
        const int8x16_t s1 = vdupq_n_s8(int8_t(-is)), s2 = vdupq_n_s8(int8_t(-is - 1));
        const auto bit = [&](uint8x16_t h, int8x16_t s) { return vshlq_n_u8(vandq_u8(vshlq_u8(h, s), one), 4); };
        lo0 = vorrq_u8(lo0, bit(qh0, s1)), lo1 = vorrq_u8(lo1, bit(qh1, s1));
        hi0 = vorrq_u8(hi0, bit(qh0, s2)), hi1 = vorrq_u8(hi1, bit(qh1, s2));
      }
      const float* xa = x + 32 * is;
      a0 = dot16_dm(lo0, xa, d1, m1, a0);
      a1 = dot16_dm(lo1, xa + 16, d1, m1, a1);
      a2 = dot16_dm(hi0, xa + 32, d2, m2, a2);
      a3 = dot16_dm(hi1, xa + 48, d2, m2, a3);
    }
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
}

float dot_q4_k(const uint8_t* w, const float* x, int64_t k) { return dot_k4<false>(w, x, k); }
float dot_q5_k(const uint8_t* w, const float* x, int64_t k) { return dot_k4<true>(w, x, k); }

// acc[0, 4) + 16 signed bytes q decoded as s q (the reference's weights, rounded as it rounds them) times
// x[0, 16), one chain per four lanes.
void dot16_s(int8x16_t q, const float* x, float s, float32x4_t* acc) {
  const int16x8_t l = vmovl_s8(vget_low_s8(q)), h = vmovl_high_s8(q);
  const auto w = [&](int16x4_t v) { return vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(v)), s); };
  acc[0] = vfmaq_f32(acc[0], w(vget_low_s16(l)), vld1q_f32(x));
  acc[1] = vfmaq_f32(acc[1], w(vget_high_s16(l)), vld1q_f32(x + 4));
  acc[2] = vfmaq_f32(acc[2], w(vget_low_s16(h)), vld1q_f32(x + 8));
  acc[3] = vfmaq_f32(acc[3], w(vget_high_s16(h)), vld1q_f32(x + 12));
}

// 16 Q3_K weights of quarter u of half n, as signed bytes: bits 2u of qs, minus 4 unless bit 4n + u of hm is set.
uint8x16_t q3(uint8x16_t qs, uint8x16_t hm, int n, int u) {
  const uint8x16_t lo = vandq_u8(vshlq_u8(qs, vdupq_n_s8(int8_t(-2 * u))), vdupq_n_u8(3));
  return vsubq_u8(lo, vbicq_u8(vdupq_n_u8(4), vtstq_u8(hm, vdupq_n_u8(uint8_t(1 << (4 * n + u))))));
}

// Each weight decoded before its product, in dot_k4's shape: scaling each sub-block's sum of q x once, as
// dot_q6_k does, ran Q3_K 14% slower than Q4_K on the M5 though it reads 22% fewer bytes.
float dot_q3_k(const uint8_t* w, const float* x, int64_t k) {
  float32x4_t acc[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
  for (int64_t b = 0; b < k / 256; ++b, w += 110, x += 256) {
    const float d = h2f(w + 108);
    int sc[16];
    scales_q3_k(w + 96, sc);
    float dl[16];
    for (int i = 0; i < 16; ++i) dl[i] = d * float(sc[i]);
    const uint8x16_t h0 = vld1q_u8(w), h1 = vld1q_u8(w + 16);
    for (int n = 0; n < 2; ++n) {
      const uint8x16_t q0 = vld1q_u8(w + 32 + 32 * n), q1 = vld1q_u8(w + 48 + 32 * n);
      for (int u = 0; u < 4; ++u) {  // weights [128 n + 32 u, + 32): bits 2u of qs, bit 4n + u of hmask
        const float* xa = x + 128 * n + 32 * u;
        const int is = 8 * n + 2 * u;
        dot16_s(vreinterpretq_s8_u8(q3(q0, h0, n, u)), xa, dl[is], acc);
        dot16_s(vreinterpretq_s8_u8(q3(q1, h1, n, u)), xa + 16, dl[is + 1], acc);
      }
    }
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])));
}

float dot_q6_k(const uint8_t* w, const float* x, int64_t k) {
  const uint8x16_t m4 = vdupq_n_u8(0x0F), m2 = vdupq_n_u8(3);
  const int8x16_t off = vdupq_n_s8(32);
  float32x4_t acc = vdupq_n_f32(0);
  for (int64_t b = 0; b < k / 256; ++b, w += 210, x += 256) {
    const uint8_t* ql = w;
    const uint8_t* qh = w + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(w + 192);
    const float d = h2f(w + 208);
    for (int n = 0; n < 256; n += 128, ql += 64, qh += 32, sc += 8) {
      const float* xa = x + n;
      for (int half = 0; half < 2; ++half) {  // l in [16 half, 16 half + 16): scales sc[half], sc[2 + half], ...
        const int l = 16 * half;
        const uint8x16_t a = vld1q_u8(ql + l), c = vld1q_u8(ql + l + 32), h = vld1q_u8(qh + l);
        const auto q = [&](uint8x16_t nib, int shift) {
          const uint8x16_t hb = vandq_u8(vshlq_u8(h, vdupq_n_s8(int8_t(-shift))), m2);
          return vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(nib, vshlq_n_u8(hb, 4))), off);
        };
        const float32x4_t z = vdupq_n_f32(0);
        acc = vfmaq_n_f32(acc, dot16(q(vandq_u8(a, m4), 0), xa + l, z), d * float(sc[half]));
        acc = vfmaq_n_f32(acc, dot16(q(vandq_u8(c, m4), 2), xa + l + 32, z), d * float(sc[2 + half]));
        acc = vfmaq_n_f32(acc, dot16(q(vshrq_n_u8(a, 4), 4), xa + l + 64, z), d * float(sc[4 + half]));
        acc = vfmaq_n_f32(acc, dot16(q(vshrq_n_u8(c, 4), 6), xa + l + 96, z), d * float(sc[6 + half]));
      }
    }
  }
  return vaddvq_f32(acc);
}

float dot_q8_0(const uint8_t* w, const float* x, int64_t k) {
  float32x4_t acc = vdupq_n_f32(0);
  for (int64_t b = 0; b < k / 32; ++b, w += 34, x += 32) {
    const int8_t* q = reinterpret_cast<const int8_t*>(w + 2);
    const float32x4_t p = dot16(vld1q_s8(q + 16), x + 16, dot16(vld1q_s8(q), x, vdupq_n_f32(0)));
    acc = vfmaq_n_f32(acc, p, h2f(w));
  }
  return vaddvq_f32(acc);
}

// 16 bytes (U unsigned) as f32 times s, minus m, into y[0, 16): a product then a difference, as the reference
// rounds them.
template <bool U>
void put16(float* y, uint8x16_t b, float s, float m) {
  const float32x4_t mv = vdupq_n_f32(m);
  int32x4_t v[4];
  if (U) {
    const uint16x8_t l = vmovl_u8(vget_low_u8(b)), h = vmovl_high_u8(b);
    v[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(l))), v[1] = vreinterpretq_s32_u32(vmovl_high_u16(l));
    v[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(h))), v[3] = vreinterpretq_s32_u32(vmovl_high_u16(h));
  } else {
    const int8x16_t q = vreinterpretq_s8_u8(b);
    const int16x8_t l = vmovl_s8(vget_low_s8(q)), h = vmovl_high_s8(q);
    v[0] = vmovl_s16(vget_low_s16(l)), v[1] = vmovl_high_s16(l), v[2] = vmovl_s16(vget_low_s16(h)), v[3] = vmovl_high_s16(h);
  }
  for (int i = 0; i < 4; ++i) vst1q_f32(y + 4 * i, vsubq_f32(vmulq_n_f32(vcvtq_f32_s32(v[i]), s), mv));
}

template <bool Q5>
void decode_k4(const uint8_t* w, int64_t k, float* y) {
  constexpr int BYTES = Q5 ? 176 : 144;
  const uint8x16_t m4 = vdupq_n_u8(0x0F), one = vdupq_n_u8(1);
  for (int64_t b = 0; b < k / 256; ++b, w += BYTES) {
    const float d = h2f(w), min = h2f(w + 2);
    const uint8x16_t qh0 = vld1q_u8(w + 16), qh1 = vld1q_u8(w + 32);
    const uint8_t* q = w + (Q5 ? 48 : 16);
    for (int is = 0; is < 8; is += 2, q += 32, y += 64) {
      uint8_t sc, m;
      scale_min_k4(is, w + 4, &sc, &m);
      const float d1 = d * sc, m1 = min * m;
      scale_min_k4(is + 1, w + 4, &sc, &m);
      const float d2 = d * sc, m2 = min * m;
      const uint8x16_t v0 = vld1q_u8(q), v1 = vld1q_u8(q + 16);
      uint8x16_t lo0 = vandq_u8(v0, m4), lo1 = vandq_u8(v1, m4), hi0 = vshrq_n_u8(v0, 4), hi1 = vshrq_n_u8(v1, 4);
      if (Q5) {
        const int8x16_t s1 = vdupq_n_s8(int8_t(-is)), s2 = vdupq_n_s8(int8_t(-is - 1));
        const auto bit = [&](uint8x16_t h, int8x16_t s) { return vshlq_n_u8(vandq_u8(vshlq_u8(h, s), one), 4); };
        lo0 = vorrq_u8(lo0, bit(qh0, s1)), lo1 = vorrq_u8(lo1, bit(qh1, s1));
        hi0 = vorrq_u8(hi0, bit(qh0, s2)), hi1 = vorrq_u8(hi1, bit(qh1, s2));
      }
      put16<true>(y, lo0, d1, m1);
      put16<true>(y + 16, lo1, d1, m1);
      put16<true>(y + 32, hi0, d2, m2);
      put16<true>(y + 48, hi1, d2, m2);
    }
  }
}

void decode_q4_k(const uint8_t* w, int64_t k, float* y) { decode_k4<false>(w, k, y); }
void decode_q5_k(const uint8_t* w, int64_t k, float* y) { decode_k4<true>(w, k, y); }

void decode_q3_k(const uint8_t* w, int64_t k, float* y) {
  for (int64_t b = 0; b < k / 256; ++b, w += 110) {
    const float d = h2f(w + 108);
    int sc[16];
    scales_q3_k(w + 96, sc);
    const uint8x16_t h0 = vld1q_u8(w), h1 = vld1q_u8(w + 16);
    for (int n = 0; n < 2; ++n) {
      const uint8x16_t q0 = vld1q_u8(w + 32 + 32 * n), q1 = vld1q_u8(w + 48 + 32 * n);
      for (int u = 0; u < 4; ++u, y += 32) {  // y[128 n + 32 u + l] = (d sc[8 n + 2 u + l / 16]) q, as the reference
        const int is = 8 * n + 2 * u;
        put16<false>(y, q3(q0, h0, n, u), d * float(sc[is]), 0);
        put16<false>(y + 16, q3(q1, h1, n, u), d * float(sc[is + 1]), 0);
      }
    }
  }
}

void decode_q6_k(const uint8_t* w, int64_t k, float* y) {
  const uint8x16_t m4 = vdupq_n_u8(0x0F), m2 = vdupq_n_u8(3);
  const int8x16_t off = vdupq_n_s8(32);
  for (int64_t b = 0; b < k / 256; ++b, w += 210) {
    const uint8_t* ql = w;
    const uint8_t* qh = w + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(w + 192);
    const float d = h2f(w + 208);
    for (int n = 0; n < 256; n += 128, ql += 64, qh += 32, sc += 8, y += 128) {
      for (int half = 0; half < 2; ++half) {  // l in [16 half, 16 half + 16): scales sc[half], sc[2 + half], ...
        const int l = 16 * half;
        const uint8x16_t a = vld1q_u8(ql + l), c = vld1q_u8(ql + l + 32), h = vld1q_u8(qh + l);
        const auto q = [&](uint8x16_t nib, int shift) {
          const uint8x16_t hb = vandq_u8(vshlq_u8(h, vdupq_n_s8(int8_t(-shift))), m2);
          return vreinterpretq_u8_s8(vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(nib, vshlq_n_u8(hb, 4))), off));
        };
        put16<false>(y + l, q(vandq_u8(a, m4), 0), d * float(sc[half]), 0);
        put16<false>(y + l + 32, q(vandq_u8(c, m4), 2), d * float(sc[2 + half]), 0);
        put16<false>(y + l + 64, q(vshrq_n_u8(a, 4), 4), d * float(sc[4 + half]), 0);
        put16<false>(y + l + 96, q(vshrq_n_u8(c, 4), 6), d * float(sc[6 + half]), 0);
      }
    }
  }
}

void decode_q8_0(const uint8_t* w, int64_t k, float* y) {
  for (int64_t b = 0; b < k / 32; ++b, w += 34, y += 32) {
    const float d = h2f(w);
    put16<false>(y, vld1q_u8(w + 2), d, 0);
    put16<false>(y + 16, vld1q_u8(w + 18), d, 0);
  }
}

}  // namespace

namespace cpu {

void fast_dot_tile(const float* const* w, const float* const* x, int64_t k, float* out) {
  // 4 w by 4 x: 16 accumulators, the 4 x of a step held, one w at a time (21 of 32 registers).
  float32x4_t a[16];
  for (float32x4_t& v : a) v = vdupq_n_f32(0);
  for (int64_t i = 0; i < k; i += 4) {
    float32x4_t xs[4];
    for (int p = 0; p < 4; ++p) xs[p] = vld1q_f32(x[p] + i);
    for (int c = 0; c < 4; ++c) {
      const float32x4_t u = vld1q_f32(w[c] + i);
      for (int p = 0; p < 4; ++p) a[4 * c + p] = vfmaq_f32(a[4 * c + p], u, xs[p]);
    }
  }
  for (int t = 0; t < 16; ++t) out[t] = vaddvq_f32(a[t]);
}

}  // namespace cpu
