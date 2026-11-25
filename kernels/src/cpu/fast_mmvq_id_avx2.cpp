// The fast CPU backend's mmvq_id on x86-64 with AVX2 and FMA (kernels/CMakeLists.txt gives this file its
// flags; fast_cpu.cpp checks the CPU before it calls in).
#include <immintrin.h>

#include "larch.h"
#include "cpu/fast_mmvq_id.inc"

namespace {

float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}

// Bytes [8 i, 8 i + 8) of b as f32, unsigned (U) or signed.
template <bool U>
__m256 f8(__m128i b) {
  return _mm256_cvtepi32_ps(U ? _mm256_cvtepu8_epi32(b) : _mm256_cvtepi8_epi32(b));
}

// sum over 32 bytes q (values as f32, U unsigned) times x[0, 32), into 8 lanes.
template <bool U>
__m256 dot32(__m256i q, const float* x) {
  const __m128i lo = _mm256_castsi256_si128(q), hi = _mm256_extracti128_si256(q, 1);
  __m256 p = _mm256_mul_ps(f8<U>(lo), _mm256_loadu_ps(x));
  p = _mm256_fmadd_ps(f8<U>(_mm_srli_si128(lo, 8)), _mm256_loadu_ps(x + 8), p);
  p = _mm256_fmadd_ps(f8<U>(hi), _mm256_loadu_ps(x + 16), p);
  return _mm256_fmadd_ps(f8<U>(_mm_srli_si128(hi, 8)), _mm256_loadu_ps(x + 24), p);
}

// acc + sum over 32 unsigned bytes q decoded as d q - m (the reference's weights, rounded as it rounds them)
// times x[0, 32).
__m256 dot32_dm(__m256i q, const float* x, float d, float m, __m256 acc) {
  const __m128i lo = _mm256_castsi256_si128(q), hi = _mm256_extracti128_si256(q, 1);
  const __m256 dv = _mm256_set1_ps(d), mv = _mm256_set1_ps(m);
  const auto w = [&](__m128i b) { return _mm256_sub_ps(_mm256_mul_ps(dv, f8<true>(b)), mv); };
  acc = _mm256_fmadd_ps(w(lo), _mm256_loadu_ps(x), acc);
  acc = _mm256_fmadd_ps(w(_mm_srli_si128(lo, 8)), _mm256_loadu_ps(x + 8), acc);
  acc = _mm256_fmadd_ps(w(hi), _mm256_loadu_ps(x + 16), acc);
  return _mm256_fmadd_ps(w(_mm_srli_si128(hi, 8)), _mm256_loadu_ps(x + 24), acc);
}

// The same with two scales: bytes [0, 16) by s0 and [16, 32) by s1.
__m256 dot32x2(__m256i q, const float* x, float s0, float s1, __m256 acc) {
  const __m128i lo = _mm256_castsi256_si128(q), hi = _mm256_extracti128_si256(q, 1);
  __m256 p = _mm256_mul_ps(f8<false>(lo), _mm256_loadu_ps(x));
  p = _mm256_fmadd_ps(f8<false>(_mm_srli_si128(lo, 8)), _mm256_loadu_ps(x + 8), p);
  acc = _mm256_fmadd_ps(_mm256_set1_ps(s0), p, acc);
  p = _mm256_mul_ps(f8<false>(hi), _mm256_loadu_ps(x + 16));
  p = _mm256_fmadd_ps(f8<false>(_mm_srli_si128(hi, 8)), _mm256_loadu_ps(x + 24), p);
  return _mm256_fmadd_ps(_mm256_set1_ps(s1), p, acc);
}

template <bool Q5>
float dot_k4(const uint8_t* w, const float* x, int64_t k) {
  constexpr int BYTES = Q5 ? 176 : 144;
  const __m256i m4 = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
  // Two accumulators: one chain serializes on FMA latency (the NEON twin ran 2x slower that way).
  __m256 a0 = _mm256_setzero_ps(), a1 = a0;
  for (int64_t b = 0; b < k / 256; ++b, w += BYTES, x += 256) {
    const float d = h2f(w), min = h2f(w + 2);
    const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16));
    const uint8_t* q = w + (Q5 ? 48 : 16);
    for (int is = 0; is < 8; is += 2, q += 32) {
      uint8_t sc, m;
      scale_min_k4(is, w + 4, &sc, &m);
      const float d1 = d * sc, m1 = min * m;
      scale_min_k4(is + 1, w + 4, &sc, &m);
      const float d2 = d * sc, m2 = min * m;
      const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
      __m256i lo = _mm256_and_si256(v, m4), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
      if (Q5) {
        const __m128i s1 = _mm_cvtsi32_si128(is), s2 = _mm_cvtsi32_si128(is + 1);
        lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, s1), one), 4));
        hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, s2), one), 4));
      }
      a0 = dot32_dm(lo, x + 32 * is, d1, m1, a0);
      a1 = dot32_dm(hi, x + 32 * is + 32, d2, m2, a1);
    }
  }
  return hsum(_mm256_add_ps(a0, a1));
}

float dot_q4_k(const uint8_t* w, const float* x, int64_t k) { return dot_k4<false>(w, x, k); }
float dot_q5_k(const uint8_t* w, const float* x, int64_t k) { return dot_k4<true>(w, x, k); }

// 32 Q3_K weights of quarter u of half n, as signed bytes: bits 2u of qs, minus 4 unless bit 4n + u of hm is set.
__m256i q3(__m256i qs, __m256i hm, int n, int u) {
  const __m256i lo = _mm256_and_si256(_mm256_srl_epi16(qs, _mm_cvtsi32_si128(2 * u)), _mm256_set1_epi8(3));
  const __m256i hb = _mm256_and_si256(_mm256_srl_epi16(hm, _mm_cvtsi32_si128(4 * n + u)), _mm256_set1_epi8(1));
  return _mm256_sub_epi8(_mm256_or_si256(lo, _mm256_slli_epi16(hb, 2)), _mm256_set1_epi8(4));
}

float dot_q3_k(const uint8_t* w, const float* x, int64_t k) {
  __m256 a0 = _mm256_setzero_ps(), a1 = a0;  // two chains, as dot_k4
  for (int64_t b = 0; b < k / 256; ++b, w += 110, x += 256) {
    const float d = h2f(w + 108);
    int sc[16];
    scales_q3_k(w + 96, sc);
    const __m256i hm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
    for (int n = 0; n < 2; ++n) {
      const __m256i qs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32 + 32 * n));
      for (int u = 0; u < 4; ++u) {
        const int is = 8 * n + 2 * u;
        __m256& acc = u & 1 ? a1 : a0;
        acc = dot32x2(q3(qs, hm, n, u), x + 128 * n + 32 * u, d * float(sc[is]), d * float(sc[is + 1]), acc);
      }
    }
  }
  return hsum(_mm256_add_ps(a0, a1));
}

float dot_q6_k(const uint8_t* w, const float* x, int64_t k) {
  const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(3), off = _mm256_set1_epi8(32);
  __m256 acc = _mm256_setzero_ps();
  for (int64_t b = 0; b < k / 256; ++b, w += 210, x += 256) {
    const uint8_t* ql = w;
    const uint8_t* qh = w + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(w + 192);
    const float d = h2f(w + 208);
    for (int n = 0; n < 256; n += 128, ql += 64, qh += 32, sc += 8) {
      const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
      const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
      const auto high = [&](int shift) {  // bits shift, shift + 1 of h, moved to 4 and 5
        return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), m2), 4);
      };
      const auto q = [&](__m256i nib, int shift) { return _mm256_sub_epi8(_mm256_or_si256(nib, high(shift)), off); };
      const __m256i q1 = q(_mm256_and_si256(a, m4), 0), q2 = q(_mm256_and_si256(c, m4), 2);
      const __m256i q3 = q(_mm256_and_si256(_mm256_srli_epi16(a, 4), m4), 4);
      const __m256i q4 = q(_mm256_and_si256(_mm256_srli_epi16(c, 4), m4), 6);
      const float* xa = x + n;
      acc = dot32x2(q1, xa, d * float(sc[0]), d * float(sc[1]), acc);
      acc = dot32x2(q2, xa + 32, d * float(sc[2]), d * float(sc[3]), acc);
      acc = dot32x2(q3, xa + 64, d * float(sc[4]), d * float(sc[5]), acc);
      acc = dot32x2(q4, xa + 96, d * float(sc[6]), d * float(sc[7]), acc);
    }
  }
  return hsum(acc);
}

float dot_q8_0(const uint8_t* w, const float* x, int64_t k) {
  __m256 acc = _mm256_setzero_ps();
  for (int64_t b = 0; b < k / 32; ++b, w += 34, x += 32) {
    const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 2));
    acc = _mm256_fmadd_ps(_mm256_set1_ps(h2f(w)), dot32<false>(q, x), acc);
  }
  return hsum(acc);
}

// 8 bytes of b as f32 times s, minus m: a product then a difference, as the reference rounds them.
template <bool U>
void put8(float* y, __m128i b, __m256 s, __m256 m) {
  _mm256_storeu_ps(y, _mm256_sub_ps(_mm256_mul_ps(s, f8<U>(b)), m));
}

// 32 bytes q decoded as s q - m into y[0, 32).
template <bool U>
void put32(float* y, __m256i q, float s, float m) {
  const __m256 sv = _mm256_set1_ps(s), mv = _mm256_set1_ps(m);
  const __m128i lo = _mm256_castsi256_si128(q), hi = _mm256_extracti128_si256(q, 1);
  put8<U>(y, lo, sv, mv);
  put8<U>(y + 8, _mm_srli_si128(lo, 8), sv, mv);
  put8<U>(y + 16, hi, sv, mv);
  put8<U>(y + 24, _mm_srli_si128(hi, 8), sv, mv);
}

template <bool Q5>
void decode_k4(const uint8_t* w, int64_t k, float* y) {
  constexpr int BYTES = Q5 ? 176 : 144;
  const __m256i m4 = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
  for (int64_t b = 0; b < k / 256; ++b, w += BYTES) {
    const float d = h2f(w), min = h2f(w + 2);
    const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16));
    const uint8_t* q = w + (Q5 ? 48 : 16);
    for (int is = 0; is < 8; is += 2, q += 32, y += 64) {
      uint8_t sc, m;
      scale_min_k4(is, w + 4, &sc, &m);
      const float d1 = d * sc, m1 = min * m;
      scale_min_k4(is + 1, w + 4, &sc, &m);
      const float d2 = d * sc, m2 = min * m;
      const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
      __m256i lo = _mm256_and_si256(v, m4), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
      if (Q5) {
        const __m128i s1 = _mm_cvtsi32_si128(is), s2 = _mm_cvtsi32_si128(is + 1);
        lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, s1), one), 4));
        hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, s2), one), 4));
      }
      put32<true>(y, lo, d1, m1);
      put32<true>(y + 32, hi, d2, m2);
    }
  }
}

void decode_q4_k(const uint8_t* w, int64_t k, float* y) { decode_k4<false>(w, k, y); }
void decode_q5_k(const uint8_t* w, int64_t k, float* y) { decode_k4<true>(w, k, y); }

void decode_q3_k(const uint8_t* w, int64_t k, float* y) {
  const __m256 zero = _mm256_setzero_ps();
  for (int64_t b = 0; b < k / 256; ++b, w += 110) {
    const float d = h2f(w + 108);
    int sc[16];
    scales_q3_k(w + 96, sc);
    const __m256i hm = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
    for (int n = 0; n < 2; ++n) {
      const __m256i qs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 32 + 32 * n));
      for (int u = 0; u < 4; ++u, y += 32) {  // y[128 n + 32 u + l] = (d sc[8 n + 2 u + l / 16]) q, as the reference
        const __m256i q = q3(qs, hm, n, u);
        const __m128i lo = _mm256_castsi256_si128(q), hi = _mm256_extracti128_si256(q, 1);
        const __m256 s0 = _mm256_set1_ps(d * float(sc[8 * n + 2 * u])), s1 = _mm256_set1_ps(d * float(sc[8 * n + 2 * u + 1]));
        put8<false>(y, lo, s0, zero);
        put8<false>(y + 8, _mm_srli_si128(lo, 8), s0, zero);
        put8<false>(y + 16, hi, s1, zero);
        put8<false>(y + 24, _mm_srli_si128(hi, 8), s1, zero);
      }
    }
  }
}

void decode_q6_k(const uint8_t* w, int64_t k, float* y) {
  const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(3), off = _mm256_set1_epi8(32);
  const __m256 zero = _mm256_setzero_ps();
  for (int64_t b = 0; b < k / 256; ++b, w += 210) {
    const uint8_t* ql = w;
    const uint8_t* qh = w + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(w + 192);
    const float d = h2f(w + 208);
    for (int n = 0; n < 256; n += 128, ql += 64, qh += 32, sc += 8, y += 128) {
      const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
      const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
      const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
      const auto high = [&](int shift) { return _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, shift), m2), 4); };
      const auto q = [&](__m256i nib, int shift) { return _mm256_sub_epi8(_mm256_or_si256(nib, high(shift)), off); };
      const __m256i qs[4] = {q(_mm256_and_si256(a, m4), 0), q(_mm256_and_si256(c, m4), 2),
                             q(_mm256_and_si256(_mm256_srli_epi16(a, 4), m4), 4),
                             q(_mm256_and_si256(_mm256_srli_epi16(c, 4), m4), 6)};
      for (int u = 0; u < 4; ++u) {  // y[32 u + l] = (d sc[2 u + l / 16]) q, as the reference's (d sc) q
        const __m128i lo = _mm256_castsi256_si128(qs[u]), hi = _mm256_extracti128_si256(qs[u], 1);
        const __m256 s0 = _mm256_set1_ps(d * float(sc[2 * u])), s1 = _mm256_set1_ps(d * float(sc[2 * u + 1]));
        put8<false>(y + 32 * u, lo, s0, zero);
        put8<false>(y + 32 * u + 8, _mm_srli_si128(lo, 8), s0, zero);
        put8<false>(y + 32 * u + 16, hi, s1, zero);
        put8<false>(y + 32 * u + 24, _mm_srli_si128(hi, 8), s1, zero);
      }
    }
  }
}

void decode_q8_0(const uint8_t* w, int64_t k, float* y) {
  const __m256 zero = _mm256_setzero_ps();
  for (int64_t b = 0; b < k / 32; ++b, w += 34, y += 32) {
    const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 2));
    const __m256 d = _mm256_set1_ps(h2f(w));
    const __m128i lo = _mm256_castsi256_si128(q), hi = _mm256_extracti128_si256(q, 1);
    put8<false>(y, lo, d, zero);
    put8<false>(y + 8, _mm_srli_si128(lo, 8), d, zero);
    put8<false>(y + 16, hi, d, zero);
    put8<false>(y + 24, _mm_srli_si128(hi, 8), d, zero);
  }
}

}  // namespace

namespace cpu {

void fast_dot_tile(const float* const* w, const float* const* x, int64_t k, float* out) {
  // 4 w by 3 x: 12 accumulators, the 3 x of a step held, one w at a time (16 registers).
  __m256 a[12];
  for (__m256& v : a) v = _mm256_setzero_ps();
  for (int64_t i = 0; i < k; i += 8) {
    const __m256 x0 = _mm256_loadu_ps(x[0] + i), x1 = _mm256_loadu_ps(x[1] + i), x2 = _mm256_loadu_ps(x[2] + i);
    for (int c = 0; c < 4; ++c) {
      const __m256 u = _mm256_loadu_ps(w[c] + i);
      a[3 * c] = _mm256_fmadd_ps(u, x0, a[3 * c]);
      a[3 * c + 1] = _mm256_fmadd_ps(u, x1, a[3 * c + 1]);
      a[3 * c + 2] = _mm256_fmadd_ps(u, x2, a[3 * c + 2]);
    }
  }
  for (int t = 0; t < 12; ++t) out[t] = hsum(a[t]);
}

}  // namespace cpu
