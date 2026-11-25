// The fast CPU backend's Q3_K and Q4_K mmvq_id on x86-64 with AVX-512 VNNI (kernels/CMakeLists.txt gives this
// file AVX-512 F/CD/BW/DQ/VL; VNNI comes from the attribute below; fast_cpu.cpp checks the CPU before it calls
// in). x is exact integers, not int8 activations: each 256 block of x as round(x 2^(22 - e)) for its max below
// 2^e, in three balanced int8 limbs, so vpdpbusd's sums of q x are exact and only x's bits under 2^(e - 23) are
// lost, where f32 products round at 2^-24 of each partial sum. A Q3_K weight decoding to 0 adds exactly 0. Q4_K's
// d sc sum(q x) - dmin m sum(x) cancels integer sums that are exact until their f32 conversion (up to 2^28, so
// rounded at 2^-24 like the products): what it leaves is f32 rounding, as in the AVX2 path.
#include <immintrin.h>

#include <cstring>

#include "larch.h"
#include "cpu/fast_cpu.h"

// INLINE for the dots, so the row loop holds their constants.
#if defined(_MSC_VER) && !defined(__clang__)
#define VNNI
#define INLINE __forceinline
#else
#define VNNI __attribute__((target("avx512vnni")))
#define INLINE inline __attribute__((always_inline))
#endif

namespace {

// 64 weights' u8 codes q times x's three limbs at p (64 bytes each, top first): per lane, the sum of its 4
// q xi, exact in int32 (|q| < 16, |xi| <= 2^22).
VNNI __m512i dot64(__m512i q, const int8_t* p) {
  __m512i a = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q, _mm512_loadu_si512(p));
  a = _mm512_dpbusd_epi32(_mm512_slli_epi32(a, 8), q, _mm512_loadu_si512(p + 64));
  return _mm512_dpbusd_epi32(_mm512_slli_epi32(a, 8), q, _mm512_loadu_si512(p + 128));
}

__m512i both(__m256i v) { return _mm512_broadcast_i64x4(v); }

// Lane counts for 16-bit shifts: a in the low 256 bits, b in the high.
__m512i counts(short a, short b) { return _mm512_inserti64x4(_mm512_set1_epi16(a), _mm256_set1_epi16(b), 1); }

// One x row as fast_vnni_prep writes it: xl 192 bytes per 64 weights, xs 16 per 64, xm 1 per 32, xe 1 per 256.
struct xrow {
  const int8_t* xl;
  const int32_t* xs;
  const float* xm;
  const float* xe;
};

// Q3_K: codes q + 4 = lo | hb << 2 (0..7) with lo bits 2u of qs[32 n + l], hb bit 4 n + u of hm[l], for weight
// 128 n + 32 u + l; sub-block 16 weights. Group g = 2 n + h of a block: quarters 2 h, 2 h + 1 of half n, weights
// 64 g + [0, 64), sub-blocks 4 g + lane / 4. The 4 sum(x) a lane's codes overcount (xs) go first, in integers.
INLINE VNNI __m512 q3_group(__m512i lo, __m512i hm, int g, const int8_t* xl, const int32_t* xs, __m512 s16,
                            __m512 acc) {
  // hm's bits 2 g, 2 g + 1 moved to bit 2 (left for g = 0), joined to lo's bits 0 and 1.
  const __m512i hb = g ? _mm512_srlv_epi16(hm, counts(short(2 * g - 2), short(2 * g - 1)))
                        : _mm512_sllv_epi16(hm, counts(2, 1));
  const __m512i q =  // (lo & 3) | (hb & 4)
      _mm512_ternarylogic_epi32(lo, _mm512_and_si512(hb, _mm512_set1_epi8(4)), _mm512_set1_epi8(3), 0xEC);
  const __m512i v = _mm512_sub_epi32(dot64(q, xl + 192 * g), _mm512_loadu_si512(xs + 16 * g));
  const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const __m512i idx = _mm512_srli_epi32(_mm512_add_epi32(_mm512_set1_epi32(16 * g), lane), 2);  // 4 g + lane / 4
  return _mm512_fmadd_ps(_mm512_cvtepi32_ps(v), _mm512_permutexvar_ps(idx, s16), acc);
}

INLINE VNNI float dot_q3_k(const uint8_t* w, xrow x, int64_t k) {
  const __m128i lo4 = _mm_set1_epi8(0x0F);
  const __m128i hi_idx = _mm_setr_epi8(8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 9, 10, 11);
  const __m128i hi_cnt = _mm_setr_epi16(0, 0, 2, 2, 4, 4, 6, 6);
  __m512 a0 = _mm512_setzero_ps(), a1 = a0;  // two chains: one waits on its fma
  for (int64_t b = 0; b < k / 256; ++b, w += 110) {
    // The 12 scale bytes and d, masked: a 16-byte load would read past the tensor's last block.
    const __m128i s = _mm_maskz_loadu_epi8(0x3FFF, w + 96);
    const uint16_t dh = uint16_t(_mm_extract_epi16(s, 6));
    // The 16 scales as the reference's scales_q3_k: low nibbles of s[i] then s[i - 8], bits 2 (i / 4) of s[8 + i % 4].
    const __m128i low = _mm_unpacklo_epi64(_mm_and_si128(s, lo4), _mm_and_si128(_mm_srli_epi16(s, 4), lo4));
    const __m128i high = _mm_and_si128(_mm_srlv_epi16(_mm_shuffle_epi8(s, hi_idx), hi_cnt), _mm_set1_epi8(3));
    const __m128i sc = _mm_sub_epi8(_mm_or_si128(low, _mm_slli_epi16(high, 4)), _mm_set1_epi8(32));
    const __m512 d = _mm512_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(dh))));  // MSVC lacks _cvtsh_ss
    const __m512 s16 =
        _mm512_mul_ps(_mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(sc))), _mm512_set1_ps(x.xe[b]));
    const __m512i hm = both(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w)));
    // Half n's qs as [qs, qs >> 2]: quarters 0 and 1 in its bits 0 and 1, 2 and 3 in bits 4 and 5.
    const auto half = [](const uint8_t* p) {
      return _mm512_srlv_epi16(both(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))), counts(0, 2));
    };
    const __m512i q0 = half(w + 32), q1 = half(w + 64);
    const int8_t* xl = x.xl + 768 * b;
    const int32_t* xs = x.xs + 64 * b;
    a0 = q3_group(q0, hm, 0, xl, xs, s16, a0);
    a1 = q3_group(_mm512_srli_epi16(q0, 4), hm, 1, xl, xs, s16, a1);
    a0 = q3_group(q1, hm, 2, xl, xs, s16, a0);
    a1 = q3_group(_mm512_srli_epi16(q1, 4), hm, 3, xl, xs, s16, a1);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
}

// Q4_K: nibbles of qs[32 g + l], low for weight 64 g + l (sub-block 2 g), high for 64 g + 32 + l (2 g + 1);
// d sc sum(q x) per lane, then dmin m sum(x) per sub-block, both sums exact integers rounded once to f32.
INLINE VNNI __m512 q4_group(const uint8_t* w, int g, const int8_t* xl, __m512 v16, __m512 acc) {
  const __m512i q = _mm512_and_si512(
      _mm512_srlv_epi16(both(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16 + 32 * g))), counts(0, 4)),
      _mm512_set1_epi8(0x0F));
  const __m512i di = _mm512_inserti64x4(_mm512_set1_epi32(2 * g), _mm256_set1_epi32(2 * g + 1), 1);
  return _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot64(q, xl + 192 * g)), _mm512_permutexvar_ps(di, v16), acc);
}

INLINE VNNI float dot_q4_k(const uint8_t* w, xrow x, int64_t k) {
  const __m128i k1 = _mm_set1_epi8(0x3F), k2 = _mm_set1_epi8(0x0F), k3 = _mm_set1_epi8(3);
  __m512 a0 = _mm512_setzero_ps(), a1 = a0;
  for (int64_t b = 0; b < k / 256; ++b, w += 144) {
    // d and dmin, then the 12 scale bytes u0 u1 u2 (dwords 1-3); the 8 scales and the 8 mins as the reference's
    // scale_min_k4: u0 & k1, u2 & k2 | (u0 >> 6 & k3) << 4, then u1 & k1, u2 >> 4 & k2 | (u1 >> 6 & k3) << 4.
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w));
    const __m128i a = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 2, 3, 1)), c = _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 2, 1, 1));
    const __m128i odd = _mm_or_si128(_mm_and_si128(_mm_srlv_epi32(a, _mm_setr_epi32(0, 0, 0, 4)), k2),
                                     _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(c, 6), k3), 4));
    const __m128i sm = _mm_blend_epi32(_mm_and_si128(a, k1), odd, 0b1010);
    const __m128 dd = _mm_cvtph_ps(v);  // d, dmin
    const __m512 dm = _mm512_insertf32x8(_mm512_broadcastss_ps(dd), _mm256_broadcastss_ps(_mm_movehdup_ps(dd)), 1);
    // (d sc) and (dmin m) as the reference rounds them, then x's exact power of 2.
    const __m512 v16 =
        _mm512_mul_ps(_mm512_mul_ps(dm, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(sm))), _mm512_set1_ps(x.xe[b]));
    const int8_t* xl = x.xl + 768 * b;
    a0 = q4_group(w, 0, xl, v16, a0);
    a1 = q4_group(w, 1, xl, v16, a1);
    a0 = q4_group(w, 2, xl, v16, a0);
    a1 = q4_group(w, 3, xl, v16, a1);
    // The mins: lanes 8-15 of v16 times the 8 sub-blocks' sum(x), lanes 0-7 times 0.
    a0 = _mm512_fnmadd_ps(v16, _mm512_insertf32x8(_mm512_setzero_ps(), _mm256_loadu_ps(x.xm + 8 * b), 1), a0);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
}

// ponytail: 4 rows ahead measured -4 % on a Q3_K down at 8 rows (9800X3D, WSL2, interleaved), neutral on gate
// and up; re-derive where RAM latency or a row's compute differs.
constexpr int64_t PREFETCH = 4;

// One dot per instance, so it inlines and its constants leave the row loop.
template <bool Q3>
VNNI void items(const cpu::fast_id_args& a, int64_t first, int64_t last) {
  const bool weighted = a.wt != nullptr;
  const int64_t blocks = (a.n + cpu::FAST_COLS - 1) / cpu::FAST_COLS;
  for (int64_t i = first; i < last; ++i) {
    const int64_t pair = i / blocks, c0 = i % blocks * cpu::FAST_COLS;
    const int64_t c1 = c0 + cpu::FAST_COLS < a.n ? c0 + cpu::FAST_COLS : a.n;
    const int64_t r = weighted ? pair : pair / a.slots;
    const int64_t j0 = weighted ? 0 : pair % a.slots, j1 = weighted ? a.slots : j0 + 1;
    for (int64_t c = c0; c < c1; ++c) {
      double sum = 0;  // the biased, weighted slots in ascending order in double, rounded once, as the reference
      for (int64_t j = j0; j < j1; ++j) {
        const int64_t row = int64_t(a.ids[r * a.ids_ld + j]) * a.n + c, at = r * a.x_ld + j * a.x_slot;
        const xrow x{a.xl + at / 64 * 192, a.xs + at / 4, a.xm + at / 32, a.xe + at / 256};
        const uint8_t* wr = a.w + row * a.w_rb;
        // The weights come from RAM: the row PREFETCH on (this expert's, for a later column), ahead of the hardware,
        // clamped to w's last row.
        const uint8_t* ahead = a.w + (row + PREFETCH < a.w_rows ? row + PREFETCH : a.w_rows - 1) * a.w_rb;
        for (int64_t o = 0; o < a.w_rb; o += 64) _mm_prefetch(reinterpret_cast<const char*>(ahead + o), _MM_HINT_T0);
        double v = Q3 ? dot_q3_k(wr, x, a.k) : dot_q4_k(wr, x, a.k);
        if (a.bias) v += a.bias[row];
        if (weighted)
          sum += double(a.wt[r * a.wt_ld + j]) * v;
        else
          a.y[(r * a.slots + j) * a.n + c] = float(v);
      }
      if (weighted) a.y[r * a.n + c] = float(sum);
    }
  }
}

}  // namespace

namespace cpu {

bool fast_vnni_supports(int dtype) { return dtype == LARCH_Q3_K || dtype == LARCH_Q4_K; }

bool fast_vnni_prep(const float* x, int64_t n, int8_t* xl, int32_t* xs, float* xm, float* xe) {
  for (int64_t b = 0; b < n / 256; ++b, x += 256) {
    __m512 amax = _mm512_setzero_ps();
    __mmask16 bad = 0;
    for (int j = 0; j < 16; ++j) {
      const __m512 v = _mm512_loadu_ps(x + 16 * j);
      bad |= _mm512_fpclass_ps_mask(v, 0x99);  // QNaN, +-Inf, SNaN
      amax = _mm512_max_ps(amax, _mm512_abs_ps(v));
    }
    const float top = _mm512_reduce_max_ps(amax);
    uint32_t bits;
    std::memcpy(&bits, &top, 4);
    const uint32_t ex = bits >> 23;  // top in [2^(ex - 127), 2^(ex - 126)): scale 2^(148 - ex), xe 2^(ex - 148)
    if (bad || (top > 0 && ex < 22)) return false;
    const uint32_t xe_bits = top > 0 ? (ex - 21) << 23 : 0, scale_bits = top > 0 ? (275 - ex) << 23 : 0;
    float sf;
    std::memcpy(&xe[b], &xe_bits, 4);
    std::memcpy(&sf, &scale_bits, 4);
    const __m512 scale = _mm512_set1_ps(sf);
    int32_t s32 = 0;  // the even j's 16-sum
    for (int j = 0; j < 16; ++j) {
      const __m512i xi = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + 16 * j), scale));  // |xi| <= 2^22
      // Balanced base-256 digits: d0 = xi's low byte as signed, d1 the next of (xi - d0) / 256, d2 the rest.
      const __m512i d0 = _mm512_srai_epi32(_mm512_slli_epi32(xi, 24), 24);
      const __m512i r1 = _mm512_srai_epi32(_mm512_sub_epi32(xi, d0), 8);
      const __m512i d1 = _mm512_srai_epi32(_mm512_slli_epi32(r1, 24), 24);
      const __m512i d2 = _mm512_srai_epi32(_mm512_sub_epi32(r1, d1), 8);
      int8_t* p = xl + (4 * b + j / 4) * 192 + 16 * (j % 4);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm512_cvtepi32_epi8(d2));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p + 64), _mm512_cvtepi32_epi8(d1));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(p + 128), _mm512_cvtepi32_epi8(d0));
      __m512i s = _mm512_add_epi32(xi, _mm512_shuffle_epi32(xi, _MM_PERM_CDAB));
      s = _mm512_add_epi32(s, _mm512_shuffle_epi32(s, _MM_PERM_BADC));  // each 128-bit lane: its 4-sum, 4 times
      const __m512i pick = _mm512_setr_epi32(0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      const __m128i s4 = _mm512_castsi512_si128(_mm512_permutexvar_epi32(pick, s));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(xs + 64 * b + 4 * j), _mm_slli_epi32(s4, 2));  // Q3_K's 4 sum
      const int32_t s16 = _mm_extract_epi32(_mm_hadd_epi32(_mm_hadd_epi32(s4, s4), s4), 0);
      if (j % 2) xm[8 * b + j / 2] = float(s32 + s16);  // Q4_K's sub-block sum(x), rounded once
      s32 = s16;
    }
  }
  return true;
}

void fast_mmvq_id_vnni(const fast_id_args& a, int64_t first, int64_t last) {
  a.dtype == LARCH_Q3_K ? items<true>(a, first, last) : items<false>(a, first, last);
}

}  // namespace cpu
