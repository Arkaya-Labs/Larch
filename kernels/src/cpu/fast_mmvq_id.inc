// mmvq_id for the fast CPU backend, included first by the ISA file of the target (fast_mmvq_id_avx2.cpp,
// fast_mmvq_id_neon.cpp), which then defines the dot products below in its intrinsics. The dots are f32
// where the reference decodes each weight and sums in double, which mmvq_id's tolerance allows. Only the
// entry points leave this file (fast_cpu.h).
#include <cstring>

#include "cpu/fast_cpu.h"

namespace {

// IEEE f16 to f32, exact (as the reference's f16_to_f32).
float h2f(const uint8_t* p) {
  const uint32_t h = uint32_t(p[0]) | uint32_t(p[1]) << 8;
  const uint32_t sign = (h & 0x8000u) << 16, e = h >> 10 & 0x1F, m = h & 0x3FF;
  uint32_t bits;
  if (e == 0x1F) {
    bits = sign | 0x7F800000u | m << 13;
  } else if (e) {
    bits = sign | (e + 112) << 23 | m << 13;
  } else {
    float f = float(m) * 5.9604644775390625e-08f;  // m 2^-24
    std::memcpy(&bits, &f, 4);
    bits |= sign;
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = uint8_t((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

// The 16 signed 6-bit scales of a Q3_K block from its 12 scale bytes s, as the reference's scales_q3_k.
void scales_q3_k(const uint8_t* s, int* sc) {
  for (int i = 0; i < 16; ++i)
    sc[i] = ((i < 8 ? s[i] & 0xF : s[i - 8] >> 4) | (s[8 + i % 4] >> (2 * (i / 4)) & 3) << 4) - 32;
}

// The ISA file's dot products of a quant row of k weights with x (f32), in f32 lanes. The K quants with mins
// decode each weight as the reference does (d sc q - dmin m, so a weight that decodes to 0 is 0) before its
// product: d sc sum(q x) - dmin m sum(x) cancels in another order and leaves what the weights do not hold. The
// others scale each sub-block's sum of q x once (NEON's Q3_K decodes each weight too, which is faster
// there).
float dot_q3_k(const uint8_t* w, const float* x, int64_t k);
float dot_q4_k(const uint8_t* w, const float* x, int64_t k);
float dot_q5_k(const uint8_t* w, const float* x, int64_t k);
float dot_q6_k(const uint8_t* w, const float* x, int64_t k);
float dot_q8_0(const uint8_t* w, const float* x, int64_t k);

using dot_fn = float (*)(const uint8_t*, const float*, int64_t);

// The ISA file's row decoders, as the reference's to_f32 rounds: d sc q - dmin m as a product then a difference,
// d sc q as (d sc) q.
void decode_q3_k(const uint8_t* w, int64_t k, float* y);
void decode_q4_k(const uint8_t* w, int64_t k, float* y);
void decode_q5_k(const uint8_t* w, int64_t k, float* y);
void decode_q6_k(const uint8_t* w, int64_t k, float* y);
void decode_q8_0(const uint8_t* w, int64_t k, float* y);

using decode_fn = void (*)(const uint8_t*, int64_t, float*);

decode_fn decode_for(int dtype) {
  switch (dtype) {
    case LARCH_Q3_K: return decode_q3_k;
    case LARCH_Q4_K: return decode_q4_k;
    case LARCH_Q5_K: return decode_q5_k;
    case LARCH_Q6_K: return decode_q6_k;
    case LARCH_Q8_0: return decode_q8_0;
    default: return nullptr;
  }
}

// larch.h's dtype codes, as the ISA files may not include the reference's headers.
dot_fn dot_for(int dtype) {
  switch (dtype) {
    case LARCH_Q3_K: return dot_q3_k;
    case LARCH_Q4_K: return dot_q4_k;
    case LARCH_Q5_K: return dot_q5_k;
    case LARCH_Q6_K: return dot_q6_k;
    case LARCH_Q8_0: return dot_q8_0;
    default: return nullptr;
  }
}

}  // namespace

namespace cpu {

bool fast_id_supports(int dtype) { return dot_for(dtype) != nullptr; }

bool fast_decode_row_isa(int dtype, const uint8_t* w, int64_t k, float* out) {
  const decode_fn f = decode_for(dtype);
  if (f) f(w, k, out);
  return f != nullptr;
}

void fast_mmvq_id_isa(const fast_id_args& a, int64_t first, int64_t last) {
  const dot_fn dot = dot_for(a.dtype);
  const int64_t blocks = (a.n + FAST_COLS - 1) / FAST_COLS;
  const bool weighted = a.wt != nullptr;
  for (int64_t i = first; i < last; ++i) {
    const int64_t pair = i / blocks, c0 = i % blocks * FAST_COLS;
    const int64_t c1 = c0 + FAST_COLS < a.n ? c0 + FAST_COLS : a.n;
    const int64_t r = weighted ? pair : pair / a.slots;
    const int64_t j0 = weighted ? 0 : pair % a.slots, j1 = weighted ? a.slots : j0 + 1;
    for (int64_t c = c0; c < c1; ++c) {
      double sum = 0;  // the biased, weighted slots in ascending order in double, rounded once, as the reference
      for (int64_t j = j0; j < j1; ++j) {
        const int64_t row = int64_t(a.ids[r * a.ids_ld + j]) * a.n + c;
        double v = dot(a.w + row * a.w_rb, a.x + r * a.x_ld + j * a.x_slot, a.k);
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

}  // namespace cpu
