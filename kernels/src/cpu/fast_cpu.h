// The fast CPU backend's ISA kernels (lane c1) and what they take. Plain data and declarations only: the ISA
// files include it, and must share no inline code with the default-flag files (kernels/CMakeLists.txt).
#pragma once

#include <cstdint>

namespace cpu {

// Columns of w one work item computes.
constexpr int64_t FAST_COLS = 16;

// mmvq_id, checked: x as f32 [rows, x_slots, k] (x_slots 1 or slots), every id in range, k a multiple of 32. y gets f32
// [rows, slots, n], or [rows, n] when weighted (wt f32 [rows, slots]).
struct fast_id_args {
  int dtype;
  const uint8_t* w;
  int64_t w_rb;
  const float* x;
  int64_t x_ld, x_slot;
  const int32_t* ids;
  int64_t ids_ld;
  const float* bias;  // f32 [experts * n] or null
  const float* wt;    // null unless weighted
  int64_t wt_ld;
  float* y;
  int64_t n, k, slots;
  const int8_t* xl = nullptr;  // fast_vnni_prep's x, for fast_mmvq_id_vnni only
  const int32_t* xs = nullptr;
  const float *xm = nullptr, *xe = nullptr;
  int64_t w_rows = 0;  // w's rows, for fast_mmvq_id_vnni's prefetch to stay inside it
};

// Work items [first, last): item i is (row, slot, column block) unweighted, (row, column block) weighted, the
// blocks FAST_COLS wide. False for a dtype the kernel lacks.
bool fast_id_supports(int dtype);
void fast_mmvq_id_isa(const fast_id_args& a, int64_t first, int64_t last);

// The AVX-512 VNNI kernel (fast_mmvq_id_avx512.cpp) for the dtypes fast_vnni_supports: fast_mmvq_id_isa's work
// items on x as exact integers. fast_vnni_prep writes them for n f32 of x (a multiple of 256): xl 3 n bytes, xs
// n / 4, xm n / 32, xe n / 256; false when x holds an infinity or NaN, or a block too small for a normal scale.
bool fast_vnni_supports(int dtype);
bool fast_vnni_prep(const float* x, int64_t n, int8_t* xl, int32_t* xs, float* xm, float* xe);
void fast_mmvq_id_vnni(const fast_id_args& a, int64_t first, int64_t last);

// Grouped prefill's tile: out[FAST_TILE_X c + p] = w[c] . x[p] for FAST_TILE_W decoded rows w and FAST_TILE_X x
// rows, all f32 of k (a multiple of 32): 4 by 3 on AVX2 (12 accumulators, 3 x and a w in its 16 registers), 4 by 4
// on NEON (21 of its 32).
constexpr int FAST_TILE_W = 4;
#if defined(__aarch64__) || defined(_M_ARM64)
constexpr int FAST_TILE_X = 4;
#else
constexpr int FAST_TILE_X = 3;
#endif
void fast_dot_tile(const float* const* w, const float* const* x, int64_t k, float* out);

// A quant row of k weights as f32, each as the reference's to_f32 decodes it (d q - m as a product then a
// difference), for grouped prefill's tile. False for a dtype the kernel lacks.
bool fast_decode_row_isa(int dtype, const uint8_t* w, int64_t k, float* out);

}  // namespace cpu
