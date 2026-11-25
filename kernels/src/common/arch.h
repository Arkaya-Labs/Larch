// Whether a build's GPU code runs on a device, from the device's architecture alone. No context is
// created, so counting devices has no side effects and a busy or broken GPU cannot hide the others.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string_view>

namespace arch {

// CUDA. `built` lists the build's architectures as __CUDA_ARCH__ spells them (nvcc's
// __CUDA_ARCH_LIST__: 860, 890, 1200) and `cc` is the device's the same way. SASS for X.Y runs on
// X.Z for Z >= Y; the highest entry's PTX JITs on every later capability.
// ponytail: __CUDA_ARCH_LIST__ does not say which entries carry PTX, so the highest is taken to
// (the default list's 120-virtual, which backend-builds checks); read CMake's list if a build drops it

// The __CUDA_ARCH__ of the code the driver runs on `cc`: the highest same-major entry <= cc, else the highest's PTX.
inline int cuda_image(const int* built, size_t n, int cc) {
  int same = 0, top = 0;
  for (size_t i = 0; i < n; ++i) {
    if (built[i] / 100 == cc / 100 && built[i] <= cc) same = std::max(same, built[i]);
    top = std::max(top, built[i]);
  }
  return same ? same : top;
}

// Whether that code runs: SASS of the device's major, or PTX from an earlier capability.
inline bool cuda_runs(const int* built, size_t n, int cc) { return n > 0 && cuda_image(built, n, cc) <= cc; }

// HIP. A code object runs only on its own gfx target: `built` is the comma-separated
// CMAKE_HIP_ARCHITECTURES and `gcn` the device's gcnArchName ("gfx1030:sramecc-:xnack-"); target
// features after ':' are ignored on both sides.
inline bool hip_runs(std::string_view built, std::string_view gcn) {
  gcn = gcn.substr(0, gcn.find(':'));
  while (!built.empty()) {
    const size_t comma = built.find(',');
    const std::string_view one = built.substr(0, comma);
    if (one.substr(0, one.find(':')) == gcn) return true;
    built = comma == std::string_view::npos ? std::string_view() : built.substr(comma + 1);
  }
  return false;
}

// The archs where the tensor-core prefill defaults on: its ctest and parity passed and prefill gained there.
// ponytail: sm_86 (3050) and sm_120 (5070 Ti), Oct 2026: m6 heads (24q/8kv, hd 128) tensor-core vs CUDA-core
// prefill is 1.4x at 8 rows to 26x at 256 rows over 1024 history; attn_prefill_tc_k on sm_86 takes 222 registers and
// STACK:0 (CUDA 12.8). The sm_86 numbers are a 16-SM 3050 Laptop's: re-measure on a large sm_86 part (e.g. a 3090,
// 82 SMs) when one enters a bench. sm_89 by judgment, not measured: the same mma.sync f16 code as sm_86 (prefill.cuh
// records STACK:0 for the sm_89 build; re-check with cuobjdump); drop 89 if ctest attention_prefill_tc or
// `larch parity --mode all` fails or its W4 rows lose on the 4060.
// ponytail: HD256 (attn_prefill_hd256_k) shares this list, without prefill_tc's crowding rule (it won crowded tiles
// too). sm_120 measured (Nov 2026, numbers at the route in ops.cu); sm_86 and sm_89 by the judgment above. Give HD256
// its own list without 86 (3050) or 89 (4060) if, on that GPU, `ctest -R attention` fails or
// `LARCH_PREFILL_TC=1 attention --bench hd256 512 65536` is slower than the same with LARCH_PREFILL_TC=0.
inline constexpr int PREFILL_TC_DEFAULT_ARCHES[] = {86, 89, 120};

// CUDA. The most rows at which quant.cu's MMVQ_ROUTE takes the multi-row mmvq_rows_k (2-16 rows, byte-equal to one-row
// mmvq), 1 where it never does. sm_120 measured every row (#281). sm_89 (2-8) and sm_86 (2-4) by judgment, not
// measurement: the kernel is FMA-bound from 8 rows on sm_120 (0.61 of copy peak); a 4060 has about sm_120's FP32 FMA
// per byte of DRAM bandwidth and a 3050 Laptop about two thirds, so each keeps only the rows where sm_120 had that
// much headroom. Raise either when the lever (mmvq_bench) runs on the 4060 or the 3050.
inline int mmvq_rows_up_to(int arch_code) {
  switch (arch_code) {
    case 120: return 16;
    case 89: return 8;
    case 86: return 4;
    default: return 1;
  }
}

// CUDA. Whether the tensor-core prefill runs `rows` rows over `seqs` sequences on `arch_code` (major*10+minor);
// `switch_value` is LARCH_PREFILL_TC: 1 forces it on, 0 off, -1 the default.
// The default leaves it where its 64-row tiles hold too many of the S sequences over R rows: a tile walks each of its
// 64 S / max(R, 64) sequences in series, and it stays in one wave to 256 rows while the CUDA-core cost grows with R, so
// it loses past max(R, 64) / 16 a tile: S * 1024 > max(R, 64)^2. Under 32 rows it stays: the CUDA-core tiles (4 rows
// under 64 rows, 8 from 64) straddle sequences too. sm_120, m1 heads, tensor-core vs CUDA-core us (Oct 2026): one
// sequence, it wins at every measured point, 8 rows without history included (3.8 vs 4.3; 32 at 0: 6.4 vs 13.8; 8 at
// 4096: 334 vs 1503); routed (4096 history) 64 x 8 2683 vs 1441, 192 x 48 5659 vs 5189; kept 64 x 2 678 vs 1427,
// 256 x 64 5674 vs 6943, 16 x 16 5375 vs 5878; the kept 64 x 4 and 128 x 16 tie at 256 history.
// ponytail: measured on equal splits to 256 rows; per-sequence route if ragged multi-sequence prefill shows in a W4
// profile.
// ponytail: sm_86 also wins crowded tiles (64 rows x 8 sequences at 4096 history: 4027 vs 14219 us) but shares
// sm_120's threshold; a per-arch crowding threshold if crowded multi-sequence prefill shows in an sm_86 W4 profile.
inline bool prefill_tc(int switch_value, int arch_code, int64_t rows, int64_t seqs) {
  if (switch_value >= 0) return switch_value == 1;
  const int64_t span = std::max(rows, int64_t(64));
  return std::count(std::begin(PREFILL_TC_DEFAULT_ARCHES), std::end(PREFILL_TC_DEFAULT_ARCHES), arch_code) > 0 &&
         !(rows >= 32 && seqs * 1024 > span * span);
}

// CUDA. The archs where one-row mmvq's Q4_K, Q6_K and Q3_K decode in half2 math (LARCH_DECODE_H2=0 turns it off; the
// switch cannot force it on elsewhere). sm_86 and sm_89: the 1-row decode is FMA- and issue-bound at 20 W on sm_89, and
// half2 does two weights per instruction. sm_120 keeps the f32 decoder. ctest verify_f32 pins the off path's bytes
// (Q3_K's once verify.cpp's PINNED has its hashes).
// ponytail: sm_86 by judgment (same half2 rate as sm_89), untimed and without a 3050 parity or bench run; drop 86 if
// ctest or `larch parity` fails on the 3050 or its decode bench loses to LARCH_DECODE_H2=0.
inline constexpr int MMVQ_H2_DEFAULT_ARCHES[] = {86, 89};
/// Whether `arch_code` is in MMVQ_H2_DEFAULT_ARCHES; constexpr for quant.cu's H2 kernels, which trap elsewhere.
inline constexpr bool mmvq_h2_arch(int arch_code) {
  return std::count(std::begin(MMVQ_H2_DEFAULT_ARCHES), std::end(MMVQ_H2_DEFAULT_ARCHES), arch_code) > 0;
}
inline bool mmvq_h2(int arch_code, int switch_value) { return switch_value != 0 && mmvq_h2_arch(arch_code); }

// The archs where the tensor-core decode defaults on: its ctest and parity passed and decode gained there.
// ponytail: sm_120 only (5070 Ti, Oct 2026); add 86 and 89 when ctest attention_decode_tc and `larch parity` with
// LARCH_DECODE_TC=1 pass on the 3050 and the 4060 and their W2/W4 rows gain, and flip the sm_86 decode row of
// check_route in kernels/tests/attention.cpp with them.
inline constexpr int DECODE_TC_DEFAULT_ARCHES[] = {120};

// CUDA. Whether decode takes the tensor-core kernel on `arch_code` (major*10+minor); `switch_value` is
// LARCH_DECODE_TC: 1 forces it on, 0 off, -1 the arch default.
// The default takes every group the kernel runs (ops.cu: to 8), all measured winning on sm_120 (Oct 2026): groups 4
// and 7 (m1/m5 heads) at every point of attention --bench decode's grid, and at kv 8 x rows 1/8/32 x 1k/4k/16k
// against the CUDA-core decode group 1 by 7.5-34 %, group 2 by 14-39 %, group 8 by 60-71 %. HD256 (16q/2kv, group 8)
// shares the list: one row at 32k and 128k context 107 vs 612 us and 348 vs 2692 us (sm_120, Nov 2026).
inline bool decode_tc(int switch_value, int arch_code) {
  if (switch_value >= 0) return switch_value == 1;
  return std::count(std::begin(DECODE_TC_DEFAULT_ARCHES), std::end(DECODE_TC_DEFAULT_ARCHES), arch_code) > 0;
}

}  // namespace arch
