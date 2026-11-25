// Included by ops.cu inside namespace cuda after split_len, on CUDA only: the kernel is sm_80 inline PTX.
#pragma once

// HD128/F16 KV, page16 split-K decode, FA2-style. A CTA is one (row, kv head, split) and reads each K/V page once for
// the whole GQA group: its G <= 8 query heads are the mma M rows (padded to 16). Each of the four warps computes the
// page's full S = Q K^T and the same softmax state, then PV for its own 32 output dims, so no warp merges with
// another. Pages of 16 positions from the split's start (so one may straddle two blocks) are staged by cp.async,
// STAGES deep. The partials are attn_tiled_k's, so attn_merge_k and the capture arena's budget serve both.
// Decode reads every page once, so the copies set its speed: a warp copies whole 128-byte lines of four positions into
// row-major pages whose 16-byte chunks are XOR-swizzled by position (no bank conflict for the copies or ldmatrix).
// sm_120, m1/m5 heads at 16 rows x 4k with the math stubbed: prefill's chunk-major map (a warp copies 32 bytes of 16
// positions) moved 463/480 GB/s, whole lines into chunk-major pages 574/553, this layout 826/798.
namespace decode {
constexpr int KT = 16, HD = 128, LD = KT + 1;  // LD pads the FP32 P rows off one bank
// Pages in flight a CTA. sm_120, the m1/m5 grid of attention --bench decode: 3 (27.8 KB, 3 CTAs an SM instead of 4) was
// up to 22 % slower at 1k-2k contexts and 1 row x 16k, at most 3.5 % faster (m5 1 row x 4k/8k).
constexpr int STAGES = 2;
constexpr int KV_BYTES = KT * HD * 2;
constexpr int SMEM = STAGES * 2 * KV_BYTES + 4 * 8 * LD * 4;  // the K/V ring, then each warp's FP32 P of 8 heads

// The 16-byte chunk c (eight dims) of position p of a page sits at byte swizzle(p, c).
__device__ __forceinline__ uint32_t swizzle(int p, int c) { return uint32_t(p * 16 + (c ^ p % 8)) * 16; }
}  // namespace decode

// On sm_120 (CUDA 13.3) it takes 119 registers, STACK:0, 4 CTAs an SM; a 5-CTA bound (96) spilled 24 bytes and was
// 1-10 % slower at 37 of the 40 grid points.
__global__ void __launch_bounds__(128) attn_decode_tc_k(attn_args a, int64_t /* rows: the grid covers them */,
                                                        int splits, int share, int64_t span, float* partial) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace decode;
  using prefill::hi_lo, prefill::ldmatrix, prefill::mma, prefill::pack, prefill::scales, prefill::scan_nonfinite;
  __shared__ __align__(128) uint8_t tiles[SMEM];
  const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  // attend_split keeps rows * kv_heads * splits under 2^31.
  const int split = int(blockIdx.x % unsigned(splits));
  const int kvh = int(blockIdx.x / unsigned(splits) % unsigned(a.kv_heads));
  const int64_t row = blockIdx.x / unsigned(splits) / unsigned(a.kv_heads), n = a.lens[row * a.lens_ld];
  const int64_t per_split = split_len(n, splits, share, span);
  // A capped split past its row's length has no work, and the merge never reads it.
  if (span && split * per_split >= n) return;
  const int64_t begin = split * per_split, end = min(n, begin + per_split);
  // Row g of the A fragments is head g: q[k][0] holds dims 16k + 2t, +1 and q[k][2] dims 16k + 2t + 8, +9; rows g + 8
  // (q[k][1], q[k][3]) and heads past the group stay zero. q may be only 2-byte aligned, and each thread loads it once.
  const int group = int(a.group), h = kvh * group + g;
  uint32_t q[8][4] = {};
  if (g < group) {
    const __half* at = a.q + row * a.q_ld + int64_t(h) * HD + 2 * t;
#pragma unroll
    for (int k = 0; k < 8; ++k) q[k][0] = pack(at[16 * k], at[16 * k + 1]), q[k][2] = pack(at[16 * k + 8], at[16 * k + 9]);
  }
  // This thread copies chunks c0 and c0 + 8 (at copy_at and copy_at + 128) of position tid / 8 of every K and V page,
  // so eight lanes read a whole 128-byte line; only the table lookup (block_tokens is KT) moves with the page.
  const int copy_pos = tid / 8, c0 = tid % 8;
  const uint32_t copy_at = swizzle(copy_pos, c0);
  const int64_t slot_ld = a.kv_heads * HD;
  const __half *const k_src = a.k + kvh * HD + c0 * 8, *const v_src = a.v + kvh * HD + c0 * 8;
  const int32_t* const blocks = a.table + row * a.table_ld;
  const auto stage = [&](int64_t p0, int buffer) {
    const int64_t pos = p0 + copy_pos;
    const bool real = pos < end;
    const int64_t at = real ? (int64_t(blocks[pos / KT]) * KT + pos % KT) * slot_ld : 0;
    uint8_t* const k = tiles + buffer * 2 * KV_BYTES + copy_at;
    // A position past the split is zero-filled, never read: past the row it may be another sequence's or garbage.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      __pipeline_memcpy_async(k + i * 128, k_src + at + i * 64, 16, real ? 0 : 16);
      __pipeline_memcpy_async(k + KV_BYTES + i * 128, v_src + at + i * 64, 16, real ? 0 : 16);
    }
  };
  const uint32_t base = uint32_t(__cvta_generic_to_shared(tiles));
  float* const fp32_p = reinterpret_cast<float*>(tiles + STAGES * 2 * KV_BYTES) + warp * 8 * LD;
  // o[i] holds head g's dims 32 warp + 8i + 2t, +1.
  float o[4][2] = {}, m = -INFINITY, l = 0;
  // One commit group a page, empty past the end, so wait_prior counts pages.
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (begin + i * KT < end) stage(begin + i * KT, i);
    __pipeline_commit();
  }
  int buffer = 0;
  for (int64_t p0 = begin; p0 < end; p0 += KT) {
    const uint32_t k_at = buffer * 2 * KV_BYTES, v_at = k_at + KV_BYTES;
    __pipeline_wait_prior(STAGES - 2);
    // cuda_pipeline_helpers.h's wait_group asm has no "memory" clobber: keep the V scan below it.
    asm volatile("" ::: "memory");
    bool bad = false;  // a non-finite V among this thread's own copies
#pragma unroll
    for (int i = 0; i < 2; ++i) scan_nonfinite(*reinterpret_cast<const uint4*>(tiles + v_at + copy_at + i * 128), bad);
    // One barrier a page: every copy has landed, every warp is done with the slot the next stage refills, the flag is
    // shared.
    const bool nonfinite = __syncthreads_or(bad);
    if (const int64_t ahead = p0 + (STAGES - 1) * KT; ahead < end) stage(ahead, (buffer + STAGES - 1) % STAGES);
    __pipeline_commit();
    const int count = int(min(int64_t(KT), end - p0));  // positions of the page in the split: at least one
    float s[2][4] = {};  // s[j][e]: head g, position 8j + 2t + e (e = 2, 3: the zero rows g + 8)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        uint32_t kf[4];
        ldmatrix<false>(kf, base + k_at + swizzle(8 * j + lane % 8, 4 * k + lane / 8));
        mma(s[j], q[2 * k], kf[0], kf[1]);
        mma(s[j], q[2 * k + 1], kf[2], kf[3]);
      }
    // Head g's softmax, as prefill's per row; the exponentials are computed unmasked and then selected.
    float top = -INFINITY;
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e)
        if (8 * j + 2 * t + e < count) top = fmaxf(top, s[j][e] * a.scale);
    top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
    top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
    const float next = fmaxf(m, top), corr = l == 0 ? 0.0f : expf(m - next);
    // The page's panels are split at its own scale (scales()): P far below the running max keeps 22 bits.
    float largest = 0;
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float& p = s[j][e];
        const float score = __fmul_rn(p, a.scale), weight = expf(__fsub_rn(score, next));
        // -Inf weighs 0 also while next is -Inf (exp(-Inf + Inf) = NaN), as under the reference's global max.
        p = 8 * j + 2 * t + e < count && score != -INFINITY ? weight : 0.0f;
        largest = fmaxf(largest, p);
      }
    largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, 1));
    largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, 2));
    float up, down, total = 0;
    scales(largest, up, down);
    uint32_t ph[4] = {}, pl[4] = {};  // P * 2^shift as A fragments: its half (high) and half((it - high) * 2^11)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      __half high[2], low[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        total += s[j][e];
        hi_lo(s[j][e], up, high[e], low[e]);
      }
      ph[2 * j] = pack(high[0], high[1]);
      pl[2 * j] = pack(low[0], low[1]);
    }
    total += __shfl_xor_sync(0xffffffffu, total, 1);
    total += __shfl_xor_sync(0xffffffffu, total, 2);
    l = l * corr + total;
    m = next;
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i][0] *= corr, o[i][1] *= corr;
    // A masked P is exactly 0 in both panels and a zero-filled position holds V = 0; only a non-finite V would make
    // 0 * Inf = NaN, so such a page takes FP32 over the positions in the split.
    if (!nonfinite) {
#pragma unroll
      for (int i = 0; i < 4; i += 2) {
        uint32_t vf[4];  // V fragments of this warp's dims 8i.., then 8i + 8..
        ldmatrix<true>(vf, base + v_at + swizzle(lane % 16, 4 * warp + i + lane / 16));
#pragma unroll
        for (int x = 0; x < 2; ++x) {
          float high[4] = {}, low[4] = {};
          mma(high, ph, vf[2 * x], vf[2 * x + 1]);
          mma(low, pl, vf[2 * x], vf[2 * x + 1]);
#pragma unroll
          for (int c = 0; c < 2; ++c) o[i + x][c] += (high[c] + low[c] * 0x1p-11f) * down;
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) fp32_p[g * LD + 8 * j + 2 * t + e] = s[j][e];
      __syncwarp();
      const float* weight = fp32_p + g * LD;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        // The page's FP32 sum meets O once, as the tensor panels do.
        float2 add{};
        for (int p = 0; p < count; ++p) {
          const float2 v = __half22float2(
              *reinterpret_cast<const __half2*>(tiles + v_at + swizzle(p, 4 * warp + i) + 4 * t));
          add.x += weight[p] * v.x;
          add.y += weight[p] * v.y;
        }
        o[i][0] += add.x;
        o[i][1] += add.y;
      }
    }
    buffer = buffer + 1 < STAGES ? buffer + 1 : 0;
  }
  // Only the group's heads write: head g past it is the next kv head's (or past the row).
  if (g >= group) return;
  if (partial) {
    float* const part = partial + ((row * a.heads + h) * splits + split) * (HD + 2);
    if (warp == 0 && t == 0) part[HD] = m, part[HD + 1] = l;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int e = 0; e < 2; ++e) part[warp * 32 + 8 * i + 2 * t + e] = o[i][e];
  } else {
    // l is also 0 after -Inf scores alone: 0 / 0 is the reference's NaN, and only a padding row writes zeros.
    __half* const out = a.out + row * a.out_ld + int64_t(h) * HD + warp * 32 + 2 * t;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int e = 0; e < 2; ++e) out[8 * i + e] = __float2half_rn(n ? o[i][e] / l : 0.0f);
  }
#elif defined(__CUDA_ARCH__)
  asm volatile("trap;");  // The host routes here only from sm_80; a build for an older arch must not run it.
#endif
}
