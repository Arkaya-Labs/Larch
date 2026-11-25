// Included by ops.cu inside namespace cuda after decode.cuh, on CUDA only: the kernels are sm_80 inline PTX.
#pragma once

// Head dim 256 on the tensor cores (the catalog's HD256 full-attention layers: 16 q heads, 2 kv heads), F16 KV in
// page16 tables. The decode is its HD128 sibling's design (decode.cuh: fragment helpers, P panels, non-finite V
// fallback, partial layout and attn_merge_k) retiled so twice the dims neither spill nor overflow 48 KB; the prefill
// takes prefill.cuh's helpers, panels and fallback but packs the GQA group into its tile.
// ponytail: these bodies repeat the HD128 kernels' math, so a numeric fix lands twice, to keep the HD128 hot path
// untouched; template the decode on HD (and try the packed prefill at HD128) once the K5 kill rule passes.
namespace hd256 {
constexpr int HD = 256, KT = 16, LD = KT + 1;  // LD pads the FP32 P rows off one bank
constexpr int KV_BYTES = KT * HD * 2;          // 8 KB: one page of K or V
// Decode: two pages of K and V in flight (32 KB; decode.cuh's depth, its own constant so the 48 KB limit holds when
// that one is retuned), then each warp's FP32 P of 8 heads.
constexpr int STAGES = 2, DECODE_SMEM = STAGES * 2 * KV_BYTES + 4 * 8 * LD * 4;
// Prefill: tiles of M = 64 virtual rows (a query row and head each; 32 KB of Q) or 32, and one stage of K and V: 48 KB
// at 64, the static limit.
constexpr int M = 64;
// The small-grid prefill (attn_prefill_hd256_head_k): 32 query rows of one head (16 KB of Q), two stages of K and V.
constexpr int HEAD_ROWS = 32, HEAD_Q_BYTES = HEAD_ROWS * HD * 2, HEAD_SMEM = HEAD_Q_BYTES + 4 * KV_BYTES;

// Decode pages are row-major: position p's 32 chunks of 16 bytes, chunk c at swizzle(p, c). The XOR stays inside c's
// 128-byte line, so eight lanes copying a line and every ldmatrix phase (eight positions of one chunk) hit 32 banks.
__device__ __forceinline__ uint32_t swizzle(int p, int c) { return uint32_t(p * 32 + (c ^ p % 8)) * 16; }
}  // namespace hd256

// decode.cuh's split-K decode at HD256: a CTA is one (row, kv head, split), its G <= 8 query heads the mma M rows;
// every warp computes the page's full S and softmax, then PV for its own 64 output dims. Each row of a multi-row step
// takes its own CTAs, so at the same splits its bytes are its one-row decode's.
__global__ void __launch_bounds__(128) attn_decode_hd256_k(attn_args a, int64_t /* rows: the grid covers them */,
                                                           int splits, int share, int64_t span, float* partial) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace hd256;
  using prefill::hi_lo, prefill::ldmatrix, prefill::mma, prefill::pack, prefill::scales, prefill::scan_nonfinite;
  __shared__ __align__(128) uint8_t tiles[DECODE_SMEM];
  const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  // attend_split keeps rows * kv_heads * splits under 2^31.
  const int split = int(blockIdx.x % unsigned(splits));
  const int kvh = int(blockIdx.x / unsigned(splits) % unsigned(a.kv_heads));
  const int64_t row = blockIdx.x / unsigned(splits) / unsigned(a.kv_heads), n = a.lens[row * a.lens_ld];
  const int64_t per_split = split_len(n, splits, share, span);
  // A capped split past its row's length has no work, and the merge never reads it.
  if (span && split * per_split >= n) return;
  const int64_t begin = split * per_split, end = min(n, begin + per_split);
  // Row g of the A fragments is head g (rows g + 8 and heads past the group stay zero), as in decode.cuh.
  const int group = int(a.group), h = kvh * group + g;
  uint32_t q[16][4] = {};
  if (g < group) {
    const __half* at = a.q + row * a.q_ld + int64_t(h) * HD + 2 * t;
#pragma unroll
    for (int k = 0; k < 16; ++k)
      q[k][0] = pack(at[16 * k], at[16 * k + 1]), q[k][2] = pack(at[16 * k + 8], at[16 * k + 9]);
  }
  // This thread copies chunks c0 + 8i (at copy_at + 128 i) of position tid / 8: eight lanes read a whole line.
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
    for (int i = 0; i < 4; ++i) {
      __pipeline_memcpy_async(k + i * 128, k_src + at + i * 64, 16, real ? 0 : 16);
      __pipeline_memcpy_async(k + KV_BYTES + i * 128, v_src + at + i * 64, 16, real ? 0 : 16);
    }
  };
  const uint32_t base = uint32_t(__cvta_generic_to_shared(tiles));
  float* const fp32_p = reinterpret_cast<float*>(tiles + STAGES * 2 * KV_BYTES) + warp * 8 * LD;
  // o[i] holds head g's dims 64 warp + 8i + 2t, +1.
  float o[8][2] = {}, m = -INFINITY, l = 0;
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
    for (int i = 0; i < 4; ++i) scan_nonfinite(*reinterpret_cast<const uint4*>(tiles + v_at + copy_at + i * 128), bad);
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
      for (int k = 0; k < 8; ++k) {
        uint32_t kf[4];
        ldmatrix<false>(kf, base + k_at + swizzle(8 * j + lane % 8, 4 * k + lane / 8));
        mma(s[j], q[2 * k], kf[0], kf[1]);
        mma(s[j], q[2 * k + 1], kf[2], kf[3]);
      }
    // Head g's softmax, as decode.cuh's.
    float top = -INFINITY;
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e)
        if (8 * j + 2 * t + e < count) top = fmaxf(top, s[j][e] * a.scale);
    top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
    top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
    const float next = fmaxf(m, top), corr = l == 0 ? 0.0f : expf(m - next);
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
    for (int i = 0; i < 8; ++i) o[i][0] *= corr, o[i][1] *= corr;
    // A non-finite V would make a masked 0 * Inf = NaN in the panels: such a page takes FP32 over its positions.
    if (!nonfinite) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        uint32_t vf[4];  // V fragments of this warp's dims 8i.., then 8i + 8..
        ldmatrix<true>(vf, base + v_at + swizzle(lane % 16, 8 * warp + i + lane / 16));
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
      for (int i = 0; i < 8; ++i) {
        float2 add{};
        for (int p = 0; p < count; ++p) {
          const float2 v = __half22float2(
              *reinterpret_cast<const __half2*>(tiles + v_at + swizzle(p, 8 * warp + i) + 4 * t));
          add.x += weight[p] * v.x;
          add.y += weight[p] * v.y;
        }
        o[i][0] += add.x;
        o[i][1] += add.y;
      }
    }
    buffer = buffer + 1 < STAGES ? buffer + 1 : 0;
  }
  if (g >= group) return;
  if (partial) {
    float* const part = partial + ((row * a.heads + h) * splits + split) * (HD + 2);
    if (warp == 0 && t == 0) part[HD] = m, part[HD + 1] = l;
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int e = 0; e < 2; ++e) part[warp * 64 + 8 * i + 2 * t + e] = o[i][e];
  } else {
    // l is also 0 after -Inf scores alone: 0 / 0 is the reference's NaN, and only a padding row writes zeros.
    __half* const out = a.out + row * a.out_ld + int64_t(h) * HD + warp * 64 + 2 * t;
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int e = 0; e < 2; ++e) out[8 * i + e] = __float2half_rn(n ? o[i][e] / l : 0.0f);
  }
#elif defined(__CUDA_ARCH__)
  asm volatile("trap;");  // The host routes here only from sm_80; a build for an older arch must not run it.
#endif
}

// The small-grid HD256 prefill: under a grid that fills the SMs with the GQA-packed kernel below, each of its CTAs
// walks the whole history with one or two warps, where this kernel's eight split the dims (128 rows after 110k
// history: 11.7 against 20.9 ms). prefill.cuh's FA2 prefill at HD256: a CTA is one query head x 32 rows, eight warps.
// Warp w takes rows 16 (w % 2).. and output dims 64 (w / 2)..; a warp holding all 256 dims of its 16 rows would need
// 128 O and 64 Q registers and spills (so did two warps of 128 dims each: 255 registers, spill loads in the tile loop
// on sm_86/89). The four warps of a row group each compute S over their 64 dims, swap the quarters through shared
// memory and add them in one order, so all four hold the same S, then run the same softmax and their own PV. One CTA
// an SM, as the HD128 prefill's two of four warps (CUDA 13.4): 184 registers on sm_120, 166 on sm_86/89, STACK:0; a
// two-CTA bound (128) spilled on all three.
// ponytail: two HD256 prefills, this one only for grids the packed one leaves under the SMs; split the packed kernel's
// history across CTAs (attend_split's merge) and delete this one if that wins at 32-128 rows after 110k history. The
// 48 KB tile alone would allow two CTAs; retry a 128-register bound when ptxas keeps it at STACK:0.
__global__ void __launch_bounds__(256, 1) attn_prefill_hd256_head_k(attn_args a, int64_t rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace hd256;
  using prefill::hi_lo, prefill::ldmatrix, prefill::mma, prefill::pack, prefill::scales, prefill::scan_nonfinite;
  __shared__ __align__(128) uint8_t tiles[HEAD_SMEM];
  const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int rw = warp % 2, dq = warp / 2;  // row group, dim quarter
  // attend keeps rows * heads under 2^31, so heads, groups and rows take 32-bit arithmetic.
  const int h = int(blockIdx.x % unsigned(a.heads)), kvh = h / int(a.group);
  const int row0 = int(blockIdx.x / unsigned(a.heads)) * HEAD_ROWS, end = min(int(rows), row0 + HEAD_ROWS);
  const uint32_t base = uint32_t(__cvta_generic_to_shared(tiles));
  // Q is chunk-major, as in prefill.cuh: chunk c of row r at (c * HEAD_ROWS + r) * 16.
  if (((uintptr_t(a.q) | uintptr_t(a.q_ld) * 2) & 15) == 0) {
    for (int i = tid; i < HEAD_ROWS * HD / 8; i += 256) {
      const int r = row0 + i % HEAD_ROWS;
      const int64_t at = r < end ? int64_t(r) * a.q_ld + h * HD + i / HEAD_ROWS * 8 : 0;
      __pipeline_memcpy_async(tiles + i * 16, a.q + at, 16, r < end ? 0 : 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  } else {
    // ponytail: a q not 16-byte aligned keeps 2-byte element loads, about 32 serial load latencies a thread (unmeasured
    // here; prefill.cuh's marker has the HD128 cost); serving's q is aligned. Stage it in wider pieces if a strided q
    // shows in a profile.
    for (int i = tid; i < HEAD_ROWS * HD; i += 256) {
      const int r = i / HD, d = i % HD;
      *reinterpret_cast<__half*>(tiles + (d / 8 * HEAD_ROWS + r) * 16 + d % 8 * 2) =
          row0 + r < end ? a.q[int64_t(row0 + r) * a.q_ld + h * HD + d] : __float2half_rn(0);
    }
  }
  __syncthreads();
  uint32_t q[4][4];  // this warp's 16 rows x 64 dims
#pragma unroll
  for (int k = 0; k < 4; ++k)
    ldmatrix<false>(q[k], base + ((dq * 8 + 2 * k + lane / 16) * HEAD_ROWS + rw * 16 + lane % 16) * 16);
  // Rows g (x = 0) and g + 8 (x = 1) of this warp; o[n] holds their dims 64 dq + 8n + 2t, +1.
  float o[8][4] = {}, m[2] = {-INFINITY, -INFINITY}, l[2] = {};
  // The Q tile is dead once every warp holds its fragments (past the first sequence's barrier): 1 KB of it a warp
  // takes the warp's quarter of S (lane-major, conflict-free), or its FP32 P on a non-finite tile.
  float* const quarters = reinterpret_cast<float*>(tiles) + rw * 256 + lane;  // + 512 d: dim quarter d's
  float* const fp32_p = reinterpret_cast<float*>(tiles) + warp * 256;
  // This thread copies position (tid / 2) % 16, chunks c0 and c0 + 16, of every K and V tile: a lane pair reads one
  // 32-byte sector, as in prefill.cuh.
  const int copy_pos = tid / 2 % KT, c0 = tid % 2 + tid / 32 * 2;
  const uint32_t copy_at = (c0 * KT + copy_pos) * 16;
  const uint32_t k_lane = (lane / 8 * KT + lane % 8) * 16 + dq * 8 * KT * 16;
  const uint32_t v_lane = (lane / 16 * KT + lane % 16) * 16 + dq * 8 * KT * 16;
  int64_t seq = 0;
  int first = 0;
  while (seq + 1 < a.seqs && first + int64_t(a.lens[seq * a.lens_ld]) <= row0) first += a.lens[seq++ * a.lens_ld];
  for (;; first += a.lens[seq++ * a.lens_ld]) {
    const int to = seq + 1 < a.seqs ? int(min(int64_t(end), first + int64_t(a.lens[seq * a.lens_ld]))) : end;
    // A row group's four warps share mine, row_lo, row_hi and every skip below, so all reach each group barrier.
    const int mine = row0 + rw * 16, row_lo = min(16, max(0, first - mine)), row_hi = min(16, max(0, to - mine));
    const int64_t w = int64_t(a.start[seq * a.start_ld]) - first + 1 + mine, longest = w - mine + to - 1;
    if (max(first, row0) < to) {
      const int32_t* blocks = a.table + seq * a.table_ld;
      const auto stage = [&](int64_t p0, int buffer) {
        const int64_t pos = p0 + copy_pos;
        const bool real = pos < longest;
        const int64_t e = real ? ((int64_t(blocks[pos / 16]) * 16 + pos % 16) * a.kv_heads + kvh) * HD + c0 * 8 : 0;
        uint8_t* const k = tiles + HEAD_Q_BYTES + buffer * 2 * KV_BYTES + copy_at;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          __pipeline_memcpy_async(k + i * 16 * KT * 16, a.k + e + i * 128, 16, real ? 0 : 16);
          __pipeline_memcpy_async(k + KV_BYTES + i * 16 * KT * 16, a.v + e + i * 128, 16, real ? 0 : 16);
        }
        __pipeline_commit();
      };
      __syncthreads();  // the previous sequence's last tile is consumed before buffer 0 refills
      stage(0, 0);
      for (int64_t p0 = 0; p0 < longest; p0 += KT) {
        const int buffer = int(p0 / KT % 2);
        const uint32_t k_at = HEAD_Q_BYTES + buffer * 2 * KV_BYTES, v_at = k_at + KV_BYTES;
        __pipeline_wait_prior(0);
        asm volatile("" ::: "memory");  // keep the V scan below the wait (no "memory" clobber in its asm)
        bool bad = false;
#pragma unroll
        for (int i = 0; i < 2; ++i)
          scan_nonfinite(*reinterpret_cast<const uint4*>(tiles + v_at + copy_at + i * 16 * KT * 16), bad);
        // One barrier per tile: copies landed, the other buffer and every S quarter consumed, the flag shared.
        const bool nonfinite = __syncthreads_or(bad);
        if (p0 + KT < longest) stage(p0 + KT, buffer ^ 1);
        const int rel = int(min(int64_t(KT), max(int64_t(-16), w - p0)));
        if (row_lo >= row_hi || rel + row_hi <= 1) continue;  // causal skip: no row of this row group sees the tile
        const int count = int(min(int64_t(KT), longest - p0));
        float s[2][4] = {};
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            uint32_t kf[4];
            ldmatrix<false>(kf, base + k_at + k_lane + (k * 4 * KT + j * 8) * 16);
            mma(s[j], q[2 * k], kf[0], kf[1]);
            mma(s[j], q[2 * k + 1], kf[2], kf[3]);
          }
#pragma unroll
        for (int i = 0; i < 8; ++i) quarters[dq * 512 + i * 32] = s[i / 4][i % 4];
        // This row group's four warps; immediate ids, since a register id makes ptxas reserve all 16 barriers.
        if (rw) asm volatile("bar.sync 2, 128;" ::: "memory");
        else asm volatile("bar.sync 1, 128;" ::: "memory");
#pragma unroll
        for (int i = 0; i < 8; ++i)
          s[i / 4][i % 4] = (quarters[i * 32] + quarters[512 + i * 32]) + (quarters[1024 + i * 32] + quarters[1536 + i * 32]);
        int allowed[2];
        float down[2];
        uint32_t ph[4], pl[4];
#pragma unroll
        for (int x = 0; x < 2; ++x) {
          const int i = g + 8 * x;
          allowed[x] = i >= row_lo && i < row_hi ? max(0, min(count, rel + i)) : 0;
          float top = -INFINITY;
#pragma unroll
          for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e)
              if (8 * j + 2 * t + e < allowed[x]) top = fmaxf(top, s[j][2 * x + e] * a.scale);
          top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
          top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
          const float next = allowed[x] ? fmaxf(m[x], top) : m[x], rescale = expf(m[x] - next);
          const float corr = allowed[x] ? (l[x] == 0 ? 0.0f : rescale) : 1.0f;
          float largest = 0;
#pragma unroll
          for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              float& p = s[j][2 * x + e];
              const float score = __fmul_rn(p, a.scale), weight = expf(__fsub_rn(score, next));
              p = 8 * j + 2 * t + e < allowed[x] && score != -INFINITY ? weight : 0.0f;
              largest = fmaxf(largest, p);
            }
          largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, 1));
          largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, 2));
          float up;
          scales(largest, up, down[x]);
          float total = 0;
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            __half high[2], low[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              total += s[j][2 * x + e];
              hi_lo(s[j][2 * x + e], up, high[e], low[e]);
            }
            ph[j * 2 + x] = pack(high[0], high[1]);
            pl[j * 2 + x] = pack(low[0], low[1]);
          }
          total += __shfl_xor_sync(0xffffffffu, total, 1);
          total += __shfl_xor_sync(0xffffffffu, total, 2);
          l[x] = l[x] * corr + total;
          m[x] = next;
#pragma unroll
          for (int n = 0; n < 8; ++n) o[n][2 * x] *= corr, o[n][2 * x + 1] *= corr;
        }
        // As in prefill.cuh: masked P is exactly 0 in both panels and zero-filled V is 0; only non-finite V takes FP32.
        if (!nonfinite) {
#pragma unroll
          for (int n = 0; n < 8; n += 2) {
            uint32_t vf[4];
            ldmatrix<true>(vf, base + v_at + v_lane + n * KT * 16);
#pragma unroll
            for (int i = 0; i < 2; ++i) {
              float high[4] = {}, low[4] = {};
              mma(high, ph, vf[2 * i], vf[2 * i + 1]);
              mma(low, pl, vf[2 * i], vf[2 * i + 1]);
#pragma unroll
              for (int c = 0; c < 4; ++c) o[n + i][c] += (high[c] + low[c] * 0x1p-11f) * down[c / 2];
            }
          }
        } else {
          // The quarters are read before this warp's P overwrites its 1 KB (16 x 16 floats; this path is rare).
          if (rw) asm volatile("bar.sync 2, 128;" ::: "memory");
          else asm volatile("bar.sync 1, 128;" ::: "memory");
#pragma unroll
          for (int x = 0; x < 2; ++x)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
              for (int e = 0; e < 2; ++e) fp32_p[(g + 8 * x) * KT + 8 * j + 2 * t + e] = s[j][2 * x + e];
          __syncwarp();
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            const float* weight = fp32_p + (g + 8 * x) * KT;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
              float2 add{};
              for (int p = 0; p < allowed[x]; ++p) {
                const uint8_t* at = tiles + v_at + ((dq * 8 + c) * KT + p) * 16 + 4 * t;
                const float2 v = __half22float2(*reinterpret_cast<const __half2*>(at));
                add.x += weight[p] * v.x;
                add.y += weight[p] * v.y;
              }
              o[c][2 * x] += add.x;
              o[c][2 * x + 1] += add.y;
            }
          }
        }
      }
    }
    if (to >= end) break;
  }
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const int r = row0 + rw * 16 + g + 8 * x;
    if (r >= end) continue;
    __half* out = a.out + int64_t(r) * a.out_ld + h * HD + dq * 64 + 2 * t;
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) out[8 * n + e] = __float2half_rn(o[n][2 * x + e] / l[x]);
  }
#elif defined(__CUDA_ARCH__)
  asm volatile("trap;");  // The host routes here only from sm_80; a build for an older arch must not run it.
#endif
}

// prefill.cuh's FA2 prefill at HD256 with the GQA group packed into the tile (llama.cpp's ncols2, FlashInfer's
// head-group fusion): a CTA is one kv head x R = 64 / group query rows x the group's q heads, M = 64 virtual rows,
// v = r * group + j for row r and head j of the group, so each K/V tile serves all the heads that share it. Four warps
// along M, each 16 virtual rows with all 256 output dims in registers (o: 128); Q stays in shared memory and each tile
// reloads its fragments, so no warp exchanges S and a warp skips a tile none of its rows sees. One K/V stage: Q (32 KB)
// and the stage fill the 48 KB static limit, and two CTAs an SM overlap one's copies with the other's math. Registers:
// 255 and STACK:0 needs the row and head packed in one register, the QK loop unrolled by 2, and the panels' PV and the
// FP32 path apart (S live through PV spilled).
// ponytail: single-stage K/V and 16-position tiles; double-buffer through dynamic shared memory (one CTA an SM) or
// widen the tiles if a profile shows the copies or the barriers bound once this ships.
// TM, the tile's virtual rows, is M or M / 2: the half tile (two warps) doubles a grid that would leave SMs idle (256
// rows of 16q/2kv: 64 CTAs of 64 on the 5070 Ti's 70 SMs lost 3 % to attn_prefill_hd256_head_k).
template <int TM>
__global__ void __launch_bounds__(TM * 2, 128 / TM) attn_prefill_hd256_k(attn_args a, int64_t rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace hd256;
  using prefill::hi_lo, prefill::ldmatrix, prefill::mma, prefill::pack, prefill::scales, prefill::scan_nonfinite;
  constexpr int THREADS = TM * 2, WARPS = TM / 16, Q_BYTES = TM * HD * 2;
  __shared__ __align__(128) uint8_t tiles[Q_BYTES + 2 * KV_BYTES];
  const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  // attend keeps rows * heads under 2^31 and group <= TM, so rows, heads and groups take 32-bit arithmetic.
  const int group = int(a.group), span = TM / group;  // span: the query rows of a CTA
  const int kvh = int(blockIdx.x % unsigned(a.kv_heads));
  const int row0 = int(blockIdx.x / unsigned(a.kv_heads)) * span, end = min(int(rows), row0 + span);
  const uint32_t base = uint32_t(__cvta_generic_to_shared(tiles));
  // Virtual row v is query row row0 + v / group, head kvh * group + v % group; v >= span * group is padding.
  const auto q_at = [=](int v) {
    return int64_t(row0 + v / group) * a.q_ld + int64_t(kvh * group + v % group) * HD;
  };
  const auto real = [=](int v) { return v < span * group && row0 + v / group < end; };
  // Q is chunk-major: chunk c of virtual row v at (c * TM + v) * 16.
  if (((uintptr_t(a.q) | uintptr_t(a.q_ld) * 2) & 15) == 0) {
    for (int i = tid; i < TM * HD / 8; i += THREADS) {
      const int v = i % TM;
      __pipeline_memcpy_async(tiles + i * 16, a.q + (real(v) ? q_at(v) + i / TM * 8 : 0), 16, real(v) ? 0 : 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  } else {
    // ponytail: a q not 16-byte aligned keeps 2-byte element loads (as prefill.cuh's marker says); serving's q is
    // aligned. Stage it in wider pieces if a strided q shows in a profile.
    for (int i = tid; i < TM * HD; i += THREADS) {
      const int v = i / HD, d = i % HD;
      *reinterpret_cast<__half*>(tiles + (d / 8 * TM + v) * 16 + d % 8 * 2) =
          real(v) ? a.q[q_at(v) + d] : __float2half_rn(0);
    }
  }
  // Rows g (x = 0) and g + 8 (x = 1) of this warp's 16; o[n] holds their dims 8n + 2t, +1.
  float o[32][4] = {}, m[2] = {-INFINITY, -INFINITY}, l[2] = {};
  // Each one's query row and head in the group as row * M + head (attend keeps rows * M under 2^31), -1 for padding:
  // one register a row, which the output's head reads too (a separate head was kept and spilled).
  int vrow[2];
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const int v = warp * 16 + g + 8 * x;
    vrow[x] = real(v) ? (row0 + v / group) * M + v % group : -1;
  }
  const uint32_t q_lane = (lane / 16 * TM + warp * 16 + lane % 16) * 16;
  // This thread copies position (tid / 2) % 16, chunks c0 + 2 WARPS i, of every K and V tile: a lane pair reads one
  // 32-byte sector, as in prefill.cuh.
  const int copy_pos = tid / 2 % KT, c0 = tid % 2 + warp * 2;
  const uint32_t copy_at = (c0 * KT + copy_pos) * 16, k_at = Q_BYTES, v_at = Q_BYTES + KV_BYTES;
  // K: lanes 8i.. take chunk i % 2 of positions 8 (i / 2).., so one ldmatrix holds both 8-position B fragments.
  const uint32_t k_lane = (lane / 8 % 2 * KT + lane / 16 * 8 + lane % 8) * 16;
  const uint32_t v_lane = (lane / 16 * KT + lane % 16) * 16;
  // A non-finite tile's FP32 P (16 x 16 a warp) goes where its K was, once every warp is done with K.
  float* const fp32_p = reinterpret_cast<float*>(tiles + k_at) + warp * 16 * KT;
  int64_t seq = 0;
  int first = 0;
  while (seq + 1 < a.seqs && first + int64_t(a.lens[seq * a.lens_ld]) <= row0) first += a.lens[seq++ * a.lens_ld];
  for (;; first += a.lens[seq++ * a.lens_ld]) {
    const int to = seq + 1 < a.seqs ? int(min(int64_t(end), first + int64_t(a.lens[seq * a.lens_ld]))) : end;
    // Row r of the sequence sees positions below start - first + 1 + r; the CTA's last row the most.
    const int64_t shift = int64_t(a.start[seq * a.start_ld]) - first + 1, longest = shift + to - 1;
    if (max(first, row0) < to) {
      const int32_t* blocks = a.table + seq * a.table_ld;
      for (int64_t p0 = 0; p0 < longest; p0 += KT) {
        __syncthreads();  // every warp is done with the stage (and, at a sequence's start, Q has landed)
        {
          const int64_t pos = p0 + copy_pos;
          const bool live = pos < longest;
          const int64_t e = live ? ((int64_t(blocks[pos / 16]) * 16 + pos % 16) * a.kv_heads + kvh) * HD + c0 * 8 : 0;
#pragma unroll
          for (int i = 0; i < 16 / WARPS; ++i) {
            const int at = i * 2 * WARPS * KT * 16, from = i * 2 * WARPS * 8;
            __pipeline_memcpy_async(tiles + k_at + copy_at + at, a.k + e + from, 16, live ? 0 : 16);
            __pipeline_memcpy_async(tiles + v_at + copy_at + at, a.v + e + from, 16, live ? 0 : 16);
          }
          __pipeline_commit();
        }
        __pipeline_wait_prior(0);
        asm volatile("" ::: "memory");  // keep the V scan below the wait (no "memory" clobber in its asm)
        bool bad = false;
#pragma unroll
        for (int i = 0; i < 16 / WARPS; ++i)
          scan_nonfinite(*reinterpret_cast<const uint4*>(tiles + v_at + copy_at + i * 2 * WARPS * KT * 16), bad);
        // Copies landed, the flag shared.
        const bool nonfinite = __syncthreads_or(bad);
        const int count = int(min(int64_t(KT), longest - p0));
        int allowed[2];
#pragma unroll
        for (int x = 0; x < 2; ++x) {
          const int row = vrow[x] / M;  // -1 / M is 0, but -1 fails the first test
          const bool in = vrow[x] >= 0 && row >= max(first, row0) && row < to;
          allowed[x] = in ? int(max(int64_t(0), min(int64_t(count), shift + row - p0))) : 0;
        }
        // Causal skip, per warp: no row of this warp sees the tile.
        const bool skip = !__any_sync(0xffffffffu, allowed[0] > 0 || allowed[1] > 0);
        float s[2][4] = {}, down[2];
        uint32_t ph[4], pl[4];
        if (!skip) {
#if __CUDA_ARCH__ >= 1200
#pragma unroll 2
#else
#pragma unroll
#endif
          for (int k = 0; k < 16; ++k) {  // 16 dims a step: one A fragment, one K ldmatrix for both 8 positions
            uint32_t qa[4], kf[4];
            ldmatrix<false>(qa, base + q_lane + 2 * k * TM * 16);
            ldmatrix<false>(kf, base + k_at + k_lane + 2 * k * KT * 16);
            mma(s[0], qa, kf[0], kf[1]);
            mma(s[1], qa, kf[2], kf[3]);
          }
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            float top = -INFINITY;
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
              for (int e = 0; e < 2; ++e)
                if (8 * j + 2 * t + e < allowed[x]) top = fmaxf(top, s[j][2 * x + e] * a.scale);
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
            const float next = allowed[x] ? fmaxf(m[x], top) : m[x], rescale = expf(m[x] - next);
            const float corr = allowed[x] ? (l[x] == 0 ? 0.0f : rescale) : 1.0f;
            float largest = 0;
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
              for (int e = 0; e < 2; ++e) {
                float& p = s[j][2 * x + e];
                const float score = __fmul_rn(p, a.scale), weight = expf(__fsub_rn(score, next));
                p = 8 * j + 2 * t + e < allowed[x] && score != -INFINITY ? weight : 0.0f;
                largest = fmaxf(largest, p);
              }
            largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, 1));
            largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, 2));
            float up;
            scales(largest, up, down[x]);
            float total = 0;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              __half high[2], low[2];
#pragma unroll
              for (int e = 0; e < 2; ++e) {
                total += s[j][2 * x + e];
                hi_lo(s[j][2 * x + e], up, high[e], low[e]);
              }
              ph[j * 2 + x] = pack(high[0], high[1]);
              pl[j * 2 + x] = pack(low[0], low[1]);
            }
            total += __shfl_xor_sync(0xffffffffu, total, 1);
            total += __shfl_xor_sync(0xffffffffu, total, 2);
            l[x] = l[x] * corr + total;
            m[x] = next;
#pragma unroll
            for (int n = 0; n < 32; ++n) o[n][2 * x] *= corr, o[n][2 * x + 1] *= corr;
          }
        }
        // As in prefill.cuh: masked P is exactly 0 in both panels and zero-filled V is 0; only non-finite V takes FP32.
        // A non-finite V would make a masked 0 x Inf = NaN in the panels: such a tile takes FP32 over its positions.
        // The flag is the CTA's, so every warp, skipping or not, reaches the barrier that frees K for the FP32 P.
        // Apart, so S is dead in the panels' branch and the panels in this one.
        if (!nonfinite) {
          if (!skip) {
#pragma unroll
            for (int n = 0; n < 32; n += 2) {
#if __CUDA_ARCH__ < 1200
              // A fence every two steps keeps ptxas's V loads at most two steps ahead: five spilled a register a tile
              // on sm_86/89.
              if (n % 4 == 0) __syncwarp();
#endif
              uint32_t vf[4];
              ldmatrix<true>(vf, base + v_at + v_lane + n * KT * 16);
#pragma unroll
              for (int i = 0; i < 2; ++i) {
                float high[4] = {}, low[4] = {};
                mma(high, ph, vf[2 * i], vf[2 * i + 1]);
                mma(low, pl, vf[2 * i], vf[2 * i + 1]);
#pragma unroll
                for (int c = 0; c < 4; ++c) o[n + i][c] += (high[c] + low[c] * 0x1p-11f) * down[c / 2];
              }
            }
          }
        } else {
          __syncthreads();
          if (!skip) {
#pragma unroll
            for (int x = 0; x < 2; ++x)
#pragma unroll
              for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) fp32_p[(g + 8 * x) * KT + 8 * j + 2 * t + e] = s[j][2 * x + e];
            __syncwarp();
#pragma unroll
            for (int x = 0; x < 2; ++x) {
              const float* weight = fp32_p + (g + 8 * x) * KT;
#pragma unroll
              for (int c = 0; c < 32; ++c) {
                float2 add{};
                for (int p = 0; p < allowed[x]; ++p) {
                  const uint8_t* at = tiles + v_at + (c * KT + p) * 16 + 4 * t;
                  const float2 v = __half22float2(*reinterpret_cast<const __half2*>(at));
                  add.x += weight[p] * v.x;
                  add.y += weight[p] * v.y;
                }
                o[c][2 * x] += add.x;
                o[c][2 * x + 1] += add.y;
              }
            }
          }
        }
      }
    }
    if (to >= end) break;
  }
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    if (vrow[x] < 0) continue;
    __half* out = a.out + int64_t(vrow[x] / M) * a.out_ld + int64_t(kvh * group + vrow[x] % M) * HD + 2 * t;
#pragma unroll
    for (int n = 0; n < 32; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) out[8 * n + e] = __float2half_rn(o[n][2 * x + e] / l[x]);
  }
#elif defined(__CUDA_ARCH__)
  asm volatile("trap;");  // The host routes here only from sm_80; a build for an older arch must not run it.
#endif
}
