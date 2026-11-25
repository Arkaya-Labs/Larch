// Included by ops.cu inside namespace cuda after attn_args, on CUDA only: the kernel is sm_80 inline PTX.
#pragma once

// HD128/F16 KV, page16, concatenated ragged prefill, FA2-style. A CTA is one query head x 64 rows; each of its four
// warps owns 16 rows and keeps Q, O, P and the softmax state in registers. K/V pages (16 positions) are staged by
// cp.async, double-buffered, so the next page loads while this one computes. No split-K or persistent allocation.
// ponytail: 16-position tiles; at 32 a warp needs more than 255 registers and spills (sm_120, CUDA 13.3). Retry 32
// when a KT = 32 build shows STACK:0 for attn_prefill_tc_k in cuobjdump --dump-resource-usage.
namespace prefill {
constexpr int ROWS = 64, KT = 16, HD = 128, LD = KT + 1;  // LD pads the FP32 P rows off one bank
constexpr int Q_BYTES = ROWS * HD * 2, KV_BYTES = KT * HD * 2;
constexpr int SMEM = Q_BYTES + 4 * KV_BYTES;  // Q (then each warp's FP32 P), and two stages of K and V tiles

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Volatile keeps the shared reads after the barrier that publishes the tile.
template <bool TRANS>
__device__ __forceinline__ void ldmatrix(uint32_t (&x)[4], uint32_t address) {
  if constexpr (TRANS)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]) : "r"(address));
  else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3]) : "r"(address));
}

// c += a b on one m16n8k16 tile: f16 A (row-major) and B (column-major) fragments, FP32 accumulators.
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two halves as one fragment register, lo in the low 16 bits.
__device__ __forceinline__ uint32_t pack(__half lo, __half hi) {
  return uint32_t(__half_as_ushort(lo)) | uint32_t(__half_as_ushort(hi)) << 16;
}

// Sets bad when any of the eight halves of v is Inf or NaN (all exponent bits set).
__device__ __forceinline__ void scan_nonfinite(uint4 v, bool& bad) {
  const uint32_t half2s[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int j = 0; j < 4; ++j)
    bad |= (half2s[j] & 0x7c00u) == 0x7c00u || (half2s[j] & 0x7c000000u) == 0x7c000000u;
}

// up = 2^shift puts a tile's largest P, L, in [2^14, 2^15) (lower when the clamp binds: L < 2^-112 or 0), so P far
// below the running max (most of a long context) keeps the panels' 22 bits; down = 2^-shift undoes it when the panels
// meet O. Powers of two scale exactly both ways; shift <= 126 keeps 2^-shift a normal float.
__device__ __forceinline__ void scales(float largest, float& up, float& down) {
  const int shift = min(126, max(0, 14 - (int(__float_as_uint(largest) >> 23) - 127)));
  up = __uint_as_float(uint32_t(127 + shift) << 23);
  down = __uint_as_float(uint32_t(127 - shift) << 23);
}

// p times up as the two panels' halves: high = half(p up) and low = half((p up - high) * 2^11). The residual is at
// most half an ulp of high: 2^11 lifts it to high's scale so its half keeps 11 bits (PV unscales it in FP32). Where
// that half is subnormal (residual under 2^-25 at this scale) its rounding drops under 2^-36 * 2^-shift: at most 2^-50
// L (2^-162 when the clamp binds), so 2^-46 L max|V| a tile. The tiles' L sum to at most l, so the output moves under
// 2^-46 max|V| <= 2^-30, below one f16 output ulp: no FP32 PV is owed for it.
__device__ __forceinline__ void hi_lo(float p, float up, __half& high, __half& low) {
  const float scaled = p * up;
  high = __float2half_rn(scaled);
  low = __float2half_rn((scaled - __half2float(high)) * 2048.0f);
}
#endif
}  // namespace prefill

// Shared tiles are chunk-major: the 16-byte chunk c (eight dims) of row r sits at (c * rows + r) * 16. An ldmatrix
// phase's eight rows are then 128 contiguous bytes (no bank conflict) and every fragment is a fixed offset from one
// per-lane address. On sm_120 (CUDA 13.3) the 128-thread bound gets 192 registers, STACK:0 (two CTAs an SM);
// unbounded ptxas takes 209, and a three-CTA bound spills (8 bytes of stack). The sm_86/sm_89 builds take 188,
// STACK:0 (see PREFILL_TC_DEFAULT_ARCHES in common/arch.h).
__global__ void __launch_bounds__(128) attn_prefill_tc_k(attn_args a, int64_t rows) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using namespace prefill;
  __shared__ __align__(128) uint8_t tiles[SMEM];
  const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  // attend() keeps rows * heads under 2^31, so heads, groups and rows take 32-bit arithmetic.
  const int h = int(blockIdx.x % unsigned(a.heads)), kvh = h / int(a.group);
  const int row0 = int(blockIdx.x / unsigned(a.heads)) * ROWS, end = min(int(rows), row0 + ROWS);
  const uint32_t base = uint32_t(__cvta_generic_to_shared(tiles));
  if (((uintptr_t(a.q) | uintptr_t(a.q_ld) * 2) & 15) == 0) {
    // cp.async keeps every copy in flight (the generic element loads below run one at a time: each may alias the
    // shared store before it). Chunk i is row i % ROWS, dims i / ROWS * 8.., so it lands at i * 16: a warp stores 512
    // contiguous bytes, no bank conflict. Rows past end are zero-filled.
    for (int i = tid; i < ROWS * HD / 8; i += 128) {
      const int r = row0 + i % ROWS;
      __pipeline_memcpy_async(tiles + i * 16, a.q + (r < end ? int64_t(r) * a.q_ld + h * HD + i / ROWS * 8 : 0), 16,
                              r < end ? 0 : 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
  } else {
    // ponytail: a q not 16-byte aligned (its rows need not even be 4-byte aligned) keeps 2-byte element loads, about
    // 64 serial load latencies a thread (7.8 us of 64 rows at 0 history, sm_120); serving's q is aligned. Stage it in
    // wider pieces if a strided q shows in a profile.
    for (int i = tid; i < ROWS * HD; i += 128) {
      const int r = i / HD, d = i % HD;
      *reinterpret_cast<__half*>(tiles + (d / 8 * ROWS + r) * 16 + d % 8 * 2) =
          row0 + r < end ? a.q[int64_t(row0 + r) * a.q_ld + h * HD + d] : __float2half_rn(0);
    }
  }
  __syncthreads();
  uint32_t q[8][4];
#pragma unroll
  for (int k = 0; k < 8; ++k)
    ldmatrix<false>(q[k], base + (lane / 16 * ROWS + warp * 16 + lane % 16) * 16 + k * 2 * ROWS * 16);
  // Rows g (x = 0) and g + 8 (x = 1) of this warp; o[n] holds their columns 8n + 2t and 8n + 2t + 1.
  float o[16][4] = {}, m[2] = {-INFINITY, -INFINITY}, l[2] = {};
  // The Q tile is dead once every warp holds its fragments (past the first sequence's barrier).
  float* const fp32_p = reinterpret_cast<float*>(tiles + warp * 16 * HD * 2);
  // This thread copies position (tid / 2) % 16, chunks c0 and c0 + 8, of every K and V tile: a lane pair reads one
  // 32-byte sector and a warp writes two 256-byte runs.
  const int copy_pos = tid / 2 % KT, c0 = tid % 2 + tid / 32 * 2;
  const uint32_t copy_at = (c0 * KT + copy_pos) * 16;
  const uint32_t k_lane = (lane / 8 * KT + lane % 8) * 16, v_lane = (lane / 16 * KT + lane % 16) * 16;
  int64_t seq = 0;
  int first = 0;
  while (seq + 1 < a.seqs && first + int64_t(a.lens[seq * a.lens_ld]) <= row0) first += a.lens[seq++ * a.lens_ld];
  // Monotone concatenation: each sequence is one contiguous run of the tile's rows, staging its own pages.
  for (;; first += a.lens[seq++ * a.lens_ld]) {
    const int to = seq + 1 < a.seqs ? int(min(int64_t(end), first + int64_t(a.lens[seq * a.lens_ld]))) : end;
    // A warp's 16 rows may straddle sequences: those outside [row_lo, row_hi) see none of this one's positions.
    const int mine = row0 + warp * 16, row_lo = min(16, max(0, first - mine)), row_hi = min(16, max(0, to - mine));
    // Row offset i of this warp attends to positions [0, w + i); the CTA stages [0, longest).
    const int64_t w = int64_t(a.start[seq * a.start_ld]) - first + 1 + mine, longest = w - mine + to - 1;
    if (max(first, row0) < to) {
      const int32_t* blocks = a.table + seq * a.table_ld;
      const auto stage = [&](int64_t p0, int buffer) {
        const int64_t pos = p0 + copy_pos;
        const bool real = pos < longest;
        const int64_t e = real ? ((int64_t(blocks[pos / 16]) * 16 + pos % 16) * a.kv_heads + kvh) * HD + c0 * 8 : 0;
        uint8_t* const k = tiles + Q_BYTES + buffer * 2 * KV_BYTES + copy_at;
        // 16 bytes global to shared by cp.async (no register round trip); a position past the sequence is zero-filled.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          __pipeline_memcpy_async(k + i * 8 * KT * 16, a.k + e + i * 64, 16, real ? 0 : 16);
          __pipeline_memcpy_async(k + KV_BYTES + i * 8 * KT * 16, a.v + e + i * 64, 16, real ? 0 : 16);
        }
        __pipeline_commit();
      };
      __syncthreads();  // the previous sequence's last tile is consumed before buffer 0 refills
      stage(0, 0);
      for (int64_t p0 = 0; p0 < longest; p0 += KT) {
        const int buffer = int(p0 / KT % 2);
        const uint32_t k_at = Q_BYTES + buffer * 2 * KV_BYTES, v_at = k_at + KV_BYTES;
        __pipeline_wait_prior(0);
        // cuda_pipeline_helpers.h's wait_group asm has no "memory" clobber: keep the V scan below it.
        asm volatile("" ::: "memory");
        bool bad = false;  // a non-finite V among this thread's own copies
#pragma unroll
        for (int i = 0; i < 2; ++i)
          scan_nonfinite(*reinterpret_cast<const uint4*>(tiles + v_at + copy_at + i * 8 * KT * 16), bad);
        // One barrier per tile: every copy has landed, every warp is done with the other buffer, the flag is shared.
        const bool nonfinite = __syncthreads_or(bad);
        if (p0 + KT < longest) stage(p0 + KT, buffer ^ 1);
        // Row offset i sees min(rel + i, count) positions of this tile; rel is clamped where that stops changing.
        const int rel = int(min(int64_t(KT), max(int64_t(-16), w - p0)));
        if (row_lo >= row_hi || rel + row_hi <= 1) continue;  // causal skip: no row of this warp sees the tile
        const int count = int(min(int64_t(KT), longest - p0));
        float s[2][4] = {};
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            uint32_t kf[4];
            ldmatrix<false>(kf, base + k_at + k_lane + (k * 4 * KT + j * 8) * 16);
            mma(s[j], q[2 * k], kf[0], kf[1]);
            mma(s[j], q[2 * k + 1], kf[2], kf[3]);
          }
        int allowed[2];
        float down[2];          // 2^-shift of each row: undoes the panels' scale when they meet O
        uint32_t ph[4], pl[4];  // P * 2^shift as A fragments: its half (high) and half((it - high) * 2^11)
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
          // A row with no allowed position here keeps its max and sum. The exponentials are computed unmasked and
          // then selected: branching around each one costs more than it.
          const float next = allowed[x] ? fmaxf(m[x], top) : m[x], rescale = expf(m[x] - next);
          const float corr = allowed[x] ? (l[x] == 0 ? 0.0f : rescale) : 1.0f;
          // The row's panels are split at its own scale (scales()).
          float largest = 0;
#pragma unroll
          for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              float& p = s[j][2 * x + e];
              const float score = __fmul_rn(p, a.scale), weight = expf(__fsub_rn(score, next));
              // -Inf weighs 0 also while next is -Inf (exp(-Inf + Inf) = NaN), as under the reference's global max.
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
          for (int n = 0; n < 16; ++n) o[n][2 * x] *= corr, o[n][2 * x + 1] *= corr;
        }
        // Tensor PV multiplies every staged position into every row: a masked P, and every P of a row outside the
        // sequence or seeing none of the tile, is exactly 0 in both panels (shift clamps to 126, so up and down stay
        // finite); zero-filled positions hold V = 0 and 0 * finite V = 0. A non-finite V would make 0 * Inf = NaN.
        if (!nonfinite) {
#pragma unroll
          for (int n = 0; n < 16; n += 2) {
            uint32_t vf[4];  // V fragments of dims 8n..8n+7, then 8n+8..8n+15
            ldmatrix<true>(vf, base + v_at + v_lane + n * KT * 16);
            float high[2][4] = {}, low[2][4] = {};
#pragma unroll
            for (int i = 0; i < 2; ++i) {
              mma(high[i], ph, vf[2 * i], vf[2 * i + 1]);
              mma(low[i], pl, vf[2 * i], vf[2 * i + 1]);
#pragma unroll
              for (int c = 0; c < 4; ++c) o[n + i][c] += (high[i][c] + low[i][c] * 0x1p-11f) * down[c / 2];
            }
          }
        } else {
          // Only positions a row sees meet a non-finite V (a masked one would add 0 x Inf), and FP32 needs no panels.
#pragma unroll
          for (int x = 0; x < 2; ++x)
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
              for (int e = 0; e < 2; ++e) fp32_p[(g + 8 * x) * LD + 8 * j + 2 * t + e] = s[j][2 * x + e];
          __syncwarp();
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            const float* weight = fp32_p + (g + 8 * x) * LD;
#pragma unroll
            for (int c = 0; c < 16; ++c) {
              // The tile's FP32 sum meets O once, as the tensor panels do: O is large, and rounding into it per
              // position loses the cancellations the reference keeps.
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
    if (to >= end) break;
  }
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const int r = row0 + warp * 16 + g + 8 * x;
    if (r >= end) continue;
    // Every row sees a position, so l is 0 only after -Inf scores alone: 0 / 0 is the reference's NaN.
    __half* out = a.out + int64_t(r) * a.out_ld + h * HD + 2 * t;
#pragma unroll
    for (int n = 0; n < 16; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) out[8 * n + e] = __float2half_rn(o[n][2 * x + e] / l[x]);
  }
#elif defined(__CUDA_ARCH__)
  asm volatile("trap;");  // The host routes here only from sm_80; a build for an older arch must not run it.
#endif
}

// The caller proves tensor/layout and device capability eligibility; all metadata stays device-side.
static void attend_prefill_tc(const stream& s, const attn_args& a, int64_t rows) {
  const int64_t blocks = (rows + prefill::ROWS - 1) / prefill::ROWS * a.heads;
  if (blocks > INT_MAX) throw failure{LARCH_ERR_UNSUPPORTED, "more than 2^31 prefill tiles"};
  launch(attn_prefill_tc_k, unsigned(blocks), 128, s.s, a, rows);
}
