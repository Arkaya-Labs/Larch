// The staged decode tile (design 2.1): mmvq at 1 to 32 rows on one kernel, so a row's bits do not depend on the rows
// beside it or on how many there are. Changed from Splash (https://github.com/incoai/splash at d8af1054, Apache-2.0,
// splash/LICENSE): the stage, prefetch, spread walk and last-part K reduce of runtime/metal/kernels/common/
// gguf_staged_tile.h, shared/gguf_linear.metal and common/split_reduce.h, here on GGUF blocks read in place with
// larch's decoders, f16 activations, a K split fixed per shape and one-simdgroup threadgroups (splash/VENDORED).
// Keep the source order of float operations, so a row's sum is the same at every row count. Before the includes.
#pragma clang fp reassociate(off)
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#include "args.h"

using namespace metal;
using namespace mpp::tensor_ops;

typedef device const uchar* bytes_in;

// Four bytes at a 2-byte aligned p: Q3_K, Q6_K and Q8_0 blocks are 110, 210 and 34 bytes, so only every other
// block of a row is 4-byte aligned.
// ponytail: two 2-byte loads a word. Aligned uint loads on even blocks if the Mac shows q3_k or q6_k load bound.
static inline uint u32_at(bytes_in p) {
  device const ushort* h = (device const ushort*)p;
  return uint(h[0]) | uint(h[1]) << 16;
}

// The formats. Lane l of a tile decodes item j = l % 4 of 4 columns of each unit: elements 32 u + 8 j .. + 7 of the
// unit for u = 0..3, so the 4 lanes of a column read 32 adjacent bytes. load() reads an item's quant bytes into
// registers (issued a matmul ahead of their use); stage() decodes group u of them into 8 halves at o, each the f16
// rounding of the f32 value ops.metal's deq_<fmt> computes, as vv_gemm_q's stage holds it (Q3_K and Q6_K: +0 for
// its -0, one fma a weight instead of a sub and a mul).

// Q3_K, unit = half n of a block: bytes 8 j .. + 7 of the half's qs and of hmask, the 12 scale bytes and d.
struct st_q3_k {
  struct item {
    uint q[2], h[2], s[3];
    half d;
  };
  static item load(bytes_in row, uint unit, uint j) {
    bytes_in b = row + unit / 2 * 110, q = b + 32 + 32 * (unit % 2) + 8 * j, hm = b + 8 * j;
    item v;
    v.q[0] = u32_at(q);
    v.q[1] = u32_at(q + 4);
    v.h[0] = u32_at(hm);
    v.h[1] = u32_at(hm + 4);
    v.s[0] = u32_at(b + 96);
    v.s[1] = u32_at(b + 100);
    v.s[2] = u32_at(b + 104);
    v.d = *(device const half*)(b + 108);
    return v;
  }
  // q = bit pair u of the qs byte, minus 4 where hmask bit 4 n + u is clear, times d (scale i - 32), i = 8 n + 2 u +
  // j / 2. Four codes a word: code = q + 4, and fma(code, dl, -4 dl) rounds (code - 4) dl once, as dl * q does.
  static void stage(thread const item& v, uint unit, uint j, uint u, threadgroup half4* o) {
    const uint n = unit % 2, i = 8 * n + 2 * u + j / 2, b = i % 8;
    const uint lo = (b < 4 ? v.s[0] : v.s[1]) >> (8 * (b % 4) + 4 * n) & 0xF;
    const uint hi = v.s[2] >> (8 * (i % 4) + 2 * (i / 4)) & 3;
    const float dl = float(v.d) * float(int(lo | hi << 4) - 32);
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      const uint c = (v.q[w] >> (2 * u) & 0x03030303u) | (v.h[w] >> (4 * n + u) & 0x01010101u) << 2;
      o[w] = half4(fma(float4(as_type<uchar4>(c)), float4(dl), float4(-4.0f * dl)));
    }
  }
};

// The 6-bit scale and min of sub-block c of a Q4_K/Q5_K block whose first 16 bytes are hdr (ggml's get_scale_min_k4).
static inline uint2 k4_scale_min(uint4 hdr, uint c) {
  if (c < 4) return uint2(hdr.y >> (8 * c) & 63, hdr.z >> (8 * c) & 63);
  const uint sh = 8 * (c - 4), w = hdr.w >> sh;
  return uint2((w & 0xF) | (hdr.y >> sh >> 6 & 3) << 4, (w >> 4 & 0xF) | (hdr.z >> sh >> 6 & 3) << 4);
}

// Q4_K, unit = half h of a block (sub-blocks 4 h .. 4 h + 3): the 16-byte header, and qs bytes 64 h + 8 j .. + 7
// (sub-blocks 4 h, 4 h + 1 in their low and high nibbles) and the 8 bytes 32 on (4 h + 2, 4 h + 3).
// Wide loads: the host sends only 16-byte aligned rows (GGUF aligns tensors to 32 bytes; a block is 144).
struct st_q4_k {
  struct item {
    uint4 hdr;
    uint2 q[2];
  };
  static item load(bytes_in row, uint unit, uint j) {
    bytes_in b = row + unit / 2 * 144, q = b + 16 + 64 * (unit % 2) + 8 * j;
    item v;
    v.hdr = *(device const uint4*)b;
    v.q[0] = *(device const uint2*)q;
    v.q[1] = *(device const uint2*)(q + 32);
    return v;
  }
  static void stage(thread const item& v, uint unit, uint, uint u, threadgroup half4* o) {
    const uint2 sm = k4_scale_min(v.hdr, 4 * (unit % 2) + u);
    const half2 dm = as_type<half2>(v.hdr.x);
    const float d = float(dm.x) * float(sm.x), mn = float(dm.y) * float(sm.y);
#pragma unroll
    for (int w = 0; w < 2; ++w)
      o[w] = half4(d * float4(as_type<uchar4>(v.q[u / 2][w] >> (4 * (u % 2)) & 0x0F0F0F0Fu)) - mn);
  }
};

// Q5_K: Q4_K's bytes (qs from 48) and qh bytes 8 j .. + 7, whose bit c is sub-block c's fifth bit. 16-byte rows
// (a block is 176).
struct st_q5_k {
  struct item {
    uint4 hdr;
    uint2 h, q[2];
  };
  static item load(bytes_in row, uint unit, uint j) {
    bytes_in b = row + unit / 2 * 176, q = b + 48 + 64 * (unit % 2) + 8 * j;
    item v;
    v.hdr = *(device const uint4*)b;
    v.h = *(device const uint2*)(b + 16 + 8 * j);
    v.q[0] = *(device const uint2*)q;
    v.q[1] = *(device const uint2*)(q + 32);
    return v;
  }
  static void stage(thread const item& v, uint unit, uint, uint u, threadgroup half4* o) {
    const uint c = 4 * (unit % 2) + u;
    const uint2 sm = k4_scale_min(v.hdr, c);
    const half2 dm = as_type<half2>(v.hdr.x);
    const float d = float(dm.x) * float(sm.x), mn = float(dm.y) * float(sm.y);
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      const uint q = (v.q[u / 2][w] >> (4 * (u % 2)) & 0x0F0F0F0Fu) | (v.h[w] >> c & 0x01010101u) << 4;
      o[w] = half4(d * float4(as_type<uchar4>(q)) - mn);
    }
  }
};

// Q6_K, unit = half n of a block: ql bytes 64 n + 8 j .. + 7 (quarters 0 and 2 in their low and high nibbles) and 32
// on (1 and 3), qh bytes 128 + 32 n + 8 j .. + 7 (bit pair u: quarter u's top bits), the half's 8 scales and d.
struct st_q6_k {
  struct item {
    uint l[2][2], h[2], s[2];
    half d;
  };
  static item load(bytes_in row, uint unit, uint j) {
    const uint n = unit % 2;
    bytes_in b = row + unit / 2 * 210, ql = b + 64 * n + 8 * j, qh = b + 128 + 32 * n + 8 * j;
    item v;
    v.l[0][0] = u32_at(ql);
    v.l[0][1] = u32_at(ql + 4);
    v.l[1][0] = u32_at(ql + 32);
    v.l[1][1] = u32_at(ql + 36);
    v.h[0] = u32_at(qh);
    v.h[1] = u32_at(qh + 4);
    v.s[0] = u32_at(b + 192 + 8 * n);
    v.s[1] = u32_at(b + 196 + 8 * n);
    v.d = *(device const half*)(b + 208);
    return v;
  }
  // deq_q6_k's (d sc) q with q = code - 32: fma(code, d sc, -32 d sc) rounds it once, as the product does.
  static void stage(thread const item& v, uint, uint j, uint u, threadgroup half4* o) {
    const char sc = as_type<char>(uchar(v.s[u / 2] >> (8 * (2 * (u % 2) + j / 2))));
    const float s = float(v.d) * float(sc);
#pragma unroll
    for (int w = 0; w < 2; ++w) {
      const uint q = (v.l[u % 2][w] >> (4 * (u / 2)) & 0x0F0F0F0Fu) | (v.h[w] >> (2 * u) & 0x03030303u) << 4;
      o[w] = half4(fma(float4(as_type<uchar4>(q)), float4(s), float4(-32.0f * s)));
    }
  }
};

// Q8_0, unit = blocks 4 unit .. + 3 (group u = block 4 unit + u): bytes 8 j .. + 7 of each block's qs and its d.
struct st_q8_0 {
  struct item {
    uint q[4][2];
    half d[4];
  };
  static item load(bytes_in row, uint unit, uint j) {
    item v;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      bytes_in b = row + (4 * unit + g) * 34;
      v.q[g][0] = u32_at(b + 2 + 8 * j);
      v.q[g][1] = u32_at(b + 6 + 8 * j);
      v.d[g] = *(device const half*)b;
    }
    return v;
  }
  static void stage(thread const item& v, uint, uint, uint u, threadgroup half4* o) {
#pragma unroll
    for (int w = 0; w < 2; ++w) o[w] = half4(float4(as_type<char4>(v.q[u][w])) * float(v.d[u]));
  }
};

// Element (r, n) of out: bias[n] added when there is one, then stored as f32 or f16 (ops.metal's store_c).
static void st_store(device uchar* c, constant mm_args& p, bytes_in bias, uint r, uint n, float v) {
  if (p.has_bias) v += *(device const float*)(bias + n * p.bias_rs);
  device uchar* row = c + r * p.c_rs;
  if (p.c_f32)
    ((device float*)row)[n] = v;
  else
    ((device half*)row)[n] = half(v);
}

// fn(r, c, i) for each element i of acc in row r < m, column c < n of the tile at n0.
template <class Acc, class Fn>
static void st_each(thread Acc& acc, uint m, uint n, uint n0, Fn fn) {
#pragma unroll
  for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    const uint r = uint(index[1]), c = n0 + uint(index[0]);
    if (r < m && c < n) fn(r, c, i);
  }
}

// The tile's halves a column in the stage: KS, 8 more where both buffers still fit 16 KB, the most static threadgroup
// memory shader validation (which counts it twice against 32 KB) allows.
constexpr int st_lead(int ks) { return ks + (2 * ST_N * (ks + 8) * 2 <= 16384 ? 8 : 0); }

using st_dev_t = tensor<device half, dextents<int, 2>, tensor_inline>;
using st_tg_t = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;

// The destination of an 8 RT-row tile, left for st_walk to zero: Splash's staged_accumulator returns it uninitialized
// (an initialized one returned lost its values on Apple9).
template <int KS, int RT>
static auto st_acc() {
  constexpr auto descriptor = matmul2d_descriptor(8 * RT, ST_N, KS, false, true, false,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> op;
  return op.template get_destination_cooperative_tensor<st_dev_t, st_tg_t, float>();
}

// Where a column tile's walk of units [ub, ue) starts: Splash's spread start, tile t at (t * 40503 mod 2^16) / 2^16
// of the way, the golden ratio's fraction, so the tiles do not all wait on the same freshly written slice of x; ub
// when spread is 0. The tile alone picks it, never the rows, so a row's sums run in one order at every row count.
static uint st_first(uint spread, uint tile, uint ub, uint ue) {
  return spread ? ub + uint(ulong(tile * 40503u & 0xFFFFu) * (ue - ub) >> 16) : ub;
}

// The staged walk: acc = x[0 .. m) . w[n0 .. n0 + ST_N) over units [ub, ue) of k, from first round to
// first - 1, through a stage of 2 ST_N L halves. w's row c is at w + c w_rs, x's row r at x + r x_ld halves (read in
// place, m <= 8 RT). Per unit the lanes hold its quant bytes (args.h), stage it as f16 KS inputs a step into the free
// half of a double buffer, load the next unit's bytes behind the unit's last stage, and run matmul2d(8 RT, ST_N, KS)
// on it, f32 accumulate.
template <class F, int KS, int RT, int L, class Acc>
static void st_walk(bytes_in w, u64 w_rs, uint n, uint n0, device half* x, int x_ld, uint m, uint ub, uint ue,
                    uint first, ushort lane, threadgroup half* stage, thread Acc& acc) {
  constexpr int S = ST_UNIT / KS, G = KS / 32;
  constexpr auto descriptor = matmul2d_descriptor(8 * RT, ST_N, KS, false, true, false,
                                                   matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> op;
  st_tg_t w0(stage, dextents<int, 2>(KS, ST_N), array<int, 2>({1, L}));
  st_tg_t w1(stage + ST_N * L, dextents<int, 2>(KS, ST_N), array<int, 2>({1, L}));
  const uint j = lane % 4;
  bytes_in rows[4];
  typename F::item v[4];
#pragma unroll
  for (int t = 0; t < 4; ++t) {
    // A ragged last tile decodes column n - 1 again for the columns past n, and stores none of them.
    rows[t] = w + ulong(min(n0 + lane / 4 + 8 * t, n - 1)) * w_rs;
    v[t] = F::load(rows[t], first, j);
  }
#pragma unroll
  for (uint16_t i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  // The walk: units [first, ue), then [ub, first). The last unit of the first run loads ub next.
  uint walked = 0, step = 0;
  for (uint run = 0; run < 2; ++run) {
    const uint from = run ? ub : first, to = run ? first : ue;
    for (uint u = from; u < to; ++u, ++walked) {
      const uint next = u + 1 < to ? u + 1 : ub;
      const bool more = walked + 1 < ue - ub;
#pragma unroll
      for (int s = 0; s < S; ++s, ++step) {
        const uint buf = step & 1;
        threadgroup half* st = stage + buf * (ST_N * L);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
          for (int g = 0; g < G; ++g)
            F::stage(v[t], u, j, uint(s * G + g), (threadgroup half4*)(st + (lane / 4 + 8 * t) * L + g * 32 + 8 * j));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (s == S - 1 && more) {
#pragma unroll
          for (int t = 0; t < 4; ++t) v[t] = F::load(rows[t], next, j);
        }
        st_dev_t xt(x + u * ST_UNIT + uint(s * KS), dextents<int, 2>(KS, int(m)), array<int, 2>({1, x_ld}));
        if (buf)
          op.run(xt, w1, acc);
        else
          op.run(xt, w0, acc);
      }
    }
  }
}

// split_reduce.h's arrival at a column tile's counter, after this part's sums are out: true in the last of the
// splits parts to arrive, which adds them and then zeroes the counter.
static bool st_last(device atomic_uint* counter, uint splits, ushort lane) {
  threadgroup_barrier(mem_flags::mem_device);  // every lane's parts out before lane 0 counts this part in
  uint arrived = 0;
  if (lane == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    arrived = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_device);
  return simd_broadcast(arrived, 0) == splits - 1;
}

/// mmvq on the staged tile: out[r, n] = x[r, :] . w[n, :] + bias[n] for up to 8 RT rows. Grid (ceil(n / ST_N),
/// splits), one simdgroup a threadgroup. Threadgroup (t, p) walks (st_walk) columns ST_N t .. + ST_N - 1 over part p
/// of k's units. With splits > 1 each part writes its f32 sums to parts [splits, m, n] and the last part to arrive
/// (counters[t], zero between calls) adds them in part order: no extra dispatch, the same bits every run.
template <class F, int KS, int RT>
kernel void mmvq_st(bytes_in w [[buffer(0)]], device half* x [[buffer(1)]], bytes_in bias [[buffer(2)]],
                    device uchar* out [[buffer(3)]], device coherent(device) float* parts [[buffer(4)]],
                    device atomic_uint* counters [[buffer(5)]], constant st_args& a [[buffer(6)]],
                    uint2 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup half stage[2 * ST_N * st_lead(KS)];
  const uint m = a.mm.m, n = a.mm.n, n0 = tg.x * ST_N, units = a.mm.k / ST_UNIT;
  const uint per = (units + a.splits - 1) / a.splits, ub = tg.y * per, ue = min(ub + per, units);
  auto acc = st_acc<KS, RT>();
  st_walk<F, KS, RT, st_lead(KS)>(w, a.mm.b_rs, n, n0, x, int(a.mm.a_rs / 2), m, ub, ue,
                                  st_first(a.spread, tg.x, ub, ue), lane, stage, acc);
  if (a.splits == 1) {
    st_each(acc, m, n, n0, [&](uint r, uint c, uint16_t i) { st_store(out, a.mm, bias, r, c, acc[i]); });
    return;
  }
  st_each(acc, m, n, n0, [&](uint r, uint c, uint16_t i) { parts[(ulong(tg.y) * m + r) * n + c] = acc[i]; });
  if (!st_last(counters + tg.x, a.splits, lane)) return;
  st_each(acc, m, n, n0, [&](uint r, uint c, uint16_t i) {
    float total = 0.0f;
    for (uint p = 0; p < a.splits; ++p) total += p == tg.y ? float(acc[i]) : float(parts[(ulong(p) * m + r) * n + c]);
    st_store(out, a.mm, bias, r, c, total);
  });
  if (lane == 0) atomic_store_explicit(counters + tg.x, 0u, memory_order_relaxed);
}

typedef decltype(mmvq_st<st_q8_0, 64, 1>) mmvq_st_t;
#define ST(fmt, ks)                                                                                    \
  template [[host_name("vv_mmvq_st" #ks "_m8_" #fmt)]] kernel mmvq_st_t mmvq_st<st_##fmt, ks, 1>;   \
  template [[host_name("vv_mmvq_st" #ks "_m16_" #fmt)]] kernel mmvq_st_t mmvq_st<st_##fmt, ks, 2>;  \
  template [[host_name("vv_mmvq_st" #ks "_m32_" #fmt)]] kernel mmvq_st_t mmvq_st<st_##fmt, ks, 4>;
#define ST_FORMAT(fmt) ST(fmt, 32) ST(fmt, 64) ST(fmt, 128)
// skipped: Q4_0 and Q5_0 (design: they come free), add them with the decode route if a model routes them here.
ST_FORMAT(q3_k)
ST_FORMAT(q4_k)
ST_FORMAT(q5_k)
ST_FORMAT(q6_k)
ST_FORMAT(q8_0)
