// larch_attn_full on M5 (Apple GPU family 10): MLX's steel_attention_nax (Metal 4 tensor matmul), vendored
// verbatim under mlx/ (MIT), with the same fixed function constants as attn_full_sg.metal.
// ponytail: head_dim 64 only; 72 (the 35B's encoder) takes the simdgroup kernel. MLX pads 72 to 96 for NAX:
// do that if the 35B's encode profile shows attn_full over half its time.
//
// Also prefill attention over the paged KV pool (vv_attn_paged_nax*, below): MLX's head-dim split NAX kernel
// (attention_nax_dsplit), reading K and V through the block table. It lives here because it needs the same
// macOS 26.2 cooperative-tensor layout as MLX's kernels (metal.cmake builds this file alone for 26.2).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

#include "args.h"

instantiate_kernel("vv_attn_full_nax64", attention_nax, half, 64, 32, 64, 64, 4, 1, float, float)

// Prefill attention, larch_attn_prefill's contract (backend.mm attend), on NAX fragments: MLX's
// attention_nax_dsplit (fc344c9) with Larch's addressing. Threadgroup (tile, q head) takes BQ = 16 WM consecutive q
// rows of one sequence; simdgroup (row group, d group) owns 16 of those rows and BD / WN of head_dim, keeps its Q
// slice in registers and walks the positions the tile sees BK = 32 at a time. Each 16-position half of a step is one
// 16-slot pool block (block_tokens 16, the NAX fragment's height), so K and V load straight from the pool through
// the block table, with no staging. With WN > 1 the d groups add their partial Q K^T in a fixed order through
// threadgroup memory, so every d group runs the same softmax. S, the online softmax and O stay f32; P meets V as f32
// against f16 (MLX's mixed mma). Masked scores take finite_min, so a dead row (past the tile's rows, whose Q loads
// as zeros) stays finite and is never stored. Tile x counts the sequences' BQ-row tiles in order; the host launches
// an upper bound, and the tiles past the last return at once. head_dim is exactly BD.
template <int BD, int WM, int WN, bool HM>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attn_paged_nax(
    const device uchar* q [[buffer(0)]], const device half* kp [[buffer(1)]], const device half* vp [[buffer(2)]],
    const device uchar* bt [[buffer(3)]], const device uchar* lens [[buffer(4)]],
    const device uchar* start [[buffer(5)]], device uchar* out [[buffer(6)]], constant attn_args& a [[buffer(7)]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr short kU = 16, BQ = WM * kU, BK = 2 * kU, TK = BK / kU, TD = BD / kU, TDh = TD / WN, BDh = BD / WN;
  static_assert(TD % WN == 0 && TDh % 2 == 0, "each d group holds pairs of 16-wide head_dim fragments");
  using stile_t = NAXTile<float, 1, TK>;
  using otile_t = NAXTile<float, 1, TDh>;
  using frag_t = stile_t::NAXFrag_t;
  constexpr short kEPF = frag_t::kElemsPerFrag, kRowsPT = otile_t::kRowsPerThread;

  uint t = HM ? tg.y : tg.x, s = 0, first = 0, len = 0;
  for (; s < a.seqs; ++s, first += len, t -= (len + BQ - 1) / BQ) {
    len = uint(max(*(const device int*)(lens + s * a.lens_rs), 0));
    if (t < (len + BQ - 1) / BQ) break;
  }
  // Past the last tile, or past q's rows: lens that sum past them are a caller bug the host cannot see.
  if (s == a.seqs || first + t * BQ >= a.q_rows) return;  // the whole threadgroup
  const uint j0 = t * BQ, rows = min(min(uint(BQ), len - j0), a.q_rows - first - j0);
  const uint h = HM ? tg.x : tg.y, kvh = h / (a.heads / a.kv_heads);
  const int pos0 = *(const device int*)(start + s * a.start_rs) + int(j0), kend = pos0 + int(rows);  // keys [0, kend)
  const device int* table = (const device int*)(bt + s * a.bt_rs);

  const short row_group = short(simd_group_id) / WN, d_group = short(simd_group_id) % WN, tm = kU * row_group;
  const short lim_q = short(rows) - tm;  // this simdgroup's live rows; 0 or less loads zeros and stores nothing
  const int q_ld = int(a.q_rs / 2), o_ld = int(a.out_rs / 2), kv_ld = int(a.kv_heads * a.hd);
  const device half* Q = (const device half*)(q + u64(first + j0 + tm) * a.q_rs) + h * a.hd + d_group * BDh;
  device half* O = (device half*)(out + u64(first + j0 + tm) * a.out_rs) + h * a.hd + d_group * BDh;

  NAXTile<half, 1, 1> Qtiles[TDh];
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TDh; id++) Qtiles[id].load_rows(Q + id * kU, q_ld, lim_q);

  otile_t Otile;
  Otile.clear();
  metal::vec<float, kRowsPT> max_score, sum_score{0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) max_score[i] = Limits<float>::finite_min;
  const float scale2 = a.scale * 1.44269504089f;  // exp2 softmax
  const short2 coord = otile_t::NAXFrag_t::get_coord();
  const short sm = coord.y, sn = coord.x;
  threadgroup float s_xchg[WM][WN][TK * kEPF * 32];

  const int kb_lim = (kend + BK - 1) / BK, kb_min_causal = pos0 / BK;
  for (int kb = 0; kb < kb_lim; kb++) {
    // The step's two pool blocks and their live slots; a block wholly past kend is never read (0 rows).
    const device half* kblk[TK];
    const device half* vblk[TK];
    short nk[TK];
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      const int c0 = kb * BK + ik * kU;
      nk[ik] = short(clamp(kend - c0, 0, int(kU)));
      const u64 base = nk[ik] > 0 ? (u64(uint(table[c0 / kU])) * kU * a.kv_heads + kvh) * a.hd : 0;
      kblk[ik] = kp + base + d_group * BDh;
      vblk[ik] = vp + base + d_group * BDh;
    }

    stile_t Stile;
    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        NAXTile<half, 2, 1> Ktile;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
          if (nk[ik + i] == kU)
            frag_t::load(Ktile.frag_at(i, 0), kblk[ik + i] + id * kU, kv_ld, Int<1>{});
          else
            frag_t::load_rows(Ktile.frag_at(i, 0), kblk[ik + i] + id * kU, kv_ld, Int<1>{}, nk[ik + i]);
        }
        frag_t::mma(Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1), Qtiles[id].frag_at(0, 0), metal::false_type{},
                    Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
      }
    }

    if constexpr (WN > 1) {  // every d group's partial scores, added in d group order
      threadgroup float* slot = s_xchg[row_group][d_group];
      const short base = short(simd_lane_id) * (TK * kEPF);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& sf = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEPF; i++) slot[base + ik * kEPF + i] = sf[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& sf = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEPF; i++) sf[i] = s_xchg[row_group][0][base + ik * kEPF + i];
        STEEL_PRAGMA_UNROLL
        for (short peer = 1; peer < WN; peer++)
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < kEPF; i++) sf[i] += s_xchg[row_group][peer][base + ik * kEPF + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) Stile.elems()[ii] *= scale2;

    if (kb >= kb_min_causal) {  // causal: row position r sees keys c <= r (and so none at or past kend)
      const int base_row = pos0 + tm, base_col = kb * BK;
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const int r = base_row + ii * stile_t::kFragRowsJump + sm, c = base_col + ik * kU + jj + sn;
            const short loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = r < c ? Limits<float>::finite_min : fg[loc];
          }
        }
      }
    }

    metal::vec<float, kRowsPT> new_max, factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
      sum_score[i] = sum_score[i] * factor[i];
    }
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);
    simdgroup_barrier(mem_flags::mem_none);

    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<half, 1, 2> Vtile;
        if (nk[ik] == kU)
          Vtile.load(vblk[ik] + id * kU, kv_ld);
        else
          Vtile.load_rows(vblk[ik] + id * kU, kv_ld, nk[ik]);
        frag_t::mma(Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Stile.frag_at(0, ik), metal::false_type{},
                    Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
      }
    }
  }

  metal::vec<float, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) rcp[i] = 1.f / sum_score[i];
  Otile.template row_bin_op<MulOp>(rcp);
  if (lim_q <= 0) return;
  if (lim_q < kU)
    Otile.store_rows(O, o_ld, lim_q);
  else
    Otile.store(O, o_ld);
}
// head_dim 64 and 128 without a split, 256 split in 4 (MLX's wide-head shape): 64 q rows, 4 simdgroups at 64 and
// 128, 32 rows in 2 x 4 at 256 (16 KB of score exchange). HM (the _hm kernels, LARCH_METAL_ATTN_NAX_HM) swaps the
// grid so x is the q head: the threadgroups that run together then read every kv head's half of the same pool slots,
// for A/B timing at long contexts (1.5x MLX at 32K positions with x the tile, M5, 27 Nov).
#define PAGED_NAX(name, bd, wm, wn, hm)                                                     \
  template [[host_name(name)]] kernel decltype(attn_paged_nax<bd, wm, wn, hm>) attn_paged_nax<bd, wm, wn, hm>;
PAGED_NAX("vv_attn_paged_nax64", 64, 4, 1, false)
PAGED_NAX("vv_attn_paged_nax128", 128, 4, 1, false)
PAGED_NAX("vv_attn_paged_nax256", 256, 2, 4, false)
PAGED_NAX("vv_attn_paged_nax64_hm", 64, 4, 1, true)
PAGED_NAX("vv_attn_paged_nax128_hm", 128, 4, 1, true)
PAGED_NAX("vv_attn_paged_nax256_hm", 256, 2, 4, true)
#pragma clang diagnostic pop

// mmvq at 2 to 16 rows on NAX fragments, Q3_K: MLX's affine_qmv_nax (quantized_nax.h, "few rows of x times
// transposed weights") with GGUF Q3_K blocks read in place. Threadgroup t takes columns 32 t .. + 31 (two 16-column
// B fragments); its MMVQ_NAX_WK simdgroups split k's units (half blocks, ST_UNIT inputs), simdgroup s taking units
// s, s + WK, .... Lane (fm, fn) holds x rows fm and fm + 8 and w columns fm and fm + 8 of each fragment. It decodes
// item j = fn / 4 of each unit with staged.metal's st_q3_k lanes (bytes 8 j .. + 7 of the half's qs and hmask), so
// its 32 inputs are 32 u + 8 j .. + 7 for u = 0..3, and it loads the matching x; A and B order k the same way. Each
// weight is decoded once to the f16 value vv_mmvq_st stages, and meets every row in one 16x32x16 matmul per 16
// inputs, so 2 rows cost what 16 do in MMA work but the decode is not repeated. The simdgroups' f32 sums add in
// simdgroup order through threadgroup memory. The host sends k a multiple of 256 and x rows 16-byte aligned.
#define MMVQ_NAX_WK 8

static inline uint nax_u32_at(const device uchar* p) {  // 4 bytes at a 2-byte aligned p (Q3_K blocks are 110 bytes)
  const device ushort* h = (const device ushort*)p;
  return uint(h[0]) | uint(h[1]) << 16;
}

[[kernel, max_total_threads_per_threadgroup(MMVQ_NAX_WK * 32)]] void vv_mmvq_nax_q3_k(
    const device uchar* w [[buffer(0)]], const device uchar* x [[buffer(1)]], const device uchar* bias [[buffer(2)]],
    device uchar* out [[buffer(3)]], constant mm_args& a [[buffer(4)]], uint3 tid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  using Frag = BaseNAXFrag;
  using afrag = typename Frag::template dtype_frag_t<half>;
  using cfrag = typename Frag::template dtype_frag_t<float>;
  constexpr short E = Frag::kElemsPerFrag;
  const short2 coord = Frag::get_coord();
  const uint fm = uint(coord.y), fn = uint(coord.x), j = fn / 4;
  const uint m = a.m, n = a.n, units = a.k / 128, col0 = tid.x * 32;
  const device uchar* rows[2][2];
  STEEL_PRAGMA_UNROLL
  for (short t = 0; t < 2; t++) {
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; r++)  // a ragged last tile decodes column n - 1 for the columns past n, stored nowhere
      rows[t][r] = w + ulong(min(col0 + 16 * t + 8 * r + fm, n - 1)) * a.b_rs;
  }
  const bool x_rows[2] = {fm < m, fm + 8 < m};
  const device uchar* xr[2] = {x + ulong(fm) * a.a_rs, x + ulong(fm + 8) * a.a_rs};
  cfrag D[2] = {cfrag(0), cfrag(0)};
  for (uint u = sg; u < units; u += MMVQ_NAX_WK) {
    vec<half, 8> xs[2][4];
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; r++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++)
        xs[r][i] = x_rows[r] ? *(const device vec<half, 8>*)(xr[r] + 2 * (u * 128 + 32 * i + 8 * j)) : vec<half, 8>(0);
    }
    // st_q3_k::load and stage: bit pair i of the qs byte, minus 4 where hmask bit 4 h + i is clear, times
    // d (scale 8 h + 2 i + j / 2 - 32); fma(code, dl, -4 dl) with code = q + 4.
    const uint h = u % 2;
    uint q[2][2][2], hm[2][2][2];
    float dl[2][2][4];
    STEEL_PRAGMA_UNROLL
    for (short t = 0; t < 2; t++) {
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < 2; r++) {
        const device uchar* b = rows[t][r] + u / 2 * 110;
        const device uchar *qp = b + 32 + 32 * h + 8 * j, *hp = b + 8 * j;
        q[t][r][0] = nax_u32_at(qp);
        q[t][r][1] = nax_u32_at(qp + 4);
        hm[t][r][0] = nax_u32_at(hp);
        hm[t][r][1] = nax_u32_at(hp + 4);
        const uint s0 = nax_u32_at(b + 96), s1 = nax_u32_at(b + 100), s2 = nax_u32_at(b + 104);
        const float d = float(*(const device half*)(b + 108));
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 4; i++) {
          const uint si = 8 * h + 2 * i + j / 2, sb = si % 8;
          const uint lo = (sb < 4 ? s0 : s1) >> (8 * (sb % 4) + 4 * h) & 0xF;
          const uint hi = s2 >> (8 * (si % 4) + 2 * (si / 4)) & 3;
          dl[t][r][i] = d * float(int(lo | hi << 4) - 32);
        }
      }
    }
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < 8; f++) {  // inputs 32 (f / 2) + 8 j + 4 (f % 2) .. + 3
      const short i = f / 2, word = f % 2;
      afrag af, bf[2];
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < 2; r++) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 4; e++) af[4 * r + e] = xs[r][i][4 * word + e];
      }
      STEEL_PRAGMA_UNROLL
      for (short t = 0; t < 2; t++) {
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 2; r++) {
          const uint c = (q[t][r][word] >> (2 * i) & 0x03030303u) | (hm[t][r][word] >> (4 * h + i) & 0x01010101u) << 2;
          const float s = dl[t][r][i];
          const half4 v = half4(fma(float4(as_type<uchar4>(c)), float4(s), float4(-4.0f * s)));
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 4; e++) bf[t][4 * r + e] = v[e];
        }
      }
      Frag::mma(D[0], D[1], af, metal::bool_constant<false>{}, bf[0], bf[1], metal::bool_constant<true>{});
    }
  }
  threadgroup float part[(MMVQ_NAX_WK - 1) * 2 * E * 32];
  if (sg > 0) {
    STEEL_PRAGMA_UNROLL
    for (short t = 0; t < 2; t++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < E; e++) part[(((sg - 1) * 2 + t) * E + e) * 32 + lane] = D[t][e];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  for (uint s = 1; s < MMVQ_NAX_WK; s++) {
    STEEL_PRAGMA_UNROLL
    for (short t = 0; t < 2; t++) {
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < E; e++) D[t][e] += part[(((s - 1) * 2 + t) * E + e) * 32 + lane];
    }
  }
  STEEL_PRAGMA_UNROLL
  for (short t = 0; t < 2; t++) {
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < E; e++) {
      const uint r = fm + 8 * (e / 4), c = col0 + 16 * t + fn + e % 4;
      if (r >= m || c >= n) continue;
      float v = D[t][e];
      if (a.has_bias) v += *(const device float*)(bias + c * a.bias_rs);
      device uchar* o = out + r * a.c_rs;
      if (a.c_f32)
        ((device float*)o)[c] = v;
      else
        ((device half*)o)[c] = half(v);
    }
  }
}
