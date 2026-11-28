// Kernel arguments, shared by backend.mm (setBytes) and ops.metal. Every `*_rs` is the byte stride
// between rows, which for a vector (larch.h allows stride[0] > 1) is the stride between elements.
#pragma once

#ifdef __METAL_VERSION__
typedef ulong u64;
typedef uint u32;
#else
#include <cstdint>
typedef uint64_t u64;
typedef uint32_t u32;
#endif

/// Elementwise ops over [rows, cols] (embed, dequant, silu_mul, add, kv_write): up to 3 tensors.
struct ew_args {
  u64 rs0, rs1, rs2;
  u32 cols;
};

/// rmsnorm and qk_norm: `heads` groups of n elements per row, each normalised on its own.
struct rms_args {
  u64 x_rs, w_rs, out_rs;
  u32 n, heads;
  float eps;
};

/// rope: the host-side constants of ggml's rope_yarn.
struct rope_args {
  u64 rs, pos_rs, ff_rs;
  u32 heads, hd, n_dims, neox, has_ff;
  float theta_scale, freq_scale, ext_factor, mscale, lo, hi;
};

/// layernorm: rms_args' row of n, minus its mean first, plus an optional bias.
struct ln_args {
  u64 x_rs, w_rs, b_rs, out_rs;
  u32 n, has_b;
  float eps;
};

/// rope_multi: rope_args' constants, the four sections and the mode (larch_rope_mode); pair i turns with i + pairs.
struct rope_multi_args {
  u64 rs, pos_rs, ff_rs;
  u32 hd, pairs, mode, has_ff;
  u32 s[4];
  float theta_scale, freq_scale, ext_factor, mscale, lo, hi;
};

/// attn_full: the simdgroups of MLX's attention kernels (WM 4, WN 1).
#define ATTN_FULL_SG 4

/// The tensor GEMM's tile of c: GEMM_TENSOR_M rows of a by GEMM_TENSOR_N rows of b_t, GEMM_TENSOR_SG
/// simdgroups. Measured (M5, 9 Oct, m 512-2048): 128 x 64 beat 64 x 64 by 10-25 % at every prefill shape, and
/// 64 x 128, 128 x 128, 128 x 32 and 8-simdgroup tiles.
#define GEMM_TENSOR_M 128
#define GEMM_TENSOR_N 64
#define GEMM_TENSOR_SG 4
/// The deep variant, 256 x 64 on 8 simdgroups, for k > n at 256+ rows (ffn down in a prefill chunk): 512 x 4096
/// x 14336 took 7-10 ms against 13 ms; equal at 1024 rows, slower at 64.
#define GEMM_TENSOR_DEEP_M 256
#define GEMM_TENSOR_DEEP_SG 8

/// mmvq_ext's lanes per column of w: 32 / MMVQ_EXT_NX columns per simdgroup, 4 simdgroups per threadgroup.
/// Measured (M5, 9 Oct, m1 shapes, 2-4 rows): 8 beat 4 and 16 by 3-8 %, except 4 on the 128256-wide output.
#define MMVQ_EXT_NX 8

/// mmvq_mma: simdgroups per threadgroup, each over every MMVQ_MMA_SG-th 64-wide k step, and 8-column tiles of w
/// per threadgroup. Measured (M5, 10 Oct, m1 shapes, 2-16 rows): 2 tiles was within 3 % of the best of 1, 2
/// and 4 everywhere but n 128256 (4 tiles, 2.5 % faster) and n 1024 (1 tile, up to 10 % faster, 3 us).
#define MMVQ_MMA_SG 8
#define MMVQ_MMA_NT 2

/// mmvq and gemm: c[m, n] = alpha * a[m, :] . b[n, :] + beta * c[m, n] + bias[n].
struct mm_args {
  u64 a_rs, b_rs, c_rs, bias_rs;
  u32 m, n, k, has_bias, c_f32;
  float alpha, beta;
  u32 m_fast;  // gemm: the grid's x walks c's row tiles (else its column tiles); 0 elsewhere
};

/// mmvq_id: row r's slot j multiplies x's row r (x_slot_rs 0) or (r, j) by row ids[r, j] * n + column of w.
/// Weighted, out's row r is the slots' sum by wt[r, j]; else out's (r, j) is that slot's product.
struct id_args {
  u64 x_rs, x_slot_rs, w_rs, ids_rs, wt_rs, out_rs, out_slot_rs, bias_rs;
  u32 n, k, slots, has_bias, out_f32, weighted, experts, rows;
};

/// mmvq_id grouped by expert (a prefill): from MOE_GROUP_PAIRS (row, slot) pairs per expert, each simdgroup
/// decodes one column of an expert once, MOE_GROUP_T sub-blocks per lane (so k up to 1024 MOE_GROUP_T), and
/// dots it with every pair that chose the expert, where the per-pair kernel decodes it again for each.
#define MOE_GROUP_PAIRS 4
#define MOE_GROUP_T 2
/// The grouped kernel's weighted f32 partials per call at most (a 1024-row qwen3moe down: 8 x 2048 x 4 bytes a
/// row); the engine's fit charges it with the pair lists.
#define MOE_PARTIAL_BYTES (64ull << 20)

/// mmvq_id grouped on the tensor path (vv_mmvq_id_mm): MOE_MM_SG simdgroups per tile of an expert's pairs and
/// columns, k MOE_MM_K at a time. qwen3moe averages 32 pairs an expert at a 512-token chunk. vv_moe_gather lays
/// x's rows out in expert order, MOE_MM_K-padded, for its 32 x 64 tile; the engine's fit charges them (fit.rs).
#define MOE_MM_SG 4
#define MOE_MM_K 64

/// moe_route: one simdgroup per row holds MOE_ROUTE_MAX / 32 logits per lane.
#define MOE_ROUTE_MAX 512
struct route_args {
  u64 lg_rs, ids_rs, w_rs;
  u32 experts, k;
  float scale;
};

/// gemm on a deferred dequant (gemm_q): a GEMM_Q_M (GEMM_Q_M_SMALL for few rows) x GEMM_Q_N (half that for
/// k <= n) tile of c per GEMM_Q_SG simdgroups, k in steps of GEMM_Q_K. Measured (M5, 10 Oct, m1 shapes at 52 and
/// 512 rows): k steps of 128 beat 64 by 3 % end to end; 32-row tiles lost at 52 rows.
#define GEMM_Q_M 128
#define GEMM_Q_M_SMALL 64
#define GEMM_Q_N 64
#define GEMM_Q_K 128
#define GEMM_Q_SG 4

/// The staged decode tile (vv_mmvq_st<KS>_m<ROWS>_<fmt>, staged.metal): mmvq at 1 to 32 rows on one kernel, so a
/// row's bits do not depend on the rows beside it. A threadgroup of one simdgroup owns ST_N columns of w and walks
/// k in units of ST_UNIT inputs (half a K-quant block, four Q8_0 blocks), each lane holding its share of the next
/// unit's quant bytes in registers while the current unit is staged as f16, KS inputs a step (32, 64 or 128),
/// into a double buffer that matmul2d multiplies by up to ROWS (8, 16 or 32) rows of x.
#define ST_N 32
#define ST_UNIT 128
/// The K split: parts of whole units until the grid has ST_TG_PER_CORE threadgroups a core (the design's 8 of two
/// simdgroups), each part at least ST_MIN_PART inputs, at most ST_MAX_SPLITS parts.
#define ST_TG_PER_CORE 16
#define ST_MIN_PART 512
#define ST_MAX_SPLITS 8

/// The staged tile's arguments: mm (a = x, b = w, c = out; alpha 1, beta 0, m_fast unused), the K split's parts
/// (st_splits), and spread 1 to start each column tile's walk at its own unit (0: every tile from its part's first).
struct st_args {
  mm_args mm;
  u32 splits, spread;
};

/// The staged tile's K split for an n x k weight on a GPU of `cores` cores. Shape and core count only, never the
/// row count, so a row's sums are added in the same order at every row count. Every part gets units.
static inline u32 st_splits(u32 n, u32 k, u32 cores) {
  const u32 tiles = (n + ST_N - 1) / ST_N, units = k / ST_UNIT, most = k / ST_MIN_PART;
  u32 s = (ST_TG_PER_CORE * cores + tiles - 1) / tiles;
  s = s < ST_MAX_SPLITS ? s : ST_MAX_SPLITS;
  s = s < most ? s : most;
  if (s <= 1) return 1;
  const u32 per = (units + s - 1) / s;
  return (units + per - 1) / per;
}

/// The simdgroup-matrix kernels for GPUs without the tensor API (M1 to M4, or LARCH_METAL_TENSOR=0): gemm on a
/// deferred dequant (vv_gemm_qs) and the grouped expert mm (vv_mmvq_id_mm_sg) take a GEMM_QS_M x GEMM_QS_N tile
/// of c (rows of a or an expert's pairs, by rows of w) on GEMM_QS_SG simdgroups, k GEMM_QS_K at a time: thread t
/// decodes one 32-element sub-block of w per step. llama.cpp's kernel_mul_mm has the same tile.
#define GEMM_QS_M 32
#define GEMM_QS_N 64
#define GEMM_QS_K 64
#define GEMM_QS_SG 4

/// gemm at up to GEMV_M rows of a: vv_gemv, a threadgroup of GEMV_SG simdgroups per column of c splitting k, so a
/// narrow matrix (a gated-delta ssm_alpha or ssm_beta, 32 columns; a shared-expert gate, 1) still fills the GPU.
#define GEMV_M 8
#define GEMV_SG 4

/// mmvq at 1 row (Q4_K, Q6_K): MMVQ1_SG simdgroups per threadgroup, MMVQ1_NR columns each, as llama.cpp's
/// mul_mv for these formats.
#define MMVQ1_SG 2
#define MMVQ1_NR 2
#define MMVQ1N_C 4  // columns per simdgroup of mmvq_id1n (experts under 4 blocks wide)

/// Attention's shape limits, shared so host dispatch and kernel layout agree: one simdgroup serves up to
/// ATTN_GROUP q heads of a kv head, each lane holds 2 chunks of 4 dims per head, and the split-K reduce
/// weighs two splits per lane. Splits past 16 run only for sequences of ATTN_SPLIT_POS positions a split or more
/// (lengths are device data): few-head models (2 kv heads of 16 q: 4 head groups) at long contexts, where
/// 16 splits left most of the GPU idle (8k decode 357 us, MLX 144, M5, 1 Nov), without the 2x loss 64 splits
/// had at 128-512 positions.
#define ATTN_GROUP 4
#define ATTN_MAX_HEAD_DIM 256
#define ATTN_MAX_SPLITS 64
#define ATTN_SPLIT_POS 128

/// Flash prefill attention: ATTN_FLASH_ROWS q rows of one sequence per threadgroup of ATTN_FLASH_SG simdgroups,
/// one pool block of ATTN_FLASH_BLOCK slots per step, head_dim a multiple of 8 up to 256. Measured (M5, 9 Oct,
/// m1 heads, one sequence): 16 and 64 rows were no faster than 32, two blocks per step no faster than one.
/// The tensor kernel takes head_dim 129-256 in ATTN_FLASH_ROWS256 rows: 32 rows used 41,496 bytes of
/// threadgroup memory there, past the 32 KB limit (M5, run 12, Metal API validation).
#define ATTN_FLASH_ROWS 32
#define ATTN_FLASH_ROWS256 16
#define ATTN_FLASH_SG 4
#define ATTN_FLASH_BLOCK 16

/// Attention's tensor strides and geometry; split-K decode also uses splits and partial_stride, and flash
/// prefill bounds its rows by q_rows.
struct attn_args {
  u64 q_rs, out_rs, bt_rs, lens_rs, start_rs;
  u32 heads, kv_heads, hd, block_tokens, seqs, prefill, splits, partial_stride;
  float scale;
  u32 q_rows;
};

/// gdn_conv and gdn (Gated DeltaNet): the strides, the layer's state slots (state_rs bytes apart) and geometry.
/// gdn: one simdgroup holds GDN_ROWS rows of a v head's delta-rule matrix, head_dim / 32 entries a row per lane,
/// so head_dim is a multiple of 32 up to GDN_MAX_HD. gdn_conv keeps a channel's last taps - 1 inputs in
/// registers, taps up to GDN_MAX_TAPS.
#define GDN_ROWS 4
#define GDN_SG 4          // simdgroups per gdn threadgroup
#define GDN_WIDE_ROWS 32  // gdn rows a sequence, on average, up to which each simdgroup takes one state row
#define GDN_MAX_HD 256
#define GDN_MAX_TAPS 8
struct gdn_args {
  u64 in_rs, out_rs, w_rs, alpha_rs, beta_rs, dt_rs, a_rs, slots_rs, lens_rs, snaps_rs, state_rs;
  u32 seqs, channels, taps, k_heads, v_heads, hd, has_snaps;
  float eps, scale;
};

/// draft_conv: the row strides, d, its channels per group and the groups, the stage's first tap row and an f32 out.
struct draft_conv_args {
  u64 x_rs, dyn_rs, base_rs, out_rs;
  u32 d, group, groups, t0, f32_out;
};

/// sample_prep: the strides of logits, penalty ids and the three outputs.
struct sample_args {
  u64 lg_rs, pen_rs, vals_rs, ids_rs, lse_rs;
  u32 vocab;
};

/// sample_prep: vv_sp_part sorts SP_CHUNK logits of a row per threadgroup of SP_THREADS into a partial of SP_PART
/// floats (max, sum of exp, then 64 values and 64 ids), and vv_sp_merge merges a row's partials on SP_MERGE_THREADS,
/// one part per lane. Each keeps its threadgroup memory under 16 KB: shader validation doubles it, against 32 KB.
/// Measured (M5, 10 Oct, 128256 ids, top 64, rows 1/4/16): chunks of 1024 on 128 threads 79/212/761 us, 1024 on
/// 256 81/225/819, 2048 on 256 93/253/912; one threadgroup per row (before) 1042/1074/3288.
#define SP_CHUNK 1024
#define SP_PART 130
#define SP_THREADS 128
#define SP_MERGE_THREADS 512

/// larch_sample_params, field for field (backend.mm asserts the layout).
struct sample_row {
  float temperature;
  u32 top_k;
  float repeat_penalty;
  u32 penalty_window;
};
