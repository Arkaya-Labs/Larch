/* larch.h: the C ABI between larch-sys and the kernel backends (architecture section 5).
 * Everything is prefixed larch_. Every call returns larch_status, except the destroy/free calls
 * and the queries named below. Handles are opaque. Device memory is reached only through
 * larch_buffer* + byte offset; raw device pointers never cross the ABI. Device::open checks
 * LARCH_ABI_VERSION against larch_abi_version().
 *
 * One library holds two CPU devices and at most one GPU backend. Ordinal -1 opens the CPU reference,
 * the oracle; -2 the CPU device the engine computes on (the fast CPU backend's, multi-threaded:
 * larch_device_set_threads; until that backend lands it runs the reference's ops); 0.. the GPU
 * devices. Every handle then calls the backend it came from (src/common/dispatch.cpp). Two things
 * cross devices: host buffers (larch_buffer_alloc_host), which any CPU device's ops take, and events
 * (larch_stream_wait).
 *
 * All ops are asynchronous on a stream. Host memory: larch_upload's src, and everything an op takes
 * by pointer (tensors, params, pools, per_row), may be reused as soon as the call returns.
 * larch_download's dst must stay allocated, at the same address, until larch_stream_sync on that
 * stream returns, or larch_event_sync on an event recorded on it after the download, and holds the
 * data once that returns LARCH_OK (a backend may finish earlier, never later).
 *
 * ABI 3 to 5 add entries a backend may lack: it then answers a call LARCH_ERR_UNSUPPORTED, after the
 * same argument checks as every backend (so a malformed call is still INVALID_ARG). The CPU devices
 * have them all. */
#ifndef LARCH_H
#define LARCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LARCH_ABI_VERSION 5

typedef enum {
  LARCH_OK = 0,
  LARCH_ERR_INVALID_ARG,
  LARCH_ERR_OOM,
  LARCH_ERR_UNSUPPORTED,
  LARCH_ERR_LAUNCH,
  LARCH_ERR_DEVICE_LOST,
  LARCH_ERR_TIMEOUT,
  LARCH_ERR_DRIVER
} larch_status;

typedef enum { LARCH_BACKEND_CPU = 0, LARCH_BACKEND_CUDA, LARCH_BACKEND_HIP, LARCH_BACKEND_METAL } larch_backend;

/* GGML block layouts, byte-identical. */
typedef enum {
  LARCH_F32 = 0,
  LARCH_F16,
  LARCH_BF16,
  LARCH_I32,
  LARCH_Q4_K,
  LARCH_Q5_K,
  LARCH_Q6_K,
  LARCH_Q8_0,
  LARCH_Q4_0,
  LARCH_Q5_0,
  LARCH_FP8_E4M3, /* KV pages only: OCP E4M3FN, one byte (bias 7, no infinity, S.1111.111 is NaN) */
  LARCH_Q3_K,     /* weights only: a GPU backend without its kernels answers UNSUPPORTED */
  LARCH_Q2_K      /* ABI 5, weights only: a GPU backend answers UNSUPPORTED where it has no kernel for it (CUDA has
                      embed, dequant and the mmvq_id calls) */
} larch_dtype;

/* larch_device_props.flags */
#define LARCH_CAP_GRAPHS (1u << 0)
#define LARCH_CAP_BF16 (1u << 1)
#define LARCH_CAP_ASYNC_COPY (1u << 2)
#define LARCH_CAP_UNIFIED (1u << 3)

typedef struct larch_device larch_device;
typedef struct larch_stream larch_stream;
typedef struct larch_buffer larch_buffer;
typedef struct larch_event larch_event;
typedef struct larch_graph larch_graph;

typedef struct {
  char name[64];
  larch_backend backend;
  uint32_t arch_code; /* CUDA: major*10+minor (86, 89, 120); HIP: the gfx number (1030, 1100, 1201);
                         Metal: the raw MTLGPUFamily value of the highest Apple family (1010 = Apple10) */
  uint32_t compute_units; /* SMs or CUs; a CPU device: its threads (the reference: 1) */
  uint32_t warp_size;
  uint32_t max_threads_per_group;
  uint64_t shared_mem_per_group;
  uint64_t l2_bytes;
  uint64_t vram_total; /* bytes the GPU may use: device memory, or on unified memory Metal's
                          recommendedMaxWorkingSetSize (not physical RAM) */
  uint64_t vram_free;
  uint64_t recommended_working_set; /* Apple: recommendedMaxWorkingSetSize; = vram_free elsewhere */
  float peak_gbps;                  /* from bus width x clock, or Apple table fallback */
  uint32_t driver_version;
  uint32_t runtime_version;
  uint32_t flags; /* LARCH_CAP_GRAPHS | _BF16 | _ASYNC_COPY | _UNIFIED */
} larch_device_props;

typedef struct {
  larch_buffer* buf;
  uint64_t offset;
  larch_dtype dtype;
  uint32_t ndim;
  int64_t shape[4];
  int64_t stride[4]; /* in elements (blocks for quant) */
} larch_tensor;

/* The paged KV pool, viewed at one layer: the state of one attention layer group, which may differ
 * per layer (dtype here, a sliding window at larch_attn). Each of k and v is [n_layers][n_blocks *
 * block_tokens][n_kv_heads][head_dim] of dtype; block_tokens, n_kv_heads and head_dim are > 0 and
 * layer < n_layers. Position p of a sequence lives in slot
 * block_table[seq][p / block_tokens] * block_tokens + p % block_tokens.
 * dtype: f16; q8_0, with head_dim a multiple of 32 so each slot's row is whole GGML q8_0 blocks; or
 * fp8_e4m3. kv_write rounds into it: q8_0 as ggml's quantize_row_q8_0_ref, fp8 to nearest even,
 * saturating at +-448. A backend without a dtype's pages answers every op on them UNSUPPORTED. */
typedef struct {
  larch_buffer* k;
  larch_buffer* v;
  uint32_t n_blocks, block_tokens, n_layers, n_kv_heads, head_dim;
  larch_dtype dtype;
  uint32_t layer;
} larch_kv_pool;

/* The parameters of ggml_rope_ext, so results match llama.cpp's on the same GGUF. */
typedef struct {
  uint32_t n_dims;     /* rotated dimensions per head, <= head_dim */
  uint32_t neox;       /* 0 = normal (adjacent pairs), 1 = neox (first and second half) */
  uint32_t n_ctx_orig; /* YaRN: the training context */
  float freq_base;     /* theta */
  float freq_scale;    /* linear scaling: 1 / factor */
  float ext_factor;    /* YaRN; 0 = off */
  float attn_factor;
  float beta_fast;
  float beta_slow;
  const larch_tensor* freq_factors; /* NULL, or f32 [n_dims / 2] dividing each frequency (Llama 3 rope_freqs) */
} larch_rope_params;

/* How larch_rope_multi gives each rotated pair its position (ggml_rope_multi's modes). */
typedef enum {
  LARCH_ROPE_MROPE = 0,  /* Qwen2-VL: contiguous sections t, h, w, e */
  LARCH_ROPE_IMROPE = 1, /* Qwen3-VL and Qwen 3.5 text: the sections interleaved */
  LARCH_ROPE_VISION = 2  /* Qwen-VL encoders: contiguous, each section's frequency starting over */
} larch_rope_mode;

/* The parameters of ggml_rope_multi: larch_rope_params (its neox ignored) and four sections. */
typedef struct {
  larch_rope_params rope;
  uint32_t sections[4]; /* pairs per position (t, h, w, e); s0 + s1 + s2 > 0 */
  uint32_t mode;        /* a larch_rope_mode */
} larch_rope_multi_params;

/* Per row of larch_sample_prep. */
typedef struct {
  /* Dead: no output of sample_prep depends on it (the top-64 order and the T = 1 LSE are the same at
   * every temperature; larch_sample and the host sampler read larch_draw_params'). ABI 3 keeps it:
   * dropping it moves every field that Metal mirrors and every caller fills, for 4 bytes. */
  float temperature;
  uint32_t top_k;          /* <= 64; 0 = 64 */
  float repeat_penalty;    /* > 0; 1 = off */
  uint32_t penalty_window; /* ids used from the row of penalty_ids_i32, <= its width */
} larch_sample_params;

/* Per row of larch_sample: the host sampler's settings (architecture 9.6). */
typedef struct {
  float temperature; /* <= 0: greedy */
  float top_p;       /* in (0, 1]; 1 = off */
  uint32_t top_k;    /* <= 64; 0 = every candidate */
  uint32_t pad;      /* 0 */
  uint64_t seed;
} larch_draw_params;

/* The recurrent state of one Gated DeltaNet layer group (Qwen 3.5's linear attention), in n_slots
 * slots per layer. A slot holds a sequence's live state or a snapshot; restoring a snapshot is
 * pointing the sequence at its slot. channels = (2 k_heads + v_heads) head_dim.
 * conv: f16 [n_layers][n_slots][kernel - 1][channels], the last kernel - 1 conv inputs, oldest first.
 * ssm: f32 [n_layers][n_slots][v_heads][head_dim][head_dim], the delta-rule matrix M of each v head,
 * row j giving output j: out = M q / sqrt(head_dim) (llama.cpp's state, transposed as it stores it).
 * Neither buffer is NULL, every count is > 0, kernel >= 2, v_heads a multiple of k_heads and
 * layer < n_layers. */
typedef struct {
  larch_buffer* conv;
  larch_buffer* ssm;
  uint32_t n_slots, n_layers, layer;
  uint32_t kernel, k_heads, v_heads, head_dim;
} larch_gdn_state;

uint32_t larch_abi_version(void);
larch_status larch_backend_info(larch_backend* out, uint32_t* runtime_version, uint32_t* driver_version);
larch_status larch_device_count(int32_t* out);
larch_status larch_device_open(int32_t ordinal, larch_device** out); /* -1 = CPU reference, -2 = CPU */
larch_status larch_device_props_get(const larch_device*, larch_device_props* out);
void larch_device_close(larch_device*);
/* The threads a CPU device's ops may use; 0, the default, is one per logical CPU. The reference (-1)
 * keeps the setting and stays single-threaded. GPU devices: UNSUPPORTED. */
larch_status larch_device_set_threads(larch_device*, uint32_t threads);
/* The calling thread's latest error detail, for logs; the device may be NULL (e.g. after a failed open) */
const char* larch_last_error(const larch_device*);

larch_status larch_buffer_alloc(larch_device*, uint64_t bytes, larch_buffer** out);
/* Metal: zero-copy mmap. CUDA: a range inside page-locked host memory (larch_buffer_alloc_host), which
 * the device's ops read in place over the bus, until larch_buffer_free of the wrap, which must come
 * before that host buffer's; any other memory is UNSUPPORTED there. Others: UNSUPPORTED */
larch_status larch_buffer_wrap_host(larch_device*, void* host, uint64_t bytes, larch_buffer** out);
/* Host memory at *host, until larch_buffer_free, that the device copies to and from at full speed
 * and asynchronously (page-locked on a GPU, plain memory on a CPU device). Every CPU device's ops take
 * it as a tensor buffer, as they take their own; a GPU's ops refuse it, and its copies run through
 * larch_upload and larch_download from *host. */
larch_status larch_buffer_alloc_host(larch_device*, uint64_t bytes, larch_buffer** out, void** host);
void larch_buffer_free(larch_buffer*);
larch_status larch_upload(larch_stream*, larch_buffer*, uint64_t off, const void* src, uint64_t bytes);
larch_status larch_download(larch_stream*, const larch_buffer*, uint64_t off, void* dst, uint64_t bytes);
larch_status larch_stream_create(larch_device*, larch_stream** out);
larch_status larch_stream_sync(larch_stream*); /* returns DEVICE_LOST / TIMEOUT on fault */
void larch_stream_destroy(larch_stream*);
larch_status larch_event_record(larch_stream*, larch_event** out);
larch_status larch_event_elapsed_ns(const larch_event* a, const larch_event* b, uint64_t* out);
void larch_event_destroy(larch_event*);
/* Blocks the host until the work queued before the event is done; DEVICE_LOST or TIMEOUT on a fault,
 * as larch_stream_sync. A CPU device's event is done when it is recorded. */
larch_status larch_event_sync(const larch_event*);
/* Work queued on the stream after this call starts once the event's work is done. The event may come
 * from another stream or device; another backend's is waited for on the host here (larch_event_sync),
 * which still orders the stream's later work, and a backend without a wait of its own answers its own
 * events LARCH_ERR_UNSUPPORTED. */
larch_status larch_stream_wait(larch_stream*, const larch_event*);

/* graphs (CUDA graphs, HIP graphs, Metal indirect command buffers); UNSUPPORTED is a valid answer */
larch_status larch_graph_begin(larch_stream*);
larch_status larch_graph_end(larch_stream*, larch_graph** out);
larch_status larch_graph_launch(larch_graph*, larch_stream*);
void larch_graph_destroy(larch_graph*);

/* ops. Their contract is architecture 6.1 "Activation dtypes": f16 activations, an f32 residual
 * stream, f16 KV (8-bit pages where larch_kv_pool says). On every backend:
 * - Shapes are [rows, cols] row-major unless noted; batch = rows. Every dim after the first is dense
 *   (stride = the product of the later dims; for quant the last dim counts blocks) and stride[0] is
 *   at least that product, so rows never overlap. Quant tensors are 2-D with whole blocks per row.
 * - Any dtype, ndim, shape or host parameter outside what an op lists is LARCH_ERR_INVALID_ARG (a
 *   NULL tensor, buffer, pool or params where one is required, a buffer of another backend, a params
 *   field out of its range), and so is a tensor or pool that reaches past its buffer. The checks run on the host, on metadata only, before
 *   anything is enqueued, so a rejected call writes nothing.
 * - Values in device memory (ids, positions, slots, block tables, lengths, logits) are not part of
 *   those checks: one out of range, or a NaN logit, is undefined behaviour, which the CPU reference
 *   reports as INVALID_ARG only as a debugging aid.
 * - Every dot product, norm statistic and softmax accumulates in f32, never f16 (the CPU reference
 *   may use double), but for larch_mmvq and larch_mmvq_verify on CUDA sm_86/89 (see larch_mmvq).
 * - An output overlaps no input unless its op says so. An overlap is undefined behaviour: it is not
 *   checked. */

/* out_f32[r] = table[ids_i32[r]], dequantized: the residual stream entering layer 0.
 * table: [vocab, d] as the GGUF stores it, any quant dtype, f16 or f32. ids_i32: i32 [rows].
 * out_f32: f32 [rows, d]. */
larch_status larch_embed(larch_stream*, const larch_tensor* table, const larch_tensor* ids_i32,
                           const larch_tensor* out_f32);
/* out = x / sqrt(mean(x^2) + eps) * w per row: the residual stream into a linear layer's input.
 * x: f32 [rows, d]. w: f32 [d], as stored. out: f16 [rows, d]. */
larch_status larch_rmsnorm(larch_stream*, const larch_tensor* x, const larch_tensor* w, float eps,
                             const larch_tensor* out);
/* out = (x - mean) / sqrt(var + eps) * w + b per row, mean and var (the mean of (x - mean)^2) over the
 * row, as ggml_norm, ggml_mul and ggml_add: a ViT's LayerNorm, the residual stream into a linear layer's
 * input. x: f32 [rows, d]. w: f32 [d], as stored. b_or_null: f32 [d] or NULL (no bias). out: f16
 * [rows, d]. A GPU backend may lack it and answer a valid call UNSUPPORTED. */
larch_status larch_layernorm(larch_stream*, const larch_tensor* x, const larch_tensor* w,
                               const larch_tensor* b_or_null, float eps, const larch_tensor* out);
/* ggml_rope_ext in place, so results match llama.cpp's on the same GGUF. qk: f16 [rows, heads,
 * head_dim], called once for q and once for k; row r sits at position pos_i32[r], i32 [rows]. The
 * first p->n_dims (even, 0 < n_dims <= head_dim) of each head rotate and the rest pass through. Pair
 * i (normal: elements 2i and 2i + 1; neox: i and i + n_dims / 2) turns by
 * pos * freq_base^(-2i / n_dims) / ff[i], which ggml's rope_yarn then scales with the rest of p.
 * ff = p->freq_factors (f32 [n_dims / 2]), or all 1 when it is NULL. */
larch_status larch_rope(larch_stream*, const larch_tensor* qk, const larch_tensor* pos_i32,
                          const larch_rope_params* p);
/* ggml_rope_multi in place (Qwen-VL's M-RoPE), so results match llama.cpp's on the same GGUF: as
 * larch_rope, but each rotated pair takes its angle from one of four positions of its row. qk: f16
 * [rows, heads, head_dim]. pos_i32: i32 [rows, 4], each row's positions t, h, w and e. Pair i
 * has sector j = i mod (s0 + s1 + s2 + s3) and takes position k:
 * - MROPE and VISION: t for j < s0, h for j < s0 + s1, w for j < s0 + s1 + s2, else e;
 * - IMROPE: h if j % 3 == 1 and j < 3 s1, w if j % 3 == 2 and j < 3 s2, t if j % 3 == 0 and j < 3 s0,
 *   else e.
 * Its angle is pos_k * freq_base^(-2 n / rope.n_dims) / ff[i], scaled by rope_yarn as larch_rope's,
 * with n = i, or under VISION the pairs since its section's first (j - the section's start).
 * MROPE and IMROPE rotate the first rope.n_dims dims as neox (pair i is i with i + n_dims / 2); VISION
 * rotates every dim, rope.n_dims being head_dim / 2 (pair i is i with i + n_dims). A GPU backend may
 * lack it and answer a valid call UNSUPPORTED. */
larch_status larch_rope_multi(larch_stream*, const larch_tensor* qk, const larch_tensor* pos_i32,
                                const larch_rope_multi_params* p);
/* The decode GEMV, its own op and not gemm with alpha 1 and beta 0: out[r, n] = sum_k w[n, k] x[r, k]
 * + bias[n], reading w once for all rows; out is written, never read.
 * w_quant: [n, k] in Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q4_0 or Q5_0. x_f16: f16 [rows, k] with rows <= 16 (more
 * rows go through dequant + gemm). bias_or_null: f32 [n] or NULL. out: f16 [rows, n]; f32 only for
 * the lm_head. On CUDA sm_86 and sm_89, one row, the rows routed to larch_mmvq_verify's kernel (2 to 8
 * rows on sm_89, 2 to 4 on sm_86, where the route takes the format) and larch_mmvq_verify may sum up
 * to 16 products of one 16-weight run in f16, x scaled per run into range by a power of two, then
 * accumulate in f32, so a row's bytes never depend on its batch; LARCH_DECODE_H2=0 keeps f32
 * throughout. */
larch_status larch_mmvq(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                          const larch_tensor* bias_or_null, const larch_tensor* out);
/* The verify GEMV of exact speculation: mmvq's shapes and dtypes with up to 32 rows, each row
 * independent of the others: row r of out is bit-identical to larch_mmvq of x's row r alone on the
 * same backend, so verifying draft tokens gives the target's own one-row logits. On Metal GPU family
 * 10, mmvq_verify rounds weights to f16 within tiles, as quant_gemm does, at every row count, and row r
 * is bit-identical to mmvq_verify of x's row r alone instead, which a guessing engine also runs its
 * one-row decode through. */
larch_status larch_mmvq_verify(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                                 const larch_tensor* bias_or_null, const larch_tensor* out);
/* MoE expert GEMV. Row r's slot j uses expert e = ids_i32[r, j] (i32 [rows, slots], each in
 * [0, experts)): y[r, j, n] = sum_k w[e * N + n, k] x[r, k] + bias[e * N + n], where x_f16 is
 * f16 [rows, k] (every slot reads row r: gate and up) or [rows, slots, k] (slot j reads x[r, j]: down).
 * w_quant: the GGUF's [experts, N, k] expert tensor seen as [experts * N, k], in mmvq's dtypes.
 * bias_or_null: f32 [experts * N] or NULL. Without weights, out is y, f16 or f32 [rows, slots, N].
 * With weights_f32 (f32 [rows, slots]), out is f16 or f32 [rows, N]: the sum over j ascending of
 * weights[r, j] y[r, j], rounded once (the down projection and the experts' combine in one call). */
larch_status larch_mmvq_id(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                             const larch_tensor* ids_i32, const larch_tensor* bias_or_null,
                             const larch_tensor* weights_or_null, const larch_tensor* out);
/* ABI 5: larch_mmvq_id over a mixed layer, whose experts' rows lie in two tensors of their own dtypes, the hot
 * experts' w_hot ([hot * N, k]) and the cold ones' w_cold ([cold * N, k]), each in mmvq's dtypes. table_i32: i32
 * [experts]; expert e is slot s = table[e] of w_hot (rows s N to s N + N - 1) when s >= 0, else slot -1 - s of w_cold.
 * x_f16, ids_i32 (each in [0, experts)), weights_or_null and out as larch_mmvq_id's, without a bias. Table values,
 * as ids, are not checked. A (row, slot) pair reads only its expert's rows. On CUDA and the CPU reference its result
 * is larch_mmvq_id's on one tensor holding every expert's rows at its dtype, bit for bit, so a layer's hot experts
 * give the bytes they gave in the whole tensor. A GPU backend may lack it and answer a valid call UNSUPPORTED; CUDA
 * takes Q2_K cold experts only. */
larch_status larch_mmvq_id_mixed(larch_stream*, const larch_tensor* w_hot, const larch_tensor* w_cold,
                                   const larch_tensor* table_i32, const larch_tensor* x_f16,
                                   const larch_tensor* ids_i32, const larch_tensor* weights_or_null,
                                   const larch_tensor* out);
/* Tiled quantized matrix multiply: out = x_f16 * w_quant^T + bias, for any positive row count.
 * Shapes and output dtypes match mmvq. Weights are rounded to f16 within tiles, matching dequant
 * followed by gemm, with f32 accumulation. Returns UNSUPPORTED when the backend or shape has no
 * direct path; callers retain their existing mmvq or dequant + gemm fallback. */
larch_status larch_quant_gemm(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                                const larch_tensor* bias_or_null, const larch_tensor* out);
/* Explicit approximate quantized GEMM, separate from the f16-rounding contract above.
 * Shapes/output dtypes match quant_gemm, except Q2_K and Q3_K. Q4_K, Q5_K, Q6_K, Q8_0, Q4_0 and Q5_0 weights retain their
 * original scales and minima (Q6_K: a scale per 16 weights).
 * Each contiguous group of 32 f16 activations is quantized to signed i8 with scale max(abs(x))/127
 * and nearest-even rounding. Weight-minimum correction uses the sum of the original activations;
 * scaled integer dot products accumulate in f32. A zero activation group has zero scale and codes.
 * A group containing any nonfinite activation has NaN scale/sum metadata and zero codes; its
 * sequence row produces NaN outputs without affecting other rows.
 * CPU provides the numerical reference. Unsupported backends/shapes return UNSUPPORTED before
 * enqueueing work or writing output; callers retain a reference fallback. */
larch_status larch_quant_gemm_i8(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                                   const larch_tensor* bias_or_null, const larch_tensor* out);
/* CUDA-only bounded split-K variant of quant_gemm_i8, with the same group32 numerical policy.
 * Requires an underfilled output grid, sufficient K work and two resident producer blocks per SM.
 * Ordered f32 partial sums may reassociate accumulation. Combined packing/partial scratch is
 * limited to 32 MiB. Ineligible resources/shapes return UNSUPPORTED before enqueueing work or
 * writing output; callers retain their existing projection route. CPU quant_gemm_i8 is its reference. */
larch_status larch_quant_gemm_i8_split(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                                         const larch_tensor* bias_or_null, const larch_tensor* out);
/* out_f16 = w_quant dequantized, the scratch gemm reads. w_quant: [n, k], any quant dtype.
 * out_f16: f16 [n, k]. */
larch_status larch_dequant(larch_stream*, const larch_tensor* w_quant, const larch_tensor* out_f16);
/* c = alpha * a b_t^T + beta * c + bias, as BLAS with the cuBLASLt bias epilogue (bias is not scaled);
 * c is not read when beta == 0. cuBLAS/hipBLAS/MPS or own. a_f16: f16 [m, k]. b_f16_t: f16 [n, k],
 * the weight, transposed. bias_or_null: f32 [n] or NULL. c: f16 [m, n]; f32 for the lm_head, and for an
 * f32 residual the product accumulates into with beta = 1 (a vision encoder's). */
larch_status larch_gemm(larch_stream*, const larch_tensor* a_f16, const larch_tensor* b_f16_t,
                          const larch_tensor* bias_or_null, const larch_tensor* c, float alpha, float beta);
/* out = silu(gate) * up, silu(x) = x / (1 + e^-x). gate, up, out: f16 [rows, ff]. */
larch_status larch_silu_mul(larch_stream*, const larch_tensor* gate, const larch_tensor* up,
                              const larch_tensor* out);
/* out = x * sigmoid(gate), sigmoid(g) = 1 / (1 + e^-g): Qwen 3.5's attention output gate (gate
 * f16 [rows, d]) and shared-expert gate (gate f16 [rows, 1], one value per row). x, out: f16 [rows, d];
 * out may be x itself (same buffer, offset and strides), no other overlap. */
larch_status larch_sigmoid_mul(larch_stream*, const larch_tensor* x, const larch_tensor* gate,
                                 const larch_tensor* out);
/* out = 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))) in f32, ggml_gelu's tanh form: a ViT's MLP
 * activation. x, out: f16 [rows, d]; out may be x itself (same buffer, offset and strides), no other
 * overlap. A GPU backend may lack it and answer a valid call UNSUPPORTED. */
larch_status larch_gelu(larch_stream*, const larch_tensor* x, const larch_tensor* out);
/* The residual update out = a_f32 + b_f16, no broadcast. a_f32: f32 [rows, d], the residual stream.
 * b_f16: f16 [rows, d]. out: f32 [rows, d]; out may be a_f32 itself (same buffer, offset and
 * strides), no other overlap. */
larch_status larch_add(larch_stream*, const larch_tensor* a_f32, const larch_tensor* b_f16,
                         const larch_tensor* out);
/* Stores row r of k and v at slot slots_i32[r] of the pool's layer (layout at larch_kv_pool); a
 * negative slot skips the row (bucket padding); the other slots are distinct. k, v: f16 [rows,
 * n_kv_heads, head_dim]. slots_i32: i32 [rows]. */
larch_status larch_kv_write(larch_stream*, const larch_tensor* k, const larch_tensor* v,
                              const larch_tensor* slots_i32, const larch_kv_pool* pool);
/* Attention, all three ops: per head, softmax(scale * q k^T) v over the pool's layer. q: f16 [rows,
 * n_heads, head_dim], head_dim the pool's, n_heads a multiple of n_kv_heads; q head h reads kv head
 * h / (n_heads / n_kv_heads) (GQA). out: f16, q's shape. Positions are read from the pool, so
 * kv_write for this step runs first. larch_attn is the general entry; decode and prefill are its
 * special cases, kept as their own entries.
 *
 * Decode: one query row per sequence. Row r reads row r of block_table_i32 (i32 [rows, max_blocks])
 * and attends to positions [0, seq_lens_i32[r]) (i32 [rows]), its own token included; seq_lens 0
 * marks a padding row, whose output is zero. splits (0 = auto) is a speed hint: it changes the result
 * only by f32 rounding. */
larch_status larch_attn_decode(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool,
                                 const larch_tensor* block_table_i32, const larch_tensor* seq_lens_i32,
                                 uint32_t splits /* 0 = auto */, float scale, const larch_tensor* out);
/* Prefill, causal and chunk-aware: q holds q_lens_i32[s] rows per sequence s in sequence order
 * (i32 [seqs], summing to rows). Row j of sequence s sits at position q_start_pos_i32[s] + j
 * (i32 [seqs]) and attends to positions [0, that position] through row s of block_table_i32
 * (i32 [seqs, max_blocks]); earlier positions come from cached or earlier chunks. */
larch_status larch_attn_prefill(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool,
                                  const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                                  const larch_tensor* q_lens_i32, float scale, const larch_tensor* out);
/* The general entry: a span of q_lens_i32[s] query rows per sequence s, laid out as prefill's. Row j
 * of sequence s sits at position p = q_start_pos_i32[s] + j and attends to every position before
 * q_start_pos_i32[s], then to span positions q_start_pos_i32[s] + i: those with i <= j (causal) when
 * mask_or_null is NULL, else those whose bit i is set in the row's entry of mask_or_null (i32 [rows],
 * read as unsigned; every q_lens <= 32). A draft tree is a mask whose row keeps its ancestors and
 * itself. window > 0 keeps only positions p' with p - p' < window (sliding-window layers).
 * sinks_or_null: f32 [n_heads], one more logit per head in each softmax, with no value (attention
 * sinks). splits as decode's.
 * The verify contract: without sinks and window, each row is bit-identical to larch_attn_decode, at
 * the same nonzero splits, of that row alone over a sequence whose positions hold, in order, the K and
 * V of the positions the row attends to; so a chain's row j equals decode with seq_len
 * q_start_pos + j + 1, and a tree's rows equal decode along each branch. */
larch_status larch_attn(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool,
                          const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                          const larch_tensor* q_lens_i32, const larch_tensor* mask_or_null,
                          const larch_tensor* sinks_or_null, uint32_t window, uint32_t splits, float scale,
                          const larch_tensor* out);
/* Non-causal attention over dense rows, as a ViT's: rows is a whole number of segments of seg_len rows
 * (one image tile each), and in each segment, per head, out = softmax(scale q k^T) v over the segment's
 * rows, f32 softmax and sums. q, k, v, out: f16 [rows, heads, head_dim], one kv head per q head; out
 * overlaps none of them. A GPU backend may lack it, a head_dim, or a layout (CUDA: each tensor 4-byte
 * aligned with an even row stride), and answer a valid call UNSUPPORTED. */
larch_status larch_attn_full(larch_stream*, const larch_tensor* q, const larch_tensor* k, const larch_tensor* v,
                               uint32_t seg_len, float scale, const larch_tensor* out);
/* Qwen3: rmsnorm over head_dim per head, in place: each head of each row is normalised on its own,
 * then times w. qk: f16 [rows, heads, head_dim]. w: f32 [head_dim]. */
larch_status larch_qk_norm(larch_stream*, const larch_tensor* qk, const larch_tensor* w, float eps);
/* Per row r of logits (f32 [rows, vocab]), with per_row[r] (host memory, rows entries):
 * 1. The repetition penalty, llama.cpp's: once per distinct id among the first penalty_window ids of
 *    row r of penalty_ids_i32 (i32 [rows, width], or NULL for none; negative ids are padding), a
 *    logit <= 0 is multiplied by repeat_penalty and a positive one divided.
 * 2. out_lse_f32[r] (f32 [rows]): the log-sum-exp of the penalised row over the whole vocabulary.
 * 3. out_topk_vals_f32, out_topk_ids_i32 (f32, i32 [rows, 64]): the k = min(top_k ? top_k : 64, vocab)
 *    largest penalised logits and their ids, largest first, ties to the lower id; slots from k on
 *    hold -inf and id -1. */
larch_status larch_sample_prep(larch_stream*, const larch_tensor* logits, const larch_sample_params* per_row,
                                 const larch_tensor* penalty_ids_i32, const larch_tensor* out_topk_vals_f32,
                                 const larch_tensor* out_topk_ids_i32, const larch_tensor* out_lse_f32);
/* Draws one token per row on the device with the host sampler's math, so only the ids (and logprobs)
 * come back. Inputs are sample_prep's outputs: topk_vals_f32, topk_ids_i32 (f32, i32 [rows, 64]) and
 * lse_f32 (f32 [rows]); per_row[r] (host memory, rows entries) is row r's settings. Candidates are row
 * r's 64 in order, less any whose id is among the row's ban_ids_or_null (i32 [rows, width] or NULL;
 * negative entries are padding).
 * - temperature <= 0: the first candidate whose id is >= 0.
 * - Else, with v_i the candidates' values, n0 the number of leading ids >= 0 (at least 1), k = top_k ?
 *   min(top_k, n0) : n0, l_i = (double)(v_i - v_0) (an f32 difference) and T = sum over i < k of e^l_i:
 *   n is the number of leading i < k with m_i < top_p, at least 1, where m_0 = 0 and m_(i+1) = m_i +
 *   e^l_i / T; w_i = e^(l_i / temperature); u = (z >> 11) * 2^-53 * (sum over i < n of w_i), in that
 *   order; the pick is the first i < n whose running sum of w reaches past u, else n - 1. z is the
 *   row's random word, a function of its seed and position alone, so a verify step redraws the same
 *   tokens: splitmix64 at index pos + 1, z = seed + (pos + 1) * 0x9E3779B97F4A7C15, then
 *   z = (z ^ z >> 30) * 0xBF58476D1CE4E5B9, z = (z ^ z >> 27) * 0x94D049BB133111EB, z ^= z >> 31
 *   (uint64_t arithmetic), with pos = pos_i32[r] (i32 [rows], >= 0). Sums run in f64, in index order.
 * out_ids_i32 (i32 [rows]) gets the pick's id and out_logprobs_f32 (f32 [rows]) its value minus lse;
 * a row with no candidate left gets id -1 and -inf. */
larch_status larch_sample(larch_stream*, const larch_tensor* topk_vals_f32, const larch_tensor* topk_ids_i32,
                            const larch_tensor* lse_f32, const larch_draw_params* per_row,
                            const larch_tensor* pos_i32, const larch_tensor* ban_ids_or_null,
                            const larch_tensor* out_ids_i32, const larch_tensor* out_logprobs_f32);

/* MoE routing, llama.cpp's softmax gating: per row of logits_f32 (f32 [rows, experts]), the k largest
 * logits (ties to the lower id) go to out_ids_i32 (i32 [rows, k], 0 < k <= experts) largest first,
 * and out_weights_f32 (f32 [rows, k]) gets each one's softmax probability over all experts, divided by
 * the k's sum (clamped to at least 2^-14, as llama.cpp does), times scale (0 means 1). */
larch_status larch_moe_route(larch_stream*, const larch_tensor* logits_f32, float scale,
                               const larch_tensor* out_ids_i32, const larch_tensor* out_weights_f32);

/* Training: the router-delta trainer's forward and backward (doc I4), every tensor f32. ABI 3 entries
 * that the CPU devices and CUDA have; HIP and Metal answer UNSUPPORTED.
 * c = alpha op(a) op(b) + beta c, op(x) = x^T when its flag is nonzero, else x: op(a) [m, k], op(b)
 * [k, n], c [m, n]; c is not read when beta == 0. */
larch_status larch_gemm_ex(larch_stream*, const larch_tensor* a, const larch_tensor* b, const larch_tensor* c,
                             uint32_t trans_a, uint32_t trans_b, float alpha, float beta);
/* larch_rmsnorm with out f32 [rows, d]. */
larch_status larch_rmsnorm_f32(larch_stream*, const larch_tensor* x, const larch_tensor* w, float eps,
                                 const larch_tensor* out);
/* The gradient of larch_rmsnorm_f32 at x, onto dx (accumulate != 0) or into it: per row, with
 * s = 1 / sqrt(mean(x^2) + eps) and c = s^3 / d * sum_i dy_i w_i x_i, dx_i += s w_i dy_i - x_i c.
 * x, dy, dx: f32 [rows, d]. w: f32 [d]. */
larch_status larch_rmsnorm_backward(larch_stream*, const larch_tensor* x, const larch_tensor* w,
                                      const larch_tensor* dy, float eps, const larch_tensor* dx,
                                      uint32_t accumulate);
/* larch_rope with qk f32. Positions may be negative: rope at -p is the inverse (the transpose) of rope
 * at p, which is its backward pass. */
larch_status larch_rope_f32(larch_stream*, const larch_tensor* qk, const larch_tensor* pos_i32,
                              const larch_rope_params* p);
/* larch_silu_mul with gate, up and out f32. */
larch_status larch_silu_mul_f32(larch_stream*, const larch_tensor* gate, const larch_tensor* up,
                                  const larch_tensor* out);
/* The gradient of silu_mul: dgate = dout up sig(g) (1 + g (1 - sig(g))), dup = dout silu(g), with g the
 * gate and sig(g) = 1 / (1 + e^-g). All f32 [rows, ff]. */
larch_status larch_silu_mul_backward(larch_stream*, const larch_tensor* gate, const larch_tensor* up,
                                       const larch_tensor* dout, const larch_tensor* dgate,
                                       const larch_tensor* dup);
/* out[r] = softmax(scale x[r]) per row of f32 [rows, n]; causal != 0 keeps only the columns
 * c <= r mod n of row r and writes 0 to the rest (each head's [T, T] scores, heads stacked as rows).
 * out may be x itself (same buffer, offset and strides), no other overlap. */
larch_status larch_softmax(larch_stream*, const larch_tensor* x, float scale, uint32_t causal,
                             const larch_tensor* out);
/* The gradient through out = softmax(scale x) per row: ds = scale p (dp - sum_c p dp), with p that out.
 * p, dp, ds: f32 [rows, n]; ds may be dp itself, no other overlap. */
larch_status larch_softmax_backward(larch_stream*, const larch_tensor* p, const larch_tensor* dp, float scale,
                                      const larch_tensor* ds);

/* Gated DeltaNet (Qwen 3.5's linear attention), llama.cpp's math, in two ops over one layer's
 * larch_gdn_state. Both take q_lens_i32[s] rows per sequence s in sequence order (i32 [seqs], summing
 * to rows) and sequence s's live slot slots_i32[s] (i32 [seqs]): its state before the first row, and
 * after the last one. A row r whose snap_slots_or_null[r] (i32 [rows], or NULL for none) is >= 0 also
 * leaves the state as of that row in that slot: the snapshots a draft's rollback or a block boundary
 * needs. Every slot named in one call is distinct.
 *
 * Step 1, the short causal conv and SiLU. x: f16 [rows, channels], the in-projection's q, k and v
 * channels. Per channel c, the conv inputs are the slot's conv state followed by the sequence's rows,
 * and out[r, c] = silu(sum over i < kernel of w[c, i] in[c][t + i]), where row r is the sequence's
 * row t; f16 [rows, channels]. w: f32 [channels, kernel], the GGUF's conv1d weight. */
larch_status larch_gdn_conv(larch_stream*, const larch_tensor* x, const larch_tensor* w,
                              const larch_gdn_state* state, const larch_tensor* slots_i32,
                              const larch_tensor* q_lens_i32, const larch_tensor* snap_slots_or_null,
                              const larch_tensor* out);
/* Step 2, the delta rule. qkv: f16 [rows, channels], gdn_conv's output: q (k_heads x head_dim), then
 * k (the same), then v (v_heads x head_dim). For each row and v head h, which reads q and k head
 * h % k_heads (llama.cpp's tiling): q and k are scaled by 1 / sqrt(their sum of squares + eps);
 * g = softplus(alpha[r, h] + dt_bias[h]) * a[h], softplus(x) = x > 20 ? x : log(1 + e^x);
 * b = sigmoid(beta[r, h]); then M = e^g M, d = b (v - M k), M = M + d k^T and
 * out[r, h] = M q / sqrt(head_dim), f16 [rows, v_heads * head_dim]. alpha, beta: f32 [rows, v_heads],
 * the in-projection's. dt_bias, a: f32 [v_heads] as the GGUF stores them (a = -exp(A_log)). */
larch_status larch_gdn(larch_stream*, const larch_tensor* qkv, const larch_tensor* alpha,
                         const larch_tensor* beta, const larch_tensor* dt_bias, const larch_tensor* a, float eps,
                         const larch_gdn_state* state, const larch_tensor* slots_i32,
                         const larch_tensor* q_lens_i32, const larch_tensor* snap_slots_or_null,
                         const larch_tensor* out);

/* Fused layer entries (ABI 4): each is a sequence of the ops above in one call, for one- and few-row decode.
 * Each takes exactly the arguments of its sequence's ops, under their checks, and gives the bytes of that
 * sequence on the same device, except where an entry names a gemm it replaces: there its own f32 product
 * stands in for the backend's gemm, and every later step reads that product. The CPU devices run the
 * sequence itself. A backend may lack an entry, a weight dtype or a shape, and answer a valid call
 * UNSUPPORTED before it writes anything; the caller then runs the sequence. Scratch the sequence writes
 * and nothing after it reads (the gate and up halves, a projection before its residual add) is not
 * written. */

/* larch_mmvq_id(gate_w, x_f16, ids_i32, NULL, NULL, g), the same with up_w into u, then
 * larch_silu_mul(g, u, out): the routed experts' activation. out: f16 [rows, slots, N]. */
larch_status larch_fused_moe_gate_up(larch_stream*, const larch_tensor* gate_w, const larch_tensor* up_w,
                                       const larch_tensor* x_f16, const larch_tensor* ids_i32,
                                       const larch_tensor* out);
/* The down projections into the f32 residual x_f32 [rows, d], in place. With shared_w (all three shared
 * arguments or none): larch_mmvq(shared_w, shared_act_f16, NULL, t), larch_sigmoid_mul(t, shared_gate_f16,
 * t) (shared_gate_f16: f16 [rows, 1]) and larch_add(x, t, x); then larch_mmvq_id(down_w, act_f16, ids_i32,
 * NULL, weights_f32, t) and larch_add(x, t, x), t f16 [rows, d]. */
larch_status larch_fused_moe_down(larch_stream*, const larch_tensor* down_w, const larch_tensor* act_f16,
                                    const larch_tensor* ids_i32, const larch_tensor* weights_f32,
                                    const larch_tensor* shared_w_or_null, const larch_tensor* shared_act_or_null,
                                    const larch_tensor* shared_gate_or_null, const larch_tensor* x_f32);
/* The MoE block's inputs: larch_rmsnorm(x_f32, norm_w, eps, h_f16); the router's logits larch_gemm(h, router_f16,
 * NULL, logits_f32, 1, 0) and larch_moe_route(logits_f32, 0, ids_i32, weights_f32); with shared_gate_w (the five
 * shared arguments, or none): larch_mmvq(shared_gate_w, h, NULL, g), larch_mmvq(shared_up_w, h, NULL, u),
 * larch_silu_mul(g, u, shared_act_f16) and the shared expert's gate input larch_gemm(h, gate_inp_f16, NULL,
 * shared_gate_f16, 1, 0) (gate_inp_f16: f16 [1, d]; shared_gate_f16: f16 [rows, 1]). Both gemms are replaced. */
larch_status larch_fused_moe_in(larch_stream*, const larch_tensor* x_f32, const larch_tensor* norm_w, float eps,
                                  const larch_tensor* h_f16, const larch_tensor* router_f16,
                                  const larch_tensor* logits_f32, const larch_tensor* ids_i32,
                                  const larch_tensor* weights_f32, const larch_tensor* shared_gate_w_or_null,
                                  const larch_tensor* shared_up_w_or_null, const larch_tensor* shared_act_or_null,
                                  const larch_tensor* gate_inp_or_null, const larch_tensor* shared_gate_or_null);
/* A Gated DeltaNet layer's inputs: larch_rmsnorm(x_f32, norm_w, eps, h_f16), then h's projections by qkv_w into
 * qkv_f16, by z_w into z_f16, and by alpha_w and beta_w into alpha_f32 and beta_f32 (f32 [rows, v_heads]): each
 * larch_mmvq(w, h, NULL, out) for a quant w, larch_gemm(h, w, NULL, out, 1, 0) for an f16 one, which is replaced. */
larch_status larch_fused_gdn_in(larch_stream*, const larch_tensor* x_f32, const larch_tensor* norm_w, float eps,
                                  const larch_tensor* h_f16, const larch_tensor* qkv_w, const larch_tensor* z_w,
                                  const larch_tensor* alpha_w, const larch_tensor* beta_w,
                                  const larch_tensor* qkv_f16, const larch_tensor* z_f16,
                                  const larch_tensor* alpha_f32, const larch_tensor* beta_f32);
/* A projection onto the f32 residual x_f32 [rows, n], in place: with gate_f16 (f16 [rows, k] or [rows, 1]),
 * larch_sigmoid_mul(in_f16, gate_f16, g) first (g f16 [rows, k]), else g is in_f16; then larch_mmvq(w_quant, g,
 * NULL, t) and larch_add(x_f32, t, x_f32), t f16 [rows, n]. A layer's output projection: the state-space one, and
 * the attention one behind its gate (entries 6 and 8 of the fused layer). */
larch_status larch_fused_proj_add(larch_stream*, const larch_tensor* w_quant, const larch_tensor* in_f16,
                                    const larch_tensor* gate_or_null, const larch_tensor* x_f32);
/* A Gated DeltaNet layer's recurrence: larch_gdn_conv(qkv_f16, conv_w, state, slots_i32, q_lens_i32, snaps, conv_f16),
 * larch_gdn(conv_f16, alpha_f32, beta_f32, dt_bias, a, eps, state, slots_i32, q_lens_i32, snaps, o), then llama.cpp's
 * gated norm: larch_qk_norm(o as f16 [rows, v_heads, head_dim], norm_w, eps), its per-head reduction as on an o
 * whose rows are 16-byte aligned, and larch_silu_mul(z_f16, o, out_f16), o f16 [rows, v_heads head_dim]. conv_f16
 * is written; o is not. */
larch_status larch_fused_gdn(larch_stream*, const larch_tensor* qkv_f16, const larch_tensor* conv_w,
                               const larch_tensor* alpha_f32, const larch_tensor* beta_f32,
                               const larch_tensor* dt_bias, const larch_tensor* a, float eps,
                               const larch_gdn_state* state, const larch_tensor* slots_i32,
                               const larch_tensor* q_lens_i32, const larch_tensor* snap_slots_or_null,
                               const larch_tensor* norm_w, const larch_tensor* z_f16, const larch_tensor* conv_f16,
                               const larch_tensor* out_f16);
/* Attention's inputs after its projections. With gate_f16, q_in_f16 holds Qwen 3.5's [q, gate] pair per head (f16
 * [rows, heads, 2 head_dim]) and each half is copied out, into q_f16 and gate_f16 ([rows, heads, head_dim]), as the
 * gemms by 0/1 selector matrices copy them, but for the sign of a zero and a non-finite element (the gemm turns an inf
 * or NaN into NaN across its half of the head, as inf * 0 is NaN; the copy keeps it in place); without it q_f16 is
 * q_in_f16 itself. Then
 * larch_qk_norm(q_f16, q_norm_w, eps) and of k by k_norm_w (both or neither), larch_rope(q_f16, pos_i32, p) and
 * of k, and larch_kv_write(k, v_f16, slots_i32, pool), with k_f16 left as it was: its normed and roped rows go to
 * the pool only. q and k heads are one head_dim wide. */
larch_status larch_fused_attn_in(larch_stream*, const larch_tensor* q_in_f16, const larch_tensor* k_f16,
                                   const larch_tensor* v_f16, const larch_tensor* q_norm_or_null,
                                   const larch_tensor* k_norm_or_null, float eps, const larch_tensor* pos_i32,
                                   const larch_rope_params* p, const larch_tensor* slots_i32,
                                   const larch_kv_pool* pool, const larch_tensor* q_f16,
                                   const larch_tensor* gate_or_null);

/* DFlash2's dynamic conv over one block's rows, stage 0 or 1, in f32 in this order: o = out (f32) or 0 (f16),
 * o += x[r, c] (base[2 stage, c] + dyn[r, 2 stage, g]), then for r > 0 o += x[r - 1, c] (base[2 stage + 1, c] +
 * dyn[r, 2 stage + 1, g]), g = c / (d / groups); out[r, c] = o. x: f16 [rows, d]. dyn: f32 [rows, 4, groups],
 * groups dividing d. base: f32 [4, d]. out: f16 or f32 [rows, d]. A GPU backend may answer UNSUPPORTED. */
larch_status larch_draft_conv(larch_stream*, const larch_tensor* x_f16, const larch_tensor* dyn_f32,
                                const larch_tensor* base_f32, uint32_t stage, const larch_tensor* out);

#ifdef __cplusplus
}
#endif

#endif
