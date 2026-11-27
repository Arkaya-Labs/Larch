// How the public larch_* functions reach a backend. dispatch.cpp owns every larch_* symbol.
// Each backend fills one vv_ops table and derives its handle types from the structs below, setting
// `ops` to that table, so every handle carries the backend that created it.
#pragma once

#include <string>

#include "larch.h"

struct vv_ops;

struct larch_device {
  const vv_ops* ops;
};
struct larch_stream {
  const vv_ops* ops;
};
struct larch_buffer {
  const vv_ops* ops;
  uint64_t bytes = 0;  // set by dispatch.cpp at alloc and wrap, for checks a backend does not do itself
  uint8_t* host = nullptr;  // CPU memory and host buffers: where every CPU device's ops read and write it
};
struct larch_event {
  const vv_ops* ops;
};
struct larch_graph {
  const vv_ops* ops;
};

// One entry per larch_* function except larch_abi_version and larch_last_error, which
// dispatch.cpp answers itself, plus the debug op. Same signatures and semantics as larch.h.
struct vv_ops {
  larch_status (*backend_info)(larch_backend* out, uint32_t* runtime_version, uint32_t* driver_version);
  larch_status (*device_count)(int32_t* out);
  larch_status (*device_open)(int32_t ordinal, larch_device** out);
  larch_status (*device_props_get)(const larch_device*, larch_device_props* out);
  void (*device_close)(larch_device*);

  larch_status (*buffer_alloc)(larch_device*, uint64_t bytes, larch_buffer** out);
  larch_status (*buffer_wrap_host)(larch_device*, void* host, uint64_t bytes, larch_buffer** out);
  void (*buffer_free)(larch_buffer*);
  larch_status (*upload)(larch_stream*, larch_buffer*, uint64_t off, const void* src, uint64_t bytes);
  larch_status (*download)(larch_stream*, const larch_buffer*, uint64_t off, void* dst, uint64_t bytes);
  larch_status (*stream_create)(larch_device*, larch_stream** out);
  larch_status (*stream_sync)(larch_stream*);
  void (*stream_destroy)(larch_stream*);
  larch_status (*event_record)(larch_stream*, larch_event** out);
  larch_status (*event_elapsed_ns)(const larch_event* a, const larch_event* b, uint64_t* out);
  void (*event_destroy)(larch_event*);

  larch_status (*graph_begin)(larch_stream*);
  larch_status (*graph_end)(larch_stream*, larch_graph** out);
  larch_status (*graph_launch)(larch_graph*, larch_stream*);
  void (*graph_destroy)(larch_graph*);

  larch_status (*embed)(larch_stream*, const larch_tensor* table, const larch_tensor* ids_i32,
                         const larch_tensor* out_f32);
  larch_status (*rmsnorm)(larch_stream*, const larch_tensor* x, const larch_tensor* w, float eps,
                           const larch_tensor* out);
  larch_status (*rope)(larch_stream*, const larch_tensor* qk, const larch_tensor* pos_i32,
                        const larch_rope_params* p);
  larch_status (*mmvq)(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                        const larch_tensor* bias_or_null, const larch_tensor* out);
  larch_status (*dequant)(larch_stream*, const larch_tensor* w_quant, const larch_tensor* out_f16);
  larch_status (*gemm)(larch_stream*, const larch_tensor* a_f16, const larch_tensor* b_f16_t,
                        const larch_tensor* bias_or_null, const larch_tensor* c, float alpha, float beta);
  larch_status (*silu_mul)(larch_stream*, const larch_tensor* gate, const larch_tensor* up,
                            const larch_tensor* out);
  larch_status (*add)(larch_stream*, const larch_tensor* a_f32, const larch_tensor* b_f16,
                       const larch_tensor* out);
  larch_status (*kv_write)(larch_stream*, const larch_tensor* k, const larch_tensor* v,
                            const larch_tensor* slots_i32, const larch_kv_pool* pool);
  larch_status (*attn_decode)(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool,
                               const larch_tensor* block_table_i32, const larch_tensor* seq_lens_i32,
                               uint32_t splits, float scale, const larch_tensor* out);
  larch_status (*attn_prefill)(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool,
                                const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                                const larch_tensor* q_lens_i32, float scale, const larch_tensor* out);
  larch_status (*qk_norm)(larch_stream*, const larch_tensor* qk, const larch_tensor* w, float eps);
  larch_status (*sample_prep)(larch_stream*, const larch_tensor* logits, const larch_sample_params* per_row,
                               const larch_tensor* penalty_ids_i32, const larch_tensor* out_topk_vals_f32,
                               const larch_tensor* out_topk_ids_i32, const larch_tensor* out_lse_f32);

  // The one entry that is not a larch.h function: debug only (architecture 9.11), reached through
  // dispatch.cpp's vv_fault_device_lost. Queues a kernel that faults, so the next sync is a real
  // DEVICE_LOST and every later one too; INVALID_ARG while the stream captures. Null, the default, on a
  // backend without it (CPU, Metal, HIP): UNSUPPORTED. The default also keeps
  // -Wmissing-field-initializers quiet on those tables.
  larch_status (*fault_device_lost)(larch_stream*) = nullptr;
  // Optional direct quantized GEMM; a missing slot preserves the backend's existing linear path.
  larch_status (*quant_gemm)(larch_stream*, const larch_tensor*, const larch_tensor*,
                              const larch_tensor*, const larch_tensor*) = nullptr;
  larch_status (*quant_gemm_i8)(larch_stream*, const larch_tensor*, const larch_tensor*,
                                 const larch_tensor*, const larch_tensor*) = nullptr;
  // Optional CUDA split-K entry; CPU/Metal/HIP preserve their existing projection fallbacks.
  larch_status (*quant_gemm_i8_split)(larch_stream*, const larch_tensor*, const larch_tensor*,
                                       const larch_tensor*, const larch_tensor*) = nullptr;

  // ABI 3. A null entry answers UNSUPPORTED after the shared checks (dispatch.cpp), and the pool dtypes a
  // backend lacks the pages of are refused the same way before its kv_write and attention ops run.
  uint32_t kv_dtypes = 1u << LARCH_F16;
  // The weight dtypes this backend has no kernels for, refused the same way before its quant ops run.
  uint32_t no_weights = 1u << LARCH_Q3_K | 1u << LARCH_Q2_K | 1u << LARCH_NVFP4;
  larch_status (*device_set_threads)(larch_device*, uint32_t threads) = nullptr;
  larch_status (*buffer_alloc_host)(larch_device*, uint64_t bytes, larch_buffer** out, void** host) = nullptr;
  larch_status (*event_sync)(const larch_event*) = nullptr;
  // Waits for this table's own events only; dispatch.cpp waits on the host for another table's.
  larch_status (*stream_wait)(larch_stream*, const larch_event*) = nullptr;
  larch_status (*mmvq_verify)(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                               const larch_tensor* bias_or_null, const larch_tensor* out) = nullptr;
  larch_status (*mmvq_id)(larch_stream*, const larch_tensor* w_quant, const larch_tensor* x_f16,
                           const larch_tensor* ids_i32, const larch_tensor* bias_or_null,
                           const larch_tensor* weights_or_null, const larch_tensor* out) = nullptr;
  // ABI 5.
  larch_status (*mmvq_id_mixed)(larch_stream*, const larch_tensor* w_hot, const larch_tensor* w_cold,
                                 const larch_tensor* table_i32, const larch_tensor* x_f16,
                                 const larch_tensor* ids_i32, const larch_tensor* weights_or_null,
                                 const larch_tensor* out) = nullptr;
  larch_status (*sigmoid_mul)(larch_stream*, const larch_tensor* x, const larch_tensor* gate,
                               const larch_tensor* out) = nullptr;
  larch_status (*attn)(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool,
                        const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                        const larch_tensor* q_lens_i32, const larch_tensor* mask_or_null,
                        const larch_tensor* sinks_or_null, uint32_t window, uint32_t splits, float scale,
                        const larch_tensor* out) = nullptr;
  larch_status (*sample)(larch_stream*, const larch_tensor* topk_vals_f32, const larch_tensor* topk_ids_i32,
                          const larch_tensor* lse_f32, const larch_draw_params* per_row,
                          const larch_tensor* pos_i32, const larch_tensor* ban_ids_or_null,
                          const larch_tensor* out_ids_i32, const larch_tensor* out_logprobs_f32) = nullptr;
  larch_status (*moe_route)(larch_stream*, const larch_tensor* logits_f32, float scale,
                             const larch_tensor* out_ids_i32, const larch_tensor* out_weights_f32) = nullptr;
  larch_status (*gdn_conv)(larch_stream*, const larch_tensor* x, const larch_tensor* w,
                            const larch_gdn_state* state, const larch_tensor* slots_i32,
                            const larch_tensor* q_lens_i32, const larch_tensor* snap_slots_or_null,
                            const larch_tensor* out) = nullptr;
  larch_status (*gdn)(larch_stream*, const larch_tensor* qkv, const larch_tensor* alpha,
                       const larch_tensor* beta, const larch_tensor* dt_bias, const larch_tensor* a, float eps,
                       const larch_gdn_state* state, const larch_tensor* slots_i32,
                       const larch_tensor* q_lens_i32, const larch_tensor* snap_slots_or_null,
                       const larch_tensor* out) = nullptr;
  // The training entries (larch.h): CPU and CUDA.
  // ponytail: Metal has none (UNSUPPORTED); add them to its backend when the trainer runs on Apple silicon.
  larch_status (*gemm_ex)(larch_stream*, const larch_tensor* a, const larch_tensor* b, const larch_tensor* c,
                           uint32_t trans_a, uint32_t trans_b, float alpha, float beta) = nullptr;
  larch_status (*rmsnorm_f32)(larch_stream*, const larch_tensor* x, const larch_tensor* w, float eps,
                               const larch_tensor* out) = nullptr;
  larch_status (*rmsnorm_backward)(larch_stream*, const larch_tensor* x, const larch_tensor* w,
                                    const larch_tensor* dy, float eps, const larch_tensor* dx,
                                    uint32_t accumulate) = nullptr;
  larch_status (*rope_f32)(larch_stream*, const larch_tensor* qk, const larch_tensor* pos_i32,
                            const larch_rope_params* p) = nullptr;
  larch_status (*silu_mul_f32)(larch_stream*, const larch_tensor* gate, const larch_tensor* up,
                                const larch_tensor* out) = nullptr;
  larch_status (*silu_mul_backward)(larch_stream*, const larch_tensor* gate, const larch_tensor* up,
                                     const larch_tensor* dout, const larch_tensor* dgate,
                                     const larch_tensor* dup) = nullptr;
  larch_status (*softmax)(larch_stream*, const larch_tensor* x, float scale, uint32_t causal,
                           const larch_tensor* out) = nullptr;
  larch_status (*softmax_backward)(larch_stream*, const larch_tensor* p, const larch_tensor* dp, float scale,
                                    const larch_tensor* ds) = nullptr;
  larch_status (*layernorm)(larch_stream*, const larch_tensor* x, const larch_tensor* w,
                             const larch_tensor* b_or_null, float eps, const larch_tensor* out) = nullptr;
  larch_status (*gelu)(larch_stream*, const larch_tensor* x, const larch_tensor* out) = nullptr;
  larch_status (*attn_full)(larch_stream*, const larch_tensor* q, const larch_tensor* k, const larch_tensor* v,
                             uint32_t seg_len, float scale, const larch_tensor* out) = nullptr;
  larch_status (*rope_multi)(larch_stream*, const larch_tensor* qk, const larch_tensor* pos_i32,
                              const larch_rope_multi_params* p) = nullptr;
  // ABI 4's fused entries: null answers UNSUPPORTED after the shared checks, as above.
  larch_status (*fused_moe_gate_up)(larch_stream*, const larch_tensor* gate_w, const larch_tensor* up_w,
                                     const larch_tensor* x_f16, const larch_tensor* ids_i32,
                                     const larch_tensor* out) = nullptr;
  larch_status (*fused_moe_down)(larch_stream*, const larch_tensor* down_w, const larch_tensor* act_f16,
                                  const larch_tensor* ids_i32, const larch_tensor* weights_f32,
                                  const larch_tensor* shared_w, const larch_tensor* shared_act,
                                  const larch_tensor* shared_gate, const larch_tensor* x_f32) = nullptr;
  larch_status (*fused_moe_in)(larch_stream*, const larch_tensor* x_f32, const larch_tensor* norm_w, float eps,
                                const larch_tensor* h_f16, const larch_tensor* router_f16,
                                const larch_tensor* logits_f32, const larch_tensor* ids_i32,
                                const larch_tensor* weights_f32, const larch_tensor* shared_gate_w,
                                const larch_tensor* shared_up_w, const larch_tensor* shared_act,
                                const larch_tensor* gate_inp, const larch_tensor* shared_gate) = nullptr;
  larch_status (*fused_gdn_in)(larch_stream*, const larch_tensor* x_f32, const larch_tensor* norm_w, float eps,
                                const larch_tensor* h_f16, const larch_tensor* qkv_w, const larch_tensor* z_w,
                                const larch_tensor* alpha_w, const larch_tensor* beta_w,
                                const larch_tensor* qkv_f16, const larch_tensor* z_f16,
                                const larch_tensor* alpha_f32, const larch_tensor* beta_f32) = nullptr;
  larch_status (*fused_proj_add)(larch_stream*, const larch_tensor* w_quant, const larch_tensor* in_f16,
                                  const larch_tensor* gate, const larch_tensor* x_f32) = nullptr;
  larch_status (*fused_gdn)(larch_stream*, const larch_tensor* qkv, const larch_tensor* conv_w,
                             const larch_tensor* alpha, const larch_tensor* beta, const larch_tensor* dt_bias,
                             const larch_tensor* a, float eps, const larch_gdn_state* state,
                             const larch_tensor* slots_i32, const larch_tensor* q_lens_i32,
                             const larch_tensor* snap_slots, const larch_tensor* norm_w, const larch_tensor* z,
                             const larch_tensor* conv, const larch_tensor* out) = nullptr;
  larch_status (*fused_attn_in)(larch_stream*, const larch_tensor* q_in, const larch_tensor* k,
                                 const larch_tensor* v, const larch_tensor* q_norm, const larch_tensor* k_norm,
                                 float eps, const larch_tensor* pos_i32, const larch_rope_params* p,
                                 const larch_tensor* slots_i32, const larch_kv_pool* pool, const larch_tensor* q,
                                 const larch_tensor* gate) = nullptr;
  larch_status (*draft_conv)(larch_stream*, const larch_tensor* x, const larch_tensor* dyn,
                              const larch_tensor* base, uint32_t stage, const larch_tensor* out) = nullptr;
  // ABI 6, Inkling 2: null answers UNSUPPORTED after the shared checks, as above.
  larch_status (*short_conv)(larch_stream*, const larch_tensor* x, const larch_tensor* w, const larch_tensor* dyn,
                              const larch_tensor* residual, const larch_conv_state* state, uint32_t offset,
                              const larch_tensor* slots_i32, const larch_tensor* q_lens_i32,
                              const larch_tensor* snap_slots, const larch_tensor* out) = nullptr;
  larch_status (*clip_norm)(larch_stream*, const larch_tensor* x, const larch_tensor* scale, float t,
                             const larch_tensor* pos_i32, uint32_t floor, float alpha, const larch_tensor* out) = nullptr;
  larch_status (*depth_mix)(larch_stream*, const larch_tensor* states, const larch_tensor* query,
                             const larch_tensor* out) = nullptr;
  larch_status (*loop_gate)(larch_stream*, const larch_tensor* x, const larch_tensor* msg, const larch_tensor* gain,
                             const larch_tensor* forget, const larch_tensor* out) = nullptr;
  larch_status (*situ_mul)(larch_stream*, const larch_tensor* gate_up, const larch_tensor* ids,
                            const larch_tensor* scale, float beta, float lin_beta, const larch_tensor* out) = nullptr;
  larch_status (*moe_route_sigmoid)(larch_stream*, const larch_tensor* logits, const larch_tensor* bias, float eps,
                                     float scale, const larch_tensor* down_scale, const larch_tensor* out_ids,
                                     const larch_tensor* out_weights) = nullptr;
  larch_status (*attn_rel)(larch_stream*, const larch_tensor* q, const larch_kv_pool* pool,
                            const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                            const larch_tensor* q_lens_i32, const larch_tensor* mask, const larch_tensor* rel_q,
                            const larch_tensor* rel_w, const larch_tensor* sink, uint32_t window, uint32_t splits,
                            float scale, const larch_tensor* out) = nullptr;
  larch_status (*softcap)(larch_stream*, const larch_tensor* x, float cap, const larch_tensor* out) = nullptr;
};

// The debug op's entry (dispatch.cpp): INVALID_ARG for a null handle or while the stream captures,
// UNSUPPORTED on a backend without the slot, else the backend's answer.
extern "C" larch_status vv_fault_device_lost(larch_stream* s);
// Debug only (dispatch.cpp): a nonzero runtime error code here fails the next CUDA stream's INT8 tile
// probe with it (tests/quant_gemm.cpp). Every backend links it; only CUDA reads it.
extern "C" int vv_fault_quant_i8_probe;

// The table of each backend: src/cpu always (the reference and the CPU device), plus the one GPU
// backend that is compiled.
const vv_ops* vv_cpu_ops();
const vv_ops* vv_cpu_device_ops();  // ordinal -2: the reference's ops until the fast CPU backend fills its slots
const vv_ops* vv_cuda_ops();  // CUDA, and HIP built from the hipified src/cuda
const vv_ops* vv_metal_ops();

// Records the calling thread's detail string for larch_last_error and returns `status`.
larch_status vv_fail(larch_status status, const std::string& detail);
