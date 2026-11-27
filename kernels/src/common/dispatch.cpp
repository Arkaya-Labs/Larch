// The public larch_* symbols. Ordinal -1 opens the CPU reference, -2 the CPU device and 0.. the
// compiled GPU backend; every later call goes to the backend recorded in its handle (backend.hpp).
#include <initializer_list>
#include <new>

#include "backend.hpp"
#include "cpu/cpu.h"

static thread_local std::string last_error;

larch_status vv_fail(larch_status status, const std::string& detail) {
  last_error = detail;
  return status;
}

// Out of host memory: an empty detail, so no earlier failure's shows (setting one could throw again).
static larch_status oom() noexcept {
  last_error.clear();
  return LARCH_ERR_OOM;
}

// Anything else a backend throws: LAUNCH, as the CPU backend's `run` reports it, with no detail for
// the same reason. An exception that left a C entry would end the process.
static larch_status escaped() noexcept {
  last_error.clear();
  return LARCH_ERR_LAUNCH;
}

static const vv_ops* gpu() {
#if defined(LARCH_BACKEND_IS_CUDA) || defined(LARCH_BACKEND_IS_HIP)  // HIP: src/cuda, hipified
  return vv_cuda_ops();
#elif defined(LARCH_BACKEND_IS_METAL)
  return vv_metal_ops();
#else
  return nullptr;
#endif
}

// Calls `fn` on the backend that created handle `h`; a null handle is INVALID_ARG. Host memory running
// out in a backend (a handle, an error string) is OOM: nothing may unwind into the C caller.
#define VV_FORWARD(h, fn, ...)                                                                              \
  try {                                                                                                     \
    return (h) ? (h)->ops->fn(__VA_ARGS__) : vv_fail(LARCH_ERR_INVALID_ARG, "larch_" #fn ": null handle"); \
  } catch (const std::bad_alloc&) {                                                                         \
    return oom();                                                                                           \
  } catch (...) {                                                                                           \
    return escaped();                                                                                       \
  }

// After the shared checks `check` pass with the stream's own buffers, so a malformed call is INVALID_ARG on
// every backend: UNSUPPORTED, for an entry the backend lacks (`why`).
template <class Check>
static larch_status unsupported(const char* name, const larch_stream* s, const char* why, const Check& check) try {
  check([s](const larch_buffer* b, const std::string& what) {
    if (!b || b->ops != s->ops) throw cpu::bad_arg(what + " is null or another backend's tensor");
    return b->bytes;
  });
  return vv_fail(LARCH_ERR_UNSUPPORTED, std::string(name) + ": " + why);
} catch (const cpu::bad_arg& e) {
  return vv_fail(LARCH_ERR_INVALID_ARG, std::string(name) + ": " + e.what());
} catch (const std::bad_alloc&) {
  return oom();
} catch (...) {
  return escaped();
}

static constexpr const char* NO_OP = "not on this backend";
static constexpr const char* NO_PAGES = "no KV pages of this dtype on this backend";

// s's backend lacks the pages of pool's dtype (vv_ops::kv_dtypes); an unknown dtype counts as lacking.
static bool no_pages(const larch_stream* s, const larch_kv_pool* pool) {
  return s && pool && !(uint32_t(pool->dtype) < 32 && (s->ops->kv_dtypes >> pool->dtype & 1));
}

static constexpr const char* NO_WEIGHTS = "no kernels for this weight dtype on this backend";

// s's backend lacks kernels for w's dtype (vv_ops::no_weights).
static bool no_weights(const larch_stream* s, const larch_tensor* w) {
  return s && w && uint32_t(w->dtype) < 32 && (s->ops->no_weights >> w->dtype & 1);
}

uint32_t larch_abi_version(void) { return LARCH_ABI_VERSION; }

const char* larch_last_error(const larch_device*) { return last_error.c_str(); }

larch_status larch_backend_info(larch_backend* out, uint32_t* runtime_version, uint32_t* driver_version) try {
  return (gpu() ? gpu() : vv_cpu_ops())->backend_info(out, runtime_version, driver_version);
} catch (const std::bad_alloc&) {
  return oom();
} catch (...) {
  return escaped();
}

larch_status larch_device_count(int32_t* out) try {
  if (gpu()) return gpu()->device_count(out);
  *out = 0;
  return LARCH_OK;
} catch (const std::bad_alloc&) {
  return oom();
} catch (...) {
  return escaped();
}

larch_status larch_device_open(int32_t ordinal, larch_device** out) try {
  if (ordinal == -1) return vv_cpu_ops()->device_open(ordinal, out);
  if (ordinal == -2) return vv_cpu_device_ops()->device_open(ordinal, out);
  if (ordinal >= 0 && gpu()) return gpu()->device_open(ordinal, out);
  return vv_fail(LARCH_ERR_INVALID_ARG, "no device " + std::to_string(ordinal));
} catch (const std::bad_alloc&) {
  return oom();
} catch (...) {
  return escaped();
}

larch_status larch_device_props_get(const larch_device* d, larch_device_props* out) {
  VV_FORWARD(d, device_props_get, d, out);
}

void larch_device_close(larch_device* d) try {
  if (d) d->ops->device_close(d);
} catch (...) {  // a teardown has no status to return, and nothing may unwind into the C caller
}

larch_status larch_device_set_threads(larch_device* d, uint32_t threads) {
  if (d && !d->ops->device_set_threads)
    return vv_fail(LARCH_ERR_UNSUPPORTED, "larch_device_set_threads: not a CPU device");
  VV_FORWARD(d, device_set_threads, d, threads);
}

static larch_status sized(larch_status status, larch_buffer** out, uint64_t bytes) {
  if (status == LARCH_OK) (*out)->bytes = bytes;
  return status;
}

larch_status larch_buffer_alloc(larch_device* d, uint64_t bytes, larch_buffer** out) try {
  return d ? sized(d->ops->buffer_alloc(d, bytes, out), out, bytes)
           : vv_fail(LARCH_ERR_INVALID_ARG, "larch_buffer_alloc: null handle");
} catch (const std::bad_alloc&) {
  return oom();
} catch (...) {
  return escaped();
}

larch_status larch_buffer_wrap_host(larch_device* d, void* host, uint64_t bytes, larch_buffer** out) try {
  return d ? sized(d->ops->buffer_wrap_host(d, host, bytes, out), out, bytes)
           : vv_fail(LARCH_ERR_INVALID_ARG, "larch_buffer_wrap_host: null handle");
} catch (const std::bad_alloc&) {
  return oom();
} catch (...) {
  return escaped();
}

larch_status larch_buffer_alloc_host(larch_device* d, uint64_t bytes, larch_buffer** out, void** host) try {
  if (!d) return vv_fail(LARCH_ERR_INVALID_ARG, "larch_buffer_alloc_host: null handle");
  if (!d->ops->buffer_alloc_host)
    return vv_fail(LARCH_ERR_UNSUPPORTED, "larch_buffer_alloc_host: not on this backend");
  return sized(d->ops->buffer_alloc_host(d, bytes, out, host), out, bytes);
} catch (const std::bad_alloc&) {
  return oom();
} catch (...) {
  return escaped();
}

void larch_buffer_free(larch_buffer* b) try {
  if (b) b->ops->buffer_free(b);
} catch (...) {  // a teardown has no status to return, and nothing may unwind into the C caller
}

larch_status larch_upload(larch_stream* s, larch_buffer* b, uint64_t off, const void* src, uint64_t bytes) {
  VV_FORWARD(s, upload, s, b, off, src, bytes);
}

larch_status larch_download(larch_stream* s, const larch_buffer* b, uint64_t off, void* dst, uint64_t bytes) {
  VV_FORWARD(s, download, s, b, off, dst, bytes);
}

larch_status larch_stream_create(larch_device* d, larch_stream** out) { VV_FORWARD(d, stream_create, d, out); }

larch_status larch_stream_sync(larch_stream* s) { VV_FORWARD(s, stream_sync, s); }

void larch_stream_destroy(larch_stream* s) try {
  if (s) s->ops->stream_destroy(s);
} catch (...) {  // a teardown has no status to return, and nothing may unwind into the C caller
}

larch_status larch_event_record(larch_stream* s, larch_event** out) { VV_FORWARD(s, event_record, s, out); }

larch_status larch_event_elapsed_ns(const larch_event* a, const larch_event* b, uint64_t* out) {
  VV_FORWARD(a && b ? a : nullptr, event_elapsed_ns, a, b, out);
}

void larch_event_destroy(larch_event* e) try {
  if (e) e->ops->event_destroy(e);
} catch (...) {  // a teardown has no status to return, and nothing may unwind into the C caller
}

larch_status larch_event_sync(const larch_event* e) {
  if (e && !e->ops->event_sync) return vv_fail(LARCH_ERR_UNSUPPORTED, "larch_event_sync: not on this backend");
  VV_FORWARD(e, event_sync, e);
}

// Another backend's event is waited for on the host, which orders the stream's later work too; a backend's
// own events need its own wait.
larch_status larch_stream_wait(larch_stream* s, const larch_event* e) {
  if (!s || !e) return vv_fail(LARCH_ERR_INVALID_ARG, "larch_stream_wait: null handle");
  if (e->ops != s->ops) return larch_event_sync(e);
  if (!s->ops->stream_wait) return vv_fail(LARCH_ERR_UNSUPPORTED, "larch_stream_wait: not on this backend");
  VV_FORWARD(s, stream_wait, s, e);
}

larch_status larch_graph_begin(larch_stream* s) { VV_FORWARD(s, graph_begin, s); }

larch_status larch_graph_end(larch_stream* s, larch_graph** out) { VV_FORWARD(s, graph_end, s, out); }

larch_status larch_graph_launch(larch_graph* g, larch_stream* s) { VV_FORWARD(g, graph_launch, g, s); }

void larch_graph_destroy(larch_graph* g) try {
  if (g) g->ops->graph_destroy(g);
} catch (...) {  // a teardown has no status to return, and nothing may unwind into the C caller
}

larch_status larch_embed(larch_stream* s, const larch_tensor* table, const larch_tensor* ids_i32,
                           const larch_tensor* out_f32) {
  if (no_weights(s, table))
    return unsupported("embed", s, NO_WEIGHTS, [&](const cpu::sizer& b) { cpu::check_embed(b, table, ids_i32, out_f32); });
  VV_FORWARD(s, embed, s, table, ids_i32, out_f32);
}

larch_status larch_rmsnorm(larch_stream* s, const larch_tensor* x, const larch_tensor* w, float eps,
                             const larch_tensor* out) {
  VV_FORWARD(s, rmsnorm, s, x, w, eps, out);
}

larch_status larch_rope_multi(larch_stream* s, const larch_tensor* qk, const larch_tensor* pos_i32,
                                const larch_rope_multi_params* p) {
  if (s && !s->ops->rope_multi)
    return unsupported("rope_multi", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_rope_multi(b, qk, pos_i32, p); });
  VV_FORWARD(s, rope_multi, s, qk, pos_i32, p);
}

larch_status larch_layernorm(larch_stream* s, const larch_tensor* x, const larch_tensor* w,
                               const larch_tensor* b_or_null, float eps, const larch_tensor* out) {
  if (s && !s->ops->layernorm)
    return unsupported("layernorm", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_layernorm(b, x, w, b_or_null, out); });
  VV_FORWARD(s, layernorm, s, x, w, b_or_null, eps, out);
}

larch_status larch_rope(larch_stream* s, const larch_tensor* qk, const larch_tensor* pos_i32,
                          const larch_rope_params* p) {
  VV_FORWARD(s, rope, s, qk, pos_i32, p);
}

larch_status larch_mmvq(larch_stream* s, const larch_tensor* w_quant, const larch_tensor* x_f16,
                          const larch_tensor* bias_or_null, const larch_tensor* out) {
  if (no_weights(s, w_quant))
    return unsupported("mmvq", s, NO_WEIGHTS,
                       [&](const cpu::sizer& b) { cpu::check_mmvq(b, w_quant, x_f16, bias_or_null, out); });
  VV_FORWARD(s, mmvq, s, w_quant, x_f16, bias_or_null, out);
}

larch_status larch_mmvq_verify(larch_stream* s, const larch_tensor* w_quant, const larch_tensor* x_f16,
                                 const larch_tensor* bias_or_null, const larch_tensor* out) {
  if (s && (!s->ops->mmvq_verify || no_weights(s, w_quant)))
    return unsupported("mmvq_verify", s, s->ops->mmvq_verify ? NO_WEIGHTS : NO_OP,
                       [&](const cpu::sizer& b) { cpu::check_mmvq_verify(b, w_quant, x_f16, bias_or_null, out); });
  VV_FORWARD(s, mmvq_verify, s, w_quant, x_f16, bias_or_null, out);
}

larch_status larch_mmvq_id(larch_stream* s, const larch_tensor* w_quant, const larch_tensor* x_f16,
                             const larch_tensor* ids_i32, const larch_tensor* bias_or_null,
                             const larch_tensor* weights_or_null, const larch_tensor* out) {
  if (s && (!s->ops->mmvq_id || no_weights(s, w_quant)))
    return unsupported("mmvq_id", s, s->ops->mmvq_id ? NO_WEIGHTS : NO_OP, [&](const cpu::sizer& b) {
      cpu::check_mmvq_id(b, w_quant, x_f16, ids_i32, bias_or_null, weights_or_null, out);
    });
  VV_FORWARD(s, mmvq_id, s, w_quant, x_f16, ids_i32, bias_or_null, weights_or_null, out);
}

larch_status larch_mmvq_id_mixed(larch_stream* s, const larch_tensor* w_hot, const larch_tensor* w_cold,
                                   const larch_tensor* table_i32, const larch_tensor* x_f16,
                                   const larch_tensor* ids_i32, const larch_tensor* weights_or_null,
                                   const larch_tensor* out) {
  if (s && (!s->ops->mmvq_id_mixed || no_weights(s, w_hot) || no_weights(s, w_cold)))
    return unsupported("mmvq_id_mixed", s, s->ops->mmvq_id_mixed ? NO_WEIGHTS : NO_OP, [&](const cpu::sizer& b) {
      cpu::check_mmvq_id_mixed(b, w_hot, w_cold, table_i32, x_f16, ids_i32, weights_or_null, out);
    });
  VV_FORWARD(s, mmvq_id_mixed, s, w_hot, w_cold, table_i32, x_f16, ids_i32, weights_or_null, out);
}

larch_status larch_dequant(larch_stream* s, const larch_tensor* w_quant, const larch_tensor* out_f16) {
  if (no_weights(s, w_quant))
    return unsupported("dequant", s, NO_WEIGHTS, [&](const cpu::sizer& b) { cpu::check_dequant(b, w_quant, out_f16); });
  VV_FORWARD(s, dequant, s, w_quant, out_f16);
}

// The quantized GEMM entries: a backend without the slot answers UNSUPPORTED so callers keep their fallback.
static larch_status quant_entry(const char* name, larch_stream* s, decltype(vv_ops::quant_gemm) vv_ops::*slot,
                                 const larch_tensor* w_quant, const larch_tensor* x_f16,
                                 const larch_tensor* bias_or_null, const larch_tensor* out) try {
  if (!s) return vv_fail(LARCH_ERR_INVALID_ARG, std::string("larch_") + name + ": null handle");
  if (s->ops->*slot && !no_weights(s, w_quant)) return (s->ops->*slot)(s, w_quant, x_f16, bias_or_null, out);
  return unsupported(name, s, s->ops->*slot ? NO_WEIGHTS : NO_OP,
                     [&](const cpu::sizer& b) { cpu::check_quant_gemm(b, w_quant, x_f16, bias_or_null, out); });
} catch (const std::bad_alloc&) {
  return oom();
} catch (...) {
  return escaped();
}

larch_status larch_quant_gemm(larch_stream* s, const larch_tensor* w_quant, const larch_tensor* x_f16,
                                const larch_tensor* bias_or_null, const larch_tensor* out) {
  return quant_entry("quant_gemm", s, &vv_ops::quant_gemm, w_quant, x_f16, bias_or_null, out);
}

larch_status larch_quant_gemm_i8(larch_stream* s, const larch_tensor* w_quant, const larch_tensor* x_f16,
                                   const larch_tensor* bias_or_null, const larch_tensor* out) {
  return quant_entry("quant_gemm_i8", s, &vv_ops::quant_gemm_i8, w_quant, x_f16, bias_or_null, out);
}

larch_status larch_quant_gemm_i8_split(larch_stream* s, const larch_tensor* w_quant, const larch_tensor* x_f16,
                                         const larch_tensor* bias_or_null, const larch_tensor* out) {
  return quant_entry("quant_gemm_i8_split", s, &vv_ops::quant_gemm_i8_split, w_quant, x_f16, bias_or_null, out);
}

larch_status larch_gemm(larch_stream* s, const larch_tensor* a_f16, const larch_tensor* b_f16_t,
                          const larch_tensor* bias_or_null, const larch_tensor* c, float alpha, float beta) {
  VV_FORWARD(s, gemm, s, a_f16, b_f16_t, bias_or_null, c, alpha, beta);
}

larch_status larch_silu_mul(larch_stream* s, const larch_tensor* gate, const larch_tensor* up,
                              const larch_tensor* out) {
  VV_FORWARD(s, silu_mul, s, gate, up, out);
}

larch_status larch_sigmoid_mul(larch_stream* s, const larch_tensor* x, const larch_tensor* gate,
                                 const larch_tensor* out) {
  if (s && !s->ops->sigmoid_mul)
    return unsupported("sigmoid_mul", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_sigmoid_mul(b, x, gate, out); });
  VV_FORWARD(s, sigmoid_mul, s, x, gate, out);
}

larch_status larch_attn_full(larch_stream* s, const larch_tensor* q, const larch_tensor* k, const larch_tensor* v,
                               uint32_t seg_len, float scale, const larch_tensor* out) {
  if (s && !s->ops->attn_full)
    return unsupported("attn_full", s, NO_OP,
                       [&](const cpu::sizer& b) { cpu::check_attn_full(b, q, k, v, seg_len, out); });
  VV_FORWARD(s, attn_full, s, q, k, v, seg_len, scale, out);
}

larch_status larch_gelu(larch_stream* s, const larch_tensor* x, const larch_tensor* out) {
  if (s && !s->ops->gelu) return unsupported("gelu", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_gelu(b, x, out); });
  VV_FORWARD(s, gelu, s, x, out);
}

larch_status larch_add(larch_stream* s, const larch_tensor* a_f32, const larch_tensor* b_f16,
                         const larch_tensor* out) {
  VV_FORWARD(s, add, s, a_f32, b_f16, out);
}

larch_status larch_kv_write(larch_stream* s, const larch_tensor* k, const larch_tensor* v,
                              const larch_tensor* slots_i32, const larch_kv_pool* pool) {
  if (no_pages(s, pool))
    return unsupported("kv_write", s, NO_PAGES,
                       [&](const cpu::sizer& b) { cpu::check_kv_write(b, k, v, slots_i32, pool); });
  VV_FORWARD(s, kv_write, s, k, v, slots_i32, pool);
}

larch_status larch_attn_decode(larch_stream* s, const larch_tensor* q, const larch_kv_pool* pool,
                                 const larch_tensor* block_table_i32, const larch_tensor* seq_lens_i32,
                                 uint32_t splits, float scale, const larch_tensor* out) {
  if (no_pages(s, pool))
    return unsupported("attn_decode", s, NO_PAGES, [&](const cpu::sizer& b) {
      cpu::check_attn_decode(b, q, pool, block_table_i32, seq_lens_i32, out);
    });
  VV_FORWARD(s, attn_decode, s, q, pool, block_table_i32, seq_lens_i32, splits, scale, out);
}

larch_status larch_attn_prefill(larch_stream* s, const larch_tensor* q, const larch_kv_pool* pool,
                                  const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                                  const larch_tensor* q_lens_i32, float scale, const larch_tensor* out) {
  if (no_pages(s, pool))
    return unsupported("attn_prefill", s, NO_PAGES, [&](const cpu::sizer& b) {
      cpu::check_attn_prefill(b, q, pool, block_table_i32, q_start_pos_i32, q_lens_i32, out);
    });
  VV_FORWARD(s, attn_prefill, s, q, pool, block_table_i32, q_start_pos_i32, q_lens_i32, scale, out);
}

larch_status larch_attn(larch_stream* s, const larch_tensor* q, const larch_kv_pool* pool,
                          const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                          const larch_tensor* q_lens_i32, const larch_tensor* mask_or_null,
                          const larch_tensor* sinks_or_null, uint32_t window, uint32_t splits, float scale,
                          const larch_tensor* out) {
  if (s && (!s->ops->attn || no_pages(s, pool)))
    return unsupported("attn", s, s->ops->attn ? NO_PAGES : NO_OP, [&](const cpu::sizer& b) {
      cpu::check_attn(b, q, pool, block_table_i32, q_start_pos_i32, q_lens_i32, mask_or_null, sinks_or_null, out);
    });
  VV_FORWARD(s, attn, s, q, pool, block_table_i32, q_start_pos_i32, q_lens_i32, mask_or_null, sinks_or_null, window,
             splits, scale, out);
}

larch_status larch_qk_norm(larch_stream* s, const larch_tensor* qk, const larch_tensor* w, float eps) {
  VV_FORWARD(s, qk_norm, s, qk, w, eps);
}

larch_status larch_sample_prep(larch_stream* s, const larch_tensor* logits, const larch_sample_params* per_row,
                                 const larch_tensor* penalty_ids_i32, const larch_tensor* out_topk_vals_f32,
                                 const larch_tensor* out_topk_ids_i32, const larch_tensor* out_lse_f32) {
  VV_FORWARD(s, sample_prep, s, logits, per_row, penalty_ids_i32, out_topk_vals_f32, out_topk_ids_i32, out_lse_f32);
}

larch_status larch_sample(larch_stream* s, const larch_tensor* topk_vals_f32, const larch_tensor* topk_ids_i32,
                            const larch_tensor* lse_f32, const larch_draw_params* per_row,
                            const larch_tensor* pos_i32, const larch_tensor* ban_ids_or_null,
                            const larch_tensor* out_ids_i32, const larch_tensor* out_logprobs_f32) {
  if (s && !s->ops->sample)
    return unsupported("sample", s, NO_OP, [&](const cpu::sizer& b) {
      cpu::check_sample(b, topk_vals_f32, topk_ids_i32, lse_f32, per_row, pos_i32, ban_ids_or_null, out_ids_i32,
                        out_logprobs_f32);
    });
  VV_FORWARD(s, sample, s, topk_vals_f32, topk_ids_i32, lse_f32, per_row, pos_i32, ban_ids_or_null, out_ids_i32,
             out_logprobs_f32);
}

larch_status larch_moe_route(larch_stream* s, const larch_tensor* logits_f32, float scale,
                               const larch_tensor* out_ids_i32, const larch_tensor* out_weights_f32) {
  if (s && !s->ops->moe_route)
    return unsupported("moe_route", s, NO_OP, [&](const cpu::sizer& b) {
      cpu::check_moe_route(b, logits_f32, scale, out_ids_i32, out_weights_f32);
    });
  VV_FORWARD(s, moe_route, s, logits_f32, scale, out_ids_i32, out_weights_f32);
}

// The training entries: a backend without one answers UNSUPPORTED after the shared checks.
larch_status larch_gemm_ex(larch_stream* s, const larch_tensor* a, const larch_tensor* b, const larch_tensor* c,
                             uint32_t trans_a, uint32_t trans_b, float alpha, float beta) {
  if (s && !s->ops->gemm_ex)
    return unsupported("gemm_ex", s, NO_OP, [&](const cpu::sizer& z) { cpu::check_gemm_ex(z, a, b, c, trans_a, trans_b); });
  VV_FORWARD(s, gemm_ex, s, a, b, c, trans_a, trans_b, alpha, beta);
}

larch_status larch_rmsnorm_f32(larch_stream* s, const larch_tensor* x, const larch_tensor* w, float eps,
                                 const larch_tensor* out) {
  if (s && !s->ops->rmsnorm_f32)
    return unsupported("rmsnorm_f32", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_rmsnorm(b, x, w, out, LARCH_F32); });
  VV_FORWARD(s, rmsnorm_f32, s, x, w, eps, out);
}

larch_status larch_rmsnorm_backward(larch_stream* s, const larch_tensor* x, const larch_tensor* w,
                                      const larch_tensor* dy, float eps, const larch_tensor* dx,
                                      uint32_t accumulate) {
  if (s && !s->ops->rmsnorm_backward)
    return unsupported("rmsnorm_backward", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_rmsnorm_backward(b, x, w, dy, dx); });
  VV_FORWARD(s, rmsnorm_backward, s, x, w, dy, eps, dx, accumulate);
}

larch_status larch_rope_f32(larch_stream* s, const larch_tensor* qk, const larch_tensor* pos_i32,
                              const larch_rope_params* p) {
  if (s && !s->ops->rope_f32)
    return unsupported("rope_f32", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_rope(b, qk, pos_i32, p, LARCH_F32); });
  VV_FORWARD(s, rope_f32, s, qk, pos_i32, p);
}

larch_status larch_silu_mul_f32(larch_stream* s, const larch_tensor* gate, const larch_tensor* up,
                                  const larch_tensor* out) {
  if (s && !s->ops->silu_mul_f32)
    return unsupported("silu_mul_f32", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_silu_mul(b, gate, up, out, LARCH_F32); });
  VV_FORWARD(s, silu_mul_f32, s, gate, up, out);
}

larch_status larch_silu_mul_backward(larch_stream* s, const larch_tensor* gate, const larch_tensor* up,
                                       const larch_tensor* dout, const larch_tensor* dgate,
                                       const larch_tensor* dup) {
  if (s && !s->ops->silu_mul_backward)
    return unsupported("silu_mul_backward", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_silu_mul_backward(b, gate, up, dout, dgate, dup); });
  VV_FORWARD(s, silu_mul_backward, s, gate, up, dout, dgate, dup);
}

larch_status larch_softmax(larch_stream* s, const larch_tensor* x, float scale, uint32_t causal,
                             const larch_tensor* out) {
  if (s && !s->ops->softmax)
    return unsupported("softmax", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_softmax(b, x, out); });
  VV_FORWARD(s, softmax, s, x, scale, causal, out);
}

larch_status larch_softmax_backward(larch_stream* s, const larch_tensor* p, const larch_tensor* dp, float scale,
                                      const larch_tensor* ds) {
  if (s && !s->ops->softmax_backward)
    return unsupported("softmax_backward", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_softmax_backward(b, p, dp, ds); });
  VV_FORWARD(s, softmax_backward, s, p, dp, scale, ds);
}

larch_status larch_gdn_conv(larch_stream* s, const larch_tensor* x, const larch_tensor* w,
                              const larch_gdn_state* state, const larch_tensor* slots_i32,
                              const larch_tensor* q_lens_i32, const larch_tensor* snap_slots_or_null,
                              const larch_tensor* out) {
  if (s && !s->ops->gdn_conv)
    return unsupported("gdn_conv", s, NO_OP, [&](const cpu::sizer& b) {
      cpu::check_gdn_conv(b, x, w, state, slots_i32, q_lens_i32, snap_slots_or_null, out);
    });
  VV_FORWARD(s, gdn_conv, s, x, w, state, slots_i32, q_lens_i32, snap_slots_or_null, out);
}

larch_status larch_gdn(larch_stream* s, const larch_tensor* qkv, const larch_tensor* alpha,
                         const larch_tensor* beta, const larch_tensor* dt_bias, const larch_tensor* a, float eps,
                         const larch_gdn_state* state, const larch_tensor* slots_i32,
                         const larch_tensor* q_lens_i32, const larch_tensor* snap_slots_or_null,
                         const larch_tensor* out) {
  if (s && !s->ops->gdn)
    return unsupported("gdn", s, NO_OP, [&](const cpu::sizer& b) {
      cpu::check_gdn(b, qkv, alpha, beta, dt_bias, a, eps, state, slots_i32, q_lens_i32, snap_slots_or_null, out);
    });
  VV_FORWARD(s, gdn, s, qkv, alpha, beta, dt_bias, a, eps, state, slots_i32, q_lens_i32, snap_slots_or_null, out);
}

// The fused entries (ABI 4): why s's backend answers one UNSUPPORTED after the shared checks (it lacks the entry, a
// weight's kernels or the pool's pages), or null when it runs it.
static const char* lacks(const larch_stream* s, bool slot, std::initializer_list<const larch_tensor*> weights,
                         const larch_kv_pool* pool = nullptr) {
  if (!s) return nullptr;
  if (!slot) return NO_OP;
  for (const larch_tensor* w : weights)
    if (no_weights(s, w)) return NO_WEIGHTS;
  return no_pages(s, pool) ? NO_PAGES : nullptr;
}

larch_status larch_fused_moe_gate_up(larch_stream* s, const larch_tensor* gate_w, const larch_tensor* up_w,
                                       const larch_tensor* x_f16, const larch_tensor* ids_i32,
                                       const larch_tensor* out) {
  if (const char* why = lacks(s, s && s->ops->fused_moe_gate_up, {gate_w, up_w}))
    return unsupported("fused_moe_gate_up", s, why, [&](const cpu::sizer& b) {
      cpu::check_fused_moe_gate_up(b, gate_w, up_w, x_f16, ids_i32, out);
    });
  VV_FORWARD(s, fused_moe_gate_up, s, gate_w, up_w, x_f16, ids_i32, out);
}

larch_status larch_fused_moe_down(larch_stream* s, const larch_tensor* down_w, const larch_tensor* act_f16,
                                    const larch_tensor* ids_i32, const larch_tensor* weights_f32,
                                    const larch_tensor* shared_w_or_null, const larch_tensor* shared_act_or_null,
                                    const larch_tensor* shared_gate_or_null, const larch_tensor* x_f32) {
  if (const char* why = lacks(s, s && s->ops->fused_moe_down, {down_w, shared_w_or_null}))
    return unsupported("fused_moe_down", s, why, [&](const cpu::sizer& b) {
      cpu::check_fused_moe_down(b, down_w, act_f16, ids_i32, weights_f32, shared_w_or_null, shared_act_or_null,
                                shared_gate_or_null, x_f32);
    });
  VV_FORWARD(s, fused_moe_down, s, down_w, act_f16, ids_i32, weights_f32, shared_w_or_null, shared_act_or_null,
             shared_gate_or_null, x_f32);
}

larch_status larch_fused_moe_in(larch_stream* s, const larch_tensor* x_f32, const larch_tensor* norm_w, float eps,
                                  const larch_tensor* h_f16, const larch_tensor* router_f16,
                                  const larch_tensor* logits_f32, const larch_tensor* ids_i32,
                                  const larch_tensor* weights_f32, const larch_tensor* shared_gate_w_or_null,
                                  const larch_tensor* shared_up_w_or_null, const larch_tensor* shared_act_or_null,
                                  const larch_tensor* gate_inp_or_null, const larch_tensor* shared_gate_or_null) {
  if (const char* why = lacks(s, s && s->ops->fused_moe_in, {shared_gate_w_or_null, shared_up_w_or_null}))
    return unsupported("fused_moe_in", s, why, [&](const cpu::sizer& b) {
      cpu::check_fused_moe_in(b, x_f32, norm_w, h_f16, router_f16, logits_f32, ids_i32, weights_f32,
                              shared_gate_w_or_null, shared_up_w_or_null, shared_act_or_null, gate_inp_or_null,
                              shared_gate_or_null);
    });
  VV_FORWARD(s, fused_moe_in, s, x_f32, norm_w, eps, h_f16, router_f16, logits_f32, ids_i32, weights_f32,
             shared_gate_w_or_null, shared_up_w_or_null, shared_act_or_null, gate_inp_or_null, shared_gate_or_null);
}

larch_status larch_fused_gdn_in(larch_stream* s, const larch_tensor* x_f32, const larch_tensor* norm_w, float eps,
                                  const larch_tensor* h_f16, const larch_tensor* qkv_w, const larch_tensor* z_w,
                                  const larch_tensor* alpha_w, const larch_tensor* beta_w,
                                  const larch_tensor* qkv_f16, const larch_tensor* z_f16,
                                  const larch_tensor* alpha_f32, const larch_tensor* beta_f32) {
  if (const char* why = lacks(s, s && s->ops->fused_gdn_in, {qkv_w, z_w, alpha_w, beta_w}))
    return unsupported("fused_gdn_in", s, why, [&](const cpu::sizer& b) {
      cpu::check_fused_gdn_in(b, x_f32, norm_w, h_f16, qkv_w, z_w, alpha_w, beta_w, qkv_f16, z_f16, alpha_f32,
                              beta_f32);
    });
  VV_FORWARD(s, fused_gdn_in, s, x_f32, norm_w, eps, h_f16, qkv_w, z_w, alpha_w, beta_w, qkv_f16, z_f16, alpha_f32,
             beta_f32);
}

larch_status larch_fused_proj_add(larch_stream* s, const larch_tensor* w_quant, const larch_tensor* in_f16,
                                    const larch_tensor* gate_or_null, const larch_tensor* x_f32) {
  if (const char* why = lacks(s, s && s->ops->fused_proj_add, {w_quant}))
    return unsupported("fused_proj_add", s, why,
                       [&](const cpu::sizer& b) { cpu::check_fused_proj_add(b, w_quant, in_f16, gate_or_null, x_f32); });
  VV_FORWARD(s, fused_proj_add, s, w_quant, in_f16, gate_or_null, x_f32);
}

larch_status larch_fused_gdn(larch_stream* s, const larch_tensor* qkv_f16, const larch_tensor* conv_w,
                               const larch_tensor* alpha_f32, const larch_tensor* beta_f32,
                               const larch_tensor* dt_bias, const larch_tensor* a, float eps,
                               const larch_gdn_state* state, const larch_tensor* slots_i32,
                               const larch_tensor* q_lens_i32, const larch_tensor* snap_slots_or_null,
                               const larch_tensor* norm_w, const larch_tensor* z_f16, const larch_tensor* conv_f16,
                               const larch_tensor* out_f16) {
  if (const char* why = lacks(s, s && s->ops->fused_gdn, {}))
    return unsupported("fused_gdn", s, why, [&](const cpu::sizer& b) {
      cpu::check_fused_gdn(b, qkv_f16, conv_w, alpha_f32, beta_f32, dt_bias, a, eps, state, slots_i32, q_lens_i32,
                           snap_slots_or_null, norm_w, z_f16, conv_f16, out_f16);
    });
  VV_FORWARD(s, fused_gdn, s, qkv_f16, conv_w, alpha_f32, beta_f32, dt_bias, a, eps, state, slots_i32, q_lens_i32,
             snap_slots_or_null, norm_w, z_f16, conv_f16, out_f16);
}

larch_status larch_fused_attn_in(larch_stream* s, const larch_tensor* q_in_f16, const larch_tensor* k_f16,
                                   const larch_tensor* v_f16, const larch_tensor* q_norm_or_null,
                                   const larch_tensor* k_norm_or_null, float eps, const larch_tensor* pos_i32,
                                   const larch_rope_params* p, const larch_tensor* slots_i32,
                                   const larch_kv_pool* pool, const larch_tensor* q_f16,
                                   const larch_tensor* gate_or_null) {
  if (const char* why = lacks(s, s && s->ops->fused_attn_in, {}, pool))
    return unsupported("fused_attn_in", s, why, [&](const cpu::sizer& b) {
      cpu::check_fused_attn_in(b, q_in_f16, k_f16, v_f16, q_norm_or_null, k_norm_or_null, pos_i32, p, slots_i32, pool,
                               q_f16, gate_or_null);
    });
  VV_FORWARD(s, fused_attn_in, s, q_in_f16, k_f16, v_f16, q_norm_or_null, k_norm_or_null, eps, pos_i32, p, slots_i32,
             pool, q_f16, gate_or_null);
}

larch_status larch_draft_conv(larch_stream* s, const larch_tensor* x_f16, const larch_tensor* dyn_f32,
                                const larch_tensor* base_f32, uint32_t stage, const larch_tensor* out) {
  if (s && !s->ops->draft_conv)
    return unsupported("draft_conv", s, NO_OP,
                       [&](const cpu::sizer& b) { cpu::check_draft_conv(b, x_f16, dyn_f32, base_f32, stage, out); });
  VV_FORWARD(s, draft_conv, s, x_f16, dyn_f32, base_f32, stage, out);
}

larch_status larch_short_conv(larch_stream* s, const larch_tensor* x_f16, const larch_tensor* w_f32,
                                const larch_tensor* dyn_or_null, const larch_tensor* residual_or_null,
                                const larch_conv_state* state, uint32_t offset, const larch_tensor* slots_i32,
                                const larch_tensor* q_lens_i32, const larch_tensor* snap_slots_or_null,
                                const larch_tensor* out) {
  if (s && !s->ops->short_conv)
    return unsupported("short_conv", s, NO_OP, [&](const cpu::sizer& b) {
      cpu::check_short_conv(b, x_f16, w_f32, dyn_or_null, residual_or_null, state, offset, slots_i32, q_lens_i32,
                            snap_slots_or_null, out);
    });
  VV_FORWARD(s, short_conv, s, x_f16, w_f32, dyn_or_null, residual_or_null, state, offset, slots_i32, q_lens_i32,
             snap_slots_or_null, out);
}

larch_status larch_clip_norm(larch_stream* s, const larch_tensor* x, const larch_tensor* scale_or_null, float t,
                               const larch_tensor* pos_or_null, uint32_t floor, float alpha, const larch_tensor* out) {
  if (s && !s->ops->clip_norm)
    return unsupported("clip_norm", s, NO_OP, [&](const cpu::sizer& b) {
      cpu::check_clip_norm(b, x, scale_or_null, t, pos_or_null, floor, alpha, out);
    });
  VV_FORWARD(s, clip_norm, s, x, scale_or_null, t, pos_or_null, floor, alpha, out);
}

larch_status larch_depth_mix(larch_stream* s, const larch_tensor* states_f32, const larch_tensor* query_f32,
                               const larch_tensor* out_f32) {
  if (s && !s->ops->depth_mix)
    return unsupported("depth_mix", s, NO_OP,
                       [&](const cpu::sizer& b) { cpu::check_depth_mix(b, states_f32, query_f32, out_f32); });
  VV_FORWARD(s, depth_mix, s, states_f32, query_f32, out_f32);
}

larch_status larch_loop_gate(larch_stream* s, const larch_tensor* x_f32, const larch_tensor* msg_f32,
                               const larch_tensor* gain, const larch_tensor* forget, const larch_tensor* out) {
  if (s && !s->ops->loop_gate)
    return unsupported("loop_gate", s, NO_OP,
                       [&](const cpu::sizer& b) { cpu::check_loop_gate(b, x_f32, msg_f32, gain, forget, out); });
  VV_FORWARD(s, loop_gate, s, x_f32, msg_f32, gain, forget, out);
}

larch_status larch_situ_mul(larch_stream* s, const larch_tensor* gate_up, const larch_tensor* ids_i32,
                              const larch_tensor* scale_or_null, float beta, float lin_beta, const larch_tensor* out) {
  if (s && !s->ops->situ_mul)
    return unsupported("situ_mul", s, NO_OP, [&](const cpu::sizer& b) {
      cpu::check_situ_mul(b, gate_up, ids_i32, scale_or_null, beta, lin_beta, out);
    });
  VV_FORWARD(s, situ_mul, s, gate_up, ids_i32, scale_or_null, beta, lin_beta, out);
}

larch_status larch_moe_route_sigmoid(larch_stream* s, const larch_tensor* logits_f32, const larch_tensor* bias_or_null,
                                       float eps, float scale, const larch_tensor* down_scale_or_null,
                                       const larch_tensor* out_ids_i32, const larch_tensor* out_weights_f32) {
  if (s && !s->ops->moe_route_sigmoid)
    return unsupported("moe_route_sigmoid", s, NO_OP, [&](const cpu::sizer& b) {
      cpu::check_moe_route_sigmoid(b, logits_f32, bias_or_null, eps, scale, down_scale_or_null, out_ids_i32,
                                   out_weights_f32);
    });
  VV_FORWARD(s, moe_route_sigmoid, s, logits_f32, bias_or_null, eps, scale, down_scale_or_null, out_ids_i32,
             out_weights_f32);
}

larch_status larch_attn_rel(larch_stream* s, const larch_tensor* q, const larch_kv_pool* pool,
                              const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                              const larch_tensor* q_lens_i32, const larch_tensor* mask_or_null,
                              const larch_tensor* rel_q, const larch_tensor* rel_w_f32, const larch_tensor* sink,
                              uint32_t window, uint32_t splits, float scale, const larch_tensor* out) {
  const char* why = !s ? nullptr : !s->ops->attn_rel ? NO_OP : no_pages(s, pool) ? NO_PAGES : nullptr;
  if (why)
    return unsupported("attn_rel", s, why, [&](const cpu::sizer& b) {
      cpu::check_attn_rel(b, q, pool, block_table_i32, q_start_pos_i32, q_lens_i32, mask_or_null, rel_q, rel_w_f32,
                          sink, out);
    });
  VV_FORWARD(s, attn_rel, s, q, pool, block_table_i32, q_start_pos_i32, q_lens_i32, mask_or_null, rel_q, rel_w_f32,
             sink, window, splits, scale, out);
}

larch_status larch_softcap(larch_stream* s, const larch_tensor* x_f32, float cap, const larch_tensor* out) {
  if (s && !s->ops->softcap)
    return unsupported("softcap", s, NO_OP, [&](const cpu::sizer& b) { cpu::check_softcap(b, x_f32, cap, out); });
  VV_FORWARD(s, softcap, s, x_f32, cap, out);
}

extern "C" {
int vv_fault_quant_i8_probe = 0;
}

// Debug only (architecture 9.11), so not in larch.h: its callers (tests/device_lost.cpp, larch-kernels)
// declare it themselves. A backend without the op (CPU, Metal, HIP) answers UNSUPPORTED; one with it
// answers INVALID_ARG while the stream captures.
extern "C" larch_status vv_fault_device_lost(larch_stream* s) try {
  if (!s) return vv_fail(LARCH_ERR_INVALID_ARG, "vv_fault_device_lost: null handle");
  if (!s->ops->fault_device_lost) return vv_fail(LARCH_ERR_UNSUPPORTED, "vv_fault_device_lost: not on this backend");
  return s->ops->fault_device_lost(s);
} catch (const std::bad_alloc&) {
  return oom();
} catch (...) {
  return escaped();
}
