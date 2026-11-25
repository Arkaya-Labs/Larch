// CPU reference: device, buffers, streams, events and the ops tables of both CPU devices. Everything runs
// at the call, so a stream is only a handle, sync is a no-op and an event is the time it was recorded,
// done once recorded. No graphs.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "cpu/cpu.h"

namespace cpu {

float view::get(int64_t r, int64_t c) const {
  const uint8_t* p = row(r) + c * elem_bytes;
  switch (dtype) {
    case LARCH_F32: { float v; std::memcpy(&v, p, 4); return v; }
    case LARCH_F16: return half(p);
    case LARCH_BF16: return bf16_to_f32(u16(p));
    default: throw bad_arg("get: not a float tensor");
  }
}

void view::set(int64_t r, int64_t c, float v) const {
  uint8_t* p = row(r) + c * elem_bytes;
  switch (dtype) {
    case LARCH_F32: std::memcpy(p, &v, 4); return;
    case LARCH_F16: { const uint16_t h = f32_to_f16(v); std::memcpy(p, &h, 2); return; }
    case LARCH_BF16: throw bad_arg("bf16 outputs are not supported");
    default: throw bad_arg("set: not a float tensor");
  }
}

int32_t view::i32(int64_t r, int64_t c) const {
  int32_t v;
  std::memcpy(&v, row(r) + c * 4, 4);
  return v;
}

void view::set_i32(int64_t r, int64_t c, int32_t v) const { std::memcpy(row(r) + c * 4, &v, 4); }

meta check_tensor(const sizer& bytes, const larch_tensor* t, const char* what) {
  const std::string w = what;
  const uint64_t size = bytes(t ? t->buf : nullptr, w);
  // A sizer refuses a null buffer; this keeps a lax one from crashing or passing a tensor off as absent.
  if (!t || !t->buf) throw bad_arg(w + " is null");
  if (t->ndim < 1 || t->ndim > 4) throw bad_arg(w + ": ndim " + std::to_string(t->ndim));
  const type_info ti = info(t->dtype);
  if (t->dtype == LARCH_FP8_E4M3) throw bad_arg(w + ": fp8_e4m3 is for KV pages only");
  if (ti.block > 1 && (t->ndim != 2 || t->shape[1] % ti.block))
    throw bad_arg(w + ": a quant tensor is 2-D with whole blocks per row");
  meta v;
  v.buf = t->buf;
  v.offset = t->offset;
  v.dtype = t->dtype;
  v.ndim = t->ndim;
  v.rows = t->shape[0];
  v.cols = 1;
  // Dims after the first are dense (in elements, or blocks for quant) and rows do not overlap.
  int64_t row = 1;  // elements (blocks for quant) per row
  for (uint32_t d = t->ndim; d-- > 1;) {
    if (t->shape[d] <= 0 || t->stride[d] != row) throw bad_arg(w + ": dims after the first must be dense");
    row = mul(row, d == t->ndim - 1 ? t->shape[d] / ti.block : t->shape[d]);
    v.cols = mul(v.cols, t->shape[d]);
  }
  if (v.rows < 0 || t->stride[0] < row) throw bad_arg(w + ": bad row count or overlapping rows");
  std::copy(t->shape, t->shape + t->ndim, v.shape);
  v.ld = t->stride[0];
  v.aligned = t->offset % uint64_t(ti.block > 1 ? 2 : ti.bytes) == 0;
  v.row_bytes = mul(v.ld, ti.bytes);
  const uint64_t span = v.rows ? uint64_t(mul(v.rows - 1, v.row_bytes)) + uint64_t(mul(row, ti.bytes)) : 0;
  if (t->offset > size || span > size - t->offset) throw bad_arg(w + ": reads past the end of its buffer");
  return v;
}

uint64_t buffer_bytes(const larch_buffer* b, const std::string& what) {
  if (!ours(b)) throw bad_arg(what + " is null or not a CPU tensor");
  return b->bytes;
}

view::view(const meta& m) : meta(m) {
  if (!buf) return;
  if (!ours(buf)) throw bad_arg("view: not a CPU buffer");  // a GPU backend's meta has no CPU memory
  elem_bytes = info(dtype).bytes;  // per element for plain types
  base = buf->host + offset;
}

static larch_status backend_info(larch_backend* out, uint32_t* runtime_version, uint32_t* driver_version) {
  *out = LARCH_BACKEND_CPU;
  *runtime_version = *driver_version = 0;
  return LARCH_OK;
}

static larch_status device_count(int32_t* out) {
  *out = 0;
  return LARCH_OK;
}

struct device : larch_device {
  uint32_t threads = 0;  // larch_device_set_threads'; 0 = one per logical CPU
};

// dispatch.cpp opens -1 through vv_cpu_ops() and -2 through vv_cpu_device_ops().
static larch_status device_open(int32_t ordinal, larch_device** out) {
  *out = new device{{ordinal == -2 ? vv_cpu_device_ops() : vv_cpu_ops()}};
  return LARCH_OK;
}

// The same on every machine, so a CPU run sizes its KV pool the same everywhere: a fixed 2 GiB budget,
// not the host's RAM. Nothing is subtracted as buffers are allocated.
// ponytail: the CPU device reports the reference's budget; the host's RAM once the planner sizes a CPU
// tier on it (v2.1).
static larch_status device_props_get(const larch_device* vd, larch_device_props* out) {
  const auto* d = static_cast<const device*>(vd);
  const bool reference = d->ops == vv_cpu_ops();
  *out = {};
  const char* name = reference ? "CPU reference" : "CPU";
  std::memcpy(out->name, name, std::strlen(name) + 1);
  out->backend = LARCH_BACKEND_CPU;
  out->warp_size = out->max_threads_per_group = 1;
  out->compute_units = reference ? 1 : d->threads ? d->threads : std::max(1u, std::thread::hardware_concurrency());
  out->vram_total = out->vram_free = out->recommended_working_set = 2ull << 30;
  return LARCH_OK;
}

static void device_close(larch_device* d) { delete static_cast<device*>(d); }

static larch_status device_set_threads(larch_device* d, uint32_t threads) {
  static_cast<device*>(d)->threads = threads;
  return LARCH_OK;
}

// Filled with 0xFF (NaN in every float dtype, -1 in i32), so a read before any write shows up as NaN
// instead of a quiet zero.
static larch_status buffer_alloc(larch_device* d, uint64_t bytes, larch_buffer** out) {
  if (!bytes) return vv_fail(LARCH_ERR_INVALID_ARG, "buffer_alloc: 0 bytes");
  try {
    auto* b = new buffer{{d->ops, bytes, nullptr}, std::unique_ptr<uint8_t[]>(new uint8_t[size_t(bytes)])};
    b->host = b->data.get();
    std::memset(b->host, 0xFF, size_t(bytes));
    *out = b;
  } catch (const std::bad_alloc&) {
    return vv_fail(LARCH_ERR_OOM, "buffer_alloc: " + std::to_string(bytes) + " bytes");
  }
  return LARCH_OK;
}

// A CPU device's memory is host memory already.
static larch_status buffer_alloc_host(larch_device* d, uint64_t bytes, larch_buffer** out, void** host) {
  const larch_status st = buffer_alloc(d, bytes, out);
  if (st == LARCH_OK) *host = (*out)->host;
  return st;
}

static larch_status buffer_wrap_host(larch_device*, void*, uint64_t, larch_buffer**) {
  return vv_fail(LARCH_ERR_UNSUPPORTED, "buffer_wrap_host: Metal only");
}

static void buffer_free(larch_buffer* b) { delete static_cast<buffer*>(b); }

// b is ours, host is set when bytes are moved, and [off, off + bytes) lies inside b.
static bool in_range(const larch_buffer* b, uint64_t off, const void* host, uint64_t bytes) {
  if (!ours(b) || (bytes && !host)) return false;
  return off <= b->bytes && bytes <= b->bytes - off;
}

static larch_status upload(larch_stream*, larch_buffer* b, uint64_t off, const void* src, uint64_t bytes) {
  if (!in_range(b, off, src, bytes)) return vv_fail(LARCH_ERR_INVALID_ARG, "upload: bad buffer, source or range");
  if (bytes) std::memcpy(b->host + off, src, size_t(bytes));
  return LARCH_OK;
}

static larch_status download(larch_stream*, const larch_buffer* b, uint64_t off, void* dst, uint64_t bytes) {
  if (!in_range(b, off, dst, bytes)) return vv_fail(LARCH_ERR_INVALID_ARG, "download: bad buffer, target or range");
  if (bytes) std::memcpy(dst, b->host + off, size_t(bytes));
  return LARCH_OK;
}

// Its device, for the threads the fast CPU backend's ops may use.
struct stream : larch_stream {
  const device* dev;
};

static larch_status stream_create(larch_device* d, larch_stream** out) {
  *out = new stream{{d->ops}, static_cast<const device*>(d)};
  return LARCH_OK;
}

uint32_t stream_threads(const larch_stream* s) {
  const uint32_t t = static_cast<const stream*>(s)->dev->threads;
  return t ? t : std::max(1u, std::thread::hardware_concurrency());
}

static larch_status stream_sync(larch_stream*) { return LARCH_OK; }
static void stream_destroy(larch_stream* s) { delete static_cast<stream*>(s); }

struct event : larch_event {
  uint64_t ns;
};

static larch_status event_record(larch_stream* s, larch_event** out) {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  *out = new event{{s->ops}, uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count())};
  return LARCH_OK;
}

static larch_status event_sync(const larch_event*) { return LARCH_OK; }
static larch_status stream_wait(larch_stream*, const larch_event*) { return LARCH_OK; }

static larch_status event_elapsed_ns(const larch_event* a, const larch_event* b, uint64_t* out) {
  if (!cpu_ops(b->ops)) return vv_fail(LARCH_ERR_INVALID_ARG, "event_elapsed_ns: b is not a CPU event");
  const uint64_t t0 = static_cast<const event*>(a)->ns, t1 = static_cast<const event*>(b)->ns;
  if (t1 < t0) return vv_fail(LARCH_ERR_INVALID_ARG, "event_elapsed_ns: b was recorded before a");
  *out = t1 - t0;
  return LARCH_OK;
}

static void event_destroy(larch_event* e) { delete static_cast<event*>(e); }

static larch_status graph_begin(larch_stream*) { return vv_fail(LARCH_ERR_UNSUPPORTED, "CPU reference: no graphs"); }
static larch_status graph_end(larch_stream*, larch_graph**) {
  return vv_fail(LARCH_ERR_UNSUPPORTED, "CPU reference: no graphs");
}
static larch_status graph_launch(larch_graph*, larch_stream*) {
  return vv_fail(LARCH_ERR_UNSUPPORTED, "CPU reference: no graphs");
}
static void graph_destroy(larch_graph*) {}

// ops.cpp
larch_status embed(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status rmsnorm(larch_stream*, const larch_tensor*, const larch_tensor*, float, const larch_tensor*);
larch_status rope(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_rope_params*);
larch_status mmvq(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                   const larch_tensor*);
larch_status quant_gemm_i8(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                            const larch_tensor*);
larch_status dequant(larch_stream*, const larch_tensor*, const larch_tensor*);
larch_status gemm(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                   const larch_tensor*, float, float);
larch_status silu_mul(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status add(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status kv_write(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                       const larch_kv_pool*);
larch_status attn_decode(larch_stream*, const larch_tensor*, const larch_kv_pool*, const larch_tensor*,
                          const larch_tensor*, uint32_t, float, const larch_tensor*);
larch_status attn_prefill(larch_stream*, const larch_tensor*, const larch_kv_pool*, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, float, const larch_tensor*);
larch_status qk_norm(larch_stream*, const larch_tensor*, const larch_tensor*, float);
larch_status sample_prep(larch_stream*, const larch_tensor*, const larch_sample_params*, const larch_tensor*,
                          const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status mmvq_verify(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                          const larch_tensor*);
larch_status mmvq_id(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                      const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status mmvq_id_mixed(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                            const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status sigmoid_mul(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status layernorm(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*, float,
                        const larch_tensor*);
larch_status gelu(larch_stream*, const larch_tensor*, const larch_tensor*);
larch_status rope_multi(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_rope_multi_params*);
larch_status attn_full(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*, uint32_t, float,
                        const larch_tensor*);
larch_status attn(larch_stream*, const larch_tensor*, const larch_kv_pool*, const larch_tensor*,
                   const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*, uint32_t,
                   uint32_t, float, const larch_tensor*);
larch_status sample(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                     const larch_draw_params*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                     const larch_tensor*);
larch_status moe_route(larch_stream*, const larch_tensor*, float, const larch_tensor*, const larch_tensor*);
larch_status gemm_ex(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*, uint32_t,
                      uint32_t, float, float);
larch_status rmsnorm_f32(larch_stream*, const larch_tensor*, const larch_tensor*, float, const larch_tensor*);
larch_status rmsnorm_backward(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*, float,
                               const larch_tensor*, uint32_t);
larch_status rope_f32(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_rope_params*);
larch_status silu_mul_f32(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status silu_mul_backward(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                                const larch_tensor*, const larch_tensor*);
larch_status softmax(larch_stream*, const larch_tensor*, float, uint32_t, const larch_tensor*);
larch_status softmax_backward(larch_stream*, const larch_tensor*, const larch_tensor*, float, const larch_tensor*);
larch_status draft_conv(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*, uint32_t,
                         const larch_tensor*);
// fast_cpu.cpp (lane c1)
larch_status fast_mmvq_id(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                           const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status fast_mmvq_id_mixed(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                                 const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
// gdn.cpp
larch_status gdn_conv(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_gdn_state*,
                       const larch_tensor*, const larch_tensor*, const larch_tensor*, const larch_tensor*);
larch_status gdn(larch_stream*, const larch_tensor*, const larch_tensor*, const larch_tensor*,
                  const larch_tensor*, const larch_tensor*, float, const larch_gdn_state*, const larch_tensor*,
                  const larch_tensor*, const larch_tensor*, const larch_tensor*);

}  // namespace cpu

static constexpr vv_ops table = {
    .backend_info = cpu::backend_info,
    .device_count = cpu::device_count,
    .device_open = cpu::device_open,
    .device_props_get = cpu::device_props_get,
    .device_close = cpu::device_close,
    .buffer_alloc = cpu::buffer_alloc,
    .buffer_wrap_host = cpu::buffer_wrap_host,
    .buffer_free = cpu::buffer_free,
    .upload = cpu::upload,
    .download = cpu::download,
    .stream_create = cpu::stream_create,
    .stream_sync = cpu::stream_sync,
    .stream_destroy = cpu::stream_destroy,
    .event_record = cpu::event_record,
    .event_elapsed_ns = cpu::event_elapsed_ns,
    .event_destroy = cpu::event_destroy,
    .graph_begin = cpu::graph_begin,
    .graph_end = cpu::graph_end,
    .graph_launch = cpu::graph_launch,
    .graph_destroy = cpu::graph_destroy,
    .embed = cpu::embed,
    .rmsnorm = cpu::rmsnorm,
    .rope = cpu::rope,
    .mmvq = cpu::mmvq,
    .dequant = cpu::dequant,
    .gemm = cpu::gemm,
    .silu_mul = cpu::silu_mul,
    .add = cpu::add,
    .kv_write = cpu::kv_write,
    .attn_decode = cpu::attn_decode,
    .attn_prefill = cpu::attn_prefill,
    .qk_norm = cpu::qk_norm,
    .sample_prep = cpu::sample_prep,
    .quant_gemm_i8 = cpu::quant_gemm_i8,
    .kv_dtypes = 1u << LARCH_F16 | 1u << LARCH_Q8_0 | 1u << LARCH_FP8_E4M3,
    .no_weights = 0,
    .device_set_threads = cpu::device_set_threads,
    .buffer_alloc_host = cpu::buffer_alloc_host,
    .event_sync = cpu::event_sync,
    .stream_wait = cpu::stream_wait,
    .mmvq_verify = cpu::mmvq_verify,
    .mmvq_id = cpu::mmvq_id,
    // The fast CPU device runs its own (fast_mmvq_id_mixed): its mmvq_id's sums for the hot experts.
    .mmvq_id_mixed = cpu::mmvq_id_mixed,
    .sigmoid_mul = cpu::sigmoid_mul,
    .attn = cpu::attn,
    .sample = cpu::sample,
    .moe_route = cpu::moe_route,
    .gdn_conv = cpu::gdn_conv,
    .gdn = cpu::gdn,
    .gemm_ex = cpu::gemm_ex,
    .rmsnorm_f32 = cpu::rmsnorm_f32,
    .rmsnorm_backward = cpu::rmsnorm_backward,
    .rope_f32 = cpu::rope_f32,
    .silu_mul_f32 = cpu::silu_mul_f32,
    .silu_mul_backward = cpu::silu_mul_backward,
    .softmax = cpu::softmax,
    .softmax_backward = cpu::softmax_backward,
    .layernorm = cpu::layernorm,
    .gelu = cpu::gelu,
    .attn_full = cpu::attn_full,
    .rope_multi = cpu::rope_multi,
    .fused_moe_gate_up = cpu::fused_moe_gate_up,
    .fused_moe_down = cpu::fused_moe_down,
    .fused_moe_in = cpu::fused_moe_in,
    .fused_gdn_in = cpu::fused_gdn_in,
    .fused_proj_add = cpu::fused_proj_add,
    .fused_gdn = cpu::fused_gdn,
    .fused_attn_in = cpu::fused_attn_in,
    .draft_conv = cpu::draft_conv,
};
// A table of its own, so handles know their device and the fast CPU backend (lane c1) can replace these
// slots without touching the reference's.
static constexpr vv_ops cpu_device = [] {
  vv_ops o = table;
  o.mmvq_id = cpu::fast_mmvq_id;
  o.mmvq_id_mixed = cpu::fast_mmvq_id_mixed;
  return o;
}();

const vv_ops* vv_cpu_ops() { return &table; }
const vv_ops* vv_cpu_device_ops() { return &cpu_device; }
