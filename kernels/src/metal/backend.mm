// Metal backend (architecture 6.4): device, props, buffers, streams, events and the ops of ops.metal.
// A stream is one MTLCommandQueue. Work encodes into its open command buffer; sync, download, event_record
// and the early commit (feed) commit it. Any command-buffer error is LARCH_ERR_DEVICE_LOST (no reset, D2).
// Every entry point is static and reached through vv_metal_ops() (common/backend.hpp).
#import <Metal/Metal.h>
#include <IOKit/IOKitLib.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/backend.hpp"
#include "cpu/cpu.h"  // the GGML block table, size checks and bad_arg the CPU reference's checks use
#include "metal/args.h"
#include "metal/mlx/backend/metal/kernels/steel/attn/params.h"  // MLX attention's AttnParams

extern "C" const unsigned char larch_metallib[];
extern "C" const std::size_t larch_metallib_len;
extern "C" const unsigned char larch_nax_metallib[];  // attn_full_nax.metal, built for macOS 26.2
extern "C" const std::size_t larch_nax_metallib_len;

/// Every kernel of ops.metal as a pipeline, built at open. A per-dtype slot is nil where the op has no
/// kernel for that dtype; the op's host checks rule those dtypes out first.
struct kernels {
  id<MTLComputePipelineState> embed[LARCH_Q3_K + 1], dequant[LARCH_Q3_K + 1], mmvq[LARCH_Q3_K + 1];
  id<MTLComputePipelineState> mmvq_mm[2][LARCH_Q3_K + 1];  // up to 8 and up to 16 rows
  id<MTLComputePipelineState> mmvq_ext[2][LARCH_Q3_K + 1];  // up to 4 and up to 8 rows
  id<MTLComputePipelineState> mmvq_mmc8_q3_k;  // Q3_K at 2-8 rows under LARCH_METAL_MMVQ_MMC
  id<MTLComputePipelineState> mmvq1[LARCH_Q3_K + 1];        // 1 row: Q3_K, Q4_K, Q5_K and Q6_K, nil for the others
  id<MTLComputePipelineState> mmvq_mma[2][LARCH_Q3_K + 1];  // up to 8 and up to 16 rows: the K-quants
  // mmvq_verify's staged tile at 8, 16 and 32 rows (vv_mmvq_st, staged.metal): Q3_K to Q8_0 on Apple GPU family 10,
  // nil elsewhere
  id<MTLComputePipelineState> mmvq_st[3][LARCH_Q3_K + 1];
  id<MTLComputePipelineState> gemm_q[2][2][LARCH_Q3_K + 1];  // [rows > GEMM_Q_M_SMALL][k <= n]; nil: none
  id<MTLComputePipelineState> mmvq_id[LARCH_Q3_K + 1], mmvq_id_grouped[LARCH_Q3_K + 1];
  id<MTLComputePipelineState> mmvq_id_mm[2][LARCH_Q3_K + 1];  // the tensor path's grouped mmvq_id, [64 x 32 tile]
  id<MTLComputePipelineState> mmvq_id1[LARCH_Q3_K + 1];  // Q3_K, Q4_K, Q5_K and Q6_K, nil for the others
  id<MTLComputePipelineState> mmvq_id1n[LARCH_Q3_K + 1];  // the same formats, for experts under 1024 wide
  id<MTLComputePipelineState> attn_flash[3];                 // head_dim up to 64, 128 and 256
  // The simdgroup-matrix kernels for GPUs without the tensor API; nil when LARCH_METAL_SIMD_KERNELS=0.
  id<MTLComputePipelineState> gemm_qs[LARCH_Q3_K + 1], mmvq_id_mm_sg[LARCH_Q3_K + 1];
  id<MTLComputePipelineState> attn_flash_sg[3];  // head_dim up to 64, 128 and 256
  id<MTLComputePipelineState> gemm, gemv, gemm_tensor, gemm_tensor_deep, rmsnorm, qk_norm, rope, silu_mul, add, kv_write, attn, attn_split,
      attn_split_reduce, sp_copy, sp_penalty, sp_part, sp_merge, moe_route, moe_group, moe_gather, moe_reduce,
      sigmoid_mul, gdn_conv, gdn, gdn1, layernorm, gelu, rope_multi, draft_conv, attn_in;
  id<MTLComputePipelineState> attn_full_sg[2], attn_full_nax;  // head_dim 64 and 72; NAX 64, nil without the tensor API
  // Paged prefill attention on NAX fragments, head_dim exactly 64, 128 and 256 (attn_full_nax.metal); nil without
  // macOS 26.2 and the tensor API, or with LARCH_METAL_ATTN_NAX=0
  id<MTLComputePipelineState> attn_paged_nax[3];
  id<MTLComputePipelineState> gdn_chunk[2];  // head_dim 64 and 128; nil when LARCH_METAL_GDN_CHUNK=0
};
// How a device's streams feed the GPU, read from the environment at device_open, which rejects a value other
// than 0 or 1:
// LARCH_METAL_CONCURRENT (default 1): dispatches share a concurrent encoder, with a barrier only where one binds
//   bytes that a dispatch since the last barrier wrote, or writes bytes one read. A serial encoder puts a barrier
//   between every two, and a decode step of a MoE model is about 1,000 small dispatches.
// LARCH_METAL_COMMIT (default 1): commit the open command buffer after 32 dispatches, then after every 128, so
//   the GPU starts while the host still encodes the step (llama.cpp's n_nodes_0; CUDA replays a graph instead).
//   64,256 was within noise of 32,128, and 0 cost 2.6 % MoE decode (M5, 26 Oct).
// LARCH_METAL_PROFILE (default 0): each dispatch commits alone and its GPU time adds to its kernel's row of a
//   table printed to stderr at exit. Profiling runs only: the gaps between dispatches are not timed.
// LARCH_METAL_SIMD_KERNELS (default 1): 0 leaves out the simdgroup-matrix prefill kernels (vv_gemm_qs,
//   vv_mmvq_id_mm_sg, vv_attn_flash_sg), so a GPU without the tensor API takes dequant + vv_gemm, mmvq_id_grouped
//   and vv_attn.
// LARCH_METAL_MOE_PREFETCH (default 1): the 64 x 32 grouped expert tile for Q3_K (gate/up prefill) loads the next
//   k step into registers while the current one multiplies (vv_mmvq_id_mm64x32p_*); 0 takes the plain tile, for
//   A/B timing on one build. With LARCH_METAL_PROFILE=1, the table also shows how many pairs each expert got.
// LARCH_METAL_GDN_WIDE (default 1): gdn steps of up to GDN_WIDE_ROWS rows a sequence on average (decode and
//   check steps) give each simdgroup one state row (vv_gdn1); 0 keeps GDN_ROWS rows a simdgroup, for A/B timing.
// LARCH_METAL_GDN_CHUNK (default 1): the other gdn steps (prefill) at head_dim 64 or 128 take vv_gdn_chunk, 8 rows
//   at a time on simdgroup matrices (MLX's chunked delta rule); 0 keeps vv_gdn's row-by-row walk, for A/B timing.
// LARCH_METAL_MOE_NARROW (default 1): mmvq_id at decode and check row counts on experts under 1024 wide takes
//   vv_mmvq_id1n, which keeps all 32 lanes busy; 0 keeps vv_mmvq_id1, for A/B timing.
// LARCH_METAL_MMVQ_MMC (default 0): Q3_K mmvq at 2-8 rows takes vv_mmvq_mmc8_q3_k (codes into the MMA, each group's
//   scale after its chain, as Splash's register tile) instead of vv_mmvq_mma8_q3_k; for A/B timing.
// LARCH_METAL_ATTN_NAX (default 1): flash prefill at head_dim 64, 128 or 256 takes vv_attn_paged_nax (MLX's NAX
//   attention over the paged pool, macOS 26.2 and the tensor API); 0 keeps vv_attn_flash, for A/B timing.
// LARCH_METAL_ATTN_NAX_HM (default 0): 1 launches vv_attn_paged_nax with the q head as grid x (the _hm kernels),
//   for A/B timing at long contexts.
struct feed {
  bool concurrent = true, commit = true, profile = false, simd_kernels = true, moe_prefetch = true, gdn_wide = true,
       moe_narrow = true, gdn_chunk = true, attn_nax = true, attn_nax_hm = false, mmvq_mmc = false;
};
static constexpr uint32_t first_commit = 32, commit_every = 128;
struct mtl_device : larch_device {
  id<MTLDevice> dev;
  larch_device_props props;
  kernels k;
  bool tensor_gemm;  // gemm uses vv_gemm_tensor (forced, or faster when timed at open)
  feed fed;          // read at open
};
struct mtl_buffer : larch_buffer {
  id<MTLBuffer> buf;
  uint64_t bytes;         // what the caller asked for; buf.length may be rounded up to whole pages
  bool readonly = false;  // wrap_host's: the GGUF mapping is read-only, and a GPU write to it is lost
  id<MTLResidencySet> resident = nil;  // keeps buf resident from allocation on (see resident())
};
struct mtl_stream : larch_stream {
  id<MTLCommandQueue> queue;
  id<MTLCommandBuffer> cb;                         // open, not committed; nil when nothing is encoded
  id<MTLCommandBuffer> last;                       // the last one committed; nil before the first
  NSMutableArray<id<MTLCommandBuffer>>* pending;  // committed since the last sync
  id<MTLBuffer> staging;                           // every copy's, made on the first; nil before
  uint64_t staged;                                 // blits not yet synced use [0, staged) of it
  const mtl_device* dev;
  id<MTLBuffer> scratch;  // sample_prep's penalised logits, grown on demand; nil before
  id<MTLBuffer> attn_scratch;  // split-K attention's f32 partials, grown on demand; nil before
  id<MTLBuffer> sp_parts;      // sample_prep's partials (vv_sp_part), grown on demand; nil before
  id<MTLBuffer> moe_scratch;   // grouped mmvq_id's pair lists and weighted partials, grown on demand; nil before
  id<MTLBuffer> st_parts, st_counters;  // mmvq_verify's K-split parts and per-tile counters (zero between calls); nil before
  id<MTLComputeCommandEncoder> enc;  // open on cb for every dispatch until a blit or commit; nil when none
  // A dequant this stream deferred (see dequant()): w's and out's buffers, offsets and row strides, w's shape and
  // format.
  struct lazy_dequant {
    id<MTLBuffer> w, out;
    bool w_readonly;
    uint64_t w_off, w_rs, out_off, out_rs;
    int64_t rows, cols;
    larch_dtype dtype;
    uint64_t out_end() const { return out_off + uint64_t(rows - 1) * out_rs + uint64_t(cols) * sizeof(uint16_t); }
  };
  std::vector<lazy_dequant> lazy;  // pending, outputs disjoint; none reads another's output or writes its w
  // A byte range a dispatch binds, [lo, hi) of buf; write unless the op marked it read-only (see arg).
  struct range {
    const void* buf;
    uint64_t lo, hi;
    bool write;
  };
  std::vector<range> since_barrier;  // concurrent encoder: what the dispatches since its last barrier bind
  uint32_t in_cb = 0;                // dispatches in cb, for the early commit
  bool committed_early = false;      // since the last sync: later command buffers take the longer run
  bool profiled = true;              // LARCH_METAL_PROFILE times its dispatches; pick_gemm's stream stays out
  // Drafter mode: the last dense GEMV was mmvq_verify (set there, cleared by mmvq and dequant), as in every step of a
  // guessing engine (larch.h's mmvq_verify) and none of a plain one; mmvq_id then runs per pair at any row count.
  bool check = false;
};
// An event sits between two command buffers: `prev`, the last one committed before it (nil if none),
// and `next`, the one later work encodes into. The time from a to b runs from the GPU starting a's next
// to it finishing b's prev, idle gaps between command buffers included, so the GPU waiting for the host
// counts, as in a CUDA event pair. a's next starts only once committed, so the time leaves out the host
// encoding it up to its early commit (feed), or all of it with LARCH_METAL_COMMIT=0.
struct mtl_event : larch_event {
  id<MTLCommandBuffer> prev, next;
};

// b as this backend's buffer, or nullptr for a null or foreign handle: a CPU reference buffer can reach
// a Metal stream, since section 15's tests keep both alive in one process.
static const mtl_buffer* as_mtl(const larch_buffer* b) {
  return b && b->ops == vv_metal_ops() ? static_cast<const mtl_buffer*>(b) : nullptr;
}

// feed's switches from the environment; INVALID_ARG names the first set to neither 0 nor 1.
static larch_status read_feed(feed* f) {
  for (const auto& [name, on] : {std::pair{"LARCH_METAL_CONCURRENT", &f->concurrent},
                                 std::pair{"LARCH_METAL_COMMIT", &f->commit},
                                 std::pair{"LARCH_METAL_PROFILE", &f->profile},
                                 std::pair{"LARCH_METAL_SIMD_KERNELS", &f->simd_kernels},
                                 std::pair{"LARCH_METAL_MOE_PREFETCH", &f->moe_prefetch},
                                 std::pair{"LARCH_METAL_GDN_WIDE", &f->gdn_wide},
                                 std::pair{"LARCH_METAL_MOE_NARROW", &f->moe_narrow},
                                 std::pair{"LARCH_METAL_GDN_CHUNK", &f->gdn_chunk},
                                 std::pair{"LARCH_METAL_ATTN_NAX", &f->attn_nax},
                                 std::pair{"LARCH_METAL_ATTN_NAX_HM", &f->attn_nax_hm},
                                 std::pair{"LARCH_METAL_MMVQ_MMC", &f->mmvq_mmc}}) {
    const char* c = std::getenv(name);
    if (c && std::strcmp(c, "0") && std::strcmp(c, "1"))
      return vv_fail(LARCH_ERR_INVALID_ARG, std::string(name) + " must be 0 or 1");
    if (c) *on = *c == '1';
  }
  return LARCH_OK;
}

// LARCH_METAL_PROFILE's table: per kernel name, calls and GPU seconds. Pipelines map to names at open.
struct profile_row {
  uint64_t calls = 0;
  double seconds = 0;
};
static std::mutex profile_lock;
static std::unordered_map<const void*, std::string>& pipeline_names() {
  static auto* m = new std::unordered_map<const void*, std::string>;
  return *m;
}
static std::unordered_map<std::string, profile_row>& profile_rows() {
  static auto* m = new std::unordered_map<std::string, profile_row>;
  return *m;
}
// Under LARCH_METAL_PROFILE, per grouped mmvq_id shape ("q3_k 768x2048, 2048 rows"): calls, and its experts
// by pairs (0, 1-31, 32-63, 64-127, 128-255, 256+) summed over the calls, with the 64-pair tile's fill (pairs
// over the rows its tiles multiply): the per-expert row counts a tile choice has to fit.
struct moe_hist_row {
  uint64_t calls = 0, pairs = 0, tile_rows = 0, bins[6] = {};
};
static std::map<std::string, moe_hist_row>& moe_hist_rows() {
  static auto* m = new std::map<std::string, moe_hist_row>;
  return *m;
}
static void print_moe_hist() {
  if (moe_hist_rows().empty()) return;
  std::fprintf(stderr, "larch metal moe pairs an expert (summed over calls)\n%-34s %6s %7s %7s %7s %7s %7s %7s %6s\n",
               "shape", "calls", "0", "1-31", "32-63", "64-127", "128-255", "256+", "fill64");
  for (const auto& [name, r] : moe_hist_rows())
    std::fprintf(stderr, "%-34s %6llu %7llu %7llu %7llu %7llu %7llu %7llu %5.1f%%\n", name.c_str(),
                 (unsigned long long)r.calls, (unsigned long long)r.bins[0], (unsigned long long)r.bins[1],
                 (unsigned long long)r.bins[2], (unsigned long long)r.bins[3], (unsigned long long)r.bins[4],
                 (unsigned long long)r.bins[5], r.tile_rows ? 100.0 * double(r.pairs) / double(r.tile_rows) : 0.0);
}
static void print_profile() {
  std::lock_guard<std::mutex> g(profile_lock);
  print_moe_hist();
  std::vector<std::pair<std::string, profile_row>> rows(profile_rows().begin(), profile_rows().end());
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.seconds > b.second.seconds; });
  double total = 0;
  uint64_t calls = 0;
  for (const auto& r : rows) total += r.second.seconds, calls += r.second.calls;
  std::fprintf(stderr, "larch metal profile: %llu dispatches, %.3f ms of kernel time\n%-36s %10s %12s %10s %7s\n",
               (unsigned long long)calls, total * 1e3, "kernel", "calls", "total ms", "us/call", "share");
  for (const auto& [name, r] : rows)
    std::fprintf(stderr, "%-36s %10llu %12.3f %10.2f %6.1f%%\n", name.c_str(), (unsigned long long)r.calls,
                 r.seconds * 1e3, r.seconds * 1e6 / double(std::max<uint64_t>(1, r.calls)),
                 total > 0 ? 100 * r.seconds / total : 0.0);
}

static uint32_t os_version() {  // 27.0 -> 27000, CUDA's encoding
  NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
  return uint32_t(v.majorVersion * 1000 + v.minorVersion * 10);
}

static larch_status backend_info(larch_backend* out, uint32_t* runtime_version, uint32_t* driver_version) {
  *out = LARCH_BACKEND_METAL;
  *runtime_version = *driver_version = os_version();  // the Metal runtime and driver ship with macOS
  return LARCH_OK;
}

// The devices at the 6.4 floor (Metal 4, unified memory); ordinals index this list.
static NSArray<id<MTLDevice>>* devices() {
  NSPredicate* floor = [NSPredicate predicateWithBlock:^BOOL(id<MTLDevice> d, NSDictionary*) {
    return [d supportsFamily:MTLGPUFamilyMetal4] && d.hasUnifiedMemory;
  }];
  return [MTLCopyAllDevices() filteredArrayUsingPredicate:floor];
}

static larch_status device_count(int32_t* out) {
  @autoreleasepool {
    *out = int32_t(devices().count);
  }
  return LARCH_OK;
}

// IORegistry's gpu-core-count, or 0 (unknown, like peak_gbps's 0) when the chip does not report it;
// launch shapes then derive their default from the other props (section 5).
static uint32_t gpu_cores(id<MTLDevice> dev) {
  io_registry_entry_t e = IOServiceGetMatchingService(kIOMainPortDefault, IORegistryEntryIDMatching(dev.registryID));
  id n = CFBridgingRelease(IORegistryEntrySearchCFProperty(e, kIOServicePlane, CFSTR("gpu-core-count"),
                                                           kCFAllocatorDefault,
                                                           kIORegistryIterateRecursively | kIORegistryIterateParents));
  IOObjectRelease(e);
  return [n isKindOfClass:NSNumber.class] ? [n unsignedIntValue] : 0;
}

// The suffix of a per-dtype kernel's name in ops.metal.
static const char* const dtype_name[] = {"f32",  "f16",  "bf16", "i32",  "q4_k",     "q5_k",
                                         "q6_k", "q8_0", "q4_0", "q5_0", "fp8_e4m3", "q3_k"};
static_assert(std::size(dtype_name) == LARCH_Q3_K + 1, "one name per larch_dtype");

// Apple's figures (6.4); the Rust side reports them as Estimated. Unknown chips get 0 (Missing).
static float peak_gbps(NSString* name, uint32_t cores) {
  const auto is = [&](NSString* chip) { return [name isEqualToString:chip]; };
  if (is(@"Apple M1")) return 68;
  if (is(@"Apple M2") || is(@"Apple M3")) return 100;
  if (is(@"Apple M4")) return 120;
  if (is(@"Apple M5")) return 153;
  if (is(@"Apple M3 Pro")) return 150;
  if (is(@"Apple M1 Pro") || is(@"Apple M2 Pro")) return 200;
  if (is(@"Apple M4 Pro")) return 273;
  if (is(@"Apple M5 Pro")) return 307;
  if (is(@"Apple M1 Max") || is(@"Apple M2 Max")) return 400;
  // An M3, M4 or M5 Max with fewer GPU cores has fewer memory channels; 0 cores is unknown.
  if (is(@"Apple M3 Max")) return cores >= 40 ? 400 : cores ? 300 : 0;
  if (is(@"Apple M4 Max")) return cores >= 40 ? 546 : cores ? 410 : 0;
  if (is(@"Apple M5 Max")) return cores >= 40 ? 614 : cores ? 460 : 0;
  if (is(@"Apple M1 Ultra") || is(@"Apple M2 Ultra")) return 800;
  if (is(@"Apple M3 Ultra")) return 819;
  return 0;
}

// Architecture 6.4: time one f16 GEMM of a prefill shape (a 512-token chunk through a 2048-wide layer) on
// each path, through the public op so it times exactly what prefill runs, and keep the faster.
static larch_status pick_gemm(mtl_device* d) {
  constexpr int64_t rows[3] = {512, 2048, 512}, cols[3] = {2048, 2048, 2048};  // a, b_t, c
  const std::vector<uint8_t> zeros(size_t(rows[1] * cols[1] * 2));
  larch_stream* s = nullptr;
  larch_buffer* buf[3] = {};
  larch_event* ev[4] = {};
  larch_tensor t[3] = {};
  uint64_t ns[2] = {};
  larch_status st = larch_stream_create(d, &s);
  if (!st) static_cast<mtl_stream*>(s)->profiled = false;
  for (int i = 0; i < 3 && !st; ++i) {
    const uint64_t bytes = uint64_t(rows[i] * cols[i]) * 2;
    st = larch_buffer_alloc(d, bytes, &buf[i]);
    if (!st) st = larch_upload(s, buf[i], 0, zeros.data(), bytes);
    t[i] = {buf[i], 0, LARCH_F16, 2, {rows[i], cols[i]}, {cols[i], 1}};
  }
  for (int path = 0; path < 2 && !st; ++path) {  // simdgroup, then tensor; one warm-up each
    d->tensor_gemm = path == 1;
    st = larch_gemm(s, &t[0], &t[1], nullptr, &t[2], 1, 0);
    if (!st) st = larch_event_record(s, &ev[2 * path]);
    if (!st) st = larch_gemm(s, &t[0], &t[1], nullptr, &t[2], 1, 0);
    if (!st) st = larch_event_record(s, &ev[2 * path + 1]);
  }
  if (!st) st = larch_stream_sync(s);
  for (int path = 0; path < 2 && !st; ++path) st = larch_event_elapsed_ns(ev[2 * path], ev[2 * path + 1], &ns[path]);
  d->tensor_gemm = !st && ns[1] < ns[0];
  for (larch_event* e : ev)
    if (e) larch_event_destroy(e);
  for (larch_buffer* b : buf)
    if (b) larch_buffer_free(b);
  if (s) larch_stream_destroy(s);
  return st;
}

static larch_status device_open(int32_t ordinal, larch_device** out) {
  @autoreleasepool {
    NSArray<id<MTLDevice>>* all = devices();
    if (ordinal < 0 || ordinal >= int32_t(all.count))
      return vv_fail(LARCH_ERR_INVALID_ARG,
                     "no Metal device " + std::to_string(ordinal) + " with Metal 4 and unified memory");
    id<MTLDevice> dev = all[NSUInteger(ordinal)];
    // LARCH_METAL_TENSOR=0|1 forces a GEMM path, and 0 also turns off flash prefill, the other tensor-API
    // kernel; unset, a family-10 device times both GEMM paths below (6.4).
    const char* const tensor_env = std::getenv("LARCH_METAL_TENSOR");
    if (tensor_env && std::strcmp(tensor_env, "0") && std::strcmp(tensor_env, "1"))
      return vv_fail(LARCH_ERR_INVALID_ARG, "LARCH_METAL_TENSOR must be 0 or 1");
    feed fed;
    if (larch_status st = read_feed(&fed)) return st;
    const bool tensor_capable = [dev supportsFamily:MTLGPUFamilyApple10];
    const bool tensor_forced = tensor_env && *tensor_env == '1', tensor_timed = !tensor_env && tensor_capable;
    const bool tensor_off = tensor_env && *tensor_env == '0';
    if (tensor_forced && !tensor_capable)
      return vv_fail(LARCH_ERR_UNSUPPORTED, "Metal tensor GEMM needs Apple GPU family 10 or later");

    // The metallib is static data, so Metal reads it in place (empty destructor).
    dispatch_data_t data = dispatch_data_create(larch_metallib, larch_metallib_len, nullptr, ^{});
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithData:data error:&err];
    if (!lib) return vv_fail(LARCH_ERR_DRIVER, std::string("larch.metallib: ") + err.localizedDescription.UTF8String);

    // Each kernel's pipeline, against the threadgroup size its launch below fixes. A missing kernel is a
    // build bug (a renamed or silently dropped instantiation). The first failure sticks.
    larch_status st = LARCH_OK;
    if (fed.profile) {
      static std::once_flag printer;
      std::call_once(printer, [] { std::atexit(print_profile); });
    }
    const auto named = [&fed](id<MTLComputePipelineState> ps, const std::string& name) {
      if (ps && fed.profile) {
        std::lock_guard<std::mutex> g(profile_lock);
        pipeline_names()[(__bridge const void*)ps] = name;
      }
      return ps;
    };
    // fc: the kernel's function constants, for MLX's attention; nil for ours, which have none.
    id<MTLLibrary> from = lib;  // the library pipe and opt take kernels from
    const auto fn = [&](const std::string& name, MTLFunctionConstantValues* fc) -> id<MTLFunction> {
      return fc ? [from newFunctionWithName:@(name.c_str()) constantValues:fc error:nil]
                : [from newFunctionWithName:@(name.c_str())];
    };
    const auto pipe = [&](const std::string& name, NSUInteger threads,
                          MTLFunctionConstantValues* fc = nil) -> id<MTLComputePipelineState> {
      if (st) return nil;
      id<MTLFunction> f = fn(name, fc);
      id<MTLComputePipelineState> ps = f ? [dev newComputePipelineStateWithFunction:f error:&err] : nil;
      if (!f)
        st = vv_fail(LARCH_ERR_DRIVER, "larch.metallib has no kernel " + name);
      else if (!ps)
        st = vv_fail(LARCH_ERR_DRIVER, name + ": " + err.localizedDescription.UTF8String);
      else if (ps.maxTotalThreadsPerThreadgroup < threads)
        st = vv_fail(LARCH_ERR_UNSUPPORTED, "Metal: " + name + " allows " +
                                                 std::to_string(ps.maxTotalThreadsPerThreadgroup) +
                                                 " threads per group, its launch uses " + std::to_string(threads));
      return named(ps, name);
    };
    // An optional kernel: nil when the device cannot build it or launch `threads` threads of it in 32-lane
    // simdgroups, and its caller keeps a slower path. A missing one is still a build bug.
    const auto opt = [&](const std::string& name, NSUInteger threads,
                         MTLFunctionConstantValues* fc = nil) -> id<MTLComputePipelineState> {
      if (st) return nil;
      id<MTLFunction> f = fn(name, fc);
      if (!f) st = vv_fail(LARCH_ERR_DRIVER, "larch.metallib has no kernel " + name);
      id<MTLComputePipelineState> ps = f ? [dev newComputePipelineStateWithFunction:f error:nil] : nil;
      const bool fits = ps.threadExecutionWidth == 32 && ps.maxTotalThreadsPerThreadgroup >= threads &&
                        dev.maxThreadsPerThreadgroup.width >= threads &&
                        ps.staticThreadgroupMemoryLength <= dev.maxThreadgroupMemoryLength;
      return fits ? named(ps, name) : nil;
    };
    kernels k{};
    for (const larch_dtype t :
         {LARCH_F32, LARCH_F16, LARCH_Q3_K, LARCH_Q4_K, LARCH_Q5_K, LARCH_Q6_K, LARCH_Q8_0, LARCH_Q4_0, LARCH_Q5_0}) {
      k.embed[t] = pipe(std::string("vv_embed_") + dtype_name[t], 256);  // each
      if (!cpu::is_quant(t)) continue;
      k.dequant[t] = pipe(std::string("vv_dequant_") + dtype_name[t], 256);  // each
      k.mmvq[t] = pipe(std::string("vv_mmvq_") + dtype_name[t], 128);
      k.mmvq_id[t] = pipe(std::string("vv_mmvq_id_") + dtype_name[t], 128);
      k.mmvq_id_grouped[t] = pipe(std::string("vv_mmvq_id_grouped_") + dtype_name[t], 128);
      if (t == LARCH_Q3_K || t == LARCH_Q4_K || t == LARCH_Q5_K || t == LARCH_Q6_K) {
        k.mmvq1[t] = pipe(std::string("vv_mmvq1_") + dtype_name[t], 32 * MMVQ1_SG);
        if (t == LARCH_Q3_K) k.mmvq_mmc8_q3_k = pipe("vv_mmvq_mmc8_q3_k", 32 * MMVQ_MMA_SG);
        k.mmvq_id1[t] = pipe(std::string("vv_mmvq_id1_") + dtype_name[t], 32 * MMVQ1_SG);
        if (fed.moe_narrow) k.mmvq_id1n[t] = opt(std::string("vv_mmvq_id1n_") + dtype_name[t], 32 * MMVQ1_SG);
      }
      // G1's best step (M5, 14 Nov): 32 inputs for the K-quants, 128 for Q8_0. All three row tiles or none, so a format
      // never changes kernel with the row count.
      if (tensor_capable && !tensor_off && t != LARCH_Q4_0 && t != LARCH_Q5_0) {
        auto& st_tiles = k.mmvq_st;
        for (int m = 0; m < 3; ++m)
          st_tiles[m][t] = opt(std::string("vv_mmvq_st") + (t == LARCH_Q8_0 ? "128" : "32") + "_m" +
                                   std::to_string(8 << m) + "_" + dtype_name[t],
                               32);
        if (!st_tiles[0][t] || !st_tiles[1][t] || !st_tiles[2][t]) st_tiles[0][t] = st_tiles[1][t] = st_tiles[2][t] = nil;
      }
      const bool k_quant = t == LARCH_Q3_K || t == LARCH_Q4_K || t == LARCH_Q5_K || t == LARCH_Q6_K;
      for (int wide = 0; wide < 2 && k_quant; ++wide)  // nil keeps the others
        k.mmvq_mma[wide][t] =
            opt(std::string(wide ? "vv_mmvq_mma16_" : "vv_mmvq_mma8_") + dtype_name[t], 32 * MMVQ_MMA_SG);
      for (int wide = 0; wide < 2; ++wide)
        k.mmvq_ext[wide][t] = pipe(std::string(wide ? "vv_mmvq_ext8_" : "vv_mmvq_ext4_") + dtype_name[t], 128);
      for (int tiles = 0; tiles < 2; ++tiles)  // nil keeps mmvq's one-column kernel
        k.mmvq_mm[tiles][t] = opt(std::string(tiles ? "vv_mmvq_mm16_" : "vv_mmvq_mm8_") + dtype_name[t], 128);
      if (fed.simd_kernels) {  // nil keeps dequant + vv_gemm, and mmvq_id_grouped
        k.gemm_qs[t] = opt(std::string("vv_gemm_qs_") + dtype_name[t], GEMM_QS_SG * 32);
        k.mmvq_id_mm_sg[t] = opt(std::string("vv_mmvq_id_mm_sg_") + dtype_name[t], GEMM_QS_SG * 32);
      }
    }
    k.gemm = pipe("vv_gemm", 128);
    k.gemv = pipe("vv_gemv", 32 * GEMV_SG);
    if (tensor_forced || tensor_timed) {  // timed and nil: simdgroup GEMM without a race
      k.gemm_tensor = opt("vv_gemm_tensor", 32 * GEMM_TENSOR_SG);
      if (!st && !k.gemm_tensor && tensor_forced)
        st = vv_fail(LARCH_ERR_UNSUPPORTED,
                     "Metal tensor GEMM has no " + std::to_string(GEMM_TENSOR_SG) + "-simdgroup launch");
    }
    k.rmsnorm = pipe("vv_rmsnorm", 256);
    k.qk_norm = pipe("vv_qk_norm", 256);
    k.rope = pipe("vv_rope", 256);  // each, as are the five below
    k.silu_mul = pipe("vv_silu_mul", 256);
    k.add = pipe("vv_add", 256);
    k.kv_write = pipe("vv_kv_write", 256);
    k.sp_copy = pipe("vv_sp_copy", 256);
    k.sp_penalty = pipe("vv_sp_penalty", 256);
    k.attn = pipe("vv_attn", 32);
    k.attn_split = pipe("vv_attn_split", 32);
    k.attn_split_reduce = pipe("vv_attn_split_reduce", 32);
    k.sp_part = pipe("vv_sp_part", SP_THREADS);
    k.sp_merge = pipe("vv_sp_merge", 32);  // sized from its pipeline, in whole simdgroups
    k.moe_route = pipe("vv_moe_route", 32);
    k.moe_group = pipe("vv_moe_group", 256);
    k.moe_gather = pipe("vv_moe_gather", 256);  // each
    k.moe_reduce = pipe("vv_moe_reduce", 256);
    k.sigmoid_mul = pipe("vv_sigmoid_mul", 256);
    k.layernorm = pipe("vv_layernorm", 256);
    k.gelu = pipe("vv_gelu", 256);  // each
    k.rope_multi = pipe("vv_rope_multi", 256);  // each
    k.draft_conv = pipe("vv_draft_conv", 256);  // each
    k.attn_in = pipe("vv_attn_in", 256);
    // MLX attention's constants: no mask, causal, sinks, window or lse; unaligned, so the last query and key
    // blocks always take the bounds-checked loads (a uniform branch per block, nothing on whole blocks).
    MTLFunctionConstantValues* dense = [MTLFunctionConstantValues new];
    const bool no = false;
    const int no_window = 0;
    for (NSUInteger i : {200, 201, 300, 301, 302, 303}) [dense setConstantValue:&no type:MTLDataTypeBool atIndex:i];
    [dense setConstantValue:&no_window type:MTLDataTypeInt atIndex:304];
    k.attn_full_sg[0] = pipe("vv_attn_full_sg64", 32 * ATTN_FULL_SG, dense);
    k.attn_full_sg[1] = pipe("vv_attn_full_sg72", 32 * ATTN_FULL_SG, dense);
    if (@available(macOS 26.2, *)) {
      if (tensor_capable && !tensor_off) {
        dispatch_data_t nax = dispatch_data_create(larch_nax_metallib, larch_nax_metallib_len, nullptr, ^{});
        from = [dev newLibraryWithData:nax error:&err];
        if (!from) return vv_fail(LARCH_ERR_DRIVER, std::string("larch_nax.metallib: ") + err.localizedDescription.UTF8String);
        k.attn_full_nax = opt("vv_attn_full_nax64", 32 * ATTN_FULL_SG, dense);
        for (int wide = 0; wide < 3 && fed.attn_nax; ++wide)  // 4, 4 and 2 x 4 simdgroups (attn_full_nax.metal)
          k.attn_paged_nax[wide] = opt("vv_attn_paged_nax" + std::to_string(64 << wide) + (fed.attn_nax_hm ? "_hm" : ""),
                                       wide == 2 ? 256 : 128);
        from = lib;
      }
    }
    k.gdn_conv = pipe("vv_gdn_conv", 256);
    k.gdn = pipe("vv_gdn", 32 * GDN_SG);
    k.gdn1 = pipe("vv_gdn1", 32 * GDN_SG);
    for (int wide = 0; wide < 2 && fed.gdn_chunk; ++wide)
      k.gdn_chunk[wide] = opt("vv_gdn_chunk" + std::to_string(64 << wide), 32 * GDN_SG);
    for (int wide = 0; wide < 3 && tensor_capable && !tensor_off; ++wide)  // nil keeps vv_attn
      k.attn_flash[wide] = opt("vv_attn_flash" + std::to_string(64 << wide), ATTN_FLASH_SG * 32);
    for (int wide = 0; wide < 3 && fed.simd_kernels; ++wide)  // nil keeps vv_attn
      k.attn_flash_sg[wide] = opt("vv_attn_flash_sg" + std::to_string(64 << wide), ATTN_FLASH_SG * 32);
    if (st) return st;
    if (k.attn.threadExecutionWidth != 32 || k.attn_split.threadExecutionWidth != 32 ||
        k.attn_split_reduce.threadExecutionWidth != 32)
      return vv_fail(LARCH_ERR_UNSUPPORTED, "Metal: attention SIMD width is not 32");

    larch_device_props p = {};
    std::strncpy(p.name, dev.name.UTF8String, sizeof p.name - 1);
    p.backend = LARCH_BACKEND_METAL;
    for (NSInteger f = MTLGPUFamilyApple1; [dev supportsFamily:MTLGPUFamily(f)]; ++f) p.arch_code = uint32_t(f);
    p.compute_units = gpu_cores(dev);
    p.warp_size = uint32_t(k.add.threadExecutionWidth);
    if (p.warp_size != 32)  // every kernel in ops.metal assumes 32 lanes, as on every Apple GPU so far
      return vv_fail(LARCH_ERR_UNSUPPORTED, "Metal: SIMD width " + std::to_string(p.warp_size) + ", not 32");
    p.max_threads_per_group = uint32_t(dev.maxThreadsPerThreadgroup.width);
    p.shared_mem_per_group = dev.maxThreadgroupMemoryLength;
    p.l2_bytes = 0;  // Metal does not expose it
    p.vram_total = p.recommended_working_set = dev.recommendedMaxWorkingSetSize;
    p.peak_gbps = peak_gbps(dev.name, p.compute_units);
    p.runtime_version = p.driver_version = os_version();
    p.flags = LARCH_CAP_UNIFIED | LARCH_CAP_BF16;
    auto* d = new mtl_device{{vv_metal_ops()}, dev, p, k, tensor_forced, fed};
    // A lost device takes the fault path (D2, 6.4); any other failed timing keeps the simdgroup GEMM.
    const larch_status timed = tensor_timed && k.gemm_tensor ? pick_gemm(d) : LARCH_OK;
    if (timed == LARCH_ERR_DEVICE_LOST) {
      delete d;
      return timed;
    }
    if (timed) d->tensor_gemm = false;
    if (d->tensor_gemm) {  // only for the tensor path: nil keeps the shallow tile, and the deferred dequant
      d->k.gemm_tensor_deep = opt("vv_gemm_tensor_deep", 32 * GEMM_TENSOR_DEEP_SG);
      for (const larch_dtype t :
           {LARCH_Q3_K, LARCH_Q4_K, LARCH_Q5_K, LARCH_Q6_K, LARCH_Q8_0, LARCH_Q4_0, LARCH_Q5_0})
        for (int v = 0; v < 4; ++v) {
          const int big = v / 2, narrow = v % 2;
          d->k.gemm_q[big][narrow][t] = opt(
              std::string(big ? "vv_gemm_q_" : "vv_gemm_q_small_") + (narrow ? "narrow_" : "") + dtype_name[t],
              GEMM_Q_SG * 32);
        }
      for (const larch_dtype t :
           {LARCH_Q3_K, LARCH_Q4_K, LARCH_Q5_K, LARCH_Q6_K, LARCH_Q8_0, LARCH_Q4_0, LARCH_Q5_0})
        for (int tall = 0; tall < 2; ++tall)
          d->k.mmvq_id_mm[tall][t] =
              opt(std::string(tall ? "vv_mmvq_id_mm64x32_" : "vv_mmvq_id_mm_") + dtype_name[t], MOE_MM_SG * 32);
      // The prefetching Q3_K tile replaces the plain one where it builds.
      if (fed.moe_prefetch)
        if (id<MTLComputePipelineState> pf = opt("vv_mmvq_id_mm64x32p_q3_k", MOE_MM_SG * 32))
          d->k.mmvq_id_mm[1][LARCH_Q3_K] = pf;
      if (st) {
        delete d;
        return st;
      }
    }
    *out = d;
  }
  return LARCH_OK;
}

static larch_status device_props_get(const larch_device* vd, larch_device_props* out) {
  const auto* d = static_cast<const mtl_device*>(vd);
  *out = d->props;
  out->vram_free = out->vram_total - std::min<uint64_t>(d->dev.currentAllocatedSize, out->vram_total);
  return LARCH_OK;
}

static void device_close(larch_device* d) { delete static_cast<mtl_device*>(d); }

// Metal maps an allocation's pages when a command buffer first uses it: for m1's 2 x 4.3 GB KV pool and
// its 4.9 GB mapping, 220 to 270 ms inside the first step (TTFT 545 to 320 ms on 52 tokens, M5, 10 Oct).
// Requesting residency here moves that to load, as cudaMalloc commits memory up front. nil when the set
// cannot be made: the buffer works as before and pays on first use.
static id<MTLResidencySet> resident(id<MTLDevice> dev, id<MTLBuffer> b) {
  id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:nil];
  [rs addAllocation:b];
  [rs commit];
  [rs requestResidency];
  return rs;
}

static larch_status buffer_alloc(larch_device* vd, uint64_t bytes, larch_buffer** out) {
  auto* d = static_cast<mtl_device*>(vd);
  if (!bytes) return vv_fail(LARCH_ERR_INVALID_ARG, "buffer_alloc: 0 bytes");
  // Checked here: past maxBufferLength the validation layer aborts instead of returning nil.
  if (bytes > d->dev.maxBufferLength)
    return vv_fail(LARCH_ERR_OOM, "buffer_alloc: " + std::to_string(bytes) + " bytes > maxBufferLength " +
                                       std::to_string(d->dev.maxBufferLength));
  id<MTLBuffer> b = [d->dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
  if (!b) return vv_fail(LARCH_ERR_OOM, "buffer_alloc: " + std::to_string(bytes) + " bytes");
  *out = new mtl_buffer{{vv_metal_ops()}, b, bytes, false, resident(d->dev, b)};
  return LARCH_OK;
}

// Wraps a mapping without copying. `host` must be page-aligned (the start of an mmap); tensors in it
// are addressed by offset. The length is rounded up to whole pages, which the mapping already covers.
// The buffer is read-only: upload rejects it, and so must every op's output.
static larch_status buffer_wrap_host(larch_device* vd, void* host, uint64_t bytes, larch_buffer** out) {
  auto* d = static_cast<mtl_device*>(vd);
  const uint64_t page = uint64_t(getpagesize());
  if (!host || !bytes || uintptr_t(host) % page)
    return vv_fail(LARCH_ERR_INVALID_ARG, "wrap_host: needs a page-aligned address and a nonzero length");
  // ponytail: one buffer per mapping; split it like llama.cpp if a model exceeds maxBufferLength.
  const std::string what = "wrap_host: " + std::to_string(bytes) + " bytes (maxBufferLength " +
                           std::to_string(d->dev.maxBufferLength) + ")";
  if (bytes > d->dev.maxBufferLength) return vv_fail(LARCH_ERR_OOM, what);  // also keeps the rounding below
  id<MTLBuffer> b = [d->dev newBufferWithBytesNoCopy:host
                                              length:(bytes + page - 1) / page * page
                                             options:MTLResourceStorageModeShared
                                         deallocator:nil];
  if (!b) return vv_fail(LARCH_ERR_OOM, what);
  *out = new mtl_buffer{{vv_metal_ops()}, b, bytes, true, resident(d->dev, b)};
  return LARCH_OK;
}

static void buffer_free(larch_buffer* b) { delete static_cast<mtl_buffer*>(b); }

// b is ours, host is set when bytes are moved, and [off, off + bytes) lies inside what was allocated.
static bool in_range(const mtl_buffer* b, uint64_t off, const void* host, uint64_t bytes) {
  return b && (host || !bytes) && off <= b->bytes && bytes <= b->bytes - off;
}

static id<MTLCommandBuffer> current(mtl_stream* s) {
  if (!s->cb) s->cb = [s->queue commandBuffer];
  return s->cb;
}

// Ends the stream's compute encoder before a blit or a commit. One encoder per command buffer serves every
// dispatch: a serial one (LARCH_METAL_CONCURRENT=0) runs them in order, and in the default concurrent one only
// encode's barriers order them. +3 % decode tok/s on m1 (M5, 9 Oct), TTFT unchanged, against an encoder per
// dispatch.
static void end_compute(mtl_stream* s) {
  [s->enc endEncoding];
  s->enc = nil;
  s->since_barrier.clear();  // Metal orders encoders that share a tracked buffer
}

static void settle(mtl_stream* s);

// Whether a dispatch or copy that binds buffer b must run the deferred dequants first: b holds an output, or a w
// that is writable (a GGUF mapping is read-only, so reading other weights from it never does).
static bool touches_lazy(const mtl_stream* s, id<MTLBuffer> b) {
  for (const auto& l : s->lazy)
    if (b == l.out || (b == l.w && !l.w_readonly)) return true;
  return false;
}

// Commits the open command buffer as it stands. The deferred dequants stay deferred: they encode into a later
// command buffer of the same queue, still before anything that binds their output (encode, blit).
static void commit(mtl_stream* s) {
  if (!s->cb) return;
  end_compute(s);
  [s->cb commit];
  [s->pending addObject:s->cb];
  s->last = s->cb;
  s->cb = nil;
  s->in_cb = 0;
}

// Commits the open command buffer, a deferred dequant included: another stream or the host sees the result.
static void flush(mtl_stream* s) {
  settle(s);
  commit(s);
}

static larch_status wait(id<MTLCommandBuffer> cb) {
  [cb waitUntilCompleted];
  if (cb.status == MTLCommandBufferStatusError)
    return vv_fail(LARCH_ERR_DEVICE_LOST,
                std::string("Metal command buffer: ") + cb.error.localizedDescription.UTF8String);
  return LARCH_OK;
}

static larch_status stream_create(larch_device* vd, larch_stream** out) {
  auto* d = static_cast<mtl_device*>(vd);
  id<MTLCommandQueue> q = [d->dev newCommandQueue];
  if (!q) return vv_fail(LARCH_ERR_DRIVER, "newCommandQueue failed");
  *out = new mtl_stream{{vv_metal_ops()}, q, nil, nil, [NSMutableArray new], nil, 0, d, nil, nil, nil, nil, nil, nil,
                        nil, {}, {}, 0, false};
  return LARCH_OK;
}

static larch_status stream_sync(larch_stream* vs) {
  auto* s = static_cast<mtl_stream*>(vs);
  larch_status st = LARCH_OK;
  @autoreleasepool {
    flush(s);
    for (id<MTLCommandBuffer> cb in s->pending) {
      larch_status e = wait(cb);
      if (st == LARCH_OK) st = e;
    }
    [s->pending removeAllObjects];
    s->staged = 0;
    s->committed_early = false;
  }
  return st;
}

static void stream_destroy(larch_stream* vs) {
  auto* s = static_cast<mtl_stream*>(vs);
  flush(s);  // encoded work still runs, as with a destroyed CUDA stream
  delete s;
}

// Copies through the stream's one 64 MiB staging buffer, filled from the front until the next sync. A
// copy that finds it full syncs the stream first, which waits for everything queued and returns any
// earlier fault. So a model loaded in many uploads holds 64 MiB of staging, not a second copy of
// itself, and no copy of any size needs a buffer near maxBufferLength.
static const uint64_t staging_cap = 64 << 20;

// Reserves n <= staging_cap bytes of the staging buffer, at *at.
static larch_status stage(mtl_stream* s, uint64_t n, uint64_t* at) {
  if (!s->staging) s->staging = [s->queue.device newBufferWithLength:staging_cap options:MTLResourceStorageModeShared];
  if (!s->staging) return vv_fail(LARCH_ERR_OOM, "staging buffer");
  if (s->staged + n > staging_cap) {
    if (larch_status st = stream_sync(s)) return st;
  }
  *at = s->staged;
  s->staged += n;
  return LARCH_OK;
}

static void blit(mtl_stream* s, id<MTLBuffer> from, uint64_t from_off, id<MTLBuffer> to, uint64_t to_off, uint64_t n) {
  if (touches_lazy(s, from) || touches_lazy(s, to)) settle(s);
  end_compute(s);
  id<MTLBlitCommandEncoder> e = [current(s) blitCommandEncoder];
  [e copyFromBuffer:from sourceOffset:from_off toBuffer:to destinationOffset:to_off size:n];
  [e endEncoding];
  // Under LARCH_METAL_PROFILE a copy commits alone too, so its time does not land on the next dispatch's row.
  if (s->dev->fed.profile && s->profiled) {
    const std::string name = n >= (1u << 20) ? "blit >= 1 MiB" : "blit < 1 MiB";
    [s->cb addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      if (cb.status != MTLCommandBufferStatusCompleted) return;
      std::lock_guard<std::mutex> g(profile_lock);
      profile_row& r = profile_rows()[name];
      r.calls += 1;
      r.seconds += std::max(0.0, cb.GPUEndTime - cb.GPUStartTime);
    }];
    commit(s);
  }
}

// Staged, so the copy lands in stream order and `src` can be reused on return (like cudaMemcpyAsync
// from pageable memory).
static larch_status upload(larch_stream* vs, larch_buffer* vb, uint64_t off, const void* src, uint64_t bytes) {
  auto* s = static_cast<mtl_stream*>(vs);
  const mtl_buffer* b = as_mtl(vb);
  if (!in_range(b, off, src, bytes) || b->readonly)
    return vv_fail(LARCH_ERR_INVALID_ARG, "upload: bad buffer, source or range, or a read-only wrapped buffer");
  @autoreleasepool {
    for (uint64_t at = 0; at < bytes;) {
      const uint64_t n = std::min(staging_cap, bytes - at);
      uint64_t st = 0;
      if (larch_status e = stage(s, n, &st)) return e;
      std::memcpy(static_cast<uint8_t*>(s->staging.contents) + st, static_cast<const uint8_t*>(src) + at, n);
      blit(s, s->staging, st, b->buf, off + at, n);
      at += n;
    }
  }
  return LARCH_OK;
}

// Synchronous: each chunk waits for everything queued on the stream, then is copied out.
static larch_status download(larch_stream* vs, const larch_buffer* vb, uint64_t off, void* dst, uint64_t bytes) {
  auto* s = static_cast<mtl_stream*>(vs);
  const mtl_buffer* b = as_mtl(vb);
  if (!in_range(b, off, dst, bytes)) return vv_fail(LARCH_ERR_INVALID_ARG, "download: bad buffer, target or range");
  @autoreleasepool {
    for (uint64_t at = 0; at < bytes;) {
      const uint64_t n = std::min(staging_cap, bytes - at);
      uint64_t st = 0;
      if (larch_status e = stage(s, n, &st)) return e;
      blit(s, b->buf, off + at, s->staging, st, n);
      if (larch_status e = stream_sync(s)) return e;
      std::memcpy(static_cast<uint8_t*>(dst) + at, static_cast<const uint8_t*>(s->staging.contents) + st, n);
      at += n;
    }
  }
  return LARCH_OK;
}

// Commits the stream's open command buffer and opens the next one (see mtl_event).
static larch_status event_record(larch_stream* vs, larch_event** out) {
  auto* s = static_cast<mtl_stream*>(vs);
  @autoreleasepool {
    flush(s);
    *out = new mtl_event{{vv_metal_ops()}, s->last, current(s)};
  }
  return LARCH_OK;
}

// Blocks until both ends are complete. a's next is committed once later work or b's record on the
// same stream flushed it; otherwise waiting would never return, so that is INVALID_ARG.
static larch_status event_elapsed_ns(const larch_event* va, const larch_event* vb, uint64_t* out) {
  const auto* a = static_cast<const mtl_event*>(va);
  const auto* b = static_cast<const mtl_event*>(vb);
  if (a == b) return *out = 0, LARCH_OK;
  if (b->ops != vv_metal_ops() || !b->prev || b->prev.commandQueue != a->next.commandQueue ||
      a->next.status < MTLCommandBufferStatusCommitted)
    return vv_fail(LARCH_ERR_INVALID_ARG, "event_elapsed_ns: b is not a later Metal event on a's stream");
  if (larch_status st = wait(a->next)) return st;
  if (larch_status st = wait(b->prev)) return st;
  const double s = b->prev.GPUEndTime - a->next.GPUStartTime;
  if (s < 0) return vv_fail(LARCH_ERR_INVALID_ARG, "event_elapsed_ns: b completed before a");
  *out = uint64_t(s * 1e9);
  return LARCH_OK;
}

static void event_destroy(larch_event* e) { delete static_cast<mtl_event*>(e); }

// Graphs are UNSUPPORTED on day 1 (6.4: indirect command buffers are a day-2 stretch). Parameters are
// deduced from the vv_ops slot each fills.
template <class... A>
static larch_status no_graphs(A...) {
  return vv_fail(LARCH_ERR_UNSUPPORTED, "Metal: no graphs");
}
static void graph_destroy(larch_graph*) {}  // graph_end never returns a graph

// Ops share the CPU reference's metadata checks, then apply Metal's refusals before encoding.

namespace {  // generic names (view, arg) that other translation units also define: keep them out of the link

struct unsupported : std::runtime_error {  // a valid call this backend cannot run: LARCH_ERR_UNSUPPORTED
  using std::runtime_error::runtime_error;
};

template <class F>
larch_status run(const char* op, F&& f) {
  try {
    f();
    return LARCH_OK;
  } catch (const cpu::bad_arg& e) {
    return vv_fail(LARCH_ERR_INVALID_ARG, std::string(op) + ": " + e.what());
  } catch (const unsupported& e) {
    return vv_fail(LARCH_ERR_UNSUPPORTED, std::string(op) + ": " + e.what());
  } catch (const std::bad_alloc&) {
    return vv_fail(LARCH_ERR_OOM, std::string(op) + ": out of memory");
  }
}

void need(bool ok, const char* what) {
  if (!ok) throw cpu::bad_arg(what);
}

// A checked tensor: rows x cols (cols = the product of the dims after the first), rows rs bytes apart.
struct view : cpu::meta {
  const mtl_buffer* b = nullptr;
  uint64_t off = 0, rs = 0;
  bool misaligned = false;  // offset not a multiple of the element size (2 bytes for quant blocks)

  view() = default;
  view(const cpu::meta& m)
      : cpu::meta(m), b(static_cast<const mtl_buffer*>(m.buf)), off(m.offset), rs(uint64_t(m.row_bytes)),
        misaligned(!m.aligned) {}
};

cpu::sizer sizer(larch_stream* vs) {
  const auto* s = static_cast<mtl_stream*>(vs);
  return [s](const larch_buffer* buf, const std::string& what) -> uint64_t {
    const mtl_buffer* b = as_mtl(buf);
    if (!b) throw cpu::bad_arg(what + " is null or not a Metal tensor");
    if (b->buf.device != s->dev->dev) throw cpu::bad_arg(what + " is on another Metal device");
    return b->bytes;
  };
}

void writable(const view& v) { need(!v.b->readonly, "an output is a read-only wrapped buffer"); }

struct arg {  // a kernel buffer argument: bytes [off, off + len) of buf, which the kernel writes unless ro
  id<MTLBuffer> buf;
  uint64_t off;
  uint64_t len = UINT64_MAX;  // the rest of the buffer
  bool ro = false;
};
// The kernels address tensors through typed pointers, so a misaligned offset would read and write the
// wrong bytes. larch.h has no alignment rule yet, so such a call is valid, and this backend cannot run it.
// Called only after every check, so a call that is also invalid still gets INVALID_ARG.
arg at(const view& v) {
  if (v.misaligned) throw unsupported("a tensor offset is not a multiple of its element size (2 bytes for quant)");
  return {v.b->buf, v.off, uint64_t(v.rows) * v.rs};  // whole row strides: never short of what the kernel touches
}
arg at_or(const view& v, const arg& unused) { return v.b ? at(v) : unused; }  // optional tensors
// An input: the op's kernel only reads these bytes, so dispatches that read them may overlap.
arg input(arg a) {
  a.ro = true;
  return a;
}
arg input(const view& v) { return input(at(v)); }
arg input_or(const view& v, const arg& unused) { return input(at_or(v, unused)); }

// Encodes one dispatch of pso: bufs at indices 0.., args after them. grid counts threads (per_thread) or
// threadgroups of tg threads. An empty grid encodes nothing.
template <class A>
void encode(mtl_stream* s, id<MTLComputePipelineState> pso, std::initializer_list<arg> bufs, const A& args,
            MTLSize grid, MTLSize tg, bool per_thread) {
  if (!grid.width || !grid.height || !grid.depth) return;
  for (const arg& b : bufs)
    if (touches_lazy(s, b.buf)) {
      settle(s);
      break;
    }
  const feed& f = s->dev->fed;
  @autoreleasepool {
    if (!s->enc)
      s->enc = f.concurrent ? [current(s) computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                            : [current(s) computeCommandEncoder];
    id<MTLComputeCommandEncoder> e = s->enc;
    if (f.concurrent) {
      // A barrier where this dispatch binds bytes an earlier one since the last barrier writes, or writes
      // bytes one reads. An arg is a write unless its op marked it read-only (input), so a missed mark costs
      // concurrency, never a race.
      // ponytail: linear scan of the window; index it by buffer if it passes ~256 entries in a profile.
      const size_t before = s->since_barrier.size();
      bool hazard = false;
      for (const arg& b : bufs) {
        const mtl_stream::range m{(__bridge const void*)b.buf, b.off,
                                  b.len > UINT64_MAX - b.off ? UINT64_MAX : b.off + b.len, !b.ro};
        for (size_t i = 0; i < before && !hazard; ++i) {
          const auto& r = s->since_barrier[i];
          hazard = r.buf == m.buf && r.lo < m.hi && m.lo < r.hi && (r.write || m.write);
        }
        s->since_barrier.push_back(m);
      }
      if (hazard) {
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        s->since_barrier.erase(s->since_barrier.begin(), s->since_barrier.begin() + ptrdiff_t(before));
      }
    }
    [e setComputePipelineState:pso];
    NSUInteger i = 0;
    for (const arg& b : bufs) [e setBuffer:b.buf offset:b.off atIndex:i++];
    [e setBytes:&args length:sizeof args atIndex:i];
    if (per_thread)
      [e dispatchThreads:grid threadsPerThreadgroup:tg];
    else
      [e dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    ++s->in_cb;
    if (f.profile) {
      if (s->profiled) {
        const void* key = (__bridge const void*)pso;
        // On completion, so work still running at a sync-free destroy counts too; a faulted buffer's times don't.
        [s->cb addCompletedHandler:^(id<MTLCommandBuffer> cb) {
          if (cb.status != MTLCommandBufferStatusCompleted) return;
          std::lock_guard<std::mutex> g(profile_lock);
          profile_row& r = profile_rows()[pipeline_names()[key]];
          r.calls += 1;
          r.seconds += std::max(0.0, cb.GPUEndTime - cb.GPUStartTime);
        }];
      }
      commit(s);
    } else if (f.commit && s->in_cb >= (s->committed_early ? commit_every : first_commit)) {
      commit(s);
      s->committed_early = true;
    }
  }
}

// One thread per point of grid (x, y, z), in threadgroups of up to 256.
template <class A>
void each(mtl_stream* s, id<MTLComputePipelineState> pso, std::initializer_list<arg> bufs, const A& args,
          int64_t x, int64_t y, int64_t z = 1) {
  const NSUInteger w = NSUInteger(std::clamp<int64_t>(x, 1, 256)), h = NSUInteger(std::clamp<int64_t>(y, 1, 256 / w));
  encode(s, pso, bufs, args, MTLSizeMake(NSUInteger(x), NSUInteger(y), NSUInteger(z)), MTLSizeMake(w, h, 1), true);
}

// n_groups threadgroups of `threads` threads along x (by y).
template <class A>
void groups(mtl_stream* s, id<MTLComputePipelineState> pso, std::initializer_list<arg> bufs, const A& args,
            int64_t n_groups, NSUInteger threads, int64_t y = 1) {
  encode(s, pso, bufs, args, MTLSizeMake(NSUInteger(n_groups), NSUInteger(y), 1), MTLSizeMake(threads, 1, 1), false);
}

}  // namespace

// Encodes deferred dequant l where the stream now stands. The others stay pending: no output of theirs meets l's.
static void run_lazy(mtl_stream* s, const mtl_stream::lazy_dequant& l) {
  auto rest = std::exchange(s->lazy, {});  // so that encode's hook does not run them too
  each(s, s->dev->k.dequant[l.dtype], {input(arg{l.w, l.w_off}), {l.out, l.out_off}}, ew_args{l.w_rs, l.out_rs, 0, 0},
       l.cols / 32, l.rows);
  s->lazy = std::move(rest);
}

// Encodes every deferred dequant where the stream now stands.
static void settle(mtl_stream* s) {
  for (const auto& l : std::exchange(s->lazy, {})) run_lazy(s, l);
}

namespace {

// GPU cores: IORegistry's gpu-core-count; a chip that does not report it gets a derived stand-in, the
// simdgroups one threadgroup of pso can hold.
uint32_t core_count(const mtl_stream* s, id<MTLComputePipelineState> pso) {
  const NSUInteger limit = std::min(pso.maxTotalThreadsPerThreadgroup, s->dev->dev.maxThreadsPerThreadgroup.width);
  const uint32_t reported = s->dev->props.compute_units;
  return reported ? reported : uint32_t(std::max<NSUInteger>(1, limit / pso.threadExecutionWidth));
}

// The pipelines of the stream's device.
const kernels& ks(larch_stream* vs) { return static_cast<mtl_stream*>(vs)->dev->k; }

// The MPP tensors index in int: a tile's start (row * row stride) must fit, not just each factor.
bool tensor_gemm_eligible(const mtl_stream* stream, const view& a, const view& bt) {
  constexpr uint64_t max_int = uint64_t(std::numeric_limits<int>::max());
  const auto fits = [&](const view& v) {
    return v.rows <= int64_t(max_int) && !(v.rs % sizeof(uint16_t)) && v.rs / sizeof(uint16_t) <= max_int &&
           uint64_t(v.rows) * (v.rs / sizeof(uint16_t)) <= max_int;
  };
  return stream->dev->tensor_gemm && a.rows > 16 && a.cols <= int64_t(max_int) && fits(a) && fits(bt);
}

// A norm's threadgroup: whole simdgroups, at most 256 threads, no more than n needs.
NSUInteger norm_threads(int64_t n) { return NSUInteger(std::min<int64_t>(256, (n + 31) / 32 * 32)); }

larch_status embed(larch_stream* vs, const larch_tensor* table, const larch_tensor* ids_i32,
                    const larch_tensor* out_f32) {
  return run("embed", [&] {
    const auto checked = cpu::check_embed(sizer(vs), table, ids_i32, out_f32);
    const view t = checked.table, ids = checked.ids, o = checked.out;
    writable(o);
    const int64_t per_thread = cpu::is_quant(t.dtype) ? 32 : 1;
    each(static_cast<mtl_stream*>(vs), ks(vs).embed[t.dtype], {input(t), input(ids), at(o)},
         ew_args{t.rs, ids.rs, o.rs, 0}, t.cols / per_thread, ids.rows);
  });
}

larch_status rmsnorm(larch_stream* vs, const larch_tensor* x, const larch_tensor* w, float eps,
                      const larch_tensor* out) {
  return run("rmsnorm", [&] {
    const auto checked = cpu::check_rmsnorm(sizer(vs), x, w, out);
    const view xv = checked.x, wv = checked.w, o = checked.out;
    writable(o);
    groups(static_cast<mtl_stream*>(vs), ks(vs).rmsnorm, {input(xv), input(wv), at(o)},
           rms_args{xv.rs, wv.rs, o.rs, u32(xv.cols), 1, eps}, xv.rows, norm_threads(xv.cols));
  });
}

// vv_rope's arguments: ggml's rope_yarn constants, computed as the CPU reference computes them.
rope_args rope_arguments(const cpu::rope_checked& c, const larch_rope_params* p) {
  const view q = c.qk, pos = c.pos, ff = c.freq_factors;
  const int64_t nd = p->n_dims;
  const auto corr_dim = [&](float n_rot) {
    return float(nd) * logf(float(p->n_ctx_orig) / (n_rot * 2 * float(M_PI))) / (2 * logf(p->freq_base));
  };
  const float lo = std::max(0.0f, floorf(corr_dim(p->beta_fast)));
  const float hi = std::min(float(nd - 1), ceilf(corr_dim(p->beta_slow)));
  float mscale = p->attn_factor;
  if (p->ext_factor != 0.0f) mscale *= 1.0f + 0.1f * logf(1.0f / p->freq_scale);
  return rope_args{q.rs, pos.rs, ff.rs, u32(q.shape[1]), u32(q.shape[2]), u32(nd), p->neox ? 1u : 0u,
                   p->freq_factors ? 1u : 0u, powf(p->freq_base, -2.0f / float(nd)), p->freq_scale,
                   p->ext_factor, mscale, lo, hi};
}

larch_status rope(larch_stream* vs, const larch_tensor* qk, const larch_tensor* pos_i32,
                   const larch_rope_params* p) {
  return run("rope", [&] {
    const auto checked = cpu::check_rope(sizer(vs), qk, pos_i32, p);
    const view q = checked.qk, pos = checked.pos, ff = checked.freq_factors;
    writable(q);
    each(static_cast<mtl_stream*>(vs), ks(vs).rope, {at(q), input(pos), input_or(ff, at(pos))},
         rope_arguments(checked, p), p->n_dims / 2, q.shape[1], q.rows);
  });
}

larch_status layernorm(larch_stream* vs, const larch_tensor* x, const larch_tensor* w, const larch_tensor* b_or_null,
                        float eps, const larch_tensor* out) {
  return run("layernorm", [&] {
    const auto checked = cpu::check_layernorm(sizer(vs), x, w, b_or_null, out);
    const view xv = checked.x, wv = checked.w, bv = checked.b, o = checked.out;
    writable(o);
    groups(static_cast<mtl_stream*>(vs), ks(vs).layernorm, {input(xv), input(wv), input_or(bv, at(wv)), at(o)},
           ln_args{xv.rs, wv.rs, bv.rs, o.rs, u32(xv.cols), bv.b ? 1u : 0u, eps}, xv.rows, norm_threads(xv.cols));
  });
}

larch_status rope_multi(larch_stream* vs, const larch_tensor* qk, const larch_tensor* pos_i32,
                         const larch_rope_multi_params* p) {
  return run("rope_multi", [&] {
    const auto checked = cpu::check_rope_multi(sizer(vs), qk, pos_i32, p);
    const view q = checked.qk, pos = checked.pos, ff = checked.freq_factors;
    const larch_rope_params* r = &p->rope;
    const int64_t nd = r->n_dims, pairs = p->mode == LARCH_ROPE_VISION ? nd : nd / 2;
    writable(q);
    // rope's rope_yarn constants, as the CPU reference's yarn computes them.
    const auto corr_dim = [&](float n_rot) {
      return float(nd) * logf(float(r->n_ctx_orig) / (n_rot * 2 * float(M_PI))) / (2 * logf(r->freq_base));
    };
    const float lo = std::max(0.0f, floorf(corr_dim(r->beta_fast)));
    const float hi = std::min(float(nd - 1), ceilf(corr_dim(r->beta_slow)));
    float mscale = r->attn_factor;
    if (r->ext_factor != 0.0f) mscale *= 1.0f + 0.1f * logf(1.0f / r->freq_scale);
    const rope_multi_args a{q.rs, pos.rs, ff.rs, u32(q.shape[2]), u32(pairs), p->mode, r->freq_factors ? 1u : 0u,
                            {p->sections[0], p->sections[1], p->sections[2], p->sections[3]},
                            powf(r->freq_base, -2.0f / float(nd)), r->freq_scale, r->ext_factor, mscale, lo, hi};
    each(static_cast<mtl_stream*>(vs), ks(vs).rope_multi, {at(q), input(pos), input_or(ff, at(pos))}, a, pairs,
         q.shape[1], q.rows);
  });
}

larch_status gelu(larch_stream* vs, const larch_tensor* x, const larch_tensor* out) {
  return run("gelu", [&] {
    const auto checked = cpu::check_gelu(sizer(vs), x, out);
    const view xv = checked.a, o = checked.out;
    writable(o);
    each(static_cast<mtl_stream*>(vs), ks(vs).gelu, {input(xv), at(o)}, ew_args{xv.rs, 0, o.rs, 0}, xv.cols, xv.rows);
  });
}

larch_status attn_full(larch_stream* vs, const larch_tensor* q, const larch_tensor* k, const larch_tensor* v,
                        uint32_t seg_len, float scale, const larch_tensor* out) {
  return run("attn_full", [&] {
    const auto checked = cpu::check_attn_full(sizer(vs), q, k, v, seg_len, out);
    const view qv = checked.q, kv = checked.k, vv = checked.v, o = checked.out;
    writable(o);
    const int64_t heads = qv.shape[1], hd = qv.shape[2];
    if (hd != 64 && hd != 72) throw unsupported("attn_full: head_dim other than 64 or 72");
    if (qv.rows > INT32_MAX || heads > 65535) throw unsupported("attn_full: 2^31 rows or 65536 heads or more");
    for (const view* t : {&qv, &kv, &vv, &o})
      if (t->rs % 2) throw unsupported("attn_full: a row stride that is not whole halves");
    const kernels& kk = ks(vs);
    const bool nax = hd == 64 && kk.attn_full_nax;
    const int bq = nax ? 64 : 32, bk = 32, n = int(seg_len);
    // MLX's batch is our segment, its sequence a segment's rows; strides in halves (B, H, L; D = 1).
    const auto strides = [&](const view& t, int64_t (&st)[3]) {
      st[0] = int64_t(seg_len) * int64_t(t.rs / 2), st[1] = hd, st[2] = int64_t(t.rs / 2);
    };
    mlx::steel::AttnParams p{};
    p.B = int(qv.rows / seg_len), p.H = int(heads), p.D = int(hd), p.qL = n, p.kL = n, p.gqa_factor = 1;
    p.scale = scale, p.NQ = (n + bq - 1) / bq, p.NK = (n + bk - 1) / bk, p.NQ_aligned = n / bq, p.NK_aligned = n / bk;
    p.qL_rem = n - p.NQ_aligned * bq, p.kL_rem = n - p.NK_aligned * bk, p.qL_off = 0;
    strides(qv, p.Q_strides), strides(kv, p.K_strides), strides(vv, p.V_strides), strides(o, p.O_strides);
    encode(static_cast<mtl_stream*>(vs), nax ? kk.attn_full_nax : kk.attn_full_sg[hd == 72],
           {input(qv), input(kv), input(vv), at(o)}, p, MTLSizeMake(NSUInteger(p.NQ), NSUInteger(p.H), NSUInteger(p.B)),
           MTLSizeMake(32 * ATTN_FULL_SG, 1, 1), false);
  });
}

// mmvq's dispatch on checked views.
void mmvq_rows(larch_stream* vs, const view& w, const view& x, const view& b, const view& o) {
  const mm_args a{x.rs, w.rs, o.rs, b.rs, u32(x.rows), u32(w.rows), u32(w.cols), b.b ? 1u : 0u,
                  o.dtype == LARCH_F32 ? 1u : 0u, 1.0f, 0.0f, 0};
  auto* stream = static_cast<mtl_stream*>(vs);
  if (x.rows == 1 && ks(vs).mmvq1[w.dtype]) {  // m1 decode 24.6 to 26.0 tok/s (M5, 10 Oct)
    const int64_t cols = MMVQ1_SG * MMVQ1_NR;
    groups(stream, ks(vs).mmvq1[w.dtype], {input(w), input(x), input_or(b, at(w)), at(o)}, a, (w.rows + cols - 1) / cols,
           32 * MMVQ1_SG);
    return;
  }
  // The K-quants from 3 rows (Q3_K from 2): the few-row MMA kernel beat the others at every m1 shape and row count
  // (M5, 10 Oct: 8 rows of 14336 x 4096 Q4_K 473 to 318 us, llama.cpp 319). At 2 rows the shared-x kernel won
  // for Q4_K, Q5_K and Q6_K (Q5_K 12288 x 2048: 140 vs 168 us), but on Q3_K it took 361 us to the MMA kernel's 138
  // (M5, 27 Nov; 8 rows 467 to 139).
  if (w.dtype == LARCH_Q3_K && x.rows >= 2 && x.rows <= 8 && stream->dev->fed.mmvq_mmc && ks(vs).mmvq_mmc8_q3_k) {
    const int64_t cols = 8 * MMVQ_MMA_NT;
    groups(stream, ks(vs).mmvq_mmc8_q3_k, {input(w), input(x), input_or(b, at(w)), at(o)}, a, (w.rows + cols - 1) / cols,
           32 * MMVQ_MMA_SG);
    return;
  }
  if (x.rows >= (w.dtype == LARCH_Q3_K ? 2 : 3) && ks(vs).mmvq_mma[x.rows > 8][w.dtype]) {
    const int64_t cols = 8 * MMVQ_MMA_NT;
    groups(stream, ks(vs).mmvq_mma[x.rows > 8][w.dtype], {input(w), input(x), input_or(b, at(w)), at(o)}, a,
           (w.rows + cols - 1) / cols, 32 * MMVQ_MMA_SG);
    return;
  }
  // The other formats, and 2 rows. Measured on an idle M5, 9 Oct, n 256-128256, rows 1-16. The shared-x
  // kernel (16 columns per threadgroup) beats the others at 2-4 rows, and at 5-7 rows where w is too narrow for
  // the tile kernel, once it has 4 threadgroups per core (n 256 lost, 512 tied, 1024 won). The tile kernel (32
  // columns per threadgroup) wins from 5 rows with 4 threadgroups per core, from 8 rows with 2. Otherwise the
  // one-column kernel.
  // A device whose tile pipeline cannot run 128 threads keeps the others (device_open).
  const id<MTLComputePipelineState> mm = ks(vs).mmvq_mm[x.rows > 8][w.dtype], one = ks(vs).mmvq[w.dtype],
                                    shared = ks(vs).mmvq_ext[x.rows > 4][w.dtype];
  const int64_t cores = core_count(stream, one), tiles = (w.rows + 31) / 32;
  const int64_t shared_groups = (w.rows + 4 * (32 / MMVQ_EXT_NX) - 1) / (4 * (32 / MMVQ_EXT_NX));
  const bool tiled = mm && x.rows >= 5 && tiles >= cores * (x.rows >= 8 ? 2 : 4);
  const bool shared_x = !tiled && x.rows >= 2 && x.rows <= 8 && shared_groups >= cores * 4;
  groups(stream, tiled ? mm : shared_x ? shared : one, {input(w), input(x), input_or(b, at(w)), at(o)}, a,
         tiled ? tiles : shared_x ? shared_groups : x.rows ? (w.rows + 3) / 4 : 0, 128);  // 4 simdgroups each
}

larch_status mmvq(larch_stream* vs, const larch_tensor* w_quant, const larch_tensor* x_f16,
                   const larch_tensor* bias_or_null, const larch_tensor* out) {
  return run("mmvq", [&] {
    const auto checked = cpu::check_mmvq(sizer(vs), w_quant, x_f16, bias_or_null, out);
    writable(checked.c);
    static_cast<mtl_stream*>(vs)->check = false;
    mmvq_rows(vs, checked.b_t, checked.a, checked.bias, checked.c);
  });
}

// larch.h's verify GEMV on the staged tile: one kernel at every row count, its K split from the shape and the core
// count alone, so a row's bits never depend on the rows beside it. A drafter's one-row decode runs it too (the engine's
// check steps), which keeps guessed output exact within that mode. A format or shape the tile does not take (Q4_0,
// Q5_0, k not a multiple of ST_UNIT, a Q4_K or Q5_K w or any x off 16-byte alignment) runs one-row mmvq row by row,
// which is row-independent too. UNSUPPORTED without the tile (no Apple GPU family 10), so the engine drops drafters.
larch_status mmvq_verify(larch_stream* vs, const larch_tensor* w_quant, const larch_tensor* x_f16,
                          const larch_tensor* bias_or_null, const larch_tensor* out) {
  return run("mmvq_verify", [&] {
    const auto checked = cpu::check_mmvq_verify(sizer(vs), w_quant, x_f16, bias_or_null, out);
    const view w = checked.b_t, x = checked.a, o = checked.c, b = checked.bias;
    writable(o);
    // Every family-10 device builds every format's tile; Q4_K stands for them.
    if (!ks(vs).mmvq_st[0][LARCH_Q4_K]) throw unsupported("no staged tile (Apple GPU family 10 and the tensor API)");
    auto* s = static_cast<mtl_stream*>(vs);
    s->check = true;
    if (!x.rows) return;
    const id<MTLComputePipelineState> tile = ks(vs).mmvq_st[x.rows > 16 ? 2 : x.rows > 8][w.dtype];
    const bool wide_loads = w.dtype == LARCH_Q4_K || w.dtype == LARCH_Q5_K;  // uint4 loads of each block
    if (!tile || w.cols % ST_UNIT || (wide_loads && w.off % 16) || x.off % 16) {
      for (int64_t r = 0; r < x.rows; ++r) {
        view xr = x, orow = o;
        xr.rows = orow.rows = 1;
        xr.off += uint64_t(r) * x.rs;
        orow.off += uint64_t(r) * o.rs;
        mmvq_rows(vs, w, xr, b, orow);
      }
      return;
    }
    const uint32_t n = u32(w.rows), k = u32(w.cols), splits = st_splits(n, k, core_count(s, tile));
    const uint64_t tiles = (n + ST_N - 1) / ST_N;
    const st_args a{{x.rs, w.rs, o.rs, b.rs, u32(x.rows), n, k, b.b ? 1u : 0u, o.dtype == LARCH_F32 ? 1u : 0u, 1.0f,
                     0.0f, 0},
                    splits,
                    1};
    // Unused bindings (no bias; parts and counters at one part): the weight buffer from offset 0, read-only so it
    // costs no barrier. Offset 0, because validation wants typed buffer offsets aligned and w's may be 2-aligned.
    const arg unused = input(arg{w.b->buf, 0});
    arg parts = unused, counters = unused;
    if (splits > 1) {  // work queued with an old buffer keeps it alive
      const uint64_t part_bytes = uint64_t(splits) * uint64_t(x.rows) * n * 4, counter_bytes = tiles * 4;
      if (part_bytes > s->dev->dev.maxBufferLength) throw std::bad_alloc();
      if (s->st_parts.length < part_bytes) {
        s->st_parts = [s->dev->dev newBufferWithLength:part_bytes options:MTLResourceStorageModePrivate];
        if (!s->st_parts) throw std::bad_alloc();
      }
      if (s->st_counters.length < counter_bytes) {  // the tile leaves them zero; a new buffer starts so
        s->st_counters = [s->dev->dev newBufferWithLength:counter_bytes options:MTLResourceStorageModeShared];
        if (!s->st_counters) throw std::bad_alloc();
        std::memset(s->st_counters.contents, 0, counter_bytes);
      }
      // Written: two split calls never share the counters without a barrier.
      // ponytail: one counter set a stream, so split calls run one after another; a ring of sets if a check
      // step's profile shows the barriers.
      parts = {s->st_parts, 0, part_bytes};
      counters = {s->st_counters, 0, counter_bytes};
    }
    groups(s, tile, {input(w), input(x), input_or(b, unused), at(o), parts, counters}, a, int64_t(tiles), 32, splits);
  });
}

// LARCH_METAL_PROFILE's pairs an expert: copies vv_moe_group's offsets (experts + 1 at lists' start) to the host
// and adds them to moe_hist_rows()[name] when the copy completes. Profiling runs only: it commits a buffer.
static void moe_hist(mtl_stream* s, id<MTLBuffer> lists, uint32_t experts, std::string name) {
  const uint64_t bytes = uint64_t(experts + 1) * 4;
  id<MTLBuffer> h = [s->dev->dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!h) return;
  blit(s, lists, 0, h, 0, bytes);
  [current(s) addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    if (cb.status != MTLCommandBufferStatusCompleted) return;
    const uint32_t* g = static_cast<const uint32_t*>(h.contents);
    std::lock_guard<std::mutex> l(profile_lock);
    moe_hist_row& row = moe_hist_rows()[name];
    row.calls += 1;
    for (uint32_t e = 0; e < experts; ++e) {
      const uint64_t c = g[e + 1] - g[e];
      row.pairs += c;
      row.tile_rows += (c + 63) / 64 * 64;
      row.bins[c == 0 ? 0 : c < 32 ? 1 : c < 64 ? 2 : c < 128 ? 3 : c < 256 ? 4 : 5] += 1;
    }
  }];
  commit(s);
}

// 4 simdgroups per threadgroup over w's columns, by (row, slot) pairs, or by rows when weighted.
larch_status mmvq_id(larch_stream* vs, const larch_tensor* w_quant, const larch_tensor* x_f16,
                      const larch_tensor* ids_i32, const larch_tensor* bias_or_null,
                      const larch_tensor* weights_or_null, const larch_tensor* out) {
  return run("mmvq_id", [&] {
    const auto c = cpu::check_mmvq_id(sizer(vs), w_quant, x_f16, ids_i32, bias_or_null, weights_or_null, out);
    const view w = c.w, x = c.x, ids = c.ids, b = c.bias, wt = c.weights, o = c.out;
    writable(o);
    const int64_t slots = ids.cols, n = o.shape[o.ndim - 1];
    // The shader indexes rows (expert * n + column), k, slots and (row, slot) pairs in 32 bits.
    if (w.rows > UINT32_MAX || w.cols > UINT32_MAX || slots > UINT32_MAX || x.rows * slots > int64_t(UINT32_MAX))
      throw unsupported("mmvq_id: a dimension or rows x slots of 2^32 or more");
    const bool weighted = wt.b != nullptr;
    const uint64_t elem = o.dtype == LARCH_F32 ? 4 : 2;
    const int64_t experts = w.rows / n, pairs = x.rows * slots;
    const id_args a{x.rs, x.ndim == 3 ? uint64_t(w.cols) * 2 : 0, w.rs, ids.rs, weighted ? wt.rs : 0, o.rs,
                    weighted ? 0 : uint64_t(n) * elem, b.rs, u32(n), u32(w.cols), u32(slots), b.b ? 1u : 0u,
                    o.dtype == LARCH_F32 ? 1u : 0u, weighted ? 1u : 0u, u32(experts), u32(x.rows)};
    mtl_stream* s = static_cast<mtl_stream*>(vs);
    // ponytail: grouped only up to k = 1024 MOE_GROUP_T (qwen3moe's 2048); raise MOE_GROUP_T when a model's k does.
    // Drafter mode runs per pair at any row count, 1-row decode included, so a row's bits never depend on the rows
    // beside it (the engine's check steps send up to 32 rows a call).
    if (s->check || pairs < MOE_GROUP_PAIRS * experts || experts > MOE_ROUTE_MAX || w.cols > 1024 * MOE_GROUP_T) {
      id<MTLComputePipelineState> narrow = w.cols < 1024 ? ks(vs).mmvq_id1n[w.dtype] : nil;
      const int64_t cols = MMVQ1_SG * (narrow ? MMVQ1N_C : MMVQ1_NR);
      if (id<MTLComputePipelineState> one = narrow ? narrow : ks(vs).mmvq_id1[w.dtype])
        groups(s, one, {input(w), input(x), input(ids), input_or(b, at(w)), input_or(wt, at(w)), at(o)}, a, (n + cols - 1) / cols,
               32 * MMVQ1_SG, weighted ? x.rows : pairs);
      else
        groups(s, ks(vs).mmvq_id[w.dtype], {input(w), input(x), input(ids), input_or(b, at(w)), input_or(wt, at(w)), at(o)}, a,
               (n + 3) / 4, 128, weighted ? x.rows : pairs);
      return;
    }
    // Weighted, the f32 partials of up to MOE_PARTIAL_BYTES at a time: blocks of rows, each grouped on its own,
    // so the scratch the engine's fit charges (larch-engine fit.rs) bounds every shape. The tensor path adds
    // the block's x rows in expert order, which its 32 x 64 tile reads.
    const int64_t block = weighted ? std::max<int64_t>(1, MOE_PARTIAL_BYTES / (slots * n * 4)) : x.rows;
    const int64_t kp = (w.cols + MOE_MM_K - 1) / MOE_MM_K * MOE_MM_K, pairs_max = std::min(x.rows, block) * slots;
    // ponytail: gate and up's gathered x is not blocked: 256 MiB at the 8,192-token chunk maximum for k 2048. Block
    // it under MOE_PARTIAL_BYTES too if a 16 GB Mac's fit shows it.
    const bool gathered = ks(vs).mmvq_id_mm[0][w.dtype];  // only the 32 x 64 tile reads it
    const uint64_t lists = (uint64_t(experts + 1 + pairs_max) * 4 + 15) / 16 * 16;
    const uint64_t parts = weighted ? (uint64_t(pairs_max * n) * 4 + 15) / 16 * 16 : 0;
    const uint64_t bytes = lists + parts + (gathered ? uint64_t(pairs_max * kp) * 2 : 0);
    if (s->moe_scratch.length < bytes) {  // work queued with the old one keeps it alive
      if (bytes > s->dev->dev.maxBufferLength) throw std::bad_alloc();  // the validation layer aborts past it
      s->moe_scratch = [s->dev->dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
      if (!s->moe_scratch) throw std::bad_alloc();
    }
    const arg lst{s->moe_scratch, 0, lists}, part{s->moe_scratch, lists, parts}, xsort{s->moe_scratch, lists + parts};
    for (int64_t r0 = 0; r0 < x.rows; r0 += block) {
      const int64_t r = std::min(block, x.rows - r0);
      id_args ab = a;
      ab.rows = u32(r);
      const auto from = [&](const view& v, uint64_t rs) { return arg{at(v).buf, at(v).off + uint64_t(r0) * rs}; };
      const arg xa = input(from(x, x.rs)), ia = input(from(ids, ids.rs)), oa = from(o, o.rs);
      groups(s, ks(vs).moe_group, {ia, lst}, ab, 1, 256);
      if (s->dev->fed.profile && s->profiled)
        moe_hist(s, s->moe_scratch, ab.experts,
                 std::string(dtype_name[w.dtype]) + " " + std::to_string(n) + "x" + std::to_string(w.cols) + ", " +
                     std::to_string(r) + " rows");
      // 64 pairs x 32 columns where the decode outweighs the half-empty tiles. M5, 128 experts, 8 slots, 2048 rows:
      // Q3_K 768 x 2048 17.18 -> 14.71 ms, Q6_K 2048 x 768 21.34 -> 17.48 ms; Q4_K and Q5_K stay 32 x 64 (8.34 vs
      // 9.31, 12.21 vs 14.03 ms), as does Q3_K at 512 rows (6.46 vs 9.10 ms).
      const bool tall = (w.dtype == LARCH_Q3_K || w.dtype == LARCH_Q6_K) && r * slots >= 64 * experts;
      if (id<MTLComputePipelineState> mm = ks(vs).mmvq_id_mm[tall][w.dtype]) {
        if (!tall) each(s, ks(vs).moe_gather, {xa, input(lst), xsort}, ab, kp / 4, r * slots);  // tall stages x
        groups(s, mm, {input(w), tall ? xa : input(xsort), input(lst), input_or(b, at(w)), weighted ? part : oa}, ab,
               (n + (tall ? 31 : 63)) / (tall ? 32 : 64), MOE_MM_SG * 32, experts);
      } else if (id<MTLComputePipelineState> sg = ks(vs).mmvq_id_mm_sg[w.dtype])  // no tensor API
        groups(s, sg, {input(w), xa, lst, input_or(b, at(w)), weighted ? part : oa}, ab,
               (n + GEMM_QS_N - 1) / GEMM_QS_N, GEMM_QS_SG * 32, experts);
      else
        groups(s, ks(vs).mmvq_id_grouped[w.dtype], {input(w), xa, lst, input_or(b, at(w)), weighted ? part : oa}, ab,
               (n + 3) / 4, 128, experts);
      if (weighted) each(s, ks(vs).moe_reduce, {part, input(from(wt, wt.rs)), oa}, ab, n, r);
    }
  });
}

larch_status moe_route(larch_stream* vs, const larch_tensor* logits_f32, float scale,
                        const larch_tensor* out_ids_i32, const larch_tensor* out_weights_f32) {
  return run("moe_route", [&] {
    const auto c = cpu::check_moe_route(sizer(vs), logits_f32, scale, out_ids_i32, out_weights_f32);
    const view lg = c.logits, ids = c.ids, w = c.weights;
    writable(ids);
    writable(w);
    if (lg.cols > MOE_ROUTE_MAX) throw unsupported("moe_route: more than 512 experts");
    groups(static_cast<mtl_stream*>(vs), ks(vs).moe_route, {input(lg), at(ids), at(w)},
           route_args{lg.rs, ids.rs, w.rs, u32(lg.cols), u32(ids.cols), scale}, lg.rows, 32);
  });
}

larch_status dequant(larch_stream* vs, const larch_tensor* w_quant, const larch_tensor* out_f16) {
  return run("dequant", [&] {
    const auto checked = cpu::check_dequant(sizer(vs), w_quant, out_f16);
    const view w = checked.w, o = checked.out;
    writable(o);
    const arg wa = at(w), oa = at(o);
    auto* s = static_cast<mtl_stream*>(vs);
    s->check = false;
    if (!w.rows) return;
    // Deferred: the gemm that reads out next can read w instead (gemm_q), never writing the f16 copy. Anything
    // else that binds out's buffer, or w's when writable, runs it first (encode, blit), and so does every flush
    // (sync, download, event_record); an early commit leaves it deferred. A later dequant replaces the rows of it
    // that its gapless output covers: the model's chain of dequants into one scratch (q k v o gate up down, then
    // q again) runs none but a row across an end. m1 TTFT (M5, 10 Oct): 52 tokens 316 to 115 ms, 1594 tokens 3.1
    // to 2.1 s (164 ms and 2.3 s with one pending).
    const mtl_stream::lazy_dequant n{wa.buf, oa.buf, w.b->readonly, wa.off, w.rs, oa.off, o.rs, w.rows, w.cols, w.dtype};
    const uint64_t start = n.out_off, end = n.out_end();
    bool all = s->lazy.size() >= 8;  // ponytail: 8 pending at most; the model keeps 3
    for (const auto& l : s->lazy) all |= l.out == wa.buf || (l.w == oa.buf && !l.w_readonly);
    if (all) settle(s);
    const bool dense = o.rs == uint64_t(o.cols) * sizeof(uint16_t);
    std::vector<mtl_stream::lazy_dequant> keep;
    for (auto l : std::exchange(s->lazy, {})) {
      const uint64_t from = l.out_off, to = l.out_end();
      if (l.out != oa.buf || to <= start || from >= end) {  // disjoint
        keep.push_back(l);
      } else if (dense && start <= from && to > end) {  // rows r0 on lie past end and stay
        const int64_t r0 = int64_t((end - from + l.out_rs - 1) / l.out_rs);
        if (from + uint64_t(r0 - 1) * l.out_rs + uint64_t(l.cols) * sizeof(uint16_t) > end) {  // row r0 - 1 crosses it
          auto row = l;
          row.w_off += uint64_t(r0 - 1) * l.w_rs, row.out_off += uint64_t(r0 - 1) * l.out_rs, row.rows = 1;
          run_lazy(s, row);
        }
        l.w_off += uint64_t(r0) * l.w_rs, l.out_off += uint64_t(r0) * l.out_rs, l.rows -= r0;
        if (l.rows > 0) keep.push_back(l);
      } else if (!dense || start > from) {  // overlapped some other way: it runs now
        run_lazy(s, l);
      }  // else covered: replaced unrun
    }
    keep.push_back(n);
    s->lazy = std::move(keep);
  });
}

larch_status gemm(larch_stream* vs, const larch_tensor* a_f16, const larch_tensor* b_f16_t,
                   const larch_tensor* bias_or_null, const larch_tensor* c, float alpha, float beta) {
  return run("gemm", [&] {
    const auto checked = cpu::check_gemm(sizer(vs), a_f16, b_f16_t, bias_or_null, c);
    const view a = checked.a, bt = checked.b_t, cv = checked.c, b = checked.bias;
    writable(cv);
    // Threadgroups run x first. With x on c's rows, every row tile of one b_t (weight) tile runs together
    // and shares it from cache; with x on c's columns, a's row tile is shared instead. The deeper k's tile
    // is the bigger, so x goes to rows when k <= n (M5, 9 Oct, m 512: 14336 x 4096 5.0 -> 11.4 TFLOP/s and
    // 4096 x 4096 5.8 -> 11.3 with rows; 4096 x 14336 3.4 with rows, 3.9 with columns).
    const uint32_t m_fast = a.cols <= bt.rows ? 1u : 0u;
    const mm_args p{a.rs, bt.rs, cv.rs, b.rs, u32(a.rows), u32(bt.rows), u32(a.cols), b.b ? 1u : 0u,
                    cv.dtype == LARCH_F32 ? 1u : 0u, alpha, beta, m_fast};
    auto* stream = static_cast<mtl_stream*>(vs);
    const arg aa = input(a), bb = input(bt), bias = input_or(b, aa), cc = at(cv);
    const bool tensor = tensor_gemm_eligible(stream, a, bt);
    // b_t is exactly a deferred dequant's output, and nothing else here touches one: read w in place, where the
    // dequant would write its f16 copy for the GEMM to read back. The tensor path takes gemm_q when k is whole
    // GEMM_Q_K tiles; a device without tensor GEMM takes vv_gemm_qs past GEMV_M rows.
    const auto& lazy = stream->lazy;
    const auto l = std::find_if(lazy.begin(), lazy.end(), [&](const auto& e) {
      return e.out == bb.buf && e.out_off == bb.off && e.out_rs == bt.rs && e.rows == bt.rows && e.cols == bt.cols;
    });
    const bool alone = l != lazy.end() && !touches_lazy(stream, aa.buf) && !touches_lazy(stream, bias.buf) &&
                       !touches_lazy(stream, cc.buf);
    const bool q = alone && tensor && l->cols % GEMM_Q_K == 0;
    const bool qs = alone && !stream->dev->tensor_gemm && a.rows > GEMV_M;
    const id<MTLComputePipelineState> fused = q    ? ks(vs).gemm_q[a.rows > GEMM_Q_M_SMALL][m_fast][l->dtype]
                                              : qs ? ks(vs).gemm_qs[l->dtype]
                                                   : nil;
    if (fused) {
      // gemm_q, k <= n: half-width tiles, twice as many threadgroups (M5: 4096 x 4096 at 512 rows 1450 to 1330
      // us); k > n keeps the full width, which reads a half as often (4096 x 14336: 4930 against 7800 us).
      const int64_t tm = !q ? GEMM_QS_M : a.rows > GEMM_Q_M_SMALL ? GEMM_Q_M : GEMM_Q_M_SMALL;
      const int64_t tn = !q ? GEMM_QS_N : m_fast ? GEMM_Q_N / 2 : GEMM_Q_N;
      const int64_t rows = (a.rows + tm - 1) / tm, cols = (bt.rows + tn - 1) / tn;
      mm_args pw = p;
      pw.b_rs = l->w_rs;
      const arg wq = input(arg{l->w, l->w_off});
      auto keep = std::exchange(stream->lazy, {});  // the kernel reads w, as the dequant will: it stays deferred
      groups(stream, fused, {aa, wq, bias, cc}, pw, m_fast ? rows : cols, (q ? GEMM_Q_SG : GEMM_QS_SG) * 32,
             m_fast ? cols : rows);
      stream->lazy = std::move(keep);
      return;
    }
    if (a.rows > 0 && a.rows <= GEMV_M) {  // 0 rows keeps the empty-grid gemm below: no dispatch
      groups(stream, ks(vs).gemv, {aa, bb, bias, cc}, p, bt.rows, 32 * GEMV_SG);
      return;
    }
    // An f16 weight (no deferred dequant: a projector's): the tensor tiles run x on rows whatever k, and the
    // deep tile only under 1024 rows (M5, 11 Nov, metal_perf: 3680 x 4096 -> 1024 6.4 -> 14.7 TFLOP/s, 6400
    // rows 6.0 -> 14.5). A dequantized text weight keeps the old order: on it this cost serve ttft p99 18-21 %.
    // ponytail: f16 weights only; retry for dequantized ones with a serve A/B that isolates the ttft tail.
    const bool f16_weight = l == lazy.end();
    const bool deep = tensor && !m_fast && a.rows >= GEMM_TENSOR_DEEP_M && (!f16_weight || a.rows < 1024) &&
                      ks(vs).gemm_tensor_deep;
    mm_args pt = p;
    pt.m_fast = tensor && f16_weight ? 1u : m_fast;
    const int64_t tm = deep ? GEMM_TENSOR_DEEP_M : tensor ? GEMM_TENSOR_M : 32, tn = tensor ? GEMM_TENSOR_N : 32;
    const int64_t rows = (a.rows + tm - 1) / tm, cols = (bt.rows + tn - 1) / tn;
    const id<MTLComputePipelineState> pipeline = deep ? ks(vs).gemm_tensor_deep : tensor ? ks(vs).gemm_tensor : ks(vs).gemm;
    groups(stream, pipeline, {aa, bb, bias, cc}, pt, pt.m_fast ? rows : cols,
           tensor ? pipeline.threadExecutionWidth * (deep ? GEMM_TENSOR_DEEP_SG : GEMM_TENSOR_SG) : 128,
           pt.m_fast ? cols : rows);
  });
}

larch_status silu_mul(larch_stream* vs, const larch_tensor* gate, const larch_tensor* up,
                       const larch_tensor* out) {
  return run("silu_mul", [&] {
    const auto checked = cpu::check_silu_mul(sizer(vs), gate, up, out);
    const view g = checked.a, u = checked.b, o = checked.out;
    writable(o);
    each(static_cast<mtl_stream*>(vs), ks(vs).silu_mul, {input(g), input(u), at(o)}, ew_args{g.rs, u.rs, o.rs, 0}, g.cols, g.rows);
  });
}

larch_status add(larch_stream* vs, const larch_tensor* a_f32, const larch_tensor* b_f16,
                  const larch_tensor* out) {
  return run("add", [&] {
    const auto checked = cpu::check_add(sizer(vs), a_f32, b_f16, out);
    const view a = checked.a, b = checked.b, o = checked.out;
    writable(o);
    each(static_cast<mtl_stream*>(vs), ks(vs).add, {input(a), input(b), at(o)}, ew_args{a.rs, b.rs, o.rs, 0}, a.cols, a.rows);
  });
}

larch_status sigmoid_mul(larch_stream* vs, const larch_tensor* x, const larch_tensor* gate,
                          const larch_tensor* out) {
  return run("sigmoid_mul", [&] {
    const auto checked = cpu::check_sigmoid_mul(sizer(vs), x, gate, out);
    const view xv = checked.a, g = checked.b, o = checked.out;
    writable(o);
    each(static_cast<mtl_stream*>(vs), ks(vs).sigmoid_mul, {input(xv), input(g), at(o)},
         ew_args{xv.rs, g.rs, o.rs, u32(g.cols)}, xv.cols, xv.rows);
  });
}

larch_status draft_conv(larch_stream* vs, const larch_tensor* x, const larch_tensor* dyn, const larch_tensor* base,
                         uint32_t stage, const larch_tensor* out) {
  return run("draft_conv", [&] {
    const auto checked = cpu::check_draft_conv(sizer(vs), x, dyn, base, stage, out);
    const view xv = checked.x, dv = checked.dyn, bv = checked.base, o = checked.out;
    writable(o);
    const u32 groups = u32(dv.shape[2]);
    each(static_cast<mtl_stream*>(vs), ks(vs).draft_conv, {input(xv), input(dv), input(bv), at(o)},
         draft_conv_args{xv.rs, dv.rs, bv.rs, o.rs, u32(xv.cols), u32(xv.cols) / groups, groups, 2 * stage,
                         o.dtype == LARCH_F32 ? 1u : 0u},
         xv.cols, xv.rows);
  });
}

// Gated DeltaNet's arguments from the shared checks: the state view is the layer's [n_slots, per slot].
gdn_args gdn_arguments(const cpu::gdn_checked& c, const larch_gdn_state* state, float eps) {
  const view in = c.in, out = c.out, w = c.w, al = c.alpha, be = c.beta, dt = c.dt_bias, av = c.a, st = c.state,
             slots = c.slots, lens = c.lens, snaps = c.snaps;
  return {in.rs, out.rs, w.rs, al.rs, be.rs, dt.rs, av.rs, slots.rs, lens.rs, snaps.rs, st.rs, u32(slots.rows),
          u32(in.cols), state->kernel, state->k_heads, state->v_heads, state->head_dim, snaps.b ? 1u : 0u, eps,
          1.0f / sqrtf(float(state->head_dim))};
}

larch_status gdn_conv(larch_stream* vs, const larch_tensor* x, const larch_tensor* w, const larch_gdn_state* state,
                       const larch_tensor* slots_i32, const larch_tensor* q_lens_i32,
                       const larch_tensor* snap_slots_or_null, const larch_tensor* out) {
  return run("gdn_conv", [&] {
    const cpu::gdn_checked c =
        cpu::check_gdn_conv(sizer(vs), x, w, state, slots_i32, q_lens_i32, snap_slots_or_null, out);
    const view xv = c.in, wv = c.w, o = c.out, st = c.state, slots = c.slots, lens = c.lens, snaps = c.snaps;
    writable(o);
    writable(st);
    if (state->kernel > GDN_MAX_TAPS)
      throw unsupported("gdn_conv: a conv kernel wider than " + std::to_string(GDN_MAX_TAPS));
    each(static_cast<mtl_stream*>(vs), ks(vs).gdn_conv,
         {input(xv), input(wv), at(st), input(slots), input(lens), input_or(snaps, at(lens)), at(o)},
         gdn_arguments(c, state, 0.0f), xv.cols, slots.rows);
  });
}

larch_status gdn(larch_stream* vs, const larch_tensor* qkv, const larch_tensor* alpha, const larch_tensor* beta,
                  const larch_tensor* dt_bias, const larch_tensor* a_log, float eps, const larch_gdn_state* state,
                  const larch_tensor* slots_i32, const larch_tensor* q_lens_i32,
                  const larch_tensor* snap_slots_or_null, const larch_tensor* out) {
  return run("gdn", [&] {
    const cpu::gdn_checked c = cpu::check_gdn(sizer(vs), qkv, alpha, beta, dt_bias, a_log, eps, state, slots_i32,
                                              q_lens_i32, snap_slots_or_null, out);
    const view in = c.in, al = c.alpha, be = c.beta, dt = c.dt_bias, av = c.a, o = c.out, st = c.state,
               slots = c.slots, lens = c.lens, snaps = c.snaps;
    writable(o);
    writable(st);
    if (state->head_dim % 32 || state->head_dim > GDN_MAX_HD)
      throw unsupported("gdn: head_dim not a multiple of 32 up to 256");
    mtl_stream* s = static_cast<mtl_stream*>(vs);
    // ponytail: by the call's average rows a sequence, so a prefill batched with many decodes walks rows one
    // simdgroup a row; split the call by sequence length when mixed batches show gdn in a profile.
    const bool few = in.rows <= GDN_WIDE_ROWS * slots.rows, wide = few && s->dev->fed.gdn_wide;
    id<MTLComputePipelineState> chunk =
        few || (state->head_dim != 64 && state->head_dim != 128) ? nil : ks(vs).gdn_chunk[state->head_dim / 128];
    const uint32_t rows_per_group = (chunk ? 8 : wide ? 1 : GDN_ROWS) * GDN_SG;
    encode(s, chunk ? chunk : wide ? ks(vs).gdn1 : ks(vs).gdn,
           {input(in), input(al), input(be), input(dt), input(av), at(st), input(slots), input(lens),
            input_or(snaps, at(lens)), at(o)},
           gdn_arguments(c, state, eps),
           MTLSizeMake((state->head_dim + rows_per_group - 1) / rows_per_group, state->v_heads, NSUInteger(slots.rows)),
           MTLSizeMake(32, GDN_SG, 1), false);
  });
}

larch_status kv_write(larch_stream* vs, const larch_tensor* k, const larch_tensor* v,
                       const larch_tensor* slots_i32, const larch_kv_pool* pool) {
  return run("kv_write", [&] {
    const auto checked = cpu::check_kv_write(sizer(vs), k, v, slots_i32, pool);
    const view kv = checked.k, vv = checked.v, slots = checked.slots, pk = checked.pool_k, pv = checked.pool_v;
    need(!pk.b->readonly && !pv.b->readonly, "the pool is a read-only wrapped buffer");
    each(static_cast<mtl_stream*>(vs), ks(vs).kv_write,
         {input(kv), input(vv), input(slots), at(pk), at(pv)}, ew_args{kv.rs, vv.rs, slots.rs, u32(kv.cols)},
         kv.cols, kv.rows);
  });
}

// The gated split, both norms, rope and the KV write in one vv_attn_in, which takes each step's arithmetic, so its
// bytes are the sequence's. Head dims past ATTN_IN_HD answer UNSUPPORTED.
larch_status fused_attn_in(larch_stream* vs, const larch_tensor* q_in, const larch_tensor* k, const larch_tensor* v,
                            const larch_tensor* q_norm, const larch_tensor* k_norm, float eps,
                            const larch_tensor* pos, const larch_rope_params* p, const larch_tensor* slots,
                            const larch_kv_pool* pool, const larch_tensor* q, const larch_tensor* gate) {
  return run("fused_attn_in", [&] {
    const auto c = cpu::check_fused_attn_in(sizer(vs), q_in, k, v, q_norm, k_norm, pos, p, slots, pool, q, gate);
    const view qi = c.q_in, qo = c.q_rope.qk, g = c.gate, kv = c.kv.k, vv = c.kv.v, sl = c.kv.slots, pk = c.kv.pool_k,
               pv = c.kv.pool_v, ps = c.q_rope.pos, ff = c.q_rope.freq_factors, qn = c.q_norm.w, kn = c.k_norm.w;
    writable(qo);
    if (gate) writable(g);
    need(!pk.b->readonly && !pv.b->readonly, "the pool is a read-only wrapped buffer");
    const int64_t heads = qo.shape[1], kv_heads = kv.shape[1], hd = qo.shape[2];
    if (hd > ATTN_IN_HD) throw unsupported("head_dim past " + std::to_string(ATTN_IN_HD));
    const attn_in_args a{qi.rs, qo.rs, g.rs, kv.rs, vv.rs, sl.rs, qn.rs, kn.rs, u32(heads), u32(kv_heads), u32(hd),
                         gate ? 1u : 0u, q_norm ? 1u : 0u, eps, rope_arguments(c.q_rope, p)};
    // As qk_norm: a simdgroup a head, one a threadgroup under 16 rows, up to 8 from 16.
    const int64_t total = heads + kv_heads, per = qo.rows < 16 ? 1 : std::min<int64_t>(total, 8);
    groups(static_cast<mtl_stream*>(vs), ks(vs).attn_in,
           {input(qi), at(qo), at_or(g, at(qo)), input(kv), input(vv), input(sl), input(ps), input_or(ff, at(ps)),
            input_or(qn, at(ps)), input_or(kn, at(ps)), at(pk), at(pv)},
           a, qo.rows * ((total + per - 1) / per), NSUInteger(per * 32));
  });
}

// Both attention ops use the same Metal views of the shared checker's result.
struct attn_view {
  view q, out, table, k, v, start, lens;

  attn_view(const cpu::attn_checked& a)
      : q(a.q), out(a.out), table(a.table), k(a.k), v(a.v), start(a.start), lens(a.lens) {}
};

// The attention kernels' simdgroups per row: each kv head's q heads in groups of ATTN_GROUP (ops.metal head_group).
int64_t head_groups(const attn_view& a, const larch_kv_pool* pool) {
  return int64_t(pool->n_kv_heads) * ((a.q.shape[1] / pool->n_kv_heads + ATTN_GROUP - 1) / ATTN_GROUP);
}

// Called by both ops after all their checks, so an invalid call gets INVALID_ARG, never UNSUPPORTED. flash false keeps
// a prefill on vv_attn, decode's kernel (larch_attn's spans).
void attend(larch_stream* vs, const attn_view& a, const larch_kv_pool* pool, uint32_t prefill, float scale,
            bool flash_ok = true) {
  writable(a.out);
  // ponytail: each lane holds 2 chunks of 4 dims per head; add chunks if a model with head_dim > 256 lands.
  if (pool->head_dim > ATTN_MAX_HEAD_DIM) throw unsupported("head_dim > 256");
  const attn_args args{a.q.rs, a.out.rs, a.table.rs, a.lens.rs, a.start.rs, u32(a.q.shape[1]), pool->n_kv_heads,
                       pool->head_dim, pool->block_tokens, u32(a.table.rows), prefill, 1, 0, scale, u32(a.q.rows)};
  const std::initializer_list<arg> bufs = {input(a.q),     input(a.k),    input(a.v),
                                           input(a.table), input(a.lens), input_or(a.start, at(a.lens)),
                                           at(a.out)};
  auto* stream = static_cast<mtl_stream*>(vs);
  // Flash prefill reads K and V in place as tensors of whole blocks with 16-byte aligned rows (head_dim a
  // multiple of 8, aligned pool offsets); every other prefill keeps vv_attn. Flash was faster at every chunk
  // of one sequence (M5, 9 Oct, m1 heads, 512-2048 tokens): 1.5x at 1 row, 3.2x at 32, 3.6x at 512.
  // ponytail: a batch of many 1-row sequences leaves 31 of 32 tile rows idle; send it to vv_attn if mixed
  // prefill batches show attention in the profile.
  // Without the tensor API, the simdgroup-matrix flash kernel takes the same calls.
  const int wide = pool->head_dim <= 64 ? 0 : pool->head_dim <= 128 ? 1 : 2;
  const id<MTLComputePipelineState> flash =
      ks(vs).attn_flash[wide] ? ks(vs).attn_flash[wide] : ks(vs).attn_flash_sg[wide];
  if (prefill && flash_ok && flash && pool->block_tokens == ATTN_FLASH_BLOCK && pool->head_dim % 8 == 0 && !(a.k.off % 16) &&
      !(a.v.off % 16)) {
    // Each sequence's rows in tiles: at most one partial tile per sequence.
    // head_dim exactly 64, 128 or 256 on NAX: 64 q rows a tile (32 at 256), K and V read in place, no staging.
    // At 512 rows and 8,192 positions vv_attn_flash took 3.2x MLX's time (21.1 vs 6.5 ms at 16 q / 2 kv heads,
    // d256, M5, 27 Nov).
    if (const id<MTLComputePipelineState> nax = ks(vs).attn_paged_nax[wide];
        nax && pool->head_dim == (64u << wide) && !(a.q.rs % 2) && !(a.out.rs % 2)) {
      const int64_t tile = wide == 2 ? 32 : 64;
      const int64_t tiles = (a.q.rows + tile - 1) / tile + a.table.rows;
      const bool hm = stream->dev->fed.attn_nax_hm;
      groups(stream, nax, bufs, args, hm ? a.q.shape[1] : tiles, wide == 2 ? 256 : 128, hm ? tiles : a.q.shape[1]);
      return;
    }
    const int64_t tile = wide == 2 && flash == ks(vs).attn_flash[wide] ? ATTN_FLASH_ROWS256 : ATTN_FLASH_ROWS;
    groups(stream, flash, bufs, args, (a.q.rows + tile - 1) / tile + a.table.rows, ATTN_FLASH_SG * 32,
           a.q.shape[1]);
    return;
  }
  groups(stream, ks(vs).attn, bufs, args, head_groups(a, pool), 32, a.q.rows);
}

uint32_t automatic_attn_splits(const mtl_stream* stream, const attn_view& a, const larch_kv_pool* pool) {
  const id<MTLComputePipelineState> pipeline = stream->dev->k.attn_split;
  const NSUInteger width = pipeline.threadExecutionWidth;
  const NSUInteger limit = std::min(pipeline.maxTotalThreadsPerThreadgroup, stream->dev->dev.maxThreadsPerThreadgroup.width);
  const uint32_t cores = core_count(stream, pipeline);
  // Each (row, head group, split) simdgroup walks its positions serially, so the split count aims at 4
  // simdgroups per core thread slot. The 4 is measured (M5, 9 Oct, 32 q / 8 kv heads, rows 1/4/16, 128-8192
  // tokens): one simdgroup per slot left 1-row 8k decode at 1/7 of its speed. Past 16, vv_attn_split uses only
  // the splits a sequence has ATTN_SPLIT_POS positions each for (64 for all lost 2x at 128-512 tokens).
  const int64_t groups = cpu::mul(a.q.rows, head_groups(a, pool));
  const int64_t desired = (cpu::mul(int64_t(cores), int64_t(limit / width) * 4) + groups - 1) / groups;
  return u32(std::clamp(desired, int64_t(1), std::min<int64_t>(ATTN_MAX_SPLITS, a.table.cols)));
}

// Split-K decode, or with span 1 larch_attn's causal spans on decode's kernels (vv_attn_split's prefill rows).
void attend_split_k(larch_stream* vs, const attn_view& a, const larch_kv_pool* pool, uint32_t splits, float scale,
                    uint32_t span = 0) {
  const auto* stream = static_cast<const mtl_stream*>(vs);
  const uint32_t split_count = !a.q.rows ? 1 : splits ? splits : automatic_attn_splits(stream, a, pool);
  if (split_count == 1) {
    attend(vs, a, pool, span, scale, false);
    return;
  }
  writable(a.out);
  if (pool->head_dim > ATTN_MAX_HEAD_DIM) throw unsupported("head_dim > 256");

  const arg query = input(a.q), key = input(a.k), value = input(a.v), table = input(a.table),
            lengths = input(a.lens), starts = input_or(a.start, at(a.lens)), output = at(a.out);
  const uint32_t partial_stride = pool->head_dim + 2;
  const int64_t partials = cpu::mul(cpu::mul(a.q.rows, a.q.shape[1]), int64_t(split_count));
  const uint64_t bytes = uint64_t(cpu::mul(cpu::mul(partials, int64_t(partial_stride)), int64_t(sizeof(float))));
  auto* mutable_stream = static_cast<mtl_stream*>(vs);
  if (bytes > mutable_stream->dev->dev.maxBufferLength) throw std::bad_alloc();
  if (mutable_stream->attn_scratch.length < bytes) {
    mutable_stream->attn_scratch = [mutable_stream->dev->dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    if (!mutable_stream->attn_scratch) throw std::bad_alloc();
  }
  const attn_args args{a.q.rs, a.out.rs, a.table.rs, a.lens.rs, a.start.rs, u32(a.q.shape[1]), pool->n_kv_heads,
                       pool->head_dim, pool->block_tokens, u32(a.table.rows), span, split_count, partial_stride, scale,
                       u32(a.q.rows)};
  groups(mutable_stream, ks(vs).attn_split,
         {query, key, value, table, lengths, starts, {mutable_stream->attn_scratch, 0}}, args,
         cpu::mul(head_groups(a, pool), int64_t(split_count)), 32, a.q.rows);
  groups(mutable_stream, ks(vs).attn_split_reduce, {input(arg{mutable_stream->attn_scratch, 0}), output}, args, a.q.shape[1], 32,
         a.q.rows);
}

larch_status attn_decode(larch_stream* vs, const larch_tensor* q, const larch_kv_pool* pool,
                          const larch_tensor* block_table_i32, const larch_tensor* seq_lens_i32, uint32_t splits,
                          float scale, const larch_tensor* out) {
  return run("attn_decode", [&] {
    const attn_view a = cpu::check_attn_decode(sizer(vs), q, pool, block_table_i32, seq_lens_i32, out);
    static_assert(16 <= ATTN_MAX_SPLITS, "the split-K reduce weighs one split per lane");
    attend_split_k(vs, a, pool, std::min(splits, 16u), scale);  // a speed hint (larch.h): 16 is the most kept
  });
}

larch_status attn_prefill(larch_stream* vs, const larch_tensor* q, const larch_kv_pool* pool,
                           const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                           const larch_tensor* q_lens_i32, float scale, const larch_tensor* out) {
  return run("attn_prefill", [&] {
    const attn_view a = cpu::check_attn_prefill(sizer(vs), q, pool, block_table_i32, q_start_pos_i32, q_lens_i32, out);
    attend(vs, a, pool, 1, scale);
  });
}

// larch.h's general entry, causal spans only (a check step's): each row runs as attn_decode's row of seq_len start + j
// + 1 at the same split count, kernels and split bounds, so it is bit-identical to attn_decode at the same nonzero
// splits (the verify contract). A mask, sinks or a window answer UNSUPPORTED, after every check.
// ponytail: no draft-tree mask, attention sinks or sliding window (the engine checks chains, no Metal model has
// either); add 50e0010's position map if a tree drafter lands.
larch_status attn(larch_stream* vs, const larch_tensor* q, const larch_kv_pool* pool,
                   const larch_tensor* block_table_i32, const larch_tensor* q_start_pos_i32,
                   const larch_tensor* q_lens_i32, const larch_tensor* mask_or_null,
                   const larch_tensor* sinks_or_null, uint32_t window, uint32_t splits, float scale,
                   const larch_tensor* out) {
  return run("attn", [&] {
    const attn_view a = cpu::check_attn(sizer(vs), q, pool, block_table_i32, q_start_pos_i32, q_lens_i32,
                                        mask_or_null, sinks_or_null, out);
    if (mask_or_null || sinks_or_null || window) throw unsupported("a mask, attention sinks or a sliding window");
    attend_split_k(vs, a, pool, std::min(splits, 16u), scale, 1);  // attn_decode's split count
  });
}

larch_status qk_norm(larch_stream* vs, const larch_tensor* qk, const larch_tensor* w, float eps) {
  return run("qk_norm", [&] {
    const auto checked = cpu::check_qk_norm(sizer(vs), qk, w);
    const view q = checked.x, wv = checked.w;
    writable(q);
    // Simdgroups a threadgroup, a head each: 8 took prefill from 1.7 to 0.85 ms a call against a threadgroup a head,
    // but decode from 6.9 to 9.0 us (M5 profile, 27 Oct), so under 16 rows one (8.7 us), and a row's heads use every
    // core.
    const int64_t heads = q.shape[1], per = q.rows < 16 ? 1 : std::min<int64_t>(heads, 8);
    groups(static_cast<mtl_stream*>(vs), ks(vs).qk_norm, {at(q), input(wv)},
           rms_args{q.rs, wv.rs, q.rs, u32(q.shape[2]), u32(heads), eps}, q.rows * ((heads + per - 1) / per),
           NSUInteger(per * 32));
  });
}

static_assert(sizeof(sample_row) == sizeof(larch_sample_params) &&
              offsetof(sample_row, top_k) == offsetof(larch_sample_params, top_k) &&
              offsetof(sample_row, repeat_penalty) == offsetof(larch_sample_params, repeat_penalty) &&
              offsetof(sample_row, penalty_window) == offsetof(larch_sample_params, penalty_window));

larch_status sample_prep(larch_stream* vs, const larch_tensor* logits, const larch_sample_params* per_row,
                          const larch_tensor* penalty_ids_i32, const larch_tensor* out_topk_vals_f32,
                          const larch_tensor* out_topk_ids_i32, const larch_tensor* out_lse_f32) {
  return run("sample_prep", [&] {
    auto* s = static_cast<mtl_stream*>(vs);
    const auto checked = cpu::check_sample_prep(sizer(vs), logits, per_row, penalty_ids_i32, out_topk_vals_f32,
                                                 out_topk_ids_i32, out_lse_f32);
    const view lg = checked.logits, vals = checked.vals, ids = checked.ids, lse = checked.lse, pen = checked.penalty_ids;
    const int64_t n = lg.rows, vocab = lg.cols;
    writable(vals);
    writable(ids);
    writable(lse);
    if (!n) return;
    const int64_t parts = (vocab + SP_CHUNK - 1) / SP_CHUNK;
    const NSUInteger merge_threads =
        std::min<NSUInteger>(SP_MERGE_THREADS, ks(vs).sp_merge.maxTotalThreadsPerThreadgroup) / 32 * 32;
    // ponytail: vv_sp_merge takes one part per lane, at most SP_MERGE_THREADS parts (524288 ids); a larger
    // vocabulary needs a second merge level.
    if (parts > int64_t(merge_threads))
      throw unsupported("sample_prep: vocab above " + std::to_string(merge_threads * SP_CHUNK));

    arg src = input(lg);  // the rows vv_sp_part reads, a.lg_rs apart
    const arg vals_at = at(vals), ids_at = at(ids), lse_at = at(lse);
    id<MTLDevice> dev = s->dev->dev;
    // ponytail: a small MTLBuffer per call for per_row; stage it in the stream's staging buffer if the
    // allocation shows in the decode step profile.
    id<MTLBuffer> rows = [dev newBufferWithBytes:per_row length:NSUInteger(n) * sizeof *per_row
                                         options:MTLResourceStorageModeShared];
    if (!rows) throw std::bad_alloc();
    sample_args a{lg.rs, pen.rs, vals.rs, ids.rs, lse.rs, u32(vocab)};  // each dispatch copies it when encoded
    // The engine's default (repeat_penalty 1 or window 0 on every row) reads the logits in place.
    if (pen.b && std::any_of(per_row, per_row + n, [](const larch_sample_params& p) {
          return p.repeat_penalty != 1.0f && p.penalty_window > 0;
        })) {
      const arg pen_at = input(pen);  // reject unsupported addressing before encoding either dispatch
      const uint64_t bytes = uint64_t(cpu::mul(cpu::mul(n, vocab), 4));
      if (bytes > dev.maxBufferLength) throw std::bad_alloc();  // the validation layer aborts past it
      if (s->scratch.length < bytes) {  // work queued with the old one keeps it alive
        s->scratch = [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        if (!s->scratch) throw std::bad_alloc();
      }
      each(s, ks(vs).sp_copy, {src, {s->scratch, 0}}, a, vocab, n);
      each(s, ks(vs).sp_penalty, {src, pen_at, input(arg{rows, 0}), {s->scratch, 0}}, a, pen.cols, n);
      src = input(arg{s->scratch, 0});  // the parts read the penalised copy
      a.lg_rs = uint64_t(vocab) * 4;
    }
    const uint64_t part_bytes = uint64_t(cpu::mul(cpu::mul(n, parts), int64_t(SP_PART * sizeof(float))));
    if (part_bytes > dev.maxBufferLength) throw std::bad_alloc();
    if (s->sp_parts.length < part_bytes) {  // work queued with the old one keeps it alive
      s->sp_parts = [dev newBufferWithLength:part_bytes options:MTLResourceStorageModePrivate];
      if (!s->sp_parts) throw std::bad_alloc();
    }
    groups(s, ks(vs).sp_part, {src, input(arg{rows, 0}), {s->sp_parts, 0}}, a, parts, SP_THREADS, n);
    groups(s, ks(vs).sp_merge, {input(arg{s->sp_parts, 0}), input(arg{rows, 0}), vals_at, ids_at, lse_at}, a, n, merge_threads);
  });
}

}  // namespace

static const vv_ops table = {
    .backend_info = backend_info,
    .device_count = device_count,
    .device_open = device_open,
    .device_props_get = device_props_get,
    .device_close = device_close,
    .buffer_alloc = buffer_alloc,
    .buffer_wrap_host = buffer_wrap_host,
    .buffer_free = buffer_free,
    .upload = upload,
    .download = download,
    .stream_create = stream_create,
    .stream_sync = stream_sync,
    .stream_destroy = stream_destroy,
    .event_record = event_record,
    .event_elapsed_ns = event_elapsed_ns,
    .event_destroy = event_destroy,
    .graph_begin = no_graphs,
    .graph_end = no_graphs,
    .graph_launch = no_graphs,
    .graph_destroy = graph_destroy,
    .embed = embed,
    .rmsnorm = rmsnorm,
    .rope = rope,
    .mmvq = mmvq,
    .dequant = dequant,
    .gemm = gemm,
    .silu_mul = silu_mul,
    .add = add,
    .kv_write = kv_write,
    .attn_decode = attn_decode,
    .attn_prefill = attn_prefill,
    .qk_norm = qk_norm,
    .sample_prep = sample_prep,
    // ponytail: no Q2_K kernels; add them to the per-dtype kernels when a mixed-expert file must run on Metal.
    .no_weights = 1u << LARCH_Q2_K,
    .mmvq_verify = mmvq_verify,
    .mmvq_id = mmvq_id,
    .sigmoid_mul = sigmoid_mul,
    .attn = attn,
    .moe_route = moe_route,
    .gdn_conv = gdn_conv,
    .gdn = gdn,
    .layernorm = layernorm,
    .gelu = gelu,
    .attn_full = attn_full,
    .rope_multi = rope_multi,
    .fused_attn_in = fused_attn_in,
    .draft_conv = draft_conv,
};

const vv_ops* vv_metal_ops() { return &table; }
