// CUDA backend (architecture 6.2): device, buffers, streams, events, graphs, the shared argument checks
// and the ops table. A stream is a cudaStream_t with its own cuBLAS handle. A sticky error is
// LARCH_ERR_DEVICE_LOST and nothing is reset in-process (D2). Reached only through vv_cuda_ops().
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <new>
#include <vector>

#include "common/arch.h"
#include "common/graph.h"
#include "cuda/cuda.cuh"

namespace cuda {

larch_status status_of(cudaError_t e) {
  switch (e) {
    case cudaSuccess: return LARCH_OK;
    case cudaErrorMemoryAllocation: return LARCH_ERR_OOM;
    case cudaErrorECCUncorrectable: return LARCH_ERR_DEVICE_LOST;
    case cudaErrorInvalidConfiguration:
    case cudaErrorLaunchOutOfResources:
    case cudaErrorNoKernelImageForDevice:
    case cudaErrorInvalidDeviceFunction: return LARCH_ERR_LAUNCH;
    // The 7xx codes this backend can meet are the sticky kernel faults: illegal address, launch
    // failure or timeout, device assert, misaligned address, illegal instruction, invalid PC. The
    // rest of the range is peer access, host registration and cooperative launch, which it never
    // calls. A range because hipify has no HIP names for several, and HIP keeps CUDA's numbers.
    default: return e >= 700 && e < 800 ? LARCH_ERR_DEVICE_LOST : LARCH_ERR_DRIVER;
  }
}

// CUDA 13's static runtime has no error strings without a driver (cudaGetErrorString returns NULL).
static std::string error_text(cudaError_t e) {
  const char* s = cudaGetErrorString(e);
  return s ? s : "CUDA error " + std::to_string(int(e));
}

void ck(cudaError_t e, const char* what) {
  if (e == cudaSuccess) return;
  cudaGetLastError();  // a non-sticky error would otherwise fail the next launch's check
  throw failure{status_of(e), std::string(what) + ": " + error_text(e)};
}

void ck(cublasStatus_t e, const char* what) {
  if (e == CUBLAS_STATUS_SUCCESS) return;
  // A kernel fault inside cuBLAS shows up as a CUDA error; prefer it, it may be DEVICE_LOST.
  const cudaError_t cuda = cudaGetLastError();
  if (cuda != cudaSuccess) ck(cuda, what);
  const larch_status st = e == CUBLAS_STATUS_ALLOC_FAILED     ? LARCH_ERR_OOM
                           : e == CUBLAS_STATUS_NOT_SUPPORTED  ? LARCH_ERR_UNSUPPORTED
                           : e == CUBLAS_STATUS_EXECUTION_FAILED ? LARCH_ERR_LAUNCH
                                                                 : LARCH_ERR_DRIVER;
  throw failure{st, std::string(what) + ": cuBLAS status " + std::to_string(int(e))};
}

view::view(const cpu::meta& m) : meta(m), p(m.buf ? static_cast<const buffer*>(m.buf)->ptr + m.offset : nullptr) {}

cpu::sizer sizer(const stream& s) {
  return [ordinal = s.ordinal](const larch_buffer* b, const std::string& what) -> uint64_t {
    if (!b || b->ops != vv_cuda_ops()) throw bad_arg(what + " is null or not a CUDA tensor");
    const auto* cb = static_cast<const buffer*>(b);
    if (cb->ordinal != ordinal) throw bad_arg(what + " is on another device than the stream");
    return cb->bytes;
  };
}

// The debug-only device-loss kernel (architecture 9.11): one thread stores through the null pointer the
// host passes (a parameter, so the compiler cannot fold the store into a trap), an illegal address.
// Built on every backend (CI wants device code in each .cu's object); only the op below is CUDA-only.
__global__ void fault_k(int* p) { *p = 0; }

}  // namespace cuda

using namespace cuda;

// The entry points below run outside run(): each maps its own errors, clearing them as ck() does.
static larch_status fail(cudaError_t e, const std::string& what) {
  cudaGetLastError();
  return vv_fail(status_of(e), what + ": " + error_text(e));
}

static larch_status backend_info(larch_backend* out, uint32_t* runtime_version, uint32_t* driver_version) {
#if defined(__HIP_PLATFORM_AMD__)
  *out = LARCH_BACKEND_HIP;
#else
  *out = LARCH_BACKEND_CUDA;
#endif
  int rt = 0, drv = 0;  // the driver version is 0 when no driver is installed
  if (cudaError_t e = cudaRuntimeGetVersion(&rt)) return fail(e, "cudaRuntimeGetVersion");
  if (cudaError_t e = cudaDriverGetVersion(&drv)) return fail(e, "cudaDriverGetVersion");
  *runtime_version = uint32_t(rt);
  *driver_version = uint32_t(drv);
  return LARCH_OK;
}

#if !defined(__HIP_PLATFORM_AMD__)
static constexpr int built[] = {__CUDA_ARCH_LIST__};  // as __CUDA_ARCH__ spells them: 860, 890, 1200
#endif

// Whether this build has code for runtime device `id`, from its architecture alone (common/arch.h);
// `name` gets that architecture. A device whose architecture cannot be read is kept: opening it
// reports why.
static bool has_code(int id, std::string& name) {
#if defined(__HIP_PLATFORM_AMD__)
#if defined(LARCH_HIP_ARCHS)
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, id)) {
    cudaGetLastError();
    return true;
  }
  name = p.gcnArchName;
  return arch::hip_runs(LARCH_HIP_ARCHS, p.gcnArchName);
#else
  // ponytail: every AMD device counts until kernels/CMakeLists.txt passes LARCH_HIP_ARCHS (a1)
  return true;
#endif
#else
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id) ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id)) {
    cudaGetLastError();
    return true;
  }
  name = "sm_" + std::to_string(major * 10 + minor);
  return arch::cuda_runs(built, std::size(built), major * 100 + minor * 10);
#endif
}

// No driver (or one too old for the runtime) and no device both count as 0 devices, so GPU-less
// machines run the CPU reference; Device::open reports the driver version on its own. The rest are
// the devices this build has code for, so public ordinals depend only on the build and the hardware.
// `skipped` names the others for an error message.
static larch_status usable_devices(std::vector<int>& ids, std::string* skipped = nullptr) {
  int n = 0;
  const cudaError_t e = cudaGetDeviceCount(&n);
  if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver) {
    cudaGetLastError();
    n = 0;
  } else if (e) {
    return fail(e, "cudaGetDeviceCount");
  }
  for (int id = 0; id < n; ++id) {
    std::string name;
    if (has_code(id, name)) {
      ids.push_back(id);
    } else if (skipped) {
      *skipped += (skipped->empty() ? "runtime id " : ", runtime id ") + std::to_string(id) + " (" + name + ")";
    }
  }
  return LARCH_OK;
}

static larch_status device_count(int32_t* out) {
  std::vector<int> ids;
  if (larch_status st = usable_devices(ids)) return st;
  *out = int32_t(ids.size());
  return LARCH_OK;
}

static larch_status device_open(int32_t ordinal, larch_device** out) {
  std::vector<int> ids;
  std::string skipped;
  if (larch_status st = usable_devices(ids, &skipped)) return st;
  // UNSUPPORTED when a GPU is there but this build has no code for it: the app says so, as it
  // does not for a machine with no GPU at all.
  if (ordinal < 0 || size_t(ordinal) >= ids.size())
    return vv_fail(skipped.empty() ? LARCH_ERR_INVALID_ARG : LARCH_ERR_UNSUPPORTED,
                   "no CUDA device " + std::to_string(ordinal) +
                       (skipped.empty() ? "" : "; no code in this build for " + skipped));
  // Only this boundary translates dense public ordinals. Handles and every per-device call keep
  // the real runtime id, including streams, buffers and events derived from this device.
  const int id = ids[size_t(ordinal)];
  cudaDeviceProp cp;
  int mem_khz = 0, rt = 0, drv = 0;
  cudaMemPool_t pool;
  uint64_t keep = UINT64_MAX;
  if (cudaError_t e = cudaSetDevice(id)) return fail(e, "cudaSetDevice");
  if (cudaError_t e = cudaGetDeviceProperties(&cp, id)) return fail(e, "cudaGetDeviceProperties");
  if (cudaError_t e = cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, id))
    return fail(e, "cudaDeviceGetAttribute");
  // Op scratch (cudaMallocAsync) stays in the pool between steps instead of going back to the driver
  // at every sync.
  if (cudaError_t e = cudaDeviceGetDefaultMemPool(&pool, id)) return fail(e, "cudaDeviceGetDefaultMemPool");
  if (cudaError_t e = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep))
    return fail(e, "cudaMemPoolSetAttribute");
  cudaRuntimeGetVersion(&rt);
  cudaDriverGetVersion(&drv);

  larch_device_props p = {};
  std::strncpy(p.name, cp.name, sizeof p.name - 1);
  p.backend = LARCH_BACKEND_CUDA;
  p.arch_code = uint32_t(cp.major * 10 + cp.minor);
#if defined(__HIP_PLATFORM_AMD__)
  p.backend = LARCH_BACKEND_HIP;
  p.arch_code = uint32_t(std::strtoul(cp.gcnArchName + 3, nullptr, 10));  // "gfx1100:sramecc-..." -> 1100
#endif
  p.compute_units = uint32_t(cp.multiProcessorCount);
  p.warp_size = uint32_t(cp.warpSize);
  p.max_threads_per_group = uint32_t(cp.maxThreadsPerBlock);
  p.shared_mem_per_group = cp.sharedMemPerBlock;
  p.l2_bytes = uint64_t(cp.l2CacheSize);
  p.vram_total = cp.totalGlobalMem;
  // Double data rate: two transfers per memory clock, bus width in bits.
  p.peak_gbps = float(2.0 * mem_khz * 1e3 * (cp.memoryBusWidth / 8) / 1e9);
#if defined(__HIP_PLATFORM_AMD__)
  p.peak_gbps = 0;  // unknown: ROCm's memory clock on GDDR6 is not half the data rate; measure on RDNA first
#endif
  p.runtime_version = uint32_t(rt);
  p.driver_version = uint32_t(drv);
  // Graphs: begin/end/launch below. No bf16 op, no async-copy op and discrete memory: those stay clear.
  p.flags = LARCH_CAP_GRAPHS;
  if (!(*out = new (std::nothrow) device{{vv_cuda_ops()}, id, p})) return vv_fail(LARCH_ERR_OOM, "device_open");
  return LARCH_OK;
}

// Op scratch stays cached in the pool between steps (device_open); it is handed back first, so
// vram_free counts it as free.
static larch_status device_props_get(const larch_device* vd, larch_device_props* out) {
  const auto* d = static_cast<const device*>(vd);
  size_t free = 0, total = 0;
  cudaMemPool_t pool;
  if (cudaError_t e = cudaSetDevice(d->ordinal)) return fail(e, "cudaSetDevice");
  if (cudaError_t e = cudaDeviceGetDefaultMemPool(&pool, d->ordinal)) return fail(e, "cudaDeviceGetDefaultMemPool");
  if (cudaError_t e = cudaMemPoolTrimTo(pool, 0)) return fail(e, "cudaMemPoolTrimTo");
  if (cudaError_t e = cudaMemGetInfo(&free, &total)) return fail(e, "cudaMemGetInfo");
  *out = d->props;
  out->vram_free = out->recommended_working_set = free;
  return LARCH_OK;
}

static void device_close(larch_device* d) { delete static_cast<device*>(d); }

// cudaMalloc on the current device. Op scratch stays cached in the pool (device_open), so an allocation
// that finds the device full hands that back to the driver and tries once more.
static cudaError_t device_malloc(void** p, uint64_t bytes, int ordinal) {
  cudaError_t e = cudaMalloc(p, bytes);
  cudaMemPool_t pool;
  if (e != cudaErrorMemoryAllocation || cudaDeviceGetDefaultMemPool(&pool, ordinal) || cudaMemPoolTrimTo(pool, 0))
    return e;
  cudaGetLastError();
  return cudaMalloc(p, bytes);
}

static larch_status buffer_alloc(larch_device* vd, uint64_t bytes, larch_buffer** out) {
  const auto* d = static_cast<const device*>(vd);
  if (!bytes) return vv_fail(LARCH_ERR_INVALID_ARG, "buffer_alloc: 0 bytes");
  void* p = nullptr;
  if (cudaError_t e = cudaSetDevice(d->ordinal)) return fail(e, "cudaSetDevice");
  if (cudaError_t e = device_malloc(&p, bytes, d->ordinal))
    return fail(e, "buffer_alloc: " + std::to_string(bytes) + " bytes");
  if (!(*out = new (std::nothrow) buffer{{vv_cuda_ops()}, d->ordinal, static_cast<uint8_t*>(p), bytes})) {
    cudaFree(p);
    return vv_fail(LARCH_ERR_OOM, "buffer_alloc: host memory");
  }
  return LARCH_OK;
}

static larch_status buffer_wrap_host(larch_device*, void*, uint64_t, larch_buffer**) {
  return vv_fail(LARCH_ERR_UNSUPPORTED, "buffer_wrap_host: Metal only");
}

// cudaFree waits for the device, so work still queued on the buffer finishes first (larch-kernels'
// Buffer relies on it).
static void buffer_free(larch_buffer* vb) {
  auto* b = static_cast<buffer*>(vb);
  cudaSetDevice(b->ordinal);
  cudaFree(b->ptr);
  delete b;
}

// b is a buffer of s's device, host is set when bytes are moved, and [off, off + bytes) lies inside b.
static const buffer* in_range(const stream* s, const larch_buffer* vb, uint64_t off, const void* host,
                              uint64_t bytes) {
  if (!vb || vb->ops != vv_cuda_ops() || (bytes && !host)) return nullptr;
  const auto* b = static_cast<const buffer*>(vb);
  return b->ordinal == s->ordinal && off <= b->bytes && bytes <= b->bytes - off ? b : nullptr;
}

// Uploads go through the stream's two pinned staging halves, so `src` is free when upload returns
// (larch.h), whatever kind of host memory it is, and a model loaded in many uploads pins 64 MiB, not its
// size. One half fills while the other's copies run; refilling a half waits for its own copies only.
static const uint64_t staging_half = 32 << 20;

// A copy queued while s captures a graph would replay host memory that is gone (or staging that was
// reused), and a sync or event record would break the capture, so all of them refuse it.
static larch_status not_capturing(const stream* s, const char* what) {
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (cudaError_t e = cudaStreamIsCapturing(s->s, &cap)) return fail(e, what);
  if (cap != cudaStreamCaptureStatusNone)
    return vv_fail(LARCH_ERR_INVALID_ARG, std::string(what) + ": the stream is capturing a graph");
  return LARCH_OK;
}

static larch_status stream_sync(larch_stream* vs) {
  auto* s = static_cast<stream*>(vs);
  if (cudaError_t e = cudaSetDevice(s->ordinal)) return fail(e, "cudaSetDevice");
  if (larch_status st = not_capturing(s, "stream_sync")) return st;  // a sync would break the capture
  if (cudaError_t e = cudaStreamSynchronize(s->s)) return fail(e, "cudaStreamSynchronize");
  s->staged = 0;
  return LARCH_OK;
}

static larch_status upload(larch_stream* vs, larch_buffer* vb, uint64_t off, const void* src, uint64_t bytes) {
  auto* s = static_cast<stream*>(vs);
  const buffer* b = in_range(s, vb, off, src, bytes);
  if (!b) return vv_fail(LARCH_ERR_INVALID_ARG, "upload: bad buffer, source or range");
  if (cudaError_t e = cudaSetDevice(s->ordinal)) return fail(e, "cudaSetDevice");
  if (larch_status st = not_capturing(s, "upload")) return st;
  if (!s->staging && bytes)
    if (cudaError_t e = cudaHostAlloc(reinterpret_cast<void**>(&s->staging), 2 * staging_half, cudaHostAllocDefault))
      return fail(e, "upload: pinned staging");
  for (uint64_t at = 0; at < bytes;) {
    if (s->staged == staging_half) {  // switch halves once the other's copies are done (never recorded = done)
      if (cudaError_t e = cudaEventRecord(s->staged_done[s->half], s->s)) return fail(e, "upload");
      if (cudaError_t e = cudaEventSynchronize(s->staged_done[s->half ^ 1])) return fail(e, "upload");
      s->half ^= 1;
      s->staged = 0;
    }
    uint8_t* stage = s->staging + s->half * staging_half + s->staged;
    const uint64_t n = std::min(staging_half - s->staged, bytes - at);
    std::memcpy(stage, static_cast<const uint8_t*>(src) + at, n);
    if (cudaError_t e = cudaMemcpyAsync(b->ptr + off + at, stage, n, cudaMemcpyHostToDevice, s->s))
      return fail(e, "upload");
    s->staged += n;
    at += n;
  }
  return LARCH_OK;
}

// Straight into dst: it stays put until the next stream_sync (larch.h), which this copy precedes.
static larch_status download(larch_stream* vs, const larch_buffer* vb, uint64_t off, void* dst, uint64_t bytes) {
  auto* s = static_cast<stream*>(vs);
  const buffer* b = in_range(s, vb, off, dst, bytes);
  if (!b) return vv_fail(LARCH_ERR_INVALID_ARG, "download: bad buffer, target or range");
  if (!bytes) return LARCH_OK;
  if (cudaError_t e = cudaSetDevice(s->ordinal)) return fail(e, "cudaSetDevice");
  if (larch_status st = not_capturing(s, "download")) return st;
  if (cudaError_t e = cudaMemcpyAsync(dst, b->ptr + off, bytes, cudaMemcpyDeviceToHost, s->s))
    return fail(e, "download");
  return LARCH_OK;
}

// Frees what a stream holds, whichever parts exist (stream_create's failure paths, stream_destroy).
static void release(stream* s) {
  if (s->blas) cublasDestroy(s->blas);
  cudaFree(s->blas_workspace);
  cudaFree(s->capture_scratch);
  cudaFreeHost(s->staging);
  for (cudaEvent_t e : s->staged_done)
    if (e) cudaEventDestroy(e);
  cudaStreamDestroy(s->s);
  delete s;
}

static larch_status stream_create(larch_device* vd, larch_stream** out) {
  const auto* d = static_cast<const device*>(vd);
  if (cudaError_t e = cudaSetDevice(d->ordinal)) return fail(e, "cudaSetDevice");
  cudaStream_t cs = nullptr;
  if (cudaError_t e = cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking)) return fail(e, "cudaStreamCreate");
  // cuBLAS's own workspace, set once so a captured gemm never allocates (cuBLAS's sizes: 32 MiB from
  // Hopper on, 4 MiB before).
  const size_t ws_bytes = d->props.arch_code >= 90 ? 32u << 20 : 4u << 20;
  static std::atomic<uint64_t> ids{0};
  auto* s = new (std::nothrow) stream{{vv_cuda_ops()}, d->ordinal, ids++, int(d->props.warp_size),
                                    int(d->props.compute_units), int(d->props.arch_code), cs, nullptr, nullptr};
  if (!s) {
    cudaStreamDestroy(cs);
    return vv_fail(LARCH_ERR_OOM, "stream_create: host memory");
  }
  // Frees the parts made so far on every early return, also when formatting its error runs out of host
  // memory and throws (dispatch turns that into OOM).
  std::unique_ptr<stream, void (*)(stream*)> made(s, release);
  if (cudaError_t e = cudaEventCreateWithFlags(&s->staged_done[0], cudaEventDisableTiming))
    return fail(e, "stream_create: staging event");
  if (cudaError_t e = cudaEventCreateWithFlags(&s->staged_done[1], cudaEventDisableTiming))
    return fail(e, "stream_create: staging event");
  if (cudaError_t e = device_malloc(&s->blas_workspace, ws_bytes, d->ordinal))
    return fail(e, "stream_create: cuBLAS workspace");
  if (cublasStatus_t e = cublasCreate(&s->blas))
    return vv_fail(e == CUBLAS_STATUS_ALLOC_FAILED ? LARCH_ERR_OOM : LARCH_ERR_DRIVER,
                   "stream_create: cublasCreate status " + std::to_string(int(e)));
  if (cublasSetStream(s->blas, cs) ||
#if defined(__HIP_PLATFORM_AMD__)
      // Older hipify versions do not map SetWorkspace. rocBLAS accepts only the default math mode;
      // GEMM still requests f32 accumulation explicitly in common/blas.h.
      hipblasSetWorkspace(s->blas, s->blas_workspace, ws_bytes) ||
      hipblasSetMathMode(s->blas, HIPBLAS_DEFAULT_MATH))
#else
      cublasSetWorkspace(s->blas, s->blas_workspace, ws_bytes) ||
      // Split-K partial sums in f32, never in c's f16 (larch.h: every dot product accumulates in f32).
      cublasSetMathMode(s->blas,
                        cublasMath_t(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)))
#endif
    return vv_fail(LARCH_ERR_DRIVER, "stream_create: cuBLAS stream, workspace or math mode");
  s->shared_mem_per_block = int(d->props.shared_mem_per_group);
#if !defined(__HIP_PLATFORM_AMD__)
  s->image_arch = arch::cuda_image(built, std::size(built), s->arch_code * 10);
  quant_i8_init(*s);  // HIP registers no INT8 slot
#endif
  *out = made.release();
  return LARCH_OK;
}

// Queued work still runs, as with any destroyed CUDA stream; the staging buffer is freed after it.
static void stream_destroy(larch_stream* vs) {
  auto* s = static_cast<stream*>(vs);
  cudaSetDevice(s->ordinal);
  cudaStreamSynchronize(s->s);
  release(s);
}

struct event : larch_event {
  int ordinal;
  cudaEvent_t e;
};

static larch_status event_record(larch_stream* vs, larch_event** out) {
  auto* s = static_cast<stream*>(vs);
  cudaEvent_t e = nullptr;
  if (cudaError_t r = cudaSetDevice(s->ordinal)) return fail(r, "cudaSetDevice");
  if (larch_status st = not_capturing(s, "event_record")) return st;
  if (cudaError_t r = cudaEventCreate(&e)) return fail(r, "cudaEventCreate");
  if (cudaError_t r = cudaEventRecord(e, s->s)) {
    cudaEventDestroy(e);
    return fail(r, "cudaEventRecord");
  }
  if (!(*out = new (std::nothrow) event{{vv_cuda_ops()}, s->ordinal, e})) {
    cudaEventDestroy(e);
    return vv_fail(LARCH_ERR_OOM, "event_record: host memory");
  }
  return LARCH_OK;
}

// Waits for both; CUDA's resolution is about half a microsecond.
static larch_status event_elapsed_ns(const larch_event* va, const larch_event* vb, uint64_t* out) {
  const auto* a = static_cast<const event*>(va);
  const auto* b = static_cast<const event*>(vb);
  if (b->ops != vv_cuda_ops() || b->ordinal != a->ordinal)
    return vv_fail(LARCH_ERR_INVALID_ARG, "event_elapsed_ns: b is not a CUDA event of a's device");
  float ms = 0;
  if (cudaError_t e = cudaSetDevice(a->ordinal)) return fail(e, "cudaSetDevice");
  if (cudaError_t e = cudaEventSynchronize(a->e)) return fail(e, "cudaEventSynchronize");
  if (cudaError_t e = cudaEventSynchronize(b->e)) return fail(e, "cudaEventSynchronize");
  if (cudaError_t e = cudaEventElapsedTime(&ms, a->e, b->e)) return fail(e, "cudaEventElapsedTime");
  if (ms < 0) return vv_fail(LARCH_ERR_INVALID_ARG, "event_elapsed_ns: b was recorded before a");
  *out = uint64_t(double(ms) * 1e6);
  return LARCH_OK;
}

static void event_destroy(larch_event* ve) {
  auto* e = static_cast<event*>(ve);
  cudaEventDestroy(e->e);
  delete e;
}

// A captured gemm keeps the address of its stream's cuBLAS workspace, so a graph replays only on the
// stream that captured it: that workspace is then alive (the stream is) and used by one queue at a time.
struct graph : larch_graph {
  uint64_t stream;  // the capturing stream's id
  cudaGraphExec_t exec;
};

// graph_begin and graph_end must be paired (larch-kernels' Graph::capture always does). Between them
// every op on the stream is recorded, not run, and leases its scratch from the stream's arena; a
// begin left open keeps the stream capturing, so its later ops never run, a sync breaks the capture
// and a second begin is INVALID_ARG.
static larch_status graph_begin(larch_stream* vs) {
  auto* s = static_cast<stream*>(vs);
  if (cudaError_t e = cudaSetDevice(s->ordinal)) return fail(e, "cudaSetDevice");
  if (s->capturing) return vv_fail(LARCH_ERR_INVALID_ARG, "graph_begin: stream already capturing");
  // Allocate before capture. A graph never owns scratch allocation nodes, and all its addresses
  // remain live until this stream is synchronized and destroyed. An allocation failure leaves the
  // existing eager path available; sticky device errors still propagate normally.
  if (!s->capture_scratch) {
    void* arena = nullptr;
    const cudaError_t e = device_malloc(&arena, capture_scratch_bytes, s->ordinal);
    if (e == cudaErrorMemoryAllocation) {
      cudaGetLastError();
      return vv_fail(LARCH_ERR_UNSUPPORTED, "graph_begin: bounded scratch arena does not fit");
    }
    if (e) return fail(e, "graph_begin: scratch arena");
    s->capture_scratch = static_cast<uint8_t*>(arena);
  }
  if (cudaError_t e = vv_graph_begin(s->s)) return fail(e, "graph_begin");
  s->capture_used = 0;
  s->capturing = true;
  return LARCH_OK;
}

static larch_status graph_end(larch_stream* vs, larch_graph** out) {
  auto* s = static_cast<stream*>(vs);
  cudaGraphExec_t exec = nullptr;
  // Whatever fails below (the device switch, EndCapture, even an exception), later eager scratch
  // must use its normal stream-ordered allocator again.
  s->capturing = false;
  s->capture_used = 0;
  if (cudaError_t e = cudaSetDevice(s->ordinal)) return fail(e, "cudaSetDevice");
  const cudaError_t ended = vv_graph_end(s->s, &exec);
  if (ended == cudaErrorNotSupported)
    return vv_fail(LARCH_ERR_UNSUPPORTED, "graph_end: allocation nodes or node bound exceed capture policy");
  if (ended) return fail(ended, "graph_end");
  if (!(*out = new (std::nothrow) graph{{vv_cuda_ops()}, s->id, exec})) {
    vv_graph_destroy(exec);
    return vv_fail(LARCH_ERR_OOM, "graph_end: host memory");
  }
  return LARCH_OK;
}

static larch_status graph_launch(larch_graph* vg, larch_stream* vs) {
  const auto* g = static_cast<const graph*>(vg);
  auto* s = static_cast<stream*>(vs);
  if (!vs || vs->ops != vv_cuda_ops() || s->id != g->stream)
    return vv_fail(LARCH_ERR_INVALID_ARG, "graph_launch: a graph replays only on the stream that captured it");
  if (cudaError_t e = cudaSetDevice(s->ordinal)) return fail(e, "cudaSetDevice");
  if (cudaError_t e = vv_graph_launch(g->exec, s->s)) return fail(e, "graph_launch");
  return LARCH_OK;
}

static void graph_destroy(larch_graph* vg) {
  auto* g = static_cast<graph*>(vg);
  vv_graph_destroy(g->exec);
  delete g;
}

// The debug-only device-loss op (architecture 9.11): queues fault_k like any op; the fault surfaces at
// the next sync as DEVICE_LOST and sticks. On a context already lost, cudaSetDevice, the capture query
// or the launch check reports DEVICE_LOST instead. Refused while the stream captures: a captured fault
// would replay. Not on HIP: ROCm aborts the process on a GPU page fault ("Memory access fault by GPU
// node") instead of returning it, which would skip the engine's ordered fault path (D2), so HIP keeps
// the synthetic host error.
#if !defined(__HIP_PLATFORM_AMD__)
static larch_status fault_device_lost(larch_stream* vs) {
  auto* s = static_cast<stream*>(vs);
  if (cudaError_t e = cudaSetDevice(s->ordinal)) return fail(e, "cudaSetDevice");
  if (larch_status st = not_capturing(s, "fault_device_lost")) return st;
  return run("fault_device_lost", vs, [](const stream& s) { launch(fault_k, 1, 1, s.s, static_cast<int*>(nullptr)); });
}
#endif

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
    .graph_begin = graph_begin,
    .graph_end = graph_end,
    .graph_launch = graph_launch,
    .graph_destroy = graph_destroy,
    .embed = cuda::embed,
    .rmsnorm = cuda::rmsnorm,
    .rope = cuda::rope,
    .mmvq = cuda::mmvq,
    .dequant = cuda::dequant,
    .gemm = cuda::gemm_rows,
    .silu_mul = cuda::silu_mul,
    .add = cuda::add,
    .kv_write = cuda::kv_write,
    .attn_decode = cuda::attn_decode,
    .attn_prefill = cuda::attn_prefill,
    .qk_norm = cuda::qk_norm,
    .sample_prep = cuda::sample_prep,
#if !defined(__HIP_PLATFORM_AMD__)
    .fault_device_lost = fault_device_lost,
    .quant_gemm = cuda::quant_gemm,
    .quant_gemm_i8 = cuda::quant_gemm_i8,
    .quant_gemm_i8_split = cuda::quant_gemm_i8_split,
    // ponytail: HIP keeps the default (Q3_K refused) and builds no Q3_K kernels; take Q3_K there when ctest ops
    // passes it on an RDNA card.
    .no_weights = 0,
    .mmvq_verify = cuda::mmvq_verify,
#endif
    .mmvq_id = cuda::mmvq_id,
    .mmvq_id_mixed = cuda::mmvq_id_mixed,
    .sigmoid_mul = cuda::sigmoid_mul,
    .attn = cuda::attn,
    .moe_route = cuda::moe_route,
// ponytail: gdn.cu's warps are 32 lanes; HIP answers UNSUPPORTED until an RDNA run checks them (wave32).
// Likewise the training entries: hipBLAS's sgemm and these kernels are unbuilt and unchecked on an RDNA card.
#if !defined(__HIP_PLATFORM_AMD__)
    .gdn_conv = cuda::gdn_conv,
    .gdn = cuda::gdn,
    .gemm_ex = cuda::gemm_ex,
    .rmsnorm_f32 = cuda::rmsnorm_f32,
    .rmsnorm_backward = cuda::rmsnorm_backward,
    .rope_f32 = cuda::rope_f32,
    .silu_mul_f32 = cuda::silu_mul_f32,
    .silu_mul_backward = cuda::silu_mul_backward,
    .softmax = cuda::softmax,
    .softmax_backward = cuda::softmax_backward,
#endif
    .layernorm = cuda::layernorm,
    .gelu = cuda::gelu,
    .attn_full = cuda::attn_full,
    .rope_multi = cuda::rope_multi,
// ponytail: the fused entries reuse quant.cu's CUDA-only decoders; HIP answers UNSUPPORTED and runs each sequence.
#if !defined(__HIP_PLATFORM_AMD__)
    .fused_moe_gate_up = cuda::fused_moe_gate_up,
    .fused_moe_down = cuda::fused_moe_down,
    .fused_moe_in = cuda::fused_moe_in,
    .fused_gdn_in = cuda::fused_gdn_in,
    .fused_proj_add = cuda::fused_proj_add,
    .fused_gdn = cuda::fused_gdn,
    .fused_attn_in = cuda::fused_attn_in,
#endif
    .draft_conv = cuda::draft_conv,
};

const vv_ops* vv_cuda_ops() { return &table; }
