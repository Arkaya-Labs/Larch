#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../tests/testdata.h"
#include "args.h"
#include "larch.h"

extern "C" const unsigned char larch_metallib[];
extern "C" const std::size_t larch_metallib_len;

namespace {

constexpr uint64_t kMaxIterations = 1000;
constexpr uint64_t kMaxDimension = 1U << 20;
constexpr uint64_t kMaxCopyBytes = 1ULL << 30;
constexpr uint64_t kMaxCopyPayloadPerCommandBuffer = kMaxCopyBytes;
constexpr uint64_t kZeroChunkBytes = 1ULL << 20;
constexpr uint64_t kCopyVectorBytes = sizeof(uint32_t) * 4;

struct arguments {
  std::string mode;
  std::unordered_map<std::string, std::string> values;
};

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void usage() {
  std::puts("usage:");
  std::puts("  metal_perf copy --device ORDINAL --bytes BYTES --iterations COUNT");
  std::puts("  metal_perf mmvq --device ORDINAL --format FORMAT --rows ROWS --n COLS --k WIDTH --iterations COUNT");
  std::puts("  metal_perf st --device ORDINAL --format FORMAT|all --n COLS --k WIDTH --iterations COUNT");
  std::puts("  metal_perf dequant --device ORDINAL --format FORMAT --n ROWS --k WIDTH --iterations COUNT");
  std::puts("  metal_perf mmvq-id --device ORDINAL --format FORMAT --experts COUNT --rows ROWS --slots USED --n COLS "
            "--k WIDTH --weighted 0|1 --iterations COUNT");
  std::puts("  metal_perf repack-id --device ORDINAL --experts COUNT --rows ROWS --slots USED --n COLS --k WIDTH "
            "--weighted 0|1 --iterations COUNT   (Q3_K, k a multiple of 512: the repacked-expert prototype)");
  std::puts("  metal_perf affine-q3 --device ORDINAL --rows ROWS --n COLS --k WIDTH --iterations COUNT   (Q3_K, k a "
            "multiple of 512: the exact MLX-style affine prototype)");
  std::puts("  metal_perf gemm --device ORDINAL --m ROWS --n COLS --k WIDTH --iterations COUNT");
  std::puts("  metal_perf add --device ORDINAL --d WIDTH --iterations COUNT");
  std::puts("  metal_perf linear --device ORDINAL --format FORMAT --m ROWS --n COLS --k WIDTH --iterations COUNT");
  std::puts("  metal_perf attn-decode --device ORDINAL --rows ROWS --seq-len TOKENS --heads HEADS --kv-heads HEADS "
            "--head-dim DIM --splits 0..16 --scale SCALE --iterations COUNT");
  std::puts("  metal_perf attn-prefill --device ORDINAL --rows CHUNK --seq-len TOKENS --heads HEADS --kv-heads HEADS "
            "--head-dim DIM --scale SCALE --iterations COUNT");
  std::puts("  metal_perf sample --device ORDINAL --rows ROWS --vocab IDS --top-k 0..64 --iterations COUNT");
  std::puts("formats: q3_k q4_k q5_k q6_k q8_0 q4_0 q5_0");
  std::puts("All shape and timing inputs are required. Run timing commands under flock ~/src/gpu.lock.");
}

arguments parse_arguments(int argc, char** argv) {
  if (argc < 2) fail("missing measurement mode; pass --help for usage");
  arguments parsed{argv[1], {}};
  if (parsed.mode != "copy" && parsed.mode != "mmvq" && parsed.mode != "st" && parsed.mode != "mmvq-id" && parsed.mode != "repack-id" && parsed.mode != "affine-q3" && parsed.mode != "add" && parsed.mode != "dequant" &&
      parsed.mode != "attn-decode" &&
      parsed.mode != "attn-prefill" && parsed.mode != "gemm" && parsed.mode != "linear" && parsed.mode != "sample")
    fail("unknown measurement mode '" + parsed.mode + "'");
  for (int index = 2; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag.rfind("--", 0) != 0) fail("expected an option, got '" + flag + "'");
    if (index + 1 == argc) fail("missing value for " + flag);
    const std::string value = argv[++index];
    if (!parsed.values.emplace(flag, value).second) fail("duplicate option " + flag);
  }
  return parsed;
}

const std::string& required(const arguments& parsed, const char* name) {
  const auto found = parsed.values.find(name);
  if (found == parsed.values.end()) fail(std::string("missing required option ") + name);
  return found->second;
}

void require_only(const arguments& parsed, std::initializer_list<const char*> names) {
  for (const auto& [name, value] : parsed.values) {
    (void)value;
    bool known = false;
    for (const char* allowed : names) known = known || name == allowed;
    if (!known) fail("option " + name + " does not apply to " + parsed.mode);
  }
}

uint64_t parse_u64(const std::string& text, const char* name) {
  uint64_t value = 0;
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr != end) fail(std::string(name) + " must be an unsigned integer");
  return value;
}

uint64_t bounded(const arguments& parsed, const char* name, uint64_t minimum, uint64_t maximum) {
  const uint64_t value = parse_u64(required(parsed, name), name);
  if (value < minimum || value > maximum)
    fail(std::string(name) + " must be in [" + std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
  return value;
}

float parse_scale(const arguments& parsed) {
  const std::string& text = required(parsed, "--scale");
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end != text.c_str() + text.size() || !std::isfinite(value) || value <= 0 ||
      value > std::numeric_limits<float>::max())
    fail("--scale must be a finite positive number");
  return static_cast<float>(value);
}

uint64_t product(uint64_t left, uint64_t right, const char* what) {
  if (left && right > std::numeric_limits<uint64_t>::max() / left)
    fail(std::string(what) + " is too large");
  return left * right;
}

uint64_t sum(uint64_t left, uint64_t right, const char* what) {
  if (right > std::numeric_limits<uint64_t>::max() - left) fail(std::string(what) + " is too large");
  return left + right;
}

int64_t dimension(uint64_t value, const char* name) {
  if (value > uint64_t(std::numeric_limits<int64_t>::max())) fail(std::string(name) + " is too large");
  return static_cast<int64_t>(value);
}

void check(larch_status status, const larch_device* device, const char* operation) {
  if (status == LARCH_OK) return;
  const char* detail = larch_last_error(device);
  fail(std::string(operation) + " failed" + (detail && *detail ? ": " + std::string(detail) : ""));
}

struct session {
  larch_device* device = nullptr;
  larch_stream* stream = nullptr;
  larch_device_props props = {};
  std::vector<larch_buffer*> buffers;

  explicit session(uint64_t ordinal) {
    int32_t count = 0;
    check(larch_device_count(&count), nullptr, "larch_device_count");
    if (count <= 0) fail("no Metal device with Metal 4 and unified memory is available");
    if (ordinal >= uint64_t(count)) fail("Metal device " + std::to_string(ordinal) + " is unavailable");
    check(larch_device_open(static_cast<int32_t>(ordinal), &device), nullptr, "larch_device_open");
    check(larch_stream_create(device, &stream), device, "larch_stream_create");
    check(larch_device_props_get(device, &props), device, "larch_device_props_get");
  }

  session(const session&) = delete;
  session& operator=(const session&) = delete;

  ~session() {
    if (stream) (void)larch_stream_sync(stream);
    for (larch_buffer* buffer : buffers) larch_buffer_free(buffer);
    if (stream) larch_stream_destroy(stream);
    if (device) larch_device_close(device);
  }

  larch_buffer* allocate(uint64_t bytes, const char* label) {
    larch_buffer* buffer = nullptr;
    check(larch_buffer_alloc(device, bytes, &buffer), device, label);
    buffers.push_back(buffer);
    return buffer;
  }

  void zero(larch_buffer* buffer, uint64_t bytes, const char* label) {
    const uint64_t chunk = std::min(bytes, kZeroChunkBytes);
    std::vector<uint8_t> zeros(static_cast<size_t>(chunk), 0);
    for (uint64_t offset = 0; offset < bytes;) {
      const uint64_t count = std::min(chunk, bytes - offset);
      check(larch_upload(stream, buffer, offset, zeros.data(), count), device, label);
      offset += count;
    }
  }

  void upload(larch_buffer* buffer, const void* data, uint64_t bytes, const char* label) {
    check(larch_upload(stream, buffer, 0, data, bytes), device, label);
  }

  void sync(const char* operation) { check(larch_stream_sync(stream), device, operation); }
};

struct event {
  larch_event* value = nullptr;
  ~event() {
    if (value) larch_event_destroy(value);
  }
};

template <class F>
uint64_t gpu_elapsed(session& run, F&& encode) {
  event begin;
  event end;
  check(larch_event_record(run.stream, &begin.value), run.device, "larch_event_record(start)");
  std::forward<F>(encode)();
  check(larch_event_record(run.stream, &end.value), run.device, "larch_event_record(end)");
  uint64_t elapsed_ns = 0;
  check(larch_event_elapsed_ns(begin.value, end.value, &elapsed_ns), run.device, "larch_event_elapsed_ns");
  if (!elapsed_ns) fail("Metal returned a zero GPU elapsed time");
  return elapsed_ns;
}

larch_tensor tensor(larch_buffer* buffer, larch_dtype dtype, uint32_t ndim, std::initializer_list<int64_t> shape,
                     std::initializer_list<int64_t> stride) {
  larch_tensor value = {};
  value.buf = buffer;
  value.dtype = dtype;
  value.ndim = ndim;
  size_t index = 0;
  for (int64_t dimension_value : shape) value.shape[index++] = dimension_value;
  index = 0;
  for (int64_t stride_value : stride) value.stride[index++] = stride_value;
  return value;
}

struct quant_format {
  larch_dtype dtype;
  uint64_t block_elements;
  uint64_t block_bytes;
};

quant_format format(const std::string& name) {
  if (name == "q3_k") return {LARCH_Q3_K, 256, 110};
  if (name == "q4_k") return {LARCH_Q4_K, 256, 144};
  if (name == "q5_k") return {LARCH_Q5_K, 256, 176};
  if (name == "q6_k") return {LARCH_Q6_K, 256, 210};
  if (name == "q8_0") return {LARCH_Q8_0, 32, 34};
  if (name == "q4_0") return {LARCH_Q4_0, 32, 18};
  if (name == "q5_0") return {LARCH_Q5_0, 32, 22};
  fail("--format must be one of: q3_k q4_k q5_k q6_k q8_0 q4_0 q5_0");
}

double bytes_per_ns(uint64_t bytes, uint64_t elapsed_ns) {
  return static_cast<double>(bytes) / static_cast<double>(elapsed_ns);
}

NSArray<id<MTLDevice>>* direct_devices() {
  NSMutableArray<id<MTLDevice>>* devices = [NSMutableArray array];
  for (id<MTLDevice> device in MTLCopyAllDevices()) {
    if ([device supportsFamily:MTLGPUFamilyMetal4] && device.hasUnifiedMemory) [devices addObject:device];
  }
  return devices;
}

id<MTLDevice> direct_device(uint64_t ordinal) {
  NSArray<id<MTLDevice>>* devices = direct_devices();
  if (ordinal >= devices.count) fail("Metal device " + std::to_string(ordinal) + " is unavailable");
  return devices[NSUInteger(ordinal)];
}

void wait_for(id<MTLCommandBuffer> command_buffer, const char* operation) {
  [command_buffer waitUntilCompleted];
  if (command_buffer.status == MTLCommandBufferStatusError) {
    const char* detail = command_buffer.error.localizedDescription.UTF8String;
    fail(std::string(operation) + " failed" + (detail ? ": " + std::string(detail) : ""));
  }
}

id<MTLComputePipelineState> metal_pipeline(id<MTLDevice> device, const std::string& name) {
  static id<MTLLibrary> library = nil;  // loaded once a device: st builds 45 pipelines from it
  NSError* error = nil;
  if (library.device != device) {
    dispatch_data_t data = dispatch_data_create(larch_metallib, larch_metallib_len, nullptr, ^{});
    library = [device newLibraryWithData:data error:&error];
    if (!library) {
      const char* detail = error.localizedDescription.UTF8String;
      fail(std::string("larch.metallib could not load") + (detail ? ": " + std::string(detail) : ""));
    }
  }
  id<MTLFunction> function = [library newFunctionWithName:@(name.c_str())];
  if (!function) fail("larch.metallib has no compute kernel " + name);
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) {
    const char* detail = error.localizedDescription.UTF8String;
    fail(name + " pipeline failed" + (detail ? ": " + std::string(detail) : ""));
  }
  return pipeline;
}

MTLSize copy_threads(id<MTLComputePipelineState> pipeline, id<MTLDevice> device) {
  const NSUInteger width = pipeline.threadExecutionWidth;
  const NSUInteger limit = std::min(pipeline.maxTotalThreadsPerThreadgroup, device.maxThreadsPerThreadgroup.width);
  if (!width || limit < width) fail("vv_perf_copy has no usable SIMD-width threadgroup");
  return MTLSizeMake(limit / width * width, 1, 1);
}

void encode_copy(id<MTLCommandBuffer> command_buffer, id<MTLComputePipelineState> pipeline, id<MTLBuffer> source,
                 id<MTLBuffer> destination, uint64_t bytes, uint64_t iterations) {
  const uint64_t vectors = (bytes + kCopyVectorBytes - 1) / kCopyVectorBytes;
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:source offset:0 atIndex:0];
  [encoder setBuffer:destination offset:0 atIndex:1];
  [encoder setBytes:&bytes length:sizeof bytes atIndex:2];
  const MTLSize grid = MTLSizeMake(NSUInteger(vectors), 1, 1);
  const MTLSize threads = copy_threads(pipeline, command_buffer.device);
  for (uint64_t iteration = 0; iteration < iterations; ++iteration)
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
  [encoder endEncoding];
}

void verify_copy(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, id<MTLDevice> device) {
  constexpr uint64_t bytes = kCopyVectorBytes + 3;
  id<MTLBuffer> source = [device newBufferWithLength:NSUInteger(bytes) options:MTLResourceStorageModeShared];
  id<MTLBuffer> destination = [device newBufferWithLength:NSUInteger(bytes) options:MTLResourceStorageModeShared];
  if (!source || !destination) fail("copy correctness buffers could not be allocated");
  auto* source_bytes = static_cast<uint8_t*>(source.contents);
  auto* destination_bytes = static_cast<uint8_t*>(destination.contents);
  for (uint64_t index = 0; index < bytes; ++index) {
    source_bytes[index] = static_cast<uint8_t>(index * 17 + 3);
    destination_bytes[index] = 0xA5;
  }
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  encode_copy(command_buffer, pipeline, source, destination, bytes, 1);
  [command_buffer commit];
  wait_for(command_buffer, "copy correctness check");
  for (uint64_t index = 0; index < bytes; ++index)
    if (destination_bytes[index] != source_bytes[index])
      fail("compute copy correctness check failed at byte " + std::to_string(index));
}

uint64_t measure_copy(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, id<MTLBuffer> source,
                      id<MTLBuffer> destination, uint64_t bytes, uint64_t iterations) {
  const uint64_t per_command_buffer = std::max<uint64_t>(1, kMaxCopyPayloadPerCommandBuffer / bytes);
  uint64_t remaining = iterations, elapsed_ns = 0;
  while (remaining) {
    const uint64_t batch = std::min(remaining, per_command_buffer);
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    encode_copy(command_buffer, pipeline, source, destination, bytes, batch);
    [command_buffer commit];
    wait_for(command_buffer, "copy measurement");
    const double seconds = command_buffer.GPUEndTime - command_buffer.GPUStartTime;
    if (!(seconds > 0.0)) fail("Metal returned no GPU timestamps for copy");
    const uint64_t batch_ns = static_cast<uint64_t>(seconds * 1e9);
    if (!batch_ns) fail("Metal returned a zero GPU elapsed time for copy");
    elapsed_ns = sum(elapsed_ns, batch_ns, "copy GPU elapsed");
    remaining -= batch;
  }
  return elapsed_ns;
}

void run_copy(const arguments& parsed) {
  require_only(parsed, {"--device", "--bytes", "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const uint64_t bytes = bounded(parsed, "--bytes", 1, kMaxCopyBytes);
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  id<MTLDevice> device = direct_device(ordinal);
  if (bytes > device.maxBufferLength) fail("--bytes exceeds this device's maxBufferLength");
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (!queue) fail("newCommandQueue failed");
  id<MTLComputePipelineState> pipeline = metal_pipeline(device, "vv_perf_copy");
  verify_copy(queue, pipeline, device);
  id<MTLBuffer> source = [device newBufferWithLength:NSUInteger(bytes) options:MTLResourceStorageModePrivate];
  id<MTLBuffer> destination = [device newBufferWithLength:NSUInteger(bytes) options:MTLResourceStorageModePrivate];
  if (!source || !destination) fail("copy buffers could not be allocated");

  id<MTLCommandBuffer> initialize = [queue commandBuffer];
  id<MTLBlitCommandEncoder> initialize_blit = [initialize blitCommandEncoder];
  [initialize_blit fillBuffer:source range:NSMakeRange(0, NSUInteger(bytes)) value:0];
  [initialize_blit fillBuffer:destination range:NSMakeRange(0, NSUInteger(bytes)) value:0];
  [initialize_blit endEncoding];
  [initialize commit];
  wait_for(initialize, "copy initialization");

  id<MTLCommandBuffer> warmup = [queue commandBuffer];
  encode_copy(warmup, pipeline, source, destination, bytes, 1);
  [warmup commit];
  wait_for(warmup, "copy warmup");

  const uint64_t elapsed_ns = measure_copy(queue, pipeline, source, destination, bytes, iterations);
  const uint64_t payload_bytes = product(bytes, iterations, "copy payload");
  std::printf("mode=copy device=%llu bytes=%llu iterations=%llu gpu_ns=%llu payload_gbps=%.6f read_write_gbps=%.6f\n",
              static_cast<unsigned long long>(ordinal), static_cast<unsigned long long>(bytes),
              static_cast<unsigned long long>(iterations), static_cast<unsigned long long>(elapsed_ns),
              bytes_per_ns(payload_bytes, elapsed_ns), bytes_per_ns(product(payload_bytes, 2, "copy traffic"), elapsed_ns));
}

// The prefill path's dequant of one weight to f16: quant bytes read plus f16 bytes written, per second.
void run_dequant(const arguments& parsed) {
  require_only(parsed, {"--device", "--format", "--n", "--k", "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const quant_format quant = format(required(parsed, "--format"));
  const uint64_t n = bounded(parsed, "--n", 1, kMaxDimension);
  const uint64_t k = bounded(parsed, "--k", 1, kMaxDimension);
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  if (k % quant.block_elements) fail("--k must be a whole number of the selected quantization blocks");
  const uint64_t weight_bytes = product(product(n, k / quant.block_elements, "dequant blocks"), quant.block_bytes,
                                        "dequant weight bytes");
  const uint64_t out_bytes = product(product(n, k, "dequant elements"), 2, "dequant output bytes");
  session run(ordinal);
  larch_buffer* weights = run.allocate(weight_bytes, "dequant weights");
  larch_buffer* out = run.allocate(out_bytes, "dequant output");
  run.zero(weights, weight_bytes, "dequant weight upload");
  run.sync("dequant setup");
  const larch_tensor weight = tensor(weights, quant.dtype, 2, {dimension(n, "--n"), dimension(k, "--k")},
                                      {dimension(k / quant.block_elements, "--k"), 1});
  const larch_tensor output = tensor(out, LARCH_F16, 2, {dimension(n, "--n"), dimension(k, "--k")},
                                      {dimension(k, "--k"), 1});
  check(larch_dequant(run.stream, &weight, &output), run.device, "dequant warmup");
  run.sync("dequant warmup");
  const uint64_t elapsed_ns = gpu_elapsed(run, [&] {
    for (uint64_t iteration = 0; iteration < iterations; ++iteration)
      check(larch_dequant(run.stream, &weight, &output), run.device, "larch_dequant");
  });
  const uint64_t moved = product(sum(weight_bytes, out_bytes, "dequant bytes"), iterations, "dequant traffic");
  std::printf("mode=dequant device=%llu format=%s n=%llu k=%llu iterations=%llu gpu_ns=%llu traffic_gbps=%.6f\n",
              static_cast<unsigned long long>(ordinal), required(parsed, "--format").c_str(),
              static_cast<unsigned long long>(n), static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(iterations), static_cast<unsigned long long>(elapsed_ns),
              bytes_per_ns(moved, elapsed_ns));
}

void run_mmvq(const arguments& parsed) {
  require_only(parsed, {"--device", "--format", "--rows", "--n", "--k", "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const quant_format quant = format(required(parsed, "--format"));
  const uint64_t rows = bounded(parsed, "--rows", 1, 16);
  const uint64_t n = bounded(parsed, "--n", 1, kMaxDimension);
  const uint64_t k = bounded(parsed, "--k", 1, kMaxDimension);
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  if (k % quant.block_elements) fail("--k must be a whole number of the selected quantization blocks");

  const uint64_t weight_blocks = product(n, k / quant.block_elements, "mmvq weight blocks");
  const uint64_t weight_bytes = product(weight_blocks, quant.block_bytes, "mmvq weight bytes");
  const uint64_t x_bytes = product(product(rows, k, "mmvq x elements"), 2, "mmvq x bytes");
  const uint64_t out_bytes = product(product(rows, n, "mmvq output elements"), 2, "mmvq output bytes");
  session run(ordinal);
  larch_buffer* weights = run.allocate(weight_bytes, "mmvq weights");
  larch_buffer* x = run.allocate(x_bytes, "mmvq x");
  larch_buffer* out = run.allocate(out_bytes, "mmvq output");
  run.zero(weights, weight_bytes, "mmvq weight upload");
  run.zero(x, x_bytes, "mmvq x upload");
  run.sync("mmvq setup");

  const larch_tensor weight = tensor(weights, quant.dtype, 2, {dimension(n, "--n"), dimension(k, "--k")},
                                      {dimension(k / quant.block_elements, "--k"), 1});
  const larch_tensor activation = tensor(x, LARCH_F16, 2, {dimension(rows, "--rows"), dimension(k, "--k")},
                                          {dimension(k, "--k"), 1});
  const larch_tensor output = tensor(out, LARCH_F16, 2, {dimension(rows, "--rows"), dimension(n, "--n")},
                                      {dimension(n, "--n"), 1});
  check(larch_mmvq(run.stream, &weight, &activation, nullptr, &output), run.device, "mmvq warmup");
  run.sync("mmvq warmup");
  const uint64_t elapsed_ns = gpu_elapsed(run, [&] {
    for (uint64_t iteration = 0; iteration < iterations; ++iteration)
      check(larch_mmvq(run.stream, &weight, &activation, nullptr, &output), run.device, "larch_mmvq");
  });
  const uint64_t logical_weight_bytes = product(weight_bytes, iterations, "mmvq logical weight bytes");
  std::printf("mode=mmvq device=%llu format=%s rows=%llu n=%llu k=%llu iterations=%llu gpu_ns=%llu "
              "weight_bytes=%llu logical_weight_gbps=%.6f\n",
              static_cast<unsigned long long>(ordinal), required(parsed, "--format").c_str(),
              static_cast<unsigned long long>(rows), static_cast<unsigned long long>(n), static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(iterations), static_cast<unsigned long long>(elapsed_ns),
              static_cast<unsigned long long>(weight_bytes), bytes_per_ns(logical_weight_bytes, elapsed_ns));
}

// The staged tile's steps and walks, by config: k32 .. k128 spread from a tile's own unit, k32ls .. lockstep.
constexpr uint32_t kStSteps[] = {32, 64, 128};
constexpr uint64_t kStRows[] = {1, 2, 4, 8, 16, 32};
constexpr uint64_t kStMaxRows = 32;
constexpr const char* kStFormats[] = {"q3_k", "q4_k", "q5_k", "q6_k", "q8_0"};

// metal_perf st on one format: the staged decode tile (staged.metal, f32 out) on one weight shape against
// larch_mmvq at 1 row (what plain decode runs today) on the same weights, both cache-warm. For each config (step
// 32, 64 or 128; spread or lockstep walk) on 32 random x rows: every row of a 2-, 4-, 8-, 16- and 32-row call must
// have the bytes of that row alone (with st_splits' K split for the shape), and every row alone must match
// larch_mmvq to 1e-2. Then each config is timed at each row count. One line per row count, with design 6.1's rule
// on rows 1, 2, 4 and 8 (at most 1.01, 1.04, 1.10 and 1.18 times mmvq at 1 row; 1.10 at 8 rows for a vocab-wide
// lm_head, n of 65536 or more), and a summary of the configs that pass at all four. False when a check failed.
bool run_st_format(const arguments& parsed, const std::string& name) {
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const quant_format qf = format(name);
  const uint64_t n = bounded(parsed, "--n", 1, kMaxDimension);
  const uint64_t k = bounded(parsed, "--k", ST_UNIT, kMaxDimension);
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  if (k % ST_UNIT) fail("--k must be a multiple of 128 (the staged tile's unit)");
  id<MTLDevice> device = direct_device(ordinal);
  session run(ordinal);
  const uint32_t cores = run.props.compute_units;
  if (!cores) fail("the device reports no GPU core count, which st_splits needs");
  const uint32_t splits = st_splits(uint32_t(n), uint32_t(k), cores);
  const uint64_t tiles = (n + ST_N - 1) / ST_N, row_bytes = k / qf.block_elements * qf.block_bytes;

  rng g(35);
  const bytes w = quant(qf.dtype, int64_t(n), int64_t(k), g);
  const bytes x = halves(kStMaxRows * k, g);

  // The reference: larch_mmvq at 1 row on each x row, then its time on row 0.
  larch_buffer* w_vv = run.allocate(w.size(), "st mmvq weights");
  larch_buffer* x_vv = run.allocate(2 * k, "st mmvq x");
  larch_buffer* out_vv = run.allocate(n * 4, "st mmvq output");
  run.upload(w_vv, w.data(), w.size(), "st mmvq weight upload");
  const int64_t nn = dimension(n, "--n"), kk = dimension(k, "--k");
  const larch_tensor weight = tensor(w_vv, qf.dtype, 2, {nn, kk}, {kk / int64_t(qf.block_elements), 1});
  const larch_tensor activation = tensor(x_vv, LARCH_F16, 2, {1, kk}, {kk, 1});
  const larch_tensor output = tensor(out_vv, LARCH_F32, 2, {1, nn}, {nn, 1});
  std::vector<std::vector<float>> ref(kStMaxRows, std::vector<float>(size_t(n)));
  for (uint64_t r = kStMaxRows; r-- > 0;) {  // row 0 last, so it stays in x_vv for the timing
    run.upload(x_vv, x.data() + r * 2 * k, 2 * k, "st mmvq x upload");
    check(larch_mmvq(run.stream, &weight, &activation, nullptr, &output), run.device, "st mmvq reference");
    check(larch_download(run.stream, out_vv, 0, ref[r].data(), n * 4), run.device, "st mmvq download");
    run.sync("st mmvq reference");
  }
  const uint64_t mmvq_ns = gpu_elapsed(run, [&] {
    for (uint64_t i = 0; i < iterations; ++i)
      check(larch_mmvq(run.stream, &weight, &activation, nullptr, &output), run.device, "larch_mmvq");
  });
  const double mmvq_us = double(mmvq_ns) / 1e3 / double(iterations);

  // A step whose pipelines do not build is reported and skipped, so the other steps still run; the format fails.
  id<MTLComputePipelineState> pipes[3][3];  // [step][8-row tiles: 1, 2, 4]
  bool built[3] = {true, true, true};
  for (int s = 0; s < 3; ++s) {
    try {
      for (int t = 0; t < 3; ++t)
        pipes[s][t] = metal_pipeline(device, "vv_mmvq_st" + std::to_string(kStSteps[s]) + "_m" +
                                                 std::to_string(8 << t) + "_" + name);
    } catch (const std::exception& error) {
      built[s] = false;
      std::printf("mode=st-missing format=%s step=%u reason=%s\n", name.c_str(), kStSteps[s], error.what());
    }
  }
  id<MTLCommandQueue> queue = [device newCommandQueue];
  constexpr uint64_t check_rows = kStMaxRows + 2 + 4 + 8 + 16 + 32;  // each row alone, then each multi-row call
  id<MTLBuffer> w_buf = [device newBufferWithBytes:w.data() length:w.size() options:MTLResourceStorageModeShared];
  id<MTLBuffer> x_buf = [device newBufferWithBytes:x.data() length:x.size() options:MTLResourceStorageModeShared];
  id<MTLBuffer> out_buf = [device newBufferWithLength:NSUInteger(check_rows * n * 4)
                                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> parts = [device newBufferWithLength:NSUInteger(splits * kStMaxRows * n * 4)
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> counters = [device newBufferWithLength:NSUInteger(tiles * 4) options:MTLResourceStorageModeShared];
  if (!queue || !w_buf || !x_buf || !out_buf || !parts || !counters) fail("st buffers could not be allocated");
  std::memset(counters.contents, 0, NSUInteger(tiles * 4));  // the tile keeps them zero between calls
  st_args args = {};
  args.mm.a_rs = k * 2;
  args.mm.b_rs = row_bytes;
  args.mm.c_rs = n * 4;
  args.mm.n = uint32_t(n);
  args.mm.k = uint32_t(k);
  args.mm.c_f32 = 1;
  args.mm.alpha = 1.0f;
  args.splits = splits;
  const auto encode = [&](id<MTLComputeCommandEncoder> encoder, int step, bool spread, uint64_t rows, uint64_t x_row,
                          uint64_t out_row) {
    args.mm.m = uint32_t(rows);
    args.spread = spread;
    [encoder setComputePipelineState:pipes[step][rows <= 8 ? 0 : rows <= 16 ? 1 : 2]];
    [encoder setBuffer:w_buf offset:0 atIndex:0];
    [encoder setBuffer:x_buf offset:NSUInteger(x_row * k * 2) atIndex:1];
    [encoder setBuffer:w_buf offset:0 atIndex:2];  // no bias (has_bias 0): any bound buffer
    [encoder setBuffer:out_buf offset:NSUInteger(out_row * n * 4) atIndex:3];
    [encoder setBuffer:parts offset:0 atIndex:4];
    [encoder setBuffer:counters offset:0 atIndex:5];
    [encoder setBytes:&args length:sizeof args atIndex:6];  // last, as backend.mm's encode binds it
    [encoder dispatchThreadgroups:MTLSizeMake(NSUInteger(tiles), NSUInteger(splits), 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];  // one simdgroup (staged.metal)
  };

  constexpr int configs = 6;  // step s, spread: config s; lockstep: config 3 + s
  const auto config_name = [](int c) { return "k" + std::to_string(kStSteps[c % 3]) + (c >= 3 ? "ls" : ""); };
  uint64_t unequal[std::size(kStRows)] = {};
  double max_rel = 0;
  double us[configs][std::size(kStRows)] = {};
  const auto* out = static_cast<const uint8_t*>(out_buf.contents);
  for (int c = 0; c < configs; ++c) {
    if (!built[c % 3]) {
      std::fill(std::begin(us[c]), std::end(us[c]), NAN);  // compares false: never best, never passing
      continue;
    }
    id<MTLCommandBuffer> checks = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [checks computeCommandEncoder];
    for (uint64_t r = 0; r < kStMaxRows; ++r) encode(encoder, c % 3, c < 3, 1, r, r);
    for (uint64_t ri = 1, at = kStMaxRows; ri < std::size(kStRows); at += kStRows[ri++])
      encode(encoder, c % 3, c < 3, kStRows[ri], 0, at);
    [encoder endEncoding];
    [checks commit];
    wait_for(checks, "st row checks");
    for (uint64_t ri = 1, at = kStMaxRows; ri < std::size(kStRows); at += kStRows[ri++])
      for (uint64_t r = 0; r < kStRows[ri]; ++r)
        unequal[ri] += std::memcmp(out + (at + r) * n * 4, out + r * n * 4, n * 4) != 0;
    for (uint64_t r = 0; r < kStMaxRows; ++r) {
      const auto* mine = reinterpret_cast<const float*>(out + r * n * 4);
      double peak = 0, err = 0;
      for (uint64_t col = 0; col < n; ++col) {
        peak = std::max(peak, double(std::fabs(ref[r][col])));
        err = std::max(err, double(std::fabs(mine[col] - ref[r][col])));
      }
      const double rel = peak > 0 ? err / peak : err;
      max_rel = std::isnan(rel) ? INFINITY : std::max(max_rel, rel);
    }
    for (size_t ri = 0; ri < std::size(kStRows); ++ri) {
      id<MTLCommandBuffer> warm = [queue commandBuffer];
      encoder = [warm computeCommandEncoder];
      encode(encoder, c % 3, c < 3, kStRows[ri], 0, 0);
      [encoder endEncoding];
      [warm commit];
      wait_for(warm, "st warmup");
      id<MTLCommandBuffer> timed = [queue commandBuffer];
      encoder = [timed computeCommandEncoder];
      for (uint64_t i = 0; i < iterations; ++i) encode(encoder, c % 3, c < 3, kStRows[ri], 0, 0);
      [encoder endEncoding];
      [timed commit];
      wait_for(timed, "st timing");
      us[c][ri] = (timed.GPUEndTime - timed.GPUStartTime) * 1e6 / double(iterations);
    }
  }

  const bool head = n >= 65536, values_ok = max_rel < 1e-2 && built[0] && built[1] && built[2];
  const double limits[] = {1.01, 1.04, 1.10, head ? 1.10 : 1.18};
  bool rows_ok = values_ok, passing[configs];
  std::fill(std::begin(passing), std::end(passing), true);
  for (size_t ri = 0; ri < std::size(kStRows); ++ri) {
    int best = 0;
    for (int c = 1; c < configs; ++c) best = std::isnan(us[best][ri]) || us[c][ri] < us[best][ri] ? c : best;
    const bool ruled = ri < std::size(limits);
    for (int c = 0; ruled && c < configs; ++c) passing[c] = passing[c] && us[c][ri] <= limits[ri] * mmvq_us;
    rows_ok = rows_ok && !unequal[ri];
    std::printf("mode=st format=%s n=%llu k=%llu rows=%llu cores=%u splits=%u mmvq_us=%.3f", name.c_str(),
                static_cast<unsigned long long>(n), static_cast<unsigned long long>(k),
                static_cast<unsigned long long>(kStRows[ri]), cores, splits, mmvq_us);
    for (int c = 0; c < configs; ++c) std::printf(" %s_us=%.3f", config_name(c).c_str(), us[c][ri]);
    const double ratio = us[best][ri] / mmvq_us;
    std::printf(" best=%s ratio=%.3f", config_name(best).c_str(), ratio);
    if (ruled)
      std::printf(" limit=%.2f speed=%s", limits[ri], ratio <= limits[ri] ? "pass" : "FAIL");
    else
      std::printf(" limit=none speed=none");
    std::printf(" unequal=%llu max_rel_vs_mmvq=%.2e rows_ok=%s\n", static_cast<unsigned long long>(unequal[ri]),
                max_rel, !unequal[ri] && values_ok ? "pass" : "FAIL");
  }
  std::string pass;
  for (int c = 0; c < configs; ++c)
    if (passing[c]) pass += (pass.empty() ? "" : ",") + config_name(c);
  std::printf("mode=st-summary format=%s n=%llu k=%llu passing=%s rows_ok=%s\n", name.c_str(),
              static_cast<unsigned long long>(n), static_cast<unsigned long long>(k),
              pass.empty() ? "none" : pass.c_str(), rows_ok ? "pass" : "FAIL");
  return rows_ok;
}

// metal_perf st: run_st_format on --format, or on each of the five with --format all. Exits 1 when a row check
// failed, 77 (ctest's skip) without an Apple GPU family 10 device: the tile needs the tensor API.
void run_st(const arguments& parsed) {
  require_only(parsed, {"--device", "--format", "--n", "--k", "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  NSArray<id<MTLDevice>>* devices = direct_devices();
  if (ordinal >= devices.count || ![devices[NSUInteger(ordinal)] supportsFamily:MTLGPUFamilyApple10]) {
    std::printf("mode=st skipped: no Apple GPU family 10 device %llu\n", static_cast<unsigned long long>(ordinal));
    std::exit(77);
  }
  const std::string& name = required(parsed, "--format");
  std::vector<std::string> names;
  if (name == "all") names.assign(std::begin(kStFormats), std::end(kStFormats));
  else if (std::find(std::begin(kStFormats), std::end(kStFormats), name) != std::end(kStFormats)) names = {name};
  else fail("--format must be one of: q3_k q4_k q5_k q6_k q8_0 all");
  bool ok = true;
  for (const std::string& each : names) ok = run_st_format(parsed, each) && ok;
  if (!ok) fail("a staged row differs from that row alone, or from larch_mmvq by 1e-2 or more (rows_ok=FAIL above)");
}

// An MoE layer's expert matmul at decode or a small prefill: row r, slot j picks expert (r * slots + j) % experts,
// so up to rows * slots experts are read. Weighted is the down projection: x per slot, out their weighted sum.
void run_mmvq_id(const arguments& parsed) {
  require_only(parsed, {"--device", "--format", "--experts", "--rows", "--slots", "--n", "--k", "--weighted",
                        "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const quant_format quant = format(required(parsed, "--format"));
  const uint64_t experts = bounded(parsed, "--experts", 1, 512);
  const uint64_t rows = bounded(parsed, "--rows", 1, 4096);
  const uint64_t slots = bounded(parsed, "--slots", 1, experts);
  const uint64_t n = bounded(parsed, "--n", 1, kMaxDimension);
  const uint64_t k = bounded(parsed, "--k", 1, kMaxDimension);
  const bool weighted = bounded(parsed, "--weighted", 0, 1) == 1;
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  if (k % quant.block_elements) fail("--k must be a whole number of the selected quantization blocks");

  const uint64_t expert_bytes = product(product(n, k / quant.block_elements, "expert blocks"), quant.block_bytes,
                                        "expert bytes");
  const uint64_t weight_bytes = product(expert_bytes, experts, "mmvq-id weight bytes");
  const uint64_t pairs = product(rows, slots, "mmvq-id pairs");
  const uint64_t x_bytes = product(product(weighted ? pairs : rows, k, "mmvq-id x elements"), 2, "mmvq-id x bytes");
  const uint64_t out_bytes = product(product(weighted ? rows : pairs, n, "mmvq-id output elements"), 2, "out bytes");
  std::vector<int32_t> ids(static_cast<size_t>(pairs));
  for (uint64_t p = 0; p < pairs; ++p) ids[size_t(p)] = int32_t(p % experts);
  const std::vector<float> weights(static_cast<size_t>(pairs), 1.0f / float(slots));
  session run(ordinal);
  larch_buffer* w_buffer = run.allocate(weight_bytes, "mmvq-id weights");
  larch_buffer* x = run.allocate(x_bytes, "mmvq-id x");
  larch_buffer* id_buffer = run.allocate(pairs * 4, "mmvq-id ids");
  larch_buffer* wt_buffer = run.allocate(pairs * 4, "mmvq-id combine weights");
  larch_buffer* out = run.allocate(out_bytes, "mmvq-id output");
  run.zero(w_buffer, weight_bytes, "mmvq-id weight upload");
  run.zero(x, x_bytes, "mmvq-id x upload");
  run.upload(id_buffer, ids.data(), pairs * 4, "mmvq-id ids upload");
  run.upload(wt_buffer, weights.data(), pairs * 4, "mmvq-id combine weights upload");
  run.sync("mmvq-id setup");

  const int64_t r = dimension(rows, "--rows"), m = dimension(slots, "--slots"), kk = dimension(k, "--k"),
                nn = dimension(n, "--n");
  const larch_tensor weight = tensor(w_buffer, quant.dtype, 2, {dimension(experts * n, "--experts"), kk},
                                      {dimension(k / quant.block_elements, "--k"), 1});
  const larch_tensor activation = weighted ? tensor(x, LARCH_F16, 3, {r, m, kk}, {m * kk, kk, 1})
                                            : tensor(x, LARCH_F16, 2, {r, kk}, {kk, 1});
  const larch_tensor chosen = tensor(id_buffer, LARCH_I32, 2, {r, m}, {m, 1});
  const larch_tensor combine = tensor(wt_buffer, LARCH_F32, 2, {r, m}, {m, 1});
  const larch_tensor output = weighted ? tensor(out, LARCH_F16, 2, {r, nn}, {nn, 1})
                                        : tensor(out, LARCH_F16, 3, {r, m, nn}, {m * nn, nn, 1});
  const larch_tensor* wt = weighted ? &combine : nullptr;
  check(larch_mmvq_id(run.stream, &weight, &activation, &chosen, nullptr, wt, &output), run.device,
        "mmvq_id warmup");
  run.sync("mmvq_id warmup");
  const uint64_t elapsed_ns = gpu_elapsed(run, [&] {
    for (uint64_t iteration = 0; iteration < iterations; ++iteration)
      check(larch_mmvq_id(run.stream, &weight, &activation, &chosen, nullptr, wt, &output), run.device,
            "larch_mmvq_id");
  });
  const uint64_t read = product(product(std::min(pairs, experts), expert_bytes, "read bytes"), iterations, "traffic");
  std::printf("mode=mmvq-id device=%llu format=%s experts=%llu rows=%llu slots=%llu n=%llu k=%llu weighted=%d "
              "iterations=%llu gpu_ns=%llu us_per_call=%.3f expert_gbps=%.6f\n",
              static_cast<unsigned long long>(ordinal), required(parsed, "--format").c_str(),
              static_cast<unsigned long long>(experts), static_cast<unsigned long long>(rows),
              static_cast<unsigned long long>(slots), static_cast<unsigned long long>(n),
              static_cast<unsigned long long>(k), int(weighted), static_cast<unsigned long long>(iterations),
              static_cast<unsigned long long>(elapsed_ns), double(elapsed_ns) / 1e3 / double(iterations),
              bytes_per_ns(read, elapsed_ns));
}

// Q3_K rows of nb blocks rearranged for vv_mmvq_id_rq3 (ops.metal), bits moved and none changed: weight i of
// block b (as ggml orders it: sub-block c = i / 32, n = c / 4, bit pair u = c % 4, l = i % 32) has its low bits
// at qs[32 n + l] >> 2 u and its high bit at hmask[l] bit 4 n + u. Rows land rs bytes apart (16-byte aligned).
bytes repack_q3_k(const bytes& w, uint64_t rows, uint64_t nb, uint64_t rs) {
  bytes out(size_t(rows * rs), 0);
  for (uint64_t r = 0; r < rows; ++r) {
    uint8_t* o = out.data() + r * rs;
    for (uint64_t b = 0; b < nb; ++b) {
      const uint8_t* q = w.data() + (r * nb + b) * 110;
      for (int i = 0; i < 256; ++i) {
        const int c = i / 32, n = c / 4, u = c % 4, l = i % 32;
        const int lo = (q[32 + 32 * n + l] >> (2 * u)) & 3, hi = (q[l] >> (4 * n + u)) & 1;
        o[b * 64 + i / 4] |= uint8_t(lo << (2 * (i % 4)));
        o[nb * 64 + b * 32 + i / 8] |= uint8_t(hi << (i % 8));
      }
      std::memcpy(o + nb * 96 + b * 12, q + 96, 12);
      std::memcpy(o + nb * 108 + b * 2, q + 108, 2);
    }
  }
  return out;
}

// metal_perf repack-id: run_mmvq_id's shapes and expert choice on Q3_K, larch_mmvq_id on the GGUF blocks against
// vv_mmvq_id_rq3 on the same blocks repacked, f32 out. Prints both times and the largest difference relative to
// the output's peak; exits 1 past 1e-3.
void run_repack_id(const arguments& parsed) {
  require_only(parsed, {"--device", "--experts", "--rows", "--slots", "--n", "--k", "--weighted", "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const uint64_t experts = bounded(parsed, "--experts", 1, 512);
  const uint64_t rows = bounded(parsed, "--rows", 1, 64);
  const uint64_t slots = bounded(parsed, "--slots", 1, experts);
  const uint64_t n = bounded(parsed, "--n", 1, kMaxDimension);
  const uint64_t k = bounded(parsed, "--k", 512, kMaxDimension);
  const bool weighted = bounded(parsed, "--weighted", 0, 1) == 1;
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  if (k % 512) fail("--k must be a multiple of 512");
  const uint64_t nb = k / 256, rs = (nb * 110 + 15) / 16 * 16, pairs = rows * slots;
  rng g(41);
  const bytes w = quant(LARCH_Q3_K, int64_t(experts * n), int64_t(k), g);
  const bytes wr = repack_q3_k(w, experts * n, nb, rs);
  const bytes x = halves(size_t((weighted ? pairs : rows) * k), g);
  std::vector<int32_t> ids(static_cast<size_t>(pairs));
  for (uint64_t p = 0; p < pairs; ++p) ids[size_t(p)] = int32_t(p % experts);
  std::vector<float> wts(static_cast<size_t>(pairs));
  for (auto& v : wts) v = std::uniform_real_distribution<float>(0, 1)(g);
  const uint64_t out_elems = (weighted ? rows : pairs) * n;

  session run(ordinal);
  larch_buffer* w_vv = run.allocate(w.size(), "repack-id weights");
  larch_buffer* x_vv = run.allocate(x.size(), "repack-id x");
  larch_buffer* id_vv = run.allocate(pairs * 4, "repack-id ids");
  larch_buffer* wt_vv = run.allocate(pairs * 4, "repack-id combine weights");
  larch_buffer* out_vv = run.allocate(out_elems * 4, "repack-id output");
  run.upload(w_vv, w.data(), w.size(), "repack-id weight upload");
  run.upload(x_vv, x.data(), x.size(), "repack-id x upload");
  run.upload(id_vv, ids.data(), pairs * 4, "repack-id ids upload");
  run.upload(wt_vv, wts.data(), pairs * 4, "repack-id combine weights upload");
  run.sync("repack-id setup");
  const int64_t r = int64_t(rows), m = int64_t(slots), kk = int64_t(k), nn = int64_t(n);
  const larch_tensor weight = tensor(w_vv, LARCH_Q3_K, 2, {int64_t(experts) * nn, kk}, {int64_t(nb), 1});
  const larch_tensor activation = weighted ? tensor(x_vv, LARCH_F16, 3, {r, m, kk}, {m * kk, kk, 1})
                                            : tensor(x_vv, LARCH_F16, 2, {r, kk}, {kk, 1});
  const larch_tensor chosen = tensor(id_vv, LARCH_I32, 2, {r, m}, {m, 1});
  const larch_tensor combine = tensor(wt_vv, LARCH_F32, 2, {r, m}, {m, 1});
  const larch_tensor output = weighted ? tensor(out_vv, LARCH_F32, 2, {r, nn}, {nn, 1})
                                        : tensor(out_vv, LARCH_F32, 3, {r, m, nn}, {m * nn, nn, 1});
  const larch_tensor* wt = weighted ? &combine : nullptr;
  check(larch_mmvq_id(run.stream, &weight, &activation, &chosen, nullptr, wt, &output), run.device, "repack-id ref");
  std::vector<float> ref(static_cast<size_t>(out_elems));
  check(larch_download(run.stream, out_vv, 0, ref.data(), out_elems * 4), run.device, "repack-id ref download");
  run.sync("repack-id reference");
  const uint64_t ref_ns = gpu_elapsed(run, [&] {
    for (uint64_t i = 0; i < iterations; ++i)
      check(larch_mmvq_id(run.stream, &weight, &activation, &chosen, nullptr, wt, &output), run.device,
            "larch_mmvq_id");
  });
  run.sync("repack-id reference timing");

  id<MTLDevice> device = direct_device(ordinal);
  id<MTLComputePipelineState> pipe = metal_pipeline(device, "vv_mmvq_id_rq3");
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLBuffer> w_buf = [device newBufferWithBytes:wr.data() length:wr.size() options:MTLResourceStorageModeShared];
  id<MTLBuffer> x_buf = [device newBufferWithBytes:x.data() length:x.size() options:MTLResourceStorageModeShared];
  id<MTLBuffer> id_buf = [device newBufferWithBytes:ids.data() length:pairs * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> wt_buf = [device newBufferWithBytes:wts.data() length:pairs * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> out_buf = [device newBufferWithLength:out_elems * 4 options:MTLResourceStorageModeShared];
  if (!queue || !w_buf || !x_buf || !id_buf || !wt_buf || !out_buf) fail("repack-id buffers could not be allocated");
  id_args a = {};
  a.x_rs = (weighted ? slots : 1) * k * 2;
  a.x_slot_rs = weighted ? k * 2 : 0;
  a.w_rs = rs;
  a.ids_rs = slots * 4;
  a.wt_rs = weighted ? slots * 4 : 0;
  a.out_rs = (weighted ? 1 : slots) * n * 4;
  a.out_slot_rs = weighted ? 0 : n * 4;
  a.n = uint32_t(n);
  a.k = uint32_t(k);
  a.slots = uint32_t(slots);
  a.out_f32 = 1;
  a.weighted = weighted ? 1 : 0;
  a.experts = uint32_t(experts);
  a.rows = uint32_t(rows);
  const auto encode = [&](id<MTLComputeCommandEncoder> e) {
    [e setComputePipelineState:pipe];
    [e setBuffer:w_buf offset:0 atIndex:0];
    [e setBuffer:x_buf offset:0 atIndex:1];
    [e setBuffer:id_buf offset:0 atIndex:2];
    [e setBuffer:w_buf offset:0 atIndex:3];  // no bias
    [e setBuffer:wt_buf offset:0 atIndex:4];
    [e setBuffer:out_buf offset:0 atIndex:5];
    [e setBytes:&a length:sizeof a atIndex:6];
    [e dispatchThreadgroups:MTLSizeMake(NSUInteger((n + 7) / 8), NSUInteger(weighted ? rows : pairs), 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  };
  {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    encode(e);
    [e endEncoding];
    [cb commit];
    wait_for(cb, "repack-id check");
  }
  const auto* mine = static_cast<const float*>(out_buf.contents);
  double peak = 0, err = 0;
  for (uint64_t i = 0; i < out_elems; ++i) {
    peak = std::max(peak, double(std::fabs(ref[size_t(i)])));
    err = std::max(err, double(std::fabs(mine[i] - ref[size_t(i)])));
  }
  const double rel = peak > 0 ? err / peak : err;
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];  // serial: each call waits for the one before
  for (uint64_t i = 0; i < iterations; ++i) encode(e);
  [e endEncoding];
  [cb commit];
  wait_for(cb, "repack-id timing");
  const double rq_ns = (cb.GPUEndTime - cb.GPUStartTime) * 1e9;
  const uint64_t read = std::min(pairs, experts) * n * nb * 110;
  std::printf("mode=repack-id experts=%llu rows=%llu slots=%llu n=%llu k=%llu weighted=%d iterations=%llu "
              "ref_us=%.3f repacked_us=%.3f ref_gbps=%.3f repacked_gbps=%.3f max_rel=%.3g %s\n",
              static_cast<unsigned long long>(experts), static_cast<unsigned long long>(rows),
              static_cast<unsigned long long>(slots), static_cast<unsigned long long>(n),
              static_cast<unsigned long long>(k), int(weighted), static_cast<unsigned long long>(iterations),
              double(ref_ns) / 1e3 / double(iterations), rq_ns / 1e3 / double(iterations),
              double(read) * double(iterations) / double(ref_ns), double(read) * double(iterations) / rq_ns, rel,
              rel <= 1e-3 ? "pass" : "FAIL");
  if (rel > 1e-3) fail("vv_mmvq_id_rq3 differs from larch_mmvq_id by more than 1e-3 of the output's peak");
}

// ggml's dequantize_row_q3_K for one 110-byte block, written out on its own here (not from repack_aq3's indexing)
// so the affine prototype's weights are checked against Q3_K's: 256 f32 weights in ggml's order.
float half_to_float(uint16_t h) {  // exact: arm64's native fp16
  _Float16 v;
  std::memcpy(&v, &h, 2);
  return float(v);
}

void q3_k_block_ref(const uint8_t* blk, float* y) {
  uint16_t dh;
  std::memcpy(&dh, blk + 108, 2);
  const float d_all = half_to_float(dh);
  const uint8_t *hm = blk, *q = blk + 32;
  uint32_t aux[4];
  std::memcpy(aux, blk + 96, 12);
  const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f, tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  int8_t scales[16];
  std::memcpy(scales, aux, 16);
  int is = 0;
  uint8_t m = 1;
  for (int n = 0; n < 256; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      float dl = d_all * float(scales[is++] - 32);
      for (int l = 0; l < 16; ++l) *y++ = dl * float(int8_t((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
      dl = d_all * float(scales[is++] - 32);
      for (int l = 0; l < 16; ++l) *y++ = dl * float(int8_t((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
      shift += 2;
      m <<= 1;
    }
    q += 32;
  }
}

// Q3_K rows of nb blocks in vv_mmvq_aq3's layout (ops.metal): codes q + 4, 8 in 3 bytes; int8 sc - 32 per 16
// weights; fp16 d per block. Rows rs bytes apart (16-byte aligned). Weight i of block b: sub-block c = i / 32,
// n = c / 4, bit pair u = c % 4, l = i % 32; scale index 8 n + 2 u + l / 16, which is i / 16.
bytes repack_aq3(const bytes& w, uint64_t rows, uint64_t nb, uint64_t rs) {
  bytes out(size_t(rows * rs), 0);
  const uint64_t k = nb * 256, code_bytes = k * 3 / 8, groups = k / 16;
  for (uint64_t r = 0; r < rows; ++r) {
    uint8_t* o = out.data() + r * rs;
    for (uint64_t b = 0; b < nb; ++b) {
      const uint8_t* q = w.data() + (r * nb + b) * 110;
      uint32_t aux[3];
      std::memcpy(aux, q + 96, 12);
      for (int i = 0; i < 256; ++i) {
        const int c = i / 32, n = c / 4, u = c % 4, l = i % 32;
        const int lo = (q[32 + 32 * n + l] >> (2 * u)) & 3, hi = (q[l] >> (4 * n + u)) & 1;
        const uint64_t e = b * 256 + i, bit = 3 * e;
        const uint32_t code = uint32_t(lo | hi << 2);  // q + 4
        o[bit / 8] |= uint8_t(code << (bit % 8));
        if (bit % 8 > 5) o[bit / 8 + 1] |= uint8_t(code >> (8 - bit % 8));
      }
      for (int si = 0; si < 16; ++si) {  // ggml's get_scale: low nibble from bytes 0-7, top 2 bits from 8-11
        const int lo = si < 8 ? (q[96 + si] & 0xF) : (q[96 + si - 8] >> 4);
        const int hi = (q[104 + si % 4] >> (2 * (si / 4))) & 3;
        o[code_bytes + b * 16 + si] = uint8_t(int8_t((lo | hi << 4) - 32));
      }
      std::memcpy(o + code_bytes + groups + b * 2, q + 108, 2);
      (void)aux;
    }
  }
  return out;
}

// metal_perf affine-q3: larch_mmvq on Q3_K blocks (today's kernels) against vv_mmvq_aq3 on the same weights in the
// exact affine layout, f32 out. First checks every repacked weight equals ggml's Q3_K value bit for bit, then prints
// both times, the bytes per weight and the largest output difference relative to the output's peak; exits 1 on a
// weight mismatch or past 1e-3.
void run_affine_q3(const arguments& parsed) {
  require_only(parsed, {"--device", "--rows", "--n", "--k", "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const uint64_t rows = bounded(parsed, "--rows", 1, 64);
  const uint64_t n = bounded(parsed, "--n", 1, kMaxDimension);
  const uint64_t k = bounded(parsed, "--k", 512, kMaxDimension);
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  if (k % 512) fail("--k must be a multiple of 512");
  const uint64_t nb = k / 256, code_bytes = k * 3 / 8, groups = k / 16;
  const uint64_t rs = (code_bytes + groups + nb * 2 + 15) / 16 * 16;
  rng g(43);
  const bytes w = quant(LARCH_Q3_K, int64_t(n), int64_t(k), g);
  const bytes wr = repack_aq3(w, n, nb, rs);
  uint64_t mismatched = 0;
  for (uint64_t r = 0; r < n; ++r) {
    const uint8_t* o = wr.data() + r * rs;
    for (uint64_t b = 0; b < nb; ++b) {
      float ref[256];
      q3_k_block_ref(w.data() + (r * nb + b) * 110, ref);
      uint16_t dh;
      std::memcpy(&dh, o + code_bytes + groups + b * 2, 2);
      for (int i = 0; i < 256; ++i) {
        const uint64_t e = b * 256 + i, bit = 3 * e;
        const uint32_t word = uint32_t(o[bit / 8]) | (bit / 8 + 1 < code_bytes ? uint32_t(o[bit / 8 + 1]) << 8 : 0);
        const int code = int(word >> (bit % 8) & 7);
        const float s = half_to_float(dh) * float(int8_t(o[code_bytes + e / 16]));
        const float mine = s * float(code - 4);
        if (std::memcmp(&mine, &ref[i], 4)) ++mismatched;
      }
    }
  }
  const bytes x = halves(size_t(rows * k), g);
  session run(ordinal);
  larch_buffer* w_vv = run.allocate(w.size(), "affine-q3 weights");
  larch_buffer* x_vv = run.allocate(x.size(), "affine-q3 x");
  larch_buffer* out_vv = run.allocate(rows * n * 4, "affine-q3 output");
  run.upload(w_vv, w.data(), w.size(), "affine-q3 weight upload");
  run.upload(x_vv, x.data(), x.size(), "affine-q3 x upload");
  run.sync("affine-q3 setup");
  const int64_t rr = int64_t(rows), kk = int64_t(k), nn = int64_t(n);
  const larch_tensor weight = tensor(w_vv, LARCH_Q3_K, 2, {nn, kk}, {int64_t(nb), 1});
  const larch_tensor activation = tensor(x_vv, LARCH_F16, 2, {rr, kk}, {kk, 1});
  const larch_tensor output = tensor(out_vv, LARCH_F32, 2, {rr, nn}, {nn, 1});
  check(larch_mmvq(run.stream, &weight, &activation, nullptr, &output), run.device, "affine-q3 ref");
  std::vector<float> ref(static_cast<size_t>(rows * n));
  check(larch_download(run.stream, out_vv, 0, ref.data(), rows * n * 4), run.device, "affine-q3 ref download");
  run.sync("affine-q3 reference");
  const uint64_t ref_ns = gpu_elapsed(run, [&] {
    for (uint64_t i = 0; i < iterations; ++i)
      check(larch_mmvq(run.stream, &weight, &activation, nullptr, &output), run.device, "larch_mmvq");
  });
  run.sync("affine-q3 reference timing");

  id<MTLDevice> device = direct_device(ordinal);
  id<MTLComputePipelineState> pipe = metal_pipeline(device, "vv_mmvq_aq3");
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLBuffer> w_buf = [device newBufferWithBytes:wr.data() length:wr.size() options:MTLResourceStorageModeShared];
  id<MTLBuffer> x_buf = [device newBufferWithBytes:x.data() length:x.size() options:MTLResourceStorageModeShared];
  id<MTLBuffer> out_buf = [device newBufferWithLength:rows * n * 4 options:MTLResourceStorageModeShared];
  if (!queue || !w_buf || !x_buf || !out_buf) fail("affine-q3 buffers could not be allocated");
  mm_args a = {};
  a.a_rs = k * 2;
  a.b_rs = rs;
  a.c_rs = n * 4;
  a.m = uint32_t(rows);
  a.n = uint32_t(n);
  a.k = uint32_t(k);
  a.c_f32 = 1;
  const auto encode = [&](id<MTLComputeCommandEncoder> e) {
    [e setComputePipelineState:pipe];
    [e setBuffer:w_buf offset:0 atIndex:0];
    [e setBuffer:x_buf offset:0 atIndex:1];
    [e setBuffer:out_buf offset:0 atIndex:2];
    [e setBytes:&a length:sizeof a atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(NSUInteger(rows), NSUInteger((n + 7) / 8), 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  };
  {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    encode(e);
    [e endEncoding];
    [cb commit];
    wait_for(cb, "affine-q3 check");
  }
  const auto* mine = static_cast<const float*>(out_buf.contents);
  double peak = 0, err = 0;
  for (uint64_t i = 0; i < rows * n; ++i) {
    peak = std::max(peak, double(std::fabs(ref[size_t(i)])));
    err = std::max(err, double(std::fabs(mine[i] - ref[size_t(i)])));
  }
  const double rel = peak > 0 ? err / peak : err;
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];  // serial: each call waits for the one before
  for (uint64_t i = 0; i < iterations; ++i) encode(e);
  [e endEncoding];
  [cb commit];
  wait_for(cb, "affine-q3 timing");
  const double aq_ns = (cb.GPUEndTime - cb.GPUStartTime) * 1e9;
  std::printf("mode=affine-q3 rows=%llu n=%llu k=%llu iterations=%llu ref_us=%.3f affine_us=%.3f "
              "q3k_bits_per_weight=%.4f affine_bits_per_weight=%.4f weight_mismatches=%llu max_rel=%.3g %s\n",
              static_cast<unsigned long long>(rows), static_cast<unsigned long long>(n),
              static_cast<unsigned long long>(k), static_cast<unsigned long long>(iterations),
              double(ref_ns) / 1e3 / double(iterations), aq_ns / 1e3 / double(iterations), 110.0 * 8 / 256,
              double(code_bytes + groups + nb * 2) * 8 / double(k), static_cast<unsigned long long>(mismatched), rel,
              !mismatched && rel <= 1e-3 ? "pass" : "FAIL");
  if (mismatched) fail("affine-q3 weights differ from ggml's Q3_K values");
  if (rel > 1e-3) fail("vv_mmvq_aq3 differs from larch_mmvq by more than 1e-3 of the output's peak");
}

// A chain of dependent one-row adds, x += h: at a small d, the cost of one dispatch in a serial encoder, which
// a decode pays for each of its few-microsecond ops.
void run_add(const arguments& parsed) {
  require_only(parsed, {"--device", "--d", "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const uint64_t d = bounded(parsed, "--d", 1, kMaxDimension);
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  session run(ordinal);
  larch_buffer* x = run.allocate(d * 4, "add x");
  larch_buffer* h = run.allocate(d * 2, "add h");
  run.zero(x, d * 4, "add x upload");
  run.zero(h, d * 2, "add h upload");
  run.sync("add setup");
  const int64_t dd = dimension(d, "--d");
  const larch_tensor xt = tensor(x, LARCH_F32, 2, {1, dd}, {dd, 1});
  const larch_tensor ht = tensor(h, LARCH_F16, 2, {1, dd}, {dd, 1});
  check(larch_add(run.stream, &xt, &ht, &xt), run.device, "add warmup");
  run.sync("add warmup");
  const uint64_t elapsed_ns = gpu_elapsed(run, [&] {
    for (uint64_t iteration = 0; iteration < iterations; ++iteration)
      check(larch_add(run.stream, &xt, &ht, &xt), run.device, "larch_add");
  });
  std::printf("mode=add device=%llu d=%llu iterations=%llu gpu_ns=%llu us_per_dispatch=%.3f\n",
              static_cast<unsigned long long>(ordinal), static_cast<unsigned long long>(d),
              static_cast<unsigned long long>(iterations), static_cast<unsigned long long>(elapsed_ns),
              double(elapsed_ns) / 1e3 / double(iterations));
}

void run_gemm(const arguments& parsed) {
  require_only(parsed, {"--device", "--m", "--n", "--k", "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const uint64_t m = bounded(parsed, "--m", 1, kMaxDimension);
  const uint64_t n = bounded(parsed, "--n", 1, kMaxDimension);
  const uint64_t k = bounded(parsed, "--k", 1, kMaxDimension);
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  const uint64_t a_bytes = product(product(m, k, "gemm A elements"), 2, "gemm A bytes");
  const uint64_t b_bytes = product(product(n, k, "gemm B elements"), 2, "gemm B bytes");
  const uint64_t c_bytes = product(product(m, n, "gemm C elements"), 2, "gemm C bytes");

  session run(ordinal);
  larch_buffer* a_buffer = run.allocate(a_bytes, "gemm A");
  larch_buffer* b_buffer = run.allocate(b_bytes, "gemm B");
  larch_buffer* c_buffer = run.allocate(c_bytes, "gemm C");
  run.zero(a_buffer, a_bytes, "gemm A upload");
  run.zero(b_buffer, b_bytes, "gemm B upload");
  run.zero(c_buffer, c_bytes, "gemm C upload");
  run.sync("gemm setup");

  const larch_tensor a = tensor(a_buffer, LARCH_F16, 2, {dimension(m, "--m"), dimension(k, "--k")},
                                 {dimension(k, "--k"), 1});
  const larch_tensor b = tensor(b_buffer, LARCH_F16, 2, {dimension(n, "--n"), dimension(k, "--k")},
                                 {dimension(k, "--k"), 1});
  const larch_tensor c = tensor(c_buffer, LARCH_F16, 2, {dimension(m, "--m"), dimension(n, "--n")},
                                 {dimension(n, "--n"), 1});
  check(larch_gemm(run.stream, &a, &b, nullptr, &c, 1.0f, 0.0f), run.device, "gemm warmup");
  run.sync("gemm warmup");
  const uint64_t elapsed_ns = gpu_elapsed(run, [&] {
    for (uint64_t iteration = 0; iteration < iterations; ++iteration)
      check(larch_gemm(run.stream, &a, &b, nullptr, &c, 1.0f, 0.0f), run.device, "larch_gemm");
  });
  const uint64_t operations = product(product(product(product(m, n, "gemm operations"), k, "gemm operations"),
                                              2, "gemm operations"), iterations, "gemm operations");
  const uint64_t logical_bytes_per_iteration =
      sum(sum(a_bytes, b_bytes, "gemm logical bytes"), c_bytes, "gemm logical bytes");
  const uint64_t logical_bytes = product(logical_bytes_per_iteration, iterations, "gemm logical bytes");
  const double logical_tflops = static_cast<double>(operations) / static_cast<double>(elapsed_ns) / 1000.0;
  std::printf("mode=gemm device=%llu m=%llu n=%llu k=%llu iterations=%llu gpu_ns=%llu logical_tflops=%.6f "
              "logical_read_write_gbps=%.6f\n",
              static_cast<unsigned long long>(ordinal), static_cast<unsigned long long>(m),
              static_cast<unsigned long long>(n), static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(iterations), static_cast<unsigned long long>(elapsed_ns), logical_tflops,
              bytes_per_ns(logical_bytes, elapsed_ns));
}

// The model's linear layer above mmvq's rows: dequant w into an f16 scratch, then gemm on it.
void run_linear(const arguments& parsed) {
  require_only(parsed, {"--device", "--format", "--m", "--n", "--k", "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const quant_format quant = format(required(parsed, "--format"));
  const uint64_t m = bounded(parsed, "--m", 1, kMaxDimension);
  const uint64_t n = bounded(parsed, "--n", 1, kMaxDimension);
  const uint64_t k = bounded(parsed, "--k", 1, kMaxDimension);
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  if (k % quant.block_elements) fail("--k must be a whole number of the selected quantization blocks");
  const uint64_t w_bytes = product(product(n, k / quant.block_elements, "linear blocks"), quant.block_bytes,
                                   "linear weight bytes");
  const uint64_t dq_bytes = product(product(n, k, "linear scratch elements"), 2, "linear scratch bytes");
  const uint64_t a_bytes = product(product(m, k, "linear x elements"), 2, "linear x bytes");
  const uint64_t c_bytes = product(product(m, n, "linear out elements"), 2, "linear out bytes");
  session run(ordinal);
  larch_buffer* w_buffer = run.allocate(w_bytes, "linear weights");
  larch_buffer* dq_buffer = run.allocate(dq_bytes, "linear scratch");
  larch_buffer* a_buffer = run.allocate(a_bytes, "linear x");
  larch_buffer* c_buffer = run.allocate(c_bytes, "linear out");
  run.zero(w_buffer, w_bytes, "linear weight upload");
  run.zero(a_buffer, a_bytes, "linear x upload");
  run.sync("linear setup");
  const int64_t mi = dimension(m, "--m"), ni = dimension(n, "--n"), ki = dimension(k, "--k");
  const larch_tensor w = tensor(w_buffer, quant.dtype, 2, {ni, ki}, {dimension(k / quant.block_elements, "--k"), 1});
  const larch_tensor dq = tensor(dq_buffer, LARCH_F16, 2, {ni, ki}, {ki, 1});
  const larch_tensor a = tensor(a_buffer, LARCH_F16, 2, {mi, ki}, {ki, 1});
  const larch_tensor c = tensor(c_buffer, LARCH_F16, 2, {mi, ni}, {ni, 1});
  const auto step = [&] {
    check(larch_dequant(run.stream, &w, &dq), run.device, "larch_dequant");
    check(larch_gemm(run.stream, &a, &dq, nullptr, &c, 1.0f, 0.0f), run.device, "larch_gemm");
  };
  step();
  run.sync("linear warmup");
  const uint64_t elapsed_ns = gpu_elapsed(run, [&] {
    for (uint64_t iteration = 0; iteration < iterations; ++iteration) step();
  });
  const double flops = 2.0 * double(m) * double(n) * double(k) * double(iterations);
  std::printf("mode=linear device=%llu format=%s m=%llu n=%llu k=%llu iterations=%llu gpu_ns=%llu "
              "logical_tflops=%.6f\n",
              static_cast<unsigned long long>(ordinal), required(parsed, "--format").c_str(),
              static_cast<unsigned long long>(m), static_cast<unsigned long long>(n), static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(iterations), static_cast<unsigned long long>(elapsed_ns),
              flops / double(elapsed_ns) / 1000.0);
}

void run_attention(const arguments& parsed) {
  require_only(parsed, {"--device", "--rows", "--seq-len", "--heads", "--kv-heads", "--head-dim", "--splits",
                        "--scale", "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const uint64_t rows = bounded(parsed, "--rows", 1, 16);
  const uint64_t sequence_length = bounded(parsed, "--seq-len", 1, 65536);
  const uint64_t heads = bounded(parsed, "--heads", 1, 128);
  const uint64_t kv_heads = bounded(parsed, "--kv-heads", 1, heads);
  const uint64_t head_dim = bounded(parsed, "--head-dim", 1, 256);
  const uint64_t splits = bounded(parsed, "--splits", 0, 16);
  const float scale = parse_scale(parsed);
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  if (heads % kv_heads) fail("--heads must be divisible by --kv-heads");

  constexpr uint64_t block_tokens = 16;
  const uint64_t blocks = (sequence_length + block_tokens - 1) / block_tokens;
  const uint64_t q_elements = product(product(rows, heads, "attention q rows"), head_dim, "attention q elements");
  const uint64_t q_bytes = product(q_elements, 2, "attention q bytes");
  const uint64_t pool_elements =
      product(product(product(rows, blocks, "attention pool blocks") * block_tokens, kv_heads, "attention pool slots"),
              head_dim, "attention pool elements");
  const uint64_t pool_bytes = product(pool_elements, 2, "attention pool bytes");
  const uint64_t table_elements = product(rows, blocks, "attention block table elements");
  const uint64_t table_bytes = product(table_elements, sizeof(int32_t), "attention block table bytes");
  const uint64_t lens_bytes = product(rows, sizeof(int32_t), "attention sequence length bytes");

  session run(ordinal);
  larch_buffer* q_buffer = run.allocate(q_bytes, "attention q");
  larch_buffer* out_buffer = run.allocate(q_bytes, "attention output");
  larch_buffer* key_buffer = run.allocate(pool_bytes, "attention key pool");
  larch_buffer* value_buffer = run.allocate(pool_bytes, "attention value pool");
  larch_buffer* table_buffer = run.allocate(table_bytes, "attention block table");
  larch_buffer* lens_buffer = run.allocate(lens_bytes, "attention sequence lengths");
  run.zero(q_buffer, q_bytes, "attention q upload");
  run.zero(key_buffer, pool_bytes, "attention key upload");
  run.zero(value_buffer, pool_bytes, "attention value upload");
  std::vector<int32_t> table(static_cast<size_t>(table_elements));
  for (uint64_t row = 0; row < rows; ++row)
    for (uint64_t block = 0; block < blocks; ++block)
      table[static_cast<size_t>(row * blocks + block)] = static_cast<int32_t>(row * blocks + block);  // a row's own
  std::vector<int32_t> lengths(static_cast<size_t>(rows), static_cast<int32_t>(sequence_length));
  run.upload(table_buffer, table.data(), table_bytes, "attention block table upload");
  run.upload(lens_buffer, lengths.data(), lens_bytes, "attention sequence length upload");
  run.sync("attention setup");

  const larch_tensor q = tensor(q_buffer, LARCH_F16, 3,
                                 {dimension(rows, "--rows"), dimension(heads, "--heads"), dimension(head_dim, "--head-dim")},
                                 {dimension(heads * head_dim, "--heads"), dimension(head_dim, "--head-dim"), 1});
  const larch_tensor out = tensor(out_buffer, LARCH_F16, 3,
                                   {dimension(rows, "--rows"), dimension(heads, "--heads"), dimension(head_dim, "--head-dim")},
                                   {dimension(heads * head_dim, "--heads"), dimension(head_dim, "--head-dim"), 1});
  const larch_tensor block_table = tensor(table_buffer, LARCH_I32, 2,
                                            {dimension(rows, "--rows"), dimension(blocks, "attention blocks")},
                                            {dimension(blocks, "attention blocks"), 1});
  const larch_tensor seq_lens = tensor(lens_buffer, LARCH_I32, 1, {dimension(rows, "--rows")}, {1});
  const larch_kv_pool pool = {key_buffer, value_buffer, static_cast<uint32_t>(rows * blocks), static_cast<uint32_t>(block_tokens),
                               1, static_cast<uint32_t>(kv_heads), static_cast<uint32_t>(head_dim), LARCH_F16, 0};
  check(larch_attn_decode(run.stream, &q, &pool, &block_table, &seq_lens, static_cast<uint32_t>(splits), scale, &out),
        run.device, "attn-decode warmup");
  run.sync("attn-decode warmup");
  const uint64_t elapsed_ns = gpu_elapsed(run, [&] {
    for (uint64_t iteration = 0; iteration < iterations; ++iteration)
      check(larch_attn_decode(run.stream, &q, &pool, &block_table, &seq_lens, static_cast<uint32_t>(splits), scale, &out),
            run.device, "larch_attn_decode");
  });
  const uint64_t kv_bytes_per_iteration =
      product(product(product(product(rows, heads, "attention rows and heads"), sequence_length, "attention positions"), head_dim,
                      "attention dimensions"),
              4, "attention key/value bytes");
  const uint64_t logical_kv_bytes = product(kv_bytes_per_iteration, iterations, "attention logical key/value bytes");
  const uint64_t decoded_rows = product(rows, iterations, "attention decoded rows");
  const double rows_per_second = static_cast<double>(decoded_rows) * 1e9 / static_cast<double>(elapsed_ns);
  std::printf("mode=attn-decode device=%llu rows=%llu seq_len=%llu heads=%llu kv_heads=%llu head_dim=%llu splits=%llu "
              "iterations=%llu gpu_ns=%llu logical_kv_gbps=%.6f decoded_rows_per_s=%.6f\n",
              static_cast<unsigned long long>(ordinal), static_cast<unsigned long long>(rows),
              static_cast<unsigned long long>(sequence_length), static_cast<unsigned long long>(heads),
              static_cast<unsigned long long>(kv_heads), static_cast<unsigned long long>(head_dim),
              static_cast<unsigned long long>(splits), static_cast<unsigned long long>(iterations),
              static_cast<unsigned long long>(elapsed_ns), bytes_per_ns(logical_kv_bytes, elapsed_ns), rows_per_second);
}

// One prefill chunk of --rows tokens, the last of a --seq-len sequence (causal: row j sees positions up to
// seq_len - rows + j), the shape a chunked prompt's attention has.
void run_attention_prefill(const arguments& parsed) {
  require_only(parsed, {"--device", "--rows", "--seq-len", "--heads", "--kv-heads", "--head-dim", "--scale",
                        "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const uint64_t sequence_length = bounded(parsed, "--seq-len", 1, 65536);
  const uint64_t rows = bounded(parsed, "--rows", 1, sequence_length);
  const uint64_t heads = bounded(parsed, "--heads", 1, 128);
  const uint64_t kv_heads = bounded(parsed, "--kv-heads", 1, heads);
  const uint64_t head_dim = bounded(parsed, "--head-dim", 1, 256);
  const float scale = parse_scale(parsed);
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  if (heads % kv_heads) fail("--heads must be divisible by --kv-heads");
  constexpr uint64_t block_tokens = 16;
  const uint64_t blocks = (sequence_length + block_tokens - 1) / block_tokens;
  const uint64_t q_bytes = rows * heads * head_dim * 2, pool_bytes = blocks * block_tokens * kv_heads * head_dim * 2;
  session run(ordinal);
  larch_buffer* q_buffer = run.allocate(q_bytes, "prefill q");
  larch_buffer* out_buffer = run.allocate(q_bytes, "prefill output");
  larch_buffer* key_buffer = run.allocate(pool_bytes, "prefill key pool");
  larch_buffer* value_buffer = run.allocate(pool_bytes, "prefill value pool");
  larch_buffer* table_buffer = run.allocate(blocks * sizeof(int32_t), "prefill block table");
  larch_buffer* meta_buffer = run.allocate(2 * sizeof(int32_t), "prefill start and length");
  run.zero(q_buffer, q_bytes, "prefill q upload");
  run.zero(key_buffer, pool_bytes, "prefill key upload");
  run.zero(value_buffer, pool_bytes, "prefill value upload");
  std::vector<int32_t> table(static_cast<size_t>(blocks));
  for (uint64_t block = 0; block < blocks; ++block) table[static_cast<size_t>(block)] = static_cast<int32_t>(block);
  const int32_t meta[2] = {static_cast<int32_t>(sequence_length - rows), static_cast<int32_t>(rows)};
  run.upload(table_buffer, table.data(), blocks * sizeof(int32_t), "prefill block table upload");
  run.upload(meta_buffer, meta, sizeof meta, "prefill start and length upload");
  run.sync("prefill setup");
  const int64_t r = dimension(rows, "--rows"), h = dimension(heads, "--heads"), d = dimension(head_dim, "--head-dim");
  const larch_tensor q = tensor(q_buffer, LARCH_F16, 3, {r, h, d}, {h * d, d, 1});
  const larch_tensor out = tensor(out_buffer, LARCH_F16, 3, {r, h, d}, {h * d, d, 1});
  const larch_tensor block_table = tensor(table_buffer, LARCH_I32, 2, {1, dimension(blocks, "blocks")},
                                           {dimension(blocks, "blocks"), 1});
  larch_tensor start = tensor(meta_buffer, LARCH_I32, 1, {1}, {1}), lengths = start;
  lengths.offset = sizeof(int32_t);
  const larch_kv_pool pool = {key_buffer, value_buffer, static_cast<uint32_t>(blocks), static_cast<uint32_t>(block_tokens),
                               1, static_cast<uint32_t>(kv_heads), static_cast<uint32_t>(head_dim), LARCH_F16, 0};
  check(larch_attn_prefill(run.stream, &q, &pool, &block_table, &start, &lengths, scale, &out), run.device,
        "attn-prefill warmup");
  run.sync("attn-prefill warmup");
  const uint64_t elapsed_ns = gpu_elapsed(run, [&] {
    for (uint64_t iteration = 0; iteration < iterations; ++iteration)
      check(larch_attn_prefill(run.stream, &q, &pool, &block_table, &start, &lengths, scale, &out), run.device,
            "larch_attn_prefill");
  });
  std::printf("mode=attn-prefill device=%llu rows=%llu seq_len=%llu heads=%llu kv_heads=%llu head_dim=%llu "
              "iterations=%llu gpu_ns=%llu\n",
              static_cast<unsigned long long>(ordinal), static_cast<unsigned long long>(rows),
              static_cast<unsigned long long>(sequence_length), static_cast<unsigned long long>(heads),
              static_cast<unsigned long long>(kv_heads), static_cast<unsigned long long>(head_dim),
              static_cast<unsigned long long>(iterations), static_cast<unsigned long long>(elapsed_ns));
}

// sample_prep of `rows` decode rows (no penalty) over random logits.
void run_sample(const arguments& parsed) {
  require_only(parsed, {"--device", "--rows", "--vocab", "--top-k", "--iterations"});
  const uint64_t ordinal = bounded(parsed, "--device", 0, std::numeric_limits<int32_t>::max());
  const uint64_t rows = bounded(parsed, "--rows", 1, 4096);
  const uint64_t vocab = bounded(parsed, "--vocab", 1, kMaxDimension);
  const uint64_t top_k = bounded(parsed, "--top-k", 0, 64);
  const uint64_t iterations = bounded(parsed, "--iterations", 1, kMaxIterations);
  const uint64_t elements = product(rows, vocab, "sample logits");
  std::vector<float> host(elements);
  std::mt19937 random(1);
  for (float& x : host) x = std::uniform_real_distribution<float>(-8, 8)(random);
  session run(ordinal);
  larch_buffer* logits = run.allocate(elements * 4, "sample logits");
  larch_buffer* out = run.allocate(rows * (64 * 8 + 4), "sample outputs");
  check(larch_upload(run.stream, logits, 0, host.data(), elements * 4), run.device, "sample logits upload");
  run.sync("sample setup");
  const int64_t r = dimension(rows, "--rows");
  const larch_tensor lg = tensor(logits, LARCH_F32, 2, {r, dimension(vocab, "--vocab")}, {dimension(vocab, "--vocab"), 1});
  larch_tensor vals = tensor(out, LARCH_F32, 2, {r, 64}, {64, 1}), ids = vals, lse = tensor(out, LARCH_F32, 1, {r}, {1});
  ids.dtype = LARCH_I32;
  ids.offset = rows * 256;
  lse.offset = rows * 512;
  const std::vector<larch_sample_params> per_row(rows, larch_sample_params{1, uint32_t(top_k), 1, 0});
  const auto once = [&] {
    check(larch_sample_prep(run.stream, &lg, per_row.data(), nullptr, &vals, &ids, &lse), run.device, "larch_sample_prep");
  };
  once();
  run.sync("sample warmup");
  const uint64_t elapsed_ns = gpu_elapsed(run, [&] {
    for (uint64_t iteration = 0; iteration < iterations; ++iteration) once();
  });
  std::printf("mode=sample device=%llu rows=%llu vocab=%llu iterations=%llu gpu_ns=%llu\n",
              static_cast<unsigned long long>(ordinal), static_cast<unsigned long long>(rows),
              static_cast<unsigned long long>(vocab), static_cast<unsigned long long>(iterations),
              static_cast<unsigned long long>(elapsed_ns));
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) == "--help") {
          usage();
          return 0;
        }
      }
      const arguments parsed = parse_arguments(argc, argv);
      if (parsed.mode == "copy") run_copy(parsed);
      else if (parsed.mode == "mmvq") run_mmvq(parsed);
      else if (parsed.mode == "st") run_st(parsed);
      else if (parsed.mode == "mmvq-id") run_mmvq_id(parsed);
      else if (parsed.mode == "repack-id") run_repack_id(parsed);
      else if (parsed.mode == "affine-q3") run_affine_q3(parsed);
      else if (parsed.mode == "add") run_add(parsed);
      else if (parsed.mode == "dequant") run_dequant(parsed);
      else if (parsed.mode == "attn-prefill") run_attention_prefill(parsed);
      else if (parsed.mode == "gemm") run_gemm(parsed);
      else if (parsed.mode == "linear") run_linear(parsed);
      else if (parsed.mode == "sample") run_sample(parsed);
      else run_attention(parsed);
      return 0;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "metal_perf: %s\n", error.what());
      return 1;
    }
  }
}
