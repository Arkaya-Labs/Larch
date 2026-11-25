// Step-1 smoke test of the Metal plumbing, through the C ABI only: device, props, a round trip
// through a private buffer in two staging chunks timed by two events, and a zero-copy wrap of a file
// mapping.
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <vector>

#include "larch.h"

#define CHECK(x)                                                                                     \
  do {                                                                                               \
    if (!(x)) {                                                                                      \
      std::fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, #x, larch_last_error(dev)); \
      return 1;                                                                                      \
    }                                                                                                \
  } while (0)

static uint64_t now_ns() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

int main() {
  larch_device* dev = nullptr;
  int32_t n = 0;
  CHECK(larch_device_count(&n) == LARCH_OK && n > 0);
  CHECK(larch_device_open(0, &dev) == LARCH_OK);

  larch_device_props p;
  CHECK(larch_device_props_get(dev, &p) == LARCH_OK);
  std::printf("%s: Apple family %u, %u GPU cores, SIMD width %u, working set %" PRIu64 " MiB\n", p.name, p.arch_code,
              p.compute_units, p.warp_size, p.recommended_working_set >> 20);
  CHECK(p.backend == LARCH_BACKEND_METAL && p.compute_units > 0 && p.warp_size > 0 && p.max_threads_per_group > 0 &&
        p.recommended_working_set == p.vram_total && p.vram_free <= p.vram_total);

  larch_stream* s = nullptr;
  CHECK(larch_stream_create(dev, &s) == LARCH_OK);

  // Round trip through a private buffer at an odd offset and length, uploaded in two staging chunks.
  // The events around the upload count GPU time, so they fit inside the host's wall time around it.
  const uint64_t len = (96 << 20) + 3, off = 4096 + 4;
  std::vector<uint8_t> src(len), dst(len);
  for (uint64_t i = 0; i < len; ++i) src[i] = uint8_t(i * 31 + 7);
  larch_buffer *buf = nullptr, *huge = nullptr;
  larch_event *first = nullptr, *e0 = nullptr, *e1 = nullptr;
  CHECK(larch_buffer_alloc(dev, 1ull << 62, &huge) == LARCH_ERR_OOM);  // past maxBufferLength
  CHECK(larch_buffer_alloc(dev, off + len, &buf) == LARCH_OK);
  CHECK(larch_upload(s, buf, off, src.data(), len + 1) == LARCH_ERR_INVALID_ARG);
  CHECK(larch_event_record(s, &first) == LARCH_OK);  // so e0 has a command buffer before it
  const uint64_t t0 = now_ns();
  CHECK(larch_event_record(s, &e0) == LARCH_OK);
  CHECK(larch_upload(s, buf, off, src.data(), len) == LARCH_OK);
  CHECK(larch_event_record(s, &e1) == LARCH_OK);
  CHECK(larch_download(s, buf, off, dst.data(), len) == LARCH_OK);
  const uint64_t wall = now_ns() - t0;
  CHECK(src == dst);
  uint64_t ns = 0;
  CHECK(larch_event_elapsed_ns(e0, e1, &ns) == LARCH_OK && ns > 0 && ns <= wall);
  CHECK(larch_event_elapsed_ns(e1, e0, &ns) == LARCH_ERR_INVALID_ARG);
  CHECK(larch_event_elapsed_ns(e1, e1, &ns) == LARCH_OK && ns == 0);
  // Events on two streams do not compare.
  larch_stream* s2 = nullptr;
  larch_event *f0 = nullptr, *f1 = nullptr;
  CHECK(larch_stream_create(dev, &s2) == LARCH_OK && larch_event_record(s2, &f0) == LARCH_OK &&
        larch_event_record(s2, &f1) == LARCH_OK && larch_stream_sync(s2) == LARCH_OK);
  CHECK(larch_event_elapsed_ns(e0, f1, &ns) == LARCH_ERR_INVALID_ARG);
  larch_event_destroy(f0);
  larch_event_destroy(f1);
  larch_stream_destroy(s2);

  // Uploads share the stream's one staging buffer, so 8 back-to-back 16 MiB uploads with no sync
  // between them allocate at most that buffer, not 128 MiB of staging.
  CHECK(larch_device_props_get(dev, &p) == LARCH_OK);
  const int64_t free0 = int64_t(p.vram_free);
  for (int i = 0; i < 8; ++i) CHECK(larch_upload(s, buf, 0, src.data(), 16 << 20) == LARCH_OK);
  CHECK(larch_device_props_get(dev, &p) == LARCH_OK && free0 - int64_t(p.vram_free) <= (64 << 20));
  CHECK(larch_stream_sync(s) == LARCH_OK);

  // A null buffer, or one from the CPU reference, is INVALID_ARG on a Metal stream.
  larch_device* cpu = nullptr;
  larch_buffer* cpu_buf = nullptr;
  CHECK(larch_device_open(-1, &cpu) == LARCH_OK && larch_buffer_alloc(cpu, 4, &cpu_buf) == LARCH_OK);
  CHECK(larch_upload(s, cpu_buf, 0, src.data(), 4) == LARCH_ERR_INVALID_ARG);
  CHECK(larch_download(s, nullptr, 0, dst.data(), 4) == LARCH_ERR_INVALID_ARG);
  larch_buffer_free(cpu_buf);
  larch_device_close(cpu);
  std::printf("upload of %" PRIu64 " MiB: %" PRIu64 " us of GPU time, %" PRIu64 " us of wall time\n", len >> 20,
              ns / 1000, wall / 1000);

  // Zero copy: wrap a read-only mapping whose length is not a page multiple, change the file
  // through its descriptor, and read the change back through the GPU buffer at an unaligned offset.
  const long page = getpagesize();
  const uint64_t flen = uint64_t(3 * page + 123), at = uint64_t(2 * page + 36);
  FILE* f = std::tmpfile();
  CHECK(f && ftruncate(fileno(f), off_t(flen)) == 0);
  void* map = mmap(nullptr, flen, PROT_READ, MAP_SHARED, fileno(f), 0);
  CHECK(map != MAP_FAILED);
  larch_buffer *wrapped = nullptr, *unaligned = nullptr;
  CHECK(larch_buffer_wrap_host(dev, static_cast<char*>(map) + 64, flen - 64, &unaligned) == LARCH_ERR_INVALID_ARG);
  CHECK(larch_device_props_get(dev, &p) == LARCH_OK);
  const uint64_t free_before = p.vram_free;
  CHECK(larch_buffer_wrap_host(dev, map, flen, &wrapped) == LARCH_OK);
  CHECK(larch_device_props_get(dev, &p) == LARCH_OK && p.vram_free < free_before);  // wrapped bytes count
  const uint32_t marker = 0xdeadbeef;
  uint32_t got = 0;
  CHECK(larch_download(s, wrapped, flen - 2, &got, sizeof got) == LARCH_ERR_INVALID_ARG);  // past flen, same page
  CHECK(larch_upload(s, wrapped, at, &marker, sizeof marker) == LARCH_ERR_INVALID_ARG);   // read-only
  CHECK(larch_buffer_wrap_host(dev, map, ~0ull - 100, &unaligned) == LARCH_ERR_OOM);       // past maxBufferLength
  CHECK(pwrite(fileno(f), &marker, sizeof marker, off_t(at)) == sizeof marker);
  CHECK(larch_download(s, wrapped, at, &got, sizeof got) == LARCH_OK);
  CHECK(got == marker);

  larch_buffer_free(wrapped);
  munmap(map, flen);
  std::fclose(f);
  larch_event_destroy(first);
  larch_event_destroy(e0);
  larch_event_destroy(e1);
  larch_buffer_free(buf);
  larch_stream_destroy(s);
  larch_device_close(dev);
  std::printf("metal smoke: ok\n");
  return 0;
}
