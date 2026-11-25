// The debug-only device-loss op (architecture 9.11), in its own process: a sticky error poisons all of
// it. On the CPU reference the op is UNSUPPORTED. On the device under test (LARCH_TEST_DEVICE, else
// GPU 0 when the build has one, else nothing more): on CUDA the op is INVALID_ARG while the stream
// captures (the stream stays healthy), then loses the device for real, so the sync after it, the op
// itself, the next op (an upload into a buffer from before the fault) and that op's sync are all
// DEVICE_LOST; on Metal and HIP the slot is unset, so the op is UNSUPPORTED. Nothing is reset in-process
// (D2): after the fault the process ends with _Exit, since the runtime's exit teardown of a lost context
// can hang (#73).
#define _CRT_SECURE_NO_WARNINGS  // std::getenv on MSVC
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "larch.h"

// dispatch.cpp's entry, outside larch.h: its callers declare it (larch-kernels does the same).
extern "C" larch_status vv_fault_device_lost(larch_stream*);

static int wrong = 0;

// One line per check: `what` returned `st`, which is `ok` or not.
static void expect(const char* what, larch_status st, bool ok) {
  std::printf("%s: %s: status %d%s%s\n", ok ? "ok" : "WRONG", what, int(st), st ? ", " : "",
              st ? larch_last_error(nullptr) : "");
  wrong += !ok;
}

// LARCH_TEST_DEVICE, or GPU 0 when there is one, else -1 (the CPU reference check alone).
static bool pick(int32_t gpus, int32_t* ordinal) {
  const char* e = std::getenv("LARCH_TEST_DEVICE");
  if (!e || !*e) return *ordinal = gpus > 0 ? 0 : -1, true;
  char* end = nullptr;
  const long v = std::strtol(e, &end, 10);
  if (*end || v < -1 || v >= gpus) {
    std::fprintf(stderr, "LARCH_TEST_DEVICE=%s: want -1 or a GPU ordinal below %d\n", e, gpus);
    return false;
  }
  return *ordinal = int32_t(v), true;
}

int main() {
  larch_device* cpu = nullptr;
  larch_stream* cs = nullptr;
  if (larch_device_open(-1, &cpu) != LARCH_OK || larch_stream_create(cpu, &cs) != LARCH_OK) {
    std::fprintf(stderr, "CPU reference: open failed: %s\n", larch_last_error(nullptr));
    return 1;
  }
  larch_status st = vv_fault_device_lost(cs);
  expect("CPU reference: fault_device_lost is UNSUPPORTED", st, st == LARCH_ERR_UNSUPPORTED);
  larch_stream_destroy(cs);
  larch_device_close(cpu);

  int32_t gpus = 0, ordinal = -1;
  larch_backend backend = LARCH_BACKEND_CPU;
  uint32_t rt = 0, drv = 0;
  if (larch_device_count(&gpus) != LARCH_OK || larch_backend_info(&backend, &rt, &drv) != LARCH_OK) {
    std::fprintf(stderr, "device count or backend info failed: %s\n", larch_last_error(nullptr));
    return 1;
  }
  if (!pick(gpus, &ordinal)) return 1;
  if (ordinal < 0) {
    std::printf("device_lost: no GPU under test, the CPU reference check only\n");
    return wrong != 0;
  }
  larch_device* dev = nullptr;
  larch_stream* s = nullptr;
  larch_buffer* buf = nullptr;
  if (larch_device_open(ordinal, &dev) != LARCH_OK || larch_stream_create(dev, &s) != LARCH_OK ||
      larch_buffer_alloc(dev, 64, &buf) != LARCH_OK) {
    std::fprintf(stderr, "device %d: open failed: %s\n", ordinal, larch_last_error(nullptr));
    return 1;
  }
  std::printf("device_lost: device %d\n", ordinal);
  if (backend != LARCH_BACKEND_CUDA) {
    st = vv_fault_device_lost(s);
    expect("GPU (not CUDA): fault_device_lost is UNSUPPORTED", st, st == LARCH_ERR_UNSUPPORTED);
    larch_buffer_free(buf);
    larch_stream_destroy(s);
    larch_device_close(dev);
    return wrong != 0;
  }
  // A captured fault would replay: refused, and the (empty) capture ends normally.
  larch_graph* g = nullptr;
  st = larch_graph_begin(s);
  expect("GPU (CUDA): graph_begin is OK", st, st == LARCH_OK);
  st = vv_fault_device_lost(s);
  expect("GPU (CUDA): fault_device_lost while the stream captures is INVALID_ARG", st, st == LARCH_ERR_INVALID_ARG);
  st = larch_graph_end(s, &g);
  expect("GPU (CUDA): graph_end after the refusal is OK", st, st == LARCH_OK);
  larch_graph_destroy(g);
  st = larch_stream_sync(s);
  expect("GPU (CUDA): the stream is still healthy (sync OK)", st, st == LARCH_OK);
  st = vv_fault_device_lost(s);
  expect("GPU (CUDA): fault_device_lost queued (OK), or the device was already lost", st,
         st == LARCH_OK || st == LARCH_ERR_DEVICE_LOST);
  st = larch_stream_sync(s);
  expect("GPU: the sync after it is DEVICE_LOST", st, st == LARCH_ERR_DEVICE_LOST);
  st = vv_fault_device_lost(s);
  expect("GPU: fault_device_lost on the lost device is DEVICE_LOST", st, st == LARCH_ERR_DEVICE_LOST);
  const uint8_t bytes[4] = {};
  st = larch_upload(s, buf, 0, bytes, sizeof bytes);
  expect("GPU: the next op (an upload into a buffer from before the fault) is DEVICE_LOST", st,
         st == LARCH_ERR_DEVICE_LOST);
  st = larch_stream_sync(s);
  expect("GPU: its sync is DEVICE_LOST (sticky)", st, st == LARCH_ERR_DEVICE_LOST);
  // The lost context's exit teardown can hang (#73): out without it. stdout to a pipe is fully buffered.
  std::fflush(stdout);
  std::_Exit(wrong != 0);
}
