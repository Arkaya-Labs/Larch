// The usable-device rule without a GPU (common/arch.h), then the count it gives: the CPU build has
// no GPU ordinals, and on a GPU build every ordinal below the count opens (and on CUDA and HIP runs
// an op: a launch is what fails without code for the device) while the count itself does not.
#include <cstdint>
#include <cstdio>

#include "../src/common/arch.h"
#include "larch.h"

static int wrong = 0;

static void expect(bool ok, const char* what, long value) {
  if (!ok) std::fprintf(stderr, "WRONG: %s (%ld): %s\n", what, value, larch_last_error(nullptr));
  wrong += !ok;
}

int main() {
  // The default 86-real;89-real;120-real;120-virtual, as __CUDA_ARCH_LIST__ spells it.
  const int built[] = {860, 890, 1200};
  const struct {
    int cc;
    bool runs;
  } cuda[] = {{860, true},  {870, true},  {890, true},   {800, false},  {750, false},
              {900, false}, {1000, false}, {1200, true}, {1210, true}, {1300, true}};
  for (auto c : cuda) expect(arch::cuda_runs(built, 3, c.cc) == c.runs, "cuda_runs for capability", c.cc);
  expect(!arch::cuda_runs(built, 0, 860), "cuda_runs with no architectures", 0);
  const char* hip = "gfx1030,gfx1100,gfx1101,gfx1102,gfx1200,gfx1201";
  const struct {
    const char* gcn;
    bool runs;
  } amd[] = {{"gfx1030:sramecc-:xnack-", true}, {"gfx1201", true}, {"gfx1032", false},
             {"gfx1036:xnack-", false},         {"gfx103", false}, {"gfx11000", false}};
  for (int i = 0; i < int(sizeof amd / sizeof amd[0]); ++i)
    expect(arch::hip_runs(hip, amd[i].gcn) == amd[i].runs, "hip_runs for row", i);
  expect(arch::hip_runs("gfx1030:xnack-,gfx1100", "gfx1030"), "hip_runs ignores built features", 0);

  int32_t count = -1;
  larch_backend backend;
  uint32_t runtime = 0, driver = 0;
  if (larch_device_count(&count) != LARCH_OK || count < 0 ||
      larch_backend_info(&backend, &runtime, &driver) != LARCH_OK) {
    std::fprintf(stderr, "device_count or backend_info failed: %s\n", larch_last_error(nullptr));
    return 1;
  }
  expect(backend != LARCH_BACKEND_CPU || count == 0, "GPUs on the CPU build", count);
  for (int32_t k = 0; k < count; ++k) {
    larch_device* d = nullptr;
    larch_stream* s = nullptr;
    larch_buffer* a = nullptr;
    larch_buffer* b = nullptr;
    if (larch_device_open(k, &d) || larch_stream_create(d, &s) || larch_buffer_alloc(d, 16, &a) ||
        larch_buffer_alloc(d, 8, &b)) {
      expect(false, "open, stream and buffers on ordinal", k);
      continue;
    }
    if (backend == LARCH_BACKEND_CUDA || backend == LARCH_BACKEND_HIP) {
      const char zeros[16] = {};
      larch_tensor ta{a, 0, LARCH_F32, 2, {1, 4}, {4, 1}}, tb{b, 0, LARCH_F16, 2, {1, 4}, {4, 1}};
      expect(!larch_upload(s, a, 0, zeros, 16) && !larch_upload(s, b, 0, zeros, 8) &&
                 !larch_add(s, &ta, &tb, &ta) && !larch_stream_sync(s),
             "add on ordinal", k);
    }
    larch_buffer_free(b);
    larch_buffer_free(a);
    larch_stream_destroy(s);
    larch_device_close(d);
  }
  larch_device* none = nullptr;
  expect(larch_device_open(count, &none) == LARCH_ERR_INVALID_ARG, "open at the count", count);
  std::printf("device_count: %d usable GPUs; %s\n", count, wrong ? "WRONG" : "ok");
  return wrong != 0;
}
