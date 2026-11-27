// larch_buffer_wrap_host on GPU 0 over page-locked host memory (larch_buffer_alloc_host): mmvq_id over a wrapped range
// of Q4_K and Q8_0 experts writes the same bytes as the same call over a device copy of them, and freeing the wrap
// leaves the host buffer whole. A malloc'd pointer and a range past the host buffer are UNSUPPORTED. 77 (skip) where
// there is no GPU or it keeps no host buffers, or (Metal, others) wraps only what it wraps already.
#include <cstdio>
#include <cstdlib>

#include "testdata.h"

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)
#define OK(call) CHECK((call) == LARCH_OK)

namespace {

larch_tensor dense(larch_buffer* b, uint64_t off, larch_dtype t, std::vector<int64_t> shape) {
  larch_tensor x = {b, off, t, uint32_t(shape.size()), {}, {}};
  for (int64_t d = int64_t(shape.size()) - 1, stride = 1; d >= 0; --d) {
    x.shape[d] = shape[size_t(d)];
    x.stride[d] = stride;
    stride *= d + 1 == int64_t(shape.size()) ? shape[size_t(d)] / block_of(t) : shape[size_t(d)];
  }
  return x;
}

}  // namespace

int main() {
  int32_t gpus = 0;
  OK(larch_device_count(&gpus));
  if (gpus < 1) return 77;
  larch_device* dev = nullptr;
  larch_stream* s = nullptr;
  OK(larch_device_open(0, &dev));
  OK(larch_stream_create(dev, &s));
  larch_device_props props{};
  OK(larch_device_props_get(dev, &props));
  if (props.backend != LARCH_BACKEND_CUDA) return 77;
  const int64_t e = 8, n = 64, k = 512, rows = 3, used = 2;
  rng g(7);
  for (const larch_dtype t : {LARCH_Q4_K, LARCH_Q8_0}) {
    const bytes w = quant(t, e * n, k, g);
    const uint64_t at = 4096, held = at + w.size() + 4096;
    larch_buffer* hb = nullptr;
    void* host = nullptr;
    const larch_status hs = larch_buffer_alloc_host(dev, held, &hb, &host);
    if (hs == LARCH_ERR_UNSUPPORTED) return 77;
    OK(hs);
    auto* h = static_cast<uint8_t*>(host);
    std::memcpy(h + at, w.data(), w.size());
    larch_buffer* wrapped = nullptr;
    OK(larch_buffer_wrap_host(dev, h + at, w.size(), &wrapped));
    larch_buffer* copy = nullptr;
    OK(larch_buffer_alloc(dev, w.size(), &copy));
    OK(larch_upload(s, copy, 0, w.data(), w.size()));
    // x, ids, and the two outputs on the device.
    const bytes x = halves(size_t(rows * k), g);
    std::vector<int32_t> ids(size_t(rows * used));
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = int32_t((i * 5 + 3) % size_t(e));
    const uint64_t xo = 0, io = x.size(), oo = io + 4 * ids.size(), ob = 2 * uint64_t(rows * used * n);
    larch_buffer* io_buf = nullptr;
    OK(larch_buffer_alloc(dev, oo + 2 * ob, &io_buf));
    OK(larch_upload(s, io_buf, xo, x.data(), x.size()));
    OK(larch_upload(s, io_buf, io, ids.data(), 4 * ids.size()));
    const larch_tensor xt = dense(io_buf, xo, LARCH_F16, {rows, k});
    const larch_tensor it = dense(io_buf, io, LARCH_I32, {rows, used});
    const larch_tensor o1 = dense(io_buf, oo, LARCH_F16, {rows, used, n});
    const larch_tensor o2 = dense(io_buf, oo + ob, LARCH_F16, {rows, used, n});
    const larch_tensor ww = dense(wrapped, 0, t, {e * n, k});
    const larch_tensor wc = dense(copy, 0, t, {e * n, k});
    OK(larch_mmvq_id(s, &ww, &xt, &it, nullptr, nullptr, &o1));
    OK(larch_mmvq_id(s, &wc, &xt, &it, nullptr, nullptr, &o2));
    bytes a(ob), b(ob);
    OK(larch_download(s, io_buf, oo, a.data(), ob));
    OK(larch_download(s, io_buf, oo + ob, b.data(), ob));
    OK(larch_stream_sync(s));
    CHECK(a == b);
    CHECK(fnv(a.data(), a.size()) != fnv(bytes(ob).data(), ob));  // not all zeros
    // Outside every host buffer: a malloc'd pointer, and a range that runs one byte past this one.
    std::vector<uint8_t> plain(w.size());
    larch_buffer* bad = nullptr;
    CHECK(larch_buffer_wrap_host(dev, plain.data(), plain.size(), &bad) == LARCH_ERR_UNSUPPORTED);
    CHECK(larch_buffer_wrap_host(dev, h + at, held - at + 1, &bad) == LARCH_ERR_UNSUPPORTED);
    larch_buffer_free(wrapped);
    CHECK(std::memcmp(h + at, w.data(), w.size()) == 0);  // the host buffer outlives its wrap, whole
    larch_buffer_free(io_buf);
    larch_buffer_free(copy);
    larch_buffer_free(hb);
  }
  larch_stream_destroy(s);
  larch_device_close(dev);
  std::puts("wrap_host: ok");
  return 0;
}
