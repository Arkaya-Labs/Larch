/* G1 (not part of the build): what a cold pair costs a split layer's decode step on GPU 0. One layer at m13's expert
   shapes (gate and up Q3_K [256 * 512, 2048], down Q4_K [256 * 2048, 512]), 100 of its 256 experts cold: their
   slices in page-locked host memory (larch_buffer_alloc_host) read in place through larch_buffer_wrap_host, the other
   156 in device memory, an i32 table between them. A step is the split layer's routed experts as the model runs them
   (larch_mmvq_id_mixed for gate and up, silu_mul, larch_mmvq_id_mixed for down, weighted), at 1, 4 and 8 rows of 8
   experts, each row with 0, 1 or 2 cold experts (and a step with one cold pair in all). testdata.h's replay_time
   captures a run of steps in one graph, each step routed to the next of 512 random id sets, and keeps the best of
   four replays. The same steps over one device tensor holding every expert (larch_mmvq_id) are the resident layer.

     nvcc -O3 -std=c++20 -arch=sm_120 -ccbin g++-14 -Ikernels/include -Ikernels/tests g1.cu \
       <build>/liblarch_kernels.a -lcublas -o g1
     flock ~/src/gpu.lock ./g1 */
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "testdata.h"
#include "larch.h"

static larch_device* dev;
static larch_stream* str;
static rng g(11);
constexpr int64_t D = 2048, E = 256, EF = 512, SLOTS = 8, IDSETS = 512, COLD = 100;

[[noreturn]] static void die(const char* what) {
  std::fprintf(stderr, "%s: %s\n", what, larch_last_error(dev));
  std::exit(1);
}
static larch_buffer* buffer(const bytes& b) {
  larch_buffer* p = nullptr;
  if (larch_buffer_alloc(dev, std::max<size_t>(b.size(), 1), &p) != LARCH_OK) die("alloc");
  if (larch_upload(str, p, 0, b.data(), b.size()) != LARCH_OK || larch_stream_sync(str) != LARCH_OK) die("upload");
  return p;
}
static larch_tensor view(larch_buffer* b, uint64_t off, larch_dtype t, std::vector<int64_t> shape) {
  larch_tensor x = {b, off, t, uint32_t(shape.size()), {}, {}};
  for (int64_t i = int64_t(shape.size()) - 1, stride = 1; i >= 0; --i) {
    x.shape[i] = shape[size_t(i)];
    x.stride[i] = stride;
    stride *= i == int64_t(shape.size()) - 1 ? shape[size_t(i)] / block_of(t) : shape[size_t(i)];
  }
  return x;
}

int main() {
  if (larch_device_open(0, &dev) != LARCH_OK || larch_stream_create(dev, &str) != LARCH_OK) die("open GPU 0");
  // Cold: 25 of every 64 experts, 100 in all, spread over the ids.
  std::vector<int32_t> table(E);
  std::vector<int32_t> cold, hot;
  for (int32_t e = 0; e < E; ++e) (e % 64 < 25 ? cold : hot).push_back(e);
  const larch_dtype dt[3] = {LARCH_Q3_K, LARCH_Q3_K, LARCH_Q4_K};
  const int64_t rows_of[3] = {EF, EF, D}, k_of[3] = {D, D, EF};
  // Each projection's whole tensor on the device, its hot slots on the device and its cold slots on the host.
  larch_buffer *whole[3], *hotb[3];
  uint64_t cold_at[3], cold_bytes[3], held = 0;
  std::vector<bytes> coldv(3);
  for (int p = 0; p < 3; ++p) {
    const bytes full = quant(dt[p], E * rows_of[p], k_of[p], g);
    const size_t slice = size_t(nbytes(dt[p], {rows_of[p], k_of[p]}));
    bytes hb;
    for (int32_t e : hot) hb.insert(hb.end(), full.begin() + ptrdiff_t(e * slice), full.begin() + ptrdiff_t((e + 1) * slice));
    for (int32_t e : cold) coldv[p].insert(coldv[p].end(), full.begin() + ptrdiff_t(e * slice), full.begin() + ptrdiff_t((e + 1) * slice));
    whole[p] = buffer(full);
    hotb[p] = buffer(hb);
    cold_at[p] = held;
    cold_bytes[p] = coldv[p].size();
    held += (cold_bytes[p] + 255) / 256 * 256;
  }
  for (size_t i = 0; i < hot.size(); ++i) table[size_t(hot[i])] = int32_t(i);
  for (size_t i = 0; i < cold.size(); ++i) table[size_t(cold[i])] = -1 - int32_t(i);
  larch_buffer* hostb = nullptr;
  void* host = nullptr;
  if (larch_buffer_alloc_host(dev, held, &hostb, &host) != LARCH_OK) die("alloc_host");
  for (int p = 0; p < 3; ++p) std::memcpy(static_cast<uint8_t*>(host) + cold_at[p], coldv[p].data(), cold_bytes[p]);
  larch_buffer* wrapped = nullptr;
  if (larch_buffer_wrap_host(dev, host, held, &wrapped) != LARCH_OK) die("wrap_host");
  larch_buffer* tb = buffer(bytes(reinterpret_cast<uint8_t*>(table.data()), reinterpret_cast<uint8_t*>(table.data()) + 4 * E));
  const larch_tensor tt = view(tb, 0, LARCH_I32, {E});
  std::printf("| rows | cold a row | resident us | split us | split gate+up us | split down us |\n|---|---|---|---|---|---|\n");
  for (const int64_t rows : {1, 4, 8}) {
    larch_buffer* h = buffer(halves(size_t(rows * D), g));
    larch_buffer* wts = nullptr;
    {
      bytes w(size_t(rows * SLOTS * 4));
      for (int64_t i = 0; i < rows * SLOTS; ++i) {
        const float v = 0.125f;
        std::memcpy(&w[size_t(i) * 4], &v, 4);
      }
      wts = buffer(w);
    }
    const uint64_t act = uint64_t(rows * SLOTS * EF * 2);
    larch_buffer* scratch = buffer(bytes(size_t(3 * act + rows * D * 2)));
    const larch_tensor ht = view(h, 0, LARCH_F16, {rows, D}), wt = view(wts, 0, LARCH_F32, {rows, SLOTS});
    const larch_tensor g3 = view(scratch, 0, LARCH_F16, {rows, SLOTS, EF}),
                       u3 = view(scratch, act, LARCH_F16, {rows, SLOTS, EF}),
                       a3 = view(scratch, 2 * act, LARCH_F16, {rows, SLOTS, EF}),
                       g2 = view(scratch, 0, LARCH_F16, {rows * SLOTS, EF}),
                       u2 = view(scratch, act, LARCH_F16, {rows * SLOTS, EF}),
                       a2 = view(scratch, 2 * act, LARCH_F16, {rows * SLOTS, EF}),
                       out = view(scratch, 3 * act, LARCH_F16, {rows, D});
    // c cold experts a row (c = -1: one cold pair in the whole step, on row 0).
    for (const int c : {0, 1, 2, -1}) {
      if (c == -1 && rows == 1) continue;
      std::vector<int32_t> idv;
      for (int64_t i = 0; i < IDSETS * rows; ++i) {
        const int nc = c >= 0 ? c : (i % rows == 0 ? 1 : 0);
        std::set<int32_t> pick;
        while (int(pick.size()) < nc) pick.insert(cold[g() % cold.size()]);
        while (int(pick.size()) < SLOTS) pick.insert(hot[g() % hot.size()]);
        std::vector<int32_t> row(pick.begin(), pick.end());
        std::shuffle(row.begin(), row.end(), g);
        idv.insert(idv.end(), row.begin(), row.end());
      }
      larch_buffer* ids = buffer(bytes(reinterpret_cast<uint8_t*>(idv.data()), reinterpret_cast<uint8_t*>(idv.data()) + 4 * idv.size()));
      const auto id_t = [&](int i) { return view(ids, uint64_t(i % IDSETS) * rows * SLOTS * 4, LARCH_I32, {rows, SLOTS}); };
      larch_tensor W[3], H[3], C[3];
      for (int p = 0; p < 3; ++p) {
        W[p] = view(whole[p], 0, dt[p], {E * rows_of[p], k_of[p]});
        H[p] = view(hotb[p], 0, dt[p], {int64_t(hot.size()) * rows_of[p], k_of[p]});
        C[p] = view(wrapped, cold_at[p], dt[p], {int64_t(cold.size()) * rows_of[p], k_of[p]});
      }
      const auto gate_up = [&](int i, bool split) {
        const larch_tensor id = id_t(i);
        larch_status s = split ? larch_mmvq_id_mixed(str, &H[0], &C[0], &tt, &ht, &id, nullptr, &g3)
                               : larch_mmvq_id(str, &W[0], &ht, &id, nullptr, nullptr, &g3);
        if (s == LARCH_OK)
          s = split ? larch_mmvq_id_mixed(str, &H[1], &C[1], &tt, &ht, &id, nullptr, &u3)
                    : larch_mmvq_id(str, &W[1], &ht, &id, nullptr, nullptr, &u3);
        return s == LARCH_OK ? larch_silu_mul(str, &g2, &u2, &a2) : s;
      };
      const auto down = [&](int i, bool split) {
        const larch_tensor id = id_t(i);
        return split ? larch_mmvq_id_mixed(str, &H[2], &C[2], &tt, &a3, &id, &wt, &out)
                     : larch_mmvq_id(str, &W[2], &a3, &id, nullptr, &wt, &out);
      };
      const double resident = replay_time(str, [&](int i) {
        const larch_status s = gate_up(i, false);
        return s == LARCH_OK ? down(i, false) : s;
      });
      const double split = replay_time(str, [&](int i) {
        const larch_status s = gate_up(i, true);
        return s == LARCH_OK ? down(i, true) : s;
      });
      const double sgu = replay_time(str, [&](int i) { return gate_up(i, true); });
      const double sdn = replay_time(str, [&](int i) { return down(i, true); });
      if (resident < 0 || split < 0 || sgu < 0 || sdn < 0) die("replay");
      std::printf("| %lld | %s | %.1f | %.1f | %.1f | %.1f |\n", (long long)rows,
                  c >= 0 ? std::to_string(c).c_str() : "1 in the step", resident, split, sgu, sdn);
      larch_buffer_free(ids);
    }
    larch_buffer_free(h);
    larch_buffer_free(wts);
    larch_buffer_free(scratch);
  }
  larch_buffer_free(wrapped);
  larch_buffer_free(hostb);
  larch_stream_destroy(str);
  larch_device_close(dev);
  return 0;
}
