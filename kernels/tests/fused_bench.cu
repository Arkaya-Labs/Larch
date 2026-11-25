/* fused_bench: each fused entry (larch.h, ABI 4) against the sequence of ops it stands for, inside a CUDA graph
   (testdata.h's replay_time, best of four replays), at the hybrid MoE target's layer shapes on GPU 0. Experts are
   cold as in decode: each call routes to the next of many random id sets over all 256 experts, and the dense weights
   rotate through copies past L2. Prints a markdown table of microseconds per call, unfused and fused, and the
   speedup; an entry is kept on by default only at 1.3x or better. Not a ctest (the test glob is .cpp only): timing
   takes the exclusive GPU lock.

     nvcc -O3 -std=c++20 -arch=sm_120 -ccbin g++-14 -Ikernels/include kernels/tests/fused_bench.cu \
       <build>/liblarch_kernels.a -lcublas -o <build>/fused_bench
     flock ~/src/gpu.lock <build>/fused_bench [--rows 1,4] */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "testdata.h"
#include "larch.h"

static larch_device* dev;
static larch_stream* str;
static std::vector<larch_buffer*> owned;
static rng g(3);

[[noreturn]] static void die(const char* what) {
  std::fprintf(stderr, "%s: %s\n", what, larch_last_error(dev));
  std::exit(1);
}
static larch_buffer* buffer(const bytes& b) {
  larch_buffer* p = nullptr;
  if (larch_buffer_alloc(dev, b.size(), &p) != LARCH_OK) die("alloc");
  if (larch_upload(str, p, 0, b.data(), b.size()) != LARCH_OK || larch_stream_sync(str) != LARCH_OK) die("upload");
  owned.push_back(p);
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
static bytes f32s(size_t n, float lo, float hi) {
  bytes b(n * 4);
  std::uniform_real_distribution<float> u(lo, hi);
  for (size_t i = 0; i < n; ++i) {
    const float v = u(g);
    std::memcpy(&b[i * 4], &v, 4);
  }
  return b;
}
// copies x [rows, cols] of t, back to back, and the bytes of one.
static larch_buffer* copies(larch_dtype t, int64_t rows, int64_t cols, int n, uint64_t* each) {
  *each = nbytes(t, {rows, cols});
  bytes all;
  for (int i = 0; i < n; ++i) {
    const bytes b = block_of(t) > 1 ? quant(t, rows, cols, g) : halves(size_t(rows * cols), g);
    all.insert(all.end(), b.begin(), b.end());
  }
  return buffer(all);
}

// The target's MoE block: d 2048, 256 experts of 512, 8 a token, a shared expert of 512.
constexpr int64_t D = 2048, E = 256, EF = 512, SLOTS = 8, SF = 512, IDSETS = 512, COPIES = 64;

struct row {
  std::string name;
  double unfused, fused;
};
static std::vector<row> table;

static void time(const std::string& name, const std::function<larch_status(int)>& unfused,
                 const std::function<larch_status(int)>& fused) {
  const double u = replay_time(str, unfused), f = replay_time(str, fused);
  if (u < 0 || f < 0) {
    std::printf("| %s | failed: %s | | |\n", name.c_str(), larch_last_error(dev));
    return;
  }
  table.push_back({name, u, f});
  std::printf("| %s | %.2f | %.2f | %.2fx |\n", name.c_str(), u, f, u / f);
  std::fflush(stdout);
}

static void moe(larch_dtype gate_up, larch_dtype down, larch_dtype shared, int64_t rows) {
  const std::string tag = std::string(" (") + DTYPE[gate_up] + "/" + DTYPE[down] + ", shared " + DTYPE[shared] +
                          ") rows " + std::to_string(rows);
  uint64_t each = 0;
  larch_buffer* gw = copies(gate_up, E * EF, D, 1, &each);
  larch_buffer* uw = copies(gate_up, E * EF, D, 1, &each);
  larch_buffer* dw = copies(down, E * D, EF, 1, &each);
  uint64_t sw_each = 0;
  larch_buffer* sw = copies(shared, D, SF, COPIES, &sw_each);
  std::vector<int32_t> idv;
  for (int64_t i = 0; i < IDSETS * rows; ++i) {
    std::vector<int32_t> all(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) all[size_t(e)] = int32_t(e);
    std::shuffle(all.begin(), all.end(), g);
    idv.insert(idv.end(), all.begin(), all.begin() + SLOTS);
  }
  bytes idb(idv.size() * 4);
  std::memcpy(idb.data(), idv.data(), idb.size());
  larch_buffer* ids = buffer(idb);
  larch_buffer* wts = buffer(f32s(size_t(rows * SLOTS), 0, 0.3f));
  larch_buffer* h = buffer(halves(size_t(rows * D), g));
  larch_buffer* act = buffer(halves(size_t(rows * SLOTS * EF), g));
  larch_buffer* sact = buffer(halves(size_t(rows * SF), g));
  larch_buffer* sg = buffer(halves(size_t(rows), g));
  larch_buffer* x = buffer(f32s(size_t(rows * D), -1, 1));
  larch_buffer* scratch = buffer(bytes(size_t(rows * SLOTS * EF * 2 * 3)));
  const uint64_t act_bytes = uint64_t(rows * SLOTS * EF * 2);
  const auto id_t = [&](int i) { return view(ids, uint64_t(i % IDSETS) * rows * SLOTS * 4, LARCH_I32, {rows, SLOTS}); };
  const larch_tensor gt = view(gw, 0, gate_up, {E * EF, D}), ut = view(uw, 0, gate_up, {E * EF, D});
  const larch_tensor dt = view(dw, 0, down, {E * D, EF}), ht = view(h, 0, LARCH_F16, {rows, D});
  const larch_tensor g3 = view(scratch, 0, LARCH_F16, {rows, SLOTS, EF}),
                      u3 = view(scratch, act_bytes, LARCH_F16, {rows, SLOTS, EF}),
                      a3 = view(scratch, 2 * act_bytes, LARCH_F16, {rows, SLOTS, EF});
  const larch_tensor g2 = view(scratch, 0, LARCH_F16, {rows * SLOTS, EF}),
                      u2 = view(scratch, act_bytes, LARCH_F16, {rows * SLOTS, EF}),
                      a2 = view(scratch, 2 * act_bytes, LARCH_F16, {rows * SLOTS, EF});
  time("1 moe_gate_up" + tag,
       [&](int i) {
         const larch_tensor id = id_t(i);
         larch_status s = larch_mmvq_id(str, &gt, &ht, &id, nullptr, nullptr, &g3);
         if (s == LARCH_OK) s = larch_mmvq_id(str, &ut, &ht, &id, nullptr, nullptr, &u3);
         return s == LARCH_OK ? larch_silu_mul(str, &g2, &u2, &a2) : s;
       },
       [&](int i) {
         const larch_tensor id = id_t(i);
         return larch_fused_moe_gate_up(str, &gt, &ut, &ht, &id, &a3);
       });
  const larch_tensor at = view(act, 0, LARCH_F16, {rows, SLOTS, EF}), wt = view(wts, 0, LARCH_F32, {rows, SLOTS});
  const larch_tensor sat = view(sact, 0, LARCH_F16, {rows, SF}), sgt = view(sg, 0, LARCH_F16, {rows, 1});
  const larch_tensor xt = view(x, 0, LARCH_F32, {rows, D}), t = view(scratch, 0, LARCH_F16, {rows, D});
  const auto sw_t = [&](int i) { return view(sw, uint64_t(i % COPIES) * sw_each, shared, {D, SF}); };
  time("2 moe_down" + tag,
       [&](int i) {
         const larch_tensor id = id_t(i), swt = sw_t(i);
         larch_status s = larch_mmvq(str, &swt, &sat, nullptr, &t);
         if (s == LARCH_OK) s = larch_sigmoid_mul(str, &t, &sgt, &t);
         if (s == LARCH_OK) s = larch_add(str, &xt, &t, &xt);
         if (s == LARCH_OK) s = larch_mmvq_id(str, &dt, &at, &id, nullptr, &wt, &t);
         return s == LARCH_OK ? larch_add(str, &xt, &t, &xt) : s;
       },
       [&](int i) {
         const larch_tensor id = id_t(i), swt = sw_t(i);
         return larch_fused_moe_down(str, &dt, &at, &id, &wt, &swt, &sat, &sgt, &xt);
       });
  // Entry 3 at the shared expert's dtype (gate and up), the f16 router and gate input; weights rotate past L2.
  uint64_t r_each = 0, gi_each = 0;
  larch_buffer* rw = copies(LARCH_F16, E, D, 16, &r_each);
  larch_buffer* sgw = copies(shared, SF, D, COPIES, &sw_each);
  larch_buffer* suw = copies(shared, SF, D, COPIES, &sw_each);
  larch_buffer* giw = copies(LARCH_F16, 1, D, 16, &gi_each);
  larch_buffer* nw = buffer(f32s(size_t(D), 0.5f, 1.5f));
  larch_buffer* lg = buffer(bytes(size_t(rows * E * 4)));
  larch_buffer* rt = buffer(bytes(size_t(rows * SLOTS * 8)));
  larch_buffer* hh = buffer(bytes(size_t(rows * D * 2)));
  larch_buffer* so = buffer(bytes(size_t(rows * SF * 2 * 3 + rows * 2)));
  const larch_tensor nwt = view(nw, 0, LARCH_F32, {D}), hht = view(hh, 0, LARCH_F16, {rows, D});
  const larch_tensor lgt = view(lg, 0, LARCH_F32, {rows, E}), it = view(rt, 0, LARCH_I32, {rows, SLOTS}),
                      wtt = view(rt, uint64_t(rows * SLOTS * 4), LARCH_F32, {rows, SLOTS});
  const larch_tensor sgo = view(so, 0, LARCH_F16, {rows, SF}), suo = view(so, uint64_t(rows * SF * 2), LARCH_F16, {rows, SF}),
                      sao = view(so, uint64_t(rows * SF * 4), LARCH_F16, {rows, SF}),
                      ggo = view(so, uint64_t(rows * SF * 6), LARCH_F16, {rows, 1});
  const auto w3 = [&](int i) {
    return std::vector<larch_tensor>{view(rw, uint64_t(i % 16) * r_each, LARCH_F16, {E, D}),
                                      view(sgw, uint64_t(i % COPIES) * sw_each, shared, {SF, D}),
                                      view(suw, uint64_t(i % COPIES) * sw_each, shared, {SF, D}),
                                      view(giw, uint64_t(i % 16) * gi_each, LARCH_F16, {1, D})};
  };
  time("3 moe_in" + tag,
       [&](int i) {
         const std::vector<larch_tensor> w = w3(i);
         larch_status s = larch_rmsnorm(str, &xt, &nwt, 1e-6f, &hht);
         if (s == LARCH_OK) s = larch_gemm(str, &hht, &w[0], nullptr, &lgt, 1, 0);
         if (s == LARCH_OK) s = larch_moe_route(str, &lgt, 0, &it, &wtt);
         if (s == LARCH_OK) s = larch_mmvq(str, &w[1], &hht, nullptr, &sgo);
         if (s == LARCH_OK) s = larch_mmvq(str, &w[2], &hht, nullptr, &suo);
         if (s == LARCH_OK) s = larch_silu_mul(str, &sgo, &suo, &sao);
         return s == LARCH_OK ? larch_gemm(str, &hht, &w[3], nullptr, &ggo, 1, 0) : s;
       },
       [&](int i) {
         const std::vector<larch_tensor> w = w3(i);
         return larch_fused_moe_in(str, &xt, &nwt, 1e-6f, &hht, &w[0], &lgt, &it, &wtt, &w[1], &w[2], &sao, &w[3],
                                    &ggo);
       });
  larch_stream_sync(str);
  for (larch_buffer* b : owned) larch_buffer_free(b);
  owned.clear();
}

// The hybrid layer's entries: 4 (GDN inputs), 5 (the recurrence), 6 (state-space out + residual), 7 (attention
// inputs), 8 (attention gate, out + residual).
static void hybrid(int64_t rows) {
  const std::string tag = " rows " + std::to_string(rows);
  constexpr int64_t CH = 8192, VD = 4096, VH = 32, KH = 16, HD = 128, KERN = 4, SLOTS_ = 8;
  uint64_t q_each = 0, z_each = 0, ab_each = 0;
  larch_buffer* wq = copies(LARCH_Q5_K, CH, D, 4, &q_each);
  larch_buffer* wz = copies(LARCH_Q3_K, VD, D, 4, &z_each);
  larch_buffer* wa = copies(LARCH_F16, VH, D, 16, &ab_each);
  larch_buffer* wb = copies(LARCH_F16, VH, D, 16, &ab_each);
  larch_buffer* nw = buffer(f32s(size_t(D), 0.5f, 1.5f));
  larch_buffer* x = buffer(f32s(size_t(rows * D), -1, 1));
  larch_buffer* out = buffer(bytes(size_t(rows * (D + CH + VD) * 2 + rows * VH * 8)));
  const larch_tensor xt = view(x, 0, LARCH_F32, {rows, D}), nwt = view(nw, 0, LARCH_F32, {D});
  const larch_tensor ht = view(out, 0, LARCH_F16, {rows, D}),
                      qo = view(out, uint64_t(rows * D * 2), LARCH_F16, {rows, CH}),
                      zo = view(out, uint64_t(rows * (D + CH) * 2), LARCH_F16, {rows, VD}),
                      ao = view(out, uint64_t(rows * (D + CH + VD) * 2), LARCH_F32, {rows, VH}),
                      bo = view(out, uint64_t(rows * (D + CH + VD) * 2 + rows * VH * 4), LARCH_F32, {rows, VH});
  const auto w4 = [&](int i) {
    return std::vector<larch_tensor>{view(wq, uint64_t(i % 4) * q_each, LARCH_Q5_K, {CH, D}),
                                      view(wz, uint64_t(i % 4) * z_each, LARCH_Q3_K, {VD, D}),
                                      view(wa, uint64_t(i % 16) * ab_each, LARCH_F16, {VH, D}),
                                      view(wb, uint64_t(i % 16) * ab_each, LARCH_F16, {VH, D})};
  };
  time("4 gdn_in (q5_k/q3_k, f16 alpha/beta)" + tag,
       [&](int i) {
         const std::vector<larch_tensor> w = w4(i);
         larch_status s = larch_rmsnorm(str, &xt, &nwt, 1e-6f, &ht);
         if (s == LARCH_OK) s = larch_mmvq(str, &w[0], &ht, nullptr, &qo);
         if (s == LARCH_OK) s = larch_mmvq(str, &w[1], &ht, nullptr, &zo);
         if (s == LARCH_OK) s = larch_gemm(str, &ht, &w[2], nullptr, &ao, 1, 0);
         return s == LARCH_OK ? larch_gemm(str, &ht, &w[3], nullptr, &bo, 1, 0) : s;
       },
       [&](int i) {
         const std::vector<larch_tensor> w = w4(i);
         return larch_fused_gdn_in(str, &xt, &nwt, 1e-6f, &ht, &w[0], &w[1], &w[2], &w[3], &qo, &zo, &ao, &bo);
       });

  // 5: one decode row per sequence, each in its own slot of a 2-layer state.
  const int64_t conv_slot = (KERN - 1) * CH, ssm_slot = VH * HD * HD, nslots = SLOTS_;
  larch_buffer* cs = buffer(halves(size_t(2 * nslots * conv_slot), g));
  larch_buffer* ss = buffer(f32s(size_t(2 * nslots * ssm_slot), -0.1f, 0.1f));
  larch_buffer* cw = buffer(f32s(size_t(CH * KERN), 0.05f, 0.45f));
  larch_buffer* small = buffer(f32s(size_t(2 * rows * VH + 2 * VH + HD), -1, 1));
  larch_buffer* zb = buffer(halves(size_t(rows * VD), g));
  std::vector<int32_t> iv;
  for (int64_t r = 0; r < rows; ++r) iv.push_back(int32_t(r));
  for (int64_t r = 0; r < rows; ++r) iv.push_back(1);
  bytes ib(iv.size() * 4);
  std::memcpy(ib.data(), iv.data(), ib.size());
  larch_buffer* idx = buffer(ib);
  larch_buffer* tmp = buffer(bytes(size_t(rows * (CH + 2 * VD) * 2)));
  const larch_gdn_state gs = {cs, ss, uint32_t(nslots), 2, 1, uint32_t(KERN), uint32_t(KH), uint32_t(VH), uint32_t(HD)};
  const larch_tensor cwt = view(cw, 0, LARCH_F32, {CH, KERN}), al = view(small, 0, LARCH_F32, {rows, VH}),
                      be = view(small, uint64_t(rows * VH * 4), LARCH_F32, {rows, VH}),
                      dt = view(small, uint64_t(2 * rows * VH * 4), LARCH_F32, {VH}),
                      av = view(small, uint64_t((2 * rows + 1) * VH * 4), LARCH_F32, {VH}),
                      gn = view(small, uint64_t((2 * rows + 2) * VH * 4), LARCH_F32, {HD});
  const larch_tensor slt = view(idx, 0, LARCH_I32, {rows}), lt = view(idx, uint64_t(rows * 4), LARCH_I32, {rows});
  const larch_tensor zt = view(zb, 0, LARCH_F16, {rows, VD}), conv = view(tmp, 0, LARCH_F16, {rows, CH}),
                      o2 = view(tmp, uint64_t(rows * CH * 2), LARCH_F16, {rows, VD}),
                      o3 = view(tmp, uint64_t(rows * CH * 2), LARCH_F16, {rows, VH, HD}),
                      gated = view(tmp, uint64_t(rows * (CH + VD) * 2), LARCH_F16, {rows, VD});
  time("5 gdn (hd 128, 32 v heads)" + tag,
       [&](int) {
         larch_status s = larch_gdn_conv(str, &qo, &cwt, &gs, &slt, &lt, nullptr, &conv);
         if (s == LARCH_OK) s = larch_gdn(str, &conv, &al, &be, &dt, &av, 1e-6f, &gs, &slt, &lt, nullptr, &o2);
         if (s == LARCH_OK) s = larch_qk_norm(str, &o3, &gn, 1e-6f);
         return s == LARCH_OK ? larch_silu_mul(str, &zt, &o2, &gated) : s;
       },
       [&](int) {
         return larch_fused_gdn(str, &qo, &cwt, &al, &be, &dt, &av, 1e-6f, &gs, &slt, &lt, nullptr, &gn, &zt, &conv,
                                 &gated);
       });

  // 6 and 8: the out projections onto x (8 behind the attention gate).
  for (const larch_dtype t : {LARCH_Q3_K, LARCH_Q8_0, LARCH_Q4_K}) {
    const bool gate = t == LARCH_Q4_K;
    uint64_t each = 0;
    larch_buffer* wo = copies(t, D, VD, 8, &each);
    larch_buffer* gb = buffer(halves(size_t(rows * VD), g));
    const larch_tensor gt = view(gb, 0, LARCH_F16, {rows, VD}), td = view(tmp, 0, LARCH_F16, {rows, D}),
                        gg = view(tmp, uint64_t(rows * D * 2), LARCH_F16, {rows, VD});
    time(std::string(gate ? "8 proj_add, gated (" : "6 proj_add (") + DTYPE[t] + ")" + tag,
         [&](int i) {
           const larch_tensor w = view(wo, uint64_t(i % 8) * each, t, {D, VD});
           larch_status s = gate ? larch_sigmoid_mul(str, &zt, &gt, &gg) : LARCH_OK;
           if (s == LARCH_OK) s = larch_mmvq(str, &w, gate ? &gg : &zt, nullptr, &td);
           return s == LARCH_OK ? larch_add(str, &xt, &td, &xt) : s;
         },
         [&](int i) {
           const larch_tensor w = view(wo, uint64_t(i % 8) * each, t, {D, VD});
           return larch_fused_proj_add(str, &w, &zt, gate ? &gt : nullptr, &xt);
         });
  }

  // 7: the gated Q's split, q and k norms, partial neox rope (64 of 256) and the KV write, 16 q heads, 2 kv heads.
  constexpr int64_t H = 16, KVH = 2, AHD = 256, BT = 16, BLOCKS = 64;
  larch_buffer* qin = buffer(halves(size_t(rows * H * 2 * AHD), g));
  larch_buffer* kvb = buffer(halves(size_t(2 * rows * KVH * AHD), g));
  larch_buffer* norms = buffer(f32s(size_t(2 * AHD), 0.5f, 1.5f));
  larch_buffer* pk = buffer(bytes(size_t(BLOCKS * BT * KVH * AHD * 2)));
  larch_buffer* pv = buffer(bytes(size_t(BLOCKS * BT * KVH * AHD * 2)));
  larch_buffer* qg = buffer(bytes(size_t(2 * rows * H * AHD * 2 + rows * KVH * AHD * 2)));
  bytes sel(size_t(2 * AHD * 2 * AHD) * 2, 0);
  for (int64_t half = 0; half < 2; ++half)
    for (int64_t i = 0; i < AHD; ++i) sel[size_t((half * AHD + i) * 2 * AHD + half * AHD + i) * 2 + 1] = 0x3c;
  larch_buffer* selb = buffer(sel);
  std::vector<int32_t> pv_(size_t(2 * rows));
  for (int64_t r = 0; r < rows; ++r) pv_[size_t(r)] = int32_t(1000 + r), pv_[size_t(rows + r)] = int32_t(r * 3);
  bytes pb(pv_.size() * 4);
  std::memcpy(pb.data(), pv_.data(), pb.size());
  larch_buffer* posb = buffer(pb);
  const larch_kv_pool pool = {pk, pv, uint32_t(BLOCKS), uint32_t(BT), 1, uint32_t(KVH), uint32_t(AHD), LARCH_F16, 0};
  larch_rope_params rp = {};
  rp.n_dims = 64, rp.neox = 1, rp.n_ctx_orig = 262144, rp.freq_base = 1e7f, rp.freq_scale = 1, rp.attn_factor = 1;
  rp.beta_fast = 32, rp.beta_slow = 1;
  const larch_tensor qi = view(qin, 0, LARCH_F16, {rows, H, 2 * AHD}), pr = view(qin, 0, LARCH_F16, {rows * H, 2 * AHD});
  const larch_tensor kt = view(kvb, 0, LARCH_F16, {rows, KVH, AHD}),
                      vt = view(kvb, uint64_t(rows * KVH * AHD * 2), LARCH_F16, {rows, KVH, AHD});
  const larch_tensor qn = view(norms, 0, LARCH_F32, {AHD}), kn = view(norms, AHD * 4, LARCH_F32, {AHD});
  const larch_tensor q3 = view(qg, 0, LARCH_F16, {rows, H, AHD}), q2 = view(qg, 0, LARCH_F16, {rows * H, AHD}),
                      g3 = view(qg, uint64_t(rows * H * AHD * 2), LARCH_F16, {rows, H, AHD}),
                      g2 = view(qg, uint64_t(rows * H * AHD * 2), LARCH_F16, {rows * H, AHD});
  const larch_tensor s0 = view(selb, 0, LARCH_F16, {AHD, 2 * AHD}),
                      s1 = view(selb, uint64_t(AHD * 2 * AHD * 2), LARCH_F16, {AHD, 2 * AHD});
  const larch_tensor pos = view(posb, 0, LARCH_I32, {rows}), slots = view(posb, uint64_t(rows * 4), LARCH_I32, {rows});
  time("7 attn_in (gated, normed, rope 64 of 256)" + tag,
       [&](int) {
         larch_status s = larch_gemm(str, &pr, &s0, nullptr, &q2, 1, 0);
         if (s == LARCH_OK) s = larch_gemm(str, &pr, &s1, nullptr, &g2, 1, 0);
         if (s == LARCH_OK) s = larch_qk_norm(str, &q3, &qn, 1e-6f);
         if (s == LARCH_OK) s = larch_qk_norm(str, &kt, &kn, 1e-6f);
         if (s == LARCH_OK) s = larch_rope(str, &q3, &pos, &rp);
         if (s == LARCH_OK) s = larch_rope(str, &kt, &pos, &rp);
         return s == LARCH_OK ? larch_kv_write(str, &kt, &vt, &slots, &pool) : s;
       },
       [&](int) {
         return larch_fused_attn_in(str, &qi, &kt, &vt, &qn, &kn, 1e-6f, &pos, &rp, &slots, &pool, &q3, &g3);
       });
  larch_stream_sync(str);
  for (larch_buffer* b : owned) larch_buffer_free(b);
  owned.clear();
}

int main(int argc, char** argv) {
  std::vector<int64_t> rows = {1, 4};
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) {
      rows.clear();
      for (char* p = argv[++i]; *p;) {
        rows.push_back(std::strtol(p, &p, 10));
        if (*p == ',') ++p;
      }
    } else {
      std::fprintf(stderr, "usage: fused_bench [--rows 1,4]\n");
      return 2;
    }
  }
  if (larch_device_open(0, &dev) != LARCH_OK || larch_stream_create(dev, &str) != LARCH_OK) die("open GPU 0");
  std::printf("| entry (target shapes) | unfused us | fused us | speedup |\n|---|---|---|---|\n");
  for (const int64_t r : rows) {
    moe(LARCH_Q3_K, LARCH_Q3_K, LARCH_Q5_K, r);
    moe(LARCH_Q3_K, LARCH_Q4_K, LARCH_Q8_0, r);
    hybrid(r);
  }
  larch_stream_destroy(str);
  larch_device_close(dev);
  return 0;
}
