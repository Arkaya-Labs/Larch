// Inkling 2's larch.h entries (ABI 6) on Metal, in the CPU reference's f32 order (kernels/src/cpu/inkling.cpp and
// attn_rel in ops.cpp). backend.mm checks every shape and dtype on the host; nothing here checks data, except that a
// bad expert id or relative distance reads nothing out of bounds. Sums accumulate in f32 where the reference's are in
// double: the tolerances of kernels/tests/inkling.cpp allow that.
#include <metal_stdlib>

#include "args.h"

using namespace metal;

typedef device const uchar* bytes_in;

template <typename T>
static inline T at(bytes_in p, u64 rs, uint i) {  // element i of a vector, or row i's first element
  return *(device const T*)(p + i * rs);
}
// Element i of a row that is f32 or f16.
static inline float load(bytes_in row, uint i, uint f32) {
  return f32 ? ((device const float*)row)[i] : float(((device const half*)row)[i]);
}
static inline void store(device uchar* row, uint i, uint f32, float v) {
  if (f32)
    ((device float*)row)[i] = v;
  else
    ((device half*)row)[i] = half(v);
}

/// softcap: out = cap tanh(x / cap), thread (column, row). Each thread reads the x it writes, so out may be x.
kernel void vv_softcap(bytes_in x [[buffer(0)]], device uchar* out [[buffer(1)]], constant softcap_args& a [[buffer(2)]],
                       uint2 g [[thread_position_in_grid]]) {
  const float v = ((device const float*)(x + u64(g.y) * a.x_rs))[g.x];
  ((device float*)(out + u64(g.y) * a.out_rs))[g.x] = a.cap * precise::tanh(v * (1.0f / a.cap));
}

/// loop_gate: out = (x - tanh(forget) x) + tanh(gain) msg, thread (column, row). out may be x.
kernel void vv_loop_gate(bytes_in x [[buffer(0)]], bytes_in msg [[buffer(1)]], bytes_in gain [[buffer(2)]],
                         bytes_in forget [[buffer(3)]], device uchar* out [[buffer(4)]],
                         constant loop_gate_args& a [[buffer(5)]], uint2 g [[thread_position_in_grid]]) {
  const float xi = ((device const float*)(x + u64(g.y) * a.x_rs))[g.x];
  const float m = ((device const float*)(msg + u64(g.y) * a.msg_rs))[g.x];
  const float gn = load(gain + u64(g.y) * a.gain_rs, g.x, a.f32_gain);
  const float fg = load(forget + u64(g.y) * a.forget_rs, g.x, a.f32_forget);
  const float kept = xi - precise::tanh(fg) * xi;
  ((device float*)(out + u64(g.y) * a.out_rs))[g.x] = kept + precise::tanh(gn) * m;
}

/// situ_mul: thread (k, slot j, row r) of the fused gate-up output. An expert id outside the scale table gives NaN, so
/// a bad id shows in the output instead of reading past the table.
kernel void vv_situ_mul(bytes_in gu [[buffer(0)]], bytes_in ids [[buffer(1)]], bytes_in scale [[buffer(2)]],
                        device uchar* out [[buffer(3)]], constant situ_args& a [[buffer(4)]],
                        uint3 g [[thread_position_in_grid]]) {
  const uint k = g.x, j = g.y, r = g.z;
  bytes_in row = gu + u64(r) * a.gu_rs;
  float gt = load(row, j * 2 * a.ff + k, a.f32_in), u = load(row, j * 2 * a.ff + a.ff + k, a.f32_in);
  if (a.has_scale) {
    const int e = ((device const int*)(ids + u64(r) * a.ids_rs))[j];
    const float s = e >= 0 && uint(e) < a.experts ? at<float>(scale, a.scale_rs, uint(e)) : NAN;
    gt *= s;
    u *= s;
  }
  const float act = a.beta * precise::tanh(gt * (1.0f / a.beta)) * (1.0f / (1.0f + precise::exp(-gt)));
  const float up = a.lin_beta > 0.0f ? a.lin_beta * precise::tanh(u * (1.0f / a.lin_beta)) : u;
  ((device half*)(out + u64(r) * a.out_rs))[j * a.ff + k] = half(act * up);
}

/// clip_norm: simdgroup (head h, row r); lane l holds elements l, l + 32, ... of the head. out may be x: each lane
/// reads its elements before it writes them.
kernel void vv_clip_norm(bytes_in x [[buffer(0)]], bytes_in scale [[buffer(1)]], bytes_in pos [[buffer(2)]],
                         device uchar* out [[buffer(3)]], constant clip_args& a [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint h = tg.x, r = tg.y;
  bytes_in xr = x + u64(r) * a.x_rs;
  device const float* sc = (device const float*)(scale + u64(h) * a.scale_rs);
  float y[CLIP_MAX_DIM / 32];
  float ss = 0;
  for (uint m = 0; m < CLIP_MAX_DIM / 32; ++m) {
    const uint i = lane + 32 * m;
    y[m] = 0;
    if (i >= a.dim) continue;
    float v = load(xr, h * a.dim + i, a.f32_in);
    if (a.has_scale) v *= sc[i];
    y[m] = v;
    ss += v * v;
  }
  const float inv = 1.0f / max(precise::sqrt(simd_sum(ss)), a.t);
  float tau = 1.0f;
  if (a.has_pos) {
    float p = float(at<int>(pos, a.pos_rs, r)) + 1.0f;
    p = max(p * (1.0f / float(a.floor)), 1.0f);
    tau = precise::log(p) * a.alpha + 1.0f;
  }
  device uchar* o = out + u64(r) * a.out_rs;
  for (uint m = 0; m < CLIP_MAX_DIM / 32; ++m) {
    const uint i = lane + 32 * m;
    if (i >= a.dim) break;
    float v = y[m] * inv * a.t;
    if (a.has_pos) v *= tau;
    store(o, h * a.dim + i, a.f32_out, v);
  }
}

/// depth_mix: threadgroup r takes row r. One pass over the d columns sums each state's squares and its dot with
/// each query (a state's norm scales its dots, so they need not wait for it); then each query's softmax over the
/// states, and the weighted sum of the normed states.
kernel void vv_depth_mix(bytes_in states [[buffer(0)]], bytes_in query [[buffer(1)]], device uchar* out [[buffer(2)]],
                         constant depth_args& a [[buffer(3)]], uint r [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint SUMS = DEPTH_MAX_STATES * (DEPTH_MAX_Q + 1);
  threadgroup float part[DEPTH_THREADS / 32][SUMS];
  threadgroup float total[SUMS];
  const uint per = a.nq + 1, used = a.n * per;  // sum j * per: state j's squares; + 1 + i: its dot with query i
  float acc[SUMS];
  for (uint i = 0; i < SUMS; ++i) acc[i] = 0;
  for (uint k = tid; k < a.d; k += DEPTH_THREADS) {
    float qk[DEPTH_MAX_Q];
    for (uint i = 0; i < a.nq; ++i) qk[i] = ((device const float*)(query + u64(i) * a.q_rs))[k];
    for (uint j = 0; j < a.n; ++j) {
      const float v = ((device const float*)(states + u64(j) * a.st_rs))[u64(r) * a.d + k];
      acc[j * per] += v * v;
      for (uint i = 0; i < a.nq; ++i) acc[j * per + 1 + i] += v * qk[i];
    }
  }
  for (uint i = 0; i < used; ++i) {
    const float s = simd_sum(acc[i]);
    if (lane == 0) part[sg][i] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < used) {
    float s = 0;
    for (uint w = 0; w < DEPTH_THREADS / 32; ++w) s += part[w][tid];
    total[tid] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv[DEPTH_MAX_STATES];
  for (uint j = 0; j < a.n; ++j) inv[j] = 1.0f / precise::sqrt(total[j * per] / float(a.d) + 1e-6f);
  const float root_d = precise::sqrt(float(a.d));
  device float* o = (device float*)(out + u64(r) * a.out_rs);
  for (uint i = 0; i < a.nq; ++i) {
    float p[DEPTH_MAX_STATES];
    float mx = -INFINITY, z = 0;
    for (uint j = 0; j < a.n; ++j) {
      p[j] = total[j * per + 1 + i] * inv[j] / root_d;
      mx = max(mx, p[j]);
    }
    for (uint j = 0; j < a.n; ++j) {
      p[j] = precise::exp(p[j] - mx);
      z += p[j];
    }
    for (uint k = tid; k < a.d; k += DEPTH_THREADS) {
      float v = 0;
      for (uint j = 0; j < a.n; ++j)
        v += p[j] * (((device const float*)(states + u64(j) * a.st_rs))[u64(r) * a.d + k] * inv[j]);
      o[i * a.d + k] = v / z;
    }
  }
}

/// moe_route_sigmoid: one simdgroup per row, as vv_moe_route. Lane l holds experts l, l + 32, ...: p = sigmoid(logit)
/// and the score p + bias; each of the k picks takes the largest score not yet taken, the lower id on a tie. The
/// weights are the picks' p over (their sum + eps), times the scale and the expert's down scale.
kernel void vv_moe_route_sigmoid(bytes_in lg [[buffer(0)]], bytes_in bias [[buffer(1)]], bytes_in down [[buffer(2)]],
                                 device uchar* ids [[buffer(3)]], device uchar* wts [[buffer(4)]],
                                 constant route_sigmoid_args& a [[buffer(5)]], uint r [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  device const float* l = (device const float*)(lg + u64(r) * a.lg_rs);
  device int* out_ids = (device int*)(ids + u64(r) * a.ids_rs);
  device float* out_w = (device float*)(wts + u64(r) * a.w_rs);
  const uint per = (a.experts + 31) / 32;
  float p[MOE_ROUTE_MAX / 32], v[MOE_ROUTE_MAX / 32];
  for (uint i = 0; i < per; ++i) {
    const uint e = lane + 32 * i;
    p[i] = e < a.experts ? 1.0f / (1.0f + precise::exp(-l[e])) : 0.0f;
    v[i] = e >= a.experts ? -INFINITY : a.has_bias ? p[i] + at<float>(bias, a.bias_rs, e) : p[i];
  }
  uint taken = 0;  // bit i: expert lane + 32 i is picked
  float sum = 0;
  for (uint j = 0; j < a.k; ++j) {
    float best = -INFINITY;
    uint id = UINT_MAX;
    for (uint i = 0; i < per; ++i) {
      const uint e = lane + 32 * i;
      if (e < a.experts && !(taken >> i & 1) && (id == UINT_MAX || v[i] > best)) {
        best = v[i];
        id = e;
      }
    }
    const float m = simd_max(id == UINT_MAX ? -INFINITY : best);
    const uint pick = simd_min(id != UINT_MAX && best == m ? id : UINT_MAX);
    if (lane == pick % 32) taken |= 1u << (pick / 32);
    const float pp = simd_shuffle(p[pick / 32], ushort(pick % 32));
    sum += pp;
    if (!lane) {
      out_ids[j] = int(pick);
      out_w[j] = pp;
    }
  }
  const float denom = sum + a.eps, mult = a.scale == 0.0f ? 1.0f : a.scale;
  if (!lane)
    for (uint j = 0; j < a.k; ++j) {
      float w = out_w[j] / denom * mult;
      if (a.has_down) w *= at<float>(down, a.down_rs, uint(out_ids[j]));
      out_w[j] = w;
    }
}

/// short_conv: thread (channel c, sequence s) walks the sequence's rows in order with the channel's last taps - 1
/// inputs in registers (the slot's state first), as vv_gdn_conv. Per row: y = w[c] . window, plus each lag's input
/// times its dynamic tap, then (x + y) * scale, plus the residual. out may be the residual: the thread reads its
/// element first. Snapshots and the live slot get the window as of their row (x is f16, so exact).
kernel void vv_short_conv(bytes_in x [[buffer(0)]], bytes_in w [[buffer(1)]], bytes_in dyn [[buffer(2)]],
                          bytes_in res [[buffer(3)]], device uchar* st [[buffer(4)]], bytes_in slots [[buffer(5)]],
                          bytes_in lens [[buffer(6)]], bytes_in snaps [[buffer(7)]], device uchar* out [[buffer(8)]],
                          constant short_conv_args& a [[buffer(9)]], uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, s = g.y, hist = a.taps - 1, at_c = a.offset + c, head = c / (a.c / a.heads);
  uint r0 = 0;
  for (uint i = 0; i < s; ++i) r0 += uint(max(at<int>(lens, a.lens_rs, i), 0));
  const uint len = uint(max(at<int>(lens, a.lens_rs, s), 0));
  device half* live = (device half*)(st + u64(uint(at<int>(slots, a.slots_rs, s))) * a.state_rs);
  device const float* wc = (device const float*)(w + u64(c) * a.w_rs);
  float h[GDN_MAX_TAPS - 1], wk[GDN_MAX_TAPS];
  for (uint i = 0; i < hist; ++i) h[i] = float(live[i * a.channels + at_c]);
  for (uint i = 0; i < a.taps; ++i) wk[i] = wc[i];
  for (uint t = 0; t < len; ++t) {
    const uint r = r0 + t;
    const float in = float(((device const half*)(x + u64(r) * a.x_rs))[c]);
    float y = 0;
    for (uint i = 0; i < hist; ++i) y += wk[i] * h[i];
    y += wk[hist] * in;
    if (a.has_dyn) {
      bytes_in dr = dyn + u64(r) * a.dyn_rs;
      for (uint lag = 0; lag < a.taps; ++lag)
        y += (lag == 0 ? in : h[hist - lag]) * load(dr, head * a.taps + lag, a.f32_dyn);
    }
    y = (in + y) * a.scale;
    if (a.has_res) y = ((device const float*)(res + u64(r) * a.res_rs))[c] + y;
    store(out + u64(r) * a.out_rs, c, a.f32_out, y);
    for (uint i = 0; i + 1 < hist; ++i) h[i] = h[i + 1];
    h[hist - 1] = in;
    if (a.has_snaps) {
      const int snap = at<int>(snaps, a.snaps_rs, r);
      if (snap >= 0) {
        device half* d = (device half*)(st + u64(uint(snap)) * a.state_rs);
        for (uint i = 0; i < hist; ++i) d[i * a.channels + at_c] = half(h[i]);
      }
    }
  }
  for (uint i = 0; i < hist; ++i) live[i * a.channels + at_c] = half(h[i]);
}

/// attn_rel, 1 of 2: simdgroup (split, q head hh, row r) over its share of the positions row r attends to, 32 at a
/// time, one a lane: logit = scale (q . k) + (b - sink), b = rel_q . rel_w[p - p'] within the extent; an online
/// softmax as vv_attn_split's, into the share's max, sum and head_dim accumulators. The split bounds depend only on
/// the row's own positions and the call's splits, so a row of a span equals that row as its own span, and as a
/// decode row (larch.h's decode form: row r its own sequence of lens[r] positions, 0 a padding row), bit for bit.
/// A masked-out position counts as absent; a share with none live has sum 0, which the reduce skips. A mask bit past
/// the row's own position (a negative distance) gets no bias.
kernel void vv_attn_rel(bytes_in q [[buffer(0)]], device const half* kp [[buffer(1)]], device const half* vp [[buffer(2)]],
                        bytes_in bt [[buffer(3)]], bytes_in lens [[buffer(4)]], bytes_in start [[buffer(5)]],
                        bytes_in mask [[buffer(6)]], bytes_in relq [[buffer(7)]], bytes_in relw [[buffer(8)]],
                        bytes_in sink [[buffer(9)]], device float* partial [[buffer(10)]],
                        constant attn_rel_args& ra [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
  constant attn_args& a = ra.at;
  const uint split = tg.x / a.heads, hh = tg.x % a.heads, r = tg.y;
  uint s = r, p0, p, hi;
  if (ra.decode) {  // p0 at the end: no position is a span position
    hi = p0 = uint(max(at<int>(lens, a.lens_rs, r), 0));
    p = hi - 1;  // used only when hi > 0
  } else {
    uint first = 0;
    for (s = 0; s + 1 < a.seqs && r >= first + uint(at<int>(lens, a.lens_rs, s)); ++s)
      first += uint(at<int>(lens, a.lens_rs, s));
    p0 = uint(at<int>(start, a.start_rs, s));
    p = p0 + r - first;
    hi = ra.has_mask ? p0 + uint(at<int>(lens, a.lens_rs, s)) : p + 1;
  }
  const uint end = p + 1;  // the row's window ends at its own position (0 for a padding row: p wrapped)
  const uint lo = ra.window && end > ra.window ? end - ra.window : 0;
  const uint bits = ra.has_mask ? uint(at<int>(mask, ra.mask_rs, r)) : 0;
  const uint n = hi > lo ? hi - lo : 0, used = max(1u, min(a.splits, n / ATTN_SPLIT_POS));
  const uint b0 = split < used ? lo + uint(u64(n) * split / used) : hi;
  const uint b1 = split < used ? lo + uint(u64(n) * (split + 1) / used) : hi;
  device float* dst = partial + ((u64(r) * a.heads + hh) * a.splits + split) * a.partial_stride;
  if (b0 >= b1) {  // the same for the whole simdgroup
    if (lane == 0) dst[1] = 0.0f;
    return;
  }
  threadgroup half qs[ATTN_MAX_HEAD_DIM];
  threadgroup float rq[ATTN_REL_MAX_NREL];
  device const half* qr = (device const half*)(q + u64(r) * a.q_rs) + hh * a.hd;
  for (uint d = lane; d < a.hd; d += 32) qs[d] = qr[d];
  bytes_in rq_row = relq + u64(r) * ra.relq_rs;
  for (uint i = lane; i < ra.n_rel; i += 32) rq[i] = load(rq_row, hh * ra.n_rel + i, ra.f32_relq);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float sk = load(sink + u64(r) * ra.sink_rs, hh, ra.f32_sink);
  const uint kvh = hh / (a.heads / a.kv_heads);
  device const int* table = (device const int*)(bt + u64(s) * a.bt_rs);
  float mx = -INFINITY, sum = 0, acc[ATTN_MAX_HEAD_DIM / 32] = {};
  for (uint c0 = b0; c0 < b1; c0 += 32) {
    const uint t = c0 + lane;
    bool live = t < b1;
    if (live && t >= p0) live = ra.has_mask ? (bits >> (t - p0) & 1) != 0 : t <= p;
    float logit = -INFINITY;
    uint slot = 0;
    if (live) {
      slot = uint(table[t / a.block_tokens]) * a.block_tokens + t % a.block_tokens;
      device const half* kr = kp + (u64(slot) * a.kv_heads + kvh) * a.hd;
      float dot = 0;
      for (uint d = 0; d < a.hd; ++d) dot += float(qs[d]) * float(kr[d]);
      float b = 0;
      if (t <= p && p - t < ra.extent) {
        device const float* wr = (device const float*)(relw + u64(p - t) * ra.relw_rs);
        for (uint i = 0; i < ra.n_rel; ++i) b += rq[i] * wr[i];
      }
      logit = a.scale * dot + (b - sk);
    }
    const float nmx = max(mx, simd_max(logit));
    if (nmx == -INFINITY) continue;  // the same for the whole simdgroup: nothing live yet
    const float corr = exp(mx - nmx), e = live ? exp(logit - nmx) : 0.0f;
    sum = sum * corr + simd_sum(e);
    for (uint i = 0; i < ATTN_MAX_HEAD_DIM / 32; ++i) acc[i] *= corr;
    mx = nmx;
    for (uint m = 0; m < 32 && c0 + m < b1; ++m) {
      const float em = simd_shuffle(e, ushort(m));
      if (em == 0.0f) continue;  // the same for the whole simdgroup
      device const half* vr = vp + (u64(simd_shuffle(slot, ushort(m))) * a.kv_heads + kvh) * a.hd;
      for (uint i = 0; i < ATTN_MAX_HEAD_DIM / 32; ++i)
        if (lane + 32 * i < a.hd) acc[i] += em * float(vr[lane + 32 * i]);
    }
  }
  if (lane == 0) {
    dst[0] = mx;
    dst[1] = sum;
  }
  for (uint i = 0; i < ATTN_MAX_HEAD_DIM / 32; ++i)
    if (lane + 32 * i < a.hd) dst[2 + lane + 32 * i] = acc[i];
}

/// attn_rel, 2 of 2: vv_attn_split_reduce's merge of a (q head, row)'s splits, plus the one more logit 0 with no
/// value: it joins the max and adds exp(0 - max) to the sum. A row with no live position gives 0.
kernel void vv_attn_rel_reduce(device const float* partial [[buffer(0)]], device uchar* out [[buffer(1)]],
                               constant attn_args& a [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint hh = tg.x, r = tg.y;
  device const float* base = partial + (u64(r) * a.heads + hh) * a.splits * a.partial_stride;
  const bool live = lane < a.splits && base[lane * a.partial_stride + 1] != 0.0f;
  const bool live1 = lane + 32 < a.splits && base[(lane + 32) * a.partial_stride + 1] != 0.0f;
  const float m = live ? base[lane * a.partial_stride] : -INFINITY;
  const float z = live ? base[lane * a.partial_stride + 1] : 0.0f;
  const float m1 = live1 ? base[(lane + 32) * a.partial_stride] : -INFINITY;
  const float z1 = live1 ? base[(lane + 32) * a.partial_stride + 1] : 0.0f;
  const float mx = max(0.0f, simd_max(max(m, m1))), w = live ? exp(m - mx) : 0.0f, w1 = live1 ? exp(m1 - mx) : 0.0f;
  const float total = simd_sum(w * z + w1 * z1) + exp(-mx);
  float accum[ATTN_MAX_HEAD_DIM / 32] = {};
  for (uint split = 0; split < a.splits; ++split) {
    const float ws = simd_shuffle(split < 32 ? w : w1, ushort(split % 32));
    if (ws == 0.0f) continue;  // the same for every lane
    device const float* src = base + split * a.partial_stride + 2;
    for (uint i = 0; i < ATTN_MAX_HEAD_DIM / 32; ++i)
      if (lane + 32 * i < a.hd) accum[i] += ws * src[lane + 32 * i];
  }
  device half* output = (device half*)(out + u64(r) * a.out_rs) + hh * a.hd;
  for (uint i = 0; i < ATTN_MAX_HEAD_DIM / 32; ++i)
    if (lane + 32 * i < a.hd) output[lane + 32 * i] = half(accum[i] / total);
}
