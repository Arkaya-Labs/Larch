//! The router-delta trainer on the model's device: [`super::Suffix`]'s forward and backward in f32
//! through the training ops (larch.h), reading the model's resident weights one projection, one
//! expert or one head chunk at a time into f32 scratch, so nothing is dequantized to host memory and
//! there is no second copy of the model. Runs where those ops do: CUDA, and the CPU devices, where
//! the tests hold it to the reference.
// ponytail: per-head GEMMs and one f32 weight copy at a time; batched GEMMs and a fused dequant GEMM
// when a 2 s job budget needs them on a real window.

use std::ops::Range;

use larch_kernels::{Buffer, DType, KError, Stream, Tensor, ops};

use super::{
    LearnError, Net, P, Route, Sel, Window, column_sums, objective_of, route, router_grad,
};
use crate::transformer::TrainLayer;
use crate::{Model, ModelConfig, MoeCfg, Transformer};

/// Head rows dequantized per chunk; also the longest window.
const CHUNK: i64 = 8192;

/// A qwen3moe model's layers `first..`, final norm and head, trained on the model's device.
pub struct DeviceSuffix<'m> {
    cfg: ModelConfig,
    moe: MoeCfg,
    first: u16,
    layers: Vec<TrainLayer<'m>>,
    out_norm: Tensor<'m>,
    head: Tensor<'m>,
    rope: ops::RopeParams<'m>,
    s: &'m Stream,
    /// i32 `0, 1, ..`: embed's row ids where it copies or dequantizes whole rows.
    iota: Buffer,
    /// f32 rows of one weight.
    w: Buffer,
}

/// A layer's forward activations on the device (f32), kept for the backward pass.
struct Cache {
    x_in: Buffer,
    q_raw: Buffer,
    k_raw: Buffer,
    /// After qk_norm and rope.
    q: Buffer,
    k: Buffer,
    v: Buffer,
    x_mid: Buffer,
    routes: Vec<Route>,
    /// Every (token, slot), grouped by expert; `groups`: (expert, first, count) of each group.
    slots: Vec<(usize, usize)>,
    groups: Vec<(i64, i64, i64)>,
    /// `[slots, expert ffn]` in `slots` order.
    g: Buffer,
    u: Buffer,
    /// Host `[T, used, d]`: each chosen expert's output.
    y: Vec<f32>,
}

struct Pass {
    /// f32 `[head rows, vocab]`, on the host.
    logits: Vec<f32>,
    /// The residual at the head rows, `[head rows, d]`.
    xr: Buffer,
    sel: Vec<Sel>,
    caches: Vec<Cache>,
}

fn f32t<'b>(b: &'b Buffer, shape: &[i64]) -> Result<Tensor<'b>, KError> {
    Tensor::new(b, 0, DType::F32, shape)
}

/// `[T, slots]` with slot `s`'s `value(s)` at its token's row: the experts' combine as a GEMM.
fn one_hot(t_n: usize, slots: &[(usize, usize)], value: impl Fn(usize) -> f32) -> Vec<f32> {
    let mut m = vec![0f32; t_n * slots.len()];
    for (s, &(t, _)) in slots.iter().enumerate() {
        m[t * slots.len() + s] = value(s);
    }
    m
}

impl<'m> DeviceSuffix<'m> {
    /// Layers `first..` of `m`, run on `s` (a stream of `m`'s device). Invalid unless `m` is qwen3moe
    /// with those layers' routed experts on the device (see [`crate::Placement`]).
    pub fn new(
        m: &'m Transformer,
        s: &'m Stream,
        first: u16,
    ) -> Result<DeviceSuffix<'m>, LearnError> {
        let cfg = m.config().clone();
        if u32::from(first) >= cfg.n_layers {
            return Err(LearnError::Invalid("first layer"));
        }
        let (moe, layers, out_norm, head) = m
            .train_view(first)?
            .ok_or(LearnError::Invalid("arch, or experts in host memory"))?;
        let (d, kvd) = (
            i64::from(cfg.d_model),
            i64::from(cfg.n_kv_heads * cfg.head_dim),
        );
        let (qd, ef, ne) = (
            i64::from(cfg.n_heads * cfg.head_dim),
            i64::from(moe.ffn),
            i64::from(moe.experts),
        );
        let widest = [qd, kvd, ef, ne, CHUNK.min(i64::from(cfg.vocab))]
            .iter()
            .map(|r| r * d)
            .max()
            .unwrap_or(0);
        let iota = Buffer::alloc(s.device(), DType::I32.bytes(CHUNK.max(qd) as u64))?;
        s.upload(&iota, 0, &(0..CHUNK.max(qd) as i32).collect::<Vec<_>>())?;
        Ok(DeviceSuffix {
            moe,
            first,
            layers,
            out_norm,
            head,
            rope: m.rope_params()?,
            s,
            iota,
            w: Buffer::alloc(s.device(), DType::F32.bytes(widest as u64))?,
            cfg,
        })
    }

    fn buf(&self, n: i64) -> Result<Buffer, KError> {
        Buffer::alloc(self.s.device(), DType::F32.bytes(n.max(1) as u64))
    }

    fn up(&self, v: &[f32]) -> Result<Buffer, KError> {
        let b = self.buf(v.len() as i64)?;
        self.s.upload(&b, 0, v)?;
        Ok(b)
    }

    fn up_i32(&self, v: &[i32]) -> Result<Buffer, KError> {
        let b = self.buf(v.len() as i64)?;
        self.s.upload(&b, 0, v)?;
        Ok(b)
    }

    fn ids(&self, n: i64) -> Result<Tensor<'_>, KError> {
        Tensor::new(&self.iota, 0, DType::I32, &[n])
    }

    /// Rows `r` of weight `w`, dequantized into the f32 scratch.
    fn deq(&self, w: &Tensor, r: Range<i64>) -> Result<Tensor<'_>, KError> {
        let n = r.end - r.start;
        let out = f32t(&self.w, &[n, w.shape()[1]])?;
        ops::embed(self.s, &w.rows(r.start, n)?, &self.ids(n)?, &out)?;
        Ok(out)
    }

    /// A copy of f32 `x` `[rows, cols]`.
    fn copy(&self, x: &Tensor) -> Result<Buffer, KError> {
        let b = self.buf(x.shape().iter().product())?;
        ops::embed(self.s, x, &self.ids(x.shape()[0])?, &f32t(&b, x.shape())?)?;
        Ok(b)
    }

    /// `c = op(a) op(b) + beta c`.
    fn mm(
        &self,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        ta: bool,
        tb: bool,
        beta: f32,
    ) -> Result<(), KError> {
        ops::gemm_ex(self.s, a, b, c, ta, tb, 1.0, beta)
    }

    /// The forward pass with `deltas` (by suffix layer), the head on `head_rows`.
    fn forward(
        &self,
        win: &Window,
        deltas: &[Option<P>],
        head_rows: &[usize],
        keep: bool,
    ) -> Result<Pass, LearnError> {
        let c = &self.cfg;
        let (s, t_n) = (self.s, win.tokens.len());
        if t_n as i64 > CHUNK {
            return Err(LearnError::Invalid("window length"));
        }
        let tn = t_n as i64;
        let (d, hd, nh, nkv) = (
            i64::from(c.d_model),
            i64::from(c.head_dim),
            i64::from(c.n_heads),
            i64::from(c.n_kv_heads),
        );
        let (qd, kvd, eps) = (nh * hd, nkv * hd, c.rms_eps);
        let (ne, used, ef) = (
            self.moe.experts as usize,
            self.moe.used as usize,
            i64::from(self.moe.ffn),
        );
        let scale = 1.0 / (hd as f32).sqrt();
        let pos = self.up_i32(win.positions)?;
        let pos = Tensor::new(&pos, 0, DType::I32, &[tn])?;
        let x = self.up(win.residual)?;
        let xt = f32t(&x, &[tn, d])?;
        let h = self.buf(tn * d)?;
        let ht = f32t(&h, &[tn, d])?;
        let attn = self.buf(tn * qd)?;
        let at = f32t(&attn, &[tn, qd])?;
        let (mut sel, mut caches) = (Vec::new(), Vec::new());
        for (li, l) in self.layers.iter().enumerate() {
            let x_in = self.copy(&xt)?;
            ops::rmsnorm_f32(s, &xt, &l.attn_norm, eps, &ht)?;
            let [q_raw, q, k_raw, k, v] = [qd, qd, kvd, kvd, kvd].map(|w| self.buf(tn * w));
            let (q_raw, q, k_raw, k, v) = (q_raw?, q?, k_raw?, k?, v?);
            let (qt, kt, vt) = (
                f32t(&q, &[tn, qd])?,
                f32t(&k, &[tn, kvd])?,
                f32t(&v, &[tn, kvd])?,
            );
            for (w, out, rows, norm, raw, heads) in [
                (&l.wq, &q, qd, &l.q_norm, &q_raw, nh),
                (&l.wk, &k, kvd, &l.k_norm, &k_raw, nkv),
            ] {
                let rt = f32t(raw, &[tn, rows])?;
                self.mm(&ht, &self.deq(w, 0..rows)?, &rt, false, true, 0.0)?;
                let o = f32t(out, &[tn * heads, hd])?;
                ops::rmsnorm_f32(s, &rt.reshape(&[tn * heads, hd])?, norm, eps, &o)?;
                ops::rope_f32(s, &o.reshape(&[tn, heads, hd])?, &pos, &self.rope)?;
            }
            self.mm(&ht, &self.deq(&l.wv, 0..kvd)?, &vt, false, true, 0.0)?;
            // Per head: P = causal softmax(q kᵀ scale), then P v (GQA: head h reads kv head h / group).
            // One head's P at a time, recomputed in the backward: every head's is 512 MB a layer at
            // a served row's 2k tokens.
            let p = self.buf(tn * tn)?;
            let pt = f32t(&p, &[tn, tn])?;
            let kv = |h: i64| h / (nh / nkv) * hd;
            for h in 0..nh {
                let (qh, kh) = (qt.cols(h * hd, hd)?, kt.cols(kv(h), hd)?);
                self.mm(&qh, &kh, &pt, false, true, 0.0)?;
                ops::softmax(s, &pt, scale, true, &pt)?;
                self.mm(
                    &pt,
                    &vt.cols(kv(h), hd)?,
                    &at.cols(h * hd, hd)?,
                    false,
                    false,
                    0.0,
                )?;
            }
            self.mm(&at, &self.deq(&l.wo, 0..d)?, &xt, false, true, 1.0)?;
            let x_mid = self.copy(&xt)?;
            ops::rmsnorm_f32(s, &xt, &l.ffn_norm, eps, &ht)?;
            let lb = self.buf(tn * ne as i64)?;
            let router = self.deq(&l.router, 0..ne as i64)?;
            self.mm(
                &ht,
                &router,
                &f32t(&lb, &[tn, ne as i64])?,
                false,
                true,
                0.0,
            )?;
            let mut lg = vec![0f32; t_n * ne];
            s.download(&lb, 0, &mut lg)?;
            let delta = deltas[li];
            let routes = lg
                .chunks_exact(ne)
                .map(|r| {
                    let r: Vec<f32> = (0..ne)
                        .map(|e| r[e] + delta.and_then(|p| p.router()).map_or(0.0, |dl| dl[e]))
                        .collect();
                    route(&r, used)
                })
                .collect::<Result<Vec<_>, _>>()?;
            // Experts ascending, each one's (token, slot)s in order; one GEMM per weight per expert.
            let (mut slots, mut groups) = (Vec::with_capacity(t_n * used), Vec::new());
            for e in 0..ne {
                let at = slots.len();
                slots.extend(
                    (0..t_n)
                        .flat_map(|t| (0..used).map(move |j| (t, j)))
                        .filter(|&(t, j)| routes[t].ids[j] == e),
                );
                if slots.len() > at {
                    groups.push((e as i64, at as i64, (slots.len() - at) as i64));
                }
            }
            let ns = slots.len() as i64;
            let ids: Vec<i32> = slots.iter().map(|&(t, _)| t as i32).collect();
            let ids = self.up_i32(&ids)?;
            let [g, u] = [0; 2].map(|_| self.buf(ns * ef));
            let (g, u) = (g?, u?);
            let (gt, ut) = (f32t(&g, &[ns, ef])?, f32t(&u, &[ns, ef])?);
            // A group's inputs, activation and output live in scratch the size of the largest group,
            // and it joins the residual on its own: the whole `[slots, d]` ones (8 per token) do not
            // fit beside a resident model at a served row's 2k tokens, and WSL spills what does not.
            let widest = groups.iter().map(|g| g.2).max().unwrap_or(1);
            let (xe, act, y) = (
                self.buf(widest * d)?,
                self.buf(widest * ef)?,
                self.buf(widest * d)?,
            );
            let du = d as usize;
            let mut yh = vec![0f32; if keep { t_n * used * du } else { 0 }];
            let mut yg = Vec::new();
            for &(e, at, n) in &groups {
                let (xs, a_s, ys) = (
                    f32t(&xe, &[n, d])?,
                    f32t(&act, &[n, ef])?,
                    f32t(&y, &[n, d])?,
                );
                let gid = Tensor::new(&ids, DType::I32.bytes(at as u64), DType::I32, &[n])?;
                ops::embed(s, &ht, &gid, &xs)?;
                let (gs, us) = (gt.rows(at, n)?, ut.rows(at, n)?);
                self.mm(
                    &xs,
                    &self.deq(&l.gate, e * ef..(e + 1) * ef)?,
                    &gs,
                    false,
                    true,
                    0.0,
                )?;
                self.mm(
                    &xs,
                    &self.deq(&l.up, e * ef..(e + 1) * ef)?,
                    &us,
                    false,
                    true,
                    0.0,
                )?;
                ops::silu_mul_f32(s, &gs, &us, &a_s)?;
                let down = self.deq(&l.down, e * d..(e + 1) * d)?;
                self.mm(&a_s, &down, &ys, false, true, 0.0)?;
                // The adapter joins the expert's output before the routing weights combine them.
                if let Some((r, (_, b, a))) =
                    delta.and_then(|p| Some((p.rank as i64, p.adapter(e as usize, d as usize)?)))
                {
                    let (bb, ab, hb) = (self.up(b)?, self.up(a)?, self.buf(n * r)?);
                    let hs = f32t(&hb, &[n, r])?;
                    self.mm(&a_s, &f32t(&ab, &[r, ef])?, &hs, false, true, 0.0)?;
                    self.mm(&hs, &f32t(&bb, &[d, r])?, &ys, false, true, 1.0)?;
                }
                let group = &slots[at as usize..(at + n) as usize];
                // ponytail: a buffer and an upload per group (and a download with `keep`); one per
                // layer, reused, when the ~9 s served-row step is what a profile shows.
                let cw = self.up(&one_hot(t_n, group, |i| routes[group[i].0].w[group[i].1]))?;
                self.mm(&f32t(&cw, &[tn, n])?, &ys, &xt, false, false, 1.0)?;
                if keep {
                    yg.resize(n as usize * du, 0.0);
                    s.download(&y, 0, &mut yg)?;
                    for (i, &(t, j)) in group.iter().enumerate() {
                        yh[(t * used + j) * du..][..du].copy_from_slice(&yg[i * du..(i + 1) * du]);
                    }
                }
            }
            if delta.is_some() {
                sel.push(Sel::of(&routes, head_rows));
            }
            if keep {
                caches.push(Cache {
                    x_in,
                    q_raw,
                    k_raw,
                    q,
                    k,
                    v,
                    x_mid,
                    routes,
                    slots,
                    groups,
                    g,
                    u,
                    y: yh,
                });
            }
        }
        let n = head_rows.len() as i64;
        let rows: Vec<i32> = head_rows.iter().map(|&t| t as i32).collect();
        let rows = self.up_i32(&rows)?;
        let xr = self.buf(n * d)?;
        let xrt = f32t(&xr, &[n, d])?;
        ops::embed(s, &xt, &Tensor::new(&rows, 0, DType::I32, &[n])?, &xrt)?;
        let hf = self.buf(n * d)?;
        let hft = f32t(&hf, &[n, d])?;
        ops::rmsnorm_f32(s, &xrt, &self.out_norm, eps, &hft)?;
        let v = i64::from(c.vocab);
        let lc = self.buf(n * CHUNK.min(v))?;
        let mut logits = vec![0f32; (n * v) as usize];
        let mut part = Vec::new();
        for c0 in (0..v).step_by(CHUNK as usize) {
            let cn = CHUNK.min(v - c0);
            let lt = f32t(&lc, &[n, cn])?;
            self.mm(
                &hft,
                &self.deq(&self.head, c0..c0 + cn)?,
                &lt,
                false,
                true,
                0.0,
            )?;
            part.resize((n * cn) as usize, 0.0);
            s.download(&lc, 0, &mut part)?;
            for (r, p) in part.chunks_exact(cn as usize).enumerate() {
                logits[(r as i64 * v + c0) as usize..][..cn as usize].copy_from_slice(p);
            }
        }
        Ok(Pass {
            logits,
            xr,
            sel,
            caches,
        })
    }

    /// dL/ddelta for each adapted layer (suffix-layer order), from the head's dlogits down to the
    /// lowest adapted layer's router.
    fn backward(
        &self,
        win: &Window,
        deltas: &[Option<P>],
        head_rows: &[usize],
        pass: &Pass,
        dlogits: &[f32],
    ) -> Result<Vec<Vec<f32>>, LearnError> {
        let c = &self.cfg;
        let (s, t_n) = (self.s, win.tokens.len());
        let tn = t_n as i64;
        let (d, hd, nh, nkv) = (
            i64::from(c.d_model),
            i64::from(c.head_dim),
            i64::from(c.n_heads),
            i64::from(c.n_kv_heads),
        );
        let (qd, kvd, eps) = (nh * hd, nkv * hd, c.rms_eps);
        let (ne, ef) = (self.moe.experts as usize, i64::from(self.moe.ffn));
        let scale = 1.0 / (hd as f32).sqrt();
        let lowest = deltas.iter().position(Option::is_some).unwrap_or(0);
        // dhf = dlogits · head, a chunk of the vocabulary at a time.
        let (n, v) = (head_rows.len() as i64, i64::from(c.vocab));
        let dhf = self.buf(n * d)?;
        let dhft = f32t(&dhf, &[n, d])?;
        let dl = self.buf(n * CHUNK.min(v))?;
        for c0 in (0..v).step_by(CHUNK as usize) {
            let cn = CHUNK.min(v - c0);
            let part: Vec<f32> = dlogits
                .chunks_exact(v as usize)
                .flat_map(|r| r[c0 as usize..(c0 + cn) as usize].iter().copied())
                .collect();
            s.upload(&dl, 0, &part)?;
            let head = self.deq(&self.head, c0..c0 + cn)?;
            let beta = if c0 == 0 { 0.0 } else { 1.0 };
            self.mm(&f32t(&dl, &[n, cn])?, &head, &dhft, false, false, beta)?;
        }
        let dxr = self.buf(n * d)?;
        let xrt = f32t(&pass.xr, &[n, d])?;
        ops::rmsnorm_backward(
            s,
            &xrt,
            &self.out_norm,
            &dhft,
            eps,
            &f32t(&dxr, &[n, d])?,
            false,
        )?;
        let rows: Vec<(usize, usize)> = head_rows.iter().map(|&t| (t, 0)).collect();
        let scatter = self.up(&one_hot(t_n, &rows, |_| 1.0))?;
        let dx = self.buf(tn * d)?;
        let dxt = f32t(&dx, &[tn, d])?;
        self.mm(
            &f32t(&scatter, &[tn, n])?,
            &f32t(&dxr, &[n, d])?,
            &dxt,
            false,
            false,
            0.0,
        )?;
        let neg: Vec<i32> = win.positions.iter().map(|&p| -p).collect();
        let neg = self.up_i32(&neg)?;
        let neg = Tensor::new(&neg, 0, DType::I32, &[tn])?;
        let mut dxh = vec![0f32; t_n * d as usize];
        let mut grads = Vec::new();
        for li in (lowest..self.layers.len()).rev() {
            let (l, k) = (&self.layers[li], &pass.caches[li]);
            s.download(&dx, 0, &mut dxh)?;
            let dlog = router_grad(&k.routes, &k.y, &dxh, d as usize, ne);
            let ad = deltas[li].filter(|p| !p.ids.is_empty());
            if deltas[li].is_some() && ad.is_none() {
                grads.push(column_sums(&dlog, ne));
            }
            if li == lowest && ad.is_none() {
                break;
            }
            let mut ga = vec![0f32; ad.map_or(0, |p| p.v.len())];
            // dh2: through the router, then through each chosen expert.
            let dlb = self.up(&dlog)?;
            let dh2 = self.buf(tn * d)?;
            let dh2t = f32t(&dh2, &[tn, d])?;
            let router = self.deq(&l.router, 0..ne as i64)?;
            self.mm(
                &f32t(&dlb, &[tn, ne as i64])?,
                &router,
                &dh2t,
                false,
                false,
                0.0,
            )?;
            let ns = k.slots.len() as i64;
            // Per group, in scratch the size of the largest (as in the forward): dy = Cᵀ dx through
            // the combine, the expert's backward, and its dx joining dh2.
            let widest = k.groups.iter().map(|g| g.2).max().unwrap_or(1);
            let [dyb, dxb] = [0; 2].map(|_| self.buf(widest * d));
            let [dab, dgb, dub] = [0; 3].map(|_| self.buf(widest * ef));
            let (dyb, dxb, dab, dgb, dub) = (dyb?, dxb?, dab?, dgb?, dub?);
            let (gt, ut) = (f32t(&k.g, &[ns, ef])?, f32t(&k.u, &[ns, ef])?);
            for &(e, at, n) in &k.groups {
                let (dys, dxs) = (f32t(&dyb, &[n, d])?, f32t(&dxb, &[n, d])?);
                let (das, dgs, dus) = (
                    f32t(&dab, &[n, ef])?,
                    f32t(&dgb, &[n, ef])?,
                    f32t(&dub, &[n, ef])?,
                );
                let group = &k.slots[at as usize..(at + n) as usize];
                let cw = self.up(&one_hot(t_n, group, |i| k.routes[group[i].0].w[group[i].1]))?;
                self.mm(&f32t(&cw, &[tn, n])?, &dxt, &dys, true, false, 0.0)?;
                let down = self.deq(&l.down, e * d..(e + 1) * d)?;
                self.mm(&dys, &down, &das, false, false, 0.0)?;
                if let Some((r, (range, b, a))) =
                    ad.and_then(|p| Some((p.rank as i64, p.adapter(e as usize, d as usize)?)))
                {
                    // ha = ae Aᵀ, hb = dy B; dB = dyᵀ ha, dA = hbᵀ ae, dae += hb A.
                    let ae = self.buf(n * ef)?;
                    let aes = f32t(&ae, &[n, ef])?;
                    ops::silu_mul_f32(s, &gt.rows(at, n)?, &ut.rows(at, n)?, &aes)?;
                    let [bb, ab, hab, hbb, gb, gab] =
                        [d * r, r * ef, n * r, n * r, d * r, r * ef].map(|len| self.buf(len));
                    let (bb, ab, hab, hbb, gb, gab) = (bb?, ab?, hab?, hbb?, gb?, gab?);
                    s.upload(&bb, 0, b)?;
                    s.upload(&ab, 0, a)?;
                    let (bt, at_, ha, hb) = (
                        f32t(&bb, &[d, r])?,
                        f32t(&ab, &[r, ef])?,
                        f32t(&hab, &[n, r])?,
                        f32t(&hbb, &[n, r])?,
                    );
                    self.mm(&aes, &at_, &ha, false, true, 0.0)?;
                    self.mm(&dys, &bt, &hb, false, false, 0.0)?;
                    self.mm(&hb, &at_, &das, false, false, 1.0)?;
                    self.mm(&dys, &ha, &f32t(&gb, &[d, r])?, true, false, 0.0)?;
                    self.mm(&hb, &aes, &f32t(&gab, &[r, ef])?, true, false, 0.0)?;
                    let (gd, gf) = ga[range].split_at_mut((d * r) as usize);
                    s.download(&gb, 0, gd)?;
                    s.download(&gab, 0, gf)?;
                }
                ops::silu_mul_backward(s, &gt.rows(at, n)?, &ut.rows(at, n)?, &das, &dgs, &dus)?;
                self.mm(
                    &dgs,
                    &self.deq(&l.gate, e * ef..(e + 1) * ef)?,
                    &dxs,
                    false,
                    false,
                    0.0,
                )?;
                self.mm(
                    &dus,
                    &self.deq(&l.up, e * ef..(e + 1) * ef)?,
                    &dxs,
                    false,
                    false,
                    1.0,
                )?;
                let sw = self.up(&one_hot(t_n, group, |_| 1.0))?;
                self.mm(&f32t(&sw, &[tn, n])?, &dxs, &dh2t, false, false, 1.0)?;
            }
            if ad.is_some() {
                grads.push(ga);
            }
            if li == lowest {
                break;
            }
            let x_mid = f32t(&k.x_mid, &[tn, d])?;
            ops::rmsnorm_backward(s, &x_mid, &l.ffn_norm, &dh2t, eps, &dxt, true)?;
            // Attention: per head, dV = Pᵀ dO, dP = dO Vᵀ, dS = softmax_backward, dQ = dS K, dK = dSᵀ Q;
            // a GQA group's heads sum into their kv head.
            let dattn = self.buf(tn * qd)?;
            let dat = f32t(&dattn, &[tn, qd])?;
            self.mm(&dxt, &self.deq(&l.wo, 0..d)?, &dat, false, false, 0.0)?;
            let [dq, dk, dv] = [qd, kvd, kvd].map(|w| self.buf(tn * w));
            let (dq, dk, dv, dp) = (dq?, dk?, dv?, self.buf(tn * tn)?);
            let (dqt, dkt, dvt, dpt) = (
                f32t(&dq, &[tn, qd])?,
                f32t(&dk, &[tn, kvd])?,
                f32t(&dv, &[tn, kvd])?,
                f32t(&dp, &[tn, tn])?,
            );
            let (qt, kt, vt) = (
                f32t(&k.q, &[tn, qd])?,
                f32t(&k.k, &[tn, kvd])?,
                f32t(&k.v, &[tn, kvd])?,
            );
            let p = self.buf(tn * tn)?;
            let ph = f32t(&p, &[tn, tn])?;
            let group = nh / nkv;
            for h in 0..nh {
                let kh = h / group * hd;
                let beta = if h % group == 0 { 0.0 } else { 1.0 };
                // The forward's P for this head, recomputed (it keeps none).
                self.mm(
                    &qt.cols(h * hd, hd)?,
                    &kt.cols(kh, hd)?,
                    &ph,
                    false,
                    true,
                    0.0,
                )?;
                ops::softmax(s, &ph, scale, true, &ph)?;
                let doh = dat.cols(h * hd, hd)?;
                self.mm(&ph, &doh, &dvt.cols(kh, hd)?, true, false, beta)?;
                self.mm(&doh, &vt.cols(kh, hd)?, &dpt, false, true, 0.0)?;
                ops::softmax_backward(s, &ph, &dpt, scale, &dpt)?;
                self.mm(
                    &dpt,
                    &kt.cols(kh, hd)?,
                    &dqt.cols(h * hd, hd)?,
                    false,
                    false,
                    0.0,
                )?;
                self.mm(
                    &dpt,
                    &qt.cols(h * hd, hd)?,
                    &dkt.cols(kh, hd)?,
                    true,
                    false,
                    beta,
                )?;
            }
            // rope backward (the inverse rotation), then qk_norm's, then the projections'.
            let dh1 = self.buf(tn * d)?;
            let dh1t = f32t(&dh1, &[tn, d])?;
            for (i, (g, raw, norm, w, heads)) in [
                (&dq, &k.q_raw, &l.q_norm, &l.wq, nh),
                (&dk, &k.k_raw, &l.k_norm, &l.wk, nkv),
            ]
            .into_iter()
            .enumerate()
            {
                let gt = f32t(g, &[tn * heads, hd])?;
                ops::rope_f32(s, &gt.reshape(&[tn, heads, hd])?, &neg, &self.rope)?;
                let draw = self.buf(tn * heads * hd)?;
                let drt = f32t(&draw, &[tn * heads, hd])?;
                let rt = f32t(raw, &[tn * heads, hd])?;
                ops::rmsnorm_backward(s, &rt, norm, &gt, eps, &drt, false)?;
                let wt = self.deq(w, 0..heads * hd)?;
                let beta = if i == 0 { 0.0 } else { 1.0 };
                self.mm(
                    &drt.reshape(&[tn, heads * hd])?,
                    &wt,
                    &dh1t,
                    false,
                    false,
                    beta,
                )?;
            }
            self.mm(&dvt, &self.deq(&l.wv, 0..kvd)?, &dh1t, false, false, 1.0)?;
            let x_in = f32t(&k.x_in, &[tn, d])?;
            ops::rmsnorm_backward(s, &x_in, &l.attn_norm, &dh1t, eps, &dxt, true)?;
        }
        grads.reverse();
        Ok(grads)
    }
}

impl Net for DeviceSuffix<'_> {
    fn cfg(&self) -> &ModelConfig {
        &self.cfg
    }

    fn moe(&self) -> MoeCfg {
        self.moe
    }

    fn first(&self) -> u16 {
        self.first
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn head_logits(
        &self,
        win: &Window,
        d: &[Option<P>],
        rows: &[usize],
    ) -> Result<Vec<f32>, LearnError> {
        Ok(self.forward(win, d, rows, false)?.logits)
    }

    fn objective(
        &self,
        win: &Window,
        d: &[Option<P>],
        rows: &[usize],
        grad: bool,
        base: Option<&[f32]>,
    ) -> Result<(f64, Option<Vec<Vec<f32>>>, Vec<Sel>), LearnError> {
        let pass = self.forward(win, d, rows, grad)?;
        let (loss, dlogits) =
            objective_of(&pass.logits, self.cfg.vocab as usize, win, rows, grad, base)?;
        if !grad {
            return Ok((loss, None, pass.sel));
        }
        let g = self.backward(win, d, rows, &pass, &dlogits)?;
        Ok((loss, Some(g), pass.sel))
    }
}
