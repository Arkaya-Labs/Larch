//! Qwen 3.5's multi-token-prediction block as an exact drafter, as llama.cpp runs it
//! (`src/models/qwen35moe.cpp`, `graph_mtp`; the driver `common/speculative.cpp`,
//! `common_speculative_impl_draft_mtp`): row p pairs token p with the target's final hidden state
//! (after the output norm) at p - 1, `eh_proj(concat(enorm(embed(token)), hnorm(hidden)))`, then one
//! full-attention decoder layer with its own KV, the shared head norm and the target's head. Its KV
//! position p holds that pair, so every committed position goes through [`MtpDrafter::observe`];
//! [`MtpDrafter::propose`] chains its guesses, each step feeding its own output hidden state.

use std::ops::Range;

use super::*;
use crate::Taps;
use larch_kernels::SampleParams;

/// The MTP block's weights: its projection and norms beside its decoder layer.
pub(super) struct Nextn {
    pub eh_proj: W,
    pub enorm: W,
    pub hnorm: W,
    pub head_norm: W,
    pub layer: Layer,
}

/// Positions per page of an [`MtpCache`].
const PAGE: u32 = 16;
/// The most guesses one [`MtpDrafter::propose`] makes.
pub const MTP_GUESSES: u32 = 16;

/// One sequence's MTP context: the block's K and V of every observed position, and the target's
/// final hidden state at the last one, which the next position pairs with.
pub struct MtpCache {
    k: Buffer,
    v: Buffer,
    /// f32 `[1, d]`.
    h: Buffer,
    len: u32,
}

impl MtpCache {
    /// Positions observed so far: the next observed position, and the anchor's.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// No position observed yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Forgets every position from `len` on.
    // ponytail: the kept hidden state stays the old last position's, so after any cut below it the
    // first position observed pairs with that one (a worse guess, never other output): edited
    // history, and any chat whose template re-renders earlier turns (think blocks dropped). Keep
    // one hidden state per position if multi-turn chats show in the hit rate.
    pub fn truncate(&mut self, len: u32) {
        self.len = self.len.min(len);
    }
}

/// The drafter beside a [`Transformer`] loaded by [`Transformer::load_mtp`]: the target holds the
/// block's weights and lends its activations, this holds the block's inputs and outputs.
pub struct MtpDrafter {
    /// i32: tokens `[rows]`, positions `[rows]`, hidden-state gather ids or decode lengths `[rows]`,
    /// then the chunk's start and length.
    aux: Buffer,
    /// f16 `[rows, 2 d]`: the normed embedding and hidden state.
    cat: Buffer,
    /// i32 `0..rows`.
    iota: Buffer,
    /// f32 `[1, d]` zeros: the hidden state before position 0.
    zero: Buffer,
    /// i32 `[1, pages]`: `0..pages`, every cache's own pages.
    table: Buffer,
    pages: u32,
    logits: Buffer,
    /// sample_prep's top-64 values, ids and LSE, one row a guess.
    top: Buffer,
    /// f32 `[rows, d]`: every batch row's final hidden state ([`MtpDrafter::taps`]).
    taps: Buffer,
    rows: i64,
    d: i64,
    vocab: i64,
}

impl Transformer {
    /// The device bytes of the MTP block's weights in `g` (layer `c.n_layers`, as [`Transformer::load_mtp`]
    /// takes them), with the f16 side copies of its gates; 0 without one.
    pub fn nextn_bytes(g: &Gguf, c: &ModelConfig) -> u64 {
        let at = format!("blk.{}.", c.n_layers);
        (g.tensors.iter())
            .filter(|t| t.name.starts_with(&at))
            .fold(0, |a, t| {
                let side = t.name.contains("ffn_gate_inp");
                let n = t.dims.iter().fold(1u64, |a, &d| a.saturating_mul(d));
                // bf16 (GGML type 30) has no DType, so the header leaves it at 0 bytes.
                let raw = if t.ggml_type == 30 {
                    n.saturating_mul(2)
                } else {
                    t.bytes
                };
                a.saturating_add(raw)
                    .saturating_add(if side { n.saturating_mul(2) } else { 0 })
            })
    }

    /// The MTP block on the step `b` describes, rows `[rows]` of `b.token_ids` paired with the
    /// hidden states the scratch's `t` holds (f32 `[rows, d]`), its K and V in `kv`; leaves the
    /// block's output in the scratch's `x`.
    fn nextn_block(
        &self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        cat: &Tensor,
        iota: &Tensor,
    ) -> Result<(), KError> {
        // Its rows are plain rows, whatever step ran last.
        self.check.set(false);
        let m = self.mtp.as_ref().ok_or(KError::InvalidArg)?;
        let rows = b.token_ids.shape()[0];
        let (d, eps) = (i64::from(self.cfg.d_model), self.cfg.rms_eps);
        if rows > self.max_rows || cat.shape() != [rows, 2 * d] {
            return Err(KError::InvalidArg);
        }
        let sc = &self.sc;
        let x = Tensor::new(&sc.x, 0, DType::F32, &[rows, d])?;
        let hin = Tensor::new(&sc.t, 0, DType::F32, &[rows, d])?;
        let h = Tensor::new(&sc.h, 0, DType::F16, &[rows, d])?;
        ops::embed(s, &self.embed_table()?, &b.token_ids, &x)?;
        ops::rmsnorm(s, &x, &self.t(m.enorm)?, eps, &cat.cols(0, d)?)?;
        ops::rmsnorm(s, &hin, &self.t(m.hnorm)?, eps, &cat.cols(d, d)?)?;
        self.linear(s, m.eh_proj, None, cat, &h)?;
        // f16 to the f32 residual stream: a row gather widens.
        ops::embed(s, &h, &iota.rows(0, rows)?, &x)?;
        let io = self.attn_io(rows)?;
        self.layer(s, self.layers.len(), &m.layer, b, None, kv, &io)
    }

    /// The logits `[1, vocab]` of the MTP block's one-row output in `x`, its head norm left in `h`.
    fn nextn_head(&self, s: &Stream, logits: &Tensor) -> Result<(), KError> {
        let m = self.mtp.as_ref().ok_or(KError::InvalidArg)?;
        let d = i64::from(self.cfg.d_model);
        let x = Tensor::new(&self.sc.x, 0, DType::F32, &[1, d])?;
        let h = Tensor::new(&self.sc.h, 0, DType::F16, &[1, d])?;
        ops::rmsnorm(s, &x, &self.t(m.head_norm)?, self.cfg.rms_eps, &h)?;
        self.linear(s, self.output, None, &h, logits)
    }
}

impl MtpDrafter {
    /// The drafter of `t`'s MTP block, for steps of up to `rows` rows and sequences of up to
    /// `max_context` positions; None when `t` was loaded without one.
    pub fn new(
        t: &Transformer,
        s: &Stream,
        rows: usize,
        max_context: u32,
    ) -> Result<Option<MtpDrafter>, KError> {
        if t.mtp.is_none() {
            return Ok(None);
        }
        let dev = s.device();
        let (d, vocab) = (i64::from(t.cfg.d_model), i64::from(t.cfg.vocab));
        let r = (rows as i64).max(i64::from(MTP_GUESSES));
        // Below 2^30 positions, the u32 position sums below cannot overflow.
        let pages = (max_context.checked_add(MTP_GUESSES))
            .filter(|&n| n < 1 << 30)
            .ok_or(KError::InvalidArg)?
            .div_ceil(PAGE);
        let g = i64::from(MTP_GUESSES);
        let buf = |t: DType, n: i64| Buffer::alloc(dev, t.bytes(n as u64).max(1));
        let m = MtpDrafter {
            aux: buf(DType::I32, 3 * r + 2)?,
            cat: buf(DType::F16, r * 2 * d)?,
            iota: buf(DType::I32, r)?,
            zero: buf(DType::F32, d)?,
            table: buf(DType::I32, i64::from(pages))?,
            pages,
            logits: buf(DType::F32, vocab)?,
            top: buf(DType::F32, g * 129)?,
            taps: buf(DType::F32, r * d)?,
            rows: r,
            d,
            vocab,
        };
        s.upload(&m.iota, 0, &(0..r as i32).collect::<Vec<_>>())?;
        s.upload(&m.zero, 0, &vec![0f32; d as usize])?;
        s.upload(&m.table, 0, &(0..pages as i32).collect::<Vec<_>>())?;
        Ok(Some(m))
    }

    /// Device bytes of [`MtpDrafter::new`] for a target of config `c`.
    pub fn scratch_bytes(c: &ModelConfig, rows: usize, max_context: u32) -> u64 {
        let (d, vocab) = (u64::from(c.d_model), u64::from(c.vocab));
        let r = (rows as u64).max(u64::from(MTP_GUESSES));
        let pages = u64::from(max_context.saturating_add(MTP_GUESSES).div_ceil(PAGE));
        4 * (3 * r + 2)
            + 4 * r * d
            + 4 * r
            + 4 * d
            + 4 * pages
            + 4 * vocab
            + 4 * 129 * u64::from(MTP_GUESSES)
            + 4 * r * d
    }

    /// Device bytes of one [`MtpCache`] of a target of config `c` at `max_context`.
    pub fn cache_bytes(c: &ModelConfig, max_context: u32) -> u64 {
        let page = u64::from(PAGE);
        let positions = (u64::from(max_context) + u64::from(MTP_GUESSES)).div_ceil(page) * page;
        let kvd = u64::from(c.n_kv_heads) * u64::from(c.head_dim);
        2 * DType::F16.bytes(positions * kvd) + 4 * u64::from(c.d_model)
    }

    /// An empty cache, sized as the drafter.
    pub fn cache(&self, t: &Transformer, s: &Stream) -> Result<MtpCache, KError> {
        let dev = s.device();
        let c = &t.cfg;
        let kvd = u64::from(c.n_kv_heads) * u64::from(c.head_dim);
        let bytes = DType::F16.bytes(u64::from(self.pages) * u64::from(PAGE) * kvd);
        Ok(MtpCache {
            k: Buffer::alloc(dev, bytes)?,
            v: Buffer::alloc(dev, bytes)?,
            h: Buffer::alloc(dev, 4 * self.d as u64)?,
            len: 0,
        })
    }

    /// The taps of a step of `rows` batch rows: every row's final hidden state.
    pub fn taps(&self, rows: usize) -> Result<Taps<'_>, KError> {
        let n = rows as i64;
        Ok(Taps {
            layers: &[],
            final_state: true,
            rows: Tensor::new(&self.iota, 0, DType::I32, &[n])?,
            out: Tensor::new(&self.taps, 0, DType::F32, &[n, self.d])?,
        })
    }

    fn pool<'a>(&self, t: &Transformer, c: &'a MtpCache) -> KvPool<'a> {
        KvPool {
            k: &c.k,
            v: &c.v,
            n_blocks: self.pages,
            block_tokens: PAGE,
            n_layers: 1,
            n_kv_heads: t.cfg.n_kv_heads,
            head_dim: t.cfg.head_dim,
            dtype: DType::F16,
        }
    }

    fn i32s(&self, at: i64, n: i64) -> Result<Tensor<'_>, KError> {
        Tensor::new(&self.aux, 4 * at as u64, DType::I32, &[n])
    }

    /// Appends rows `rows` of the step whose [`MtpDrafter::taps`] described `n` rows to `c`: their
    /// tokens `tokens`, at the positions from `c.len()` on, each a token the target ran and the
    /// sequence keeps. Row j pairs with the hidden state of the position before it: the cache's
    /// kept one for the first, the taps' row before it for the rest; the last row's becomes the kept
    /// one.
    // ponytail: one MTP pass per sequence a step (host experts sync each); batch the seats' rows
    // into one pass when a multi-chat profile of the hybrid MoE target shows these passes.
    pub fn observe(
        &self,
        t: &Transformer,
        s: &Stream,
        c: &mut MtpCache,
        n: usize,
        rows: Range<u32>,
        tokens: &[u32],
    ) -> Result<(), KError> {
        let k = rows.len() as i64;
        let fits = rows.start <= rows.end
            && rows.end as usize <= n
            && n as i64 <= self.rows
            && tokens.len() as i64 == k
            && tokens.iter().all(|&x| i64::from(x) < self.vocab)
            && c.len + k as u32 + MTP_GUESSES <= self.pages * PAGE;
        if !fits {
            return Err(KError::InvalidArg);
        }
        if k == 0 {
            return Ok(());
        }
        let (r, first) = (self.rows, rows.start as i32);
        let mut v: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
        v.resize(r as usize, 0);
        v.extend((0..r as i32).map(|j| c.len as i32 + j));
        // Rows 1.. read the taps' row before them; the last row's tap is kept.
        v.extend((first..first + k as i32).take(r as usize));
        v.resize(3 * r as usize, 0);
        v.extend([c.len as i32, k as i32]);
        s.upload(&self.aux, 0, &v)?;
        let d = self.d;
        let taps = Tensor::new(&self.taps, 0, DType::F32, &[n as i64, d])?;
        let hin = Tensor::new(&t.sc.t, 0, DType::F32, &[k, d])?;
        let kept = if c.len == 0 { &self.zero } else { &c.h };
        let one = Tensor::new(&self.iota, 0, DType::I32, &[1])?;
        ops::embed(
            s,
            &Tensor::new(kept, 0, DType::F32, &[1, d])?,
            &one,
            &hin.rows(0, 1)?,
        )?;
        if k > 1 {
            ops::embed(s, &taps, &self.i32s(2 * r, k - 1)?, &hin.rows(1, k - 1)?)?;
        }
        let last = self.i32s(2 * r + k - 1, 1)?;
        ops::embed(s, &taps, &last, &Tensor::new(&c.h, 0, DType::F32, &[1, d])?)?;
        let (tok, pos) = (self.i32s(0, k)?, self.i32s(r, k)?);
        let b = StepBatch {
            token_ids: tok,
            positions: pos,
            slot_ids: pos,
            decode_rows: 0,
            prefill_rows: k as u32,
            block_table: Tensor::new(&self.table, 0, DType::I32, &[1, i64::from(self.pages)])?,
            seq_lens: self.i32s(0, 0)?,
            prefill_q_start: self.i32s(3 * r, 1)?,
            prefill_q_len: self.i32s(3 * r + 1, 1)?,
            logits_rows: tok,
            attn_splits: 0,
            check: false,
        };
        let cat = Tensor::new(&self.cat, 0, DType::F16, &[k, 2 * d])?;
        let iota = Tensor::new(&self.iota, 0, DType::I32, &[r])?;
        t.nextn_block(s, &b, &self.pool(t, c), &cat, &iota)?;
        c.len += k as u32;
        Ok(())
    }

    /// Up to `n` greedy guesses (at most [`MTP_GUESSES`]) for the tokens after `anchor`, the token
    /// at position `c.len()`: each step one row, on the device, with one download at the end.
    /// Leaves `c`'s positions as they were (its K and V past them are the next observe's to
    /// overwrite).
    // ponytail: sequences propose one after another, a download each; chain the seats together
    // when a multi-chat profile of the hybrid MoE target shows the device idle between them.
    pub fn propose(
        &self,
        t: &Transformer,
        s: &Stream,
        c: &MtpCache,
        anchor: u32,
        n: u32,
    ) -> Result<Vec<u32>, KError> {
        let fits = (1..=MTP_GUESSES).contains(&n)
            && i64::from(anchor) < self.vocab
            && c.len + n <= self.pages * PAGE;
        if !fits {
            return Err(KError::InvalidArg);
        }
        let (r, d, g) = (self.rows, self.d, i64::from(MTP_GUESSES));
        let mut v = vec![0i32; r as usize];
        v[0] = anchor as i32;
        v.extend((0..r as i32).map(|j| c.len as i32 + j));
        v.extend((0..r as i32).map(|j| c.len as i32 + j + 1));
        v.resize(3 * r as usize + 2, 0);
        s.upload(&self.aux, 0, &v)?;
        let one = Tensor::new(&self.iota, 0, DType::I32, &[1])?;
        let iota = Tensor::new(&self.iota, 0, DType::I32, &[r])?;
        let cat = Tensor::new(&self.cat, 0, DType::F16, &[1, 2 * d])?;
        let logits = Tensor::new(&self.logits, 0, DType::F32, &[1, self.vocab])?;
        let ids = |i: i64| Tensor::new(&self.top, 4 * 64 * (g + i) as u64, DType::I32, &[1, 64]);
        let kept = if c.len == 0 { &self.zero } else { &c.h };
        let pool = self.pool(t, c);
        let p = SampleParams {
            temperature: 0.0,
            top_k: 1,
            repeat_penalty: 1.0,
            penalty_window: 0,
        };
        for i in 0..i64::from(n) {
            let hin = Tensor::new(&t.sc.t, 0, DType::F32, &[1, d])?;
            // The first step pairs with the target's hidden state, each later one with its own.
            match i {
                0 => ops::embed(s, &Tensor::new(kept, 0, DType::F32, &[1, d])?, &one, &hin)?,
                _ => ops::embed(
                    s,
                    &Tensor::new(&t.sc.h, 0, DType::F16, &[1, d])?,
                    &one,
                    &hin,
                )?,
            }
            let tok = match i {
                0 => self.i32s(0, 1)?,
                _ => ids(i - 1)?.reshape(&[64])?.rows(0, 1)?,
            };
            let b = StepBatch {
                token_ids: tok,
                positions: self.i32s(r + i, 1)?,
                slot_ids: self.i32s(r + i, 1)?,
                decode_rows: 1,
                prefill_rows: 0,
                block_table: Tensor::new(&self.table, 0, DType::I32, &[1, i64::from(self.pages)])?,
                seq_lens: self.i32s(2 * r + i, 1)?,
                prefill_q_start: self.i32s(0, 0)?,
                prefill_q_len: self.i32s(0, 0)?,
                logits_rows: tok,
                attn_splits: 0,
                check: false,
            };
            t.nextn_block(s, &b, &pool, &cat, &iota)?;
            t.nextn_head(s, &logits)?;
            let vals = Tensor::new(&self.top, 4 * 64 * i as u64, DType::F32, &[1, 64])?;
            let lse = Tensor::new(&self.top, 4 * (128 * g + i) as u64, DType::F32, &[1])?;
            ops::sample_prep(s, &logits, &[p], None, &vals, &ids(i)?, &lse)?;
        }
        let mut got = vec![0i32; 64 * n as usize];
        s.download(&self.top, 4 * 64 * g as u64, &mut got)?;
        Ok(got.chunks(64).map(|r| r[0] as u32).collect())
    }

    /// The f32 logits `[1, vocab]` of the last [`MtpDrafter::propose`]'s last guess (tests).
    pub fn last_logits(&self, s: &Stream) -> Result<Vec<f32>, KError> {
        let mut v = vec![0f32; self.vocab as usize];
        s.download(&self.logits, 0, &mut v)?;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Arch, tiny};
    use larch_kernels::CPU;

    /// Weight `w` as f32 rows: a row gather dequantizes any table.
    fn host(t: &Transformer, s: &Stream, w: W) -> Vec<f32> {
        let wt = t.t(w).unwrap();
        if w.ndim == 1 {
            let mut v = vec![0f32; w.shape[0] as usize];
            s.download(wt.buffer(), wt.offset(), &mut v).unwrap();
            return v;
        }
        let [r, c] = w.shape;
        let dev = s.device();
        let ids = Buffer::alloc(dev, 4 * r as u64).unwrap();
        s.upload(&ids, 0, &(0..r as i32).collect::<Vec<_>>())
            .unwrap();
        let out = Buffer::alloc(dev, 4 * (r * c) as u64).unwrap();
        let ids = Tensor::new(&ids, 0, DType::I32, &[r]).unwrap();
        ops::embed(
            s,
            &wt,
            &ids,
            &Tensor::new(&out, 0, DType::F32, &[r, c]).unwrap(),
        )
        .unwrap();
        let mut v = vec![0f32; (r * c) as usize];
        s.download(&out, 0, &mut v).unwrap();
        v
    }

    fn mv(w: &[f32], x: &[f32]) -> Vec<f32> {
        w.chunks_exact(x.len())
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    fn rms(x: &[f32], w: &[f32]) -> Vec<f32> {
        let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let k = 1.0 / (ms + 1e-6).sqrt();
        x.iter().zip(w).map(|(v, w)| v * k * w).collect()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn swiglu(g: &[f32], u: &[f32]) -> Vec<f32> {
        g.iter().zip(u).map(|(g, u)| g * sigmoid(*g) * u).collect()
    }

    /// The MTP block's logits for a fixed hidden state and token at position 0 (where attention
    /// returns V) match a direct f32 recomputation from the dequantized weights.
    #[test]
    fn mtp_logits_match_an_f32_recomputation() {
        let path =
            std::env::temp_dir().join(format!("larch-mtp-logits-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf_mtp(Arch::Qwen35Moe, 5)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let t = Transformer::load_mtp(&path, &dev, &s, 8, Placement::default()).unwrap();
        std::fs::remove_file(&path).unwrap();
        let dr = MtpDrafter::new(&t, &s, 8, 64).unwrap().unwrap();
        let c = dr.cache(&t, &s).unwrap();
        // The fit reserves what the drafter and a cache allocate.
        let bufs = [
            &dr.aux, &dr.cat, &dr.iota, &dr.zero, &dr.table, &dr.logits, &dr.top, &dr.taps,
        ];
        let got: u64 = bufs.iter().map(|b| b.bytes()).sum();
        assert_eq!(got, MtpDrafter::scratch_bytes(&t.cfg, 8, 64));
        let got = c.k.bytes() + c.v.bytes() + c.h.bytes();
        assert_eq!(got, MtpDrafter::cache_bytes(&t.cfg, 64));
        let d = tiny::D as usize;
        let mut seed = 0x1234_5678u32;
        let hidden: Vec<f32> = (0..d)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
            })
            .collect();
        // Position 0 pairs with `zero`: here the fixed hidden state.
        s.upload(&dr.zero, 0, &hidden).unwrap();
        let tok = 17u32;
        dr.propose(&t, &s, &c, tok, 1).unwrap();
        let got = dr.last_logits(&s).unwrap();

        let m = t.mtp.as_ref().unwrap();
        let w = |w: W| host(&t, &s, w);
        let e = &w(t.tok_embd.unwrap())[tok as usize * d..][..d];
        let mut cat = rms(e, &w(m.enorm));
        cat.extend(rms(&hidden, &w(m.hnorm)));
        let mut x = mv(&w(m.eh_proj), &cat);
        let (
            Mixer::Attn(a),
            Ffn::Moe {
                router,
                gate,
                up,
                down,
                shared: Some(sh),
                ..
            },
        ) = (&m.layer.mixer, m.layer.ffn)
        else {
            panic!("a gated attention layer and a shared-expert MoE");
        };
        let hn = rms(&x, &w(m.layer.attn_norm));
        let (q, v) = (mv(&w(a.wq), &hn), mv(&w(a.wv), &hn));
        let hd = tiny::HEAD_DIM as usize;
        let group = (tiny::HEADS / tiny::KV_HEADS) as usize;
        // One position: each head's attention output is its kv head's V, gated.
        let attn: Vec<f32> = (0..tiny::HEADS as usize)
            .flat_map(|h| {
                let gate = &q[h * 2 * hd + hd..][..hd];
                let v = &v[h / group * hd..][..hd];
                v.iter()
                    .zip(gate)
                    .map(|(v, g)| v * sigmoid(*g))
                    .collect::<Vec<_>>()
            })
            .collect();
        for (x, o) in x.iter_mut().zip(mv(&w(a.wo), &attn)) {
            *x += o;
        }
        let f = rms(&x, &w(m.layer.ffn_norm));
        let logits = mv(&w(router), &f);
        let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
        let p: Vec<f32> = logits.iter().map(|l| (l - mx).exp()).collect();
        let mut order: Vec<usize> = (0..p.len()).collect();
        order.sort_by(|&i, &j| p[j].total_cmp(&p[i]));
        let top = &order[..tiny::EXPERTS_USED as usize];
        let norm: f32 = top.iter().map(|&j| p[j]).sum();
        let (g, u, dn) = (w(gate), w(up), w(down));
        let ef = tiny::EXPERT_FFN as usize;
        for &j in top {
            let act = swiglu(
                &mv(&g[j * ef * d..][..ef * d], &f),
                &mv(&u[j * ef * d..][..ef * d], &f),
            );
            for (x, o) in x.iter_mut().zip(mv(&dn[j * d * ef..][..d * ef], &act)) {
                *x += p[j] / norm * o;
            }
        }
        let sg = sigmoid(mv(&w(sh.gate_inp), &f)[0]);
        let act = swiglu(&mv(&w(sh.gate), &f), &mv(&w(sh.up), &f));
        for (x, o) in x.iter_mut().zip(mv(&w(sh.down), &act)) {
            *x += sg * o;
        }
        let want = mv(&w(t.output), &rms(&x, &w(m.head_norm)));

        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(&want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err <= 0.02 * scale,
            "max error {err} against logits up to {scale}"
        );
        let argmax = |v: &[f32]| (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j]));
        assert_eq!(argmax(&got), argmax(&want));
    }

    /// One observe of a step's four rows leaves the cache as four one-row observes do, which pair
    /// each row with the kept hidden state only: the same guesses and logits.
    #[test]
    fn observing_rows_together_pairs_them_as_one_by_one() {
        let path = std::env::temp_dir().join(format!("larch-mtp-rows-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf_mtp(Arch::Qwen35Moe, 7)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let t = Transformer::load_mtp(&path, &dev, &s, 8, Placement::default()).unwrap();
        std::fs::remove_file(&path).unwrap();
        let dr = MtpDrafter::new(&t, &s, 8, 64).unwrap().unwrap();
        // Stand-in final states of a 4-row step.
        let mut seed = 0x9e37_79b9u32;
        let taps: Vec<f32> = (0..4 * dr.d)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect();
        s.upload(&dr.taps, 0, &taps).unwrap();
        let toks = [3, 41, 7, 19];
        let (mut a, mut b) = (dr.cache(&t, &s).unwrap(), dr.cache(&t, &s).unwrap());
        dr.observe(&t, &s, &mut a, 4, 0..4, &toks).unwrap();
        for j in 0..4u32 {
            let one = &toks[j as usize..j as usize + 1];
            dr.observe(&t, &s, &mut b, 4, j..j + 1, one).unwrap();
        }
        assert_eq!((a.len(), b.len()), (4, 4));
        let (ga, la) = (
            dr.propose(&t, &s, &a, 23, 2).unwrap(),
            dr.last_logits(&s).unwrap(),
        );
        let (gb, lb) = (
            dr.propose(&t, &s, &b, 23, 2).unwrap(),
            dr.last_logits(&s).unwrap(),
        );
        assert_eq!(ga, gb);
        let err = la
            .iter()
            .zip(&lb)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(err < 1e-3, "{err}");
    }

    /// The fit's figure holds the bf16 tensors' raw bytes, which the header's size leaves at 0
    /// (no [`DType`] has them), and the side copies of both routers.
    #[test]
    fn nextn_bytes_counts_the_bf16_routers() {
        let g = Gguf::parse(&tiny::gguf_mtp(Arch::Qwen35Moe, 2)).unwrap();
        let c = ModelConfig::from_gguf(&g).unwrap();
        let at = format!("blk.{}.", c.n_layers);
        let (mut want, mut bf16) = (0, 0);
        for t in g.tensors.iter().filter(|t| t.name.starts_with(&at)) {
            let n: u64 = t.dims.iter().product();
            bf16 += u32::from(t.ggml_type == 30);
            want += if t.ggml_type == 30 { 2 * n } else { t.bytes };
            want += if t.name.contains("ffn_gate_inp") {
                2 * n
            } else {
                0
            };
        }
        assert_eq!(bf16, 2);
        assert_eq!(Transformer::nextn_bytes(&g, &c), want);
    }

    /// With any layer's experts in host memory the MTP block's go there too: it drafts the same
    /// bytes with one host layer as with all of them, and the logits of the device's within the
    /// host path's f16 rounding.
    #[test]
    fn mtp_experts_go_to_the_host_with_the_first_host_layer() {
        let path = std::env::temp_dir().join(format!("larch-mtp-host-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf_mtp(Arch::Qwen35Moe, 6)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let logits = |layers| {
            let place = Placement {
                cpu_expert_layers: layers,
                ..Default::default()
            };
            let t = Transformer::load_mtp(&path, &dev, &s, 8, place).unwrap();
            let host = matches!(
                t.mtp.as_ref().unwrap().layer.ffn,
                Ffn::Moe { host: true, .. }
            );
            assert_eq!(host, layers > 0, "{layers}");
            let dr = MtpDrafter::new(&t, &s, 8, 64).unwrap().unwrap();
            let c = dr.cache(&t, &s).unwrap();
            dr.propose(&t, &s, &c, 9, 2).unwrap();
            dr.last_logits(&s).unwrap()
        };
        let (device, first, host) = (logits(0), logits(1), logits(tiny::LAYERS));
        std::fs::remove_file(&path).unwrap();
        assert_eq!(first, host);
        let err = device
            .iter()
            .zip(&host)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-2, "{err}");
    }

    /// A per-expert load carries the block: its experts stay on the device, where the fit charges
    /// them whole, and it drafts as the whole-device load does.
    #[test]
    fn a_planned_load_keeps_the_block_on_the_device() {
        let path = std::env::temp_dir().join(format!("larch-mtp-plan-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf_mtp(Arch::Qwen35Moe, 8)).unwrap();
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let logits = |t: &Transformer| {
            let dr = MtpDrafter::new(t, &s, 8, 64).unwrap().unwrap();
            let c = dr.cache(t, &s).unwrap();
            dr.propose(t, &s, &c, 9, 2).unwrap();
            dr.last_logits(&s).unwrap()
        };
        let whole = Transformer::load_mtp(&path, &dev, &s, 8, Placement::default()).unwrap();
        let plan = ExpertPlan {
            resident: vec![vec![0]; tiny::LAYERS as usize],
            dynamic: false,
        };
        let split = Transformer::load_cached_mtp(&path, &dev, &s, 8, &plan).unwrap();
        std::fs::remove_file(&path).unwrap();
        let block = split.mtp.as_ref().unwrap().layer.ffn;
        assert!(matches!(block, Ffn::Moe { host: false, .. }));
        assert_eq!(logits(&whole), logits(&split));
    }

    /// A recording load counts the trunk's routes only: with every layer's experts on the host,
    /// the block's (host too) have no row in the histogram.
    #[test]
    fn the_block_stays_out_of_the_host_layers_histogram() {
        let path = std::env::temp_dir().join(format!("larch-mtp-hist-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf_mtp(Arch::Qwen35Moe, 9)).unwrap();
        let hist = Histogram::path(&path);
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let place = Placement {
            cpu_expert_layers: tiny::LAYERS,
            ..Default::default()
        };
        let mut t = Transformer::load_mtp(&path, &dev, &s, 8, place).unwrap();
        let start = Histogram::uniform(0, tiny::LAYERS, tiny::EXPERTS);
        t.record_routes(&s, hist.clone(), start.clone()).unwrap();
        let dr = MtpDrafter::new(&t, &s, 8, 64).unwrap().unwrap();
        let c = dr.cache(&t, &s).unwrap();
        dr.propose(&t, &s, &c, 9, 2).unwrap();
        assert_eq!(t.cache.as_ref().unwrap().borrow().hist.counts, start.counts);
        drop(t);
        std::fs::remove_file(&path).unwrap();
        let _ = std::fs::remove_file(&hist);
    }

    /// With a resident layer the route log is live: the block's rows stay out of it, so a step of
    /// `max_rows` rows does not run past the trunk's log.
    #[test]
    fn the_block_stays_out_of_the_route_log() {
        let path = std::env::temp_dir().join(format!("larch-mtp-log-{}.gguf", std::process::id()));
        std::fs::write(&path, tiny::gguf_mtp(Arch::Qwen35Moe, 10)).unwrap();
        let hist = Histogram::path(&path);
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let mut resident = vec![(0..tiny::EXPERTS).collect::<Vec<u32>>(); tiny::LAYERS as usize];
        resident[0] = vec![0];
        let plan = ExpertPlan {
            resident,
            dynamic: false,
        };
        let mut t = Transformer::load_cached_mtp(&path, &dev, &s, 8, &plan).unwrap();
        let start = Histogram::uniform(0, tiny::LAYERS, tiny::EXPERTS);
        t.record_routes(&s, hist.clone(), start).unwrap();
        assert!(t.route_log.is_some());
        let dr = MtpDrafter::new(&t, &s, 8, 64).unwrap().unwrap();
        let mut c = dr.cache(&t, &s).unwrap();
        s.upload(&dr.taps, 0, &vec![0.5f32; 8 * dr.d as usize])
            .unwrap();
        dr.observe(&t, &s, &mut c, 8, 0..8, &[3; 8]).unwrap();
        assert_eq!(c.len(), 8);
        drop(t);
        std::fs::remove_file(&path).unwrap();
        let _ = std::fs::remove_file(&hist);
    }
}
