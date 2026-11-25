//! `FakeModel`: the engine's test double. Its next token is always (input token + 1) % vocab, so a
//! stream from a prompt ending in t reads t+1, t+2, ... A tapped row whose input token is t reads
//! `[t, l, 2, 3, ..., d_model - 1]` after layer l, and the final state takes l = n_layers. It reads
//! its inputs back to the host and writes no KV: tests only.

use larch_kernels::{DType, KError, KvPool, Stream, Tensor};

use crate::{Arch, Model, ModelConfig, RopeCfg, StepBatch, Taps};

/// A model whose next token is always input + 1 (see the module doc).
pub struct FakeModel {
    cfg: ModelConfig,
}

impl FakeModel {
    /// A fake with `vocab` tokens and one layer.
    pub fn new(vocab: u32) -> FakeModel {
        FakeModel::with_layers(vocab, 1)
    }

    /// A fake with `vocab` tokens and `n_layers` layers to tap.
    pub fn with_layers(vocab: u32, n_layers: u32) -> FakeModel {
        let rope = RopeCfg {
            n_dims: 16,
            neox: false,
            freq_base: 10000.0,
            freq_scale: 1.0,
            yarn: false,
            n_ctx_orig: 4096,
            sections: None,
        };
        FakeModel {
            cfg: ModelConfig {
                arch: Arch::Llama,
                n_layers,
                kv_layers: n_layers,
                d_model: 16,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 16,
                ffn: 16,
                vocab,
                ctx_train: 4096,
                rope,
                rms_eps: 1e-5,
                qkv_bias: false,
                qk_norm: false,
                tie_embeddings: true,
            },
        }
    }
}

fn read(s: &Stream, t: &Tensor) -> Result<Vec<i32>, KError> {
    let mut v = vec![0i32; t.shape()[0] as usize];
    s.download(t.buffer(), t.offset(), &mut v)?;
    Ok(v)
}

impl Model for FakeModel {
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    fn forward(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        _kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        self.pass(s, b, logits, None)
    }

    fn forward_taps(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        _kv: &KvPool,
        logits: &Tensor,
        taps: Option<&Taps>,
    ) -> Result<(), KError> {
        self.pass(s, b, logits, taps)
    }
}

impl FakeModel {
    /// The forward pass both [`Model`] methods run, with or without taps.
    fn pass(
        &self,
        s: &Stream,
        b: &StepBatch,
        logits: &Tensor,
        taps: Option<&Taps>,
    ) -> Result<(), KError> {
        let blocks = taps.map(|t| t.blocks(&self.cfg)).transpose()?;
        let (tokens, rows) = (read(s, &b.token_ids)?, read(s, &b.logits_rows)?);
        let vocab = self.cfg.vocab as usize;
        if logits.dtype() != DType::F32 || logits.shape() != [rows.len() as i64, vocab as i64] {
            return Err(KError::InvalidArg);
        }
        let mut l = vec![0f32; rows.len() * vocab];
        for (i, &r) in rows.iter().enumerate() {
            let tok = tokens.get(r as usize).map(|&t| usize::try_from(t));
            let Some(Ok(tok)) = tok else {
                return Err(KError::InvalidArg);
            };
            // Far ahead of the rest, so sampling at temperature <= 2 still picks t+1.
            l[i * vocab + (tok + 1) % vocab] = 100.0;
        }
        if let (Some(t), Some(blocks)) = (taps, blocks) {
            let rows = read(s, &t.rows)?;
            let labels = t.layers.iter().map(|&l| u32::from(l));
            let labels = labels.chain(t.final_state.then_some(self.cfg.n_layers));
            for (out, label) in blocks.iter().zip(labels) {
                let mut v = Vec::with_capacity(rows.len() * self.cfg.d_model as usize);
                for &r in &rows {
                    let tok = usize::try_from(r).ok().and_then(|r| tokens.get(r));
                    let tok = *tok.ok_or(KError::InvalidArg)?;
                    v.extend([tok as f32, label as f32]);
                    v.extend((2..self.cfg.d_model).map(|i| i as f32));
                }
                s.upload(out.buffer(), out.offset(), &v)?;
            }
        }
        s.upload(logits.buffer(), logits.offset(), &l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larch_kernels::{Buffer, CPU, Device};

    /// Taps as `run` takes them: `rows` is viewed as `rows_as` (a dtype and shape), and `out` gives
    /// out's dtype and row count.
    struct Spec {
        layers: Vec<u16>,
        final_state: bool,
        rows: Vec<i32>,
        rows_as: (DType, Vec<i64>),
        out: (DType, i64),
    }

    /// Well-formed taps: i32 `[n]` rows and an f32 `out` of `out_rows` rows.
    fn spec(layers: &[u16], final_state: bool, rows: &[i32], out_rows: i64) -> Spec {
        Spec {
            layers: layers.to_vec(),
            final_state,
            rows: rows.to_vec(),
            rows_as: (DType::I32, vec![rows.len() as i64]),
            out: (DType::F32, out_rows),
        }
    }

    /// One prefill of tokens 5, 9, 13 whose logits row is the last, with `spec`'s taps: the logits
    /// and the tap rows.
    fn run(m: &mut dyn Model, spec: Option<Spec>) -> Result<(Vec<f32>, Vec<f32>), KError> {
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        let ints = Buffer::alloc(&dev, 64).unwrap();
        s.upload(&ints, 0, &[5i32, 9, 13, 2]).unwrap();
        let t = |off: u64, n: i64| Tensor::new(&ints, off, DType::I32, &[n]).unwrap();
        let batch = StepBatch {
            token_ids: t(0, 3),
            positions: t(0, 3),
            slot_ids: t(0, 3),
            decode_rows: 0,
            prefill_rows: 3,
            block_table: t(0, 1),
            seq_lens: t(0, 0),
            prefill_q_start: t(0, 1),
            prefill_q_len: t(0, 1),
            logits_rows: t(12, 1),
            attn_splits: 0,
            check: false,
        };
        let (k, v) = (
            Buffer::alloc(&dev, 4).unwrap(),
            Buffer::alloc(&dev, 4).unwrap(),
        );
        let kv = KvPool {
            k: &k,
            v: &v,
            n_blocks: 0,
            block_tokens: 16,
            n_layers: 1,
            n_kv_heads: 1,
            head_dim: 16,
            dtype: DType::F16,
        };
        let vocab = i64::from(m.config().vocab);
        let lb = Buffer::alloc(&dev, vocab as u64 * 4).unwrap();
        let logits = Tensor::new(&lb, 0, DType::F32, &[1, vocab]).unwrap();
        let tb = Buffer::alloc(&dev, 1 << 12).unwrap();
        let mut out = vec![];
        match spec {
            None => m.forward_taps(&s, &batch, &kv, &logits, None)?,
            Some(t) => {
                s.upload(&tb, 0, &t.rows).unwrap();
                let (dtype, out_rows) = t.out;
                let taps = Taps {
                    layers: &t.layers,
                    final_state: t.final_state,
                    rows: Tensor::new(&tb, 0, t.rows_as.0, &t.rows_as.1).unwrap(),
                    out: Tensor::new(&tb, 64, dtype, &[out_rows, 16]).unwrap(),
                };
                m.forward_taps(&s, &batch, &kv, &logits, Some(&taps))?;
                out = vec![0f32; out_rows as usize * 16];
                s.download(&tb, 64, &mut out).unwrap();
            }
        }
        let mut l = vec![0f32; vocab as usize];
        s.download(&lb, 0, &mut l).unwrap();
        Ok((l, out))
    }

    fn tap(t: f32, l: f32) -> Vec<f32> {
        [t, l]
            .into_iter()
            .chain((2..16).map(|i| i as f32))
            .collect()
    }

    #[test]
    fn taps_come_back_for_the_first_and_last_layer_and_the_final_state() {
        let mut m = FakeModel::with_layers(64, 3);
        let (plain, _) = run(&mut m, None).unwrap();
        let (logits, out) = run(&mut m, Some(spec(&[0, 2], true, &[2, 0], 6))).unwrap();
        assert_eq!(logits, plain, "taps change the logits");
        assert_eq!(plain[14], 100.0, "after 13 comes 14");
        // Blocks layer 0, layer 2, final (labelled 3), each with rows 2 then 0 (tokens 13, 5).
        let want: Vec<f32> = [0., 2., 3.]
            .into_iter()
            .flat_map(|l| [tap(13., l), tap(5., l)].concat())
            .collect();
        assert_eq!(out, want);
    }

    #[test]
    fn a_bad_tap_is_an_error() {
        let mut m = FakeModel::with_layers(64, 3);
        // Layer 0 at row 2, with `rows` viewed as `rows_as` and an `out` of dtype `out`.
        let viewed = |rows_as: DType, shape: &[i64], out: DType| Spec {
            rows_as: (rows_as, shape.to_vec()),
            out: (out, 1),
            ..spec(&[0], false, &[2], 1)
        };
        for (name, taps) in [
            ("past the depth", spec(&[3], false, &[2], 1)),
            ("nothing named", spec(&[], false, &[2], 0)),
            ("no rows", spec(&[0], true, &[], 0)),
            ("f16 out", viewed(DType::I32, &[1], DType::F16)),
            ("short out", spec(&[0, 1], true, &[2], 2)),
            ("f32 rows", viewed(DType::F32, &[1], DType::F32)),
            ("2-D rows", viewed(DType::I32, &[1, 1], DType::F32)),
            ("row past the batch", spec(&[0], false, &[3], 1)),
        ] {
            let got = run(&mut m, Some(taps)).err();
            assert_eq!(got, Some(KError::InvalidArg), "{name}");
        }
    }

    /// A model that never learnt taps, as every wrapper in the engine's tests is.
    struct Plain(FakeModel);

    impl Model for Plain {
        fn config(&self) -> &ModelConfig {
            self.0.config()
        }

        fn forward(
            &mut self,
            s: &Stream,
            b: &StepBatch,
            kv: &KvPool,
            logits: &Tensor,
        ) -> Result<(), KError> {
            self.0.forward(s, b, kv, logits)
        }
    }

    #[test]
    fn a_model_without_taps_refuses_them() {
        let mut m = Plain(FakeModel::new(64));
        let (logits, _) = run(&mut m, None).unwrap();
        assert_eq!(logits[14], 100.0);
        let taps = spec(&[0], false, &[2], 1);
        assert_eq!(run(&mut m, Some(taps)).err(), Some(KError::Unsupported));
    }
}
