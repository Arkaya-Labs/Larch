//! Inkling 2 Mini (`inkling2_loop`), as llama.cpp b12723 builds it (`src/models/inkling2.cpp` and
//! `inkling2-loop.cpp`), through larch.h's ops on any device that has them (the CPU devices have all):
//! - The loop: the prelude's layers, the core's run [`InklingCfg::passes`] times over shared weights,
//!   then the coda's; each visit keeps its own K and V (KV layer = visit) and short-convolution state.
//!   Passes after the first restart from the prelude's output and mix the previous pass in through
//!   rank-`gate_rank` controllers on the Q, K, V and FFN inputs (`loop_gate`).
//! - Depth mixing: a layer's attention and FFN inputs, and the head's, are a softmax mix of the
//!   finished blocks (`block` layers each) and the current state by learned queries (`depth_mix`);
//!   the first layer of a block starts a new residual stream.
//! - Attention: Q scaled per channel and clipped, K clipped, V RMS-normed, the full layers' Q and
//!   relative query times the log-scaled position; a learned relative-position bias by distance and a
//!   per-query sink logit (`attn_rel`); a sigmoid output gate. No rotary positions.
//! - Four causal 4-tap convolutions with a residual a layer (K, V, attention out, FFN out), the full
//!   layers' K and V with input-dependent taps (`short_conv`).
//! - MoE: sigmoid routing with a selection bias, top-k weights normalised and scaled
//!   (`moe_route_sigmoid`), one fused gate-up tensor per expert with SITU (`situ_mul`), per-expert
//!   weight scales (`<name>.scale`).
//! - Head: depth mix, RMS norm, the unpadded vocabulary's rows of the padded output, soft cap.
//!
//! Activations are f16 and the residual streams f32 (architecture 6.1); the router and the
//! controllers, f32 in the file, run in f32 (`gemm_ex`), as llama.cpp keeps them. Linear layers take
//! `mmvq` in slices of 16 rows (quant weights) or `gemm` (f16 copies of float weights).

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use larch_kernels::{Buffer, ConvState, DType, Device, KError, KvPool, Stream, Tensor, ops};
use larch_records::LoadError;

use crate::gguf::{self, Gguf, Value, code};
use crate::{Arch, Model, ModelConfig, RopeCfg, StateBatch, StepBatch, Taps};

/// llama.cpp's limit on cache slots (`LLAMA_MAX_LAYERS`), which bounds the visits.
const MAX_VISITS: u32 = 512;
/// The depth mix's and the controllers' RMS norm epsilon (inkling2.cpp's 1e-6f).
const MIX_EPS: f32 = 1e-6;
/// The router weights' normalisation epsilon (`expert_weights_norm_eps`).
const ROUTE_EPS: f32 = 1e-20;
/// Rows a quant `mmvq` call takes (ABI 2).
const MMVQ_ROWS: i64 = 16;
/// The controllers: Q, K, V and FFN (llama.cpp's `LOOP_CTRL_*` order).
const CTRLS: usize = 4;

/// Inkling 2's hyperparameters, as llama.cpp's `load_arch_hparams` reads and checks them.
#[derive(Clone, Debug, PartialEq)]
pub struct InklingCfg {
    pub d_model: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// Physical layers.
    pub layers: u32,
    /// Sliding-window (true) or full attention, per physical layer.
    pub swa: Vec<bool>,
    pub window: u32,
    /// Depth-mix block size.
    pub block: u32,
    /// The loop core `[loop_start, loop_end)` and how often it runs.
    pub loop_start: u32,
    pub loop_end: u32,
    pub passes: u32,
    pub gate_rank: u32,
    /// Relative-bias width and the full layers' extent.
    pub rel: u32,
    pub extent: u32,
    pub rel_clip: f32,
    pub qk_clip: f32,
    pub log_floor: u32,
    pub log_alpha: f32,
    pub experts: u32,
    pub used: u32,
    pub ffn: u32,
    /// `expert_weights_scale`; 0 means 1.
    pub weights_scale: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    /// `final_logit_softcapping`; 0 for none.
    pub softcap: f32,
    /// The short convolutions' taps (`shortconv.l_cache`).
    pub conv: u32,
    pub vocab: u32,
    pub vocab_padded: u32,
    pub ctx_train: u32,
    pub rms_eps: f32,
}

impl InklingCfg {
    /// The hyperparameters of an `inkling2_loop` GGUF, `None` for any other architecture.
    /// `BadGguf(KEY, 0)` where llama.cpp throws on the keys; `UnsupportedArch` for a layout llama.cpp
    /// takes and this port does not (layers of differing attention or expert widths).
    pub fn from_gguf(g: &Gguf) -> Result<Option<InklingCfg>, LoadError> {
        const NAME: &str = "inkling2_loop";
        if g.get("general.architecture").and_then(Value::as_str) != Some(NAME) {
            return Ok(None);
        }
        let bad = LoadError::BadGguf(code::KEY, 0);
        let get = |k: &str| g.get(&format!("{NAME}.{k}"));
        let u = |k: &str| {
            get(k)
                .map(|v| v.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or(bad))
                .transpose()
        };
        let need = |k: &str| u(k)?.ok_or(bad);
        let f = |k: &str| get(k).map(|v| v.as_f32().ok_or(bad)).transpose();
        // A key that may be one value for every layer or an array of one per layer, which must agree
        // here (the KV pool and the scratch take one width).
        let per_layer = |k: &str, n: u32| -> Result<Option<u32>, LoadError> {
            match get(k) {
                None => Ok(None),
                Some(Value::Arr(a)) => {
                    let v: Option<Vec<u32>> = a
                        .iter()
                        .map(|v| v.as_u64().and_then(|v| u32::try_from(v).ok()))
                        .collect();
                    let v = v.filter(|v| v.len() >= n as usize).ok_or(bad)?;
                    match v[..n as usize].iter().all(|&x| x == v[0]) {
                        true => Ok(Some(v[0])),
                        false => Err(LoadError::UnsupportedArch),
                    }
                }
                Some(v) => v
                    .as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or(bad)
                    .map(Some),
            }
        };
        let layers = need("block_count")?;
        // Every layer has tensors of its own, so the table bounds the count.
        if layers == 0 || layers as usize > g.tensors.len() {
            return Err(bad);
        }
        if u("nextn_predict_layers")?.unwrap_or(0) != 0 {
            return Err(bad);
        }
        let d_model = need("embedding_length")?;
        let heads = per_layer("attention.head_count", layers)?.ok_or(bad)?;
        let kv_heads = per_layer("attention.head_count_kv", layers)?.unwrap_or(heads);
        let head_dim = u("attention.key_length")?.unwrap_or(d_model / heads.max(1));
        let v_dim = u("attention.value_length")?.unwrap_or(d_model / heads.max(1));
        // llama.cpp reads the window layers' widths from their own keys; one width here.
        let swa_k = u("attention.key_length_swa")?.unwrap_or(head_dim);
        let swa_v = u("attention.value_length_swa")?.unwrap_or(v_dim);
        let swa: Vec<bool> = match get("attention.sliding_window_pattern").and_then(Value::as_arr) {
            Some(a) if a.len() == layers as usize => a
                .iter()
                .map(|v| match v {
                    Value::Bool(b) => Some(*b),
                    v => v.as_u64().map(|x| x != 0),
                })
                .collect::<Option<_>>()
                .ok_or(bad)?,
            _ => return Err(bad),
        };
        let c = InklingCfg {
            d_model,
            heads,
            kv_heads,
            head_dim,
            layers,
            window: need("attention.sliding_window")?,
            block: need("attn_res.block_size")?,
            loop_start: need("loop.start")?,
            loop_end: need("loop.end")?,
            passes: need("loop.count")?,
            gate_rank: need("loop.gate_rank")?,
            rel: need("attention.relative_length")?,
            extent: need("attention.relative_extent")?,
            rel_clip: f("attention.relative_clip")?.ok_or(bad)?,
            qk_clip: f("attention.qk_clip")?.ok_or(bad)?,
            log_floor: need("attention.log_scale_floor")?,
            log_alpha: f("attention.log_scale_alpha")?.ok_or(bad)?,
            experts: need("expert_count")?,
            used: per_layer("expert_used_count", layers)?.ok_or(bad)?,
            ffn: per_layer("expert_feed_forward_length", layers)?.ok_or(bad)?,
            weights_scale: f("expert_weights_scale")?.unwrap_or(0.0),
            situ_beta: f("activation.situ_beta")?.ok_or(bad)?,
            situ_linear_beta: f("activation.situ_linear_beta")?.ok_or(bad)?,
            softcap: f("final_logit_softcapping")?.unwrap_or(0.0),
            conv: need("shortconv.l_cache")?,
            vocab: need("vocab_size_unpadded")?,
            vocab_padded: need("vocab_size")?,
            ctx_train: need("context_length")?,
            rms_eps: f("attention.layer_norm_rms_epsilon")?.ok_or(bad)?,
            swa,
        };
        let varying = matches!(get("loop.varying"), Some(Value::Bool(true)));
        // llama.cpp's checks (inkling2.cpp, inkling2-loop.cpp), then this port's own bounds.
        let positive = |x: f32| x.is_finite() && x > 0.0;
        let ok = c.rel > 0
            && c.extent > 0
            && c.window > 0
            && c.block > 0
            && c.log_floor > 0
            && (2..=64).contains(&c.conv)
            && c.extent <= i32::MAX as u32
            && c.window <= i32::MAX as u32
            && c.vocab > 0
            && c.vocab <= c.vocab_padded
            && [
                c.rms_eps,
                c.qk_clip,
                c.rel_clip,
                c.situ_beta,
                c.situ_linear_beta,
            ]
            .into_iter()
            .all(positive)
            && c.log_alpha.is_finite()
            && c.log_alpha >= 0.0
            && c.weights_scale.is_finite()
            && c.softcap.is_finite()
            && c.softcap >= 0.0
            && kv_heads > 0
            && heads.is_multiple_of(kv_heads)
            && head_dim > 0
            && head_dim == v_dim
            && c.used > 0
            && c.used <= c.experts
            && c.ffn > 0
            && c.passes >= if varying { 1 } else { 2 }
            && c.gate_rank > 0
            && c.loop_start > 0
            && c.loop_start < c.loop_end
            && c.loop_end <= layers
            && c.loop_start.is_multiple_of(c.block)
            && c.loop_end.is_multiple_of(c.block)
            && c.visits_count().is_some_and(|v| v <= MAX_VISITS)
            && heads.checked_mul(head_dim).is_some()
            && d_model.checked_mul(4).is_some();
        if !ok {
            return Err(bad);
        }
        if swa_k != head_dim || swa_v != v_dim {
            return Err(LoadError::UnsupportedArch);
        }
        // llama.cpp: the tokenizer's size must be the unpadded vocabulary.
        let tokens = g.get("tokenizer.ggml.tokens").and_then(Value::as_arr);
        if tokens.is_some_and(|t| t.len() != c.vocab as usize) {
            return Err(bad);
        }
        Ok(Some(c))
    }

    /// Layer visits a token makes: the prelude, the core `passes` times, the coda.
    fn visits_count(&self) -> Option<u32> {
        let core = self.loop_end - self.loop_start;
        self.passes
            .checked_sub(1)?
            .checked_mul(core)?
            .checked_add(self.layers)
    }

    /// Short-convolution channels per visit: K, V, the attention output, the FFN output.
    pub fn channels(&self) -> u32 {
        2 * self.kv_heads * self.head_dim + 2 * self.d_model
    }

    /// States a depth mix can see: a block per `block` layers, plus the current one.
    fn states(&self) -> u32 {
        self.layers.div_ceil(self.block) + 1
    }

    /// The config the engine and the [`Model`] trait read: every visit a layer of the KV pool.
    fn model_config(&self) -> ModelConfig {
        let visits = self.visits_count().expect("checked");
        ModelConfig {
            arch: Arch::Inkling2Loop,
            n_layers: visits,
            kv_layers: visits,
            d_model: self.d_model,
            n_heads: self.heads,
            n_kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            ffn: self.ffn,
            vocab: self.vocab,
            ctx_train: self.ctx_train,
            // No rotary positions: the relative bias carries them.
            rope: RopeCfg {
                n_dims: 0,
                neox: false,
                freq_base: 10000.0,
                freq_scale: 1.0,
                yarn: false,
                n_ctx_orig: self.ctx_train,
                sections: None,
            },
            rms_eps: self.rms_eps,
            qkv_bias: false,
            qk_norm: false,
            tie_embeddings: false,
        }
    }
}

/// A weight in [`Inkling`]'s weights buffer.
#[derive(Clone, Debug)]
struct W {
    off: u64,
    dtype: DType,
    /// Row-major; stacked experts as one `[experts * n, k]` matrix.
    shape: Vec<i64>,
}

impl W {
    fn t<'a>(&self, b: &'a Buffer) -> Result<Tensor<'a>, KError> {
        Tensor::new(b, self.off, self.dtype, &self.shape)
    }
}

/// One physical layer's weights.
struct Layer {
    swa: bool,
    /// The window layers' input norm; the full layers' Q, K and V input norms.
    norms: Vec<W>,
    wq: W,
    wk: W,
    wv: W,
    wo: W,
    gate: W,
    q_clip: W,
    sink: W,
    rel: W,
    rel_bias: W,
    rel_clip: W,
    k_conv: W,
    v_conv: W,
    attn_post_conv: W,
    ffn_post_conv: W,
    /// The full layers' dynamic taps.
    k_taps: Option<W>,
    v_taps: Option<W>,
    /// `[queries, d]`; none on layer 0.
    attn_query: Option<W>,
    ffn_query: W,
    ffn_norm: W,
    router: W,
    probs_bias: Option<W>,
    gate_up: W,
    down: W,
    gate_up_scale: Option<W>,
    down_scale: Option<W>,
}

/// A core layer's controllers.
struct LoopW {
    /// `[4, d]`: the Q, K, V and FFN message queries.
    query: W,
    /// Per controller: `[rank, 2 d]`, `[d, rank]`, `[d, rank]`.
    down: [W; CTRLS],
    gain: [W; CTRLS],
    forget: [W; CTRLS],
}

/// One layer visit: its physical layer and pass; its KV and state layer is its index.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Visit {
    layer: u32,
    pass: u32,
}

/// The visits in order: prelude, core per pass, coda.
fn visits(c: &InklingCfg) -> Vec<Visit> {
    let mut v: Vec<Visit> = (0..c.loop_start)
        .map(|layer| Visit { layer, pass: 0 })
        .collect();
    for pass in 0..c.passes {
        v.extend((c.loop_start..c.loop_end).map(|layer| Visit { layer, pass }));
    }
    v.extend((c.loop_end..c.layers).map(|layer| Visit { layer, pass: 0 }));
    v
}

/// Activations for up to `rows` rows, each its own buffer.
struct Scratch {
    rows: i64,
    /// f32 `[2][states][rows, d]`: two regions of residual states (a pass's blocks then its current
    /// state), so a pass's bank stays whole while the next pass builds its own.
    stack: Buffer,
    emb: Buffer,
    mix: Buffer,
    msg: Buffer,
    /// f32 `[4][rows, d]`: the controlled Q, K, V and FFN inputs.
    ctl: Buffer,
    cat: Buffer,
    low: Buffer,
    gain: Buffer,
    forget: Buffer,
    /// f16 `[3][rows, d]`: the normed Q, K and V inputs (the FFN input in the first).
    h: Buffer,
    h32: Buffer,
    q: Buffer,
    kraw: Buffer,
    vraw: Buffer,
    k: Buffer,
    v: Buffer,
    ktaps: Buffer,
    vtaps: Buffer,
    rel: Buffer,
    sink: Buffer,
    gate: Buffer,
    attn: Buffer,
    o: Buffer,
    logits: Buffer,
    ids: Buffer,
    weights: Buffer,
    gu: Buffer,
    act: Buffer,
    down: Buffer,
    head: Buffer,
    head16: Buffer,
    /// f32 ones, `max(d, head_dim)` of them: the weight of the weightless norms.
    ones: Buffer,
    /// i32 `0..`: the rows a copy of residual states gathers.
    ids_iota: Buffer,
}

/// Inkling 2 Mini on a device: weights, scratch for `max_rows` rows and the short convolutions' state.
pub struct Inkling {
    cfg: ModelConfig,
    ink: InklingCfg,
    w: Buffer,
    layers: Vec<Layer>,
    loops: Vec<LoopW>,
    visits: Vec<Visit>,
    tok_embd: W,
    tok_norm: W,
    output: W,
    output_norm: W,
    output_query: W,
    sc: Scratch,
    /// f16 `[visits][slots][conv - 1][channels]` and its slots.
    state: Option<(Buffer, u32)>,
}

/// The f16 and the f32 tensor of `shape` at the start of `buf`.
fn t16<'a>(buf: &'a Buffer, shape: &[i64]) -> Result<Tensor<'a>, KError> {
    Tensor::new(buf, 0, DType::F16, shape)
}
fn t32<'a>(buf: &'a Buffer, shape: &[i64]) -> Result<Tensor<'a>, KError> {
    Tensor::new(buf, 0, DType::F32, shape)
}

/// A kernel error as a load error.
fn backend(e: KError) -> LoadError {
    LoadError::Backend(e.code())
}

/// What a tensor becomes on the device.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// f32 as stored (vectors, convolutions, queries, clips, biases, scales, the relative table);
    /// a float matrix the f32 ops read (router, controllers), f16 and bf16 widened.
    F32,
    /// A matrix the linear layers read: quant as stored, float as f16 (gemm).
    Mat,
    /// Quant only: experts (mmvq_id) and the head (mmvq).
    Quant,
    /// The token embedding: quant, f16 or f32 as stored (embed).
    Embed,
}

/// GGML type ids of the floats a float tensor may be stored in.
const F32_TYPE: u32 = 0;
const F16_TYPE: u32 = 1;
const BF16_TYPE: u32 = gguf::BF16;

impl Inkling {
    /// Loads an `inkling2_loop` GGUF onto `dev` with scratch for `max_rows` rows a step, checking
    /// every hyperparameter, tensor name, shape, type and byte count against llama.cpp's loader. A type
    /// larch has no dtype for is `UnsupportedQuant`; NVFP4 experts load everywhere but run only on the
    /// CPU devices for now (a GPU answers their first op `Unsupported`).
    pub fn load(
        path: &Path,
        dev: &Device,
        s: &Stream,
        max_rows: u32,
    ) -> Result<Inkling, LoadError> {
        let file = File::open(path).map_err(|_| LoadError::Io)?;
        let len = file.metadata().map_err(|_| LoadError::Io)?.len();
        let mut f = BufReader::new(file);
        let g = Gguf::read(&mut f, len)?;
        let ink = InklingCfg::from_gguf(&g)?.ok_or(LoadError::UnsupportedArch)?;
        if max_rows == 0 {
            return Err(LoadError::Backend(KError::InvalidArg.code()));
        }
        let (d, hd) = (i64::from(ink.d_model), i64::from(ink.head_dim));
        let (qd, kvd) = (i64::from(ink.heads) * hd, i64::from(ink.kv_heads) * hd);
        let (h, kvh, rel) = (
            i64::from(ink.heads),
            i64::from(ink.kv_heads),
            i64::from(ink.rel),
        );
        let (e, ffn, conv) = (
            i64::from(ink.experts),
            i64::from(ink.ffn),
            i64::from(ink.conv),
        );
        let (vp, rank) = (i64::from(ink.vocab_padded), i64::from(ink.gate_rank));

        // Every tensor this port reads, in file order, with the shape and kind it must have; the
        // weights buffer holds them packed (256-byte aligned) in that order.
        let mut plan: Vec<(String, Vec<i64>, Kind, W)> = Vec::new();
        let mut at = 0u64;
        let mut add = |name: String, shape: Vec<i64>, kind: Kind| -> Result<W, LoadError> {
            let t = g
                .tensor(&name)
                .ok_or(LoadError::BadGguf(code::MISSING_TENSOR, 0))?;
            if t.shape() != shape {
                return Err(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset));
            }
            // The table's byte count must be the type's for the shape (Gguf::read sizes the types it
            // knows and bounds them by the file; 0 for any other, which the match below refuses).
            let float = matches!(t.ggml_type, F32_TYPE | F16_TYPE | BF16_TYPE);
            let elems: i64 = shape.iter().product();
            let stored = match t.ggml_type {
                BF16_TYPE => Some(elems as u64 * 2),
                ty => gguf::dtype(ty)
                    .ok()
                    .and_then(|dt| dt.checked_bytes(elems as u64)),
            };
            if stored.is_some_and(|b| b != t.bytes) {
                return Err(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset));
            }
            let dtype = match (kind, float) {
                (Kind::F32, true) => DType::F32,
                (Kind::Mat, true) => DType::F16,
                (Kind::Embed, _) if t.ggml_type == BF16_TYPE => {
                    return Err(LoadError::UnsupportedQuant(t.ggml_type));
                }
                (Kind::Mat | Kind::Quant | Kind::Embed, _) => {
                    let dt = gguf::dtype(t.ggml_type)?;
                    match (kind, dt.block() > 1) {
                        (Kind::Embed, _) | (_, true) => dt,
                        _ => return Err(LoadError::UnsupportedQuant(t.ggml_type)),
                    }
                }
                (Kind::F32, false) => return Err(LoadError::UnsupportedQuant(t.ggml_type)),
            };
            // Stacked experts [experts, n, k] are one [experts * n, k] matrix, as mmvq_id takes them.
            let dshape = match shape[..] {
                [x, n, k] if kind == Kind::Quant => vec![x * n, k],
                _ => shape.clone(),
            };
            let n: i64 = dshape.iter().product();
            let bytes = dtype
                .checked_bytes(n as u64)
                .ok_or(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset))?;
            let w = W {
                off: at,
                dtype,
                shape: dshape,
            };
            at = (at + bytes).next_multiple_of(256);
            plan.push((name, shape, kind, w.clone()));
            Ok(w)
        };
        let tok_embd = add("token_embd.weight".into(), vec![vp, d], Kind::Embed)?;
        let tok_norm = add("token_embd_norm.weight".into(), vec![d], Kind::F32)?;
        let output = add("output.weight".into(), vec![vp, d], Kind::Quant)?;
        let output_norm = add("output_norm.weight".into(), vec![d], Kind::F32)?;
        let output_query = add("output_res_query.weight".into(), vec![d], Kind::F32)?;
        let mut layers = Vec::new();
        for i in 0..ink.layers {
            let n = |s: &str| format!("blk.{i}.{s}");
            let swa = ink.swa[i as usize];
            let extent = i64::from(if swa { ink.window } else { ink.extent });
            let wq = add(n("attn_q.weight"), vec![qd, d], Kind::Mat)?;
            let wk = add(n("attn_k.weight"), vec![kvd, d], Kind::Mat)?;
            let wv = add(n("attn_v.weight"), vec![kvd, d], Kind::Mat)?;
            let wo = add(n("attn_output.weight"), vec![d, qd], Kind::Mat)?;
            let gate = add(n("attn_gate.weight"), vec![qd, d], Kind::Mat)?;
            let q_clip = add(n("attn_q_clip.weight"), vec![h, hd], Kind::F32)?;
            let sink = add(n("attn_sink_proj.weight"), vec![h, d], Kind::Mat)?;
            let rel_w = add(n("attn_rel.weight"), vec![rel * h, d], Kind::Mat)?;
            let rel_bias = add(n("attn_rel_bias.weight"), vec![extent, rel], Kind::F32)?;
            let rel_clip = add(n("attn_rel_clip.weight"), vec![h, rel], Kind::F32)?;
            let k_conv = add(n("attn_k_conv.weight"), vec![kvd, conv], Kind::F32)?;
            let v_conv = add(n("attn_v_conv.weight"), vec![kvd, conv], Kind::F32)?;
            let attn_post_conv = add(n("attn_post_conv.weight"), vec![d, conv], Kind::F32)?;
            let ffn_post_conv = add(n("ffn_post_conv.weight"), vec![d, conv], Kind::F32)?;
            let (mut norms, mut k_taps, mut v_taps) = (Vec::new(), None, None);
            if swa {
                norms.push(add(n("attn_norm.weight"), vec![d], Kind::F32)?);
            } else {
                for x in ["q", "k", "v"] {
                    let name = n(&format!("attn_{x}_inp_norm.weight"));
                    norms.push(add(name, vec![d], Kind::F32)?);
                }
                let taps = vec![conv * kvh, d];
                k_taps = Some(add(n("attn_k_conv_proj.weight"), taps.clone(), Kind::Mat)?);
                v_taps = Some(add(n("attn_v_conv_proj.weight"), taps, Kind::Mat)?);
            }
            // [queries, d]: one on window layers, Q's, K's and V's on full layers; none on layer 0.
            let attn_query = match (i, swa) {
                (0, _) => None,
                (_, true) => Some(add(n("attn_res_query.weight"), vec![d], Kind::F32)?),
                (_, false) => Some(add(n("attn_res_query.weight"), vec![3, d], Kind::F32)?),
            }
            .map(|q| W {
                shape: vec![if swa { 1 } else { 3 }, d],
                ..q
            });
            let ffn_query = W {
                shape: vec![1, d],
                ..add(n("ffn_res_query.weight"), vec![d], Kind::F32)?
            };
            let ffn_norm = add(n("ffn_norm.weight"), vec![d], Kind::F32)?;
            let router = add(n("ffn_gate_inp.weight"), vec![e, d], Kind::F32)?;
            let probs_bias = match g.tensor(&n("exp_probs_b.bias")) {
                Some(_) => Some(add(n("exp_probs_b.bias"), vec![e], Kind::F32)?),
                None => None,
            };
            let gate_up = add(
                n("ffn_gate_up_exps.weight"),
                vec![e, 2 * ffn, d],
                Kind::Quant,
            )?;
            let down = add(n("ffn_down_exps.weight"), vec![e, d, ffn], Kind::Quant)?;
            // llama.cpp's `get_w_scale`: a `<weight>.scale` of one value per expert scales it.
            let mut scale = |name: &str| match g.tensor(&n(name)) {
                Some(_) => add(n(name), vec![e], Kind::F32).map(Some),
                None => Ok(None),
            };
            let gate_up_scale = scale("ffn_gate_up_exps.scale")?;
            let down_scale = scale("ffn_down_exps.scale")?;
            layers.push(Layer {
                swa,
                norms,
                wq,
                wk,
                wv,
                wo,
                gate,
                q_clip,
                sink,
                rel: rel_w,
                rel_bias,
                rel_clip,
                k_conv,
                v_conv,
                attn_post_conv,
                ffn_post_conv,
                k_taps,
                v_taps,
                attn_query,
                ffn_query,
                ffn_norm,
                router,
                probs_bias,
                gate_up,
                down,
                gate_up_scale,
                down_scale,
            });
        }
        let mut loops = Vec::new();
        for i in ink.loop_start..ink.loop_end {
            let n = |s: &str| format!("blk.{i}.{s}");
            let query = add(n("loop_msg_query.weight"), vec![CTRLS as i64, d], Kind::F32)?;
            let mut ctrl = |part: &str, shape: [i64; 2]| -> Result<[W; CTRLS], LoadError> {
                let mut out = Vec::new();
                for c in ["q", "k", "v", "ffn"] {
                    let name = n(&format!("loop_{c}_ctrl_{part}.weight"));
                    out.push(add(name, shape.to_vec(), Kind::F32)?);
                }
                Ok(out.try_into().expect("four controllers"))
            };
            let down = ctrl("down", [rank, 2 * d])?;
            let gain = ctrl("gain", [d, rank])?;
            let forget = ctrl("forget", [d, rank])?;
            loops.push(LoopW {
                query,
                down,
                gain,
                forget,
            });
        }

        // The upload: each tensor read from the file, widened or narrowed to its device float type.
        let w = Buffer::alloc(dev, at.max(256)).map_err(backend)?;
        for (name, _, _, dst) in &plan {
            let t = g.tensor(name).expect("planned");
            let start = g
                .data_offset
                .checked_add(t.offset)
                .filter(|&o| o.checked_add(t.bytes).is_some_and(|end| end <= len))
                .ok_or(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset))?;
            let mut raw = vec![0u8; t.bytes as usize];
            f.seek(SeekFrom::Start(start)).map_err(|_| LoadError::Io)?;
            f.read_exact(&mut raw).map_err(|_| LoadError::Io)?;
            let bytes = convert(&raw, t.ggml_type, dst.dtype);
            s.upload(&w, dst.off, &bytes).map_err(backend)?;
        }
        let rows = i64::from(max_rows);
        let sc = Scratch::alloc(dev, s, &ink, rows).map_err(backend)?;
        s.sync().map_err(backend)?;
        let cfg = ink.model_config();
        Ok(Inkling {
            cfg,
            visits: visits(&ink),
            ink,
            w,
            layers,
            loops,
            tok_embd,
            tok_norm,
            output,
            output_norm,
            output_query,
            sc,
            state: None,
        })
    }

    /// The hyperparameters.
    pub fn inkling(&self) -> &InklingCfg {
        &self.ink
    }

    /// Bytes one state slot takes: every visit's short-convolution history.
    pub fn slot_bytes(&self) -> u64 {
        u64::from(self.cfg.n_layers)
            * u64::from(self.ink.conv - 1)
            * u64::from(self.ink.channels())
            * 2
    }

    /// Allocates `slots` zeroed slots of the short convolutions' state, replacing any.
    pub fn alloc_state(&mut self, s: &Stream, slots: u32) -> Result<(), LoadError> {
        let bytes = self
            .slot_bytes()
            .checked_mul(u64::from(slots))
            .filter(|&b| b > 0)
            .ok_or(LoadError::Backend(KError::InvalidArg.code()))?;
        let buf = Buffer::alloc(s.device(), bytes).map_err(backend)?;
        s.upload(&buf, 0, &vec![0u8; bytes as usize])
            .map_err(backend)?;
        self.state = Some((buf, slots));
        Ok(())
    }

    fn conv_state(&self) -> Result<ConvState<'_>, KError> {
        let (buf, slots) = self.state.as_ref().ok_or(KError::InvalidArg)?;
        Ok(ConvState {
            conv: buf,
            n_slots: *slots,
            n_layers: self.cfg.n_layers,
            kernel: self.ink.conv,
            channels: self.ink.channels(),
        })
    }

    /// Byte ranges of `slot` in the state buffer, one per visit.
    fn slot_ranges(&self, slot: u32) -> Result<Vec<(u64, u64)>, KError> {
        let (_, slots) = self.state.as_ref().ok_or(KError::InvalidArg)?;
        if slot >= *slots {
            return Err(KError::InvalidArg);
        }
        let per = self.slot_bytes() / u64::from(self.cfg.n_layers);
        Ok((0..u64::from(self.cfg.n_layers))
            .map(|v| ((v * u64::from(*slots) + u64::from(slot)) * per, per))
            .collect())
    }

    /// `out = x · wᵀ`: quant weights by mmvq in slices of [`MMVQ_ROWS`], f16 ones by gemm.
    fn linear(&self, s: &Stream, w: &W, x: &Tensor, out: &Tensor) -> Result<(), KError> {
        let wt = w.t(&self.w)?;
        if w.dtype == DType::F16 {
            return ops::gemm(s, x, &wt, None, out, 1.0, 0.0);
        }
        let rows = x.shape()[0];
        if rows != out.shape()[0] {
            return Err(KError::InvalidArg);
        }
        for at in (0..rows).step_by(MMVQ_ROWS as usize) {
            let n = (rows - at).min(MMVQ_ROWS);
            ops::mmvq(s, &wt, &x.rows(at, n)?, None, &out.rows(at, n)?)?;
        }
        Ok(())
    }

    /// mmvq_id in slices of [`MMVQ_ROWS`] rows.
    fn experts(
        &self,
        s: &Stream,
        w: &W,
        x: &Tensor,
        ids: &Tensor,
        weights: Option<&Tensor>,
        out: &Tensor,
    ) -> Result<(), KError> {
        let wt = w.t(&self.w)?;
        let rows = x.shape()[0];
        for at in (0..rows).step_by(MMVQ_ROWS as usize) {
            let n = (rows - at).min(MMVQ_ROWS);
            let wr = weights.map(|w| w.rows(at, n)).transpose()?;
            ops::mmvq_id(
                s,
                &wt,
                &x.rows(at, n)?,
                &ids.rows(at, n)?,
                None,
                wr.as_ref(),
                &out.rows(at, n)?,
            )?;
        }
        Ok(())
    }

    /// The controller of one input: `x` and its message through the layer's rank projections, into
    /// `out` (llama.cpp's `control`).
    #[allow(clippy::too_many_arguments)]
    fn control(
        &self,
        s: &Stream,
        lw: &LoopW,
        c: usize,
        x: &Tensor,
        msg: &Tensor,
        out: &Tensor,
        rows: i64,
    ) -> Result<(), KError> {
        let (d, rank) = (i64::from(self.ink.d_model), i64::from(self.ink.gate_rank));
        let sc = &self.sc;
        let ones = Tensor::new(&sc.ones, 0, DType::F32, &[d])?;
        let cat = Tensor::new(&sc.cat, 0, DType::F32, &[rows, 2 * d])?;
        ops::rmsnorm_f32(s, x, &ones, MIX_EPS, &cat.cols(0, d)?)?;
        ops::rmsnorm_f32(s, msg, &ones, MIX_EPS, &cat.cols(d, d)?)?;
        let low = Tensor::new(&sc.low, 0, DType::F32, &[rows, rank])?;
        let gain = Tensor::new(&sc.gain, 0, DType::F32, &[rows, d])?;
        let forget = Tensor::new(&sc.forget, 0, DType::F32, &[rows, d])?;
        ops::gemm_ex(
            s,
            &cat,
            &lw.down[c].t(&self.w)?,
            &low,
            false,
            true,
            1.0,
            0.0,
        )?;
        ops::gemm_ex(
            s,
            &low,
            &lw.gain[c].t(&self.w)?,
            &gain,
            false,
            true,
            1.0,
            0.0,
        )?;
        ops::gemm_ex(
            s,
            &low,
            &lw.forget[c].t(&self.w)?,
            &forget,
            false,
            true,
            1.0,
            0.0,
        )?;
        ops::loop_gate(s, x, msg, &gain, &forget, out)
    }

    /// One visit of physical layer `vis.layer` at KV and state layer `v`: `blocks` (stack slots) and
    /// `cur` as llama.cpp's `build_layer` updates them; `bank` the previous pass's states (stack
    /// slots, contiguous) on a looped visit.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        s: &Stream,
        b: &StepBatch,
        st: &StateBatch,
        kv: &KvPool,
        v: u32,
        vis: Visit,
        blocks: &mut Vec<usize>,
        cur: &mut usize,
        bank: Option<(usize, usize)>,
        region: usize,
    ) -> Result<(), KError> {
        let ink = &self.ink;
        let l = &self.layers[vis.layer as usize];
        let sc = &self.sc;
        let rows = i64::from(b.decode_rows + b.prefill_rows);
        let (d, hd) = (i64::from(ink.d_model), i64::from(ink.head_dim));
        let (h, kvh) = (i64::from(ink.heads), i64::from(ink.kv_heads));
        let (qd, kvd, conv) = (h * hd, kvh * hd, i64::from(ink.conv));
        let ip = vis.layer;
        let slot = |i: usize| self.slot(i, rows);
        let wt = |w: &W| w.t(&self.w);
        let state = self.conv_state()?;
        let looped = bank.map(|(first, n)| (first, n, &self.loops[(ip - ink.loop_start) as usize]));

        // The messages of a looped visit: the previous pass's bank mixed by the layer's queries.
        let msg = t32(&sc.msg, &[rows, CTRLS as i64, d])?;
        if let Some((first, n, lw)) = looped {
            ops::depth_mix(s, &self.states(first, n, rows)?, &wt(&lw.query)?, &msg)?;
        }
        let msg_of = |c: usize| {
            msg.reshape(&[rows, CTRLS as i64 * d])?
                .cols(c as i64 * d, d)
        };

        // Attention inputs: the depth mix (one query, or Q, K and V's on full layers), then the
        // controllers, then the norms.
        let attn_q = l.attn_query.as_ref();
        let queries = attn_q.map_or(1, |q| q.shape[0]);
        let mix = t32(&sc.mix, &[rows, queries, d])?;
        if let Some(q) = attn_q {
            ops::depth_mix(s, &self.states_with(blocks, *cur, rows)?, &wt(q)?, &mix)?;
        }
        let input = |j: i64| -> Result<Tensor, KError> {
            match attn_q {
                Some(_) => mix
                    .reshape(&[rows, queries * d])?
                    .cols(if l.swa { 0 } else { j } * d, d),
                None => slot(*cur),
            }
        };
        let mut inputs = [input(0)?, input(1)?, input(2)?];
        let ctl = t32(&sc.ctl, &[CTRLS as i64 * rows, d])?;
        if let Some((_, _, lw)) = looped {
            for (j, inp) in inputs.iter_mut().enumerate() {
                let out = ctl.rows(j as i64 * rows, rows)?;
                self.control(s, lw, j, inp, &msg_of(j)?, &out, rows)?;
                *inp = out;
            }
        }
        let hq = t16(&sc.h, &[3 * rows, d])?;
        let hs: [Tensor; 3] = [
            hq.rows(0, rows)?,
            hq.rows(rows, rows)?,
            hq.rows(2 * rows, rows)?,
        ];
        let eps = ink.rms_eps;
        // A window layer without messages has one input; with them, or on a full layer, three.
        let three = looped.is_some() || !l.swa;
        for j in 0..if three { 3 } else { 1 } {
            let norm = &l.norms[if l.swa { 0 } else { j }];
            ops::rmsnorm(s, &inputs[j], &wt(norm)?, eps, &hs[j])?;
        }
        let (hq, hk, hv) = match three {
            true => (&hs[0], &hs[1], &hs[2]),
            false => (&hs[0], &hs[0], &hs[0]),
        };

        // Q, K, V and the relative query; K and V through their convolutions.
        let q = t16(&sc.q, &[rows, qd])?;
        self.linear(s, &l.wq, hq, &q)?;
        let (kraw, vraw) = (t16(&sc.kraw, &[rows, kvd])?, t16(&sc.vraw, &[rows, kvd])?);
        self.linear(s, &l.wk, hk, &kraw)?;
        self.linear(s, &l.wv, hv, &vraw)?;
        let (ktaps, vtaps) = (
            t16(&sc.ktaps, &[rows, kvh, conv])?,
            t16(&sc.vtaps, &[rows, kvh, conv])?,
        );
        if let (Some(kt), Some(vt)) = (&l.k_taps, &l.v_taps) {
            self.linear(s, kt, hk, &ktaps.reshape(&[rows, kvh * conv])?)?;
            self.linear(s, vt, hv, &vtaps.reshape(&[rows, kvh * conv])?)?;
        }
        let taps = l.k_taps.is_some();
        let (k, vv) = (t16(&sc.k, &[rows, kvd])?, t16(&sc.v, &[rows, kvd])?);
        let conv_call = |x: &Tensor,
                         w: &W,
                         dyn_t: Option<&Tensor>,
                         res: Option<&Tensor>,
                         off: u32,
                         out: &Tensor| {
            ops::short_conv(
                s,
                x,
                &wt(w)?,
                dyn_t,
                res,
                &state,
                v,
                off,
                &st.slots,
                &st.q_lens,
                None,
                out,
            )
        };
        conv_call(&kraw, &l.k_conv, taps.then_some(&ktaps), None, 0, &k)?;
        conv_call(
            &vraw,
            &l.v_conv,
            taps.then_some(&vtaps),
            None,
            kvd as u32,
            &vv,
        )?;
        let rel_t = t16(&sc.rel, &[rows, h * i64::from(ink.rel)])?;
        self.linear(s, &l.rel, hq, &rel_t)?;
        let q3 = q.reshape(&[rows, h, hd])?;
        let (k3, v3) = (k.reshape(&[rows, kvh, hd])?, vv.reshape(&[rows, kvh, hd])?);
        let rel3 = rel_t.reshape(&[rows, h, i64::from(ink.rel)])?;
        // The full layers' log-scaled positions on Q and the relative query.
        let pos = (!l.swa).then_some(&b.positions);
        let (floor, alpha) = (ink.log_floor, ink.log_alpha);
        ops::clip_norm(
            s,
            &q3,
            Some(&wt(&l.q_clip)?),
            ink.qk_clip,
            pos,
            floor,
            alpha,
            &q3,
        )?;
        ops::clip_norm(s, &k3, None, ink.qk_clip, None, floor, alpha, &k3)?;
        let ones = Tensor::new(&sc.ones, 0, DType::F32, &[hd])?;
        ops::qk_norm(s, &v3, &ones, eps)?;
        let rc = wt(&l.rel_clip)?;
        ops::clip_norm(s, &rel3, Some(&rc), ink.rel_clip, pos, floor, alpha, &rel3)?;
        let sink = t16(&sc.sink, &[rows, h])?;
        self.linear(s, &l.sink, hq, &sink)?;
        let gate = t16(&sc.gate, &[rows, qd])?;
        self.linear(s, &l.gate, hq, &gate)?;
        ops::kv_write(s, &k3, &v3, &b.slot_ids, kv, v)?;

        // Attention: decode rows in decode's form, then the prefill chunks as spans.
        let attn = t16(&sc.attn, &[rows, h, hd])?;
        let (nd, np) = (i64::from(b.decode_rows), i64::from(b.prefill_rows));
        let window = if l.swa { ink.window } else { 0 };
        let scale = 1.0 / (hd as f32).sqrt();
        let rel_w = wt(&l.rel_bias)?;
        if nd > 0 {
            ops::attn_rel(
                s,
                &q3.rows(0, nd)?,
                kv,
                v,
                &b.block_table.rows(0, nd)?,
                &b.seq_lens,
                None,
                &rel3.rows(0, nd)?,
                &rel_w,
                &sink.rows(0, nd)?,
                window,
                b.attn_splits,
                scale,
                &attn.rows(0, nd)?,
            )?;
        }
        if np > 0 {
            let seqs = b.prefill_q_len.shape()[0];
            ops::attn_rel(
                s,
                &q3.rows(nd, np)?,
                kv,
                v,
                &b.block_table.rows(nd, seqs)?,
                &b.prefill_q_start,
                Some(&b.prefill_q_len),
                &rel3.rows(nd, np)?,
                &rel_w,
                &sink.rows(nd, np)?,
                window,
                b.attn_splits,
                scale,
                &attn.rows(nd, np)?,
            )?;
        }
        let attn2 = attn.reshape(&[rows, qd])?;
        ops::sigmoid_mul(s, &attn2, &gate, &attn2)?;
        let o = t16(&sc.o, &[rows, d])?;
        self.linear(s, &l.wo, &attn2, &o)?;
        let (post, ffn_off) = (2 * kvd as u32, 2 * kvd as u32 + ink.d_model);
        // The first layer of a block starts a new residual stream; the earlier one joins the blocks.
        if ip.is_multiple_of(ink.block) {
            blocks.push(*cur);
            let next = self.next_slot(region, blocks.len())?;
            conv_call(&o, &l.attn_post_conv, None, None, post, &slot(next)?)?;
            *cur = next;
        } else {
            let c = slot(*cur)?;
            conv_call(&o, &l.attn_post_conv, None, Some(&c), post, &c)?;
        }
        // MoE: the depth mix, the controller, the norm, the router and the experts.
        let fmix = t32(&sc.mix, &[rows, d])?;
        ops::depth_mix(
            s,
            &self.states_with(blocks, *cur, rows)?,
            &wt(&l.ffn_query)?,
            &fmix.reshape(&[rows, 1, d])?,
        )?;
        let mut ffn_in = fmix;
        if let Some((_, _, lw)) = looped {
            let out = ctl.rows(3 * rows, rows)?;
            self.control(s, lw, 3, &ffn_in, &msg_of(3)?, &out, rows)?;
            ffn_in = out;
        }
        let (e, k, ffn) = (
            i64::from(ink.experts),
            i64::from(ink.used),
            i64::from(ink.ffn),
        );
        let h32 = t32(&sc.h32, &[rows, d])?;
        ops::rmsnorm_f32(s, &ffn_in, &wt(&l.ffn_norm)?, eps, &h32)?;
        ops::rmsnorm(s, &ffn_in, &wt(&l.ffn_norm)?, eps, &hs[0])?;
        let logits = t32(&sc.logits, &[rows, e])?;
        ops::gemm_ex(s, &h32, &wt(&l.router)?, &logits, false, true, 1.0, 0.0)?;
        let ids = Tensor::new(&sc.ids, 0, DType::I32, &[rows, k])?;
        let weights = t32(&sc.weights, &[rows, k])?;
        let pb = l.probs_bias.as_ref().map(&wt).transpose()?;
        let ds = l.down_scale.as_ref().map(&wt).transpose()?;
        let ws = ink.weights_scale;
        ops::moe_route_sigmoid(
            s,
            &logits,
            pb.as_ref(),
            ROUTE_EPS,
            ws,
            ds.as_ref(),
            &ids,
            &weights,
        )?;
        let gu = t32(&sc.gu, &[rows, k, 2 * ffn])?;
        self.experts(s, &l.gate_up, &hs[0], &ids, None, &gu)?;
        let act = t16(&sc.act, &[rows, k, ffn])?;
        let gs = l.gate_up_scale.as_ref().map(&wt).transpose()?;
        ops::situ_mul(
            s,
            &gu,
            &ids,
            gs.as_ref(),
            ink.situ_beta,
            ink.situ_linear_beta,
            &act,
        )?;
        let down = t16(&sc.down, &[rows, d])?;
        self.experts(s, &l.down, &act, &ids, Some(&weights), &down)?;
        let c = slot(*cur)?;
        conv_call(&down, &l.ffn_post_conv, None, Some(&c), ffn_off, &c)
    }

    /// Stack slot `i` as f32 `[rows, d]` (the stack is laid out per step, `rows` a slot).
    fn slot(&self, i: usize, rows: i64) -> Result<Tensor<'_>, KError> {
        let d = i64::from(self.ink.d_model);
        Tensor::new(
            &self.sc.stack,
            (i as u64) * (rows * d * 4) as u64,
            DType::F32,
            &[rows, d],
        )
    }

    /// Slots `first..first + n` as f32 `[n, rows, d]`.
    fn states(&self, first: usize, n: usize, rows: i64) -> Result<Tensor<'_>, KError> {
        let d = i64::from(self.ink.d_model);
        let off = (first as u64) * (rows * d * 4) as u64;
        Tensor::new(&self.sc.stack, off, DType::F32, &[n as i64, rows, d])
    }

    /// The blocks and the current state, which are consecutive slots of one region.
    fn states_with(&self, blocks: &[usize], cur: usize, rows: i64) -> Result<Tensor<'_>, KError> {
        let first = blocks.first().copied().unwrap_or(cur);
        let whole = blocks
            .iter()
            .chain([&cur])
            .enumerate()
            .all(|(i, &x)| x == first + i);
        if !whole {
            return Err(KError::InvalidArg);
        }
        self.states(first, blocks.len() + 1, rows)
    }

    /// The slot after `n` states of `region`.
    fn next_slot(&self, region: usize, n: usize) -> Result<usize, KError> {
        let per = self.ink.states() as usize;
        if n >= per {
            return Err(KError::InvalidArg);
        }
        Ok(region * per + n)
    }

    /// The forward pass (llama.cpp's `inkling2_loop` graph), logits for `b.logits_rows`.
    fn pass(
        &self,
        s: &Stream,
        b: &StepBatch,
        st: &StateBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        let ink = &self.ink;
        let rows = i64::from(b.decode_rows + b.prefill_rows);
        let d = i64::from(ink.d_model);
        let ok = rows > 0
            && rows <= self.sc.rows
            && b.token_ids.shape() == [rows]
            && b.positions.shape() == [rows]
            && !b.check;
        if !ok {
            return Err(KError::InvalidArg);
        }
        let wt = |w: &W| w.t(&self.w);
        let per = ink.states() as usize;
        // Embedding and its norm into region 0's first slot.
        let emb = Tensor::new(&self.sc.emb, 0, DType::F32, &[rows, d])?;
        ops::embed(s, &wt(&self.tok_embd)?, &b.token_ids, &emb)?;
        let (mut blocks, mut cur) = (Vec::new(), 0usize);
        ops::rmsnorm_f32(
            s,
            &emb,
            &wt(&self.tok_norm)?,
            ink.rms_eps,
            &self.slot(cur, rows)?,
        )?;
        let (start, end) = (ink.loop_start, ink.loop_end);
        let (mut v, mut region) = (0u32, 0usize);
        let mut visit = |layer, pass, blocks: &mut Vec<usize>, cur: &mut usize, bank, region| {
            let r = self.layer(
                s,
                b,
                st,
                kv,
                v,
                Visit { layer, pass },
                blocks,
                cur,
                bank,
                region,
            );
            v += 1;
            r
        };
        for layer in 0..start {
            visit(layer, 0, &mut blocks, &mut cur, None, region)?;
        }
        // Every pass starts from the prelude, which stays in region 0's first slots: no later layer
        // writes them (the core's first layer starts a block, so its stream takes the next slot).
        let prelude = blocks.len();
        for pass in 0..ink.passes {
            let mut bank = None;
            if pass > 0 {
                // The previous pass's bank, its blocks after the first core block and its output,
                // stays in its region while this pass builds on the other, from the prelude.
                let first = blocks.get(prelude + 1).copied().unwrap_or(cur);
                bank = Some((first, cur + 1 - first));
                region = 1 - region;
                if region == 1 {
                    let src = self.states(0, prelude + 1, rows)?;
                    self.copy_slots(s, &src, per, prelude + 1, rows)?;
                }
                blocks = (region * per..region * per + prelude).collect();
                cur = region * per + prelude;
            }
            for layer in start..end {
                visit(layer, pass, &mut blocks, &mut cur, bank, region)?;
            }
        }
        for layer in end..ink.layers {
            visit(layer, 0, &mut blocks, &mut cur, None, region)?;
        }
        debug_assert_eq!(v as usize, self.visits.len());

        // Head: the depth mix, the logits rows, the norm, the unpadded vocabulary, the soft cap.
        let mixed = Tensor::new(&self.sc.mix, 0, DType::F32, &[rows, 1, d])?;
        ops::depth_mix(
            s,
            &self.states_with(&blocks, cur, rows)?,
            &wt(&self.output_query)?.reshape(&[1, d])?,
            &mixed,
        )?;
        let n = b.logits_rows.shape()[0];
        let head = Tensor::new(&self.sc.head, 0, DType::F32, &[n, d])?;
        ops::embed(s, &mixed.reshape(&[rows, d])?, &b.logits_rows, &head)?;
        let h16 = Tensor::new(&self.sc.head16, 0, DType::F16, &[n, d])?;
        ops::rmsnorm(s, &head, &wt(&self.output_norm)?, ink.rms_eps, &h16)?;
        let out_w = W {
            shape: vec![i64::from(ink.vocab), d],
            ..self.output.clone()
        };
        self.linear(s, &out_w, &h16, logits)?;
        if ink.softcap > 0.0 {
            ops::softcap(s, logits, ink.softcap, logits)?;
        }
        Ok(())
    }

    /// Copies `n` states `src` (f32 `[n, rows, d]`) into slots `first..first + n`, as a gather of
    /// the identity rows.
    fn copy_slots(
        &self,
        s: &Stream,
        src: &Tensor,
        first: usize,
        n: usize,
        rows: i64,
    ) -> Result<(), KError> {
        let d = i64::from(self.ink.d_model);
        let total = n as i64 * rows;
        let ids = Tensor::new(&self.sc.ids_iota, 0, DType::I32, &[total])?;
        let dst = self.states(first, n, rows)?.reshape(&[total, d])?;
        ops::embed(s, &src.reshape(&[total, d])?, &ids, &dst)
    }
}

impl Scratch {
    fn alloc(dev: &Device, s: &Stream, c: &InklingCfg, rows: i64) -> Result<Scratch, KError> {
        let (d, hd) = (i64::from(c.d_model), i64::from(c.head_dim));
        let (h, kvh) = (i64::from(c.heads), i64::from(c.kv_heads));
        let (qd, kvd) = (h * hd, kvh * hd);
        let (e, k, ffn) = (i64::from(c.experts), i64::from(c.used), i64::from(c.ffn));
        let states = i64::from(c.states());
        let f32b = |n: i64| Buffer::alloc(dev, (n.max(1) * 4) as u64);
        let f16b = |n: i64| Buffer::alloc(dev, (n.max(1) * 2) as u64);
        let ones = Buffer::alloc(dev, (d.max(hd) * 4) as u64)?;
        s.upload(&ones, 0, &vec![1.0f32; d.max(hd) as usize])?;
        // The identity rows a slot copy gathers (2 regions of states).
        let iota = Buffer::alloc(dev, (2 * states * rows * 4) as u64)?;
        s.upload(
            &iota,
            0,
            &(0..(2 * states * rows) as i32).collect::<Vec<i32>>(),
        )?;
        Ok(Scratch {
            rows,
            stack: f32b(2 * states * rows * d)?,
            emb: f32b(rows * d)?,
            mix: f32b(rows * CTRLS as i64 * d)?,
            msg: f32b(rows * CTRLS as i64 * d)?,
            ctl: f32b(CTRLS as i64 * rows * d)?,
            cat: f32b(rows * 2 * d)?,
            low: f32b(rows * i64::from(c.gate_rank))?,
            gain: f32b(rows * d)?,
            forget: f32b(rows * d)?,
            h: f16b(3 * rows * d)?,
            h32: f32b(rows * d)?,
            q: f16b(rows * qd)?,
            kraw: f16b(rows * kvd)?,
            vraw: f16b(rows * kvd)?,
            k: f16b(rows * kvd)?,
            v: f16b(rows * kvd)?,
            ktaps: f16b(rows * kvh * i64::from(c.conv))?,
            vtaps: f16b(rows * kvh * i64::from(c.conv))?,
            rel: f16b(rows * h * i64::from(c.rel))?,
            sink: f16b(rows * h)?,
            gate: f16b(rows * qd)?,
            attn: f16b(rows * qd)?,
            o: f16b(rows * d)?,
            logits: f32b(rows * e)?,
            ids: Buffer::alloc(dev, (rows * k * 4) as u64)?,
            weights: f32b(rows * k)?,
            gu: f32b(rows * k * 2 * ffn)?,
            act: f16b(rows * k * ffn)?,
            down: f16b(rows * d)?,
            head: f32b(rows * d)?,
            head16: f16b(rows * d)?,
            ones,
            ids_iota: iota,
        })
    }
}

/// A tensor's file bytes as `to` (its device type): floats converted, quants as stored.
fn convert(raw: &[u8], ggml_type: u32, to: DType) -> Vec<u8> {
    let floats = || -> Vec<f32> {
        match ggml_type {
            F32_TYPE => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            F16_TYPE => raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            _ => raw
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect(),
        }
    };
    match (ggml_type, to) {
        (F32_TYPE, DType::F32) | (F16_TYPE, DType::F16) => raw.to_vec(),
        (_, DType::F32) => floats().iter().flat_map(|x| x.to_le_bytes()).collect(),
        (_, DType::F16) => floats()
            .iter()
            .flat_map(|&x| f32_to_f16(x).to_le_bytes())
            .collect(),
        _ => raw.to_vec(),
    }
}

/// IEEE half to single.
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let (exp, man) = (u32::from(h >> 10) & 0x1f, u32::from(h & 0x3ff));
    let mag = match exp {
        0 => {
            let v = man as f32 * 2f32.powi(-24); // zero or subnormal, exact in f32
            return if sign != 0 { -v } else { v };
        }
        0x1f => 0x7f80_0000 | (man << 13),
        _ => ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(sign | mag)
}

/// Single to half, rounding to nearest even (as the CPU reference's `f32_to_f16`).
fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let (ax, sign) = (x & 0x7fff_ffff, ((x >> 16) & 0x8000) as u16);
    if ax > 0x7f80_0000 {
        return sign | 0x7e00;
    }
    if ax >= 0x477f_f000 {
        return sign | 0x7c00;
    }
    if ax < 0x3880_0000 {
        return sign | (f32::from_bits(ax) * 16_777_216.0).round_ties_even() as u16;
    }
    sign | ((ax + 0xfff + ((ax >> 13) & 1) - 0x3800_0000) >> 13) as u16
}

impl Model for Inkling {
    fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// A forward pass needs each row's short-convolution state: [`Model::forward_state`].
    fn forward(
        &mut self,
        _s: &Stream,
        _b: &StepBatch,
        _kv: &KvPool,
        _logits: &Tensor,
    ) -> Result<(), KError> {
        Err(KError::InvalidArg)
    }

    fn forward_state(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        st: &StateBatch,
        kv: &KvPool,
        logits: &Tensor,
        taps: Option<&Taps>,
    ) -> Result<(), KError> {
        if taps.is_some() {
            return Err(KError::Unsupported);
        }
        self.pass(s, b, st, kv, logits)
    }

    fn state_slots(&self) -> u32 {
        self.state.as_ref().map_or(0, |(_, n)| *n)
    }

    fn reset_state(&mut self, s: &Stream, slot: u32) -> Result<(), KError> {
        let (buf, _) = self.state.as_ref().ok_or(KError::InvalidArg)?;
        for (off, bytes) in self.slot_ranges(slot)? {
            s.upload(buf, off, &vec![0u8; bytes as usize])?;
        }
        Ok(())
    }

    fn copy_state(&mut self, s: &Stream, from: u32, to: u32) -> Result<(), KError> {
        let mut bytes = vec![0u8; self.state_size()];
        self.save_state(s, from, &mut bytes)?;
        self.load_state(s, to, &bytes)
    }

    fn state_size(&self) -> usize {
        self.slot_bytes() as usize
    }

    fn save_state(&self, s: &Stream, slot: u32, out: &mut [u8]) -> Result<(), KError> {
        if out.len() != self.state_size() {
            return Err(KError::InvalidArg);
        }
        let (buf, _) = self.state.as_ref().ok_or(KError::InvalidArg)?;
        let mut at = 0;
        for (off, bytes) in self.slot_ranges(slot)? {
            s.download(buf, off, &mut out[at..at + bytes as usize])?;
            at += bytes as usize;
        }
        s.sync()
    }

    fn load_state(&mut self, s: &Stream, slot: u32, bytes: &[u8]) -> Result<(), KError> {
        if bytes.len() != self.state_size() {
            return Err(KError::InvalidArg);
        }
        let (buf, _) = self.state.as_ref().ok_or(KError::InvalidArg)?;
        let mut at = 0;
        for (off, n) in self.slot_ranges(slot)? {
            s.upload(buf, off, &bytes[at..at + n as usize])?;
            at += n as usize;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_convert_as_their_types() {
        // bf16 1.0 and 0.1 to f16, f16 to f32.
        assert_eq!(
            convert(&[0x80, 0x3f, 0xcd, 0x3d], BF16_TYPE, DType::F16),
            [0x00, 0x3c, 0x68, 0x2e]
        );
        assert_eq!(
            convert(&[0x00, 0x3c], F16_TYPE, DType::F32),
            1.0f32.to_le_bytes()
        );
    }

    #[test]
    fn halves_round_trip_as_the_reference_rounds() {
        for x in [0.0f32, -0.0, 1.0, -2.5, 65504.0, 6.1e-5, 3e-8, 1e-3, 0.1] {
            let h = f32_to_f16(x);
            assert_eq!(f32_to_f16(f16_to_f32(h)), h, "{x}");
        }
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00); // a tie rounds to even
    }

    #[test]
    fn visits_follow_llama_cpps_loop() {
        let mut c = InklingCfg::from_gguf(&Gguf::parse(&crate::tiny::inkling_gguf(7)).unwrap())
            .unwrap()
            .unwrap();
        let layers: Vec<u32> = visits(&c).iter().map(|v| v.layer).collect();
        assert_eq!(layers, [0, 1, 2, 3, 2, 3, 2, 3, 4, 5]);
        // The released model: 6 + 12 x 3 + 6 = 48 visits.
        (c.layers, c.loop_start, c.loop_end, c.passes) = (24, 6, 18, 3);
        assert_eq!(visits(&c).len(), 48);
    }
}
