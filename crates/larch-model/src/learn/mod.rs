//! The native router-delta trainer (doc I4, MoE Rewiring style test-time training): a qwen3moe model's
//! suffix, layers `first..n_layers`, the final norm and the head, run forward and backward in f32 on
//! the CPU with every weight frozen, optimizing one f32 delta per selected MoE layer that is added to
//! that layer's router logits before softmax and top k, or (doc I5, [`Adapt`]) rank-r adapters on the
//! down projection of each layer's most-routed experts with the router frozen. Weights stay quantized in host memory and are
//! dequantized by row block through the CPU kernels' `embed`, so no layer is ever whole in f32.
// ponytail: CPU f32 reference, minutes per step on the target MoE's suffix; the CUDA path is `device`
// (parity: device_trainer_matches_the_reference); add HIP backward kernels when an AMD card must learn.

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use larch_kernels::{Buffer, CPU, DType, Device, KError, Stream, Tensor, ops};
use larch_records::LoadError;

use crate::gguf::{self, Gguf, code};
use crate::{Arch, ModelConfig, MoeCfg};

mod device;
pub use device::DeviceSuffix;

/// Adam's default learning rate for a job.
pub const LR: f32 = 0.01;
/// The default global gradient-norm clip.
pub const CLIP: f32 = 1.0;
/// The persistent mode's pull toward the parent: `STABILITY * mean((delta - parent)^2)`.
pub const STABILITY: f64 = 0.001;
/// Rows dequantized per block.
const BLOCK: usize = 64;
/// Worker threads at most.
const MAX_THREADS: usize = 16;

/// Why a job or a pass failed.
#[derive(Clone, Debug, PartialEq)]
pub enum LearnError {
    Load(LoadError),
    Kernel(KError),
    /// An input that does not fit the model or the contract; names the field.
    Invalid(&'static str),
    /// A loss, gradient or delta that is not finite; names which.
    NonFinite(&'static str),
}

impl fmt::Display for LearnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LearnError::Load(e) => write!(f, "load: {e:?}"),
            LearnError::Kernel(e) => write!(f, "kernel: {e}"),
            LearnError::Invalid(what) => write!(f, "invalid {what}"),
            LearnError::NonFinite(what) => write!(f, "non-finite {what}"),
        }
    }
}

impl std::error::Error for LearnError {}

impl From<LoadError> for LearnError {
    fn from(e: LoadError) -> Self {
        LearnError::Load(e)
    }
}

impl From<KError> for LearnError {
    fn from(e: KError) -> Self {
        LearnError::Kernel(e)
    }
}

/// What a job reports.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    /// Masked cross-entropy at the starting deltas (the stability term excluded).
    pub loss_before: f64,
    /// Masked cross-entropy at the returned deltas.
    pub loss_after: f64,
    pub steps: u32,
    pub seconds: f64,
    /// Fraction of (token, selected layer) top-k sets that differ from the starting deltas' sets.
    pub churn: f64,
    /// L2 norm of the returned vector's distance from where the cap is measured: the job's init
    /// (router: zero), or with `cap_from_parent` the parent.
    pub deviation: f64,
}

/// Per-layer router-logit deltas, the fields of `larch-router-candidate/1` (Contract A's
/// `RouterDeltas`), with the job's report. With `rank != 0` it is an expert-adapter candidate instead:
/// layer `i`'s vector holds, for each expert of `ids[i]`, B `[d_model, rank]` then A `[rank, expert
/// ffn]` of `W_down + B A`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouterCandidate {
    pub layers: Vec<u16>,
    pub experts: u32,
    pub deltas: Vec<Vec<f32>>,
    /// 0 for router deltas; 4 or 8 for expert adapters.
    pub rank: u32,
    /// The adapted experts per layer, ascending; empty for router deltas.
    pub ids: Vec<Vec<u32>>,
    pub stats: Stats,
    /// Not a contract field: the init the cap is measured from, as the trainer started it (empty:
    /// zero router deltas, or B = 0 with the hashed A).
    pub origin: Vec<Vec<f32>>,
}

/// One token window: the residual stream entering the suffix's first layer at every position.
#[derive(Clone, Copy)]
pub struct Window<'a> {
    /// f32 `[tokens, d_model]`.
    pub residual: &'a [f32],
    pub tokens: &'a [u32],
    pub positions: &'a [i32],
    /// Position `i` predicts token `i + 1` where `mask[i + 1]`.
    pub mask: &'a [bool],
    /// The window's share of its batch's loss and gradient against the others' (1: an ordinary row).
    pub weight: f32,
    /// What the window's masked positions train.
    pub row: Row,
}

/// What a window's masked positions train.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Row {
    /// Cross-entropy on the user's verified tokens.
    User,
    /// Cross-entropy on general text (replay).
    Replay,
    /// KL(base || candidate) on general text: the base is the same forward with the deltas off,
    /// run each step (doc I5's frozen-reference penalty).
    Kl,
}

/// Selection (`Select`) scores the distinct windows of a schedule's first batches.
pub const SELECT_BATCHES: usize = 8;

/// A persistent job's steps at most: more total steps over the user's rows is what changes behaviour,
/// and the candidate's time budget bounds them first.
pub const MAX_PERSISTENT_STEPS: u32 = 1024;

/// Temporary jobs start at zero; persistent ones at the parent (`None`: the base, zero) and pay
/// [`STABILITY`] for moving away from it.
#[derive(Clone, Copy)]
pub enum Mode<'a> {
    Temporary,
    Persistent(Option<&'a RouterCandidate>),
}

/// Train low-rank adapters on the down projection of `k` experts of each layer (doc I5) instead of
/// router deltas, which stay frozen.
#[derive(Clone, Copy)]
pub struct Adapt {
    /// 4, 8 or 16.
    pub rank: u32,
    pub k: u32,
    /// LoRA+: B's learning rate over A's (the job's `lr`).
    pub ratio: f32,
    pub select: Select,
    pub init: Init,
}

impl Default for Adapt {
    fn default() -> Self {
        Adapt {
            rank: 8,
            k: 4,
            ratio: 1.0,
            select: Select::Hits,
            init: Init::Hash,
        }
    }
}

/// How a fresh job picks each layer's experts, scored at the supervised rows on the base model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Select {
    /// Most routed.
    Hits,
    /// Highest summed gate probability (ESFT's average gate score).
    Gate,
    /// Largest |dL/dW_down| (Frobenius) on the user rows, sketched as |dL/dB| at B = 0 with the
    /// hashed A.
    Grad,
    /// Highest specificity: the expert's share of the user rows' routes over its share of the replay
    /// rows' (`Row::Replay`), each add-one smoothed, so adapting it moves the user's text more than
    /// general text.
    Specificity,
}

/// A fresh adapter's A (B is always 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Init {
    /// Uniform in ±1/sqrt(ffn) from a hash.
    Hash,
    /// LoRA-GA: orthogonal rows spanning the top right singular vectors of dL/dW_down, found by one
    /// randomized range-finder pass (two extra backwards), at the hashed rows' norm.
    Grad,
}

/// One training job.
pub struct Job<'a> {
    /// Unique MoE layers inside the suffix.
    pub layers: Vec<u16>,
    pub mode: Mode<'a>,
    /// 1, 2, 5 or 8 for a temporary job (the doc's grid); 1 to [`MAX_PERSISTENT_STEPS`] for a
    /// persistent one.
    pub steps: u32,
    pub lr: f32,
    pub clip: f32,
    /// The global L2 bound on all deltas, strict after f32 rounding.
    pub cap: f32,
    /// Checked before each step; the closing evaluation always runs.
    pub deadline: Option<Instant>,
    /// `None` trains router deltas.
    pub adapt: Option<Adapt>,
    /// With a parent, the cap bounds the distance from the parent instead of from the init.
    pub cap_from_parent: bool,
    /// Centres the cap here (job layer order) instead, so a chain of jobs shares one ball.
    pub origin: Option<&'a [Vec<f32>]>,
}

/// One layer's trainable vector as the forward reads it: router-logit deltas (`ids` empty), or the
/// expert adapters of a [`RouterCandidate`].
#[derive(Clone, Copy)]
struct P<'a> {
    v: &'a [f32],
    ids: &'a [u32],
    rank: usize,
}

impl<'a> P<'a> {
    /// The router-logit deltas, unless this layer trains adapters.
    fn router(self) -> Option<&'a [f32]> {
        self.ids.is_empty().then_some(self.v)
    }

    /// Expert `e`'s range in `v`, with its B `[d, rank]` and A `[rank, ffn]`.
    fn adapter(self, e: usize, d: usize) -> Option<(Range<usize>, &'a [f32], &'a [f32])> {
        let i = self.ids.iter().position(|&x| x as usize == e)?;
        let per = self.v.len() / self.ids.len();
        let r = i * per..(i + 1) * per;
        let (b, a) = self.v[r.clone()].split_at(d * self.rank);
        Some((r, b, a))
    }
}

/// A quantized (or f16/f32) row-major matrix held as its file bytes.
struct Mat {
    raw: Vec<u8>,
    dtype: DType,
    cols: usize,
}

impl Mat {
    fn row_bytes(&self) -> usize {
        self.dtype.bytes(self.cols as u64) as usize
    }
}

struct Layer {
    attn_norm: Vec<f32>,
    wq: Mat,
    wk: Mat,
    wv: Mat,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wo: Mat,
    ffn_norm: Vec<f32>,
    /// f32 `[experts, d]`.
    router: Vec<f32>,
    /// `[experts * ffn, d]`, `[experts * ffn, d]`, `[experts * d, ffn]`.
    gate: Mat,
    up: Mat,
    down: Mat,
}

/// A qwen3moe model's layers `first..`, final norm and head, in host memory.
pub struct Suffix {
    cfg: ModelConfig,
    moe: MoeCfg,
    first: u16,
    layers: Vec<Layer>,
    out_norm: Vec<f32>,
    head: Mat,
    rope_freqs: Option<Vec<f32>>,
    dev: Device,
}

/// One token's route: the full softmax, the chosen experts in order, their weights and their
/// probability sum (before the 2^-14 clamp).
struct Route {
    p: Vec<f64>,
    ids: Vec<usize>,
    w: Vec<f32>,
    top: f64,
}

/// A layer's forward activations, kept for the backward pass.
struct Cache {
    x_in: Vec<f32>,
    q_raw: Vec<f32>,
    k_raw: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// Per head `[T, T]`.
    probs: Vec<Vec<f32>>,
    x_mid: Vec<f32>,
    routes: Vec<Route>,
    /// `[T, used, ffn]` gate and up, `[T, used, d]` each chosen expert's output.
    g: Vec<f32>,
    u: Vec<f32>,
    y: Vec<f32>,
}

/// One adapted layer's routing in a pass: `[T, used]` sorted chosen ids, and each expert's gate
/// probability summed over the head rows.
#[derive(Clone)]
struct Sel {
    ids: Vec<usize>,
    gate: Vec<f64>,
}

impl Sel {
    fn of(routes: &[Route], head_rows: &[usize]) -> Sel {
        let mut gate = vec![0f64; routes.first().map_or(0, |r| r.p.len())];
        for &t in head_rows {
            gate.iter_mut().zip(&routes[t].p).for_each(|(g, p)| *g += p);
        }
        let ids = routes
            .iter()
            .flat_map(|r| {
                let mut ids = r.ids.clone();
                ids.sort_unstable();
                ids
            })
            .collect();
        Sel { ids, gate }
    }
}

struct Pass {
    /// f32 `[head rows, vocab]`.
    logits: Vec<f32>,
    x_final: Vec<f32>,
    /// Per adapted layer.
    sel: Vec<Sel>,
    caches: Vec<Cache>,
}

fn threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(MAX_THREADS)
}

/// `f(i)` for `i < n` over the worker threads, results in order.
fn par_map<R: Send>(n: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let mut out: Vec<(usize, R)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads().min(n).max(1))
            .map(|_| {
                sc.spawn(|| {
                    let mut v = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            return v;
                        }
                        v.push((i, f(i)));
                    }
                })
            })
            .collect();
        hs.into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
            .collect()
    });
    out.sort_unstable_by_key(|e| e.0);
    out.into_iter().map(|e| e.1).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    acc.iter().sum::<f32>() + tail
}

fn axpy(y: &mut [f32], a: f32, x: &[f32]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += a * x;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The CPU kernel's rmsnorm of one span: sum of squares in f64, the scale rounded once to f32.
fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let ss: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    (1.0 / (ss / x.len() as f64 + f64::from(eps)).sqrt()) as f32
}

fn rms(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let s = rms_scale(x, eps);
    for i in 0..x.len() {
        out[i] = x[i] * s * w[i];
    }
}

/// `dx += d(rms(x) * w)/dx · dy`.
fn rms_back(x: &[f32], w: &[f32], eps: f32, dy: &[f32], dx: &mut [f32]) {
    let s = rms_scale(x, eps);
    let n = x.len() as f32;
    let c: f32 = (0..x.len()).map(|i| dy[i] * w[i] * x[i]).sum::<f32>() * s * s * s / n;
    for i in 0..x.len() {
        dx[i] += s * w[i] * dy[i] - x[i] * c;
    }
}

/// Every row of `[rows, n]` normed.
fn rms_rows(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (x, o) in x.chunks_exact(w.len()).zip(out.chunks_exact_mut(w.len())) {
        rms(x, w, eps, o);
    }
    out
}

fn add(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
}

/// A row block dequantizer on its own CPU stream: raw rows in, f32 rows out, through `embed`.
struct Dequant {
    s: Stream,
    q: Buffer,
    ids: Buffer,
    out: Buffer,
    host: Vec<f32>,
}

impl Dequant {
    fn new(dev: &Device, m: &Mat) -> Result<Dequant, KError> {
        let s = Stream::new(dev)?;
        let q = Buffer::alloc(dev, (m.row_bytes() * BLOCK) as u64)?;
        let ids = Buffer::alloc(dev, 4 * BLOCK as u64)?;
        let out = Buffer::alloc(dev, (4 * m.cols * BLOCK) as u64)?;
        s.upload(&ids, 0, &(0..BLOCK as i32).collect::<Vec<_>>())?;
        Ok(Dequant {
            s,
            q,
            ids,
            out,
            host: vec![0f32; m.cols * BLOCK],
        })
    }

    fn rows(&mut self, m: &Mat, r0: usize, n: usize) -> Result<&[f32], KError> {
        let rb = m.row_bytes();
        self.s.upload(&self.q, 0, &m.raw[r0 * rb..(r0 + n) * rb])?;
        let (rows, cols) = (n as i64, m.cols as i64);
        let table = Tensor::new(&self.q, 0, m.dtype, &[rows, cols])?;
        let ids = Tensor::new(&self.ids, 0, DType::I32, &[rows])?;
        let out = Tensor::new(&self.out, 0, DType::F32, &[rows, cols])?;
        ops::embed(&self.s, &table, &ids, &out)?;
        let host = &mut self.host[..n * m.cols];
        self.s.download(&self.out, 0, host)?;
        Ok(host)
    }
}

impl Suffix {
    /// Reads layers `first..` of a qwen3moe GGUF, its final norm and head; the shapes and types are
    /// checked as the [`crate::Transformer`] loader checks them.
    pub fn load(path: &Path, first: u16) -> Result<Suffix, LearnError> {
        let file = File::open(path).map_err(|_| LoadError::Io)?;
        let len = file.metadata().map_err(|_| LoadError::Io)?.len();
        let mut f = BufReader::new(file);
        let g = Gguf::read(&mut f, len)?;
        let cfg = ModelConfig::from_gguf(&g)?;
        let moe = MoeCfg::from_gguf(&g, cfg.arch)?.ok_or(LearnError::Invalid("arch"))?;
        if cfg.arch != Arch::Qwen3Moe || u32::from(first) >= cfg.n_layers {
            return Err(LearnError::Invalid("first layer"));
        }
        let (d, hd) = (i64::from(cfg.d_model), i64::from(cfg.head_dim));
        let (qd, kvd) = (i64::from(cfg.n_heads) * hd, i64::from(cfg.n_kv_heads) * hd);
        let (e, ef) = (i64::from(moe.experts), i64::from(moe.ffn));
        let mut read = |name: &str, shape: &[i64]| -> Result<(DType, Vec<u8>), LearnError> {
            let t = g
                .tensor(name)
                .ok_or(LoadError::BadGguf(code::MISSING_TENSOR, 0))?;
            if t.shape() != shape {
                return Err(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset).into());
            }
            let dtype = gguf::dtype(t.ggml_type)?;
            if shape.len() == 1 && dtype != DType::F32 || dtype == DType::BF16 {
                return Err(LoadError::UnsupportedQuant(t.ggml_type).into());
            }
            let mut raw = vec![0u8; t.bytes as usize];
            f.seek(SeekFrom::Start(g.data_offset + t.offset))
                .and_then(|_| f.read_exact(&mut raw))
                .map_err(|_| LoadError::Io)?;
            Ok((dtype, raw))
        };
        let mut mat = |name: &str, shape: &[i64]| -> Result<Mat, LearnError> {
            let (dtype, raw) = read(name, shape)?;
            Ok(Mat {
                raw,
                dtype,
                cols: shape[shape.len() - 1] as usize,
            })
        };
        let vector = |m: Mat| -> Vec<f32> {
            m.raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let dev = Device::open(CPU)?;
        let mut layers = Vec::new();
        for i in first..cfg.n_layers as u16 {
            let n = |s: &str| format!("blk.{i}.{s}");
            let router = mat(&n("ffn_gate_inp.weight"), &[e, d])?;
            if !matches!(router.dtype, DType::F32 | DType::F16) {
                return Err(LearnError::Invalid("router type"));
            }
            layers.push(Layer {
                attn_norm: vector(mat(&n("attn_norm.weight"), &[d])?),
                wq: mat(&n("attn_q.weight"), &[qd, d])?,
                wk: mat(&n("attn_k.weight"), &[kvd, d])?,
                wv: mat(&n("attn_v.weight"), &[kvd, d])?,
                q_norm: vector(mat(&n("attn_q_norm.weight"), &[hd])?),
                k_norm: vector(mat(&n("attn_k_norm.weight"), &[hd])?),
                wo: mat(&n("attn_output.weight"), &[d, qd])?,
                ffn_norm: vector(mat(&n("ffn_norm.weight"), &[d])?),
                router: Suffix::dense(&dev, &router, e as usize)?,
                gate: mat(&n("ffn_gate_exps.weight"), &[e, ef, d])?,
                up: mat(&n("ffn_up_exps.weight"), &[e, ef, d])?,
                down: mat(&n("ffn_down_exps.weight"), &[e, d, ef])?,
            });
        }
        let vocab = i64::from(cfg.vocab);
        let head = match g.tensor("output.weight") {
            Some(_) => mat("output.weight", &[vocab, d])?,
            None => mat("token_embd.weight", &[vocab, d])?,
        };
        let out_norm = vector(mat("output_norm.weight", &[d])?);
        let rope_freqs = match g.tensor("rope_freqs.weight") {
            Some(_) => Some(vector(mat(
                "rope_freqs.weight",
                &[i64::from(cfg.rope.n_dims / 2)],
            )?)),
            None => None,
        };
        Ok(Suffix {
            cfg,
            moe,
            first,
            layers,
            out_norm,
            head,
            rope_freqs,
            dev,
        })
    }

    /// All of a small matrix in f32.
    fn dense(dev: &Device, m: &Mat, rows: usize) -> Result<Vec<f32>, KError> {
        let mut dq = Dequant::new(dev, m)?;
        let mut out = Vec::with_capacity(rows * m.cols);
        for r0 in (0..rows).step_by(BLOCK) {
            out.extend_from_slice(dq.rows(m, r0, BLOCK.min(rows - r0))?);
        }
        Ok(out)
    }

    /// `f` folds each dequantized row block of `rows` (its first row, its values) into a thread's
    /// accumulator; one accumulator per worker.
    fn par_rows<R: Send>(
        &self,
        m: &Mat,
        rows: Range<usize>,
        init: impl Fn() -> R + Sync,
        f: impl Fn(&mut R, usize, &[f32]) + Sync,
    ) -> Result<Vec<R>, LearnError> {
        let next = AtomicUsize::new(rows.start);
        let workers = threads().min(rows.len().div_ceil(BLOCK)).max(1);
        let out: Vec<Result<R, KError>> = par_map(workers, |_| {
            let mut dq = Dequant::new(&self.dev, m)?;
            let mut acc = init();
            loop {
                let r0 = next.fetch_add(BLOCK, Ordering::Relaxed);
                if r0 >= rows.end {
                    return Ok(acc);
                }
                f(&mut acc, r0, dq.rows(m, r0, BLOCK.min(rows.end - r0))?);
            }
        });
        Ok(out.into_iter().collect::<Result<_, _>>()?)
    }

    /// `y[t, o] = x[t] · W[rows.start + o]`: x `[n, cols]` → `[n, rows.len()]`.
    fn mm(&self, m: &Mat, rows: Range<usize>, x: &[f32]) -> Result<Vec<f32>, LearnError> {
        let (c, start, width) = (m.cols, rows.start, rows.len());
        let n = x.len() / c;
        let parts = self.par_rows(m, rows, Vec::new, |acc, r0, b| {
            let k = b.len() / c;
            let mut y = vec![0f32; n * k];
            for t in 0..n {
                for o in 0..k {
                    y[t * k + o] = dot(&x[t * c..(t + 1) * c], &b[o * c..(o + 1) * c]);
                }
            }
            acc.push((r0, y));
        })?;
        let mut out = vec![0f32; n * width];
        for (r0, y) in parts.into_iter().flatten() {
            let k = y.len() / n;
            for t in 0..n {
                out[t * width + r0 - start..][..k].copy_from_slice(&y[t * k..(t + 1) * k]);
            }
        }
        Ok(out)
    }

    /// `dx[t] = Σ_o dy[t, o] W[rows.start + o]`: dy `[n, rows.len()]` → `[n, cols]`.
    fn mm_t(&self, m: &Mat, rows: Range<usize>, dy: &[f32]) -> Result<Vec<f32>, LearnError> {
        let (c, start, width) = (m.cols, rows.start, rows.len());
        let n = dy.len() / width;
        let parts = self.par_rows(
            m,
            rows,
            || vec![0f32; n * c],
            |acc, r0, b| {
                for t in 0..n {
                    for o in 0..b.len() / c {
                        let g = dy[t * width + r0 - start + o];
                        if g != 0.0 {
                            axpy(&mut acc[t * c..(t + 1) * c], g, &b[o * c..(o + 1) * c]);
                        }
                    }
                }
            },
        )?;
        let mut out = vec![0f32; n * c];
        parts.iter().for_each(|p| add(&mut out, p));
        Ok(out)
    }

    /// cos and sin of each pair at `pos`: ggml_rope_ext as the CPU kernel computes it.
    fn rope_cs(&self, pos: i32) -> Vec<f32> {
        let r = &self.cfg.rope;
        let nd = r.n_dims as usize;
        let (ext, attn_factor) = (if r.yarn { 1.0f32 } else { 0.0 }, 1.0f32);
        let corr = |n_rot: f32| {
            nd as f32 * (r.n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * r.freq_base.ln())
        };
        let lo = corr(32.0).floor().max(0.0);
        let hi = corr(1.0).ceil().min((nd - 1) as f32);
        let theta_scale = r.freq_base.powf(-2.0 / nd as f32);
        let mut cs = vec![0f32; nd];
        let mut theta = pos as f32;
        for i0 in (0..nd).step_by(2) {
            let extrap = theta / self.rope_freqs.as_ref().map_or(1.0, |f| f[i0 / 2]);
            let interp = r.freq_scale * extrap;
            let (mut t, mut mscale) = (interp, attn_factor);
            if ext != 0.0 {
                let y = ((i0 / 2) as f32 - lo) / (hi - lo).max(0.001);
                let mix = (1.0 - y.clamp(0.0, 1.0)) * ext;
                t = interp * (1.0 - mix) + extrap * mix;
                mscale *= 1.0 + 0.1 * (1.0 / r.freq_scale).ln();
            }
            cs[i0] = t.cos() * mscale;
            cs[i0 + 1] = t.sin() * mscale;
            theta *= theta_scale;
        }
        cs
    }

    /// Rotates every head of one row in place; `back` applies the transpose (the backward pass).
    fn rope(&self, row: &mut [f32], cs: &[f32], back: bool) {
        let (hd, nd, neox) = (self.cfg.head_dim as usize, cs.len(), self.cfg.rope.neox);
        for h in row.chunks_exact_mut(hd) {
            for i0 in (0..nd).step_by(2) {
                let a = if neox { i0 / 2 } else { i0 };
                let b = if neox { a + nd / 2 } else { a + 1 };
                let (c, s) = (cs[i0], if back { -cs[i0 + 1] } else { cs[i0 + 1] });
                let (x0, x1) = (h[a], h[b]);
                h[a] = x0 * c - x1 * s;
                h[b] = x0 * s + x1 * c;
            }
        }
    }

    /// The forward pass over the window with `deltas` (by suffix layer), the head on `head_rows`.
    fn forward(
        &self,
        win: &Window,
        deltas: &[Option<P>],
        head_rows: &[usize],
        keep: bool,
    ) -> Result<Pass, LearnError> {
        let c = &self.cfg;
        let t_n = win.tokens.len();
        let (d, hd, nh, nkv) = (
            c.d_model as usize,
            c.head_dim as usize,
            c.n_heads as usize,
            c.n_kv_heads as usize,
        );
        let (qd, kvd, eps) = (nh * hd, nkv * hd, c.rms_eps);
        let (used, ef) = (self.moe.used as usize, self.moe.ffn as usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let cs: Vec<Vec<f32>> = win.positions.iter().map(|&p| self.rope_cs(p)).collect();
        let mut x = win.residual.to_vec();
        let (mut sel, mut caches) = (Vec::new(), Vec::new());
        for (li, l) in self.layers.iter().enumerate() {
            let x_in = x.clone();
            let h1 = rms_rows(&x, &l.attn_norm, eps);
            let q_raw = self.mm(&l.wq, 0..qd, &h1)?;
            let k_raw = self.mm(&l.wk, 0..kvd, &h1)?;
            let v = self.mm(&l.wv, 0..kvd, &h1)?;
            let (mut q, mut k) = (
                rms_rows(&q_raw, &l.q_norm, eps),
                rms_rows(&k_raw, &l.k_norm, eps),
            );
            for t in 0..t_n {
                self.rope(&mut q[t * qd..(t + 1) * qd], &cs[t], false);
                self.rope(&mut k[t * kvd..(t + 1) * kvd], &cs[t], false);
            }
            // Per head: causal softmax(q·k * scale) in f64 as the kernel does, then P·v.
            let heads = par_map(nh, |h| {
                let kh = h / (nh / nkv);
                let (mut p, mut o) = (vec![0f32; t_n * t_n], vec![0f32; t_n * hd]);
                let mut s = vec![0f64; t_n];
                for i in 0..t_n {
                    let qi = &q[i * qd + h * hd..][..hd];
                    let mut mx = f64::NEG_INFINITY;
                    for j in 0..=i {
                        s[j] = f64::from(dot(qi, &k[j * kvd + kh * hd..][..hd]) * scale);
                        mx = mx.max(s[j]);
                    }
                    let z: f64 = s[..=i].iter().map(|v| (v - mx).exp()).sum();
                    for j in 0..=i {
                        let pij = ((s[j] - mx).exp() / z) as f32;
                        p[i * t_n + j] = pij;
                        axpy(
                            &mut o[i * hd..(i + 1) * hd],
                            pij,
                            &v[j * kvd + kh * hd..][..hd],
                        );
                    }
                }
                (p, o)
            });
            let mut attn = vec![0f32; t_n * qd];
            for (h, (_, o)) in heads.iter().enumerate() {
                for t in 0..t_n {
                    attn[t * qd + h * hd..][..hd].copy_from_slice(&o[t * hd..(t + 1) * hd]);
                }
            }
            add(&mut x, &self.mm(&l.wo, 0..d, &attn)?);
            let x_mid = x.clone();
            let h2 = rms_rows(&x, &l.ffn_norm, eps);
            let delta = deltas[li];
            let routes = (0..t_n)
                .map(|t| {
                    let lg: Vec<f32> = (0..self.moe.experts as usize)
                        .map(|e| {
                            dot(&h2[t * d..(t + 1) * d], &l.router[e * d..(e + 1) * d])
                                + delta.and_then(|p| p.router()).map_or(0.0, |dl| dl[e])
                        })
                        .collect();
                    route(&lg, used)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (mut g, mut u, mut y) = (
                vec![0f32; t_n * used * ef],
                vec![0f32; t_n * used * ef],
                vec![0f32; t_n * used * d],
            );
            for e in 0..self.moe.experts as usize {
                let slots: Vec<(usize, usize)> = (0..t_n)
                    .flat_map(|t| (0..used).map(move |j| (t, j)))
                    .filter(|&(t, j)| routes[t].ids[j] == e)
                    .collect();
                if slots.is_empty() {
                    continue;
                }
                let xe: Vec<f32> = slots
                    .iter()
                    .flat_map(|&(t, _)| h2[t * d..(t + 1) * d].iter().copied())
                    .collect();
                let ge = self.mm(&l.gate, e * ef..(e + 1) * ef, &xe)?;
                let ue = self.mm(&l.up, e * ef..(e + 1) * ef, &xe)?;
                let ae: Vec<f32> = ge.iter().zip(&ue).map(|(&a, &b)| silu(a) * b).collect();
                let mut ye = self.mm(&l.down, e * d..(e + 1) * d, &ae)?;
                // Before the routing weight scales it, as the base projection's output does.
                if let Some((_, b, a)) = delta.and_then(|p| p.adapter(e, d)) {
                    adapter_add(&mut ye, &ae, b, a, ef);
                }
                for (s, &(t, j)) in slots.iter().enumerate() {
                    let at = t * used + j;
                    g[at * ef..(at + 1) * ef].copy_from_slice(&ge[s * ef..(s + 1) * ef]);
                    u[at * ef..(at + 1) * ef].copy_from_slice(&ue[s * ef..(s + 1) * ef]);
                    y[at * d..(at + 1) * d].copy_from_slice(&ye[s * d..(s + 1) * d]);
                    axpy(
                        &mut x[t * d..(t + 1) * d],
                        routes[t].w[j],
                        &ye[s * d..(s + 1) * d],
                    );
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
                    probs: heads.into_iter().map(|h| h.0).collect(),
                    x_mid,
                    routes,
                    g,
                    u,
                    y,
                });
            }
        }
        let hf: Vec<f32> = head_rows
            .iter()
            .flat_map(|&t| {
                let mut o = vec![0f32; d];
                rms(&x[t * d..(t + 1) * d], &self.out_norm, eps, &mut o);
                o
            })
            .collect();
        let logits = self.mm(&self.head, 0..c.vocab as usize, &hf)?;
        Ok(Pass {
            logits,
            x_final: x,
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
        rows: &[usize],
        pass: &Pass,
        dlogits: &[f32],
    ) -> Result<Vec<Vec<f32>>, LearnError> {
        let c = &self.cfg;
        let t_n = win.tokens.len();
        let (d, hd, nh, nkv) = (
            c.d_model as usize,
            c.head_dim as usize,
            c.n_heads as usize,
            c.n_kv_heads as usize,
        );
        let (qd, kvd, eps) = (nh * hd, nkv * hd, c.rms_eps);
        let (used, ef, ne) = (
            self.moe.used as usize,
            self.moe.ffn as usize,
            self.moe.experts as usize,
        );
        let scale = 1.0 / (hd as f32).sqrt();
        let lowest = deltas.iter().position(Option::is_some).unwrap_or(0);
        let dhf = self.mm_t(&self.head, 0..c.vocab as usize, dlogits)?;
        let mut dx = vec![0f32; t_n * d];
        for (r, &t) in rows.iter().enumerate() {
            let x = &pass.x_final[t * d..(t + 1) * d];
            rms_back(
                x,
                &self.out_norm,
                eps,
                &dhf[r * d..(r + 1) * d],
                &mut dx[t * d..(t + 1) * d],
            );
        }
        let mut grads = Vec::new();
        for li in (lowest..self.layers.len()).rev() {
            let (l, k) = (&self.layers[li], &pass.caches[li]);
            let dlog = router_grad(&k.routes, &k.y, &dx, d, ne);
            let ad = deltas[li].filter(|p| !p.ids.is_empty());
            if deltas[li].is_some() && ad.is_none() {
                grads.push(column_sums(&dlog, ne));
            }
            if li == lowest && ad.is_none() {
                break;
            }
            let mut ga = vec![0f32; ad.map_or(0, |p| p.v.len())];
            // dh2: through the router, then through each chosen expert.
            let mut dh2 = vec![0f32; t_n * d];
            for t in 0..t_n {
                for e in 0..ne {
                    let g = dlog[t * ne + e];
                    if g != 0.0 {
                        axpy(
                            &mut dh2[t * d..(t + 1) * d],
                            g,
                            &l.router[e * d..(e + 1) * d],
                        );
                    }
                }
            }
            for e in 0..ne {
                let slots: Vec<(usize, usize)> = (0..t_n)
                    .flat_map(|t| (0..used).map(move |j| (t, j)))
                    .filter(|&(t, j)| k.routes[t].ids[j] == e)
                    .collect();
                if slots.is_empty() {
                    continue;
                }
                let dy: Vec<f32> = slots
                    .iter()
                    .flat_map(|&(t, j)| {
                        let w = k.routes[t].w[j];
                        dx[t * d..(t + 1) * d].iter().map(move |&g| g * w)
                    })
                    .collect();
                let mut da = self.mm_t(&l.down, e * d..(e + 1) * d, &dy)?;
                if let Some((p, (r, b, a))) = ad.and_then(|p| Some((p, p.adapter(e, d)?))) {
                    let ae: Vec<f32> = slots
                        .iter()
                        .flat_map(|&(t, j)| {
                            let at = (t * used + j) * ef;
                            (at..at + ef).map(|i| silu(k.g[i]) * k.u[i])
                        })
                        .collect();
                    adapter_back(&dy, &ae, b, a, p.rank, &mut da, &mut ga[r]);
                }
                let (mut dg, mut du) = (vec![0f32; da.len()], vec![0f32; da.len()]);
                for (s, &(t, j)) in slots.iter().enumerate() {
                    let at = (t * used + j) * ef;
                    for f in 0..ef {
                        let (gv, uv, a) = (k.g[at + f], k.u[at + f], da[s * ef + f]);
                        let sg = 1.0 / (1.0 + (-gv).exp());
                        dg[s * ef + f] = a * uv * sg * (1.0 + gv * (1.0 - sg));
                        du[s * ef + f] = a * silu(gv);
                    }
                }
                let dxe = self.mm_t(&l.gate, e * ef..(e + 1) * ef, &dg)?;
                let dxu = self.mm_t(&l.up, e * ef..(e + 1) * ef, &du)?;
                for (s, &(t, _)) in slots.iter().enumerate() {
                    add(&mut dh2[t * d..(t + 1) * d], &dxe[s * d..(s + 1) * d]);
                    add(&mut dh2[t * d..(t + 1) * d], &dxu[s * d..(s + 1) * d]);
                }
            }
            if ad.is_some() {
                grads.push(ga);
            }
            // ponytail: at the lowest adapted layer the gate/up backward above is unused work; skip
            // it when a profile shows this layer's share matters.
            if li == lowest {
                break;
            }
            for t in 0..t_n {
                let x = &k.x_mid[t * d..(t + 1) * d];
                let mut g = dx[t * d..(t + 1) * d].to_vec();
                rms_back(x, &l.ffn_norm, eps, &dh2[t * d..(t + 1) * d], &mut g);
                dx[t * d..(t + 1) * d].copy_from_slice(&g);
            }
            // Attention block.
            let dattn = self.mm_t(&l.wo, 0..d, &dx)?;
            let heads = par_map(nh, |h| {
                let kh = h / (nh / nkv);
                let p = &k.probs[h];
                let (mut dq, mut dk, mut dv) = (
                    vec![0f32; t_n * hd],
                    vec![0f32; t_n * hd],
                    vec![0f32; t_n * hd],
                );
                for i in 0..t_n {
                    let doi = &dattn[i * qd + h * hd..][..hd];
                    let dp: Vec<f32> = (0..=i)
                        .map(|j| dot(doi, &k.v[j * kvd + kh * hd..][..hd]))
                        .collect();
                    let s: f32 = (0..=i).map(|j| p[i * t_n + j] * dp[j]).sum();
                    for j in 0..=i {
                        let pij = p[i * t_n + j];
                        axpy(&mut dv[j * hd..(j + 1) * hd], pij, doi);
                        let ds = pij * (dp[j] - s) * scale;
                        axpy(
                            &mut dq[i * hd..(i + 1) * hd],
                            ds,
                            &k.k[j * kvd + kh * hd..][..hd],
                        );
                        axpy(
                            &mut dk[j * hd..(j + 1) * hd],
                            ds,
                            &k.q[i * qd + h * hd..][..hd],
                        );
                    }
                }
                (dq, dk, dv)
            });
            let (mut dq, mut dk, mut dv) = (
                vec![0f32; t_n * qd],
                vec![0f32; t_n * kvd],
                vec![0f32; t_n * kvd],
            );
            for (h, (q, kk, vv)) in heads.iter().enumerate() {
                let kh = h / (nh / nkv);
                for t in 0..t_n {
                    dq[t * qd + h * hd..][..hd].copy_from_slice(&q[t * hd..(t + 1) * hd]);
                    add(
                        &mut dk[t * kvd + kh * hd..][..hd],
                        &kk[t * hd..(t + 1) * hd],
                    );
                    add(
                        &mut dv[t * kvd + kh * hd..][..hd],
                        &vv[t * hd..(t + 1) * hd],
                    );
                }
            }
            // rope, then qk_norm, backward: per row, per head.
            let mut dq_raw = vec![0f32; dq.len()];
            let mut dk_raw = vec![0f32; dk.len()];
            for t in 0..t_n {
                let cs = self.rope_cs(win.positions[t]);
                self.rope(&mut dq[t * qd..(t + 1) * qd], &cs, true);
                self.rope(&mut dk[t * kvd..(t + 1) * kvd], &cs, true);
            }
            for (raw, g, w, out) in [
                (&k.q_raw, &dq, &l.q_norm, &mut dq_raw),
                (&k.k_raw, &dk, &l.k_norm, &mut dk_raw),
            ] {
                for ((x, gy), o) in raw
                    .chunks_exact(hd)
                    .zip(g.chunks_exact(hd))
                    .zip(out.chunks_exact_mut(hd))
                {
                    rms_back(x, w, eps, gy, o);
                }
            }
            let mut dh1 = self.mm_t(&l.wq, 0..qd, &dq_raw)?;
            add(&mut dh1, &self.mm_t(&l.wk, 0..kvd, &dk_raw)?);
            add(&mut dh1, &self.mm_t(&l.wv, 0..kvd, &dv)?);
            for t in 0..t_n {
                rms_back(
                    &k.x_in[t * d..(t + 1) * d],
                    &l.attn_norm,
                    eps,
                    &dh1[t * d..(t + 1) * d],
                    &mut dx[t * d..(t + 1) * d],
                );
            }
        }
        grads.reverse();
        Ok(grads)
    }
}

/// A suffix the trainer runs: the CPU reference ([`Suffix`]) or the model's device ([`DeviceSuffix`]).
trait Net {
    fn cfg(&self) -> &ModelConfig;
    fn moe(&self) -> MoeCfg;
    /// The suffix's first layer.
    fn first(&self) -> u16;
    fn n_layers(&self) -> usize;
    /// f32 logits `[rows.len(), vocab]` with `d` (by suffix layer).
    fn head_logits(
        &self,
        win: &Window,
        d: &[Option<P>],
        rows: &[usize],
    ) -> Result<Vec<f32>, LearnError>;

    /// The window's loss at `d` (`objective_of`; `base`: the base model's logits at `rows`, for a
    /// `Row::Kl` window); with `grad`, its gradient per adapted layer in suffix-layer order; and the
    /// adapted layers' selections.
    #[allow(clippy::type_complexity)]
    fn objective(
        &self,
        win: &Window,
        d: &[Option<P>],
        rows: &[usize],
        grad: bool,
        base: Option<&[f32]>,
    ) -> Result<(f64, Option<Vec<Vec<f32>>>, Vec<Sel>), LearnError>;

    /// The base model's logits at `rows` for a `Row::Kl` window, else None.
    // ponytail: a fresh base forward each step (and the KL row's candidate forward repeats its replay
    // row's); cache the base's top-k per replay row per load when step time matters more than memory.
    fn base(&self, win: &Window, rows: &[usize]) -> Result<Option<Vec<f32>>, LearnError> {
        (win.row == Row::Kl)
            .then(|| self.head_logits(win, &vec![None; self.n_layers()], rows))
            .transpose()
    }

    /// Checks the window against the model; the rows that predict a masked target.
    fn check(&self, win: &Window) -> Result<Vec<usize>, LearnError> {
        let t = win.tokens.len();
        if t < 2 || win.positions.len() != t || win.mask.len() != t {
            return Err(LearnError::Invalid("window lengths"));
        }
        if win.residual.len() != t * self.cfg().d_model as usize {
            return Err(LearnError::Invalid("residual length"));
        }
        if win.residual.iter().any(|x| !x.is_finite()) {
            return Err(LearnError::NonFinite("residual"));
        }
        if !(win.weight.is_finite() && win.weight > 0.0) {
            return Err(LearnError::Invalid("window weight"));
        }
        if win.tokens.iter().any(|&id| id >= self.cfg().vocab)
            || win.positions.iter().any(|&p| p < 0)
        {
            return Err(LearnError::Invalid("token or position"));
        }
        let rows: Vec<usize> = (0..t - 1).filter(|&i| win.mask[i + 1]).collect();
        if rows.is_empty() {
            return Err(LearnError::Invalid("mask: no target"));
        }
        Ok(rows)
    }

    /// The per-suffix-layer deltas of `c`, once its shape is checked.
    fn by_layer<'a>(&self, c: &'a RouterCandidate) -> Result<Vec<Option<P<'a>>>, LearnError> {
        let n = self.n_layers();
        let mut out = vec![None; n];
        if c.experts != self.moe().experts
            || c.deltas.len() != c.layers.len()
            || c.layers.is_empty()
        {
            return Err(LearnError::Invalid("deltas shape"));
        }
        let (d, ef) = (self.cfg().d_model as usize, self.moe().ffn as usize);
        let r = c.rank as usize;
        if ![0, 4, 8, 16].contains(&c.rank)
            || c.ids.len() != if r == 0 { 0 } else { c.layers.len() }
        {
            return Err(LearnError::Invalid("adapter shape"));
        }
        for (i, (&l, v)) in c.layers.iter().zip(&c.deltas).enumerate() {
            let ids = c.ids.get(i).map_or(&[][..], |v| &v[..]);
            // Ascending ids below `experts` are unique; each adapts r (d + ffn) values.
            if r != 0
                && (ids.is_empty()
                    || ids.windows(2).any(|w| w[0] >= w[1])
                    || ids.iter().any(|&e| e >= c.experts)
                    || v.len() != ids.len() * r * (d + ef))
            {
                return Err(LearnError::Invalid("adapter shape"));
            }
            let i = usize::from(
                l.checked_sub(self.first())
                    .ok_or(LearnError::Invalid("layers"))?,
            );
            if i >= n || out[i].is_some() || (r == 0 && v.len() != c.experts as usize) {
                return Err(LearnError::Invalid("layers"));
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err(LearnError::NonFinite("delta"));
            }
            out[i] = Some(P { v, ids, rank: r });
        }
        Ok(out)
    }

    /// Every position's f32 logits `[tokens, vocab]` with `deltas` (None: the base model).
    fn logits(
        &self,
        win: &Window,
        deltas: Option<&RouterCandidate>,
    ) -> Result<Vec<f32>, LearnError> {
        self.check(win)?;
        let d = deltas.map(|c| self.by_layer(c)).transpose()?;
        let d = d.unwrap_or_else(|| vec![None; self.n_layers()]);
        let rows: Vec<usize> = (0..win.tokens.len()).collect();
        self.head_logits(win, &d, &rows)
    }

    /// The window's loss with `deltas` (None: the base model): masked next-token cross-entropy, or
    /// for a `Row::Kl` window KL(base || deltas).
    fn loss(&self, win: &Window, deltas: Option<&RouterCandidate>) -> Result<f64, LearnError> {
        let rows = self.check(win)?;
        let d = deltas.map(|c| self.by_layer(c)).transpose()?;
        let d = d.unwrap_or_else(|| vec![None; self.n_layers()]);
        let base = self.base(win, &rows)?;
        Ok(self.objective(win, &d, &rows, false, base.as_deref())?.0)
    }

    /// The window's loss (`loss`) and, at a candidate's deltas, its gradient.
    fn loss_grad(
        &self,
        win: &Window,
        c: &RouterCandidate,
    ) -> Result<(f64, Vec<Vec<f32>>), LearnError> {
        let rows = self.check(win)?;
        let d = self.by_layer(c)?;
        let base = self.base(win, &rows)?;
        let (loss, grad, _) = self.objective(win, &d, &rows, true, base.as_deref())?;
        // `grad` is in layer order; a layer's rank among the candidate's layers is its index there.
        let grad = grad.unwrap_or_default();
        let out = c
            .layers
            .iter()
            .map(|l| grad[c.layers.iter().filter(|&o| o < l).count()].clone())
            .collect();
        Ok((loss, out))
    }

    /// The vectors the cap is measured from, per layer of `c`: its stored origin, else zero router
    /// deltas, or B = 0 with the hashed A.
    fn origin(&self, c: &RouterCandidate) -> Vec<Vec<f32>> {
        if !c.origin.is_empty() {
            return c.origin.clone();
        }
        let (d, ef) = (self.cfg().d_model as usize, self.moe().ffn as usize);
        (0..c.layers.len())
            .map(|i| match c.ids.get(i) {
                Some(ids) if c.rank != 0 => init_adapters(c.layers[i], ids, c.rank as usize, d, ef),
                _ => vec![0f32; c.deltas[i].len()],
            })
            .collect()
    }

    /// The job's starting point: zero router deltas, or, for adapters, each layer's `k` experts
    /// (scored on the base model at the supervised rows, [`Select`]) with B = 0 and A per [`Init`].
    fn fresh(
        &self,
        wins: &[Window],
        rows: &[Vec<usize>],
        job: &Job,
    ) -> Result<RouterCandidate, LearnError> {
        let (ne, d, ef) = (
            self.moe().experts as usize,
            self.cfg().d_model as usize,
            self.moe().ffn as usize,
        );
        let mut c = RouterCandidate {
            layers: job.layers.clone(),
            experts: self.moe().experts,
            deltas: vec![vec![0f32; ne]; job.layers.len()],
            ..Default::default()
        };
        self.by_layer(&c)?;
        let Some(a) = job.adapt else { return Ok(c) };
        let r = a.rank as usize;
        if a.k == 0 || a.k as usize > ne || ![4, 8, 16].contains(&a.rank) {
            return Err(LearnError::Invalid("experts per layer or rank"));
        }
        if !(a.ratio.is_finite() && a.ratio > 0.0) {
            return Err(LearnError::Invalid("lr ratio"));
        }
        // Adapters with B = 0 are the base model; the gradient at them is the base model's.
        let probe = |ids: Vec<Vec<u32>>, deltas: Vec<Vec<f32>>| {
            let p = RouterCandidate {
                layers: job.layers.clone(),
                experts: self.moe().experts,
                deltas,
                rank: a.rank,
                ids,
                ..Default::default()
            };
            let mut sum: Vec<Vec<f32>> = Vec::new();
            // Every user row counts the same here: weights shape the loss, not which experts adapt.
            // General rows are left out: a KL row's gradient at the base is zero, and a replay row's
            // says nothing about the user's text.
            let ce: Vec<&Window> = wins.iter().filter(|w| w.row == Row::User).collect();
            if ce.is_empty() {
                return Err(LearnError::Invalid("no user rows to select on"));
            }
            for win in &ce {
                let g = self.loss_grad(win, &p)?.1;
                if sum.is_empty() {
                    sum = g.iter().map(|v| vec![0f32; v.len()]).collect();
                }
                for (s, g) in sum.iter_mut().zip(&g) {
                    axpy(s, 1.0 / ce.len() as f32, g);
                }
            }
            Ok::<_, LearnError>(sum)
        };
        let hashed = |ids: &[Vec<u32>]| -> Vec<Vec<f32>> {
            ids.iter()
                .zip(&job.layers)
                .map(|(ids, &l)| init_adapters(l, ids, r, d, ef))
                .collect()
        };
        let specific = a.select == Select::Specificity;
        let score: Vec<Vec<f64>> = if a.select == Select::Grad {
            let all = vec![(0..ne as u32).collect::<Vec<_>>(); job.layers.len()];
            probe(all.clone(), hashed(&all))?
                .iter()
                .map(|g| {
                    g.chunks_exact(r * (d + ef))
                        .map(|e| f64::from(dot(&e[..d * r], &e[..d * r])))
                        .collect()
                })
                .collect()
        } else {
            // A zero router delta is the base model; it only makes the forward report its routes.
            let (zero, mut at) = (vec![0f32; ne], vec![None; self.n_layers()]);
            for &l in &job.layers {
                at[usize::from(l - self.first())] = Some(P {
                    v: &zero,
                    ids: &[],
                    rank: 0,
                });
            }
            let mut order: Vec<usize> = (0..job.layers.len()).collect();
            order.sort_unstable_by_key(|&i| job.layers[i]);
            // Per window kind (user, general): each layer's summed score per expert.
            let mut score = [0; 2].map(|_| vec![vec![0f64; ne]; job.layers.len()]);
            // A KL row routes as the replay row it scores again: only user and replay rows count.
            for (win, rows) in wins.iter().zip(rows).filter(|(w, _)| w.row != Row::Kl) {
                let (_, _, sel) = self.objective(win, &at, rows, false, None)?;
                let kind = usize::from(win.row == Row::Replay && specific);
                for (s, &i) in sel.iter().zip(&order) {
                    let add = match a.select {
                        Select::Gate => s.gate.clone(),
                        _ => hits(&s.ids, self.moe().used as usize, rows, ne),
                    };
                    score[kind][i]
                        .iter_mut()
                        .zip(add)
                        .for_each(|(x, y)| *x += y);
                }
            }
            let [user, general] = score;
            if !specific {
                user
            } else {
                user.iter()
                    .zip(&general)
                    .map(|(u, g)| specificity(u, g))
                    .collect()
            }
        };
        c.ids = score.iter().map(|s| top_k(s, a.k as usize)).collect();
        c.deltas = hashed(&c.ids);
        if a.init == Init::Grad {
            // Y = G Aᵀ from dB at (B = 0, hashed A); Q = orth(Y); Z = Qᵀ G from dA at (B = Q, A = 0).
            let y = probe(c.ids.clone(), c.deltas.clone())?;
            let mut qa = c.deltas.clone();
            for (v, y) in qa.iter_mut().zip(&y) {
                for (e, y) in v
                    .chunks_exact_mut(r * (d + ef))
                    .zip(y.chunks_exact(r * (d + ef)))
                {
                    let mut q = transpose(&y[..d * r], d, r);
                    orthonormalize(&mut q, d, 1.0);
                    e[..d * r].copy_from_slice(&transpose(&q, r, d));
                    e[d * r..].fill(0.0);
                }
            }
            let z = probe(c.ids.clone(), qa)?;
            // Same row norm as the hashed A: E|row|^2 = ffn * bound^2 / 3 = 1/3.
            let scale = (1.0f32 / 3.0).sqrt();
            for (v, z) in c.deltas.iter_mut().zip(&z) {
                for (e, z) in v
                    .chunks_exact_mut(r * (d + ef))
                    .zip(z.chunks_exact(r * (d + ef)))
                {
                    let mut a_rows = z[d * r..].to_vec();
                    // An expert no supervised row reaches has no gradient: keep its hashed A.
                    if orthonormalize(&mut a_rows, ef, scale) {
                        e[d * r..].copy_from_slice(&a_rows);
                    }
                }
            }
        }
        c.rank = a.rank;
        c.origin = c.deltas.clone();
        self.by_layer(&c)?;
        Ok(c)
    }

    /// Runs a job over a schedule of window batches (step `s` on batch `s % len`): Adam (fresh
    /// state), global clip, projection to the cap after each step; selection is recomputed every step.
    fn train(&self, schedule: &[&[Window]], job: &Job) -> Result<RouterCandidate, LearnError> {
        let started = Instant::now();
        if schedule.iter().any(|b| b.is_empty()) || schedule.is_empty() {
            return Err(LearnError::Invalid("schedule"));
        }
        let rows = schedule
            .iter()
            .map(|b| b.iter().map(|w| self.check(w)).collect())
            .collect::<Result<Vec<Vec<_>>, _>>()?;
        let steps_ok = match job.mode {
            Mode::Temporary => [1, 2, 5, 8].contains(&job.steps),
            Mode::Persistent(_) => (1..=MAX_PERSISTENT_STEPS).contains(&job.steps),
        };
        if !steps_ok {
            return Err(LearnError::Invalid("steps"));
        }
        let ok = |x: f32| x.is_finite() && x > 0.0;
        if !ok(job.lr) || !ok(job.clip) || !ok(job.cap) {
            return Err(LearnError::Invalid("lr, clip or cap"));
        }
        // Selection scores the distinct windows of the first batches (a row repeats across steps).
        let (mut pick, mut pick_rows): (Vec<Window>, Vec<Vec<usize>>) = (Vec::new(), Vec::new());
        for (b, r) in schedule.iter().zip(&rows).take(SELECT_BATCHES) {
            for (w, r) in b.iter().zip(r) {
                let same = |p: &Window| {
                    std::ptr::eq(p.residual, w.residual)
                        && std::ptr::eq(p.mask, w.mask)
                        && p.row == w.row
                };
                if !pick.iter().any(same) {
                    pick.push(*w);
                    pick_rows.push(r.clone());
                }
            }
        }
        let (start, persistent) = match job.mode {
            Mode::Temporary => (self.fresh(&pick, &pick_rows, job)?, false),
            Mode::Persistent(None) => (self.fresh(&pick, &pick_rows, job)?, true),
            Mode::Persistent(Some(p)) => {
                if p.layers != job.layers {
                    return Err(LearnError::Invalid("parent layers"));
                }
                self.by_layer(p)?;
                if p.rank != job.adapt.map_or(0, |a| a.rank) {
                    return Err(LearnError::Invalid("parent adapters"));
                }
                if !job.cap_from_parent && dist(&p.deltas, &self.origin(p)) > f64::from(job.cap) {
                    return Err(LearnError::Invalid("parent exceeds cap"));
                }
                (p.clone(), true)
            }
        };
        // Work in suffix-layer order; `perm[i]` is the job's index of the i-th adapted layer.
        let mut perm: Vec<usize> = (0..job.layers.len()).collect();
        perm.sort_unstable_by_key(|&i| job.layers[i]);
        let anchor: Vec<Vec<f32>> = perm.iter().map(|&i| start.deltas[i].clone()).collect();
        let mut cur = anchor.clone();
        let init = self.origin(&start);
        let origin_job = match (job.origin, job.mode) {
            (Some(o), _) => o.to_vec(),
            (None, Mode::Persistent(Some(p))) if job.cap_from_parent => p.deltas.clone(),
            _ => init.clone(),
        };
        let origin: Vec<Vec<f32>> = perm.iter().map(|&i| origin_job[i].clone()).collect();
        let count: usize = anchor.iter().map(Vec::len).sum();
        // Batch `b`'s mean loss and gradient (by the windows' weights) and its first window's
        // selections.
        let run = |cur: &[Vec<f32>], grad: bool, b: usize| {
            let mut d = vec![None; self.n_layers()];
            for (k, &i) in perm.iter().enumerate() {
                d[usize::from(job.layers[i] - self.first())] = Some(P {
                    v: &cur[k],
                    ids: start.ids.get(i).map_or(&[][..], |v| &v[..]),
                    rank: start.rank as usize,
                });
            }
            let (total, mut out) = (schedule[b].iter().map(|w| w.weight).sum::<f32>(), None);
            for (win, rows) in schedule[b].iter().zip(&rows[b]) {
                let share = win.weight / total;
                let base = self.base(win, rows)?;
                let (l, g, sel) = self.objective(win, &d, rows, grad, base.as_deref())?;
                let (loss, sum, _) = out.get_or_insert_with(|| {
                    let zero: Option<Vec<Vec<f32>>> = g
                        .as_ref()
                        .map(|g| g.iter().map(|v| vec![0f32; v.len()]).collect());
                    (0.0, zero, sel)
                });
                *loss += l * f64::from(share);
                if let (Some(sum), Some(g)) = (sum.as_mut(), g) {
                    for (s, g) in sum.iter_mut().zip(&g) {
                        axpy(s, share, g);
                    }
                }
            }
            Ok::<_, LearnError>(out.expect("a batch is never empty"))
        };
        let (mut m, mut v): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (
            cur.iter().map(|c| vec![0f32; c.len()]).collect(),
            cur.iter().map(|c| vec![0f32; c.len()]).collect(),
        );
        let (b1, b2, adam_eps) = (0.9f32, 0.999f32, 1e-8f32);
        // LoRA+: each expert's B, its first d * rank values, steps at `ratio` times the lr.
        let (r, d) = (start.rank as usize, self.cfg().d_model as usize);
        let per = r * (d + self.moe().ffn as usize);
        let lr_b = job.lr * job.adapt.map_or(1.0, |a| a.ratio);
        let (mut loss_before, mut base_sel, mut steps) = (None, None, 0);
        // A step is not started when the last one's measured time would pass the deadline.
        let mut last = std::time::Duration::ZERO;
        while steps < job.steps && job.deadline.is_none_or(|d| Instant::now() + last < d) {
            let began = Instant::now();
            let (loss, g, sel) = run(&cur, true, steps as usize % schedule.len())?;
            let mut g = g.unwrap_or_default();
            loss_before.get_or_insert(loss);
            base_sel.get_or_insert(sel);
            if persistent {
                for (k, gk) in g.iter_mut().enumerate() {
                    for (e, g) in gk.iter_mut().enumerate() {
                        *g += (2.0 * STABILITY * f64::from(cur[k][e] - anchor[k][e]) / count as f64)
                            as f32;
                    }
                }
            }
            let gn = norm(&g);
            if !gn.is_finite() {
                return Err(LearnError::NonFinite("gradient"));
            }
            if gn == 0.0 {
                return Err(LearnError::Invalid("zero router gradient"));
            }
            // torch's clip_grad_norm_.
            let coef = f64::from(job.clip) / (gn + 1e-6);
            if coef < 1.0 {
                g.iter_mut()
                    .flatten()
                    .for_each(|x| *x = (f64::from(*x) * coef) as f32);
            }
            steps += 1;
            let (c1, c2) = (1.0 - b1.powi(steps as i32), 1.0 - b2.powi(steps as i32));
            for k in 0..cur.len() {
                for e in 0..cur[k].len() {
                    m[k][e] = b1 * m[k][e] + (1.0 - b1) * g[k][e];
                    v[k][e] = b2 * v[k][e] + (1.0 - b2) * g[k][e] * g[k][e];
                    let lr = if r != 0 && e % per < d * r {
                        lr_b
                    } else {
                        job.lr
                    };
                    cur[k][e] -= lr * (m[k][e] / c1) / ((v[k][e] / c2).sqrt() + adam_eps);
                }
            }
            project(&mut cur, &origin, job.cap)?;
            last = began.elapsed();
        }
        let (loss_after, _, sel) = run(&cur, false, 0)?;
        let loss_before = loss_before.unwrap_or(loss_after);
        let base = base_sel.unwrap_or_else(|| sel.clone());
        let used = self.moe().used as usize;
        let sets = base.iter().map(|s| s.ids.len() / used).sum::<usize>();
        let changed = base
            .iter()
            .zip(&sel)
            .map(|(a, b)| {
                a.ids
                    .chunks_exact(used)
                    .zip(b.ids.chunks_exact(used))
                    .filter(|(x, y)| x != y)
                    .count()
            })
            .sum::<usize>();
        let deviation = dist(&cur, &origin);
        let mut deltas = vec![Vec::new(); cur.len()];
        for (k, &i) in perm.iter().enumerate() {
            deltas[i] = std::mem::take(&mut cur[k]);
        }
        Ok(RouterCandidate {
            layers: job.layers.clone(),
            experts: self.moe().experts,
            deltas,
            rank: start.rank,
            ids: start.ids,
            origin: init,
            stats: Stats {
                loss_before,
                loss_after,
                steps,
                seconds: started.elapsed().as_secs_f64(),
                churn: changed as f64 / sets.max(1) as f64,
                deviation,
            },
        })
    }
}

impl Net for Suffix {
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
        Ok((
            loss,
            Some(self.backward(win, d, rows, &pass, &dlogits)?),
            pass.sel,
        ))
    }
}

/// The trainer's entry points, the same on [`Suffix`] and [`DeviceSuffix`].
macro_rules! entry_points {
    ($t:ty) => {
        impl $t {
            /// The model's config.
            pub fn config(&self) -> &ModelConfig {
                Net::cfg(self)
            }

            /// The model's experts.
            pub fn moe(&self) -> MoeCfg {
                Net::moe(self)
            }

            /// Every position's f32 logits `[tokens, vocab]` with `deltas` (None: the base model).
            pub fn logits(
                &self,
                win: &Window,
                deltas: Option<&RouterCandidate>,
            ) -> Result<Vec<f32>, LearnError> {
                Net::logits(self, win, deltas)
            }

            /// Masked next-token cross-entropy with `deltas` (None: the base model).
            pub fn loss(
                &self,
                win: &Window,
                deltas: Option<&RouterCandidate>,
            ) -> Result<f64, LearnError> {
                Net::loss(self, win, deltas)
            }

            /// Masked next-token cross-entropy and, at a candidate's deltas, its gradient.
            pub fn loss_grad(
                &self,
                win: &Window,
                c: &RouterCandidate,
            ) -> Result<(f64, Vec<Vec<f32>>), LearnError> {
                Net::loss_grad(self, win, c)
            }

            /// Runs a job on one window: Adam (fresh state), global clip, projection to the cap after
            /// each step; selection is recomputed every step.
            pub fn train(&self, win: &Window, job: &Job) -> Result<RouterCandidate, LearnError> {
                Net::train(self, &[std::slice::from_ref(win)], job)
            }

            /// [`Self::train`] over a schedule: step `s` trains on batch `s % schedule.len()`, each of
            /// its windows weighing the same, under one Adam state. Experts are picked on the first
            /// batch, and `loss_before`/`loss_after` are its loss at the start and the end.
            pub fn train_schedule(
                &self,
                schedule: &[&[Window]],
                job: &Job,
            ) -> Result<RouterCandidate, LearnError> {
                Net::train(self, schedule, job)
            }
        }
    };
}

entry_points!(Suffix);
entry_points!(DeviceSuffix<'_>);

/// The window's loss from its `logits` (`[rows.len(), v]`): KL(base || logits) with `base` (the
/// base model's logits at the same rows), else masked cross-entropy; with `grad`, dL/dlogits.
pub(crate) fn objective_of(
    logits: &[f32],
    v: usize,
    win: &Window,
    rows: &[usize],
    grad: bool,
    base: Option<&[f32]>,
) -> Result<(f64, Vec<f32>), LearnError> {
    match base {
        Some(b) => kl(logits, b, v, rows.len(), grad),
        None => ce(logits, v, win, rows, grad),
    }
}

/// Mean over `n` rows of KL(p || q), p = softmax(`base`), q = softmax(`logits`) (`[n, v]` each),
/// and with `grad` dKL/dlogits = (q - p) / n.
fn kl(
    logits: &[f32],
    base: &[f32],
    v: usize,
    n: usize,
    grad: bool,
) -> Result<(f64, Vec<f32>), LearnError> {
    if base.len() != logits.len() || logits.len() != n * v {
        return Err(LearnError::Invalid("base logits"));
    }
    let log_softmax = |x: &[f32]| -> Vec<f64> {
        let mx = f64::from(x.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let lz = mx
            + x.iter()
                .map(|&y| (f64::from(y) - mx).exp())
                .sum::<f64>()
                .ln();
        x.iter().map(|&y| f64::from(y) - lz).collect()
    };
    let mut loss = 0f64;
    let mut dlogits = vec![0f32; if grad { n * v } else { 0 }];
    for r in 0..n {
        let (q, p) = (&logits[r * v..(r + 1) * v], &base[r * v..(r + 1) * v]);
        for (i, (lp, lq)) in log_softmax(p).into_iter().zip(log_softmax(q)).enumerate() {
            let pp = lp.exp();
            loss += pp * (lp - lq);
            if grad {
                dlogits[r * v + i] = ((lq.exp() - pp) / n as f64) as f32;
            }
        }
    }
    loss /= n as f64;
    if !loss.is_finite() {
        return Err(LearnError::NonFinite("loss"));
    }
    Ok((loss, dlogits))
}

/// Masked next-token cross-entropy of `logits` (`[rows.len(), v]`) and, with `grad`, dL/dlogits.
fn ce(
    logits: &[f32],
    v: usize,
    win: &Window,
    rows: &[usize],
    grad: bool,
) -> Result<(f64, Vec<f32>), LearnError> {
    let n = rows.len();
    let mut loss = 0f64;
    let mut dlogits = vec![0f32; if grad { n * v } else { 0 }];
    for (r, &t) in rows.iter().enumerate() {
        let lg = &logits[r * v..(r + 1) * v];
        let mx = f64::from(lg.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let z: f64 = lg.iter().map(|&x| (f64::from(x) - mx).exp()).sum();
        let target = win.tokens[t + 1] as usize;
        loss += mx + z.ln() - f64::from(lg[target]);
        if grad {
            let dl = &mut dlogits[r * v..(r + 1) * v];
            for (o, &x) in dl.iter_mut().zip(lg) {
                *o = ((f64::from(x) - mx).exp() / z / n as f64) as f32;
            }
            dl[target] -= 1.0 / n as f32;
        }
    }
    loss /= n as f64;
    if !loss.is_finite() {
        return Err(LearnError::NonFinite("loss"));
    }
    Ok((loss, dlogits))
}

/// llama.cpp's softmax gating as the CPU `moe_route` computes it: softmax in f64, top k (lower
/// id first on ties), weights over their sum clamped at 2^-14.
fn route(l: &[f32], used: usize) -> Result<Route, LearnError> {
    if l.iter().any(|x| x.is_nan()) {
        return Err(LearnError::NonFinite("router logit"));
    }
    let mx = f64::from(l.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let z: f64 = l.iter().map(|&x| (f64::from(x) - mx).exp()).sum();
    let p: Vec<f64> = l.iter().map(|&x| (f64::from(x) - mx).exp() / z).collect();
    let mut order: Vec<usize> = (0..l.len()).collect();
    order.sort_by(|&x, &y| l[y].total_cmp(&l[x]).then(x.cmp(&y)));
    order.truncate(used);
    let top: f64 = order.iter().map(|&i| p[i]).sum();
    let norm = 1.0 / top.max(2f64.powi(-14));
    let w = order.iter().map(|&i| (p[i] * norm) as f32).collect();
    Ok(Route {
        p,
        ids: order,
        w,
        top,
    })
}

/// dL/d(router logits) `[T, experts]` from dL/d(MoE block output) `dx` `[T, d]` and each chosen
/// expert's output `y` `[T, used, d]`: out[t] = Σ_j w_tj y_tj, so dw_tj = dout_t · y_tj, through the route.
fn router_grad(routes: &[Route], y: &[f32], dx: &[f32], d: usize, ne: usize) -> Vec<f32> {
    let mut dlog = vec![0f32; routes.len() * ne];
    for (t, r) in routes.iter().enumerate() {
        let used = r.ids.len();
        let dout = &dx[t * d..(t + 1) * d];
        let dw: Vec<f64> = (0..used)
            .map(|j| f64::from(dot(dout, &y[(t * used + j) * d..][..d])))
            .collect();
        route_back(r, &dw, &mut dlog[t * ne..(t + 1) * ne]);
    }
    dlog
}

/// How often each expert is routed at `rows` (the positions whose next token is supervised), from a
/// layer's sorted ids `[T, used]`.
fn hits(sel: &[usize], used: usize, rows: &[usize], ne: usize) -> Vec<f64> {
    let mut n = vec![0f64; ne];
    for &t in rows {
        sel[t * used..(t + 1) * used]
            .iter()
            .for_each(|&e| n[e] += 1.0);
    }
    n
}

/// Each expert's share of the user rows' routes over its share of the general rows', add-one smoothed.
fn specificity(user: &[f64], general: &[f64]) -> Vec<f64> {
    let n = user.len() as f64;
    let (su, sg) = (user.iter().sum::<f64>(), general.iter().sum::<f64>());
    (user.iter().zip(general))
        .map(|(u, g)| ((u + 1.0) / (su + n)) / ((g + 1.0) / (sg + n)))
        .collect()
}

/// The `k` highest-scoring experts, ties to the lower id; ascending.
fn top_k(score: &[f64], k: usize) -> Vec<u32> {
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_by(|&x, &y| score[y].total_cmp(&score[x]).then(x.cmp(&y)));
    let mut top: Vec<u32> = order[..k].iter().map(|&e| e as u32).collect();
    top.sort_unstable();
    top
}

/// `[rows, cols]` to `[cols, rows]`.
fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| m[(i % rows) * cols + i / rows])
        .collect()
}

/// Gram-Schmidt on the rows (length `n`) of `m`, in order, each to norm `scale`. False, with `m`
/// partly changed, when a row is (numerically) in the span of the earlier ones.
fn orthonormalize(m: &mut [f32], n: usize, scale: f32) -> bool {
    let mut q: Vec<Vec<f64>> = Vec::new();
    let mut ok = true;
    for row in m.chunks_exact_mut(n) {
        let mut v: Vec<f64> = row.iter().map(|&x| f64::from(x)).collect();
        let before = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for u in &q {
            let c: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
            v.iter_mut().zip(u).for_each(|(a, b)| *a -= c * b);
        }
        let len = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if len.is_nan() || len <= 1e-6 * before {
            ok = false;
            row.fill(0.0);
            continue;
        }
        v.iter_mut().for_each(|x| *x /= len);
        row.iter_mut()
            .zip(&v)
            .for_each(|(o, &x)| *o = (x * f64::from(scale)) as f32);
        q.push(v);
    }
    ok
}

/// B = 0, so a fresh adapter is the base model, and A uniform in ±1/sqrt(ffn) from a hash of
/// (layer, expert, index): the same start on every device, with no RNG state to carry.
fn init_adapters(layer: u16, ids: &[u32], r: usize, d: usize, ef: usize) -> Vec<f32> {
    let bound = 1.0 / (ef as f32).sqrt();
    ids.iter()
        .flat_map(|&e| {
            let a = (0..r * ef).map(move |i| {
                let mut z = ((u64::from(layer) << 48) ^ (u64::from(e) << 32) ^ i as u64)
                    .wrapping_add(0x9e37_79b9_7f4a_7c15);
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                z ^= z >> 31;
                ((z >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * bound
            });
            std::iter::repeat_n(0.0, d * r).chain(a)
        })
        .collect()
}

/// `y += (ae Aᵀ) Bᵀ` for rows `ae` `[n, ffn]`, A `[r, ffn]`, B `[d, r]`, y `[n, d]`: the adapter's part
/// of an expert's down projection.
pub(crate) fn adapter_add(y: &mut [f32], ae: &[f32], b: &[f32], a: &[f32], ef: usize) {
    let r = a.len() / ef;
    for (y, x) in y.chunks_exact_mut(b.len() / r).zip(ae.chunks_exact(ef)) {
        let h: Vec<f32> = a.chunks_exact(ef).map(|row| dot(x, row)).collect();
        for (o, brow) in y.iter_mut().zip(b.chunks_exact(r)) {
            *o += dot(&h, brow);
        }
    }
}

/// The adapter's backward for one expert: from dL/dy `dy` `[n, d]` and its input `ae` `[n, ffn]`, adds
/// dL/dae to `da` and writes dB `[d, r]` then dA `[r, ffn]` to `g`.
fn adapter_back(
    dy: &[f32],
    ae: &[f32],
    b: &[f32],
    a: &[f32],
    r: usize,
    da: &mut [f32],
    g: &mut [f32],
) {
    let (ef, d) = (a.len() / r, b.len() / r);
    let (gb, ga) = g.split_at_mut(d * r);
    for ((dy, x), da) in dy
        .chunks_exact(d)
        .zip(ae.chunks_exact(ef))
        .zip(da.chunks_exact_mut(ef))
    {
        let ha: Vec<f32> = a.chunks_exact(ef).map(|row| dot(x, row)).collect();
        let mut hb = vec![0f32; r];
        for (&g, (brow, gbrow)) in dy.iter().zip(b.chunks_exact(r).zip(gb.chunks_exact_mut(r))) {
            axpy(&mut hb, g, brow);
            axpy(gbrow, g, &ha);
        }
        for ((&h, arow), garow) in hb
            .iter()
            .zip(a.chunks_exact(ef))
            .zip(ga.chunks_exact_mut(ef))
        {
            axpy(da, h, arow);
            axpy(garow, h, x);
        }
    }
}

/// The sum over rows of `[rows, n]`, in f64.
fn column_sums(m: &[f32], n: usize) -> Vec<f32> {
    let mut sum = vec![0f64; n];
    for row in m.chunks_exact(n) {
        sum.iter_mut()
            .zip(row)
            .for_each(|(s, &x)| *s += f64::from(x));
    }
    sum.into_iter().map(|g| g as f32).collect()
}

/// dL/dlogits of one route from dL/dw of its chosen experts, onto `dl` (zeros or unset elsewhere).
fn route_back(r: &Route, dw: &[f64], dl: &mut [f32]) {
    let used = r.ids.len();
    if r.top >= 2f64.powi(-14) {
        // w_j = softmax over the chosen logits: only those logits get a gradient.
        let s: f64 = (0..used).map(|j| f64::from(r.w[j]) * dw[j]).sum();
        for j in 0..used {
            dl[r.ids[j]] = (f64::from(r.w[j]) * (dw[j] - s)) as f32;
        }
    } else {
        // Clamped: w_j = p_j / 2^-14, through the full softmax.
        let inv = 2f64.powi(14);
        let s: f64 = (0..used).map(|j| r.p[r.ids[j]] * dw[j]).sum();
        for (i, o) in dl.iter_mut().enumerate() {
            *o = (-inv * r.p[i] * s) as f32;
        }
        for j in 0..used {
            dl[r.ids[j]] += (inv * r.p[r.ids[j]] * dw[j]) as f32;
        }
    }
}

/// The global L2 norm of f32 vectors, in f64.
pub fn norm(v: &[Vec<f32>]) -> f64 {
    v.iter()
        .flatten()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt()
}

/// The L2 distance between f32 vectors, in f64.
fn dist(v: &[Vec<f32>], o: &[Vec<f32>]) -> f64 {
    v.iter()
        .flatten()
        .zip(o.iter().flatten())
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Scales `v`'s deviation from `origin` into the L2 ball of radius `cap`, strictly after f32 rounding.
fn project(v: &mut [Vec<f32>], origin: &[Vec<f32>], cap: f32) -> Result<(), LearnError> {
    let n = dist(v, origin);
    if !n.is_finite() {
        return Err(LearnError::NonFinite("delta"));
    }
    let mut scale = f64::from(cap) / n;
    let orig: Vec<Vec<f32>> = v.to_vec();
    while dist(v, origin) > f64::from(cap) {
        for ((o, x), z) in v
            .iter_mut()
            .flatten()
            .zip(orig.iter().flatten())
            .zip(origin.iter().flatten())
        {
            *o = (f64::from(*z) + (f64::from(*x) - f64::from(*z)) * scale) as f32;
        }
        scale *= 1.0 - 1e-7;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specificity_prefers_what_user_rows_route_to_and_general_text_does_not() {
        // Expert 0 is the users' most routed but general text's too; expert 1 is theirs alone.
        let (user, general) = ([40.0, 30.0, 10.0, 0.0], [60.0, 0.0, 5.0, 55.0]);
        assert_eq!(top_k(&specificity(&user, &general), 2), [1, 2]);
        assert_eq!(top_k(&user, 2), [0, 1]);
    }

    /// The projection holds the cap exactly on f32 values, as the candidate reader checks it.
    #[test]
    fn projection_is_strict_after_rounding() {
        let mut v = vec![vec![0.3f32, 0.4, 0.1], vec![1.0, -2.0, 0.25]];
        project(&mut v, &[vec![0.0; 3], vec![0.0; 3]], 0.1).unwrap();
        assert!(norm(&v) <= 0.1);
        assert!(norm(&v) > 0.0999);
    }
}
