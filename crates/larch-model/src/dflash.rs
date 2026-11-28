//! The DFlash block drafter (arXiv 2602.06036; z-lab's `dflash/model_mlx.py`): a few Qwen3-style
//! layers that guess a block of tokens at once. The target's residuals after
//! [`DflashDrafter::target_layers`], concatenated per position, go through `fc` and `hidden_norm`
//! into context K/V of every draft layer, cached per sequence in a [`DraftCache`]. A block is the
//! anchor (the last sampled token, not yet run by the target) and mask tokens after it, attending to
//! the context and bidirectionally to each other; the mask rows' greedy tokens are the guesses. Each
//! block row attends as a decode row over the context and the whole block, so the drafter needs only
//! `attn_decode`, which every backend has, and no masked `larch_attn`. A sliding layer's row is
//! causal: a decode row over its own position's window and the block rows up to itself. The
//! embedding and head are the target's ([`crate::Model::embed_and_head`]). GGUFs come from
//! `scripts/dflash_convert.py`.
//!
//! DFlash2 (Inco's `DFlash2DraftModel`; its math ported from Splash, Apache-2.0: what and how in
//! kernels/src/metal/splash/VENDORED, the licence beside it) adds three parts. Each layer's
//! attention and FFN inputs go through a dynamic conv over the block's rows (`ops::draft_conv`): a
//! channel of a row and of the row before it, each by a base tap plus a weight that the row's norm
//! projects per group of channels, and again on the half's output beside the residual. Its mask rows may
//! read a trained mask embedding. Its guesses come from a selector (`select`): the head's top
//! candidates per position, each scored by its logit plus a low-rank edge from the previous guess.
//! Its block rows see each other in every layer, and a cache whose layers all slide is a ring of
//! the window, not the whole context.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;

use larch_kernels::{Buffer, DType, Device, KError, KvPool, SampleParams, Stream, Tensor, ops};
use larch_records::LoadError;

use crate::gguf::{self, Gguf, Value, code};
use crate::transformer::{backend, fit_err, flat};
use crate::vision::f16_value;

/// Rows per op: mmvq's limit, and the longest block.
const ROWS: i64 = 16;
/// Positions per page of a [`DraftCache`].
const PAGE: u32 = 16;
/// Upload chunk of the weights.
const CHUNK: usize = 64 << 20;

/// A weight's place in [`DflashDrafter::weights`].
#[derive(Clone, Copy)]
struct W {
    off: u64,
    dtype: DType,
    shape: [i64; 2],
    ndim: usize,
}

struct Layer {
    attn_norm: W,
    wq: W,
    wk: W,
    wv: W,
    wo: W,
    q_norm: W,
    k_norm: W,
    ffn_norm: W,
    gate: W,
    up: W,
    down: W,
    /// Causal over [`DraftShape`]'s window, else the whole context and block.
    sliding: bool,
}

/// One sequence's draft context: K and V of every observed position in every draft layer.
pub struct DraftCache {
    k: Buffer,
    v: Buffer,
    /// i32 `[ROWS, pages]`: page i is block i, in one row per block row (attn_decode's table).
    table: Buffer,
    pages: u32,
    len: u32,
    /// Its pages are a ring over the positions (every layer sliding): position p in slot p % slots.
    ring: bool,
    /// The furthest [`DraftCache::truncate`] cuts back and keeps its window (a ring's).
    max_cut: u32,
}

impl DraftCache {
    /// Positions observed so far: the next observed position, and the anchor's.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// No position observed yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Forgets every position from `len` on (rejected rows, or a sequence cut back); a ring cut
    /// back past what it still holds forgets every position.
    pub fn truncate(&mut self, len: u32) {
        self.len = match self.len.saturating_sub(len) > self.max_cut {
            true => 0,
            false => self.len.min(len),
        };
    }
}

/// Activations for [`ROWS`] rows; `aux` holds a call's i32 inputs (gather ids or tokens, positions,
/// each row's attended length in a full and in a sliding layer: [`DflashDrafter::aux`]).
struct Scratch {
    x: Buffer,
    h: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    t: Buffer,
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    /// f32 `[ROWS, taps * d]`, then its f16 norm in `cat16`.
    cat: Buffer,
    cat16: Buffer,
    /// f32 `[ROWS, d]`: fc's output.
    ctx: Buffer,
    logits: Buffer,
    /// sample_prep's top-64 values, ids and LSE.
    top: Buffer,
    aux: Buffer,
    /// f32 ones `[taps * d]`.
    ones: Buffer,
    /// i32 `[ROWS, window_pages]`: a sliding layer's block table, each row from its window's page.
    window_table: Buffer,
    /// DFlash2: f32 `[ROWS, 4 * d / group]`, a conv's dynamic weights, and f32 `[ROWS, rank]`, the
    /// selector's projection.
    dynw: Buffer,
    sel: Buffer,
}

/// A drafter GGUF's layout and weights' places, checked against its header: what a load's plan
/// counts before it loads ([`DraftShape::read`]).
pub struct DraftShape {
    d: i64,
    heads: i64,
    kv_heads: i64,
    head_dim: i64,
    vocab: i64,
    eps: f32,
    rope_base: f32,
    block: u32,
    mask_token: i32,
    target_layers: Vec<u16>,
    /// The sliding layers' window (one for all of them), 0 without.
    window: u32,
    fc: W,
    hidden_norm: W,
    out_norm: W,
    layers: Vec<Layer>,
    ffn: i64,
    /// Where the weights start in the file, and the bytes the device holds (DFlash2's codebooks
    /// past them stay on the host).
    data_offset: u64,
    data: u64,
    v2: Option<V2Shape>,
}

/// What a DFlash2 draft adds to a DFlash one.
struct V2Shape {
    /// Each layer's attention and FFN conv: its base taps (f32 `[4 * d]`: stage, tap, channel)
    /// and dynamic projection (`[4 * d / group, d]`).
    convs: Vec<[(W, W); 2]>,
    group: i64,
    /// `[rank, d]`: the final norm's rows to the selector's rank.
    selector: W,
    rank: i64,
    /// Candidates a position ranks, at most sample_prep's 64.
    top_k: usize,
    /// Q8_0 `[vocab, rank]` codebooks (file offset, bytes), host only.
    pred: (u64, u64),
    succ: (u64, u64),
    /// f32 `[d]`: the trained mask row, else the target's row of the mask token.
    mask: Option<W>,
}

/// DFlash2's host-side weights: the selector's codebooks.
struct V2Host {
    pred: Vec<u8>,
    succ: Vec<u8>,
}

impl DraftShape {
    /// Reads the header of the drafter GGUF at `path`.
    pub fn read(path: &Path) -> Result<DraftShape, LoadError> {
        Ok(open(path)?.1)
    }

    /// Rows of a full block: the anchor and `block - 1` guesses.
    pub fn block(&self) -> u32 {
        self.block
    }

    /// The target layers whose residuals it observes.
    pub fn target_layers(&self) -> &[u16] {
        &self.target_layers
    }

    /// [`DraftCache`] bytes per position: K and V of every layer, f16.
    pub fn cache_bytes_per_position(&self) -> u64 {
        4 * self.layers.len() as u64 * (self.kv_heads * self.head_dim) as u64
    }

    /// Positions a [`DraftCache`] for `max_positions` holds: a ring of the window when every
    /// layer slides, else all of them and a block, in whole pages.
    pub fn cache_positions(&self, max_positions: u32) -> u32 {
        self.cache_pages(max_positions).saturating_mul(PAGE)
    }

    fn ring(&self) -> bool {
        self.layers.iter().all(|l| l.sliding)
    }

    fn cache_pages(&self, max_positions: u32) -> u32 {
        match self.ring() {
            // Every row's window, and its block rows past it, without wrapping onto itself.
            true => window_pages(self.window) + 2,
            false => max_positions
                .saturating_add(self.block)
                .div_ceil(PAGE)
                .max(1),
        }
    }

    /// Device bytes of [`DflashDrafter::load`]: the weights and one set of activations.
    pub fn load_bytes(&self) -> u64 {
        self.data + Scratch::sizes(self).iter().sum::<u64>()
    }
}

/// The GGUF at `path`, to read its weights from, and its shape.
fn open(path: &Path) -> Result<(BufReader<File>, DraftShape), LoadError> {
    let file = File::open(path).map_err(|_| LoadError::Io)?;
    let len = file.metadata().map_err(|_| LoadError::Io)?.len();
    let mut f = BufReader::new(file);
    let g = Gguf::read(&mut f, len)?;
    if g.get("general.architecture").and_then(Value::as_str) != Some("dflash") {
        return Err(LoadError::UnsupportedArch);
    }
    let bad = || LoadError::BadGguf(code::KEY, 0);
    let int = |k: &str| {
        g.get(&format!("dflash.{k}"))
            .and_then(Value::as_u64)
            .filter(|&v| v > 0 && v < 1 << 31)
            .map(|v| v as i64)
            .ok_or_else(bad)
    };
    let float = |k: &str| {
        g.get(&format!("dflash.{k}"))
            .and_then(Value::as_f32)
            .ok_or_else(bad)
    };
    let (d, n_layers, ffn) = (
        int("embedding_length")?,
        int("block_count")?,
        int("feed_forward_length")?,
    );
    let (heads, kv_heads, head_dim) = (
        int("attention.head_count")?,
        int("attention.head_count_kv")?,
        int("attention.key_length")?,
    );
    let (vocab, block) = (int("vocab_size")?, int("block_size")?);
    let mask_token = g
        .get("dflash.mask_token_id")
        .and_then(Value::as_u64)
        .filter(|&m| m < vocab as u64)
        .ok_or_else(bad)? as i32;
    let target_layers = g
        .get("dflash.target_layers")
        .and_then(Value::as_arr)
        .and_then(|a| {
            a.iter()
                .map(|v| v.as_u64().and_then(|v| u16::try_from(v).ok()))
                .collect::<Option<Vec<u16>>>()
        })
        .filter(|t| !t.is_empty())
        .ok_or_else(bad)?;
    if !(2..=ROWS).contains(&block) || heads % kv_heads != 0 || head_dim % 2 != 0 {
        return Err(bad());
    }
    let windows = match g.get("dflash.attention.sliding_window") {
        // Absent is "no windows", not a table sized by `block_count` before any tensor is checked.
        None => Vec::new(),
        Some(v) => v
            .as_arr()
            .and_then(|a| {
                a.iter()
                    .map(|v| v.as_u64().filter(|&v| v <= MAX_WINDOW).map(|v| v as u32))
                    .collect::<Option<Vec<u32>>>()
            })
            .filter(|w| w.len() == n_layers as usize)
            .ok_or_else(bad)?,
    };
    // ponytail: one window for every sliding layer (z-lab's configs); a table per window if a
    // drafter mixes them.
    let window = windows.iter().copied().max().unwrap_or(0);
    if windows.iter().any(|&w| w != 0 && w != window) {
        return Err(bad());
    }

    let find = |name: &str, shape: &[i64]| -> Result<W, LoadError> {
        let t = g
            .tensor(name)
            .ok_or(LoadError::BadGguf(code::MISSING_TENSOR, 0))?;
        if t.shape() != shape {
            return Err(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset));
        }
        let dtype = gguf::dtype(t.ggml_type)?;
        // Vectors f32, matrices quantized (mmvq's weights).
        if (shape.len() == 1) != (dtype == DType::F32) || dtype == DType::F16 {
            return Err(LoadError::UnsupportedQuant(t.ggml_type));
        }
        let mut s = [0; 2];
        s[..shape.len()].copy_from_slice(shape);
        Ok(W {
            off: t.offset,
            dtype,
            shape: s,
            ndim: shape.len(),
        })
    };
    let (qd, kvd, taps) = (
        heads * head_dim,
        kv_heads * head_dim,
        target_layers.len() as i64,
    );
    let layers = (0..n_layers)
        .map(|i| {
            let n = |s: &str| format!("blk.{i}.{s}");
            Ok(Layer {
                attn_norm: find(&n("attn_norm.weight"), &[d])?,
                wq: find(&n("attn_q.weight"), &[qd, d])?,
                wk: find(&n("attn_k.weight"), &[kvd, d])?,
                wv: find(&n("attn_v.weight"), &[kvd, d])?,
                wo: find(&n("attn_output.weight"), &[d, qd])?,
                q_norm: find(&n("attn_q_norm.weight"), &[head_dim])?,
                k_norm: find(&n("attn_k_norm.weight"), &[head_dim])?,
                ffn_norm: find(&n("ffn_norm.weight"), &[d])?,
                gate: find(&n("ffn_gate.weight"), &[ffn, d])?,
                up: find(&n("ffn_up.weight"), &[ffn, d])?,
                down: find(&n("ffn_down.weight"), &[d, ffn])?,
                sliding: windows.get(i as usize).is_some_and(|&w| w > 0),
            })
        })
        .collect::<Result<Vec<_>, LoadError>>()?;
    let v2 = match g.tensor("selector.weight") {
        None => None,
        Some(_) => Some(open_v2(&g, find, d, vocab, n_layers)?),
    };
    // The device holds everything before the host-only codebooks, which the converter writes last.
    let data = match &v2 {
        None => len.saturating_sub(g.data_offset),
        Some(v) => {
            let end = v.pred.0.min(v.succ.0);
            let host = [v.pred.0, v.succ.0];
            if (g.tensors.iter()).any(|t| !host.contains(&t.offset) && t.offset + t.bytes > end) {
                return Err(bad());
            }
            end
        }
    };
    let sh = DraftShape {
        d,
        heads,
        kv_heads,
        head_dim,
        vocab,
        eps: float("attention.layer_norm_rms_epsilon")?,
        rope_base: float("rope.freq_base")?,
        block: block as u32,
        mask_token,
        fc: find("fc.weight", &[d, taps * d])?,
        hidden_norm: find("hidden_norm.weight", &[d])?,
        out_norm: find("output_norm.weight", &[d])?,
        target_layers,
        window,
        layers,
        ffn,
        data_offset: g.data_offset,
        data,
        v2,
    };
    Ok((f, sh))
}

/// DFlash2's tensors and keys: `blk.N.{attn,ffn}_conv.{base,weight}`, `selector.weight`, the
/// `selector.{pred,succ}` codebooks, an optional `mask_embedding`, `dflash.conv_group_size` and
/// `dflash.selector_top_k`.
fn open_v2(
    g: &Gguf,
    find: impl Fn(&str, &[i64]) -> Result<W, LoadError>,
    d: i64,
    vocab: i64,
    n_layers: i64,
) -> Result<V2Shape, LoadError> {
    let bad = || LoadError::BadGguf(code::KEY, 0);
    let key = |k: &str| g.get(&format!("dflash.{k}")).and_then(Value::as_u64);
    let group = key("conv_group_size")
        .filter(|&v| v > 0 && (d as u64).is_multiple_of(v))
        .ok_or_else(bad)? as i64;
    let top_k = key("selector_top_k")
        .filter(|&v| (1..=64).contains(&v) && v <= vocab as u64)
        .ok_or_else(bad)? as usize;
    let selector = g.tensor("selector.weight").ok_or_else(bad)?;
    let rank = selector
        .shape()
        .first()
        .copied()
        // Not zero: `select` chunks the selector by its rank.
        .filter(|&r| r > 0 && r % 32 == 0)
        .ok_or_else(bad)?;
    let selector = find("selector.weight", &[rank, d])?;
    let dyn_rows = 4 * d / group;
    let convs = (0..n_layers)
        .map(|i| {
            let conv = |c: &str| {
                let n = |s: &str| format!("blk.{i}.{c}_conv.{s}");
                Ok((
                    find(&n("base"), &[4 * d])?,
                    find(&n("weight"), &[dyn_rows, d])?,
                ))
            };
            Ok([conv("attn")?, conv("ffn")?])
        })
        .collect::<Result<Vec<_>, LoadError>>()?;
    // The host decodes Q8_0 rows: blocks of 32 values, an f16 scale and 32 i8.
    let codebook = |name: &str| {
        let t = g
            .tensor(name)
            .ok_or(LoadError::BadGguf(code::MISSING_TENSOR, 0))?;
        if t.shape() != [vocab, rank] || gguf::dtype(t.ggml_type)? != DType::Q8_0 {
            return Err(LoadError::BadGguf(code::TENSOR, g.data_offset + t.offset));
        }
        Ok((t.offset, t.bytes))
    };
    let mask = match g.tensor("mask_embedding") {
        None => None,
        Some(_) => Some(find("mask_embedding", &[d])?),
    };
    Ok(V2Shape {
        convs,
        group,
        selector,
        rank,
        top_k,
        pred: codebook("selector.pred")?,
        succ: codebook("selector.succ")?,
        mask,
    })
}

/// The longest window `open` accepts: the window table is sized by it (4 MiB at the limit), not by
/// the GGUF's 32-bit range (8 GiB).
const MAX_WINDOW: u64 = 1 << 20;

/// Pages of a sliding layer's table row: its window, from a page boundary, and a block's rows
/// past its own (DFlash2's rows see the whole block).
fn window_pages(window: u32) -> u32 {
    (window + PAGE - 1).div_ceil(PAGE) + 1
}

impl Scratch {
    /// Each buffer's bytes, in field order.
    fn sizes(sh: &DraftShape) -> [u64; 20] {
        let (d, ffn, taps) = (sh.d, sh.ffn, sh.target_layers.len() as i64);
        let (qd, kvd) = (sh.heads * sh.head_dim, sh.kv_heads * sh.head_dim);
        let buf = |t: DType, cols: i64| t.bytes((ROWS * cols) as u64);
        let (f16, f32) = (DType::F16, DType::F32);
        [
            buf(f32, d),
            buf(f16, d),
            buf(f16, qd),
            buf(f16, kvd),
            buf(f16, kvd),
            buf(f16, qd),
            buf(f16, d),
            buf(f16, ffn),
            buf(f16, ffn),
            buf(f16, ffn),
            buf(f32, taps * d),
            buf(f16, taps * d),
            buf(f32, d),
            buf(f32, sh.vocab),
            buf(f32, 129),
            buf(DType::I32, taps + 4),
            f32.bytes((taps * d) as u64),
            buf(DType::I32, i64::from(window_pages(sh.window))),
            sh.v2.as_ref().map_or(0, |v| buf(f32, 4 * d / v.group)),
            sh.v2.as_ref().map_or(0, |v| buf(f32, v.rank)),
        ]
    }
}

/// A loaded DFlash drafter: its weights and one set of activations; the sequences' context is in
/// their [`DraftCache`]s.
pub struct DflashDrafter {
    weights: Buffer,
    sh: DraftShape,
    sc: Scratch,
    v2: Option<V2Host>,
}

impl DflashDrafter {
    /// Loads a drafter GGUF onto `dev`.
    pub fn load(path: &Path, dev: &Device, s: &Stream) -> Result<DflashDrafter, LoadError> {
        let (mut f, sh) = open(path)?;
        let data = sh.data;

        // ponytail: uploaded, not wrapped as the target is on Metal; wrap it if the drafter's
        // load time or its second copy in host memory shows.
        let alloc = |b: u64| Buffer::alloc(dev, b.max(1)).map_err(|e| fit_err(dev, b, e));
        let weights = alloc(data)?;
        f.seek(SeekFrom::Start(sh.data_offset))
            .map_err(|_| LoadError::Io)?;
        let mut staging = vec![0u8; CHUNK.min(data as usize)];
        for at in (0..data).step_by(CHUNK) {
            let c = &mut staging[..CHUNK.min((data - at) as usize)];
            f.read_exact(c).map_err(|_| LoadError::Io)?;
            s.upload(&weights, at, c).map_err(backend)?;
        }

        let [
            x,
            h,
            q,
            k,
            v,
            attn,
            t,
            gate,
            up,
            act,
            cat,
            cat16,
            ctx,
            logits,
            top,
            aux,
            ones,
            window_table,
            dynw,
            sel,
        ] = Scratch::sizes(&sh).map(alloc);
        let sc = Scratch {
            x: x?,
            h: h?,
            q: q?,
            k: k?,
            v: v?,
            attn: attn?,
            t: t?,
            gate: gate?,
            up: up?,
            act: act?,
            cat: cat?,
            cat16: cat16?,
            ctx: ctx?,
            logits: logits?,
            top: top?,
            aux: aux?,
            ones: ones?,
            window_table: window_table?,
            dynw: dynw?,
            sel: sel?,
        };
        let n = sh.target_layers.len() * sh.d as usize;
        s.upload(&sc.ones, 0, &vec![1f32; n]).map_err(backend)?;
        let v2 = match &sh.v2 {
            None => None,
            Some(v) => {
                let mut read = |(off, bytes): (u64, u64)| -> Result<Vec<u8>, LoadError> {
                    let mut b = vec![0u8; bytes as usize];
                    f.seek(SeekFrom::Start(sh.data_offset + off))
                        .and_then(|_| f.read_exact(&mut b))
                        .map_err(|_| LoadError::Io)?;
                    Ok(b)
                };
                Some(V2Host {
                    pred: read(v.pred)?,
                    succ: read(v.succ)?,
                })
            }
        };
        Ok(DflashDrafter {
            weights,
            sh,
            sc,
            v2,
        })
    }

    /// The target layers whose output residuals [`DflashDrafter::observe`] takes, in its order: the
    /// engine's [`crate::Taps::layers`].
    pub fn target_layers(&self) -> &[u16] {
        &self.sh.target_layers
    }

    /// Whether it drafts for a target of config `c`: the same width and vocabulary, and target
    /// layers it has.
    pub fn fits(&self, c: &crate::ModelConfig) -> bool {
        self.sh.d == i64::from(c.d_model)
            && self.sh.vocab == i64::from(c.vocab)
            && self
                .sh
                .target_layers
                .iter()
                .all(|&l| u32::from(l) < c.n_layers)
    }

    /// Rows of a full block: the anchor and `block - 1` guesses.
    pub fn block(&self) -> u32 {
        self.sh.block
    }

    /// What its header says.
    pub fn shape(&self) -> &DraftShape {
        &self.sh
    }

    /// An empty cache for up to `max_positions` observed positions.
    pub fn cache(
        &self,
        dev: &Device,
        s: &Stream,
        max_positions: u32,
    ) -> Result<DraftCache, KError> {
        max_positions
            .checked_add(self.sh.block)
            .ok_or(KError::InvalidArg)?;
        let (pages, ring) = (self.sh.cache_pages(max_positions), self.sh.ring());
        let bytes = self.sh.cache_bytes_per_position() / 2 * u64::from(pages * PAGE);
        let table = Buffer::alloc(dev, 4 * ROWS as u64 * u64::from(pages))?;
        let ids: Vec<i32> = (0..ROWS).flat_map(|_| 0..pages as i32).collect();
        s.upload(&table, 0, &ids)?;
        Ok(DraftCache {
            k: Buffer::alloc(dev, bytes)?,
            v: Buffer::alloc(dev, bytes)?,
            table,
            pages,
            len: 0,
            ring,
            // What a ring keeps past its window and the block a propose writes after it.
            max_cut: match ring {
                true => (pages * PAGE).saturating_sub(self.sh.window + 2 * self.sh.block + PAGE),
                false => u32::MAX,
            },
        })
    }

    fn t(&self, w: W) -> Result<Tensor<'_>, KError> {
        Tensor::new(&self.weights, w.off, w.dtype, &w.shape[..w.ndim])
    }

    fn pool<'a>(&self, c: &'a DraftCache) -> KvPool<'a> {
        KvPool {
            k: &c.k,
            v: &c.v,
            n_blocks: c.pages,
            block_tokens: PAGE,
            n_layers: self.sh.layers.len() as u32,
            n_kv_heads: self.sh.kv_heads as u32,
            head_dim: self.sh.head_dim as u32,
            dtype: DType::F16,
        }
    }

    fn rope(&self) -> ops::RopeParams<'static> {
        ops::RopeParams {
            n_dims: self.sh.head_dim as u32,
            neox: true,
            n_ctx_orig: 0,
            freq_base: self.sh.rope_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            freq_factors: None,
        }
    }

    /// The first page a sliding layer's row reads once it has seen `seen` positions: the page
    /// that holds its window's first position.
    // ponytail: up to PAGE - 1 positions more than z-lab's window, as attn_decode reads whole
    // pages; a windowed attn_decode (CUDA's larch_attn refuses windows) if a long context's hit
    // rate shows it.
    fn window_page(&self, seen: u32) -> u32 {
        match self.sh.window {
            0 => 0,
            w => seen.saturating_sub(w) / PAGE,
        }
    }

    /// Uploads a call's i32 inputs into `c`'s positions: `ids` at 0, then `rows` positions from
    /// `at`, then each row's attended length (the context and the whole block), then a sliding
    /// layer's (its own position's window from [`Self::window_page`] and the block rows up to
    /// itself, or DFlash2's whole block), then each row's cache slot; returns (positions, lengths,
    /// sliding lengths, slots).
    fn aux(
        &self,
        s: &Stream,
        c: &DraftCache,
        ids: &[i32],
        at: u32,
        rows: i64,
    ) -> Result<[Tensor<'_>; 4], KError> {
        let taps = self.sh.target_layers.len() as i64;
        let mut v = ids.to_vec();
        v.resize((ROWS * taps) as usize, 0);
        v.extend((0..rows as i32).map(|r| at as i32 + r));
        let end = at + rows as u32;
        v.extend(std::iter::repeat_n(end as i32, rows as usize));
        let seen = (1..=rows as u32).map(|r| at + r);
        let bidir = self.sh.v2.is_some();
        v.extend(seen.map(|n| (if bidir { end } else { n } - self.window_page(n) * PAGE) as i32));
        let slots = c.pages * PAGE;
        v.extend((at..end).map(|p| if c.ring { p % slots } else { p } as i32));
        s.upload(&self.sc.aux, 0, &v)?;
        let i32 = |off: i64, n: i64| Tensor::new(&self.sc.aux, 4 * off as u64, DType::I32, &[n]);
        let p = ROWS * taps;
        Ok([
            i32(p, rows)?,
            i32(p + rows, rows)?,
            i32(p + 2 * rows, rows)?,
            i32(p + 3 * rows, rows)?,
        ])
    }

    /// A sliding layer's block table for a block of `rows` rows from `c.len()`: row r from its
    /// [`Self::window_page`], as wide as the window or the cache.
    fn window_table(&self, s: &Stream, c: &DraftCache, rows: i64) -> Result<Tensor<'_>, KError> {
        let cols = window_pages(self.sh.window).min(c.pages);
        let table: Vec<i32> = (1..=rows as u32)
            .flat_map(|r| {
                let first = self.window_page(c.len + r);
                (first..first + cols).map(|p| match c.ring {
                    true => (p % c.pages) as i32,
                    false => p.min(c.pages - 1) as i32,
                })
            })
            .collect();
        s.upload(&self.sc.window_table, 0, &table)?;
        Tensor::new(
            &self.sc.window_table,
            0,
            DType::I32,
            &[rows, i64::from(cols)],
        )
    }

    /// Appends rows `rows` of `taps` to `c`: f32 `[taps * n, d]` laid out as [`crate::Taps::out`]
    /// writes the [`DflashDrafter::target_layers`] (block k = layer k's `n` rows), for the positions
    /// from `c.len()` on, each a token the target ran and the sequence keeps. One step's taps of
    /// several sequences feed each one's cache by its own rows.
    pub fn observe(
        &self,
        s: &Stream,
        c: &mut DraftCache,
        taps: &Tensor,
        rows: Range<u32>,
    ) -> Result<(), KError> {
        let k = self.sh.target_layers.len() as i64;
        let (d, hd, eps) = (self.sh.d, self.sh.head_dim, self.sh.eps);
        let n = taps.shape()[0] / k;
        let (first, take) = (i64::from(rows.start), rows.len() as u32);
        let fits = taps.dtype() == DType::F32
            && taps.shape() == [n * k, d]
            && rows.start <= rows.end
            && i64::from(rows.end) <= n
            && (c.ring || c.len + take + self.sh.block <= c.pages * PAGE);
        if !fits {
            return Err(KError::InvalidArg);
        }
        let (sc, pool, rope) = (&self.sc, self.pool(c), self.rope());
        // ponytail: 16-row mmvq slices re-read fc, k and v (~130 MB on an 8B drafter) per slice of a
        // prefill chunk; dequant + gemm when a long prompt's prefill profile shows it.
        for a in (0..i64::from(take)).step_by(ROWS as usize) {
            let m = (i64::from(take) - a).min(ROWS);
            // Row r, block j of the gather: tap j of row first + a + r, so `cat` is [m, taps * d].
            let ids: Vec<i32> = (0..m)
                .flat_map(|r| (0..k).map(move |j| (j * n + first + a + r) as i32))
                .collect();
            let [pos, _, _, slots] = self.aux(s, c, &ids, c.len + a as u32, m)?;
            let gather = Tensor::new(&sc.aux, 0, DType::I32, &[m * k])?;
            let cat = Tensor::new(&sc.cat, 0, DType::F32, &[m * k, d])?;
            ops::embed(s, taps, &gather, &cat)?;
            // mmvq reads f16 and no op narrows f32 but a norm: `hidden_norm` after fc is blind to
            // the row's scale, so normalising each concatenated row first changes only its eps.
            let cat16 = Tensor::new(&sc.cat16, 0, DType::F16, &[m, k * d])?;
            let ones = Tensor::new(&sc.ones, 0, DType::F32, &[k * d])?;
            ops::rmsnorm(s, &cat.reshape(&[m, k * d])?, &ones, eps, &cat16)?;
            let ctx = Tensor::new(&sc.ctx, 0, DType::F32, &[m, d])?;
            ops::mmvq(s, &self.t(self.sh.fc)?, &cat16, None, &ctx)?;
            let hc = Tensor::new(&sc.h, 0, DType::F16, &[m, d])?;
            ops::rmsnorm(s, &ctx, &self.t(self.sh.hidden_norm)?, eps, &hc)?;
            let kk = Tensor::new(&sc.k, 0, DType::F16, &[m, self.sh.kv_heads, hd])?;
            let vv = Tensor::new(&sc.v, 0, DType::F16, &[m, self.sh.kv_heads, hd])?;
            for (i, l) in self.sh.layers.iter().enumerate() {
                ops::mmvq(s, &self.t(l.wk)?, &hc, None, &flat(&kk)?)?;
                ops::mmvq(s, &self.t(l.wv)?, &hc, None, &flat(&vv)?)?;
                ops::qk_norm(s, &kk, &self.t(l.k_norm)?, eps)?;
                ops::rope(s, &kk, &pos, &rope)?;
                ops::kv_write(s, &kk, &vv, &slots, &pool, i as u32)?;
            }
        }
        c.len += take;
        Ok(())
    }

    /// Up to `n` greedy guesses for the tokens after `anchor`, the token at position `c.len()`, and
    /// the draft's softmax chance of each: one block of `n + 1` rows (`n < block`), embedded and
    /// projected with the target's `embed` and `head` ([`crate::Model::embed_and_head`]). `before`,
    /// the sequence's tokens before `anchor` (only its last 63 are read), picks the head's rows
    /// ([`draft_vocab`]). Leaves `c`'s positions as they were.
    #[allow(clippy::too_many_arguments)]
    pub fn propose(
        &self,
        s: &Stream,
        c: &DraftCache,
        embed: &Tensor,
        head: &Tensor,
        anchor: u32,
        before: &[u32],
        n: u32,
    ) -> Result<(Vec<u32>, Vec<f32>), KError> {
        let (d, hd, eps) = (self.sh.d, self.sh.head_dim, self.sh.eps);
        let (nh, nkv) = (self.sh.heads, self.sh.kv_heads);
        let v2 = self.sh.v2.as_ref().zip(self.v2.as_ref());
        // DFlash2 runs its whole block: its rows see each other and its convs the row before.
        let rows = match v2 {
            Some(_) => i64::from(self.sh.block),
            None => i64::from(n) + 1,
        };
        let vd = [self.sh.vocab, d];
        let fits = (1..self.sh.block).contains(&n)
            && embed.shape() == vd
            && head.shape() == vd
            && i64::from(anchor) < self.sh.vocab
            && (c.ring || i64::from(c.len) + rows <= i64::from(c.pages * PAGE));
        if !fits {
            return Err(KError::InvalidArg);
        }
        let (sc, pool, rope) = (&self.sc, self.pool(c), self.rope());
        // A trained mask row is row 0 of its own table.
        let mask = v2.and_then(|(v, _)| v.mask);
        let mut ids = vec![anchor as i32];
        ids.resize(rows as usize, mask.map_or(self.sh.mask_token, |_| 0));
        let [pos, lens, window_lens, slots] = self.aux(s, c, &ids, c.len, rows)?;
        let tokens = Tensor::new(&sc.aux, 0, DType::I32, &[rows])?;
        let table = Tensor::new(&c.table, 0, DType::I32, &[rows, i64::from(c.pages)])?;
        let window_table = match self.sh.window {
            0 => None,
            _ => Some(self.window_table(s, c, rows)?),
        };
        let x = Tensor::new(&sc.x, 0, DType::F32, &[rows, d])?;
        let h = Tensor::new(&sc.h, 0, DType::F16, &[rows, d])?;
        let t = Tensor::new(&sc.t, 0, DType::F16, &[rows, d])?;
        let q = Tensor::new(&sc.q, 0, DType::F16, &[rows, nh, hd])?;
        let k = Tensor::new(&sc.k, 0, DType::F16, &[rows, nkv, hd])?;
        let v = Tensor::new(&sc.v, 0, DType::F16, &[rows, nkv, hd])?;
        let attn = Tensor::new(&sc.attn, 0, DType::F16, &[rows, nh, hd])?;
        let ffn = self.sh.layers[0].gate.shape[0];
        let g = Tensor::new(&sc.gate, 0, DType::F16, &[rows, ffn])?;
        let u = Tensor::new(&sc.up, 0, DType::F16, &[rows, ffn])?;
        let act = Tensor::new(&sc.act, 0, DType::F16, &[rows, ffn])?;
        let scale = 1.0 / (hd as f32).sqrt();
        let w = |w: W| self.t(w);
        // LARCH_DRAFT_TILE=1: the block's projections on the check's GEMV (Metal's staged tile reads
        // each weight once for all rows), not decode's.
        let mm = |w: &Tensor, x: &Tensor, out: &Tensor| match draft_tile() {
            true => ops::mmvq_verify(s, w, x, None, out),
            false => ops::mmvq(s, w, x, None, out),
        };
        // Layer i's attention and FFN of `h` into `t`.
        let attend = |i: usize, l: &Layer| -> Result<(), KError> {
            mm(&w(l.wq)?, &h, &flat(&q)?)?;
            mm(&w(l.wk)?, &h, &flat(&k)?)?;
            mm(&w(l.wv)?, &h, &flat(&v)?)?;
            ops::qk_norm(s, &q, &w(l.q_norm)?, eps)?;
            ops::qk_norm(s, &k, &w(l.k_norm)?, eps)?;
            ops::rope(s, &q, &pos, &rope)?;
            ops::rope(s, &k, &pos, &rope)?;
            // The block's K and V sit past the context, where the next observe overwrites them.
            ops::kv_write(s, &k, &v, &slots, &pool, i as u32)?;
            let (rows_table, rows_lens) = match (&window_table, l.sliding) {
                (Some(wt), true) => (wt, &window_lens),
                _ => (&table, &lens),
            };
            ops::attn_decode(
                s, &q, &pool, i as u32, rows_table, rows_lens, 0, scale, &attn,
            )?;
            mm(&w(l.wo)?, &flat(&attn)?, &t)
        };
        let feed = |l: &Layer| -> Result<(), KError> {
            mm(&w(l.gate)?, &h, &g)?;
            mm(&w(l.up)?, &h, &u)?;
            ops::silu_mul(s, &g, &u, &act)?;
            mm(&w(l.down)?, &act, &t)
        };

        match mask {
            None => ops::embed(s, embed, &tokens, &x)?,
            Some(m) => {
                ops::embed(s, embed, &tokens.rows(0, 1)?, &x.rows(0, 1)?)?;
                let m = w(m)?.reshape(&[1, d])?;
                ops::embed(s, &m, &tokens.rows(1, rows - 1)?, &x.rows(1, rows - 1)?)?;
            }
        }
        match v2 {
            None => {
                for (i, l) in self.sh.layers.iter().enumerate() {
                    ops::rmsnorm(s, &x, &w(l.attn_norm)?, eps, &h)?;
                    attend(i, l)?;
                    ops::add(s, &x, &t, &x)?;
                    ops::rmsnorm(s, &x, &w(l.ffn_norm)?, eps, &h)?;
                    feed(l)?;
                    ops::add(s, &x, &t, &x)?;
                }
            }
            Some((v, _)) => {
                let dynw = Tensor::new(&sc.dynw, 0, DType::F32, &[rows, 4, d / v.group])?;
                // One half of a layer: its norm (into `t`) and dynamic weights, the conv's first stage
                // into `h`, `mid` into `t`, and the second stage of `t` added into the residual.
                let half = |norm: W, (base, proj): (W, W), mid: &dyn Fn() -> Result<(), KError>| {
                    let base = w(base)?.reshape(&[4, d])?;
                    ops::rmsnorm(s, &x, &w(norm)?, eps, &t)?;
                    mm(&w(proj)?, &t, &flat(&dynw)?)?;
                    ops::draft_conv(s, &t, &dynw, &base, 0, &h)?;
                    mid()?;
                    ops::draft_conv(s, &t, &dynw, &base, 1, &x)
                };
                for (i, l) in self.sh.layers.iter().enumerate() {
                    let [a, f] = v.convs[i];
                    half(l.attn_norm, a, &|| attend(i, l))?;
                    half(l.ffn_norm, f, &|| feed(l))?;
                }
            }
        }
        // The mask rows only: the anchor's row predicts nothing new.
        let n = i64::from(n);
        let hn = h.rows(0, n)?;
        ops::rmsnorm(s, &x.rows(1, n)?, &w(self.sh.out_norm)?, eps, &hn)?;
        // The head's first `nv` rows only: BPE ids run roughly by merge order, so the low ids hold the common
        // tokens (FR-Spec's frequency-ranked draft vocabulary, as a prefix). A guess past it misses its check.
        let nv = draft_vocab(self.sh.vocab, anchor, before);
        let logits = Tensor::new(&sc.logits, 0, DType::F32, &[n, nv])?;
        mm(&head.rows(0, nv)?, &hn, &logits)?;
        let p = SampleParams {
            temperature: 0.0,
            top_k: v2.map_or(1, |(v, _)| v.top_k as u32),
            repeat_penalty: 1.0,
            penalty_window: 0,
        };
        let vals = Tensor::new(&sc.top, 0, DType::F32, &[n, 64])?;
        let top = Tensor::new(&sc.top, 4 * 64 * n as u64, DType::I32, &[n, 64])?;
        let lse = Tensor::new(&sc.top, 8 * 64 * n as u64, DType::F32, &[n])?;
        ops::sample_prep(s, &logits, &vec![p; n as usize], None, &vals, &top, &lse)?;
        if let Some((v, _)) = v2 {
            let sel = Tensor::new(&sc.sel, 0, DType::F32, &[n, v.rank])?;
            mm(&w(v.selector)?, &hn, &sel)?;
        }
        // sample_prep's values, ids and LSE lie back to back: one download, one sync.
        let g = n as usize;
        let mut raw = vec![0i32; 129 * g];
        s.download(&sc.top, 0, &mut raw)?;
        let f32s =
            |b: &[i32]| -> Vec<f32> { b.iter().map(|&b| f32::from_bits(b as u32)).collect() };
        let (vals, ids, lse) = (
            f32s(&raw[..64 * g]),
            &raw[64 * g..128 * g],
            f32s(&raw[128 * g..]),
        );
        let picked = match v2 {
            None => (0..g).map(|r| (ids[r * 64] as u32, vals[r * 64])).collect(),
            Some((v, host)) => {
                let mut sel = vec![0f32; (n * v.rank) as usize];
                s.download(&sc.sel, 0, &mut sel)?;
                select(v, host, anchor, ids, &vals, &sel)
            }
        };
        // The selector may take a candidate below the top one: its chance is its own logit's.
        Ok(picked
            .into_iter()
            .zip(lse)
            .map(|((id, val), l)| (id, (val - l).exp()))
            .unzip())
    }

    /// The f32 logits of the last [`DflashDrafter::propose`]'s guesses, `[n, vocab]`, where it ran
    /// on the whole vocabulary (tests).
    pub fn last_logits(&self, s: &Stream, n: u32) -> Result<Vec<f32>, KError> {
        let mut v = vec![0f32; (i64::from(n) * self.sh.vocab) as usize];
        s.download(&self.sc.logits, 0, &mut v)?;
        Ok(v)
    }
}

/// Qwen 3.5 and 3.6's 248,320-token vocabulary, the ids below [`QWEN35_DRAFT_VOCAB`] a draft head
/// scores while the text stays inside them, and the specials from [`QWEN35_SPECIALS`] on: Splash's
/// measured subset for these targets (its `kQwenDraftVocabulary`), 40% fewer head rows a round.
const QWEN35_VOCAB: i64 = 248_320;
const QWEN35_DRAFT_VOCAB: i64 = 148_288;
const QWEN35_SPECIALS: i64 = 248_000;

/// How many of the head's first rows a round scores (the rest of the vocabulary is never guessed,
/// and a guess its check would keep past them misses): LARCH_DRAFT_VOCAB=N gives the first N always.
/// Otherwise, for a Qwen 3.5 or 3.6 vocabulary, the first [`QWEN35_DRAFT_VOCAB`] while `anchor` and
/// the last 63 tokens `before` it each lie below it or among the specials, as Splash keeps its subset
/// (Splash: French text, a tenth of whose tokens lie past it, kept 0.2% fewer tokens a round through
/// this window than over the whole vocabulary), else the whole vocabulary.
// ponytail: one prefix view, so unlike Splash's subset head the specials are not scored and a turn's
// end or `</think>` is never guessed; score their 320 rows too if a bench shows it costs kept tokens.
fn draft_vocab(vocab: i64, anchor: u32, before: &[u32]) -> i64 {
    static N: std::sync::OnceLock<Option<i64>> = std::sync::OnceLock::new();
    let n = N.get_or_init(|| std::env::var("LARCH_DRAFT_VOCAB").ok()?.parse().ok());
    if let Some(n) = *n {
        return n.clamp(64, vocab);
    }
    let inside = |&t: &u32| !(QWEN35_DRAFT_VOCAB..QWEN35_SPECIALS).contains(&i64::from(t));
    let window = &before[before.len().saturating_sub(63)..];
    match vocab == QWEN35_VOCAB && inside(&anchor) && window.iter().all(inside) {
        true => QWEN35_DRAFT_VOCAB,
        false => vocab,
    }
}

fn draft_tile() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LARCH_DRAFT_TILE").is_ok_and(|v| v == "1"))
}

/// Row `id` of a Q8_0 `[vocab, rank]` codebook.
fn q8_row(book: &[u8], rank: usize, id: usize) -> impl Iterator<Item = f32> + '_ {
    let blocks = rank / 32;
    book[id * blocks * 34..][..blocks * 34]
        .chunks_exact(34)
        .flat_map(|b| {
            let scale = f16_value(u16::from_le_bytes([b[0], b[1]]));
            b[2..].iter().map(move |&q| scale * f32::from(q as i8))
        })
}

/// DFlash2's selector (Splash's draft_select_edges and draft_select_dflash, greedy): position p
/// takes the candidate of the largest logit plus edge, the edge from the previous guess (the anchor
/// first) being `sum_j pred[prev, j] sel[p, j] succ[cand, j]`. `ids` and `vals` are sample_prep's
/// top-64 rows, of which the first `top_k` are the candidates; ties go to the better-ranked one.
/// Each guess comes with its logit.
fn select(
    v: &V2Shape,
    host: &V2Host,
    anchor: u32,
    ids: &[i32],
    vals: &[f32],
    sel: &[f32],
) -> Vec<(u32, f32)> {
    let rank = v.rank as usize;
    let mut prev = anchor as usize;
    let mut out = Vec::new();
    for (p, s) in sel.chunks(rank).enumerate() {
        let ctx: Vec<f32> = q8_row(&host.pred, rank, prev)
            .zip(s)
            .map(|(a, b)| a * b)
            .collect();
        let mut best = (f32::NEG_INFINITY, None);
        for (&id, &val) in ids[p * 64..][..v.top_k].iter().zip(&vals[p * 64..]) {
            if id < 0 {
                break;
            }
            let edge: f32 = q8_row(&host.succ, rank, id as usize)
                .zip(&ctx)
                .map(|(a, b)| a * b)
                .sum();
            if val + edge > best.0 {
                best = (val + edge, Some((id as usize, val)));
            }
        }
        let Some((id, val)) = best.1 else { break };
        out.push((id as u32, val));
        prev = id;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The subset holds while the anchor and the 63 tokens before it lie below it or among the
    /// specials; one token between, in that window, gives the whole vocabulary back.
    #[test]
    fn the_draft_vocabulary_follows_the_window() {
        if std::env::var_os("LARCH_DRAFT_VOCAB").is_some() {
            return;
        }
        let (v, sub) = (QWEN35_VOCAB, QWEN35_DRAFT_VOCAB);
        let (past, special) = (sub as u32, QWEN35_SPECIALS as u32 + 46);
        let mut text = vec![past; 1];
        text.extend([7u32; 63]);
        assert_eq!(draft_vocab(v, 9, &text), sub, "past the window");
        assert_eq!(draft_vocab(v, 9, &text[..0]), sub, "nothing before");
        assert_eq!(draft_vocab(v, past, &text), v, "the anchor");
        text[1] = past;
        assert_eq!(draft_vocab(v, 9, &text), v, "the window's first");
        text[1] = special;
        assert_eq!(draft_vocab(v, special, &text), sub, "specials");
        assert_eq!(draft_vocab(1000, 9, &text[1..]), 1000, "another vocabulary");
    }
}
