//! Image input through Qwen3-VL's projector (`clip.projector_type = qwen3vl_merger`, the mmproj of
//! Qwen 3.5), as llama.cpp's mtmd builds it at the oracle's tag (tools/mtmd/models/qwen3vl.cpp):
//! - [`QwenPixels`]: the whole image near its own size ([`QwenVisionConfig::fit`], "smart resize":
//!   each side a multiple of 32, the area within a token range), Pillow's bicubic into a black canvas
//!   (PAD_CEIL), normalized and cut into 16-pixel patches in 2 x 2 merge order. No tiles.
//! - [`QwenVision`]: the ViT on the device's ops. The patch conv's two time frames (a still image is
//!   both) and the learned position grid resized bilinearly to the image's grid; per layer LayerNorm,
//!   fused QKV with bias, 2D rope over each patch's (y, x), non-causal attention over every patch,
//!   tanh GELU; the post LayerNorm; then the merger (4 patches into one row, fc, GELU, fc):
//!   [`QwenVisionConfig::tokens_for`] f32 rows.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use larch_kernels::ops::{self, RopeMode, RopeParams};
use larch_kernels::{Buffer, DType, Device, KError, Stream, Tensor};
use larch_records::LoadError;

use crate::gguf::{Gguf, Value, code};
use crate::transformer::{backend, f16_bits, fit_err, flat};
use crate::vision::{
    Plan, Rgb, W, clip_eps, clip_kind, clip_opt, clip_three, clip_u32, merge_order, resize_padded,
};

/// The most rows one image takes. Qwen's own limit is llama.cpp's default, 4096 (1920 x 1080 at its
/// size: 2040 rows); 1600 keeps a screenshot at about its size (1280 x 720: 920 rows) and bounds the
/// encoder, whose attention grows with the square of the patches (1600 rows: 6400 patches).
// ponytail: a fixed cap, as MAX_TILES; the plan's accuracy run sets it, and the gate passes the same
// cap to llama.cpp (--image-max-tokens).
pub const MAX_IMAGE_TOKENS: u32 = 1600;

/// llama.cpp's row limits for a Qwen-VL projector whose file names no pixel limits.
const MIN_TOKENS: u32 = 8;

/// Text-model widths of the 0.8B class (Qwen3.5-0.8B is 1,024 wide). A model this small reads
/// small text better from a larger image, so its default floor is [`SMALL_MIN_TOKENS`] rows: a
/// smaller image is scaled up toward it, at most [`SMALL_MAX_UPSCALE`] a side, and one under
/// [`SMALL_MIN_NATIVE`] rows (a word crop) only to llama.cpp's floor. On the image eval
/// (bench/image-eval) Qwen3.5-0.8B gained 0.032 (35 items better, 10 worse of 256; small fonts,
/// code, scene text) and 0.044 on 180 held-out public items (10 / 2, no category worse), at about
/// 0.4 s more to the first token on the images it scales. The 2B (2,048 wide) gained nothing, so it
/// keeps llama.cpp's fit.
const SMALL_TEXT: std::ops::RangeInclusive<u32> = 512..=1024;
/// See [`SMALL_TEXT`].
pub const SMALL_MIN_TOKENS: u32 = 1024;
/// See [`SMALL_TEXT`].
const SMALL_MAX_UPSCALE: f32 = 2.0;
/// See [`SMALL_TEXT`].
const SMALL_MIN_NATIVE: u32 = 16;

const MAX_TOKENS: u32 = 4096;

/// The text before and after an image's rows in a prompt (llama.cpp's mtmd for Qwen-VL).
pub const IMAGE_BEGIN: &str = "<|vision_start|>";
/// See [`IMAGE_BEGIN`].
pub const IMAGE_END: &str = "<|vision_end|>";

/// The projector's geometry, from the mmproj GGUF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenVisionConfig {
    /// Patch side in pixels (16).
    pub patch: u32,
    /// Merge side (2): merge² patches make one row.
    pub merge: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_head: u32,
    pub n_layer: u32,
    pub eps: f32,
    /// The projector's output width: the text model's d_model.
    pub proj_dim: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Side of the learned position grid in patches (image_size / patch: 48).
    pub pos_side: u32,
    /// The fewest pixels a resized image has (llama.cpp's 8 rows, or the file's).
    pub min_pixels: u32,
    /// The most (4096 rows, or the file's), capped at [`MAX_IMAGE_TOKENS`] rows.
    pub max_pixels: u32,
}

impl QwenVisionConfig {
    /// Reads and range-checks the `clip.*` keys of a Qwen3-VL projector. `UnsupportedArch` for another
    /// projector or one with deepstack layers; `BadGguf(KEY, 0)` for a missing or out-of-range key.
    pub fn from_gguf(g: &Gguf) -> Result<QwenVisionConfig, LoadError> {
        let bad = LoadError::BadGguf(code::KEY, 0);
        clip_kind(g, "qwen3vl_merger")?;
        // Deepstack layers feed rows into the text model's own layers (Qwen3-VL); Qwen 3.5 has none.
        if let Some(v) = g.get("clip.vision.is_deepstack_layers") {
            let a = v.as_arr().ok_or(bad)?;
            if !a.iter().all(|x| matches!(x, Value::Bool(_))) {
                return Err(bad);
            }
            if a.contains(&Value::Bool(true)) {
                return Err(LoadError::UnsupportedArch);
            }
        }
        let u = |k: &str, lo: u64, hi: u64| clip_u32(g, k, lo, hi);
        let (image_size, patch) = (u("image_size", 1, 8192)?, u("patch_size", 1, 64)?);
        let merge = clip_opt(g, "spatial_merge_size", 2, 8)?;
        let area = patch * merge * patch * merge;
        // LARCH_IMAGE_MAX_TOKENS lowers the cap (a tighter card, or faster large images: 1024 on
        // Qwen3.5-0.8B kept its accuracy and halved a large image's time to first token,
        // bench/image-eval/docs/IMAGE-OPT-RESULTS.md).
        let cap = (std::env::var("LARCH_IMAGE_MAX_TOKENS").ok())
            .and_then(|v| v.parse::<u32>().ok())
            .map_or(MAX_IMAGE_TOKENS, |t| t.clamp(MIN_TOKENS, MAX_IMAGE_TOKENS));
        let max_cap = cap * area;
        let pixels = |k: &str, default: u32| clip_opt(g, k, default, u64::from(u32::MAX));
        let proj_dim = u("projection_dim", 1, 65536)?;
        let floor = match SMALL_TEXT.contains(&proj_dim) {
            true => SMALL_MIN_TOKENS.min(cap),
            false => MIN_TOKENS,
        };
        let c = QwenVisionConfig {
            patch,
            merge,
            n_embd: u("embedding_length", 1, 8192)?,
            n_ff: u("feed_forward_length", 1, 65536)?,
            n_head: u("attention.head_count", 1, 256)?,
            n_layer: u("block_count", 1, 256)?,
            eps: clip_eps(g)?,
            proj_dim,
            mean: clip_three(g, "image_mean")?,
            std: clip_three(g, "image_std")?,
            pos_side: image_size / patch,
            min_pixels: pixels("image_min_pixels", floor * area)?,
            max_pixels: pixels("image_max_pixels", MAX_TOKENS * area)?.min(max_cap),
        };
        // Whole patches; llama.cpp's graph is written for a 2 x 2 merge; a head width attn_full
        // takes (64 everywhere, 72 for the 35B's encoder on the CPU reference and Metal); stds that
        // divide; a position grid that resizes (and fits 16k rows); a token range that is one.
        // ponytail: CUDA's attn_full is 64 only, so the engine refuses a 72-wide encoder on CUDA at
        // load; add 72 to attn_full_tc_k when the 35B's images are wanted on CUDA.
        let ok = image_size.is_multiple_of(patch)
            && c.merge == 2
            && c.n_embd.is_multiple_of(c.n_head)
            && matches!(c.n_embd / c.n_head, 64 | 72)
            && c.std.iter().all(|&v| v != 0.0)
            && (2..=128).contains(&c.pos_side)
            && c.min_pixels <= c.max_pixels
            && c.max_pixels >= area;
        if !ok {
            return Err(bad);
        }
        Ok(c)
    }

    /// Values per patch: 3 channels of patch².
    pub fn patch_len(&self) -> u32 {
        3 * self.patch * self.patch
    }

    /// The most rows an image takes: the encoder's scratch holds four patches for each.
    pub fn max_tokens(&self) -> u32 {
        self.max_pixels / (self.patch * self.merge).pow(2)
    }

    /// The size llama.cpp resizes a `w` x `h` image to (calc_size_preserved_ratio, transformers'
    /// smart_resize): each side the nearest multiple of patch · merge, then both scaled down to at
    /// most max_pixels or up to at least min_pixels, the aspect about kept; f32 as the reference.
    pub fn fit(&self, w: u32, h: u32) -> (u32, u32) {
        if w == 0 || h == 0 {
            return (0, 0);
        }
        let f = i64::from(self.patch * self.merge);
        let by = |x: f32, op: fn(f32) -> f32| op(x / f as f32) as i64 * f;
        let (wf, hf) = (w as f32, h as f32);
        let (mut wb, mut hb) = (f.max(by(wf, f32::round)), f.max(by(hf, f32::round)));
        if wb * hb > i64::from(self.max_pixels) {
            let beta = (hf * wf / self.max_pixels as f32).sqrt();
            hb = f.max(by(hf / beta, f32::floor));
            wb = f.max(by(wf / beta, f32::floor));
        } else if wb * hb < i64::from(self.min_pixels) {
            // A small text model's floor ([`SMALL_TEXT`]): at most 2x a side, and a word crop only to
            // llama.cpp's floor.
            let small = SMALL_TEXT.contains(&self.proj_dim);
            let (floor, most) = match (small, wb * hb / (f * f) < i64::from(SMALL_MIN_NATIVE)) {
                (true, true) => (
                    (MIN_TOKENS * (f * f) as u32).min(self.min_pixels),
                    f32::INFINITY,
                ),
                (true, false) => (self.min_pixels, SMALL_MAX_UPSCALE),
                (false, _) => (self.min_pixels, f32::INFINITY),
            };
            if wb * hb < i64::from(floor) {
                let beta = (floor as f32 / (hf * wf)).sqrt().min(most);
                hb = by(hf * beta, f32::ceil);
                wb = by(wf * beta, f32::ceil);
            }
            if wb * hb > i64::from(self.max_pixels) {
                let beta = (hf * wf / self.max_pixels as f32).sqrt();
                hb = f.max(by(hf / beta, f32::floor));
                wb = f.max(by(wf / beta, f32::floor));
            }
        }
        (wb as u32, hb as u32)
    }

    /// Patches across and down for a `w` x `h` image.
    pub fn grid_for(&self, w: u32, h: u32) -> (u32, u32) {
        let (w, h) = self.fit(w, h);
        (w / self.patch, h / self.patch)
    }

    /// Embedding rows of a `w` x `h` image.
    pub fn tokens_for(&self, w: u32, h: u32) -> u32 {
        let (gx, gy) = self.grid_for(w, h);
        gx * gy / (self.merge * self.merge)
    }
}

/// An image cut into the encoder's input: f16 patch rows in merge order (the merge² patches of one
/// output row together), so the merger's concatenation is a reshape.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenPixels {
    /// Patches across and down, each a multiple of the merge side.
    pub gx: u32,
    pub gy: u32,
    /// f16 bits, `[gx * gy, 3 * patch²]`.
    pub patches: Vec<u16>,
}

impl QwenPixels {
    /// `img` as llama.cpp's Qwen-VL preprocessor makes it: resized whole to [`QwenVisionConfig::fit`]
    /// (bicubic, aspect kept, centered on black), normalized, cut into patches. A patch is
    /// c · p² + ky · p + kx of (v / 255 - mean) / std, rounded to f16 (im2col's).
    pub fn new(img: &Rgb, c: &QwenVisionConfig) -> QwenPixels {
        let (w, h) = c.fit(img.width, img.height);
        let r = resize_padded(img, w, h);
        let (p, m) = (c.patch as usize, c.merge as usize);
        let (gx, gy) = (w as usize / p, h as usize / p);
        let planes: [Vec<f32>; 3] = std::array::from_fn(|ch| {
            (r.data.chunks_exact(3))
                .map(|px| (f32::from(px[ch]) / 255.0 - c.mean[ch]) / c.std[ch])
                .collect()
        });
        let mut patches = Vec::with_capacity(gx * gy * c.patch_len() as usize);
        for (px, py) in merge_order(gx, gy, m) {
            for plane in &planes {
                for ky in 0..p {
                    let row = &plane[(py * p + ky) * w as usize + px * p..][..p];
                    patches.extend(row.iter().map(|&v| f16_bits(v)));
                }
            }
        }
        QwenPixels {
            gx: gx as u32,
            gy: gy as u32,
            patches,
        }
    }

    /// Embedding rows of the image.
    pub fn tokens(&self, c: &QwenVisionConfig) -> u32 {
        self.gx * self.gy / (c.merge * c.merge)
    }
}

/// The learned position grid `src` (f32 `[side², e]`, row y · side + x) resized to `gx` x `gy`, row
/// y · gx + x: ggml_interpolate's bilinear with aligned corners in the same f32 expressions, and
/// `src` itself at its own size (llama.cpp skips the resize there).
pub fn position_grid(src: &[f32], side: usize, e: usize, gx: usize, gy: usize) -> Vec<f32> {
    if (gx, gy) == (side, side) {
        return src.to_vec();
    }
    let scale = |out: usize| match out > 1 && side > 1 {
        true => (out - 1) as f32 / (side - 1) as f32,
        false => out as f32 / side as f32,
    };
    // Per output index: the two source indices and the weight of the second.
    let taps = |out: usize| -> Vec<(usize, usize, f32)> {
        let sf = scale(out);
        (0..out)
            .map(|i| {
                let v = i as f32 / sf;
                let f = v.floor() as i64;
                let last = side as i64 - 1;
                let (i0, i1) = (f.clamp(0, last), (f + 1).clamp(0, last));
                (i0 as usize, i1 as usize, (v - i0 as f32).clamp(0.0, 1.0))
            })
            .collect()
    };
    let (tx, ty) = (taps(gx), taps(gy));
    let mut out = Vec::with_capacity(gx * gy * e);
    for &(y0, y1, dy) in &ty {
        for &(x0, x1, dx) in &tx {
            let row = |x: usize, y: usize| &src[(y * side + x) * e..][..e];
            let (a, b, c, d) = (row(x0, y0), row(x1, y0), row(x0, y1), row(x1, y1));
            out.extend((0..e).map(|j| {
                a[j] * (1.0 - dx) * (1.0 - dy)
                    + b[j] * dx * (1.0 - dy)
                    + c[j] * (1.0 - dx) * dy
                    + d[j] * dx * dy
            }));
        }
    }
    out
}

/// One encoder layer: LayerNorm, the fused QKV (rows q, then k, then v), out, LayerNorm, up, down.
struct QLayer {
    ln1: (W, W),
    qkv: (W, W),
    o: (W, W),
    ln2: (W, W),
    up: (W, W),
    down: (W, W),
}

/// The encoder and merger on a device, with scratch for the largest image.
pub struct QwenVision {
    cfg: QwenVisionConfig,
    weights: Buffer,
    /// The patch conv's kernels for the two time frames; a still image goes through both.
    patch: [W; 2],
    patch_b: W,
    /// The learned position grid on the host, f32 `[pos_side², n_embd]`.
    pos: Vec<f32>,
    layers: Vec<QLayer>,
    post_ln: (W, W),
    mm0: (W, W),
    mm2: (W, W),
    sc: Scratch,
}

/// One image's activations, for up to [`QwenVisionConfig::max_tokens`] rows.
/// `work` holds each phase's buffers in turn, as the stream runs them: the patches until the conv,
/// then q, k, v and the attention output (`slot` bytes apart), then the FFN's middle, then the
/// merger's. Each phase's last reader runs before the next phase's first writer.
struct Scratch {
    x: Buffer,
    h: Buffer,
    work: Buffer,
    /// Bytes from one attention buffer to the next in `work`.
    slot: u64,
    /// i32 `[patches, 4]`: each patch's (y, x, y, x), the rope's positions.
    rope: Buffer,
    /// i32 `0..max_tokens`: the identity an image's rows are gathered through ([`QwenVision::ids`]).
    ids: Buffer,
}

impl QwenVision {
    /// Loads the projector at `path` onto `dev` for a text model of width `d_model`. Every key, tensor
    /// name, dtype and shape is checked; `BadGguf(MISMATCH, 0)` when its output is not `d_model` wide.
    pub fn load(
        path: &Path,
        dev: &Device,
        s: &Stream,
        d_model: u32,
    ) -> Result<QwenVision, LoadError> {
        let file = File::open(path).map_err(|_| LoadError::Io)?;
        let len = file.metadata().map_err(|_| LoadError::Io)?.len();
        let g = Gguf::read(BufReader::new(&file), len)?;
        let cfg = QwenVisionConfig::from_gguf(&g)?;
        if cfg.proj_dim != d_model {
            return Err(LoadError::BadGguf(code::MISMATCH, 0));
        }
        let (e, f, pl) = (
            i64::from(cfg.n_embd),
            i64::from(cfg.n_ff),
            i64::from(cfg.patch_len()),
        );
        let mm = e * i64::from(cfg.merge * cfg.merge);
        let dm = i64::from(d_model);
        let mut plan = Plan {
            g: &g,
            file: &file,
            bytes: 0,
            uploads: Vec::new(),
        };
        let [pd, ed, fd, side] = [cfg.patch, cfg.n_embd, cfg.n_ff, cfg.pos_side];
        let patch = [
            plan.mat("v.patch_embd.weight", e, pl, &[pd, pd, 3, ed], None)?,
            plan.mat("v.patch_embd.weight.1", e, pl, &[pd, pd, 3, ed], None)?,
        ];
        let patch_b = plan.vec("v.patch_embd.bias", e, None)?;
        let pos = plan.host_f32("v.position_embd.weight", &[ed, side * side])?;
        let lin = |plan: &mut Plan, name: &str, rows: i64, cols: i64, dims: &[u32]| {
            Ok::<_, LoadError>((
                plan.mat(&format!("{name}.weight"), rows, cols, dims, None)?,
                plan.vec(&format!("{name}.bias"), rows, None)?,
            ))
        };
        let norm = |plan: &mut Plan, name: &str| {
            Ok::<_, LoadError>((
                plan.vec(&format!("{name}.weight"), e, None)?,
                plan.vec(&format!("{name}.bias"), e, None)?,
            ))
        };
        let mut layers = Vec::new();
        for l in 0..cfg.n_layer {
            let b = |t: &str| format!("v.blk.{l}.{t}");
            layers.push(QLayer {
                ln1: norm(&mut plan, &b("ln1"))?,
                qkv: lin(&mut plan, &b("attn_qkv"), 3 * e, e, &[ed, 3 * ed])?,
                o: lin(&mut plan, &b("attn_out"), e, e, &[ed, ed])?,
                ln2: norm(&mut plan, &b("ln2"))?,
                up: lin(&mut plan, &b("ffn_up"), f, e, &[ed, fd])?,
                down: lin(&mut plan, &b("ffn_down"), e, f, &[fd, ed])?,
            });
        }
        let post_ln = norm(&mut plan, "v.post_ln")?;
        let mm0 = lin(&mut plan, "mm.0", mm, mm, &[mm as u32, mm as u32])?;
        let mm2 = lin(&mut plan, "mm.2", dm, mm, &[mm as u32, d_model])?;
        let weights = plan.upload(dev, s)?;
        let need = Scratch::sizes(&cfg).0.iter().sum();
        let sc = Scratch::alloc(dev, s, &cfg).map_err(|e| fit_err(dev, need, e))?;
        s.sync().map_err(backend)?;
        Ok(QwenVision {
            cfg,
            weights,
            patch,
            patch_b,
            pos,
            layers,
            post_ln,
            mm0,
            mm2,
            sc,
        })
    }

    /// The projector's geometry.
    pub fn config(&self) -> &QwenVisionConfig {
        &self.cfg
    }

    /// The i32 identity `0..n` on the device, for gathering `n` rows of an image's embeddings
    /// (`n` at most [`QwenVisionConfig::max_tokens`]).
    pub fn ids(&self, n: u32) -> Result<Tensor<'_>, KError> {
        Tensor::new(&self.sc.ids, 0, DType::I32, &[i64::from(n)])
    }

    /// Encodes `px` into `out`, f32 `[px.tokens(), proj_dim]`, the whole image in one pass on `s`:
    /// [`Self::begin`], every layer, [`Self::finish`].
    // ponytail: the position grid is resized on the host per image (tens of ms at 1600 rows); cache it
    // per grid size if it shows in TTFT.
    pub fn encode(&self, s: &Stream, px: &QwenPixels, out: &Tensor) -> Result<(), KError> {
        self.check_out(px.gx, px.gy, out)?;
        let mut job = self.begin(s, px)?;
        self.advance(s, &mut job, u32::MAX)?;
        self.finish(s, job, out)
    }

    /// `out` takes `gx` x `gy` patches' rows.
    fn check_out(&self, gx: u32, gy: u32, out: &Tensor) -> Result<(), KError> {
        let rows = i64::from(gx) * i64::from(gy) / i64::from(self.cfg.merge * self.cfg.merge);
        match out.dtype() == DType::F32 && out.shape() == [rows, i64::from(self.cfg.proj_dim)] {
            true => Ok(()),
            false => Err(KError::InvalidArg),
        }
    }

    /// Starts encoding `px`: its position rows and its patches through the conv, into the scratch.
    /// [`Self::advance`] then runs its layers, as many at a time as a caller's step can spare, and
    /// [`Self::finish`] writes its rows. The scratch holds one image: any other encode, whole or
    /// begun, before this one finishes clobbers it.
    pub fn begin(&self, s: &Stream, px: &QwenPixels) -> Result<EncodeJob, KError> {
        let c = &self.cfg;
        let m = c.merge;
        let (gx, gy) = (px.gx, px.gy);
        let n = i64::from(gx) * i64::from(gy);
        let pl = i64::from(c.patch_len());
        if n == 0
            || !gx.is_multiple_of(m)
            || !gy.is_multiple_of(m)
            || n / i64::from(m * m) > i64::from(c.max_tokens())
            || px.patches.len() as i64 != n * pl
        {
            return Err(KError::InvalidArg);
        }
        let sc = &self.sc;
        let w = |w: W| w.t(&self.weights);
        let t = self.tensors(n)?;
        let pxt = Tensor::new(&sc.work, 0, DType::F16, &[n, pl])?;
        // The residual starts as the position rows, and the conv's two frames land on it (beta 1):
        // llama.cpp's sum of the two convs, the bias and the position rows, all in f32.
        let (side, ew) = (c.pos_side as usize, c.n_embd as usize);
        let grid = position_grid(&self.pos, side, ew, gx as usize, gy as usize);
        let order: Vec<(usize, usize)> =
            merge_order(gx as usize, gy as usize, m as usize).collect();
        let rows: Vec<f32> = (order.iter())
            .flat_map(|&(x, y)| &grid[(y * gx as usize + x) * ew..][..ew])
            .copied()
            .collect();
        s.upload(&sc.x, 0, &rows)?;
        let at: Vec<i32> = (order.iter())
            .flat_map(|&(x, y)| [y as i32, x as i32, y as i32, x as i32])
            .collect();
        s.upload(&sc.rope, 0, &at)?;
        s.upload(&sc.work, 0, &px.patches)?;
        ops::gemm(
            s,
            &pxt,
            &w(self.patch[0])?,
            Some(&w(self.patch_b)?),
            &t.x,
            1.0,
            1.0,
        )?;
        ops::gemm(s, &pxt, &w(self.patch[1])?, None, &t.x, 1.0, 1.0)?;
        Ok(EncodeJob { gx, gy, layer: 0 })
    }

    /// Runs up to `layers` more of `job`'s encoder layers; whether all have run.
    pub fn advance(&self, s: &Stream, job: &mut EncodeJob, layers: u32) -> Result<bool, KError> {
        let c = &self.cfg;
        let n = i64::from(job.gx) * i64::from(job.gy);
        let w = |w: W| w.t(&self.weights);
        let t = self.tensors(n)?;
        let hd = i64::from(c.n_embd / c.n_head);
        // ggml_rope_multi as qwen3vl.cpp calls it: vision mode, n_dims head_dim / 2, four sections of
        // head_dim / 4 (y, x, y, x), base 10000, no YaRN.
        let rope = RopeParams {
            n_dims: (hd / 2) as u32,
            neox: true,
            n_ctx_orig: 32768,
            freq_base: 10_000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            freq_factors: None,
        };
        let sections = [(hd / 4) as u32; 4];
        let scale = 1.0 / (hd as f32).sqrt();
        let end = (job.layer as usize)
            .saturating_add(layers as usize)
            .min(self.layers.len());
        for l in &self.layers[job.layer as usize..end] {
            ops::layernorm(s, &t.x, &w(l.ln1.0)?, Some(&w(l.ln1.1)?), c.eps, &t.h)?;
            // One gemm, q, k and v strided views of its rows; q and k sit side by side, so one
            // rope_multi turns both (each row's heads share its positions).
            ops::gemm(s, &t.h, &w(l.qkv.0)?, Some(&w(l.qkv.1)?), &t.qkv, 1.0, 0.0)?;
            ops::rope_multi(s, &t.qk, &t.pos, &rope, sections, RopeMode::Vision)?;
            ops::attn_full(s, &t.q, &t.k, &t.v, n as u32, scale, &t.attn)?;
            ops::gemm(
                s,
                &flat(&t.attn)?,
                &w(l.o.0)?,
                Some(&w(l.o.1)?),
                &t.x,
                1.0,
                1.0,
            )?;
            ops::layernorm(s, &t.x, &w(l.ln2.0)?, Some(&w(l.ln2.1)?), c.eps, &t.h)?;
            ops::gemm(s, &t.h, &w(l.up.0)?, Some(&w(l.up.1)?), &t.u, 1.0, 0.0)?;
            ops::gelu(s, &t.u, &t.u)?;
            ops::gemm(s, &t.u, &w(l.down.0)?, Some(&w(l.down.1)?), &t.x, 1.0, 1.0)?;
        }
        job.layer = end as u32;
        Ok(job.layer as usize == self.layers.len())
    }

    /// The post LayerNorm and the merger of `job`, every layer run, into `out`, f32 `[rows,
    /// proj_dim]`.
    pub fn finish(&self, s: &Stream, job: EncodeJob, out: &Tensor) -> Result<(), KError> {
        if job.layer as usize != self.layers.len() {
            return Err(KError::InvalidArg);
        }
        self.check_out(job.gx, job.gy, out)?;
        let c = &self.cfg;
        let (m, e) = (c.merge, i64::from(c.n_embd));
        let n = i64::from(job.gx) * i64::from(job.gy);
        let tokens = n / i64::from(m * m);
        let w = |w: W| w.t(&self.weights);
        let t = self.tensors(n)?;
        ops::layernorm(
            s,
            &t.x,
            &w(self.post_ln.0)?,
            Some(&w(self.post_ln.1)?),
            c.eps,
            &t.h,
        )?;
        // The merge-ordered rows, four to a row, are the merger's input as a reshape.
        let mm = e * i64::from(m * m);
        let merged = t.h.reshape(&[tokens, mm])?;
        let mid = Tensor::new(&self.sc.work, 0, DType::F16, &[tokens, mm])?;
        ops::gemm(
            s,
            &merged,
            &w(self.mm0.0)?,
            Some(&w(self.mm0.1)?),
            &mid,
            1.0,
            0.0,
        )?;
        ops::gelu(s, &mid, &mid)?;
        ops::gemm(
            s,
            &mid,
            &w(self.mm2.0)?,
            Some(&w(self.mm2.1)?),
            out,
            1.0,
            0.0,
        )?;
        Ok(())
    }

    /// The scratch as `n` patches' activations.
    fn tensors(&self, n: i64) -> Result<Acts<'_>, KError> {
        let (sc, c) = (&self.sc, &self.cfg);
        let (e, f) = (i64::from(c.n_embd), i64::from(c.n_ff));
        let nh = i64::from(c.n_head);
        // The QKV rows fill the first three slots; the attention output is the fourth.
        let qkv = Tensor::new(&sc.work, 0, DType::F16, &[n, 3 * e])?;
        let part = |i: i64, w: i64, heads: i64| qkv.cols(i * e, w)?.heads(heads);
        Ok(Acts {
            x: Tensor::new(&sc.x, 0, DType::F32, &[n, e])?,
            h: Tensor::new(&sc.h, 0, DType::F16, &[n, e])?,
            qk: part(0, 2 * e, 2 * nh)?,
            q: part(0, e, nh)?,
            k: part(1, e, nh)?,
            v: part(2, e, nh)?,
            qkv,
            attn: Tensor::new(&sc.work, 3 * sc.slot, DType::F16, &[n, nh, e / nh])?,
            u: Tensor::new(&sc.work, 0, DType::F16, &[n, f])?,
            pos: Tensor::new(&sc.rope, 0, DType::I32, &[n, 4])?,
        })
    }
}

/// An image's encode in flight ([`QwenVision::begin`]): its grid and the next layer to run; its
/// activations are in the projector's scratch.
#[derive(Debug)]
pub struct EncodeJob {
    gx: u32,
    gy: u32,
    layer: u32,
}

impl EncodeJob {
    /// Encoder layers run so far.
    pub fn layers_done(&self) -> u32 {
        self.layer
    }
}

/// One image's activations in the scratch: the f32 residual, the f16 LayerNorm output, the
/// attention's q, k, v and output, the FFN's middle, the rope's positions.
struct Acts<'a> {
    x: Tensor<'a>,
    h: Tensor<'a>,
    /// f16 `[n, 3 e]`, the QKV gemm's output; `qk`, `q`, `k` and `v` are views of its rows.
    qkv: Tensor<'a>,
    qk: Tensor<'a>,
    q: Tensor<'a>,
    k: Tensor<'a>,
    v: Tensor<'a>,
    attn: Tensor<'a>,
    u: Tensor<'a>,
    pos: Tensor<'a>,
}

impl Scratch {
    /// The bytes of x, h, work, rope and ids, and work's attention slot.
    fn sizes(c: &QwenVisionConfig) -> ([u64; 5], u64) {
        let n = u64::from(c.max_tokens()) * u64::from(c.merge * c.merge);
        let (e, f) = (u64::from(c.n_embd), u64::from(c.n_ff));
        let slot = (n * e * 2).next_multiple_of(256);
        // The patches, the four attention buffers, the FFN's middle; the merger's (n / 4 rows of
        // 4 e) is within the attention's.
        let work = (n * u64::from(c.patch_len()) * 2)
            .max(4 * slot)
            .max(n * f * 2);
        let bytes = [
            n * e * 4,
            n * e * 2,
            work,
            n * 4 * 4,
            u64::from(c.max_tokens()) * 4,
        ];
        (bytes, slot)
    }

    fn alloc(dev: &Device, s: &Stream, c: &QwenVisionConfig) -> Result<Scratch, KError> {
        let ([x, h, work, rope, ids], slot) = Self::sizes(c);
        let a = |bytes: u64| Buffer::alloc(dev, bytes);
        let sc = Scratch {
            x: a(x)?,
            h: a(h)?,
            work: a(work)?,
            slot,
            rope: a(rope)?,
            ids: a(ids)?,
        };
        s.upload(&sc.ids, 0, &(0..c.max_tokens() as i32).collect::<Vec<_>>())?;
        Ok(sc)
    }
}
