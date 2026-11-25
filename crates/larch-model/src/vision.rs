//! Image input through InternVL's projector (`clip.projector_type = internvl`), as llama.cpp's mtmd
//! builds it at the oracle's tag (tools/llama-logits):
//! - [`Image`]: the caller's file bytes, named by their sha256, decoded as stb_image does (RGB, alpha
//!   dropped, 16-bit samples by their high byte), behind limits on bytes and pixels checked from the
//!   header before anything is allocated. Where it parts from stb, on purpose: the EXIF orientation is
//!   applied (stb ignores it: the model sees what the person sees); WebP is read (stb reads none); a
//!   GIF's transparent pixels keep their palette colour (stb's are black); and JPEG goes through
//!   zune-jpeg, whose IDCT and upsampling round otherwise (a few levels in 255). The image gate holds
//!   JPEG against stb itself, and the other three against twins of the intended pixels.
//! - [`Pixels`]: llava-uhd tiles, the best grid of 448-pixel tiles (up to [`MAX_TILES`]) and an
//!   overview, resized with Pillow's fixed-point bicubic, normalized and cut into patches.
//! - [`Vision`]: InternViT (LayerNorm, non-causal attention, tanh GELU, layer scale), pixel shuffle and
//!   the MLP projector on the device's ops: [`VisionConfig::tile_tokens`] f32 rows per tile.

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;

use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};
use larch_kernels::{Buffer, DType, Device, KError, Stream, Tensor, ops};
use larch_records::LoadError;
use sha2::{Digest, Sha256};

use crate::gguf::TensorInfo;
use crate::gguf::{Gguf, Value, code};
use crate::transformer::{backend, f16_bits, fit_err, flat};

/// Tiles of one image beside its overview. The model takes 12 (llama.cpp's default); 6 caps one image
/// at 7 × 256 = 1792 tokens, so a picture never takes most of a laptop's context. The grid then
/// differs from llama.cpp's for most large images (1920 × 1080: 3 × 2 here, 4 × 3 there): the gate
/// compares at 6, and 6 read screenshots as well as 12 (the PR's accuracy table).
// ponytail: a fixed cap; derive it from max_context when contexts past 32k make 12 tiles cheap.
pub const MAX_TILES: u32 = 6;
/// Bytes of one image file.
pub const MAX_BYTES: usize = 32 << 20;
/// Pixels of one decoded image, checked from the header before decoding (a decompression bomb).
pub const MAX_PIXELS: u64 = 40_000_000;
/// The longest side, checked the same way.
pub const MAX_SIDE: u32 = 16_384;

/// The text before and after an image's rows in a prompt (llama.cpp's mtmd for InternVL), tokenised
/// with the text around them; the rows themselves are embeddings, not tokens.
pub const IMAGE_BEGIN: &str = "<img>";
/// See [`IMAGE_BEGIN`].
pub const IMAGE_END: &str = "</img>";

/// The bit an image row's token id carries in a prompt: no vocabulary reaches it, so a prompt's ids
/// name each image row and the engine's prefix matching tells images apart.
pub const IMAGE_TOKEN: u32 = 1 << 31;

/// The ids of an image's `n` rows: [`IMAGE_TOKEN`] and 31 bits of sha256(digest, row), so every row
/// depends on all 256 bits of the image's digest. Two different images share a row's id with odds of
/// 2^-31 and a whole image's ids with 2^-(31 n); no state narrower than the digest sits between them,
/// so a crafted pair cannot share every row by colliding a short seed.
pub fn image_tokens(digest: &[u8; 32], n: u32) -> impl Iterator<Item = u32> + '_ {
    (0..n).map(move |i| {
        let h = Sha256::new()
            .chain_update(b"larch image row")
            .chain_update(digest)
            .chain_update(i.to_le_bytes())
            .finalize();
        IMAGE_TOKEN | (u32::from_le_bytes([h[0], h[1], h[2], h[3]]) & !IMAGE_TOKEN)
    })
}

/// Why an image cannot be used, in words the chat shows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageError {
    Empty,
    TooManyBytes(usize),
    /// A format this build cannot read, by name.
    Unsupported(&'static str),
    /// Width and height past the limits (or zero).
    Size(u64, u64),
    /// Past the decoder's limits (a side, or the memory a decode takes) before its size is known.
    TooLarge,
    /// The decoder's own message: a truncated or corrupt file.
    Corrupt(String),
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ImageError::Empty => write!(f, "the image is empty"),
            ImageError::TooManyBytes(n) => write!(
                f,
                "the image is {} MB; the limit is {} MB",
                n.div_ceil(1 << 20),
                MAX_BYTES >> 20
            ),
            ImageError::Unsupported(what) => {
                write!(
                    f,
                    "{what} images are not supported: use PNG, JPEG, WebP or GIF"
                )
            }
            ImageError::Size(w, h) => write!(
                f,
                "the image is {w}x{h} pixels; the limit is {} megapixels and {MAX_SIDE} per side",
                MAX_PIXELS / 1_000_000
            ),
            ImageError::TooLarge => write!(
                f,
                "the image is too large; the limit is {} megapixels and {MAX_SIDE} per side",
                MAX_PIXELS / 1_000_000
            ),
            ImageError::Corrupt(e) => write!(f, "the image cannot be read: {e}"),
        }
    }
}

impl std::error::Error for ImageError {}

/// An image file the caller supplies, named by the sha256 of its bytes; width and height are after
/// its EXIF orientation. Checked at construction (format, limits, header); decoded on demand. The
/// fields are private, so a digest always names these bytes (the tile cache and the row ids trust
/// it) and the size is always the file's.
#[derive(Clone, Debug)]
pub struct Image {
    digest: [u8; 32],
    width: u32,
    height: u32,
    bytes: Arc<[u8]>,
}

impl PartialEq for Image {
    fn eq(&self, other: &Image) -> bool {
        self.digest == other.digest
    }
}

impl Image {
    /// Checks `bytes` (the limits, the format, the header) without decoding the pixels.
    pub fn new(bytes: Arc<[u8]>) -> Result<Image, ImageError> {
        let (width, height) = {
            let (mut d, _) = decoder(&bytes)?;
            let (w, h) = d.dimensions();
            match d.orientation() {
                Ok(o) if swaps(o) => (h, w),
                _ => (w, h),
            }
        };
        Ok(Image {
            digest: Sha256::digest(&bytes).into(),
            width,
            height,
            bytes,
        })
    }

    /// The file's bytes.
    pub fn bytes(&self) -> &Arc<[u8]> {
        &self.bytes
    }

    /// The sha256 of [`bytes`](Self::bytes).
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// The width in pixels, after the EXIF orientation.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The height in pixels, after the EXIF orientation.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The digest as 64 lowercase hex digits, the name the image is stored under.
    pub fn hex(&self) -> String {
        hex(&self.digest)
    }

    /// The pixels as RGB, row-major.
    pub fn decode(&self) -> Result<Rgb, ImageError> {
        let (mut d, _) = decoder(&self.bytes)?;
        let orientation = d.orientation().ok();
        let mut img = DynamicImage::from_decoder(d).map_err(corrupt)?;
        if let Some(o) = orientation {
            img.apply_orientation(o);
        }
        let (width, height) = (img.width(), img.height());
        // stb_image's 3 channels: 16-bit samples keep their high byte, alpha is dropped. Straight from
        // the decoded samples, with no 16-bit RGB copy (a 40 MP image's would be 240 MB more).
        let hi = |v: u16| (v >> 8) as u8;
        let data = match &img {
            DynamicImage::ImageLuma16(b) => b.iter().flat_map(|&v| [hi(v); 3]).collect(),
            DynamicImage::ImageLumaA16(b) => {
                b.chunks_exact(2).flat_map(|p| [hi(p[0]); 3]).collect()
            }
            DynamicImage::ImageRgb16(b) => b.iter().map(|&v| hi(v)).collect(),
            DynamicImage::ImageRgba16(b) => (b.chunks_exact(4))
                .flat_map(|p| [hi(p[0]), hi(p[1]), hi(p[2])])
                .collect(),
            _ => img.to_rgb8().into_raw(),
        };
        drop(img);
        Ok(Rgb {
            width,
            height,
            data,
        })
    }
}

/// 64 lowercase hex digits.
pub fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn corrupt(e: image::ImageError) -> ImageError {
    match e {
        image::ImageError::Limits(_) => ImageError::TooLarge,
        image::ImageError::Unsupported(_) => ImageError::Unsupported("this kind of"),
        e => ImageError::Corrupt(e.to_string()),
    }
}

fn swaps(o: image::metadata::Orientation) -> bool {
    use image::metadata::Orientation::*;
    matches!(o, Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH)
}

/// A decoder over `bytes`, once the byte limit, the format and the header's size pass.
fn decoder(bytes: &[u8]) -> Result<(Box<dyn ImageDecoder + '_>, ImageFormat), ImageError> {
    if bytes.is_empty() {
        return Err(ImageError::Empty);
    }
    if bytes.len() > MAX_BYTES {
        return Err(ImageError::TooManyBytes(bytes.len()));
    }
    let text = bytes.trim_ascii_start();
    if text.starts_with(b"<svg") || text.starts_with(b"<?xml") {
        return Err(ImageError::Unsupported("SVG"));
    }
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        return Err(ImageError::Unsupported(match &bytes[8..12] {
            b"avif" | b"avis" => "AVIF",
            _ => "HEIC",
        }));
    }
    let mut r = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| ImageError::Corrupt(e.to_string()))?;
    let format = match r.format() {
        Some(f @ (ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif)) => {
            f
        }
        Some(ImageFormat::Bmp) => return Err(ImageError::Unsupported("BMP")),
        Some(ImageFormat::Tiff) => return Err(ImageError::Unsupported("TIFF")),
        _ => {
            return Err(ImageError::Corrupt(
                "not a PNG, JPEG, WebP or GIF file".into(),
            ));
        }
    };
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_SIDE);
    limits.max_image_height = Some(MAX_SIDE);
    // RGBA16 at the pixel limit, plus the RGB copy.
    limits.max_alloc = Some(MAX_PIXELS * 11);
    r.limits(limits);
    let d = r.into_decoder().map_err(|e| match e {
        image::ImageError::Limits(_) => ImageError::TooLarge,
        e => corrupt(e),
    })?;
    let (w, h) = d.dimensions();
    let (w, h) = (u64::from(w), u64::from(h));
    if w == 0 || h == 0 || w > u64::from(MAX_SIDE) || h > u64::from(MAX_SIDE) || w * h > MAX_PIXELS
    {
        return Err(ImageError::Size(w, h));
    }
    Ok((Box::new(d), format))
}

/// RGB pixels, 3 bytes each, row-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Rgb {
    /// A `width` x `height` image of one colour.
    pub fn filled(width: u32, height: u32, c: [u8; 3]) -> Rgb {
        Rgb {
            width,
            height,
            data: c.repeat((width * height) as usize),
        }
    }
}

/// The projector's geometry, from the mmproj GGUF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionConfig {
    /// Tile side in pixels (448).
    pub image_size: u32,
    /// Patch side in pixels (14).
    pub patch: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_head: u32,
    pub n_layer: u32,
    pub eps: f32,
    /// The projector's output width: the text model's d_model.
    pub proj_dim: u32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Pixel-shuffle factor (2): merge² patches make one token.
    pub merge: u32,
    pub min_tiles: u32,
    /// The model's tile limit, before [`MAX_TILES`].
    pub max_tiles: u32,
}

impl VisionConfig {
    /// Reads and range-checks the `clip.*` keys of an InternVL projector. `UnsupportedArch` for another
    /// projector; `BadGguf(KEY, 0)` for a missing or out-of-range key.
    pub fn from_gguf(g: &Gguf) -> Result<VisionConfig, LoadError> {
        clip_kind(g, "internvl")?;
        let (u, opt) = (
            |k: &str, lo: u64, hi: u64| clip_u32(g, k, lo, hi),
            |k: &str, default: u32, hi: u64| clip_opt(g, k, default, hi),
        );
        let c = VisionConfig {
            image_size: u("image_size", 1, 4096)?,
            patch: u("patch_size", 1, 64)?,
            n_embd: u("embedding_length", 1, 8192)?,
            n_ff: u("feed_forward_length", 1, 65536)?,
            n_head: u("attention.head_count", 1, 256)?,
            n_layer: u("block_count", 1, 256)?,
            eps: clip_eps(g)?,
            proj_dim: u("projection_dim", 1, 65536)?,
            mean: clip_three(g, "image_mean")?,
            std: clip_three(g, "image_std")?,
            merge: opt("projector.scale_factor", 1, 8)?,
            min_tiles: opt("preproc_min_tiles", 1, 64)?,
            max_tiles: opt("preproc_max_tiles", 12, 64)?,
        };
        let grid = c.image_size / c.patch;
        // Whole patches, whole merge blocks, the head width attn_full takes on every backend (64),
        // stds that divide.
        let ok = c.image_size.is_multiple_of(c.patch)
            && grid.is_multiple_of(c.merge)
            && c.n_embd.is_multiple_of(c.n_head)
            && c.n_embd / c.n_head == 64
            && c.min_tiles <= c.max_tiles
            && c.std.iter().all(|&v| v != 0.0)
            && u64::from(grid) * u64::from(grid) <= 16_384;
        if !ok {
            return Err(LoadError::BadGguf(code::KEY, 0));
        }
        Ok(c)
    }

    /// Patches per tile side.
    pub fn grid(&self) -> u32 {
        self.image_size / self.patch
    }

    /// Patches per tile.
    pub fn patches(&self) -> u32 {
        self.grid() * self.grid()
    }

    /// Embedding rows per tile.
    pub fn tile_tokens(&self) -> u32 {
        self.patches() / (self.merge * self.merge)
    }

    /// Values per patch: 3 channels of patch².
    pub fn patch_len(&self) -> u32 {
        3 * self.patch * self.patch
    }

    /// The llava-uhd grid llama.cpp picks for a `w` x `h` image, `(cols, rows)` of tiles, or `(0, 0)`
    /// when the overview alone covers it.
    pub fn grid_for(&self, w: u32, h: u32) -> (u32, u32) {
        let s = self.image_size;
        if w <= s && h <= s {
            return (0, 0);
        }
        let (min, max) = (
            self.min_tiles,
            self.max_tiles.min(MAX_TILES).max(self.min_tiles),
        );
        let (mut best, mut best_eff, mut best_waste) = ((1, 1), 0i64, i64::MAX);
        for a in min..=max {
            for b in min.div_ceil(a).max(min)..=(max / a).min(max) {
                let (cw, ch) = (a * s, b * s);
                // resize_maintain_aspect_ratio, in float as the reference.
                let scale = (cw as f32 / w as f32).min(ch as f32 / h as f32);
                let (tw, th) = ((w as f32 * scale) as i64, (h as f32 * scale) as i64);
                let eff = (tw * th).min(i64::from(w) * i64::from(h));
                let waste = i64::from(cw) * i64::from(ch) - eff;
                if eff > best_eff || (eff == best_eff && waste < best_waste) {
                    (best, best_eff, best_waste) = ((a, b), eff, waste);
                }
            }
        }
        best
    }

    /// Tiles a `w` x `h` image takes, the overview included.
    pub fn tiles_for(&self, w: u32, h: u32) -> u32 {
        let (c, r) = self.grid_for(w, h);
        c * r + 1
    }

    /// Embedding rows of a `w` x `h` image.
    pub fn tokens_for(&self, w: u32, h: u32) -> u32 {
        self.tiles_for(w, h) * self.tile_tokens()
    }
}

/// `UnsupportedArch` unless `g` is a clip projector of type `kind` with a vision encoder and GELU.
pub(crate) fn clip_kind(g: &Gguf, kind: &str) -> Result<(), LoadError> {
    let s = |k: &str| g.get(k).and_then(Value::as_str);
    if s("general.architecture") != Some("clip")
        || s("clip.projector_type") != Some(kind)
        || g.get("clip.has_vision_encoder") != Some(&Value::Bool(true))
        || g.get("clip.use_gelu") != Some(&Value::Bool(true))
    {
        return Err(LoadError::UnsupportedArch);
    }
    Ok(())
}

/// `clip.vision.{k}` in `lo..=hi`, or `BadGguf(KEY, 0)`.
pub(crate) fn clip_u32(g: &Gguf, k: &str, lo: u64, hi: u64) -> Result<u32, LoadError> {
    match g.get(&format!("clip.vision.{k}")).and_then(Value::as_u64) {
        Some(v) if (lo..=hi).contains(&v) => Ok(v as u32),
        _ => Err(LoadError::BadGguf(code::KEY, 0)),
    }
}

/// `clip.vision.{k}` in `1..=hi`, `default` when absent.
pub(crate) fn clip_opt(g: &Gguf, k: &str, default: u32, hi: u64) -> Result<u32, LoadError> {
    match g.get(&format!("clip.vision.{k}")) {
        None => Ok(default),
        Some(_) => clip_u32(g, k, 1, hi),
    }
}

/// Three finite f32 values at `clip.vision.{k}` (a mean or std per channel).
pub(crate) fn clip_three(g: &Gguf, k: &str) -> Result<[f32; 3], LoadError> {
    let bad = LoadError::BadGguf(code::KEY, 0);
    let a = g
        .get(&format!("clip.vision.{k}"))
        .and_then(Value::as_arr)
        .ok_or(bad)?;
    match a
        .iter()
        .map(Value::as_f32)
        .collect::<Option<Vec<_>>>()
        .as_deref()
    {
        Some(&[x, y, z]) if [x, y, z].iter().all(|v| v.is_finite()) => Ok([x, y, z]),
        _ => Err(bad),
    }
}

/// The encoder's LayerNorm epsilon, in (0, 1).
pub(crate) fn clip_eps(g: &Gguf) -> Result<f32, LoadError> {
    g.get("clip.vision.attention.layer_norm_epsilon")
        .and_then(Value::as_f32)
        .filter(|e| e.is_finite() && *e > 0.0 && *e < 1.0)
        .ok_or(LoadError::BadGguf(code::KEY, 0))
}

/// llama.cpp's Pillow-exact separable resample (bicubic, a = -0.5; 22-bit fixed point; the horizontal
/// pass stored as u8 before the vertical one).
fn resize(src: &Rgb, w: u32, h: u32) -> Rgb {
    const BITS: u32 = 22;
    let clip8 = |v: i32| (v >> BITS).clamp(0, 255) as u8;
    let filter = |x: f64| {
        let x = x.abs();
        let a = -0.5;
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    // Per output pixel: the first input index, the taps, and their fixed-point weights.
    let weights = |inp: u32, out: u32| -> (usize, Vec<(i32, i32)>, Vec<i32>) {
        let scale = f64::from(inp) / f64::from(out);
        let filterscale = scale.max(1.0);
        let support = 2.0 * filterscale;
        let ksize = support.ceil() as usize * 2 + 1;
        let mut bounds = Vec::with_capacity(out as usize);
        let mut w = vec![0i32; out as usize * ksize];
        for xx in 0..out as usize {
            let center = (xx as f64 + 0.5) * scale;
            let ss = 1.0 / filterscale;
            let xmin = ((center - support + 0.5) as i32).max(0);
            let xmax = ((center + support + 0.5) as i32).min(inp as i32) - xmin;
            let pre: Vec<f64> = (0..xmax)
                .map(|x| filter((f64::from(x + xmin) - center + 0.5) * ss))
                .collect();
            let ww: f64 = pre.iter().sum();
            for (x, p) in pre.iter().enumerate() {
                let p = if ww != 0.0 { p / ww } else { *p };
                let fx = p * f64::from(1u32 << BITS) + if p < 0.0 { -0.5 } else { 0.5 };
                w[xx * ksize + x] = fx as i32;
            }
            bounds.push((xmin, xmax));
        }
        (ksize, bounds, w)
    };
    let horizontal = |s: &Rgb, out: u32| -> Rgb {
        let (k, bounds, w) = weights(s.width, out);
        let mut data = vec![0u8; out as usize * s.height as usize * 3];
        for y in 0..s.height as usize {
            let row = &s.data[y * s.width as usize * 3..];
            for (xx, &(xmin, n)) in bounds.iter().enumerate() {
                let mut acc = [1i32 << (BITS - 1); 3];
                for x in 0..n as usize {
                    let p = &row[(xmin as usize + x) * 3..];
                    for c in 0..3 {
                        acc[c] += i32::from(p[c]) * w[xx * k + x];
                    }
                }
                let o = (y * out as usize + xx) * 3;
                for c in 0..3 {
                    data[o + c] = clip8(acc[c]);
                }
            }
        }
        Rgb {
            width: out,
            height: s.height,
            data,
        }
    };
    let vertical = |s: &Rgb, out: u32| -> Rgb {
        let (k, bounds, w) = weights(s.height, out);
        let row = s.width as usize * 3;
        let mut data = vec![0u8; row * out as usize];
        let mut acc = vec![0i32; row];
        for (yy, &(ymin, n)) in bounds.iter().enumerate() {
            acc.fill(1 << (BITS - 1));
            for y in 0..n as usize {
                let src = &s.data[(ymin as usize + y) * row..][..row];
                let wy = w[yy * k + y];
                for (a, &v) in acc.iter_mut().zip(src) {
                    *a += i32::from(v) * wy;
                }
            }
            for (d, &a) in data[yy * row..][..row].iter_mut().zip(&acc) {
                *d = clip8(a);
            }
        }
        Rgb {
            width: s.width,
            height: out,
            data,
        }
    };
    match (w != src.width, h != src.height) {
        (true, true) => vertical(&horizontal(src, w), h),
        (true, false) => horizontal(src, w),
        (false, true) => vertical(src, h),
        (false, false) => src.clone(),
    }
}

/// `img` resized into a `w` x `h` black canvas, aspect kept and centered (llama.cpp's PAD_CEIL).
pub(crate) fn resize_padded(img: &Rgb, w: u32, h: u32) -> Rgb {
    if (img.width, img.height) == (w, h) {
        return img.clone();
    }
    let scale = (w as f32 / img.width as f32).min(h as f32 / img.height as f32);
    let nw = ((img.width as f32 * scale).ceil() as u32).min(w);
    let nh = ((img.height as f32 * scale).ceil() as u32).min(h);
    let inner = resize(img, nw, nh);
    let mut out = Rgb::filled(w, h, [0, 0, 0]);
    let (ox, oy) = ((w - nw) / 2, (h - nh) / 2);
    for y in 0..nh as usize {
        let at = ((oy as usize + y) * w as usize + ox as usize) * 3;
        out.data[at..at + nw as usize * 3]
            .copy_from_slice(&inner.data[y * nw as usize * 3..][..nw as usize * 3]);
    }
    out
}

/// An image cut into the encoder's input: f16 patch rows per tile, the tiles row-major and then the
/// overview, each tile's patches in merge order (the merge² patches of one output token together),
/// so the pixel shuffle after the encoder is a reshape.
#[derive(Clone, Debug, PartialEq)]
pub struct Pixels {
    pub tiles: u32,
    /// f16 bits, `[tiles * patches, 3 * patch²]`.
    pub patches: Vec<u16>,
}

impl Pixels {
    /// Tiles and overview of `img`, as llama.cpp's InternVL preprocessor makes them.
    pub fn new(img: &Rgb, c: &VisionConfig) -> Pixels {
        let s = c.image_size;
        let (cols, rows) = c.grid_for(img.width, img.height);
        let mut tiles = Vec::new();
        if cols > 0 {
            let refined = resize_padded(img, cols * s, rows * s);
            for ty in 0..rows {
                for tx in 0..cols {
                    let mut t = Rgb::filled(s, s, [0, 0, 0]);
                    for y in 0..s as usize {
                        let from = ((ty * s) as usize + y) * (cols * s) as usize * 3
                            + (tx * s) as usize * 3;
                        t.data[y * s as usize * 3..][..s as usize * 3]
                            .copy_from_slice(&refined.data[from..][..s as usize * 3]);
                    }
                    tiles.push(t);
                }
            }
        }
        tiles.push(resize(img, s, s));
        let mut patches = Vec::with_capacity(tiles.len() * (c.patches() * c.patch_len()) as usize);
        for t in &tiles {
            patchify(t, c, &mut patches);
        }
        Pixels {
            tiles: tiles.len() as u32,
            patches,
        }
    }
}

/// Appends one tile's patches in merge order: row `merge² t + k` is patch (px, py) of output token
/// t = (py / m) (G / m) + px / m, k = px % m + m (py % m), as llama.cpp's pixel shuffle orders the
/// channels. A patch is c · p² + ky · p + kx of (v / 255 - mean) / std, rounded to f16 (im2col's).
fn patchify(t: &Rgb, c: &VisionConfig, out: &mut Vec<u16>) {
    let (p, g, m) = (c.patch as usize, c.grid() as usize, c.merge as usize);
    let planes: [Vec<f32>; 3] = std::array::from_fn(|ch| {
        (t.data.chunks_exact(3))
            .map(|px| (f32::from(px[ch]) / 255.0 - c.mean[ch]) / c.std[ch])
            .collect()
    });
    let side = c.image_size as usize;
    for (px, py) in merge_order(g, g, m) {
        for plane in &planes {
            for ky in 0..p {
                let row = &plane[(py * p + ky) * side + px * p..][..p];
                out.extend(row.iter().map(|&v| f16_bits(v)));
            }
        }
    }
}

/// The patches (px, py) of a gx x gy grid in merge order (see [`patchify`]): output token
/// t = (py / m) (gx / m) + px / m, its merge² patches together.
pub(crate) fn merge_order(gx: usize, gy: usize, m: usize) -> impl Iterator<Item = (usize, usize)> {
    let gm = gx / m;
    (0..gx * gy).map(move |r| {
        let (t, k) = (r / (m * m), r % (m * m));
        ((t % gm) * m + k % m, (t / gm) * m + k / m)
    })
}

/// A weight's place in the one weight buffer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct W {
    pub(crate) at: u64,
    pub(crate) dtype: DType,
    pub(crate) rows: i64,
    /// 0 for a vector.
    pub(crate) cols: i64,
}

impl W {
    /// The weight as a tensor of `weights`.
    pub(crate) fn t(self, weights: &Buffer) -> Result<Tensor<'_>, KError> {
        match self.cols {
            0 => Tensor::new(weights, self.at, self.dtype, &[self.rows]),
            c => Tensor::new(weights, self.at, self.dtype, &[self.rows, c]),
        }
    }
}

/// One encoder layer: LayerNorm, q, k, v, out (scaled by ls1), LayerNorm, up, down (scaled by ls2).
struct VLayer {
    ln1: (W, W),
    q: (W, W),
    k: (W, W),
    v: (W, W),
    o: (W, W),
    ln2: (W, W),
    up: (W, W),
    down: (W, W),
}

/// The encoder and projector on a device, with scratch for one tile.
pub struct Vision {
    cfg: VisionConfig,
    weights: Buffer,
    patch: (W, W),
    /// f32 `[patches + 1, n_embd]`: position rows in merge order, the CLS row last with the class
    /// embedding added (llama.cpp appends CLS, so it takes the last position row).
    pos: W,
    layers: Vec<VLayer>,
    mm0: (W, W),
    mm1: (W, W),
    mm3: (W, W),
    sc: Scratch,
}

/// One tile's activations and the identity the row gathers read.
struct Scratch {
    px: Buffer,
    x: Buffer,
    h: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    t: Buffer,
    u: Buffer,
    /// i32 `0..`: a tile's rows, or the most rows one image feeds ([`Vision::ids`]).
    ids: Buffer,
}

impl Vision {
    /// Loads the projector at `path` onto `dev` for a text model of width `d_model`. Every key, tensor
    /// name, dtype and shape is checked; `BadGguf(MISMATCH, 0)` when its output is not `d_model` wide.
    pub fn load(path: &Path, dev: &Device, s: &Stream, d_model: u32) -> Result<Vision, LoadError> {
        let file = File::open(path).map_err(|_| LoadError::Io)?;
        let len = file.metadata().map_err(|_| LoadError::Io)?.len();
        let g = Gguf::read(BufReader::new(&file), len)?;
        let cfg = VisionConfig::from_gguf(&g)?;
        if cfg.proj_dim != d_model {
            return Err(LoadError::BadGguf(code::MISMATCH, 0));
        }
        let (e, f, n) = (
            i64::from(cfg.n_embd),
            i64::from(cfg.n_ff),
            i64::from(cfg.patches()),
        );
        let (pl, mm) = (
            i64::from(cfg.patch_len()),
            e * i64::from(cfg.merge * cfg.merge),
        );
        let dm = i64::from(d_model);
        let mut plan = Plan {
            g: &g,
            file: &file,
            bytes: 0,
            uploads: Vec::new(),
        };
        let [pd, ed, fd] = [cfg.patch, cfg.n_embd, cfg.n_ff];
        let patch = (
            plan.mat("v.patch_embd.weight", e, pl, &[pd, pd, 3, ed], None)?,
            plan.vec("v.patch_embd.bias", e, None)?,
        );
        let class = plan.host_f32("v.class_embd", &[ed, 1, 1])?;
        let pos_src = plan.host_f32("v.position_embd.weight", &[ed, cfg.patches() + 1, 1])?;
        let pos = plan.alloc(DType::F32, n + 1, e);
        let mut layers = Vec::new();
        for l in 0..cfg.n_layer {
            let b = |t: &str| format!("v.blk.{l}.{t}");
            let ls1: Arc<[f32]> = plan.host_f32(&b("ls1.weight"), &[ed])?.into();
            let ls2: Arc<[f32]> = plan.host_f32(&b("ls2.weight"), &[ed])?.into();
            let mut lin =
                |name: &str, rows: i64, cols: i64, dims: &[u32], ls: Option<&Arc<[f32]>>| {
                    Ok::<_, LoadError>((
                        plan.mat(&b(&format!("{name}.weight")), rows, cols, dims, ls.cloned())?,
                        plan.vec(&b(&format!("{name}.bias")), rows, ls.cloned())?,
                    ))
                };
            let q = lin("attn_q", e, e, &[ed, ed], None)?;
            let k = lin("attn_k", e, e, &[ed, ed], None)?;
            let v = lin("attn_v", e, e, &[ed, ed], None)?;
            let o = lin("attn_out", e, e, &[ed, ed], Some(&ls1))?;
            let up = lin("ffn_up", f, e, &[ed, fd], None)?;
            let down = lin("ffn_down", e, f, &[fd, ed], Some(&ls2))?;
            layers.push(VLayer {
                ln1: (
                    plan.vec(&b("ln1.weight"), e, None)?,
                    plan.vec(&b("ln1.bias"), e, None)?,
                ),
                q,
                k,
                v,
                o,
                ln2: (
                    plan.vec(&b("ln2.weight"), e, None)?,
                    plan.vec(&b("ln2.bias"), e, None)?,
                ),
                up,
                down,
            });
        }
        let mm0 = (
            plan.vec("mm.model.mlp.0.weight", mm, None)?,
            plan.vec("mm.model.mlp.0.bias", mm, None)?,
        );
        let mm1 = (
            plan.mat("mm.model.mlp.1.weight", dm, mm, &[mm as u32, d_model], None)?,
            plan.vec("mm.model.mlp.1.bias", dm, None)?,
        );
        let mm3 = (
            plan.mat("mm.model.mlp.3.weight", dm, dm, &[d_model, d_model], None)?,
            plan.vec("mm.model.mlp.3.bias", dm, None)?,
        );
        let weights = plan.upload(dev, s)?;
        // The position rows in merge order, then CLS: llama.cpp's patch p takes row p.
        let (side, ew) = (cfg.grid() as usize, cfg.n_embd as usize);
        let mut rows = Vec::with_capacity((n as usize + 1) * ew);
        for (px, py) in merge_order(side, side, cfg.merge as usize) {
            rows.extend_from_slice(&pos_src[(py * side + px) * ew..][..ew]);
        }
        rows.extend(
            pos_src[n as usize * ew..][..ew]
                .iter()
                .zip(&class)
                .map(|(p, c)| c + p),
        );
        s.upload(&weights, pos.at, &rows).map_err(backend)?;
        let need = Scratch::sizes(&cfg, d_model).iter().sum();
        let sc = Scratch::alloc(dev, s, &cfg, d_model).map_err(|e| fit_err(dev, need, e))?;
        s.sync().map_err(backend)?;
        Ok(Vision {
            cfg,
            weights,
            patch,
            pos,
            layers,
            mm0,
            mm1,
            mm3,
            sc,
        })
    }

    /// The projector's geometry.
    pub fn config(&self) -> &VisionConfig {
        &self.cfg
    }

    fn t(&self, w: W) -> Result<Tensor<'_>, KError> {
        w.t(&self.weights)
    }

    /// The i32 identity `0..n` on the device, for gathering `n` rows of an embedding table.
    pub fn ids(&self, n: u32) -> Result<Tensor<'_>, KError> {
        Tensor::new(&self.sc.ids, 0, DType::I32, &[i64::from(n)])
    }

    /// Encodes `px` into `out`, f32 `[px.tiles * tile_tokens, proj_dim]`, one tile at a time on `s`.
    pub fn encode(&self, s: &Stream, px: &Pixels, out: &Tensor) -> Result<(), KError> {
        let (tt, dm) = (
            i64::from(self.cfg.tile_tokens()),
            i64::from(self.cfg.proj_dim),
        );
        let tiles = i64::from(px.tiles);
        if out.dtype() != DType::F32 || out.shape() != [tiles * tt, dm] {
            return Err(KError::InvalidArg);
        }
        for tile in 0..px.tiles {
            self.encode_tile(s, px, tile, &out.rows(i64::from(tile) * tt, tt)?)?;
        }
        Ok(())
    }

    /// Encodes tile `tile` of `px` (the overview is the last) into `out`, f32 `[tile_tokens,
    /// proj_dim]`.
    // ponytail: one tile per pass; batch tiles in one pass (attn_full's segments) when a 6-tile
    // image's encode shows in TTFT.
    pub fn encode_tile(
        &self,
        s: &Stream,
        px: &Pixels,
        tile: u32,
        out: &Tensor,
    ) -> Result<(), KError> {
        let c = &self.cfg;
        let (e, f, n) = (
            i64::from(c.n_embd),
            i64::from(c.n_ff),
            i64::from(c.patches()),
        );
        let (tt, pl) = (i64::from(c.tile_tokens()), i64::from(c.patch_len()));
        let rows = n + 1;
        if px.patches.len() as i64 != i64::from(px.tiles) * n * pl
            || tile >= px.tiles
            || out.dtype() != DType::F32
            || out.shape() != [tt, i64::from(c.proj_dim)]
        {
            return Err(KError::InvalidArg);
        }
        let sc = &self.sc;
        let w = |w: W| self.t(w);
        let hd = e / i64::from(c.n_head);
        let x = Tensor::new(&sc.x, 0, DType::F32, &[rows, e])?;
        let h = Tensor::new(&sc.h, 0, DType::F16, &[rows, e])?;
        let heads = [rows, i64::from(c.n_head), hd];
        let (q, k, v) = (
            Tensor::new(&sc.q, 0, DType::F16, &heads)?,
            Tensor::new(&sc.k, 0, DType::F16, &heads)?,
            Tensor::new(&sc.v, 0, DType::F16, &heads)?,
        );
        let attn = Tensor::new(&sc.attn, 0, DType::F16, &heads)?;
        let t = Tensor::new(&sc.t, 0, DType::F16, &[rows, e])?;
        let u = Tensor::new(&sc.u, 0, DType::F16, &[rows, f])?;
        let ids = Tensor::new(&sc.ids, 0, DType::I32, &[rows])?;
        let scale = 1.0 / (hd as f32).sqrt();
        let pos = w(self.pos)?;
        {
            let at = (i64::from(tile) * n * pl) as usize;
            s.upload(&sc.px, 0, &px.patches[at..at + (n * pl) as usize])?;
            let pxt = Tensor::new(&sc.px, 0, DType::F16, &[n, pl])?;
            // Patches through the conv (as a gemm), plus their position rows; CLS with its row last.
            ops::gemm(
                s,
                &pxt,
                &w(self.patch.0)?,
                Some(&w(self.patch.1)?),
                &t.rows(0, n)?,
                1.0,
                0.0,
            )?;
            ops::add(s, &pos.rows(0, n)?, &t.rows(0, n)?, &x.rows(0, n)?)?;
            ops::embed(s, &pos, &ids.rows(n, 1)?, &x.rows(n, 1)?)?;
            for l in &self.layers {
                ops::layernorm(s, &x, &w(l.ln1.0)?, Some(&w(l.ln1.1)?), c.eps, &h)?;
                for (p, out) in [(l.q, &q), (l.k, &k), (l.v, &v)] {
                    ops::gemm(s, &h, &w(p.0)?, Some(&w(p.1)?), &flat(out)?, 1.0, 0.0)?;
                }
                ops::attn_full(s, &q, &k, &v, rows as u32, scale, &attn)?;
                // The projections land on the f32 residual (beta 1), as llama.cpp adds in f32: no f16
                // rounding of them and no separate add.
                ops::gemm(s, &flat(&attn)?, &w(l.o.0)?, Some(&w(l.o.1)?), &x, 1.0, 1.0)?;
                ops::layernorm(s, &x, &w(l.ln2.0)?, Some(&w(l.ln2.1)?), c.eps, &h)?;
                ops::gemm(s, &h, &w(l.up.0)?, Some(&w(l.up.1)?), &u, 1.0, 0.0)?;
                ops::gelu(s, &u, &u)?;
                ops::gemm(s, &u, &w(l.down.0)?, Some(&w(l.down.1)?), &x, 1.0, 1.0)?;
            }
            // Drop CLS; the merge-ordered patch rows are the pixel shuffle as a reshape. The projector's
            // LayerNorm eps is PyTorch's default, as llama.cpp hard-codes it.
            let merged = x.rows(0, n)?.reshape(&[tt, e * n / tt])?;
            let hm = Tensor::new(&sc.u, 0, DType::F16, &[tt, e * n / tt])?;
            ops::layernorm(
                s,
                &merged,
                &w(self.mm0.0)?,
                Some(&w(self.mm0.1)?),
                1e-5,
                &hm,
            )?;
            let dm = i64::from(c.proj_dim);
            let y = Tensor::new(&sc.h, 0, DType::F16, &[tt, dm])?;
            ops::gemm(s, &hm, &w(self.mm1.0)?, Some(&w(self.mm1.1)?), &y, 1.0, 0.0)?;
            ops::gelu(s, &y, &y)?;
            ops::gemm(s, &y, &w(self.mm3.0)?, Some(&w(self.mm3.1)?), out, 1.0, 0.0)?;
        }
        Ok(())
    }
}

impl Scratch {
    /// Each buffer's bytes, in field order.
    fn sizes(c: &VisionConfig, d_model: u32) -> [u64; 10] {
        let (e, f, n) = (
            u64::from(c.n_embd),
            u64::from(c.n_ff),
            u64::from(c.patches()),
        );
        let rows = n + 1;
        // The merged rows' LayerNorm output reuses `u` and the projector's middle `h`.
        let tt = u64::from(c.tile_tokens());
        // The identity covers a tile's rows and the most rows one image injects (Vision::ids).
        let ids = rows.max(u64::from(MAX_TILES + 1) * tt);
        let u = (rows * f).max(tt * e * n / tt).max(1) * 2;
        let h = (rows * e).max(tt * u64::from(d_model)) * 2;
        let half = rows * e * 2;
        [
            n * u64::from(c.patch_len()) * 2,
            rows * e * 4,
            h,
            half,
            half,
            half,
            half,
            half,
            u,
            ids * 4,
        ]
    }

    fn alloc(dev: &Device, s: &Stream, c: &VisionConfig, d_model: u32) -> Result<Scratch, KError> {
        let [px, x, h, q, k, v, attn, t, u, ids] = Self::sizes(c, d_model);
        let a = |bytes: u64| Buffer::alloc(dev, bytes);
        let sc = Scratch {
            px: a(px)?,
            x: a(x)?,
            h: a(h)?,
            q: a(q)?,
            k: a(k)?,
            v: a(v)?,
            attn: a(attn)?,
            t: a(t)?,
            u: a(u)?,
            ids: a(ids)?,
        };
        s.upload(&sc.ids, 0, &(0..(ids / 4) as i32).collect::<Vec<_>>())?;
        Ok(sc)
    }
}

/// A weight, its tensor in the file (or the rows of it), and the layer scale folded into its rows.
type Upload = (W, TensorInfo, Option<Arc<[f32]>>);

/// Where each tensor goes in the weight buffer (and the layer scale folded into it), and the f32
/// tensors read on the host.
pub(crate) struct Plan<'a> {
    pub(crate) g: &'a Gguf,
    pub(crate) file: &'a File,
    pub(crate) bytes: u64,
    pub(crate) uploads: Vec<Upload>,
}

impl Plan<'_> {
    /// The weight buffer on `dev`, every planned weight in it.
    pub(crate) fn upload(&self, dev: &Device, s: &Stream) -> Result<Buffer, LoadError> {
        let weights = Buffer::alloc(dev, self.bytes).map_err(|e| fit_err(dev, self.bytes, e))?;
        for (w, info, ls) in &self.uploads {
            let mut bytes = self.read(info)?;
            if let Some(ls) = ls {
                scale_rows(&mut bytes, w, ls);
            }
            s.upload(&weights, w.at, &bytes).map_err(backend)?;
        }
        Ok(weights)
    }

    /// `name` with GGML dims `dims` and dtype `dt`, or `BadGguf`.
    pub(crate) fn find(
        &self,
        name: &str,
        dims: &[u32],
        dt: DType,
    ) -> Result<TensorInfo, LoadError> {
        let t = self
            .g
            .tensor(name)
            .ok_or(LoadError::BadGguf(code::MISSING_TENSOR, 0))?;
        let want: Vec<u64> = dims.iter().map(|&d| u64::from(d)).collect();
        if t.dims != want || crate::gguf::dtype(t.ggml_type)? != dt {
            return Err(LoadError::BadGguf(code::TENSOR, t.offset));
        }
        Ok(t.clone())
    }

    pub(crate) fn alloc(&mut self, dtype: DType, rows: i64, cols: i64) -> W {
        let at = self.bytes.next_multiple_of(256);
        self.bytes = at + dtype.bytes((rows * cols.max(1)) as u64);
        W {
            at,
            dtype,
            rows,
            cols,
        }
    }

    /// An f16 matrix `[rows, cols]` (GGML dims `dims`), each row scaled by `ls` when given.
    pub(crate) fn mat(
        &mut self,
        name: &str,
        rows: i64,
        cols: i64,
        dims: &[u32],
        ls: Option<Arc<[f32]>>,
    ) -> Result<W, LoadError> {
        let t = self.find(name, dims, DType::F16)?;
        let w = self.alloc(DType::F16, rows, cols);
        self.uploads.push((w, t, ls));
        Ok(w)
    }

    /// An f32 vector `[n]`, scaled by `ls` when given.
    pub(crate) fn vec(
        &mut self,
        name: &str,
        n: i64,
        ls: Option<Arc<[f32]>>,
    ) -> Result<W, LoadError> {
        let t = self.find(name, &[n as u32], DType::F32)?;
        let w = self.alloc(DType::F32, n, 0);
        self.uploads.push((w, t, ls));
        Ok(w)
    }

    /// A tensor's bytes from the file (its range was checked against the file at parse).
    pub(crate) fn read(&self, t: &TensorInfo) -> Result<Vec<u8>, LoadError> {
        let mut f = self.file;
        let mut bytes = vec![0u8; t.bytes as usize];
        f.seek(SeekFrom::Start(self.g.data_offset + t.offset))
            .and_then(|_| f.read_exact(&mut bytes))
            .map_err(|_| LoadError::Io)?;
        Ok(bytes)
    }

    /// An f32 tensor read into host memory.
    pub(crate) fn host_f32(&self, name: &str, dims: &[u32]) -> Result<Vec<f32>, LoadError> {
        let t = self.find(name, dims, DType::F32)?;
        let b = self.read(&t)?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// Scales each row r of the matrix or vector `w` (bytes as stored) by `ls[r]`, as llama.cpp's
/// `ggml_mul` by ls1/ls2 after the projection: folded into the weights, f16 rounded once.
fn scale_rows(bytes: &mut [u8], w: &W, ls: &[f32]) {
    match w.dtype {
        DType::F16 => {
            let cols = w.cols as usize;
            for (i, h) in bytes.chunks_exact_mut(2).enumerate() {
                let v = f16_value(u16::from_le_bytes([h[0], h[1]])) * ls[i / cols];
                h.copy_from_slice(&f16_bits(v).to_le_bytes());
            }
        }
        _ => {
            for (i, b) in bytes.chunks_exact_mut(4).enumerate() {
                let v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * ls[i];
                b.copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

/// An f16's exact value.
pub(crate) fn f16_value(h: u16) -> f32 {
    let (e, m) = (i32::from(h >> 10 & 31), f32::from(h & 1023));
    let v = match e {
        31 if m != 0.0 => f32::NAN,
        31 => f32::INFINITY,
        0 => m * 2f32.powi(-24),
        _ => (m + 1024.0) * 2f32.powi(e - 25),
    };
    if h & 0x8000 != 0 { -v } else { v }
}
