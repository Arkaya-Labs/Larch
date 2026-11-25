//! Either image projector the engine pairs with a text model, by its GGUF header: InternVL's
//! (`internvl`, [`crate::vision`]: 448-pixel tiles, sequential positions) or Qwen3-VL's
//! (`qwen3vl_merger`, [`crate::qwen_vision`]: the whole image at about its size, M-RoPE positions).
//!
//! The engine encodes an image in **units** and caches encoded units: an InternVL tile of
//! `tile_tokens` rows, or a Qwen image whole (one unit of all its rows), so a step that reaches an
//! image row encodes the unit holding it at most once.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use larch_kernels::{Device, KError, Stream, Tensor};
use larch_records::LoadError;

use crate::gguf::Gguf;
use crate::qwen_vision::{self, EncodeJob, QwenPixels, QwenVision, QwenVisionConfig};
use crate::vision::{self, Pixels, Rgb, Vision, VisionConfig};

/// Either projector's geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProjectorConfig {
    InternVl(VisionConfig),
    Qwen(QwenVisionConfig),
}

impl ProjectorConfig {
    /// The geometry of the projector `g` describes: `UnsupportedArch` for any other projector type,
    /// the type's own errors for a bad file of a known one.
    pub fn from_gguf(g: &Gguf) -> Result<ProjectorConfig, LoadError> {
        match VisionConfig::from_gguf(g) {
            Err(LoadError::UnsupportedArch) => QwenVisionConfig::from_gguf(g).map(Self::Qwen),
            r => r.map(Self::InternVl),
        }
    }

    /// The projector's output width: the text model's d_model.
    pub fn proj_dim(&self) -> u32 {
        match self {
            Self::InternVl(c) => c.proj_dim,
            Self::Qwen(c) => c.proj_dim,
        }
    }

    /// Embedding rows of a `w` x `h` image.
    pub fn tokens_for(&self, w: u32, h: u32) -> u32 {
        match self {
            Self::InternVl(c) => c.tokens_for(w, h),
            Self::Qwen(c) => c.tokens_for(w, h),
        }
    }

    /// Rows of one encode unit of a `w` x `h` image: a tile's, or the whole image's.
    pub fn unit_rows(&self, w: u32, h: u32) -> u32 {
        match self {
            Self::InternVl(c) => c.tile_tokens(),
            Self::Qwen(c) => c.tokens_for(w, h),
        }
    }

    /// The most rows of one unit, any image: a cache slot's.
    pub fn max_unit_rows(&self) -> u32 {
        match self {
            Self::InternVl(c) => c.tile_tokens(),
            Self::Qwen(c) => c.max_tokens(),
        }
    }

    /// The most rows of one image.
    pub fn max_image_rows(&self) -> u32 {
        match self {
            Self::InternVl(c) => (vision::MAX_TILES + 1) * c.tile_tokens(),
            Self::Qwen(c) => c.max_tokens(),
        }
    }

    /// The (nx, ny) grid of a `w` x `h` image's rows on an M-RoPE model (row i at x = i % nx,
    /// y = i / nx), which then takes max(nx, ny) positions; None for sequential positions.
    pub fn grid_for(&self, w: u32, h: u32) -> Option<(u32, u32)> {
        match self {
            Self::InternVl(_) => None,
            Self::Qwen(c) => {
                let (gx, gy) = c.grid_for(w, h);
                Some((gx / c.merge, gy / c.merge))
            }
        }
    }

    /// The text before and after an image's rows in a prompt, as llama.cpp's mtmd writes them.
    pub fn markers(&self) -> (&'static str, &'static str) {
        match self {
            Self::InternVl(_) => (vision::IMAGE_BEGIN, vision::IMAGE_END),
            Self::Qwen(_) => (qwen_vision::IMAGE_BEGIN, qwen_vision::IMAGE_END),
        }
    }

    /// `img` cut for the encoder.
    pub fn cut(&self, img: &Rgb) -> ImagePixels {
        match self {
            Self::InternVl(c) => ImagePixels::InternVl(Pixels::new(img, c)),
            Self::Qwen(c) => ImagePixels::Qwen(QwenPixels::new(img, c)),
        }
    }

    /// The smallest input the encoder takes, all zeros (one row on Qwen, one tile on InternVL).
    pub fn blank(&self) -> ImagePixels {
        match self {
            Self::InternVl(c) => ImagePixels::InternVl(Pixels {
                tiles: 1,
                patches: vec![0; (c.patches() * c.patch_len()) as usize],
            }),
            Self::Qwen(c) => ImagePixels::Qwen(QwenPixels {
                gx: c.merge,
                gy: c.merge,
                patches: vec![0; (c.merge * c.merge * c.patch_len()) as usize],
            }),
        }
    }

    /// The engine's warm-up input, all zeros. An InternVL tile is always one size, so it is
    /// [`Self::blank`]. On Qwen it is a square of [`WARM_UP_SIDE`]² rows (256, a small screenshot's),
    /// within the cap: after one row, the first real image still paid its GEMMs' and attention's
    /// first launch at real sizes. With 3 s idle after the load, the first image's time to first
    /// token went 456 → 377 ms on Qwen3.5-2B and 364 → 290 on the 0.8B (later images: 208, 135).
    pub fn warm_up_input(&self) -> ImagePixels {
        match self {
            Self::InternVl(_) => self.blank(),
            Self::Qwen(c) => {
                let mut side = WARM_UP_SIDE;
                while side > 1 && side * side > c.max_tokens() {
                    side -= 1;
                }
                let g = side * c.merge;
                ImagePixels::Qwen(QwenPixels {
                    gx: g,
                    gy: g,
                    patches: vec![0; (g * g * c.patch_len()) as usize],
                })
            }
        }
    }
}

/// Rows per side of the Qwen warm-up input ([`ProjectorConfig::warm_up_input`]).
pub const WARM_UP_SIDE: u32 = 16;

/// An image cut for either encoder.
#[derive(Clone, Debug, PartialEq)]
pub enum ImagePixels {
    InternVl(Pixels),
    Qwen(QwenPixels),
}

impl ImagePixels {
    /// Encode units: tiles, or one.
    pub fn units(&self) -> u32 {
        match self {
            Self::InternVl(p) => p.tiles,
            Self::Qwen(_) => 1,
        }
    }

    /// Rows of each of these pixels' units on `c`: a tile's, or the whole image's.
    pub fn unit_rows(&self, c: &ProjectorConfig) -> u32 {
        match (self, c) {
            (Self::Qwen(p), ProjectorConfig::Qwen(q)) => p.tokens(q),
            _ => c.max_unit_rows(),
        }
    }

    /// Whether these are an image of `rows` rows cut for `c`'s encoder: the same projector type,
    /// whole units, and each unit's patches.
    pub fn fits(&self, c: &ProjectorConfig, rows: u32) -> bool {
        match (self, c) {
            (Self::InternVl(p), ProjectorConfig::InternVl(v)) => {
                p.tiles * v.tile_tokens() == rows
                    && p.patches.len() == (p.tiles * v.patches() * v.patch_len()) as usize
            }
            (Self::Qwen(p), ProjectorConfig::Qwen(q)) => {
                let m = q.merge;
                p.gx.is_multiple_of(m)
                    && p.gy.is_multiple_of(m)
                    && p.tokens(q) == rows
                    && rows <= q.max_tokens()
                    && p.patches.len() == (p.gx * p.gy * q.patch_len()) as usize
            }
            _ => false,
        }
    }
}

/// Either projector on a device.
pub enum Projector {
    InternVl(Vision),
    Qwen(QwenVision),
}

impl Projector {
    /// Loads the projector at `path` onto `dev` for a text model of width `d_model`, whichever type
    /// its header names; `UnsupportedArch` for another.
    pub fn load(
        path: &Path,
        dev: &Device,
        s: &Stream,
        d_model: u32,
    ) -> Result<Projector, LoadError> {
        let file = File::open(path).map_err(|_| LoadError::Io)?;
        let len = file.metadata().map_err(|_| LoadError::Io)?.len();
        let g = Gguf::read(BufReader::new(&file), len)?;
        match ProjectorConfig::from_gguf(&g)? {
            ProjectorConfig::InternVl(_) => Vision::load(path, dev, s, d_model).map(Self::InternVl),
            ProjectorConfig::Qwen(_) => QwenVision::load(path, dev, s, d_model).map(Self::Qwen),
        }
    }

    /// The width of the rows the projector at `path` writes (its `projection_dim`), from its header:
    /// the text model's `d_model` it must go with.
    pub fn width(path: &Path) -> Result<u32, LoadError> {
        let file = File::open(path).map_err(|_| LoadError::Io)?;
        let len = file.metadata().map_err(|_| LoadError::Io)?.len();
        Ok(ProjectorConfig::from_gguf(&Gguf::read(BufReader::new(&file), len)?)?.proj_dim())
    }

    /// The projector's geometry.
    pub fn config(&self) -> ProjectorConfig {
        match self {
            Self::InternVl(v) => ProjectorConfig::InternVl(*v.config()),
            Self::Qwen(v) => ProjectorConfig::Qwen(*v.config()),
        }
    }

    /// Encodes unit `unit` of `px` into `out`, f32 `[unit rows, proj_dim]`: tile `unit` (the
    /// overview the last), or the whole Qwen image (unit 0). `InvalidArg` for pixels of the other
    /// projector or a unit past them.
    pub fn encode_unit(
        &self,
        s: &Stream,
        px: &ImagePixels,
        unit: u32,
        out: &Tensor,
    ) -> Result<(), KError> {
        match (self, px) {
            (Self::InternVl(v), ImagePixels::InternVl(p)) => v.encode_tile(s, p, unit, out),
            (Self::Qwen(v), ImagePixels::Qwen(p)) if unit == 0 => v.encode(s, p, out),
            _ => Err(KError::InvalidArg),
        }
    }

    /// Whether a unit can be encoded some layers at a time ([`Self::begin`]): a Qwen image can, an
    /// InternVL tile (about 110 ms on a 3050) encodes in one call.
    pub fn sliced(&self) -> bool {
        matches!(self, Self::Qwen(_))
    }

    /// The encoder's layers: what [`Self::advance`] counts.
    pub fn layers(&self) -> u32 {
        match self {
            Self::InternVl(v) => v.config().n_layer,
            Self::Qwen(v) => v.config().n_layer,
        }
    }

    /// Starts encoding unit `unit` of `px` in slices (see [`QwenVision::begin`]): `Unsupported` for a
    /// projector that does not slice, `InvalidArg` for pixels of the other projector or a unit past
    /// them. The projector's scratch holds the encode until [`Self::finish`].
    pub fn begin(&self, s: &Stream, px: &ImagePixels, unit: u32) -> Result<EncodeJob, KError> {
        match (self, px) {
            (Self::Qwen(v), ImagePixels::Qwen(p)) if unit == 0 => v.begin(s, p),
            (Self::Qwen(_), _) => Err(KError::InvalidArg),
            _ => Err(KError::Unsupported),
        }
    }

    /// Runs up to `layers` more of `job`'s layers; whether all have run.
    pub fn advance(&self, s: &Stream, job: &mut EncodeJob, layers: u32) -> Result<bool, KError> {
        match self {
            Self::Qwen(v) => v.advance(s, job, layers),
            Self::InternVl(_) => Err(KError::Unsupported),
        }
    }

    /// Writes `job`'s rows, every layer run, into `out`, f32 `[unit rows, proj_dim]`.
    pub fn finish(&self, s: &Stream, job: EncodeJob, out: &Tensor) -> Result<(), KError> {
        match self {
            Self::Qwen(v) => v.finish(s, job, out),
            Self::InternVl(_) => Err(KError::Unsupported),
        }
    }

    /// The i32 identity `0..n` on the device, for gathering `n` rows of an image's embeddings
    /// (`n` at most the most rows one image takes).
    pub fn ids(&self, n: u32) -> Result<Tensor<'_>, KError> {
        match self {
            Self::InternVl(v) => v.ids(n),
            Self::Qwen(v) => v.ids(n),
        }
    }
}
