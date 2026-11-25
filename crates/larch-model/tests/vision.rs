//! Image input on the CPU reference: a generated InternVL projector against its graph written out
//! naively in f64 (llama.cpp's order: CLS last, position row p for patch p, tanh GELU, layer scale,
//! the pixel shuffle's channel order), the mmproj trust boundary, llava-uhd's grid, and decoding.

use std::sync::Arc;

use larch_kernels::{Buffer, CPU, DType, Device, Stream, Tensor};
use larch_model::gguf::{Value, code};
use larch_model::tiny;
use larch_model::vision::{
    IMAGE_TOKEN, Image, ImageError, MAX_TILES, Pixels, Rgb, Vision, VisionConfig, image_tokens,
};
use larch_records::LoadError;

const E: usize = tiny::VIT as usize;
const FF: usize = tiny::VIT_FFN as usize;
const LAYERS: usize = tiny::VIT_LAYERS as usize;
const SIZE: u32 = tiny::TILE;
const PATCH: u32 = tiny::PATCH;
const DM: usize = 32;

/// The generated projector, its tensors kept as values for the reference.
type Tiny = tiny::Projector;

/// A projector `DM` wide whose `key` is `v`.
fn with_key(t: &Tiny, key: &str, v: Value) -> Vec<u8> {
    let mut t2 = Tiny {
        tensors: t.tensors.clone(),
        kv: t.kv.clone(),
    };
    t2.kv.iter_mut().find(|e| e.0 == key).unwrap().1 = v;
    t2.gguf()
}

/// `bytes` in a fresh temporary file, which lives as long as the path.
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(bytes: &[u8]) -> TempFile {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("larch-vision-{}-{n}.gguf", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        TempFile(p)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn load(bytes: &[u8], d_model: u32) -> Result<(Device, Stream, Vision), LoadError> {
    let f = TempFile::new(bytes);
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    let v = Vision::load(&f.0, &dev, &s, d_model)?;
    Ok((dev, s, v))
}

fn layernorm(x: &[f64], w: &[f32], b: &[f32], eps: f64) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    x.iter()
        .enumerate()
        .map(|(i, v)| (v - mean) / (var + eps).sqrt() * f64::from(w[i]) + f64::from(b[i]))
        .collect()
}

/// `w` `[rows, cols]` (rows the output) times x, plus b.
fn linear(w: &[f32], b: &[f32], x: &[f64]) -> Vec<f64> {
    let cols = x.len();
    (0..b.len())
        .map(|r| {
            f64::from(b[r])
                + (0..cols)
                    .map(|c| f64::from(w[r * cols + c]) * x[c])
                    .sum::<f64>()
        })
        .collect()
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// The projector on one 56 x 56 tile, as llama.cpp's graph reads: f64 throughout, with the pixels
/// rounded to f16 as its im2col rounds them.
fn reference(t: &Tiny, tile: &Rgb, c: &VisionConfig) -> Vec<Vec<f64>> {
    let (g, p) = ((SIZE / PATCH) as usize, PATCH as usize);
    let n = g * g;
    let px = |x: usize, y: usize, ch: usize| {
        let v = (f32::from(tile.data[(y * SIZE as usize + x) * 3 + ch]) / 255.0 - c.mean[ch])
            / c.std[ch];
        f64::from(half(v))
    };
    let (pw, pb, cls, pos) = (
        t.get("v.patch_embd.weight"),
        t.get("v.patch_embd.bias"),
        t.get("v.class_embd"),
        t.get("v.position_embd.weight"),
    );
    // Rows 0..n the patches in raster order (patch p = py · g + px), row n the CLS token.
    let mut x: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let (gx, gy) = (i % g, i / g);
            let v: Vec<f64> = (0..3)
                .flat_map(|ch| (0..p).flat_map(move |ky| (0..p).map(move |kx| (ch, ky, kx))))
                .map(|(ch, ky, kx)| px(gx * p + kx, gy * p + ky, ch))
                .collect();
            linear(pw, pb, &v)
        })
        .collect();
    x.push(cls.iter().map(|&v| f64::from(v)).collect());
    for (r, row) in x.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v += f64::from(pos[r * E + j]);
        }
    }
    for l in 0..LAYERS {
        let w = |s: &str| t.get(&format!("v.blk.{l}.{s}"));
        let h: Vec<_> = x
            .iter()
            .map(|r| layernorm(r, w("ln1.weight"), w("ln1.bias"), 1e-6))
            .collect();
        let q: Vec<_> = h
            .iter()
            .map(|r| linear(w("attn_q.weight"), w("attn_q.bias"), r))
            .collect();
        let k: Vec<_> = h
            .iter()
            .map(|r| linear(w("attn_k.weight"), w("attn_k.bias"), r))
            .collect();
        let v: Vec<_> = h
            .iter()
            .map(|r| linear(w("attn_v.weight"), w("attn_v.bias"), r))
            .collect();
        for i in 0..x.len() {
            let s: Vec<f64> = k
                .iter()
                .map(|kr| q[i].iter().zip(kr).map(|(a, b)| a * b).sum::<f64>() / (E as f64).sqrt())
                .collect();
            let mx = s.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
            let z: f64 = e.iter().sum();
            let a: Vec<f64> = (0..E)
                .map(|j| (0..x.len()).map(|r| e[r] / z * v[r][j]).sum())
                .collect();
            let o = linear(w("attn_out.weight"), w("attn_out.bias"), &a);
            for j in 0..E {
                x[i][j] += o[j] * f64::from(w("ls1.weight")[j]);
            }
        }
        for row in x.iter_mut() {
            let h = layernorm(row, w("ln2.weight"), w("ln2.bias"), 1e-6);
            let u: Vec<f64> = linear(w("ffn_up.weight"), w("ffn_up.bias"), &h)
                .into_iter()
                .map(gelu)
                .collect();
            let d = linear(w("ffn_down.weight"), w("ffn_down.bias"), &u);
            for j in 0..E {
                row[j] += d[j] * f64::from(w("ls2.weight")[j]);
            }
        }
    }
    // Pixel shuffle: token (py / 2) (g / 2) + px / 2, channel c + E (px % 2) + 2 E (py % 2).
    let mut merged = vec![vec![0.0; 4 * E]; n / 4];
    for (i, row) in x.iter().take(n).enumerate() {
        let (gx, gy) = (i % g, i / g);
        let tok = (gy / 2) * (g / 2) + gx / 2;
        let base = E * (gx % 2) + 2 * E * (gy % 2);
        merged[tok][base..base + E].copy_from_slice(row);
    }
    merged
        .iter()
        .map(|m| {
            let h = layernorm(
                m,
                t.get("mm.model.mlp.0.weight"),
                t.get("mm.model.mlp.0.bias"),
                1e-5,
            );
            let y: Vec<f64> = linear(
                t.get("mm.model.mlp.1.weight"),
                t.get("mm.model.mlp.1.bias"),
                &h,
            )
            .into_iter()
            .map(gelu)
            .collect();
            linear(
                t.get("mm.model.mlp.3.weight"),
                t.get("mm.model.mlp.3.bias"),
                &y,
            )
        })
        .collect()
}

/// `v` rounded to f16 and back.
fn half(v: f32) -> f32 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    let e = v.abs().log2().floor().max(-14.0);
    let q = 2f32.powi(e as i32 - 10);
    (v / q).round_ties_even() * q
}

/// A tile with structure in every channel, so a wrong patch or channel order shows.
fn pattern(w: u32, h: u32) -> Rgb {
    let mut img = Rgb::filled(w, h, [0, 0, 0]);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = (y * w as usize + x) * 3;
            img.data[i] = (x * 7 + y * 3) as u8;
            img.data[i + 1] = (x * y % 251) as u8;
            img.data[i + 2] = 255u8.wrapping_sub((x * 2 + y) as u8);
        }
    }
    img
}

fn encode(v: &Vision, s: &Stream, dev: &Device, px: &Pixels) -> Vec<f32> {
    let c = v.config();
    let rows = (px.tiles * c.tile_tokens()) as i64;
    let buf = Buffer::alloc(dev, (rows * DM as i64 * 4) as u64).unwrap();
    let out = Tensor::new(&buf, 0, DType::F32, &[rows, DM as i64]).unwrap();
    v.encode(s, px, &out).unwrap();
    let mut host = vec![0f32; (rows * DM as i64) as usize];
    s.download(&buf, 0, &mut host).unwrap();
    host
}

#[test]
fn encoder_matches_the_graph_written_out() {
    let t = Tiny::new(DM as u32, 0x9e37);
    let (dev, s, v) = load(&t.gguf(), DM as u32).unwrap();
    let c = *v.config();
    assert_eq!((c.grid(), c.tile_tokens()), (4, 4));
    let img = pattern(SIZE, SIZE);
    let px = Pixels::new(&img, &c);
    assert_eq!(px.tiles, 1, "a tile-sized image is its overview alone");
    let got = encode(&v, &s, &dev, &px);
    let want = reference(&t, &img, &c);
    for (r, w) in want.iter().enumerate() {
        let g = &got[r * DM..(r + 1) * DM];
        let dot: f64 = g.iter().zip(w).map(|(a, b)| f64::from(*a) * b).sum();
        let (ng, nw) = (
            g.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt(),
            w.iter().map(|b| b * b).sum::<f64>().sqrt(),
        );
        let cos = dot / (ng * nw);
        let max = g
            .iter()
            .zip(w)
            .map(|(a, b)| (f64::from(*a) - b).abs())
            .fold(0.0, f64::max);
        // f16 activations at every linear input against f64: the order of the patches, channels,
        // CLS and position rows would each move a row far past this.
        assert!(
            cos > 0.9995 && max < 0.05 * nw / (DM as f64).sqrt() + 1e-2,
            "token {r}: cos {cos}, max {max}"
        );
    }
}

#[test]
fn a_projector_of_another_width_is_refused() {
    let t = Tiny::new(DM as u32, 1);
    let e = load(&t.gguf(), DM as u32 + 1).err();
    assert_eq!(e, Some(LoadError::BadGguf(code::MISMATCH, 0)));
}

#[test]
fn bad_projectors_are_load_errors() {
    let t = Tiny::new(DM as u32, 2);
    let with = |key: &str, v: Value| with_key(&t, key, v);
    let bad_key = LoadError::BadGguf(code::KEY, 0);
    for (key, v, want) in [
        (
            "clip.projector_type",
            Value::Str("qwen3vl_merger".into()),
            LoadError::UnsupportedArch,
        ),
        (
            "clip.use_gelu",
            Value::Bool(false),
            LoadError::UnsupportedArch,
        ),
        ("clip.vision.image_size", Value::U32(0), bad_key),
        ("clip.vision.image_size", Value::U32(50), bad_key), // not whole patches
        ("clip.vision.patch_size", Value::Str("14".into()), bad_key),
        ("clip.vision.attention.head_count", Value::U32(3), bad_key),
        ("clip.vision.block_count", Value::U32(u32::MAX), bad_key),
        (
            "clip.vision.image_std",
            Value::Arr(vec![Value::F32(0.0); 3]),
            bad_key,
        ),
        (
            "clip.vision.image_mean",
            Value::Arr(vec![Value::F32(0.5); 2]),
            bad_key,
        ),
        (
            "clip.vision.attention.layer_norm_epsilon",
            Value::F32(f32::NAN),
            bad_key,
        ),
        // Layers the file does not hold.
        (
            "clip.vision.block_count",
            Value::U32(3),
            LoadError::BadGguf(code::MISSING_TENSOR, 0),
        ),
    ] {
        assert_eq!(
            load(&with(key, v.clone()), DM as u32).err(),
            Some(want),
            "{key} = {v:?}"
        );
    }
    // A tensor of the wrong shape or dtype.
    let mut wrong = Tiny::new(DM as u32, 2);
    wrong
        .tensors
        .iter_mut()
        .find(|e| e.0 == "v.blk.1.ffn_up.weight")
        .unwrap()
        .1 = vec![FF as u64, E as u64];
    assert!(
        matches!(load(&wrong.gguf(), DM as u32).err(), Some(LoadError::BadGguf(c, _)) if c == code::TENSOR)
    );
    let mut f32_mat = Tiny::new(DM as u32, 2);
    f32_mat
        .tensors
        .iter_mut()
        .find(|e| e.0 == "mm.model.mlp.3.weight")
        .unwrap()
        .3 = false;
    assert!(
        matches!(load(&f32_mat.gguf(), DM as u32).err(), Some(LoadError::BadGguf(c, _)) if c == code::TENSOR)
    );
    // Truncated: the tensor table points past the end.
    let full = t.gguf();
    assert!(matches!(
        load(&full[..full.len() - 100], DM as u32).err(),
        Some(LoadError::BadGguf(..))
    ));
}

/// The real model's geometry (InternVL3-1B's mmproj keys).
fn internvl() -> VisionConfig {
    VisionConfig {
        image_size: 448,
        patch: 14,
        n_embd: 1024,
        n_ff: 4096,
        n_head: 16,
        n_layer: 24,
        eps: 1e-6,
        proj_dim: 896,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
        merge: 2,
        min_tiles: 1,
        max_tiles: 12,
    }
}

#[test]
fn tiles_follow_llava_uhd_and_are_capped() {
    let c = internvl();
    assert_eq!(c.tile_tokens(), 256);
    // Small images: the overview alone.
    for (w, h) in [(1, 1), (448, 448), (448, 10), (10, 448)] {
        assert_eq!(
            (c.grid_for(w, h), c.tokens_for(w, h)),
            ((0, 0), 256),
            "{w}x{h}"
        );
    }
    // The spec's example, 1000 x 600: 3 x 2 tiles and the overview.
    assert_eq!(c.grid_for(1000, 600), (3, 2));
    assert_eq!(c.tokens_for(1000, 600), 7 * 256);
    assert_eq!(c.grid_for(600, 1000), (2, 3));
    // llava's rule takes the most effective pixels first: 2 x 1 tiles would shrink this image, 3 x 2
    // hold all of it.
    assert_eq!(c.grid_for(900, 449), (3, 2));
    // No image takes more than MAX_TILES tiles and its overview, whatever its shape.
    for (w, h) in [
        (10_000, 10),
        (10, 10_000),
        (16_384, 2441),
        (4000, 3000),
        (449, 449),
    ] {
        let (cols, rows) = c.grid_for(w, h);
        assert!(
            cols * rows <= MAX_TILES && cols * rows >= 1,
            "{w}x{h}: {cols}x{rows}"
        );
        assert!(c.tokens_for(w, h) <= (MAX_TILES + 1) * 256);
    }
    assert_eq!(c.grid_for(10_000, 10), (6, 1));
}

#[test]
fn pixels_cut_tiles_then_the_overview() {
    let c = internvl();
    let img = pattern(1000, 600);
    let px = Pixels::new(&img, &c);
    assert_eq!(px.tiles, 7);
    assert_eq!(px.patches.len(), 7 * 1024 * 588);
    // A uniform image normalizes to one value per channel everywhere, padding aside.
    let grey = Pixels::new(&Rgb::filled(300, 200, [128, 128, 128]), &c);
    let v = (128.0f32 / 255.0 - 0.485) / 0.229;
    assert_eq!(grey.tiles, 1);
    assert!(grey.patches[..196].iter().all(|&h| h == grey.patches[0]));
    assert!((half(v) - v).abs() < 1e-3);
    // 10000 x 10 and 1 x 1 are fine.
    assert_eq!(Pixels::new(&pattern(1, 1), &c).tiles, 1);
    assert_eq!(
        Pixels::new(&Rgb::filled(10_000, 10, [9, 9, 9]), &c).tiles,
        7
    );
}

fn png(img: image::DynamicImage) -> Vec<u8> {
    let mut b = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut b), image::ImageFormat::Png)
        .unwrap();
    b
}

fn img(bytes: Vec<u8>) -> Result<Image, ImageError> {
    Image::new(Arc::from(bytes))
}

/// 16-bit images keep each sample's high byte, as stb_image (llama.cpp's decoder) does: RGBA drops
/// alpha, grey with alpha repeats the grey.
#[test]
fn sixteen_bit_samples_keep_their_high_byte() {
    use image::{DynamicImage, ImageBuffer, ImageFormat, LumaA, Rgba};
    let png = |d: DynamicImage| {
        let mut out = Vec::new();
        d.write_to(&mut std::io::Cursor::new(&mut out), ImageFormat::Png)
            .unwrap();
        out
    };
    let rgba = ImageBuffer::<Rgba<u16>, _>::from_fn(3, 2, |x, y| {
        Rgba([
            x as u16 * 0x1234 + 0xff,
            y as u16 * 0x4321 + 0x180,
            0xfedc,
            1,
        ])
    });
    let got = img(png(DynamicImage::ImageRgba16(rgba.clone())))
        .unwrap()
        .decode()
        .unwrap();
    let want: Vec<u8> = (rgba.pixels())
        .flat_map(|p| [(p[0] >> 8) as u8, (p[1] >> 8) as u8, (p[2] >> 8) as u8])
        .collect();
    assert_eq!((got.width, got.height, got.data), (3, 2, want));
    let grey =
        ImageBuffer::<LumaA<u16>, _>::from_fn(2, 1, |x, _| LumaA([0x8001 + x as u16 * 0x100, 7]));
    let got = img(png(DynamicImage::ImageLumaA16(grey)))
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(got.data, [0x80, 0x80, 0x80, 0x81, 0x81, 0x81]);
}

/// A PNG whose header claims `w` x `h` (its CRC right), with no image data behind it.
fn png_header(w: u32, h: u32) -> Vec<u8> {
    let mut ihdr = b"IHDR".to_vec();
    ihdr.extend(w.to_be_bytes());
    ihdr.extend(h.to_be_bytes());
    ihdr.extend([8, 2, 0, 0, 0]);
    let mut b = b"\x89PNG\r\n\x1a\n".to_vec();
    b.extend(13u32.to_be_bytes());
    b.extend(&ihdr);
    b.extend(crc32(&ihdr).to_be_bytes());
    // The decoder reads the header up to the first data chunk; give it one.
    let idat = b"IDAT\x78\x9c\x03\x00\x00\x00\x00\x01";
    b.extend(((idat.len() - 4) as u32).to_be_bytes());
    b.extend(idat);
    b.extend(crc32(idat).to_be_bytes());
    b
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
    }
    !c
}

#[test]
fn bad_files_are_errors_never_panics() {
    assert_eq!(img(vec![]), Err(ImageError::Empty));
    assert!(matches!(
        img(vec![0; (32 << 20) + 1]),
        Err(ImageError::TooManyBytes(_))
    ));
    assert_eq!(
        img(b"<svg xmlns='http://www.w3.org/2000/svg'/>".to_vec()),
        Err(ImageError::Unsupported("SVG"))
    );
    assert_eq!(
        img(b"  <?xml version='1.0'?><svg/>".to_vec()),
        Err(ImageError::Unsupported("SVG"))
    );
    assert_eq!(
        img(b"\0\0\0\x18ftypheic\0\0\0\0mif1heic".to_vec()),
        Err(ImageError::Unsupported("HEIC"))
    );
    assert!(matches!(
        img(b"just some text, saved as photo.png".to_vec()),
        Err(ImageError::Corrupt(_))
    ));
    // Truncated: the header reads, the pixels do not.
    let good = png(image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        64,
        64,
        image::Rgb([1, 2, 3]),
    )));
    let cut = img(good[..good.len() / 2].to_vec());
    assert!(cut.is_err() || cut.unwrap().decode().is_err());
    // A decompression bomb: refused from the header, before any pixel is allocated.
    assert_eq!(
        img(png_header(10_000, 10_000)),
        Err(ImageError::Size(10_000, 10_000))
    );
    // Past the decoder's side limit before the size is read: no "0x0 pixels" in the message.
    let wide = img(png_header(100_000, 1));
    assert_eq!(wide, Err(ImageError::TooLarge));
    assert!(!wide.unwrap_err().to_string().contains("0x0"));
    assert_eq!(
        img(png_header(0, 5)).err().map(|e| matches!(
            e,
            ImageError::Size(..) | ImageError::TooLarge | ImageError::Corrupt(_)
        )),
        Some(true)
    );
    // Extremes that are fine.
    for (w, h) in [(1, 1), (10_000, 10)] {
        let b = png(image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            w,
            h,
            image::Rgb([5, 6, 7]),
        )));
        let i = img(b).unwrap();
        assert_eq!((i.width(), i.height()), (w, h));
        assert_eq!(&i.decode().unwrap().data[..3], &[5, 6, 7]);
    }
}

#[test]
fn decoding_keeps_stb_images_rgb() {
    // Transparency: alpha dropped, the colour under it kept (stb_image with 3 channels).
    let mut rgba = image::RgbaImage::from_pixel(2, 1, image::Rgba([200, 100, 50, 255]));
    rgba.put_pixel(1, 0, image::Rgba([10, 20, 30, 0]));
    let d = img(png(image::DynamicImage::ImageRgba8(rgba)))
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(d.data, vec![200, 100, 50, 10, 20, 30]);
    // Grey replicated; 16 bits by the high byte.
    let grey = image::GrayImage::from_pixel(1, 1, image::Luma([77]));
    assert_eq!(
        img(png(image::DynamicImage::ImageLuma8(grey)))
            .unwrap()
            .decode()
            .unwrap()
            .data,
        vec![77; 3]
    );
    let deep = image::ImageBuffer::<image::Rgb<u16>, _>::from_pixel(
        1,
        1,
        image::Rgb([0x12ab, 0xff00, 0x00ff]),
    );
    assert_eq!(
        img(png(image::DynamicImage::ImageRgb16(deep)))
            .unwrap()
            .decode()
            .unwrap()
            .data,
        vec![0x12, 0xff, 0x00]
    );
    // The same bytes are the same image; other bytes another.
    let a = png(image::DynamicImage::ImageLuma8(
        image::GrayImage::from_pixel(3, 3, image::Luma([1])),
    ));
    let b = png(image::DynamicImage::ImageLuma8(
        image::GrayImage::from_pixel(3, 3, image::Luma([2])),
    ));
    assert_eq!(img(a.clone()).unwrap().digest(), img(a).unwrap().digest());
    assert_ne!(
        img(b.clone()).unwrap().hex(),
        img(png(image::DynamicImage::ImageLuma8(
            image::GrayImage::from_pixel(3, 3, image::Luma([1]))
        )))
        .unwrap()
        .hex()
    );
    let _ = b;
}

/// The fixtures in tests/data/vision, made with Pillow (`make.py` beside them).
fn fixture(name: &str) -> Image {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/vision")
        .join(name);
    img(std::fs::read(p).unwrap()).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn every_format_decodes() {
    // 4 x 2 images, left half red, right half blue, unless said.
    let red_left = |d: &Rgb, tol: i32| {
        let px = |x: usize, y: usize| &d.data[(y * d.width as usize + x) * 3..][..3];
        let near = |p: &[u8], c: [i32; 3]| {
            p.iter()
                .zip(c)
                .all(|(&a, b)| (i32::from(a) - b).abs() <= tol)
        };
        near(px(0, 0), [255, 0, 0])
            && near(px(d.width as usize - 1, d.height as usize - 1), [0, 0, 255])
    };
    for (name, tol) in [
        ("rgba.png", 0),
        ("palette.png", 0),
        ("grey16.png", 255), // a grey image: only decoding is checked
        ("progressive.jpg", 40),
        ("cmyk.jpg", 60),
        ("lossy.webp", 40),
        ("lossless.webp", 0),
        ("animated.gif", 0),
    ] {
        let i = fixture(name);
        assert_eq!((i.width(), i.height()), (4, 2), "{name}");
        let d = i.decode().unwrap();
        assert!(red_left(&d, tol), "{name}: {:?}", &d.data[..6]);
    }
    // EXIF orientation 6 (rotate 90 clockwise): stored 4 x 2, shown 2 x 4, red on top.
    let i = fixture("exif6.jpg");
    assert_eq!((i.width(), i.height()), (2, 4));
    let d = i.decode().unwrap();
    assert_eq!((d.width, d.height), (2, 4));
    assert!(d.data[0] > 200 && d.data[2] < 60, "{:?}", &d.data[..3]);
}

/// The image gate (architecture 15, for images) on GPU 0 against llama.cpp's `llama-logits --mmproj`
/// dump: per image, every embedding row's cosine with the reference's and the largest absolute
/// difference; per prompt position, the oracle's KL and top-1 rule. Runs when LARCH_VISION_ORACLE
/// names a folder holding `prompts.jsonl` and its images (with their `.oracle.png` twins),
/// `reference.jsonl` (the dump, its embedding files beside it) and `bounds.json` (calibrated on
/// llama.cpp against itself, see tests/data/vision-bounds.json), with LARCH_VISION_MODEL and
/// LARCH_VISION_MMPROJ the GGUFs.
#[test]
#[ignore = "a GPU and the model files: LARCH_VISION_ORACLE, LARCH_VISION_MODEL, LARCH_VISION_MMPROJ"]
fn image_oracle() {
    use larch_model::oracle::{self, Dump, IMAGE_ROW, Top};
    use larch_model::{EmbedRun, StepBatch, Transformer};
    use std::path::PathBuf;

    let var =
        |k: &str| PathBuf::from(std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set")));
    let (dir, model, mmproj) = (
        var("LARCH_VISION_ORACLE"),
        var("LARCH_VISION_MODEL"),
        var("LARCH_VISION_MMPROJ"),
    );
    let bounds: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("bounds.json")).unwrap()).unwrap();
    let f = |a: &str, b: &str| {
        bounds[a][b]
            .as_f64()
            .unwrap_or_else(|| panic!("bounds.json: {a}.{b}"))
    };
    // LARCH_VISION_MUTATION breaks one step on purpose, and the gate must then fail: proof that each
    // check sees the bug it is there for. `tile-order` reverses an image's grid tiles,
    // `overview` blanks the thumbnail tile, `normalization` uses CLIP's mean and std for
    // InternVL's, `shuffle-transpose` transposes each tile's 16 x 16 rows, `offset` puts the image
    // one position late.
    let mutation = std::env::var("LARCH_VISION_MUTATION").ok();
    let (min_cos, max_abs, bound, margin) = (
        f("embd", "min_cos"),
        f("embd", "max_abs"),
        f("logits", "bound"),
        f("logits", "margin"),
    );
    let sha = oracle::sha256(&model).unwrap();
    let dump = Dump::read(
        std::io::BufReader::new(std::fs::File::open(dir.join("reference.jsonl")).unwrap()),
        &sha,
    )
    .unwrap();

    const CHUNK: i64 = 64;
    let dev = Device::open(0).unwrap();
    let s = Stream::new(&dev).unwrap();
    let mut text = Transformer::load(&model, &dev, &s, CHUNK as u32).unwrap();
    let c = larch_model::Model::config(&text).clone();
    let vision = Vision::load(&mmproj, &dev, &s, c.d_model).unwrap();
    let (d, vocab) = (c.d_model as usize, c.vocab as usize);
    let longest = dump.prompts.iter().map(|p| p.tokens.len()).max().unwrap() as i64;
    let blocks = (longest + 15) / 16;
    let kv_bytes = (i64::from(c.kv_layers)
        * blocks
        * 16
        * i64::from(c.n_kv_heads)
        * i64::from(c.head_dim)
        * 2) as u64;
    let (k, v) = (
        Buffer::alloc(&dev, kv_bytes).unwrap(),
        Buffer::alloc(&dev, kv_bytes).unwrap(),
    );
    let pool = larch_kernels::KvPool {
        k: &k,
        v: &v,
        n_blocks: blocks as u32,
        block_tokens: 16,
        n_layers: c.kv_layers,
        n_kv_heads: c.n_kv_heads,
        head_dim: c.head_dim,
        dtype: DType::F16,
    };
    let input = Buffer::alloc(&dev, ((CHUNK * 4 + blocks + 4) * 4) as u64).unwrap();
    let logits = Buffer::alloc(&dev, (CHUNK as usize * vocab * 4) as u64).unwrap();
    let (mut worst_cos, mut worst_abs, mut encode_ms, mut tiles) = (1.0f64, 0.0f64, 0.0f64, 0u32);
    // Each image file's worst row, so a failure names the image.
    let mut per_file: std::collections::BTreeMap<String, (f64, f64)> = Default::default();
    let mut run: Vec<Vec<Top>> = Vec::new();
    for (pi, p) in dump.prompts.iter().enumerate() {
        // Encode each image and hold its rows on the device for the forward passes.
        let mut held = Vec::new();
        for im in &p.images {
            // llama.cpp read `<image>.oracle.png`, the pixels larch means to show where stb cannot
            // (EXIF orientation, WebP, a GIF's transparent index; images/make.py): larch reads the image.
            let file = im.file.strip_suffix(".oracle.png").unwrap_or(&im.file);
            let bytes = std::fs::read(dir.join(file)).unwrap();
            let mut px = Pixels::new(&img(bytes).unwrap().decode().unwrap(), vision.config());
            let tile_len = (vision.config().patches() * vision.config().patch_len()) as usize;
            match mutation.as_deref() {
                // The grid's tiles in reverse (a raster-order bug); the overview stays last. Swapping
                // only the first two missed: wide.png's are both plain sky.
                Some("tile-order") if px.tiles > 2 => {
                    let grid = (px.tiles - 1) as usize;
                    let tiles: Vec<Vec<u16>> = px.patches[..grid * tile_len]
                        .chunks_exact(tile_len)
                        .rev()
                        .map(<[u16]>::to_vec)
                        .collect();
                    px.patches[..grid * tile_len].copy_from_slice(&tiles.concat());
                }
                Some("overview") => {
                    let n = px.patches.len();
                    px.patches[n - tile_len..].fill(0);
                }
                // ImageNet's mean 0.449 and std 0.226 undone, CLIP's 0.459 and 0.2686 applied.
                Some("normalization") => {
                    for h in &mut px.patches {
                        *h = f32_to_f16((f16_to_f32(*h) * 0.226 + 0.449 - 0.459) / 0.2686);
                    }
                }
                _ => {}
            }
            assert_eq!(
                (px.tiles * vision.config().tile_tokens()) as usize,
                im.n,
                "prompt {pi}: {}",
                im.file
            );
            let buf = Buffer::alloc(&dev, (im.n * d * 4) as u64).unwrap();
            let out = Tensor::new(&buf, 0, DType::F32, &[im.n as i64, d as i64]).unwrap();
            let t0 = larch_kernels::Event::record(&s).unwrap();
            vision.encode(&s, &px, &out).unwrap();
            let t1 = larch_kernels::Event::record(&s).unwrap();
            let mut got = vec![0f32; im.n * d];
            s.download(&buf, 0, &mut got).unwrap();
            if mutation.as_deref() == Some("shuffle-transpose") {
                let tt = vision.config().tile_tokens() as usize;
                let side = (tt as f64).sqrt() as usize;
                let old = got.clone();
                for t in 0..im.n / tt {
                    for (r, c) in (0..side).flat_map(|r| (0..side).map(move |c| (r, c))) {
                        let (to, from) = (t * tt + r * side + c, t * tt + c * side + r);
                        got[to * d..(to + 1) * d].copy_from_slice(&old[from * d..(from + 1) * d]);
                    }
                }
                s.upload(&buf, 0, &got).unwrap();
            }
            encode_ms += t0.elapsed_ns(&t1).unwrap() as f64 / 1e6;
            tiles += px.tiles;
            let want: Vec<f32> = std::fs::read(dir.join(&im.embd))
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            assert_eq!(want.len(), got.len());
            let per = per_file.entry(im.file.clone()).or_insert((1.0f64, 0.0f64));
            for (g, w) in got.chunks_exact(d).zip(want.chunks_exact(d)) {
                let dot: f64 = g
                    .iter()
                    .zip(w)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                let n = |x: &[f32]| x.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
                per.0 = per.0.min(dot / (n(g) * n(w)));
                per.1 = g
                    .iter()
                    .zip(w)
                    .map(|(a, b)| f64::from(a - b).abs())
                    .fold(per.1, f64::max);
            }
            worst_cos = worst_cos.min(per.0);
            worst_abs = worst_abs.max(per.1);
            held.push((im, buf));
        }
        // The prompt as prefill chunks, every row's logits.
        let len = p.tokens.len() as i64;
        let mut rows: Vec<Top> = Vec::new();
        for start in (0..len).step_by(CHUNK as usize) {
            let m = CHUNK.min(len - start);
            let mut packed: Vec<i32> = Vec::new();
            packed.extend(
                p.tokens[start as usize..(start + m) as usize]
                    .iter()
                    .map(|&t| if t == IMAGE_ROW { 0 } else { t as i32 }),
            );
            packed.extend((start..start + m).map(|x| x as i32)); // positions
            packed.extend((start..start + m).map(|x| x as i32)); // slots: block b holds positions 16b..
            packed.extend(0..blocks as i32); // the block table
            packed.push(start as i32);
            packed.push(m as i32);
            packed.extend(0..m as i32); // logits rows
            s.upload(&input, 0, &packed).unwrap();
            let at = |i: i64| (i * 4) as u64;
            let t =
                |off: i64, shape: &[i64]| Tensor::new(&input, at(off), DType::I32, shape).unwrap();
            let batch = StepBatch {
                token_ids: t(0, &[m]),
                positions: t(m, &[m]),
                slot_ids: t(2 * m, &[m]),
                decode_rows: 0,
                prefill_rows: m as u32,
                block_table: t(3 * m, &[1, blocks]),
                seq_lens: t(0, &[0]),
                prefill_q_start: t(3 * m + blocks, &[1]),
                prefill_q_len: t(3 * m + blocks + 1, &[1]),
                logits_rows: t(3 * m + blocks + 2, &[m]),
                attn_splits: 0,
                check: false,
            };
            let runs: Vec<EmbedRun> = held
                .iter()
                .flat_map(|(im, buf)| {
                    let (lo, hi) = (
                        (im.at as i64).max(start),
                        ((im.at + im.n) as i64).min(start + m),
                    );
                    let table = Tensor::new(buf, 0, DType::F32, &[im.n as i64, d as i64]).unwrap();
                    // Position `at` takes the image's row `from`, for `n` rows.
                    let run = |at: i64, from: i64, n: i64| EmbedRun {
                        row: (at - start) as u32,
                        table: table.rows(from, n).unwrap(),
                        ids: vision.ids(n as u32).unwrap(),
                    };
                    let first = lo - im.at as i64;
                    match () {
                        _ if lo >= hi => Vec::new(),
                        // One position late: each row reads the one before it (the first its own).
                        _ if mutation.as_deref() == Some("offset") && first == 0 => {
                            let mut v = vec![run(lo, 0, 1)];
                            if hi - lo > 1 {
                                v.push(run(lo + 1, 0, hi - lo - 1));
                            }
                            v
                        }
                        _ if mutation.as_deref() == Some("offset") => {
                            vec![run(lo, first - 1, hi - lo)]
                        }
                        _ => vec![run(lo, first, hi - lo)],
                    }
                })
                .collect();
            let out = Tensor::new(&logits, 0, DType::F32, &[m, vocab as i64]).unwrap();
            larch_model::Model::forward_embeds(
                &mut text, &s, &batch, None, &pool, &out, None, &runs,
            )
            .unwrap();
            let mut host = vec![0f32; m as usize * vocab];
            s.download(&logits, 0, &mut host).unwrap();
            for row in host.chunks_exact(vocab) {
                let mx = row.iter().fold(f64::MIN, |a, &b| a.max(f64::from(b)));
                let lse = mx
                    + row
                        .iter()
                        .map(|&x| (f64::from(x) - mx).exp())
                        .sum::<f64>()
                        .ln();
                let mut ids: Vec<u32> = (0..vocab as u32).collect();
                ids.select_nth_unstable_by(63, |&a, &b| {
                    row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b))
                });
                ids.truncate(64);
                ids.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
                rows.push(
                    ids.iter()
                        .map(|&i| (i, (f64::from(row[i as usize]) - lse) as f32))
                        .collect(),
                );
            }
        }
        run.push(rows);
    }
    let report = oracle::compare(&dump, &run, margin).unwrap();
    println!(
        "image gate: {} prompts, {tiles} tiles; embeddings min cos {worst_cos:.6} (bound {min_cos}), max abs {worst_abs:.4} (bound {max_abs}); \
         logits mean KL {:.5} p99 {:.5} (bound {bound:.5}), decisive top-1 {:.4}, top-1 {:.4}; encode {:.1} ms/tile",
        dump.prompts.len(),
        report.all.mean_kl,
        report.all.p99_kl,
        report.all.decisive_top1_rate(),
        report.all.top1_rate(),
        encode_ms / f64::from(tiles.max(1)),
    );
    let mut files: Vec<_> = per_file.iter().collect();
    files.sort_by(|a, b| a.1.0.total_cmp(&b.1.0));
    for (f, (cos, abs)) in files.iter().take(4) {
        println!("  {f}: min cos {cos:.6}, max abs {abs:.4}");
    }
    // The text positions (and each image's last row) against their own bound when bounds.json has
    // one: an image's other rows predict nothing that is sampled, and llama.cpp disagrees with
    // itself there, which widens the all-positions bound.
    let text = bounds.get("logits_text").map(|t| {
        let f = |k: &str| {
            t[k].as_f64()
                .unwrap_or_else(|| panic!("bounds.json: logits_text.{k}"))
        };
        (f("bound"), f("margin"))
    });
    let mut text_ok = true;
    if let Some((text_bound, text_margin)) = text {
        let td = oracle::text_only(&dump);
        let trun: Vec<Vec<Top>> = (dump.prompts.iter().zip(&run))
            .map(|(p, r)| {
                oracle::text_positions(p)
                    .into_iter()
                    .map(|i| r[i].clone())
                    .collect()
            })
            .collect();
        let tr = oracle::compare(&td, &trun, text_margin).unwrap();
        println!(
            "text positions: {}; logits mean KL {:.5} p99 {:.5} (bound {text_bound:.5}), decisive top-1 \
             {:.4}, top-1 {:.4}",
            tr.all.positions,
            tr.all.mean_kl,
            tr.all.p99_kl,
            tr.all.decisive_top1_rate(),
            tr.all.top1_rate(),
        );
        text_ok = tr.passes(text_bound);
    }
    let embd_ok = worst_cos >= min_cos && worst_abs <= max_abs;
    let all_ok = report.passes(bound);
    if let Some(m) = mutation {
        println!(
            "mutation {m}: embeddings pass {embd_ok}, text logits pass {text_ok}, all logits pass {all_ok}"
        );
        assert!(
            !(embd_ok && text_ok && all_ok),
            "the gate missed the mutation {m}"
        );
        return;
    }
    assert!(text_ok, "text logits past their bound");
    assert!(embd_ok, "embeddings past their bounds");
    assert!(all_ok, "logits past the oracle's bound");
}

/// An f16's value (the mutation test's pixels).
fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, man) = (
        u32::from(h >> 15) << 31,
        u32::from((h >> 10) & 0x1f),
        u32::from(h & 0x3ff),
    );
    f32::from_bits(match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            return f32::from_bits(sign)
                + (man as f32) * 2f32.powi(-24) * if sign != 0 { -1.0 } else { 1.0 };
        }
        (31, _) => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    })
}

/// The f16 nearest `f` (ties up), for values in f16's normal range.
fn f32_to_f16(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let man = b & 0x7f_ffff;
    sign | (((exp as u32) << 10) + (man >> 13) + ((man >> 12) & 1)) as u16
}

#[test]
fn f16_round_trips_for_the_mutation_test() {
    for v in [0.0f32, 1.0, -2.5, 0.000_061_035_156, 65504.0, 0.333_251_95] {
        assert_eq!(f16_to_f32(f32_to_f16(v)), v, "{v}");
    }
}

#[test]
fn image_rows_have_ids_no_vocabulary_reaches_and_images_differ() {
    let (a, b) = ([1u8; 32], [2u8; 32]);
    let ta: Vec<u32> = image_tokens(&a, 256).collect();
    let tb: Vec<u32> = image_tokens(&b, 256).collect();
    assert!(ta.iter().chain(&tb).all(|&t| t & IMAGE_TOKEN != 0));
    assert_eq!(
        ta,
        image_tokens(&a, 256).collect::<Vec<_>>(),
        "the same image, the same ids"
    );
    assert!(
        ta.iter().zip(&tb).all(|(x, y)| x != y),
        "two images share no row"
    );
    let distinct: std::collections::HashSet<_> = ta.iter().collect();
    assert_eq!(distinct.len(), 256);
}

#[test]
fn image_row_ids_depend_on_the_whole_digest() {
    // Ids seeded from a digest's first 8 bytes gave two images that share them every row's id, so a
    // crafted pair (2^32 work) would share KV. Digests that differ only in their last byte differ.
    let a = [7u8; 32];
    let mut b = a;
    b[31] ^= 1;
    let ta: Vec<u32> = image_tokens(&a, 1792).collect();
    let tb: Vec<u32> = image_tokens(&b, 1792).collect();
    let same = ta.iter().zip(&tb).filter(|(x, y)| x == y).count();
    assert!(same <= 1, "{same} of 1792 rows share an id");
}
