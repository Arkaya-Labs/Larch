//! Qwen 3.5's image input on the CPU reference: a generated Qwen3-VL projector against its graph
//! written out naively in f64 from llama.cpp's qwen3vl.cpp (patches in raster order through both conv
//! frames, the position grid resized bilinearly with aligned corners, 2D vision rope over (y, x),
//! attention over every patch, tanh GELU, the post LayerNorm, the 2 x 2 merge's channel order), smart
//! resize against llama.cpp's own function, and the mmproj trust boundary.

use larch_kernels::{Buffer, CPU, DType, Device, KError, Stream, Tensor};
use larch_model::gguf::{Value, code};
use larch_model::qwen_vision::{
    MAX_IMAGE_TOKENS, QwenPixels, QwenVision, QwenVisionConfig, SMALL_MIN_TOKENS, position_grid,
};
use larch_model::tiny;
use larch_model::vision::{Rgb, Vision};
use larch_records::LoadError;

const E: usize = tiny::QWEN_VIT as usize;
const HEADS: usize = tiny::QWEN_HEADS as usize;
const HD: usize = E / HEADS;
const LAYERS: usize = tiny::QWEN_VIT_LAYERS as usize;
const SIDE: usize = tiny::QWEN_POS_SIDE as usize;
const P: usize = 16;
const DM: usize = 32;

type Tiny = tiny::Projector;

/// `bytes` in a fresh temporary file, which lives as long as the path.
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(bytes: &[u8]) -> TempFile {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("larch-qwenvl-{}-{n}.gguf", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        TempFile(p)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn load(bytes: &[u8], d_model: u32) -> Result<(Device, Stream, QwenVision), LoadError> {
    let f = TempFile::new(bytes);
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    let v = QwenVision::load(&f.0, &dev, &s, d_model)?;
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

/// `w` `[rows, cols]` (rows the output) times x, plus b when given.
fn linear(w: &[f32], b: Option<&[f32]>, x: &[f64]) -> Vec<f64> {
    let cols = x.len();
    (0..w.len() / cols)
        .map(|r| {
            b.map_or(0.0, |b| f64::from(b[r]))
                + (0..cols)
                    .map(|c| f64::from(w[r * cols + c]) * x[c])
                    .sum::<f64>()
        })
        .collect()
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
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

/// The position table resized to `gx` x `gy` in f64, row y · gx + x: bilinear, the corners aligned
/// (output i at source i (SIDE - 1) / (out - 1)).
fn grid_f64(pos: &[f32], gx: usize, gy: usize) -> Vec<Vec<f64>> {
    let at = |i: usize, out: usize| match out {
        1 => 0.0,
        _ => i as f64 * (SIDE - 1) as f64 / (out - 1) as f64,
    };
    let mut rows = Vec::new();
    for y in 0..gy {
        for x in 0..gx {
            let (sx, sy) = (at(x, gx), at(y, gy));
            let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(SIDE - 1), (y0 + 1).min(SIDE - 1));
            let (dx, dy) = (sx - x0 as f64, sy - y0 as f64);
            let v = |x: usize, y: usize, j: usize| f64::from(pos[(y * SIDE + x) * E + j]);
            rows.push(
                (0..E)
                    .map(|j| {
                        v(x0, y0, j) * (1.0 - dx) * (1.0 - dy)
                            + v(x1, y0, j) * dx * (1.0 - dy)
                            + v(x0, y1, j) * (1.0 - dx) * dy
                            + v(x1, y1, j) * dx * dy
                    })
                    .collect(),
            );
        }
    }
    rows
}

/// The projector on an image already at its fitted size, as llama.cpp's graph reads: f64 throughout,
/// with the pixels rounded to f16 as its im2col rounds them.
fn reference(t: &Tiny, img: &Rgb) -> Vec<Vec<f64>> {
    let (w, gx, gy) = (
        img.width as usize,
        img.width as usize / P,
        img.height as usize / P,
    );
    let n = gx * gy;
    let px = |x: usize, y: usize, ch: usize| {
        f64::from(half(
            (f32::from(img.data[(y * w + x) * 3 + ch]) / 255.0 - 0.5) / 0.5,
        ))
    };
    let grid = grid_f64(t.get("v.position_embd.weight"), gx, gy);
    // Patch i in raster order: (i % gx, i / gx). Both time frames of the conv see the same image.
    let mut x: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let (cx, cy) = (i % gx, i / gx);
            let v: Vec<f64> = (0..3)
                .flat_map(|ch| (0..P).flat_map(move |ky| (0..P).map(move |kx| (ch, ky, kx))))
                .map(|(ch, ky, kx)| px(cx * P + kx, cy * P + ky, ch))
                .collect();
            let a = linear(
                t.get("v.patch_embd.weight"),
                Some(t.get("v.patch_embd.bias")),
                &v,
            );
            let b = linear(t.get("v.patch_embd.weight.1"), None, &v);
            (0..E).map(|j| a[j] + b[j] + grid[i][j]).collect()
        })
        .collect();
    // Vision rope: head_dim / 2 pairs (k, k + head_dim / 2); the first quarter of the head turns by y,
    // the second by x, each with its frequency starting over (base 10000 over head_dim / 2 dims).
    let rope = |v: &[f64], i: usize| -> Vec<f64> {
        let (cx, cy) = ((i % gx) as f64, (i / gx) as f64);
        let mut o = v.to_vec();
        for h in 0..HEADS {
            let b = h * HD;
            for k in 0..HD / 2 {
                let (p, j) = if k < HD / 4 {
                    (cy, k)
                } else {
                    (cx, k - HD / 4)
                };
                let theta = p * 10000f64.powf(-2.0 * j as f64 / (HD / 2) as f64);
                let (a, c) = (v[b + k], v[b + k + HD / 2]);
                o[b + k] = a * theta.cos() - c * theta.sin();
                o[b + k + HD / 2] = a * theta.sin() + c * theta.cos();
            }
        }
        o
    };
    for l in 0..LAYERS {
        let w = |s: &str| t.get(&format!("v.blk.{l}.{s}"));
        let qkv: Vec<Vec<f64>> = x
            .iter()
            .map(|r| {
                let h = layernorm(r, w("ln1.weight"), w("ln1.bias"), 1e-6);
                linear(w("attn_qkv.weight"), Some(w("attn_qkv.bias")), &h)
            })
            .collect();
        let q: Vec<_> = (0..n).map(|i| rope(&qkv[i][..E], i)).collect();
        let k: Vec<_> = (0..n).map(|i| rope(&qkv[i][E..2 * E], i)).collect();
        let v: Vec<_> = (0..n).map(|i| qkv[i][2 * E..].to_vec()).collect();
        for i in 0..n {
            let mut a = vec![0.0; E];
            for h in 0..HEADS {
                let r = h * HD..(h + 1) * HD;
                let s: Vec<f64> = (k.iter())
                    .map(|kr| {
                        (q[i][r.clone()].iter().zip(&kr[r.clone()]))
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            / (HD as f64).sqrt()
                    })
                    .collect();
                let mx = s.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = s.iter().map(|v| (v - mx).exp()).collect();
                let z: f64 = e.iter().sum();
                for j in r {
                    a[j] = (0..n).map(|m| e[m] / z * v[m][j]).sum();
                }
            }
            let o = linear(w("attn_out.weight"), Some(w("attn_out.bias")), &a);
            for j in 0..E {
                x[i][j] += o[j];
            }
        }
        for row in x.iter_mut() {
            let h = layernorm(row, w("ln2.weight"), w("ln2.bias"), 1e-6);
            let u: Vec<f64> = linear(w("ffn_up.weight"), Some(w("ffn_up.bias")), &h)
                .into_iter()
                .map(gelu)
                .collect();
            let d = linear(w("ffn_down.weight"), Some(w("ffn_down.bias")), &u);
            for j in 0..E {
                row[j] += d[j];
            }
        }
    }
    // The merge: row (y / 2) (gx / 2) + x / 2, channel c + E (x % 2) + 2 E (y % 2).
    let mut merged = vec![vec![0.0; 4 * E]; n / 4];
    for (i, row) in x.iter().enumerate() {
        let (cx, cy) = (i % gx, i / gx);
        let base = E * (cx % 2) + 2 * E * (cy % 2);
        let h = layernorm(
            row,
            t.get("v.post_ln.weight"),
            t.get("v.post_ln.bias"),
            1e-6,
        );
        merged[(cy / 2) * (gx / 2) + cx / 2][base..base + E].copy_from_slice(&h);
    }
    (merged.iter())
        .map(|m| {
            let y: Vec<f64> = linear(t.get("mm.0.weight"), Some(t.get("mm.0.bias")), m)
                .into_iter()
                .map(gelu)
                .collect();
            linear(t.get("mm.2.weight"), Some(t.get("mm.2.bias")), &y)
        })
        .collect()
}

/// An image with structure in every channel, so a wrong patch or channel order shows.
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

fn encode(v: &QwenVision, s: &Stream, dev: &Device, px: &QwenPixels) -> Vec<f32> {
    let rows = i64::from(px.tokens(v.config()));
    let buf = Buffer::alloc(dev, (rows * DM as i64 * 4) as u64).unwrap();
    let out = Tensor::new(&buf, 0, DType::F32, &[rows, DM as i64]).unwrap();
    v.encode(s, px, &out).unwrap();
    let mut host = vec![0f32; (rows * DM as i64) as usize];
    s.download(&buf, 0, &mut host).unwrap();
    host
}

#[test]
fn encoder_matches_the_graph_written_out() {
    let t = Tiny::qwen(DM as u32, 0x51ab);
    let (dev, s, v) = load(&t.gguf(), DM as u32).unwrap();
    let c = *v.config();
    assert_eq!(
        (c.pos_side, c.min_pixels, c.max_pixels),
        (6, 8 * 1024, MAX_IMAGE_TOKENS * 1024)
    );
    // The grid at the table's size (no resize), stretched down the side, and both ways (up across,
    // down the side); each already a fitted size, so the pixels are the image's.
    for (iw, ih) in [(96, 96), (96, 128), (160, 64)] {
        let img = pattern(iw, ih);
        assert_eq!(c.fit(iw, ih), (iw, ih));
        let px = QwenPixels::new(&img, &c);
        assert_eq!((px.gx, px.gy), (iw / 16, ih / 16));
        let got = encode(&v, &s, &dev, &px);
        let want = reference(&t, &img);
        assert_eq!(got.len(), want.len() * DM);
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
            // f16 activations at every linear input and in the rope against f64: the order of the
            // patches, channels, position rows or rope planes would each move a row far past this.
            assert!(
                cos > 0.9995 && max < 0.05 * nw / (DM as f64).sqrt() + 1e-2,
                "{iw}x{ih} row {r}: cos {cos}, max {max}"
            );
        }
    }
}

/// An encode in slices of layers ([`QwenVision::begin`], `advance`, `finish`) writes the rows the
/// whole encode writes, bit for bit; a finish before the last layer, or into the wrong shape, is
/// refused.
#[test]
fn an_encode_in_slices_is_the_whole_encode() {
    let t = Tiny::qwen(DM as u32, 0x51ab);
    let (dev, s, v) = load(&t.gguf(), DM as u32).unwrap();
    let c = *v.config();
    let px = QwenPixels::new(&pattern(96, 128), &c);
    let whole = encode(&v, &s, &dev, &px);
    let rows = i64::from(px.tokens(&c));
    let buf = Buffer::alloc(&dev, (rows * DM as i64 * 4) as u64).unwrap();
    let out = Tensor::new(&buf, 0, DType::F32, &[rows, DM as i64]).unwrap();
    for slice in [1, 2, 5] {
        let mut job = v.begin(&s, &px).unwrap();
        let mut steps = 0;
        while !v.advance(&s, &mut job, slice).unwrap() {
            steps += 1;
        }
        assert_eq!(job.layers_done(), LAYERS as u32);
        assert_eq!(steps, (LAYERS as u32).div_ceil(slice) - 1, "slice {slice}");
        v.finish(&s, job, &out).unwrap();
        let mut got = vec![0f32; whole.len()];
        s.download(&buf, 0, &mut got).unwrap();
        assert_eq!(got, whole, "slices of {slice}");
    }
    let mut early = v.begin(&s, &px).unwrap();
    v.advance(&s, &mut early, 1).unwrap();
    assert_eq!(v.finish(&s, early, &out), Err(KError::InvalidArg));
    let mut job = v.begin(&s, &px).unwrap();
    v.advance(&s, &mut job, u32::MAX).unwrap();
    let small = Tensor::new(&buf, 0, DType::F32, &[rows - 1, DM as i64]).unwrap();
    assert_eq!(v.finish(&s, job, &small), Err(KError::InvalidArg));
}

#[test]
fn a_bad_image_or_output_is_refused() {
    let t = Tiny::qwen(DM as u32, 7);
    let (dev, s, v) = load(&t.gguf(), DM as u32).unwrap();
    let c = *v.config();
    let px = QwenPixels::new(&pattern(96, 96), &c);
    let buf = Buffer::alloc(&dev, 64 * DM as u64 * 4).unwrap();
    let out = |rows: i64| Tensor::new(&buf, 0, DType::F32, &[rows, DM as i64]).unwrap();
    assert_eq!(v.encode(&s, &px, &out(8)), Err(KError::InvalidArg));
    let odd = QwenPixels {
        gx: 3,
        gy: 12,
        patches: px.patches.clone(),
    };
    assert_eq!(v.encode(&s, &odd, &out(9)), Err(KError::InvalidArg));
    let short = QwenPixels {
        patches: px.patches[1..].to_vec(),
        ..px.clone()
    };
    assert_eq!(v.encode(&s, &short, &out(9)), Err(KError::InvalidArg));
    v.encode(&s, &px, &out(9)).unwrap();
}

/// A Qwen 3.5 projector's geometry (the 0.8B's header) at `cap` rows.
fn qwen35(cap: u32) -> QwenVisionConfig {
    QwenVisionConfig {
        patch: 16,
        merge: 2,
        n_embd: 768,
        n_ff: 3072,
        n_head: 12,
        n_layer: 12,
        eps: 1e-6,
        proj_dim: 1024,
        mean: [0.5; 3],
        std: [0.5; 3],
        pos_side: 48,
        min_pixels: 8 * 1024,
        max_pixels: cap * 1024,
    }
}

#[test]
fn smart_resize_is_llama_cpps() {
    // calc_size_preserved_ratio at llama.cpp's tag, compiled and run on these sizes (align 32, 8 rows
    // at least), at this build's cap and at llama.cpp's default.
    #[rustfmt::skip]
    let cases: [(u32, u32, u32, u32, u32, u32); 28] = [
        (1600, 1280, 720, 1280, 736, 920), (1600, 1920, 1080, 1696, 960, 1590),
        (1600, 1, 1, 96, 96, 9), (1600, 448, 448, 448, 448, 196),
        (1600, 16384, 16384, 1280, 1280, 1600), (1600, 100, 50, 128, 64, 8),
        (1600, 3000, 4000, 1088, 1472, 1564), (1600, 2560, 1440, 1696, 960, 1590),
        (1600, 800, 600, 800, 608, 475), (1600, 16384, 1, 16384, 32, 512),
        (1600, 1, 16384, 32, 16384, 512), (1600, 1170, 2532, 864, 1856, 1566),
        (1600, 33, 17, 128, 96, 12), (1600, 4000, 30, 4000, 32, 125),
        (4096, 1280, 720, 1280, 736, 920), (4096, 1920, 1080, 1920, 1088, 2040),
        (4096, 1, 1, 96, 96, 9), (4096, 448, 448, 448, 448, 196),
        (4096, 16384, 16384, 2048, 2048, 4096), (4096, 100, 50, 128, 64, 8),
        (4096, 3000, 4000, 1760, 2336, 4015), (4096, 2560, 1440, 2560, 1440, 3600),
        (4096, 800, 600, 800, 608, 475), (4096, 16384, 1, 16384, 32, 512),
        (4096, 1, 16384, 32, 16384, 512), (4096, 1170, 2532, 1184, 2528, 2923),
        (4096, 33, 17, 128, 96, 12), (4096, 4000, 30, 4000, 32, 125),
    ];
    for (cap, w, h, fw, fh, tokens) in cases {
        let c = qwen35(cap);
        assert_eq!(c.fit(w, h), (fw, fh), "{w}x{h} at {cap}");
        assert_eq!(c.tokens_for(w, h), tokens, "{w}x{h} at {cap}");
        assert_eq!(c.grid_for(w, h), (fw / 16, fh / 16));
    }
}

/// A 0.8B-class text model (512 to 1,024 wide) reads small text better from a larger image: its
/// default floor is SMALL_MIN_TOKENS rows, reached by at most 2x a side, and a word crop (under 16
/// rows) keeps llama.cpp's fit. A 2,048-wide model, and a header that names its own floor, keep
/// llama.cpp's.
#[test]
fn a_small_text_model_scales_small_images_up() {
    let t = Tiny::qwen(DM as u32, 21);
    let header = |dim: u32, key: Option<(&str, Value)>| {
        let mut kv = t.kv.clone();
        kv.retain(|e| e.0 != "clip.vision.projection_dim");
        kv.push(("clip.vision.projection_dim".into(), Value::U32(dim)));
        if let Some((k, v)) = key {
            kv.push((k.into(), v));
        }
        let b = Tiny {
            tensors: t.tensors.clone(),
            kv,
        }
        .gguf();
        QwenVisionConfig::from_gguf(&larch_model::gguf::Gguf::parse(&b).unwrap())
    };
    let area = |c: &QwenVisionConfig| (c.patch * c.merge).pow(2);
    let small = header(1024, None).unwrap();
    assert_eq!(
        small.min_pixels,
        SMALL_MIN_TOKENS.min(small.max_tokens()) * area(&small)
    );
    let big = header(2048, None).unwrap();
    assert_eq!(big.min_pixels, 8 * area(&big), "llama.cpp's floor");
    let named = header(
        1024,
        Some(("clip.vision.image_min_pixels", Value::U32(8 * area(&small)))),
    );
    assert_eq!(
        named.unwrap().min_pixels,
        8 * area(&small),
        "the file's own floor"
    );
    // The fit, on the 0.8B's geometry with the floor its header gets, and on the 2B's width.
    let c = QwenVisionConfig {
        min_pixels: SMALL_MIN_TOKENS * 1024,
        ..qwen35(MAX_IMAGE_TOKENS)
    };
    let wide = QwenVisionConfig {
        proj_dim: 2048,
        ..qwen35(MAX_IMAGE_TOKENS)
    };
    // A 640 x 360 screenshot (220 rows) at 2x a side, not the 1,032 rows the floor alone gives.
    assert_eq!(
        (c.fit(640, 360), c.tokens_for(640, 360)),
        ((1280, 736), 920)
    );
    assert_eq!(
        wide.fit(640, 360),
        (640, 352),
        "the 2B's fit is llama.cpp's"
    );
    // A 1280 x 720 screenshot (920 rows) up to the floor (past it by f32 rounding, as llama.cpp
    // rounds); a word crop (6 rows) only to 8 rows.
    assert_eq!(
        (c.fit(1280, 720), c.tokens_for(1280, 720)),
        ((1376, 800), 1075)
    );
    assert_eq!((c.fit(100, 50), c.tokens_for(100, 50)), ((128, 64), 8));
    // A large image is only scaled down, as before.
    assert_eq!(c.fit(1920, 1080), wide.fit(1920, 1080));
}

#[test]
fn every_image_size_fits_the_scratch() {
    // The encoder holds max_tokens rows; every size an Image may have lands within it and the minimum.
    let c = qwen35(MAX_IMAGE_TOKENS);
    let sides = [
        1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 100, 333, 720, 1000, 1080, 1440, 1920, 2560, 4000,
        8191, 10000, 16383, 16384,
    ];
    for &w in &sides {
        for &h in &sides {
            let (fw, fh) = c.fit(w, h);
            let t = c.tokens_for(w, h);
            assert!(
                fw % 32 == 0 && fh % 32 == 0 && (8..=c.max_tokens()).contains(&t),
                "{w}x{h} -> {fw}x{fh}, {t} rows"
            );
        }
    }
}

#[test]
fn patches_come_in_merge_order() {
    let t = Tiny::qwen(DM as u32, 3);
    let (_dev, _s, v) = load(&t.gguf(), DM as u32).unwrap();
    let c = *v.config();
    // 128 x 64 (exactly the least area): 8 x 4 patches, each one colour naming its place.
    let mut img = Rgb::filled(128, 64, [0, 0, 0]);
    for y in 0..64usize {
        for x in 0..128usize {
            let i = (y * 128 + x) * 3;
            img.data[i] = (x / 16 * 16 + 8) as u8;
            img.data[i + 1] = (y / 16 * 16 + 8) as u8;
        }
    }
    let px = QwenPixels::new(&img, &c);
    assert_eq!((px.gx, px.gy, px.tokens(&c)), (8, 4, 8));
    let f16 = |v: u8| {
        let x = half((f32::from(v) / 255.0 - 0.5) / 0.5);
        let mut h = 0u16;
        // The f16 bits of x by search: the patch values are in [-1, 1].
        while f16_value(h) != x {
            h += 1;
        }
        h
    };
    let pl = c.patch_len() as usize;
    for r in 0..32usize {
        let (tok, k) = (r / 4, r % 4);
        let (x, y) = ((tok % 4) * 2 + k % 2, (tok / 4) * 2 + k / 2);
        let row = &px.patches[r * pl..(r + 1) * pl];
        assert_eq!(row[0], f16((x * 16 + 8) as u8), "row {r} x");
        assert_eq!(row[256], f16((y * 16 + 8) as u8), "row {r} y");
        assert_eq!(row[512], f16(0), "row {r} blue");
    }
}

/// An f16's value (positive or negative, normal or subnormal).
fn f16_value(h: u16) -> f32 {
    let (e, m) = (i32::from(h >> 10 & 31), f32::from(h & 1023));
    let v = match e {
        0 => m * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (m + 1024.0) * 2f32.powi(e - 25),
    };
    if h & 0x8000 != 0 { -v } else { v }
}

#[test]
fn the_position_grid_resizes_with_aligned_corners() {
    // A table linear in x and y: bilinear reproduces it at the aligned-corner coordinates, which the
    // half-pixel convention would not (its output 0 reads source -0.25 at 3 -> 5).
    let (side, e) = (3, 2);
    let src: Vec<f32> = (0..side * side)
        .flat_map(|i| {
            let (x, y) = ((i % side) as f32, (i / side) as f32);
            [x + 10.0 * y, 100.0 + x - y]
        })
        .collect();
    let check = |gx: usize, gy: usize| {
        let out = position_grid(&src, side, e, gx, gy);
        assert_eq!(out.len(), gx * gy * e);
        for y in 0..gy {
            for x in 0..gx {
                let sx = x as f32 * (side - 1) as f32 / (gx - 1).max(1) as f32;
                let sy = y as f32 * (side - 1) as f32 / (gy - 1).max(1) as f32;
                let at = (y * gx + x) * e;
                assert!(
                    (out[at] - (sx + 10.0 * sy)).abs() < 1e-4,
                    "{gx}x{gy} ({x}, {y})"
                );
                assert!((out[at + 1] - (100.0 + sx - sy)).abs() < 1e-4);
            }
        }
    };
    check(5, 3);
    check(2, 7);
    check(3, 5);
    assert_eq!(
        position_grid(&src, side, e, 3, 3),
        src,
        "its own size is the table"
    );
}

/// A projector `DM` wide whose `key` is `v` (added when absent).
fn with_key(t: &Tiny, key: &str, v: Value) -> Vec<u8> {
    let mut kv = t.kv.clone();
    match kv.iter_mut().find(|e| e.0 == key) {
        Some(e) => e.1 = v,
        None => kv.push((key.into(), v)),
    }
    Tiny {
        tensors: t.tensors.clone(),
        kv,
    }
    .gguf()
}

/// A projector with tensor `name` changed by `f`.
fn with_tensor(
    t: &Tiny,
    name: &str,
    f: impl FnOnce(&mut (String, Vec<u64>, Vec<f32>, bool)),
) -> Vec<u8> {
    let mut tensors = t.tensors.clone();
    f(tensors.iter_mut().find(|e| e.0 == name).unwrap());
    Tiny {
        tensors,
        kv: t.kv.clone(),
    }
    .gguf()
}

#[test]
fn bad_projectors_are_load_errors() {
    let t = Tiny::qwen(DM as u32, 11);
    let err = |bytes: Vec<u8>| load(&bytes, DM as u32).err().expect("a load error");
    let key = LoadError::BadGguf(code::KEY, 0);
    assert_eq!(
        load(&t.gguf(), 64).err(),
        Some(LoadError::BadGguf(code::MISMATCH, 0))
    );
    // Each loader refuses the other's projector, so pairing can try both.
    let internvl = Tiny::new(DM as u32, 11).gguf();
    assert_eq!(err(internvl), LoadError::UnsupportedArch);
    {
        let f = TempFile::new(&t.gguf());
        let dev = Device::open(CPU).unwrap();
        let s = Stream::new(&dev).unwrap();
        assert_eq!(
            Vision::load(&f.0, &dev, &s, DM as u32).err(),
            Some(LoadError::UnsupportedArch)
        );
    }
    let deep = Value::Arr(vec![Value::Bool(false), Value::Bool(true)]);
    assert_eq!(
        err(with_key(&t, "clip.vision.is_deepstack_layers", deep)),
        LoadError::UnsupportedArch
    );
    let not_bools = Value::Arr(vec![Value::U32(0), Value::U32(0)]);
    assert_eq!(
        err(with_key(&t, "clip.vision.is_deepstack_layers", not_bools)),
        key
    );
    for (k, v) in [
        ("clip.vision.spatial_merge_size", Value::U32(3)),
        ("clip.vision.attention.head_count", Value::U32(1)),
        ("clip.vision.image_size", Value::U32(100)),
        ("clip.vision.image_size", Value::U32(16)),
        (
            "clip.vision.image_std",
            Value::Arr(vec![Value::F32(0.0); 3]),
        ),
        ("clip.vision.image_min_pixels", Value::U32(1 << 30)),
        ("clip.vision.image_max_pixels", Value::U32(1000)),
        ("clip.vision.attention.layer_norm_epsilon", Value::F32(0.0)),
        ("clip.vision.patch_size", Value::Str("16".into())),
    ] {
        assert_eq!(err(with_key(&t, k, v.clone())), key, "{k} = {v:?}");
    }
    assert_eq!(
        err(with_key(
            &t,
            "clip.projector_type",
            Value::Str("qwen2vl_merger".into())
        )),
        LoadError::UnsupportedArch
    );
    assert_eq!(
        err(with_key(&t, "clip.use_gelu", Value::Bool(false))),
        LoadError::UnsupportedArch
    );
    let missing = Tiny {
        tensors: (t.tensors.iter())
            .filter(|e| e.0 != "v.patch_embd.weight.1")
            .cloned()
            .collect(),
        kv: t.kv.clone(),
    };
    assert_eq!(
        err(missing.gguf()),
        LoadError::BadGguf(code::MISSING_TENSOR, 0)
    );
    // A transposed QKV, and an f32 matrix where f16 is read.
    let transposed = with_tensor(&t, "v.blk.1.attn_qkv.weight", |e| e.1.swap(0, 1));
    assert!(matches!(
        err(transposed),
        LoadError::BadGguf(code::TENSOR, _)
    ));
    let f32_matrix = with_tensor(&t, "mm.0.weight", |e| e.3 = false);
    assert!(matches!(
        err(f32_matrix),
        LoadError::BadGguf(code::TENSOR, _)
    ));
}

#[test]
fn the_files_pixel_limits_are_read_and_capped() {
    let t = Tiny::qwen(DM as u32, 5);
    let mut kv = t.kv.clone();
    kv.push(("clip.vision.image_max_pixels".into(), Value::U32(64 * 1024)));
    kv.push(("clip.vision.image_min_pixels".into(), Value::U32(56 * 56)));
    let small = Tiny {
        tensors: t.tensors.clone(),
        kv,
    };
    let (_d, _s, v) = load(&small.gguf(), DM as u32).unwrap();
    let c = v.config();
    assert_eq!(
        (c.min_pixels, c.max_pixels, c.max_tokens()),
        (3136, 65536, 64)
    );
    assert_eq!(c.fit(1920, 1080), (320, 192));
    let big = with_key(&t, "clip.vision.image_max_pixels", Value::U32(16384 * 1024));
    let (_d, _s, v) = load(&big, DM as u32).unwrap();
    assert_eq!(v.config().max_tokens(), MAX_IMAGE_TOKENS);
}

/// The top 64 of a row of logits as log-probabilities, by logit descending (ties by id), as
/// llama-logits writes them.
fn top64(row: &[f32]) -> Vec<(u32, f32)> {
    let mx = row.iter().fold(f64::MIN, |a, &b| a.max(f64::from(b)));
    let lse = mx
        + row
            .iter()
            .map(|&x| (f64::from(x) - mx).exp())
            .sum::<f64>()
            .ln();
    let mut ids: Vec<u32> = (0..row.len() as u32).collect();
    let by = |a: &u32, b: &u32| row[*b as usize].total_cmp(&row[*a as usize]).then(a.cmp(b));
    ids.select_nth_unstable_by(63, by);
    ids.truncate(64);
    ids.sort_by(by);
    ids.iter()
        .map(|&i| (i, (f64::from(row[i as usize]) - lse) as f32))
        .collect()
}

/// Each row's M-RoPE positions (t, h, w, e) as llama.cpp's mtmd helpers lay them out: a text row one
/// position on from the last, an image's row i at (p0, p0 + i / nx, p0 + i % nx, 0) from its first
/// position p0, and the rows after the image going on from p0 + n_pos (max(nx, ny)).
fn mrope_positions(p: &larch_model::oracle::Prompt, mutation: Option<&str>) -> Vec<[i32; 4]> {
    let mut pos = Vec::with_capacity(p.tokens.len());
    let (mut next, mut r) = (0i32, 0usize);
    let mut images = p.images.iter().peekable();
    while r < p.tokens.len() {
        let Some(im) = images.next_if(|im| im.at == r) else {
            pos.push([next, next, next, 0]);
            next += 1;
            r += 1;
            continue;
        };
        let (nx, _) = im.grid.expect("an M-RoPE dump: each image has a grid");
        for i in 0..im.n as i32 {
            let (y, x) = (i / nx as i32, i % nx as i32);
            pos.push(match mutation {
                // InternVL's layout: the image's rows one position each.
                Some("positions-sequential") => [next + i; 4],
                Some("grid-transpose") => [next, next + x, next + y, 0],
                _ => [next, next + y, next + x, 0],
            });
        }
        next += match mutation {
            Some("positions-sequential") => im.n as i32,
            _ => im.n_pos.expect("an M-RoPE dump: each image has n_pos") as i32,
        };
        r += im.n;
    }
    pos
}

/// The image gate for Qwen 3.5 against a `llama-logits --mmproj` dump (M-RoPE positions, each image's
/// grid): larch's projector rows against llama.cpp's, and every row's logits with the image rows fed
/// at their (t, h, w) positions. LARCH_VISION_ORACLE holds reference.jsonl, the images and
/// bounds.json; LARCH_VISION_DEVICE picks the device (0 by default; -2 the CPU), and
/// LARCH_VISION_RUN_OUT names a file for each prompt's top-1 ids. With
/// LARCH_VISION_MUTATION one step breaks on purpose and the gate must fail: `positions-sequential`
/// (InternVL's positions), `grid-transpose` (x and y swapped), `normalization` (ImageNet's mean and
/// std for 0.5), `offset` (the image one row late).
#[test]
#[ignore = "needs a Qwen 3.5 model, its projector and a llama-logits dump: LARCH_VISION_*"]
fn qwen_image_oracle() {
    use larch_model::oracle::{self, Dump, IMAGE_ROW, Top};
    use larch_model::vision::Image;
    use larch_model::{EmbedRun, Model, StepBatch, Transformer};
    use std::path::PathBuf;

    let var =
        |k: &str| PathBuf::from(std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set")));
    let (dir, model, mmproj) = (
        var("LARCH_VISION_ORACLE"),
        var("LARCH_VISION_MODEL"),
        var("LARCH_VISION_MMPROJ"),
    );
    let device: i32 = std::env::var("LARCH_VISION_DEVICE").map_or(0, |v| v.parse().unwrap());
    let bounds: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("bounds.json")).unwrap()).unwrap();
    let f = |a: &str, b: &str| {
        bounds[a][b]
            .as_f64()
            .unwrap_or_else(|| panic!("bounds.json: {a}.{b}"))
    };
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
    let dev = Device::open(device).unwrap();
    let s = Stream::new(&dev).unwrap();
    let mut text = Transformer::load(&model, &dev, &s, CHUNK as u32).unwrap();
    if text.config().arch.hybrid() {
        text.alloc_state(&s, 1).unwrap();
    }
    let c = larch_model::Model::config(&text).clone();
    assert!(c.rope.sections.is_some(), "a model with M-RoPE sections");
    let vision = QwenVision::load(&mmproj, &dev, &s, c.d_model).unwrap();
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
    let input = Buffer::alloc(&dev, ((CHUNK * 8 + blocks + 8) * 4) as u64).unwrap();
    let logits = Buffer::alloc(&dev, (CHUNK as usize * vocab * 4) as u64).unwrap();
    let (mut worst_cos, mut worst_abs, mut encode_ms, mut rows_encoded) =
        (1.0f64, 0.0f64, 0.0f64, 0usize);
    let mut per_file: std::collections::BTreeMap<String, (f64, f64)> = Default::default();
    let mut run: Vec<Vec<Top>> = Vec::new();
    for (pi, p) in dump.prompts.iter().enumerate() {
        let mut held = Vec::new();
        for im in &p.images {
            let file = im.file.strip_suffix(".oracle.png").unwrap_or(&im.file);
            let bytes = std::fs::read(dir.join(file)).unwrap();
            let rgb = Image::new(bytes.into()).unwrap().decode().unwrap();
            let mut cfg = *vision.config();
            if mutation.as_deref() == Some("normalization") {
                (cfg.mean, cfg.std) = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
            }
            let px = QwenPixels::new(&rgb, &cfg);
            assert_eq!(
                (px.tokens(&cfg) as usize, Some((px.gx / 2, px.gy / 2))),
                (im.n, im.grid),
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
            encode_ms += t0.elapsed_ns(&t1).unwrap() as f64 / 1e6;
            rows_encoded += im.n;
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
        if text.config().arch.hybrid() {
            text.reset_state(&s, 0).unwrap();
        }
        let pos = mrope_positions(p, mutation.as_deref());
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
            packed.extend(pos[start as usize..(start + m) as usize].iter().flatten()); // [m, 4]
            packed.extend((start..start + m).map(|x| x as i32)); // KV slots: row r in slot r
            packed.extend(0..blocks as i32); // the block table
            packed.push(start as i32); // the chunk's first KV index
            packed.push(m as i32);
            packed.extend(0..m as i32); // logits rows
            packed.push(0); // the state slot
            packed.push(m as i32); // its rows
            s.upload(&input, 0, &packed).unwrap();
            let at = |i: i64| (i * 4) as u64;
            let t =
                |off: i64, shape: &[i64]| Tensor::new(&input, at(off), DType::I32, shape).unwrap();
            let batch = StepBatch {
                token_ids: t(0, &[m]),
                positions: t(m, &[m, 4]),
                slot_ids: t(5 * m, &[m]),
                decode_rows: 0,
                prefill_rows: m as u32,
                block_table: t(6 * m, &[1, blocks]),
                seq_lens: t(0, &[0]),
                prefill_q_start: t(6 * m + blocks, &[1]),
                prefill_q_len: t(6 * m + blocks + 1, &[1]),
                logits_rows: t(6 * m + blocks + 2, &[m]),
                attn_splits: 0,
                check: false,
            };
            let state = larch_model::StateBatch {
                slots: t(7 * m + blocks + 2, &[1]),
                q_lens: t(7 * m + blocks + 3, &[1]),
            };
            let runs: Vec<EmbedRun> = held
                .iter()
                .flat_map(|(im, buf)| {
                    let (lo, hi) = (
                        (im.at as i64).max(start),
                        ((im.at + im.n) as i64).min(start + m),
                    );
                    let table = Tensor::new(buf, 0, DType::F32, &[im.n as i64, d as i64]).unwrap();
                    let run = |at: i64, from: i64, n: i64| EmbedRun {
                        row: (at - start) as u32,
                        table: table.rows(from, n).unwrap(),
                        ids: vision.ids(n as u32).unwrap(),
                    };
                    let first = lo - im.at as i64;
                    match () {
                        _ if lo >= hi => Vec::new(),
                        // One row late: each row reads the one before it (the first its own).
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
            let state = text.config().arch.hybrid().then_some(&state);
            larch_model::Model::forward_embeds(
                &mut text, &s, &batch, state, &pool, &out, None, &runs,
            )
            .unwrap();
            let mut host = vec![0f32; m as usize * vocab];
            s.download(&logits, 0, &mut host).unwrap();
            rows.extend(host.chunks_exact(vocab).map(top64));
        }
        run.push(rows);
    }
    // LARCH_VISION_RUN_OUT: each prompt's top-1 ids, one JSON line a prompt, to find where a run
    // parts from the reference.
    if let Ok(path) = std::env::var("LARCH_VISION_RUN_OUT") {
        let lines: Vec<String> = (run.iter().enumerate())
            .map(|(i, rows)| {
                let ids: Vec<u32> = rows.iter().map(|r| r[0].0).collect();
                serde_json::json!({"index": i, "top1": ids}).to_string()
            })
            .collect();
        std::fs::write(path, lines.join("\n") + "\n").unwrap();
    }
    let report = oracle::compare(&dump, &run, margin).unwrap();
    println!(
        "qwen image gate: {} prompts, {rows_encoded} image rows; embeddings min cos {worst_cos:.6} (bound {min_cos}), max abs {worst_abs:.4} (bound {max_abs}); \
         logits mean KL {:.5} p99 {:.5} (bound {bound:.5}), decisive top-1 {:.4}, top-1 {:.4}; encode {:.2} ms/row",
        dump.prompts.len(),
        report.all.mean_kl,
        report.all.p99_kl,
        report.all.decisive_top1_rate(),
        report.all.top1_rate(),
        encode_ms / rows_encoded.max(1) as f64,
    );
    let mut files: Vec<_> = per_file.iter().collect();
    files.sort_by(|a, b| a.1.0.total_cmp(&b.1.0));
    for (f, (cos, abs)) in files.iter().take(4) {
        println!("  {f}: min cos {cos:.6}, max abs {abs:.4}");
    }
    let text_bounds = bounds.get("logits_text").map(|t| {
        let f = |k: &str| {
            t[k].as_f64()
                .unwrap_or_else(|| panic!("bounds.json: logits_text.{k}"))
        };
        (f("bound"), f("margin"))
    });
    let mut text_ok = true;
    if let Some((text_bound, text_margin)) = text_bounds {
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

/// The projector layer the engine drives: the header picks the type, a Qwen image is one unit of all
/// its rows and an InternVL image a unit per tile, units encode as the projector itself does, and
/// pixels fit only their own projector.
#[test]
fn either_projector_by_its_header() {
    use larch_model::projector::{ImagePixels, Projector, ProjectorConfig};
    let (q, i) = (Tiny::qwen(DM as u32, 21), Tiny::new(DM as u32, 21));
    let config = |b: &[u8]| ProjectorConfig::from_gguf(&larch_model::gguf::Gguf::parse(b).unwrap());
    let (qc, ic) = (config(&q.gguf()).unwrap(), config(&i.gguf()).unwrap());
    assert!(matches!(qc, ProjectorConfig::Qwen(_)) && matches!(ic, ProjectorConfig::InternVl(_)));
    let other = with_key(&q, "clip.projector_type", Value::Str("gemma3".into()));
    assert_eq!(config(&other).err(), Some(LoadError::UnsupportedArch));
    let bad_qwen = with_key(&q, "clip.vision.spatial_merge_size", Value::U32(3));
    assert_eq!(
        config(&bad_qwen).err(),
        Some(LoadError::BadGguf(code::KEY, 0))
    );
    // A 160 x 64 image: 10 x 4 patches, 10 rows on a 5 x 2 grid, one unit; InternVL's tiles of 4.
    assert_eq!(
        (
            qc.tokens_for(160, 64),
            qc.unit_rows(160, 64),
            qc.grid_for(160, 64)
        ),
        (10, 10, Some((5, 2)))
    );
    assert_eq!((ic.unit_rows(160, 64), ic.grid_for(160, 64)), (4, None));
    assert_eq!(qc.markers(), ("<|vision_start|>", "<|vision_end|>"));
    assert_eq!(ic.markers(), ("<img>", "</img>"));
    assert_eq!(qc.max_unit_rows(), MAX_IMAGE_TOKENS);

    let f = TempFile::new(&q.gguf());
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    let p = Projector::load(&f.0, &dev, &s, DM as u32).unwrap();
    assert_eq!(p.config(), qc);
    let img = pattern(160, 64);
    let px = qc.cut(&img);
    assert_eq!(px.units(), 1);
    assert!(px.fits(&qc, 10) && !px.fits(&qc, 9) && !px.fits(&ic, 10));
    let buf = Buffer::alloc(&dev, 10 * DM as u64 * 4).unwrap();
    let out = Tensor::new(&buf, 0, DType::F32, &[10, DM as i64]).unwrap();
    p.encode_unit(&s, &px, 0, &out).unwrap();
    let mut got = vec![0f32; 10 * DM];
    s.download(&buf, 0, &mut got).unwrap();
    let ImagePixels::Qwen(direct) = &px else {
        unreachable!()
    };
    let (_d, s2, v) = load(&q.gguf(), DM as u32).unwrap();
    assert_eq!(
        got,
        encode(&v, &s2, &_d, direct),
        "a unit is the whole image"
    );
    assert_eq!(p.encode_unit(&s, &px, 1, &out), Err(KError::InvalidArg));
    let internvl = ic.cut(&Rgb::filled(56, 56, [9, 9, 9]));
    assert_eq!(
        p.encode_unit(&s, &internvl, 0, &out),
        Err(KError::InvalidArg)
    );
    // The warm-up's blank input is one row through either encoder.
    let blank = qc.blank();
    assert!(blank.fits(&qc, 1));
    let one = Tensor::new(&buf, 0, DType::F32, &[1, DM as i64]).unwrap();
    p.encode_unit(&s, &blank, 0, &one).unwrap();
    assert!(ic.blank().fits(&ic, 4));
    // The engine's warm-up input: on Qwen the largest square of up to 16 x 16 rows within the cap
    // (a small screenshot's size, so the first image does not pay for the kernels' first launch
    // at real sizes); on InternVL the blank tile.
    let warm = qc.warm_up_input();
    let rows = warm.unit_rows(&qc);
    let want = (1..=16u32)
        .rev()
        .map(|side| side * side)
        .find(|&n| n <= qc.max_unit_rows())
        .unwrap();
    assert_eq!(
        (rows, want),
        (256, 256),
        "the test projector's cap is 1600 rows"
    );
    assert!(warm.fits(&qc, rows) && warm.units() == 1);
    let wbuf = Buffer::alloc(&dev, u64::from(rows) * DM as u64 * 4).unwrap();
    let wout = Tensor::new(&wbuf, 0, DType::F32, &[i64::from(rows), DM as i64]).unwrap();
    p.encode_unit(&s, &warm, 0, &wout).unwrap();
    assert_eq!(ic.warm_up_input(), ic.blank());
    assert_eq!(p.ids(10).unwrap().shape(), [10]);
}
