//! Qwen 3.5 images through the engine on the CPU reference: the tiny hybrid model with the tiny
//! Qwen3-VL projector. The engine lays out M-RoPE positions from each row's KV index and the image's
//! grid; a forward pass written out here with llama.cpp's layout (mtmd's helpers: the image's rows at
//! (p0, p0 + y, p0 + x), the text after it from p0 + max(nx, ny)) gives the engine's logprobs, and
//! the image's rows one position each (InternVL's layout) does not.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use larch_engine::{EngineEvent, EngineOptions, ImageSpan, Request, start};
use larch_kernels::{Buffer, CPU, DType, Device, KvPool, Stream, Tensor};
use larch_model::projector::{ImagePixels, ProjectorConfig};
use larch_model::qwen_vision::{QwenPixels, QwenVision};
use larch_model::vision::{IMAGE_TOKEN, Image, image_tokens};
use larch_model::{Arch, EmbedRun, Model, StateBatch, StepBatch, Transformer, tiny};
use larch_records::knob_id;

fn files() -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir();
    // One pair per test: tests run in parallel and would rewrite each other's files.
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let id = format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let model = dir.join(format!("larch-mrope-tiny-{id}.gguf"));
    let mmproj = dir.join(format!("larch-mrope-mmproj-{id}.gguf"));
    std::fs::write(&model, tiny::gguf(Arch::Qwen35, 5)).unwrap();
    std::fs::write(&mmproj, tiny::Projector::qwen(tiny::D as u32, 5).gguf()).unwrap();
    (model, mmproj)
}

fn palette() -> Arc<Image> {
    let png = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../larch-model/tests/data/vision/palette.png"
    ))
    .unwrap();
    Arc::new(Image::new(png.into()).unwrap())
}

/// A `w` x `h` RGB PNG of a pattern (so the encoder's rows differ), deflate's stored blocks: no
/// encoder crate here.
fn png(w: u32, h: u32) -> Vec<u8> {
    let crc = |bytes: &[u8]| {
        let mut c = !0u32;
        for &b in bytes {
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
    };
    let mut raw = Vec::new();
    for y in 0..h {
        raw.push(0); // filter: none
        for x in 0..w {
            raw.extend([(x * 7 + y * 3) as u8, (x * y) as u8, (x ^ y) as u8]);
        }
    }
    let mut z = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65535).collect();
    for (k, b) in blocks.iter().enumerate() {
        z.push(u8::from(k + 1 == blocks.len()));
        let n = b.len() as u16;
        z.extend(n.to_le_bytes());
        z.extend((!n).to_le_bytes());
        z.extend_from_slice(b);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    z.extend(((b << 16) | a).to_be_bytes());
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |kind: &[u8], data: &[u8]| {
        out.extend((data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend(crc(&body).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend(w.to_be_bytes());
    ihdr.extend(h.to_be_bytes());
    ihdr.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    out
}

fn text(n: usize, seed: u32) -> Vec<u32> {
    (0..n).map(|i| (i as u32 * 7 + seed) % 200 + 1).collect()
}

/// Logprobs of every row of `tokens` (image rows at `at..at + n` fed the projector's rows of
/// `px`) through the model alone, each row at `positions(row)`.
fn reference(
    model: &Path,
    mmproj: &Path,
    tokens: &[u32],
    px: &QwenPixels,
    at: usize,
    positions: impl Fn(usize) -> [i32; 4],
) -> Vec<Vec<f64>> {
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    let rows = tokens.len() as i64;
    let mut m = Transformer::load(model, &dev, &s, rows as u32).unwrap();
    m.alloc_state(&s, 1).unwrap();
    let c = m.config().clone();
    let d = c.d_model as i64;
    let v = QwenVision::load(mmproj, &dev, &s, c.d_model).unwrap();
    let n = i64::from(px.tokens(v.config()));
    let embd = Buffer::alloc(&dev, (n * d * 4) as u64).unwrap();
    v.encode(&s, px, &Tensor::new(&embd, 0, DType::F32, &[n, d]).unwrap())
        .unwrap();
    let blocks = (rows + 15) / 16;
    let kv = (i64::from(c.kv_layers)
        * blocks
        * 16
        * i64::from(c.n_kv_heads)
        * i64::from(c.head_dim)
        * 2) as u64;
    let (k, vb) = (
        Buffer::alloc(&dev, kv).unwrap(),
        Buffer::alloc(&dev, kv).unwrap(),
    );
    let pool = KvPool {
        k: &k,
        v: &vb,
        n_blocks: blocks as u32,
        block_tokens: 16,
        n_layers: c.kv_layers,
        n_kv_heads: c.n_kv_heads,
        head_dim: c.head_dim,
        dtype: DType::F16,
    };
    let mut packed: Vec<i32> = (tokens.iter())
        .map(|&t| if t & IMAGE_TOKEN != 0 { 0 } else { t as i32 })
        .collect();
    packed.extend((0..rows as usize).flat_map(&positions));
    packed.extend(0..rows as i32); // KV slots
    packed.extend(0..blocks as i32);
    packed.extend([0, rows as i32]); // q start, q len
    packed.extend(0..rows as i32); // logits rows
    packed.extend([0, rows as i32]); // state slot, its rows
    let input = Buffer::alloc(&dev, packed.len() as u64 * 4).unwrap();
    s.upload(&input, 0, &packed).unwrap();
    let t =
        |off: i64, shape: &[i64]| Tensor::new(&input, (off * 4) as u64, DType::I32, shape).unwrap();
    let r = rows;
    let batch = StepBatch {
        token_ids: t(0, &[r]),
        positions: t(r, &[r, 4]),
        slot_ids: t(5 * r, &[r]),
        decode_rows: 0,
        prefill_rows: r as u32,
        block_table: t(6 * r, &[1, blocks]),
        seq_lens: t(0, &[0]),
        prefill_q_start: t(6 * r + blocks, &[1]),
        prefill_q_len: t(6 * r + blocks + 1, &[1]),
        logits_rows: t(6 * r + blocks + 2, &[r]),
        attn_splits: 0,
        check: false,
    };
    let state = StateBatch {
        slots: t(7 * r + blocks + 2, &[1]),
        q_lens: t(7 * r + blocks + 3, &[1]),
    };
    let vocab = i64::from(c.vocab);
    let logits = Buffer::alloc(&dev, (r * vocab * 4) as u64).unwrap();
    let out = Tensor::new(&logits, 0, DType::F32, &[r, vocab]).unwrap();
    let ids = v.ids(n as u32).unwrap();
    let run = EmbedRun {
        row: at as u32,
        table: Tensor::new(&embd, 0, DType::F32, &[n, d]).unwrap(),
        ids,
    };
    m.forward_embeds(&s, &batch, Some(&state), &pool, &out, None, &[run])
        .unwrap();
    let mut host = vec![0f32; (r * vocab) as usize];
    s.download(&logits, 0, &mut host).unwrap();
    host.chunks_exact(vocab as usize)
        .map(|l| {
            let mx = l.iter().fold(f64::MIN, |a, &b| a.max(f64::from(b)));
            let lse = mx
                + l.iter()
                    .map(|&x| (f64::from(x) - mx).exp())
                    .sum::<f64>()
                    .ln();
            l.iter().map(|&x| f64::from(x) - lse).collect()
        })
        .collect()
}

#[test]
fn image_rows_take_their_grid_positions() {
    grid_positions(10, 6, None, None);
}

/// An image-only prompt one row past a 64-row prefill chunk: a 13 x 5 grid of rows, 65. Its last
/// image row runs as a decode row, and keeps its grid position as the rows prefilled before it do.
#[test]
fn a_last_image_row_decoded_keeps_its_grid_position() {
    grid_positions(0, 0, Some(64), Some((13, 5)));
}

/// The engine's logprobs for `before` text rows, an image's rows and `after` text rows (with
/// `chunk` the prefill chunk), against a forward pass with llama.cpp's layout and one with
/// sequential positions. The image is the palette's 8 rows on a 4 x 2 grid, or a PNG of a `grid`
/// of rows.
fn grid_positions(before: usize, after: usize, chunk: Option<u32>, grid: Option<(u32, u32)>) {
    let (model, mmproj) = files();
    let mut o = EngineOptions::new(model.clone(), CPU);
    o.mmproj = Some(mmproj.clone());
    o.max_context = 512;
    o.kv_pool_fraction = 0.3;
    if let Some(c) = chunk {
        o.knob_overrides = vec![(knob_id::PREFILL_CHUNK_TOKENS, f64::from(c))];
    }
    let e = start(o).unwrap();
    let v = e.vision().expect("a projector");
    let ProjectorConfig::Qwen(q) = v else {
        panic!("not Qwen's projector")
    };
    // The palette (4 x 2 pixels resize to 128 x 64: 8 rows on a 4 x 2 grid, which take 4
    // positions), or a PNG of `grid` rows of 32 x 32 pixels.
    let (image, nx, ny) = match grid {
        None => (palette(), 4, 2),
        Some((nx, ny)) => (
            Arc::new(Image::new(png(nx * 32, ny * 32).into()).unwrap()),
            nx,
            ny,
        ),
    };
    let (w, h) = (image.width(), image.height());
    assert_eq!(
        (v.tokens_for(w, h), v.grid_for(w, h)),
        (nx * ny, Some((nx, ny)))
    );
    let px = QwenPixels::new(&image.decode().unwrap(), &q);
    let rows = (nx * ny) as usize;
    assert_eq!(px.tokens(&q) as usize, rows);
    let (before, after) = (text(before, 1), text(after, 2));
    let at = before.len();
    let mut prompt = before.clone();
    prompt.extend(image_tokens(image.digest(), rows as u32));
    prompt.extend(&after);
    let mut r = Request::new(prompt.clone(), 6);
    r.logprobs = 4;
    r.images = vec![ImageSpan {
        at: at as u32,
        rows: rows as u32,
        tile_rows: v.unit_rows(w, h),
        image: image.clone(),
        pixels: Some(Arc::new(ImagePixels::Qwen(px.clone()))),
    }];
    let h = e.submit(r).unwrap();
    let mut got: Vec<(u32, Vec<(u32, f32)>)> = Vec::new();
    for ev in h.events.iter() {
        match ev {
            EngineEvent::Token { id, top, .. } => got.push((id, top)),
            EngineEvent::Done { .. } => break,
            EngineEvent::Prefilled { .. } => {}
            end => panic!("{end:?}"),
        }
    }
    e.shutdown();
    assert_eq!(got.len(), 6);
    // The prompt and the tokens the engine fed back; row L - 1 + j predicts token j.
    let mut fed = prompt.clone();
    fed.extend(got[..got.len() - 1].iter().map(|g| g.0));
    let llama = |row: usize| -> [i32; 4] {
        match row {
            r if r < at => [r as i32, r as i32, r as i32, 0],
            r if r < at + rows => {
                let (i, p0, nx) = ((r - at) as i32, at as i32, nx as i32);
                [p0, p0 + i / nx, p0 + i % nx, 0]
            }
            // The image took max(nx, ny) positions for its rows (the palette's 8: 4).
            r => {
                let q = (r - rows) as i32 + nx.max(ny) as i32;
                [q, q, q, 0]
            }
        }
    };
    let sequential = |row: usize| [row as i32, row as i32, row as i32, 0];
    let worst = |positions: &dyn Fn(usize) -> [i32; 4]| {
        let lp = reference(&model, &mmproj, &fed, &px, at, positions);
        let mut worst = 0f64;
        for (j, (_, top)) in got.iter().enumerate() {
            for &(id, l) in top {
                worst = worst.max((lp[prompt.len() - 1 + j][id as usize] - f64::from(l)).abs());
            }
        }
        worst
    };
    let (right, wrong) = (worst(&llama), worst(&sequential));
    println!(
        "engine against llama.cpp's layout: {right:.2e}; against sequential positions: {wrong:.2e}"
    );
    // The engine runs the prompt in chunks and decodes one row a step (mmvq), the reference one
    // prefill (dequant + gemm): rounding apart, far below what a wrong layout moves.
    assert!(
        right < 5e-3 && wrong > 10.0 * right,
        "right {right}, wrong {wrong}"
    );
    let _ = std::fs::remove_file(&model);
    let _ = std::fs::remove_file(&mmproj);
}

/// Beside another chat, a Qwen image encodes a slice of its layers a step (the other chat keeps
/// decoding), and the image chat's first prediction is the one it makes alone.
#[test]
fn an_image_beside_another_chat_encodes_in_slices() {
    let (model, mmproj) = files();
    let opts = || {
        let mut o = EngineOptions::new(model.clone(), CPU);
        o.mmproj = Some(mmproj.clone());
        o.max_context = 512;
        o.kv_pool_fraction = 0.3;
        o
    };
    let image = palette();
    let image_request = |e: &larch_engine::EngineHandle| {
        let v = e.vision().expect("a projector");
        let (w, h) = (image.width(), image.height());
        let mut prompt = text(10, 1);
        let at = prompt.len() as u32;
        prompt.extend(image_tokens(image.digest(), v.tokens_for(w, h)));
        prompt.extend(text(6, 2));
        let mut r = Request::new(prompt, 3);
        r.logprobs = 4;
        r.images = vec![ImageSpan {
            at,
            rows: v.tokens_for(w, h),
            tile_rows: v.unit_rows(w, h),
            image: image.clone(),
            pixels: Some(Arc::new(v.cut(&image.decode().unwrap()))),
        }];
        r
    };
    let first = |h: larch_engine::RequestHandle| -> Vec<(u32, f32)> {
        match h
            .events
            .iter()
            .find(|e| !matches!(e, EngineEvent::Prefilled { .. }))
            .expect("an event")
        {
            EngineEvent::Token { top, .. } => top,
            end => panic!("{end:?}"),
        }
    };
    // Alone: the image encodes whole, in the step that reads it.
    let e = start(opts()).unwrap();
    let alone = first(e.submit(image_request(&e)).unwrap());
    assert_eq!(e.encode_slices(), 0, "alone, an image encodes in one step");
    e.shutdown();
    // Beside a chat that is decoding.
    let e = start(opts()).unwrap();
    let other = e.submit(Request::new(text(12, 3), 400)).unwrap();
    match other
        .events
        .iter()
        .find(|e| !matches!(e, EngineEvent::Prefilled { .. }))
        .unwrap()
    {
        EngineEvent::Token { .. } => {}
        end => panic!("{end:?}"),
    }
    let beside = first(e.submit(image_request(&e)).unwrap());
    let slices = e.encode_slices();
    e.shutdown();
    drop(other);
    // The tiny projector's 2 layers go a slice each.
    assert!(slices >= 2, "{slices} slices");
    // Apart only by the rows each step multiplied together (mmvq beside decode rows, a gemm alone).
    let worst = alone
        .iter()
        .map(|&(id, lp)| {
            let b = beside
                .iter()
                .find(|x| x.0 == id)
                .map_or(f32::NEG_INFINITY, |x| x.1);
            (lp - b).abs()
        })
        .fold(0f32, f32::max);
    assert!(
        worst < 5e-3,
        "first prediction apart by {worst}: {alone:?} vs {beside:?}"
    );
}

/// A pasted image encodes beside a chat that is decoding, a slice a step, before its message is
/// sent: it no longer waits for the engine to go idle.
#[test]
fn a_pasted_image_encodes_beside_a_decoding_chat() {
    let (model, mmproj) = files();
    let mut o = EngineOptions::new(model, CPU);
    o.mmproj = Some(mmproj);
    o.max_context = 512;
    o.kv_pool_fraction = 0.3;
    let e = start(o).unwrap();
    let other = e.submit(Request::new(text(12, 3), 400)).unwrap();
    match other
        .events
        .iter()
        .find(|e| !matches!(e, EngineEvent::Prefilled { .. }))
        .unwrap()
    {
        EngineEvent::Token { .. } => {}
        end => panic!("{end:?}"),
    }
    let before = e.tiles_encoded();
    let image = palette();
    let v = e.vision().expect("a projector");
    e.prepare_image(&image, Arc::new(v.cut(&image.decode().unwrap())))
        .unwrap();
    // The warm-up encodes only when idle; the pasted image goes ahead of it beside the chat.
    let mut decoded = 0;
    while e.tiles_encoded() == before {
        match other.events.recv().unwrap() {
            EngineEvent::Token { .. } => decoded += 1,
            end => panic!("the chat ended first: {end:?}"),
        }
        assert!(decoded < 300, "no encode beside the chat");
    }
    assert!(e.encode_slices() >= 2, "{} slices", e.encode_slices());
    e.shutdown();
}

/// A request with another image arriving while a pasted image's encode is in flight: the paste
/// goes on in slices first (not settled whole in one step), then the request's image; each encodes
/// once and the request answers.
#[test]
fn a_request_image_waits_for_the_paste_in_flight() {
    let (model, mmproj) = files();
    let mut o = EngineOptions::new(model, CPU);
    o.mmproj = Some(mmproj);
    o.max_context = 512;
    o.kv_pool_fraction = 0.3;
    let e = start(o).unwrap();
    let other = e.submit(Request::new(text(12, 3), 400)).unwrap();
    match other
        .events
        .iter()
        .find(|e| !matches!(e, EngineEvent::Prefilled { .. }))
        .unwrap()
    {
        EngineEvent::Token { .. } => {}
        end => panic!("{end:?}"),
    }
    let before = e.tiles_encoded();
    let v = e.vision().expect("a projector");
    let pasted = palette();
    e.prepare_image(&pasted, Arc::new(v.cut(&pasted.decode().unwrap())))
        .unwrap();
    let png = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../larch-model/tests/data/vision/rgba.png"
    ))
    .unwrap();
    let image = Arc::new(Image::new(png.into()).unwrap());
    let (w, h) = (image.width(), image.height());
    let mut prompt = text(4, 1);
    let at = prompt.len() as u32;
    prompt.extend(image_tokens(image.digest(), v.tokens_for(w, h)));
    prompt.extend(text(3, 2));
    let mut r = Request::new(prompt, 2);
    r.images = vec![ImageSpan {
        at,
        rows: v.tokens_for(w, h),
        tile_rows: v.unit_rows(w, h),
        image: image.clone(),
        pixels: Some(Arc::new(v.cut(&image.decode().unwrap()))),
    }];
    let asked = e.submit(r).unwrap();
    loop {
        match asked.events.recv().unwrap() {
            EngineEvent::Token { .. } => {}
            EngineEvent::Done { .. } => break,
            EngineEvent::Prefilled { .. } => {}
            end => panic!("{end:?}"),
        }
    }
    assert_eq!(e.tiles_encoded(), before + 2, "each image encodes once");
    e.shutdown();
}
