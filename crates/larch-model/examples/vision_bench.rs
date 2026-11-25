//! Times Qwen 3.5's image encoder alone (no text model): host prep (resize, patches) and the device
//! encode (ViT + merger), per image size, warm, median of N. With LARCH_METAL_PROFILE=1 the Metal
//! backend also prints each kernel's GPU time at exit.
//!   cargo run --release -p larch-model --example vision_bench -- <mmproj.gguf> <d_model> \
//!       [device: 0] [reps: 5] [sizes: 640x480,1280x720,max]
//! `max` is the largest image the cap allows (MAX_IMAGE_TOKENS rows). d_model is the text model's
//! width (0.8B 1024, 2B 2048, 35B-A3B 2048).

use std::path::Path;
use std::time::Instant;

use larch_kernels::{Buffer, DType, Device, Stream, Tensor};
use larch_model::qwen_vision::{QwenPixels, QwenVision};
use larch_model::vision::Rgb;

fn pattern(w: u32, h: u32) -> Rgb {
    let mut data = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            data.extend([
                (x * 7 + y) as u8,
                (y * 5 + x / 3) as u8,
                ((x ^ y) * 3) as u8,
            ]);
        }
    }
    Rgb {
        width: w,
        height: h,
        data,
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(
        a.len() >= 3,
        "usage: vision_bench <mmproj.gguf> <d_model> [device] [reps] [sizes]"
    );
    let d_model: u32 = a[2].parse().unwrap();
    let device: i32 = a.get(3).map_or(0, |d| d.parse().unwrap());
    let reps: usize = a.get(4).map_or(5, |n| n.parse().unwrap());
    let sizes = a.get(5).map_or("640x480,1280x720,max", |s| s.as_str());
    let dev = Device::open(device).unwrap();
    let s = Stream::new(&dev).unwrap();
    let t = Instant::now();
    let v = QwenVision::load(Path::new(&a[1]), &dev, &s, d_model).unwrap();
    let c = *v.config();
    println!(
        "loaded in {:.2} s: {} layers, width {}, ff {}, heads {}, cap {} rows",
        t.elapsed().as_secs_f64(),
        c.n_layer,
        c.n_embd,
        c.n_ff,
        c.n_head,
        c.max_tokens()
    );
    let out = Buffer::alloc(&dev, u64::from(c.max_tokens()) * u64::from(d_model) * 4).unwrap();
    println!("size,resized,patches,rows,prep_ms,encode_ms_median,encode_ms_min");
    for size in sizes.split(',') {
        let (w, h) = match size {
            // A 16:9 image past the cap: smart resize brings it to the cap.
            "max" => (3840, 2160),
            _ => {
                let (w, h) = size.split_once('x').expect("WxH");
                (w.parse().unwrap(), h.parse().unwrap())
            }
        };
        let img = pattern(w, h);
        let t = Instant::now();
        let px = QwenPixels::new(&img, &c);
        let prep = t.elapsed().as_secs_f64() * 1e3;
        let rows = px.tokens(&c);
        let o = Tensor::new(&out, 0, DType::F32, &[i64::from(rows), i64::from(d_model)]).unwrap();
        let mut times = Vec::new();
        for i in 0..reps + 2 {
            s.sync().unwrap();
            let t = Instant::now();
            v.encode(&s, &px, &o).unwrap();
            s.sync().unwrap();
            if i >= 2 {
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let (fw, fh) = c.fit(w, h);
        println!(
            "{size},{fw}x{fh},{},{rows},{prep:.1},{:.1},{min:.1}",
            px.gx * px.gy,
            median(times)
        );
    }
}
