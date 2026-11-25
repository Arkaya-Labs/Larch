//! How an image encode sits beside another chat: chat A streams a long answer; after A's first
//! tokens chat B sends an image and a question. Prints A's longest gap between tokens while B's
//! image is read (and A's median gap), and B's time to first token. With `paste`, the image is
//! prepared (encode on paste) 3 s before B sends, as the app does when it is attached.
//!   cargo run --release -p larch --example vision_pause -- <model.gguf> <mmproj.gguf> <image> \
//!       [paste|nopaste] [device: 0]

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use larch::{GenParams, Larch, LoadOptions, SessionOptions, TextEvent};
use larch_model::vision::Image;

fn params(max_tokens: u32) -> GenParams {
    GenParams {
        max_tokens,
        temperature: Some(0.0),
        thinking: Some(false),
        ..GenParams::default()
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        a.len() >= 4,
        "usage: vision_pause <model> <mmproj> <image> [paste|nopaste] [device]"
    );
    let paste = a.get(4).is_some_and(|m| m == "paste");
    let device: i32 = a.get(5).map_or(Ok(0), |d| d.parse())?;
    let mut opts = LoadOptions::new(a[1].clone().into(), device);
    opts.mmproj = Some(a[2].clone().into());
    opts.max_context = 8192;
    let v = Larch::load(opts).map_err(|e| anyhow::anyhow!("load: {e:?}"))?;
    let bytes = std::fs::read(&a[3]).with_context(|| format!("reading {}", a[3]))?;
    let image = Arc::new(Image::new(bytes.into()).map_err(|e| anyhow::anyhow!("{e}"))?);
    println!(
        "image rows {}",
        v.image_rows(&image).context("no projector")?
    );
    let t0 = Instant::now();
    let (b_send, b_first) = std::thread::scope(|sc| -> Result<(Duration, Duration)> {
        let chat_a = sc.spawn(|| -> Result<Vec<Duration>> {
            let mut s = v.session(SessionOptions::default());
            let mut stamps = Vec::new();
            let q = "Write a long, detailed story about a lighthouse keeper, at least 600 words.";
            for ev in s
                .send(q, params(700))
                .map_err(|e| anyhow::anyhow!("{e:?}"))?
            {
                match ev {
                    TextEvent::Text { .. } => stamps.push(t0.elapsed()),
                    TextEvent::Done { .. } => break,
                    other => anyhow::bail!("{other:?}"),
                }
            }
            Ok(stamps)
        });
        std::thread::sleep(Duration::from_secs(2));
        if paste {
            v.prepare_image(&image)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            std::thread::sleep(Duration::from_secs(3));
        }
        let mut s = v.session(SessionOptions::default());
        let send = t0.elapsed();
        let mut first = None;
        for ev in (s.send_images(
            "What does this image show? One sentence.",
            std::slice::from_ref(&image),
            params(32),
        ))
        .map_err(|e| anyhow::anyhow!("{e:?}"))?
        {
            match ev {
                TextEvent::Text { .. } => {
                    first.get_or_insert(t0.elapsed());
                }
                TextEvent::Done { .. } => break,
                other => anyhow::bail!("{other:?}"),
            }
        }
        let stamps = chat_a.join().expect("chat A")?;
        let first = first.unwrap_or(t0.elapsed());
        let gaps: Vec<(Duration, Duration)> =
            stamps.windows(2).map(|w| (w[1], w[1] - w[0])).collect();
        let mut all: Vec<Duration> = gaps.iter().map(|g| g.1).collect();
        all.sort();
        let window_start = if paste {
            send - Duration::from_secs(3)
        } else {
            send
        };
        let worst = (gaps.iter())
            .filter(|(at, _)| *at >= window_start && *at <= first + Duration::from_millis(200))
            .map(|g| g.1)
            .max()
            .unwrap_or_default();
        println!(
            "mode {},chat A tokens {},A median gap {:.1} ms,A worst gap during B's image {:.1} ms",
            if paste { "paste" } else { "nopaste" },
            stamps.len(),
            all.get(all.len() / 2)
                .map_or(0.0, |d| d.as_secs_f64() * 1e3),
            worst.as_secs_f64() * 1e3
        );
        Ok((send, first))
    })?;
    println!(
        "B time to first token {:.0} ms",
        (b_first - b_send).as_secs_f64() * 1e3
    );
    Ok(())
}
