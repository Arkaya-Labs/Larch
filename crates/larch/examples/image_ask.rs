//! One question about one image, through a session (the path the desktop app takes): loads the model
//! with its projector, sends the image and the question, streams the answer and the time to the
//! first token and per token.
//!   cargo run --release -p larch --example image_ask -- <model.gguf> <mmproj.gguf> <image> \
//!       "<question>" [device: 0, -1 the CPU reference, -2 the CPU] [max tokens: 64]

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use larch::{GenParams, Larch, LoadOptions, SessionOptions, TextEvent};
use larch_model::vision::Image;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        a.len() >= 5,
        "usage: image_ask <model.gguf> <mmproj.gguf> <image> <question> [device] [max tokens]"
    );
    let device: i32 = a.get(5).map_or(Ok(0), |d| d.parse())?;
    let max_tokens: u32 = a.get(6).map_or(Ok(64), |n| n.parse())?;
    let t0 = Instant::now();
    let mut opts = LoadOptions::new(a[1].clone().into(), device);
    opts.mmproj = Some(a[2].clone().into());
    opts.max_context = 4096;
    // The CPU reference encodes a whole image and prefills in one step of a minute or more, past
    // its watchdog's 60 s floor; a smoke run waits.
    if device < 0 {
        opts.step_timeout_floor = Some(std::time::Duration::from_secs(1800));
    }
    let v = Larch::load(opts).map_err(|e| anyhow::anyhow!("load: {e:?}"))?;
    let bytes = std::fs::read(&a[3]).with_context(|| format!("reading {}", a[3]))?;
    let image = Arc::new(Image::new(bytes.into()).map_err(|e| anyhow::anyhow!("{e}"))?);
    let rows = v.image_rows(&image).context("no projector loaded")?;
    println!(
        "loaded in {:.1} s; {}x{} image, {rows} rows",
        t0.elapsed().as_secs_f64(),
        image.width(),
        image.height()
    );
    let mut s = v.session(SessionOptions::default());
    let p = GenParams {
        max_tokens,
        temperature: Some(0.0),
        thinking: Some(false),
        ..GenParams::default()
    };
    let t1 = Instant::now();
    let (mut first, mut tokens) = (None, 0u32);
    for ev in s
        .send_images(&a[4], &[image], p)
        .map_err(|e| anyhow::anyhow!("send: {e:?}"))?
    {
        match ev {
            TextEvent::Text {
                delta, tokens: n, ..
            } => {
                first.get_or_insert_with(|| t1.elapsed());
                tokens += n;
                print!("{delta}");
                use std::io::Write;
                std::io::stdout().flush()?;
            }
            TextEvent::Done { .. } => break,
            other => anyhow::bail!("{other:?}"),
        }
    }
    let total = t1.elapsed();
    let first = first.unwrap_or(total);
    println!(
        "\n\nfirst token {:.2} s; {tokens} tokens, {:.0} ms each after it",
        first.as_secs_f64(),
        (total - first).as_secs_f64() * 1e3 / f64::from(tokens.saturating_sub(1).max(1))
    );
    Ok(())
}
