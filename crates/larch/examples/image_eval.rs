//! The image eval: each item of an eval set (one JSON object a line: `id`, `image`, `kind`,
//! `prompt`) asked in a fresh session, under a settings profile; one JSON line a result with the
//! answer and its timings, for `score.py` to grade.
//!   cargo run --release -p larch --example image_eval -- <model.gguf> <mmproj.gguf> <items.jsonl> \
//!       <out.jsonl> '<profile json>' [ids file]
//! Profile keys (all optional): thinking (bool), temperature, top_p, top_k, repeat_penalty, seed,
//! ctx (context, default 8192), max (per kind: {"transcribe":700,"qa":96,"absent":96,
//! "describe":900}), stop (strings), loop_stop (bool: `TextStream::stop_loops`), two_pass (bool:
//! a question after a transcription), warm ("text" or "image:<path>": one turn before the items),
//! pause_ms (idle time after the load, before the first item).

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use larch::{GenParams, Larch, LoadOptions, SessionOptions, TextEvent};
use larch_model::vision::Image;
use serde_json::{Value, json};

fn vram_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

fn params(p: &Value, kind: &str) -> GenParams {
    let mut g = GenParams::default();
    let max = |k: &str, d: u32| p["max"][k].as_u64().map_or(d, |v| v as u32);
    g.max_tokens = match kind {
        "transcribe" => max("transcribe", 700),
        "describe" => max("describe", 900),
        k => max(k, 96),
    };
    g.temperature = Some(p["temperature"].as_f64().unwrap_or(0.0) as f32);
    g.top_p = p["top_p"].as_f64().map(|v| v as f32);
    g.top_k = p["top_k"]
        .as_u64()
        .map(|v| v as u32)
        .or((g.temperature == Some(0.0)).then_some(1));
    g.repeat_penalty = p["repeat_penalty"].as_f64().map(|v| v as f32);
    g.seed = Some(p["seed"].as_u64().unwrap_or(1));
    g.thinking = Some(p["thinking"].as_bool().unwrap_or(false));
    // The app's chats: thinking on, hidden, closed after `thinking_budget` tokens (default 100).
    if let Some(b) = p["thinking_budget"].as_u64() {
        g.thinking_budget = Some(b as u32);
        g.hide_thinking = true;
        g.max_tokens += b as u32;
    }
    g.stop = (p["stop"].as_array().into_iter().flatten())
        .filter_map(|s| s.as_str().map(String::from))
        .collect();
    g
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        a.len() >= 6,
        "usage: image_eval <model> <mmproj> <items> <out> <profile> [ids]"
    );
    let profile: Value = serde_json::from_str(&a[5])?;
    let only: Option<HashSet<String>> = a.get(6).map(|f| {
        std::fs::read_to_string(f)
            .expect("ids file")
            .lines()
            .map(|l| l.trim().to_owned())
            .filter(|l| !l.is_empty())
            .collect()
    });
    let mut opts = LoadOptions::new(PathBuf::from(&a[1]), 0);
    opts.mmproj = Some(PathBuf::from(&a[2]));
    opts.max_context = profile["ctx"].as_u64().unwrap_or(8192) as u32;
    let t_load = Instant::now();
    let v = Larch::load(opts).map_err(|e| anyhow::anyhow!("load: {e:?}"))?;
    let load_s = t_load.elapsed().as_secs_f64();
    let mut out = std::fs::File::create(&a[4])?;
    writeln!(
        out,
        "{}",
        json!({"meta": true, "model": a[1], "profile": profile, "load_s": load_s, "vram_mib": vram_mib()})
    )?;
    // "pause_ms": idle time after the load, as a person takes before pasting the first image; the
    // engine's own warm-up runs in it.
    if let Some(ms) = profile["pause_ms"].as_u64() {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
    // "warm": "text" sends one short text turn, "image:<path>" one image turn, before the items, so
    // the first item's timings show what a warm-up at load would save.
    if let Some(w) = profile["warm"].as_str() {
        let t = Instant::now();
        let mut s = v.session(SessionOptions::default());
        let mut g = params(&profile, "qa");
        g.max_tokens = 4;
        let stream = match w.strip_prefix("image:") {
            Some(path) => {
                let img = Arc::new(
                    Image::new(std::fs::read(path)?.into()).map_err(|e| anyhow::anyhow!("{e}"))?,
                );
                s.send_images("What is this?", &[img], g)
            }
            None => s.send("Hi", g),
        };
        if let Ok(stream) = stream {
            for _ in stream {}
        }
        writeln!(
            out,
            "{}",
            json!({"meta": true, "warm": w, "warm_ms": t.elapsed().as_secs_f64() * 1e3})
        )?;
    }
    let mut seen: HashMap<String, usize> = HashMap::new();
    for row in std::fs::read_to_string(&a[3])?.lines() {
        let it: Value = serde_json::from_str(row)?;
        let id = it["id"].as_str().unwrap_or_default().to_owned();
        if only.as_ref().is_some_and(|o| !o.contains(&id)) {
            continue;
        }
        let kind = it["kind"].as_str().unwrap_or("qa");
        let path = it["image"].as_str().unwrap_or_default().to_owned();
        let first = {
            let n = seen.entry(path.clone()).or_default();
            *n += 1;
            *n == 1
        };
        let bytes: Arc<[u8]> = std::fs::read(&path)?.into();
        let image = match Image::new(bytes) {
            Ok(i) => Arc::new(i),
            Err(e) => {
                writeln!(out, "{}", json!({"id": id, "err": format!("image: {e}")}))?;
                continue;
            }
        };
        let rows = v.image_rows(&image);
        let p = params(&profile, kind);
        let max = p.max_tokens;
        let mut s = v.session(SessionOptions::default());
        let t0 = Instant::now();
        let (mut text, mut err) = (String::new(), None::<String>);
        let (mut prompt_tokens, mut completion, mut ttft, mut tok_s) = (0, 0, 0.0, 0.0);
        let mut first_ms = None;
        let mut looped = false;
        // "two_pass": a question is asked after a transcription of the image in the same session
        // (read, then reason); the answer's timings start before the transcription.
        let mut pass1_ms = None;
        let sent = if profile["two_pass"].as_bool().unwrap_or(false)
            && (kind == "qa" || kind == "absent")
        {
            let read = "Transcribe all the text in this image exactly, line by line. Output only the text.";
            let mut tp = params(&profile, "transcribe");
            tp.thinking = Some(false);
            if let Ok(stream) = s.send_images(read, &[image], tp) {
                for _ in stream {}
            }
            pass1_ms = Some(t0.elapsed().as_secs_f64() * 1e3);
            s.send(it["prompt"].as_str().unwrap_or_default(), p)
        } else {
            s.send_images(it["prompt"].as_str().unwrap_or_default(), &[image], p)
        };
        match sent {
            Err(e) => err = Some(format!("{e:?}")),
            Ok(mut stream) => {
                stream.stop_loops(profile["loop_stop"].as_bool().unwrap_or(false));
                for ev in stream.by_ref() {
                    match ev {
                        TextEvent::Prefilled { .. } | TextEvent::Thinking { .. } => {}
                        TextEvent::Text { delta, .. } => {
                            first_ms.get_or_insert(t0.elapsed().as_secs_f64() * 1e3);
                            text.push_str(&delta);
                        }
                        TextEvent::Done { usage, timings, .. } => {
                            prompt_tokens = usage.prompt_tokens;
                            completion = usage.completion_tokens;
                            ttft = timings.ttft_ms;
                            tok_s = timings.tok_s;
                        }
                        TextEvent::Error(e) => err = Some(format!("{e:?}")),
                    }
                }
                looped = stream.looped();
            }
        }
        let wall = t0.elapsed().as_secs_f64() * 1e3;
        writeln!(
            out,
            "{}",
            json!({
                "id": id, "kind": kind, "text": text, "err": err, "image_rows": rows,
                "first_visit": first, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion, "hit_max": completion >= max, "looped": looped,
                "ttft_ms": ttft, "first_text_ms": first_ms, "pass1_ms": pass1_ms, "tok_s": tok_s, "wall_ms": wall,
            })
        )?;
        out.flush()?;
    }
    writeln!(
        out,
        "{}",
        json!({"meta": true, "end": true, "vram_mib": vram_mib()})
    )?;
    Ok(())
}
