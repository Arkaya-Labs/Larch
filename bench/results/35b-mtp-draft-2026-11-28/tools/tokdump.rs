//! Bench-only (not committed): greedy or seeded token ids and top-2 logprobs per prompt, for the
//! draft pass's F0 and E exactness checks. Usage:
//! tokdump <model> <prompts.jsonl> <out.jsonl> <max_tokens> <chats> [temperature] [seed]
use std::{io::Write, path::PathBuf, sync::{Arc, Mutex}};

use larch::{GenParams, Larch, LoadOptions, Prompt, TextEvent};
use larch_model::Message;
use larch_records::knob_id;
use serde_json::{Value, json};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (max_tokens, chats): (u32, usize) = (a[4].parse()?, a[5].parse()?);
    let temp: f32 = a.get(6).map_or(Ok(0.0), |t| t.parse())?;
    let seed: Option<u64> = a.get(7).map(|s| s.parse()).transpose()?;
    let mut opts = LoadOptions::new(PathBuf::from(&a[1]), 0);
    opts.max_context = 2560;
    opts.knob_overrides = vec![(knob_id::MAX_CONTEXT, 2560.0)];
    if let Ok(n) = std::env::var("TOKDUMP_SPLITS") {
        opts.knob_overrides.push((knob_id::ATTN_DECODE_SPLITS, n.parse()?));
    }
    if let Ok(n) = std::env::var("TOKDUMP_SEQS") {
        opts.knob_overrides.push((knob_id::MAX_BATCH_SEQS, n.parse()?));
    }
    let v = Arc::new(Larch::load(opts).map_err(|e| anyhow::anyhow!("load: {e:?}"))?);
    let lines: Vec<Value> = std::fs::read_to_string(&a[2])?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let queue = Arc::new(Mutex::new(lines.into_iter().enumerate().collect::<Vec<_>>()));
    let out = Arc::new(Mutex::new(Vec::new()));
    let t0 = std::time::Instant::now();
    let workers: Vec<_> = (0..chats)
        .map(|_| {
            let (v, queue, out) = (v.clone(), queue.clone(), out.clone());
            std::thread::spawn(move || -> anyhow::Result<()> {
                loop {
                    let Some((i, line)) = queue.lock().unwrap().pop() else {
                        return Ok(());
                    };
                    let msgs: Vec<Message> = (line["messages"].as_array().unwrap().iter())
                        .map(|m| Message {
                            role: m["role"].as_str().unwrap().into(),
                            content: m["content"].as_str().unwrap_or("").into(),
                        })
                        .collect();
                    let p = GenParams {
                        max_tokens,
                        temperature: Some(temp),
                        seed: seed.map(|s| s + i as u64),
                        logprobs: 2,
                        prefix_cache: false,
                        hide_thinking: false,
                        ..GenParams::default()
                    };
                    let (mut ids, mut top) = (Vec::new(), Vec::new());
                    for ev in v.generate(Prompt::Messages(msgs), p)? {
                        match ev {
                            TextEvent::Text { logprobs: Some(l), .. } => {
                                for t in l {
                                    ids.push(t.id);
                                    top.push(t.top.iter().flat_map(|&(i, lp)| [json!(i), json!(lp)]).collect::<Vec<_>>());
                                }
                            }
                            TextEvent::Error(e) => anyhow::bail!("{e:?}"),
                            _ => {}
                        }
                    }
                    let id = line["id"].clone();
                    out.lock().unwrap().push((i, json!({"id": id, "ids": ids, "top": top})));
                }
            })
        })
        .collect();
    for w in workers {
        w.join().unwrap()?;
    }
    let secs = t0.elapsed().as_secs_f64();
    let mut out = Arc::try_unwrap(out).unwrap().into_inner().unwrap();
    out.sort_by_key(|p| p.0);
    let mut f = std::fs::File::create(&a[3])?;
    let n: usize = out.iter().map(|(_, o)| o["ids"].as_array().unwrap().len()).sum();
    for (_, o) in out {
        writeln!(f, "{o}")?;
    }
    eprintln!("tokdump,{n} tokens,{secs:.2} s,{:.1} tok/s,{}", n as f64 / secs, v.guess_stats());
    Ok(())
}
