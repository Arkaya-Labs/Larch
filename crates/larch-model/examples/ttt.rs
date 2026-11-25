//! Temporary test-time training on a real qwen3moe GGUF: per window, adapt the last four layers'
//! router deltas on the first half, then compare next-token CE on the second half with and without
//! them, at lr 0.01 x 2 steps and lr 0.05 x 5 steps. One window per run. With a GPU, the model loads
//! on GPU 0 (`LARCH_CPU_MOE` layers' experts in host memory, as `--n-cpu-moe`), computes the prefix
//! and trains there ([`DeviceSuffix`]); without one, a window's first run only computes and caches
//! its prefix in CACHE and the next trains on the CPU reference ([`Suffix`]):
//! `LARCH_TTT_TARGET=experts` trains rank-8 adapters on the down projection of the last 8 layers' 4
//! experts with the largest |dL/dW_down| instead (doc I5; cap 4, lr 2e-3 x 5 steps, a persistent
//! job's cap measured from its parent: the second sweep's recipe), with `norm` the distance from
//! where the cap is measured. Knobs: `LARCH_TTT_CAP`, `_LRS` (comma list), `_STEPS` (past 5, chained
//! 5-step jobs), `_RANK`, `_K`, `_LAYERS`, `_RATIO` (LoRA+), `_SELECT` (hits, gate, grad), `_INIT`
//! (hash, grad), `_CHUNKS` (dynamic evaluation: the training half in that many in-order chunks, STEPS
//! each), `_CAP_FROM_PARENT` (1: a persistent job's cap is measured from its parent);
//! `LARCH_TTT_ANCHOR=FILE` and `_ANCHOR2` add each candidate's CE change on a fixed 512-token span
//! of FILE; `LARCH_TTT_CHAIN=N` trains windows 0..N in turn (persistent, each from the last, then
//! decayed toward the init by `_DECAY`) and scores the rest. Anchors and chain need a GPU.
//! `cargo run --release -p larch-model --example ttt -- MODEL TEXT CACHE WINDOW [WINDOWS] [LEN]`.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use larch_kernels::{Buffer, CPU_COMPUTE, DType, Device, KvPool, Stream, Tensor, device_count};
use larch_model::gguf::Gguf;
use larch_model::learn::{
    Adapt, CLIP, DeviceSuffix, Init, Job, LR, Mode, RouterCandidate, Row, Select, Suffix, Window,
    norm,
};
use larch_model::{Model, ModelConfig, Placement, StepBatch, Taps, Tokenizer, Transformer};

/// The residual after layer `tap` at every row of one prefill of `tokens`.
fn residual(m: &mut Transformer, s: &Stream, tokens: &[u32], tap: u16) -> Vec<f32> {
    let dev = s.device();
    let c = m.config().clone();
    let (n, d) = (tokens.len(), c.d_model as usize);
    let bt = 16usize;
    let blocks = n.div_ceil(bt);
    let kv = u64::from(c.n_layers * c.n_kv_heads * c.head_dim) * (blocks * bt) as u64 * 2;
    let (k, v) = (
        Buffer::alloc(dev, kv).unwrap(),
        Buffer::alloc(dev, kv).unwrap(),
    );
    let pool = KvPool {
        k: &k,
        v: &v,
        n_blocks: blocks as u32,
        block_tokens: bt as u32,
        n_layers: c.n_layers,
        n_kv_heads: c.n_kv_heads,
        head_dim: c.head_dim,
        dtype: DType::F16,
    };
    let tok: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let pos: Vec<i32> = (0..n as i32).collect();
    let table: Vec<i32> = (0..blocks as i32).collect();
    let parts: [(&[i32], Vec<i64>); 7] = [
        (&tok, vec![n as i64]),
        (&pos, vec![n as i64]),
        (&table, vec![1, blocks as i64]),
        (&[], vec![0]),
        (&[0], vec![1]),
        (&[n as i32], vec![1]),
        (&[n as i32 - 1], vec![1]),
    ];
    let input = Buffer::alloc(dev, 1 << 16).unwrap();
    let (mut at, mut t) = (0, Vec::new());
    for (data, shape) in &parts {
        s.upload(&input, at, data).unwrap();
        t.push(Tensor::new(&input, at, DType::I32, shape).unwrap());
        at += data.len() as u64 * 4;
    }
    let batch = StepBatch {
        token_ids: t[0],
        positions: t[1],
        slot_ids: t[1],
        decode_rows: 0,
        prefill_rows: n as u32,
        block_table: t[2],
        seq_lens: t[3],
        prefill_q_start: t[4],
        prefill_q_len: t[5],
        logits_rows: t[6],
        attn_splits: 0,
        check: false,
    };
    let lb = Buffer::alloc(dev, u64::from(c.vocab) * 4).unwrap();
    let ob = Buffer::alloc(dev, (n * d * 4) as u64).unwrap();
    let logits = Tensor::new(&lb, 0, DType::F32, &[1, i64::from(c.vocab)]).unwrap();
    let taps = Taps {
        layers: &[tap],
        final_state: false,
        rows: t[1],
        out: Tensor::new(&ob, 0, DType::F32, &[n as i64, d as i64]).unwrap(),
    };
    m.forward_taps(s, &batch, &pool, &logits, Some(&taps))
        .unwrap();
    let mut out = vec![0f32; n * d];
    s.download(&ob, 0, &mut out).unwrap();
    out
}

/// The first `mask.len()` positions of `tokens`, scoring `mask`.
fn span<'a>(
    tokens: &'a [u32],
    residual: &'a [f32],
    positions: &'a [i32],
    mask: &'a [bool],
) -> Window<'a> {
    let (n, d) = (mask.len(), residual.len() / tokens.len());
    Window {
        residual: &residual[..n * d],
        tokens: &tokens[..n],
        positions: &positions[..n],
        mask,
        weight: 1.0,
        row: Row::User,
    }
}

/// An env knob, or `default`.
fn knob<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).map_or(default, |v| v.parse().ok().expect(name))
}

/// Tokens of an anchor's fixed span: Moby Dick's first chapter from its first line, else from the
/// middle of the file.
const ANCHOR: usize = 512;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, default: usize| args.get(i).map_or(default, |a| a.parse().unwrap());
    let (path, text) = (
        Path::new(&args[1]),
        std::fs::read_to_string(&args[2]).unwrap(),
    );
    let (cache, w, windows, len) = (Path::new(&args[3]), arg(4, 0), arg(5, 8), arg(6, 256));
    let f = File::open(path).unwrap();
    let flen = f.metadata().unwrap().len();
    let g = Gguf::read(BufReader::new(f), flen).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    // The story itself, past the licence header.
    let body = &text[text.find("Chapter 1").unwrap_or(0)..];
    let ids = tok.encode(body, false);
    let anchor_ids: Vec<(&str, Vec<u32>)> = [("", "LARCH_TTT_ANCHOR"), ("2", "LARCH_TTT_ANCHOR2")]
        .into_iter()
        .filter_map(|(name, var)| {
            let a = std::fs::read_to_string(std::env::var(var).ok()?).unwrap();
            let from = a
                .find("Call me Ishmael")
                .unwrap_or_else(|| a.floor_char_boundary(a.len() / 2));
            let end = a.ceil_char_boundary(from + 16 * ANCHOR);
            Some((name, tok.encode(&a[from..end], false)[..ANCHOR].to_vec()))
        })
        .collect();
    let cfg = ModelConfig::from_gguf(&g).unwrap();
    let experts = std::env::var("LARCH_TTT_TARGET").is_ok_and(|v| v == "experts");
    let (cap, rank, k, n_layers) = (
        knob("LARCH_TTT_CAP", if experts { 4.0f32 } else { 1.0 }),
        knob("LARCH_TTT_RANK", 8u32),
        knob("LARCH_TTT_K", 4u32),
        knob("LARCH_TTT_LAYERS", if experts { 8u16 } else { 4 }),
    );
    let (ratio, chunks, decay, cap_from_parent) = (
        knob("LARCH_TTT_RATIO", 1.0f32),
        knob("LARCH_TTT_CHUNKS", 1usize),
        knob("LARCH_TTT_DECAY", 0.0f32),
        knob("LARCH_TTT_CAP_FROM_PARENT", u8::from(experts)) != 0,
    );
    let select = match std::env::var("LARCH_TTT_SELECT").as_deref() {
        Ok("hits") => Select::Hits,
        Ok("gate") => Select::Gate,
        Ok("grad") | Err(_) => Select::Grad,
        Ok(v) => panic!("LARCH_TTT_SELECT {v}"),
    };
    let init = match std::env::var("LARCH_TTT_INIT").as_deref() {
        Ok("hash") | Err(_) => Init::Hash,
        Ok("grad") => Init::Grad,
        Ok(v) => panic!("LARCH_TTT_INIT {v}"),
    };
    // Train windows 0..chain one after another, each job continuing from the last candidate.
    let chain = knob("LARCH_TTT_CHAIN", 0usize);
    let steps = std::env::var("LARCH_TTT_STEPS")
        .ok()
        .map(|v| v.parse::<u32>().unwrap());
    let sweep: Vec<(f32, u32)> = match std::env::var("LARCH_TTT_LRS") {
        Ok(l) => l
            .split(',')
            .map(|x| (x.parse().unwrap(), steps.unwrap_or(5)))
            .collect(),
        Err(_) if experts => vec![(2e-3, steps.unwrap_or(5))],
        Err(_) => vec![(LR, steps.unwrap_or(2)), (0.05, steps.unwrap_or(5))],
    };
    let first = cfg.n_layers as u16 - n_layers;
    let stride = ids.len() / windows;
    let window = |w: usize| &ids[w * stride..w * stride + len];
    let tokens = window(w);
    // Both trainers have the same entry points; one body runs either. `$res` maps a window to its
    // residual, `$anchor` is the anchor span's.
    macro_rules! adapt {
        ($suffix:expr, $res:expr, $prefix_s:expr, $anchor:expr) => {{
            let (suffix, res, prefix_s, anchor) = (&$suffix, $res, $prefix_s, $anchor);
            let layers: Vec<u16> = (first..first + n_layers).collect();
            let half = len / 2;
            let pos: Vec<i32> = (0..len as i32).collect();
            // Training sees only the first half: its tokens, residual and an observed-span mask.
            let mut train_mask = vec![true; half];
            train_mask[0] = false;
            let train_of = |w: usize| span(window(w), res(w), &pos, &train_mask);
            let eval_mask: Vec<bool> = (0..len).map(|i| i >= half).collect();
            let eval_of = |w: usize| span(window(w), res(w), &pos, &eval_mask);
            let apos: Vec<i32> = (0..ANCHOR as i32).collect();
            let amask: Vec<bool> = (0..ANCHOR).map(|i| i > 0).collect();
            let anchors: Vec<_> = anchor
                .into_iter()
                .map(|(name, t, r)| {
                    let a = span(t, r, &apos, &amask);
                    let b = suffix.loss(&a, None).unwrap();
                    (name, a, b)
                })
                .collect();
            let anchor_of = |c: &RouterCandidate| {
                anchors
                    .iter()
                    .map(|(name, a, b)| {
                        let after = suffix.loss(a, Some(c)).unwrap();
                        format!(" anchor{name} {:+.2}%", (after / b - 1.0) * 100.0)
                    })
                    .collect::<String>()
            };
            // Window `w`'s training half in `chunks` in-order pieces, each with the tokens before it
            // as context; the trainer takes at most 5 steps a job, so longer runs and later chunks
            // continue from the last candidate.
            let run = |w: usize, mode: Mode, lr: f32, steps: u32| {
                let (mut c, mut seconds, mut before) = (None::<RouterCandidate>, 0.0, None);
                for ch in 0..chunks {
                    let (lo, hi) = ((half * ch / chunks).max(1), half * (ch + 1) / chunks);
                    let mask: Vec<bool> = (0..hi).map(|i| i >= lo).collect();
                    let train = span(window(w), res(w), &pos, &mask);
                    let mut left = steps;
                    while left > 0 {
                        let n = left.min(5);
                        let job = Job {
                            layers: layers.clone(),
                            mode: c.as_ref().map_or(mode, |p| Mode::Persistent(Some(p))),
                            steps: n,
                            lr,
                            clip: CLIP,
                            cap,
                            deadline: None,
                            adapt: experts.then_some(Adapt {
                                rank,
                                k,
                                ratio,
                                select,
                                init,
                            }),
                            cap_from_parent,
                            origin: None,
                        };
                        let next = suffix.train(&train, &job).unwrap();
                        seconds += next.stats.seconds;
                        before.get_or_insert(next.stats.loss_before);
                        c = Some(next);
                        left -= n;
                    }
                }
                let mut c = c.unwrap();
                (c.stats.seconds, c.stats.loss_before, c.stats.steps) =
                    (seconds, before.unwrap(), steps * chunks as u32);
                c
            };
            let label = format!(
                "cap {cap} rank {rank} k {k} layers {n_layers} ratio {ratio} select {select:?} init {init:?} chunks {chunks} pcap {}",
                u8::from(cap_from_parent)
            );
            if chain > 0 {
                let (lr, steps) = sweep[0];
                println!(
                    "chain {label} decay {decay} lr {lr} steps {steps}: train 0..{chain}, eval {chain}..{windows}"
                );
                let mut c: Option<RouterCandidate> = None;
                for cw in 0..chain {
                    let mut next = run(cw, Mode::Persistent(c.as_ref()), lr, steps);
                    for (v, o) in next.deltas.iter_mut().zip(&next.origin) {
                        v.iter_mut().zip(o).for_each(|(x, z)| *x -= decay * (*x - z));
                    }
                    println!(
                        "chain train window {cw} train {:.5}->{:.5} norm {:.4}{}",
                        next.stats.loss_before,
                        next.stats.loss_after,
                        next.stats.deviation,
                        anchor_of(&next)
                    );
                    c = Some(next);
                }
                let c = c.unwrap();
                for ew in chain..windows {
                    let eval = eval_of(ew);
                    let base = suffix.loss(&eval, None).unwrap();
                    let adapted = suffix.loss(&eval, Some(&c)).unwrap();
                    println!(
                        "chain eval window {ew} delta {:+.6} base {base:.5} adapted {adapted:.5}{}",
                        adapted - base,
                        anchor_of(&c)
                    );
                }
                return;
            }
            let (train, eval) = (train_of(w), eval_of(w));
            let zero = RouterCandidate {
                layers: layers.clone(),
                experts: suffix.moe().experts,
                deltas: vec![vec![0.0; suffix.moe().experts as usize]; layers.len()],
                ..Default::default()
            };
            if experts {
                println!("window {w}: prefix {prefix_s:.1}s, expert adapters rank {rank}, {k} per layer");
            } else {
                let (_, grad) = suffix.loss_grad(&train, &zero).unwrap();
                println!(
                    "window {w}: prefix {prefix_s:.1}s, |dL/ddelta| per layer at zero {:?}",
                    grad.iter()
                        .map(|g| norm(std::slice::from_ref(g)))
                        .collect::<Vec<_>>()
                );
            }
            let base = suffix.loss(&eval, None).unwrap();
            for &(lr, steps) in &sweep {
                let c = run(w, Mode::Temporary, lr, steps);
                let adapted = suffix.loss(&eval, Some(&c)).unwrap();
                let params = if experts {
                    format!(" params {}", c.deltas.iter().map(Vec::len).sum::<usize>())
                } else {
                    String::new()
                };
                println!(
                    "row {label} lr {lr} steps {steps} window {w} delta {:+.6} base {base:.5} adapted {adapted:.5} train {:.5}->{:.5} s/step {:.2} churn {:.4} norm {:.4}{params}{}",
                    adapted - base,
                    c.stats.loss_before,
                    c.stats.loss_after,
                    c.stats.seconds / f64::from(c.stats.steps.max(1)),
                    c.stats.churn,
                    c.stats.deviation,
                    anchor_of(&c)
                );
            }
        }};
    }
    let t = Instant::now();
    if device_count().unwrap() > 0 {
        let dev = Device::open(0).unwrap();
        let s = Stream::new(&dev).unwrap();
        let cpu_expert_layers = std::env::var("LARCH_CPU_MOE").map_or(0, |v| v.parse().unwrap());
        let place = Placement {
            cpu_expert_layers,
            ..Default::default()
        };
        let rows = len.max(if anchor_ids.is_empty() { 0 } else { ANCHOR });
        let mut m = Transformer::load_placed(path, &dev, &s, rows as u32, place).unwrap();
        let mut res = vec![Vec::new(); windows];
        for i in (0..windows).filter(|&i| chain > 0 || i == w) {
            res[i] = residual(&mut m, &s, window(i), first - 1);
        }
        let anchor: Vec<_> = anchor_ids
            .iter()
            .map(|(name, a)| (*name, a.as_slice(), residual(&mut m, &s, a, first - 1)))
            .collect();
        let suffix = DeviceSuffix::new(&m, &s, first).unwrap();
        adapt!(
            suffix,
            |i: usize| res[i].as_slice(),
            t.elapsed().as_secs_f64(),
            anchor.iter().map(|(n, t, r)| (*n, *t, r.as_slice()))
        );
        return;
    }
    assert!(
        chain == 0 && anchor_ids.is_empty(),
        "the chain and the anchor need a GPU"
    );
    // The serving engine's prefix (the Transformer on the engine's CPU compute device), cached per window.
    let file = cache.join(format!("w{w}-{len}.f32"));
    let res: Vec<f32> = match std::fs::read(&file) {
        Ok(b) => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Err(_) => {
            let dev = Device::open(CPU_COMPUTE).unwrap();
            let s = Stream::new(&dev).unwrap();
            let mut m = Transformer::load(path, &dev, &s, len as u32).unwrap();
            let r = residual(&mut m, &s, tokens, first - 1);
            let bytes: Vec<u8> = r.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(&file, bytes).unwrap();
            // The prefix alone takes minutes on the CPU reference: train in the next run.
            println!(
                "window {w}: prefix {:.1}s, cached",
                t.elapsed().as_secs_f64()
            );
            return;
        }
    };
    let suffix = Suffix::load(path, first).unwrap();
    adapt!(
        suffix,
        |_: usize| res.as_slice(),
        t.elapsed().as_secs_f64(),
        std::iter::empty::<(&str, &[u32], &[f32])>()
    );
}
