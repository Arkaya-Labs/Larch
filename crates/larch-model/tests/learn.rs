//! The native router-delta trainer on the generated qwen3moe model: forward parity with the
//! Transformer, a finite-difference gradcheck, selection changes, the job's contract, and the device
//! trainer against the CPU reference.

use std::path::{Path, PathBuf};
use std::time::Instant;

use larch_kernels::{Buffer, CPU, DType, Device, KvPool, Stream, Tensor};
use larch_model::learn::{
    Adapt, CLIP, DeviceSuffix, Init, Job, LearnError, MAX_PERSISTENT_STEPS, Mode, RouterCandidate,
    Row, Select, Suffix, Window, norm,
};
use larch_model::{Arch, Model, RouterDeltas, StepBatch, Taps, Transformer, tiny};

const TOKENS: [u32; 12] = [5, 9, 2, 200, 31, 7, 64, 9, 5, 2, 31, 7];

fn fixture(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("larch-learn-{name}-{}.gguf", std::process::id()));
    std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 1)).unwrap();
    path
}

/// One CPU prefill from position 0: every row's logits, and the residual after layer 0.
fn transformer(path: &Path, device: i32, d: Option<&RouterDeltas>) -> (Vec<f32>, Vec<f32>) {
    let dev = Device::open(device).unwrap();
    let s = Stream::new(&dev).unwrap();
    let n = TOKENS.len();
    let mut m = Transformer::load(path, &dev, &s, n as u32).unwrap();
    m.set_router_deltas(&s, d).unwrap();
    let c = m.config().clone();
    let (d, vocab) = (c.d_model as usize, c.vocab as usize);
    let kv = u64::from(c.n_layers * c.n_kv_heads * c.head_dim) * 16 * 2;
    let (k, v) = (
        Buffer::alloc(&dev, kv).unwrap(),
        Buffer::alloc(&dev, kv).unwrap(),
    );
    let pool = KvPool {
        k: &k,
        v: &v,
        n_blocks: 1,
        block_tokens: 16,
        n_layers: c.n_layers,
        n_kv_heads: c.n_kv_heads,
        head_dim: c.head_dim,
        dtype: DType::F16,
    };
    let tok: Vec<i32> = TOKENS.iter().map(|&t| t as i32).collect();
    let pos: Vec<i32> = (0..n as i32).collect();
    let parts: [(&[i32], Vec<i64>); 6] = [
        (&tok, vec![n as i64]),
        (&pos, vec![n as i64]),
        (&[0], vec![1, 1]),
        (&[], vec![0]),
        (&[0], vec![1]),
        (&[n as i32], vec![1]),
    ];
    let input = Buffer::alloc(&dev, 1 << 12).unwrap();
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
        logits_rows: t[1],
        attn_splits: 0,
        check: false,
    };
    let lb = Buffer::alloc(&dev, (n * vocab * 4) as u64).unwrap();
    let ob = Buffer::alloc(&dev, (n * d * 4) as u64).unwrap();
    let logits = Tensor::new(&lb, 0, DType::F32, &[n as i64, vocab as i64]).unwrap();
    let taps = Taps {
        layers: &[0],
        final_state: false,
        rows: t[1],
        out: Tensor::new(&ob, 0, DType::F32, &[n as i64, d as i64]).unwrap(),
    };
    m.forward_taps(&s, &batch, &pool, &logits, Some(&taps))
        .unwrap();
    let (mut l, mut r) = (vec![0f32; n * vocab], vec![0f32; n * d]);
    s.download(&lb, 0, &mut l).unwrap();
    s.download(&ob, 0, &mut r).unwrap();
    (l, r)
}

fn window<'a>(residual: &'a [f32], positions: &'a [i32], mask: &'a [bool]) -> Window<'a> {
    Window {
        residual,
        tokens: &TOKENS,
        positions,
        mask,
        weight: 1.0,
        row: Row::User,
    }
}

fn candidate(layers: Vec<u16>, deltas: Vec<Vec<f32>>) -> RouterCandidate {
    RouterCandidate {
        layers,
        experts: tiny::EXPERTS,
        deltas,
        ..Default::default()
    }
}

/// Rank-`rank` adapters on `ids` per layer, with every B and A entry nonzero (`b_scale` sets B).
fn adapters(layers: Vec<u16>, ids: Vec<Vec<u32>>, rank: u32, b_scale: f32) -> RouterCandidate {
    let (d, ef, r) = (tiny::D as usize, tiny::EXPERT_FFN as usize, rank as usize);
    let deltas = ids
        .iter()
        .zip(&layers)
        .map(|(ids, &l)| {
            let mut v = Vec::new();
            for e in 0..ids.len() {
                let mut x = residual(r * (d + ef));
                x.rotate_left(e + usize::from(l));
                v.extend(x[..d * r].iter().map(|b| b * b_scale));
                v.extend(x[d * r..].iter().map(|a| a * 0.2));
            }
            v
        })
        .collect();
    RouterCandidate {
        layers,
        experts: tiny::EXPERTS,
        deltas,
        rank,
        ids,
        ..Default::default()
    }
}

fn job(layers: Vec<u16>, mode: Mode, steps: u32, lr: f32, cap: f32) -> Job {
    Job {
        layers,
        mode,
        steps,
        lr,
        clip: CLIP,
        cap,
        deadline: None,
        adapt: None,
        cap_from_parent: false,
        origin: None,
    }
}

/// Fed the Transformer's own layer-0 residual, the suffix (layer 1, final norm, head) gives the
/// Transformer's logits. Tolerance: the Transformer keeps activations, K/V and the router in f16, the
/// suffix in f32, so each logit agrees to 2 % of the largest |logit| and every argmax matches.
#[test]
fn suffix_forward_matches_the_transformer() {
    let path = fixture("parity");
    let (want, residual) = transformer(&path, CPU, None);
    let suffix = Suffix::load(&path, 1).unwrap();
    std::fs::remove_file(&path).unwrap();
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let mask = [true; TOKENS.len()];
    let got = suffix
        .logits(&window(&residual, &pos, &mask), None)
        .unwrap();
    let v = tiny::VOCAB as usize;
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    let worst = want
        .iter()
        .zip(&got)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("parity: max |dlogit| {worst:e}, max |logit| {scale:e}");
    assert!(worst <= 0.02 * scale, "{worst} vs {scale}");
    let argmax = |r: &[f32]| (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
    for (a, b) in want.chunks_exact(v).zip(got.chunks_exact(v)) {
        assert_eq!(argmax(a), argmax(b));
    }
}

/// Served expert adapters (Q8_0 on the device: GPU 0 when the build has one) against the trainer's
/// f32 forward of the same candidate: logits within the forward parity bound, the same argmax, and
/// not the base model's.
#[test]
fn served_adapters_match_the_trainer() {
    let path = fixture("served");
    let gpu = larch_kernels::device_count().unwrap() > 0;
    let dev = if gpu { 0 } else { CPU };
    let c = adapters(vec![1], vec![vec![0, 2]], 8, 0.5);
    let d = RouterDeltas {
        layers: c.layers.clone(),
        experts: c.experts,
        deltas: c.deltas.clone(),
        rank: c.rank,
        ids: c.ids.clone(),
    };
    let (base, residual) = transformer(&path, dev, None);
    let (served, _) = transformer(&path, dev, Some(&d));
    let suffix = Suffix::load(&path, 1).unwrap();
    std::fs::remove_file(&path).unwrap();
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let mask = [true; TOKENS.len()];
    let want = suffix
        .logits(&window(&residual, &pos, &mask), Some(&c))
        .unwrap();
    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    let (worst, moved) = (diff(&want, &served), diff(&base, &served));
    eprintln!("served adapters: max |dlogit| {worst:e}, moved {moved:e}, max |logit| {scale:e}");
    assert!(worst <= 0.02 * scale, "{worst} vs {scale}");
    assert!(
        moved > 10.0 * worst,
        "the adapters must matter: {moved} vs {worst}"
    );
    let argmax = |r: &[f32]| (0..r.len()).max_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap();
    let v = tiny::VOCAB as usize;
    for (a, b) in want.chunks_exact(v).zip(served.chunks_exact(v)) {
        assert_eq!(argmax(a), argmax(b));
    }
}

/// Uniform in [-1, 1): a residual of the scale the model sees.
fn residual(n: usize) -> Vec<f32> {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// dL/ddelta against central differences on both layers (so the backward runs through layer 1's
/// attention to layer 0's router), at a point whose top-k margins exceed the step.
#[test]
fn gradient_matches_finite_differences() {
    let path = fixture("gradcheck");
    let suffix = Suffix::load(&path, 0).unwrap();
    std::fs::remove_file(&path).unwrap();
    let res = residual(TOKENS.len() * tiny::D as usize);
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let mut mask = [true; TOKENS.len()];
    mask[..4].fill(false);
    let w = window(&res, &pos, &mask);
    let c = candidate(
        vec![1, 0],
        vec![vec![0.02, -0.01, 0.0, 0.01], vec![0.0, 0.01, -0.02, 0.0]],
    );
    let (_, grad) = suffix.loss_grad(&w, &c).unwrap();
    let h = 1e-2f32;
    let mut checked = 0;
    for (l, gl) in grad.iter().enumerate() {
        for (e, &an) in gl.iter().enumerate() {
            let mut p = c.clone();
            p.deltas[l][e] += h;
            let up = suffix.loss(&w, Some(&p)).unwrap();
            p.deltas[l][e] -= 2.0 * h;
            let down = suffix.loss(&w, Some(&p)).unwrap();
            let fd = (up - down) / (2.0 * f64::from(h));
            let an = f64::from(an);
            eprintln!(
                "layer {} expert {e}: analytic {an:e} fd {fd:e}",
                c.layers[l]
            );
            assert!((fd - an).abs() <= 1e-4 + 0.03 * an.abs(), "{an} vs {fd}");
            checked += usize::from(an != 0.0);
        }
    }
    assert!(checked >= 4, "too few nonzero gradients");
}

/// Forcing experts 0 and 1 everywhere: the other experts' deltas get exactly zero gradient (only
/// chosen weights carry it); a large step then changes selections, which is reported as churn.
#[test]
fn selection_changes_are_handled() {
    let path = fixture("churn");
    let suffix = Suffix::load(&path, 1).unwrap();
    std::fs::remove_file(&path).unwrap();
    let res = residual(TOKENS.len() * tiny::D as usize);
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let mask = [true; TOKENS.len()];
    let w = window(&res, &pos, &mask);
    let forced = candidate(vec![1], vec![vec![2.0, 2.0, -2.0, -2.0]]);
    let (_, g) = suffix.loss_grad(&w, &forced).unwrap();
    assert!(g[0][0] != 0.0 && g[0][1] != 0.0);
    assert_eq!(&g[0][2..], &[0.0, 0.0]);
    let out = suffix
        .train(
            &w,
            &job(vec![1], Mode::Persistent(Some(&forced)), 5, 1.0, 10.0),
        )
        .unwrap();
    eprintln!("churn {:?}", out.stats);
    assert!(out.stats.churn > 0.0 && out.stats.loss_after.is_finite());
}

/// The temporary job lowers the masked loss within the cap; a passed deadline runs no step; bad
/// shapes and parents are errors.
#[test]
fn training_lowers_the_loss_within_the_cap() {
    let path = fixture("train");
    let suffix = Suffix::load(&path, 0).unwrap();
    std::fs::remove_file(&path).unwrap();
    let res = residual(TOKENS.len() * tiny::D as usize);
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let mut mask = [false; TOKENS.len()];
    mask[6..].fill(true);
    let w = window(&res, &pos, &mask);
    let out = suffix
        .train(&w, &job(vec![0, 1], Mode::Temporary, 5, 0.1, 0.25))
        .unwrap();
    eprintln!("{:?}", out.stats);
    assert_eq!(out.stats.steps, 5);
    assert!(out.stats.loss_after < out.stats.loss_before);
    assert!(norm(&out.deltas) <= 0.25);

    let mut late = job(vec![0, 1], Mode::Temporary, 5, 0.1, 0.25);
    late.deadline = Some(Instant::now());
    let out = suffix.train(&w, &late).unwrap();
    assert_eq!(out.stats.steps, 0);
    assert_eq!(out.stats.loss_after, out.stats.loss_before);

    let parent = candidate(vec![1], vec![vec![0.0; tiny::EXPERTS as usize]]);
    let mismatched = job(vec![0, 1], Mode::Persistent(Some(&parent)), 1, 0.1, 0.25);
    assert_eq!(
        suffix.train(&w, &mismatched),
        Err(LearnError::Invalid("parent layers"))
    );
    let short = candidate(vec![0, 1], vec![vec![0.0; 3], vec![0.0; 4]]);
    let bad = job(vec![0, 1], Mode::Persistent(Some(&short)), 1, 0.1, 0.25);
    assert_eq!(suffix.train(&w, &bad), Err(LearnError::Invalid("layers")));
    let over = candidate(vec![0, 1], vec![vec![1.0; 4], vec![0.0; 4]]);
    let big = job(vec![0, 1], Mode::Persistent(Some(&over)), 1, 0.1, 0.25);
    assert_eq!(
        suffix.train(&w, &big),
        Err(LearnError::Invalid("parent exceeds cap"))
    );
    assert_eq!(
        suffix.train(&w, &job(vec![0, 1], Mode::Temporary, 3, 0.1, 0.25)),
        Err(LearnError::Invalid("steps"))
    );
    // A persistent job takes any step count up to the ceiling; temporary ones keep the doc's grid.
    assert!(
        suffix
            .train(&w, &job(vec![0, 1], Mode::Persistent(None), 3, 0.1, 0.25))
            .is_ok()
    );
    for steps in [0, MAX_PERSISTENT_STEPS + 1] {
        let j = job(vec![0, 1], Mode::Persistent(None), steps, 0.1, 0.25);
        assert_eq!(suffix.train(&w, &j), Err(LearnError::Invalid("steps")));
    }
    let w = window(&res[1..], &pos, &mask);
    assert_eq!(
        suffix.train(&w, &job(vec![0], Mode::Temporary, 1, 0.1, 0.25)),
        Err(LearnError::Invalid("residual length"))
    );
}

/// The device trainer (GPU 0 when the build has one, else the CPU reference device, which runs the
/// training ops' CPU implementations) against the CPU reference trainer: logits, loss and gradient
/// through both layers, and a two-step job. Both are f32; only summation order differs.
#[test]
fn device_trainer_matches_the_reference() {
    let path = fixture("device");
    let suffix = Suffix::load(&path, 0).unwrap();
    let gpu = larch_kernels::device_count().unwrap() > 0;
    let dev = Device::open(if gpu { 0 } else { CPU }).unwrap();
    let s = Stream::new(&dev).unwrap();
    let m = Transformer::load(&path, &dev, &s, 600).unwrap();
    std::fs::remove_file(&path).unwrap();
    let ds = DeviceSuffix::new(&m, &s, 0).unwrap();
    assert!(matches!(
        DeviceSuffix::new(&m, &s, 2),
        Err(LearnError::Invalid("first layer"))
    ));
    let res = residual(TOKENS.len() * tiny::D as usize);
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let mut mask = [true; TOKENS.len()];
    mask[..4].fill(false);
    let w = window(&res, &pos, &mask);
    let close = |a: &[f32], b: &[f32], what: &str| {
        let scale = a.iter().fold(0f32, |m, x| m.max(x.abs()));
        let worst = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        eprintln!("{what}: max |diff| {worst:e}, max |ref| {scale:e}");
        assert!(worst <= 1e-4 * scale + 1e-7, "{what}: {worst} vs {scale}");
    };
    close(
        &suffix.logits(&w, None).unwrap(),
        &ds.logits(&w, None).unwrap(),
        "logits",
    );
    let c = candidate(
        vec![1, 0],
        vec![vec![0.02, -0.01, 0.0, 0.01], vec![0.0, 0.01, -0.02, 0.0]],
    );
    let (lr, gr) = suffix.loss_grad(&w, &c).unwrap();
    let (ld, gd) = ds.loss_grad(&w, &c).unwrap();
    assert!((lr - ld).abs() <= 1e-5 * lr.abs(), "loss {lr} vs {ld}");
    close(&gr.concat(), &gd.concat(), "gradient");
    let j = job(vec![0, 1], Mode::Temporary, 2, 0.1, 0.25);
    let (a, b) = (suffix.train(&w, &j).unwrap(), ds.train(&w, &j).unwrap());
    assert_eq!(a.stats.steps, b.stats.steps);
    close(&a.deltas.concat(), &b.deltas.concat(), "trained deltas");

    // Expert adapters: the same loss, gradient, zero-B identity and trained adapters.
    let c = adapters(vec![1, 0], vec![vec![0, 2], vec![1, 3]], 4, 0.05);
    let (lr, gr) = suffix.loss_grad(&w, &c).unwrap();
    let (ld, gd) = ds.loss_grad(&w, &c).unwrap();
    assert!(
        (lr - ld).abs() <= 1e-5 * lr.abs(),
        "adapter loss {lr} vs {ld}"
    );
    close(&gr.concat(), &gd.concat(), "adapter gradient");
    // A window of 600 tokens (several expert groups of each size): the same loss and gradient.
    let long: Vec<u32> = TOKENS.iter().copied().cycle().take(600).collect();
    let (lres, lpos) = (
        residual(600 * tiny::D as usize),
        (0..600).collect::<Vec<i32>>(),
    );
    let lmask: Vec<bool> = (0..600).map(|i| i >= 4).collect();
    let lw = Window {
        tokens: &long,
        ..window(&lres, &lpos, &lmask)
    };
    let (lr, gr) = suffix.loss_grad(&lw, &c).unwrap();
    let (ld, gd) = ds.loss_grad(&lw, &c).unwrap();
    assert!((lr - ld).abs() <= 1e-5 * lr.abs(), "long loss {lr} vs {ld}");
    close(&gr.concat(), &gd.concat(), "long-window gradient");
    // The KL to the base: the same loss and gradient.
    let kw = Window { row: Row::Kl, ..w };
    let (lr, gr) = suffix.loss_grad(&kw, &c).unwrap();
    let (ld, gd) = ds.loss_grad(&kw, &c).unwrap();
    assert!((lr - ld).abs() <= 1e-4 * lr.abs() + 1e-9, "KL {lr} vs {ld}");
    close(&gr.concat(), &gd.concat(), "KL gradient");
    let mut j = job(vec![0, 1], Mode::Temporary, 2, 0.01, 1.0);
    j.adapt = Some(Adapt {
        rank: 4,
        k: 2,
        ..Default::default()
    });
    let (a, b) = (suffix.train(&w, &j).unwrap(), ds.train(&w, &j).unwrap());
    assert_eq!((a.rank, &a.ids), (4, &b.ids));
    assert!(a.stats.deviation <= 1.0 && a.stats.loss_after < a.stats.loss_before);
    close(&a.deltas.concat(), &b.deltas.concat(), "trained adapters");

    // Gate-probability and gradient selection, the LoRA-GA init and LoRA+ agree too.
    for (select, init) in [(Select::Gate, Init::Hash), (Select::Grad, Init::Grad)] {
        j.adapt = Some(Adapt {
            rank: 4,
            k: 2,
            ratio: 4.0,
            select,
            init,
        });
        let (a, b) = (suffix.train(&w, &j).unwrap(), ds.train(&w, &j).unwrap());
        assert_eq!(a.ids, b.ids, "{select:?}");
        close(&a.origin.concat(), &b.origin.concat(), "init");
        // Every entry's gradient at the trained point (B != 0, so the dA path is live).
        let (_, gr) = suffix.loss_grad(&w, &a).unwrap();
        let (_, gd) = ds.loss_grad(&w, &a).unwrap();
        close(&gr.concat(), &gd.concat(), "gradient at the trained point");
        // Adam (eps 1e-8) can turn a gradient under the gradient check's own resolution (1e-4 of the
        // largest, what `close` allows) into any step up to lr, so such an entry's trained value is
        // not comparable. Compared: every entry whose gradient at each step of the reference's
        // trajectory is exactly 0 (dA at B = 0, on both) or resolved.
        let mut one = job(vec![0, 1], Mode::Temporary, 1, 0.01, 1.0);
        one.adapt = j.adapt;
        let start = RouterCandidate {
            deltas: a.origin.clone(),
            ..a.clone()
        };
        let steps = [start, suffix.train(&w, &one).unwrap()]
            .map(|c| suffix.loss_grad(&w, &c).unwrap().1.concat());
        let resolved = |i: usize| {
            steps.iter().all(|g| {
                let top = g.iter().fold(0f32, |m, x| m.max(x.abs()));
                g[i] == 0.0 || g[i].abs() >= 1e-4 * top
            })
        };
        let keep = |v: &[Vec<f32>]| -> Vec<f32> {
            let v = v.concat();
            (0..v.len())
                .filter(|&i| resolved(i))
                .map(|i| v[i])
                .collect()
        };
        let kept = keep(&a.deltas).len();
        eprintln!(
            "compared {kept} of {} trained entries",
            a.deltas.concat().len()
        );
        assert!(
            2 * kept >= a.deltas.concat().len(),
            "{select:?}: too few comparable entries"
        );
        close(&keep(&a.deltas), &keep(&b.deltas), "trained adapters");
    }
}

/// A schedule batch trains on its windows' mean: its first loss is their mean loss and its first Adam
/// step follows the sign of their mean gradient; later steps rotate through the batches.
#[test]
fn schedules_average_their_windows() {
    let path = fixture("schedule");
    let suffix = Suffix::load(&path, 0).unwrap();
    std::fs::remove_file(&path).unwrap();
    let res = residual(TOKENS.len() * tiny::D as usize);
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let (mut early, mut late) = ([false; TOKENS.len()], [false; TOKENS.len()]);
    early[2..6].fill(true);
    late[7..].fill(true);
    let (a, b) = (window(&res, &pos, &early), window(&res, &pos, &late));
    let zero = candidate(vec![0, 1], vec![vec![0.0; 4]; 2]);
    let (la, ga) = suffix.loss_grad(&a, &zero).unwrap();
    let (lb, gb) = suffix.loss_grad(&b, &zero).unwrap();
    let lr = 0.01;
    let out = suffix
        .train_schedule(&[&[a, b]], &job(vec![0, 1], Mode::Temporary, 1, lr, 10.0))
        .unwrap();
    assert!((out.stats.loss_before - (la + lb) / 2.0).abs() < 1e-9);
    for ((d, ga), gb) in out.deltas.concat().iter().zip(ga.concat()).zip(gb.concat()) {
        let g = (ga + gb) / 2.0;
        if g.abs() > 1e-6 {
            assert!(
                (d + lr * g.signum()).abs() < 1e-4,
                "step {d} for mean gradient {g}"
            );
        }
    }
    // A window's weight is its share of the batch: 3:1 here.
    let heavy = Window {
        weight: 3.0,
        ..window(&res, &pos, &late)
    };
    let out = suffix
        .train_schedule(
            &[&[window(&res, &pos, &early), heavy]],
            &job(vec![0, 1], Mode::Temporary, 1, lr, 10.0),
        )
        .unwrap();
    assert!((out.stats.loss_before - (la + 3.0 * lb) / 4.0).abs() < 1e-9);
    for ((d, ga), gb) in out.deltas.concat().iter().zip(ga.concat()).zip(gb.concat()) {
        let g = (ga + 3.0 * gb) / 4.0;
        if g.abs() > 1e-6 {
            assert!(
                (d + lr * g.signum()).abs() < 1e-4,
                "weighted step {d} for {g}"
            );
        }
    }
    // Weights shape the loss, never the choice of experts: a heavy window picks the same ones.
    let mut grad = job(vec![0, 1], Mode::Temporary, 1, lr, 10.0);
    grad.adapt = Some(Adapt {
        rank: 4,
        k: 1,
        select: Select::Grad,
        ..Default::default()
    });
    let picked = |w: f32| {
        let b = Window {
            weight: w,
            ..window(&res, &pos, &late)
        };
        let batch = [window(&res, &pos, &early), b];
        suffix.train_schedule(&[&batch], &grad).unwrap().ids
    };
    assert_eq!(picked(1.0), picked(1000.0));
    let (a, b) = (window(&res, &pos, &early), window(&res, &pos, &late));
    let two = suffix
        .train_schedule(
            &[&[a], &[b]],
            &job(vec![0, 1], Mode::Temporary, 8, lr, 10.0),
        )
        .unwrap();
    assert_eq!(two.stats.steps, 8);
    assert!((two.stats.loss_before - la).abs() < 1e-9);
    assert_eq!(
        suffix.train_schedule(&[], &job(vec![0, 1], Mode::Temporary, 1, lr, 10.0)),
        Err(LearnError::Invalid("schedule"))
    );
}

/// LoRA-GA's A is orthogonal at the hashed rows' norm and catches more of the down projection's
/// gradient than the hashed A; LoRA+ steps B `ratio` times further; `cap_from_parent` bounds the
/// distance from the parent, which may itself be past the cap from the init.
#[test]
fn adapter_levers() {
    let path = fixture("levers");
    let suffix = Suffix::load(&path, 0).unwrap();
    std::fs::remove_file(&path).unwrap();
    let res = residual(TOKENS.len() * tiny::D as usize);
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let mut mask = [true; TOKENS.len()];
    mask[..4].fill(false);
    let w = window(&res, &pos, &mask);
    let (d, ef, r) = (tiny::D as usize, tiny::EXPERT_FFN as usize, 4);
    let per = r * (d + ef);
    let fresh = |adapt: Adapt| {
        let mut j = job(vec![0, 1], Mode::Temporary, 1, 0.01, 1.0);
        j.adapt = Some(adapt);
        j.deadline = Some(Instant::now());
        suffix.train(&w, &j).unwrap()
    };
    let base = Adapt {
        rank: 4,
        k: 2,
        ..Default::default()
    };
    let (hashed, ga) = (
        fresh(base),
        fresh(Adapt {
            init: Init::Grad,
            ..base
        }),
    );
    assert_eq!(hashed.ids, ga.ids);
    let caught = |c: &RouterCandidate| {
        let (_, g) = suffix.loss_grad(&w, c).unwrap();
        g.iter()
            .flat_map(|g| g.chunks_exact(per).map(|e| norm(&[e[..d * r].to_vec()])))
            .collect::<Vec<_>>()
    };
    for (h, g) in caught(&hashed).iter().zip(caught(&ga)) {
        assert!(g > 1.5 * h, "GA catches {g}, hashed {h}");
    }
    for e in ga.deltas.iter().flat_map(|v| v.chunks_exact(per)) {
        assert!(e[..d * r].iter().all(|&b| b == 0.0));
        let a: Vec<&[f32]> = e[d * r..].chunks_exact(ef).collect();
        for (i, x) in a.iter().enumerate() {
            for (k, y) in a.iter().enumerate() {
                let dot: f32 = x.iter().zip(*y).map(|(p, q)| p * q).sum();
                let want = if i == k { 1.0 / 3.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-5, "A rows {i}, {k}: {dot}");
            }
        }
    }

    let step = |ratio: f32| {
        let mut j = job(vec![0, 1], Mode::Temporary, 1, 0.01, 100.0);
        j.adapt = Some(Adapt { ratio, ..base });
        let c = suffix.train(&w, &j).unwrap();
        let moved = |part: std::ops::Range<usize>| {
            c.deltas
                .iter()
                .zip(&c.origin)
                .flat_map(|(v, o)| v.chunks_exact(per).zip(o.chunks_exact(per)))
                .flat_map(|(v, o)| {
                    v[part.clone()]
                        .iter()
                        .zip(&o[part.clone()])
                        .map(|(x, y)| (x - y).abs())
                })
                .fold(0f32, f32::max)
        };
        (moved(0..d * r), moved(d * r..per))
    };
    let ((b1, a1), (b4, a4)) = (step(1.0), step(4.0));
    assert_eq!(a1, a4);
    assert!((b4 / b1 - 4.0).abs() < 1e-3, "B moved {b4} vs {b1}");

    let mut far = hashed.clone();
    far.deltas
        .iter_mut()
        .for_each(|v| v.iter_mut().step_by(7).for_each(|x| *x += 0.5));
    let mut j = job(vec![0, 1], Mode::Persistent(Some(&far)), 2, 0.01, 0.05);
    j.adapt = Some(base);
    assert_eq!(
        suffix.train(&w, &j),
        Err(LearnError::Invalid("parent exceeds cap"))
    );
    j.cap_from_parent = true;
    let out = suffix.train(&w, &j).unwrap();
    assert!(out.stats.deviation <= 0.05 && out.stats.deviation > 0.0);
    assert_eq!(out.origin, hashed.origin);

    // A pinned origin shares one ball across a chain: the second job stays within the cap of the first's start.
    let mut j = job(vec![0, 1], Mode::Persistent(Some(&out)), 2, 0.01, 0.05);
    j.adapt = Some(base);
    j.cap_from_parent = true;
    j.origin = Some(&far.deltas);
    let chained = suffix.train(&w, &j).unwrap();
    let from_far: f64 = chained
        .deltas
        .iter()
        .zip(&far.deltas)
        .flat_map(|(x, y)| x.iter().zip(y).map(|(x, y)| f64::from(x - y).powi(2)))
        .sum::<f64>()
        .sqrt();
    assert!(from_far <= 0.05, "{from_far}");
}

/// dL/d(B, A) of the expert adapters against central differences at a nonzero B, through layer 1's
/// attention to layer 0's adapters; and a zero B gives the base model's logits exactly.
#[test]
fn adapter_gradient_matches_finite_differences() {
    let path = fixture("adapter-grad");
    let suffix = Suffix::load(&path, 0).unwrap();
    std::fs::remove_file(&path).unwrap();
    let res = residual(TOKENS.len() * tiny::D as usize);
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let mut mask = [true; TOKENS.len()];
    mask[..4].fill(false);
    let w = window(&res, &pos, &mask);
    let c = adapters(vec![1, 0], vec![vec![0, 2], vec![1, 3]], 4, 0.05);
    let (_, grad) = suffix.loss_grad(&w, &c).unwrap();
    let h = 1e-2f32;
    let mut checked = 0;
    for (l, gl) in grad.iter().enumerate() {
        for (i, &an) in gl.iter().enumerate().step_by(61) {
            let mut p = c.clone();
            p.deltas[l][i] += h;
            let up = suffix.loss(&w, Some(&p)).unwrap();
            p.deltas[l][i] -= 2.0 * h;
            let down = suffix.loss(&w, Some(&p)).unwrap();
            let fd = (up - down) / (2.0 * f64::from(h));
            let an = f64::from(an);
            assert!(
                (fd - an).abs() <= 1e-4 + 0.03 * an.abs(),
                "{l}/{i}: {an} vs {fd}"
            );
            checked += usize::from(an != 0.0);
        }
    }
    assert!(checked >= 20, "too few nonzero gradients: {checked}");

    let mut zero = c.clone();
    for (v, ids) in zero.deltas.iter_mut().zip(&c.ids) {
        let per = v.len() / ids.len();
        for e in 0..ids.len() {
            v[e * per..e * per + tiny::D as usize * 4].fill(0.0);
        }
    }
    assert_eq!(
        suffix.logits(&w, Some(&zero)).unwrap(),
        suffix.logits(&w, None).unwrap()
    );
}

/// The KL to the base (`Row::Kl`) is zero at the base model and at zero-B adapters, positive away
/// from them, and its gradient (from dKL/dlogits = q - p) matches central differences.
#[test]
fn kl_gradient_matches_finite_differences() {
    let path = fixture("kl-grad");
    let suffix = Suffix::load(&path, 0).unwrap();
    std::fs::remove_file(&path).unwrap();
    let res = residual(TOKENS.len() * tiny::D as usize);
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let mut mask = [true; TOKENS.len()];
    mask[0] = false;
    let w = Window {
        row: Row::Kl,
        ..window(&res, &pos, &mask)
    };
    assert_eq!(suffix.loss(&w, None).unwrap(), 0.0);
    let c = adapters(vec![1, 0], vec![vec![0, 2], vec![1, 3]], 4, 0.05);
    let mut zero = c.clone();
    for (v, ids) in zero.deltas.iter_mut().zip(&c.ids) {
        let per = v.len() / ids.len();
        for e in 0..ids.len() {
            v[e * per..e * per + tiny::D as usize * 4].fill(0.0);
        }
    }
    assert!(suffix.loss(&w, Some(&zero)).unwrap().abs() < 1e-12);
    let (kl, grad) = suffix.loss_grad(&w, &c).unwrap();
    assert!(kl > 1e-6, "the adapters must move the distribution: {kl}");
    let h = 1e-2f32;
    let mut checked = 0;
    for (l, gl) in grad.iter().enumerate() {
        for (i, &an) in gl.iter().enumerate().step_by(61) {
            let mut p = c.clone();
            p.deltas[l][i] += h;
            let up = suffix.loss(&w, Some(&p)).unwrap();
            p.deltas[l][i] -= 2.0 * h;
            let down = suffix.loss(&w, Some(&p)).unwrap();
            let fd = (up - down) / (2.0 * f64::from(h));
            let an = f64::from(an);
            assert!(
                (fd - an).abs() <= 1e-6 + 0.03 * an.abs(),
                "{l}/{i}: {an} vs {fd}"
            );
            checked += usize::from(an.abs() > 1e-7);
        }
    }
    assert!(checked >= 20, "too few nonzero gradients: {checked}");
}

/// A KL row (general text scored against the base) never moves which experts selection picks: it
/// routes as the replay row it scores again.
#[test]
fn a_kl_row_does_not_change_the_selection() {
    let path = fixture("kl-select");
    let suffix = Suffix::load(&path, 0).unwrap();
    std::fs::remove_file(&path).unwrap();
    let pos: Vec<i32> = (0..TOKENS.len() as i32).collect();
    let mut mask = [true; TOKENS.len()];
    mask[0] = false;
    let (ru, mut rg) = (
        residual(TOKENS.len() * tiny::D as usize),
        residual(TOKENS.len() * tiny::D as usize),
    );
    rg.reverse();
    let user = window(&ru, &pos, &mask);
    let replay = Window {
        row: Row::Replay,
        ..window(&rg, &pos, &mask)
    };
    // General text scored against the base: counted with the user's routes, it would move the picks.
    let mut rk = residual(TOKENS.len() * tiny::D as usize);
    rk.rotate_left(97);
    let kl = Window {
        row: Row::Kl,
        ..window(&rk, &pos, &mask)
    };
    for select in [Select::Specificity, Select::Hits] {
        let picked = |batch: &[Window]| {
            let mut j = job(vec![0, 1], Mode::Temporary, 1, 0.01, 1.0);
            j.adapt = Some(Adapt {
                rank: 4,
                k: 1,
                select,
                ..Default::default()
            });
            suffix.train_schedule(&[batch], &j).unwrap().ids
        };
        assert_eq!(
            picked(&[user, replay]),
            picked(&[user, replay, kl]),
            "{select:?}"
        );
    }
}
