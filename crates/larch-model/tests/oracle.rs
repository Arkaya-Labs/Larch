//! The oracle against hand-computed examples: every expected KL, rate and quantile below is worked out
//! from the formula in `oracle`'s module doc, not by running it. The dumps are written here in
//! `tools/llama-logits`' format; `model_prefill_matches_llama_cpp` runs a real one.

use std::f64::consts::LN_2;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use larch_kernels::{Buffer, CPU, DType, Device, KvPool, Stream, Tensor};
use larch_model::gguf::Gguf;
use larch_model::oracle::{
    self, Dump, Header, OracleError, Prompt, Report, Stats, Top, calibrate, compare,
};
use larch_model::{Model, StateBatch, StepBatch, TokcheckError, Tokenizer, Transformer, tiny};
use serde_json::{Value, json};

const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const PROMPTS_SHA: &str = "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

fn header(n_vocab: u32, prompts: usize, backend: &str) -> Value {
    json!({
        "larch_oracle": 1, "llama_cpp_tag": "b11458", "llama_cpp_commit": "c479922",
        "model_sha256": SHA, "prompts_sha256": PROMPTS_SHA, "n_vocab": n_vocab,
        "prompts": prompts, "top_k": 64, "backend": backend, "n_ubatch": 512,
    })
}

fn line(index: usize, tokens: &[u32], top: &[Top]) -> String {
    let top: Vec<Vec<Value>> = top
        .iter()
        .map(|t| t.iter().map(|&(id, lp)| json!([id, lp])).collect())
        .collect();
    json!({ "index": index, "prompt": "", "tokens": tokens, "top64": top }).to_string()
}

/// A dump of `prompts` (tokens, rows) with `header`, as `llama-logits` writes it.
fn text(header: &Value, prompts: &[(Vec<u32>, Vec<Top>)]) -> String {
    let mut s = header.to_string() + "\n";
    for (i, (tokens, top)) in prompts.iter().enumerate() {
        s += &(line(i, tokens, top) + "\n");
    }
    s
}

fn read(text: &str) -> Result<Dump, OracleError> {
    Dump::read(text.as_bytes(), SHA)
}

/// A whole 4-token vocabulary: p = (1/2, 1/4, 1/8, 1/8).
fn quarters() -> Top {
    [0.5f64, 0.25, 0.125, 0.125]
        .iter()
        .enumerate()
        .map(|(i, p)| (i as u32, p.ln() as f32))
        .collect()
}

/// Ids 0..64 with p_k = 2^-(k+2): half the mass, the rest outside the top 64.
fn halves() -> Top {
    (0..64)
        .map(|k| (k, (-(f64::from(k) + 2.0) * LN_2) as f32))
        .collect()
}

/// `top` with every log-probability moved by `c`.
fn shift(top: &Top, c: f64) -> Top {
    top.iter()
        .map(|&(id, lp)| (id, (f64::from(lp) + c) as f32))
        .collect()
}

fn close(got: f64, want: f64) {
    assert!((got - want).abs() < 1e-6, "{got} != {want}");
}

#[test]
fn reads_what_llama_logits_writes() {
    let d = read(&text(
        &header(4, 2, "CPU"),
        &[(vec![1, 2], vec![quarters(), quarters()]), (vec![], vec![])],
    ))
    .unwrap();
    assert_eq!(
        (d.header.backend.as_str(), d.header.n_vocab, d.prompts.len()),
        ("CPU", 4, 2)
    );
    assert_eq!(d.header.raw["n_ubatch"], 512);
    assert_eq!(d.prompts[0].tokens, [1, 2]);
    assert_eq!(d.prompts[0].top[1], quarters());
    // Windows tools print digests in upper case.
    let text = text(&header(4, 1, "CPU"), &[(vec![1], vec![quarters()])]);
    assert!(Dump::read(text.as_bytes(), &SHA.to_uppercase()).is_ok());
}

/// An M-RoPE dump's images carry their grid and positions, which must hold the image's rows; an
/// InternVL dump's carry neither.
#[test]
fn reads_image_grids() {
    let image = |extra: Value| {
        let mut im = json!({ "at": 1, "n": 6, "embd": "p0-i0.f32", "file": "red.png", "id": "x" });
        for (k, v) in extra.as_object().unwrap() {
            im[k] = v.clone();
        }
        let mut tokens: Vec<i64> = vec![1];
        tokens.extend([-1; 6]);
        tokens.push(2);
        let top: Vec<Vec<Value>> = (0..8)
            .map(|_| quarters().iter().map(|&(id, lp)| json!([id, lp])).collect())
            .collect();
        let l = json!({ "index": 0, "prompt": "", "tokens": tokens, "top64": top, "images": [im] });
        read(&format!("{}\n{l}\n", header(4, 1, "CPU")))
    };
    let d = image(json!({ "grid": [3, 2], "n_pos": 3 })).unwrap();
    let im = &d.prompts[0].images[0];
    assert_eq!(
        (im.at, im.n, im.grid, im.n_pos),
        (1, 6, Some((3, 2)), Some(3))
    );
    let d = image(json!({})).unwrap();
    assert_eq!(
        (d.prompts[0].images[0].grid, d.prompts[0].images[0].n_pos),
        (None, None)
    );
    for bad in [
        json!({ "grid": [2, 2], "n_pos": 2 }),
        json!({ "grid": [3, 2] }),
        json!({ "n_pos": 3 }),
        json!({ "grid": [6], "n_pos": 6 }),
        json!({ "grid": [3, -2], "n_pos": 3 }),
    ] {
        assert!(image(bad.clone()).is_err(), "{bad}");
    }
}

#[test]
fn refuses_malformed_dumps() {
    let ok = |h: &Value| text(h, &[(vec![1], vec![quarters()])]);
    let with = |k: &str, v: Value| {
        let mut h = header(4, 1, "CPU");
        h[k] = v;
        read(&ok(&h)).unwrap_err()
    };
    assert_eq!(read("").unwrap_err(), OracleError::BadLine(1));
    assert_eq!(read("{").unwrap_err(), OracleError::BadLine(1));
    assert_eq!(
        with("larch_oracle", json!(2)),
        OracleError::BadHeader("larch_oracle")
    );
    assert_eq!(with("top_k", json!(20)), OracleError::BadHeader("top_k"));
    assert_eq!(with("n_vocab", json!(0)), OracleError::BadHeader("n_vocab"));
    assert_eq!(
        with("backend", json!("")),
        OracleError::BadHeader("backend")
    );
    assert_eq!(
        with("model_sha256", json!(SHA.to_uppercase())),
        OracleError::BadHeader("model_sha256")
    );
    assert_eq!(
        with("prompts_sha256", json!(&PROMPTS_SHA[1..])),
        OracleError::BadHeader("prompts_sha256")
    );
    assert_eq!(
        with("prompts", Value::Null),
        OracleError::BadHeader("prompts")
    );
    assert_eq!(
        Dump::read(ok(&header(4, 1, "CPU")).as_bytes(), PROMPTS_SHA).unwrap_err(),
        OracleError::WrongModel {
            expected: PROMPTS_SHA.into(),
            found: SHA.into()
        }
    );

    // A cut-off dump, an extra prompt, a prompt out of order.
    assert_eq!(
        read(&ok(&header(4, 2, "CPU"))).unwrap_err(),
        OracleError::PromptCount {
            expected: 2,
            found: 1
        }
    );
    assert_eq!(
        with("prompts", json!(0)),
        OracleError::PromptCount {
            expected: 0,
            found: 1
        }
    );
    let swapped = header(4, 2, "CPU").to_string()
        + "\n"
        + &line(1, &[1], &[quarters()])
        + "\n"
        + &line(0, &[1], &[quarters()]);
    assert_eq!(read(&swapped).unwrap_err(), OracleError::BadLine(2));
    // Not UTF-8.
    let mut bytes = ok(&header(4, 1, "CPU")).into_bytes();
    bytes.extend(b"\xff\n");
    assert_eq!(
        Dump::read(&bytes[..], SHA).unwrap_err(),
        OracleError::Io(std::io::ErrorKind::InvalidData)
    );
}

#[test]
fn refuses_missing_positions_and_invalid_values() {
    let one = |tokens: Vec<u32>, top: Vec<Top>| read(&text(&header(4, 1, "CPU"), &[(tokens, top)]));
    let bad = |position, what| OracleError::BadValue {
        prompt: 0,
        position,
        what,
    };
    assert_eq!(
        one(vec![1, 2], vec![quarters()]).unwrap_err(),
        OracleError::Positions {
            prompt: 0,
            tokens: 2,
            positions: 1
        }
    );
    assert_eq!(
        one(vec![1, 4], vec![quarters(), quarters()]).unwrap_err(),
        bad(1, "token outside the vocabulary")
    );
    let edit = |f: &dyn Fn(&mut Top)| {
        let mut t = quarters();
        f(&mut t);
        one(vec![1, 2], vec![quarters(), t]).unwrap_err()
    };
    assert_eq!(edit(&|t| t.truncate(3)), bad(1, "not 64 entries"));
    assert_eq!(edit(&|t| t[3].0 = 4), bad(1, "id outside the vocabulary"));
    assert_eq!(edit(&|t| t[3].0 = 0), bad(1, "repeated id"));
    assert_eq!(
        edit(&|t| t[0].1 = 0.25),
        bad(1, "log-probability not finite and <= 0")
    );
    assert_eq!(edit(&|t| t.swap(0, 1)), bad(1, "not in descending order"));
    // A near-certain row whose f32 log-sum-exp lands a hair under the max logit is not invalid.
    let sure = vec![(0, 1e-7), (1, -11.5), (2, -13.8), (3, -16.1)];
    assert!(one(vec![1], vec![sure]).is_ok());
    // Logits shifted by their max but not normalised: the top two alone have probability 1.
    assert_eq!(
        edit(&|t| *t = shift(t, (2.0f64).ln())),
        bad(1, "probabilities sum over 1")
    );

    // Shapes that are not [[id, logprob], ...].
    for top64 in [
        json!([[[-1, -0.5]]]),
        json!([[[0.5, -0.5]]]),
        json!([[[0, "-0.5"]]]),
        json!([[[0, -0.5, 1]]]),
        json!([[0, -0.5]]),
    ] {
        let l = json!({ "index": 0, "prompt": "", "tokens": [1], "top64": top64 });
        let t = format!("{}\n{l}\n", header(4, 1, "CPU"));
        assert_eq!(read(&t).unwrap_err(), OracleError::BadLine(2), "{top64}");
    }
}

#[test]
fn identical_runs_agree_exactly() {
    let d = read(&text(
        &header(100, 1, "CPU"),
        &[(vec![1, 2, 3], vec![halves(); 3])],
    ))
    .unwrap();
    let r = compare(&d, &[vec![halves(); 3]], 0.0).unwrap();
    assert_eq!(
        r.all,
        Stats {
            positions: 3,
            top1: 3,
            decisive: 3,
            decisive_top1: 3,
            flip_gap: 0.0,
            mean_kl: 0.0,
            p99_kl: 0.0,
            max_kl: 0.0
        }
    );
    assert!(r.passes(0.0));
}

#[test]
fn kl_examples() {
    // The whole vocabulary: KL((1/2,1/4,1/8,1/8) || (1/4,1/2,1/8,1/8)) = 1/2 ln 2 - 1/4 ln 2 = ln 2 / 4,
    // and q's top-1 (id 1) is not p's.
    let d = read(&text(&header(4, 1, "CPU"), &[(vec![0], vec![quarters()])])).unwrap();
    let mut q = quarters();
    q.swap(0, 1);
    q[0].1 = quarters()[0].1;
    q[1].1 = quarters()[1].1;
    let r = compare(&d, &[vec![q]], 0.0).unwrap();
    close(r.all.mean_kl, LN_2 / 4.0);
    assert_eq!(r.all.top1, 0);

    // q = p e^c on all 64 ids, P = Σ p = 1/2 - 2^-65 of p on them: -c P from the ids plus the rest's
    // (1 - P) ln((1 - P) / (1 - P e^c)), the KL of (P, 1 - P) against (P e^c, 1 - P e^c).
    let d = read(&text(&header(100, 1, "CPU"), &[(vec![0], vec![halves()])])).unwrap();
    let mass = 0.5 - 2f64.powi(-65);
    let shifted = |c: f64| -c * mass + (1.0 - mass) * ((1.0 - mass) / (1.0 - mass * c.exp())).ln();
    let r = compare(&d, &[vec![shift(&halves(), -0.01)]], 0.0).unwrap();
    close(r.all.mean_kl, shifted(-0.01));
    assert_eq!(r.all.top1, 1);
    // q above p on the same ids is no better: the rest's term keeps the sum positive.
    let r = compare(&d, &[vec![shift(&halves(), 0.01)]], 0.0).unwrap();
    close(r.all.mean_kl, shifted(0.01));
    assert!(r.all.mean_kl > 0.0);
    // A run whose f32 log-sum-exp put 1.0005 of mass on the whole vocabulary (within the validity slack)
    // is renormalised: 0 to rounding, never the -ln(1.0005) = -5e-4 its raw values give.
    let d4 = read(&text(&header(4, 1, "CPU"), &[(vec![0], vec![quarters()])])).unwrap();
    let over = compare(&d4, &[vec![shift(&quarters(), 1.0005f64.ln())]], 0.0).unwrap();
    assert!(over.all.mean_kl.abs() < 1e-6, "{}", over.all.mean_kl);

    // An id missing from q's top 64 takes q's 64th: p = 1/2 on id 0 and 1/126 on ids 1..63; q drops id 0
    // and keeps ids 1..63 as p has them, with id 64 last at 1/1000. KL = 1/2 (ln 1/2 - ln 1/1000) =
    // ln(500) / 2, a lower bound: q's true mass on id 0 is at most 1/1000.
    let lp = |p: f64| p.ln() as f32;
    let mut p: Top = vec![(0, lp(0.5))];
    p.extend((1..64).map(|id| (id, lp(0.5 / 63.0))));
    let mut q: Top = (1..64).map(|id| (id, lp(0.5 / 63.0))).collect();
    q.push((64, lp(0.001)));
    let d = read(&text(&header(100, 1, "CPU"), &[(vec![0], vec![p])])).unwrap();
    let r = compare(&d, &[vec![q]], 0.0).unwrap();
    close(r.all.mean_kl, 500f64.ln() / 2.0);
    assert_eq!(r.all.top1, 0, "q's top-1 is id 1 at 1/126, p's id 0 at 1/2");
}

/// A run sharper than the reference with the same top-1, q = softmax(1.3 z) against p = softmax(z) over
/// 200 tokens with z_i = -i/20: their top 64 are the same ids, so the score is the KL coarsened to them
/// and the rest, positive and at most the full KL (computed here over all 200). Without the rest's term
/// the sum is about -0.003 and passes every bound >= 0.
#[test]
fn sharper_runs_fail_the_gate() {
    let log_softmax = |t: f64| -> Vec<f64> {
        let z: Vec<f64> = (0..200).map(|i| -f64::from(i) / 20.0 * t).collect();
        let lse = z.iter().map(|x| x.exp()).sum::<f64>().ln();
        z.iter().map(|x| x - lse).collect()
    };
    let (p, q) = (log_softmax(1.0), log_softmax(1.3));
    let full: f64 = p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum();
    let top = |l: &[f64]| -> Top { (0..64).map(|i| (i, l[i as usize] as f32)).collect() };
    let d = read(&text(&header(200, 1, "CPU"), &[(vec![0], vec![top(&p)])])).unwrap();
    let r = compare(&d, &[vec![top(&q)]], 0.0).unwrap();
    // Top-1 agrees at the one decisive position: the gate fails on KL alone.
    assert_eq!((r.all.top1, r.all.decisive, r.all.decisive_top1), (1, 1, 1));
    assert!(
        r.all.mean_kl > 0.03 && r.all.mean_kl <= full,
        "{} vs full {full}",
        r.all.mean_kl
    );
    assert!(!r.passes(0.0) && !r.passes(0.03));
}

/// p puts 1/128 on each of 64 ids (its rest 1/2); q puts all its mass on them and, past an f32
/// log-sum-exp, 1.0005 of it: q's rest is floored at f32's epsilon e and q renormalised by
/// Z = 1.0005 + e. KL = 1/2 ln((1/128) Z / (1.0005/64)) + 1/2 ln((1/2) Z / e).
#[test]
fn floored_rests_are_renormalised() {
    let flat = |p: f64| -> Top { (0..64).map(|id| (id, p.ln() as f32)).collect() };
    let d = read(&text(
        &header(100, 1, "CPU"),
        &[(vec![0], vec![flat(1.0 / 128.0)])],
    ))
    .unwrap();
    let r = compare(&d, &[vec![flat(1.0005 / 64.0)]], 0.0).unwrap();
    let (e, z) = (f64::from(f32::EPSILON), 1.0005 + f64::from(f32::EPSILON));
    let want = 0.5 * (z / (2.0 * 1.0005)).ln() + 0.5 * (0.5 * z / e).ln();
    assert!(
        (r.all.mean_kl - want).abs() < 1e-5,
        "{} vs {want}",
        r.all.mean_kl
    );
}

/// The validity check lets either side's mass on the whole 4-token vocabulary reach 1 + 1e-3 (an f32
/// log-sum-exp); both are renormalised, so only the shapes count: equal shapes score 0 whichever side
/// is over (the identical pair is (1/2, 1/4, 1/8, 1/8) x 1.0005, whose top-1 is decisive, so the gate
/// reads it), and (1/2, 1/4, 1/8, 1/8) x 1.0005 against q from `kl_examples` scores that example's
/// ln 2 / 4.
#[test]
fn over_mass_on_either_side_is_renormalised() {
    let flat = |m: f64| -> Top { (0..4).map(|id| (id, (m / 4.0).ln() as f32)).collect() };
    let one = |p: Top, q: Top| {
        let d = read(&text(&header(4, 1, "CPU"), &[(vec![0], vec![p])])).unwrap();
        compare(&d, &[vec![q]], 0.0).unwrap()
    };
    let over = shift(&quarters(), 1.0005f64.ln());
    let same = one(over.clone(), over.clone());
    assert_eq!(same.all.max_kl, 0.0);
    assert!(same.passes(0.0));
    for (p, q) in [(1.0005, 1.0), (1.0, 1.0005)] {
        let r = one(flat(p), flat(q));
        assert!(r.all.mean_kl.abs() < 1e-6, "{p} vs {q}: {}", r.all.mean_kl);
    }
    let mut q = quarters();
    q.swap(0, 1);
    (q[0].1, q[1].1) = (quarters()[0].1, quarters()[1].1);
    close(one(over, q).all.mean_kl, LN_2 / 4.0);
}

#[test]
fn ties_in_the_reference_count_either_way() {
    // p is flat over the whole 4-token vocabulary, so a run whose top-1 is id 2 agrees with it.
    let flat: Top = (0..4).map(|id| (id, 0.25f64.ln() as f32)).collect();
    let mut q = flat.clone();
    q.rotate_left(2);
    let d = read(&text(&header(4, 1, "CPU"), &[(vec![0], vec![flat])])).unwrap();
    assert_eq!(compare(&d, &[vec![q.clone()]], 0.0).unwrap().all.top1, 1);
    // Against (1/2, 1/4, 1/8, 1/8), a run sure of id 2 misses; the flat run lists id 0 among its tied
    // top, so it agrees.
    let d = read(&text(&header(4, 1, "CPU"), &[(vec![0], vec![quarters()])])).unwrap();
    let sure2: Top = [2, 3, 0, 1]
        .iter()
        .zip(quarters())
        .map(|(&id, e)| (id, e.1))
        .collect();
    assert_eq!(compare(&d, &[vec![sure2]], 0.0).unwrap().all.top1, 0);
    assert_eq!(compare(&d, &[vec![q]], 0.0).unwrap().all.top1, 1);
    // A run tied at its top between id 0 (p's best) and id 1 agrees, whichever it lists first.
    let mut tied = quarters();
    tied.swap(0, 1);
    tied[0].1 = quarters()[0].1;
    tied[1].1 = quarters()[0].1;
    tied[2..].iter_mut().for_each(|e| e.1 = -20.0);
    assert_eq!(compare(&d, &[vec![tied]], 0.0).unwrap().all.top1, 1);
}

#[test]
fn quantiles_rates_and_the_gate() {
    // 2 prompts x 100 positions over the whole 4-token vocabulary, q = p e^-c_t with c_t = (t+1)/1000,
    // so KL_t = c_t (Σ p = 1): 0.001 .. 0.200.
    let rows = |from: usize| -> Vec<Top> {
        (from..from + 100)
            .map(|t| shift(&quarters(), -((t + 1) as f64) / 1000.0))
            .collect()
    };
    let d = read(&text(
        &header(4, 2, "CPU"),
        &[
            (vec![0; 100], vec![quarters(); 100]),
            (vec![0; 100], vec![quarters(); 100]),
        ],
    ))
    .unwrap();
    let r = compare(&d, &[rows(0), rows(100)], 0.0).unwrap();
    // Nearest rank: ceil(0.99 x 200) = 198th smallest = 0.198; mean = 100.5 / 1000.
    assert_eq!((r.all.positions, r.all.top1), (200, 200));
    close(r.all.mean_kl, 0.1005);
    close(r.all.p99_kl, 0.198);
    close(r.all.max_kl, 0.2);
    // ceil(0.99 x 3) = 3: of three positions, p99 is the largest.
    let d3 = read(&text(
        &header(4, 1, "CPU"),
        &[(vec![0; 3], vec![quarters(); 3])],
    ))
    .unwrap();
    close(
        compare(&d3, &[rows(0)[..3].to_vec()], 0.0)
            .unwrap()
            .all
            .p99_kl,
        0.003,
    );
    // Per prompt: the 99th of 100 values.
    close(r.prompts[0].p99_kl, 0.099);
    close(r.prompts[1].mean_kl, 0.1505);
    close(r.prompts[1].max_kl, 0.2);

    // The gate: mean KL at most the bound; top-1 at least 99 %, so 1 miss in 100 passes and 2 fail.
    assert!(r.passes(0.1006) && !r.passes(0.1004));
    let miss = |n: usize| {
        let d = read(&text(
            &header(4, 1, "CPU"),
            &[(vec![0; 100], vec![quarters(); 100])],
        ))
        .unwrap();
        let mut q = vec![quarters(); 100];
        let mut wrong = quarters();
        wrong.swap(0, 1);
        (wrong[0].1, wrong[1].1) = (quarters()[0].1, quarters()[1].1);
        q[..n].fill(wrong);
        compare(&d, &[q], 0.0).unwrap()
    };
    assert!(miss(1).passes(1.0));
    assert!(!miss(2).passes(1.0));
    let empty = Report {
        all: Stats {
            positions: 0,
            top1: 0,
            decisive: 0,
            decisive_top1: 0,
            flip_gap: 0.0,
            mean_kl: 0.0,
            p99_kl: 0.0,
            max_kl: 0.0,
        },
        prompts: vec![],
    };
    assert!(!empty.passes(1.0), "no positions never passes");
}

#[test]
fn refuses_runs_that_do_not_cover_the_dump() {
    let d = read(&text(
        &header(4, 1, "CPU"),
        &[(vec![0, 1], vec![quarters(); 2])],
    ))
    .unwrap();
    assert_eq!(
        compare(&d, &[vec![quarters(); 2], vec![]], 0.0).unwrap_err(),
        OracleError::PromptCount {
            expected: 1,
            found: 2
        }
    );
    assert_eq!(
        compare(&d, &[vec![quarters()]], 0.0).unwrap_err(),
        OracleError::Positions {
            prompt: 0,
            tokens: 2,
            positions: 1
        }
    );
    let mut q = quarters();
    q.reverse();
    assert_eq!(
        compare(&d, &[vec![quarters(), q]], 0.0).unwrap_err(),
        OracleError::BadValue {
            prompt: 0,
            position: 1,
            what: "not in descending order"
        }
    );
    // A dump edited after reading is checked too.
    let mut cut = d.clone();
    cut.prompts[0].top.pop();
    assert_eq!(
        compare(&cut, &[vec![quarters(); 2]], 0.0).unwrap_err(),
        OracleError::Positions {
            prompt: 0,
            tokens: 2,
            positions: 1
        }
    );
}

/// The protocol's calibration set over the 4-token vocabulary: 100 prompts, 4 of 1101 tokens and 96
/// of 1, 4500 positions; `kl(t)` sets position t's KL against [`quarters`].
fn calibration_dump(backend: &str, kl: impl Fn(usize) -> f64) -> String {
    let mut t = 0;
    let prompts: Vec<(Vec<u32>, Vec<Top>)> = (0..100)
        .map(|i| {
            let n = if i < 4 { 1101 } else { 1 };
            let rows = (0..n)
                .map(|_| {
                    t += 1;
                    shift(&quarters(), -kl(t - 1))
                })
                .collect();
            (vec![0; n], rows)
        })
        .collect();
    text(&header(4, 100, backend), &prompts)
}

#[test]
fn calibrates_on_the_worst_backend_pair() {
    let cpu = read(&calibration_dump("CPU", |_| 0.0)).unwrap();
    // p99 of 4500 positions is the 4455th smallest: 45 outliers leave it alone, 46 move it.
    let cuda = read(&calibration_dump(
        "CUDA",
        |t| if t < 45 { 0.5 } else { 0.001 },
    ))
    .unwrap();
    let metal = read(&calibration_dump(
        "MTL",
        |t| if t < 46 { 0.5 } else { 0.002 },
    ))
    .unwrap();
    let c = calibrate(&cpu, &[cuda.clone(), metal.clone()]).unwrap();
    assert_eq!(c.pairs[0].0, "CUDA");
    close(c.pairs[0].1.p99_kl, 0.001);
    close(c.pairs[1].1.p99_kl, 0.5);
    close(c.bound, 0.75);
    let c = calibrate(&cpu, std::slice::from_ref(&cuda)).unwrap();
    close(c.bound, 1.5 * 0.001);
    assert_eq!(oracle::BOUND_FACTOR, 1.5);
}

#[test]
fn calibration_refuses_mixed_or_short_sets() {
    let cpu = read(&calibration_dump("CPU", |_| 0.0)).unwrap();
    let cuda = read(&calibration_dump("CUDA", |_| 0.001)).unwrap();
    let mismatch = |f: &dyn Fn(&mut Dump)| {
        let mut o = cuda.clone();
        f(&mut o);
        calibrate(&cpu, &[o]).unwrap_err()
    };
    assert_eq!(
        mismatch(&|o| o.header.model_sha256 = PROMPTS_SHA.into()),
        OracleError::Mismatch("model_sha256")
    );
    assert_eq!(
        mismatch(&|o| o.header.prompts_sha256 = SHA.into()),
        OracleError::Mismatch("prompts_sha256")
    );
    assert_eq!(
        mismatch(&|o| o.header.llama_cpp_commit = "0000000".into()),
        OracleError::Mismatch("llama_cpp_commit")
    );
    assert_eq!(
        mismatch(&|o| o.prompts[7].tokens[0] = 1),
        OracleError::Mismatch("tokens")
    );
    assert_eq!(
        mismatch(&|o| o.header.raw = cpu.header.raw.clone()),
        OracleError::Mismatch("run")
    );
    assert_eq!(
        calibrate(&cpu, &[]).unwrap_err(),
        OracleError::Mismatch("no other run")
    );
    let same = read(&calibration_dump("CUDA", |_| 0.0)).unwrap();
    assert_eq!(
        calibrate(&cpu, &[same]).unwrap_err(),
        OracleError::NoBound(0.0)
    );
    // Identical rows whose mass is 1.0005 (within the validity slack) are as identical.
    let over = |b| read(&calibration_dump(b, |_| -(1.0005f64.ln()))).unwrap();
    assert_eq!(
        calibrate(&over("CPU"), &[over("CUDA")]).unwrap_err(),
        OracleError::NoBound(0.0)
    );

    let mut short = cpu.clone();
    short.prompts.pop();
    assert_eq!(
        calibrate(&short, std::slice::from_ref(&cuda)).unwrap_err(),
        OracleError::CalibrationSet {
            prompts: 99,
            long: 4
        }
    );
    let mut few_long = cpu.clone();
    few_long.prompts[0].tokens.truncate(1100);
    few_long.prompts[0].top.truncate(1100);
    assert_eq!(
        calibrate(&few_long, &[cuda]).unwrap_err(),
        OracleError::CalibrationSet {
            prompts: 100,
            long: 3
        }
    );
}

/// A 4-token row whose top-1, id 0, leads id 1 by `gap` nats (exact for a dyadic gap), and the run
/// that flips it: ids 0 and 1 swapped.
fn gapped(gap: f32) -> (Top, Top) {
    let p: Top = vec![(0, -1.0), (1, -1.0 - gap), (2, -3.0), (3, -3.0)];
    let mut flip = p.clone();
    (flip[0].0, flip[1].0) = (1, 0);
    (p, flip)
}

/// Single-token prompts 10, 11 and 12 get reference gaps 0.1, 0.3 and 0.2; the MTL run flips at 0.2,
/// the CUDA run at all three, the Vulkan run at 0.1. The margin is the widest flip, 0.3, which is
/// neither the first nor the last pair's, nor the first or the last flip of its run; the bound is
/// still 1.5 x the worst p99, which three flipped positions of 4500 leave alone.
#[test]
fn the_margin_is_the_widest_flip() {
    let gaps = [0.1, 0.3, 0.2];
    let run = |backend, kl: f64, flips: &[f32]| {
        let mut o = read(&calibration_dump(backend, |_| kl)).unwrap();
        for (i, &g) in gaps.iter().enumerate() {
            let (p, flip) = gapped(g);
            o.prompts[10 + i].top[0] = if flips.contains(&g) { flip } else { p };
        }
        o
    };
    let cpu = run("CPU", 0.0, &[]);
    let c = calibrate(
        &cpu,
        &[
            run("MTL", 0.002, &[0.2]),
            run("CUDA", 0.001, &gaps),
            run("Vulkan", 0.0005, &[0.1]),
        ],
    )
    .unwrap();
    close(c.pairs[0].1.flip_gap, 0.2);
    close(c.pairs[1].1.flip_gap, 0.3);
    close(c.pairs[2].1.flip_gap, 0.1);
    close(c.margin, 0.3);
    close(c.bound, 1.5 * 0.002);
}

/// Margin 0.25 over 102 positions: reference gaps 0.125 and 0.25, then 100 of 0.375. A run that flips
/// only the first two passes on 100/100 decisive positions although its raw top-1, 100/102, is under
/// 99 %; flipping two more, above the margin, leaves 98/100 and fails.
#[test]
fn a_confident_flip_fails_the_gate() {
    let rows: Vec<(Top, Top)> = [0.125, 0.25]
        .into_iter()
        .chain([0.375; 100])
        .map(gapped)
        .collect();
    let reference = rows.iter().map(|r| r.0.clone()).collect();
    let d = read(&text(&header(4, 1, "CPU"), &[(vec![0; 102], reference)])).unwrap();
    let flip = |n: usize, margin| {
        let q: Vec<Top> = rows
            .iter()
            .enumerate()
            .map(|(t, (p, f))| if t < n { f } else { p }.clone())
            .collect();
        compare(&d, &[q], margin).unwrap()
    };
    let near = flip(2, 0.25);
    let a = &near.all;
    assert_eq!(
        (a.positions, a.top1, a.decisive, a.decisive_top1),
        (102, 100, 100, 100)
    );
    assert_eq!(a.flip_gap, 0.25);
    assert!(a.top1_rate() < oracle::TOP1_MIN && near.passes(1.0));
    // At margin 0 the same two flips are decisive: 100/102.
    assert!(!flip(2, 0.0).passes(1.0));
    let sure = flip(4, 0.25);
    let a = &sure.all;
    assert_eq!((a.top1, a.decisive, a.decisive_top1), (98, 100, 98));
    assert_eq!(a.flip_gap, 0.375);
    assert!(!sure.passes(1.0));

    // At margin 0 an exact tie is not decisive; a 1-token vocabulary's single entry is.
    let decisive = |n_vocab, row: Top| {
        let d = read(&text(
            &header(n_vocab, 1, "CPU"),
            &[(vec![0], vec![row.clone()])],
        ))
        .unwrap();
        compare(&d, &[vec![row]], 0.0).unwrap().all.decisive
    };
    let flat: Top = (0..4).map(|id| (id, 0.25f64.ln() as f32)).collect();
    assert_eq!((decisive(4, flat), decisive(1, vec![(0, 0.0)])), (0, 1));
}

/// Another run need not be another backend: the reference's backend at another n_ubatch pairs with it
/// (an identical header is refused in `calibration_refuses_mixed_or_short_sets`).
#[test]
fn calibrates_on_another_run_of_one_backend() {
    let cpu = read(&calibration_dump("CPU", |_| 0.0)).unwrap();
    let mut ub32 = read(&calibration_dump("CPU", |_| 0.001)).unwrap();
    ub32.header.raw["n_ubatch"] = json!(32);
    let c = calibrate(&cpu, &[ub32]).unwrap();
    assert_eq!(c.pairs[0].0, "CPU");
    close(c.bound, 1.5 * 0.001);
}

/// llama.cpp's tokens (`synth-llama3.tokens.jsonl`, from tokdump at b11458) as a dump of the tiny
/// tokenizer fixture, whose vocabulary they are.
#[test]
fn dump_tokens_are_checked_against_the_tokenizer() {
    let bytes = std::fs::read(tiny::TOKENIZER_GGUF).unwrap();
    let g = Gguf::parse(&bytes).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/synth-llama3.tokens.jsonl"
    );
    let prompts: Vec<Prompt> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap();
            let tokens: Vec<u32> = serde_json::from_value(v["tokens"].clone()).unwrap();
            Prompt {
                text: v["prompt"].as_str().unwrap().into(),
                top: vec![halves(); tokens.len()],
                tokens,
                images: Vec::new(),
            }
        })
        .collect();
    let n_vocab = g
        .get("tokenizer.ggml.tokens")
        .unwrap()
        .as_arr()
        .unwrap()
        .len() as u32;
    let header = Header {
        llama_cpp_tag: "b11458".into(),
        llama_cpp_commit: "c479922".into(),
        model_sha256: SHA.into(),
        prompts_sha256: PROMPTS_SHA.into(),
        backend: "CPU".into(),
        n_vocab,
        prompts: prompts.len(),
        raw: Value::Null,
    };
    let mut d = Dump { header, prompts };
    assert_eq!(d.tokcheck(&tok), Ok(1100));
    d.prompts[3].tokens[0] += 1;
    assert!(matches!(
        d.tokcheck(&tok),
        Err(TokcheckError::Mismatch {
            prompt: 3,
            position: 0,
            ..
        })
    ));
    d.prompts.clear();
    assert_eq!(d.tokcheck(&tok), Err(TokcheckError::Empty));
}

/// Rows per model step in [`model_prefill_matches_llama_cpp`].
const ROWS: usize = 64;

/// The top 64 of one row of logits, as `llama-logits` takes it: log-softmax in f64, by logit descending,
/// ties by id.
fn top64(logits: &[f32]) -> Top {
    let max = logits.iter().fold(f64::MIN, |m, &l| m.max(f64::from(l)));
    let lse = max
        + logits
            .iter()
            .map(|&l| (f64::from(l) - max).exp())
            .sum::<f64>()
            .ln();
    let by_logit = |a: &u32, b: &u32| {
        logits[*b as usize]
            .total_cmp(&logits[*a as usize])
            .then(a.cmp(b))
    };
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    if ids.len() > oracle::TOP_K {
        ids.select_nth_unstable_by(oracle::TOP_K, by_logit);
        ids.truncate(oracle::TOP_K);
    }
    ids.sort_by(by_logit);
    ids.iter()
        .map(|&i| (i, (f64::from(logits[i as usize]) - lse) as f32))
        .collect()
}

/// The platform hasher's digest of a known file, also at a relative path that reads as an option; a
/// missing file is the open error and a hasher that ran and failed is its failure, never "no hasher".
#[test]
fn sha256_hashes_a_file() {
    let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    let name = format!("larch-sha-{}", std::process::id());
    // The second is relative to the package root, the test's working directory.
    for p in [std::env::temp_dir().join(&name), format!("-{name}").into()] {
        std::fs::write(&p, b"abc").unwrap();
        let got = oracle::sha256(&p);
        std::fs::remove_file(&p).unwrap();
        assert_eq!(got.unwrap(), abc, "{}", p.display());
    }
    let e = oracle::sha256(Path::new("no-such-file")).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
    assert!(
        !e.to_string().contains("sha256sum"),
        "not the hasher's: {e}"
    );
    // A directory opens, except on Windows, and every hasher fails on it.
    let dir = std::env::temp_dir();
    let e = oracle::sha256(&dir).unwrap_err();
    assert!(!e.to_string().contains("no sha256sum"), "{e}");
    let hasher = e.kind() == std::io::ErrorKind::Other;
    assert!(hasher || File::open(&dir).is_err(), "{e}");
}

/// The CPU reference against a llama.cpp dump of the same GGUF: the tokenizer on the dump's texts, then
/// one model-level prefill per prompt (64-row chunks over a paged KV pool) compared position by
/// position; a model with recurrent state (Qwen 3.5) runs each prompt from a reset state slot. This is
/// the model's own numbers, not D16's engine parity, which runs through the engine's
/// teacher-forcing interface. The CPU reference is slow: `--release`, and LARCH_ORACLE_PROMPTS caps
/// the prompts. Gated as `larch parity` gates: top-1 >= 99 % at the positions decisive at
/// LARCH_ORACLE_MARGIN (the calibrated tie margin, default 0), and LARCH_ORACLE_BOUND, once
/// calibrated, adds the mean-KL bound. The dump must be of the GGUF itself: its header's sha256
/// against the file's (`WrongModel` otherwise). LARCH_ORACLE_DEVICE (an ordinal, 0 = the first GPU)
/// runs the model on a GPU instead.
///   LARCH_ORACLE_GGUF=<gguf> LARCH_ORACLE_DUMP=<CPU dump.jsonl> [LARCH_ORACLE_MARGIN=<nats>]
///   [LARCH_ORACLE_BOUND=<bound>] [LARCH_ORACLE_DEVICE=<ordinal>]
///   cargo test --release -p larch-model --test oracle -- --ignored --nocapture model_prefill
#[test]
#[ignore = "needs a GGUF and its llama-logits dump: LARCH_ORACLE_GGUF and LARCH_ORACLE_DUMP"]
fn model_prefill_matches_llama_cpp() {
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"));
    let gguf = env("LARCH_ORACLE_GGUF");
    let f = File::open(env("LARCH_ORACLE_DUMP")).unwrap();
    let mut d = Dump::read(
        BufReader::new(f),
        &oracle::sha256(Path::new(&gguf)).unwrap(),
    )
    .unwrap();
    if let Ok(n) = std::env::var("LARCH_ORACLE_PROMPTS") {
        d.prompts.truncate(n.parse().unwrap());
    }
    let f = File::open(&gguf).unwrap();
    let g = Gguf::read(BufReader::new(&f), f.metadata().unwrap().len()).unwrap();
    let n = d.tokcheck(&Tokenizer::from_gguf(&g).unwrap()).unwrap();
    eprintln!("tokcheck: all {n} prompts tokenise as llama.cpp's dump");

    let device = std::env::var("LARCH_ORACLE_DEVICE").map_or(CPU, |o| o.parse().unwrap());
    let dev = Device::open(device).unwrap();
    let s = Stream::new(&dev).unwrap();
    let mut model = Transformer::load(Path::new(&gguf), &dev, &s, ROWS as u32).unwrap();
    let c = model.config().clone();
    if c.arch.hybrid() {
        model.alloc_state(&s, 1).unwrap();
    }
    assert_eq!(c.vocab, d.header.n_vocab);
    let longest = d.prompts.iter().map(|p| p.tokens.len()).max().unwrap_or(0);
    let blocks = longest.div_ceil(16).max(1);
    let kv = (c.kv_layers * c.n_kv_heads * c.head_dim) as u64 * (blocks * 16 * 2) as u64;
    let (k, v) = (
        Buffer::alloc(&dev, kv).unwrap(),
        Buffer::alloc(&dev, kv).unwrap(),
    );
    let pool = KvPool {
        k: &k,
        v: &v,
        n_blocks: blocks as u32,
        block_tokens: 16,
        n_layers: c.kv_layers,
        n_kv_heads: c.n_kv_heads,
        head_dim: c.head_dim,
        dtype: DType::F16,
    };
    let input = Buffer::alloc(&dev, ((4 * ROWS + blocks + 4) * 4) as u64).unwrap();
    let vocab = c.vocab as usize;
    let out = Buffer::alloc(&dev, (ROWS * vocab * 4) as u64).unwrap();
    // One sequence over blocks 0, 1, 2, ...: a position's KV slot is the position.
    let table: Vec<i32> = (0..blocks as i32).collect();
    let mut run = Vec::new();
    for p in &d.prompts {
        if model.state_slots() > 0 {
            model.reset_state(&s, 0).unwrap();
        }
        let mut rows = Vec::new();
        for start in (0..p.tokens.len()).step_by(ROWS) {
            let end = (start + ROWS).min(p.tokens.len());
            let tokens: Vec<i32> = p.tokens[start..end].iter().map(|&t| t as i32).collect();
            let pos: Vec<i32> = (start as i32..end as i32).collect();
            let want: Vec<i32> = (0..tokens.len() as i32).collect();
            let parts: [(&[i32], Vec<i64>); 9] = [
                (&tokens, vec![tokens.len() as i64]),
                (&pos, vec![pos.len() as i64]),
                (&table, vec![1, blocks as i64]),
                (&[], vec![0]),
                (&[start as i32], vec![1]),
                (&[tokens.len() as i32], vec![1]),
                (&want, vec![want.len() as i64]),
                (&[0], vec![1]),
                (&[tokens.len() as i32], vec![1]),
            ];
            let mut at = 0;
            let mut t = Vec::new();
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
                prefill_rows: tokens.len() as u32,
                block_table: t[2],
                seq_lens: t[3],
                prefill_q_start: t[4],
                prefill_q_len: t[5],
                logits_rows: t[6],
                attn_splits: 0,
                check: false,
            };
            let shape = [tokens.len() as i64, vocab as i64];
            let logits = Tensor::new(&out, 0, DType::F32, &shape).unwrap();
            // State slot 0 holds the prompt's recurrent state; an attention model ignores it.
            let state = StateBatch {
                slots: t[7],
                q_lens: t[8],
            };
            model
                .forward_state(&s, &batch, &state, &pool, &logits, None)
                .unwrap();
            let mut l = vec![0f32; tokens.len() * vocab];
            s.download(&out, 0, &mut l).unwrap();
            rows.extend(l.chunks(vocab).map(top64));
        }
        run.push(rows);
    }
    let margin = std::env::var("LARCH_ORACLE_MARGIN").map_or(0.0, |m| m.parse().unwrap());
    let r = compare(&d, &run, margin).unwrap();
    for (i, p) in r.prompts.iter().enumerate() {
        eprintln!(
            "prompt {i}: {} positions, top-1 {}/{}, KL mean {:.3e} max {:.3e}",
            p.positions, p.top1, p.positions, p.mean_kl, p.max_kl
        );
    }
    let a = &r.all;
    eprintln!(
        "model prefill, CPU reference vs llama.cpp {}: {} positions, top-1 {:.4} (raw), {} decisive \
         at margin {margin} with top-1 {:.4}, KL mean {:.3e} p99 {:.3e} max {:.3e}",
        d.header.backend,
        a.positions,
        a.top1_rate(),
        a.decisive,
        a.decisive_top1_rate(),
        a.mean_kl,
        a.p99_kl,
        a.max_kl
    );
    assert!(a.decisive_top1_rate() >= oracle::TOP1_MIN);
    if let Ok(b) = std::env::var("LARCH_ORACLE_BOUND") {
        assert!(r.passes(b.parse().unwrap()), "the gate at bound {b}");
    }
}
