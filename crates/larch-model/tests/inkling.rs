//! Inkling 2 Mini (`inkling2_loop`) on the CPU reference: the generated model ([`tiny::inkling_gguf`])
//! against llama.cpp b12723's dump of the same bytes (`tests/data/inkling`), and the same logits however a
//! sequence's rows are split into steps (one prefill, chunks, one-row decode, beside other sequences).

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use larch_kernels::{Buffer, CPU, DType, Device, KvPool, Stream, Tensor};
use larch_model::gguf::Gguf;
use larch_model::oracle::{self, Dump, Top, compare};
use larch_model::{Inkling, InklingCfg, Model, StateBatch, StepBatch, Tokenizer, tiny};
use larch_records::LoadError;

/// Rows a step may hold; positions a sequence may reach (blocks of [`BLOCK`]); sequences.
const ROWS: u32 = 64;
const BLOCK: i64 = 16;
const CTX: i64 = 512;
const SEQS: u32 = 5;
/// The generated models' seed and the sha256 of their bytes, which the committed dumps were made from.
const SEED: u64 = 7;
const TINY_SHA256: &str = "68f4332972f4a574326b547b32d9d615c83d81b81d2309b249e25ebf068ec300";
const NVFP4_SHA256: &str = "06b2fd458a56d42f4838b3e0471244a68c923c4758c1f1a27cc4a771f4f2d7e3";

/// A generated model: its seed to its GGUF.
type Generator = fn(u64) -> Vec<u8>;

/// The generated models as llama.cpp's dumps name them: the base model, and its NVFP4 experts' variant.
const MODELS: [(&str, Generator, &str); 2] = [
    ("tiny", tiny::inkling_gguf, TINY_SHA256),
    ("tiny-nvfp4", tiny::inkling_gguf_nvfp4, NVFP4_SHA256),
];

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/inkling")
        .join(name)
}

/// The generated model at a path of its own (tests run in parallel).
fn tiny_file(tag: &str) -> PathBuf {
    model_file(tag, tiny::inkling_gguf)
}

fn model_file(tag: &str, gguf: Generator) -> PathBuf {
    let p = std::env::temp_dir().join(format!("larch-inkling-{tag}-{}.gguf", std::process::id()));
    std::fs::write(&p, gguf(SEED)).unwrap();
    p
}

/// One sequence's part of a step: its slot (state and KV blocks), the position of its first row, its
/// tokens; a decode part is one token.
struct Part {
    slot: i32,
    start: i32,
    tokens: Vec<i32>,
    decode: bool,
}

fn prefill(slot: i32, start: i32, tokens: &[i32]) -> Part {
    Part {
        slot,
        start,
        tokens: tokens.to_vec(),
        decode: false,
    }
}

fn decode(slot: i32, start: i32, token: i32) -> Part {
    Part {
        slot,
        start,
        tokens: vec![token],
        decode: true,
    }
}

/// The model on the CPU reference with a KV pool of [`CTX`] positions for each of [`SEQS`] sequences.
struct Run {
    s: Stream,
    model: Inkling,
    k: Buffer,
    v: Buffer,
    input: Buffer,
    out: Buffer,
    blocks: i64,
    /// Positions a sequence may reach.
    ctx: i64,
}

impl Run {
    fn new(path: &Path) -> Run {
        Run::on(path, CPU, CTX, SEQS)
    }

    /// The model on device `ordinal` with `seqs` sequences of up to `ctx` positions.
    fn on(path: &Path, ordinal: i32, ctx: i64, seqs: u32) -> Run {
        let dev = Device::open(ordinal).unwrap();
        let s = Stream::new(&dev).unwrap();
        let mut model = Inkling::load(path, &dev, &s, ROWS).unwrap();
        model.alloc_state(&s, seqs).unwrap();
        let c = model.config().clone();
        let ctx = (ctx + BLOCK - 1) / BLOCK * BLOCK;
        let blocks = ctx / BLOCK * i64::from(seqs);
        let kv = (c.kv_layers * c.n_kv_heads * c.head_dim) as u64 * (blocks * BLOCK * 2) as u64;
        Run {
            k: Buffer::alloc(&dev, kv).unwrap(),
            v: Buffer::alloc(&dev, kv).unwrap(),
            input: Buffer::alloc(&dev, 64 << 10).unwrap(),
            out: Buffer::alloc(&dev, u64::from(ROWS * c.vocab) * 4).unwrap(),
            model,
            blocks,
            ctx,
            s,
        }
    }

    /// One step over `parts` (decode parts first), the logits of every row.
    fn step(&mut self, parts: &[Part]) -> Vec<Vec<f32>> {
        let per = self.ctx / BLOCK;
        let table = |slot: i32| (0..per as i32).map(move |b| slot * per as i32 + b);
        let slot_of = |slot: i32, p: i32| {
            (i64::from(slot) * per + i64::from(p) / BLOCK) * BLOCK + i64::from(p) % BLOCK
        };
        let (dec, pre): (Vec<&Part>, Vec<&Part>) = parts.iter().partition(|p| p.decode);
        assert!(
            parts.iter().take(dec.len()).all(|p| p.decode),
            "decode parts first"
        );
        let all: Vec<&Part> = dec.iter().chain(&pre).copied().collect();
        let tokens: Vec<i32> = all.iter().flat_map(|p| p.tokens.clone()).collect();
        let pos: Vec<i32> = all
            .iter()
            .flat_map(|p| (p.start..p.start + p.tokens.len() as i32).collect::<Vec<_>>())
            .collect();
        let kv_slots: Vec<i32> = all
            .iter()
            .flat_map(|p| (0..p.tokens.len() as i32).map(|i| slot_of(p.slot, p.start + i) as i32))
            .collect();
        let tables: Vec<i32> = all.iter().flat_map(|p| table(p.slot)).collect();
        let lens: Vec<i32> = dec.iter().map(|p| p.start + 1).collect();
        let q_start: Vec<i32> = pre.iter().map(|p| p.start).collect();
        let q_len: Vec<i32> = pre.iter().map(|p| p.tokens.len() as i32).collect();
        let rows = tokens.len() as i64;
        let every: Vec<i32> = (0..rows as i32).collect();
        let state_slots: Vec<i32> = all.iter().map(|p| p.slot).collect();
        let state_lens: Vec<i32> = all.iter().map(|p| p.tokens.len() as i32).collect();
        let parts: [(&[i32], Vec<i64>); 11] = [
            (&tokens, vec![rows]),
            (&pos, vec![rows]),
            (&kv_slots, vec![rows]),
            (&tables, vec![all.len() as i64, per]),
            (&lens, vec![lens.len() as i64]),
            (&q_start, vec![q_start.len() as i64]),
            (&q_len, vec![q_len.len() as i64]),
            (&every, vec![rows]),
            (&state_slots, vec![all.len() as i64]),
            (&state_lens, vec![all.len() as i64]),
            (&[], vec![0]),
        ];
        let mut at = 0;
        let mut t = Vec::new();
        for (data, shape) in &parts {
            self.s.upload(&self.input, at, data).unwrap();
            t.push(Tensor::new(&self.input, at, DType::I32, shape).unwrap());
            at += (data.len() as u64 * 4).next_multiple_of(16);
        }
        let batch = StepBatch {
            token_ids: t[0],
            positions: t[1],
            slot_ids: t[2],
            decode_rows: dec.len() as u32,
            prefill_rows: (rows - dec.len() as i64) as u32,
            block_table: t[3],
            seq_lens: t[4],
            prefill_q_start: t[5],
            prefill_q_len: t[6],
            logits_rows: t[7],
            attn_splits: 0,
            check: false,
        };
        let state = StateBatch {
            slots: t[8],
            q_lens: t[9],
        };
        let vocab = i64::from(self.model.config().vocab);
        let logits = Tensor::new(&self.out, 0, DType::F32, &[rows, vocab]).unwrap();
        let pool = KvPool {
            k: &self.k,
            v: &self.v,
            n_blocks: self.blocks as u32,
            block_tokens: BLOCK as u32,
            n_layers: self.model.config().kv_layers,
            n_kv_heads: self.model.config().n_kv_heads,
            head_dim: self.model.config().head_dim,
            dtype: DType::F16,
        };
        if let Err(e) = self
            .model
            .forward_state(&self.s, &batch, &state, &pool, &logits, None)
        {
            panic!("forward: {e:?}: {}", larch_kernels::last_error());
        }
        let mut l = vec![0f32; (rows * vocab) as usize];
        self.s.download(&self.out, 0, &mut l).unwrap();
        l.chunks(vocab as usize).map(<[f32]>::to_vec).collect()
    }

    /// A whole sequence from a reset `slot`, in prefill chunks of at most `chunk` rows.
    fn sequence(&mut self, slot: i32, tokens: &[i32], chunk: usize) -> Vec<Vec<f32>> {
        self.model.reset_state(&self.s, slot as u32).unwrap();
        let mut rows = Vec::new();
        for start in (0..tokens.len()).step_by(chunk) {
            let end = (start + chunk).min(tokens.len());
            rows.extend(self.step(&[prefill(slot, start as i32, &tokens[start..end])]));
        }
        rows
    }
}

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

fn tokens(n: usize, seed: i32) -> Vec<i32> {
    (0..n as i32)
        .map(|i| (i * 37 + seed * 11 + 5) % 256)
        .collect()
}

fn same(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: rows");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert!(bits(x) == bits(y), "{what}: row {i} differs");
    }
}

#[test]
fn the_generated_models_are_the_dumps() {
    for (name, gguf, want) in MODELS {
        let p = model_file(&format!("sha-{name}"), gguf);
        let sha = oracle::sha256(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        assert_eq!(
            sha, want,
            "{name}: the generator changed: rewrite its dumps (tests/data/inkling)"
        );
    }
    let bytes = tiny::inkling_gguf(SEED);
    let c = InklingCfg::from_gguf(&Gguf::parse(&bytes).unwrap())
        .unwrap()
        .unwrap();
    assert_eq!((c.layers, c.passes, c.loop_start, c.loop_end), (6, 3, 2, 4));
    assert_eq!(c.swa, [true, true, true, false, true, false]);
}

/// llama.cpp's checks on the keys, as load errors.
#[test]
fn malformed_keys_are_refused() {
    use larch_model::gguf::{Value, write};
    let bytes = tiny::inkling_gguf(SEED);
    let g = Gguf::parse(&bytes).unwrap();
    let with = |key: &str, val: Option<Value>| {
        let mut kv: Vec<(String, Value)> = g.kv.clone().into_iter().collect();
        kv.retain(|(k, _)| k != key);
        if let Some(v) = val {
            kv.push((key.to_string(), v));
        }
        let tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = g
            .tensors
            .iter()
            .map(|t| {
                let at = (g.data_offset + t.offset) as usize;
                let data = bytes[at..at + t.bytes as usize].to_vec();
                (t.name.clone(), t.dims.clone(), t.ggml_type, data)
            })
            .collect();
        InklingCfg::from_gguf(&Gguf::parse(&write(&kv, &tensors)).unwrap())
    };
    let bad = Err(LoadError::BadGguf(8, 0));
    let k = |s: &str| format!("inkling2_loop.{s}");
    for (key, val) in [
        ("loop.start", Some(Value::U32(1))), // not a block boundary
        ("loop.end", Some(Value::U32(7))),   // past the layers
        ("loop.count", Some(Value::U32(1))), // a fixed loop runs at least twice
        ("loop.gate_rank", Some(Value::U32(0))),
        ("shortconv.l_cache", Some(Value::U32(1))),
        ("vocab_size_unpadded", Some(Value::U32(300))), // past the padded width
        ("attention.qk_clip", Some(Value::F32(0.0))),
        ("attention.log_scale_alpha", Some(Value::F32(-1.0))),
        (
            "attention.sliding_window_pattern",
            Some(Value::Arr(vec![Value::Bool(true); 5])),
        ),
        ("attention.relative_extent", None),
    ] {
        assert_eq!(with(&k(key), val), bad, "{key}");
    }
    // Layers of differing widths are llama.cpp's, not this port's.
    let ragged = Value::Arr([4, 4, 4, 4, 4, 2].map(Value::I32).to_vec());
    assert_eq!(
        with(&k("attention.head_count"), Some(ragged)),
        Err(LoadError::UnsupportedArch)
    );
    // Not this architecture.
    assert_eq!(
        with("general.architecture", Some(Value::Str("llama".into()))),
        Ok(None)
    );
}

/// llama.cpp b12723's dumps of the generated model, all on the CPU: the reference (flash attention on, the whole
/// prompt one micro-batch, as `llama-logits` runs by default), then the same with flash attention off and at
/// micro-batches of 7, which differ from it only by rounding.
fn dumps(model: &str, sha: &str) -> (Dump, [Dump; 2]) {
    let read = |run: &str| {
        let f = File::open(data(&format!("{model}.b12723.{run}.jsonl"))).unwrap();
        Dump::read(BufReader::new(f), sha).unwrap()
    };
    (read("cpu"), [read("cpu-fa0"), read("cpu-ub7")])
}

/// The CPU reference against llama.cpp's dumps of the generated models, gated as `larch parity` gates, with the
/// bound and margin calibrated as [`oracle::calibrate`] calibrates them, on llama.cpp's own runs of these prompts
/// (the model is too small for the calibration protocol's prompt set): the bound 1.5 x the worst pair's p99 KL,
/// the margin the widest reference gap at which a pair's top-1 differs. Random weights make this model chaotic:
/// llama.cpp's two attention paths alone disagree by a mean KL of about 1e-3, and rounding flips the experts of
/// near-tied tokens (tests/data/inkling's tensordump tool finds them), so the gate catches a wrong formula, not
/// rounding. The tokenizer is checked on the dump's prompts first.
#[test]
fn prefill_matches_llama_cpp() {
    for (name, gguf, sha) in MODELS {
        oracle_gate(name, gguf, sha);
    }
}

/// [`prefill_matches_llama_cpp`] for one model. The NVFP4 variant checks larch's NVFP4 decoding against
/// llama.cpp's in the model's own experts.
fn oracle_gate(name: &str, gguf: Generator, sha: &str) {
    let (d, others) = dumps(name, sha);
    let p = model_file(&format!("oracle-{name}"), gguf);
    let g = Gguf::parse(&std::fs::read(&p).unwrap()).unwrap();
    assert_eq!(
        d.tokcheck(&Tokenizer::from_gguf(&g).unwrap()),
        Ok(d.prompts.len())
    );
    let pairs: Vec<_> = others
        .iter()
        .map(|o| {
            let tops: Vec<Vec<Top>> = o.prompts.iter().map(|p| p.top.clone()).collect();
            compare(&d, &tops, 0.0).unwrap().all
        })
        .collect();
    let bound = oracle::BOUND_FACTOR * pairs.iter().map(|s| s.p99_kl).fold(0.0, f64::max);
    let margin = pairs.iter().map(|s| s.flip_gap).fold(0.0, f64::max);
    let mut run = Run::new(&p);
    std::fs::remove_file(&p).unwrap();
    let rows: Vec<Vec<Top>> = d
        .prompts
        .iter()
        .map(|pr| {
            let toks: Vec<i32> = pr.tokens.iter().map(|&t| t as i32).collect();
            run.sequence(0, &toks, ROWS as usize)
                .iter()
                .map(|l| top64(l))
                .collect()
        })
        .collect();
    let r = compare(&d, &rows, margin).unwrap();
    for (s, what) in pairs
        .iter()
        .zip(["flash attention off", "micro-batches of 7"])
    {
        eprintln!(
            "{name}, llama.cpp {what}: top-1 {:.4}, KL mean {:.3e} p99 {:.3e} max {:.3e}, flip gap {:.4}",
            s.top1_rate(),
            s.mean_kl,
            s.p99_kl,
            s.max_kl,
            s.flip_gap
        );
    }
    let a = &r.all;
    eprintln!(
        "{name}, larch's CPU reference: {} positions, top-1 {:.4}, {} decisive at margin {margin:.4} with top-1 {:.4}, KL mean \
         {:.3e} p99 {:.3e} max {:.3e}; bound {bound:.3e}",
        a.positions,
        a.top1_rate(),
        a.decisive,
        a.decisive_top1_rate(),
        a.mean_kl,
        a.p99_kl,
        a.max_kl
    );
    assert!(
        r.passes(bound),
        "{name}: the CPU reference misses llama.cpp's gate"
    );
}

/// One prefill, chunks of 7, and a prefill then one-row decode steps give the same logits, bit for bit: the
/// state, the KV and the attention see the same inputs however the rows are split.
#[test]
fn every_split_of_a_sequence_agrees() {
    let p = tiny_file("split");
    let mut run = Run::new(&p);
    std::fs::remove_file(&p).unwrap();
    let toks = tokens(40, 1);
    let whole = run.sequence(1, &toks, ROWS as usize);
    same(&whole, &run.sequence(1, &toks, 7), "chunks of 7");
    run.model.reset_state(&run.s, 2).unwrap();
    let mut rows = run.step(&[prefill(2, 0, &toks[..30])]);
    for (i, &t) in toks.iter().enumerate().skip(30) {
        rows.extend(run.step(&[decode(2, i as i32, t)]));
    }
    same(&whole, &rows, "decode after a prefill");
}

/// Four sequences decoding in one step, beside a prefill chunk of a fifth, each give their own run's logits.
#[test]
fn sequences_in_one_step_are_independent() {
    let p = tiny_file("seqs");
    let mut run = Run::new(&p);
    std::fs::remove_file(&p).unwrap();
    let toks: Vec<Vec<i32>> = (0..5).map(|s| tokens(20 + 3 * s, s as i32 + 2)).collect();
    let alone: Vec<Vec<Vec<f32>>> = toks
        .iter()
        .map(|t| run.sequence(0, t, ROWS as usize))
        .collect();
    // Slots 0-3 prefilled to all but their last token, which they decode beside slot 4's prefill.
    for (s, t) in toks.iter().enumerate().take(4) {
        run.model.reset_state(&run.s, s as u32).unwrap();
        run.step(&[prefill(s as i32, 0, &t[..t.len() - 1])]);
    }
    run.model.reset_state(&run.s, 4).unwrap();
    let mut parts: Vec<Part> = (0..4)
        .map(|s| {
            let t = &toks[s];
            decode(s as i32, t.len() as i32 - 1, t[t.len() - 1])
        })
        .collect();
    parts.push(prefill(4, 0, &toks[4]));
    let rows = run.step(&parts);
    for s in 0..4 {
        same(
            &rows[s..s + 1],
            &alone[s][alone[s].len() - 1..],
            &format!("decode row {s}"),
        );
    }
    same(&rows[4..], &alone[4], "the prefill beside them");
}

/// A copied state slot holds its source's bytes and a saved one reloads them; a reset slot is zeros; and the
/// source runs on as an unbroken prefill.
#[test]
fn state_slots_copy_save_and_reset() {
    let p = tiny_file("copy");
    let mut run = Run::new(&p);
    std::fs::remove_file(&p).unwrap();
    let toks = tokens(24, 9);
    let whole = run.sequence(0, &toks, ROWS as usize);
    run.model.reset_state(&run.s, 0).unwrap();
    run.step(&[prefill(0, 0, &toks[..20])]);
    let save = |run: &Run, slot| {
        let mut b = vec![0u8; run.model.state_size()];
        run.model.save_state(&run.s, slot, &mut b).unwrap();
        b
    };
    let saved = save(&run, 0);
    assert!(saved.iter().any(|&b| b != 0));
    run.model.copy_state(&run.s, 0, 1).unwrap();
    run.model.load_state(&run.s, 2, &saved).unwrap();
    assert!(save(&run, 1) == saved && save(&run, 2) == saved);
    run.model.reset_state(&run.s, 1).unwrap();
    assert!(save(&run, 1).iter().all(|&b| b == 0));
    let mut short = vec![0u8; 3];
    assert!(run.model.save_state(&run.s, 0, &mut short).is_err());
    same(
        &run.step(&[prefill(0, 20, &toks[20..])]),
        &whole[20..],
        "the source",
    );
}

/// Writes the generated models into LARCH_INKLING_TINY_OUT for llama.cpp's dumps (`tests/data/inkling`):
///   LARCH_INKLING_TINY_OUT=<dir> cargo test -p larch-model --test inkling -- --ignored write_tiny
/// then, with `tools/llama-logits` built at b12723 (`-DLLAMA_TAG=b12723`), for each model `<m>` (`tiny`,
/// `tiny-nvfp4`): `llama-logits <dir>/<m>.gguf prompts.jsonl --threads 1 > <m>.b12723.cpu.jsonl`, the same
/// with `--fa off` into `<m>.b12723.cpu-fa0.jsonl` and with `--batch 7 --ubatch 7 --threads 4` into
/// `<m>.b12723.cpu-ub7.jsonl`.
#[test]
#[ignore = "writes the generated GGUFs for llama.cpp's dumps: LARCH_INKLING_TINY_OUT"]
fn write_tiny_ggufs() {
    let dir = PathBuf::from(std::env::var("LARCH_INKLING_TINY_OUT").unwrap());
    std::fs::write(dir.join("tiny.gguf"), tiny::inkling_gguf(SEED)).unwrap();
    std::fs::write(dir.join("tiny-nvfp4.gguf"), tiny::inkling_gguf_nvfp4(SEED)).unwrap();
}

/// m15 (or another `inkling2_loop` GGUF) against its llama-logits dump, by the gate of
/// [`prefill_matches_llama_cpp`] with the calibrated bound and margin given (oracle-calibrate's), each prompt
/// one prefill in 64-row chunks from a reset state. LARCH_ORACLE_DEVICE (an ordinal) runs it on a GPU; the CPU
/// reference takes hours on m15. LARCH_ORACLE_PROMPTS caps the prompts.
///   LARCH_ORACLE_GGUF=<gguf> LARCH_ORACLE_DUMP=<dump.jsonl> LARCH_ORACLE_BOUND=<bound>
///   LARCH_ORACLE_MARGIN=<nats> [LARCH_ORACLE_DEVICE=<ordinal>]
///   cargo test --release -p larch-model --test inkling -- --ignored --nocapture gguf_prefill
#[test]
#[ignore = "needs an inkling2_loop GGUF and its llama-logits dump: LARCH_ORACLE_GGUF and LARCH_ORACLE_DUMP"]
fn gguf_prefill_matches_llama_cpp() {
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"));
    let gguf = PathBuf::from(env("LARCH_ORACLE_GGUF"));
    let f = File::open(env("LARCH_ORACLE_DUMP")).unwrap();
    let mut d = Dump::read(BufReader::new(f), &oracle::sha256(&gguf).unwrap()).unwrap();
    if let Ok(n) = std::env::var("LARCH_ORACLE_PROMPTS") {
        d.prompts.truncate(n.parse().unwrap());
    }
    let file = File::open(&gguf).unwrap();
    let g = Gguf::read(BufReader::new(&file), file.metadata().unwrap().len()).unwrap();
    assert_eq!(
        d.tokcheck(&Tokenizer::from_gguf(&g).unwrap()),
        Ok(d.prompts.len())
    );
    let ordinal = std::env::var("LARCH_ORACLE_DEVICE").map_or(CPU, |o| o.parse().unwrap());
    let longest = d.prompts.iter().map(|p| p.tokens.len()).max().unwrap_or(1) as i64;
    let mut run = Run::on(&gguf, ordinal, longest, 1);
    let rows: Vec<Vec<Top>> = d
        .prompts
        .iter()
        .map(|pr| {
            let toks: Vec<i32> = pr.tokens.iter().map(|&t| t as i32).collect();
            run.sequence(0, &toks, ROWS as usize)
                .iter()
                .map(|l| top64(l))
                .collect()
        })
        .collect();
    let (bound, margin): (f64, f64) = (
        env("LARCH_ORACLE_BOUND").parse().unwrap(),
        env("LARCH_ORACLE_MARGIN").parse().unwrap(),
    );
    let r = compare(&d, &rows, margin).unwrap();
    let a = &r.all;
    eprintln!(
        "{} positions, top-1 {:.4}, {} decisive at margin {margin} with top-1 {:.4}, KL mean {:.3e} p99 {:.3e} \
         max {:.3e}; bound {bound:.3e}",
        a.positions,
        a.top1_rate(),
        a.decisive,
        a.decisive_top1_rate(),
        a.mean_kl,
        a.p99_kl,
        a.max_kl
    );
    assert!(r.passes(bound), "misses llama.cpp's gate");
}
