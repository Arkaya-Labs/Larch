//! Check chunks, every arch: each row is a one-row decode, bit for bit, on the CPU reference and (an
//! ignored test) on a real GGUF and device.
//!
//! Qwen 3.5 (`qwen35`, `qwen35moe`) on the CPU reference: the generated hybrid models give the same
//! logits however a sequence's rows are split into steps (one prefill, chunks, decode steps, beside
//! another sequence), the recurrent state is per slot, resets and copies, and a hybrid model refuses
//! a step without its state. `write_tiny_ggufs` writes the generated models with a byte vocabulary
//! llama.cpp loads, for tests/oracle.rs' `model_prefill_matches_llama_cpp`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use larch_kernels::{
    Buffer, CPU, CPU_COMPUTE, DType, Device, Graph, KError, KvPool, Stream, Tensor,
};
use larch_model::gguf::{Gguf, Value, write};
use larch_model::{
    Arch, ExpertPlan, Model, ModelConfig, MoeCfg, Placement, StateBatch, StepBatch, Taps,
    Transformer, tiny,
};
use larch_records::LoadError;

/// Rows a step may hold, and positions per sequence.
const ROWS: u32 = 64;
const BLOCK: i64 = 16;
const CTX: i64 = 512;

/// One sequence's part of a step: its slot (state and KV), the position of its first row, its
/// tokens; `decode` makes it a decode row (one token), `pad` a padding decode row (no KV write,
/// nothing attended) whose recurrent state still runs in `slot`, as the engine pads.
struct Part {
    slot: i32,
    start: i32,
    tokens: Vec<i32>,
    decode: bool,
    pad: bool,
}

impl Part {
    fn prefill(slot: i32, start: i32, tokens: &[i32]) -> Part {
        let tokens = tokens.to_vec();
        Part {
            slot,
            start,
            tokens,
            decode: false,
            pad: false,
        }
    }

    fn decode(slot: i32, start: i32, token: i32) -> Part {
        Part {
            slot,
            start,
            tokens: vec![token],
            decode: true,
            pad: false,
        }
    }
}

/// A model on one device with a KV pool of `CTX` positions per slot and room for one step's inputs.
struct Run {
    s: Stream,
    model: Transformer,
    k: Buffer,
    v: Buffer,
    input: Buffer,
    out: Buffer,
    slots: u32,
    /// The steps' [`StepBatch::check`] and `attn_splits`.
    check: bool,
    splits: u32,
    /// Each row's four M-RoPE positions from its position, when the steps pass `[rows, 4]`.
    planes: Option<fn(i32) -> [i32; 4]>,
    /// Added to every row's position, not to its KV index: rows apart from their index.
    shift: i32,
}

impl Run {
    fn new(path: &Path, dev: &Device, slots: u32) -> Run {
        Run::placed(path, dev, slots, Placement::default())
    }

    /// [`Run::new`] with the weights where `place` puts them.
    fn placed(path: &Path, dev: &Device, slots: u32, place: Placement) -> Run {
        let s = Stream::new(dev).unwrap();
        let model = Transformer::load_placed(path, dev, &s, ROWS, place).unwrap();
        Run::on(s, model, dev, slots)
    }

    /// [`Run::placed`] with the token embedding in host memory exactly when `host_embed`.
    fn embedded(path: &Path, dev: &Device, slots: u32, place: Placement, host_embed: bool) -> Run {
        let s = Stream::new(dev).unwrap();
        let model =
            Transformer::load_planned(path, dev, &s, ROWS, place, None, host_embed).unwrap();
        Run::on(s, model, dev, slots)
    }

    /// [`Run::new`] with each layer's experts where `plan` puts them.
    fn cached(path: &Path, dev: &Device, slots: u32, plan: &ExpertPlan) -> Run {
        let s = Stream::new(dev).unwrap();
        let model = Transformer::load_cached(path, dev, &s, ROWS, plan).unwrap();
        Run::on(s, model, dev, slots)
    }

    /// The loaded `model`'s state, KV pool and inputs.
    fn on(s: Stream, mut model: Transformer, dev: &Device, slots: u32) -> Run {
        if model.config().arch.hybrid() {
            model.alloc_state(&s, slots).unwrap();
            model.alloc_checks(&s, Transformer::CHECK_SLOTS).unwrap();
            assert_eq!(model.state_slots(), slots);
        }
        let c = model.config().clone();
        let kv = DType::F16.bytes(
            u64::from(c.n_layers * c.n_kv_heads * c.head_dim) * (u64::from(slots) * CTX as u64),
        );
        let blocks = (CTX / BLOCK) as usize * slots as usize;
        let input = Buffer::alloc(dev, ((11 * ROWS as usize + blocks + 8) * 4) as u64).unwrap();
        let out = Buffer::alloc(dev, u64::from(ROWS * c.vocab) * 4).unwrap();
        Run {
            k: Buffer::alloc(dev, kv).unwrap(),
            v: Buffer::alloc(dev, kv).unwrap(),
            s,
            model,
            input,
            out,
            slots,
            check: false,
            splits: 0,
            planes: None,
            shift: 0,
        }
    }

    /// One step over `parts` (decode parts first): every row's logits, in row order.
    fn step(&mut self, parts: &[Part]) -> Vec<Vec<f32>> {
        self.step_with(parts, true).unwrap()
    }

    /// [`Run::step`] through `forward_state`, or through `forward` (no state) when `state` is false.
    fn step_with(&mut self, parts: &[Part], state: bool) -> Result<Vec<Vec<f32>>, KError> {
        self.step_by(parts, |m, s, b, st, kv, l| match state {
            true => m.forward_state(s, b, st, kv, l, None),
            false => m.forward(s, b, kv, l),
        })
    }

    /// [`Run::step`] with `run` in place of the forward pass: a graph's capture or replay.
    fn step_by(
        &mut self,
        parts: &[Part],
        run: impl FnOnce(
            &mut Transformer,
            &Stream,
            &StepBatch,
            &StateBatch,
            &KvPool,
            &Tensor,
        ) -> Result<(), KError>,
    ) -> Result<Vec<Vec<f32>>, KError> {
        let per = CTX / BLOCK;
        let (dec, pre): (Vec<&Part>, Vec<&Part>) = parts.iter().partition(|p| p.decode);
        assert!(dec.iter().all(|p| p.tokens.len() == 1));
        let seqs: Vec<&Part> = dec.iter().chain(&pre).copied().collect();
        let tokens: Vec<i32> = seqs.iter().flat_map(|p| p.tokens.clone()).collect();
        let pos: Vec<i32> = seqs
            .iter()
            .flat_map(|p| (p.start..).take(p.tokens.len()))
            .collect();
        let pos: Vec<i32> = pos.into_iter().map(|p| p + self.shift).collect();
        let (pos, pos_shape) = match self.planes {
            Some(f) => {
                let n = pos.len() as i64;
                (pos.into_iter().flat_map(f).collect(), vec![n, 4])
            }
            None => (pos.clone(), vec![pos.len() as i64]),
        };
        let slot_ids: Vec<i32> = seqs
            .iter()
            .flat_map(|p| {
                (p.start..)
                    .take(p.tokens.len())
                    .map(|q| if p.pad { -1 } else { p.slot * CTX as i32 + q })
            })
            .collect();
        let table: Vec<i32> = seqs
            .iter()
            .flat_map(|p| (0..per as i32).map(move |b| p.slot * per as i32 + b))
            .collect();
        let rows = tokens.len() as i64;
        let all: Vec<i32> = (0..rows as i32).collect();
        let lens: Vec<i32> = dec
            .iter()
            .map(|p| if p.pad { 0 } else { p.start + 1 })
            .collect();
        let starts: Vec<i32> = pre.iter().map(|p| p.start).collect();
        let q_lens: Vec<i32> = pre.iter().map(|p| p.tokens.len() as i32).collect();
        let slots: Vec<i32> = seqs.iter().map(|p| p.slot).collect();
        let state_lens: Vec<i32> = seqs.iter().map(|p| p.tokens.len() as i32).collect();
        let n = seqs.len() as i64;
        let parts: [(&[i32], Vec<i64>); 11] = [
            (&tokens, vec![rows]),
            (&pos, pos_shape),
            (&slot_ids, vec![rows]),
            (&table, vec![n, per]),
            (&lens, vec![lens.len() as i64]),
            (&starts, vec![starts.len() as i64]),
            (&q_lens, vec![q_lens.len() as i64]),
            (&all, vec![rows]),
            (&slots, vec![n]),
            (&state_lens, vec![n]),
            (&[], vec![0]),
        ];
        let mut at = 0;
        let mut t = Vec::new();
        for (data, shape) in &parts {
            self.s.upload(&self.input, at, data).unwrap();
            t.push(Tensor::new(&self.input, at, DType::I32, shape).unwrap());
            at += data.len() as u64 * 4;
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
            attn_splits: self.splits,
            check: self.check,
        };
        let batch_state = StateBatch {
            slots: t[8],
            q_lens: t[9],
        };
        let c = self.model.config().clone();
        let pool = KvPool {
            k: &self.k,
            v: &self.v,
            n_blocks: (per * i64::from(self.slots)) as u32,
            block_tokens: BLOCK as u32,
            n_layers: c.n_layers,
            n_kv_heads: c.n_kv_heads,
            head_dim: c.head_dim,
            dtype: DType::F16,
        };
        let vocab = c.vocab as usize;
        let logits = Tensor::new(&self.out, 0, DType::F32, &[rows, vocab as i64]).unwrap();
        // The engine's host step before each pass, eager or replayed: a host embedding's rows.
        self.model.embed_tokens(&self.s, &tokens)?;
        run(
            &mut self.model,
            &self.s,
            &batch,
            &batch_state,
            &pool,
            &logits,
        )?;
        let mut l = vec![0f32; rows as usize * vocab];
        self.s.download(&self.out, 0, &mut l).unwrap();
        Ok(l.chunks(vocab).map(<[f32]>::to_vec).collect())
    }

    /// [`Run::sequence`] in a reset `slot`, its rows' four positions from `planes`.
    fn planes_at(
        &mut self,
        slot: i32,
        tokens: &[i32],
        planes: fn(i32) -> [i32; 4],
        chunk: usize,
    ) -> Vec<Vec<f32>> {
        self.model.reset_state(&self.s, slot as u32).unwrap();
        self.planes = Some(planes);
        let out = self.sequence(slot, tokens, chunk);
        self.planes = None;
        out
    }

    /// `tokens` from position 0 in slot `slot`, `chunk` rows per prefill step, or one decode step
    /// per token with `chunk` 0: every position's logits.
    fn sequence(&mut self, slot: i32, tokens: &[i32], chunk: usize) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        let step = chunk.max(1);
        for (i, c) in tokens.chunks(step).enumerate() {
            let start = (i * step) as i32;
            out.extend(self.step(&[if chunk == 0 {
                Part::decode(slot, start, c[0])
            } else {
                Part::prefill(slot, start, c)
            }]));
        }
        out
    }
}

fn model_file(arch: Arch, name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("larch-{name}-{}.gguf", std::process::id()));
    std::fs::write(&path, tiny::gguf(arch, 7)).unwrap();
    path
}

fn tokens(n: usize, seed: i32) -> Vec<i32> {
    (0..n as i32)
        .map(|i| (i * 37 + seed * 11 + 5) % tiny::VOCAB as i32)
        .collect()
}

/// `a` and `b` are the same logits, bit for bit, and not trivially so.
fn same(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    assert!(
        a.iter().flatten().all(|x| x.is_finite()),
        "{what}: not finite"
    );
    assert!(a.iter().flatten().any(|&x| x != 0.0), "{what}: all zero");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(x == y, "{what}: row {i} differs");
    }
}

/// The fused layer entries (larch.h ABI 4) give the unfused ops' logits bit for bit on both CPU devices: every
/// arch's generated model over a prefill, decode steps and a step of two decode rows, all fused against none, and
/// the MoE models again with their first layer's experts in host memory. Every entry refused
/// ([`Transformer::refuse_fused`]) gives them too: the fallbacks a refusing backend (the CUDA H2 route) takes.
#[test]
fn fused_entries_give_the_unfused_logits() {
    let archs = [
        Arch::Qwen35Moe,
        Arch::Qwen35,
        Arch::Qwen3Moe,
        Arch::Qwen3,
        Arch::Qwen2,
        Arch::Llama,
    ];
    for arch in archs {
        let path = model_file(arch, &format!("fused-{arch:?}"));
        let moe = matches!(arch, Arch::Qwen35Moe | Arch::Qwen3Moe);
        let hosts: &[u32] = if moe { &[0, 1] } else { &[0] };
        let runs = [CPU, CPU_COMPUTE].map(|o| hosts.iter().map(move |&h| (o, h)));
        for (ordinal, host) in runs.into_iter().flatten() {
            let dev = Device::open(ordinal).unwrap();
            let place = Placement {
                cpu_expert_layers: host,
                ..Placement::default()
            };
            let logits = |fused, refuse| {
                let s = Stream::new(&dev).unwrap();
                let mut model = Transformer::load_placed(&path, &dev, &s, ROWS, place).unwrap();
                model.set_fused(Some(fused));
                model.refuse_fused(refuse);
                let mut run = Run::on(s, model, &dev, 2);
                let toks = tokens(6, 3);
                let mut out = run.sequence(0, &toks[..3], 3);
                for (i, &t) in toks[3..5].iter().enumerate() {
                    out.extend(run.step(&[Part::decode(0, 3 + i as i32, t)]));
                }
                out.extend(run.step(&[Part::decode(0, 5, toks[5]), Part::decode(1, 0, toks[0])]));
                out
            };
            let what = format!("{arch:?} on {ordinal}, {host} host layers");
            let unfused = logits(false, false);
            same(&logits(true, false), &unfused, &what);
            same(
                &logits(true, true),
                &unfused,
                &format!("{what}, every entry refused"),
            );
        }
        std::fs::remove_file(path).unwrap();
    }
}

/// The hybrid MoE with its token embedding in host memory computes what the device's does, bit
/// for bit: a prefill, chunks and decode steps of two sequences, beside a host-expert layer.
#[test]
fn a_host_embedding_runs_the_hybrid_as_the_device_one() {
    let path = model_file(Arch::Qwen35Moe, "q35-embd");
    let dev = Device::open(CPU).unwrap();
    for layers in [0, 1] {
        let place = Placement {
            cpu_expert_layers: layers,
            ..Default::default()
        };
        let run = |host: bool| {
            let mut r = Run::embedded(&path, &dev, 2, place, host);
            assert_eq!(r.model.embed_and_head().is_none(), host);
            let mut got = r.sequence(0, &tokens(20, 1), 7);
            got.extend(r.sequence(1, &tokens(5, 2), 0));
            got.extend(r.step(&[Part::decode(0, 20, 9), Part::prefill(1, 5, &tokens(3, 3))]));
            got
        };
        same(&run(true), &run(false), &format!("{layers} host layers"));
    }
    std::fs::remove_file(&path).unwrap();
}

/// Every split of a sequence into steps gives the same logits, bit for bit: one prefill, chunks,
/// decode steps, and prefill then decode, in either arch. A step over 16 rows multiplies by f16
/// weights (dequant + gemm) and one of up to 16 rows by the quantized weights (mmvq), which round
/// differently, as in every model: so each family is compared within itself, where a split changes
/// only which step a row is in. Both families against llama.cpp is tests/oracle.rs'.
#[test]
fn every_split_of_a_sequence_agrees() {
    for (arch, name) in [
        (Arch::Qwen35, "q35-split"),
        (Arch::Qwen35Moe, "q35moe-split"),
    ] {
        let path = model_file(arch, name);
        let dev = Device::open(CPU).unwrap();
        let mut run = Run::new(&path, &dev, 6);
        std::fs::remove_file(&path).unwrap();
        let t = tokens(40, 1);
        // dequant + gemm: 40 rows in one step, then in two.
        let whole = run.sequence(0, &t, 40);
        same(&whole, &run.sequence(1, &t, 20), &format!("{name} halves"));
        // mmvq: steps of 16 rows, of 5, one decode row each, and 16 rows of prefill then decode.
        let sixteens = run.sequence(2, &t, 16);
        same(&sixteens, &run.sequence(3, &t, 5), &format!("{name} fives"));
        same(
            &sixteens,
            &run.sequence(4, &t, 0),
            &format!("{name} decode"),
        );
        let mut mixed = run.sequence(5, &t[..16], 16);
        for (i, &tok) in t.iter().enumerate().skip(16) {
            mixed.extend(run.step(&[Part::decode(5, i as i32, tok)]));
        }
        same(&sixteens, &mixed, &format!("{name} prefill then decode"));
    }
}

/// Two sequences in one step, one decoding beside a padding row and one prefilling, each in its own
/// slot, give what each gives alone; a reset slot starts again from position 0. Every step holds at
/// most 16 rows, so all of them multiply alike (see `every_split_of_a_sequence_agrees`).
#[test]
fn slots_are_independent_and_reset() {
    let path = model_file(Arch::Qwen35Moe, "q35-slots");
    let dev = Device::open(CPU).unwrap();
    let mut run = Run::new(&path, &dev, 4);
    std::fs::remove_file(&path).unwrap();
    let (a, b) = (tokens(16, 2), tokens(12, 3));
    let alone_a = run.sequence(0, &a, 16);
    let alone_b = run.sequence(1, &b, 12);
    // Slot 2 decodes a[..10]; then one step holds its next token, a padding row on slot 3, and all
    // of b in slot 1, reset.
    run.model.reset_state(&run.s, 1).unwrap();
    let mut got_a = run.sequence(2, &a[..10], 0);
    let pad = Part {
        pad: true,
        ..Part::decode(3, 0, 0)
    };
    let both = run.step(&[Part::decode(2, 10, a[10]), pad, Part::prefill(1, 0, &b)]);
    got_a.push(both[0].clone());
    same(&alone_a[..11], &got_a, "a beside b");
    same(&alone_b, &both[2..], "b beside a");
    // Without the reset, slot 1's state runs on from b and the same tokens predict otherwise.
    let again = run.sequence(1, &b, 12);
    let moved = again
        .iter()
        .zip(&alone_b)
        .any(|(x, y)| x.iter().zip(y).any(|(p, q)| (p - q).abs() > 1e-3));
    assert!(moved, "a state that was not reset changes nothing");
    run.model.reset_state(&run.s, 1).unwrap();
    same(&alone_b, &run.sequence(1, &b, 12), "b after a reset");
}

/// A slot copied from another runs on exactly as its source, which the copy leaves as it was; a slot
/// out of range is refused.
#[test]
fn a_copied_slot_runs_on_as_its_source() {
    let path = model_file(Arch::Qwen35Moe, "q35-copy");
    let dev = Device::open(CPU).unwrap();
    let mut run = Run::new(&path, &dev, 2);
    std::fs::remove_file(&path).unwrap();
    let t = tokens(16, 4);
    let alone = run.sequence(0, &t, 16);
    // Both slots hold t[..10]'s KV; slot 1's state is zeroed, then copied from slot 0's.
    run.model.reset_state(&run.s, 0).unwrap();
    run.sequence(0, &t[..10], 10);
    run.sequence(1, &t[..10], 10);
    run.model.reset_state(&run.s, 1).unwrap();
    let state = |run: &Run, slot| run.model.state_bytes(&run.s, slot).unwrap();
    assert!(
        state(&run, 1).iter().all(|&b| b == 0),
        "a reset slot reads zero"
    );
    run.model.copy_slot(&run.s, 0, 1).unwrap();
    assert!(state(&run, 0).iter().any(|&b| b != 0));
    assert!(
        state(&run, 1) == state(&run, 0),
        "the copy reads as its source"
    );
    assert_eq!(run.model.state_bytes(&run.s, 2), Err(KError::InvalidArg));
    // A saved state written back reads as it was; a wrong length writes nothing.
    let saved = state(&run, 0);
    run.model.reset_state(&run.s, 1).unwrap();
    run.model.set_state_bytes(&run.s, 1, &saved).unwrap();
    assert!(state(&run, 1) == saved, "the written state reads back");
    let short = &saved[1..];
    assert_eq!(
        run.model.set_state_bytes(&run.s, 1, short),
        Err(KError::InvalidArg)
    );
    assert_eq!(
        run.model
            .read_state(&run.s, 1, &mut vec![0; saved.len() + 1]),
        Err(KError::InvalidArg)
    );
    assert!(state(&run, 1) == saved, "a refused write leaves the slot");
    same(
        &alone[10..],
        &run.step(&[Part::prefill(1, 10, &t[10..])]),
        "the copy",
    );
    same(
        &alone[10..],
        &run.step(&[Part::prefill(0, 10, &t[10..])]),
        "the source",
    );
    assert_eq!(run.model.copy_slot(&run.s, 0, 2), Err(KError::InvalidArg));
    assert_eq!(run.model.copy_slot(&run.s, 2, 0), Err(KError::InvalidArg));
}

/// Growing the state between steps keeps every slot as it stood, a rewound one living in a spare
/// too: the steps after it match a run that never grew, bit for bit. The new slots read zero; a
/// grow stops at the room and is refused while a check is begun.
#[test]
fn a_grown_state_runs_on_as_before() {
    let path = model_file(Arch::Qwen35Moe, "q35-grow");
    let dev = Device::open(CPU).unwrap();
    let (mut a, mut b) = (Run::new(&path, &dev, 2), Run::new(&path, &dev, 2));
    std::fs::remove_file(&path).unwrap();
    let (p, g, q) = (tokens(10, 11), tokens(4, 12), tokens(6, 13));
    for run in [&mut a, &mut b] {
        run.sequence(0, &p, 10);
        run.sequence(1, &q[..4], 0);
        // Slot 0 keeps 2 of 4 rows: it lives in a spare until a step moves it home.
        check(run, 10, &g, 2, &[]);
    }
    let state = |run: &Run, slot| run.model.state_bytes(&run.s, slot).unwrap();
    let before = [state(&a, 0), state(&a, 1)];
    let slot = before[0].len() as u64;
    let spares = u64::from(Transformer::CHECK_SLOTS);
    // The new buffers hold the spares too: a byte short of a third slot grows nothing, room for
    // it grows 3 of the 6 asked.
    let grow = |run: &mut Run, want, room| run.model.grow_state(&run.s, want, room);
    assert_eq!(grow(&mut a, 6, (spares + 3) * slot - 1), Ok(2));
    assert_eq!(grow(&mut a, 6, (spares + 3) * slot), Ok(3));
    assert_eq!(grow(&mut a, 2, u64::MAX), Ok(3));
    assert_eq!(a.model.state_slots(), 3);
    assert_eq!([state(&a, 0), state(&a, 1)], before);
    assert!(state(&a, 2).iter().all(|&x| x == 0));
    a.model.begin_check(&a.s, 1, 2).unwrap();
    assert_eq!(grow(&mut a, 4, u64::MAX), Err(KError::InvalidArg));
    a.model.end_check(&a.s, 1, 1).unwrap_err();
    for i in 0..2 {
        let parts = [
            Part::decode(0, 12 + i, q[4 + i as usize]),
            Part::decode(1, 4 + i, q[4 + i as usize]),
        ];
        same(&b.step(&parts), &a.step(&parts), "after the grow");
    }
    // A check after the grow still finds its spares.
    same(
        &check(&mut b, 14, &g[..3], 1, &[]),
        &check(&mut a, 14, &g[..3], 1, &[]),
        "a check after the grow",
    );
}

/// A hybrid model refuses a step without its state that it runs with it, and a slot out of range to
/// reset; an attention model has no state slots.
#[test]
fn a_hybrid_step_needs_its_state() {
    let path = model_file(Arch::Qwen35, "q35-nostate");
    let dev = Device::open(CPU).unwrap();
    let mut run = Run::new(&path, &dev, 1);
    std::fs::remove_file(&path).unwrap();
    assert_eq!(
        run.model.reset_state(&run.s, 1).err(),
        Some(KError::InvalidArg)
    );
    let part = || [Part::prefill(0, 0, &[5, 9, 2])];
    assert_eq!(
        run.step_with(&part(), false).err(),
        Some(KError::InvalidArg)
    );
    assert!(run.step_with(&part(), true).is_ok());

    let path = model_file(Arch::Qwen3, "q3-nostate");
    let s = Stream::new(&dev).unwrap();
    let mut dense = Transformer::load(&path, &dev, &s, ROWS).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(dense.state_slots(), 0);
    assert!(matches!(
        dense.alloc_state(&s, 1),
        Err(LoadError::UnsupportedArch)
    ));
}

/// A check of `rows` rows on slot 0 (from position `at`, beside `beside`) rewound to `kept`.
fn check(run: &mut Run, at: i32, guess: &[i32], kept: usize, beside: &[Part]) -> Vec<Vec<f32>> {
    let rows = guess.len() as u32;
    run.model.begin_check(&run.s, 0, rows).unwrap();
    let mut parts: Vec<Part> = beside
        .iter()
        .map(|p| Part {
            tokens: p.tokens.clone(),
            ..*p
        })
        .collect();
    parts.push(Part::prefill(0, at, guess));
    run.check = true;
    let out = run.step(&parts);
    run.check = false;
    run.model.end_check(&run.s, 0, kept as u32).unwrap();
    out
}

/// A check step rewound to its first `kept` rows (every `kept` of every length up to 8) leaves the
/// state, recurrent and conv, as those rows alone would: the logits after it match a run that never
/// saw the other rows, bit for bit, through a second check on the rewound slot and decode steps
/// that move it home.
#[test]
fn a_check_rewinds_to_its_kept_rows() {
    let path = model_file(Arch::Qwen35Moe, "q35-check");
    let dev = Device::open(CPU).unwrap();
    let mut run = Run::new(&path, &dev, 2);
    std::fs::remove_file(&path).unwrap();
    let (p, g, next) = (tokens(10, 5), tokens(8, 6), tokens(5, 7));
    for rows in 1..=8 {
        for kept in 1..=rows {
            let what = format!("{rows} rows, {kept} kept");
            // Slot 1 runs only what slot 0 keeps: the prompt, the kept guesses, then the next
            // tokens, 3 of them as one chunk (the second check keeps 2), then decode rows.
            let at = p.len() as i32 + kept as i32;
            run.model.reset_state(&run.s, 1).unwrap();
            run.sequence(1, &p, 10);
            let mut want = run.step(&[Part::prefill(1, 10, &g[..kept])]);
            want.extend(run.step(&[Part::prefill(1, at, &next[..2])]));
            for (i, &t) in next.iter().enumerate().skip(2) {
                want.extend(run.step(&[Part::decode(1, at + i as i32, t)]));
            }
            run.model.reset_state(&run.s, 0).unwrap();
            run.sequence(0, &p, 10);
            let mut got = check(&mut run, 10, &g[..rows], kept, &[]);
            got.truncate(kept);
            got.extend(check(&mut run, at, &next[..3], 2, &[])[..2].to_vec());
            for (i, &t) in next.iter().enumerate().skip(2) {
                got.extend(run.step(&[Part::decode(0, at + i as i32, t)]));
            }
            same(&want, &got, &what);
        }
    }
}

/// A check beside another sequence's decode rewinds only its own slot; the other runs as alone.
#[test]
fn a_check_beside_another_sequence() {
    let path = model_file(Arch::Qwen35, "q35-check2");
    let dev = Device::open(CPU).unwrap();
    let mut run = Run::new(&path, &dev, 2);
    std::fs::remove_file(&path).unwrap();
    let (p, g, b) = (tokens(10, 8), tokens(4, 9), tokens(8, 10));
    let alone_b = run.sequence(1, &b, 0);
    run.model.reset_state(&run.s, 0).unwrap();
    let want: Vec<_> = run.sequence(0, &[&p[..], &g[..2], &b[..3]].concat(), 0)[10..].to_vec();
    run.model.reset_state(&run.s, 0).unwrap();
    run.model.reset_state(&run.s, 1).unwrap();
    run.sequence(0, &p, 0);
    run.sequence(1, &b[..4], 0);
    let one = check(&mut run, 10, &g, 2, &[Part::decode(1, 4, b[4])]);
    let mut got = one[1..3].to_vec();
    let mut got_b = vec![one[0].clone()];
    for i in 0..3 {
        let both = run.step(&[
            Part::decode(1, 5 + i, b[5 + i as usize]),
            Part::decode(0, 12 + i, b[i as usize]),
        ]);
        got_b.push(both[0].clone());
        got.push(both[1].clone());
    }
    same(&want, &got, "the checked sequence");
    same(&alone_b[4..], &got_b, "the one beside it");
}

/// Check steps give each row, bit for bit, the logits of decoding it alone at the same attention
/// splits: two chunks of 10 rows in one step (past every one-row kernel's row count), then a chunk
/// beside another sequence's decode row. After the same prompt prefill in three slots.
fn check_chunks_match_decode(run: &mut Run, what: &str) {
    let (p, g) = (tokens(8, 11), tokens(20, 12));
    run.splits = 2;
    for slot in 0..3 {
        if run.model.state_slots() > 0 {
            run.model.reset_state(&run.s, slot).unwrap();
        }
        run.step(&[Part::prefill(slot as i32, 0, &p)]);
    }
    let want: Vec<_> = (0..20)
        .flat_map(|i| run.step(&[Part::decode(0, 8 + i as i32, g[i])]))
        .collect();
    run.check = true;
    let two = run.step(&[Part::prefill(1, 8, &g[..10]), Part::prefill(2, 8, &g[..10])]);
    let beside = run.step(&[Part::decode(2, 18, g[10]), Part::prefill(1, 18, &g[10..])]);
    run.check = false;
    same(
        &want[..10],
        &two[..10],
        &format!("{what}: first of two chunks"),
    );
    same(
        &want[..10],
        &two[10..],
        &format!("{what}: second of two chunks"),
    );
    same(
        &want[10..11],
        &beside[..1],
        &format!("{what}: decode row beside"),
    );
    same(&want[10..], &beside[1..], &format!("{what}: chunk beside"));
}

#[test]
fn a_check_chunk_is_decode_bit_for_bit() {
    for arch in [
        Arch::Llama,
        Arch::Qwen2,
        Arch::Qwen3,
        Arch::Qwen3Moe,
        Arch::Qwen35,
        Arch::Qwen35Moe,
    ] {
        let what = format!("{arch:?}");
        let path = model_file(arch, &format!("check-{what}"));
        let dev = Device::open(CPU).unwrap();
        let mut run = Run::new(&path, &dev, 3);
        std::fs::remove_file(&path).unwrap();
        check_chunks_match_decode(&mut run, &what);
    }
}

/// The same for the MoE archs on the optimized CPU device, which also runs a GPU model's host
/// experts: its mmvq_id groups a call's pairs by expert from 4 per expert used, which a 10-row chunk
/// over 4 experts reaches and a one-row decode never does.
#[test]
fn a_moe_check_chunk_is_decode_bit_for_bit_on_cpu_compute() {
    for arch in [Arch::Qwen3Moe, Arch::Qwen35Moe] {
        let what = format!("{arch:?} on CPU_COMPUTE");
        let path = model_file(arch, &format!("check-fast-{arch:?}"));
        let dev = Device::open(CPU_COMPUTE).unwrap();
        let mut run = Run::new(&path, &dev, 3);
        std::fs::remove_file(&path).unwrap();
        check_chunks_match_decode(&mut run, &what);
    }
}

/// A check step on a model with a pool stays on the CPU path: the bits of a model without one,
/// while the prefill-only step before it streams.
#[test]
fn a_check_step_does_not_stream() {
    let path = model_file(Arch::Qwen3Moe, "stream-check");
    let dev = Device::open(CPU).unwrap();
    let place = |stream_rows| Placement {
        cpu_expert_layers: tiny::LAYERS,
        stream_rows,
    };
    let (mut streamed, mut cpu) = (
        Run::placed(&path, &dev, 3, place(1)),
        Run::placed(&path, &dev, 3, place(0)),
    );
    std::fs::remove_file(&path).unwrap();
    assert!(streamed.model.alloc_stream_pool(&streamed.s).unwrap());
    let (p, g) = (tokens(8, 11), tokens(10, 12));
    streamed.step(&[Part::prefill(0, 0, &p)]);
    assert_eq!(streamed.model.streamed_steps(), 1, "a prefill step streams");
    streamed.check = true;
    cpu.check = true;
    same(
        &cpu.step(&[Part::prefill(1, 0, &g)]),
        &streamed.step(&[Part::prefill(1, 0, &g)]),
        "a check step",
    );
    assert_eq!(
        streamed.model.streamed_steps(),
        1,
        "the check step streamed"
    );
}

/// The same with every layer split on the CPU reference: a check step's device and host parts are
/// a one-row decode's, whichever side an expert is on.
#[test]
fn a_split_check_chunk_is_decode_bit_for_bit() {
    for arch in [Arch::Qwen3Moe, Arch::Qwen35Moe] {
        let what = format!("{arch:?} split");
        let path = model_file(arch, &format!("check-split-{arch:?}"));
        let dev = Device::open(CPU).unwrap();
        let plan = partial_plan(&path, 0.5);
        let mut run = Run::cached(&path, &dev, 3, &plan);
        std::fs::remove_file(&path).unwrap();
        check_chunks_match_decode(&mut run, &what);
        assert!(run.model.expert_hits().is_some_and(|(h, m)| h > 0 && m > 0));
    }
}

/// `check_chunks_match_decode` on a real GGUF and device (a Mac's Metal is device 0), with the
/// routed experts of the first LARCH_CHECK_HOST_LAYERS layers (default 0) on CPU_COMPUTE, as a
/// MoE too big for the device runs:
///   LARCH_CHECK_GGUF=<model.gguf> cargo test --release -p larch-model --test qwen35 -- --ignored --nocapture a_check_chunk_on
#[test]
#[ignore = "needs a GGUF: LARCH_CHECK_GGUF, LARCH_CHECK_DEVICE (default 0), LARCH_CHECK_HOST_LAYERS (default 0)"]
fn a_check_chunk_on_a_device() {
    let path = PathBuf::from(std::env::var("LARCH_CHECK_GGUF").unwrap());
    let device = std::env::var("LARCH_CHECK_DEVICE").map_or(0, |d| d.parse().unwrap());
    let dev = Device::open(device).unwrap();
    let host = std::env::var("LARCH_CHECK_HOST_LAYERS").map_or(0, |n| n.parse().unwrap());
    let place = Placement {
        cpu_expert_layers: host,
        ..Default::default()
    };
    let mut run = Run::placed(&path, &dev, 3, place);
    check_chunks_match_decode(&mut run, &path.display().to_string());
}

/// A decode step captured in pieces, cut at each host-expert layer, replays (piece, host experts by
/// tag on the live rows, ..., last piece) as the eager step of the same device shape, bit for bit:
/// over 4 decode steps with new tokens, the live rows' logits, every KV byte and the live
/// sequences' recurrent state, then 2 eager steps after them. Twice: two sequences in a bucket of
/// 2, and three beside a padding row in a bucket of 4. A device kernel may compute a row by the
/// batch's row count, so each eager step runs the replay's rows, padding included; its host
/// sections then run the padding row too, which a replay skips. That changes no live row: at a
/// decode's few rows CPU_COMPUTE's mmvq_id runs each (row, expert) pair on its own (it groups by
/// expert only from 4 pairs per expert used, which 4 rows reach only by all choosing the same
/// experts). A hybrid runs `forward_state`, any other model `forward`, as the engine does:
/// With LARCH_CHECK_RESIDENT (a fraction, 0 to 1) every layer is split instead, that part of its
/// experts resident: a split layer's device part runs in its piece, so step-dependent routes replay
/// as eager too:
///   LARCH_CHECK_GGUF=<model.gguf> LARCH_CHECK_HOST_LAYERS=<n> cargo test --release
///   -p larch-model --test qwen35 -- --ignored --nocapture a_host_expert_step_replays
#[test]
#[ignore = "needs a GGUF and a device that captures graphs: LARCH_CHECK_GGUF, LARCH_CHECK_DEVICE (default 0), LARCH_CHECK_HOST_LAYERS (default 0) or LARCH_CHECK_RESIDENT (a fraction)"]
fn a_host_expert_step_replays_as_eager() {
    let path = PathBuf::from(std::env::var("LARCH_CHECK_GGUF").unwrap());
    let dev = check_device();
    let fraction = check_fraction();
    let (mut run, host) = check_run(&path, &dev, fraction);
    for (live, pad) in [(2, false), (3, true)] {
        let (eager, bytes) = replay_phase(&mut run, None, live, pad, None, &[]);
        let (replayed, bytes_replayed) = replay_phase(&mut run, Some(host), live, pad, None, &[]);
        let what = format!("{}, {live} live rows, padded {pad}", path.display());
        same(&eager, &replayed, &what);
        assert!(bytes == bytes_replayed, "{what}: KV or state differs");
    }
    drop(run);
    // This device captured the plan above, so a None here is the model's refusal: a dynamic
    // plan's uploads map in the next pass, which a replay never makes.
    if let Some(fraction) = fraction {
        let plan = ExpertPlan {
            dynamic: true,
            ..partial_plan(&path, fraction)
        };
        let mut run = Run::cached(&path, &dev, 4, &plan);
        let hybrid = run.model.config().arch.hybrid();
        run.step_by(&[Part::decode(0, 0, 1)], |m, s, b, st, kv, l| {
            let captured = Graph::capture_pieces(s, |s| match hybrid {
                true => m.forward_state(s, b, st, kv, l, None),
                false => m.forward(s, b, kv, l),
            })?;
            assert!(captured.is_none(), "a dynamic plan captured");
            Ok(())
        })
        .unwrap();
    }
}

/// A step that taps what a drafter reads, target layers or only the final hidden state, replays
/// as it ran eagerly, the same env and shapes as [`a_host_expert_step_replays_as_eager`]: logits,
/// KV, state and the live rows' tap bytes, in a bucket of 3 live rows and a padding row too (blocks
/// at `k * (live + pad)`):
///   LARCH_CHECK_GGUF=<model.gguf> cargo test --release -p larch-model --test qwen35 --
///   --ignored --nocapture a_tapped_step_replays
#[test]
#[ignore = "needs a GGUF and a device that captures graphs: LARCH_CHECK_GGUF, LARCH_CHECK_DEVICE (default 0), LARCH_CHECK_HOST_LAYERS (default 0) or LARCH_CHECK_RESIDENT (a fraction)"]
fn a_tapped_step_replays_as_eager() {
    let path = PathBuf::from(std::env::var("LARCH_CHECK_GGUF").unwrap());
    let dev = check_device();
    let (mut run, host) = check_run(&path, &dev, check_fraction());
    let c = run.model.config().clone();
    let layers: Vec<u16> = (0..c.n_layers as u16)
        .step_by(c.n_layers.div_ceil(3) as usize)
        .collect();
    for (layers, final_state) in [(layers, false), (Vec::new(), true)] {
        let tap = Tap::new(&dev, &run.s, layers, final_state, c.d_model);
        for (live, pad) in [(2, false), (3, true)] {
            let (eager, bytes) = replay_phase(&mut run, None, live, pad, Some(&tap), &[]);
            let (replayed, replayed_bytes) =
                replay_phase(&mut run, Some(host), live, pad, Some(&tap), &[]);
            let what = format!(
                "{}, {live} live rows, padded {pad}, final state {final_state}",
                path.display()
            );
            same(&eager, &replayed, &what);
            assert!(bytes == replayed_bytes, "{what}: KV, state or taps differ");
        }
    }
}

/// A check step (decode rows, then chunks checked at decode's splits) replays as it ran eagerly: its
/// rows' logits, KV and state, over 4 steps with new tokens, then 2 eager decode steps. Per-chunk
/// lengths are data: two chunks swap their lengths each step on the same capture. A hybrid has one
/// check a step. The same env as [`a_host_expert_step_replays_as_eager`]:
///   LARCH_CHECK_GGUF=<model.gguf> LARCH_CHECK_HOST_LAYERS=<n> cargo test --release
///   -p larch-model --test qwen35 -- --ignored --nocapture a_check_step_replays
#[test]
#[ignore = "needs a GGUF and a device that captures graphs: LARCH_CHECK_GGUF, LARCH_CHECK_DEVICE (default 0), LARCH_CHECK_HOST_LAYERS (default 0) or LARCH_CHECK_RESIDENT (a fraction)"]
fn a_check_step_replays_as_eager() {
    let path = PathBuf::from(std::env::var("LARCH_CHECK_GGUF").unwrap());
    let dev = check_device();
    let (mut run, host) = check_run(&path, &dev, check_fraction());
    run.splits = 16;
    let mut shapes: Vec<(usize, &[usize])> = vec![(0, &[5]), (2, &[5])];
    if !run.model.config().arch.hybrid() {
        shapes.push((1, &[3, 5]));
    }
    for (live, chunks) in shapes {
        let (eager, bytes) = replay_phase(&mut run, None, live, false, None, chunks);
        let (replayed, replayed_bytes) =
            replay_phase(&mut run, Some(host), live, false, None, chunks);
        let what = format!("{}, {live} decode rows, chunks {chunks:?}", path.display());
        same(&eager, &replayed, &what);
        assert!(bytes == replayed_bytes, "{what}: KV or state differs");
    }
}

/// A guessing hybrid replays as it runs eager: the live rows' logits and the state after every
/// step, and the KV pools, through checks beside decode rows and alone, rewinds that leave a slot
/// in a spare and steps that move it home, spares reused by a later check, and a padded bucket whose
/// padding row sits on a moved free slot.
#[test]
#[ignore = "needs a Qwen 3.5 GGUF and a device that captures graphs: LARCH_CHECK_GGUF, LARCH_CHECK_DEVICE (default 0), LARCH_CHECK_HOST_LAYERS (default 0) or LARCH_CHECK_RESIDENT (a fraction)"]
fn a_guessing_hybrid_replays_as_eager() {
    let path = PathBuf::from(std::env::var("LARCH_CHECK_GGUF").unwrap());
    let dev = check_device();
    let (mut run, _) = check_run(&path, &dev, check_fraction());
    assert!(run.model.config().arch.hybrid());
    run.splits = 16;
    let (eager, bytes) = rewound_phase(&mut run, false);
    let (replayed, replayed_bytes) = rewound_phase(&mut run, true);
    let what = path.display().to_string();
    same(&eager, &replayed, &what);
    assert!(bytes == replayed_bytes, "{what}: KV or state differs");
}

/// One step of [`rewound_phase`]: decode slots, a check as (slot, rows, kept), a padding row's slot.
type Rewound = (&'static [i32], Option<(i32, usize, usize)>, Option<i32>);

/// Three prefilled sequences through [`Rewound`] steps, eager or (each shape captured on its first
/// step) replayed, then 2 eager decode steps: the live rows' logits, then the state of the slots
/// still held after each step and the K and V pools. Slot 2 ends unheld, under the padding row.
fn rewound_phase(run: &mut Run, graphs: bool) -> (Vec<Vec<f32>>, Vec<u8>) {
    const STEPS: [Rewound; 9] = [
        (&[0, 1], None, None),
        (&[1], Some((0, 3, 2)), None),
        (&[0, 1], None, None),
        (&[0], Some((1, 3, 1)), None),
        (&[0, 1], None, None),
        (&[], Some((2, 3, 2)), None),
        (&[0, 1], None, Some(2)),
        (&[0, 1], None, Some(2)),
        (&[0, 1], None, None),
    ];
    let zero = vec![0u8; run.k.bytes() as usize];
    run.s.upload(&run.k, 0, &zero).unwrap();
    run.s.upload(&run.v, 0, &zero).unwrap();
    for slot in 0..4 {
        run.model.reset_state(&run.s, slot).unwrap();
    }
    let prompts = [tokens(9, 5), tokens(7, 6), tokens(5, 8)];
    let mut logits = Vec::new();
    for (slot, p) in prompts.iter().enumerate() {
        logits.extend(run.step(&[Part::prefill(slot as i32, 0, p)]));
    }
    let mut at: [i32; 3] = std::array::from_fn(|s| prompts[s].len() as i32);
    let (next, guess) = (tokens(64, 7), tokens(64, 13));
    let token = |slot: i32, at: i32| next[(at + slot) as usize % 64];
    let mut cache: HashMap<Vec<(bool, usize)>, Pieces> = HashMap::new();
    let mut bytes = Vec::new();
    let mut held = 3;
    for (decode, check, pad) in STEPS {
        let mut parts: Vec<Part> = decode
            .iter()
            .map(|&q| Part::decode(q, at[q as usize], token(q, at[q as usize])))
            .collect();
        if let Some((slot, rows, _)) = check {
            let from = at[slot as usize];
            parts.push(Part::prefill(
                slot,
                from,
                &guess[from as usize..from as usize + rows],
            ));
            run.model
                .begin_check(&run.s, slot as u32, rows as u32)
                .unwrap();
        }
        let live: usize = parts.iter().map(|p| p.tokens.len()).sum();
        if let Some(slot) = pad {
            parts.push(Part {
                pad: true,
                ..Part::decode(slot, 0, 0)
            });
            held = 2;
        }
        run.check = check.is_some();
        let got = run
            .step_by(&parts, |m, s, b, st, kv, l| {
                if !graphs {
                    return forward(m, s, b, st, kv, l, true, None);
                }
                let shape: Vec<_> = parts.iter().map(|p| (p.decode, p.tokens.len())).collect();
                if !cache.contains_key(&shape) {
                    let captured =
                        Graph::capture_pieces(s, |s| forward(m, s, b, st, kv, l, true, None))?;
                    let pieces = captured.expect("the device refused the capture");
                    cache.insert(shape.clone(), pieces);
                }
                replay(m, s, &parts, &cache[&shape], live as u32)
            })
            .unwrap();
        run.check = false;
        // The padding row's logits are never sampled, and a replay leaves its experts stale.
        logits.extend(got.into_iter().take(live));
        for &q in decode {
            at[q as usize] += 1;
        }
        if let Some((slot, _, kept)) = check {
            run.model
                .end_check(&run.s, slot as u32, kept as u32)
                .unwrap();
            at[slot as usize] += kept as i32;
        }
        for slot in 0..held {
            bytes.extend(run.model.state_bytes(&run.s, slot).unwrap());
        }
    }
    let mut k = zero.clone();
    run.s.download(&run.k, 0, &mut k).unwrap();
    let mut v = zero;
    run.s.download(&run.v, 0, &mut v).unwrap();
    bytes.extend(k);
    bytes.extend(v);
    for q in [0, 1] {
        logits.extend(run.step(&[Part::decode(q, at[q as usize], token(q, at[q as usize]))]));
    }
    (logits, bytes)
}

/// [`a_host_expert_step_replays_as_eager`]'s device.
fn check_device() -> Device {
    let device = std::env::var("LARCH_CHECK_DEVICE").map_or(0, |d| d.parse().unwrap());
    Device::open(device).unwrap()
}

/// LARCH_CHECK_RESIDENT.
fn check_fraction() -> Option<f64> {
    std::env::var("LARCH_CHECK_RESIDENT")
        .ok()
        .map(|f| f.parse::<f64>().unwrap())
}

/// A 4-slot run of the GGUF, split by `fraction` or with LARCH_CHECK_HOST_LAYERS on the host, and
/// the host layers it cut at.
fn check_run(path: &Path, dev: &Device, fraction: Option<f64>) -> (Run, u32) {
    match fraction {
        Some(fraction) => {
            let plan = partial_plan(path, fraction);
            let layers = plan.resident.len() as u32;
            (Run::cached(path, dev, 4, &plan), layers)
        }
        None => {
            let host = std::env::var("LARCH_CHECK_HOST_LAYERS").map_or(0, |n| n.parse().unwrap());
            let place = Placement {
                cpu_expert_layers: host,
                ..Default::default()
            };
            (Run::placed(path, dev, 4, place), host)
        }
    }
}

/// A graph's pieces, as [`Graph::capture_pieces`] returns them.
type Pieces = (Vec<(Graph, u64)>, Graph);

/// One replay as the engine runs it: a hybrid's slot table for `parts` first (a guessing load's
/// capture reads it), then each piece and its host step for the step's `live` rows.
fn replay(
    m: &mut Transformer,
    s: &Stream,
    parts: &[Part],
    pieces: &Pieces,
    live: u32,
) -> Result<(), KError> {
    if m.config().arch.hybrid() {
        let (dec, pre): (Vec<&Part>, Vec<&Part>) = parts.iter().partition(|p| p.decode);
        let (slots, lens): (Vec<i32>, Vec<i32>) = dec
            .iter()
            .chain(&pre)
            .map(|p| (p.slot, p.tokens.len() as i32))
            .unzip();
        m.route_table(s, &slots, &lens)?;
    }
    for (g, tag) in &pieces.0 {
        g.launch(s)?;
        m.run_host(s, *tag, live)?;
    }
    pieces.1.launch(s)
}

/// A plan that splits every layer of the GGUF at `path`, its first `fraction` of experts resident
/// (at least one, never all).
fn partial_plan(path: &Path, fraction: f64) -> ExpertPlan {
    let file = std::fs::File::open(path).unwrap();
    let len = file.metadata().unwrap().len();
    let g = Gguf::read(std::io::BufReader::new(file), len).unwrap();
    let c = ModelConfig::from_gguf(&g).unwrap();
    let m = MoeCfg::from_gguf(&g, c.arch).unwrap().expect("a MoE GGUF");
    let n = ((fraction * f64::from(m.experts)) as u32).clamp(1, m.experts - 1);
    ExpertPlan {
        resident: vec![(0..n).collect(); c.n_layers as usize],
        dynamic: false,
    }
}

/// `live` sequences (slots from 0) from a zeroed KV pool and state: prefills, 4 decode steps (with
/// `pad`, beside a padding row on slot 3; with `graphs`, the host layers, in graph pieces captured
/// on the first; with `tap`, every row tapped), then 2 eager decode steps of the live rows. The live
/// rows' logits, and the K and V pools, the live slots' state after the 4 steps and the live rows'
/// taps of each of the 4. With `chunks`, a check step: after the decode rows, one sequence per chunk
/// (slots from `live`), whose lengths rotate step by step; the eager steps then decode them too.
fn replay_phase(
    run: &mut Run,
    graphs: Option<u32>,
    live: usize,
    pad: bool,
    tap: Option<&Tap>,
    chunks: &[usize],
) -> (Vec<Vec<f32>>, Vec<u8>) {
    let hybrid = run.model.config().arch.hybrid();
    let zero = vec![0u8; run.k.bytes() as usize];
    run.s.upload(&run.k, 0, &zero).unwrap();
    run.s.upload(&run.v, 0, &zero).unwrap();
    if hybrid {
        for slot in 0..4 {
            run.model.reset_state(&run.s, slot).unwrap();
        }
    }
    let prompts = &[tokens(9, 5), tokens(7, 6), tokens(5, 8)][..live + chunks.len()];
    let mut logits = Vec::new();
    for (slot, p) in prompts.iter().enumerate() {
        let part = Part::prefill(slot as i32, 0, p);
        logits.extend(run.step_with(&[part], hybrid).unwrap());
    }
    let next = tokens(6, 7);
    let guess = tokens(64, 13);
    // The chunk length of chunk `c` at step `i`, and where it starts.
    let len = |c: usize, i: usize| chunks[(c + i) % chunks.len()];
    let start =
        |c: usize, i: usize| prompts[live + c].len() + (0..i).map(|j| len(c, j)).sum::<usize>();
    let step = |i: usize, pad: bool| {
        let mut parts: Vec<Part> = (0..live)
            .map(|q| Part::decode(q as i32, (prompts[q].len() + i) as i32, next[(i + q) % 6]))
            .collect();
        for c in 0..chunks.len() {
            let (slot, at) = ((live + c) as i32, start(c, i));
            parts.push(if i < 4 {
                Part::prefill(slot, at as i32, &guess[at..at + len(c, i)])
            } else {
                Part::decode(slot, at as i32, guess[at])
            });
        }
        if pad {
            parts.push(Part {
                pad: true,
                ..Part::decode(3, 0, 0)
            });
        }
        parts
    };
    let mut pieces = None;
    let live_rows = live + chunks.iter().sum::<usize>();
    let rows = live_rows + usize::from(pad);
    let taps = tap.map(|t| t.taps(rows));
    let taps = taps.as_ref();
    let mut tapped = Vec::new();
    run.check = !chunks.is_empty();
    for i in 0..4 {
        let got = match graphs {
            None => run.step_by(&step(i, pad), |m, s, b, st, kv, l| {
                forward(m, s, b, st, kv, l, hybrid, taps)
            }),
            Some(host) => {
                let parts = step(i, pad);
                run.step_by(&parts, |m, s, b, st, kv, l| {
                    if pieces.is_none() {
                        let captured = Graph::capture_pieces(s, |s| {
                            forward(m, s, b, st, kv, l, hybrid, taps)
                        })?;
                        pieces = Some(captured.expect("the device refused the capture"));
                    }
                    let got = pieces.as_ref().unwrap();
                    let layers = m.config().n_layers;
                    assert_eq!(got.0.len() as u32, host.min(layers), "a cut per host layer");
                    replay(m, s, &parts, got, live_rows as u32)
                })
            }
        };
        // The padding row's logits are never sampled, and a replay leaves its experts stale.
        logits.extend(got.unwrap().into_iter().take(live_rows));
        if let Some(t) = tap {
            t.live_bytes(&run.s, rows, live_rows, &mut tapped);
        }
    }
    run.check = false;
    let mut bytes = zero.clone();
    run.s.download(&run.k, 0, &mut bytes).unwrap();
    let mut v = zero;
    run.s.download(&run.v, 0, &mut v).unwrap();
    bytes.extend(v);
    if hybrid {
        for slot in 0..(live + chunks.len()) as u32 {
            bytes.extend(run.model.state_bytes(&run.s, slot).unwrap());
        }
    }
    bytes.extend(tapped);
    for i in 4..6 {
        logits.extend(run.step_with(&step(i, false), hybrid).unwrap());
    }
    (logits, bytes)
}

/// One forward as the engine runs it: a hybrid through `forward_state`, others through `forward_taps`.
#[allow(clippy::too_many_arguments)]
fn forward(
    m: &mut Transformer,
    s: &Stream,
    b: &StepBatch,
    st: &StateBatch,
    kv: &KvPool,
    l: &Tensor,
    hybrid: bool,
    taps: Option<&Taps>,
) -> Result<(), KError> {
    match hybrid {
        true => m.forward_state(s, b, st, kv, l, taps),
        false => m.forward_taps(s, b, kv, l, taps),
    }
}

/// A drafter's taps of every row of a step, in buffers the test owns: [`Taps::out`]'s layout.
struct Tap {
    layers: Vec<u16>,
    final_state: bool,
    iota: Buffer,
    out: Buffer,
    d: usize,
}

impl Tap {
    fn new(dev: &Device, s: &Stream, layers: Vec<u16>, final_state: bool, d_model: u32) -> Tap {
        let d = d_model as usize;
        let iota = Buffer::alloc(dev, 4 * u64::from(ROWS)).unwrap();
        s.upload(&iota, 0, &(0..ROWS as i32).collect::<Vec<_>>())
            .unwrap();
        let blocks = layers.len() + usize::from(final_state);
        let out = Buffer::alloc(dev, (4 * blocks * ROWS as usize * d) as u64).unwrap();
        Tap {
            layers,
            final_state,
            iota,
            out,
            d,
        }
    }

    /// `out`'s blocks.
    fn blocks(&self) -> usize {
        self.layers.len() + usize::from(self.final_state)
    }

    /// The taps of a step of `rows` rows.
    fn taps(&self, rows: usize) -> Taps<'_> {
        let (n, k, d) = (rows as i64, self.blocks() as i64, self.d as i64);
        Taps {
            layers: &self.layers,
            final_state: self.final_state,
            rows: Tensor::new(&self.iota, 0, DType::I32, &[n]).unwrap(),
            out: Tensor::new(&self.out, 0, DType::F32, &[k * n, d]).unwrap(),
        }
    }

    /// Appends to `to` the first `live` rows of every block of a step of `rows` rows.
    fn live_bytes(&self, s: &Stream, rows: usize, live: usize, to: &mut Vec<u8>) {
        let row = 4 * self.d;
        let mut all = vec![0u8; self.blocks() * rows * row];
        s.download(&self.out, 0, &mut all).unwrap();
        for block in all.chunks(rows * row) {
            to.extend_from_slice(&block[..live * row]);
        }
    }
}

/// What a check refuses: more rows than spares, a second begin, an end without its step or with
/// `kept` out of range; a model without recurrent state has no checks.
#[test]
fn a_check_refuses_what_it_cannot_rewind() {
    let path = model_file(Arch::Qwen35, "q35-check3");
    let dev = Device::open(CPU).unwrap();
    let mut run = Run::new(&path, &dev, 1);
    std::fs::remove_file(&path).unwrap();
    let (m, s) = (&mut run.model, &run.s);
    let spares = Transformer::CHECK_SLOTS;
    assert_eq!(m.begin_check(s, 0, spares + 2), Err(KError::Unsupported));
    assert_eq!(m.begin_check(s, 1, 2), Err(KError::InvalidArg));
    assert_eq!(m.begin_check(s, 0, 0), Err(KError::InvalidArg));
    assert_eq!(m.end_check(s, 0, 1), Err(KError::InvalidArg));
    m.begin_check(s, 0, 3).unwrap();
    assert_eq!(m.begin_check(s, 0, 3), Err(KError::InvalidArg));
    assert_eq!(m.end_check(s, 0, 4), Err(KError::InvalidArg));
    assert_eq!(m.end_check(s, 0, 2), Err(KError::InvalidArg), "no step ran");
    // A step that is not the begun check's shape is refused before it runs.
    m.begin_check(s, 0, 3).unwrap();
    assert_eq!(
        run.step_with(&[Part::prefill(0, 0, &[1, 2])], true).err(),
        Some(KError::InvalidArg)
    );
    run.model.end_check(&run.s, 0, 1).unwrap_err();

    let path = model_file(Arch::Qwen3, "q3-check");
    let mut dense = Transformer::load(&path, &dev, &run.s, ROWS).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(dense.begin_check(&run.s, 0, 2), Err(KError::Unsupported));
}

/// What a hybrid load allocates beside the weights, for the plan before it: the scratch of its
/// one Gated DeltaNet layer and full-attention Q gate for `ROWS` rows, 3 slots of state, the f16
/// copies of its float alpha and its shared-expert gates, and the dequant copy of the gated Q.
/// A second recurrent layer adds 3 slots; a header whose state passes u64 is refused. The check
/// spares count apart.
#[test]
fn hybrid_bytes_counts_what_the_load_takes() {
    use tiny::{D, GDN_HEAD_DIM as GD, GDN_K_HEADS, GDN_KERNEL, GDN_V_HEADS, HEAD_DIM, HEADS};
    // The config comes from the tiny header before `keys`: with both of its 2 layers recurrent it
    // keeps no KV layer, which ModelConfig refuses; hybrid_bytes reads the layout from the header.
    let header = |arch, keys: &[(&str, u32)]| {
        let b = tiny::gguf(arch, 7);
        let mut g = Gguf::read(std::io::Cursor::new(&b), b.len() as u64).unwrap();
        let c = larch_model::ModelConfig::from_gguf(&g).unwrap();
        for &(k, v) in keys {
            g.kv.insert(format!("qwen35moe.{k}"), Value::U32(v));
        }
        (g, c)
    };
    let bytes = |arch, keys: &[(&str, u32)], slots| {
        let (g, c) = header(arch, keys);
        Transformer::hybrid_bytes(&g, &c, ROWS, slots)
    };
    let (rows, ch, vd, qd) = (
        u64::from(ROWS),
        (2 * GDN_K_HEADS + GDN_V_HEADS) * GD,
        GDN_V_HEADS * GD,
        HEADS * HEAD_DIM,
    );
    // f16 qkv, conv, z, o, gated, q pairs, gates and the shared gate; f32 alpha and beta; the
    // 0/1 selectors.
    let scratch = 2 * rows * (2 * ch + 3 * vd + 2 * qd + qd + 1)
        + 4 * rows * 2 * GDN_V_HEADS
        + 2 * (2 * HEAD_DIM) * (2 * HEAD_DIM);
    let slot = 2 * (GDN_KERNEL - 1) * ch + 4 * GDN_V_HEADS * GD * GD;
    // Layer 0's f32 alpha (its beta is quantized) and both layers' shared-expert gates.
    let side = 2 * GDN_V_HEADS * D + 2 * 2 * D;
    let dq = 2 * D * ((2 * qd).max(ch) - qd.max(D));
    // The 3 slots; the route of a step's slots and snapshots.
    let slots = 3;
    let route = 4 * 2 * rows;
    let one = bytes(Arch::Qwen35Moe, &[], 3).unwrap();
    assert_eq!(one, scratch + slots * slot + side + dq + route);
    let two = bytes(Arch::Qwen35Moe, &[("full_attention_interval", 3)], 3).unwrap();
    assert_eq!(two - one, slots * slot);
    assert_eq!(bytes(Arch::Qwen3Moe, &[], 3).unwrap(), 0);
    let spares = |arch| {
        let (g, c) = header(arch, &[]);
        Transformer::check_bytes(&g, &c, 4).unwrap()
    };
    assert_eq!(spares(Arch::Qwen35Moe), 4 * slot);
    assert_eq!(spares(Arch::Qwen3Moe), 0);
    let huge = [
        ("ssm.group_count", 1),
        ("ssm.time_step_rank", 1),
        ("ssm.state_size", 1 << 31),
        ("ssm.inner_size", 1 << 31),
    ];
    assert!(matches!(
        bytes(Arch::Qwen35Moe, &huge, 3),
        Err(LoadError::BadGguf(..))
    ));
}

/// Some row of `a` and `b` differs.
fn differ(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    assert!(
        a.iter().zip(b).any(|(x, y)| x != y),
        "{what}: no row differs"
    );
}

/// Qwen 3.5's attention layers take four positions a row (M-RoPE, interleaved sections): a text
/// row's (p, p, p, 0) is the one position p bit for bit, in prefill and decode, in either arch; the
/// t, h and w planes each turn some pairs (the e plane, with no section, none); and a sequence whose
/// rows sit at positions apart from their KV index (as the text after an image does) is the same
/// sequence at those positions.
#[test]
fn mrope_positions_rope_as_llama_cpp_lays_them_out() {
    for (arch, name) in [
        (Arch::Qwen35, "q35-mrope"),
        (Arch::Qwen35Moe, "q35moe-mrope"),
    ] {
        let path = model_file(arch, name);
        let dev = Device::open(CPU).unwrap();
        let mut run = Run::new(&path, &dev, 6);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(run.model.config().rope.sections, Some([6, 5, 5, 0]));
        let t = tokens(20, 4);
        let text = (run.sequence(0, &t, 20), run.sequence(1, &t, 0));
        let four: fn(i32) -> [i32; 4] = |p| [p, p, p, 0];
        same(
            &text.0,
            &run.planes_at(2, &t, four, 20),
            &format!("{name} prefill"),
        );
        same(
            &text.1,
            &run.planes_at(3, &t, four, 0),
            &format!("{name} decode"),
        );
        same(
            &text.0,
            &run.planes_at(4, &t, |p| [p, p, p, 77], 20),
            &format!("{name} e"),
        );
        for (plane, f) in [
            ("t", (|p| [p + 3, p, p, 0]) as fn(i32) -> [i32; 4]),
            ("h", |p| [p, p + 3, p, 0]),
            ("w", |p| [p, p, p + 3, 0]),
        ] {
            differ(
                &text.0,
                &run.planes_at(5, &t, f, 20),
                &format!("{name} {plane}"),
            );
        }
        // Positions 7.. on KV slots 0..: rope turns as the positions say, attention reads the KV
        // index; the same rows at one position each and at (p, p, p, 0) agree.
        run.shift = 7;
        run.model.reset_state(&run.s, 2).unwrap();
        let one = run.sequence(2, &t, 20);
        let four_shifted = run.planes_at(3, &t, four, 20);
        run.shift = 0;
        same(&one, &four_shifted, &format!("{name} shifted"));
        differ(&text.0, &one, &format!("{name} shifted from text"));
    }
}

/// Four positions a row are refused before any work by a model without rope sections, and in any
/// other shape by one with them; the refused step leaves the recurrent state as it was.
#[test]
fn four_positions_need_rope_sections() {
    let dev = Device::open(CPU).unwrap();
    let path = model_file(Arch::Qwen3, "q3-mrope");
    let mut run = Run::new(&path, &dev, 2);
    std::fs::remove_file(&path).unwrap();
    assert_eq!(run.model.config().rope.sections, None);
    run.planes = Some(|p| [p, p, p, 0]);
    let t = tokens(8, 5);
    assert_eq!(
        run.step_with(&[Part::prefill(0, 0, &t)], false).err(),
        Some(KError::InvalidArg)
    );
    let path = model_file(Arch::Qwen35, "q35-mrope-shape");
    let mut run = Run::new(&path, &dev, 2);
    std::fs::remove_file(&path).unwrap();
    let alone = run.sequence(0, &t, 8);
    run.planes = Some(|p| [p, p, p, 0]);
    let refused = run.step_by(&[Part::prefill(1, 0, &t)], |m, s, b, st, kv, l| {
        let bad = StepBatch {
            positions: b.positions.reshape(&[b.token_ids.shape()[0] * 2, 2])?,
            ..*b
        };
        m.forward_state(s, &bad, st, kv, l, None)
    });
    assert_eq!(refused.err(), Some(KError::InvalidArg));
    run.planes = None;
    same(&alone, &run.sequence(1, &t, 8), "after a refused step");
}

/// `rope.dimension_sections`: read for Qwen 3.5, at most four counts with some t, h or w pairs, no
/// more pairs than the rope turns.
#[test]
fn rope_sections_are_checked() {
    let cfg = |v: Value| {
        let b = tiny::gguf(Arch::Qwen35, 7);
        let mut g = Gguf::read(std::io::Cursor::new(&b), b.len() as u64).unwrap();
        g.kv.insert("qwen35.rope.dimension_sections".into(), v);
        ModelConfig::from_gguf(&g)
    };
    let arr = |v: &[i32]| Value::Arr(v.iter().map(|&x| Value::I32(x)).collect());
    assert_eq!(cfg(arr(&[8, 8])).unwrap().rope.sections, Some([8, 8, 0, 0]));
    assert_eq!(
        cfg(arr(&[6, 5, 5, 0])).unwrap().rope.sections,
        Some([6, 5, 5, 0])
    );
    let key = Some(LoadError::BadGguf(larch_model::gguf::code::KEY, 0));
    for bad in [
        arr(&[0, 0, 0, 4]),
        arr(&[6, 5, 6, 0]),
        arr(&[4, 4, 4, 4, 0]),
        arr(&[-1, 5, 5, 0]),
        Value::U32(16),
    ] {
        assert_eq!(cfg(bad.clone()).err(), key, "{bad:?}");
    }
}

/// GPT-2's byte-to-unicode map: printable bytes stand for themselves, the rest for 256 + n.
fn byte_token(b: u8) -> String {
    let keep = |b: u8| (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
    let c = if keep(b) {
        u32::from(b)
    } else {
        256 + (0..b).filter(|&x| !keep(x)).count() as u32
    };
    char::from_u32(c).unwrap().to_string()
}

/// The generated qwen35 and qwen35moe models, with a 256-token byte vocabulary (one unused merge) so that
/// llama.cpp loads them, into LARCH_QWEN35_TINY_OUT; then dump each with llama-logits on CPU and
/// run tests/oracle.rs' `model_prefill_matches_llama_cpp` on it.
///   LARCH_QWEN35_TINY_OUT=<dir> cargo test -p larch-model --test qwen35 -- --ignored write_tiny
#[test]
#[ignore = "writes GGUFs for a llama.cpp dump: LARCH_QWEN35_TINY_OUT"]
fn write_tiny_ggufs() {
    let dir = PathBuf::from(std::env::var("LARCH_QWEN35_TINY_OUT").unwrap());
    // qwen3 and qwen3moe: the attention models' agreement on the same vocabulary, for scale.
    let archs = [
        (Arch::Qwen35, "qwen35"),
        (Arch::Qwen35Moe, "qwen35moe"),
        (Arch::Qwen3, "qwen3"),
        (Arch::Qwen3Moe, "qwen3moe"),
    ];
    for (arch, name) in archs {
        let bytes = tiny::gguf(arch, 7);
        let g = Gguf::parse(&bytes).unwrap();
        let mut kv: Vec<(String, Value)> = g.kv.clone().into_iter().collect();
        let tokens = (0..=255u8).map(|b| Value::Str(byte_token(b))).collect();
        kv.extend([
            ("tokenizer.ggml.model".into(), Value::Str("gpt2".into())),
            ("tokenizer.ggml.pre".into(), Value::Str("qwen2".into())),
            ("tokenizer.ggml.tokens".into(), Value::Arr(tokens)),
            (
                "tokenizer.ggml.token_type".into(),
                Value::Arr(vec![Value::I32(1); 256]),
            ),
            // One merge of two control bytes no prompt holds: llama.cpp refuses an empty list.
            (
                "tokenizer.ggml.merges".into(),
                Value::Arr(vec![Value::Str(format!(
                    "{} {}",
                    byte_token(1),
                    byte_token(2)
                ))]),
            ),
            ("tokenizer.ggml.eos_token_id".into(), Value::U32(10)),
            ("tokenizer.ggml.add_bos_token".into(), Value::Bool(false)),
        ]);
        let tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = g
            .tensors
            .iter()
            .map(|t| {
                let at = (g.data_offset + t.offset) as usize;
                let data = bytes[at..at + t.bytes as usize].to_vec();
                (t.name.clone(), t.dims.clone(), t.ggml_type, data)
            })
            .collect();
        std::fs::write(dir.join(format!("tiny-{name}.gguf")), write(&kv, &tensors)).unwrap();
    }
}
