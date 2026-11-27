//! The tiny model on the CPU reference: the GGUF reader on good and broken files, the loader, and the
//! ways a prompt can be fed (one prefill, one token per step, chunks beside decode rows and padding)
//! giving the same logits, and the forward taps. This checks the paths against each other only;
//! correctness is D16's comparison with llama.cpp, which needs `larch parity`. One ignored test
//! checks a real model's layer-0 tap on the GPU against the CPU reference, and the GPU's final state
//! against the CPU reference's norm of the GPU's own last-layer tap.

use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};

use larch_kernels::{
    Buffer, CPU, CPU_COMPUTE, DType, Device, Graph, KError, KvPool, Stream, Tensor, ops,
};
use larch_model::gguf::{self, Gguf, Value, code};
use larch_model::learn::{DeviceSuffix, LearnError};
use larch_model::{
    Arch, ExpertPlan, Histogram, MMVQ_MAX_ROWS, Model, ModelConfig, MoeCfg, Placement,
    RouterDeltas, StepBatch, Taps, Transformer, tiny,
};
use larch_records::LoadError;

const MAX_ROWS: u32 = MMVQ_MAX_ROWS as u32 + 1;

const ARCHS: [Arch; 4] = [Arch::Llama, Arch::Qwen2, Arch::Qwen3, Arch::Qwen3Moe];

/// The KV pool pages only the attention layers: every layer of the other archs, the one attention
/// layer of the tiny Qwen 3.5 models (the other is Gated DeltaNet).
#[test]
fn kv_layers_are_the_attention_layers() {
    for arch in [Arch::Llama, Arch::Qwen3Moe, Arch::Qwen35, Arch::Qwen35Moe] {
        let c = ModelConfig::from_gguf(&Gguf::parse(&tiny::gguf(arch, 1)).unwrap()).unwrap();
        let attention = if arch.hybrid() { 1 } else { tiny::LAYERS };
        assert_eq!(
            (c.n_layers, c.kv_layers),
            (tiny::LAYERS, attention),
            "{arch:?}"
        );
    }
}

#[test]
fn gguf_reader() {
    for arch in ARCHS {
        let bytes = tiny::gguf(arch, 1);
        let g = Gguf::parse(&bytes).unwrap();
        let c = ModelConfig::from_gguf(&g).unwrap();
        assert_eq!(
            (c.arch, c.n_layers, c.vocab, c.head_dim),
            (arch, tiny::LAYERS, tiny::VOCAB, tiny::HEAD_DIM as u32)
        );
        assert_eq!(
            (c.qkv_bias, c.qk_norm, c.rope.neox),
            (
                arch == Arch::Qwen2,
                matches!(arch, Arch::Qwen3 | Arch::Qwen3Moe),
                arch != Arch::Llama
            )
        );

        // Every truncation is a clean error, and corrupted header bytes never panic.
        let header = g.data_offset as usize + 64;
        for n in 0..header {
            assert!(Gguf::parse(&bytes[..n]).is_err(), "prefix {n} parsed");
        }
        let mut seed = 7u64;
        for _ in 0..2000 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut b = bytes[..header].to_vec();
            b[(seed >> 33) as usize % header] ^= (seed >> 17) as u8 | 1;
            let _ = Gguf::parse(&b);
        }

        // A tensor offset near u64::MAX (out of reach of the single-byte flips above).
        let name = b"output.weight";
        let at = bytes.windows(name.len()).position(|w| w == name).unwrap() + name.len() + 24;
        let mut b = bytes.clone();
        b[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(Gguf::parse(&b).is_err());
    }
}

/// Loads `bytes` onto the CPU reference, with room beyond the largest mmvq crossover.
fn load(bytes: &[u8], name: &str) -> Result<(Stream, Transformer), LoadError> {
    load_placed(bytes, name, Placement::default())
}

/// [`load`] with the weights where `place` puts them.
fn load_placed(
    bytes: &[u8],
    name: &str,
    place: Placement,
) -> Result<(Stream, Transformer), LoadError> {
    load_by(bytes, name, |p, d, s| {
        Transformer::load_placed(p, d, s, MAX_ROWS, place)
    })
}

/// [`load`] with each layer's experts where `plan` puts them.
fn load_cached(
    bytes: &[u8],
    name: &str,
    plan: &ExpertPlan,
) -> Result<(Stream, Transformer), LoadError> {
    load_by(bytes, name, |p, d, s| {
        Transformer::load_cached(p, d, s, MAX_ROWS, plan)
    })
}

/// `bytes` written to a file, loaded by `f` onto the CPU reference.
fn load_by(
    bytes: &[u8],
    name: &str,
    f: impl FnOnce(&std::path::Path, &Device, &Stream) -> Result<Transformer, LoadError>,
) -> Result<(Stream, Transformer), LoadError> {
    load_by_on(bytes, name, CPU, f)
}

fn load_by_on(
    bytes: &[u8],
    name: &str,
    backend: i32,
    f: impl FnOnce(&std::path::Path, &Device, &Stream) -> Result<Transformer, LoadError>,
) -> Result<(Stream, Transformer), LoadError> {
    // Unique per call: parallel tests may load files of the same name.
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("larch-{name}-{}-{call}.gguf", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    let dev = Device::open(backend).unwrap();
    let s = Stream::new(&dev).unwrap();
    let model = f(&path, &dev, &s);
    std::fs::remove_file(&path).unwrap();
    model.map(|m| (s, m))
}

/// A tensor table entry as `gguf::write` takes it: name, GGML dims, GGML type, data.
type Entry = (String, Vec<u64>, u32, Vec<u8>);

/// tiny's file (seed 1) with its metadata and tensor table edited by `f`.
fn edit(arch: Arch, f: impl FnOnce(&mut Vec<(String, Value)>, &mut Vec<Entry>)) -> Vec<u8> {
    edit_bytes(&tiny::gguf(arch, 1), f)
}

/// The GGUF `bytes` with its metadata and tensor table edited by `f`.
fn edit_bytes(bytes: &[u8], f: impl FnOnce(&mut Vec<(String, Value)>, &mut Vec<Entry>)) -> Vec<u8> {
    let g = Gguf::parse(bytes).unwrap();
    let mut tensors = g
        .tensors
        .iter()
        .map(|t| {
            let at = (g.data_offset + t.offset) as usize;
            let data = bytes[at..at + t.bytes as usize].to_vec();
            (t.name.clone(), t.dims.clone(), t.ggml_type, data)
        })
        .collect();
    let mut kv = g.kv.into_iter().collect();
    f(&mut kv, &mut tensors);
    gguf::write(&kv, &tensors)
}

fn set(kv: &mut Vec<(String, Value)>, key: &str, v: Value) {
    kv.retain(|e| e.0 != key);
    kv.push((key.into(), v));
}

#[test]
fn config_defaults() {
    // llama.cpp's: a scaling factor without a type is linear scaling.
    let bytes = edit(Arch::Llama, |kv, _| {
        set(kv, "llama.rope.scaling.factor", Value::F32(4.0))
    });
    let c = ModelConfig::from_gguf(&Gguf::parse(&bytes).unwrap()).unwrap();
    assert_eq!((c.rope.freq_scale, c.rope.yarn), (0.25, false));
}

#[test]
fn load_rejects_bad_files() {
    let bad_key = Some(LoadError::BadGguf(code::KEY, 0));
    // R9: each value is in range, but head_count x key_length overflows the widths derived from it.
    let huge = edit(Arch::Qwen3, |kv, _| {
        set(kv, "qwen3.attention.head_count", Value::U32(u32::MAX - 1));
        set(kv, "qwen3.attention.key_length", Value::U32(u32::MAX - 1));
    });
    assert_eq!(load(&huge, "huge").err(), bad_key);
    let empty = edit(Arch::Llama, |kv, _| {
        set(kv, "llama.block_count", Value::U32(0))
    });
    assert_eq!(load(&empty, "empty").err(), bad_key);
    // R9: more blocks than the file has tensors, refused before any per-layer loop runs.
    let endless = edit(Arch::Llama, |kv, _| {
        set(kv, "llama.block_count", Value::U32(u32::MAX))
    });
    assert_eq!(load(&endless, "endless").err(), bad_key);

    // Vectors are f32; matrices are quants, except an embedding table that is not also the lm_head.
    let f16 = |name: &str, tied: bool| {
        edit(Arch::Llama, |_, t| {
            let e = t.iter_mut().find(|e| e.0 == name).unwrap();
            e.2 = 1; // GGML F16
            e.3 = vec![0; e.1.iter().product::<u64>() as usize * 2];
            if tied {
                t.retain(|e| e.0 != "output.weight");
            }
        })
    };
    let f16_err = Some(LoadError::UnsupportedQuant(1));
    for (name, tied) in [
        ("blk.0.attn_norm.weight", false),
        ("blk.0.attn_q.weight", false),
        ("token_embd.weight", true),
    ] {
        let err = load(&f16(name, tied), "f16").err();
        assert_eq!(err, f16_err, "{name} tied={tied}");
    }
    assert!(load(&f16("token_embd.weight", false), "f16-embd").is_ok());
}

#[test]
fn load_needs_the_arch_tensors() {
    for (arch, name) in [
        (Arch::Qwen2, "blk.1.attn_k.bias"),
        (Arch::Qwen3, "blk.1.attn_k_norm.weight"),
        (Arch::Qwen3Moe, "blk.1.ffn_gate_inp.weight"),
        (Arch::Qwen3Moe, "blk.1.ffn_down_exps.weight"),
    ] {
        let mut bytes = tiny::gguf(arch, 1);
        let at = bytes
            .windows(name.len())
            .position(|w| w == name.as_bytes())
            .unwrap();
        bytes[at + name.len() - 1] = b'X'; // renamed, so the layer lacks it
        let err = load(&bytes, &format!("missing-{arch:?}")).err();
        assert_eq!(err, Some(LoadError::BadGguf(code::MISSING_TENSOR, 0)));
    }
}

/// GGML type 11 is Q3_K: 110 bytes per 256 weights, loaded and used on the CPU reference; a partial
/// block, a short data section or an offset past it are refused.
#[test]
fn q3_k_tensors_load_and_run() {
    assert_eq!(gguf::dtype(11), Ok(DType::Q3K));
    assert_eq!(gguf::ggml_type(DType::Q3K), Some(11));
    // output.weight as Q3_K: bytes from an LCG, d (the f16 at byte 108) 2^-7 in every block.
    let mut seed = 3u64;
    let mut q3 = vec![0u8; 256 * 110];
    for b in q3.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (seed >> 33) as u8;
    }
    q3.chunks_mut(110)
        .for_each(|b| b[108..].copy_from_slice(&0x2000u16.to_le_bytes()));
    let bytes = edit(Arch::Llama, |_, t| {
        let out = t.iter_mut().find(|e| e.0 == "output.weight").unwrap();
        (out.2, out.3) = (11, q3.clone());
    });
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(g.tensor("output.weight").unwrap().bytes, 256 * 110);
    let logits = prefill7(&bytes, "q3k");
    assert!(logits.iter().all(|x| x.is_finite()));
    assert_ne!(logits, prefill7(&tiny::gguf(Arch::Llama, 1), "q3k-base"));

    let one = |dims: Vec<u64>, data: Vec<u8>| {
        gguf::write(&[] as &[(&str, Value)], &[("w", dims, 11u32, data)])
    };
    assert!(Gguf::parse(&one(vec![200, 1], vec![0; 110])).is_err());
    let file = one(vec![256, 1], vec![0; 110]);
    let at = Gguf::parse(&file).unwrap().data_offset as usize;
    assert!(Gguf::parse(&file[..at + 109]).is_err());
    let mut far = file.clone();
    set_offset(&mut far, "w", 1 << 40);
    assert!(Gguf::parse(&far).is_err());
}

/// GGML type 10 is Q2_K: 84 bytes per 256 weights, loaded and used on the CPU reference; a partial
/// block or a short data section are refused.
#[test]
fn q2_k_tensors_load_and_run() {
    assert_eq!(gguf::dtype(10), Ok(DType::Q2K));
    assert_eq!(gguf::ggml_type(DType::Q2K), Some(10));
    // output.weight as Q2_K: bytes from an LCG, d and dmin (the f16s at bytes 80 and 82) 2^-7.
    let mut seed = 5u64;
    let mut q2 = vec![0u8; 256 * 84];
    for b in q2.iter_mut() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (seed >> 33) as u8;
    }
    q2.chunks_mut(84).for_each(|b| {
        b[80..82].copy_from_slice(&0x2000u16.to_le_bytes());
        b[82..].copy_from_slice(&0x2000u16.to_le_bytes());
    });
    let bytes = edit(Arch::Llama, |_, t| {
        let out = t.iter_mut().find(|e| e.0 == "output.weight").unwrap();
        (out.2, out.3) = (10, q2.clone());
    });
    let g = Gguf::parse(&bytes).unwrap();
    assert_eq!(g.tensor("output.weight").unwrap().bytes, 256 * 84);
    let logits = prefill7(&bytes, "q2k");
    assert!(logits.iter().all(|x| x.is_finite()));
    assert_ne!(logits, prefill7(&tiny::gguf(Arch::Llama, 1), "q2k-base"));

    let one = |dims: Vec<u64>, data: Vec<u8>| {
        gguf::write(&[] as &[(&str, Value)], &[("w", dims, 10u32, data)])
    };
    assert!(Gguf::parse(&one(vec![200, 1], vec![0; 84])).is_err());
    let file = one(vec![256, 1], vec![0; 84]);
    let at = Gguf::parse(&file).unwrap().data_offset as usize;
    assert!(Gguf::parse(&file[..at + 83]).is_err());
}

/// `bytes` with tensor `name`'s data offset (relative to the data section) set to `to`.
fn set_offset(bytes: &mut [u8], name: &str, to: u64) {
    let at = bytes
        .windows(name.len())
        .position(|w| w == name.as_bytes())
        .unwrap()
        + name.len();
    let n_dims = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let off = at + 4 + 8 * n_dims + 4;
    bytes[off..off + 8].copy_from_slice(&to.to_le_bytes());
}

#[test]
fn a_tensor_meeting_a_host_expert_is_refused() {
    // Another tensor at a host expert's offset, or inside it, would be in neither buffer.
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let g = Gguf::parse(&bytes).unwrap();
    let exps = g.tensor("blk.0.ffn_gate_exps.weight").unwrap().offset;
    let place = Placement {
        cpu_expert_layers: 1,
        ..Default::default()
    };
    for (to, what) in [(exps, "alias"), (exps + 32, "inside")] {
        let mut b = bytes.clone();
        set_offset(&mut b, "blk.0.attn_norm.weight", to);
        let err = load_placed(&b, &format!("overlap-{what}"), place).err();
        assert!(
            matches!(err, Some(LoadError::BadGguf(code::TENSOR, _))),
            "{what}: {err:?}"
        );
    }
}

#[test]
fn the_expert_width_defaults_as_llama_cpp_does() {
    // Without expert_feed_forward_length: feed_forward_length / expert_used_count, checked against the
    // expert tensors.
    assert_eq!(tiny::FFN / u64::from(tiny::EXPERTS_USED), tiny::EXPERT_FFN);
    let without = |ffn: Option<u32>| {
        edit(Arch::Qwen3Moe, |kv, _| {
            kv.retain(|e| e.0 != "qwen3moe.expert_feed_forward_length");
            if let Some(ffn) = ffn {
                set(kv, "qwen3moe.feed_forward_length", Value::U32(ffn));
            }
        })
    };
    let want = prefill7(&tiny::gguf(Arch::Qwen3Moe, 1), "width-set");
    assert_eq!(prefill7(&without(None), "width-default"), want);
    // A width the tensors do not have.
    let err = load(&without(Some(1000)), "width-wrong").err();
    assert!(
        matches!(err, Some(LoadError::BadGguf(code::TENSOR, _))),
        "{err:?}"
    );
    // An explicit 0 is malformed, as before.
    let zero = edit(Arch::Qwen3Moe, |kv, _| {
        set(kv, "qwen3moe.expert_feed_forward_length", Value::U32(0))
    });
    assert_eq!(
        load(&zero, "width-zero").err(),
        Some(LoadError::BadGguf(code::KEY, 0))
    );
}

const BLOCKS: u32 = 32;
const BLOCK_TOKENS: i32 = 16;
const MAX_BLOCKS: usize = 4;
/// Two sequences' block tables: reversed and disjoint, so no slot equals its position.
const TABLE_A: [i32; MAX_BLOCKS] = [31, 30, 29, 28];
const TABLE_B: [i32; MAX_BLOCKS] = [27, 26, 25, 24];
const V: usize = tiny::VOCAB as usize;

fn slot(table: &[i32], pos: i32) -> i32 {
    table[(pos / BLOCK_TOKENS) as usize] * BLOCK_TOKENS + pos % BLOCK_TOKENS
}

/// A decode row (token, position, block table), or `None` for a padding row.
type Decode<'a> = Option<(i32, i32, &'a [i32])>;
/// A prefill chunk (tokens, first position, block table).
type Prefill<'a> = (&'a [i32], i32, &'a [i32]);

/// The KV pool geometry over `k` and `v`.
fn pool<'a>(k: &'a Buffer, v: &'a Buffer) -> KvPool<'a> {
    KvPool {
        k,
        v,
        n_blocks: BLOCKS,
        block_tokens: BLOCK_TOKENS as u32,
        n_layers: tiny::LAYERS,
        n_kv_heads: tiny::KV_HEADS as u32,
        head_dim: tiny::HEAD_DIM as u32,
        dtype: DType::F16,
    }
}

struct Rig {
    s: Stream,
    model: Transformer,
    k: Buffer,
    v: Buffer,
    input: Buffer,
    logits: Buffer,
    /// Hand each step's ids to `Model::embed_tokens` before the forward pass, as the engine does.
    prime: bool,
    /// The steps' [`StepBatch::check`]: the prefill chunks are check chunks.
    check: bool,
    /// Capture each step's forward pass in one graph (no cut allowed) and replay it.
    capture: bool,
}

impl Rig {
    fn new(arch: Arch) -> Rig {
        Rig::with(&tiny::gguf(arch, 1), &format!("tiny-{arch:?}"))
    }

    fn with(bytes: &[u8], name: &str) -> Rig {
        Rig::placed(bytes, name, Placement::default())
    }

    fn placed(bytes: &[u8], name: &str, place: Placement) -> Rig {
        Rig::loaded(load_placed(bytes, name, place).unwrap())
    }

    fn cached(bytes: &[u8], name: &str, plan: &ExpertPlan) -> Rig {
        Rig::loaded(load_cached(bytes, name, plan).unwrap())
    }

    fn loaded((s, model): (Stream, Transformer)) -> Rig {
        let dev = s.device();
        let input = Buffer::alloc(dev, 1 << 16).unwrap();
        // [layers][blocks * block_tokens][kv heads][head_dim] of f16 (KvPool::bytes is private).
        let bytes = u64::from(tiny::LAYERS * BLOCKS)
            * BLOCK_TOKENS as u64
            * tiny::KV_HEADS
            * tiny::HEAD_DIM
            * 2;
        let (k, v) = (
            Buffer::alloc(dev, bytes).unwrap(),
            Buffer::alloc(dev, bytes).unwrap(),
        );
        let logits = Buffer::alloc(dev, u64::from(MAX_ROWS) * u64::from(tiny::VOCAB) * 4).unwrap();
        Rig {
            s,
            model,
            k,
            v,
            input,
            logits,
            prime: false,
            check: false,
            capture: false,
        }
    }

    /// Fills the KV pool with f16 NaN, so reading a slot this run did not write shows in the logits.
    fn clear_kv(&self) {
        let nan = vec![0x7e00u16; (self.k.bytes() / 2) as usize];
        self.s.upload(&self.k, 0, &nan).unwrap();
        self.s.upload(&self.v, 0, &nan).unwrap();
    }

    /// One step of `decode` rows then `prefill` chunks; returns the logits of rows `want`.
    fn step(&mut self, decode: &[Decode], prefill: &[Prefill], want: &[i32]) -> Vec<f32> {
        let (mut tok, mut pos, mut slots, mut table, mut lens) =
            (vec![], vec![], vec![], vec![], vec![]);
        for d in decode {
            let (t, p, tb, len) = match *d {
                Some((t, p, tb)) => (t, p, tb, p + 1),
                None => (0, 0, &[0; MAX_BLOCKS][..], 0),
            };
            tok.push(t);
            pos.push(p);
            slots.push(if len > 0 { slot(tb, p) } else { -1 });
            table.extend(tb);
            lens.push(len);
        }
        let (mut starts, mut q_lens) = (vec![], vec![]);
        for &(ts, start, tb) in prefill {
            for (j, &t) in ts.iter().enumerate() {
                tok.push(t);
                pos.push(start + j as i32);
                slots.push(slot(tb, start + j as i32));
            }
            table.extend(tb);
            starts.push(start);
            q_lens.push(ts.len() as i32);
        }
        let mut at = 0;
        let mut put = |v: &[i32], shape: &[i64]| {
            self.s.upload(&self.input, at, v).unwrap();
            let t = (at, shape.to_vec());
            at += v.len() as u64 * 4;
            t
        };
        let len = |v: &[i32]| v.len() as i64;
        let seqs = (decode.len() + prefill.len()) as i64;
        let spec = [
            put(&tok, &[len(&tok)]),
            put(&pos, &[len(&pos)]),
            put(&slots, &[len(&slots)]),
            put(&table, &[seqs, MAX_BLOCKS as i64]),
            put(&lens, &[len(&lens)]),
            put(&starts, &[len(&starts)]),
            put(&q_lens, &[len(&q_lens)]),
            put(want, &[len(want)]),
        ];
        let t = |i: usize| Tensor::new(&self.input, spec[i].0, DType::I32, &spec[i].1).unwrap();
        let batch = StepBatch {
            token_ids: t(0),
            positions: t(1),
            slot_ids: t(2),
            decode_rows: decode.len() as u32,
            prefill_rows: (tok.len() - decode.len()) as u32,
            block_table: t(3),
            seq_lens: t(4),
            prefill_q_start: t(5),
            prefill_q_len: t(6),
            logits_rows: t(7),
            attn_splits: 0,
            check: self.check,
        };
        let shape = [len(want), i64::from(tiny::VOCAB)];
        let logits = Tensor::new(&self.logits, 0, DType::F32, &shape).unwrap();
        let kv = pool(&self.k, &self.v);
        if self.prime {
            self.model.embed_tokens(&self.s, &tok).unwrap();
        }
        if self.capture {
            let model = &mut self.model;
            let (cut, last) =
                Graph::capture_pieces(&self.s, |s| model.forward(s, &batch, &kv, &logits))
                    .map_err(|e| (e, larch_kernels::last_error()))
                    .unwrap()
                    .expect("a device that captures graphs");
            assert!(cut.is_empty(), "the step cut its graph {} times", cut.len());
            last.launch(&self.s).unwrap();
        } else {
            self.model
                .forward(&self.s, &batch, &kv, &logits)
                .map_err(|e| (e, larch_kernels::last_error()))
                .unwrap();
        }
        let mut out = vec![0f32; want.len() * V];
        self.s.download(&self.logits, 0, &mut out).unwrap();
        out
    }
}

fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
}

/// `got` matches the reference rows `want` within 2e-3 of the largest |logit|, and has the same top
/// token wherever its lead over the runner-up exceeds the difference.
fn agree(name: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{name}");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "{name}: non-finite logits"
    );
    let scale = want.iter().fold(1f32, |m, x| m.max(x.abs()));
    let diff = want
        .iter()
        .zip(got)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("{name}: max |diff| {diff:.2e}, max |logit| {scale:.2}");
    assert!(diff <= 2e-3 * scale, "{name}: {diff}");
    for (row, (f, o)) in want.chunks(V).zip(got.chunks(V)).enumerate() {
        let best = argmax(f);
        let lead = f[best]
            - (0..V)
                .filter(|&i| i != best)
                .map(|i| f[i])
                .fold(f32::MIN, f32::max);
        if lead > 2.0 * diff {
            assert_eq!(best, argmax(o), "{name} row {row}");
        }
    }
}

/// Feeds `p` in chunks of 7 rows from a cleared pool; returns every row's logits.
fn chunked(r: &mut Rig, p: &[i32], table: &[i32]) -> Vec<f32> {
    r.clear_kv();
    let mut out = Vec::new();
    for (i, c) in p.chunks(7).enumerate() {
        let rows: Vec<i32> = (0..c.len() as i32).collect();
        out.extend(r.step(&[], &[(c, 7 * i as i32, table)], &rows));
    }
    out
}

#[test]
fn prefill_decode_and_mixed_steps_agree() {
    let prompt = |m: i32| -> Vec<i32> {
        (0..MAX_ROWS as i32)
            .map(|i| (i * m + 11) % tiny::VOCAB as i32)
            .collect()
    };
    let (a, b) = (prompt(37), prompt(91));
    let all: Vec<i32> = (0..MAX_ROWS as i32).collect();
    for arch in ARCHS {
        let mut r = Rig::new(arch);
        let name = |run: &str| format!("{arch:?} {run}");
        // The reference prefill exceeds every backend/weight crossover, reaching f16 dequant + gemm.
        r.clear_kv();
        let full = r.step(&[], &[(&a, 0, &TABLE_A)], &all);
        // NaN or a dead model would make every comparison below pass.
        assert!(full.iter().all(|x| x.is_finite()), "{arch:?}: non-finite");
        let (lo, hi) = full
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
        assert!(hi - lo > 0.5, "{arch:?}: logits barely vary ({lo}..{hi})");

        // One token per step, as decode rows (mmvq, attn_decode).
        r.clear_kv();
        let decoded: Vec<f32> = (0..a.len())
            .flat_map(|i| r.step(&[Some((a[i], i as i32, &TABLE_A))], &[], &[0]))
            .collect();
        agree(&name("decode"), &full, &decoded);
        // Prefill chunks of 7 rows (mmvq), each attending to the chunks before it.
        let chunked_a = chunked(&mut r, &a, &TABLE_A);
        agree(&name("chunked"), &full, &chunked_a);

        // Mixed steps: A prefills 7 rows and asks for the last one only, as the executor does; then A
        // decodes one token per step beside a padding row while B prefills in chunks of 7. B's
        // reference is B's chunks alone: the same arithmetic per row, so only the batch layout differs.
        let alone_b = chunked(&mut r, &b, &TABLE_B);
        r.clear_kv();
        let mut got_a = r.step(&[], &[(&a[..7], 0, &TABLE_A)], &[6]);
        let mut got_b = Vec::new();
        for (i, p) in (7..a.len()).enumerate() {
            let chunk: Vec<Prefill> = b
                .chunks(7)
                .nth(i)
                .map(|c| (c, 7 * i as i32, &TABLE_B[..]))
                .into_iter()
                .collect();
            // Rows: A's decode row 0, the padding row 1, then B's chunk.
            let nb = chunk.first().map_or(0, |c| c.0.len() as i32);
            let want: Vec<i32> = std::iter::once(0).chain(2..2 + nb).collect();
            let out = r.step(&[Some((a[p], p as i32, &TABLE_A)), None], &chunk, &want);
            got_a.extend(&out[..V]);
            got_b.extend(&out[V..]);
        }
        agree(&name("mixed A"), &full[6 * V..], &got_a);
        agree(&name("mixed B"), &alone_b, &got_b);

        // Two prefill chunks in one step, each with its own block table row.
        r.clear_kv();
        let rows: Vec<i32> = (0..14).collect();
        let two = r.step(
            &[],
            &[(&a[..7], 0, &TABLE_A), (&b[..7], 0, &TABLE_B)],
            &rows,
        );
        agree(&name("two chunks A"), &chunked_a[..7 * V], &two[..7 * V]);
        agree(&name("two chunks B"), &alone_b[..7 * V], &two[7 * V..]);
    }
}

/// The logits of one 7-row prefill of `bytes` from a cleared pool.
fn prefill7(bytes: &[u8], name: &str) -> Vec<f32> {
    prefill7_placed(bytes, name, Placement::default())
}

/// [`prefill7`] with the weights where `place` puts them.
fn prefill7_placed(bytes: &[u8], name: &str, place: Placement) -> Vec<f32> {
    let mut r = Rig::placed(bytes, name, place);
    r.clear_kv();
    let rows: Vec<i32> = (0..7).collect();
    r.step(&[], &[(&[5, 9, 2, 200, 31, 7, 64], 0, &TABLE_A)], &rows)
}

#[test]
fn tied_lm_head_is_the_embedding_table() {
    let tied = edit(Arch::Llama, |_, t| t.retain(|e| e.0 != "output.weight"));
    let copied = edit(Arch::Llama, |_, t| {
        let embd = t
            .iter()
            .find(|e| e.0 == "token_embd.weight")
            .unwrap()
            .clone();
        let out = t.iter_mut().find(|e| e.0 == "output.weight").unwrap();
        (out.2, out.3) = (embd.2, embd.3);
    });
    let got = prefill7(&tied, "tied");
    assert_eq!(got, prefill7(&copied, "copied"));
    assert_ne!(got, prefill7(&tiny::gguf(Arch::Llama, 1), "untied"));
}

/// The qwen3moe file with each layer's experts (and router rows) rewritten by `f`, which gets the
/// tensor's bytes and its number of experts.
fn experts(f: impl Fn(&str, &mut Vec<u8>, usize)) -> Vec<u8> {
    edit(Arch::Qwen3Moe, |_, t| {
        for e in t
            .iter_mut()
            .filter(|e| e.0.contains("_exps") || e.0.contains("gate_inp"))
        {
            f(&e.0, &mut e.3, tiny::EXPERTS as usize);
        }
    })
}

#[test]
fn moe_experts_are_indexed_by_the_router() {
    // Reversing the experts and the router's rows together permutes nothing the model computes.
    let reverse = |_: &str, d: &mut Vec<u8>, n: usize| {
        let each = d.len() / n;
        *d = d.chunks(each).rev().flatten().copied().collect();
    };
    let base = prefill7(&tiny::gguf(Arch::Qwen3Moe, 1), "moe");
    assert_eq!(prefill7(&experts(reverse), "moe-reversed"), base);
    // Reversing the experts alone routes each row to other experts.
    let only_experts = |name: &str, d: &mut Vec<u8>, n: usize| {
        if name.contains("_exps") {
            reverse(name, d, n);
        }
    };
    assert_ne!(prefill7(&experts(only_experts), "moe-misrouted"), base);
}

#[test]
fn host_experts_compute_what_device_experts_do() {
    // Experts in host memory compute what the device's do, in a prefill (gate, up and down per
    // chosen expert over 7 rows), then a decode row, within mmvq_id's tolerance (kernels/tests/ops.cpp):
    // the CPU device's fast mmvq_id sums in f32 where the reference sums in double. The rest of the
    // weights move to other offsets of the device buffer; each must still be found.
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let run = |layers: u32| {
        let place = Placement {
            cpu_expert_layers: layers,
            ..Default::default()
        };
        let mut r = Rig::placed(&bytes, &format!("moe-host-{layers}"), place);
        r.clear_kv();
        let rows: Vec<i32> = (0..7).collect();
        let mut got = r.step(&[], &[(&[5, 9, 2, 200, 31, 7, 64], 0, &TABLE_A)], &rows);
        got.extend(r.step(&[Some((17, 7, &TABLE_A))], &[], &[0]));
        got
    };
    let want = run(0);
    assert!(want.iter().all(|x| x.is_finite()));
    let max = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    for layers in [1, tiny::LAYERS, 99] {
        let got = run(layers);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let tol = 1e-3 * w.abs() + 1e-3 * max;
            assert!(
                (g - w).abs() <= tol,
                "{layers} host layers: [{i}] {g} vs {w}"
            );
        }
    }
    // A dense model has no routed experts to place.
    let dense = tiny::gguf(Arch::Qwen3, 1);
    let p = Placement {
        cpu_expert_layers: 1,
        ..Default::default()
    };
    assert_eq!(
        prefill7(&dense, "dense"),
        prefill7_placed(&dense, "dense-placed", p)
    );
}

/// A token embedding kept in host memory (the discrete-CUDA MoE load) computes what the device's
/// does, bit for bit, in a prefill and a decode step, beside a host-expert layer and a split plan:
/// with the ids handed over before each pass, as the engine does, and read back without them.
#[test]
fn a_host_embedding_computes_what_the_device_one_does() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let place = Placement {
        cpu_expert_layers: 1,
        ..Default::default()
    };
    let split = ExpertPlan {
        resident: vec![vec![0, 1], vec![3]],
        dynamic: false,
    };
    for plan in [None, Some(&split)] {
        let run = |host: bool, prime: bool| {
            let name = format!("embd-{host}-{prime}-{}", plan.is_some());
            let loaded = load_by(&bytes, &name, |p, d, s| {
                Transformer::load_planned(p, d, s, MAX_ROWS, place, plan, host)
            });
            let mut r = Rig::loaded(loaded.unwrap());
            r.prime = prime;
            assert_eq!(r.model.embed_and_head().is_none(), host);
            r.clear_kv();
            let rows: Vec<i32> = (0..7).collect();
            let mut got = r.step(&[], &[(&[5, 9, 2, 200, 31, 7, 64], 0, &TABLE_A)], &rows);
            got.extend(r.step(&[Some((17, 7, &TABLE_A))], &[], &[0]));
            got
        };
        let want = run(false, false);
        assert!(want.iter().all(|x| x.is_finite()));
        for prime in [true, false] {
            let what = format!("split plan {}, primed {prime}", plan.is_some());
            assert_eq!(run(true, prime), want, "{what}");
        }
    }
}

#[test]
fn streamed_prefill_computes_what_the_cpu_path_does() {
    // The experts of the host layers, streamed through the pool onto the device, compute what the CPU
    // device computes on them within mmvq_id's tolerance, and the slots are refilled every step; a
    // mixed layer's hot and cold tensors stream together.
    let plain = tiny::gguf(Arch::Qwen3Moe, 1);
    let rows: Vec<i32> = (0..7).collect();
    for (bytes, layers) in [&plain[..], cold_tiny().1]
        .into_iter()
        .flat_map(|b| [1, tiny::LAYERS, 99].map(|l| (b, l)))
    {
        let bytes = bytes.to_vec();
        let place = |stream_rows| Placement {
            cpu_expert_layers: layers,
            stream_rows,
        };
        let want = prefill7_placed(&bytes, &format!("stream-cpu-{layers}"), place(0));
        let max = want.iter().fold(0f32, |m, x| m.max(x.abs()));
        let mut r = Rig::placed(&bytes, &format!("stream-{layers}"), place(7));
        assert!(r.model.stream_pool_want().0 > 0, "{layers} host layers");
        assert!(r.model.alloc_stream_pool(&r.s).unwrap());
        for run in 0..2 {
            r.clear_kv();
            let got = r.step(&[], &[(&[5, 9, 2, 200, 31, 7, 64], 0, &TABLE_A)], &rows);
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let tol = 1e-3 * w.abs() + 1e-3 * max;
                assert!(
                    (g - w).abs() <= tol,
                    "{layers} layers, run {run}: [{i}] {g} vs {w}"
                );
            }
        }
        assert_eq!(r.model.streamed_steps(), 2, "{layers} host layers");
    }
}

/// On CUDA (GPU 0): a split layer's decode rows read its cold experts in place (`mmvq_id_mixed`
/// over the host experts' page-locked weights) inside the step's graph, so a split load decodes as
/// the resident load does, bit for bit: at 1 and 4 rows, a check step's rows as their one-row
/// decodes, a decode row beside a prompt chunk as the same row alone, and each decode step replayed
/// from one graph piece as it ran eagerly. Layer 0 keeps 2 of its 4 experts and layer 1 one, so
/// every row's 2 experts in layer 1 include a cold one; the prompt rows, which keep the host path,
/// count those cold pairs from their routed ids, and the decode rows count none. Skips without CUDA.
#[test]
fn split_layers_decode_their_cold_experts_in_place() {
    let cuda = larch_kernels::Device::open(0)
        .ok()
        .and_then(|d| d.props().ok())
        .is_some_and(|p| p.backend == larch_kernels::Backend::Cuda);
    if !cuda {
        return;
    }
    let b = tiny::gguf(Arch::Qwen3Moe, 1);
    let split = plan(&[&[0, 1], &[3]], false);
    let gpu = |name: &str, p: Option<&ExpertPlan>| {
        let loaded = load_by_on(&b, name, 0, |path, d, s| match p {
            Some(p) => Transformer::load_cached(path, d, s, MAX_ROWS, p),
            None => Transformer::load_placed(path, d, s, MAX_ROWS, Placement::default()),
        });
        Rig::loaded(loaded.unwrap())
    };
    let (mut res, mut sp) = (
        gpu("mixed-resident", None),
        gpu("mixed-split", Some(&split)),
    );
    let prompt: Vec<i32> = (0..12)
        .map(|i| (i * 37 + 11) % tiny::VOCAB as i32)
        .collect();
    let one = |r: &mut Rig| {
        r.clear_kv();
        (0..prompt.len())
            .flat_map(|i| r.step(&[Some((prompt[i], i as i32, &TABLE_A))], &[], &[0]))
            .collect::<Vec<f32>>()
    };
    let want = one(&mut res);
    assert!(want.iter().all(|x| x.is_finite()));
    assert_eq!(one(&mut sp), want, "1 row");
    let tables: [&[i32]; 4] = [&TABLE_A, &TABLE_B, &[23, 22, 21, 20], &[19, 18, 17, 16]];
    let four = |r: &mut Rig| {
        r.clear_kv();
        (0..6)
            .flat_map(|i| {
                let d: Vec<Decode> = (0..4)
                    .map(|q| Some((prompt[(i + 3 * q) % 12], i as i32, tables[q])))
                    .collect();
                r.step(&d, &[], &[0, 1, 2, 3])
            })
            .collect::<Vec<f32>>()
    };
    assert_eq!(four(&mut sp), four(&mut res), "4 rows");
    assert_eq!(
        sp.model.expert_hits(),
        Some((0, 0)),
        "decode rows count no pairs"
    );
    // Three guesses checked at once: their one-row decodes' logits.
    sp.clear_kv();
    for (i, &t) in prompt[..4].iter().enumerate() {
        sp.step(&[Some((t, i as i32, &TABLE_A))], &[], &[0]);
    }
    sp.check = true;
    let checked = sp.step(&[], &[(&prompt[4..7], 4, &TABLE_A)], &[0, 1, 2]);
    sp.check = false;
    assert_eq!(checked, want[4 * V..7 * V], "a check step's rows");
    // A decode row beside a prompt chunk: the row's logits alone, and the chunk's (on split_ffn
    // from the row after it) the resident load's; the chunk's 7 rows count their 28 pairs in the 2
    // split layers, a cold one in layer 1 for each row at least.
    let chunk = [5, 9, 2, 200, 31, 7, 64];
    let mixed = |r: &mut Rig| {
        r.clear_kv();
        for (i, &t) in prompt[..5].iter().enumerate() {
            r.step(&[Some((t, i as i32, &TABLE_A))], &[], &[0]);
        }
        let alone = r.step(&[Some((prompt[5], 5, &TABLE_A))], &[], &[0]);
        let decode = [Some((prompt[5], 5, &TABLE_A[..]))];
        let beside = r.step(&decode, &[(&chunk, 0, &TABLE_B)], &[0, 7]);
        (alone, beside)
    };
    let (alone, beside) = mixed(&mut sp);
    assert_eq!(beside[..V], alone[..], "a decode row beside a prompt chunk");
    let resident = mixed(&mut res).1;
    close(
        "a prompt chunk beside a decode row",
        &resident[V..],
        &beside[V..],
    );
    let (hits, misses) = sp.model.expert_hits().unwrap();
    assert_eq!(
        hits + misses,
        7 * 2 * 2,
        "only the prompt rows took the host path"
    );
    assert!(misses >= 7, "{misses} cold pairs");
    // Each decode step in one graph piece, replayed as it ran.
    for n in [1usize, 4] {
        sp.clear_kv();
        let d: Vec<Decode> = (0..n).map(|q| Some((prompt[q], 0, tables[q]))).collect();
        let rows: Vec<i32> = (0..n as i32).collect();
        let eager = sp.step(&d, &[], &rows);
        sp.capture = true;
        let replayed = sp.step(&d, &[], &rows);
        sp.capture = false;
        assert_eq!(replayed, eager, "{n} rows replayed");
    }
}

#[test]
fn tensor_slot_prefill_is_bit_identical_at_64_and_512_rows() {
    // Child processes isolate the slot setting from parallel tests in this binary.
    if std::env::var("LARCH_PREFILL_STREAM_SLOTS").is_err() {
        for slots in ["1", "2"] {
            assert!(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "tensor_slot_prefill_is_bit_identical_at_64_and_512_rows"
                    ])
                    .env("LARCH_PREFILL_STREAM_SLOTS", slots)
                    .status()
                    .unwrap()
                    .success(),
                "{slots} slots"
            );
        }
        return;
    }
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    for rows in [64usize, 512] {
        let tokens: Vec<i32> = (0..rows).map(|i| (i % V) as i32).collect();
        let tables: Vec<[i32; MAX_BLOCKS]> = (0..rows / 64)
            .map(|i| {
                [
                    4 * i as i32,
                    4 * i as i32 + 1,
                    4 * i as i32 + 2,
                    4 * i as i32 + 3,
                ]
            })
            .collect();
        let prefill: Vec<Prefill<'_>> = tokens
            .chunks(64)
            .zip(&tables)
            .map(|(t, table)| (t, 0, &table[..]))
            .collect();
        let want: Vec<i32> = (0..rows / 64).map(|i| (i * 64 + 63) as i32).collect();
        let run = |stream_rows| {
            let place = Placement {
                cpu_expert_layers: tiny::LAYERS,
                stream_rows,
            };
            let mut r = Rig::loaded(
                load_by_on(&bytes, "tensor-slot-bit", CPU_COMPUTE, |p, d, s| {
                    Transformer::load_placed(p, d, s, rows as u32, place)
                })
                .unwrap(),
            );
            if stream_rows > 0 {
                assert!(r.model.alloc_stream_pool(&r.s).unwrap());
            }
            r.clear_kv();
            r.step(&[], &prefill, &want)
        };
        let (got, cold) = (run(64), run(0));
        let first = got.iter().zip(&cold).position(|(a, b)| a != b);
        let max = got
            .iter()
            .zip(&cold)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            first.is_none(),
            "{rows} rows: first mismatch {first:?}, max diff {max}"
        );
    }
}

#[test]
fn ineligible_steps_stay_on_the_cpu_path_bit_for_bit() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let place = |stream_rows| Placement {
        cpu_expert_layers: tiny::LAYERS,
        stream_rows,
    };
    let prompt: Prefill = (&[5, 9, 2, 200, 31, 7, 64], 0, &TABLE_A);
    let rows: Vec<i32> = (0..7).collect();
    let cpu = |name: &str| Rig::placed(&bytes, name, place(0));
    // No pool (the fit left none, or it was never allocated).
    let want = prefill7_placed(&bytes, "ineligible-cpu", place(0));
    let mut r = Rig::placed(&bytes, "ineligible-no-pool", place(7));
    r.clear_kv();
    assert_eq!(r.step(&[], &[prompt], &rows), want);
    assert_eq!(r.model.streamed_steps(), 0, "no pool");
    // Below the crossover.
    let mut r = Rig::placed(&bytes, "ineligible-below", place(8));
    assert!(r.model.alloc_stream_pool(&r.s).unwrap());
    r.clear_kv();
    assert_eq!(r.step(&[], &[prompt], &rows), want);
    assert_eq!(r.model.streamed_steps(), 0, "below R");
    // A mixed step: decode rows run beside the prefill. Its chunk streams; its decode row keeps the
    // CPU path, and every row matches the CPU path's bits here.
    let mixed = |r: &mut Rig| {
        r.clear_kv();
        r.step(&[], &[prompt], &rows);
        let b: Prefill = (&[8, 4, 77], 0, &TABLE_B);
        r.step(&[Some((17, 7, &TABLE_A))], &[b], &[0, 3])
    };
    let mut r = Rig::placed(&bytes, "ineligible-mixed", place(1));
    assert!(r.model.alloc_stream_pool(&r.s).unwrap());
    let got = mixed(&mut r);
    assert_eq!(got, mixed(&mut cpu("ineligible-mixed-cpu")));
    assert_eq!(
        r.model.streamed_steps(),
        2,
        "the prefill step and the mixed step's chunk"
    );
    // Expert adapters run on the CPU; router deltas alone stream.
    let ad = adapters(1.0);
    let [want_ad, _] = prefill7_deltas(&bytes, "ineligible-ad-cpu", place(0), &ad);
    let mut r = Rig::placed(&bytes, "ineligible-ad", place(7));
    assert!(r.model.alloc_stream_pool(&r.s).unwrap());
    r.model.set_router_deltas(&r.s, Some(&ad)).unwrap();
    r.clear_kv();
    assert_eq!(r.step(&[], &[prompt], &rows), want_ad);
    assert_eq!(r.model.streamed_steps(), 0, "adapters");
    r.model
        .set_router_deltas(&r.s, Some(&every_layer(&[1.0, 0.0, -1.0, 0.0])))
        .unwrap();
    r.clear_kv();
    r.step(&[], &[prompt], &rows);
    assert_eq!(r.model.streamed_steps(), 1, "router deltas alone stream");
}

#[test]
fn moe_of_identical_experts_is_the_dense_ffn() {
    // Every expert a copy of expert 0: the weights over the top k sum to 1, so the block is expert 0.
    let copies = |name: &str, d: &mut Vec<u8>, n: usize| {
        if name.contains("_exps") {
            *d = d[..d.len() / n].repeat(n);
        }
    };
    let moe = experts(copies);
    // The same file as qwen3: expert 0 as the dense gate, up and down, and no router.
    let dense = edit(Arch::Qwen3Moe, |kv, t| {
        for e in kv.iter_mut() {
            e.0 = e.0.replace("qwen3moe.", "qwen3.");
            if e.0 == "general.architecture" {
                e.1 = Value::Str("qwen3".into());
            }
        }
        set(
            kv,
            "qwen3.feed_forward_length",
            Value::U32(tiny::EXPERT_FFN as u32),
        );
        t.retain(|e| !e.0.contains("gate_inp"));
        for e in t.iter_mut().filter(|e| e.0.contains("_exps")) {
            e.0 = e.0.replace("_exps", "");
            e.3.truncate(e.3.len() / tiny::EXPERTS as usize);
            e.1.pop();
        }
    });
    // Bit-equal: 7 rows run mmvq dense and mmvq_id routed, the same GEMV per row.
    assert_eq!(prefill7(&moe, "moe-copies"), prefill7(&dense, "dense"));
}

/// The same delta on every layer of tiny's qwen3moe.
fn every_layer(delta: &[f32]) -> RouterDeltas {
    RouterDeltas {
        layers: (0..tiny::LAYERS as u16).collect(),
        experts: tiny::EXPERTS,
        deltas: vec![delta.to_vec(); tiny::LAYERS as usize],
        ..Default::default()
    }
}

/// [`prefill7_placed`] with router deltas `d` set, then (same rig, same prompt) with them cleared.
fn prefill7_deltas(bytes: &[u8], name: &str, place: Placement, d: &RouterDeltas) -> [Vec<f32>; 2] {
    let mut r = Rig::placed(bytes, name, place);
    let rows: Vec<i32> = (0..7).collect();
    let mut run = |d: Option<&RouterDeltas>| {
        r.model.set_router_deltas(&r.s, d).unwrap();
        r.clear_kv();
        r.step(&[], &[(&[5, 9, 2, 200, 31, 7, 64], 0, &TABLE_A)], &rows)
    };
    [run(Some(d)), run(None)]
}

#[test]
fn zero_router_deltas_are_the_base_model_bit_for_bit() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    for layers in [0, 99] {
        let place = Placement {
            cpu_expert_layers: layers,
            ..Default::default()
        };
        let base = prefill7_placed(&bytes, &format!("deltas-base-{layers}"), place);
        let zero = every_layer(&[0.0; tiny::EXPERTS as usize]);
        let [with, cleared] =
            prefill7_deltas(&bytes, &format!("deltas-zero-{layers}"), place, &zero);
        assert_eq!(with, base, "{layers} host layers: zero deltas");
        assert_eq!(cleared, base, "{layers} host layers: cleared");
        // A nonzero set changes the answer, and clearing it restores the base exactly.
        let some = every_layer(&[3.0, -3.0, 0.5, 0.0]);
        let [with, cleared] =
            prefill7_deltas(&bytes, &format!("deltas-some-{layers}"), place, &some);
        assert_ne!(with, base, "{layers} host layers: deltas");
        assert_eq!(cleared, base, "{layers} host layers: cleared after deltas");
    }
}

/// With every router zeroed, routing is the deltas' alone: the host predicts the top k of the delta
/// (ties to the lower id, as moe_route), and the model with the deltas computes what the model
/// without them computes with those experts moved to slots 0..k, where all-zero logits tie.
#[test]
fn router_deltas_route_as_the_host_predicts() {
    let zero_router = |name: &str, d: &mut Vec<u8>, _: usize| {
        if name.contains("gate_inp") {
            d.fill(0);
        }
    };
    // Experts 1 and 3 tie far above the rest: weights 0.5 each, exactly as four zero logits give.
    let delta = [-100f32, 0.0, -100.0, 0.0];
    let mut ids: Vec<usize> = (0..delta.len()).collect();
    ids.sort_by(|&a, &b| delta[b].total_cmp(&delta[a]).then(a.cmp(&b)));
    assert_eq!(ids, [1, 3, 0, 2]);
    let moved = |name: &str, d: &mut Vec<u8>, n: usize| {
        zero_router(name, d, n);
        if name.contains("_exps") {
            let each = d.chunks(d.len() / n).collect::<Vec<_>>();
            *d = ids.iter().flat_map(|&i| each[i]).copied().collect();
        }
    };
    let (zeroed, moved) = (experts(zero_router), experts(moved));
    for layers in [0, 99] {
        let place = Placement {
            cpu_expert_layers: layers,
            ..Default::default()
        };
        let name = format!("deltas-route-{layers}");
        let [got, plain] = prefill7_deltas(&zeroed, &name, place, &every_layer(&delta));
        let want = prefill7_placed(&moved, &format!("deltas-moved-{layers}"), place);
        assert_eq!(got, want, "{layers} host layers");
        assert_ne!(got, plain, "{layers} host layers: the deltas route");
    }
}

/// Rank-2 adapters on experts 1 and 3 of every layer; `b` scales B.
fn adapters(b: f32) -> RouterDeltas {
    let (d, ef, r) = (tiny::D as usize, tiny::EXPERT_FFN as usize, 2);
    let ramp = |n: usize, s: f32| (0..n).map(move |i| ((i * 7 % 13) as f32 - 6.0) * s);
    let one: Vec<f32> = ramp(d * r, 0.02 * b).chain(ramp(r * ef, 0.05)).collect();
    RouterDeltas {
        layers: (0..tiny::LAYERS as u16).collect(),
        experts: tiny::EXPERTS,
        deltas: vec![[one.clone(), one].concat(); tiny::LAYERS as usize],
        rank: r as u32,
        ids: vec![vec![1, 3]; tiny::LAYERS as usize],
    }
}

#[test]
fn expert_adapters_are_the_base_at_zero_and_agree_on_host_experts() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let mut got = Vec::new();
    for layers in [0, 99] {
        let place = Placement {
            cpu_expert_layers: layers,
            ..Default::default()
        };
        let base = prefill7_placed(&bytes, &format!("ad-base-{layers}"), place);
        let [zero, _] =
            prefill7_deltas(&bytes, &format!("ad-zero-{layers}"), place, &adapters(0.0));
        assert_eq!(zero, base, "{layers} host layers: B = 0");
        let [with, cleared] =
            prefill7_deltas(&bytes, &format!("ad-some-{layers}"), place, &adapters(1.0));
        assert_ne!(with, base, "{layers} host layers: adapters");
        assert_eq!(cleared, base, "{layers} host layers: cleared");
        got.push(with);
    }
    assert_eq!(got[0], got[1], "device and host experts");
    let (s, mut m) = load(&bytes, "ad-bad").unwrap();
    for bad in [
        RouterDeltas {
            ids: vec![vec![3, 1]; 2],
            ..adapters(1.0)
        },
        RouterDeltas {
            ids: vec![vec![1, 4]; 2],
            ..adapters(1.0)
        },
        RouterDeltas {
            ids: vec![vec![1, 3]],
            ..adapters(1.0)
        },
        RouterDeltas {
            rank: 3,
            ..adapters(1.0)
        },
    ] {
        assert_eq!(m.set_router_deltas(&s, Some(&bad)), Err(KError::InvalidArg));
    }
}

#[test]
fn router_deltas_are_validated() {
    let (s, mut m) = load(&tiny::gguf(Arch::Qwen3Moe, 1), "deltas-bad").unwrap();
    let good = every_layer(&[0.0; tiny::EXPERTS as usize]);
    let bad = |f: &dyn Fn(&mut RouterDeltas)| {
        let mut d = good.clone();
        f(&mut d);
        d
    };
    for d in [
        bad(&|d| d.layers[1] = tiny::LAYERS as u16),
        bad(&|d| d.layers[1] = 0),
        bad(&|d| d.experts += 1),
        bad(&|d| d.deltas.pop().map(drop).unwrap()),
        bad(&|d| d.deltas[0].push(0.0)),
        bad(&|d| d.deltas[1][2] = f32::NAN),
        bad(&|d| d.deltas[1][2] = f32::INFINITY),
    ] {
        assert_eq!(
            m.set_router_deltas(&s, Some(&d)),
            Err(KError::InvalidArg),
            "{d:?}"
        );
    }
    // One layer alone is a valid set.
    let one = RouterDeltas {
        layers: vec![1],
        experts: tiny::EXPERTS,
        deltas: vec![vec![1.0; tiny::EXPERTS as usize]],
        ..Default::default()
    };
    assert_eq!(m.set_router_deltas(&s, Some(&one)), Ok(()));
    // A dense model has no router.
    let (s, mut dense) = load(&tiny::gguf(Arch::Qwen3, 1), "deltas-dense").unwrap();
    assert_eq!(dense.set_router_deltas(&s, None), Err(KError::Unsupported));
}

#[test]
fn qk_norms_run_on_qwen3_only() {
    // Norm weights of 3: applied, they would change every logit.
    let with_norms = edit(Arch::Llama, |_, t| {
        let w: Vec<u8> = (0..tiny::HEAD_DIM)
            .flat_map(|_| 3f32.to_le_bytes())
            .collect();
        for i in 0..tiny::LAYERS {
            for n in ["attn_q_norm", "attn_k_norm"] {
                t.push((
                    format!("blk.{i}.{n}.weight"),
                    vec![tiny::HEAD_DIM],
                    0,
                    w.clone(),
                ));
            }
        }
    });
    let plain = prefill7(&tiny::gguf(Arch::Llama, 1), "plain");
    assert_eq!(prefill7(&with_norms, "with-norms"), plain);
}

#[test]
fn rope_freqs_tensor_is_applied() {
    // The GGUF converter stores frequency factors in this tensor, without a
    // special scaling-type metadata key.
    let scaled = edit(Arch::Llama, |_, t| {
        let factors: Vec<u8> = (0..tiny::HEAD_DIM / 2)
            .flat_map(|_| 2f32.to_le_bytes())
            .collect();
        t.push((
            "rope_freqs.weight".into(),
            vec![tiny::HEAD_DIM / 2],
            0, // GGML F32
            factors,
        ));
    });
    let plain = prefill7(&tiny::gguf(Arch::Llama, 1), "plain-rope");
    let got = prefill7(&scaled, "scaled-rope");
    assert!(got.iter().all(|x| x.is_finite()));
    assert_ne!(got, plain, "rope_freqs.weight must affect the forward pass");
}

/// Taps as [`prefill_taps`] takes them: layers, final state, rows.
type TapSpec<'a> = (&'a [u16], bool, &'a [i32]);

/// One prefill of `tokens` from position 0 over a pool sized from `model`'s config, asking for every
/// row's logits in reverse order (so their gather is no identity): the logits, and with `taps` the
/// tap blocks (`taps: None` runs `forward` itself).
fn prefill_taps(
    s: &Stream,
    model: &mut Transformer,
    tokens: &[i32],
    taps: Option<TapSpec>,
) -> Result<(Vec<f32>, Vec<f32>), KError> {
    let (c, dev) = (model.config().clone(), s.device());
    let n = tokens.len() as i64;
    let blocks = tokens.len().div_ceil(BLOCK_TOKENS as usize);
    let slots = blocks as u64 * BLOCK_TOKENS as u64;
    let kv = u64::from(c.n_layers * c.n_kv_heads * c.head_dim) * slots * 2;
    let (k, v) = (
        Buffer::alloc(dev, kv).unwrap(),
        Buffer::alloc(dev, kv).unwrap(),
    );
    let pool = KvPool {
        k: &k,
        v: &v,
        n_blocks: blocks as u32,
        block_tokens: BLOCK_TOKENS as u32,
        n_layers: c.n_layers,
        n_kv_heads: c.n_kv_heads,
        head_dim: c.head_dim,
        dtype: DType::F16,
    };
    // One sequence over blocks 0, 1, 2, ...: a position's KV slot is the position.
    let pos: Vec<i32> = (0..n as i32).collect();
    let table: Vec<i32> = (0..blocks as i32).collect();
    let want: Vec<i32> = pos.iter().rev().copied().collect();
    let rows = taps.map_or(&[][..], |t| t.2);
    let parts: [(&[i32], Vec<i64>); 8] = [
        (tokens, vec![n]),
        (&pos, vec![n]),
        (&table, vec![1, blocks as i64]),
        (&[], vec![0]),
        (&[0], vec![1]),
        (&[n as i32], vec![1]),
        (&want, vec![n]),
        (rows, vec![rows.len() as i64]),
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
    let vocab = i64::from(c.vocab);
    let lb = Buffer::alloc(dev, (n * vocab * 4) as u64).unwrap();
    let logits = Tensor::new(&lb, 0, DType::F32, &[n, vocab]).unwrap();
    let mut out = Vec::new();
    match taps {
        None => model.forward(s, &batch, &pool, &logits)?,
        Some((layers, final_state, _)) => {
            let blocks = layers.len() + usize::from(final_state);
            let shape = [(blocks * rows.len()) as i64, i64::from(c.d_model)];
            out = vec![0f32; (shape[0] * shape[1]) as usize];
            let ob = Buffer::alloc(dev, (out.len() as u64 * 4).max(4)).unwrap();
            let taps = Taps {
                layers,
                final_state,
                rows: t[7],
                out: Tensor::new(&ob, 0, DType::F32, &shape).unwrap(),
            };
            model.forward_taps(s, &batch, &pool, &logits, Some(&taps))?;
            s.download(&ob, 0, &mut out).unwrap();
        }
    }
    let mut l = vec![0f32; (n * vocab) as usize];
    s.download(&lb, 0, &mut l).unwrap();
    Ok((l, out))
}

const TAPPED: [i32; 7] = [5, 9, 2, 200, 31, 7, 64];

#[test]
fn taps_leave_the_logits_unchanged() {
    for arch in ARCHS {
        let (s, mut m) = load(&tiny::gguf(arch, 1), &format!("taps-same-{arch:?}")).unwrap();
        let (plain, _) = prefill_taps(&s, &mut m, &TAPPED, None).unwrap();
        let taps: TapSpec = (&[1, 0], true, &[6, 2]);
        let (tapped, _) = prefill_taps(&s, &mut m, &TAPPED, Some(taps)).unwrap();
        assert!(plain.iter().all(|x| x.is_finite()), "{arch:?}");
        assert_eq!(tapped, plain, "{arch:?}");
    }
}

/// tiny's model cut to its first layer: the same layer 0, output norm and head.
fn one_layer(arch: Arch) -> Vec<u8> {
    edit(arch, |kv, t| {
        let name = kv.iter().find(|e| e.0 == "general.architecture");
        let name = name.and_then(|e| e.1.as_str()).unwrap().to_string();
        set(kv, &format!("{name}.block_count"), Value::U32(1));
        t.retain(|e| !e.0.starts_with("blk.1."));
    })
}

/// The CPU reference's rmsnorm of the f32 rows `x` (rows of `w.len()`) into f16, widened back to
/// f32 with `embed`: the final-state tap of rows whose last-layer residual is `x`.
fn cpu_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    let (d, n) = (w.len() as i64, (x.len() / w.len()) as i64);
    let ids: Vec<i32> = (0..n as i32).collect();
    let buf = |bytes: i64| Buffer::alloc(&dev, bytes as u64).unwrap();
    let [xb, wb, hb, ib, ob] = [4 * n * d, 4 * d, 2 * n * d, 4 * n, 4 * n * d].map(buf);
    s.upload(&xb, 0, x).unwrap();
    s.upload(&wb, 0, w).unwrap();
    s.upload(&ib, 0, &ids).unwrap();
    let xs = Tensor::new(&xb, 0, DType::F32, &[n, d]).unwrap();
    let ws = Tensor::new(&wb, 0, DType::F32, &[d]).unwrap();
    let h = Tensor::new(&hb, 0, DType::F16, &[n, d]).unwrap();
    let is = Tensor::new(&ib, 0, DType::I32, &[n]).unwrap();
    let out = Tensor::new(&ob, 0, DType::F32, &[n, d]).unwrap();
    ops::rmsnorm(&s, &xs, &ws, eps, &h).unwrap();
    ops::embed(&s, &h, &is, &out).unwrap();
    let mut v = vec![0f32; x.len()];
    s.download(&ob, 0, &mut v).unwrap();
    v
}

/// The output norm's f32 weights and epsilon, from a GGUF of `len` bytes.
fn out_norm(mut f: impl Read + Seek, len: u64) -> (Vec<f32>, f32) {
    let g = Gguf::read(&mut f, len).unwrap();
    let t = g.tensor("output_norm.weight").unwrap();
    let dtype = gguf::dtype(t.ggml_type).ok();
    assert_eq!(dtype, Some(DType::F32), "output_norm.weight");
    let mut b = vec![0u8; t.bytes as usize];
    f.seek(SeekFrom::Start(g.data_offset + t.offset)).unwrap();
    f.read_exact(&mut b).unwrap();
    let w = b
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()));
    (w.collect(), ModelConfig::from_gguf(&g).unwrap().rms_eps)
}

/// Each f16 value (widened) of `got` is `want`'s or its f16 neighbour: one f16 step is at most
/// 2^-10 of a normal value's size, and 2^-24 below the normal range.
fn one_f16_step(name: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{name}");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let step = w.abs() / 1024.0 + 2f32.powi(-24);
        assert!((w - g).abs() <= step, "{name}[{i}]: {g}, want {w}");
    }
}

#[test]
fn taps_copy_each_named_layer_and_the_final_state() {
    let d = tiny::D as usize;
    let block = |v: &[f32], k: usize| v[k * 2 * d..(k + 1) * 2 * d].to_vec();
    for arch in ARCHS {
        let bytes = tiny::gguf(arch, 1);
        let (w, eps) = out_norm(Cursor::new(&bytes), bytes.len() as u64);
        let tap = |(s, m): &mut (Stream, Transformer), taps: TapSpec| {
            prefill_taps(s, m, &TAPPED, Some(taps)).unwrap().1
        };
        let mut two = load(&bytes, &format!("taps-two-{arch:?}")).unwrap();
        let b = tap(&mut two, (&[1, 0], true, &[6, 2]));
        let (layer1, layer0, last) = (block(&b, 0), block(&b, 1), block(&b, 2));
        let mut one = load(&one_layer(arch), &format!("taps-one-{arch:?}")).unwrap();
        let c = tap(&mut one, (&[0], true, &[6, 2]));
        // Layer 0 is the same arithmetic in both models, and layer 1 is not layer 0.
        assert_eq!(layer0, block(&c, 0), "{arch:?} layer 0");
        assert_ne!(layer1, layer0, "{arch:?}");
        // The final state is the output norm of the last layer's residual: per row the CPU
        // reference's rmsnorm of the same f32 input, so the same bits.
        assert_eq!(cpu_norm(&layer1, &w, eps), last, "{arch:?} final");
        assert_eq!(cpu_norm(&layer0, &w, eps), block(&c, 1), "{arch:?} cut");
        // A layer alone and the final state alone fill the one block they ask for.
        let alone = tap(&mut two, (&[1], false, &[6, 2]));
        assert_eq!(alone, layer1, "{arch:?} layer 1 alone");
        let alone = tap(&mut two, (&[], true, &[6, 2]));
        assert_eq!(alone, last, "{arch:?} final alone");
        // Each block's rows follow `rows`.
        let swapped = tap(&mut two, (&[1, 0], true, &[2, 6]));
        for k in 0..3 {
            let rows = block(&b, k);
            let (r6, r2) = rows.split_at(d);
            assert_eq!(block(&swapped, k), [r2, r6].concat(), "{arch:?} block {k}");
        }
    }
}

/// A pass after `embed_tokens` embeds the rows handed to it, without reading the batch's ids back
/// (the engine's eager step): primed with other tokens of the same count, a prefill computes what
/// a prefill of those tokens does. A device embedding ignores the call.
#[test]
fn a_primed_pass_embeds_the_handed_rows() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let (fed, handed) = ([5, 9, 2, 200, 31, 7, 64], [8, 1, 77, 3, 250, 6, 40]);
    let rows: Vec<i32> = (0..7).collect();
    let run = |host: bool, tokens: &[i32], primed: Option<&[i32]>| {
        let loaded = load_by(&bytes, &format!("primed-{host}"), |p, d, s| {
            Transformer::load_planned(p, d, s, MAX_ROWS, Placement::default(), None, host)
        });
        let mut r = Rig::loaded(loaded.unwrap());
        r.clear_kv();
        if let Some(ids) = primed {
            r.model.embed_tokens(&r.s, ids).unwrap();
        }
        r.step(&[], &[(tokens, 0, &TABLE_A)], &rows)
    };
    let want = run(true, &handed, None);
    assert_ne!(want, run(true, &fed, None), "the tokens change the logits");
    assert_eq!(run(true, &fed, Some(&handed)), want);
    assert_eq!(run(false, &fed, Some(&handed)), run(false, &fed, None));
}

#[test]
fn a_tap_past_the_depth_is_an_error() {
    let (s, mut m) = load(&tiny::gguf(Arch::Llama, 1), "taps-deep").unwrap();
    for layers in [&[2][..], &[0, u16::MAX]] {
        let got = prefill_taps(&s, &mut m, &TAPPED, Some((layers, false, &[6])));
        assert_eq!(got.err(), Some(KError::InvalidArg), "{layers:?}");
    }
}

/// The Mac check on m4 (bench/models.toml): one 16-token prompt on GPU 0 and on the CPU reference,
/// tapping layer 0, the last layer and the final state at the last row.
/// - Layer 0: max |GPU - CPU| <= 1e-3 of the largest |CPU value|. Its residual is the embedding row
///   plus two f16 deltas (the attention and feed-forward outputs). The backends sum each delta's dot
///   products in different orders, so a delta can round to the neighbouring f16 value: at most 2^-10
///   (9.8e-4) of its size. 1e-3 of the largest value covers that step on both deltas wherever their
///   sizes add up to no more than that value.
/// - The GPU's final state: within one f16 step of the CPU reference's norm of the GPU's own
///   last-layer tap. The input is the same f32 rows; only the norm's f32 sum order differs.
/// - Printed for each block, gated for none but layer 0: the GPU-CPU difference. No bound across 24
///   layers of f16 activations has been measured.
///   LARCH_MODELS=<dir> cargo test --release -p larch-model --test tiny -- --ignored --nocapture tapped
#[test]
#[ignore = "needs m4 and a GPU: LARCH_MODELS=<dir of the GGUFs>"]
fn tapped_residual_matches_the_cpu_reference() {
    let dir = std::env::var("LARCH_MODELS").expect("LARCH_MODELS=<dir of the GGUFs>");
    let path = std::path::PathBuf::from(dir).join("m4.gguf");
    let tokens: Vec<i32> = (0..16).map(|i| i * 997 + 11).collect();
    let run = |ordinal: i32| {
        let dev = Device::open(ordinal).unwrap();
        let s = Stream::new(&dev).unwrap();
        let mut m = Transformer::load(&path, &dev, &s, MAX_ROWS).unwrap();
        let last = u16::try_from(m.config().n_layers - 1).unwrap();
        let taps: TapSpec = (&[0, last], true, &[15]);
        prefill_taps(&s, &mut m, &tokens, Some(taps)).unwrap().1
    };
    let (gpu, cpu) = (run(0), run(CPU));
    assert!(gpu.iter().chain(&cpu).all(|x| x.is_finite()), "non-finite");
    let d = cpu.len() / 3;
    let block = |v: &[f32], k: usize| v[k * d..(k + 1) * d].to_vec();
    let names = ["layer 0", "last layer", "final state"];
    for (k, name) in names.into_iter().enumerate() {
        let (g, c) = (block(&gpu, k), block(&cpu, k));
        let scale = c.iter().fold(0f32, |m, x| m.max(x.abs()));
        let diff = g
            .iter()
            .zip(&c)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("{name}: max |gpu - cpu| {diff:.3e}, max |cpu| {scale:.3e}");
        if k == 0 {
            assert!(diff <= 1e-3 * scale, "{name}: {diff} > 1e-3 x {scale}");
        }
    }
    let f = File::open(&path).unwrap();
    let len = f.metadata().unwrap().len();
    let (w, eps) = out_norm(BufReader::new(f), len);
    let want = cpu_norm(&block(&gpu, 1), &w, eps);
    one_f16_step("gpu final", &want, &block(&gpu, 2));
}

/// A router skewed towards the low experts, as real routing is: expert 0 most, 3 least.
fn skewed(r: &mut Rig) {
    let skew = every_layer(&[1.5, 0.75, -0.75, -1.5]);
    r.model.set_router_deltas(&r.s, Some(&skew)).unwrap();
}

/// The skewed model's 7-row prefill then 3 decode rows, every row's logits.
fn serve(r: &mut Rig) -> Vec<f32> {
    skewed(r);
    r.clear_kv();
    let p = [5, 9, 2, 200, 31, 7, 64];
    let rows: Vec<i32> = (0..7).collect();
    let mut got = r.step(&[], &[(&p, 0, &TABLE_A)], &rows);
    for (i, t) in [17, 99, 3].into_iter().enumerate() {
        got.extend(r.step(&[Some((t, 7 + i as i32, &TABLE_A))], &[], &[0]));
    }
    got
}

/// `got` within mmvq_id's tolerance of `want`, as host experts are of device experts: the CPU
/// device sums in f32 where the reference sums in double.
fn close(name: &str, want: &[f32], got: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}");
    let max = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-3 * w.abs() + 1e-3 * max,
            "{name}: [{i}] {g} vs {w}"
        );
    }
}

fn plan(resident: &[&[u32]], dynamic: bool) -> ExpertPlan {
    ExpertPlan {
        resident: resident.iter().map(|r| r.to_vec()).collect(),
        dynamic,
    }
}

#[test]
fn split_experts_serve_what_whole_layers_do() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let want = serve(&mut Rig::with(&bytes, "split-device"));
    assert!(want.iter().all(|x| x.is_finite()));
    let host = Placement {
        cpu_expert_layers: 99,
        ..Default::default()
    };
    let whole = serve(&mut Rig::placed(&bytes, "split-whole", host));
    close("whole layers", &want, &whole);
    let plans = [
        ("hot", plan(&[&[0, 1], &[0]], false)),
        ("cold", plan(&[&[3], &[1, 2, 3]], false)),
        ("one layer", plan(&[&[2, 0]], false)),
        ("host only", plan(&[&[], &[]], false)),
        ("dynamic", plan(&[&[3], &[2]], true)),
    ];
    for (name, p) in plans {
        let got = serve(&mut Rig::cached(&bytes, name, &p));
        close(name, &want, &got);
        close(name, &whole, &got);
        // Frozen residency reproduces bit for bit.
        if !p.dynamic {
            assert_eq!(got, serve(&mut Rig::cached(&bytes, name, &p)), "{name}");
        }
    }
    // Every expert resident is no split: the plain device path, bit for bit.
    let all = plan(&[&[3, 2, 1, 0], &[0, 1, 2, 3]], false);
    assert_eq!(serve(&mut Rig::cached(&bytes, "all", &all)), want);
}

#[test]
fn a_bad_plan_is_refused() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let bad = [
        &[4][..],
        &[1, 1],
        &[0, 0, 0, 0],
        &[0, 1, 2, 99],
        &[0, 1, 2, 3, 3],
    ];
    for bad in bad.map(|r| plan(&[r], false)) {
        assert!(load_cached(&bytes, "bad-plan", &bad).is_err(), "{bad:?}");
    }
}

/// A split row's bytes do not depend on its batch: a row whose experts are all resident takes the
/// device part alone, and beside a row with host experts it also gets an exact 0 from the host.
#[test]
fn split_rows_do_not_depend_on_their_batch() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let p = plan(&[&[0], &[0, 1]], false);
    let mut r = Rig::cached(&bytes, "split-rows", &p);
    skewed(&mut r);
    let (a, b) = ([5, 9, 2, 200, 31, 7, 64], [1, 77, 140, 3, 250, 8, 19]);
    let prefill = |r: &mut Rig| {
        r.clear_kv();
        r.step(&[], &[(&a, 0, &TABLE_A)], &[6]);
        r.step(&[], &[(&b, 0, &TABLE_B)], &[6]);
    };
    let mut both = Vec::new();
    let mut alone = Vec::new();
    for t in [17, 99, 3, 128, 64] {
        prefill(&mut r);
        both.extend(r.step(
            &[Some((t, 7, &TABLE_A)), Some((t + 1, 7, &TABLE_B))],
            &[],
            &[0, 1],
        ));
        prefill(&mut r);
        alone.extend(r.step(&[Some((t, 7, &TABLE_A))], &[], &[0]));
        alone.extend(r.step(&[Some((t + 1, 7, &TABLE_B))], &[], &[0]));
    }
    assert_eq!(both, alone);
    let (hits, misses) = r.model.expert_hits().unwrap();
    assert!(hits > 0 && misses > 0, "{hits} hits, {misses} misses");
}

#[test]
fn dynamic_residency_moves_hot_experts_onto_the_device() {
    // Every slot starts on the coldest expert; the skewed router's hot ones move in.
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let rate = |dynamic| {
        let mut r = Rig::cached(&bytes, "dyn", &plan(&[&[3, 2], &[3, 2]], dynamic));
        serve(&mut r);
        serve(&mut r);
        let (h, m) = r.model.expert_hits().unwrap();
        h as f64 / (h + m) as f64
    };
    let (frozen, dynamic) = (rate(false), rate(true));
    assert!(dynamic > frozen, "{dynamic} vs {frozen}");
}

#[test]
fn a_split_model_records_its_routes() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let dir = std::env::temp_dir().join(format!("larch-record-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = Histogram::path(&dir.join("tiny.gguf"));
    let (len, e) = (bytes.len() as u64, tiny::EXPERTS);
    // Layer 0 splits, so its routes come with its host round trip; layer 1 is resident and its
    // routes come from the log.
    let mut r = Rig::cached(&bytes, "record", &plan(&[&[0, 1]], false));
    let start = Histogram {
        counts: vec![1; (tiny::LAYERS * e) as usize],
        ..Histogram::uniform(len, tiny::LAYERS, e)
    };
    let short = Histogram {
        counts: Vec::new(),
        ..Histogram::uniform(len, tiny::LAYERS, e)
    };
    for foreign in [Histogram::uniform(len, tiny::LAYERS, e + 1), short] {
        let err = r.model.record_routes(&r.s, path.clone(), foreign).err();
        assert_eq!(err, Some(KError::InvalidArg));
    }
    r.model.record_routes(&r.s, path.clone(), start).unwrap();
    serve(&mut r);
    drop(r);
    let h = Histogram::read(&path, len, tiny::LAYERS, e).unwrap();
    // 10 rows of 2 experts a layer, beside the starting 1 per expert; the resident layer's last
    // step (one row) is read at the next step, which never comes.
    let sum = |l: usize| h.counts[l * e as usize..][..e as usize].iter().sum::<u64>();
    assert_eq!(sum(0), 10 * 2 + u64::from(e), "{:?}", h.counts);
    assert_eq!(sum(1), 9 * 2 + u64::from(e), "{:?}", h.counts);
    assert!(h.counts[0] > h.counts[3], "skewed: {:?}", h.counts);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A whole-layer load counts its host layers and its resident ones and saves beside the model, so
/// the load after it plans per expert.
#[test]
fn a_whole_layer_load_records_for_the_next_plan() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let dir = std::env::temp_dir().join(format!("larch-cold-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = Histogram::path(&dir.join("tiny.gguf"));
    let (len, e) = (bytes.len() as u64, tiny::EXPERTS);
    // A streamed prefill step counts its host layers as the CPU path does.
    for stream_rows in [0, 1] {
        let place = Placement {
            cpu_expert_layers: 1,
            stream_rows,
        };
        let mut r = Rig::placed(&bytes, &format!("cold-{stream_rows}"), place);
        if stream_rows > 0 {
            assert!(r.model.alloc_stream_pool(&r.s).unwrap());
        }
        let start = Histogram::uniform(len, tiny::LAYERS, e);
        r.model.record_routes(&r.s, path.clone(), start).unwrap();
        serve(&mut r);
        assert_eq!(r.model.expert_hits(), None, "recording is not a split");
        assert_eq!(r.model.streamed_steps() > 0, stream_rows > 0);
        drop(r);
        let h = Histogram::read(&path, len, tiny::LAYERS, e).unwrap();
        let sum = |l: usize| h.counts[l * e as usize..][..e as usize].iter().sum::<u64>();
        // 10 rows of 2 experts; the last step (one row) is read at the next step, which never
        // comes, and its host counts wait with its resident ones.
        assert_eq!(sum(0), 9 * 2, "host layer, R {stream_rows}: {:?}", h.counts);
        assert_eq!(sum(1), 9 * 2, "resident, R {stream_rows}: {:?}", h.counts);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// One prompt and an end leaves no resident evidence, so no host evidence either: the file stays
/// all zero, which the next load reads as no histogram.
#[test]
fn a_step_never_read_back_counts_on_no_layer() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let dir = std::env::temp_dir().join(format!("larch-unread-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = Histogram::path(&dir.join("tiny.gguf"));
    let (len, e) = (bytes.len() as u64, tiny::EXPERTS);
    let place = Placement {
        cpu_expert_layers: 1,
        ..Default::default()
    };
    let mut r = Rig::placed(&bytes, "unread", place);
    let start = Histogram::uniform(len, tiny::LAYERS, e);
    r.model.record_routes(&r.s, path.clone(), start).unwrap();
    let rows: Vec<i32> = (0..7).collect();
    r.step(&[], &[(&[5, 9, 2, 200, 31, 7, 64], 0, &TABLE_A)], &rows);
    drop(r);
    let h = Histogram::read(&path, len, tiny::LAYERS, e).unwrap();
    assert!(h.counts.iter().all(|&n| n == 0), "{:?}", h.counts);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A serve is killed, never dropped, so a recording whole-layer load saves while it runs, well
/// before a split model's interval, with both kinds of layer in the file.
#[test]
fn a_cold_serve_saves_before_it_ends() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let dir = std::env::temp_dir().join(format!("larch-early-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = Histogram::path(&dir.join("tiny.gguf"));
    let (len, e) = (bytes.len() as u64, tiny::EXPERTS);
    let place = Placement {
        cpu_expert_layers: 1,
        ..Default::default()
    };
    let mut r = Rig::placed(&bytes, "early", place);
    let start = Histogram::uniform(len, tiny::LAYERS, e);
    r.model.record_routes(&r.s, path.clone(), start).unwrap();
    // 4 passes each.
    for _ in 0..3 {
        serve(&mut r);
    }
    let mut saved = None;
    for _ in 0..500 {
        saved = Histogram::read(&path, len, tiny::LAYERS, e);
        if saved.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let h = saved.expect("saved while serving");
    let sum = |l: usize| h.counts[l * e as usize..][..e as usize].iter().sum::<u64>();
    assert!(sum(0) > 0 && sum(1) > 0, "{:?}", h.counts);
    drop(r);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Recording alone is not an expert cache: adapters and suffix training stay as whole layers have
/// them, unlike a_cached_model_refuses_adapters_and_suffix_training.
#[test]
fn a_recording_load_keeps_adapters_and_suffix_training() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let (d, ef, rank) = (tiny::D as usize, tiny::EXPERT_FFN as usize, 4);
    let a = RouterDeltas {
        layers: vec![1],
        experts: tiny::EXPERTS,
        deltas: vec![vec![0.01; 2 * rank * (d + ef)]],
        rank: rank as u32,
        ids: vec![vec![0, 2]],
    };
    let dir = std::env::temp_dir().join(format!("larch-learn-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = Histogram::path(&dir.join("tiny.gguf"));
    let place = Placement {
        cpu_expert_layers: 1,
        ..Default::default()
    };
    let mut r = Rig::placed(&bytes, "learn", place);
    let start = Histogram::uniform(bytes.len() as u64, tiny::LAYERS, tiny::EXPERTS);
    r.model.record_routes(&r.s, path, start).unwrap();
    r.model.set_router_deltas(&r.s, Some(&a)).unwrap();
    assert!(DeviceSuffix::new(&r.model, &r.s, 1).is_ok());
    drop(r);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A row without a resident expert never runs one on the device: with the last layer's only
/// resident expert (3) NaN, a row routed elsewhere stays what it is alone beside a row routed to
/// 3, where a weight-0 copy of expert 3 would make it NaN.
#[test]
fn a_host_only_row_never_runs_a_resident_expert() {
    let blown = edit(Arch::Qwen3Moe, |_, t| {
        let e = t
            .iter_mut()
            .find(|e| e.0 == "blk.1.ffn_gate_exps.weight")
            .unwrap();
        // Each GGML type's block bytes and where its f16 scale sits.
        let (block, at) = match e.2 {
            2 => (18, 0),     // Q4_0
            6 => (22, 0),     // Q5_0
            8 => (34, 0),     // Q8_0
            12 => (144, 0),   // Q4_K
            13 => (176, 0),   // Q5_K
            14 => (210, 208), // Q6_K
            t => panic!("type {t}"),
        };
        let each = e.3.len() / tiny::EXPERTS as usize;
        for b in e.3[3 * each..].chunks_exact_mut(block) {
            b[at..at + 2].copy_from_slice(&0x7e00u16.to_le_bytes()); // an f16 NaN
        }
    });
    // The last layer's expert 3, resident, is NaN: no router reads it, so a row routed to it
    // returns NaN logits and a row that is not must stay finite in any batch.
    let mut r = Rig::cached(&blown, "blown", &plan(&[&[0, 1, 2, 3], &[3]], false));
    // Rows then route to 3 some of the time.
    let some3 = every_layer(&[0.0, 0.0, 0.0, -1.5]);
    r.model.set_router_deltas(&r.s, Some(&some3)).unwrap();
    let (a, b) = ([5, 9, 2, 200, 31, 7, 64], [1, 77, 140, 3, 250, 8, 19]);
    let prefill = |r: &mut Rig| {
        r.clear_kv();
        r.step(&[], &[(&a, 0, &TABLE_A)], &[6]);
        r.step(&[], &[(&b, 0, &TABLE_B)], &[6]);
    };
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let mut seen = [0; 4];
    for t in (0..40).map(|i| i * 6 + 1) {
        prefill(&mut r);
        let both = r.step(
            &[Some((t, 7, &TABLE_A)), Some((t + 1, 7, &TABLE_B))],
            &[],
            &[0, 1],
        );
        prefill(&mut r);
        let mut alone = r.step(&[Some((t, 7, &TABLE_A))], &[], &[0]);
        alone.extend(r.step(&[Some((t + 1, 7, &TABLE_B))], &[], &[0]));
        let ok: Vec<bool> = alone
            .chunks(V)
            .map(|row| row.iter().all(|x| x.is_finite()))
            .collect();
        seen[ok[0] as usize * 2 + ok[1] as usize] += 1;
        assert_eq!(bits(both), bits(alone), "token {t}");
    }
    // A batch of a NaN row beside a finite one happened, either way round.
    assert!(seen[1] + seen[2] > 0, "{seen:?}");
}

#[test]
fn a_saturated_count_stays_saturated() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let dir = std::env::temp_dir().join(format!("larch-saturate-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = Histogram::path(&dir.join("tiny.gguf"));
    let (len, e) = (bytes.len() as u64, tiny::EXPERTS);
    let mut r = Rig::cached(&bytes, "saturate", &plan(&[&[0]], false));
    let full = Histogram {
        counts: vec![u64::MAX; (tiny::LAYERS * e) as usize],
        ..Histogram::uniform(len, tiny::LAYERS, e)
    };
    r.model
        .record_routes(&r.s, path.clone(), full.clone())
        .unwrap();
    serve(&mut r);
    serve(&mut r);
    drop(r);
    assert_eq!(Histogram::read(&path, len, tiny::LAYERS, e), Some(full));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Four or more rows on one host expert run as that row alone does: CPU_COMPUTE groups a call's pairs
/// from 4 per expert, whose arithmetic differs from the per-pair path a lone row takes.
#[test]
fn many_rows_on_one_host_expert_match_the_row_alone() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let plans = [
        ("all host", plan(&[&[], &[]], false)),
        ("split", plan(&[&[0], &[0]], false)),
    ];
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let prompt = [5, 9, 2, 200, 31, 7, 64];
    let all: Vec<i32> = (0..8).collect();
    for (name, p) in plans {
        let mut r = Rig::cached(&bytes, name, &p);
        skewed(&mut r);
        for t in [17, 99, 3, 128, 64, 250] {
            r.clear_kv();
            r.step(&[], &[(&prompt, 0, &TABLE_A)], &[6]);
            let rows: Vec<Decode> = vec![Some((t, 7, &TABLE_A)); 8];
            let batch = r.step(&rows, &[], &all);
            r.clear_kv();
            r.step(&[], &[(&prompt, 0, &TABLE_A)], &[6]);
            let alone = r.step(&[Some((t, 7, &TABLE_A))], &[], &[0]);
            for (i, row) in batch.chunks(V).enumerate() {
                assert_eq!(bits(row), bits(&alone), "{name}: token {t}, row {i}");
            }
        }
    }
}

/// Adapters and suffix training refuse on any model with an expert cache, even on a resident
/// layer, so a later run's plan never takes away what this one served. Whole layers still take them.
#[test]
fn a_cached_model_refuses_adapters_and_suffix_training() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let (d, ef, rank) = (tiny::D as usize, tiny::EXPERT_FFN as usize, 4);
    let a = RouterDeltas {
        layers: vec![1],
        experts: tiny::EXPERTS,
        deltas: vec![vec![0.01; 2 * rank * (d + ef)]],
        rank: rank as u32,
        ids: vec![vec![0, 2]],
    };
    // Layer 1 resident, as a first run's uniform plan leaves a trained suffix.
    let mut r = Rig::cached(&bytes, "refuse", &plan(&[&[], &[0, 1, 2, 3]], false));
    let refused = r.model.set_router_deltas(&r.s, Some(&a)).err();
    assert_eq!(refused, Some(KError::Unsupported));
    assert!(matches!(
        DeviceSuffix::new(&r.model, &r.s, 1),
        Err(LearnError::Invalid(_))
    ));
    let mut whole = Rig::with(&bytes, "refuse-whole");
    whole.model.set_router_deltas(&whole.s, Some(&a)).unwrap();
}

/// A crafted expert_count sizes nothing: without expert tensors that hold that many experts the
/// plan has no histogram, and the load behind it refuses the file.
#[test]
fn a_crafted_expert_count_gets_no_histogram() {
    let arch = Arch::Qwen3Moe;
    let counts = |bytes: &[u8]| {
        let g = Gguf::parse(bytes).unwrap();
        let c = ModelConfig::from_gguf(&g).unwrap();
        let m = MoeCfg::from_gguf(&g, c.arch).unwrap().unwrap();
        let none = std::path::Path::new("/nonexistent/m.experts");
        Histogram::for_load(&g, &c, m, none, bytes.len() as u64).map(|h| h.counts.len())
    };
    let whole = (tiny::LAYERS * tiny::EXPERTS) as usize;
    assert_eq!(counts(&tiny::gguf(arch, 1)), Some(whole));
    for count in [1 << 20, u32::MAX] {
        let crafted = edit(arch, |kv, _| {
            set(kv, "qwen3moe.expert_count", Value::U32(count));
        });
        assert_eq!(counts(&crafted), None, "{count}");
        let place = Placement::default();
        assert!(matches!(
            load_placed(&crafted, "crafted", place).err(),
            Some(LoadError::BadGguf(..))
        ));
    }
    let gone = edit(arch, |_, t| {
        t.retain(|e| e.0 != "blk.1.ffn_down_exps.weight")
    });
    assert_eq!(counts(&gone), None);
}

/// Near-even counts move nothing: with every count at 1000, one more routing does not take a slot,
/// so dynamic residency hits as the frozen plan does.
#[test]
fn near_even_counts_do_not_swap_experts() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let dir = std::env::temp_dir().join(format!("larch-even-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = Histogram::path(&dir.join("tiny.gguf"));
    let (len, e) = (bytes.len() as u64, tiny::EXPERTS);
    let even = Histogram {
        counts: vec![1000; (tiny::LAYERS * e) as usize],
        ..Histogram::uniform(len, tiny::LAYERS, e)
    };
    let hits = |dynamic| {
        let mut r = Rig::cached(&bytes, "even", &plan(&[&[3, 2], &[3, 2]], dynamic));
        r.model
            .record_routes(&r.s, path.clone(), even.clone())
            .unwrap();
        serve(&mut r);
        serve(&mut r);
        r.model.expert_hits().unwrap()
    };
    assert_eq!(hits(true), hits(false));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The periodic save runs off the step, even on a plan with no route log: its temporary file is a
/// FIFO here, whose open blocks as a stalled disk would, and the steps go on regardless.
#[cfg(unix)]
#[test]
fn a_periodic_save_never_stalls_a_step() {
    use std::sync::mpsc::channel;
    use std::time::Duration;
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let dir = std::env::temp_dir().join(format!("larch-fifo-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = Histogram::path(&dir.join("tiny.gguf"));
    let tmp = format!("{}.{}.tmp", path.display(), std::process::id());
    let made = std::process::Command::new("mkfifo").arg(&tmp).status();
    assert!(made.unwrap().success());
    let (len, e) = (bytes.len() as u64, tiny::EXPERTS);
    let (stepped, steps_done) = channel();
    let (finish, finished) = channel::<()>();
    let hist = path.clone();
    let worker = std::thread::spawn(move || {
        let mut r = Rig::cached(&bytes, "fifo", &plan(&[&[0], &[1]], false));
        let start = Histogram::uniform(len, tiny::LAYERS, e);
        r.model.record_routes(&r.s, hist, start).unwrap();
        let prompt = [5, 9, 2, 200, 31, 7, 64];
        r.clear_kv();
        r.step(&[], &[(&prompt, 0, &TABLE_A)], &[6]);
        for _ in 0..256 {
            r.step(&[Some((17, 7, &TABLE_A))], &[], &[0]);
        }
        stepped.send(()).unwrap();
        finished.recv().unwrap();
    });
    steps_done
        .recv_timeout(Duration::from_secs(120))
        .expect("a step waited for the save");
    // The save got past the FIFO only now: a reader sees the whole file.
    let (read, got) = channel();
    let fifo = tmp.clone();
    std::thread::spawn(move || read.send(std::fs::read(fifo)));
    let saved = got.recv_timeout(Duration::from_secs(120)).expect("a save");
    assert_eq!(
        saved.unwrap().len() as u64,
        24 + 8 * u64::from(tiny::LAYERS * e)
    );
    // Still there unless this OS can fsync a FIFO: the drop's save needs it gone.
    std::fs::remove_file(&tmp).ok();
    finish.send(()).unwrap();
    worker.join().unwrap();
    let h = Histogram::read(&path, len, tiny::LAYERS, e).expect("the drop's save");
    assert!(h.counts.iter().sum::<u64>() > 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Expert tensors that share their bytes cannot size the histogram past the file: every (layer,
/// expert) pair needs bytes of its own, so a count that only aliased tensors hold plans nothing.
#[test]
fn aliased_expert_tensors_get_no_histogram() {
    let bytes = tiny::gguf(Arch::Qwen3Moe, 1);
    let mut g = Gguf::parse(&bytes).unwrap();
    let c = ModelConfig::from_gguf(&g).unwrap();
    let mut m = MoeCfg::from_gguf(&g, c.arch).unwrap().unwrap();
    let none = std::path::Path::new("/nonexistent/m.experts");
    let len = bytes.len() as u64;
    assert!(Histogram::for_load(&g, &c, m, none, len).is_some());
    // As many experts as one tensor's extent holds, every expert tensor at the same offset.
    let widest = g
        .tensors
        .iter()
        .filter(|t| t.name.contains("_exps"))
        .map(|t| t.bytes / u64::from(m.experts))
        .max()
        .unwrap();
    m.experts = (len / widest) as u32;
    for t in g.tensors.iter_mut().filter(|t| t.name.contains("_exps")) {
        let slice = t.bytes / u64::from(tiny::EXPERTS);
        (t.dims[2], t.bytes, t.offset) = (u64::from(m.experts), slice * u64::from(m.experts), 0);
    }
    assert!(
        Histogram::for_load(&g, &c, m, none, len).is_none(),
        "{}",
        m.experts
    );
}

/// Tensor `t`'s rows decoded to f32 by the CPU reference (embed).
fn decode_f32(g: &Gguf, file: &[u8], t: &gguf::TensorInfo) -> Vec<f32> {
    let (k, rows) = (
        t.dims[0] as i64,
        (t.dims.iter().product::<u64>() / t.dims[0]) as i64,
    );
    let dev = Device::open(CPU).unwrap();
    let s = Stream::new(&dev).unwrap();
    let w = Buffer::alloc(&dev, t.bytes).unwrap();
    let ids = Buffer::alloc(&dev, 4 * rows as u64).unwrap();
    let out = Buffer::alloc(&dev, 4 * (rows * k) as u64).unwrap();
    let at = (g.data_offset + t.offset) as usize;
    s.upload(&w, 0, &file[at..at + t.bytes as usize]).unwrap();
    s.upload(&ids, 0, &(0..rows as i32).collect::<Vec<_>>())
        .unwrap();
    let dtype = gguf::dtype(t.ggml_type).unwrap();
    let table = Tensor::new(&w, 0, dtype, &[rows, k]).unwrap();
    let idt = Tensor::new(&ids, 0, DType::I32, &[rows]).unwrap();
    let o = Tensor::new(&out, 0, DType::F32, &[rows, k]).unwrap();
    ops::embed(&s, &table, &idt, &o).unwrap();
    let mut y = vec![0f32; (rows * k) as usize];
    s.download(&out, 0, &mut y).unwrap();
    s.sync().unwrap();
    y
}

/// `larch repack` of the tiny MoE model's layer 1: its routed experts become Q2_K, each the
/// quantizer's bytes of the source tensor's CPU-reference decode; every other tensor, its type and
/// the metadata are the source's byte for byte, in a standard GGUF v3. The file loads and runs on
/// CPU_COMPUTE, one token per step, as the CPU reference's f32 prefill of the same file does. A
/// layer the file lacks is refused and writes nothing.
#[test]
fn repacked_expert_layers_load_and_run() {
    let dir = std::env::temp_dir().join(format!("larch-repack-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, dst) = (dir.join("src.gguf"), dir.join("dst.gguf"));
    let a = tiny::gguf(Arch::Qwen3Moe, 1);
    std::fs::write(&src, &a).unwrap();
    larch_model::q2k::repack(&src, &dst, &[1]).unwrap();
    let b = std::fs::read(&dst).unwrap();
    assert_eq!(&b[..8], b"GGUF\x03\0\0\0");
    let (ga, gb) = (Gguf::parse(&a).unwrap(), Gguf::parse(&b).unwrap());
    assert_eq!(ga.kv, gb.kv);
    assert_eq!(ga.tensors.len(), gb.tensors.len());
    let data = |g: &Gguf, f: &'static [u8], t: &gguf::TensorInfo| -> Vec<u8> {
        f[(g.data_offset + t.offset) as usize..][..t.bytes as usize].to_vec()
    };
    let (a, b): (&'static [u8], &'static [u8]) = (a.leak(), b.leak());
    let mut repacked = 0;
    for (ta, tb) in ga.tensors.iter().zip(&gb.tensors) {
        assert_eq!((&ta.name, &ta.dims), (&tb.name, &tb.dims));
        let n = &ta.name;
        if n.starts_with("blk.1.ffn_") && n.ends_with("_exps.weight") {
            assert_eq!(tb.ggml_type, 10, "{n}");
            let want = larch_model::q2k::quantize(&decode_f32(&ga, a, ta));
            assert!(data(&gb, b, tb) == want, "{n}");
            repacked += 1;
        } else {
            assert_eq!(tb.ggml_type, ta.ggml_type, "{n}");
            assert!(data(&gb, b, tb) == data(&ga, a, ta), "{n}");
        }
    }
    assert_eq!(repacked, 3);

    let prompt: Vec<i32> = (0..16)
        .map(|i| (i * 37 + 11) % tiny::VOCAB as i32)
        .collect();
    let mut reference = Rig::with(b, "repacked");
    reference.clear_kv();
    let all: Vec<i32> = (0..prompt.len() as i32).collect();
    let full = reference.step(&[], &[(&prompt, 0, &TABLE_A)], &all);
    let mut compute = Rig::loaded(
        load_by_on(b, "repacked-compute", CPU_COMPUTE, |p, d, s| {
            Transformer::load_placed(p, d, s, MAX_ROWS, Placement::default())
        })
        .unwrap(),
    );
    compute.clear_kv();
    let decoded: Vec<f32> = (0..prompt.len())
        .flat_map(|i| compute.step(&[Some((prompt[i], i as i32, &TABLE_A))], &[], &[0]))
        .collect();
    agree("repacked on CPU_COMPUTE", &full, &decoded);
    let mut original = Rig::new(Arch::Qwen3Moe);
    original.clear_kv();
    assert_ne!(full, original.step(&[], &[(&prompt, 0, &TABLE_A)], &all));

    let bad = dir.join("bad.gguf");
    assert!(larch_model::q2k::repack(&src, &bad, &[tiny::LAYERS]).is_err());
    assert!(!bad.exists());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `larch repack --cold-fraction 0.5`: in each layer the two experts the histogram counts least
/// (layer 0: 1 and 3; layer 1: 0 and 2, a tie broken by id) move to `ffn_*_cold_exps` as Q2_K, the
/// quantizer's bytes of their decode; the hot experts stay in `ffn_*_exps` byte for byte, in id
/// order; `larch.cold_experts.<layer>` lists the cold ids. The file loads and runs on CPU_COMPUTE,
/// one token per step, as the CPU reference's f32 prefill of the same file does.
#[test]
fn cold_experts_repack_beside_the_hot_ones() {
    let (a, b) = cold_tiny();
    let (ga, gb) = (Gguf::parse(a).unwrap(), Gguf::parse(b).unwrap());
    let cold = |l: u32| -> Vec<u64> {
        let v = gb
            .get(&format!("larch.cold_experts.{l}"))
            .unwrap()
            .as_arr()
            .unwrap();
        v.iter().map(|x| x.as_u64().unwrap()).collect()
    };
    assert_eq!((cold(0), cold(1)), (vec![1, 3], vec![0, 2]));
    assert_eq!(gb.kv.len(), ga.kv.len() + 2);
    let data = |g: &Gguf, f: &'static [u8], t: &gguf::TensorInfo| {
        &f[(g.data_offset + t.offset) as usize..][..t.bytes as usize]
    };
    for t in &ga.tensors {
        let hot = gb.tensor(&t.name).unwrap();
        let Some((l, p)) = (0..tiny::LAYERS)
            .flat_map(|l| ["gate", "up", "down"].map(|p| (l, p)))
            .find(|(l, p)| t.name == format!("blk.{l}.ffn_{p}_exps.weight"))
        else {
            assert_eq!(
                (hot.ggml_type, data(&gb, b, hot)),
                (t.ggml_type, data(&ga, a, t))
            );
            continue;
        };
        let c = gb
            .tensor(&format!("blk.{l}.ffn_{p}_cold_exps.weight"))
            .unwrap();
        let (k, n) = (t.dims[0], t.dims[1]);
        assert_eq!(
            (hot.dims.clone(), hot.ggml_type),
            (vec![k, n, 2], t.ggml_type)
        );
        assert_eq!((c.dims.clone(), c.ggml_type), (vec![k, n, 2], 10));
        let slice = (t.bytes / 4) as usize;
        let src = data(&ga, a, t);
        let (h, cl): (Vec<usize>, Vec<usize>) =
            (0..4).partition(|&e| !cold(l).contains(&(e as u64)));
        let want: Vec<u8> = h
            .iter()
            .flat_map(|&e| &src[e * slice..][..slice])
            .copied()
            .collect();
        assert!(data(&gb, b, hot) == want, "{}", t.name);
        let f = decode_f32(&ga, a, t);
        let per = (k * n) as usize;
        let x: Vec<f32> = cl
            .iter()
            .flat_map(|&e| &f[e * per..][..per])
            .copied()
            .collect();
        assert!(
            data(&gb, b, c) == larch_model::q2k::quantize(&x),
            "{}",
            c.name
        );
    }

    let prompt: Vec<i32> = (0..16)
        .map(|i| (i * 37 + 11) % tiny::VOCAB as i32)
        .collect();
    let mut reference = Rig::with(b, "cold");
    reference.clear_kv();
    let all: Vec<i32> = (0..prompt.len() as i32).collect();
    let full = reference.step(&[], &[(&prompt, 0, &TABLE_A)], &all);
    let mut compute = Rig::loaded(
        load_by_on(b, "cold-compute", CPU_COMPUTE, |p, d, s| {
            Transformer::load_placed(p, d, s, MAX_ROWS, Placement::default())
        })
        .unwrap(),
    );
    compute.clear_kv();
    let decoded: Vec<f32> = (0..prompt.len())
        .flat_map(|i| compute.step(&[Some((prompt[i], i as i32, &TABLE_A))], &[], &[0]))
        .collect();
    agree("cold experts on CPU_COMPUTE", &full, &decoded);
}

/// repack refuses what it cannot write faithfully, with no output: a layer already mixed, a layer
/// past the histogram, and metadata holding an empty array. A mixed file's stream pool slot holds a
/// layer's hot and cold tensors of one projection together.
#[test]
fn repack_refuses_what_it_cannot_copy() {
    let (a, b) = cold_tiny();
    let dir = std::env::temp_dir().join(format!("larch-refuse-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (mixed, plain, out) = (dir.join("m.gguf"), dir.join("p.gguf"), dir.join("o.gguf"));
    std::fs::write(&mixed, b).unwrap();
    std::fs::write(&plain, a).unwrap();
    let hist = |len: usize| Histogram::uniform(len as u64, tiny::LAYERS, tiny::EXPERTS);
    let q2k = larch_model::q2k::repack_cold;
    assert!(q2k(&mixed, &out, &[0], 0.5, &hist(b.len())).is_err());
    assert!(larch_model::q2k::repack(&mixed, &out, &[1]).is_err());
    let short = Histogram::uniform(a.len() as u64, 1, tiny::EXPERTS);
    assert!(q2k(&plain, &out, &[1], 0.5, &short).is_err());
    let empty = edit(Arch::Qwen3Moe, |kv, _| {
        set(kv, "general.tags", Value::Arr(vec![]))
    });
    std::fs::write(&plain, &empty).unwrap();
    assert!(larch_model::q2k::repack(&plain, &out, &[1]).is_err());
    assert!(!out.exists());
    std::fs::remove_dir_all(&dir).unwrap();
    let pool = |f: &[u8]| Transformer::stream_pool_bytes(&Gguf::parse(f).unwrap());
    let g = Gguf::parse(b).unwrap();
    let span = |n: &str| g.tensor(n).unwrap().bytes;
    let both = span("blk.0.ffn_gate_exps.weight") + span("blk.0.ffn_gate_cold_exps.weight");
    assert!(pool(a) > 0);
    assert!(pool(b) >= both + (32 << 20), "{} < {both}", pool(b));
}

/// repack never replaces a file: an output naming its own source, or any existing file, is refused
/// and both stay byte for byte; a source expert weight that is not finite (here an f16 NaN scale) is
/// refused with no output, as Q2_K has no code for it.
#[test]
fn repack_never_replaces_a_file_or_quantizes_a_non_finite_weight() {
    let dir = std::env::temp_dir().join(format!("larch-clobber-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, other, out) = (dir.join("s.gguf"), dir.join("x.gguf"), dir.join("o.gguf"));
    let a = tiny::gguf(Arch::Qwen3Moe, 1);
    std::fs::write(&src, &a).unwrap();
    std::fs::write(&other, b"not a model").unwrap();
    assert!(larch_model::q2k::repack(&src, &src, &[1]).is_err());
    assert!(larch_model::q2k::repack(&src, &other, &[1]).is_err());
    assert_eq!(std::fs::read(&src).unwrap(), a);
    assert_eq!(std::fs::read(&other).unwrap(), b"not a model");
    let nan = edit(Arch::Qwen3Moe, |_, t| {
        let e = t
            .iter_mut()
            .find(|e| e.0 == "blk.1.ffn_gate_exps.weight")
            .unwrap();
        let at = if e.2 == 14 { 208 } else { 0 }; // Q6_K keeps its f16 scale last, the rest first
        e.3[at..at + 2].copy_from_slice(&0x7e00u16.to_le_bytes());
    });
    std::fs::write(&src, &nan).unwrap();
    assert!(larch_model::q2k::repack(&src, &out, &[1]).is_err());
    assert!(!out.exists());
    larch_model::q2k::repack(&src, &out, &[0]).unwrap(); // layer 0 is finite
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The tiny MoE file and its `repack_cold` at 0.5: layer 0's experts 1 and 3 cold, layer 1's 0
/// and 2.
fn cold_tiny() -> (&'static [u8], &'static [u8]) {
    // Unique per call: parallel tests build it too.
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("larch-cold-{}-{call}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, dst) = (dir.join("src.gguf"), dir.join("dst.gguf"));
    let a: &'static [u8] = tiny::gguf(Arch::Qwen3Moe, 1).leak();
    std::fs::write(&src, a).unwrap();
    let hist = Histogram {
        counts: vec![9, 1, 7, 2, 3, 8, 3, 9],
        ..Histogram::uniform(a.len() as u64, tiny::LAYERS, tiny::EXPERTS)
    };
    larch_model::q2k::repack_cold(&src, &dst, &[0, 1], 0.5, &hist).unwrap();
    let b = std::fs::read(&dst).unwrap().leak();
    std::fs::remove_dir_all(&dir).unwrap();
    (a, b)
}

/// A mixed layer whose experts stay in host memory (its cold tensors and expert table on the CPU
/// device) computes what it does on the device, within mmvq_id's tolerance, as whole host layers do
/// (host_experts_compute_what_device_experts_do): the CPU device's hot experts sum in f32 through
/// its own mmvq_id.
/// Without its cold tensors, or with a cold list that is unsorted, repeated, out of range, empty or
/// every expert, or a list that disagrees with the tensors' shapes, the file is refused.
#[test]
fn mixed_layers_run_on_the_host_and_refuse_a_bad_list() {
    let (_, b) = cold_tiny();
    let run = |layers: u32| {
        let place = Placement {
            cpu_expert_layers: layers,
            ..Default::default()
        };
        let mut r = Rig::placed(b, &format!("cold-host-{layers}"), place);
        r.clear_kv();
        let rows: Vec<i32> = (0..7).collect();
        let mut got = r.step(&[], &[(&[5, 9, 2, 200, 31, 7, 64], 0, &TABLE_A)], &rows);
        got.extend(r.step(&[Some((17, 7, &TABLE_A))], &[], &[0]));
        got
    };
    let want = run(0);
    assert!(want.iter().all(|x| x.is_finite()));
    let max = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    for layers in [1, tiny::LAYERS] {
        for (i, (g, w)) in run(layers).iter().zip(&want).enumerate() {
            let tol = 1e-3 * w.abs() + 1e-3 * max;
            assert!(
                (g - w).abs() <= tol,
                "{layers} host layers: [{i}] {g} vs {w}"
            );
        }
    }

    let key = larch_model::q2k::cold_key(0);
    let list = |ids: &[u32]| Value::Arr(ids.iter().map(|&i| Value::U32(i)).collect());
    let mut bad: Vec<(Vec<u8>, LoadError)> = [&[3, 1][..], &[1, 1], &[1, 4], &[], &[0, 1, 2, 3]]
        .iter()
        .map(|ids| {
            let f = edit_bytes(b, |kv, _| set(kv, &key, list(ids)));
            (f, LoadError::BadGguf(code::KEY, 0))
        })
        .collect();
    let one = edit_bytes(b, |kv, _| set(kv, &key, list(&[1])));
    let at = Gguf::parse(&one).unwrap();
    let t = at.tensor("blk.0.ffn_gate_exps.weight").unwrap();
    bad.push((
        one.clone(),
        LoadError::BadGguf(code::TENSOR, at.data_offset + t.offset),
    ));
    let gone = edit_bytes(b, |_, t| t.retain(|e| !e.0.contains("_cold_exps")));
    bad.push((gone, LoadError::BadGguf(code::MISSING_TENSOR, 0)));
    for (i, (f, want)) in bad.into_iter().enumerate() {
        assert_eq!(
            load(&f, &format!("bad-cold-{i}")).err(),
            Some(want),
            "case {i}"
        );
    }
}

/// The tiny MoE model repacked with whole Q2_K layers and with cold experts runs on GPU 0, a
/// prefill and one token per step, as the CPU reference's f32 prefill of the same file does:
///   LARCH_BACKEND=CUDA cargo test --release -p larch-model --test tiny -- --ignored repacked_tiny
#[test]
#[ignore = "needs a GPU"]
fn repacked_tiny_runs_on_the_gpu_as_on_the_cpu_reference() {
    let dir = std::env::temp_dir().join(format!("larch-repack-gpu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, whole) = (dir.join("src.gguf"), dir.join("whole.gguf"));
    std::fs::write(&src, tiny::gguf(Arch::Qwen3Moe, 1)).unwrap();
    larch_model::q2k::repack(&src, &whole, &[0, 1]).unwrap();
    let files = [
        (
            "whole",
            std::fs::read(&whole).unwrap().leak() as &'static [u8],
        ),
        ("cold", cold_tiny().1),
    ];
    std::fs::remove_dir_all(&dir).unwrap();
    let prompt: Vec<i32> = (0..16)
        .map(|i| (i * 37 + 11) % tiny::VOCAB as i32)
        .collect();
    let all: Vec<i32> = (0..prompt.len() as i32).collect();
    for (name, b) in files {
        let mut cpu = Rig::with(b, name);
        cpu.clear_kv();
        let want = cpu.step(&[], &[(&prompt, 0, &TABLE_A)], &all);
        let mut gpu = Rig::loaded(
            load_by_on(b, &format!("{name}-gpu"), 0, |p, d, s| {
                Transformer::load_placed(p, d, s, MAX_ROWS, Placement::default())
            })
            .unwrap(),
        );
        gpu.clear_kv();
        agree(
            &format!("{name} gpu prefill"),
            &want,
            &gpu.step(&[], &[(&prompt, 0, &TABLE_A)], &all),
        );
        gpu.clear_kv();
        let decoded: Vec<f32> = (0..prompt.len())
            .flat_map(|i| gpu.step(&[Some((prompt[i], i as i32, &TABLE_A))], &[], &[0]))
            .collect();
        agree(&format!("{name} gpu decode"), &want, &decoded);
    }
    // A Q2_K matrix outside the experts needs mmvq, which CUDA lacks for Q2_K: refused at load.
    let dense = edit(Arch::Qwen3Moe, |_, t| {
        let out = t.iter_mut().find(|e| e.0 == "output.weight").unwrap();
        (out.2, out.3) = (
            10,
            vec![0; (out.1.iter().product::<u64>() / 256 * 84) as usize],
        );
    });
    let err = load_by_on(&dense, "dense-q2k-gpu", 0, |p, d, s| {
        Transformer::load_placed(p, d, s, MAX_ROWS, Placement::default())
    })
    .err();
    assert_eq!(err, Some(LoadError::UnsupportedQuant(10)));
}
