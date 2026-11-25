//! Decode graph replay (9.5 step 2) on the device `LARCH_TEST_DEVICE` names (an ordinal; unset is
//! the CPU reference, as in CI): graphs on and off sample the same tokens and teacher-forced tops,
//! also after a forced run regrew the buffers the graphs hold; steady-state decode retains no extra
//! device memory. Where the device has no graphs, `decode_graphs` stays unsupported. Score requests'
//! device gather and their rows through padded replays match the CPU reference. Tests serialize
//! their engines so the free-memory snapshots cannot race another test's allocations.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use larch_engine::{
    EngineEvent, EngineHandle, EngineOptions, Forced, Loader, ReloadChange, Request, RequestHandle,
    start, start_with,
};
use larch_kernels::{Buffer, CPU, DType, Device, DeviceProps, KError, KvPool, Stream, Tensor, ops};
use larch_model::{Arch, FakeModel, Model, ModelConfig, StepBatch, Transformer, tiny};
use larch_records::{
    Authority, Backend, EngineError, EngineSample, HealthState, KnobReply, KnobRequest, MState,
    Outcome, Phase, Read, ReadFrom, Record, RejectReason, RevertTarget, knob_id,
};

static DEVICE_TEST: Mutex<()> = Mutex::new(());

/// The request's next event past its prefill progress.
fn first_event(h: &RequestHandle) -> EngineEvent {
    loop {
        match h.events.recv_timeout(Duration::from_secs(10)).unwrap() {
            EngineEvent::Prefilled { .. } => {}
            e => return e,
        }
    }
}

fn device() -> i32 {
    std::env::var("LARCH_TEST_DEVICE").map_or(CPU, |v| {
        v.parse().expect("LARCH_TEST_DEVICE is a device ordinal")
    })
}

/// The tiny llama model, written once per test binary.
fn tiny_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let p = std::env::temp_dir().join(format!("larch-graphs-tiny-{}.gguf", std::process::id()));
        std::fs::write(&p, tiny::gguf(Arch::Llama, 3)).unwrap();
        p
    })
    .clone()
}

/// The same generated weights with two HD128 query heads and one KV head. Projection widths stay
/// 256/128, and the loader derives a consistent 128-dimensional rotary embedding from the metadata.
fn tiny_hd128_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let mut bytes = tiny::gguf(Arch::Llama, 3);
        for (key, before, after) in [
            ("llama.attention.head_count", 4u32, 2u32),
            ("llama.attention.head_count_kv", 2u32, 1u32),
        ] {
            let mut tag = (key.len() as u64).to_le_bytes().to_vec();
            tag.extend_from_slice(key.as_bytes());
            tag.extend_from_slice(&4u32.to_le_bytes()); // GGUF UINT32 metadata value.
            let matches: Vec<_> = bytes
                .windows(tag.len())
                .enumerate()
                .filter_map(|(i, value)| (value == tag).then_some(i + tag.len()))
                .collect();
            assert_eq!(matches.len(), 1, "generated metadata key {key}");
            let at = matches[0];
            assert_eq!(&bytes[at..at + 4], &before.to_le_bytes());
            bytes[at..at + 4].copy_from_slice(&after.to_le_bytes());
        }
        let g = larch_model::gguf::Gguf::read(bytes.as_slice(), bytes.len() as u64).unwrap();
        let config = ModelConfig::from_gguf(&g).unwrap();
        assert_eq!(
            (config.d_model, config.n_heads, config.n_kv_heads),
            (256, 2, 1)
        );
        assert_eq!(
            (config.head_dim, config.rope.n_dims, config.n_layers),
            (128, 128, 2)
        );
        let p =
            std::env::temp_dir().join(format!("larch-graphs-hd128-{}.gguf", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p
    })
    .clone()
}

/// The tiny model behind a gate the test holds while it submits, so a mix's requests are all in by
/// its second step. A replay never reaches it.
struct Gated(
    Transformer,
    Arc<Mutex<()>>,
    Option<(u32, SyncSender<()>, Receiver<()>)>,
);

impl Model for Gated {
    fn config(&self) -> &ModelConfig {
        self.0.config()
    }

    fn forward(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        drop(self.1.lock().unwrap());
        if b.prefill_rows == 0
            && self
                .2
                .as_ref()
                .is_some_and(|(rows, _, _)| *rows == b.decode_rows)
            && let Some((_, entered, release)) = self.2.take()
        {
            entered.send(()).unwrap();
            release.recv_timeout(Duration::from_secs(10)).unwrap();
        }
        self.0.forward(s, b, kv, logits)
    }
}

struct Run {
    e: EngineHandle,
    gate: Arc<Mutex<()>>,
}

/// An engine on the test device with up to 32 decode rows, graphs on or off. Its pool gets 0.30 (the
/// registry's least) of what is free, so a GPU keeps room for the next test's engine.
fn engine(graphs: bool) -> Run {
    engine_with_decode_gate(graphs, None)
}

fn engine_with_decode_gate(
    graphs: bool,
    decode_gate: Option<(SyncSender<()>, Receiver<()>)>,
) -> Run {
    engine_with_decode_gate_at_path(graphs, decode_gate, tiny_path())
}

fn engine_with_decode_gate_at_path(
    graphs: bool,
    decode_gate: Option<(SyncSender<()>, Receiver<()>)>,
    path: PathBuf,
) -> Run {
    let gate = Arc::new(Mutex::new(()));
    let (g, load_path) = (gate.clone(), path.clone());
    let load: Loader = Box::new(move |dev, s, rows| {
        Ok(Box::new(Gated(
            Transformer::load(&load_path, dev, s, rows)?,
            g,
            decode_gate.map(|(entered, release)| (32, entered, release)),
        )) as Box<dyn Model>)
    });
    let mut o = EngineOptions::new(path, device());
    o.kv_pool_fraction = 0.3;
    o.knob_overrides = vec![
        (knob_id::MAX_BATCH_SEQS, 32.0),
        (knob_id::DECODE_GRAPHS, f64::from(u8::from(graphs))),
    ];
    let e = start_with(o, load).unwrap();
    let flags = Device::open(device()).unwrap().props().unwrap().flags;
    let can = flags & DeviceProps::CAP_GRAPHS != 0;
    assert_eq!(e.capabilities().graph_replay, can);
    let refused = e.rejected_overrides();
    if graphs && !can {
        assert_eq!(
            refused,
            [(knob_id::DECODE_GRAPHS, 1.0, RejectReason::Unsupported)]
        );
    } else {
        assert!(refused.is_empty(), "{refused:?}");
    }
    Run { e, gate }
}

/// 32 greedy requests decoding together and finishing about one a step, so the decode steps pass
/// through every bucket from 32 down to 1: their tokens.
fn mix(r: &Run) -> Vec<Vec<u32>> {
    turnover(r, false)
}

fn turnover(r: &Run, cancel_tail: bool) -> Vec<Vec<u32>> {
    let gate = r.gate.lock().unwrap();
    // One engine turn: CUDA results depend on batch shape, so a racy first step flips near ties.
    let hs =
        r.e.submit_group(
            (0..32u32)
                .map(|i| Request::new(vec![i + 1; i as usize % 5 + 1], i + 3))
                .collect(),
        )
        .unwrap();
    let hs: Vec<_> = hs
        .into_iter()
        .enumerate()
        .filter_map(|(i, h)| {
            if cancel_tail && i >= 24 {
                drop(h);
                None
            } else {
                Some((i, h))
            }
        })
        .collect();
    drop(gate);
    hs.into_iter()
        .map(|(i, h)| {
            let mut t = Vec::new();
            for ev in h.events.iter() {
                match ev {
                    EngineEvent::Token { id, .. } => t.push(id),
                    EngineEvent::Done { .. } => {
                        assert_eq!(t.len(), i + 3, "request {i} returned incomplete tokens");
                        return t;
                    }
                    EngineEvent::Prefilled { .. } => {}
                    ev => panic!("{ev:?}"),
                }
            }
            panic!("no Done")
        })
        .collect()
}

/// The top token after each position of two teacher-forced runs: an 80-token prefill that grows past
/// the 64 serving rows, then decode; and prompt 1, every later position a forced decode step (the
/// parity decode path).
fn forced(e: &EngineHandle) -> Vec<Vec<u32>> {
    [(100, 80), (48, 1)]
        .map(|(end, prompt)| {
            let f = e
                .force(Forced {
                    tokens: (1..end).collect(),
                    prompt,
                    session: 0,
                    keep_kv: false,
                })
                .unwrap();
            let done = f.request.events.recv().unwrap();
            assert!(matches!(done, EngineEvent::Done { .. }), "{done:?}");
            let tops: Vec<_> = f.positions.try_iter().map(|p| p.top[0].0).collect();
            assert_eq!(
                tops.len(),
                end as usize - 1,
                "prompt {prompt}: incomplete positions"
            );
            tops
        })
        .to_vec()
}

/// A four-token greedy request from token 2: its tokens and the event that ended it.
fn run_four(e: &EngineHandle) -> (Vec<u32>, Option<EngineEvent>) {
    let h = e.submit(Request::new(vec![2], 4)).unwrap();
    let mut tokens = Vec::new();
    for event in h.events.iter() {
        match event {
            EngineEvent::Token { id, .. } => tokens.push(id),
            EngineEvent::Prefilled { .. } => {}
            event => return (tokens, Some(event)),
        }
    }
    (tokens, None)
}

/// Join the engine so the last Done's decode sample is published before reading the ring.
fn decode_samples(e: EngineHandle) -> Vec<EngineSample> {
    let mut r = e.ring().reader(ReadFrom::Oldest);
    e.shutdown();
    let mut out = Vec::new();
    while let Read::Record(rec) = r.next() {
        if let Record::EngineSample(_, s) = rec
            && s.phase() == Phase::Decode
        {
            out.push(s);
        }
    }
    out
}

/// Graphs on and off sample the same tokens and forced tops, also after the forced run regrew the
/// step buffers the graphs held; every pure decode step replays where the device captures graphs
/// (CUDA's exact 9 to 32 rows too), and elsewhere none does.
#[test]
fn graphs_on_and_off_agree() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let off = engine(false);
    let (tokens, tops) = (mix(&off), forced(&off.e));
    assert!(
        decode_samples(off.e)
            .iter()
            .all(|s| { s.graph_replayed == 0 && s.graph_captured() == Some(false) })
    );

    let on = engine(true);
    assert_eq!(mix(&on), tokens);
    assert_eq!(forced(&on.e), tops);
    assert_eq!(mix(&on), tokens);
    assert_eq!(on.e.health().state, HealthState::Ready);
    let can = u8::from(on.e.capabilities().graph_replay);
    let backend = on.e.capabilities().backend;
    let samples = decode_samples(on.e);
    assert!(!samples.is_empty());
    assert!(samples.iter().all(|s| s.graph_captured().is_some()));
    assert!(
        samples
            .iter()
            .all(|s| { s.graph_captured() != Some(true) || s.graph_replayed != 0 })
    );
    assert!(
        samples.iter().all(|s| s.graph_replayed == can),
        "replayed {can} expected"
    );
    if can != 0 && backend == Backend::Cuda {
        assert!(samples.iter().any(|s| s.graph_captured() == Some(true)));
        for rows in 1..=32 {
            assert!(
                samples
                    .iter()
                    .any(|s| s.n_decode_seqs == rows && s.graph_replayed == 1)
            );
        }
    }
}

/// A hybrid (Gated DeltaNet) model: graphs on and off sample the same tokens through every bucket,
/// padding rows on state slots no sequence holds; every pure decode step replays where the device
/// captures graphs, and elsewhere none does.
#[test]
fn hybrid_graphs_on_and_off_agree() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let path =
        std::env::temp_dir().join(format!("larch-graphs-hybrid-{}.gguf", std::process::id()));
    std::fs::write(&path, tiny::gguf(Arch::Qwen35, 3)).unwrap();
    let run = |graphs: bool| {
        let mut o = EngineOptions::new(path.clone(), device());
        o.kv_pool_fraction = 0.3;
        o.knob_overrides = vec![
            (knob_id::MAX_BATCH_SEQS, 32.0),
            (knob_id::DECODE_GRAPHS, f64::from(u8::from(graphs))),
        ];
        let r = Run {
            e: start(o).unwrap(),
            gate: Arc::new(Mutex::new(())),
        };
        let tokens = mix(&r);
        let can = u8::from(r.e.capabilities().graph_replay);
        (tokens, can, decode_samples(r.e))
    };
    let (off, _, _) = run(false);
    let (on, can, samples) = run(true);
    std::fs::remove_file(&path).unwrap();
    assert_eq!(on, off);
    assert!(!samples.is_empty());
    assert!(
        samples.iter().all(|s| s.graph_replayed == can),
        "replayed {can} expected"
    );
}

/// A tiny DFlash drafter guessing: with graphs on and off, the same greedy tokens
/// at bucket sizes that never pad (one and two sequences, so eager and replay run the same shapes
/// and the drafter reads the same taps); decode and check steps replay where the device captures.
/// LARCH_TEST_DEVICE=<ordinal> cargo test --release -p larch-engine --test graphs -- --ignored
/// a_drafter_replays
#[test]
#[ignore = "needs a device that captures graphs: LARCH_TEST_DEVICE"]
fn a_drafter_replays_decode_and_check_steps_as_eager() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let draft =
        std::env::temp_dir().join(format!("larch-graphs-dflash-{}.gguf", std::process::id()));
    let hybrid = std::env::temp_dir().join(format!(
        "larch-graphs-dflash-hybrid-{}.gguf",
        std::process::id()
    ));
    std::fs::write(&draft, tiny::dflash_gguf(7)).unwrap();
    std::fs::write(&hybrid, tiny::gguf(Arch::Qwen35, 3)).unwrap();
    let run = |model: &PathBuf, graphs: bool, seqs: u32| {
        let mut o = EngineOptions::new(model.clone(), device());
        o.kv_pool_fraction = 0.3;
        o.knob_overrides = vec![
            (knob_id::MAX_BATCH_SEQS, 8.0),
            (knob_id::DECODE_GRAPHS, f64::from(u8::from(graphs))),
        ];
        (o.drafter, o.drafter_guesses) = (Some(draft.clone()), 4);
        let e = start(o).unwrap();
        let hs = (e.submit_group(
            (0..seqs)
                .map(|i| Request::new(vec![i + 2; 7], 96))
                .collect(),
        ))
        .unwrap();
        let tokens: Vec<Vec<u32>> = hs
            .into_iter()
            .map(|h| {
                let mut t = Vec::new();
                for ev in h.events.iter() {
                    match ev {
                        EngineEvent::Token { id, .. } => t.push(id),
                        EngineEvent::Done { .. } => return t,
                        EngineEvent::Prefilled { .. } => {}
                        ev => panic!("{ev:?}"),
                    }
                }
                panic!("no Done")
            })
            .collect();
        let stats = e.guess_stats();
        let mut r = e.ring().reader(ReadFrom::Oldest);
        e.shutdown();
        let mut samples = Vec::new();
        while let Read::Record(rec) = r.next() {
            if let Record::EngineSample(_, s) = rec {
                samples.push(s);
            }
        }
        (tokens, stats, samples)
    };
    // The Qwen 3.5 rows are the hybrid engine path: a guessing drafter reaches `route_table`.
    for (model, seqs) in [
        (tiny_path(), 1),
        (tiny_path(), 2),
        (hybrid.clone(), 1),
        (hybrid.clone(), 2),
    ] {
        let (tokens, stats, _) = run(&model, false, seqs);
        let (replayed, replay_stats, samples) = run(&model, true, seqs);
        assert_eq!(replayed, tokens, "{model:?} {seqs} sequences");
        // Tokens only: with measured k every GuessStats figure (rounds and kept too) follows the
        // draft length, which follows real step times that differ between replay and eager.
        for st in [&stats, &replay_stats] {
            assert!(st.rounds > 0, "{model:?} {seqs} sequences: no check ran");
        }
        // A sample counts a check step as decode (`StepPlan::counts`), so these are the decode and
        // check steps together. Only a check shape's first hit runs eager (second-hit admission);
        // a plain bucket captures on its first. Were no check to replay, every check step would be
        // eager, at least `rounds / seqs` of them; were no plain step to, the replays would be the
        // check steps alone, at most `rounds`.
        let steps: Vec<_> = samples.iter().filter(|s| s.n_prefill_tokens == 0).collect();
        let eager = steps.iter().filter(|s| s.graph_replayed == 0).count();
        let replayed = steps.len() - eager;
        // The samples are the replay run's, so are its rounds.
        let rounds = replay_stats.rounds as usize;
        assert!(
            eager < rounds / seqs as usize,
            "{model:?} {seqs} sequences: {} eager steps, {rounds} check rounds: check steps did not replay",
            eager
        );
        assert!(
            replayed > rounds,
            "{model:?} {seqs} sequences: {} replays, {rounds} check rounds: plain steps did not replay",
            replayed
        );
    }
    std::fs::remove_file(&draft).unwrap();
    std::fs::remove_file(&hybrid).unwrap();
}

/// Captures for changing batches remain correct after split changes and a KV/model reload.
#[test]
fn changing_batches_survive_split_changes_and_reload() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let off = engine(false);
    let tokens = mix(&off);
    let tops = forced(&off.e);
    off.e.shutdown();
    let on = engine(true);
    assert_eq!(mix(&on), tokens);
    let reply = on.e.knobs().set(KnobRequest::new(
        knob_id::ATTN_DECODE_SPLITS,
        2.0,
        Authority::Human,
    ));
    if on.e.capabilities().backend == Backend::Cpu {
        assert_eq!(
            reply,
            KnobReply::Rejected {
                reason: RejectReason::Unsupported
            }
        );
    } else {
        assert!(
            matches!(reply, KnobReply::Applied { effective: 2.0, .. }),
            "{reply:?}"
        );
    }
    assert_eq!(mix(&on), tokens);
    on.e.reload(ReloadChange::default()).unwrap();
    assert_eq!(mix(&on), tokens);
    assert_eq!(forced(&on.e), tops);
    assert_eq!(on.e.health().state, HealthState::Ready);
    let can = u8::from(on.e.capabilities().graph_replay);
    assert!(decode_samples(on.e).iter().all(|s| s.graph_replayed == can));
}

fn mixed_full_chunk_campaign(
    graphs: bool,
    chunk: usize,
    max_seqs: u32,
    path: PathBuf,
) -> (Vec<Vec<u32>>, Vec<EngineSample>) {
    let gate = Arc::new(Mutex::new(()));
    let (entered_tx, entered_rx) = sync_channel(1);
    let (release_tx, release_rx) = sync_channel(1);
    let (g, load_path) = (gate.clone(), path.clone());
    let load: Loader = Box::new(move |dev, stream, rows| {
        let first_decode_rows =
            if graphs && dev.props().unwrap().flags & DeviceProps::CAP_GRAPHS != 0 {
                8
            } else {
                7
            };
        Ok(Box::new(Gated(
            Transformer::load(&load_path, dev, stream, rows)?,
            g,
            Some((first_decode_rows, entered_tx, release_rx)),
        )) as Box<dyn Model>)
    });
    let mut opts = EngineOptions::new(path, device());
    opts.kv_pool_fraction = 0.3;
    opts.knob_overrides = vec![
        (knob_id::MAX_BATCH_SEQS, f64::from(max_seqs)),
        (knob_id::PREFILL_CHUNK_TOKENS, chunk as f64),
        (knob_id::DECODE_GRAPHS, f64::from(u8::from(graphs))),
    ];
    let e = start_with(opts, load).unwrap();
    let held = gate.lock().unwrap();
    // One engine turn, as in `turnover`.
    let mut handles = e
        .submit_group((0..7).map(|i| Request::new(vec![i + 1], 64)).collect())
        .unwrap();
    drop(held);
    let mut outputs = vec![Vec::new(); 7];
    for (h, tokens) in handles.iter().zip(&mut outputs) {
        match first_event(h) {
            EngineEvent::Token { id, .. } => tokens.push(id),
            event => panic!("{event:?}"),
        }
    }
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    // Three identical full-chunk prompt shapes exercise both intermediate and final logits keys.
    for i in 0..3 {
        handles.push(e.submit(Request::new(vec![i + 11; chunk * 3], 1)).unwrap());
        outputs.push(Vec::new());
    }
    release_tx.send(()).unwrap();
    for (index, h) in handles.into_iter().enumerate() {
        loop {
            match h.events.recv_timeout(Duration::from_secs(10)).unwrap() {
                EngineEvent::Token { id, .. } => outputs[index].push(id),
                EngineEvent::Done { usage, .. } => {
                    assert_eq!(usage.completion_tokens as usize, outputs[index].len());
                    assert_eq!(outputs[index].len(), if index < 7 { 64 } else { 1 });
                    break;
                }
                EngineEvent::Prefilled { .. } => {}
                event => panic!("{event:?}"),
            }
        }
    }
    let mut reader = e.ring().reader(ReadFrom::Oldest);
    // Reload destroys all retained graphs before replacing KV/model/buffers; a short follow-up
    // also exercises admission and KV reuse on the replacement.
    e.reload(ReloadChange::default()).unwrap();
    let h = e.submit(Request::new(vec![3; chunk + 1], 3)).unwrap();
    let mut reused = Vec::new();
    loop {
        match h.events.recv_timeout(Duration::from_secs(10)).unwrap() {
            EngineEvent::Token { id, .. } => reused.push(id),
            EngineEvent::Done { .. } => break,
            EngineEvent::Prefilled { .. } => {}
            event => panic!("{event:?}"),
        }
    }
    assert_eq!(reused.len(), 3);
    outputs.push(reused);
    e.shutdown();
    let mut samples = Vec::new();
    while let Read::Record(record) = reader.next() {
        if let Record::EngineSample(_, sample) = record {
            samples.push(sample);
        }
    }
    (outputs, samples)
}

/// Repeated full chunks keep exact ragged metadata and logits offsets across mixed captures.
/// Run with LARCH_MIXED_GRAPHS=1 to exercise the opt-in CUDA path; CPU checks eager parity.
#[test]
fn bounded_mixed_full_chunks_match_eager_and_reuse_after_reload() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let diagnostic = larch_kernels::routes().mixed_graphs;
    let props = Device::open(device()).unwrap().props().unwrap();
    let captures = diagnostic
        && props.backend == larch_kernels::Backend::Cuda
        && props.flags & DeviceProps::CAP_GRAPHS != 0;
    for (chunk, max_seqs) in [(64, 8), (128, 16)] {
        let (expected, eager) = mixed_full_chunk_campaign(false, chunk, max_seqs, tiny_path());
        assert!(eager.iter().all(|s| s.graph_replayed == 0));
        let (actual, samples) = mixed_full_chunk_campaign(true, chunk, max_seqs, tiny_path());
        assert_eq!(actual, expected);
        let mixed: Vec<_> = samples
            .iter()
            .filter(|s| {
                s.n_decode_seqs == 7 && s.n_prefill_seqs == 1 && s.n_prefill_tokens == chunk as u32
            })
            .collect();
        assert!(mixed.len() >= 9, "missing actual mixed chunk steps");
        assert_eq!(
            mixed
                .iter()
                .filter(|s| s.graph_captured() == Some(true))
                .count(),
            if captures { 2 } else { 0 }
        );
        if captures {
            assert_eq!(mixed[0].graph_replayed, 0);
            assert!(mixed.iter().filter(|s| s.graph_replayed != 0).count() >= 7);
        } else {
            assert!(mixed.iter().all(|s| s.graph_replayed == 0));
        }
    }
}

/// Cancelled rows leave no stale metadata when their KV blocks are reused in another changing batch.
#[test]
fn cancelled_batches_leave_reusable_graph_inputs() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let off = engine(false);
    let cancelled = turnover(&off, true);
    off.e.shutdown();
    assert_eq!(cancelled.len(), 24);
    let on = engine(true);
    let tokens = mix(&on);
    assert_eq!(turnover(&on, true), cancelled);
    assert_eq!(mix(&on), tokens);
    let mut reader = on.e.ring().reader(ReadFrom::Oldest);
    let mut cancellations = 0;
    while let Read::Record(rec) = reader.next() {
        if let Record::RequestSpan(_, span) = rec {
            cancellations += usize::from(span.outcome() == Outcome::Cancelled);
        }
    }
    assert_eq!(cancellations, 8);
    let can = u8::from(on.e.capabilities().graph_replay);
    assert!(decode_samples(on.e).iter().all(|s| s.graph_replayed == can));
}

fn active_cancellation_campaign(graphs: bool, path: PathBuf) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let (entered_tx, entered) = sync_channel(1);
    let (release, held) = sync_channel(1);
    let run = engine_with_decode_gate_at_path(graphs, Some((entered_tx, held)), path);
    let can = run.e.capabilities().graph_replay;
    let gate = run.gate.lock().unwrap();
    // One engine turn, as in `turnover`.
    let handles = run
        .e
        .submit_group(
            (0..32u32)
                .map(|i| Request::new(vec![i + 1; i as usize % 5 + 1], 128))
                .collect(),
        )
        .unwrap();
    let cancelled_ids: Vec<_> = handles[24..].iter().map(|h| h.id).collect();
    drop(gate);
    let mut tokens = vec![Vec::new(); 32];
    for (i, h) in handles.iter().enumerate() {
        let event = first_event(h);
        let EngineEvent::Token { id, .. } = event else {
            panic!("{event:?}");
        };
        tokens[i].push(id);
    }
    entered.recv_timeout(Duration::from_secs(10)).unwrap();
    for h in &handles[24..] {
        h.cancel();
    }
    release.send(()).unwrap();
    for (i, h) in handles.into_iter().enumerate() {
        loop {
            match h.events.recv_timeout(Duration::from_secs(10)).unwrap() {
                EngineEvent::Token { id, .. } => tokens[i].push(id),
                EngineEvent::Done { usage, .. } if i < 24 => {
                    assert_eq!(usage.completion_tokens, 128);
                    assert_eq!(tokens[i].len(), 128);
                    break;
                }
                EngineEvent::Error(EngineError::Cancelled) if i >= 24 => {
                    assert!((1..128).contains(&tokens[i].len()));
                    break;
                }
                EngineEvent::Prefilled { .. } => {}
                event => panic!("request {i}: {event:?}"),
            }
        }
    }
    let mut reader = run.e.ring().reader(ReadFrom::Oldest);
    let mut cancelled = 0;
    let mut decode32 = false;
    let mut decode24 = false;
    while let Read::Record(record) = reader.next() {
        match record {
            Record::RequestSpan(_, span) if cancelled_ids.contains(&span.req_id) => {
                assert_eq!(span.outcome(), Outcome::Cancelled);
                assert!((1..128).contains(&span.gen_tokens));
                let index = 24
                    + cancelled_ids
                        .iter()
                        .position(|id| *id == span.req_id)
                        .unwrap();
                assert_eq!(span.gen_tokens as usize, tokens[index].len());
                cancelled += 1;
            }
            Record::EngineSample(_, sample)
                if sample.phase() == Phase::Decode && sample.n_decode_seqs == 32 =>
            {
                assert_eq!(sample.graph_replayed, u8::from(graphs && can));
                assert_eq!(sample.graph_captured(), Some(graphs && can));
                decode32 = true;
            }
            Record::EngineSample(_, sample)
                if sample.phase() == Phase::Decode && sample.n_decode_seqs == 24 =>
            {
                assert_eq!(sample.graph_replayed, u8::from(graphs && can));
                decode24 = true;
            }
            _ => {}
        }
    }
    assert_eq!(cancelled, 8);
    assert!(decode32, "all 32 requests must have entered active decode");
    assert!(
        decode24,
        "active cancellation did not leave a 24-row decode step"
    );
    let reuse = mix(&run);
    run.e.shutdown();
    (tokens.into_iter().take(24).collect::<Vec<_>>(), reuse)
}

/// Cancel only after every sequence generated a token. A one-shot first-decode gate makes the
/// cancellation boundary deterministic even when graph replay outruns the host event reader.
#[test]
fn active_cancellation_reuses_graph_rows_and_kv() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(
        active_cancellation_campaign(false, tiny_path()),
        active_cancellation_campaign(true, tiny_path())
    );
}

/// HD128/GQA2 real Transformer attention survives captured full chunks and active cancellation.
/// LARCH_PREFILL_TC=1 selects tensor-core prefill on capable CUDA devices; CPU and unsupported
/// devices still validate the real-loader fallback. Operator tests check the CPU reference.
#[test]
fn hd128_attention_graphs_cancel_and_reuse_paged_inputs() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let path = tiny_hd128_path();
    assert_eq!(
        active_cancellation_campaign(false, path.clone()),
        active_cancellation_campaign(true, path.clone())
    );
    let (expected, eager) = mixed_full_chunk_campaign(false, 64, 8, path.clone());
    assert!(eager.iter().all(|s| s.graph_replayed == 0));
    let (actual, samples) = mixed_full_chunk_campaign(true, 64, 8, path);
    assert_eq!(actual, expected);
    let mixed: Vec<_> = samples
        .iter()
        .filter(|s| s.n_decode_seqs == 7 && s.n_prefill_seqs == 1 && s.n_prefill_tokens == 64)
        .collect();
    assert!(
        mixed.len() >= 9,
        "missing eligible HD128 repeated full chunks"
    );
    let props = Device::open(device()).unwrap().props().unwrap();
    let captures = larch_kernels::routes().mixed_graphs
        && props.backend == larch_kernels::Backend::Cuda
        && props.flags & DeviceProps::CAP_GRAPHS != 0;
    assert_eq!(
        mixed
            .iter()
            .filter(|s| s.graph_captured() == Some(true))
            .count(),
        if captures { 2 } else { 0 }
    );
    if captures {
        assert!(mixed.iter().filter(|s| s.graph_replayed != 0).count() >= 7);
    } else {
        assert!(mixed.iter().all(|s| s.graph_replayed == 0));
    }
}

/// Public custom loaders may use host copies in forward, which CUDA capture refuses. Omitted or
/// invalid graph overrides retain eager decode, and a valid explicit opt-out does too.
#[test]
fn custom_loader_defaults_to_eager_decode() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    for override_value in [None, Some(0.0), Some(2.0), Some(f64::NAN)] {
        let mut opts = EngineOptions::new(PathBuf::from("fake"), device());
        opts.max_context = 512;
        opts.kv_pool_fraction = 0.3;
        if let Some(value) = override_value {
            opts.knob_overrides.push((knob_id::DECODE_GRAPHS, value));
        }
        let e = start_with(opts, Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64))))).unwrap();
        let (tokens, terminal) = run_four(&e);
        let graphs = e.knobs().effective().values[knob_id::DECODE_GRAPHS as usize];
        let health = e.health().state;
        let rejected = e.rejected_overrides().to_vec();
        let samples = decode_samples(e);
        assert_eq!(
            tokens,
            [3, 4, 5, 6],
            "override {override_value:?}: {terminal:?}"
        );
        assert!(
            matches!(terminal, Some(EngineEvent::Done { usage, .. }) if usage.completion_tokens == 4)
        );
        assert_eq!(health, HealthState::Ready);
        assert_eq!(graphs, 0.0);
        assert_eq!(samples.len(), 3);
        assert!(samples.iter().all(|s| s.graph_replayed == 0));
        if let Some(value) = override_value.filter(|&v| v != 0.0) {
            assert_eq!(rejected.len(), 1);
            assert_eq!(rejected[0].0, knob_id::DECODE_GRAPHS);
            assert_eq!(rejected[0].1.to_bits(), value.to_bits());
            assert_eq!(rejected[0].2, RejectReason::WrongType);
        } else {
            assert!(rejected.is_empty());
        }
    }
}

/// Ordinary Transformer startup keeps the measured-arch CUDA default (sm_120) and honors both valid
/// explicit settings.
#[test]
fn ordinary_loader_preserves_graph_defaults_and_overrides() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let arch = Device::open(device()).unwrap().props().unwrap().arch_code;
    for override_value in [None, Some(0.0), Some(1.0)] {
        let mut opts = EngineOptions::new(tiny_path(), device());
        opts.max_context = 512;
        opts.kv_pool_fraction = 0.3;
        if let Some(value) = override_value {
            opts.knob_overrides.push((knob_id::DECODE_GRAPHS, value));
        }
        let e = start(opts).unwrap();
        let caps = e.capabilities();
        let expected = f64::from(u8::from(
            caps.graph_replay
                && override_value
                    .map_or(caps.backend == Backend::Cuda && arch == 120, |v| v != 0.0),
        ));
        assert_eq!(
            e.knobs().effective().values[knob_id::DECODE_GRAPHS as usize],
            expected
        );
        if override_value == Some(1.0) && !caps.graph_replay {
            assert_eq!(
                e.rejected_overrides(),
                [(knob_id::DECODE_GRAPHS, 1.0, RejectReason::Unsupported)]
            );
        } else {
            assert!(e.rejected_overrides().is_empty());
        }
        let (tokens, terminal) = run_four(&e);
        let samples = decode_samples(e);
        assert_eq!(tokens.len(), 4, "override {override_value:?}: {terminal:?}");
        assert!(
            matches!(terminal, Some(EngineEvent::Done { usage, .. }) if usage.completion_tokens == 4)
        );
        assert_eq!(samples.len(), 3);
        assert!(
            samples
                .iter()
                .all(|s| f64::from(s.graph_replayed) == expected)
        );
    }
}

/// A custom replacement must not inherit the ordinary CUDA Transformer's graph default.
#[test]
fn custom_reload_defaults_to_eager_decode() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let mut opts = EngineOptions::new(tiny_path(), device());
    opts.max_context = 512;
    opts.kv_pool_fraction = 0.3;
    let e = start(opts).unwrap();
    let old_graph_rev = if e.knobs().effective().values[knob_id::DECODE_GRAPHS as usize] != 0.0 {
        let KnobReply::Applied { rev, .. } = e.knobs().set(KnobRequest::new(
            knob_id::DECODE_GRAPHS,
            1.0,
            Authority::Human,
        )) else {
            panic!("graph setting refused");
        };
        Some(rev)
    } else {
        None
    };
    let before = e.knobs().effective();
    e.reload_with(
        Default::default(),
        Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
    )
    .unwrap();
    let effective = e.knobs().effective();
    if effective.values[knob_id::DECODE_GRAPHS as usize] != 0.0 {
        e.shutdown();
        panic!("a custom replacement inherited decode graphs");
    }
    assert_eq!(
        effective.rev,
        before.rev + u64::from(before.values[knob_id::DECODE_GRAPHS as usize] != 0.0)
    );
    // Neither an older graph setting nor the reload's own revision reverts graphs back on.
    if let Some(rev) = old_graph_rev {
        assert_eq!(
            e.knobs().revert(RevertTarget::Rev(rev), Authority::Human),
            KnobReply::Rejected {
                reason: RejectReason::Superseded {
                    by_rev: effective.rev,
                },
            }
        );
    }
    assert_eq!(
        e.knobs()
            .revert(RevertTarget::Rev(effective.rev), Authority::Human),
        KnobReply::Rejected {
            reason: RejectReason::NothingToRevert,
        }
    );
    assert_eq!(
        e.knobs().effective().values[knob_id::DECODE_GRAPHS as usize],
        0.0
    );
    let (tokens, terminal) = run_four(&e);
    let health = e.health().state;
    let samples = decode_samples(e);
    assert_eq!(tokens, [3, 4, 5, 6], "{terminal:?}");
    assert!(matches!(terminal, Some(EngineEvent::Done { .. })));
    assert_eq!(health, HealthState::Ready);
    assert_eq!(samples.len(), 3);
    assert!(samples.iter().all(|s| s.graph_replayed == 0));
}

/// A plain reload after a custom one is the ordinary loader again: the startup setting comes back,
/// the arch default or an explicit `--set decode_graphs=0`.
#[test]
fn plain_reload_after_custom_restores_graph_default() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    for override_value in [None, Some(0.0)] {
        let mut opts = EngineOptions::new(tiny_path(), device());
        opts.max_context = 512;
        opts.kv_pool_fraction = 0.3;
        opts.knob_overrides
            .extend(override_value.map(|v| (knob_id::DECODE_GRAPHS, v)));
        let e = start(opts).unwrap();
        let startup = e.knobs().effective().values[knob_id::DECODE_GRAPHS as usize];
        e.reload_with(
            Default::default(),
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
        )
        .unwrap();
        assert_eq!(
            e.knobs().effective().values[knob_id::DECODE_GRAPHS as usize],
            0.0
        );
        e.reload(Default::default()).unwrap();
        let graphs = e.knobs().effective().values[knob_id::DECODE_GRAPHS as usize];
        e.shutdown();
        assert_eq!(graphs, startup, "override {override_value:?}");
    }
}

/// A live setting made after startup is what the plain reload restores, not the startup value.
/// Needs a device that captures graphs: elsewhere turning them on is refused, so nothing is flipped.
#[test]
fn plain_reload_after_custom_restores_live_graph_setting() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let mut opts = EngineOptions::new(tiny_path(), device());
    opts.max_context = 512;
    opts.kv_pool_fraction = 0.3;
    let e = start(opts).unwrap();
    let live = 1.0 - e.knobs().effective().values[knob_id::DECODE_GRAPHS as usize];
    let req = KnobRequest::new(knob_id::DECODE_GRAPHS, live, Authority::Human);
    if !matches!(e.knobs().set(req), KnobReply::Applied { .. }) {
        e.shutdown();
        return;
    }
    e.reload_with(
        Default::default(),
        Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
    )
    .unwrap();
    e.reload(Default::default()).unwrap();
    let graphs = e.knobs().effective().values[knob_id::DECODE_GRAPHS as usize];
    e.shutdown();
    assert_eq!(graphs, live);
}

/// After warming every bucket, the same mix retains no extra device memory. These snapshots do
/// not detect transient allocations that are freed before the next device-property query.
#[test]
fn steady_state_decode_allocates_nothing() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::open(device()).unwrap();
    let r = engine(true);
    let tokens = mix(&r);
    let free = dev.props().unwrap().vram_free;
    assert_eq!(mix(&r), tokens);
    assert_eq!(
        dev.props().unwrap().vram_free,
        free,
        "decode retained extra device memory after the warm-up"
    );
    r.e.shutdown();
}

/// A DeviceSample with `sample_tick` 0 follows the load and each reload; where the engine opened
/// NVML (CUDA) its sampler adds one a second, ticks from 1. On CUDA the VRAM is measured, the used
/// part inside the total, and with NVML the device fields are read (Measured, or Unsupported where
/// NVML refuses them), never Missing, and without NVML Missing; elsewhere (the CPU reference in CI)
/// every field is Unsupported and no sampler runs.
#[test]
fn device_sample_carries_vram_at_load_and_reload() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let r = engine(false);
    let caps = r.e.capabilities();
    let (cuda, nvml) = (
        caps.backend == Backend::Cuda,
        caps.device_sample == MState::Measured,
    );
    if nvml {
        std::thread::sleep(Duration::from_millis(2500));
    }
    r.e.reload(ReloadChange::default()).unwrap();
    let mut reader = r.e.ring().reader(ReadFrom::Oldest);
    r.e.shutdown();
    let mut samples = Vec::new();
    while let Read::Record(rec) = reader.next() {
        if let Record::DeviceSample(_, d) = rec {
            samples.push(d);
        }
    }
    let ticks: Vec<u64> = samples
        .iter()
        .map(|d| d.sample_tick)
        .filter(|&t| t > 0)
        .collect();
    assert_eq!(samples.len() - ticks.len(), 2, "one at load, one at reload");
    if nvml {
        assert!(ticks.len() >= 2, "{ticks:?}");
        assert!(ticks.iter().zip(1..).all(|(&t, i)| t == i), "{ticks:?}");
    } else {
        assert_eq!(ticks, [0; 0]);
    }
    for d in samples {
        let (used, total) = (d.vram_used_bytes, d.vram_total_bytes);
        let device = [
            d.sm_clock_mhz,
            d.temp_c,
            d.power_w,
            d.power_limit_w,
            d.gpu_util,
        ];
        if cuda {
            assert_eq!(
                (used.state(), total.state()),
                (MState::Measured, MState::Measured)
            );
            assert!(0 < used.v && used.v <= total.v, "{used:?} of {total:?}");
            for m in device {
                if nvml {
                    assert_ne!(m.state(), MState::Missing, "{d:?}");
                } else {
                    assert_eq!(m.state(), MState::Missing, "{d:?}");
                }
            }
        } else {
            assert_eq!(
                (used.state(), total.state()),
                (MState::Unsupported, MState::Unsupported)
            );
            for m in device {
                assert_eq!(m.state(), MState::Unsupported, "{d:?}");
            }
        }
    }
}

/// The score gather (`embed` over one logits row viewed as an F32 `[vocab, 1]` table, as the engine
/// runs it) on the test device copies exactly what the CPU reference copies: repeated and
/// out-of-order ids, a table and ids that start mid-buffer, and an output that starts past a prefix
/// it must leave alone.
#[test]
fn score_gather_matches_cpu_reference() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    const VOCAB: usize = 37;
    // Three logits rows; the gather reads the last. The first three ids are outside the view, and
    // -1 would be out of range if the op read them.
    let logits: Vec<f32> = (0..3 * VOCAB)
        .map(|i| (i as f32 * 0.37).sin() * 9.0)
        .collect();
    let ids = [-1i32, -1, -1, 5, 36, 5, 0, 17, 36];
    let gather = |ordinal: i32| {
        let dev = Device::open(ordinal).unwrap();
        let s = Stream::new(&dev).unwrap();
        let up = |bytes: &[u8]| {
            let b = Buffer::alloc(&dev, bytes.len() as u64).unwrap();
            s.upload(&b, 0, bytes).unwrap();
            b
        };
        let (table, named) = (
            up(bytemuck::cast_slice(&logits)),
            up(bytemuck::cast_slice(&ids)),
        );
        let gathered = up(bytemuck::cast_slice(&[-7f32; 12]));
        let table_row = Tensor::new(
            &table,
            (2 * VOCAB * 4) as u64,
            DType::F32,
            &[VOCAB as i64, 1],
        );
        ops::embed(
            &s,
            &table_row.unwrap(),
            &Tensor::new(&named, 12, DType::I32, &[6]).unwrap(),
            &Tensor::new(&gathered, 16, DType::F32, &[6, 1]).unwrap(),
        )
        .unwrap();
        let mut out = [0f32; 12];
        s.download(&gathered, 0, &mut out).unwrap();
        s.sync().unwrap();
        out.to_vec()
    };
    let mut want = vec![-7f32; 4];
    want.extend(ids[3..].iter().map(|&id| logits[2 * VOCAB + id as usize]));
    want.extend([-7f32; 2]);
    assert_eq!(gather(CPU), want, "CPU reference");
    assert_eq!(gather(device()), want, "device {}", device());
}

/// A token stream's ids, each with its `top`.
fn tops(h: RequestHandle) -> Vec<(u32, Vec<(u32, f32)>)> {
    let mut out = Vec::new();
    for ev in h.events.iter() {
        match ev {
            EngineEvent::Token { id, top, .. } => out.push((id, top)),
            EngineEvent::Done { .. } => return out,
            EngineEvent::Prefilled { .. } => {}
            ev => panic!("{ev:?}"),
        }
    }
    panic!("no Done")
}

/// f16 bits of a small whole number (the kernel tests' conversion).
fn f16(x: f32) -> u16 {
    let b = x.to_bits();
    if x == 0.0 {
        return 0;
    }
    ((b >> 16) & 0x8000 | (((b >> 23) & 0xff) - 112) << 10 | (b >> 13) & 0x3ff) as u16
}

/// The tiny model with logits row r raised by 8 (r mod 16), on the device so a graph captures it:
/// every logprob is unchanged, but rows' LSEs differ by whole units, so a row paired with another
/// row's logits or LSE shows far past the CPU tolerance.
struct RowShifted(Transformer, Buffer);

impl Model for RowShifted {
    fn config(&self) -> &ModelConfig {
        self.0.config()
    }

    fn forward(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        self.0.forward(s, b, kv, logits)?;
        let shift = Tensor::new(&self.1, 0, DType::F16, logits.shape())?;
        ops::add(s, logits, &shift, logits)
    }
}

/// Score requests interleaved with ordinary ones on the test device, graphs on: the decode steps of
/// 5 and 3 rows replay the 8- and 4-row graphs with padding rows wherever the device has graphs.
/// Every token's logprobs, scores and ordinary top 5 alike, match the CPU reference scoring that
/// request's own prefix (prefix.rs's 1e-2), so each row's ids, logits and LSE stay paired.
#[test]
fn score_requests_match_cpu_reference_through_padded_replays() {
    let _device_guard = DEVICE_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let load: Loader = Box::new(|dev, s, rows| {
        let model = Transformer::load(&tiny_path(), dev, s, rows)?;
        let vocab = model.config().vocab as usize;
        let shift: Vec<u16> = (0..rows as usize * vocab)
            .map(|i| f16(8.0 * (i / vocab % 16) as f32))
            .collect();
        let buf = Buffer::alloc(dev, (shift.len() * 2) as u64).unwrap();
        s.upload(&buf, 0, &shift).unwrap();
        Ok(Box::new(RowShifted(model, buf)) as Box<dyn Model>)
    });
    let mut o = EngineOptions::new(tiny_path(), device());
    o.kv_pool_fraction = 0.3;
    o.knob_overrides = vec![(knob_id::DECODE_GRAPHS, 1.0)];
    let e = start_with(o, load).unwrap();
    let can = u8::from(e.capabilities().graph_replay);
    // Repeated, out of order, outside any top 64, and in a different order for each request.
    let score_ids = |i: usize| {
        let mut ids = vec![200, 3, 200, 0, 255, 17];
        ids.rotate_left(i);
        ids
    };
    let reqs: Vec<Request> = [4u32, 3, 5, 2, 4]
        .into_iter()
        .enumerate()
        .map(|(i, n)| {
            let mut q = Request::new((0..=i as u32).map(|t| t * 37 + 11).collect(), n);
            if i % 2 == 0 {
                q.score_ids = score_ids(i);
            } else {
                q.logprobs = 5;
            }
            q
        })
        .collect();
    let runs: Vec<_> = (e.submit_group(reqs.clone()).unwrap().into_iter())
        .map(tops)
        .collect();
    let samples = decode_samples(e);
    for rows in [5, 3] {
        assert!(samples.iter().any(|s| s.n_decode_seqs == rows), "{rows}");
    }
    assert!(samples.iter().all(|s| s.graph_replayed == can));
    let oracle = start(EngineOptions::new(tiny_path(), CPU)).unwrap();
    let mut worst = 0f32;
    for (q, run) in reqs.iter().zip(&runs) {
        assert_eq!(run.len(), q.max_tokens as usize);
        for (k, (_, top)) in run.iter().enumerate() {
            let named: Vec<u32> = top.iter().map(|t| t.0).collect();
            if q.score_ids.is_empty() {
                assert_eq!(named.len(), 5);
            } else {
                assert_eq!(named, q.score_ids);
            }
            let mut prefix = q.prompt.clone();
            prefix.extend(run[..k].iter().map(|t| t.0));
            let reference = Request {
                score_ids: named,
                ..Request::new(prefix, 1)
            };
            let want = tops(oracle.submit(reference).unwrap()).remove(0).1;
            for (&(id, lp), &(want_id, want_lp)) in top.iter().zip(&want) {
                assert_eq!(id, want_id);
                // f32::max drops a NaN, so a NaN logprob must fail here.
                assert!(
                    lp.is_finite() && want_lp.is_finite(),
                    "id {id}: {lp} vs {want_lp}"
                );
                worst = worst.max((lp - want_lp).abs());
            }
        }
    }
    assert!(worst < 1e-2, "CPU-reference logprob delta {worst}");
}
