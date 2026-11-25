//! Shared KV lifecycle through the real executor. Numerical checks use the independent CPU kernel
//! reference on the same generated GGUF and the existing cached/fresh logprob tolerance (1e-2).

use std::path::PathBuf;
use std::sync::OnceLock;

use larch_engine::{
    EngineEvent, EngineHandle, EngineOptions, Forced, ReloadChange, Request, RequestHandle, start,
    start_with,
};
use larch_kernels::CPU;
use larch_model::{Arch, FakeModel, tiny};
use larch_records::{Read, ReadFrom, Record, knob_id};

fn path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let p = std::env::temp_dir().join(format!("larch-prefix-tiny-{}.gguf", std::process::id()));
        std::fs::write(&p, tiny::gguf(Arch::Llama, 3)).unwrap();
        p
    })
    .clone()
}

fn options(device: i32, blocks: usize) -> EngineOptions {
    let mut o = EngineOptions::new(path(), device);
    o.max_context = 512;
    o.kv_pool_fraction = 0.3;
    o.prefix_cache_blocks = blocks;
    o.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 8.0)];
    o
}

fn device() -> i32 {
    std::env::var("LARCH_TEST_DEVICE").map_or(CPU, |s| s.parse().unwrap())
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n).map(|i| (i as u32 * 7 + 3) % 200).collect()
}

type Tokens = Vec<(u32, Vec<(u32, f32)>)>;

fn collect(h: RequestHandle) -> (Tokens, u32) {
    let mut tokens = Vec::new();
    for ev in h.events.iter() {
        match ev {
            EngineEvent::Token { id, top, .. } => tokens.push((id, top)),
            EngineEvent::Done { usage, .. } => {
                assert_eq!(tokens.len(), usage.completion_tokens as usize);
                return (tokens, usage.cached_tokens);
            }
            EngineEvent::Prefilled { .. } => {}
            ev => panic!("{ev:?}"),
        }
    }
    panic!("missing terminal event")
}

fn request(p: Vec<u32>, outputs: u32) -> Request {
    let mut r = Request::new(p, outputs);
    r.logprobs = 5;
    r
}

/// CPU teacher forcing computes the reference at every position, including the reused prefix.
/// Compare every candidate-generated logprob with that independent computation, without assuming
/// a cached candidate is itself a correctness oracle.
fn cpu_reference(e: &EngineHandle, p: &[u32], tokens: &Tokens) -> f32 {
    let mut all = p.to_vec();
    all.extend(tokens.iter().take(tokens.len() - 1).map(|t| t.0));
    let h = e
        .force(Forced {
            tokens: all,
            prompt: p.len() as u32,
            session: 0,
            keep_kv: false,
        })
        .unwrap();
    let positions: Vec<_> = h.positions.iter().collect();
    assert!(matches!(
        h.request.events.recv().unwrap(),
        EngineEvent::Done { .. }
    ));
    assert_eq!(positions.len(), p.len() + tokens.len() - 1);
    let mut maximum = 0f32;
    for (position, (_, top)) in positions[p.len() - 1..].iter().zip(tokens) {
        for &(id, value) in top {
            let oracle = position
                .top
                .iter()
                .find(|(i, _)| *i == id)
                .expect("candidate top5 in CPU top64")
                .1;
            maximum = maximum.max((value - oracle).abs());
        }
    }
    assert!(maximum < 1e-2, "CPU-reference logprob delta {maximum}");
    maximum
}

#[test]
fn automatic_shared_graph_tables_match_cpu_reference() {
    let e = start(options(device(), 128)).unwrap();
    let oracle = start(options(CPU, 0)).unwrap();
    let p = prompt(65);
    let (cold, cached) = collect(e.submit(request(p.clone(), 1)).unwrap());
    assert_eq!(cached, 0);
    cpu_reference(&oracle, &p, &cold);
    let mut max_delta = 0f32;
    for _round in 0..3 {
        // Both begin with four shared blocks; changing suffixes/lengths give different private
        // block tables on consecutive replays of the same decode graph shape.
        let group: Vec<_> = (0..4)
            .map(|i| {
                let mut p = p.clone();
                p[64] = 20 + i;
                request(p, 18 + i)
            })
            .collect();
        let handles = e.submit_group(group.clone()).unwrap();
        for (r, h) in group.iter().zip(handles) {
            let (tokens, cached) = collect(h);
            assert_eq!(cached, 64);
            max_delta = max_delta.max(cpu_reference(&oracle, &r.prompt, &tokens));
        }
    }
    let stats = e.prefix_cache_stats().unwrap();
    assert_eq!(stats.hits, 12);
    assert_eq!(stats.computed_prompt_tokens, 65 + 12);
    assert_eq!(stats.unique_blocks_used, 4);
    assert_eq!(stats.unique_tokens_used, 64);
    let can_graph = e.capabilities().graph_replay;
    let mut reader = e.ring().reader(ReadFrom::Oldest);
    e.shutdown();
    oracle.shutdown();
    let mut replays = 0;
    while let Read::Record(record) = reader.next() {
        if let Record::EngineSample(_, s) = record {
            replays += u64::from(s.graph_replayed != 0);
        }
    }
    if can_graph {
        assert!(replays > 0, "graph-capable device must actually replay");
    }
    eprintln!("automatic CPU-oracle max logprob delta={max_delta}, graph replays={replays}");
}

#[test]
fn automatic_coalesced_cold_publishers_match_cpu_reference() {
    let e = start(options(device(), 128)).unwrap();
    let oracle = start(options(CPU, 0)).unwrap();
    let group: Vec<_> = (0..4)
        .map(|i| {
            let mut p = prompt(33);
            p[16] = 20 + i;
            request(p, 18)
        })
        .collect();
    let handles = e.submit_group(group.clone()).unwrap();
    let mut max_delta = 0f32;
    for (r, h) in group.iter().zip(handles) {
        let (tokens, cached) = collect(h);
        assert_eq!(
            cached, 0,
            "coalescing happens after computation and cannot claim saved work"
        );
        max_delta = max_delta.max(cpu_reference(&oracle, &r.prompt, &tokens));
    }
    let stats = e.prefix_cache_stats().unwrap();
    assert_eq!(stats.computed_prompt_tokens, 132);
    assert_eq!(stats.unique_blocks_used, 5);
    assert_eq!(stats.unique_tokens_used, 80);
    e.shutdown();
    oracle.shutdown();
    eprintln!("coalesced cold-publisher CPU-oracle max logprob delta={max_delta}");
}

#[test]
fn automatic_policy_observation_and_reload() {
    let e = start(options(device(), 128)).unwrap();
    collect(e.submit(request(prompt(65), 1)).unwrap());
    let mut disabled = request(prompt(65), 1);
    disabled.prefix_cache = false;
    assert_eq!(collect(e.submit(disabled).unwrap()).1, 0);
    let mut isolated = request(prompt(65), 1);
    isolated.cache_namespace = 1;
    assert_eq!(collect(e.submit(isolated).unwrap()).1, 0);
    let mut no_retention = request(prompt(65), 1);
    no_retention.session = 9;
    no_retention.keep_kv = false;
    assert_eq!(collect(e.submit(no_retention).unwrap()).1, 0);
    let f = e
        .force(Forced {
            tokens: prompt(65),
            prompt: 65,
            session: 0,
            keep_kv: false,
        })
        .unwrap();
    assert_eq!(f.positions.iter().count(), 65);
    assert!(
        matches!(f.request.events.recv().unwrap(), EngineEvent::Done { usage, .. } if usage.cached_tokens == 0)
    );
    e.reload(ReloadChange::default()).unwrap();
    assert_eq!(e.prefix_cache_stats().unwrap().indexed_blocks, 0);
    assert_eq!(collect(e.submit(request(prompt(65), 1)).unwrap()).1, 0);
    assert_eq!(collect(e.submit(request(prompt(65), 1)).unwrap()).1, 64);
    e.shutdown();
}

#[test]
fn automatic_custom_loaders_require_explicit_compatibility() {
    for compatible in [false, true] {
        let mut o = options(CPU, 128);
        o.prefix_cache_custom_compatible = compatible;
        let e = start_with(o, Box::new(|_, _, _| Ok(Box::new(FakeModel::new(256))))).unwrap();
        collect(e.submit(request(prompt(65), 1)).unwrap());
        assert_eq!(
            collect(e.submit(request(prompt(65), 1)).unwrap()).1,
            if compatible { 64 } else { 0 }
        );
        e.shutdown();
    }
}
