//! Reproducible token-ID workloads for prefix-cache cold/warm, independent histories, unique
//! traffic, and memory/cancellation pressure. No prepopulation outside measured rounds. Emits
//! individual token gaps and protected telemetry records as JSONL; model loading is outside clocks.
//! Usage: prefix_bench GGUF DEVICE CONCURRENCY WORKLOAD(C|D|E|F) ROUNDS

use std::sync::Arc;
use std::time::Instant;

use larch_engine::{EngineEvent, EngineOptions, Request, start};
use larch_records::{Read, ReadFrom, Record, knob_id};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 6, "GGUF DEVICE CONCURRENCY WORKLOAD ROUNDS");
    let concurrency: usize = args[3].parse().unwrap();
    let workload = args[4].as_str();
    assert!(matches!(workload, "C" | "D" | "E" | "F"));
    let rounds: usize = args[5].parse().unwrap();
    let mut opts = EngineOptions::new(args[1].clone().into(), args[2].parse().unwrap());
    opts.max_context = 2560;
    opts.kv_pool_fraction = if workload == "F" { 0.3 } else { 0.85 };
    opts.ring_capacity = 65536;
    opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 32.0)];
    let e = start(opts).unwrap();
    let vocab = e.deployment().vocab;
    let mut reader = e.ring().reader(ReadFrom::Oldest);
    // Identical unrelated warmup on parent and candidate; its costs are reported separately.
    let warm_start = Instant::now();
    let warm = e.submit(Request::new(vec![1], 1)).unwrap();
    assert!(matches!(
        warm.events.iter().last(),
        Some(EngineEvent::Done { .. })
    ));
    println!(
        "{{\"kind\":\"warmup\",\"wall_ns\":{}}}",
        warm_start.elapsed().as_nanos()
    );
    let clock = Arc::new(Instant::now());
    for round in 0..rounds {
        let phase_start = clock.elapsed().as_nanos();
        // Fixed closed-loop groups; concurrency=1 is sequential reuse, >1 overlaps arrivals.
        for group in 0..(16usize.div_ceil(concurrency)) {
            let first = group * concurrency;
            let n = concurrency.min(16 - first);
            let prompts: Vec<Vec<u32>> = (first..first + n)
                .map(|i| {
                    let length = match workload {
                        "F" => 2304,
                        "D" => 512 + (round + 1) * 64,
                        _ => 1056,
                    };
                    let mut seed = if workload == "D" {
                        (i + 1) as u32
                    } else {
                        (round * 16 + i + 1) as u32
                    };
                    (0..length)
                        .map(|pos| {
                            if (workload == "D" && pos < 512) || (workload == "C" && pos < 1024) {
                                (pos as u32 * 7 + 3) % vocab
                            } else {
                                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                                seed % vocab
                            }
                        })
                        .collect()
                })
                .collect();
            let requests = prompts
                .iter()
                .enumerate()
                .map(|(j, p)| {
                    let mut r = Request::new(p.clone(), 64);
                    r.ignore_eos = true;
                    if workload == "D" {
                        r.session = (first + j + 1) as u64;
                        r.keep_kv = true;
                    }
                    r
                })
                .collect();
            let submit = clock.elapsed().as_nanos();
            let handles = e.submit_group(requests).unwrap();
            let workers: Vec<_> = handles.into_iter().enumerate().map(|(j, h)| {
                let clock = clock.clone();
                let cancel = workload == "F" && (first + j).is_multiple_of(2);
                std::thread::spawn(move || {
                    let mut times = Vec::new();
                    let mut ids = Vec::new();
                    for event in h.events.iter() {
                        match event {
                            EngineEvent::Token { id, .. } => {
                                ids.push(id); times.push(clock.elapsed().as_nanos());
                                if cancel && ids.len() == 1 { h.cancel(); }
                            }
                            EngineEvent::Prefilled { .. } => {}
                            EngineEvent::Done { usage, .. } => {
                                println!("{{\"kind\":\"request\",\"round\":{round},\"id\":{},\"submit_ns\":{submit},\"done_ns\":{},\"prompt_tokens\":{},\"cached_tokens\":{},\"outputs\":{},\"times_ns\":{:?},\"tokens\":{:?},\"outcome\":\"done\"}}", h.id, clock.elapsed().as_nanos(), usage.prompt_tokens, usage.cached_tokens, ids.len(), times, ids);
                                return;
                            }
                            EngineEvent::Error(error) => {
                                println!("{{\"kind\":\"request\",\"round\":{round},\"id\":{},\"submit_ns\":{submit},\"done_ns\":{},\"outputs\":{},\"times_ns\":{:?},\"tokens\":{:?},\"outcome\":\"{:?}\"}}", h.id, clock.elapsed().as_nanos(), ids.len(), times, ids, error);
                                return;
                            }
                        }
                    }
                    panic!("no terminal event")
                })
            }).collect();
            for worker in workers {
                worker.join().unwrap();
            }
        }
        println!(
            "{{\"kind\":\"round\",\"round\":{round},\"wall_ns\":{}}}",
            clock.elapsed().as_nanos() - phase_start
        );
        let stats = e.prefix_cache_stats().unwrap();
        println!(
            "{{\"kind\":\"cache\",\"round\":{round},\"capacity\":{},\"indexed\":{},\"unique_blocks\":{},\"unique_tokens\":{},\"metadata_bytes\":{},\"lookups\":{},\"hits\":{},\"reused\":{},\"computed\":{},\"evictions\":{}}}",
            stats.capacity_blocks,
            stats.indexed_blocks,
            stats.unique_blocks_used,
            stats.unique_tokens_used,
            stats.index_metadata_bytes,
            stats.lookups,
            stats.hits,
            stats.reused_prompt_tokens,
            stats.computed_prompt_tokens,
            stats.evictions
        );
    }
    // Drain the engine before final records are read. Cancellations are not completed throughput.
    e.shutdown();
    loop {
        match reader.next() {
            Read::Record(Record::EngineSample(_, s)) => println!(
                "{{\"kind\":\"step\",\"step\":{},\"duration_ns\":{},\"gpu_ns\":{},\"decode\":{},\"prefill\":{},\"prefill_tokens\":{},\"replay\":{},\"capture\":{},\"unique_blocks\":{},\"unique_tokens\":{},\"preempted\":{}}}",
                s.step,
                s.dur_ns,
                s.gpu_ns.get().map_or("null".to_owned(), |v| v.to_string()),
                s.n_decode_seqs,
                s.n_prefill_seqs,
                s.n_prefill_tokens,
                s.graph_replayed,
                u8::from(s.graph_captured().unwrap()),
                s.kv_blocks_used,
                s.kv_tokens_used,
                s.n_preempted
            ),
            Read::Record(Record::RequestSpan(_, s)) => println!(
                "{{\"kind\":\"span\",\"id\":{},\"submit_ns\":{},\"admit_ns\":{},\"first_ns\":{},\"done_ns\":{},\"cached\":{},\"prompt\":{},\"generated\":{},\"preemptions\":{},\"outcome\":{}}}",
                s.req_id,
                s.t_submit_ns,
                s.t_admit_ns,
                s.t_first_ns,
                s.t_done_ns,
                s.cached_tokens,
                s.prompt_tokens,
                s.gen_tokens,
                s.preemptions,
                s.outcome
            ),
            Read::Record(_) => {}
            Read::Lagged { dropped } => panic!("telemetry lost {dropped} records"),
            _ => break,
        }
    }
}
