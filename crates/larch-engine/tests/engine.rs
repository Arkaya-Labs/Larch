//! The engine end to end on the CPU reference: streaming and stops, admission, cancel, chunked
//! prefill and preemption through the executor, the knob path and its telemetry, the reported
//! configuration, faults and shutdown (FakeModel), and the D14 session tests of section 15 on the tiny
//! model; then teacher forcing (`force`), A/B and the guardrail (9.9), and the telemetry counters.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, OnceLock};
use std::time::{Duration, Instant};

use larch_engine::grammar::{Compiled, Constraint};
use larch_engine::{
    EngineEvent, EngineHandle, EngineOptions, Finish, Forced, ForcedHandle, MAX_SCORE_IDS,
    Position, ReloadChange, Request, RequestHandle, SamplingParams, TelemetryTier, invalid, start,
    start_with,
};
use larch_kernels::{CPU, KError, KvPool, Stream, Tensor};
use larch_model::{Arch, FakeModel, Model, ModelConfig, StepBatch, Taps, Transformer, gguf, tiny};
use larch_records::{
    AbCommand, AbEnd, AbReply, ActivationClass, ActivationResult, Allow, Authority, Backend, Cause,
    EngineError, HealthState, IndeterminateReason, KnobControl, KnobDelta, KnobReply, KnobRequest,
    LoadError, MState, Outcome, Phase, Read, ReadFrom, ReclaimReason, Record, RejectReason,
    RequestSpan, ReuseMiss, RevertTarget, RiskClass, knob_id,
};

fn fake_opts(overrides: Vec<(u16, f64)>, max_context: u32) -> EngineOptions {
    let mut opts = EngineOptions::new(PathBuf::from("fake"), CPU);
    opts.knob_overrides = overrides;
    opts.max_context = max_context;
    opts
}

fn fake(overrides: Vec<(u16, f64)>, max_context: u32) -> EngineHandle {
    start_with(
        fake_opts(overrides, max_context),
        Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)) as Box<dyn Model>)),
    )
    .unwrap()
}

/// The tiny llama model, written once per test binary.
fn tiny_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let p = std::env::temp_dir().join(format!("larch-engine-tiny-{}.gguf", std::process::id()));
        std::fs::write(&p, tiny::gguf(Arch::Llama, 3)).unwrap();
        p
    })
    .clone()
}

fn tiny_engine() -> EngineHandle {
    start(EngineOptions::new(tiny_path(), CPU)).unwrap()
}

/// Generated tokens, each with its top logprobs.
type Tokens = Vec<(u32, Vec<(u32, f32)>)>;

/// Tokens and the terminal event.
fn run(h: RequestHandle) -> (Tokens, EngineEvent) {
    let mut tokens = Vec::new();
    for e in h.events.iter() {
        match e {
            EngineEvent::Token { id, top, .. } => tokens.push((id, top)),
            EngineEvent::Prefilled { .. } => {}
            end => return (tokens, end),
        }
    }
    panic!("stream ended without Done or Error");
}

/// The request's next event past its prefill progress.
fn event_of(h: &RequestHandle) -> EngineEvent {
    loop {
        match h.events.recv().unwrap() {
            EngineEvent::Prefilled { .. } => {}
            e => return e,
        }
    }
}

fn ids(t: &[(u32, Vec<(u32, f32)>)]) -> Vec<u32> {
    t.iter().map(|x| x.0).collect()
}

fn done(e: &EngineEvent) -> (Finish, u32, u32) {
    match e {
        EngineEvent::Done { finish, usage, .. } => {
            (*finish, usage.completion_tokens, usage.cached_tokens)
        }
        e => panic!("not Done: {e:?}"),
    }
}

/// Every record in the ring so far.
fn records(e: &EngineHandle) -> Vec<Record> {
    let mut r = e.ring().reader(ReadFrom::Oldest);
    let mut out = Vec::new();
    while let Read::Record(rec) = r.next() {
        out.push(rec);
    }
    out
}

#[test]
fn streams_and_stops() {
    let e = fake(vec![], 4096);
    let (t, end) = run(e.submit(Request::new(vec![5], 4)).unwrap());
    assert_eq!(
        (ids(&t), done(&end)),
        (vec![6, 7, 8, 9], (Finish::Length, 4, 0))
    );
    let stop = Request {
        stop_tokens: vec![8],
        ..Request::new(vec![5], 10)
    };
    let (t, end) = run(e.submit(stop.clone()).unwrap());
    assert_eq!((ids(&t), done(&end).0), (vec![6, 7, 8], Finish::Eos));
    let (t, _) = run(e
        .submit(Request {
            ignore_eos: true,
            max_tokens: 5,
            ..stop
        })
        .unwrap());
    // 8 is banned (#61); FakeModel's other logits tie and ties go to the lower id.
    assert_eq!(ids(&t), vec![6, 7, 0, 1, 2]);
    // Eight at once all finish, each with its own stream.
    let hs: Vec<_> = (0..8)
        .map(|i| e.submit(Request::new(vec![i, i + 1], 3)).unwrap())
        .collect();
    for (i, h) in hs.into_iter().enumerate() {
        let i = i as u32;
        assert_eq!(ids(&run(h).0), vec![i + 2, i + 3, i + 4]);
    }
}

#[test]
fn admission() {
    let e = fake(vec![], 64);
    let err = |r: Request| e.submit(r).err().unwrap();
    assert_eq!(
        err(Request::new(vec![], 4)),
        EngineError::InvalidRequest(invalid::EMPTY_PROMPT)
    );
    assert_eq!(
        err(Request::new(vec![1], 0)),
        EngineError::InvalidRequest(invalid::NO_TOKENS)
    );
    assert_eq!(
        err(Request::new(vec![64], 1)),
        EngineError::InvalidRequest(invalid::TOKEN_OUT_OF_VOCAB)
    );
    assert_eq!(
        err(Request::new(vec![1; 60], 10)),
        EngineError::PromptTooLong {
            tokens: 70,
            limit: 64
        }
    );
    for sampling in [
        SamplingParams {
            repeat_penalty: 0.0,
            ..SamplingParams::GREEDY
        },
        SamplingParams {
            temperature: f32::NAN,
            ..SamplingParams::GREEDY
        },
        SamplingParams {
            top_p: 1.5,
            ..SamplingParams::GREEDY
        },
        SamplingParams {
            top_k: 65,
            ..SamplingParams::GREEDY
        },
    ] {
        let mut request = Request::new(vec![1], 1);
        request.sampling = sampling;
        assert_eq!(err(request), EngineError::InvalidRequest(invalid::SAMPLING));
    }
    let score = |ids: Vec<u32>, repeat_penalty: f32| Request {
        score_ids: ids,
        sampling: SamplingParams {
            repeat_penalty,
            ..SamplingParams::GREEDY
        },
        ..Request::new(vec![1], 1)
    };
    assert_eq!(
        err(score(vec![2, 64], 1.0)),
        EngineError::InvalidRequest(invalid::TOKEN_OUT_OF_VOCAB)
    );
    for penalty in [1.1, 0.9] {
        assert_eq!(
            err(score(vec![2], penalty)),
            EngineError::InvalidRequest(invalid::SCORE_PENALTY)
        );
    }
    assert_eq!(
        err(score(vec![2; MAX_SCORE_IDS + 1], 1.0)),
        EngineError::InvalidRequest(invalid::SCORE_IDS)
    );
    let (t, _) = run(e.submit(score(vec![2; MAX_SCORE_IDS], 1.0)).unwrap());
    assert_eq!(t[0].1.len(), MAX_SCORE_IDS, "the cap itself is admitted");
    let stops = |n: u32| Request {
        stop_tokens: (0..n).collect(),
        ignore_eos: true,
        ..Request::new(vec![1], 4)
    };
    assert_eq!(
        err(stops(64)),
        EngineError::InvalidRequest(invalid::IGNORED_STOPS)
    );
    let (t, end) = run(e.submit(stops(63)).unwrap());
    assert_eq!((ids(&t), done(&end).0), (vec![63; 4], Finish::Length));
    let small = start_with(
        fake_opts(vec![], 512),
        Box::new(|_, _, _| Ok(Box::new(FakeModel::new(8)) as Box<dyn Model>)),
    )
    .unwrap();
    assert_eq!(
        small.submit(stops(8)).err(),
        Some(EngineError::InvalidRequest(invalid::IGNORED_STOPS))
    );
    let (t, end) = run(small.submit(stops(7)).unwrap());
    assert_eq!((ids(&t), done(&end).0), (vec![7; 4], Finish::Length));
    // A malformed request must not fault the engine or prevent a later valid request.
    let (_, end) = run(e.submit(Request::new(vec![1], 1)).unwrap());
    assert_eq!(done(&end).0, Finish::Length);
    // top_p 0 is the registry's minimum (top_p_default): it keeps the top candidate.
    let mut request = Request::new(vec![1], 1);
    request.sampling = SamplingParams {
        temperature: 1.0,
        top_p: 0.0,
        ..SamplingParams::GREEDY
    };
    assert_eq!(done(&run(e.submit(request).unwrap()).1).0, Finish::Length);
    let full = fake(vec![(knob_id::MAX_WAITING, 0.0)], 4096);
    assert!(matches!(
        full.submit(Request::new(vec![1], 1)),
        Err(EngineError::Overloaded { limit: 0, .. })
    ));
}

/// A bad ring capacity, pool fraction (registry bounds) or zero context fails load once, in the
/// engine (rejected `--set` overrides are reported instead, see
/// `effective_config_matches_the_engine`); Off is a working tier.
#[test]
fn bad_startup_options_are_load_errors() {
    let bad: [fn(&mut EngineOptions); 5] = [
        |o| o.max_context = 0,
        |o| o.ring_capacity = 3,
        |o| o.ring_capacity = 1,
        |o| o.kv_pool_fraction = f32::NAN,
        |o| o.kv_pool_fraction = 0.2,
    ];
    for f in bad {
        let mut opts = fake_opts(vec![], 4096);
        f(&mut opts);
        let result = start_with(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)) as Box<dyn Model>)),
        );
        assert_eq!(
            result.err(),
            Some(LoadError::Backend(KError::InvalidArg.code()))
        );
    }
    let mut opts = fake_opts(vec![], 4096);
    opts.telemetry = TelemetryTier::Off;
    let e = start_with(
        opts,
        Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)) as Box<dyn Model>)),
    )
    .unwrap();
    assert_eq!(
        done(&run(e.submit(Request::new(vec![1], 2)).unwrap()).1).1,
        2
    );
    assert!(records(&e).is_empty());
}

/// 10.1.1 StopSeq: the acknowledgement parks exactly the fed tokens under the session and returns
/// what the engine decided, even with a cancel racing it. Cancellation wins over a stop.
#[test]
fn finish_stop_parks_fed_tokens_once() {
    let e = fake(vec![(knob_id::MAX_BATCH_SEQS, 1.0)], 4096);
    let r = Request {
        session: 7,
        keep_kv: true,
        ..Request::new(vec![1, 2, 3], 1000)
    };
    let h = e.submit(r).unwrap();
    assert!(matches!(event_of(&h), EngineEvent::Token { .. }));
    let (usage, ..) = e.finish_stop(&h).unwrap();
    assert!(usage.completion_tokens >= 1);
    assert_eq!(e.parked_tokens(7), 3 + usage.completion_tokens - 1);
    assert_eq!(
        e.finish_stop(&h),
        Err(EngineError::InvalidRequest(invalid::TERMINAL_TAKEN)),
        "the terminal was taken"
    );

    // A stop for a request still queued behind the one batch slot is a cancellation: no span
    // claims a first token, and the session keeps its parked sequence. At a millisecond a step,
    // `busy` holds the slot for seconds.
    let (slow, _) = delayed(1000, vec![(knob_id::MAX_BATCH_SEQS, 1.0)]);
    let kept = Request {
        session: 7,
        keep_kv: true,
        ..Request::new(vec![1, 2, 3], 2)
    };
    run(slow.submit(kept.clone()).unwrap());
    let parked = slow.parked_tokens(7);
    assert!(parked > 0);
    let busy = slow.submit(Request::new(vec![1], 4000)).unwrap();
    let queued = slow.submit(kept).unwrap();
    assert_eq!(slow.finish_stop(&queued), Err(EngineError::Cancelled));
    assert_eq!(slow.parked_tokens(7), parked);
    drop(busy);

    // The result is what the engine recorded at the step boundary, even with a cancel racing it,
    // and a cancelled stop never leaves KV parked. Each span is read as soon as its stop returns,
    // before later requests can wrap the ring.
    for session in 100..300 {
        let h = e
            .submit(Request {
                session,
                keep_kv: true,
                ..Request::new(vec![1, 2, 3], 1000)
            })
            .unwrap();
        event_of(&h);
        let canceller = h.canceller();
        let race = std::thread::spawn(move || canceller.cancel());
        let stopped = e.finish_stop(&h);
        race.join().unwrap();
        assert!(
            stopped.is_ok() || e.parked_tokens(session) == 0,
            "{stopped:?}"
        );
        let recorded = records(&e).into_iter().rev().find_map(|r| match r {
            Record::RequestSpan(_, s) if s.req_id == h.id => Some(s.outcome() == Outcome::Ok),
            _ => None,
        });
        assert_eq!(recorded, Some(stopped.is_ok()), "request {}", h.id);
        e.release_session(session);
    }

    let h = e.submit(Request::new(vec![1], 4000)).unwrap();
    event_of(&h);
    h.canceller().cancel();
    assert_eq!(e.finish_stop(&h), Err(EngineError::Cancelled));
    let h = e.submit(Request::new(vec![1], 4000)).unwrap();
    assert_eq!(
        fake(vec![], 4096).finish_stop(&h),
        Err(EngineError::InvalidRequest(invalid::OTHER_ENGINE))
    );
}

/// Row R3: a cancelled request frees its KV; used blocks return to 0, read from the ring.
#[test]
fn cancel_frees_everything() {
    let e = fake(vec![], 4096);
    let h = e.submit(Request::new(vec![1], 1000)).unwrap();
    for _ in 0..3 {
        assert!(matches!(event_of(&h), EngineEvent::Token { .. }));
    }
    drop(h); // cancels
    run(e.submit(Request::new(vec![1], 2)).unwrap());
    // Done is sent inside the step; this round trip waits for the step's EngineSample.
    assert_eq!(e.parked_tokens(0), 0);
    let (mut cancelled, mut last_used) = (0, None);
    for rec in records(&e) {
        match rec {
            Record::RequestSpan(_, s) if s.outcome() == Outcome::Cancelled => cancelled += 1,
            Record::EngineSample(_, s) => last_used = Some(s.kv_blocks_used),
            _ => {}
        }
    }
    assert_eq!((cancelled, last_used), (1, Some(0)));
}

/// A prompt longer than `prefill_chunk_tokens` runs in chunks, steps with no sampled row included.
#[test]
fn chunked_prefill() {
    let e = fake(vec![(knob_id::PREFILL_CHUNK_TOKENS, 64.0)], 4096);
    let (t, end) = run(e.submit(Request::new(vec![5; 200], 3)).unwrap());
    assert_eq!((ids(&t), done(&end).0), (vec![6, 7, 8], Finish::Length));
    let chunks: Vec<u32> = records(&e)
        .into_iter()
        .filter_map(|r| match r {
            Record::EngineSample(_, s) if s.n_prefill_tokens > 0 => Some(s.n_prefill_tokens),
            _ => None,
        })
        .collect();
    assert_eq!(chunks, vec![64, 64, 64, 8]);

    // On the tiny model, chunked and whole prefill give the same next-token distribution.
    let p: Vec<u32> = (0..200).map(|i| (i * 13 + 5) % 250).collect();
    let mut opts = EngineOptions::new(tiny_path(), CPU);
    opts.knob_overrides = vec![(knob_id::PREFILL_CHUNK_TOKENS, 64.0)];
    let chunked = start(opts).unwrap();
    let first = |e: &EngineHandle| {
        let r = Request {
            logprobs: 5,
            ..Request::new(p.clone(), 1)
        };
        run(e.submit(r).unwrap()).0.remove(0).1
    };
    same_distribution(&first(&chunked), &first(&tiny_engine()));
}

/// A pool of 8 blocks (one 128-token context) with three sequences of 120 tokens: decodes run out of
/// blocks, the newest is preempted and recomputes its prompt and generated tokens, and every stream is
/// still exactly FakeModel's sequence.
#[test]
fn preemption_recomputes() {
    // The first step waits until all three are submitted, so they run together and preempt.
    let (go, wait) = std::sync::mpsc::channel();
    let model = Gated(FakeModel::new(64), Some(wait));
    let load = move |_: &_, _: &_, _| Ok(Box::new(model) as Box<dyn Model>);
    let opts = fake_opts(vec![(knob_id::MAX_BATCH_SEQS, 1.0)], 128);
    let e = start_with(opts, Box::new(load)).unwrap();
    assert_eq!(e.deployment().kv_pool_blocks, 8);
    let k = e.knobs();
    let more = KnobRequest::new(knob_id::MAX_BATCH_SEQS, 3.0, Authority::Human);
    assert!(matches!(k.set(more), KnobReply::Applied { .. }));
    let hs: Vec<_> = (0..3)
        .map(|i| e.submit(Request::new(vec![i; 40], 80)).unwrap())
        .collect();
    go.send(()).unwrap();
    for (i, h) in hs.into_iter().enumerate() {
        let (t, end) = run(h);
        let want: Vec<u32> = (1..=80).map(|j| (i as u32 + j) % 64).collect();
        assert_eq!((ids(&t), done(&end).0), (want, Finish::Length));
    }
    let preempted: u16 = records(&e)
        .into_iter()
        .filter_map(|r| match r {
            Record::RequestSpan(_, s) => Some(s.preemptions),
            _ => None,
        })
        .sum();
    assert!(preempted > 0, "nothing was preempted");
    // S05: what the steps computed beyond each request's fed positions (its prompt and all but its
    // last token) is the recompute, and the counter has all of it.
    assert_eq!(e.parked_tokens(0), 0); // the last step's sample is written
    let recs = records(&e);
    let samples: Vec<_> = recs
        .iter()
        .filter_map(|r| match r {
            Record::EngineSample(_, s) => Some(s),
            _ => None,
        })
        .collect();
    let rows: u64 = samples
        .iter()
        .map(|s| u64::from(s.n_prefill_tokens + s.n_decode_seqs))
        .sum();
    let spans: Vec<_> = recs
        .iter()
        .filter_map(|r| match r {
            Record::RequestSpan(_, s) => Some(s),
            _ => None,
        })
        .collect();
    let fed: u64 = spans
        .iter()
        .map(|s| u64::from(s.prompt_tokens + s.gen_tokens - 1))
        .sum();
    let recomputed = samples.last().unwrap().tot_recomputed_tokens;
    assert!(recomputed > 0);
    assert_eq!(recomputed, rows - fed);
    // A stream's longest token gap is one of its gaps: at least their mean, at most its span.
    for s in spans {
        let (gaps, ns) = (u64::from(s.gen_tokens - 1), s.t_done_ns - s.t_first_ns);
        assert!(
            s.max_gap_ns * gaps >= ns.saturating_sub(1_000_000) && s.max_gap_ns <= ns,
            "{s:?}"
        );
    }
}

/// S01's input: a step counts the waiting requests only when every seat is taken and the queue's
/// head would fit the KV pool, so `max_batch_seqs` alone holds them; 0 when the KV does too.
#[test]
fn cap_blocked_counts_only_what_the_cap_holds() {
    for (max_context, prompt, blocked) in [(4096, 4, 1), (128, 100, 0)] {
        let (go, wait) = std::sync::mpsc::channel();
        let model = Gated(FakeModel::new(64), Some(wait));
        let load = move |_: &_, _: &_, _| Ok(Box::new(model) as Box<dyn Model>);
        let opts = fake_opts(vec![(knob_id::MAX_BATCH_SEQS, 1.0)], max_context);
        let e = start_with(opts, Box::new(load)).unwrap();
        let hs: Vec<_> = (0..2)
            .map(|_| e.submit(Request::new(vec![1; prompt], 8)).unwrap())
            .collect();
        go.send(()).unwrap();
        hs.into_iter().for_each(|h| drop(run(h)));
        assert_eq!(e.parked_tokens(0), 0);
        let held: Vec<(u32, u32, u32)> = records(&e)
            .into_iter()
            .filter_map(|r| match r {
                Record::EngineSample(_, s) => Some((s.n_running, s.n_waiting, s.cap_blocked_ready)),
                _ => None,
            })
            .collect();
        assert!(held.contains(&(1, 1, blocked)), "{held:?}");
        assert!(held.iter().all(|&(_, w, b)| b <= w), "{held:?}");
        assert!(
            held.iter().filter(|s| s.1 == 0).all(|s| s.2 == 0),
            "{held:?}"
        );
    }
}

#[test]
fn knob_change_shows_in_the_next_sample() {
    let e = fake(vec![], 4096);
    let k = e.knobs();
    let set = |knob, value, who| k.set(KnobRequest::new(knob, value, who));
    let KnobReply::Applied {
        rev,
        previous,
        effective,
        ..
    } = set(knob_id::MAX_BATCH_SEQS, 8.0, Authority::Human)
    else {
        panic!("not applied");
    };
    assert_eq!((rev, previous, effective), (1, 16.0, 8.0));
    assert_eq!(k.effective().rev, 1);
    run(e.submit(Request::new(vec![1], 3)).unwrap());
    let mut seen_rev = false;
    let mut r = e.ring().reader(ReadFrom::Oldest);
    loop {
        match r.next() {
            Read::Record(Record::ConfigurationRevision(_, c)) if c.rev == 1 => seen_rev = true,
            Read::Record(Record::EngineSample(h, _)) if seen_rev => {
                assert_eq!(
                    h.config_rev, 1,
                    "the first sample after the change carries its revision"
                );
                break;
            }
            Read::Record(_) => {}
            other => panic!("no EngineSample after the revision: {other:?}"),
        }
    }

    // Rejections: past the scratch (sized for 1024 at the 512 default), a quality knob from gait,
    // a value off its step.
    assert_eq!(
        set(knob_id::PREFILL_CHUNK_TOKENS, 1088.0, Authority::Human),
        KnobReply::Rejected {
            reason: RejectReason::ExceedsFit { limit: 1024.0 }
        }
    );
    assert!(matches!(
        set(knob_id::TEMPERATURE_DEFAULT, 1.0, Authority::Gait),
        KnobReply::Rejected {
            reason: RejectReason::Unauthorized { .. }
        }
    ));
    assert!(matches!(
        set(knob_id::PREFILL_CHUNK_TOKENS, 100.0, Authority::Human),
        KnobReply::Rejected {
            reason: RejectReason::NotOnStep
        }
    ));
    // The declared inverse, once; then it is superseded by the inverse itself.
    assert!(matches!(
        k.revert(RevertTarget::Rev(1), Authority::Human),
        KnobReply::Applied {
            effective: 16.0,
            rev: 2,
            ..
        }
    ));
    assert_eq!(
        k.revert(RevertTarget::Rev(1), Authority::Human),
        KnobReply::Rejected {
            reason: RejectReason::Superseded { by_rev: 2 }
        }
    );
    // max_batch_seqs rises live up to the registry max; the scratch is sized for it.
    assert!(matches!(
        set(knob_id::MAX_BATCH_SEQS, 32.0, Authority::Human),
        KnobReply::Applied {
            rev: 3,
            effective: 32.0,
            ..
        }
    ));
    // A trial's changes revert together, and a second revert has nothing to do.
    let trial = KnobRequest {
        trial_id: 9,
        ..KnobRequest::new(knob_id::MAX_BATCH_SEQS, 4.0, Authority::Gait)
    };
    assert!(matches!(k.set(trial), KnobReply::Applied { .. }));
    assert!(matches!(
        k.set(KnobRequest {
            value: 2.0,
            ..trial
        }),
        KnobReply::Applied { .. }
    ));
    assert!(matches!(
        k.revert(RevertTarget::Trial(9), Authority::Gait),
        KnobReply::Applied {
            effective: 32.0,
            ..
        }
    ));
    assert_eq!(
        k.revert(RevertTarget::Trial(9), Authority::Gait),
        KnobReply::Rejected {
            reason: RejectReason::NothingToRevert
        }
    );
    // D7: gait cannot undo a human's change, here to a quality knob it may not set either.
    let KnobReply::Applied { rev, .. } = set(knob_id::TEMPERATURE_DEFAULT, 1.2, Authority::Human)
    else {
        panic!("not applied");
    };
    assert!(matches!(
        k.revert(RevertTarget::Rev(rev), Authority::Gait),
        KnobReply::Rejected {
            reason: RejectReason::Unauthorized { .. }
        }
    ));
    assert_eq!(
        k.effective().values[knob_id::TEMPERATURE_DEFAULT as usize],
        1.2
    );
    // The chunk rises live to the scratch's cap, which the deployment publishes.
    assert!(matches!(
        set(knob_id::PREFILL_CHUNK_TOKENS, 1024.0, Authority::Human),
        KnobReply::Applied {
            effective: 1024.0,
            ..
        }
    ));
    run(e.submit(Request::new(vec![1; 1100], 2)).unwrap());
    assert_eq!(e.deployment().prefill_chunk_max, 1024);
}

/// `effective()` reports what runs: the options, the clamps and the startup overrides; an override
/// the engine cannot apply is rejected in the ring and listed by `rejected_overrides`.
#[test]
fn effective_config_matches_the_engine() {
    let mut opts = fake_opts(
        vec![
            (knob_id::KV_POOL_FRACTION, 0.5),
            (knob_id::DECODE_GRAPHS, 1.0),
            (99, 1.0),
        ],
        1024,
    );
    opts.telemetry = TelemetryTier::T0;
    let e = start_with(
        opts,
        Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)) as Box<dyn Model>)),
    )
    .unwrap();
    let v = e.knobs().effective().values;
    let at = |k: u16| v[k as usize];
    assert_eq!(
        (
            at(knob_id::MAX_CONTEXT),
            at(knob_id::KV_POOL_FRACTION),
            at(knob_id::TELEMETRY_TIER),
            at(knob_id::DECODE_GRAPHS),
            at(knob_id::DEVICE)
        ),
        (1024.0, 0.5, 0.0, 0.0, f64::from(CPU))
    );
    assert_eq!(e.deployment().ctx_max, 1024);
    let rejected: Vec<(u16, u16)> = records(&e)
        .into_iter()
        .filter_map(|r| match r {
            Record::ActivationEvent(_, a) if a.result() == ActivationResult::Rejected => {
                Some((a.knob, a.reason))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        rejected,
        vec![
            (knob_id::DECODE_GRAPHS, RejectReason::Unsupported.code()),
            (99, RejectReason::UnknownKnob.code())
        ]
    );
    assert_eq!(
        e.rejected_overrides(),
        [
            (knob_id::DECODE_GRAPHS, 1.0, RejectReason::Unsupported),
            (99, 1.0, RejectReason::UnknownKnob)
        ]
    );
    // T0: one EngineSample per 64 steps, so none for a 3-token request.
    run(e.submit(Request::new(vec![1], 3)).unwrap());
    assert_eq!(e.parked_tokens(0), 0);
    let samples = records(&e)
        .into_iter()
        .filter(|r| matches!(r, Record::EngineSample(..)))
        .count();
    assert_eq!(samples, 0);
}

/// `shutdown` returns once the engine thread is done: health Draining, its HealthEvent in the ring,
/// and every later call refused.
#[test]
fn shutdown_drains() {
    let (e, _) = delayed(1000, vec![]);
    let other = e.clone();
    let h = e.submit(Request::new(vec![1], 4000)).unwrap(); // 4 s at least
    assert!(matches!(event_of(&h), EngineEvent::Token { .. }));
    e.shutdown();
    assert_eq!(other.health().state, HealthState::Draining);
    assert_eq!(
        run(h).1,
        EngineEvent::Error(EngineError::NotReady(HealthState::Draining))
    );
    assert_eq!(
        other.submit(Request::new(vec![1], 1)).err(),
        Some(EngineError::NotReady(HealthState::Draining))
    );
    let drained = records(&other)
        .into_iter()
        .any(|r| matches!(r, Record::HealthEvent(_, h) if h.to == HealthState::Draining.raw()));
    assert!(drained);
}

/// FakeModel with a per-step delay in microseconds that the test can change.
struct Delay(FakeModel, Arc<AtomicU64>);

impl Model for Delay {
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
        std::thread::sleep(Duration::from_micros(self.1.load(Ordering::Relaxed)));
        self.0.forward(s, b, kv, logits)
    }
}

/// FakeModel whose steps take `.1` µs, plus `.2` µs more on arm-B steps (read from the engine's
/// published config, which flips at the step boundary before the step runs).
struct ArmBDelay(
    FakeModel,
    u64,
    Arc<AtomicU64>,
    Arc<OnceLock<Arc<dyn KnobControl>>>,
);

impl Model for ArmBDelay {
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
        let arm_b = self.3.get().is_some_and(|k| k.effective().ab_arm == 2);
        let extra = if arm_b {
            self.2.load(Ordering::Relaxed)
        } else {
            0
        };
        std::thread::sleep(Duration::from_micros(self.1 + extra));
        self.0.forward(s, b, kv, logits)
    }
}

/// FakeModel whose first step waits for a message, so requests submitted meanwhile run together.
struct Gated(FakeModel, Option<std::sync::mpsc::Receiver<()>>);

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
        if let Some(go) = self.1.take() {
            let _ = go.recv();
        }
        self.0.forward(s, b, kv, logits)
    }
}

fn delayed(us: u64, overrides: Vec<(u16, f64)>) -> (EngineHandle, Arc<AtomicU64>) {
    let d = Arc::new(AtomicU64::new(us));
    let model = Delay(FakeModel::new(64), d.clone());
    let load = move |_: &_, _: &_, _| Ok(Box::new(model) as Box<dyn Model>);
    (
        start_with(fake_opts(overrides, 4096), Box::new(load)).unwrap(),
        d,
    )
}

/// FakeModel that fails (or panics) at its `n`th forward.
struct Faulty {
    inner: FakeModel,
    n: u32,
    panic: bool,
}

impl Model for Faulty {
    fn config(&self) -> &ModelConfig {
        self.inner.config()
    }

    fn forward(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        self.n -= 1;
        if self.n == 0 {
            if self.panic {
                panic!("injected");
            }
            return Err(KError::DeviceLost);
        }
        self.inner.forward(s, b, kv, logits)
    }
}

/// A model whose `at`th forward waits twice on `hold` before it runs: the test queues work behind
/// that step between the two waits, so the engine takes all of it at the next boundary.
struct Held {
    inner: Box<dyn Model>,
    at: u32,
    hold: Arc<Barrier>,
}

impl Model for Held {
    fn config(&self) -> &ModelConfig {
        self.inner.config()
    }

    fn forward(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        if self.at == 1 {
            self.hold.wait();
            self.hold.wait();
        }
        self.at = self.at.saturating_sub(1);
        self.inner.forward(s, b, kv, logits)
    }
}

/// 9.11: a fault or a panic in a step fails the running request with its error and the queued one
/// with `NotReady(Failed)`, and leaves health Failed.
#[test]
fn fault_fails_every_request() {
    for (panic, want) in [
        (false, EngineError::DeviceLost),
        (true, EngineError::Internal(0)),
    ] {
        // One seat, so the second request stays queued.
        let opts = fake_opts(vec![(knob_id::MAX_BATCH_SEQS, 1.0)], 4096);
        let load = move |_: &_, _: &_, _| {
            let m = Faulty {
                inner: FakeModel::new(64),
                n: 3,
                panic,
            };
            Ok(Box::new(m) as Box<dyn Model>)
        };
        let e = start_with(opts, Box::new(load)).unwrap();
        let running = e.submit(Request::new(vec![1], 100)).unwrap();
        let queued = e.submit(Request::new(vec![2], 100)).unwrap();
        assert_eq!(run(running).1, EngineEvent::Error(want));
        assert_eq!(
            run(queued).1,
            EngineEvent::Error(EngineError::NotReady(HealthState::Failed))
        );
        assert_eq!(e.health().state, HealthState::Failed);
        let after = e.submit(Request::new(vec![1], 1)).err();
        assert_eq!(after, Some(EngineError::NotReady(HealthState::Failed)));
    }
}

/// The model lane rejects an unknown tensor type only when the model asks for that tensor.
#[test]
fn start_accepts_an_unused_tensor_of_unknown_type() {
    let bytes = tiny::gguf(Arch::Llama, 3);
    let g = gguf::Gguf::parse(&bytes).unwrap();
    let mut tensors: Vec<_> = g
        .tensors
        .iter()
        .map(|t| {
            let at = (g.data_offset + t.offset) as usize;
            let data = bytes[at..at + t.bytes as usize].to_vec();
            (t.name.clone(), t.dims.clone(), t.ggml_type, data)
        })
        .collect();
    tensors.push(("unused".into(), vec![4], 999, Vec::new()));
    let kv: Vec<_> = g.kv.into_iter().collect();
    let path =
        std::env::temp_dir().join(format!("larch-engine-unknown-{}.gguf", std::process::id()));
    std::fs::write(&path, gguf::write(&kv, &tensors)).unwrap();
    let e = start(EngineOptions::new(path.clone(), CPU));
    std::fs::remove_file(path).unwrap();
    e.unwrap().shutdown();
}

/// An MoE model reports its class, and keeping its experts in host memory (on the CPU device)
/// generates what keeping them on the device does: the same ops, bit for bit.
#[test]
fn moe_experts_in_host_memory_generate_the_same_tokens() {
    let path = std::env::temp_dir().join(format!("larch-engine-moe-{}.gguf", std::process::id()));
    std::fs::write(&path, tiny::gguf(Arch::Qwen3Moe, 3)).unwrap();
    let generate = |layers: Option<u32>| {
        let mut opts = EngineOptions::new(path.clone(), CPU);
        opts.cpu_expert_layers = layers;
        let e = start(opts).unwrap();
        let class = e.model_class();
        let (t, _) = run(e.submit(Request::new(vec![5, 9, 2, 200], 6)).unwrap());
        e.shutdown();
        (class, ids(&t))
    };
    let (planned, want) = generate(None);
    // The CPU reference plans no host experts.
    assert!(!planned.dense && !planned.experts_in_ram, "{planned:?}");
    for layers in [1, 99] {
        let (class, got) = generate(Some(layers));
        assert!(!class.dense && class.experts_in_ram, "{class:?}");
        assert_eq!(got, want, "{layers} host layers");
    }
    std::fs::remove_file(path).unwrap();
}

/// A Qwen 3.5 model, whose Gated DeltaNet layers keep a recurrent state per sequence, generates
/// the same tokens for a prompt alone, beside other prompts, and as a session's next turn: each
/// sequence gets a slot of its own, zeroed at position 0, and no start reuses KV without its state;
/// a cancelled request gives its slot back, and a prompt in chunks matches it whole. Each
/// comparison keeps both sides on one matmul route (16 rows or fewer, or more than 16), so the
/// tokens match exactly. The scheduler's proptest covers preemption and live batch cuts.
#[test]
fn hybrid_sequences_keep_their_own_state() {
    for arch in [Arch::Qwen35, Arch::Qwen35Moe] {
        let name = format!("larch-engine-{arch:?}-{}.gguf", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, tiny::gguf(arch, 3)).unwrap();
        let e = start(EngineOptions::new(path.clone(), CPU)).unwrap();
        let prompts: Vec<Vec<u32>> = [(4, 7), (5, 11), (6, 13)]
            .map(|(n, k)| (0..n).map(|i| (i * k + 3) % tiny::VOCAB).collect())
            .into();
        // Tokens with their top logprobs, which show a wrong state where the tiny model's tokens
        // may not.
        let generate = |p: &Vec<u32>, session| {
            let r = Request {
                logprobs: 5,
                session,
                keep_kv: session != 0,
                ..Request::new(p.clone(), 8)
            };
            e.submit(r).unwrap()
        };
        let alone: Vec<_> = prompts.iter().map(|p| run(generate(p, 0)).0).collect();
        let handles: Vec<_> = prompts.iter().map(|p| generate(p, 0)).collect();
        let together: Vec<_> = handles.into_iter().map(|h| run(h).0).collect();
        assert_eq!(together, alone, "{arch:?}: beside other prompts");
        let (first, _) = run(generate(&prompts[0], 7));
        let mut next = prompts[0].clone();
        next.extend(ids(&first));
        next.push(9);
        let fresh = run(generate(&next, 0)).0;
        assert_eq!(run(generate(&next, 7)).0, fresh, "{arch:?}: next turn");
        let long = e.submit(Request::new(prompts[2].clone(), 400)).unwrap();
        assert!(matches!(event_of(&long), EngineEvent::Token { .. }));
        long.cancel();
        assert_eq!(run(long).1, EngineEvent::Error(EngineError::Cancelled));
        let handles: Vec<_> = prompts.iter().map(|p| generate(p, 0)).collect();
        let after: Vec<_> = handles.into_iter().map(|h| run(h).0).collect();
        assert_eq!(after, alone, "{arch:?}: after a cancel");
        let prompt: Vec<u32> = (0..100).map(|i| (i * 17 + 5) % tiny::VOCAB).collect();
        let whole = run(generate(&prompt, 0)).0;
        let mut opts = EngineOptions::new(path.clone(), CPU);
        opts.knob_overrides = vec![(knob_id::PREFILL_CHUNK_TOKENS, 64.0)];
        let chunked = start(opts).unwrap();
        let r = Request {
            logprobs: 5,
            ..Request::new(prompt, 8)
        };
        assert_eq!(
            run(chunked.submit(r).unwrap()).0,
            whole,
            "{arch:?}: in chunks"
        );
        chunked.shutdown();
        e.shutdown();
        std::fs::remove_file(path).unwrap();
    }
}

/// A hybrid session's next turn runs on from the state its last turn parked: at its snapshot (the
/// chat history, before the generation prompt) or at the turn's end. Each turn gives the tokens and
/// top logprobs a fresh run of its prompt gives, and reuses exactly the parked tokens, or none when
/// the prompt does not repeat them all. Every step holds at most 16 rows, so all of them multiply
/// alike (larch-model's tests/qwen35.rs).
#[test]
fn hybrid_turns_run_on_from_the_parked_state() {
    for arch in [Arch::Qwen35, Arch::Qwen35Moe] {
        let name = format!("larch-engine-turns-{arch:?}-{}.gguf", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, tiny::gguf(arch, 3)).unwrap();
        let e = start(EngineOptions::new(path.clone(), CPU)).unwrap();
        let generate = |prompt: &[u32], session, snapshot: usize| {
            let r = Request {
                logprobs: 5,
                session,
                keep_kv: session != 0,
                history_end: snapshot as u32,
                ..Request::new(prompt.to_vec(), 4)
            };
            run(e.submit(r).unwrap())
        };
        let check = |prompt: &[u32], snapshot, reused: usize, what: &str| {
            let (tokens, end) = generate(prompt, 5, snapshot);
            assert_eq!(tokens, generate(prompt, 0, 0).0, "{arch:?}: {what}");
            assert_eq!(done(&end).2, reused as u32, "{arch:?}: {what}, reused");
            tokens
        };
        let g = [7, 8]; // the generation prompt, which no later prompt repeats
        let h1 = [3, 1, 4];
        let h2 = [&h1[..], &[9, 2, 5]].concat();
        let p3 = [&h2[..], &[5, 8], &g].concat();
        check(&[&h1[..], &g].concat(), h1.len(), 0, "turn 1");
        // Turn 2 rewrites turn 1's answer: it reuses the history up to the snapshot.
        check(
            &[&h2[..], &g].concat(),
            h2.len(),
            h1.len(),
            "from a snapshot",
        );
        // Turn 3 parks at its end, which turn 4 repeats with all of turn 3's answer.
        let third = check(&p3, 0, h2.len(), "from the next snapshot");
        let p4 = [&p3[..], &ids(&third)].concat();
        check(&p4, 0, p3.len() + 3, "from the turn's end");
        check(&[2, 7, 1], 0, 0, "a new chat");
        e.shutdown();
        std::fs::remove_file(path).unwrap();
    }
}

/// The first generated token's top logprobs agree between two runs (ids present in both).
fn same_distribution(a: &[(u32, f32)], b: &[(u32, f32)]) {
    let common: Vec<(f32, f32)> = a
        .iter()
        .filter_map(|(i, x)| b.iter().find(|(j, _)| j == i).map(|(_, y)| (*x, *y)))
        .collect();
    assert!(
        common.len() >= 4,
        "top-5 lists barely overlap: {a:?} vs {b:?}"
    );
    let diff = common.iter().fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(diff < 1e-2, "logprobs differ by {diff}: {a:?} vs {b:?}");
}

fn turn(
    e: &EngineHandle,
    prompt: Vec<u32>,
    max_tokens: u32,
    session: u64,
) -> (Tokens, EngineEvent) {
    let r = Request {
        logprobs: 5,
        session,
        keep_kv: session != 0,
        ..Request::new(prompt, max_tokens)
    };
    run(e.submit(r).unwrap())
}

/// A score request returns exact logprobs for the ids it names, in its order, even outside the top 64.
#[test]
fn score_ids_are_exact() {
    let e = tiny_engine();
    let prompt: Vec<u32> = (0..20).map(|i| (i * 5 + 1) % 250).collect();
    let r = Request {
        logprobs: 20,
        ..Request::new(prompt.clone(), 1)
    };
    let top = run(e.submit(r).unwrap()).0.remove(0).1;
    let all: Vec<u32> = (0..tiny::VOCAB).rev().collect();
    let r = Request {
        score_ids: all.clone(),
        ..Request::new(prompt, 1)
    };
    let scored = run(e.submit(r).unwrap()).0.remove(0).1;
    assert_eq!(scored.iter().map(|s| s.0).collect::<Vec<_>>(), all);
    let mass: f64 = scored.iter().map(|s| f64::from(s.1).exp()).sum();
    assert!((mass - 1.0).abs() < 1e-3, "probabilities sum to {mass}");
    for (id, lp) in top {
        let s = scored.iter().find(|s| s.0 == id).unwrap().1;
        assert!((s - lp).abs() < 1e-5, "id {id}: {s} vs {lp}");
    }
}

/// FakeModel whose predicted id gets its own value as its logit instead of 100, so rows that predict
/// different ids have different LSEs.
struct HotIsId(FakeModel);

impl Model for HotIsId {
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
        let mut l = vec![0f32; logits.shape().iter().product::<i64>() as usize];
        s.download(logits.buffer(), logits.offset(), &mut l)?;
        for row in l.chunks_mut(self.0.config().vocab as usize) {
            for (id, v) in row.iter_mut().enumerate() {
                if *v != 0.0 {
                    *v = id as f32;
                }
            }
        }
        s.upload(logits.buffer(), logits.offset(), &l)
    }
}

/// Score requests started in one step each read their own logits row and that row's LSE, not
/// another row's, with a request that scores nothing between them.
#[test]
fn score_ids_read_their_own_row() {
    let model = HotIsId(FakeModel::new(64));
    let load = move |_: &_, _: &_, _| Ok(Box::new(model) as Box<dyn Model>);
    let e = start_with(fake_opts(vec![], 64), Box::new(load)).unwrap();
    // A prompt ending in t predicts t + 1 with logit t + 1; the other 63 ids have 0.
    let score = |last: u32| Request {
        score_ids: vec![11, 21],
        ..Request::new(vec![1, last], 1)
    };
    let group = vec![score(10), Request::new(vec![1, 5], 1), score(20)];
    let hs = e.submit_group(group).unwrap();
    for (h, own) in hs.into_iter().zip([Some(11), None, Some(21)]) {
        let (id, top) = run(h).0.remove(0);
        let Some(own) = own else {
            assert_eq!((id, top), (6, vec![]), "the request that scores nothing");
            continue;
        };
        assert_eq!(id, own);
        assert_eq!(top.iter().map(|s| s.0).collect::<Vec<_>>(), [11, 21]);
        let lse = (f64::from(own).exp() + 63.0).ln();
        for (id, lp) in top {
            let want = f64::from(if id == own { own } else { 0 }) - lse;
            let lp = f64::from(lp);
            assert!(
                (lp - want).abs() < 1e-4,
                "{own}: id {id} at {lp}, want {want}"
            );
        }
    }
}

/// A step's largest batch, every request at the score cap, fits the step's buffers, and each request
/// reads its own scores.
#[test]
fn score_ids_fill_a_full_batch() {
    let e = fake(vec![(knob_id::MAX_BATCH_SEQS, 64.0)], 64);
    // Each request scores the id it predicts, logit 100 against 0: about 0, where another
    // request's id would be about -100.
    let group = (0..64)
        .map(|t| Request {
            score_ids: vec![(t + 1) % 64; MAX_SCORE_IDS],
            ..Request::new(vec![1, t], 1)
        })
        .collect();
    for (t, h) in e.submit_group(group).unwrap().into_iter().enumerate() {
        let (id, top) = run(h).0.remove(0);
        assert_eq!(id, (t as u32 + 1) % 64);
        assert_eq!(top.len(), MAX_SCORE_IDS);
        assert!(top.iter().all(|&(s, lp)| s == id && lp.abs() < 1e-3), "{t}");
    }
}

/// D14 one_token_then_continue: a turn that generates one token and stops (by length, and by a stop
/// token) parks exactly its fed tokens; the next turn reuses that many, never one more, and its logits
/// match a fresh prefill of the same prompt.
#[test]
fn one_token_then_continue() {
    let e = tiny_engine();
    let p1: Vec<u32> = (0..30).map(|i| (i * 7 + 3) % 250).collect();
    for (session, stop) in [(1u64, false), (2, true)] {
        let first = if stop {
            let g = ids(&turn(&e, p1.clone(), 1, 0).0)[0];
            let r = Request {
                stop_tokens: vec![g],
                session,
                keep_kv: true,
                ..Request::new(p1.clone(), 5)
            };
            let (t, end) = run(e.submit(r).unwrap());
            assert_eq!(done(&end).0, Finish::Eos);
            t
        } else {
            let (t, end) = turn(&e, p1.clone(), 1, session);
            assert_eq!(done(&end).0, Finish::Length);
            t
        };
        assert_eq!(
            e.parked_tokens(session),
            30,
            "the sampled token is not in the KV"
        );
        let mut p2 = p1.clone();
        p2.extend([first[0].0, 11, 12, 13]);
        let (cached, end) = turn(&e, p2.clone(), 3, session);
        assert_eq!(done(&end).2, 30);
        let (fresh, end) = turn(&e, p2, 3, 0);
        assert_eq!(done(&end).2, 0);
        same_distribution(&cached[0].1, &fresh[0].1);
    }
}

/// D14 cancel_then_continue: a cancelled turn parks nothing; the next turn re-prefills and is correct.
#[test]
fn cancel_then_continue() {
    let e = tiny_engine();
    let p1: Vec<u32> = (0..20).map(|i| (i * 5 + 1) % 250).collect();
    let h = e
        .submit(Request {
            session: 3,
            keep_kv: true,
            ..Request::new(p1.clone(), 200)
        })
        .unwrap();
    let EngineEvent::Token { id: g, .. } = event_of(&h) else {
        panic!("no token")
    };
    drop(h);
    let mut p2 = p1.clone();
    p2.extend([g, 40, 41]);
    let (cached, end) = turn(&e, p2.clone(), 2, 3);
    assert_eq!(done(&end).2, 0, "nothing was parked");
    let (fresh, _) = turn(&e, p2, 2, 0);
    same_distribution(&cached[0].1, &fresh[0].1);
}

/// D14 reset_race (engine side): a turn under a released session id parks nothing, whenever the
/// release lands (the release mid-turn is the scheduler test of the same name); a new id parks as
/// usual.
#[test]
fn release_is_sticky() {
    let e = tiny_engine();
    let p: Vec<u32> = (0..20).collect();
    e.release_session(4);
    let (_, end) = turn(&e, p.clone(), 6, 4);
    assert_eq!(done(&end).0, Finish::Length);
    assert_eq!(e.parked_tokens(4), 0);
    turn(&e, p, 2, 5);
    assert_eq!(e.parked_tokens(5), 21);
}

/// S03 and S04's inputs across five turns of one conversation on an 8-block pool: the first turn,
/// a full reuse, a reuse after another request reclaimed part of the parked KV (its ReclaimEvent
/// joins the span by id, and precedes it in the ring), a diverged prompt, and a turn after a reload
/// dropped the KV.
#[test]
fn session_turns_record_their_reuse() {
    let e = fake(vec![(knob_id::MAX_BATCH_SEQS, 1.0)], 128);
    assert_eq!(e.deployment().kv_pool_blocks, 8);
    let fields = |s: &RequestSpan| {
        (
            s.turn_id,
            s.reuse_miss(),
            s.lcp_tokens,
            s.parked_kv_at_match,
            s.reuse_available_tokens,
            s.prior_kv_applies,
            s.cached_tokens,
        )
    };
    let run_turn = |prompt: Vec<u32>| {
        let h = e.submit(kept(7, prompt, 2)).unwrap();
        let id = h.id;
        let tokens = ids(&run(h).0);
        e.parked_tokens(7); // Done comes before the step's span reaches the ring
        (tokens, span(&e, id))
    };
    // 20 prompt tokens and 21, 22: 21 fed and parked.
    let first: Vec<u32> = (1..=20).collect();
    let (out, s) = run_turn(first.clone());
    assert_eq!(out, [21, 22]);
    assert_eq!(fields(&s), (1, ReuseMiss::FirstTurn, 0, 0, 0, 0, 0));
    // The whole history and a new token: 41, 42 follow 40, and 24 tokens park in two blocks.
    let second: Vec<u32> = (1..=22).chain([40]).collect();
    let (_, s) = run_turn(second.clone());
    assert_eq!(fields(&s), (2, ReuseMiss::None, 21, 21, 21, 1, 21));
    assert_eq!(e.parked_tokens(7), 24);
    // 100 tokens need seven of the eight blocks: one comes from the parked two, 8 tokens with it.
    run(e.submit(Request::new(vec![50; 100], 2)).unwrap());
    assert_eq!(e.parked_tokens(7), 16);
    let reclaims: Vec<_> = records(&e)
        .into_iter()
        .filter_map(|r| match r {
            Record::ReclaimEvent(_, c) => Some(c),
            _ => None,
        })
        .collect();
    let [demand] = reclaims[..] else {
        panic!("{reclaims:?}")
    };
    assert_eq!(
        (
            demand.session_id,
            demand.blocks,
            demand.tokens,
            demand.admitted_work_tokens
        ),
        (7, 1, 8, 100)
    );
    assert_eq!(demand.reason(), ReclaimReason::Demand);
    // The history still matches all 24 fed tokens; only the 16 parked are reused.
    let third: Vec<u32> = second.iter().copied().chain([41, 42, 60]).collect();
    let (_, s) = run_turn(third);
    assert_eq!(fields(&s), (3, ReuseMiss::Reclaimed, 24, 16, 16, 1, 16));
    assert_eq!(s.reclaim_id, demand.reclaim_id);
    // A prompt that leaves the history at its first token.
    let (_, s) = run_turn(vec![9, 9, 9]);
    assert_eq!(fields(&s), (4, ReuseMiss::PrefixDiverged, 0, 27, 0, 1, 0));
    assert_eq!(s.reclaim_id, 0);
    // A reload drops the parked KV as a Reload reclaim; the next turn still sees its history.
    let parked = e.parked_tokens(7);
    e.reload_with(
        ReloadChange::default(),
        Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)) as Box<dyn Model>)),
    )
    .unwrap();
    let reload = records(&e)
        .into_iter()
        .rev()
        .find_map(|r| match r {
            Record::ReclaimEvent(_, c) => Some(c),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        (reload.session_id, reload.tokens, reload.reason()),
        (7, parked, ReclaimReason::Reload)
    );
    let (_, s) = run_turn(vec![9, 9, 9, 10]);
    assert_eq!(fields(&s), (5, ReuseMiss::Reloaded, 4, 0, 0, 0, 0));
    assert_eq!(s.reclaim_id, reload.reclaim_id);
    // A span's reclaim reached the ring before the span.
    let mut written = Vec::new();
    for r in records(&e) {
        match r {
            Record::ReclaimEvent(_, c) => written.push(c.reclaim_id),
            Record::RequestSpan(_, s) if s.reclaim_id != 0 => {
                assert!(written.contains(&s.reclaim_id), "{s:?} before {written:?}")
            }
            _ => {}
        }
    }
}

/// The ring's span for request `id`.
fn span(e: &EngineHandle, id: u64) -> RequestSpan {
    records(e)
        .into_iter()
        .rev()
        .find_map(|r| match r {
            Record::RequestSpan(_, s) if s.req_id == id => Some(s),
            _ => None,
        })
        .expect("span")
}

/// A turn of conversation `session` that parks its KV at Done.
fn kept(session: u64, prompt: Vec<u32>, max_tokens: u32) -> Request {
    Request {
        session,
        keep_kv: true,
        ..Request::new(prompt, max_tokens)
    }
}

/// The handle's terminal events, spans and parked KV agree across a conversation (10.1.1).
#[test]
fn terminal_rows_end_to_end() {
    // ponytail: a 4 s margin (4000 steps at 1 ms) before Length lets the stop and the cancel below
    // land first; gate every step on the test if a loaded runner stalls this thread that long.
    let (e, _) = delayed(1000, vec![]);
    // Length: [1, 2, 3] gives 4, 5 and parks [1, 2, 3, 4]; 5 was never fed.
    let h = e.submit(kept(9, vec![1, 2, 3], 2)).unwrap();
    let id = h.id;
    let (t, end) = run(h);
    assert_eq!((ids(&t), done(&end)), (vec![4, 5], (Finish::Length, 2, 0)));
    assert_eq!(e.parked_tokens(9), 4); // also waits for the step's span
    let s = span(&e, id);
    assert_eq!((s.outcome(), s.finish()), (Outcome::Ok, Finish::Length));

    // Error, rejected at intake: the parked sequence stays as it was.
    assert!(matches!(
        e.submit(kept(9, vec![1; 4096], 1)),
        Err(EngineError::PromptTooLong { .. })
    ));
    assert_eq!(
        e.submit(kept(9, vec![1, 99], 1)).err(),
        Some(EngineError::InvalidRequest(invalid::TOKEN_OUT_OF_VOCAB))
    );
    assert_eq!(e.parked_tokens(9), 4);
    let rejected = records(&e)
        .into_iter()
        .filter(|r| {
            matches!(r, Record::RequestSpan(_, s) if s.session_id == 9
                && s.outcome() == Outcome::Rejected && s.finish() == Finish::None)
        })
        .count();
    assert_eq!(rejected, 2);

    // StopSeq: the history 4, 5 and a new message 6; the stream matches a stop string on the first
    // token, 7. Every fed token parks, but not the last sampled one, wherever the acknowledgement
    // landed.
    let h = e.submit(kept(9, vec![1, 2, 3, 4, 5, 6], 4000)).unwrap();
    skip(&h, 1);
    let (usage, ..) = e.finish_stop(&h).unwrap();
    assert_eq!((usage.prompt_tokens, usage.cached_tokens), (6, 4));
    assert_eq!(e.parked_tokens(9), 6 + usage.completion_tokens - 1);
    let s = span(&e, h.id);
    assert_eq!(
        (s.outcome(), s.finish(), s.gen_tokens, s.cached_tokens),
        (Outcome::Ok, Finish::StopSeq, usage.completion_tokens, 4)
    );

    // Cancelled: the next turn's history leaves out the stop string, so its reuse ends at 6. The
    // cancel frees all of it, the reused prefix included.
    let h = e.submit(kept(9, vec![1, 2, 3, 4, 5, 6, 20], 4000)).unwrap();
    let id = h.id;
    skip(&h, 1);
    drop(h);
    // The next turn re-prefills the history once.
    let (_, end) = run(e
        .submit(kept(9, vec![1, 2, 3, 4, 5, 6, 20, 21, 30], 1))
        .unwrap());
    assert_eq!(done(&end), (Finish::Length, 1, 0));
    assert_eq!(e.parked_tokens(9), 9);
    let s = span(&e, id);
    assert_eq!(
        (s.outcome(), s.finish(), s.error, s.cached_tokens),
        (
            Outcome::Cancelled,
            Finish::None,
            EngineError::Cancelled.code(),
            6
        )
    );

    // Eos (a stop token): reuses every parked token and never one more; parks like Length.
    let r = Request {
        stop_tokens: vec![42],
        ..kept(9, vec![1, 2, 3, 4, 5, 6, 20, 21, 30, 31, 40], 10)
    };
    let (t, end) = run(e.submit(r).unwrap());
    assert_eq!((ids(&t), done(&end)), (vec![41, 42], (Finish::Eos, 2, 9)));
    assert_eq!(e.parked_tokens(9), 12);
}

/// A mid-turn fault emits a DeviceLost terminal event and span (10.1.1, 9.11).
#[test]
fn a_fault_ends_the_turn_device_lost() {
    let load = |_: &_, _: &_, _| {
        let m = Faulty {
            inner: FakeModel::new(64),
            n: 3,
            panic: false,
        };
        Ok(Box::new(m) as Box<dyn Model>)
    };
    let e = start_with(fake_opts(vec![], 4096), Box::new(load)).unwrap();
    run(e.submit(kept(9, vec![1, 2, 3], 1)).unwrap()); // forward 1
    assert_eq!(e.parked_tokens(9), 3);
    let h = e.submit(kept(9, vec![1, 2, 3, 4, 5], 100)).unwrap(); // forward 2 runs, 3 fails
    let id = h.id;
    assert_eq!(run(h).1, EngineEvent::Error(EngineError::DeviceLost));
    // The stopped-engine public fallback guarantees zero; this does not prove physical KV cleanup.
    assert_eq!(e.parked_tokens(9), 0);
    let ring = e.clone();
    e.shutdown(); // joins the engine thread, so its last spans are in the ring
    let s = span(&ring, id);
    assert_eq!(
        (s.outcome(), s.finish()),
        (Outcome::DeviceLost, Finish::None)
    );
}

/// A stop cannot resurrect released KV or replace a natural end's usage and finish (10.1.1).
#[test]
fn stop_races_a_release_or_the_last_token() {
    let (e, _) = delayed(1000, vec![]);
    // The 4 s margin of terminal_rows_end_to_end: the stop lands before Length.
    // Session drop releases without cancelling; either release order must leave no parked KV.
    for (session, release_first) in [(20u64, true), (21, false)] {
        let h = e.submit(kept(session, vec![1, 2, 3], 4000)).unwrap();
        skip(&h, 1);
        if release_first {
            e.release_session(session);
        }
        let (usage, ..) = e.finish_stop(&h).unwrap();
        assert_eq!(
            span(&e, h.id).finish(),
            Finish::StopSeq,
            "session {session}"
        );
        if !release_first {
            assert_eq!(
                e.parked_tokens(session),
                3 + usage.completion_tokens - 1,
                "session {session}"
            );
            e.release_session(session);
        }
        assert_eq!(e.parked_tokens(session), 0, "session {session}");
    }
    // The stop matched on the last token: Done(Length) was queued behind it, and that end stands.
    let h = e.submit(kept(22, vec![1, 2, 3], 2)).unwrap();
    skip(&h, 2);
    let (usage, ..) = e.finish_stop(&h).unwrap();
    assert_eq!(usage.completion_tokens, 2);
    assert_eq!(e.parked_tokens(22), 4);
    assert_eq!(span(&e, h.id).finish(), Finish::Length);
}

/// Stops racing shutdown or faults check valid outcomes; regression detection is probabilistic.
#[test]
fn a_stop_racing_the_engines_end() {
    // Either message may reach the engine first; both orders are valid outcomes.
    for i in 0..20 {
        let e = fake(vec![], 4096);
        let h = e.submit(kept(9, vec![1], 4000)).unwrap();
        skip(&h, 1);
        let other = e.clone();
        let stopping = std::thread::spawn(move || other.shutdown());
        let got = e.finish_stop(&h);
        stopping.join().unwrap();
        assert!(
            matches!(
                got,
                Ok(_) | Err(EngineError::NotReady(HealthState::Draining))
            ),
            "run {i}: {got:?}"
        );
    }

    // Forward 3 holds, then fails: a stop racing that fault gets DeviceLost.
    let hold = Arc::new(Barrier::new(2));
    let h2 = hold.clone();
    let load = move |_: &_, _: &_, _| {
        let faulty = Faulty {
            inner: FakeModel::new(64),
            n: 3,
            panic: false,
        };
        let m = Held {
            inner: Box::new(faulty),
            at: 3,
            hold: h2,
        };
        Ok(Box::new(m) as Box<dyn Model>)
    };
    let e = start_with(fake_opts(vec![], 4096), Box::new(load)).unwrap();
    let h = e.submit(kept(9, vec![1], 100)).unwrap();
    skip(&h, 2); // forwards 1 and 2
    hold.wait(); // the engine is inside forward 3
    let other = e.clone();
    let stop = std::thread::spawn(move || other.finish_stop(&h));
    hold.wait();
    assert_eq!(stop.join().unwrap(), Err(EngineError::DeviceLost));
    // The stopped-engine public fallback guarantees zero; this does not prove physical KV cleanup.
    assert_eq!(e.parked_tokens(9), 0);
}

/// Waits for `n` tokens of a running request.
fn skip(h: &RequestHandle, n: usize) {
    for _ in 0..n {
        assert!(matches!(event_of(h), EngineEvent::Token { .. }));
    }
}

fn forced(tokens: &[u32], prompt: u32, session: u64) -> Forced {
    Forced {
        tokens: tokens.to_vec(),
        prompt,
        session,
        keep_kv: false,
    }
}

/// A forced run's positions and its terminal event; it streams no tokens.
fn observe(e: &EngineHandle, f: Forced) -> (Vec<Position>, EngineEvent) {
    let ForcedHandle { request, positions } = e.force(f).unwrap();
    let (tokens, end) = run(request);
    assert!(tokens.is_empty(), "a forced run streams no tokens");
    (positions.try_iter().collect(), end)
}

fn agree(a: &[Position], b: &[Position]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert_eq!(x.pos, y.pos);
        same_distribution(&x.top[..5], &y.top[..5]);
    }
}

/// Section 15's modes through `force` on the tiny model: one-batch prefill, decode token by token,
/// chunked, cached (the reuse path, from the parked `n_kv` on) and eight in flight observe the same
/// distributions at every position.
#[test]
fn forced_modes_agree() {
    let e = tiny_engine();
    let refs: Vec<Vec<u32>> = (0..8)
        .map(|j| (0..40).map(|i| (i * 7 + j * 31 + 3) % 250).collect())
        .collect();
    let t = &refs[0];
    let (prefill, end) = observe(&e, forced(t, 40, 0));
    assert_eq!(done(&end), (Finish::Length, 1, 0));
    let pos: Vec<u32> = prefill.iter().map(|p| p.pos).collect();
    assert_eq!(pos, (0..40).collect::<Vec<_>>());
    for p in &prefill {
        assert_eq!(p.top.len(), 64);
        assert!(p.top[0].1 <= 0.0 && p.top.windows(2).all(|w| w[0].1 >= w[1].1));
    }
    let (decode, end) = observe(&e, forced(t, 1, 0));
    assert_eq!(done(&end), (Finish::Length, 40, 0));
    agree(&prefill, &decode);

    // Chunked, with decode rows in the chunk steps (D16): the decode-mode run is admitted first, as
    // a chunked run admitted first takes the whole chunk budget each step. Both wait behind a held
    // first step, so they are admitted at one boundary, in that order.
    let long: Vec<u32> = (0..150).map(|i| (i * 13 + 5) % 250).collect();
    let mut opts = EngineOptions::new(tiny_path(), CPU);
    opts.knob_overrides = vec![(knob_id::PREFILL_CHUNK_TOKENS, 64.0)];
    let hold = Arc::new(Barrier::new(2));
    let (h, path) = (hold.clone(), tiny_path());
    let load = move |dev: &_, s: &_, rows| {
        let m = Transformer::load(&path, dev, s, rows)?;
        let held = Held {
            inner: Box::new(m),
            at: 1,
            hold: h,
        };
        Ok(Box::new(held) as Box<dyn Model>)
    };
    let c = start_with(opts, Box::new(load)).unwrap();
    let first = c.submit(Request::new(vec![1], 1)).unwrap();
    hold.wait(); // step 1 runs `first`
    let runs = [c.force(forced(t, 1, 0)), c.force(forced(&long, 150, 0))];
    hold.wait();
    run(first);
    let [decoded, chunked] = runs.map(|h| {
        let h = h.unwrap();
        assert!(run(h.request).0.is_empty());
        h.positions.try_iter().collect::<Vec<_>>()
    });
    agree(&prefill, &decoded);
    agree(&observe(&e, forced(&long, 150, 0)).0, &chunked);
    let mixed = records(&c).into_iter().any(|r| {
        matches!(r, Record::EngineSample(_, s) if s.n_decode_seqs > 0 && s.n_prefill_tokens > 0)
    });
    assert!(mixed, "no step had decode rows and a chunk");

    // Cached: the first half is a session turn that generates one token and parks.
    let first = Request {
        session: 9,
        keep_kv: true,
        ..Request::new(t[..20].to_vec(), 1)
    };
    run(e.submit(first).unwrap());
    let (cached, end) = observe(&e, forced(t, 40, 9));
    assert_eq!(done(&end).2, 20, "positions from the parked n_kv on");
    agree(&prefill[20..], &cached);

    let hs: Vec<ForcedHandle> = refs
        .iter()
        .map(|r| e.force(forced(r, 1, 0)).unwrap())
        .collect();
    for (h, r) in hs.into_iter().zip(&refs) {
        run(h.request);
        let parallel: Vec<Position> = h.positions.try_iter().collect();
        agree(&observe(&e, forced(r, 40, 0)).0, &parallel);
    }
}

/// The reference tokens are what is fed (FakeModel predicts input + 1), and a run preempted for KV
/// recomputes its prefix without reporting a position twice.
#[test]
fn forced_runs_survive_preemption() {
    // One 128-token context. 2 ms a step, so each run lasts ~200 ms and the three, submitted
    // microseconds apart, overlap however this thread is scheduled: the preemption is certain.
    let model = Delay(FakeModel::new(64), Arc::new(AtomicU64::new(2000)));
    let load = move |_: &_, _: &_, _| Ok(Box::new(model) as Box<dyn Model>);
    let opts = fake_opts(vec![(knob_id::MAX_BATCH_SEQS, 1.0)], 128);
    let e = start_with(opts, Box::new(load)).unwrap();
    let more = KnobRequest::new(knob_id::MAX_BATCH_SEQS, 3.0, Authority::Human);
    assert!(matches!(e.knobs().set(more), KnobReply::Applied { .. }));
    let refs: Vec<Vec<u32>> = (0..3)
        .map(|j| (0..100).map(|i| (i * 5 + j * 17) % 64).collect())
        .collect();
    let hs: Vec<ForcedHandle> = refs
        .iter()
        .map(|r| e.force(forced(r, 1, 0)).unwrap())
        .collect();
    for (h, r) in hs.into_iter().zip(&refs) {
        assert_eq!(done(&run(h.request).1).0, Finish::Length);
        let got: Vec<(u32, u32)> = h
            .positions
            .try_iter()
            .map(|p| (p.pos, p.top[0].0))
            .collect();
        let want: Vec<(u32, u32)> = (0..100).map(|p| (p, (r[p as usize] + 1) % 64)).collect();
        assert_eq!(got, want);
    }
    let preempted = records(&e)
        .into_iter()
        .any(|r| matches!(r, Record::RequestSpan(_, s) if s.preemptions > 0));
    assert!(preempted, "nothing was preempted");
    // Every token is admitted like a prompt token, and a run may fill max_context exactly.
    let full: Vec<u32> = (0..128).map(|i| i % 64).collect();
    let (positions, end) = observe(&e, forced(&full, 1, 0));
    assert_eq!((positions.len(), done(&end).0), (128, Finish::Length));
    let over: Vec<u32> = (0..129).map(|i| i % 64).collect();
    assert_eq!(
        e.force(forced(&over, 1, 0)).err(),
        Some(EngineError::PromptTooLong {
            tokens: 129,
            limit: 128
        })
    );
    assert_eq!(
        e.force(forced(&[1, 64], 1, 0)).err(),
        Some(EngineError::InvalidRequest(invalid::TOKEN_OUT_OF_VOCAB))
    );
    let err = |prompt| e.force(forced(&[1, 2], prompt, 0)).err();
    assert_eq!(
        err(0),
        Some(EngineError::InvalidRequest(invalid::EMPTY_PROMPT))
    );
    assert_eq!(
        e.force(forced(&[], 1, 0)).err(),
        Some(EngineError::InvalidRequest(invalid::EMPTY_PROMPT))
    );
    assert_eq!(
        err(3),
        Some(EngineError::InvalidRequest(invalid::NO_TOKENS))
    );
}

/// Arm B is `knob` at `value`.
fn delta(knob: u16, value: f64) -> [KnobDelta; 4] {
    let mut b = [KnobDelta::default(); 4];
    b[0] = KnobDelta { knob, value };
    b
}

/// A gait start without an allowance.
fn ab(b: [KnobDelta; 4], period_steps: u32, max_steps: u64, trial_id: u64) -> AbCommand {
    AbCommand::Start {
        b,
        period_steps,
        max_steps,
        trial_id,
        authority: Authority::Gait,
        allow: Allow::NONE,
    }
}

/// 9.8, 9.12: `ab` and `revert` sent during a step longer than the 250 ms default wait for that step
/// to end instead of answering Busy or Indeterminate: the engine takes commands only between
/// steps, and a step under the watchdog's limit is slow, not stuck (#172).
#[test]
fn ab_and_revert_wait_out_a_slow_step() {
    let (e, _) = delayed(600_000, vec![]);
    let k = e.knobs();
    let long = e.submit(Request::new(vec![1], 4000)).unwrap();
    skip(&long, 1);
    // Inside the next 600 ms step, so at least 500 ms of it remain when each command arrives.
    std::thread::sleep(Duration::from_millis(100));
    let started = k.ab(ab(delta(knob_id::MAX_BATCH_SEQS, 8.0), 1000, 0, 7));
    assert!(matches!(started, AbReply::Started { .. }), "{started:?}");
    std::thread::sleep(Duration::from_millis(100));
    let undone = k.revert(RevertTarget::Trial(7), Authority::Gait);
    assert!(matches!(undone, KnobReply::Applied { .. }), "{undone:?}");
}

/// 9.9: arms alternate every `period_steps` at step boundaries and stamp every record, the A/B ends
/// on arm A at `max_steps`, and each request is attributed to the arm of its first token.
#[test]
fn ab_alternates_and_ends_on_arm_a() {
    let (e, _) = delayed(1000, vec![]);
    let k = e.knobs();
    let long = e.submit(Request::new(vec![1], 4000)).unwrap();
    skip(&long, 3);
    let AbReply::Started {
        arm_a_rev,
        step: s0,
    } = k.ab(ab(delta(knob_id::MAX_BATCH_SEQS, 8.0), 4, 24, 3))
    else {
        panic!("not started");
    };
    assert_eq!(arm_a_rev, 1);
    // A late read can be past a flip: the published arm must be the one its revision is in.
    let cfg = k.effective();
    let n = cfg.rev - arm_a_rev; // one revision per flip; the 6th after the start is the end
    assert_eq!(cfg.ab_arm, if n < 6 { [1, 2][n as usize % 2] } else { 0 });
    run(e.submit(Request::new(vec![2], 3)).unwrap());
    // Wait for the end itself: tokens from before the start may still be buffered in `long`.
    let until = Instant::now() + Duration::from_secs(20);
    while k.effective().ab_arm != 0 {
        assert!(Instant::now() < until, "the A/B never reached max_steps");
        std::thread::sleep(Duration::from_millis(5));
    }
    let stopped = AbReply::Stopped {
        steps: 24,
        end: AbEnd::MaxSteps,
    };
    assert_eq!(k.ab(AbCommand::Stop { trial_id: 3 }), stopped);
    let cfg = k.effective();
    assert_eq!(
        (cfg.ab_arm, cfg.values[knob_id::MAX_BATCH_SEQS as usize]),
        (0, 16.0)
    );
    long.cancel();
    assert_eq!(run(long).1, EngineEvent::Error(EngineError::Cancelled));
    assert_eq!(e.parked_tokens(0), 0); // after the step that wrote its span
    let recs = records(&e);
    let arm = |step: u64| match step.checked_sub(s0) {
        Some(n) if n < 24 => [1, 2][(n / 4 % 2) as usize],
        _ => 0,
    };
    let mut flips = Vec::new();
    for r in &recs {
        match r {
            Record::EngineSample(h, s) => assert_eq!(h.ab_arm, arm(s.step), "step {}", s.step),
            Record::ConfigurationRevision(_, c) if c.cause() == Cause::AbFlip => {
                flips.push((c.step_applied - s0, c.effective, c.trial_id));
            }
            _ => {}
        }
    }
    // The start, five flips and the end.
    let want: Vec<(u64, f64, u64)> = (0..=6)
        .map(|i| (i * 4, if i % 2 == 1 { 8.0 } else { 16.0 }, 3))
        .collect();
    assert_eq!(flips, want);
    let mut spans = 0;
    for r in &recs {
        if let Record::RequestSpan(h, s) = r {
            let first_step = recs.iter().find_map(|r| match r {
                Record::EngineSample(sh, _) if sh.t_mono_ns >= s.t_first_ns => Some(sh.ab_arm),
                _ => None,
            });
            assert_eq!(Some(h.ab_arm), first_step, "span {}", s.req_id);
            spans += 1;
        }
    }
    assert_eq!(spans, 2);
}

/// 9.9: a request is attributed to the arm of its first token, not of its last.
#[test]
fn a_span_carries_the_arm_of_its_first_token() {
    let e = fake(vec![], 4096);
    let k = e.knobs();
    let started = k.ab(ab(delta(knob_id::MAX_BATCH_SEQS, 8.0), 1, 0, 5));
    assert!(matches!(started, AbReply::Started { step: 1, .. }));
    run(e.submit(Request::new(vec![1], 2)).unwrap()); // token 1 on arm A, token 2 on arm B
    assert_eq!(e.parked_tokens(0), 0); // after the step that wrote its span
    let arms: Vec<u8> = records(&e)
        .into_iter()
        .filter_map(|r| match r {
            Record::RequestSpan(h, _) => Some(h.ab_arm),
            _ => None,
        })
        .collect();
    assert_eq!(arms, vec![1]);
}

/// A human set on an A/B knob ends the A/B on arm A first, so its `previous` is the value outside the
/// trial; the trial's revert then leaves a superseding human change alone.
#[test]
fn a_human_change_supersedes_a_trial() {
    let e = fake(vec![], 4096);
    let k = e.knobs();
    let started = k.ab(ab(delta(knob_id::MAX_BATCH_SEQS, 8.0), 1, 0, 5));
    assert!(matches!(started, AbReply::Started { .. }));
    run(e.submit(Request::new(vec![1], 2)).unwrap()); // two steps: arm A, then arm B
    assert_eq!(k.effective().values[knob_id::MAX_BATCH_SEQS as usize], 8.0);
    let human = k.set(KnobRequest::new(
        knob_id::MAX_BATCH_SEQS,
        4.0,
        Authority::Human,
    ));
    assert!(matches!(
        human,
        KnobReply::Applied {
            previous: 16.0,
            effective: 4.0,
            ..
        }
    ));
    assert!(matches!(
        k.ab(AbCommand::Stop { trial_id: 5 }),
        AbReply::Stopped {
            end: AbEnd::Stop,
            ..
        }
    ));
    let nothing = KnobReply::Rejected {
        reason: RejectReason::NothingToRevert,
    };
    assert_eq!(k.revert(RevertTarget::Trial(5), Authority::Gait), nothing);
    assert_eq!(k.effective().values[knob_id::MAX_BATCH_SEQS as usize], 4.0);

    let gait = KnobRequest {
        trial_id: 6,
        ..KnobRequest::new(knob_id::PREFILL_CHUNK_TOKENS, 256.0, Authority::Gait)
    };
    assert!(matches!(k.set(gait), KnobReply::Applied { .. }));
    let human = KnobRequest::new(knob_id::PREFILL_CHUNK_TOKENS, 128.0, Authority::Human);
    assert!(matches!(k.set(human), KnobReply::Applied { .. }));
    assert_eq!(k.revert(RevertTarget::Trial(6), Authority::Gait), nothing);
    assert_eq!(
        k.effective().values[knob_id::PREFILL_CHUNK_TOKENS as usize],
        128.0
    );
}

/// The trial's revert ends its A/B with an inverse revision per knob even on arm A, and undoes the
/// trial's other changes; a gait start without an allowance is validated like its `set`.
#[test]
fn trial_revert_ends_the_ab() {
    let e = fake(vec![], 4096);
    let k = e.knobs();
    let start = |b, trial| k.ab(ab(b, 1000, 0, trial));
    let rejected = |reason| AbReply::Rejected { reason };
    let mut twice = delta(knob_id::MAX_BATCH_SEQS, 8.0);
    twice[1] = twice[0];
    assert_eq!(
        start(delta(knob_id::MAX_BATCH_SEQS, 8.0), 0),
        rejected(RejectReason::WrongType)
    );
    assert_eq!(
        start(twice, 1),
        rejected(RejectReason::Conflict {
            with: knob_id::MAX_BATCH_SEQS
        })
    );
    assert_eq!(
        start(delta(knob_id::MAX_BATCH_SEQS, 99.0), 1),
        rejected(RejectReason::OutOfRange {
            min: 1.0,
            max: 64.0
        })
    );
    assert_eq!(
        start(delta(knob_id::DECODE_GRAPHS, 1.0), 1),
        rejected(RejectReason::Unsupported)
    );
    assert_eq!(
        start(delta(knob_id::MAX_CONTEXT, 1024.0), 1),
        rejected(RejectReason::NotLive {
            class: ActivationClass::ModelReload
        })
    );
    // D7, gait without an allowance: never a quality or resource knob.
    assert_eq!(
        start(delta(knob_id::TEMPERATURE_DEFAULT, 1.0), 1),
        rejected(RejectReason::Unauthorized {
            risk: RiskClass::QualityAffecting
        })
    );
    assert_eq!(
        start(delta(knob_id::MAX_WAITING, 0.0), 1),
        rejected(RejectReason::Unauthorized {
            risk: RiskClass::ResourceOrAvailability
        })
    );
    // An A/B that would end before arm B runs; `period_steps` 0 is 200.
    let short = rejected(RejectReason::OutOfRange {
        min: 201.0,
        max: u64::MAX as f64,
    });
    let b = delta(knob_id::MAX_BATCH_SEQS, 8.0);
    assert_eq!(k.ab(ab(b, 200, 50, 1)), short);
    assert_eq!(k.ab(ab(b, 0, 200, 1)), short);
    // Arm B may not loosen its own guard, and an A/B changes at least one knob.
    assert_eq!(
        start(delta(knob_id::GUARDRAIL_RATIO, 3.0), 1),
        rejected(RejectReason::Conflict {
            with: knob_id::GUARDRAIL_RATIO
        })
    );
    assert_eq!(
        start([KnobDelta::default(); 4], 1),
        rejected(RejectReason::WrongType)
    );

    let policy = KnobRequest::new(knob_id::SCHEDULE_POLICY, 1.0, Authority::Human);
    let KnobReply::Applied { rev: human, .. } = k.set(policy) else {
        panic!("not applied");
    };
    let earlier = KnobRequest {
        trial_id: 6,
        ..KnobRequest::new(knob_id::MAX_BATCH_SEQS, 32.0, Authority::Gait)
    };
    assert!(matches!(k.set(earlier), KnobReply::Applied { .. }));
    let mut b = delta(knob_id::MAX_BATCH_SEQS, 8.0);
    b[1] = KnobDelta {
        knob: knob_id::SCHEDULE_POLICY,
        value: 2.0,
    };
    // Two revisions before it, one per knob at its start.
    assert!(matches!(start(b, 7), AbReply::Started { arm_a_rev: 4, .. }));
    assert_eq!(start(b, 8), rejected(RejectReason::Busy));
    // Neither one revision nor another trial may change a knob the A/B flips.
    let busy = KnobReply::Rejected {
        reason: RejectReason::Busy,
    };
    assert_eq!(k.revert(RevertTarget::Rev(human), Authority::Human), busy);
    assert_eq!(k.revert(RevertTarget::Trial(6), Authority::Gait), busy);
    let gait = KnobRequest {
        trial_id: 7,
        ..KnobRequest::new(knob_id::PREFILL_CHUNK_TOKENS, 256.0, Authority::Gait)
    };
    assert!(matches!(k.set(gait), KnobReply::Applied { .. }));
    assert!(matches!(
        k.revert(RevertTarget::Trial(7), Authority::Gait),
        KnobReply::Applied { .. }
    ));
    let stopped = AbReply::Stopped {
        steps: 0,
        end: AbEnd::Revert,
    };
    assert_eq!(k.ab(AbCommand::Stop { trial_id: 7 }), stopped);
    let v = k.effective().values;
    assert_eq!((v[1], v[2], v[3]), (32.0, 512.0, 1.0));
    assert_eq!(
        k.revert(RevertTarget::Trial(7), Authority::Gait),
        KnobReply::Rejected {
            reason: RejectReason::NothingToRevert
        }
    );
    // With the A/B over, the earlier trial's revert goes through.
    assert!(matches!(
        k.revert(RevertTarget::Trial(6), Authority::Gait),
        KnobReply::Applied {
            effective: 16.0,
            ..
        }
    ));
    assert_eq!(e.parked_tokens(0), 0); // every message above is handled
    let inverse: Vec<(u16, f64, u64)> = records(&e)
        .into_iter()
        .filter_map(|r| match r {
            Record::ConfigurationRevision(_, c) if c.cause() == Cause::Inverse => {
                Some((c.knob, c.effective, c.trial_id))
            }
            _ => None,
        })
        .collect();
    let want = vec![(1, 32.0, 7), (3, 1.0, 7), (2, 512.0, 7), (1, 16.0, 6)];
    assert_eq!(inverse, want);
}

/// A start is validated under its own authority and allowance (D7, #54): a resource knob needs
/// `Allow::RESOURCE` from gait and nothing from a human; a quality knob never runs live (decision
/// 15); an arm with `max_waiting` 0 would refuse every submit, so it is out of range.
#[test]
fn ab_starts_carry_their_authority() {
    let e = fake(vec![], 4096);
    let k = e.knobs();
    let start = |knob, value, authority, allow, trial| {
        k.ab(AbCommand::Start {
            b: delta(knob, value),
            period_steps: 1000,
            max_steps: 0,
            trial_id: trial,
            authority,
            allow,
        })
    };
    let rejected = |reason| AbReply::Rejected { reason };
    let watermark = knob_id::KV_PREEMPT_WATERMARK;
    assert_eq!(
        start(watermark, 0.85, Authority::Gait, Allow::NONE, 1),
        rejected(RejectReason::Unauthorized {
            risk: RiskClass::ResourceOrAvailability
        })
    );
    assert_eq!(
        start(
            knob_id::MAX_WAITING,
            0.0,
            Authority::Gait,
            Allow::RESOURCE,
            1
        ),
        rejected(RejectReason::OutOfRange {
            min: 1.0,
            max: 4096.0
        })
    );
    assert_eq!(
        start(
            knob_id::TOP_K_DEFAULT,
            8.0,
            Authority::Human,
            Allow::NONE,
            1
        ),
        rejected(RejectReason::Unauthorized {
            risk: RiskClass::QualityAffecting
        })
    );
    for (authority, allow, trial) in [
        (Authority::Gait, Allow::RESOURCE, 2),
        (Authority::Human, Allow::NONE, 3),
    ] {
        let started = start(watermark, 0.85, authority, allow, trial);
        assert!(matches!(started, AbReply::Started { .. }), "{started:?}");
        let stopped = k.ab(AbCommand::Stop { trial_id: trial });
        assert!(matches!(stopped, AbReply::Stopped { .. }), "{stopped:?}");
    }
}

/// An A/B ends at `max_steps` in the step that reaches it, so an engine that then idles is on arm A;
/// its start, flips and end are all in the ring with its trial.
#[test]
fn max_steps_ends_before_the_engine_idles() {
    let e = fake(vec![], 4096);
    let k = e.knobs();
    let started = k.ab(ab(delta(knob_id::MAX_BATCH_SEQS, 8.0), 1, 3, 12));
    assert!(matches!(started, AbReply::Started { arm_a_rev: 1, .. }));
    run(e.submit(Request::new(vec![1], 3)).unwrap()); // arms A, B, A; then idle
    let stopped = AbReply::Stopped {
        steps: 3,
        end: AbEnd::MaxSteps,
    };
    assert_eq!(k.ab(AbCommand::Stop { trial_id: 12 }), stopped);
    let cfg = k.effective();
    assert_eq!(
        (cfg.ab_arm, cfg.values[knob_id::MAX_BATCH_SEQS as usize]),
        (0, 16.0)
    );
    let edges: Vec<(u8, f64)> = records(&e)
        .into_iter()
        .filter_map(|r| match r {
            Record::ConfigurationRevision(h, c) if c.trial_id == 12 => {
                Some((h.ab_arm, c.effective))
            }
            _ => None,
        })
        .collect();
    assert_eq!(edges, [(1, 16.0), (2, 8.0), (1, 16.0), (0, 16.0)]);
}

/// Waits up to `within` for the effective value of `knob` to become `value`.
fn wait_for(e: &EngineHandle, knob: u16, value: f64, within: Duration) {
    let until = Instant::now() + within;
    while e.knobs().effective().values[knob as usize] != value {
        assert!(Instant::now() < until, "knob {knob} never became {value}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The first cause-Guardrail revision on `knob`: its step and trial, and the trial of the
/// ActivationEvent that follows it.
fn guardrail_records(e: &EngineHandle, knob: u16) -> (u64, u64, u64) {
    let recs = records(e);
    let i = recs
        .iter()
        .position(|r| {
            matches!(r, Record::ConfigurationRevision(_, c)
                if c.cause() == Cause::Guardrail && c.knob == knob)
        })
        .expect("a guardrail revision");
    let Record::ConfigurationRevision(_, c) = &recs[i] else {
        unreachable!()
    };
    let named = recs[i..]
        .iter()
        .find_map(|r| match r {
            Record::ActivationEvent(_, a) if a.knob == knob => Some(a.trial_id),
            _ => None,
        })
        .expect("its ActivationEvent");
    (c.step_applied, c.trial_id, named)
}

/// The steps the guardrail counts, as (step, arm, ns per token): the decode-dominant ones, from
/// their EngineSample, which carries the `dur_ns` the guardrail read (9.9). These tests run one
/// sequence, so every such step is in the same bucket.
fn guarded_steps(e: &EngineHandle) -> Vec<(u64, u8, u64)> {
    records(e)
        .into_iter()
        .filter_map(|r| match r {
            Record::EngineSample(h, s)
                if s.n_decode_seqs >= 1 && s.n_prefill_tokens <= s.n_decode_seqs =>
            {
                Some((s.step, h.ab_arm, s.dur_ns / u64::from(s.n_decode_seqs)))
            }
            _ => None,
        })
        .collect()
}

/// Nearest-rank percentile `p` of a non-empty sample, as the guardrail takes it (9.9).
fn nearest_rank(mut v: Vec<u64>, p: usize) -> u64 {
    v.sort_unstable();
    v[(v.len() * p).div_ceil(100) - 1]
}

/// The baseline a watch starting at `step` snapshots: the nearest-rank p90 of the last 200 counted
/// steps before it, arm B's left out.
fn baseline_ns(e: &EngineHandle, step: u64) -> u64 {
    let mut before: Vec<u64> = guarded_steps(e)
        .into_iter()
        .filter(|g| g.0 < step && g.1 != 2)
        .map(|g| g.2)
        .collect();
    nearest_rank(before.split_off(before.len().saturating_sub(200)), 90)
}

/// A step delay (µs) that trips a watch starting at `step` in its first full window, however noisy
/// the machine was during the baseline: over 3 × the baseline (`guardrail_ratio` is at most 3.0),
/// and at least `floor_us`.
fn tripping_delay(e: &EngineHandle, step: u64, floor_us: u64) -> u64 {
    (3 * baseline_ns(e, step) / 1000).max(floor_us)
}

/// The step that trips a watch starting at `step`, replayed from the ring: over the watch's steps
/// (arm B's for an A/B, the others for a change), the first whose window of the last 32 has a
/// nearest-rank p90 over `guardrail_ratio` × the baseline (arm A's latest 200 steps for an A/B, #213).
fn trip_step(e: &EngineHandle, step: u64, ab: bool) -> u64 {
    let ratio = e.knobs().effective().values[knob_id::GUARDRAIL_RATIO as usize];
    let mut window = VecDeque::new();
    for (s, arm, ns) in guarded_steps(e) {
        if s < step || (arm == 2) != ab {
            continue;
        }
        if window.len() == 32 {
            window.pop_front();
        }
        window.push_back(ns);
        // An A/B judges arm B against arm A's latest steps; a change against its start snapshot.
        let limit = ratio * baseline_ns(e, if ab { s } else { step }) as f64;
        if window.len() == 32 && nearest_rank(window.iter().copied().collect(), 90) as f64 > limit {
            return s;
        }
    }
    panic!("no step trips the watch that started at step {step}");
}

/// 9.9: a gait change followed by slower decode steps is reverted by the guardrail at the next step
/// boundary, with an ActivationEvent naming the trial.
#[test]
fn guardrail_reverts_a_slow_change() {
    // 2 ms a step: the request runs for seconds after the baseline, so it is still running when the
    // change lands however late this thread is scheduled.
    let (e, delay) = delayed(2000, vec![]);
    let long = e.submit(Request::new(vec![1], 4000)).unwrap();
    skip(&long, 250); // the baseline's 200 decode steps
    let change = KnobRequest {
        trial_id: 9,
        timeout_ms: 10_000,
        ..KnobRequest::new(knob_id::SCHEDULE_POLICY, 1.0, Authority::Gait)
    };
    let KnobReply::Applied { step, .. } = e.knobs().set(change) else {
        panic!("not applied");
    };
    assert!(step < 4000, "the request ended before the change");
    // Slow at once: the watch's 400 steps then last 20 s, not 0.8 s. Then slow against the
    // baseline the guardrail took, which a loaded machine can inflate.
    delay.store(50_000, Ordering::Relaxed);
    let slow = tripping_delay(&e, step, 50_000);
    delay.store(slow, Ordering::Relaxed);
    let within = Duration::from_micros(64 * slow) + Duration::from_secs(10);
    wait_for(&e, knob_id::SCHEDULE_POLICY, 0.0, within);
    delay.store(0, Ordering::Relaxed);
    drop(long);
    assert_eq!(e.parked_tokens(0), 0);
    let (applied, trial, named) = guardrail_records(&e, knob_id::SCHEDULE_POLICY);
    assert_eq!(
        (applied, trial, named),
        (trip_step(&e, step, false) + 1, 9, 9)
    );
}

/// 9.9: slow arm-B steps end the A/B on arm A with cause Guardrail.
#[test]
fn guardrail_ends_a_slow_ab() {
    // 2 ms a step, as in guardrail_reverts_a_slow_change: the request outlives a late A/B start.
    // Only arm B slows: a slowdown on both arms is traffic drift, not the candidate (#213).
    let (delay, knobs) = (Arc::new(AtomicU64::new(0)), Arc::new(OnceLock::new()));
    let model = ArmBDelay(FakeModel::new(64), 2000, delay.clone(), knobs.clone());
    let load = move |_: &_, _: &_, _| Ok(Box::new(model) as Box<dyn Model>);
    let e = start_with(fake_opts(vec![], 4096), Box::new(load)).unwrap();
    let k = e.knobs();
    let _ = knobs.set(k.clone());
    let long = e.submit(Request::new(vec![1], 4000)).unwrap();
    skip(&long, 250);
    let AbReply::Started { step, .. } = k.ab(ab(delta(knob_id::SCHEDULE_POLICY, 1.0), 1, 0, 11))
    else {
        panic!("not started");
    };
    assert!(step < 4000, "the request ended before the A/B started");
    let slow = tripping_delay(&e, step, 30_000);
    delay.store(slow, Ordering::Relaxed);
    let until = Instant::now() + Duration::from_micros(64 * slow) + Duration::from_secs(10);
    while k.effective().ab_arm != 0 {
        assert!(Instant::now() < until, "the guardrail never tripped");
        std::thread::sleep(Duration::from_millis(5));
    }
    delay.store(0, Ordering::Relaxed);
    assert!(matches!(
        k.ab(AbCommand::Stop { trial_id: 11 }),
        AbReply::Stopped {
            end: AbEnd::Guardrail,
            ..
        }
    ));
    assert_eq!(k.effective().values[knob_id::SCHEDULE_POLICY as usize], 0.0);
    drop(long);
    assert_eq!(e.parked_tokens(0), 0);
    let (applied, trial, named) = guardrail_records(&e, knob_id::SCHEDULE_POLICY);
    assert_eq!(
        (applied, trial, named),
        (trip_step(&e, step, true) + 1, 11, 11)
    );
}

/// 9.9, 12.4: a change still in its watch trips on the arm-A steps of a later A/B on its knob. The
/// guardrail reverts the change under its own trial. Someone else's change ends the A/B with Stop,
/// as a superseding `set` would, so nothing names the A/B's trial as the regression; the trial's own
/// earlier change ends it with Guardrail, naming it.
#[test]
fn a_tripped_change_ends_an_ab_on_its_knob() {
    let knob = knob_id::SCHEDULE_POLICY;
    for (who, trial, end) in [
        (Authority::Human, 0, AbEnd::Stop),
        (Authority::Gait, 10, AbEnd::Guardrail),
    ] {
        // 2 ms a step: the request outlives the change and the start however late this thread runs.
        let (e, delay) = delayed(2000, vec![]);
        let k = e.knobs();
        let long = e.submit(Request::new(vec![1], 4000)).unwrap();
        skip(&long, 250); // the baseline's 200 decode steps
        let change = KnobRequest {
            trial_id: trial,
            timeout_ms: 10_000,
            ..KnobRequest::new(knob, 1.0, who)
        };
        let KnobReply::Applied { step: watched, .. } = k.set(change) else {
            panic!("not applied");
        };
        // A period past the request's end: arm B never runs, the trip is on arm A.
        let AbReply::Started { step, .. } = k.ab(ab(delta(knob, 2.0), 4000, 0, 10)) else {
            panic!("not started");
        };
        assert!(step < 4000, "the request ended before the A/B started");
        // Arm A runs the change's value, slow at once: the watch's 400 steps then last seconds.
        delay.store(30_000, Ordering::Relaxed);
        let slow = tripping_delay(&e, watched, 30_000);
        delay.store(slow, Ordering::Relaxed);
        wait_for(
            &e,
            knob,
            0.0,
            Duration::from_micros(64 * slow) + Duration::from_secs(10),
        );
        delay.store(0, Ordering::Relaxed);
        assert!(matches!(
            k.ab(AbCommand::Stop { trial_id: 10 }),
            AbReply::Stopped { end: got, .. } if got == end
        ));
        drop(long);
        assert_eq!(e.parked_tokens(0), 0);
        let (applied, named_trial, named) = guardrail_records(&e, knob);
        let tripped = trip_step(&e, watched, false);
        assert_eq!((applied, named_trial, named), (tripped + 1, trial, trial));
        let blamed = records(&e).into_iter().any(|r| match r {
            Record::ConfigurationRevision(_, c) => {
                c.cause() == Cause::Guardrail && c.trial_id == 10
            }
            Record::ActivationEvent(_, a) => a.trial_id == 10,
            _ => false,
        });
        assert_eq!(
            blamed,
            trial == 10,
            "the A/B's trial is named only for its own change"
        );
    }
}

/// 9.8: a change the engine does not reach within its timeout is Indeterminate; it stays queued and
/// its ActivationEvent records it when it lands.
#[test]
fn a_late_change_is_indeterminate() {
    let (e, delay) = delayed(300_000, vec![]);
    let long = e.submit(Request::new(vec![1], 4000)).unwrap();
    skip(&long, 1);
    std::thread::sleep(Duration::from_millis(50)); // inside the next step's 300 ms
    let req = KnobRequest {
        timeout_ms: 1,
        ..KnobRequest::new(knob_id::MAX_BATCH_SEQS, 8.0, Authority::Human)
    };
    assert_eq!(
        e.knobs().set(req),
        KnobReply::Indeterminate {
            reason: IndeterminateReason::TimedOutBeforeStep,
            rev: None
        }
    );
    wait_for(&e, knob_id::MAX_BATCH_SEQS, 8.0, Duration::from_secs(20));
    delay.store(0, Ordering::Relaxed);
    drop(long);
    assert_eq!(e.parked_tokens(0), 0);
    let landed = records(&e).into_iter().any(|r| {
        matches!(r, Record::ActivationEvent(_, a)
            if a.knob == knob_id::MAX_BATCH_SEQS && a.result() == ActivationResult::Applied)
    });
    assert!(landed);
}

/// A trial that set a knob and then A/Bs it is reverted to the value before the trial, from arm B.
#[test]
fn a_trial_revert_restores_the_value_before_the_trial() {
    let e = fake(vec![], 4096);
    let k = e.knobs();
    let set = KnobRequest {
        trial_id: 5,
        ..KnobRequest::new(knob_id::MAX_BATCH_SEQS, 8.0, Authority::Gait)
    };
    assert!(matches!(k.set(set), KnobReply::Applied { .. }));
    let started = k.ab(ab(delta(knob_id::MAX_BATCH_SEQS, 4.0), 1, 0, 5));
    assert!(matches!(started, AbReply::Started { .. }));
    run(e.submit(Request::new(vec![1], 2)).unwrap()); // arm A, then arm B
    assert_eq!(k.effective().values[knob_id::MAX_BATCH_SEQS as usize], 4.0);
    assert!(matches!(
        k.revert(RevertTarget::Trial(5), Authority::Gait),
        KnobReply::Applied {
            effective: 16.0,
            ..
        }
    ));
    let cfg = k.effective();
    assert_eq!(
        (cfg.ab_arm, cfg.values[knob_id::MAX_BATCH_SEQS as usize]),
        (0, 16.0)
    );
    assert_eq!(
        k.revert(RevertTarget::Trial(5), Authority::Gait),
        KnobReply::Rejected {
            reason: RejectReason::NothingToRevert
        }
    );
}

/// 9.9/9.11: a fault ends the A/B on arm A (the library keeps its process); the failed request is
/// attributed to the arm of its first token.
#[test]
fn a_fault_ends_the_ab() {
    let load = |_: &_, _: &_, _| {
        let m = Faulty {
            inner: FakeModel::new(64),
            n: 3,
            panic: true,
        };
        Ok(Box::new(m) as Box<dyn Model>)
    };
    let e = start_with(fake_opts(vec![], 4096), Box::new(load)).unwrap();
    let k = e.knobs();
    let started = k.ab(ab(delta(knob_id::MAX_BATCH_SEQS, 8.0), 2, 0, 4));
    assert!(matches!(started, AbReply::Started { step: 1, .. }));
    // Steps 1 and 2 run on arm A; the fault is in step 3, on arm B.
    let (_, end) = run(e.submit(Request::new(vec![1], 100)).unwrap());
    assert_eq!(end, EngineEvent::Error(EngineError::Internal(0)));
    let cfg = k.effective();
    assert_eq!(
        (
            e.health().state,
            cfg.ab_arm,
            cfg.values[knob_id::MAX_BATCH_SEQS as usize]
        ),
        (HealthState::Failed, 0, 16.0)
    );
    assert_eq!(
        k.ab(AbCommand::Stop { trial_id: 4 }),
        AbReply::Rejected {
            reason: RejectReason::NotReady
        }
    );
    // The engine writes a span after its request hears of the end, and a failed engine answers no
    // round trip to wait on: wait for the span itself.
    let until = Instant::now() + Duration::from_secs(20);
    let arms = loop {
        let arms: Vec<u8> = records(&e)
            .into_iter()
            .filter_map(|r| match r {
                Record::RequestSpan(h, s) if s.outcome() == Outcome::Error => Some(h.ab_arm),
                _ => None,
            })
            .collect();
        if !arms.is_empty() {
            break arms;
        }
        assert!(Instant::now() < until, "the failed request has no span");
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(arms, vec![1]);
    let restored = records(&e).into_iter().any(|r| {
        matches!(r, Record::ConfigurationRevision(h, c)
            if c.cause() == Cause::AbFlip && c.effective == 16.0 && h.ab_arm == 0)
    });
    assert!(restored, "the end on arm A is in the ring");
}

/// 9.8, 9.11: changes still queued when the engine fails are Rejected NotReady, each with the
/// ActivationEvent its Indeterminate caller waits for, and every later command is NotReady, not the
/// Busy of a queue the fault left full.
#[test]
fn a_fault_answers_the_queued_changes() {
    let hold = Arc::new(Barrier::new(2));
    let h = hold.clone();
    let load = move |_: &_, _: &_, _| {
        let faulty = Faulty {
            inner: FakeModel::new(64),
            n: 2,
            panic: false,
        };
        let m = Held {
            inner: Box::new(faulty),
            at: 2,
            hold: h,
        };
        Ok(Box::new(m) as Box<dyn Model>)
    };
    let e = start_with(fake_opts(vec![], 4096), Box::new(load)).unwrap();
    let k = e.knobs();
    let running = e.submit(Request::new(vec![1], 100)).unwrap();
    hold.wait(); // the engine is inside step 2, which fails
    let late = KnobRequest {
        timeout_ms: 1,
        ..KnobRequest::new(knob_id::MAX_BATCH_SEQS, 8.0, Authority::Human)
    };
    for _ in 0..64 {
        assert!(matches!(k.set(late), KnobReply::Indeterminate { .. }));
    }
    let busy = KnobReply::Rejected {
        reason: RejectReason::Busy,
    };
    assert_eq!(k.set(late), busy, "the pending bound");
    hold.wait();
    // The drain is done before the running request hears of the fault.
    assert_eq!(run(running).1, EngineEvent::Error(EngineError::DeviceLost));
    let not_ready = RejectReason::NotReady;
    assert_eq!(
        k.revert(RevertTarget::Trial(1), Authority::Gait),
        KnobReply::Rejected { reason: not_ready }
    );
    assert_eq!(
        k.ab(AbCommand::Stop { trial_id: 1 }),
        AbReply::Rejected { reason: not_ready }
    );
    let refused = records(&e)
        .into_iter()
        .filter(|r| {
            matches!(r, Record::ActivationEvent(_, a)
                if a.knob == knob_id::MAX_BATCH_SEQS
                    && a.result == ActivationResult::Rejected.raw()
                    && a.reason == not_ready.code())
        })
        .count();
    assert_eq!(refused, 64);
}

/// The counters a consumer differences: rejections are spans and counted, tokens are counted as they
/// are sampled and prompts as they start, `tot_finished` counts Done only; the estimates say how
/// they arrive (bandwidth from the GGUF's weights, Unsupported on the CPU reference); the capability
/// report admits there is no graph replay.
#[test]
fn telemetry_counts_and_states() {
    let (e, _) = delayed(1000, vec![]); // slow enough that the cancel lands mid-request
    let caps = e.capabilities();
    assert_eq!(
        (caps.backend, caps.graph_replay, caps.prefill_chunk_max),
        (Backend::Cpu, false, 1024)
    );
    assert_eq!(
        [
            caps.achieved_gbps,
            caps.hbm_util,
            caps.sm_occupancy,
            caps.device_sample
        ],
        [
            MState::Missing,
            MState::Unsupported,
            MState::Unsupported,
            MState::Unsupported
        ]
    );
    let graphs = KnobRequest::new(knob_id::DECODE_GRAPHS, 1.0, Authority::Human);
    assert_eq!(
        e.knobs().set(graphs),
        KnobReply::Rejected {
            reason: RejectReason::Unsupported
        }
    );
    assert!(e.submit(Request::new(vec![], 1)).is_err());
    assert!(e.submit(Request::new(vec![1; 4096], 10)).is_err());
    run(e.submit(Request::new(vec![1, 2, 3], 4)).unwrap());
    let h = e.submit(Request::new(vec![5], 3000)).unwrap();
    skip(&h, 2);
    drop(h);
    run(e.submit(Request::new(vec![1], 1)).unwrap());
    assert_eq!(e.parked_tokens(0), 0);
    let recs = records(&e);
    let rejected: Vec<u16> = recs
        .iter()
        .filter_map(|r| match r {
            Record::RequestSpan(_, s) if s.outcome() == Outcome::Rejected => Some(s.error),
            _ => None,
        })
        .collect();
    let codes = [
        EngineError::InvalidRequest(invalid::EMPTY_PROMPT).code(),
        EngineError::PromptTooLong {
            tokens: 0,
            limit: 0,
        }
        .code(),
    ];
    assert_eq!(rejected, codes);
    let (mut gen_tokens, mut prompt_tokens) = (0, 0);
    for r in &recs {
        if let Record::RequestSpan(_, s) = r
            && s.outcome() != Outcome::Rejected
        {
            gen_tokens += u64::from(s.gen_tokens);
            prompt_tokens += u64::from(s.prompt_tokens);
        }
    }
    let last = recs
        .iter()
        .rev()
        .find_map(|r| match r {
            Record::EngineSample(_, s) => Some(*s),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        (last.tot_rejected, last.tot_finished, last.tot_queued),
        (2, 2, 3)
    );
    assert_eq!(
        (last.tot_gen_tokens, last.tot_prompt_tokens),
        (gen_tokens, prompt_tokens)
    );
    assert_eq!(last.achieved_gbps.state(), MState::Missing);
    assert_eq!(last.hbm_util.state(), MState::Unsupported);
    assert_eq!(last.graph_replayed, 0);

    // From a GGUF: the weights give the deployment's identity and a bandwidth estimate.
    let e = tiny_engine();
    assert_eq!(e.capabilities().achieved_gbps, MState::Estimated);
    run(e.submit(Request::new(vec![1, 2, 3], 2)).unwrap());
    assert_eq!(e.parked_tokens(0), 0);
    let d = e.deployment();
    assert_eq!(d.quant.as_str(), "Q5_K", "GGML's name, not DType's");
    assert!(d.params_b > 0.0 && d.weights_bytes > 0);
    assert_eq!(d.peak_gbps.state(), MState::Unsupported);
    let recs = records(&e);
    let gbps = recs.iter().find_map(|r| match r {
        Record::EngineSample(_, s) => Some(s.achieved_gbps),
        _ => None,
    });
    assert!(gbps.is_some_and(|g| g.state() == MState::Estimated && g.v > 0.0));
    // Past 16 rows the model dequantizes and runs gemms, which the weight bytes undercount.
    run(e.submit(Request::new((0..40).collect(), 1)).unwrap());
    assert_eq!(e.parked_tokens(0), 0);
    let wide = records(&e).into_iter().find_map(|r| match r {
        Record::EngineSample(_, s) if s.n_prefill_tokens == 40 => Some(s.achieved_gbps.state()),
        _ => None,
    });
    assert_eq!(wide, Some(MState::Missing));
    let device = recs.iter().find_map(|r| match r {
        Record::DeviceSample(_, d) => Some(d.temp_c.state()),
        _ => None,
    });
    assert_eq!(device, Some(MState::Unsupported));

    // Mid-request the counters already hold what has happened, not what an end would add: tokens as
    // sampled, the one start, nothing finished; the request queued behind the one seat has not started.
    let (e, _) = delayed(1000, vec![(knob_id::MAX_BATCH_SEQS, 1.0)]);
    let running = e.submit(Request::new(vec![1], 4000)).unwrap();
    skip(&running, 3);
    let queued = e.submit(Request::new(vec![2], 1)).unwrap();
    let until = Instant::now() + Duration::from_secs(20);
    let sample = loop {
        let waiting = records(&e).into_iter().find_map(|r| match r {
            Record::EngineSample(_, s) if s.n_waiting == 1 => Some(s),
            _ => None,
        });
        if let Some(s) = waiting {
            break s;
        }
        assert!(Instant::now() < until, "the second request never waited");
        std::thread::sleep(Duration::from_millis(5));
    };
    assert!(
        sample.tot_gen_tokens >= 3,
        "tokens so far: {}",
        sample.tot_gen_tokens
    );
    assert_eq!(
        (
            sample.tot_queued,
            sample.tot_prompt_tokens,
            sample.tot_finished
        ),
        (1, 1, 0)
    );
    drop((running, queued));
}

/// #61: with ignore_eos every stop token has a -inf logit before sampling, as under llama-server's
/// ignore_eos, so it is never sampled and the request still runs to max_tokens.
#[test]
fn ignore_eos_never_samples_an_eog_token() {
    let mut o = EngineOptions::new("fake".into(), CPU);
    o.max_context = 512;
    o.ring_capacity = 256;
    let e = start_with(o, Box::new(|_, _, _| Ok(Box::new(FakeModel::new(8))))).unwrap();
    // FakeModel's argmax after token t is t + 1, so after 4 it is 5, the EOS here.
    let h = e
        .submit(Request {
            stop_tokens: vec![5],
            ignore_eos: true,
            ..Request::new(vec![4], 3)
        })
        .unwrap();
    let sampled: Vec<u32> = h
        .events
        .iter()
        .map_while(|ev| match ev {
            EngineEvent::Token { id, .. } => Some(Some(id)),
            EngineEvent::Prefilled { .. } => Some(None),
            _ => None,
        })
        .flatten()
        .collect();
    assert_eq!(sampled.len(), 3);
    assert!(!sampled.contains(&5), "sampled {sampled:?}");
}

/// #64: parked sessions resumed with one submit_group decode as one batch from the first step: the
/// group is admitted in one turn and each one-token remainder is a decode row, not a prefill.
#[test]
fn parked_group_resumes_as_one_decode_batch() {
    let e = fake(vec![], 512);
    let (pl, pp, tg) = (4u64, 16u32, 4u32);
    let prompts: Vec<Vec<u32>> = (0..pl)
        .map(|i| (0..pp).map(|t| (t + i as u32) % 50).collect())
        .collect();
    let prefill = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| kept(i as u64 + 1, p.clone(), 1));
    let first: Vec<u32> = e
        .submit_group(prefill.collect())
        .unwrap()
        .into_iter()
        .map(|h| ids(&run(h).0)[0])
        .collect();
    for i in 1..=pl {
        assert_eq!(e.parked_tokens(i), pp); // also a barrier: the prefill samples are written
    }
    let mut ring = e.ring().reader(ReadFrom::Latest);
    let group = prompts.iter().zip(&first).enumerate().map(|(i, (p, &f))| {
        let mut p = p.clone();
        p.push(f);
        kept(i as u64 + 1, p, tg)
    });
    for h in e.submit_group(group.collect()).unwrap() {
        assert_eq!(done(&run(h).1), (Finish::Length, tg, pp));
    }
    e.parked_tokens(1); // barrier: the last decode step's sample is written
    let mut recs = Vec::new();
    ring.drain(&mut recs, usize::MAX);
    let steps: Vec<_> = recs
        .into_iter()
        .filter_map(|r| match r {
            Record::EngineSample(_, s) => Some((s.phase(), s.n_decode_seqs, s.n_prefill_seqs)),
            _ => None,
        })
        .collect();
    assert_eq!(steps, vec![(Phase::Decode, pl as u32, 0); tg as usize]);
}

/// #64 races: a cancel, or a reset (cancel, then release), landing while the group's first decode
/// step runs ends that request Cancelled and parks nothing for it; the rest decode on together and park.
#[test]
fn a_cancel_or_reset_during_the_groups_first_step() {
    for reset in [false, true] {
        let hold = Arc::new(Barrier::new(2));
        let held = hold.clone();
        // Forward 1 prefills the group; forward 2 is its first decode step.
        let load = move |_: &_, _: &_, _| {
            let m = Held {
                inner: Box::new(FakeModel::new(64)),
                at: 2,
                hold: held,
            };
            Ok(Box::new(m) as Box<dyn Model>)
        };
        let e = start_with(fake_opts(vec![], 512), Box::new(load)).unwrap();
        let prompts: Vec<Vec<u32>> = (0..4u32)
            .map(|i| (0..16).map(|t| t + i).collect())
            .collect();
        let prefill = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| kept(i as u64 + 1, p.clone(), 1));
        let first: Vec<u32> = e
            .submit_group(prefill.collect())
            .unwrap()
            .into_iter()
            .map(|h| ids(&run(h).0)[0])
            .collect();
        e.parked_tokens(1); // barrier: the prefill step's sample is written before the reader
        let mut ring = e.ring().reader(ReadFrom::Latest);
        let group = prompts.iter().zip(&first).enumerate().map(|(i, (p, &f))| {
            let mut p = p.clone();
            p.push(f);
            kept(i as u64 + 1, p, 4)
        });
        let mut hs = e.submit_group(group.collect()).unwrap();
        hold.wait(); // the group's first decode step is in flight
        hs[1].cancel();
        if reset {
            e.release_session(2);
        }
        hold.wait();
        assert_eq!(
            run(hs.remove(1)).1,
            EngineEvent::Error(EngineError::Cancelled),
            "reset={reset}"
        );
        for h in hs {
            assert_eq!(done(&run(h).1), (Finish::Length, 4, 16), "reset={reset}");
        }
        assert_eq!(e.parked_tokens(2), 0, "reset={reset}");
        for s in [1, 3, 4] {
            assert_eq!(e.parked_tokens(s), 20, "reset={reset}");
        }
        let mut recs = Vec::new();
        ring.drain(&mut recs, usize::MAX);
        let steps: Vec<_> = recs
            .into_iter()
            .filter_map(|r| match r {
                Record::EngineSample(_, s) => Some((s.phase(), s.n_decode_seqs, s.n_prefill_seqs)),
                _ => None,
            })
            .collect();
        assert_eq!(steps[0], (Phase::Decode, 4, 0), "reset={reset}");
        assert!(
            steps[1..].iter().all(|&s| s == (Phase::Decode, 3, 0)),
            "reset={reset}: {steps:?}"
        );
    }
}

/// submit_group admits all or none: one refusal refuses the whole group, each request with a
/// Rejected span, and none of it runs.
#[test]
fn a_refused_request_refuses_its_group() {
    let e = fake(vec![], 64);
    let group = vec![Request::new(vec![1], 4), Request::new(vec![], 4)];
    assert_eq!(
        e.submit_group(group).err(),
        Some(EngineError::InvalidRequest(invalid::EMPTY_PROMPT))
    );
    let h = e
        .submit_group(vec![Request::new(vec![1], 2)])
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(done(&run(h).1).0, Finish::Length);
    e.parked_tokens(0); // barrier: every span is written
    let outcomes: Vec<Outcome> = records(&e)
        .into_iter()
        .filter_map(|r| match r {
            Record::RequestSpan(_, s) => Some(s.outcome()),
            _ => None,
        })
        .collect();
    assert_eq!(
        outcomes,
        [Outcome::Rejected, Outcome::Rejected, Outcome::Ok]
    );
}

/// submit_group leaves the queue's accounting exact after a refusal and after every member runs.
#[test]
fn a_group_keeps_the_queue_count_exact() {
    let e = fake(vec![(knob_id::MAX_WAITING, 2.0)], 512);
    let reqs = |n| (0..n).map(|_| Request::new(vec![1], 2)).collect::<Vec<_>>();
    assert!(matches!(
        e.submit_group(reqs(3)),
        Err(EngineError::Overloaded { .. })
    ));
    for _ in 0..2 {
        for h in e.submit_group(reqs(2)).unwrap() {
            assert_eq!(done(&run(h).1).0, Finish::Length);
        }
        // The engine has published the scheduler queue count before this reply.
        e.parked_tokens(0);
    }
}

/// Advances the test clock by 7 ms per forward; nothing else moves engine time.
struct Ticking(FakeModel, Arc<AtomicU64>);
impl Model for Ticking {
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
        self.1.fetch_add(7_000_000, Ordering::AcqRel);
        self.0.forward(s, b, kv, logits)
    }
}

#[test]
fn a_test_clock_sets_each_step_duration() {
    let clock = Arc::new(AtomicU64::new(0));
    let mut opts = fake_opts(Vec::new(), 256);
    opts.clock = Some(clock.clone());
    let e = start_with(
        opts,
        Box::new(move |_, _, _| {
            Ok(Box::new(Ticking(FakeModel::new(64), clock.clone())) as Box<dyn Model>)
        }),
    )
    .unwrap();
    run(e.submit(Request::new(Vec::from([1]), 3)).unwrap());
    let durs: Vec<u64> = records(&e)
        .into_iter()
        .filter_map(|r| match r {
            Record::EngineSample(_, s) => Some(s.dur_ns),
            _ => None,
        })
        .collect();
    assert!(!durs.is_empty());
    assert!(durs.iter().all(|&d| d == 7_000_000), "{durs:?}");
}

/// FakeModel's tapped row for input token `t` after layer `l` (d_model 16).
fn tap(t: u32, l: u16) -> Vec<f32> {
    [t as f32, f32::from(l)]
        .into_iter()
        .chain((2..16).map(|i| i as f32))
        .collect()
}

/// A FakeModel engine with `n_layers` layers to tap.
fn fake_layers(n_layers: u32, overrides: Vec<(u16, f64)>) -> EngineHandle {
    start_with(
        fake_opts(overrides, 4096),
        Box::new(move |_, _, _| {
            Ok(Box::new(FakeModel::with_layers(64, n_layers)) as Box<dyn Model>)
        }),
    )
    .unwrap()
}

/// A prefill-only request for `layers`.
fn activations(prompt: Vec<u32>, layers: &[u16]) -> Request {
    Request {
        activations: layers.to_vec(),
        ..Request::new(prompt, 0)
    }
}

/// The rows of an activations request, which ends with Done (Length) and no token.
fn rows(h: RequestHandle) -> Vec<Vec<f32>> {
    match run(h) {
        (
            t,
            EngineEvent::Done {
                finish: Finish::Length,
                usage,
                activations,
                ..
            },
        ) if t.is_empty() && usage.completion_tokens == 0 => activations,
        e => panic!("{e:?}"),
    }
}

/// An activations request prefills only and reads the last prompt token after each named layer, in
/// the order named, also when its prompt ends in a one-token chunk (64, 64, then 1).
#[test]
fn activations_read_the_last_token_after_each_layer() {
    let e = fake_layers(3, vec![]);
    let h = e.submit(activations(vec![5, 9, 13], &[2, 0, 2])).unwrap();
    assert_eq!(rows(h), [tap(13, 2), tap(13, 0), tap(13, 2)]);
    let e = fake_layers(3, vec![(knob_id::PREFILL_CHUNK_TOKENS, 64.0)]);
    let mut prompt = vec![5; 128];
    prompt.push(11);
    let h = e.submit(activations(prompt, &[1])).unwrap();
    assert_eq!(rows(h), [tap(11, 1)]);
}

/// Two activations requests and a generating one in the same step: each gets its own rows.
#[test]
fn two_activations_in_flight_get_their_own_rows() {
    let e = fake_layers(4, vec![]);
    let mut hs = e
        .submit_group(vec![
            activations(vec![1, 7], &[3, 0]),
            Request::new(vec![4], 2),
            activations(vec![2, 9, 20], &[1]),
        ])
        .unwrap()
        .into_iter();
    let (a, g, b) = (hs.next().unwrap(), hs.next().unwrap(), hs.next().unwrap());
    assert_eq!(rows(a), [tap(7, 3), tap(7, 0)]);
    assert_eq!(rows(b), [tap(20, 1)]);
    assert_eq!(ids(&run(g).0), [5, 6]);
    let first = records(&e).into_iter().find_map(|r| match r {
        Record::EngineSample(_, s) => Some(s.n_prefill_tokens),
        _ => None,
    });
    assert_eq!(first, Some(6), "all three prefilled in one step");
}

/// A layer past the model's depth, more layers than it has, or an activations request that would
/// also sample or touch a session is refused at submit.
#[test]
fn activations_are_checked_at_intake() {
    let e = fake_layers(3, vec![]);
    let err = |r: Request| e.submit(r).err();
    let layer = Some(EngineError::InvalidRequest(invalid::ACTIVATION_LAYER));
    let sampling = Some(EngineError::InvalidRequest(invalid::ACTIVATIONS));
    assert_eq!(err(activations(vec![1], &[3])), layer);
    assert_eq!(err(activations(vec![1], &[0, 1, 2, 0])), layer);
    let one = Request {
        max_tokens: 1,
        ..activations(vec![1], &[0])
    };
    assert_eq!(err(one), sampling);
    let scored = Request {
        score_ids: vec![2],
        ..activations(vec![1], &[0])
    };
    assert_eq!(err(scored), sampling);
    let session = Request {
        session: 5,
        ..activations(vec![1], &[0])
    };
    assert_eq!(err(session), sampling);
    assert_eq!(err(activations(vec![1], &[2, 0, 1])), None);
}

/// On the tiny model, rows read after a prefix from the automatic cache match a fresh prefill, within
/// `same_distribution`'s 1e-2 scaled to the value (the f16 KV paths of a cached prefix and a fresh
/// chunk round differently).
#[test]
fn activations_after_a_cached_prefix_match_a_fresh_prefill() {
    let p: Vec<u32> = (0..60).map(|i| (i * 13 + 5) % 250).collect();
    let mut opts = EngineOptions::new(tiny_path(), CPU);
    opts.prefix_cache_blocks = 64;
    let cached = start(opts).unwrap();
    run(cached.submit(Request::new(p[..48].to_vec(), 1)).unwrap());
    let read = |e: &EngineHandle| match run(e.submit(activations(p.clone(), &[0, 1])).unwrap()) {
        (
            t,
            EngineEvent::Done {
                usage, activations, ..
            },
        ) if t.is_empty() => (usage.cached_tokens, activations),
        e => panic!("{e:?}"),
    };
    let (hit, warm) = read(&cached);
    let (miss, cold) = read(&tiny_engine());
    assert_eq!((hit, miss), (48, 0));
    for (name, rows) in [("warm", &warm), ("cold", &cold)] {
        assert_eq!(rows.len(), 2, "{name} rows");
        for r in rows {
            assert_eq!(r.len(), tiny::D as usize, "{name} width");
            assert!(r.iter().all(|x| x.is_finite()), "{name}: {r:?}");
        }
    }
    // Same shapes, so every element is compared.
    for (l, (w, c)) in warm.iter().zip(&cold).enumerate() {
        for (i, (a, b)) in w.iter().zip(c).enumerate() {
            let diff = (a - b).abs() / b.abs().max(1.0);
            assert!(diff < 1e-2, "row {l} [{i}]: {a} vs {b}");
        }
    }
}

/// FakeModel counting the passes: `forward` in `plain`, `forward_taps` in `tapped`.
struct Counted {
    model: FakeModel,
    plain: Arc<AtomicU64>,
    tapped: Arc<AtomicU64>,
}

impl Model for Counted {
    fn config(&self) -> &ModelConfig {
        self.model.config()
    }

    fn forward(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        self.plain.fetch_add(1, Ordering::Relaxed);
        self.model.forward(s, b, kv, logits)
    }

    fn forward_taps(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
        taps: Option<&Taps>,
    ) -> Result<(), KError> {
        self.tapped.fetch_add(1, Ordering::Relaxed);
        self.model.forward_taps(s, b, kv, logits, taps)
    }
}

/// With no activations request admitted, every step runs `forward`, never `forward_taps`, as
/// before activations existed: chunked prefill, decode, and steps mixing them. The one tapped
/// step runs `forward_taps`, and the steps after it `forward` again.
#[test]
fn steps_without_activations_run_forward_only() {
    let (plain, tapped) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let (p, t) = (plain.clone(), tapped.clone());
    let e = start_with(
        fake_opts(vec![(knob_id::PREFILL_CHUNK_TOKENS, 64.0)], 4096),
        Box::new(move |_, _, _| {
            Ok(Box::new(Counted {
                model: FakeModel::with_layers(64, 2),
                plain: p.clone(),
                tapped: t.clone(),
            }) as Box<dyn Model>)
        }),
    )
    .unwrap();
    let long: Vec<u32> = (0..129).map(|i| i % 50 + 1).collect();
    let group = vec![Request::new(vec![3], 6), Request::new(long, 3)];
    for h in e.submit_group(group).unwrap() {
        let end = run(h).1;
        assert!(
            matches!(
                end,
                EngineEvent::Done {
                    finish: Finish::Length,
                    ..
                }
            ),
            "{end:?}"
        );
    }
    let before = plain.load(Ordering::Relaxed);
    assert!(before >= 6, "{before} passes");
    assert_eq!(
        tapped.load(Ordering::Relaxed),
        0,
        "forward_taps without taps"
    );
    assert_eq!(
        rows(e.submit(activations(vec![5, 9], &[1])).unwrap()),
        [tap(9, 1)]
    );
    assert_eq!(tapped.load(Ordering::Relaxed), 1);
    assert_eq!(plain.load(Ordering::Relaxed), before);
    run(e.submit(Request::new(vec![4], 2)).unwrap());
    assert_eq!(tapped.load(Ordering::Relaxed), 1);
    assert_eq!(plain.load(Ordering::Relaxed), before + 2);
}

/// FakeModel without taps: `forward_taps` stays the trait's default, which refuses them.
struct Untapped(FakeModel);

impl Model for Untapped {
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
        self.0.forward(s, b, kv, logits)
    }
}

/// A model without taps ends only the activations request, with NO_TAPS; the request in the same
/// step generates and the engine stays Ready.
#[test]
fn a_model_without_taps_refuses_only_the_activations_request() {
    let e = start_with(
        fake_opts(vec![], 4096),
        Box::new(|_, _, _| Ok(Box::new(Untapped(FakeModel::with_layers(64, 2))) as Box<dyn Model>)),
    )
    .unwrap();
    let mut hs = e
        .submit_group(vec![
            activations(vec![1, 7], &[0]),
            Request::new(vec![4], 2),
        ])
        .unwrap()
        .into_iter();
    let (a, g) = (hs.next().unwrap(), hs.next().unwrap());
    let (t, end) = run(a);
    assert!(t.is_empty());
    assert_eq!(
        end,
        EngineEvent::Error(EngineError::InvalidRequest(invalid::NO_TAPS))
    );
    assert_eq!(ids(&run(g).0), [5, 6]);
    assert_eq!(e.health().state, HealthState::Ready);
}

/// Token id b is byte b, as the tiny model's facade vocabulary shows it.
fn grammar(trigger: Option<&str>, pattern: &str, vocab: Vec<Vec<u8>>) -> Constraint {
    Constraint {
        trigger: trigger.map(Into::into),
        compiled: Arc::new(Compiled::new(pattern, vocab.into()).unwrap()),
        replay: Vec::new(),
    }
}

fn bytes() -> Vec<Vec<u8>> {
    (0..=255u8).map(|b| vec![b]).collect()
}

fn sampled(temperature: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_p: 0.95,
        top_k: 40,
        repeat_penalty: 1.1,
        seed: Some(7),
    }
}

/// Token events in full (ids, logprobs, top lists) and the grammar's forced picks.
fn tokens(h: RequestHandle) -> (Vec<EngineEvent>, u32) {
    let mut out = Vec::new();
    for e in h.events.iter() {
        match e {
            EngineEvent::Done { timings, .. } => return (out, timings.grammar_forced),
            EngineEvent::Error(e) => panic!("{e:?}"),
            EngineEvent::Prefilled { .. } => {}
            token => out.push(token),
        }
    }
    panic!("stream ended without Done");
}

/// Free mode is the old path: a seeded request whose trigger never shows is bit-identical to the
/// same request with no grammar.
#[test]
fn grammar_free_mode_is_bit_identical() {
    let e = tiny_engine();
    let r = Request {
        sampling: sampled(1.0),
        logprobs: 5,
        stop_tokens: vec![255],
        ..Request::new((0..20).map(|i| (i * 7 + 3) % 250).collect(), 48)
    };
    let free = tokens(e.submit(r.clone()).unwrap());
    let never = Request {
        grammar: Some(grammar(Some("\0\0\0\0\0\0\0\0"), "x", bytes())),
        ..r
    };
    assert_eq!(tokens(e.submit(never).unwrap()), free);
    assert_eq!(free.0.len(), 48);
}

/// Constrained from the first token: every token keeps the pattern alive, and once it can go no
/// further only the stop is allowed.
#[test]
fn grammar_constrains_then_stops() {
    let e = tiny_engine();
    for t in [0.0, 1.0] {
        let r = Request {
            sampling: sampled(t),
            stop_tokens: vec![255],
            grammar: Some(grammar(None, "[a-c]{5}", bytes())),
            ..Request::new(vec![1, 2, 3], 40)
        };
        let (events, forced) = tokens(e.submit(r).unwrap());
        let ids: Vec<u32> = (events.iter())
            .map(|e| match e {
                EngineEvent::Token { id, .. } => *id,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(ids.len(), 6, "{ids:?}");
        assert!(ids[..5].iter().all(|&i| (97..=99).contains(&i)), "{ids:?}");
        assert_eq!((ids[5], forced), (255, 0));
    }
}

/// A greedy grammar row gets all 64 candidates from the device, so a wide state's filter finds an
/// allowed one below the model's (disallowed) pick instead of forcing.
#[test]
fn greedy_grammar_rows_filter_every_candidate() {
    let e = start_with(
        fake_opts(Vec::new(), 512),
        Box::new(|_, _, _| Ok(Box::new(FakeModel::new(1000)) as Box<dyn Model>)),
    )
    .unwrap();
    // Even ids are letters (500 of them: a wide state), odd ids digits.
    let vocab: Vec<Vec<u8>> = (0..1000u32)
        .map(|id| match id % 2 {
            0 => vec![b'a' + (id / 2 % 26) as u8],
            _ => vec![b'0' + (id / 2 % 10) as u8],
        })
        .collect();
    let r = Request {
        grammar: Some(grammar(None, "[a-z]{6}", vocab)),
        stop_tokens: vec![999],
        ..Request::new(vec![1], 20)
    };
    let (events, forced) = tokens(e.submit(r).unwrap());
    assert_eq!(forced, 0);
    assert_eq!(events.len(), 7);
}

#[test]
fn grammar_refusals() {
    let e = tiny_engine();
    let g = || Some(grammar(None, "a", bytes()));
    let refused = |r: Request| e.submit(r).err();
    let base = Request {
        stop_tokens: vec![255],
        ..Request::new(vec![1], 4)
    };
    let code = |c| Some(EngineError::InvalidRequest(c));
    assert_eq!(
        refused(Request {
            grammar: g(),
            score_ids: vec![1],
            ..base.clone()
        }),
        code(invalid::GRAMMAR_SCORE_IDS)
    );
    assert_eq!(
        refused(Request {
            grammar: g(),
            ignore_eos: true,
            ..base.clone()
        }),
        code(invalid::GRAMMAR_IGNORE_EOS)
    );
    assert_eq!(
        refused(Request {
            grammar: Some(grammar(Some(""), "a", bytes())),
            ..base.clone()
        }),
        code(invalid::GRAMMAR_PATTERN)
    );
    assert_eq!(
        refused(Request {
            grammar: g(),
            stop_tokens: Vec::new(),
            ..base.clone()
        }),
        code(invalid::GRAMMAR_STOPS)
    );
    // A triggered one too: `[0-9]+` matches and continues, so only a stop can end it there.
    assert_eq!(
        refused(Request {
            grammar: Some(grammar(Some("z"), "[0-9]+", bytes())),
            stop_tokens: Vec::new(),
            ..base.clone()
        }),
        code(invalid::GRAMMAR_STOPS)
    );
    // Its ids are gathered on the device: none may pass the model's vocabulary.
    let mut wide = bytes();
    wide.push(b"x".to_vec());
    assert_eq!(
        refused(Request {
            grammar: Some(grammar(None, "a", wide)),
            ..base.clone()
        }),
        code(invalid::TOKEN_OUT_OF_VOCAB)
    );
    assert_eq!(
        refused(Request {
            grammar: g(),
            stop_tokens: vec![256],
            ..base.clone()
        }),
        code(invalid::TOKEN_OUT_OF_VOCAB)
    );
    assert!(
        e.submit(Request {
            grammar: g(),
            ..base
        })
        .is_ok()
    );
}
