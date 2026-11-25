//! Workers on the CPU engine with the fake model and the tiny tokenizer: answers in order, the
//! system prompt, the batch cap, KV kept for the group, sessions freed after the run, the one
//! retry, and the system prompt's KV shared through the prefix cache.

use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use larch::{
    ChatMessage, EngineError, Finish, GenParams, Larch, Prompt, Role, SessionOptions, TextEvent,
};
use larch_agent::workers::{Answer, Workers};
use larch_engine::{EngineHandle, EngineOptions, start_with};
use larch_kernels::{CPU, KError, KvPool, Stream, Tensor};
use larch_model::{
    ChatTemplate, FakeModel, Model, ModelConfig, StepBatch, Tokenizer, gguf::Gguf, tiny,
};
use larch_records::{Read, ReadFrom, Record};

/// The fake model at 2 ms a step, recording the most sequences one step carried.
struct Counting {
    inner: FakeModel,
    most: Arc<AtomicUsize>,
}

fn nonzero(s: &Stream, t: &Tensor) -> Result<usize, KError> {
    let mut v = vec![0i32; t.shape()[0] as usize];
    s.download(t.buffer(), t.offset(), &mut v)?;
    Ok(v.iter().filter(|&&n| n > 0).count())
}

impl Model for Counting {
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
        // Padding decode rows attend to nothing (seq_lens 0); every prefill chunk has rows.
        let seqs = nonzero(s, &b.seq_lens)? + nonzero(s, &b.prefill_q_len)?;
        self.most.fetch_max(seqs, Ordering::Relaxed);
        thread::sleep(Duration::from_millis(2));
        self.inner.forward(s, b, kv, logits)
    }
}

/// An engine whose template ends the prompt with the system message when there is one, else
/// with the user's: the fake model continues from the last token, so an answer shows which
/// message it ended on, and prompts without a system message each get their own answer.
fn engine() -> (Larch, EngineHandle, Arc<AtomicUsize>) {
    engine_with(
        "{{ messages[-1].content }}{% if messages[0].role == 'system' %}{{ messages[0].content }}{% endif %}",
    )
}

/// The fake model behind `template`.
fn engine_with(template: &str) -> (Larch, EngineHandle, Arc<AtomicUsize>) {
    engine_opts(template, EngineOptions::new("fake".into(), CPU))
}

/// The fake model behind `template`, with the prefix cache on, and its tokenizer.
fn cached(template: &str) -> (Larch, EngineHandle, Tokenizer) {
    let mut opts = EngineOptions::new("fake".into(), CPU);
    opts.prefix_cache_blocks = 64;
    opts.prefix_cache_custom_compatible = true;
    let (larch, engine, _) = engine_opts(template, opts);
    let g = Gguf::parse(&std::fs::read(tiny::TOKENIZER_GGUF).unwrap()).unwrap();
    (larch, engine, Tokenizer::from_gguf(&g).unwrap())
}

fn engine_opts(template: &str, opts: EngineOptions) -> (Larch, EngineHandle, Arc<AtomicUsize>) {
    let bytes = std::fs::read(tiny::TOKENIZER_GGUF).unwrap();
    let g = Gguf::parse(&bytes).unwrap();
    let tokenizer = Tokenizer::from_gguf(&g).unwrap();
    let template = ChatTemplate::new(template, &tokenizer).unwrap();
    let most = Arc::new(AtomicUsize::new(0));
    let model = Counting {
        inner: FakeModel::new(512),
        most: most.clone(),
    };
    let engine = start_with(opts, Box::new(move |_, _, _| Ok(Box::new(model)))).unwrap();
    let larch = Larch::from_engine(engine.clone(), tokenizer, template);
    (larch, engine, most)
}

fn params() -> GenParams {
    GenParams {
        max_tokens: 8,
        temperature: Some(0.0),
        ..GenParams::default()
    }
}

/// The prompt's answer from one session on its own.
fn alone(larch: &Larch, system: &str, prompt: &str) -> (String, Finish) {
    let mut session = larch.session(SessionOptions {
        system: (!system.is_empty()).then(|| system.into()),
        keep_kv: false,
        ..SessionOptions::default()
    });
    let mut text = String::new();
    for event in session.send(prompt, params()).unwrap() {
        match event {
            TextEvent::Prefilled { .. } | TextEvent::Thinking { .. } => {}
            TextEvent::Text { delta, .. } => text.push_str(&delta),
            TextEvent::Done { finish, .. } => return (text, finish),
            TextEvent::Error(e) => panic!("{e:?}"),
        }
    }
    panic!("no Done")
}

fn prompts() -> Vec<String> {
    ["a", "f", "k", "p", "u"].map(String::from).to_vec()
}

/// Each prompt's only answer, as text and finish, or the error before it: no answer was retried.
fn firsts(
    got: Vec<(Vec<Answer>, Option<EngineError>)>,
) -> Vec<Result<(String, Finish), EngineError>> {
    got.into_iter()
        .map(|(a, error)| match error {
            Some(e) => {
                assert!(a.is_empty(), "{a:?}");
                Err(e)
            }
            None => {
                assert_eq!(a.len(), 1, "{a:?}");
                Ok((a[0].text.clone(), a[0].finish))
            }
        })
        .collect()
}

#[test]
fn workers_return_every_answer_in_prompt_order() {
    let (larch, _, _) = engine();
    let want: Vec<_> = prompts().iter().map(|p| alone(&larch, "", p)).collect();
    assert!(want.windows(2).all(|w| w[0] != w[1]), "{want:?}");
    let got = Workers {
        max_workers: 3,
        prefix_cache: false,
    }
    .run(&larch, "", &prompts(), &params(), |_| None);
    assert_eq!(firsts(got), want.into_iter().map(Ok).collect::<Vec<_>>());
}

#[test]
fn every_worker_gets_the_system_prompt() {
    let (larch, _, _) = engine();
    let mut answers = Vec::new();
    for system in ["Review this.", "Sum it up!"] {
        let want: Vec<_> = prompts()
            .iter()
            .map(|p| Ok(alone(&larch, system, p)))
            .collect();
        let got = firsts(
            Workers {
                max_workers: 3,
                prefix_cache: false,
            }
            .run(&larch, system, &prompts(), &params(), |_| None),
        );
        assert_eq!(got, want, "{system}");
        answers.push(got);
    }
    // Another system prompt, other answers: the comparison above saw the system prompt.
    assert_ne!(answers[0], answers[1]);
}

#[test]
fn a_failed_prompt_keeps_its_place() {
    let (larch, _, _) = engine();
    // Thousands of tokens: past the fake model's 4,096-token context.
    let long: String = (0..3000).map(|i| format!("{i} ")).collect();
    let mut asked = prompts();
    asked.insert(2, long);
    let got = firsts(Workers::default().run(&larch, "", &asked, &params(), |_| None));
    assert_eq!(got.len(), 6);
    assert!(
        matches!(got[2], Err(EngineError::PromptTooLong { .. })),
        "{:?}",
        got[2]
    );
    for (i, answer) in got.iter().enumerate().filter(|(i, _)| *i != 2) {
        assert!(
            answer.as_ref().is_ok_and(|(a, _)| !a.is_empty()),
            "{i}: {answer:?}"
        );
    }
}

#[test]
fn workers_leave_one_batch_slot_free() {
    for (max_workers, cap) in [(3, 2), (1, 1)] {
        let (larch, _, most) = engine();
        let got = Workers {
            max_workers,
            prefix_cache: false,
        }
        .run(&larch, "Review this.", &prompts(), &params(), |_| None);
        assert!(got.iter().all(|(_, error)| error.is_none()), "{got:?}");
        // At most the cap, and the cap reached, so the count saw the workers run together.
        assert_eq!(
            most.load(Ordering::Relaxed),
            cap,
            "max_workers {max_workers}"
        );
    }
}

#[test]
fn workers_keep_their_kv_for_the_group_and_say_how_each_answer_ended() {
    let (larch, engine, _) = engine();
    // "Hi!\n" ends in the token before end-of-text, so that worker stops after a few steps
    // while the other runs to max_tokens.
    let asked = ["Hi!\n", "a"].map(String::from);
    let got = firsts(
        Workers {
            max_workers: 3,
            prefix_cache: true,
        }
        .run(&larch, "", &asked, &params(), |_| None),
    );
    let finish: Vec<_> = got.iter().map(|a| a.as_ref().unwrap().1).collect();
    assert_eq!(finish, [Finish::Eos, Finish::Length]);
    // The engine samples every step after its sequences finish: a worker keeps its KV, which the
    // prefix cache needs, parked until its session is dropped after the group.
    let mut ring = engine.ring().reader(ReadFrom::Oldest);
    let (mut steps, mut parked) = (0, 0);
    while let Read::Record(record) = ring.next() {
        if let Record::EngineSample(_, s) = record {
            steps += 1;
            parked = parked.max(s.n_parked);
        }
    }
    assert!(steps > 0);
    assert!(parked > 0);
}

/// A system prompt of several 16-token blocks, and seven prompts after it.
const SHARED: &str = "Read the excerpt and list each fact in it that helps answer the question. \
                      Question: when does the train to Leith leave on a Sunday morning?";

/// Prompt tokens a worker on `SHARED` takes from the prefix cache: the tokens its prompt and the
/// anchor's share, in whole 16-token blocks.
fn shared_len(tokenizer: &Tokenizer) -> u64 {
    let anchor = tokenizer.encode(&format!("{SHARED}|"), true);
    let worker = tokenizer.encode(&format!("{SHARED}|a"), true);
    let common = anchor
        .iter()
        .zip(&worker)
        .take_while(|(a, b)| a == b)
        .count();
    (common / 16 * 16) as u64
}

#[test]
fn seven_workers_share_the_anchored_system_prompt() {
    let (larch, engine, tokenizer) = cached("{{ messages[0].content }}|{{ messages[-1].content }}");
    let shared = shared_len(&tokenizer);
    assert!(shared >= 32, "{shared}");
    let prompts: Vec<String> = ["a", "f", "k", "p", "u", "z", "e"]
        .map(String::from)
        .to_vec();
    let before = engine.prefix_cache_stats().unwrap();
    // One group of seven.
    let got = Workers {
        max_workers: 8,
        prefix_cache: true,
    }
    .run(&larch, SHARED, &prompts, &params(), |_| None);
    let after = engine.prefix_cache_stats().unwrap();
    for (answers, error) in &got {
        assert!(error.is_none(), "{error:?}");
        assert_eq!(
            u64::from(answers[0].usage.cached_tokens),
            shared,
            "{answers:?}"
        );
    }
    let reused = after.reused_prompt_tokens - before.reused_prompt_tokens;
    assert_eq!(reused, 7 * shared, "{before:?} {after:?}");
}

#[test]
fn sessions_that_keep_no_kv_share_nothing_with_the_anchor() {
    // Workers before the prefix cache: the anchor's prefix is there, and they cannot take it.
    let (larch, engine, tokenizer) = cached("{{ messages[0].content }}|{{ messages[-1].content }}");
    assert!(shared_len(&tokenizer) >= 32);
    let message = |role: &str, content: &str| larch_model::Message {
        role: role.into(),
        content: content.into(),
    };
    let anchor = Prompt::Messages(vec![message("system", SHARED), message("user", "")]);
    let p = GenParams {
        max_tokens: 1,
        ..params()
    };
    assert!(
        larch
            .generate(anchor, p)
            .unwrap()
            .any(|e| matches!(e, TextEvent::Done { .. }))
    );
    let before = engine.prefix_cache_stats().unwrap();
    let opts = SessionOptions {
        system: Some(SHARED.into()),
        keep_kv: false,
        ..SessionOptions::default()
    };
    let mut sessions: Vec<_> = (0..7).map(|_| larch.session(opts.clone())).collect();
    let streams: Vec<_> = (sessions.iter_mut())
        .zip(["a", "f", "k", "p", "u", "z", "e"])
        .map(|(s, prompt)| s.send(prompt, params()).unwrap())
        .collect();
    for stream in streams {
        assert!(
            stream
                .into_iter()
                .any(|e| matches!(e, TextEvent::Done { .. }))
        );
    }
    let after = engine.prefix_cache_stats().unwrap();
    assert_eq!(after.reused_prompt_tokens, before.reused_prompt_tokens);
}

#[test]
fn a_finished_worker_frees_its_session() {
    let (larch, engine, _) = engine();
    let before = larch.session(SessionOptions::default()).id();
    let got = Workers::default().run(&larch, "Review this.", &prompts(), &params(), |_| None);
    assert_eq!(got.len(), 5);
    let after = larch.session(SessionOptions::default());
    assert_eq!(after.tokens_in_kv(), 0);
    // Session IDs only grow, so the workers' sessions lie between the two: each was dropped,
    // which frees any KV it parked.
    assert!((before + 1..after.id()).all(|id| engine.parked_tokens(id) == 0));
}

#[test]
fn a_refused_answer_is_retried_once_in_its_own_session() {
    // The prompt ends with the number of messages, so an answer shows how long a history it was
    // given.
    let (larch, _, _) = engine_with("{{ messages[-1].content }}{{ messages|length }}");
    let got = Workers {
        max_workers: 3,
        prefix_cache: false,
    }
    .run(&larch, "", &prompts(), &params(), |_| Some("k".into()));
    for (prompt, (answers, error)) in prompts().iter().zip(got) {
        assert!(error.is_none(), "{prompt}: {error:?}");
        assert_eq!(answers.len(), 2, "{prompt}: {answers:?}");
        // The same conversation, continued by hand: the prompt, its answer, then the feedback.
        let mut session = larch.session(SessionOptions::default());
        let message = |role, content: &str| ChatMessage {
            role,
            content: content.into(),
            ..ChatMessage::default()
        };
        let history = vec![
            message(Role::User, prompt),
            message(Role::Assistant, &answers[0].text),
        ];
        session.restore(SessionOptions::default(), history).unwrap();
        let mut want = String::new();
        for event in session.send("k", params()).unwrap() {
            if let TextEvent::Text { delta, .. } = event {
                want.push_str(&delta);
            }
        }
        assert_eq!(answers[1].text, want, "{prompt}");
        // A fresh session given only the feedback answers otherwise: the history reached it.
        assert_ne!(answers[1].text, alone(&larch, "", "k").0, "{prompt}");
    }
}

#[test]
fn only_refused_answers_are_retried_and_a_retry_is_not_checked() {
    let (larch, _, _) = engine();
    let refused = alone(&larch, "", "a").0;
    let checked = Cell::new(0);
    let got = Workers {
        max_workers: 3,
        prefix_cache: false,
    }
    .run(&larch, "", &prompts(), &params(), |a| {
        checked.set(checked.get() + 1);
        (a.text == refused).then(|| "again".into())
    });
    assert!(got.iter().all(|(_, error)| error.is_none()), "{got:?}");
    let counts: Vec<_> = got.iter().map(|(answers, _)| answers.len()).collect();
    assert_eq!(counts, [2, 1, 1, 1, 1]);
    assert_eq!(checked.get(), prompts().len());
    // Each answer carries the engine's counts and timings.
    let first = &got[1].0[0];
    assert!(first.usage.prompt_tokens > 0 && first.usage.completion_tokens > 0);
    assert!(first.timings.ttft_ms > 0.0, "{first:?}");
}

#[test]
fn a_failed_retry_keeps_the_first_answer() {
    let (larch, _, _) = engine();
    // Thousands of tokens: past the fake model's 4,096-token context.
    let long: String = (0..3000).map(|i| format!("{i} ")).collect();
    let got = Workers {
        max_workers: 3,
        prefix_cache: false,
    }
    .run(&larch, "", &prompts()[..1], &params(), |_| {
        Some(long.clone())
    });
    let (answers, error) = &got[0];
    assert_eq!(answers.len(), 1, "{answers:?}");
    assert_eq!(answers[0].text, alone(&larch, "", "a").0);
    assert!(
        matches!(error, Some(EngineError::PromptTooLong { .. })),
        "{error:?}"
    );
}

#[test]
fn the_retries_of_a_group_run_together() {
    let (larch, _, most) = engine();
    let p = GenParams {
        max_tokens: 64,
        ..params()
    };
    // Two groups of two (the cap with max_workers 3); the count restarts before each retry is
    // sent, so it ends with what the last group's retries carried at once.
    let got = Workers {
        max_workers: 3,
        prefix_cache: false,
    }
    .run(&larch, "", &prompts()[..4], &p, |_| {
        most.store(0, Ordering::Relaxed);
        Some("again".into())
    });
    assert!(
        got.iter().all(|(a, e)| a.len() == 2 && e.is_none()),
        "{got:?}"
    );
    assert_eq!(most.load(Ordering::Relaxed), 2);
}

#[test]
fn workers_without_the_cache_send_no_anchor_and_park_nothing() {
    // The engine's cache is off: an anchor or parked KV would only cost a prefill and the pool.
    let (larch, engine, _) = engine();
    assert_eq!(engine.prefix_cache_stats().unwrap().capacity_blocks, 0);
    let got = Workers {
        max_workers: 3,
        prefix_cache: false,
    }
    .run(&larch, "Review this.", &prompts(), &params(), |_| None);
    let answers: Vec<_> = got.iter().flat_map(|(a, _)| a).collect();
    assert_eq!(answers.len(), 5, "{got:?}");
    // Every prompt token the engine computed is a worker's: none is an anchor's.
    let fed: u64 = answers
        .iter()
        .map(|a| u64::from(a.usage.prompt_tokens))
        .sum();
    let computed = engine.prefix_cache_stats().unwrap().computed_prompt_tokens;
    assert_eq!(computed, fed, "{got:?}");
    let mut ring = engine.ring().reader(ReadFrom::Oldest);
    let mut parked = 0;
    while let Read::Record(record) = ring.next() {
        if let Record::EngineSample(_, s) = record {
            parked = parked.max(s.n_parked);
        }
    }
    assert_eq!(parked, 0);
}

#[test]
fn a_run_stopped_before_it_starts_sends_not_even_the_anchor() {
    // With the cache on the system prompt's anchor goes first, and no stop could cancel it once
    // sent: a run whose stop is already set (research's editor past its writing time) sends
    // the engine nothing.
    let (larch, engine, _) = cached("{{ messages[0].content }}|{{ messages[-1].content }}");
    let before = engine.prefix_cache_stats().unwrap().computed_prompt_tokens;
    let stop = std::sync::atomic::AtomicBool::new(true);
    let got = Workers {
        max_workers: 3,
        prefix_cache: true,
    }
    .run_helpers(
        &larch,
        SHARED,
        &prompts(),
        &[],
        &params(),
        |_| None,
        |_, _| {},
        &stop,
    );
    assert!(
        got.iter()
            .all(|(a, e)| a.is_empty() && *e == Some(EngineError::Cancelled)),
        "{got:?}"
    );
    let after = engine.prefix_cache_stats().unwrap().computed_prompt_tokens;
    assert_eq!(after, before);
}

#[cfg(feature = "runtime-admission")]
#[test]
fn admitted_workers_preserve_prompt_order_and_retry_identity() {
    use larch_agent::runtime::inference::{Admission, AdmissionPermit};
    use std::sync::atomic::AtomicBool;
    #[derive(Default)]
    struct Tracker {
        acquired: AtomicUsize,
        settled: AtomicUsize,
    }
    struct Handle(Arc<Tracker>);
    struct OwnedPermit(Arc<Tracker>);
    impl AdmissionPermit for OwnedPermit {
        type Error = String;
        fn settle(self, _: u64) -> Result<(), String> {
            self.0.settled.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }
    impl Admission for Handle {
        type Permit = OwnedPermit;
        type Error = String;
        fn acquire(
            &self,
            root: &str,
            required: u64,
            _: &AtomicBool,
        ) -> Result<Self::Permit, String> {
            assert_eq!(root, "root");
            assert!(required >= 8);
            self.0.acquired.fetch_add(1, Ordering::SeqCst);
            Ok(OwnedPermit(self.0.clone()))
        }
    }
    let (larch, _, _) = engine();
    let expected: Vec<_> = prompts()[..2]
        .iter()
        .map(|p| alone(&larch, "", p).0)
        .collect();
    let tracker = Arc::new(Tracker::default());
    let admission = Handle(tracker.clone());
    let mut live_indices = Vec::new();
    let got = Workers {
        max_workers: 3,
        prefix_cache: false,
    }
    .run_admitted_with_live(
        &larch,
        "",
        &prompts()[..2],
        &params(),
        |_| Some("again".into()),
        &admission,
        "root",
        512,
        &AtomicBool::new(false),
        |index, _| live_indices.push(index),
    );
    assert!(live_indices.contains(&0) && live_indices.contains(&1));
    assert!(
        got.iter()
            .all(|(answers, error)| error.is_none() && answers.len() == 2)
    );
    assert_eq!(
        got.iter()
            .map(|(answers, _)| answers[0].text.clone())
            .collect::<Vec<_>>(),
        expected
    );
    assert!(
        got.iter()
            .all(|(answers, _)| answers[0].session == answers[1].session)
    );
    assert_eq!(tracker.acquired.load(Ordering::SeqCst), 4);
    assert_eq!(tracker.settled.load(Ordering::SeqCst), 4);
}

#[cfg(feature = "runtime-admission")]
#[test]
fn admitted_workers_do_not_start_after_cancellation() {
    use larch_agent::runtime::inference::SeatAdmission;
    use std::sync::atomic::AtomicBool;
    let (larch, _, _) = engine();
    let admission = SeatAdmission::new(1).unwrap();
    let stop = AtomicBool::new(true);
    let got = Workers::default().run_admitted(
        &larch,
        "",
        &prompts()[..2],
        &params(),
        |_| None,
        &admission,
        "foreground",
        512,
        &stop,
    );
    assert!(
        got.iter()
            .all(|(answers, error)| answers.is_empty() && error.as_deref() == Some("cancelled"))
    );
}

/// `params` for work nobody waits on token by token, which waits for its seats.
fn background() -> GenParams {
    GenParams {
        class: larch::api::Class::Background,
        ..params()
    }
}

#[test]
fn two_background_worker_runs_on_one_seat_never_run_side_by_side() {
    use larch_agent::runtime::inference::SeatAdmission;
    let (larch, _, most) = engine();
    let want: Vec<_> = prompts().iter().map(|p| alone(&larch, "", p)).collect();
    most.store(0, Ordering::Relaxed);
    SeatAdmission::install(&larch, 1).unwrap();
    // Each run alone would send two at once (`at_once`); two runs would batch four.
    let workers = Workers {
        max_workers: 3,
        prefix_cache: false,
    };
    let run = || firsts(workers.run(&larch, "", &prompts(), &background(), |_| None));
    let got = thread::scope(|s| {
        let other = s.spawn(run);
        (run(), other.join().unwrap())
    });
    let want: Vec<_> = want.into_iter().map(Ok).collect();
    assert_eq!((got.0, got.1), (want.clone(), want));
    assert_eq!(most.load(Ordering::Relaxed), 1, "one sequence a step");
}

#[test]
fn a_foreground_worker_group_keeps_its_size_and_never_waits_behind_background_seats() {
    use larch_agent::runtime::inference::SeatAdmission;
    use std::sync::{atomic::AtomicBool, mpsc};
    let (larch, _, most) = engine();
    let want: Vec<_> = prompts().iter().map(|p| alone(&larch, "", p)).collect();
    most.store(0, Ordering::Relaxed);
    let seats = SeatAdmission::install(&larch, 1).unwrap();
    let held = (seats.take(1, larch::api::Class::Background, &AtomicBool::default())).unwrap();
    let workers = Workers {
        max_workers: 3,
        prefix_cache: false,
    };
    let (tx, done) = mpsc::channel();
    thread::scope(|s| {
        s.spawn(|| {
            tx.send(firsts(workers.run(
                &larch,
                "",
                &prompts(),
                &params(),
                |_| None,
            )))
        });
        let got = done.recv_timeout(Duration::from_secs(60));
        drop(held);
        let want: Vec<_> = want.into_iter().map(Ok).collect();
        assert_eq!(got.expect("the group waited for seats"), want);
    });
    // `at_once` of three workers, as before seats, though the pool has one.
    assert_eq!(most.load(Ordering::Relaxed), 2);
}

#[test]
fn a_loader_sets_the_seats_a_caller_found_unless_one_is_taken() {
    use larch_agent::runtime::inference::SeatAdmission;
    use std::sync::atomic::AtomicBool;
    let (larch, _, _) = engine();
    let found = SeatAdmission::of(&larch);
    SeatAdmission::install(&larch, 3).unwrap();
    assert_eq!(found.capacity(), 3);
    let _held = (found.take(1, larch::api::Class::Foreground, &AtomicBool::default())).unwrap();
    assert!(SeatAdmission::install(&larch, 2).is_err());
    assert_eq!(found.capacity(), 3);
}

#[test]
fn each_prompt_is_a_helper_that_waits_works_and_ends_with_its_group() {
    use larch_agent::workers::{Helper, HelperState::*, telling};
    let (larch, _, _) = engine();
    let tasks: Vec<String> = (0..5).map(|i| format!("part {i}")).collect();
    let (to, heard) = std::sync::mpsc::channel();
    let stop = std::sync::atomic::AtomicBool::new(false);
    let pool = Workers {
        max_workers: 3,
        prefix_cache: false,
    };
    telling(to, 0, || {
        let none = |_: &Answer| None;
        pool.run_helpers(
            &larch,
            "",
            &prompts(),
            &tasks,
            &params(),
            none,
            |_, _| {},
            &stop,
        )
    });
    let said: Vec<(usize, _)> = heard.try_iter().map(|h: Helper| (h.id, h.state)).collect();
    // Two at a time, as the pool keeps the chat's slot.
    let want = [
        (0, Queued),
        (1, Queued),
        (2, Queued),
        (3, Queued),
        (4, Queued),
        (0, Working),
        (1, Working),
        (0, Done),
        (1, Done),
        (2, Working),
        (3, Working),
        (2, Done),
        (3, Done),
        (4, Working),
        (4, Done),
    ];
    assert_eq!(said, want);
}

#[test]
fn a_second_skill_in_a_turn_numbers_its_helpers_after_the_first() {
    use larch_agent::workers::telling;
    let (larch, _, _) = engine();
    let tasks: Vec<String> = (0..5).map(|i| format!("part {i}")).collect();
    let (to, heard) = std::sync::mpsc::channel();
    let stop = std::sync::atomic::AtomicBool::new(false);
    let pool = Workers {
        max_workers: 3,
        prefix_cache: false,
    };
    let none = |_: &Answer| None;
    // The board already holds 2 helpers of an earlier skill; this one runs its workers twice.
    telling(to, 2, || {
        for _ in 0..2 {
            let p = params();
            pool.run_helpers(&larch, "", &prompts(), &tasks, &p, none, |_, _| {}, &stop);
        }
    });
    let ids: std::collections::BTreeSet<usize> = heard.try_iter().map(|h| h.id).collect();
    assert_eq!(ids, (2..12).collect());
}
