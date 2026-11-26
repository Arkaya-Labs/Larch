//! Reload runs the production drain, fit, allocation and publication path on the CPU backend.
use std::io::Write;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::{Duration, Instant};

use larch_engine::{
    EngineEvent, EngineHandle, EngineOptions, Loader, ReloadChange, Request, RequestHandle,
    start_with,
};
use larch_kernels::{CPU, KError, KvPool, Stream, Tensor};
use larch_model::{FakeModel, Model, ModelConfig, StepBatch};
use larch_records::{
    Authority, EngineError, HealthState, KnobReply, KnobRequest, LoadError, Read, ReadFrom, Record,
    knob_id,
};

const DEADLINE: Duration = Duration::from_secs(10);

struct Held {
    inner: FakeModel,
    entered: Option<SyncSender<()>>,
    release: Receiver<()>,
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
        if let Some(entered) = self.entered.take() {
            entered.send(()).unwrap();
            self.release
                .recv_timeout(DEADLINE)
                .expect("release held forward");
        }
        self.inner.forward(s, b, kv, logits)
    }
}

fn loader(vocab: u32) -> Loader {
    Box::new(move |_, _, _| Ok(Box::new(FakeModel::new(vocab))))
}
fn options() -> EngineOptions {
    let mut opts = EngineOptions::new("fake".into(), CPU);
    opts.max_context = 512;
    // These tests time the watchdog itself, so they keep the GPU floor on the fast fake model.
    opts.step_timeout_floor = Some(Duration::from_secs(5));
    opts
}
fn finish(h: RequestHandle) -> Vec<u32> {
    let mut tokens = Vec::new();
    loop {
        match h.events.recv_timeout(DEADLINE).expect("request terminal") {
            EngineEvent::Token { id, .. } => tokens.push(id),
            EngineEvent::Done { .. } => return tokens,
            EngineEvent::Prefilled { .. } => {}
            e => panic!("request failed: {e:?}"),
        }
    }
}
fn wait_state(e: &EngineHandle, state: HealthState) {
    let until = Instant::now() + DEADLINE;
    while e.health().state != state {
        assert!(Instant::now() < until, "health did not reach {state:?}");
        std::thread::yield_now();
    }
}

// Catches closing admission too late, aborting accepted work, ring replacement, and stale models.
#[test]
fn reload_drains_and_resumes() {
    let (entered_tx, entered) = sync_channel(1);
    let (release, release_rx) = sync_channel(1);
    let e = start_with(
        options(),
        Box::new(move |_, _, _| {
            Ok(Box::new(Held {
                inner: FakeModel::new(64),
                entered: Some(entered_tx),
                release: release_rx,
            }))
        }),
    )
    .unwrap();
    let mut reader = e.ring().reader(ReadFrom::Oldest);
    let accepted = e.submit(Request::new(vec![15], 1)).unwrap();
    entered.recv_timeout(DEADLINE).unwrap();
    let queued = e.submit(Request::new(vec![3], 1)).unwrap();
    let reloader = e.clone();
    let (done_tx, done) = sync_channel(1);
    let reload = std::thread::spawn(move || {
        done_tx
            .send(reloader.reload_with(ReloadChange::default(), loader(16)))
            .unwrap();
    });
    wait_state(&e, HealthState::Draining);
    assert!(matches!(
        e.submit(Request::new(vec![1], 1)),
        Err(EngineError::NotReady(HealthState::Draining))
    ));
    assert!(matches!(
        e.reload_with(ReloadChange::default(), loader(16)),
        Err(EngineError::NotReady(HealthState::Draining))
    ));
    assert!(
        done.try_recv().is_err(),
        "reload must wait for accepted work"
    );
    release.send(()).unwrap();
    assert_eq!(finish(accepted), vec![16]);
    assert_eq!(finish(queued), vec![4]);
    done.recv_timeout(DEADLINE).unwrap().unwrap();
    reload.join().unwrap();
    assert_eq!(e.health().state, HealthState::Ready);
    assert_eq!(
        finish(e.submit(Request::new(vec![15], 1)).unwrap()),
        vec![0]
    );
    let mut states = Vec::new();
    let mut engine_ids = Vec::new();
    while let Read::Record(r) = reader.next() {
        if let Record::HealthEvent(h, event) = r {
            states.push(event.to);
            engine_ids.push(h.engine_id);
        }
    }
    assert_eq!(
        states,
        [
            HealthState::Ready.raw(),
            HealthState::Draining.raw(),
            HealthState::Loading.raw(),
            HealthState::Ready.raw()
        ]
    );
    assert!(engine_ids.iter().all(|id| *id == engine_ids[0]));
    e.shutdown();
}

// Catches retaining old KV/admission limits or resetting live knobs and revisions during rebuild.
#[test]
fn reload_drops_parked_kv_and_updates_limits() {
    let e = start_with(options(), loader(64)).unwrap();
    finish(
        e.submit(Request {
            session: 7,
            keep_kv: true,
            ..Request::new(vec![1, 2], 1)
        })
        .unwrap(),
    );
    assert_eq!(e.parked_tokens(7), 2);
    let knobs = e.knobs();
    assert!(matches!(
        knobs.set(KnobRequest::new(
            knob_id::MAX_BATCH_SEQS,
            8.0,
            Authority::Human
        )),
        KnobReply::Applied { .. }
    ));
    let rev = knobs.effective().rev;
    assert!(matches!(
        e.submit(Request::new(vec![1; 600], 1)),
        Err(EngineError::PromptTooLong { limit: 512, .. })
    ));
    e.reload_with(
        ReloadChange {
            model: Some("replacement".into()),
            max_context: Some(1024),
            kv_pool_fraction: Some(0.5),
            ..ReloadChange::default()
        },
        loader(32),
    )
    .unwrap();
    assert_eq!(e.parked_tokens(7), 0);
    let cfg = knobs.effective();
    assert!(cfg.rev > rev);
    assert_eq!(cfg.values[knob_id::MAX_CONTEXT as usize], 1024.0);
    assert_eq!(cfg.values[knob_id::KV_POOL_FRACTION as usize], 0.5);
    assert_eq!(cfg.values[knob_id::MAX_BATCH_SEQS as usize], 8.0);
    assert_eq!(e.deployment().ctx_max, 1024);
    assert_eq!(e.deployment().vocab, 32);
    assert_eq!(e.deployment().model.as_str(), "replacement");
    assert_eq!(
        finish(e.submit(Request::new(vec![1; 600], 1)).unwrap()),
        vec![2]
    );
    assert!(e.submit(Request::new(vec![32], 1)).is_err());
    let mut reader = e.ring().reader(ReadFrom::Oldest);
    let mut previous = 0;
    let mut changes = Vec::new();
    while let Read::Record(r) = reader.next() {
        if let Record::ConfigurationRevision(h, c) = r {
            assert!(h.config_rev > previous);
            previous = h.config_rev;
            changes.push((c.knob, c.effective));
        }
    }
    assert!(changes.contains(&(knob_id::MAX_CONTEXT, 1024.0)));
    assert!(changes.contains(&(knob_id::KV_POOL_FRACTION, 0.5)));
    e.shutdown();
}

// Catches bypassing the registry bounds or mutating state before validation finishes.
#[test]
fn reload_rejects_invalid_values() {
    let e = start_with(options(), loader(64)).unwrap();
    let before = e.knobs().effective();
    let health = e.health();
    for change in [
        ReloadChange {
            max_context: Some(0),
            ..Default::default()
        },
        ReloadChange {
            max_context: Some(511),
            ..Default::default()
        },
        ReloadChange {
            max_context: Some(131073),
            ..Default::default()
        },
        ReloadChange {
            kv_pool_fraction: Some(f32::NAN),
            ..Default::default()
        },
        ReloadChange {
            kv_pool_fraction: Some(f32::INFINITY),
            ..Default::default()
        },
        ReloadChange {
            kv_pool_fraction: Some(0.29),
            ..Default::default()
        },
        ReloadChange {
            kv_pool_fraction: Some(0.96),
            ..Default::default()
        },
    ] {
        assert!(e.reload_with(change, loader(16)).is_err());
        assert_eq!(e.health(), health);
        assert_eq!(*e.knobs().effective(), *before);
    }
    assert_eq!(
        finish(e.submit(Request::new(vec![15], 1)).unwrap()),
        vec![16]
    );
    e.shutdown();
}

#[test]
fn reload_rejects_bad_file_without_losing_the_old_model() {
    let e = start_with(options(), loader(64)).unwrap();
    finish(
        e.submit(Request {
            session: 9,
            keep_kv: true,
            ..Request::new(vec![1, 2], 1)
        })
        .unwrap(),
    );
    let before = e.deployment();
    let config = e.knobs().effective();
    let path = std::env::temp_dir().join(format!("larch-bad-reload-{}.gguf", std::process::id()));
    for bytes in [None, Some(b"not a GGUF".as_slice())] {
        if let Some(bytes) = bytes {
            std::fs::write(&path, bytes).unwrap();
        }
        let result = e.reload(ReloadChange {
            model: Some(path.clone()),
            max_context: Some(1024),
            ..Default::default()
        });
        assert!(
            matches!(result, Err(EngineError::InvalidRequest(_))),
            "{result:?}"
        );
        assert_eq!(e.health().state, HealthState::Ready);
        assert_eq!(e.deployment(), before);
        assert_eq!(*e.knobs().effective(), *config);
        assert_eq!(e.parked_tokens(9), 2);
        assert_eq!(
            finish(e.submit(Request::new(vec![15], 1)).unwrap()),
            vec![16]
        );
    }
    std::fs::remove_file(path).unwrap();
    e.shutdown();
}

#[test]
fn reload_longer_than_step_timeout_succeeds() {
    let e = start_with(options(), loader(64)).unwrap();
    e.reload_with(
        ReloadChange::default(),
        Box::new(|_, _, _| {
            std::thread::sleep(Duration::from_millis(5500));
            Ok(Box::new(FakeModel::new(16)))
        }),
    )
    .unwrap();
    assert_eq!(e.health().state, HealthState::Ready);
    assert_eq!(
        finish(e.submit(Request::new(vec![15], 1)).unwrap()),
        vec![0]
    );
    e.shutdown();
}

#[test]
fn shutdown_records_the_ready_to_draining_edge() {
    let e = start_with(options(), loader(64)).unwrap();
    let mut reader = e.ring().reader(ReadFrom::Latest);
    e.shutdown();
    let mut edges = Vec::new();
    while let Read::Record(record) = reader.next() {
        if let Record::HealthEvent(_, event) = record {
            edges.push((event.from, event.to));
        }
    }
    assert_eq!(
        edges,
        [(HealthState::Ready.raw(), HealthState::Draining.raw())]
    );
}

// Catches publishing Ready after a failed rebuild or allowing another in-process rebuild.
#[test]
fn reload_failure_never_publishes_ready() {
    let e = start_with(options(), loader(64)).unwrap();
    let mut reader = e.ring().reader(ReadFrom::Latest);
    let result = e.reload_with(
        ReloadChange::default(),
        Box::new(|_, _, _| Err(LoadError::DoesNotFit { need: 2, have: 1 })),
    );
    assert!(result.is_err());
    assert_eq!(e.health().state, HealthState::Failed);
    assert!(matches!(
        e.submit(Request::new(vec![1], 1)),
        Err(EngineError::NotReady(HealthState::Failed))
    ));
    assert!(e.reload_with(ReloadChange::default(), loader(16)).is_err());
    let mut states = Vec::new();
    while let Read::Record(r) = reader.next() {
        if let Record::HealthEvent(_, event) = r {
            states.push(event.to);
        }
    }
    assert_eq!(
        states,
        [
            HealthState::Draining.raw(),
            HealthState::Loading.raw(),
            HealthState::Failed.raw()
        ]
    );
    e.clone().shutdown();
    assert_eq!(e.health().state, HealthState::Failed);
}

// Catches shrinking the allocation ceiling when a live prefill value was temporarily reduced.
#[test]
fn reload_preserves_live_knob_revert_capacity() {
    let e = start_with(options(), loader(64)).unwrap();
    let knobs = e.knobs();
    let KnobReply::Applied { rev, .. } = knobs.set(KnobRequest::new(
        knob_id::PREFILL_CHUNK_TOKENS,
        256.0,
        Authority::Human,
    )) else {
        panic!("prefill change refused");
    };
    e.reload_with(ReloadChange::default(), loader(64)).unwrap();
    assert_eq!(
        knobs.effective().values[knob_id::PREFILL_CHUNK_TOKENS as usize],
        256.0
    );
    assert!(matches!(
        knobs.set(KnobRequest::new(
            knob_id::PREFILL_CHUNK_TOKENS,
            512.0,
            Authority::Human
        )),
        KnobReply::Applied { .. }
    ));
    assert!(knobs.effective().rev > rev);
    e.shutdown();
}

// Catches publishing Ready after shutdown has closed admission during a loader callback.
#[test]
fn reload_shutdown_during_loading_never_resumes() {
    let e = start_with(options(), loader(64)).unwrap();
    let (entered_tx, entered) = sync_channel(1);
    let (release, release_rx) = sync_channel(1);
    let reloader = e.clone();
    let reload = std::thread::spawn(move || {
        reloader.reload_with(
            ReloadChange::default(),
            Box::new(move |_, _, _| {
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(DEADLINE).unwrap();
                Ok(Box::new(FakeModel::new(16)))
            }),
        )
    });
    entered.recv_timeout(DEADLINE).unwrap();
    assert_eq!(e.health().state, HealthState::Loading);
    assert!(matches!(
        e.reload_with(ReloadChange::default(), loader(16)),
        Err(EngineError::NotReady(HealthState::Loading))
    ));
    let mut reader = e.ring().reader(ReadFrom::Latest);
    let closer = e.clone();
    let shutdown = std::thread::spawn(move || closer.shutdown());
    wait_state(&e, HealthState::Draining);
    release.send(()).unwrap();
    assert!(reload.join().unwrap().is_err());
    shutdown.join().unwrap();
    assert_eq!(e.health().state, HealthState::Draining);
    while let Read::Record(r) = reader.next() {
        if let Record::HealthEvent(_, event) = r {
            assert_ne!(event.to, HealthState::Ready.raw());
            assert_eq!(
                (event.from, event.to),
                (HealthState::Loading.raw(), HealthState::Draining.raw())
            );
        }
    }
}

// Catches retaining a consumed startup loader instead of recreating it from the remembered path.
#[test]
fn reload_reopens_current_model_path() {
    let path = std::env::temp_dir().join(format!("larch-reload-{}.gguf", std::process::id()));
    std::fs::write(&path, larch_model::tiny::gguf(larch_model::Arch::Llama, 3)).unwrap();
    let e = start_with(options(), loader(64)).unwrap();
    e.reload(ReloadChange {
        model: Some(path.clone()),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(e.deployment().vocab, 256);
    let first = finish(e.submit(Request::new(vec![1, 2], 2)).unwrap());
    e.reload(ReloadChange::default()).unwrap();
    assert_eq!(
        finish(e.submit(Request::new(vec![1, 2], 2)).unwrap()),
        first
    );
    e.shutdown();
    std::fs::remove_file(path).unwrap();
}

// Catches resetting the scheduler's sticky release tombstones while discarding parked KV.
#[test]
fn reload_keeps_released_sessions_released() {
    let e = start_with(options(), loader(64)).unwrap();
    e.release_session(9);
    assert_eq!(e.parked_tokens(9), 0);
    e.reload_with(ReloadChange::default(), loader(64)).unwrap();
    finish(
        e.submit(Request {
            session: 9,
            keep_kv: true,
            ..Request::new(vec![1, 2], 1)
        })
        .unwrap(),
    );
    assert_eq!(
        e.parked_tokens(9),
        0,
        "reload must not resurrect a released session"
    );
    e.shutdown();
}

fn assert_failed(e: &EngineHandle, fault: larch_records::FaultCode) {
    assert_eq!(e.health().state, HealthState::Failed);
    assert_eq!(e.health().last_fault, fault);
    let mut reader = e.ring().reader(ReadFrom::Oldest);
    let mut events = Vec::new();
    while let Read::Record(r) = reader.next() {
        if let Record::HealthEvent(_, event) = r {
            events.push(event);
        }
    }
    let last = events.last().expect("failed health event");
    assert_eq!(
        (last.to, last.fault),
        (HealthState::Failed.raw(), fault.raw())
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| e.to == HealthState::Ready.raw())
            .count(),
        1
    );
}

// Catches treating LoadError's Backend tag as the contained device status, including shutdown.
#[test]
fn reload_backend_faults_keep_their_identity() {
    use larch_records::FaultCode;
    for (failure, error, fault) in [
        (
            LoadError::Backend(KError::DeviceLost.code()),
            EngineError::DeviceLost,
            FaultCode::DeviceLost,
        ),
        (
            LoadError::Backend(KError::Timeout.code()),
            EngineError::DeviceLost,
            FaultCode::DeviceLost,
        ),
        (
            LoadError::Backend(KError::Oom.code()),
            EngineError::Internal(2),
            FaultCode::Oom,
        ),
        (
            LoadError::Backend(KError::Launch.code()),
            EngineError::Internal(4),
            FaultCode::LaunchFailure,
        ),
        (
            LoadError::Backend(99),
            EngineError::Internal(99),
            FaultCode::LaunchFailure,
        ),
        (
            LoadError::Io,
            EngineError::Internal(larch_engine::invalid::RELOAD_MODEL | 1),
            FaultCode::LoadError,
        ),
        (
            LoadError::DoesNotFit { need: 2, have: 1 },
            EngineError::Internal(larch_engine::invalid::RELOAD_MODEL | 5),
            FaultCode::LoadError,
        ),
    ] {
        for overlap_shutdown in [false, true] {
            let e = start_with(options(), loader(64)).unwrap();
            let (entered_tx, entered) = sync_channel(1);
            let (release, release_rx) = sync_channel(1);
            let reloader = e.clone();
            let reload = std::thread::spawn(move || {
                let result = reloader.reload_with(
                    ReloadChange::default(),
                    Box::new(move |_, _, _| {
                        entered_tx.send(()).unwrap();
                        release_rx.recv_timeout(DEADLINE).unwrap();
                        Err(failure)
                    }),
                );
                (result, reloader.health())
            });
            entered.recv_timeout(DEADLINE).unwrap();
            let shutdown = overlap_shutdown.then(|| {
                let closer = e.clone();
                std::thread::spawn(move || closer.shutdown())
            });
            if overlap_shutdown {
                wait_state(&e, HealthState::Draining);
            }
            release.send(()).unwrap();
            let (result, health) = reload.join().unwrap();
            assert_eq!(
                result,
                Err(error),
                "{failure:?}, shutdown={overlap_shutdown}"
            );
            assert_eq!(
                (health.state, health.last_fault),
                (HealthState::Failed, fault)
            );
            assert_failed(&e, fault);
            if let Some(shutdown) = shutdown {
                shutdown.join().unwrap();
            }
            e.clone().shutdown();
            assert_failed(&e, fault);
        }
    }
}

struct DropPanic {
    inner: FakeModel,
    entered: SyncSender<()>,
    release: Receiver<()>,
}
impl Model for DropPanic {
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
        self.inner.forward(s, b, kv, logits)
    }
}
impl Drop for DropPanic {
    fn drop(&mut self) {
        self.entered.send(()).unwrap();
        self.release
            .recv_timeout(DEADLINE)
            .expect("release old model destructor");
        panic!("old model destructor failed");
    }
}

// Catches dropping the reload reply during unwind before the panic fault is published.
#[test]
fn reload_old_model_drop_panic_publishes_failed_before_reply() {
    use larch_records::FaultCode;
    for overlap_shutdown in [false, true] {
        let (entered_tx, entered) = sync_channel(1);
        let (release, release_rx) = sync_channel(1);
        let e = start_with(
            options(),
            Box::new(move |_, _, _| {
                Ok(Box::new(DropPanic {
                    inner: FakeModel::new(64),
                    entered: entered_tx,
                    release: release_rx,
                }))
            }),
        )
        .unwrap();
        let reloader = e.clone();
        let (done_tx, done) = sync_channel(1);
        let reload = std::thread::spawn(move || {
            let result = reloader.reload_with(ReloadChange::default(), loader(16));
            done_tx.send((result, reloader.health())).unwrap();
        });
        entered.recv_timeout(DEADLINE).unwrap();
        assert_eq!(e.health().state, HealthState::Loading);
        assert!(
            done.try_recv().is_err(),
            "reload returned before old-model teardown"
        );
        let shutdown = overlap_shutdown.then(|| {
            let closer = e.clone();
            std::thread::spawn(move || closer.shutdown())
        });
        if overlap_shutdown {
            wait_state(&e, HealthState::Draining);
        }
        release.send(()).unwrap();
        let (result, health) = done.recv_timeout(DEADLINE).unwrap();
        reload.join().unwrap();
        assert_eq!(
            result,
            Err(EngineError::Internal(0)),
            "shutdown={overlap_shutdown}"
        );
        assert_eq!(
            (health.state, health.last_fault),
            (HealthState::Failed, FaultCode::Panic)
        );
        assert_failed(&e, FaultCode::Panic);
        if let Some(shutdown) = shutdown {
            shutdown.join().unwrap();
        }
        e.clone().shutdown();
        assert_failed(&e, FaultCode::Panic);
    }
}

// Catches a watchdog that cannot terminate work while the executor is held, and late double ends.
#[test]
fn watchdog_fails_pending_work() {
    let (entered_tx, entered) = sync_channel(1);
    let (release, release_rx) = sync_channel(1);
    let e = start_with(
        options(),
        Box::new(move |_, _, _| {
            Ok(Box::new(Held {
                inner: FakeModel::new(64),
                entered: Some(entered_tx),
                release: release_rx,
            }))
        }),
    )
    .unwrap();
    let active = e.submit(Request::new(vec![1], 1)).unwrap();
    entered.recv_timeout(DEADLINE).unwrap();
    let queued = e.submit(Request::new(vec![2], 1)).unwrap();
    let knobs = e.knobs();
    let (knob_tx, knob_rx) = sync_channel(1);
    let knob = std::thread::spawn(move || {
        knob_tx
            .send(knobs.set(KnobRequest {
                timeout_ms: 10_000,
                ..KnobRequest::new(knob_id::MAX_BATCH_SEQS, 8.0, Authority::Human)
            }))
            .unwrap();
    });
    let result = active.events.recv_timeout(Duration::from_secs(7));
    if result.is_err() {
        let _ = release.send(());
    }
    assert_eq!(result.unwrap(), EngineEvent::Error(EngineError::DeviceLost));
    assert_eq!(
        queued.events.recv_timeout(DEADLINE).unwrap(),
        EngineEvent::Error(EngineError::NotReady(HealthState::Failed))
    );
    assert_eq!(
        knob_rx.recv_timeout(Duration::from_millis(100)).unwrap(),
        KnobReply::Rejected {
            reason: larch_records::RejectReason::NotReady
        }
    );
    knob.join().unwrap();
    assert_failed(&e, larch_records::FaultCode::Timeout);
    let control = e.clone();
    let (done_tx, done) = sync_channel(1);
    let controls = std::thread::spawn(move || {
        assert_eq!(control.parked_tokens(1), 0);
        assert!(control.finish_stop(&active).is_err());
        assert!(
            control
                .reload_with(ReloadChange::default(), loader(16))
                .is_err()
        );
        assert!(control.set_fault_flush(|_, _| {}).is_err());
        assert!(matches!(
            control
                .knobs()
                .ab(larch_records::AbCommand::Stop { trial_id: 1 }),
            larch_records::AbReply::Rejected {
                reason: larch_records::RejectReason::NotReady
            }
        ));
        control.shutdown();
        done_tx.send((active, queued)).unwrap();
    });
    let bounded = done.recv_timeout(Duration::from_millis(100));
    // Release even on failure, so the parent never leaves an uncontrolled hung executor.
    let _ = release.send(());
    controls.join().unwrap();
    let (active, queued) = bounded.expect("public calls must return while executor is held");
    std::thread::sleep(Duration::from_millis(30));
    assert!(active.events.try_recv().is_err());
    assert!(queued.events.try_recv().is_err());
    assert_failed(&e, larch_records::FaultCode::Timeout);
}

struct Faulting {
    inner: FakeModel,
    failure: Option<KError>,
    drop_wait: Option<(SyncSender<()>, Receiver<()>)>,
}
impl Model for Faulting {
    fn config(&self) -> &ModelConfig {
        self.inner.config()
    }
    fn forward(&mut self, _: &Stream, _: &StepBatch, _: &KvPool, _: &Tensor) -> Result<(), KError> {
        match self.failure {
            Some(error) => Err(error),
            None => panic!("controlled step panic"),
        }
    }
}
impl Drop for Faulting {
    fn drop(&mut self) {
        if let Some((entered, release)) = self.drop_wait.take() {
            let _ = entered.send(());
            let _ = release.recv_timeout(DEADLINE);
        }
    }
}

#[test]
fn stream_timeout_maps_to_device_lost() {
    use larch_records::FaultCode;
    for (failure, error, fault) in [
        (
            Some(KError::Timeout),
            EngineError::DeviceLost,
            FaultCode::DeviceLost,
        ),
        (
            Some(KError::DeviceLost),
            EngineError::DeviceLost,
            FaultCode::DeviceLost,
        ),
        (Some(KError::Oom), EngineError::Internal(2), FaultCode::Oom),
        (None, EngineError::Internal(0), FaultCode::Panic),
    ] {
        let e = start_with(
            options(),
            Box::new(move |_, _, _| {
                Ok(Box::new(Faulting {
                    inner: FakeModel::new(64),
                    failure,
                    drop_wait: None,
                }))
            }),
        )
        .unwrap();
        let request = e.submit(Request::new(vec![1], 1)).unwrap();
        assert_eq!(
            request.events.recv_timeout(DEADLINE).unwrap(),
            EngineEvent::Error(error)
        );
        assert_failed(&e, fault);
        e.shutdown();
        assert!(request.events.try_recv().is_err());
    }
}

#[test]
fn fault_does_not_wait_for_device_drop() {
    let (entered_tx, entered) = sync_channel(1);
    let (release, release_rx) = sync_channel(1);
    let e = start_with(
        options(),
        Box::new(move |_, _, _| {
            Ok(Box::new(Faulting {
                inner: FakeModel::new(64),
                failure: Some(KError::DeviceLost),
                drop_wait: Some((entered_tx, release_rx)),
            }))
        }),
    )
    .unwrap();
    let request = e.submit(Request::new(vec![1], 1)).unwrap();
    assert_eq!(
        request.events.recv_timeout(DEADLINE).unwrap(),
        EngineEvent::Error(EngineError::DeviceLost)
    );
    let (done_tx, done) = sync_channel(1);
    let close = std::thread::spawn(move || {
        e.shutdown();
        let _ = done_tx.send(());
    });
    let result = done.recv_timeout(Duration::from_millis(100));
    // Do not leave a hung destructor on RED.
    let _ = release.send(());
    close.join().unwrap();
    assert!(
        result.is_ok(),
        "faulted shutdown waited for device destruction"
    );
    assert!(
        entered.try_recv().is_err(),
        "faulted resources must be retained"
    );
}

fn child_run(name: &str, mode: &str, restart: Option<String>) -> std::process::Output {
    use std::process::{Command, Stdio};
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", name, "--nocapture"])
        .env("LARCH_RECOVERY_CHILD", mode)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(value) = restart {
        command.env("LARCH_RESTARTED_AT", value);
    }
    let child = command.spawn().unwrap();
    let pid = child.id().to_string();
    let (done, deadline) = sync_channel(1);
    let killer = std::thread::spawn(move || {
        if deadline.recv_timeout(DEADLINE).is_ok() {
            return false;
        }
        #[cfg(windows)]
        let _ = Command::new("taskkill").args(["/F", "/PID", &pid]).output();
        #[cfg(unix)]
        let _ = Command::new("kill").args(["-KILL", &pid]).output();
        true
    });
    let output = child.wait_with_output().unwrap();
    let _ = done.send(());
    assert!(
        !killer.join().unwrap(),
        "child deadline: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
fn recovery_timestamp_is_measured() {
    if std::env::var("LARCH_RECOVERY_CHILD").as_deref() == Ok("timestamp") {
        let e = start_with(options(), loader(64)).unwrap();
        let mut reader = e.ring().reader(ReadFrom::Oldest);
        let mut measured = None;
        while let Read::Record(record) = reader.next() {
            if let Record::HealthEvent(_, event) = record {
                measured = Some(event.recovery_ms);
            }
        }
        let expected = std::env::var("LARCH_RESTARTED_AT").unwrap();
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let valid = expected.parse::<u128>().ok().filter(|&v| v > 0 && v <= ns);
        if valid.is_some() {
            assert!(measured.unwrap() >= 200);
        } else {
            assert_eq!(measured, Some(0));
        }
        e.shutdown();
        return;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    for restart in [
        (now - 250_000_000).to_string(),
        (now + 60_000_000_000).to_string(),
        "0".into(),
        "bad".into(),
        "-1".into(),
        u128::MAX.to_string(),
    ] {
        let output = child_run("recovery_timestamp_is_measured", "timestamp", Some(restart));
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

#[test]
fn fault_flush_sees_final_records() {
    let e = start_with(
        options(),
        Box::new(|_, _, _| {
            Ok(Box::new(Faulting {
                inner: FakeModel::new(64),
                failure: Some(KError::DeviceLost),
                drop_wait: None,
            }))
        }),
    )
    .unwrap();
    let (tx, rx) = sync_channel(1);
    e.set_fault_flush(move |ring, deadline| {
        let mut records = ring.reader(ReadFrom::Oldest);
        let (mut failed, mut spans) = (0, 0);
        while let Read::Record(record) = records.next() {
            match record {
                Record::HealthEvent(_, event) if event.to == HealthState::Failed.raw() => {
                    failed += 1
                }
                Record::RequestSpan(_, _) => spans += 1,
                _ => {}
            }
        }
        tx.send((failed, spans, deadline > Instant::now())).unwrap();
    })
    .unwrap();
    let request = e.submit(Request::new(vec![1], 1)).unwrap();
    assert_eq!(
        request.events.recv_timeout(DEADLINE).unwrap(),
        EngineEvent::Error(EngineError::DeviceLost)
    );
    let records = rx.recv_timeout(Duration::from_secs(1));
    e.shutdown();
    assert_eq!(records.unwrap(), (1, 1, true));
}

#[test]
fn fault_flush_is_bounded() {
    let mode = std::env::var("LARCH_RECOVERY_CHILD").unwrap_or_default();
    if mode == "flush" || mode == "panic_flush" {
        let mut opts = options();
        opts.on_fault = larch_engine::OnFault::Exit;
        let (drop_tx, _drop_rx) = sync_channel(1);
        let (_release, drop_release) = sync_channel(1);
        let e = start_with(
            opts,
            Box::new(move |_, _, _| {
                Ok(Box::new(Faulting {
                    inner: FakeModel::new(64),
                    failure: Some(KError::DeviceLost),
                    drop_wait: Some((drop_tx, drop_release)),
                }))
            }),
        )
        .unwrap();
        let (_release, held) = sync_channel::<()>(1);
        e.set_fault_flush(move |_, _| {
            eprintln!("flush_entered");
            assert_ne!(mode, "panic_flush", "controlled flush panic");
            // Held until the process ends, with a line each 1 ms: the last says how long it ran.
            // Busy, as a timed wait oversleeps where macOS throttles the process (a background
            // job, as hosted CI runs tests), and the lines would show the throttle, not the hook.
            let (until, mut next) = (Instant::now() + DEADLINE, Instant::now());
            while Instant::now() < until && held.try_recv().is_err() {
                if Instant::now() >= next {
                    next += Duration::from_millis(1);
                    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH);
                    eprintln!("hook_ran_ns={}", now.unwrap().as_nanos());
                }
                std::thread::yield_now();
            }
        })
        .unwrap();
        let _request = e.submit(Request::new(vec![1], 1)).unwrap();
        std::thread::sleep(DEADLINE);
        panic!("fault did not exit");
    }
    for mode in ["flush", "panic_flush"] {
        let output = child_run("fault_flush_is_bounded", mode, None);
        let exited_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        assert_eq!(output.status.code(), Some(3));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("flush_entered"), "{stderr}");
        assert!(stderr.contains("fault=DeviceLost"), "{stderr}");
        let detected = stderr
            .lines()
            .find_map(|line| {
                line.split_once("detected_ns=")
                    .and_then(|(_, ns)| ns.parse::<u128>().ok())
            })
            .expect("fault detection timestamp");
        assert!(exited_ns >= detected, "exit before detection");
        let gap = exited_ns - detected;
        let ran: Vec<u128> = (stderr.lines())
            .filter_map(|line| line.strip_prefix("hook_ran_ns=")?.parse().ok())
            .collect();
        let last = ran.last().map_or(0, |at| at.saturating_sub(detected));
        let pause = ran.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
        // Past libtest's capture, so each CI log carries the measured exit cost (the hook budget).
        let _ = writeln!(
            std::io::stderr(),
            "{mode} exit gap {gap} ns; the held hook last ran {last} ns after detection, its longest pause {pause} ns"
        );
        assert!(gap <= 100_000_000, "exit gap {gap} ns");
    }
}

/// 9.11, D2: a CLI caller that ends its process once its run fails (`larch parity`) still exits 3.
#[test]
fn shutdown_after_a_fault_leaves_the_exit_to_the_engine() {
    if std::env::var("LARCH_RECOVERY_CHILD").as_deref() == Ok("caller_exit") {
        let mut opts = options();
        opts.on_fault = larch_engine::OnFault::Exit;
        let e = start_with(
            opts,
            Box::new(|_, _, _| {
                Ok(Box::new(Faulting {
                    inner: FakeModel::new(64),
                    failure: Some(KError::DeviceLost),
                    drop_wait: None,
                }))
            }),
        )
        .unwrap();
        // The hook holds its whole budget, so the caller hears its error before the exit.
        let (_release, held) = sync_channel::<()>(1);
        e.set_fault_flush(move |_, _| {
            let _ = held.recv_timeout(DEADLINE);
        })
        .unwrap();
        let request = e.submit(Request::new(vec![1], 1)).unwrap();
        let event = request.events.recv_timeout(DEADLINE);
        assert_eq!(event, Ok(EngineEvent::Error(EngineError::DeviceLost)));
        e.shutdown();
        std::process::exit(1);
    }
    let output = child_run(
        "shutdown_after_a_fault_leaves_the_exit_to_the_engine",
        "caller_exit",
        None,
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(3), "{stderr}");
}

#[test]
fn watchdog_ignores_idle() {
    let e = start_with(options(), loader(64)).unwrap();
    std::thread::sleep(Duration::from_millis(5100));
    assert_eq!(e.health().state, HealthState::Ready);
    assert_eq!(finish(e.submit(Request::new(vec![1], 1)).unwrap()), vec![2]);
    e.shutdown();
}

#[test]
fn watchdog_exit_does_not_wait_for_engine() {
    if std::env::var("LARCH_RECOVERY_CHILD").as_deref() == Ok("watchdog_exit") {
        let mut opts = options();
        opts.on_fault = larch_engine::OnFault::Exit;
        let (entered, _entry) = sync_channel(1);
        let (_release, held) = sync_channel::<()>(1);
        let e = start_with(
            opts,
            Box::new(move |_, _, _| {
                Ok(Box::new(Held {
                    inner: FakeModel::new(64),
                    entered: Some(entered),
                    release: held,
                }))
            }),
        )
        .unwrap();
        let _request = e.submit(Request::new(vec![1], 1)).unwrap();
        std::thread::sleep(DEADLINE);
        panic!("watchdog did not exit");
    }
    let output = child_run(
        "watchdog_exit_does_not_wait_for_engine",
        "watchdog_exit",
        None,
    );
    let exited_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    assert_eq!(output.status.code(), Some(3));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("fault=Timeout"), "{stderr}");
    let detected = stderr
        .lines()
        .find_map(|line| {
            line.split_once("detected_ns=")
                .and_then(|(_, ns)| ns.parse::<u128>().ok())
        })
        .expect("watchdog detection timestamp");
    assert!(
        exited_ns - detected <= 100_000_000,
        "exit gap {} ns",
        exited_ns - detected
    );
}

#[test]
fn watchdog_closes_forced_positions_while_executor_is_held() {
    let (entered_tx, entered) = sync_channel(1);
    let (release, held) = sync_channel(1);
    let e = start_with(
        options(),
        Box::new(move |_, _, _| {
            Ok(Box::new(Held {
                inner: FakeModel::new(64),
                entered: Some(entered_tx),
                release: held,
            }))
        }),
    )
    .unwrap();
    let forced = e
        .force(larch_engine::Forced {
            tokens: vec![1, 2],
            prompt: 2,
            session: 0,
            keep_kv: false,
        })
        .unwrap();
    entered.recv_timeout(DEADLINE).unwrap();
    let terminal = forced.request.events.recv_timeout(Duration::from_secs(7));
    let closed = forced.positions.recv_timeout(Duration::from_millis(100));
    release.send(()).unwrap();
    assert_eq!(
        terminal.unwrap(),
        EngineEvent::Error(EngineError::DeviceLost)
    );
    assert_eq!(closed, Err(std::sync::mpsc::RecvTimeoutError::Disconnected));
    e.shutdown();
}

struct HeldFault {
    inner: Faulting,
    entered: SyncSender<()>,
    release: Receiver<()>,
}
impl Model for HeldFault {
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
        self.entered.send(()).unwrap();
        self.release.recv_timeout(DEADLINE).unwrap();
        self.inner.forward(s, b, kv, logits)
    }
}

struct HeldCaptureDrop {
    entered: SyncSender<()>,
    release: Receiver<()>,
}
impl Drop for HeldCaptureDrop {
    fn drop(&mut self) {
        let _ = self.entered.send(());
        let _ = self.release.recv_timeout(DEADLINE);
    }
}

// Catches arbitrary replaced-hook cleanup running inside the fault publication boundary.
#[test]
fn replacing_fault_flush_drops_captures_outside_publication_locks() {
    let (forward_tx, forward_entered) = sync_channel(1);
    let (forward_release, forward_rx) = sync_channel(1);
    let e = start_with(
        options(),
        Box::new(move |_, _, _| {
            Ok(Box::new(HeldFault {
                inner: Faulting {
                    inner: FakeModel::new(64),
                    failure: Some(KError::DeviceLost),
                    drop_wait: None,
                },
                entered: forward_tx,
                release: forward_rx,
            }))
        }),
    )
    .unwrap();
    let (drop_tx, drop_entered) = sync_channel(1);
    let (drop_release, drop_rx) = sync_channel(1);
    let capture = HeldCaptureDrop {
        entered: drop_tx,
        release: drop_rx,
    };
    e.set_fault_flush(move |_, _| drop(capture)).unwrap();
    let request = e.submit(Request::new(vec![1], 1)).unwrap();
    forward_entered.recv_timeout(DEADLINE).unwrap();
    let setter = e.clone();
    let replacing = std::thread::spawn(move || setter.set_fault_flush(|_, _| {}));
    drop_entered.recv_timeout(DEADLINE).unwrap();
    forward_release.send(()).unwrap();
    let terminal = request.events.recv_timeout(Duration::from_millis(200));
    let observer = e.clone();
    let (health_tx, health_rx) = sync_channel(1);
    let observing = std::thread::spawn(move || {
        health_tx.send(observer.health()).unwrap();
    });
    let health = health_rx.recv_timeout(Duration::from_millis(200));
    // Always release the destructor before asserting, including on the old implementation.
    drop_release.send(()).unwrap();
    replacing.join().unwrap().unwrap();
    observing.join().unwrap();
    assert_eq!(
        terminal.unwrap(),
        EngineEvent::Error(EngineError::DeviceLost)
    );
    assert_eq!(health.unwrap().state, HealthState::Failed);
    e.shutdown();
}

// Catches a receiver retained by surviving handles after its engine consumer has joined.
#[test]
fn completed_shutdown_closes_surviving_handle_calls() {
    use larch_records::{AbCommand, AbReply, RejectReason, RevertTarget};
    let mode = std::env::var("LARCH_RECOVERY_CHILD").unwrap_or_default();
    if mode == "parked_after_shutdown" || mode == "stop_after_shutdown" {
        let e = start_with(options(), loader(64)).unwrap();
        let knobs = e.knobs();
        let request = e.submit(Request::new(vec![1], 1)).unwrap();
        assert!(matches!(
            request
                .events
                .iter()
                .find(|e| !matches!(e, EngineEvent::Prefilled { .. })),
            Some(EngineEvent::Token { .. })
        ));
        e.clone().shutdown();
        assert_eq!(e.health().state, HealthState::Draining);
        assert_eq!(
            knobs.set(KnobRequest::new(
                knob_id::MAX_BATCH_SEQS,
                8.0,
                Authority::Human
            )),
            KnobReply::Rejected {
                reason: RejectReason::NotReady
            }
        );
        assert_eq!(
            knobs.revert(RevertTarget::Trial(1), Authority::Human),
            KnobReply::Rejected {
                reason: RejectReason::NotReady
            }
        );
        assert_eq!(
            knobs.ab(AbCommand::Stop { trial_id: 1 }),
            AbReply::Rejected {
                reason: RejectReason::NotReady
            }
        );
        assert!(matches!(
            e.reload_with(ReloadChange::default(), loader(16)),
            Err(EngineError::NotReady(HealthState::Draining))
        ));
        let observer = e.clone();
        let (done_tx, done) = sync_channel(1);
        let caller = std::thread::spawn(move || {
            if mode == "parked_after_shutdown" {
                assert_eq!(observer.parked_tokens(1), 0);
            } else {
                assert_eq!(
                    observer.finish_stop(&request).unwrap().0.completion_tokens,
                    1
                );
            }
            done_tx.send(()).unwrap();
        });
        done.recv_timeout(Duration::from_millis(200))
            .expect("call after completed shutdown");
        caller.join().unwrap();
        e.shutdown();
        return;
    }
    let results: Vec<_> = ["parked_after_shutdown", "stop_after_shutdown"]
        .into_iter()
        .map(|mode| {
            (
                mode,
                child_run(
                    "completed_shutdown_closes_surviving_handle_calls",
                    mode,
                    None,
                ),
            )
        })
        .collect();
    let failures: Vec<_> = results
        .into_iter()
        .filter(|(_, output)| !output.status.success())
        .map(|(mode, output)| format!("{mode}: {}", String::from_utf8_lossy(&output.stderr)))
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

struct DropHeldModel {
    inner: FakeModel,
    _drop_hold: HeldCaptureDrop,
}
impl Model for DropHeldModel {
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
        self.inner.forward(s, b, kv, logits)
    }
}

// The controlled healthy destructor locates the gap after the final host drain, before join.
#[test]
fn shutdown_disconnects_calls_arriving_after_final_drain() {
    if std::env::var("LARCH_RECOVERY_CHILD").as_deref() == Ok("after_final_drain") {
        let (entered_tx, entered) = sync_channel(1);
        let (release, release_rx) = sync_channel(1);
        let e = start_with(
            options(),
            Box::new(move |_, _, _| {
                Ok(Box::new(DropHeldModel {
                    inner: FakeModel::new(64),
                    _drop_hold: HeldCaptureDrop {
                        entered: entered_tx,
                        release: release_rx,
                    },
                }))
            }),
        )
        .unwrap();
        let closer = e.clone();
        let shutdown = std::thread::spawn(move || closer.shutdown());
        entered.recv_timeout(DEADLINE).unwrap();
        let observer = e.clone();
        let (done_tx, done) = sync_channel(1);
        let parked = std::thread::spawn(move || {
            done_tx.send(observer.parked_tokens(1)).unwrap();
        });
        let result = done.recv_timeout(Duration::from_millis(200));
        release.send(()).unwrap();
        shutdown.join().unwrap();
        assert_eq!(result.unwrap(), 0);
        parked.join().unwrap();
        return;
    }
    let output = child_run(
        "shutdown_disconnects_calls_arriving_after_final_drain",
        "after_final_drain",
        None,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn reload_draining_keeps_intake_available() {
    let (entered_tx, entered) = sync_channel(1);
    let (release, held) = sync_channel(1);
    let e = start_with(
        options(),
        Box::new(move |_, _, _| {
            Ok(Box::new(Held {
                inner: FakeModel::new(64),
                entered: Some(entered_tx),
                release: held,
            }))
        }),
    )
    .unwrap();
    let request = e.submit(Request::new(vec![1], 1)).unwrap();
    entered.recv_timeout(DEADLINE).unwrap();
    let reloader = e.clone();
    let reload =
        std::thread::spawn(move || reloader.reload_with(ReloadChange::default(), loader(16)));
    wait_state(&e, HealthState::Draining);
    let observer = e.clone();
    let (started_tx, started) = sync_channel(1);
    let (done_tx, done) = sync_channel(1);
    let parked = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        done_tx.send(observer.parked_tokens(1)).unwrap();
    });
    started.recv_timeout(DEADLINE).unwrap();
    let early = done.recv_timeout(Duration::from_millis(50));
    release.send(()).unwrap();
    assert_eq!(early, Err(std::sync::mpsc::RecvTimeoutError::Timeout));
    assert_eq!(finish(request), vec![2]);
    reload.join().unwrap().unwrap();
    assert_eq!(done.recv_timeout(DEADLINE).unwrap(), 0);
    parked.join().unwrap();
    assert_eq!(e.health().state, HealthState::Ready);
    e.shutdown();
}

#[test]
fn shutdown_of_pending_work_keeps_watchdog_active() {
    let (entered_tx, entered) = sync_channel(1);
    let (release, held) = sync_channel(1);
    let e = start_with(
        options(),
        Box::new(move |_, _, _| {
            Ok(Box::new(Held {
                inner: FakeModel::new(64),
                entered: Some(entered_tx),
                release: held,
            }))
        }),
    )
    .unwrap();
    let request = e.submit(Request::new(vec![1], 1)).unwrap();
    entered.recv_timeout(DEADLINE).unwrap();
    let closer = e.clone();
    let (done_tx, done) = sync_channel(1);
    let shutdown = std::thread::spawn(move || {
        closer.shutdown();
        done_tx.send(()).unwrap();
    });
    wait_state(&e, HealthState::Draining);
    let terminal = request.events.recv_timeout(Duration::from_secs(7));
    let closed = done.recv_timeout(Duration::from_millis(200));
    release.send(()).unwrap();
    shutdown.join().unwrap();
    assert_eq!(
        terminal.unwrap(),
        EngineEvent::Error(EngineError::DeviceLost)
    );
    assert_eq!(closed, Ok(()));
    assert_failed(&e, larch_records::FaultCode::Timeout);
}
