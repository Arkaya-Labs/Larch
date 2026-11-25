//! Environment triggers use isolated CPU children; every child has a parent kill/reap deadline.
use std::process::{Command, Output, Stdio};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use larch_engine::{
    EngineEvent, EngineHandle, EngineOptions, Forced, OnFault, ReloadChange, Request,
    RequestHandle, start_with,
};
use larch_kernels::{CPU, KError, KvPool, Stream, Tensor};
use larch_model::{FakeModel, Model, ModelConfig, StepBatch};
use larch_records::{
    EngineError, FaultCode, HealthState, LoadError, Read, ReadFrom, Record, RingHandle, knob_id,
};

const DEADLINE: Duration = Duration::from_secs(9);
// Integration-test dependencies are not compiled with cfg(test).
const ACTIVE: bool = cfg!(debug_assertions) || cfg!(feature = "fault-inject");

struct Controlled {
    inner: FakeModel,
    forward: Option<(SyncSender<()>, Receiver<()>)>,
    drop_wait: Option<Receiver<()>>,
}
impl Model for Controlled {
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
        if let Some((entered, release)) = self.forward.take() {
            entered.send(()).unwrap();
            release.recv_timeout(DEADLINE).unwrap();
        }
        self.inner.forward(s, b, kv, logits)
    }
}
impl Drop for Controlled {
    fn drop(&mut self) {
        if let Some(release) = &self.drop_wait {
            eprintln!("partial_model_drop");
            let _ = release.recv_timeout(DEADLINE);
        }
    }
}
fn options(exit: bool) -> EngineOptions {
    let mut opts = EngineOptions::new("fake".into(), CPU);
    opts.max_context = 512;
    opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 1.0)];
    opts.on_fault = if exit { OnFault::Exit } else { OnFault::Fail };
    // The hang trigger times the watchdog, so it keeps the GPU floor on the fast fake model.
    opts.step_timeout_floor = Some(Duration::from_secs(5));
    opts
}
fn terminal(r: &RequestHandle) -> EngineEvent {
    loop {
        let event = r.events.recv_timeout(DEADLINE).expect("terminal event");
        if !matches!(
            event,
            EngineEvent::Token { .. } | EngineEvent::Prefilled { .. }
        ) {
            assert_eq!(
                r.events.recv_timeout(Duration::from_millis(100)),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
            );
            return event;
        }
    }
}
fn records(ring: RingHandle, fault: FaultCode, expected: &[(u64, EngineError)]) {
    let mut reader = ring.reader(ReadFrom::Oldest);
    let mut faults = Vec::new();
    let mut spans = Vec::new();
    while let Read::Record(record) = reader.next() {
        match record {
            Record::HealthEvent(_, event) if event.to == HealthState::Failed.raw() => {
                faults.push(event.fault)
            }
            Record::RequestSpan(_, span) => spans.push((span.req_id, span.error)),
            _ => {}
        }
    }
    assert_eq!(faults, [fault.raw()]);
    spans.sort_unstable();
    let mut want: Vec<_> = expected.iter().map(|&(id, err)| (id, err.code())).collect();
    want.sort_unstable();
    assert_eq!(spans, want);
}
fn failed(e: &EngineHandle, code: FaultCode) {
    assert_eq!(e.health().state, HealthState::Failed);
    assert_eq!(e.health().last_fault, code);
    assert!(matches!(
        e.submit(Request::new(vec![1], 1)),
        Err(EngineError::NotReady(HealthState::Failed))
    ));
    assert_eq!(
        e.reload_with(
            ReloadChange::default(),
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64))))
        ),
        Err(EngineError::NotReady(HealthState::Failed))
    );
}
fn step_child(trigger: &str, exit: bool) {
    let (entered_tx, entered) = sync_channel(1);
    let (release, held) = sync_channel(1);
    let mut opts = options(exit);
    if trigger.starts_with("oom@") {
        // Growth happens when observed rows are scheduled, beside the held running request.
        opts.knob_overrides = vec![(knob_id::MAX_BATCH_SEQS, 2.0)];
    }
    let e = start_with(
        opts,
        Box::new(move |_, _, _| {
            Ok(Box::new(Controlled {
                inner: FakeModel::new(64),
                forward: Some((entered_tx, held)),
                drop_wait: None,
            }))
        }),
    )
    .unwrap();
    let running = e.submit(Request::new(vec![1], 3)).unwrap();
    entered.recv_timeout(DEADLINE).unwrap();
    let forced = trigger.starts_with("oom@").then(|| {
        e.force(Forced {
            tokens: vec![1; 80],
            prompt: 80,
            session: 0,
            keep_kv: false,
        })
        .unwrap()
    });
    // OOM: both seats are taken before this request, so it still receives the queued fault error.
    let queued = e.submit(Request::new(vec![10], 1)).unwrap();
    let (code, error) = if trigger.starts_with("panic@") {
        (FaultCode::Panic, EngineError::Internal(0))
    } else if trigger.starts_with("device_lost@") {
        (FaultCode::DeviceLost, EngineError::DeviceLost)
    } else if trigger.starts_with("hang@") {
        (FaultCode::Timeout, EngineError::DeviceLost)
    } else {
        (FaultCode::Oom, EngineError::Internal(2))
    };
    if ACTIVE {
        let mut expected = vec![
            (running.id, error),
            (queued.id, EngineError::NotReady(HealthState::Failed)),
        ];
        if let Some(f) = &forced {
            expected.push((f.request.id, error));
        }
        let (flushed_tx, flushed) = sync_channel(1);
        e.set_fault_flush(move |ring, deadline| {
            assert!(deadline > Instant::now());
            records(ring, code, &expected);
            assert_eq!(terminal(&running), EngineEvent::Error(error));
            assert_eq!(
                terminal(&queued),
                EngineEvent::Error(EngineError::NotReady(HealthState::Failed))
            );
            if let Some(f) = forced {
                assert_eq!(terminal(&f.request), EngineEvent::Error(error));
                assert_eq!(
                    f.positions.recv_timeout(Duration::from_millis(100)),
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
                );
            }
            eprintln!("verified_final_records_and_terminals");
            let _ = flushed_tx.send(());
        })
        .unwrap();
        release.send(()).unwrap();
        flushed.recv_timeout(DEADLINE).unwrap();
        if exit {
            std::thread::sleep(DEADLINE);
            panic!("fault must exit");
        }
        failed(&e, code);
    } else {
        release.send(()).unwrap();
        assert!(matches!(terminal(&running), EngineEvent::Done { .. }));
        assert!(matches!(terminal(&queued), EngineEvent::Done { .. }));
        if let Some(f) = forced {
            assert!(matches!(terminal(&f.request), EngineEvent::Done { .. }));
        }
        assert_eq!(e.health().state, HealthState::Ready);
    }
    e.shutdown();
}
fn allocation_child(scenario: &str, exit: bool) {
    let loaded = start_with(
        options(exit),
        Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
    );
    if scenario == "startup" && ACTIVE {
        assert!(
            matches!(loaded, Err(LoadError::DoesNotFit { need, have }) if need > 0 && have > 0)
        );
        return;
    }
    let e = loaded.unwrap();
    if scenario == "reload" {
        let (release, held) = sync_channel(1);
        let (flushed_tx, flushed) = sync_channel(1);
        e.set_fault_flush(move |ring, _| {
            records(ring, FaultCode::Oom, &[]);
            eprintln!("verified_reload_oom_flush");
            let _ = flushed_tx.send(());
        })
        .unwrap();
        let before = Instant::now();
        let result = e.reload_with(
            ReloadChange::default(),
            Box::new(move |_, _, _| {
                Ok(Box::new(Controlled {
                    inner: FakeModel::new(64),
                    forward: None,
                    drop_wait: Some(held),
                }))
            }),
        );
        if ACTIVE {
            assert_eq!(result, Err(EngineError::Internal(2)));
            assert!(
                before.elapsed() < Duration::from_millis(500),
                "OOM publication waited for cleanup"
            );
            flushed.recv_timeout(Duration::from_secs(1)).unwrap();
            failed(&e, FaultCode::Oom);
            if exit {
                std::thread::sleep(DEADLINE);
                panic!("fault must exit");
            }
            e.shutdown();
            // Retention must survive shutdown as well as publication.
            std::thread::sleep(Duration::from_millis(20));
            drop(release);
            return;
        }
        result.unwrap();
        release.send(()).unwrap();
    }
    assert!(matches!(
        terminal(&e.submit(Request::new(vec![1], 1)).unwrap()),
        EngineEvent::Done { .. }
    ));
    assert_eq!(e.health().state, HealthState::Ready);
    e.shutdown();
}
fn child(scenario: &str, trigger: &str, exit: bool) -> Output {
    let mut process = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "fault_child", "--nocapture"])
        .env("LARCH_INJECT_CHILD", scenario)
        .env("LARCH_FAULT", trigger)
        .env("LARCH_INJECT_EXIT", if exit { "1" } else { "0" })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(12);
    loop {
        if process.try_wait().unwrap().is_some() {
            return process.wait_with_output().unwrap();
        }
        if Instant::now() >= deadline {
            let _ = process.kill();
            let output = process.wait_with_output().unwrap();
            panic!(
                "killed/reaped {scenario}/{trigger}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn check(scenario: &str, trigger: &str, exit: bool) {
    let output = child(scenario, trigger, exit);
    let ended = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let status = if ACTIVE && exit && scenario != "startup" && scenario != "ignored" {
        3
    } else {
        0
    };
    assert_eq!(
        output.status.code(),
        Some(status),
        "{scenario}/{trigger}: {stderr}\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    if ACTIVE && scenario == "reload" {
        assert!(!stderr.contains("partial_model_drop"), "{stderr}");
    }
    if status == 3 {
        let detected = stderr
            .lines()
            .find_map(|line| {
                line.split_once("detected_ns=")
                    .and_then(|(_, ns)| ns.parse::<u128>().ok())
            })
            .expect("detection timestamp");
        assert!(
            ended >= detected && ended - detected <= 100_000_000,
            "exit exceeded 100 ms: {stderr}"
        );
        assert!(
            stderr.contains(if scenario == "reload" {
                "verified_reload_oom_flush"
            } else {
                "verified_final_records_and_terminals"
            }),
            "{stderr}"
        );
    }
}

#[test]
fn fault_child() {
    if let Ok(scenario) = std::env::var("LARCH_INJECT_CHILD") {
        let trigger = std::env::var("LARCH_FAULT").unwrap();
        let exit = std::env::var("LARCH_INJECT_EXIT").unwrap() == "1";
        if scenario == "step" {
            step_child(&trigger, exit);
        } else {
            allocation_child(&scenario, exit);
        }
    } else {
        // The default invocation also asserts a real engine round-trip.
        check("ignored", "invalid", false);
    }
}
#[test]
fn step_faults_fail_or_exit_with_final_records() {
    for trigger in ["panic@step:2", "device_lost@step:2", "hang@step:2"] {
        for exit in [false, true] {
            check("step", trigger, exit);
        }
    }
}
#[test]
fn allocation_counter_covers_startup_observation_and_reload() {
    for n in 1..=5 {
        for exit in [false, true] {
            check("startup", &format!("oom@alloc:{n}"), exit);
        }
    }
    for n in 6..=8 {
        for exit in [false, true] {
            check("step", &format!("oom@alloc:{n}"), exit);
        }
    }
    for n in 6..=10 {
        for exit in [false, true] {
            check("reload", &format!("oom@alloc:{n}"), exit);
        }
    }
}
#[test]
fn malformed_and_unreached_targets_do_not_fault() {
    for trigger in [
        "",
        "panic",
        "panic@step:0",
        "oom@alloc:0",
        "panic@alloc:1",
        "oom@step:1",
        "hang@step:-1",
        "device_lost@step:+1",
        "panic@step:1:2",
        "panic@step:18446744073709551616",
        "panic@step:2",
        "oom@alloc:6",
    ] {
        check("ignored", trigger, false);
    }
}
