//! Host-only terminal ownership and the watchdog (9.11–9.13). No lock here spans device work.
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, sync_channel};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{EngineEvent, Msg, OnFault, Position, Shared, TelemetryTier, Timings, Usage};
use larch_records::{
    AbEnd, AbReply, ActivationEvent, ActivationResult, Cause, ConfigurationRevision, DeviceSample,
    EffectiveConfig, EngineError, FaultCode, Finish, HealthEvent, HealthState, KnobReply, MF32,
    MU64, Outcome, RecordBody, RecordHeader, RejectReason, RequestSpan, RingHandle, RingWriter,
    Tier, knob_id,
};
use larch_sys::nvml::{Nvml, Read};

pub(crate) const POLL: Duration = Duration::from_millis(5);
pub(crate) const IDLE_WAIT: Duration = Duration::from_millis(250);
pub(crate) type Flush = Box<dyn FnOnce(RingHandle, Instant) + Send>;

/// Shared by the real request and watchdog; no scheduling or KV state is duplicated here.
pub(crate) struct Completion {
    events: Option<Sender<EngineEvent>>,
    pub(crate) positions: Option<Sender<Position>>,
    /// An activations request's rows, set by the step that tapped them; Done takes them.
    pub(crate) activations: Vec<Vec<f32>>,
    pub(crate) span: RequestSpan,
    pub(crate) arm: u8,
    pub(crate) running: bool,
    terminal: bool,
    /// The request's forced grammar picks, set when it ends.
    pub(crate) grammar_forced: u32,
    /// The adaptation version that produced it, set when it ends.
    pub(crate) adapter_version: u64,
}
impl Completion {
    pub(crate) fn new(events: Sender<EngineEvent>, span: RequestSpan) -> Self {
        Self {
            events: Some(events),
            positions: None,
            activations: Vec::new(),
            span,
            arm: 0,
            running: false,
            terminal: false,
            grammar_forced: 0,
            adapter_version: 0,
        }
    }
    pub(crate) fn token(&self, event: EngineEvent) {
        if let Some(events) = &self.events {
            let _ = events.send(event);
        }
    }
    pub(crate) fn position(&self, position: Position) {
        if let Some(positions) = &self.positions {
            let _ = positions.send(position);
        }
    }
    pub(crate) fn finish(
        &mut self,
        span: RequestSpan,
        arm: u8,
        end: Result<Finish, EngineError>,
    ) -> bool {
        if self.terminal {
            return false;
        }
        self.span = span;
        self.arm = arm;
        self.terminal = true;
        self.positions.take();
        let event = match end {
            Ok(finish) => EngineEvent::Done {
                finish,
                usage: Usage {
                    prompt_tokens: span.prompt_tokens,
                    completion_tokens: span.gen_tokens,
                    cached_tokens: span.cached_tokens,
                },
                timings: Timings {
                    grammar_forced: self.grammar_forced,
                    ..Timings::of(&span)
                },
                activations: std::mem::take(&mut self.activations),
                adapter_version: self.adapter_version,
            },
            Err(error) => EngineEvent::Error(error),
        };
        if let Some(events) = self.events.take() {
            let _ = events.send(event);
        }
        true
    }
    pub(crate) fn snapshot(&mut self, span: RequestSpan, arm: u8, running: bool) {
        if !self.terminal {
            self.span = span;
            self.arm = arm;
            self.running = running;
        }
    }
}

// Proposed §3/§4.4 decision: serialize the ring writes of the engine, the watchdog and the caller
// thread in `shutdown` (its Draining edge); no lock spans GPU work.
pub(crate) struct Host {
    pub(crate) ab: Option<crate::engine::Ab>,
    pub(crate) ab_done: Option<(u64, u64, AbEnd)>,
    pub(crate) step: u64,
    pub(crate) error: Option<EngineError>,
    ring: RingWriter,
    pub(crate) requests: HashMap<u64, Arc<Mutex<Completion>>>,
    pub(crate) flush: Option<Flush>,
    pub(crate) failed: bool,
    header: RecordHeader,
    enabled: bool,
}
impl Host {
    pub(crate) fn new(ring: RingWriter, header: RecordHeader, enabled: bool) -> Self {
        Self {
            ab: None,
            ab_done: None,
            step: 0,
            error: None,
            ring,
            requests: HashMap::new(),
            flush: None,
            failed: false,
            header,
            enabled,
        }
    }
    pub(crate) fn record<R: RecordBody>(&mut self, header: RecordHeader, body: &R) {
        if !self.failed {
            self.header = header;
            self.write(header, body);
        }
    }
    fn write<R: RecordBody>(&mut self, header: RecordHeader, body: &R) {
        if self.enabled {
            self.ring.push(header, body);
        }
    }
    pub(crate) fn span(&mut self, header: RecordHeader, span: &RequestSpan) {
        if self.requests.remove(&span.req_id).is_some() {
            self.record(header, span);
        }
    }
    /// A record from a thread other than the engine's, under the last header the engine wrote and
    /// the arm and revision in `cfg`, the published config: the last header may be a span's, under
    /// the arm its request was pinned to. The engine publishes under the host lock, which the caller
    /// holds, so `cfg` is the config in force.
    fn stamped<R: RecordBody>(&mut self, cfg: &EffectiveConfig, tier: Tier, now: u64, body: &R) {
        let mut header = self.header;
        header.t_mono_ns = now;
        header.t_wall_ns = wall_ns() as i64;
        header.tier = tier.raw();
        header.ab_arm = cfg.ab_arm;
        header.config_rev = cfg.rev;
        self.record(header, body);
    }
    pub(crate) fn draining(&mut self, cfg: &EffectiveConfig, from: HealthState, now: u64) {
        self.stamped(
            cfg,
            Tier::T0,
            now,
            &HealthEvent {
                from: from.raw(),
                to: HealthState::Draining.raw(),
                ..bytemuck::Zeroable::zeroed()
            },
        );
    }
}

/// The sampler's period (9.6).
const SAMPLE_PERIOD: Duration = Duration::from_secs(1);

/// One NVML reading as a DeviceSample, with `sample_tick` 0. A field NVML refuses is Unsupported,
/// one it fails to read Missing.
pub(crate) fn device_sample(nvml: &Nvml) -> DeviceSample {
    let s = nvml.sample();
    let f = |r: Read<u32>, scale: f32| match r {
        Read::Value(v) => MF32::measured(v as f32 * scale),
        Read::Unsupported => MF32::unsupported(),
        Read::Failed => MF32::missing(),
    };
    let u = |r: Read<u64>| match r {
        Read::Value(v) => MU64::measured(v),
        Read::Unsupported => MU64::unsupported(),
        Read::Failed => MU64::missing(),
    };
    DeviceSample {
        sm_clock_mhz: f(s.sm_clock_mhz, 1.0),
        mem_clock_mhz: f(s.mem_clock_mhz, 1.0),
        temp_c: f(s.temp_c, 1.0),
        power_w: f(s.power_mw, 1e-3),
        gpu_util: f(s.gpu_util, 0.01),
        vram_used_bytes: u(s.memory.map(|(used, _)| used)),
        vram_total_bytes: u(s.memory.map(|(_, total)| total)),
        power_limit_w: f(s.power_limit_mw, 1e-3),
        // ponytail: Missing until B0 (v2 spec 4.3) detects the Windows sysmem spill.
        spill: MF32::missing(),
        clocks_event_reasons: u(s.clocks_event_reasons),
        other_procs: u(s.other_procs.map(u64::from)),
        sample_tick: 0,
    }
}

/// The engine while it runs: gone once the engine has stopped or failed.
fn live(weak: &Weak<Shared>) -> Option<Arc<Shared>> {
    weak.upgrade()
        .filter(|sh| !sh.watch.stopped.load(Ordering::Acquire))
}

/// Writes a DeviceSample a second, ticks counted from 1, until the engine stops; none without
/// NVML or telemetry. At T0 it writes one in 64, as the engine does EngineSamples.
pub(crate) fn sampler(
    sh: &Arc<Shared>,
    telemetry: TelemetryTier,
) -> std::io::Result<Option<std::thread::JoinHandle<()>>> {
    if sh.device.is_none() || telemetry == TelemetryTier::Off {
        return Ok(None);
    }
    let weak = Arc::downgrade(sh);
    std::thread::Builder::new()
        .name("larch-sampler".into())
        .spawn(move || {
            let mut deadline = Instant::now();
            for tick in 1u64.. {
                // Ticks stay a period apart however long a sample takes.
                deadline += SAMPLE_PERIOD;
                // Woken early by the engine's stop, or spuriously.
                while let Some(wait) = deadline.checked_duration_since(Instant::now()) {
                    if live(&weak).is_none() {
                        return;
                    }
                    std::thread::park_timeout(wait);
                }
                let Some(sh) = live(&weak) else {
                    return;
                };
                let t1 = sh
                    .effective
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .values[knob_id::TELEMETRY_TIER as usize]
                    == 1.0;
                if t1 || tick.is_multiple_of(64) {
                    let nvml = sh.device.as_ref().expect("checked at spawn");
                    let sample = DeviceSample {
                        sample_tick: tick,
                        ..device_sample(nvml)
                    };
                    let now = sh.now();
                    let tier = if t1 { Tier::T1 } else { Tier::T0 };
                    let mut host = sh.host.lock().unwrap_or_else(|e| e.into_inner());
                    let cfg = **sh.effective.read().unwrap_or_else(|e| e.into_inner());
                    host.stamped(&cfg, tier, now, &sample);
                }
            }
        })
        .map(Some)
}

pub(crate) struct Watch {
    pub(crate) heartbeat: AtomicU64,
    pub(crate) pending: AtomicBool,
    pub(crate) stopped: AtomicBool,
    timeout: AtomicU64,
    floor: AtomicU64,
    samples: Mutex<VecDeque<u64>>,
}
impl Watch {
    pub(crate) fn new(floor: Duration) -> Self {
        let floor = u64::try_from(floor.as_nanos()).unwrap_or(u64::MAX);
        Self {
            heartbeat: AtomicU64::new(0),
            pending: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            timeout: AtomicU64::new(floor),
            floor: AtomicU64::new(floor),
            samples: Mutex::new(VecDeque::new()),
        }
    }
    /// A reload's floor: the old model's step times say nothing of the new one's.
    pub(crate) fn set_floor(&self, floor: Duration) {
        let floor = u64::try_from(floor.as_nanos()).unwrap_or(u64::MAX);
        let mut samples = self.samples.lock().expect("watchdog samples");
        samples.clear();
        self.floor.store(floor, Ordering::Release);
        self.timeout.store(floor, Ordering::Release);
    }
    pub(crate) fn begin(&self, now: u64) {
        if !self.pending.load(Ordering::Acquire) {
            self.heartbeat.store(now, Ordering::Release);
            self.pending.store(true, Ordering::Release);
        }
    }
    pub(crate) fn idle(&self) {
        self.pending.store(false, Ordering::Release);
    }
    pub(crate) fn complete(&self, now: u64, duration: Option<u64>) {
        self.heartbeat.store(now, Ordering::Release);
        if let Some(duration) = duration {
            let mut samples = self.samples.lock().expect("watchdog samples");
            if samples.len() == 200 {
                samples.pop_front();
            }
            samples.push_back(duration);
            self.timeout.store(
                crate::guard::p99(samples.iter().copied())
                    .saturating_mul(10)
                    .max(self.floor.load(Ordering::Acquire)),
                Ordering::Release,
            );
        }
    }
    /// How long a step may run before the watchdog faults the engine (9.12).
    pub(crate) fn step_limit(&self) -> Duration {
        Duration::from_nanos(self.timeout.load(Ordering::Acquire))
    }
    fn expired(&self, now: u64) -> bool {
        self.pending.load(Ordering::Acquire)
            && now.saturating_sub(self.heartbeat.load(Ordering::Acquire))
                > self.timeout.load(Ordering::Acquire)
    }
}

pub(crate) fn wall_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

pub(crate) fn recovery_ms(value: Option<&str>, now: u64) -> u32 {
    value
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0 && v <= now)
        .map_or(0, |v| {
            ((now - v) / 1_000_000).min(u64::from(u32::MAX)) as u32
        })
}

pub(crate) fn watchdog(
    sh: &Arc<Shared>,
    on_fault: OnFault,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    let weak = Arc::downgrade(sh);
    std::thread::Builder::new()
        .name("larch-watchdog".into())
        .spawn(move || {
            loop {
                let Some(sh) = weak.upgrade() else {
                    return;
                };
                if sh.watch.stopped.load(Ordering::Acquire) {
                    return;
                }
                if sh.watch.expired(sh.now()) {
                    // ponytail: EngineError has no Timeout contract variant; upgrade when the records
                    // owner adds the approved variant. DeviceLost requires the same process restart.
                    fault(&sh, (FaultCode::Timeout, EngineError::DeviceLost), on_fault);
                    if sh.watch.stopped.load(Ordering::Acquire) {
                        return;
                    }
                }
                let delay = if sh.watch.pending.load(Ordering::Acquire) {
                    POLL
                } else {
                    IDLE_WAIT
                };
                drop(sh);
                std::thread::park_timeout(delay);
            }
        })
}

/// Serializes final host records and closes admission before notifying any caller.
pub(crate) fn fault(sh: &Shared, (code, error): (FaultCode, EngineError), on_fault: OnFault) {
    let detected = Instant::now();
    let detected_ns = wall_ns();
    // CI runners exited up to 59 ms past a held hook (macOS; windows-2022 16 ms; runs at 10 and
    // 30 ms), so 20 ms keeps the worst seen near 79 ms of 9.11's 100 ms.
    let deadline = detected + Duration::from_millis(20);
    let (hook, inbox) = {
        let mut health = sh.health.lock().unwrap_or_else(|e| e.into_inner());
        if health.state == HealthState::Failed {
            return;
        }
        // A completion may have landed after the watchdog sampled its heartbeat.
        if code == FaultCode::Timeout && !sh.watch.expired(sh.now()) {
            return;
        }
        let mut host = sh.host.lock().unwrap_or_else(|e| e.into_inner());
        let now = sh.now();
        let from = health.state;
        health.state = HealthState::Failed;
        sh.failed.store(true, Ordering::Release);
        health.since_mono_ns = now;
        health.last_fault = code;
        let mut header = host.header;
        header.t_mono_ns = now;
        header.t_wall_ns = detected_ns as i64;
        header.tier = Tier::T0.raw();
        host.write(
            header,
            &HealthEvent {
                from: from.raw(),
                to: HealthState::Failed.raw(),
                fault: code.raw(),
                ..bytemuck::Zeroable::zeroed()
            },
        );
        host.error = Some(error);
        let step = host.step;
        if let Some(ab) = host.ab.take() {
            let trial_id = ab.trial;
            host.ab_done = Some((trial_id, (step + 1).saturating_sub(ab.start), AbEnd::Fault));
            let mut effective = sh.effective.write().unwrap_or_else(|e| e.into_inner());
            let mut config = **effective;
            config.ab_arm = 0;
            header.ab_arm = 0;
            for (knob, value, _) in ab.knobs {
                let previous = config.values[knob as usize];
                let prev_rev = config.rev;
                config.rev += 1;
                config.values[knob as usize] = value;
                header.config_rev = config.rev;
                host.write(
                    header,
                    &ConfigurationRevision {
                        rev: config.rev,
                        prev_rev,
                        knob,
                        cause: Cause::AbFlip.raw(),
                        requested: value,
                        effective: value,
                        previous,
                        step_applied: step + 1,
                        trial_id,
                        ..bytemuck::Zeroable::zeroed()
                    },
                );
            }
            *effective = Arc::new(config);
        }
        let inbox = sh.inbox.lock().unwrap_or_else(|e| e.into_inner()).take();
        let messages: Vec<_> = inbox.iter().flat_map(|rx| rx.try_iter()).collect();
        let reason = RejectReason::NotReady;
        let mut acknowledgements = Vec::new();
        for message in messages {
            match message {
                Msg::Knob(req, t, reply) => {
                    host.write(
                        header,
                        &ActivationEvent {
                            knob: req.knob,
                            result: ActivationResult::Rejected.raw(),
                            reason: reason.code(),
                            requested: req.value,
                            effective: sh
                                .effective
                                .read()
                                .unwrap_or_else(|e| e.into_inner())
                                .values
                                .get(req.knob as usize)
                                .copied()
                                .unwrap_or(0.0),
                            t_request_ns: t,
                            t_applied_ns: now,
                            step,
                            trial_id: req.trial_id,
                            ..bytemuck::Zeroable::zeroed()
                        },
                    );
                    let _ = reply.send(KnobReply::Rejected { reason });
                }
                Msg::Revert(.., reply) => {
                    let _ = reply.send(KnobReply::Rejected { reason });
                }
                Msg::Ab(.., reply) => {
                    let _ = reply.send(AbReply::Rejected { reason });
                }
                Msg::PrefixStats(reply) => drop(reply),
                Msg::DocFit(_, reply) => {
                    let _ = reply.send(Err(EngineError::NotReady(HealthState::Failed)));
                }
                Msg::PinOp(_, reply) => {
                    let _ = reply.send(Err(EngineError::NotReady(HealthState::Failed)));
                }
                Msg::Pin(.., reply) => {
                    let _ = reply.send(Err(EngineError::NotReady(HealthState::Failed)));
                }
                Msg::Unpin(_) => {}
                Msg::Parked(_, reply) => {
                    let _ = reply.send(0);
                }
                Msg::Reload(_, load, reply) => {
                    std::mem::forget(load);
                    let _ = reply.send(Err(EngineError::NotReady(HealthState::Failed)));
                }
                Msg::RouterDeltas(.., reply) => {
                    let _ = reply.send(Err(EngineError::NotReady(HealthState::Failed)));
                }
                Msg::Job(job) => drop(job),
                Msg::Prepare(..) => {}
                Msg::Rejected(span) => host.write(header, &span),
                Msg::FinishStop(_, reply) => acknowledgements.push(reply),
                Msg::Submit(_) | Msg::Release(_) | Msg::Shutdown => {}
            }
        }
        sh.pending_knobs.store(0, Ordering::Release);
        for (_, notification) in std::mem::take(&mut host.requests) {
            let mut notification = notification.lock().unwrap_or_else(|e| e.into_inner());
            let mut span = notification.span;
            if !notification.terminal {
                let error = if notification.running {
                    error
                } else {
                    EngineError::NotReady(HealthState::Failed)
                };
                span.t_done_ns = now;
                span.outcome = if error == EngineError::DeviceLost {
                    Outcome::DeviceLost
                } else {
                    Outcome::Error
                }
                .raw();
                span.finish = Finish::None.raw();
                span.error = error.code();
                // Publish each span before its terminal event is observable.
                header.ab_arm = notification.arm;
                host.write(header, &span);
                let arm = notification.arm;
                notification.finish(span, arm, Err(error));
            } else {
                header.ab_arm = notification.arm;
                host.write(header, &span);
            }
        }
        drop(acknowledgements);
        host.failed = true;
        (host.flush.take(), inbox)
    };
    // Receiver disposal can drop queued captures; no user cleanup runs under publication locks.
    drop(inbox);
    sh.watch.stopped.store(true, Ordering::Release);
    // Neither stderr nor a contributor's fsync callback can delay the fault thread indefinitely.
    let ring = sh.ring.clone();
    let (tx, rx) = sync_channel(1);
    let _ = std::thread::Builder::new()
        .name("larch-fault-flush".into())
        .spawn(move || {
            let _ = writeln!(
                std::io::stderr(),
                "larch fault={code:?} detected_ns={detected_ns}"
            );
            if let Some(hook) = hook {
                let _ =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| hook(ring, deadline)));
            }
            let _ = tx.send(());
        });
    let _ = rx.recv_timeout(deadline.saturating_duration_since(Instant::now()));
    if on_fault == OnFault::Exit {
        std::process::exit(3);
    }
}

impl Shared {
    /// Device work may never return; synchronous callers poll only host health while waiting.
    pub(crate) fn receive<T>(
        &self,
        rx: &Receiver<T>,
        deadline: Option<Instant>,
    ) -> Result<T, RecvTimeoutError> {
        loop {
            if let Ok(reply) = rx.try_recv() {
                return Ok(reply);
            }
            if self.health.lock().unwrap_or_else(|e| e.into_inner()).state == HealthState::Failed {
                return rx.try_recv().map_err(|_| RecvTimeoutError::Disconnected);
            }
            let wait = deadline.map_or(POLL, |d| {
                d.saturating_duration_since(Instant::now()).min(POLL)
            });
            match rx.recv_timeout(wait) {
                Err(RecvTimeoutError::Timeout) if deadline.is_none_or(|d| Instant::now() < d) => {}
                result => return result,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineOptions, Request, start_with};
    use larch_kernels::CPU;
    use larch_model::FakeModel;

    #[test]
    fn terminal_fault_disconnects_the_original_inbox() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let e = start_with(opts, Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64))))).unwrap();
        fault(
            &e.shared,
            (FaultCode::Panic, EngineError::Internal(0)),
            OnFault::Fail,
        );
        assert!(
            e.tx.send(Msg::Release(1)).is_err(),
            "failed resources must not retain connected intake"
        );
        e.shutdown();
    }

    #[test]
    fn watchdog_rechecks_expiry_after_completion_wins_publication() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        // No engine loop or watchdog thread runs, so only this test moves the heartbeat.
        let (_engine, e) = crate::engine::Engine::init(
            opts,
            Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64)))),
            None,
        )
        .unwrap();
        let health = e.shared.health.lock().unwrap();
        e.shared.watch.timeout.store(0, Ordering::Release);
        e.shared.watch.begin(0);
        let sh = e.shared.clone();
        let (sampled_tx, sampled) = sync_channel(1);
        let watchdog = std::thread::spawn(move || {
            assert!(sh.watch.expired(sh.now()));
            sampled_tx.send(()).unwrap();
            fault(
                &sh,
                (FaultCode::Timeout, EngineError::DeviceLost),
                OnFault::Fail,
            );
        });
        sampled.recv_timeout(Duration::from_secs(1)).unwrap();
        e.shared.watch.complete(e.shared.now(), None);
        e.shared.watch.idle();
        drop(health);
        watchdog.join().unwrap();
        assert_eq!(e.health().state, HealthState::Ready);
    }

    #[test]
    fn engine_does_not_wait_for_a_callers_health_lock() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let e = start_with(opts, Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64))))).unwrap();
        // admit() holds this lock across its send; waking on it hands the CPU to the engine.
        let health = e.shared.health.lock().unwrap();
        let (events_tx, events) = std::sync::mpsc::channel();
        let id = e.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let cancel = Arc::new(AtomicBool::new(false));
        let r = crate::sched::Req::new(id, Request::new(vec![1], 2), events_tx, cancel, 1);
        e.shared.in_flight.fetch_add(1, Ordering::AcqRel);
        e.tx.send(Msg::Submit(vec![r])).unwrap();
        // The token needs a whole turn: the failed check, handle and a step.
        let token = events
            .iter()
            .find(|e| !matches!(e, EngineEvent::Prefilled { .. }));
        assert!(
            matches!(token, Some(EngineEvent::Token { .. })),
            "{token:?}"
        );
        drop(health);
        e.shutdown();
    }

    #[test]
    fn watchdog_ignores_idle_and_restarts_its_clock() {
        let watch = Watch::new(Duration::from_secs(5));
        assert!(!watch.expired(60_000_000_000));
        watch.begin(60_000_000_000);
        assert!(!watch.expired(65_000_000_000));
        assert!(watch.expired(65_000_000_001));
        watch.idle();
        assert!(!watch.expired(120_000_000_000));
        watch.begin(120_000_000_000);
        assert!(!watch.expired(120_000_000_001));
    }

    #[test]
    fn a_longer_watchdog_floor_holds() {
        let watch = Watch::new(Duration::from_secs(60));
        watch.begin(0);
        watch.complete(0, Some(1_000_000));
        assert!(!watch.expired(60_000_000_000));
        assert!(watch.expired(60_000_000_001));
    }

    #[test]
    fn a_reload_floor_replaces_the_old_models_samples() {
        let watch = Watch::new(Duration::from_secs(5));
        watch.complete(0, Some(1_000_000_000));
        assert_eq!(watch.step_limit(), Duration::from_secs(10));
        watch.set_floor(Duration::from_secs(60));
        assert_eq!(watch.step_limit(), Duration::from_secs(60));
        watch.complete(0, Some(1_000_000));
        assert_eq!(watch.step_limit(), Duration::from_secs(60));
    }

    #[test]
    fn a_zero_floor_leaves_only_ten_times_p99() {
        let watch = Watch::new(Duration::ZERO);
        watch.begin(0);
        assert!(watch.expired(1));
        watch.complete(0, Some(100_000_000));
        assert!(!watch.expired(1_000_000_000));
        assert!(watch.expired(1_000_000_001));
    }

    #[test]
    fn watchdog_uses_ten_times_nearest_rank_p99() {
        let watch = Watch::new(Duration::from_secs(5));
        watch.begin(0);
        for _ in 0..99 {
            watch.complete(1_000_000_000, Some(600_000_000));
        }
        // One outlier in 100 is above the nearest-rank p99.
        watch.complete(1_000_000_000, Some(2_000_000_000));
        assert!(!watch.expired(7_000_000_000));
        assert!(watch.expired(7_000_000_001));
        watch.complete(2_000_000_000, Some(2_000_000_000));
        assert!(!watch.expired(22_000_000_000));
        assert!(watch.expired(22_000_000_001));
    }

    #[test]
    fn restart_timestamp_validation_does_not_invent_measurements() {
        for (value, expected) in [
            (None, 0),
            (Some("bad"), 0),
            (Some("-1"), 0),
            (Some("0"), 0),
            (Some("1000000001"), 0),
            (Some("500000000"), 500),
        ] {
            assert_eq!(recovery_ms(value, 1_000_000_000), expected);
        }
    }

    #[test]
    fn a_taken_knob_reports_fault_during_apply() {
        use larch_records::{Authority, IndeterminateReason, KnobControl, KnobRequest, knob_id};
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let e = start_with(opts, Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64))))).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let knobs = crate::knobs::EngineKnobs {
            shared: e.shared.clone(),
            tx,
        };
        let (done_tx, done) = sync_channel(1);
        let caller = std::thread::spawn(move || {
            done_tx
                .send(knobs.set(KnobRequest::new(
                    knob_id::MAX_BATCH_SEQS,
                    8.0,
                    Authority::Human,
                )))
                .unwrap();
        });
        let Msg::Knob(_, _, reply) = rx.recv_timeout(Duration::from_secs(1)).unwrap() else {
            panic!("knob command");
        };
        fault(
            &e.shared,
            (FaultCode::Panic, EngineError::Internal(0)),
            OnFault::Fail,
        );
        let result = done.recv_timeout(Duration::from_millis(100));
        drop(reply);
        caller.join().unwrap();
        assert_eq!(
            result.unwrap(),
            KnobReply::Indeterminate {
                reason: IndeterminateReason::FaultDuringApply,
                rev: None
            }
        );
        e.shutdown();
    }

    #[test]
    fn a_taken_ab_reply_is_bounded_after_fault() {
        use larch_records::{AbCommand, KnobControl};
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let e = start_with(opts, Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64))))).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let knobs = crate::knobs::EngineKnobs {
            shared: e.shared.clone(),
            tx,
        };
        let (done_tx, done) = sync_channel(1);
        let caller = std::thread::spawn(move || {
            done_tx
                .send(knobs.ab(AbCommand::Stop { trial_id: 1 }))
                .unwrap();
        });
        let Msg::Ab(_, _, taken, reply) = rx.recv_timeout(Duration::from_secs(1)).unwrap() else {
            panic!("A/B command");
        };
        taken.store(true, Ordering::Release);
        fault(
            &e.shared,
            (FaultCode::Panic, EngineError::Internal(0)),
            OnFault::Fail,
        );
        let result = done.recv_timeout(Duration::from_millis(100));
        drop(reply);
        caller.join().unwrap();
        assert_eq!(
            result.unwrap(),
            AbReply::Rejected {
                reason: RejectReason::NotReady
            }
        );
        e.shutdown();
    }

    #[test]
    fn poisoned_host_metadata_does_not_replace_the_fault_path() {
        let mut opts = EngineOptions::new("fake".into(), CPU);
        opts.max_context = 512;
        let e = start_with(opts, Box::new(|_, _, _| Ok(Box::new(FakeModel::new(64))))).unwrap();
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _host = e.shared.host.lock().unwrap();
            panic!("controlled host metadata poison");
        }));
        let request = e.submit(Request::new(vec![1], 1)).unwrap();
        fault(
            &e.shared,
            (FaultCode::Panic, EngineError::Internal(0)),
            OnFault::Fail,
        );
        assert_eq!(e.health().state, HealthState::Failed);
        let _ = request.events.recv_timeout(Duration::from_secs(1)).unwrap();
        e.shutdown();
    }

    /// A record from another thread carries the published arm and revision, not those of the last
    /// span, which ran under the arm its request was pinned to.
    #[test]
    fn stamped_records_carry_the_published_config() {
        let (ring, handle) = larch_records::ring(16);
        let mut host = Host::new(ring, bytemuck::Zeroable::zeroed(), true);
        let span = RecordHeader {
            ab_arm: 1,
            config_rev: 3,
            ..bytemuck::Zeroable::zeroed()
        };
        host.record(span, &<RequestSpan as bytemuck::Zeroable>::zeroed());
        let cfg = EffectiveConfig {
            rev: 4,
            values: [0.0; 32],
            ab_arm: 2,
        };
        host.draining(&cfg, HealthState::Ready, 5);
        let mut reader = handle.reader(larch_records::ReadFrom::Oldest);
        let mut last = None;
        while let larch_records::Read::Record(r) = reader.next() {
            last = Some(r);
        }
        let Some(larch_records::Record::HealthEvent(h, _)) = last else {
            panic!("{last:?}")
        };
        assert_eq!((h.ab_arm, h.config_rev), (2, 4));
    }
}
