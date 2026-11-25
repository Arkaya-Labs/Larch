//! `EngineKnobs`, the engine's `KnobControl` (architecture 4.6, 9.8): validation runs on the caller's
//! thread, then the change waits in the engine's queue for the next step boundary.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, SyncSender, sync_channel};
use std::time::{Duration, Instant};

use larch_records::{
    AbCommand, AbReply, Authority, DEFAULT_TIMEOUT_MS, EffectiveConfig, Health, HealthState,
    IndeterminateReason, KnobControl, KnobReply, KnobRequest, KnobSpec, REGISTRY, RejectReason,
    RevertTarget, RiskClass, knob_id, validate,
};

use crate::{Msg, Shared};

/// Pending changes the engine holds at most (9.8).
const QUEUE: usize = 64;

/// `period_steps` of an A/B that gives 0 (9.9).
const AB_PERIOD: u32 = 200;

pub struct EngineKnobs {
    pub(crate) shared: Arc<Shared>,
    pub(crate) tx: Sender<Msg>,
}

impl EngineKnobs {
    /// Takes a pending slot (9.8) and sends the command; the engine replies on the receiver. A
    /// failed or draining engine answers NotReady (9.11), whatever its queue held when it stopped.
    fn enqueue<R>(
        &self,
        msg: impl FnOnce(SyncSender<R>) -> Msg,
    ) -> Result<Receiver<R>, RejectReason> {
        let health = self.shared.health.lock().unwrap_or_else(|e| e.into_inner());
        if health.state != HealthState::Ready {
            return Err(RejectReason::NotReady);
        }
        if self.shared.pending_knobs.fetch_add(1, Ordering::Relaxed) >= QUEUE {
            self.shared.pending_knobs.fetch_sub(1, Ordering::Relaxed);
            return Err(RejectReason::Busy);
        }
        let (tx, rx) = sync_channel(1);
        if self.tx.send(msg(tx)).is_err() {
            self.shared.pending_knobs.fetch_sub(1, Ordering::Relaxed);
            return Err(RejectReason::NotReady);
        }
        Ok(rx)
    }

    /// How long `ab` and `revert` wait to be taken: the engine reads commands only between steps,
    /// and a step may run up to the watchdog's limit before it is a fault (9.12), so a slow step
    /// delays them rather than refusing them.
    fn step_wait(&self) -> Duration {
        Duration::from_millis(u64::from(DEFAULT_TIMEOUT_MS)).max(self.shared.watch.step_limit())
    }

    fn queue(&self, wait: Duration, msg: impl FnOnce(SyncSender<KnobReply>) -> Msg) -> KnobReply {
        let rx = match self.enqueue(msg) {
            Ok(rx) => rx,
            Err(reason) => return KnobReply::Rejected { reason },
        };
        // On timeout the change stays queued; its ActivationEvent records how it ended. Without a
        // reply the engine ended while it held the change: a shutdown never applies one after its
        // last drain, a fault may have been in the middle of it.
        let reason = match self.shared.receive(&rx, Some(Instant::now() + wait)) {
            Ok(reply) => return reply,
            Err(RecvTimeoutError::Timeout) => IndeterminateReason::TimedOutBeforeStep,
            Err(RecvTimeoutError::Disconnected) if self.health().state == HealthState::Draining => {
                return KnobReply::Rejected {
                    reason: RejectReason::NotReady,
                };
            }
            Err(RecvTimeoutError::Disconnected) => IndeterminateReason::FaultDuringApply,
        };
        KnobReply::Indeterminate { reason, rev: None }
    }
}

impl KnobControl for EngineKnobs {
    fn registry(&self) -> &'static [KnobSpec] {
        REGISTRY
    }

    fn effective(&self) -> Arc<EffectiveConfig> {
        self.shared.effective.read().expect("config").clone()
    }

    fn set(&self, req: KnobRequest) -> KnobReply {
        let health = self.health().state;
        if let Err(reason) = validate(
            &req,
            self.shared.info.read().expect("model info").caps.backend,
            health,
        ) {
            return KnobReply::Rejected { reason };
        }
        let t = self.shared.now();
        let wait = Duration::from_millis(u64::from(req.timeout_ms));
        self.queue(wait, |tx| Msg::Knob(req, t, tx))
    }

    /// The inverse is recorded with cause Inverse; gait may undo only its own changes (D7).
    fn revert(&self, target: RevertTarget, authority: Authority) -> KnobReply {
        let t = self.shared.now();
        self.queue(self.step_wait(), |tx| Msg::Revert(target, authority, t, tx))
    }

    /// Validates a start's deltas as a `set` under the command's authority and allowance (D7).
    /// A QualityAffecting knob is `Unauthorized` under any authority, `Human` too: quality trials
    /// run on calibration load, never as a live A/B (architecture 4.6, v2 plan decision 15).
    /// `max_waiting` 0 is `OutOfRange`: its arm would refuse every submit, and flips need steps.
    /// Trial 0 (no trial) or no knob at all is `WrongType`, a repeated knob `Conflict`, and
    /// `guardrail_ratio` `Conflict` with itself (arm B would loosen its own guard). `period_steps` 0
    /// is 200; a `max_steps` that ends the A/B before arm B runs is `OutOfRange`. Commands share the
    /// pending bound of `set`. `AbReply` has no Indeterminate: a command the engine has not taken
    /// within [`Self::step_wait`] is withdrawn and answered Busy, and one it took is waited for.
    fn ab(&self, mut cmd: AbCommand) -> AbReply {
        let reject = |reason| AbReply::Rejected { reason };
        if let AbCommand::Start {
            b,
            period_steps,
            max_steps,
            trial_id,
            authority,
            allow,
        } = &mut cmd
        {
            if *trial_id == 0 || b.iter().all(|d| d.knob == 0) {
                return reject(RejectReason::WrongType);
            }
            if *period_steps == 0 {
                *period_steps = AB_PERIOD;
            }
            let period = u64::from(*period_steps);
            if *max_steps != 0 && *max_steps <= period {
                return reject(RejectReason::OutOfRange {
                    min: (period + 1) as f64,
                    max: u64::MAX as f64,
                });
            }
            let health = self.health().state;
            for (i, d) in b.iter().enumerate().filter(|(_, d)| d.knob != 0) {
                if b[..i].iter().any(|e| e.knob == d.knob) || d.knob == knob_id::GUARDRAIL_RATIO {
                    return reject(RejectReason::Conflict { with: d.knob });
                }
                let req = KnobRequest {
                    allow: *allow,
                    ..KnobRequest::new(d.knob, d.value, *authority)
                };
                let spec = match validate(
                    &req,
                    self.shared.info.read().expect("model info").caps.backend,
                    health,
                ) {
                    Ok(spec) => spec,
                    Err(reason) => return reject(reason),
                };
                if spec.risk() == RiskClass::QualityAffecting {
                    return reject(RejectReason::Unauthorized { risk: spec.risk() });
                }
                if d.knob == knob_id::MAX_WAITING && d.value == 0.0 {
                    return reject(RejectReason::OutOfRange {
                        min: 1.0,
                        max: spec.max,
                    });
                }
            }
        }
        let t = self.shared.now();
        let taken = Arc::new(AtomicBool::new(false));
        let rx = match self.enqueue(|tx| Msg::Ab(cmd, t, taken.clone(), tx)) {
            Ok(rx) => rx,
            Err(reason) => return reject(reason),
        };
        match self
            .shared
            .receive(&rx, Some(Instant::now() + self.step_wait()))
        {
            Ok(reply) => reply,
            Err(RecvTimeoutError::Timeout) if !taken.swap(true, Ordering::AcqRel) => {
                reject(RejectReason::Busy)
            }
            Err(_) => self
                .shared
                .receive(&rx, None)
                .unwrap_or(reject(RejectReason::NotReady)),
        }
    }

    fn health(&self) -> Health {
        *self.shared.health.lock().unwrap_or_else(|e| e.into_inner())
    }
}
