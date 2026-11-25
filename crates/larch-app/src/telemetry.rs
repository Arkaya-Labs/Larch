//! The ring reader thread: EngineSample, DeviceSample, ConfigurationRevision and HealthEvent
//! records become the `engine:*` events.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use larch_records::{
    DeploymentSnapshot, DeviceSample, HealthState, MState, MU64, Read, ReadFrom, Record,
    RingHandle, knob_spec,
};

use crate::{
    Emit, Event, camel,
    contract::{ConfigChange, EngineSample, Failed, Health, PowerCap},
};

/// About 4 samples a second (contract.rs).
const TICK: Duration = Duration::from_millis(250);

/// NVML clock-event reasons a firmware power budget sets: SW power cap and SW thermal slowdown.
const CAPPED: u64 = 0x4 | 0x20;

/// Reads `ring` from its oldest record (the load's snapshot and revisions) until `stop` is set.
pub(crate) fn spawn(ring: RingHandle, engine: u64, emit: Emit, stop: Arc<AtomicBool>) {
    std::thread::Builder::new()
        .name("larch-app-telemetry".into())
        .spawn(move || {
            let mut reader = ring.reader(ReadFrom::Oldest);
            let mut t = Telemetry {
                engine,
                ..Telemetry::default()
            };
            let mut tick = Instant::now();
            loop {
                std::thread::sleep(TICK);
                // Checked after the sleep, so an unloaded engine sends no stale status.
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                loop {
                    match reader.next() {
                        Read::Record(r) => t.record(r, &emit),
                        // Lost samples only shorten the window; the totals stay exact.
                        Read::Lagged { .. } => {}
                        Read::Empty => break,
                    }
                }
                if let Some(s) = t.sample(tick.elapsed()) {
                    emit(Event::Sample(s));
                }
                tick = Instant::now();
            }
        })
        .expect("spawning the telemetry thread");
}

#[derive(Default)]
struct Telemetry {
    /// The load this engine is, stamped on its changes.
    engine: u64,
    deployment: Option<DeploymentSnapshot>,
    device: Option<DeviceSample>,
    /// The highest enforced power limit seen since load, watts.
    // ponytail: the load's limit stands in for the GPU's full one, so an engine loaded while
    // already capped shows no cap; read NVML's default limit when DeviceSample carries it.
    full_w: f32,
    last: Option<larch_records::EngineSample>,
    /// `last` at the start of this window: the engine's totals start at zero.
    base: (u64, u64, u64),
    /// A step ran in this window.
    fresh: bool,
}

impl Telemetry {
    fn record(&mut self, r: Record, emit: &Emit) {
        match r {
            Record::DeploymentSnapshot(_, d) => self.deployment = Some(d),
            Record::DeviceSample(_, d) => self.device(d),
            Record::EngineSample(_, s) => {
                self.last = Some(s);
                self.fresh = true;
            }
            Record::ConfigurationRevision(h, c) => emit(Event::Config(ConfigChange {
                engine: self.engine,
                rev: c.rev,
                knob: knob_spec(c.knob)
                    .map_or("unknown", |k| k.name.as_str())
                    .into(),
                old: c.previous,
                new: c.effective,
                cause: camel(c.cause()),
                at: (h.t_wall_ns.max(0) / 1_000_000) as u64,
            })),
            Record::HealthEvent(_, e) if e.to() == HealthState::Failed => {
                emit(Event::Failed(Failed {
                    message: format!("The GPU stopped ({:?}). Restart Larch.", e.fault()),
                }));
                emit(Event::Health(Health::Failed));
            }
            _ => {}
        }
    }

    /// From the load on: before the engine's first step a sample with nothing running, so the
    /// memory a loaded model takes shows at once; None before the load's snapshot.
    fn sample(&mut self, window: Duration) -> Option<EngineSample> {
        let d = self.deployment?;
        // ponytail: f16 K and V per layer, as larch-engine's fit.rs sizes a block; read the
        // engine's block bytes once DeploymentSnapshot or the facade carries them. Before the first
        // step no EngineSample carries kv_block_tokens: the engine's 16 (sched.rs) stands in.
        let block = |tokens: u32| {
            2 * u64::from(d.n_layers)
                * u64::from(tokens)
                * u64::from(d.n_kv_heads)
                * u64::from(d.head_dim)
                * 2
        };
        let measured = |m: MU64| (m.state() == MState::Measured).then_some(m.v);
        let (gpu_used_bytes, gpu_total_bytes) = (
            self.device.and_then(|d| measured(d.vram_used_bytes)),
            self.device.and_then(|d| measured(d.vram_total_bytes)),
        );
        let power_cap = self.power_cap();
        let Some(s) = self.last else {
            return Some(EngineSample {
                tok_s: 0.0,
                running: 0,
                waiting: 0,
                queue_wait_ms: 0.0,
                weights_bytes: d.weights_bytes,
                kv_used_bytes: 0,
                kv_total_bytes: u64::from(d.kv_pool_blocks) * block(16),
                gpu_used_bytes,
                gpu_total_bytes,
                power_cap,
            });
        };
        let block = block(s.kv_block_tokens);
        let (generated, queue_ns, queued) = self.base;
        let fresh = std::mem::take(&mut self.fresh);
        self.base = (s.tot_gen_tokens, s.tot_queue_ns, s.tot_queued);
        // No step in the window: nothing ran or waited (a waiting request is admitted at a step).
        let (tok_s, running, waiting, queue_wait_ms) = if fresh {
            let queued = s.tot_queued - queued;
            (
                (s.tot_gen_tokens - generated) as f64 / window.as_secs_f64(),
                s.n_running,
                s.n_waiting,
                if queued == 0 {
                    0.0
                } else {
                    (s.tot_queue_ns - queue_ns) as f64 / queued as f64 / 1e6
                },
            )
        } else {
            (0.0, 0, 0, 0.0)
        };
        Some(EngineSample {
            tok_s,
            running,
            waiting,
            queue_wait_ms,
            weights_bytes: d.weights_bytes,
            kv_used_bytes: u64::from(s.kv_blocks_used) * block,
            kv_total_bytes: u64::from(s.kv_blocks_total) * block,
            gpu_used_bytes,
            gpu_total_bytes,
            power_cap,
        })
    }

    /// Keeps the latest sample and the highest enforced limit so far.
    fn device(&mut self, d: DeviceSample) {
        self.full_w = self.full_w.max(d.power_limit_w.get().unwrap_or(0.0));
        self.device = Some(d);
    }

    /// The cap while a capping reason is set and the limit is below the highest seen.
    fn power_cap(&self) -> Option<PowerCap> {
        let d = self.device?;
        let limit_w = d.power_limit_w.get()?;
        let reasons = d.clocks_event_reasons.get()?;
        (reasons & CAPPED != 0 && limit_w < self.full_w).then_some(PowerCap {
            limit_w,
            full_w: self.full_w,
        })
    }
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;
    use larch_records::{DeviceSample, MF32, MU64};

    use super::*;

    #[test]
    fn power_cap_follows_the_limit_and_the_capping_reasons() {
        let sample = |limit: f32, reasons: u64| DeviceSample {
            power_limit_w: MF32::measured(limit),
            clocks_event_reasons: MU64::measured(reasons),
            ..DeviceSample::zeroed()
        };
        let mut t = Telemetry::default();
        let mut see = |d: DeviceSample| {
            t.device(d);
            t.power_cap()
        };
        assert_eq!(see(sample(35.0, 0x1)), None);
        // XPS 15, 27 Oct: 35 -> 20 W with SW power cap + SW thermal (0x24).
        let cap = Some(PowerCap {
            limit_w: 20.0,
            full_w: 35.0,
        });
        assert_eq!(see(sample(20.0, 0x24)), cap);
        assert_eq!(see(sample(20.0, 0x4)), cap);
        // A lower limit without a capping reason, or a capping reason at the full limit, is no cap.
        assert_eq!(see(sample(20.0, 0x1)), None);
        assert_eq!(see(sample(35.0, 0x24)), None);
        let unmeasured = DeviceSample {
            clocks_event_reasons: MU64::unsupported(),
            ..sample(20.0, 0)
        };
        assert_eq!(see(unmeasured), None);
    }
}
