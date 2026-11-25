//! The knob API (4.6): the trait strided, gait and the CLI call, implemented by larch-engine.

use std::sync::Arc;

use crate::{
    ActivationClass, Backend, FaultCode, HealthState, KnobSpec, KnobType, RiskClass, knob_spec,
};

pub trait KnobControl: Send + Sync {
    fn registry(&self) -> &'static [KnobSpec];
    /// An arc-swap load; never blocks.
    fn effective(&self) -> Arc<EffectiveConfig>;
    /// Blocks for at most `req.timeout_ms`.
    fn set(&self, req: KnobRequest) -> KnobReply;
    /// Applies the declared inverse (9.8).
    fn revert(&self, target: RevertTarget, authority: Authority) -> KnobReply;
    fn ab(&self, cmd: AbCommand) -> AbReply;
    fn health(&self) -> Health;
}

pub const DEFAULT_TIMEOUT_MS: u32 = 250;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct KnobRequest {
    pub knob: u16,
    pub value: f64,
    pub authority: Authority,
    pub allow: Allow,
    pub trial_id: u64,
    pub timeout_ms: u32,
}

impl KnobRequest {
    /// No extra allowances, no trial, the default timeout.
    pub const fn new(knob: u16, value: f64, authority: Authority) -> Self {
        Self {
            knob,
            value,
            authority,
            allow: Allow::NONE,
            trial_id: 0,
            timeout_ms: DEFAULT_TIMEOUT_MS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub enum Authority {
    Human,
    Gait,
    Guardrail,
}

/// Bitflags a caller grants with a request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct Allow(pub u8);

impl Allow {
    pub const NONE: Allow = Allow(0);
    /// Gait may change `ResourceOrAvailability` knobs (`--allow-resource`, D7).
    pub const RESOURCE: Allow = Allow(1 << 0);

    pub const fn contains(self, other: Allow) -> bool {
        self.0 & other.0 == other.0
    }
}

/// One revision, or everything done under a trial id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub enum RevertTarget {
    Rev(u64),
    Trial(u64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub enum KnobReply {
    Applied {
        rev: u64,
        effective: f64,
        previous: f64,
        step: u64,
        latency_us: u32,
    },
    Rejected {
        reason: RejectReason,
    },
    Indeterminate {
        reason: IndeterminateReason,
        rev: Option<u64>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub enum RejectReason {
    UnknownKnob,
    WrongType,
    OutOfRange { min: f64, max: f64 },
    NotOnStep,
    Conflict { with: u16 },
    MissingDependency { needs: u16 },
    NotLive { class: ActivationClass },
    Unauthorized { risk: RiskClass },
    ExceedsFit { limit: f64 },
    Unsupported,
    NotReady,
    Busy,
    Superseded { by_rev: u64 },
    NothingToRevert,
}

impl RejectReason {
    /// The stable code written to `ActivationEvent.reason` when `result` is Rejected.
    pub const fn code(&self) -> u16 {
        match self {
            RejectReason::UnknownKnob => 1,
            RejectReason::WrongType => 2,
            RejectReason::OutOfRange { .. } => 3,
            RejectReason::NotOnStep => 4,
            RejectReason::Conflict { .. } => 5,
            RejectReason::MissingDependency { .. } => 6,
            RejectReason::NotLive { .. } => 7,
            RejectReason::Unauthorized { .. } => 8,
            RejectReason::ExceedsFit { .. } => 9,
            RejectReason::Unsupported => 10,
            RejectReason::NotReady => 11,
            RejectReason::Busy => 12,
            RejectReason::Superseded { .. } => 13,
            RejectReason::NothingToRevert => 14,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub enum IndeterminateReason {
    TimedOutBeforeStep,
    FaultDuringApply,
    EngineRestarted,
}

impl IndeterminateReason {
    /// The stable code written to `ActivationEvent.reason` when `result` is Indeterminate.
    pub const fn code(&self) -> u16 {
        match self {
            IndeterminateReason::TimedOutBeforeStep => 1,
            IndeterminateReason::FaultDuringApply => 2,
            IndeterminateReason::EngineRestarted => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub enum AbCommand {
    /// Arm B's deltas are checked as a `set` under `authority` and `allow` would be.
    Start {
        b: [KnobDelta; 4],
        period_steps: u32,
        max_steps: u64,
        trial_id: u64,
        authority: Authority,
        allow: Allow,
    },
    Stop {
        trial_id: u64,
    },
}

/// Knob 0 marks an unused slot (ids start at 1).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct KnobDelta {
    pub knob: u16,
    pub value: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub enum AbReply {
    Started { arm_a_rev: u64, step: u64 },
    Stopped { steps: u64, end: AbEnd },
    Rejected { reason: RejectReason },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub enum AbEnd {
    MaxSteps,
    Stop,
    Guardrail,
    Revert,
    Fault,
}

/// The live configuration. `values` is indexed by knob id (index 0 is unused).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct EffectiveConfig {
    pub rev: u64,
    pub values: [f64; 32],
    pub ab_arm: u8,
}

impl EffectiveConfig {
    /// Revision 0: every knob in `registry` at its default.
    pub fn defaults(registry: &[KnobSpec]) -> Self {
        let mut values = [0.0; 32];
        for k in registry {
            values[k.id as usize] = k.default;
        }
        Self {
            rev: 0,
            values,
            ab_arm: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct Health {
    pub state: HealthState,
    pub since_mono_ns: u64,
    pub last_fault: FaultCode,
    pub restarts: u32,
}

/// The checks `KnobControl::set` runs on the caller's thread before anything is queued, in the
/// order of 4.6: known knob, type, step, bounds, activation, authority vs risk, backend, health.
/// Dynamic preconditions (e.g. `ExceedsFit`) are the engine's, at the step boundary.
pub fn validate(
    req: &KnobRequest,
    backend: Backend,
    health: HealthState,
) -> Result<&'static KnobSpec, RejectReason> {
    let spec = knob_spec(req.knob).ok_or(RejectReason::UnknownKnob)?;
    let v = req.value;

    let well_typed = match spec.ty() {
        KnobType::Bool => v == 0.0 || v == 1.0,
        KnobType::U32 | KnobType::Enum => {
            v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v <= u32::MAX as f64
        }
        KnobType::F32 => v.is_finite(),
        KnobType::Unknown => false,
    };
    if !well_typed {
        return Err(RejectReason::WrongType);
    }
    if spec.step > 0.0 {
        let k = (v - spec.min) / spec.step;
        if (k - k.round()).abs() > 1e-9 {
            return Err(RejectReason::NotOnStep);
        }
    }
    if v < spec.min || v > spec.max {
        return Err(RejectReason::OutOfRange {
            min: spec.min,
            max: spec.max,
        });
    }

    match spec.activation() {
        ActivationClass::Live | ActivationClass::RequestScoped => {}
        class => return Err(RejectReason::NotLive { class }),
    }

    let risk = spec.risk();
    if req.authority == Authority::Gait {
        let allowed = match risk {
            RiskClass::PerformanceOnly => true,
            RiskClass::ResourceOrAvailability => req.allow.contains(Allow::RESOURCE),
            RiskClass::QualityAffecting | RiskClass::Unknown => false,
        };
        if !allowed {
            return Err(RejectReason::Unauthorized { risk });
        }
    }

    if !spec.supports(backend) {
        return Err(RejectReason::Unsupported);
    }
    if health != HealthState::Ready {
        return Err(RejectReason::NotReady);
    }
    Ok(spec)
}
