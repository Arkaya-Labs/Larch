use bytemuck::{Pod, Zeroable};

use crate::{Backend, FixedStr};

/// One knob's schema (4.5). Values travel as `f64`: Bool is 0/1, Enum is the index.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct KnobSpec {
    pub id: u16,
    /// [`KnobType`].
    pub ty: u8,
    /// [`Unit`].
    pub unit: u8,
    /// [`ActivationClass`].
    pub activation: u8,
    /// [`RiskClass`].
    pub risk: u8,
    /// [`Inverse`].
    pub inverse: u8,
    /// Bitmask of `BACKEND_*`.
    pub backends: u8,
    pub name: FixedStr<32>,
    pub min: f64,
    pub max: f64,
    pub default: f64,
    /// Granularity from `min`; 0 = any value in range.
    pub step: f64,
    /// Knob ids; 0 = none. All zero for now (see `knob!`).
    pub conflicts: [u16; 4],
    /// Knob ids; 0 = none. All zero for now (see `knob!`).
    pub requires: [u16; 4],
    pub enum_names: [FixedStr<16>; 4],
}

const _: () = assert!(size_of::<KnobSpec>() == 152);

raw_enum! {
    pub enum KnobType: u8 {
        Bool = 0,
        U32 = 1,
        F32 = 2,
        Enum = 3,
    }
}

raw_enum! {
    pub enum Unit: u8 {
        None = 0,
        Tokens = 1,
        Seqs = 2,
        Requests = 3,
        Ratio = 4,
        Count = 5,
        Ms = 6,
    }
}

raw_enum! {
    /// When a change takes effect. Independent of [`RiskClass`] (issue #61).
    pub enum ActivationClass: u8 {
        RequestScoped = 0,
        Live = 1,
        ModelReload = 2,
        ProcessRestart = 3,
    }
}

raw_enum! {
    /// Who may change a knob (D7).
    pub enum RiskClass: u8 {
        PerformanceOnly = 0,
        ResourceOrAvailability = 1,
        QualityAffecting = 2,
    }
}

raw_enum! {
    /// What `KnobControl::revert` restores (9.8).
    pub enum Inverse: u8 {
        RestorePrevious = 0,
        RestoreDefault = 1,
    }
}

pub const BACKEND_CUDA: u8 = 1 << 0;
pub const BACKEND_HIP: u8 = 1 << 1;
pub const BACKEND_METAL: u8 = 1 << 2;
pub const BACKEND_CPU: u8 = 1 << 3;
pub const BACKEND_ALL: u8 = BACKEND_CUDA | BACKEND_HIP | BACKEND_METAL | BACKEND_CPU;

impl Backend {
    /// This backend's bit in [`KnobSpec::backends`]; 0 for `Unknown`.
    pub const fn mask(self) -> u8 {
        match self {
            Backend::Cuda => BACKEND_CUDA,
            Backend::Hip => BACKEND_HIP,
            Backend::Metal => BACKEND_METAL,
            Backend::Cpu => BACKEND_CPU,
            Backend::Unknown => 0,
        }
    }
}

impl KnobSpec {
    pub const fn ty(&self) -> KnobType {
        KnobType::from_raw(self.ty)
    }

    pub const fn unit(&self) -> Unit {
        Unit::from_raw(self.unit)
    }

    pub const fn activation(&self) -> ActivationClass {
        ActivationClass::from_raw(self.activation)
    }

    pub const fn risk(&self) -> RiskClass {
        RiskClass::from_raw(self.risk)
    }

    pub const fn inverse(&self) -> Inverse {
        Inverse::from_raw(self.inverse)
    }

    pub const fn supports(&self, backend: Backend) -> bool {
        self.backends & backend.mask() != 0
    }

    /// The names of an Enum knob's values, by index.
    pub fn enum_names(&self) -> impl Iterator<Item = &str> {
        self.enum_names
            .iter()
            .map(FixedStr::as_str)
            .take_while(|s| !s.is_empty())
    }
}

/// Knob ids; the index into [`REGISTRY`] is `id - 1`, and into `EffectiveConfig::values` is `id`.
pub mod knob_id {
    pub const MAX_BATCH_SEQS: u16 = 1;
    pub const PREFILL_CHUNK_TOKENS: u16 = 2;
    pub const SCHEDULE_POLICY: u16 = 3;
    pub const ATTN_DECODE_SPLITS: u16 = 4;
    pub const DECODE_GRAPHS: u16 = 5;
    pub const MAX_WAITING: u16 = 6;
    pub const KV_PREEMPT_WATERMARK: u16 = 7;
    pub const GUARDRAIL_RATIO: u16 = 8;
    pub const TELEMETRY_TIER: u16 = 9;
    pub const TEMPERATURE_DEFAULT: u16 = 10;
    pub const TOP_P_DEFAULT: u16 = 11;
    pub const TOP_K_DEFAULT: u16 = 12;
    pub const REPEAT_PENALTY_DEFAULT: u16 = 13;
    pub const KV_POOL_FRACTION: u16 = 14;
    pub const MAX_CONTEXT: u16 = 15;
    pub const DEVICE: u16 = 16;
}

/// One registry row: `knob!(id, name, type, unit, min..=max, default d, step s, activation, risk,
/// backends [, [enum names]])`. Step 0 means any value in range. No knob has conflicts or
/// requirements yet: the first one that needs them gets their meaning approved with it.
macro_rules! knob {
    ($id:expr, $name:literal, $ty:ident, $unit:ident, $min:literal..=$max:literal, default $default:literal,
     step $step:literal, $activation:ident, $risk:ident, $backends:expr $(, [$($enum:literal),+])?) => {
        KnobSpec {
            id: $id,
            ty: KnobType::$ty.raw(),
            unit: Unit::$unit.raw(),
            activation: ActivationClass::$activation.raw(),
            risk: RiskClass::$risk.raw(),
            inverse: Inverse::RestorePrevious.raw(),
            backends: $backends,
            name: FixedStr::from_str_truncating($name),
            min: $min as f64,
            max: $max as f64,
            default: $default as f64,
            step: $step as f64,
            conflicts: [0; 4],
            requires: [0; 4],
            enum_names: enum_names(&[$($($enum),+)?]),
        }
    };
}

const fn enum_names(names: &[&str]) -> [FixedStr<16>; 4] {
    let mut out = [FixedStr([0; 16]); 4];
    let mut i = 0;
    while i < names.len() {
        out[i] = FixedStr::from_str_truncating(names[i]);
        i += 1;
    }
    out
}

const GPU: u8 = BACKEND_CUDA | BACKEND_HIP | BACKEND_METAL;
const ALL: u8 = BACKEND_ALL;

/// The knobs for the 3 days (4.5), ordered by id. Shared by larch, strided, gait and cuprum.
/// `max_batch_seqs` is further clamped by fit (`ExceedsFit`); `attn_decode_splits` 0 = auto;
/// `decode_graphs` is rejected `Unsupported` where the backend has no graphs; `max_context` is
/// also bounded by the model at reload; `device` is reported only.
#[rustfmt::skip]
pub static REGISTRY: &[KnobSpec] = {
    use knob_id::*;
    &[
        knob!(MAX_BATCH_SEQS,         "max_batch_seqs",         U32,  Seqs,     1..=64,       default 16,   step 1,  Live,           PerformanceOnly,        ALL),
        knob!(PREFILL_CHUNK_TOKENS,   "prefill_chunk_tokens",   U32,  Tokens,   64..=8192,    default 512,  step 64, Live,           PerformanceOnly,        ALL),
        knob!(SCHEDULE_POLICY,        "schedule_policy",        Enum, None,     0..=2,        default 0,    step 1,  Live,           PerformanceOnly,        ALL,
              ["decode_first", "balanced", "prefill_first"]),
        knob!(ATTN_DECODE_SPLITS,     "attn_decode_splits",     U32,  Count,    0..=16,       default 0,    step 1,  Live,           PerformanceOnly,        GPU),
        knob!(DECODE_GRAPHS,          "decode_graphs",          Bool, None,     0..=1,        default 1,    step 1,  Live,           PerformanceOnly,        ALL),
        knob!(MAX_WAITING,            "max_waiting",            U32,  Requests, 0..=4096,     default 256,  step 1,  Live,           ResourceOrAvailability, ALL),
        knob!(KV_PREEMPT_WATERMARK,   "kv_preempt_watermark",   F32,  Ratio,    0.80..=0.99,  default 0.95, step 0,  Live,           ResourceOrAvailability, ALL),
        knob!(GUARDRAIL_RATIO,        "guardrail_ratio",        F32,  Ratio,    1.2..=3.0,    default 1.5,  step 0,  Live,           ResourceOrAvailability, ALL),
        knob!(TELEMETRY_TIER,         "telemetry_tier",         Enum, None,     0..=1,        default 1,    step 1,  Live,           PerformanceOnly,        ALL, ["t0", "t1"]),
        knob!(TEMPERATURE_DEFAULT,    "temperature_default",    F32,  None,     0..=2,        default 0.7,  step 0,  RequestScoped,  QualityAffecting,       ALL),
        knob!(TOP_P_DEFAULT,          "top_p_default",          F32,  Ratio,    0..=1,        default 0.95, step 0,  RequestScoped,  QualityAffecting,       ALL),
        knob!(TOP_K_DEFAULT,          "top_k_default",          U32,  Count,    0..=64,       default 40,   step 1,  RequestScoped,  QualityAffecting,       ALL),
        knob!(REPEAT_PENALTY_DEFAULT, "repeat_penalty_default", F32,  None,     1..=2,        default 1,    step 0,  RequestScoped,  QualityAffecting,       ALL),
        knob!(KV_POOL_FRACTION,       "kv_pool_fraction",       F32,  Ratio,    0.30..=0.95,  default 0.85, step 0,  ModelReload,    ResourceOrAvailability, ALL),
        knob!(MAX_CONTEXT,            "max_context",            U32,  Tokens,   512..=131072, default 8192, step 1,  ModelReload,    ResourceOrAvailability, ALL),
        knob!(DEVICE,                 "device",                 U32,  None,     0..=63,       default 0,    step 1,  ProcessRestart, ResourceOrAvailability, ALL),
    ]
};

/// The spec for a knob id, if it exists.
pub fn knob_spec(id: u16) -> Option<&'static KnobSpec> {
    REGISTRY.iter().find(|k| k.id == id)
}
