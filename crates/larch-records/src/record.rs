use bytemuck::{Pod, Zeroable};

use crate::{FixedStr, MF32, MU64};

/// Bytes per ring slot: the header plus the body.
pub const SLOT_BYTES: usize = 256;
/// The slot as the ring stores it, in `u64` words.
pub const SLOT_WORDS: usize = SLOT_BYTES / 8;
pub const HEADER_BYTES: usize = size_of::<RecordHeader>();
/// The largest body that fits in a slot after the header.
pub const BODY_MAX: usize = SLOT_BYTES - HEADER_BYTES;

/// Common to every record (4.3). 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct RecordHeader {
    /// [`RecordKind`]; stamped by the ring from the body type.
    pub kind: u16,
    /// `LAYOUT_VERSION` of that kind; stamped by the ring from the body type.
    pub layout: u16,
    /// [`Tier`]: 0 = T0 always-on, 1 = T1 sampled.
    pub tier: u8,
    /// [`AbArm`]: 0 = no A/B, 1 = arm A, 2 = arm B.
    pub ab_arm: u8,
    /// Device ordinal, -1 = host/CPU.
    pub device: i16,
    /// Ring sequence, stamped by the ring.
    pub seq: u64,
    /// Nanoseconds since the engine's monotonic epoch (`Instant` captured at `Engine::load`).
    pub t_mono_ns: u64,
    /// Unix-epoch nanoseconds.
    pub t_wall_ns: i64,
    /// Random per process start (identity scope).
    pub engine_id: u64,
    /// `EffectiveConfig` revision in force.
    pub config_rev: u64,
}

impl RecordHeader {
    pub const fn kind(&self) -> RecordKind {
        RecordKind::from_raw(self.kind)
    }

    pub const fn tier(&self) -> Tier {
        Tier::from_raw(self.tier)
    }

    pub const fn ab_arm(&self) -> AbArm {
        AbArm::from_raw(self.ab_arm)
    }
}

raw_enum! {
    pub enum RecordKind: u16 {
        EngineSample = 1,
        RequestSpan = 2,
        DeviceSample = 3,
        DeploymentSnapshot = 4,
        HostSnapshot = 5,
        ConfigurationRevision = 6,
        ActivationEvent = 7,
        HealthEvent = 8,
        ReclaimEvent = 9,
    }
}

raw_enum! {
    pub enum Tier: u8 {
        /// Always on.
        T0 = 0,
        /// Sampled.
        T1 = 1,
    }
}

raw_enum! {
    pub enum AbArm: u8 {
        None = 0,
        A = 1,
        B = 2,
    }
}

raw_enum! {
    /// `EngineSample.phase`.
    pub enum Phase: u8 {
        Decode = 0,
        Prefill = 1,
        Mixed = 2,
    }
}

raw_enum! {
    /// `RequestSpan.outcome`.
    pub enum Outcome: u8 {
        Ok = 0,
        Cancelled = 1,
        Rejected = 2,
        Error = 3,
        DeviceLost = 4,
    }
}

raw_enum! {
    /// `RequestSpan.finish`.
    pub enum Finish: u8 {
        Eos = 0,
        Length = 1,
        StopSeq = 2,
        None = 3,
    }
}

raw_enum! {
    /// `RequestSpan.class`: the engine's request classes.
    pub enum RequestClass: u8 {
        Foreground = 0,
        Background = 1,
    }
}

raw_enum! {
    /// `RequestSpan.reuse_miss`: why a session turn did not reuse all of its prefix (S03's reasons).
    pub enum ReuseMiss: u8 {
        /// Reused all of the match, or not a session turn.
        None = 0,
        /// A demand reclaim took parked blocks the prompt matched.
        Reclaimed = 1,
        /// A reload or a KV invalidation dropped the parked sequence.
        Reloaded = 2,
        /// Parked metadata overflowed and the sequence was dropped.
        OverflowDrop = 3,
        /// The prompt diverged from the fed history inside it.
        PrefixDiverged = 4,
        /// Resident KV matched but was not reused.
        MatchFailure = 5,
        /// The session's first turn.
        FirstTurn = 6,
        /// The session was released, or its turn kept no KV.
        Reset = 7,
        /// The previous turn was cancelled.
        Cancelled = 8,
        /// The previous turn ended in an engine error.
        EngineError = 9,
    }
}

raw_enum! {
    /// `ReclaimEvent.reason`.
    pub enum ReclaimReason: u8 {
        /// Taken for admitted work.
        Demand = 0,
        /// Dropped by a model or pool reload.
        Reload = 1,
    }
}

raw_enum! {
    /// `DeploymentSnapshot.backend`; the same values as `larch_backend` in `larch.h`.
    pub enum Backend: u8 {
        Cpu = 0,
        Cuda = 1,
        Hip = 2,
        Metal = 3,
    }
}

raw_enum! {
    /// `ConfigurationRevision.cause`.
    pub enum Cause: u8 {
        Human = 0,
        Gait = 1,
        Guardrail = 2,
        AbFlip = 3,
        Inverse = 4,
        Reload = 5,
    }
}

raw_enum! {
    /// `ActivationEvent.result`.
    pub enum ActivationResult: u8 {
        Applied = 0,
        Rejected = 1,
        Indeterminate = 2,
    }
}

raw_enum! {
    /// `HealthEvent.from` / `.to`, and [`crate::Health::state`].
    pub enum HealthState: u8 {
        Loading = 0,
        Ready = 1,
        Degraded = 2,
        Failed = 3,
        Draining = 4,
    }
}

raw_enum! {
    /// `HealthEvent.fault`, and [`crate::Health::last_fault`].
    pub enum FaultCode: u16 {
        None = 0,
        Panic = 1,
        DeviceLost = 2,
        LaunchFailure = 3,
        Timeout = 4,
        Oom = 5,
        LoadError = 6,
    }
}

/// A record body that the ring can carry. The ring stamps `KIND` and `LAYOUT_VERSION` into the
/// header, so a body can never be pushed under the wrong kind.
pub trait RecordBody: Pod {
    const KIND: RecordKind;
    /// Bumped on any layout change, which is a contract change (4.1).
    const LAYOUT_VERSION: u16;
}

/// Engine, one per step. T1 (T0 keeps a 1-in-64 decimation).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct EngineSample {
    pub step: u64,
    /// Plan start to outputs ready.
    pub dur_ns: u64,
    /// From events: Measured on CUDA/HIP/Metal.
    pub gpu_ns: MU64,
    /// Scheduler + sampling CPU time.
    pub host_ns: u64,
    /// [`Phase`].
    pub phase: u8,
    pub graph_replayed: u8,
    #[cfg_attr(feature = "schema", serde(skip))]
    pub _pad0: [u8; 2],
    pub n_decode_seqs: u32,
    pub n_prefill_seqs: u32,
    pub n_prefill_tokens: u32,
    pub n_running: u32,
    pub n_waiting: u32,
    /// Preempted this step.
    pub n_preempted: u32,
    /// Parked conversations (D14).
    pub n_parked: u32,
    /// Blocks held parked (D14); included in `kv_blocks_used`.
    pub parked_blocks: u32,
    /// Blocks taken from parked sequences this step (D14).
    pub parked_reclaimed: u32,
    pub kv_blocks_total: u32,
    pub kv_blocks_used: u32,
    pub kv_block_tokens: u32,
    pub kv_tokens_used: u32,
    /// Measured: 1 − used_tokens / (used_blocks × block_tokens).
    pub kv_frag: MF32,
    /// Measured: used_blocks / total.
    pub kv_util: MF32,
    pub tot_preemptions: u64,
    pub tot_finished: u64,
    pub tot_prompt_tokens: u64,
    pub tot_gen_tokens: u64,
    pub tot_queue_ns: u64,
    pub tot_queued: u64,
    pub tot_rejected: u64,
    /// Estimated: bytes moved by the step's weights + KV reads / gpu_ns.
    pub achieved_gbps: MF32,
    /// Estimated: achieved / peak.
    pub hbm_util: MF32,
    /// Estimated from the occupancy API for the dominant decode kernel on CUDA/HIP;
    /// Unsupported on Metal.
    pub sm_occupancy: MF32,
    /// KV positions computed again after a preemption discarded them (S05).
    pub tot_recomputed_tokens: u64,
    /// Requests waiting while every seat is taken and the queue's head would fit the KV pool: what
    /// `max_batch_seqs`, and nothing else, holds back this step (S01).
    pub cap_blocked_ready: u32,
    #[cfg_attr(feature = "schema", serde(skip))]
    pub _pad1: [u8; 4],
}

impl EngineSample {
    pub const fn phase(&self) -> Phase {
        Phase::from_raw(self.phase)
    }

    /// Successful graph capture on this step. Legacy records leave the reserved byte zero and
    /// report unknown; the encoding consumes no new bytes or layout version (1=no, 2=yes).
    pub const fn graph_captured(&self) -> Option<bool> {
        match self._pad0[0] {
            1 => Some(false),
            2 => Some(true),
            _ => None,
        }
    }

    /// Marks whether this step captured a graph, in the reserved byte [`Self::graph_captured`] reads.
    pub fn set_graph_captured(&mut self, captured: bool) {
        self._pad0[0] = if captured { 2 } else { 1 };
    }
}

/// Engine, on request end. T0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct RequestSpan {
    pub req_id: u64,
    pub t_submit_ns: u64,
    pub t_admit_ns: u64,
    pub t_first_ns: u64,
    pub t_done_ns: u64,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    /// Prompt tokens reused from a parked sequence (D14).
    pub cached_tokens: u32,
    /// The session's turn, from 1; 0 outside a session.
    pub turn_id: u32,
    pub session_id: u64,
    pub preemptions: u16,
    /// [`Outcome`].
    pub outcome: u8,
    /// [`Finish`].
    pub finish: u8,
    /// [`crate::EngineError::code`], 0 = no error.
    pub error: u16,
    /// [`RequestClass`].
    pub class: u8,
    /// [`ReuseMiss`].
    pub reuse_miss: u8,
    /// The longest wait between two of its tokens, a preemption's re-prefill included.
    pub max_gap_ns: u64,
    /// Longest common prefix of the prompt with the session's fed history, or with the shared
    /// prefix cache's match when that is longer.
    pub lcp_tokens: u32,
    /// The parked sequence's KV tokens when the turn matched it, 0 if none.
    pub parked_kv_at_match: u32,
    /// Prompt tokens with resident, reusable KV: at most `lcp_tokens` and one less than the prompt.
    pub reuse_available_tokens: u32,
    /// 1 when the session's parked KV, not the shared cache, bounds `reuse_available_tokens`.
    pub prior_kv_applies: u8,
    #[cfg_attr(feature = "schema", serde(skip))]
    pub _pad0: [u8; 3],
    /// The last [`ReclaimEvent`] that took this session's parked blocks since its previous turn,
    /// 0 if none.
    pub reclaim_id: u64,
}

impl RequestSpan {
    pub const fn outcome(&self) -> Outcome {
        Outcome::from_raw(self.outcome)
    }

    pub const fn finish(&self) -> Finish {
        Finish::from_raw(self.finish)
    }

    /// `class` decoded.
    pub const fn class(&self) -> RequestClass {
        RequestClass::from_raw(self.class)
    }

    /// `reuse_miss` decoded.
    pub const fn reuse_miss(&self) -> ReuseMiss {
        ReuseMiss::from_raw(self.reuse_miss)
    }
}

/// Sampler thread → engine, 1 Hz. T1. NVML on Windows/Linux CUDA; Unsupported on Metal and HIP.
/// A field NVML refuses is Unsupported, one it fails to read Missing: never a stale value.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct DeviceSample {
    pub sm_clock_mhz: MF32,
    pub mem_clock_mhz: MF32,
    pub temp_c: MF32,
    pub power_w: MF32,
    /// Fraction of the last sample period a kernel ran, 0 to 1.
    pub gpu_util: MF32,
    pub vram_used_bytes: MU64,
    pub vram_total_bytes: MU64,
    /// The enforced power limit.
    pub power_limit_w: MF32,
    /// 1 while weights or KV spill to shared system memory (v2 spec 4.3, the B0 check), else 0.
    pub spill: MF32,
    /// NVML's clock-event (throttle) reasons bitmask.
    pub clocks_event_reasons: MU64,
    /// Compute processes on the device besides this one.
    pub other_procs: MU64,
    /// The sampler's seconds, from 1: consecutive at T1, every 64th at T0. 0 on the record each
    /// load and reload writes, which is not the sampler's. A reader tells a stalled sampler from a
    /// steady card by it.
    pub sample_tick: u64,
}

/// Engine, at load and reload. T0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct DeploymentSnapshot {
    pub model: FixedStr<64>,
    pub arch: FixedStr<16>,
    pub quant: FixedStr<16>,
    pub params_b: f32,
    pub n_layers: u16,
    pub n_heads: u16,
    pub n_kv_heads: u16,
    pub head_dim: u16,
    pub ctx_max: u32,
    pub vocab: u32,
    /// [`Backend`].
    pub backend: u8,
    pub device_name: FixedStr<48>,
    #[cfg_attr(feature = "schema", serde(skip))]
    pub _pad0: [u8; 3],
    /// sm 86/89/120, gfx 1030…, Apple family.
    pub arch_code: u32,
    pub peak_gbps: MF32,
    /// The largest `prefill_chunk_tokens` the scratch was sized for; 0 in records from before
    /// this field, which took the pad bytes it sits in (no layout change).
    pub prefill_chunk_max: u32,
    pub weights_bytes: u64,
    pub kv_pool_blocks: u32,
    pub fit_max_seqs: u32,
}

impl DeploymentSnapshot {
    pub const fn backend(&self) -> Backend {
        Backend::from_raw(self.backend)
    }
}

/// Engine, at load. T0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct HostSnapshot {
    pub os: FixedStr<16>,
    pub os_version: FixedStr<32>,
    pub cpu: FixedStr<48>,
    pub ram_bytes: u64,
    pub larch_version: FixedStr<16>,
    pub git_sha: FixedStr<12>,
    pub driver: FixedStr<16>,
    /// CUDA/Metal/HIP version.
    pub runtime: FixedStr<16>,
    #[cfg_attr(feature = "schema", serde(skip))]
    pub _pad0: [u8; 4],
}

/// Engine, on every applied change. T0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct ConfigurationRevision {
    pub rev: u64,
    pub prev_rev: u64,
    pub knob: u16,
    /// [`Cause`].
    pub cause: u8,
    #[cfg_attr(feature = "schema", serde(skip))]
    pub _pad0: [u8; 5],
    pub requested: f64,
    pub effective: f64,
    pub previous: f64,
    pub step_applied: u64,
    pub trial_id: u64,
}

impl ConfigurationRevision {
    pub const fn cause(&self) -> Cause {
        Cause::from_raw(self.cause)
    }
}

/// Engine, on every knob request outcome. T0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct ActivationEvent {
    pub knob: u16,
    /// [`ActivationResult`].
    pub result: u8,
    #[cfg_attr(feature = "schema", serde(skip))]
    pub _pad0: u8,
    /// [`crate::RejectReason::code`] or [`crate::IndeterminateReason::code`], by `result`.
    pub reason: u16,
    #[cfg_attr(feature = "schema", serde(skip))]
    pub _pad1: [u8; 2],
    pub requested: f64,
    pub effective: f64,
    pub t_request_ns: u64,
    pub t_applied_ns: u64,
    pub step: u64,
    pub trial_id: u64,
}

impl ActivationEvent {
    pub const fn result(&self) -> ActivationResult {
        ActivationResult::from_raw(self.result)
    }
}

/// Engine and watchdog. T0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct HealthEvent {
    /// [`HealthState`].
    pub from: u8,
    /// [`HealthState`].
    pub to: u8,
    /// [`FaultCode`].
    pub fault: u16,
    pub detail: u32,
    /// On the first Ready after a supervised restart: the previous process's fault → Ready,
    /// from `LARCH_RESTARTED_AT`.
    pub recovery_ms: u32,
}

/// Engine, when a demand reclaim takes a parked session's blocks or a reload drops them (S04). T0.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[cfg_attr(feature = "schema", derive(serde::Serialize, schemars::JsonSchema))]
pub struct ReclaimEvent {
    /// From 1 per engine; a resumed turn's `RequestSpan.reclaim_id` points back to it.
    pub reclaim_id: u64,
    /// The session whose parked blocks went.
    pub session_id: u64,
    /// KV blocks taken from it.
    pub blocks: u32,
    /// Fed tokens whose KV went with those blocks.
    pub tokens: u32,
    /// The tokens the growth that needed the blocks adds; every event of one growth carries it.
    pub admitted_work_tokens: u32,
    /// [`ReclaimReason`].
    pub reason: u8,
    #[cfg_attr(feature = "schema", serde(skip))]
    pub _pad0: [u8; 3],
}

impl ReclaimEvent {
    /// `reason` decoded.
    pub const fn reason(&self) -> ReclaimReason {
        ReclaimReason::from_raw(self.reason)
    }
}

impl HealthEvent {
    pub const fn from(&self) -> HealthState {
        HealthState::from_raw(self.from)
    }

    pub const fn to(&self) -> HealthState {
        HealthState::from_raw(self.to)
    }

    pub const fn fault(&self) -> FaultCode {
        FaultCode::from_raw(self.fault)
    }
}

/// Implements [`RecordBody`] for each kind (`Kind: size @ layout version`), checks its size at
/// compile time and generates the decode dispatch, so the list of kinds is written once.
macro_rules! record_bodies {
    ($($name:ident: $size:literal @ $version:literal),+ $(,)?) => {
        $(
            impl RecordBody for $name {
                const KIND: RecordKind = RecordKind::$name;
                const LAYOUT_VERSION: u16 = $version;
            }
            const _: () = assert!(size_of::<$name>() == $size && $size <= BODY_MAX);
        )+

        fn decode_body(h: RecordHeader, bytes: &[u8]) -> Record {
            match h.kind() {
                $(RecordKind::$name => body(h, bytes, Record::$name),)+
                RecordKind::Unknown => Record::Unknown(h),
            }
        }
    };
}

record_bodies! {
    EngineSample: 208 @ 2,
    RequestSpan: 104 @ 2,
    DeviceSample: 128 @ 2,
    DeploymentSnapshot: 200 @ 1,
    HostSnapshot: 168 @ 1,
    ConfigurationRevision: 64 @ 1,
    ActivationEvent: 56 @ 1,
    HealthEvent: 12 @ 1,
    ReclaimEvent: 32 @ 1,
}

const _: () = assert!(size_of::<RecordHeader>() == 48);

/// A decoded slot. This is a view for readers; the ring stores raw words.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Record {
    EngineSample(RecordHeader, EngineSample),
    RequestSpan(RecordHeader, RequestSpan),
    DeviceSample(RecordHeader, DeviceSample),
    DeploymentSnapshot(RecordHeader, DeploymentSnapshot),
    HostSnapshot(RecordHeader, HostSnapshot),
    ConfigurationRevision(RecordHeader, ConfigurationRevision),
    ActivationEvent(RecordHeader, ActivationEvent),
    HealthEvent(RecordHeader, HealthEvent),
    ReclaimEvent(RecordHeader, ReclaimEvent),
    /// A kind or layout version this build does not know (a newer producer).
    Unknown(RecordHeader),
}

impl Record {
    pub const fn header(&self) -> &RecordHeader {
        match self {
            Record::EngineSample(h, _)
            | Record::RequestSpan(h, _)
            | Record::DeviceSample(h, _)
            | Record::DeploymentSnapshot(h, _)
            | Record::HostSnapshot(h, _)
            | Record::ConfigurationRevision(h, _)
            | Record::ActivationEvent(h, _)
            | Record::HealthEvent(h, _)
            | Record::ReclaimEvent(h, _)
            | Record::Unknown(h) => h,
        }
    }

    /// Lays out a header and body as one slot. Only the ring calls this, after stamping `kind`,
    /// `layout` and `seq`, so a slot's header always matches its body.
    pub(crate) fn encode<R: RecordBody>(header: &RecordHeader, body: &R) -> [u64; SLOT_WORDS] {
        let mut words = [0u64; SLOT_WORDS];
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut words);
        bytes[..HEADER_BYTES].copy_from_slice(bytemuck::bytes_of(header));
        bytes[HEADER_BYTES..HEADER_BYTES + size_of::<R>()]
            .copy_from_slice(bytemuck::bytes_of(body));
        words
    }

    /// Decodes a slot. An unknown kind, or a layout version other than this build's, decodes to
    /// [`Record::Unknown`].
    pub(crate) fn decode(words: &[u64; SLOT_WORDS]) -> Record {
        let bytes: &[u8] = bytemuck::cast_slice(words);
        decode_body(bytemuck::pod_read_unaligned(&bytes[..HEADER_BYTES]), bytes)
    }
}

fn body<R: RecordBody>(
    h: RecordHeader,
    bytes: &[u8],
    wrap: fn(RecordHeader, R) -> Record,
) -> Record {
    if h.layout != R::LAYOUT_VERSION {
        return Record::Unknown(h);
    }
    wrap(
        h,
        bytemuck::pod_read_unaligned(&bytes[HEADER_BYTES..HEADER_BYTES + size_of::<R>()]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn health_slot(kind: u16, layout: u16) -> (RecordHeader, [u64; SLOT_WORDS]) {
        let h = RecordHeader {
            kind,
            layout,
            ..RecordHeader::zeroed()
        };
        (h, Record::encode(&h, &HealthEvent::zeroed()))
    }

    #[test]
    fn unknown_kind_or_layout_decodes_to_unknown() {
        let (h, slot) = health_slot(999, 1);
        assert_eq!(h.kind(), RecordKind::Unknown);
        assert_eq!(Record::decode(&slot), Record::Unknown(h));

        let (h, slot) = health_slot(
            RecordKind::HealthEvent.raw(),
            HealthEvent::LAYOUT_VERSION + 1,
        );
        assert_eq!(Record::decode(&slot), Record::Unknown(h));

        let (h, slot) = health_slot(RecordKind::HealthEvent.raw(), HealthEvent::LAYOUT_VERSION);
        assert_eq!(
            Record::decode(&slot),
            Record::HealthEvent(h, HealthEvent::zeroed())
        );
    }

    /// Records v2: this build reads a changed kind's layout 1 as Unknown, never as new fields.
    #[test]
    fn changed_kinds_decode_layout_1_as_unknown() {
        fn old<R: RecordBody>() {
            assert_eq!(R::LAYOUT_VERSION, 2);
            let h = RecordHeader {
                kind: R::KIND.raw(),
                layout: 1,
                ..RecordHeader::zeroed()
            };
            assert_eq!(
                Record::decode(&Record::encode(&h, &R::zeroed())),
                Record::Unknown(h)
            );
        }
        old::<EngineSample>();
        old::<RequestSpan>();
        old::<DeviceSample>();
    }
}
