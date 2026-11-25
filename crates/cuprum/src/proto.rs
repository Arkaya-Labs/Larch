//! The four router/worker messages. Payloads reuse the `larch-records` serde mirrors, so the
//! schema stays shared with 4.8.
// ponytail: serialize-only, because the records mirrors have no `Deserialize`; add the decode
// side when a transport exists and records grows it (a contract change).

use larch_records::{Backend, DeploymentSnapshot, FixedStr, KnobRequest, Record, RecordHeader};
use schemars::{JsonSchema, SchemaGenerator};
use serde::Serialize;

/// Router-assigned worker identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, JsonSchema)]
pub struct WorkerId(pub u64);

/// A generation request on the wire; the same fields as the engine's `Request` that a worker needs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, JsonSchema)]
pub struct RequestWire {
    pub prompt: Vec<u32>,
    pub max_tokens: u32,
    pub stop_tokens: Vec<u32>,
}

/// One ring record on the wire: a header and its decoded body.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, JsonSchema)]
pub enum RecordWire {
    EngineSample(RecordHeader, larch_records::EngineSample),
    RequestSpan(RecordHeader, larch_records::RequestSpan),
    DeviceSample(RecordHeader, larch_records::DeviceSample),
    DeploymentSnapshot(RecordHeader, DeploymentSnapshot),
    HostSnapshot(RecordHeader, larch_records::HostSnapshot),
    ConfigurationRevision(RecordHeader, larch_records::ConfigurationRevision),
    ActivationEvent(RecordHeader, larch_records::ActivationEvent),
    HealthEvent(RecordHeader, larch_records::HealthEvent),
    ReclaimEvent(RecordHeader, larch_records::ReclaimEvent),
    /// A kind or layout version the sender did not know.
    Unknown(RecordHeader),
}

impl From<Record> for RecordWire {
    fn from(r: Record) -> Self {
        match r {
            Record::EngineSample(h, b) => Self::EngineSample(h, b),
            Record::RequestSpan(h, b) => Self::RequestSpan(h, b),
            Record::DeviceSample(h, b) => Self::DeviceSample(h, b),
            Record::DeploymentSnapshot(h, b) => Self::DeploymentSnapshot(h, b),
            Record::HostSnapshot(h, b) => Self::HostSnapshot(h, b),
            Record::ConfigurationRevision(h, b) => Self::ConfigurationRevision(h, b),
            Record::ActivationEvent(h, b) => Self::ActivationEvent(h, b),
            Record::HealthEvent(h, b) => Self::HealthEvent(h, b),
            Record::ReclaimEvent(h, b) => Self::ReclaimEvent(h, b),
            Record::Unknown(h) => Self::Unknown(h),
        }
    }
}

/// Worker to router, once at connect.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, JsonSchema)]
pub struct WorkerHello {
    pub engine_id: u64,
    pub backend: Backend,
    pub device_name: FixedStr<48>,
    pub deployment: DeploymentSnapshot,
}

/// Router to worker: run this request on `target`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, JsonSchema)]
pub struct RouteRequest {
    pub request: RequestWire,
    pub target: WorkerId,
}

/// Router to worker: forward a knob change to the engine's `KnobControl`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, JsonSchema)]
pub struct KnobForward {
    pub req: KnobRequest,
}

/// Worker to router: records drained from the ring.
#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub struct TelemetryBatch {
    pub records: Vec<RecordWire>,
}

/// JSON Schema of the four messages; the records definitions they reference are included.
pub fn schema() -> serde_json::Value {
    let mut generator = SchemaGenerator::default();
    generator.subschema_for::<WorkerHello>();
    generator.subschema_for::<RouteRequest>();
    generator.subschema_for::<KnobForward>();
    generator.subschema_for::<TelemetryBatch>();
    serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "cuprum",
        "$defs": generator.definitions(),
    })
}
