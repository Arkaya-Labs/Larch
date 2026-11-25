//! Writes `larch-records.schema.json` (4.8): the JSON Schema of every contract type, for cuprum
//! and any future transport. Usage: `records-schema [OUT_PATH]`.

use larch_records::*;
use schemars::SchemaGenerator;

fn main() -> std::io::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "larch-records.schema.json".into());
    let mut generator = SchemaGenerator::default();
    macro_rules! define {
        ($($t:ty),+ $(,)?) => {$( generator.subschema_for::<$t>(); )+};
    }
    define!(
        RecordHeader,
        EngineSample,
        RequestSpan,
        DeviceSample,
        DeploymentSnapshot,
        HostSnapshot,
        ConfigurationRevision,
        ActivationEvent,
        HealthEvent,
        ReclaimEvent,
        RecordKind,
        Tier,
        AbArm,
        MState,
        MF32,
        MF64,
        MU64,
        Phase,
        Outcome,
        Finish,
        RequestClass,
        ReuseMiss,
        ReclaimReason,
        Backend,
        Cause,
        ActivationResult,
        HealthState,
        FaultCode,
        KnobSpec,
        KnobType,
        Unit,
        ActivationClass,
        RiskClass,
        Inverse,
        KnobRequest,
        KnobReply,
        RevertTarget,
        AbCommand,
        AbReply,
        EffectiveConfig,
        Health,
        EngineError,
        LoadError,
    );
    let schema = serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "larch-records",
        "$defs": generator.definitions(),
    });
    let mut text = serde_json::to_string_pretty(&schema)?;
    text.push('\n');
    std::fs::write(&path, text)?;
    println!("wrote {path}");
    Ok(())
}
