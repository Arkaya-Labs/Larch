use cuprum::proto::*;
use larch_records::{Authority, KnobRequest, Record, RecordHeader};

#[test]
fn schema_has_messages_and_shared_records_defs() {
    let schema = schema();
    let defs = schema["$defs"].as_object().unwrap();
    for name in [
        "WorkerHello",
        "RouteRequest",
        "KnobForward",
        "TelemetryBatch",
        "DeploymentSnapshot",
        "RequestWire",
        "WorkerId",
        "RecordWire",
        "RecordHeader",
        "KnobRequest",
        "EngineSample",
    ] {
        assert!(defs.contains_key(name), "missing {name}");
    }
}

fn refs(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, x) in m {
                match x.as_str() {
                    Some(r) if k == "$ref" => out.push(r.to_owned()),
                    _ => refs(x, out),
                }
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| refs(x, out)),
        _ => {}
    }
}

#[test]
fn every_schema_ref_resolves() {
    let schema = schema();
    let mut found = Vec::new();
    refs(&schema, &mut found);
    assert!(!found.is_empty());
    for r in found {
        let name = r
            .strip_prefix("#/$defs/")
            .unwrap_or_else(|| panic!("odd ref {r}"));
        assert!(schema["$defs"].get(name).is_some(), "dangling {r}");
    }
}

#[test]
fn route_request_serializes() {
    let msg = RouteRequest {
        request: RequestWire {
            prompt: vec![1, 2],
            max_tokens: 8,
            stop_tokens: vec![],
        },
        target: WorkerId(3),
    };
    let json = serde_json::to_value(msg).unwrap();
    assert_eq!(json["target"], 3);
    assert_eq!(json["request"]["max_tokens"], 8);
}

#[test]
fn messages_serialize() {
    let fwd = KnobForward {
        req: KnobRequest::new(1, 4.0, Authority::Gait),
    };
    assert_eq!(serde_json::to_value(fwd).unwrap()["req"]["knob"], 1);

    let header = RecordHeader {
        kind: 0,
        layout: 0,
        tier: 0,
        ab_arm: 0,
        device: -1,
        seq: 7,
        t_mono_ns: 0,
        t_wall_ns: 0,
        engine_id: 0,
        config_rev: 0,
    };
    let batch = TelemetryBatch {
        records: vec![Record::Unknown(header).into()],
    };
    let json = serde_json::to_value(batch).unwrap();
    assert_eq!(json["records"][0]["Unknown"]["seq"], 7);
}
