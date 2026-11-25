#![no_main]
use libfuzzer_sys::fuzz_target;
use larch_records::{Allow, Authority, Backend, HealthState, KnobRequest, REGISTRY, validate};

fuzz_target!(|data: &[u8]| {
    let Some(d) = data.first_chunk::<15>() else {
        return;
    };
    let mut knob = u16::from_le_bytes([d[1], d[2]]);
    // Even selector: land on a real knob so most inputs get past UnknownKnob to the deeper checks.
    if d[0] % 2 == 0 {
        knob = REGISTRY[knob as usize % REGISTRY.len()].id;
    }
    let value = f64::from_le_bytes(d[3..11].try_into().unwrap());
    let authority = match d[11] % 3 {
        0 => Authority::Human,
        1 => Authority::Gait,
        _ => Authority::Guardrail,
    };
    let mut req = KnobRequest::new(knob, value, authority);
    req.allow = Allow(d[12]);
    let res = validate(&req, Backend::from_raw(d[13]), HealthState::from_raw(d[14]));
    if let Ok(spec) = res {
        assert!(
            value.is_finite() && spec.min <= value && value <= spec.max,
            "validate accepted {value} for knob {}",
            spec.id
        );
    }
});
