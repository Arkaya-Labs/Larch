//! The registry of 4.5 and the validation order of 4.6 (row R8).

use larch_records::*;

#[test]
fn registry_has_the_sixteen_knobs_in_id_order() {
    let names: Vec<&str> = REGISTRY.iter().map(|k| k.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "max_batch_seqs",
            "prefill_chunk_tokens",
            "schedule_policy",
            "attn_decode_splits",
            "decode_graphs",
            "max_waiting",
            "kv_preempt_watermark",
            "guardrail_ratio",
            "telemetry_tier",
            "temperature_default",
            "top_p_default",
            "top_k_default",
            "repeat_penalty_default",
            "kv_pool_fraction",
            "max_context",
            "device",
        ]
    );
    for (i, k) in REGISTRY.iter().enumerate() {
        assert_eq!(k.id as usize, i + 1);
        assert_eq!(knob_spec(k.id), Some(k));
    }
    assert_eq!(knob_spec(0), None);
    assert_eq!(knob_spec(17), None);
}

/// Their meaning is not defined yet; the first knob that needs one gets it approved with it.
#[test]
fn no_knob_has_conflicts_or_requirements() {
    assert!(
        REGISTRY
            .iter()
            .all(|k| k.conflicts == [0; 4] && k.requires == [0; 4])
    );
}

#[test]
fn every_default_passes_validation_shape_checks() {
    for k in REGISTRY {
        assert!(k.min <= k.default && k.default <= k.max, "{}", k.name);
        if k.step > 0.0 {
            let n = (k.default - k.min) / k.step;
            assert_eq!(n, n.round(), "{} default off step", k.name);
        }
        match k.ty() {
            KnobType::Enum => assert_eq!(k.enum_names().count() as f64, k.max + 1.0, "{}", k.name),
            KnobType::Bool => assert_eq!((k.min, k.max), (0.0, 1.0)),
            _ => assert_eq!(k.enum_names().count(), 0),
        }
    }
}

fn check(req: KnobRequest) -> Result<u16, RejectReason> {
    check_on(req, Backend::Cuda, HealthState::Ready)
}

fn check_on(req: KnobRequest, backend: Backend, health: HealthState) -> Result<u16, RejectReason> {
    validate(&req, backend, health).map(|k| k.id)
}

fn human(knob: u16, value: f64) -> KnobRequest {
    KnobRequest::new(knob, value, Authority::Human)
}

fn gait(knob: u16, value: f64) -> KnobRequest {
    KnobRequest::new(knob, value, Authority::Gait)
}

#[test]
fn accepts_valid_requests() {
    assert_eq!(
        check(human(knob_id::MAX_BATCH_SEQS, 32.0)),
        Ok(knob_id::MAX_BATCH_SEQS)
    );
    assert_eq!(
        check(gait(knob_id::PREFILL_CHUNK_TOKENS, 1024.0)),
        Ok(knob_id::PREFILL_CHUNK_TOKENS)
    );
    assert_eq!(
        check(gait(knob_id::SCHEDULE_POLICY, 2.0)),
        Ok(knob_id::SCHEDULE_POLICY)
    );
    assert_eq!(
        check(gait(knob_id::DECODE_GRAPHS, 0.0)),
        Ok(knob_id::DECODE_GRAPHS)
    );
    assert_eq!(
        check(human(knob_id::KV_PREEMPT_WATERMARK, 0.9)),
        Ok(knob_id::KV_PREEMPT_WATERMARK)
    );
    assert_eq!(
        check(human(knob_id::TEMPERATURE_DEFAULT, 0.0)),
        Ok(knob_id::TEMPERATURE_DEFAULT)
    );
    // Bounds are inclusive.
    assert_eq!(
        check(human(knob_id::MAX_BATCH_SEQS, 64.0)),
        Ok(knob_id::MAX_BATCH_SEQS)
    );
    assert_eq!(
        check(human(knob_id::PREFILL_CHUNK_TOKENS, 8192.0)),
        Ok(knob_id::PREFILL_CHUNK_TOKENS)
    );
}

#[test]
fn unknown_knob() {
    assert_eq!(check(human(0, 1.0)), Err(RejectReason::UnknownKnob));
    assert_eq!(check(human(17, 1.0)), Err(RejectReason::UnknownKnob));
}

#[test]
fn wrong_type() {
    for (knob, v) in [
        (knob_id::DECODE_GRAPHS, 0.5),
        (knob_id::DECODE_GRAPHS, 2.0),
        (knob_id::MAX_BATCH_SEQS, 1.5),
        (knob_id::MAX_BATCH_SEQS, -1.0),
        (knob_id::MAX_BATCH_SEQS, f64::NAN),
        (knob_id::MAX_BATCH_SEQS, f64::INFINITY),
        (knob_id::SCHEDULE_POLICY, 0.5),
        (knob_id::GUARDRAIL_RATIO, f64::NAN),
        (knob_id::GUARDRAIL_RATIO, f64::NEG_INFINITY),
    ] {
        assert_eq!(
            check(human(knob, v)),
            Err(RejectReason::WrongType),
            "{knob} = {v}"
        );
    }
}

#[test]
fn not_on_step() {
    assert_eq!(
        check(human(knob_id::PREFILL_CHUNK_TOKENS, 100.0)),
        Err(RejectReason::NotOnStep)
    );
    // Step is checked before bounds.
    assert_eq!(
        check(human(knob_id::PREFILL_CHUNK_TOKENS, 9000.0)),
        Err(RejectReason::NotOnStep)
    );
}

#[test]
fn out_of_range() {
    assert_eq!(
        check(human(knob_id::MAX_BATCH_SEQS, 0.0)),
        Err(RejectReason::OutOfRange {
            min: 1.0,
            max: 64.0
        })
    );
    assert_eq!(
        check(human(knob_id::PREFILL_CHUNK_TOKENS, 8256.0)),
        Err(RejectReason::OutOfRange {
            min: 64.0,
            max: 8192.0
        })
    );
    assert_eq!(
        check(human(knob_id::SCHEDULE_POLICY, 3.0)),
        Err(RejectReason::OutOfRange { min: 0.0, max: 2.0 })
    );
    assert_eq!(
        check(human(knob_id::KV_PREEMPT_WATERMARK, 0.5)),
        Err(RejectReason::OutOfRange {
            min: 0.80,
            max: 0.99
        })
    );
}

#[test]
fn not_live() {
    assert_eq!(
        check(human(knob_id::KV_POOL_FRACTION, 0.5)),
        Err(RejectReason::NotLive {
            class: ActivationClass::ModelReload
        })
    );
    assert_eq!(
        check(human(knob_id::MAX_CONTEXT, 4096.0)),
        Err(RejectReason::NotLive {
            class: ActivationClass::ModelReload
        })
    );
    assert_eq!(
        check(human(knob_id::DEVICE, 1.0)),
        Err(RejectReason::NotLive {
            class: ActivationClass::ProcessRestart
        })
    );
}

#[test]
fn authority_vs_risk() {
    let quality = Err(RejectReason::Unauthorized {
        risk: RiskClass::QualityAffecting,
    });
    assert_eq!(check(gait(knob_id::TEMPERATURE_DEFAULT, 0.5)), quality);
    let mut with_resource = gait(knob_id::TEMPERATURE_DEFAULT, 0.5);
    with_resource.allow = Allow::RESOURCE;
    assert_eq!(
        check(with_resource),
        quality,
        "RESOURCE never unlocks quality knobs"
    );

    let resource = Err(RejectReason::Unauthorized {
        risk: RiskClass::ResourceOrAvailability,
    });
    assert_eq!(check(gait(knob_id::MAX_WAITING, 128.0)), resource);
    let mut allowed = gait(knob_id::MAX_WAITING, 128.0);
    allowed.allow = Allow::RESOURCE;
    assert_eq!(check(allowed), Ok(knob_id::MAX_WAITING));

    assert_eq!(
        check(human(knob_id::TEMPERATURE_DEFAULT, 0.5)),
        Ok(knob_id::TEMPERATURE_DEFAULT)
    );
    assert_eq!(
        check(human(knob_id::MAX_WAITING, 128.0)),
        Ok(knob_id::MAX_WAITING)
    );
    let guardrail = KnobRequest::new(knob_id::MAX_WAITING, 128.0, Authority::Guardrail);
    assert_eq!(check(guardrail), Ok(knob_id::MAX_WAITING));
}

#[test]
fn backend_support() {
    let splits = human(knob_id::ATTN_DECODE_SPLITS, 4.0);
    assert_eq!(
        check_on(splits, Backend::Cpu, HealthState::Ready),
        Err(RejectReason::Unsupported)
    );
    for b in [Backend::Cuda, Backend::Hip, Backend::Metal] {
        assert_eq!(
            check_on(splits, b, HealthState::Ready),
            Ok(knob_id::ATTN_DECODE_SPLITS)
        );
    }
    assert_eq!(
        check_on(
            human(knob_id::MAX_BATCH_SEQS, 4.0),
            Backend::Unknown,
            HealthState::Ready
        ),
        Err(RejectReason::Unsupported)
    );
}

#[test]
fn health_must_be_ready() {
    for state in [
        HealthState::Loading,
        HealthState::Degraded,
        HealthState::Failed,
        HealthState::Draining,
    ] {
        assert_eq!(
            check_on(human(knob_id::MAX_BATCH_SEQS, 4.0), Backend::Cuda, state),
            Err(RejectReason::NotReady)
        );
    }
    // Static checks come first: a bad request is reported as bad even while not ready.
    assert_eq!(
        check_on(
            human(knob_id::MAX_BATCH_SEQS, 0.0),
            Backend::Cuda,
            HealthState::Failed
        ),
        Err(RejectReason::OutOfRange {
            min: 1.0,
            max: 64.0
        })
    );
}

#[test]
fn effective_config_defaults() {
    let cfg = EffectiveConfig::defaults(REGISTRY);
    assert_eq!(cfg.rev, 0);
    assert_eq!(cfg.values[knob_id::MAX_BATCH_SEQS as usize], 16.0);
    assert_eq!(cfg.values[knob_id::DECODE_GRAPHS as usize], 1.0);
    let req = KnobRequest::new(knob_id::MAX_BATCH_SEQS, 8.0, Authority::Gait);
    assert_eq!(
        (req.allow, req.trial_id, req.timeout_ms),
        (Allow::NONE, 0, DEFAULT_TIMEOUT_MS)
    );
}
