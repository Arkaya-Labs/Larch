use bytemuck::Zeroable;
use larch_records::*;

fn header() -> RecordHeader {
    RecordHeader {
        tier: 1,
        ab_arm: 2,
        device: -1,
        t_mono_ns: 11,
        t_wall_ns: -12,
        engine_id: 13,
        config_rev: 14,
        ..RecordHeader::zeroed()
    }
}

/// What the ring stores for `header()`: the kind and layout of `R`, and seq 0.
fn stamped<R: RecordBody>() -> RecordHeader {
    RecordHeader {
        kind: R::KIND.raw(),
        layout: R::LAYOUT_VERSION,
        seq: 0,
        ..header()
    }
}

/// Pushes one body through a fresh ring and reads it back.
fn round_trip<R: RecordBody>(body: &R) -> Record {
    let (mut w, handle) = ring(2);
    let mut r = handle.reader(ReadFrom::Oldest);
    w.push(header(), body);
    match r.next() {
        Read::Record(rec) => rec,
        other => panic!("expected a record, got {other:?}"),
    }
}

#[test]
fn graph_capture_marker_preserves_legacy_records_and_round_trips() {
    let mut sample = EngineSample::zeroed();
    assert_eq!(sample.graph_captured(), None);
    assert_eq!(std::mem::size_of::<EngineSample>(), 208);
    assert_eq!(EngineSample::LAYOUT_VERSION, 2);
    for captured in [false, true] {
        sample.set_graph_captured(captured);
        assert_eq!(sample.graph_captured(), Some(captured));
        assert_eq!(sample._pad0[1], 0);
        let Record::EngineSample(_, decoded) = round_trip(&sample) else {
            panic!("expected engine sample");
        };
        assert_eq!(decoded.graph_captured(), Some(captured));
    }
    sample._pad0[0] = 255;
    assert_eq!(sample.graph_captured(), None);
}

#[test]
fn every_kind_round_trips() {
    let h = header();
    let engine = EngineSample {
        step: 1,
        gpu_ns: MU64::measured(2),
        phase: Phase::Mixed.raw(),
        kv_util: MF32::measured(0.5),
        sm_occupancy: MF32::unsupported(),
        ..EngineSample::zeroed()
    };
    let span = RequestSpan {
        req_id: 3,
        cached_tokens: 4,
        turn_id: 2,
        session_id: 5,
        class: RequestClass::Background.raw(),
        reuse_miss: ReuseMiss::Reclaimed.raw(),
        max_gap_ns: 7,
        reclaim_id: 6,
        outcome: Outcome::DeviceLost.raw(),
        finish: Finish::StopSeq.raw(),
        error: EngineError::DeviceLost.code(),
        ..RequestSpan::zeroed()
    };
    let device = DeviceSample {
        temp_c: MF32::measured(71.0),
        clocks_event_reasons: MU64::measured(1 << 3),
        sample_tick: 9,
        vram_total_bytes: MU64::measured(1 << 34),
        ..DeviceSample::zeroed()
    };
    let deploy = DeploymentSnapshot {
        model: FixedStr::from_str_truncating("a-model"),
        device_name: FixedStr::from_str_truncating("a-device"),
        backend: Backend::Cuda.raw(),
        arch_code: 120,
        fit_max_seqs: 64,
        prefill_chunk_max: 1024,
        ..DeploymentSnapshot::zeroed()
    };
    let host = HostSnapshot {
        git_sha: FixedStr::from_str_truncating("0123456789abcdef"),
        ..HostSnapshot::zeroed()
    };
    let rev = ConfigurationRevision {
        rev: 2,
        prev_rev: 1,
        knob: 3,
        cause: Cause::AbFlip.raw(),
        ..ConfigurationRevision::zeroed()
    };
    let act = ActivationEvent {
        knob: 4,
        result: ActivationResult::Rejected.raw(),
        reason: RejectReason::NotOnStep.code(),
        ..ActivationEvent::zeroed()
    };
    let reclaim = ReclaimEvent {
        reclaim_id: 6,
        session_id: 5,
        blocks: 2,
        tokens: 32,
        admitted_work_tokens: 17,
        reason: ReclaimReason::Demand.raw(),
        ..ReclaimEvent::zeroed()
    };
    let health = HealthEvent {
        from: HealthState::Loading.raw(),
        to: HealthState::Ready.raw(),
        fault: FaultCode::Timeout.raw(),
        detail: 0,
        recovery_ms: 1500,
    };

    assert_eq!(
        round_trip(&engine),
        Record::EngineSample(stamped::<EngineSample>(), engine)
    );
    assert_eq!(
        round_trip(&span),
        Record::RequestSpan(stamped::<RequestSpan>(), span)
    );
    assert_eq!(
        round_trip(&device),
        Record::DeviceSample(stamped::<DeviceSample>(), device)
    );
    assert_eq!(
        round_trip(&deploy),
        Record::DeploymentSnapshot(stamped::<DeploymentSnapshot>(), deploy)
    );
    assert_eq!(
        round_trip(&host),
        Record::HostSnapshot(stamped::<HostSnapshot>(), host)
    );
    assert_eq!(
        round_trip(&rev),
        Record::ConfigurationRevision(stamped::<ConfigurationRevision>(), rev)
    );
    assert_eq!(
        round_trip(&act),
        Record::ActivationEvent(stamped::<ActivationEvent>(), act)
    );
    assert_eq!(
        round_trip(&health),
        Record::HealthEvent(stamped::<HealthEvent>(), health)
    );
    assert_eq!(
        round_trip(&reclaim),
        Record::ReclaimEvent(stamped::<ReclaimEvent>(), reclaim)
    );

    assert_eq!(engine.phase(), Phase::Mixed);
    assert_eq!(span.outcome(), Outcome::DeviceLost);
    assert_eq!(span.finish(), Finish::StopSeq);
    assert_eq!(
        (span.class(), span.reuse_miss()),
        (RequestClass::Background, ReuseMiss::Reclaimed)
    );
    assert_eq!(reclaim.reason(), ReclaimReason::Demand);
    assert_eq!(deploy.backend(), Backend::Cuda);
    assert_eq!(host.git_sha.as_str(), "0123456789ab");
    assert_eq!(rev.cause(), Cause::AbFlip);
    assert_eq!(act.result(), ActivationResult::Rejected);
    assert_eq!(
        (health.from(), health.to(), health.fault()),
        (HealthState::Loading, HealthState::Ready, FaultCode::Timeout)
    );
    assert_eq!((h.tier(), h.ab_arm()), (Tier::T1, AbArm::B));
}

#[test]
fn unknown_raw_values_never_panic() {
    assert_eq!(Phase::from_raw(77), Phase::Unknown);
    assert_eq!(Outcome::from_raw(u8::MAX), Outcome::Unknown);
    assert_eq!(FaultCode::from_raw(1234), FaultCode::Unknown);
    assert_eq!(Backend::from_raw(9), Backend::Unknown);
    let span = RequestSpan {
        outcome: 200,
        finish: 201,
        ..RequestSpan::zeroed()
    };
    assert_eq!(
        (span.outcome(), span.finish()),
        (Outcome::Unknown, Finish::Unknown)
    );
}

#[test]
fn zeroed_records_are_honest() {
    let e = EngineSample::zeroed();
    for m in [
        e.kv_frag,
        e.kv_util,
        e.achieved_gbps,
        e.hbm_util,
        e.sm_occupancy,
    ] {
        assert_eq!(m.state(), MState::Missing);
    }
    assert_eq!(e.gpu_ns.get(), None);
    assert_eq!(HealthEvent::zeroed().fault(), FaultCode::None);
}

/// `use larch_records::*` must not shadow the prelude (`impl From` in strided, gait, cuprum).
#[test]
fn glob_import_keeps_the_prelude() {
    struct Wrapper(u64);
    impl From<u64> for Wrapper {
        fn from(v: u64) -> Self {
            Wrapper(v)
        }
    }
    assert_eq!(Wrapper::from(7).0, 7);
}
