use chrono::{Duration, TimeZone, Utc};
use larch_task::{
    admission::Request,
    schedule::{Policy, Schedule},
    store::SqliteStore,
    types::{PlanningPolicy, SCHEMA_VERSION, WorkflowDefinition},
    watch::Truth,
};
use std::collections::BTreeMap;
fn path() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("larch-trigger-{}-{}.sqlite", std::process::id(), {
        // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{t}-{n}")
    }))
}
fn definition() -> WorkflowDefinition {
    WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "test".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: Vec::new(),
        budgets: BTreeMap::new(),
        acceptance: Vec::new(),
    }
}
#[test]
fn trigger_watch_and_reservation_survive_reopen() {
    let file = path();
    let anchor = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    {
        let mut s = SqliteStore::open(&file).unwrap();
        s.create_schedule(
            "clock",
            &Schedule::Interval {
                anchor,
                seconds: 60,
            },
            Policy::default(),
            &definition(),
            anchor,
        )
        .unwrap();
        s.enable_schedule("clock", 0, true, anchor).unwrap();
        let first = s.pump_schedule("clock", 1, anchor, 100).unwrap();
        assert_eq!(first.len(), 1);
        assert!(s.pump_schedule("clock", 0, anchor, 100).is_err());
        assert!(
            s.pump_schedule("clock", 2, anchor + Duration::minutes(1), 100)
                .unwrap()
                .is_empty()
        );
        assert_eq!(s.pending_outbox(10).unwrap().len(), 1);
        s.create_watch("watch").unwrap();
        s.update_watch("watch", 0, Truth::True, Some("cursor".into()))
            .unwrap();
        s.create_admission_root("root", 10).unwrap();
        s.reserve_permit(
            &Request {
                id: "permit".into(),
                root: "root".into(),
                reserve: 8,
            },
            1,
        )
        .unwrap();
    }
    {
        let mut s = SqliteStore::open(&file).unwrap();
        assert_eq!(s.pending_outbox(10).unwrap().len(), 1);
        assert_eq!(s.list().unwrap().len(), 1);
        assert_eq!(
            s.watch("watch").unwrap().unwrap().cursor.as_deref(),
            Some("cursor")
        );
        assert!(s.update_watch("watch", 0, Truth::False, None).is_err());
        s.update_watch("watch", 1, Truth::Unknown, Some("bad".into()))
            .unwrap();
        assert_eq!(
            s.watch("watch").unwrap().unwrap().cursor.as_deref(),
            Some("cursor")
        );
        assert!(
            s.reserve_permit(
                &Request {
                    id: "another".into(),
                    root: "root".into(),
                    reserve: 1,
                },
                1,
            )
            .is_err()
        );
        s.settle_permit("permit", 4).unwrap();
        s.reserve_permit(
            &Request {
                id: "another".into(),
                root: "root".into(),
                reserve: 6,
            },
            1,
        )
        .unwrap();
    }
    let _ = std::fs::remove_file(file);
}
#[test]
fn child_persistence() {
    let Ok(file) = std::env::var("LARCH_TEST_CHILD_DB") else {
        return;
    };
    let anchor = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let mut s = SqliteStore::open(file).unwrap();
    s.create_schedule(
        "child",
        &Schedule::Once { at: anchor },
        Policy::default(),
        &definition(),
        anchor,
    )
    .unwrap();
    s.enable_schedule("child", 0, true, anchor).unwrap();
    s.pump_schedule("child", 1, anchor, 10).unwrap();
    s.create_admission_root("root", 20).unwrap();
    s.reserve_permit(
        &Request {
            id: "inflight".into(),
            root: "root".into(),
            reserve: 10,
        },
        1,
    )
    .unwrap();
}
#[test]
fn committed_trigger_and_unknown_reservation_survive_process_exit() {
    let file = path();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_persistence"])
        .env("LARCH_TEST_CHILD_DB", &file)
        .status()
        .unwrap();
    assert!(status.success());
    let mut s = SqliteStore::open(&file).unwrap();
    assert_eq!(s.list().unwrap().len(), 1);
    assert_eq!(s.pending_outbox(10).unwrap().len(), 1);
    assert!(
        s.reserve_permit(
            &Request {
                id: "second".into(),
                root: "root".into(),
                reserve: 1,
            },
            1,
        )
        .is_err()
    );
    s.settle_permit("inflight", 10).unwrap();
    drop(s);
    let _ = std::fs::remove_file(file);
}
#[test]
fn pump_due_drives_independent_schedules() {
    let file = path();
    let anchor = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let mut s = SqliteStore::open(&file).unwrap();
    for id in ["a", "b"] {
        s.create_schedule(
            id,
            &Schedule::Once { at: anchor },
            Policy::default(),
            &definition(),
            anchor,
        )
        .unwrap();
        s.enable_schedule(id, 0, true, anchor).unwrap();
    }
    let result = s.pump_due(anchor, 10).unwrap();
    assert_eq!(result.len(), 2);
    assert!(
        result
            .iter()
            .all(|r| r.outcome.as_ref().unwrap().len() == 1)
    );
    assert!(s.pump_due(anchor, 10).unwrap().is_empty());
    assert_eq!(s.pending_outbox(10).unwrap().len(), 2);
    drop(s);
    let _ = std::fs::remove_file(file);
}
#[test]
fn sqlite_integer_ranges_are_checked_without_losing_reservations() {
    let file = path();
    let mut s = SqliteStore::open(&file).unwrap();
    assert!(s.create_admission_root("too-big", u64::MAX).is_err());
    s.create_admission_root("root", u64::try_from(i64::MAX).unwrap())
        .unwrap();
    assert!(
        s.reserve_permit(
            &Request {
                id: "overflow".into(),
                root: "root".into(),
                reserve: u64::MAX,
            },
            1,
        )
        .is_err()
    );
    s.reserve_permit(
        &Request {
            id: "held".into(),
            root: "root".into(),
            reserve: 1,
        },
        1,
    )
    .unwrap();
    assert!(s.settle_permit("held", u64::MAX).is_err());
    assert!(
        s.reserve_permit(
            &Request {
                id: "second".into(),
                root: "root".into(),
                reserve: 1,
            },
            1,
        )
        .is_err()
    );
    s.settle_permit("held", 1).unwrap();
    drop(s);
    let _ = std::fs::remove_file(file);
}
#[test]
fn a_latest_schedule_recovers_after_more_missed_instants_than_the_scan_bound() {
    let file = path();
    let anchor = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let mut s = SqliteStore::open(&file).unwrap();
    s.create_schedule(
        "every-second",
        &Schedule::Interval { anchor, seconds: 1 },
        Policy::default(),
        &definition(),
        anchor,
    )
    .unwrap();
    s.enable_schedule("every-second", 0, true, anchor).unwrap();
    let now = anchor + Duration::minutes(20);
    let result = s.pump_due(now, 1024).unwrap();
    let started = result[0].outcome.as_ref().unwrap();
    assert_eq!(started.len(), 1, "exactly the newest instant starts");
    assert_eq!(started[0].scheduled_at, now);
    assert!(s.pump_due(now, 1024).unwrap().is_empty(), "and only once");
    assert_eq!(s.pending_outbox(10).unwrap().len(), 1);
    drop(s);
    let _ = std::fs::remove_file(file);
}
