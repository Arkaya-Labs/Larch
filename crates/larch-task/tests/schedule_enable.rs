use chrono::{TimeZone, Utc};
use larch_task::{
    schedule::{Policy, Schedule},
    store::SqliteStore,
    types::{PlanningPolicy, SCHEMA_VERSION, WorkflowDefinition},
};
use std::collections::BTreeMap;
#[test]
fn saved_schedule_remains_disabled_until_explicitly_enabled() {
    let file =
        std::env::temp_dir().join(format!("larch-enable-{}-{}.sqlite", std::process::id(), {
            // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            format!("{t}-{n}")
        }));
    let now = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "schedule".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let mut s = SqliteStore::open(&file).unwrap();
    s.create_schedule(
        "s",
        &Schedule::Once { at: now },
        Policy::default(),
        &definition,
        now,
    )
    .unwrap();
    let records = s.list_schedules().unwrap();
    assert_eq!(records.len(), 1);
    assert!(!records[0].enabled);
    assert_eq!(records[0].next_due, Some(now));
    assert_eq!(records[0].definition.goal, "schedule");
    assert!(s.pump_due(now, 10).unwrap().is_empty());
    assert!(s.pending_outbox(10).unwrap().is_empty());
    s.enable_schedule("s", 0, true, now).unwrap();
    assert!(s.enable_schedule("s", 0, false, now).is_err());
    assert_eq!(s.pump_due(now, 10).unwrap().len(), 1);
    assert_eq!(s.pending_outbox(10).unwrap().len(), 1);
    drop(s);
    let s = SqliteStore::open(&file).unwrap();
    assert!(s.list_schedules().unwrap()[0].enabled);
    drop(s);
    let _ = std::fs::remove_file(file);
}
/// A daily schedule saved off and turned on the next day runs at its next occurrence, not the
/// moment it is turned on for the one that fell while it was off.
#[test]
fn turning_a_schedule_on_skips_what_fell_while_it_was_off() {
    let file = std::env::temp_dir().join(format!(
        "larch-enable-later-{}-{}.sqlite",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let anchor = Utc.with_ymd_and_hms(2024, 1, 1, 7, 45, 0).unwrap();
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "schedule".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let daily = Schedule::Interval {
        anchor,
        seconds: 86_400,
    };
    let mut s = SqliteStore::open(&file).unwrap();
    s.create_schedule("s", &daily, Policy::default(), &definition, anchor)
        .unwrap();
    let on = anchor + chrono::Duration::hours(25);
    s.enable_schedule("s", 0, true, on).unwrap();
    assert!(s.pump_due(on, 10).unwrap().is_empty());
    let next = anchor + chrono::Duration::days(2);
    assert_eq!(s.list_schedules().unwrap()[0].next_due, Some(next));
    assert_eq!(s.pump_due(next, 10).unwrap().len(), 1);
    drop(s);
    let _ = std::fs::remove_file(file);
}
#[test]
fn upgrading_existing_schedules_preserves_their_enabled_behavior() {
    let file = std::env::temp_dir().join(format!(
        "larch-enable-upgrade-{}-{}.sqlite",
        std::process::id(),
        {
            // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            format!("{t}-{n}")
        }
    ));
    let now = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "schedule".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    {
        let mut s = SqliteStore::open(&file).unwrap();
        s.create_schedule(
            "legacy",
            &Schedule::Once { at: now },
            Policy::default(),
            &definition,
            now,
        )
        .unwrap();
    }
    {
        let db = rusqlite::Connection::open(&file).unwrap();
        db.execute_batch("ALTER TABLE schedules DROP COLUMN enabled; DROP TABLE artifact_pins; DROP TABLE storage_policy; DROP TABLE child_origins; DROP TABLE missed_occurrences; DROP TABLE notification_claims; DROP TABLE step_stats; DROP INDEX occurrences_unfinished; ALTER TABLE occurrences DROP COLUMN finished; ALTER TABLE schedules DROP COLUMN pumped_at; PRAGMA user_version=3;")
            .unwrap();
    }
    let s = SqliteStore::open(&file).unwrap();
    assert!(s.list_schedules().unwrap()[0].enabled);
    drop(s);
    let _ = std::fs::remove_file(file);
}
