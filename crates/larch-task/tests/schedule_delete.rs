use chrono::{TimeZone, Utc};
use larch_task::{
    schedule::{Policy, Schedule, occurrence_run_id},
    store::SqliteStore,
    types::{
        Command, CommandEnvelope, PlanningPolicy, RunStatus, SCHEMA_VERSION, WorkflowDefinition,
    },
};
use std::collections::BTreeMap;

/// A schedule is deleted only at the revision shown and once every run it started is settled;
/// it then starts nothing again, and the runs it started stay until they are deleted too.
#[test]
fn a_deleted_schedule_starts_nothing_and_leaves_its_settled_runs() {
    let file = std::env::temp_dir().join(format!(
        "larch-delete-{}-{}.sqlite",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
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
    let mut s = SqliteStore::open(&file).unwrap();
    let every_minute = Schedule::Interval {
        anchor: now,
        seconds: 60,
    };
    s.create_schedule("s", &every_minute, Policy::default(), &definition, now)
        .unwrap();
    s.enable_schedule("s", 0, true, now).unwrap();
    assert_eq!(s.pump_due(now, 10).unwrap().len(), 1);
    let run = occurrence_run_id("s", now);
    assert!(s.get(&run).unwrap().is_some());
    // Its run has not settled.
    let shown = s.schedule_revision("s").unwrap().unwrap();
    assert_eq!(
        s.delete_schedule("s", shown),
        Err("a run of this schedule is still unfinished".into())
    );
    assert!(s.delete_run(&run).is_err());
    let revision = s.get(&run).unwrap().unwrap().revision;
    s.execute(
        &run,
        &CommandEnvelope {
            id: "cancel".into(),
            expected_revision: revision,
            command: Command::Cancel,
        },
    )
    .unwrap();
    assert_eq!(s.get(&run).unwrap().unwrap().status, RunStatus::Cancelled);
    // The schedule still holds its settled run, and a revision not shown is refused.
    assert!(s.delete_run(&run).is_err());
    assert_eq!(
        s.delete_schedule("s", shown - 1),
        Err("unknown schedule or stale revision".into())
    );
    s.delete_schedule("s", shown).unwrap();
    assert!(s.list_schedules().unwrap().is_empty());
    assert!(s.missed_occurrences("s", 10).unwrap().is_empty());
    let later = now + chrono::Duration::minutes(5);
    assert!(s.pump_due(later, 10).unwrap().is_empty());
    assert!(s.delete_schedule("s", shown).is_err());
    // Its run stays, and goes on its own once deleted.
    assert!(s.get(&run).unwrap().is_some());
    s.delete_run(&run).unwrap();
    assert!(s.get(&run).unwrap().is_none());
    drop(s);
    let _ = std::fs::remove_file(file);
}
