use chrono::{TimeZone, Utc};
use larch_task::{
    execution::{ActivityDescriptor, ExecutionPlan},
    graph::{Graph, Node, NodeKind},
    schedule::{Policy, Schedule},
    store::SqliteStore,
    types::{PlanningPolicy, SCHEMA_VERSION, WorkflowDefinition},
};
use std::collections::BTreeMap;
#[test]
fn occurrence_inherits_validated_plan_in_same_transaction() {
    let file = std::env::temp_dir().join(format!("larch-plan-{}-{}.sqlite", std::process::id(), {
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
    let def = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "scheduled".into(),
        initial_graph: Some(Graph {
            nodes: vec![Node {
                when: None,
                optional: Vec::new(),
                id: "work".into(),
                depends_on: vec![],
                kind: NodeKind::Activity,
                inputs: BTreeMap::new(),
                outputs: BTreeMap::new(),
            }],
        }),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([(
            "work".into(),
            ActivityDescriptor {
                kind: "fake".into(),
                payload: serde_json::Value::Null,
            },
        )]),
    };
    let mut s = SqliteStore::open(&file).unwrap();
    s.create_schedule_with_plan(
        "s",
        &Schedule::Once { at: now },
        Policy::default(),
        &def,
        &plan,
        now,
    )
    .unwrap();
    s.enable_schedule("s", 0, true, now).unwrap();
    s.pump_due(now, 10).unwrap();
    let run = s.pending_outbox(10).unwrap().pop().unwrap().payload;
    assert_eq!(s.execution_plan(&run).unwrap().as_ref(), Some(&plan));
    drop(s);
    let mut s = SqliteStore::open(&file).unwrap();
    assert!(s.execution_plan(&run).unwrap().is_some());
    // An edit replaces the rule, goal and steps for later occurrences only, under its revision.
    let later = now + chrono::Duration::days(1);
    let mut edited = def.clone();
    edited.goal = "scheduled, edited".into();
    let revision = s.schedule_revision("s").unwrap().unwrap();
    let stale = s.update_schedule_with_plan(
        "s",
        revision + 1,
        &Schedule::Once { at: later },
        Policy::default(),
        &edited,
        &plan,
        now,
    );
    assert!(stale.is_err());
    s.update_schedule_with_plan(
        "s",
        revision,
        &Schedule::Once { at: later },
        // The first occurrence never ran; beside it, not skipped behind it.
        Policy {
            overlap: larch_task::schedule::Overlap::Parallel { max_active: 2 },
            ..Policy::default()
        },
        &edited,
        &plan,
        now,
    )
    .unwrap();
    let saved = s.list_schedules().unwrap().pop().unwrap();
    assert_eq!(
        (saved.definition.goal.as_str(), saved.next_due),
        ("scheduled, edited", Some(later))
    );
    assert_eq!(s.get(&run).unwrap().unwrap().definition.goal, "scheduled");
    s.pump_due(later, 10).unwrap();
    let next = larch_task::schedule::occurrence_run_id("s", later);
    assert_eq!(
        s.get(&next).unwrap().unwrap().definition.goal,
        "scheduled, edited"
    );
    drop(s);
    let _ = std::fs::remove_file(file);
}
