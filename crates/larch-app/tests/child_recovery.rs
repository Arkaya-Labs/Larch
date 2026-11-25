//! Child recovery is projected from committed task state, not inferred from Waiting.
use larch_app::task::TaskCard;
use larch_task::types::{
    ChildFailure, ChildFailureKind, PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION,
    WorkflowDefinition,
};
use std::collections::BTreeMap;

#[test]
fn card_projects_committed_child_failure_and_accepts_older_json() {
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "Find evidence".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let mut run = RunState::new("parent".into(), definition).unwrap();
    run.status = RunStatus::Waiting;
    run.child_failure = Some(ChildFailure {
        child_id: "child-7".into(),
        kind: ChildFailureKind::FailedAttempt,
        reason: Some("permission denied".into()),
    });
    let card = TaskCard::from(&run);
    let json = serde_json::to_value(&card).unwrap();
    assert_eq!(json["childFailure"]["child_id"], "child-7");
    assert_eq!(json["childFailure"]["kind"], "failed_attempt");
    assert_eq!(json["childFailure"]["reason"], "permission denied");
    let roundtrip: TaskCard = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(roundtrip.child_failure, run.child_failure);
    let mut older = json;
    older.as_object_mut().unwrap().remove("childFailure");
    let restored: TaskCard = serde_json::from_value(older).unwrap();
    assert!(restored.child_failure.is_none());
}
