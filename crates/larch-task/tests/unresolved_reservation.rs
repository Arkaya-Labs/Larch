use larch_task::{
    graph::{Graph, Node, NodeKind},
    store::SqliteStore,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION,
        WorkflowDefinition,
    },
};
use std::collections::BTreeMap;
fn state(id: &str) -> RunState {
    let node = Node {
        when: None,
        optional: Vec::new(),
        id: "step".into(),
        depends_on: vec![],
        kind: NodeKind::Activity,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
    };
    RunState::new(
        id.into(),
        WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "work".into(),
            initial_graph: Some(Graph { nodes: vec![node] }),
            planning_policy: PlanningPolicy::Disabled,
            grants: vec![],
            budgets: BTreeMap::from([("model_tokens".into(), 50)]),
            acceptance: vec![],
        },
    )
    .unwrap()
}
fn command(id: &str, revision: u64, command: Command) -> CommandEnvelope {
    CommandEnvelope {
        id: id.into(),
        expected_revision: revision,
        command,
    }
}
fn start(id: &str, epoch: u64) -> Command {
    Command::StartAttempt {
        attempt: Attempt {
            iteration: None,
            id: id.into(),
            node_id: "step".into(),
            epoch,
        },
    }
}

#[test]
fn resume_requires_reconciliation_and_ids_are_matched_literally() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    let run = "job_%_id";
    let other = "job-AXid";
    assert_eq!(run.len(), other.len());
    store.create_idempotent("create", &state(run)).unwrap();
    store.create_idempotent("create", &state(other)).unwrap();
    store
        .dispatch_budgeted_once(run, &command("start", 0, start("attempt", 1)), 10, 2)
        .unwrap();
    store
        .dispatch_budgeted_once(
            other,
            &command("start", 0, start("other-attempt", 1)),
            10,
            2,
        )
        .unwrap();
    store
        .execute(run, &command("recover", 1, Command::Recover))
        .unwrap();
    assert!(
        store
            .execute(run, &command("resume-before-receipt", 2, Command::Resume))
            .is_err()
    );
    assert_eq!(store.get(run).unwrap().unwrap().status, RunStatus::Waiting);
    store
        .reconcile_budgeted_usage(run, "attempt", 3, "verified usage")
        .unwrap();
    // The other run still holds a reservation and has a name that matches this run under SQL LIKE.
    store
        .execute(run, &command("resume-after-receipt", 2, Command::Resume))
        .unwrap();
    assert_eq!(store.get(run).unwrap().unwrap().status, RunStatus::Ready);
    assert!(
        store
            .dispatch_budgeted_once(run, &command("next", 3, start("next-attempt", 2)), 10, 2)
            .unwrap()
            .1
    );
}

#[test]
fn cancellation_remains_available_with_unresolved_usage() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    store.create_idempotent("create", &state("root")).unwrap();
    store
        .dispatch_budgeted_once("root", &command("start", 0, start("attempt", 1)), 10, 1)
        .unwrap();
    store
        .execute("root", &command("recover", 1, Command::Recover))
        .unwrap();
    assert!(
        store
            .execute("root", &command("resume", 2, Command::Resume))
            .is_err()
    );
    store
        .execute("root", &command("cancel", 2, Command::Cancel))
        .unwrap();
    assert_eq!(
        store.get("root").unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    store
        .reconcile_budgeted_usage("root", "attempt", 3, "verified usage")
        .unwrap();
    assert_eq!(
        store.get("root").unwrap().unwrap().status,
        RunStatus::Cancelled
    );
}
