use larch_task::{
    admission::Request,
    execution::{ActivityDescriptor, ChildSpawnRequest, ExecutionPlan},
    graph::{Graph, Node, NodeKind},
    store::SqliteStore,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, RunState, SCHEMA_VERSION,
        WorkflowDefinition,
    },
};
use std::collections::BTreeMap;

fn state(id: &str, grants: &[&str]) -> RunState {
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
            grants: grants.iter().map(|g| (*g).into()).collect(),
            budgets: BTreeMap::new(),
            acceptance: vec![],
        },
    )
    .unwrap()
}
fn plan() -> ExecutionPlan {
    ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([(
            "step".into(),
            ActivityDescriptor {
                kind: "script".into(),
                payload: serde_json::Value::Null,
            },
        )]),
    }
}
fn setup() -> SqliteStore {
    let mut store = SqliteStore::open(":memory:").unwrap();
    store
        .create_idempotent("create", &state("parent", &["read", "write"]))
        .unwrap();
    store
        .execute(
            "parent",
            &CommandEnvelope {
                id: "start".into(),
                expected_revision: 0,
                command: Command::StartAttempt {
                    attempt: Attempt {
                        iteration: None,
                        id: "a".into(),
                        node_id: "step".into(),
                        epoch: 1,
                    },
                },
            },
        )
        .unwrap();
    store
}
fn proposal(id: &str, command: &str) -> ChildSpawnRequest {
    ChildSpawnRequest {
        parent_id: "parent".into(),
        expected_revision: 1,
        command_id: command.into(),
        parent_attempt_id: "a".into(),
        parent_epoch: 1,
        expected_graph_revision: 0,
        child: state(id, &["read"]),
        plan: plan(),
    }
}
fn assert_absent(store: &mut SqliteStore, id: &str) {
    assert!(store.get(id).unwrap().is_none());
    assert!(store.execution_plan(id).unwrap().is_none());
    assert_eq!(store.root_id(id).unwrap(), id);
    assert!(
        store
            .message_child("parent", id, "message", "hello")
            .is_err()
    );
    assert_eq!(store.get("parent").unwrap().unwrap().revision, 1);
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 1);
}

#[test]
fn widened_grants_and_mismatched_budget_are_atomic_failures() {
    let mut store = setup();
    let mut widened = proposal("widened", "widen");
    widened.child.definition.grants.push("admin".into());
    assert!(store.spawn_child_with_plan(&widened).is_err());
    assert_absent(&mut store, "widened");
    let mut mismatch = proposal("budget-child", "budget");
    mismatch
        .child
        .definition
        .budgets
        .insert("model_tokens".into(), 100);
    assert!(store.spawn_child_with_plan(&mismatch).is_err());
    assert_absent(&mut store, "budget-child");
}

#[test]
fn independent_child_ledger_blocks_spawn_without_being_modified() {
    let mut store = setup();
    store.create_admission_root("child", 50).unwrap();
    assert!(
        store
            .spawn_child_with_plan(&proposal("child", "spawn"))
            .is_err()
    );
    assert_absent(&mut store, "child");
    let permit = store
        .reserve_permit(
            &Request {
                id: "proof".into(),
                root: "child".into(),
                reserve: 50,
            },
            1,
        )
        .unwrap();
    store.settle_permit(&permit.id, 0).unwrap();
}

#[test]
fn existing_non_spawn_command_id_cannot_be_reused() {
    let mut store = setup();
    assert!(
        store
            .spawn_child_with_plan(&proposal("child", "start"))
            .is_err()
    );
    assert_absent(&mut store, "child");
}

#[test]
fn child_authority_is_scoped_and_follows_parent_revocation() {
    let mut store = setup();
    store
        .spawn_child_with_plan(&proposal("child", "spawn"))
        .unwrap();
    let (before, grants) = store.effective_grants("child").unwrap();
    assert_eq!(grants, vec!["read".to_string()]);
    store
        .execute(
            "parent",
            &CommandEnvelope {
                id: "revoke".into(),
                expected_revision: 2,
                command: Command::RevokeGrant {
                    grant: "read".into(),
                },
            },
        )
        .unwrap();
    let (after, grants) = store.effective_grants("child").unwrap();
    assert!(after > before);
    assert!(grants.is_empty());
    assert_eq!(store.execution_plan("child").unwrap(), Some(plan()));
}
