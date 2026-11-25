//! Pausing a root blocks new child effects while preserving already-admitted receipts.
use larch_task::{
    execution::{ActivityDescriptor, ChildSpawnRequest, ExecutionPlan},
    graph::{Graph, Node, NodeKind},
    store::SqliteStore,
    types::{
        Attempt, Command, CommandEnvelope, EffectStatus, PlanningPolicy, RunState, RunStatus,
        SCHEMA_VERSION, WorkflowDefinition,
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
            grants: vec!["read".into()],
            budgets: BTreeMap::new(),
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
fn start(id: &str) -> Command {
    Command::StartAttempt {
        attempt: Attempt {
            iteration: None,
            id: id.into(),
            node_id: "step".into(),
            epoch: 1,
        },
    }
}
fn intent(id: &str, revision: u64) -> CommandEnvelope {
    command(
        id,
        revision,
        Command::RecordIntent {
            effect_id: id.into(),
            attempt_id: "child-attempt".into(),
            epoch: 1,
            operation_digest: format!("digest-{id}"),
        },
    )
}

#[test]
fn pause_rejects_new_child_effects_but_allows_inflight_receipt() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    store.create_idempotent("create", &state("parent")).unwrap();
    store
        .execute(
            "parent",
            &command("parent-start", 0, start("parent-attempt")),
        )
        .unwrap();
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([(
            "step".into(),
            ActivityDescriptor {
                kind: "script".into(),
                payload: serde_json::Value::Null,
            },
        )]),
    };
    store
        .spawn_child_and_wait(&ChildSpawnRequest {
            parent_id: "parent".into(),
            expected_revision: 1,
            command_id: "spawn-wait".into(),
            parent_attempt_id: "parent-attempt".into(),
            parent_epoch: 1,
            expected_graph_revision: 0,
            child: state("child"),
            plan,
        })
        .unwrap();
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Waiting
    );
    store
        .execute("child", &command("child-start", 0, start("child-attempt")))
        .unwrap();
    let (generation, grants) = store.effective_grants("child").unwrap();
    assert!(grants.contains(&"read".to_string()));
    let (_, inserted) = store
        .reserve_granted_effect("child", &intent("before-pause", 1), "read", generation)
        .unwrap();
    assert!(inserted);
    store
        .execute("parent", &command("pause", 2, Command::Pause))
        .unwrap();
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Paused
    );
    assert!(
        store
            .reserve_granted_effect("child", &intent("after-pause", 2), "read", generation)
            .is_err()
    );
    assert!(
        !store
            .get("child")
            .unwrap()
            .unwrap()
            .effects
            .contains_key("after-pause")
    );
    store
        .execute(
            "child",
            &command(
                "receipt",
                2,
                Command::RecordReceipt {
                    effect_id: "before-pause".into(),
                    attempt_id: "child-attempt".into(),
                    epoch: 1,
                    receipt: "completed".into(),
                },
            ),
        )
        .unwrap();
    let child = store.get("child").unwrap().unwrap();
    assert_eq!(
        child.effects["before-pause"].status,
        EffectStatus::Completed {
            receipt: "completed".into()
        }
    );
    assert!(!child.effects.contains_key("after-pause"));
}
