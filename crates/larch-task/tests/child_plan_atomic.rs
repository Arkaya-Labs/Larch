use larch_task::{
    execution::{ActivityDescriptor, ChildSpawnRequest, ExecutionPlan},
    graph::{Graph, Node, NodeKind},
    runtime::ActorHandle,
    store::SqliteStore,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, RunState, SCHEMA_VERSION,
        WorkflowDefinition,
    },
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);
impl TestDirectory {
    fn new() -> Self {
        let name = format!(
            "larch-child-plan-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        );
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn database(&self) -> PathBuf {
        self.0.join("tasks.sqlite")
    }
}
impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

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
fn plan() -> ExecutionPlan {
    ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([(
            "step".into(),
            ActivityDescriptor {
                kind: "script".into(),
                payload: serde_json::json!({"text": "result"}),
            },
        )]),
    }
}
fn start() -> CommandEnvelope {
    CommandEnvelope {
        id: "start".into(),
        expected_revision: 0,
        command: Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "attempt-a".into(),
                node_id: "step".into(),
                epoch: 1,
            },
        },
    }
}
fn prepare(store: &mut SqliteStore) {
    store.create_idempotent("create", &state("parent")).unwrap();
    store.execute("parent", &start()).unwrap();
}
fn request(child: &str, command: &str) -> ChildSpawnRequest {
    ChildSpawnRequest {
        parent_id: "parent".into(),
        expected_revision: 1,
        command_id: command.into(),
        parent_attempt_id: "attempt-a".into(),
        parent_epoch: 1,
        expected_graph_revision: 0,
        child: state(child),
        plan: plan(),
    }
}
fn assert_no_child(store: &mut SqliteStore, child: &str, parent_revision: u64) {
    assert!(store.get(child).unwrap().is_none());
    assert!(store.execution_plan(child).unwrap().is_none());
    assert_eq!(store.root_id(child).unwrap(), child);
    assert!(
        store
            .message_child("parent", child, "message", "hello")
            .is_err()
    );
    assert_eq!(
        store.get("parent").unwrap().unwrap().revision,
        parent_revision
    );
}

#[test]
fn committed_child_plan_survives_reopen_and_exact_replay() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let reply = {
        let mut store = SqliteStore::open(&path).unwrap();
        prepare(&mut store);
        let reply = store
            .spawn_child_with_plan(&request("child", "spawn"))
            .unwrap();
        assert_eq!(reply.revision, 2);
        assert_eq!(store.get("child").unwrap(), Some(state("child")));
        assert_eq!(store.execution_plan("child").unwrap(), Some(plan()));
        assert_eq!(store.root_id("child").unwrap(), "parent");
        reply
    };
    let mut store = SqliteStore::open(&path).unwrap();
    assert_eq!(store.get("child").unwrap(), Some(state("child")));
    assert_eq!(store.execution_plan("child").unwrap(), Some(plan()));
    assert_eq!(store.root_id("child").unwrap(), "parent");
    assert_eq!(
        store
            .spawn_child_with_plan(&request("child", "spawn"))
            .unwrap(),
        reply
    );
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 2);
    store
        .message_child("parent", "child", "message", "hello")
        .unwrap();
}

#[test]
fn invalid_plan_and_stale_parent_leave_no_child_link_or_plan() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    prepare(&mut store);
    let mut invalid_plans = Vec::new();
    let mut missing = plan();
    missing.activities.clear();
    invalid_plans.push(missing);
    let mut extra = plan();
    extra
        .activities
        .insert("extra".into(), extra.activities["step"].clone());
    invalid_plans.push(extra);
    let mut wrong_graph = plan();
    wrong_graph.graph_revision = 1;
    invalid_plans.push(wrong_graph);
    let mut empty_kind = plan();
    empty_kind.activities.get_mut("step").unwrap().kind.clear();
    invalid_plans.push(empty_kind);
    for (index, invalid) in invalid_plans.into_iter().enumerate() {
        let id = format!("child-{index}");
        let mut proposal = request(&id, &format!("invalid-{index}"));
        proposal.plan = invalid;
        assert!(store.spawn_child_with_plan(&proposal).is_err());
        assert_no_child(&mut store, &id, 1);
    }
    let mut stale = request("stale", "stale");
    stale.expected_revision = 0;
    assert!(store.spawn_child_with_plan(&stale).is_err());
    assert_no_child(&mut store, "stale", 1);
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 1);
}

#[test]
fn replaced_attempt_cannot_spawn_even_after_reading_current_revision() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    prepare(&mut store);
    store
        .execute(
            "parent",
            &CommandEnvelope {
                id: "recover".into(),
                expected_revision: 1,
                command: Command::Recover,
            },
        )
        .unwrap();
    store
        .execute(
            "parent",
            &CommandEnvelope {
                id: "resume".into(),
                expected_revision: 2,
                command: Command::Resume,
            },
        )
        .unwrap();
    store
        .execute(
            "parent",
            &CommandEnvelope {
                id: "start-b".into(),
                expected_revision: 3,
                command: Command::StartAttempt {
                    attempt: Attempt {
                        iteration: None,
                        id: "attempt-b".into(),
                        node_id: "step".into(),
                        epoch: 2,
                    },
                },
            },
        )
        .unwrap();
    let current_revision = store.get("parent").unwrap().unwrap().revision;
    let mut stale = request("stale-child", "stale-attempt");
    stale.expected_revision = current_revision;
    assert!(store.spawn_child_with_plan(&stale).is_err());
    assert_no_child(&mut store, "stale-child", current_revision);
    let mut stale_graph = request("graph-child", "stale-graph");
    stale_graph.expected_revision = current_revision;
    stale_graph.parent_attempt_id = "attempt-b".into();
    stale_graph.parent_epoch = 2;
    stale_graph.expected_graph_revision = 1;
    assert!(store.spawn_child_with_plan(&stale_graph).is_err());
    assert_no_child(&mut store, "graph-child", current_revision);
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 4);
}

#[test]
fn command_identity_and_existing_child_collisions_are_rejected() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    prepare(&mut store);
    store
        .spawn_child_with_plan(&request("child", "spawn"))
        .unwrap();
    let mut changed_plan = request("child", "spawn");
    changed_plan
        .plan
        .activities
        .get_mut("step")
        .unwrap()
        .payload = serde_json::json!({"text": "other"});
    assert!(store.spawn_child_with_plan(&changed_plan).is_err());
    let mut changed_child = request("child", "spawn");
    changed_child.child.definition.goal = "different".into();
    assert!(store.spawn_child_with_plan(&changed_child).is_err());
    let mut same_child = request("child", "different-command");
    same_child.expected_revision = 2;
    assert!(store.spawn_child_with_plan(&same_child).is_err());
    assert_eq!(store.get("child").unwrap(), Some(state("child")));
    assert_eq!(store.execution_plan("child").unwrap(), Some(plan()));
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 2);
}

#[test]
fn failed_plan_insert_rolls_back_every_part_and_retry_succeeds() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let mut store = SqliteStore::open(&path).unwrap();
    prepare(&mut store);
    let fault = rusqlite::Connection::open(&path).unwrap();
    fault.execute_batch("CREATE TRIGGER reject_child_plan BEFORE INSERT ON execution_plans BEGIN SELECT RAISE(ABORT, 'injected failure'); END;").unwrap();
    assert!(
        store
            .spawn_child_with_plan(&request("child", "spawn"))
            .is_err()
    );
    assert_no_child(&mut store, "child", 1);
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 1);
    fault
        .execute_batch("DROP TRIGGER reject_child_plan;")
        .unwrap();
    store
        .spawn_child_with_plan(&request("child", "spawn"))
        .unwrap();
    assert_eq!(store.execution_plan("child").unwrap(), Some(plan()));
    assert_eq!(store.root_id("child").unwrap(), "parent");
}

#[test]
fn budgeted_parent_cannot_publish_child_before_suspend_accounting_exists() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    let mut parent = state("parent");
    parent.definition.budgets.insert("model_tokens".into(), 20);
    store.create_idempotent("create", &parent).unwrap();
    store
        .dispatch_budgeted_once("parent", &start(), 10, 1)
        .unwrap();
    let mut proposal = request("child", "spawn");
    proposal.child.definition.budgets = parent.definition.budgets;
    assert!(store.spawn_child_with_plan(&proposal).is_err());
    assert_no_child(&mut store, "child", 1);
}

#[test]
fn actor_exposes_atomic_spawn_and_plan() {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    actor.create("create".into(), state("parent")).unwrap();
    actor.execute("parent".into(), start()).unwrap();
    let reply = actor
        .spawn_child_with_plan(request("child", "spawn"))
        .unwrap();
    assert_eq!(actor.get("child".into()).unwrap(), Some(state("child")));
    assert_eq!(actor.execution_plan("child".into()).unwrap(), Some(plan()));
    assert_eq!(
        actor
            .spawn_child_with_plan(request("child", "spawn"))
            .unwrap(),
        reply
    );
    actor.shutdown().unwrap();
}
