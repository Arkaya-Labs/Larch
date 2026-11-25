use larch_task::{
    execution::{ActivityDescriptor, ChildSpawnRequest, ExecutionPlan},
    graph::{Graph, Node, NodeKind},
    runtime::ActorHandle,
    store::SqliteStore,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION,
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
            "larch-child-wait-{}-{}",
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
            grants: vec![],
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
fn setup(store: &mut SqliteStore) {
    store.create_idempotent("create", &state("parent")).unwrap();
    store
        .execute("parent", &command("start-a", 0, start("attempt-a", 1)))
        .unwrap();
}
fn request(child: &str, command_id: &str) -> ChildSpawnRequest {
    ChildSpawnRequest {
        parent_id: "parent".into(),
        expected_revision: 1,
        command_id: command_id.into(),
        parent_attempt_id: "attempt-a".into(),
        parent_epoch: 1,
        expected_graph_revision: 0,
        child: state(child),
        plan: plan(),
    }
}
fn assert_no_child(store: &mut SqliteStore, child: &str, revision: u64) {
    assert!(store.get(child).unwrap().is_none());
    assert!(store.execution_plan(child).unwrap().is_none());
    assert_eq!(store.root_id(child).unwrap(), child);
    assert!(
        store
            .message_child("parent", child, "message", "hello")
            .is_err()
    );
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.revision, revision);
    assert_eq!(parent.status, RunStatus::Running);
    assert_eq!(parent.attempt.unwrap().id, "attempt-a");
    assert!(parent.waiting_child.is_none());
}
fn finish_child(store: &mut SqliteStore, child: &str) {
    store
        .execute(child, &command("child-start", 0, start("child-attempt", 1)))
        .unwrap();
    store
        .execute(
            child,
            &command(
                "child-finish",
                1,
                Command::FinishAttempt {
                    attempt_id: "child-attempt".into(),
                    epoch: 1,
                    result: "evidence".into(),
                },
            ),
        )
        .unwrap();
    store
        .execute(
            child,
            &command(
                "child-complete",
                2,
                Command::CompleteRun {
                    result: "evidence".into(),
                },
            ),
        )
        .unwrap();
}

#[test]
fn spawn_and_wait_survive_reopen_and_exact_replay() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let original = {
        let mut store = SqliteStore::open(&path).unwrap();
        setup(&mut store);
        let reply = store
            .spawn_child_and_wait(&request("child", "spawn-wait"))
            .unwrap();
        assert_eq!(reply.revision, 2);
        assert_eq!(reply.status, RunStatus::Waiting);
        reply
    };
    let mut store = SqliteStore::open(&path).unwrap();
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Waiting);
    assert!(parent.attempt.is_none());
    let waiting = parent.waiting_child.unwrap();
    assert_eq!(waiting.child_id, "child");
    assert_eq!(waiting.node_id, "step");
    assert_eq!(waiting.graph_revision, 0);
    assert_eq!(waiting.origin_attempt_id.as_deref(), Some("attempt-a"));
    assert_eq!(waiting.origin_epoch, Some(1));
    assert_eq!(store.get("child").unwrap(), Some(state("child")));
    assert_eq!(store.root_id("child").unwrap(), "parent");
    assert_eq!(store.execution_plan("child").unwrap(), Some(plan()));
    assert_eq!(
        store
            .spawn_child_and_wait(&request("child", "spawn-wait"))
            .unwrap(),
        original
    );
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 2);
    assert!(store.collect_waited_child("parent", "child").is_err());
    finish_child(&mut store, "child");
    assert_eq!(
        store
            .collect_waited_child("parent", "child")
            .unwrap()
            .status,
        RunStatus::Succeeded
    );
    assert!(
        store
            .execute(
                "parent",
                &command(
                    "wrong-result",
                    2,
                    Command::WakeChild {
                        child_id: "child".into(),
                        result: "forged".into(),
                    }
                )
            )
            .is_err()
    );
    store
        .execute(
            "parent",
            &command(
                "wake",
                2,
                Command::WakeChild {
                    child_id: "child".into(),
                    result: "evidence".into(),
                },
            ),
        )
        .unwrap();
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Ready);
    assert_eq!(
        parent.node_results.get("step").map(String::as_str),
        Some("evidence")
    );
}

#[test]
fn database_failure_rolls_back_child_plan_link_and_parent_wait() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let mut store = SqliteStore::open(&path).unwrap();
    setup(&mut store);
    let fault = rusqlite::Connection::open(&path).unwrap();
    fault.execute_batch("CREATE TRIGGER reject_child_plan BEFORE INSERT ON execution_plans BEGIN SELECT RAISE(ABORT, 'injected plan failure'); END;").unwrap();
    assert!(
        store
            .spawn_child_and_wait(&request("child", "spawn-wait"))
            .is_err()
    );
    assert_no_child(&mut store, "child", 1);
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 1);
    fault.execute_batch("DROP TRIGGER reject_child_plan; CREATE TRIGGER reject_parent_wait BEFORE UPDATE ON runs WHEN NEW.id = 'parent' BEGIN SELECT RAISE(ABORT, 'injected parent failure'); END;").unwrap();
    assert!(
        store
            .spawn_child_and_wait(&request("child", "spawn-wait"))
            .is_err()
    );
    assert_no_child(&mut store, "child", 1);
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 1);
    fault
        .execute_batch("DROP TRIGGER reject_parent_wait;")
        .unwrap();
    store
        .spawn_child_and_wait(&request("child", "spawn-wait"))
        .unwrap();
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Waiting
    );
}

#[test]
fn stale_attempt_or_graph_and_invalid_plan_publish_nothing() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    setup(&mut store);
    let mut bad_plan = request("bad-plan", "bad-plan");
    bad_plan.plan.activities.clear();
    assert!(store.spawn_child_and_wait(&bad_plan).is_err());
    assert_no_child(&mut store, "bad-plan", 1);
    let mut bad_epoch = request("bad-epoch", "bad-epoch");
    bad_epoch.parent_epoch = 2;
    assert!(store.spawn_child_and_wait(&bad_epoch).is_err());
    assert_no_child(&mut store, "bad-epoch", 1);
    let mut bad_graph = request("bad-graph", "bad-graph");
    bad_graph.expected_graph_revision = 1;
    assert!(store.spawn_child_and_wait(&bad_graph).is_err());
    assert_no_child(&mut store, "bad-graph", 1);
    let mut bad_revision = request("bad-revision", "bad-revision");
    bad_revision.expected_revision = 0;
    assert!(store.spawn_child_and_wait(&bad_revision).is_err());
    assert_no_child(&mut store, "bad-revision", 1);
}

#[test]
fn prior_attempt_child_cannot_be_adopted_by_replacement_attempt() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    setup(&mut store);
    store
        .spawn_child_with_plan(&request("child-a", "spawn-a"))
        .unwrap();
    finish_child(&mut store, "child-a");
    store
        .execute("parent", &command("recover", 2, Command::Recover))
        .unwrap();
    store
        .execute("parent", &command("resume", 3, Command::Resume))
        .unwrap();
    store
        .execute("parent", &command("start-b", 4, start("attempt-b", 2)))
        .unwrap();
    let wait = Command::WaitForChild {
        attempt_id: "attempt-b".into(),
        epoch: 2,
        child_id: "child-a".into(),
    };
    assert!(store.execute("parent", &command("adopt", 5, wait)).is_err());
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.revision, 5);
    assert_eq!(parent.status, RunStatus::Running);
    assert_eq!(parent.attempt.unwrap().id, "attempt-b");
    assert!(parent.waiting_child.is_none());
    assert!(store.collect_waited_child("parent", "child-a").is_err());
    assert!(
        store
            .execute(
                "parent",
                &command(
                    "forged-wake",
                    5,
                    Command::WakeChild {
                        child_id: "child-a".into(),
                        result: "evidence".into(),
                    }
                )
            )
            .is_err()
    );
}

#[test]
fn legacy_unbound_child_cannot_unlock_parent() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    setup(&mut store);
    store
        .spawn_child("parent", 1, "legacy-spawn", &state("legacy"))
        .unwrap();
    assert!(
        store
            .execute(
                "parent",
                &command(
                    "wait",
                    2,
                    Command::WaitForChild {
                        attempt_id: "attempt-a".into(),
                        epoch: 1,
                        child_id: "legacy".into(),
                    }
                )
            )
            .is_err()
    );
    assert!(store.collect_waited_child("parent", "legacy").is_err());
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Running
    );
}

#[test]
fn recovery_preserves_wait_and_cancellation_cannot_wake_parent() {
    let dir = TestDirectory::new();
    let path = dir.database();
    {
        let mut store = SqliteStore::open(&path).unwrap();
        setup(&mut store);
        store
            .spawn_child_and_wait(&request("child", "spawn-wait"))
            .unwrap();
    }
    let mut store = SqliteStore::open(&path).unwrap();
    store
        .execute("parent", &command("recover", 2, Command::Recover))
        .unwrap();
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Waiting);
    assert!(parent.attempt.is_none());
    assert_eq!(parent.waiting_child.unwrap().child_id, "child");
    assert!(
        store
            .execute("parent", &command("resume", 3, Command::Resume))
            .is_err()
    );
    store
        .execute("parent", &command("cancel", 3, Command::Cancel))
        .unwrap();
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    assert_eq!(
        store.get("child").unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    assert!(store.collect_waited_child("parent", "child").is_err());
    assert!(
        store
            .execute(
                "parent",
                &command(
                    "wake",
                    4,
                    Command::WakeChild {
                        child_id: "child".into(),
                        result: "evidence".into(),
                    }
                )
            )
            .is_err()
    );
}

#[test]
fn unresolved_effect_prevents_atomic_wait_and_child_publication() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    setup(&mut store);
    store
        .execute(
            "parent",
            &command(
                "intent",
                1,
                Command::RecordIntent {
                    effect_id: "e".into(),
                    attempt_id: "attempt-a".into(),
                    epoch: 1,
                    operation_digest: "digest".into(),
                },
            ),
        )
        .unwrap();
    let mut proposal = request("child", "spawn-wait");
    proposal.expected_revision = 2;
    assert!(store.spawn_child_and_wait(&proposal).is_err());
    assert_no_child(&mut store, "child", 2);
}

#[test]
fn actor_exposes_atomic_spawn_and_wait() {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    actor.create("create".into(), state("parent")).unwrap();
    actor
        .execute(
            "parent".into(),
            command("start-a", 0, start("attempt-a", 1)),
        )
        .unwrap();
    let reply = actor
        .spawn_child_and_wait(request("child", "spawn-wait"))
        .unwrap();
    assert_eq!(reply.status, RunStatus::Waiting);
    assert_eq!(actor.execution_plan("child".into()).unwrap(), Some(plan()));
    assert_eq!(
        actor
            .spawn_child_and_wait(request("child", "spawn-wait"))
            .unwrap(),
        reply
    );
    actor.shutdown().unwrap();
}

#[test]
fn conflicting_replay_and_prior_command_identity_do_not_change_wait() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    setup(&mut store);
    let mut prior_id = request("no-child", "start-a");
    prior_id.expected_revision = 1;
    assert!(store.spawn_child_and_wait(&prior_id).is_err());
    assert_no_child(&mut store, "no-child", 1);
    store
        .spawn_child_and_wait(&request("child", "spawn-wait"))
        .unwrap();
    let mut changed = request("child", "spawn-wait");
    changed.plan.activities.get_mut("step").unwrap().payload = serde_json::json!({"changed": true});
    assert!(store.spawn_child_and_wait(&changed).is_err());
    let mut wrong_child = request("other", "spawn-wait");
    wrong_child.expected_revision = 2;
    assert!(store.spawn_child_and_wait(&wrong_child).is_err());
    assert!(store.get("other").unwrap().is_none());
    assert!(store.execution_plan("other").unwrap().is_none());
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.revision, 2);
    assert_eq!(parent.status, RunStatus::Waiting);
    assert_eq!(parent.waiting_child.unwrap().child_id, "child");
    assert_eq!(store.execution_plan("child").unwrap(), Some(plan()));
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 2);
}

#[test]
fn cancelled_parent_cannot_publish_new_child() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    setup(&mut store);
    store
        .execute("parent", &command("cancel", 1, Command::Cancel))
        .unwrap();
    let mut proposal = request("late", "spawn-late");
    proposal.expected_revision = 2;
    assert!(store.spawn_child_and_wait(&proposal).is_err());
    assert!(store.get("late").unwrap().is_none());
    assert!(store.execution_plan("late").unwrap().is_none());
    assert_eq!(store.root_id("late").unwrap(), "late");
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Cancelled
    );
}

#[test]
fn pausing_child_wait_preserves_origin_and_resumes_without_redoing_parent() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    setup(&mut store);
    store
        .spawn_child_and_wait(&request("child", "spawn-wait"))
        .unwrap();
    store
        .execute(
            "child",
            &command("child-start", 0, start("child-attempt", 1)),
        )
        .unwrap();
    store
        .execute("parent", &command("pause", 2, Command::Pause))
        .unwrap();
    let paused = store.get("parent").unwrap().unwrap();
    assert_eq!(paused.status, RunStatus::Paused);
    assert_eq!(paused.waiting_child.unwrap().child_id, "child");
    store
        .execute(
            "child",
            &command(
                "child-finish",
                1,
                Command::FinishAttempt {
                    attempt_id: "child-attempt".into(),
                    epoch: 1,
                    result: "evidence".into(),
                },
            ),
        )
        .unwrap();
    store
        .execute(
            "child",
            &command(
                "child-complete",
                2,
                Command::CompleteRun {
                    result: "evidence".into(),
                },
            ),
        )
        .unwrap();
    assert!(
        store
            .execute(
                "parent",
                &command(
                    "wake-while-paused",
                    3,
                    Command::WakeChild {
                        child_id: "child".into(),
                        result: "evidence".into(),
                    }
                )
            )
            .is_err()
    );
    store
        .execute("parent", &command("resume", 3, Command::Resume))
        .unwrap();
    let waiting = store.get("parent").unwrap().unwrap();
    assert_eq!(waiting.status, RunStatus::Waiting);
    assert!(waiting.attempt.is_none());
    assert_eq!(waiting.waiting_child.unwrap().child_id, "child");
    store
        .execute(
            "parent",
            &command(
                "wake",
                4,
                Command::WakeChild {
                    child_id: "child".into(),
                    result: "evidence".into(),
                },
            ),
        )
        .unwrap();
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Ready);
    assert_eq!(
        parent.node_results.get("step").map(String::as_str),
        Some("evidence")
    );
    assert_eq!(parent.last_epoch, 1);
    assert!(
        store
            .execute(
                "parent",
                &command(
                    "wake-again",
                    5,
                    Command::WakeChild {
                        child_id: "child".into(),
                        result: "evidence".into(),
                    }
                )
            )
            .is_err()
    );
}
