//! Defensive recovery fixtures: residual child reservations must block failure retry.
use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, ChildSpawnRequest, Driver,
        ExecutionPlan, Executor,
    },
    graph::{Graph, Node, NodeKind},
    runtime::ActorHandle,
    types::{
        Attempt, ChildFailureKind, Command, CommandEnvelope, PlanningPolicy, RunState, RunStatus,
        SCHEMA_VERSION, WorkflowDefinition,
    },
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};
static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
struct TestDirectory(PathBuf);
impl TestDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "larch-child-failure-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
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
struct NeverRun(Arc<AtomicUsize>);
impl Driver for NeverRun {
    fn execute(&self, _: ActivityRequest) -> Result<ActivityOutcome, String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err("unexpected automatic execution".into())
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
fn setup(path: &std::path::Path) -> (ActorHandle, Executor, Arc<AtomicUsize>) {
    let actor = ActorHandle::start(path, 8).unwrap();
    actor.create("create".into(), state("parent")).unwrap();
    actor
        .execute(
            "parent".into(),
            command("parent-start", 0, start("parent-attempt", 1)),
        )
        .unwrap();
    actor
        .spawn_child_and_wait(ChildSpawnRequest {
            parent_id: "parent".into(),
            expected_revision: 1,
            command_id: "spawn".into(),
            parent_attempt_id: "parent-attempt".into(),
            parent_epoch: 1,
            expected_graph_revision: 0,
            child: state("child_%x"),
            plan: plan(),
        })
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let executor = Executor::new(actor.clone(), Arc::new(NeverRun(calls.clone())));
    (actor, executor, calls)
}
fn fail_child(actor: &ActorHandle) {
    actor
        .execute(
            "child_%x".into(),
            command("start", 0, start("child-attempt", 1)),
        )
        .unwrap();
    actor
        .execute(
            "child_%x".into(),
            command(
                "fail",
                1,
                Command::FailAttempt {
                    attempt_id: "child-attempt".into(),
                    epoch: 1,
                    reason: "failure".into(),
                },
            ),
        )
        .unwrap();
}
/// Explicit synthetic recovery fixture: test that residual permits fail closed even if other
/// persisted state says a child attempt failed. Normal settlement should not create this state.
fn seed_residual_permit(path: &std::path::Path, attempt: &str) {
    let db = rusqlite::Connection::open(path).unwrap();
    db.execute(
        "INSERT OR IGNORE INTO admission_roots(id,budget,spent,reserved) VALUES('parent',50,0,0)",
        [],
    )
    .unwrap();
    db.execute(
        "UPDATE admission_roots SET reserved=reserved+5 WHERE id='parent'",
        [],
    )
    .unwrap();
    let child = "child_%x";
    let id = format!("dispatch:{}:{}:{}", child.len(), child, attempt);
    db.execute(
        "INSERT INTO admission_permits(id,root_id,reserved) VALUES(?1,'parent',5)",
        [&id],
    )
    .unwrap();
}

#[test]
fn residual_child_permit_blocks_failure_projection_until_reconciled() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let (actor, executor, calls) = setup(&path);
    fail_child(&actor);
    seed_residual_permit(&path, "child-attempt");
    assert!(matches!(executor.tick("parent"), Ok(false) | Err(_)));
    let parent = actor.get("parent".into()).unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Waiting);
    assert!(parent.child_failure.is_none());
    assert_eq!(parent.revision, 2);
    assert!(
        actor
            .execute(
                "parent".into(),
                command(
                    "retry",
                    2,
                    Command::RetryChild {
                        child_id: "child_%x".into()
                    }
                )
            )
            .is_err()
    );
    actor
        .reconcile_budgeted_usage(
            "child_%x".into(),
            "child-attempt".into(),
            3,
            "verified usage".into(),
        )
        .unwrap();
    assert!(executor.tick("parent").unwrap());
    assert_eq!(
        actor
            .get("parent".into())
            .unwrap()
            .unwrap()
            .child_failure
            .unwrap()
            .kind,
        ChildFailureKind::FailedAttempt
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    actor.shutdown().unwrap();
}

#[test]
fn retry_rechecks_reservations_even_after_failure_was_recorded() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let (actor, executor, _) = setup(&path);
    fail_child(&actor);
    assert!(executor.tick("parent").unwrap());
    seed_residual_permit(&path, "older-attempt");
    assert!(
        actor
            .execute(
                "parent".into(),
                command(
                    "retry",
                    3,
                    Command::RetryChild {
                        child_id: "child_%x".into()
                    }
                )
            )
            .is_err()
    );
    let parent = actor.get("parent".into()).unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Waiting);
    assert_eq!(parent.revision, 3);
    assert_eq!(
        parent.child_failure.unwrap().kind,
        ChildFailureKind::FailedAttempt
    );
    assert_eq!(
        actor.get("child_%x".into()).unwrap().unwrap().status,
        RunStatus::Waiting
    );
    actor.shutdown().unwrap();
}

#[test]
fn retrying_then_recovering_child_clears_prior_failure_marker() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let (actor, executor, calls) = setup(&path);
    fail_child(&actor);
    actor
        .execute("child_%x".into(), command("resume", 2, Command::Resume))
        .unwrap();
    actor
        .execute(
            "child_%x".into(),
            command("start-b", 3, start("child-b", 2)),
        )
        .unwrap();
    actor
        .execute("child_%x".into(), command("recover-b", 4, Command::Recover))
        .unwrap();
    let child = actor.get("child_%x".into()).unwrap().unwrap();
    assert_eq!(child.status, RunStatus::Waiting);
    assert!(child.attempt_failure.is_none());
    assert!(matches!(executor.tick("parent"), Ok(false) | Err(_)));
    assert!(
        actor
            .get("parent".into())
            .unwrap()
            .unwrap()
            .child_failure
            .is_none()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    actor.shutdown().unwrap();
}
