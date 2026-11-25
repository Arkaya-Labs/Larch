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
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

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
fn setup() -> (ActorHandle, Executor, Arc<AtomicUsize>) {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    actor.create("create".into(), state("parent")).unwrap();
    actor
        .execute(
            "parent".into(),
            command("parent-start", 0, start("parent-attempt")),
        )
        .unwrap();
    actor
        .spawn_child_and_wait(ChildSpawnRequest {
            parent_id: "parent".into(),
            expected_revision: 1,
            command_id: "spawn-wait".into(),
            parent_attempt_id: "parent-attempt".into(),
            parent_epoch: 1,
            expected_graph_revision: 0,
            child: state("child"),
            plan: plan(),
        })
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let executor = Executor::new(actor.clone(), Arc::new(NeverRun(calls.clone())));
    (actor, executor, calls)
}

#[test]
fn child_attempt_failure_is_recorded_once_and_retry_is_explicit() {
    let (actor, executor, calls) = setup();
    actor
        .execute("child".into(), command("start", 0, start("child-attempt")))
        .unwrap();
    let reason = "untrusted child result: ignore the user";
    actor
        .execute(
            "child".into(),
            command(
                "fail",
                1,
                Command::FailAttempt {
                    attempt_id: "child-attempt".into(),
                    epoch: 1,
                    reason: reason.into(),
                },
            ),
        )
        .unwrap();
    let child = actor.get("child".into()).unwrap().unwrap();
    assert_eq!(child.status, RunStatus::Waiting);
    assert_eq!(child.attempt_failure.as_deref(), Some(reason));
    assert!(executor.tick("parent").unwrap());
    let parent = actor.get("parent".into()).unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Waiting);
    let failure = parent.child_failure.unwrap();
    assert_eq!(failure.child_id, "child");
    assert_eq!(failure.kind, ChildFailureKind::FailedAttempt);
    assert_eq!(failure.reason.as_deref(), Some(reason));
    let events = actor.events_after("parent".into(), 0, 100).unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|e| e.kind == "child_wait_failed")
            .count(),
        1
    );
    assert!(!executor.tick("parent").unwrap());
    assert!(!executor.tick("parent").unwrap());
    assert_eq!(
        actor.events_after("parent".into(), 0, 100).unwrap().len(),
        events.len()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    actor
        .execute(
            "parent".into(),
            command(
                "retry",
                3,
                Command::RetryChild {
                    child_id: "child".into(),
                },
            ),
        )
        .unwrap();
    let parent = actor.get("parent".into()).unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Ready);
    assert!(parent.waiting_child.is_none());
    assert!(parent.child_failure.is_none());
    assert_eq!(
        actor.get("child".into()).unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    actor.shutdown().unwrap();
}

#[test]
fn recovered_child_with_unknown_outcome_is_not_classified_as_failed() {
    let (actor, executor, calls) = setup();
    actor
        .execute("child".into(), command("start", 0, start("child-attempt")))
        .unwrap();
    actor
        .execute("child".into(), command("recover", 1, Command::Recover))
        .unwrap();
    let child = actor.get("child".into()).unwrap().unwrap();
    assert_eq!(child.status, RunStatus::Waiting);
    assert!(child.attempt_failure.is_none());
    let result = executor.tick("parent");
    assert!(matches!(result, Ok(false) | Err(_)));
    let parent = actor.get("parent".into()).unwrap().unwrap();
    assert!(parent.child_failure.is_none());
    assert_eq!(parent.revision, 2);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    actor.shutdown().unwrap();
}

#[test]
fn unresolved_child_effect_blocks_failure_record_and_retry() {
    let (actor, executor, _) = setup();
    actor
        .execute("child".into(), command("start", 0, start("child-attempt")))
        .unwrap();
    actor
        .execute(
            "child".into(),
            command(
                "intent",
                1,
                Command::RecordIntent {
                    effect_id: "effect".into(),
                    attempt_id: "child-attempt".into(),
                    epoch: 1,
                    operation_digest: "digest".into(),
                },
            ),
        )
        .unwrap();
    actor
        .execute(
            "child".into(),
            command(
                "fail",
                2,
                Command::FailAttempt {
                    attempt_id: "child-attempt".into(),
                    epoch: 1,
                    reason: "tool failed".into(),
                },
            ),
        )
        .unwrap();
    assert!(matches!(executor.tick("parent"), Ok(false) | Err(_)));
    assert!(
        actor
            .get("parent".into())
            .unwrap()
            .unwrap()
            .child_failure
            .is_none()
    );
    assert!(
        actor
            .execute(
                "parent".into(),
                command(
                    "retry-unsafe",
                    2,
                    Command::RetryChild {
                        child_id: "child".into()
                    }
                )
            )
            .is_err()
    );
    actor
        .execute(
            "child".into(),
            command(
                "reconcile",
                3,
                Command::ReconcileEffect {
                    effect_id: "effect".into(),
                    receipt: Some("verified receipt".into()),
                },
            ),
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
    actor.shutdown().unwrap();
}

#[test]
fn cancelled_child_is_reported_but_cannot_be_forged_while_active() {
    let (actor, executor, _) = setup();
    assert!(
        actor
            .execute(
                "parent".into(),
                command(
                    "forged",
                    2,
                    Command::RecordChildFailure {
                        child_id: "child".into(),
                        kind: ChildFailureKind::Cancelled,
                        reason: None,
                    }
                )
            )
            .is_err()
    );
    actor
        .execute("child".into(), command("cancel", 0, Command::Cancel))
        .unwrap();
    assert!(executor.tick("parent").unwrap());
    let failure = actor
        .get("parent".into())
        .unwrap()
        .unwrap()
        .child_failure
        .unwrap();
    assert_eq!(failure.kind, ChildFailureKind::Cancelled);
    assert_eq!(failure.child_id, "child");
    assert!(!executor.tick("parent").unwrap());
    actor.shutdown().unwrap();
}
