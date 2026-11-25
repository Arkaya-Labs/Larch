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
use std::{collections::BTreeMap, sync::Arc};
struct NeverRun;
impl Driver for NeverRun {
    fn execute(&self, _: ActivityRequest) -> Result<ActivityOutcome, String> {
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
fn failed_child() -> (ActorHandle, Executor) {
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
            command_id: "spawn".into(),
            parent_attempt_id: "parent-attempt".into(),
            parent_epoch: 1,
            expected_graph_revision: 0,
            child: state("child"),
            plan: plan(),
        })
        .unwrap();
    actor
        .execute(
            "child".into(),
            command("child-start", 0, start("child-attempt")),
        )
        .unwrap();
    actor
        .execute(
            "child".into(),
            command(
                "fail",
                1,
                Command::FailAttempt {
                    attempt_id: "child-attempt".into(),
                    epoch: 1,
                    reason: "failed".into(),
                },
            ),
        )
        .unwrap();
    let executor = Executor::new(actor.clone(), Arc::new(NeverRun));
    assert!(executor.tick("parent").unwrap());
    (actor, executor)
}

#[test]
fn child_cancelled_after_recorded_failure_can_be_explicitly_retried() {
    let (actor, executor) = failed_child();
    actor
        .execute(
            "child".into(),
            command("independent-cancel", 2, Command::Cancel),
        )
        .unwrap();
    assert_eq!(
        actor.get("child".into()).unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    let failure = actor
        .get("parent".into())
        .unwrap()
        .unwrap()
        .child_failure
        .unwrap();
    assert_eq!(failure.kind, ChildFailureKind::FailedAttempt);
    assert_eq!(failure.reason.as_deref(), Some("failed"));
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
    assert!(parent.child_failure.is_none());
    assert!(parent.waiting_child.is_none());
    assert_eq!(
        actor.get("child".into()).unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    drop(executor);
    actor.shutdown().unwrap();
}

#[test]
fn child_cannot_resume_after_parent_records_failure_decision() {
    let (actor, executor) = failed_child();
    assert!(
        actor
            .execute("child".into(), command("resume", 2, Command::Resume))
            .is_err()
    );
    let child = actor.get("child".into()).unwrap().unwrap();
    assert_eq!(child.status, RunStatus::Waiting);
    assert_eq!(child.attempt_failure.as_deref(), Some("failed"));
    let parent = actor.get("parent".into()).unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Waiting);
    assert_eq!(parent.revision, 3);
    assert_eq!(
        parent.child_failure.unwrap().kind,
        ChildFailureKind::FailedAttempt
    );
    assert!(!executor.tick("parent").unwrap());
    actor.shutdown().unwrap();
}
