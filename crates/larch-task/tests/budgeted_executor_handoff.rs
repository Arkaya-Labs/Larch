use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, BudgetedChildSpawnRequest,
        ChildSpawnRequest, Driver, ExecutionPlan, Executor,
    },
    graph::{Graph, Node, NodeKind},
    runtime::ActorHandle,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION,
        WorkflowDefinition,
    },
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
struct TestDirectory(PathBuf);
impl TestDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "larch-handoff-executor-{}-{}",
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
fn plan() -> ExecutionPlan {
    ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([(
            "step".into(),
            ActivityDescriptor {
                kind: "delegate".into(),
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
fn spawn(request: &ActivityRequest, invalid: bool) -> ChildSpawnRequest {
    let mut child_plan = plan();
    if invalid {
        child_plan.activities.clear();
    }
    ChildSpawnRequest {
        parent_id: request.run_id.clone(),
        expected_revision: 1,
        command_id: "child-handoff".into(),
        parent_attempt_id: request.attempt_id.clone(),
        parent_epoch: request.epoch,
        expected_graph_revision: request.graph_revision,
        child: state("child"),
        plan: child_plan,
    }
}
#[derive(Clone, Copy)]
enum Mode {
    Proposal,
    InvalidPlan,
    Overrun,
    MissingReceipt,
    Precommitted,
    PrecommittedMismatch,
}
struct Delegate {
    actor: ActorHandle,
    mode: Mode,
}
impl Driver for Delegate {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        assert_eq!(request.max_tokens, Some(20));
        if matches!(self.mode, Mode::Precommitted | Mode::PrecommittedMismatch) {
            self.actor
                .spawn_child_and_wait_budgeted(BudgetedChildSpawnRequest {
                    spawn: spawn(&request, false),
                    used: 7,
                    usage_receipt: "parent-usage".into(),
                })?;
        }
        Ok(ActivityOutcome {
            output: String::new(),
        })
    }
    fn enforces_budget(&self) -> bool {
        true
    }
    fn reservation(&self, _: &ActivityRequest) -> Option<(u64, usize)> {
        Some((20, 1))
    }
    fn measured_usage(
        &self,
        _: &ActivityRequest,
        _: &Result<ActivityOutcome, String>,
    ) -> Option<u64> {
        Some(if matches!(self.mode, Mode::Overrun) {
            21
        } else {
            7
        })
    }
    fn usage_receipt(
        &self,
        _: &ActivityRequest,
        _: &Result<ActivityOutcome, String>,
    ) -> Option<String> {
        match self.mode {
            Mode::MissingReceipt => None,
            Mode::PrecommittedMismatch => Some("different".into()),
            _ => Some("parent-usage".into()),
        }
    }
    fn budgeted_child(
        &self,
        request: &ActivityRequest,
        _: &Result<ActivityOutcome, String>,
    ) -> Option<ChildSpawnRequest> {
        Some(spawn(request, matches!(self.mode, Mode::InvalidPlan)))
    }
    fn suspended_child(
        &self,
        _: &ActivityRequest,
        _: &Result<ActivityOutcome, String>,
    ) -> Option<String> {
        matches!(self.mode, Mode::Precommitted | Mode::PrecommittedMismatch).then(|| "child".into())
    }
}
fn ledger(path: &std::path::Path) -> (i64, i64, i64, i64) {
    let db = rusqlite::Connection::open(path).unwrap();
    let (spent, reserved) = db
        .query_row(
            "SELECT spent,reserved FROM admission_roots WHERE id='parent'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    let permits = db
        .query_row("SELECT count(*) FROM admission_permits", [], |r| r.get(0))
        .unwrap();
    let receipts = db
        .query_row("SELECT count(*) FROM budget_receipts", [], |r| r.get(0))
        .unwrap();
    (spent, reserved, permits, receipts)
}
fn setup(path: &std::path::Path, mode: Mode) -> (ActorHandle, Executor) {
    let actor = ActorHandle::start(path, 8).unwrap();
    actor.create("create".into(), state("parent")).unwrap();
    let executor = Executor::new(
        actor.clone(),
        Arc::new(Delegate {
            actor: actor.clone(),
            mode,
        }),
    );
    executor.install_plan("parent", plan()).unwrap();
    (actor, executor)
}

#[test]
fn executor_commits_proposed_handoff_and_one_seat_child_can_start() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let (actor, executor) = setup(&path, Mode::Proposal);
    assert!(executor.tick("parent").unwrap());
    let parent = actor.get("parent".into()).unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Waiting);
    assert_eq!(parent.waiting_child.unwrap().child_id, "child");
    assert_eq!(ledger(&path), (7, 0, 0, 1));
    let start = command(
        "child-start",
        0,
        Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "child-attempt".into(),
                node_id: "step".into(),
                epoch: 1,
            },
        },
    );
    assert!(
        actor
            .dispatch_budgeted_once("child".into(), start, 20, 1)
            .unwrap()
            .1
    );
    assert_eq!(ledger(&path), (7, 20, 1, 1));
    actor.shutdown().unwrap();
}

#[test]
fn invalid_plan_and_overrun_charge_actual_usage_without_publishing_child() {
    for (mode, expected) in [(Mode::InvalidPlan, 7), (Mode::Overrun, 21)] {
        let dir = TestDirectory::new();
        let path = dir.database();
        let (actor, executor) = setup(&path, mode);
        assert!(executor.tick("parent").is_err());
        let parent = actor.get("parent".into()).unwrap().unwrap();
        assert_eq!(parent.status, RunStatus::Waiting);
        assert!(parent.attempt.is_none());
        assert!(actor.get("child".into()).unwrap().is_none());
        assert_eq!(ledger(&path), (expected, 0, 0, 1));
        actor.shutdown().unwrap();
    }
}

#[test]
fn missing_receipt_keeps_unknown_permit_and_blocks_resume() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let (actor, executor) = setup(&path, Mode::MissingReceipt);
    assert!(executor.tick("parent").is_err());
    let parent = actor.get("parent".into()).unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Waiting);
    assert!(parent.attempt.is_none());
    assert!(actor.get("child".into()).unwrap().is_none());
    assert_eq!(ledger(&path), (0, 20, 1, 0));
    assert!(
        actor
            .execute(
                "parent".into(),
                command("resume", parent.revision, Command::Resume)
            )
            .is_err()
    );
    assert_eq!(ledger(&path), (0, 20, 1, 0));
    actor.shutdown().unwrap();
}

#[test]
fn precommitted_wait_is_accepted_once_only_when_receipt_matches() {
    for (mode, succeeds) in [
        (Mode::Precommitted, true),
        (Mode::PrecommittedMismatch, false),
    ] {
        let dir = TestDirectory::new();
        let path = dir.database();
        let (actor, executor) = setup(&path, mode);
        let result = executor.tick("parent");
        assert_eq!(result.is_ok(), succeeds);
        if succeeds {
            assert!(result.unwrap());
        }
        let parent = actor.get("parent".into()).unwrap().unwrap();
        assert_eq!(parent.status, RunStatus::Waiting);
        assert_eq!(parent.revision, 2);
        assert_eq!(parent.waiting_child.unwrap().child_id, "child");
        assert!(actor.get("child".into()).unwrap().is_some());
        assert_eq!(ledger(&path), (7, 0, 0, 1));
        assert_eq!(
            actor.events_after("parent".into(), 0, 100).unwrap().len(),
            2
        );
        actor.shutdown().unwrap();
    }
}
