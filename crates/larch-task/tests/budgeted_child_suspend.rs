use larch_task::{
    execution::{ActivityDescriptor, BudgetedChildSpawnRequest, ChildSpawnRequest, ExecutionPlan},
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
        let path = std::env::temp_dir().join(format!(
            "larch-budget-wait-{}-{}",
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
            grants: vec!["read".into()],
            budgets: BTreeMap::from([("model_tokens".into(), 30)]),
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
    let (_, claimed) = store
        .dispatch_budgeted_once(
            "parent",
            &command("start", 0, start("parent-attempt", 1)),
            20,
            1,
        )
        .unwrap();
    assert!(claimed);
}
fn request() -> BudgetedChildSpawnRequest {
    BudgetedChildSpawnRequest {
        spawn: ChildSpawnRequest {
            parent_id: "parent".into(),
            expected_revision: 1,
            command_id: "spawn-wait".into(),
            parent_attempt_id: "parent-attempt".into(),
            parent_epoch: 1,
            expected_graph_revision: 0,
            child: state("child"),
            plan: plan(),
        },
        used: 7,
        usage_receipt: "verified-parent-usage".into(),
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
fn assert_unchanged(store: &SqliteStore, path: &std::path::Path) {
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.revision, 1);
    assert_eq!(parent.status, RunStatus::Running);
    assert_eq!(parent.attempt.unwrap().id, "parent-attempt");
    assert!(parent.waiting_child.is_none());
    assert!(store.get("child").unwrap().is_none());
    assert!(store.execution_plan("child").unwrap().is_none());
    assert_eq!(store.root_id("child").unwrap(), "child");
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 1);
    assert_eq!(ledger(path), (0, 20, 1, 0));
}

#[test]
fn one_seat_handoff_survives_reopen_and_charges_both_attempts() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let reply = {
        let mut store = SqliteStore::open(&path).unwrap();
        setup(&mut store);
        let reply = store.spawn_child_and_wait_budgeted(&request()).unwrap();
        assert_eq!(reply.revision, 2);
        assert_eq!(reply.status, RunStatus::Waiting);
        assert_eq!(ledger(&path), (7, 0, 0, 1));
        reply
    };
    let mut store = SqliteStore::open(&path).unwrap();
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Waiting
    );
    assert_eq!(store.get("child").unwrap(), Some(state("child")));
    assert_eq!(store.execution_plan("child").unwrap(), Some(plan()));
    assert_eq!(store.root_id("child").unwrap(), "parent");
    assert_eq!(
        store.spawn_child_and_wait_budgeted(&request()).unwrap(),
        reply
    );
    assert_eq!(ledger(&path), (7, 0, 0, 1));
    let (_, claimed) = store
        .dispatch_budgeted_once(
            "child",
            &command("child-start", 0, start("child-attempt", 1)),
            15,
            1,
        )
        .unwrap();
    assert!(claimed);
    assert_eq!(ledger(&path), (7, 15, 1, 1));
    store
        .settle_budgeted(
            "child",
            &command(
                "child-finish",
                1,
                Command::FinishAttempt {
                    attempt_id: "child-attempt".into(),
                    epoch: 1,
                    result: "child evidence".into(),
                },
            ),
            8,
        )
        .unwrap();
    assert_eq!(ledger(&path), (15, 0, 0, 1));
    store
        .execute(
            "child",
            &command(
                "child-complete",
                2,
                Command::CompleteRun {
                    result: "child evidence".into(),
                },
            ),
        )
        .unwrap();
    store
        .execute(
            "parent",
            &command(
                "wake",
                2,
                Command::WakeChild {
                    child_id: "child".into(),
                    result: "child evidence".into(),
                },
            ),
        )
        .unwrap();
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Ready);
    assert_eq!(
        parent.node_results.get("step").map(String::as_str),
        Some("child evidence")
    );
    drop(store);
    let store = SqliteStore::open(&path).unwrap();
    assert_eq!(ledger(&path), (15, 0, 0, 1));
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Ready
    );
}

#[test]
fn missing_receipt_overrun_and_integer_overflow_leave_attempt_reserved() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let mut store = SqliteStore::open(&path).unwrap();
    setup(&mut store);
    let mut missing = request();
    missing.usage_receipt = "  ".into();
    assert!(store.spawn_child_and_wait_budgeted(&missing).is_err());
    assert_unchanged(&store, &path);
    let mut overrun = request();
    overrun.used = 21;
    assert!(store.spawn_child_and_wait_budgeted(&overrun).is_err());
    assert_unchanged(&store, &path);
    let mut overflow = request();
    overflow.used = u64::MAX;
    assert!(store.spawn_child_and_wait_budgeted(&overflow).is_err());
    assert_unchanged(&store, &path);
}

#[test]
fn conflicting_receipts_or_reused_command_cannot_mutate_ledger() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let mut store = SqliteStore::open(&path).unwrap();
    setup(&mut store);
    let original = store.spawn_child_and_wait_budgeted(&request()).unwrap();
    let mut other_receipt = request();
    other_receipt.usage_receipt = "different".into();
    assert!(store.spawn_child_and_wait_budgeted(&other_receipt).is_err());
    let mut other_usage = request();
    other_usage.used = 8;
    assert!(store.spawn_child_and_wait_budgeted(&other_usage).is_err());
    assert_eq!(
        store.spawn_child_and_wait_budgeted(&request()).unwrap(),
        original
    );
    assert_eq!(ledger(&path), (7, 0, 0, 1));
    assert_eq!(store.get("parent").unwrap().unwrap().revision, 2);
    assert_eq!(store.events_after("parent", 0, 100).unwrap().len(), 2);
}

#[test]
fn transaction_faults_leave_child_unpublished_and_parent_reservation_intact() {
    for trigger in [
        "CREATE TRIGGER injected BEFORE INSERT ON execution_plans BEGIN SELECT RAISE(ABORT,'plan failure'); END;",
        "CREATE TRIGGER injected BEFORE INSERT ON budget_receipts BEGIN SELECT RAISE(ABORT,'receipt failure'); END;",
        "CREATE TRIGGER injected BEFORE DELETE ON admission_permits BEGIN SELECT RAISE(ABORT,'permit failure'); END;",
        "CREATE TRIGGER injected BEFORE UPDATE ON runs WHEN NEW.id='parent' BEGIN SELECT RAISE(ABORT,'parent failure'); END;",
    ] {
        let dir = TestDirectory::new();
        let path = dir.database();
        let mut store = SqliteStore::open(&path).unwrap();
        setup(&mut store);
        let fault = rusqlite::Connection::open(&path).unwrap();
        fault.execute_batch(trigger).unwrap();
        assert!(store.spawn_child_and_wait_budgeted(&request()).is_err());
        assert_unchanged(&store, &path);
        fault.execute_batch("DROP TRIGGER injected;").unwrap();
        store.spawn_child_and_wait_budgeted(&request()).unwrap();
        assert_eq!(ledger(&path), (7, 0, 0, 1));
    }
}

#[test]
fn stale_parent_attempt_or_graph_preserves_reservation_and_publishes_nothing() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let mut store = SqliteStore::open(&path).unwrap();
    setup(&mut store);
    let mut stale_attempt = request();
    stale_attempt.spawn.parent_attempt_id = "replaced".into();
    assert!(store.spawn_child_and_wait_budgeted(&stale_attempt).is_err());
    assert_unchanged(&store, &path);
    let mut stale_epoch = request();
    stale_epoch.spawn.parent_epoch = 2;
    assert!(store.spawn_child_and_wait_budgeted(&stale_epoch).is_err());
    assert_unchanged(&store, &path);
    let mut stale_graph = request();
    stale_graph.spawn.expected_graph_revision = 1;
    assert!(store.spawn_child_and_wait_budgeted(&stale_graph).is_err());
    assert_unchanged(&store, &path);
}

#[test]
fn unknown_usage_recovery_does_not_release_seat_or_publish_child() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let mut store = SqliteStore::open(&path).unwrap();
    setup(&mut store);
    store
        .execute("parent", &command("recover", 1, Command::Recover))
        .unwrap();
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Waiting);
    assert!(parent.attempt.is_none());
    assert!(store.get("child").unwrap().is_none());
    assert_eq!(ledger(&path), (0, 20, 1, 0));
    drop(store);
    let store = SqliteStore::open(&path).unwrap();
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Waiting
    );
    assert_eq!(ledger(&path), (0, 20, 1, 0));
}

#[test]
fn actor_accepts_budgeted_spawn_and_wait_idempotently() {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    actor.create("create".into(), state("parent")).unwrap();
    actor
        .dispatch_budgeted_once(
            "parent".into(),
            command("start", 0, start("parent-attempt", 1)),
            20,
            1,
        )
        .unwrap();
    let reply = actor.spawn_child_and_wait_budgeted(request()).unwrap();
    assert_eq!(reply.status, RunStatus::Waiting);
    assert_eq!(actor.execution_plan("child".into()).unwrap(), Some(plan()));
    assert_eq!(
        actor.spawn_child_and_wait_budgeted(request()).unwrap(),
        reply
    );
    actor.shutdown().unwrap();
}

#[test]
fn preexisting_conflicting_usage_receipt_blocks_the_handoff() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let mut store = SqliteStore::open(&path).unwrap();
    setup(&mut store);
    let db = rusqlite::Connection::open(&path).unwrap();
    let permit: String = db
        .query_row("SELECT id FROM admission_permits", [], |r| r.get(0))
        .unwrap();
    db.execute("INSERT INTO budget_receipts(permit_id,run_id,used,receipt) VALUES(?1,'parent',6,'earlier')", [&permit]).unwrap();
    assert!(store.spawn_child_and_wait_budgeted(&request()).is_err());
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Running);
    assert_eq!(parent.revision, 1);
    assert!(store.get("child").unwrap().is_none());
    assert!(store.execution_plan("child").unwrap().is_none());
    assert_eq!(ledger(&path), (0, 20, 1, 1));
    let old: (i64, String) = db
        .query_row(
            "SELECT used,receipt FROM budget_receipts WHERE permit_id=?1",
            [&permit],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(old, (6, "earlier".into()));
}

#[test]
fn cancellation_before_handoff_retains_unknown_usage_until_reconciled() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let mut store = SqliteStore::open(&path).unwrap();
    setup(&mut store);
    store
        .execute("parent", &command("cancel", 1, Command::Cancel))
        .unwrap();
    let mut proposal = request();
    proposal.spawn.expected_revision = 2;
    assert!(store.spawn_child_and_wait_budgeted(&proposal).is_err());
    assert!(store.get("child").unwrap().is_none());
    assert!(store.execution_plan("child").unwrap().is_none());
    assert_eq!(ledger(&path), (0, 20, 1, 0));
    store
        .reconcile_budgeted_usage("parent", "parent-attempt", 7, "verified-parent-usage")
        .unwrap();
    assert_eq!(ledger(&path), (7, 0, 0, 1));
    assert!(
        store
            .reconcile_budgeted_usage("parent", "parent-attempt", 8, "verified-parent-usage")
            .is_err()
    );
    assert!(
        store
            .reconcile_budgeted_usage("parent", "parent-attempt", 7, "different")
            .is_err()
    );
}

#[test]
fn cancellation_after_handoff_cancels_child_without_losing_receipt() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let mut store = SqliteStore::open(&path).unwrap();
    setup(&mut store);
    let reply = store.spawn_child_and_wait_budgeted(&request()).unwrap();
    store
        .execute("parent", &command("cancel", 2, Command::Cancel))
        .unwrap();
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    assert_eq!(
        store.get("child").unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    assert_eq!(ledger(&path), (7, 0, 0, 1));
    assert_eq!(
        store.spawn_child_and_wait_budgeted(&request()).unwrap(),
        reply
    );
    let mut changed = request();
    changed.usage_receipt = "different".into();
    assert!(store.spawn_child_and_wait_budgeted(&changed).is_err());
    let mut changed = request();
    changed.used = 8;
    assert!(store.spawn_child_and_wait_budgeted(&changed).is_err());
    assert_eq!(ledger(&path), (7, 0, 0, 1));
}
