use larch_task::{
    graph::{Graph, Node, NodeKind},
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
            "larch-failure-receipt-{}-{}",
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
fn state() -> RunState {
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
        "run".into(),
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
fn envelope(id: &str, revision: u64, command: Command) -> CommandEnvelope {
    CommandEnvelope {
        id: id.into(),
        expected_revision: revision,
        command,
    }
}
fn failure() -> CommandEnvelope {
    envelope(
        "failed",
        1,
        Command::FailAttempt {
            attempt_id: "attempt".into(),
            epoch: 1,
            reason: "driver failed".into(),
        },
    )
}
fn setup(store: &mut SqliteStore) {
    store.create_idempotent("create", &state()).unwrap();
    let start = envelope(
        "start",
        0,
        Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "attempt".into(),
                node_id: "step".into(),
                epoch: 1,
            },
        },
    );
    assert!(
        store
            .dispatch_budgeted_once("run", &start, 20, 1)
            .unwrap()
            .1
    );
}
fn ledger(path: &std::path::Path) -> (i64, i64, i64, i64) {
    let db = rusqlite::Connection::open(path).unwrap();
    let (spent, reserved) = db
        .query_row(
            "SELECT spent,reserved FROM admission_roots WHERE id='run'",
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
fn assert_pending(store: &SqliteStore, path: &std::path::Path) {
    let state = store.get("run").unwrap().unwrap();
    assert_eq!(state.status, RunStatus::Running);
    assert_eq!(state.revision, 1);
    assert_eq!(state.attempt.unwrap().id, "attempt");
    assert_eq!(ledger(path), (0, 20, 1, 0));
    assert_eq!(store.events_after("run", 0, 100).unwrap().len(), 1);
}

#[test]
fn failed_attempt_charges_actual_overrun_and_replays_exact_receipt() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let reply = {
        let mut store = SqliteStore::open(&path).unwrap();
        setup(&mut store);
        let reply = store
            .fail_budgeted_with_receipt("run", &failure(), 21, "verified-usage")
            .unwrap();
        assert_eq!(reply.status, RunStatus::Waiting);
        assert_eq!(reply.revision, 2);
        assert_eq!(ledger(&path), (21, 0, 0, 1));
        reply
    };
    let mut store = SqliteStore::open(&path).unwrap();
    assert_eq!(
        store
            .fail_budgeted_with_receipt("run", &failure(), 21, "verified-usage")
            .unwrap(),
        reply
    );
    assert!(
        store
            .fail_budgeted_with_receipt("run", &failure(), 20, "verified-usage")
            .is_err()
    );
    assert!(
        store
            .fail_budgeted_with_receipt("run", &failure(), 21, "different")
            .is_err()
    );
    assert_eq!(ledger(&path), (21, 0, 0, 1));
    assert_eq!(store.events_after("run", 0, 100).unwrap().len(), 2);
    let db = rusqlite::Connection::open(&path).unwrap();
    let receipt: (String, i64, String) = db
        .query_row("SELECT run_id,used,receipt FROM budget_receipts", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!(receipt, ("run".into(), 21, "verified-usage".into()));
}

#[test]
fn empty_receipt_stale_attempt_wrong_command_and_overflow_preserve_permit() {
    let dir = TestDirectory::new();
    let path = dir.database();
    let mut store = SqliteStore::open(&path).unwrap();
    setup(&mut store);
    assert!(
        store
            .fail_budgeted_with_receipt("run", &failure(), 7, " ")
            .is_err()
    );
    assert_pending(&store, &path);
    let stale = envelope(
        "stale",
        1,
        Command::FailAttempt {
            attempt_id: "attempt".into(),
            epoch: 2,
            reason: "bad".into(),
        },
    );
    assert!(
        store
            .fail_budgeted_with_receipt("run", &stale, 7, "receipt")
            .is_err()
    );
    assert_pending(&store, &path);
    let wrong = envelope(
        "wrong",
        1,
        Command::FinishAttempt {
            attempt_id: "attempt".into(),
            epoch: 1,
            result: "done".into(),
        },
    );
    assert!(
        store
            .fail_budgeted_with_receipt("run", &wrong, 7, "receipt")
            .is_err()
    );
    assert_pending(&store, &path);
    assert!(
        store
            .fail_budgeted_with_receipt("run", &failure(), u64::MAX, "receipt")
            .is_err()
    );
    assert_pending(&store, &path);
}

#[test]
fn each_transaction_failure_keeps_original_attempt_and_reservation() {
    for trigger in [
        "CREATE TRIGGER injected BEFORE INSERT ON budget_receipts BEGIN SELECT RAISE(ABORT,'receipt failure'); END;",
        "CREATE TRIGGER injected BEFORE DELETE ON admission_permits BEGIN SELECT RAISE(ABORT,'permit failure'); END;",
        "CREATE TRIGGER injected BEFORE UPDATE ON runs WHEN NEW.id='run' BEGIN SELECT RAISE(ABORT,'run failure'); END;",
    ] {
        let dir = TestDirectory::new();
        let path = dir.database();
        let mut store = SqliteStore::open(&path).unwrap();
        setup(&mut store);
        let fault = rusqlite::Connection::open(&path).unwrap();
        fault.execute_batch(trigger).unwrap();
        assert!(
            store
                .fail_budgeted_with_receipt("run", &failure(), 7, "verified-usage")
                .is_err()
        );
        assert_pending(&store, &path);
        fault.execute_batch("DROP TRIGGER injected;").unwrap();
        store
            .fail_budgeted_with_receipt("run", &failure(), 7, "verified-usage")
            .unwrap();
        assert_eq!(ledger(&path), (7, 0, 0, 1));
    }
}

#[test]
fn actor_commits_receipt_and_failure_once() {
    use larch_task::runtime::ActorHandle;
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    actor.create("create".into(), state()).unwrap();
    let start = envelope(
        "start",
        0,
        Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "attempt".into(),
                node_id: "step".into(),
                epoch: 1,
            },
        },
    );
    actor
        .dispatch_budgeted_once("run".into(), start, 20, 1)
        .unwrap();
    let reply = actor
        .fail_budgeted_with_receipt("run".into(), failure(), 21, "verified-usage".into())
        .unwrap();
    assert_eq!(reply.status, RunStatus::Waiting);
    assert_eq!(
        actor
            .fail_budgeted_with_receipt("run".into(), failure(), 21, "verified-usage".into())
            .unwrap(),
        reply
    );
    assert!(
        actor
            .fail_budgeted_with_receipt("run".into(), failure(), 20, "verified-usage".into())
            .is_err()
    );
    actor.shutdown().unwrap();
}
