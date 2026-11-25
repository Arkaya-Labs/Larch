use larch_task::{
    graph::{Graph, Node, NodeKind},
    store::SqliteStore,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, RunState, SCHEMA_VERSION,
        WorkflowDefinition,
    },
};
use std::collections::BTreeMap;
fn path() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("larch-budget-{}-{}.sqlite", std::process::id(), {
        // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{t}-{n}")
    }))
}
fn run(id: &str) -> RunState {
    RunState::new(
        id.into(),
        WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "budgeted".into(),
            initial_graph: Some(Graph {
                nodes: vec![Node {
                    when: None,
                    optional: Vec::new(),
                    id: "node".into(),
                    depends_on: vec![],
                    kind: NodeKind::Activity,
                    inputs: BTreeMap::new(),
                    outputs: BTreeMap::new(),
                }],
            }),
            planning_policy: PlanningPolicy::Disabled,
            grants: vec![],
            budgets: BTreeMap::from([("model_tokens".into(), 10)]),
            acceptance: vec![],
        },
    )
    .unwrap()
}
fn start(id: &str) -> CommandEnvelope {
    CommandEnvelope {
        id: format!("start-{id}"),
        expected_revision: 0,
        command: Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: id.into(),
                node_id: "node".into(),
                epoch: 1,
            },
        },
    }
}
#[test]
fn children_cannot_reset_root_budget_and_unknown_reservations_survive_restart() {
    let file = path();
    {
        let mut s = SqliteStore::open(&file).unwrap();
        s.create(&run("root")).unwrap();
        s.spawn_child("root", 0, "child", &run("child")).unwrap();
        assert!(s.create_admission_root("child", 10).is_err());
        let root_start = CommandEnvelope {
            expected_revision: 1,
            ..start("root-attempt")
        };
        s.dispatch_budgeted("root", &root_start, 7, 2).unwrap();
        // Budgeted child-plan publication is not yet supported; seed the durable
        // origin only to exercise the shared ledger and fencing invariants.
        let db = rusqlite::Connection::open(&file).unwrap();
        db.execute(
            "INSERT INTO child_origins(parent_id,child_id,attempt_id,epoch,graph_revision,node_id) VALUES('root','child','root-attempt',1,0,'node')",
            [],
        )
        .unwrap();
        drop(db);
        assert!(
            s.dispatch_budgeted("child", &start("child-attempt"), 4, 2)
                .is_err()
        );
    }
    {
        let mut s = SqliteStore::open(&file).unwrap();
        assert!(
            s.dispatch_budgeted("child", &start("child-attempt"), 4, 2)
                .is_err()
        );
        assert!(
            s.dispatch_budgeted("child", &start("child-attempt"), 3, 2)
                .is_ok()
        );
        let finish = CommandEnvelope {
            id: "finish-root".into(),
            expected_revision: 2,
            command: Command::FinishAttempt {
                attempt_id: "root-attempt".into(),
                epoch: 1,
                result: "ok".into(),
            },
        };
        s.settle_budgeted("root", &finish, 7).unwrap();
        let child_finish = CommandEnvelope {
            id: "finish-child".into(),
            expected_revision: 1,
            command: Command::FinishAttempt {
                attempt_id: "child-attempt".into(),
                epoch: 1,
                result: "ok".into(),
            },
        };
        s.settle_budgeted("child", &child_finish, 3).unwrap();
        let db = rusqlite::Connection::open(&file).unwrap();
        let spent: i64 = db
            .query_row(
                "SELECT spent FROM admission_roots WHERE id='root'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(spent, 10);
    }
    let _ = std::fs::remove_file(file);
}
#[test]
fn cancelled_work_keeps_reservation_until_verified_usage_reconciles() {
    let file = path();
    let mut s = SqliteStore::open(&file).unwrap();
    s.create(&run("root")).unwrap();
    s.create(&run("other")).unwrap();
    s.dispatch_budgeted("root", &start("cancelled"), 7, 1)
        .unwrap();
    s.execute(
        "root",
        &CommandEnvelope {
            id: "cancel".into(),
            expected_revision: 1,
            command: Command::Cancel,
        },
    )
    .unwrap();
    let other_start = start("other-attempt");
    assert!(s.dispatch_budgeted("other", &other_start, 1, 1).is_err());
    assert!(
        s.reconcile_budgeted_usage("root", "cancelled", 4, "")
            .is_err()
    );
    s.reconcile_budgeted_usage("root", "cancelled", 4, "verified-provider-usage")
        .unwrap();
    s.reconcile_budgeted_usage("root", "cancelled", 4, "verified-provider-usage")
        .unwrap();
    assert!(
        s.reconcile_budgeted_usage("root", "cancelled", 5, "different")
            .is_err()
    );
    assert!(s.dispatch_budgeted("other", &other_start, 6, 1).is_ok());
    drop(s);
    let _ = std::fs::remove_file(file);
}
