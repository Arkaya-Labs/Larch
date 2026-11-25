use larch_task::{
    graph::{Graph, Node, NodeKind},
    store::SqliteStore,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, RunState, SCHEMA_VERSION,
        WorkflowDefinition,
    },
};
use std::collections::BTreeMap;
fn state(id: &str, budgeted: bool) -> RunState {
    RunState::new(
        id.into(),
        WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "test".into(),
            initial_graph: Some(Graph {
                nodes: vec![Node {
                    when: None,
                    optional: Vec::new(),
                    id: "work".into(),
                    depends_on: vec![],
                    kind: NodeKind::Activity,
                    inputs: BTreeMap::new(),
                    outputs: BTreeMap::new(),
                }],
            }),
            planning_policy: PlanningPolicy::Disabled,
            grants: vec![],
            budgets: if budgeted {
                BTreeMap::from([("model_tokens".into(), 10)])
            } else {
                BTreeMap::new()
            },
            acceptance: vec![],
        },
    )
    .unwrap()
}
#[test]
fn scheduled_claim_cannot_dispatch_child_of_paused_root() {
    for budgeted in [false, true] {
        let path = std::env::temp_dir().join(format!(
            "larch-root-claim-{}-{}-{}.sqlite",
            std::process::id(),
            budgeted,
            {
                // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                format!("{t}-{n}")
            }
        ));
        let mut s = SqliteStore::open(&path).unwrap();
        s.create(&state("root", budgeted)).unwrap();
        s.spawn_child("root", 0, "spawn", &state("child", budgeted))
            .unwrap();
        s.execute(
            "root",
            &CommandEnvelope {
                id: "pause".into(),
                expected_revision: 1,
                command: Command::Pause,
            },
        )
        .unwrap();
        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute(
            "INSERT INTO task_outbox(id,kind,payload,delivered) VALUES('out','start','child',0)",
            [],
        )
        .unwrap();
        db.execute("INSERT INTO child_origins(parent_id,child_id,attempt_id,epoch,graph_revision,node_id) VALUES('root','child','origin',1,0,'work')", []).unwrap();
        drop(db);
        let cmd = CommandEnvelope {
            id: "start".into(),
            expected_revision: 0,
            command: Command::StartAttempt {
                attempt: Attempt {
                    iteration: None,
                    id: "attempt".into(),
                    node_id: "work".into(),
                    epoch: 1,
                },
            },
        };
        if budgeted {
            assert!(
                s.claim_budgeted_scheduled_start_once("out", "child", &cmd, 1, 1)
                    .is_err()
            );
        } else {
            assert!(s.claim_scheduled_start_once("out", &cmd).is_err());
        }
        assert_eq!(s.pending_outbox(10).unwrap().len(), 1);
        assert!(s.get("child").unwrap().unwrap().attempt.is_none());
        drop(s);
        let _ = std::fs::remove_file(path);
    }
}
