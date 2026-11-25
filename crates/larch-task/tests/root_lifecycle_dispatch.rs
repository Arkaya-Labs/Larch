use larch_task::{
    graph::{Graph, Node, NodeKind},
    store::SqliteStore,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION,
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
                BTreeMap::from([("model_tokens".into(), 20)])
            } else {
                BTreeMap::new()
            },
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
fn start(id: &str, revision: u64) -> CommandEnvelope {
    envelope(
        id,
        revision,
        Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: id.into(),
                node_id: "work".into(),
                epoch: 1,
            },
        },
    )
}
fn dispatch(
    s: &mut SqliteStore,
    run: &str,
    command: &CommandEnvelope,
    budgeted: bool,
) -> Result<(), String> {
    if budgeted {
        s.dispatch_budgeted(run, command, 1, 2).map(|_| ())
    } else {
        s.execute(run, command).map(|_| ())
    }
}
fn finish(s: &mut SqliteStore, command: &CommandEnvelope, budgeted: bool) {
    if budgeted {
        s.settle_budgeted("root", command, 1).unwrap();
    } else {
        s.execute("root", command).unwrap();
    }
}
fn prepare(budgeted: bool) -> SqliteStore {
    let mut s = SqliteStore::open(":memory:").unwrap();
    s.create(&state("root", budgeted)).unwrap();
    s.spawn_child("root", 0, "spawn", &state("child", budgeted))
        .unwrap();
    s
}
#[test]
fn paused_roots_cannot_dispatch_children() {
    for budgeted in [false, true] {
        let mut paused = prepare(budgeted);
        paused
            .execute("root", &envelope("pause", 1, Command::Pause))
            .unwrap();
        assert!(dispatch(&mut paused, "child", &start("child-paused", 0), budgeted).is_err());
        assert_eq!(
            paused.get("child").unwrap().unwrap().status,
            RunStatus::Ready
        );
    }
}
#[test]
fn root_waiting_after_failure_cannot_dispatch_child() {
    for budgeted in [false, true] {
        let mut s = prepare(budgeted);
        dispatch(&mut s, "root", &start("root-attempt", 1), budgeted).unwrap();
        finish(
            &mut s,
            &envelope(
                "fail",
                2,
                Command::FailAttempt {
                    attempt_id: "root-attempt".into(),
                    epoch: 1,
                    reason: "waiting".into(),
                },
            ),
            budgeted,
        );
        assert_eq!(s.get("root").unwrap().unwrap().status, RunStatus::Waiting);
        assert!(dispatch(&mut s, "child", &start("child-waiting", 0), budgeted).is_err());
    }
}
#[test]
fn terminal_persisted_root_cannot_dispatch_child() {
    for (budgeted, terminal) in [
        (false, RunStatus::Failed),
        (true, RunStatus::Failed),
        (false, RunStatus::Succeeded),
        (true, RunStatus::Succeeded),
    ] {
        let path = std::env::temp_dir().join(format!(
            "larch-failed-root-{}-{}-{}.sqlite",
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
        let mut root = s.get("root").unwrap().unwrap();
        root.status = terminal;
        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute(
            "UPDATE runs SET state=?1 WHERE id='root'",
            [serde_json::to_string(&root).unwrap()],
        )
        .unwrap();
        drop(db);
        assert!(dispatch(&mut s, "child", &start("child-failed", 0), budgeted).is_err());
        drop(s);
        let _ = std::fs::remove_file(path);
    }
}
#[test]
fn waiting_for_legacy_unbound_child_is_rejected() {
    let mut s = prepare(false);
    dispatch(&mut s, "root", &start("root-attempt", 1), false).unwrap();
    assert!(
        s.execute(
            "root",
            &envelope(
                "wait",
                2,
                Command::WaitForChild {
                    attempt_id: "root-attempt".into(),
                    epoch: 1,
                    child_id: "child".into()
                }
            )
        )
        .is_err()
    );
}
#[test]
fn planning_attempt_cannot_start_after_root_is_paused() {
    for budgeted in [false, true] {
        let mut parent = state("root", budgeted);
        parent.definition.initial_graph = None;
        parent.graph = None;
        parent.definition.planning_policy = PlanningPolicy::Initial;
        let mut child = state("child", budgeted);
        child.definition.initial_graph = None;
        child.graph = None;
        child.definition.planning_policy = PlanningPolicy::Initial;
        let mut s = SqliteStore::open(":memory:").unwrap();
        s.create(&parent).unwrap();
        s.spawn_child("root", 0, "spawn", &child).unwrap();
        s.execute("root", &envelope("pause", 1, Command::Pause))
            .unwrap();
        let plan = envelope(
            "child-planning",
            0,
            Command::StartAttempt {
                attempt: Attempt {
                    iteration: None,
                    id: "planning".into(),
                    node_id: "__planning__".into(),
                    epoch: 1,
                },
            },
        );
        assert!(dispatch(&mut s, "child", &plan, budgeted).is_err());
    }
}

#[test]
fn root_cannot_complete_while_child_is_nonterminal() {
    for budgeted in [false, true] {
        let mut s = prepare(budgeted);
        dispatch(&mut s, "root", &start("root-attempt", 1), budgeted).unwrap();
        finish(
            &mut s,
            &envelope(
                "finish",
                2,
                Command::FinishAttempt {
                    attempt_id: "root-attempt".into(),
                    epoch: 1,
                    result: "ok".into(),
                },
            ),
            budgeted,
        );
        assert!(
            s.execute(
                "root",
                &envelope(
                    "complete",
                    3,
                    Command::CompleteRun {
                        result: "ok".into()
                    }
                )
            )
            .is_err()
        );
    }
}
#[test]
fn root_waiting_on_bound_child_allows_child_dispatch() {
    use larch_task::execution::{ActivityDescriptor, ChildSpawnRequest, ExecutionPlan};
    let mut s = SqliteStore::open(":memory:").unwrap();
    s.create(&state("root", false)).unwrap();
    dispatch(&mut s, "root", &start("root-attempt", 0), false).unwrap();
    let request = ChildSpawnRequest {
        parent_id: "root".into(),
        expected_revision: 1,
        command_id: "spawn-wait".into(),
        parent_attempt_id: "root-attempt".into(),
        parent_epoch: 1,
        expected_graph_revision: 0,
        child: state("child", false),
        plan: ExecutionPlan {
            graph_revision: 0,
            activities: BTreeMap::from([(
                "work".into(),
                ActivityDescriptor {
                    kind: "test".into(),
                    payload: serde_json::Value::Null,
                },
            )]),
        },
    };
    s.spawn_child_and_wait(&request).unwrap();
    assert_eq!(s.get("root").unwrap().unwrap().status, RunStatus::Waiting);
    assert!(dispatch(&mut s, "child", &start("child-waiting", 0), false).is_ok());
}
#[test]
fn child_pinned_to_replaced_parent_attempt_cannot_start() {
    use larch_task::execution::{ActivityDescriptor, ChildSpawnRequest, ExecutionPlan};
    let mut s = SqliteStore::open(":memory:").unwrap();
    s.create(&state("root", false)).unwrap();
    dispatch(&mut s, "root", &start("old", 0), false).unwrap();
    let request = ChildSpawnRequest {
        parent_id: "root".into(),
        expected_revision: 1,
        command_id: "spawn".into(),
        parent_attempt_id: "old".into(),
        parent_epoch: 1,
        expected_graph_revision: 0,
        child: state("child", false),
        plan: ExecutionPlan {
            graph_revision: 0,
            activities: BTreeMap::from([(
                "work".into(),
                ActivityDescriptor {
                    kind: "test".into(),
                    payload: serde_json::Value::Null,
                },
            )]),
        },
    };
    s.spawn_child_with_plan(&request).unwrap();
    s.execute("root", &envelope("recover", 2, Command::Recover))
        .unwrap();
    s.execute("root", &envelope("resume", 3, Command::Resume))
        .unwrap();
    s.execute(
        "root",
        &envelope(
            "start-new",
            4,
            Command::StartAttempt {
                attempt: Attempt {
                    iteration: None,
                    id: "new".into(),
                    node_id: "work".into(),
                    epoch: 2,
                },
            },
        ),
    )
    .unwrap();
    assert!(dispatch(&mut s, "child", &start("stale-child", 0), false).is_err());
}
#[test]
fn legacy_unbound_child_cannot_start_when_root_ready_or_running() {
    for budgeted in [false, true] {
        let mut ready = prepare(budgeted);
        assert!(dispatch(&mut ready, "child", &start("unbound-ready", 0), budgeted).is_err());
        let mut running = prepare(budgeted);
        dispatch(&mut running, "root", &start("root-attempt", 1), budgeted).unwrap();
        assert!(
            dispatch(
                &mut running,
                "child",
                &start("unbound-running", 0),
                budgeted
            )
            .is_err()
        );
    }
}
#[test]
fn origin_bound_child_cannot_start_under_inactive_or_unrelated_waiting_root() {
    for budgeted in [false, true] {
        for status in [
            RunStatus::Paused,
            RunStatus::Waiting,
            RunStatus::Succeeded,
            RunStatus::Failed,
        ] {
            let path = std::env::temp_dir().join(format!(
                "larch-origin-root-{}-{}-{}.sqlite",
                std::process::id(),
                budgeted,
                {
                    // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
                    static NEXT: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(0);
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
            let mut root = s.get("root").unwrap().unwrap();
            root.status = status;
            let db = rusqlite::Connection::open(&path).unwrap();
            db.execute(
                "UPDATE runs SET state=?1 WHERE id='root'",
                [serde_json::to_string(&root).unwrap()],
            )
            .unwrap();
            db.execute("INSERT INTO child_origins(parent_id,child_id,attempt_id,epoch,graph_revision,node_id) VALUES('root','child','origin',1,0,'work')",[]).unwrap();
            drop(db);
            assert!(dispatch(&mut s, "child", &start("origin-child", 0), budgeted).is_err());
            assert!(s.get("child").unwrap().unwrap().attempt.is_none());
            drop(s);
            let _ = std::fs::remove_file(path);
        }
    }
}
