//! Migration from v5 must preserve legacy data without fabricating a trusted child origin.
use larch_task::{
    execution::{ActivityDescriptor, ChildSpawnRequest, ExecutionPlan},
    graph::{Graph, Node, NodeKind},
    store::SqliteStore,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION,
        WorkflowDefinition,
    },
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
struct TestDirectory(PathBuf);
impl TestDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "larch-origin-migration-{}-{}",
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
            budgets: BTreeMap::new(),
            acceptance: vec![],
        },
    )
    .unwrap()
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

/// Construct a v5-shaped database with an old unbound child wait and a settled child. We remove
/// only the v6 table and optional v6 wait fields; no origin metadata is synthesized for old rows.
/// What the v8 migration adds, dropped to rebuild an older store.
const V8_OBJECTS: &str = "DROP TABLE step_stats; DROP TABLE missed_occurrences; DROP TABLE notification_claims; DROP INDEX occurrences_unfinished; ALTER TABLE occurrences DROP COLUMN finished; ALTER TABLE schedules DROP COLUMN pumped_at;";
fn make_v5(path: &Path) {
    {
        let mut store = SqliteStore::open(path).unwrap();
        store.create_idempotent("create", &state("parent")).unwrap();
        store
            .execute(
                "parent",
                &command("parent-start", 0, start("parent-attempt")),
            )
            .unwrap();
        let plan = ExecutionPlan {
            graph_revision: 0,
            activities: BTreeMap::from([(
                "step".into(),
                ActivityDescriptor {
                    kind: "script".into(),
                    payload: serde_json::Value::Null,
                },
            )]),
        };
        store
            .spawn_child_with_plan(&ChildSpawnRequest {
                parent_id: "parent".into(),
                expected_revision: 1,
                command_id: "spawn".into(),
                parent_attempt_id: "parent-attempt".into(),
                parent_epoch: 1,
                expected_graph_revision: 0,
                child: state("child"),
                plan,
            })
            .unwrap();
        store
            .execute("child", &command("child-start", 0, start("child-attempt")))
            .unwrap();
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
    }
    let db = rusqlite::Connection::open(path).unwrap();
    let json: String = db
        .query_row("SELECT state FROM runs WHERE id = 'parent'", [], |r| {
            r.get(0)
        })
        .unwrap();
    let mut parent: serde_json::Value = serde_json::from_str(&json).unwrap();
    parent["status"] = serde_json::json!("waiting");
    parent["revision"] = serde_json::json!(3);
    parent["attempt"] = serde_json::Value::Null;
    parent["waiting_child"] =
        serde_json::json!({"child_id": "child", "node_id": "step", "graph_revision": 0});
    db.execute(
        "UPDATE runs SET revision = 3, state = ?1 WHERE id = 'parent'",
        [serde_json::to_string(&parent).unwrap()],
    )
    .unwrap();
    db.execute_batch(&format!(
        "DROP TABLE IF EXISTS child_origins; {V8_OBJECTS} PRAGMA user_version = 5;"
    ))
    .unwrap();
}

#[test]
fn v5_unbound_wait_survives_migration_without_becoming_trusted() {
    let dir = TestDirectory::new();
    let path = dir.database();
    make_v5(&path);
    let mut store = SqliteStore::open(&path).unwrap();
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Waiting);
    assert_eq!(parent.revision, 3);
    let wait = parent.waiting_child.unwrap();
    assert_eq!(wait.child_id, "child");
    assert_eq!(wait.node_id, "step");
    assert_eq!(wait.origin_attempt_id, None);
    assert_eq!(wait.origin_epoch, None);
    assert_eq!(
        store.collect_child("parent", "child").unwrap().status,
        RunStatus::Succeeded
    );
    assert!(store.collect_waited_child("parent", "child").is_err());
    assert!(
        store
            .execute(
                "parent",
                &command(
                    "wake",
                    3,
                    Command::WakeChild {
                        child_id: "child".into(),
                        result: "evidence".into(),
                    }
                )
            )
            .is_err()
    );
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Waiting
    );
    drop(store);
    let db = rusqlite::Connection::open(&path).unwrap();
    let version: u32 = db
        .pragma_query_value(None, "user_version", |r| r.get(0))
        .unwrap();
    assert_eq!(version, 9);
    let origins: u32 = db
        .query_row("SELECT count(*) FROM child_origins", [], |r| r.get(0))
        .unwrap();
    assert_eq!(origins, 0);
}

#[test]
fn failed_v6_migration_preserves_v5_version_and_data_until_repaired() {
    let dir = TestDirectory::new();
    let path = dir.database();
    make_v5(&path);
    {
        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute_batch("CREATE TABLE child_origins (collision TEXT);")
            .unwrap();
    }
    assert!(SqliteStore::open(&path).is_err());
    {
        let db = rusqlite::Connection::open(&path).unwrap();
        let version: u32 = db
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .unwrap();
        assert_eq!(version, 5);
        let runs: u32 = db
            .query_row("SELECT count(*) FROM runs", [], |r| r.get(0))
            .unwrap();
        let links: u32 = db
            .query_row("SELECT count(*) FROM children", [], |r| r.get(0))
            .unwrap();
        assert_eq!((runs, links), (2, 1));
        db.execute_batch("DROP TABLE child_origins;").unwrap();
    }
    let store = SqliteStore::open(&path).unwrap();
    assert_eq!(
        store.get("parent").unwrap().unwrap().status,
        RunStatus::Waiting
    );
    assert_eq!(
        store.collect_child("parent", "child").unwrap().status,
        RunStatus::Succeeded
    );
    assert!(store.collect_waited_child("parent", "child").is_err());
}

#[test]
fn v6_rows_migrate_to_v7_with_failure_markers_defaulted() {
    let dir = TestDirectory::new();
    let path = dir.database();
    make_v5(&path);
    drop(SqliteStore::open(&path).unwrap());
    {
        let db = rusqlite::Connection::open(&path).unwrap();
        for id in ["parent", "child"] {
            let json: String = db
                .query_row("SELECT state FROM runs WHERE id=?1", [id], |r| r.get(0))
                .unwrap();
            let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
            value.as_object_mut().unwrap().remove("child_failure");
            value.as_object_mut().unwrap().remove("attempt_failure");
            db.execute(
                "UPDATE runs SET state=?1 WHERE id=?2",
                rusqlite::params![serde_json::to_string(&value).unwrap(), id],
            )
            .unwrap();
        }
        db.execute_batch(&format!("{V8_OBJECTS} PRAGMA user_version=6;"))
            .unwrap();
    }
    let store = SqliteStore::open(&path).unwrap();
    let parent = store.get("parent").unwrap().unwrap();
    assert_eq!(parent.status, RunStatus::Waiting);
    assert!(parent.child_failure.is_none());
    assert!(parent.attempt_failure.is_none());
    assert!(store.collect_waited_child("parent", "child").is_err());
    drop(store);
    let db = rusqlite::Connection::open(&path).unwrap();
    let version: u32 = db
        .pragma_query_value(None, "user_version", |r| r.get(0))
        .unwrap();
    assert_eq!(version, 9);
}
