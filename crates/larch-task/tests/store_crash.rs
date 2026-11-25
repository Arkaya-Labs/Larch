use larch_task::{store::SqliteStore, types::*};
use std::{
    collections::BTreeMap,
    process::{Command as Process, Stdio},
};
fn initial() -> RunState {
    RunState::new(
        "crash-run".into(),
        WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "recover".into(),
            initial_graph: None,
            planning_policy: PlanningPolicy::Initial,
            grants: vec![],
            budgets: BTreeMap::new(),
            acceptance: vec![],
        },
    )
    .unwrap()
}
#[test]
fn child_process_commit() {
    let Some(path) = std::env::var_os("LARCH_CRASH_TEST_DB") else {
        return;
    };
    let mut store = SqliteStore::open(path).unwrap();
    store.create_idempotent("create", &initial()).unwrap();
    store
        .execute(
            "crash-run",
            &CommandEnvelope {
                id: "pause".into(),
                expected_revision: 0,
                command: Command::Pause,
            },
        )
        .unwrap();
    std::process::abort();
}
#[test]
fn committed_command_survives_abrupt_process_exit() {
    let root = std::env::temp_dir().join(format!("larch-crash-{}-{}", std::process::id(), {
        // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{t}-{n}")
    }));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("state.sqlite");
    let status = Process::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_process_commit"])
        .env("LARCH_CRASH_TEST_DB", &path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(!status.success());
    let mut store = SqliteStore::open(&path).unwrap();
    let state = store.get("crash-run").unwrap().unwrap();
    assert_eq!(state.status, RunStatus::Paused);
    let retry = store
        .execute(
            "crash-run",
            &CommandEnvelope {
                id: "pause".into(),
                expected_revision: 0,
                command: Command::Pause,
            },
        )
        .unwrap();
    assert_eq!(retry.revision, 1);
    drop(store);
    std::fs::remove_dir_all(root).unwrap();
}
