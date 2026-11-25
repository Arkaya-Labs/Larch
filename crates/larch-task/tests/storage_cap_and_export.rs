use larch_task::{
    artifacts::ArtifactStore,
    store::SqliteStore,
    types::{
        Command, CommandEnvelope, PlanningPolicy, RunState, SCHEMA_VERSION, WorkflowDefinition,
    },
};
use std::{collections::BTreeMap, fs, path::PathBuf};
fn workspace() -> PathBuf {
    let p = std::env::temp_dir().join(format!("larch-ops-{}-{}", std::process::id(), {
        // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{t}-{n}")
    }));
    fs::create_dir_all(&p).unwrap();
    p
}
fn state() -> RunState {
    RunState::new(
        "run".into(),
        WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "portable goal".into(),
            initial_graph: None,
            planning_policy: PlanningPolicy::Initial,
            grants: vec!["write".into()],
            budgets: BTreeMap::new(),
            acceptance: vec!["check".into()],
        },
    )
    .unwrap()
}
#[test]
fn cap_blocks_new_publication_without_losing_existing_pins() {
    let dir = workspace();
    let objects = ArtifactStore::open(dir.join("objects")).unwrap();
    let mut s = SqliteStore::open(dir.join("db.sqlite")).unwrap();
    s.create(&state()).unwrap();
    s.set_storage_cap(&objects, 4).unwrap();
    assert!(objects.put(b"bypass").is_err());
    let first = s.put_pinned(&objects, "run", b"abc", "undo").unwrap();
    assert!(s.put_pinned(&objects, "run", b"de", "decision").is_err());
    assert_eq!(objects.read(&first).unwrap(), b"abc");
    assert!(!s.collect_unpinned(&objects).unwrap().contains(&first));
    drop(s);
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn workflow_export_strips_grants_and_deletion_respects_pins() {
    let dir = workspace();
    let objects = ArtifactStore::open(dir.join("objects")).unwrap();
    let mut s = SqliteStore::open(dir.join("db.sqlite")).unwrap();
    s.create(&state()).unwrap();
    let exported = s.export_workflow("run").unwrap();
    let imported = SqliteStore::import_workflow(&exported).unwrap();
    assert_eq!(imported.goal, "portable goal");
    assert!(imported.grants.is_empty());
    assert_eq!(imported.acceptance, vec!["check"]);
    let digest = s
        .put_pinned(&objects, "run", b"evidence", "evidence")
        .unwrap();
    assert!(s.delete_run("run").is_err());
    s.execute(
        "run",
        &CommandEnvelope {
            id: "cancel".into(),
            expected_revision: 0,
            command: Command::Cancel,
        },
    )
    .unwrap();
    assert!(s.delete_run("run").is_err());
    s.unpin_artifact("run", &digest, "evidence").unwrap();
    s.delete_run("run").unwrap();
    assert!(s.get("run").unwrap().is_none());
    drop(s);
    fs::remove_dir_all(dir).unwrap();
}
