use larch_task::{
    artifacts::ArtifactStore,
    store::SqliteStore,
    types::{PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION, WorkflowDefinition},
};
use std::{collections::BTreeMap, fs, path::PathBuf};
fn workspace() -> PathBuf {
    let p = std::env::temp_dir().join(format!("larch-bundle-{}-{}", std::process::id(), {
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
            goal: "backup".into(),
            initial_graph: None,
            planning_policy: PlanningPolicy::Initial,
            grants: vec!["local-write".into()],
            budgets: BTreeMap::new(),
            acceptance: vec![],
        },
    )
    .unwrap()
}
#[test]
fn referenced_artifacts_survive_collection_and_restore_is_paused_without_grants() {
    let dir = workspace();
    let db = dir.join("source.sqlite");
    let objects = ArtifactStore::open(dir.join("objects")).unwrap();
    let unused = objects.put(b"unused").unwrap();
    let mut store = SqliteStore::open(&db).unwrap();
    store.create(&state()).unwrap();
    let pinned = store
        .put_pinned(&objects, "run", b"pending decision", "pending_decision")
        .unwrap();
    let deleted = store.collect_unpinned(&objects).unwrap();
    assert!(deleted.contains(&unused));
    assert!(objects.read(&pinned).is_ok());
    let bundle = dir.join("bundle");
    store.backup_bundle(&objects, &bundle).unwrap();
    let restored_root = dir.join("restored");
    SqliteStore::restore_root(&bundle, &restored_root).unwrap();
    let restored = SqliteStore::open(restored_root.join("tasks.sqlite"))
        .unwrap()
        .get("run")
        .unwrap()
        .unwrap();
    assert_eq!(restored.status, RunStatus::Paused);
    assert!(restored.definition.grants.is_empty());
    assert_eq!(
        ArtifactStore::open(restored_root.join("objects"))
            .unwrap()
            .read(&pinned)
            .unwrap(),
        b"pending decision"
    );
    drop(store);
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn corrupt_bundle_does_not_publish_partial_restore() {
    let dir = workspace();
    let objects = ArtifactStore::open(dir.join("objects")).unwrap();
    let mut store = SqliteStore::open(dir.join("source.sqlite")).unwrap();
    store.create(&state()).unwrap();
    let digest = store
        .put_pinned(&objects, "run", b"reference", "undo")
        .unwrap();
    let bundle = dir.join("bundle");
    store.backup_bundle(&objects, &bundle).unwrap();
    fs::write(bundle.join("objects").join(&digest), b"corrupt").unwrap();
    let destination = dir.join("restored");
    assert!(SqliteStore::restore_root(&bundle, &destination).is_err());
    assert!(!destination.exists());
    drop(store);
    fs::remove_dir_all(dir).unwrap();
}
