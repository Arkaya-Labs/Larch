use larch_task::{
    artifacts::ArtifactStore,
    store::SqliteStore,
    types::{PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION, WorkflowDefinition},
};
use std::{collections::BTreeMap, fs};
#[test]
fn restore_publishes_one_complete_directory_and_corruption_publishes_nothing() {
    let root = std::env::temp_dir().join(format!("larch-atomic-{}-{}", std::process::id(), {
        // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{t}-{n}")
    }));
    fs::create_dir_all(&root).unwrap();
    let mut store = SqliteStore::open(root.join("source.sqlite")).unwrap();
    let objects = ArtifactStore::open(root.join("source-objects")).unwrap();
    let def = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "goal".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: vec!["write".into()],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    store
        .create(&RunState::new("run".into(), def).unwrap())
        .unwrap();
    let digest = store
        .put_pinned(&objects, "run", b"contents", "evidence")
        .unwrap();
    let bundle = root.join("bundle");
    store.backup_bundle(&objects, &bundle).unwrap();
    let restored = root.join("restored");
    SqliteStore::restore_bundle(&bundle, &restored).unwrap();
    let reopened = SqliteStore::open(restored.join("tasks.sqlite")).unwrap();
    assert_eq!(
        reopened.get("run").unwrap().unwrap().status,
        RunStatus::Paused
    );
    assert_eq!(
        ArtifactStore::open(restored.join("objects"))
            .unwrap()
            .read(&digest)
            .unwrap(),
        b"contents"
    );
    drop(reopened);
    fs::write(bundle.join("objects").join(&digest), b"corrupt").unwrap();
    let rejected = root.join("rejected");
    assert!(SqliteStore::restore_bundle(&bundle, &rejected).is_err());
    assert!(!rejected.exists());
    drop(store);
    fs::remove_dir_all(root).unwrap();
}
