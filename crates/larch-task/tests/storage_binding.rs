use larch_task::{
    artifacts::ArtifactStore,
    store::SqliteStore,
    types::{PlanningPolicy, RunState, SCHEMA_VERSION, WorkflowDefinition},
};
use std::{collections::BTreeMap, fs};
fn state(id: &str) -> RunState {
    RunState::new(
        id.into(),
        WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "goal".into(),
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
fn capped_artifact_root_rejects_a_second_database_but_allows_owner_reopen() {
    let root = std::env::temp_dir().join(format!("larch-binding-{}-{}", std::process::id(), {
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
    let artifacts = ArtifactStore::open(root.join("objects")).unwrap();
    {
        let mut owner = SqliteStore::open(root.join("owner.sqlite")).unwrap();
        owner.create(&state("owner")).unwrap();
        owner.set_storage_cap(&artifacts, 4).unwrap();
        owner
            .put_pinned(&artifacts, "owner", b"abc", "evidence")
            .unwrap();
    }
    {
        let mut other = SqliteStore::open(root.join("other.sqlite")).unwrap();
        other.create(&state("other")).unwrap();
        assert!(
            other
                .put_pinned(&artifacts, "other", b"d", "evidence")
                .is_err()
        );
        assert!(other.set_storage_cap(&artifacts, 100).is_err());
        assert!(other.collect_unpinned(&artifacts).is_err());
    }
    {
        let mut owner = SqliteStore::open(root.join("owner.sqlite")).unwrap();
        owner
            .put_pinned(&artifacts, "owner", b"d", "evidence")
            .unwrap();
        assert!(
            owner
                .put_pinned(&artifacts, "owner", b"e", "evidence")
                .is_err()
        );
    }
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn atomic_restore_rebinds_capped_objects_to_restored_database() {
    let root = std::env::temp_dir().join(format!(
        "larch-binding-restore-{}-{}",
        std::process::id(),
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
    fs::create_dir_all(&root).unwrap();
    let artifacts = ArtifactStore::open(root.join("objects")).unwrap();
    let bundle = root.join("bundle");
    {
        let mut owner = SqliteStore::open(root.join("owner.sqlite")).unwrap();
        owner.create(&state("owner")).unwrap();
        owner.set_storage_cap(&artifacts, 4).unwrap();
        owner
            .put_pinned(&artifacts, "owner", b"abc", "evidence")
            .unwrap();
        owner.backup_bundle(&artifacts, &bundle).unwrap();
    }
    let restored = root.join("restored");
    SqliteStore::restore_bundle(&bundle, &restored).unwrap();
    let restored_objects = ArtifactStore::open(restored.join("objects")).unwrap();
    {
        let mut other = SqliteStore::open(root.join("other.sqlite")).unwrap();
        other.create(&state("other")).unwrap();
        assert!(other.set_storage_cap(&restored_objects, 100).is_err());
    }
    {
        let mut owner = SqliteStore::open(restored.join("tasks.sqlite")).unwrap();
        owner
            .put_pinned(&restored_objects, "owner", b"d", "evidence")
            .unwrap();
    }
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn an_uncapped_root_belongs_to_its_first_database_and_moves_with_it() {
    let root = std::env::temp_dir().join(format!(
        "larch-binding-uncapped-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    {
        let artifacts = ArtifactStore::open(root.join("objects")).unwrap();
        let mut owner = SqliteStore::open(root.join("owner.sqlite")).unwrap();
        owner.create(&state("owner")).unwrap();
        let kept = owner
            .put_pinned(&artifacts, "owner", b"owner evidence", "evidence")
            .unwrap();
        let mut other = SqliteStore::open(root.join("other.sqlite")).unwrap();
        other.create(&state("other")).unwrap();
        assert!(
            other
                .put_pinned(&artifacts, "other", b"other", "evidence")
                .is_err()
        );
        assert!(
            other.collect_unpinned(&artifacts).is_err(),
            "another database never collects the owner's objects"
        );
        assert!(artifacts.read(&kept).is_ok());
    }
    let moved = root.with_extension("moved");
    fs::rename(&root, &moved).unwrap();
    {
        let artifacts = ArtifactStore::open(moved.join("objects")).unwrap();
        let mut owner = SqliteStore::open(moved.join("owner.sqlite")).unwrap();
        owner
            .put_pinned(&artifacts, "owner", b"after the move", "evidence")
            .unwrap();
        assert!(owner.collect_unpinned(&artifacts).unwrap().is_empty());
    }
    fs::remove_dir_all(moved).unwrap();
}
