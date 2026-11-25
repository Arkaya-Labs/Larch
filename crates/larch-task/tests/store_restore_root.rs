use larch_task::{artifacts::ArtifactStore, store::SqliteStore, types::*};
use std::{collections::BTreeMap, fs, path::PathBuf};
fn workspace() -> PathBuf {
    let path =
        std::env::temp_dir().join(format!("larch-restore-root-{}-{}", std::process::id(), {
            // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            format!("{t}-{n}")
        }));
    fs::create_dir_all(&path).unwrap();
    path
}
#[test]
fn restore_publishes_one_root_after_full_verification() {
    let dir = workspace();
    let objects = ArtifactStore::open(dir.join("objects")).unwrap();
    let mut store = SqliteStore::open(dir.join("source.sqlite")).unwrap();
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "recover".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: vec!["write".into()],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    store
        .create(&RunState::new("run".into(), definition).unwrap())
        .unwrap();
    let digest = store
        .put_pinned(&objects, "run", b"output", "result")
        .unwrap();
    let bundle = dir.join("bundle");
    store.backup_bundle(&objects, &bundle).unwrap();
    let root = dir.join("restored");
    SqliteStore::restore_root(&bundle, &root).unwrap();
    let state = SqliteStore::open(root.join("tasks.sqlite"))
        .unwrap()
        .get("run")
        .unwrap()
        .unwrap();
    assert_eq!(state.status, RunStatus::Paused);
    assert!(state.definition.grants.is_empty());
    assert_eq!(
        ArtifactStore::open(root.join("objects"))
            .unwrap()
            .read(&digest)
            .unwrap(),
        b"output"
    );
    drop(store);
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn corrupt_bundle_never_publishes_root() {
    let dir = workspace();
    let objects = ArtifactStore::open(dir.join("objects")).unwrap();
    let mut store = SqliteStore::open(dir.join("source.sqlite")).unwrap();
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "recover".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    store
        .create(&RunState::new("run".into(), definition).unwrap())
        .unwrap();
    let digest = store
        .put_pinned(&objects, "run", b"output", "result")
        .unwrap();
    let bundle = dir.join("bundle");
    store.backup_bundle(&objects, &bundle).unwrap();
    fs::write(bundle.join("objects").join(digest), b"broken").unwrap();
    let root = dir.join("restored");
    assert!(SqliteStore::restore_root(&bundle, &root).is_err());
    assert!(!root.exists());
    drop(store);
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn a_restored_in_flight_intent_reconciles_then_resumes() {
    let dir = workspace();
    let objects = ArtifactStore::open(dir.join("objects")).unwrap();
    let mut store = SqliteStore::open(dir.join("source.sqlite")).unwrap();
    let node = larch_task::graph::Node {
        id: "step".into(),
        depends_on: vec![],
        kind: larch_task::graph::NodeKind::Activity,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
        when: None,
        optional: vec![],
    };
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "recover".into(),
        initial_graph: Some(larch_task::graph::Graph { nodes: vec![node] }),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    store
        .create(&RunState::new("run".into(), definition).unwrap())
        .unwrap();
    for (revision, command) in [
        Command::StartAttempt {
            attempt: Attempt {
                id: "a".into(),
                node_id: "step".into(),
                epoch: 1,
                iteration: None,
            },
        },
        Command::RecordIntent {
            effect_id: "write".into(),
            attempt_id: "a".into(),
            epoch: 1,
            operation_digest: "digest".into(),
        },
    ]
    .into_iter()
    .enumerate()
    {
        store
            .execute(
                "run",
                &CommandEnvelope {
                    id: format!("c{revision}"),
                    expected_revision: revision as u64,
                    command,
                },
            )
            .unwrap();
    }
    let bundle = dir.join("bundle");
    store.backup_bundle(&objects, &bundle).unwrap();
    let root = dir.join("restored");
    SqliteStore::restore_root(&bundle, &root).unwrap();
    let mut restored = SqliteStore::open(root.join("tasks.sqlite")).unwrap();
    let paused = restored.get("run").unwrap().unwrap();
    assert_eq!(paused.status, RunStatus::Paused);
    assert_eq!(paused.effects["write"].status, EffectStatus::Uncertain);
    let control = |store: &mut SqliteStore, id: &str, command| {
        let revision = store.get("run").unwrap().unwrap().revision;
        store.execute(
            "run",
            &CommandEnvelope {
                id: id.into(),
                expected_revision: revision,
                command,
            },
        )
    };
    assert!(
        control(&mut restored, "early", Command::Resume).is_err(),
        "an uncertain effect blocks Resume"
    );
    control(
        &mut restored,
        "reconcile",
        Command::ReconcileEffect {
            effect_id: "write".into(),
            receipt: None,
        },
    )
    .unwrap();
    let resumed = control(&mut restored, "resume", Command::Resume).unwrap();
    assert_eq!(resumed.status, RunStatus::Ready);
    // Windows deletes no file that is still open.
    drop((store, restored));
    fs::remove_dir_all(dir).unwrap();
}
