use larch_task::{store::SqliteStore, types::*};

fn initial() -> RunState {
    RunState::new(
        "run".into(),
        WorkflowDefinition {
            version: 1,
            goal: "goal".into(),
            initial_graph: None,
            planning_policy: PlanningPolicy::Initial,
            grants: vec![],
            budgets: Default::default(),
            acceptance: vec![],
        },
    )
    .unwrap()
}
#[test]
fn commands_are_deduplicated_and_conflicting_reuse_is_rejected() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    let state = initial();
    assert_eq!(
        store.create_idempotent("create", &state).unwrap(),
        store.create_idempotent("create", &state).unwrap()
    );
    let command = CommandEnvelope {
        id: "pause".into(),
        expected_revision: 0,
        command: Command::Pause,
    };
    let first = store.execute("run", &command).unwrap();
    assert_eq!(store.execute("run", &command).unwrap(), first);
    let conflicting = CommandEnvelope {
        command: Command::Cancel,
        ..command
    };
    assert!(store.execute("run", &conflicting).is_err());
    assert_eq!(store.events_after("run", 0, 100).unwrap().len(), 1);
    assert_eq!(store.get("run").unwrap().unwrap().revision, 1);
}
#[test]
fn cancelling_parent_cancels_child_atomically() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    let parent = initial();
    store.create_idempotent("create", &parent).unwrap();
    let mut child = initial();
    child.id = "child".into();
    store.spawn_child("run", 0, "spawn", &child).unwrap();
    store.message_child("run", "child", "m", "hello").unwrap();
    store.message_child("run", "child", "m", "hello").unwrap();
    assert!(
        store
            .message_child("run", "child", "m", "different")
            .is_err()
    );
    assert_eq!(
        store.message_child("run", "stranger", "m", "hello"),
        Err("unknown child".into())
    );
    store
        .execute(
            "run",
            &CommandEnvelope {
                id: "cancel".into(),
                expected_revision: 1,
                command: Command::Cancel,
            },
        )
        .unwrap();
    assert_eq!(
        store.get("child").unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    assert_eq!(
        store.collect_child("run", "child").unwrap().status,
        RunStatus::Cancelled
    );
}
#[test]
fn sqlite_rejects_cursor_outside_signed_range() {
    let store = SqliteStore::open(":memory:").unwrap();
    assert!(store.events_after("run", u64::MAX, 1).is_err());
}

#[test]
fn database_path_is_shared_under_runtime_service() {
    assert_eq!(
        larch_task::store::database_path("data"),
        std::path::Path::new("data")
            .join("runtime-service")
            .join("tasks.sqlite")
    );
}
#[test]
fn events_reads_past_the_page_cap() {
    let mut store = SqliteStore::open(":memory:").unwrap();
    store.create_idempotent("create", &initial()).unwrap();
    for revision in 0..1001u64 {
        let command = if revision % 2 == 0 {
            Command::Pause
        } else {
            Command::Resume
        };
        store
            .execute(
                "run",
                &CommandEnvelope {
                    id: format!("c{revision}"),
                    expected_revision: revision,
                    command,
                },
            )
            .unwrap();
    }
    let events = store.events("run").unwrap();
    assert_eq!(events.len(), 1001);
    assert_eq!(events.last().unwrap().revision, 1001);
    let mut done = initial();
    done.id = "done".into();
    store.create_idempotent("create-done", &done).unwrap();
    store
        .execute(
            "done",
            &CommandEnvelope {
                id: "cancel".into(),
                expected_revision: 0,
                command: Command::Cancel,
            },
        )
        .unwrap();
    assert_eq!(store.list().unwrap().len(), 2);
    let active: Vec<_> = store
        .list_active()
        .unwrap()
        .into_iter()
        .map(|r| r.id)
        .collect();
    assert_eq!(active, ["run"]);
}
