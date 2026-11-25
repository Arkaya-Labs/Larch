#![cfg(unix)]
use larch_agent::runtime::{
    ipc::{read_frame, write_frame},
    service::{Action, MAX_FRAME, PROTOCOL_VERSION, Request, Response},
};
use std::{
    fs,
    os::unix::net::UnixStream,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

/// A spawned service that is killed if its test panics before shutting it down: a leaked daemon
/// keeps its inherited descriptors (and any lock behind them) open after the test run ends.
struct Reap(std::process::Child);
impl Drop for Reap {
    fn drop(&mut self) {
        // Already exited after a clean shutdown; kill then only reports that, which is fine here.
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn send(socket: &std::path::Path, token: String, action: Action) -> Response {
    let mut stream = UnixStream::connect(socket).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let req = Request {
        version: PROTOCOL_VERSION,
        token,
        action,
    };
    write_frame(&mut stream, &serde_json::to_vec(&req).unwrap(), MAX_FRAME).unwrap();
    serde_json::from_slice(&read_frame(&mut stream, MAX_FRAME).unwrap()).unwrap()
}
#[test]
fn service_owns_directory_authenticates_and_shuts_down() {
    let dir = tempfile::tempdir().unwrap();
    let binary = env!("CARGO_BIN_EXE_larch-agent");
    let mut child = Command::new(binary)
        .args(["service", "--data-dir"])
        .arg(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(Reap)
        .unwrap();
    let runtime = dir.path().join("runtime-service");
    let socket = runtime.join("service.sock");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !socket.exists() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert!(socket.exists(), "service did not start");
    let token = fs::read_to_string(runtime.join("token")).unwrap();
    let other = Command::new(binary)
        .args(["service", "--data-dir"])
        .arg(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert!(!other.success());
    assert!(send(&socket, "wrong".into(), Action::List).error.is_some());
    assert!(send(&socket, token.clone(), Action::List).error.is_none());
    assert!(send(&socket, token, Action::Shutdown).error.is_none());
    assert!(child.0.wait().unwrap().success());
}

#[test]
fn client_submits_replays_and_controls_a_workflow_across_restart() {
    use larch_agent::runtime::ipc::Client;
    use larch_task::types::{
        Command as TaskCommand, CommandEnvelope, RunState, WorkflowDefinition,
    };
    let dir = tempfile::tempdir().unwrap();
    let binary = env!("CARGO_BIN_EXE_larch-agent");
    let launch = || {
        Command::new(binary)
            .args(["service", "--data-dir"])
            .arg(dir.path())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map(Reap)
            .unwrap()
    };
    let wait = || {
        let socket = dir.path().join("runtime-service/service.sock");
        let deadline = Instant::now() + Duration::from_secs(5);
        while !socket.exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(socket.exists());
    };
    let mut child = launch();
    wait();
    let client = Client::connect(dir.path()).unwrap();
    let definition: WorkflowDefinition = serde_json::from_value(
        serde_json::json!({"version":1, "goal":"write a report", "planning_policy":"initial"}),
    )
    .unwrap();
    let run = RunState::new("run-1".into(), definition).unwrap();
    assert_eq!(client.create("create-1", &run).unwrap().revision, 0);
    assert_eq!(
        client.get("run-1").unwrap().unwrap().definition.goal,
        "write a report"
    );
    // A saved run stays paused until it is explicitly started.
    assert_eq!(
        client.get("run-1").unwrap().unwrap().status,
        larch_task::types::RunStatus::Paused
    );
    let resume = CommandEnvelope {
        id: "resume-1".into(),
        expected_revision: 0,
        command: TaskCommand::Resume,
    };
    // Only Start or StartGoal starts a run, so the service's start checks always apply.
    let error = client.control("run-1", &resume).unwrap_err();
    assert!(error.contains("never started"), "{error}");
    let stale = CommandEnvelope {
        id: "stale-1".into(),
        expected_revision: 1,
        command: TaskCommand::Pause,
    };
    assert!(client.control("run-1", &stale).is_err());
    assert_eq!(client.list().unwrap().len(), 1);
    assert_eq!(client.events("run-1", 0, 10).unwrap().len(), 0);
    client.shutdown().unwrap();
    assert!(child.0.wait().unwrap().success());
    let mut child = launch();
    wait();
    let client = Client::connect(dir.path()).unwrap();
    assert_eq!(client.get("run-1").unwrap().unwrap().revision, 0);
    assert_eq!(client.events("run-1", 0, 10).unwrap().len(), 0);
    client.shutdown().unwrap();
    assert!(child.0.wait().unwrap().success());
}

#[test]
fn cli_creates_and_inspects_a_workflow() {
    use larch_agent::runtime::ipc::Client;
    let dir = tempfile::tempdir().unwrap();
    let binary = env!("CARGO_BIN_EXE_larch-agent");
    let mut child = Command::new(binary)
        .args(["service", "--data-dir"])
        .arg(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(Reap)
        .unwrap();
    let socket = dir.path().join("runtime-service/service.sock");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !socket.exists() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert!(socket.exists());
    let definition = dir.path().join("definition.json");
    fs::write(
        &definition,
        r#"{"version":1,"goal":"test task","planning_policy":"initial"}"#,
    )
    .unwrap();
    let create = Command::new(binary)
        .args(["task", "--data-dir"])
        .arg(dir.path())
        .args([
            "create",
            "--run-id",
            "r1",
            "--command-id",
            "c1",
            "--definition",
        ])
        .arg(&definition)
        .output()
        .unwrap();
    assert!(
        create.status.success(),
        "{}",
        String::from_utf8_lossy(&create.stderr)
    );
    let get = Command::new(binary)
        .args(["task", "--data-dir"])
        .arg(dir.path())
        .args(["get", "r1"])
        .output()
        .unwrap();
    assert!(
        get.status.success(),
        "{}",
        String::from_utf8_lossy(&get.stderr)
    );
    assert!(String::from_utf8_lossy(&get.stdout).contains("test task"));
    Client::connect(dir.path()).unwrap().shutdown().unwrap();
    assert!(child.0.wait().unwrap().success());
}

#[test]
fn client_reports_unavailable_service() {
    let dir = tempfile::tempdir().unwrap();
    assert!(larch_agent::runtime::ipc::Client::connect(dir.path()).is_err());
}

#[test]
fn cli_start_refuses_when_no_driver_is_configured() {
    use larch_agent::runtime::ipc::Client;
    let dir = tempfile::tempdir().unwrap();
    let binary = env!("CARGO_BIN_EXE_larch-agent");
    let mut child = Command::new(binary)
        .args(["service", "--data-dir"])
        .arg(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(Reap)
        .unwrap();
    let socket = dir.path().join("runtime-service/service.sock");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !socket.exists() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert!(socket.exists());
    let plan = dir.path().join("plan.json");
    fs::write(&plan, r#"{"graph_revision":0,"activities":{}}"#).unwrap();
    let out = Command::new(binary)
        .args(["task", "--data-dir"])
        .arg(dir.path())
        .args(["start", "r1", "--plan"])
        .arg(plan)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("activity execution is not configured"));
    Client::connect(dir.path()).unwrap().shutdown().unwrap();
    assert!(child.0.wait().unwrap().success());
}

#[test]
fn authenticated_client_can_revoke_a_saved_grant() {
    use larch_agent::runtime::ipc::Client;
    use larch_task::types::{
        Command as TaskCommand, CommandEnvelope, RunState, WorkflowDefinition,
    };
    let dir = tempfile::tempdir().unwrap();
    let binary = env!("CARGO_BIN_EXE_larch-agent");
    let mut child = Command::new(binary)
        .args(["service", "--data-dir"])
        .arg(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(Reap)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let client = loop {
        if let Ok(client) = Client::connect(dir.path()) {
            break client;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(20));
    };
    let definition: WorkflowDefinition = serde_json::from_value(serde_json::json!({"version":1,"goal":"test", "planning_policy":"initial", "grants":["write:test"]})).unwrap();
    client
        .create("create", &RunState::new("r".into(), definition).unwrap())
        .unwrap();
    client
        .control(
            "r",
            &CommandEnvelope {
                id: "revoke".into(),
                expected_revision: 0,
                command: TaskCommand::RevokeGrant {
                    grant: "write:test".into(),
                },
            },
        )
        .unwrap();
    assert!(
        client
            .get("r")
            .unwrap()
            .unwrap()
            .definition
            .grants
            .is_empty()
    );
    client.shutdown().unwrap();
    assert!(child.0.wait().unwrap().success());
}

#[test]
fn device_lease_is_exclusive_and_released_on_drop() {
    use larch_agent::runtime::service::DeviceLease;
    let dir = tempfile::tempdir().unwrap();
    let owner = DeviceLease::acquire(dir.path()).unwrap();
    assert!(DeviceLease::acquire(dir.path()).is_err());
    drop(owner);
    assert!(DeviceLease::acquire(dir.path()).is_ok());
}

#[test]
fn a_model_service_refuses_a_data_folder_whose_lease_is_held() {
    let dir = tempfile::tempdir().unwrap();
    let _held = larch_agent::runtime::service::DeviceLease::acquire(dir.path()).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_larch-agent"))
        .args(["service", "--data-dir"])
        .arg(dir.path())
        .args(["--model", "missing-local-model", "--cpu"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("already owned"), "{stderr}");
}

#[test]
fn cli_reconcile_lets_a_run_with_an_uncertain_effect_resume() {
    use larch_agent::runtime::ipc::Client;
    use larch_task::{
        store::{SqliteStore, database_path},
        types::{Attempt, Command as TaskCommand, CommandEnvelope, EffectStatus, RunState},
    };
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    // A run that died mid-effect: the service's start recovers its attempt, leaving the
    // effect uncertain and the run waiting.
    fs::create_dir(dir.path().join("runtime-service")).unwrap();
    fs::set_permissions(
        dir.path().join("runtime-service"),
        fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    {
        let mut store = SqliteStore::open(database_path(dir.path())).unwrap();
        let definition = serde_json::from_value(
            serde_json::json!({"version":1,"goal":"test","planning_policy":"initial"}),
        )
        .unwrap();
        store
            .create(&RunState::new("r".into(), definition).unwrap())
            .unwrap();
        for (revision, command) in [
            TaskCommand::StartAttempt {
                attempt: Attempt {
                    iteration: None,
                    id: "a".into(),
                    node_id: "__planning__".into(),
                    epoch: 1,
                },
            },
            TaskCommand::RecordIntent {
                effect_id: "e".into(),
                attempt_id: "a".into(),
                epoch: 1,
                operation_digest: "search".into(),
            },
        ]
        .into_iter()
        .enumerate()
        {
            store
                .execute(
                    "r",
                    &CommandEnvelope {
                        id: format!("c{revision}"),
                        expected_revision: revision as u64,
                        command,
                    },
                )
                .unwrap();
        }
    }
    let binary = env!("CARGO_BIN_EXE_larch-agent");
    let mut child = Command::new(binary)
        .args(["service", "--data-dir"])
        .arg(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(Reap)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let client = loop {
        if let Ok(client) = Client::connect(dir.path()) {
            break client;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(20));
    };
    let state = client.get("r").unwrap().unwrap();
    assert_eq!(state.effects["e"].status, EffectStatus::Uncertain);
    let task = |args: &[&str]| {
        let out = Command::new(binary)
            .args(["task", "--data-dir"])
            .arg(dir.path())
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    let revision = state.revision.to_string();
    task(&[
        "reconcile",
        "r",
        "e",
        "--command-id",
        "rec",
        "--revision",
        &revision,
    ]);
    let state = client.get("r").unwrap().unwrap();
    assert_eq!(state.effects["e"].status, EffectStatus::ReconciledAbsent);
    let revision = state.revision.to_string();
    task(&[
        "resume",
        "r",
        "--command-id",
        "res",
        "--revision",
        &revision,
    ]);
    client.shutdown().unwrap();
    assert!(child.0.wait().unwrap().success());
}

#[test]
fn model_service_reports_missing_model_before_accepting_work() {
    let dir = tempfile::tempdir().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_larch-agent"))
        .args(["service", "--data-dir"])
        .arg(dir.path())
        .args(["--model", "missing-local-model", "--cpu"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("model not found"));
}

#[test]
fn cli_can_list_saved_schedules_without_enabling_execution() {
    use larch_agent::runtime::ipc::Client;
    let dir = tempfile::tempdir().unwrap();
    let binary = env!("CARGO_BIN_EXE_larch-agent");
    let mut child = Command::new(binary)
        .args(["service", "--data-dir"])
        .arg(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(Reap)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let client = loop {
        if let Ok(client) = Client::connect(dir.path()) {
            break client;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(20));
    };
    let out = Command::new(binary)
        .args(["task", "--data-dir"])
        .arg(dir.path())
        .arg("schedules")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "[]");
    client.shutdown().unwrap();
    assert!(child.0.wait().unwrap().success());
}

#[test]
fn authenticated_retry_child_reaches_the_actor_and_deduplicates() {
    use larch_agent::runtime::{ipc::Client, service};
    use larch_task::{
        execution::{
            ActivityDescriptor, ActivityOutcome, ActivityRequest, ChildSpawnRequest, Driver,
            ExecutionPlan, Executor,
        },
        graph::{Graph, Node, NodeKind},
        runtime::ActorHandle,
        types::{
            Attempt, ChildFailureKind, Command as TaskCommand, CommandEnvelope, PlanningPolicy,
            RunState, RunStatus, SCHEMA_VERSION, WorkflowDefinition,
        },
    };
    use std::{collections::BTreeMap, sync::Arc};
    struct Never;
    impl Driver for Never {
        fn execute(&self, _: ActivityRequest) -> Result<ActivityOutcome, String> {
            Err("unexpected driver call".into())
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let db = larch_task::store::database_path(dir.path());
    std::fs::create_dir_all(db.parent().unwrap()).unwrap();
    let actor = ActorHandle::start(&db, 4).unwrap();
    let node = Node {
        when: None,
        optional: Vec::new(),
        id: "step".into(),
        depends_on: vec![],
        kind: NodeKind::Activity,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
    };
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "test".into(),
        initial_graph: Some(Graph { nodes: vec![node] }),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let parent = RunState::new("parent".into(), definition.clone()).unwrap();
    let child = RunState::new("child".into(), definition).unwrap();
    let envelope = |id: &str, expected_revision: u64, command| CommandEnvelope {
        id: id.into(),
        expected_revision,
        command,
    };
    let attempt = |id: &str| TaskCommand::StartAttempt {
        attempt: Attempt {
            iteration: None,
            id: id.into(),
            node_id: "step".into(),
            epoch: 1,
        },
    };
    actor.create("create".into(), parent).unwrap();
    actor
        .execute(
            "parent".into(),
            envelope("start-parent", 0, attempt("parent-attempt")),
        )
        .unwrap();
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([(
            "step".into(),
            ActivityDescriptor {
                kind: "test".into(),
                payload: serde_json::Value::Null,
            },
        )]),
    };
    actor
        .spawn_child_and_wait(ChildSpawnRequest {
            parent_id: "parent".into(),
            expected_revision: 1,
            command_id: "spawn".into(),
            parent_attempt_id: "parent-attempt".into(),
            parent_epoch: 1,
            expected_graph_revision: 0,
            child,
            plan,
        })
        .unwrap();
    actor
        .execute(
            "child".into(),
            envelope("start-child", 0, attempt("child-attempt")),
        )
        .unwrap();
    actor
        .execute(
            "child".into(),
            envelope(
                "fail-child",
                1,
                TaskCommand::FailAttempt {
                    attempt_id: "child-attempt".into(),
                    epoch: 1,
                    reason: "child failed".into(),
                },
            ),
        )
        .unwrap();
    Executor::new(actor.clone(), Arc::new(Never))
        .tick("parent")
        .unwrap();
    assert_eq!(
        actor
            .get("parent".into())
            .unwrap()
            .unwrap()
            .child_failure
            .unwrap()
            .kind,
        ChildFailureKind::FailedAttempt
    );
    actor.shutdown().unwrap();
    let service_dir = dir.path().to_owned();
    let server = thread::spawn(move || service::serve(&service_dir));
    let deadline = Instant::now() + Duration::from_secs(5);
    let client = loop {
        if let Ok(client) = Client::connect(dir.path()) {
            break client;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(20));
    };
    let forged = envelope(
        "forged",
        3,
        TaskCommand::RecordChildFailure {
            child_id: "child".into(),
            kind: ChildFailureKind::Cancelled,
            reason: None,
        },
    );
    assert!(client.control("parent", &forged).is_err());
    let retry = envelope(
        "retry-child",
        3,
        TaskCommand::RetryChild {
            child_id: "child".into(),
        },
    );
    let first = client.control("parent", &retry).unwrap();
    assert_eq!(first.status, RunStatus::Ready);
    assert_eq!(client.control("parent", &retry).unwrap(), first);
    assert_eq!(
        client.get("child").unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    client.shutdown().unwrap();
    server.join().unwrap().unwrap();
}

#[test]
fn worker_reports_waited_child_failure_once_without_running_new_work() {
    use larch_agent::runtime::{ipc::Client, service};
    use larch_task::{
        execution::{
            ActivityDescriptor, ActivityOutcome, ActivityRequest, ChildSpawnRequest, Driver,
            ExecutionPlan,
        },
        graph::{Graph, Node, NodeKind},
        runtime::ActorHandle,
        types::{
            Attempt, ChildFailureKind, Command as TaskCommand, CommandEnvelope, PlanningPolicy,
            RunState, RunStatus, SCHEMA_VERSION, WorkflowDefinition,
        },
    };
    use std::{
        collections::BTreeMap,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };
    struct Never(Arc<AtomicUsize>);
    impl Driver for Never {
        fn execute(&self, _: ActivityRequest) -> Result<ActivityOutcome, String> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err("unexpected driver call".into())
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let db = larch_task::store::database_path(dir.path());
    std::fs::create_dir_all(db.parent().unwrap()).unwrap();
    let actor = ActorHandle::start(&db, 4).unwrap();
    let node = Node {
        when: None,
        optional: Vec::new(),
        id: "step".into(),
        depends_on: vec![],
        kind: NodeKind::Activity,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
    };
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "test".into(),
        initial_graph: Some(Graph { nodes: vec![node] }),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let envelope = |id: &str, expected_revision: u64, command| CommandEnvelope {
        id: id.into(),
        expected_revision,
        command,
    };
    let attempt = |id: &str| TaskCommand::StartAttempt {
        attempt: Attempt {
            iteration: None,
            id: id.into(),
            node_id: "step".into(),
            epoch: 1,
        },
    };
    actor
        .create(
            "create".into(),
            RunState::new("parent".into(), definition.clone()).unwrap(),
        )
        .unwrap();
    actor
        .execute(
            "parent".into(),
            envelope("start-parent", 0, attempt("parent-attempt")),
        )
        .unwrap();
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([(
            "step".into(),
            ActivityDescriptor {
                kind: "test".into(),
                payload: serde_json::Value::Null,
            },
        )]),
    };
    actor
        .spawn_child_and_wait(ChildSpawnRequest {
            parent_id: "parent".into(),
            expected_revision: 1,
            command_id: "spawn".into(),
            parent_attempt_id: "parent-attempt".into(),
            parent_epoch: 1,
            expected_graph_revision: 0,
            child: RunState::new("child".into(), definition).unwrap(),
            plan,
        })
        .unwrap();
    actor
        .execute(
            "child".into(),
            envelope("start-child", 0, attempt("child-attempt")),
        )
        .unwrap();
    actor
        .execute(
            "child".into(),
            envelope(
                "fail-child",
                1,
                TaskCommand::FailAttempt {
                    attempt_id: "child-attempt".into(),
                    epoch: 1,
                    reason: "child failed".into(),
                },
            ),
        )
        .unwrap();
    assert!(
        actor
            .get("parent".into())
            .unwrap()
            .unwrap()
            .child_failure
            .is_none()
    );
    actor.shutdown().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let driver = Arc::new(Never(calls.clone()));
    let service_dir = dir.path().to_owned();
    let server = thread::spawn(move || service::serve_with_driver(&service_dir, driver));
    let deadline = Instant::now() + Duration::from_secs(5);
    let client = loop {
        if let Ok(client) = Client::connect(dir.path()) {
            break client;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(20));
    };
    loop {
        let parent = client.get("parent").unwrap().unwrap();
        assert_eq!(parent.status, RunStatus::Waiting);
        if let Some(failure) = parent.child_failure {
            assert_eq!(failure.kind, ChildFailureKind::FailedAttempt);
            break;
        }
        assert!(Instant::now() < deadline, "waiting parent was not polled");
        thread::sleep(Duration::from_millis(20));
    }
    thread::sleep(Duration::from_millis(250));
    assert_eq!(
        client
            .events("parent", 0, 100)
            .unwrap()
            .iter()
            .filter(|event| event.kind == "child_wait_failed")
            .count(),
        1
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    client.shutdown().unwrap();
    server.join().unwrap().unwrap();
}
