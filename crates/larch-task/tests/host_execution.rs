#![cfg(unix)]
use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, ChildSpawnRequest, Driver,
        ExecutionPlan,
    },
    graph::{Graph, Node, NodeKind},
    host::{self, InitialPlanner},
    ipc::Client,
    runtime::ActorHandle,
    types::{
        ChildFailureKind, Command, CommandEnvelope, PlanningPolicy, RunState, RunStatus,
        SCHEMA_VERSION, WorkflowDefinition,
    },
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
static NEXT_DIR: AtomicU64 = AtomicU64::new(0);
struct TestDirectory(PathBuf);
impl TestDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "larch-host-execution-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct Server {
    dir: PathBuf,
    done: mpsc::Receiver<Result<(), String>>,
    thread: Option<JoinHandle<()>>,
}
impl Server {
    fn start<F>(dir: &Path, run: F) -> Self
    where
        F: FnOnce(PathBuf) -> Result<(), String> + Send + 'static,
    {
        let path = dir.to_owned();
        let target = path.clone();
        let (send, done) = mpsc::channel();
        let thread = thread::spawn(move || {
            let _ = send.send(run(target));
        });
        let server = Self {
            dir: path,
            done,
            thread: Some(thread),
        };
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            if Client::connect(&server.dir).is_ok() {
                break;
            }
            if let Ok(result) = server.done.try_recv() {
                panic!("host exited during startup: {result:?}");
            }
            assert!(Instant::now() < deadline, "host did not start");
            thread::sleep(Duration::from_millis(20));
        }
        server
    }
    fn stop(&mut self) {
        Client::connect(&self.dir).unwrap().shutdown().unwrap();
        self.done
            .recv_timeout(Duration::from_secs(8))
            .expect("host did not stop")
            .unwrap();
        self.thread.take().unwrap().join().unwrap();
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        if self.thread.is_some() {
            if let Ok(client) = Client::connect(&self.dir) {
                let _ = client.shutdown();
            }
            if self.done.recv_timeout(Duration::from_secs(5)).is_ok()
                && let Some(thread) = self.thread.take()
            {
                let _ = thread.join();
            }
        }
    }
}
fn graph() -> Graph {
    Graph {
        nodes: vec![Node {
            when: None,
            optional: Vec::new(),
            id: "step".into(),
            depends_on: vec![],
            kind: NodeKind::Activity,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }],
    }
}
fn definition(goal: &str, planned: bool) -> WorkflowDefinition {
    WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: goal.into(),
        initial_graph: (!planned).then(graph),
        planning_policy: if planned {
            PlanningPolicy::Initial
        } else {
            PlanningPolicy::Disabled
        },
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    }
}
fn plan(kind: &str, graph_revision: u64, payload: serde_json::Value) -> ExecutionPlan {
    ExecutionPlan {
        graph_revision,
        activities: BTreeMap::from([(
            "step".into(),
            ActivityDescriptor {
                kind: kind.into(),
                payload,
            },
        )]),
    }
}
fn wait_for(client: &Client, id: &str, predicate: impl Fn(&RunState) -> bool) -> RunState {
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        let state = client.get(id).unwrap().expect("run missing");
        if predicate(&state) {
            return state;
        }
        assert!(
            Instant::now() < deadline,
            "run {id} did not reach expected state: {:?}",
            state.status
        );
        thread::sleep(Duration::from_millis(20));
    }
}
struct Script;
impl Driver for Script {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        if request.descriptor.kind != "script" {
            return Err("unknown activity".into());
        }
        Ok(ActivityOutcome {
            output: "done".into(),
        })
    }
    fn registered_kinds(&self) -> Option<std::collections::BTreeSet<String>> {
        Some(["script".to_owned()].into())
    }
}
struct ScriptPlanner(Arc<AtomicUsize>);
impl InitialPlanner for ScriptPlanner {
    fn plan(&self, actor: &ActorHandle, run_id: &str) -> Result<(), String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let state = actor.get(run_id.into())?.ok_or("unknown run")?;
        let command = CommandEnvelope {
            id: "plan".into(),
            expected_revision: state.revision,
            command: Command::PatchGraph {
                graph: graph(),
                expected_graph_revision: 0,
                grants: state.definition.grants.clone(),
                acceptance: state.definition.acceptance.clone(),
            },
        };
        actor.install_initial_plan(
            run_id.into(),
            command,
            plan("script", 1, serde_json::Value::Null),
        )?;
        Ok(())
    }
}
struct Delegate {
    actor: ActorHandle,
}
impl Driver for Delegate {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        match request.descriptor.kind.as_str() {
            "delegate" => {
                let parent = self
                    .actor
                    .get(request.run_id.clone())?
                    .ok_or("missing parent")?;
                let child_id = format!("{}-child-{}", request.run_id, request.epoch);
                let first_fails = request
                    .descriptor
                    .payload
                    .get("fail_first")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let kind = if first_fails && request.epoch == 1 {
                    "child-fail"
                } else {
                    "child-pass"
                };
                self.actor.spawn_child_and_wait(ChildSpawnRequest {
                    parent_id: request.run_id.clone(),
                    expected_revision: parent.revision,
                    command_id: format!("spawn:{}", request.attempt_id),
                    parent_attempt_id: request.attempt_id.clone(),
                    parent_epoch: request.epoch,
                    expected_graph_revision: request.graph_revision,
                    child: RunState::new(child_id, definition("child", false))?,
                    plan: plan(kind, 0, serde_json::Value::Null),
                })?;
                Ok(ActivityOutcome {
                    output: String::new(),
                })
            }
            "child-pass" => Ok(ActivityOutcome {
                output: "child evidence".into(),
            }),
            "child-fail" => Err("scripted child failure".into()),
            _ => Err("unknown activity".into()),
        }
    }
    fn suspended_child(
        &self,
        request: &ActivityRequest,
        _: &Result<ActivityOutcome, String>,
    ) -> Option<String> {
        (request.descriptor.kind == "delegate")
            .then(|| format!("{}-child-{}", request.run_id, request.epoch))
    }
}
fn delegated_server(dir: &Path) -> Server {
    Server::start(dir, |path| {
        host::serve_with_factory(&path, |actor, _stop| {
            Ok(Arc::new(Delegate { actor }) as Arc<dyn Driver>)
        })
    })
}

#[test]
fn predefined_plan_runs_from_saved_paused_state_after_restart() {
    let dir = TestDirectory::new();
    let mut first = Server::start(&dir.0, |path| {
        host::serve_with_driver(&path, Arc::new(Script))
    });
    let client = Client::connect(&dir.0).unwrap();
    client
        .create(
            "create",
            &RunState::new("predefined".into(), definition("predefined", false)).unwrap(),
        )
        .unwrap();
    assert_eq!(
        client.get("predefined").unwrap().unwrap().status,
        RunStatus::Paused
    );
    first.stop();
    let mut second = Server::start(&dir.0, |path| {
        host::serve_with_driver(&path, Arc::new(Script))
    });
    let client = Client::connect(&dir.0).unwrap();
    let saved = client.get("predefined").unwrap().unwrap();
    assert_eq!(saved.status, RunStatus::Paused);
    assert_eq!(saved.revision, 0);
    let activity = plan("script", 0, serde_json::Value::Null);
    assert!(
        client
            .start_plan("predefined", &activity, "stale", 5)
            .is_err()
    );
    // A kind the driver does not run is refused before the immutable plan is stored.
    let unknown = plan("model", 0, serde_json::Value::Null);
    let error = client
        .start_plan("predefined", &unknown, "unknown-kind", 0)
        .unwrap_err();
    assert!(error.contains("not registered"), "{error}");
    assert_eq!(
        client.get("predefined").unwrap().unwrap().status,
        RunStatus::Paused
    );
    // A client can gate a start on the kinds the service names, from its connect answer.
    let offered = client.capabilities_at_connect();
    assert_eq!(offered, &client.capabilities().unwrap());
    assert_eq!(offered.activity_kinds, Some(["script".to_owned()].into()));
    // A control Resume cannot start the run around Start's checks.
    let resume = CommandEnvelope {
        id: "resume".into(),
        expected_revision: 0,
        command: Command::Resume,
    };
    let error = client.control("predefined", &resume).unwrap_err();
    assert!(error.contains("never started"), "{error}");
    client
        .start_plan("predefined", &activity, "start", 0)
        .unwrap();
    let completed = wait_for(&client, "predefined", |state| {
        state.status == RunStatus::Succeeded
    });
    assert_eq!(completed.result.as_deref(), Some("done"));
    second.stop();
}

#[test]
fn goal_only_run_is_planned_after_explicit_start_and_same_host_executes_it() {
    let dir = TestDirectory::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let planner_calls = calls.clone();
    let mut server = Server::start(&dir.0, move |path| {
        host::serve_with_planner(
            &path,
            Arc::new(Script),
            Arc::new(ScriptPlanner(planner_calls)),
        )
    });
    let client = Client::connect(&dir.0).unwrap();
    client
        .create(
            "create",
            &RunState::new("goal".into(), definition("planned", true)).unwrap(),
        )
        .unwrap();
    let saved = client.get("goal").unwrap().unwrap();
    assert_eq!(saved.status, RunStatus::Paused);
    assert!(saved.graph.is_none());
    thread::sleep(Duration::from_millis(300));
    let observed = client.get("goal").unwrap().unwrap();
    assert_eq!(observed.status, RunStatus::Paused);
    assert!(observed.graph.is_none());
    assert_eq!(observed.revision, saved.revision);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    client
        .start_goal("goal", "start-goal", saved.revision)
        .unwrap();
    let completed = wait_for(&client, "goal", |state| {
        state.status == RunStatus::Succeeded
    });
    assert_eq!(completed.result.as_deref(), Some("done"));
    assert!(completed.graph.is_some());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    server.stop();
}

#[test]
fn host_wakes_parent_after_scripted_child_succeeds() {
    let dir = TestDirectory::new();
    let mut server = delegated_server(&dir.0);
    let client = Client::connect(&dir.0).unwrap();
    client
        .create(
            "create",
            &RunState::new("parent".into(), definition("delegate", false)).unwrap(),
        )
        .unwrap();
    client
        .start_plan(
            "parent",
            &plan("delegate", 0, serde_json::json!({"fail_first": false})),
            "start",
            0,
        )
        .unwrap();
    let parent = wait_for(&client, "parent", |state| {
        state.status == RunStatus::Succeeded
    });
    assert_eq!(
        parent.node_results.get("step").map(String::as_str),
        Some("child evidence")
    );
    assert_eq!(
        client.get("parent-child-1").unwrap().unwrap().status,
        RunStatus::Succeeded
    );
    let events = client.events("parent", 0, 100).unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.kind == "child_wait_started")
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.kind == "child_wait_completed")
            .count(),
        1
    );
    server.stop();
}

#[test]
fn host_records_child_failure_once_then_explicit_retry_can_succeed() {
    let dir = TestDirectory::new();
    let mut server = delegated_server(&dir.0);
    let client = Client::connect(&dir.0).unwrap();
    client
        .create(
            "create",
            &RunState::new("parent".into(), definition("delegate", false)).unwrap(),
        )
        .unwrap();
    client
        .start_plan(
            "parent",
            &plan("delegate", 0, serde_json::json!({"fail_first": true})),
            "start",
            0,
        )
        .unwrap();
    let waiting = wait_for(&client, "parent", |state| state.child_failure.is_some());
    assert_eq!(waiting.status, RunStatus::Waiting);
    assert_eq!(
        waiting.child_failure.unwrap().kind,
        ChildFailureKind::FailedAttempt
    );
    assert_eq!(
        client
            .events("parent", 0, 100)
            .unwrap()
            .iter()
            .filter(|event| event.kind == "child_wait_failed")
            .count(),
        1
    );
    let retry = CommandEnvelope {
        id: "retry".into(),
        expected_revision: waiting.revision,
        command: Command::RetryChild {
            child_id: "parent-child-1".into(),
        },
    };
    assert_eq!(
        client.control("parent", &retry).unwrap().status,
        RunStatus::Ready
    );
    let completed = wait_for(&client, "parent", |state| {
        state.status == RunStatus::Succeeded
    });
    assert_eq!(
        completed.node_results.get("step").map(String::as_str),
        Some("child evidence")
    );
    assert_eq!(
        client.get("parent-child-1").unwrap().unwrap().status,
        RunStatus::Cancelled
    );
    assert_eq!(
        client.get("parent-child-2").unwrap().unwrap().status,
        RunStatus::Succeeded
    );
    assert_eq!(
        client
            .events("parent", 0, 100)
            .unwrap()
            .iter()
            .filter(|event| event.kind == "child_wait_failed")
            .count(),
        1
    );
    server.stop();
}

struct Panics;
impl Driver for Panics {
    fn execute(&self, _: ActivityRequest) -> Result<ActivityOutcome, String> {
        panic!("scripted driver panic");
    }
}

#[test]
fn a_panicked_worker_stops_offering_execution() {
    let dir = TestDirectory::new();
    let mut server = Server::start(&dir.0, |path| {
        host::serve_with_driver(&path, Arc::new(Panics))
    });
    let client = Client::connect(&dir.0).unwrap();
    assert!(client.capabilities_at_connect().execution);
    client
        .create(
            "create",
            &RunState::new("doomed".into(), definition("doomed", false)).unwrap(),
        )
        .unwrap();
    client
        .start_plan(
            "doomed",
            &plan("any", 0, serde_json::Value::Null),
            "start",
            0,
        )
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(8);
    while client.capabilities().unwrap().execution {
        assert!(Instant::now() < deadline, "execution still offered");
        thread::sleep(Duration::from_millis(20));
    }
    let error = client
        .start_plan(
            "doomed",
            &plan("any", 0, serde_json::Value::Null),
            "again",
            0,
        )
        .unwrap_err();
    assert!(error.contains("not configured"), "{error}");
    server.stop();
}

#[test]
fn serve_notify_reports_its_own_token_once_it_listens() {
    let dir = TestDirectory::new();
    // A folder an older build left open to others is made private before anything lands in it.
    let private = dir.0.join("runtime-service");
    fs::create_dir(&private).unwrap();
    fs::set_permissions(
        &private,
        std::os::unix::fs::PermissionsExt::from_mode(0o755),
    )
    .unwrap();
    let (ready, token) = mpsc::channel();
    let mut server = Server::start(&dir.0, move |path| {
        host::serve_notify(
            &path,
            Box::new(move |token| {
                let _ = ready.send(token.to_owned());
            }),
        )
    });
    let token = token.recv_timeout(Duration::from_secs(8)).unwrap();
    assert_eq!(fs::read_to_string(private.join("token")).unwrap(), token);
    let mode =
        std::os::unix::fs::PermissionsExt::mode(&fs::metadata(&private).unwrap().permissions());
    assert_eq!(mode & 0o777, 0o700);
    server.stop();
}
