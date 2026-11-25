#![cfg(unix)]
use larch_task::{
    host::{self, Action, MAX_FRAME, PROTOCOL_VERSION, Request, Response},
    ipc::{Client, read_frame, write_frame},
    types::{PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION, WorkflowDefinition},
};
use std::{
    fs,
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
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
            "larch-host-life-{}-{}",
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
    fn start(dir: &Path) -> Self {
        let path = dir.to_owned();
        let target = path.clone();
        let (send, done) = mpsc::channel();
        let thread = thread::spawn(move || {
            let _ = send.send(host::serve(&target));
        });
        let server = Self {
            dir: path,
            done,
            thread: Some(thread),
        };
        let deadline = Instant::now() + Duration::from_secs(5);
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
            .recv_timeout(Duration::from_secs(5))
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
fn raw_request(dir: &Path, token: &str, action: Action) -> Response {
    let mut stream = UnixStream::connect(dir.join("runtime-service/service.sock")).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let request = Request {
        version: PROTOCOL_VERSION,
        token: token.into(),
        action,
    };
    write_frame(
        &mut stream,
        &serde_json::to_vec(&request).unwrap(),
        MAX_FRAME,
    )
    .unwrap();
    serde_json::from_slice(&read_frame(&mut stream, MAX_FRAME).unwrap()).unwrap()
}
fn definition() -> WorkflowDefinition {
    WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "saved goal".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: vec![],
        budgets: Default::default(),
        acceptance: vec![],
    }
}

#[test]
fn socket_authentication_and_directory_ownership_are_enforced() {
    let dir = TestDirectory::new();
    let mut server = Server::start(&dir.0);
    assert!(
        raw_request(&dir.0, "wrong-token", Action::List)
            .error
            .is_some()
    );
    let token = fs::read_to_string(dir.0.join("runtime-service/token")).unwrap();
    assert!(raw_request(&dir.0, &token, Action::List).error.is_none());
    let path = dir.0.clone();
    let (send, done) = mpsc::channel();
    let contender = thread::spawn(move || {
        let _ = send.send(host::serve(&path));
    });
    let second = done
        .recv_timeout(Duration::from_secs(5))
        .expect("second host did not reject ownership");
    assert!(second.is_err());
    contender.join().unwrap();
    server.stop();
}

#[test]
fn saved_paused_workflow_remains_paused_across_host_restart() {
    let dir = TestDirectory::new();
    let mut first = Server::start(&dir.0);
    let client = Client::connect(&dir.0).unwrap();
    let state = RunState::new("run".into(), definition()).unwrap();
    client.create("create", &state).unwrap();
    let saved = client.get("run").unwrap().unwrap();
    assert_eq!(saved.status, RunStatus::Paused);
    assert_eq!(saved.revision, 0);
    assert!(saved.graph.is_none());
    first.stop();
    let mut second = Server::start(&dir.0);
    let client = Client::connect(&dir.0).unwrap();
    let restored = client.get("run").unwrap().unwrap();
    assert_eq!(restored.status, RunStatus::Paused);
    assert_eq!(restored.revision, 0);
    assert_eq!(restored.definition.goal, "saved goal");
    assert!(restored.graph.is_none());
    second.stop();
}
