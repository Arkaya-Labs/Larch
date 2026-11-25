//! Socket-level tests of the production local task host and authenticated client.
#![cfg(unix)]

use larch_task::{
    host::{self, Action, MAX_FRAME, PROTOCOL_VERSION, Request, Response},
    ipc::{self, Client},
    types::{
        ChildFailureKind, Command, CommandEnvelope, PlanningPolicy, RunState, RunStatus,
        SCHEMA_VERSION, WorkflowDefinition,
    },
};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    net::Shutdown,
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
const DEADLINE: Duration = Duration::from_secs(10);
const IO_TIMEOUT: Duration = Duration::from_secs(4);

struct Server {
    root: PathBuf,
    thread: Option<JoinHandle<Result<(), String>>>,
}

impl Server {
    fn start() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "vt-host-{}-{}-{nonce}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let host_root = root.clone();
        let thread = thread::spawn(move || host::serve(&host_root));
        let server = Self {
            root,
            thread: Some(thread),
        };
        let deadline = Instant::now() + DEADLINE;
        loop {
            if Client::connect(&server.root).is_ok() {
                return server;
            }
            assert!(Instant::now() < deadline, "task host did not become ready");
            assert!(
                !server.thread.as_ref().unwrap().is_finished(),
                "task host exited during startup"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn client(&self) -> Client {
        Client::connect(&self.root).expect("production client connects to host")
    }

    fn socket(&self) -> PathBuf {
        self.root.join("runtime-service/service.sock")
    }

    fn token(&self) -> String {
        fs::read_to_string(self.root.join("runtime-service/token")).unwrap()
    }

    fn stop(&mut self) {
        let Some(handle) = self.thread.take() else {
            return;
        };
        if !handle.is_finished() {
            let _ = self.client().shutdown();
        }
        let deadline = Instant::now() + DEADLINE;
        while !handle.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(handle.is_finished(), "task host did not shut down");
        handle.join().unwrap().unwrap();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // Do not block a failing test indefinitely if host startup or shutdown failed.
        if let Some(handle) = self.thread.take() {
            if !handle.is_finished()
                && let Ok(client) = Client::connect(&self.root)
            {
                let _ = client.shutdown();
            }
            let deadline = Instant::now() + Duration::from_secs(2);
            while !handle.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(10));
            }
            if handle.is_finished() {
                let _ = handle.join();
            }
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn connect(socket: &Path) -> UnixStream {
    let stream = UnixStream::connect(socket).unwrap();
    stream.set_read_timeout(Some(IO_TIMEOUT)).unwrap();
    stream.set_write_timeout(Some(IO_TIMEOUT)).unwrap();
    stream
}

fn request(socket: &Path, request: &Request) -> Response {
    let mut stream = connect(socket);
    let body = serde_json::to_vec(request).unwrap();
    ipc::write_frame(&mut stream, &body, MAX_FRAME).unwrap();
    let reply = ipc::read_frame(&mut stream, MAX_FRAME).unwrap();
    serde_json::from_slice(&reply).unwrap()
}

fn state() -> RunState {
    RunState::new(
        "run-1".into(),
        WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "Review the notes".into(),
            initial_graph: None,
            planning_policy: PlanningPolicy::Initial,
            grants: vec![],
            budgets: BTreeMap::new(),
            acceptance: vec![],
        },
    )
    .unwrap()
}

fn envelope(id: &str, revision: u64, command: Command) -> CommandEnvelope {
    CommandEnvelope {
        id: id.into(),
        expected_revision: revision,
        command,
    }
}

#[test]
fn invalid_token_and_version_cannot_mutate_the_store() {
    let mut server = Server::start();
    let client = server.client();
    client.create("create", &state()).unwrap();
    let cancel = || Action::Control {
        run_id: "run-1".into(),
        command: envelope("forged-cancel", 0, Command::Cancel),
    };
    let invalid_token = Request {
        version: PROTOCOL_VERSION,
        token: "not-the-service-token".into(),
        action: cancel(),
    };
    assert!(request(&server.socket(), &invalid_token).error.is_some());
    let invalid_version = Request {
        version: PROTOCOL_VERSION + 1,
        token: server.token(),
        action: cancel(),
    };
    assert!(request(&server.socket(), &invalid_version).error.is_some());
    let run = client.get("run-1").unwrap().unwrap();
    assert_eq!(run.status, RunStatus::Paused);
    assert_eq!(run.revision, 0);
    assert!(client.events("run-1", 0, 100).unwrap().is_empty());
    client.capabilities().unwrap();
    server.stop();
}

#[test]
fn oversized_and_truncated_frames_are_rejected_without_stalling_the_host() {
    let mut server = Server::start();
    let client = server.client();
    let mut oversized = connect(&server.socket());
    oversized.write_all(&u32::MAX.to_be_bytes()).unwrap();
    let response: Response =
        serde_json::from_slice(&ipc::read_frame(&mut oversized, MAX_FRAME).unwrap()).unwrap();
    assert!(response.error.is_some());
    client.capabilities().unwrap();

    let mut truncated = connect(&server.socket());
    truncated.write_all(&20u32.to_be_bytes()).unwrap();
    truncated.write_all(b"{}").unwrap();
    truncated.shutdown(Shutdown::Write).unwrap();
    let response: Response =
        serde_json::from_slice(&ipc::read_frame(&mut truncated, MAX_FRAME).unwrap()).unwrap();
    assert!(response.error.is_some());
    client.capabilities().unwrap();
    server.stop();
}

#[test]
fn clients_cannot_forge_receipts_or_child_failure() {
    let mut server = Server::start();
    let client = server.client();
    client.create("create", &state()).unwrap();
    let commands = [
        Command::RecordReceipt {
            effect_id: "effect".into(),
            attempt_id: "attempt".into(),
            epoch: 1,
            receipt: "forged".into(),
        },
        Command::RecordChildFailure {
            child_id: "child".into(),
            kind: ChildFailureKind::FailedAttempt,
            reason: Some("forged".into()),
        },
        Command::ReconcileEffect {
            effect_id: "effect".into(),
            receipt: Some("forged".into()),
        },
    ];
    for (index, command) in commands.into_iter().enumerate() {
        let error = client
            .control("run-1", &envelope(&format!("forged-{index}"), 0, command))
            .unwrap_err();
        assert!(error.contains("trusted") || error.contains("actor"));
    }
    let run = client.get("run-1").unwrap().unwrap();
    assert_eq!(run.status, RunStatus::Paused);
    assert_eq!(run.revision, 0);
    assert!(run.effects.is_empty());
    assert!(run.child_failure.is_none());
    assert!(client.events("run-1", 0, 100).unwrap().is_empty());
    // Saying an uncertain effect did not happen is the user's to say: it reaches the reducer.
    let absent = Command::ReconcileEffect {
        effect_id: "effect".into(),
        receipt: None,
    };
    let error = client
        .control("run-1", &envelope("absent", 0, absent))
        .unwrap_err();
    assert!(error.contains("unknown effect"), "{error}");
    server.stop();
}

#[test]
fn lost_reply_can_be_retried_without_repeating_a_committed_command() {
    let mut server = Server::start();
    let client = server.client();
    client.create("create", &state()).unwrap();
    let cancel = envelope("cancel-once", 0, Command::Cancel);
    let request = Request {
        version: PROTOCOL_VERSION,
        token: server.token(),
        action: Action::Control {
            run_id: "run-1".into(),
            command: cancel.clone(),
        },
    };
    let mut stream = connect(&server.socket());
    ipc::write_frame(
        &mut stream,
        &serde_json::to_vec(&request).unwrap(),
        MAX_FRAME,
    )
    .unwrap();
    // The client disappears after sending. The next connection is served only
    // after the host has processed this complete request.
    drop(stream);
    let retried = client.control("run-1", &cancel);
    if retried.is_err() {
        let deadline = Instant::now() + Duration::from_millis(500);
        while !server.thread.as_ref().unwrap().is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
    }
    assert!(
        !server.thread.as_ref().unwrap().is_finished(),
        "task host exited after a client disconnected before reading its reply"
    );
    let retried = retried.unwrap();
    assert_eq!(retried.revision, 1);
    assert_eq!(retried.status, RunStatus::Cancelled);
    let events = client.events("run-1", 0, 100).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].command_id, "cancel-once");
    assert!(
        client
            .control("run-1", &envelope("cancel-once", 0, Command::Pause))
            .is_err()
    );
    server.stop();
}

#[test]
fn repeated_disconnected_peers_do_not_stop_the_host() {
    let mut server = Server::start();
    let client = server.client();
    for _ in 0..16 {
        let stream = UnixStream::connect(server.socket()).unwrap();
        drop(stream);
    }
    let capabilities = client.capabilities();
    if capabilities.is_err() {
        let deadline = Instant::now() + Duration::from_millis(500);
        while !server.thread.as_ref().unwrap().is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
    }
    assert!(
        !server.thread.as_ref().unwrap().is_finished(),
        "task host exited after peers disconnected"
    );
    capabilities.unwrap();
    server.stop();
}

#[test]
fn client_disappearing_mid_frame_does_not_stop_the_host() {
    let mut server = Server::start();
    let client = server.client();
    let mut stream = connect(&server.socket());
    stream.write_all(&20u32.to_be_bytes()).unwrap();
    stream.write_all(b"{").unwrap();
    drop(stream);
    let capabilities = client.capabilities();
    if capabilities.is_err() {
        let deadline = Instant::now() + Duration::from_millis(500);
        while !server.thread.as_ref().unwrap().is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
    }
    assert!(
        !server.thread.as_ref().unwrap().is_finished(),
        "task host exited after an incomplete frame was abandoned"
    );
    capabilities.unwrap();
    server.stop();
}

#[test]
fn a_null_success_stays_distinct_from_an_empty_envelope() {
    let null = Response {
        result: Some(serde_json::Value::Null),
        error: None,
    };
    let encoded = serde_json::to_string(&null).unwrap();
    assert_eq!(encoded, r#"{"result":null}"#);
    let decoded: Response = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.result, Some(serde_json::Value::Null));
    let empty: Response = serde_json::from_str("{}").unwrap();
    assert!(empty.result.is_none() && empty.error.is_none());
}

#[test]
fn step_stats_past_one_frame_arrive_in_pages() {
    let server = Server::start();
    let client = server.client();
    assert!(client.capabilities_at_connect().step_stats);
    client.create("create", &state()).unwrap();
    // 8,000 small loop attempts: about 1.06 MB of stats, past the 1 MiB frame in one answer.
    let mut store =
        larch_task::store::SqliteStore::open(larch_task::store::database_path(&server.root))
            .unwrap();
    let at = chrono::Utc::now();
    for i in 0..8000 {
        let stat = larch_task::execution::StepStat {
            attempt_id: format!("attempt-{i:05}"),
            node_id: "loop".into(),
            started_at: at,
            finished_at: at,
            tokens: Some((i, 1)),
        };
        store.record_step("run-1", &stat).unwrap();
    }
    let stats = client.step_stats("run-1").unwrap();
    assert_eq!(stats.len(), 8000);
    assert_eq!(stats[7999].attempt_id, "attempt-07999");
    assert!(serde_json::to_vec(&stats).unwrap().len() > MAX_FRAME);
}
