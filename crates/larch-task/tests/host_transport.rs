//! The service's local transport on every platform it ships on: the Unix socket, and the named
//! pipe on Windows.
use larch_task::{
    host,
    ipc::{Client, read_frame},
    types::{PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION, WorkflowDefinition},
};
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

/// The service's endpoint, as the platform names it.
#[cfg(unix)]
fn endpoint(dir: &Path) -> PathBuf {
    dir.join("runtime-service").join("service.sock")
}
#[cfg(windows)]
fn endpoint(dir: &Path) -> PathBuf {
    std::fs::read_to_string(dir.join("runtime-service").join("service.pipe"))
        .unwrap()
        .into()
}
/// A raw connection to the endpoint, bypassing the client.
#[cfg(unix)]
fn raw(endpoint: &Path) -> std::io::Result<std::os::unix::net::UnixStream> {
    std::os::unix::net::UnixStream::connect(endpoint)
}
#[cfg(windows)]
fn raw(endpoint: &Path) -> std::io::Result<larch_sys::pipe::PipeStream> {
    larch_sys::pipe::PipeStream::connect(&endpoint.to_string_lossy(), Duration::from_millis(500))
}
/// A running service in a fresh directory, and a client that reached it.
fn start(tag: &str) -> (PathBuf, thread::JoinHandle<Result<(), String>>, Client) {
    let dir = std::env::temp_dir().join(format!("larch-transport-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let target = dir.clone();
    let server = thread::spawn(move || host::serve(&target));
    let deadline = Instant::now() + Duration::from_secs(10);
    let client = loop {
        match Client::connect(&dir) {
            Ok(client) => break client,
            Err(error) if Instant::now() > deadline => panic!("service never listened: {error}"),
            Err(_) => thread::sleep(Duration::from_millis(20)),
        }
    };
    (dir, server, client)
}
fn state() -> RunState {
    RunState::new(
        "r1".into(),
        WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "transport".into(),
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
fn a_client_creates_reads_and_shuts_down_the_service() {
    let (dir, server, client) = start("basic");
    client.create("c1", &state()).unwrap();
    // Each call is its own connection: several in a row prove the next instance is listening.
    for _ in 0..3 {
        assert_eq!(client.get("r1").unwrap().unwrap().status, RunStatus::Paused);
    }
    assert!(client.get("missing").unwrap().is_none());
    let endpoint = endpoint(&dir);
    client.shutdown().unwrap();
    server.join().unwrap().unwrap();
    assert!(raw(&endpoint).is_err(), "the endpoint itself is gone");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn early_leavers_oversized_frames_tricklers_and_concurrent_clients_keep_the_service_up() {
    let (dir, server, client) = start("abuse");
    client.create("c1", &state()).unwrap();
    let endpoint = endpoint(&dir);
    // A client that connects and leaves without a word.
    drop(raw(&endpoint).unwrap());
    assert!(client.get("r1").unwrap().is_some());
    // A frame larger than the limit is refused with an error reply, not served.
    let mut big = raw(&endpoint).unwrap();
    big.write_all(&(host::MAX_FRAME as u32 + 1).to_be_bytes())
        .unwrap();
    let reply: host::Response =
        serde_json::from_slice(&read_frame(&mut big, host::MAX_FRAME).unwrap()).unwrap();
    assert!(reply.error.is_some());
    drop(big);
    // A client that trickles one byte at a time loses the server at the request deadline,
    // instead of holding the only server thread for as long as it likes.
    let trickler = {
        let endpoint = endpoint.clone();
        thread::spawn(move || {
            let mut slow = raw(&endpoint).unwrap();
            slow.write_all(&100u32.to_be_bytes()).unwrap();
            for _ in 0..40 {
                thread::sleep(Duration::from_millis(250));
                if slow.write_all(b" ").is_err() {
                    break;
                }
            }
            let mut rest = Vec::new();
            let _ = slow.read_to_end(&mut rest);
        })
    };
    thread::sleep(Duration::from_millis(100));
    let started = Instant::now();
    assert!(client.get("r1").unwrap().is_some());
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "served after {:?}",
        started.elapsed()
    );
    // Clients calling at once all get through.
    let callers: Vec<_> = (0..3)
        .map(|_| {
            let dir = dir.clone();
            thread::spawn(move || Client::connect(&dir).unwrap().get("r1").unwrap().is_some())
        })
        .collect();
    for caller in callers {
        assert!(caller.join().unwrap());
    }
    client.shutdown().unwrap();
    server.join().unwrap().unwrap();
    trickler.join().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}
