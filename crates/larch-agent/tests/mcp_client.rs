//! The MCP client against this crate's own MCP server (`larch-agent mcp`, tiny model, CPU):
//! connect over stdio, list the tools, call one, and see a refusal come back as the server's error.

use std::collections::BTreeMap;
use std::fs;

use larch_agent::mcp::client::{Client, McpError, Server, Transport};
use larch_agent::tool::{Permission, Tool, ToolCtx};
use larch_model::{
    Arch,
    gguf::{Gguf, Value as Kv, write},
    tiny,
};
use serde_json::json;

/// Renders only the user's text, so every engine step holds a few dozen tokens.
const TEMPLATE: &str = "{% for m in messages %}{% if m.role == 'assistant' %}{{ bos_token }}{% elif m.role == 'user' %}{{ m.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}{% endif %}";

/// A data folder holding `models/tiny.gguf`: the tiny weights, one token per byte, TEMPLATE.
// ponytail: the third copy (cli.rs, mcp_cli.rs); move all three to tests/common together
fn data_dir() -> tempfile::TempDir {
    let bytes = tiny::gguf(Arch::Llama, 3);
    let g = Gguf::parse(&bytes).unwrap();
    let tensors: Vec<_> = (g.tensors.iter())
        .map(|t| {
            let at = (g.data_offset + t.offset) as usize;
            let data = bytes[at..at + t.bytes as usize].to_vec();
            (t.name.clone(), t.dims.clone(), t.ggml_type, data)
        })
        .collect();
    let mut kv: Vec<_> = (g.kv.into_iter())
        .filter(|(k, _)| k != "llama.context_length")
        .collect();
    let bytes = (0..256).map(|b| Kv::Str(format!("<0x{b:02X}>")));
    kv.extend([
        ("llama.context_length".into(), Kv::U32(4096)),
        ("tokenizer.ggml.model".into(), Kv::Str("llama".into())),
        ("tokenizer.ggml.tokens".into(), Kv::Arr(bytes.collect())),
        (
            "tokenizer.ggml.token_type".into(),
            Kv::Arr(vec![Kv::I32(6); 256]),
        ),
        ("tokenizer.ggml.bos_token_id".into(), Kv::U32(97)),
        ("tokenizer.ggml.eos_token_id".into(), Kv::U32(255)),
        ("tokenizer.ggml.add_bos_token".into(), Kv::Bool(false)),
        ("tokenizer.ggml.add_space_prefix".into(), Kv::Bool(false)),
        ("tokenizer.chat_template".into(), Kv::Str(TEMPLATE.into())),
    ]);
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("models")).unwrap();
    fs::write(dir.path().join("models/tiny.gguf"), write(&kv, &tensors)).unwrap();
    dir
}

fn server(dir: &std::path::Path, allowed: &std::path::Path) -> Transport {
    Transport::Stdio {
        command: env!("CARGO_BIN_EXE_larch-agent").into(),
        args: ["mcp", "-m", "tiny", "--cpu", "--data-dir"]
            .map(String::from)
            .into_iter()
            .chain([
                dir.to_str().unwrap().into(),
                "--allow".into(),
                allowed.to_str().unwrap().into(),
            ])
            .collect(),
        env: BTreeMap::new(),
        secret_args: Vec::new(),
    }
}

#[test]
fn the_client_lists_and_calls_the_servers_tools() {
    let dir = data_dir();
    let allowed = dir.path().join("notes");
    std::fs::create_dir_all(&allowed).unwrap();
    std::fs::write(allowed.join("a.md"), "inside the folder").unwrap();
    let log = dir.path().join("server.log");
    let client = Client::connect(&server(dir.path(), &allowed), Some(log)).unwrap();
    assert_eq!(client.protocol, "2025-11-25");

    let tools = client.tools().unwrap();
    let names: Vec<_> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"read_file"), "{names:?}");
    let read = tools.iter().find(|t| t.name == "read_file").unwrap();
    assert!(
        read.schema["properties"]["path"].is_object(),
        "{:?}",
        read.schema
    );

    let text = client.call("read_file", &json!({"path": "a.md"})).unwrap();
    assert!(
        text.as_ref().unwrap().contains("inside the folder"),
        "{text:?}"
    );
    // A path outside --allow: the server's refusal is the tool's error, not a broken connection.
    let outside = client
        .call("read_file", &json!({"path": "/etc/hostname"}))
        .unwrap();
    assert!(outside.unwrap_err().contains("access denied"));
    // An unknown tool is a JSON-RPC error or an error result, never a hang.
    match client.call("no_such_tool", &json!({})) {
        Ok(Err(_)) | Err(McpError::Server { .. }) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn outside_tools_wrap_the_servers_tools_for_a_chat() {
    let dir = data_dir();
    let allowed = dir.path().join("notes");
    std::fs::create_dir_all(&allowed).unwrap();
    std::fs::write(allowed.join("a.md"), "hello").unwrap();
    let server = Server::new("my-notes", server(dir.path(), &allowed), None);
    assert!(server.chat_tools().is_empty(), "no tools before a connect");
    assert!(!server.connect().unwrap().is_empty());
    let tools = server.chat_tools();
    let read = tools
        .iter()
        .find(|t| t.spec().name == "my_notes__read_file")
        .expect("named <server>__<tool>");
    let spec = read.spec();
    assert!(spec.untrusted_results);
    // The server marks read_file read-only but not closed-world, so it asks as a write would.
    assert!(matches!(
        spec.permission,
        Permission::Read | Permission::WriteExternal
    ));
    assert_eq!(spec.schema["type"], "object");
    let ctx = ToolCtx {
        allowed: vec![],
        data_dir: dir.path().into(),
        source: "test".into(),
    };
    assert_eq!(
        read.call(&json!({"path": "a.md"}), &ctx).unwrap().trim(),
        "hello"
    );
    // Turned off: a chat that still holds the tool gets an error, not a call.
    server.close();
    let off = read
        .call(&json!({"path": "a.md"}), &ctx)
        .unwrap_err()
        .to_string();
    assert!(off.contains("turned off"), "{off}");
    // On again: it reconnects.
    server.connect().unwrap();
    assert_eq!(
        read.call(&json!({"path": "a.md"}), &ctx).unwrap().trim(),
        "hello"
    );
}

/// Whether `pid` still runs: gone, or a zombie waiting for its parent, counts as stopped.
#[cfg(target_os = "linux")]
fn running(pid: &str) -> bool {
    fs::read_to_string(format!("/proc/{}/stat", pid.trim())).is_ok_and(|s| {
        s.rsplit(") ")
            .next()
            .is_some_and(|rest| !rest.starts_with('Z'))
    })
}

/// A server that answers `initialize` with a revision the client does not speak, then ignores
/// its closed stdin, is stopped with its children: the failed start leaves no program running.
#[cfg(target_os = "linux")]
#[test]
fn a_server_whose_start_fails_is_stopped_with_its_children() {
    let dir = tempfile::tempdir().unwrap();
    let pids = dir.path().join("pids");
    let script = format!(
        "sleep 300 & echo $! > {p}.child; echo $$ > {p}; read line; \
         echo '{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"protocolVersion\":\"1999-01-01\"}}}}'; \
         trap '' HUP; exec sleep 300",
        p = pids.display()
    );
    let transport = Transport::Stdio {
        command: "sh".into(),
        args: vec!["-c".into(), script],
        env: BTreeMap::new(),
        secret_args: Vec::new(),
    };
    match Client::connect(&transport, None) {
        Err(McpError::Protocol(m)) => assert!(m.contains("1999-01-01"), "{m}"),
        other => panic!("{:?}", other.map(|c| c.server_name)),
    }
    let (server, child) = (
        fs::read_to_string(&pids).unwrap(),
        fs::read_to_string(pids.with_extension("child")).unwrap(),
    );
    assert!(!running(&server), "the server {server} still runs");
    // Its child is killed with its group; init reaps it a moment later.
    let t = std::time::Instant::now();
    while running(&child) && t.elapsed() < std::time::Duration::from_secs(5) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(!running(&child), "its child {child} still runs");
}

/// Windows: a stdio server runs in a Job Object that closes with the app, so an app that ends
/// without its destructors (a crash, a kill from Task Manager) leaves no server running. This test
/// runs itself again as that app: it connects, then exits with the client still held.
#[cfg(windows)]
#[test]
fn a_stdio_server_ends_with_the_app_however_the_app_ends() {
    const NAME: &str = "a_stdio_server_ends_with_the_app_however_the_app_ends";
    if let Ok(pid_file) = std::env::var("LARCH_TEST_JOB_APP") {
        let stub = format!(
            "$PID | Out-File -Encoding ascii '{pid_file}'; [void][Console]::In.ReadLine(); \
             [Console]::Out.WriteLine('{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"protocolVersion\":\"{}\",\"capabilities\":{{}},\"serverInfo\":{{\"name\":\"stub\"}}}}}}'); \
             [Console]::Out.Flush(); Start-Sleep 300",
            larch_agent::mcp::client::PROTOCOL
        );
        let transport = Transport::Stdio {
            command: "powershell".into(),
            args: vec!["-NoProfile".into(), "-Command".into(), stub],
            env: BTreeMap::new(),
            secret_args: Vec::new(),
        };
        let _client = Client::connect(&transport, None).unwrap();
        std::process::exit(0);
    }
    let dir = tempfile::tempdir().unwrap();
    let pid_file = dir.path().join("pid");
    let app = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", NAME, "--nocapture"])
        .env("LARCH_TEST_JOB_APP", &pid_file)
        .status()
        .unwrap();
    assert!(app.success(), "{app}");
    let pid = fs::read_to_string(&pid_file).unwrap().trim().to_owned();
    let alive = || {
        let list = std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .unwrap();
        String::from_utf8_lossy(&list.stdout).contains(&pid)
    };
    let t = std::time::Instant::now();
    while alive() && t.elapsed() < std::time::Duration::from_secs(10) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(!alive(), "the server {pid} outlived the app");
}
