//! Recorded JSON-RPC sessions replayed over in-process pipes against a fake catalogue, plus the
//! audit, cap, pool and HTTP checks. No model runs.

use std::{
    fs,
    io::{self, BufRead, BufReader, Read, Write},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use tempfile::TempDir;

use super::*;
use crate::tool::Permission::{self, *};

fn entry(
    name: &str,
    permission: Permission,
    schema: Value,
    run: impl Fn(&Value, &Ctx) -> Result<Output, String> + Send + Sync + 'static,
) -> Entry {
    Entry {
        name: name.into(),
        description: format!("{name} for the tests"),
        schema,
        permission,
        run: Box::new(run),
    }
}

fn text(text: String) -> Result<Output, String> {
    Ok(Output {
        text,
        structured: None,
    })
}

fn fake() -> Vec<Entry> {
    let int =
        |key: &str| json!({"type":"object","properties":{key:{"type":"integer"}},"required":[key]});
    vec![
        entry(
            "read_note",
            Read,
            json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}),
            |args, _| {
                Ok(Output {
                    text: format!("note {}", args["path"].as_str().unwrap()),
                    structured: Some(json!({"path": args["path"]})),
                })
            },
        ),
        entry(
            "write_note",
            WriteLocal,
            json!({"type":"object","properties":{"path":{"type":"string"},"text":{"type":"string"}},"required":["path","text"]}),
            |args, _| {
                let body = args["text"].as_str().unwrap();
                fs::write(args["path"].as_str().unwrap(), body).map_err(|e| e.to_string())?;
                text(format!("wrote {} bytes", body.len()))
            },
        ),
        entry(
            "post_note",
            WriteExternal,
            json!({"type":"object"}),
            |_, _| panic!("never runs"),
        ),
        entry("buy", Spend, json!({"type":"object"}), |_, _| {
            panic!("never runs")
        }),
        entry("slow", Read, json!({"type":"object"}), |_, ctx| {
            let start = Instant::now();
            while !ctx.stopped() && start.elapsed() < Duration::from_secs(5) {
                thread::sleep(Duration::from_millis(5));
            }
            text("slow done".into())
        }),
        entry("count", Read, int("n"), |args, ctx| {
            let n = args["n"].as_u64().unwrap();
            for i in 1..=n {
                ctx.progress(i as f64, Some(n as f64), &format!("{i} of {n}"));
            }
            text(format!("counted {n}"))
        }),
        entry("big", Read, int("lines"), |args, _| {
            let lines = args["lines"].as_u64().unwrap();
            let body: String = (0..lines).map(|i| format!("line {i}\n")).collect();
            Ok(Output {
                structured: Some(json!({ "text": body })),
                text: body,
            })
        }),
        entry("boom", Read, json!({"type":"object"}), |_, _| {
            panic!("boom")
        }),
    ]
}

fn server(settings: Settings) -> (Server, TempDir) {
    let data = tempfile::tempdir().unwrap();
    (Server::new(fake(), settings, data.path()).unwrap(), data)
}

/// Whether `got` is `want`, where a `"*"` in `want` stands for any value.
fn fits(want: &Value, got: &Value) -> bool {
    match (want, got) {
        (Value::String(s), _) if s == "*" => true,
        (Value::Object(w), Value::Object(g)) => {
            w.len() == g.len() && w.iter().all(|(k, v)| g.get(k).is_some_and(|g| fits(v, g)))
        }
        (Value::Array(w), Value::Array(g)) => {
            w.len() == g.len() && w.iter().zip(g).all(|(w, g)| fits(w, g))
        }
        _ => want == got,
    }
}

/// Plays a recorded session in lockstep: each `>` line goes to the server, each `<` line is the
/// next line it must answer; after the last line nothing more may come.
fn replay(script: &str, settings: Settings) -> TempDir {
    let (server, data) = server(settings);
    let (input, mut to_server) = io::pipe().unwrap();
    let (from_server, output) = io::pipe().unwrap();
    let run = thread::spawn(move || server.stdio(BufReader::new(input), output));
    let mut from_server = BufReader::new(from_server);
    for (n, line) in script.lines().enumerate() {
        if let Some(msg) = line.strip_prefix("> ") {
            writeln!(to_server, "{msg}").unwrap();
        } else if let Some(want) = line.strip_prefix("< ") {
            let mut got = String::new();
            from_server.read_line(&mut got).unwrap();
            let parsed: Value = serde_json::from_str(&got).unwrap_or(Value::Null);
            assert!(
                fits(&serde_json::from_str(want).unwrap(), &parsed),
                "line {}:\nwant {want}\n got {got}",
                n + 1
            );
        }
    }
    drop(to_server);
    run.join().unwrap().unwrap();
    let mut rest = String::new();
    from_server.read_to_string(&mut rest).unwrap();
    assert_eq!(rest, "", "the server sent more than the session expects");
    data
}

/// Sends every request, closes the input and returns all the server wrote.
fn session(server: Server, requests: &[Value]) -> Vec<Value> {
    let input: String = requests.iter().map(|r| format!("{r}\n")).collect();
    talk(server, input.into_bytes())
}

/// Sends `input` as it is, closes it and returns all the server wrote.
fn talk(server: Server, input: Vec<u8>) -> Vec<Value> {
    let (from_server, output) = io::pipe().unwrap();
    let run = thread::spawn(move || server.stdio(input.as_slice(), output));
    let lines: Vec<Value> = BufReader::new(from_server)
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    run.join().unwrap().unwrap();
    lines
}

fn init() -> Value {
    json!({"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"claude-code","version":"2.0.0"}}})
}

fn call(id: u64, name: &str, args: Value) -> Value {
    json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":name,"arguments":args}})
}

fn answer(lines: &[Value], id: u64) -> &Value {
    lines.iter().find(|l| l["id"] == id).unwrap()
}

fn audit_lines(data: &Path) -> Vec<Value> {
    let dir = data.join("agent").join("log");
    let mut lines = Vec::new();
    for file in fs::read_dir(dir).unwrap() {
        let text = fs::read_to_string(file.unwrap().path()).unwrap();
        lines.extend(
            text.lines()
                .map(|l| serde_json::from_str::<Value>(l).unwrap()),
        );
    }
    lines
}

fn with_folder(dir: &Path) -> Settings {
    Settings {
        write_folder: Some(dir.into()),
        ..Settings::default()
    }
}

#[test]
fn initialize_ping_and_protocol_errors() {
    replay(
        include_str!("fixtures/initialize.jsonl"),
        Settings::default(),
    );
}

#[test]
fn tools_list_shows_read_entries_only_without_a_write_folder() {
    replay(
        include_str!("fixtures/tools_list.jsonl"),
        Settings::default(),
    );
}

#[test]
fn a_read_call_returns_text_and_structured_content() {
    replay(
        include_str!("fixtures/read_call.jsonl"),
        Settings::default(),
    );
}

#[test]
fn bad_arguments_are_refused_before_the_entry_runs() {
    replay(include_str!("fixtures/bad_args.jsonl"), Settings::default());
}

#[test]
fn a_write_local_call_without_a_write_folder_is_refused() {
    replay(
        include_str!("fixtures/write_no_folder.jsonl"),
        Settings::default(),
    );
}

#[test]
fn a_write_local_call_runs_inside_the_write_folder_only() {
    let folder = tempfile::tempdir().unwrap();
    replay(
        include_str!("fixtures/write_folder.jsonl"),
        with_folder(folder.path()),
    );
    assert_eq!(
        fs::read_to_string(folder.path().join("n.md")).unwrap(),
        "hi"
    );
    assert!(!folder.path().parent().unwrap().join("out.md").exists());
}

#[test]
fn write_external_and_spend_calls_are_refused() {
    replay(
        include_str!("fixtures/write_external.jsonl"),
        Settings::default(),
    );
}

#[test]
fn a_cancelled_call_stops_and_is_not_answered() {
    replay(include_str!("fixtures/cancel.jsonl"), Settings::default());
}

#[test]
fn a_call_with_a_progress_token_streams_progress() {
    replay(include_str!("fixtures/progress.jsonl"), Settings::default());
}

#[test]
fn task_augmented_calls_answer_at_once_and_keep_their_result() {
    replay(include_str!("fixtures/tasks.jsonl"), Settings::default());
}

#[test]
fn a_line_that_is_not_utf8_is_a_parse_error_and_serving_goes_on() {
    let (server, _data) = server(Settings::default());
    let ping = json!({"jsonrpc":"2.0","id":1,"method":"ping"});
    let lines = talk(
        server,
        [b"\xff\n".as_slice(), format!("{ping}\n").as_bytes()].concat(),
    );
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert_eq!(lines[0]["error"]["code"], -32700);
    assert_eq!(lines[1], json!({"jsonrpc":"2.0","id":1,"result":{}}));
}

#[test]
fn write_entries_are_listed_only_with_a_write_folder() {
    let folder = tempfile::tempdir().unwrap();
    let (server, _data) = server(with_folder(folder.path()));
    let lines = session(
        server,
        &[
            init(),
            json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
        ],
    );
    let names: Vec<&str> = (answer(&lines, 1)["result"]["tools"].as_array().unwrap())
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        ["read_note", "write_note", "slow", "count", "big", "boom"]
    );
}

#[test]
fn every_call_logs_an_intent_and_an_outcome_with_the_client() {
    let folder = tempfile::tempdir().unwrap();
    let data = replay(
        include_str!("fixtures/write_folder.jsonl"),
        with_folder(folder.path()),
    );
    let lines = audit_lines(data.path());
    let mcp = json!({"created_by": "mcp", "client": "claude-code"});
    assert!(lines.iter().all(|l| l["detail"] == mcp), "{lines:#?}");
    let kinds: Vec<(&str, Option<&str>)> = (lines.iter())
        .map(|l| (l["kind"].as_str().unwrap(), l["error"].as_str()))
        .collect();
    let denied = "access denied: '../out.md' is outside the allowed folders";
    assert_eq!(
        kinds,
        [
            ("intent", None),
            ("outcome", None),
            // A refused call never ran: one outcome line says why.
            ("outcome", Some(denied)),
            (
                "outcome",
                Some("access denied: '/etc/out.md' is outside the allowed folders")
            ),
        ]
    );
    assert_eq!(lines[0]["tool"], "write_note");
    assert_eq!(lines[0]["permission"], "write_local");
    assert_eq!(lines[1]["result_bytes"], 13);
    assert_eq!(lines[0]["args_hash"], lines[1]["args_hash"]);
}

#[test]
fn results_are_capped_with_the_cut_stated() {
    let (server, _data) = server(Settings {
        max_result_tokens: 64,
        ..Settings::default()
    });
    let lines = session(server, &[init(), call(1, "big", json!({"lines": 1000}))]);
    let result = &answer(&lines, 1)["result"];
    let text = result["content"][0]["text"].as_str().unwrap();
    let (kept, tail) = text.rsplit_once('\n').unwrap();
    assert!(kept.len() <= 256, "{} bytes kept", kept.len());
    assert!(
        tail.starts_with("... ") && tail.contains("more lines"),
        "{tail}"
    );
    // Structured content past the cap is dropped, and the text says so.
    assert!(result.get("structuredContent").is_none());
    assert!(kept.contains("structured content"), "{kept}");
    assert_eq!(Settings::default().max_result_tokens * 4, 32 * 1024);
}

#[test]
fn a_panicking_entry_answers_an_error_and_logs_its_outcome() {
    let (server, data) = server(Settings::default());
    let lines = session(server, &[init(), call(1, "boom", json!({}))]);
    let result = &answer(&lines, 1)["result"];
    assert_eq!(result["isError"], true);
    assert_eq!(result["content"][0]["text"], "boom panicked: boom");
    let log = audit_lines(data.path());
    assert_eq!(log.last().unwrap()["kind"], "outcome");
    assert_eq!(log.last().unwrap()["error"], "boom panicked: boom");
}

#[test]
fn calls_past_the_worker_cap_wait_their_turn() {
    let active = Arc::new(AtomicUsize::new(0));
    let most = Arc::new(AtomicUsize::new(0));
    let (a, m) = (active.clone(), most.clone());
    let gate = entry("gate", Read, json!({"type":"object"}), move |_, _| {
        m.fetch_max(a.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
        thread::sleep(Duration::from_millis(30));
        a.fetch_sub(1, Ordering::SeqCst);
        text("ok".into())
    });
    let data = tempfile::tempdir().unwrap();
    let settings = Settings {
        workers: 2,
        ..Settings::default()
    };
    let server = Server::new(vec![gate], settings, data.path()).unwrap();
    let mut requests = vec![init()];
    requests.extend((1..=6).map(|id| call(id, "gate", json!({}))));
    let lines = session(server, &requests);
    assert_eq!(lines.len(), 7);
    assert_eq!(most.load(Ordering::SeqCst), 2);
}

#[test]
fn a_write_local_entry_must_take_a_path() {
    let data = tempfile::tempdir().unwrap();
    let pathless = entry("w", WriteLocal, json!({"type":"object"}), |_, _| {
        text(String::new())
    });
    let err = Server::new(vec![pathless], Settings::default(), data.path()).err();
    assert_eq!(err.map(|e| e.kind()), Some(io::ErrorKind::InvalidInput));
}

#[cfg(unix)]
mod http {
    use std::net::{TcpListener, TcpStream};

    use super::*;

    fn post(port: u16, headers: &str, body: &str) -> String {
        let mut conn = TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(
            conn,
            "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAccept: application/json, text/event-stream\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n{headers}\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut response = String::new();
        conn.read_to_string(&mut response).unwrap();
        response
    }

    fn start() -> (u16, String) {
        let (server, data) = server(Settings::default());
        let token = token(data.path()).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let _data = data;
            server.http(listener)
        });
        (port, token)
    }

    #[test]
    fn http_refuses_a_bind_beyond_loopback() {
        let (server, _data) = server(Settings::default());
        let wide = TcpListener::bind("0.0.0.0:0").unwrap();
        let err = server.http(wide).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn http_refuses_a_missing_or_wrong_token() {
        let (port, token) = start();
        let body = init().to_string();
        assert!(post(port, "", &body).starts_with("HTTP/1.1 401"));
        let wrong = "Authorization: Bearer nope\r\n";
        assert!(post(port, wrong, &body).starts_with("HTTP/1.1 401"));
        let right = format!("Authorization: Bearer {token}\r\n");
        let ok = post(port, &right, &body);
        assert!(ok.starts_with("HTTP/1.1 200"), "{ok}");
        assert!(ok.contains("Content-Type: text/event-stream"), "{ok}");
        assert!(ok.contains(r#""protocolVersion":"2025-11-25""#), "{ok}");
        let note = json!({"jsonrpc":"2.0","method":"notifications/initialized"}).to_string();
        assert!(post(port, &right, &note).starts_with("HTTP/1.1 202"));
        // A page on another site that learned the port gets nothing, token or not.
        let foreign = format!("{right}Origin: https://evil.example\r\n");
        assert!(post(port, &foreign, &body).starts_with("HTTP/1.1 403"));
        // Only a body of known length is read: an unread chunked one would reset the connection.
        let chunked = format!("{right}Transfer-Encoding: chunked\r\n");
        assert!(post(port, &chunked, "").starts_with("HTTP/1.1 411"));
        let local = format!("{right}Origin: http://localhost:{port}\r\n");
        assert!(post(port, &local, &body).starts_with("HTTP/1.1 200"));
    }

    #[test]
    fn the_token_is_made_once_and_only_its_owner_may_read_it() {
        use std::os::unix::fs::PermissionsExt;
        let data = tempfile::tempdir().unwrap();
        let first = token(data.path()).unwrap();
        assert_eq!(first.len(), 64);
        assert!(first.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(token(data.path()).unwrap(), first);
        let file = data.path().join("agent").join("mcp-token");
        assert_eq!(
            fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600
        );
        fs::set_permissions(&file, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(token(data.path()).is_err());
        // A file cut short before its write would hold no whole token.
        fs::write(&file, "").unwrap();
        fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(token(data.path()).is_err());
        assert_ne!(token(tempfile::tempdir().unwrap().path()).unwrap(), first);
    }
}
