//! `larch-agent mcp` as an MCP client runs it: the built binary over stdio, on the tiny
//! random-weight model with a byte vocabulary on the CPU reference. Its answers are noise, so the
//! tests check what is around them: the catalogue, the order, the folders, the write rule and the
//! web gate.

use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader, Write},
    net::TcpListener,
    path::Path,
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    thread,
};

use larch_model::{
    Arch,
    gguf::{Gguf, Value as Kv, write},
    tiny,
};
use serde_json::{Value, json};

/// Renders only the user's text, so every engine step holds a few dozen tokens.
const TEMPLATE: &str = "{% for m in messages %}{% if m.role == 'assistant' %}{{ bos_token }}{% elif m.role == 'user' %}{{ m.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}{% endif %}";

/// The read tools every flag set lists, in catalogue order, when config.toml leaves git_tools off.
const READS: [&str; 8] = [
    "iris_fanout",
    "list_dir",
    "read_file",
    "find_files",
    "search_files",
    "recall",
    "iris_ask_docs",
    "iris_read_papers",
];

/// A data folder holding `models/tiny.gguf`: the tiny weights, one token per byte, TEMPLATE.
// ponytail: the same builder as cli.rs's; move both to tests/common once a third file needs it
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

/// The tools `--web` adds, by the tool each comes after in the catalogue.
const WEB: [(&str, &str); 4] = [
    ("recall", "web_search"),
    ("web_search", "fetch_page"),
    ("iris_ask_docs", "iris_answer"),
    ("iris_read_papers", "iris_research"),
];

/// The variable a debug build of the binary reads as its fake web_search's result, which also lets
/// its fetches reach this machine.
const TEST_WEB: &str = "LARCH_AGENT_TEST_WEB";

/// One MCP session: `initialize`, its notification, then `calls` as requests 2, 3, ...; stdin
/// closes after the last, and the server answers every call before it exits. The answers by ID.
fn session(dir: &Path, flags: &[&str], calls: &[(&str, Value)]) -> HashMap<u64, Value> {
    session_env(dir, flags, &[], calls)
}

/// `session` with the binary's environment plus `env`.
fn session_env(
    dir: &Path,
    flags: &[&str],
    env: &[(&str, &str)],
    calls: &[(&str, Value)],
) -> HashMap<u64, Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_larch-agent"))
        .current_dir(dir)
        .args(["mcp", "-m", "tiny", "--cpu", "--data-dir"])
        .arg(dir)
        .args(flags)
        .envs(env.iter().copied())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25", "capabilities": {},
        "clientInfo": {"name": "mcp_cli", "version": "0"}}});
    let mut lines = vec![
        init,
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    ];
    lines.extend((2..).zip(calls).map(|(id, (method, params))| {
        json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
    }));
    let mut stdin = child.stdin.take().unwrap();
    for line in lines {
        writeln!(stdin, "{line}").unwrap();
    }
    drop(stdin);
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let stdout = String::from_utf8(out.stdout).unwrap();
    let answers: HashMap<u64, Value> = (stdout.lines())
        .map(|line| serde_json::from_str::<Value>(line).expect(line))
        .map(|m| (m["id"].as_u64().expect("only answers on stdout"), m))
        .collect();
    assert_eq!(
        answers[&1]["result"]["serverInfo"]["name"], "iris",
        "{stderr}"
    );
    assert_eq!(answers.len(), calls.len() + 1, "{stdout}\n{stderr}");
    answers
}

/// The names `tools/list` gave.
fn names(list: &Value) -> Vec<String> {
    (list["result"]["tools"].as_array().unwrap().iter())
        .map(|t| t["name"].as_str().unwrap().to_owned())
        .collect()
}

fn call(name: &str, args: Value) -> (&'static str, Value) {
    ("tools/call", json!({"name": name, "arguments": args}))
}

fn text(answer: &Value) -> &str {
    answer["result"]["content"][0]["text"].as_str().unwrap()
}

fn fanout(prompts: &[&str]) -> (&'static str, Value) {
    call(
        "iris_fanout",
        json!({"system": "Answer in one line.", "prompts": prompts, "max_tokens": 8}),
    )
}

#[test]
fn without_write_the_catalogue_is_the_reads_and_a_write_is_refused() {
    let dir = data_dir();
    let allowed = dir.path().join("allowed");
    fs::create_dir(&allowed).unwrap();
    fs::write(allowed.join("a.md"), "inside\n").unwrap();
    fs::write(dir.path().join("secret.md"), "outside\n").unwrap();
    let outside = dir.path().join("secret.md");
    let a = session(
        dir.path(),
        &["--allow", allowed.to_str().unwrap()],
        &[
            ("tools/list", json!({})),
            call("read_file", json!({"path": "a.md"})),
            call("read_file", json!({"path": outside.to_str().unwrap()})),
            call("write_file", json!({"path": "b.md", "text": "x"})),
        ],
    );
    assert_eq!(names(&a[&2]), READS);
    assert_eq!(a[&3]["result"]["isError"], false, "{}", a[&3]);
    assert!(text(&a[&3]).contains("inside"), "{}", a[&3]);
    assert_eq!(a[&4]["result"]["isError"], true, "{}", a[&4]);
    assert!(text(&a[&4]).contains("access denied"), "{}", a[&4]);
    assert!(!text(&a[&4]).contains("outside\n"), "{}", a[&4]);
    assert_eq!(a[&5]["result"]["isError"], true, "{}", a[&5]);
    assert!(text(&a[&5]).contains("without a write folder"), "{}", a[&5]);
    assert!(!allowed.join("b.md").exists());
}

#[test]
fn with_write_write_file_is_listed_and_writes_only_in_its_folder() {
    let dir = data_dir();
    // Its own folder: write_file never writes inside the data folder.
    let work = tempfile::tempdir().unwrap();
    let out = work.path().join("out");
    fs::create_dir(&out).unwrap();
    let a = session(
        dir.path(),
        &["--write", out.to_str().unwrap()],
        &[
            ("tools/list", json!({})),
            call("write_file", json!({"path": "b.md", "text": "written"})),
            call("write_file", json!({"path": "../escape.md", "text": "x"})),
        ],
    );
    let mut expected = READS.map(String::from).to_vec();
    expected.insert(5, "write_file".into());
    assert_eq!(names(&a[&2]), expected);
    assert_eq!(a[&3]["result"]["isError"], false, "{}", a[&3]);
    assert_eq!(fs::read_to_string(out.join("b.md")).unwrap(), "written");
    assert_eq!(a[&4]["result"]["isError"], true, "{}", a[&4]);
    assert!(!work.path().join("escape.md").exists());
}

#[test]
fn config_gives_the_mcp_folders_never_the_chat_folders_and_turns_on_git() {
    let dir = data_dir();
    let (chat, mcp) = (dir.path().join("chat"), dir.path().join("mcp"));
    for folder in [&chat, &mcp] {
        fs::create_dir(folder).unwrap();
        fs::write(folder.join("a.md"), "here\n").unwrap();
    }
    fs::create_dir(dir.path().join("agent")).unwrap();
    let config = format!("allowed = [{chat:?}]\nmcp_allow = [{mcp:?}]\ngit_tools = true\n");
    fs::write(dir.path().join("agent/config.toml"), config).unwrap();
    let path = |folder: &Path| json!({"path": folder.join("a.md").to_str().unwrap()});
    let a = session(
        dir.path(),
        &[],
        &[
            call("read_file", path(&mcp)),
            call("read_file", path(&chat)),
            ("tools/list", json!({})),
        ],
    );
    assert_eq!(a[&2]["result"]["isError"], false, "{}", a[&2]);
    assert!(text(&a[&3]).contains("access denied"), "{}", a[&3]);
    // As in a chat, after the file tools.
    let mut expected = READS.map(String::from).to_vec();
    let git = ["changed_files", "diff_file", "show_file", "log"];
    expected.splice(5..5, git.map(String::from));
    assert_eq!(names(&a[&4]), expected);
}

#[test]
fn fanout_answers_each_prompt_in_order_with_its_finish() {
    let dir = data_dir();
    let prompts = ["one", "two two", "three three three"];
    let mut calls = vec![fanout(&prompts)];
    calls.extend(prompts.iter().map(|p| fanout(&[p])));
    let a = session(dir.path(), &[], &calls);
    let batch = &a[&2]["result"];
    assert_eq!(batch["isError"], false, "{batch}");
    let answers = batch["structuredContent"]["answers"].as_array().unwrap();
    assert_eq!(answers.len(), 3, "{batch}");
    // Greedy and seeded: each prompt run alone gives the answer the batch put at its place.
    for (i, answer) in answers.iter().enumerate() {
        let alone = &a[&(3 + i as u64)]["result"]["structuredContent"]["answers"][0];
        assert_eq!(answer, alone, "prompt {i}");
        // The tiny model never ends its turn, so each answer stops at max_tokens.
        assert_eq!(answer["finish"], "length", "{answer}");
    }
    let text = batch["content"][0]["text"].as_str().unwrap();
    assert!(text.find("[1]") < text.find("[2]") && text.find("[2]") < text.find("[3]"));
}

#[test]
fn the_skills_return_their_sources_as_structured_content() {
    let dir = data_dir();
    let papers = dir.path().join("papers");
    fs::create_dir(&papers).unwrap();
    fs::write(papers.join("p.md"), "# A paper\n\nIt measures 42 things.\n").unwrap();
    let a = session(
        dir.path(),
        &["--allow", papers.to_str().unwrap()],
        &[
            call("iris_read_papers", json!({"papers": ["p.md"]})),
            call(
                "iris_ask_docs",
                json!({"folder": ".", "question": "How many things?"}),
            ),
        ],
    );
    let result = &a[&2]["result"];
    assert_eq!(result["isError"], false, "{result}");
    let readers = result["structuredContent"]["readers"].as_array().unwrap();
    assert_eq!(readers.len(), 1, "{result}");
    assert!(
        readers[0]["file"].as_str().unwrap().ends_with("p.md"),
        "{result}"
    );
    assert!(text(&a[&2]).contains("p.md"), "{result}");
    let docs = &a[&3]["result"];
    assert_eq!(docs["isError"], false, "{docs}");
    assert!(docs["structuredContent"].get("facts").is_some(), "{docs}");
}

/// A web server on this machine that serves `body` as Markdown at /page, and 404 elsewhere; the
/// paths asked for but robots.txt, in order.
fn site(body: &'static str) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let asked = Arc::new(Mutex::new(Vec::new()));
    let log = asked.clone();
    thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let path = line.split(' ').nth(1).unwrap_or("/").to_string();
            // The headers, up to the empty line.
            while reader.read_line(&mut line).unwrap() > 2 {
                line.clear();
            }
            if path != "/robots.txt" {
                log.lock().unwrap().push(path.clone());
            }
            let (status, body) = match path.as_str() {
                "/page" => ("200 OK", body),
                _ => ("404 Not Found", "not found"),
            };
            let reply = format!(
                "HTTP/1.1 {status}\r\nContent-Type: text/markdown\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(reply.as_bytes()).unwrap();
        }
    });
    (base, asked)
}

const PAGE: &str = "# The tower\n\nThe tower is 212 metres tall.\n\nIt has 48 floors.\n";

#[test]
fn the_web_tools_are_offered_only_with_web_and_fetch_page_never_opens_this_machine() {
    let dir = data_dir();
    let (base, asked) = site(PAGE);
    let page = format!("{base}/page");
    let without = session(dir.path(), &[], &[("tools/list", json!({}))]);
    assert_eq!(names(&without[&2]), READS);
    let with = session(
        dir.path(),
        &["--web"],
        &[
            ("tools/list", json!({})),
            call("fetch_page", json!({ "url": page })),
        ],
    );
    let mut expected = READS.map(String::from).to_vec();
    for (after, tool) in WEB {
        let at = expected.iter().position(|t| t == after).unwrap();
        expected.insert(at + 1, tool.into());
    }
    assert_eq!(names(&with[&2]), expected);
    // A caller picks a level, never a run that holds the GPU past deep's budget.
    let tools = with[&2]["result"]["tools"].as_array().unwrap();
    let research = tools.iter().find(|t| t["name"] == "iris_research").unwrap();
    let props = &research["inputSchema"]["properties"];
    assert_eq!(props["effort"]["enum"], json!(["standard", "deep"]));
    assert!(
        props.get("rounds").is_none() && props.get("pages").is_none(),
        "{props}"
    );
    // The caller's own URL is not held to the provenance rule, but the private-address block holds.
    assert_eq!(with[&3]["result"]["isError"], true, "{}", with[&3]);
    assert!(text(&with[&3]).contains("this machine"), "{}", with[&3]);
    assert!(asked.lock().unwrap().is_empty());
}

// The fake search and the loopback fetches exist only in a debug build of the binary.
#[cfg_attr(not(debug_assertions), ignore = "needs a debug build's test hooks")]
#[test]
fn iris_answer_cites_only_the_served_page_and_iris_research_says_why_it_stopped() {
    let dir = data_dir();
    let (base, asked) = site(PAGE);
    let hits = json!([{"title": "The tower", "url": format!("{base}/page"), "snippet": "..."}]);
    let question = "How tall is the tower?";
    let a = session_env(
        dir.path(),
        &["--web"],
        &[(TEST_WEB, &hits.to_string())],
        &[
            call("iris_answer", json!({ "question": question })),
            call(
                "iris_research",
                json!({"question": question, "effort": "standard"}),
            ),
        ],
    );
    let answer = &a[&2]["result"];
    assert_eq!(answer["isError"], false, "{answer}");
    let structured = &answer["structuredContent"];
    assert!(structured["answer"].is_string(), "{answer}");
    // The tiny model's noise cites nothing it can verify, so the list may be empty; any source it
    // keeps quotes the page.
    for source in structured["sources"].as_array().unwrap() {
        let quote = source["cited"]["quote"].as_str().unwrap();
        assert!(PAGE.contains(quote), "{source}");
    }
    assert!(asked.lock().unwrap().contains(&"/page".into()), "{answer}");
    let report = &a[&3]["result"];
    assert_eq!(report["isError"], false, "{report}");
    let structured = &report["structuredContent"];
    assert_eq!(structured["type"], "report", "{report}");
    assert!(structured["sections"].is_array(), "{report}");
    assert!(structured["sources"].is_array(), "{report}");
    assert!(structured["unverified"].is_array(), "{report}");
    let stopped = structured["stopped"].as_str().unwrap();
    assert!(!stopped.is_empty(), "{report}");
    assert!(
        text(&a[&3]).contains(&format!("_Stopped: {stopped}._")),
        "{report}"
    );
}
