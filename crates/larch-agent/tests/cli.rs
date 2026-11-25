//! `larch-agent chat`, `bench` and `run ask-docs` as a user runs them, on the tiny random-weight
//! model with a byte vocabulary: what it says is noise, so the tests check the lines around it.

use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Output, Stdio},
};

use larch_agent::bench::Report;
use larch_model::{
    Arch,
    gguf::{Gguf, Value, write},
    tiny,
};
use serde_json::json;

/// Renders only the user's text, so the system prompt and the noise the model answers stay out of
/// the prompt and every engine step holds a few dozen tokens.
// ponytail: the engine's watchdog gives a first step 5 s, and a 465-token prefill of the tiny model
// takes 1.4 s on the Linux container and over 5 s on the Windows runner; render the whole
// conversation again once the first step's limit fits a slow CPU (#149)
const TEMPLATE: &str = "{% for m in messages %}{% if m.role == 'assistant' %}{{ bos_token }}{% elif m.role == 'user' %}{{ m.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}{% endif %}";
/// TEMPLATE, refusing any date but the benchmark's (bench/README.md's `date_string`).
const BENCH_TEMPLATE: &str = "{% if date_string != '26 Jul 2024' %}{{ raise_exception('rendered on ' ~ date_string) }}{% endif %}{% for m in messages %}{% if m.role == 'assistant' %}{{ bos_token }}{% elif m.role == 'user' %}{{ m.content }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}{% endif %}";

/// `template`, with a worker's system prompt in front when its question is the long "word word"
/// one: the question is in a worker's system prompt, which `template` leaves out.
fn long_question(template: &str) -> String {
    format!(
        "{{% if 'Question: word word' in messages[0].content %}}{{{{ messages[0].content }}}}\
         {{% endif %}}{template}"
    )
}

/// A data folder holding `models/tiny.gguf`: the tiny weights, one token per byte, `template`, and
/// a context long enough for a 512-token answer.
fn data_dir(template: &str) -> tempfile::TempDir {
    data_dir_of(template, 16384)
}

/// `data_dir` for a model trained for `context` tokens.
fn data_dir_of(template: &str, context: u32) -> tempfile::TempDir {
    let bytes = tiny::gguf(Arch::Llama, 3);
    let g = Gguf::parse(&bytes).unwrap();
    let tensors: Vec<_> = g
        .tensors
        .iter()
        .map(|t| {
            let at = (g.data_offset + t.offset) as usize;
            let data = bytes[at..at + t.bytes as usize].to_vec();
            (t.name.clone(), t.dims.clone(), t.ggml_type, data)
        })
        .collect();
    let mut kv: Vec<_> =
        g.kv.into_iter()
            .filter(|(k, _)| k != "llama.context_length")
            .collect();
    let bytes = (0..256).map(|b| Value::Str(format!("<0x{b:02X}>")));
    kv.extend([
        ("llama.context_length".into(), Value::U32(context)),
        ("tokenizer.ggml.model".into(), Value::Str("llama".into())),
        ("tokenizer.ggml.tokens".into(), Value::Arr(bytes.collect())),
        (
            "tokenizer.ggml.token_type".into(),
            Value::Arr(vec![Value::I32(6); 256]),
        ),
        ("tokenizer.ggml.bos_token_id".into(), Value::U32(97)),
        ("tokenizer.ggml.eos_token_id".into(), Value::U32(255)),
        ("tokenizer.ggml.add_bos_token".into(), Value::Bool(false)),
        ("tokenizer.ggml.add_space_prefix".into(), Value::Bool(false)),
        (
            "tokenizer.chat_template".into(),
            Value::Str(template.into()),
        ),
    ]);
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("models")).unwrap();
    fs::write(dir.path().join("models/tiny.gguf"), write(&kv, &tensors)).unwrap();
    dir
}

/// Runs the binary in `dir`, which is no git checkout.
fn agent(dir: &Path, args: &[&str], stdin: &str) -> Output {
    agent_with(dir, args, stdin, &[], "tiny", true)
}

/// Runs with explicit model and backend choices so CLI tests exercise the real loader.
fn agent_with(
    dir: &Path,
    args: &[&str],
    stdin: &str,
    env: &[(&str, &str)],
    model: &str,
    cpu: bool,
) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_larch-agent"))
        .current_dir(dir)
        .envs(env.iter().copied())
        .args(args)
        .args(["-m", model])
        .args(cpu.then_some("--cpu"))
        .arg("--data-dir")
        .arg(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn model_lookup_uses_file_then_models_filename_then_alias() {
    let dir = data_dir(TEMPLATE);
    let models = dir.path().join("models");
    let direct = dir.path().join("direct.gguf");
    fs::copy(models.join("tiny.gguf"), &direct).unwrap();
    // Invalid later candidates expose a lookup that tries them before the existing file.
    fs::write(models.join("direct.gguf"), "invalid").unwrap();
    fs::write(models.join("tiny.gguf.gguf"), "invalid").unwrap();
    // Directories are not model files, even if they match before the alias.
    fs::create_dir(dir.path().join("tiny")).unwrap();
    fs::create_dir(models.join("tiny")).unwrap();
    for model in [direct.to_str().unwrap(), "direct.gguf", "tiny.gguf", "tiny"] {
        let out = agent_with(dir.path(), &["chat"], "/quit\n", &[], model, true);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{model}: {stderr}");
        assert!(
            stderr.lines().any(|line| line.starts_with("chat ")),
            "{stderr}"
        );
        assert!(!stderr.contains("no usable GPU"), "{stderr}");
    }
}

#[test]
fn model_not_found_lists_every_candidate_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let out = agent_with(dir.path(), &["chat"], "", &[], "missing", true);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains(&format!(
            "model not found; looked in: missing, {}, {}",
            dir.path().join("models").join("missing").display(),
            dir.path().join("models").join("missing.gguf").display()
        )),
        "{stderr}"
    );
}

#[test]
fn model_help_explains_lookup_order() {
    let dir = tempfile::tempdir().unwrap();
    let out = agent(dir.path(), &["chat", "--help"], "");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Model lookup order: an existing file path, a file name in `<data-dir>/models/`, then an alias at `<data-dir>/models/<alias>.gguf`"),
        "{stdout}"
    );
}

#[test]
fn model_without_cpu_loads_and_warns_only_without_gpu() {
    let no_gpu = larch_kernels::device_count().unwrap() == 0;
    let dir = data_dir(TEMPLATE);
    let out = agent_with(dir.path(), &["chat"], "/quit\n", &[], "tiny", false);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let warning = "larch: no usable GPU found; running on the CPU reference (slow)";
    assert_eq!(
        stderr.lines().filter(|line| *line == warning).count(),
        usize::from(no_gpu),
        "{stderr}"
    );
    assert!(
        stderr.lines().any(|line| line.starts_with("chat ")),
        "{stderr}"
    );
}

/// The chat ID `chat` printed, after checking it ran a turn and printed its timings.
fn chatted(out: &Output) -> String {
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stdout.starts_with("> "), "{stdout}");
    assert!(
        stdout
            .lines()
            .any(|l| l.contains(" tok/s · TTFT ") && l.ends_with(" cached")),
        "{stdout}"
    );
    stderr
        .lines()
        .find_map(|l| l.strip_prefix("chat "))
        .unwrap_or_else(|| panic!("{stderr}"))
        .to_owned()
}

#[test]
fn chat_answers_saves_and_reopens() {
    let dir = data_dir(TEMPLATE);
    // The tiny model never says its end of turn, and a chat without tools has no cap on its
    // answer: 64 tokens keep the turn short, and fit beside the chat's tools in the 8,192-token
    // context with a macOS temp folder too, whose path is about 100 tokens longer in the prompt.
    config(dir.path(), "answer_max_tokens = 64\n");
    let id = chatted(&agent(dir.path(), &["chat"], "hello there\n/quit\n"));
    let saved = dir.path().join(format!("chat/{id}.jsonl"));
    assert!(fs::read_to_string(&saved).unwrap().contains("hello there"));
    // The end of input ends a chat as /quit does.
    let again = chatted(&agent(dir.path(), &["chat", "--chat", &id], "and again\n"));
    assert_eq!(again, id);
    let text = fs::read_to_string(&saved).unwrap();
    assert!(
        text.contains("hello there") && text.contains("and again"),
        "{text}"
    );
}

/// A tool-call prompt that expects no tool.
fn line(id: &str, user: &str) -> serde_json::Value {
    json!({"id": id, "tools": [], "user": user, "expect": {"tool": null, "args_required": {}}})
}

/// A prompts file in `dir`: one prompt longer than the context, which the engine refuses, then a
/// short one.
fn prompts(dir: &Path) -> String {
    let (long, short) = (
        line("none-001", &"word ".repeat(20_000)),
        line("none-002", "Hi."),
    );
    let path = dir.join("prompts.jsonl");
    fs::write(&path, format!("{long}\n{short}\n")).unwrap();
    path.to_str().unwrap().to_owned()
}

fn bench(dir: &Path, json: &Path, more: &[&str]) -> Output {
    let prompts = prompts(dir);
    let json = json.to_str().unwrap();
    // The tiny model never says its end of turn, and a prompt without tools has no cap on its
    // answer: 128 tokens keep the row short, and fit beside the chat's tools, which the cap is
    // checked against, in the 8,192-token context. The tiny model reads a byte a token, so every
    // byte of the system prompt and the tool list takes the cap's room: 256 left almost none.
    let args = [
        "bench",
        "--machine",
        "test",
        "--answer-max-tokens",
        "128",
        "--prompts",
        &prompts,
        "--json",
        json,
    ];
    agent(dir, &[&args, more].concat(), "")
}

#[test]
fn bench_refuses_an_answer_cap_the_context_cannot_hold() {
    // The tiny model's context is 16,384 tokens, so a cap as large leaves no room for a prompt.
    let dir = data_dir(BENCH_TEMPLATE);
    let prompts = prompts(dir.path());
    let args = [
        "bench",
        "--machine",
        "test",
        "--answer-max-tokens",
        "16384",
        "--prompts",
        &prompts,
        "--json",
        "report.json",
    ];
    let out = agent(dir.path(), &args, "");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("--answer-max-tokens 16384 does not fit the model's"),
        "{stderr}"
    );
}

#[test]
fn bench_writes_its_report_and_exits_1_when_a_prompt_errored() {
    // The template refuses any date but the benchmark's, so the short prompt runs only with it.
    let dir = data_dir(BENCH_TEMPLATE);
    let json = dir.path().join("report.json");
    let out = bench(dir.path(), &json, &[]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: Report = serde_json::from_str(&fs::read_to_string(&json).unwrap()).unwrap();
    assert_eq!(report.schema, "larch-agent-bench/1");
    assert_eq!(
        (report.model.as_str(), report.machine.as_str()),
        ("tiny", "test")
    );
    assert_eq!((report.summary.prompts, report.summary.errors), (2, 1));
    assert!(report.rows[0].error.is_some());
    assert_eq!(report.rows[1].error, None, "{:?}", report.rows[1]);
    assert_eq!(report.docs, None);
    // No docs, so no docs switches.
    assert_eq!((report.docs_workers, report.docs_window), (None, None));
    // The default budget, which a report from before it lacks.
    assert_eq!(
        (report.thinking_budget, report.answer_max_tokens),
        (Some(100), Some(128))
    );
    // The commit checked out in the source folder, though the bench ran outside it.
    let source = Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    match source {
        Some(head) => assert!(
            report.commit.as_ref().is_some_and(|c| c.starts_with(&head)),
            "{:?} is not {head}",
            report.commit
        ),
        None => assert_eq!(report.commit, None),
    }
}

#[test]
fn bench_reports_the_alias_whatever_path_names_the_model() {
    let dir = data_dir(BENCH_TEMPLATE);
    fs::write(
        dir.path().join("short.jsonl"),
        format!("{}\n", line("none-002", "Hi.")),
    )
    .unwrap();
    let path = dir.path().join("models").join("tiny.gguf");
    let args = [
        "bench",
        "--machine",
        "test",
        "--prompts",
        "short.jsonl",
        "--answer-max-tokens",
        "16",
        "--json",
        "report.json",
    ];
    let out = agent_with(dir.path(), &args, "", &[], path.to_str().unwrap(), true);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let report = fs::read_to_string(dir.path().join("report.json")).expect(&stderr);
    let report: Report = serde_json::from_str(&report).unwrap();
    // A committed report names no folder of the machine, and one key per model.
    assert_eq!(report.model, "tiny", "{stderr}");
}

#[test]
fn the_prefix_cache_is_on_unless_the_flag_or_the_variable_turns_it_off() {
    let dir = data_dir(BENCH_TEMPLATE);
    fs::write(
        dir.path().join("short.jsonl"),
        format!("{}\n", line("none-002", "Hi.")),
    )
    .unwrap();
    // The tiny model never says its end of turn: a short cap keeps the row short.
    let cap = ["--answer-max-tokens", "16"];
    let args = [
        &["bench", "--machine", "test", "--prompts", "short.jsonl"][..],
        &cap,
    ]
    .concat();
    let blocks = |more: &[&str], env: &[(&str, &str)]| {
        let json = ["--json", "report.json"];
        let out = agent_with(
            dir.path(),
            &[&args[..], &json, more].concat(),
            "",
            env,
            "tiny",
            true,
        );
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = fs::read_to_string(dir.path().join("report.json")).unwrap();
        let report: Report = serde_json::from_str(&text).unwrap();
        report.prefix_cache[0].capacity_blocks
    };
    // The default, at most a quarter of the KV pool.
    let on = blocks(&[], &[]);
    assert!(on > 0 && on <= 1024, "{on}");
    assert_eq!(blocks(&["--prefix-cache-blocks", "0"], &[]), 0);
    assert_eq!(blocks(&[], &[("LARCH_PREFIX_CACHE_BLOCKS", "0")]), 0);
    // The flag wins over the variable.
    let flag = &["--prefix-cache-blocks", "8"];
    assert_eq!(blocks(flag, &[("LARCH_PREFIX_CACHE_BLOCKS", "0")]), 8);
}

#[test]
fn bench_refuses_a_report_path_before_the_model_loads() {
    let dir = data_dir(TEMPLATE);
    let out = bench(dir.path(), &dir.path().join("no/such/dir/report.json"), &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(stderr.contains("report.json"), "{stderr}");
    // No prompt ran.
    assert!(!stderr.contains("none-001"), "{stderr}");
    // With no model in the folder, the report path still fails first.
    let empty = tempfile::tempdir().unwrap();
    let out = bench(
        empty.path(),
        &empty.path().join("no/such/dir/report.json"),
        &[],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("report.json") && !stderr.contains("tiny.gguf"),
        "{stderr}"
    );
}

#[test]
fn bench_refuses_a_router_this_build_lacks_by_name() {
    let dir = data_dir(TEMPLATE);
    let json = dir.path().join("report.json");
    let out = bench(dir.path(), &json, &["--router", "score"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(
        stderr.contains("router score") && stderr.contains("v2 PR 13 (agent/score-router)"),
        "{stderr}"
    );
    assert!(!json.exists());
}

#[test]
fn a_quick_report_is_marked_and_kept_out_of_bench_results() {
    let dir = data_dir(BENCH_TEMPLATE);
    let results = dir.path().join("bench/results");
    fs::create_dir_all(&results).unwrap();
    let kept = results.join("agent-test-tiny.json");
    let out = bench(dir.path(), &kept, &["--quick"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(stderr.contains("bench/results"), "{stderr}");
    assert!(!kept.exists() && !stderr.contains("none-001"), "{stderr}");
    let json = dir.path().join("quick.json");
    let out = bench(dir.path(), &json, &["--quick"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The long prompt errors, as in the full run.
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    let report: Report = serde_json::from_str(&fs::read_to_string(&json).unwrap()).unwrap();
    assert!(report.quick);
    assert_eq!(report.summary.prompts, 2);
    // A run on renamed tools is marked so, with the router's margin only when a router gates.
    assert!(!report.rename_tools && report.router_margin.is_none());
    let renamed = dir.path().join("renamed.json");
    let out = bench(
        dir.path(),
        &renamed,
        &[
            "--quick",
            "--rename-tools",
            "--router",
            "logprob",
            "--router-margin",
            "3",
        ],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    let report: Report = serde_json::from_str(&fs::read_to_string(&renamed).unwrap()).unwrap();
    assert!(report.rename_tools && report.quick);
    assert_eq!(report.router_margin, Some(3.0));
}

/// A replay set at `replay/` in `dir`: one web question on a one-line saved page, asked on 20 Oct
/// 2026.
fn replay_set(dir: &Path) {
    let saved = dir.join("replay/web/market");
    fs::create_dir_all(&saved).unwrap();
    let q = json!({"id": "market-open", "kind": "hours", "question": "When does it open?",
                   "today": "2026-10-20", "answer": ["8 am"], "pages": "web/market"});
    fs::write(dir.join("replay/questions.jsonl"), format!("{q}\n")).unwrap();
    let url = "https://riverton.example/";
    let hits = json!({"query": "", "hits": [{"title": "Market", "url": url, "snippet": "Hours."}]});
    fs::write(saved.join("hits.json"), hits.to_string()).unwrap();
    let text = "Opens at 8 am.\n";
    let page = json!({"url": url, "final_url": url, "fetched_at": "2026-10-20T07:00:00Z",
                      "sha256": "aa", "title": "Market", "text": text, "bytes": text.len(),
                      "truncated": false, "published": null});
    fs::write(saved.join("1.json"), page.to_string()).unwrap();
}

#[test]
fn bench_replay_renders_its_rows_day_and_keeps_its_audit_and_its_table() {
    // The template refuses any date but the rows' day, so the readers run only with it.
    let dir = data_dir(&format!(
        "{{% if date_string != '20 Oct 2026' %}}{{{{ raise_exception('rendered on ' ~ \
         date_string) }}}}{{% endif %}}{TEMPLATE}"
    ));
    replay_set(dir.path());
    let results = dir.path().join("bench/results");
    fs::create_dir_all(&results).unwrap();
    // The tiny model never says its end of turn: a short cap keeps the answers short.
    let args = ["bench", "--machine", "test", "--answer-max-tokens", "16"];
    let run = |name: &str| {
        let json = format!("bench/results/{name}");
        let more = ["--replay", "replay", "--json", &json];
        agent(dir.path(), &[&args[..], &more].concat(), "")
    };
    // In bench/results/, a replay report takes an agent-replay- name: agent-<machine>-<alias>.json
    // there is the tool-call report's.
    for name in ["agent-test-tiny.json", "replay-test-tiny.json"] {
        let out = run(name);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success() && stderr.contains("agent-replay-"),
            "{stderr}"
        );
        assert!(!results.join(name).exists());
    }
    let out = run("agent-replay-test-tiny.json");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    let json = results.join("agent-replay-test-tiny.json");
    let report: Report = serde_json::from_str(&fs::read_to_string(&json).unwrap()).unwrap();
    let replay = report.replay.unwrap();
    assert_eq!(replay.rows.len(), 2, "{stderr}");
    // The engine keeps a prefix cache, but the chunks workers run without it, as the app's do.
    assert!(!replay.workers_prefix_cache);
    // The run's audit log beside the report as one file of whole lines, as the throwaway data
    // folder is gone; both names are the agent lane's.
    let mut names: Vec<_> = (fs::read_dir(&results).unwrap())
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    let audit = "agent-replay-test-tiny-audit.jsonl";
    assert_eq!(names, [audit, "agent-replay-test-tiny.json"]);
    let lines = fs::read_to_string(results.join(audit)).unwrap();
    assert!(lines.lines().count() > 0);
    for line in lines.lines() {
        serde_json::from_str::<serde_json::Value>(line).unwrap();
    }
    // The table again, from the saved report alone.
    let table = Command::new(env!("CARGO_BIN_EXE_larch-agent"))
        .arg("replay-table")
        .arg(&json)
        .output()
        .unwrap();
    assert!(table.status.success(), "{table:?}");
    assert_eq!(
        String::from_utf8_lossy(&table.stdout),
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn bench_reads_the_document_questions_first_and_asks_them_last() {
    let dir = data_dir(BENCH_TEMPLATE);
    let json = dir.path().join("report.json");
    // A document set that is not there fails before any prompt runs.
    let out = bench(dir.path(), &json, &["--docs", "docs"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(
        stderr.contains("questions.json") && !stderr.contains("none-001"),
        "{stderr}"
    );
    let docs = dir.path().join("docs");
    fs::create_dir(&docs).unwrap();
    fs::write(docs.join("lease.md"), "Rent: $1,450 a month.\n").unwrap();
    let questions = json!([{"id": "q01", "question": "When does the train leave?", "facts": []}]);
    fs::write(docs.join("questions.json"), questions.to_string()).unwrap();
    // The docs switches need the docs and take what config.toml's keys take.
    for (flags, says) in [
        (&["--docs-workers", "14"][..], "--docs <DOCS>"),
        (
            &["--docs", "docs", "--docs-workers", "0"][..],
            "--docs-workers",
        ),
        (
            &["--docs", "docs", "--docs-window", "4294967296"][..],
            "--docs-window",
        ),
    ] {
        let out = bench(dir.path(), &json, flags);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains(says), "{stderr}");
    }
    let flags = [
        "--docs",
        "docs",
        "--docs-workers",
        "14",
        "--docs-window",
        "128",
    ];
    let out = bench(dir.path(), &json, &flags);
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The long prompt errors, as in the run without documents.
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    // The timed tool-call prompts come first, on a GPU the document questions have not warmed.
    let line = |id: &str| stderr.lines().position(|l| l.starts_with(id));
    let (last_prompt, question) = (line("none-002 "), line("q01 "));
    assert!(
        last_prompt.is_some() && question.is_some() && last_prompt < question,
        "{stderr}"
    );
    // The tiny model's noise is no list of facts, so the answer says the documents do not answer.
    let report: Report = serde_json::from_str(&fs::read_to_string(&json).unwrap()).unwrap();
    // The switches the docs ran with are in the report.
    assert_eq!(
        (report.docs_workers, report.docs_window),
        (Some(14), Some(128))
    );
    let docs = report.docs.unwrap();
    assert_eq!(docs.abstain_rate, Some(1.0), "{stderr}");
    assert!(docs.errors.is_empty(), "{stderr}");
}

#[test]
fn bench_exits_1_when_only_a_document_question_errored() {
    let dir = data_dir(&long_question(BENCH_TEMPLATE));
    fs::write(
        dir.path().join("short.jsonl"),
        format!("{}\n", line("none-002", "Hi.")),
    )
    .unwrap();
    // The question alone is past the context with the reply's tokens, so the engine refuses it.
    let docs = dir.path().join("docs");
    fs::create_dir(&docs).unwrap();
    fs::write(docs.join("lease.md"), "Rent: $1,450 a month.\n").unwrap();
    let questions = json!([{"id": "q01", "question": "word ".repeat(2000), "facts": []}]);
    fs::write(docs.join("questions.json"), questions.to_string()).unwrap();
    let args = [
        "bench",
        "--machine",
        "test",
        "--answer-max-tokens",
        "128",
        "--prompts",
        "short.jsonl",
        "--docs",
        "docs",
        "--json",
        "report.json",
    ];
    let out = agent(dir.path(), &args, "");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    let json = dir.path().join("report.json");
    let report: Report = serde_json::from_str(&fs::read_to_string(&json).unwrap()).unwrap();
    assert_eq!(report.summary.errors, 0, "{stderr}");
    let errors = report.docs.unwrap().errors;
    assert!(errors["q01"].contains("limit"), "{errors:?}");
}

#[test]
fn a_failed_bench_leaves_its_header_report_and_nothing_beside_it() {
    // No model in the folder, so the run fails at the load, after the report's header is written.
    let dir = tempfile::tempdir().unwrap();
    let new = dir.path().join("new.json");
    let out = bench(dir.path(), &new, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("tiny.gguf"), "{stderr}");
    let report: Report = serde_json::from_str(&fs::read_to_string(&new).unwrap()).unwrap();
    assert_eq!(
        (report.model.as_str(), report.machine.as_str()),
        ("tiny", "test")
    );
    assert_eq!((report.summary.prompts, report.rows.len()), (0, 0));
    assert_eq!(report.prompts.sha256.len(), 64);
    // An older report is replaced whole.
    let old = dir.path().join("old.json");
    fs::write(&old, "older\n").unwrap();
    assert!(!bench(dir.path(), &old, &[]).status.success());
    let report: Report = serde_json::from_str(&fs::read_to_string(&old).unwrap()).unwrap();
    assert!(report.rows.is_empty());
    // The checkout's held-out prompts go in by their name in it; a section asked for is there,
    // empty, and one not asked for is null.
    let rows = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/agent/multistep.jsonl"
    );
    let args = [
        "bench",
        "--machine",
        "test",
        "--holdout",
        "--multistep",
        rows,
    ];
    assert!(
        !agent(dir.path(), &[&args[..], &["--json", "h.json"]].concat(), "")
            .status
            .success()
    );
    let report: Report =
        serde_json::from_str(&fs::read_to_string(dir.path().join("h.json")).unwrap()).unwrap();
    assert_eq!(report.prompts.file, "bench/agent/holdout.jsonl");
    let steps = report.multistep.unwrap();
    assert_eq!((steps.prompts.sha256.len(), steps.rows.len()), (64, 0));
    assert_eq!((report.docs, report.memory), (None, None));
    // Nothing else is left beside them.
    let mut names: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    names.sort();
    assert_eq!(names, ["h.json", "new.json", "old.json", "prompts.jsonl"]);
}

/// `run ask-docs` in `dir`, on `folder`, with `docs` the allowed folder.
fn ask_docs(dir: &Path, folder: &str, question: &str) -> Output {
    let args = ["run", "ask-docs", "--allow", "docs", "--folder", folder];
    agent(dir, &[&args[..], &["--question", question]].concat(), "")
}

#[test]
fn ask_docs_takes_its_folders_from_the_current_one() {
    // Only a scan, so no model is asked and the answer says no document was read.
    let dir = data_dir(TEMPLATE);
    fs::create_dir_all(dir.path().join("docs/scans")).unwrap();
    fs::write(dir.path().join("docs/scans/scan.pdf"), "%PDF-1.4").unwrap();
    let out = ask_docs(dir.path(), "docs/scans", "When?");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let shown = stdout
        .strip_prefix(
            "The documents do not answer this question.\n\n_No .pdf, .md or .txt file was read from ",
        )
        .and_then(|s| s.strip_suffix("._\n"))
        .unwrap_or_else(|| panic!("{stdout}"));
    // The child reads its current folder back without links (macOS: /var is /private/var).
    assert_eq!(
        fs::canonicalize(shown).unwrap(),
        fs::canonicalize(dir.path().join("docs/scans")).unwrap()
    );
}

/// Writes `text` as the data folder `dir`'s config.toml.
fn config(dir: &Path, text: &str) {
    fs::create_dir_all(dir.join("agent")).unwrap();
    fs::write(dir.join("agent/config.toml"), text).unwrap();
}

#[test]
fn ask_docs_takes_its_folders_from_config_toml() {
    let dir = data_dir(TEMPLATE);
    fs::create_dir_all(dir.path().join("docs/scans")).unwrap();
    fs::write(dir.path().join("docs/scans/scan.pdf"), "%PDF-1.4").unwrap();
    let docs = serde_json::to_string(&dir.path().join("docs")).unwrap();
    config(
        dir.path(),
        &format!("allowed = [{docs}]\nmax_workers = 2\n"),
    );
    // No --allow: the folder is allowed by config.toml alone.
    let args = [
        "run",
        "ask-docs",
        "--folder",
        "docs/scans",
        "--question",
        "When?",
    ];
    let out = agent(dir.path(), &args, "");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.starts_with("The documents do not answer this question."),
        "{stdout}"
    );
}

#[test]
fn ask_docs_runs_config_tomls_worker_cap_and_logs_replies_only_when_asked() {
    let dir = data_dir(TEMPLATE);
    fs::create_dir_all(dir.path().join("docs")).unwrap();
    for (name, text) in [
        ("lease.md", "The rent is $1,450.\n"),
        ("bank.md", "The rent went out on the 1st.\n"),
        ("notes.md", "Ask about the rent.\n"),
    ] {
        fs::write(dir.path().join("docs").join(name), text).unwrap();
    }
    // Two sessions in all, one kept free for the chat: the workers go one at a time.
    config(dir.path(), "max_workers = 2\n");
    let args = [
        "run",
        "ask-docs",
        "--allow",
        "docs",
        "--folder",
        "docs",
        "--question",
        "What is the rent?",
    ];
    let out = agent_with(
        dir.path(),
        &args,
        "",
        &[("LARCH_AGENT_LOG_REPLIES", "1")],
        "tiny",
        true,
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    // The run prints each audit line to stderr, as it logs it.
    let steps: Vec<serde_json::Value> = stderr
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let detail = |step: &str| {
        let event = steps.iter().find(|e| e["step"] == step);
        event.unwrap_or_else(|| panic!("{stderr}"))["detail"].clone()
    };
    assert_eq!(detail("workers")["workers"], 3, "{stderr}");
    assert_eq!(detail("workers")["at_once"], 1, "{stderr}");
    // Asked for, each worker's reply is in the log beside its hash.
    let worker = detail("worker");
    assert!(
        worker["reply"].is_string() && worker["reply_sha256"].is_string(),
        "{worker}"
    );
}

#[test]
fn ask_docs_fails_on_a_config_toml_key_it_does_not_know() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join("docs")).unwrap();
    config(dir.path(), "max_wrokers = 2\n");
    let out = ask_docs(dir.path(), "docs", "When?");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("'max_wrokers'"), "{stderr}");
}

#[test]
fn ask_docs_with_no_folder_allowed_says_how_to_allow_one() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join("docs")).unwrap();
    let args = ["run", "ask-docs", "--folder", "docs", "--question", "When?"];
    let out = agent(dir.path(), &args, "");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    let config = dir.path().join("agent").join("config.toml");
    let says = format!(
        "no folder is allowed: pass --allow <folder>, or list one under `allowed` in {}",
        config.display()
    );
    assert!(stderr.contains(&says), "{stderr}");
}

#[test]
fn chat_reads_config_toml_before_it_loads_the_model() {
    // No model in the data folder, so config.toml's error shows only if it comes first.
    let dir = tempfile::tempdir().unwrap();
    config(dir.path(), "git_tool = true\n");
    let out = agent(dir.path(), &["chat"], "");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(
        stderr.contains("config.toml:1: unknown key 'git_tool'") && !stderr.contains("tiny.gguf"),
        "{stderr}"
    );
}

#[test]
fn ask_docs_checks_its_folder_before_it_loads_the_model() {
    // No model in the data folder, so the folder's error shows only if it comes first.
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join("docs")).unwrap();
    fs::create_dir_all(dir.path().join("private")).unwrap();
    for (folder, error) in [("private", "access denied"), ("docs/typo", "typo")] {
        let out = ask_docs(dir.path(), folder, "When?");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{stderr}");
        assert!(
            stderr.contains(error) && !stderr.contains("tiny.gguf"),
            "{stderr}"
        );
    }
}

#[test]
fn a_failed_ask_docs_names_the_audit_log() {
    // A question past the context with the reply's tokens fails the workers step, which the
    // audit log keeps.
    let dir = data_dir(&long_question(TEMPLATE));
    fs::create_dir_all(dir.path().join("docs")).unwrap();
    fs::write(dir.path().join("docs/lease.md"), "Rent: $1,450 a month.\n").unwrap();
    let out = ask_docs(dir.path(), "docs", &"word ".repeat(2000));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    let log = dir.path().join("agent").join("log");
    assert!(stderr.contains(&log.display().to_string()), "{stderr}");
    assert!(stderr.contains("limit"), "{stderr}");
    let logged: String = fs::read_dir(&log)
        .unwrap()
        .map(|e| fs::read_to_string(e.unwrap().path()).unwrap())
        .collect();
    assert!(
        logged.contains("\"step\":\"workers\"") && logged.contains("limit"),
        "{logged}"
    );
}

/// `run research` prints its planning line (and the brief and each progress line) as they come,
/// before the report, so the time to the brief shows from the CLI; nothing reaches the web.
#[test]
fn run_research_prints_its_planning_line_before_the_report() {
    let dir = data_dir("{{ messages[-1].content }}");
    let out = agent_with(
        dir.path(),
        &["run", "research", "--question", "the harbor tower"],
        "",
        &[("LARCH_AGENT_TEST_WEB", "[]")],
        "tiny",
        true,
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(
        stdout.starts_with("Planning research on: the harbor tower\n\n"),
        "{stdout}"
    );
    assert!(stdout.contains("# the harbor tower"), "{stdout}");
}

/// A turn's chat shows at most `MAX_OUTSIDE` outside tools: asking a turn bench for more is
/// refused, as its report would name tools the model never saw.
#[test]
fn a_turn_bench_refuses_more_extra_tools_than_a_chat_shows() {
    let dir = data_dir(BENCH_TEMPLATE);
    let over = (larch_agent::MAX_OUTSIDE + 1).to_string();
    let args = [
        "bench",
        "--machine",
        "test",
        "--json",
        "report.json",
        "--turn",
        "--extra-tools",
        &over,
    ];
    let out = agent_with(dir.path(), &args, "", &[], "tiny", true);
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("outside tools"), "{err}");
}

#[test]
fn folio_bench_reads_the_documents_asks_the_questions_and_writes_its_report() {
    let dir = data_dir_of("{% for m in messages %}{{ m.content }}{% endfor %}", 32768);
    let docs: Vec<_> = ["a.md", "b.md"]
        .iter()
        .map(|n| {
            let path = dir.path().join(n);
            fs::write(
                &path,
                "Rent is $1,200 a month. The lease ends 3 March 2027.\n".repeat(20),
            )
            .unwrap();
            path.to_str().unwrap().to_owned()
        })
        .collect();
    // The tiny model never ends its turn: a short cap keeps each answer quick.
    fs::create_dir_all(dir.path().join("agent")).unwrap();
    fs::write(
        dir.path().join("agent/config.toml"),
        "answer_max_tokens = 8\n",
    )
    .unwrap();
    let questions = dir.path().join("q.jsonl");
    let row = json!({"id": "q1", "question": "When does the lease end?",
        "gold": ["The lease ends 3 March 2027."], "expect": ["3 March 2027"]});
    fs::write(&questions, format!("{row}\n")).unwrap();
    let report = dir.path().join("folio.json");
    // One token a byte: the chat's own prompt and tools alone pass the engine's default 8192.
    let args = [
        "folio-bench",
        "--max-context",
        "16384",
        "--context",
        "32768",
        "--json",
        report.to_str().unwrap(),
    ];
    let out = agent(
        dir.path(),
        &[
            &args[..],
            &[
                "--questions",
                questions.to_str().unwrap(),
                "--docs",
                &docs[0],
                &docs[1],
            ],
        ]
        .concat(),
        "",
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let r: serde_json::Value = serde_json::from_slice(&fs::read(&report).unwrap()).unwrap();
    assert_eq!(r["schema"], "larch-agent-folio/2");
    assert_eq!(r["documents"].as_array().unwrap().len(), 2, "{r}");
    assert_eq!(r["questions"].as_array().unwrap().len(), 1, "{r}");
    assert_eq!(r["decode_default"]["tok_s"].as_array().unwrap().len(), 3);
    assert!(r["tokens"].as_u64().unwrap() > 1000, "{r}");
    // --reopen opens the kept chat again on a fresh engine and measures with its documents held.
    let chat = r["chat"].as_str().unwrap();
    let again = dir.path().join("reopen.json");
    let reopen = [
        "folio-bench",
        "--context",
        "32768",
        "--json",
        again.to_str().unwrap(),
        "--reopen",
        chat,
        "--questions",
        questions.to_str().unwrap(),
        "--docs",
        &docs[0],
        &docs[1],
    ];
    let out = agent(dir.path(), &reopen, "");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let r: serde_json::Value = serde_json::from_slice(&fs::read(&again).unwrap()).unwrap();
    assert_eq!(r["schema"], "larch-agent-folio-reopen/1");
    assert_eq!(r["decode_documents"]["tok_s"].as_array().unwrap().len(), 3);
    assert!(r["stamps"].as_array().unwrap().len() >= 2, "{r}");
}
