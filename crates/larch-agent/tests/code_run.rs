//! Code mode's run_tests and bash: the five test-output parsers on recorded output, and both
//! tools through the sandbox. A test that needs the sandbox prints why and passes when this host
//! cannot sandbox (no unshare, no Landlock, or root), unless LARCH_SANDBOX_REQUIRED=1 makes that
//! a failure.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use larch::toolcall::{Family, Mode, ToolCall, grammar};
use larch_agent::{
    CHAT_TOOLS, FREE_TOOLS, GIT_TOOLS,
    code::task::{Budget, CodeTask, TestRun},
    tool::{Permission, Registry, ToolCtx, ToolError},
};
use serde_json::{Value, json};

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_output");
    // In LF, as a Windows checkout may have made it CRLF: the tests add the CRLF they check.
    let text = fs::read_to_string(path.join(name)).unwrap();
    text.replace("\r\n", "\n")
}

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|n| n.to_string()).collect()
}

#[test]
fn cargo_test_output_parses() {
    // Two test binaries, the unit tests and the doc-tests, summed; ignored is neither.
    let pass = TestRun::parse(&fixture("cargo_pass.txt"), Some(0));
    assert_eq!((pass.passed, pass.failed), (3, 0));
    assert!(pass.failing.is_empty());
    let out = fixture("cargo_fail.txt");
    let fail = TestRun::parse(&out, Some(101));
    assert_eq!((fail.passed, fail.failed), (1, 2));
    // The names the second `failures:` lists, without `- should panic`.
    assert_eq!(
        fail.failing,
        names(&["tests::adds_zero", "tests::overflows"])
    );
    assert_eq!(fail.tail, out, "a short output is its own tail");
    // A test's captured output may print a `failures:` block of its own.
    let captured = out.replace(
        "---- tests::adds_zero stdout ----\n",
        "---- tests::adds_zero stdout ----\nfailures:\n    not a test name\n",
    );
    assert_eq!(TestRun::parse(&captured, Some(101)).failing, fail.failing);
}

#[test]
fn pytest_output_parses() {
    let pass = TestRun::parse(&fixture("pytest_pass.txt"), Some(0));
    assert_eq!((pass.passed, pass.failed), (3, 0));
    assert!(pass.failing.is_empty());
    // An error at setup is a test that did not pass.
    let fail = TestRun::parse(&fixture("pytest_fail.txt"), Some(1));
    assert_eq!((fail.passed, fail.failed), (4, 3));
    assert_eq!(
        fail.failing,
        names(&[
            "tests/test_math.py::test_sub",
            "tests/test_math.py::test_half[0]",
            "tests/test_math.py::test_uses_fixture",
        ])
    );
    // Windows line ends leave no `\r` on a name.
    let crlf = fixture("pytest_fail.txt").replace('\n', "\r\n");
    assert_eq!(TestRun::parse(&crlf, Some(1)).failing, fail.failing);
    // A captured log line at ERROR level names no test.
    let logged = fixture("pytest_fail.txt").replace(
        "=========================== short test summary info",
        "------------------------------ Captured log call -------------------------------\n\
         ERROR    root:test_math.py:5 could not connect\n\
         =========================== short test summary info",
    );
    assert_eq!(TestRun::parse(&logged, Some(1)).failing, fail.failing);
    // -q prints the summary without its rules.
    let quiet = TestRun::parse("..F\n1 failed, 2 passed in 0.01s\n", Some(1));
    assert_eq!((quiet.passed, quiet.failed), (2, 1));
}

#[test]
fn jest_output_parses() {
    let pass = TestRun::parse(&fixture("jest_pass.txt"), Some(0));
    assert_eq!((pass.passed, pass.failed), (2, 0));
    assert!(pass.failing.is_empty());
    let out = fixture("jest_fail.txt");
    let fail = TestRun::parse(&out, Some(1));
    assert_eq!((fail.passed, fail.failed), (1, 2));
    assert_eq!(fail.failing, names(&["adds negatives", "sum › adds 2 + 2"]));
    let crlf = out.replace('\n', "\r\n");
    assert_eq!(TestRun::parse(&crlf, Some(1)).failing, fail.failing);
    // A line of counts above the summary is not pytest's.
    let counted = format!("1 passed in 2 steps\n{out}");
    assert_eq!(TestRun::parse(&counted, Some(1)).failed, 2);
    // The summary of all failing tests names them again; each is listed once.
    let summary = format!("{out}\nSummary of all failing tests\n{out}");
    assert_eq!(TestRun::parse(&summary, Some(1)).failing, fail.failing);
}

#[test]
fn unittest_output_parses() {
    // A skip is neither passed nor failed.
    let pass = TestRun::parse(&fixture("unittest_pass.txt"), Some(0));
    assert_eq!((pass.passed, pass.failed), (4, 0));
    // An error is a test that did not pass.
    let out = fixture("unittest_fail.txt");
    let fail = TestRun::parse(&out, Some(1));
    assert_eq!((fail.passed, fail.failed), (2, 2));
    let want = [
        "test_crash (test_math.Math.test_crash)",
        "test_sub (test_math.Math.test_sub)",
    ];
    assert_eq!(fail.failing, names(&want));
    let crlf = out.replace('\n', "\r\n");
    assert_eq!(TestRun::parse(&crlf, Some(1)).failing, fail.failing);
    let one = TestRun::parse("F\nRan 1 test in 0.000s\n\nFAILED (failures=1)\n", Some(1));
    assert_eq!((one.passed, one.failed), (0, 1));
}

#[test]
fn node_test_output_parses() {
    let pass = TestRun::parse(&fixture("node_pass.txt"), Some(0));
    assert_eq!((pass.passed, pass.failed), (5, 0));
    assert!(pass.failing.is_empty());
    // TAP, node's reporter when stdout is not a terminal: a nested failure and the group it
    // fails are both named, as node counts both.
    let tap = TestRun::parse(&fixture("node_fail.txt"), Some(1));
    assert_eq!((tap.passed, tap.failed), (2, 3));
    assert_eq!(tap.failing, names(&["subtracts", "inner fails", "group"]));
    // The spec reporter lists each failure again under "failing tests"; each is named once.
    let spec = fixture("node_spec_fail.txt");
    let got = TestRun::parse(&spec, Some(1));
    assert_eq!((got.passed, got.failed), (2, 3));
    assert_eq!(got.failing, names(&["subtracts", "inner fails"]));
    let crlf = spec.replace('\n', "\r\n");
    assert_eq!(TestRun::parse(&crlf, Some(1)).failing, got.failing);
}

#[test]
fn output_no_parser_reads_counts_as_one_test_and_a_failed_exit_never_passes() {
    let run = |out: &str, exit| {
        let r = TestRun::parse(out, exit);
        (r.passed, r.failed)
    };
    assert_eq!(run("all good\n", Some(0)), (1, 0));
    // Counts with a time that is not pytest's summary.
    assert_eq!(run("build ok\n3 warnings in 2 files\n", Some(0)), (1, 0));
    assert_eq!(
        run("error[E0425]: cannot find value `x`\n", Some(101)),
        (0, 1)
    );
    assert_eq!(run("still going\n", None), (0, 1), "killed: no exit code");
    // A crashed test binary prints no `test result:`; the binary before it passed.
    let crashed = fixture("cargo_pass.txt") + "error: test failed (signal: 6, SIGABRT)\n";
    assert_eq!(run(&crashed, Some(101)), (3, 1));
    // jest with a suite that never ran: zero tests counted.
    let unrun = "  ● Test suite failed to run\n\nTests:       0 total\n";
    assert_eq!(run(unrun, Some(1)), (0, 1));
    // The tail is the output's end, cut at a character boundary.
    let long = "é".repeat(5000) + "the end\n";
    let tail = TestRun::parse(&long, Some(0)).tail;
    assert!(
        tail.len() <= 4000 && tail.ends_with("é\u{e9}the end\n"),
        "{}",
        tail.len()
    );
}

/// One allowed folder `root` inside a temp dir, with nothing else in it.
struct Fixture {
    tmp: tempfile::TempDir,
    ctx: ToolCtx,
}

impl Fixture {
    fn new() -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        fs::create_dir(&root).unwrap();
        let ctx = ToolCtx {
            allowed: vec![root],
            data_dir: tmp.path().join("data"),
            source: "test".into(),
        };
        Fixture { tmp, ctx }
    }

    fn root(&self) -> &Path {
        &self.ctx.allowed[0]
    }

    fn call(&self, registry: &Registry, name: &str, args: Value) -> Result<String, ToolError> {
        registry.get(name).unwrap().call(&args, &self.ctx)
    }
}

/// Whether this host may skip the sandbox's half of a test: it cannot sandbox, and
/// LARCH_SANDBOX_REQUIRED=1 does not say it must.
fn may_skip(why: &str) -> bool {
    assert!(
        env::var_os("LARCH_SANDBOX_REQUIRED").is_none_or(|v| v != "1"),
        "LARCH_SANDBOX_REQUIRED=1 but the sandbox is unavailable: {why}"
    );
    println!("skipped: {why}");
    true
}

/// A tool's result, or None, with the reason printed, when this host cannot sandbox.
fn sandboxed(result: Result<String, ToolError>) -> Option<String> {
    match result {
        Err(ToolError::Invalid(why))
            if (why.starts_with("the sandbox is unavailable")
                || why.starts_with("commands run only in the sandbox"))
                && may_skip(&why) =>
        {
            None
        }
        result => Some(result.expect("the command runs")),
    }
}

#[test]
fn bash_never_runs_outside_the_sandbox() {
    let f = Fixture::new();
    let escaped = f.tmp.path().join("escaped");
    // The sandbox's shell is cmd.exe on Windows.
    let cmd = if cfg!(windows) {
        format!(
            "echo in>ran& echo out>\"{}\"& echo %HOME%",
            escaped.display()
        )
    } else {
        format!(
            "echo in > ran; echo out > '{}'; echo \"$HOME\"",
            escaped.display()
        )
    };
    let got = f.call(&Registry::builtin(), "bash", json!({"command": cmd}));
    // No sandbox on this host: the command was refused before anything ran.
    assert!(
        !f.root().join("ran").exists() || got.is_ok(),
        "ran unconfined: {got:?}"
    );
    assert!(!escaped.exists(), "{got:?}");
    match sandboxed(got) {
        None => {}
        Some(out) => {
            assert!(f.root().join("ran").exists(), "cwd is the folder: {out}");
            assert!(!escaped.exists(), "wrote outside the folder: {out}");
            // HOME is the sandbox's private tmp, not the folder's or ours.
            let home = PathBuf::from(out.lines().last().unwrap());
            assert!(home.is_absolute(), "{out}");
            assert_ne!(Some(home.as_os_str()), env::var_os("HOME").as_deref());
            assert!(!home.starts_with(f.tmp.path()), "{out}");
        }
    }
}

/// A pipe fails with its first failing command, so `pytest | tail` cannot hide failed tests.
#[cfg(unix)]
#[test]
fn a_pipe_fails_with_its_first_failing_command() {
    let f = Fixture::new();
    let got = f.call(
        &Registry::builtin(),
        "bash",
        json!({"command": "false | cat"}),
    );
    let Some(out) = sandboxed(got) else {
        return;
    };
    assert!(out.starts_with("exit code 1"), "{out}");
}

#[test]
fn bash_caps_its_output() {
    let f = Fixture::new();
    // About 2 MB: past the sandbox's 1 MiB, which keeps the end, and far past the result's
    // 2000 tokens, which keep the start and the end of what the sandbox kept.
    let got = f.call(
        &Registry::builtin(),
        "bash",
        // Raw bytes: cmd's echo and Python's text mode both end lines with CRLF there.
        json!({"command": if cfg!(windows) {
            r#"python -c "import sys; sys.stdout.buffer.write(b'\n'.join(b'%d' % i for i in range(1, 300001)) + b'\n')""#
        } else {
            "seq 1 300000"
        }}),
    );
    let Some(out) = sandboxed(got) else {
        return;
    };
    let (status, shown) = out.split_once('\n').unwrap();
    assert_eq!(status, "exit code 0; the output was cut");
    assert!(out.len() <= 2000 * 4 + 100, "{}", out.len());
    assert!(shown.contains("\n[...]\n"), "{out}");
    assert!(
        out.contains("\n299999\n300000\n["),
        "the end is kept: {out}"
    );
    // The start was dropped: the 1 MiB begins inside a number, so the first whole line is the
    // second.
    let first: u32 = shown.lines().nth(1).unwrap().parse().unwrap();
    assert!(first > 100_000, "{first}");
}

#[test]
fn bash_kills_a_command_past_its_timeout() {
    let f = Fixture::new();
    let registry = Registry::builtin();
    let start = Instant::now();
    let got = f.call(
        &registry,
        "bash",
        json!({"command": if cfg!(windows) {
            "echo started& powershell -NoProfile -Command Start-Sleep 30"
        } else {
            "echo started; sleep 30"
        }, "timeout": 1}),
    );
    let Some(out) = sandboxed(got) else {
        return;
    };
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "{:?}",
        start.elapsed()
    );
    assert_eq!(out, "killed after 1 s, its timeout\nstarted");
    // 120 s is the default and the most a call may ask for.
    let call = |timeout| ToolCall {
        name: "bash".into(),
        args: json!({"command": "true", "timeout": timeout}),
    };
    assert!(registry.resolve(&call(120), &["bash"]).is_ok());
    assert!(registry.resolve(&call(121), &["bash"]).is_err());
}

/// A task whose current attempt's worktree is `worktree`.
fn task(worktree: &Path, test_cmd: Option<&str>) -> CodeTask {
    CodeTask {
        request: "fix the adder".into(),
        repo: worktree.into(),
        base_sha: "0".repeat(40),
        branch: Some("iris/fix-the-adder-0".into()),
        worktree: Some(worktree.into()),
        allowed_paths: vec![".".into()],
        test_cmd: test_cmd.map(String::from),
        check_cmds: vec![],
        budget: Budget {
            seconds: 60,
            tool_calls: 10,
            tokens: 1000,
            k: 1,
        },
        tag: None,
        folder: None,
        drafts: vec![],
        base_tests: None,
        verify: Default::default(),
        read_only: false,
    }
}

#[test]
fn run_tests_runs_the_tasks_command_in_its_worktree_and_parses_it() {
    let f = Fixture::new();
    let out = fixture("cargo_fail.txt");
    fs::write(f.root().join("recorded.txt"), &out).unwrap();
    // Only a task makes one: the registry has no run_tests that could never succeed.
    assert!(Registry::builtin().get("run_tests").is_none());
    let replay = if cfg!(windows) {
        "type recorded.txt& exit 101"
    } else {
        "cat recorded.txt; exit 101"
    };
    // A setup with no tests on the base, as `code` records it: every failure is the change's.
    let ready = CodeTask {
        base_tests: Some(TestRun {
            passed: 0,
            failed: 0,
            failing: Vec::new(),
            shared: Vec::new(),
            tail: String::new(),
        }),
        ..task(f.root(), Some(replay))
    };
    let invalid = |task: CodeTask| match task.test() {
        Err(ToolError::Invalid(why)) => why,
        other => panic!("not invalid: {other:?}"),
    };
    let unstarted = CodeTask {
        branch: None,
        worktree: None,
        ..ready.clone()
    };
    assert!(invalid(unstarted).contains("no worktree"));
    let commandless = CodeTask {
        test_cmd: None,
        ..ready.clone()
    };
    assert!(invalid(commandless).contains("no test command"));
    // The worktree, whatever folder the chat names first: recorded.txt is not in this one.
    let elsewhere = ToolCtx {
        allowed: vec![f.tmp.path().into()],
        ..f.ctx.clone()
    };
    let mut registry = Registry::builtin();
    registry.replace(ready.run_tests(None));
    let tool = registry.get("run_tests").unwrap();
    let Some(got) = sandboxed(tool.call(&json!({}), &elsewhere)) else {
        return;
    };
    assert_eq!(
        got,
        format!(
            "exit code 101: 1 passed, 2 failed\nfailing:\ntests::adds_zero\ntests::overflows\n\n{out}"
        )
    );
    // A command the shell does not have runs no test, now or after any change: an error, so the
    // loop ends before the model's first step.
    let missing = task(f.root(), Some("larch-no-such-test-runner"));
    let why = invalid(missing);
    assert!(
        why.contains("names a command this computer does not have"),
        "{why}"
    );
    // The run itself, for the packet.
    let (run, shown) = ready.test().unwrap();
    assert_eq!((run.passed, run.failed), (1, 2));
    assert_eq!(
        run.failing,
        names(&["tests::adds_zero", "tests::overflows"])
    );
    assert_eq!(shown, got);
}

#[test]
fn run_tests_builds_a_crate_with_a_dependency_offline() {
    let f = Fixture::new();
    // strsim is this crate's own dependency, so the registry that built these tests holds it.
    fs::write(
        f.root().join("Cargo.toml"),
        "[package]\nname = \"toy\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n\
         [dependencies]\nstrsim = \"0.11\"\n",
    )
    .unwrap();
    fs::create_dir(f.root().join("src")).unwrap();
    fs::write(
        f.root().join("src/lib.rs"),
        "#[test]\nfn near() {\n    assert_eq!(strsim::levenshtein(\"kitten\", \"sitting\"), 3);\n}\n",
    )
    .unwrap();
    // The toolchain these tests run on: rustup names it to cargo in a variable the sandbox
    // does not pass.
    if let Ok(toolchain) = env::var("RUSTUP_TOOLCHAIN") {
        fs::write(f.root().join("rust-toolchain"), toolchain).unwrap();
    }
    let got = task(f.root(), Some("cargo test")).test();
    let Some(got) = sandboxed(got.map(|(_, shown)| shown)) else {
        return;
    };
    assert!(
        got.starts_with("exit code 0: 1 passed, 0 failed\n"),
        "{got}"
    );
}

#[test]
fn run_tests_and_bash_schemas_pass_the_grammar() {
    let tmp = tempfile::tempdir().unwrap();
    let specs = [
        task(tmp.path(), Some("true")).run_tests(None).spec(),
        Registry::builtin().get("bash").unwrap().spec(),
    ];
    for spec in specs {
        let name = &spec.name;
        assert_eq!(spec.permission, Permission::WriteLocal, "{name}");
        // bash reads any file, inside the allowed folders or not.
        assert_eq!(spec.untrusted_results, name == "bash", "{name}");
        assert!(spec.description.ends_with('.'), "{name}");
        jsonschema::validator_for(&spec.schema).unwrap();
        let tools = [spec.function()];
        for (family, xml) in [
            (Family::Qwen3, false),
            (Family::Qwen3, true),
            (Family::Llama3, false),
            (Family::Mistral, false),
        ] {
            grammar(family, &tools, xml, Mode::Required)
                .unwrap_or_else(|e| panic!("{name} {family:?} xml={xml}: {e}"));
        }
    }
    // bash is offered only in code mode's free mode, never in a chat.
    assert_eq!(FREE_TOOLS, ["bash"]);
    assert!(!CHAT_TOOLS.contains(&"bash") && !GIT_TOOLS.contains(&"bash"));
}

/// In a folder done in place a command can write the folder's own repository, and the user's git
/// would later run what it planted there outside the sandbox: a new hook and a changed config are
/// put back as they were, and the command's result says so; a whole repository made where there
/// was none goes.
#[test]
fn a_command_cannot_leave_a_hook_or_config_in_the_folders_repository() {
    let f = Fixture::new();
    let git = |args: &[&str]| {
        let ran = (Command::new("git").args(args).current_dir(f.root()))
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("HOME", f.tmp.path())
            .output()
            .unwrap();
        assert!(ran.status.success(), "{args:?}");
    };
    git(&["init", "-q"]);
    let config = fs::read(f.root().join(".git/config")).unwrap();
    let hook = f.root().join(".git/hooks/pre-commit");
    let plant = if cfg!(windows) {
        r"echo evil> .git\hooks\pre-commit& echo [core]fsmonitor=evil>> .git\config& echo done"
    } else {
        "echo evil > .git/hooks/pre-commit; echo '[core]fsmonitor=evil' >> .git/config; echo done"
    };
    let got = f.call(&Registry::builtin(), "bash", json!({"command": plant}));
    let Some(got) = sandboxed(got) else {
        return;
    };
    assert!(got.contains("done"), "{got}");
    assert!(
        got.contains("put back as it was") && got.contains("pre-commit"),
        "{got}"
    );
    assert!(!hook.exists());
    assert_eq!(fs::read(f.root().join(".git/config")).unwrap(), config);
    // No repository before: one the command makes goes whole.
    fs::remove_dir_all(f.root().join(".git")).unwrap();
    let made = if cfg!(windows) {
        "git init -q& echo done"
    } else {
        "git init -q; echo done"
    };
    let got = sandboxed(f.call(&Registry::builtin(), "bash", json!({"command": made}))).unwrap();
    assert!(got.contains("put back as it was: .git"), "{got}");
    assert!(!f.root().join(".git").exists());
}

/// A link a command leaves in the folder's repository goes without being followed: putting the
/// config back through it would write outside the sandbox, where it points.
#[cfg(unix)]
#[test]
fn a_link_in_the_folders_repository_goes_without_a_write_through_it() {
    let f = Fixture::new();
    let init = Command::new("git")
        .args(["init", "-q"])
        .current_dir(f.root())
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("HOME", f.tmp.path())
        .status()
        .unwrap();
    assert!(init.success());
    let config = fs::read(f.root().join(".git/config")).unwrap();
    let outside = f.tmp.path().join("outside");
    fs::write(&outside, "kept\n").unwrap();
    let plant = format!(
        "rm .git/config && ln -s '{}' .git/config && echo done",
        outside.display()
    );
    let got = f.call(&Registry::builtin(), "bash", json!({"command": plant}));
    let Some(got) = sandboxed(got) else {
        return;
    };
    assert!(got.contains("put back as it was: .git/config"), "{got}");
    assert_eq!(fs::read_to_string(&outside).unwrap(), "kept\n");
    let put = f.root().join(".git/config");
    assert!(!fs::symlink_metadata(&put).unwrap().is_symlink());
    assert_eq!(fs::read(put).unwrap(), config);
}
