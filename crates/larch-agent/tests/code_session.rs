//! Code mode's session loop on the fake model: one kept session per attempt, its transcript only
//! appended to, so each step reuses the KV the last one parked; a dropped reply is asked for again
//! from the same prompt; a transcript near the context's end is compacted once; a folder state
//! reached twice stops the run; Stop ends a generation at once.
// The in-place, venv and suite tests run a Python toolchain only on unix; their helpers idle elsewhere.
#![cfg_attr(not(unix), allow(dead_code, unused_imports))]

mod common;

use std::{
    env, fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use common::{Fake, qwen3_call as call, qwen3_code as code, qwen3_plan as plan};
use larch_agent::{
    AgentEvent,
    audit::AuditEvent,
    code::{
        allow::Decision,
        r#loop::{BUDGET, Code, Outcome},
        room::Room,
        task::{CodeTask, Overrides, Verdict, Verify, checks},
    },
    tool::ToolError,
    workers::Workers,
};
use serde_json::json;

/// The toy's test command: it exits 1 while add subtracts.
const TEST: &str = if cfg!(windows) {
    "python test_calc.py"
} else {
    "python3 test_calc.py"
};

/// Python's unittest: Windows' Python has no python3.
const UNITTEST: &str = if cfg!(windows) {
    "python -m unittest"
} else {
    "python3 -m unittest"
};

/// The toy repository in `tmp`, committed, with `more` files: add subtracts, and its one test
/// says so.
fn toy(tmp: &Path, more: &[(&str, &str)]) -> std::path::PathBuf {
    let repo = tmp.join("toy");
    fs::create_dir_all(&repo).unwrap();
    fs::write(repo.join("calc.py"), "def add(a, b):\n    return a - b\n").unwrap();
    fs::write(
        repo.join("test_calc.py"),
        "from calc import add\n\nassert add(2, 3) == 5, add(2, 3)\nprint(\"ok\")\n",
    )
    .unwrap();
    for (name, text) in more {
        fs::write(repo.join(name), text).unwrap();
    }
    common::commit(&repo);
    repo
}

/// The task "make add add" on `repo`, with its tests.
fn task(repo: &Path) -> CodeTask {
    let overrides = Overrides {
        test_cmd: Some(TEST.into()),
        allowed_paths: None,
        check_cmds: vec![],
        budget: BUDGET,
    };
    CodeTask::new("make add add", repo, overrides).unwrap()
}

/// `request` in `folder`, done in place without tests.
fn untested(request: &str, folder: &Path, data: &Path) -> CodeTask {
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: vec![],
        budget: BUDGET,
    };
    CodeTask::in_folder(request, folder, data, overrides).unwrap()
}

/// Scripts the edit that replaces `old` with `new` in calc.py.
fn edit(fake: &Fake, old: &str, new: &str) {
    let args = json!({"path": "calc.py", "old_text": old, "new_text": new});
    call(fake, "edit", args, false);
}

/// Scripts a reply cut at the step's cap, the largest call's room (replace_symbol's 2048 tokens),
/// inside an edit's new_text.
fn cut(fake: &Fake) {
    cut_of(fake, "calc.py");
}

/// `cut` for an edit of `path`.
fn cut_of(fake: &Fake, path: &str) {
    let mut cut = vec![fake.token("<tool_call>")];
    let open = format!(
        "\n{{\"name\": \"edit\", \"arguments\": {{\"path\":\"{path}\",\"old_text\":\"a - b\",\"new_text\":\""
    );
    cut.extend(fake.tokenizer.encode(&open, false));
    cut.resize(2048, fake.byte(b'x'));
    fake.say_tokens(&cut);
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

/// Each model step's audit line, in order: its prompt tokens and those the session's KV held.
fn turns(data: &Path) -> Vec<(u32, u32)> {
    let log = fs::read_dir(data.join("agent/log")).unwrap();
    (log.map(|f| fs::read_to_string(f.unwrap().path()).unwrap()))
        .flat_map(|text| {
            text.lines()
                .map(|l| serde_json::from_str::<AuditEvent>(l).unwrap())
                .collect::<Vec<_>>()
        })
        .filter(|e| e.step.as_deref() == Some("attempt 0"))
        .map(|e| (e.prompt_tokens.unwrap(), e.cached_tokens.unwrap()))
        .collect()
}

/// A multi-step fix ends with the tests agreeing; every step after the first reads the last
/// step's prompt from the session's parked KV and prefills only the reply and its result, and
/// the edit's result shows the file's lines as they are now.
#[test]
fn a_multi_step_fix_solves_and_each_step_reuses_the_kv() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path(), &[]);
    call(&fake, "view", json!({"path": "calc.py"}), false);
    edit(&fake, "a - b", "a + b");
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done: add adds now.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo), &data)) else {
        return;
    };
    let p = &outcome.packet;
    let done = "the model said it is done and the tests agree";
    assert_eq!(p.stopped.as_deref(), Some(done), "{}", fake.fed());
    assert!(p.diff.contains("+    return a + b\n"), "{}", p.diff);
    let fed = fake.fed();
    assert_eq!(fed.matches("Request: make add add").count(), 1, "{fed}");
    // A tested folder is checked by its tests, and told so with where commands run.
    assert!(fed.contains(", where commands run; its tests, which run_tests runs, check the work."));
    assert!(!fed.contains("no test command"), "{fed}");
    let now = "replaced 1 match in calc.py at line 2\ncalc.py now, lines 1-2 of 2:\n1: def add(a, \
               b):\n2:     return a + b";
    assert!(fed.contains(now), "{fed}");
    let turns = turns(&data);
    assert_eq!(turns.len(), 4, "{turns:?}");
    for (k, pair) in turns.windows(2).enumerate() {
        let ((last, _), (prompt, cached)) = (pair[0], pair[1]);
        println!(
            "step {}: prompt {prompt}, cached {cached}, after {last}",
            k + 2
        );
        // The template renders a past reply without the empty think block the generation
        // prompt opened, so the reuse ends a few tokens before the last prompt's end.
        assert!(cached + 16 >= last && cached < prompt, "{turns:?}");
    }
    assert_eq!(fake.unsaid(), 0);
    outcome.task.discard().unwrap();
}

/// A reply cut at its cap is dropped, not appended: the same prompt is sent again, held to a
/// call, and read whole from the KV the cut reply's turn parked; no later prompt holds it.
#[test]
fn a_cut_reply_is_dropped_and_asked_for_again_from_the_parked_kv() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path(), &[]);
    cut(&fake);
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let mut notices = Vec::new();
    let mut sink = |e: AgentEvent| {
        if let AgentEvent::Notice(n) = e {
            notices.push(n);
        }
    };
    let got = code().run_with(&fake.larch, &task(&repo), &data, &mut sink, &mut |_| {
        Decision::No
    });
    let Some(mut outcome) = ran(got) else {
        return;
    };
    let done = "the model said it is done and the tests agree";
    assert_eq!(outcome.packet.stopped.as_deref(), Some(done));
    let dropped = "the reply was dropped and asked for again: the reply was cut at the token limit";
    assert!(
        notices.iter().any(|n| n.starts_with(dropped)),
        "{notices:?}"
    );
    // Fed once, as it was said; never again in a prompt.
    let xs = "x".repeat(1500);
    assert_eq!(fake.fed().matches(&xs).count(), 1);
    let turns = turns(&data);
    assert_eq!(turns.len(), 3, "{turns:?}");
    let ((first, _), (again, cached)) = (turns[0], turns[1]);
    assert_eq!(again, first);
    assert_eq!(cached, first - 1, "{turns:?}");
    outcome.task.discard().unwrap();
}

/// `n` lines of notes, which the model views and leaves alone.
fn notes(n: usize) -> String {
    (1..=n)
        .map(|n| format!("line {n} of the notes, which the model reads and leaves alone\n"))
        .collect()
}

/// Views each of `names` in the toy with `files`, then fixes add and says done, on the session
/// loop with a `ctx` context: the outcome, the notices, and the number of tool results before
/// each compaction.
fn viewed(
    ctx: u32,
    files: &[(&str, &str)],
    names: &[&str],
) -> Option<(Fake, Vec<String>, Vec<usize>)> {
    let fake = Fake::qwen3(ctx);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path(), files);
    for name in names {
        call(&fake, "view", json!({"path": name}), false);
    }
    edit(&fake, "a - b", "a + b");
    fake.say("Done: add adds.");
    let data = tmp.path().join("data");
    let (mut notices, mut results, mut at) = (Vec::new(), 0, Vec::new());
    let mut sink = |e: AgentEvent| match e {
        AgentEvent::ToolResult { .. } => results += 1,
        AgentEvent::Notice(n) => {
            if n == "the transcript neared the context's end: compacted" {
                at.push(results);
            }
            notices.push(n);
        }
        _ => {}
    };
    let got = code().run_with(&fake.larch, &task(&repo), &data, &mut sink, &mut |_| {
        Decision::No
    });
    let mut outcome = ran(got)?;
    let done = "the model said it is done and the tests agree";
    assert_eq!(outcome.packet.stopped.as_deref(), Some(done), "{notices:?}");
    outcome.task.discard().unwrap();
    Some((fake, notices, at))
}

/// A transcript that nears the context's end is compacted into a fresh session: the steps
/// folded to a line but the last, and the other files in play shown whole as they are now, last
/// used first; the run goes on to its fix.
#[test]
fn a_transcript_near_the_context_end_is_compacted_and_the_run_goes_on() {
    let notes = notes(100);
    let names = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"];
    let mut files: Vec<(&str, &str)> = names.iter().map(|n| (*n, notes.as_str())).collect();
    files.push(("s.txt", "a short note\n"));
    let viewing = [
        "calc.py", "a.txt", "b.txt", "s.txt", "c.txt", "d.txt", "e.txt",
    ];
    let Some((fake, notices, at)) = viewed(32768, &files, &viewing) else {
        return;
    };
    assert_eq!(at.len(), 1, "{notices:?}");
    let fed = fake.fed();
    assert_eq!(fed.matches("Request: make add add").count(), 2, "{fed}");
    let compacted = &fed[fed.rfind("Request: make add add").unwrap()..];
    let folded = "[2] view {\"path\":\"a.txt\"}: 1: line 1 of the notes, which the model reads \
                  and leaves alone (100 lines)";
    assert!(compacted.contains(folded), "{compacted}");
    // The last step's view is whole once, not again as a file in play.
    let last = viewing[at[0] - 1];
    assert!(
        !compacted.contains(&format!("{last} as it is now")),
        "{compacted}"
    );
    let (newer, calc) = (
        compacted.find("s.txt as it is now:\n1: a short note\n"),
        compacted.find("calc.py as it is now:\n1: def add(a, b):\n"),
    );
    assert!(
        newer.is_some() && calc.is_some() && newer < calc,
        "{compacted}"
    );
}

/// A context whose first prompt takes half of it (a small one, with a full map) keeps the
/// view that set off a compaction, its last lines cut to fit and where to view the rest: folded,
/// the model would view it again, and fold it again at the next compaction.
#[test]
fn a_small_context_keeps_the_last_view_through_a_compaction() {
    let notes = notes(100);
    let names = ["a.txt", "b.txt", "c.txt"];
    let files: Vec<(&str, &str)> = names.iter().map(|n| (*n, notes.as_str())).collect();
    let Some((fake, notices, at)) = viewed(16384, &files, &names) else {
        return;
    };
    assert!(!at.is_empty(), "{notices:?}");
    let fed = fake.fed();
    let compacted = &fed[fed.find("Request: make add add").unwrap() + 1..];
    let compacted = &compacted[compacted.find("Request: make add add").unwrap()..];
    let last = format!(
        "[{}] view {{\"path\":\"{}\"}}\n1: line 1 of the notes",
        at[0],
        names[at[0] - 1]
    );
    assert!(compacted.contains(&last), "{compacted}");
    assert!(
        compacted.contains("... more lines cut for room; view from start "),
        "{compacted}"
    );
}

/// A compaction leaves the steps after it room: the last step whole takes at most half of what
/// the compaction point leaves above the first prompt, so two steps in a row of views the room
/// sizes never both compact. The tiny vocabulary takes a token for about a byte and a half, a
/// third of what a view counts, so its views are a third as long.
#[test]
fn two_steps_in_a_row_never_both_compact() {
    let notes = notes(35);
    let names = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt", "f.txt"];
    let files: Vec<(&str, &str)> = names.iter().map(|n| (*n, notes.as_str())).collect();
    let viewing: Vec<&str> = ["calc.py"].into_iter().chain(names).collect();
    let Some((_, notices, at)) = viewed(16384, &files, &viewing) else {
        return;
    };
    assert!(!at.is_empty(), "{notices:?}");
    assert!(at.windows(2).all(|w| w[1] > w[0] + 1), "{at:?}");
}

/// Without tests, an honest revert (try a change, undo it) goes on; the same change undone a
/// second time, the same return again, stops the run.
#[test]
fn a_second_return_to_a_folder_state_stops_the_run_without_tests() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("game");
    fs::create_dir(&folder).unwrap();
    fs::write(folder.join("script.js"), "let n = 1;\nlet m = 1;\n").unwrap();
    let data = tmp.path().join("data");
    // n's fix, then m tried and undone (step 1's state again), then the same try undone again.
    let edits = [("n = 1", "n = 2"), ("m = 1", "m = 5"), ("m = 5", "m = 1")];
    let again = [("m = 1", "m = 5"), ("m = 5", "m = 1"), ("m = 1", "m = 4")];
    for (old, new) in edits.into_iter().chain(again) {
        let args = json!({"path": "script.js", "old_text": old, "new_text": new});
        call(&fake, "edit", args, false);
    }
    let task = untested("make n right", &folder, &data);
    let outcome = (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let why = "the change put the folder back as step 1 left it, a second time; stopped";
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(why),
        "{}",
        fake.fed()
    );
    assert_eq!(outcome.steps, 5);
    assert!(fake.unsaid() > 0);
}

/// Two fixes tried and undone in turn, each tested and back to the start, are exploration, not a
/// flip-flop: the run goes on to the fix that works.
#[test]
fn two_honest_reverts_of_different_fixes_do_not_stop_the_run() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path(), &[]);
    call(&fake, "run_tests", json!({}), false);
    for wrong in ["a * b", "b - a"] {
        edit(&fake, "a - b", wrong);
        call(&fake, "run_tests", json!({}), false);
        edit(&fake, wrong, "a - b");
    }
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo), &data)) else {
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

/// A reply dropped again while held to a call is told why, with the steer to write_file a cut
/// edit earns, so the next ask is not the same greedy prompt; a third drop in a row stops.
#[test]
fn replies_dropped_three_times_stop_and_the_held_drop_is_told_why() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path(), &[]);
    for _ in 0..3 {
        cut(&fake);
    }
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo), &data)) else {
        return;
    };
    let why = "3 replies in a row were dropped: the reply was cut at the token limit; make one \
               tool call, or say you are done in one line";
    assert_eq!(outcome.packet.stopped.as_deref(), Some(why));
    let told = "(no call: cut)<|im_end|>\n<|im_start|>user\nthe reply was cut at the token \
                limit; make one tool call, or say you are done in one line\n2 edits of \
                'calc.py' failed: write it whole with write_file";
    let fed = fake.fed();
    assert_eq!(fed.matches(told).count(), 1, "{fed}");
    assert_eq!(fake.unsaid(), 0);
    outcome.task.discard().unwrap();
}

/// A template that does not render a transcript's history as the start of the next prompt (the
/// system message after the turns) runs the same loop to its fix: each step reuses only the
/// prefix that still matches, and the engine reads the rest again.
#[test]
fn a_template_whose_history_moves_runs_the_same_loop_without_reuse() {
    let last = "{% for m in messages %}{% if m.role != 'system' %}<|im_start|>{{ m.role }}\n{{ \
                m.content }}<|im_end|>\n{% endif %}{% endfor %}{% for m in messages %}{% if \
                m.role == 'system' %}<|im_start|>system\n{{ m.content }}<|im_end|>\n{% endif \
                %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    let fake = Fake::qwen3_with(32768, last);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path(), &[]);
    call(&fake, "view", json!({"path": "calc.py"}), false);
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let data = tmp.path().join("data");
    let mut notices = Vec::new();
    let mut sink = |e: AgentEvent| {
        if let AgentEvent::Notice(n) = e {
            notices.push(n);
        }
    };
    let got = code().run_with(&fake.larch, &task(&repo), &data, &mut sink, &mut |_| {
        Decision::No
    });
    let Some(mut outcome) = ran(got) else {
        return;
    };
    let done = "the model said it is done and the tests agree";
    assert_eq!(outcome.packet.stopped.as_deref(), Some(done), "{notices:?}");
    // The transcript, not a prompt rebuilt each step: the request is read once per prompt.
    let fed = fake.fed();
    assert!(!fed.contains("[1] view"), "{fed}");
    let turns = turns(&data);
    assert_eq!(turns.len(), 3, "{turns:?}");
    // The system message moved to the end: what follows the request is read again.
    for pair in turns.windows(2) {
        let ((last, _), (prompt, cached)) = (pair[0], pair[1]);
        assert!(cached + 16 < last && last < prompt, "{turns:?}");
    }
    assert_eq!(fake.unsaid(), 0);
    outcome.task.discard().unwrap();
}

/// A first prompt longer than the context is one clear failure: nothing to compact.
#[test]
fn a_first_prompt_too_long_is_one_notice_not_a_compaction() {
    let fake = Fake::qwen3(2048);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("game");
    fs::create_dir(&folder).unwrap();
    fs::write(folder.join("script.js"), "let n = 1;\n").unwrap();
    let data = tmp.path().join("data");
    let mut notices = Vec::new();
    let mut sink = |e: AgentEvent| {
        if let AgentEvent::Notice(n) = e {
            notices.push(n);
        }
    };
    let task = untested("make n right", &folder, &data);
    let outcome = (code().run_with(&fake.larch, &task, &data, &mut sink, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let stopped = outcome.packet.stopped.unwrap();
    assert!(
        stopped.starts_with("the engine failed: prompt is "),
        "{stopped}"
    );
    assert!(
        !notices.iter().any(|n| n.contains("compacted")),
        "{notices:?}"
    );
}

/// Stop pressed while a step's reply crawls, one forward pass taking seconds, ends the run at
/// once, and the packet says the user stopped it.
#[test]
fn a_stop_during_a_step_ends_the_run_at_once() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("game");
    fs::create_dir(&folder).unwrap();
    fs::write(folder.join("script.js"), "let n = 1;\n").unwrap();
    let data = tmp.path().join("data");
    call(&fake, "view", json!({"path": "script.js"}), false);
    call(&fake, "view", json!({"path": "script.js"}), false);
    let task = untested("make n right", &folder, &data);
    let (stop, armed) = (Arc::new(AtomicBool::new(false)), AtomicBool::new(false));
    let pressed = std::sync::Mutex::new(None);
    // The next step's first pass stalls once the view's result is in.
    let mut sink = |e: AgentEvent| {
        if matches!(e, AgentEvent::ToolResult { .. }) {
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
}

/// A replace_symbol writes as an edit does: its result shows the lines around it as they are now.
#[test]
fn a_replace_symbol_shows_the_lines_it_changed() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("calc");
    fs::create_dir(&folder).unwrap();
    let mut calc: String = (1..=25).map(|n| format!("x_{n} = {n}\n")).collect();
    calc += "def add(a, b):\n    return a - b\n";
    fs::write(folder.join("calc.py"), calc).unwrap();
    let data = tmp.path().join("data");
    let new_text = "def add(a, b):\n    return a + b\n";
    let args = json!({"path": "calc.py", "name": "add", "new_text": new_text});
    call(&fake, "replace_symbol", args, false);
    fake.say("Done.");
    let task = untested("make add add", &folder, &data);
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap();
    let fed = fake.fed();
    let now = "calc.py now, lines 17-27 of 27:\n17: x_17 = 17\n";
    assert!(fed.contains(now), "{fed}");
    assert!(fed.contains("\n27:     return a + b"), "{fed}");
}

/// A call's note goes to the user as progress, with each result's status line, and never into
/// the transcript: no later prompt holds it.
#[test]
fn a_note_is_progress_and_stays_out_of_the_transcript() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path(), &[]);
    let noted = json!({"path": "calc.py", "note": "see what add does"});
    call(&fake, "view", noted, false);
    edit(&fake, "a - b", "a + b");
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done.");
    let mut progress = Vec::new();
    let mut sink = |e: AgentEvent| {
        if let AgentEvent::Progress(p) = e {
            progress.push(p);
        }
    };
    let data = tmp.path().join("data");
    let got = code().run_with(&fake.larch, &task(&repo), &data, &mut sink, &mut |_| {
        Decision::No
    });
    let Some(mut outcome) = ran(got) else {
        return;
    };
    let fed = fake.fed();
    // Fed once as the model wrote it, then its transcript holds the call without it.
    assert_eq!(fed.matches("see what add does").count(), 1, "{fed}");
    assert!(fed.contains(r#"{"path":"calc.py"}}"#), "{fed}");
    assert_eq!(progress[..2], ["see what add does", "read calc.py"]);
    assert!(
        progress.contains(&"tests: 1 of 1 pass".into()),
        "{progress:?}"
    );
    outcome.task.discard().unwrap();
}

/// Runs `request` in place in `folder` on `code`, saying yes to each ask; a notice of writers
/// at once holds the first one's first pass until the others are in, so none runs ahead alone.
/// The outcome and every progress line and notice, in order.
fn run_in(
    fake: &Fake,
    code: Code,
    request: &str,
    folder: &Path,
    data: &Path,
) -> (Outcome, Vec<String>) {
    let task = untested(request, folder, data);
    let mut told = Vec::new();
    let mut sink = |e: AgentEvent| match e {
        AgentEvent::Notice(n) | AgentEvent::Progress(n) => {
            if n.contains(" at once") {
                fake.stall(Duration::from_secs(1));
            }
            told.push(n);
        }
        _ => {}
    };
    let outcome =
        (code.run_with(&fake.larch, &task, data, &mut sink, &mut |_| Decision::Once)).unwrap();
    (outcome, told)
}

/// The brief line that carries a plan's contract to each of its writers.
const CONTRACT: &str =
    "The contract every file of the plan follows exactly:\nnotes start with their letter";

/// A new folder's first step is held to a plan; its files are written side by side, as many at
/// once as the room's writers, each writer's brief carrying the contract, after one progress
/// line that names the plan.
#[test]
fn a_new_folder_fans_out_by_plan_with_the_contract_in_every_brief() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    let data = tmp.path().join("data");
    plan(
        &fake,
        r#"{"contract": "notes start with their letter", "files": [{"path": "a.txt", "about": "the alpha note"}, {"path": "b.txt", "about": "the beta note"}, {"path": "c.txt", "about": "the gamma note"}]}"#,
    );
    // One script for the three writers: side by side each takes about a third of the x's; one
    // after another, the first would take them all and the others none.
    const X: usize = 12;
    let mut ids = vec![fake.byte(b'x'); 3 * X];
    ids.extend([fake.tokenizer.eos().unwrap(); 3]);
    fake.say_tokens(&ids);
    fake.say("Done.");
    fake.say("Done.");
    let wide = Code {
        workers: Workers {
            max_workers: 4,
            prefix_cache: false,
        },
        ..code()
    };
    let (_, told) = run_in(&fake, wide, "write three notes", &folder, &data);
    let at = |line: &str| told.iter().position(|t| t == line);
    let planned = at("Plan: 3 files: a.txt, b.txt, c.txt");
    let at_once = at("writing a.txt, b.txt, c.txt at once");
    assert!(planned.is_some() && planned < at_once, "{told:?}");
    for name in ["a.txt", "b.txt", "c.txt"] {
        let text = fs::read_to_string(folder.join(name)).unwrap();
        let x = text.trim_end();
        assert!(!x.is_empty() && x.len() < 3 * X, "{name}: {text:?}");
    }
    let fed = fake.fed();
    assert_eq!(fed.matches(CONTRACT).count(), 3, "{fed}");
}

/// A plan over files that are there has each one's editor rewrite it from its text and the
/// change; when the result fails a cross-file check, the session is told what does not fit and
/// makes the rest of the change itself.
#[test]
fn editors_that_break_a_cross_file_check_fall_back_to_the_session() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("game");
    fs::create_dir(&folder).unwrap();
    let page = "<!doctype html>\n<html><body>\n<div id=\"board\"></div>\n<script src=\"game.js\"></script>\n</body></html>\n";
    fs::write(folder.join("index.html"), page).unwrap();
    fs::write(
        folder.join("game.js"),
        "document.getElementById('board').textContent = 'go';\n",
    )
    .unwrap();
    let data = tmp.path().join("data");
    plan(
        &fake,
        r#"{"contract": "the board's id is grid", "files": [{"path": "index.html", "about": "rename the id board to grid"}, {"path": "game.js", "about": "look up grid"}]}"#,
    );
    fake.say(&page.replace("board", "grid"));
    // The script's editor misses the rename: the script looks up an id no element has.
    fake.say("document.getElementById('board').textContent = 'start';\n");
    let args = json!({"path": "game.js", "old_text": "'board'", "new_text": "'grid'"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    let (_, told) = run_in(&fake, code(), "rename the board to grid", &folder, &data);
    assert!(
        told.iter()
            .any(|t| t == "Plan: 2 files: index.html, game.js"),
        "{told:?}"
    );
    let fed = fake.fed();
    assert!(
        fed.contains("they do not fit together:\ngame.js line 1"),
        "{fed}"
    );
    assert!(
        fed.contains("Make the rest of this change yourself"),
        "{fed}"
    );
    // Told once, after the last wave: not again per wave, judged before the later files.
    assert_eq!(fed.matches("looks up #board").count(), 1, "{fed}");
    let contract = "follows exactly:\nthe board's id is grid";
    assert_eq!(fed.matches(contract).count(), 2, "{fed}");
    let js = fs::read_to_string(folder.join("game.js")).unwrap();
    assert_eq!(
        js,
        "document.getElementById('grid').textContent = 'start';\n"
    );
    assert!(
        fs::read_to_string(folder.join("index.html"))
            .unwrap()
            .contains("id=\"grid\"")
    );
}

/// An empty folder's first reply is held to a plan: a write_file the model would make instead
/// is not taken, and the reply is dropped and asked for again, free.
#[test]
fn an_empty_folders_first_reply_is_held_to_a_plan() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("hello");
    fs::create_dir(&folder).unwrap();
    let data = tmp.path().join("data");
    let args = json!({"path": "hello.py", "about": "prints hello"});
    call(&fake, "write_file", args, false);
    fake.say("print('hello')\n");
    fake.say("Done.");
    fake.say("Done.");
    let (_, told) = run_in(&fake, code(), "write hello.py", &folder, &data);
    // The grammar takes none of the call's tokens: the reply runs to its cap and is dropped.
    let dropped = "the reply was dropped and asked for again: the reply was cut";
    assert!(told[2].starts_with(dropped), "{told:?}");
    assert!(!folder.join("hello.py").exists(), "{told:?}");
}

/// In the XML call body, a plan's files come as the schema's list once checked, so the plan
/// fans out as in the JSON body.
#[test]
fn an_xml_body_plan_fans_out() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    let data = tmp.path().join("data");
    let body = "\n<function=plan>\n<parameter=contract>\nc\n</parameter>\n<parameter=files>\n[{\"path\": \"a.txt\", \"about\": \"alpha\"}, {\"path\": \"b.txt\", \"about\": \"beta\"}]\n</parameter>\n</function>\n";
    fake.say_tokens(&[fake.token("<tool_call>")]);
    let mut ids = fake.tokenizer.encode(body, false);
    ids.extend([fake.token("</tool_call>"), fake.tokenizer.eos().unwrap()]);
    fake.say_tokens(&ids);
    fake.say("alpha");
    fake.say("beta");
    fake.say("Done.");
    fake.say("Done.");
    let xml = Code {
        xml_body: true,
        ..code()
    };
    let (_, told) = run_in(&fake, xml, "write notes", &folder, &data);
    assert!(
        told.iter().any(|t| t == "Plan: 2 files: a.txt, b.txt"),
        "{told:?}"
    );
    assert_eq!(fs::read_to_string(folder.join("a.txt")).unwrap(), "alpha\n");
    assert_eq!(fs::read_to_string(folder.join("b.txt")).unwrap(), "beta\n");
}

/// A planned file the user declines is named to leave, not to make, and a second plan never
/// fans out: the plan is offered once, and the grammar no longer takes it.
#[test]
fn a_declined_file_is_left_and_the_plan_is_offered_once() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    let data = tmp.path().join("data");
    let args = r#"{"contract": "", "files": [{"path": "a.txt", "about": "alpha"}, {"path": "b.txt", "about": "beta"}]}"#;
    plan(&fake, args);
    fake.say("alpha");
    fake.say("beta");
    plan(&fake, args);
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("write notes", &folder, &data);
    let mut told = Vec::new();
    let mut sink = |e: AgentEvent| {
        if let AgentEvent::Notice(n) | AgentEvent::Progress(n) = e {
            told.push(n);
        }
    };
    let ask = &mut |a: &larch_agent::code::allow::Ask| match a.path.as_deref() {
        Some("b.txt") => Decision::No,
        _ => Decision::Once,
    };
    (code().run_with(&fake.larch, &task, &data, &mut sink, ask)).unwrap();
    let fed = fake.fed();
    assert!(fed.contains("the user declined b.txt: leave them"), "{fed}");
    assert!(!fed.contains("yourself"), "{fed}");
    assert!(!fed.contains("follows exactly"), "{fed}");
    assert_eq!(
        told.iter().filter(|t| t.starts_with("Plan:")).count(),
        1,
        "{told:?}"
    );
    assert!(!folder.join("b.txt").exists());
}

/// One writer of a plan's wave that the engine fails leaves its file to the session, not the
/// run: the other file is written, and the session is told the one left and goes on.
#[test]
fn a_writer_the_engine_fails_leaves_its_file_to_the_session() {
    let fake = Fake::qwen3(12_288);
    let tmp = tempfile::tempdir().unwrap();
    // A line the tiny vocabulary takes a token a byte: b.txt fits the brief's bytes, and its
    // writer's prompt passes the context.
    let long: String = "qzjxkv".repeat(2016) + "\n";
    let (folder, data) = folder(tmp.path(), &[("b.txt", long.as_str())]);
    let args = r#"{"contract": "", "files": [{"path": "a.txt", "about": "alpha"}, {"path": "b.txt", "about": "beta"}]}"#;
    plan(&fake, args);
    fake.say("alpha");
    fake.say("Done.");
    fake.say("Done.");
    let (outcome, told) = run_in(&fake, code(), "write notes", &folder, &data);
    let stopped = outcome.packet.stopped.unwrap_or_default();
    assert!(!stopped.contains("engine failed"), "{stopped} {told:?}");
    assert_eq!(fs::read_to_string(folder.join("a.txt")).unwrap(), "alpha\n");
    let fed = fake.fed();
    assert!(
        fed.contains("the writer of 'b.txt' failed: prompt is "),
        "{told:?}"
    );
    assert!(fed.contains("not changed: b.txt"), "{told:?}");
}

/// A path named twice, a path outside the folder and a file there too long to rewrite whole
/// stay out of the wave: the other files are still written side by side, and the session is
/// told the ones left.
#[test]
fn a_bad_entry_in_a_plan_leaves_the_rest_of_its_wave_side_by_side() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    let long: String = (0..1000).map(|i| format!("{i}\n")).collect();
    fs::write(folder.join("long.txt"), &long).unwrap();
    let data = tmp.path().join("data");
    plan(
        &fake,
        r#"{"contract": "", "files": [{"path": "a.txt", "about": "alpha"}, {"path": "a.txt", "about": "again"}, {"path": "../escape.txt", "about": "out"}, {"path": "long.txt", "about": "add a line"}, {"path": "b.txt", "about": "beta"}]}"#,
    );
    const X: usize = 12;
    let mut ids = vec![fake.byte(b'x'); 2 * X];
    ids.extend([fake.tokenizer.eos().unwrap(); 2]);
    fake.say_tokens(&ids);
    fake.say("Done.");
    fake.say("Done.");
    let wide = Code {
        workers: Workers {
            max_workers: 4,
            prefix_cache: false,
        },
        ..code()
    };
    let (_, told) = run_in(&fake, wide, "write notes", &folder, &data);
    assert!(
        told.iter().any(|t| t == "writing a.txt, b.txt at once"),
        "{told:?}"
    );
    assert!(folder.join("a.txt").exists() && folder.join("b.txt").exists());
    assert_eq!(fs::read_to_string(folder.join("long.txt")).unwrap(), long);
    assert!(!tmp.path().join("escape.txt").exists());
    let fed = fake.fed();
    assert!(
        fed.contains("not changed: ../escape.txt, long.txt"),
        "{fed}"
    );
}

/// A plan's contract counts in each brief's bytes: a file that fits the writer's brief alone but
/// not beside the contract goes to the session, never to an editor that would not see it whole.
#[test]
fn a_file_that_fits_no_brief_beside_the_contract_goes_to_the_session() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = tmp.path().join("notes");
    fs::create_dir(&folder).unwrap();
    // A page just inside the room's brief and its rewrite lines, which a 600-byte contract fills.
    let room = Room::new(32768, 0, code().workers, code().thinking);
    let width = (room.brief_bytes - 300) / 200;
    assert!(200 <= room.rewrite_max_lines, "{room:?}");
    let page: String = (0..200)
        .map(|i| format!("{i:>w$}\n", w = width - 1))
        .collect();
    fs::write(folder.join("page.txt"), &page).unwrap();
    let data = tmp.path().join("data");
    let contract = "c".repeat(600);
    let args = format!(
        r#"{{"contract": "{contract}", "files": [{{"path": "page.txt", "about": "add a line"}}]}}"#
    );
    plan(&fake, &args);
    fake.say("Done.");
    fake.say("Done.");
    let (_, told) = run_in(&fake, code(), "add a line", &folder, &data);
    let fed = fake.fed();
    assert!(fed.contains("not changed: page.txt"), "{told:?}");
    assert!(!fed.contains("Write page.txt whole"), "{told:?}");
}

/// `folder` in `tmp` holding `files`, and the data folder beside it.
fn folder(tmp: &Path, files: &[(&str, &str)]) -> (std::path::PathBuf, std::path::PathBuf) {
    let folder = tmp.join("game");
    fs::create_dir(&folder).unwrap();
    for (name, text) in files {
        fs::write(folder.join(name), text).unwrap();
    }
    (folder, tmp.join("data"))
}

/// Runs `request` in place without tests on the session loop, saying yes to each ask.
fn in_place(fake: &Fake, request: &str, folder: &Path, data: &Path) -> Outcome {
    let task = untested(request, folder, data);
    (code().run_with(&fake.larch, &task, data, &mut |_| {}, &mut |_| {
        Decision::Once
    }))
    .unwrap()
}

/// An edit that fails on a file the room rewrites whole is made by the writer in the same step:
/// no second edit. At 64k the brief holds a file past the 12 KB an 8k context shows the writer.
#[test]
fn a_failed_edit_of_a_file_the_room_rewrites_is_written_whole() {
    let fake = Fake::qwen3(65536);
    let tmp = tempfile::tempdir().unwrap();
    let long: String = (1..=400)
        .map(|n| format!("let cell_{n:03} = {n}; // a board cell\n"))
        .collect();
    assert!(long.len() > 12_000);
    let (folder, data) = folder(tmp.path(), &[("script.js", &long)]);
    let args = json!({"path": "script.js", "old_text": "let n = 1;", "new_text": "let n = 2;"});
    call(&fake, "edit", args, false);
    let fixed = format!("{long}let n = 2;\n");
    fake.say(&format!("```js\n{fixed}```"));
    fake.say("Done.");
    fake.say("Done.");
    in_place(&fake, "make n 2", &folder, &data);
    let fed = fake.fed();
    assert!(
        fed.contains("script.js is written whole with this change instead"),
        "{fed}"
    );
    assert!(fed.contains("where it has:\nlet n = 1;\nit has instead:\nlet n = 2;"));
    assert_eq!(fs::read_to_string(folder.join("script.js")).unwrap(), fixed);
}

/// A file longer than the room rewrites whole is edited: write_file of it is refused.
#[test]
fn a_file_longer_than_the_room_rewrites_is_edited_not_written() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let long = long_py();
    let (folder, data) = folder(tmp.path(), &[("big.py", &long)]);
    let args = json!({"path": "big.py", "about": "the same with x1 = 0"});
    call(&fake, "write_file", args, false);
    fake.say("Done.");
    fake.say("Done.");
    in_place(&fake, "make x1 0", &folder, &data);
    let fed = fake.fed();
    let refused = "'big.py' is longer than the 729 lines written whole here: change it with \
                   replace_symbol or edit";
    assert!(fed.contains(refused), "{fed}");
    assert_eq!(fs::read_to_string(folder.join("big.py")).unwrap(), long);
}

/// The first prompt carries the code map, each file's lines and named blocks, the tools that
/// take a block by name, and, without tests, where commands run and what checks the work.
#[test]
fn the_first_prompt_has_the_code_map_and_says_there_are_no_tests() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let js = "function dropDisc(col) {\n  return col;\n}\n";
    let (folder, data) = folder(tmp.path(), &[("script.js", js)]);
    fake.say("Done.");
    fake.say("Done.");
    in_place(&fake, "make discs drop", &folder, &data);
    let fed = fake.fed();
    assert!(
        fed.contains("script.js, 3 lines\n  function dropDisc 1-3\n"),
        "{fed}"
    );
    assert!(fed.contains("replace_symbol to replace a named block whole"));
    assert!(fed.contains("given symbol, one named block whole"));
    // The build check the JavaScript gets, or none where the sandbox cannot run it on the base.
    assert!(
        fed.contains(", where commands run; no test command"),
        "{fed}"
    );
}

/// A write's result ends with what it leaves broken: here a page that links a script that is
/// not there.
#[test]
fn a_write_that_breaks_a_reference_is_told_in_its_result() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let page = "<html><body><p>hi</p></body></html>\n";
    let (folder, data) = folder(tmp.path(), &[("index.html", page)]);
    let args = json!({"path": "index.html", "old_text": "</body>", "new_text": "<script src=\"app.js\"></script></body>"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    in_place(&fake, "add the app", &folder, &data);
    let fed = fake.fed();
    let tail = fed.split("replaced 1 match in ").nth(1).unwrap_or_default();
    assert!(tail.contains("\nchecks: index.html links app.js"), "{fed}");
}

/// The same edit made again with the folder as the first left it is not run: it gets the
/// first's result back; a third time asks for the answer, and the next reply's words end the run.
#[test]
fn the_same_call_in_an_unchanged_folder_is_told_then_ends_with_the_models_words() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, data) = folder(tmp.path(), &[("script.js", "let n = 1;\n")]);
    let args = json!({"path": "script.js", "old_text": "n = 1", "new_text": "n = 2"});
    for _ in 0..3 {
        call(&fake, "edit", args.clone(), false);
    }
    fake.say("n is 2 now.");
    let outcome = in_place(&fake, "make n right", &folder, &data);
    let why = "the model made the same call 3 times with the folder unchanged; ended with the \
               model's answer";
    let fed = fake.fed();
    assert_eq!(outcome.packet.stopped.as_deref(), Some(why), "{fed}");
    assert_eq!(outcome.packet.pr_body, "n is 2 now.", "{fed}");
    assert!(fed.contains("no more tool calls"), "{fed}");
    let told = "not run: step 1 made this same call and the folder is as it was then, so it \
                gets the same: replaced 1 match in ";
    // The second and the third, which asks for the answer.
    assert_eq!(fed.matches(told).count(), 2, "{fed}");
    assert!(fed.contains("change your approach"), "{fed}");
    // The run's trace keeps each step and the stop.
    let dir = data.join("agent/trace");
    let file = fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
    let lines: Vec<serde_json::Value> = (fs::read_to_string(file).unwrap().lines())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines[0]["step"], 1);
    assert!(
        lines[0]["call"].as_str().unwrap().starts_with("edit "),
        "{lines:?}"
    );
    assert!(lines[0]["status"].is_string(), "{lines:?}");
    assert!(
        lines[1]["result"]
            .as_str()
            .unwrap()
            .starts_with("not run: step 1")
    );
    assert_eq!(lines.last().unwrap()["stop"], why);
}

/// Another call that gets what an earlier one got in the same folder state is told so.
#[test]
fn a_result_got_before_in_the_same_state_is_told_as_nothing_new() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let same = [
        ("a.js", "let n = 1;\nn++;\n"),
        ("b.js", "let n = 1;\nn++;\n"),
    ];
    let (folder, data) = folder(tmp.path(), &same);
    call(&fake, "view", json!({"path": "a.js"}), false);
    call(&fake, "view", json!({"path": "b.js"}), false);
    fake.say("Done.");
    fake.say("Done.");
    in_place(&fake, "make n right", &folder, &data);
    let fed = fake.fed();
    let told = "step 1 got this same result with the folder as it is: nothing new";
    assert_eq!(fed.matches(told).count(), 1, "{fed}");
}

/// A command made again after another command changed what git does not see (an ignored file)
/// runs again: a command's effects outrun the folder's state.
// The fake model's commands are POSIX shell, which cmd.exe does not run.
#[cfg(not(windows))]
#[test]
fn the_same_command_after_a_command_runs_again() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let files = [(".gitignore", "out.txt\n"), ("a.txt", "a\n")];
    let (folder, data) = folder(tmp.path(), &files);
    let cat = json!({"command": "cat out.txt"});
    call(&fake, "bash", cat.clone(), false);
    call(
        &fake,
        "bash",
        json!({"command": "echo hi > out.txt"}),
        false,
    );
    call(&fake, "bash", cat, false);
    let args = json!({"path": "a.txt", "old_text": "a", "new_text": "b"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let task = untested("make out", &folder, &data);
    let got = code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    });
    let Some(_) = ran(got) else {
        return;
    };
    let fed = fake.fed();
    if fed.contains("the sandbox is unavailable") {
        assert!(env::var_os("LARCH_SANDBOX_REQUIRED").is_none_or(|v| v != "1"));
        println!("skipped: the sandbox is unavailable");
        return;
    }
    assert!(!fed.contains("not run:"), "{fed}");
    assert!(fed.contains("exit code 0\nhi"), "{fed}");
}

/// A read made again gets its text again, not a line: compaction may have folded the first
/// away, and a read changes nothing.
#[test]
fn a_view_made_again_gets_the_file_again() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, data) = folder(tmp.path(), &[("script.js", "let n = 1;\n")]);
    for _ in 0..3 {
        call(&fake, "view", json!({"path": "script.js"}), false);
    }
    fake.say("Done.");
    fake.say("Done.");
    in_place(&fake, "make n right", &folder, &data);
    let fed = fake.fed();
    assert_eq!(fed.matches("1: let n = 1;").count(), 3, "{fed}");
    assert!(!fed.contains("not run:"), "{fed}");
}

/// Scripts `text`, then a view of `args`, as one reply.
fn said_then_view(fake: &Fake, text: &str, args: serde_json::Value) {
    fake.say_tokens(&fake.tokenizer.encode(&format!("{text}\n"), false));
    call(fake, "view", args, false);
}

/// Reads that bring nothing new, at a context that never compacts, end the run: the model is
/// asked for its answer in words with what it has, and that reply's words end it.
#[test]
fn reads_that_bring_nothing_new_end_with_the_models_words() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let files = [
        ("a.js", "let a = 1;\nlet b = 2;\n"),
        ("b.js", "let c = 3;\nlet d = 4;\n"),
    ];
    let (folder, data) = folder(tmp.path(), &files);
    for i in 0..30 {
        said_then_view(
            &fake,
            &format!("Look {i}."),
            json!({"path": files[i % 2].0}),
        );
    }
    let outcome = in_place(&fake, "make d 5", &folder, &data);
    let why = outcome.packet.stopped.as_deref().unwrap_or_default();
    assert!(why.contains("nothing new"), "{why}");
    // Two new reads, the stale ones, then the reply that answers.
    assert_eq!(outcome.steps, 2 + 6 + 1, "{}", fake.fed());
    assert_eq!(outcome.packet.pr_body, "Look 8.");
    assert!(fake.fed().contains("no more tool calls"), "{}", fake.fed());
}

/// An ending reply that is a bare call, after reads that brought nothing new, stops at the call's
/// marker and ends the run with a plain line, never an empty answer.
#[test]
fn an_ending_reply_that_calls_ends_with_words() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let files = [
        ("a.js", "let a = 1;\nlet b = 2;\n"),
        ("b.js", "let c = 3;\nlet d = 4;\n"),
    ];
    let (folder, data) = folder(tmp.path(), &files);
    for i in 0..8 {
        said_then_view(
            &fake,
            &format!("Look {i}."),
            json!({"path": files[i % 2].0}),
        );
    }
    let before = fake.unsaid();
    call(&fake, "view", json!({"path": "a.js"}), false);
    // From the ending reply's marker on, so the stream hears it before the engine says the rest.
    let rest = fake.unsaid() - before - 1;
    fake.pace_last(rest, Duration::from_millis(100));
    let outcome = in_place(&fake, "make d 5", &folder, &data);
    let why = outcome.packet.stopped.as_deref().unwrap_or_default();
    assert!(why.contains("nothing new"), "{why}");
    assert!(!outcome.packet.pr_body.trim().is_empty(), "{}", fake.fed());
    // The reply stopped at its marker: the call's arguments were never said.
    assert!(fake.unsaid() > 0, "{}", fake.fed());
}

/// Reads that bring nothing new, the last a long file's that sets off a compaction: the
/// compacted prompt still asks for the answer, though it cuts that read's result short.
#[test]
fn a_stale_read_that_compacts_still_asks_for_the_answer() {
    let fake = Fake::qwen3(16384);
    let tmp = tempfile::tempdir().unwrap();
    let big: String = (1..=400)
        .map(|n| format!("v{n} = '{}';\n", "x".repeat(120)))
        .collect();
    let mut files = vec![("big.js".to_string(), big)];
    files.extend((0..15).map(|i| (format!("s{i}.txt"), format!("note {i}\n"))));
    let named: Vec<(&str, &str)> = files
        .iter()
        .map(|(n, t)| (n.as_str(), t.as_str()))
        .collect();
    let (folder, data) = folder(tmp.path(), &named);
    // The long file and the short ones, five short ones again, then the long one again.
    for (name, _) in named.iter().chain(&named[1..6]).chain(&named[..1]) {
        said_then_view(&fake, "Look.", json!({"path": name}));
    }
    fake.say("Done.");
    let outcome = in_place(&fake, "make it", &folder, &data);
    let why = outcome.packet.stopped.as_deref().unwrap_or_default();
    assert!(why.contains("nothing new"), "{why}");
    let fed = fake.fed();
    let compacted = &fed[fed.rfind("Request: make it").unwrap()..];
    assert!(compacted.contains("no more tool calls"), "{compacted}");
}

/// A chess game's script of 330 lines, 13 KB.
fn chess() -> String {
    (1..=330)
        .map(|n| format!("board[{n}] = piece({n}, 'black'); // a square\n"))
        .collect()
}

/// A compaction keeps the part of a file too long to show whole that the model viewed, as it is
/// now: whole or nothing, it would be dropped and read again from its first line.
#[test]
fn a_compaction_keeps_the_part_of_a_long_file_it_viewed() {
    let fake = Fake::qwen3(16384);
    let tmp = tempfile::tempdir().unwrap();
    // Short notes after the long ones, so a short view sets off the compaction and leaves room.
    let short: Vec<(String, String)> = (0..12)
        .map(|i| (format!("s{i}.txt"), format!("a short note {i}\n")))
        .collect();
    let mut files = vec![
        ("game.js".to_string(), chess()),
        ("a.txt".into(), notes(100)),
        ("b.txt".into(), notes(50)),
    ];
    files.extend(short);
    let named: Vec<(&str, &str)> = files
        .iter()
        .map(|(n, t)| (n.as_str(), t.as_str()))
        .collect();
    let (folder, data) = folder(tmp.path(), &named);
    call(
        &fake,
        "view",
        json!({"path": "game.js", "start": 300}),
        false,
    );
    for (name, _) in &named[1..] {
        call(&fake, "view", json!({"path": name}), false);
    }
    fake.say("Done.");
    fake.say("Done.");
    in_place(&fake, "make the computer play black", &folder, &data);
    let fed = fake.fed();
    let compacted = &fed[fed.rfind("Request: make the computer").unwrap()..];
    let part = "game.js lines 300-330 as they are now:\n300: board[300] = piece(300, 'black')";
    assert!(compacted.contains(part), "{compacted}");
}

/// The chess follow-up: a model that pages through a ~330-line file a view at a time, starting
/// again at its first line once a compaction folded the pages away. At 8k, where the fake's tools
/// fill most of the context, as at 16k, the re-reads end the run with the model's words in
/// bounded steps and compactions (a compaction with nothing new since the last, or reads that
/// bring nothing new), never at its tool-call budget after dozens of compactions.
#[test]
fn paging_a_long_file_at_8k_ends_with_the_models_words() {
    for ctx in [8192, 16384] {
        let fake = Fake::qwen3(ctx);
        let tmp = tempfile::tempdir().unwrap();
        let (folder, data) = folder(tmp.path(), &[("game.js", &chess())]);
        for round in 0..20 {
            for start in [1, 109, 210] {
                let text = format!("Round {round}: game.js from line {start}.");
                said_then_view(&fake, &text, json!({"path": "game.js", "start": start}));
            }
        }
        let mut notices = Vec::new();
        let task = untested(
            "make the black pieces operated by the computer",
            &folder,
            &data,
        );
        let mut sink = |e| {
            if let AgentEvent::Notice(n) = e {
                notices.push(n);
            }
        };
        let outcome = (code().run_with(&fake.larch, &task, &data, &mut sink, &mut |_| {
            Decision::Once
        }))
        .unwrap();
        let compactions = notices.iter().filter(|n| n.ends_with("compacted")).count();
        // At most one per page it read, and the one that ends it.
        assert!(compactions <= 4, "{ctx}: {notices:?}");
        assert!(outcome.steps <= 12, "{ctx}: {} {notices:?}", outcome.steps);
        assert!(fake.unsaid() > 0, "{ctx}: {notices:?}");
        let why = outcome.packet.stopped.unwrap_or_default();
        assert!(why.contains("nothing new"), "{ctx}: {why} {notices:?}");
        let words = &outcome.packet.pr_body;
        assert!(words.starts_with("Round "), "{ctx}: {words:?} {notices:?}");
        assert!(fake.fed().contains("no more tool calls"), "{}", fake.fed());
    }
}

/// A command made again after a real change runs again: the folder's state tells them apart.
#[test]
fn the_same_command_after_an_edit_runs_again() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, data) = folder(tmp.path(), &[("script.js", "let n = 1;\n")]);
    call(&fake, "bash", json!({"command": "cat script.js"}), false);
    let args = json!({"path": "script.js", "old_text": "n = 1", "new_text": "n = 2"});
    call(&fake, "edit", args, false);
    call(&fake, "bash", json!({"command": "cat script.js"}), false);
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("make n right", &folder, &data);
    let got = code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |_| {
        Decision::Once
    });
    let Some(_) = ran(got) else {
        return;
    };
    let fed = fake.fed();
    assert!(!fed.contains("not run:"), "{fed}");
    assert!(fed.contains("let n = 2;"), "{fed}");
}

/// A file of 800 lines, past the 729 a 32k context rewrites whole.
fn long_py() -> String {
    (1..=800).map(|n| format!("x{n} = {n}\n")).collect()
}

/// An edit the user says no to is never rewritten by the writer: one ask, no second one.
#[test]
fn a_declined_edit_is_not_rewritten() {
    let fake = Fake::qwen3(65536);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, data) = folder(tmp.path(), &[("script.js", "let n = 1;\n")]);
    let args = json!({"path": "script.js", "old_text": "let n = 1;", "new_text": "let n = 2;"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    let task = untested("make n 2", &folder, &data);
    let mut asked = Vec::new();
    (code().run_with(&fake.larch, &task, &data, &mut |_| {}, &mut |a| {
        asked.push(a.name.clone());
        Decision::No
    }))
    .unwrap();
    let fed = fake.fed();
    assert!(fed.contains("the user said no"), "{fed}");
    assert!(
        !fed.contains("is written whole with this change instead"),
        "{fed}"
    );
    assert_eq!(asked, ["edit"]);
}

/// A file longer than the room rewrites is edited, and written whole by the writer once two of
/// its edits failed.
#[test]
fn a_long_file_is_rewritten_after_two_failed_edits() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let long = long_py();
    let (folder, data) = folder(tmp.path(), &[("big.py", &long)]);
    let args = json!({"path": "big.py", "old_text": "x1 = 9", "new_text": "x1 = 0"});
    call(&fake, "edit", args.clone(), false);
    call(&fake, "edit", args, false);
    let fixed = long.replacen("x1 = 1\n", "x1 = 0\n", 1);
    fake.say(&format!("```python\n{fixed}```"));
    fake.say("Done.");
    fake.say("Done.");
    in_place(&fake, "make x1 0", &folder, &data);
    let fed = fake.fed();
    let rewrite = "big.py is written whole with this change instead";
    assert_eq!(fed.matches(rewrite).count(), 1, "{fed}");
    assert_eq!(fs::read_to_string(folder.join("big.py")).unwrap(), fixed);
}

/// Once two edits of a long file failed, the model's own write_file of it is no longer refused.
#[test]
fn write_file_of_a_long_file_is_allowed_after_two_failed_edits() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let long = long_py();
    let (folder, data) = folder(tmp.path(), &[("big.py", &long)]);
    let args = json!({"path": "big.py", "old_text": "x1 = 9", "new_text": "x1 = 0"});
    call(&fake, "edit", args, false);
    cut_of(&fake, "big.py");
    let write = json!({"path": "big.py", "about": "the same with x1 = 0"});
    call(&fake, "write_file", write, false);
    let fixed = long.replacen("x1 = 1\n", "x1 = 0\n", 1);
    fake.say(&format!("```python\n{fixed}```"));
    fake.say("Done.");
    fake.say("Done.");
    in_place(&fake, "make x1 0", &folder, &data);
    let fed = fake.fed();
    assert!(!fed.contains("is longer than the 729 lines"), "{fed}");
    assert_eq!(fs::read_to_string(folder.join("big.py")).unwrap(), fixed);
}

/// The map of a repository whose file names alone would fill it shows each folder with its
/// count of files, then the files closest to the request with their blocks, never a run of
/// "more symbols" lines.
#[test]
fn a_big_repository_maps_by_folder_and_the_files_closest_to_the_request() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, data) = folder(tmp.path(), &[]);
    fs::create_dir(folder.join("src")).unwrap();
    for f in 0..600 {
        let name = if f == 321 { "zebra" } else { "step" };
        let text = format!("def {name}(x):\n    return x\n\n\ndef other_{f}(x):\n    return x\n");
        fs::write(folder.join(format!("src/m{f:04}.py")), text).unwrap();
    }
    fake.say("Done.");
    fake.say("Done.");
    in_place(&fake, "speed up zebra", &folder, &data);
    let fed = fake.fed();
    let map = fed.split("Repository map:").nth(1).unwrap_or_default();
    assert!(
        map.contains("600 files, by folder:\nsrc/: 600 files\n"),
        "{map}"
    );
    let closest = "The files closest to the request:\nsrc/m0321.py, 6 lines\n  def zebra 1-2\n";
    assert!(map.contains(closest), "{map}");
    assert!(!map.contains("more symbols"), "{map}");
}

/// An editor's rewrite goes through write_file's checks: one that drops a function its change
/// does not name is refused, and the file goes back to the session.
#[test]
fn an_editor_that_drops_a_function_falls_back_to_the_session() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, data) = folder(
        tmp.path(),
        &[
            (
                "a.js",
                "function a() {\n  return 1;\n}\n\nfunction b() {\n  return 2;\n}\n",
            ),
            ("b.js", "function c() {\n  return a();\n}\n"),
        ],
    );
    plan(
        &fake,
        r#"{"contract": "", "files": [{"path": "a.js", "about": "a returns 3"}, {"path": "b.js", "about": "c returns 4"}]}"#,
    );
    fake.say("function a() {\n  return 3;\n}\n");
    fake.say("function c() {\n  return 4;\n}\n");
    fake.say("Done.");
    fake.say("Done.");
    let (_, told) = run_in(&fake, code(), "a returns 3, c returns 4", &folder, &data);
    assert!(fake.fed().contains("not changed: a.js\n"), "{told:?}");
    assert!(
        fs::read_to_string(folder.join("a.js"))
            .unwrap()
            .contains("function b")
    );
    assert_eq!(
        fs::read_to_string(folder.join("b.js")).unwrap(),
        "function c() {\n  return 4;\n}\n"
    );
}

/// Why a done with no tests ends a run.
const NO_TESTS: &str = "the model said it is done; the folder has no tests";

/// A folder in `tmp` with `files`, done in place without tests on `fake`'s data: "write the
/// notes".
fn notes_task(fake: &Fake, tmp: &Path, files: &[(&str, &str)]) -> CodeTask {
    let folder = tmp.join("notes");
    fs::create_dir(&folder).unwrap();
    for (name, text) in files {
        fs::write(folder.join(name), text).unwrap();
    }
    untested("write the notes", &folder, &fake.data())
}

/// Scripts the edit that replaces `old` with `new` in `path`.
fn edit_in(fake: &Fake, path: &str, old: &str, new: &str) {
    let args = json!({"path": path, "old_text": old, "new_text": new});
    call(fake, "edit", args, false);
}

/// `task` on the session loop: the outcome, the progress lines and the notices.
fn watched(fake: &Fake, task: &CodeTask) -> Option<(Outcome, Vec<String>, Vec<String>)> {
    let (mut progress, mut notices) = (Vec::new(), Vec::new());
    let mut sink = |e: AgentEvent| match e {
        AgentEvent::Progress(p) => progress.push(p),
        AgentEvent::Notice(n) => notices.push(n),
        _ => {}
    };
    let data = fake.data();
    let got = code().run_with(&fake.larch, task, &data, &mut sink, &mut |_| Decision::Once);
    Some((ran(got)?, progress, notices))
}

/// A reviewer is a same-model judge with no ground truth: its doubt of a done with no tests
/// behind it goes with the done, unverified, and never sends it back. The first done gets the
/// static review alone, which sends it back anyway.
#[test]
fn a_reviewers_doubt_goes_with_the_done_and_never_sends_it_back() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let task = notes_task(&fake, tmp.path(), &[("notes.txt", "draft\n")]);
    edit_in(&fake, "notes.txt", "draft", "final");
    fake.say("Done.");
    fake.say("Done.");
    fake.say("not done: the notes have no title");
    let Some((outcome, progress, notices)) = watched(&fake, &task) else {
        return;
    };
    let doubt = format!("{NO_TESTS}; a reviewer doubts it: the notes have no title");
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(doubt.as_str()),
        "{progress:?}"
    );
    assert_eq!(outcome.packet.verdict, Verdict::Unverified);
    let fed = fake.fed();
    assert!(!fed.contains("a reviewer read the change"), "{fed}");
    assert!(
        fed.contains("The changed lines:\nnotes.txt now, lines 1-1 of 1:\n1: final"),
        "{fed}"
    );
    let line = "Reviewer: not done: the notes have no title";
    assert!(progress.iter().any(|p| p == line), "{progress:?}");
    let checking = progress
        .iter()
        .filter(|p| p.starts_with("Reviewer: checking"));
    assert_eq!(checking.count(), 1, "{progress:?}");
    assert_eq!(
        notices.iter().filter(|n| *n == NO_TESTS).count(),
        1,
        "{notices:?}"
    );
    assert_eq!(fake.unsaid(), 0);
    // The run's trace keeps the reviewer's verdict beside the steps.
    let dir = fake.data().join("agent/trace");
    let file = fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
    let trace = fs::read_to_string(file).unwrap();
    assert!(
        trace.contains(&format!("\"reviewer\":\"{line}\"")),
        "{trace}"
    );
}

/// A pool that holds one context has no reviewer: a tested run never says so, and a run
/// without tests says it once, at the first done it would have read.
#[test]
fn a_reviewer_off_is_told_once_and_only_where_it_would_read() {
    let off = |progress: &[String]| {
        progress
            .iter()
            .filter(|p| p.starts_with("Reviewer: off"))
            .count()
    };
    let tmp = tempfile::tempdir().unwrap();
    let fake = Fake::qwen3_one_context(32768);
    let notes = notes_task(&fake, tmp.path(), &[("notes.txt", "draft\n")]);
    edit_in(&fake, "notes.txt", "draft", "final");
    for _ in 0..3 {
        fake.say("Done.");
    }
    let (outcome, progress, _) = watched(&fake, &notes).unwrap();
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{progress:?}"
    );
    assert_eq!(off(&progress), 1, "{progress:?}");
    let fake = Fake::qwen3_one_context(32768);
    let repo = toy(tmp.path(), &[]);
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let Some((mut outcome, progress, _)) = watched(&fake, &task(&repo)) else {
        return;
    };
    assert_eq!(off(&progress), 0, "{progress:?}");
    outcome.task.discard().unwrap();
}

/// The tests are the judge: a done they agree with is never reviewed.
#[test]
fn a_done_the_tests_pass_is_never_reviewed() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path(), &[]);
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let Some((mut outcome, progress, _)) = watched(&fake, &task(&repo)) else {
        return;
    };
    let done = "the model said it is done and the tests agree";
    assert_eq!(outcome.packet.stopped.as_deref(), Some(done));
    assert!(
        !progress.iter().any(|p| p.starts_with("Reviewer: checking")),
        "{progress:?}"
    );
    assert_eq!(fake.unsaid(), 0);
    outcome.task.discard().unwrap();
}

/// The reviewer reads the changed lines first; a file too long to add whole after them is left
/// out, and the reviewer is told so.
#[test]
fn the_reviewer_is_told_when_part_of_the_files_is_cut() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let mut big: String = (1..=1500)
        .map(|n| format!("line {n} of the notes\n"))
        .collect();
    big += "the end\n";
    let task = notes_task(&fake, tmp.path(), &[("big.txt", &big)]);
    edit_in(&fake, "big.txt", "the end", "the real end");
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let Some((outcome, progress, _)) = watched(&fake, &task) else {
        return;
    };
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{progress:?}"
    );
    let fed = fake.fed();
    let review = &fed[fed.rfind("You review a change").expect("the reviewer ran")..];
    assert!(review.contains("1501: the real end"), "{review}");
    assert!(
        review.contains("Parts of the files were left out for room"),
        "{review}"
    );
    assert!(!review.contains("big.txt whole:"), "{review}");
    assert_eq!(fake.unsaid(), 0);
}

/// A test file that takes a while and says, in overlap.log, when another run of it was running.
const SLOW_TEST: &str = "import fcntl, time\nlock = open('lock', 'w')\ntry:\n    \
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\nexcept OSError:\n    \
open('overlap.log', 'a').write('overlap\\n')\ntime.sleep(1.5)\nfrom calc import add\n\
assert add(2, 3) == 5, add(2, 3)\nprint(\"ok\")\n";

/// The toy with `SLOW_TEST` as its test.
fn slow_toy(tmp: &Path) -> std::path::PathBuf {
    toy(
        tmp,
        &[
            ("test_calc.py", SLOW_TEST),
            (".gitignore", "lock\noverlap.log\n"),
        ],
    )
}

/// Tests still running when an attempt stops (its tokens spent after a view, which they go on
/// beside) stop before the packet runs the tests and commits.
#[test]
fn background_checks_stop_before_the_packet() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = slow_toy(tmp.path());
    edit(&fake, "a - b", "a + b");
    call(&fake, "view", json!({"path": "calc.py"}), false);
    let args = json!({"path": "calc.py", "old_text": "a - b", "new_text": "a + b"});
    let said = format!("\n{{\"name\": \"edit\", \"arguments\": {args}}}\n");
    let tokens = fake.tokenizer.encode(&said, false).len() as u64 + 3;
    let mut task = task(&repo);
    task.budget.tokens = tokens + 2;
    let Some((mut outcome, progress, _)) = watched(&fake, &task) else {
        return;
    };
    let stopped = outcome.packet.stopped.clone().unwrap_or_default();
    assert!(
        stopped.starts_with("the tokens budget"),
        "{stopped} {progress:?}"
    );
    assert!(
        progress
            .iter()
            .any(|p| p == "Tests running in the background")
    );
    let dir = outcome.task.dir().unwrap().to_path_buf();
    assert!(
        !dir.join("overlap.log").exists(),
        "two test runs overlapped"
    );
    outcome.task.discard().unwrap();
}

/// Tests a newer change outdates stop at once: the step never waits for them, their result
/// never reaches the model, and the next run of the tests never overlaps them.
#[test]
fn a_change_stops_the_checks_it_outdates() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = slow_toy(tmp.path());
    edit(&fake, "a - b", "a * b");
    edit(&fake, "a * b", "a + b");
    fake.say("Done.");
    let Some((mut outcome, progress, _)) = watched(&fake, &task(&repo)) else {
        return;
    };
    let done = "the model said it is done and the tests agree";
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(done),
        "{progress:?}"
    );
    // The second edit ends the first's tests without a line; the done says it runs them again.
    let stopped = "Tests in the background stopped: running them again for the done";
    let lines: Vec<&str> = (progress.iter().map(String::as_str))
        .filter(|p| p.starts_with("Tests in the background stopped"))
        .collect();
    assert_eq!(lines, [stopped], "{progress:?}");
    assert!(!fake.fed().contains("the tests after your previous change"));
    let dir = outcome.task.dir().unwrap().to_path_buf();
    assert!(
        !dir.join("overlap.log").exists(),
        "two test runs overlapped"
    );
    outcome.task.discard().unwrap();
}

/// A change's tests run in the background while the model writes its next step; a step that
/// only reads gets what they said, labelled as the previous change's, and a done with nothing
/// changed since reads them instead of running the tests again.
#[test]
fn a_background_check_joins_a_later_step_and_the_done_reuses_it() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let counted = "open('runs.log', 'a').write('x\\n')\nfrom calc import add\nassert add(2, 3) == 5, add(2, 3)\n";
    let repo = toy(
        tmp.path(),
        &[("test_calc.py", counted), (".gitignore", "runs.log\n")],
    );
    edit(&fake, "a - b", "a * b");
    call(&fake, "view", json!({"path": "calc.py"}), false);
    edit(&fake, "a * b", "a + b");
    fake.say("Done: add adds now, and the tests after the change pass as they should.");
    // Each step's reply outlasts a test run.
    fake.pace(Duration::from_millis(100));
    let Some((mut outcome, progress, _)) = watched(&fake, &task(&repo)) else {
        return;
    };
    let done = "the model said it is done and the tests agree";
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(done),
        "{progress:?}"
    );
    let fed = fake.fed();
    let view = fed.find("{\"path\":\"calc.py\"}").unwrap();
    let tests = fed.find("the tests after your previous change: ");
    assert!(tests > Some(view), "{fed}");
    let joined = progress
        .iter()
        .filter(|p| p.starts_with("In the background: tests: "));
    assert_eq!(joined.count(), 2, "{progress:?}");
    let dir = outcome.task.dir().unwrap().to_path_buf();
    let runs = fs::read_to_string(dir.join("runs.log"))
        .unwrap()
        .lines()
        .count();
    assert_eq!(runs, 3, "the base, then each change in the background");
    assert_eq!(fake.unsaid(), 0);
    outcome.task.discard().unwrap();
}

/// A plan's editors' files reach the reviewer as a reply's own writes do: a done after the fan-out
/// shows it each edited file's changed lines against its earlier text, not the whole file.
#[test]
fn a_fan_out_then_done_shows_the_reviewer_the_edited_lines() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let text = |name: &str| -> String { (1..=30).map(|n| format!("{name} {n}\n")).collect() };
    let (a, b) = (text("a"), text("b"));
    let task = notes_task(&fake, tmp.path(), &[("a.txt", &a), ("b.txt", &b)]);
    plan(
        &fake,
        r#"{"contract": "", "files": [{"path": "a.txt", "about": "line 15 says final"}, {"path": "b.txt", "about": "line 15 says final"}]}"#,
    );
    fake.say(&a.replace("a 15\n", "final\n"));
    fake.say(&b.replace("b 15\n", "final\n"));
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let Some((outcome, progress, _)) = watched(&fake, &task) else {
        return;
    };
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{progress:?}"
    );
    let fed = fake.fed();
    let review = fed.split("The changed lines:").nth(1).unwrap_or_default();
    for name in ["a.txt", "b.txt"] {
        let lines = format!("{name} now, lines 5-25 of 30:");
        assert!(review.contains(&lines), "{name}: {review}");
    }
    let ok = "Reviewer: the change does what was asked";
    assert!(progress.iter().any(|p| p == ok), "{progress:?}");
}

/// A folder in `tmp` whose map passes a quarter of a 32k context: `net/` and `ui/`, each of `n`
/// files, one file in `net/` defining zebra.
fn two_areas(tmp: &Path, n: usize) -> (std::path::PathBuf, std::path::PathBuf) {
    let (folder, data) = folder(tmp, &[]);
    for area in ["net", "ui"] {
        fs::create_dir(folder.join(area)).unwrap();
        for f in 0..n {
            let name = if (area, f) == ("net", 7) {
                "zebra"
            } else {
                "step"
            };
            let text =
                format!("def {name}(x):\n    return x\n\n\ndef {area}_{f}(x):\n    return x\n");
            fs::write(folder.join(format!("{area}/m{f:03}.py")), text).unwrap();
        }
    }
    (folder, data)
}

/// The files of each area of `two_areas` in a big repository.
const AREA_FILES: usize = 400;

/// Code mode on the session loop with area experts on.
fn with_experts() -> Code {
    Code {
        experts: true,
        ..code()
    }
}

/// Asks the experts `question`.
fn ask_experts(fake: &Fake, question: &str) {
    call(fake, "ask_experts", json!({ "question": question }), false);
}

/// The tools each audited call of a run in `data` ran, in order.
fn audited_tools(data: &Path) -> Vec<String> {
    let log = fs::read_dir(data.join("agent/log")).unwrap();
    (log.map(|f| fs::read_to_string(f.unwrap().path()).unwrap()))
        .flat_map(|text| {
            text.lines()
                .map(|l| serde_json::from_str::<AuditEvent>(l).unwrap())
                .collect::<Vec<_>>()
        })
        .filter_map(|e| e.tool)
        .collect()
}

/// A repository whose map passes a quarter of the context gets experts. An ask goes to the area
/// whose code matches it, with one live line, and its places come back as path:line, through
/// the audited call. On a model without a prefix cache (a hybrid's case) an expert parks
/// nothing: the second ask reads its map again, and the lead's next step still reads its
/// transcript from its own parked KV.
#[test]
fn area_experts_answer_the_lead_with_file_and_line() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, data) = two_areas(tmp.path(), AREA_FILES);
    ask_experts(&fake, "where is zebra");
    fake.say("net/m007.py:1: def zebra takes x");
    ask_experts(&fake, "what does zebra take");
    fake.say("net/m007.py:2: zebra returns x");
    fake.say("Done.");
    fake.say("Done.");
    let (_, told) = run_in(&fake, with_experts(), "speed up zebra", &folder, &data);
    assert!(
        told.iter()
            .any(|t| t.starts_with("Area experts:") && t.ends_with(" areas, one sequence an ask")),
        "{told:?}"
    );
    let asking = "Asking 1 area experts: where is zebra";
    assert!(told.iter().any(|t| t == asking), "{told:?}");
    let fed = fake.fed();
    let places = "The area experts' places:\nnet/m007.py:1: def zebra takes x";
    assert!(fed.contains(places), "{fed}");
    assert!(!fed.contains("What lookup finds follows"), "{fed}");
    // The expert holds its area's map, and reads it again on the second ask.
    assert_eq!(
        fed.matches("The area's files:\nnet/m000.py, 6 lines")
            .count(),
        2
    );
    assert_eq!(
        audited_tools(&data)
            .iter()
            .filter(|t| *t == "ask_experts")
            .count(),
        2
    );
    // The lead's step after each ask reads its last prompt from its parked KV.
    let turns = turns(&data);
    for pair in turns.windows(2).take(2) {
        let ((prompt, _), (_, cached)) = (pair[0], pair[1]);
        // All but the template tail of the last turn, which each prompt renders again.
        assert!(cached + 16 >= prompt, "{turns:?}");
    }
}

/// An expert that names a line outside every named block keeps no place, and the step falls
/// back to the single session's lookup, audited, which finds zebra.
#[test]
fn a_failing_expert_falls_back_to_lookup() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, data) = two_areas(tmp.path(), AREA_FILES);
    ask_experts(&fake, "where is zebra");
    fake.say("net/m007.py:3: zebra");
    fake.say("Done.");
    fake.say("Done.");
    let (_, told) = run_in(&fake, with_experts(), "speed up zebra", &folder, &data);
    let back = "Area experts: falling back to lookup";
    assert!(told.iter().any(|t| t == back), "{told:?}");
    let fed = fake.fed();
    let dropped = "dropped, not in its area's named blocks: net/m007.py:3: zebra (is no line of a named block)";
    assert!(fed.contains(dropped), "{fed}");
    assert!(fed.contains("What lookup finds follows."), "{fed}");
    let found = "net/m007.py, def zebra, lines 1-2:\n1: def zebra(x):";
    assert!(fed.contains(found), "{fed}");
    let tools = audited_tools(&data);
    assert!(tools.contains(&"lookup".to_string()), "{tools:?}");
}

/// Stop pressed while the experts are asked ends the run there: no lookup runs after them.
#[test]
fn a_stop_during_an_expert_ask_runs_no_lookup() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, data) = two_areas(tmp.path(), AREA_FILES);
    ask_experts(&fake, "where is zebra");
    fake.say("net/m007.py:1: zebra");
    fake.say("Done.");
    let task = untested("speed up zebra", &folder, &data);
    let stop = Arc::new(AtomicBool::new(false));
    let mut told = Vec::new();
    let mut sink = |e: AgentEvent| {
        if let AgentEvent::Notice(n) | AgentEvent::Progress(n) = e {
            if n.starts_with("Asking ") {
                stop.store(true, Ordering::Relaxed);
            }
            told.push(n);
        }
    };
    let outcome = (with_experts().run_until(
        &fake.larch,
        &task,
        &data,
        &mut sink,
        &mut |_| Decision::Once,
        &stop,
    ))
    .unwrap();
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some("the user stopped the run")
    );
    assert!(!told.iter().any(|t| t.contains("falling back")), "{told:?}");
    let tools = audited_tools(&data);
    assert!(!tools.contains(&"lookup".to_string()), "{tools:?}");
}

/// No experts for a small repository, for a big one with them off, or for the same big one at
/// 128k, whose map fits a quarter of that context: the lead is never offered ask_experts.
#[test]
fn experts_stay_off_for_a_small_repository_when_off_or_when_the_map_fits() {
    for (ctx, files, code) in [
        (32768, 3, with_experts()),
        (32768, AREA_FILES, code()),
        (131_072, AREA_FILES, with_experts()),
    ] {
        let fake = Fake::qwen3(ctx);
        let tmp = tempfile::tempdir().unwrap();
        let (folder, data) = two_areas(tmp.path(), files);
        fake.say("Done.");
        fake.say("Done.");
        let (_, told) = run_in(&fake, code, "speed up zebra", &folder, &data);
        assert!(
            !told.iter().any(|t| t.starts_with("Area experts")),
            "{told:?}"
        );
        assert!(!fake.fed().contains("ask_experts"), "{ctx} {files}");
    }
}

/// `make f right` in `folder` in place without tests, its build checks `checks`: the outcome,
/// the status lines and the notices, or None where the sandbox is unavailable.
fn checked_run(
    fake: &Fake,
    folder: &Path,
    checks: &[String],
) -> Option<(Outcome, Vec<String>, Vec<String>)> {
    let overrides = Overrides {
        test_cmd: None,
        allowed_paths: None,
        check_cmds: checks.to_vec(),
        budget: BUDGET,
    };
    let data = fake.data();
    let task = CodeTask::in_folder("make f right", folder, &data, overrides).unwrap();
    let (mut progress, mut notices) = (Vec::new(), Vec::new());
    let mut sink = |e: AgentEvent| match e {
        AgentEvent::Progress(p) => progress.push(p),
        AgentEvent::Notice(n) => notices.push(n),
        _ => {}
    };
    let got = code().run_with(&fake.larch, &task, &data, &mut sink, &mut |_| {
        Decision::Once
    });
    let outcome = ran(got)?;
    sandboxed(fake, &notices)?;
    Some((outcome, progress, notices))
}

/// Some(()), or None, with the reason printed, when a check or command the sandbox cannot run
/// was told, as a run it cannot sandbox is (`ran`).
fn sandboxed(fake: &Fake, notices: &[String]) -> Option<()> {
    let fed = fake.fed();
    let unsandboxed = (notices.iter().map(String::as_str))
        .chain(fed.lines())
        .find(|n| n.contains("the sandbox is unavailable"));
    if let Some(why) = unsandboxed {
        assert!(
            env::var_os("LARCH_SANDBOX_REQUIRED").is_none_or(|v| v != "1"),
            "LARCH_SANDBOX_REQUIRED=1 but {why}"
        );
        println!("skipped: {why}");
        return None;
    }
    Some(())
}

/// A done whose changed Python file does not compile, written by a command, gets the compiler's
/// error back as its result; the fix's done goes through.
// Build checks are POSIX shell, so task::checks gives Windows none.
#[cfg(not(windows))]
#[test]
fn a_done_with_a_failing_build_check_gets_its_output_back() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, _) = folder(tmp.path(), &[("f.py", "def f():\n    return 1\n")]);
    // A command's write: an edit's own syntax gate would have refused it.
    let broken = json!({"command": "printf 'def f(:\\n    return 1\\n' > f.py"});
    call(&fake, "bash", broken, false);
    fake.say("Done.");
    let fix = json!({"old_text": "def f(:\n    return 1", "new_text": "def f():\n    return 2"});
    let args = json!({"path": "f.py", "old_text": fix["old_text"], "new_text": fix["new_text"]});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let checks = checks(&["f.py".into()]);
    let Some((outcome, progress, _)) = checked_run(&fake, &folder, &checks) else {
        return;
    };
    let fed = fake.fed();
    let told = "a build check of the changed files fails, so it is not done yet:\n\
                `PYTHONPYCACHEPREFIX=\"$TMPDIR\" python3 -B -c ";
    assert_eq!(fed.matches(told).count(), 1, "{fed}");
    assert!(fed.contains("SyntaxError"), "{fed}");
    // Its bytecode went to the private tmp, not the folder.
    assert!(!folder.join("__pycache__").exists());
    assert!(
        fed.contains("an import of each changed .py module the base has"),
        "{fed}"
    );
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{progress:?}"
    );
}

/// A change that compiles but breaks its module's import (an enum member that extends an enum)
/// is caught by the build check: the done gets the import's error back.
// Build checks are POSIX shell, so `task::checks` gives Windows none.
#[cfg(not(windows))]
#[test]
fn a_done_whose_module_no_longer_imports_gets_the_error_back() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let text = "import enum\n\nclass A(enum.Enum):\n    x = 1\n";
    let (folder, _) = folder(tmp.path(), &[("e.py", text)]);
    let more = "    x = 1\n\nclass B(A):\n    y = 2\n";
    let args = json!({"path": "e.py", "old_text": "    x = 1\n", "new_text": more});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    let args =
        json!({"path": "e.py", "old_text": "class B(A):", "new_text": "class B(enum.Enum):"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let checks = checks(&["e.py".into()]);
    let Some((outcome, progress, _)) = checked_run(&fake, &folder, &checks) else {
        return;
    };
    let fed = fake.fed();
    // "Cannot extend enumerations" before Python 3.11.
    assert_eq!(fed.matches("annot extend").count(), 1, "{fed}");
    // Results name the folder's files from its top, never where it is on the host.
    assert!(fed.contains("replaced 1 match in e.py at line 4"), "{fed}");
    let host = fs::canonicalize(&folder).unwrap();
    assert!(!fed.contains(&*host.to_string_lossy()), "{fed}");
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{progress:?}"
    );
}

/// A new script that reads its input as it loads (no `__main__` guard) is compiled, not
/// imported: there is no base to tell its import's failure from, and its done goes through. The
/// first prompt names the checks, not their commands.
// The fake model's commands are POSIX shell, which cmd.exe does not run.
#[cfg(not(windows))]
#[test]
fn a_new_script_is_compiled_not_imported() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, _) = folder(tmp.path(), &[("f.py", "def f():\n    return 1\n")]);
    let script = json!({"command": "printf 'name = input()\\nprint(name)\\n' > game.py"});
    call(&fake, "bash", script, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let checks = checks(&["f.py".into()]);
    let Some((outcome, progress, _)) = checked_run(&fake, &folder, &checks) else {
        return;
    };
    let fed = fake.fed();
    assert!(
        !fed.contains("a build check of the changed files fails"),
        "{fed}"
    );
    assert!(
        progress.iter().any(|p| p.contains("py_compile 'game.py'")),
        "{progress:?}"
    );
    assert!(
        fed.contains("an import of each changed .py module the base has"),
        "{fed}"
    );
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{progress:?}"
    );
}

/// Where no test is near the change in a suite too big to run whole, no tests ran: that says
/// nothing, so it is no pass, the build checks run at the done instead, and the packet says the
/// tests were not run and the change is unverified.
#[test]
fn no_tests_near_the_change_is_no_pass() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let tests: Vec<(String, String)> = (0..40)
        .map(|n| (format!("test_x{n}.py"), "import unittest\n".to_string()))
        .collect();
    let mut more: Vec<(&str, &str)> = tests
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    more.push(("util.py", "def u():\n    return 1\n"));
    let repo = toy(tmp.path(), &more);
    let args = json!({"path": "util.py", "old_text": "return 1", "new_text": "return 2"});
    call(&fake, "edit", args, false);
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done.");
    // The reviewer reads a done no test checked.
    fake.say("ok");
    let mut task = task(&repo);
    task.test_cmd = Some(format!("{UNITTEST} {{tests}}"));
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task, &data)) else {
        return;
    };
    let fed = fake.fed();
    assert!(fed.contains("no tests ran"), "{fed}");
    assert!(!fed.contains("The change passes its checks"), "{fed}");
    let p = &outcome.packet;
    let unverified = "the model said it is done; no tests ran near the change";
    assert_eq!(p.stopped.as_deref(), Some(unverified), "{fed}");
    assert_eq!(p.verdict, Verdict::Unverified);
    assert!(
        p.render().contains("tests after: not run\n"),
        "{}",
        p.render()
    );
    outcome.task.discard().unwrap();
}

/// Where none is near the change in a suite too big to run whole, run_tests says to name the
/// tests; the ones the model names run, and its done runs those same ones again, which pass.
#[test]
fn a_done_runs_the_tests_the_model_named() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let tests: Vec<(String, String)> = (0..40)
        .map(|n| (format!("test_x{n}.py"), "import unittest\n".to_string()))
        .collect();
    let mut more: Vec<(&str, &str)> = tests
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    more.push(("util.py", "def u():\n    return 1\n"));
    let check = "import unittest\nfrom util import u\n\nclass T(unittest.TestCase):\n    \
                 def test_u(self):\n        self.assertEqual(u(), 2)\n";
    more.push(("test_u.py", check));
    let repo = toy(tmp.path(), &more);
    let args = json!({"path": "util.py", "old_text": "return 1", "new_text": "return 2"});
    call(&fake, "edit", args, false);
    call(&fake, "run_tests", json!({}), false);
    call(&fake, "run_tests", json!({"tests": ["test_u.py"]}), false);
    fake.say("Done.");
    fake.say("ok");
    let mut task = task(&repo);
    task.test_cmd = Some(format!("{UNITTEST} {{tests}}"));
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task, &data)) else {
        return;
    };
    // The named run is the first that needs the sandbox; as root it is refused.
    if sandboxed(&fake, &[]).is_none() {
        return;
    }
    let fed = fake.fed();
    assert!(fed.contains("run_tests with tests=[...]"), "{fed}");
    assert!(fed.contains("1 passed, 0 failed"), "{fed}");
    let p = &outcome.packet;
    let agree = "the model said it is done and the tests agree";
    assert_eq!(p.stopped.as_deref(), Some(agree), "{fed}");
    let after = p.tests_after.as_ref().expect("the done's run");
    assert_eq!((after.passed, after.failed), (1, 0));
    assert_eq!(p.verdict, Verdict::Verified);
    outcome.task.discard().unwrap();
}

/// The toy with a unittest of add, so a failure has a name, run in simple mode: run_tests
/// reports the run as it is, and a done ends the attempt with the last run since the last
/// change as the verdict, never held or run again. Its test leaves a file behind, as a coverage
/// run does; `more` adds files.
fn simple(fake: &Fake, tmp: &Path, more: &[(&str, &str)]) -> Option<Outcome> {
    let test = "import unittest\nfrom calc import add\n\nopen(\"left.txt\", \"w\").write(\"x\")\n\n\
                class T(unittest.TestCase):\n    \
                def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n";
    let repo = toy(tmp, &[more, &[("test_add.py", test)]].concat());
    let mut task = task(&repo);
    task.test_cmd = Some(format!("{UNITTEST} test_add"));
    let code = Code {
        verify: Verify::Simple,
        ..code()
    };
    ran(code.run(&fake.larch, &task, &tmp.join("data")))
}

/// In simple mode a failing run is shown, and a done after it ends the attempt with the failures
/// listed, not held.
#[test]
fn simple_ends_a_done_on_the_failures_it_saw() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done.");
    let Some(mut outcome) = simple(&fake, tmp.path(), &[]) else {
        return;
    };
    let fed = fake.fed();
    assert!(fed.contains("0 passed, 1 failed"), "{fed}");
    assert!(!fed.contains("could not compare"), "{fed}");
    let p = &outcome.packet;
    assert_eq!(p.stopped.as_deref(), Some("the model said it is done"));
    match &p.verdict {
        Verdict::Partial(left) => assert!(left.iter().any(|t| t.contains("test_add")), "{left:?}"),
        v => panic!("{v:?}"),
    }
    assert_eq!(outcome.steps, 2);
    outcome.task.discard().unwrap();
}

/// In simple mode a passing run then a done is verified; a done with no run since the last
/// change is not.
#[test]
fn simple_verifies_a_passing_run_since_the_last_change() {
    for run_after in [true, false] {
        let fake = Fake::qwen3(32768);
        let tmp = tempfile::tempdir().unwrap();
        call(&fake, "run_tests", json!({}), false);
        edit(&fake, "a - b", "a + b");
        if run_after {
            call(&fake, "run_tests", json!({}), false);
        }
        fake.say("Done.");
        let Some(mut outcome) = simple(&fake, tmp.path(), &[]) else {
            return;
        };
        let p = &outcome.packet;
        assert_eq!(p.stopped.as_deref(), Some("the model said it is done"));
        let want = if run_after {
            Verdict::Verified
        } else {
            Verdict::Unverified
        };
        assert_eq!(p.verdict, want, "{}", fake.fed());
        outcome.task.discard().unwrap();
    }
}

/// New reads cost no call only while their results since the last change fit the room's read
/// tokens (two contexts): a run that only reads at 16k ends at its tool_calls budget.
#[test]
fn reads_past_the_rooms_read_tokens_count() {
    let fake = Fake::qwen3(16384);
    let tmp = tempfile::tempdir().unwrap();
    let long = "y".repeat(70);
    let big: String = (1..=8000).map(|n| format!("{n} {long}\n")).collect();
    let repo = toy(tmp.path(), &[("big.txt", big.as_str())]);
    for n in 0..60 {
        call(
            &fake,
            "view",
            json!({"path": "big.txt", "start": 1 + n * 100}),
            false,
        );
    }
    fake.say("Done.");
    let mut task = task(&repo);
    task.budget.tool_calls = 1;
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task, &data)) else {
        return;
    };
    let stopped = outcome.packet.stopped.as_deref();
    assert_eq!(stopped, Some("the tool_calls budget (1) ran out"));
    assert!(outcome.steps < 60, "{}", outcome.steps);
    outcome.task.discard().unwrap();
}

/// A change that passes the checks covering it is told so once, as a fact, where it did not
/// before: by its tests, run again in the same state without a second telling, and without
/// tests by its build check as it is made.
// Build checks are POSIX shell, so task::checks gives Windows none.
#[cfg(not(windows))]
#[test]
fn a_change_that_passes_its_checks_is_told_once() {
    let passes = "The change passes its checks: if the request is met, your next reply is done.";
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let repo = toy(tmp.path(), &[]);
    edit(&fake, "a - b", "a + b");
    call(&fake, "run_tests", json!({}), false);
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done.");
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task(&repo), &data)) else {
        return;
    };
    assert_eq!(fake.fed().matches(passes).count(), 1, "{}", fake.fed());
    outcome.task.discard().unwrap();
    let fake = Fake::qwen3(32768);
    let (folder, _) = folder(tmp.path(), &[("f.py", "def f():\n    return 1\n")]);
    let args = json!({"path": "f.py", "old_text": "return 1", "new_text": "return 2"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("ok");
    let checks = checks(&["f.py".into()]);
    if checked_run(&fake, &folder, &checks).is_some() {
        assert_eq!(fake.fed().matches(passes).count(), 1, "{}", fake.fed());
    }
}

/// The checks run once on a folder state: a second done with nothing changed since reads what
/// they said there, it does not run them again.
#[test]
fn a_check_already_run_on_the_folder_as_it_is_is_not_run_again() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, _) = folder(tmp.path(), &[("f.txt", "a\n")]);
    let args = json!({"path": "f.txt", "old_text": "a", "new_text": "b"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let Some((_, progress, _)) = checked_run(&fake, &folder, &["true".into()]) else {
        return;
    };
    let ran = progress.iter().filter(|p| *p == "Checking: true").count();
    assert_eq!(ran, 1, "{progress:?}");
}

/// A check killed at a time limit says nothing about the change: the user is told it was not
/// done, and the done goes on as without it.
// The fake model's commands are POSIX shell, which cmd.exe does not run.
#[cfg(not(windows))]
#[test]
fn a_timed_out_check_does_not_block_the_done() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, _) = folder(tmp.path(), &[("f.txt", "a\n")]);
    let args = json!({"path": "f.txt", "old_text": "a", "new_text": "b"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let slow = ["timeout 1 sleep 5".to_string()];
    let Some((outcome, _, notices)) = checked_run(&fake, &folder, &slow) else {
        return;
    };
    let fed = fake.fed();
    assert!(
        !fed.contains("a build check of the changed files fails"),
        "{fed}"
    );
    let told = "the check `timeout 1 sleep 5` was not done: exit code 124";
    assert!(notices.iter().any(|n| n.starts_with(told)), "{notices:?}");
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{notices:?}"
    );
}

/// A changed module whose import fails on the base too (a library not installed) says nothing
/// of the change: the done is not blocked, and the user is told the check was dropped.
// Build checks are POSIX shell, so `task::checks` gives Windows none.
#[cfg(not(windows))]
#[test]
fn a_check_that_fails_on_the_base_is_dropped() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let text = "import not_installed_here\n\ndef f():\n    return 1\n";
    let (folder, _) = folder(tmp.path(), &[("f.py", text)]);
    let args = json!({"path": "f.py", "old_text": "return 1", "new_text": "return 2"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let checks = checks(&["f.py".into()]);
    let Some((outcome, _, notices)) = checked_run(&fake, &folder, &checks) else {
        return;
    };
    let fed = fake.fed();
    assert!(
        !fed.contains("a build check of the changed files fails"),
        "{fed}"
    );
    assert!(
        notices
            .iter()
            .any(|n| n.ends_with("fails on the base too: dropped")),
        "{notices:?}"
    );
    assert_eq!(
        fs::read_to_string(folder.join("f.py")).unwrap(),
        text.replace("1", "2")
    );
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{notices:?}"
    );
}

/// A module whose import the base fails too hides no other: the import that the change broke
/// still sends the done back, and the base's own failure alone is dropped.
// Build checks are POSIX shell, so `task::checks` gives Windows none.
#[cfg(not(windows))]
#[test]
fn a_module_the_base_fails_hides_no_other() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let a = "import not_installed_here\n\ndef f():\n    return 1\n";
    let b = "def g():\n    return 1\n";
    let (folder, _) = folder(tmp.path(), &[("a.py", a), ("b.py", b)]);
    let args = json!({"path": "a.py", "old_text": "return 1", "new_text": "return 2"});
    call(&fake, "edit", args, false);
    let broken = "import also_missing\n\ndef g():";
    let args = json!({"path": "b.py", "old_text": "def g():", "new_text": broken});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    let args = json!({"path": "b.py", "old_text": broken, "new_text": "def g():"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let checks = checks(&["a.py".into(), "b.py".into()]);
    let Some((outcome, _, notices)) = checked_run(&fake, &folder, &checks) else {
        return;
    };
    let fed = fake.fed();
    let told = "a build check of the changed files fails, so it is not done yet";
    assert_eq!(fed.matches(told).count(), 1, "{fed}");
    assert!(fed.contains("import failed: b.py"), "{fed}");
    assert!(!fed.contains("import failed: a.py"), "{fed}");
    assert!(
        notices
            .iter()
            .any(|n| n.ends_with("fails on the base too: dropped")),
        "{notices:?}"
    );
    assert_eq!(fs::read_to_string(folder.join("b.py")).unwrap(), b);
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{notices:?}"
    );
}

/// A view longer than the room from the first prompt to the compaction is cut, by the view or
/// by the compaction it causes, its start kept with where to view from for the rest, so it is
/// never folded to a line and asked for again. At 16k the fake's first prompt takes most of
/// the room, as an 8k model's map and tools do.
#[test]
fn a_view_past_the_room_is_cut_with_where_to_view_from() {
    let notes = notes(200);
    let Some((fake, notices, _)) = viewed(16384, &[("big.txt", notes.as_str())], &["big.txt"])
    else {
        return;
    };
    let fed = fake.fed();
    assert!(
        fed.contains("1: line 1 of the notes, which the model reads and leaves alone"),
        "{fed}"
    );
    assert!(fed.contains("; view from start "), "{fed}");
    // Cut, not whole: the last line is never in a prompt.
    assert!(!fed.contains("200: line 200 of the notes"), "{fed}");
    assert_eq!(fake.unsaid(), 0, "{notices:?}");
}

/// A plan made after a done the review sent back is no done: the reviewer reads the change
/// only at the next done, after the plan wrote it, never the files as they were before.
#[test]
fn a_plan_after_a_sent_back_done_is_not_reviewed_as_a_done() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let task = notes_task(&fake, tmp.path(), &[("notes.txt", "draft\n")]);
    edit_in(&fake, "notes.txt", "draft", "final");
    fake.say("Done.");
    plan(
        &fake,
        r##"{"contract": "the notes start with a title", "files": [{"path": "notes.txt", "about": "put the title # Notes first"}]}"##,
    );
    fake.say("# Notes\nfinal\n");
    fake.say("Done: the notes have a title.");
    fake.say("ok");
    let Some((outcome, progress, notices)) = watched(&fake, &task) else {
        return;
    };
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{progress:?} {notices:?}"
    );
    let checking = progress
        .iter()
        .filter(|p| p.starts_with("Reviewer: checking"));
    assert_eq!(checking.count(), 1, "{progress:?}");
    let notes = fs::read_to_string(tmp.path().join("notes/notes.txt")).unwrap();
    assert_eq!(notes, "# Notes\nfinal\n");
    assert_eq!(fake.unsaid(), 0);
}

/// `make add add` in place in `folder`, its tests `cmd`: the outcome, the status lines and the
/// notices, or None where the sandbox is unavailable.
fn tested_in_place(
    fake: &Fake,
    folder: &Path,
    cmd: &str,
) -> Option<(Outcome, Vec<String>, Vec<String>)> {
    let overrides = Overrides {
        test_cmd: Some(cmd.into()),
        allowed_paths: None,
        check_cmds: vec![],
        budget: BUDGET,
    };
    let task = CodeTask::in_folder("make add add", folder, &fake.data(), overrides).unwrap();
    watched(fake, &task)
}

/// A folder of the calculator, its test file `test` and more `files`, past `SMALL_SUITE` test
/// files so no test runs whole, committed.
fn big_suite(tmp: &Path, test: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
    let mut all: Vec<(String, String)> = (0..40)
        .map(|n| (format!("test_x{n}.py"), "import unittest\n".into()))
        .collect();
    all.push(("test_calc.py".into(), test.into()));
    all.extend(files.iter().map(|(a, b)| (a.to_string(), b.to_string())));
    let all: Vec<(&str, &str)> = all.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let (folder, _) = folder(tmp, &all);
    common::commit(&folder);
    folder
}

/// The unittest file of the calculator's add, with a test that fails before and after any
/// change (it needs the network).
const ADD_TESTS: &str = "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n    def test_net(self):\n        \
self.fail('needs the network')\n";

/// The stop of a done whose failures could not be compared with the base.
const UNCOMPARED_DONE: &str = "the model said it is done; tests fail that could not be compared \
                               with the base, so it is unverified";

/// The outcome a user sees of a done the base could not be compared with: stopped unverified,
/// never "the tests agree" nor "passes its checks", and the packet unverified.
fn unverified(outcome: &Outcome, fed: &str, notices: &[String]) {
    unverified_as(UNCOMPARED_DONE, outcome, fed, notices);
}

/// `unverified`, stopped as `stop` says.
fn unverified_as(stop: &str, outcome: &Outcome, fed: &str, notices: &[String]) {
    let p = &outcome.packet;
    assert_eq!(p.stopped.as_deref(), Some(stop), "{notices:?}\n{fed}");
    assert_eq!(p.verdict, Verdict::Unverified);
    assert!(!fed.contains("The change passes its checks"), "{fed}");
    assert!(
        !notices.iter().any(|n| n.contains("tests agree")),
        "{notices:?}"
    );
}

/// A base run that crashes in its checkout (it needs a local settings file git ignores) cannot
/// be compared: a test that failed before the change neither holds the done nor passes it.
#[cfg(not(windows))]
#[test]
fn a_base_that_cannot_run_neither_holds_nor_passes_the_done() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let settings = [
        ("calc.py", "def add(a, b):\n    return a - b\n"),
        (".gitignore", "local_settings.py\n"),
        ("local_settings.py", "DEBUG = True\n"),
    ];
    let folder = big_suite(tmp.path(), ADD_TESTS, &settings);
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let cmd = "python3 -c 'import local_settings' && python3 -m unittest {tests}";
    let Some((outcome, _, notices)) = tested_in_place(&fake, &folder, cmd) else {
        return;
    };
    let fed = fake.fed();
    assert!(!fed.contains("the tests do not pass yet"), "{fed}");
    unverified(&outcome, &fed, &notices);
    let shown = outcome.packet.render();
    assert!(
        shown.contains("tests after: 1 passed, 1 failed: test_net"),
        "{shown}"
    );
}

/// A virtualenv at `venv` whose site-packages holds an editable install's .pth file naming
/// `folder/src`, as `pip install -e .` leaves it: its python3.
fn editable_venv(venv: &Path, folder: &Path) -> std::path::PathBuf {
    let made = std::process::Command::new("python3")
        .args(["-m", "venv", "--without-pip"])
        .arg(venv)
        .status()
        .unwrap();
    assert!(made.success());
    let lib = fs::read_dir(venv.join("lib"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let pth = lib.path().join("site-packages/__editable__.calc-0.pth");
    fs::write(pth, format!("{}\n", folder.join("src").display())).unwrap();
    venv.join("bin/python3")
}

/// A project installed editable: the base run imports the base's code from its checkout, not
/// the folder's through the .pth, so a change that breaks add is its own failure and holds the
/// done; the fix's done goes through.
#[cfg(not(windows))]
#[test]
fn an_editable_install_never_makes_the_changes_failure_the_bases() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let test = "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
                def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n";
    let (folder, _) = folder(tmp.path(), &[]);
    fs::create_dir_all(folder.join("src/calc")).unwrap();
    fs::create_dir_all(folder.join("tests")).unwrap();
    let add = "def add(a, b):\n    return a + b\n";
    fs::write(folder.join("src/calc/__init__.py"), add).unwrap();
    fs::write(folder.join("tests/__init__.py"), "").unwrap();
    fs::write(folder.join("tests/test_calc.py"), test).unwrap();
    common::commit(&folder);
    let python = editable_venv(&tmp.path().join("venv"), &folder);
    edit_in(&fake, "src/calc/__init__.py", "a + b", "a - b");
    fake.say("Done.");
    edit_in(&fake, "src/calc/__init__.py", "a - b", "b + a");
    fake.say("Done.");
    let cmd = format!("{} -m unittest {{tests}}", python.display());
    let Some((outcome, _, notices)) = tested_in_place(&fake, &folder, &cmd) else {
        return;
    };
    let fed = fake.fed();
    assert_eq!(fed.matches("the tests do not pass yet").count(), 1, "{fed}");
    assert!(!fed.contains("as before your change"), "{fed}");
    let agree = "the model said it is done and the tests agree";
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(agree),
        "{notices:?}"
    );
    let now = fs::read_to_string(folder.join("src/calc/__init__.py")).unwrap();
    assert_eq!(now, add.replace("a + b", "b + a"));
}

/// The same project with the base's own `src/` hidden from its run (an empty one): the base
/// run loads the folder's code through the .pth, which its output names, so the change's
/// failure could not be compared: the done goes through unverified, never as agreeing.
#[cfg(not(windows))]
#[test]
fn a_base_run_that_loads_the_folders_code_is_not_compared() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let test = "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
                def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n";
    let (folder, _) = folder(tmp.path(), &[]);
    fs::create_dir_all(folder.join("lib/calc")).unwrap();
    fs::create_dir_all(folder.join("tests")).unwrap();
    fs::write(
        folder.join("lib/calc/__init__.py"),
        "def add(a, b):\n    return a + b\n",
    )
    .unwrap();
    fs::write(folder.join("tests/__init__.py"), "").unwrap();
    fs::write(folder.join("tests/test_calc.py"), test).unwrap();
    common::commit(&folder);
    // The .pth names lib/, which the base run does not put first.
    let python = editable_venv(&tmp.path().join("venv"), &folder);
    let pth = fs::read_dir(tmp.path().join("venv/lib"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let pth = pth.path().join("site-packages/__editable__.calc-0.pth");
    fs::write(pth, format!("{}\n", folder.join("lib").display())).unwrap();
    edit_in(&fake, "lib/calc/__init__.py", "a + b", "a - b");
    fake.say("Done.");
    let cmd = format!("{} -m unittest {{tests}}", python.display());
    let Some((outcome, _, notices)) = tested_in_place(&fake, &folder, &cmd) else {
        return;
    };
    unverified(&outcome, &fake.fed(), &notices);
}

/// A base checkout that fails (here a path that fits under the folder but is too long under
/// the checkout's longer tmp path) ends neither the run nor the done's check: the failures
/// could not be compared, so the done goes through unverified.
#[cfg(target_os = "linux")]
#[test]
fn a_base_checkout_that_fails_is_no_comparison_not_an_error() {
    let fake = Fake::qwen3(32768);
    // A short folder path: /tmp/vXXX, 9 bytes, against the checkout's 16 or more.
    let tmp = tempfile::Builder::new()
        .prefix("v")
        .rand_bytes(3)
        .tempdir_in("/tmp")
        .unwrap();
    let folder = tmp.path();
    assert_eq!(folder.as_os_str().len(), 9);
    let test = "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
                def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n";
    fs::write(folder.join("test_calc.py"), test).unwrap();
    fs::write(folder.join("calc.py"), "def add(a, b):\n    return a + b\n").unwrap();
    // 4082 bytes: under PATH_MAX after /tmp/vXXX/, past it after /tmp/.tmpXXXXXX/.
    let deep = vec!["a".repeat(250); 16].join("/");
    fs::create_dir_all(folder.join(&deep)).unwrap();
    fs::write(folder.join(&deep).join("b".repeat(4082 - 16 * 251)), "x\n").unwrap();
    common::commit(folder);
    edit(&fake, "a + b", "a - b");
    fake.say("Done.");
    let cmd = "python3 -m unittest {tests}";
    let Some((outcome, _, notices)) = tested_in_place(&fake, folder, cmd) else {
        return;
    };
    unverified(&outcome, &fake.fed(), &notices);
}

/// A changed script that parses its arguments as it loads (argparse, no main guard) exits at
/// its import, on the base too: it neither ends the build check nor reads the other files as
/// its arguments, so the module the change broke after it still sends the done back.
// Build checks are POSIX shell, so `task::checks` gives Windows none.
#[cfg(not(windows))]
#[test]
fn a_script_that_exits_at_its_import_hides_no_other_module() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let cli = "import argparse\n\np = argparse.ArgumentParser()\np.add_argument('name')\n\
               args = p.parse_args()\nprint(args.name)\n";
    let b = "def g():\n    return 1\n";
    let (folder, _) = folder(tmp.path(), &[("a_cli.py", cli), ("b.py", b)]);
    let args =
        json!({"path": "a_cli.py", "old_text": "print(args.name)", "new_text": "print(args)"});
    call(&fake, "edit", args, false);
    let broken = "g = missing_name\n\ndef g():";
    let args = json!({"path": "b.py", "old_text": "def g():", "new_text": broken});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    let args = json!({"path": "b.py", "old_text": broken, "new_text": "def g():"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let checks = checks(&["a_cli.py".into(), "b.py".into()]);
    let Some((outcome, _, notices)) = checked_run(&fake, &folder, &checks) else {
        return;
    };
    let fed = fake.fed();
    let told = "a build check of the changed files fails, so it is not done yet";
    assert_eq!(fed.matches(told).count(), 1, "{notices:?}\n{fed}");
    assert!(fed.contains("import failed: b.py"), "{fed}");
    assert_eq!(fs::read_to_string(folder.join("b.py")).unwrap(), b);
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{notices:?}"
    );
}

/// A repository that keeps a file broken on purpose (Django's syntax-error test fixture) still
/// compiles each new file: the fixture is never part of that check, so a new module that does
/// not compile sends the done back.
// Build checks are POSIX shell, so task::checks gives Windows none.
#[cfg(not(windows))]
#[test]
fn a_broken_fixture_never_drops_the_new_files_compile() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let fixture = "tests/test_runner_apps/tagged/tests_syntax_error.py";
    let (folder, data) = folder(tmp.path(), &[("a.py", "def f():\n    return 1\n")]);
    fs::create_dir_all(folder.join("tests/test_runner_apps/tagged")).unwrap();
    fs::write(folder.join(fixture), "1syntax_error\n").unwrap();
    let broken = json!({"command": "printf 'def g(:\\n    return 2\\n' > new.py"});
    call(&fake, "bash", broken, false);
    fake.say("Done.");
    let fixed = json!({"command": "printf 'def g():\\n    return 2\\n' > new.py"});
    call(&fake, "bash", fixed, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let task = untested("make f right", &folder, &data);
    let Some((outcome, progress, notices)) = watched(&fake, &task) else {
        return;
    };
    if sandboxed(&fake, &notices).is_none() {
        return;
    }
    let fed = fake.fed();
    assert!(fed.contains("a compile of each new .py file"), "{fed}");
    let told = "a build check of the changed files fails, so it is not done yet";
    assert_eq!(fed.matches(told).count(), 1, "{notices:?}\n{fed}");
    assert!(
        progress.iter().any(|p| p.ends_with("py_compile 'new.py'")),
        "{progress:?}"
    );
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{notices:?}"
    );
}

/// Changed modules whose imports the base fails too with long tracebacks put the line of the
/// module the change broke, between them, in the middle the output's cut leaves out: the names
/// are read before the cut, so that module still sends the done back.
// Build checks are POSIX shell, so `task::checks` gives Windows none.
#[cfg(not(windows))]
#[test]
fn a_failed_import_in_the_cut_middle_is_still_named() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let long = "raise RuntimeError(' '.join(str(i) for i in range(4000)))\n\ndef f():\n    \
                return 1\n";
    let b = "def g():\n    return 1\n";
    let (folder, _) = folder(tmp.path(), &[("a.py", long), ("b.py", b), ("c.py", long)]);
    for path in ["a.py", "c.py"] {
        let args = json!({"path": path, "old_text": "return 1", "new_text": "return 2"});
        call(&fake, "edit", args, false);
    }
    let broken = "g = missing_name\n\ndef g():";
    let args = json!({"path": "b.py", "old_text": "def g():", "new_text": broken});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    let args = json!({"path": "b.py", "old_text": broken, "new_text": "def g():"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let checks = checks(&["a.py".into(), "b.py".into(), "c.py".into()]);
    let Some((outcome, _, notices)) = checked_run(&fake, &folder, &checks) else {
        return;
    };
    let fed = fake.fed();
    let told = "a build check of the changed files fails, so it is not done yet";
    assert_eq!(fed.matches(told).count(), 1, "{notices:?}");
    assert!(fed.contains("import failed: b.py"), "{notices:?}");
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{notices:?}"
    );
}

/// A changed module that fails only after one the base fails too passes when the check runs
/// again on it alone: that rerun is what the user is told, and it neither holds the done nor
/// passes it.
// Build checks are POSIX shell, so `task::checks` gives Windows none.
#[cfg(not(windows))]
#[test]
fn a_module_that_passes_alone_is_told_so_and_holds_nothing() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let a = "import builtins\n\nbuiltins.SERVER_DOWN = 1\nraise RuntimeError('a needs a server')\n\
             \ndef f():\n    return 1\n";
    let b = "def g():\n    return 1\n";
    let (folder, _) = folder(tmp.path(), &[("a.py", a), ("b.py", b)]);
    let args = json!({"path": "a.py", "old_text": "return 1", "new_text": "return 2"});
    call(&fake, "edit", args, false);
    let after = "import builtins\n\nif getattr(builtins, 'SERVER_DOWN', 0):\n    \
                 raise RuntimeError('b after a')\n\ndef g():";
    let args = json!({"path": "b.py", "old_text": "def g():", "new_text": after});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("Done.");
    fake.say("ok");
    let checks = checks(&["a.py".into(), "b.py".into()]);
    let Some((outcome, _, notices)) = checked_run(&fake, &folder, &checks) else {
        return;
    };
    let fed = fake.fed();
    assert!(
        !fed.contains("a build check of the changed files fails"),
        "{fed}"
    );
    assert!(
        notices.iter().any(|n| n.contains("passes alone")),
        "{notices:?}"
    );
    assert!(!fed.contains("The change passes its checks"), "{fed}");
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(NO_TESTS),
        "{notices:?}"
    );
}

/// A done no test near the change could check gets the reviewer, as one without tests does:
/// its doubt goes with the done, which ends unverified.
#[test]
fn a_done_no_test_near_could_check_is_reviewed() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let tests: Vec<(String, String)> = (0..40)
        .map(|n| (format!("test_x{n}.py"), "import unittest\n".to_string()))
        .collect();
    let mut more: Vec<(&str, &str)> = tests
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    more.push(("util.py", "def u():\n    return 1\n"));
    let repo = toy(tmp.path(), &more);
    let args = json!({"path": "util.py", "old_text": "return 1", "new_text": "return 2"});
    call(&fake, "edit", args, false);
    fake.say("Done.");
    fake.say("not done: u should return 3");
    let mut task = task(&repo);
    task.test_cmd = Some(format!("{UNITTEST} {{tests}}"));
    let Some((mut outcome, progress, notices)) = watched(&fake, &task) else {
        return;
    };
    let fed = fake.fed();
    assert!(
        !fed.contains("a reviewer read the change"),
        "{progress:?}\n{fed}"
    );
    let unverified = "the model said it is done; no tests ran near the change";
    let doubt = format!("{unverified}; a reviewer doubts it: u should return 3");
    assert_eq!(outcome.packet.stopped.as_deref(), Some(doubt.as_str()));
    assert_eq!(outcome.packet.verdict, Verdict::Unverified);
    assert_eq!(
        notices.iter().filter(|n| *n == unverified).count(),
        1,
        "{notices:?}"
    );
    assert_eq!(fake.unsaid(), 0);
    outcome.task.discard().unwrap();
}

/// The calculator whose add subtracts, beside a mul that works: test_add fails on the base.
const BROKEN_ADD: &str = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n";

/// "Fix the failing test_add", and a change that leaves it failing as it did on the base: the
/// first done is sent back naming it, the second ends the attempt, never as a pass.
#[cfg(not(windows))]
#[test]
fn a_failure_the_base_shares_is_never_a_pass() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let folder = big_suite(tmp.path(), ADD_TESTS, &[("calc.py", BROKEN_ADD)]);
    edit(&fake, "a * b", "b * a");
    fake.say("Done.");
    fake.say("Done.");
    let cmd = "python3 -m unittest {tests}";
    let Some((outcome, _, notices)) = tested_in_place(&fake, &folder, cmd) else {
        return;
    };
    let fed = fake.fed();
    let before = "still failing, as before your change: ";
    assert!(fed.contains(before) && fed.contains("test_add"), "{fed}");
    assert!(!fed.contains("The change passes its checks"), "{fed}");
    assert!(
        !notices.iter().any(|n| n.contains("tests agree")),
        "{notices:?}"
    );
    let p = &outcome.packet;
    let except = "passes except 2 failures that were there before the change";
    assert!(
        p.stopped.as_deref().is_some_and(|s| s.contains(except)),
        "{:?} {notices:?}",
        p.stopped
    );
    assert_ne!(p.verdict, Verdict::Verified);
    assert!(p.verdict.to_string().contains(except), "{}", p.verdict);
    assert_eq!(fake.unsaid(), 0, "{notices:?}");
}

/// The same test fixed: none fails, the only verified pass.
#[cfg(not(windows))]
#[test]
fn a_fix_of_the_failing_test_is_the_verified_pass() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let test = "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
                def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n";
    let folder = big_suite(tmp.path(), test, &[("calc.py", BROKEN_ADD)]);
    edit(&fake, "a - b", "a + b");
    fake.say("Done.");
    let cmd = "python3 -m unittest {tests}";
    let Some((outcome, _, notices)) = tested_in_place(&fake, &folder, cmd) else {
        return;
    };
    let p = &outcome.packet;
    let agree = "the model said it is done and the tests agree";
    assert_eq!(p.stopped.as_deref(), Some(agree), "{notices:?}");
    assert_eq!(p.verdict, Verdict::Verified);
    assert_eq!(fake.unsaid(), 0, "{notices:?}");
}

/// The calculator installed editable from `folder/<dir>` by a virtualenv beside it, committed:
/// "make add add" in worktrees, its tests run by that virtualenv's python.
fn editable_task(tmp: &Path, dir: &str) -> (CodeTask, std::path::PathBuf) {
    let test = "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
                def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n";
    let (folder, _) = folder(tmp, &[]);
    fs::create_dir_all(folder.join(dir).join("calc")).unwrap();
    fs::create_dir_all(folder.join("tests")).unwrap();
    let add = "def add(a, b):\n    return a + b\n";
    fs::write(folder.join(dir).join("calc/__init__.py"), add).unwrap();
    fs::write(folder.join("tests/__init__.py"), "").unwrap();
    fs::write(folder.join("tests/test_calc.py"), test).unwrap();
    common::commit(&folder);
    let python = editable_venv(&tmp.join("venv"), &folder);
    let pth = fs::read_dir(tmp.join("venv/lib"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let pth = pth.path().join("site-packages/__editable__.calc-0.pth");
    fs::write(pth, format!("{}\n", folder.join(dir).display())).unwrap();
    let overrides = Overrides {
        test_cmd: Some(format!("{} -m unittest {{tests}}", python.display())),
        allowed_paths: None,
        check_cmds: vec![],
        budget: BUDGET,
    };
    (
        CodeTask::new("make add add", &folder, overrides).unwrap(),
        folder,
    )
}

/// In worktrees, an editable install's .pth names the user's checkout: the run after the change
/// imports the worktree's src/ first, so a change that breaks add fails there and holds the
/// done, and the files the run writes stay out of the diff.
#[cfg(not(windows))]
#[test]
fn a_worktrees_run_imports_the_worktree_not_the_editable_checkout() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (task, _) = editable_task(tmp.path(), "src");
    edit_in(&fake, "src/calc/__init__.py", "a + b", "a - b");
    fake.say("Done.");
    edit_in(&fake, "src/calc/__init__.py", "a - b", "b + a");
    fake.say("Done.");
    let Some((outcome, _, notices)) = watched(&fake, &task) else {
        return;
    };
    let fed = fake.fed();
    assert_eq!(fed.matches("the tests do not pass yet").count(), 1, "{fed}");
    let p = &outcome.packet;
    let agree = "the model said it is done and the tests agree";
    assert_eq!(p.stopped.as_deref(), Some(agree), "{notices:?}");
    assert!(!p.diff.contains(".larch-site"), "{}", p.diff);
}

/// The same with the package under lib/, which the worktree's run does not put first: it loads
/// the user's checkout, so it could not be compared, never a pass, and its stop never says a
/// test fails, as none does.
#[cfg(not(windows))]
#[test]
fn a_worktrees_run_that_loads_the_users_checkout_is_not_compared() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (task, _) = editable_task(tmp.path(), "lib");
    edit_in(&fake, "lib/calc/__init__.py", "a + b", "a - b");
    fake.say("Done.");
    let Some((outcome, _, notices)) = watched(&fake, &task) else {
        return;
    };
    let stop = "the model said it is done; the tests pass, but they may have run the user's \
                installed copy, not the change, so it is unverified";
    unverified_as(stop, &outcome, &fake.fed(), &notices);
}

/// The calculator's src/ package, its add `add`, importing `dep`, a dependency installed in the
/// project's own `.venv/` inside `folder`, git-ignored, which also holds the editable install's
/// .pth naming `folder/src`, committed: that virtualenv's python3.
fn venv_inside(folder: &Path, add: &str) -> std::path::PathBuf {
    let test = "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    \
                def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n";
    fs::create_dir_all(folder.join("src/calc")).unwrap();
    fs::create_dir_all(folder.join("tests")).unwrap();
    let calc = format!("import dep\n\ndef add(a, b):\n    return {add}\n");
    fs::write(folder.join("src/calc/__init__.py"), calc).unwrap();
    fs::write(folder.join("tests/__init__.py"), "").unwrap();
    fs::write(folder.join("tests/test_calc.py"), test).unwrap();
    fs::write(folder.join(".gitignore"), ".venv/\n").unwrap();
    common::commit(folder);
    let venv = folder.join(".venv");
    let python = editable_venv(&venv, folder);
    let lib = fs::read_dir(venv.join("lib"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    fs::write(lib.path().join("site-packages/dep.py"), "X = 1\n").unwrap();
    python
}

/// Overrides whose tests are `python -m unittest {tests}`, run by `python`.
fn venv_overrides(python: &Path) -> Overrides {
    Overrides {
        test_cmd: Some(format!("{} -m unittest {{tests}}", python.display())),
        allowed_paths: None,
        check_cmds: vec![],
        budget: BUDGET,
    }
}

/// In worktrees, a project's own `.venv/` inside the checkout is no editable install: the fix's
/// run, which loads its dependency from there, is compared and its done Verified.
#[cfg(not(windows))]
#[test]
fn a_venv_inside_the_checkout_leaves_a_worktrees_fix_verified() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, _) = folder(tmp.path(), &[]);
    let python = venv_inside(&folder, "a - b");
    edit_in(&fake, "src/calc/__init__.py", "a - b", "a + b");
    fake.say("Done.");
    let task = CodeTask::new("make add add", &folder, venv_overrides(&python)).unwrap();
    let Some((outcome, _, notices)) = watched(&fake, &task) else {
        return;
    };
    let p = &outcome.packet;
    let agree = "the model said it is done and the tests agree";
    assert_eq!(
        p.stopped.as_deref(),
        Some(agree),
        "{notices:?}\n{}",
        fake.fed()
    );
    assert_eq!(p.verdict, Verdict::Verified);
}

/// The same `.venv/` in worktrees: a change that breaks add is compared with the base, which
/// passes, so its done is sent back; the fix's done agrees.
#[cfg(not(windows))]
#[test]
fn a_venv_inside_the_checkout_never_hides_a_worktrees_break() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, _) = folder(tmp.path(), &[]);
    let python = venv_inside(&folder, "a + b");
    edit_in(&fake, "src/calc/__init__.py", "a + b", "a - b");
    fake.say("Done.");
    edit_in(&fake, "src/calc/__init__.py", "a - b", "b + a");
    fake.say("Done.");
    let task = CodeTask::new("make add add", &folder, venv_overrides(&python)).unwrap();
    let Some((outcome, _, notices)) = watched(&fake, &task) else {
        return;
    };
    let fed = fake.fed();
    assert_eq!(fed.matches("the tests do not pass yet").count(), 1, "{fed}");
    let agree = "the model said it is done and the tests agree";
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(agree),
        "{notices:?}"
    );
}

/// The same `.venv/` inside a folder worked in place: a change that breaks add is compared
/// with the base, which passes, so its done is sent back.
#[cfg(not(windows))]
#[test]
fn a_venv_inside_the_folder_never_hides_a_break_in_place() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let (folder, _) = folder(tmp.path(), &[]);
    let python = venv_inside(&folder, "a + b");
    edit_in(&fake, "src/calc/__init__.py", "a + b", "a - b");
    fake.say("Done.");
    edit_in(&fake, "src/calc/__init__.py", "a - b", "b + a");
    fake.say("Done.");
    let cmd = format!("{} -m unittest {{tests}}", python.display());
    let Some((outcome, _, notices)) = tested_in_place(&fake, &folder, &cmd) else {
        return;
    };
    let fed = fake.fed();
    assert_eq!(fed.matches("the tests do not pass yet").count(), 1, "{fed}");
    let agree = "the model said it is done and the tests agree";
    assert_eq!(
        outcome.packet.stopped.as_deref(),
        Some(agree),
        "{notices:?}"
    );
}

/// A whole-suite command (no `{tests}`) with a test that failed before the change (it needs the
/// network): compared with the suite's run on the base, so it never holds the done, and the
/// done it lets through is never a pass.
#[cfg(not(windows))]
#[test]
fn a_whole_suites_old_failure_neither_holds_nor_passes_the_done() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let calc = "def add(a, b):\n    return a + b\n";
    let (folder, _) = folder(
        tmp.path(),
        &[("calc.py", calc), ("test_calc.py", ADD_TESTS)],
    );
    common::commit(&folder);
    edit(&fake, "a + b", "b + a");
    fake.say("Done.");
    fake.say("Done.");
    let Some((outcome, _, notices)) =
        tested_in_place(&fake, &folder, "python3 -m unittest test_calc")
    else {
        return;
    };
    let fed = fake.fed();
    assert!(!fed.contains("the tests do not pass yet"), "{fed}");
    assert!(!fed.contains("The change passes its checks"), "{fed}");
    let p = &outcome.packet;
    let except = "passes except 1 failures that were there before the change";
    assert!(
        p.stopped.as_deref().is_some_and(|s| s.contains(except)),
        "{:?} {notices:?}",
        p.stopped
    );
    assert_ne!(p.verdict, Verdict::Verified);
    assert_eq!(fake.unsaid(), 0, "{fed}");
}

/// Full mode: a done reruns every test the model named in a run that ran, so a passing unrelated
/// test cannot stand in for a failing one the change broke.
#[test]
fn a_done_reruns_every_test_the_model_named() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let ok =
        "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        pass\n";
    let check = "import unittest\nfrom util import u\n\nclass T(unittest.TestCase):\n    \
                 def test_u(self):\n        self.assertEqual(u(), 1)\n";
    let files = [
        ("util.py", "def u():\n    return 1\n"),
        ("test_ok.py", ok),
        ("test_u.py", check),
    ];
    let repo = toy(tmp.path(), &files);
    let args = json!({"path": "util.py", "old_text": "return 1", "new_text": "return 3"});
    call(&fake, "edit", args, false);
    call(&fake, "run_tests", json!({"tests": ["test_ok.py"]}), false);
    call(&fake, "run_tests", json!({"tests": ["test_u.py"]}), false);
    for _ in 0..3 {
        fake.say("Done.");
    }
    let mut task = task(&repo);
    task.test_cmd = Some(format!("{UNITTEST} {{tests}}"));
    let data = tmp.path().join("data");
    let Some(mut outcome) = ran(code().run(&fake.larch, &task, &data)) else {
        return;
    };
    let p = &outcome.packet;
    let agree = "the model said it is done and the tests agree";
    assert_ne!(p.stopped.as_deref(), Some(agree), "{}", fake.fed());
    assert_ne!(p.verdict, Verdict::Verified);
    outcome.task.discard().unwrap();
}

/// Simple mode is the loop and its tools alone: no tests in the background, and no telling the
/// model its change passes.
#[test]
fn simple_neither_runs_tests_beside_nor_says_a_change_passes() {
    let fake = Fake::qwen3(32768);
    let tmp = tempfile::tempdir().unwrap();
    let names: Vec<String> = (0..20).map(|n| format!("f{n}.py")).collect();
    let more: Vec<(&str, &str)> = names.iter().map(|n| (n.as_str(), "x = 1\n")).collect();
    edit(&fake, "a - b", "a + b");
    for n in &names {
        call(&fake, "view", json!({"path": n}), false);
    }
    call(&fake, "run_tests", json!({}), false);
    fake.say("Done.");
    fake.pace(std::time::Duration::from_millis(30));
    let Some(mut outcome) = simple(&fake, tmp.path(), &more) else {
        return;
    };
    let fed = fake.fed();
    assert!(fed.contains("1 passed, 0 failed"), "{fed}");
    assert!(
        !fed.contains("the tests after your previous change"),
        "{fed}"
    );
    assert!(!fed.contains("The change passes its checks"), "{fed}");
    assert_eq!(outcome.packet.verdict, Verdict::Verified);
    outcome.task.discard().unwrap();
}

/// In simple mode a run_tests that never ran (a refused label) keeps the last real run, and the
/// files a test run leaves behind are no change: a done after either is still verified.
#[test]
fn simple_keeps_its_last_run_past_a_refused_one_and_test_leftovers() {
    for refused in [true, false] {
        let fake = Fake::qwen3(32768);
        let tmp = tempfile::tempdir().unwrap();
        edit(&fake, "a - b", "a + b");
        call(&fake, "run_tests", json!({}), false);
        match refused {
            true => call(&fake, "run_tests", json!({"tests": ["a b"]}), false),
            false => call(
                &fake,
                "bash",
                json!({"command": "grep -n add calc.py"}),
                false,
            ),
        }
        fake.say("Done.");
        let Some(mut outcome) = simple(&fake, tmp.path(), &[]) else {
            return;
        };
        assert_eq!(outcome.packet.verdict, Verdict::Verified, "{}", fake.fed());
        outcome.task.discard().unwrap();
    }
}
