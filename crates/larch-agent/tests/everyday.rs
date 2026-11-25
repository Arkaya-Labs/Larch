//! The everyday set (bench/agent/everyday.jsonl): its rows' shape, the deterministic checks, and
//! `bench --everyday`, which runs each row through a chat with nothing attached.

mod common;

use std::{fs, path::Path};

use larch_agent::{
    bench::{self, EverydayQuestion, MustNot, Report, Route},
    config::{self, Thinking},
    parse::ToolCall,
};
use serde_json::json;

use common::{Fake, spaced};

/// The tiny set the tests run.
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/everyday.jsonl");

fn row(id: &str) -> EverydayQuestion {
    let set = bench::everyday_set(Path::new(FIXTURE)).unwrap();
    set.questions.into_iter().find(|q| q.id == id).unwrap()
}

fn called(name: &str) -> ToolCall {
    ToolCall {
        name: name.into(),
        args: json!({"question": "Lisbon weather this weekend"}),
    }
}

#[test]
fn the_fixture_loads_and_a_broken_row_fails_the_set() {
    let set = bench::everyday_set(Path::new(FIXTURE)).unwrap();
    assert_eq!(set.questions.len(), 4);
    let follow = &set.questions[3];
    assert_eq!((follow.history.len(), follow.route), (2, Route::Model));
    assert_eq!(
        set.questions[1].must_not,
        [MustNot::ClaimFiles, MustNot::InventCapability]
    );
    let good = fs::read_to_string(FIXTURE).unwrap();
    let first = good.lines().next().unwrap();
    let broken = [
        (first.replace("\"user\":", "\"usr\":"), "usr"),
        (
            first.replace("smalltalk\",\"user", "gossip\",\"user"),
            "gossip",
        ),
        (
            first.replace("\"tool\":null", "\"tool\":\"web_answer\""),
            "route none",
        ),
        (
            first.replace("everyday-smalltalk-001", "smalltalk-001"),
            "smalltalk-001",
        ),
        (format!("{first}\n{first}"), "twice"),
    ];
    for (text, says) in broken {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("everyday.jsonl");
        fs::write(&path, text).unwrap();
        let e = bench::everyday_set(&path).unwrap_err();
        assert!(format!("{e:#}").contains(says), "{says}: {e:#}");
    }
}

#[test]
fn a_model_row_passes_with_no_call_and_a_right_answer() {
    let q = row("everyday-smalltalk-001");
    let r = bench::score_everyday(&q, &[], "Doing well, thanks for asking!", false);
    assert!(r.right && r.answer_scored, "{r:?}");
    let r = bench::score_everyday(&q, &[called("web_answer")], "", false);
    assert!(!r.right && r.wrong_tool, "{r:?}");
    let r = bench::score_everyday(&q, &[], "I can't help with that.", false);
    assert!(!r.right && r.deflected, "{r:?}");
}

#[test]
fn a_file_tool_with_nothing_attached_is_a_miss_and_its_own_counter() {
    let q = row("everyday-myfiles-001");
    let calls = [called("list_dir"), called("read_file")];
    let r = bench::score_everyday(&q, &calls, "", false);
    assert_eq!((r.right, r.file_tool_calls, r.wrong_tool), (false, 2, true));
    let r = bench::score_everyday(&q, &[], "I looked at your notes: groceries.", false);
    assert!(!r.right && r.claimed_files, "{r:?}");
    let r = bench::score_everyday(
        &q,
        &[],
        "I can't see your files here; attach a folder.",
        false,
    );
    assert!(r.right && !r.claimed_files, "{r:?}");
    let r = bench::score_everyday(
        &q,
        &[],
        "Attach the folder and I'll email you a summary.",
        false,
    );
    assert!(!r.right && r.invented, "{r:?}");
}

#[test]
fn a_web_row_scores_its_answer_only_when_the_run_had_the_web() {
    let q = row("everyday-travel-001");
    let calls = [called("web_answer")];
    let offline = bench::score_everyday(&q, &calls, "", false);
    assert!(offline.right && !offline.answer_scored, "{offline:?}");
    let live = bench::score_everyday(&q, &calls, "Sunny, 24 C.", true);
    assert!(!live.right && live.answer_scored, "{live:?}");
    let live = bench::score_everyday(&q, &calls, "Lisbon: sunny, 24 C [1].", true);
    assert!(live.right, "{live:?}");
    let none = bench::score_everyday(&q, &[], "It is usually mild.", false);
    assert!(!none.right && none.no_tool, "{none:?}");
}

#[test]
fn a_question_back_on_a_plain_request_is_asking_back() {
    for (said, back) in [
        ("Which city are you in?", true),
        ("Sure. Could you tell me which city?", true),
        ("Which one? The 2012 model or the 2013?", true),
        ("Doing well, thanks for asking!", false),
        // smalltalk-003 on the Mac: a joke opens with its question and answers it.
        (
            "Why don't skeletons fight each other?  \nThey don't have the guts. 😄",
            false,
        ),
        ("What's a good sci-fi movie to watch tonight?", true),
        ("Which city are you in? 😊", true),
        (
            "Here is a draft: Dear Sam, thank you for the lovely dinner last night. Best, Alex. \
             Want it warmer?",
            false,
        ),
        (
            "# Does creatine work, and is it safe?\n\n## Effectiveness\n\nCreatine is well studied.",
            false,
        ),
    ] {
        assert_eq!(bench::asks_back(said), back, "{said}");
    }
}

/// Runs `rows` of the fixture once on `fake`, without the web, and its everyday section.
fn everyday(fake: &Fake, rows: &[&str]) -> bench::Everyday {
    let mut set = bench::everyday_set(Path::new(FIXTURE)).unwrap();
    set.questions.retain(|q| rows.contains(&q.id.as_str()));
    let routing = bench::Routing {
        arch: "llama".into(),
        tool_grammar: true,
        ..bench::Routing::default()
    };
    let mut report = Report::default();
    bench::run_everyday(
        &fake.larch,
        Thinking::default(),
        &routing,
        false,
        &set,
        false,
        &mut report,
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    report.everyday.unwrap()
}

#[test]
fn bench_everyday_runs_each_row_as_a_chat_with_nothing_attached() {
    let fake = Fake::new(32768);
    fake.say("Doing well, thanks for asking!");
    // A files question with nothing attached gets the fixed answer, with no model step.
    let e = everyday(&fake, &["everyday-smalltalk-001", "everyday-myfiles-001"]);
    // A dropped turn's last pass may still run: the next row's reply waits for it.
    fake.settle();
    fake.say(&spaced(
        "web_answer",
        json!({"question": "Lisbon weather this weekend"}),
    ));
    let web = everyday(&fake, &["everyday-travel-001"]);
    let rows: Vec<_> = (e.rows.iter().chain(&web.rows))
        .map(|r| (r.id.as_str(), r.right, r.tools.clone(), r.error.clone()))
        .collect();
    assert_eq!(
        rows,
        [
            ("everyday-smalltalk-001", true, vec![], None),
            ("everyday-myfiles-001", true, vec![], None),
            (
                "everyday-travel-001",
                true,
                vec!["web_answer".to_string()],
                None
            ),
        ]
    );
    assert!(!web.rows[0].answer_scored);
    assert!(e.rows[0].stages.unwrap().first_visible_ms.is_some());
    let files = &e.by_category["myfiles"];
    assert_eq!((files.rows, files.passed, files.file_tool_calls), (1, 1, 0));
    assert_eq!(e.by_category["smalltalk"].pass_rate, 1.0);
    let table = bench::everyday_table(&e);
    assert!(table.contains("myfiles"), "{table}");
    assert!(table.contains("file tool calls 0"), "{table}");
}

#[test]
fn with_the_web_a_turn_runs_to_its_end_and_every_call_counts() {
    let fake = Fake::new(32768);
    let mut set = bench::everyday_set(Path::new(FIXTURE)).unwrap();
    set.questions.truncate(1);
    set.repeat = 2;
    let out = tempfile::tempdir().unwrap();
    let audit = out.path().join("r-audit.jsonl");
    set.audit = Some(audit.clone());
    for _ in 0..2 {
        fake.say(&spaced("recall", json!({"query": "how the user is"})));
        fake.say("Doing well, thanks for asking!");
    }
    let routing = bench::Routing {
        arch: "llama".into(),
        tool_grammar: true,
        ..bench::Routing::default()
    };
    let mut report = Report::default();
    bench::run_everyday(
        &fake.larch,
        Thinking::default(),
        &routing,
        false,
        &set,
        true,
        &mut report,
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    assert_eq!(fake.unsaid(), 0);
    let e = report.everyday.unwrap();
    let r = &e.rows[0];
    assert_eq!(
        (r.tools.clone(), r.wrong_tool, r.run),
        (vec!["recall".to_string()], true, 1)
    );
    assert!(r.answer.contains("Doing well"), "{r:?}");
    assert_eq!(e.rows[1].run, 2);
    let repeated = e.repeated.unwrap();
    assert_eq!((repeated.runs, repeated.all_pass), (2, 0.0));
    // The chats' audit lines outlive the run's throwaway data folder.
    let kept = std::fs::read_to_string(&audit).unwrap();
    assert!(kept.contains("recall"), "{kept}");
}

/// `larch-agent bench` with `args` and a missing model: its exit code and its error.
fn bench_cli(args: &[&str]) -> (Option<i32>, String) {
    let tmp = tempfile::tempdir().unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_larch-agent"))
        .args(["bench", "--machine", "test", "-m", "missing", "--cpu"])
        .args(args)
        .arg("--json")
        .arg(tmp.path().join("r.json"))
        .arg("--data-dir")
        .arg(tmp.path())
        .output()
        .unwrap();
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
}

#[test]
fn a_broken_everyday_set_or_a_lone_live_fails_before_the_load() {
    let tmp = tempfile::tempdir().unwrap();
    let broken = tmp.path().join("everyday.jsonl");
    fs::write(&broken, r#"{"id":"everyday-gossip-001"}"#).unwrap();
    let cases = [
        (
            vec!["--everyday", broken.to_str().unwrap()],
            "prompt line 1",
        ),
        (vec!["--live"], "--live goes with --replay or --everyday"),
        (vec!["--app"], "--app opens chats"),
        (
            vec!["--capture"],
            "--capture goes with --turn or --everyday",
        ),
        (
            vec!["--capture", "--replay", "x"],
            "--capture goes with --turn or --everyday",
        ),
    ];
    for (args, says) in cases {
        let (code, stderr) = bench_cli(&args);
        assert_eq!(code, Some(1), "{args:?}: {stderr}");
        assert!(stderr.contains(says), "{args:?}: {stderr}");
        assert!(!stderr.contains("missing.gguf"), "{args:?}: {stderr}");
    }
    // The checkout's set by default, which loads before the model does.
    let (_, stderr) = bench_cli(&["--everyday", "--app"]);
    assert!(
        stderr.contains("missing.gguf") || stderr.contains("everyday.jsonl"),
        "{stderr}"
    );
}

#[test]
fn the_shipped_set_loads_and_its_my_files_rows_take_the_no_folder_answer() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/agent/everyday.jsonl"
    );
    let set = bench::everyday_set(Path::new(path)).unwrap();
    let mine: Vec<_> = (set.questions.iter())
        .filter(|q| q.category == "myfiles" && q.history.is_empty())
        .collect();
    assert!(mine.len() > 10);
    let said =
        "I can't read your files in this chat yet. Attach a folder or the files, then ask again.";
    for q in mine {
        let r = bench::score_everyday(q, &[], said, false);
        assert!(r.right, "{}: {r:?}", q.id);
        let r = bench::score_everyday(q, &[called("ask_docs")], "", false);
        assert!(!r.right, "{}: {r:?}", q.id);
    }
}

/// The shipped set's row `id`.
fn shipped(id: &str) -> EverydayQuestion {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/agent/everyday.jsonl"
    );
    let set = bench::everyday_set(Path::new(path)).unwrap();
    set.questions.into_iter().find(|q| q.id == id).unwrap()
}

/// The review's replies at 1920a81: a right answer with a short question after it, or a draft
/// that is itself a question, is no question back.
#[test]
fn a_right_answer_that_ends_in_a_question_is_not_asking_back() {
    for (id, said) in [
        (
            "everyday-homework-008",
            "World War I started in 1914 and ended in 1918. Want to know what caused it?",
        ),
        (
            "everyday-homework-001",
            "17% of 2350 is 399.5. Need anything else?",
        ),
        (
            "everyday-homework-006",
            "The capital of Australia is Canberra, not Sydney. Want a few fun facts to settle it?",
        ),
        (
            "everyday-writing-014",
            "Sorry I missed our call this morning. Could we find a time later today or tomorrow \
             to catch up?",
        ),
        (
            "everyday-writing-003",
            "Hi Mark, the heater is broken again. Can you get someone out to fix it this week?",
        ),
    ] {
        let r = bench::score_everyday(&shipped(id), &[], said, false);
        assert!(r.right && !r.asked_back, "{id}: {r:?}");
    }
    // A question with no answer before it, on a row whose checks it fails, still is one.
    let q = shipped("everyday-homework-008");
    let r = bench::score_everyday(&q, &[], "Do you mean World War I or II?", false);
    assert!(!r.right && r.asked_back, "{r:?}");
}

/// The review's replies at 1920a81: a chat answer that says what it lacks or cannot choose for
/// the user, and then answers, is no refusal and no deflection; a whole refusal still is.
#[test]
fn a_hedged_or_honest_chat_answer_is_no_refusal() {
    for (id, said) in [
        (
            "everyday-smalltalk-007",
            "I don't have feelings like people do, but I'm glad to chat about how you feel.",
        ),
        (
            "everyday-smalltalk-010",
            "I don't have a favourite colour, but I'd pick a deep blue if I had to.",
        ),
        (
            "everyday-life-005",
            "I can't tell you what to choose, but here are a few things to weigh: your savings, \
             how long a search takes in your field, and your health.",
        ),
        (
            "everyday-life-007",
            "I can't tell you what to choose, but here are a few things to weigh: pick a calm \
             time, show them your plan, and say why art matters to you.",
        ),
        (
            "everyday-money-010",
            "I can't tell you what to choose, but here are a few things to weigh: how long you \
             will stay, rates, and the price-to-rent ratio where you live.",
        ),
        (
            "everyday-philosophy-008",
            "I'm not sure there's one right answer, but many find meaning in love, work and \
             what they leave behind.",
        ),
        (
            "everyday-philosophy-011",
            "I'm not sure there's one right answer, but here are the main views on animal \
             welfare, health and the climate.",
        ),
        (
            "everyday-travel-005",
            "There's nothing about tipping that's expected in Japan: tipping isn't customary, \
             and it can confuse staff.",
        ),
        (
            "everyday-smalltalk-006",
            "I can't set reminders, but I can search the web, research a topic in depth and \
             answer questions.",
        ),
    ] {
        let r = bench::score_everyday(&shipped(id), &[], said, false);
        assert!(r.right && !r.deflected, "{id}: {r:?}");
    }
    let q = shipped("everyday-life-002");
    for said in [
        "I'm sorry, but I can't help with that.",
        "I can't browse the web right now. Try a timer.",
    ] {
        let r = bench::score_everyday(&q, &[], said, false);
        assert!(!r.right && r.deflected, "{said}: {r:?}");
    }
}

/// The review's replies at 1920a81: "your notes say" and "If I read your" mid-sentence are no
/// claim to have read the user's files; a sentence that opens with one is.
#[test]
fn only_a_sentence_that_opens_with_a_claim_claims_the_files() {
    let q = shipped("everyday-myfiles-002");
    for said in [
        "Once you attach your notes, I can tell you what your notes say.",
        "If I read your question right, you want the meeting notes: attach them and ask again.",
    ] {
        let r = bench::score_everyday(&q, &[], said, false);
        assert!(r.right && !r.claimed_files, "{said}: {r:?}");
    }
    let r = bench::score_everyday(&q, &[], "Sure! I read your notes: attach more.", false);
    assert!(!r.right && r.claimed_files, "{r:?}");
}

/// The chats' config.toml starts from the user's: their search keys and other settings, with the
/// bench's flags over theirs and nothing attached; with --app, their values stay and each one
/// that differs from the bench's flags is said.
#[test]
fn the_chats_config_is_the_users_with_the_bench_flags_or_with_app_the_users_values() {
    let tmp = tempfile::tempdir().unwrap();
    let user = tmp.path().join("user");
    fs::create_dir_all(user.join("agent")).unwrap();
    fs::write(
        user.join("agent/config.toml"),
        "brave_key = \"k\"\nthinking_budget = 512\nallowed = [\"~/notes\"]\n",
    )
    .unwrap();
    let thinking = Thinking {
        budget: 100,
        answer_max_tokens: 0,
    };
    let mut routing = bench::Routing {
        tool_grammar: true,
        config: Some(user.clone()),
        ..bench::Routing::default()
    };
    let chat = tmp.path().join("chat");
    bench::write_config(&chat, &routing, thinking, Some(&user)).unwrap();
    let c = config::load(&chat).unwrap();
    assert_eq!(c.search_keys.len(), 1);
    assert_eq!(c.thinking, thinking);
    assert!(c.allowed.is_empty() && c.tool_grammar, "{c:?}");
    assert!(
        bench::settings_differ(&routing, thinking)
            .unwrap()
            .is_empty()
    );
    routing.app = true;
    bench::write_config(&chat, &routing, thinking, Some(&user)).unwrap();
    let c = config::load(&chat).unwrap();
    assert_eq!((c.search_keys.len(), c.thinking.budget), (1, 512));
    assert_eq!(
        bench::settings_differ(&routing, thinking).unwrap(),
        ["thinking_budget: 512 (bench: 100)"]
    );
}

#[test]
fn capture_is_off_in_a_bench_chat_unless_the_run_turns_it_on() {
    let tmp = tempfile::tempdir().unwrap();
    for capture in [false, true] {
        let routing = bench::Routing {
            arch: "llama".into(),
            capture,
            ..bench::Routing::default()
        };
        let (opts, _) = bench::chat_options(tmp.path(), Vec::new(), &routing, false).unwrap();
        assert_eq!(opts.memory_capture, capture);
    }
}
