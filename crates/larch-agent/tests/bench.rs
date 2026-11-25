//! The frozen prompts, the scorer, the runner over scripted replies and the report.

mod common;

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use larch_agent::{
    Agent, AgentOptions,
    bench::{
        self, BenchPrompt, CitedQuote, Cut, DATE, Docs, DocsQuestion, Expect, Report, SCHEMA,
        ScriptedStep, StepPrompt, ToolcallScore, score_answer, score_toolcall,
    },
    config::Thinking,
    parse::{Family, ParseError, Parsed, ToolCall},
    router::Router,
    tool::Registry,
};
use serde_json::{Value, json};

use common::{Fake, call, spaced};

fn agent_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/agent")
}

fn prompt(
    id: &str,
    tools: &[&str],
    user: &str,
    tool: Option<&str>,
    args: serde_json::Value,
) -> BenchPrompt {
    BenchPrompt {
        id: id.into(),
        tools: tools.iter().map(|t| t.to_string()).collect(),
        history: Vec::new(),
        user: user.into(),
        expect: Expect {
            tool: tool.map(Into::into),
            args_required: args.as_object().unwrap().clone(),
            ..Expect::default()
        },
        servers: Vec::new(),
    }
}

const CHAT: [&str; 7] = [
    "list_dir",
    "read_file",
    "write_file",
    "recall",
    "remember",
    "forget",
    "ask_docs",
];

#[test]
fn the_prompts_file_holds_the_mix() {
    let (prompts, file) = bench::load(&agent_dir().join("toolcalls.jsonl")).unwrap();
    assert_eq!(prompts.len(), 200);
    assert_eq!(file.sha256.len(), 64);
    let mut kinds = BTreeMap::new();
    let registry = Registry::builtin();
    for (i, p) in prompts.iter().enumerate() {
        let (kind, n) = p.id.split_once('-').unwrap();
        let count = kinds.entry(kind).or_insert(0);
        *count += 1;
        assert_eq!(
            n,
            format!("{count:03}"),
            "{} is out of order at line {}",
            p.id,
            i + 1
        );
        for tool in &p.tools {
            assert!(
                registry.get(tool).is_some(),
                "{}: {tool} is not a v1 tool",
                p.id
            );
        }
        assert_eq!(p.expect.tool.is_none(), kind == "none", "{}", p.id);
        if let Some(tool) = &p.expect.tool {
            assert!(p.tools.contains(tool), "{}: {tool} is not shown", p.id);
            let spec = &registry.specs(&[tool.as_str()])[0];
            for key in p.expect.args_required.keys() {
                assert!(
                    spec.schema["properties"].get(key).is_some(),
                    "{}: {tool} has no {key}",
                    p.id
                );
            }
        } else {
            assert!(p.expect.args_required.is_empty(), "{}", p.id);
        }
        // An inferred date is never in the text.
        if kind == "infer" {
            assert!(!p.user.contains(DATE), "{}", p.id);
        }
    }
    assert_eq!(
        kinds,
        BTreeMap::from([("clear", 120), ("infer", 20), ("near", 20), ("none", 40)])
    );
}

#[test]
fn the_source_routing_holdout_has_each_class_the_chat_tools_and_no_policy_worded_row() {
    let (rows, _) = bench::load(&agent_dir().join("source-routing-holdout.jsonl")).unwrap();
    // source-routing.jsonl's rows were written in the chat policy's own words; the holdout's were
    // not, so none of its prompts may be one of them.
    let (worded, _) = bench::load(&agent_dir().join("source-routing.jsonl")).unwrap();
    let worded: HashSet<_> = worded.iter().map(|r| r.user.to_lowercase()).collect();
    let mut classes = BTreeMap::new();
    for r in &rows {
        assert_eq!(r.tools, larch_agent::CHAT_TOOLS, "{}", r.id);
        assert!(!worded.contains(&r.user.to_lowercase()), "{}", r.id);
        let class =
            r.id.trim_start_matches("holdout-")
                .rsplit_once('-')
                .unwrap()
                .0;
        *classes.entry(class.to_owned()).or_insert(0) += 1;
    }
    let want = [
        ("ambiguous", 2),
        ("files-absent", 6),
        ("files-present", 4),
        ("none", 5),
        ("second-turn", 3),
        ("skills", 2),
        ("web", 7),
    ];
    assert_eq!(classes, want.map(|(c, n)| (c.to_owned(), n)).into());
    // What changes over time is a web question.
    for user in ["who is the CEO of Apple", "what is the latest iPhone"] {
        let r = rows.iter().find(|r| r.user == user).unwrap();
        assert_eq!(r.expect.tool.as_deref(), Some("web_answer"), "{}", r.id);
    }
    // Only these rows have earlier turns.
    let turns: Vec<_> = (rows.iter())
        .filter(|r| !r.history.is_empty())
        .map(|r| &r.id)
        .collect();
    assert_eq!(
        turns,
        [
            "holdout-second-turn-001",
            "holdout-second-turn-002",
            "holdout-second-turn-003"
        ]
    );
    // -003 is -002 as a chat sends it now: ask_docs's first line over the document set first.
    let sent = |id: &str| {
        let row = rows.iter().find(|r| r.id == id).unwrap();
        row.history.last().unwrap().content.clone()
    };
    let head = "Read 24 files (0 pdf, 15 md, 9 txt); no text in scan-passport.pdf";
    assert_eq!(
        sent("holdout-second-turn-003"),
        format!("{head}\n\n{}", sent("holdout-second-turn-002"))
    );
}

/// Each run of four words in `text`, lower case.
fn runs_of_four(text: &str) -> HashSet<String> {
    let words: Vec<String> = (text.to_lowercase().replace('\u{2019}', "'"))
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(Into::into)
        .collect();
    words.windows(4).map(|w| w.join(" ")).collect()
}

/// Every string in `value`.
fn strings(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::String(s) => out.extend([s.as_str(), "\n"]),
        serde_json::Value::Array(a) => a.iter().for_each(|v| strings(v, out)),
        serde_json::Value::Object(o) => o.values().for_each(|v| strings(v, out)),
        _ => {}
    }
}

#[test]
fn no_holdout_prompt_shares_four_words_with_the_prompt_the_tools_or_the_old_rows() {
    // What the model or the older rows already say: the system prompt's house text, the tool
    // descriptions, and the rows written before the holdout.
    let mut seen = bench::system_prompt();
    for spec in Registry::builtin().specs(&larch_agent::CHAT_TOOLS) {
        strings(&spec.function(), &mut seen);
    }
    for file in ["toolcalls.jsonl", "source-routing.jsonl"] {
        for r in bench::load(&agent_dir().join(file)).unwrap().0 {
            seen.extend([r.user.as_str(), "\n"]);
        }
    }
    let seen: HashSet<String> = seen.lines().flat_map(runs_of_four).collect();
    let (rows, _) = bench::load(&agent_dir().join("source-routing-holdout.jsonl")).unwrap();
    let mut shared = Vec::new();
    for r in &rows {
        let asked = (r.history.iter())
            .filter(|m| m.role == larch::Role::User)
            .map(|m| m.content.as_str());
        for text in asked.chain([r.user.as_str()]) {
            shared.extend(
                runs_of_four(text)
                    .intersection(&seen)
                    .map(|w| format!("{}: {w}", r.id)),
            );
        }
    }
    assert!(shared.is_empty(), "{shared:#?}");
}

#[test]
fn the_document_rows_load_and_each_expects_ask_docs_among_the_chat_tools() {
    let (rows, _) = bench::load(&agent_dir().join("docs-routing.jsonl")).unwrap();
    assert_eq!(rows.len(), 10);
    for p in &rows {
        assert_eq!(p.expect.tool.as_deref(), Some("ask_docs"), "{}", p.id);
        assert_eq!(p.tools, CHAT, "{}", p.id);
    }
}

/// Runs `script args` with Python 3: whether it passed, and what it printed to stderr.
fn python(script: &Path, args: &[&str]) -> (bool, String) {
    // Windows names Python 3 `py`; `python3` there is the Store's placeholder.
    let python = if cfg!(windows) { "py" } else { "python3" };
    let out = Command::new(python)
        .arg(script)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("{python} runs {}: {e}", script.display()));
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    (out.status.success(), stderr)
}

#[test]
fn gen_py_wrote_the_prompts_file() {
    let (passed, stderr) = python(&agent_dir().join("gen.py"), &["--check"]);
    assert!(passed, "{stderr}");
}

#[test]
fn make_docs_wrote_the_document_set_and_passes_over_a_finder_file() {
    // On a copy: Finder writes a .DS_Store into a folder it opens, which the check passes over.
    let tmp = tempfile::tempdir().unwrap();
    for entry in fs::read_dir(agent_dir().join("docs")).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), tmp.path().join(entry.file_name())).unwrap();
    }
    fs::write(tmp.path().join(".DS_Store"), [0u8; 8]).unwrap();
    let (passed, stderr) = python(&tmp.path().join("make_docs.py"), &["--check"]);
    assert!(passed, "{stderr}");
    // Any other name it does not write is reported, a hidden one too.
    fs::write(tmp.path().join(".old-notes.md"), "Gate code 1234.\n").unwrap();
    let (passed, stderr) = python(&tmp.path().join("make_docs.py"), &["--check"]);
    let extra = ".old-notes.md is not written by make_docs.py";
    assert!(!passed && stderr.contains(extra), "{stderr}");
}

#[test]
fn the_research_evals_run_end_to_end_on_the_fake_model() {
    let (passed, stderr) = python(&agent_dir().join("research/run.py"), &["--fake"]);
    assert!(passed, "{stderr}");
}

#[test]
fn the_research_eval_unit_tests_pass() {
    let (passed, stderr) = python(&agent_dir().join("research/test_research.py"), &[]);
    assert!(passed, "{stderr}");
}

#[test]
fn the_coding_evals_report_the_committed_tool_outputs() {
    let (passed, stderr) = python(&agent_dir().join("coding/run.py"), &["--fake"]);
    assert!(passed, "{stderr}");
}

#[test]
fn the_coding_eval_unit_tests_pass() {
    let (passed, stderr) = python(&agent_dir().join("coding/test_coding.py"), &[]);
    assert!(passed, "{stderr}");
}

#[test]
fn a_call_scores_on_its_tool_and_its_required_arguments() {
    let expect = Expect {
        tool: Some("log".into()),
        args_required: json!({"branch": "Main", "n": 5, "base": null})
            .as_object()
            .unwrap()
            .clone(),
        ..Expect::default()
    };
    let parsed = |name: &str, args| Parsed {
        call: Some(ToolCall {
            name: name.into(),
            args,
        }),
        ..Parsed::default()
    };
    let all = ToolcallScore {
        parsed_ok: true,
        right_tool: true,
        args_ok: true,
    };
    let good = parsed(
        "log",
        json!({"branch": "origin/main", "n": 5, "base": "dev"}),
    );
    assert_eq!(score_toolcall(&expect, &good), all);
    // An extra call is still the first call, scored.
    let more = Parsed {
        error: Some(ParseError::MultipleCalls),
        ..good
    };
    assert_eq!(score_toolcall(&expect, &more), all);
    let bad_args = ToolcallScore {
        args_ok: false,
        ..all
    };
    let missing = parsed("log", json!({"branch": "main", "n": 5}));
    assert_eq!(score_toolcall(&expect, &missing), bad_args);
    // A number must be equal: 15 holds the digit 5 but is not 5.
    let other = parsed("log", json!({"branch": "main", "n": 15, "base": "dev"}));
    assert_eq!(score_toolcall(&expect, &other), bad_args);
    // 5.0 is 5, as the tool's schema takes it.
    let float = parsed("log", json!({"branch": "main", "n": 5.0, "base": "dev"}));
    assert_eq!(score_toolcall(&expect, &float), all);
    let wrong = parsed(
        "changed_files",
        json!({"branch": "main", "n": 5, "base": "dev"}),
    );
    assert_eq!(
        score_toolcall(&expect, &wrong),
        ToolcallScore {
            parsed_ok: true,
            ..ToolcallScore::default()
        }
    );
    let broken = Parsed {
        error: Some(ParseError::MissingName),
        ..Parsed::default()
    };
    assert_eq!(score_toolcall(&expect, &broken), ToolcallScore::default());
    let answer = Expect::default();
    assert_eq!(score_toolcall(&answer, &Parsed::default()), all);
}

#[test]
fn an_answer_scores_on_its_text_and_a_refusal_is_wrong_where_no_tool_is_right() {
    let expect = |tool: Option<&str>, must: &[&str], not: &[&str]| Expect {
        tool: tool.map(Into::into),
        answer_must_contain: must.iter().map(|s| s.to_string()).collect(),
        answer_must_not_contain: not.iter().map(|s| s.to_string()).collect(),
        ..Expect::default()
    };
    let say = |text: &str| Parsed {
        text_before: text.into(),
        ..Parsed::default()
    };
    let refusal = say("The documents do not answer this question.");
    // A row without answer fields that expects a tool is not checked, a refusal included.
    assert_eq!(
        score_answer(&expect(Some("ask_docs"), &[], &[]), &refusal),
        None
    );
    assert_eq!(score_answer(&expect(None, &[], &[]), &say("Salt.")), None);
    assert_eq!(
        score_answer(&expect(None, &[], &[]), &refusal).as_deref(),
        Some("refuses: 'the documents do not'")
    );
    let salt = expect(None, &["Salt"], &["lead"]);
    assert_eq!(score_answer(&salt, &say("SALT tightens the gluten.")), None);
    assert_eq!(
        score_answer(&salt, &say("It tightens the gluten.")).as_deref(),
        Some("lacks 'Salt'")
    );
    assert_eq!(
        score_answer(&salt, &say("Salt, like LEAD, is heavy.")).as_deref(),
        Some("has 'lead'")
    );
    // A call is scored on its tool, not its text.
    let called = Parsed {
        call: Some(ToolCall {
            name: "recall".into(),
            args: json!({}),
        }),
        ..say("Let me check.")
    };
    assert_eq!(score_answer(&salt, &called), None);
}

fn holdout(id: &str) -> BenchPrompt {
    let (rows, _) = bench::load(&agent_dir().join("source-routing-holdout.jsonl")).unwrap();
    rows.into_iter().find(|r| r.id == id).unwrap()
}

fn say(text: &str) -> Parsed {
    Parsed {
        text_before: text.into(),
        ..Parsed::default()
    }
}

#[test]
fn a_refusal_is_its_shape_whatever_the_apostrophe_and_a_recipe_step_is_not_one() {
    let row = holdout("holdout-second-turn-001");
    for refusal in [
        "I don't have a chicken stew recipe in your files.",
        "I don\u{2019}t have a chicken stew recipe in your files.",
        "Sorry, I'm unable to find a chicken stew recipe.",
        "The documents do not have a chicken recipe.",
        "A chicken stew is not in your files.",
        "Your files don't mention a chicken stew.",
        "The documents don't have one.",
        "I couldn't find a chicken stew recipe.",
        "I could not find a chicken stew recipe.",
        "There is nothing about chicken stew in your notes.",
    ] {
        assert!(
            score_answer(&row.expect, &say(refusal)).is_some_and(|r| r.starts_with("refuses")),
            "{refusal}"
        );
    }
    let recipe =
        "Brown the chicken, add onions and stock, simmer 40 minutes. Do not include the bones.";
    assert_eq!(score_answer(&row.expect, &say(recipe)), None);
}

#[test]
fn an_absent_fact_must_be_refused_without_a_number_or_a_date() {
    for id in ["holdout-files-absent-001", "holdout-files-absent-002"] {
        let expect = holdout(id).expect;
        for invented in [
            "Your blood type is AB negative, from your notes.",
            "Your dentist appointment is Tuesday at noon, don't be late.",
            "The rent is nine hundred dollars a month, if I'm not mistaken.",
            "I don't have it, but it may be on 3 June.",
            "I don't have the date; try Friday.",
            "I don't have your blood type on file, but it is most likely O positive.",
            "I don't know your exact warranty, but dishwashers usually come with a year of cover.",
            "I don't have the time, but eye exams are usually in the morning.",
            "I don't know; maybe tomorrow.",
            "I don't have it; try in the a.m.",
            "I don't have it, but it is some time p.m., I think.",
            "I can't say, but it was some weeks ago.",
        ] {
            assert!(
                score_answer(&expect, &say(invented)).is_some(),
                "{id}: {invented}"
            );
        }
        for refusal in [
            "I don't have that in your files.",
            "Your files do not say.",
            "I can\u{2019}t find it in your notes or memory.",
            "Your files don't mention it.",
            "I don't know. No one knows but you.",
        ] {
            assert_eq!(
                score_answer(&expect, &say(refusal)),
                None,
                "{id}: {refusal}"
            );
        }
    }
}

#[test]
fn answer_texts_match_whole_words_and_a_star_lets_the_last_one_run_on() {
    let expect = Expect {
        answer_must_contain: vec!["minute*".into()],
        answer_must_not_contain: vec!["one".into()],
        ..Expect::default()
    };
    assert_eq!(
        score_answer(&expect, &say("Boil it for ten minutes; check your phone.")),
        None
    );
    assert_eq!(
        score_answer(&expect, &say("Ten minutes, one egg.")).as_deref(),
        Some("has 'one'")
    );
    assert_eq!(
        score_answer(&expect, &say("Boil it until done.")).as_deref(),
        Some("lacks 'minute*'")
    );
}

#[test]
fn a_general_answer_must_say_more_than_the_question_and_the_right_way_round() {
    let cases = [
        (
            "holdout-none-001",
            "Salt controls the yeast.",
            "Bread needs salt.",
        ),
        (
            "holdout-none-002",
            "Gold is heavier than lead.",
            "Lead is heavier than gold.",
        ),
        (
            "holdout-none-003",
            "It is efficient but weak in the cold.",
            "A heat pump pumps heat.",
        ),
        (
            "holdout-none-004",
            "A small Dobsonian.",
            "A telescope you like.",
        ),
        (
            "holdout-none-005",
            "Boil it for nine minutes.",
            "Boil an egg in water.",
        ),
    ];
    for (id, right, wrong) in cases {
        let expect = holdout(id).expect;
        assert_eq!(score_answer(&expect, &say(right)), None, "{id}: {right}");
        assert!(
            score_answer(&expect, &say(wrong)).is_some(),
            "{id}: {wrong}"
        );
    }
}

#[test]
fn a_row_can_take_any_of_several_tools_and_an_answer_that_holds_one_of_several_texts() {
    let expect: Expect = serde_json::from_value(json!({
        "tool_any": ["ask_docs", "recall", null],
        "args_required": {},
        "answer_must_contain_any": ["do not", "no record"],
        "answer_must_not_contain": ["0", "7"],
    }))
    .unwrap();
    let all = ToolcallScore {
        parsed_ok: true,
        right_tool: true,
        args_ok: true,
    };
    let call = |name: &str| Parsed {
        call: Some(ToolCall {
            name: name.into(),
            args: json!({}),
        }),
        ..Parsed::default()
    };
    let say = |text: &str| Parsed {
        text_before: text.into(),
        ..Parsed::default()
    };
    assert_eq!(score_toolcall(&expect, &call("recall")), all);
    assert!(!score_toolcall(&expect, &call("web_answer")).right_tool);
    assert_eq!(score_toolcall(&expect, &say("I do not know.")), all);
    // A refusal is what this row wants, so REFUSALS do not apply.
    let refusal = say("The documents do not answer this question.");
    assert_eq!(score_answer(&expect, &refusal), None);
    assert_eq!(score_answer(&expect, &say("No RECORD of it.")), None);
    assert_eq!(
        score_answer(&expect, &say("It is type A.")).as_deref(),
        Some("lacks any of 'do not', 'no record'")
    );
    assert_eq!(
        score_answer(&expect, &say("I do not know; maybe day 7.")).as_deref(),
        Some("has '7'")
    );
    // Without null among them, an answer is not a right tool.
    let tools: Expect = serde_json::from_value(json!({
        "tool_any": ["ask_docs", "recall"],
        "args_required": {},
    }))
    .unwrap();
    assert!(!score_toolcall(&tools, &say("Type A.")).right_tool);
}

#[test]
fn a_second_turn_row_sends_its_history_before_the_prompt() {
    let fake = Fake::new(32768);
    let row: BenchPrompt = serde_json::from_value(json!({
        "id": "holdout-second-turn-001",
        "tools": CHAT,
        "history": [
            {"role": "user", "content": "recipe for chicken stew", "truncated": false},
            {"role": "assistant", "content": "", "truncated": false, "tool_calls": [
                {"name": "ask_docs", "arguments": {"question": "recipe for chicken stew"}, "id": "000000001"}
            ]},
            {"role": "tool", "content": "The documents do not answer this question.", "truncated": false, "tool_call_id": "000000001"},
            {"role": "assistant", "content": "The documents do not answer this question.", "truncated": false}
        ],
        "user": "give me a recipe for chicken stew",
        "expect": {"tool": null, "args_required": {}, "answer_must_contain": ["chicken"]},
    }))
    .unwrap();
    fake.say("Brown the chicken, then simmer it with onions.");
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &[row],
        |_| {},
    );
    assert_eq!(rows[0].error, None);
    assert!(
        rows[0].after.right_tool && rows[0].wrong_answer.is_none(),
        "{:?}",
        rows[0]
    );
    let fed = fake.fed();
    let (first, second) = (
        fed.find("recipe for chicken stew").unwrap(),
        fed.find("give me a recipe").unwrap(),
    );
    assert!(first < second, "{fed}");
    assert!(
        fed.contains("The documents do not answer this question."),
        "{fed}"
    );
}

#[test]
fn a_row_whose_history_ends_with_a_tool_result_scores_the_answer_to_it() {
    let fake = Fake::new(32768);
    let row = holdout("holdout-second-turn-002");
    assert_eq!(row.user, "");
    fake.say("Brown the chicken, add onions and stock, and simmer it.");
    fake.say("I don't have a chicken stew recipe in your files.");
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &[row.clone(), row],
        |_| {},
    );
    assert_eq!(rows[0].error, None);
    assert!(
        rows[0].after.right_tool && rows[0].wrong_answer.is_none(),
        "{:?}",
        rows[0]
    );
    assert_eq!(
        rows[1].wrong_answer.as_deref(),
        Some("refuses: 'i don't have'")
    );
    let fed = fake.fed();
    assert!(fed.contains("chicken stew, how would I make it"), "{fed}");
    assert!(
        fed.contains("The documents do not answer this question."),
        "{fed}"
    );
}

#[test]
fn a_refusal_where_an_answer_is_right_marks_the_row_and_keeps_its_tool_score() {
    let fake = Fake::new(32768);
    let prompts = [prompt(
        "none-001",
        &CHAT,
        "recipe for chicken stew",
        None,
        json!({}),
    )];
    fake.say("The documents do not answer this question.");
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &prompts,
        |_| {},
    );
    assert!(rows[0].after.right_tool, "{:?}", rows[0]);
    assert_eq!(
        rows[0].wrong_answer.as_deref(),
        Some("refuses: 'the documents do not'")
    );
}

#[test]
fn the_summary_counts_each_class_and_lists_the_failing_rows() {
    let row = |id: &str, right: bool, answer: Option<&str>, error: bool| bench::Row {
        id: id.into(),
        after: ToolcallScore {
            parsed_ok: true,
            right_tool: right,
            args_ok: right,
        },
        wrong_answer: answer.map(Into::into),
        error: error.then(|| "too long".into()),
        ..bench::Row::default()
    };
    let rows = [
        row("source-none-001", true, None, false),
        row(
            "source-none-002",
            true,
            Some("refuses: 'do not say'"),
            false,
        ),
        row("source-web-001", false, None, false),
        row("holdout-files-present-001", false, None, true),
        row("clear-001", true, None, false),
        row("near-002", false, None, false),
    ];
    let s = bench::summary(&rows);
    let count = |rows, errors, wrong_tool, wrong_answer| bench::ClassCount {
        rows,
        errors,
        wrong_tool,
        wrong_answer,
    };
    let want: BTreeMap<String, bench::ClassCount> = [
        ("none", count(2, 0, 0, 1)),
        ("web", count(1, 0, 1, 0)),
        ("files-present", count(1, 1, 0, 0)),
        ("other", count(2, 0, 1, 0)),
    ]
    .into_iter()
    .map(|(c, n)| (c.into(), n))
    .collect();
    assert_eq!(s.by_class, want);
    assert_eq!(
        s.failing,
        [
            "source-none-002",
            "source-web-001",
            "holdout-files-present-001",
            "near-002"
        ]
    );
    // Two of the six rows are right.
    assert_eq!(s.right_rate, 2.0 / 6.0);
}

#[test]
fn a_run_over_scripted_replies_counts_what_happened() {
    let fake = Fake::new(32768);
    let date = format!("journal/{DATE}.md");
    let prompts = [
        prompt(
            "clear-001",
            &CHAT,
            "Read notes/todo.txt.",
            Some("read_file"),
            json!({"path": "notes/todo.txt"}),
        ),
        prompt("none-001", &CHAT, "What is 2 and 2?", None, json!({})),
        prompt(
            "infer-001",
            &CHAT,
            "Write \"ok\" to today's journal.",
            Some("write_file"),
            json!({"path": DATE}),
        ),
        prompt(
            "near-001",
            &CHAT,
            "Use read_files on a.txt.",
            Some("read_file"),
            json!({"path": "a.txt"}),
        ),
        prompt(
            "clear-002",
            &CHAT,
            "What do you remember about my dentist?",
            Some("recall"),
            json!({"query": "dentist"}),
        ),
        prompt(
            "clear-003",
            &CHAT,
            "List the files in notes.",
            Some("list_dir"),
            json!({"path": "notes"}),
        ),
    ];
    fake.say(&call("read_file", json!({"path": "notes/todo.txt"})));
    fake.say("4.");
    // Not JSON, so the retry is the marked user message.
    fake.say("{\"name\": \"write_file\", \"parameters\": {\"path\": ");
    fake.say(&call("write_file", json!({"path": date, "text": "ok"})));
    // An unknown tool, so the retry is the error as the tool result.
    fake.say(&call("read_files", json!({"path": "a.txt"})));
    fake.say(&call("read_file", json!({"path": "a.txt"})));
    fake.say(&call("list_dir", json!({"path": "."})));
    // A plain answer where a call was needed: no call, so not a wrong tool.
    fake.say("There are three files.");
    let mut seen = Vec::new();
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &prompts,
        |row| seen.push(row.id.clone()),
    );
    assert_eq!(fake.unsaid(), 0);
    assert_eq!(
        seen,
        prompts.iter().map(|p| p.id.clone()).collect::<Vec<_>>()
    );
    let all = ToolcallScore {
        parsed_ok: true,
        right_tool: true,
        args_ok: true,
    };
    let scores: Vec<_> = rows
        .iter()
        .map(|r| (r.before, r.after, r.retried, r.tool.as_deref()))
        .collect();
    assert_eq!(
        scores,
        [
            (all, all, false, Some("read_file")),
            (all, all, false, None),
            (ToolcallScore::default(), all, true, Some("write_file")),
            (
                ToolcallScore {
                    parsed_ok: true,
                    ..ToolcallScore::default()
                },
                all,
                true,
                Some("read_file")
            ),
            (
                ToolcallScore {
                    parsed_ok: true,
                    ..ToolcallScore::default()
                },
                ToolcallScore {
                    parsed_ok: true,
                    ..ToolcallScore::default()
                },
                false,
                Some("list_dir")
            ),
            (
                ToolcallScore {
                    parsed_ok: true,
                    ..ToolcallScore::default()
                },
                ToolcallScore {
                    parsed_ok: true,
                    ..ToolcallScore::default()
                },
                false,
                None
            ),
        ]
    );
    for row in &rows {
        assert_eq!(row.error, None);
        assert!(row.prompt_tokens > 0 && row.generated_tokens > 0, "{row:?}");
    }
    let summary = bench::summary(&rows);
    assert_eq!((summary.prompts, summary.errors), (6, 0));
    assert_eq!(summary.parse_rate_before, 5.0 / 6.0);
    assert_eq!(summary.parse_rate_after, 1.0);
    // list_dir for read_file only; the plain answer to clear-003 is not a wrong tool.
    assert_eq!(summary.wrong_tool_rate, 1.0 / 6.0);
    assert!(summary.ttft_ms.p50 <= summary.ttft_ms.p99);
    // Each retry goes back as the loop sends it: the parse error as the marked user message, the
    // unknown tool as the result of its call (the template's tool role).
    let fed = fake.fed();
    let user = "user<|end_header_id|>\n\n[agent] Your last reply had no valid tool call: tool call is not valid JSON";
    assert!(fed.contains(user), "{fed}");
    let tool =
        "ipython<|end_header_id|>\n\n\"unknown tool 'read_files'; did you mean 'read_file'?\"";
    assert!(fed.contains(tool), "{fed}");
}

#[test]
fn the_routing_rows_give_each_family_a_mirror_that_expects_the_opposite() {
    let (rows, _) = bench::load(&agent_dir().join("routing/dev.jsonl")).unwrap();
    assert!((30..=50).contains(&rows.len()), "{}", rows.len());
    let mut labels: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for r in &rows {
        assert_eq!(r.tools, larch_agent::CHAT_TOOLS, "{}", r.id);
        // routing-<family>-<label>-<n>: the label names what the row takes.
        let parts: Vec<&str> = r.id.split('-').collect();
        let [_, family, label, _] = parts[..] else {
            panic!("{}", r.id)
        };
        let takes = match label {
            "none" | "refuse" => None,
            "web" => Some("web_answer"),
            "read" => Some("read_papers"),
            "list" => Some("list_dir"),
            other => Some(other),
        };
        assert_eq!(r.expect.tool.as_deref(), takes, "{}", r.id);
        assert_eq!(r.expect.refuse, label == "refuse", "{}", r.id);
        labels.entry(family).or_default().insert(label);
    }
    let alone: Vec<_> = (labels.iter())
        // An absent fact is refused whatever the wording.
        .filter(|(f, l)| l.len() < 2 && **f != "absent")
        .collect();
    assert!(alone.is_empty(), "families without a mirror: {alone:?}");
}

#[test]
fn an_answer_that_says_it_cannot_reach_the_web_is_its_own_counter() {
    let fake = Fake::new(32768);
    let prompts = [
        prompt(
            "web-001",
            &CHAT,
            "Search for today's news.",
            Some("web_answer"),
            json!({}),
        ),
        prompt("none-001", &CHAT, "What is 2 and 2?", None, json!({})),
    ];
    fake.say("I can\u{2019}t browse the internet, so I can't check the news.");
    fake.say("4.");
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &prompts,
        |_| {},
    );
    assert_eq!(
        rows.iter().map(|r| r.offline_claim).collect::<Vec<_>>(),
        [true, false]
    );
    assert_eq!(bench::summary(&rows).counters.unwrap().offline_claims, 1);
}

/// A call to a tool not in the list and a call that fails its tool's schema are counted apart, per
/// row and over the run, in both sections: with the grammar on each is a bug (v3 D11).
#[test]
fn unknown_tools_and_schema_fails_are_their_own_counters() {
    let fake = Fake::new(32768);
    let prompts = [
        prompt(
            "near-001",
            &CHAT,
            "Read a.txt.",
            Some("read_file"),
            json!({"path": "a.txt"}),
        ),
        prompt(
            "near-002",
            &CHAT,
            "Read a.txt.",
            Some("read_file"),
            json!({"path": "a.txt"}),
        ),
    ];
    fake.say(&call("read_files", json!({"path": "a.txt"})));
    fake.say(&call("read_file", json!({"path": "a.txt"})));
    fake.say(&call("read_file", json!({"path": 3})));
    fake.say(&call("read_file", json!({"path": "a.txt"})));
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &prompts,
        |_| {},
    );
    let counts: Vec<_> = rows
        .iter()
        .map(|r| (r.unknown_tools, r.schema_fails))
        .collect();
    assert_eq!(counts, [(1, 0), (0, 1)], "{rows:?}");
    let counters = bench::summary(&rows).counters.unwrap();
    assert_eq!((counters.unknown_tools, counters.schema_fails), (1, 1));
    fake.say(&call("list_dirs", json!({"path": "invoices"})));
    fake.say(&call("list_dir", json!({"path": 3})));
    let rows = bench::run_steps(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &[step_row("chain-001")],
        |_| {},
    );
    assert_eq!(
        (rows[0].unknown_tools, rows[0].schema_fails),
        (1, 1),
        "{rows:?}"
    );
    let counters = bench::step_summary(&rows).counters;
    assert_eq!((counters.unknown_tools, counters.schema_fails), (1, 1));
}

/// A missing argument and a value under its minimum count as schema fails, as a wrong type does;
/// a reply whose JSON does not parse counts in neither.
#[test]
fn every_schema_fail_counts_and_a_reply_that_does_not_parse_counts_in_neither() {
    let fake = Fake::new(32768);
    let read = |id| {
        prompt(
            id,
            &CHAT,
            "Read a.txt.",
            Some("read_file"),
            json!({"path": "a.txt"}),
        )
    };
    let prompts = [read("near-001"), read("near-002"), read("near-003")];
    let right = call("read_file", json!({"path": "a.txt"}));
    fake.say(&call("read_file", json!({})));
    fake.say(&right);
    fake.say(&call(
        "read_file",
        json!({"path": "a.txt", "start_line": 0}),
    ));
    fake.say(&right);
    fake.say(r#"{"name": "read_file", "parameters": {"path": "#);
    fake.say(&right);
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &prompts,
        |_| {},
    );
    let counts: Vec<_> = rows
        .iter()
        .map(|r| (r.unknown_tools, r.schema_fails, r.retried))
        .collect();
    assert_eq!(
        counts,
        [(0, 1, true), (0, 1, true), (0, 0, true)],
        "{rows:?}"
    );
}

#[test]
fn a_row_times_the_first_reply_not_the_retry() {
    let fake = Fake::new(32768);
    let prompts = [prompt(
        "near-001",
        &CHAT,
        "Use read_files on a.txt.",
        Some("read_file"),
        json!({"path": "a.txt"}),
    )];
    fake.say(&call("read_files", json!({"path": "a.txt"})));
    fake.say(&call("read_file", json!({"path": "a.txt"})));
    // Only the first reply's prefill is slow.
    fake.stall(Duration::from_secs(1));
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &prompts,
        |_| {},
    );
    assert!(rows[0].retried, "{:?}", rows[0]);
    assert!(rows[0].ttft_ms >= 1000.0, "{:?}", rows[0]);
}

#[test]
fn a_prompts_file_without_prompts_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty.jsonl");
    fs::write(&empty, "").unwrap();
    let error = bench::load(&empty).unwrap_err().to_string();
    assert!(error.contains("no prompts"), "{error}");
}

#[test]
fn a_prompt_the_engine_refuses_is_an_error_row_and_the_run_goes_on() {
    // The tools and a chat step's 1,024 tokens fit; the long prompt does not.
    let fake = Fake::new(6144);
    let prompts = [
        prompt(
            "none-001",
            &CHAT,
            &"many words ".repeat(1000),
            None,
            json!({}),
        ),
        prompt("none-002", &CHAT, "Hi.", None, json!({})),
    ];
    fake.say("Hello.");
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &prompts,
        |_| {},
    );
    assert!(rows[0].error.is_some(), "{:?}", rows[0]);
    assert_eq!(rows[1].error, None);
    let summary = bench::summary(&rows);
    assert_eq!((summary.prompts, summary.errors), (2, 1));
    // Rates are over the prompts that ran.
    assert_eq!(summary.parse_rate_before, 1.0);
}

#[test]
fn the_report_round_trips_with_and_without_docs() {
    let (_, prompts) = bench::load(&agent_dir().join("toolcalls.jsonl")).unwrap();
    let mut report = Report {
        schema: SCHEMA.into(),
        model: "m0".into(),
        machine: "test".into(),
        commit: None,
        started_at: "2024-07-26T00:00:00Z".into(),
        quick: false,
        router: Router::Off,
        router_margin: None,
        rename_tools: false,
        extra_tools: 0,
        app_prompt: false,
        turn: true,
        app: None,
        tool_grammar: true,
        thinking_budget: None,
        answer_max_tokens: None,
        docs_workers: None,
        docs_window: None,
        prompts,
        summary: bench::summary(&[]),
        multistep: None,
        docs: None,
        memory: None,
        prefix_cache: Vec::new(),
        kv_pool_blocks: None,
        prefix_cache_blocks: None,
        anchor_ms: None,
        replay: None,
        live: None,
        everyday: None,
        memory_chats: None,
        rows: vec![bench::Row {
            id: "none-001".into(),
            ..bench::Row::default()
        }],
    };
    let text = serde_json::to_string(&report).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["schema"], "larch-agent-bench/1");
    assert_eq!(value["docs"], serde_json::Value::Null);
    assert_eq!(serde_json::from_str::<Report>(&text).unwrap(), report);
    // A report written before the docs section existed still reads.
    let mut old = value.clone();
    old.as_object_mut().unwrap().remove("docs");
    assert_eq!(serde_json::from_value::<Report>(old).unwrap(), report);
    // A measure with nothing to count is null.
    report.docs = Some(Docs {
        precision: Some(0.9),
        recall: Some(0.8),
        abstain_rate: None,
        errors: [(
            "q05".into(),
            "prompt is 9000 tokens, the limit is 8192".into(),
        )]
        .into(),
        questions: vec![DocsQuestion {
            id: "q01".into(),
            answer: "$1,850 (lease-alder-street.md:14).".into(),
            cited: vec![CitedQuote {
                file: "lease-alder-street.md".into(),
                line: 14,
                quote: "Monthly rent: $1,850".into(),
            }],
            facts_emitted: 3,
            facts_verified: 2,
            recalled: vec![0],
            abstained: false,
            wall_ms: 1234.5,
            outcome: Some(bench::Outcome::Answered),
            stages: None,
            before: Vec::new(),
            right: true,
        }],
    });
    let text = serde_json::to_string(&report).unwrap();
    assert_eq!(serde_json::from_str::<Report>(&text).unwrap(), report);
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let cited = &value["docs"]["questions"][0]["cited"][0];
    assert_eq!(cited["file"], "lease-alder-street.md");
    // A docs section written before it had questions still reads.
    let mut old = value.clone();
    old["docs"].as_object_mut().unwrap().remove("questions");
    let read = serde_json::from_value::<Report>(old).unwrap();
    assert_eq!(read.docs.unwrap().questions, []);
    // The v2 sections, a quick run's mark and the router.
    report.quick = true;
    report.multistep = Some(bench::MultiStep {
        rows: vec![bench::StepRow {
            id: "parallel-001".into(),
            steps: vec![true, true, false],
            calls_in_one_step: Some(false),
            cut: vec![Cut::Call],
            ..bench::StepRow::default()
        }],
        ..bench::MultiStep::default()
    });
    report.memory = Some(bench::Memory {
        recall_at_5: Some(1.0),
        questions: vec![bench::MemoryRow {
            id: "m01".into(),
            hits: vec!["facts/a.md:7".into()],
            in_range: Some(true),
            ..bench::MemoryRow::default()
        }],
        ..bench::Memory::default()
    });
    let text = serde_json::to_string(&report).unwrap();
    assert_eq!(serde_json::from_str::<Report>(&text).unwrap(), report);
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        (&value["quick"], &value["router"]),
        (&json!(true), &json!("off"))
    );
    assert_eq!(value["multistep"]["rows"][0]["cut"], json!(["call"]));
    // A routed run: its margin, the renamed tools' mark, and what the router did per row.
    report.router = Router::Logprob;
    report.router_margin = Some(2.0);
    report.rename_tools = true;
    report.rows[0].routed = Some(bench::Routed {
        margin: Some(0.5),
        call: Some(false),
        again: true,
        ..bench::Routed::default()
    });
    report.summary.wrong_tool_by_group = [("files".into(), 0.25)].into();
    let text = serde_json::to_string(&report).unwrap();
    assert_eq!(serde_json::from_str::<Report>(&text).unwrap(), report);
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        (
            &value["router"],
            &value["router_margin"],
            &value["rename_tools"]
        ),
        (&json!("logprob"), &json!(2.0), &json!(true))
    );
    assert_eq!(value["rows"][0]["routed"]["again"], json!(true));
    // A report from before the router reads as one without it.
    let mut old = value.clone();
    for key in ["router_margin", "rename_tools"] {
        old.as_object_mut().unwrap().remove(key);
    }
    old["rows"][0].as_object_mut().unwrap().remove("routed");
    let read = serde_json::from_value::<Report>(old).unwrap();
    assert_eq!((read.router_margin, read.rename_tools), (None, false));
    assert_eq!(read.rows[0].routed, None);
}

/// Whether `new` holds every value of `old`, whatever it adds.
fn holds(old: &serde_json::Value, new: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (old, new) {
        (Value::Object(o), Value::Object(n)) => {
            o.iter().all(|(k, v)| n.get(k).is_some_and(|n| holds(v, n)))
        }
        (Value::Array(o), Value::Array(n)) => {
            o.len() == n.len() && o.iter().zip(n).all(|(o, n)| holds(o, n))
        }
        _ => old == new,
    }
}

#[test]
fn a_v1_report_loads_unchanged() {
    for name in ["agent-m5-m1.json", "agent-m5-m2.json"] {
        let text = fs::read_to_string(agent_dir().join("../results").join(name)).unwrap();
        let report: Report = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(report.schema, SCHEMA);
        let v2 = (
            report.quick,
            report.router,
            &report.multistep,
            &report.memory,
        );
        assert_eq!(v2, (false, Router::Off, &None, &None), "{name}");
        let summary = (report.summary.visible_ms, report.summary.counters);
        assert_eq!(summary, (None, None), "{name}");
        let rows = report.rows.iter();
        assert!(
            rows.clone()
                .all(|r| r.visible_ms.is_none() && r.cut.is_empty() && !r.wrong_but_valid),
            "{name}"
        );
        assert_eq!(
            report.docs.as_ref().map(|d| d.questions.len()),
            Some(0),
            "{name}"
        );
        // Written again, it keeps every v1 value as it was.
        let old: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(
            holds(&old, &serde_json::to_value(&report).unwrap()),
            "{name}"
        );
    }
}

#[test]
fn the_quick_subset_has_the_counts() {
    let (prompts, _) = bench::load(&agent_dir().join("toolcalls.jsonl")).unwrap();
    let quick = bench::quick_prompts(&prompts);
    let mut groups = BTreeMap::new();
    for p in &quick {
        *groups.entry(p.id.split_once('-').unwrap().0).or_insert(0) += 1;
    }
    assert_eq!(
        groups,
        BTreeMap::from([("clear", 10), ("infer", 10), ("near", 10), ("none", 10)])
    );
    // Fixed ids: each group's prompts spread evenly from its first.
    let none: Vec<_> = (quick.iter())
        .filter(|p| p.id.starts_with("none-"))
        .map(|p| p.id.as_str())
        .collect();
    let every_fourth: Vec<_> = (0..10).map(|i| format!("none-{:03}", 1 + 4 * i)).collect();
    assert_eq!(none, every_fourth);
    let inputs = bench::Inputs {
        prompts,
        steps: Some(bench::load_steps(&agent_dir().join("multistep.jsonl")).unwrap()),
        docs: Some(bench::doc_set(&agent_dir().join("docs")).unwrap()),
        memory: Some(memory_set()),
    }
    .quick();
    let steps = inputs.steps.unwrap().0;
    let counts = (
        inputs.prompts.len(),
        steps.len(),
        inputs.docs.unwrap().questions.len(),
        inputs.memory.unwrap().questions.len(),
    );
    assert_eq!(counts, (40, 10, 6, 5));
    // Every kind of multi-step row is in the ten.
    let kinds: BTreeSet<_> = steps
        .iter()
        .map(|r| r.id.split_once('-').unwrap().0)
        .collect();
    assert_eq!(kinds, BTreeSet::from(["chain", "error", "one", "parallel"]));
}

#[test]
fn every_tool_row_of_a_bench_starts_from_the_anchored_prompt() {
    let fake = Fake::new(32768);
    let ask = |id: &str, user: &str| prompt(id, &CHAT, user, None, json!({}));
    let inputs = bench::Inputs {
        prompts: vec![
            ask("none-001", "What is 2 and 2?"),
            ask("none-002", "What is 2 and 3?"),
            ask("none-003", "What is 3 and 3?"),
        ],
        steps: None,
        docs: None,
        memory: None,
    };
    // The anchor takes no token of the script (common/mod.rs), so the rows' replies.
    for reply in ["4.", "5.", "6."] {
        fake.say(reply);
    }
    let mut report = Report::default();
    bench::run_all(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &bench::Routing::default(),
        &inputs,
        &mut report,
        || fake.engine.prefix_cache_stats(),
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    // Each row reuses the system prompt and the tool list; the first one too, as no row computed
    // them before it.
    let shown: Vec<_> = (larch_agent::tool::Registry::builtin().specs(&CHAT).iter())
        .map(larch_agent::tool::ToolSpec::function)
        .collect();
    let tools =
        fake.tokens(&[("user", "")], &shown).len() - fake.tokens(&[("user", "")], &[]).len();
    let cached: Vec<_> = report.rows.iter().map(|r| r.cached_tokens).collect();
    assert!(cached.iter().all(|&c| c == cached[0]), "{cached:?}");
    assert!(
        cached[0] as usize > tools,
        "{cached:?}, the tool list {tools}"
    );
    assert_eq!(report.prefix_cache[0].hits, 3, "{:?}", report.prefix_cache);
    assert_eq!(fake.unsaid(), 0);
}

#[test]
fn the_app_prompt_holds_folders_names_and_memory_the_same_bytes_each_time() {
    let app = bench::app_prompt("qwen3moe").unwrap();
    assert_eq!(app, bench::app_prompt("qwen3moe").unwrap());
    for part in [
        "You may read and write files under:\n/Users/alex/papers-ml\n/Users/alex/notes",
        "## Memory index",
        "Chicago",
        "\"folder\":\"/Users/alex/papers-ml\"",
        "attention-is-all-you-need.pdf",
        "groceries.md",
        &format!("Today is {DATE}. Your own knowledge ends about Mar 2025,"),
    ] {
        assert!(app.contains(part), "{part} in {app}");
    }
    assert!(app.starts_with(&bench::system_prompt()[..100]), "{app}");
}

#[test]
fn a_bench_with_the_app_prompt_sends_it_to_every_row() {
    let fake = Fake::new(32768);
    let inputs = bench::Inputs {
        prompts: vec![prompt("none-001", &CHAT, "Hi.", None, json!({}))],
        steps: None,
        docs: None,
        memory: None,
    };
    fake.say("Hello.");
    let routing = bench::Routing {
        system: bench::app_prompt("llama").unwrap(),
        ..bench::Routing::default()
    };
    let mut report = Report::default();
    bench::run_all(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &routing,
        &inputs,
        &mut report,
        || fake.engine.prefix_cache_stats(),
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    assert!(fake.fed().contains("attention-is-all-you-need.pdf"));
    assert_eq!(report.rows[0].error, None);
}

#[test]
fn a_failed_anchor_is_a_progress_line_and_the_rows_still_run() {
    // The system prompt and the tool list are longer than this context, so the anchor fails.
    let fake = Fake::new(512);
    let inputs = bench::Inputs {
        prompts: vec![prompt("none-001", &CHAT, "Hi.", None, json!({}))],
        steps: None,
        docs: None,
        memory: None,
    };
    let mut report = Report::default();
    let mut lines = Vec::new();
    bench::run_all(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &bench::Routing::default(),
        &inputs,
        &mut report,
        || fake.engine.prefix_cache_stats(),
        |_| Ok(()),
        |l| lines.push(l),
    )
    .unwrap();
    let why = "the system prompt did not reach the prefix cache: ";
    assert!(lines[0].starts_with(why), "{lines:?}");
    assert_eq!(report.rows.len(), 1, "{lines:?}");
}

#[test]
fn the_sections_run_in_order_and_the_report_is_saved_after_each() {
    let fake = Fake::new(32768);
    let docs = tempfile::tempdir().unwrap();
    fs::write(
        docs.path().join("trains.md"),
        "The train to Leith leaves at nine.\n",
    )
    .unwrap();
    let asked = "When does the train to Leith leave?";
    let questions = json!([{"id": "q01", "question": asked, "facts": []}]);
    fs::write(docs.path().join("questions.json"), questions.to_string()).unwrap();
    let mut memory = memory_set();
    memory.questions.truncate(1);
    let remembered = memory.questions[0].question.clone();
    let step = step_row("one-001");
    let inputs = bench::Inputs {
        prompts: vec![prompt(
            "none-001",
            &CHAT,
            "What is 2 and 2?",
            None,
            json!({}),
        )],
        steps: Some((vec![step.clone()], Default::default())),
        docs: Some(bench::doc_set(docs.path()).unwrap()),
        memory: Some(memory),
    };
    // The anchors (of the chat's tool list, of none, and the docs question's worker anchor)
    // take no token of the script.
    fake.say("4.");
    fake.say(&call("read_file", json!({"path": "notes/todo.txt"})));
    fake.say("Milk, the plumber and the passport.");
    // The one excerpt's worker finds nothing, so no answer step runs.
    fake.say("[]");
    fake.say("41 Alder Street.");
    let (mut lines, mut saved) = (Vec::new(), Vec::new());
    let mut report = Report::default();
    bench::run_all(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &bench::Routing::default(),
        &inputs,
        &mut report,
        || fake.engine.prefix_cache_stats(),
        |r| {
            let sections = (r.multistep.is_some(), r.docs.is_some(), r.memory.is_some());
            saved.push((r.rows.len(), sections));
            Ok(())
        },
        |line| lines.push(line),
    )
    .unwrap();
    assert_eq!(fake.unsaid(), 0);
    let ids: Vec<_> = lines.iter().map(|l| l.split(' ').next().unwrap()).collect();
    assert_eq!(
        ids,
        [
            "2", "none-001", "prefix", "one-001", "prefix", "q01", "prefix", "m01", "prefix"
        ],
        "{lines:?}"
    );
    // The anchors' time, which no row holds, in the report and said.
    let anchor_ms = report.anchor_ms.unwrap();
    assert_eq!(lines[0], format!("2 anchors: {anchor_ms:.0} ms"));
    // The prefix cache's counters after each section, in the report and said.
    let sections: Vec<_> = report
        .prefix_cache
        .iter()
        .map(|c| c.section.as_str())
        .collect();
    assert_eq!(sections, ["toolcalls", "multistep", "docs", "memory"]);
    let hits: Vec<_> = report.prefix_cache.iter().map(|c| c.hits).collect();
    // Every section reuses an anchored prefix, the docs worker its own.
    assert!(
        hits[0] > 0 && hits.windows(2).all(|w| w[0] < w[1]),
        "{hits:?}"
    );
    let last = report.prefix_cache.last().unwrap();
    let said = format!(
        "prefix cache after memory: {} hits, {} reused prompt tokens",
        last.hits, last.reused_prompt_tokens
    );
    assert_eq!(lines.last(), Some(&said));
    assert_eq!(
        saved,
        [
            (1, (false, false, false)),
            (1, (true, false, false)),
            (1, (true, true, false)),
            (1, (true, true, true)),
        ]
    );
    // The model heard them in that order.
    let fed = fake.fed();
    let at = |text: &str| {
        fed.find(text)
            .unwrap_or_else(|| panic!("{text} not in {fed}"))
    };
    let order = [
        at("What is 2 and 2?"),
        at(&step.user),
        at(asked),
        at(&remembered),
    ];
    assert!(order.is_sorted(), "{order:?}");
    assert!(!report.failed(), "{report:?}");
    let multistep = report.multistep.unwrap();
    assert_eq!(multistep.summary.right_rate, 1.0);
    assert_eq!(report.memory.unwrap().holds_value, Some(1.0));
}

/// Runs git in `dir` for the test's setup, without the host's configuration.
fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args([
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
        ])
        .args(["-c", "commit.gpgsign=false"])
        .args(args)
        .current_dir(dir)
        .env("HOME", dir)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}: {out:?}");
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn the_report_names_the_checkouts_commit_and_its_local_edits() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    git(root, &["init", "-q"]);
    fs::write(root.join("a.txt"), "one\n").unwrap();
    git(root, &["add", "."]);
    git(root, &["commit", "-q", "-m", "start"]);
    let head = git(root, &["rev-parse", "HEAD"]).trim().to_owned();
    assert_eq!(bench::commit(root), Some(head.clone()));
    fs::write(root.join("a.txt"), "two\n").unwrap();
    assert_eq!(bench::commit(root), Some(format!("{head}-dirty")));
}

fn read_todo(id: &str) -> BenchPrompt {
    prompt(
        id,
        &CHAT,
        "Read notes/todo.txt.",
        Some("read_file"),
        json!({"path": "notes/todo.txt"}),
    )
}

#[test]
fn a_step_cut_at_the_token_limit_is_counted_with_where_it_was_cut() {
    // A call that never ends stops at the step's token limit inside its JSON; the retry goes on
    // with the same bytes, which are plain text there.
    let fake = Fake::new(32768);
    let start = "{\"name\": \"read_file\", \"parameters\": {\"path\": \"";
    let mut ids: Vec<u32> = start.bytes().map(|b| fake.byte(b)).collect();
    ids.extend([fake.byte(b'a'); 3000]);
    fake.say_tokens(&ids);
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &[read_todo("clear-001")],
        |_| {},
    );
    assert_eq!(rows[0].cut, [Cut::Call, Cut::Text], "{:?}", rows[0]);
    // Whole calls, then text to the limit, were cut in the text, whatever the parser says of the
    // calls: two of them (MultipleCalls), or one whose arguments are not an object and another
    // (ArgsNotObject, so a retry).
    let two = format!(
        "{}; {} ",
        call("read_file", json!({"path": "notes/todo.txt"})),
        call("read_file", json!({"path": "a.txt"}))
    );
    let bad = format!(
        "{{\"name\": \"read_file\", \"parameters\": \"notes/todo.txt\"}}; {} ",
        call("read_file", json!({"path": "a.txt"}))
    );
    // A Qwen3 call closed by its tag, then text with a bracket in it.
    let tagged = format!(
        "<tool_call>{}</tool_call>\nI will explain [the contents ",
        json!({"name": "read_file", "arguments": {"path": "notes/todo.txt"}})
    );
    for (id, family, calls, retried) in [
        ("clear-003", Family::Llama3, two, false),
        ("clear-004", Family::Llama3, bad, true),
        ("clear-005", Family::Qwen3, tagged, false),
    ] {
        let fake = Fake::new(32768);
        let mut ids: Vec<u32> = calls.bytes().map(|b| fake.byte(b)).collect();
        ids.extend([fake.byte(b'a'); 3000]);
        fake.say_tokens(&ids);
        let rows = bench::run(
            &fake.larch,
            family,
            Thinking::default(),
            0,
            &[read_todo(id)],
            |_| {},
        );
        assert_eq!(rows[0].cut.first(), Some(&Cut::Text), "{:?}", rows[0]);
        assert_eq!(
            (rows[0].before.parsed_ok, rows[0].retried),
            (!retried, retried),
            "{:?}",
            rows[0]
        );
    }
    // A step cut inside the hidden think block.
    let fake = Fake::new(32768);
    let mut ids = vec![fake.token("<think>")];
    ids.extend([fake.token("x"); 3000]);
    fake.say_tokens(&ids);
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &[read_todo("clear-002")],
        |_| {},
    );
    assert_eq!(rows[0].cut.first(), Some(&Cut::Think), "{:?}", rows[0]);
    let counters = bench::summary(&rows).counters.unwrap();
    assert_eq!(
        (counters.length_stops, counters.in_call, counters.in_think),
        (2, 0, 1)
    );
    assert_eq!(counters.retries, 1);
}

#[test]
fn a_valid_call_to_the_wrong_tool_is_wrong_but_valid_and_a_parse_error_is_not() {
    let fake = Fake::new(32768);
    let read = |id, path: &str| {
        prompt(
            id,
            &CHAT,
            &format!("Read {path}."),
            Some("read_file"),
            json!({ "path": path }),
        )
    };
    let prompts = [
        read("clear-001", "notes/todo.txt"),
        read("clear-002", "a.txt"),
        read("clear-003", "b.txt"),
        prompt("none-001", &CHAT, "What is 2 and 2?", None, json!({})),
    ];
    // The wrong tool, valid; a broken call, then the right one; the right tool on another file.
    fake.say(&call("list_dir", json!({"path": "notes"})));
    fake.say("{\"name\": \"read_file\", \"parameters\": {\"path\": ");
    fake.say(&call("read_file", json!({"path": "a.txt"})));
    fake.say(&call("read_file", json!({"path": "c.txt"})));
    fake.say("4.");
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &prompts,
        |_| {},
    );
    assert_eq!(fake.unsaid(), 0);
    let flags: Vec<_> = rows.iter().map(|r| r.wrong_but_valid).collect();
    assert_eq!(flags, [true, false, true, false]);
    // The report keeps the last reply, so a Mac run shows what a wrong row did.
    let replies: Vec<_> = rows.iter().map(|r| r.reply.as_deref()).collect();
    assert_eq!(
        replies,
        [
            Some(r#"list_dir {"path":"notes"}"#),
            Some(r#"read_file {"path":"a.txt"}"#),
            Some(r#"read_file {"path":"c.txt"}"#),
            Some("4."),
        ]
    );
    let counters = bench::summary(&rows).counters.unwrap();
    assert_eq!((counters.wrong_but_valid, counters.retries), (2, 1));
    assert_eq!(counters.length_stops, 0);
}

/// Every forward pass of a paced fake takes this long, so a step's times count its tokens.
const PACE: Duration = Duration::from_millis(25);

#[test]
fn visible_ms_after_hidden_thinking_is_the_first_visible_text_not_the_first_token() {
    let fake = Fake::new(32768);
    fake.pace(PACE);
    // The first token comes at once; eleven hidden ones follow before the answer's first.
    fake.say("<think>xxxxxxxxxx</think>It is 4.");
    let prompts = [prompt(
        "none-001",
        &CHAT,
        "What is 2 and 2?",
        None,
        json!({}),
    )];
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &prompts,
        |_| {},
    );
    let row = &rows[0];
    let visible = row.visible_ms.unwrap_or_else(|| panic!("{row:?}"));
    assert!(visible - row.ttft_ms >= 10.0 * 25.0, "{row:?}");
    let summary = bench::summary(&rows);
    assert_eq!(summary.visible_ms.map(|v| v.p50), Some(visible));
}

#[test]
fn visible_ms_on_a_call_row_is_the_time_to_the_call_line() {
    let fake = Fake::new(32768);
    fake.pace(PACE);
    // The call's text is held back while it streams; the user sees its line when it is whole.
    fake.say(&call("read_file", json!({"path": "notes/todo.txt"})));
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &[read_todo("clear-001")],
        |_| {},
    );
    let row = &rows[0];
    let visible = row.visible_ms.unwrap_or_else(|| panic!("{row:?}"));
    let after_first = f64::from(row.generated_tokens - 2) * 25.0;
    assert!(visible - row.ttft_ms >= after_first, "{row:?}");
}

#[test]
fn a_chat_with_a_router_this_build_lacks_does_not_open() {
    let fake = Fake::new(32768);
    let opts = AgentOptions {
        router: Router::Score,
        ..fake.opts()
    };
    let Err(e) = Agent::new(&fake.larch, opts) else {
        panic!("a chat opened with the score router");
    };
    let e = e.to_string();
    assert!(
        e.contains("router score") && e.contains("v2 PR 13 (agent/score-router)"),
        "{e}"
    );
}

#[test]
fn the_holdout_rows_do_not_overlap_the_200_and_hold_the_mix() {
    let same = |user: &str| {
        user.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    };
    let (main, _) = bench::load(&agent_dir().join("toolcalls.jsonl")).unwrap();
    let (held, file) = bench::load(&agent_dir().join("holdout.jsonl")).unwrap();
    assert_eq!(file.sha256.len(), 64);
    let seen: HashSet<String> = main.iter().map(|p| same(&p.user)).collect();
    for p in &held {
        assert!(!seen.contains(&same(&p.user)), "{} is in the 200", p.id);
    }
    let registry = Registry::builtin();
    let mut kinds = BTreeMap::new();
    for p in &held {
        let (kind, n) = p.id.split_once('-').unwrap();
        let count = kinds.entry(kind).or_insert(0);
        *count += 1;
        assert_eq!(n, format!("{count:03}"), "{} is out of order", p.id);
        assert_eq!(p.expect.tool.is_none(), kind == "none", "{}", p.id);
        if let Some(tool) = &p.expect.tool {
            assert!(
                registry.get(tool).is_some() && p.tools.contains(tool),
                "{}",
                p.id
            );
        }
    }
    assert_eq!(
        kinds,
        BTreeMap::from([("clear", 40), ("infer", 5), ("near", 5), ("none", 50)])
    );
}

fn memory_set() -> bench::MemorySet {
    bench::memory_set(&agent_dir().join("memory")).unwrap()
}

#[test]
fn tier_1_finds_every_planted_memory_and_misses_a_deleted_fact() {
    let set = memory_set();
    let dated = set.questions.iter().filter(|q| q.range.is_some()).count();
    assert_eq!((set.questions.len(), dated), (20, 5));
    let mut seen = 0;
    let m = bench::run_memory(None, Thinking::default(), &set, |_| seen += 1);
    assert_eq!(seen, 20);
    let missed: Vec<_> = (m.questions.iter())
        .filter(|r| !r.found)
        .map(|r| (&r.id, &r.hits))
        .collect();
    assert!(missed.is_empty(), "{missed:?}");
    assert_eq!(m.recall_at_5, Some(1.0));
    // "in June" keeps June's rows: by its words alone, "the car insurance I chose in June" best
    // matched the July fact that says "renews on 27 June".
    let out_of_range: Vec<_> = (m.questions.iter())
        .filter(|r| r.in_range == Some(false))
        .map(|r| (r.id.as_str(), r.hits[0].as_str()))
        .collect();
    assert_eq!(out_of_range, []);
    assert_eq!(m.in_range, Some(1.0));
    let m19 = m.questions.iter().find(|r| r.id == "m19").unwrap();
    assert_eq!(m19.hits[0], "daily/2024-06-27.md:1");
    assert!(
        !m19.hits.iter().any(|h| h.contains("renews-on-27")),
        "{m19:?}"
    );
    for (r, q) in m.questions.iter().zip(&set.questions) {
        assert_eq!((&r.error, &r.answer), (&None, &None), "{r:?}");
        assert_eq!(r.in_range.is_some(), q.range.is_some(), "{r:?}");
        assert!(r.hits.len() <= 5, "{r:?}");
    }
    assert_eq!(m.holds_value, None);
    // The plumber's number is in its fact alone.
    let fact = "agent/memory/facts/the-plumber-is-tomasz-nowak-phone.md";
    fs::remove_file(set.data_dir().join(fact)).unwrap();
    let m = bench::run_memory(None, Thinking::default(), &set, |_| {});
    let missed: Vec<_> = (m.questions.iter())
        .filter(|r| !r.found)
        .map(|r| r.id.as_str())
        .collect();
    assert_eq!(missed, ["m07"]);
    assert_eq!(m.recall_at_5, Some(19.0 / 20.0));
}

#[test]
fn tier_1_reads_no_transcript() {
    // The hits go to the model unasked, so a chat that said the answer is not one of them.
    let set = memory_set();
    let chat = set.data_dir().join("chat");
    fs::create_dir_all(&chat).unwrap();
    let line = json!({"role": "user", "content": "the van is booked with Rapid Removals, my new address is 41 Alder Street"});
    fs::write(chat.join("1-1.jsonl"), format!("{line}\n")).unwrap();
    assert!(
        larch_agent::memory::recall(set.data_dir(), "Rapid Removals")
            .unwrap()
            .iter()
            .any(|h| h.path == "chat/1-1.jsonl")
    );
    let m = bench::run_memory(None, Thinking::default(), &set, |_| {});
    let hits: Vec<_> = m.questions.iter().flat_map(|r| &r.hits).collect();
    assert!(hits.iter().all(|h| !h.starts_with("chat/")), "{hits:?}");
    assert_eq!(m.recall_at_5, Some(1.0));
}

#[test]
fn tier_1_with_keys_scores_at_least_tier_1_without() {
    let off = bench::run_memory(None, Thinking::default(), &memory_set(), |_| {});
    let fake = Fake::new(32768);
    let mut set = memory_set();
    set.keys = true;
    // One key turn per note line, in the notes' order: its last words, as a model picks names
    // and amounts.
    let notes = larch_agent::memory::notes(set.data_dir()).unwrap();
    assert_eq!(notes.len(), 20);
    for n in &notes {
        let words: Vec<&str> = (n.text.split_whitespace())
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .collect();
        let at = words.len();
        fake.say(&format!(
            "{}\n{}",
            words[at - 2..].join(" "),
            words[at - 4..at - 2].join(" ")
        ));
    }
    let one_at_a_time = larch_agent::workers::Workers {
        max_workers: 1,
        prefix_cache: false,
    };
    let pass = bench::memory_keys(&fake.larch, Family::Llama3, 0, &one_at_a_time, &set);
    assert_eq!(
        (pass.lines, pass.keys, pass.errors.len()),
        (20, 40, 0),
        "{pass:?}"
    );
    assert_eq!(fake.unsaid(), 0);
    let on = bench::run_memory(None, Thinking::default(), &set, |_| {});
    assert!(on.recall_at_5 >= off.recall_at_5, "{on:?}");
    assert!(on.in_range >= off.in_range, "{on:?}");
    assert_eq!(on.recall_at_5, Some(1.0));
    // A key's hit is the note line it points at, so the answer's file and line still match.
    let keys = fs::read_to_string(set.data_dir().join("agent/memory/keys/2024-07-20.txt")).unwrap();
    assert_eq!(
        keys,
        "1 camera adapter\n1 raincoat boots\n2 at 20:00\n2 22 July\n"
    );
}

#[test]
fn a_key_pass_that_saves_no_keys_fails_the_report_and_a_run_without_keys_does_not() {
    let run = |keys: bool| {
        let fake = Fake::new(32768);
        let mut memory = memory_set();
        memory.questions.truncate(1);
        memory.keys = keys;
        // Nothing scripted: every key turn's reply is empty, so no line gets keys, and the
        // question's answer is empty too, a wrong answer and no error.
        let inputs = bench::Inputs {
            prompts: Vec::new(),
            steps: None,
            docs: None,
            memory: Some(memory),
        };
        let mut report = Report::default();
        bench::run_all(
            &fake.larch,
            Family::Llama3,
            Thinking::default(),
            0,
            &bench::Routing::default(),
            &inputs,
            &mut report,
            || fake.engine.prefix_cache_stats(),
            |_| Ok(()),
            |_| {},
        )
        .unwrap();
        report
    };
    let on = run(true);
    let m = on.memory.as_ref().unwrap();
    assert!(m.questions.iter().all(|q| q.error.is_none()), "{m:?}");
    let keys = m.keys.as_ref().unwrap();
    assert_eq!((keys.keys, keys.errors.len()), (0, 20), "{keys:?}");
    assert!(on.failed(), "{on:?}");
    let off = run(false);
    assert!(off.memory.as_ref().unwrap().keys.is_none(), "{off:?}");
    assert!(!off.failed(), "{off:?}");
}

#[test]
fn tier_2_answers_without_a_cap_as_a_chat_answer_does() {
    // A chat's answer has no cap by default, and nor does a memory answer: 600 tokens are not
    // cut, past the 100 a Qwen3 step thinks.
    let fake = Fake::new(32768);
    let mut set = memory_set();
    set.questions.truncate(1);
    fake.say_tokens(&[fake.token("x"); 600]);
    let m = bench::run_memory(
        Some((&fake.larch, Family::Qwen3, 0)),
        Thinking::default(),
        &set,
        |_| {},
    );
    let row = &m.questions[0];
    assert_eq!(row.cut, None, "{row:?}");
    assert_eq!(row.answer.as_deref(), Some("x".repeat(600).as_str()));
}

#[test]
fn tier_2_scores_the_planted_value_in_the_answer() {
    let fake = Fake::new(32768);
    let mut set = memory_set();
    // The June boiler question, too long for the context with its hits, comes last.
    let mut boiler = set
        .questions
        .iter()
        .find(|q| q.id == "m08")
        .unwrap()
        .clone();
    boiler.question += &" word".repeat(40_000);
    set.questions.truncate(3);
    set.questions.push(boiler);
    // The address, case aside; a wrong company; an answer cut at the token limit.
    fake.say("You live at 41 alder street, flat 3.");
    fake.say("It is booked with Swift Vans.");
    fake.say_tokens(&[fake.token("x"); 600]);
    // A cap on answers, so the last one is cut.
    let capped = Thinking {
        answer_max_tokens: 512,
        ..Thinking::default()
    };
    let m = bench::run_memory(Some((&fake.larch, Family::Llama3, 0)), capped, &set, |_| {});
    let rows: Vec<_> = (m.questions.iter())
        .map(|r| (r.found, r.holds_value, r.cut, r.error.is_some()))
        .collect();
    assert_eq!(
        rows,
        [
            (true, Some(true), None, false),
            (true, Some(false), None, false),
            (true, Some(false), Some(Cut::Text), false),
            (true, None, None, true),
        ]
    );
    // The engine refused the last question's model step, not its recall, which still counts.
    assert_eq!((m.recall_at_5, m.in_range), (Some(1.0), Some(1.0)));
    assert_eq!(
        m.questions[0].answer.as_deref(),
        Some("You live at 41 alder street, flat 3.")
    );
    assert_eq!(m.holds_value, Some(1.0 / 3.0));
    // The model got each question with recall's hits.
    let fed = fake.fed();
    let hit = "agent/memory/facts/my-new-address-from-8-june.md";
    assert!(
        fed.contains(&set.questions[1].question) && fed.contains(hit),
        "{fed}"
    );
}

/// Whether text `v` (or each path in a list) is in an earlier step's result and not in the user's
/// text, case aside: after a listing, a listed name under the listed folder, as the file tools take
/// a relative path from the first allowed folder, not from the folder listed.
fn from_results(earlier: &[ScriptedStep], user: &str, v: &serde_json::Value) -> bool {
    let found = |v: &str| {
        let new = !user.to_lowercase().contains(&v.to_lowercase());
        new && earlier.iter().any(|s| match s.expect.tool.as_deref() {
            Some("list_dir") => {
                let folder = s.expect.args_required["path"].as_str().unwrap();
                (s.result.lines())
                    .any(|name| v == format!("{folder}/{}", name.trim_end_matches('/')))
            }
            _ => s.result.contains(v),
        })
    };
    match v {
        serde_json::Value::Array(items) => items.iter().all(|i| i.as_str().is_some_and(found)),
        v => v.as_str().is_some_and(found),
    }
}

fn multistep() -> Vec<StepPrompt> {
    bench::load_steps(&agent_dir().join("multistep.jsonl"))
        .unwrap()
        .0
}

fn step_row(id: &str) -> StepPrompt {
    multistep().into_iter().find(|r| r.id == id).unwrap()
}

#[test]
fn the_multistep_rows_parse_and_hold_the_mix() {
    let (rows, file) = bench::load_steps(&agent_dir().join("multistep.jsonl")).unwrap();
    assert_eq!((rows.len(), file.sha256.len()), (50, 64));
    let registry = Registry::builtin();
    let mut kinds = BTreeMap::new();
    for r in &rows {
        let (kind, n) = r.id.split_once('-').unwrap();
        let (count, steps) = kinds.entry(kind).or_insert((0, BTreeSet::new()));
        *count += 1;
        assert_eq!(n, format!("{count:03}"), "{} is out of order", r.id);
        steps.insert(r.steps.len());
        assert_eq!(r.r#final.tool, None, "{}", r.id);
        for s in &r.steps {
            let tool = s.expect.tool.as_deref().unwrap();
            assert!(
                r.tools.iter().any(|t| t == tool),
                "{}: {tool} is not shown",
                r.id
            );
            assert!(
                registry.get(tool).is_some(),
                "{}: {tool} is not a tool",
                r.id
            );
            assert!(!s.result.is_empty(), "{}", r.id);
        }
        if kind == "chain" {
            // The second call needs what the first one found.
            let mut args = r.steps[1].expect.args_required.values();
            assert!(
                args.any(|v| from_results(&r.steps[..1], &r.user, v)),
                "{}",
                r.id
            );
        }
    }
    let mix = |n, steps| (n, BTreeSet::from([steps]));
    assert_eq!(
        kinds,
        BTreeMap::from([
            ("chain", mix(15, 2)),
            ("error", mix(5, 1)),
            ("one", mix(20, 1)),
            ("parallel", mix(10, 3)),
        ])
    );
}

#[test]
fn the_chat_multistep_rows_show_the_chat_tools_and_take_each_last_call_from_a_result() {
    let (rows, _) = bench::load_steps(&agent_dir().join("routing/multistep.jsonl")).unwrap();
    let registry = Registry::builtin();
    let mut kinds = BTreeMap::new();
    for r in &rows {
        assert_eq!(r.tools, larch_agent::CHAT_TOOLS, "{}", r.id);
        let (kind, n) = r.id.split_once('-').unwrap();
        let count = kinds.entry(kind).or_insert(0);
        *count += 1;
        assert_eq!(n, format!("{count:03}"), "{} is out of order", r.id);
        let calls = (r
            .steps
            .iter()
            .map(|s| (&s.expect.tool, &s.expect.args_required)))
        .chain([(&r.r#final.tool, &r.r#final.args_required)]);
        for (tool, args) in calls.filter_map(|(t, a)| Some((t.as_ref()?, a))) {
            let spec = &registry.specs(&[tool.as_str()])[0];
            for key in args.keys() {
                assert!(
                    spec.schema["properties"].get(key).is_some(),
                    "{}: {tool} has no {key}",
                    r.id
                );
            }
        }
        assert!(r.steps.iter().all(|s| !s.result.is_empty()), "{}", r.id);
        // The last call needs what an earlier one found, not what the user said.
        let (last, earlier) = match &r.r#final.tool {
            Some(_) => (&r.r#final.args_required, &r.steps[..]),
            None => {
                let (last, earlier) = r.steps.split_last().unwrap();
                (&last.expect.args_required, earlier)
            }
        };
        let mut args = last.values();
        assert!(args.any(|v| from_results(earlier, &r.user, v)), "{}", r.id);
    }
    let want = [
        ("find", 5),
        ("fix", 5),
        ("papers", 4),
        ("recall", 3),
        ("research", 3),
        ("web", 6),
        ("write", 4),
    ];
    assert_eq!(kinds, BTreeMap::from(want));
}

#[test]
fn a_two_step_row_scores_each_step_and_fails_on_the_other_tool() {
    let fake = Fake::new(32768);
    // List the invoices, read the newest one the listing names, answer.
    let row = step_row("chain-001");
    fake.say(&call("list_dir", json!({"path": "invoices"})));
    fake.say(&call("read_file", json!({"path": "invoices/2024-07.csv"})));
    fake.say("July came to 412.50 EUR.");
    // The second call is valid but to the other tool.
    fake.say(&call("list_dir", json!({"path": "invoices"})));
    fake.say(&call(
        "write_file",
        json!({"path": "invoices/2024-07.csv", "text": "x"}),
    ));
    // The second step's call first, without the listing it needs.
    fake.say(&call("read_file", json!({"path": "invoices/2024-07.csv"})));
    let mut seen = Vec::new();
    let rows = [row.clone(), row.clone(), row];
    let rows = bench::run_steps(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &rows,
        |r| seen.push(r.id.clone()),
    );
    assert_eq!(fake.unsaid(), 0);
    assert_eq!(seen, ["chain-001"; 3]);
    let scores: Vec<_> = rows
        .iter()
        .map(|r| {
            let s = (r.steps.clone(), r.answered, r.wrong_but_valid);
            (s, r.steps_used, r.right())
        })
        .collect();
    assert_eq!(
        scores,
        [
            ((vec![true, true], true, false), 3, true),
            ((vec![true, false], false, true), 2, false),
            ((vec![false, false], false, true), 1, false),
        ]
    );
    for r in &rows {
        assert_eq!(r.error, None);
        assert!(r.generated_tokens > 0 && r.visible_ms.is_some(), "{r:?}");
    }
    // Each model step's call or answer, so a Mac run shows what a wrong row did.
    assert_eq!(
        rows[0].replies,
        [
            r#"list_dir {"path":"invoices"}"#,
            r#"read_file {"path":"invoices/2024-07.csv"}"#,
            "July came to 412.50 EUR.",
        ]
    );
    assert_eq!(
        rows[1].replies[1],
        r#"write_file {"path":"invoices/2024-07.csv","text":"x"}"#
    );
    // Each matched call got its step's result; the wrong one got none.
    let fed = fake.fed();
    assert_eq!(fed.matches("2024-06.csv").count(), 2, "{fed}");
    assert_eq!(fed.matches("July: 3 items").count(), 1, "{fed}");
    let summary = bench::step_summary(&rows);
    assert_eq!(
        (
            summary.rows,
            summary.errors,
            summary.right_rate,
            summary.steps
        ),
        (3, 0, 1.0 / 3.0, 2.0)
    );
    assert_eq!(summary.counters.wrong_but_valid, 2);
    assert_eq!(summary.calls_in_one_step, None);
}

#[test]
fn a_last_skill_call_scores_its_arguments_and_a_list_as_its_set_of_paths() {
    let fake = Fake::new(32768);
    // The app ends the turn on a skill's answer, so the skill call is the row's last step.
    let row: StepPrompt = serde_json::from_value(json!({
        "id": "papers-002",
        "tools": ["list_dir", "read_file", "read_papers", "web_answer"],
        "user": "Compare the two mixture-of-experts papers in research.",
        "steps": [{"expect": {"tool": "list_dir", "args_required": {"path": "research"}},
                   "result": "dense-scaling.pdf\nmixtral-moe.pdf\nswitch-moe.pdf"}],
        "final": {"tool": "read_papers", "args_required":
                  {"papers": ["research/mixtral-moe.pdf", "research/switch-moe.pdf"]}}
    }))
    .unwrap();
    let papers = |p: &[&str]| call("read_papers", json!({ "papers": p }));
    let (mixtral, switch) = ("research/mixtral-moe.pdf", "research/switch-moe.pdf");
    // (reply, answered, wrong_but_valid)
    let replies = [
        (papers(&[mixtral, switch]), true, false),
        (papers(&["./Research/Switch-MoE.pdf", mixtral]), true, false),
        (papers(&[switch]), false, true),
        (
            papers(&["research/dense-scaling.pdf", mixtral, switch]),
            false,
            true,
        ),
        (papers(&[mixtral, mixtral]), false, true),
        (
            call("web_answer", json!({"question": "mixtral"})),
            false,
            true,
        ),
        // An answer from memory, with no skill call, is not the row's last step either.
        (
            "Mixtral routes each token to two experts.".into(),
            false,
            false,
        ),
    ];
    for (reply, ..) in &replies {
        fake.say(&call("list_dir", json!({"path": "research"})));
        fake.say(reply);
    }
    let rows = bench::run_steps(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &vec![row; replies.len()],
        |_| {},
    );
    assert_eq!(fake.unsaid(), 0);
    for (r, (reply, answered, wrong)) in rows.iter().zip(&replies) {
        assert_eq!(r.error, None);
        let got = (r.steps.clone(), r.answered, r.wrong_but_valid);
        assert_eq!(got, (vec![true], *answered, *wrong), "{reply}");
    }
}

#[test]
fn three_reads_score_in_any_order_and_a_reply_with_several_calls_is_marked() {
    let fake = Fake::new(32768);
    let row = step_row("parallel-001");
    let read = |day: &str| call("read_file", json!({ "path": format!("notes/{day}.txt") }));
    // All three in the first reply, out of order: the first runs, the model hears the others did
    // not, and makes them one by one.
    fake.say(&format!(
        "{}; {}; {}",
        read("wed"),
        read("mon"),
        read("tue")
    ));
    fake.say(&read("mon"));
    fake.say(&read("tue"));
    fake.say("You booked the dentist on Tuesday.");
    // One per reply, in another order.
    for day in ["tue", "wed", "mon"] {
        fake.say(&read(day));
    }
    fake.say("Tuesday.");
    let rows = bench::run_steps(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &[row.clone(), row],
        |_| {},
    );
    assert_eq!(fake.unsaid(), 0);
    let scores: Vec<_> = rows
        .iter()
        .map(|r| (r.right(), r.steps_used, r.calls_in_one_step))
        .collect();
    assert_eq!(scores, [(true, 4, Some(true)), (true, 4, Some(false))]);
    let fed = fake.fed();
    let note = ParseError::MultipleCalls.to_string();
    assert_eq!(fed.matches(&note).count(), 1, "{fed}");
    assert_eq!(bench::step_summary(&rows).calls_in_one_step, Some(0.5));
}

/// In the Qwen format the reply's calls that fit open steps run together, as the chat runs them:
/// a three-file row takes two steps, and nobody hears of dropped calls.
#[test]
fn a_qwen_reply_of_three_reads_is_one_step() {
    let fake = Fake::qwen3(32768);
    let read = |day: &str| {
        let call = json!({"name": "read_file", "arguments": {"path": format!("notes/{day}.txt")}});
        format!("<tool_call>\n{call}\n</tool_call>")
    };
    fake.say(&format!(
        "{}\n{}\n{}",
        read("wed"),
        read("mon"),
        read("tue")
    ));
    fake.say("You booked the dentist on Tuesday.");
    let rows = bench::run_steps(
        &fake.larch,
        Family::Qwen3,
        Thinking::default(),
        0,
        &[step_row("parallel-001")],
        |_| {},
    );
    assert_eq!(fake.unsaid(), 0);
    let r = &rows[0];
    assert_eq!(
        (r.right(), r.steps_used, r.calls_in_one_step),
        (true, 2, Some(true)),
        "{r:?}"
    );
    assert!(!fake.fed().contains(&ParseError::MultipleCalls.to_string()));
}

#[test]
fn every_parallel_row_counts_in_calls_in_one_step_by_its_first_reply() {
    let fake = Fake::new(32768);
    let row = step_row("parallel-001");
    let read = |day: &str| call("read_file", json!({ "path": format!("notes/{day}.txt") }));
    let all = format!("{}; {}; {}", read("mon"), read("tue"), read("wed"));
    let rest = |fake: &Fake| {
        fake.say(&read("tue"));
        fake.say(&read("wed"));
        fake.say("Tuesday.");
    };
    // All three in the first reply.
    fake.say(&all);
    rest(&fake);
    // Eight answers with no call.
    for _ in 0..8 {
        fake.say("I cannot tell.");
    }
    // A broken first reply, then all three in the retry: the first reply held one.
    fake.say("{\"name\": \"read_file\", \"parameters\": {\"path\": ");
    fake.say(&all);
    rest(&fake);
    let rows = bench::run_steps(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &vec![row; 10],
        |_| {},
    );
    assert_eq!(fake.unsaid(), 0);
    let mut want = vec![Some(false); 10];
    want[0] = Some(true);
    let marks: Vec<_> = rows.iter().map(|r| r.calls_in_one_step).collect();
    assert_eq!(marks, want);
    assert!(rows[0].right() && rows[9].right(), "{rows:?}");
    assert_eq!(bench::step_summary(&rows).calls_in_one_step, Some(0.1));
}

#[test]
fn a_chained_read_names_the_listed_folder_not_a_bare_name_or_another_folder() {
    let fake = Fake::new(32768);
    let row = step_row("chain-001");
    // The file tools take a relative path from the first allowed folder, whatever was listed.
    let paths = [
        ("invoices/2024-07.csv", true),
        ("./Invoices/2024-07.csv", true),
        ("2024-07.csv", false),
        ("unrelated/2024-07.csv", false),
        ("old/invoices/2024-07.csv", false),
        // A file's path with a trailing slash fails in the tool.
        ("invoices/2024-07.csv/", false),
    ];
    for (path, right) in paths {
        fake.say(&call("list_dir", json!({"path": "invoices"})));
        fake.say(&call("read_file", json!({ "path": path })));
        if right {
            fake.say("July came to 412.50 EUR.");
        }
    }
    let rows = bench::run_steps(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &vec![row; paths.len()],
        |_| {},
    );
    assert_eq!(fake.unsaid(), 0);
    for ((path, right), r) in paths.iter().zip(&rows) {
        let want = (vec![true, *right], *right, !*right);
        assert_eq!(
            (r.steps.clone(), r.answered, r.wrong_but_valid),
            want,
            "{path}"
        );
    }
}

#[test]
fn a_trailing_slash_fits_a_folder_step_and_not_a_file_step() {
    // `projects/` lists the folder; `notes/todo.txt/` and `notes/reminder.txt/` are not folders,
    // so the read and the write would fail.
    let fake = Fake::new(32768);
    fake.say(&call("list_dir", json!({"path": "projects/"})));
    fake.say("It holds garden and kitchen.");
    fake.say(&call("read_file", json!({"path": "notes/todo.txt/"})));
    fake.say(&call(
        "write_file",
        json!({"path": "notes/reminder.txt/", "text": "pick up the keys at 5"}),
    ));
    let rows = ["one-007", "one-001", "one-015"].map(step_row);
    let rows = bench::run_steps(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &rows,
        |_| {},
    );
    assert_eq!(fake.unsaid(), 0);
    let scores: Vec<_> = rows
        .iter()
        .map(|r| (r.id.as_str(), r.steps.clone(), r.right(), r.wrong_but_valid))
        .collect();
    assert_eq!(
        scores,
        [
            ("one-007", vec![true], true, false),
            ("one-001", vec![false], false, true),
            ("one-015", vec![false], false, true),
        ]
    );
}

/// A folder step takes its trailing `/` under `--rename-tools` too: the step's tool is the
/// stand-in's, and the folder rule is the real tool's.
#[test]
fn a_trailing_slash_fits_a_renamed_folder_step() {
    let fake = Fake::new(32768);
    fake.say(&json!({"name": "qv7_tarn", "parameters": {"path": "projects/"}}).to_string());
    fake.say("It holds garden and kitchen.");
    let routing = bench::Routing {
        rename: true,
        ..bench::Routing::default()
    };
    let rows = bench::run_steps_routed(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &routing,
        &[step_row("one-007")],
        |_| {},
    );
    assert_eq!(fake.unsaid(), 0);
    assert_eq!(rows[0].steps, [true], "{:?}", rows[0]);
    assert!(rows[0].right(), "{:?}", rows[0]);
}

#[test]
fn a_cut_step_and_its_retry_count_though_an_engine_error_ends_the_row() {
    // A call cut at the token limit, then a retry the context has no room for.
    let cut_call = |fake: &Fake| {
        let start = "{\"name\": \"read_file\", \"parameters\": {\"path\": \"";
        let mut ids: Vec<u32> = start.bytes().map(|b| fake.byte(b)).collect();
        ids.extend([fake.byte(b'a'); 3000]);
        fake.say_tokens(&ids);
    };
    // Room for the first step only: its prompt and a step's tokens, from a plain answer cut at
    // the limit, and a little.
    let plain = |fake: &Fake| fake.say_tokens(&[fake.byte(b'a'); 3000]);
    let probe = Fake::new(32768);
    plain(&probe);
    let p = bench::run(
        &probe.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &[read_todo("clear-001")],
        |_| {},
    );
    assert_eq!(
        (p[0].cut.as_slice(), &p[0].error),
        (&[Cut::Text][..], &None)
    );
    let fake = Fake::new(p[0].prompt_tokens + p[0].generated_tokens + 32);
    cut_call(&fake);
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &[read_todo("clear-001")],
        |_| {},
    );
    assert!(rows[0].error.is_some() && rows[0].retried, "{:?}", rows[0]);
    assert_eq!(rows[0].cut, [Cut::Call]);
    let s = bench::summary(&rows);
    let c = s.counters.unwrap();
    assert_eq!(
        (s.errors, c.length_stops, c.in_call, c.retries),
        (1, 1, 1, 1)
    );
    // The same on a multi-step row.
    let row = step_row("one-001");
    let probe = Fake::new(32768);
    plain(&probe);
    let p = bench::run_steps(
        &probe.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        std::slice::from_ref(&row),
        |_| {},
    );
    assert_eq!(
        (p[0].cut.as_slice(), &p[0].error),
        (&[Cut::Text][..], &None)
    );
    let fake = Fake::new(p[0].prompt_tokens + p[0].generated_tokens + 32);
    cut_call(&fake);
    let rows = bench::run_steps(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &[row],
        |_| {},
    );
    assert!(
        rows[0].error.is_some() && rows[0].retries == 1,
        "{:?}",
        rows[0]
    );
    let s = bench::step_summary(&rows);
    let c = s.counters;
    assert_eq!(
        (s.errors, c.length_stops, c.in_call, c.retries),
        (1, 1, 1, 1)
    );
}

#[test]
fn a_bench_with_the_cache_off_sends_no_anchor() {
    let fake = Fake::uncached(32768);
    let inputs = bench::Inputs {
        prompts: vec![prompt(
            "none-001",
            &CHAT,
            "What is 2 and 2?",
            None,
            json!({}),
        )],
        steps: None,
        docs: None,
        memory: None,
    };
    fake.say("4.");
    let mut report = Report::default();
    let mut lines = Vec::new();
    bench::run_all(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &bench::Routing::default(),
        &inputs,
        &mut report,
        || fake.engine.prefix_cache_stats(),
        |_| Ok(()),
        |l| lines.push(l),
    )
    .unwrap();
    assert_eq!(report.anchor_ms, None);
    assert!(lines[0].starts_with("none-001 "), "{lines:?}");
    // The row's prompt is all the engine computed.
    let row = report.rows[0].prompt_tokens;
    assert_eq!(
        report.prefix_cache[0].computed_prompt_tokens,
        u64::from(row)
    );
}

#[test]
fn renamed_tools_show_the_model_other_names_and_score_by_the_real_ones() {
    assert_eq!(bench::shown_name("list_dir"), "qv7_tarn");
    assert_eq!(bench::real_name("qv7_tarn"), "list_dir");
    // A name with no stand-in, such as an unknown tool's, stays as it is both ways.
    assert_eq!(bench::shown_name("open"), "open");
    assert_eq!(bench::real_name("open"), "open");
    let fake = Fake::new(32768);
    let inputs = bench::Inputs {
        prompts: vec![prompt(
            "files-001",
            &CHAT,
            "What is in taxes?",
            Some("list_dir"),
            json!({"path": "taxes"}),
        )],
        steps: None,
        docs: None,
        memory: None,
    };
    // The row: a call by the name the model was shown.
    fake.say(&json!({"name": "qv7_tarn", "parameters": {"path": "taxes"}}).to_string());
    let routing = bench::Routing {
        rename: true,
        ..bench::Routing::default()
    };
    let mut report = Report::default();
    bench::run_all(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &routing,
        &inputs,
        &mut report,
        || fake.engine.prefix_cache_stats(),
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    let row = &report.rows[0];
    assert_eq!(row.tool.as_deref(), Some("list_dir"), "{row:?}");
    assert!(row.before.right_tool && row.before.args_ok, "{row:?}");
    // The model saw the stand-ins in the tool list, never the real names.
    let fed = fake.fed();
    assert!(fed.contains("\"qv7_tarn\"") && fed.contains("\"wx2_plom\""));
    assert!(!fed.contains("\"list_dir\"") && !fed.contains("\"read_file\""));
    assert_eq!(fake.unsaid(), 0);
}

#[test]
fn extra_tools_follow_each_rows_own_and_a_call_to_one_is_the_wrong_tool() {
    let fake = Fake::new(32768);
    let row = |id| {
        prompt(
            id,
            &CHAT,
            "What is in taxes?",
            Some("list_dir"),
            json!({"path": "taxes"}),
        )
    };
    let inputs = bench::Inputs {
        prompts: vec![row("files-001"), row("files-002")],
        steps: None,
        docs: None,
        memory: None,
    };
    fake.say(&json!({"name": "list_dir", "parameters": {"path": "taxes"}}).to_string());
    fake.say(
        &json!({"name": "calendar_list_events", "parameters": {"start": "a", "end": "b"}})
            .to_string(),
    );
    let routing = bench::Routing {
        extra_tools: 2,
        ..bench::Routing::default()
    };
    let mut report = Report::default();
    bench::run_all(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &routing,
        &inputs,
        &mut report,
        || fake.engine.prefix_cache_stats(),
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    assert!(report.rows[0].before.right_tool, "{:?}", report.rows[0]);
    // The second row's call is a tool the list holds, and not the row's.
    let row = &report.rows[1];
    assert_eq!(row.tool.as_deref(), Some("calendar_list_events"), "{row:?}");
    assert!(!row.before.right_tool, "{row:?}");
    // The model saw the first two extra tools after the chat's, and no third.
    let fed = fake.fed();
    assert!(fed.contains("\"calendar_list_events\"") && fed.contains("\"calendar_create_event\""));
    assert!(!fed.contains("\"calendar_update_event\""));
    assert!(fed.find("\"list_dir\"") < fed.find("\"calendar_list_events\""));
}

#[test]
fn a_line_number_written_as_a_string_scores_as_the_call_the_loop_runs() {
    let fake = Fake::new(32768);
    // One tool row: the model writes `start_line` as "2"; the loop runs it with 2.
    let prompts = vec![prompt(
        "files-002",
        &CHAT,
        "What is on line 2 of a.txt?",
        Some("read_file"),
        json!({"path": "a.txt", "start_line": 2}),
    )];
    fake.say(&call(
        "read_file",
        json!({"path": "a.txt", "start_line": "2"}),
    ));
    let rows = bench::run(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &prompts,
        |_| {},
    );
    let row = &rows[0];
    assert!(
        row.before.args_ok && row.after.args_ok && !row.retried,
        "{row:?}"
    );
    assert!(!row.wrong_but_valid, "{row:?}");
    // A multi-step row's step matches the coerced call too, so it gets its result.
    let steps: bench::StepPrompt = serde_json::from_value(json!({
        "id": "one-901",
        "tools": CHAT,
        "user": "What is on line 2 of a.txt?",
        "steps": [{
            "expect": {"tool": "read_file", "args_required": {"path": "a.txt", "start_line": 2}},
            "result": "2: the second line"
        }],
        "final": {"tool": null}
    }))
    .unwrap();
    fake.say(&call(
        "read_file",
        json!({"path": "a.txt", "start_line": "2"}),
    ));
    fake.say("It says the second line.");
    let rows = bench::run_steps(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &[steps],
        |_| {},
    );
    let row = &rows[0];
    assert_eq!(
        (row.steps.clone(), row.answered),
        (vec![true], true),
        "{row:?}"
    );
    assert!(row.right() && !row.wrong_but_valid, "{row:?}");
    assert!(fake.fed().contains("2: the second line"));
    assert_eq!(fake.unsaid(), 0);
}

#[test]
fn a_gated_row_records_what_its_gate_read_from_the_prefix_cache() {
    let fake = Fake::new(32768);
    let inputs = bench::Inputs {
        prompts: vec![prompt(
            "none-001",
            &CHAT,
            "What is 2 and 2?",
            None,
            json!({}),
        )],
        steps: None,
        docs: None,
        memory: None,
    };
    // The gate's one token, then the row's reply.
    fake.say_tokens(&[fake.token("H")]);
    fake.say("4.");
    let routing = bench::Routing {
        router: Router::Logprob,
        ..bench::Routing::default()
    };
    let mut report = Report::default();
    bench::run_all(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &routing,
        &inputs,
        &mut report,
        || fake.engine.prefix_cache_stats(),
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    let row = &report.rows[0];
    let routed = row.routed.unwrap();
    // The gate's prompt is the step's, and it reads the anchored blocks the step reads.
    assert!(row.cached_tokens > 0, "{row:?}");
    assert_eq!(
        routed.gate_prompt_tokens,
        Some(row.prompt_tokens),
        "{row:?}"
    );
    assert_eq!(
        routed.gate_cached_tokens,
        Some(row.cached_tokens),
        "{row:?}"
    );
    // Its time is part of the row's, and not of the step's time to first token.
    let ms = routed.gate_ms.unwrap();
    assert!(ms > 0.0 && ms <= row.wall_ms, "{row:?}");
    assert_eq!(fake.unsaid(), 0);
}

/// The arguments of a `tool` call with `expected`'s values in the tool's schema order, and a
/// value of its type for each other required one, as the call grammar takes them.
fn schema_args(tool: &str, expected: &serde_json::Map<String, serde_json::Value>) -> String {
    let spec = &Registry::builtin().specs(&[tool])[0];
    let required = spec.schema["required"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut args = serde_json::Map::new();
    for (key, schema) in spec.schema["properties"].as_object().unwrap() {
        let value = match expected.get(key).filter(|v| !v.is_null()) {
            Some(v) => v.clone(),
            None if required.contains(&json!(key)) => match &schema["enum"] {
                serde_json::Value::Array(e) => e[0].clone(),
                _ if schema["type"] == "integer" => json!(1),
                _ => json!("x"),
            },
            None => continue,
        };
        args.insert(key.clone(), value);
    }
    serde_json::Value::Object(args).to_string()
}

/// Each toolcalls row through the fake model, whose every call is cut just before its last
/// brace: with the grammar the first reply of every row parses, so the gate holds with the retry
/// off (`parse_rate_before`); without it, no call row does.
#[test]
fn every_toolcalls_row_parses_on_its_first_reply_with_the_grammar() {
    let (prompts, _) = bench::load(&agent_dir().join("toolcalls.jsonl")).unwrap();
    for on in [true, false] {
        let fake = Fake::qwen3(32768);
        let routing = bench::Routing {
            tool_grammar: on,
            ..bench::Routing::default()
        };
        let mut rows = Vec::new();
        for prompt in &prompts {
            match &prompt.expect.tool {
                Some(tool) => {
                    let args = schema_args(tool, &prompt.expect.args_required);
                    fake.say_tokens(&[fake.token("<tool_call>")]);
                    fake.say(&format!("\n{{\"name\": \"{tool}\", \"arguments\": {args}"));
                }
                None => fake.say("Done."),
            }
            let one = std::slice::from_ref(prompt);
            let thinking = Thinking::default();
            rows.extend(bench::run_routed(
                &fake.larch,
                Family::Qwen3,
                thinking,
                0,
                &routing,
                one,
                |_| {},
            ));
            // The grammar's last forced picks and the row's retry say what the fake was not told.
            fake.settle();
        }
        let summary = bench::summary(&rows);
        assert_eq!(summary.errors, 0, "{on}");
        let answers = prompts.iter().filter(|p| p.expect.tool.is_none()).count();
        let share = answers as f64 / prompts.len() as f64;
        let want = if on { 1.0 } else { share };
        assert_eq!(summary.parse_rate_before, want, "{on}");
    }
}

/// The bench's rows with the chat's tool list, as a turn runs them.
fn turn_row(id: &str, history: serde_json::Value, user: &str, tool: Option<&str>) -> BenchPrompt {
    BenchPrompt {
        history: serde_json::from_value(history).unwrap(),
        ..prompt(id, &larch_agent::CHAT_TOOLS, user, tool, json!({}))
    }
}

/// Runs `rows` as chat turns (`--turn`), with the call grammar on as the app has it.
fn turns(fake: &Fake, rows: Vec<BenchPrompt>) -> Report {
    let inputs = bench::Inputs {
        prompts: rows,
        steps: None,
        docs: None,
        memory: None,
    };
    let routing = bench::Routing {
        turn: true,
        arch: "llama".into(),
        tool_grammar: true,
        ..bench::Routing::default()
    };
    let mut report = Report::default();
    bench::run_all(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &routing,
        &inputs,
        &mut report,
        || fake.engine.prefix_cache_stats(),
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    report
}

/// A `--turn` row runs as a chat's turn after its planted history: the model reads the agent's
/// own prompt and the turn's tool note, and the row is scored on the turn's first call, which the
/// bench never runs (v3 D8: the routing rows measure what ships).
#[test]
fn a_turn_row_runs_as_a_chat_turn_and_scores_its_first_call() {
    let fake = Fake::new(32768);
    let row = turn_row(
        "routing-notools-web-001",
        json!([
            {"role": "user", "content": "explain how BERT works, without using any tools", "truncated": false},
            {"role": "assistant", "content": "BERT is a transformer encoder.", "truncated": false}
        ]),
        "now search the web for the newest BERT variant",
        Some("web_answer"),
    );
    // "search the web" holds the step to web_answer, so the call is the grammar's spacing.
    fake.say(&spaced(
        "web_answer",
        json!({"question": "the newest BERT variant"}),
    ));
    let report = turns(&fake, vec![row]);
    let r = &report.rows[0];
    assert_eq!(r.error, None);
    assert_eq!(r.tool.as_deref(), Some("web_answer"));
    assert!(r.after.right_tool && r.after.args_ok, "{r:?}");
    assert!(!r.withdrawn);
    let fed = fake.fed();
    assert!(fed.contains("explain how BERT works"), "{fed}");
    assert!(
        fed.contains("[agent] Tools are on for this message."),
        "{fed}"
    );
    assert!(fed.contains("Today is"), "{fed}");
}

/// A `--turn` row shaped as bench/agent/no-folders.jsonl's (#400), its tools without the file
/// tools, runs as a chat with nothing attached: no folder in its prompt, and the fixed answer
/// that says to attach one is right, with no call; a row with the file tools still has the home
/// folder.
#[test]
fn a_turn_row_without_the_file_tools_runs_with_nothing_attached() {
    let line = r#"{"id": "nofolders-files-001", "tools": ["recall", "remember", "forget", "web_answer", "read_papers", "research"], "user": "what do my notes say about the trip to Lisbon?", "expect": {"tool": null, "args_required": {}, "answer_must_contain": ["attach a folder"]}}"#;
    let row: BenchPrompt = serde_json::from_str(line).unwrap();
    assert!(bench::chat_tools(&row.tools));
    // Every row of the real set passes --turn's check (it refused them all on the Mac).
    let (rows, _) = bench::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/agent/no-folders.jsonl"),
    )
    .unwrap();
    assert!(rows.iter().all(|r| bench::chat_tools(&r.tools)));
    let mut some = row.tools.clone();
    some.pop();
    assert!(!bench::chat_tools(&some));
    let fake = Fake::new(32768);
    fake.say(
        "I can't read your files in this chat yet. Attach a folder or the files, then ask again.",
    );
    let report = turns(&fake, vec![row]);
    let r = &report.rows[0];
    assert_eq!((r.error.as_deref(), r.tool.as_deref()), (None, None));
    assert!(r.after.right_tool && r.wrong_answer.is_none(), "{r:?}");
    let fed = fake.fed();
    assert!(!fed.contains("You may read and write files under"), "{fed}");
    // A row with the chat's file tools has the home folder, as before.
    let fake = Fake::new(32768);
    fake.say("Fine, thanks.");
    let row = turn_row("routing-none-001", json!([]), "how are you?", None);
    let report = turns(&fake, vec![row]);
    assert_eq!(report.rows[0].error, None);
    let fed = fake.fed();
    assert!(fed.contains("You may read and write files under"), "{fed}");
}

/// A reply a check takes back (D5) counts as withdrawn, and the row scores the call that the
/// message went again with.
#[test]
fn a_turn_row_counts_a_withdrawn_reply_and_scores_the_call_after_it() {
    let fake = Fake::new(32768);
    let row = turn_row(
        "routing-offline-web-001",
        json!([]),
        // Not about the latest, which a first message would force.
        "who won the F1 race in Austin?",
        Some("web_answer"),
    );
    // A paced engine runs at most one pass after the cancel; it takes the newline the grammar
    // lets a forced call start with, or the forced step does.
    fake.pace(Duration::from_millis(20));
    fake.say_tokens(&fake.tokenizer.encode("I can't access the internet.", false));
    fake.say_tokens(&fake.tokenizer.encode("\n", false));
    fake.say(&spaced(
        "web_answer",
        json!({"question": "the last F1 race"}),
    ));
    let report = turns(&fake, vec![row]);
    fake.pace(Duration::ZERO);
    let r = &report.rows[0];
    assert_eq!(r.error, None);
    assert!(r.withdrawn, "{r:?}");
    assert_eq!(r.tool.as_deref(), Some("web_answer"));
    assert_eq!(report.summary.counters.unwrap().withdrawn, 1);
    // The speed is the first step that ran to its end, not the withdrawn one; the tokens are
    // every step's.
    assert!(r.ttft_ms > 0.0 && r.tok_s > 0.0, "{r:?}");
    assert!(r.prompt_tokens > 0 && r.generated_tokens > 0, "{r:?}");
}

/// A plain answer is scored as the turn shows it, with the row's answer checks.
#[test]
fn a_turn_row_scores_a_plain_answer() {
    let fake = Fake::new(32768);
    let row = |id| {
        let mut row = turn_row(id, json!([]), "What is 2 and 2?", None);
        row.expect.answer_must_contain = vec!["4".into()];
        row
    };
    fake.say("4.");
    fake.say("Five.");
    let report = turns(&fake, vec![row("math-none-001"), row("math-none-002")]);
    let [right, wrong] = &report.rows[..] else {
        panic!("{:?}", report.rows)
    };
    assert_eq!(
        (right.error.as_deref(), right.tool.as_deref()),
        (None, None)
    );
    assert!(
        right.after.right_tool && right.wrong_answer.is_none(),
        "{right:?}"
    );
    assert!(wrong.wrong_answer.is_some(), "{wrong:?}");
}

/// A turn that ends in a failure, with no answer, is the row's error, not an empty answer.
#[test]
fn a_turn_that_fails_is_an_error_not_an_answer() {
    // The system prompt and the tool list alone are longer than this context.
    let fake = Fake::new(1024);
    let row = turn_row("math-none-001", json!([]), "What is 2 and 2?", None);
    let report = turns(&fake, vec![row]);
    let r = &report.rows[0];
    assert!(r.error.is_some(), "{r:?}");
    assert_eq!(report.summary.errors, 1);
}

/// A planted history carries the turn notes its messages would have had, so a kept limit holds
/// as it would in the app.
#[test]
fn a_planted_history_carries_its_turn_notes() {
    let fake = Fake::new(32768);
    let row = turn_row(
        "routing-nosearch-none-009",
        json!([
            {"role": "user", "content": "from now on, no tools", "truncated": false},
            {"role": "assistant", "content": "OK.", "truncated": false}
        ]),
        "What is 2 and 2?",
        None,
    );
    fake.say("4.");
    let report = turns(&fake, vec![row]);
    assert_eq!(report.rows[0].error, None);
    let kept = "[agent] No tools until the user turns them back on: the user asked.";
    let fed = fake.fed();
    assert!(
        fed.contains(&format!("from now on, no tools\n\n{kept}")),
        "{fed}"
    );
    assert!(
        fed.contains(&format!("What is 2 and 2?\n\n{kept}")),
        "{fed}"
    );
}

/// The multi-step rows of a `--turn` run still start from their anchored prompt: only the tool
/// rows' chats anchor their own.
#[test]
fn a_turn_run_still_anchors_its_multistep_rows() {
    let fake = Fake::new(32768);
    let inputs = bench::Inputs {
        prompts: Vec::new(),
        steps: Some((vec![step_row("one-007")], Default::default())),
        docs: None,
        memory: None,
    };
    let routing = bench::Routing {
        turn: true,
        arch: "llama".into(),
        ..bench::Routing::default()
    };
    let mut report = Report::default();
    bench::run_all(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        &routing,
        &inputs,
        &mut report,
        || fake.engine.prefix_cache_stats(),
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    assert!(report.anchor_ms.is_some(), "{report:?}");
}

/// `--turn` takes rows that show the chat's tools, with the router off as the app has it; anything
/// else fails before the model loads.
#[test]
fn a_turn_run_refuses_other_rows_and_a_router_before_the_load() {
    for more in [
        &[][..],
        &[
            "--prompts",
            "../../bench/agent/routing/dev.jsonl",
            "--router",
            "logprob",
        ],
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_larch-agent"))
            .args([
                "bench",
                "--turn",
                "--machine",
                "test",
                "-m",
                "missing",
                "--cpu",
            ])
            .args(more)
            .arg("--json")
            .arg(tmp.path().join("r.json"))
            .arg("--data-dir")
            .arg(tmp.path())
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{more:?}: {stderr}");
        assert!(stderr.contains("--turn"), "{more:?}: {stderr}");
        assert!(!stderr.contains("missing.gguf"), "{more:?}: {stderr}");
    }
}

#[test]
fn renamed_tools_hide_find_files_and_search_files_too() {
    for tool in ["find_files", "search_files"] {
        let shown = bench::shown_name(tool);
        assert_ne!(shown, tool);
        assert_eq!(bench::real_name(shown), tool);
    }
}

/// The `routing/multistep.jsonl` row `id` run on the replies `fake` holds: what each step scored,
/// whether the turn answered, and how many scripted tokens were left unsaid.
fn chat_row(fake: &Fake, routing: &bench::Routing, id: &str) -> (Vec<bool>, bool, usize) {
    let (rows, _) = bench::load_steps(&agent_dir().join("routing/multistep.jsonl")).unwrap();
    let row = rows.into_iter().find(|r| r.id == id).unwrap();
    let rows = bench::run_steps_routed(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        0,
        routing,
        &[row],
        |_| {},
    );
    (rows[0].steps.clone(), rows[0].answered, fake.unsaid())
}

#[test]
fn find_files_on_the_folder_is_right_where_a_find_or_fix_row_lists_it() {
    let fake = Fake::new(32768);
    let routing = bench::Routing::default();
    let find = |pattern, path| call("find_files", json!({"pattern": pattern, "path": path}));
    fake.say(&find("*.md", "documents/jobs"));
    fake.say(&call(
        "read_file",
        json!({"path": "documents/jobs/cv-2024.md"}),
    ));
    fake.say("Four years of Rust.");
    assert_eq!(
        chat_row(&fake, &routing, "find-002"),
        (vec![true; 2], true, 0)
    );
    fake.say(&call("read_file", json!({"path": "notes/todo.txt"})));
    fake.say(&find("todo", "notes"));
    fake.say(&call("read_file", json!({"path": "notes/todo-list.md"})));
    fake.say("Renew the passport.");
    assert_eq!(
        chat_row(&fake, &routing, "fix-001"),
        (vec![true; 3], true, 0)
    );
}

#[test]
fn a_fix_row_is_right_when_the_near_name_is_read_from_the_errors_listing() {
    // The not-found error lists the folder, so the listing step may be skipped; a name that is
    // not in the listing is still wrong.
    let fake = Fake::new(32768);
    let routing = bench::Routing::default();
    let read = |path| call("read_file", json!({ "path": path }));
    fake.say(&read("notes/todo.txt"));
    fake.say(&read("notes/todo-list.md"));
    fake.say("Renew the passport.");
    assert_eq!(
        chat_row(&fake, &routing, "fix-001"),
        (vec![true; 3], true, 0)
    );
    fake.say(&read("notes/todo.txt"));
    fake.say(&read("notes/todo.md"));
    assert_eq!(
        chat_row(&fake, &routing, "fix-001"),
        (vec![true, false, false], false, 0)
    );
}

#[test]
fn a_steps_other_right_tools_take_their_shown_names_under_rename_tools() {
    let fake = Fake::new(32768);
    let routing = bench::Routing {
        rename: true,
        ..bench::Routing::default()
    };
    let named = |name, args| json!({"name": bench::shown_name(name), "parameters": args});
    for (tool, args) in [
        ("list_dir", json!({"path": "documents/jobs"})),
        (
            "find_files",
            json!({"pattern": "*.md", "path": "documents/jobs"}),
        ),
    ] {
        fake.say(&named(tool, args).to_string());
        let read = named("read_file", json!({"path": "documents/jobs/cv-2024.md"}));
        fake.say(&read.to_string());
        fake.say("Four years of Rust.");
        assert_eq!(
            chat_row(&fake, &routing, "find-002"),
            (vec![true; 2], true, 0),
            "{tool}"
        );
    }
}

/// A fix row's not-found result is what `read_file` writes for its path in a folder that holds its
/// listing, so the row measures the recovery the app goes through.
#[cfg(unix)]
#[test]
fn a_fix_rows_not_found_result_is_what_read_file_writes() {
    let (rows, _) = bench::load_steps(&agent_dir().join("routing/multistep.jsonl")).unwrap();
    for r in rows.iter().filter(|r| r.id.starts_with("fix-")) {
        let (read, list) = (&r.steps[0], &r.steps[1]);
        let tmp = tempfile::tempdir().unwrap();
        let folder = tmp
            .path()
            .join(list.expect.args_required["path"].as_str().unwrap());
        fs::create_dir_all(&folder).unwrap();
        for name in list.result.lines() {
            match name.strip_suffix('/') {
                Some(dir) => fs::create_dir(folder.join(dir)).unwrap(),
                None => fs::write(folder.join(name), "").unwrap(),
            }
        }
        let ctx = larch_agent::tool::ToolCtx {
            allowed: vec![tmp.path().into()],
            data_dir: tmp.path().join("data"),
            source: String::new(),
        };
        let args = serde_json::Value::Object(read.expect.args_required.clone());
        let e = (Registry::builtin().get("read_file").unwrap())
            .call(&args, &ctx)
            .unwrap_err();
        assert_eq!(read.result, e.to_string(), "{}", r.id);
    }
}

#[test]
fn search_files_is_a_right_first_step_where_an_absent_fact_may_be_in_the_notes() {
    let (rows, _) = bench::load(&agent_dir().join("source-routing-holdout.jsonl")).unwrap();
    let search = Parsed {
        call: Some(ToolCall {
            name: "search_files".into(),
            args: json!({}),
        }),
        ..Parsed::default()
    };
    let absent = || {
        rows.iter()
            .filter(|r| r.id.starts_with("holdout-files-absent-"))
    };
    assert_eq!(absent().count(), 6);
    for r in absent() {
        assert!(score_toolcall(&r.expect, &search).right_tool, "{}", r.id);
    }
}

#[test]
fn some_source_rows_expect_find_files_and_some_search_files() {
    let (rows, _) = bench::load(&agent_dir().join("source-routing.jsonl")).unwrap();
    for name in ["find_files", "search_files"] {
        let expecting = (rows.iter())
            .filter(|r| r.expect.tool.as_deref() == Some(name))
            .count();
        assert!(expecting >= 2, "{name}: {expecting}");
    }
}

#[test]
fn a_rows_outcome_comes_from_the_tool_results_first_line_then_its_score() {
    use bench::{Outcome, outcome};
    use larch_agent::skills::{ask_docs, web_answer};
    let cases = [
        (
            Some("Paris is the capital [1]."),
            false,
            true,
            Outcome::Answered,
        ),
        (Some("Lyon is the capital."), false, false, Outcome::Wrong),
        (None, false, true, Outcome::Answered),
        (Some(web_answer::NOT_FOUND), false, false, Outcome::Refused),
        (
            Some("I couldn't find that in your notes."),
            false,
            false,
            Outcome::Refused,
        ),
        (Some(web_answer::NO_PAGES), false, false, Outcome::NoFiles),
        (
            Some("No readable files in ~/docs (saw: .docx)"),
            false,
            false,
            Outcome::NoFiles,
        ),
        (Some("No files match *.pdf"), false, false, Outcome::NoFiles),
        (
            Some("the file has no text (a scanned PDF needs OCR first)"),
            false,
            false,
            Outcome::Unreadable,
        ),
        (
            Some("No page could be fetched: a.example 403"),
            false,
            false,
            Outcome::Unreadable,
        ),
        (
            Some("The web search failed: rate-limited"),
            false,
            false,
            Outcome::Error,
        ),
        (Some("Paris."), true, false, Outcome::Error),
        // ask_docs's first line (#400): the rest says answered or refused.
        (
            Some("Read 3 files (1 pdf, 2 md, 0 txt)\nThe rent is $1,850 (lease.md:14)."),
            false,
            true,
            Outcome::Answered,
        ),
        (
            Some("Read 3 files (1 pdf, 2 md, 0 txt)\nThe rent is $1,950."),
            false,
            false,
            Outcome::Wrong,
        ),
        (
            Some(&format!(
                "Read 1 file (0 pdf, 1 md, 0 txt)\n{}",
                ask_docs::NOT_FOUND
            )),
            false,
            false,
            Outcome::Refused,
        ),
        (
            Some(&format!(
                "Read 2 files (2 pdf, 0 md, 0 txt); no text in scan.pdf\n{}",
                ask_docs::NOT_FOUND
            )),
            false,
            false,
            Outcome::Unreadable,
        ),
        (
            Some("No text in scan-passport.pdf"),
            false,
            false,
            Outcome::Unreadable,
        ),
        (
            Some("No readable files in ~/docs (saw: .docx, .pages)"),
            false,
            false,
            Outcome::NoFiles,
        ),
        (
            Some("No files match ~/papers"),
            false,
            false,
            Outcome::NoFiles,
        ),
        (
            Some(
                "I can't read your files in this chat yet. Attach a folder or the files, then ask again.",
            ),
            false,
            true,
            Outcome::NoFiles,
        ),
    ];
    for (first, error, right, want) in cases {
        assert_eq!(outcome(first, error, right, false), want, "{first:?}");
    }
    // A right abstention is a refusal.
    assert_eq!(
        outcome(Some(ask_docs::NOT_FOUND), false, true, true),
        Outcome::Refused
    );
    // A right answer that hedges first is answered; on an abstain row, the hedge is its refusal.
    let hedged = Some("I'm not sure of the exact date, but it was 3 Oct.");
    assert_eq!(outcome(hedged, false, true, false), Outcome::Answered);
    assert_eq!(outcome(hedged, false, true, true), Outcome::Refused);
    // Each line matches as the skill writes it, whatever its case.
    let fetched = format!("{}: a; b", larch_agent::skills::web_answer::FETCH_FAILED);
    assert_eq!(
        outcome(Some(&fetched), false, false, false),
        Outcome::Unreadable
    );
    let none = larch_agent::skills::web_answer::NO_PAGES;
    assert_eq!(outcome(Some(none), false, false, false), Outcome::NoFiles);
    let names: Vec<String> = [Outcome::NoFiles, Outcome::Unreadable]
        .iter()
        .map(|o| serde_json::to_string(o).unwrap())
        .collect();
    assert_eq!(names, ["\"no_files\"", "\"unreadable\""]);
}

#[test]
fn a_rows_stages_add_up_its_audit_steps_and_turns() {
    use larch_agent::audit::{AuditEvent, EventKind};
    let step = |name: &str, ms: f64| AuditEvent {
        kind: EventKind::Step,
        step: Some(name.into()),
        wall_ms: Some(ms),
        ..AuditEvent::default()
    };
    let turn = |ttft: f64, tokens: u32, tok_s: f64| AuditEvent {
        kind: EventKind::Turn,
        ttft_ms: Some(ttft),
        generated_tokens: Some(tokens),
        tok_s: Some(tok_s),
        ..AuditEvent::default()
    };
    let events = [
        step("search", 300.0),
        step("search", 200.0),
        step("fetch", 900.0),
        turn(1000.0, 50, 50.0),
        turn(500.0, 100, 25.0),
        step("recheck", 4.0),
    ];
    let s = bench::stages(&events, Some(120.0));
    assert_eq!(
        (
            s.search_ms,
            s.fetch_ms,
            s.prefill_ms_sum,
            s.generation_ms_sum
        ),
        (Some(500.0), Some(900.0), Some(1500.0), Some(5000.0))
    );
    assert_eq!((s.recheck_ms, s.first_visible_ms), (Some(4.0), Some(120.0)));
    // A run with no web steps has none, not 0.
    let s = bench::stages(&[turn(10.0, 0, 0.0)], None);
    assert_eq!((s.search_ms, s.fetch_ms, s.recheck_ms), (None, None, None));
    assert_eq!(
        (s.prefill_ms_sum, s.generation_ms_sum),
        (Some(10.0), Some(0.0))
    );
    // An old report's row, with no stages, loads.
    let s: bench::Stages = serde_json::from_str("{}").unwrap();
    assert_eq!(s, bench::Stages::default());
    // So does one from before the sums were named so.
    let s: bench::Stages =
        serde_json::from_str(r#"{"prefill_ms": 3.0, "generation_ms": 4.0}"#).unwrap();
    assert_eq!(
        (s.prefill_ms_sum, s.generation_ms_sum),
        (Some(3.0), Some(4.0))
    );
}

#[test]
fn repeated_runs_keep_each_and_say_which_rows_passed_every_one() {
    let runs = [
        ("a", true),
        ("b", true),
        ("a", true),
        ("b", false),
        ("a", true),
        ("b", true),
    ];
    let r = bench::repeated(runs.iter().map(|(id, pass)| (id.to_string(), *pass)));
    assert_eq!(r.runs, 3);
    assert_eq!((r.rows.len(), r.all_pass), (2, 0.5));
    assert!((r.mean - 5.0 / 6.0).abs() < 1e-9);
    let b = &r.rows[1];
    assert_eq!(
        (b.id.as_str(), b.passes, b.runs, b.all_pass),
        ("b", 2, 3, false)
    );
    assert!(r.rows[0].all_pass);
}

#[test]
fn the_apps_options_come_from_one_function_and_the_bench_says_where_they_differ() {
    let app = AgentOptions::app("/data".into(), vec!["/home/a".into()], "llama".into());
    assert_eq!((app.router, app.prefix_cache), (Router::Off, true));
    let ours = AgentOptions {
        allowed: Vec::new(),
        prefix_cache: false,
        ..app.clone()
    };
    assert_eq!(
        bench::differs(&ours, &app),
        [
            r#"allowed: ["/home/a"] (bench: [])"#,
            "prefix_cache: true (bench: false)"
        ]
    );
    assert!(bench::differs(&app, &app).is_empty());
}

#[test]
fn every_scored_row_is_one_evidence_line_and_a_row_for_a_human_is_none() {
    let replay_row = |id: &str, right: bool, check: bool| bench::ReplayRow {
        id: id.into(),
        right,
        check,
        answer: "Villa won 2-1 [1].".into(),
        sources: vec!["https://results.example/villa".into()],
        ..bench::ReplayRow::default()
    };
    let report = Report {
        replay: Some(bench::Replay {
            rows: vec![replay_row("sport-villa", true, false)],
            ..bench::Replay::default()
        }),
        live: Some(bench::Replay {
            rows: vec![
                replay_row("sport-villa", false, false),
                replay_row("sport-stale", false, true),
                bench::ReplayRow {
                    error: Some("engine".into()),
                    ..replay_row("sport-error", false, false)
                },
            ],
            ..bench::Replay::default()
        }),
        everyday: Some(bench::Everyday {
            rows: vec![
                bench::EverydayRow {
                    id: "everyday-smalltalk-001".into(),
                    right: true,
                    answer: "Doing well!".into(),
                    answer_scored: true,
                    ..bench::EverydayRow::default()
                },
                // A run without the web stops at the call: no answer was scored.
                bench::EverydayRow {
                    id: "everyday-travel-001".into(),
                    right: true,
                    tools: vec!["web_answer".into()],
                    ..bench::EverydayRow::default()
                },
            ],
            ..bench::Everyday::default()
        }),
        docs: Some(Docs {
            questions: vec![DocsQuestion {
                id: "q01".into(),
                answer: "$1,850 (lease.md:14).".into(),
                right: true,
                cited: vec![CitedQuote {
                    file: "lease.md".into(),
                    line: 14,
                    quote: "rent".into(),
                }],
                ..DocsQuestion::default()
            }],
            ..Docs::default()
        }),
        ..Report::default()
    };
    let asked: BTreeMap<String, String> = [
        ("sport-villa", "Who won?"),
        ("everyday-smalltalk-001", "hey, how's it going?"),
        ("everyday-travel-001", "weather in Lisbon this weekend?"),
        ("q01", "What is the rent?"),
    ]
    .map(|(id, q)| (id.to_string(), q.to_string()))
    .into();
    let lines = bench::evidence(&report, &asked, "2026-10-30T12:00:00Z");
    let got: Vec<_> = (lines.iter())
        .map(|e| (e.question.as_str(), e.tools.clone(), e.verdict.as_str()))
        .collect();
    let web = vec!["web_answer".to_string()];
    assert_eq!(
        got,
        [
            ("Who won?", web.clone(), "right"),
            ("Who won?", web, "wrong"),
            ("hey, how's it going?", vec![], "right"),
            ("What is the rent?", vec!["ask_docs".to_string()], "right"),
        ]
    );
    assert_eq!(lines[0].sources, ["https://results.example/villa"]);
    assert_eq!(lines[3].sources, ["lease.md:14"]);
    assert!(lines.iter().all(|e| e.source_of_verdict == "bench"));
    let tmp = tempfile::tempdir().unwrap();
    bench::append_evidence(tmp.path(), &lines[..2]).unwrap();
    bench::append_evidence(tmp.path(), &lines[2..]).unwrap();
    let text = fs::read_to_string(tmp.path().join("agent/evidence.jsonl")).unwrap();
    assert_eq!(text.lines().count(), 4);
    // A --quick run is a smoke test: it keeps none.
    let quick = Report {
        quick: true,
        ..report.clone()
    };
    assert!(bench::evidence(&quick, &asked, "now").is_empty());
    let first: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    let keys: BTreeSet<&str> = first
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let want = [
        "answer",
        "at",
        "question",
        "source_of_verdict",
        "sources",
        "tools",
        "verdict",
    ];
    assert_eq!(keys, want.into());
}

/// A docs question can read the paper first (`before: read_papers`), as a chat that read ResNet
/// and then asks about it, and its follow-up is right only when the answer cites the planted
/// quote from the paper's methods section.
#[test]
fn a_docs_follow_up_after_read_papers_quotes_the_methods_section() {
    let fake = Fake::new(32768);
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("resnet.md"),
        "# ResNet: Deep Residual Learning\n\n## Abstract\n\nResNet eases the training of deep \
         networks.\n\n## Methods\n\nResNet layers learn residual functions of their inputs.\n\
         Shortcut connections perform identity mapping, added to the stacked layers.\n\n\
         ## Results\n\nResNet won ILSVRC 2015.\n",
    )
    .unwrap();
    let questions = json!([{
        "id": "resnet-methods",
        "question": "What does the methods section of ResNet say about shortcut connections?",
        "facts": [{"file": "resnet.md", "quote": "Shortcut connections perform identity mapping", "line": 10}],
        "before": [{"tool": "read_papers", "args": {"papers": ["resnet.md"]}}]
    }]);
    fs::write(dir.path().join("questions.json"), questions.to_string()).unwrap();
    let set = bench::doc_set(dir.path()).unwrap();
    // The reader's summary, then ask_docs: one worker on the paper's one chunk, then the answer.
    fake.say("ResNet learns residual functions through shortcut connections.");
    fake.say(r#"[{"line": 10, "claim": "Shortcut connections perform identity mapping"}]"#);
    fake.say("Shortcut connections perform identity mapping [1].");
    let docs = bench::run_docs(
        &fake.larch,
        None,
        Thinking::default(),
        false,
        &set,
        larch_agent::workers::Workers::default(),
        |_, _| {},
    )
    .unwrap();
    assert_eq!(fake.unsaid(), 0);
    assert!(docs.errors.is_empty(), "{:?}", docs.errors);
    let q = &docs.questions[0];
    assert!(q.before[0].contains("resnet.md"), "{:?}", q.before);
    assert_eq!(q.recalled, [0], "{q:?}");
    assert!(q.right, "{q:?}");
    assert_eq!(q.cited[0].line, 10);
}

/// The tiny memory chat set the runner's test scripts.
const MEMORY_CHATS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/memory-chats.jsonl"
);

#[test]
fn the_memory_chat_set_loads_and_a_broken_case_fails_it() {
    let set = bench::memory_chat_set(&agent_dir().join("memory-chats/chats.jsonl")).unwrap();
    assert_eq!(set.cases.len(), 40);
    let from = set.cases.iter().filter(|c| c.from.is_some()).count();
    assert_eq!(from, 24, "the #436 fixtures");
    // Every case runs; the memory-off one with memory off.
    assert!(set.cases.iter().all(|c| c.skip.is_none()));
    assert!(set.cases.iter().any(|c| c.off));
    let good = fs::read_to_string(MEMORY_CHATS).unwrap();
    let first = good.lines().next().unwrap();
    let dir = tempfile::tempdir().unwrap();
    for (text, says) in [
        (first.replace("mc-01-save", "save-01"), "save-01"),
        (format!("{first}\n{first}"), "twice"),
        (first.replace("\"user\": \"My", "\"usr\": \"My"), "usr"),
        (
            first.replace("[{\"user\": \"My brother Arun lives in Denver.\"}]", "[]"),
            "no message",
        ),
        (
            first.replace("\"Denver\"]}]", "\"Denver|\"]}]"),
            "empty detail",
        ),
    ] {
        let path = dir.path().join("chats.jsonl");
        fs::write(&path, &text).unwrap();
        let e = format!("{:#}", bench::memory_chat_set(&path).unwrap_err());
        assert!(e.contains(says), "{says}: {e}");
    }
}

#[test]
fn a_detail_reads_its_alternatives_and_secrets_are_seen() {
    assert!(bench::has("Mi hermana Lucía vive en Madrid", "Lucia|Lucía"));
    assert!(bench::has(
        "She works as an architect.",
        "architect*|arquitect*"
    ));
    assert!(!bench::has("Mayan ruins", "Maya"));
    assert!(bench::secret_shaped("key sk-proj-AbC123xyz789LmNoP"));
    assert!(bench::secret_shaped("-----BEGIN OPENSSH PRIVATE KEY-----"));
    assert!(!bench::secret_shaped("I use a desk-lamp"));
}

#[test]
fn the_memory_chat_runner_scores_a_save_a_partial_save_a_false_save_and_a_wrong_merge() {
    let fake = Fake::new(32768);
    let set = bench::memory_chat_set(Path::new(MEMORY_CHATS)).unwrap();
    let remembers = |text: &str| {
        fake.say(&spaced("remember", json!({"text": text, "type": "user"})));
        fake.say("Noted.");
    };
    // mc-01: saved whole, then used in a new chat.
    remembers("The user's brother Arun lives in Denver.");
    fake.say("Arun lives in Denver.");
    // mc-02: saved without the job; its probe is for a human.
    remembers("The user's sister Maya lives in Austin.");
    fake.say("Maybe a sketchbook.");
    // mc-03: a hypothetical sister saved.
    remembers("The user has a sister called Lila.");
    // mc-04: the near-duplicate update overwrites the seeded Maya with Mira.
    remembers("My sister Mira lives in Austin and works as a designer at Figma");
    let routing = bench::Routing {
        arch: "llama".into(),
        tool_grammar: true,
        ..bench::Routing::default()
    };
    let mut report = Report::default();
    let mut lines = Vec::new();
    bench::run_memory_chats(
        &fake.larch,
        Thinking::default(),
        &routing,
        false,
        &set,
        &mut report,
        |_| Ok(()),
        |line| lines.push(line),
    )
    .unwrap();
    assert_eq!(fake.unsaid(), 0);
    let m = report.memory_chats.unwrap();
    let row = |id: &str| m.rows.iter().find(|r| r.id == id).unwrap();
    let errors: Vec<_> = m.rows.iter().filter_map(|r| r.error.clone()).collect();
    assert!(errors.is_empty(), "{errors:?}");

    let save = row("mc-01-save");
    assert_eq!((save.gold[0].saved, save.gold[0].missing.len()), (true, 0));
    assert_eq!(
        (save.approvals, save.false_saves, save.merges),
        (0, 0, 0),
        "remember runs unasked"
    );
    assert_eq!(save.turns[0].saved.len(), 1);
    assert_eq!(save.turns[0].tools, ["remember"]);
    let probe = &save.probes[0];
    assert_eq!(
        (probe.captured, probe.retrieved, probe.right),
        (true, true, Some(true))
    );
    // The probe's chat is a new one, so its turn is no chat's second.
    assert_eq!(save.turns[1].chat, 1);

    let partial = row("mc-02-partial");
    assert!(partial.gold[0].saved);
    assert_eq!(partial.gold[0].missing, ["designer"]);
    assert_eq!(partial.probes[0].right, None);

    let negative = row("mc-03-false-save");
    assert_eq!(
        (negative.false_saves, negative.never_held.clone()),
        (1, vec!["Lila".to_string()])
    );

    let merge = row("mc-04-merge");
    assert_eq!(merge.merges, 1, "{:?}", merge.facts);
    assert_eq!(merge.lost, ["Maya + Figma"]);
    assert!(merge.gold[0].saved);

    assert_eq!(
        row("mc-05-later").skipped.as_deref(),
        Some("needs the memory off switch")
    );
    assert!(row("mc-05-later").turns.is_empty());

    let s = &m.summary;
    assert_eq!((s.cases, s.skipped, s.errors), (5, 1, 0));
    assert_eq!((s.unasked_facts, s.unasked_saved, s.whole_facts), (3, 3, 2));
    assert_eq!(s.whole_share, Some(2.0 / 3.0));
    // The hypothetical alone: the probes are counted apart, so they do not dilute the share.
    assert_eq!((s.negative_messages, s.false_saves), (1, 1));
    assert_eq!(s.false_save_share, Some(1.0));
    assert_eq!((s.probe_turns, s.probe_saves), (2, 0));
    // Every remember here came unasked; the hypothetical's is on a negative.
    assert_eq!(
        (s.remember_explicit, s.remember_unasked, s.remember_negative),
        (0, 3, 1)
    );
    assert_eq!((s.never_held, s.merges, s.secrets), (1, 2, 0));
    assert_eq!((s.probes, s.probes_right, s.probe_checks), (1, 1, 1));
    assert_eq!(s.approvals, 0);
    let table = bench::memory_chats_table(&m);
    assert!(
        table.contains("remembers on its own  100% (3/3)"),
        "{table}"
    );
    assert!(table.contains("mc-05-later skipped"), "{table}");
    assert_eq!(lines.len(), 5);
}

#[test]
fn a_broken_memory_chat_set_or_a_wrong_flag_fails_before_the_load() {
    let tmp = tempfile::tempdir().unwrap();
    let broken = tmp.path().join("chats.jsonl");
    fs::write(&broken, r#"{"id":"mc-1-x","chats":[[{"user":"hi"}]]}"#).unwrap();
    let run = |args: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_larch-agent"))
            .args(["bench", "--machine", "test", "-m", "missing", "--cpu"])
            .args(args)
            .arg("--json")
            .arg(tmp.path().join("r.json"))
            .arg("--data-dir")
            .arg(tmp.path())
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        (out.status.code(), stderr)
    };
    for (args, says) in [
        (
            vec!["--memory-chats", broken.to_str().unwrap()],
            "mc-<nn>-<name>",
        ),
        (vec!["--memory-chats", "--live"], "--live"),
        (vec!["--memory-chats", "--router", "logprob"], "router off"),
    ] {
        let (code, stderr) = run(&args);
        assert_ne!(code, Some(0), "{args:?}: {stderr}");
        assert!(stderr.contains(says), "{args:?}: {stderr}");
        assert!(!stderr.contains("missing.gguf"), "{args:?}: {stderr}");
    }
    // The checkout's set by default, which loads before the model does.
    let (_, stderr) = run(&["--memory-chats", "--app"]);
    assert!(stderr.contains("missing.gguf"), "{stderr}");
}

/// Runs the memory chat cases `jsonl` on `fake`, with `audit` as the run's audit log.
fn memory_chats(fake: &Fake, jsonl: &str, audit: Option<PathBuf>) -> bench::MemoryChats {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chats.jsonl");
    fs::write(&path, jsonl).unwrap();
    let mut set = bench::memory_chat_set(&path).unwrap();
    set.audit = audit;
    let routing = bench::Routing {
        arch: "llama".into(),
        tool_grammar: true,
        ..bench::Routing::default()
    };
    let mut report = Report::default();
    bench::run_memory_chats(
        &fake.larch,
        Thinking::default(),
        &routing,
        false,
        &set,
        &mut report,
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    assert_eq!(fake.unsaid(), 0);
    report.memory_chats.unwrap()
}

#[test]
fn a_seeded_fact_or_an_old_one_never_counts_as_saved() {
    let fake = Fake::new(32768);
    // Nothing is saved: the seeded Arun fact holds the key of both gold facts.
    fake.say("Congratulations to them!");
    // The correction is not captured: the first message's fact holds the key Maya.
    fake.say(&spaced(
        "remember",
        json!({"text": "The user's sister Maya lives in Austin.", "type": "user"}),
    ));
    fake.say("Noted.");
    fake.say("Got it.");
    let m = memory_chats(
        &fake,
        concat!(
            r#"{"id":"mc-01-seeded","memory":["The user's brother Arun lives in Denver."],"chats":[[{"user":"Arun just got engaged to Priya!"}]],"facts":[{"details":["Priya","Arun"]},{"details":["Arun","Denver"]}]}"#,
            "\n",
            r#"{"id":"mc-02-correction","chats":[[{"user":"My sister Maya lives in Austin."},{"user":"Correction: Maya moved to Denver."}]],"facts":[{"details":["Maya","Austin"]},{"details":["Denver","Maya"],"stale":["Austin"]}]}"#,
            "\n"
        ),
        None,
    );
    let saved = |r: usize| m.rows[r].gold.iter().map(|g| g.saved).collect::<Vec<_>>();
    assert_eq!(saved(0), [false, false]);
    assert_eq!(saved(1), [true, false]);
    assert!(m.rows[1].gold[1].stale);
    assert_eq!(m.summary.corrections, 1);
    assert_eq!((m.summary.unasked_facts, m.summary.unasked_saved), (4, 1));
}

#[test]
fn a_fact_forgotten_after_the_chats_still_counts_as_saved() {
    let fake = Fake::new(32768);
    fake.say(&spaced(
        "remember",
        json!({"text": "The user's project codename is Northstar.", "type": "project"}),
    ));
    fake.say("Saved.");
    fake.say("I have no codename saved for you.");
    let m = memory_chats(
        &fake,
        r#"{"id":"mc-15-forget","chats":[[{"user":"Remember that my project codename is Northstar.","explicit":true}]],"forget":["Northstar"],"facts":[{"details":["Northstar"],"explicit":true}],"probes":[{"user":"What codename do you remember for me?","not":["Northstar"]}]}"#,
        None,
    );
    let r = &m.rows[0];
    assert_eq!(
        (r.gold[0].saved, r.forgot, r.remember_explicit),
        (true, 1, 1)
    );
    assert!(r.probes[0].captured);
    assert_eq!(r.probes[0].right, Some(true));
    assert_eq!((m.summary.explicit_facts, m.summary.explicit_saved), (1, 1));
}

#[test]
fn a_quoted_sentence_after_a_fact_is_not_scored_as_the_fact() {
    let fake = Fake::new(32768);
    // As background capture will store it: the model's sentence, then the user's words.
    let text = "The user's sister Maya lives in Austin.\nSaid: \"My sister Maya lives in Austin and works as a designer.\"";
    fake.say(&spaced("remember", json!({"text": text, "type": "user"})));
    fake.say("Noted.");
    let m = memory_chats(
        &fake,
        r#"{"id":"mc-01-quote","chats":[[{"user":"My sister Maya lives in Austin and works as a designer."}]],"facts":[{"details":["Maya","Austin","designer"]}],"never":[["designer"]]}"#,
        None,
    );
    let r = &m.rows[0];
    assert_eq!(r.gold[0].missing, ["designer"]);
    assert!(r.never_held.is_empty(), "{r:?}");
    assert_eq!(r.facts[0].text, "The user's sister Maya lives in Austin.");
    assert_eq!(bench::sentence_of("A fact.\nSaid: \"words\""), "A fact.");
    assert_eq!(bench::sentence_of("Said: nothing else"), "");
    assert_eq!(bench::sentence_of("He said: hi"), "He said: hi");
}

#[test]
fn the_runs_audit_log_holds_every_cases_chats() {
    let fake = Fake::new(32768);
    fake.say("Hello.");
    fake.say("Hello again.");
    let out = tempfile::tempdir().unwrap();
    let audit = out.path().join("r-audit.jsonl");
    // A log from an earlier run is replaced, not added to.
    fs::write(&audit, "{\"old\":true}\n").unwrap();
    memory_chats(
        &fake,
        concat!(
            r#"{"id":"mc-01-a","chats":[[{"user":"hi","no_save":true}]]}"#,
            "\n",
            r#"{"id":"mc-02-b","chats":[[{"user":"hi again","no_save":true}]]}"#,
            "\n"
        ),
        Some(audit.clone()),
    );
    let text = fs::read_to_string(&audit).unwrap();
    assert!(!text.contains("\"old\""), "{text}");
    let chats: BTreeSet<String> = (text.lines())
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| v["chat"].as_str().map(String::from))
        .collect();
    assert_eq!(chats.len(), 2, "{text}");
}

#[test]
fn an_errored_case_counts_in_the_shares_with_what_it_saved() {
    let row = |error: Option<&str>, saved: bool| bench::MemoryChatRow {
        id: "mc-01-x".into(),
        gold: vec![bench::GoldRow {
            details: vec!["Maya".into()],
            saved,
            ..bench::GoldRow::default()
        }],
        error: error.map(String::from),
        ..bench::MemoryChatRow::default()
    };
    let s = bench::memory_chat_summary(&[row(None, true), row(Some("the web step failed"), false)]);
    assert_eq!(s.errors, 1);
    assert_eq!((s.unasked_facts, s.unasked_saved), (2, 1));
    assert_eq!(s.remembered_share, Some(0.5));
}

#[test]
fn a_fact_with_a_gold_facts_key_alone_does_not_capture_it() {
    let fake = Fake::new(32768);
    fake.say(&spaced(
        "remember",
        json!({"text": "The user's dog is Biscuit.", "type": "user"}),
    ));
    fake.say("Cute!");
    let m = memory_chats(
        &fake,
        concat!(
            r#"{"id":"mc-01-dog","chats":[[{"user":"My dog Biscuit is a 4 year old beagle."}]],"facts":[{"details":["Biscuit","beagle"]},{"details":["Biscuit","4"]}]}"#,
            "\n"
        ),
        None,
    );
    let gold = &m.rows[0].gold;
    assert_eq!(
        gold.iter().map(|g| g.saved).collect::<Vec<_>>(),
        [false, false],
        "{gold:?}"
    );
}

#[test]
fn a_remember_after_a_tool_result_is_asked_and_the_bench_says_yes() {
    let fake = Fake::new(32768);
    // A web question first: its result, even a failure, is in the chat, so the save asks.
    fake.say(&spaced(
        "web_answer",
        json!({"question": "Where is Denver?"}),
    ));
    fake.say("Denver is in Colorado.");
    fake.say(&spaced(
        "remember",
        json!({"text": "The user's brother Arun lives in Denver.", "type": "user"}),
    ));
    fake.say("Noted.");
    let m = memory_chats(
        &fake,
        concat!(
            r#"{"id":"mc-01-asked","chats":[[{"user":"Where is Denver?"},{"user":"My brother Arun lives in Denver."}]],"facts":[{"details":["Arun","Denver"]}]}"#,
            "\n"
        ),
        None,
    );
    let r = &m.rows[0];
    assert_eq!(r.approvals, 1, "{r:?}");
    assert!(r.gold[0].saved, "{r:?}");
}

#[test]
fn a_memory_off_case_saves_nothing_and_its_probe_runs_with_memory_on() {
    let fake = Fake::new(32768);
    // Off, the chat shows no memory tool; a remember that comes anyway is refused.
    fake.say(&spaced(
        "remember",
        json!({"text": "The user's sister Lila lives in Boston.", "type": "user"}),
    ));
    fake.say("Nice.");
    fake.say("I don't know where she lives.");
    let m = memory_chats(
        &fake,
        concat!(
            r#"{"id":"mc-16-memory-off","chats":[[{"user":"My sister Lila lives in Boston.","no_save":true}]],"never":[["Lila"]],"probes":[{"user":"Where does my sister Lila live?","not":["Boston"]}],"off":true}"#,
            "\n"
        ),
        None,
    );
    let r = &m.rows[0];
    assert_eq!(r.skipped, None);
    assert_eq!((r.false_saves, r.never_held.len()), (0, 0), "{r:?}");
    assert_eq!(r.probes[0].right, Some(true), "{r:?}");
    assert_eq!(fake.unsaid(), 0);
}
