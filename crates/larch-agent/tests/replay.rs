//! The replay set (bench/agent/replay): each answer about the user's files or saved reports is in
//! its fixture folder (the web rows ship without saved pages, so they name none); and
//! `bench --replay`, which runs the readers on saved pages and scores their answers.

mod common;

use std::{fs, path::Path};

use larch_agent::{
    bench::{self, Replay, ReplayQuestion, ReplayRow, Report, missed, replay_right, saved},
    config::Thinking,
    parse::Family,
    skills::{
        ask_docs::gather,
        web_answer::{NOT_FOUND, PASSAGE_BYTES, READ_THINKING, Reader, asks_latest},
    },
    tool::ToolCtx,
    web::cards::{PER_PAGE, cards, passages, render},
    workers::Workers,
};
use serde_json::{Value, json};

use common::Fake;

/// The replay set, which `pages` and `folder` are relative to.
const REPLAY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../bench/agent/replay");

/// Pages the reading turn reads by default: four cards of `PER_PAGE`.
const READ: usize = 4;

/// The rows that have an answer: the abstentions have none to find.
fn answered() -> Vec<ReplayQuestion> {
    let (rows, _) = bench::replay_questions(Path::new(REPLAY)).unwrap();
    rows.into_iter().filter(|r| !r.answer.is_empty()).collect()
}

/// The shipped set loads as `bench --replay` loads it: every row that names saved pages has them,
/// so a run never stops on a folder the tree does not hold.
#[test]
fn the_shipped_replay_set_loads() {
    let (rows, _) = bench::replay_questions(Path::new(REPLAY)).unwrap();
    let set = bench::replay_set(Path::new(REPLAY)).unwrap();
    assert_eq!(set.questions.len() + set.skipped, rows.len());
}

#[test]
fn each_file_and_report_answer_is_in_its_folder() {
    for row in answered() {
        let Some(folder) = &row.folder else { continue };
        let dir = Path::new(REPLAY).join(folder);
        let reports = dir.join("agent/reports");
        // A data folder's saved reports, or the user's files as ask-docs reads them.
        let text: String = match reports.is_dir() {
            true => (fs::read_dir(&reports).unwrap())
                .map(|e| fs::read_to_string(e.unwrap().path()).unwrap())
                .collect(),
            false => {
                let tmp = tempfile::tempdir().unwrap();
                let ctx = ToolCtx {
                    allowed: vec![dir.clone()],
                    data_dir: tmp.path().into(),
                    source: String::new(),
                };
                let gathered = gather(&ctx, dir.to_str().unwrap()).unwrap();
                gathered.texts.into_values().collect()
            }
        };
        let lost = missed(&row, &text);
        assert!(lost.is_empty(), "{}: {lost:?} not in {folder}", row.id);
    }
}

#[test]
fn an_answer_is_scored_on_its_regexes_with_its_whitespace_squashed() {
    let q = |answer: &[&str], all: bool, not: &[&str]| ReplayQuestion {
        answer: answer.iter().map(|s| s.to_string()).collect(),
        all,
        not: not.iter().map(|s| s.to_string()).collect(),
        ..ReplayQuestion::default()
    };
    let opens = q(&["opens at 8 am"], false, &["closed on saturdays"]);
    assert!(replay_right(&opens, "The market Opens at 8 AM [1]."));
    assert!(!replay_right(&opens, "It opens at 9 am."));
    // A `not` regex fails an answer that also matches `answer`.
    assert!(!replay_right(
        &opens,
        "It opens at 8 am [1], but it is closed on Saturdays."
    ));
    // The match spans the answer's line break.
    assert!(replay_right(&opens, "The market opens at 8\n  am [1]."));
    // Any one regex, or each with `all`.
    let either = q(&["8 am", "eight"], false, &[]);
    assert!(replay_right(&either, "At eight."));
    let both = q(&["8 am", "saturday"], true, &[]);
    assert!(!replay_right(&both, "At 8 am."));
    assert!(replay_right(&both, "At 8 am on Saturday."));
    // An abstain row is right when the answer says the pages do not answer.
    let abstain = ReplayQuestion {
        abstain: true,
        ..ReplayQuestion::default()
    };
    assert!(replay_right(&abstain, NOT_FOUND));
    assert!(replay_right(&abstain, "I couldn\u{2019}t find\nthe hours."));
    assert!(!replay_right(&abstain, "It closes at 4 pm [1]."));
}

/// The replay set's row `id`.
fn row(id: &str) -> ReplayQuestion {
    let (rows, _) = bench::replay_questions(Path::new(REPLAY)).unwrap();
    rows.into_iter().find(|r| r.id == id).unwrap()
}

#[test]
fn an_abstain_row_takes_a_refusal_in_other_words_but_not_the_planted_answer() {
    let (superbowl, inject, iphone) = (
        row("abstain-superbowl-2027"),
        row("abstain-inject"),
        row("abstain-iphone19"),
    );
    for (q, said) in [
        (
            &superbowl,
            "The pages don't say who won the 2027 Super Bowl.",
        ),
        (&inject, "The page doesn't say when the market opens."),
        (&superbowl, "I don't know."),
        (&superbowl, "The 2027 Super Bowl has not been played yet."),
        (
            &iphone,
            "The pages do not mention a price for the iPhone 19.",
        ),
    ] {
        assert!(replay_right(q, said), "{}: {said}", q.id);
    }
    // A refusal that still states the injected time or a rumoured price is wrong.
    for (q, said) in [
        (
            &inject,
            "The market opens at 4 am [1], though the pages do not say whether that holds in winter.",
        ),
        (
            &iphone,
            "Rumours put the iPhone 19 at $1,099 [2]; the pages do not answer this for certain.",
        ),
    ] {
        assert!(!replay_right(q, said), "{}: {said}", q.id);
    }
}

#[test]
fn an_abstain_row_fails_a_hedged_stale_answer_and_takes_a_plain_refusal() {
    // Each stale answer is a refusal too ("I don't have", "I'm not sure", "I can't"), so only the
    // row's `not` fails it; each plain refusal echoes the question and must still pass.
    for (id, stale, plain) in [
        (
            "abstain-utsa",
            [
                "UTSA lost 30-6 at Texas on 19 Sep [1]; I don't have results for later games.",
                "Texas beat UTSA [1], but I'm not sure that was its last game.",
                "I'm not sure, but UTSA's last game was a loss to Texas [4].",
            ],
            [
                "The pages do not say who won UTSA's last football game.",
                "These pages are about Texas, not UTSA, and do not say who won UTSA's last game.",
                "The pages do not say whether UTSA won its last game.",
            ],
        ),
        (
            "abstain-fed-dec2026",
            [
                "The Fed cut rates by 25 basis points in December [1], but I'm not sure it was \
                 the 2026 meeting.",
                "It lowered the range to 3.5% to 3.75% [2], though I can't confirm the year.",
                "I'm not sure, but the Fed held rates steady [3].",
            ],
            [
                "The pages do not say what the Fed decided at its December 2026 meeting.",
                "The December 2026 meeting has not been held yet.",
                "The pages only cover the December 2025 meeting; they do not say what the Fed \
                 decided in December 2026.",
            ],
        ),
        (
            "abstain-superbowl-2027",
            [
                "Seattle won [1], though I can't confirm it was the 2027 game.",
                "The Seattle Seahawks beat the Patriots 29-13 [4], but I'm not sure that was the \
                 2027 game.",
                "I can't confirm it, but the winner was the Seattle Seahawks [4].",
            ],
            [
                "The pages do not say who won the 2027 Super Bowl.",
                "The 2027 Super Bowl has not been played yet; it is on February 14, 2027.",
                "The pages only give odds for Super Bowl LXI, with the Los Angeles Rams favourites \
                 to win; they do not say who won.",
            ],
        ),
    ] {
        let q = row(id);
        for said in stale {
            assert!(!replay_right(&q, said), "{id}: {said}");
        }
        for said in plain {
            assert!(replay_right(&q, said), "{id}: {said}");
        }
    }
}

#[test]
fn an_abstain_row_takes_a_refusal_of_an_event_still_to_come_or_one_its_pages_do_not_cover() {
    // The final review's probe: each a right refusal in its own words, with no `not` regex
    // involved.
    let wrong: Vec<_> = [
        (
            "abstain-fed-dec2026",
            "The Fed's December 2026 meeting has not happened yet; it is scheduled for December \
             8-9.",
        ),
        (
            "abstain-fed-dec2026",
            "The December 2026 meeting has not taken place yet.",
        ),
        (
            "abstain-fed-dec2026",
            "The Fed has not yet announced a decision for December 2026.",
        ),
        (
            "abstain-fed-dec2026",
            "The pages only cover the December 2025 meeting [1]; the December 2026 meeting is \
             still to come.",
        ),
        (
            "abstain-fed-dec2026",
            "None of the pages covers the December 2026 meeting; they are about December 2025 [1].",
        ),
        (
            "abstain-superbowl-2027",
            "Super Bowl LXI is not until February 2027, so there is no winner yet.",
        ),
        (
            "abstain-superbowl-2027",
            "No one has won the 2027 Super Bowl yet; it is played in February 2027.",
        ),
        (
            "abstain-utsa",
            "These pages do not cover UTSA's latest game.",
        ),
        (
            "abstain-iphone19",
            "There is no iPhone 19 yet; the pages only hold rumours.",
        ),
    ]
    .into_iter()
    .filter(|(id, said)| !replay_right(&row(id), said))
    .collect();
    assert!(wrong.is_empty(), "{wrong:#?}");
}

#[test]
fn a_claim_after_a_hedge_and_a_conjunction_fails_and_betting_odds_are_no_score() {
    // A hedge, then the claim after "but"; and a real score after one.
    let stated: Vec<_> = [
        ("abstain-utsa", "I'm not sure, but UTSA won."),
        (
            "abstain-fed-dec2026",
            "I can't confirm, but the Fed cut rates again in December.",
        ),
        (
            "abstain-superbowl-2027",
            "I don't know, but the Kansas City Chiefs won it.",
        ),
        (
            "abstain-superbowl-2027",
            "I can't confirm the year, but the game was won 27-24 [4].",
        ),
    ]
    .into_iter()
    .filter(|(id, said)| replay_right(&row(id), said))
    .collect();
    assert!(stated.is_empty(), "{stated:#?}");
    // The row's own page 3 shows odds such as 8-1 and 12-1, which a refusal may quote.
    let odds = "I can't tell who won because the pages only show odds, with the Bills at 8-1 and \
                the Jets at 12-1.";
    assert!(replay_right(&row("abstain-superbowl-2027"), odds), "{odds}");
}

#[test]
fn a_citation_inside_a_sentence_does_not_break_the_regexes() {
    let mancity = row("sport-mancity");
    for said in [
        "Aston Villa [1] beat Manchester City 2-1.",
        "Aston Villa [could not verify] beat Manchester City 2-1.",
        "Aston Villa [1, could not verify] beat Manchester City 2-1.",
        "Aston Villa [2-3] beat Manchester City 2-1.",
    ] {
        assert!(replay_right(&mancity, said), "{said}");
    }
}

#[test]
fn a_row_with_no_answer_regex_or_an_unknown_key_fails_the_set() {
    let mut typo = open();
    typo["nots"] = json!(["closed on saturdays"]);
    let empty = web("market-open", "When does it open?", &[], false);
    for (row, says) in [(typo, "nots"), (empty, "market-open")] {
        let dir = market(&[row]);
        let e = bench::replay_questions(dir.path()).unwrap_err();
        assert!(format!("{e:#}").contains(says), "{e:#}");
    }
}

/// The market page: its opening hour is on line 3.
const PAGE: &str = "# Riverton Market\n\nThe Riverton market opens at 8 am every Saturday.\n";
/// A worker's list for the market page.
const FACT: &str = r#"[{"line": 3, "claim": "The Riverton market opens at 8 am"}]"#;
/// The rows' day, which is not the machine's: the readers are told it.
const DAY: &str = "2026-10-20";

/// A web question on the saved market page.
fn web(id: &str, question: &str, answer: &[&str], abstain: bool) -> Value {
    json!({"id": id, "kind": if abstain { "abstain" } else { "hours" }, "question": question,
           "today": DAY, "answer": answer, "not": ["closed on saturdays"], "abstain": abstain,
           "pages": "web/market"})
}

fn open() -> Value {
    web(
        "market-open",
        "When does the Riverton market open?",
        &["opens at 8 am"],
        false,
    )
}

fn close() -> Value {
    web(
        "market-close",
        "When does the Riverton market close?",
        &[],
        true,
    )
}

/// A replay set of `questions` and a question about files, whose saved search found the market
/// page.
fn market(questions: &[Value]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let files = json!({"id": "files-x", "kind": "files", "question": "What is in my notes?",
                       "today": DAY, "answer": ["x"], "folder": "files"});
    let lines: Vec<String> = (questions.iter().chain([&files]))
        .map(Value::to_string)
        .collect();
    fs::write(dir.path().join("questions.jsonl"), lines.join("\n")).unwrap();
    let url = "https://riverton.example/";
    saved_page(dir.path(), "market", url, "Riverton Market", PAGE);
    dir
}

/// Saves `web/<name>` in the set `dir`: a search that found one page, `url`, of `title` and `text`.
fn saved_page(dir: &Path, name: &str, url: &str, title: &str, text: &str) {
    let saved = dir.join("web").join(name);
    fs::create_dir_all(&saved).unwrap();
    let hits = json!({"query": "", "hits": [{"title": title, "url": url,
                      "snippet": "Stalls and hours."}]});
    fs::write(saved.join("hits.json"), hits.to_string()).unwrap();
    let page = json!({"url": url, "final_url": url, "fetched_at": "2026-10-20T07:00:00Z",
                      "sha256": "aa", "title": title, "text": text,
                      "bytes": text.len(), "truncated": false, "published": null});
    fs::write(saved.join("1.json"), page.to_string()).unwrap();
}

/// The replay set's row `id` as its line holds it, its saved pages at `web/<pages>`.
fn line_of(id: &str, pages: &str) -> Value {
    let text = fs::read_to_string(Path::new(REPLAY).join("questions.jsonl")).unwrap();
    let mut row: Value = (text.lines())
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|r| r["id"] == id)
        .unwrap();
    row["pages"] = json!(format!("web/{pages}"));
    row
}

/// The rows of the set in `set`, answered by the plain reader alone, which says `said` in turn.
fn plain(set: &Path, said: &[&str]) -> Vec<ReplayRow> {
    let fake = Fake::new(32768);
    let set = bench::replay_set(set).unwrap();
    for s in said {
        fake.say(s);
    }
    let mut report = Report::default();
    bench::run_replay(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        Workers::default(),
        &set,
        &[Reader::Plain],
        &mut report,
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    assert_eq!(fake.unsaid(), 0);
    report.replay.unwrap().rows
}

#[test]
fn a_note_under_the_answer_never_changes_its_verdict() {
    // market-open's `not` holds the number note's words, and market-close passes on them as an
    // abstain, so either verdict flips if a note is scored.
    let mut open = open();
    open["not"] = json!(["not in the pages"]);
    let dir = market(&[open, close()]);
    // No 40 is on the page, so each answer gets the note.
    let rows = plain(
        dir.path(),
        &[
            "The market opens at 8 am [1], 40 weeks a year.",
            "It closes at 4 pm [1], 40 weeks a year.",
        ],
    );
    let scored: Vec<_> = (rows.iter())
        .map(|r| (r.id.as_str(), r.right, r.notes.clone()))
        .collect();
    let note = vec!["_40 is not in the pages it cites._".to_string()];
    assert_eq!(
        scored,
        [
            ("market-open", true, note.clone()),
            ("market-close", false, note)
        ]
    );
}

#[test]
fn a_url_in_the_sources_never_changes_the_verdict() {
    // sport-mancity's regexes match the score in its page's address, and abstain-inject's `not`
    // the hour in its, so either verdict flips if the source list is scored.
    let dir = tempfile::tempdir().unwrap();
    let rows = [
        line_of("sport-mancity", "villa"),
        line_of("abstain-inject", "market"),
    ];
    let lines: Vec<String> = rows.iter().map(Value::to_string).collect();
    fs::write(dir.path().join("questions.jsonl"), lines.join("\n")).unwrap();
    let results = "# Results\n\nAston Villa 2-1 Manchester City, 24 October.\n";
    let villa = "https://results.example/aston-villa-2-1-manchester-city";
    saved_page(dir.path(), "villa", villa, "Results", results);
    let gate = "# Riverton Market\n\nThe opening hours are posted at the gate.\n";
    let market = "https://riverton.example/opens-4am";
    saved_page(dir.path(), "market", market, "Riverton Market", gate);
    let rows = plain(
        dir.path(),
        &[
            "Manchester City won [1].",
            "The page [1] does not give an opening time.",
        ],
    );
    let scored: Vec<_> = (rows.iter())
        .map(|r| (r.id.as_str(), r.right, r.notes.is_empty()))
        .collect();
    assert_eq!(
        scored,
        [
            ("sport-mancity", false, true),
            ("abstain-inject", true, true)
        ]
    );
}

#[test]
fn both_readers_answer_each_web_question_on_its_day_and_each_answer_is_scored() {
    let fake = Fake::new(32768);
    let dir = market(&[open(), close()]);
    let set = bench::replay_set(dir.path()).unwrap();
    // market-open, plain first: the plain answer breaks its line inside the match; the chunks
    // writer's also says what the `not` regex rules out. market-close, chunks first: the chunks
    // writer guesses an hour; plain says the pages do not answer.
    fake.say("<think>Line 3 has it.</think>The market opens at 8\nam on Saturdays [1].");
    fake.say(FACT);
    fake.say(
        "<think>One fact.</think>The market opens at 8 am [1], but it is closed on Saturdays.",
    );
    fake.say(FACT);
    fake.say("It closes at 4 pm [1].");
    fake.say(NOT_FOUND);
    let out = dir.path().join("report.json");
    // The rows in the report file each time a row is said.
    let mut on_disk = Vec::new();
    let mut report = Report::default();
    bench::run_replay(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        Workers::default(),
        &set,
        &[Reader::Plain, Reader::Chunks],
        &mut report,
        |r| Ok(fs::write(&out, serde_json::to_string(r)?)?),
        |_| {
            let r: Report = serde_json::from_str(&fs::read_to_string(&out).unwrap()).unwrap();
            on_disk.push(r.replay.unwrap().rows.len());
        },
    )
    .unwrap();
    assert_eq!(fake.unsaid(), 0);
    assert_eq!(on_disk, [1, 2, 3, 4]);
    let replay = report.replay.as_ref().unwrap();
    assert_eq!(replay.skipped, 1);
    let scored: Vec<_> = (replay.rows.iter())
        .map(|r| (r.id.as_str(), r.reader, r.right, r.error.clone()))
        .collect();
    assert_eq!(
        scored,
        [
            ("market-open", Reader::Plain, true, None),
            ("market-open", Reader::Chunks, false, None),
            ("market-close", Reader::Plain, true, None),
            ("market-close", Reader::Chunks, false, None),
        ]
    );
    // The answers whole, as the user reads them above their sources.
    assert_eq!(
        replay.rows[0].answer,
        "The market opens at 8\nam on Saturdays [1]."
    );
    assert_eq!(
        replay.rows[1].answer,
        "The market opens at 8 am [1], but it is closed on Saturdays."
    );
    assert!(
        (replay.rows.iter()).all(|r| r.prompt_tokens > 0 && r.wall_ms > 0.0),
        "{:?}",
        replay.rows
    );
    let thought: Vec<_> = replay.rows.iter().map(|r| r.thinking_tokens > 0).collect();
    assert_eq!(thought, [true, true, false, false]);
    // Each reader was told the rows' day, not the machine's.
    let fed = fake.fed();
    assert!(fed.contains("Today is October 20 2026."), "plain: {fed}");
    assert!(fed.contains("Today is 2026-10-20."), "chunks: {fed}");
    // One untimed turn, an empty message, before the first row; then plain reads first on even
    // questions and the chunks workers on odd ones, so neither reader always runs on a GPU the
    // other warmed.
    let at = |s: &str| fed.find(s).unwrap_or_else(|| panic!("{s:?} in {fed}"));
    let first = at("Question: When does the Riverton market open?\n\n");
    assert!(at("user<|end_header_id|>\n\n<|eot_id|>") < first, "{fed}");
    for (q, plain_first) in [("open", true), ("close", false)] {
        let plain = at(&format!("Question: When does the Riverton market {q}?\n\n"));
        let chunks = at(&format!(
            "2026-10-20.\nQuestion: When does the Riverton market {q}?"
        ));
        assert_eq!(plain < chunks, plain_first, "{q}: {fed}");
    }
    // The settings each reader ran with: the plain reader's own thinking budget and no pages'
    // days (a small model's), the chunks writer's from the bench's, and the workers' prefix cache
    // as passed.
    assert_eq!(
        (
            replay.plain_thinking_budget,
            replay.page_dates,
            replay.chunks_thinking_budget,
            replay.workers_prefix_cache
        ),
        (READ_THINKING, false, Thinking::default().budget, false)
    );
    let table = bench::replay_table(replay);
    println!("{table}");
    for line in [
        "1 file and report question skipped: plain reader for files is a later task",
        "market-open plain pass, chunks fail",
        "market-close plain pass, chunks fail",
    ] {
        assert!(table.contains(line), "{line} in {table}");
    }
}

#[test]
fn reader_plain_runs_the_plain_reader_alone() {
    let fake = Fake::new(32768);
    let dir = market(&[open(), close()]);
    let set = bench::replay_set(dir.path()).unwrap();
    fake.say("The market opens at 8 am [1].");
    fake.say(NOT_FOUND);
    let mut report = Report::default();
    // Plain uses no workers, but the report still says how they were set.
    let workers = Workers {
        prefix_cache: true,
        ..Workers::default()
    };
    bench::run_replay(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        workers,
        &set,
        &[Reader::Plain],
        &mut report,
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    assert_eq!(fake.unsaid(), 0);
    let replay = report.replay.unwrap();
    assert_eq!(replay.readers, [Reader::Plain]);
    assert!(replay.workers_prefix_cache);
    let scored: Vec<_> = (replay.rows.iter())
        .map(|r| (r.id.as_str(), r.reader, r.right))
        .collect();
    assert_eq!(
        scored,
        [
            ("market-open", Reader::Plain, true),
            ("market-close", Reader::Plain, true)
        ]
    );
    assert!(!bench::replay_table(&replay).contains("disagree"));
}

#[test]
fn a_question_the_engine_refuses_is_an_error_row_and_the_next_one_runs() {
    // The market page's prompts fit; the long question's do not.
    let fake = Fake::new(6144);
    let long = web(
        "market-long",
        &"When does the Riverton market open? ".repeat(600),
        &["opens at 8 am"],
        false,
    );
    let dir = market(&[long, close()]);
    let mut set = bench::replay_set(dir.path()).unwrap();
    let audit = dir.path().join("report-audit.jsonl");
    set.audit = Some(audit.clone());
    // The lines of the audit log's copy, one file.
    let copied = || fs::read_to_string(&audit).map_or(0, |text| text.lines().count());
    // market-close runs the chunks reader first: its worker finds nothing, so no writer runs.
    fake.say("[]");
    fake.say(NOT_FOUND);
    let mut report = Report::default();
    let mut seen = Vec::new();
    bench::run_replay(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        Workers::default(),
        &set,
        &[Reader::Plain, Reader::Chunks],
        &mut report,
        |_| Ok(()),
        |_| seen.push(copied()),
    )
    .unwrap();
    assert_eq!(fake.unsaid(), 0);
    // The audit log outlives the throwaway data folder: copied after each row that errored, then
    // at the end.
    assert!(
        seen[0] > 0 && seen[2] == seen[1] && seen[3] == seen[1],
        "{seen:?}"
    );
    assert!(copied() > seen[3], "{seen:?}");
    let rows = &report.replay.as_ref().unwrap().rows;
    let ran: Vec<_> = (rows.iter())
        .map(|r| (r.id.as_str(), r.right, r.error.is_some()))
        .collect();
    assert_eq!(
        ran,
        [
            ("market-long", false, true),
            ("market-long", false, true),
            ("market-close", true, false),
            ("market-close", true, false),
        ],
        "{rows:?}"
    );
    assert!(report.failed());
}

#[test]
fn a_plain_answer_that_fell_back_to_the_chunks_reader_says_so_on_its_row() {
    // Ten hits with long snippets: the plain reader's prompt holds them and is over the context
    // at the smallest cards; the chunks reader's workers never see them.
    let dir = market(&[open()]);
    let hit = json!({"title": "Market", "url": "https://riverton.example/",
                     "snippet": "Stalls, hours and parking. ".repeat(150)});
    let hits = json!({"query": "", "hits": vec![hit; 10]});
    fs::write(dir.path().join("web/market/hits.json"), hits.to_string()).unwrap();
    let rows = plain(dir.path(), &[FACT, "The market opens at 8 am [1]."]);
    let r = &rows[0];
    assert!((r.right, r.error.as_deref()) == (true, None), "{r:?}");
    let why = r.fallback.as_deref().unwrap_or_default();
    assert!(why.starts_with("the prompt is "), "{r:?}");
    let replay = Replay {
        readers: vec![Reader::Plain],
        rows,
        ..Replay::default()
    };
    let table = bench::replay_table(&replay);
    assert!(table.contains("thinking tokens; fell back 1\n"), "{table}");
}

#[test]
fn a_plain_fallback_is_counted_and_left_out_of_the_paired_time() {
    let row = |id: &str, reader, secs: f64, fell: bool| ReplayRow {
        id: id.into(),
        kind: "news".into(),
        reader,
        right: true,
        wall_ms: secs * 1e3,
        fallback: fell.then(|| "the prompt is 9000 tokens, the room 8000".into()),
        ..ReplayRow::default()
    };
    let (p, c) = (Reader::Plain, Reader::Chunks);
    let replay = Replay {
        readers: vec![p, c],
        rows: vec![
            row("q1", p, 2.0, false),
            row("q1", c, 1.0, false),
            // The chunks reader answered for plain, so its time is chunks'.
            row("q2", p, 1.0, true),
            row("q2", c, 1.0, false),
            row("q3", p, 0.5, false),
            row("q3", c, 1.0, false),
        ],
        ..Replay::default()
    };
    let table = bench::replay_table(&replay);
    println!("{table}");
    for line in [
        "mean 0 prompt tokens, 0 thinking tokens; fell back 1\nchunks ",
        "mean 0 prompt tokens, 0 thinking tokens; fell back 0\npaired ",
        // q1 and q3 only, 2.0 and 0.5.
        "paired time plain/chunks: median 1.25x (n 2), without the 1 question plain fell back on, \
         as the chunks reader answered it\n",
        "paired time 1.25x, bar 1.2x no: plain does not pass, fell back 1\n",
    ] {
        assert!(table.contains(line), "{line} in {table}");
    }
}

#[test]
fn the_table_says_each_readers_n_the_paired_time_and_the_decision() {
    let row = |id: &str, reader, abstain, right, secs: f64, error: bool| ReplayRow {
        id: id.into(),
        kind: "news".into(),
        reader,
        abstain,
        right,
        wall_ms: secs * 1e3,
        error: error.then(|| "engine".into()),
        ..ReplayRow::default()
    };
    let (p, c) = (Reader::Plain, Reader::Chunks);
    let replay = Replay {
        readers: vec![p, c],
        rows: vec![
            row("q1", p, false, true, 2.0, false),
            row("q1", c, false, true, 1.0, false),
            row("q2", p, false, true, 1.0, false),
            row("q2", c, false, false, 1.0, false),
            row("q3", p, true, true, 3.0, false),
            row("q3", c, true, false, 0.5, true),
            row("q4", p, false, false, 0.5, true),
            row("q4", c, false, true, 2.0, false),
        ],
        ..Replay::default()
    };
    let table = bench::replay_table(&replay);
    println!("{table}");
    for line in [
        "plain: answerable 2/3 right, abstain 1/1 right; time p50 2.0 s, p90 3.0 s (n 3);",
        "chunks: answerable 2/3 right, abstain 0/1 right; time p50 1.0 s, p90 2.0 s (n 3);",
        // q1 and q2 only, 2.0 and 1.0: an error in either reader drops the question, and an even
        // count's median is between its middle two.
        "paired time plain/chunks: median 1.50x (n 2)",
        "decision, plain as the default: answerable 2/3 right, bar 3 (85%) no; abstain 1/1 right, \
         bar 1 yes; paired time 1.50x, bar 1.2x no: plain does not pass",
    ] {
        assert!(table.contains(line), "{line} in {table}");
    }
}

#[test]
fn the_table_prints_each_abstain_rows_answers_whole_for_both_readers() {
    let row = |id: &str, reader, abstain, right, answer: &str| ReplayRow {
        id: id.into(),
        kind: if abstain { "abstain" } else { "news" }.into(),
        reader,
        abstain,
        right,
        answer: answer.into(),
        wall_ms: 1e3,
        ..ReplayRow::default()
    };
    let (p, c) = (Reader::Plain, Reader::Chunks);
    // Longer than the 160 characters a disagree line shows of an answer.
    let long = "The pages only cover the December 2025 meeting, when the Fed cut its rate [1]; the \
                December 2026 meeting has not happened yet, so no page can say what it decided.";
    let replay = Replay {
        readers: vec![p, c],
        rows: vec![
            row("q1", p, false, true, "Aston Villa won."),
            row("q1", c, false, true, "Villa won 2-1."),
            row("a1", p, true, true, long),
            row("a1", c, true, true, NOT_FOUND),
            row("a2", p, true, false, "The market opens at\n4 am [1]."),
            row("a2", c, true, true, NOT_FOUND),
        ],
        ..Replay::default()
    };
    let table = bench::replay_table(&replay);
    println!("{table}");
    let (_, abstain) = table.split_once("abstain:\n").expect("an abstain section");
    // Every abstain row, the readers agreeing too, each answer whole on one line; an answerable
    // row the readers agree on is not listed.
    assert_eq!(
        abstain.lines().collect::<Vec<_>>(),
        [
            format!("a1 plain pass, chunks pass | plain: {long} | chunks: {NOT_FOUND}"),
            format!(
                "a2 plain fail, chunks pass | plain: The market opens at 4 am [1]. | chunks: \
                 {NOT_FOUND}"
            ),
        ]
    );
    // Listed once: not under disagree as well.
    assert_eq!(table.matches("a2 plain fail").count(), 1, "{table}");
    assert!(!table.contains("q1 "), "{table}");
}

#[test]
fn the_template_prints_the_rows_day() {
    let mut early = open();
    early["today"] = json!("2026-10-03");
    let dir = market(&[early.clone()]);
    let set = bench::replay_set(dir.path()).unwrap();
    // As a chat template prints today's date (`%d %b %Y`).
    assert_eq!(set.date().as_deref(), Some("03 Oct 2026"));
    let dir = market(&[open(), close()]);
    let set = bench::replay_set(dir.path()).unwrap();
    assert_eq!(set.date().as_deref(), Some("20 Oct 2026"));
    // A run renders one date, so its web questions share one day.
    let dir = market(&[early, close()]);
    let e = bench::replay_set(dir.path()).unwrap_err();
    assert!(format!("{e:#}").contains("one day"), "{e:#}");
}

#[test]
fn a_replay_row_says_its_outcome_and_where_its_time_went() {
    let dir = market(&[open(), close()]);
    let rows = plain(
        dir.path(),
        &[
            "The market opens at 8 am [1].",
            "The pages do not say when it closes.",
        ],
    );
    let outcomes: Vec<_> = rows.iter().map(|r| r.outcome).collect();
    assert_eq!(
        outcomes,
        [
            Some(bench::Outcome::Answered),
            Some(bench::Outcome::Refused)
        ]
    );
    let s = rows[0].stages.unwrap();
    // Saved pages: nothing searched or fetched; one reading turn; no chat to show anything.
    assert_eq!(
        (s.search_ms, s.fetch_ms, s.first_visible_ms),
        (None, None, None)
    );
    assert!(
        s.prefill_ms_sum.is_some() && s.generation_ms_sum.is_some(),
        "{s:?}"
    );
}

/// A fake of `web_search`, which needs a provider on the internet: it returns its hits whatever
/// the query.
struct FakeSearch(String);

impl larch_agent::tool::Tool for FakeSearch {
    fn spec(&self) -> larch_agent::tool::ToolSpec {
        larch_agent::tool::ToolSpec {
            name: "web_search".into(),
            description: "The fake web_search.".into(),
            schema: json!({"type": "object", "properties": {
                "query": {"type": "string"}, "n": {"type": "integer"}}, "required": ["query"]}),
            permission: larch_agent::tool::Permission::Read,
            untrusted_results: true,
            max_result_tokens: 1500,
            max_args_tokens: larch_agent::tool::ARGS_TOKENS,
        }
    }

    fn call(&self, _: &Value, _: &ToolCtx) -> Result<String, larch_agent::tool::ToolError> {
        Ok(self.0.clone())
    }
}

/// The run's day in the live tests, which is not the rows' `DAY`.
const LIVE_DAY: &str = "2026-10-30";

/// `open()` under another ID, a changing row whose answer was last checked on `as_of`.
fn changing(id: &str, as_of: &str) -> Value {
    let mut row = open();
    row["id"] = json!(id);
    row["answer_as_of"] = json!(as_of);
    row
}

#[test]
fn a_live_run_searches_saves_what_it_read_and_keeps_its_own_section() {
    let fake = Fake::new(32768);
    let url = common::serve(PAGE);
    // A row whose every live run is a human check, however fresh its answer.
    let mut always = changing("market-always", "2026-10-29");
    always["live_check"] = json!(true);
    let dir = market(&[
        open(),
        changing("market-stale", "2026-10-01"),
        changing("market-fresh", "2026-10-25"),
        always,
    ]);
    let mut set = bench::replay_set(dir.path()).unwrap();
    let audit = dir.path().join("live-audit.jsonl");
    set.audit = Some(audit);
    // The line a real web_search leaves; the fake one leaves none.
    let searches = set.data.path().join("agent/search");
    fs::create_dir_all(&searches).unwrap();
    fs::write(searches.join("log.jsonl"), "{\"provider\":\"brave\"}\n").unwrap();
    let mut tools = larch_agent::tool::Registry::builtin();
    let hits = json!([{"title": "Riverton Market", "url": url, "snippet": "Stalls and hours."}]);
    tools.replace(Box::new(FakeSearch(hits.to_string())));
    for _ in 0..4 {
        fake.say("The market opens at 8 am [1].");
    }
    let mut report = Report::default();
    bench::run_live(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        Workers::default(),
        &set,
        &[Reader::Plain],
        &tools,
        LIVE_DAY,
        &mut report,
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    assert_eq!(fake.unsaid(), 0);
    // Never summed with the saved pages' numbers.
    assert!(report.replay.is_none());
    let live = report.live.unwrap();
    let rows: Vec<_> = (live.rows.iter())
        .map(|r| (r.id.as_str(), r.right, r.check, r.error.is_none()))
        .collect();
    // The stale row's answer goes to a human, unscored; a changing row checked this week scores.
    assert_eq!(
        rows,
        [
            ("market-open", true, false, true),
            ("market-stale", false, true, true),
            ("market-fresh", true, false, true),
            ("market-always", false, true, true)
        ]
    );
    let s = live.rows[0].stages.unwrap();
    assert!(s.search_ms.is_some() && s.fetch_ms.is_some(), "{s:?}");
    assert_eq!(live.rows[0].sources, std::slice::from_ref(&url));
    // What it read is saved as the replay set saves pages, so it can refresh the set.
    let saved_at = dir
        .path()
        .join("web/market-open")
        .join(format!("live-{LIVE_DAY}"));
    let (hits, pages) = saved(&saved_at).unwrap();
    assert_eq!((hits.len(), pages.len()), (1, 1));
    assert!(pages[0].text.contains("opens at 8 am"), "{}", pages[0].text);
    let table = bench::replay_table(&live);
    assert!(table.contains("human check"), "{table}");
    assert!(table.contains("market-stale"), "{table}");
    // Which provider answered outlives the throwaway data folder.
    let kept = fs::read_to_string(dir.path().join("live-search.jsonl")).unwrap();
    assert!(kept.contains("brave"), "{kept}");
}

#[test]
fn a_repeated_run_keeps_every_run_and_says_which_rows_passed_each() {
    let fake = Fake::new(32768);
    let dir = market(&[open(), close()]);
    let mut set = bench::replay_set(dir.path()).unwrap();
    set.repeat = 2;
    for s in [
        "The market opens at 8 am [1].",
        "The pages do not say when it closes.",
        "The market opens at 8 am [1].",
        "It closes at 4 pm [1].",
    ] {
        fake.say(s);
    }
    let mut report = Report::default();
    bench::run_replay(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        Workers::default(),
        &set,
        &[Reader::Plain],
        &mut report,
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    let replay = report.replay.unwrap();
    let runs: Vec<_> = (replay.rows.iter())
        .map(|r| (r.id.as_str(), r.run, r.right))
        .collect();
    assert_eq!(
        runs,
        [
            ("market-open", 1, true),
            ("market-close", 1, true),
            ("market-open", 2, true),
            ("market-close", 2, false)
        ]
    );
    let r = replay.repeated.clone().unwrap();
    assert_eq!((r.runs, r.mean, r.all_pass), (2, 0.75, 0.5));
    let table = bench::replay_table(&replay);
    assert!(table.contains("all_pass"), "{table}");
}

/// Every changing row of the shipped set says the day its answer was checked, and the rows whose
/// answer moves within days, or that abstain only because their saved pages lack the answer, are
/// a human check in every live run.
#[test]
fn the_shipped_changing_rows_are_dated_and_the_fast_ones_a_live_check() {
    let (rows, _) = bench::replay_questions(Path::new(REPLAY)).unwrap();
    for r in &rows {
        if ["sports", "prices", "news"].contains(&r.kind.as_str()) {
            assert_eq!(r.answer_as_of.as_ref(), Some(&r.today), "{}", r.id);
        }
    }
    let checked: Vec<&str> = (rows.iter())
        .filter(|r| r.live_check)
        .map(|r| r.id.as_str())
        .collect();
    for id in [
        "sport-lakers",
        "sport-f1",
        "sport-ut-latest",
        "sport-ut-bare",
        "sport-tamu",
        "sport-mancity",
        "price-gold",
        "news-fed",
        "news-unemployment",
        "news-artemis",
        "abstain-utsa",
        "abstain-springfield",
    ] {
        assert!(checked.contains(&id), "{id}");
    }
}

/// A live search that finds nothing is an error row: no wrong answer, a failed run, and no
/// evidence.
#[test]
fn a_live_search_that_fails_is_an_error_not_a_wrong_answer() {
    let fake = Fake::new(32768);
    let dir = market(&[open()]);
    let set = bench::replay_set(dir.path()).unwrap();
    let mut tools = larch_agent::tool::Registry::builtin();
    tools.replace(Box::new(FakeSearch("[]".into())));
    let mut report = Report::default();
    bench::run_live(
        &fake.larch,
        Family::Llama3,
        Thinking::default(),
        Workers::default(),
        &set,
        &[Reader::Plain],
        &tools,
        LIVE_DAY,
        &mut report,
        |_| Ok(()),
        |_| {},
    )
    .unwrap();
    let row = &report.live.as_ref().unwrap().rows[0];
    assert!(row.error.is_some() && !row.right, "{row:?}");
    assert_eq!(row.outcome, Some(bench::Outcome::Error));
    assert!(report.failed());
    let asked = [("market-open".to_string(), "when does it open?".to_string())];
    assert!(bench::evidence(&report, &asked.into(), "now").is_empty());
}

#[test]
fn a_passage_card_keeps_its_best_chunk_past_a_list() {
    // At 2,000 bytes the lead and a list's first chunk (a shop's related items) left no room for
    // the answer's.
    let nav: String = (1..=60)
        .map(|n| format!("- Related item {n}: cases and stands\n"))
        .collect();
    let prose = |topic: &str| {
        format!("{topic} is covered on the next pages of this site, with photos and the history of the line and of the old factory.\n\n").repeat(4)
    };
    let page = larch_agent::web::fetch::Page {
        text: format!(
            "# Headphones\n\n{}{nav}\n{}{}{}",
            prose("The shop"),
            prose("The case"),
            "The headphones weigh 254 grams with the cushions on, lighter than the last model by a few grams, and fold flat into the case. ".repeat(6),
            prose("The cable")
        ),
        url: "https://example.com/xm".into(),
        final_url: "https://example.com/xm".into(),
        fetched_at: String::new(),
        sha256: String::new(),
        title: "Headphones".into(),
        bytes: 0,
        truncated: false,
        published: None,
        pdf: false,
    };
    let card = passages("how much do the headphones weigh", &[page], PASSAGE_BYTES).remove(0);
    assert!(card.text.contains("weigh 254 grams"), "{}", card.text);
    assert!(card.text.len() <= PASSAGE_BYTES);
}

/// Offline: how many answered rows keep their answer in the first `READ` cards at each card size,
/// for full cards and passage cards, with the median and largest evidence in tokens (bytes / 4).
/// No model runs; `cargo test --test replay -- --ignored --nocapture card_budgets`.
#[test]
#[ignore]
fn card_budgets() {
    let web: Vec<ReplayQuestion> = (answered().into_iter())
        .filter(|r| r.pages.is_some())
        .collect();
    let saved_rows: Vec<_> = (web.iter())
        .map(|r| saved(&Path::new(REPLAY).join(r.pages.as_ref().unwrap())).unwrap())
        .collect();
    for passage in [false, true] {
        for per_page in [800, 1_200, 1_600, 2_000, 2_400, 3_200, 4_000, PER_PAGE] {
            let (mut found, mut sizes, mut lost) = (0, Vec::new(), Vec::new());
            let (mut carded, mut hit_sizes) = (0, Vec::new());
            for (row, (hits, pages)) in web.iter().zip(&saved_rows) {
                let pages = &pages[..READ.min(pages.len())];
                let latest = asks_latest(&row.question);
                let shown = match passage {
                    true => passages(&row.question, pages, per_page),
                    false => cards(&row.question, pages, per_page, latest),
                };
                let text = render(hits, &shown);
                sizes.push(text.len() / 4);
                hit_sizes.push(render(hits, &[]).len() / 4);
                // In the cards alone, without the search snippets.
                carded += usize::from(missed(row, &render(&[], &shown)).is_empty());
                match missed(row, &text).is_empty() {
                    true => found += 1,
                    false => lost.push(row.id.as_str()),
                }
            }
            sizes.sort_unstable();
            hit_sizes.sort_unstable();
            println!(
                "{} {per_page:>5} B/page: {found}/{} found ({carded} in the cards alone), \
                 tokens p50 {} max {} (hits p50 {}); lost {lost:?}",
                if passage { "passages" } else { "cards   " },
                web.len(),
                sizes[sizes.len() / 2],
                sizes.last().unwrap(),
                hit_sizes[hit_sizes.len() / 2],
            );
        }
    }
}
