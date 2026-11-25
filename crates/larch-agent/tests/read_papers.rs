//! read_papers: the files a call names, each paper cut to its reader's budget, the numbers check,
//! and the chat that shows each reader's summary as the answer, with no model step after it
//! unless the call asked a question.

mod common;

use std::fs;

use common::{Fake, call, events, serve, serve_as, spaced, text};
use larch_agent::{
    AgentEvent,
    skills::read_papers::{
        self, LEFT_OUT, MAX_PAPERS, PARTS, Paper, Papers, arxiv, at_once, direct, extract, fit,
        gather, parts, refit, tidy, unchecked, without_references,
    },
    tool::ToolCtx,
    workers::Workers,
};
use serde_json::json;

fn ctx(fake: &Fake) -> ToolCtx {
    ToolCtx {
        allowed: vec![fake.files()],
        data_dir: fake.data(),
        source: String::new(),
    }
}

#[test]
fn a_paper_that_fits_is_whole_and_a_long_one_keeps_its_start_and_end() {
    let short = "Title\nAbstract.\n";
    assert_eq!(fit(short, 1000), (short.to_owned(), false));
    let lines: String = (0..400)
        .map(|i| format!("line {i:03} of the paper\n"))
        .collect();
    let (shown, cut) = fit(&lines, 500);
    assert!(cut);
    // 2,000 bytes: about 1,400 from the start and 600 from the end, at line ends.
    assert!(shown.starts_with("line 000 of the paper\n"), "{shown}");
    assert!(shown.ends_with("line 399 of the paper\n"), "{shown}");
    assert!(shown.contains(&format!("\n{LEFT_OUT}\n")), "{shown}");
    assert!(shown.len() <= 2000 + LEFT_OUT.len() + 1, "{}", shown.len());
    assert!(
        shown
            .lines()
            .all(|l| l == LEFT_OUT || l.starts_with("line "))
    );
}

/// A paper of `filler` paragraphs: a front, numbered sections, a table with its caption, and a
/// conclusion.
fn paper() -> String {
    let para = |tag: &str| format!("{tag} {}\n\n", "words of the paper go on here. ".repeat(12));
    let mut out = String::from("Sparse Experts at Scale\nA. Author, B. Author\n\nAbstract\n");
    out += &para("ABSTRACT-TEXT");
    out += "1 Introduction\n";
    out += &para("INTRO-1");
    out += &para("INTRO-2");
    out += &para("INTRO-3");
    out += "2 Related Work\n";
    out += &para("RELATED-1");
    out += "3 Method\n";
    for i in 1..=6 {
        out += &para(&format!("METHOD-{i}"));
    }
    out += "Table 1: Accuracy of each model on the test set.\n\n";
    out += "Model 81.2 93.4\nOurs 85.0 95.1\n\n";
    out += "4 Results\n";
    for i in 1..=6 {
        out += &para(&format!("RESULTS-{i}"));
    }
    out += "5 Conclusion\n";
    out += &para("CONCLUSION-1");
    out += &para("CONCLUSION-2");
    out += "Appendix\n";
    out += &para("APPENDIX-1");
    out
}

#[test]
fn an_extract_past_its_budget_in_real_tokens_is_cut_again_to_fit() {
    let text: String = (0..400)
        .map(|i| format!("Paragraph {i} reports a 9.{i}% gain on the held-out set.\n\n"))
        .collect();
    let (shown, cut) = extract(&text, 1000, None);
    let mut paper = Paper {
        file: "p.txt".into(),
        text: text.clone(),
        shown: shown.clone(),
        cut,
    };
    refit(&mut paper, 1000, 1000, None);
    assert_eq!(paper.shown, shown);
    // A fifth more tokens than the bytes say: the extract shrinks by that much.
    refit(&mut paper, 1000, 1250, None);
    assert!(paper.cut);
    assert!(paper.shown.len() <= 800 * 4, "{}", paper.shown.len());
    assert!(paper.shown.len() > 700 * 4, "{}", paper.shown.len());
    // Still a tenth over: the next cut shrinks the last one, not the first budget.
    let last = paper.shown.len();
    refit(&mut paper, 1000, 1100, None);
    assert!(
        paper.shown.len() * 1100 <= last * 1000,
        "{} after {last}",
        paper.shown.len()
    );
}

#[test]
fn a_long_paper_keeps_its_front_headings_tables_and_key_paragraphs() {
    let text = paper();
    let (shown, cut) = extract(&text, 900, None);
    assert!(cut);
    assert!(shown.len() <= 3600, "{}", shown.len());
    for kept in [
        "Sparse Experts at Scale",
        "ABSTRACT-TEXT",
        "1 Introduction",
        "2 Related Work",
        "3 Method",
        "4 Results",
        "5 Conclusion",
        "Table 1: Accuracy",
        "Ours 85.0 95.1",
        "INTRO-1",
        "INTRO-2",
        "CONCLUSION-1",
        "CONCLUSION-2",
        "METHOD-1",
        "RESULTS-1",
        "RESULTS-2",
    ] {
        assert!(shown.contains(kept), "{kept} missing:\n{shown}");
    }
    for left in [
        "INTRO-3",
        "RELATED-1",
        "METHOD-2",
        "RESULTS-3",
        "Appendix",
        "APPENDIX-1",
    ] {
        assert!(!shown.contains(left), "{left} kept:\n{shown}");
    }
    // In the paper's order, with a mark where text is left out.
    let at = |s: &str| shown.find(s).unwrap();
    assert!(at("ABSTRACT-TEXT") < at("METHOD-1") && at("METHOD-1") < at("CONCLUSION-1"));
    assert!(shown.contains(&format!("\n{LEFT_OUT}\n")), "{shown}");
    // A text with no sections keeps its start and its end.
    let plain = "a line of plain text\n".repeat(400);
    assert_eq!(extract(&plain, 500, None), fit(&plain, 500));
    assert_eq!(extract(&text, 100_000, None), (text.clone(), false));
}

#[test]
fn a_focus_keeps_the_paragraph_that_answers_it() {
    let text = paper().replace(
        "METHOD-4 words",
        "METHOD-4 We employ a residual connection around each sub-layer. words",
    );
    let (unfocused, _) = extract(&text, 900, None);
    assert!(!unfocused.contains("METHOD-4"), "{unfocused}");
    let (shown, cut) = extract(&text, 900, Some("Why does it need a residual connection?"));
    assert!(cut);
    assert!(shown.len() <= 3600, "{}", shown.len());
    assert!(shown.contains("residual connection around"), "{shown}");
}

#[test]
fn the_references_go_only_from_a_heading_in_the_second_half() {
    let body = "Intro.\nMethod.\nResults.\nConclusion.\n".repeat(3);
    let paper = format!("{body}References\n[1] A. Author. A paper. 2020.\n");
    assert_eq!(without_references(&paper), body);
    let numbered = format!("{body}7. REFERENCES:\n[1] Someone.\n");
    assert_eq!(without_references(&numbered), body);
    // A heading in the first half is a table of contents or a related-work section.
    let early = format!("References\n{body}");
    assert_eq!(without_references(&early), early);
}

#[test]
fn tidy_trims_lines_and_keeps_one_blank_line() {
    assert_eq!(
        tidy("\n\n  Title  \n\n\n\nText\r\nmore\n\n"),
        "Title\n\nText\nmore\n"
    );
}

#[test]
fn numbers_a_summary_gives_are_checked_against_its_paper() {
    let paper = "Accuracy rose to 93.40% on 1,200 images, from 81.2 (Table 2).";
    let summary = "1. It reaches 93.4% on 1200 images, up from 81.2, and 97.1 on a GPT4 split; \
                   Table 3 shows 3B and 2 runs.";
    // 93.4 is in 93.40, 1200 is 1,200; 97.1 is not there; 3 and 2 are one digit, 4 and 3B
    // are glued to letters.
    assert_eq!(unchecked(summary, paper), ["97.1"]);
    // Never a match inside a longer number, before or after; trailing zeros are the same number.
    assert_eq!(unchecked("about 93.4", "it was 193.4"), ["93.4"]);
    assert_eq!(unchecked("95 samples", "We used 950 samples."), ["95"]);
    assert_eq!(unchecked("95 samples", "We used 95.3 samples."), ["95"]);
    assert_eq!(
        unchecked("95 samples and 93.4", "95.0 samples, 93.40."),
        Vec::<String>::new()
    );
    // A lone digit is a result before "%" or "x".
    assert_eq!(
        unchecked("an 8% gain, 3x faster", "a 2% gain, 2x faster"),
        ["8", "3"]
    );
    assert_eq!(
        unchecked("an 8% gain, 3x faster", "an 8% gain, 3x faster"),
        Vec::<String>::new()
    );
}

#[test]
fn a_call_reads_files_and_folders_in_order_each_once_and_at_most_eight() {
    let fake = Fake::new(4096);
    let dir = fake.files().join("papers");
    fs::create_dir_all(&dir).unwrap();
    for name in ["b.pdf", "a.txt", "notes.docx", "c.md"] {
        fs::write(dir.join(name), "x").unwrap();
    }
    fs::write(fake.files().join("z.txt"), "x").unwrap();
    let got = gather(&ctx(&fake), &["z.txt", "papers", "papers/a.txt"]).unwrap();
    let names: Vec<_> = got
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap().to_owned())
        .collect();
    assert_eq!(names, ["z.txt", "a.txt", "b.pdf", "c.md"]);

    for i in 0..MAX_PAPERS {
        fs::write(dir.join(format!("more{i}.txt")), "x").unwrap();
    }
    let err = gather(&ctx(&fake), &["papers"]).unwrap_err().to_string();
    assert!(err.contains("at most 8"), "{err}");
    let empty = fake.files().join("empty");
    fs::create_dir_all(&empty).unwrap();
    let err = gather(&ctx(&fake), &["empty"]).unwrap_err().to_string();
    assert!(err.contains("no papers"), "{err}");
    let outside = fake.data().to_str().unwrap().to_owned();
    fs::create_dir_all(fake.data()).unwrap();
    let err = gather(&ctx(&fake), &[&outside]).unwrap_err().to_string();
    assert!(err.contains("outside the allowed folders"), "{err}");
    // Nothing attached: a title is no file, and the answer says what to do instead.
    let none = ToolCtx {
        allowed: vec![],
        ..ctx(&fake)
    };
    let err = gather(&none, &["Attention Is All You Need"])
        .unwrap_err()
        .to_string();
    assert!(err.starts_with(read_papers::NOTHING_ATTACHED), "{err}");
}

#[cfg(unix)]
#[test]
fn a_pipe_is_not_a_paper_and_a_stalled_extractor_stops() {
    let fake = Fake::new(4096);
    let dir = fake.files().join("papers");
    fs::create_dir_all(&dir).unwrap();
    let fifo = dir.join("pipe.txt");
    assert!(
        std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    // Read, it would block with no writer: named, it is refused; in a folder, it is not a paper.
    let err = gather(&ctx(&fake), &["papers/pipe.txt"])
        .unwrap_err()
        .to_string();
    assert!(err.contains("is not a file"), "{err}");
    let err = gather(&ctx(&fake), &["papers"]).unwrap_err().to_string();
    assert!(err.contains("no papers"), "{err}");

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let start = std::time::Instant::now();
    let setter = {
        let stop = stop.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        })
    };
    let mut sleep = std::process::Command::new("sleep");
    sleep.arg("30");
    let err =
        read_papers::output_until(sleep, std::time::Duration::from_secs(60), &stop).unwrap_err();
    setter.join().unwrap();
    assert!(err.contains("stopped"), "{err}");
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
    let mut sleep = std::process::Command::new("sleep");
    sleep.arg("30");
    let never = std::sync::atomic::AtomicBool::new(false);
    let err = read_papers::output_until(sleep, std::time::Duration::from_millis(200), &never)
        .unwrap_err();
    assert!(err.contains("ran past"), "{err}");
    let mut echo = std::process::Command::new("echo");
    echo.arg("text");
    assert_eq!(
        read_papers::output_until(echo, std::time::Duration::from_secs(5), &never).unwrap(),
        b"text\n"
    );
}

#[test]
fn readers_run_one_per_paper_up_to_the_workers_and_the_kv_pool() {
    let workers = |max_workers| Workers {
        max_workers,
        prefix_cache: true,
    };
    // The chat waits on the call, so all eight run in the default eight slots.
    assert_eq!(at_once(8, workers(8), 0, 5000), 8);
    assert_eq!(at_once(3, workers(8), 0, 5000), 3);
    assert_eq!(at_once(8, workers(2), 0, 5000), 2);
    // Each reader takes whole 16-token blocks: 5,000 tokens are 313 blocks, so 625 blocks
    // (10,000 tokens) hold one and 626 hold two; 5,008 tokens fill 313 exactly.
    assert_eq!(at_once(8, workers(8), 625, 5000), 1);
    assert_eq!(at_once(8, workers(8), 626, 5000), 2);
    assert_eq!(at_once(8, workers(8), 626, 5008), 2);
    assert_eq!(at_once(8, workers(8), 626, 5009), 1);
    assert_eq!(at_once(8, workers(8), 1252, 5000), 4);
    assert_eq!(at_once(8, workers(8), 1251, 5000), 3);
    assert_eq!(at_once(8, workers(8), 60, 5000), 1);
}

/// A chat with `max_workers` 1, so its readers run one at a time and each takes its scripted
/// reply in turn.
fn one_reader_at_a_time(fake: &Fake) -> larch_agent::Agent {
    fs::create_dir_all(fake.data().join("agent")).unwrap();
    fs::write(fake.data().join("agent/config.toml"), "max_workers = 1\n").unwrap();
    fake.agent()
}

#[test]
fn the_chat_shows_each_readers_summary_as_the_answer() {
    let fake = Fake::new(32768);
    let mut agent = one_reader_at_a_time(&fake);
    let dir = fake.files().join("papers");
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("a.txt"),
        "Fast Sorting\nWe sort 2.5x faster than quicksort.\nReferences\n[1] Hoare.\n",
    )
    .unwrap();
    fs::write(dir.join("b.md"), "Slow Trees\nTrees grow 10 cm a year.\n").unwrap();
    // "The papers" holds the step to a call, under the call grammar's spacing.
    fake.say(&spaced("read_papers", json!({"papers": ["papers"]})));
    fake.say("Fast Sorting sorts 2.5x faster.");
    fake.say("Slow Trees grow 12 cm a year.");
    fake.say("A model step after the readers.");
    let got = events(agent.turn("summarize the papers in papers").unwrap());
    let want = "## [1] a.txt\nFast Sorting sorts 2.5x faster.\n\n\
                ## [2] b.md\nSlow Trees grow 12 cm a year.\n\
                Check these numbers, not found in the paper's text: 12";
    assert_eq!(text(&got), want, "{got:?}");
    assert!(
        matches!(&got[0], AgentEvent::ToolCall { name, .. } if name == "read_papers"),
        "{got:?}"
    );
    // Each summary shows as its reader writes it, before the call's result.
    let result = (got.iter())
        .position(|e| matches!(e, AgentEvent::ToolResult { .. }))
        .unwrap();
    let before: String = (got[..result].iter())
        .filter_map(|e| match e {
            AgentEvent::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        before.starts_with("## [1] a.txt\nFast Sorting sorts 2.5x faster.\n\n## [2] b.md\nSlow"),
        "{before:?}"
    );
    // No model step merged the summaries: its reply is still unsaid.
    let unsaid = fake
        .tokenizer
        .encode("A model step after the readers.", false)
        .len()
        + 1;
    assert_eq!(fake.unsaid(), unsaid);
    assert_eq!(agent.history().last().unwrap().content, want);
}

#[test]
fn a_question_about_the_papers_is_answered_by_a_model_step_after_the_summaries() {
    let fake = Fake::new(32768);
    let mut agent = one_reader_at_a_time(&fake);
    fs::write(
        fake.files().join("a.txt"),
        "Fast Sorting\nWe sort 2.5x faster.\n",
    )
    .unwrap();
    let question = "Which paper is faster?";
    fake.say(&call(
        "read_papers",
        json!({"papers": ["a.txt"], "focus": question}),
    ));
    fake.say("Fast Sorting sorts 2.5x faster.");
    fake.say("Fast Sorting, at 2.5x.");
    let got = events(agent.turn(question).unwrap());
    let want = "## [1] a.txt\nFast Sorting sorts 2.5x faster.\n\nFast Sorting, at 2.5x.";
    assert_eq!(text(&got), want, "{got:?}");
    assert_eq!(fake.unsaid(), 0);
    assert_eq!(
        agent.history().last().unwrap().content,
        "Fast Sorting, at 2.5x."
    );
}

#[test]
fn a_long_lone_document_splits_into_parts_at_paragraph_ends() {
    let para = |c: char| format!("{}\n\n", c.to_string().repeat(1500));
    let text: String = ['a', 'b', 'c', 'd', 'e'].map(para).concat();
    // 500 tokens is about 2,000 bytes: a paragraph a part, and nothing lost.
    let got = parts(&text, 500, MAX_PAPERS);
    assert_eq!(got.len(), 5);
    assert!(got.iter().all(|p| p.len() == 1502), "{got:?}");
    assert_eq!(got.concat(), text);
    // At most `max` parts: the budget grows to share the text out, and the last takes the rest.
    let got = parts(&text, 500, 2);
    assert_eq!(
        got.iter().map(|p| p.len()).collect::<Vec<_>>(),
        [3004, 4506]
    );
    assert_eq!(parts("short\n", 500, MAX_PAPERS), ["short\n"]);
}

/// An essay of four paragraphs of 1,500 bytes, longer than one reader's 500 tokens.
const ESSAY: &str = concat!("# How to Get Ideas\n\n", include_str!("fixtures/essay.md"),);

#[test]
fn a_link_is_read_whole_in_parts_and_the_model_writes_one_summary() {
    let fake = Fake::new(32768);
    fs::create_dir_all(fake.data().join("agent")).unwrap();
    fs::write(
        fake.data().join("agent/config.toml"),
        "max_workers = 1\npaper_tokens = 500\n",
    )
    .unwrap();
    let mut agent = fake.agent();
    let url = serve(ESSAY);
    fake.say(&call("read_papers", json!({ "papers": [url] })));
    for n in 1..=4 {
        fake.say(&format!("Part {n} says to look for problems."));
    }
    fake.say("The essay says to look for problems you have yourself.");
    let got = events(agent.turn(&format!("summarize {url}")).unwrap());
    // The parts go to the model, not the screen: the answer is its one summary.
    assert_eq!(
        text(&got),
        "The essay says to look for problems you have yourself."
    );
    assert_eq!(fake.unsaid(), 0);
    let history = agent.history();
    let [.., result, _] = &history[..] else {
        panic!("{history:?}")
    };
    assert!(result.content.starts_with(PARTS), "{}", result.content);
    for n in 1..=4 {
        assert!(
            (result.content).contains(&format!("essay (part {n} of 4)")),
            "{}",
            result.content
        );
    }
}

/// ESSAY's paragraphs in paulgraham.com's markup: one `<font>` in a layout table, `<br /><br />`
/// between paragraphs, and the title only in `<title>`.
fn pg_essay() -> &'static str {
    // A Windows checkout gives the fixture CRLF line ends.
    let essay = include_str!("fixtures/essay.md").replace("\r\n", "\n");
    let body = essay
        .trim()
        .split("\n\n")
        .collect::<Vec<_>>()
        .join("<br /><br />");
    let page = format!(
        "<html><head><title>How to Do Great Work</title></head><body><table width='100%'><tr>\
         <td><img src='m.gif'></td><td><table width='435'><tr><td><font size='2' face='verdana'>\
         {body}</font></td></tr></table></td></tr></table></body></html>"
    );
    Box::leak(page.into_boxed_str())
}

#[test]
fn a_long_essay_link_is_read_in_parts_with_its_title() {
    let fake = Fake::new(32768);
    fs::create_dir_all(fake.data().join("agent")).unwrap();
    fs::write(
        fake.data().join("agent/config.toml"),
        "max_workers = 1\npaper_tokens = 500\n",
    )
    .unwrap();
    let mut agent = fake.agent();
    let url = serve_as("text/html", pg_essay());
    fake.say(&call("read_papers", json!({ "papers": [url] })));
    for n in 1..=4 {
        fake.say(&format!("Part {n} says to do great work."));
    }
    fake.say("The essay says to work on what you are curious about.");
    let got = events(agent.turn(&format!("summarize {url}")).unwrap());
    assert_eq!(
        text(&got),
        "The essay says to work on what you are curious about."
    );
    assert_eq!(fake.unsaid(), 0);
    // Each part's reader sees the page's title first, which its text alone does not give.
    let fed = fake.fed();
    for n in 1..=4 {
        let first = format!("File: essay (part {n} of 4)\n\n# How to Do Great Work\n\n");
        assert!(fed.contains(&first), "part {n}: {fed}");
    }
}

#[test]
fn a_caller_without_the_web_reads_no_link() {
    let fake = Fake::new(4096);
    let url = serve(ESSAY);
    let err = Papers::default()
        .run_until(
            &fake.larch,
            &ctx(&fake),
            Workers::default(),
            &[&url],
            None,
            &mut |_| {},
            None,
            &Default::default(),
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("web"), "{err}");
}

#[test]
fn a_message_without_the_web_reads_no_link() {
    let fake = Fake::new(32768);
    let mut agent = one_reader_at_a_time(&fake);
    let url = serve(ESSAY);
    fake.say(&call("read_papers", json!({ "papers": [url] })));
    fake.say("You asked me not to browse, so I did not open it.");
    let got = events(
        agent
            .turn(&format!("Don't browse the web. Summarize {url}"))
            .unwrap(),
    );
    let AgentEvent::ToolResult { summary, .. } = &got[1] else {
        panic!("{got:?}")
    };
    assert!(summary.contains("only where the web is on"), "{summary}");
}

#[test]
fn a_www_word_is_a_link_not_a_file() {
    let fake = Fake::new(4096);
    let err = Papers::default()
        .run_until(
            &fake.larch,
            &ctx(&fake),
            Workers::default(),
            &["www.example.com/essay"],
            None,
            &mut |_| {},
            None,
            &Default::default(),
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("only where the web is on"), "{err}");
}

#[test]
fn a_read_papers_call_that_fails_goes_back_to_the_model() {
    let fake = Fake::new(32768);
    let mut agent = one_reader_at_a_time(&fake);
    fake.say(&call("read_papers", json!({"papers": ["missing.pdf"]})));
    fake.say("That file is not there.");
    let got = events(agent.turn("summarize missing.pdf").unwrap());
    let AgentEvent::ToolResult { summary, .. } = &got[1] else {
        panic!("{got:?}")
    };
    assert!(summary.starts_with("failed: "), "{summary}");
    assert_eq!(text(&got), "That file is not there.");
    assert_eq!(fake.unsaid(), 0);
}

#[test]
fn papers_none_of_which_is_read_fail_in_one_short_line_and_the_model_hears_not_to_guess() {
    let fake = Fake::new(32768);
    let mut agent = one_reader_at_a_time(&fake);
    larch_agent::web::fetch::allow_this_machine();
    // Ports nothing listens on: no paper's robots.txt can be read.
    let urls: Vec<String> = (1..=3)
        .map(|n| {
            let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            format!("http://{}/{n}.pdf", free.local_addr().unwrap())
        })
        .collect();
    fake.say(&call("read_papers", json!({ "papers": urls })));
    fake.say("I could not read them.");
    let got = events(agent.turn(&format!("compare {}", urls.join(" "))).unwrap());
    let AgentEvent::ToolResult { summary, .. } = &got[1] else {
        panic!("{got:?}")
    };
    assert_eq!(
        summary,
        "failed: the skill failed: could not read any of the 3 papers: 127.0.0.1's robots.txt \
         could not be read"
    );
    let history = agent.history();
    let [.., result, _] = &history[..] else {
        panic!("{history:?}")
    };
    for url in &urls {
        assert!(result.content.contains(url.as_str()), "{}", result.content);
    }
    assert!(
        result.content.ends_with(read_papers::UNREAD),
        "{}",
        result.content
    );
    assert_eq!(text(&got), "I could not read them.");
}

#[test]
fn the_tool_outside_a_chat_says_it_needs_one() {
    let registry = larch_agent::tool::Registry::builtin();
    let tool = registry.get(read_papers::TOOL).unwrap();
    let fake = Fake::new(4096);
    let err = tool
        .call(&json!({"papers": ["a.txt"]}), &ctx(&fake))
        .unwrap_err();
    assert!(err.to_string().contains("runs only in a chat"), "{err}");
}

#[test]
fn read_file_shows_a_pdfs_text_not_its_bytes() {
    let fake = Fake::new(4096);
    fs::write(fake.files().join("p.pdf"), pdf("GLUE score 80.5")).unwrap();
    let registry = larch_agent::tool::Registry::builtin();
    let read = registry.get("read_file").unwrap();
    let text = read.call(&json!({"path": "p.pdf"}), &ctx(&fake)).unwrap();
    assert_eq!(text.trim(), "GLUE score 80.5");
}

#[test]
fn text_of_reads_a_pdf_by_its_bytes_not_its_name() {
    let fake = Fake::new(4096);
    let file = fake.files().join("paper");
    fs::write(&file, pdf("GLUE score 80.5")).unwrap();
    let stop = std::sync::atomic::AtomicBool::new(false);
    let text = read_papers::text_of(&file, &stop).unwrap();
    assert_eq!(tidy(&text).trim(), "GLUE score 80.5");
}

#[test]
fn write_file_refuses_to_replace_a_pdf() {
    let fake = Fake::new(4096);
    // The fixture is all ASCII, so the UTF-8 check alone lets it through.
    fs::write(fake.files().join("p.bin"), pdf("GLUE score 80.5")).unwrap();
    let registry = larch_agent::tool::Registry::builtin();
    let write = registry.get("write_file").unwrap();
    let args = json!({"path": "p.bin", "text": "GLUE score 80.5"});
    let err = write.prepare(&args, &ctx(&fake)).unwrap_err().to_string();
    assert!(err.contains("is a PDF"), "{err}");
    let err = write.call(&args, &ctx(&fake)).unwrap_err().to_string();
    assert!(err.contains("is a PDF"), "{err}");
    assert!(
        fs::read(fake.files().join("p.bin"))
            .unwrap()
            .starts_with(b"%PDF-")
    );
}

/// A one-page PDF that shows `text` in Helvetica, with its cross-reference table's offsets right.
fn pdf(text: &str) -> Vec<u8> {
    pdf_pages(&[text])
}

/// A PDF with one line of text on each of its pages.
fn pdf_pages(texts: &[&str]) -> Vec<u8> {
    let kids: Vec<String> = (0..texts.len())
        .map(|i| format!("{} 0 R", 4 + 2 * i))
        .collect();
    let mut objects = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
        format!(
            "<< /Type /Pages /Kids [{}] /Count {} >>",
            kids.join(" "),
            texts.len()
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
    ];
    for (i, text) in texts.iter().enumerate() {
        let stream = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET");
        objects.push(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {} 0 R \
             /Resources << /Font << /F1 3 0 R >> >> >>",
            5 + 2 * i
        ));
        objects.push(format!(
            "<< /Length {} >>\nstream\n{stream}\nendstream",
            stream.len()
        ));
    }
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, o) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend(format!("{} 0 obj\n{o}\nendobj\n", i + 1).bytes());
    }
    let xref = out.len();
    out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
    for o in offsets {
        out.extend(format!("{o:010} 00000 n \n").bytes());
    }
    out.extend(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        )
        .bytes(),
    );
    out
}

// Which page-break form is kept: pdftotext's, a form feed after each page (research v2's
// `source` and `Page` ranges expect it, and the built-in reader and PDFKit now give it too).
#[test]
fn the_built_in_reader_puts_a_form_feed_after_each_page_as_pdftotext_does() {
    let text = read_papers::pdf_text(&pdf_pages(&["First page", "Second page"])).unwrap();
    assert!(text.ends_with('\u{c}'), "{text:?}");
    let pages: Vec<&str> = text.strip_suffix('\u{c}').unwrap().split('\u{c}').collect();
    assert_eq!(pages.len(), 2, "{text:?}");
    assert!(
        pages[0].contains("First page") && pages[1].contains("Second page"),
        "{text:?}"
    );
}

#[test]
fn the_built_in_reader_reads_a_pdf_without_poppler() {
    let text = read_papers::pdf_text(&pdf("Sparse experts route tokens")).unwrap();
    assert!(text.contains("Sparse experts route tokens"), "{text:?}");
    let err = read_papers::pdf_text(b"%PDF-1.4\nnot a pdf").unwrap_err();
    assert!(err.starts_with("the PDF could not be read"), "{err}");
}

#[test]
fn a_link_in_the_first_message_of_a_chat_without_folders_offers_read_papers_to_its_grammar() {
    // The demo build: "https://arxiv.org/pdf/2610.39302 can you summarize this paper" in a new
    // chat with nothing attached showed "tool grammar: unsupported schema: no tool named
    // read_papers", as the step's tools were picked before the message was in the history.
    let fake = Fake::new(4096);
    let opts = larch_agent::AgentOptions {
        allowed: vec![],
        ..fake.opts()
    };
    let mut agent = larch_agent::Agent::new(&fake.larch, opts).unwrap();
    let url = "https://papers.invalid/pdf/2610.39302";
    fake.say(&call("read_papers", json!({ "papers": [url] })));
    fake.say("The paper could not be read.");
    let got = events(
        agent
            .turn(&format!("{url} can you summarize this paper"))
            .unwrap(),
    );
    let grammar =
        |e: &AgentEvent| matches!(e, AgentEvent::Notice(n) if n.starts_with("tool grammar"));
    assert!(!got.iter().any(grammar), "{got:?}");
}

#[test]
fn an_arxiv_link_names_its_paper_by_id() {
    let id = |u| arxiv(u);
    assert_eq!(
        id("https://arxiv.org/pdf/2610.39302").as_deref(),
        Some("2610.39302")
    );
    assert_eq!(
        id("https://arxiv.org/abs/2610.39302v1").as_deref(),
        Some("2610.39302v1")
    );
    assert_eq!(
        id("http://www.arxiv.org/pdf/2610.39302v2.pdf").as_deref(),
        Some("2610.39302v2")
    );
    assert_eq!(
        id("https://arxiv.org/html/2610.39302v1/").as_deref(),
        Some("2610.39302v1")
    );
    assert_eq!(
        id("https://arxiv.org/abs/hep-th/9901001?context=x").as_deref(),
        Some("hep-th/9901001")
    );
    assert_eq!(id("https://arxiv.org/list/cs.LG/recent"), None);
    assert_eq!(id("https://example.com/pdf/2610.39302"), None);
    assert_eq!(id("https://arxiv.org/abs/"), None);
}

#[test]
fn a_drive_or_dropbox_share_link_opens_as_its_download() {
    assert_eq!(
        direct("https://drive.google.com/file/d/1AbC-d_9/view?usp=sharing").as_deref(),
        Some("https://drive.google.com/uc?export=download&id=1AbC-d_9")
    );
    assert_eq!(
        direct("https://www.dropbox.com/s/k3y/deck.pdf?dl=0").as_deref(),
        Some("https://www.dropbox.com/s/k3y/deck.pdf?dl=1")
    );
    assert_eq!(
        direct("https://www.dropbox.com/scl/fi/abc/deck.pdf?rlkey=x&dl=0").as_deref(),
        Some("https://www.dropbox.com/scl/fi/abc/deck.pdf?rlkey=x&dl=1")
    );
    assert_eq!(direct("https://drive.google.com/drive/folders/1AbC"), None);
    assert_eq!(direct("https://www.dropbox.com/home"), None);
    assert_eq!(direct("https://example.com/file/d/1AbC/view"), None);
    // The user's link is what is seen: its download opens after untrusted content too.
    let mut seen = larch_agent::web::fetch::Seen::default();
    let shared = "https://drive.google.com/file/d/1AbC/view";
    seen.read(&format!("read {shared}"));
    seen.untrusted = true;
    read_papers::seen_as_read(&[shared], &mut seen);
    assert_eq!(
        seen.refuses("https://drive.google.com/uc?export=download&id=1AbC"),
        None
    );
}

#[test]
fn a_link_of_few_words_is_summarized_and_named_as_maybe_not_the_document() {
    let fake = Fake::new(4096);
    let url = serve("Sign in to read this report.");
    fake.say("The page asks the reader to sign in.");
    let run = Papers::default()
        .run_until(
            &fake.larch,
            &ctx(&fake),
            Workers::default(),
            &[&url],
            None,
            &mut |_| {},
            Some(&mut Default::default()),
            &Default::default(),
        )
        .unwrap();
    assert!(
        run.markdown.ends_with(
            "The page asks the reader to sign in.\n\n[1] The page has only 6 words of text, so \
             it may be a sign-in, paywall or viewer page rather than the document."
        ),
        "{}",
        run.markdown
    );
}

#[test]
fn an_arxiv_abstract_link_the_user_wrote_opens_as_its_pdf_after_untrusted_content() {
    // The abstract link is fetched as the PDF; after a web result, the PDF's URL is no stranger
    // to the provenance check, as the user wrote its paper's link. A link the user never wrote
    // stays refused.
    let pdf = "https://arxiv.org/pdf/2610.39302";
    for written in [
        "https://arxiv.org/abs/2610.39302",
        "www.arxiv.org/abs/2610.39302",
    ] {
        let mut seen = larch_agent::web::fetch::Seen::default();
        seen.read(&format!("look at {written}"));
        seen.untrusted = true;
        assert!(seen.refuses(pdf).is_some());
        read_papers::seen_as_read(&[written], &mut seen);
        assert_eq!(seen.refuses(pdf), None, "{written}");
    }
    let mut seen = larch_agent::web::fetch::Seen::default();
    seen.untrusted = true;
    read_papers::seen_as_read(&["https://arxiv.org/abs/2610.39302"], &mut seen);
    assert!(seen.refuses(pdf).is_some());
}

/// A PDF whose page draws a `v` curve with no current point, as a figure of arXiv 2610.39302
/// does: viewers read its text, the built-in reader panics.
fn v_curve() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/v-curve.pdf")
}

#[test]
fn the_built_in_reader_failing_on_a_curve_with_no_point_is_the_files_error() {
    let err = read_papers::pdf_text(&fs::read(v_curve()).unwrap()).unwrap_err();
    assert!(err.starts_with("the PDF could not be read"), "{err}");
}

#[cfg(target_os = "macos")]
#[test]
fn a_pdf_the_built_in_reader_refuses_is_read_by_pdfkit_on_a_mac() {
    // pdftotext reads it too where it is installed; without it, only PDFKit can.
    let stop = std::sync::atomic::AtomicBool::new(false);
    let text = read_papers::text_of(&v_curve(), &stop).unwrap();
    assert!(text.contains("Attention Is Not All You Need"), "{text}");
}

/// The helper board is the turn's: two reads in one turn keep both readers, and the next turn,
/// which reads nothing, starts with none, as the app shows the board as that turn's helpers.
#[test]
fn a_turns_helpers_are_its_own_reads_and_the_next_turn_starts_with_none() {
    use larch_agent::workers::{Helper, HelperState::Done};
    let fake = Fake::new(32768);
    let mut agent = one_reader_at_a_time(&fake);
    fs::write(
        fake.files().join("a.txt"),
        "Fast Sorting\nWe sort 2.5x faster.\n",
    )
    .unwrap();
    fs::write(
        fake.files().join("b.md"),
        "Slow Trees\nTrees grow 10 cm a year.\n",
    )
    .unwrap();
    let question = "Which paper is faster?";
    fake.say(&call(
        "read_papers",
        json!({"papers": ["a.txt"], "focus": question}),
    ));
    fake.say("Fast Sorting sorts 2.5x faster.");
    fake.say(&call(
        "read_papers",
        json!({"papers": ["b.md"], "focus": question}),
    ));
    fake.say("Slow Trees grow 10 cm a year.");
    fake.say("Fast Sorting.");
    let got = events(agent.turn(question).unwrap());
    assert_eq!(fake.unsaid(), 0, "{got:?}");
    let helper = |id, task: &str| Helper {
        id,
        task: task.into(),
        state: Done,
    };
    let board = agent.helpers();
    let want = [helper(0, "Reading a.txt"), helper(1, "Reading b.md")];
    assert_eq!(*board.lock().unwrap(), want);
    fake.say("You're welcome.");
    events(agent.turn("thanks").unwrap());
    assert_eq!(*board.lock().unwrap(), []);
}

#[test]
fn a_link_is_a_document_a_pdf_with_its_pages_and_a_page_as_one_page_of_text() {
    use larch_agent::folio::{self, Rep};
    use std::sync::atomic::AtomicBool;
    let dir = tempfile::tempdir().unwrap();
    let stop = AtomicBool::new(false);
    // A PDF from a local server: its pages come with it, a form feed between them.
    let bytes = pdf_pages(&["First page text", "Second page text"]);
    let body: &'static str = Box::leak(String::from_utf8(bytes).unwrap().into_boxed_str());
    let url = serve_as("application/pdf", body);
    let doc = folio::store_url(dir.path(), "c", &url, &stop)
        .unwrap()
        .unwrap();
    assert_eq!((doc.rep, doc.pages.len()), (Rep::Pdf, 2));
    assert!(!doc.text.contains('\u{c}'), "{:?}", doc.text);
    assert_eq!(doc.page_of(doc.text.find("Second").unwrap()), 1);
    // The chat's record keeps it under its letter, and its bytes are kept beside it.
    let record = folio::source::load(dir.path(), "c").unwrap();
    assert_eq!((record.len(), record[0].letter.as_str()), (1, "A"));
    assert!(folio::source::blob(dir.path(), &doc.sha256).is_file());
    // An article page: the readable text with its title, on one page.
    let url = serve_as("text/html", pg_essay_html());
    let doc = folio::store_url(dir.path(), "c", &url, &stop)
        .unwrap()
        .unwrap();
    assert_eq!((doc.rep, doc.pages.len()), (Rep::Html, 1));
    assert_eq!(doc.letter(), "B");
    assert!(doc.text.contains("curious"), "{}", doc.text);
}

fn pg_essay_html() -> &'static str {
    // About 1.5 KB of text: enough to be an article, not a shell.
    concat!(
        "<html><head><title>On Curiosity</title></head><body><article><h1>On Curiosity</h1>",
        "<p>Work on what you are curious about, and the work will follow. Curiosity is the engine of ",
        "good work: it is what keeps a person at a hard problem after the reward has gone, and what ",
        "makes the hours spent on it feel short. The people who do the best work in a field are rarely ",
        "the ones who chose it for its prospects; they are the ones who could not leave it alone.</p>",
        "<p>It is hard to fake. You can tell, reading a paper or an essay, whether its author wanted ",
        "to know the answer or only to be seen asking. The first kind of work has loose ends that the ",
        "author cares about, and the second kind has none. So when you choose what to work on, ask ",
        "what you would read about for a week with no one watching, and start there. The rest of what ",
        "you need, the skill and the stamina and the taste, grows from the time you give it.</p>",
        "<p>None of this is a plan, and it should not be one. A plan sets the goal before the ",
        "learning, and curiosity works the other way round: you find out what you want by following ",
        "what you wonder about, one question after another, for years.</p></article></body></html>"
    )
}
