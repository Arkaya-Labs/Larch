//! The follow-up reader (research v2): one question about a chat's documents answered from their
//! text, its send helper, and the chat that offers it.

mod common;

use std::{
    cell::RefCell,
    fs,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use common::{Fake, call, events, text};
use larch::{EngineError, GenParams, Prompt};
use larch_agent::{
    Agent,
    audit::Audit,
    config::Thinking,
    folio::{Doc, Page, Range, Rep, ocr, source, table},
    parse::Family,
    router::Intent,
    skill::SkillError,
    skills::doc::{
        all,
        reader::{Ask, Reader, Readers, Reply, Settings},
        record::PartState,
        send::{self, To},
    },
    skills::read_papers,
    tool::ToolCtx,
};
use serde_json::{Value, json};

fn ctx(fake: &Fake) -> ToolCtx {
    ToolCtx {
        allowed: vec![fake.files()],
        data_dir: fake.data(),
        source: String::new(),
    }
}

/// A PDF's text with `pages`, form feeds between them.
fn doc(pages: &[String]) -> Doc {
    let (mut text, mut out) = (String::new(), Vec::new());
    for (i, p) in pages.iter().enumerate() {
        if i > 0 {
            text.push('\x0c');
        }
        let start = text.len();
        text.push_str(p);
        out.push(Page {
            start,
            end: text.len(),
            label: None,
        });
    }
    Doc {
        index: 0,
        name: "lease.pdf".into(),
        sha256: "0".repeat(64),
        rep: Rep::Pdf,
        text,
        pages: out,
    }
}

fn settings(family: Family, budget: u32, followup_tokens: u32) -> Settings {
    Settings {
        family: Some(family),
        thinking: Thinking {
            budget,
            answer_max_tokens: 0,
        },
        followup_tokens,
        max_workers: 1,
    }
}

fn ask(question: &str) -> Ask<'_> {
    Ask {
        question,
        ..Ask::default()
    }
}

fn reply(fake: &Fake, reader: &mut Reader, s: &Settings, d: &Doc, question: &str) -> Reply {
    shown(fake, reader, s, d, question).0
}

/// `reply`, with each piece of text it showed as it was written.
fn shown(
    fake: &Fake,
    reader: &mut Reader,
    s: &Settings,
    d: &Doc,
    question: &str,
) -> (Reply, Vec<String>) {
    let stop = AtomicBool::default();
    let docs = std::slice::from_ref(d);
    let mut live = Vec::new();
    let r = reader.reply(
        &fake.larch,
        &ctx(fake),
        s,
        docs,
        &ask(question),
        &stop,
        &mut |t| live.push(t.to_owned()),
        &|_| {},
    );
    (r.unwrap(), live)
}

/// The trace lines the reader wrote.
fn trace(fake: &Fake) -> Vec<Value> {
    let dir = fake.data().join("agent/log");
    let files = fs::read_dir(dir).unwrap().map(|f| f.unwrap().path());
    let lines: Vec<Value> = (files.flat_map(|f| {
        let text = fs::read_to_string(f).unwrap();
        text.lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect::<Vec<_>>()
    }))
    .collect();
    lines.into_iter().filter(|l| l["skill"] == "doc").collect()
}

/// The steps of the reader's trace lines, in order.
fn steps(fake: &Fake) -> Vec<String> {
    (trace(fake).iter())
        .map(|l| l["step"].as_str().unwrap().to_owned())
        .collect()
}

/// Twenty pages of filler, `lines` lines each, a fact only on page 13.
fn long(lines: usize) -> Doc {
    let pages: Vec<String> = (1..=20)
        .map(|i| match i {
            13 => "Terms of care.\nThe warranty lasts seven years from delivery.\n".to_owned(),
            i => format!("Chapter {i} lists rooms, keys and parking for tenant number {i}.\n")
                .repeat(lines),
        })
        .collect();
    doc(&pages)
}

/// Page `i`'s range (from 1) of `d`.
fn page(d: &Doc, i: usize) -> Range {
    Range {
        doc: 0,
        start: d.pages[i - 1].start,
        end: d.pages[i - 1].end,
    }
}

#[test]
fn the_send_helper_ends_on_the_turns_own_stop_and_writes_its_trace_line() {
    let fake = Fake::new(4096);
    fake.say("too late");
    let (stop, pressed) = (AtomicBool::new(false), std::sync::Mutex::new(None));
    let p = GenParams {
        max_tokens: 8,
        ..GenParams::default()
    };
    fake.stall(Duration::from_secs(8));
    let got = std::thread::scope(|s| {
        s.spawn(|| {
            fake.wait_stalled();
            // The reader waits on the engine when Stop comes; the pass holds for seconds.
            std::thread::sleep(Duration::from_millis(300));
            *pressed.lock().unwrap() = Some(Instant::now());
            stop.store(true, Ordering::Relaxed);
        });
        let audit = Audit::new(&fake.data());
        let to = To::Once(Prompt::Text("hi".into()));
        let detail = json!({"docs": ["lease.pdf"]});
        send::send(
            &fake.larch,
            to,
            p,
            "read",
            detail,
            &audit,
            &stop,
            &mut |_| true,
        )
    });
    let took = pressed.lock().unwrap().unwrap().elapsed();
    assert!(took < Duration::from_secs(4), "{took:?}");
    assert!(
        matches!(got, Err(SkillError::Engine(EngineError::Cancelled))),
        "{got:?}"
    );
    let lines = trace(&fake);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_eq!(lines[0]["step"], "read");
    assert_eq!(lines[0]["detail"]["docs"], json!(["lease.pdf"]));
}

#[test]
fn a_whole_document_is_answered_in_one_step_and_a_kept_reader_sends_only_the_question() {
    let fake = Fake::chatml(8192);
    let d = doc(&[
        "The rent is $900 a month.\n".into(),
        "The deposit is $1,800.\n".into(),
    ]);
    let s = settings(Family::Llama3, 0, 6000);
    let mut reader = Reader::default();
    let answer = "\"The deposit is $1,800.\" [p. 2]";
    fake.say(answer);
    let r = reply(&fake, &mut reader, &s, &d, "What is the deposit?");
    assert_eq!(
        (r.text.as_str(), r.retried, r.no_room),
        (answer, false, false)
    );
    let whole = Range {
        doc: 0,
        start: 0,
        end: d.text.len(),
    };
    assert_eq!(r.ranges, [whole]);
    let fed = fake.fed();
    assert!(
        fed.contains(
            "Document: lease.pdf (2 pages)\n[p. 1]\nThe rent is $900 a month.\n[p. 2]\nThe deposit"
        ),
        "{fed}"
    );
    // The whole text needs no tools.
    assert!(!fed.contains("Exact search"), "{fed}");
    // The next question goes to the same session alone: its KV holds the document.
    let start = fake.fed_tokens();
    fake.say("\"The rent is $900 a month.\" [p. 1]");
    let r = reply(&fake, &mut reader, &s, &d, "And the rent?");
    assert_eq!(r.text, "\"The rent is $900 a month.\" [p. 1]");
    let fed = fake.fed_from(start);
    assert!(
        fed.contains("<|im_start|>user\nQuestion: And the rent?<|im_end|>"),
        "{fed}"
    );
    // ponytail: the fake engine prefills the kept session whole again (cached_tokens 0), so the
    // document shows once, from the history; assert its KV is reused once that is found
    assert_eq!(fed.matches("Document: ").count(), 1, "{fed}");
}

#[test]
fn an_empty_answer_is_asked_for_once_more_without_thinking() {
    let fake = Fake::qwen3(8192);
    let d = doc(&["The rent is $900 a month.\n".into()]);
    let s = settings(Family::Qwen3, 100, 6000);
    fake.say("");
    fake.say("\"The rent is $900 a month.\" [p. 1]");
    let r = reply(&fake, &mut Reader::default(), &s, &d, "What is the rent?");
    assert!(r.retried);
    assert_eq!(r.text, "\"The rent is $900 a month.\" [p. 1]");
    let fed = fake.fed();
    let (_, after) = fed.rsplit_once("Your answer was empty").unwrap();
    assert!(after.contains("<think>\n\n</think>"), "{after}");
    assert_eq!(steps(&fake), ["read", "retry"]);
}

#[test]
fn a_stopped_reader_asks_for_no_answer_again() {
    let fake = Fake::new(8192);
    let d = doc(&["The rent is $900 a month.\n".into()]);
    let s = settings(Family::Llama3, 0, 6000);
    fake.say("");
    let stop = AtomicBool::new(true);
    let docs = std::slice::from_ref(&d);
    let got = Reader::default().reply(
        &fake.larch,
        &ctx(&fake),
        &s,
        docs,
        &ask("Rent?"),
        &stop,
        &mut |_| {},
        &|_| {},
    );
    assert!(
        matches!(got, Err(SkillError::Engine(EngineError::Cancelled))),
        "{got:?}"
    );
    assert_eq!(steps(&fake), ["read"]);
}

#[test]
fn a_fact_far_into_a_long_document_is_in_the_first_read_by_its_words() {
    let fake = Fake::new(8192);
    let d = long(4);
    // Past the whole-document rule: a first read of about 300 tokens, and the tools.
    let s = settings(Family::Llama3, 0, 300);
    let answer = "\"The warranty lasts seven years from delivery.\" [p. 13]";
    fake.say(answer);
    let r = reply(
        &fake,
        &mut Reader::default(),
        &s,
        &d,
        "How long does the warranty last?",
    );
    assert_eq!(r.text, answer);
    assert!(r.ranges.contains(&page(&d, 13)), "{:?}", r.ranges);
    let fed = fake.fed();
    assert!(
        fed.contains("[p. 13]\nTerms of care.\nThe warranty lasts seven years"),
        "{fed}"
    );
    assert!(fed.contains("[...]"), "{fed}");
    assert!(
        fed.contains("of its 20 pages; find and read reach the rest."),
        "{fed}"
    );
    assert!(fed.contains("Exact search"), "{fed}");
}

#[test]
fn a_fact_the_first_read_missed_is_found_with_find() {
    let fake = Fake::new(8192);
    let d = long(4);
    let s = settings(Family::Llama3, 0, 100);
    // The question's words are not the page's: the first read takes other pages.
    fake.say(&call("find", json!({"words": "warranty"})));
    let answer = "\"The warranty lasts seven years from delivery.\" [p. 13]";
    fake.say(answer);
    let r = reply(
        &fake,
        &mut Reader::default(),
        &s,
        &d,
        "How long is the guarantee?",
    );
    assert_eq!(r.text, answer);
    let fed = fake.fed();
    let (_, after) = fed.split_once("\"name\": \"find\"").unwrap();
    assert!(after.contains("[p. 13]"), "{after}");
    assert!(after.contains("The warranty lasts seven years"), "{after}");
    let on_13 = |r: &Range| r.start >= page(&d, 13).start && r.end <= page(&d, 13).end;
    assert!(r.ranges.iter().any(on_13), "{:?}", r.ranges);
    assert_eq!(steps(&fake), ["read", "after tools"]);
}

#[test]
fn a_reader_out_of_room_answers_without_tools_and_says_so() {
    // A small context, of which the tiny vocabulary spends most on the tools' schemas (read_all
    // and read's pictures and tables too): the first read takes all the room an answer leaves,
    // as every page names a tenant.
    let fake = Fake::new(4608);
    let d = long(4);
    let s = settings(Family::Llama3, 0, 100_000);
    fake.say(&call("read", json!({"part": "pages 1-20"})));
    fake.say("\"The warranty lasts seven years from delivery.\" [p. 13]");
    let question = "How long does the warranty last for each tenant?";
    let r = reply(&fake, &mut Reader::default(), &s, &d, question);
    assert!(r.no_room, "{r:?}");
    assert_eq!(
        r.text,
        "\"The warranty lasts seven years from delivery.\" [p. 13]"
    );
    let fed = fake.fed();
    assert!(fed.contains("No room is left in this context"), "{fed}");
    // The room was counted right: no request was refused as too long.
    assert_eq!(steps(&fake), ["read", "after tools"]);
}

/// Writes `text` as the data folder's config.toml.
fn config(fake: &Fake, text: &str) {
    fs::create_dir_all(fake.data().join("agent")).unwrap();
    fs::write(fake.data().join("agent/config.toml"), text).unwrap();
}

#[test]
fn a_chat_that_read_a_document_keeps_read_papers_for_a_message_the_router_calls_code() {
    // No prefix cache, so each turn's whole prompt is fed and shows which tools it lists.
    let fake = Fake::uncached(32768);
    config(&fake, "max_workers = 1\n");
    let mut agent = fake.agent();
    fs::write(
        fake.files().join("lease.txt"),
        "Lease\nThe deposit is $1,800.\n",
    )
    .unwrap();
    fake.say(&call("read_papers", json!({"papers": ["lease.txt"]})));
    fake.say("A lease with a $1,800 deposit.");
    let got = events(agent.turn("summarize lease.txt").unwrap());
    assert!(
        text(&got).contains("A lease with a $1,800 deposit."),
        "{got:?}"
    );
    // Off: a code message shows no tool.
    let follow = "Answer a question about documents this chat already read";
    let start = fake.fed_tokens();
    fake.say("DEPOSIT = 1800");
    events(
        agent
            .turn_for("write the deposit as a python constant", Intent::Code)
            .unwrap(),
    );
    assert!(!fake.fed_from(start).contains("Read several papers"));
    let id = agent.chat_id().to_owned();
    drop(agent);
    // On: the same message in the reopened chat keeps read_papers, which says it answers
    // follow-ups.
    config(&fake, "max_workers = 1\ndoc_followups = \"on\"\n");
    let mut agent = Agent::open(&fake.larch, &id, fake.opts()).unwrap();
    let start = fake.fed_tokens();
    fake.say("DEPOSIT = 1800");
    events(
        agent
            .turn_for("write the deposit as a python constant", Intent::Code)
            .unwrap(),
    );
    let fed = fake.fed_from(start);
    assert!(fed.to_lowercase().contains(&follow.to_lowercase()), "{fed}");
}

#[test]
fn a_summary_tells_the_chat_model_it_is_one_only_with_followups_on() {
    for on in [false, true] {
        let fake = Fake::new(32768);
        let followups = if on { "doc_followups = \"on\"\n" } else { "" };
        config(&fake, &format!("max_workers = 1\n{followups}"));
        let mut agent = fake.agent();
        fs::write(
            fake.files().join("lease.txt"),
            "Lease\nThe deposit is $1,800.\n",
        )
        .unwrap();
        fake.say(&call("read_papers", json!({"papers": ["lease.txt"]})));
        fake.say("A lease with a $1,800 deposit.");
        let got = events(agent.turn("summarize lease.txt").unwrap());
        // The user sees the summary alone; the note is in the call's result, for the model.
        assert!(!text(&got).contains("[agent]"), "{got:?}");
        let history = agent.history();
        let result = &history[history.len() - 2].content;
        assert_eq!(result.contains(read_papers::SUMMARY_NOTE), on, "{result}");
        // The chat's documents by letter, with their pages, only with follow-ups on.
        assert_eq!(
            result.contains("Documents in this chat: A. lease.txt, 1 page."),
            on,
            "{result}"
        );
    }
}

#[test]
fn a_followup_is_the_readers_answer_with_no_chat_model_step() {
    let fake = Fake::new(32768);
    config(&fake, "max_workers = 1\ndoc_followups = \"on\"\n");
    let mut agent = fake.agent();
    fs::write(
        fake.files().join("lease.txt"),
        "Lease\nThe deposit is $1,800.\n",
    )
    .unwrap();
    let question = "What is the deposit?";
    fake.say(&call(
        "read_papers",
        json!({"papers": ["lease.txt"], "focus": question}),
    ));
    let answer = "\"The deposit is $1,800.\" [p. 1]";
    fake.say(answer);
    fake.say("A chat-model step after the reader.");
    let got = events(agent.turn(question).unwrap());
    assert_eq!(text(&got), answer, "{got:?}");
    assert_eq!(agent.history().last().unwrap().content, answer);
    // No model step after the reader: its reply is still unsaid.
    let unsaid = (fake.tokenizer)
        .encode("A chat-model step after the reader.", false)
        .len()
        + 1;
    assert_eq!(fake.unsaid(), unsaid);
}

#[test]
fn a_readers_not_found_is_shown_as_it_is_and_never_falls_back() {
    let fake = Fake::new(32768);
    config(&fake, "max_workers = 1\ndoc_followups = \"on\"\n");
    let mut agent = fake.agent();
    fs::write(
        fake.files().join("lease.txt"),
        "Lease\nThe deposit is $1,800.\n",
    )
    .unwrap();
    let question = "Are pets allowed?";
    fake.say(&call(
        "read_papers",
        json!({"papers": ["lease.txt"], "focus": question}),
    ));
    let answer = "Not found in the text. It covers the deposit.";
    fake.say(answer);
    fake.say("Pets are usually allowed.");
    let got = events(agent.turn(question).unwrap());
    assert_eq!(text(&got), answer, "{got:?}");
    assert_eq!(agent.source(), larch_agent::AnswerSource::Refused);
    assert!(fake.unsaid() > 0);
}

#[test]
fn a_followup_on_two_summarized_documents_reads_both_with_their_summaries() {
    let fake = Fake::uncached(32768);
    config(&fake, "max_workers = 1\ndoc_followups = \"on\"\n");
    let mut agent = fake.agent();
    fs::write(
        fake.files().join("a.txt"),
        "Lease\nThe deposit is $1,800.\n",
    )
    .unwrap();
    fs::write(fake.files().join("b.txt"), "Pets\nCats are allowed.\n").unwrap();
    let papers = json!(["a.txt", "b.txt"]);
    fake.say(&call("read_papers", json!({"papers": papers})));
    fake.say("A lease.");
    fake.say("A pet policy.");
    events(agent.turn("summarize a.txt and b.txt").unwrap());
    // Each document read for the summaries is in the chat's source record, in letter order.
    let record = fs::read_to_string(
        fake.data()
            .join(format!("agent/docs/{}.json", agent.chat_id())),
    );
    let record: Value = serde_json::from_str(&record.unwrap()).unwrap();
    assert_eq!(record[0]["letter"], "A");
    assert_eq!(record[1]["letter"], "B");
    let question = "What is the deposit?";
    fake.say(&call(
        "read_papers",
        json!({"papers": papers, "focus": question}),
    ));
    let answer = "\"The deposit is $1,800.\" [A, p. 1]";
    fake.say(answer);
    let start = fake.fed_tokens();
    let got = events(agent.turn(question).unwrap());
    assert_eq!(text(&got), answer, "{got:?}");
    let said = larch_agent::AgentEvent::Progress("Reading 2 documents".into());
    assert!(got.contains(&said), "{got:?}");
    let fed = fake.fed_from(start);
    assert!(fed.contains("Document A: a.txt (1 page)"), "{fed}");
    assert!(
        fed.contains("Summary from the first read:\nA lease.\n\nA pet policy."),
        "{fed}"
    );
}

#[test]
fn prose_a_tool_step_showed_before_its_call_starts_the_answer_and_shows_once() {
    let fake = Fake::new(8192);
    let d = long(4);
    let s = settings(Family::Llama3, 0, 100);
    let preamble = "I will search the whole text for it.";
    fake.say(&format!(
        "{preamble}\n{}",
        call("find", json!({"words": "warranty"}))
    ));
    let answer = "\"The warranty lasts seven years from delivery.\" [p. 13]";
    fake.say(answer);
    let (r, live) = shown(
        &fake,
        &mut Reader::default(),
        &s,
        &d,
        "How long is the guarantee?",
    );
    // The prose showed while its step ran, the call never did, and the answer came after it.
    assert_eq!(r.text, format!("{preamble}\n\n{answer}"));
    assert_eq!(live.concat(), r.text, "{live:?}");
    assert!(!live.concat().contains("\"name\""), "{live:?}");
    assert_eq!(steps(&fake), ["read", "after tools"]);
}

#[test]
fn an_answer_while_tools_are_offered_shows_before_its_step_ends() {
    let fake = Fake::new(8192);
    let d = long(4);
    // Past the whole-document rule, so the step offers find and read.
    let s = settings(Family::Llama3, 0, 300);
    let answer = "Seven years.\n\"The warranty lasts seven years from delivery.\" [p. 13]";
    fake.say(answer);
    fake.pace(Duration::from_millis(20));
    let (r, live) = shown(
        &fake,
        &mut Reader::default(),
        &s,
        &d,
        "How long does the warranty last?",
    );
    assert_eq!(r.text, answer);
    assert_eq!(live.concat(), answer, "{live:?}");
    assert!(fake.fed().contains("Exact search"));
    // The first line showed once it ended, while the rest was still being written.
    let line = &trace(&fake)[0];
    let first = line["detail"]["first_text_ms"].as_f64().unwrap();
    let wall = line["wall_ms"].as_f64().unwrap();
    assert!(first + 200.0 < wall, "{line}");
}

/// Text of at least `tokens` tokens on `fake`'s vocabulary: `head`, then filler lines.
fn filled(fake: &Fake, head: &str, tokens: usize) -> String {
    let mut text = head.to_owned();
    for i in 1.. {
        if fake.larch.token_ids(&text).len() >= tokens {
            break;
        }
        text += &format!("Line {i} lists rooms, keys and parking.\n");
    }
    text
}

/// A manual of `n` pages of about 200 tokens, each with its limits first; page `long` (from 1)
/// is about 450 and states its limit at its start and its end.
fn manual(fake: &Fake, n: usize, long: Option<usize>) -> Doc {
    let pages: Vec<String> = (1..=n)
        .map(|i| {
            let head = match i {
                1 => "Wind speed must stay under 20 m/s.\n",
                3 => "The floor load limit is 500 kg.\nWind speed must stay under 20 m/s.\n",
                4 => "Heat must stay under 60 C.\nDust must stay under 5 mg.\n",
                10 => "Noise must stay under 70 dB.\n",
                _ => "",
            };
            let head = format!("Section {i}.\n{head}");
            match Some(i) == long {
                true => filled(fake, &head, 430) + "Water must stay under 3 m.\n",
                false => filled(fake, &head, 200),
            }
        })
        .collect();
    doc(&pages)
}

/// Settings whose answers stop at `max_tokens`, with one part reader at a time, so the script's
/// replies go to the parts in order.
fn capped(followup_tokens: u32, max_tokens: u32) -> Settings {
    Settings {
        thinking: Thinking {
            budget: 0,
            answer_max_tokens: max_tokens,
        },
        ..settings(Family::Llama3, 0, followup_tokens)
    }
}

/// `reader`'s reply to `question` about `d`, with the turn's `stop` and its progress lines.
fn reply_all(
    fake: &Fake,
    reader: &mut Reader,
    s: &Settings,
    d: &Doc,
    question: &str,
    stop: &AtomicBool,
    progress: &dyn Fn(&str),
) -> Result<Reply, SkillError> {
    let docs = std::slice::from_ref(d);
    let ctx = ctx(fake);
    reader.reply(
        &fake.larch,
        &ctx,
        s,
        docs,
        &ask(question),
        stop,
        &mut |_| {},
        progress,
    )
}

/// `text` as the template writes a tool result: a JSON string, without its quote marks.
fn escaped(text: &str) -> String {
    let json = serde_json::to_string(text).unwrap();
    json[1..json.len() - 1].to_owned()
}

/// The trace lines of the part readers, in order.
fn parts(fake: &Fake) -> Vec<Value> {
    (trace(fake).into_iter())
        .filter(|l| l["step"] == "part")
        .collect()
}

#[test]
fn read_all_reads_every_part_once_and_keeps_each_item_with_its_page_and_the_coverage() {
    let fake = Fake::new(8192);
    let d = manual(&fake, 11, Some(6));
    let call = call("read_all", json!({"task": "every limit"}));
    let max = 160;
    let s = capped(300, max as u32);
    fake.say(&call);
    // Parts 1-5 are pages 1-5, parts 6 and 7 are page 6 in two pieces, parts 8-12 pages 7-11.
    // The second item is cut inside its quote.
    let cut = "- Heat: \"Heat must stay under 60 C.\" [p. 4]\n- Dust: \"Dust must stay under 5 mg";
    let mut cut: Vec<u32> = fake
        .tokenizer
        .encode(&(cut.to_owned() + &" and".repeat(200)), false);
    cut.truncate(max);
    let water = "- Water: \"Water must stay under 3 m.\" [p. 6]";
    for (i, answer) in (1..=12).zip([
        "- Wind: \"Wind speed must stay under 20 m/s.\" [p. 1]",
        "None.",
        "- Load: \"The floor load limit is 500 kg.\" [p. 3]\n\
         - Wind: \"Wind speed must stay under 20 m/s.\" [p. 3]",
        "",
        "None.",
        water,
        // The same item on the same page, from the page's other piece.
        "* water:  \"Water must stay under 3 m.\" [p. 6]",
        "None.",
        "",
        "None.",
        "1. Noise: \u{201c}Noise must stay under 70 dB.\u{201d} [p. 10]",
        "None.",
    ]) {
        match i {
            // Its answer runs to max_tokens: no end of turn is scripted.
            4 => fake.say_tokens(&cut),
            _ => fake.say(answer),
        }
    }
    let answer = "The limits: \"Wind speed must stay under 20 m/s.\" [p. 1]";
    fake.say(answer);
    assert!(fake.tokenizer.encode(&call, false).len() < max);
    let said = RefCell::new(Vec::new());
    let progress = |t: &str| said.borrow_mut().push(t.to_owned());
    let mut reader = Reader::default();
    let stop = AtomicBool::default();
    let question = "List every limit in the manual.";
    let r = reply_all(&fake, &mut reader, &s, &d, question, &stop, &progress).unwrap();
    assert_eq!(r.text, answer);
    let p = &reader.progress;
    let states: Vec<&PartState> = p.parts.iter().map(|(_, s)| s).collect();
    let empty = PartState::Failed("its reader gave an empty answer".into());
    let mut want = vec![&PartState::Done; 12];
    (want[3], want[8]) = (&PartState::Cut, &empty);
    assert_eq!(states, want);
    let page = |i: usize| d.page_of(p.parts[i].0.start) + 1;
    assert_eq!((page(5), page(6), page(7)), (6, 6, 7));
    let items: Vec<(&str, usize)> = (p.items.iter())
        .map(|i| (i.text.as_str(), d.page_of(i.at.start) + 1))
        .collect();
    assert_eq!(
        items,
        [
            ("Wind", 1),
            ("Load", 3),
            ("Wind", 3),
            ("Heat", 4),
            ("Water", 6),
            ("Noise", 10)
        ]
    );
    let fed = fake.fed();
    // The failed part's page did not reach a reader that finished.
    let result = "6 items found.\nPages that reached a reader: 1-7 and 9-11 of 11; not reached: 8.\n\
                  Parts whose reader finished without being cut: 10 of 12; cut: 4; failed: 9 \
                  (its reader gave an empty answer).\n\
                  1. Wind: \"Wind speed must stay under 20 m/s.\" [p. 1]\n\
                  2. Load: \"The floor load limit is 500 kg.\" [p. 3]\n";
    assert!(fed.contains(&escaped(result)), "{fed}");
    let last = "6. Noise: \"Noise must stay under 70 dB.\" [p. 10]\nRoom left";
    assert!(fed.contains(&escaped(last)), "{fed}");
    // Each part reader got its part whole, with the task.
    assert!(
        fed.contains("Part 3 of 12:\n[p. 3]\nSection 3.\nThe floor load limit is 500 kg.\n"),
        "{fed}"
    );
    assert!(fed.contains("Task: every limit"), "{fed}");
    // items is offered only once read_all ran.
    let (before, after) = fed.split_once(all::PART).unwrap();
    assert!(before.contains("\"read_all\"") && !before.contains("\"items\""));
    assert!(after.contains("\"items\""), "{after}");
    let lines: Vec<String> = (1..=12)
        .map(|i| format!("Reading part {i} of 12"))
        .collect();
    assert_eq!(*said.borrow(), lines);
    let steps = steps(&fake);
    assert_eq!(steps.len(), 14, "{steps:?}");
    assert_eq!(
        (steps[0].as_str(), steps[13].as_str()),
        ("read", "after tools")
    );
    assert_eq!(parts(&fake).len(), 12);
}

#[test]
fn a_stop_names_the_pages_not_read_and_after_a_reset_only_those_are_read() {
    let fake = Fake::new(8192);
    let d = manual(&fake, 12, None);
    let max = 160;
    let s = capped(300, max as u32);
    let call = call("read_all", json!({"task": "every limit"}));
    let answers = |pages: std::ops::RangeInclusive<usize>| {
        for i in pages {
            fake.say(&match i {
                1 => "- Wind: \"Wind speed must stay under 20 m/s.\" [p. 1]".into(),
                10 => "- Noise: \"Noise must stay under 70 dB.\" [p. 10]".into(),
                _ => "None.".to_owned(),
            });
        }
    };
    fake.say(&call);
    answers(1..=4);
    // The user stops the turn as part 5 starts.
    let stop = AtomicBool::default();
    let progress = |t: &str| {
        if t == "Reading part 5 of 12" {
            stop.store(true, Ordering::Relaxed);
        }
    };
    let mut reader = Reader::default();
    let q1 = "List every limit in the manual.";
    let got = reply_all(&fake, &mut reader, &s, &d, q1, &stop, &progress);
    assert!(
        matches!(got, Err(SkillError::Engine(EngineError::Cancelled))),
        "{got:?}"
    );
    let docs = std::slice::from_ref(&d);
    assert_eq!(
        reader.progress.coverage(docs),
        "Pages that reached a reader: 1-4 of 12; not reached: 5-12.\n\
         Parts whose reader finished without being cut: 4 of 12; not read: 5-12."
    );
    assert_eq!(fake.unsaid(), 0);
    // The Stop reset the session, whose call has no result; the progress stays.
    fake.say(&call);
    answers(5..=12);
    let answer = "The limits: \"Noise must stay under 70 dB.\" [p. 10]";
    fake.say(answer);
    let (start, stop) = (steps(&fake).len(), AtomicBool::default());
    let said = RefCell::new(Vec::new());
    let progress = |t: &str| said.borrow_mut().push(t.to_owned());
    let r = reply_all(
        &fake,
        &mut reader,
        &s,
        &d,
        "Go on with the list.",
        &stop,
        &progress,
    );
    assert_eq!(r.unwrap().text, answer);
    let steps = steps(&fake);
    assert_eq!(steps[start], "read", "{steps:?}");
    assert!(!steps.contains(&"reset".to_owned()), "{steps:?}");
    let read: Vec<u64> = (parts(&fake).iter())
        .filter(|l| l["error"].is_null())
        .map(|l| l["detail"]["part"].as_u64().unwrap())
        .collect();
    assert_eq!(read, (1..=12).collect::<Vec<_>>());
    let lines: Vec<String> = (5..=12)
        .map(|i| format!("Reading part {i} of 12"))
        .collect();
    assert_eq!(*said.borrow(), lines);
    assert_eq!(
        reader.progress.coverage(docs),
        "Pages that reached a reader: 1-12 of 12.\n\
         Parts whose reader finished without being cut: 12 of 12."
    );
    let items: Vec<&str> = reader
        .progress
        .items
        .iter()
        .map(|i| i.text.as_str())
        .collect();
    assert_eq!(items, ["Wind", "Noise"]);
}

#[test]
fn a_comparison_reads_document_b_again_before_it_answers_and_its_quotes_are_checked() {
    let fake = Fake::new(8192);
    let named = |index: usize, name: &str, pages: &[&str]| Doc {
        index,
        name: name.into(),
        sha256: index.to_string().repeat(64),
        ..doc(&pages
            .iter()
            .map(|p| filled(&fake, p, 80))
            .collect::<Vec<_>>())
    };
    let a = named(
        0,
        "a.pdf",
        &[
            "Model A.\n",
            "Setup.\n\nWe tested on ImageNet at 224 px.\nThe error rate is 4% on the test set.\n\n",
        ],
    );
    let b = named(
        1,
        "b.pdf",
        &[
            "Model B.\n",
            "The error rate is 3% on the test set.\n\n",
            "Setup.\n\nWe tested on CIFAR-10 at 32 px.\n\n",
        ],
    );
    // Each document is within the whole-document rule alone, not both together.
    let s = settings(Family::Llama3, 0, 300);
    let (a_says, b_says) = (
        "\"The error rate is 4% on the test set.\" [p. 2]",
        "\"The error rate is 3% on the test set.\" [p. 2]",
    );
    fake.say(a_says);
    fake.say(b_says);
    let read = call("read", json!({"part": "p. 3", "doc": "B"}));
    fake.say(&read);
    let compared = "Not comparable: \"We tested on CIFAR-10 at 32 px.\" [B, p. 3], but \"We \
                    tested on ImageNet at 224 px.\" [A, p. 2].";
    fake.say(compared);
    let said = RefCell::new(Vec::new());
    let progress = |t: &str| said.borrow_mut().push(t.to_owned());
    let mut readers = Readers::new(s);
    let question = "Which model has the lower error rate?";
    let docs = [a, b];
    let got = (readers.answer(
        &fake.larch,
        &ctx(&fake),
        &docs,
        ask(question),
        &AtomicBool::default(),
        &mut |_| {},
        &progress,
    ))
    .unwrap();
    assert_eq!(
        got,
        format!("## A. a.pdf\n{a_says}\n\n## B. b.pdf\n{b_says}\n\n## Comparison\n{compared}")
    );
    assert_eq!(
        *said.borrow(),
        ["Reading 2 documents", "Comparing the answers"]
    );
    let fed = fake.fed();
    let (_, step) = fed.rsplit_once(b_says).unwrap();
    // The step has the question, each answer with its document's letter, the paragraph around
    // each quote, and read with the document's letter.
    assert!(
        step.contains(
            "[A, p. 2]\nWe tested on ImageNet at 224 px.\nThe error rate is 4% on the test set.\n"
        ),
        "{step}"
    );
    assert!(
        step.contains("[B, p. 2]\nThe error rate is 3% on the test set.\n"),
        "{step}"
    );
    assert!(!step.contains("CIFAR-10 at 32 px.\n[...]"), "{step}");
    assert!(step.contains(&format!(
        "Question: {question}\n\nEach document's reader answered"
    )));
    assert!(step.contains("A. a.pdf:\n\"The error rate is 4% on the test set.\" [A, p. 2]"));
    assert!(step.contains("The document's letter"), "{step}");
    assert!(!step.contains("\"read_all\""), "{step}");
    // It read page 3 of B before it answered.
    let (_, after) = step.split_once(&read).unwrap();
    assert!(
        after.contains(&escaped("[B, p. 3]\nSetup.\n\nWe tested on CIFAR-10")),
        "{after}"
    );
    // Its quotes were checked like any reader's.
    let check = trace(&fake)
        .into_iter()
        .rfind(|l| l["step"] == "check")
        .unwrap();
    assert_eq!(
        (
            &check["detail"]["ran"],
            &check["detail"]["cited"],
            &check["detail"]["matched"]
        ),
        (&json!(true), &json!(2), &json!(2))
    );
    assert_eq!(
        steps(&fake),
        [
            "read",
            "check",
            "read",
            "check",
            "read",
            "after tools",
            "check"
        ]
    );
}

/// A scan (page 1), a text page with a table (page 2), and a caption over a picture of a chart's
/// labels (page 3), copied into the chat's folder.
fn scanned(fake: &Fake) -> std::path::PathBuf {
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/scanned.pdf");
    let file = fake.files().join("scanned.pdf");
    fs::copy(fixture, &file).unwrap();
    file
}

#[test]
fn read_adds_a_pages_pictures_and_tables_or_says_this_machine_cannot() {
    let fake = Fake::new(8192);
    let file = scanned(&fake);
    let data = fake.data();
    let read = source::text(&data, &file, &AtomicBool::default(), Some(&|_| {})).unwrap();
    let entry = source::Entry::file("scanned.pdf", &file, &read);
    let index = source::add(&data, "chat-1", entry).unwrap();
    let d = source::doc(index, "scanned.pdf", &read.sha256, read.rep, read.text);
    // Past the whole-document rule, so the reader has its tools.
    let s = settings(Family::Llama3, 0, 50);
    let args = json!({"part": "page 2", "pictures": true, "tables": true});
    fake.say(&call("read", args));
    fake.say("\"8 hours\" [p. 2]");
    let ask = Ask {
        question: "How fast is a severe fault fixed?",
        chat: "chat-1",
        ..Ask::default()
    };
    let docs = std::slice::from_ref(&d);
    let stop = AtomicBool::default();
    let r = Reader::default().reply(
        &fake.larch,
        &ctx(&fake),
        &s,
        docs,
        &ask,
        &stop,
        &mut |_| {},
        &|_| {},
    );
    assert_eq!(r.unwrap().text, "\"8 hours\" [p. 2]");
    let fed = fake.fed();
    // Page 2's text layer is good, so its pictures were read for this call alone.
    match ocr::available() {
        true => {
            assert!(fed.contains("read its page with pictures=true"), "{fed}");
            let pictures = format!("{}\\nSection 4 covers", ocr::mark(1));
            assert!(fed.contains(&pictures), "{fed}");
        }
        false => {
            assert!(
                fed.contains("This machine cannot read text inside pictures."),
                "{fed}"
            );
            assert!(
                fed.contains("OCR is not available on this machine"),
                "{fed}"
            );
        }
    }
    match table::available() {
        true => {
            let rows = r"| Level | Response | Fix |\n|---|---|---|\n| Severe | 1 hour | 8 hours |";
            assert!(fed.contains(rows), "{fed}");
        }
        false => assert!(
            fed.contains("Tables cannot be read: word boxes need"),
            "{fed}"
        ),
    }
    assert!(fed.contains(r"| 10 days |\nRoom left: about"), "{fed}");
}

#[test]
fn a_scans_summary_reads_its_pictures_on_the_progress_line_and_a_followup_quotes_them() {
    if !ocr::available() {
        println!("skipped: OCR is not available on this machine");
        return;
    }
    let fake = Fake::new(32768);
    config(&fake, "max_workers = 1\ndoc_followups = \"on\"\n");
    let mut agent = fake.agent();
    scanned(&fake);
    fake.say(&call("read_papers", json!({"papers": ["scanned.pdf"]})));
    fake.say("A services agreement.");
    let got = events(agent.turn("summarize scanned.pdf").unwrap());
    let said = larch_agent::AgentEvent::Progress("Reading scanned pages: 2 of 2".into());
    assert!(got.contains(&said), "{got:?}");
    let fed = fake.fed();
    assert!(fed.contains("The warranty lasts seven years"), "{fed}");
    // The follow-up reopens the scan's text from the caches, and its check matches the quote.
    let question = "How long is the warranty?";
    fake.say(&call(
        "read_papers",
        json!({"papers": ["scanned.pdf"], "focus": question}),
    ));
    let answer = "\"The warranty lasts seven years from delivery.\" [p. 1]";
    fake.say(answer);
    let got = events(agent.turn(question).unwrap());
    assert_eq!(text(&got), answer, "{got:?}");
}

#[test]
fn a_stopped_reader_reads_the_document_again_at_the_next_question() {
    let fake = Fake::new(8192);
    let lease = doc(&[
        "The rent is $900 a month.\n".into(),
        "The deposit is $1,800.\n".into(),
    ]);
    let cases = [
        (
            lease,
            6000,
            "What is the deposit?",
            "[p. 2]\nThe deposit is $1,800.",
        ),
        // Past the whole-document rule: its first read.
        (
            long(4),
            300,
            "How long does the warranty last?",
            "[p. 13]\nTerms of care.\nThe warranty lasts seven years",
        ),
    ];
    for (d, followup_tokens, question, text) in cases {
        let s = settings(Family::Llama3, 0, followup_tokens);
        let mut reader = Reader::default();
        let stop = AtomicBool::new(true);
        let got = reader.reply(
            &fake.larch,
            &ctx(&fake),
            &s,
            std::slice::from_ref(&d),
            &ask(question),
            &stop,
            &mut |_| {},
            &|_| {},
        );
        assert!(
            matches!(got, Err(SkillError::Engine(EngineError::Cancelled))),
            "{got:?}"
        );
        let start = fake.fed_tokens();
        fake.say("Seven years.");
        assert_eq!(
            reply(&fake, &mut reader, &s, &d, question).text,
            "Seven years."
        );
        let fed = fake.fed_from(start);
        assert!(fed.contains(text), "{fed}");
        assert!(!fed.contains("already above"), "{fed}");
    }
}

#[test]
fn after_a_reset_the_pages_the_last_answer_cited_come_again_only_within_the_read() {
    let fake = Fake::new(8192);
    let d = manual(&fake, 12, None);
    let max = 160;
    let s = capped(600, max as u32);
    let mut reader = Reader::default();
    let stop = AtomicBool::default();
    // The last answer cites one page with its section, and the whole manual.
    fake.say("\"Noise must stay under 70 dB.\" [p. 10, Section 10], as [pp. 1-12] say.");
    let q1 = "What is the noise limit?";
    reply_all(&fake, &mut reader, &s, &d, q1, &stop, &|_| {}).unwrap();
    // The next question no longer fits the kept session with its answer's room.
    let first = &trace(&fake)[0];
    let held =
        first["prompt_tokens"].as_u64().unwrap() + first["generated_tokens"].as_u64().unwrap();
    let over = 8192 - held - max as u64 + 64;
    let q2 = filled(&fake, "Which limits does the manual set?\n", over as usize);
    let answer = "\"Wind speed must stay under 20 m/s.\" [p. 1]";
    fake.say(answer);
    let r = reply_all(&fake, &mut reader, &s, &d, &q2, &stop, &|_| {}).unwrap();
    assert_eq!(r.text, answer);
    assert_eq!(steps(&fake), ["read", "read", "reset"]);
    // The cited page came again; the whole manual is past the read's budget.
    assert!(r.ranges.contains(&page(&d, 10)), "{:?}", r.ranges);
    let whole = d.pages[11].end - d.pages[0].start;
    assert!(
        r.ranges.iter().all(|x| x.end - x.start < whole / 2),
        "{:?}",
        r.ranges
    );
}

#[test]
fn a_kept_reader_hears_the_chats_last_exchange_when_it_is_not_its_own() {
    let fake = Fake::chatml(8192);
    let d = doc(&[
        "The rent is $900 a month.\n".into(),
        "The deposit is $1,800.\n".into(),
    ]);
    let s = settings(Family::Llama3, 0, 6000);
    let mut reader = Reader::default();
    let deposit = "\"The deposit is $1,800.\" [p. 2]";
    fake.say(deposit);
    reply(&fake, &mut reader, &s, &d, "What is the deposit?");
    let mut asked = |last: (&str, &str), question: &str| {
        let start = fake.fed_tokens();
        fake.say("\"The rent is $900 a month.\" [p. 1]");
        let ask = Ask {
            question,
            last: Some(last),
            ..Ask::default()
        };
        let docs = std::slice::from_ref(&d);
        let stop = AtomicBool::default();
        (reader.reply(
            &fake.larch,
            &ctx(&fake),
            &s,
            docs,
            &ask,
            &stop,
            &mut |_| {},
            &|_| {},
        ))
        .unwrap();
        fake.fed_from(start)
    };
    // Its own answer, with the notes the chat showed under it: the session holds it.
    let shown = format!("{deposit}\n\n_Number not found anywhere in the text: 2._");
    let fed = asked(("What is the deposit?", &shown), "And the rent?");
    assert!(
        fed.contains("<|im_start|>user\nQuestion: And the rent?"),
        "{fed}"
    );
    // The chat asked about another document since.
    let fed = asked(
        ("Are pets allowed in b.pdf?", "Cats are allowed."),
        "Does it say that?",
    );
    assert!(
        fed.contains(
            "<|im_start|>user\nLast question: Are pets allowed in b.pdf?\nLast answer: Cats are \
             allowed.\n\nQuestion: Does it say that?"
        ),
        "{fed}"
    );
}

#[test]
fn the_held_back_end_of_an_answer_that_starts_with_a_blank_line_stays_in_it() {
    let fake = Fake::new(8192);
    let d = long(4);
    let s = settings(Family::Llama3, 0, 300);
    // Its last line starts as a Llama 3 call does, so the stream holds it until the step ends.
    let answer = "The terms are:\n{seven years} [p. 13]";
    fake.say(&format!("\n{answer}"));
    let question = "How long does the warranty last?";
    let (r, live) = shown(&fake, &mut Reader::default(), &s, &d, question);
    assert_eq!(r.text, answer);
    assert_eq!(live.concat().trim(), answer, "{live:?}");
}

#[test]
fn a_scanned_pdf_links_summary_reads_its_pictures_and_a_followup_quotes_them() {
    if !ocr::available() {
        println!("skipped: OCR is not available on this machine");
        return;
    }
    let fake = Fake::new(32768);
    config(&fake, "max_workers = 1\ndoc_followups = \"on\"\n");
    let mut agent = fake.agent();
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/scanned.pdf");
    let url = common::serve_as("application/pdf", fs::read(fixture).unwrap());
    fake.say(&call("read_papers", json!({"papers": [url]})));
    fake.say("A services agreement.");
    let got = events(agent.turn(&format!("summarize {url}")).unwrap());
    let said = larch_agent::AgentEvent::Progress("Reading scanned pages: 2 of 2".into());
    assert!(got.contains(&said), "{got:?}");
    assert!(fake.fed().contains("The warranty lasts seven years"));
    // The follow-up reopens the link with its pictures' text, and its check matches the quote.
    let question = "How long is the warranty?";
    fake.say(&call(
        "read_papers",
        json!({"papers": [url], "focus": question}),
    ));
    let answer = "\"The warranty lasts seven years from delivery.\" [p. 1]";
    fake.say(answer);
    let got = events(agent.turn(question).unwrap());
    assert_eq!(text(&got), answer, "{got:?}");
}
