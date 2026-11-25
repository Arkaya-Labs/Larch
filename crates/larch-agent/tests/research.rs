//! The research skill on the CPU: pages from a local HTTP server, a fake `web_search`, and a
//! model whose template scripts each reply by the prompt it renders. Two rounds, each budget, a
//! round with nothing new, a later round's ranking, the planner's prompt (ledger only, folded
//! past K), a quote its page no longer shows, the sections' batch, the chat tool and the CLI.

use std::{
    collections::{BTreeMap, HashSet},
    io::{BufRead, BufReader, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use larch::{EngineError, Larch};
use larch_agent::{
    audit::AuditEvent,
    skill::{SkillError, SkillRun, TypedResult},
    skills::{
        ask_docs::{Cited, WORKER_SYSTEM},
        research::{
            self, Effort, Evidence, Observation, Plan, Planned, Research, Section, State, outline,
            planner_prompt, saved_id,
        },
        web_answer::MAX_PAGES,
    },
    tool::{ARGS_TOKENS, Permission, Registry, Tool, ToolCtx, ToolError, ToolSpec},
    web::fetch::{self, Seen},
    workers::Workers,
};
use larch_engine::{EngineOptions, start_with};
use larch_kernels::{CPU, KError, KvPool, Stream, Tensor};
use larch_model::{
    ChatTemplate, FakeModel, Model, ModelConfig, StepBatch, Tokenizer,
    gguf::{Gguf, Value as GgufValue, write},
};
use serde_json::{Value, json};

/// The height is on line 5; line 7 is never a fact, so it must never reach the planner.
const TOWER_PAGE: &str = "# Harbor Tower\n\nThe Harbor Tower is an office tower in Port Ellis.\n\n\
It is 212 metres tall and has 48 floors.\n\nThe lobby has a marble fountain.\n";
/// Sundays are on line 3.
const FERRY_PAGE: &str = "# Larkspur ferry\n\nSundays: 9am, 1pm and 5pm.\n";

const BOTH: &str = "Report on the Harbor Tower and the Larkspur ferry";
const TOWER_ONLY: &str = "Report on the Harbor Tower alone";
/// A request whose section writers fail.
const WRITERS_FAIL: &str = "the writers fail";
/// A request whose section writers write nothing but a blank.
const WRITERS_BLANK: &str = "the writers write nothing";
/// A request whose Tower section has a second sentence, which its outline leaves out.
const AT_LENGTH: &str = "Report on the Harbor Tower and the Larkspur ferry at length";
/// A request whose first reflection asks round 1's query again, in other words.
const REPEATS: &str = "Report on the Harbor Tower and the Larkspur ferry, searched twice";
/// A request researched with a cutoff: only its planner prompts may name one.
const CUTOFF: &str = "Report on the Harbor Tower and the Larkspur ferry, past the cutoff";
/// Its planner grows an outline as the rounds go (R5).
const GROWN: &str = "Report on the Harbor Tower and the Larkspur ferry, outlined as it goes";
/// Its tower section has two paragraphs, and its editor rewrites the join (R5).
const JOINED: &str = "Report on the Harbor Tower and the Larkspur ferry, joined";
/// JOINED, whose writers end past the editor's time: an editor turn fails the template.
const JOINED_LATE: &str = "Report on the Harbor Tower and the Larkspur ferry, joined late";
/// JOINED, whose editor drops the tower's citation.
const JOINED_UNCITED: &str = "Report on the Harbor Tower and the Larkspur ferry, joined uncited";
/// Its writers each write a sentence with a number and no source (R6).
const UNSOURCED: &str = "Report on the Harbor Tower and the Larkspur ferry, unsourced";
/// Its writers write nothing but such sentences, and keep them on their retry.
const ALL_UNSOURCED: &str = "Report on the Harbor Tower and the Larkspur ferry, all unsourced";
/// Its planner fails after round 1, as a prompt over the context does.
const REFLECT_FAILS: &str = "Report on the Harbor Tower, whose planner fails";
/// Its planner outlines round 1's fact, then gives no plan after the ferry round.
const LOST: &str = "Report on the Harbor Tower and the Larkspur ferry, last reflection lost";
/// Its planner outlines round 1's fact, then stops after the ferry round with no outline.
const NO_OUTLINE: &str = "Report on the Harbor Tower and the Larkspur ferry, outline not repeated";
/// Its plan gives eight queries, which find nothing.
const EIGHT: &str = "Report on the Harbor Tower, searched eight ways";
/// Its plan searches a spec table, whose reader picks seven of its lines.
const SPECS: &str = "Report on the Harbor Tower's spec table";

/// A table of eight rows, read as eight lines.
const SPEC_PAGE: &str = "# Specs\n\nHeight 212 m\nFloors 48\nLifts 12\nOpened 1931\nArchitect Ada Vale\n\
Steel 9000 t\nGlass 3000 m2\nTorque none\n";

/// A page of five long lines, lines 3 to 11, each a fact: its round's planner prompt holds them
/// all only in a large context.
const LONG_PAGE: &str = "# LONGPAGE\n\n\
Line one says the harbor survey of the long page measured every pier, quay and breakwater along the whole of the eastern shore, and found that the oldest of them, built of granite blocks quarried in the hills above the town, still stands within a hand's width of where its builders set it.\n\n\
Line two says the same survey counted the ships that call at the harbor in a year, and found that the ferries, the fishing boats and the small cargo ships that carry timber and grain together make some eleven thousand calls, most of them in the summer months.\n\n\
Line three says the tide at the harbor mouth rises and falls by about four metres at the springs, which is why the inner basin has gates that close at low water and keep the boats afloat until the sea comes back in over the sill.\n\n\
Line four says the lighthouse on the north mole was first lit with oil, then with gas, and has run on electricity since the year the power station opened, and that its beam can be seen from some twenty nautical miles out at sea on a clear night.\n\n\
Line five says the harbor board publishes its accounts every spring, and that the dues paid by the ships that call cover the upkeep of the piers, the dredging of the channel and the wages of the pilots who bring the larger ships in.\n";

// The scripted replies, by the prompt each one answers.
const NOTHING: &str = "[]";
const PLAN: &str = r#"{"sub_questions": ["How tall is the tower?", "When does the ferry run?"], "queries": ["harbor tower height"], "stop": "both answered"}"#;
const TOWER_FACTS: &str = r#"[{"line": 5, "claim": "The tower is 212 metres tall"}]"#;
const FERRY_FACTS: &str =
    r#"[{"line": 3, "claim": "The ferry leaves Sundays at 9am, 1pm and 5pm"}]"#;
const MORE: &str = r#"{"queries": ["larkspur ferry sunday"]}"#;
const DONE: &str = r#"{"queries": [], "stop": "both sub-questions are answered"}"#;
/// A tower-only reflection asks again for what it has: the search finds the same page.
const AGAIN: &str = r#"{"queries": ["harbor tower again"]}"#;
const OUTLINE: &str =
    r#"{"sections": [{"title": "Tower", "facts": [1]}, {"title": "Ferry", "facts": [2, 9]}]}"#;
const TOWER_SECTION: &str = "The Harbor Tower is 212 metres tall [1].";
/// Cites [1], which is not one of its facts, and [9], which is no fact.
const FERRY_SECTION: &str =
    "The ferry leaves on Sundays at 9am, 1pm and 5pm [2], unlike the tower [1, 9].";
const NOT_JSON: &str = "Let me think about the plan first.";
const BLANK_SECTION: &str = " ";
/// The answer to a question about a saved report, from its sources.
const FOLLOW_UP: &str = "It is 212 metres tall [1].";
const TOWER_AT_LENGTH: &str =
    "The Harbor Tower is 212 metres tall [1]. No page gives the year it opened.";
/// Line 5 of a saved report is its first section's text.
const REPORT_FACTS: &str =
    r#"[{"line": 5, "claim": "The Tower section ends: no page gives the year it opened"}]"#;
/// The answer to a question about a saved report itself; fact 3 is the report's, after its two
/// sources' facts.
const LAST_SENTENCE: &str = "It ends: \"No page gives the year it opened.\" [3]";
/// Round 1's query in other words and case, then the ferry's.
const AGAIN_REWORDED: &str = r#"{"queries": ["Height: harbor TOWER", "larkspur ferry sunday"]}"#;
/// GROWN's planner after the ferry round: its report and the outline it grew, with [7], no fact.
const GROWN_DONE: &str = r#"{"report": "The tower is 212 metres tall [1]; the ferry runs on Sundays [2].", "outline": [{"title": "Height", "facts": [1]}, {"title": "Sundays", "facts": [2, 7]}], "queries": [], "stop": "both answered"}"#;
/// JOINED's tower section: an opening paragraph the editor must not see, then the join's.
const TOWER_TWO: &str =
    "OPENING-PARAGRAPH: a tall tower.\n\nThe Harbor Tower is 212 metres tall [1].";
/// JOINED's editor: the tower's end keeps its [1], and the ferry's start adds the tower's.
const JOIN_REPLY: &str = r#"{"end": "All told, the Harbor Tower is 212 metres tall [1].", "start": "The ferry, unlike the tower [1], leaves on Sundays at 9am, 1pm and 5pm [2]."}"#;
/// JOINED_UNCITED's editor: the tower's end loses its [1], the ferry's start keeps its [2].
const JOIN_UNCITED: &str = r#"{"end": "All told, the Harbor Tower is very tall.", "start": "The ferry, unlike the tower, leaves on Sundays at 9am, 1pm and 5pm [2]."}"#;
/// UNSOURCED's tower writer, then its retry, which cuts the sentence.
const TOWER_UNSOURCED: &str = "The Harbor Tower is 212 metres tall [1]. It opened in 1931.";
/// UNSOURCED's ferry writer, first and on its retry, which keeps the sentence.
const FERRY_UNSOURCED: &str =
    "The ferry leaves on Sundays at 9am, 1pm and 5pm [2]. A ticket costs 4 dollars.";
/// ALL_UNSOURCED's writers, first and on their retry.
const ONLY_UNSOURCED: &str = "It opened in 1931.";
/// Its ferry writer gives the tower's height with the tower's fact, not one of its own.
const ACROSS: &str = "Report on the Harbor Tower and the Larkspur ferry, cited across";
const FERRY_ACROSS: &str =
    "The ferry leaves on Sundays at 9am, 1pm and 5pm [2]. The tower is 212 metres tall [1].";
/// Its tower writer writes curly quotes, a dash, an accent and Greek letters.
const UNICODE: &str = "Report on the Harbor Tower and the Larkspur ferry, in Unicode";
const TOWER_UNICODE: &str = "The tower’s “top” — by the café — is 212 metres up [1]. Ελλάδα!";
/// Its tower writer writes TOWER_UNSOURCED, and its retry fails.
const RETRIED_IN_VAIN: &str = "Report on the Harbor Tower and the Larkspur ferry, retried in vain";
const PLAN_SPECS: &str = r#"{"sub_questions": ["What are the tower's specs?"], "queries": ["spec table"], "stop": "done"}"#;
/// LOST's planner after round 1: an outline of the tower's fact alone.
const TOWER_OUTLINED: &str = r#"{"report": "The tower is 212 metres tall [1].", "outline": [{"title": "Tower", "facts": [1]}], "queries": ["larkspur ferry sunday"], "stop": ""}"#;
/// A plan with Python's quotes, which serde does not take (iris-009 on the Mac failed twice at
/// the same column with "key must be a string").
const MALFORMED: &str = r#"{"sub_questions": ["How tall is the tower?"], 'queries': ['harbor tower height'], 'stop': 'done'}"#;
const PLAN_EIGHT: &str = r#"{"sub_questions": ["How tall is it?"], "queries": ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"], "stop": "done"}"#;
/// Seven lines of the spec table, lines 3 to 9.
const SPEC_FACTS: &str = r#"[{"line": 3, "claim": "a"}, {"line": 4, "claim": "b"}, {"line": 5, "claim": "c"}, {"line": 6, "claim": "d"}, {"line": 7, "claim": "e"}, {"line": 8, "claim": "f"}, {"line": 9, "claim": "g"}]"#;

/// JOINED's outline: its titles share "tower", so the sections may overlap and the editor runs.
const OUTLINE_SHARED: &str = r#"{"sections": [{"title": "Tower", "facts": [1]}, {"title": "Tower ferry", "facts": [2, 9]}]}"#;

/// The bench row "Lexus ISF vs BMW M3" (bench/agent/research/briefs.jsonl): its planner names
/// both cars with their spellings and says which generations; its first query drops the IS F's
/// "F", as v3's planner did.
const LEXUS: &str = "Lexus ISF vs BMW M3";
const LEXUS_PLAN: &str = r#"{"subjects": [{"name": "Lexus IS F", "spellings": ["ISF", "IS-F"]}, {"name": "BMW M3", "spellings": ["E92 M3", "E90 M3"]}], "assumptions": ["same-generation cars: the 2008-2014 IS F against the 2008-2013 E90/E92 M3; not the IS 500 or the 2026 M3"], "sub_questions": ["How do their engines compare?", "Which is faster?"], "queries": ["Lexus IS vs BMW M3", "lexus isf e92 m3 comparison"], "stop": "both answered"}"#;
const COMPARISON_FACTS: &str =
    r#"[{"line": 3, "claim": "The 2012 IS F makes 416 hp and the 2012 M3 414 hp"}]"#;
const TRACK_FACTS: &str =
    r#"[{"line": 3, "claim": "The E92 M3 lapped Willow Springs in 1:33.9, the ISF in 1:35.2"}]"#;
/// What a reader of the IS 500 page would take: the lock must keep it from ever reading it.
const IS500_FACTS: &str = r#"[{"line": 3, "claim": "The 2026 IS 500 makes 472 hp"}]"#;
const LEXUS_OUTLINE: &str = r#"{"sections": [{"title": "Engines and pace", "facts": [1, 2]}]}"#;
const LEXUS_SECTION: &str =
    "The 2012 IS F makes 416 hp to the M3's 414 hp [1]. On track the M3 was quicker [2].";
/// A plan whose one subject, written as a plain name, no page it finds writes.
const PIER_PLAN: &str = r#"{"subjects": ["Larkspur pier"], "sub_questions": ["How long is the pier?"], "queries": ["harbor tower height"], "stop": "done"}"#;
/// A request whose first round reads LONG_PAGE.
const LONG: &str = "Report on the harbor, from the long page";
const LONG_PLAN: &str = r#"{"sub_questions": ["What does the long page say?", "When does the ferry run?"], "queries": ["long page"], "stop": "both answered"}"#;
const LONG_FACTS: &str = r#"[{"line": 3, "claim": "the oldest pier still stands"}, {"line": 5, "claim": "eleven thousand calls a year"}, {"line": 7, "claim": "four metre tides"}, {"line": 9, "claim": "the light is seen 20 miles out"}, {"line": 11, "claim": "dues pay the upkeep"}]"#;
/// A plan with two subjects, one of which no page it finds writes.
const TOWER_PIER: &str = "Report on the Harbor Tower and the Larkspur pier";
const TOWER_PIER_PLAN: &str = r#"{"subjects": ["Harbor Tower", "Larkspur pier"], "sub_questions": ["How tall is the tower?"], "queries": ["harbor tower height"], "stop": "done"}"#;

/// The fake model: the template names a cue token, and the model says the reply after it, then
/// ends the turn.
struct Scripted {
    inner: FakeModel,
    replies: HashSet<u32>,
    eos: u32,
    /// A reply and its runner-up: what the model says when a grammar refuses the reply.
    runner_up: (u32, u32),
    /// Replies (their tokens) the model takes a while to start, the first time it says them.
    slow: Vec<(u32, Duration)>,
}

impl Model for Scripted {
    fn config(&self) -> &ModelConfig {
        self.inner.config()
    }

    fn forward(
        &mut self,
        s: &Stream,
        b: &StepBatch,
        kv: &KvPool,
        logits: &Tensor,
    ) -> Result<(), KError> {
        // FakeModel says its input plus one: the reply after a cue, the next cue after a reply,
        // which becomes the end of the turn.
        self.inner.forward(s, b, kv, logits)?;
        let mut l = vec![0f32; logits.shape().iter().product::<i64>() as usize];
        s.download(logits.buffer(), logits.offset(), &mut l)?;
        let vocab = self.inner.config().vocab as usize;
        let mut wait = Duration::ZERO;
        for row in l.chunks_mut(vocab) {
            let pick = row.iter().position(|&x| x > 0.0).unwrap_or_default() as u32;
            if let Some(i) = self.slow.iter().position(|&(t, _)| t == pick) {
                wait = wait.max(self.slow.remove(i).1);
            }
            if self.replies.contains(&pick.wrapping_sub(1)) {
                row.fill(0.0);
                row[self.eos as usize] = 100.0;
            } else if pick == self.runner_up.0 {
                row[self.runner_up.1 as usize] = row[pick as usize] / 2.0;
            }
        }
        std::thread::sleep(wait);
        s.upload(logits.buffer(), logits.offset(), &l)
    }
}

fn scripted() -> Larch {
    scripted_in(None, Vec::new())
}

/// `scripted` on an engine whose sessions hold at most `max_context` tokens, when given, and
/// whose model takes a while to start the replies `slow` names (by token), the first time.
fn scripted_in(max_context: Option<u32>, slow: Vec<(u32, Duration)>) -> Larch {
    let replies = [
        NOTHING,
        PLAN,
        TOWER_FACTS,
        FERRY_FACTS,
        MORE,
        DONE,
        AGAIN,
        OUTLINE,
        TOWER_SECTION,
        FERRY_SECTION,
        NOT_JSON,
        BLANK_SECTION,
        FOLLOW_UP,
        TOWER_AT_LENGTH,
        REPORT_FACTS,
        LAST_SENTENCE,
        AGAIN_REWORDED,
        GROWN_DONE,
        TOWER_TWO,
        JOIN_REPLY,
        TOWER_UNSOURCED,
        FERRY_UNSOURCED,
        ONLY_UNSOURCED,
        PLAN_SPECS,
        SPEC_FACTS,
        TOWER_OUTLINED,
        PLAN_EIGHT,
        JOIN_UNCITED,
        FERRY_ACROSS,
        TOWER_UNICODE,
        MALFORMED,
        OUTLINE_SHARED,
        LEXUS_PLAN,
        COMPARISON_FACTS,
        TRACK_FACTS,
        IS500_FACTS,
        LEXUS_OUTLINE,
        LEXUS_SECTION,
        PIER_PLAN,
        TOWER_PIER_PLAN,
        LONG_PLAN,
        LONG_FACTS,
    ];
    let mut tokens: Vec<_> = (0..256)
        .map(|b| GgufValue::Str(format!("<0x{b:02X}>")))
        .collect();
    let mut types = vec![GgufValue::I32(6); 256];
    tokens.push(GgufValue::Str("<|eot_id|>".into()));
    types.push(GgufValue::I32(3));
    let mut ids = HashSet::new();
    for (i, reply) in replies.into_iter().enumerate() {
        tokens.extend([format!("<cue{i}>"), reply.into()].map(GgufValue::Str));
        types.extend([3, 4].map(GgufValue::I32));
        ids.insert(258 + 2 * i as u32);
    }
    // The token after the last reply, which the model picks before it ends the turn.
    tokens.push(GgufValue::Str("<pad>".into()));
    types.push(GgufValue::I32(3));
    let q = |s: &str| serde_json::to_string(s).unwrap();
    // Every planner prompt starts with the date; the cutoff shows only where the run has one.
    let dated = format!(
        "{{%- if not messages[1].content.startswith({today}) %}}\
         {{{{ raise_exception('the planner has no date: ' ~ messages[1].content) }}}}\
         {{%- elif ({cutoff} in messages[1].content) != ({ends} in messages[1].content) %}}\
         {{{{ raise_exception('the cutoff is wrong: ' ~ messages[1].content) }}}}",
        today = q(&format!("Today is {}.", &larch::bench::utc_now()[..10])),
        cutoff = q(CUTOFF),
        ends = q("Your own knowledge ends about Mar 2025."),
    );
    let template = format!(
        "{{%- if tools is defined %}}{{{{ raise_exception('a session with tools') }}}}\
         {{%- elif messages[0].content.startswith({worker}) -%}}\
         {{%- if '## Sources' in messages[1].content %}}<cue14>\
         {{%- elif 'Torque none' in messages[1].content %}}<cue24>\
         {{%- elif '212 metres' in messages[1].content %}}<cue2>\
         {{%- elif 'Sundays:' in messages[1].content %}}<cue3>\
         {{%- elif 'LONGPAGE' in messages[1].content %}}<cue41>\
         {{%- elif '416 hp' in messages[1].content and not ('Assuming: same-generation' in messages[0].content and 'Skip facts about a different time period' in messages[0].content) %}}{{{{ raise_exception('a reader without the assumptions') }}}}\
         {{%- elif '416 hp' in messages[1].content %}}<cue33>\
         {{%- elif '1:33.9' in messages[1].content %}}<cue34>\
         {{%- elif '472 hp' in messages[1].content %}}<cue35>\
         {{%- else %}}<cue0>{{% endif -%}}\
         {{%- elif messages[0].content.startswith('You plan') -%}}\
         {dated}\
         {{%- elif 'Bad plan' in messages[1].content and messages | length < 4 %}}<cue10>\
         {{%- elif 'Never a plan' in messages[1].content %}}<cue10>\
         {{%- elif 'Malformed plan' in messages[1].content %}}<cue30>\
         {{%- elif {specs} in messages[1].content %}}<cue23>\
         {{%- elif {eight} in messages[1].content %}}<cue26>\
         {{%- elif {lexus} in messages[1].content %}}<cue32>\
         {{%- elif {tower_pier} in messages[1].content %}}<cue39>\
         {{%- elif {long_page} in messages[1].content %}}<cue40>\
         {{%- elif 'Larkspur pier' in messages[1].content %}}<cue38>\
         {{%- else %}}<cue1>{{% endif -%}}\
         {{%- elif messages[0].content.startswith('You direct') -%}}\
         {{{{ messages[0].content ~ messages[1].content }}}}\
         {dated}\
         {{%- elif 'Prose reflect' in messages[1].content and 'Sundays' not in messages[1].content and messages | length < 4 %}}<cue10>\
         {{%- elif 'Prose reflect' in messages[1].content and 'Sundays' not in messages[1].content %}}<cue25>\
         {{%- elif 'fountain' in messages[1].content %}}{{{{ raise_exception('page text reached the planner') }}}}\
         {{%- elif {reflect_fails} in messages[1].content %}}{{{{ raise_exception('the planner failed') }}}}\
         {{%- elif {specs} in messages[1].content and '[6]' in messages[1].content %}}{{{{ raise_exception('a sixth fact of one page') }}}}\
         {{%- elif {specs} in messages[1].content %}}<cue5>\
         {{%- elif {lost} in messages[1].content and 'Sundays' in messages[1].content %}}<cue10>\
         {{%- elif {lost} in messages[1].content %}}<cue25>\
         {{%- elif {noout} in messages[1].content and 'Sundays' not in messages[1].content %}}<cue25>\
         {{%- elif {alone} in messages[1].content %}}<cue6>\
         {{%- elif {grown} in messages[1].content and 'Sundays' in messages[1].content %}}<cue17>\
         {{%- elif {repeats} in messages[1].content and 'Sundays' not in messages[1].content %}}<cue16>\
         {{%- elif 'Sundays' in messages[1].content %}}<cue5>\
         {{%- else %}}<cue4>{{% endif -%}}\
         {{%- elif messages[0].content.startswith('You outline') -%}}\
         {{%- if {grown} in messages[1].content %}}{{{{ raise_exception('the outline step ran') }}}}\
         {{%- elif {joined} in messages[1].content %}}<cue31>\
         {{%- elif {lexus} in messages[1].content and 'Assuming: same-generation' not in messages[1].content %}}{{{{ raise_exception('an outline without the assumptions') }}}}\
         {{%- elif {lexus} in messages[1].content %}}<cue36>\
         {{%- else %}}<cue7>{{% endif -%}}\
         {{%- elif messages[0].content.startswith('Write one section') -%}}\
         {{%- if {grown} in messages[1].content and not ('- Height' in messages[1].content and '- Sundays' in messages[1].content) %}}{{{{ raise_exception('a writer without the outline') }}}}\
         {{%- elif {lexus} in messages[1].content and 'Assuming: same-generation' not in messages[1].content %}}{{{{ raise_exception('a writer without the assumptions') }}}}\
         {{%- elif {lexus} in messages[1].content %}}<cue37>\
         {{%- elif {joined} in messages[1].content and '212 metres' in messages[1].content %}}<cue18>\
         {{%- elif {all_unsourced} in messages[1].content %}}<cue22>\
         {{%- elif {across} in messages[1].content and '212 metres' not in messages[1].content %}}<cue28>\
         {{%- elif {unicode} in messages[1].content and '212 metres' in messages[1].content %}}<cue29>\
         {{%- elif {in_vain} in messages[1].content and messages | length > 2 %}}{{{{ raise_exception('the retry failed') }}}}\
         {{%- elif {in_vain} in messages[1].content and '212 metres' in messages[1].content %}}<cue20>\
         {{%- elif {unsourced} in messages[1].content and messages | length > 2 -%}}\
         {{%- if 'It opened in 1931' in messages[3].content %}}<cue8>\
         {{%- elif 'A ticket costs 4 dollars' in messages[3].content %}}<cue21>\
         {{%- else %}}{{{{ raise_exception('a retry without its sentence: ' ~ messages[3].content) }}}}{{% endif -%}}\
         {{%- elif {unsourced} in messages[1].content and '212 metres' in messages[1].content %}}<cue20>\
         {{%- elif {unsourced} in messages[1].content %}}<cue21>\
         {{%- elif {fail} in messages[1].content %}}{{{{ raise_exception('the writer failed') }}}}\
         {{%- elif {blank} in messages[1].content %}}<cue11>\
         {{%- elif {long} in messages[1].content and '212 metres' in messages[1].content %}}<cue13>\
         {{%- elif '212 metres' in messages[1].content %}}<cue8>{{% else %}}<cue9>{{% endif -%}}\
         {{%- elif messages[0].content.startswith('You join') -%}}\
         {{%- if 'OPENING-PARAGRAPH' in messages[1].content %}}{{{{ raise_exception('the editor saw more than the join') }}}}\
         {{%- elif {late} in messages[1].content %}}{{{{ raise_exception('an editor turn past its time') }}}}\
         {{%- elif {uncited} in messages[1].content %}}<cue27>\
         {{%- elif {joined} in messages[1].content %}}<cue19>{{% else %}}<cue10>{{% endif -%}}\
         {{%- elif messages[0].content.startswith('Answer the user') -%}}\
         {{%- if 'last sentence' in messages[1].content %}}<cue15>{{% else %}}<cue12>{{% endif -%}}\
         {{%- else %}}{{{{ raise_exception('unexpected prompt: ' ~ messages[0].content) }}}}\
         {{%- endif -%}}",
        worker = q(WORKER_SYSTEM),
        alone = q(TOWER_ONLY),
        fail = q(WRITERS_FAIL),
        blank = q(WRITERS_BLANK),
        long = q(AT_LENGTH),
        repeats = q(REPEATS),
        grown = q(GROWN),
        joined = q(JOINED),
        unsourced = q(UNSOURCED),
        all_unsourced = q(ALL_UNSOURCED),
        across = q(ACROSS),
        unicode = q(UNICODE),
        in_vain = q(RETRIED_IN_VAIN),
        uncited = q(JOINED_UNCITED),
        late = q(JOINED_LATE),
        reflect_fails = q(REFLECT_FAILS),
        specs = q(SPECS),
        lost = q(LOST),
        eight = q(EIGHT),
        noout = q(NO_OUTLINE),
        lexus = q(LEXUS),
        tower_pier = q(TOWER_PIER),
        long_page = q(LONG),
    );
    let vocab = tokens.len() as u32;
    let metadata = [
        ("tokenizer.ggml.model", GgufValue::Str("llama".into())),
        ("tokenizer.ggml.tokens", GgufValue::Arr(tokens)),
        ("tokenizer.ggml.token_type", GgufValue::Arr(types)),
        ("tokenizer.ggml.eos_token_id", GgufValue::U32(256)),
        ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
        ("tokenizer.ggml.add_space_prefix", GgufValue::Bool(false)),
    ];
    let no_tensors: [(&str, Vec<u64>, u32, Vec<u8>); 0] = [];
    let g = Gguf::parse(&write(&metadata, &no_tensors)).unwrap();
    let tokenizer = Tokenizer::from_gguf(&g).unwrap();
    let template = ChatTemplate::new(&template, &tokenizer).unwrap();
    let mut options = EngineOptions::new("fake".into(), CPU);
    if let Some(n) = max_context {
        options.max_context = n;
    }
    let engine = start_with(
        options,
        Box::new(move |_, _, _| {
            Ok(Box::new(Scripted {
                inner: FakeModel::new(vocab),
                replies: ids.clone(),
                eos: 256,
                // MALFORMED's runner-up is PLAN.
                runner_up: (318, 260),
                slow: slow.clone(),
            }))
        }),
    )
    .unwrap();
    Larch::from_engine(engine, tokenizer, template)
}

/// A local HTTP server of the saved pages, which `fetch_many` fetches as any site.
struct Server {
    base: String,
    hits: Arc<Mutex<BTreeMap<String, usize>>>,
}

impl Server {
    fn start(pages: &[(&'static str, &'static str)]) -> Server {
        fetch::allow_this_machine();
        // The process's first certificate load blocks every fetch: 5 s on the M5 runner, as long
        // as HOST_WAIT, so a test's first pages time out. Loaded here, before any clock starts.
        drop(larch_fetch::agent("https://example.com"));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let pages: BTreeMap<_, _> = pages.iter().copied().collect();
        let hits: Arc<Mutex<BTreeMap<String, usize>>> = Arc::default();
        let counted = Arc::clone(&hits);
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                // The headers, up to the blank line.
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap() > 2 {
                    line.clear();
                }
                let path = first.split(' ').nth(1).unwrap_or("/").to_string();
                *counted.lock().unwrap().entry(path.clone()).or_insert(0) += 1;
                // robots.txt is not found, which allows everything.
                let reply = match pages.get(path.as_str()) {
                    Some(b) => format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/markdown\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
                        b.len()
                    ),
                    None => {
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                            .into()
                    }
                };
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        Server { base, hits }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    /// GETs of `path` so far.
    fn hits(&self, path: &str) -> usize {
        self.hits.lock().unwrap().get(path).copied().unwrap_or(0)
    }
}

/// A fake `web_search`: each query's hits, by URL. With `tamper`, the second round's search
/// first rewrites the tower page's saved copy without its height, as a cache gone bad would.
struct Search(BTreeMap<String, Vec<String>>, bool);

impl Tool for Search {
    fn spec(&self) -> ToolSpec {
        spec("web_search")
    }

    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        if self.1 && args["query"] == "larkspur ferry sunday" {
            for dir in std::fs::read_dir(ctx.data_dir.join("agent/web")).unwrap() {
                for file in std::fs::read_dir(dir.unwrap().path()).unwrap() {
                    let path = file.unwrap().path();
                    let text = std::fs::read_to_string(&path).unwrap();
                    let fabricated = text.replace("It is 212 metres tall", "It is short");
                    std::fs::write(&path, fabricated).unwrap();
                }
            }
        }
        let urls = self
            .0
            .get(args["query"].as_str().unwrap())
            .cloned()
            .unwrap_or_default();
        let hits: Vec<_> = (urls.iter())
            .map(|u| json!({"title": u, "url": u, "snippet": "..."}))
            .collect();
        Ok(json!(hits).to_string())
    }
}

/// A web search that takes `.1` to answer, as a slow engine does.
struct Slow(Search, Duration);

impl Tool for Slow {
    fn spec(&self) -> ToolSpec {
        self.0.spec()
    }

    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        std::thread::sleep(self.1);
        self.0.call(args, ctx)
    }
}

fn spec(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: format!("The fake {name}."),
        schema: json!({"type": "object"}),
        permission: Permission::Read,
        untrusted_results: true,
        max_result_tokens: 8192,
        max_args_tokens: ARGS_TOKENS,
    }
}

/// The server's pages, and a search where each query finds its page.
fn web(tamper: bool) -> (Server, Registry) {
    let server = Server::start(&[
        ("/tower", TOWER_PAGE),
        ("/ferry", FERRY_PAGE),
        ("/specs", SPEC_PAGE),
        ("/long", LONG_PAGE),
    ]);
    let search = [
        ("harbor tower height", "/tower"),
        ("harbor tower again", "/tower"),
        ("larkspur ferry sunday", "/ferry"),
        ("spec table", "/specs"),
        ("long page", "/long"),
    ]
    .map(|(q, p)| (q.to_string(), vec![server.url(p)]));
    let mut tools = Registry::builtin();
    tools.replace(Box::new(Search(search.into_iter().collect(), tamper)));
    (server, tools)
}

fn ctx(tmp: &Path) -> ToolCtx {
    ToolCtx {
        allowed: Vec::new(),
        data_dir: tmp.join("data"),
        source: String::new(),
    }
}

/// Three sessions at a time.
const WORKERS: Workers = Workers {
    max_workers: 3,
    prefix_cache: false,
};

/// `request` researched by `skill`, three sessions at a time (two at once).
fn research(
    skill: Research,
    tmp: &Path,
    tools: &Registry,
    request: &str,
) -> (TypedResult, SkillRun) {
    (skill.run_with(
        &scripted(),
        &ctx(tmp),
        WORKERS,
        request,
        tools,
        Seen::default(),
        &AtomicBool::default(),
    ))
    .unwrap()
}

/// The skill as before v4's one-round standard: two rounds, with a reflect turn after each.
fn two() -> Research {
    Research {
        rounds: 2,
        ..Research::default()
    }
}

/// The sections (title, text), the sources' and the unverified facts' numbers, and the stop.
type Parts = (Vec<(String, String)>, Vec<usize>, Vec<usize>, String);

/// The report's parts.
fn report(r: &TypedResult) -> Parts {
    let TypedResult::Report {
        sections,
        sources,
        unverified,
        stopped,
        ..
    } = r
    else {
        panic!("not a report")
    };
    (
        sections
            .iter()
            .map(|s| (s.title.clone(), s.text.clone()))
            .collect(),
        sources.iter().map(|e| e.n).collect(),
        unverified.iter().map(|u| u.fact.n).collect(),
        stopped.clone(),
    )
}

fn steps<'a>(run: &'a SkillRun, name: &str) -> Vec<&'a AuditEvent> {
    run.steps
        .iter()
        .filter(|e| e.step.as_deref() == Some(name))
        .collect()
}

#[test]
fn two_rounds_then_the_planner_stops_and_each_section_cites_its_own_facts() {
    let tmp = tempfile::tempdir().unwrap();
    let (server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, BOTH);
    let (sections, sources, unverified, stopped) = report(&result);
    assert_eq!(
        stopped,
        "the planner stopped: both sub-questions are answered"
    );
    assert_eq!(steps(&run, "search").len(), 2);
    assert_eq!(steps(&run, "reflect").len(), 2);
    assert_eq!(
        sections,
        [
            ("Tower".into(), TOWER_SECTION.into()),
            // [1] is the tower's fact and [9] no fact: both go.
            (
                "Ferry".into(),
                "The ferry leaves on Sundays at 9am, 1pm and 5pm [2], unlike the tower.".into()
            ),
        ]
    );
    assert_eq!((sources, unverified), (vec![1, 2], vec![]));
    let TypedResult::Report { sources, .. } = &result else {
        panic!("not a report")
    };
    assert_eq!(sources[1].cited.quote, "Sundays: 9am, 1pm and 5pm.");
    assert_eq!(sources[1].final_url, server.url("/ferry"));
    assert!(
        sources[1].fetched_at.ends_with('Z'),
        "{}",
        sources[1].fetched_at
    );
    assert_eq!(sources[1].sha256.len(), 64);
    // Each page was fetched once: the re-check reads the saved copies.
    assert_eq!((server.hits("/tower"), server.hits("/ferry")), (1, 1));
    assert!(run.markdown.contains("## Sources"), "{}", run.markdown);
    assert!(
        run.markdown.contains(&format!(
            "[2] {}, \"Sundays: 9am, 1pm and 5pm.\"",
            server.url("/ferry")
        )),
        "{}",
        run.markdown
    );
    assert_eq!(run.facts, serde_json::to_value(&result).unwrap());
}

/// Each page's reader is a helper the chat shows, named by its site, before the section writers:
/// in a user's demo (16 Nov) a run's first minutes read 8 pages with no helper on screen.
#[test]
fn each_pages_reader_is_a_helper_named_by_its_site_before_the_writers() {
    use larch_agent::workers::{Helper, HelperState::Done, telling};
    let tmp = tempfile::tempdir().unwrap();
    let (server, tools) = web(false);
    let (to, heard) = std::sync::mpsc::channel();
    let (result, _) = telling(to, 0, || research(two(), tmp.path(), &tools, BOTH));
    report(&result);
    // Each helper as it ended, by id.
    let mut board = BTreeMap::new();
    for h in heard.try_iter() {
        board.insert(h.id, h);
    }
    let site = server.url("").split_once("://").unwrap().1.to_string();
    let helper = |id, task: String| Helper {
        id,
        task,
        state: Done,
    };
    let reading = format!("Reading {site}");
    let want = [
        // Round 1 reads the tower page, round 2 the ferry page.
        helper(0, reading.clone()),
        helper(1, reading),
        helper(2, "Writing Tower".into()),
        helper(3, "Writing Ferry".into()),
    ];
    assert_eq!(board.into_values().collect::<Vec<_>>(), want);
}

#[test]
fn a_saved_report_leaves_its_outline_and_a_question_about_it_reads_its_sources_again() {
    let tmp = tempfile::tempdir().unwrap();
    let (server, tools) = web(false);
    let (larch, ctx, workers, stop) = (scripted(), ctx(tmp.path()), WORKERS, AtomicBool::default());
    let skill = two();
    let report = (skill.run_saved(
        &larch,
        &ctx,
        workers,
        BOTH,
        None,
        &tools,
        Seen::default(),
        &stop,
        &mut |_| {},
    ))
    .unwrap();
    let id = report
        .rsplit("report ")
        .next()
        .unwrap()
        .trim_end_matches("._");
    assert_eq!(id.len(), 12, "{report}");
    let path = ctx.data_dir.join(format!("agent/reports/{id}.json"));
    // Every page the run read is kept for a follow-up, by its saved copy's key.
    let saved: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let fetched: Vec<&str> = (saved["fetched"].as_array().unwrap().iter())
        .map(|p| p[0].as_str().unwrap())
        .collect();
    assert_eq!(fetched, [server.url("/tower"), server.url("/ferry")]);
    assert_eq!(saved["fetched"][0][1].as_str().unwrap().len(), 64);
    assert_eq!(
        outline(&report).unwrap(),
        format!(
            "# {BOTH}\n\n## Tower\n\n{TOWER_SECTION}\n\n## Ferry\n\nThe ferry leaves on Sundays at \
             9am, 1pm and 5pm [2], unlike the tower.\n\n_Stopped: the planner stopped: both \
             sub-questions are answered._\n\n_Saved as report {id}._"
        )
    );
    // A section keeps its first sentence; any other answer has no outline.
    let two = "# R\n\n## A\n\nOne [1]. Two.\n\n## Sources\n\n[1] x\n\n_Stopped: s._\n\n_Saved as report a._";
    assert_eq!(
        outline(two).unwrap(),
        "# R\n\n## A\n\nOne [1].\n\n_Stopped: s._\n\n_Saved as report a._"
    );
    assert_eq!(outline(TOWER_SECTION), None);

    let answer = skill
        .follow_up(&larch, &ctx, workers, id, "How tall is the tower?", &stop)
        .unwrap();
    assert!(answer.starts_with(FOLLOW_UP), "{answer}");
    // It names the report it read, so a report its title words picked wrongly shows.
    let named = format!("_From report {id}, on \"{BOTH}\"._");
    assert!(answer.ends_with(&named), "{answer}");
    assert!(
        answer.contains(&format!(
            "[1] {}, \"It is 212 metres tall and has 48 floors.\"",
            server.url("/tower")
        )),
        "{answer}"
    );
    // Read from the saved copies: nothing was fetched again.
    assert_eq!((server.hits("/tower"), server.hits("/ferry")), (1, 1));
    // The id names a file: only one of ours.
    let wrong = skill.follow_up(&larch, &ctx, workers, "../../x", "?", &stop);
    assert!(
        matches!(wrong, Err(SkillError::Tool(ToolError::Invalid(_)))),
        "{wrong:?}"
    );
}

#[test]
fn a_question_about_a_saved_report_itself_is_answered_from_its_own_text() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (larch, ctx, stop) = (scripted(), ctx(tmp.path()), AtomicBool::default());
    let skill = two();
    let report = (skill.run_saved(
        &larch,
        &ctx,
        WORKERS,
        AT_LENGTH,
        None,
        &tools,
        Seen::default(),
        &stop,
        &mut |_| {},
    ))
    .unwrap();
    assert!(report.contains(TOWER_AT_LENGTH), "{report}");
    assert!(!outline(&report).unwrap().contains("the year it opened"));
    let id = report
        .rsplit("report ")
        .next()
        .unwrap()
        .trim_end_matches("._");
    let question = "What is the last sentence of the report's Tower section?";
    let answer = (skill.follow_up(&larch, &ctx, WORKERS, id, question, &stop)).unwrap();
    assert!(
        answer.contains(&format!(
            "[1] {AT_LENGTH}, report {id}, \"{TOWER_AT_LENGTH}\""
        )),
        "{answer}"
    );
    assert!(
        answer.starts_with("It ends: \"No page gives the year it opened.\" [1]"),
        "{answer}"
    );
}

/// `text` saved as `fetch_many` saves the page read at `url`.
fn saved_page(data: &Path, url: &str, text: &str) -> fetch::Page {
    let hex = |b: &[u8]| larch::bench::hex(&larch::bench::sha256(b));
    let page = fetch::Page {
        url: url.into(),
        final_url: url.into(),
        fetched_at: larch::bench::utc_now(),
        sha256: hex(text.as_bytes()),
        title: String::new(),
        text: text.into(),
        bytes: text.len() as u64,
        truncated: false,
        published: None,
        pdf: false,
    };
    let dir = data.join("agent/web").join(&hex(url.as_bytes())[..16]);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join(format!("{}.json", page.sha256));
    std::fs::write(file, serde_json::to_vec(&page).unwrap()).unwrap();
    page
}

#[test]
fn a_follow_up_ranks_every_page_the_run_read_so_an_uncited_one_can_answer() {
    let tmp = tempfile::tempdir().unwrap();
    let (larch, ctx, stop) = (scripted(), ctx(tmp.path()), AtomicBool::default());
    // The report cites as many pages as a follow-up reads, none about the tower; the run read
    // the tower's page too, and cited nothing of it.
    let tower = saved_page(&ctx.data_dir, "https://pages.example/tower", TOWER_PAGE);
    let sources: Vec<Evidence> = (1..=MAX_PAGES)
        .map(|n| {
            let quote = format!("The rose garden number {n} blooms in May.");
            let url = format!("https://pages.example/garden-{n}");
            let p = saved_page(&ctx.data_dir, &url, &format!("# Garden {n}\n\n{quote}\n"));
            Evidence {
                n,
                cited: Cited {
                    claim: "It blooms in May".into(),
                    quote,
                    file: PathBuf::from(&url),
                    line: 3,
                },
                url: p.url,
                final_url: p.final_url,
                title: p.title,
                fetched_at: p.fetched_at,
                sha256: p.sha256,
                published: None,
            }
        })
        .collect();
    let mut fetched: Vec<Value> = (sources.iter()).map(|e| json!([e.url, e.sha256])).collect();
    fetched.push(json!([tower.url, tower.sha256]));
    let report = TypedResult::Report {
        sections: vec![Section {
            title: "Gardens".into(),
            text: "The roses bloom in May [1].".into(),
        }],
        sources,
        unverified: Vec::new(),
        stopped: "the planner stopped: done".into(),
        pages: MAX_PAGES + 1,
        fixed: 0,
        removed: 0,
        off_subject: 0,
        thin: 0,
        notes: Vec::new(),
    };
    let id = "0123456789ab";
    let save = |saved: Value| {
        let dir = ctx.data_dir.join("agent/reports");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{id}.json")), saved.to_string()).unwrap();
    };
    let ask = || two().follow_up(&larch, &ctx, WORKERS, id, "How tall is the tower?", &stop);
    let request = "Report on the rose gardens";
    // A report saved before it kept the pages it read still loads, and reads its cited pages.
    save(json!({"request": request, "report": report}));
    let before = ask().unwrap();
    assert!(!before.contains("212 metres"), "{before}");
    save(json!({"request": request, "report": report, "fetched": fetched}));
    let answer = ask().unwrap();
    assert!(answer.starts_with(FOLLOW_UP), "{answer}");
    assert!(
        answer.contains(&format!(
            "[1] {}, \"It is 212 metres tall and has 48 floors.\"",
            tower.url
        )),
        "{answer}"
    );
    // A lost copy of a page the run read but did not cite does not fail the follow-up.
    fetched.push(json!(["https://pages.example/gone", "00"]));
    save(json!({"request": request, "report": report, "fetched": fetched}));
    assert!(ask().unwrap().contains("212 metres"));
}

#[test]
fn each_budget_stops_the_run_and_is_named() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    for (skill, named, (rounds, reflects)) in [
        (
            Research { rounds: 1, ..two() },
            "the round budget (1 round) was spent",
            (1, 0),
        ),
        (
            Research { pages: 1, ..two() },
            "the page budget (1 page) was spent",
            (1, 1),
        ),
        (
            Research {
                wall: Duration::ZERO,
                ..two()
            },
            "the time budget (0 s) was spent",
            (0, 0),
        ),
    ] {
        let (result, run) = research(skill, tmp.path(), &tools, BOTH);
        let (.., stopped) = report(&result);
        assert_eq!(stopped, named);
        assert_eq!(steps(&run, "search").len(), rounds, "{named}");
        // A spent budget searches no more; the planner still files the last round into its
        // report and outline (R5), but a one-round run has no reflect turn (v4).
        assert_eq!(steps(&run, "reflect").len(), reflects, "{named}");
        assert!(
            run.markdown.ends_with(&format!("_Stopped: {named}._")),
            "{}",
            run.markdown
        );
        // A run stopped early still ends with the verification line (R6).
        assert!(run.markdown.contains(" verified. "), "{}", run.markdown);
    }
}

/// The levels' budgets are set in one place: quick is web_answer, standard and deep are research's
/// (v3 R4), and a run is standard unless the user picks deep.
#[test]
fn the_levels_live_in_one_place() {
    let secs = Duration::from_secs;
    // v4 PR 4: standard is 1 round, 20 pages and 60 s; deep 3 rounds, 40 pages and 4 minutes.
    assert_eq!(Effort::Standard.budget(), (1, 20, secs(60)));
    assert_eq!(Effort::Deep.budget(), (3, 40, secs(240)));
    let budget = |r: Research| (r.rounds, r.pages, r.wall);
    assert_eq!(budget(Research::default()), Effort::Standard.budget());
    assert_eq!(
        budget(Research::default().at(Effort::Deep)),
        Effort::Deep.budget()
    );
    assert_eq!(Effort::default(), Effort::Standard);
    assert_eq!(serde_json::to_value(Effort::Deep).unwrap(), json!("deep"));
}

/// A reflect turn that writes prose, not its JSON, is asked once more in the same session, held
/// to the JSON's grammar, as the first plan is; the run goes on to round 2 (the Mac's deep run
/// had one write a report). The retry's reply is TOWER_OUTLINED, the only scripted reflection
/// the grammar lets through.
#[test]
fn a_reflect_turn_in_prose_is_asked_again_and_the_run_goes_on() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, &format!("Prose reflect: {BOTH}"));
    let reflects = steps(&run, "reflect");
    assert_eq!(reflects.len(), 3);
    assert!(reflects[0].error.is_some() && reflects[1].error.is_none());
    assert_eq!(steps(&run, "search").len(), 2);
    assert_eq!(
        report(&result).3,
        "the planner stopped: both sub-questions are answered"
    );
}

#[test]
fn a_query_asked_before_is_not_searched_again() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (_, run) = research(two(), tmp.path(), &tools, REPEATS);
    let searched: Vec<_> = (steps(&run, "search").iter())
        .map(|e| e.detail.as_ref().unwrap()["queries"].clone())
        .collect();
    assert_eq!(
        searched,
        [
            json!(["harbor tower height"]),
            json!(["larkspur ferry sunday"])
        ]
    );
}

#[test]
fn the_planner_reads_the_date_and_the_cutoff() {
    // The template fails a plan or reflect prompt without today's date first, or with a cutoff
    // this run does not have: every other run here has none.
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let skill = Research {
        knows_until: Some("Mar 2025"),
        ..two()
    };
    let (result, run) = research(skill, tmp.path(), &tools, CUTOFF);
    assert_eq!(
        report(&result).3,
        "the planner stopped: both sub-questions are answered"
    );
    assert_eq!(steps(&run, "reflect").len(), 2);
}

#[test]
fn a_round_with_no_new_fact_stops_the_run() {
    let tmp = tempfile::tempdir().unwrap();
    let (server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, TOWER_ONLY);
    let (sections, sources, ..) = report(&result);
    assert_eq!(report(&result).3, "a round found no new fact");
    assert_eq!(steps(&run, "search").len(), 2);
    // The second search found only the page already read, which is not fetched again.
    assert_eq!(server.hits("/tower"), 1);
    assert_eq!(sources, [1]);
    assert_eq!(sections.len(), 1);
}

#[test]
fn one_worker_reads_each_page_whole_as_its_card() {
    // Four paragraphs that each make a chunk of their own, on one page: one card, one worker.
    const TIMETABLE: &str = "/timetable";
    let filler = "The Larkspur ferry crosses the bay. ".repeat(20);
    let sunday = "On a sunday the boat waits at the pier. ".repeat(18);
    let page = format!("{filler}\n\n{filler}\n\n{filler}\n\n{filler}\n\n{sunday}\n");
    let server = Server::start(&[("/tower", TOWER_PAGE), (TIMETABLE, page.leak())]);
    let search = [
        ("harbor tower height", "/tower"),
        ("larkspur ferry sunday", TIMETABLE),
    ]
    .map(|(q, p)| (q.to_string(), vec![server.url(p)]));
    let mut tools = Registry::builtin();
    tools.replace(Box::new(Search(search.into_iter().collect(), false)));
    let tmp = tempfile::tempdir().unwrap();
    let (_, run) = research(two(), tmp.path(), &tools, BOTH);
    let ranks = steps(&run, "rank");
    assert_eq!(ranks.len(), 2);
    for rank in ranks {
        let chosen = rank.detail.as_ref().unwrap()["chosen"].as_array().unwrap();
        assert_eq!(chosen.len(), 1, "{chosen:?}");
        assert_eq!(chosen[0]["line"], 1, "{chosen:?}");
    }
}

#[test]
fn reflection_prompts_hold_the_ledger_and_no_raw_page_text() {
    // The template fails any reflection whose prompt holds the tower page's fountain line, which
    // no worker picks: the two-round run passing is the check on the rendered prompts.
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (_, run) = research(two(), tmp.path(), &tools, BOTH);
    assert!(steps(&run, "reflect").iter().all(|e| e.error.is_none()));
    // And on the prompt itself: the request, the plan, the facts' quotes, never a page.
    let state = state(1);
    let prompt = planner_prompt(&state, usize::MAX, &str::len);
    assert!(
        prompt.contains("It is 212 metres tall and has 48 floors."),
        "{prompt}"
    );
    assert!(!prompt.contains("fountain"), "{prompt}");
}

/// A run's state after `rounds` rounds, one tower fact a round.
fn state(rounds: usize) -> State {
    let fact = |n: usize| Evidence {
        n,
        cited: Cited {
            claim: format!("claim {n}"),
            quote: format!("It is 212 metres tall and has 48 floors. ({n})"),
            file: PathBuf::from("t.example/tower"),
            line: 5,
        },
        url: "https://t.example/tower".into(),
        final_url: "https://t.example/tower".into(),
        title: "Harbor Tower".into(),
        fetched_at: "2026-10-26T12:00:00Z".into(),
        sha256: "0".repeat(64),
        published: Some("2026-10-01".into()),
    };
    let mut state = State {
        request: BOTH.into(),
        plan: serde_json::from_str::<Plan>(PLAN).unwrap(),
        ..State::default()
    };
    for r in 1..=rounds {
        state.ledger.push(fact(r));
        state.observations.push(Observation {
            queries: vec![format!("query {r}")],
            pages: vec!["https://t.example/tower".into()],
            failed: 0,
            failed_searches: Vec::new(),
            first: r,
            last: r,
        });
    }
    state
}

#[test]
fn the_planner_sees_the_report_and_the_newest_round_only() {
    let mut state = state(3);
    state.report = "Known: the tower is 212 metres tall [1]. Missing: the ferry.".into();
    state.outline = vec![Planned {
        title: "Tower".into(),
        facts: vec![1, 2],
    }];
    let prompt = planner_prompt(&state, usize::MAX, &str::len);
    assert!(prompt.contains(&state.report), "{prompt}");
    assert!(prompt.contains("- Tower: facts 1, 2\n"), "{prompt}");
    assert!(
        prompt.contains("Round 3: searched \"query 3\", 0 failed; read 1 new pages, 0 failed."),
        "{prompt}"
    );
    assert!(prompt.contains("(3)\""), "{prompt}");
    // Each fact shows its page's date (R6).
    assert!(prompt.contains("published 2026-10-01"), "{prompt}");
    // Rounds 1 and 2 live in the report: the prompt does not grow with the rounds.
    for n in [1, 2] {
        assert!(!prompt.contains(&format!("({n})\"")), "{prompt}");
        assert!(!prompt.contains(&format!("query {n}")), "{prompt}");
    }
}

#[test]
fn the_writers_use_the_outline_the_rounds_grew() {
    // The template fails the outline step on this request, and a writer without the outline.
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, GROWN);
    let (sections, sources, ..) = report(&result);
    let titles: Vec<&str> = sections.iter().map(|(t, _)| t.as_str()).collect();
    assert_eq!(titles, ["Height", "Sundays"]);
    assert_eq!(sections[0].1, TOWER_SECTION);
    assert_eq!(sources, [1, 2]);
    assert!(steps(&run, "outline").is_empty());
}

#[test]
fn the_editor_sees_only_the_joins_and_cannot_add_a_source() {
    // The template fails an editor that sees the tower section's first paragraph.
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, JOINED);
    let (sections, sources, ..) = report(&result);
    let (tower, ferry) = (&sections[0].1, &sections[1].1);
    assert_eq!(
        tower,
        "OPENING-PARAGRAPH: a tall tower.\n\nAll told, the Harbor Tower is 212 metres tall [1]."
    );
    // The start that cites the tower's fact too is not kept: the ferry reads as written.
    assert_eq!(
        ferry,
        "The ferry leaves on Sundays at 9am, 1pm and 5pm [2], unlike the tower."
    );
    assert_eq!(sources, [1, 2]);
    // One editor session for the one join.
    assert_eq!(steps(&run, "edit").len(), 1);
}

#[test]
fn writers_that_end_past_the_editors_time_get_no_editor_turn() {
    // JOINED's outline takes 2 s of the 4 s writing time and its tower writer 2.5 s of its own
    // 4 s: the writers end at least 4.5 s in, past the editor's end at 4 s, with both sections
    // written. The two tower sections' join, which `the_editor_sees_only_the_joins_and_cannot_add_
    // a_source` edits, is not edited, and with the prefix cache on (the anchor first) no editor
    // turn reaches the engine, which fails any (PR 580's review).
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    // OUTLINE_SHARED's token (reply 31) and TOWER_TWO's (reply 18).
    let slow = vec![
        (258 + 2 * 31, Duration::from_secs(2)),
        (258 + 2 * 18, Duration::from_millis(2500)),
    ];
    let skill = Research {
        write_wall: Duration::from_secs(4),
        ..two()
    };
    let workers = Workers {
        max_workers: 3,
        prefix_cache: true,
    };
    let (result, run) = (skill.run_with(
        &scripted_in(None, slow),
        &ctx(tmp.path()),
        workers,
        JOINED_LATE,
        &tools,
        Seen::default(),
        &AtomicBool::default(),
    ))
    .unwrap();
    let (sections, ..) = report(&result);
    assert_eq!(
        sections.iter().map(|s| &s.1).collect::<Vec<_>>(),
        [
            TOWER_TWO,
            "The ferry leaves on Sundays at 9am, 1pm and 5pm [2], unlike the tower."
        ],
        "{:?}",
        run.steps
    );
    assert!(steps(&run, "edit").is_empty(), "{:?}", run.steps);
    assert!(steps(&run, "edits").is_empty(), "{:?}", run.steps);
}

#[test]
fn sections_whose_titles_share_no_word_are_joined_as_written() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, BOTH);
    let (sections, ..) = report(&result);
    assert_eq!(sections[0], ("Tower".into(), TOWER_SECTION.into()));
    assert!(steps(&run, "edit").is_empty());
    let edits = steps(&run, "edits");
    assert!(edits.is_empty(), "{edits:?}");
}

#[test]
fn an_edit_that_drops_a_citation_leaves_its_paragraph_as_written() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, _) = research(two(), tmp.path(), &tools, JOINED_UNCITED);
    let (sections, sources, ..) = report(&result);
    assert_eq!(sections[0].1, TOWER_TWO);
    assert!(
        sections[1]
            .1
            .starts_with("The ferry, unlike the tower, leaves"),
        "{}",
        sections[1].1
    );
    assert_eq!(sources, [1, 2]);
}

#[test]
fn a_quote_its_saved_copy_lacks_is_dropped_and_listed_as_not_verified() {
    let tmp = tempfile::tempdir().unwrap();
    // The tower page's saved copy loses its height before the writing: its quote is fabricated
    // as far as the re-check can tell.
    let (_server, tools) = web(true);
    let (result, run) = research(two(), tmp.path(), &tools, BOTH);
    let (sections, sources, unverified, _) = report(&result);
    assert_eq!((sources, unverified), (vec![2], vec![1]));
    // Its sentence lost its only source, so it goes, and its section with it (R6); the quote,
    // listed below, is why, so it is no sentence the writer left without a source.
    let titles: Vec<&str> = sections.iter().map(|(t, _)| t.as_str()).collect();
    assert_eq!(titles, ["Ferry"]);
    assert!(
        run.markdown
            .contains("Read 2 pages. 1 quote verified. 0 sentences had no source."),
        "{}",
        run.markdown
    );
    assert!(
        run.markdown.contains(
            "## Could not verify\n\n- \"It is 212 metres tall and has 48 floors.\": the saved copy of"
        ),
        "{}",
        run.markdown
    );
    assert!(
        run.markdown.contains("does not show it"),
        "{}",
        run.markdown
    );
}

#[test]
fn a_sentence_with_a_number_and_no_source_is_fixed_or_removed_and_counted() {
    // The template fails a retry that does not name its sentence.
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, UNSOURCED);
    let (sections, ..) = report(&result);
    assert_eq!(
        sections,
        [
            ("Tower".into(), TOWER_SECTION.into()),
            (
                "Ferry".into(),
                "The ferry leaves on Sundays at 9am, 1pm and 5pm [2].".into()
            ),
        ]
    );
    let TypedResult::Report { fixed, removed, .. } = result else {
        panic!("not a report")
    };
    assert_eq!((fixed, removed), (1, 1));
    assert!(
        run.markdown
            .contains("2 sentences had no source: 1 fixed, 1 removed."),
        "{}",
        run.markdown
    );
}

#[test]
fn a_report_in_unicode_text_is_checked_and_kept() {
    // A review of #359: the sentence check sliced a character in two.
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, UNICODE);
    let (sections, ..) = report(&result);
    assert_eq!(sections[0], ("Tower".into(), TOWER_UNICODE.into()));
    let TypedResult::Report { fixed, removed, .. } = &result else {
        panic!("not a report")
    };
    assert_eq!((fixed, removed), (&0, &0));
    assert!(run.markdown.contains(TOWER_UNICODE), "{}", run.markdown);
}

#[test]
fn a_sentence_cut_for_citing_another_sections_fact_is_counted_as_removed() {
    // own() takes the ferry section's [1], so its number sentence has no source and goes; the
    // count says so (second review, Minor 4).
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, ACROSS);
    let (sections, ..) = report(&result);
    assert_eq!(
        sections[1],
        (
            "Ferry".into(),
            "The ferry leaves on Sundays at 9am, 1pm and 5pm [2].".into()
        )
    );
    let TypedResult::Report { fixed, removed, .. } = &result else {
        panic!("not a report")
    };
    assert_eq!((fixed, removed), (&0, &1));
    assert!(
        run.markdown
            .contains("1 sentence had no source: 0 fixed, 1 removed."),
        "{}",
        run.markdown
    );
}

#[test]
fn a_sentence_whose_quote_fails_the_recheck_is_not_counted_as_one_without_a_source() {
    // The tower's writer cuts its date on the retry and the ferry's keeps its price; then the
    // tower's quote fails the re-check, which takes its sentence but is no R6 count.
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(true);
    let (result, run) = research(two(), tmp.path(), &tools, UNSOURCED);
    let TypedResult::Report { fixed, removed, .. } = &result else {
        panic!("not a report")
    };
    assert_eq!((fixed, removed), (&1, &1));
    assert!(
        run.markdown
            .contains("2 sentences had no source: 1 fixed, 1 removed."),
        "{}",
        run.markdown
    );
    let titles: Vec<String> = report(&result).0.into_iter().map(|(t, _)| t).collect();
    assert_eq!(titles, ["Ferry"]);
}

#[test]
fn a_writer_whose_retry_fails_keeps_its_first_answer_less_its_sentence_without_a_source() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, RETRIED_IN_VAIN);
    let (sections, _, _, stopped) = report(&result);
    assert_eq!(sections[0], ("Tower".into(), TOWER_SECTION.into()));
    assert!(!stopped.contains("not written"), "{stopped}");
    let TypedResult::Report { fixed, removed, .. } = &result else {
        panic!("not a report")
    };
    assert_eq!((fixed, removed), (&0, &1));
    // The audit log keeps the retry's error.
    assert!(steps(&run, "sections")[0].error.is_some());
}

#[test]
fn the_last_rounds_facts_are_outlined_when_the_planner_gives_no_plan() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, _) = research(two(), tmp.path(), &tools, LOST);
    let (sections, sources, _, stopped) = report(&result);
    // The last reflect step runs after the budget too: it is the run's only outline step.
    assert_eq!(stopped, "the round budget (2 rounds) was spent");
    let titles: Vec<_> = sections.iter().map(|(t, _)| t.as_str()).collect();
    // Round 2's ferry fact is in no outline the planner gave: the outline step files it.
    assert_eq!(titles, ["Tower", "Ferry"]);
    assert_eq!(sources, [1, 2]);
}

#[test]
fn a_last_reflection_without_an_outline_still_outlines_the_newest_facts() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    // Round 2's reflection parses, stops, and gives no outline: round 1's outline lacks the ferry.
    let (result, run) = research(two(), tmp.path(), &tools, NO_OUTLINE);
    let (sections, sources, _, stopped) = report(&result);
    assert_eq!(
        stopped,
        "the planner stopped: both sub-questions are answered"
    );
    let titles: Vec<_> = sections.iter().map(|(t, _)| t.as_str()).collect();
    assert_eq!(titles, ["Tower", "Ferry"]);
    assert_eq!(sources, [1, 2]);
    assert_eq!(steps(&run, "outline").len(), 1);
}

#[test]
fn a_round_past_the_wall_time_reads_no_page() {
    let tmp = tempfile::tempdir().unwrap();
    let (server, mut tools) = web(false);
    let search = [(
        "harbor tower height".to_string(),
        vec![server.url("/tower")],
    )];
    let slow = Search(search.into_iter().collect(), false);
    tools.replace(Box::new(Slow(slow, Duration::from_secs(3))));
    let skill = Research {
        wall: Duration::from_secs(2),
        ..two()
    };
    let (result, run) = research(skill, tmp.path(), &tools, BOTH);
    let (.., stopped) = report(&result);
    assert_eq!(stopped, "the time budget (2 s) was spent");
    assert_eq!(server.hits("/tower"), 0);
    assert!(steps(&run, "fetch").is_empty(), "{:?}", run.steps);
    assert!(steps(&run, "rank").is_empty(), "{:?}", run.steps);
    assert!(steps(&run, "reflect").is_empty(), "{:?}", run.steps);
}

#[test]
fn a_round_without_a_key_runs_at_most_six_queries() {
    let (_server, tools) = web(false);
    let searched = |key: &str| {
        let tmp = tempfile::tempdir().unwrap();
        let agent = tmp.path().join("data/agent");
        std::fs::create_dir_all(&agent).unwrap();
        std::fs::write(agent.join("config.toml"), key).unwrap();
        let (_, run) = research(two(), tmp.path(), &tools, EIGHT);
        let search = steps(&run, "search");
        search[0].detail.as_ref().unwrap()["queries"].clone()
    };
    // The free engines rate-limit a dozen queries at once (a user's Mac); a key lifts the cap. The
    // plan gives two queries a sub-question, and each sub-question keeps one or both.
    assert_eq!(searched(""), json!(["a1", "a2", "a3", "a4", "a5", "a7"]));
    assert_eq!(searched("brave_key = \"k\"\n").as_array().unwrap().len(), 8);
}

#[test]
fn a_reflect_that_fails_still_writes_the_report() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, REFLECT_FAILS);
    let (sections, sources, _, stopped) = report(&result);
    assert!(stopped.starts_with("the planner failed ("), "{stopped}");
    assert!(stopped.contains("the planner failed"), "{stopped}");
    assert!(
        sections.iter().any(|(_, text)| text == TOWER_SECTION),
        "{sections:?}"
    );
    assert_eq!(sources, [1]);
    assert!(run.markdown.contains(TOWER_SECTION), "{}", run.markdown);
}

#[test]
fn one_page_gives_at_most_five_facts() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    // The reflect template fails the run if a sixth fact reaches the planner.
    let (_, run) = research(two(), tmp.path(), &tools, SPECS);
    let reflect = steps(&run, "reflect");
    assert_eq!(reflect.len(), 1, "{:?}", run.steps);
    let found = daily(&tmp.path().join("data"));
    assert!(found.contains(&format!("\"{SPECS}\": 5 facts")), "{found}");
}

/// BOTH's report, saved by `run_saved` under `tmp`: the data folder and its id.
fn saved(tmp: &Path, tools: &Registry) -> (PathBuf, String) {
    let (larch, ctx, stop) = (scripted(), ctx(tmp), AtomicBool::default());
    let report = (two().run_saved(
        &larch,
        &ctx,
        WORKERS,
        BOTH,
        None,
        tools,
        Seen::default(),
        &stop,
        &mut |_| {},
    ))
    .unwrap();
    let id = report.rsplit("report ").next().unwrap();
    (ctx.data_dir.clone(), id.trim_end_matches("._").to_owned())
}

#[test]
fn title_words_open_a_report_saved_in_another_chat() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (data, id) = saved(tmp.path(), &tools);
    assert_eq!(saved_id(&data, &id), Some(id.clone()));
    // Words of its title, in another order and case, as a new chat names it.
    let words = "the Larkspur Ferry and Harbor Tower report";
    assert_eq!(saved_id(&data, words), Some(id));
    // One shared word is not enough, and a title or an id no report has finds none.
    assert_eq!(saved_id(&data, "Harbor report"), None);
    assert_eq!(saved_id(&data, "LLM Transformer Architecture Report"), None);
    assert_eq!(saved_id(&data, "abcdefabcdef"), None);
}

#[test]
fn the_daily_note_names_the_saved_report() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (data, id) = saved(tmp.path(), &tools);
    let note = daily(&data);
    assert!(
        note.contains(&format!("\"{BOTH}\" saved as report {id}")),
        "{note}"
    );
}

/// Every daily note under `data`, one a line.
fn daily(data: &Path) -> String {
    let notes = larch_agent::memory::notes(data).unwrap();
    notes.iter().map(|n| format!("{}\n", n.text)).collect()
}

#[test]
fn a_report_whose_every_sentence_lost_its_source_still_renders_and_says_so() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, run) = research(two(), tmp.path(), &tools, ALL_UNSOURCED);
    let (sections, ..) = report(&result);
    assert!(sections.is_empty(), "{sections:?}");
    for said in [
        "No sentence of the report had a source, so none is shown.",
        "2 sentences had no source: 0 fixed, 2 removed.",
    ] {
        assert!(run.markdown.contains(said), "{}", run.markdown);
    }
    assert!(!run.markdown.contains("## Tower"), "{}", run.markdown);
}

#[test]
fn the_report_ends_with_the_verification_line() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (_, run) = research(two(), tmp.path(), &tools, BOTH);
    let paragraphs: Vec<&str> = run.markdown.split("\n\n").collect();
    let [.., line, stopped] = &paragraphs[..] else {
        panic!("{}", run.markdown)
    };
    assert_eq!(
        *line,
        "Read 2 pages. 2 quotes verified. 0 sentences had no source."
    );
    assert!(stopped.starts_with("_Stopped: "), "{stopped}");
}

#[test]
fn the_sections_are_written_by_parallel_sessions() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (_, run) = research(two(), tmp.path(), &tools, BOTH);
    let step = steps(&run, "sections")[0].detail.clone().unwrap();
    assert_eq!(step["workers"], 2);
    assert_eq!(step["at_once"], 2);
    let sessions: HashSet<_> = steps(&run, "section").iter().map(|e| e.session).collect();
    assert_eq!(sessions.len(), 2);
}

/// A planner that writes the same broken JSON twice gets its retry held to the plan's schema,
/// with thinking off, so the run plans; the log keeps the broken reply's start.
#[test]
fn a_plan_retry_is_held_to_valid_json() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (result, run) = research(
        Research { rounds: 1, ..two() },
        tmp.path(),
        &tools,
        &format!("Malformed plan: {BOTH}"),
    );
    let plans = steps(&run, "plan");
    assert_eq!(plans.len(), 2);
    assert!(
        (plans[0].error.as_deref()).is_some_and(|e| e.starts_with("key must be a string")),
        "{:?}",
        plans[0].error
    );
    assert_eq!(plans[0].detail, Some(json!({ "reply": MALFORMED })));
    assert_eq!(plans[1].error, None);
    assert_eq!(report(&result).1, [1]);
}

#[test]
fn a_plan_that_is_not_json_is_asked_once_more_then_given_up() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    // The first reply is no plan; the second, in the same session, is.
    let (result, run) = research(
        Research { rounds: 1, ..two() },
        tmp.path(),
        &tools,
        &format!("Bad plan: {BOTH}"),
    );
    let plans = steps(&run, "plan");
    assert_eq!(plans.len(), 2);
    assert!(plans[0].error.is_some() && plans[1].error.is_none());
    assert_eq!(report(&result).1, [1]);
    // Twice no plan: the run says so and reads nothing.
    let (result, run) = research(two(), tmp.path(), &tools, "Never a plan");
    assert_eq!(report(&result).3, "the planner gave no plan");
    assert_eq!(steps(&run, "search").len(), 0);
    assert!(run.markdown.contains("The searches found no facts"));
}

#[test]
fn when_the_writers_fail_the_ledger_is_the_report() {
    let tmp = tempfile::tempdir().unwrap();
    let (server, tools) = web(false);
    let request = format!("{TOWER_ONLY}, {WRITERS_FAIL}");
    let (result, run) = research(two(), tmp.path(), &tools, &request);
    let (sections, sources, _, stopped) = report(&result);
    assert_eq!(sections.len(), 1);
    let (title, text) = &sections[0];
    assert!(title.contains("not written"), "{title}");
    assert_eq!(
        text,
        &format!(
            "- [1] The tower is 212 metres tall: \"It is 212 metres tall and has 48 floors.\" ({})",
            server.url("/tower")
        )
    );
    // Re-checked like a written report's facts, so each cited fact is a source.
    assert_eq!(sources, [1]);
    assert!(
        stopped.ends_with("; the report was not written (invalid request 7)"),
        "{stopped}"
    );
    assert!(run.markdown.contains(text.as_str()), "{}", run.markdown);
    assert!(steps(&run, "sections")[0].error.is_some());
}

#[test]
fn when_the_writers_fail_a_fact_the_saved_page_no_longer_shows_is_not_cited() {
    let tmp = tempfile::tempdir().unwrap();
    // The ferry's search changes the tower page's saved copy, as in the test above.
    let (_server, tools) = web(true);
    let request = format!("{BOTH}, {WRITERS_FAIL}");
    let (result, run) = research(two(), tmp.path(), &tools, &request);
    let (sections, sources, unverified, _) = report(&result);
    assert_eq!((sources, unverified), (vec![2], vec![1]));
    let text = &sections[0].1;
    assert!(
        text.starts_with("- [2] ") && !text.contains("[1]"),
        "{text}"
    );
    assert!(
        run.markdown.contains("## Could not verify"),
        "{}",
        run.markdown
    );
}

#[test]
fn when_the_writers_write_nothing_the_ledger_is_the_report() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let request = format!("{TOWER_ONLY}, {WRITERS_BLANK}");
    let (result, _) = research(two(), tmp.path(), &tools, &request);
    let (sections, _, _, stopped) = report(&result);
    assert!(sections[0].0.contains("not written"), "{sections:?}");
    assert!(
        stopped.ends_with("; the report was not written (the writers wrote nothing)"),
        "{stopped}"
    );
}

/// A `web_search` that sets `stop` when asked `at`, as a user who stops the run then would.
struct StopAt(Search, &'static str, Arc<AtomicBool>);

impl Tool for StopAt {
    fn spec(&self) -> ToolSpec {
        self.0.spec()
    }

    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        if args["query"] == self.1 {
            self.2.store(true, Ordering::Relaxed);
        }
        self.0.call(args, ctx)
    }
}

#[test]
fn a_run_the_user_stops_before_the_writing_is_an_error_not_a_report() {
    let tmp = tempfile::tempdir().unwrap();
    let server = Server::start(&[("/tower", TOWER_PAGE)]);
    let search = ["harbor tower height", "harbor tower again"]
        .map(|q| (q.to_string(), vec![server.url("/tower")]));
    let stop = Arc::new(AtomicBool::new(false));
    // The second round finds only the page it has, so the run goes on to the writing, stopped.
    let mut tools = Registry::builtin();
    tools.replace(Box::new(StopAt(
        Search(search.into_iter().collect(), false),
        "harbor tower again",
        Arc::clone(&stop),
    )));
    let workers = Workers {
        max_workers: 3,
        prefix_cache: false,
    };
    let ran = two().run_with(
        &scripted(),
        &ctx(tmp.path()),
        workers,
        TOWER_ONLY,
        &tools,
        Seen::default(),
        &stop,
    );
    assert!(
        matches!(ran, Err(SkillError::Engine(EngineError::Cancelled))),
        "{:?}",
        ran.map(|(r, _)| r)
    );
}

/// A `web_search` whose provider is down: it answers in plain text.
struct Down;

impl Tool for Down {
    fn spec(&self) -> ToolSpec {
        spec("web_search")
    }

    fn call(&self, _: &Value, _: &ToolCtx) -> Result<String, ToolError> {
        Ok("search provider unavailable".into())
    }
}

#[test]
fn every_search_failing_is_the_named_stop() {
    let tmp = tempfile::tempdir().unwrap();
    let (server, mut tools) = web(false);
    tools.replace(Box::new(Down));
    let (result, run) = research(two(), tmp.path(), &tools, BOTH);
    assert_eq!(
        report(&result).3,
        "every search failed (harbor tower height: search provider unavailable)"
    );
    assert_eq!(server.hits("/tower"), 0);
    assert!(run.markdown.contains("The searches found no facts"));
    // The daily note says how it went, not the web's words.
    let notes = larch_agent::memory::notes(&tmp.path().join("data")).unwrap();
    assert_eq!(notes.len(), 1);
    assert!(
        notes[0]
            .text
            .ends_with("0 facts, 0 sources, 0 could not verify"),
        "{notes:?}"
    );
}

#[test]
fn the_chat_tool_says_when_to_pick_it_over_web_answer() {
    let tools = Registry::builtin();
    let spec = tools.get("research").unwrap().spec();
    assert_eq!(spec.permission, Permission::Read);
    assert!(spec.untrusted_results);
    assert!(
        spec.description.contains("web_answer"),
        "{}",
        spec.description
    );
    assert_eq!(
        spec.schema["properties"]["effort"]["enum"],
        json!(["standard", "deep"])
    );
    assert!(larch_agent::CHAT_TOOLS.contains(&"research"));
}

#[test]
fn the_cli_takes_iris_deep_and_run_research_with_the_model_flags() {
    for args in [
        &["iris", "--deep", "Compare the ferries"][..],
        &["run", "research", "--question", "Compare the ferries"],
        &["run", "research", "--json", "--question", "Ferries"],
        &[
            "run",
            "research",
            "--effort",
            "deep",
            "--question",
            "Ferries",
        ],
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_larch-agent"))
            .args(args)
            .args(["-m", "missing", "--cpu", "--data-dir"])
            .arg(tmp.path())
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(1), "{args:?}: {stderr}");
        assert!(stderr.contains("missing.gguf"), "{args:?}: {stderr}");
    }
}

/// The research bench's folder.
fn bench_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/agent/research")
}

fn brief_rows() -> Vec<research::BriefRow> {
    research::brief_rows(&bench_dir().join("briefs.jsonl")).unwrap()
}

fn matches(pattern: &str, text: &str) -> bool {
    let re = regex::RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build();
    re.unwrap().is_match(text)
}

#[test]
fn every_brief_row_parses_and_its_pages_load() {
    let rows = brief_rows();
    assert!(rows.len() >= 4);
    for row in rows {
        for p in (row.brief.iter())
            .chain(&row.brief_not)
            .chain(&row.report_not)
        {
            matches(p, "");
        }
        let (hits, pages) = larch_agent::bench::saved(&bench_dir().join(&row.pages)).unwrap();
        assert_eq!(hits.len(), pages.len(), "{}", row.id);
        assert!(!pages.is_empty(), "{}", row.id);
    }
}

/// Every query's hits are the saved search's, each at the test server's copy of its page; the
/// queries searched are kept.
struct AllHits(Vec<Value>, Arc<Mutex<Vec<String>>>);

impl Tool for AllHits {
    fn spec(&self) -> ToolSpec {
        spec("web_search")
    }

    fn call(&self, args: &Value, _: &ToolCtx) -> Result<String, ToolError> {
        let query = args["query"].as_str().unwrap().to_string();
        self.1.lock().unwrap().push(query);
        Ok(json!(self.0).to_string())
    }
}

/// v3 wrote about a 2026 IS 500 and a 2026 M3 for a user's "Lexus ISF vs BMW M3". The brief names
/// the generations; the query that dropped the "F" gets the IS F's name; the pages about the IS
/// 500 and the 2026 M3 write no spelling of the IS F, so they are never read, and the report
/// names neither.
#[test]
fn the_lexus_row_locks_the_subject_and_its_brief_names_the_generations() {
    let row = brief_rows()
        .into_iter()
        .find(|r| r.id == "lexus-isf-m3")
        .unwrap();
    let (hits, pages) = larch_agent::bench::saved(&bench_dir().join(&row.pages)).unwrap();
    let texts: Vec<(&'static str, &'static str)> = (pages.iter().enumerate())
        .map(|(i, p)| (&*format!("/{}", i + 1).leak(), &*p.text.clone().leak()))
        .collect();
    let server = Server::start(&texts);
    let hits: Vec<Value> = (hits.iter().enumerate())
        .map(|(i, h)| json!({"title": h.title, "url": server.url(&format!("/{}", i + 1)), "snippet": h.snippet}))
        .collect();
    let searched = Arc::new(Mutex::new(Vec::new()));
    let mut tools = Registry::builtin();
    tools.replace(Box::new(AllHits(hits, Arc::clone(&searched))));
    let tmp = tempfile::tempdir().unwrap();
    let mut shown = Vec::new();
    let report = (Research::default().run_saved(
        &scripted(),
        &ctx(tmp.path()),
        WORKERS,
        &row.request,
        None,
        &tools,
        Seen::default(),
        &AtomicBool::default(),
        &mut |line| shown.push(line.to_string()),
    ))
    .unwrap();
    assert_eq!(shown[0], "Planning research on: Lexus ISF vs BMW M3\n\n");
    let brief = &shown[1];
    for p in &row.brief {
        assert!(matches(p, brief), "{p} misses the brief: {brief}");
    }
    // "IS-F" is "ISF" with a hyphen: one spelling.
    assert!(brief.contains("Lexus IS F (also ISF)\n"), "{brief}");
    assert!(
        brief.ends_with("Stop me and say more if this is not what you meant.\n\n"),
        "{brief}"
    );
    assert_eq!(
        shown[2..],
        [
            "Round 1: 2 searches\n\n",
            "Reading 2 pages\n\n",
            "Writing 1 section\n\n"
        ]
    );
    // The progress lines are shown while the run goes; the reply keeps the plan and the brief.
    assert!(report.starts_with(&shown[..2].concat()), "{report}");
    assert!(!report.contains("Round 1"), "{report}");
    // The queries run on threads, so their order is not fixed: compare them sorted.
    let mut queries = searched.lock().unwrap().clone();
    queries.sort();
    assert_eq!(
        queries,
        [
            "Lexus IS vs BMW M3 Lexus IS F",
            "Lexus ISF vs BMW M3",
            "lexus isf e92 m3 comparison"
        ]
    );
    // Each page is fetched once; the IS 500 page names "Lexus" but no spelling of the IS F, and
    // the 2026 M3 CS page is dated outside the assumed years, so no reader gets either.
    let hits = |p: &str| server.hits(p);
    assert_eq!(
        (hits("/1"), hits("/2"), hits("/3"), hits("/4")),
        (1, 1, 1, 1)
    );
    assert!(
        report.contains("Read 2 pages, dropped 2 pages as off-subject."),
        "{report}"
    );
    let body = &report[shown[..2].concat().len()..];
    for p in &row.report_not {
        assert!(!matches(p, body), "{p} is in the report: {body}");
    }
    assert!(body.contains(LEXUS_SECTION), "{body}");
    assert_eq!(row.off_subject, Some(2));
}

/// The Mac scores briefs.jsonl on the real model with `larch-agent run research-briefs`; here the
/// scripted model passes the Lexus row and fails it once the row wants what its brief lacks.
#[test]
fn the_brief_scorer_says_pass_or_fail_a_row() {
    let tmp = tempfile::tempdir().unwrap();
    let mut row: Value = (std::fs::read_to_string(bench_dir().join("briefs.jsonl")).unwrap())
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|r| r["id"] == "lexus-isf-m3")
        .unwrap();
    row["pages"] = json!(bench_dir().join("briefs/lexus-isf-m3"));
    let score = |row: &Value| {
        let file = tmp.path().join("briefs.jsonl");
        std::fs::write(&file, format!("{row}\n")).unwrap();
        let mut said = Vec::new();
        let failed = (research::score_briefs(
            &Research::default(),
            &scripted(),
            WORKERS,
            &file,
            &mut |line| said.push(line),
        ))
        .unwrap();
        (failed, said)
    };
    assert_eq!(score(&row), (0, vec!["pass lexus-isf-m3".to_string()]));
    row["brief_not"] = json!(["(?m)^About:"]);
    row["report_not"] = json!(["416 hp"]);
    assert_eq!(
        score(&row),
        (
            1,
            vec!["FAIL lexus-isf-m3: brief has (?m)^About:; report has 416 hp".to_string()]
        )
    );
}

/// One subject on a page and one on none: the report is written from the pages there are, and
/// its note says which subject no page named.
/// Only pages read count against the page budget: with room for two, the round fetches three,
/// the lock drops the IS 500 page, and two are still read.
#[test]
fn a_page_the_lock_drops_leaves_its_slot_to_another() {
    let row = brief_rows()
        .into_iter()
        .find(|r| r.id == "lexus-isf-m3")
        .unwrap();
    let (hits, pages) = larch_agent::bench::saved(&bench_dir().join(&row.pages)).unwrap();
    let texts: Vec<(&'static str, &'static str)> = (pages.iter().enumerate())
        .map(|(i, p)| (&*format!("/{}", i + 1).leak(), &*p.text.clone().leak()))
        .collect();
    let server = Server::start(&texts);
    let hits: Vec<Value> = (hits.iter().enumerate())
        .map(|(i, h)| json!({"title": h.title, "url": server.url(&format!("/{}", i + 1)), "snippet": h.snippet}))
        .collect();
    let mut tools = Registry::builtin();
    tools.replace(Box::new(AllHits(hits, Arc::default())));
    let tmp = tempfile::tempdir().unwrap();
    let mut shown = Vec::new();
    let report = (Research {
        pages: 2,
        ..Research::default()
    }
    .run_saved(
        &scripted(),
        &ctx(tmp.path()),
        WORKERS,
        &row.request,
        None,
        &tools,
        Seen::default(),
        &AtomicBool::default(),
        &mut |line| shown.push(line.to_string()),
    ))
    .unwrap();
    let fetched: usize = (1..=4).map(|i| server.hits(&format!("/{i}"))).sum();
    assert_eq!(fetched, 3, "{report}");
    assert!(
        shown.contains(&"Reading 2 pages\n\n".to_string()),
        "{shown:?}"
    );
    assert!(
        report.contains("Read 2 pages, dropped 1 page as off-subject"),
        "{report}"
    );
}

#[test]
fn a_subject_no_page_names_is_a_note_under_a_report_on_the_rest() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let said = (two().run_saved(
        &scripted(),
        &ctx(tmp.path()),
        WORKERS,
        TOWER_PIER,
        None,
        &tools,
        Seen::default(),
        &AtomicBool::default(),
        &mut |_| {},
    ))
    .unwrap();
    assert!(said.contains("212 metres"), "{said}");
    assert!(said.contains(" No page named Larkspur pier."), "{said}");
    assert!(!said.contains("Research failed"), "{said}");
}

#[test]
fn a_subject_no_page_writes_is_a_failure_in_one_line_not_a_report() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let mut shown = String::new();
    let said = (two().run_saved(
        &scripted(),
        &ctx(tmp.path()),
        WORKERS,
        "Report on the Larkspur pier",
        None,
        &tools,
        Seen::default(),
        &AtomicBool::default(),
        &mut |line| shown.push_str(line),
    ))
    .unwrap();
    assert!(shown.contains("Round 1: 1 search\n\n"), "{shown}");
    // The reply keeps the plan's line and the brief, not the run's passing steps.
    let shown = shown.replace("Round 1: 1 search\n\n", "");
    let failed = "Research failed: no page of the 1 page fetched names Larkspur pier, so there \
                  is no report.";
    assert_eq!(said, format!("{shown}{failed}"));
    assert!(shown.contains("About:\n- Larkspur pier\n"), "{shown}");
    // Nothing was read, and nothing is saved as a report.
    assert!(!shown.contains("Reading"), "{shown}");
    assert!(!tmp.path().join("data/agent/reports").exists());
}

/// A round's facts that do not fit the planner's context are left out of its prompt, not the run:
/// on an engine of 2,500 tokens, whose reflect prompt with all five long facts is over 3,000 (the
/// template renders it), the run still reaches round 2 (the Mac's deep run stopped at 12,545 of
/// 8,192).
#[test]
fn a_planner_prompt_over_the_context_drops_facts_and_the_run_goes_on() {
    let tmp = tempfile::tempdir().unwrap();
    let (_server, tools) = web(false);
    let (_, run) = (two().run_with(
        &scripted_in(Some(2500), Vec::new()),
        &ctx(tmp.path()),
        WORKERS,
        LONG,
        &tools,
        Seen::default(),
        &AtomicBool::default(),
    ))
    .unwrap();
    let errors: Vec<_> = (run.steps.iter()).filter_map(|e| e.error.clone()).collect();
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(steps(&run, "search").len(), 2);
    assert_eq!(steps(&run, "reflect").len(), 2);
}

#[test]
fn the_planner_prompt_keeps_the_facts_that_fit_and_counts_the_rest() {
    let state = state(1);
    let all = planner_prompt(&state, usize::MAX, &str::len);
    let fitted = planner_prompt(&state, all.len() - 1, &str::len);
    assert!(fitted.len() < all.len(), "{fitted}");
    assert!(
        fitted.ends_with("(1 more fact not shown here; the writers have them.)\n"),
        "{fitted}"
    );
}
