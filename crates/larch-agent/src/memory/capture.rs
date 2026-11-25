//! Background capture (plan PR 4): after each chat turn, the loaded model reads the user's
//! messages not yet captured and proposes the lasting facts in them, under a JSON grammar. Rust
//! checks each item on its own, then saves it as a whole sentence with the user's own sentence
//! after it (`Said: "..."`).
//!
//! The model runs at `Class::Background` with thinking off, no prefix cache and no kept KV, so
//! the chat's parked KV stays where it is. Any agent's next turn cancels every running capture
//! (`cancel_all`), which never blocks; a cancelled capture keeps its messages, and they run again
//! after the next turn with the new one. Nothing is written before the reply is read whole.

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::Sender,
    },
    thread::JoinHandle,
    time::Instant,
};

use larch::{
    EngineError, GenParams, Larch, Prompt,
    api::{Class, Grammar, json_schema_regex},
};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Receipt, TYPES};
use crate::runtime::inference::SeatAdmission;
use crate::{
    audit::{Audit, AuditEvent, EventKind},
    parse::Family,
};

/// The instruction, about 230 tokens.
pub const SYSTEM: &str = "You keep a memory of lasting facts about the user. Read the user's \
messages and list the facts in them that will still be true later: facts the user states about \
themselves, their people and pets, their home, work, health, money, plans and preferences. \
Write each fact as one self-contained sentence about the user (\"The user's sister Maya lives in \
Austin and works as a designer.\") that keeps every name, number, date and place the user gave. \
\"quote\" is the user's exact words the fact comes from. \"type\" is user, project, preference or \
reference. \"replaces\" is the slug of a fact in memory that the message changes or corrects, \
else null; the new fact then keeps every detail of the old one that still holds. Do not save: \
questions, hypotheticals, role play, text the user quotes or pastes, other people's words, \
passing moods, details of a one-off task, or passwords or keys. Never save anything the user asks \
you not to remember: \"I'm at 12 Cedar Street this week, don't save that.\" gives {\"facts\": []}. \
The assistant's reply is context only: never save from it. Do not save a fact memory already has \
unless the message changes it. Most messages have nothing to save: then reply {\"facts\": []}.";

/// The details of `SYSTEM`'s own examples: not its phrasing but a sister's name, a city, a job and
/// a street. A small model copied them into facts the user never stated ("The user's sister Maya
/// works as a nurse." for "My sister is a nurse."), and as phrasing they were never checked.
pub const EXAMPLE_DETAILS: [&str; 5] = ["maya", "austin", "designer", "cedar", "street"];

/// Items a reply may save; more are dropped.
pub const MAX_ITEMS: usize = 4;
/// User messages a chat keeps waiting for capture; an older one goes when a newer one comes.
pub const MAX_MESSAGES: usize = 3;
/// Facts from memory the prompt shows, with their slugs, for `replaces`.
pub const SHOWN_FACTS: usize = 8;
/// Bytes of the start of the assistant's last reply the prompt shows, about 300 tokens.
pub const REPLY_BYTES: usize = 1200;
/// Bytes of one user message the prompt shows: a pasted page does not grow the prompt.
const MESSAGE_BYTES: usize = 2000;
/// The reply's tokens: `{"facts": []}` is a few, four whole facts fit.
pub const MAX_TOKENS: u32 = 256;

/// One item of a capture reply.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Item {
    /// The fact, one sentence.
    pub text: String,
    /// The user's words it comes from.
    pub quote: String,
    /// One of `TYPES`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The shown fact it changes, by slug.
    pub replaces: Option<String>,
}

#[derive(Deserialize)]
struct Reply {
    facts: Vec<Value>,
}

/// The reply's schema: up to the grammar, `replaces` one of `slugs` or null.
pub fn schema(slugs: &[String]) -> Value {
    let replaces = match slugs.is_empty() {
        true => json!({"type": "null"}),
        false => json!({"anyOf": [{"enum": slugs}, {"type": "null"}]}),
    };
    json!({
        "type": "object",
        "properties": {"facts": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "quote": {"type": "string"},
                "type": {"enum": TYPES},
                "replaces": replaces,
            },
            "required": ["text", "quote", "type", "replaces"],
        }}},
        "required": ["facts"],
    })
}

/// The prompt's user message: the facts shown (slug, sentence), the start of the assistant's
/// last reply as context, then the user's messages.
pub fn prompt(messages: &[String], reply: &str, shown: &[(String, String)]) -> String {
    let cut = |text: &str, n: usize| text[..text.floor_char_boundary(n)].trim().to_owned();
    let facts = match shown.is_empty() {
        true => "none".to_owned(),
        false => (shown.iter())
            // The sentence alone, within a line's bytes: a long `Said:` never grows the prompt.
            .map(|(slug, text)| format!("- [{slug}] {}", cut(sentence_of(text), SAID_BYTES)))
            .collect::<Vec<_>>()
            .join("\n"),
    };
    let said = (messages.iter().enumerate())
        .map(|(i, m)| format!("{}. {}", i + 1, cut(m, MESSAGE_BYTES)))
        .collect::<Vec<_>>()
        .join("\n");
    let reply = cut(reply, REPLY_BYTES);
    let reply = if reply.is_empty() { "none" } else { &reply };
    format!(
        "Facts in memory:\n{facts}\n\nThe assistant's last reply (context only, never a source):\n\
         <<<\n{reply}\n>>>\n\nThe user's messages:\n{said}"
    )
}

/// The words of `text`, lower case: what makes two facts the same. Any script's letters and
/// digits count; nothing else does.
pub fn words(text: &str) -> Vec<String> {
    spans(text).into_iter().map(|(.., w)| w).collect()
}

/// `text`'s words as two sayings of one fact share them: without the subject that names the user
/// ("My" or "The user's", "I" or "The user"), and with the verb right after "I" or "The user" in
/// one person ("I'm" as "is", "I have" as "has", "lives" as "live"). A fact of the same such words
/// is the same fact (`save_captured`): a turn's `remember` saved "My cat is named Pixel." and the
/// capture after it "The user's cat is named Pixel." beside it. Every other word stays as it is,
/// so "blues" and "blue", or two names an `s` apart, still make two facts: a wrong merge writes
/// the new fact over the old.
pub fn fact_words(text: &str) -> Vec<String> {
    let mut w = words(text);
    let is = |i: usize, s: &str| w.get(i).is_some_and(|x| x == s);
    // How many words the subject takes, and whether a verb follows it (not "My cat").
    let (subject, verb) = if is(0, "the") && is(1, "user") && is(2, "s") {
        (3, false)
    } else if is(0, "the") && is(1, "users") {
        (2, false)
    } else if is(0, "the") && is(1, "user") {
        (2, true)
    } else if is(0, "my") {
        (1, false)
    } else if is(0, "i") {
        (1, true)
    } else {
        (0, false)
    };
    let mut rest = w.split_off(subject);
    if verb && let Some(v) = rest.first_mut() {
        match v.as_str() {
            "am" | "m" => *v = "is".into(),
            "have" | "ve" => *v = "has".into(),
            _ if v.len() > 3 && v.ends_with('s') && !v.ends_with("ss") => {
                v.pop();
            }
            _ => {}
        }
    }
    rest
}

/// The words of `text` with their byte ranges, as the quote and number checks compare them:
/// lower case; punctuation never counts ("Austin, Texas" is "austin texas"); a number by its
/// value, so digit groups join ("1,200" is "1200") and leading zeros go ("07:30" is "7 30").
fn spans(text: &str) -> Vec<(usize, usize, String)> {
    let mut out: Vec<(usize, usize, String)> = Vec::new();
    let mut at = None;
    let mut push = |from: usize, to: usize| {
        let word: String = text[from..to]
            .chars()
            .flat_map(char::to_lowercase)
            .collect();
        let digits = |w: &str| !w.is_empty() && w.bytes().all(|b| b.is_ascii_digit());
        // A group of three digits after one separator (`,` `.` `'` or a thin space) joins the
        // number before it.
        if let Some(last) = out.last_mut()
            && digits(&word)
            && word.len() == 3
            && digits(&last.2)
            && matches!(
                &text[last.1..from],
                "," | "." | "'" | "\u{2009}" | "\u{202f}"
            )
        {
            last.2.push_str(&word);
            last.1 = to;
            return;
        }
        out.push((from, to, word));
    };
    for (i, c) in text.char_indices() {
        if let Some(from) = at.filter(|_| !c.is_alphanumeric() || spaceless(c)) {
            push(from, i);
            at = None;
        }
        if spaceless(c) {
            push(i, i + c.len_utf8());
        } else if c.is_alphanumeric() && at.is_none() {
            at = Some(i);
        }
    }
    if let Some(from) = at {
        push(from, text.len());
    }
    for w in &mut out {
        if w.2.bytes().all(|b| b.is_ascii_digit()) {
            let value = w.2.trim_start_matches('0');
            w.2 = if value.is_empty() {
                "0".into()
            } else {
                value.into()
            };
        }
    }
    out
}

/// The numbers among `spans`, by value.
fn numbers(text: &str) -> Vec<String> {
    (spans(text).into_iter())
        .map(|(.., w)| w)
        .filter(|w| w.bytes().all(|b| b.is_ascii_digit()))
        .collect()
}

/// A fact's sentence, without the user's words a capture keeps after it (`SAID`).
pub fn sentence_of(text: &str) -> &str {
    text.split_once(SAID).map_or(text, |(fact, _)| fact)
}

/// Up to `SHOWN_FACTS` facts of `facts` (slug, text), the ones the memory index ranks best for
/// `messages` (BM25 over the fact files), each with its sentence alone.
pub fn shown(
    data_dir: &Path,
    facts: &[(String, String)],
    messages: &[String],
) -> Vec<(String, String)> {
    let how = super::Recall {
        sources: &[super::Source::Fact],
        // A fact has a line for its sentence and one for its `Said:`.
        hits: SHOWN_FACTS * 3,
        today: super::today(),
    };
    let hits = super::recall_with(data_dir, &messages.join("\n"), how).unwrap_or_default();
    let mut out: Vec<(String, String)> = Vec::new();
    for hit in hits {
        let slug = (hit.path.rsplit('/').next()).and_then(|f| f.strip_suffix(".md"));
        let Some((slug, text)) = slug.and_then(|s| facts.iter().find(|(f, _)| f == s)) else {
            continue;
        };
        if out.len() < SHOWN_FACTS && !out.iter().any(|(s, _)| s == slug) {
            out.push((slug.clone(), sentence_of(text).to_owned()));
        }
    }
    out
}

/// Where a sentence ends: a full stop, a question or exclamation mark (Latin, CJK, Devanagari,
/// Arabic), a line.
fn ends(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '\n' | '。' | '！' | '？' | '।' | '؟')
}

/// Words a quote needs, and a message, to be captured.
pub const QUOTE_WORDS: usize = 3;
/// Bytes of the user's sentence a fact keeps (`Said:`).
pub const SAID_BYTES: usize = 300;

/// Where `quote` is in `messages`, as words (`spans`): the message and the byte range from its
/// first word to its last. At least `QUOTE_WORDS` words, so a "yes" or an "ok" anchors nothing.
fn find_quote<'a>(messages: &'a [String], quote: &str) -> Option<(&'a str, usize, usize)> {
    let q = words(quote);
    if q.len() < QUOTE_WORDS {
        return None;
    }
    for m in messages {
        let w = spans(m);
        if let Some(i) = (0..=w.len().saturating_sub(q.len()))
            .find(|&i| w.len() >= q.len() && (0..q.len()).all(|j| w[i + j].2 == q[j]))
        {
            return Some((m, w[i].0, w[i + q.len() - 1].1));
        }
    }
    None
}

/// The whole sentence (or sentences) of `messages` that holds `quote` (`find_quote`), white
/// space joined; `None` when no message holds it.
pub fn source_sentence(messages: &[String], quote: &str) -> Option<String> {
    let (m, from, to) = find_quote(messages, quote)?;
    let (begin, end) = sentence_around(m, from, to);
    Some(join_space(&m[begin..end]))
}

/// The byte range of the sentence of `m` around `from..to`.
fn sentence_around(m: &str, from: usize, to: usize) -> (usize, usize) {
    let begin = (m[..from].char_indices().rev())
        .find(|&(_, c)| ends(c))
        .map_or(0, |(i, c)| i + c.len_utf8());
    let end = (m[to..].char_indices())
        .find(|&(_, c)| ends(c))
        .map_or(m.len(), |(i, c)| to + i + c.len_utf8());
    (begin, end)
}

fn join_space(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The user's sentence a fact keeps: the sentence around the quote, within `SAID_BYTES` around
/// it, a cut marked with `…`.
fn said(m: &str, from: usize, to: usize) -> String {
    let (begin, end) = sentence_around(m, from, to);
    if end - begin <= SAID_BYTES {
        return join_space(&m[begin..end]);
    }
    let room = SAID_BYTES.saturating_sub(to - from) / 2;
    let start = m.floor_char_boundary(from.saturating_sub(room).max(begin));
    let stop = m.floor_char_boundary((start + SAID_BYTES).min(end));
    let mark = |cut: bool| if cut { "…" } else { "" };
    format!(
        "{}{}{}",
        mark(start > begin),
        join_space(&m[start..stop]),
        mark(stop < end)
    )
}

/// Whether `text` looks like it holds a secret: a key, token or password, a card number, a US
/// social security number. Shapes, not words: it is a secret detector, not a fact filter.
pub fn has_secret(text: &str) -> bool {
    let token = |rest: &str, n: usize| {
        rest.chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .count()
            >= n
    };
    let after = |prefix: &str, n: usize| {
        text.match_indices(prefix)
            .any(|(i, _)| token(&text[i + prefix.len()..], n))
    };
    const PREFIXES: [&str; 16] = [
        "sk-",
        "hf_",
        "ASIA",
        "sk_live_",
        "sk_test_",
        "rk_live_",
        "ghp_",
        "gho_",
        "github_pat_",
        "glpat-",
        "xoxb-",
        "xoxp-",
        "xoxa-",
        "AKIA",
        "AIza",
        "Bearer ",
    ];
    let lower = text.to_lowercase();
    text.contains("-----BEGIN")
        || PREFIXES.iter().any(|p| after(p, 12))
        || after("bearer ", 12)
        || jwt(text)
        || card(text)
        || ssn(text)
        || ["password is ", "password:", "passcode is ", "pw is ", "pw:"]
            .iter()
            .any(|p| {
                // A word of its own: "pw:" is not the end of "rpw:".
                lower.match_indices(p).any(|(i, _)| {
                    !lower[..i]
                        .chars()
                        .next_back()
                        .is_some_and(char::is_alphanumeric)
                })
            })
}

/// A JSON web token: `eyJ`, then three dot-separated base64url parts.
fn jwt(text: &str) -> bool {
    text.match_indices("eyJ").any(|(i, _)| {
        let t: String = (text[i..].chars())
            .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
            .collect();
        let parts: Vec<&str> = t.split('.').collect();
        parts.len() == 3 && parts.iter().all(|p| p.len() >= 8)
    })
}

/// A run of 13 to 19 digits, spaces or dashes between them, that passes the Luhn check.
fn card(text: &str) -> bool {
    let mut digits: Vec<u32> = Vec::new();
    let luhn = |d: &[u32]| {
        (13..=19).contains(&d.len())
            && (d.iter().rev().enumerate())
                .map(|(i, &x)| {
                    if i % 2 == 1 {
                        (x * 2) % 9 + u32::from(x == 9) * 9
                    } else {
                        x
                    }
                })
                .sum::<u32>()
                % 10
                == 0
    };
    for c in text.chars().chain([' ', 'x']) {
        match c {
            '0'..='9' => digits.push(c.to_digit(10).unwrap_or(0)),
            ' ' | '-' if !digits.is_empty() => {}
            _ => {
                if luhn(&digits) {
                    return true;
                }
                digits.clear();
            }
        }
    }
    false
}

/// The `ddd-dd-dddd` shape, alone between non-digits.
fn ssn(text: &str) -> bool {
    let b = text.as_bytes();
    (0..b.len().saturating_sub(10)).any(|i| {
        let w = &b[i..i + 11];
        let d = |r: std::ops::Range<usize>| r.into_iter().all(|j| w[j].is_ascii_digit());
        d(0..3)
            && w[3] == b'-'
            && d(4..6)
            && w[6] == b'-'
            && d(7..11)
            && (i == 0 || !b[i - 1].is_ascii_digit())
            && b.get(i + 11).is_none_or(|c| !c.is_ascii_digit())
    })
}

/// The checks one item must pass, on its own: why it is dropped (a reason, never the user's
/// words: it goes to the audit log), or the user's sentence it comes from. The fact is one line;
/// the quote is the user's (`find_quote`); `replaces` is one of `shown`; every number of the fact
/// is in the user's sentence and every number of the quote in the fact; nothing is shaped like a
/// secret.
pub fn check(item: &Item, messages: &[String], shown: &[String]) -> Result<String, String> {
    if super::slug(&item.text).is_empty() {
        return Err("the fact has no words".into());
    }
    // A line of its own would read as the user's words (`Said:`) or front matter.
    if item.text.contains(['\n', '\r']) {
        return Err("the fact is more than one line".into());
    }
    if !TYPES.contains(&item.kind.as_str()) {
        return Err("the type is not one of the types".into());
    }
    let (m, from, to) = find_quote(messages, &item.quote)
        .ok_or_else(|| "the quote is not the user's words, or too short".to_owned())?;
    if item.replaces.as_ref().is_some_and(|s| !shown.contains(s)) {
        return Err("it replaces a fact that was not shown".into());
    }
    // Pasted text is someone else's: the quote may not lie in it, nor the sentence around it,
    // and it grounds no fact.
    let gone = pasted(m);
    let (begin, end) = sentence_around(m, from, to);
    if (gone.iter()).any(|r| r.start <= from && to <= r.end || r.start <= begin && end <= r.end) {
        return Err("the quote is text the user pasted, not their own words".into());
    }
    let own = without(&m[begin..end], begin, &gone);
    if !grounded(&item.text, &own, item.replaces.is_some()) {
        return Err("the fact's words are mostly not in the user's sentence".into());
    }
    let user = numbers(&own);
    if numbers(&item.text).iter().any(|n| !user.contains(n)) {
        return Err("the fact has a number the user did not say".into());
    }
    let text = numbers(&item.text);
    if numbers(&item.quote).iter().any(|n| !text.contains(n)) {
        return Err("the fact lacks a number of its quote".into());
    }
    if [&item.text, &item.quote, &m[begin..end].to_owned()]
        .iter()
        .any(|t| has_secret(t))
    {
        return Err("it looks like a secret".into());
    }
    // The clause the quote starts in: "Write a card for my wife, she loves tulips" has two.
    let clause = (m[begin..from].rfind([',', ';', ':'])).map_or(begin, |i| begin + i + 1);
    if retells(&item.text, &m[clause..end]) {
        return Err("the fact retells a request to the assistant".into());
    }
    Ok(said(m, from, to))
}

/// Verbs that open a request for something made or found: "write an essay", "create a page".
const TASK_VERBS: [&str; 32] = [
    "write",
    "create",
    "make",
    "draft",
    "generate",
    "build",
    "add",
    "list",
    "summarize",
    "summarise",
    "translate",
    "explain",
    "search",
    "find",
    "fill",
    "give",
    "tell",
    "show",
    "rewrite",
    "edit",
    "compose",
    "describe",
    "outline",
    "research",
    "compare",
    "calculate",
    "convert",
    "solve",
    "code",
    "fix",
    "design",
    "draw",
];
/// Words a request's verb may follow: "please", "can you", "help me".
const OPENERS: [&str; 16] = [
    "please", "pls", "kindly", "hey", "hi", "ok", "okay", "now", "can", "could", "would", "will",
    "you", "u", "help", "me",
];
/// A fact's words that tell of a request, not of the user. Not "wants" or "needs": a small model
/// writes a lasting preference that way ("Make sure you always answer in Spanish." as "The user
/// wants answers in Spanish."), and a request kept as a weak fact loses nothing.
const ASKED: [&str; 5] = ["request", "requests", "requested", "asks", "asked"];

/// Whether `fact` only retells the request its user `clause` makes: the clause opens with a task
/// verb, and the fact asks for something or says the user did that verb. A small model saved each
/// request despite `SYSTEM` ("Please write a 1000 word essay on mount kilamanjaro" as "The user
/// requested a 1000-word essay on Mount Kilimanjaro.", "create a new page in notion" as "The user
/// created a new page in Notion."), and the assistant, not the user, made the page. A fact told in
/// passing is the user's: "write a card for my wife Jane" saves "The user's wife is Jane.", "Write
/// in British English from now on" saves "The user prefers British English.".
fn retells(fact: &str, clause: &str) -> bool {
    let said = words(clause);
    let Some(verb) = said.iter().find(|w| !OPENERS.contains(&w.as_str())) else {
        return false;
    };
    if !TASK_VERBS.contains(&verb.as_str()) {
        return false;
    }
    // "create" is "created" or "creating"; "add" is not "address".
    let stem = verb.trim_end_matches('e');
    let did = |w: &str| {
        w.strip_prefix(stem)
            .is_some_and(|rest| ["", "e", "d", "ed", "s", "es", "ing"].contains(&rest))
    };
    (words(fact).iter()).any(|w| ASKED.contains(&w.as_str()) || did(w))
}

/// A character of a script written without spaces (Chinese, Japanese): a word by itself.
fn spaceless(c: char) -> bool {
    matches!(c as u32, 0x2E80..=0x9FFF | 0xF900..=0xFAFF)
}

/// Whether most of `fact`'s words are in the user's `sentence`: half or more of its words of 3
/// characters or more (a character of a script without spaces counts), numbers aside, each the
/// same as a word of the sentence or sharing its first 4 characters ("lives", "live"). Words the
/// capture's own instructions use ("the user's", "lives", "works") are its phrasing, not the
/// user's, and are left out unless the fact has no other. A sentence of 3 user words then
/// grounds no fact made of other words, in any language. In a new fact, a detail of the prompt's
/// examples (`EXAMPLE_DETAILS`) is the user's word or the fact is not; a fact that `replaces` one
/// keeps the old one's details as told, so its sister Maya may come from memory.
fn grounded(fact: &str, sentence: &str, replaces: bool) -> bool {
    let user = words(sentence);
    let copied = |w: &String| EXAMPLE_DETAILS.contains(&w.as_str()) && !user.contains(w);
    if !replaces && words(fact).iter().any(copied) {
        return false;
    }
    let digits = |w: &str| w.bytes().all(|b| b.is_ascii_digit());
    let long = |w: &str| w.chars().count() >= 3 || w.chars().all(spaceless);
    let all: Vec<String> = (words(fact).into_iter())
        .filter(|w| !digits(w) && long(w))
        .collect();
    let phrasing: Vec<String> = (words(SYSTEM).into_iter())
        .filter(|w| replaces || !EXAMPLE_DETAILS.contains(&w.as_str()))
        .collect();
    let own: Vec<&String> = all.iter().filter(|w| !phrasing.contains(w)).collect();
    let content: Vec<&String> = match own.is_empty() {
        true => all.iter().collect(),
        false => own,
    };
    let near = |a: &str, b: &str| {
        a == b || {
            let same = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
            same >= 4
        }
    };
    let hits = (content.iter())
        .filter(|w| user.iter().any(|u| near(u, w)))
        .count();
    hits * 2 >= content.len()
}

/// Words a quoted span holds, or more, to read as pasted text rather than a name or a title.
const PASTED_WORDS: usize = 8;

/// A grounding check, like the tool and reply text the capture never saves from, not a judgment
/// of what is worth keeping: a pasted email is not the user's own words.
/// The byte ranges of `message` that are someone else's text the user pasted, by their marks
/// alone, in any language: each line that starts with `>` (an email's quoted reply), and a span
/// in quotation marks (`"…"`, `“…”`, `„…“`, `«…»`, `「…」`, `『…』`) of `PASTED_WORDS` words or
/// more, or of more than one sentence. A short quoted name or title ("Dune", 「デューン」) is the
/// user's own words. A mark never closed opens nothing; a `"` after a digit (5'11", a 12" pizza)
/// is an inch mark, not a quote; an apostrophe is no mark.
fn pasted(message: &str) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut open: Option<(usize, &[char])> = None;
    let mut before: Option<char> = None;
    for (i, c) in message.char_indices() {
        match open {
            Some((from, close)) if close.contains(&c) => {
                let inside = &message[from..i];
                // A sentence end with more text after it, inside the marks.
                let sentences = (inside.char_indices())
                    .any(|(j, e)| ends(e) && !inside[j + e.len_utf8()..].trim().is_empty());
                if words(inside).len() >= PASTED_WORDS || sentences {
                    out.push(from..i + c.len_utf8());
                }
                open = None;
            }
            Some(_) => {}
            None => {
                let close: Option<&[char]> = match c {
                    '"' if !before.is_some_and(|b| b.is_ascii_digit()) => Some(&['"']),
                    '“' => Some(&['”']),
                    '„' => Some(&['“', '”']),
                    '«' => Some(&['»']),
                    '「' => Some(&['」']),
                    '『' => Some(&['』']),
                    _ => None,
                };
                open = close.map(|close| (i, close));
            }
        }
        before = Some(c);
    }
    let mut at = 0;
    for line in message.split_inclusive('\n') {
        if line.trim_start().starts_with('>') {
            out.push(at..at + line.len());
        }
        at += line.len();
    }
    out
}

/// `text` with the bytes of `gone` blanked, so its words are the user's own.
fn without(text: &str, offset: usize, gone: &[std::ops::Range<usize>]) -> String {
    (text.char_indices())
        .map(
            |(i, c)| match gone.iter().any(|r| r.contains(&(offset + i))) {
                true => ' ',
                false => c,
            },
        )
        .collect()
}

/// What comes between a captured fact's sentence and the user's own.
pub const SAID: &str = "\nSaid: \"";

/// A fact file's text: the model's sentence, then the user's sentence it comes from.
pub fn whole(text: &str, said: &str) -> String {
    format!("{}{SAID}{said}\"", text.trim())
}

/// What one capture saved and dropped.
#[derive(Debug, Default)]
pub struct Outcome {
    /// Each save's receipt.
    pub saved: Vec<Receipt>,
    /// Each dropped item's reason.
    pub dropped: Vec<String>,
}

/// Billions of parameters, all of them (every expert of a MoE), below which a capture model's
/// `replaces` is not trusted: its fact is saved as a new one and the fact it names stays as it
/// is. On the M5 bench the 8B dropped facts on wrong merges (mc-04, mc-24, mc-38), where a
/// duplicate line would have cost nothing. Total, not active, parameters: the 35B-A3B (about 3B
/// active) is the demo model and keeps its replaces.
pub const MIN_REPLACE_PARAMS_B: f32 = 20.0;

/// Whether a capture model of `params_b` billion parameters in all (`DeploymentSnapshot::
/// params_b`) replaces the fact a `replaces` names, or adds its fact beside it.
pub fn full_replaces(params_b: f32) -> bool {
    params_b >= MIN_REPLACE_PARAMS_B
}

/// `save_as` with full replaces, as a large model's capture saves.
pub fn save(
    data_dir: &Path,
    reply: &str,
    cut: bool,
    messages: &[String],
    shown: &[String],
    source: &str,
) -> Result<Outcome, String> {
    save_as(data_dir, reply, cut, messages, shown, source, true)
}

/// Parses `reply` and saves each item that passes `check`, in `data_dir`, as from `source`: with
/// `replaces` and `replace` (`full_replaces`), in place of that fact (`updated`, which keeps its
/// type and takes `source`); otherwise added, or in place of a fact of the same words
/// (`save_captured`). One bad item drops itself alone; a reply that is not the JSON asked for
/// saves nothing. A `cut` reply (at its token cap) saves the items it finished.
pub fn save_as(
    data_dir: &Path,
    reply: &str,
    cut: bool,
    messages: &[String],
    shown: &[String],
    source: &str,
    replace: bool,
) -> Result<Outcome, String> {
    let reply = reply.trim();
    let parsed: Reply = match cut {
        true => finished(reply).ok_or("the reply was cut before its first fact ended")?,
        false => serde_json::from_str(reply).map_err(|e| format!("not the JSON asked for: {e}"))?,
    };
    let mut out = Outcome::default();
    for (i, value) in parsed.facts.into_iter().enumerate() {
        let n = i + 1;
        if i >= MAX_ITEMS {
            out.dropped
                .push(format!("item {n}: over {MAX_ITEMS} items"));
            continue;
        }
        let mut item = match serde_json::from_value::<Item>(value) {
            Ok(item) => item,
            Err(_) => {
                out.dropped.push(format!("item {n}: not an item"));
                continue;
            }
        };
        // A small model's replace is an add, so the fact it names need not have been shown.
        if !replace {
            item.replaces = None;
        }
        let said = match check(&item, messages, shown) {
            Ok(said) => said,
            Err(why) => {
                out.dropped.push(format!("item {n}: {why}"));
                continue;
            }
        };
        let text = whole(&item.text, &said);
        // A large model's call: a replace replaces. Its receipt keeps the old fact whole, for
        // Undo. A small model's adds its fact and leaves the old one.
        let saved = match &item.replaces {
            Some(slug) => match super::updated(data_dir, slug, &text, Some(source)) {
                // Forgotten since it was shown: the fact is new again.
                Err(super::MemoryError::NoFact(_)) => {
                    super::save_captured(data_dir, &text, &item.kind, source)
                }
                saved => saved,
            },
            None => super::save_captured(data_dir, &text, &item.kind, source),
        };
        match saved {
            // The file's text, which `undo` checks; the app shows the sentence before `SAID`; `old`
            // and `was`, the replaced fact as its file held it, which `undo` puts back.
            Ok(s) => out.saved.push(s.receipt(&text)),
            // Memory went off while the capture ran: nothing more is saved.
            Err(super::MemoryError::Off) => {
                out.dropped.push(format!("item {n}: memory is off"));
                break;
            }
            Err(_) => out.dropped.push(format!("item {n}: it could not be saved")),
        }
    }
    Ok(out)
}

/// The items a reply cut at its token cap finished: the reply up to the last item that closed.
fn finished(reply: &str) -> Option<Reply> {
    (reply.match_indices('}').rev())
        .find_map(|(i, _)| serde_json::from_str::<Reply>(&format!("{}]}}", &reply[..=i])).ok())
}

/// A running capture another turn can cancel, and when it was told to.
struct Running {
    id: u64,
    cancel: Arc<AtomicBool>,
    cancelled: Arc<Mutex<Option<Instant>>>,
}

/// Every capture running in the process.
static RUNNING: LazyLock<Mutex<Vec<Running>>> = LazyLock::new(Mutex::default);
static NEXT: AtomicU64 = AtomicU64::new(0);
/// Moves on at each `cancel_all`, so a capture that was starting while one ran (sent, not yet in
/// `RUNNING`) cancels itself.
static EPOCH: AtomicU64 = AtomicU64::new(0);

/// Cancels every running capture, of every chat: a foreground request starts (a turn, a plain
/// chat's reply, a model switch). It does not wait.
pub fn cancel_all() {
    EPOCH.fetch_add(1, Ordering::SeqCst);
    let running = RUNNING.lock().unwrap_or_else(|e| e.into_inner());
    for r in running.iter() {
        *r.cancelled.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
        r.cancel.store(true, Ordering::Relaxed);
    }
}

/// How many captures run in the process now.
#[doc(hidden)]
pub fn running() -> usize {
    RUNNING.lock().unwrap_or_else(|e| e.into_inner()).len()
}

/// A chat's messages waiting for capture, shared with its running capture, which takes out the
/// ones it is done with: `messages` for the next capture, `alone` cut once at the token cap, each
/// to be read again by itself with `RETRY_TOKENS`.
#[derive(Debug, Default)]
pub struct Pending {
    /// The messages not yet captured, oldest first.
    pub messages: Vec<String>,
    /// Messages whose capture reached its token cap, each read again alone.
    pub alone: std::collections::VecDeque<String>,
}

/// `Pending`, shared.
pub type Waiting = Arc<Mutex<Pending>>;

/// The token cap of a capture read again alone.
pub const RETRY_TOKENS: u32 = 512;

/// What a capture needs besides its stream.
pub struct Job {
    /// The data folder it saves in.
    pub data_dir: PathBuf,
    /// The chat's ID, for the audit line.
    pub chat: String,
    /// Where its facts come from (the chat ID and turn).
    pub source: String,
    /// The messages it reads.
    pub messages: Vec<String>,
    /// The slugs it showed.
    pub shown: Vec<String>,
    /// The chat's waiting messages.
    pub waiting: Waiting,
    /// Where each save's receipt goes.
    pub receipts: Option<Sender<Receipt>>,
    /// A message read again alone (`Pending::alone`): a second cut saves what it finished and
    /// drops the rest.
    pub retry: bool,
    /// Whether a `replaces` replaces (`full_replaces`), or its fact is added.
    pub replace: bool,
}

/// The request's parameters: the model's worker settings (greedy, thinking off), `max_tokens`,
/// background class, no prefix cache, and the grammar of `schema`.
pub fn params(
    family: Option<Family>,
    slugs: &[String],
    max_tokens: u32,
) -> Result<GenParams, String> {
    let pattern = json_schema_regex(&schema(slugs)).map_err(|e| e.to_string())?;
    let worker = crate::run::params(
        &crate::tool::Registry::builtin(),
        &[],
        family,
        crate::run::Kind::Worker,
        Default::default(),
        0,
    );
    Ok(GenParams {
        max_tokens,
        seed: Some(0),
        prefix_cache: false,
        class: Class::Background,
        grammar: Some(Grammar {
            trigger: None,
            pattern,
        }),
        ..worker
    })
}

/// A capture the chat runs: its thread, and how to cancel it alone (the chat closes).
pub struct Started {
    /// The thread that reads, saves and logs.
    pub thread: JoinHandle<()>,
    /// Cancels this capture when set, while it waits for its seat or is read.
    pub cancel: Arc<AtomicBool>,
}

/// Sends the capture of `job` to the engine and reads it on a thread of its own, which waits for
/// its seat, saves what passes, sends the receipts, takes the messages it is done with out of
/// `job.waiting` and writes the audit line ("memory capture"). A cancelled capture saves nothing
/// and leaves them there.
pub fn start(
    larch: &Larch,
    family: Option<Family>,
    reply: &str,
    shown: &[(String, String)],
    job: Job,
) -> Result<Started, String> {
    let slugs: Vec<String> = shown.iter().map(|(s, _)| s.clone()).collect();
    let cap = if job.retry { RETRY_TOKENS } else { MAX_TOKENS };
    let p = params(family, &slugs, cap)?;
    let user = prompt(&job.messages, reply, shown);
    let messages = crate::run::messages(SYSTEM, &user, false);
    let started = Instant::now();
    let epoch = EPOCH.load(Ordering::SeqCst);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let cancelled = Arc::new(Mutex::new(None));
    let cancel = Arc::new(AtomicBool::new(false));
    RUNNING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Running {
            id,
            cancel: cancel.clone(),
            cancelled: Arc::clone(&cancelled),
        });
    // A foreground request began between the epoch's read and the push: this capture gives way
    // too.
    if EPOCH.load(Ordering::SeqCst) != epoch {
        *cancelled.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
        cancel.store(true, Ordering::Relaxed);
    }
    let job = Job {
        shown: slugs,
        ..job
    };
    let (larch, stop) = (larch.clone(), Arc::clone(&cancel));
    let thread = std::thread::Builder::new()
        .name("memory capture".into())
        .spawn(move || {
            // Its seat is waited for here, not on the turn's drop that starts it: background
            // work waits for every foreground request.
            let read = SeatAdmission::of(&larch)
                .seat(p.class, &stop, || {
                    larch.generate(Prompt::Messages(messages), p)
                })
                .and_then(|stream| crate::workers::answer(0, stream, &stop));
            run(read, job, id, &cancelled, started)
        })
        .map_err(|e| e.to_string())?;
    Ok(Started { thread, cancel })
}

/// The capture thread's end of `start`: what `read` gave.
fn run(
    read: Result<crate::workers::Answer, EngineError>,
    job: Job,
    id: u64,
    cancelled: &Mutex<Option<Instant>>,
    started: Instant,
) {
    RUNNING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|r| r.id != id);
    let ended = Instant::now();
    let told = *cancelled.lock().unwrap_or_else(|e| e.into_inner());
    let a = match read {
        Ok(a) => a,
        Err(EngineError::Cancelled) => {
            // A message read again alone waits for its turn again.
            if job.retry {
                let mut waiting = job.waiting.lock().unwrap_or_else(|e| e.into_inner());
                job.messages
                    .iter()
                    .rev()
                    .for_each(|m| waiting.alone.push_front(m.clone()));
            }
            let yielded = told.map(|t| ended.saturating_duration_since(t));
            return log(&job, None, None, None, started, yielded);
        }
        Err(e) => return log(&job, None, None, Some(e.to_string()), started, None),
    };
    let cut = a.finish == larch::Finish::Length;
    // Cut the first time: nothing is saved, and each message is read again alone, with more room.
    if cut && !job.retry {
        let mut waiting = job.waiting.lock().unwrap_or_else(|e| e.into_inner());
        waiting.messages.retain(|m| !job.messages.contains(m));
        waiting.alone.extend(job.messages.iter().cloned());
        drop(waiting);
        let error =
            Some("the reply was cut at its token cap; each message is read again alone".into());
        return log(&job, Some(&a), None, error, started, None);
    }
    let saved = save_as(
        &job.data_dir,
        &a.text,
        cut,
        &job.messages,
        &job.shown,
        &job.source,
        job.replace,
    );
    // A reply read whole (or cut twice) is done with, saved or not: its messages are not asked
    // about again.
    job.waiting
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .messages
        .retain(|m| !job.messages.contains(m));
    let error = match (&saved, cut) {
        (Err(e), _) => Some(e.clone()),
        (Ok(_), true) => Some("the reply was cut again; what it finished was saved".into()),
        _ => None,
    };
    if let (Ok(out), Some(to)) = (&saved, &job.receipts) {
        for r in &out.saved {
            // A receiver that went away wants no receipts.
            let _ = to.send(r.clone());
        }
    }
    log(&job, Some(&a), saved.ok(), error, started, None);
}

/// The capture's audit line: tokens, wall time, cancelled or not and how fast it gave way, how
/// many facts it saved, and why it dropped the others: counts and reasons, never the user's words
/// or the facts. Like the key turns' line it belongs to no chat's steps, so the chat ID goes in
/// `detail`.
fn log(
    job: &Job,
    a: Option<&crate::workers::Answer>,
    out: Option<Outcome>,
    error: Option<String>,
    started: Instant,
    yielded: Option<std::time::Duration>,
) {
    let cancelled = a.is_none() && error.is_none();
    let out = out.unwrap_or_default();
    let detail = json!({
        "chat": job.chat,
        "messages": job.messages.len(),
        "retry": job.retry,
        "shown": job.shown.len(),
        "cancelled": cancelled,
        "yield_ms": yielded.map(|d| d.as_secs_f64() * 1e3),
        "saved": out.saved.len(),
        "dropped": out.dropped,
    });
    let event = AuditEvent {
        kind: EventKind::Step,
        step: Some("memory capture".into()),
        prompt_tokens: a.map(|a| a.usage.prompt_tokens),
        cached_tokens: a.map(|a| a.usage.cached_tokens),
        generated_tokens: a.map(|a| a.usage.completion_tokens),
        ttft_ms: a.map(|a| a.timings.ttft_ms),
        wall_ms: Some(started.elapsed().as_secs_f64() * 1e3),
        error,
        detail: Some(detail),
        ..AuditEvent::default()
    };
    // Nobody reads this thread's errors: a lost line loses only its numbers.
    let _ = Audit::new(&job.data_dir).append(event);
}
