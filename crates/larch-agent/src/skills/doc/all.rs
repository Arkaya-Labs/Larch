//! `read_all` (plan 7.5): a task that needs the whole text, read part by part, each part whole by
//! a fresh part reader, whose items go to the reader's `Progress`, which a reset keeps.

use std::{
    sync::{
        LazyLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use larch::{EngineError, Finish, GenParams, Larch, Prompt};
use regex::Regex;
use serde_json::json;

use super::{
    reader::{self, Settings},
    record::{Item, PartState, Progress},
    send::{self, To},
};
use crate::{
    audit::Audit,
    folio::{
        Doc, Range,
        tools::{self, phrase},
    },
    run,
    skill::SkillError,
    skills::ask_docs,
    skills::read_papers,
    tool::ToolError,
    workers::Workers,
};

/// The tool that reads the whole text for a task.
pub const READ_ALL: &str = "read_all";

/// The tool that pages through what `read_all` found.
pub const ITEMS: &str = "items";

/// Every part reader's system prompt. Each sentence is A/B tested on the Mac, so it stays short.
pub const PART: &str = "You read one part of a document for a task. List each item of this \
part that the task asks for, one per line: the item, then the exact words of the text that show \
it in quote marks, then its place as the text marks it, such as [p. 12]. List only what this \
part's text says; if it has none, write \"None.\"";

/// Tokens a part may render past its pages' own counts, for each page: where two pages meet.
const SEAM: u64 = 4;

/// Tokens of the smallest part worth a reader.
const MIN_PART: u64 = 64;

/// Bytes of the smallest piece of a long page, so each piece moves past a whole character.
const MIN_PIECE: usize = 256;

/// An item line: an optional list mark, the item, its quote in quote marks, and its page.
static LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"^\s*(?:[-*•]|\d+[.)])?\s*(.*?)[\s:\-–—]*["“](.+)["”]\s*(?:\[(?:[A-Z]{1,2},\s*)?pp?\.\s*(\d+))?"#,
    )
    .expect("a valid pattern")
});

/// What part readers run on: the reader's turn.
pub struct Turn<'a> {
    /// The chat's engine.
    pub larch: &'a Larch,
    /// How readers run.
    pub s: &'a Settings,
    /// Where each part's trace line goes.
    pub audit: &'a Audit,
    /// The data folder's prefix-cache namespace.
    pub namespace: u64,
    /// The turn's own Stop.
    pub stop: &'a AtomicBool,
    /// Where "Reading part 3 of 12" goes.
    pub progress: &'a dyn Fn(&str),
}

/// `read_all(task)` on `docs`: cut into parts once per task (`kept` is the task `progress` is
/// for; another task starts again), each part still pending read by a fresh part reader, and what
/// they list added to `progress`. The result is the count, the two coverage lines and the first
/// items in `bytes`, ending with `room` tokens left.
pub fn read_all(
    t: &Turn,
    docs: &[Doc],
    kept: &mut String,
    progress: &mut Progress,
    task: &str,
    bytes: usize,
    room: usize,
) -> String {
    let left = tools::room_line(room);
    if task.trim().is_empty() {
        return format!("read_all needs a task: what to list from the whole text.\n{left}");
    }
    let p = reader::params(t.s, t.s.thinking, t.namespace, None);
    if progress.parts.is_empty() || phrase(task) != phrase(kept) {
        let mut parts = Vec::new();
        for d in docs {
            let (_, budget) = sizes(t.larch, t.s, &p, d, task);
            if budget < MIN_PART {
                return format!(
                    "read_all cannot run here: a part reader's prompt and answer leave no room \
                     for text in this context.\n{left}"
                );
            }
            parts.extend(split(t.larch, d, budget));
        }
        *progress = Progress {
            parts: parts.into_iter().map(|r| (r, PartState::Pending)).collect(),
            items: Vec::new(),
        };
        task.clone_into(kept);
    }
    read(t, &p, docs, task, progress);
    let found = match progress.items.len() {
        0 => String::new(),
        n => format!("{n} item{} found.\n", ask_docs::plural(n, "", "").0),
    };
    let head = format!("{found}{}\n", progress.coverage(docs));
    let list = progress.batch(docs, 0, bytes.saturating_sub(head.len() + left.len() + 1));
    format!("{head}{}\n{left}", list.trim_end())
}

/// `items(from)`: the items `progress` holds from `from`, in `bytes`, ending with `room` tokens
/// left.
pub fn items(docs: &[Doc], progress: &Progress, from: usize, bytes: usize, room: usize) -> String {
    let left = tools::room_line(room);
    let list = progress.batch(docs, from, bytes.saturating_sub(left.len() + 1));
    format!("{}\n{left}", list.trim_end())
}

/// Tokens of a part reader's request on `doc` besides its pages, its reply's room too, and the
/// pages' tokens a part may hold: within `followup_tokens` and the rest of one request.
fn sizes(larch: &Larch, s: &Settings, p: &GenParams, doc: &Doc, task: &str) -> (u64, u64) {
    let none = Range {
        doc: doc.index,
        start: 0,
        end: 0,
    };
    // Part numbers as long as any part's.
    let text = message(doc, 99_999, 99_999, none, task);
    let fixed = reader::first_request(larch, PART, p, &text);
    let budget = (reader::limit(larch).saturating_sub(fixed)).min(u64::from(s.followup_tokens));
    (fixed, budget)
}

/// `doc` cut at page ends into parts of `budget` tokens at most; a page past that is cut at line
/// ends into pieces about that long, each a part.
fn split(larch: &Larch, doc: &Doc, budget: u64) -> Vec<Range> {
    let (mut parts, mut part): (Vec<Range>, Option<(Range, u64)>) = (Vec::new(), None);
    for (n, page) in doc.pages.iter().enumerate() {
        let r = Range {
            doc: doc.index,
            start: page.start,
            end: page.end,
        };
        let mut shown = String::new();
        reader::shown(&mut shown, doc, false, r);
        let tokens = reader::count(larch, &shown) + SEAM;
        if let Some((p, used)) = &mut part
            && *used + tokens <= budget
        {
            (p.end, *used) = (r.end, *used + tokens);
            continue;
        }
        parts.extend(part.take().map(|(p, _)| p));
        if tokens <= budget {
            part = Some((r, tokens));
            continue;
        }
        // A tenth spare, as the page's own bytes a token stand for vary along it.
        let bytes = (r.end - r.start) as f64 * budget as f64 / tokens as f64 * 0.9;
        let (name, mut from) = (format!("page {}", n + 1), None);
        while let Ok(piece) = tools::read(doc, &[], &name, from, (bytes as usize).max(MIN_PIECE)) {
            parts.push(piece.range);
            from = piece.next;
            if from.is_none() {
                break;
            }
        }
    }
    parts.extend(part.map(|(p, _)| p));
    parts
}

/// Part `n` of `of`, range `r` of `doc`, as its reader gets it: the document, the part's pages
/// with their places, then the task.
fn message(doc: &Doc, n: usize, of: usize, r: Range, task: &str) -> String {
    let mut out = format!("{}\nPart {n} of {of}:\n", reader::header(doc, false));
    reader::shown(&mut out, doc, false, r);
    out + &format!("\nTask: {task}")
}

/// Reads `progress`'s pending parts of `docs` for `task`, a group of part readers at a time, and
/// keeps each part's state and items. A part that Stop ends stays pending, and no group starts
/// after it.
fn read(t: &Turn, p: &GenParams, docs: &[Doc], task: &str, progress: &mut Progress) {
    let (todo, of) = (progress.remaining(), progress.parts.len());
    let each = (docs.iter())
        .map(|d| {
            let (fixed, budget) = sizes(t.larch, t.s, p, d, task);
            fixed + budget
        })
        .max()
        .unwrap_or(0);
    let workers = Workers {
        max_workers: t.s.max_workers,
        prefix_cache: false,
    };
    let pool = u64::from(t.larch.deployment().kv_pool_blocks);
    // ponytail: groups of read_papers' readers number, each after the last one's slowest part;
    // #507's seats decide how many parts run at once when it merges
    let n = read_papers::at_once(todo.len(), workers, pool, each);
    let (larch, audit, stop) = (t.larch, t.audit, t.stop);
    for group in todo.chunks(n) {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let ns: Vec<usize> = group.iter().map(|i| i + 1).collect();
        let s = ask_docs::plural(ns.len(), "", "").0;
        (t.progress)(&format!("Reading part{s} {} of {of}", tools::numbers(&ns)));
        let sent: Vec<_> = thread::scope(|scope| {
            let runs: Vec<_> = (group.iter())
                .map(|&i| {
                    let r = progress.parts[i].0;
                    scope.spawn(move || {
                        let doc = tools::doc_of(docs, r.doc).ok_or_else(|| {
                            SkillError::Tool(ToolError::Skill("its document is not open".into()))
                        })?;
                        let text = message(doc, i + 1, of, r, task);
                        let prompt = Prompt::Messages(run::messages(PART, &text, false));
                        let detail = json!({
                            "docs": [read_papers::name(std::path::Path::new(&doc.name))],
                            "task": reader::preview(task),
                            "part": i + 1,
                            "of": of,
                            "range": [r.start, r.end],
                        });
                        let to = To::Once(prompt);
                        send::send(
                            larch,
                            to,
                            p.clone(),
                            "part",
                            detail,
                            audit,
                            stop,
                            &mut |_| false,
                        )
                    })
                })
                .collect();
            (runs.into_iter())
                .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                .collect()
        });
        for (&i, sent) in group.iter().zip(sent) {
            let r = progress.parts[i].0;
            progress.parts[i].1 = match sent {
                Ok(sent) => {
                    let a = sent.answer;
                    // The part's document was found to send it.
                    if let Some(doc) = tools::doc_of(docs, r.doc) {
                        for (text, quote, page) in listed(&a.text) {
                            let at = at(doc, r, page);
                            progress.add(docs, Item { text, quote, at });
                        }
                    }
                    match (a.finish, a.text.trim().is_empty()) {
                        (Finish::Length, _) => PartState::Cut,
                        (_, true) => PartState::Failed("its reader gave an empty answer".into()),
                        _ => PartState::Done,
                    }
                }
                // Stop leaves the part to a later read_all.
                Err(SkillError::Engine(EngineError::Cancelled)) => PartState::Pending,
                Err(e) => PartState::Failed(e.to_string()),
            };
        }
    }
}

/// What a part reader listed: each line's item, quote and 1-based page; a line with no quote is
/// no item ("None.").
fn listed(answer: &str) -> Vec<(String, String, Option<usize>)> {
    (answer.lines().filter_map(|l| LINE.captures(l)))
        .map(|c| {
            let page = c.get(3).and_then(|n| n.as_str().parse().ok());
            (c[1].trim().to_owned(), c[2].trim().to_owned(), page)
        })
        .collect()
}

/// Where an item cited on 1-based `page` is: that page inside `part`, else the whole part.
fn at(doc: &Doc, part: Range, page: Option<usize>) -> Range {
    let on = (page.and_then(|n| doc.pages.get(n.checked_sub(1)?))).map(|p| Range {
        doc: part.doc,
        start: p.start.max(part.start),
        end: p.end.min(part.end),
    });
    on.filter(|r| r.start < r.end).unwrap_or(part)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::folio::tools::tests::paged;

    #[test]
    fn a_part_readers_lines_are_its_items_and_their_pages_hold_them() {
        let got = listed(
            "- Wind: \"Wind must stay under 20 m/s.\" [p. 2]\n\
             None.\n\
             2. \u{201c}Only a quote\u{201d} [pp. 3-4]\n\
             Heat \u{2013} \"under 60 C\"\n\
             a line with no quote [p. 5]",
        );
        assert_eq!(
            got,
            [
                (
                    "Wind".into(),
                    "Wind must stay under 20 m/s.".into(),
                    Some(2)
                ),
                (String::new(), "Only a quote".into(), Some(3)),
                ("Heat".into(), "under 60 C".into(), None),
            ]
        );
        let d = paged(0, &["one", "two", "three"]);
        let part = Range {
            doc: 0,
            start: d.pages[1].start,
            end: d.pages[2].end,
        };
        let page = |n: usize| Range {
            doc: 0,
            start: d.pages[n].start,
            end: d.pages[n].end,
        };
        assert_eq!(at(&d, part, Some(3)), page(2));
        // A page outside its part, or none, is the part.
        assert_eq!((at(&d, part, Some(1)), at(&d, part, None)), (part, part));
    }
}
