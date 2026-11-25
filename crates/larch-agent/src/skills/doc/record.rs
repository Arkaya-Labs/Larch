//! The searches a reader made and the progress of its task, kept apart so a reset loses no work.

use crate::folio::tools::{doc_of, listed, numbers, phrase, place};
use crate::folio::{Doc, Range, visible::Visible};
use crate::skills::ask_docs::squash;

/// Bytes `Progress::batch` keeps for its last line.
const FOOTER: usize = 96;

/// The searches a reader made, for the advisory note on a repeated one.
#[derive(Clone, Debug, Default)]
pub struct Searches {
    done: Vec<(String, usize)>,
}

impl Searches {
    /// Records the search for `words` from match `from` that found `total` matches, and returns
    /// the note for a repeat of an earlier identical one ("you searched for 'x' before and it had
    /// no matches"). Words are the same when `find` matches them the same; a different search,
    /// or the same one from another `from`, gets none.
    pub fn note(&mut self, words: &str, from: usize, total: usize) -> Option<String> {
        let key = (phrase(words), from);
        if !self.done.contains(&key) {
            self.done.push(key);
            return None;
        }
        let at = match from {
            0 => String::new(),
            n => format!(" from={n}"),
        };
        let had = match total {
            0 => "no matches".to_string(),
            1 => "1 match".to_string(),
            n => format!("{n} matches"),
        };
        Some(format!(
            "you searched for '{}'{at} before and it had {had}.",
            squash(words).0
        ))
    }
}

/// How far one part of a `read_all` got.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartState {
    /// Not read yet.
    Pending,
    /// Its reader finished with no cut.
    Done,
    /// Its reader answered, but its answer was cut.
    Cut,
    /// It failed, and why.
    Failed(String),
}

/// One item a part reader found: its text, its quote, and where the quote is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Item {
    /// The item, in the reader's words.
    pub text: String,
    /// The quote it rests on.
    pub quote: String,
    /// Where the quote is.
    pub at: Range,
}

/// A task's progress, which survives a session reset: each part's state and the items found.
#[derive(Clone, Debug, Default)]
pub struct Progress {
    /// Each part, in document order, with its state.
    pub parts: Vec<(Range, PartState)>,
    /// The items found so far, deduplicated, in document order.
    pub items: Vec<Item>,
}

/// `text`'s words in lower case, so "Revenue: $4M." and "revenue $4M" are one item.
fn words(text: &str) -> String {
    (text.split(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

impl Progress {
    /// Adds `item` in document order unless an item with the same words is on the same page of
    /// `docs`; false for such a duplicate.
    pub fn add(&mut self, docs: &[Doc], item: Item) -> bool {
        let key = |i: &Item| {
            let page = doc_of(docs, i.at.doc).map(|d| d.page_of(i.at.start));
            (i.at.doc, page, words(&i.text))
        };
        let k = key(&item);
        if self.items.iter().any(|i| key(i) == k) {
            return false;
        }
        let at = (self.items)
            .partition_point(|i| (i.at.doc, i.at.start) <= (item.at.doc, item.at.start));
        self.items.insert(at, item);
        true
    }

    /// The parts still to read, by index: a reset never sends a finished part again.
    pub fn remaining(&self) -> Vec<usize> {
        (self.parts.iter().enumerate())
            .filter(|(_, (_, s))| *s == PartState::Pending)
            .map(|(i, _)| i)
            .collect()
    }

    /// The items from `from`, numbered, each with its quote and place, as many as fit `max`
    /// bytes, with a last line when some are left out ("items 1 to 40 of 73; ask for the rest
    /// with items(from=40)"). The parent reads its batches with it, and the final list is it from
    /// 0; the numbers hold while no item is added.
    pub fn batch(&self, docs: &[Doc], from: usize, max: usize) -> String {
        let (several, n) = (docs.len() > 1, self.items.len());
        if from >= n {
            return match n {
                0 => "no items found.".into(),
                _ => format!("items(from={from}) is past the last of {n} items."),
            };
        }
        let budget = max.saturating_sub(FOOTER);
        let (mut out, mut end) = (String::new(), from);
        for (i, item) in self.items.iter().enumerate().skip(from) {
            let at = doc_of(docs, item.at.doc).map(|d| place(d, several, item.at));
            let (text, quote) = (squash(&item.text).0, squash(&item.quote).0);
            let mut line = match (text.is_empty(), quote.is_empty()) {
                (_, true) => format!("{}. {text}", i + 1),
                (true, false) => format!("{}. \"{quote}\"", i + 1),
                (false, false) => format!("{}. {text}: \"{quote}\"", i + 1),
            };
            line += &at.map(|a| format!(" {a}\n")).unwrap_or_else(|| "\n".into());
            if out.len() + line.len() > budget {
                if end > from {
                    break;
                }
                // One item over the whole batch still goes, cut, so the cursor always moves.
                line.truncate(line.floor_char_boundary(budget.saturating_sub(4)));
                line += "…\n";
            }
            out += &line;
            end = i + 1;
        }
        match (from, end) {
            (0, e) if e == n => out,
            (_, e) if e == n => out + &format!("items {} to {e} of {n}.", from + 1),
            (_, e) => {
                out + &format!(
                    "items {} to {e} of {n}; ask for the rest with items(from={e}).",
                    from + 1
                )
            }
        }
    }

    /// The two coverage lines: the pages that reached a reader and those that did not (after a
    /// Stop or a failure), then the parts whose reader finished without being cut. Neither says
    /// every item was found.
    pub fn coverage(&self, docs: &[Doc]) -> String {
        let mut reached = Visible::default();
        for (r, s) in &self.parts {
            if matches!(s, PartState::Done | PartState::Cut) {
                reached.seen(*r);
            }
        }
        let several = docs.len() > 1;
        let (mut got, mut missed) = (Vec::new(), Vec::new());
        for d in docs
            .iter()
            .filter(|d| self.parts.iter().any(|(r, _)| r.doc == d.index))
        {
            let (yes, no): (Vec<_>, Vec<_>) = (1..=d.pages.len()).partition(|&n| {
                let p = &d.pages[n - 1];
                reached.holds(Range {
                    doc: d.index,
                    start: p.start,
                    end: p.end,
                })
            });
            let letter = if several {
                d.letter() + " "
            } else {
                String::new()
            };
            got.push(format!("{letter}{} of {}", numbers(&yes), d.pages.len()));
            if !no.is_empty() {
                missed.push(format!("{letter}{}", numbers(&no)));
            }
        }
        let mut out = format!("Pages that reached a reader: {}", got.join(", "));
        if !missed.is_empty() {
            out += &format!("; not reached: {}", missed.join(", "));
        }
        let parts = |want: fn(&PartState) -> bool| -> Vec<usize> {
            (self.parts.iter().enumerate())
                .filter(|(_, (_, s))| want(s))
                .map(|(i, _)| i + 1)
                .collect()
        };
        let done = parts(|s| *s == PartState::Done).len();
        out += &format!(
            ".\nParts whose reader finished without being cut: {done} of {}",
            self.parts.len()
        );
        for (what, ns) in [
            ("cut", parts(|s| *s == PartState::Cut)),
            ("not read", parts(|s| *s == PartState::Pending)),
        ] {
            if !ns.is_empty() {
                out += &format!("; {what}: {}", numbers(&ns));
            }
        }
        let failed: Vec<_> = (self.parts.iter().enumerate())
            .filter_map(|(i, (_, s))| match s {
                PartState::Failed(why) => Some(format!("{} ({why})", i + 1)),
                _ => None,
            })
            .collect();
        if !failed.is_empty() {
            out += &format!("; failed: {}", listed(&failed));
        }
        out + "."
    }
}

#[cfg(test)]
mod tests {
    use std::slice::from_ref;

    use super::*;
    use crate::folio::tools::{read, tests::paged};

    fn r(doc: usize, start: usize, end: usize) -> Range {
        Range { doc, start, end }
    }

    #[test]
    fn a_repeated_search_gets_a_note_and_a_different_one_does_not() {
        let mut s = Searches::default();
        assert_eq!(s.note("thermal", 0, 0), None);
        assert_eq!(s.note("cooling fan", 0, 0), None);
        assert_eq!(
            s.note(" Thermal ", 0, 0).as_deref(),
            Some("you searched for 'Thermal' before and it had no matches.")
        );
        assert_eq!(s.note("x  y", 0, 3), None);
        assert_eq!(
            s.note("X\ny", 0, 3).as_deref(),
            Some("you searched for 'X y' before and it had 3 matches.")
        );
        // The next page of results is not a repeat.
        assert_eq!(s.note("x y", 20, 3), None);
    }

    #[test]
    fn a_reset_clears_what_is_seen_and_keeps_the_progress() {
        let d = paged(0, &["one\n", "two\n", "three\n"]);
        let page = |n: usize| r(0, d.pages[n].start, d.pages[n].end);
        let mut progress = Progress {
            parts: vec![
                (page(0), PartState::Done),
                (page(1), PartState::Cut),
                (page(2), PartState::Pending),
            ],
            items: Vec::new(),
        };
        let mut seen = Visible::default();
        seen.seen(page(0));
        let one = read(&d, &[], "page 1", None, 100).unwrap();
        assert!(
            one.render(from_ref(&d), &seen, 1)
                .starts_with("page 1 is already above.")
        );
        // The reset: the session holds nothing, the progress is kept whole.
        seen.clear();
        let kept = progress.clone();
        assert_eq!(progress.remaining(), [2]);
        assert!(one.render(from_ref(&d), &seen, 1).contains("[p. 1]\none\n"));
        progress.parts[2].1 = PartState::Done;
        assert_eq!((progress.remaining(), kept.parts.len()), (vec![], 3));
    }

    #[test]
    fn items_are_deduplicated_by_their_words_and_page_in_document_order() {
        let (a, b) = (
            paged(0, &["x".repeat(50).as_str(), "y".repeat(50).as_str()]),
            paged(1, &["z"]),
        );
        let docs = [a.clone(), b];
        let item = |text: &str, doc: usize, at: usize| Item {
            text: text.into(),
            quote: format!("q {text}"),
            at: r(doc, at, at + 3),
        };
        let mut p = Progress::default();
        assert!(p.add(&docs, item("Revenue: $4M.", 0, 60)));
        assert!(p.add(&docs, item("Margin 20%", 0, 10)));
        assert!(
            !p.add(&docs, item("revenue  $4M", 0, 70)),
            "same words, same page"
        );
        assert!(p.add(&docs, item("Revenue: $4M.", 0, 20)), "another page");
        assert!(
            p.add(&docs, item("Revenue: $4M.", 1, 0)),
            "another document"
        );
        assert_eq!(p.items.len(), 4);
        assert_eq!(
            p.batch(&docs, 0, 10_000),
            "1. Margin 20%: \"q Margin 20%\" [A, p. 1]\n\
             2. Revenue: $4M.: \"q Revenue: $4M.\" [A, p. 1]\n\
             3. Revenue: $4M.: \"q Revenue: $4M.\" [A, p. 2]\n\
             4. Revenue: $4M.: \"q Revenue: $4M.\" [B, p. 1]\n"
        );
    }

    #[test]
    fn sixty_part_results_reach_an_8k_parent_only_in_batches() {
        let pages: Vec<String> = (0..60).map(|i| format!("page {i} ").repeat(40)).collect();
        let d = paged(0, &pages.iter().map(String::as_str).collect::<Vec<_>>());
        let docs = [d.clone()];
        let mut p = Progress::default();
        for (n, page) in d.pages.iter().enumerate() {
            p.parts.push((r(0, page.start, page.end), PartState::Done));
            // About 300 tokens of items a part.
            for k in 0..4 {
                let text = format!("item {k} of part {n}: {}", "detail ".repeat(36));
                let added = p.add(
                    &docs,
                    Item {
                        text,
                        quote: format!("page {n}"),
                        at: r(0, page.start + k, page.start + k + 6),
                    },
                );
                assert!(added);
            }
        }
        let all: usize = p.items.iter().map(|i| i.text.len() + i.quote.len()).sum();
        assert!(all / 4 > 8192, "the items alone overflow an 8k context");
        let coverage = p.coverage(&docs);
        assert_eq!(
            coverage,
            "Pages that reached a reader: 1-60 of 60.\n\
             Parts whose reader finished without being cut: 60 of 60."
        );
        // The parent's request: its prompt, the count, the coverage and one batch of a quarter
        // of the room.
        let (max, mut from, mut listed) = (8192, 0, Vec::new());
        loop {
            let batch = p.batch(&docs, from, max);
            let prompt = 2_000 * 4 + format!("{} items", p.items.len()).len() + coverage.len();
            assert!(batch.len() <= max && (prompt + batch.len()) / 4 < 8192);
            listed.extend(
                batch
                    .lines()
                    .filter(|l| !l.starts_with("items "))
                    .map(str::to_owned),
            );
            match batch.rsplit_once("items(from=") {
                Some((_, n)) => from = n.trim_end_matches(").").parse().unwrap(),
                None => break,
            }
        }
        assert_eq!(listed.len(), 240);
        assert!(
            listed[0].starts_with("1. item 0 of part 0")
                && listed[239].starts_with("240. item 3 of part 59")
        );
        let first = p.batch(&docs, 0, max);
        let last = first.rsplit_once('\n').unwrap().1;
        assert!(
            last.starts_with("items 1 to ")
                && last.contains(" of 240; ask for the rest with items(from="),
            "{last}"
        );
    }

    #[test]
    fn coverage_names_the_pages_not_reached_and_the_parts_cut() {
        let a = paged(0, &["a1", "a2", "a3", "a4"]);
        let b = paged(1, &["b1", "", "b3"]);
        let docs = [a.clone(), b.clone()];
        let pages = |d: &Doc, x: usize, y: usize| r(d.index, d.pages[x].start, d.pages[y].end);
        let p = Progress {
            parts: vec![
                (pages(&a, 0, 1), PartState::Done),
                (pages(&a, 2, 2), PartState::Cut),
                (
                    pages(&a, 3, 3),
                    PartState::Failed("the engine stopped".into()),
                ),
                (pages(&b, 0, 2), PartState::Pending),
            ],
            items: Vec::new(),
        };
        assert_eq!(
            p.coverage(&docs),
            "Pages that reached a reader: A 1-3 of 4, B none of 3; not reached: A 4, B 1-3.\n\
             Parts whose reader finished without being cut: 1 of 4; cut: 2; not read: 4; \
             failed: 3 (the engine stopped)."
        );
    }
}
