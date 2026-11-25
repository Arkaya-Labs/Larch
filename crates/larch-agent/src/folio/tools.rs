//! The reader's read-only tools: exact search and reading by part, with ranges and cursors.

use std::sync::LazyLock;

use regex::Regex;

use super::{Doc, Page, Range, map, visible::Visible};
use crate::skills::ask_docs::squash;

/// Bytes of a line `find` gives whole; a longer one is cut to about this around its match, as
/// PDF text can hold a whole page on one line.
const LINE_BYTES: usize = 400;

/// What a hit costs in the result beyond its text: its place, a space and the line end.
const HIT_COST: usize = 16;

/// Labels the error for a missing part offers, nearest first.
const NEAREST: usize = 5;

/// "page 12", "p. 12", "pages 12-14", "pp. 12 to 14", case-insensitive.
static PAGES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:pages?|pp?)\.?\s*(\d+)(?:\s*(?:-|\u{2013}|\u{2014}|to)\s*(\d+))?$")
        .expect("a valid pattern")
});

/// One line `find` matched: its range in the text and the page it is on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    /// The matching line.
    pub range: Range,
    /// The page index it is on.
    pub page: usize,
}

/// What `find` returns: the hits that fit, the total, the cursor for the rest, and what it
/// searched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Found {
    /// The hits from `from`, as many as fit the room.
    pub hits: Vec<Hit>,
    /// Every match in the searched text.
    pub total: usize,
    /// The match the hits start at, as asked.
    pub from: usize,
    /// The `from` that gives the next hits, when some did not fit.
    pub next: Option<usize>,
    /// Pages searched, over every document given.
    pub pages: usize,
    /// (document, page) of each page that has no text, which no search can match.
    pub empty: Vec<(usize, usize)>,
}

/// `words` as `find` matches them: every run of whitespace one space, ASCII lower case.
pub(crate) fn phrase(words: &str) -> String {
    let mut p = squash(words).0;
    p.make_ascii_lowercase();
    p
}

/// Exact search, ASCII case folded, for `words` in `docs`: the matching lines from match
/// `from`, as many as fit `room` bytes. A run of whitespace in the words or the text matches
/// any other, as PDF text breaks lines anywhere; a hit holds every match on its line.
pub fn find(docs: &[Doc], words: &str, from: usize, room: usize) -> Found {
    let needle = phrase(words);
    // (position in `docs`, start, end) of each match in the original text.
    let mut matches = Vec::new();
    if !needle.is_empty() {
        for (i, d) in docs.iter().enumerate() {
            let (mut hay, at) = squash(&d.text);
            hay.make_ascii_lowercase();
            matches.extend(hay.match_indices(&needle).map(|(s, m)| {
                let last = at[s + m.len() - 1];
                let end = last + d.text[last..].chars().next().map_or(1, char::len_utf8);
                (i, at[s], end)
            }));
        }
    }
    let (mut hits, mut used, mut next) = (Vec::new(), 0, from.min(matches.len()));
    while let Some(&(i, s, e)) = matches.get(next) {
        let d = &docs[i];
        let (a, b) = line(&d.text, s, e);
        if used + b - a + HIT_COST > room {
            break;
        }
        used += b - a + HIT_COST;
        next += 1;
        while matches
            .get(next)
            .is_some_and(|&(j, s2, e2)| j == i && a <= s2 && e2 <= b)
        {
            next += 1;
        }
        hits.push(Hit {
            range: Range {
                doc: d.index,
                start: a,
                end: b,
            },
            page: d.page_of(s),
        });
    }
    let empty = docs.iter().flat_map(|d| {
        let blank = |p: &Page| {
            d.text
                .get(p.start..p.end)
                .is_none_or(|t| t.trim().is_empty())
        };
        (d.pages.iter().enumerate())
            .filter(move |(_, p)| blank(p))
            .map(move |(n, _)| (d.index, n))
    });
    Found {
        hits,
        total: matches.len(),
        from,
        next: (next < matches.len()).then_some(next),
        pages: docs.iter().map(|d| d.pages.len()).sum(),
        empty: empty.collect(),
    }
}

/// The whole lines of `text` that hold `s..e`, or, when they run over `LINE_BYTES`, about that
/// much around the match, cut at word boundaries.
fn line(text: &str, s: usize, e: usize) -> (usize, usize) {
    let breaks = ['\n', '\x0c'];
    let a = text[..s].rfind(breaks).map_or(0, |i| i + 1);
    let b = text[e..].find(breaks).map_or(text.len(), |i| e + i);
    if b - a <= LINE_BYTES {
        return (a, b);
    }
    let side = LINE_BYTES.saturating_sub(e - s) / 2;
    let x = text.ceil_char_boundary(s.saturating_sub(side).max(a));
    let x = match x > a {
        true => (text[x..s].char_indices())
            .find(|(_, c)| c.is_whitespace())
            .map_or(x, |(i, c)| x + i + c.len_utf8()),
        false => x,
    };
    let y = text.floor_char_boundary(e.saturating_add(side).min(b));
    let y = match y < b && !text[y..].starts_with(char::is_whitespace) {
        true => text[e..y].rfind(char::is_whitespace).map_or(y, |i| e + i),
        false => y,
    };
    (x, y)
}

/// What `read` returns: the range it gave, the whole part it is a piece of, and the cursor for
/// the next piece.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    /// The text given.
    pub range: Range,
    /// The whole part asked for.
    pub part: Range,
    /// The `from` byte that gives the next piece, when the part did not fit.
    pub next: Option<usize>,
    /// The part as the model asks for it again: "pages 12-14", "page 12", or its label.
    pub name: String,
}

/// The 1-based first and last page `part` names, when it names pages.
fn pages(part: &str) -> Option<(usize, usize)> {
    let c = PAGES.captures(part.trim())?;
    let a: usize = c[1].parse().ok()?;
    let b: usize = c.get(2).map_or(Ok(a), |b| b.as_str().parse()).ok()?;
    Some((a.min(b), a.max(b)))
}

/// The `part` the model names ("pages 12-14", "p. 12", a heading or a label in `labels`) of
/// `doc`, from byte `from` of it, as much as fits `room` bytes; the error names the nearest parts.
/// A piece ends at a line end when one is in its second half, else at a word's end.
pub fn read(
    doc: &Doc,
    labels: &[(String, Range)],
    part: &str,
    from: Option<usize>,
    room: usize,
) -> Result<Piece, String> {
    let n = doc.pages.len();
    let (name, whole) = match pages(part) {
        Some((a, _)) if a == 0 || a > n => {
            return Err(format!(
                "there is no page {a}: the document has {n} pages, 1 to {n}"
            ));
        }
        Some((a, b)) => {
            let b = b.min(n);
            let name = match a == b {
                true => format!("page {a}"),
                false => format!("pages {a}-{b}"),
            };
            (
                name,
                range(doc, doc.pages[a - 1].start, doc.pages[b - 1].end),
            )
        }
        None => label(doc, labels, part)?,
    };
    let start = match from {
        None => whole.start,
        Some(f) if (whole.start..whole.end).contains(&f) => doc.text.ceil_char_boundary(f),
        Some(f) => {
            return Err(format!(
                "from={f} is outside {name}; read({name:?}) gives it from its start"
            ));
        }
    };
    let end = match start.saturating_add(room) {
        max if max >= whole.end => whole.end,
        max => cut(&doc.text, start, max),
    };
    Ok(Piece {
        range: range(doc, start, end),
        part: whole,
        next: (end < whole.end).then_some(end),
        name,
    })
}

/// `start..end` of `doc`.
fn range(doc: &Doc, start: usize, end: usize) -> Range {
    Range {
        doc: doc.index,
        start,
        end,
    }
}

/// The label of `doc` that `part` names, case-insensitive: the one equal to it, else the one
/// label that holds it; the error lists the nearest labels.
fn label(doc: &Doc, labels: &[(String, Range)], part: &str) -> Result<(String, Range), String> {
    let key = |s: &str| squash(s).0.to_lowercase();
    let asked = key(part);
    let own: Vec<_> = labels.iter().filter(|(_, r)| r.doc == doc.index).collect();
    if let Some((l, r)) = own.iter().find(|(l, _)| key(l) == asked) {
        return Ok((l.clone(), *r));
    }
    let mut near: Vec<_> = own
        .iter()
        .filter(|(l, _)| key(l).contains(&asked))
        .collect();
    if let [(l, r)] = near[..] {
        return Ok((l.clone(), *r));
    }
    if near.is_empty() {
        near = own.iter().collect();
        near.sort_by(|(x, _), (y, _)| {
            strsim::jaro_winkler(&key(y), &asked).total_cmp(&strsim::jaro_winkler(&key(x), &asked))
        });
    }
    let n = doc.pages.len();
    Err(match near.is_empty() {
        true => format!("no part named {part:?}; read pages by number, 1 to {n}"),
        false => format!(
            "no part named {part:?}; the nearest: {}; or pages 1 to {n}",
            (near.iter().take(NEAREST))
                .map(|(l, _)| format!("{l:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    })
}

/// Where a piece from `start` that may run to `max` ends: after its last line end when that is
/// in its second half, else at its last word's end, else at a character boundary.
fn cut(text: &str, start: usize, max: usize) -> usize {
    let max = text.floor_char_boundary(max);
    let s = &text[start..max];
    if let Some(i) = s.rfind('\n').filter(|&i| i >= s.len() / 2) {
        return start + i + 1;
    }
    if text[max..].starts_with(char::is_whitespace) {
        return max;
    }
    (s.char_indices().rev())
        .find(|(_, c)| c.is_whitespace())
        .map_or(max, |(i, c)| start + i + c.len_utf8())
}

/// The document of `docs` with chat index `index`.
pub(crate) fn doc_of(docs: &[Doc], index: usize) -> Option<&Doc> {
    docs.iter().find(|d| d.index == index)
}

/// The 1-based first and last page of `r`.
fn span(doc: &Doc, r: Range) -> (usize, usize) {
    let last = r.end.saturating_sub(1).max(r.start);
    (doc.page_of(r.start) + 1, doc.page_of(last) + 1)
}

/// Where `r` is, as the reader cites it: "[p. 12]" or "[pp. 12-13]", with the document's letter
/// first when the chat has `several` ("[B, p. 12]").
pub fn place(doc: &Doc, several: bool, r: Range) -> String {
    let pages = match span(doc, r) {
        (a, b) if a == b => format!("p. {a}"),
        (a, b) => format!("pp. {a}-{b}"),
    };
    match several {
        true => format!("[{}, {pages}]", doc.letter()),
        false => format!("[{pages}]"),
    }
}

/// `r`'s pages in words, "page 12" or "pages 12 to 14", and whether that is more than one.
fn pages_of(doc: &Doc, r: Range) -> (String, bool) {
    match span(doc, r) {
        (a, b) if a == b => (format!("page {a}"), false),
        (a, b) => (format!("pages {a} to {b}"), true),
    }
}

/// `items` as a list in words: "A", "A and B", "A, B and C".
pub(crate) fn listed(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// Sorted 1-based numbers in runs, "3, 7-9 and 12", or "none".
pub(crate) fn numbers(ns: &[usize]) -> String {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &n in ns {
        match runs.last_mut() {
            Some((_, b)) if *b + 1 == n => *b = n,
            _ => runs.push((n, n)),
        }
    }
    let runs: Vec<_> = (runs.iter())
        .map(|&(a, b)| match a == b {
            true => a.to_string(),
            false => format!("{a}-{b}"),
        })
        .collect();
    if runs.is_empty() {
        "none".into()
    } else {
        listed(&runs)
    }
}

/// The last line of every tool result.
pub(crate) fn room_line(room_tokens: usize) -> String {
    format!("Room left: about {room_tokens} tokens.")
}

impl Found {
    /// The tool result the model reads, ending with the room left in tokens. `docs` are the
    /// chat's documents: with several, each place names its document's letter.
    pub fn render(&self, docs: &[Doc], words: &str, room_tokens: usize) -> String {
        let words = squash(words).0;
        let several = docs.len() > 1;
        let matches = match self.total {
            1 => "1 match".to_string(),
            n => format!("{n} matches"),
        };
        let mut out = match (self.total, self.hits.is_empty()) {
            (0, _) => "no literal matches in the extracted text; this does not mean the idea is \
                       absent, only that these exact words are not in the text.\n"
                .to_string(),
            (_, true) if self.from >= self.total => {
                format!("{matches}; from={} is past the last one.\n", self.from)
            }
            (_, true) => format!("{matches}; none fit the room left.\n"),
            (_, false) => {
                let end = self.next.unwrap_or(self.total);
                let more = (self.next)
                    .map(|n| format!("; more: find({words:?}, from={n})"))
                    .unwrap_or_default();
                format!("{matches}; showing {}-{end}{more}\n", self.from + 1)
            }
        };
        for h in &self.hits {
            let Some(d) = doc_of(docs, h.range.doc) else {
                continue;
            };
            let at_line_start =
                h.range.start == 0 || d.text[..h.range.start].ends_with(['\n', '\x0c']);
            let at_line_end =
                (d.text[h.range.end..].chars().next()).is_none_or(|c| c == '\n' || c == '\x0c');
            out += &format!(
                "{} {}{}{}\n",
                place(d, several, h.range),
                if at_line_start { "" } else { "…" },
                squash(&d.text[h.range.start..h.range.end]).0,
                if at_line_end { "" } else { "…" },
            );
        }
        out += &self.searched(docs);
        out + &room_line(room_tokens)
    }

    /// The coverage line: "searched all 212 pages; pages 3 and 7 have no text."
    fn searched(&self, docs: &[Doc]) -> String {
        let several = docs.len() > 1;
        let of = match several {
            true => format!(
                " of {}",
                listed(&docs.iter().map(Doc::letter).collect::<Vec<_>>())
            ),
            false => String::new(),
        };
        let mut line = match self.pages == docs.len() {
            true => format!("searched the whole text{of}"),
            false => format!("searched all {} pages{of}", self.pages),
        };
        let empty: Vec<String> = (docs.iter())
            .filter_map(|d| {
                let ns: Vec<_> = (self.empty.iter())
                    .filter(|e| e.0 == d.index)
                    .map(|e| e.1 + 1)
                    .collect();
                let page = if ns.len() == 1 { "page" } else { "pages" };
                let letter = if several {
                    d.letter() + " "
                } else {
                    String::new()
                };
                (!ns.is_empty()).then(|| format!("{letter}{page} {}", numbers(&ns)))
            })
            .collect();
        if !empty.is_empty() {
            let have = if self.empty.len() == 1 { "has" } else { "have" };
            line += &format!("; {} {have} no text", listed(&empty));
        }
        line + ".\n"
    }
}

impl Piece {
    /// The tool result the model reads: the text with a place at each page start, or "already
    /// above" for what `visible` holds, what remains, and the room left in tokens. Render it
    /// before adding the piece to `visible`.
    pub fn render(&self, docs: &[Doc], visible: &Visible, room_tokens: usize) -> String {
        let Some(doc) = doc_of(docs, self.range.doc) else {
            return format!("that document is not open.\n{}", room_line(room_tokens));
        };
        let several = docs.len() > 1;
        let seen = visible.already(self.range);
        let mut out = if self.range.start == self.range.end {
            format!("{} has no text in the extracted text.\n", self.name)
        } else if seen == [self.range] {
            format!("{} already above.\n", above(doc, self.range))
        } else {
            let (given, _) = pages_of(doc, self.range);
            let mut out = match (pages(&self.name), self.range == self.part) {
                (Some(_), _) => {
                    let within = map::section(doc, self.range.start).map_or(String::new(), |s| {
                        format!(
                            ", in {}, which runs to page {}",
                            s.title,
                            span(doc, s.range).1
                        )
                    });
                    format!("{given} of {}{within}:\n", doc.pages.len())
                }
                (None, true) => format!("all of {}, {given}:\n", self.name),
                (None, false) => format!(
                    "{given} of {}, which runs to page {}:\n",
                    self.name,
                    span(doc, self.part).1
                ),
            };
            let mut at = self.range.start;
            let end = range(doc, self.range.end, self.range.end);
            for s in seen.iter().chain([&end]) {
                text(&mut out, doc, several, at, s.start);
                if s.start < s.end {
                    out += &format!("[{} already above]\n", above(doc, *s));
                }
                at = s.end;
            }
            out
        };
        if let Some(n) = self.next {
            out += &format!("more: read({:?}, from={n})\n", self.name);
        }
        out + &room_line(room_tokens)
    }
}

/// `r` as already-above text names it: "page 12 is", "pages 12 to 14 are", "part of page 12 is".
fn above(doc: &Doc, r: Range) -> String {
    let (pages, more) = pages_of(doc, r);
    let whole =
        doc.pages.iter().any(|p| p.start == r.start) && doc.pages.iter().any(|p| p.end == r.end);
    match (whole, more) {
        (true, true) => format!("{pages} are"),
        (true, false) => format!("{pages} is"),
        (false, _) => format!("part of {pages} is"),
    }
}

/// Appends `a..b` of `doc`'s text with its place first and at each page that starts in it.
fn text(out: &mut String, doc: &Doc, several: bool, a: usize, b: usize) {
    let starts = (doc.pages.iter())
        .filter(|p| a < p.start && p.start < b && p.start < p.end)
        .map(|p| p.start);
    let cuts: Vec<_> = [a].into_iter().chain(starts).chain([b]).collect();
    // The form feed between two pages is no text of either.
    for w in cuts
        .windows(2)
        .filter(|w| !doc.text[w[0]..w[1]].trim().is_empty())
    {
        *out += &place(doc, several, range(doc, w[0], w[0] + 1));
        out.push('\n');
        out.push_str(&doc.text[w[0]..w[1]].replace('\x0c', "\n"));
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::slice::from_ref;

    use super::super::Rep;
    use super::*;

    /// A document of `pages`, joined with form feeds as extracted PDF text is.
    pub(crate) fn paged(index: usize, pages: &[&str]) -> Doc {
        let (mut text, mut ranges) = (String::new(), Vec::new());
        for (i, p) in pages.iter().enumerate() {
            if i > 0 {
                text.push('\x0c');
            }
            ranges.push(Page {
                start: text.len(),
                end: text.len() + p.len(),
                label: None,
            });
            text.push_str(p);
        }
        Doc {
            index,
            name: format!("{index}.pdf"),
            sha256: String::new(),
            rep: Rep::Pdf,
            text,
            pages: ranges,
        }
    }

    fn thermal() -> Doc {
        paged(
            0,
            &[
                "Intro\nThe Thermal\nlimit is set here.\n",
                "Nothing.\nTHERMAL   LIMIT twice: thermal limit.\n",
                "",
                "Last thermal\n  limit.\n",
            ],
        )
    }

    #[test]
    fn find_counts_every_match_and_pages_through_them() {
        let d = thermal();
        let all = find(from_ref(&d), "thermal limit", 0, 10_000);
        assert_eq!((all.total, all.hits.len(), all.next), (4, 3, None));
        assert_eq!(
            all.hits.iter().map(|h| h.page).collect::<Vec<_>>(),
            [0, 1, 3]
        );
        // Across a line break, the hit is both lines.
        let h = &all.hits[0];
        assert_eq!(
            &d.text[h.range.start..h.range.end],
            "The Thermal\nlimit is set here."
        );
        // A room for one hit gives one, and the cursor gives the rest, each once.
        let one = find(from_ref(&d), "thermal  LIMIT", 0, 50);
        assert_eq!((one.hits.len(), one.next), (1, Some(1)));
        let rest = find(from_ref(&d), "thermal limit", 1, 10_000);
        assert_eq!((rest.hits.len(), rest.next), (2, None));
        let text = one.render(from_ref(&d), "thermal  LIMIT", 900);
        assert!(
            text.starts_with("4 matches; showing 1-1; more: find(\"thermal LIMIT\", from=1)\n")
        );
        assert!(text.contains("\n[p. 1] The Thermal limit is set here.\n"));
        assert!(text.contains("searched all 4 pages; page 3 has no text.\n"));
        assert!(text.ends_with("\nRoom left: about 900 tokens."));
        // The second page's line holds two matches: one hit, matches 2 and 3.
        assert!(
            rest.render(&[d], "thermal limit", 9)
                .starts_with("4 matches; showing 2-4\n")
        );
    }

    #[test]
    fn find_with_no_match_says_so_and_names_its_places_by_document() {
        let (a, b) = (thermal(), paged(1, &["one", "the thermal limit"]));
        let none = find(from_ref(&a), "thermal limits exceeded", 0, 1000);
        assert_eq!((none.total, none.next), (0, None));
        let text = none.render(from_ref(&a), "thermal limits exceeded", 5);
        assert!(text.starts_with(
            "no literal matches in the extracted text; this does not mean the idea is absent"
        ));
        let both = find(&[a.clone(), b.clone()], "thermal limit", 0, 10_000);
        assert_eq!(both.total, 5);
        let text = both.render(&[a, b], "thermal limit", 5);
        assert!(text.contains("\n[B, p. 2] the thermal limit\n"), "{text}");
        assert!(text.contains("searched all 6 pages of A and B; A page 3 has no text.\n"));
    }

    #[test]
    fn a_long_line_is_cut_around_its_match_at_word_ends() {
        let long = format!("{} needle {}", "été ".repeat(200), "über ".repeat(200));
        let d = paged(0, &[&long]);
        let f = find(from_ref(&d), "NEEDLE", 0, 10_000);
        let h = &f.hits[0];
        let text = &d.text[h.range.start..h.range.end];
        assert!(
            text.len() <= LINE_BYTES && text.contains("needle"),
            "{text}"
        );
        assert!(text.starts_with("été") && text.ends_with("über"), "{text}");
        assert!(f.render(&[d], "needle", 1).contains("[p. 1] …été"));
    }

    #[test]
    fn read_names_pages_as_a_model_writes_them() {
        let d = thermal();
        let part = |p: &str| read(&d, &[], p, None, 10_000).map(|r| (r.part.start, r.part.end));
        let (p2, p23) = (
            Ok((d.pages[1].start, d.pages[1].end)),
            Ok((d.pages[1].start, d.pages[2].end)),
        );
        for p in ["page 2", "p. 2", "P.2", "p 2"] {
            assert_eq!(part(p), p2, "{p}");
        }
        for p in ["pages 2-3", "pp. 2-3", "Pages 2 to 3", "pp. 3\u{2013}2"] {
            assert_eq!(part(p), p23, "{p}");
        }
        let err = part("page 9").unwrap_err();
        assert!(err.contains("the document has 4 pages, 1 to 4"), "{err}");
        let r = read(&d, &[], "pages 1-2", None, 10_000).unwrap();
        let text = r.render(from_ref(&d), &Visible::default(), 7);
        assert!(
            text.starts_with("pages 1 to 2 of 4:\n[p. 1]\nIntro\n"),
            "{text}"
        );
        assert!(text.contains("\n[p. 2]\nNothing.\n"), "{text}");
        let blank = read(&d, &[], "page 3", None, 10_000).unwrap();
        assert!(
            blank
                .render(&[d], &Visible::default(), 7)
                .starts_with("page 3 has no text")
        );
    }

    fn sections() -> (Doc, Vec<(String, Range)>) {
        let body = "word ".repeat(30) + "\nend of line\n" + &"ünï ".repeat(40);
        let d = paged(0, &["Front\n", &body, &body, "After\n"]);
        let labels = vec![
            (
                "Section 4 Results".to_string(),
                range(&d, d.pages[1].start, d.pages[2].end),
            ),
            (
                "Appendix B".to_string(),
                range(&d, d.pages[3].start, d.pages[3].end),
            ),
        ];
        (d, labels)
    }

    #[test]
    fn a_long_part_comes_in_pieces_at_word_ends_and_none_is_a_repeat() {
        let (d, labels) = sections();
        let mut seen = Visible::default();
        let (mut from, mut joined) = (None, String::new());
        let first = read(&d, &labels, "section 4 RESULTS", None, 100).unwrap();
        loop {
            let p = read(&d, &labels, "Section 4 Results", from, 100).unwrap();
            assert!(p.range.end - p.range.start <= 100);
            let text = &d.text[p.range.start..p.range.end];
            // A piece ends at a line or a word's end.
            assert!(
                p.next.is_none() || text.ends_with(char::is_whitespace),
                "{text:?}"
            );
            assert!(!p.render(from_ref(&d), &seen, 1).contains("already above"));
            seen.seen(p.range);
            joined += text;
            from = p.next;
            if from.is_none() {
                break;
            }
        }
        assert_eq!(joined, d.text[first.part.start..first.part.end]);
        let text = first.render(from_ref(&d), &Visible::default(), 3);
        assert!(
            text.starts_with("page 2 of Section 4 Results, which runs to page 3:\n"),
            "{text}"
        );
        assert!(text.contains(&format!(
            "more: read(\"Section 4 Results\", from={})",
            first.next.unwrap()
        )));
        let all = read(&d, &labels, "appendix b", None, 1000).unwrap();
        assert!(
            all.render(&[d], &seen, 3)
                .starts_with("all of Appendix B, page 4:\n")
        );
    }

    #[test]
    fn a_missing_part_gets_the_nearest_ones() {
        let (d, labels) = sections();
        assert_eq!(
            read(&d, &labels, "results", None, 9).unwrap().name,
            "Section 4 Results"
        );
        let err = read(&d, &labels, "Sektion 4", None, 9).unwrap_err();
        assert!(err.starts_with("no part named \"Sektion 4\"; the nearest: \"Section 4 Results\", \"Appendix B\"; or pages 1 to 4"), "{err}");
        let err = read(&d, &[], "Methods", None, 9).unwrap_err();
        assert!(err.contains("read pages by number, 1 to 4"), "{err}");
        let r = read(&d, &labels, "Appendix B", None, 9).unwrap();
        assert!(read(&d, &labels, "Appendix B", Some(r.part.end), 9).is_err());
    }

    #[test]
    fn already_above_is_judged_on_exact_ranges_and_gives_only_what_is_new() {
        let d = thermal();
        let page = |n: usize| range(&d, d.pages[n].start, d.pages[n].end);
        let mut seen = Visible::default();
        seen.seen(page(0));
        let two = read(&d, &[], "pages 1-2", None, 10_000).unwrap();
        let text = two.render(from_ref(&d), &seen, 1);
        assert!(
            text.contains("[page 1 is already above]\n[p. 2]\nNothing."),
            "{text}"
        );
        assert!(!text.contains("Intro"));
        let one = read(&d, &[], "p. 1", None, 10_000).unwrap();
        assert!(
            one.render(from_ref(&d), &seen, 1)
                .starts_with("page 1 is already above.\n")
        );
        // Part of a page seen is not the whole page.
        let mut part = Visible::default();
        part.seen(range(&d, 0, 6));
        let text = one.render(from_ref(&d), &part, 1);
        assert!(
            text.contains("[part of page 1 is already above]\n[p. 1]\nThe Thermal"),
            "{text}"
        );
        // A reset brings the text back.
        seen.clear();
        assert!(one.render(&[d], &seen, 1).contains("Intro"));
    }
}
