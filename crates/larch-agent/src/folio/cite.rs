//! Citations in a reader's answer and the check of each quote against the span it cites.

use std::{cell::OnceCell, sync::LazyLock};

use regex::{Captures, Regex};

use super::{Doc, Range};
use crate::skills::{ask_docs, read_papers, web_answer};

/// What one quote's citation came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CiteResult {
    /// The quote is on the cited span.
    Matched,
    /// The quote is in no document.
    NotFound,
    /// The quote is in the text, but not where it was cited: this range is the first place.
    FoundElsewhere(Range),
}

/// One quote of an answer, and what its citation came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verdict {
    /// The quote, as the answer wrote it.
    pub quote: String,
    /// The position in the documents of the one the citation names; none when it names none.
    pub doc: Option<usize>,
    /// The place the citation names, as a note says it ("on p. 12").
    pub place: String,
    /// What the citation came to, or why it could not be checked ("no page given").
    pub result: Result<CiteResult, String>,
    /// Where to look for a checked quote: its own range when it was found (on the cited span, or
    /// elsewhere), else the span its citation names. None when it was not checked.
    pub at: Option<Range>,
}

/// The result of checking an answer's quotes: one verdict each, the notes for the user built from
/// them, and the counts the trace keeps. `ran` is false when there was nothing the check could
/// run on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Checked {
    /// The verdict of each quote of the answer, in order.
    pub results: Vec<Verdict>,
    /// One note per quote that was not matched on its cited span, or had no citation.
    pub notes: Vec<String>,
    /// Quotes with a citation.
    pub cited: usize,
    /// Of those, quotes matched on their cited span.
    pub matched: usize,
    /// The check ran on at least one cited quote.
    pub ran: bool,
}

/// A citation as the reader writes it, 1-based pages or printed labels, or a part by its label:
/// "[p. 12]", "[pp. 12-13]", "[B, p. iv]", "[§4.1]", "[Section 4.1]", "[B, Table 3]". What
/// follows a page after a comma is the section the reader names ("[p. 12, Section 3]"): the page
/// is the span.
static CITE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"\[(?:([A-Z]{1,2}),\s*)?(?:pp?\.\s*([0-9A-Za-z]+)(?:\s*[\-\u{2013}]\s*([0-9A-Za-z]+))?(?:,[^\[\]]*)?|(§\s*[0-9A-Z][0-9.]*[a-z]?|[A-Z][a-z]+\s+(?:[0-9]+|[A-Z])[0-9.]*[a-z]?))\]",
    )
    .expect("a valid pattern")
});

/// Signs that make a line a formula's, where a hyphen at its end is a minus sign.
pub(super) const FORMULA: [char; 8] = ['=', '+', '\u{2212}', '\u{d7}', '^', '_', '\\', '|'];

/// Checks each quote of `answer` against the span its citation names in `docs` (`sections` are
/// headings and labels with their ranges), with formula characters kept: one verdict each, and
/// the notes built from them. The answer itself is never changed.
pub fn check(answer: &str, docs: &[Doc], sections: &[(String, Range)]) -> Checked {
    let mut out = Checked::default();
    // Made once, and only when a quote cites a place.
    let texts = OnceCell::new();
    let texts = || -> &Vec<[(String, Vec<usize>); 2]> {
        texts.get_or_init(|| {
            (docs.iter())
                .map(|d| [compared(&d.text, true), compared(&d.text, false)])
                .collect()
        })
    };
    for (quote, after) in web_answer::quoted(answer, web_answer::QUOTE_CHARS) {
        // A quote across paragraphs may carry the block's page marks: they are not the page's words.
        // Markdown emphasis is the answer's, not the page's; an ellipsis is words left out.
        let bare = CITE
            .replace_all(quote, " ")
            .replace("**", "")
            .replace("__", "");
        let wanted = compared(bare.trim_matches(['*', '_']), true).0;
        let wanted: Vec<&str> = (wanted.split('\u{2026}'))
            .flat_map(|p| p.split("..."))
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .collect();
        if wanted.is_empty() {
            continue;
        }
        // The cite right after the quote (past bold or italic marks), else right before it.
        let emphasis = ['*', '_'];
        let after = after.trim_start().trim_start_matches(emphasis).trim_start();
        let at = quote.as_ptr() as usize - answer.as_ptr() as usize;
        let before = (answer[..at].trim_end_matches(['"', '\u{201c}', '\u{201d}']))
            .trim_end()
            .trim_end_matches(emphasis)
            .trim_end();
        let cite = (CITE.captures(after))
            .filter(|c| c.get(0).is_some_and(|m| m.start() == 0))
            .or_else(|| {
                (CITE.captures_iter(before).last())
                    .filter(|c| c.get(0).is_some_and(|m| m.end() == before.len()))
            });
        let mut verdict = Verdict {
            quote: quote.trim().to_owned(),
            doc: None,
            place: String::new(),
            result: Err("no page given".into()),
            at: None,
        };
        if let Some(c) = cite {
            out.cited += 1;
            verdict.place = place(&c);
            match spans(&c, docs, sections) {
                Err(why) => verdict.result = Err(why),
                Ok((d, spans)) => {
                    out.ran = true;
                    let texts = texts();
                    let first = |i: usize| hits(&texts[i], &wanted).min();
                    // `z` is where the last character starts: the range ends after the whole of
                    // it, which may be more than a byte (€, é, CJK), so it slices the text.
                    let range = |i: usize, (a, z): (usize, usize)| Range {
                        doc: docs[i].index,
                        start: a,
                        end: docs[i].text.ceil_char_boundary(z + 1),
                    };
                    let inside = (hits(&texts[d], &wanted))
                        .find(|&(a, z)| spans.iter().any(|&(s, e)| s <= a && z < e));
                    let elsewhere = || {
                        (0..docs.len())
                            .filter(|&i| i != d)
                            .find_map(|i| Some(range(i, first(i)?)))
                    };
                    verdict.doc = Some(d);
                    let found = match (inside, first(d)) {
                        (Some(_), _) => CiteResult::Matched,
                        (None, Some(hit)) => CiteResult::FoundElsewhere(range(d, hit)),
                        (None, None) => {
                            elsewhere().map_or(CiteResult::NotFound, CiteResult::FoundElsewhere)
                        }
                    };
                    verdict.at = Some(match (found, inside) {
                        (CiteResult::FoundElsewhere(r), _) => r,
                        (_, Some(hit)) => range(d, hit),
                        (_, None) => range(d, (spans[0].0, spans[0].1 - 1)),
                    });
                    verdict.result = Ok(found);
                }
            }
        }
        out.matched += usize::from(verdict.result == Ok(CiteResult::Matched));
        if let Some(note) = note(&verdict, docs)
            && !out.notes.contains(&note)
        {
            out.notes.push(note);
        }
        out.results.push(verdict);
    }
    out
}

/// The note on `v`, or none when its quote is on the span it cites.
fn note(v: &Verdict, docs: &[Doc]) -> Option<String> {
    let several = docs.len() > 1;
    let why = match &v.result {
        Ok(CiteResult::Matched) => return None,
        Err(why) => format!("not checked: {why}"),
        Ok(CiteResult::NotFound) if several => {
            format!("not found in {}", docs[v.doc?].letter())
        }
        Ok(CiteResult::NotFound) => "not found".into(),
        Ok(CiteResult::FoundElsewhere(r)) => {
            let cited = &docs[v.doc?];
            let doc = docs.iter().find(|d| d.index == r.doc).unwrap_or(cited);
            match doc.index == cited.index {
                true => format!(
                    "found {}, not {}",
                    at(doc, r.start, r.end - 1, several),
                    v.place
                ),
                false => format!(
                    "found {}, not in {}",
                    at(doc, r.start, r.end - 1, true),
                    cited.letter()
                ),
            }
        }
    };
    Some(format!("_Quote \"{}\": {why}._", v.quote))
}

/// The spans the citations in `text` name in `docs`, as `check` reads them; one that names
/// none is left out.
pub fn cited(text: &str, docs: &[Doc], sections: &[(String, Range)]) -> Vec<Range> {
    (CITE.captures_iter(text))
        .filter_map(|c| spans(&c, docs, sections).ok())
        .flat_map(|(d, spans)| {
            let doc = docs[d].index;
            spans
                .into_iter()
                .map(move |(start, end)| Range { doc, start, end })
        })
        .collect()
}

/// The note on the numbers `answer` gives that no document's text has, worded as only that: a
/// number found is not proven to be the right one. `None` when each is found.
pub fn numbers(answer: &str, docs: &[Doc]) -> Option<String> {
    let texts: Vec<&str> = docs.iter().map(|d| d.text.as_str()).collect();
    // A citation's numbers are places, not numbers the answer gives.
    let missing = read_papers::unchecked(&CITE.replace_all(answer, ""), &texts.join("\n"));
    let (s, _) = ask_docs::plural(missing.len(), "", "");
    (!missing.is_empty()).then(|| {
        format!(
            "_Number{s} not found anywhere in the text: {}._",
            missing.join(", ")
        )
    })
}

/// The document (its position in `docs`) and the byte ranges citation `c` names, or why it
/// names none it can be checked on.
fn spans(
    c: &Captures,
    docs: &[Doc],
    sections: &[(String, Range)],
) -> Result<(usize, Vec<(usize, usize)>), String> {
    let cite = &c[0];
    let d = match c.get(1) {
        Some(letter) => docs.iter().position(|d| d.letter() == letter.as_str()),
        None if docs.len() == 1 => Some(0),
        None => return Err(format!("{cite} names no document")),
    };
    let d = d.ok_or_else(|| format!("cannot find {cite}"))?;
    let doc = &docs[d];
    let spans: Vec<(usize, usize)> = match (c.get(2), c.get(4)) {
        (Some(from), _) => {
            // A number is the page's place, from 1; anything else is the label it prints.
            let page = |x: &str| match x.parse::<usize>() {
                Ok(n) => n.checked_sub(1).filter(|&i| i < doc.pages.len()),
                Err(_) => (doc.pages.iter())
                    .position(|p| (p.label.as_deref()).is_some_and(|l| l.eq_ignore_ascii_case(x))),
            };
            let to = c.get(3).unwrap_or(from);
            match (page(from.as_str()), page(to.as_str())) {
                (Some(a), Some(z)) if a <= z => vec![(doc.pages[a].start, doc.pages[z].end)],
                _ => Vec::new(),
            }
        }
        (None, Some(label)) => {
            let label = part(label.as_str());
            (sections.iter())
                .filter(|(name, r)| {
                    let rest = part(name);
                    r.doc == doc.index
                        && (rest.strip_prefix(&label))
                            .is_some_and(|rest| !rest.starts_with(char::is_alphanumeric))
                })
                .map(|(_, r)| (r.start, r.end))
                .collect()
        }
        (None, None) => Vec::new(),
    };
    if spans.is_empty() {
        return Err(format!("cannot find {cite}"));
    }
    // A page with no text (a scan) proves nothing either way.
    if (spans.iter()).all(|&(s, e)| doc.text.get(s..e).is_none_or(|t| t.trim().is_empty())) {
        return Err(format!("{cite} has no text"));
    }
    Ok((d, spans))
}

/// A part's label as a citation and a heading are compared: lower case, without "§" or
/// "section" before it or a full stop after it ("4.1", "table 3").
fn part(label: &str) -> String {
    let l = label.trim().to_lowercase();
    let l = (l.strip_prefix('§').or_else(|| l.strip_prefix("section"))).unwrap_or(&l);
    l.trim().trim_end_matches('.').to_string()
}

/// The span citation `c` names, as a note says it: "on pp. 12-13", "in Table 3".
fn place(c: &Captures) -> String {
    match (c.get(2), c.get(3), c.get(4)) {
        (Some(a), Some(z), _) => format!("on pp. {}-{}", a.as_str(), z.as_str()),
        (Some(a), None, _) => format!("on p. {}", a.as_str()),
        (_, _, label) => format!("in {}", label.map_or("", |l| l.as_str())),
    }
}

/// Where bytes `a` to `z` of `doc` are: "on p. 3", "on pp. 3-4", and "in B on p. 3" when
/// `named`.
fn at(doc: &Doc, a: usize, z: usize, named: bool) -> String {
    let (a, z) = (doc.page_of(a) + 1, doc.page_of(z) + 1);
    let pages = match a == z {
        true => format!("p. {a}"),
        false => format!("pp. {a}-{z}"),
    };
    match named {
        true => format!("in {} on {pages}", doc.letter()),
        false => format!("on {pages}"),
    }
}

/// The most bytes an ellipsis in a quote stands for: its parts must be this close on the page.
const ELLIPSIS_GAP: usize = 2_000;

/// Each place `quote` is in a text as `compared` made it, both ways, as its first and last byte
/// in the text's own offsets. A quote of several parts (an ellipsis between them) is there when
/// they follow each other in order, each within `ELLIPSIS_GAP` of the last.
fn hits<'a>(
    texts: &'a [(String, Vec<usize>)],
    quote: &'a [&'a str],
) -> impl Iterator<Item = (usize, usize)> + 'a {
    (texts.iter()).flat_map(move |(text, from)| {
        (text.match_indices(quote[0])).filter_map(move |(i, _)| {
            let mut end = i + quote[0].len();
            for part in &quote[1..] {
                let window = &text[end..text.floor_char_boundary(end + ELLIPSIS_GAP)];
                end += window.find(part)? + part.len();
            }
            Some((from[i], from[end - 1]))
        })
    })
}

/// `text` as a quote is compared with it: curly quote marks straight, each run of whitespace
/// and line breaks one space, the ends trimmed, and a word that a hyphen breaks at a line end
/// joined, with the hyphen dropped when `drop` and kept when not ("infor-mation" and
/// "state-of-the-art" both read either way). Letters are lower case and a PDF's ligatures
/// (U+FB00 to U+FB06) their letters, so a quote that starts its sentence anew or a page whose
/// "ffi" is one character still matches (#590 F9). For each byte of the result, the offset in
/// `text` it came from.
fn compared(text: &str, drop: bool) -> (String, Vec<usize>) {
    let (mut out, mut from) = (String::new(), Vec::new());
    let (mut gap, mut glued, mut at) = (false, false, 0);
    // A page break ends a line too.
    let mut lines = text.split_inclusive(['\n', '\x0c']).peekable();
    while let Some(line) = lines.next() {
        let start = at;
        at += line.len();
        let next = lines.peek().copied().unwrap_or("");
        let body = line.trim_end();
        // Only between two letters: before a digit, or on either line of a formula, the hyphen
        // is a minus sign or a range.
        let joins = (body.strip_suffix('-').and_then(|w| w.chars().next_back()))
            .is_some_and(char::is_alphabetic)
            && next.trim_start().starts_with(char::is_alphabetic)
            && !line.contains(FORMULA)
            && !next.contains(FORMULA);
        let body = match joins && drop {
            true => &body[..body.len() - 1],
            false => body,
        };
        // The rest of a broken word starts where its line's indent ends.
        let lead = match glued {
            true => body.len() - body.trim_start().len(),
            false => 0,
        };
        for (i, c) in body[lead..].char_indices() {
            let i = start + lead + i;
            if c.is_whitespace() {
                gap = !out.is_empty();
                continue;
            }
            if gap {
                out.push(' ');
                from.push(i);
                gap = false;
            }
            match c {
                '\u{201c}' | '\u{201d}' => out.push('"'),
                '\u{2018}' | '\u{2019}' => out.push('\''),
                c => match ligature(c) {
                    Some(letters) => out.push_str(letters),
                    None => out.extend(c.to_lowercase()),
                },
            }
            from.resize(out.len(), i);
        }
        glued = joins;
        if !joins {
            gap = !out.is_empty();
        }
    }
    (out, from)
}

/// The letters of a Latin ligature a PDF's text may carry as one character.
fn ligature(c: char) -> Option<&'static str> {
    Some(match c {
        '\u{fb00}' => "ff",
        '\u{fb01}' => "fi",
        '\u{fb02}' => "fl",
        '\u{fb03}' => "ffi",
        '\u{fb04}' => "ffl",
        '\u{fb05}' | '\u{fb06}' => "st",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::folio::{Page, Rep};

    const QUOTE: usize = web_answer::QUOTE_CHARS;

    /// Document `index`, one page per string, with a page break between.
    fn doc(index: usize, pages: &[&str]) -> Doc {
        let (mut text, mut spans) = (String::new(), Vec::new());
        for p in pages {
            let start = text.len();
            text.push_str(p);
            spans.push(Page {
                start,
                end: text.len(),
                label: None,
            });
            text.push('\x0c');
        }
        Doc {
            index,
            name: "a.pdf".into(),
            sha256: String::new(),
            rep: Rep::Pdf,
            text,
            pages: spans,
        }
    }

    /// `c` without its verdicts, which `each_quote_has_its_verdict` reads.
    fn without(c: Checked) -> Checked {
        Checked {
            results: vec![],
            ..c
        }
    }

    fn notes(answer: &str, docs: &[Doc]) -> Vec<String> {
        check(answer, docs, &[]).notes
    }

    /// True quotes the answer writes its own way still match their page: a lower-case first
    /// letter, an ellipsis, a PDF ligature, bold inside or around the marks, and a cite before the
    /// quote (#590 F9). A quote not on the page is still not found.
    #[test]
    fn a_true_quote_written_the_answers_way_matches() {
        let a = [doc(
            0,
            &[
                "Intro.",
                "The Transformer allows for significantly more parallelization. The e\u{fb03}cient \
                 model reaches a new state of the art.",
            ],
        )];
        let matched = |answer: &str| check(answer, &a, &[]).matched;
        for answer in [
            r#""the Transformer allows for significantly more parallelization" [p. 2]"#,
            r#""The Transformer allows for ... more parallelization" [p. 2]"#,
            "\"The Transformer allows for \u{2026} more parallelization\" [p. 2]",
            r#""The efficient model reaches" [p. 2]"#,
            r#""**reaches a new state of the art**" [p. 2]"#,
            r#"**"reaches a new state of the art"** [p. 2]"#,
            r#"[p. 2] "reaches a new state of the art""#,
        ] {
            assert_eq!(matched(answer), 1, "{answer}");
        }
        assert_eq!(matched(r#""reaches a new state of the sea" [p. 2]"#), 0);
        // An ellipsis's parts out of order, or the cite on the wrong page, are no match.
        assert_eq!(
            matched(r#""more parallelization ... The Transformer" [p. 2]"#),
            0
        );
        assert_eq!(matched(r#""reaches a new state of the art" [p. 1]"#), 0);
    }

    /// A quote that ends in a character of more than one byte (€, a CJK one) ranges over the
    /// whole of it, matched on its page or found elsewhere: `start..end` slices the text and ends
    /// with that character, so the page view can mark it (#590 Mac review).
    #[test]
    fn a_quote_ending_in_a_wide_character_ranges_over_all_of_it() {
        let a = [doc(
            0,
            &[
                "Intro.",
                "The rent is 1200 € a month. 東京は日本の首都で、大きな都市です。",
            ],
        )];
        for (answer, last, result) in [
            ("\"The rent is 1200 €\" [p. 2]", '€', CiteResult::Matched),
            (
                "\"東京は日本の首都で、大きな都市\" [p. 2]",
                '市',
                CiteResult::Matched,
            ),
        ] {
            let c = check(answer, &a, &[]);
            assert_eq!(c.results[0].result, Ok(result), "{answer}");
            let r = c.results[0].at.expect("checked");
            let text = &a[0].text;
            assert!(text.is_char_boundary(r.end), "{answer}: {r:?}");
            assert!(text[r.start..r.end].ends_with(last), "{answer}: {r:?}");
        }
        // Cited on the wrong page: the range it was found at instead.
        let c = check("\"The rent is 1200 €\" [p. 1]", &a, &[]);
        let Ok(CiteResult::FoundElsewhere(r)) = c.results[0].result else {
            panic!("{:?}", c.results[0]);
        };
        assert!(a[0].text[r.start..r.end].ends_with('€'), "{r:?}");
    }

    #[test]
    fn a_page_citation_survives_the_quote_mark_parser() {
        assert_eq!(
            web_answer::quoted(r#""the model learns fast" [p. 12]."#, QUOTE),
            [("the model learns fast", " [p. 12].")]
        );
    }

    #[test]
    fn each_quote_has_its_verdict_and_its_note_is_built_from_it() {
        let a = [
            doc(0, &["Intro.", "The model learns fast."]),
            doc(1, &["Other.", "The rate is 4%."]),
        ];
        let said = r#""The model learns fast" [A, p. 2], "The model learns" [A, p. 1], "The rate is 4%" [A, p. 2], "no such words" [A, p. 1] and "The model learns fast" [B]."#;
        let c = check(said, &a, &[]);
        let results: Vec<_> = c.results.iter().map(|v| v.result.clone()).collect();
        let at = |doc, start, end| Range { doc, start, end };
        assert_eq!(
            results,
            [
                Ok(CiteResult::Matched),
                Ok(CiteResult::FoundElsewhere(at(0, 7, 23))),
                Ok(CiteResult::FoundElsewhere(at(1, 7, 21))),
                Ok(CiteResult::NotFound),
                Err("no page given".into()),
            ]
        );
        // Matched: the quote's own bytes; not found: the cited page; unchecked: none.
        let places: Vec<_> = c.results.iter().map(|v| v.at).collect();
        assert_eq!(
            places,
            [
                Some(at(0, 7, 28)),
                Some(at(0, 7, 23)),
                Some(at(1, 7, 21)),
                Some(at(0, 0, 6)),
                None
            ]
        );
        assert_eq!(c.notes.len(), 4, "{:?}", c.notes);
        assert!(
            c.notes[1].ends_with("found in B on p. 2, not in A._"),
            "{:?}",
            c.notes
        );
    }

    #[test]
    fn a_quote_is_matched_on_its_cited_page_only() {
        let a = [doc(
            0,
            &["Intro.", "The model 'learns'\n  fast on data.", "End."],
        )];
        assert_eq!(
            without(check(
                "It says \u{201c}The model \u{2018}learns\u{2019} fast\u{201d} [p. 2].",
                &a,
                &[]
            )),
            Checked {
                results: vec![],
                notes: vec![],
                cited: 1,
                matched: 1,
                ran: true
            }
        );
        let said =
            r#""The model 'learns'" [p. 3], "The model unlearns" [p. 2] and "fast on data.""#;
        assert_eq!(
            without(check(said, &a, &[])),
            Checked {
                results: vec![],
                notes: vec![
                    "_Quote \"The model 'learns'\": found on p. 2, not on p. 3._".into(),
                    "_Quote \"The model unlearns\": not found._".into(),
                    "_Quote \"fast on data.\": not checked: no page given._".into(),
                ],
                cited: 2,
                matched: 0,
                ran: true
            }
        );
        // No cited quote: the check did not run.
        assert!(!check(r#""fast on data." [p. 9]"#, &a, &[]).ran);
    }

    #[test]
    fn pipes_and_stars_are_kept() {
        let a = [doc(0, &["Then the norm x is small, and *b* is too."])];
        assert_eq!(
            notes(r#""the norm |x| is small" [p. 1]"#, &a),
            ["_Quote \"the norm |x| is small\": not found._"]
        );
        assert_eq!(notes(r#""and *b* is too." [p. 1]"#, &a), [""; 0]);
        assert_eq!(notes(r#""and b is too." [p. 1]"#, &a).len(), 1);
    }

    #[test]
    fn a_hyphen_at_a_line_end_joins_two_letters_only() {
        let a = [doc(
            0,
            &[
                "The infor-\n  mation is kept. A state-of-the-\nart model.\nSo y = a-\nb holds; the range x-\n2 is open.",
            ],
        )];
        assert_eq!(notes(r#""The information is kept" [p. 1]"#, &a), [""; 0]);
        assert_eq!(notes(r#""A state-of-the-art model" [p. 1]"#, &a), [""; 0]);
        // A minus sign at a formula's line end, and a hyphen before a digit, are not joined.
        assert_eq!(notes(r#""So y = ab holds" [p. 1]"#, &a).len(), 1);
        assert_eq!(notes(r#""the range x2 is open" [p. 1]"#, &a).len(), 1);
    }

    #[test]
    fn the_same_quote_in_two_documents_is_checked_in_the_one_cited() {
        let docs = [
            doc(0, &["A shared sentence here."]),
            doc(1, &["Other text.", "A shared sentence here."]),
        ];
        assert_eq!(notes(r#""A shared sentence" [B, p. 2]"#, &docs), [""; 0]);
        assert_eq!(
            notes(r#""A shared sentence" [B, p. 1]"#, &docs),
            ["_Quote \"A shared sentence\": found in B on p. 2, not on p. 1._"]
        );
    }

    #[test]
    fn a_quote_cited_to_the_wrong_document_says_where_it_is() {
        let docs = [
            doc(0, &["x", "y", "Only A says this."]),
            doc(1, &["Other text."]),
        ];
        let n = |said: &str| notes(said, &docs);
        assert_eq!(
            n(r#""Only A says this" [B, p. 1]"#),
            ["_Quote \"Only A says this\": found in A on p. 3, not in B._"]
        );
        assert_eq!(
            n(r#""Nobody says this" [B, p. 1]"#),
            ["_Quote \"Nobody says this\": not found in B._"]
        );
        assert_eq!(
            n(r#""Only A says this" [p. 3]"#),
            ["_Quote \"Only A says this\": not checked: [p. 3] names no document._"]
        );
        assert_eq!(
            n(r#""Only A says this" [C, p. 3]"#),
            ["_Quote \"Only A says this\": not checked: cannot find [C, p. 3]._"]
        );
    }

    /// A quote across two paragraphs may copy the page mark the block put between them; the mark
    /// is not the page's words, so the quote still holds.
    #[test]
    fn a_page_mark_inside_a_quote_is_not_part_of_it() {
        let a = [doc(0, &["Rates rose.\n\nThen they fell."])];
        let said = r#""Rates rose. [p. 1] Then they fell" [p. 1]"#;
        assert_eq!(check(said, &a, &[]).notes, [""; 0]);
    }

    #[test]
    fn printed_labels_page_ranges_and_parts_name_their_spans() {
        let mut a = doc(
            0,
            &[
                "Roman page text.",
                "Results end here",
                "and go on here.",
                "",
            ],
        );
        a.pages[0].label = Some("iv".into());
        let (start, end) = (a.pages[1].start, a.pages[2].end);
        let sections = [
            ("4.1 Results".to_string(), Range { doc: 0, start, end }),
            ("Table 3: Errors".to_string(), Range { doc: 0, start, end }),
        ];
        let a = [a];
        let n = |said: &str| check(said, &a, &sections).notes;
        assert_eq!(n(r#""Roman page text" [p. iv]"#), [""; 0]);
        let across = r#""Results end here and go on""#;
        assert_eq!(n(&format!("{across} [pp. 2-3]")), [""; 0]);
        assert_eq!(n(&format!("{across} [p. 2\u{2013}3]")), [""; 0]);
        assert_eq!(
            n(&format!("{across} [p. 2]")),
            ["_Quote \"Results end here and go on\": found on pp. 2-3, not on p. 2._"]
        );
        for part in ["§4.1", "Section 4.1", "Table 3"] {
            assert_eq!(n(&format!("{across} [{part}]")), [""; 0], "{part}");
        }
        assert_eq!(
            n(r#""Roman page text" [§4.1]"#),
            ["_Quote \"Roman page text\": found on p. 1, not in §4.1._"]
        );
        assert_eq!(
            n(r#""Roman page text" [Table 30]"#),
            ["_Quote \"Roman page text\": not checked: cannot find [Table 30]._"]
        );
        assert_eq!(
            n(r#""Roman page text" [p. 4]"#),
            ["_Quote \"Roman page text\": not checked: [p. 4] has no text._"]
        );
    }

    #[test]
    fn a_page_with_its_section_after_it_is_checked_on_the_page() {
        let one = [doc(0, &["Intro.", "The rate is 4% on the test set."])];
        assert_eq!(
            without(check(r#""The rate is 4%" [p. 2, Section 3.4]"#, &one, &[])),
            Checked {
                results: vec![],
                notes: vec![],
                cited: 1,
                matched: 1,
                ran: true
            }
        );
        // The section's numbers are places, not numbers the answer gives.
        assert_eq!(numbers("It is 4% [p. 2, Section 3.4].", &one), None);
        let two = [one[0].clone(), doc(1, &["Other.", "x", "The rate is 3%."])];
        assert_eq!(
            notes(r#""The rate is 3%" [B, p. 2, Table 2]"#, &two),
            ["_Quote \"The rate is 3%\": found in B on p. 3, not on p. 2._"]
        );
        let cited = cited("[B, pp. 2-3, Table 2] and [p. 9]", &two, &[]);
        assert_eq!(
            cited,
            [Range {
                doc: 1,
                start: two[1].pages[1].start,
                end: two[1].pages[2].end
            }]
        );
    }

    #[test]
    fn a_number_not_in_the_text_is_named_and_a_found_one_is_not_vouched_for() {
        let a = [doc(0, &["It scored 93.4 on the test."])];
        assert_eq!(
            numbers("It scored 93.4 and 85.2 [p. 12].", &a).as_deref(),
            Some("_Number not found anywhere in the text: 85.2._")
        );
        assert_eq!(numbers("It scored 93.4 [p. 12].", &a), None);
    }
}
