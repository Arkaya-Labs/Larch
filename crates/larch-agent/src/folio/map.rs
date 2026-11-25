//! A document's map, built by Rust with no model call (plan 7.2): its sections, pages or slides,
//! tables, figures and labeled items, each with its byte range in the text.

use std::{collections::HashSet, sync::LazyLock};

use regex::Regex;

use super::{Doc, Range, cite::FORMULA, tools};
use crate::skills::{
    ask_docs,
    read_papers::{self, Part, REFERENCES},
};

/// Bytes of a title the map keeps: about a heading line's.
const TITLE_BYTES: usize = 80;

/// Words a heading line holds after its label at most; a longer line starts a paragraph.
const TITLE_WORDS: usize = 12;

/// Bytes a slide holds at most; a paper's or a contract's page holds thousands.
const SLIDE_BYTES: usize = 1200;

/// Pages with text a slide deck has at least.
const MIN_SLIDES: usize = 3;

/// Headings the first read lists; past them, the top level alone.
const OUTLINE: usize = 40;

/// What a map entry is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A heading or a numbered section at its level, 1 the top: it runs to the next one at its
    /// level or above.
    Section(usize),
    /// A page.
    Page,
    /// A slide deck's page.
    Slide,
    /// A table: its caption and its rows.
    Table,
    /// A figure's caption.
    Figure,
    /// Another labeled item: a theorem, an equation, a footnote.
    Item,
}

/// One part of a document the map names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The name `read` and a citation give it: "Section 4.1", "Item 1A", "Table 3", "page 2".
    pub label: String,
    /// Its line as the text writes it ("4.1 Results"), cut to a heading's length.
    pub title: String,
    /// What it is.
    pub kind: Kind,
    /// Its bytes in the text.
    pub range: Range,
}

/// How the part a label word names runs.
#[derive(Clone, Copy, Debug)]
enum Runs {
    /// Its caption's block and the rows after it.
    Table,
    /// Its block, as this kind.
    Block(Kind),
    /// A section from any line it starts, as a contract's numbered paragraphs are.
    Clause,
    /// A section this many levels below its number's depth, from a heading line; from a longer
    /// line, an item.
    Part(usize),
}

/// The words that label a part, one list for every kind of document: as written in lower case,
/// as a label names them, and how their part runs. A note sits two levels down: inside a 10-K's
/// Item 8, and below a datasheet's numbered sections, which a short footnote must not end.
const WORDS: [(&str, &str, Runs); 19] = [
    ("table", "Table", Runs::Table),
    ("figure", "Figure", Runs::Block(Kind::Figure)),
    ("fig.", "Figure", Runs::Block(Kind::Figure)),
    ("theorem", "Theorem", Runs::Block(Kind::Item)),
    ("lemma", "Lemma", Runs::Block(Kind::Item)),
    ("equation", "Equation", Runs::Block(Kind::Item)),
    ("eq.", "Equation", Runs::Block(Kind::Item)),
    ("section", "Section", Runs::Clause),
    ("§", "Section", Runs::Clause),
    ("clause", "Clause", Runs::Clause),
    ("article", "Article", Runs::Clause),
    ("chapter", "Chapter", Runs::Part(0)),
    ("appendix", "Appendix", Runs::Part(0)),
    ("annex", "Annex", Runs::Part(0)),
    ("schedule", "Schedule", Runs::Part(0)),
    ("exhibit", "Exhibit", Runs::Part(0)),
    ("item", "Item", Runs::Part(0)),
    ("slide", "Slide", Runs::Part(0)),
    ("note", "Note", Runs::Part(2)),
];

/// A label's number: "4", "4.1", "1A", "IV", "B", "B.2".
const NUMBER: &str = r"(?:[0-9]+[A-Za-z]?|[IVXLC]+|[A-Z])(?:\.[0-9]+[A-Za-z]?)*";

/// A line a label word starts: the word, its number and the rest of the line.
static LABELED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(r"^\s*{}({NUMBER})\b(.*)", words())).expect("a valid pattern")
});

/// A label a question names, in any case: "table 3", "§4.1", "Eq. (5)", "item 1a".
static NAMED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(r"{}\(?((?i){NUMBER})\b\)?", words())).expect("a valid pattern")
});

/// A numbered paragraph with no label word: "12.2 Termination for Cause. Either party ...".
static CLAUSE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^([0-9]+(?:\.[0-9]+)+)\.?\s+["\u{201c}]?[A-Z][A-Za-z]"#).expect("a valid pattern")
});

/// An equation's number at its line's end: "y = Wx + b    (5)".
static EQUATION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s\(([0-9]+[a-z]?|[A-Z]\.[0-9]+)\)$").expect("a valid pattern"));

/// `WORDS` as a pattern in any case, in one group: a word that ends in a letter needs a space
/// before its number, so "TABLES" is no table's.
fn words() -> String {
    let each = WORDS.iter().map(|(w, ..)| {
        let gap = match w.ends_with(char::is_alphabetic) {
            true => r"\s+",
            false => r"\s*",
        };
        format!("{}{gap}", regex::escape(w))
    });
    format!("(?i:({}))", each.collect::<Vec<_>>().join("|"))
}

/// The label name and the run of label word `w` as a pattern matched it, space and all.
fn word(w: &str) -> Option<(&'static str, Runs)> {
    let w = w.trim().to_lowercase();
    (WORDS.iter())
        .find(|(x, ..)| *x == w)
        .map(|&(_, name, runs)| (name, runs))
}

/// The label of part `number` that word `name` names: "Table 3", and "Equation (5)" as an
/// equation's number is printed.
fn label(name: &str, number: &str) -> String {
    match name {
        "Equation" => format!("Equation ({number})"),
        _ => format!("{name} {number}"),
    }
}

/// A line that starts with a label word and its number.
struct Labeled<'a> {
    /// The word's label name ("Item").
    name: &'static str,
    /// The number ("1A").
    number: &'a str,
    /// How its part runs.
    runs: Runs,
    /// The line after the number, trimmed.
    rest: &'a str,
    /// Where the number ends in the line.
    end: usize,
}

/// Line `t` as a label word and its number start it, or `None` when a sentence only names the
/// part ("Table 3 shows ..."): a label is followed by the line's end, a mark or a capital.
fn labeled(t: &str) -> Option<Labeled<'_>> {
    let c = LABELED.captures(t)?;
    let (name, runs) = word(&c[1])?;
    let (number, rest) = (c.get(2)?, c.get(3)?.as_str().trim());
    let marks = ['.', ':', '-', '\u{2013}', '\u{2014}', '|', '('];
    (rest.is_empty() || rest.starts_with(marks) || rest.starts_with(char::is_uppercase)).then_some(
        Labeled {
            name,
            number: number.as_str(),
            runs,
            rest,
            end: number.end(),
        },
    )
}

/// How many levels number `n` has: "4" one, "4.1" two.
fn depth(n: &str) -> usize {
    n.split('.').count()
}

/// The label of a section numbered `n` with no label word: an appendix's when it starts with a
/// letter that is no Roman numeral ("A.1 Proofs"), else `word`'s ("4.1 Results").
fn numbered(word: &str, n: &str) -> String {
    let first = n.split('.').next().unwrap_or(n);
    let letter = first.starts_with(|c: char| c.is_ascii_uppercase())
        && !first.bytes().all(|b| b"IVX".contains(&b));
    format!("{} {n}", if letter { "Appendix" } else { word })
}

/// The word a document uses for its numbered parts: "Clause" when it says "clause" more often
/// than "section", as a contract may, else "Section".
fn bare(text: &str) -> &'static str {
    let lower = text.to_lowercase();
    match lower.matches("clause").count() > lower.matches("section").count() {
        true => "Clause",
        false => "Section",
    }
}

/// The label and number of heading line `t`: the label word's ("Item 1A. Risk Factors" is
/// "Item 1A"), a numbered heading's ("4.1 Results" is "Section 4.1", with `word`), else the line.
fn head<'a>(t: &'a str, word: &str) -> (String, Option<&'a str>) {
    if let Some(l) = labeled(t) {
        return (label(l.name, l.number), Some(l.number));
    }
    let h = t.trim_end_matches(':');
    let number = read_papers::heading(h)
        .and_then(|name| {
            h.split_once(' ')
                .filter(|(_, r)| r.trim().to_lowercase() == name)
        })
        .map(|(n, _)| n.trim_end_matches('.'));
    match number {
        Some(n) => (numbered(word, n), Some(n)),
        None => (h.to_owned(), None),
    }
}

/// Line `t` as a title: up to its first sentence's end after byte `from`, cut to `TITLE_BYTES`.
fn title(t: &str, from: usize) -> String {
    let end = (t.get(from..).unwrap_or_default().match_indices(". "))
        .find(|&(i, _)| i > 0)
        .map_or(t.len(), |(i, _)| from + i);
    t[..t.floor_char_boundary(end.min(TITLE_BYTES))]
        .trim_end()
        .to_owned()
}

/// Whether `doc` is a slide deck: most of its pages with text are short and start with a title
/// line.
fn deck(doc: &Doc) -> bool {
    let texts: Vec<&str> = (doc.pages.iter())
        .filter_map(|p| doc.text.get(p.start..p.end).map(str::trim))
        .filter(|t| !t.is_empty())
        .collect();
    let slide = |t: &str| {
        t.len() <= SLIDE_BYTES
            && (t.lines().next())
                .is_some_and(|l| l.len() <= TITLE_BYTES && !l.ends_with(['.', ',', ';']))
    };
    texts.len() >= MIN_SLIDES && texts.iter().filter(|t| slide(t)).count() * 4 >= texts.len() * 3
}

/// `doc`'s map in text order. Sections come from its headings (read_papers' and Markdown's),
/// from lines a label word starts ("Item 1A. Risk Factors", "ARTICLE 12"), and from numbered
/// paragraphs ("12.2 Termination ..."), each running to the next at its level or above; tables,
/// figures and other items from lines a label word starts and from equation numbers; and one
/// entry per page, or per slide in a deck.
pub fn map(doc: &Doc) -> Vec<Entry> {
    let text = doc.text.as_str();
    let word = bare(text);
    // Each entry, and whether a heading or a caption gave it: its label is then not taken by a
    // line that only starts with it ("Section 2. We show ...").
    let mut found: Vec<(Entry, bool)> = Vec::new();
    // The number of the last top-level numbered section. A deeper number is one of its parts
    // ("7.5" under "7"), so a table row "2.5 MHz ..." is no section.
    let mut top: Option<&str> = None;
    let blocks = read_papers::blocks(text);
    for (i, b) in blocks.iter().enumerate() {
        let end = b.start + b.text.len();
        let rows = match blocks.get(i + 1) {
            Some(next) if next.part == Part::Table => next.start + next.text.len(),
            _ => end,
        };
        let mut at = b.start;
        for line in b.text.split_inclusive('\n') {
            let caption = b.part == Part::Caption && at == b.start;
            let start = at + line.len() - line.trim_start().len();
            at += line.len();
            let t = line.trim();
            let entry = |label, title, kind, end| Entry {
                label,
                title,
                kind,
                range: Range {
                    doc: doc.index,
                    start,
                    end,
                },
            };
            let under =
                |n: &str| depth(n) == 1 || top.is_none_or(|t| n.split('.').next() == Some(t));
            let hit = if b.part == Part::Heading {
                match head(t, word) {
                    (_, Some(n)) if !under(n) => None,
                    (label, n) => {
                        top = n.filter(|n| depth(n) == 1).or(top);
                        let level = n.map_or(1, depth);
                        Some((entry(label, title(t, 0), Kind::Section(level), end), true))
                    }
                }
            } else if let Some((level, h)) = ask_docs::heading(line) {
                let label = head(h, word).0;
                Some((entry(label, h.into(), Kind::Section(level), end), true))
            } else if REFERENCES.contains(&t.trim_end_matches(':').to_lowercase().as_str()) {
                Some((entry(t.into(), t.into(), Kind::Section(1), end), true))
            } else if let Some(l) = labeled(t) {
                let name = label(l.name, l.number);
                let title = title(t, l.end);
                let words = l.rest.trim_start_matches(|c: char| !c.is_alphanumeric());
                let shaped = words.split_whitespace().count() <= TITLE_WORDS
                    && !l.rest.ends_with(['.', ',', ';']);
                let level = depth(l.number);
                Some(match l.runs {
                    Runs::Table => (entry(name, title, Kind::Table, rows), caption),
                    Runs::Block(kind) => (entry(name, title, kind, end), caption),
                    Runs::Clause => {
                        top = (level == 1).then_some(l.number).or(top);
                        (entry(name, title, Kind::Section(level), end), false)
                    }
                    Runs::Part(below) if shaped => {
                        (entry(name, title, Kind::Section(level + below), end), false)
                    }
                    Runs::Part(_) => (entry(name, title, Kind::Item, end), false),
                })
            } else if let Some(c) = CLAUSE.captures(t).filter(|c| under(&c[1])) {
                let n = &c[1];
                let title = title(t, n.len());
                Some((
                    entry(numbered(word, n), title, Kind::Section(depth(n)), end),
                    false,
                ))
            } else if let Some(c) = EQUATION.captures(t) {
                let formula = c.get(0).is_some_and(|m| t[..m.start()].contains(FORMULA));
                let line_end = start + t.len();
                formula.then(|| {
                    let name = label("Equation", &c[1]);
                    (entry(name, title(t, 0), Kind::Item, line_end), false)
                })
            } else {
                None
            };
            found.extend(hit);
        }
    }
    let strong: HashSet<String> = (found.iter())
        .filter(|(_, strong)| *strong)
        .map(|(e, _)| e.label.to_lowercase())
        .collect();
    let mut map: Vec<Entry> = (found.into_iter())
        .filter(|(e, s)| *s || !strong.contains(&e.label.to_lowercase()))
        .map(|(e, _)| e)
        .collect();
    for i in 0..map.len() {
        if let Kind::Section(level) = map[i].kind {
            map[i].range.end = (map[i + 1..].iter())
                .find(|e| matches!(e.kind, Kind::Section(l) if l <= level))
                .map_or(text.len(), |e| e.range.start);
        }
    }
    let deck = deck(doc);
    for (n, p) in doc.pages.iter().enumerate() {
        let range = Range {
            doc: doc.index,
            start: p.start,
            end: p.end,
        };
        let page = doc.text.get(p.start..p.end).unwrap_or_default();
        let first = page.trim().lines().next().unwrap_or_default();
        map.push(match deck {
            true => Entry {
                label: format!("Slide {}", n + 1),
                title: match first.is_empty() {
                    true => format!("Slide {}", n + 1),
                    false => title(first, 0),
                },
                kind: Kind::Slide,
                range,
            },
            false => Entry {
                label: format!("page {}", n + 1),
                title: format!("page {}", n + 1),
                kind: Kind::Page,
                range,
            },
        });
    }
    map.sort_by_key(|e| e.range.start);
    map
}

/// `docs`' maps as the labels `tools::read` and `cite::check` take: each entry but a page by its
/// label, and by its title too when that differs.
pub fn labels(docs: &[Doc]) -> Vec<(String, Range)> {
    let mut out = Vec::new();
    for e in docs.iter().flat_map(map).filter(|e| e.kind != Kind::Page) {
        if !e.title.eq_ignore_ascii_case(&e.label) {
            out.push((e.title, e.range));
        }
        out.push((e.label, e.range));
    }
    out
}

/// The labels `question` names, as the map writes them: "table 3" is "Table 3", "§4.1" is
/// "Section 4.1" and "eq. 5" is "Equation (5)".
pub fn named(question: &str) -> Vec<String> {
    (NAMED.captures_iter(question))
        .filter_map(|c| Some(label(word(&c[1])?.0, &c[2])))
        .collect()
}

/// The innermost section of `doc` that holds byte `at`.
pub fn section(doc: &Doc, at: usize) -> Option<Entry> {
    map(doc)
        .into_iter()
        .rfind(|e| matches!(e.kind, Kind::Section(_)) && e.range.start <= at && at < e.range.end)
}

/// The first read's list of `doc`'s headings with their pages, or of its slides in a deck; the
/// top level alone when there are more than `OUTLINE`, and at most that many. Empty when it has
/// none.
pub fn outline(doc: &Doc) -> String {
    let map = map(doc);
    let deck = map.iter().any(|e| e.kind == Kind::Slide);
    let level = |e: &Entry| match e.kind {
        Kind::Section(l) if !deck => Some(l),
        Kind::Slide => Some(0),
        _ => None,
    };
    let mut heads: Vec<&Entry> = map.iter().filter(|e| level(e).is_some()).collect();
    if heads.len() > OUTLINE {
        let top = heads.iter().filter_map(|e| level(e)).min();
        heads.retain(|e| level(e) == top);
    }
    if heads.is_empty() {
        return String::new();
    }
    let list: Vec<String> = (heads.iter().take(OUTLINE))
        .map(|e| format!("{} {}", e.title, tools::place(doc, false, e.range)))
        .collect();
    let more = match heads.len().saturating_sub(OUTLINE) {
        0 => String::new(),
        n => format!("; and {n} more"),
    };
    let what = if deck { "Slides" } else { "Headings" };
    format!("{what}: {}{more}.\n", list.join("; "))
}

#[cfg(test)]
mod tests {
    use std::{slice::from_ref, sync::LazyLock};

    use super::super::{cite, tools::tests::paged, visible::Visible};
    use super::*;

    /// A paragraph that names no part, to give a page a real page's length.
    static FILL: LazyLock<String> = LazyLock::new(|| {
        "The text goes on in plain sentences that name no part of the document at all.\n".repeat(18)
    });

    /// `pages`, each with `FILL` where it holds "{FILL}".
    fn doc(pages: &[&str]) -> Doc {
        let pages: Vec<String> = pages.iter().map(|p| p.replace("{FILL}", &FILL)).collect();
        paged(0, &pages.iter().map(String::as_str).collect::<Vec<_>>())
    }

    /// The entry labeled `label`, and its text.
    fn part<'a>(d: &'a Doc, map: &'a [Entry], label: &str) -> (&'a Entry, &'a str) {
        let e =
            (map.iter().find(|e| e.label == label)).unwrap_or_else(|| panic!("{label}: {map:#?}"));
        (e, &d.text[e.range.start..e.range.end])
    }

    fn paper() -> Doc {
        doc(&[
            "Sparse Attention at Scale\nAna Li, Bo Chen\nAbstract\nWe study sparse attention.\n\n\
             1 Introduction\n{FILL}",
            "2 Method\nWe define the loss as\nL = sum of squared errors + a penalty (1)\n\n\
             2.1 Setup\nTheorem 1. The loss is convex.\n\nTable 1: Accuracy on the test set.\n\n\
             Model Acc F1\nA 90.1 88.2\nB 91.4 89.0\n\n{FILL}",
            "3 Results\nSection 2. We show the gains hold.\n{FILL}Figure 2. Accuracy by size.\n\n\
             4 Conclusion\nIt works.\n\nReferences\n[1] A. Author. A paper. 2020.\n",
            "A Proofs\nProof of Theorem 1.\n\nA.1 Lemmas\nLemma 3. A bound holds.\n\n\
             Table 4: Extra results.\n\n1 2 3\n4 5 6\n",
        ])
    }

    #[test]
    fn a_paper_maps_its_sections_items_and_the_appendix_after_its_references() {
        let d = paper();
        let map = map(&d);
        let sections: Vec<(&str, Kind)> = (map.iter())
            .filter(|e| matches!(e.kind, Kind::Section(_)))
            .map(|e| (e.label.as_str(), e.kind))
            .collect();
        use Kind::Section as S;
        assert_eq!(
            sections,
            [
                ("Abstract", S(1)),
                ("Section 1", S(1)),
                ("Section 2", S(1)),
                ("Section 2.1", S(2)),
                ("Section 3", S(1)),
                ("Section 4", S(1)),
                ("References", S(1)),
                ("Appendix A", S(1)),
                ("Appendix A.1", S(2)),
            ]
        );
        // A section runs to the next at its level: "Section 2." in a sentence of Section 3 is
        // not one.
        let (two, text) = part(&d, &map, "Section 2");
        assert_eq!(two.title, "2 Method");
        assert!(
            text.starts_with("2 Method") && text.ends_with(".\n\x0c"),
            "{text:?}"
        );
        assert!(text.contains("Table 1") && !text.contains("3 Results"));
        let (_, refs) = part(&d, &map, "References");
        assert!(refs.ends_with("2020.\n\x0c"), "{refs:?}");
        let (_, appendix) = part(&d, &map, "Appendix A");
        assert!(appendix.starts_with("A Proofs") && appendix.ends_with("4 5 6\n"));
        let (table, rows) = part(&d, &map, "Table 1");
        assert_eq!(table.kind, Kind::Table);
        assert!(rows.starts_with("Table 1: Accuracy") && rows.ends_with("B 91.4 89.0\n"));
        assert_eq!(
            part(&d, &map, "Table 4").1,
            "Table 4: Extra results.\n\n1 2 3\n4 5 6\n"
        );
        let (eq, line) = part(&d, &map, "Equation (1)");
        assert_eq!(
            (eq.kind, line),
            (Kind::Item, "L = sum of squared errors + a penalty (1)")
        );
        assert_eq!(
            part(&d, &map, "Theorem 1").1,
            "Theorem 1. The loss is convex.\n"
        );
        assert_eq!(part(&d, &map, "Lemma 3").0.kind, Kind::Item);
        assert_eq!(part(&d, &map, "Figure 2").0.kind, Kind::Figure);
        let pages = map.iter().filter(|e| e.kind == Kind::Page).count();
        assert_eq!(pages, 4);
        // The first read lists the headings with their pages.
        let outline = outline(&d);
        assert!(outline.starts_with("Headings: Abstract [p. 1]; 1 Introduction [p. 1]; 2 Method [p. 2]; 2.1 Setup [p. 2]; 3 Results [p. 3];"), "{outline}");
        assert!(
            outline.ends_with("; A Proofs [p. 4]; A.1 Lemmas [p. 4].\n"),
            "{outline}"
        );
    }

    #[test]
    fn a_10k_maps_its_items_and_the_notes_inside_item_8() {
        let d = doc(&[
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\nFORM 10-K\nPART I\n\
             Item 1. Business\nWe make industrial pumps.\n{FILL}",
            "Item 1A. Risk Factors\nOur supply chain depends on two vendors.\n{FILL}PART II\n\
             Item 7. Management's Discussion and Analysis of Financial Condition and Results of \
             Operations\nRevenue rose 12% to $4.2 billion.\n{FILL}",
            "Item 8. Financial Statements and Supplementary Data\nNotes to Consolidated Financial \
             Statements\nNote 11 \u{2014} Leases\nWe lease our plants.\n{FILL}\
             Note 12 \u{2014} Income Taxes\nThe effective tax rate was 21.3%.\n",
            "Item 9. Changes in and Disagreements with Accountants\nNone.\n",
        ]);
        let map = map(&d);
        let (risk, text) = part(&d, &map, "Item 1A");
        assert_eq!(risk.title, "Item 1A. Risk Factors");
        assert!(text.contains("two vendors") && !text.contains("Item 7"));
        assert!(part(&d, &map, "Item 7").1.contains("$4.2 billion"));
        let (_, eight) = part(&d, &map, "Item 8");
        assert!(eight.contains("Note 11") && eight.contains("21.3%") && !eight.contains("Item 9"));
        let (_, note) = part(&d, &map, "Note 12");
        assert!(note.starts_with("Note 12") && note.contains("21.3%") && !note.contains("Leases"));
        // `read` and the quote check name its parts by these labels.
        let labels = labels(from_ref(&d));
        let piece = tools::read(&d, &labels, "Item 1A", None, 10_000).unwrap();
        assert!(d.text[piece.range.start..piece.range.end].contains("two vendors"));
        let checked = cite::check(
            "\"The effective tax rate was 21.3%.\" [Note 12]",
            from_ref(&d),
            &labels,
        );
        assert_eq!((checked.cited, checked.matched), (1, 1), "{checked:?}");
    }

    #[test]
    fn a_datasheet_section_holds_its_table_and_footnote_and_a_row_is_no_section() {
        let d = doc(&[
            "XC1234 3-A Step-Down Converter\n1 Features\n\u{2022} 3.5-V to 28-V input range\n\n\
             2 Applications\n\u{2022} Set-top boxes\n{FILL}",
            "7 Specifications\n7.1 Absolute Maximum Ratings\nVIN Input voltage -0.3 30 V\n{FILL}\
             7.5 Electrical Characteristics\nover operating free-air temperature range\n\
             PARAMETER TEST CONDITIONS MIN TYP MAX UNIT\nIQ Quiescent current No load 110 uA\n\
             2.5 MHz Switching frequency\nNote 1: Specified by design.\n",
            "7.6 Typical Characteristics\nFigure 9. Efficiency vs Output Current\n\n{FILL}\
             8 Detailed Description\nThe device is a buck converter.\n",
        ]);
        let map = map(&d);
        let (ec, text) = part(&d, &map, "Section 7.5");
        assert_eq!(ec.kind, Kind::Section(2));
        assert!(text.contains("Quiescent current") && text.contains("Specified by design"));
        assert!(!text.contains("7.6 Typical"), "{text}");
        assert!(!map.iter().any(|e| e.label == "Section 2.5"), "{map:#?}");
        assert_eq!(part(&d, &map, "Note 1").0.kind, Kind::Item);
        let (fig, caption) = part(&d, &map, "Figure 9");
        assert_eq!(fig.kind, Kind::Figure);
        assert_eq!(caption, "Figure 9. Efficiency vs Output Current\n");
        assert!(part(&d, &map, "Section 7").1.contains("Figure 9"));
    }

    #[test]
    fn a_contract_maps_its_articles_clauses_and_schedules() {
        let d = doc(&[
            "MASTER SERVICES AGREEMENT\nThis Agreement is made between Acme Ltd and Beta Ltd.\n\
             ARTICLE 1 DEFINITIONS\n1.1 \"Services\" means the services set out in Schedule B.\n\
             1.2 Headings do not affect the meaning of any clause.\n{FILL}",
            "ARTICLE 12 TERMINATION\n12.1 Termination for Convenience. Either party may \
             terminate this Agreement on ninety (90) days' written notice.\n12.2 Termination for \
             Cause. Either party may terminate this Agreement at once if the other party commits \
             a material breach and does not remedy it within thirty (30) days.\n12.3 Effect of \
             Termination. Clauses 7 and 9 survive termination.\n{FILL}",
            "ARTICLE 13 GENERAL\n13.1 Notices. Notices must be in writing.\n{FILL}SCHEDULE B\n\
             FEES\nThe monthly fee is $10,000.\n",
        ]);
        let map = map(&d);
        let (article, text) = part(&d, &map, "Article 12");
        assert_eq!(
            (article.kind, article.title.as_str()),
            (Kind::Section(1), "ARTICLE 12 TERMINATION")
        );
        assert!(text.contains("12.3 Effect") && !text.contains("ARTICLE 13"));
        let (clause, text) = part(&d, &map, "Clause 12.2");
        assert_eq!(
            (clause.kind, clause.title.as_str()),
            (Kind::Section(2), "12.2 Termination for Cause")
        );
        assert!(
            text.starts_with("12.2 Termination") && text.ends_with("(30) days.\n"),
            "{text}"
        );
        let (_, schedule) = part(&d, &map, "Schedule B");
        assert!(schedule.starts_with("SCHEDULE B\nFEES") && schedule.contains("$10,000"));
        assert!(!part(&d, &map, "Article 13").1.contains("SCHEDULE B"));
        let labels = labels(from_ref(&d));
        let piece = tools::read(&d, &labels, "clause 12.2", None, 10_000).unwrap();
        assert_eq!(piece.part, clause.range);
    }

    #[test]
    fn a_deck_has_one_entry_per_slide_and_lists_their_titles() {
        let d = doc(&[
            "Acme Robotics\nSeries A, 2026\n",
            "The Problem\nWarehouses lose 30% of picks to errors.\n",
            "Market Size\n$40B by 2030\n",
            "Traction\n12 customers, $1.2M ARR\n",
        ]);
        let map = map(&d);
        let slides: Vec<(&str, &str)> = (map.iter())
            .filter(|e| e.kind == Kind::Slide)
            .map(|e| (e.label.as_str(), e.title.as_str()))
            .collect();
        assert_eq!(
            slides,
            [
                ("Slide 1", "Acme Robotics"),
                ("Slide 2", "The Problem"),
                ("Slide 3", "Market Size"),
                ("Slide 4", "Traction"),
            ]
        );
        assert_eq!(part(&d, &map, "Slide 3").0.range.start, d.pages[2].start);
        assert_eq!(
            outline(&d),
            "Slides: Acme Robotics [p. 1]; The Problem [p. 2]; Market Size [p. 3]; Traction [p. 4].\n"
        );
        assert!(!deck(&paper()));
    }

    #[test]
    fn a_web_page_maps_its_markdown_headings() {
        let d = doc(&[
            "# Annual Report\nIntro.\n## 4.1 Pricing\nPlans.\n## Item 1A. Risk Factors\nRisks.\n",
        ]);
        let map = map(&d);
        let (pricing, text) = part(&d, &map, "Section 4.1");
        assert_eq!(
            (pricing.kind, pricing.title.as_str()),
            (Kind::Section(2), "4.1 Pricing")
        );
        assert_eq!(text, "## 4.1 Pricing\nPlans.\n");
        assert!(part(&d, &map, "Item 1A").1.ends_with("Risks.\n"));
        assert!(part(&d, &map, "Annual Report").1.contains("Risks."));
    }

    #[test]
    fn a_long_outline_keeps_the_top_level() {
        let pages: Vec<String> = (1..=30)
            .map(|n| format!("{n} Part {n}\n{n}.1 First\n{n}.2 Second\n{}", *FILL))
            .collect();
        let d = paged(0, &pages.iter().map(String::as_str).collect::<Vec<_>>());
        let outline = outline(&d);
        assert!(
            outline.starts_with("Headings: 1 Part 1 [p. 1]; 2 Part 2 [p. 2];"),
            "{outline}"
        );
        assert!(!outline.contains("1.1 First") && outline.ends_with("30 Part 30 [p. 30].\n"));
    }

    #[test]
    fn a_question_names_its_labels_as_the_map_writes_them() {
        assert_eq!(
            named("Do table 3, §4.1 and Eq. (5) agree with Item 1a and fig. 2? Tables vary."),
            [
                "Table 3",
                "Section 4.1",
                "Equation (5)",
                "Item 1a",
                "Figure 2"
            ]
        );
    }

    #[test]
    fn a_page_read_names_the_section_it_is_in() {
        let d = paper();
        let piece = tools::read(&d, &[], "page 2", None, 100_000).unwrap();
        let text = piece.render(from_ref(&d), &Visible::default(), 9);
        assert!(
            text.starts_with("page 2 of 4, in 2 Method, which runs to page 2:\n"),
            "{text}"
        );
    }
}
