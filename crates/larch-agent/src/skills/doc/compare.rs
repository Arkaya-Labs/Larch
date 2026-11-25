//! The comparison step (plan 7.7): after each document's reader answers on its own, one step
//! compares the answers, with the paragraphs they quote and `find` and `read` on every document.

use std::{path::Path, slice::from_ref};

use super::reader::Answered;
use crate::folio::{Doc, Range, cite, tools};
use crate::skills::{read_papers, web_answer};

/// The heading the comparison is shown under, after the documents' answers.
pub const HEADING: &str = "## Comparison";

/// Bytes of a quote's paragraph kept on each side of it.
const AROUND: usize = 600;

/// The comparison's question: the user's, then each document's answer under its letter, with its
/// places named with that letter as the comparison cites them, then what to check.
pub fn question(question: &str, docs: &[Doc], answers: &[Answered]) -> String {
    let mut out = format!("{question}\n\nEach document's reader answered it from that document:");
    for (d, a) in docs.iter().zip(answers) {
        let l = d.letter();
        let text = (a.answer.replace("[p. ", &format!("[{l}, p. ")))
            .replace("[pp. ", &format!("[{l}, pp. "));
        let name = read_papers::name(Path::new(&d.name));
        out += &format!("\n\n{l}. {name}:\n{text}");
    }
    out + "\n\nCompare these answers. Before you compare two numbers, check that they measure the \
           same thing: the same dataset, units and conditions. Read again where a condition is \
           missing."
}

/// The paragraph around each quote each document's answer cites, in that document: on the page
/// the quote cites when it is there, else where the text first has it. A quote the text does not
/// have gives none.
pub fn evidence(docs: &[Doc], answers: &[Answered]) -> Vec<Range> {
    let mut out = Vec::new();
    for (d, a) in docs.iter().zip(answers) {
        for (quote, after) in web_answer::quoted(&a.answer, web_answer::QUOTE_CHARS) {
            // The citation right after the quote, as the check reads it.
            let after = after.trim_start();
            let cited = (after.starts_with('['))
                .then(|| after.split_inclusive(']').next().unwrap_or_default())
                .and_then(|c| cite::cited(c, from_ref(d), &[]).first().copied());
            let hits = tools::find(from_ref(d), quote, 0, usize::MAX).hits;
            let on = |r: &Range| cited.is_none_or(|c| c.start <= r.start && r.end <= c.end);
            let hit = (hits.iter().find(|h| on(&h.range))).or(hits.first());
            out.extend(hit.map(|h| paragraph(d, h.range)));
        }
    }
    out
}

/// The paragraph of `doc` around `r`: from the blank line before it to the one after, inside
/// the pages it starts and ends on and `AROUND` bytes on each side, at line ends. A quote can
/// run across a page break, as `find` reads the text with its breaks as spaces.
fn paragraph(doc: &Doc, r: Range) -> Range {
    let page = |at| &doc.pages[doc.page_of(at)];
    let (first, last) = (page(r.start), page(r.end.saturating_sub(1).max(r.start)));
    let t = &doc.text;
    let lo = t.ceil_char_boundary(r.start.saturating_sub(AROUND).max(first.start));
    let hi = t.floor_char_boundary(r.end.saturating_add(AROUND).min(last.end));
    let (before, after) = (&t[lo..r.start], &t[r.end..hi]);
    let start = match before.rfind("\n\n") {
        Some(i) => lo + i + 2,
        None if lo == first.start => lo,
        None => before.find('\n').map_or(r.start, |i| lo + i + 1),
    };
    let end = match after.find("\n\n") {
        Some(i) => r.end + i,
        None if hi == last.end => hi,
        None => after.rfind('\n').map_or(r.end, |i| r.end + i),
    };
    Range {
        doc: doc.index,
        start,
        end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::folio::tools::tests::paged;

    fn answered(answer: &str) -> Answered {
        Answered {
            answer: answer.into(),
            notes: Vec::new(),
            ranges: Vec::new(),
            checked: false,
            not_found: false,
        }
    }

    #[test]
    fn the_evidence_is_the_paragraph_around_each_quote_on_its_cited_page() {
        let a = paged(
            0,
            &[
                "Intro.\n\nThe error rate is 4% on the test set.\nMore.",
                "Setup.\n\nWe tested on ImageNet at 224 px.\nThe error rate is 4% on the test set.\n\nEnd.",
            ],
        );
        let b = paged(1, &["Nothing quoted here."]);
        let docs = [a.clone(), b];
        let answers = [
            answered("\"The error rate is 4% on the test set.\" [p. 2]"),
            answered("Not found in the text."),
        ];
        let got = evidence(&docs, &answers);
        let text: Vec<&str> = got.iter().map(|r| &a.text[r.start..r.end]).collect();
        assert_eq!(
            text,
            ["We tested on ImageNet at 224 px.\nThe error rate is 4% on the test set."]
        );
        let q = question("Which is better?", &docs, &answers);
        assert!(
            q.contains("A. 0.pdf:\n\"The error rate is 4% on the test set.\" [A, p. 2]"),
            "{q}"
        );
        assert!(q.contains("B. 1.pdf:\nNot found in the text."), "{q}");
    }

    #[test]
    fn a_quote_across_a_page_break_gives_its_paragraph_on_both_pages() {
        let a = paged(
            0,
            &["Intro.\n\nSentence starts", "here. Then more.\n\nEnd."],
        );
        let got = evidence(
            from_ref(&a),
            &[answered("\"Sentence starts here.\" [pp. 1-2]")],
        );
        let text: Vec<&str> = got.iter().map(|r| &a.text[r.start..r.end]).collect();
        assert_eq!(text, ["Sentence starts\x0chere. Then more."]);
    }
}
