//! PDF tables as rows and columns from word boxes, or "layout uncertain".

use std::{
    path::Path,
    process::Command,
    sync::{LazyLock, atomic::AtomicBool},
};

use regex::Regex;

use super::ocr::{self, LIMIT, installed};
use crate::{skills::read_papers::output_until, tool::no_window};

/// One word on a page and its box, in points from the page's top-left corner.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    /// The word as printed; an OCR line is one word.
    pub text: String,
    /// The left edge.
    pub x0: f64,
    /// The top edge.
    pub y0: f64,
    /// The right edge.
    pub x1: f64,
    /// The bottom edge.
    pub y1: f64,
}

/// A table read from word boxes: its rows and columns when they line up, else its lines.
#[derive(Clone, Debug, PartialEq)]
pub enum Table {
    /// Rows and columns that line up.
    Rows(Grid),
    /// The lines in reading order, as the rows and columns do not line up.
    Uncertain(Vec<String>),
}

/// The rows and columns of a table that lines up.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    /// The lines above the header that name the table ("Table 3. Electrical characteristics").
    pub caption: Vec<String>,
    /// The header rows as written, top first; a merged header is one cell over the columns it
    /// spans.
    pub header: Vec<Vec<Cell>>,
    /// The body rows, one text per column, never empty; a blank cell stays blank.
    pub body: Vec<Vec<String>>,
    /// The lines under the table: its footnotes.
    pub notes: Vec<String>,
}

/// A header cell and the columns it spans.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    /// Its text; blank over a column this row names nothing for.
    pub text: String,
    /// How many columns it spans, at least 1.
    pub span: usize,
}

/// A gap wider than this many word heights ends a cell: a word space is about a third of one.
// ponytail: CELL_GAP and MOST are set by hand on the fixtures below; the gate's hand-labeled
// tables set them, and a loose table whose cells come out joined is the trigger.
const CELL_GAP: f64 = 0.8;

/// The share of the rows with two cells or more that must leave a gap empty for it to divide two
/// columns: a merged header or a long label may cross it in the others.
const MOST: f64 = 0.8;

/// What the reader is told before the lines of a table that does not line up.
const UNCERTAIN: &str =
    "(layout uncertain: the rows and columns do not line up; these are its lines in reading order)";

/// A run of words on one line with no gap wider than a word space in it.
#[derive(Debug)]
struct Seg {
    text: String,
    x0: f64,
    x1: f64,
}

/// A segment placed in the columns: the first and the last column it is over, and its text.
type Placed = (usize, usize, String);

/// The table that `words` make: one table's words, with its caption above it and its footnotes
/// under it. Rows are the lines, columns the gaps that line up across most rows. When a row does
/// not fit the columns, it is `Table::Uncertain`: a guessed table would be read as a fact.
pub fn table(words: &[Word]) -> Table {
    let lines = lines(words);
    let text: Vec<String> = (lines.iter())
        .map(|l| {
            l.iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    let mut heights: Vec<f64> = words.iter().map(|w| w.y1 - w.y0).collect();
    heights.sort_by(f64::total_cmp);
    let h = heights.get(heights.len() / 2).copied().unwrap_or(0.0);
    let segs: Vec<Vec<Seg>> = lines.iter().map(|l| segments(l, CELL_GAP * h)).collect();
    let cols = columns(&segs);
    if h <= 0.0 || cols.len() < 2 {
        return Table::Uncertain(text);
    }
    let placed: Vec<Option<Vec<Placed>>> = segs.iter().map(|s| place(s, &cols)).collect();
    fn row(p: &Option<Vec<Placed>>) -> Option<&Vec<Placed>> {
        p.as_ref().filter(|p| p.iter().all(|e| e.0 == e.1))
    }
    let values = |p: &Option<Vec<Placed>>| row(p).is_some_and(|p| p.iter().any(|e| e.0 > 0));
    // A header names columns: nothing in it spans from the row labels, and it has no row label
    // or no number but years. A body counts.
    let header_like = |p: &Option<Vec<Placed>>| {
        p.as_ref().is_some_and(|p| {
            (p.iter().any(|e| e.0 > 0) && p.iter().all(|e| e.0 > 0 || e.1 == 0))
                && (p[0].0 > 0 || !p.iter().any(|e| e.0 > 0 && counts(&e.2)))
        })
    };
    let top = (0..placed.len())
        .find(|&i| header_like(&placed[i]) || values(&placed[i]))
        .unwrap_or(placed.len());
    let mut first = (top..placed.len())
        .find(|&i| !header_like(&placed[i]))
        .unwrap_or(placed.len());
    let last_values = |from: usize| (from..placed.len()).rev().find(|&i| values(&placed[i]));
    let last = match last_values(first) {
        Some(last) => last,
        // Every row reads as a header: a table of words, whose first row is its header.
        None if first > top + 1 => {
            first = top + 1;
            match last_values(first) {
                Some(last) => last,
                None => return Table::Uncertain(text),
            }
        }
        None => return Table::Uncertain(text),
    };
    let Some(body) = (placed[first..=last].iter())
        .map(|p| row(p).map(|p| fill(p, cols.len())))
        .collect::<Option<Vec<_>>>()
    else {
        return Table::Uncertain(text);
    };
    let head: Vec<&Vec<Placed>> = placed[top..first].iter().flatten().collect();
    Table::Rows(Grid {
        caption: text[..top].to_vec(),
        header: header(&head, cols.len()),
        body,
        notes: text[last + 1..].to_vec(),
    })
}

/// `tables`, one per page in page order, each joined onto the one before when it continues it:
/// the same number of columns, no new caption ("continued" is none), and no header but the one
/// it repeats.
pub fn join(tables: Vec<Table>) -> Vec<Table> {
    let mut out: Vec<Table> = Vec::new();
    for t in tables {
        match (out.last_mut(), t) {
            (Some(Table::Rows(a)), Table::Rows(b))
                if a.body[0].len() == b.body[0].len()
                    && (b.caption.iter()).all(|l| l.to_lowercase().contains("continued"))
                    && (b.header.is_empty() || b.header == a.header) =>
            {
                a.body.extend(b.body);
                a.notes.extend(b.notes);
            }
            (_, t) => out.push(t),
        }
    }
    out
}

impl Table {
    /// The table as the reader is shown it: its caption, a MultiMarkdown table whose merged
    /// header cell ends in one `|` for each column it spans, then its notes; or the "layout
    /// uncertain" line and the lines.
    pub fn markdown(&self) -> String {
        let g = match self {
            Table::Uncertain(lines) => return format!("{UNCERTAIN}\n{}", lines.join("\n")),
            Table::Rows(g) => g,
        };
        let cell = |t: &str| t.replace('|', "\\|");
        let mut out = g.caption.clone();
        if !out.is_empty() {
            out.push(String::new());
        }
        out.extend((g.header.iter()).map(|r| {
            let cells = r
                .iter()
                .map(|c| format!(" {} {}", cell(&c.text), "|".repeat(c.span)));
            format!("|{}", cells.collect::<String>())
        }));
        out.push(format!("|{}", "---|".repeat(g.body[0].len())));
        out.extend((g.body.iter()).map(|r| {
            format!(
                "|{}",
                r.iter()
                    .map(|c| format!(" {} |", cell(c)))
                    .collect::<String>()
            )
        }));
        if !g.notes.is_empty() {
            out.push(String::new());
            out.extend(g.notes.iter().cloned());
        }
        out.join("\n")
    }
}

/// Whether `words` can run on this machine: poppler's pdftotext is installed, or it is a Mac.
pub fn available() -> bool {
    installed("pdftotext") || cfg!(target_os = "macos")
}

/// The words of page `page` (from 0) of `pdf` with their boxes: by poppler's
/// `pdftotext -bbox-layout` when it is installed, else by Vision on the rendered page on macOS
/// (`ocr::vision`). `stop` ends either.
pub fn words(pdf: &Path, page: usize, stop: &AtomicBool) -> Result<Vec<Word>, String> {
    if installed("pdftotext") {
        let n = (page + 1).to_string();
        let mut cmd = Command::new("pdftotext");
        no_window(&mut cmd)
            .args(["-q", "-enc", "UTF-8", "-bbox-layout", "-f", &n, "-l", &n])
            .arg(pdf)
            .arg("-");
        let out = output_until(cmd, LIMIT, stop).map_err(|e| format!("pdftotext: {e}"))?;
        return bbox_words(&String::from_utf8_lossy(&out));
    }
    if cfg!(target_os = "macos") {
        return ocr::vision(pdf, page, "words", stop).map_err(|e| format!("Vision: {e}"));
    }
    Err("word boxes need poppler's pdftotext, which is not installed on this machine".into())
}

/// The words in the XHTML that `pdftotext -bbox-layout` prints, in its order.
fn bbox_words(xhtml: &str) -> Result<Vec<Word>, String> {
    static WORD: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r#"<word xMin="([^"]*)" yMin="([^"]*)" xMax="([^"]*)" yMax="([^"]*)">([^<]*)</word>"#,
        )
        .expect("the word pattern is valid")
    });
    (WORD.captures_iter(xhtml))
        .map(|c| word(&unescape(&c[5]), [&c[1], &c[2], &c[3], &c[4]]))
        .filter(|w| w.as_ref().map_or(true, |w| !w.text.is_empty()))
        .collect()
}

/// `s` with the five XML entities poppler writes decoded, `&amp;` last so `&amp;lt;` stays `&lt;`.
// ponytail: web::extract's decoder is private to its file; use it once it is pub(crate).
fn unescape(s: &str) -> String {
    (s.replace("&lt;", "<").replace("&gt;", ">"))
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// The words, or OCR lines, in `tsv`: one a line, its left, top, right and bottom edges in points
/// from the page's top-left corner, then its text, tab separated. Every script here prints this,
/// and the OCR cache keeps it.
pub(super) fn boxes(tsv: &str) -> Result<Vec<Word>, String> {
    (tsv.lines().filter(|l| !l.trim().is_empty()))
        .map(|l| match l.splitn(5, '\t').collect::<Vec<_>>()[..] {
            [x0, y0, x1, y1, text] => word(text, [x0, y0, x1, y1]),
            _ => Err(format!("a box line is not 5 tab-separated fields: {l:?}")),
        })
        .filter(|w| w.as_ref().map_or(true, |w| !w.text.is_empty()))
        .collect()
}

/// `words` as `boxes` reads them.
pub(super) fn tsv(words: &[Word]) -> String {
    (words.iter())
        .map(|w| {
            let text = w.text.replace(['\t', '\n', '\r'], " ");
            format!(
                "{:.2}\t{:.2}\t{:.2}\t{:.2}\t{text}\n",
                w.x0, w.y0, w.x1, w.y1
            )
        })
        .collect()
}

/// A word from its text and its four edges as printed, which must be numbers in order.
fn word(text: &str, edges: [&str; 4]) -> Result<Word, String> {
    let bad = || format!("a word box is not four numbers in order: {edges:?}");
    let mut n = [0.0; 4];
    for (n, e) in n.iter_mut().zip(edges) {
        *n = (e.trim().parse::<f64>().ok())
            .filter(|n| n.is_finite())
            .ok_or_else(bad)?;
    }
    let [x0, y0, x1, y1] = n;
    if x1 < x0 || y1 < y0 {
        return Err(bad());
    }
    let text = text.trim().to_owned();
    Ok(Word {
        text,
        x0,
        y0,
        x1,
        y1,
    })
}

/// `words` in lines, top to bottom, each left to right: a word is on a line when half of the
/// smaller height is shared, so a raised footnote mark stays on its line.
fn lines(words: &[Word]) -> Vec<Vec<Word>> {
    let mut sorted: Vec<&Word> = words.iter().collect();
    sorted.sort_by(|a, b| (a.y0 + a.y1).total_cmp(&(b.y0 + b.y1)));
    let mut lines: Vec<(f64, f64, Vec<Word>)> = Vec::new();
    for w in sorted {
        match lines.last_mut() {
            Some((y0, y1, line))
                if 2.0 * (y1.min(w.y1) - y0.max(w.y0)) >= (w.y1 - w.y0).min(*y1 - *y0) =>
            {
                (*y0, *y1) = (y0.min(w.y0), y1.max(w.y1));
                line.push(w.clone());
            }
            _ => lines.push((w.y0, w.y1, vec![w.clone()])),
        }
    }
    (lines.into_iter())
        .map(|(_, _, mut line)| {
            line.sort_by(|a, b| a.x0.total_cmp(&b.x0));
            line
        })
        .collect()
}

/// `line`'s words in segments split at gaps wider than `gap`. A lone currency sign joins the
/// number after it and a lone footnote symbol the cell before it: a 10-K sets its `$` apart.
fn segments(line: &[Word], gap: f64) -> Vec<Seg> {
    let mut segs: Vec<Seg> = Vec::new();
    let mut sign = false;
    for w in line {
        let mark = w.text.chars().all(|c| "*†‡§".contains(c));
        match segs.last_mut() {
            Some(s) if sign || w.x0 - s.x1 <= gap || mark => {
                s.text.push(' ');
                s.text.push_str(&w.text);
                s.x1 = s.x1.max(w.x1);
            }
            _ => segs.push(Seg {
                text: w.text.clone(),
                x0: w.x0,
                x1: w.x1,
            }),
        }
        sign = w.text.chars().all(|c| "$€£¥".contains(c));
    }
    segs
}

/// The columns, left to right, as x ranges, from the rows with two segments or more: where more
/// than all but `MOST` of them have ink, so a gap that lines up across most rows divides two
/// columns; and, in those gaps, where two cells or more that are in no such column overlap: a MIN
/// column that most rows leave blank.
fn columns(rows: &[Vec<Seg>]) -> Vec<(f64, f64)> {
    let grid: Vec<&Vec<Seg>> = rows.iter().filter(|r| r.len() >= 2).collect();
    if grid.len() < 2 {
        return Vec::new();
    }
    let allowed = grid.len() - (MOST * grid.len() as f64).ceil() as usize;
    let dense = inked(grid.iter().copied().flatten(), allowed);
    let rest = (grid.iter().copied().flatten())
        .filter(|s| !dense.iter().any(|c| s.x0 < c.1 && s.x1 > c.0));
    let mut cols = inked(rest, 1);
    cols.extend(dense);
    cols.sort_by(|a, b| a.0.total_cmp(&b.0));
    cols
}

/// The x ranges where more than `over` of `segs` overlap.
fn inked<'a>(segs: impl Iterator<Item = &'a Seg>, over: usize) -> Vec<(f64, f64)> {
    let mut edges: Vec<(f64, isize)> = segs.flat_map(|s| [(s.x0, 1), (s.x1, -1)]).collect();
    // At one x, a segment that starts there counts before one that ends there leaves.
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.cmp(&a.1)));
    let (mut ink, mut start, mut out) = (0, 0.0, Vec::new());
    for (x, step) in edges {
        let was = ink > over as isize;
        ink += step;
        match (was, ink > over as isize) {
            (false, true) => start = x,
            (true, false) => out.push((start, x)),
            _ => {}
        }
    }
    out
}

/// `segs` placed in `cols`, or `None` when one is over no column or shares a column with another:
/// then the gaps do not line up.
fn place(segs: &[Seg], cols: &[(f64, f64)]) -> Option<Vec<Placed>> {
    let mut out: Vec<Placed> = Vec::new();
    for s in segs {
        let mut over = (cols.iter().enumerate())
            .filter(|(_, c)| s.x0 < c.1 && s.x1 > c.0)
            .map(|(i, _)| i);
        let first = over.next()?;
        let last = over.next_back().unwrap_or(first);
        if out.last().is_some_and(|p| p.1 >= first) {
            return None;
        }
        out.push((first, last, s.text.clone()));
    }
    Some(out)
}

/// A body row: each segment's text in its column, the rest blank.
fn fill(row: &[Placed], width: usize) -> Vec<String> {
    let mut cells = vec![String::new(); width];
    for (c, _, text) in row {
        cells[*c].clone_from(text);
    }
    cells
}

/// The header rows as cells. A merged header sits over some of the columns it spans, so each
/// column with no cell of its own in a row, but one in a row below, goes to the nearest cell of
/// the row over that run of columns with headers below.
fn header(rows: &[&Vec<Placed>], width: usize) -> Vec<Vec<Cell>> {
    (0..rows.len())
        .map(|k| {
            let below = |c: usize| {
                rows[k + 1..]
                    .iter()
                    .any(|r| r.iter().any(|e| e.0 <= c && c <= e.1))
            };
            let mut owner: Vec<Option<usize>> = vec![None; width];
            for (j, e) in rows[k].iter().enumerate() {
                owner[e.0..=e.1].fill(Some(j));
            }
            let own = owner.clone();
            for c in (0..width).filter(|&c| own[c].is_none() && below(c)) {
                let near = |run: &mut dyn Iterator<Item = usize>| {
                    (run.take_while(|&d| below(d)))
                        .enumerate()
                        .find_map(|(i, d)| own[d].map(|j| (i, j)))
                };
                let left = near(&mut (0..c).rev());
                let right = near(&mut (c + 1..width));
                owner[c] = match (left, right) {
                    (Some(l), Some(r)) if r.0 < l.0 => Some(r.1),
                    (Some(l), _) => Some(l.1),
                    (None, r) => r.map(|r| r.1),
                };
            }
            let mut cells: Vec<Cell> = Vec::new();
            for (c, o) in owner.iter().enumerate() {
                match (o, cells.last_mut()) {
                    (Some(_), Some(last)) if c > 0 && owner[c - 1] == *o => last.span += 1,
                    _ => cells.push(Cell {
                        text: o.map_or_else(String::new, |j| rows[k][j].2.clone()),
                        span: 1,
                    }),
                }
            }
            cells
        })
        .collect()
}

/// Whether `cell` holds a number that is not a year: a header names years, a body counts.
fn counts(cell: &str) -> bool {
    cell.split_whitespace().any(|t| {
        let t = t.trim_matches(|c: char| !c.is_alphanumeric());
        let year = t.len() == 4 && t.parse::<u16>().is_ok_and(|y| (1900..2100).contains(&y));
        !t.is_empty()
            && !year
            && t.chars()
                .all(|c| c.is_ascii_digit() || c == ',' || c == '.')
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// Five points a character and a 2.5-point space, 9 points high: a 9-point sans font.
    fn width(text: &str) -> f64 {
        let words = text.split(' ').count() as f64;
        5.0 * text.chars().filter(|&c| c != ' ').count() as f64 + 2.5 * (words - 1.0)
    }

    /// A page laid out by hand: each row is its top and its cells, each cell its left edge and
    /// its text, whose words become words.
    fn page(rows: &[(f64, &[(f64, &str)])]) -> Vec<Word> {
        let mut words = Vec::new();
        for &(y, cells) in rows {
            for &(mut x, text) in cells {
                for w in text.split(' ') {
                    let (x0, x1) = (x, x + width(w));
                    words.push(Word {
                        text: w.into(),
                        x0,
                        y0: y,
                        x1,
                        y1: y + 9.0,
                    });
                    x = x1 + 2.5;
                }
            }
        }
        words
    }

    /// A cell whose right edge is at `right`, as numbers are set.
    fn r(right: f64, text: &str) -> (f64, &str) {
        (right - width(text), text)
    }

    /// The (value, row header, column header) of each filled body cell: the row header is the
    /// row's first cell, the column header the lowest header cell over the column with text.
    fn triples(t: &Table) -> BTreeSet<(String, String, String)> {
        let Table::Rows(g) = t else {
            return BTreeSet::new();
        };
        let name = |c: usize| {
            (g.header.iter().rev())
                .find_map(|row| {
                    let mut at = 0;
                    let cell = row.iter().find(|cell| {
                        at += cell.span;
                        c < at
                    })?;
                    Some(cell.text.clone()).filter(|t| !t.is_empty())
                })
                .unwrap_or_default()
        };
        (g.body.iter())
            .flat_map(|row| {
                (row.iter().enumerate().skip(1))
                    .filter(|(_, v)| !v.is_empty())
                    .map(move |(c, v)| (v.clone(), row[0].clone(), name(c)))
            })
            .collect()
    }

    /// The triples of a hand-labeled table: its column headers, then each row's header and
    /// values, a blank value being a blank cell.
    fn gold(columns: &[&str], rows: &[(&str, &[&str])]) -> BTreeSet<(String, String, String)> {
        (rows.iter())
            .flat_map(|&(head, values)| {
                (values.iter().zip(columns))
                    .filter(|(v, _)| !v.is_empty())
                    .map(move |(v, c)| (v.to_string(), head.to_string(), c.to_string()))
            })
            .collect()
    }

    fn grid(t: &Table) -> &Grid {
        match t {
            Table::Rows(g) => g,
            Table::Uncertain(lines) => panic!("layout uncertain: {lines:#?}"),
        }
    }

    #[rustfmt::skip]
    fn income_statement() -> Vec<Word> {
        page(&[
            (40.0, &[(50.0, "CONSOLIDATED STATEMENTS OF OPERATIONS")]),
            (54.0, &[(50.0, "(In millions, except per share data)")]),
            (76.0, &[(380.0, "Year Ended December 31,")]),
            (90.0, &[r(380.0, "2023"), r(460.0, "2022"), r(540.0, "2021")]),
            (108.0, &[
                (50.0, "Revenues"),
                (330.0, "$"),
                r(380.0, "96,773"),
                (410.0, "$"),
                r(460.0, "81,462"),
                (490.0, "$"),
                r(540.0, "53,823"),
            ]),
            (122.0, &[
                (50.0, "Cost of revenues"),
                r(380.0, "79,113"),
                r(460.0, "65,121"),
                r(540.0, "40,217"),
            ]),
            (136.0, &[
                (50.0, "Gross profit"),
                r(380.0, "17,660"),
                r(460.0, "16,341"),
                r(540.0, "13,606"),
            ]),
            (150.0, &[(50.0, "Operating expenses:")]),
            (164.0, &[
                (60.0, "Research and development"),
                r(380.0, "3,969"),
                r(460.0, "3,075"),
                r(540.0, "2,593"),
            ]),
            (178.0, &[
                (60.0, "Selling, general and administrative"),
                r(380.0, "4,800"),
                r(460.0, "3,946"),
                r(540.0, "4,517"),
            ]),
            (192.0, &[
                (60.0, "Restructuring and other"),
                r(380.0, "—"),
                r(460.0, "(34)"),
                r(540.0, "(27)"),
            ]),
            (206.0, &[
                (50.0, "Income from operations"),
                r(380.0, "8,891"),
                r(460.0, "9,354"),
                r(540.0, "6,523"),
            ]),
            (220.0, &[
                (50.0, "Net income"),
                (330.0, "$"),
                r(380.0, "14,997"),
                (410.0, "$"),
                r(460.0, "12,556"),
                (490.0, "$"),
                r(540.0, "5,519"),
            ]),
            (234.0, &[(50.0, "Earnings per share:")]),
            (248.0, &[
                (60.0, "Diluted (1)"),
                (330.0, "$"),
                r(380.0, "4.30"),
                (410.0, "$"),
                r(460.0, "3.62"),
                (490.0, "$"),
                r(540.0, "1.63"),
            ]),
            (268.0, &[(
                50.0,
                "(1) Diluted earnings per share include the effect of convertible notes.",
            )]),
            (280.0, &[(50.0, "See accompanying notes to consolidated financial statements.")]),
        ])
    }

    #[test]
    #[rustfmt::skip]
    fn an_income_statement_scores_every_triple() {
        let t = table(&income_statement());
        let g = grid(&t);
        let want = gold(
            &["2023", "2022", "2021"],
            &[
                ("Revenues", &["$ 96,773", "$ 81,462", "$ 53,823"]),
                ("Cost of revenues", &["79,113", "65,121", "40,217"]),
                ("Gross profit", &["17,660", "16,341", "13,606"]),
                ("Research and development", &["3,969", "3,075", "2,593"]),
                ("Selling, general and administrative", &["4,800", "3,946", "4,517"]),
                ("Restructuring and other", &["—", "(34)", "(27)"]),
                ("Income from operations", &["8,891", "9,354", "6,523"]),
                ("Net income", &["$ 14,997", "$ 12,556", "$ 5,519"]),
                ("Diluted (1)", &["$ 4.30", "$ 3.62", "$ 1.63"]),
            ],
        );
        assert_eq!(triples(&t), want);
        assert_eq!(g.caption.len(), 2);
        // The period header stays one cell over the three years, never copied into each.
        let span = |text: &str, span| Cell {
            text: text.into(),
            span,
        };
        assert_eq!(g.header[0], [span("", 1), span("Year Ended December 31,", 3)]);
        // Section labels stay rows of their own, their cells blank.
        assert!(g.body.contains(&vec!["Operating expenses:".into(), String::new(), String::new(), String::new()]));
        assert_eq!(g.notes.len(), 2);
        assert!(g.notes[0].starts_with("(1) Diluted"));
    }

    #[rustfmt::skip]
    fn datasheet() -> Vec<Word> {
        let mut words = page(&[
            (40.0, &[(40.0, "Table 5. Electrical Characteristics")]),
            (54.0, &[(40.0, "VDD = 3.3 V, TA = 25 °C, unless otherwise noted.")]),
            (76.0, &[
                (40.0, "Parameter"),
                (170.0, "Symbol"),
                (220.0, "Conditions"),
                (372.5, "Limits"),
                (470.0, "Unit"),
            ]),
            (90.0, &[(330.0, "Min"), (380.0, "Typ"), (430.0, "Max")]),
            (108.0, &[
                (40.0, "Supply voltage"),
                (170.0, "VDD"),
                (330.0, "2.7"),
                (380.0, "3.3"),
                (430.0, "3.6"),
                (470.0, "V"),
            ]),
            (122.0, &[
                (40.0, "Quiescent current"),
                (170.0, "IQ"),
                (220.0, "No load"),
                (380.0, "45"),
                (430.0, "60"),
                (470.0, "µA"),
            ]),
            (136.0, &[
                (40.0, "Shutdown current"),
                (170.0, "ISD"),
                (220.0, "EN = 0 V"),
                (380.0, "0.1"),
                (430.0, "1"),
                (470.0, "µA"),
            ]),
            (150.0, &[
                (40.0, "Output voltage accuracy"),
                (220.0, "IOUT = 1 mA"),
                (330.0, "-2"),
                (430.0, "+2"),
                (470.0, "%"),
            ]),
            (164.0, &[
                (40.0, "Dropout voltage"),
                (170.0, "VDO"),
                (220.0, "IOUT = 150 mA"),
                (380.0, "120"),
                (430.0, "200"),
                (470.0, "mV"),
            ]),
            (182.0, &[(40.0, "(1) Guaranteed by design; not tested in production.")]),
            (194.0, &[(40.0, "(2) At TA = 85 °C.")]),
        ]);
        // Raised footnote marks, set smaller: after "Quiescent current" and after the "1" µA.
        for (x, y) in [(40.0 + width("Quiescent current") + 1.0, 120.0), (436.0, 134.0)] {
            let text = if y < 130.0 { "(1)" } else { "(2)" };
            words.push(Word {
                text: text.into(),
                x0: x,
                y0: y,
                x1: x + 9.0,
                y1: y + 6.0,
            });
        }
        words
    }

    #[test]
    #[rustfmt::skip]
    fn a_datasheet_keeps_its_merged_limits_header_units_and_footnotes() {
        let t = table(&datasheet());
        let g = grid(&t);
        let want = gold(
            &["Symbol", "Conditions", "Min", "Typ", "Max", "Unit"],
            &[
                ("Supply voltage", &["VDD", "", "2.7", "3.3", "3.6", "V"]),
                ("Quiescent current (1)", &["IQ", "No load", "", "45", "60", "µA"]),
                ("Shutdown current", &["ISD", "EN = 0 V", "", "0.1", "1 (2)", "µA"]),
                ("Output voltage accuracy", &["", "IOUT = 1 mA", "-2", "", "+2", "%"]),
                ("Dropout voltage", &["VDO", "IOUT = 150 mA", "", "120", "200", "mV"]),
            ],
        );
        assert_eq!(triples(&t), want);
        let texts = |row: &[Cell]| row.iter().map(|c| (c.text.clone(), c.span)).collect::<Vec<_>>();
        let cell = |t: &str, span| (t.to_string(), span);
        assert_eq!(
            texts(&g.header[0]),
            [cell("Parameter", 1), cell("Symbol", 1), cell("Conditions", 1), cell("Limits", 3), cell("Unit", 1)]
        );
        assert_eq!(g.caption[0], "Table 5. Electrical Characteristics");
        assert_eq!(g.notes, ["(1) Guaranteed by design; not tested in production.", "(2) At TA = 85 °C."]);
        let md = t.markdown();
        assert!(md.contains("| Parameter | Symbol | Conditions | Limits ||| Unit |\n|  |  |  | Min | Typ | Max |  |\n|---|---|---|---|---|---|---|\n| Supply voltage | VDD |  | 2.7 | 3.3 | 3.6 | V |"), "{md}");
        assert!(md.ends_with("|\n\n(1) Guaranteed by design; not tested in production.\n(2) At TA = 85 °C."), "{md}");
    }

    #[test]
    #[rustfmt::skip]
    fn a_min_column_most_rows_leave_blank_keeps_each_value_under_its_header() {
        // An adjustable regulator's table: the text layer drops a blank MIN, which shifts TYP
        // and MAX one column left (both of today's paths, on the Mac); boxes keep x.
        let t = table(&page(&[
            (40.0, &[(40.0, "PARAMETER"), (150.0, "TEST CONDITIONS"), (330.0, "MIN"), (370.0, "TYP"), (410.0, "MAX"), (450.0, "UNIT")]),
            (58.0, &[(40.0, "Line regulation"), (150.0, "3 V ≤ VI − VO ≤ 40 V"), (370.0, "0.01"), (410.0, "0.04"), (450.0, "%/V")]),
            (72.0, &[(40.0, "Load regulation"), (150.0, "IO = 10 mA to 1.5 A"), (370.0, "0.1"), (410.0, "0.5"), (450.0, "%")]),
            (86.0, &[(40.0, "Thermal regulation"), (150.0, "20-ms pulse"), (370.0, "0.03"), (410.0, "0.07"), (450.0, "%/W")]),
            (100.0, &[(40.0, "Adjustment pin current"), (370.0, "50"), (410.0, "100"), (450.0, "µA")]),
            (114.0, &[(40.0, "Reference voltage"), (150.0, "IO = 10 mA to 1.5 A"), (330.0, "1.2"), (370.0, "1.25"), (410.0, "1.3"), (450.0, "V")]),
            (128.0, &[(40.0, "Temperature stability"), (150.0, "TJ = 0 °C to 125 °C"), (370.0, "1"), (450.0, "%")]),
            (142.0, &[(40.0, "Minimum load current"), (150.0, "VI − VO = 40 V"), (370.0, "3.5"), (410.0, "10"), (450.0, "mA")]),
            (156.0, &[(40.0, "Ripple rejection"), (150.0, "VO = 10 V, f = 120 Hz"), (370.0, "65"), (450.0, "dB")]),
            (170.0, &[(40.0, "RMS output noise"), (150.0, "f = 10 Hz to 10 kHz"), (370.0, "0.003"), (450.0, "%")]),
        ]));
        let want = gold(
            &["TEST CONDITIONS", "MIN", "TYP", "MAX", "UNIT"],
            &[
                ("Line regulation", &["3 V ≤ VI − VO ≤ 40 V", "", "0.01", "0.04", "%/V"]),
                ("Load regulation", &["IO = 10 mA to 1.5 A", "", "0.1", "0.5", "%"]),
                ("Thermal regulation", &["20-ms pulse", "", "0.03", "0.07", "%/W"]),
                ("Adjustment pin current", &["", "", "50", "100", "µA"]),
                ("Reference voltage", &["IO = 10 mA to 1.5 A", "1.2", "1.25", "1.3", "V"]),
                ("Temperature stability", &["TJ = 0 °C to 125 °C", "", "1", "", "%"]),
                ("Minimum load current", &["VI − VO = 40 V", "", "3.5", "10", "mA"]),
                ("Ripple rejection", &["VO = 10 V, f = 120 Hz", "", "65", "", "dB"]),
                ("RMS output noise", &["f = 10 Hz to 10 kHz", "", "0.003", "", "%"]),
            ],
        );
        assert_eq!(triples(&t), want);
        // One MIN in ten rows is too little ink for a dense column: the sparse one holds it.
        assert_eq!(grid(&t).body[4][2], "1.2");
    }

    fn revenue_page(caption: &str, rows: &[(&str, &str, &str)]) -> Vec<Word> {
        let mut layout: Vec<(f64, Vec<(f64, &str)>)> = vec![
            (40.0, vec![(50.0, caption)]),
            (
                60.0,
                vec![(50.0, "Region"), r(300.0, "2023"), r(380.0, "2022")],
            ),
        ];
        for (i, &(region, a, b)) in rows.iter().enumerate() {
            layout.push((
                78.0 + 14.0 * i as f64,
                vec![(50.0, region), r(300.0, a), r(380.0, b)],
            ));
        }
        let rows: Vec<(f64, &[(f64, &str)])> =
            layout.iter().map(|(y, c)| (*y, c.as_slice())).collect();
        page(&rows)
    }

    #[test]
    #[rustfmt::skip]
    fn a_table_that_continues_on_the_next_page_is_joined_once() {
        let first = revenue_page(
            "Table 3. Revenue by region (in millions)",
            &[("Americas", "1,204", "1,118"), ("Europe", "830", "792"), ("Japan", "301", "288")],
        );
        let next = revenue_page(
            "Table 3. Revenue by region (continued)",
            &[("Greater China", "512", "547"), ("Rest of Asia", "199", "176"), ("Total", "3,046", "2,921")],
        );
        let joined = join(vec![table(&first), table(&next)]);
        assert_eq!(joined.len(), 1, "{joined:#?}");
        let want = gold(
            &["2023", "2022"],
            &[
                ("Americas", &["1,204", "1,118"]),
                ("Europe", &["830", "792"]),
                ("Japan", &["301", "288"]),
                ("Greater China", &["512", "547"]),
                ("Rest of Asia", &["199", "176"]),
                ("Total", &["3,046", "2,921"]),
            ],
        );
        assert_eq!(triples(&joined[0]), want);
        assert_eq!(grid(&joined[0]).header.len(), 1);
        // A new caption starts a new table, whatever its columns.
        let other = revenue_page("Table 4. Headcount by region", &[("Americas", "410", "388"), ("Europe", "120", "97")]);
        assert_eq!(join(vec![table(&first), table(&other)]).len(), 2);
    }

    #[test]
    #[rustfmt::skip]
    fn a_misaligned_table_is_layout_uncertain_with_its_lines_in_reading_order() {
        let t = table(&page(&[
            (40.0, &[(40.0, "Item"), (150.0, "Qty"), (220.0, "Price")]),
            (54.0, &[(40.0, "Widget, large"), (170.0, "12"), (240.0, "4.50")]),
            (68.0, &[(40.0, "Bolt"), (100.0, "300"), (160.0, "0.05")]),
            (82.0, &[(40.0, "Gasket set for pumps"), (160.0, "7"), (200.0, "18.00")]),
        ]));
        let lines = ["Item Qty Price", "Widget, large 12 4.50", "Bolt 300 0.05", "Gasket set for pumps 7 18.00"];
        assert_eq!(t, Table::Uncertain(lines.map(String::from).into()));
        assert_eq!(t.markdown(), format!("{UNCERTAIN}\n{}", lines.join("\n")));
    }

    /// `pdftotext -bbox-layout` (poppler 24.02) on a datasheet table made with ReportLab, a
    /// raised "(1)" after "Supply current"; the head and the flow, block and line elements left
    /// out, as the parser reads only words.
    const POPPLER: &str = r#"<body>
<doc>
  <page width="400.000000" height="220.000000">
          <word xMin="20.000000" yMin="13.538000" xMax="42.509000" yMax="21.863000">Table</word>
          <word xMin="45.011000" yMin="13.538000" xMax="52.517000" yMax="21.863000">2.</word>
          <word xMin="55.019000" yMin="13.538000" xMax="91.523000" yMax="21.863000">Electrical</word>
          <word xMin="94.025000" yMin="13.538000" xMax="151.535000" yMax="21.863000">characteristics</word>
          <word xMin="20.000000" yMin="37.538000" xMax="62.012000" yMax="45.863000">Parameter</word>
          <word xMin="120.000000" yMin="37.538000" xMax="150.006000" yMax="45.863000">Symbol</word>
          <word xMin="212.000000" yMin="37.538000" xMax="235.499000" yMax="45.863000">Limits</word>
          <word xMin="300.000000" yMin="37.538000" xMax="316.002000" yMax="45.863000">Unit</word>
          <word xMin="170.000000" yMin="51.538000" xMax="184.499000" yMax="59.863000">Min</word>
          <word xMin="215.000000" yMin="51.538000" xMax="230.003000" yMax="59.863000">Typ</word>
          <word xMin="260.000000" yMin="51.538000" xMax="277.001000" yMax="59.863000">Max</word>
          <word xMin="170.000000" yMin="69.538000" xMax="182.510000" yMax="77.863000">2.7</word>
          <word xMin="20.000000" yMin="69.538000" xMax="47.513000" yMax="77.863000">Supply</word>
          <word xMin="50.015000" yMin="69.538000" xMax="79.031000" yMax="77.863000">voltage</word>
          <word xMin="120.000000" yMin="69.538000" xMax="138.999000" yMax="77.863000">VDD</word>
          <word xMin="215.000000" yMin="69.538000" xMax="227.510000" yMax="77.863000">3.3</word>
          <word xMin="260.000000" yMin="69.538000" xMax="272.510000" yMax="77.863000">3.6</word>
          <word xMin="300.000000" yMin="69.538000" xMax="306.003000" yMax="77.863000">V</word>
          <word xMin="20.000000" yMin="83.538000" xMax="47.513000" yMax="91.863000">Supply</word>
          <word xMin="50.015000" yMin="83.538000" xMax="78.023000" yMax="91.863000">current</word>
          <word xMin="83.500000" yMin="82.692000" xMax="90.832000" yMax="88.242000">(1)</word>
          <word xMin="120.000000" yMin="83.538000" xMax="135.498000" yMax="91.863000">IDD</word>
          <word xMin="215.000000" yMin="83.538000" xMax="225.008000" yMax="91.863000">45</word>
          <word xMin="260.000000" yMin="83.538000" xMax="270.008000" yMax="91.863000">60</word>
          <word xMin="300.000000" yMin="83.538000" xMax="311.007000" yMax="91.863000">µA</word>
          <word xMin="20.000000" yMin="97.538000" xMax="60.023000" yMax="105.863000">Shutdown</word>
          <word xMin="62.525000" yMin="97.538000" xMax="90.533000" yMax="105.863000">current</word>
          <word xMin="120.000000" yMin="97.538000" xMax="135.003000" yMax="105.863000">ISD</word>
          <word xMin="215.000000" yMin="97.538000" xMax="227.510000" yMax="105.863000">0.1</word>
          <word xMin="260.000000" yMin="97.538000" xMax="265.004000" yMax="105.863000">1</word>
          <word xMin="300.000000" yMin="97.538000" xMax="311.007000" yMax="105.863000">µA</word>
          <word xMin="20.000000" yMin="117.538000" xMax="30.998000" yMax="125.863000">(1)</word>
          <word xMin="33.500000" yMin="117.538000" xMax="45.002000" yMax="125.863000">No</word>
          <word xMin="47.504000" yMin="117.538000" xMax="67.016000" yMax="125.863000">load,</word>
          <word xMin="69.518000" yMin="117.538000" xMax="88.517000" yMax="125.863000">VDD</word>
          <word xMin="91.019000" yMin="117.538000" xMax="96.275000" yMax="125.863000">=</word>
          <word xMin="98.777000" yMin="117.538000" xMax="111.287000" yMax="125.863000">3.3</word>
          <word xMin="113.789000" yMin="117.538000" xMax="119.792000" yMax="125.863000">V</word>
          <word xMin="122.294000" yMin="117.538000" xMax="128.297000" yMax="125.863000">&amp;</word>
          <word xMin="130.799000" yMin="117.538000" xMax="142.301000" yMax="125.863000">TA</word>
          <word xMin="144.803000" yMin="117.538000" xMax="150.059000" yMax="125.863000">=</word>
          <word xMin="152.561000" yMin="117.538000" xMax="162.569000" yMax="125.863000">25</word>
          <word xMin="165.071000" yMin="117.538000" xMax="177.671000" yMax="125.863000">°C.</word>
  </page>
</doc>
</body>"#;

    #[test]
    fn poppler_word_boxes_parse_and_make_their_table() {
        let words = bbox_words(POPPLER).unwrap();
        assert_eq!(words.len(), 43);
        assert_eq!(
            words[38],
            Word {
                text: "&".into(),
                x0: 122.294,
                y0: 117.538,
                x1: 128.297,
                y1: 125.863
            }
        );
        let t = table(&words);
        let want = gold(
            &["Symbol", "Min", "Typ", "Max", "Unit"],
            &[
                ("Supply voltage", &["VDD", "2.7", "3.3", "3.6", "V"]),
                ("Supply current (1)", &["IDD", "", "45", "60", "µA"]),
                ("Shutdown current", &["ISD", "", "0.1", "1", "µA"]),
            ],
        );
        assert_eq!(triples(&t), want);
        assert_eq!(
            grid(&t).header[0][2],
            Cell {
                text: "Limits".into(),
                span: 3
            }
        );
        assert_eq!(grid(&t).notes, ["(1) No load, VDD = 3.3 V & TA = 25 °C."]);
        // Poppler's own escapes, as it printed them, and an escaped escape.
        let escaped = bbox_words(
            r#"<word xMin="20.000000" yMin="43.538000" xMax="35.516000" yMax="51.863000">&lt;a&gt;</word>
          <word xMin="38.018000" yMin="43.538000" xMax="49.412000" yMax="51.863000">&quot;b&quot;</word>
          <word xMin="51.914000" yMin="43.538000" xMax="59.852000" yMax="51.863000">&apos;c&apos;</word>
          <word xMin="62.354000" yMin="43.538000" xMax="80.357000" yMax="51.863000">&amp;lt;</word>"#,
        )
        .unwrap();
        let texts: Vec<&str> = escaped.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["<a>", "\"b\"", "'c'", "&lt;"]);
    }

    /// VISION's words of an adjustable regulator's electrical characteristics, as it prints them:
    /// a 612-point page, its MIN, TYP and MAX headers at x 478 to 548 as on LM317's, each value
    /// centered under its header, and MIN left blank in most rows.
    // ponytail: written in the script's format, as no Mac is here; replace it with the words the
    // Mac records on LM317's electrical-characteristics page.
    const VISION_WORDS: &str = "41.53\t91.78\t53.56\t100.72\t6.5\n\
                                54.82\t91.62\t87.94\t100.87\tElectrical\n\
                                89.29\t92.05\t141.79\t100.46\tCharacteristics\n\
                                41.72\t110.36\t91.66\t118.31\tPARAMETER\n\
                                209.66\t110.18\t231.72\t118.53\tTEST\n\
                                233.25\t110.24\t281.21\t118.91\tCONDITIONS\n\
                                477.81\t109.78\t492.95\t118.72\tMIN\n\
                                502.27\t109.62\t519.29\t118.87\tTYP\n\
                                530.35\t110.05\t548.42\t118.46\tMAX\n\
                                565.60\t110.36\t584.42\t118.31\tUNIT\n\
                                41.66\t124.18\t57.81\t132.53\tLine\n\
                                59.34\t124.24\t93.47\t132.91\tregulation\n\
                                209.53\t123.78\t218.00\t132.72\tVI\n\
                                219.26\t123.62\t224.95\t132.87\t−\n\
                                226.30\t124.05\t237.71\t132.46\tVO\n\
                                239.30\t124.36\t244.55\t132.31\t=\n\
                                246.13\t124.18\t251.30\t132.53\t3\n\
                                252.82\t124.24\t258.98\t132.91\tV\n\
                                260.23\t123.78\t267.70\t132.72\tto\n\
                                268.96\t123.62\t278.87\t132.87\t40\n\
                                280.22\t124.05\t286.30\t132.46\tV\n\
                                502.74\t124.36\t518.89\t132.31\t0.01\n\
                                531.28\t124.18\t547.56\t132.53\t0.04\n\
                                567.34\t124.24\t582.84\t132.91\t%/V\n\
                                41.53\t137.78\t95.79\t146.72\tADJUSTMENT\n\
                                97.05\t137.62\t108.16\t146.87\tpin\n\
                                109.51\t138.05\t134.09\t146.46\tcurrent\n\
                                506.07\t138.36\t515.55\t146.31\t50\n\
                                532.39\t138.18\t546.45\t146.53\t100\n\
                                569.79\t138.24\t580.40\t146.91\tµA\n\
                                41.53\t151.78\t77.54\t160.72\tReference\n\
                                78.79\t151.62\t104.79\t160.87\tvoltage\n\
                                209.62\t152.05\t217.91\t160.46\tIO\n\
                                219.50\t152.36\t224.76\t160.31\t=\n\
                                226.34\t152.18\t235.95\t160.53\t10\n\
                                237.48\t152.24\t250.04\t160.91\tmA\n\
                                251.29\t151.78\t258.76\t160.72\tto\n\
                                260.02\t151.62\t272.15\t160.87\t1.5\n\
                                273.50\t152.05\t279.58\t160.46\tA\n\
                                479.56\t152.36\t491.26\t160.31\t1.2\n\
                                502.68\t152.18\t518.96\t160.53\t1.25\n\
                                533.52\t152.24\t545.46\t160.91\t1.3\n\
                                571.86\t151.78\t578.11\t160.72\tV\n\
                                41.47\t165.62\t86.39\t174.87\tTemperature\n\
                                87.74\t166.05\t115.24\t174.46\tstability\n\
                                209.72\t166.36\t219.64\t174.31\tTJ\n\
                                221.22\t166.18\t226.61\t174.53\t=\n\
                                228.14\t166.24\t241.94\t174.91\t0°C\n\
                                243.19\t165.78\t250.66\t174.72\tto\n\
                                251.92\t165.62\t274.82\t174.87\t125°C\n\
                                504.86\t166.05\t516.71\t174.46\t0.7\n\
                                565.83\t166.36\t584.20\t174.31\t%VO\n\
                                41.66\t180.18\t73.71\t188.53\tMinimum\n\
                                75.24\t180.24\t90.32\t188.91\tload\n\
                                91.57\t179.78\t116.32\t188.72\tcurrent\n\
                                209.47\t179.62\t218.05\t188.87\tVI\n\
                                219.40\t180.05\t224.81\t188.46\t−\n\
                                226.40\t180.36\t237.66\t188.31\tVO\n\
                                239.24\t180.18\t244.62\t188.53\t=\n\
                                246.15\t180.24\t255.87\t188.91\t40\n\
                                257.12\t179.78\t263.37\t188.72\tV\n\
                                504.71\t179.62\t516.85\t188.87\t3.5\n\
                                534.57\t180.05\t544.20\t188.46\t10\n\
                                568.85\t180.36\t581.17\t188.31\tmA\n\
                                41.66\t194.18\t76.10\t202.53\tMaximum\n\
                                77.62\t194.24\t99.89\t202.91\toutput\n\
                                101.13\t193.78\t125.89\t202.72\tcurrent\n\
                                209.47\t193.62\t218.05\t202.87\tVI\n\
                                219.40\t194.05\t224.81\t202.46\t−\n\
                                226.40\t194.36\t237.66\t202.31\tVO\n\
                                239.24\t194.18\t244.34\t202.53\t≤\n\
                                245.87\t194.24\t255.59\t202.91\t15\n\
                                256.84\t193.78\t263.09\t202.72\tV\n\
                                479.31\t193.62\t491.45\t202.87\t1.5\n\
                                504.86\t194.05\t516.71\t202.46\t2.2\n\
                                572.05\t194.36\t577.97\t202.31\tA\n";

    #[test]
    #[rustfmt::skip]
    fn vision_words_put_a_sparse_min_column_and_its_neighbours_under_their_headers() {
        let words = boxes(VISION_WORDS).unwrap();
        assert_eq!(words.len(), 76);
        let t = table(&words);
        let want = gold(
            &["TEST CONDITIONS", "MIN", "TYP", "MAX", "UNIT"],
            &[
                ("Line regulation", &["VI − VO = 3 V to 40 V", "", "0.01", "0.04", "%/V"]),
                ("ADJUSTMENT pin current", &["", "", "50", "100", "µA"]),
                ("Reference voltage", &["IO = 10 mA to 1.5 A", "1.2", "1.25", "1.3", "V"]),
                ("Temperature stability", &["TJ = 0°C to 125°C", "", "0.7", "", "%VO"]),
                ("Minimum load current", &["VI − VO = 40 V", "", "3.5", "10", "mA"]),
                ("Maximum output current", &["VI − VO ≤ 15 V", "1.5", "2.2", "", "A"]),
            ],
        );
        assert_eq!(triples(&t), want);
        assert_eq!(grid(&t).caption, ["6.5 Electrical Characteristics"]);
    }

    #[test]
    fn word_boxes_parse_and_bad_boxes_are_errors() {
        // In the shape VISION prints: a turned page's words come out in the turned frame.
        let words =
            boxes("72.00\t90.25\t101.40\t99.75\tRevenue\n\n310.10\t90.25\t340.00\t99.75\t96,773\n")
                .unwrap();
        assert_eq!(words.len(), 2);
        assert_eq!(
            (words[1].text.as_str(), words[1].x0, words[1].y1),
            ("96,773", 310.1, 99.75)
        );
        assert_eq!(boxes(&tsv(&words)).unwrap(), words);
        assert!(boxes("72\t90\t60\t99\tbackwards").is_err());
        assert!(boxes("72\t90\tNaN\t99\tx").is_err());
        assert!(boxes("72 90 101 99 Revenue").is_err());
    }

    #[test]
    fn years_name_columns_and_amounts_count() {
        assert!(!counts("2023") && !counts("Q1") && !counts("FY2023") && !counts("—"));
        assert!(counts("$ 96,773") && counts("(34)") && counts("-2") && counts("December 31,"));
    }
}
