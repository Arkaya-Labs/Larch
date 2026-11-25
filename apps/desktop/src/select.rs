//! Text the mouse selects and Ctrl+C copies. Each piece of text (a line, a list item, a heading,
//! a code block, a message, a step, a figure in the side panel) registers its layout as it
//! renders, in reading order, so a drag runs from piece to piece as it does from line to line.
// ponytail: a drag past the thread's top or bottom edge does not scroll it; scroll while dragging
// when selections longer than a screen come up.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::ops::Range;
use std::rc::Rc;

use gpui::{
    CursorStyle, Div, Font, HighlightStyle, Hsla, ParentElement, Pixels, Point, SharedString,
    Styled, StyledText, TextLayout, TextRun, div,
};

/// A piece's place in its thread, (item, block, line): reading order is key order.
pub type Key = (usize, usize, usize);

/// Key items from here up are places outside the thread, each numbering its pieces as they
/// render (`Selection::label`); a drag stays in the place it started in.
const PLACES: usize = usize::MAX / 2;
/// The chat's side panel.
pub const SIDE: usize = PLACES;
/// The toast.
pub const TOAST: usize = PLACES + 1;
/// A card asking the user (the call it would send).
pub const ASK: usize = PLACES + 2;
/// The page open beside the chat (MCP tools, …).
pub const PAGE: usize = PLACES + 3;
/// The popover of what needs the user.
pub const POP: usize = PLACES + 4;
/// The screen over everything when the engine stopped or never started.
pub const ALERT: usize = PLACES + 5;

/// The place `key` is in: the thread, or one of the `PLACES`.
fn place(key: Key) -> usize {
    if key.0 < PLACES { 0 } else { key.0 }
}

struct Piece {
    key: Key,
    /// Shared with the rendered text: every frame registers the whole thread again.
    text: SharedString,
    layout: TextLayout,
    /// Set once the layout is prepainted: every TextLayout query before that panics.
    ready: Rc<Cell<bool>>,
}

/// The window's selectable text, as last rendered, and what is selected in it.
#[derive(Default)]
pub struct Selection {
    pieces: RefCell<Vec<Piece>>,
    /// Each place's pieces so far this frame (`label`).
    counts: RefCell<BTreeMap<usize, usize>>,
    /// The selected part's background (`restart`).
    bg: Cell<Hsla>,
    anchor: Option<(Key, usize)>,
    head: Option<(Key, usize)>,
    /// The left button went down on text and is still held.
    pub dragging: bool,
}

impl Selection {
    /// Forgets the last render's pieces: the window's render registers them again, a selection
    /// on `bg`.
    pub fn restart(&self, bg: Hsla) {
        self.pieces.borrow_mut().clear();
        self.counts.borrow_mut().clear();
        self.bg.set(bg);
    }

    /// Nothing selected: another chat or page opened.
    pub fn clear(&mut self) {
        (self.anchor, self.head, self.dragging) = (None, None, false);
    }

    /// Whether some of `place`'s text is selected.
    pub fn holds(&self, place: usize) -> bool {
        (self.anchor.zip(self.head))
            .is_some_and(|(a, h)| (self::place(a.0), self::place(h.0)) == (place, place) && a != h)
    }

    /// `text` in `runs`, registered at `key`, its selected part on `bg`.
    pub fn text(&self, key: Key, text: String, runs: Vec<TextRun>, bg: Hsla) -> Div {
        let runs = match self.range(key, &text) {
            Some(r) => highlight(runs, r, bg),
            None => runs,
        };
        let text = SharedString::from(text);
        self.register(key, StyledText::new(text.clone()).with_runs(runs), text)
    }

    /// `text` in the style it inherits, registered at `key`.
    pub fn styled(&self, key: Key, text: impl Into<SharedString>) -> Div {
        let text = text.into();
        let on = HighlightStyle {
            background_color: Some(self.bg.get()),
            ..HighlightStyle::default()
        };
        let lit = self.range(key, &text).map(|r| (r, on));
        self.register(
            key,
            StyledText::new(text.clone()).with_highlights(lit),
            text,
        )
    }

    /// `text` in the style it inherits, as `place`'s next piece this frame.
    pub fn label(&self, place: usize, text: impl Into<SharedString>) -> Div {
        let n = {
            let mut counts = self.counts.borrow_mut();
            let n = counts.entry(place).or_default();
            *n += 1;
            *n
        };
        self.styled((place, n, 0), text)
    }

    fn register(&self, key: Key, styled: StyledText, text: SharedString) -> Div {
        let ready = Rc::new(Cell::new(false));
        self.pieces.borrow_mut().push(Piece {
            key,
            text,
            layout: styled.layout().clone(),
            ready: ready.clone(),
        });
        div()
            .cursor(CursorStyle::IBeam)
            .on_children_prepainted(move |_, _, _| ready.set(true))
            .child(styled)
    }

    /// Where the piece at `key` starts on screen, as last rendered.
    pub fn top(&self, key: Key) -> Option<Pixels> {
        let pieces = self.pieces.borrow();
        let p = pieces.iter().find(|x| x.key == key && x.ready.get())?;
        Some(p.layout.bounds().top())
    }

    /// A press at `p`: on text it starts a selection (a word on a double click, the piece on a
    /// triple, an extension with shift) and is true; elsewhere it clears the selection.
    pub fn down(&mut self, p: Point<Pixels>, clicks: usize, shift: bool) -> bool {
        let Some((key, i, text)) = self.at(p, None) else {
            self.clear();
            return false;
        };
        let (a, h) = match clicks {
            2 => word(&text, i),
            3 => (0, text.len()),
            _ => (i, i),
        };
        // Shift extends a selection in the same place only.
        if !(shift && self.anchor.is_some_and(|(k, _)| place(k) == place(key))) {
            self.anchor = Some((key, a));
        }
        self.head = Some((key, h));
        self.dragging = true;
        true
    }

    /// The held button moved to `p`: the selection ends at the text nearest it in the place it
    /// started in.
    pub fn drag(&mut self, p: Point<Pixels>) {
        let Some((start, _)) = self.anchor else {
            return;
        };
        if let Some((key, i, _)) = self.at(p, Some(place(start))) {
            self.head = Some((key, i));
        }
    }

    /// The chat item whose text is under `p` (a key's first part), as last rendered; text in
    /// one of the `PLACES` outside the thread is no chat item.
    pub fn item_at(&self, p: Point<Pixels>) -> Option<usize> {
        self.at(p, None)
            .and_then(|(key, _, _)| (place(key) == 0).then_some(key.0))
    }

    /// The selected text, its pieces on lines of their own; None when nothing is selected.
    pub fn copied(&self) -> Option<String> {
        let pieces = self.pieces.borrow();
        let parts: Vec<&str> = (pieces.iter())
            .filter_map(|p| self.range(p.key, &p.text).map(|r| &p.text[r]))
            .collect();
        (!parts.is_empty()).then(|| parts.join("\n"))
    }

    /// The piece under `p` and the byte there; `within` a place, the nearest of its pieces above
    /// it (a drag past a line's end, or between pieces, still selects).
    fn at(&self, p: Point<Pixels>, within: Option<usize>) -> Option<(Key, usize, SharedString)> {
        let pieces = self.pieces.borrow();
        let mut ready =
            (pieces.iter()).filter(|x| x.ready.get() && within.is_none_or(|w| place(x.key) == w));
        let piece = match within {
            None => ready.find(|x| x.layout.bounds().contains(&p)),
            Some(_) => {
                let first = ready.clone().next();
                ready.rfind(|x| x.layout.bounds().top() <= p.y).or(first)
            }
        }?;
        let i = piece.layout.index_for_position(p).unwrap_or_else(|i| i);
        Some((piece.key, floor(&piece.text, i), piece.text.clone()))
    }

    /// The bytes of the piece at `key` the selection covers, on its char boundaries.
    fn range(&self, key: Key, text: &str) -> Option<Range<usize>> {
        let (a, h) = (self.anchor?, self.head?);
        let (a, b) = if a <= h { (a, h) } else { (h, a) };
        if key < a.0 || key > b.0 {
            return None;
        }
        let start = if key == a.0 { floor(text, a.1) } else { 0 };
        let end = if key == b.0 {
            floor(text, b.1)
        } else {
            text.len()
        };
        (start < end).then_some(start..end)
    }
}

/// One run for all of a plain text.
pub fn plain(text: &str, font: Font, color: Hsla) -> Vec<TextRun> {
    vec![TextRun {
        len: text.len(),
        font,
        color,
        background_color: None,
        underline: None,
        strikethrough: None,
    }]
}

/// `runs` with the bytes `r` on `bg`, split where `r` starts and ends.
fn highlight(runs: Vec<TextRun>, r: Range<usize>, bg: Hsla) -> Vec<TextRun> {
    let mut out = Vec::with_capacity(runs.len() + 2);
    let mut at = 0;
    for run in runs {
        let end = at + run.len;
        let (s, e) = (r.start.clamp(at, end), r.end.clamp(at, end));
        for (from, to, on) in [(at, s, false), (s, e, true), (e, end, false)] {
            if to > from {
                out.push(TextRun {
                    len: to - from,
                    background_color: if on { Some(bg) } else { run.background_color },
                    ..run.clone()
                });
            }
        }
        at = end;
    }
    out
}

/// The word around byte `i`: what lies between the whitespace on either side.
fn word(text: &str, i: usize) -> (usize, usize) {
    let start = text[..i].rfind(char::is_whitespace).map_or(0, |w| {
        w + text[w..].chars().next().map_or(1, char::len_utf8)
    });
    let end = text[i..]
        .find(char::is_whitespace)
        .map_or(text.len(), |w| i + w);
    (start, end)
}

/// `i` within `text`, moved back to a char boundary: a piece may have shrunk since it was clicked
/// (a streamed `**` becomes bold).
fn floor(text: &str, i: usize) -> usize {
    let mut i = i.min(text.len());
    while !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(lens: &[usize]) -> Vec<TextRun> {
        (lens.iter())
            .flat_map(|&n| plain(&"x".repeat(n), gpui::font("Figtree"), gpui::black()))
            .collect()
    }

    fn shape(runs: &[TextRun]) -> Vec<(usize, bool)> {
        (runs.iter())
            .map(|r| (r.len, r.background_color == Some(gpui::red())))
            .collect()
    }

    #[test]
    fn a_selection_splits_the_runs_it_crosses_and_keeps_the_rest() {
        let got = highlight(runs(&[4, 6]), 2..7, gpui::red());
        assert_eq!(shape(&got), [(2, false), (2, true), (3, true), (3, false)]);
        let all = highlight(runs(&[4]), 0..4, gpui::red());
        assert_eq!(shape(&all), [(4, true)]);
    }

    #[test]
    fn a_selection_covers_whole_pieces_between_its_ends_in_reading_order() {
        let mut s = Selection::default();
        // Dragged upward: the head comes before the anchor.
        (s.anchor, s.head) = (Some(((3, 0, 1), 2)), Some(((1, 2, 0), 4)));
        assert_eq!(s.range((1, 2, 0), "hello world"), Some(4..11));
        assert_eq!(s.range((2, 0, 0), "between"), Some(0..7));
        assert_eq!(s.range((3, 0, 1), "end"), Some(0..2));
        assert_eq!(s.range((3, 0, 2), "after"), None);
        assert_eq!(s.range((1, 1, 0), "before"), None);
    }

    /// Thread keys share one place; each of the others is its own, and holds only a selection
    /// that starts and ends in it.
    #[test]
    fn a_place_holds_the_selection_made_in_it() {
        assert_eq!(place((0, 3, 1)), place((41, 0, 0)));
        assert_ne!(place((41, 0, 0)), place((SIDE, 1, 0)));
        assert_ne!(place((SIDE, 1, 0)), place((TOAST, 1, 0)));
        let mut s = Selection::default();
        (s.anchor, s.head) = (Some(((SIDE, 1, 0), 0)), Some(((SIDE, 2, 0), 3)));
        assert!(s.holds(SIDE) && !s.holds(TOAST) && !s.holds(0));
        assert_eq!(s.range((SIDE, 1, 0), "tok/s"), Some(0..5));
        assert_eq!(s.range((SIDE, 2, 0), "1500"), Some(0..3));
        // A press without a drag holds nothing.
        s.head = s.anchor;
        assert!(!s.holds(SIDE));
    }

    #[test]
    fn indices_land_on_char_boundaries_and_words_end_at_whitespace() {
        assert_eq!(floor("é", 1), 0);
        assert_eq!(floor("ab", 9), 2);
        assert_eq!(word("one two\u{3000}three", 5), (4, 7));
        assert_eq!(word("solo", 2), (0, 4));
    }
}
