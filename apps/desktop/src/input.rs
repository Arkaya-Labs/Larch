//! A text field: one line (search, a port, a fact) or a wrapping, growing one (the composer).
//! Adapted from gpui 0.2.2's examples/input.rs (Apache-2.0): IME, selection and the clipboard as
//! there, plus wrapping, several lines, Up/Down and Enter to send.

use std::ops::Range;

use gpui::{
    App, Bounds, ClipboardItem, Context, CursorStyle, ElementId, ElementInputHandler, Entity,
    EntityInputHandler, EventEmitter, FocusHandle, Focusable, GlobalElementId, LayoutId,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PaintQuad, Pixels, Point,
    SharedString, Style, TextAlign, TextRun, UTF16Selection, Window, WrappedLine, actions, div,
    fill, point, prelude::*, px, relative, size,
};
use unicode_segmentation::UnicodeSegmentation;

use crate::theme::{Theme, fade};

actions!(
    text_input,
    [
        Backspace,
        Delete,
        Left,
        Right,
        Up,
        Down,
        SelectLeft,
        SelectRight,
        SelectAll,
        Home,
        End,
        Paste,
        Cut,
        Copy,
        Newline,
        Submit,
        WordLeft,
        WordRight,
        DeleteWord
    ]
);

/// Enter in a field: the parent sends or saves.
pub struct Submitted;

/// A paste with no text in a multiline field (the composer): the clipboard's image, or None when
/// the window read none (the parent may ask the OS: `images::os_clipboard_png`).
pub struct PastedImage(pub Option<std::sync::Arc<gpui::Image>>);

/// Image files pasted into a multiline field (Windows: files copied in Explorer).
#[cfg_attr(not(windows), allow(dead_code))]
pub struct PastedFiles(pub Vec<std::path::PathBuf>);

/// A text field's state.
pub struct TextInput {
    focus: FocusHandle,
    pub text: String,
    placeholder: SharedString,
    multiline: bool,
    max_rows: usize,
    sel: Range<usize>,
    reversed: bool,
    marked: Option<Range<usize>>,
    /// Each paragraph's start offset, shaped line and top, from the last paint.
    lines: Vec<(usize, WrappedLine, Pixels)>,
    bounds: Option<Bounds<Pixels>>,
    line_height: Pixels,
    scroll: Pixels,
    /// How far a one-line field's text is moved left, so the cursor stays in view.
    scroll_x: Pixels,
    selecting: bool,
    pub theme: Theme,
    /// A key or token: shown as `*`s (one per byte, so offsets stay the text's) and never copied.
    pub secret: bool,
    /// The right-click menu's window point while it is open.
    menu: Option<Point<Pixels>>,
}

/// What a secret field shows: `*` for each byte of the text.
fn masked(text: &str) -> String {
    text.chars().map(|c| "*".repeat(c.len_utf8())).collect()
}

impl EventEmitter<Submitted> for TextInput {}
impl EventEmitter<PastedImage> for TextInput {}
impl EventEmitter<PastedFiles> for TextInput {}

impl TextInput {
    /// An empty field; `multiline` wraps and grows to `max_rows` rows.
    pub fn new(
        cx: &mut Context<Self>,
        placeholder: &str,
        multiline: bool,
        max_rows: usize,
        theme: Theme,
    ) -> Self {
        TextInput {
            focus: cx.focus_handle(),
            text: String::new(),
            placeholder: placeholder.to_owned().into(),
            multiline,
            max_rows,
            sel: 0..0,
            reversed: false,
            marked: None,
            lines: Vec::new(),
            bounds: None,
            line_height: px(20.),
            scroll: px(0.),
            scroll_x: px(0.),
            selecting: false,
            theme,
            secret: false,
            menu: None,
        }
    }

    /// Replaces the text and puts the cursor at its end.
    pub fn set_text(&mut self, text: &str, cx: &mut Context<Self>) {
        self.text = text.to_owned();
        self.sel = self.text.len()..self.text.len();
        self.marked = None;
        cx.notify();
    }

    /// Changes the placeholder.
    pub fn set_placeholder(&mut self, p: &str) {
        self.placeholder = p.to_owned().into();
    }

    /// Takes the text out, leaving the field empty.
    pub fn take(&mut self, cx: &mut Context<Self>) -> String {
        let t = std::mem::take(&mut self.text);
        self.sel = 0..0;
        self.marked = None;
        self.scroll = px(0.);
        self.scroll_x = px(0.);
        cx.notify();
        t
    }

    fn cursor(&self) -> usize {
        if self.reversed {
            self.sel.start
        } else {
            self.sel.end
        }
    }

    fn move_to(&mut self, o: usize, cx: &mut Context<Self>) {
        self.sel = o..o;
        self.reversed = false;
        cx.notify();
    }

    fn select_to(&mut self, o: usize, cx: &mut Context<Self>) {
        if self.reversed {
            self.sel.start = o
        } else {
            self.sel.end = o
        }
        if self.sel.end < self.sel.start {
            self.reversed = !self.reversed;
            self.sel = self.sel.end..self.sel.start;
        }
        cx.notify();
    }

    fn prev(&self, o: usize) -> usize {
        self.text
            .grapheme_indices(true)
            .rev()
            .find_map(|(i, _)| (i < o).then_some(i))
            .unwrap_or(0)
    }

    fn next(&self, o: usize) -> usize {
        self.text
            .grapheme_indices(true)
            .find_map(|(i, _)| (i > o).then_some(i))
            .unwrap_or(self.text.len())
    }

    fn word_prev(&self, o: usize) -> usize {
        word_start(&self.text[..o])
    }

    fn word_next(&self, o: usize) -> usize {
        let rest = &self.text[o..];
        let skip = rest.len() - rest.trim_start().len();
        o + skip
            + rest[skip..]
                .find(char::is_whitespace)
                .unwrap_or(rest.len() - skip)
    }

    fn position(&self, o: usize) -> Option<Point<Pixels>> {
        let (start, line, top) = self.lines.iter().rev().find(|(s, _, _)| *s <= o)?;
        let p = line.position_for_index(o - start, self.line_height)?;
        Some(point(p.x, p.y + *top))
    }

    fn index_at(&self, p: Point<Pixels>) -> usize {
        let Some(b) = self.bounds else { return 0 };
        let local = point(p.x - b.left() + self.scroll_x, p.y - b.top() + self.scroll);
        let Some((start, line, top)) = self
            .lines
            .iter()
            .rev()
            .find(|(_, _, t)| *t <= local.y)
            .or(self.lines.first())
        else {
            return 0;
        };
        let i = line
            .closest_index_for_position(
                point(local.x, (local.y - *top).max(px(0.))),
                self.line_height,
            )
            .unwrap_or_else(|e| e);
        // An empty field shapes its placeholder, whose indices are not the text's.
        floor_char(&self.text, start + i.min(line.len()))
    }

    fn vertical(&mut self, down: bool, cx: &mut Context<Self>) {
        let Some(p) = self.position(self.cursor()) else {
            return;
        };
        let Some(b) = self.bounds else { return };
        let y = p.y
            + if down {
                self.line_height * 1.5
            } else {
                -self.line_height * 0.5
            };
        let o = if y < px(0.) {
            0
        } else {
            self.index_at(point(
                b.left() + p.x - self.scroll_x,
                b.top() + y - self.scroll,
            ))
        };
        self.move_to(o, cx);
    }

    fn left(&mut self, _: &Left, _: &mut Window, cx: &mut Context<Self>) {
        let o = if self.sel.is_empty() {
            self.prev(self.cursor())
        } else {
            self.sel.start
        };
        self.move_to(o, cx)
    }
    fn right(&mut self, _: &Right, _: &mut Window, cx: &mut Context<Self>) {
        let o = if self.sel.is_empty() {
            self.next(self.sel.end)
        } else {
            self.sel.end
        };
        self.move_to(o, cx)
    }
    fn word_left(&mut self, _: &WordLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(self.word_prev(self.cursor()), cx)
    }
    fn word_right(&mut self, _: &WordRight, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(self.word_next(self.cursor()), cx)
    }
    fn up(&mut self, _: &Up, _: &mut Window, cx: &mut Context<Self>) {
        self.vertical(false, cx)
    }
    fn down(&mut self, _: &Down, _: &mut Window, cx: &mut Context<Self>) {
        self.vertical(true, cx)
    }
    fn select_left(&mut self, _: &SelectLeft, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.prev(self.cursor()), cx)
    }
    fn select_right(&mut self, _: &SelectRight, _: &mut Window, cx: &mut Context<Self>) {
        self.select_to(self.next(self.cursor()), cx)
    }
    fn select_all(&mut self, _: &SelectAll, _: &mut Window, cx: &mut Context<Self>) {
        self.move_to(0, cx);
        self.select_to(self.text.len(), cx)
    }
    fn home(&mut self, _: &Home, _: &mut Window, cx: &mut Context<Self>) {
        let c = self.cursor();
        self.move_to(self.text[..c].rfind('\n').map_or(0, |i| i + 1), cx)
    }
    fn end(&mut self, _: &End, _: &mut Window, cx: &mut Context<Self>) {
        let c = self.cursor();
        self.move_to(
            self.text[c..].find('\n').map_or(self.text.len(), |i| c + i),
            cx,
        )
    }
    fn backspace(&mut self, _: &Backspace, w: &mut Window, cx: &mut Context<Self>) {
        if self.sel.is_empty() {
            self.select_to(self.prev(self.cursor()), cx)
        }
        self.replace_text_in_range(None, "", w, cx)
    }
    fn delete(&mut self, _: &Delete, w: &mut Window, cx: &mut Context<Self>) {
        if self.sel.is_empty() {
            self.select_to(self.next(self.cursor()), cx)
        }
        self.replace_text_in_range(None, "", w, cx)
    }
    fn delete_word(&mut self, _: &DeleteWord, w: &mut Window, cx: &mut Context<Self>) {
        if self.sel.is_empty() {
            self.select_to(self.word_prev(self.cursor()), cx)
        }
        self.replace_text_in_range(None, "", w, cx)
    }
    fn newline(&mut self, _: &Newline, w: &mut Window, cx: &mut Context<Self>) {
        if self.multiline {
            self.replace_text_in_range(None, "\n", w, cx)
        }
    }
    fn submit(&mut self, _: &Submit, _: &mut Window, cx: &mut Context<Self>) {
        cx.emit(Submitted)
    }
    fn paste(&mut self, _: &Paste, w: &mut Window, cx: &mut Context<Self>) {
        // Windows: files copied in Explorer reach gpui as one string of paths run together, so an
        // image pasted as its path. Images among them attach, as a drop does; other paths paste
        // one a line.
        #[cfg(windows)]
        if let Some(files) = (arboard::Clipboard::new().ok())
            .and_then(|mut c| c.get().file_list().ok())
            .filter(|f| !f.is_empty())
        {
            let (pictures, rest): (Vec<_>, Vec<_>) =
                (files.into_iter()).partition(|p| self.multiline && crate::images::image_path(p));
            if !pictures.is_empty() {
                cx.emit(PastedFiles(pictures));
            }
            if !rest.is_empty() {
                let sep = if self.multiline { "\n" } else { " " };
                let t: Vec<String> = rest.iter().map(|p| p.display().to_string()).collect();
                self.replace_text_in_range(None, &t.join(sep), w, cx);
            }
            return;
        }
        let item = cx.read_from_clipboard();
        // WSLg offers an empty string for a Windows image: that is no text to paste.
        if let Some(t) = item
            .as_ref()
            .and_then(|i| i.text())
            .filter(|t| !t.is_empty())
        {
            // Windows text ends lines in \r\n: a kept \r rode along in the message, and in a
            // one-line field it pasted as two spaces.
            let t = t.replace("\r\n", "\n").replace('\r', "\n");
            let t = if self.multiline {
                t
            } else {
                t.replace('\n', " ")
            };
            self.replace_text_in_range(None, &t, w, cx);
        } else if self.multiline {
            // Text pastes as text; a clipboard with none attaches its image.
            let image = item.and_then(|i| crate::images::pasted_image(i.entries()));
            cx.emit(PastedImage(image));
        }
    }
    fn copy(&mut self, _: &Copy, _: &mut Window, cx: &mut Context<Self>) {
        if !self.sel.is_empty() && !self.secret {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.text[self.sel.clone()].to_owned(),
            ));
        }
    }
    fn cut(&mut self, _: &Cut, w: &mut Window, cx: &mut Context<Self>) {
        if !self.sel.is_empty() && !self.secret {
            cx.write_to_clipboard(ClipboardItem::new_string(
                self.text[self.sel.clone()].to_owned(),
            ));
            self.replace_text_in_range(None, "", w, cx)
        }
    }
    fn mouse_down(&mut self, e: &MouseDownEvent, w: &mut Window, cx: &mut Context<Self>) {
        w.focus(&self.focus);
        self.selecting = true;
        let o = self.index_at(e.position);
        if e.modifiers.shift {
            self.select_to(o, cx)
        } else {
            self.move_to(o, cx)
        }
        if e.click_count == 2 {
            self.sel = word_at(&self.text, o);
        }
    }
    /// A right-click: the field takes the keys and its menu opens at the pointer; the selection
    /// stays, so Cut and Copy act on it.
    fn right_down(&mut self, e: &MouseDownEvent, w: &mut Window, cx: &mut Context<Self>) {
        cx.stop_propagation();
        w.focus(&self.focus);
        self.menu = Some(e.position);
        cx.notify()
    }
    /// Esc closes this field's right-click menu; with none open it goes on to the window's
    /// Dismiss (a popover, the chat's menu).
    fn dismiss(&mut self, _: &crate::ui::Dismiss, _: &mut Window, cx: &mut Context<Self>) {
        if self.menu.take().is_some() {
            cx.notify()
        } else {
            cx.propagate()
        }
    }
    /// The open right-click menu: Cut, Copy, Paste, Select All.
    fn menu_el(&self, at: Point<Pixels>, cx: &mut Context<Self>) -> impl IntoElement {
        use crate::context_menu::{card, row};
        let th = &self.theme;
        let some = !self.sel.is_empty() && !self.secret;
        type Act = fn(&mut TextInput, &mut Window, &mut Context<TextInput>);
        let rows: [(&'static str, &str, bool, Act); 4] = [
            ("field-cut", "Cut", some, |s, w, cx| s.cut(&Cut, w, cx)),
            ("field-copy", "Copy", some, |s, w, cx| s.copy(&Copy, w, cx)),
            ("field-paste", "Paste", true, |s, w, cx| {
                s.paste(&Paste, w, cx)
            }),
            (
                "field-all",
                "Select All",
                !self.text.is_empty(),
                |s, w, cx| s.select_all(&SelectAll, w, cx),
            ),
        ];
        let card = card(th)
            .on_mouse_down_out(cx.listener(|s, _, _, cx| {
                s.menu = None;
                cx.notify()
            }))
            .children(rows.into_iter().map(|(id, label, on, act)| {
                row(th, id, label, on).when(on, |d| {
                    d.on_click(cx.listener(move |s, _, w, cx| {
                        cx.stop_propagation();
                        s.menu = None;
                        act(s, w, cx);
                        cx.notify()
                    }))
                })
            }));
        crate::context_menu::at(at, card)
    }
    fn mouse_up(&mut self, _: &MouseUpEvent, _: &mut Window, _: &mut Context<Self>) {
        self.selecting = false;
    }
    fn mouse_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if self.selecting {
            self.select_to(self.index_at(e.position), cx);
        }
    }

    fn to16(&self, o: usize) -> usize {
        self.text[..o.min(self.text.len())].encode_utf16().count()
    }
    fn from16(&self, o: usize) -> usize {
        utf8_at(&self.text, o)
    }
    fn range16(&self, r: &Range<usize>) -> Range<usize> {
        self.to16(r.start)..self.to16(r.end)
    }
    fn range8(&self, r: &Range<usize>) -> Range<usize> {
        self.from16(r.start)..self.from16(r.end)
    }

    /// What an edit replaces: the IME's range, else the marked text, else the selection, always
    /// inside the text. A panic here runs inside the OS's text callback, which aborts the app.
    fn target(&self, r: Option<Range<usize>>) -> Range<usize> {
        let r = r
            .as_ref()
            .map(|r| self.range8(r))
            .or(self.marked.clone())
            .unwrap_or(self.sel.clone());
        let end = floor_char(&self.text, r.end);
        floor_char(&self.text, r.start.min(end))..end
    }
}

/// The byte offset in `s` of UTF-16 offset `o`, clamped to `s`.
fn utf8_at(s: &str, o: usize) -> usize {
    let mut n = 0;
    for (i, c) in s.char_indices() {
        if n >= o {
            return i;
        }
        n += c.len_utf16();
    }
    s.len()
}

/// Where the word before the end of `t` starts, skipping trailing whitespace of any width.
fn word_start(t: &str) -> usize {
    t.trim_end()
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace())
        .map_or(0, |(i, c)| i + c.len_utf8())
}

/// A one-line field's sideways scroll once its cursor is at `x`: unchanged while `x` shows in the
/// `room` after `scroll`, else just enough to show it; none while all the text (to `end`) fits.
fn follow(scroll: Pixels, x: Pixels, end: Pixels, room: Pixels) -> Pixels {
    if end <= room {
        px(0.)
    } else if x < scroll {
        x
    } else if x > scroll + room {
        x - room
    } else {
        scroll
    }
}

/// The word around byte `o` of `t`, what lies between the whitespace on either side: from a
/// word's first or last letter `word_prev`/`word_next` reached into the next word too.
fn word_at(t: &str, o: usize) -> Range<usize> {
    let start = (t[..o].char_indices().rev())
        .find(|(_, c)| c.is_whitespace())
        .map_or(0, |(i, c)| i + c.len_utf8());
    let end = t[o..].find(char::is_whitespace).map_or(t.len(), |i| o + i);
    start..end
}

/// `i` moved back to a char boundary of `s` and clamped to its length.
fn floor_char(s: &str, i: usize) -> usize {
    (0..=i.min(s.len()))
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0)
}

impl EntityInputHandler for TextInput {
    fn text_for_range(
        &mut self,
        r: Range<usize>,
        actual: &mut Option<Range<usize>>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<String> {
        let r = self.range8(&r);
        actual.replace(self.range16(&r));
        Some(self.text[r].to_owned())
    }
    fn selected_text_range(
        &mut self,
        _: bool,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<UTF16Selection> {
        Some(UTF16Selection {
            range: self.range16(&self.sel),
            reversed: self.reversed,
        })
    }
    fn marked_text_range(&self, _: &mut Window, _: &mut Context<Self>) -> Option<Range<usize>> {
        self.marked.as_ref().map(|r| self.range16(r))
    }
    fn unmark_text(&mut self, _: &mut Window, _: &mut Context<Self>) {
        self.marked = None;
    }
    fn replace_text_in_range(
        &mut self,
        r: Option<Range<usize>>,
        new: &str,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let r = self.target(r);
        self.text.replace_range(r.clone(), new);
        self.sel = r.start + new.len()..r.start + new.len();
        self.reversed = false;
        self.marked = None;
        cx.notify();
    }
    fn replace_and_mark_text_in_range(
        &mut self,
        r: Option<Range<usize>>,
        new: &str,
        new_sel: Option<Range<usize>>,
        _: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let r = self.target(r);
        self.text.replace_range(r.clone(), new);
        self.marked = (!new.is_empty()).then(|| r.start..r.start + new.len());
        // The IME's selection is relative to the composed text, not the field.
        self.sel = new_sel
            .map(|s| r.start + utf8_at(new, s.start)..r.start + utf8_at(new, s.end))
            .unwrap_or(r.start + new.len()..r.start + new.len());
        cx.notify();
    }
    fn bounds_for_range(
        &mut self,
        r: Range<usize>,
        b: Bounds<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<Bounds<Pixels>> {
        let r = self.range8(&r);
        let s = self.position(r.start)?;
        let e = self.position(r.end)?;
        Some(Bounds::from_corners(
            point(b.left() + s.x - self.scroll_x, b.top() + s.y - self.scroll),
            point(
                b.left() + e.x - self.scroll_x,
                b.top() + e.y + self.line_height - self.scroll,
            ),
        ))
    }
    fn character_index_for_point(
        &mut self,
        p: Point<Pixels>,
        _: &mut Window,
        _: &mut Context<Self>,
    ) -> Option<usize> {
        Some(self.to16(self.index_at(p)))
    }
}

struct Field {
    input: Entity<TextInput>,
}

impl IntoElement for Field {
    type Element = Self;
    fn into_element(self) -> Self {
        self
    }
}

struct Shaped {
    lines: Vec<(usize, WrappedLine, Pixels)>,
    under: Vec<PaintQuad>,
    cursor: Option<PaintQuad>,
}

/// The field's font, size and line height, read where the element's text style is in force: the
/// layout's measure callback runs after the parents' styles are popped, so it must not read them.
#[derive(Clone)]
struct Font {
    font: gpui::Font,
    size: Pixels,
    lh: Pixels,
}

impl Font {
    fn of(window: &Window) -> Self {
        let style = window.text_style();
        Font {
            font: style.font(),
            size: style.font_size.to_pixels(window.rem_size()),
            lh: window.line_height(),
        }
    }
}

fn shape(
    window: &mut Window,
    text: &str,
    placeholder: bool,
    ph: &str,
    width: Option<Pixels>,
    th: Theme,
    f: &Font,
) -> Vec<(usize, WrappedLine, Pixels)> {
    let (size, lh) = (f.size, f.lh);
    let (shown, color) = if placeholder {
        (ph, th.ink3)
    } else {
        (text, th.ink)
    };
    let run = TextRun {
        len: shown.len(),
        font: f.font.clone(),
        color: color.into(),
        background_color: None,
        underline: None,
        strikethrough: None,
    };
    let mut out = Vec::new();
    let mut start = 0;
    let mut top = px(0.);
    for para in shown.split('\n') {
        let r = TextRun {
            len: para.len(),
            ..run.clone()
        };
        let shaped = window
            .text_system()
            .shape_text(para.to_owned().into(), size, &[r], width, None)
            .ok()
            .and_then(|mut v| v.pop());
        if let Some(line) = shaped {
            let rows = line.wrap_boundaries().len() + 1;
            out.push((start, line, top));
            top += lh * rows as f32;
        }
        start += para.len() + 1;
    }
    out
}

fn height(lines: &[(usize, WrappedLine, Pixels)], lh: Pixels) -> Pixels {
    lines.last().map_or(lh, |(_, l, top)| {
        *top + lh * (l.wrap_boundaries().len() + 1) as f32
    })
}

impl Element for Field {
    type RequestLayoutState = ();
    type PrepaintState = Shaped;

    fn id(&self) -> Option<ElementId> {
        None
    }
    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&gpui::InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        let mut style = Style::default();
        style.size.width = relative(1.).into();
        let i = self.input.read(cx);
        let (text, ph, multi, max, th) = (
            if i.secret {
                masked(&i.text)
            } else {
                i.text.clone()
            },
            i.placeholder.to_string(),
            i.multiline,
            i.max_rows,
            i.theme,
        );
        if !multi {
            style.size.height = window.line_height().into();
            return (window.request_layout(style, [], cx), ());
        }
        let f = Font::of(window);
        let id = window.request_measured_layout(style, move |known, avail, window, _| {
            let w = known.width.or(match avail.width {
                gpui::AvailableSpace::Definite(x) => Some(x),
                _ => None,
            });
            let lh = f.lh;
            let lines = shape(window, &text, text.is_empty(), &ph, w, th, &f);
            size(
                w.unwrap_or(px(100.)),
                height(&lines, lh).min(lh * max as f32),
            )
        });
        (id, ())
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&gpui::InspectorElementId>,
        b: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) -> Shaped {
        let f = Font::of(window);
        let lh = f.lh;
        let (text, ph, multi, th, sel, cursor, focused) = {
            let i = self.input.read(cx);
            (
                if i.secret {
                    masked(&i.text)
                } else {
                    i.text.clone()
                },
                i.placeholder.to_string(),
                i.multiline,
                i.theme,
                i.sel.clone(),
                i.cursor(),
                i.focus.is_focused(window),
            )
        };
        let lines = shape(
            window,
            &text,
            text.is_empty(),
            &ph,
            multi.then_some(b.size.width),
            th,
            &f,
        );
        self.input.update(cx, |i, _| {
            i.lines = lines.clone();
            i.line_height = lh;
            i.bounds = Some(b);
            // Keep the cursor's row in view when the text is taller than the field.
            if let Some(p) = i.position(cursor) {
                if p.y < i.scroll {
                    i.scroll = p.y;
                } else if p.y + lh > i.scroll + b.size.height {
                    i.scroll = p.y + lh - b.size.height;
                }
            }
            if height(&lines, lh) <= b.size.height {
                i.scroll = px(0.);
            }
            // A one-line field does not wrap: its text moves left so the cursor stays in view (a
            // long command or path hid the caret and what was typed past the edge).
            if !multi {
                let end = i.position(text.len()).map_or(px(0.), |p| p.x);
                let x = i.position(cursor).map_or(i.scroll_x, |p| p.x);
                i.scroll_x = follow(i.scroll_x, x, end, b.size.width - px(2.));
            }
        });
        let i = self.input.read(cx);
        let mut under = Vec::new();
        let mut caret = None;
        let at = |p: Point<Pixels>| point(b.left() + p.x - i.scroll_x, b.top() + p.y - i.scroll);
        if !text.is_empty() && !sel.is_empty() {
            // One band per row the selection crosses.
            let (s, e) = (i.position(sel.start), i.position(sel.end));
            if let (Some(s), Some(e)) = (s, e) {
                let mut y = s.y;
                while y <= e.y {
                    let x0 = if y == s.y { s.x } else { px(0.) };
                    let x1 = if y == e.y { e.x } else { b.size.width };
                    under.push(fill(
                        Bounds::from_corners(at(point(x0, y)), at(point(x1, y + lh))),
                        fade(th.accent, 0.25),
                    ));
                    y += lh;
                }
            }
        } else if focused {
            let p = if text.is_empty() {
                Some(point(px(0.), px(0.)))
            } else {
                i.position(cursor)
            };
            if let Some(p) = p {
                caret = Some(fill(Bounds::new(at(p), size(px(1.5), lh)), th.accent));
            }
        }
        Shaped {
            lines,
            under,
            cursor: caret,
        }
    }

    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&gpui::InspectorElementId>,
        b: Bounds<Pixels>,
        _: &mut (),
        s: &mut Shaped,
        window: &mut Window,
        cx: &mut App,
    ) {
        let focus = self.input.read(cx).focus.clone();
        let (scroll, scroll_x) = (self.input.read(cx).scroll, self.input.read(cx).scroll_x);
        window.handle_input(&focus, ElementInputHandler::new(b, self.input.clone()), cx);
        let lh = window.line_height();
        window.with_content_mask(Some(gpui::ContentMask { bounds: b }), |window| {
            for q in s.under.drain(..) {
                window.paint_quad(q);
            }
            for (_, line, top) in &s.lines {
                line.paint(
                    point(b.left() - scroll_x, b.top() + *top - scroll),
                    lh,
                    TextAlign::Left,
                    None,
                    window,
                    cx,
                )
                .ok();
            }
            if let Some(q) = s.cursor.take() {
                window.paint_quad(q);
            }
        });
    }
}

impl Render for TextInput {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .w_full()
            .key_context(if self.multiline { "Composer" } else { "Field" })
            .track_focus(&self.focus)
            .cursor(CursorStyle::IBeam)
            .on_action(cx.listener(Self::backspace))
            .on_action(cx.listener(Self::delete))
            .on_action(cx.listener(Self::delete_word))
            .on_action(cx.listener(Self::left))
            .on_action(cx.listener(Self::right))
            .on_action(cx.listener(Self::word_left))
            .on_action(cx.listener(Self::word_right))
            .on_action(cx.listener(Self::up))
            .on_action(cx.listener(Self::down))
            .on_action(cx.listener(Self::select_left))
            .on_action(cx.listener(Self::select_right))
            .on_action(cx.listener(Self::select_all))
            .on_action(cx.listener(Self::home))
            .on_action(cx.listener(Self::end))
            .on_action(cx.listener(Self::paste))
            .on_action(cx.listener(Self::cut))
            .on_action(cx.listener(Self::copy))
            .on_action(cx.listener(Self::newline))
            .on_action(cx.listener(Self::submit))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::mouse_down))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::mouse_up))
            .on_mouse_up_out(MouseButton::Left, cx.listener(Self::mouse_up))
            .on_mouse_move(cx.listener(Self::mouse_move))
            .on_action(cx.listener(Self::dismiss))
            .on_mouse_down(MouseButton::Right, cx.listener(Self::right_down))
            .child(Field { input: cx.entity() })
            .when_some(self.menu, |d, at| d.child(self.menu_el(at, cx)))
    }
}

impl Focusable for TextInput {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

/// The field key bindings; `cmd` is ctrl on Linux and Windows.
pub fn bindings() -> Vec<gpui::KeyBinding> {
    use gpui::KeyBinding as K;
    let m = if cfg!(target_os = "macos") {
        "cmd"
    } else {
        "ctrl"
    };
    let w = if cfg!(target_os = "macos") {
        "alt"
    } else {
        "ctrl"
    };
    vec![
        K::new("backspace", Backspace, None),
        K::new(&format!("{w}-backspace"), DeleteWord, None),
        K::new("delete", Delete, None),
        K::new("left", Left, None),
        K::new("right", Right, None),
        K::new(&format!("{w}-left"), WordLeft, None),
        K::new(&format!("{w}-right"), WordRight, None),
        K::new("up", Up, Some("Composer")),
        K::new("down", Down, Some("Composer")),
        K::new("shift-left", SelectLeft, None),
        K::new("shift-right", SelectRight, None),
        K::new(&format!("{m}-a"), SelectAll, None),
        K::new(&format!("{m}-v"), Paste, None),
        K::new(&format!("{m}-c"), Copy, None),
        K::new(&format!("{m}-x"), Cut, None),
        K::new("home", Home, None),
        K::new("end", End, None),
        K::new("enter", Submit, None),
        K::new("shift-enter", Newline, Some("Composer")),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-line field follows its cursor sideways, by no more than it must, and not at all while
    /// its text fits.
    #[test]
    fn a_one_line_field_scrolls_to_keep_its_cursor_in_view() {
        let room = px(100.);
        assert_eq!(follow(px(0.), px(80.), px(90.), room), px(0.));
        assert_eq!(follow(px(0.), px(130.), px(200.), room), px(30.));
        assert_eq!(follow(px(30.), px(60.), px(200.), room), px(30.));
        assert_eq!(follow(px(30.), px(10.), px(200.), room), px(10.));
        // The text deleted back to fit: no scroll is left over.
        assert_eq!(follow(px(30.), px(50.), px(50.), room), px(0.));
    }

    /// A double click on a word's first or last letter selects that word alone.
    #[test]
    fn a_double_click_selects_the_word_under_it() {
        let t = "hello world, é ok";
        assert_eq!(&t[word_at(t, 6)], "world,");
        assert_eq!(&t[word_at(t, 11)], "world,");
        assert_eq!(&t[word_at(t, 0)], "hello");
        assert_eq!(&t[word_at(t, t.len())], "ok");
        assert_eq!(&t[word_at(t, "hello world, ".len())], "é");
    }

    #[test]
    fn word_start_steps_over_multibyte_whitespace() {
        let t = "one\u{3000}two\u{a0}";
        assert_eq!(word_start(t), "one\u{3000}".len());
        assert_eq!(&t[word_start(t)..], "two\u{a0}");
        assert_eq!(word_start("solo"), 0);
    }

    #[test]
    fn utf16_offsets_map_into_the_composed_text() {
        // A composition after the prefix "é " selects its own second char.
        let new = "日本";
        let start = "é ".len();
        assert_eq!(start + utf8_at(new, 1), start + "日".len());
        assert_eq!(utf8_at("a😀b", 3), "a😀".len());
        assert_eq!(utf8_at("ab", 9), 2);
    }

    #[test]
    fn floor_char_lands_on_a_boundary() {
        // A placeholder index past an empty or short text stays inside it.
        assert_eq!(floor_char("", 7), 0);
        assert_eq!(floor_char("é", 1), 0);
        assert_eq!(floor_char("ab", 1), 1);
    }
}
