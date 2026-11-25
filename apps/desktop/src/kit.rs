//! The pieces every window is made of: the design unit, labels, cards, serif and mono text,
//! icons, buttons, pills, tags, bars and setting rows. The app reaches them through ui.rs; the
//! installer window (apps/setup) includes this file too, so it depends on gpui and theme.rs alone.

use std::sync::Arc;

use gpui::{
    App, ClickEvent, Div, FontWeight, Hsla, InteractiveElement, IntoElement, ParentElement, Pixels,
    Rems, SharedString, Stateful, StatefulInteractiveElement, Styled, Window, div, font,
    prelude::FluentBuilder, relative, rems, svg,
};

use crate::theme::{Theme, fade};

/// A size in the design's pixels at text size 14; the whole window scales with Text size.
pub fn u(v: f32) -> Rems {
    rems(v / 14.)
}

/// A label in the design's small caps.
pub fn eyebrow(th: &Theme, t: impl Into<SharedString>) -> Div {
    eyebrow_box(th).child(t.into().to_uppercase())
}

/// The box an eyebrow's capitals sit in.
pub fn eyebrow_box(th: &Theme) -> Div {
    div()
        .text_size(u(11.))
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(th.eyebrow)
}

/// A card.
pub fn card(th: &Theme) -> Div {
    div()
        .bg(th.card)
        .border_1()
        .border_color(th.line)
        .rounded(u(14.))
        .p(u(20.))
        .flex()
        .flex_col()
        .gap(u(10.))
}

/// A serif title.
pub fn serif(t: impl Into<SharedString>, size: f32) -> Div {
    serif_box(size).child(t.into())
}

/// The box a serif title of `size` sits in.
pub fn serif_box(size: f32) -> Div {
    div()
        .font(serif_font())
        .text_size(u(size))
        .line_height(u(size * 1.15))
}

/// Fraunces with lining figures, so numbers in titles sit on the line.
pub fn serif_font() -> gpui::Font {
    gpui::Font {
        features: gpui::FontFeatures(Arc::new(vec![("lnum".into(), 1)])),
        ..font("Fraunces")
    }
}

/// Monospace text.
pub fn mono(t: impl Into<SharedString>) -> Div {
    mono_box().child(t.into())
}

/// The box monospace text sits in.
pub fn mono_box() -> Div {
    div().font_family("JetBrains Mono").text_size(u(12.))
}

/// An icon from assets/icons, tinted.
pub fn icon(name: &str, size: f32, c: impl Into<Hsla>) -> gpui::Svg {
    svg()
        .path(SharedString::from(format!("icons/{name}.svg")))
        .size(u(size))
        .flex_none()
        .text_color(c.into())
}

/// Text centred in a small shape, such as a count in a badge. On Windows and Linux gpui floors
/// each glyph's baseline to a whole device pixel, so a centred line draws up to a pixel high;
/// moving it down by `half` (half a device pixel) makes that floor a round.
pub fn mid(half: Pixels, t: impl IntoElement) -> Div {
    div().relative().top(half).child(t)
}

/// At most `n` lines `lh` tall, cut after a whole line. gpui's `line_clamp` puts all the rest
/// on the last line, which then runs past its box, and its ellipsis lands past the edge.
pub fn lines(n: f32, lh: f32) -> Div {
    div().line_height(u(lh)).max_h(u(n * lh)).overflow_hidden()
}

/// A print icon: the line drawing over its colour block, set off down and right like a
/// two-colour print. The block is gold on the current page.
pub fn picon(th: &Theme, name: &str, size: f32, on: bool) -> Div {
    print(name, size, th.block(on), if on { th.ink } else { th.ink2 })
}

/// A print icon with its block in `block` and its line in `line`.
pub fn print(name: &str, size: f32, block: impl Into<Hsla>, line: impl Into<Hsla>) -> Div {
    let layer = |path: String, c: Hsla| {
        svg()
            .path(SharedString::from(path))
            .absolute()
            .top_0()
            .left_0()
            .size_full()
            .text_color(c)
    };
    // The files are drawn on 24 units in a box of 26, so the block's offset fits.
    div()
        .relative()
        .flex_none()
        .size(u(size * 26. / 24.))
        .child(layer(format!("icons/{name}-block.svg"), block.into()))
        .child(layer(format!("icons/{name}.svg"), line.into()))
}

/// A page's name in the top bar: its print icon, current, and the title in the serif.
pub fn head(th: &Theme, name: &str, title: &str) -> Div {
    div()
        .flex()
        .flex_none()
        .items_center()
        .gap(u(10.))
        .child(picon(th, name, 22., true))
        .child(serif(title.to_owned(), 22.).text_color(th.ink))
}

/// How a button looks.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Btn {
    /// Filled forest green: the one thing to do.
    Primary,
    /// Outlined in green.
    Outline,
    /// Quiet: Not now, Cancel.
    Ghost,
    /// On the dark popover.
    Light,
    /// Red: Delete, armed.
    Danger,
}

/// A button; `on` runs on click.
pub fn button(
    th: &Theme,
    id: impl Into<SharedString>,
    label: impl Into<SharedString>,
    kind: Btn,
    on: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> Stateful<Div> {
    let (bg, fg, border): (Hsla, Hsla, Hsla) = match kind {
        Btn::Primary => (th.accent.into(), th.on_accent.into(), th.accent.into()),
        Btn::Outline => (
            gpui::transparent_black(),
            th.accent.into(),
            th.accent.into(),
        ),
        Btn::Ghost => (th.card.into(), th.ink.into(), th.line.into()),
        Btn::Light => (gpui::white(), gpui::black(), gpui::white()),
        Btn::Danger => (th.lose.into(), gpui::white(), th.lose.into()),
    };
    let hover = if kind == Btn::Ghost || kind == Btn::Outline {
        fade(th.accent, 0.08)
    } else {
        bg.opacity(0.88)
    };
    div()
        .id(gpui::ElementId::Name(id.into()))
        .flex()
        .flex_none()
        .items_center()
        .gap(u(6.))
        .px(u(14.))
        .py(u(7.))
        .rounded(u(9.))
        .border_1()
        .border_color(border)
        .bg(bg)
        .text_color(fg)
        .text_size(u(13.5))
        .font_weight(if kind == Btn::Primary {
            FontWeight::SEMIBOLD
        } else {
            FontWeight::NORMAL
        })
        .cursor_pointer()
        .hover(move |s| s.bg(hover))
        .on_click(on)
        .child(label.into())
}

/// A rounded pill: a filter, a tag, a composer chip.
pub fn pill(
    th: &Theme,
    id: impl Into<SharedString>,
    label: impl Into<SharedString>,
    on: bool,
    click: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> Stateful<Div> {
    let hover = fade(th.accent, 0.08);
    div()
        .id(gpui::ElementId::Name(id.into()))
        .flex()
        .flex_none()
        .items_center()
        .gap(u(5.))
        .px(u(13.))
        .py(u(5.))
        .rounded_full()
        .border_1()
        .border_color(if on { th.accent } else { th.line })
        .bg(if on {
            fade(th.accent, 0.12)
        } else {
            th.card.into()
        })
        .text_color(if on { th.accent } else { th.ink2 })
        .text_size(u(13.))
        .cursor_pointer()
        .hover(move |s| s.bg(hover))
        .on_click(click)
        .child(label.into())
}

/// A small static tag (`35B`, `Q4_K_M`).
pub fn tag(th: &Theme, t: impl Into<SharedString>) -> Div {
    div()
        .px(u(9.))
        .py(u(3.))
        .rounded_full()
        .bg(th.sunk)
        .text_color(th.ink2)
        .font_family("JetBrains Mono")
        .text_size(u(11.5))
        .child(t.into())
}

/// A segmented control: one of `opts`, `on` picked; `pick(i)` runs on a click.
pub fn seg(
    th: &Theme,
    id: &str,
    opts: &[&str],
    on: usize,
    pick: impl Fn(usize, &mut Window, &mut App) + Clone + 'static,
) -> Div {
    let th = *th;
    let last = opts.len().saturating_sub(1);
    div()
        .flex()
        .flex_none()
        .border_1()
        .border_color(th.line)
        .rounded(u(9.))
        .bg(th.card)
        .children(opts.iter().enumerate().map(|(i, o)| {
            let pick = pick.clone();
            // overflow_hidden clips to a rectangle, so the end segments carry the corners themselves.
            div()
                .id(gpui::ElementId::Name(format!("{id}-{i}").into()))
                .when(i == 0, |d| d.rounded_l(u(8.)))
                .when(i == last, |d| d.rounded_r(u(8.)))
                .px(u(14.))
                .py(u(6.))
                .text_size(u(13.))
                .cursor_pointer()
                .when(i == on, |d| {
                    d.bg(th.accent)
                        .text_color(th.on_accent)
                        .font_weight(FontWeight::SEMIBOLD)
                })
                .when(i != on, |d| d.text_color(th.ink2).hover(|s| s.bg(th.sunk)))
                .on_click(move |_, w, cx| pick(i, w, cx))
                .child(o.to_string())
        }))
}

/// A bar of coloured parts on a track, each `(fraction, colour)`.
pub fn bar(th: &Theme, parts: &[(f32, Hsla)], h: f32) -> Div {
    div()
        .h(u(h))
        .w_full()
        .rounded_full()
        .bg(th.sunk)
        .flex()
        .children(parts.iter().enumerate().map(|(i, (f, c))| {
            // Rounded by hand: overflow_hidden does not clip to the track's round ends.
            let full = parts.iter().map(|p| p.0).sum::<f32>() >= 0.999;
            div()
                .h_full()
                .w(relative(f.clamp(0., 1.)))
                .bg(*c)
                .when(i == 0, |d| d.rounded_l_full())
                .when(full && i + 1 == parts.len(), |d| d.rounded_r_full())
        }))
}

/// A row with a label on the left, as Settings lays them out.
pub fn setrow(th: &Theme, label: &str, ctl: impl IntoElement) -> Div {
    div()
        .flex()
        .flex_wrap()
        .items_center()
        .gap_x(u(24.))
        .gap_y(u(10.))
        .py(u(14.))
        // Below, not above: the first row's top line doubled the tab bar's.
        .border_b_1()
        .border_color(th.line)
        .child(
            div()
                .w(u(260.))
                .flex_none()
                .text_size(u(14.))
                .font_weight(FontWeight::SEMIBOLD)
                .child(label.to_owned()),
        )
        // Held to the row's width, so a sentence or a path wraps rather than running off the
        // page; a control keeps its own width. Narrower than the widest control (the name
        // field's 360), it goes under the label.
        .child(
            div()
                .flex_grow()
                .flex_basis(u(360.))
                .min_w_0()
                .flex()
                .flex_col()
                .items_start()
                .child(ctl),
        )
}
