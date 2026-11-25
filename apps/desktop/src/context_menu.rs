//! The right-click menu: a small card of rows at the pointer, as the chat row's "..." menu looks.
//! A text field's (Cut, Copy, Paste, Select All) lives in input.rs; the chat's (Copy, Copy
//! message) in ui.rs. Each caller adds its own click and close handlers.

use gpui::{
    Deferred, Div, ElementId, Pixels, Point, Stateful, anchored, deferred, div, prelude::*, px,
};

use crate::theme::Theme;
use crate::ui::u;

/// The menu's card, drawn over everything at window point `at` and kept inside the window.
pub fn at(at: Point<Pixels>, card: Div) -> Deferred {
    deferred(
        anchored()
            .position(at)
            .snap_to_window_with_margin(px(8.))
            .child(card),
    )
    .with_priority(1)
}

/// An empty card; the caller adds `row`s and closes it on `on_mouse_down_out`.
pub fn card(th: &Theme) -> Div {
    div()
        .occlude()
        .min_w(u(160.))
        .p(u(5.))
        .flex()
        .flex_col()
        .rounded(u(10.))
        .bg(th.card)
        .border_1()
        .border_color(th.line)
        .shadow_lg()
}

/// One row named `label`; a disabled one is greyed and the caller gives it no click.
pub fn row(th: &Theme, id: &'static str, label: &str, enabled: bool) -> Stateful<Div> {
    let th2 = *th;
    div()
        .id(ElementId::Name(id.into()))
        .px(u(10.))
        .py(u(6.))
        .rounded(u(7.))
        .text_size(u(13.5))
        .text_color(if enabled { th.ink } else { th.ink3 })
        .when(enabled, |d| {
            d.cursor_pointer().hover(move |s| s.bg(th2.sunk))
        })
        .child(label.to_owned())
}
