//! The first-launch welcome and the tour. The welcome asks what Larch should call the user; the
//! tour then opens each place on the real page, rings what it describes and explains it in a card
//! beside it. It shows once: `Prefs::welcomed` keeps the tour version the user finished or skipped,
//! and Settings › General replays it.

use gpui::{
    AnyElement, App, BoxShadow, Context, Corner, FontWeight, InteractiveElement, IntoElement,
    KeyBinding, MouseButton, ParentElement, StatefulInteractiveElement, Styled, actions, anchored,
    deferred, div, point, prelude::FluentBuilder, px, relative, rgb,
};

use crate::theme::{NOTE, NOTE_EYEBROW, NOTE_INK, Theme, fade};
use crate::tree::{Mood, Tree, tree};
use crate::ui::{Btn, Ecl, MTab, Page, STab, button, on_note, serif, shadow, u};

/// The tour's version. Raise it when the tour changes enough to show again to everyone who saw it.
pub const TOUR: u32 = 1;

/// The longest name kept, in characters.
const NAME_MAX: usize = 64;

actions!(tour, [Next, Back]);

/// The card's keys, while it has the focus: Enter or → for Next, ← for Back. Esc skips the tour
/// through the window's Dismiss.
pub fn bindings() -> Vec<KeyBinding> {
    vec![
        KeyBinding::new("enter", Next, Some("Tour")),
        KeyBinding::new("right", Next, Some("Tour")),
        KeyBinding::new("left", Back, Some("Tour")),
    ]
}

/// What a stop rings.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Spot {
    /// The sidebar's New chat button.
    NewChat,
    /// The message box on Home and in a chat.
    Composer,
    /// A page's row in the sidebar.
    Nav(Page),
    /// The Your name row in Settings › General.
    Name,
    /// Discover's "Recommended for this computer" card.
    Recommended,
}

/// Where a stop's card sits against the spot it hangs on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Place {
    /// To the right, level with the spot's top.
    Right,
    /// To the right, opening upward from the spot's bottom: for a spot low in the window.
    RightUp,
    /// Under the spot: for one as wide as the page.
    Below,
}

/// A stop: the page it opens, what it rings (the card sits beside the first) and what it says.
pub struct Stop {
    /// The page the stop opens (with the tab it describes in front).
    pub page: Page,
    /// What the stop rings. Its card hangs on the first of these on screen (`anchor`). A page's own
    /// element can go (Discover's Recommended card hides while the search box has text), so each
    /// stop also rings a sidebar spot, which is always drawn: the card, and the focus its keys
    /// need, then move there instead of going with the element.
    pub spots: &'static [Spot],
    /// The card's title.
    pub title: &'static str,
    /// The card's text: one or two sentences on what the ringed place does.
    pub body: &'static str,
}

/// The tour, in the order a new user needs it: a model first, since nothing runs without one.
pub const STOPS: &[Stop] = &[
    Stop {
        page: Page::Models,
        spots: &[Spot::Nav(Page::Models)],
        title: "Start with a model",
        body: "Larch runs the models you download. Discover searches Hugging Face for them; what \
               you download shows in Library.",
    },
    Stop {
        page: Page::Models,
        spots: &[Spot::Recommended, Spot::Nav(Page::Models)],
        title: "Picked for this computer",
        body: "Larch reads how much memory your GPU has and recommends the models that suit it, \
               largest first. Show finds one in the table, which checks it fits before you \
               download it.",
    },
    Stop {
        page: Page::Home,
        spots: &[Spot::NewChat, Spot::Composer],
        title: "Ask anything",
        body: "Use + Attach to add files or a whole folder. The chips under the box, like \
               Research, set how Larch works on your request.",
    },
    Stop {
        page: Page::Chats,
        spots: &[Spot::Nav(Page::Chats)],
        title: "Every chat is kept",
        body: "Pin the chats you use, rename or archive the rest. The latest also sit under \
               Recent.",
    },
    Stop {
        page: Page::Tasks,
        spots: &[Spot::Nav(Page::Tasks)],
        title: "Hand off longer work",
        body: "A task works through its steps without you, once or on a schedule, and stops to \
               ask when it needs your approval.",
    },
    Stop {
        page: Page::Mcp,
        spots: &[Spot::Nav(Page::Mcp)],
        title: "Give it more tools",
        body: "Connect MCP servers and their tools join every new chat. You choose which ones \
               tasks may use.",
    },
    Stop {
        page: Page::Settings,
        spots: &[Spot::Nav(Page::Settings), Spot::Name],
        title: "Make it yours",
        body: "Your name, theme and text size live here. Advanced adds the engine's numbers and \
               a chat's side panel. You can take this tour again from General.",
    },
];

/// Where the tour is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum At {
    Welcome,
    /// An index into `STOPS`.
    Stop(usize),
}

/// A typed name as kept: no control characters, no surrounding space, at most `NAME_MAX`
/// characters. Empty leaves the greeting without a name.
pub fn clean_name(typed: &str) -> String {
    let name: String = typed.chars().filter(|c| !c.is_control()).collect();
    let name: String = name.trim().chars().take(NAME_MAX).collect();
    name.trim_end().to_owned()
}

/// Opens the welcome, on the first launch.
pub fn welcome(s: &mut Ecl) {
    s.tour = Some(At::Welcome);
    s.tour_refocus = true;
}

/// Starts the tour at its first stop, as Settings' Take the tour does.
pub fn start(s: &mut Ecl, cx: &mut Context<Ecl>) {
    show(s, 0, cx);
}

/// Opens stop `i` on its page, the tab it describes in front.
fn show(s: &mut Ecl, i: usize, cx: &mut Context<Ecl>) {
    let stop = &STOPS[i];
    match stop.page {
        Page::Models => s.mtab = MTab::Discover,
        Page::Settings => s.stab = STab::General,
        _ => {}
    }
    s.tour = Some(At::Stop(i));
    s.tour_refocus = true;
    s.go(stop.page, cx);
}

/// Show me around, Next or Finish.
pub fn next(s: &mut Ecl, cx: &mut Context<Ecl>) {
    match s.tour {
        Some(At::Welcome) => {
            answer_welcome(s, cx);
            start(s, cx);
        }
        Some(At::Stop(i)) if i + 1 < STOPS.len() => show(s, i + 1, cx),
        Some(At::Stop(_)) => end(s, true, cx),
        None => {}
    }
}

/// Back to the stop before.
pub fn back(s: &mut Ecl, cx: &mut Context<Ecl>) {
    if let Some(At::Stop(i)) = s.tour
        && i > 0
    {
        show(s, i - 1, cx);
    }
}

/// Closes the tour, as seen. `done` (Finish) goes Home and says how to replay it; a skip leaves
/// the user where they are. A skip from the welcome keeps the name typed so far.
pub fn end(s: &mut Ecl, done: bool, cx: &mut Context<Ecl>) {
    if s.tour.take() == Some(At::Welcome) {
        answer_welcome(s, cx);
    }
    if done {
        s.go(Page::Home, cx);
        s.say("You're all set. Replay the tour from Settings › General.");
    }
    cx.notify();
}

/// Keeps the welcome's name and marks the tour seen, so quitting halfway does not bring the
/// welcome back. Saved at once: Settings' field shows the same name.
fn answer_welcome(s: &mut Ecl, cx: &mut Context<Ecl>) {
    let name = clean_name(&s.name_in.read(cx).text);
    s.name_in.update(cx, |i, cx| i.set_text(&name, cx));
    s.prefs.name = name;
    s.prefs.welcomed = TOUR;
    s.save_prefs();
}

/// The welcome: the window dimmed under a card that asks the user's name. Answered with Show me
/// around (or Enter in the field), Skip the tour or Esc.
pub fn welcome_el(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Option<impl IntoElement> {
    (s.tour == Some(At::Welcome)).then(|| {
        div()
            .id("welcome-back")
            .absolute()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(fade(th.ink, 0.35))
            .occlude()
            .child(
                div()
                    .id("welcome")
                    .w(u(460.))
                    .max_w_full()
                    .flex()
                    .flex_col()
                    .px(u(34.))
                    .pt(u(30.))
                    .pb(u(28.))
                    .rounded(u(16.))
                    .bg(th.card)
                    .border_1()
                    .border_color(th.line)
                    .shadow(shadow(th))
                    .occlude()
                    .child(div().flex().child(tree(
                        Tree::new(Mood::Ready),
                        48.,
                        225.,
                        true,
                        *th,
                        th.card,
                    )))
                    .child(serif("Welcome to Larch.", 30.).mt(u(14.)))
                    .child(
                        div()
                            .mt(u(6.))
                            .text_size(u(14.5))
                            .text_color(th.ink2)
                            .child("Larch runs AI models right here on your computer."),
                    )
                    .child(
                        div()
                            .mt(u(22.))
                            .mb(u(8.))
                            .font_weight(FontWeight::SEMIBOLD)
                            .child("What should Larch call you?"),
                    )
                    .child(
                        div()
                            .flex()
                            .child(crate::pages::field(th, s.name_in.clone())),
                    )
                    .child(
                        div()
                            .mt(u(8.))
                            .text_size(u(12.5))
                            .text_color(th.ink3)
                            .child("You can change it any time in Settings."),
                    )
                    .child(
                        div()
                            .mt(u(26.))
                            .flex()
                            .justify_end()
                            .gap(u(10.))
                            .child(button(
                                th,
                                "welcome-skip",
                                "Skip the tour",
                                Btn::Ghost,
                                cx.listener(|s, _, _, cx| end(s, false, cx)),
                            ))
                            .child(button(
                                th,
                                "welcome-go",
                                "Show me around",
                                Btn::Primary,
                                cx.listener(|s, _, _, cx| next(s, cx)),
                            )),
                    ),
            )
    })
}

/// Whether `spot` is on screen now: the sidebar's always, a page's only on its page.
fn drawn(s: &Ecl, spot: Spot, cx: &App) -> bool {
    match spot {
        Spot::NewChat | Spot::Nav(_) => true,
        Spot::Composer => matches!(s.page, Page::Home | Page::Chat),
        Spot::Name => s.page == Page::Settings && s.stab == STab::General,
        Spot::Recommended => crate::pages::shows_recommended(s, cx),
    }
}

/// The spot `stop`'s card hangs on: the first of its spots on screen.
fn anchor(s: &Ecl, stop: &Stop, cx: &App) -> Option<Spot> {
    stop.spots.iter().copied().find(|&spot| drawn(s, spot, cx))
}

/// Where the card sits against `spot`: under a page-wide card, upward from the sidebar's foot,
/// otherwise to the right.
fn place(spot: Spot) -> Place {
    match spot {
        Spot::Recommended => Place::Below,
        Spot::Nav(Page::Settings) => Place::RightUp,
        _ => Place::Right,
    }
}

/// What `spot` carries on this frame: its ring when the current stop rings it, and the stop's
/// card when the card hangs on it (`anchor`). Both sit outside the layout, so the spot keeps its
/// place; the element that carries them must be `relative`.
pub fn marks(s: &Ecl, spot: Spot, th: &Theme, cx: &mut Context<Ecl>) -> Vec<AnyElement> {
    let Some(At::Stop(i)) = s.tour else {
        return Vec::new();
    };
    let stop = &STOPS[i];
    if !stop.spots.contains(&spot) {
        return Vec::new();
    }
    let ring = div()
        .absolute()
        .top(u(-4.))
        .left(u(-4.))
        .right(u(-4.))
        .bottom(u(-4.))
        .rounded(u(12.))
        .border_2()
        .border_color(th.accent)
        .shadow(vec![BoxShadow {
            color: fade(th.accent, 0.18),
            offset: point(px(0.), px(0.)),
            blur_radius: px(0.),
            spread_radius: px(4.),
        }]);
    let mut out = vec![ring.into_any_element()];
    if anchor(s, stop, cx) == Some(spot) {
        // A zero-size holder beside the spot (right of it, or under it for a page-wide one);
        // the card is anchored there and kept inside the window.
        let (holder, corner) = match place(spot) {
            Place::Right => (
                div().absolute().left_full().ml(u(18.)).top(u(-4.)),
                Corner::TopLeft,
            ),
            Place::RightUp => (
                div().absolute().left_full().ml(u(18.)).bottom(u(-4.)),
                Corner::BottomLeft,
            ),
            Place::Below => (
                div().absolute().top_full().mt(u(18.)).left(u(16.)),
                Corner::TopLeft,
            ),
        };
        out.push(
            holder
                .child(
                    deferred(
                        anchored()
                            .anchor(corner)
                            .snap_to_window_with_margin(px(8.))
                            .child(card(s, i, th, cx)),
                    )
                    .with_priority(2),
                )
                .into_any_element(),
        );
    }
    out
}

/// Stop `i`'s card: where it is in the tour, what the ringed place does, and the way on.
fn card(s: &Ecl, i: usize, th: &Theme, cx: &mut Context<Ecl>) -> impl IntoElement {
    let stop = &STOPS[i];
    let ink = rgb(NOTE_INK);
    let last = i + 1 == STOPS.len();
    div()
        .id("tour-card")
        .track_focus(&s.tour_focus)
        // A click elsewhere takes the keys, as with any popover; a click on the card gives them
        // back.
        .on_mouse_down(
            MouseButton::Left,
            cx.listener(|s, _, w, _| w.focus(&s.tour_focus)),
        )
        .key_context("Tour")
        .on_action(cx.listener(|s, _: &Next, _, cx| next(s, cx)))
        .on_action(cx.listener(|s, _: &Back, _, cx| back(s, cx)))
        .w(u(330.))
        // The card is drawn from inside its spot, whose text style (a current page's bold) it
        // would otherwise take.
        .font_weight(FontWeight::NORMAL)
        .px(u(20.))
        .pt(u(18.))
        .pb(u(16.))
        .rounded(u(14.))
        .bg(rgb(NOTE))
        .text_color(ink)
        .shadow(shadow(th))
        .occlude()
        .child(
            div()
                .text_size(u(11.))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(rgb(NOTE_EYEBROW))
                .child(format!("{} OF {}", i + 1, STOPS.len())),
        )
        .child(serif(stop.title, 21.).mt(u(6.)))
        .child(
            div()
                .mt(u(6.))
                .text_size(u(13.5))
                .line_height(relative(1.5))
                .text_color(fade(ink, 0.86))
                .child(stop.body),
        )
        .child(
            div()
                .mt(u(16.))
                .flex()
                .items_center()
                .gap(u(8.))
                .child(
                    div()
                        .id("tour-skip")
                        .flex_1()
                        .text_size(u(12.5))
                        .text_color(fade(ink, 0.6))
                        .cursor_pointer()
                        .hover(move |d| d.text_color(ink))
                        .on_click(cx.listener(|s, _, _, cx| end(s, false, cx)))
                        .child("Skip tour"),
                )
                .child(div().flex().items_center().gap(u(5.)).mr(u(6.)).children(
                    (0..STOPS.len()).map(|j| {
                        div()
                            .h(u(6.))
                            .w(u(if j == i { 16. } else { 6. }))
                            .rounded_full()
                            .bg(if j == i {
                                rgb(NOTE_EYEBROW).into()
                            } else {
                                fade(ink, 0.25)
                            })
                    }),
                ))
                .when(i > 0, |d| {
                    d.child(on_note(button(
                        th,
                        "tour-back",
                        "Back",
                        Btn::Ghost,
                        cx.listener(|s, _, _, cx| back(s, cx)),
                    )))
                })
                .child(button(
                    th,
                    "tour-next",
                    if last { "Finish" } else { "Next" },
                    Btn::Light,
                    cx.listener(|s, _, _, cx| next(s, cx)),
                )),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A name keeps its letters and inner spaces; control characters, outer space and anything past
    /// the limit go.
    #[test]
    fn a_typed_name_is_cleaned() {
        assert_eq!(clean_name("  Alex  "), "Alex");
        assert_eq!(clean_name("Ana\tMaría\n"), "AnaMaría");
        assert_eq!(clean_name("Mary Jane"), "Mary Jane");
        assert_eq!(clean_name(" \u{7} "), "");
        let long = "é".repeat(NAME_MAX + 10);
        assert_eq!(clean_name(&long).chars().count(), NAME_MAX);
        // Cut at the limit, a space left at the end goes too.
        let spaced = format!("{} b", "a".repeat(NAME_MAX - 1));
        assert_eq!(clean_name(&spaced), "a".repeat(NAME_MAX - 1));
    }

    /// Every stop opens a page the sidebar reaches, rings something and says something, and rings a
    /// sidebar spot, always drawn, for its card to fall back on when a page's element goes.
    #[test]
    fn every_stop_is_reachable_and_says_something() {
        let places = [
            Page::Home,
            Page::Chats,
            Page::Tasks,
            Page::Models,
            Page::Mcp,
            Page::Settings,
        ];
        assert!(!STOPS.is_empty());
        for stop in STOPS {
            assert!(places.contains(&stop.page), "{:?}", stop.page);
            assert!(!stop.spots.is_empty() && !stop.title.is_empty() && !stop.body.is_empty());
            assert!(
                (stop.spots.iter()).any(|spot| matches!(spot, Spot::Nav(_) | Spot::NewChat)),
                "the card of {} has no sidebar spot to fall back on",
                stop.title
            );
            assert!(
                !stop.body.contains("  "),
                "a wrapped line kept its indent: {}",
                stop.title
            );
        }
    }
}
