//! Models (Library, Discover, This computer), Tools (MCP, Connect apps) and Settings (with Memory).

use gpui::{
    AnyElement, App, AppContext, Bounds, Context, Div, FontWeight, InteractiveElement, IntoElement,
    MouseMoveEvent, ParentElement, PathBuilder, PathPromptOptions, Pixels, SharedString, Stateful,
    StatefulInteractiveElement, Styled, Window, canvas, div, fill, linear_color_stop,
    linear_gradient, point, prelude::FluentBuilder, px, size,
};
use larch_app::contract::*;

use crate::logic;
use crate::prefs::{Mode, Motion};
use crate::select::PAGE;
use crate::theme::{Dark, Light, Theme, fade};
use crate::tree::{Mood, Tree, tree};
use crate::ui::*;

/// A setter for a segmented control or a pill: runs `f` on the window, saves the settings, redraws.
fn set(
    cx: &Context<Ecl>,
    f: impl Fn(&mut Ecl, usize, &mut Context<Ecl>) + Clone + 'static,
) -> impl Fn(usize, &mut Window, &mut App) + Clone + 'static {
    let e = cx.entity().downgrade();
    move |i, _, cx| {
        let f = f.clone();
        e.update(cx, |s, cx| {
            f(s, i, cx);
            s.save_prefs();
            cx.notify()
        })
        .ok();
    }
}

fn click(
    cx: &Context<Ecl>,
    f: impl Fn(&mut Ecl, &mut Context<Ecl>) + 'static,
) -> impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static {
    cx.listener(move |s, _, _, cx| {
        f(s, cx);
        cx.notify()
    })
}

/// A page's side margin, in `u`.
const PAGE_X: f32 = 40.;

fn page(d: impl IntoElement, id: &'static str) -> Stateful<Div> {
    div().id(id).size_full().overflow_y_scroll().child(
        div()
            .flex()
            .flex_col()
            .gap(u(20.))
            .px(u(PAGE_X))
            .pt(u(32.))
            .pb(u(48.))
            .child(d),
    )
}

/// A text field's frame: the theme's border, card fill and padding around the input `f`.
pub fn field(th: &Theme, f: impl IntoElement) -> Div {
    div()
        .flex_1()
        .min_w_0()
        .px(u(14.))
        .py(u(9.))
        .rounded(u(10.))
        .border_1()
        .border_color(th.line)
        .bg(th.card)
        .text_size(u(14.))
        .child(f)
}

fn tab(
    th: &Theme,
    id: String,
    label: impl Into<SharedString>,
    on: bool,
    h: f32,
    f: impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static,
) -> Stateful<Div> {
    div()
        .id(gpui::ElementId::Name(id.into()))
        .h(u(h))
        .flex()
        .items_center()
        .gap(u(6.))
        .px(u(20.))
        .cursor_pointer()
        .border_b_2()
        .border_color(if on {
            th.accent.into()
        } else {
            gpui::transparent_black()
        })
        .text_size(u(15.))
        .font_weight(if on {
            FontWeight::SEMIBOLD
        } else {
            FontWeight::NORMAL
        })
        .text_color(if on { th.ink } else { th.ink2 })
        .on_click(f)
        .child(label.into())
}

/// The top bar's title on Models: the page and its two tabs.
pub fn models_tabs(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let count = div()
        .text_size(u(12.))
        .text_color(th.ink3)
        .font_weight(FontWeight::NORMAL)
        .child(s.models.len().to_string());
    let on = |t: MTab| s.page == Page::Models && s.mtab == t;
    div()
        .flex()
        .items_center()
        .gap(u(4.))
        .child(head(th, "models", "Models").mr(u(24.)))
        .child(
            tab(
                th,
                "t-lib".into(),
                "Library",
                on(MTab::Library),
                56.,
                click(cx, |s, cx| {
                    s.mtab = MTab::Library;
                    s.go(Page::Models, cx)
                }),
            )
            .child(count),
        )
        .child(tab(
            th,
            "t-disc".into(),
            "Discover",
            on(MTab::Discover),
            56.,
            click(cx, |s, cx| {
                s.mtab = MTab::Discover;
                s.go(Page::Models, cx)
            }),
        ))
        // The engine's own page, for those who asked for the details.
        .when(s.prefs.advanced, |d| {
            d.child(tab(
                th,
                "t-engine".into(),
                "This computer",
                s.page == Page::Engine,
                56.,
                click(cx, |s, cx| s.go(Page::Engine, cx)),
            ))
        })
}

/// The top bar on Tools: its name and the MCP and Connect apps tabs.
pub fn tools_tabs(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    div()
        .flex()
        .items_center()
        .gap(u(4.))
        .child(head(th, "tools", "Tools").mr(u(24.)))
        .child(tab(
            th,
            "t-mcp".into(),
            "MCP tools",
            s.page == Page::Mcp,
            56.,
            click(cx, |s, cx| s.go(Page::Mcp, cx)),
        ))
        .child(tab(
            th,
            "t-connect".into(),
            "Connect apps",
            s.page == Page::Connect,
            56.,
            click(cx, |s, cx| s.go(Page::Connect, cx)),
        ))
}

/// Models: the libraries and their models, or Hugging Face.
pub fn models(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Stateful<Div> {
    match s.mtab {
        MTab::Library => page(library(s, th, cx), "models"),
        MTab::Discover => page(discover(s, th, cx), "discover"),
    }
}

fn home_short(p: &str) -> String {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"));
    under_home(p, &home.unwrap_or_default())
}

/// `p` with `home` written `~`, by whole folders: `C:\Users\alex2\x` is not under `C:\Users\alex`
/// (a string prefix showed it as `~2\x`).
fn under_home(p: &str, home: &str) -> String {
    let rest = (!home.is_empty())
        .then(|| std::path::Path::new(p).strip_prefix(home).ok())
        .flatten();
    match rest {
        Some(r) if r.as_os_str().is_empty() => "~".into(),
        Some(r) => format!("~{}{}", std::path::MAIN_SEPARATOR, r.display()),
        None => p.to_owned(),
    }
}

fn add_library(s: &mut Ecl, cx: &mut Context<Ecl>) {
    let o = PathPromptOptions {
        files: false,
        directories: true,
        multiple: false,
        prompt: Some("Add".into()),
    };
    s.pick(
        o,
        |s, p, cx| {
            let Some(dir) = p.and_then(|mut p| p.pop()) else {
                return;
            };
            let dir = dir.display().to_string();
            s.call(
                cx,
                move |c| c.add_library(&dir),
                |s, libs, cx| {
                    s.st.libraries = libs;
                    s.refresh(cx)
                },
            )
        },
        cx,
    );
}

fn library(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let th2 = *th;
    let mut libs: Vec<String> = s.st.libraries.clone();
    for m in &s.models {
        if !libs.contains(&m.library) {
            libs.push(m.library.clone());
        }
    }
    let lib_cards = libs
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let ms: Vec<&ModelEntry> = s.models.iter().filter(|m| &m.library == l).collect();
            let bytes: u64 = ms.iter().map(|m| m.size_bytes).sum();
            let source = ms.first().map_or("Larch", |m| m.source.as_str()).to_owned();
            let removable = i > 0 && s.st.libraries.contains(l);
            let lib = l.clone();
            card(th)
                .flex_1()
                .min_w(u(250.))
                .gap(u(8.))
                .child(
                    div()
                        .flex()
                        .items_center()
                        // A Remove button's height, so cards with and without one line up.
                        .min_h(u(37.))
                        .gap(u(8.))
                        .child(icon("folder", 16., th.ink2))
                        .child(
                            mono(home_short(l))
                                .flex_1()
                                .min_w_0()
                                .truncate()
                                .text_size(u(13.5)),
                        )
                        .when(removable, |d| {
                            d.child(button(
                                &th2,
                                format!("rm-lib-{i}"),
                                "Remove",
                                Btn::Ghost,
                                click(cx, move |s, cx| {
                                    let lib = lib.clone();
                                    s.call(
                                        cx,
                                        move |c| c.remove_library(&lib),
                                        |s, libs, cx| {
                                            s.st.libraries = libs;
                                            s.refresh(cx)
                                        },
                                    )
                                }),
                            ))
                        }),
                )
                .child(
                    div()
                        .flex()
                        .items_end()
                        .gap(u(10.))
                        .child(serif(logic::gb(bytes), 34.))
                        .child(
                            div()
                                .pb(u(5.))
                                .text_size(u(13.5))
                                .text_color(th.ink2)
                                .child(if ms.len() == 1 {
                                    "1 model".to_owned()
                                } else {
                                    format!("{} models", ms.len())
                                }),
                        ),
                )
                .child(
                    div()
                        .text_size(u(13.))
                        .text_color(th.ink3)
                        .child(if i == 0 {
                            "Downloads land here unless you pick another folder.".to_owned()
                        } else {
                            format!("Read in place from {source}.")
                        }),
                )
        })
        .collect::<Vec<_>>();
    let add = div()
        .id("add-lib")
        .w(u(180.))
        .min_h(u(130.))
        .flex_none()
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .gap(u(6.))
        .rounded(u(14.))
        .border_1()
        .border_dashed()
        .border_color(th.line)
        .text_color(th.ink2)
        .cursor_pointer()
        .hover(move |d| d.bg(th2.sunk))
        .on_click(cx.listener(|s, _, _, cx| add_library(s, cx)))
        .child(icon("folder", 18., th.ink2))
        .child(div().text_size(u(13.5)).child("Add a folder"));
    let mut models: Vec<&ModelEntry> = s.models.iter().collect();
    models.sort_by_key(|m| (s.st.model.as_ref() != Some(&m.path), !m.supported));
    let cards: Vec<AnyElement> = models
        .into_iter()
        .map(|m| model_card(s, th, m, cx).into_any_element())
        .collect();
    div()
        .flex()
        .flex_col()
        .gap(u(20.))
        .children(downloads_el(s, th, cx))
        .child(
            div()
                .flex()
                .flex_wrap()
                .gap(u(16.))
                .children(lib_cards)
                .child(add),
        )
        .when(cards.is_empty(), |d| {
            d.child(card(th).child(serif("No models yet", 27.)).child(
                div().text_color(th.ink2).child(
                    "Find one in Discover: Larch checks it fits your GPU before it downloads.",
                ),
            ))
        })
        .child(
            div()
                .grid()
                .grid_cols(logic::columns(s.room - PAGE_X * 2., 260., 16., 3))
                .gap(u(16.))
                .children(cards),
        )
}

/// "35B · MoE" from a model's name.
fn size_tag(name: &str) -> Option<String> {
    let p = logic::params(name)?;
    let moe = larch_app::is_moe(name, "");
    Some(format!(
        "{}B{}",
        logic::fmt_v(p),
        if moe { " · MoE" } else { "" }
    ))
}

/// The card's "Faster on this computer" section: the offer, the copy being made, or the switch
/// between the faster version and the original. None where no faster version is offered.
fn faster_el(s: &Ecl, th: &Theme, m: &ModelEntry, cx: &mut Context<Ecl>) -> Option<Div> {
    let f = m.faster.as_ref()?;
    let on = f.copy.is_some() && s.st.model == f.copy;
    let making = s
        .fasters
        .get(&m.path)
        .copied()
        .or(f.building.then_some((0., 0.)));
    let (path, th2) = (m.path.clone(), *th);
    let head = div().flex().flex_wrap().items_center().gap(u(8.)).child(
        div()
            .font_weight(FontWeight::SEMIBOLD)
            .child("Faster on this computer"),
    );
    let note = |t: String| div().text_size(u(13.)).text_color(th2.ink2).child(t);
    let body =
        if let Some((frac, secs)) = making {
            let p = path.clone();
            div()
            .flex()
            .flex_col()
            .gap(u(8.))
            .child(
                head.child(
                    div()
                        .text_size(u(13.))
                        .text_color(th.ink2)
                        .child(logic::making_left(frac, secs)),
                ),
            )
            .child(bar(th, &[(frac as f32, th.sun.into())], 6.))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .justify_between()
                    .gap(u(8.))
                    .child(div().flex_1().min_w(u(200.)).child(note(
                        "You can keep chatting meanwhile. It switches over when it is ready."
                            .into(),
                    )))
                    .child(button(
                        th,
                        format!("faster-cancel-{path}"),
                        "Cancel",
                        Btn::Ghost,
                        click(cx, move |s, cx| {
                            let p = p.clone();
                            s.call(cx, move |c| c.cancel_faster(&p), |_, _, _| {})
                        }),
                    )),
            )
        } else if f.copy.is_some() {
            let p = path.clone();
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .justify_between()
                .gap(u(8.))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(u(4.))
                        .child(head.child(tag(th, if on { "On" } else { "Off" })))
                        .child(note(match on {
                            true => "Running the faster version.".into(),
                            false => "The faster version is ready.".into(),
                        })),
                )
                .child(button(
                    th,
                    format!("faster-use-{path}"),
                    if on {
                        "Use the original"
                    } else {
                        "Use the faster version"
                    },
                    if on { Btn::Outline } else { Btn::Primary },
                    click(cx, move |s, cx| s.use_faster(p.clone(), !on, cx)),
                ))
        } else {
            let p = path.clone();
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .justify_between()
                .gap(u(8.))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(u(4.))
                        .flex_1()
                        .min_w(u(200.))
                        .child(head.child(tag(th, "Recommended")))
                        .child(note(logic::FASTER_NOTE.into())),
                )
                .child(button(
                    th,
                    format!("faster-make-{path}"),
                    "Make it faster",
                    Btn::Primary,
                    click(cx, move |s, cx| s.make_faster(p.clone(), cx)),
                ))
        };
    Some(
        div()
            .pt(u(12.))
            .border_t_1()
            .border_color(th.line)
            .child(body),
    )
}

fn model_card(s: &Ecl, th: &Theme, m: &ModelEntry, cx: &mut Context<Ecl>) -> Div {
    let th2 = *th;
    // Its faster version running is this model running.
    let copy = m.faster.as_ref().and_then(|f| f.copy.as_ref());
    let running =
        s.st.model.as_ref() == Some(&m.path) || (copy.is_some() && s.st.model.as_ref() == copy);
    let loading = running && s.st.health == Health::Loading;
    let have = s.gpu_bytes();
    let (fit, fit_c, right, frac) = match &m.fit {
        Fit::Fits { max_seqs, .. } => (
            "Fits",
            th.keep,
            format!(
                "room for {max_seqs} chat{} at once",
                if *max_seqs == 1 { "" } else { "s" }
            ),
            have.map(|h| m.size_bytes as f32 / h as f32),
        ),
        Fit::TooBig {
            need_bytes,
            have_bytes,
        } => (
            "Too big",
            th.lose,
            format!(
                "needs {} of {}",
                logic::gb(*need_bytes),
                logic::gb(*have_bytes)
            ),
            Some(1.),
        ),
        Fit::Unknown if !m.supported => {
            ("Not supported yet", th.lose, home_short(&m.library), None)
        }
        Fit::Unknown => ("Not checked", th.ink3, String::new(), None),
    };
    let tok = (s.prefs.tok_s.get(match copy.filter(|_| running) {
        Some(c) if s.st.model.as_ref() == Some(c) => c,
        _ => &m.path,
    }))
    .copied();
    let (path, del, has_copy) = (m.path.clone(), m.path.clone(), copy.is_some());
    let armed = s.armed.as_deref() == Some(m.path.as_str());
    // Larch downloads into the Hugging Face cache, so its models there delete here too.
    let deletable = matches!(m.source.as_str(), "Larch" | "Hugging Face") && !running;
    card(th)
        .min_w_0()
        .gap(u(12.))
        .when(running, |d| {
            d.border_color(th2.sun).shadow(vec![gpui::BoxShadow {
                color: fade(th2.sun, 0.25),
                offset: point(px(0.), px(0.)),
                blur_radius: px(0.),
                spread_radius: px(2.),
            }])
        })
        // In a narrow card the badge goes under a long name, which broke a letter a line beside it;
        // the tags, the fit and the buttons wrap rather than run past the card.
        .child(
            div()
                .flex()
                .flex_wrap()
                .items_start()
                .gap(u(10.))
                .child(
                    serif(m.name.clone(), 27.)
                        .flex_1()
                        .min_w(u(150.))
                        .when(!m.supported, |d| d.text_color(th2.ink3)),
                )
                .when(running, |d| {
                    d.child(
                        div()
                            .flex()
                            .items_center()
                            .gap(u(6.))
                            .px(u(12.))
                            .py(u(4.))
                            .rounded_full()
                            .border_1()
                            .border_color(th2.line)
                            .text_size(u(13.))
                            .child(div().size(u(9.)).rounded_full().bg(th2.sun))
                            .child(if loading { "Loading" } else { "Running" }),
                    )
                }),
        )
        .child(
            div()
                .flex()
                .flex_wrap()
                .gap(u(6.))
                .when(!m.supported, |d| d.opacity(0.6))
                .children(size_tag(&m.name).map(|t| tag(&th2, t)))
                .child(tag(th, m.quant.clone()))
                .child(tag(th, logic::gb(m.size_bytes)))
                .when(m.vision, |d| d.child(tag(&th2, "Reads images".to_owned()))),
        )
        .child(bar(
            th,
            &frac.map_or(Vec::new(), |f| {
                vec![(
                    f,
                    if fit == "Too big" {
                        th.lose.into()
                    } else {
                        th.accent.into()
                    },
                )]
            }),
            6.,
        ))
        .child(
            div()
                .flex()
                .flex_wrap()
                .gap_x(u(10.))
                .justify_between()
                .text_size(u(13.))
                .child(
                    div()
                        .text_color(fit_c)
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(fit),
                )
                .child(div().min_w_0().text_color(th.ink2).child(right)),
        )
        .children(faster_el(s, th, m, cx))
        .child(div().flex_1().min_h(u(16.)))
        .child(
            div()
                .flex()
                .flex_wrap()
                .gap(u(8.))
                .items_end()
                .justify_between()
                .child(match tok {
                    Some(t) => div()
                        .flex()
                        .items_end()
                        .gap(u(6.))
                        .child(serif(format!("{t:.0}"), 44.))
                        .child(
                            div()
                                .pb(u(6.))
                                .text_size(u(13.))
                                .text_color(th.ink2)
                                .child("tok/s"),
                        ),
                    None => div().text_size(u(24.)).text_color(th.ink3).child("—"),
                })
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(u(8.))
                        .when(deletable && (armed || !m.supported), |d| {
                            d.child(button(
                                &th2,
                                format!("del-{}", m.path),
                                // The cache is shared: what goes there goes for every app.
                                match (armed, m.source == "Hugging Face") {
                                    (true, true) => "Delete for every app?",
                                    (true, false) => "Delete?",
                                    (false, _) => "Delete",
                                },
                                if armed { Btn::Danger } else { Btn::Ghost },
                                click(cx, move |s, cx| {
                                    if s.armed.as_deref() != Some(del.as_str()) {
                                        s.armed = Some(del.clone());
                                        return;
                                    }
                                    s.armed = None;
                                    let p = del.clone();
                                    let name = logic::base_name(&del).to_owned();
                                    s.call(
                                        cx,
                                        move |c| c.delete_model(&p),
                                        move |s, _, cx| {
                                            // Discover's rows said "In library" and offered no
                                            // download until a restart.
                                            (s.hf.files.values_mut().flatten().flatten())
                                                .filter(|f| logic::base_name(&f.file) == name)
                                                .for_each(|f| f.downloaded = false);
                                            s.say("Deleted.");
                                            s.refresh(cx)
                                        },
                                    );
                                }),
                            ))
                        })
                        .when(m.supported && !running, |d| {
                            d.child(button(
                                &th2,
                                format!("load-{}", m.path),
                                "Load",
                                Btn::Outline,
                                // A model with its faster version made loads as that.
                                click(cx, move |s, cx| match has_copy {
                                    true => s.use_faster(path.clone(), true, cx),
                                    false => s.load_model(path.clone(), cx),
                                }),
                            ))
                        })
                        .when(deletable && m.supported && !armed, |d| {
                            let arm = m.path.clone();
                            d.child(
                                div()
                                    .id(gpui::ElementId::Name(format!("trash-{}", m.path).into()))
                                    .p(u(8.))
                                    .rounded(u(8.))
                                    .cursor_pointer()
                                    .hover(move |d| d.bg(th2.sunk))
                                    .on_click(click(cx, move |s, _| s.armed = Some(arm.clone())))
                                    .child(icon("trash", 15., th2.ink3)),
                            )
                        }),
                ),
        )
}

/// The running downloads, one line each, on both Models tabs; None when nothing downloads.
fn downloads_el(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Option<Div> {
    if s.dls.is_empty() {
        return None;
    }
    let rows = s.dls.iter().map(|(id, d)| {
        let frac = if d.total > 0 {
            d.got as f32 / d.total as f32
        } else {
            0.
        };
        let left = (d.bps > 0. && d.total > d.got).then(|| {
            let s = (d.total - d.got) as f64 / d.bps;
            if s < 90. {
                format!(" · {s:.0} s left")
            } else {
                format!(" · {:.0} min left", s / 60.)
            }
        });
        let stem = logic::base_name(&d.file).trim_end_matches(".gguf");
        let quant = QUANT_TAIL
            .find(stem)
            .map_or("", |m| &stem[m.start() + 1..])
            .to_owned();
        let name = stem[..stem.len() - quant.len()]
            .trim_end_matches(['-', '_', '.'])
            .replace(['-', '_'], " ");
        let id = id.clone();
        div()
            .flex()
            .items_center()
            .gap(u(16.))
            // A narrow page wraps the name and the numbers before Cancel runs off its edge
            // (a truncated one was cut mid-letter: gpui sizes its ellipsis before the row shrinks).
            .child(
                div()
                    .w(u(340.))
                    .min_w_0()
                    .flex()
                    .items_center()
                    .gap(u(8.))
                    .child(
                        div()
                            .min_w_0()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(name),
                    )
                    .when(!quant.is_empty(), |x| x.child(tag(th, quant))),
            )
            .child(
                div()
                    .flex_1()
                    .min_w(u(60.))
                    .child(bar(th, &[(frac, th.sun.into())], 6.)),
            )
            .child(
                div()
                    .w(u(280.))
                    .min_w_0()
                    .text_size(u(13.))
                    .text_color(th.ink2)
                    .child(if d.total > 0 {
                        format!(
                            "{:.0} % · {} of {} · {:.0} MB/s{}",
                            frac * 100.,
                            logic::gb(d.got).trim_end_matches(" GB"),
                            logic::gb(d.total),
                            d.bps / 1e6,
                            left.unwrap_or_default()
                        )
                    } else {
                        "Connecting to Hugging Face…".into()
                    }),
            )
            .child(button(
                th,
                format!("cancel-{id}"),
                "Cancel",
                Btn::Ghost,
                click(cx, move |s, cx| {
                    let id = id.clone();
                    s.call(cx, move |c| c.cancel_download(&id), |_, _, _| {})
                }),
            ))
    });
    Some(
        card(th)
            .gap(u(12.))
            .child(eyebrow(th, "Downloading"))
            .children(rows),
    )
}

static QUANT_TAIL: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r"(?i)[-_.](I?Q\d[A-Z0-9_]*|F16|BF16|F32)$").expect("a valid regex")
});

const USES: [&str; 5] = ["Chat", "Coding", "Reasoning", "Vision", "Long documents"];
const SIZES: [&str; 3] = ["Small", "Medium", "Large"];

/// The Discover filters a repo passes, from its name, whether it reads images and, for Long
/// documents, its context.
fn uses(repo: &str, ctx: Option<u64>, vision: bool) -> Vec<&'static str> {
    let r = repo.to_lowercase();
    let mut u = Vec::new();
    if r.contains("coder") || r.contains("code") {
        u.push("Coding");
    }
    if ["r1", "reason", "think", "qwq"]
        .iter()
        .any(|k| r.contains(k))
    {
        u.push("Reasoning");
    }
    if vision {
        u.push("Vision");
    }
    if ctx.is_some_and(|c| c >= 128 * 1024) {
        u.push("Long documents");
    }
    if u.is_empty() || r.contains("instruct") || r.contains("chat") {
        u.push("Chat");
    }
    u
}

/// Whether a Hub repo reads images in larch: the Hub files it as image-text-to-text (the search
/// asks for those too), or the file offered ships a projector larch loads (InternVL's, Qwen 3.5's).
fn reads_images(m: &HfModel, file: Option<&HfFile>) -> bool {
    m.tags.iter().any(|t| t == "image-text-to-text") || file.is_some_and(|f| f.mmproj.is_some())
}

/// A projector's own repo, and vision models whose projectors larch does not load ("-VL-" or
/// "vision" in the name), are left out of Discover; InternVL's and Qwen 3.5's pass.
fn hidden_repo(repo: &str) -> bool {
    static HIDDEN: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?i)mmproj|vision|-vl-|\bvl\b").expect("a valid regex")
    });
    HIDDEN.is_match(repo)
}

fn size_class(p: f64) -> &'static str {
    if p <= 8. {
        "Small"
    } else if p <= 24. {
        "Medium"
    } else {
        "Large"
    }
}

/// What one Discover row knows: the file it offers and that file's header.
struct Row<'a> {
    m: &'a HfModel,
    name: String,
    by: String,
    params: Option<f64>,
    file: Option<&'a HfFile>,
    info: Option<&'a RemoteModel>,
    err: Option<&'a str>,
}

fn rows(s: &Ecl) -> Vec<Row<'_>> {
    let have = s.gpu_room();
    let ctx = s
        .knobs
        .iter()
        .find(|k| k.name == "max_context")
        .map_or(8192., |k| k.value);
    s.hf.results
        .iter()
        .filter(|m| !hidden_repo(&m.repo))
        .map(|m| {
            let (by, repo) = m.repo.split_once('/').unwrap_or(("", &m.repo));
            let name = repo
                .trim_end_matches("-GGUF")
                .trim_end_matches("-gguf")
                .trim_end_matches("_GGUF")
                .replace(['-', '_'], " ");
            let files = s.hf.files.get(&m.repo);
            let file = files
                .and_then(|f| f.as_ref().ok())
                .and_then(|f| pick_file(f, have));
            let key = file.map(|f| (m.repo.clone(), f.file.clone()));
            let info = key.as_ref().and_then(|k| s.hf.info.get(k));
            let bad = key.as_ref().and_then(|k| s.hf.bad.get(k));
            Row {
                m,
                params: logic::params(&name),
                name,
                by: by.to_owned(),
                file,
                info,
                err: files
                    .and_then(|f| f.as_ref().err())
                    .or(bad)
                    .map(String::as_str),
            }
        })
        .filter(|r| {
            // A row whose header (a few MB) is still on its way is judged by what the search and
            // its file list already say, so a filter answers at once; the header then refines it.
            let pending = r.info.is_none() && r.err.is_none();
            let runs = match r.info {
                Some(i) => i.supported,
                // The Hub may not know the architecture: then only the header can say. A file
                // whose quant the engine cannot load is out at once, before its header.
                None => {
                    pending
                        && (r.m.arch_runs || r.m.arch.is_empty())
                        && r.file.is_none_or(|f| f.quant.is_empty() || loads(&f.quant))
                }
            };
            let fits = match (r.info, r.file) {
                (Some(i), _) => i.supported && have.is_none_or(|h| floor_bytes(i, ctx) <= h),
                // The rule the row's file was picked by; an MoE waits for its header, since its
                // experts can stay in host memory.
                (None, Some(f)) => {
                    pending
                        && (larch_app::is_moe(&r.m.repo, &r.m.arch)
                            || have.is_none_or(|h| f.size_bytes + logic::GIB as u64 <= h))
                }
                // Its files are still being listed.
                (None, None) => pending && !s.hf.files.contains_key(&r.m.repo),
            };
            let ctx_len = r.info.and_then(|i| i.context_length).or(r.m.context_length);
            (!s.hf.fits || (runs && fits))
                && (!s.hf.runs || runs)
                && s.hf.use_.is_none_or(|u| {
                    uses(&r.m.repo, ctx_len, reads_images(r.m, r.file)).contains(&u)
                })
                && s.hf
                    .size
                    .is_none_or(|z| r.params.is_some_and(|p| size_class(p) == z))
        })
        .collect()
}

fn discover(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let th2 = *th;
    let gpu = if s.st.device.is_empty() {
        "GPU".to_owned()
    } else {
        s.st.device.clone()
    };
    let have = s.gpu_room();
    let ctx = s
        .knobs
        .iter()
        .find(|k| k.name == "max_context")
        .map_or(8192., |k| k.value);
    let rs = rows(s);
    let filter_pill =
        |id: &str, label: String, on: bool, f: Box<dyn Fn(&mut Ecl)>, cx: &mut Context<Ecl>| {
            pill(
                &th2,
                id.to_owned(),
                if on { format!("✓ {label}") } else { label },
                on,
                click(cx, move |s, _| f(s)),
            )
        };
    let mut filters = div().flex().flex_wrap().items_center().gap(u(8.));
    filters = filters.when(have.is_some(), |d| {
        d.child(filter_pill(
            "f-fits",
            format!("Fits your {gpu}"),
            s.hf.fits,
            Box::new(|s| s.hf.fits = !s.hf.fits),
            cx,
        ))
    });
    filters = filters
        .child(filter_pill(
            "f-runs",
            "Runs in Larch".into(),
            s.hf.runs,
            Box::new(|s| s.hf.runs = !s.hf.runs),
            cx,
        ))
        .child(eyebrow(th, "Use").text_color(th.ink3).ml(u(14.)).mr(u(4.)));
    for u_ in USES {
        let on = s.hf.use_ == Some(u_);
        filters = filters.child(pill(
            th,
            format!("f-use-{u_}"),
            u_,
            on,
            click(cx, move |s, _| {
                s.hf.use_ = if s.hf.use_ == Some(u_) {
                    None
                } else {
                    Some(u_)
                }
            }),
        ));
    }
    filters = filters
        .child(eyebrow(th, "Size").text_color(th.ink3).ml(u(14.)).mr(u(4.)))
        .child(pill(
            th,
            "f-size-any",
            "Any",
            s.hf.size.is_none(),
            click(cx, |s, _| s.hf.size = None),
        ));
    for z in SIZES {
        filters = filters.child(pill(
            th,
            format!("f-size-{z}"),
            z,
            s.hf.size == Some(z),
            click(cx, move |s, _| {
                s.hf.size = if s.hf.size == Some(z) { None } else { Some(z) }
            }),
        ));
    }
    // Below this page width the columns wrap under each row's name, so they go unlabelled.
    let wide = s.room >= DISCOVER_ROW;
    let head = div()
        .flex()
        .items_center()
        .gap(u(16.))
        .pb(u(12.))
        .border_b_1()
        .border_color(th.line)
        .child(
            eyebrow(th, format!("{} models · from Hugging Face", rs.len()))
                .text_color(th.ink3)
                .flex_1(),
        )
        .when(wide, |d| {
            d.child(eyebrow(th, "Best for you").text_color(th.ink3).w(u(200.)))
                .child(
                    eyebrow(th, format!("On your {gpu}"))
                        .text_color(th.ink3)
                        .w(u(280.)),
                )
                .child(eyebrow(th, "Downloads").text_color(th.ink3).w(u(100.)))
                .child(div().w(u(120.)))
        });
    let empty = if s.hf.loading {
        "Searching Hugging Face…"
    } else if s.core.is_none() && s.hf.results.is_empty() {
        "Search for a model, and Larch shows how it runs on this computer before you download it."
    } else if s.hf.searched && rs.is_empty() {
        "Nothing matches. Try fewer filters."
    } else {
        ""
    };
    let table = card(th)
        .p(u(24.))
        .gap(u(0.))
        .child(head)
        .when(!empty.is_empty(), |d| {
            d.child(div().py(u(18.)).text_color(th.ink2).child(empty))
        })
        .children(rs.iter().map(|r| {
            let repo = r.m.repo.clone();
            let open = s.hf.open.as_deref() == Some(r.m.repo.as_str());
            let fit: Div = match (r.info, r.err) {
                (_, Some(e)) => div()
                    .text_size(u(13.))
                    .text_color(th2.ink3)
                    .child(logic::clip(e, 60)),
                // Until its header is read, a row says what the search and file list already
                // tell: the Hub's architecture, and the file's size against the room.
                (None, _) if !r.m.arch_runs && !r.m.arch.is_empty() => div()
                    .text_size(u(13.))
                    .text_color(th2.lose)
                    .child(format!("Larch can't run {}", r.m.arch)),
                (None, _) => {
                    div()
                        .text_size(u(13.))
                        .text_color(th2.ink3)
                        .child(match (r.file, have) {
                            (None, _) if s.hf.files.contains_key(&r.m.repo) => {
                                "No GGUF files".into()
                            }
                            (None, _) => "Listing its files…".into(),
                            (Some(f), Some(h)) if f.size_bytes + logic::GIB as u64 > h => {
                                format!("Too big, about {} · checking", logic::gb(f.size_bytes))
                            }
                            (Some(f), Some(_)) => {
                                format!("Likely fits, {} · checking", logic::gb(f.size_bytes))
                            }
                            (Some(_), None) => "Checking…".into(),
                        })
                }
                (Some(i), _) if !i.supported => div()
                    .text_size(u(13.))
                    .text_color(th2.lose)
                    .child(cannot_run(i)),
                (Some(i), _) => {
                    let need = need_bytes(i, ctx);
                    match have {
                        Some(h) => {
                            // An MoE too big whole still fits with experts in host memory.
                            let split = need > h && floor_bytes(i, ctx) <= h;
                            let need = if split { floor_bytes(i, ctx) } else { need };
                            let ok = need <= h;
                            let room = i.kv_bytes_per_token.filter(|k| *k > 0).map(|k| {
                                let free = h
                                    .saturating_sub(i.weights_bytes + (logic::GIB / 2.) as u64)
                                    / k;
                                i.context_length.map_or(free, |c| free.min(c))
                            });
                            div()
                                .flex()
                                .flex_col()
                                .gap(u(6.))
                                .child(bar(
                                    &th2,
                                    &[(
                                        need as f32 / h as f32,
                                        if ok {
                                            th2.accent.into()
                                        } else {
                                            th2.lose.into()
                                        },
                                    )],
                                    6.,
                                ))
                                .child(
                                    div()
                                        .flex()
                                        .gap(u(4.))
                                        .text_size(u(13.))
                                        .child(
                                            div()
                                                .font_weight(FontWeight::SEMIBOLD)
                                                .text_color(if ok { th2.keep } else { th2.lose })
                                                .child(if ok { "Fits" } else { "Too big" }),
                                        )
                                        .child(div().text_color(th2.ink2).child(
                                            match (ok, room) {
                                                (true, _) if split => {
                                                    "· some experts on the CPU".to_owned()
                                                }
                                                (true, Some(r)) if r < 16384 => {
                                                    format!("· tight, {}", logic::fmt_k(r))
                                                }
                                                (true, Some(r)) => {
                                                    format!("· room for {}", logic::fmt_k(r))
                                                }
                                                (true, None) => String::new(),
                                                (false, _) => {
                                                    format!("· needs {}", logic::gb(need))
                                                }
                                            },
                                        )),
                                )
                        }
                        None => div()
                            .text_size(u(13.))
                            .text_color(th2.ink2)
                            .child(format!("Needs {}", logic::gb(need))),
                    }
                }
            };
            let action: AnyElement = match r.file {
                Some(f) if f.downloaded => div()
                    .px(u(14.))
                    .py(u(6.))
                    .rounded_full()
                    .border_1()
                    .border_color(th2.line)
                    .text_size(u(13.))
                    .text_color(th2.ink2)
                    .child("In library")
                    .into_any_element(),
                Some(f) => match s
                    .dls
                    .values()
                    .find(|d| d.repo == r.m.repo && d.file == f.file)
                {
                    Some(d) => div()
                        .text_size(u(13.))
                        .text_color(th2.ink2)
                        .child(if d.total > 0 {
                            format!("{:.0} %", d.got as f64 / d.total as f64 * 100.)
                        } else {
                            "Starting…".into()
                        })
                        .into_any_element(),
                    None => {
                        let (rp, file) = (repo.clone(), f.file.clone());
                        button(
                            &th2,
                            format!("dl-{repo}"),
                            "Download",
                            Btn::Outline,
                            cx.listener(move |s, _, _, cx| {
                                cx.stop_propagation();
                                s.download(rp.clone(), file.clone(), false, cx)
                            }),
                        )
                        .into_any_element()
                    }
                },
                None => div().into_any_element(),
            };
            let toggle = repo.clone();
            div()
                .flex()
                .flex_col()
                .border_b_1()
                .border_color(th2.line)
                .child(
                    div()
                        .id(gpui::ElementId::Name(format!("row-{repo}").into()))
                        .flex()
                        .items_center()
                        .gap(u(16.))
                        .when(!wide, |d| d.flex_wrap().gap_y(u(10.)))
                        .py(u(14.))
                        .cursor_pointer()
                        .on_click(click(cx, move |s, cx| {
                            let shut = s.hf.open.as_deref() == Some(toggle.as_str());
                            s.hf.open = if shut { None } else { Some(toggle.clone()) };
                            // A listing that failed (a network blip) is asked for again on open.
                            if !shut && !matches!(s.hf.files.get(&toggle), Some(Ok(_))) {
                                s.hf_files(toggle.clone(), cx);
                            }
                        }))
                        // The tags sit under the name: beside it they left the name a few
                        // letters ("Qwen2.5 Coc").
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .when(!wide, |d| d.flex_basis(gpui::relative(1.)))
                                .flex()
                                .flex_col()
                                .gap(u(4.))
                                .child(
                                    div()
                                        .text_size(u(15.))
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .child(r.name.clone()),
                                )
                                .child(
                                    div()
                                        .flex()
                                        .flex_wrap()
                                        .items_center()
                                        .gap(u(6.))
                                        .text_size(u(13.))
                                        .text_color(th2.ink3)
                                        .child(r.by.clone())
                                        .children(size_tag(&r.name).into_iter().flat_map(|t| {
                                            t.split(" · ")
                                                .map(|p| tag(&th2, p.to_owned()))
                                                .collect::<Vec<_>>()
                                        })),
                                ),
                        )
                        .child(
                            div()
                                .w(u(200.))
                                .flex()
                                .gap(u(8.))
                                .items_baseline()
                                .children(r.file.map(|f| mono(f.quant.clone()).text_size(u(13.))))
                                .children(r.file.map(|f| {
                                    div()
                                        .text_size(u(13.))
                                        .text_color(th2.ink3)
                                        .child(logic::gb(f.size_bytes))
                                }))
                                .when(reads_images(r.m, r.file), |d| {
                                    d.child(tag(&th2, "Reads images".to_owned()))
                                }),
                        )
                        .child(div().w(u(280.)).child(fit))
                        .child(
                            mono(logic::fmt_k(r.m.downloads))
                                .w(u(100.))
                                .text_color(th2.ink2),
                        )
                        .child(div().w(u(120.)).flex().justify_end().child(action)),
                )
                .when(open, |d| d.child(files_el(s, &th2, &repo, cx)))
        }));
    let sort = s.hf.sort;
    div()
        .flex()
        .flex_col()
        .gap(u(16.))
        .child(
            div()
                .flex()
                .items_center()
                .gap(u(12.))
                .child(
                    div()
                        .flex_1()
                        .flex()
                        .items_center()
                        .gap(u(10.))
                        .px(u(18.))
                        .py(u(13.))
                        .rounded(u(14.))
                        .border_1()
                        .border_color(th.line)
                        .bg(th.card)
                        .text_size(u(15.))
                        .child(icon("search", 16., th.ink3))
                        .child(div().flex_1().child(s.search.clone())),
                )
                .child(seg(
                    th,
                    "sort",
                    &["Most downloaded", "Most liked", "Newest"],
                    sort,
                    set(cx, |s, i, cx| {
                        s.hf.sort = i;
                        s.hf_search(cx)
                    }),
                )),
        )
        .children(downloads_el(s, th, cx))
        .when(shows_recommended(s, cx), |d| {
            d.child(
                recommended(s.gpu_bytes(), th, cx)
                    .relative()
                    .children(crate::tour::marks(
                        s,
                        crate::tour::Spot::Recommended,
                        th,
                        cx,
                    )),
            )
        })
        .child(filters)
        .child(table)
}

/// Whether Discover shows its "Recommended for this computer" card: on the Discover tab, with an
/// empty search box (a search shows its own results instead). The tour asks too, to place its card.
pub fn shows_recommended(s: &Ecl, cx: &App) -> bool {
    s.page == Page::Models && s.mtab == MTab::Discover && s.search.read(cx).text.trim().is_empty()
}

/// A model the catalog recommends: where it downloads from and what it is good for.
#[derive(serde::Deserialize)]
struct Pick {
    repo: String,
    file: String,
    #[serde(default, rename = "display")]
    name: String,
    #[serde(default, rename = "note")]
    why: String,
    #[serde(default, rename = "use")]
    uses: Vec<String>,
    #[serde(default)]
    gpu_gb: f64,
}

/// The catalog's recommended models (`use = ["recommend"]` in bench/models.toml, where model names
/// live) for a GPU of `gpu` bytes: each row whose `gpu_gb` it reaches, the largest `gpu_gb` first.
/// An unknown GPU reaches only the rows for any GPU (`gpu_gb = 0`).
/// It compares the GPU's total memory, not what other programs left free (`gpu_room`), on purpose:
/// the card says what suits this computer; the table's fit check says what fits right now.
fn recommended_models(gpu: Option<u64>) -> Vec<Pick> {
    let catalog: std::collections::BTreeMap<String, Pick> =
        toml::from_str(include_str!("../../../bench/models.toml")).unwrap_or_default();
    let have = gpu.unwrap_or(0) as f64 / 1e9;
    let mut rows: Vec<Pick> = (catalog.into_values())
        .filter(|e| e.uses.iter().any(|u| u == "recommend") && e.gpu_gb <= have)
        .collect();
    // Stable: rows of one size stay in alias order.
    rows.sort_by(|a, b| b.gpu_gb.total_cmp(&a.gpu_gb));
    rows
}

/// "Recommended for this computer": each model the catalog recommends for this GPU, and a button
/// that searches for it (its files, fit and download then show in the table).
fn recommended(gpu: Option<u64>, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let mut c = card(th)
        .p(u(20.))
        .gap(u(10.))
        .child(eyebrow(th, "Recommended for this computer"));
    for Pick {
        repo, name, why, ..
    } in recommended_models(gpu)
    {
        c = c.child(
            div()
                .flex()
                .items_center()
                .gap(u(14.))
                .child(
                    div()
                        .w(u(150.))
                        .flex_none()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(name.clone()),
                )
                // min_w 0: in a narrow window the words wrap, not push Show out of the card.
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .text_size(u(13.5))
                        .text_color(th.ink2)
                        .child(why),
                )
                .child(button(
                    th,
                    format!("rec-{name}"),
                    "Show",
                    Btn::Outline,
                    click(cx, move |s, cx| {
                        s.search.update(cx, |i, cx| i.set_text(&repo, cx));
                        s.hf_search(cx);
                    }),
                )),
        );
    }
    c
}

/// Home's card while this computer has no model: up to three the catalog recommends for its GPU,
/// each a click from downloading, and a way in for model files already on disk.
pub fn first_model(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let gpu = s.gpu_bytes();
    let mut c = card(th).p(u(20.)).gap(u(12.)).child(eyebrow(
        th,
        match gpu {
            Some(b) => format!("A first model · sized to your {} GPU", logic::gb(b)),
            None => "A first model".into(),
        },
    ));
    for (
        i,
        Pick {
            repo,
            file,
            name,
            why,
            ..
        },
    ) in recommended_models(gpu).into_iter().take(3).enumerate()
    {
        let busy = s.dls.values().any(|d| d.repo == repo && d.file == file);
        c = c.child(
            div()
                .flex()
                .items_center()
                .gap(u(16.))
                .p(u(14.))
                .rounded(u(10.))
                .bg(th.sunk)
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap(u(3.))
                        .child(div().font_weight(FontWeight::SEMIBOLD).child(name.clone()))
                        .child(div().text_size(u(13.)).text_color(th.ink2).child(why)),
                )
                // No size here: the catalog's is the model file alone, and a model that reads
                // images downloads its image reader too. The download's own row below shows the
                // whole size, its progress and Cancel.
                .child(if busy {
                    div()
                        .flex_none()
                        .text_size(u(13.))
                        .text_color(th.ink3)
                        .child("Downloading")
                } else {
                    div().flex_none().child(button(
                        th,
                        format!("first-{name}"),
                        "Download",
                        // The largest that suits this GPU is the one to take.
                        if i == 0 { Btn::Primary } else { Btn::Outline },
                        click(cx, move |s, cx| {
                            s.download(repo.clone(), file.clone(), false, cx)
                        }),
                    ))
                }),
        );
    }
    c.children(downloads_el(s, th, cx)).child(
        div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap(u(10.))
            .pt(u(4.))
            .child(
                div()
                    .flex_1()
                    .min_w(u(200.))
                    .text_size(u(13.5))
                    .text_color(th.ink2)
                    .child("Already have GGUF files? Add their folder."),
            )
            .child(button(
                th,
                "first-folder",
                "Add a folder",
                Btn::Ghost,
                click(cx, add_library),
            ))
            .child(button(
                th,
                "first-more",
                "More models",
                Btn::Ghost,
                click(cx, |s, cx| {
                    s.mtab = MTab::Discover;
                    s.go(Page::Models, cx)
                }),
            )),
    )
}

/// A quiet note on a page, as a chat's notices look.
fn notice(th: &Theme, text: impl Into<SharedString>) -> Div {
    div()
        .text_size(u(13.5))
        .text_color(th.ink2)
        .px(u(14.))
        .py(u(10.))
        .rounded(u(9.))
        .bg(th.sunk)
        .child(text.into())
}

/// MCP tools: your servers (status, tools, on/off, sign-in), "add your own", and the MCP Registry's
/// servers with an install form that shows what will run before anything does.
pub fn mcp(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Stateful<Div> {
    // The tools chats take: one a chat leaves out adds no wrong pick.
    let on_tools: usize = (s.mcp.servers.iter())
        .filter(|x| x.enabled && x.status == "connected")
        .map(|x| x.tools.iter().filter(|t| t.left_out.is_none()).count())
        .sum();
    // First: a page's text registers in reading order.
    let intro = div().text_size(u(14.)).text_color(th.ink2).child(s.sel.label(
        PAGE,
        "MCP servers give your chats more tools: a calendar, a database, a service's API. Each one runs on this computer or is reached over the internet, and while it is on its tools join every new chat. A tool that does more than read asks you first.",
    ));
    let mut mine = card(th)
        .p(u(24.))
        .gap(u(14.))
        .child(eyebrow(th, "Your MCP servers"));
    if s.mcp.servers.is_empty() {
        mine = mine.child(div().text_color(th.ink2).child(s.sel.label(
            PAGE,
            "No MCP servers yet. Pick a suggested one below, search the registry, or add your own.",
        )));
    }
    for srv in &s.mcp.servers {
        mine = mine.child(mcp_server_row(s, th, srv, cx));
    }
    let status = if s.mcp.loading {
        "Searching the MCP Registry…".to_owned()
    } else if let Some(e) = &s.mcp.err {
        format!("Could not reach the MCP Registry: {}", logic::clip(e, 120))
    } else if s.mcp.searched && s.mcp.found.is_empty() {
        "The MCP Registry lists no server by that name. If you know its URL or the command that starts it, use Add your own server above.".to_owned()
    } else if s.mcp.searched {
        "From the MCP Registry".to_owned()
    } else {
        String::new()
    };
    let mut results = div().flex().flex_col().gap(u(12.));
    for (i, c) in s.mcp.found.iter().enumerate() {
        results = results.child(mcp_result(s, th, i, c, cx));
    }
    let body = div()
        .flex()
        .flex_col()
        .gap(u(20.))
        .child(intro)
        .when_some(crate::mcp::tool_warning(on_tools), |d, w| {
            d.child(notice(th, w))
        })
        .child(mine)
        .child(mcp_custom(s, th, cx))
        .child(
            div()
                .flex()
                .items_center()
                .gap(u(10.))
                .child(field(th, s.mcpq_in.clone()))
                .child(button(
                    th,
                    "mcp-search",
                    "Search",
                    Btn::Outline,
                    click(cx, |s, cx| s.mcp_search(false, cx)),
                )),
        )
        .when(!status.is_empty(), |d| {
            d.child(
                eyebrow_box(th)
                    .text_color(th.ink3)
                    .child(s.sel.label(PAGE, status.to_uppercase())),
            )
        })
        .child(results)
        .when(s.mcp.cursor.is_some(), |d| {
            d.child(button(
                th,
                "mcp-more",
                "Show more",
                Btn::Ghost,
                click(cx, |s, cx| s.mcp_search(true, cx)),
            ))
        });
    selects(page(body, "mcp"), cx)
}

fn mcp_server_row(s: &Ecl, th: &Theme, srv: &McpServerStatus, cx: &mut Context<Ecl>) -> Div {
    let busy = s.mcp.busy.contains(&srv.id);
    let (label, color) = match srv.status.as_str() {
        _ if busy => ("Connecting…".to_owned(), th.ink3),
        "connecting" => ("Connecting…".to_owned(), th.ink3),
        "connected" => (format!("On · {} tools", srv.tools.len()), th.keep),
        "needs-sign-in" => ("Needs sign-in".to_owned(), th.ember),
        "error" => ("Could not connect".to_owned(), th.lose),
        "off" => ("Off".to_owned(), th.ink3),
        other => (other.to_owned(), th.ink3),
    };
    // Every env and header value is masked already; name them all as secrets for the line.
    let secrets: Vec<String> = match &srv.transport {
        McpTransport::Stdio { env, .. } => env.keys().cloned().collect(),
        McpTransport::Http { headers, .. } => headers.keys().cloned().collect(),
    };
    let what = crate::mcp::describe(&srv.transport, &secrets);
    // A program's start may wait minutes (npx and uvx install it first): said, so "Connecting…"
    // does not read as stuck.
    let installs = (busy || srv.status == "connecting")
        && matches!(&srv.transport, McpTransport::Stdio { .. });
    let id = srv.id.clone();
    let mut buttons = div().flex().flex_wrap().items_center().gap(u(8.));
    if srv.can_sign_in && srv.status == "needs-sign-in" && !s.mcp.signing.contains_key(&id) {
        let i = id.clone();
        buttons = buttons.child(button(
            th,
            format!("mcp-in-{id}"),
            "Sign in",
            Btn::Primary,
            click(cx, move |s, cx| s.mcp_sign_in(i.clone(), cx)),
        ));
    }
    if srv.signed_in {
        let i = id.clone();
        buttons = buttons.child(button(
            th,
            format!("mcp-out-{id}"),
            "Sign out",
            Btn::Outline,
            click(cx, move |s, cx| s.mcp_sign_out(i.clone(), cx)),
        ));
    }
    let shown = s.mcp.tested.contains_key(&id);
    let i = id.clone();
    buttons = buttons.child(button(
        th,
        format!("mcp-tools-{id}"),
        if shown { "Hide tools" } else { "Show tools" },
        Btn::Ghost,
        click(cx, move |s, cx| s.mcp_test(i.clone(), cx)),
    ));
    // A server labels its own tools read-only: workflows that run unattended call them only
    // once the user trusts it, after reading what that unlocks (`mcp_trust_panel`, which has its
    // own Trust button, so this one steps aside while it is open).
    let (i, trusted) = (id.clone(), srv.trusted);
    let asking = s.mcp.trusting.as_deref() == Some(id.as_str());
    buttons = buttons.when(!asking, |b| {
        b.child(button(
            th,
            format!("mcp-trust-{id}"),
            if trusted {
                "Untrust in tasks"
            } else {
                "Trust in tasks"
            },
            Btn::Ghost,
            click(cx, move |s, cx| match trusted {
                true => s.mcp_set_trusted(i.clone(), false, cx),
                false => s.mcp.trusting = Some(i.clone()),
            }),
        ))
    });
    let (i, on) = (id.clone(), srv.enabled);
    buttons = buttons.child(button(
        th,
        format!("mcp-toggle-{id}"),
        if on { "Turn off" } else { "Turn on" },
        if on { Btn::Outline } else { Btn::Primary },
        click(cx, move |s, cx| s.mcp_set_enabled(i.clone(), !on, cx)),
    ));
    // Remove deletes the server's saved sign-in too, and sat beside Turn off: as a chat's Delete,
    // it takes a second click.
    let (i, key) = (id.clone(), format!("mcp:{id}"));
    let armed = s.armed.as_deref() == Some(key.as_str());
    buttons = buttons.child(button(
        th,
        format!("mcp-rm-{id}"),
        if armed {
            "Click again to remove"
        } else {
            "Remove"
        },
        Btn::Danger,
        click(cx, move |s, cx| {
            if s.armed.as_deref() != Some(key.as_str()) {
                s.armed = Some(key.clone());
                return;
            }
            s.armed = None;
            s.mcp_remove(i.clone(), cx)
        }),
    ));
    // Built where it shows, after the lines above it: a page's text registers in reading order.
    let tools = || -> Option<Div> {
        s.mcp.tested.get(&id).map(|t| match t {
            Err(e) => div()
                .text_size(u(13.))
                .text_color(th.lose)
                .child(s.sel.label(PAGE, e.clone())),
            Ok(list) if list.is_empty() => div()
                .text_size(u(13.))
                .text_color(th.ink2)
                .child("No tools."),
            Ok(list) => div()
                .flex()
                .flex_col()
                .gap(u(6.))
                .children(list.iter().map(|t| {
                    // One no chat takes says why in place of what it does.
                    let (kind, text) = match (&t.left_out, t.read_only) {
                        (Some(why), _) => {
                            ("chats leave it out", format!("Chats leave it out: {why}."))
                        }
                        (None, true) => ("reads only", logic::clip(&t.description, 160)),
                        (None, false) => ("asks first", logic::clip(&t.description, 160)),
                    };
                    div()
                        .flex()
                        .gap(u(10.))
                        .text_size(u(13.))
                        .child(
                            mono_box().text_color(th.ink).child(
                                s.sel
                                    .label(PAGE, larch_app::contract::mcp_tool_name(&id, &t.name)),
                            ),
                        )
                        .child(tag(th, kind))
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .text_color(th.ink2)
                                .child(s.sel.label(PAGE, text)),
                        )
                })),
        })
    };
    div()
        .flex()
        .flex_col()
        .gap(u(8.))
        .pt(u(12.))
        .border_t_1()
        .border_color(th.line)
        .child(
            div()
                .flex()
                .items_center()
                .gap(u(12.))
                .child(
                    div()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(s.sel.label(PAGE, srv.name.clone())),
                )
                .child(
                    div()
                        .text_size(u(13.))
                        .text_color(color)
                        .child(s.sel.label(PAGE, label)),
                ),
        )
        .child(
            mono_box()
                .text_color(th.ink3)
                .child(s.sel.label(PAGE, what)),
        )
        .when(installs, |d| {
            let note = "A first start can take a few minutes while npx or uvx installs it.";
            d.child(
                div()
                    .text_size(u(13.))
                    .text_color(th.ink2)
                    .child(s.sel.label(PAGE, note)),
            )
        })
        .when_some(
            srv.error
                .clone()
                .filter(|_| srv.status == "error" || srv.status == "needs-sign-in"),
            |d, e| {
                d.child(
                    div()
                        .text_size(u(13.))
                        .text_color(th.ink2)
                        .child(s.sel.label(PAGE, logic::capital(&e))),
                )
            },
        )
        .when_some(s.mcp.signing.get(&id), |d, (r, opened)| {
            d.child(mcp_sign_panel(s, th, &id, r, *opened, cx))
        })
        .when(s.mcp.trusting.as_deref() == Some(id.as_str()), |d| {
            d.child(mcp_trust_panel(th, srv, cx))
        })
        .child(buttons)
        .when_some(tools(), |d, t| d.child(t))
}

/// What trusting a server unlocks, before it does: its read tools, which workflows call without
/// the user, on the server's own word that they only read.
fn mcp_trust_panel(th: &Theme, srv: &McpServerStatus, cx: &mut Context<Ecl>) -> Div {
    let reads: Vec<String> = (srv.tools.iter())
        .filter(|t| t.read_only)
        .map(|t| larch_app::contract::mcp_tool_name(&srv.id, &t.name))
        .collect();
    let listed = match reads.is_empty() {
        true => "It lists no read tools yet; connect it to see them.".to_owned(),
        false => format!("Its read tools now: {}.", reads.join(", ")),
    };
    let (yes, no) = (srv.id.clone(), srv.id.clone());
    div()
        .flex()
        .flex_col()
        .gap(u(8.))
        .p(u(12.))
        .rounded(u(8.))
        .bg(th.sunk)
        .child(div().text_size(u(13.)).text_color(th.ink).child(format!(
            "Tasks run without you. Trusted, {} lets their steps call every tool it labels read-only, with arguments the model writes, including tools it adds later. Trust it only if you believe its labels.",
            srv.name
        )))
        .child(mono(listed).text_color(th.ink2))
        .child(
            div()
                .flex()
                .gap(u(8.))
                .child(button(
                    th,
                    format!("mcp-trust-yes-{}", srv.id),
                    "Trust in tasks",
                    Btn::Primary,
                    click(cx, move |s, cx| {
                        s.mcp.trusting = None;
                        s.mcp_set_trusted(yes.clone(), true, cx);
                    }),
                ))
                .child(button(
                    th,
                    format!("mcp-trust-no-{}", srv.id),
                    "Cancel",
                    Btn::Ghost,
                    click(cx, move |s, _| {
                        if s.mcp.trusting.as_deref() == Some(no.as_str()) {
                            s.mcp.trusting = None;
                        }
                    }),
                )),
        )
}

/// A sign-in in progress: first where it goes (the site's host); once the browser is open, a
/// place for the address it ended on, if its return could not reach Larch.
fn mcp_sign_panel(
    s: &Ecl,
    th: &Theme,
    id: &str,
    r: &McpSignIn,
    opened: bool,
    cx: &mut Context<Ecl>,
) -> Div {
    let (a, b, c) = (id.to_owned(), id.to_owned(), id.to_owned());
    let host = r.host.clone();
    let mut d = notice(th, if opened {
        format!("Finish signing in at {host} in your browser.")
    } else {
        format!("Sign in at {host} in your browser? Larch gets a token for this server only; your password stays with {host}.")
    })
    .flex()
    .flex_col()
    .gap(u(10.));
    if !opened {
        d = d.child(
            div()
                .flex()
                .gap(u(8.))
                .child(button(
                    th,
                    format!("mcp-open-{id}"),
                    "Open sign-in page",
                    Btn::Primary,
                    click(cx, move |s, cx| s.mcp_open_sign(&a, cx)),
                ))
                .child(button(
                    th,
                    format!("mcp-cancel-{id}"),
                    "Cancel",
                    Btn::Ghost,
                    click(cx, move |s, cx| s.mcp_cancel_sign_in(b.clone(), cx)),
                )),
        );
    } else {
        let again = id.to_owned();
        d = d
            .child(div().text_size(u(13.)).child("If the browser ends on an error page instead of “Signed in”, paste its address here:"))
            .child(div().flex().gap(u(8.)).child(field(th, s.mcp_paste_in.clone())).child(button(
                th,
                format!("mcp-finish-{id}"),
                "Finish",
                Btn::Outline,
                click(cx, move |s, cx| s.mcp_finish_sign(c.clone(), cx)),
            )))
            .child(div().flex().gap(u(8.)).child(button(th, format!("mcp-again-{id}"), "Open the page again", Btn::Ghost, click(cx, move |s, cx| s.mcp_open_sign(&again, cx)))).child(button(
                th,
                format!("mcp-cancel2-{id}"),
                "Cancel",
                Btn::Ghost,
                click(cx, move |s, cx| s.mcp_cancel_sign_in(b.clone(), cx)),
            )));
    }
    d
}

/// "Add your own server": a name, and the command that starts it or its URL.
fn mcp_custom(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let open = s.mcp.custom_open;
    let mut c = card(th).p(u(20.)).gap(u(10.)).child(
        div()
            .flex()
            .items_center()
            .child(eyebrow(th, "Add your own server").flex_1())
            .child(button(
                th,
                "mcp-custom-toggle",
                if open { "Close" } else { "Add your own" },
                Btn::Ghost,
                click(cx, |s, _| s.mcp.custom_open = !s.mcp.custom_open),
            )),
    );
    if open {
        let trust = s.mcp.custom_trust;
        c =
            c.child(div().text_size(u(13.)).text_color(th.ink2).child("Name"))
                .child(div().flex().child(field(th, s.mcp_name_in.clone())))
                .child(
                    div()
                        .text_size(u(13.))
                        .text_color(th.ink2)
                        .child("Command that starts it, or its URL"),
                )
                .child(div().flex().child(field(th, s.mcp_target_in.clone())))
                .child(pill(
                    th,
                    "mcp-custom-trust",
                    if trust {
                        "✓ I trust this server"
                    } else {
                        "I trust this server"
                    },
                    trust,
                    click(cx, |s, _| s.mcp.custom_trust = !s.mcp.custom_trust),
                ))
                .child(
                    div().text_size(u(12.5)).text_color(th.ink3).child(
                        "A command runs as a program on this computer with your user's rights.",
                    ),
                )
                .child(div().flex().child(button(
                    th,
                    "mcp-custom-add",
                    "Add",
                    Btn::Primary,
                    click(cx, |s, cx| s.mcp_add_custom(cx)),
                )));
    }
    c
}

/// One registry server: what it is, where it runs, and its install form when open.
fn mcp_result(s: &Ecl, th: &Theme, i: usize, c: &crate::mcp::Card, cx: &mut Context<Ecl>) -> Div {
    use crate::mcp::OptionKind;
    let opt = c.option().or(c.options.first());
    let where_ = match opt.map(|o| &o.kind) {
        Some(OptionKind::Remote { .. }) => "Remote · nothing runs on this computer".to_owned(),
        Some(OptionKind::Local { needs, .. }) => format!("Runs on this computer · needs {needs}"),
        None => "Nothing to install".to_owned(),
    };
    let open = s.mcp.open == Some(i);
    let mut tags = div().flex().flex_wrap().items_center().gap(u(8.));
    if c.suggested {
        tags = tags.child(tag(th, "Works without a sign-in"));
    }
    tags = tags.child(tag(th, where_));
    if !c.installable {
        tags = tags.child(tag(th, opt.map_or_else(String::new, |o| o.why.clone())));
    }
    if c.installable && !open {
        tags = tags.child(button(
            th,
            format!("mcp-install-{i}"),
            "Install",
            Btn::Primary,
            click(cx, move |s, cx| s.mcp_open(i, cx)),
        ));
    }
    if let Some(web) = logic::web_url(&c.website) {
        tags = tags.child(button(
            th,
            format!("mcp-web-{i}"),
            "Website",
            Btn::Ghost,
            click(cx, move |_, cx| cx.open_url(&web)),
        ));
    }
    let mut card_ = card(th)
        .p(u(20.))
        .gap(u(8.))
        .child(
            div()
                .flex()
                .items_center()
                .gap(u(10.))
                .child(
                    div()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(c.title.clone()),
                )
                .child(div().text_size(u(13.)).text_color(th.ink3).child(
                    if c.version.is_empty() {
                        c.publisher.clone()
                    } else {
                        format!("{} · {}", c.publisher, c.version)
                    },
                )),
        )
        .child(
            div()
                .text_size(u(13.5))
                .text_color(th.ink2)
                .child(c.description.clone()),
        )
        .child(tags);
    if open && let Some(o) = c.option() {
        let answers = s.mcp_answers(cx);
        let secrets: Vec<String> = o
            .inputs
            .iter()
            .filter(|x| x.secret)
            .map(|x| x.name.clone())
            .collect();
        let what = match crate::mcp::plan(c, o, &answers, &[]) {
            Ok(server) => crate::mcp::describe(&server.transport, &secrets),
            Err(e) => e.join(" "),
        };
        for (inp, (_, f)) in o.inputs.iter().zip(&s.mcp.answers) {
            card_ = card_
                .child(div().text_size(u(13.)).text_color(th.ink2).child(format!(
                    "{}{}",
                    inp.name,
                    if inp.required { " (required)" } else { "" }
                )))
                .child(div().flex().child(field(th, f.clone())));
        }
        let trust = s.mcp.trust;
        let local = matches!(o.kind, OptionKind::Local { .. });
        card_ = card_
            .child(
                div()
                    .rounded(u(10.))
                    .bg(th.code_bg)
                    .text_color(th.code_ink)
                    .px(u(14.))
                    .py(u(10.))
                    .child(mono(what)),
            )
            .child(pill(
                th,
                format!("mcp-trust-{i}"),
                if trust { "✓ I trust this server" } else { "I trust this server" },
                trust,
                click(cx, |s, _| s.mcp.trust = !s.mcp.trust),
            ))
            .child(div().text_size(u(12.5)).text_color(th.ink3).child(if local {
                "Its tools can read what the chat sends them, and it runs as a program on this computer with your user's rights."
            } else {
                "Its tools can read what the chat sends them."
            }))
            .child(
                div()
                    .flex()
                    .gap(u(8.))
                    .child(button(th, format!("mcp-do-{i}"), "Install", Btn::Primary, click(cx, |s, cx| s.mcp_install(cx))))
                    .child(button(th, format!("mcp-x-{i}"), "Cancel", Btn::Ghost, click(cx, |s, _| s.mcp_close()))),
            );
    }
    card_
}

fn files_el(s: &Ecl, th: &Theme, repo: &str, cx: &mut Context<Ecl>) -> Div {
    let th2 = *th;
    let list = match s.hf.files.get(repo) {
        None => {
            return div()
                .pb(u(14.))
                .text_size(u(13.))
                .text_color(th.ink3)
                .child("Listing its files…");
        }
        Some(Err(e)) => {
            return div()
                .pb(u(14.))
                .text_size(u(13.))
                .text_color(th.lose)
                .child(e.clone());
        }
        Some(Ok(f)) => f,
    };
    div()
        .flex()
        .flex_col()
        .gap(u(4.))
        .pb(u(14.))
        .pl(u(16.))
        .children(list.iter().map(|f| {
            let (rp, file) = (repo.to_owned(), f.file.clone());
            div()
                .flex()
                .items_center()
                .gap(u(14.))
                .py(u(4.))
                .text_size(u(13.))
                .child(
                    mono(f.file.clone())
                        .flex_1()
                        .min_w_0()
                        .truncate()
                        .text_color(th2.ink2),
                )
                .child(div().w(u(90.)).child(f.quant.clone()))
                .child(
                    div()
                        .w(u(80.))
                        .text_color(th2.ink3)
                        .child(logic::gb(f.size_bytes)),
                )
                .child(
                    div()
                        .w(u(70.))
                        .text_color(th2.ink3)
                        .child(if f.parts.len() > 1 {
                            format!("{} parts", f.parts.len())
                        } else {
                            String::new()
                        }),
                )
                // Its vision projector downloads with it (and counts in its size).
                .child(
                    div()
                        .w(u(100.))
                        .text_color(th2.ink3)
                        .child(if f.mmproj.is_some() {
                            "reads images"
                        } else {
                            ""
                        }),
                )
                .child(
                    div()
                        .w(u(120.))
                        .flex()
                        .justify_end()
                        .child(if f.downloaded {
                            div()
                                .text_color(th2.ink3)
                                .child("In library")
                                .into_any_element()
                        } else {
                            button(
                                &th2,
                                format!("dlf-{repo}-{}", f.file),
                                "Download",
                                Btn::Ghost,
                                cx.listener(move |s, _, _, cx| {
                                    cx.stop_propagation();
                                    s.download(rp.clone(), file.clone(), false, cx)
                                }),
                            )
                            .into_any_element()
                        }),
                )
        }))
}

/// "40 s ago" / "6 min ago" for a sample `ms` before now.
fn ago(ms: u64) -> String {
    let s = ms / 1000;
    if s < 60 {
        format!("{s} s ago")
    } else {
        format!("{} min ago", s / 60)
    }
}

/// The last 10 minutes of tok/s, with its peak marked and, under the pointer, the nearest sample.
fn chart(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> impl IntoElement {
    const H: f32 = 200.;
    const SPAN: f32 = 600_000.;
    let pts: Vec<(u64, f64)> = s.samples.iter().copied().collect();
    let th = *th;
    let now = now_ms();
    let frac = move |t: u64| (1. - (now.saturating_sub(t) as f32 / SPAN)).clamp(0., 1.);
    let top = pts.iter().map(|p| p.1).fold(0., f64::max).max(1.) * 1.3;
    let peak = pts
        .iter()
        .copied()
        .filter(|p| p.1 > 0.)
        .max_by(|a, b| a.1.total_cmp(&b.1));
    // The sample nearest the pointer's time.
    let hover = s.chart_at.and_then(|f| {
        let t = now as f64 - (1. - f64::from(f)) * f64::from(SPAN);
        pts.iter()
            .copied()
            .min_by(|a, b| (a.0 as f64 - t).abs().total_cmp(&(b.0 as f64 - t).abs()))
    });
    // A readout at a sample: above its point, kept inside the chart at the right edge.
    let readout = move |(t, v): (u64, f64), text: String, strong: bool| {
        let f = frac(t);
        let y = (H * (1. - (v / top) as f32) - 30.).max(0.);
        div()
            .absolute()
            .top(u(y))
            .when(f <= 0.75, |d| d.left(gpui::relative(f)).ml(u(8.)))
            .when(f > 0.75, |d| d.right(gpui::relative(1. - f)).mr(u(8.)))
            .px(u(8.))
            .py(u(3.))
            .rounded(u(6.))
            .text_size(u(12.5))
            .whitespace_nowrap()
            .when(strong, |d| {
                d.bg(th.card)
                    .border_1()
                    .border_color(th.line)
                    .text_color(th.ink)
                    .font_weight(FontWeight::SEMIBOLD)
            })
            .when(!strong, |d| d.text_color(th.ink2))
            .child(text)
    };
    let boxed = s.chart_box.clone();
    div()
        .id("throughput")
        .relative()
        .h(u(H))
        .w_full()
        .on_mouse_move(cx.listener(|s, e: &MouseMoveEvent, _, cx| {
            let at = s
                .chart_box
                .get()
                .filter(|b| b.contains(&e.position))
                .map(|b| {
                    (f32::from(e.position.x - b.origin.x) / f32::from(b.size.width)).clamp(0., 1.)
                });
            if at != s.chart_at {
                s.chart_at = at;
                cx.notify();
            }
        }))
        .on_hover(cx.listener(|s, on: &bool, _, cx| {
            if !*on {
                s.chart_at = None;
                cx.notify();
            }
        }))
        .child(
            canvas(
                move |b, _, _| boxed.set(Some(b)),
                move |b: Bounds<Pixels>, _, win: &mut Window, _: &mut App| {
                    let w = f32::from(b.size.width);
                    let h = f32::from(b.size.height);
                    let x = |t: u64| b.origin.x + px(frac(t) * w);
                    let y = |v: f64| b.origin.y + px(h - (v / top) as f32 * h);
                    for k in 1..4 {
                        let gy = b.origin.y + px(h * k as f32 / 4.);
                        let mut pb = PathBuilder::stroke(px(1.));
                        pb.move_to(point(b.origin.x, gy));
                        pb.line_to(point(b.origin.x + px(w), gy));
                        if let Ok(p) = pb.build() {
                            win.paint_path(p, fade(th.line, 1.));
                        }
                    }
                    if pts.len() < 2 {
                        return;
                    }
                    let mut area = PathBuilder::fill();
                    area.move_to(point(x(pts[0].0), b.origin.y + px(h)));
                    let mut line = PathBuilder::stroke(px(2.));
                    line.move_to(point(x(pts[0].0), y(pts[0].1)));
                    for p in &pts {
                        area.line_to(point(x(p.0), y(p.1)));
                        line.line_to(point(x(p.0), y(p.1)));
                    }
                    area.line_to(point(x(pts[pts.len() - 1].0), b.origin.y + px(h)));
                    area.close();
                    if let Ok(p) = area.build() {
                        win.paint_path(
                            p,
                            linear_gradient(
                                180.,
                                linear_color_stop(fade(th.sun, 0.32), 0.),
                                linear_color_stop(fade(th.sun, 0.), 1.),
                            ),
                        );
                    }
                    if let Ok(p) = line.build() {
                        win.paint_path(p, fade(th.sun, 1.));
                    }
                    let dot = |win: &mut Window, (t, v): (u64, f64), r: f32| {
                        let c = point(x(t) - px(r), y(v) - px(r));
                        win.paint_quad(
                            fill(Bounds::new(c, size(px(2. * r), px(2. * r))), th.sun)
                                .corner_radii(px(r)),
                        );
                    };
                    if let Some(p) = hover {
                        let mut pb = PathBuilder::stroke(px(1.));
                        pb.move_to(point(x(p.0), b.origin.y));
                        pb.line_to(point(x(p.0), b.origin.y + px(h)));
                        if let Ok(path) = pb.build() {
                            win.paint_path(path, fade(th.ink3, 0.6));
                        }
                        dot(win, p, 4.);
                    }
                    if let Some(p) = peak {
                        dot(win, p, 3.);
                    }
                },
            )
            .size_full(),
        )
        // The peak stays labelled unless the pointer's readout is on it.
        .children(
            peak.filter(|p| hover != Some(*p))
                .map(|p| readout(p, format!("Peak {:.0} tok/s", p.1), false)),
        )
        .children(hover.map(|p| {
            let tag = if Some(p) == peak { "Peak " } else { "" };
            readout(
                p,
                format!("{tag}{:.0} tok/s · {}", p.1, ago(now.saturating_sub(p.0))),
                true,
            )
        }))
}

fn kv(th: &Theme, k: &str, v: String) -> Div {
    div()
        .flex()
        .gap(u(16.))
        .text_size(u(14.))
        .child(
            div()
                .w(u(110.))
                .flex_none()
                .text_color(th.ink3)
                .child(k.to_owned()),
        )
        .child(v)
}

/// Engine: throughput, GPU memory and the live settings.
pub fn engine(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Stateful<Div> {
    let th2 = *th;
    let sm = s.last.as_ref();
    let tok = sm.map_or(0., |x| x.tok_s);
    let throughput = card(th)
        .flex_1()
        .min_w_0()
        .p(u(28.))
        .child(eyebrow(th, "Throughput · 10 min"))
        // Number and queue wrap under each other in a narrow card instead of running out of it.
        .child(
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .gap_x(u(28.))
                .gap_y(u(12.))
                .child(
                    div()
                        .flex()
                        .items_end()
                        .gap(u(6.))
                        .child(serif(format!("{tok:.0}"), 62.))
                        .child(serif("tok/s", 26.).text_color(th.ink2).pb(u(6.))),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap(u(6.))
                        .text_size(u(14.))
                        .child(
                            div()
                                .flex()
                                .gap(u(16.))
                                .child(div().w(u(60.)).text_color(th.ink3).child("Running"))
                                .child(sm.map_or("—".into(), |x| {
                                    format!(
                                        "{} chat{}",
                                        x.running,
                                        if x.running == 1 { "" } else { "s" }
                                    )
                                })),
                        )
                        .child(
                            div()
                                .flex()
                                .gap(u(16.))
                                .child(div().w(u(60.)).text_color(th.ink3).child("Waiting"))
                                .child(sm.map_or("—".into(), |x| {
                                    format!("{} · {:.0} ms", x.waiting, x.queue_wait_ms)
                                })),
                        ),
                ),
        )
        .child(chart(s, th, cx))
        .child(
            div()
                .flex()
                .justify_between()
                .text_size(u(13.))
                .text_color(th.ink3)
                .child("10 min ago")
                .child("now"),
        );
    let total = s.gpu_bytes();
    let (w, kvb) = sm.map_or((0, 0), |x| (x.weights_bytes, x.kv_total_bytes));
    let used = sm.and_then(|x| x.gpu_used_bytes).unwrap_or(w + kvb);
    // Beside a 500 wide memory card, a narrow page left the throughput a sliver ("10 min agonow"):
    // there the two stack.
    let stack = s.room - PAGE_X * 2. < ENGINE_ROW;
    let memory = card(th)
        .when(!stack, |d| d.w(u(500.)).flex_none())
        .p(u(28.))
        .child(eyebrow(
            th,
            if s.st.device.is_empty() {
                "Memory".to_owned()
            } else {
                format!("Memory · {}", s.st.device)
            },
        ))
        .child(serif(
            total.map_or_else(
                || logic::gb(used),
                |t| {
                    format!(
                        "{} of {}",
                        logic::gb(used).trim_end_matches(" GB"),
                        logic::gb(t).replace(".0 GB", " GB")
                    )
                },
            ),
            30.,
        ))
        .when_some(total, |d, t| {
            d.child(bar(
                th,
                &[
                    (w as f32 / t as f32, th2.accent.into()),
                    (kvb as f32 / t as f32, th2.sun.into()),
                ],
                10.,
            ))
            .child(
                div()
                    .flex()
                    .gap(u(18.))
                    .text_size(u(13.))
                    .text_color(th2.ink2)
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(u(6.))
                            .child(div().size(u(9.)).rounded(u(2.)).bg(th2.accent))
                            .child(format!("Weights {}", logic::gb(w))),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(u(6.))
                            .child(div().size(u(9.)).rounded(u(2.)).bg(th2.sun))
                            .child(format!("Chat memory {}", logic::gb(kvb))),
                    ),
            )
        })
        .child(
            div()
                .flex()
                .flex_col()
                .gap(u(10.))
                .pt(u(10.))
                .children(sm.map(|x| {
                    kv(
                        &th2,
                        "Chat memory",
                        format!(
                            "{} of {} in use",
                            logic::gb(x.kv_used_bytes),
                            logic::gb(x.kv_total_bytes)
                        ),
                    )
                }))
                .children(sm.map(|x| {
                    kv(
                        &th2,
                        "Power",
                        x.power_cap.map_or("Full".into(), |p| {
                            format!("{:.0} W, held below {:.0} W", p.limit_w, p.full_w)
                        }),
                    )
                }))
                .children(sm.map(|x| kv(&th2, "Queue wait", format!("{:.0} ms", x.queue_wait_ms))))
                .when(sm.is_none(), |d| {
                    d.child(
                        div()
                            .text_color(th2.ink2)
                            .child("Load a model to see the engine's numbers."),
                    )
                }),
        );
    page(
        div()
            .flex()
            .flex_col()
            .gap(u(20.))
            .child(
                div()
                    .flex()
                    .when(stack, |d| d.flex_col())
                    .gap(u(20.))
                    .child(throughput)
                    .child(memory),
            )
            .child(live_settings(s, th, cx)),
        "engine",
    )
}

/// The width, in `u`, the throughput card and the 500 wide memory card need side by side.
const ENGINE_ROW: f32 = 940.;
/// The page width, in `u`, a Discover row needs for its name beside its four columns.
const DISCOVER_ROW: f32 = 1000.;

fn live_settings(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let th2 = *th;
    card(th)
        .flex_1()
        .min_w_0()
        .p(u(28.))
        .gap(u(0.))
        .child(eyebrow(th, "Live settings").pb(u(12.)))
        .when(s.knobs.is_empty(), |d| d.child(div().pt(u(8.)).text_color(th2.ink2).child("The settings show once a model is loaded.")))
        .children(s.knobs.iter().map(|k| {
            let (last, undo) = s.ledger.last(&k.name);
            let source = match last {
                Some(c) => format!(
                    "{} · {}",
                    match c.cause.as_str() {
                        "human" => "you",
                        "gait" => "by tuning",
                        "guardrail" => "a guardrail",
                        "inverse" => "undone",
                        c => c,
                    },
                    logic::ago(c.at, now_ms())
                ),
                None if k.value == k.default => "default".into(),
                None => "set before".into(),
            };
            let editing = s.editing_knob.as_deref() == Some(k.name.as_str());
            let (name, v) = (k.name.clone(), k.value);
            div()
                .flex()
                .items_center()
                .gap(u(12.))
                .min_h(u(54.))
                .border_t_1()
                .border_color(th2.line)
                .child(mono(k.name.clone()).text_size(u(14.)).flex_1().min_w_0().truncate())
                .child(div().w(u(140.)).flex_none().text_size(u(13.)).text_color(th2.ink3).truncate().child(source))
                .child(if editing {
                    div().w(u(90.)).flex_none().child(field(&th2, s.knob_in.clone()).py(u(5.)).font_family("JetBrains Mono")).into_any_element()
                } else {
                    div()
                        .id(gpui::ElementId::Name(format!("kv-{}", k.name).into()))
                        .w(u(70.))
                        .flex_none()
                        .flex()
                        .justify_end()
                        .cursor_pointer()
                        .on_click(cx.listener(move |s, _, w, cx| {
                            s.editing_knob = Some(name.clone());
                            s.knob_in.update(cx, |i, cx| i.set_text(&logic::fmt_v(v), cx));
                            w.focus(&gpui::Focusable::focus_handle(&s.knob_in, cx));
                            cx.notify()
                        }))
                        .child(mono(logic::fmt_v(k.value)).text_size(u(14.)))
                        .into_any_element()
                })
                .child(div().w(u(64.)).flex_none().flex().justify_end().when_some(undo.zip(last.map(|c| c.engine)), |d, (rev, made)| {
                    d.child(button(&th2, format!("undo-{}", k.name), "Undo", Btn::Ghost, click(cx, move |s, cx| {
                        let engine = s.ledger.engine;
                        s.call(cx, move |c| c.knob_revert(made, rev), move |s, r, cx| {
                            match r {
                                KnobResult::Applied { .. } => {
                                    if s.ledger.undone(engine, rev) {
                                        s.say("Undone.")
                                    }
                                }
                                KnobResult::Pending { reason } => s.say(format!("Queued: {reason}")),
                                KnobResult::Rejected { reason } => s.say(format!("Not undone: {reason}")),
                            }
                            s.load_knobs(cx)
                        })
                    })))
                }))
        }))
        .when(!s.knobs.is_empty(), |d| d.child(div().pt(u(12.)).border_t_1().border_color(th2.line).text_size(u(12.5)).text_color(th2.ink3).child("Click a value to set it. Every change is logged with who made it, and Undo puts it back.")))
}

/// Settings > Memory: what the assistant keeps about you, each fact editable and forgettable.
fn memory(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let th2 = *th;
    let on = s.mem_on;
    let head = div()
        .flex()
        .items_end()
        .gap(u(16.))
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .gap(u(6.))
                .child(serif("What I remember", 40.))
                .child(div().text_size(u(14.)).text_color(th.ink2).child(
                    "Short facts the assistant uses in every chat. They stay on this computer.",
                )),
        )
        .child(seg(
            th,
            "mem-on",
            &["On", "Off"],
            if on { 0 } else { 1 },
            set(cx, |s, i, cx| {
                let on = i == 0;
                s.call(
                    cx,
                    move |c| c.memory_set_enabled(on),
                    move |s, _, _| s.mem_on = on,
                )
            }),
        ));
    let add = div()
        .flex()
        .gap(u(10.))
        .child(field(th, s.fact_in.clone()))
        .child(button(
            th,
            "remember",
            "Remember",
            Btn::Primary,
            click(cx, |s, cx| {
                s.fact_in
                    .update(cx, |_, cx| cx.emit(crate::input::Submitted));
            }),
        ));
    let list =
        card(th)
            .p(u(8.))
            .gap(u(0.))
            .when(s.facts.is_empty(), |d| {
                d.p(u(24.)).child(
                    div().text_color(th2.ink2).child(
                        "Nothing yet. Tell the assistant something about you, or add it here.",
                    ),
                )
            })
            .children(s.facts.iter().enumerate().map(|(i, f)| {
                let editing = s.editing.as_ref().is_some_and(|(slug, _)| slug == &f.slug);
                let (slug, text, forget) = (f.slug.clone(), f.text.clone(), f.slug.clone());
                div()
                    .flex()
                    .items_center()
                    .gap(u(14.))
                    .px(u(16.))
                    .py(u(12.))
                    .when(i > 0, |d| d.border_t_1().border_color(th2.line))
                    .child(icon("memory", 15., th2.ink3))
                    .child(if editing {
                        div()
                            .flex_1()
                            .flex()
                            .gap(u(8.))
                            .child(field(&th2, s.edit_in.clone()))
                            .child(button(
                                &th2,
                                "save-edit",
                                "Save",
                                Btn::Primary,
                                click(cx, |s, cx| {
                                    s.edit_in
                                        .update(cx, |_, cx| cx.emit(crate::input::Submitted))
                                }),
                            ))
                            .child(button(
                                &th2,
                                "cancel-edit",
                                "Cancel",
                                Btn::Ghost,
                                click(cx, |s, _| s.editing = None),
                            ))
                    } else {
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .flex_col()
                            .gap(u(2.))
                            .child(
                                div().text_size(u(14.5)).child(
                                    s.sel.label(PAGE, logic::fact_sentence(&f.text).to_owned()),
                                ),
                            )
                            .child(div().text_size(u(12.5)).text_color(th2.ink3).child(
                                s.sel.label(
                                    PAGE,
                                    format!(
                                        "{} · {}",
                                        // Not the chat's id ("chat 1794596154625199300-25988
                                        // turn 2"), which says nothing to the user.
                                        logic::fact_source(&f.source, |id| {
                                            (s.chats.iter().find(|c| c.id == id))
                                                .map(|c| c.title.clone())
                                        }),
                                        logic::ago(f.updated_at, now_ms())
                                    ),
                                ),
                            ))
                    })
                    .when(!editing, |d| {
                        d.child(button(
                            &th2,
                            format!("edit-{}", f.slug),
                            "Edit",
                            Btn::Ghost,
                            cx.listener(move |s, _, w, cx| {
                                s.editing = Some((slug.clone(), text.clone()));
                                s.edit_in.update(cx, |i, cx| {
                                    i.set_text(logic::fact_sentence(&text), cx)
                                });
                                w.focus(&gpui::Focusable::focus_handle(&s.edit_in, cx));
                                cx.notify()
                            }),
                        ))
                        .child(button(
                            &th2,
                            format!("forget-{}", f.slug),
                            "Forget",
                            Btn::Ghost,
                            click(cx, move |s, cx| {
                                let slug = forget.clone();
                                s.call(
                                    cx,
                                    move |c| c.memory_forget(&slug),
                                    |s, g, cx| {
                                        s.say_act("Forgotten.", "Undo", Act::Restore(g));
                                        s.load_memory(cx)
                                    },
                                )
                            }),
                        ))
                    })
            }));
    div()
        .max_w(u(980.))
        .flex()
        .flex_col()
        .gap(u(20.))
        .child(head)
        .when(on, |d| d.child(add))
        .child(list)
}

const SNIPS: [&str; 7] = [
    "Start",
    "curl",
    "PowerShell",
    "Python",
    "Anthropic SDK",
    "Claude Code",
    "MCP",
];

/// Connect apps: copyable commands that point other apps at this computer's model.
pub fn connect(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Stateful<Div> {
    let th2 = *th;
    let (port, mcp) = (
        s.port_in.read(cx).text.clone(),
        s.mcp_in.read(cx).text.clone(),
    );
    let model = s.st.model.clone().unwrap_or_default();
    let folder = s.st.agent_folder.clone().unwrap_or_default();
    let ps = s.snip == 2 || cfg!(windows) && s.snip == 5;
    let sn = logic::snippets(&logic::SnippetOptions {
        port: &port,
        mcp_port: &mcp,
        model: &model,
        id: "",
        folder: &folder,
        powershell: ps,
    });
    let code = sn.as_ref().map(|x| match s.snip {
        0 => format!("{}\n\n# check it answers\n{}", x.start, x.health),
        1 => x.curl.clone(),
        2 => x.powershell.clone(),
        3 => x.python.clone(),
        4 => x.anthropic.clone(),
        5 => x.claude.clone(),
        _ => format!("{}\n\n# or over HTTP\n{}", x.mcp, x.mcp_http),
    });
    let what = [
        "Serves the loaded model on an OpenAI-compatible API.",
        "Sends one message from a terminal.",
        "Sends one message from PowerShell.",
        "Any app that speaks the OpenAI API.",
        "Any app that speaks the Anthropic Messages API.",
        "Runs Claude Code on this computer's model.",
        "Gives another assistant this one's tools, limited to your documents folder.",
    ][s.snip];
    let copy = code.clone().unwrap_or_default();
    let server = match &s.server {
        None => div(),
        Some(Ok((st, body))) => div()
            .min_w_0()
            .text_size(u(13.))
            .text_color(if *st == 200 { th.keep } else { th.lose })
            .child(format!(
                "{} · {}",
                logic::http_message(*st),
                logic::clip(body, 120)
            )),
        Some(Err(e)) => (div().min_w_0().text_size(u(13.)))
            .text_color(th.lose)
            .child(e.clone()),
    };
    // Wraps: a refused connection's error ran past the card, last in the row.
    let ports = div()
        .flex()
        .flex_wrap()
        .items_center()
        .gap(u(14.))
        .child(
            div()
                .text_size(u(14.))
                .text_color(th.ink2)
                .child("Model API port"),
        )
        .child(
            div()
                .w(u(100.))
                .child(field(th, s.port_in.clone()).font_family("JetBrains Mono")),
        )
        .child(
            div()
                .text_size(u(14.))
                .text_color(th.ink2)
                .ml(u(10.))
                .child("Assistant MCP port"),
        )
        .child(
            div()
                .w(u(100.))
                .child(field(th, s.mcp_in.clone()).font_family("JetBrains Mono")),
        )
        .children(
            sn.as_ref()
                .ok()
                .map(|x| mono(x.base.clone()).text_size(u(13.)).text_color(th.ink2)),
        )
        .child(button(
            th,
            "check",
            "Check the server",
            Btn::Outline,
            click(cx, |s, cx| {
                let p = s.port_in.read(cx).text.clone();
                match logic::port(&p) {
                    Err(e) => s.server = Some(Err(e)),
                    Ok(p) => {
                        let task = cx.background_spawn(async move {
                            larch_app::connections::local_server_status(p, None)
                        });
                        cx.spawn(async move |this, cx| {
                            let r = task.await;
                            this.update(cx, |s, cx| {
                                s.server = Some(r.map(|r| (r.status, r.body.to_string())));
                                cx.notify()
                            })
                            .ok();
                        })
                        .detach();
                    }
                }
            }),
        ))
        .child(server);
    let pick = cx.entity().downgrade();
    page(
        div()
            .max_w(u(1100.))
            .flex()
            .flex_col()
            .gap(u(20.))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap(u(6.))
                    .child(serif("Put your model to work", 40.))
                    .child(div().text_size(u(14.)).text_color(th.ink2).child(
                        "Other apps can use the model running here. Nothing leaves this computer.",
                    )),
            )
            .child(card(th).p(u(24.)).gap(u(16.)).child(ports))
            .child(
                card(th)
                    .p(u(24.))
                    .gap(u(14.))
                    .child(div().flex().flex_wrap().gap(u(8.)).children(
                        SNIPS.iter().enumerate().map(|(i, n)| {
                            let pick = pick.clone();
                            pill(
                                &th2,
                                format!("snip-{i}"),
                                *n,
                                s.snip == i,
                                move |_, _, cx| {
                                    pick.update(cx, |s, cx| {
                                        s.snip = i;
                                        cx.notify()
                                    })
                                    .ok();
                                },
                            )
                        }),
                    ))
                    .child(div().text_size(u(14.)).text_color(th.ink2).child(what))
                    .child(match code {
                        Ok(c) => div()
                            .rounded(u(10.))
                            .bg(th.code_bg)
                            .text_color(th.code_ink)
                            .flex()
                            .flex_col()
                            .child(
                                div().flex().justify_end().px(u(14.)).pt(u(8.)).child(
                                    div()
                                        .id("copy-snip")
                                        .flex()
                                        .items_center()
                                        .gap(u(4.))
                                        .cursor_pointer()
                                        .text_size(u(12.))
                                        .text_color(fade(th.code_ink, 0.6))
                                        .child(icon("copy", 12., fade(th.code_ink, 0.6)))
                                        .child("Copy")
                                        .on_click(move |_, _, cx| {
                                            cx.write_to_clipboard(gpui::ClipboardItem::new_string(
                                                copy.clone(),
                                            ))
                                        }),
                                ),
                            )
                            .child(
                                div()
                                    .px(u(16.))
                                    .pb(u(14.))
                                    .font_family("JetBrains Mono")
                                    .text_size(u(12.5))
                                    .line_height(u(19.))
                                    .children(
                                        c.lines()
                                            .map(|l| {
                                                div()
                                                    .child(if l.is_empty() {
                                                        " ".to_owned()
                                                    } else {
                                                        l.to_owned()
                                                    })
                                                    .into_any_element()
                                            })
                                            .collect::<Vec<_>>(),
                                    ),
                            ),
                        Err(e) => div().text_color(th.lose).child(e.clone()),
                    }),
            ),
        "connect",
    )
}

/// A thumbnail of a theme: its sidebar, the greeting and the composer.
fn theme_card(
    th: &Theme,
    p: Theme,
    id: String,
    label: &str,
    on: bool,
    f: impl Fn(&gpui::ClickEvent, &mut Window, &mut App) + 'static,
) -> Stateful<Div> {
    let line = |w: f32, c: gpui::Hsla| div().h(u(4.)).w(gpui::relative(w)).rounded_full().bg(c);
    div()
        .id(gpui::ElementId::Name(id.into()))
        .flex_1()
        .min_w_0()
        .flex()
        .flex_col()
        .gap(u(10.))
        .p(u(10.))
        .rounded(u(14.))
        .bg(th.card)
        .border_color(if on { th.accent } else { th.line })
        .when(on, |d| d.border_2())
        .when(!on, |d| d.border_1())
        .cursor_pointer()
        .on_click(f)
        .child(
            div()
                .h(u(112.))
                .flex()
                .overflow_hidden()
                .rounded(u(10.))
                .border_1()
                .border_color(p.line)
                .bg(p.app)
                .child(
                    div()
                        .w(gpui::relative(0.3))
                        .flex_none()
                        .rounded_l(u(9.))
                        .bg(p.side)
                        .border_r_1()
                        .border_color(p.line)
                        .p(u(8.))
                        .flex()
                        .flex_col()
                        .gap(u(5.))
                        .child(div().mb(u(4.)).child(tree(
                            Tree::new(Mood::Ready),
                            14.,
                            225.,
                            true,
                            p,
                            p.side,
                        )))
                        .child(line(1., fade(p.cur, 1.)))
                        .child(line(1., fade(p.line, 1.)))
                        .child(line(1., fade(p.line, 1.)))
                        .child(line(0.8, fade(p.line, 1.))),
                )
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .p(u(10.))
                        .flex()
                        .flex_col()
                        .gap(u(8.))
                        .child(serif("Good evening.", 13.).text_color(p.ink).truncate())
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap(u(6.))
                                .p(u(6.))
                                .rounded(u(6.))
                                .border_1()
                                .border_color(p.line)
                                .bg(p.card)
                                .child(div().size(u(6.)).rounded_full().bg(p.sun))
                                .child(line(0.5, fade(p.ink3, 0.6)))
                                .child(div().flex_1())
                                .child(div().w(u(28.)).h(u(10.)).rounded(u(3.)).bg(p.accent)),
                        )
                        .child(
                            div()
                                .flex()
                                .gap(u(6.))
                                .p(u(6.))
                                .rounded(u(6.))
                                .border_1()
                                .border_color(p.line)
                                .child(line(0.45, fade(p.ink3, 0.5)))
                                .child(line(0.45, fade(p.ink3, 0.5))),
                        ),
                ),
        )
        .child(
            div()
                .text_size(u(14.))
                .font_weight(FontWeight::SEMIBOLD)
                .child(label.to_owned()),
        )
}

/// The version, and the build Larch Setup picked for this computer.
fn version_line(s: &Ecl) -> String {
    let build = s.install.as_ref().map(|i| match i.flavour.as_str() {
        "cuda" => " · NVIDIA build",
        "hip" => " · AMD build",
        "metal" => " · Apple silicon build",
        _ => " · processor build",
    });
    format!("Larch {}{}", env!("CARGO_PKG_VERSION"), build.unwrap_or(""))
}

/// Where the update check stands, beside its switch.
fn update_line(s: &Ecl) -> String {
    let install = s.install.as_ref();
    if let Some(v) = install.and_then(|i| i.pinned.as_ref()) {
        return format!("Pinned to {v} with larch-setup pin, so no updates are offered.");
    }
    if s.update_failed {
        return crate::ui::FAILED_UPDATE.into();
    }
    if s.checking {
        return "Checking…".into();
    }
    if let Some(e) = &s.check_err {
        return e.clone();
    }
    let ago = s.checked.map(|t| match t.elapsed().as_secs() / 60 {
        0 => "just now".to_owned(),
        1 => "1 minute ago".to_owned(),
        m if m < 120 => format!("{m} minutes ago"),
        m => format!("{} hours ago", m / 60),
    });
    match (s.update_offer(), ago) {
        (Some(v), _) => format!("Larch {v} is ready to install."),
        (None, Some(ago)) => format!("Up to date. Checked {ago}."),
        (None, None) => "Asks before installing anything.".into(),
    }
}

/// The newer release at the top of General, with its size and notes.
fn update_banner(s: &Ecl, th: &Theme, cx: &Context<Ecl>) -> Option<Div> {
    let version = s.update_offer()?;
    let size = s.found.as_ref().map_or(0, |f| f.download_size);
    let notes = crate::update::notes(&version);
    Some(
        card(th)
            .mb(u(16.))
            .flex_row()
            .flex_wrap()
            .items_center()
            .gap(u(14.))
            .child(icon("up", 18., th.accent))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap(u(2.))
                    .child(
                        div()
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(format!("Larch {version} is ready to install")),
                    )
                    .child(div().text_size(u(13.)).text_color(th.ink2).child(format!(
                        "Signed release · {} to download",
                        crate::update::size(size)
                    ))),
            )
            .child(button(
                th,
                "update-notes2",
                "Release notes",
                Btn::Ghost,
                move |_, _, cx| cx.open_url(&notes),
            ))
            .child(button(
                th,
                "update-later",
                "Later",
                Btn::Ghost,
                click(cx, |s, _| s.update_later = true),
            ))
            .child(button(
                th,
                "update-restart2",
                "Restart to update",
                Btn::Primary,
                cx.listener(|s, _, w, cx| s.restart_to_update(w, cx)),
            )),
    )
}

/// Settings and its tabs.
pub fn settings(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Stateful<Div> {
    let th2 = *th;
    const TABS: [(STab, &str); 7] = [
        (STab::General, "General"),
        (STab::Appearance, "Appearance"),
        (STab::Assistant, "Assistant"),
        (STab::Models, "Models"),
        (STab::Privacy, "Privacy"),
        (STab::Shortcuts, "Shortcuts"),
        (STab::Advanced, "Advanced"),
    ];
    // A narrow window hid Advanced past the page's edge.
    let tabs = div()
        .flex()
        .flex_wrap()
        .border_b_1()
        .border_color(th.line)
        .children(TABS.iter().map(|(t, l)| {
            let t = *t;
            tab(
                &th2,
                format!("st-{l}"),
                *l,
                s.stab == t,
                48.,
                click(cx, move |s, _| s.stab = t),
            )
        }))
        // What the assistant remembers, for those who asked for the details.
        .when(s.prefs.advanced, |d| {
            d.child(tab(
                &th2,
                "st-memory".into(),
                "Memory",
                s.stab == STab::Memory,
                48.,
                click(cx, |s, cx| {
                    s.stab = STab::Memory;
                    s.load_memory(cx)
                }),
            ))
        });
    // Before `onoff`, which holds `cx` for the rest of the page.
    let name_marks = crate::tour::marks(s, crate::tour::Spot::Name, th, cx);
    let onoff = |id: &str, on: bool, f: fn(&mut Ecl, bool)| {
        seg(
            &th2,
            id,
            &["On", "Off"],
            if on { 0 } else { 1 },
            set(cx, move |s, i, _| f(s, i == 0)),
        )
    };
    let p = &s.prefs;
    let body: Div = match s.stab {
        STab::General => div()
            .flex()
            .flex_col()
            .children(update_banner(s, th, cx))
            .child(setrow(th, "Your name", div().w(u(360.)).flex().gap(u(8.)).relative().children(name_marks).child(field(th, s.name_in.clone())).child(button(th, "save-name", "Save", Btn::Outline, click(cx, |s, cx| s.name_in.update(cx, |_, cx| cx.emit(crate::input::Submitted)))))))
            .child(setrow(th, "Advanced", onoff("adv", p.advanced, |s, v| s.prefs.advanced = v)))
            .when(p.advanced, |d| d.child(setrow(th, "Side panel", onoff("side-panel", p.side_panel, |s, v| s.prefs.side_panel = v))))
            .child(setrow(th, "Tour", button(th, "take-tour", "Take the tour", Btn::Outline, click(cx, crate::tour::start))))
            .child(setrow(th, "Version", mono(version_line(s)).text_size(u(13.5)).text_color(th.ink2)))
            .when(s.install.is_some(), |d| {
                d.child(setrow(
                    th,
                    "Check for updates",
                    div()
                        .flex_1()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap(u(12.))
                        .child(onoff("updates", p.updates, |s, v| s.prefs.updates = v))
                        .child(div().flex_1().text_size(u(13.)).text_color(th.ink3).child(update_line(s)))
                        .child(button(th, "check-now", "Check now", Btn::Ghost, click(cx, |s, cx| s.check_updates(cx)))),
                ))
                .child(setrow(
                    th,
                    "Installation",
                    div().child(button(th, "open-setup", "Open Larch Setup", Btn::Ghost, click(cx, |s, _| {
                        if let Some(Err(e)) = s.install.as_ref().map(|i| i.open_setup()) {
                            s.say(e)
                        }
                    }))),
                ))
            }),
        STab::Appearance => {
            let lights = [(Light::Paper, "Paper"), (Light::Porcelain, "Porcelain"), (Light::HighContrast, "High contrast")];
            let darks = [(Dark::Night, "Night"), (Dark::Forest, "Forest"), (Dark::Graphite, "Graphite")];
            div()
                .flex()
                .flex_col()
                .child(
                    // The dark themes go under the light ones once a row of six would squeeze them;
                    // alone on a row, three shrink with the page (a min width ran them off it).
                    div()
                        .flex()
                        .flex_wrap()
                        .gap_x(u(28.))
                        .gap_y(u(20.))
                        .pb(u(20.))
                        .child(div().flex_grow().flex_basis(u(500.)).min_w_0().flex().flex_col().gap(u(10.)).child(eyebrow(th, "Light theme")).child(div().flex().gap(u(14.)).children(lights.into_iter().map(|(l, n)| theme_card(&th2, Theme::light(l), format!("light-{n}"), n, p.light == l, click(cx, move |s, _| {
                            // Picking a theme shows it: the mode follows the side it is on.
                            s.prefs.light = l;
                            s.prefs.mode = Mode::Light;
                            s.save_prefs()
                        }))))))
                        .child(div().flex_grow().flex_basis(u(500.)).min_w_0().flex().flex_col().gap(u(10.)).child(eyebrow(th, "Dark theme")).child(div().flex().gap(u(14.)).children(darks.into_iter().map(|(d, n)| theme_card(&th2, Theme::dark(d), format!("dark-{n}"), n, p.dark == d, click(cx, move |s, _| {
                            s.prefs.dark = d;
                            s.prefs.mode = Mode::Dark;
                            s.save_prefs()
                        })))))),
                )
                .child(setrow(th, "Advanced", onoff("adv", p.advanced, |s, v| s.prefs.advanced = v)))
                .child(setrow(
                    th,
                    "Mode",
                    seg(th, "mode", &["System", "Light", "Dark"], p.mode as usize, set(cx, |s, i, _| s.prefs.mode = [Mode::System, Mode::Light, Mode::Dark][i])),
                ))
                .child(setrow(th, "Spacing", seg(th, "spacing", &["Roomy", "Compact"], p.compact as usize, set(cx, |s, i, _| s.prefs.compact = i == 1))))
                .child(setrow(th, "Text size", seg(th, "text", &["13", "14", "15", "16"], (p.text_size.clamp(13, 16) - 13) as usize, set(cx, |s, i, _| s.prefs.text_size = 13 + i as u8))))
                .child(setrow(
                    th,
                    "Motion",
                    seg(th, "motion", &["System", "Full", "Reduced", "Off"], p.motion as usize, set(cx, |s, i, _| s.prefs.motion = [Motion::System, Motion::Full, Motion::Reduced, Motion::Off][i])),
                ))
        }
        STab::Assistant => {
            let still = s.still();
            div()
                .flex()
                .flex_col()
                // As many states a row as fit: ten broke their names ("Waitin g") on a narrow page.
                .child(
                    card(th).mb(u(20.)).child(eyebrow(th, "States")).child(div().grid().grid_cols(logic::columns(s.room - PAGE_X * 2. - 40., 72., 10., 10)).gap(u(10.)).children(Mood::ALL.iter().map(|m| {
                        div().flex().flex_col().items_center().gap(u(10.)).py(u(16.)).rounded(u(12.)).bg(th2.sunk).child(tree(Tree::new(*m), 64., 225., still, th2, th2.sunk)).child(div().text_size(u(13.5)).text_color(th2.ink2).child(m.label()))
                    }))),
                )
                .child(setrow(th, "Tree", onoff("sun", p.sun, |s, v| s.prefs.sun = v)))
                .child(setrow(th, "Tree in chats", seg(th, "sun-small", &["Full size", "Always small"], if p.sun_small { 1 } else { 0 }, set(cx, |s, i, _| s.prefs.sun_small = i == 1))))
                .child(setrow(th, "Animation", seg(th, "anim", &["On", "Still"], if p.animate { 0 } else { 1 }, set(cx, |s, i, _| s.prefs.animate = i == 0))))
                .child(setrow(th, "Speech bubbles", seg(th, "bubbles", &["When it needs you", "Off"], if p.bubbles { 0 } else { 1 }, set(cx, |s, i, _| s.prefs.bubbles = i == 0))))
                .child(setrow(th, "Notify me", seg(th, "notify", &["When it needs you", "Off"], if p.notify { 0 } else { 1 }, set(cx, |s, i, _| s.prefs.notify = i == 0))))
                .child(setrow(th, "Sound", seg(th, "sound", &["Soft chime", "Off"], if p.sound { 0 } else { 1 }, set(cx, |s, i, _| s.prefs.sound = i == 0))))
        }
        STab::Models => {
            let libs = s.st.libraries.clone();
            let hf = s.st.hf_cache.clone();
            // None: the shared Hugging Face cache, or the first library when there is none.
            let target = p.download_to.clone().filter(|l| libs.contains(l)).or_else(|| libs.first().cloned().filter(|_| hf.is_none()));
            div()
                .flex()
                .flex_col()
                .child(setrow(
                    th,
                    "Libraries",
                    div()
                        .flex_1()
                        .flex()
                        .flex_col()
                        .gap(u(8.))
                        .children(libs.iter().enumerate().map(|(i, l)| {
                            let lib = l.clone();
                            div().flex().items_center().gap(u(10.)).child(icon("folder", 15., th2.ink2)).child(mono(home_short(l)).text_size(u(13.5)).flex_1().min_w_0()).when(i > 0, |d| {
                                d.child(button(&th2, format!("rml-{i}"), "Remove", Btn::Ghost, click(cx, move |s, cx| {
                                    let lib = lib.clone();
                                    s.call(cx, move |c| c.remove_library(&lib), |s, libs, cx| {
                                        s.st.libraries = libs;
                                        s.refresh(cx)
                                    })
                                })))
                            })
                        }))
                        .child(div().child(button(th, "add-lib2", "Add a folder", Btn::Outline, cx.listener(|s, _, _, cx| add_library(s, cx))))),
                ))
                .child(setrow(
                    th,
                    "Downloads go to",
                    div().flex().flex_wrap().gap(u(8.)).children(hf.map(|h| {
                        pill(&th2, "dlto-hf", format!("Hugging Face cache ({})", home_short(&h)), target.is_none(), click(cx, move |s, _| {
                            s.prefs.download_to = None;
                            s.save_prefs()
                        }))
                    })).children(libs.iter().enumerate().map(|(i, l)| {
                        let lib = l.clone();
                        pill(&th2, format!("dlto-{i}"), home_short(l), target.as_ref() == Some(l), click(cx, move |s, _| {
                            s.prefs.download_to = Some(lib.clone());
                            s.save_prefs()
                        }))
                    })),
                ))
                .child(setrow(th, "Other apps' models", div().text_size(u(14.)).text_color(th.ink2).child("Models from LM Studio, Ollama, Hugging Face and others are read where they are, never copied. The Hugging Face cache is shared: a model deleted there is gone for every app that uses it.")))
        }
        STab::Privacy => div()
            .flex()
            .flex_col()
            .child(setrow(
                th,
                "Memory",
                seg(th, "mem", &["On", "Off"], if s.mem_on { 0 } else { 1 }, set(cx, |s, i, cx| {
                    let on = i == 0;
                    s.call(cx, move |c| c.memory_set_enabled(on), move |s, _, _| s.mem_on = on)
                })),
            ))
            .child(setrow(
                th,
                "Keep documents ready after restart",
                div().flex().items_center().gap(u(14.)).child(
                    seg(th, "docs-keep", &["On", "Off"], if s.st.docs_keep_ready { 0 } else { 1 }, set(cx, |s, i, cx| {
                        let on = i == 0;
                        s.call(cx, move |c| c.set_docs_keep_ready(on), move |s, _, _| s.st.docs_keep_ready = on)
                    })),
                ).child(div().text_size(u(13.5)).text_color(th.ink2).child("A document's read is saved on this computer, so its answers stay fast after a restart.")),
            ))
            .child(setrow(th, "Where your data is", div().flex_1().flex().flex_col().gap(u(4.)).child(div().text_size(u(14.)).text_color(th.ink2).child("Chats, memory and models stay on this computer. Larch only goes online to search and download models, and to fetch pages the assistant cites.")).child(mono(s.st.data_dir.clone()).text_size(u(12.5)).text_color(th.ink3)))),
        STab::Shortcuts => {
            let m = if cfg!(target_os = "macos") { "⌘" } else { "Ctrl" };
            div().flex().flex_col().children(SHORTCUTS.iter().map(|(k, what)| setrow(&th2, what, div().flex().gap(u(6.)).children(k.split(" … ").map(|k| tag(&th2, logic::keys(k, m)))))))
        }
        STab::Memory => memory(s, th, cx),
        STab::Advanced => div()
            .flex()
            .flex_col()
            .child(setrow(th, "Data folder", mono(s.st.data_dir.clone()).text_size(u(13.5)).text_color(th.ink2)))
            .child(setrow(th, "Device", div().text_size(u(14.)).text_color(th.ink2).child(if s.st.device.is_empty() { "CPU".to_owned() } else { format!("{}{}", s.st.device, s.gpu_bytes().map_or(String::new(), |b| format!(" · {}", logic::gb(b)))) })))
            .child(setrow(th, "Code mode", div().text_size(u(14.)).text_color(th.ink2).child(if s.st.code_mode { "Available: changes run and test in a sandbox before you keep them." } else { "Off in this release." }))),
    };
    let out = page(
        div().flex().flex_col().gap(u(20.)).child(tabs).child(body),
        "settings",
    );
    // Remembered facts select and copy as a chat's text does.
    if s.stab == STab::Memory {
        selects(out, cx)
    } else {
        out
    }
}

#[cfg(test)]
mod home_tests {
    /// The home folder is `~` by whole folders: a folder whose name starts like it is not under it.
    #[test]
    fn only_folders_under_home_are_written_with_a_tilde() {
        let sep = std::path::MAIN_SEPARATOR;
        let home = format!("{sep}u{sep}alex");
        let under = format!("{home}{sep}models");
        assert_eq!(super::under_home(&under, &home), format!("~{sep}models"));
        assert_eq!(super::under_home(&home, &home), "~");
        let beside = format!("{sep}u{sep}alex2{sep}x");
        assert_eq!(super::under_home(&beside, &home), beside);
        assert_eq!(super::under_home(&under, ""), under);
    }
}

#[cfg(test)]
mod catalog_tests {
    /// The recommended card reads its models from the catalog (bench/models.toml), not from code,
    /// and picks them by the GPU's memory, the largest first.
    #[test]
    fn the_recommended_models_come_from_the_catalog_by_gpu_memory() {
        let names = |gpu| {
            let models = super::recommended_models(gpu);
            assert!((models.iter()).all(|m| m.repo.contains('/')
                && m.file.ends_with(".gguf")
                && !m.name.is_empty()
                && !m.why.is_empty()));
            models.into_iter().map(|m| m.name).collect::<Vec<_>>()
        };
        // Names by alias, as the catalog gives them.
        let catalog: toml::Table = include_str!("../../../bench/models.toml").parse().unwrap();
        let by = |aliases: &[&str]| -> Vec<String> {
            (aliases.iter())
                .map(|a| catalog[*a]["display"].as_str().unwrap().to_owned())
                .collect()
        };
        // Unknown, and a 4 GB GPU: the small models only.
        assert_eq!(names(None), by(&["m10", "m11"]));
        assert_eq!(names(Some(4_294_967_296)), by(&["m10", "m11"]));
        // A 16 GB Mac: macOS gives its GPU about two thirds of the memory.
        assert_eq!(names(Some(11_453_246_122)), by(&["m12", "m10", "m11"]));
        // A 24 GB Mac: an M5 gives its GPU about 19 GB.
        assert_eq!(
            names(Some(19_070_000_000)),
            by(&["m13", "m12", "m10", "m11"])
        );
    }
}
