//! The installer's screens. Each one is a larch-setup command: a `--dry-run` fills the review,
//! then the command itself, whose download, unpack and apply lines drive the progress screen.
//! A fresh install walks This computer, First model, Review and Install; an installed copy opens
//! on its own page (update, add or remove parts, repair, uninstall).

use std::collections::BTreeMap;
use std::path::PathBuf;

use futures::StreamExt;
use gpui::{
    AnyElement, App, ClickEvent, Context, Div, FontWeight, InteractiveElement, IntoElement,
    ParentElement, PromptLevel, Render, SharedString, StatefulInteractiveElement, Styled, Window,
    WindowAppearance, div, prelude::FluentBuilder,
};
use larch_setup::UpdateCheck;
use larch_setup::detect::Gpu;
use larch_setup::manifest::CatalogModel;
use larch_setup::txn::Receipt;
use serde::Deserialize;
use serde_json::Value;

use crate::engine::{self, Failure, Msg};
use crate::kit::{Btn, bar, button, card, eyebrow, icon, mono, print, serif, u};
use crate::theme::{Dark, Light, Theme, fade};

/// The fonts' license. The SIL Open Font License must go with each copy of the fonts, and on
/// Windows and Linux this window is downloaded as a single file, so it carries the text itself.
const FONT_LICENSE: &str = include_str!("../../desktop/assets/fonts/OFL.txt");

/// The plan larch-setup prints for a `--dry-run`: what it would download and install, and where.
#[derive(Clone, Debug, Deserialize)]
pub struct Plan {
    version: String,
    flavour: String,
    gpu: Option<Gpu>,
    #[serde(default)]
    components: BTreeMap<String, Sizes>,
    download: u64,
    required: u64,
    free: u64,
    root: PathBuf,
    data: PathBuf,
    data_size: u64,
    #[serde(default)]
    models: Vec<CatalogModel>,
}

/// One component's compressed download and unpacked size, in bytes.
#[derive(Clone, Copy, Debug, Deserialize)]
struct Sizes {
    download: u64,
    unpack: u64,
}

/// The parts a person picks; setup always comes with them.
const PARTS: [(&str, &str, &str); 2] = [
    (
        "app",
        "Larch app",
        "The desktop app: chats, documents, tasks and models.",
    ),
    (
        "cli",
        "Command line",
        "larch and larch-agent, for the terminal and scripts.",
    ),
];

/// A component as the progress list names it.
fn part_name(c: &str) -> String {
    match c {
        "app" => "the app".into(),
        "cli" => "the command line".into(),
        "setup" => "setup and updates".into(),
        "cuda" => "the NVIDIA libraries".into(),
        c if c.starts_with("rocm") => "the AMD libraries".into(),
        c => c.into(),
    }
}

fn cap(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().chain(c).collect())
        .unwrap_or_default()
}

/// A byte count as people read it: GB with one decimal, else whole MB (at least 1).
fn size(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.1} GB", b as f64 / 1e9)
    } else {
        // Anything at all reads as 1 MB, so a small part never looks free.
        let mb = (b as f64 / 1e6).round() as u64;
        format!("{} MB", if b > 0 { mb.max(1) } else { 0 })
    }
}

/// The steps of a fresh install, shown down the side.
const STEPS: [&str; 4] = ["This computer", "First model", "Review", "Install"];

/// One row of the progress list, weighed by its bytes.
#[derive(Clone, Debug)]
struct Task {
    id: String,
    label: String,
    total: f64,
    done: f64,
}

/// A command in progress.
#[derive(Debug, Default)]
struct Run {
    tasks: Vec<Task>,
    /// The row in progress; `tasks.len()` when all are done.
    at: usize,
    /// The GPU check failed: setup put everything back, and the processor build is offered.
    gpu_failed: bool,
}

impl Run {
    /// Marks `id` as `done` of `total` (0 keeps its total), and every row before it finished.
    fn step(&mut self, id: &str, done: f64, total: f64) {
        let Some(k) = self.tasks.iter().position(|t| t.id == id) else {
            return;
        };
        for t in &mut self.tasks[..k] {
            t.done = t.total;
        }
        let t = &mut self.tasks[k];
        t.done = done;
        if total > 0. {
            t.total = total;
        }
        self.at = if t.done >= t.total { k + 1 } else { k };
    }

    /// The whole run's share done, 0 to 1.
    fn share(&self) -> f32 {
        let all: f64 = self.tasks.iter().map(|t| t.total).sum();
        let got: f64 = self.tasks.iter().map(|t| t.done.min(t.total)).sum();
        if all > 0. { (got / all) as f32 } else { 0. }
    }
}

/// What a retry runs again.
#[derive(Clone, Debug)]
enum Retry {
    Boot,
    Apply(Vec<String>, Box<Plan>, Option<CatalogModel>),
    /// The processor build, after the GPU check failed: its plan, then the review.
    Cpu(Vec<String>),
    Uninstall,
    Straight(&'static str),
    Choose,
}

/// The screen on show.
#[derive(Clone, Debug)]
enum Screen {
    Loading,
    Choose,
    Model,
    Review {
        args: Vec<String>,
        plan: Box<Plan>,
    },
    Progress {
        title: &'static str,
    },
    Done {
        uninstall: bool,
        fresh: bool,
        cli_tip: bool,
        note: Option<String>,
    },
    Failed {
        failure: Failure,
        gpu: bool,
        retry: Option<Retry>,
    },
    Manage,
    Uninstall {
        plan: Box<Plan>,
    },
}

/// The installer window's state.
pub struct Setup {
    screen: Screen,
    /// The installed copy, if any.
    receipt: Option<Receipt>,
    /// A fresh install's plan with every part, for the first screens.
    plan: Option<Plan>,
    /// The update check on the installed page: None while it runs, Some(None) when it failed.
    check: Option<Option<UpdateCheck>>,
    app: bool,
    cli: bool,
    /// The first model's file; None is Skip.
    model: Option<String>,
    model_picked: bool,
    purge: bool,
    run: Run,
    /// A dry run is in flight: the button that started it says Checking.
    checking: bool,
    uninstalling: bool,
    dark: bool,
    /// The licenses page is open over the current screen.
    licenses: bool,
}

/// A screen's scrolling content, and the buttons pinned under it.
type Page = (Div, Option<Div>);

/// What one of the manage screen's rows does when clicked.
type Action = Box<dyn Fn(&mut Setup, &mut Context<Setup>)>;

impl Setup {
    /// The window, reading this computer and any installed copy as it opens.
    pub fn new(uninstalling: bool, window: &mut Window, cx: &mut Context<Self>) -> Self {
        cx.observe_window_appearance(window, |s, w, cx| {
            s.dark = dark(w.appearance());
            cx.notify();
        })
        .detach();
        // Closing stops setup mid-change, so ask first; one question at a time.
        let asking = std::rc::Rc::new(std::cell::Cell::new(false));
        window.on_window_should_close(cx, move |window, cx| {
            if !engine::busy() || asking.replace(true) {
                return !engine::busy();
            }
            let answer = window.prompt(
                PromptLevel::Warning,
                "Larch Setup is still working.",
                Some("Closing stops it now. Setup undoes an unfinished install the next time it runs."),
                &["Stop and close", "Keep going"],
                cx,
            );
            let asking = asking.clone();
            cx.spawn(async move |cx| {
                let stop = answer.await == Ok(0);
                asking.set(false);
                if stop {
                    // Quitting stops the command (main.rs), as closing the window does.
                    let _ = cx.update(|cx| cx.quit());
                }
            })
            .detach();
            false
        });
        let mut s = Setup {
            screen: Screen::Loading,
            receipt: None,
            plan: None,
            check: None,
            app: true,
            cli: true,
            model: None,
            model_picked: false,
            purge: false,
            run: Run::default(),
            checking: false,
            uninstalling,
            dark: dark(window.appearance()),
            licenses: false,
        };
        s.boot(cx);
        s
    }

    fn theme(&self) -> Theme {
        if self.dark {
            Theme::dark(Dark::Night)
        } else {
            Theme::light(Light::Paper)
        }
    }

    /// Runs `larch-setup <args>` off the UI thread: its progress lines reach `on_event`, its
    /// result reaches `then`.
    fn call(
        &mut self,
        args: Vec<String>,
        cx: &mut Context<Self>,
        then: impl FnOnce(&mut Self, Result<Value, Failure>, &mut Context<Self>) + 'static,
    ) {
        let (tx, mut rx) = futures::channel::mpsc::unbounded();
        std::thread::spawn(move || engine::run(args, |m| tx.unbounded_send(m).is_ok()));
        let mut then = Some(then);
        cx.spawn(async move |this, cx| {
            while let Some(m) = rx.next().await {
                let done = matches!(m, Msg::Done(_));
                let alive = this.update(cx, |s, cx| {
                    match m {
                        Msg::Event(e) => s.on_event(&e),
                        Msg::Done(r) => {
                            s.checking = false;
                            if let Some(f) = then.take() {
                                f(s, r, cx);
                            }
                        }
                    }
                    cx.notify();
                });
                if done || alive.is_err() {
                    break;
                }
            }
        })
        .detach();
    }

    /// `call`, with the result read as `T`.
    fn call_as<T: serde::de::DeserializeOwned + 'static>(
        &mut self,
        args: &[&str],
        cx: &mut Context<Self>,
        then: impl FnOnce(&mut Self, Result<T, Failure>, &mut Context<Self>) + 'static,
    ) {
        let args = args.iter().map(|a| a.to_string()).collect();
        self.call(args, cx, move |s, r, cx| {
            let r = r.and_then(|v| {
                serde_json::from_value(v).map_err(|e| Failure {
                    message: format!("setup's answer did not read: {e}"),
                    busy: false,
                })
            });
            then(s, r, cx)
        });
    }

    fn on_event(&mut self, e: &Value) {
        let num = |k: &str| e[k].as_f64().unwrap_or(0.);
        let component = e["component"].as_str().unwrap_or_default();
        match e["event"].as_str() {
            Some("download") => self.run.step(component, num("done"), num("total")),
            Some("unpack") => self.run.step(component, f64::INFINITY, 0.),
            Some("apply") => self.run.step("apply", 0., 0.),
            Some("gpu_failed") => self.run.gpu_failed = true,
            _ => {}
        }
    }

    fn failed(&mut self, failure: Failure, retry: Option<Retry>) {
        self.screen = Screen::Failed {
            failure,
            gpu: false,
            retry,
        };
    }

    // ---------- the flows ----------

    fn boot(&mut self, cx: &mut Context<Self>) {
        self.screen = Screen::Loading;
        self.call_as::<Option<Receipt>>(&["status"], cx, |s, r, cx| match r {
            Ok(Some(receipt)) => {
                s.receipt = Some(receipt);
                if s.uninstalling {
                    s.uninstalling = false;
                    s.uninstall(cx);
                } else {
                    s.manage(cx);
                }
            }
            Ok(None) => s.call_as::<Plan>(
                &["install", "--cli", "--app", "--dry-run"],
                cx,
                |s, r, _| match r {
                    Ok(plan) => {
                        s.plan = Some(plan);
                        s.screen = Screen::Choose;
                    }
                    Err(e) => s.failed(e, Some(Retry::Boot)),
                },
            ),
            Err(e) => s.failed(e, Some(Retry::Boot)),
        });
    }

    /// The installed copy's page, and a fresh update check behind it.
    fn manage(&mut self, cx: &mut Context<Self>) {
        self.screen = Screen::Manage;
        self.check = None;
        self.call_as::<UpdateCheck>(&["update", "--check"], cx, |s, r, _| {
            s.check = Some(r.ok());
        });
    }

    /// Continue on This computer (or Add or remove parts).
    fn chose(&mut self, cx: &mut Context<Self>) {
        let Some(receipt) = &self.receipt else {
            if self.plan.as_ref().is_some_and(|p| !p.models.is_empty()) {
                self.screen = Screen::Model;
            } else {
                self.review(cx);
            }
            return;
        };
        let have = |k: &str| receipt.components.contains_key(k);
        let want = |k: &str| if k == "app" { self.app } else { self.cli };
        let add: Vec<&str> = ["app", "cli"]
            .into_iter()
            .filter(|k| want(k) && !have(k))
            .collect();
        let remove: Vec<&str> = ["app", "cli"]
            .into_iter()
            .filter(|k| !want(k) && have(k))
            .collect();
        let mut args = vec!["modify".to_owned()];
        if !add.is_empty() {
            args.push("--add".into());
            args.extend(add.iter().map(|k| k.to_string()));
        }
        if !remove.is_empty() {
            args.push("--remove".into());
            args.extend(remove.iter().map(|k| k.to_string()));
        }
        self.dry_run(args, Retry::Choose, cx);
    }

    /// The review for a fresh install of the picked parts.
    fn review(&mut self, cx: &mut Context<Self>) {
        let mut args = vec!["install".to_owned()];
        if self.app {
            args.push("--app".into());
        }
        if self.cli {
            args.push("--cli".into());
        }
        self.dry_run(args, Retry::Choose, cx);
    }

    /// `args --dry-run`, then the review of exactly that plan.
    fn dry_run(&mut self, args: Vec<String>, retry: Retry, cx: &mut Context<Self>) {
        self.checking = true;
        let mut dry = args.clone();
        dry.push("--dry-run".into());
        self.call(dry, cx, move |s, r, _| {
            match r.and_then(|v| {
                serde_json::from_value::<Plan>(v).map_err(|e| Failure {
                    message: format!("setup's plan did not read: {e}"),
                    busy: false,
                })
            }) {
                Ok(plan) => {
                    s.screen = Screen::Review {
                        args,
                        plan: Box::new(plan),
                    }
                }
                Err(e) => s.failed(e, Some(retry)),
            }
        });
    }

    /// The fresh install's first model, if one was picked.
    fn first_model(&self) -> Option<CatalogModel> {
        let file = self.model.as_ref()?;
        (self.plan.as_ref()?.models.iter())
            .find(|m| &m.file == file)
            .cloned()
    }

    /// Runs a change: one task per download, then the transaction, then the model.
    fn apply(
        &mut self,
        args: Vec<String>,
        plan: Option<&Plan>,
        model: Option<CatalogModel>,
        cx: &mut Context<Self>,
    ) {
        let fresh = self.receipt.is_none();
        let mut tasks: Vec<Task> = (plan.map(|p| p.components.iter()).into_iter().flatten())
            .map(|(c, s)| Task {
                id: c.clone(),
                label: format!("Download {}", part_name(c)),
                total: s.download as f64,
                done: 0.,
            })
            .collect();
        let uninstall = args[0] == "uninstall";
        tasks.push(Task {
            id: "apply".into(),
            label: if uninstall {
                "Remove Larch"
            } else {
                "Install and check"
            }
            .into(),
            // The transaction is weighed as a small share of the download.
            total: (plan.map_or(0, |p| p.download) as f64 * 0.08).max(1.),
            done: 0.,
        });
        if let Some(m) = &model {
            tasks.push(Task {
                id: "model".into(),
                label: format!("Download {}", m.display),
                total: m.size as f64,
                done: 0.,
            });
        }
        let title = match args[0].as_str() {
            "install" => "Installing Larch",
            "uninstall" => "Removing Larch",
            "update" => "Updating Larch",
            "repair" => "Repairing Larch",
            _ => "Changing Larch",
        };
        self.screen = Screen::Progress { title };
        self.run = Run {
            tasks,
            ..Default::default()
        };
        let plan = plan.cloned();
        let again = args.clone();
        self.call(args.clone(), cx, move |s, r, cx| {
            if let Err(e) = r {
                if s.run.gpu_failed {
                    let cpu = s.cpu_install(&again);
                    s.screen = Screen::Failed {
                        failure: e,
                        gpu: true,
                        retry: Some(Retry::Cpu(cpu)),
                    };
                } else {
                    let retry = plan.map(|p| Retry::Apply(again, Box::new(p), model));
                    s.failed(e, retry);
                }
                return;
            }
            s.run.step("apply", 1., 1.);
            match model {
                Some(m) if !uninstall => {
                    let file = m.file.clone();
                    let total = m.size as f64;
                    s.call(
                        vec!["model".into(), "get".into(), file],
                        cx,
                        move |s, r, cx| {
                            let note = r.err().map(|e| {
                                format!(
                                    "The model did not download: {}. You can get it later from \
                                     Models in the app.",
                                    e.message
                                )
                            });
                            if note.is_none() {
                                s.run.step("model", total, total);
                            }
                            s.finish(args, fresh, note, cx);
                        },
                    );
                }
                _ => s.finish(args, fresh, None, cx),
            }
        });
    }

    /// The processor build as a new install of the parts `args` was installing, as larch-setup's
    /// own retry runs it: only `install` takes `--flavour`.
    fn cpu_install(&self, args: &[String]) -> Vec<String> {
        // The change was rolled back, so the receipt is still the copy from before it.
        let installed = self.receipt.as_ref().filter(|_| args[0] != "modify");
        let has = |k: &str| match installed {
            Some(r) => r.components.contains_key(k),
            None => (k == "app" && self.app) || (k == "cli" && self.cli),
        };
        let mut cpu = vec!["install".to_owned()];
        cpu.extend(
            ["app", "cli"]
                .into_iter()
                .filter(|k| has(k))
                .map(|k| format!("--{k}")),
        );
        if let Some(r) = &self.receipt {
            cpu.extend(["--channel".into(), r.channel.clone()]);
        }
        cpu.extend(["--flavour".into(), "cpu".into()]);
        cpu
    }

    /// Reads the receipt as it now stands, then says how it went.
    fn finish(
        &mut self,
        args: Vec<String>,
        fresh: bool,
        note: Option<String>,
        cx: &mut Context<Self>,
    ) {
        let uninstall = args[0] == "uninstall";
        let cli_tip = fresh || args.iter().any(|a| a == "cli");
        if uninstall {
            self.receipt = None;
            self.screen = Screen::Done {
                uninstall,
                fresh,
                cli_tip: false,
                note,
            };
            return;
        }
        self.call_as::<Option<Receipt>>(&["status"], cx, move |s, r, _| {
            if let Ok(r) = r {
                s.receipt = r;
            }
            let cli = s
                .receipt
                .as_ref()
                .is_some_and(|r| r.components.contains_key("cli"));
            s.screen = Screen::Done {
                uninstall,
                fresh,
                cli_tip: cli_tip && cli,
                note,
            };
        });
    }

    fn uninstall(&mut self, cx: &mut Context<Self>) {
        self.screen = Screen::Loading;
        self.call_as::<Plan>(&["uninstall", "--dry-run"], cx, |s, r, _| match r {
            Ok(plan) => {
                s.purge = false;
                s.screen = Screen::Uninstall {
                    plan: Box::new(plan),
                };
            }
            Err(e) => s.failed(e, Some(Retry::Uninstall)),
        });
    }

    /// Update and repair go straight to progress: their dry run only sizes the downloads.
    fn straight(&mut self, cmd: &'static str, cx: &mut Context<Self>) {
        self.screen = Screen::Loading;
        self.call_as::<Plan>(&[cmd, "--dry-run"], cx, move |s, r, cx| match r {
            Ok(plan) => s.apply(vec![cmd.into()], Some(&plan), None, cx),
            Err(e) => s.failed(e, Some(Retry::Straight(cmd))),
        });
    }

    fn retry(&mut self, r: Retry, cx: &mut Context<Self>) {
        match r {
            Retry::Boot => self.boot(cx),
            Retry::Apply(args, plan, model) => self.apply(args, Some(&plan), model, cx),
            Retry::Cpu(args) => self.dry_run(args, Retry::Choose, cx),
            Retry::Uninstall => self.uninstall(cx),
            Retry::Straight(cmd) => self.straight(cmd, cx),
            Retry::Choose => self.screen = Screen::Choose,
        }
    }

    fn open_app(&mut self, cx: &mut Context<Self>) {
        match larch_setup::cli::open_app() {
            Ok(()) => cx.quit(),
            Err(e) => self.failed(
                Failure {
                    message: e.to_string(),
                    busy: false,
                },
                None,
            ),
        }
    }
}

fn dark(a: WindowAppearance) -> bool {
    matches!(a, WindowAppearance::Dark | WindowAppearance::VibrantDark)
}

/// A click handler on the window's state.
fn on(
    cx: &mut Context<Setup>,
    f: impl Fn(&mut Setup, &mut Context<Setup>) + 'static,
) -> impl Fn(&ClickEvent, &mut Window, &mut App) + 'static {
    cx.listener(move |s, _, _, cx| {
        f(s, cx);
        cx.notify();
    })
}

// ---------- drawing ----------

impl Render for Setup {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let th = self.theme();
        let step = self.step();
        let (body, foot) = match self.screen.clone() {
            _ if self.licenses => self.licenses_page(&th, cx),
            Screen::Loading => self.loading(&th),
            Screen::Choose => self.choose(&th, cx),
            Screen::Model => self.model_screen(&th, cx),
            Screen::Review { args, plan } => self.review_screen(&th, args, *plan, cx),
            Screen::Progress { title } => self.progress(&th, title),
            Screen::Done {
                uninstall,
                fresh,
                cli_tip,
                note,
            } => self.done(&th, uninstall, fresh, cli_tip, note, cx),
            Screen::Failed {
                failure,
                gpu,
                retry,
            } => self.failed_screen(&th, failure, gpu, retry, cx),
            Screen::Manage => self.manage_screen(&th, cx),
            Screen::Uninstall { plan } => self.uninstall_screen(&th, *plan, cx),
        };
        div()
            .size_full()
            .flex()
            .bg(th.app)
            .text_color(th.ink)
            .font_family("Figtree")
            .text_size(u(14.))
            .child(side(
                &th,
                step,
                div()
                    .id("licenses")
                    .underline()
                    .cursor_pointer()
                    .on_click(on(cx, |s, _| s.licenses = true))
                    .child("Licenses"),
            ))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .id("pane")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .px(u(40.))
                            .pt(u(36.))
                            .pb(u(20.))
                            .child(body),
                    )
                    .children(foot.map(|f| {
                        f.px(u(40.))
                            .pt(u(14.))
                            .pb(u(22.))
                            .border_t_1()
                            .border_color(th.line)
                    })),
            )
    }
}

impl Setup {
    /// Which of the fresh install's steps is on show, if any.
    fn step(&self) -> Option<usize> {
        if self.receipt.is_some() && !matches!(self.screen, Screen::Done { fresh: true, .. }) {
            return None;
        }
        match &self.screen {
            Screen::Choose => Some(0),
            Screen::Model => Some(1),
            Screen::Review { .. } => Some(2),
            Screen::Progress { .. } => Some(3),
            Screen::Done { fresh: true, .. } => Some(4),
            _ => None,
        }
    }

    /// The licenses of what this window is built from, over whatever screen is on show.
    fn licenses_page(&self, th: &Theme, cx: &mut Context<Self>) -> Page {
        let body = screen(
            th,
            "Licenses",
            "Larch is under the Apache License 2.0. This window includes the fonts below, \
             under their own license.",
        )
        .child(mono(FONT_LICENSE).text_color(th.ink2).line_height(u(18.)));
        let back = button(
            th,
            "licenses-back",
            "Back",
            Btn::Ghost,
            on(cx, |s, _| s.licenses = false),
        );
        (body, Some(footer(th, "", vec![back])))
    }

    fn loading(&self, th: &Theme) -> Page {
        let body = screen(
            th,
            "Reading this computer",
            "Setup is checking your graphics, memory and any copy of Larch already here.",
        );
        (body, None)
    }

    fn choose(&self, th: &Theme, cx: &mut Context<Self>) -> Page {
        let modify = self.receipt.is_some();
        let have = |k: &str| {
            self.receipt
                .as_ref()
                .is_some_and(|r| r.components.contains_key(k))
        };
        let plan = self.plan.as_ref();
        let size_of = |k: &str| {
            if have(k) {
                "Installed".to_owned()
            } else {
                (plan.and_then(|p| p.components.get(k)))
                    .map(|s| size(s.download))
                    .unwrap_or_default()
            }
        };
        let same = modify && self.app == have("app") && self.cli == have("cli");
        let ready = (self.app || self.cli) && !same && !self.checking;
        let mut parts = div().flex().flex_col().gap(u(10.));
        for (k, title, line) in PARTS {
            let on_now = if k == "app" { self.app } else { self.cli };
            parts = parts.child(option(
                th,
                format!("part-{k}"),
                false,
                on_now,
                title,
                line,
                size_of(k),
                on(cx, move |s, _| {
                    if k == "app" {
                        s.app = !s.app
                    } else {
                        s.cli = !s.cli
                    }
                }),
            ));
        }
        let mut d = screen(
            th,
            if modify {
                "Add or remove parts"
            } else {
                "Install Larch"
            },
            if modify {
                "Keep the parts you use. Your models and chats stay either way."
            } else {
                "Chat, write and code with models that run on this computer. Nothing you type \
                 leaves it."
            },
        );
        if let Some(p) = plan.filter(|_| !modify) {
            d = d.child(eyebrow(th, "This computer")).child(machine(
                th,
                &p.flavour,
                p.gpu.as_ref(),
            ));
        }
        d = d.child(eyebrow(th, "What to install")).child(parts);
        if let Some(p) = plan.filter(|_| !modify) {
            d = d.child(
                card(th)
                    .flex_row()
                    .items_center()
                    .gap(u(16.))
                    .py(u(14.))
                    .child(div().text_color(th.ink2).child("Install in"))
                    .child(mono(p.root.display().to_string()).text_size(u(13.))),
            );
        }
        let foot = if modify {
            ""
        } else {
            "For you only. No administrator rights needed."
        };
        let mut actions = Vec::new();
        if modify {
            actions.push(button(
                th,
                "back",
                "Back",
                Btn::Ghost,
                on(cx, |s, cx| s.manage(cx)),
            ));
        }
        actions.push(go(
            th,
            "go",
            if self.checking {
                "Checking…"
            } else {
                "Continue"
            },
            ready,
            on(cx, move |s, cx| {
                if ready {
                    s.chose(cx)
                }
            }),
        ));
        (d, Some(footer(th, foot, actions)))
    }

    fn model_screen(&self, th: &Theme, cx: &mut Context<Self>) -> Page {
        let models = self
            .plan
            .as_ref()
            .map(|p| p.models.clone())
            .unwrap_or_default();
        let picked = if self.model_picked {
            self.model.clone()
        } else {
            // The largest model this computer fits, as the app's own first-model card leads with.
            models
                .iter()
                .max_by_key(|m| m.min_mem_gb)
                .map(|m| m.file.clone())
        };
        let mut list = div().flex().flex_col().gap(u(10.));
        for (i, m) in models.iter().enumerate() {
            let file = m.file.clone();
            list = list.child(option(
                th,
                format!("model-{i}"),
                true,
                picked.as_ref() == Some(&m.file),
                &m.display,
                &format!("For computers with {} GB of memory or more", m.min_mem_gb),
                size(m.size),
                on(cx, move |s, _| {
                    s.model = Some(file.clone());
                    s.model_picked = true;
                }),
            ));
        }
        list = list.child(option(
            th,
            "model-skip".into(),
            true,
            picked.is_none(),
            "Skip for now",
            "Download one later from Models in the app.",
            String::new(),
            on(cx, |s, _| {
                s.model = None;
                s.model_picked = true;
            }),
        ));
        let license = (models.iter().find(|m| Some(&m.file) == picked.as_ref()))
            .map(|m| {
                format!(
                    "Downloading it accepts its license: {}",
                    m.license_url.trim_start_matches("https://")
                )
            })
            .unwrap_or_default();
        let body = screen(
            th,
            "Pick a first model",
            "Models that fit this computer. You can add others later in the app.",
        )
        .child(list)
        .child(small(th, license));
        let foot = footer(
            th,
            "",
            vec![
                button(
                    th,
                    "back",
                    "Back",
                    Btn::Ghost,
                    on(cx, |s, _| s.screen = Screen::Choose),
                ),
                go(
                    th,
                    "go",
                    if self.checking {
                        "Checking…"
                    } else {
                        "Continue"
                    },
                    !self.checking,
                    on(cx, move |s, cx| {
                        if !s.checking {
                            s.model = picked.clone();
                            s.model_picked = true;
                            s.review(cx)
                        }
                    }),
                ),
            ],
        );
        (body, Some(foot))
    }

    fn review_screen(
        &self,
        th: &Theme,
        args: Vec<String>,
        plan: Plan,
        cx: &mut Context<Self>,
    ) -> Page {
        let fresh = self.receipt.is_none();
        let model = if fresh { self.first_model() } else { None };
        let mut rows: Vec<(String, String)> = (plan.components.iter())
            .map(|(c, s)| (cap(&part_name(c)), size(s.download)))
            .collect();
        if let Some(m) = &model {
            rows.push((m.display.clone(), size(m.size)));
        }
        if let Some(at) = args.iter().position(|a| a == "--remove") {
            for k in &args[at + 1..] {
                rows.push((cap(&part_name(k)), "Remove".into()));
            }
        }
        let extra = model.as_ref().map_or(0, |m| m.size);
        let download = plan.download + extra;
        let need = plan.required + extra;
        let room = need <= plan.free;
        // Archives are compressed; while installing, the download and what it unpacks to both
        // sit on disk.
        let installed = plan.components.values().map(|s| s.unpack).sum::<u64>() + extra;
        let mut sum = card(th).gap(u(0.)).py(u(8.));
        for (i, (t, s)) in rows.iter().enumerate() {
            sum = sum.child(line(th, i > 0, t, s.clone(), false));
        }
        if download > 0 {
            sum = sum.child(line(th, true, "Download", size(download), true));
        }
        if installed > 0 {
            sum = sum.child(line(th, true, "Installed size", size(installed), false));
        }
        sum = sum.child(line(
            th,
            true,
            "Space needed while installing",
            format!("{} of {} free", size(need), size(plan.free)),
            false,
        ));
        let build = match plan.flavour.as_str() {
            "cpu" => ", the processor build",
            "cuda" => ", the NVIDIA build",
            "metal" => ", the Apple silicon build",
            "hip" => ", the AMD build",
            _ => "",
        };
        let label = if fresh { "Install" } else { "Apply" };
        let apply_args = args.clone();
        let p = plan.clone();
        let body = screen(
            th,
            if fresh {
                "Ready to install"
            } else {
                "Ready to change"
            },
            &format!("Larch {}{build}.", plan.version),
        )
        .child(sum)
        .when(!room, |d| {
            d.child(small(
                th,
                "There is not enough free space here. Free some up, then try again.",
            ))
        })
        .child(small(
            th,
            "Every download is checked against the Larch release key before anything is installed.",
        ));
        let foot = footer(
            th,
            "",
            vec![
                button(
                    th,
                    "back",
                    "Back",
                    Btn::Ghost,
                    on(cx, move |s, cx| {
                        if s.receipt.is_some() {
                            s.screen = Screen::Choose
                        } else if s.plan.as_ref().is_some_and(|p| !p.models.is_empty()) {
                            s.screen = Screen::Model
                        } else {
                            s.screen = Screen::Choose
                        }
                        cx.notify()
                    }),
                ),
                go(
                    th,
                    "go",
                    label,
                    room,
                    on(cx, move |s, cx| {
                        if room {
                            let model = if s.receipt.is_none() {
                                s.first_model()
                            } else {
                                None
                            };
                            s.apply(apply_args.clone(), Some(&p), model, cx)
                        }
                    }),
                ),
            ],
        );
        (body, Some(foot))
    }

    fn progress(&self, th: &Theme, title: &str) -> Page {
        let r = &self.run;
        let share = r.share();
        let now = r.tasks.get(r.at);
        let bytes: Vec<&Task> = r.tasks.iter().filter(|t| t.id != "apply").collect();
        let got: f64 = bytes.iter().map(|t| t.done.min(t.total)).sum();
        let all: f64 = bytes.iter().map(|t| t.total).sum();
        let status = match now {
            None => "Done".to_owned(),
            Some(t) if t.id == "apply" => format!("{}…", t.label),
            Some(_) => format!("{} of {}", size(got as u64), size(all as u64)),
        };
        let mut list = card(th).gap(u(0.)).py(u(6.));
        for (i, t) in r.tasks.iter().enumerate() {
            let state = (i < r.at, i == r.at);
            let right = if state.1 && t.id != "apply" && t.done > 0. {
                format!("{}%", ((t.done.min(t.total) / t.total) * 100.).floor())
            } else if t.id != "apply" {
                size(t.total as u64)
            } else {
                String::new()
            };
            list = list.child(
                div()
                    .flex()
                    .items_center()
                    .gap(u(12.))
                    .py(u(11.))
                    .when(i > 0, |d| d.border_t_1().border_color(th.line))
                    .child(tick(th, state.0, state.1))
                    .child(
                        div()
                            .flex_1()
                            .when(state.1, |d| d.font_weight(FontWeight::SEMIBOLD))
                            .text_color(if state.0 || state.1 { th.ink } else { th.ink2 })
                            .child(t.label.clone()),
                    )
                    .child(div().text_color(th.ink2).child(right)),
            );
        }
        let body = screen(
            th,
            title,
            "Keep this window open until it finishes. Closing it stops setup, and setup puts \
             everything back.",
        )
        .child(bar(th, &[(share, th.sun.into())], 6.))
        .child(
            div()
                .flex()
                .justify_between()
                .text_color(th.ink2)
                .child(status)
                .child(format!("{}%", (share * 100.).floor())),
        )
        .child(list);
        (body, None)
    }

    fn done(
        &self,
        th: &Theme,
        uninstall: bool,
        fresh: bool,
        cli_tip: bool,
        note: Option<String>,
        cx: &mut Context<Self>,
    ) -> Page {
        if uninstall {
            let text = if self.purge {
                "Your models, chats and memory were deleted too."
            } else {
                "Your models, chats and memory are still on this computer, for next time."
            };
            return (
                screen(th, "Larch was removed", text),
                Some(footer(th, "", vec![close(th, Btn::Primary, cx)])),
            );
        }
        let app = self
            .receipt
            .as_ref()
            .is_some_and(|r| r.components.contains_key("app"));
        let place = if cfg!(windows) {
            "the Start menu"
        } else if cfg!(target_os = "macos") {
            "Applications"
        } else {
            "your app menu"
        };
        let mut text = String::new();
        if app {
            text += &format!("Find Larch in {place} any time.");
        }
        let mut d = screen(th, if fresh { "Larch is ready" } else { "All done" }, &text);
        if cli_tip {
            d = d.child(
                div()
                    .flex()
                    .items_center()
                    .gap(u(8.))
                    .text_color(th.ink2)
                    .child("In a new terminal, try")
                    .child(mono("larch chat").text_size(u(13.)).text_color(th.ink)),
            );
        }
        if let Some(n) = note {
            d = d.child(small(th, n));
        }
        let mut actions = vec![close(th, if app { Btn::Ghost } else { Btn::Primary }, cx)];
        if app {
            actions.push(button(
                th,
                "open",
                "Open Larch",
                Btn::Primary,
                on(cx, |s, cx| s.open_app(cx)),
            ));
        }
        (d, Some(footer(th, "", actions)))
    }

    fn failed_screen(
        &self,
        th: &Theme,
        failure: Failure,
        gpu: bool,
        retry: Option<Retry>,
        cx: &mut Context<Self>,
    ) -> Page {
        let (title, text) = if gpu {
            (
                "Your graphics card didn't pass the check",
                "Setup put everything back as it was. Larch can run on the processor instead: \
                 slower, but it works.",
            )
        } else if failure.busy {
            (
                "Close Larch first",
                "Larch is open, so setup can't change it yet. Close it, then try again.",
            )
        } else {
            (
                "Setup couldn't finish",
                "Nothing is left half done, so trying again is safe.",
            )
        };
        let mut d = screen(th, title, text);
        if !failure.busy {
            // larch-setup's own message, without its internal state suffix.
            let message = failure.message.split("; state:").next().unwrap_or_default();
            d = d.child(
                div()
                    .px(u(14.))
                    .py(u(10.))
                    .rounded(u(9.))
                    .bg(th.sunk)
                    .text_color(th.ink2)
                    .child(message.to_owned()),
            );
        }
        let mut actions = vec![close(th, Btn::Ghost, cx)];
        if let Some(r) = retry {
            let label = if gpu {
                "Install the processor build"
            } else {
                "Try again"
            };
            actions.push(button(
                th,
                "retry",
                label,
                Btn::Primary,
                on(cx, move |s, cx| s.retry(r.clone(), cx)),
            ));
        }
        (d, Some(footer(th, "", actions)))
    }

    fn manage_screen(&self, th: &Theme, cx: &mut Context<Self>) -> Page {
        let Some(r) = &self.receipt else {
            return self.loading(th);
        };
        let c = &r.components;
        let version = (c.get("app").or(c.get("cli")).or(c.get("setup")))
            .map(|c| c.version.clone())
            .unwrap_or_default();
        let runs = match (r.detect.flavour.as_str(), &r.detect.gpu) {
            ("cpu", _) => "running on the processor".to_owned(),
            (_, Some(g)) => format!("using {g}"),
            _ => "using this computer's graphics".to_owned(),
        };
        let has: Vec<&str> = (PARTS.iter())
            .filter(|(k, ..)| c.contains_key(*k))
            .map(|(k, ..)| {
                if *k == "app" {
                    "the app"
                } else {
                    "the command line"
                }
            })
            .collect();
        let mut d = screen(
            th,
            "Larch is installed",
            &format!("Version {version}, {runs}."),
        );
        if let Some(Some(u_)) = &self.check
            && let Some(v) = &u_.version
        {
            d = d.child(
                card(th)
                    .flex_row()
                    .items_center()
                    .gap(u(12.))
                    .border_color(th.accent)
                    .child(icon("down", 18., th.accent))
                    .child(div().flex_1().child(format!(
                        "Larch {v} is ready to install, {}.",
                        size(u_.download_size)
                    )))
                    .child(button(
                        th,
                        "update",
                        "Update",
                        Btn::Primary,
                        on(cx, |s, cx| s.straight("update", cx)),
                    )),
            );
        }
        let app = c.contains_key("app");
        let mut acts = card(th).gap(u(0.)).py(u(4.));
        let mut row =
            |id: &'static str, ic: &str, t: &str, line_: String, f: Action, first: bool| {
                div()
                    .id(id)
                    .flex()
                    .items_center()
                    .gap(u(14.))
                    .py(u(12.))
                    .cursor_pointer()
                    .when(!first, |d| d.border_t_1().border_color(th.line))
                    .hover(|s| s.bg(fade(th.accent, 0.05)))
                    .on_click(on(cx, move |s, cx| f(s, cx)))
                    .child(icon(ic, 18., th.ink2))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .child(div().font_weight(FontWeight::SEMIBOLD).child(t.to_owned()))
                            .child(div().text_size(u(13.)).text_color(th.ink2).child(line_)),
                    )
            };
        if app {
            acts = acts.child(row(
                "act-open",
                "larch",
                "Open Larch",
                "Start the app.".into(),
                Box::new(|s, cx| s.open_app(cx)),
                true,
            ));
        }
        acts = acts
            .child(row(
                "act-modify",
                "plus",
                "Add or remove parts",
                format!(
                    "Installed: {}.",
                    if has.is_empty() {
                        "setup only".to_owned()
                    } else {
                        has.join(" and ")
                    }
                ),
                Box::new(|s, _| {
                    let c = s
                        .receipt
                        .as_ref()
                        .map(|r| r.components.clone())
                        .unwrap_or_default();
                    s.app = c.contains_key("app");
                    s.cli = c.contains_key("cli");
                    s.screen = Screen::Choose;
                }),
                !app,
            ))
            .child(row(
                "act-repair",
                "settings",
                "Repair",
                "Put back anything missing or damaged.".into(),
                Box::new(|s, cx| s.straight("repair", cx)),
                false,
            ))
            .child(row(
                "act-uninstall",
                "trash",
                "Uninstall",
                "Remove Larch from this computer.".into(),
                Box::new(|s, cx| s.uninstall(cx)),
                false,
            ));
        let line_ = match &self.check {
            None => "Checking for updates…",
            Some(None) => "Couldn't check for updates.",
            Some(Some(u_)) if u_.version.is_none() => "You have the latest version.",
            Some(Some(_)) => "",
        };
        (d.child(acts).child(small(th, line_)), None)
    }

    fn uninstall_screen(&self, th: &Theme, plan: Plan, cx: &mut Context<Self>) -> Page {
        let what = if plan.data_size > 0 {
            format!("{} of models and everything else in", size(plan.data_size))
        } else {
            "Everything in".to_owned()
        };
        let purge = self.purge;
        let body = screen(
            th,
            "Uninstall Larch?",
            "This removes the app, the command line and setup from this computer.",
        )
        .child(
            card(th)
                .id("purge")
                .flex_row()
                .items_center()
                .gap(u(14.))
                .cursor_pointer()
                .on_click(on(cx, |s, _| s.purge = !s.purge))
                .child(
                    div()
                        .flex_1()
                        .flex()
                        .flex_col()
                        .child(
                            div()
                                .font_weight(FontWeight::SEMIBOLD)
                                .child("Also delete my models, chats and memory"),
                        )
                        .child(
                            div()
                                .text_size(u(13.))
                                .text_color(th.ink2)
                                .child(format!("{what} {}", plan.data.display())),
                        ),
                )
                .child(switch(th, purge)),
        );
        let foot = footer(
            th,
            "",
            vec![
                button(
                    th,
                    "cancel",
                    "Cancel",
                    Btn::Ghost,
                    on(cx, |s, cx| s.manage(cx)),
                ),
                button(
                    th,
                    "go",
                    "Uninstall",
                    Btn::Danger,
                    on(cx, move |s, cx| {
                        let mut args = vec!["uninstall".to_owned()];
                        if s.purge {
                            args.push("--purge".into());
                        }
                        s.apply(args, Some(&plan), None, cx)
                    }),
                ),
            ],
        );
        (body, Some(foot))
    }
}

// ---------- pieces ----------

/// The side: the Larch mark and name, the fresh install's steps, and the license line with the
/// link to the licenses page.
fn side(th: &Theme, step: Option<usize>, licenses: impl IntoElement) -> Div {
    let mut steps = div().flex().flex_col().gap(u(6.));
    if let Some(at) = step {
        for (i, s) in STEPS.iter().enumerate() {
            let done = i < at;
            let now = i == at;
            steps = steps.child(
                div()
                    .flex()
                    .items_center()
                    .gap(u(12.))
                    .px(u(12.))
                    .py(u(9.))
                    .rounded(u(10.))
                    .when(now, |d| d.bg(th.cur))
                    .child(if done {
                        div()
                            .size(u(22.))
                            .rounded_full()
                            .bg(th.accent)
                            .flex()
                            .items_center()
                            .justify_center()
                            .child(icon("check", 13., th.on_accent))
                    } else {
                        div()
                            .size(u(22.))
                            .rounded_full()
                            .border_1()
                            .border_color(if now { th.ink } else { th.line })
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_size(u(12.))
                            .text_color(if now { th.ink } else { th.ink3 })
                            .child((i + 1).to_string())
                    })
                    .child(
                        div()
                            .when(now, |d| d.font_weight(FontWeight::SEMIBOLD))
                            .text_color(if now || done { th.ink } else { th.ink2 })
                            .child(*s),
                    ),
            );
        }
    }
    div()
        .w(u(230.))
        .flex_none()
        .h_full()
        .flex()
        .flex_col()
        .gap(u(30.))
        .px(u(22.))
        .pt(u(34.))
        .pb(u(24.))
        .bg(th.side)
        .border_r_1()
        .border_color(th.line)
        .child(
            div()
                .flex()
                .items_center()
                .gap(u(10.))
                .child(print("larch", 26., th.sun, th.ink))
                .child(serif("Larch", 28.)),
        )
        .child(steps)
        .child(div().flex_1())
        .child(
            div()
                .flex()
                .flex_col()
                .gap(u(8.))
                .text_size(u(12.5))
                .text_color(th.ink3)
                .child("Free and open source under the Apache License 2.0.")
                .child(licenses),
        )
}

/// A screen's title and line, with its content below.
fn screen(th: &Theme, title: &str, text: &str) -> Div {
    div()
        .flex()
        .flex_col()
        .gap(u(16.))
        .child(serif(title.to_owned(), 32.))
        .when(!text.is_empty(), |d| {
            d.child(
                div()
                    .text_size(u(15.))
                    .text_color(th.ink2)
                    .line_height(u(22.))
                    .child(text.to_owned()),
            )
        })
}

/// A quiet line under the content.
fn small(th: &Theme, text: impl Into<SharedString>) -> Div {
    div()
        .text_size(u(13.))
        .text_color(th.ink2)
        .child(text.into())
}

/// The note and buttons at a screen's foot.
fn footer(th: &Theme, note: &str, actions: Vec<gpui::Stateful<Div>>) -> Div {
    div()
        .flex()
        .items_center()
        .gap(u(10.))
        .child(
            div()
                .flex_1()
                .text_size(u(13.))
                .text_color(th.ink2)
                .child(note.to_owned()),
        )
        .children(actions)
}

/// The primary button, dimmed while it cannot be pressed.
fn go(
    th: &Theme,
    id: &'static str,
    label: &str,
    ready: bool,
    f: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> gpui::Stateful<Div> {
    button(th, id, label.to_owned(), Btn::Primary, f).when(!ready, |b| b.opacity(0.45))
}

fn close(th: &Theme, kind: Btn, cx: &mut Context<Setup>) -> gpui::Stateful<Div> {
    button(
        th,
        "close",
        "Close",
        kind,
        cx.listener(|_, _, _, cx| cx.quit()),
    )
}

/// What Larch will run on: the graphics card, Apple silicon, or the processor.
fn machine(th: &Theme, flavour: &str, gpu: Option<&Gpu>) -> Div {
    let (ic, name, detail, build) = match (flavour, gpu) {
        ("cpu", _) => (
            "gpu-chip",
            "No supported graphics card".to_owned(),
            "Larch will run on the processor. It works, just slower.".to_owned(),
            "Processor build",
        ),
        ("metal", g) => (
            "gpu-chip",
            g.map_or("This Mac".to_owned(), |g| g.name.clone()),
            "Larch will use this Mac's graphics.".to_owned(),
            "Metal build",
        ),
        (f, g) => {
            let mut detail = Vec::new();
            if let Some(g) = g.filter(|g| g.memory > 0) {
                detail.push(format!(
                    "{} GB",
                    (g.memory as f64 / (1u64 << 30) as f64).round()
                ));
            }
            if let Some(d) = g.and_then(|g| g.driver.as_ref()) {
                detail.push(format!("driver {d}"));
            }
            (
                "gpu-card",
                g.map_or("Your graphics card".to_owned(), |g| g.name.clone()),
                detail.join(" · "),
                if f == "cuda" {
                    "CUDA build"
                } else {
                    "GPU build"
                },
            )
        }
    };
    card(th)
        .flex_row()
        .items_center()
        .gap(u(14.))
        .child(
            div()
                .size(u(44.))
                .flex_none()
                .rounded(u(10.))
                .bg(th.sunk)
                .flex()
                .items_center()
                .justify_center()
                .child(print(ic, 24., th.sun, th.ink)),
        )
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .child(div().font_weight(FontWeight::SEMIBOLD).child(name))
                .child(div().text_size(u(13.)).text_color(th.ink2).child(detail)),
        )
        .child(
            div()
                .flex_none()
                .px(u(12.))
                .py(u(4.))
                .rounded_full()
                .border_1()
                .border_color(th.keep)
                .text_color(th.keep)
                .text_size(u(13.))
                .child(build),
        )
}

/// A choice row: a check box (or a radio when `radio`), a title and line, and a size.
#[allow(clippy::too_many_arguments)]
fn option(
    th: &Theme,
    id: String,
    radio: bool,
    on_: bool,
    title: &str,
    line_: &str,
    right: String,
    f: impl Fn(&ClickEvent, &mut Window, &mut App) + 'static,
) -> AnyElement {
    let mark = if radio {
        div()
            .size(u(20.))
            .flex_none()
            .rounded_full()
            .border_2()
            .border_color(if on_ { th.accent } else { th.line })
            .flex()
            .items_center()
            .justify_center()
            .when(on_, |d| {
                d.child(div().size(u(10.)).rounded_full().bg(th.accent))
            })
    } else {
        div()
            .size(u(20.))
            .flex_none()
            .rounded(u(5.))
            .border_2()
            .border_color(if on_ { th.accent } else { th.line })
            .when(on_, |d| d.bg(th.accent))
            .flex()
            .items_center()
            .justify_center()
            .when(on_, |d| d.child(icon("check", 12., th.on_accent)))
    };
    div()
        .id(SharedString::from(id))
        .flex()
        .items_center()
        .gap(u(14.))
        .px(u(18.))
        .py(u(13.))
        .rounded(u(14.))
        .border_2()
        .border_color(if on_ { th.accent } else { th.line })
        .bg(th.card)
        .cursor_pointer()
        .on_click(f)
        .child(mark)
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .child(
                    div()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(title.to_owned()),
                )
                .child(
                    div()
                        .text_size(u(13.))
                        .text_color(th.ink2)
                        .child(line_.to_owned()),
                ),
        )
        .child(div().flex_none().text_color(th.ink2).child(right))
        .into_any_element()
}

/// A summary row: label and value, ruled off from the row above.
fn line(th: &Theme, ruled: bool, label: &str, value: String, strong: bool) -> Div {
    div()
        .flex()
        .justify_between()
        .gap(u(16.))
        .py(u(10.))
        .when(ruled, |d| d.border_t_1().border_color(th.line))
        .when(strong, |d| d.font_weight(FontWeight::SEMIBOLD))
        .child(
            div()
                .text_color(if strong { th.ink } else { th.ink2 })
                .child(label.to_owned()),
        )
        .child(value)
}

/// A progress row's mark: a filled tick when done, a gold ring while running, a faint ring after.
fn tick(th: &Theme, done: bool, now: bool) -> Div {
    let d = div()
        .size(u(22.))
        .flex_none()
        .rounded_full()
        .flex()
        .items_center()
        .justify_center();
    if done {
        d.bg(th.accent).child(icon("check", 13., th.on_accent))
    } else {
        d.border_2()
            .border_color(if now { th.sun } else { th.line })
    }
}

/// An on/off switch.
fn switch(th: &Theme, on_: bool) -> Div {
    div()
        .w(u(38.))
        .h(u(22.))
        .flex_none()
        .rounded_full()
        .p(u(3.))
        .flex()
        .when(on_, |d| d.justify_end())
        .bg(if on_ { th.lose } else { th.line })
        .child(div().size(u(16.)).rounded_full().bg(th.card))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(ids: &[(&str, f64)]) -> Run {
        Run {
            tasks: ids
                .iter()
                .map(|(id, total)| Task {
                    id: id.to_string(),
                    label: id.to_string(),
                    total: *total,
                    done: 0.,
                })
                .collect(),
            ..Default::default()
        }
    }

    /// A later row's progress finishes every row before it, and a finished row moves on.
    /// The window carries the license of every font built into it, as the OFL asks.
    #[test]
    fn the_licenses_page_holds_the_license_of_every_built_in_font() {
        assert!(FONT_LICENSE.contains("SIL Open Font License, Version 1.1"));
        for font in crate::FILES.iter().filter_map(|f| f.strip_prefix("fonts/")) {
            let family = font
                .split('-')
                .next()
                .unwrap()
                .replace("JetBrainsMono", "JetBrains Mono");
            assert!(
                FONT_LICENSE.contains(&format!("The {family} Project Authors")),
                "{font}"
            );
        }
    }

    #[test]
    fn progress_moves_down_the_rows_by_bytes() {
        let mut r = run(&[("app", 100.), ("cli", 300.), ("apply", 32.)]);
        r.step("app", 50., 100.);
        assert_eq!((r.at, r.share()), (0, 50. / 432.));
        r.step("cli", f64::INFINITY, 0.);
        assert_eq!(r.at, 2);
        assert_eq!(r.tasks[0].done, 100.);
        r.step("apply", 1., 1.);
        assert_eq!(r.at, 3);
        // An event for a component the plan did not list changes nothing.
        r.step("cuda", 5., 10.);
        assert_eq!(r.at, 3);
    }

    #[test]
    fn sizes_read_as_people_say_them() {
        assert_eq!(size(17_117_951_808), "17.1 GB");
        assert_eq!(size(96_400_000), "96 MB");
        assert_eq!(size(10), "1 MB");
        assert_eq!(size(0), "0 MB");
    }

    /// The plan a `--dry-run` prints reads, models and all.
    #[test]
    fn a_dry_run_plan_reads() {
        let plan: Plan = serde_json::from_value(serde_json::json!({
            "version": "0.1.0", "flavour": "cuda", "why": "",
            "gpu": {"vendor": "nvidia", "name": "GPU", "memory": 17179869184u64, "driver": "581.57",
                    "cc": "12.0", "gfx": null, "pnp": null, "verified": true},
            "components": {"app": {"download": 1, "unpack": 2}},
            "download": 1, "required": 3, "free": 9, "root": "/r", "data": "/d", "data_size": 0,
            "models": [{"display": "M", "repo": "o/r", "file": "m.gguf", "revision": "a",
                        "sha256": "b", "size": 4, "min_mem_gb": 8, "license_url": "https://l"}]
        }))
        .unwrap();
        assert_eq!(plan.models[0].display, "M");
        assert_eq!(plan.components["app"].unpack, 2);
    }
}
