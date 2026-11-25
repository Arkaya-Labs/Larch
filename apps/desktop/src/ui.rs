//! The window: its state, the core's events, the shell (sidebar, top bar, the assistant chip and
//! its popover, toasts, the GPU-stopped screen), Home, Chats and a chat. The other pages are in
//! pages.rs.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use futures::channel::mpsc::UnboundedReceiver;
use gpui::{
    AnyElement, App, AppContext, BoxShadow, ClipboardItem, Context, Div, Entity, ExternalPaths,
    FocusHandle, Focusable, FontWeight, Hsla, InteractiveElement, IntoElement, KeyBinding,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ObjectFit, ParentElement,
    PathPromptOptions, Pixels, Point, PromptLevel, Render, Rgba, ScrollHandle, SharedString,
    Stateful, StatefulInteractiveElement, Styled, StyledImage, StyledText, Subscription, TextRun,
    Window, WindowAppearance, actions, div, font, img, point, prelude::FluentBuilder, px,
};
use larch_agent::tool::no_window;
use larch_app::Event;
use larch_app::contract::*;
use serde_json::Value;

use crate::images;
// The shared pieces stay reachable as `crate::ui::*`, as every page imports them.
use crate::input::{PastedFiles, PastedImage, Submitted, TextInput};
pub use crate::kit::*;
use crate::logic::{self, Block, Span, Task};
use crate::picker;
use crate::prefs::{Mode, Motion, Prefs};
use crate::select::{self, Selection};
use crate::theme::{NOTE, NOTE_EYEBROW, NOTE_INK, Theme, fade};
use crate::tree::{Mood, Tree, tree};

actions!(
    ecl,
    [
        NewChat,
        GoHome,
        GoChats,
        GoTasks,
        GoModels,
        GoTools,
        GoSettings,
        StopReply,
        Dismiss,
        FocusComposer,
        ToggleSidePanel,
        Quit
    ]
);

/// The window's key bindings; `cmd` is ctrl off macOS.
pub fn bindings() -> Vec<KeyBinding> {
    let m = if cfg!(target_os = "macos") {
        "cmd"
    } else {
        "ctrl"
    };
    let k = |keys: &str| keys.replace('M', m);
    vec![
        KeyBinding::new(&k("M-n"), NewChat, Some("Ecl")),
        KeyBinding::new(&k("M-1"), GoHome, Some("Ecl")),
        KeyBinding::new(&k("M-2"), GoChats, Some("Ecl")),
        // In the sidebar's order.
        KeyBinding::new(&k("M-3"), GoTasks, Some("Ecl")),
        KeyBinding::new(&k("M-4"), GoModels, Some("Ecl")),
        KeyBinding::new(&k("M-5"), GoTools, Some("Ecl")),
        KeyBinding::new(&k("M-,"), GoSettings, Some("Ecl")),
        KeyBinding::new(&k("M-."), StopReply, Some("Ecl")),
        KeyBinding::new(&k("M-l"), FocusComposer, Some("Ecl")),
        KeyBinding::new(&k("M-alt-b"), ToggleSidePanel, Some("Ecl")),
        KeyBinding::new("escape", Dismiss, Some("Ecl")),
        // App-wide, so it quits wherever the focus is.
        KeyBinding::new(&k("M-q"), Quit, None),
    ]
}

/// The shortcuts as Settings lists them.
pub const SHORTCUTS: &[(&str, &str)] = &[
    ("M-n", "New chat"),
    ("M-l", "Go to the message box"),
    ("Enter", "Send"),
    ("Shift-Enter", "New line"),
    ("M-.", "Stop the reply"),
    ("M-1 … M-5", "Home, Chats, Tasks, Models, Tools"),
    ("M-,", "Settings"),
    ("M-Alt-B", "Show or hide a chat's side panel (Advanced)"),
    ("M-q", "Quit"),
    ("Esc", "Close a popover or a dialog"),
];

/// The sidebar's width, in `u`.
const SIDEBAR: f32 = 248.;
/// The page width, in `u`, a chat needs to show its side panel beside a 560 wide conversation.
const SIDE_PANEL_ROOM: f32 = 860.;

/// Unix milliseconds now.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// A page of the window.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Page {
    Home,
    Chats,
    Tasks,
    Chat,
    Models,
    Engine,
    /// MCP servers: their tools join every new chat.
    Mcp,
    Connect,
    Settings,
}

/// The Models page's tabs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MTab {
    Library,
    Discover,
}

/// Settings' tabs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum STab {
    General,
    Appearance,
    Assistant,
    Models,
    Privacy,
    Shortcuts,
    Advanced,
    /// What the assistant remembers; shown with Advanced on.
    Memory,
}

/// Where a tool call stands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepState {
    Run,
    Wait,
    Ok,
    Fail,
    No,
    Stop,
}

/// One tool call in a reply.
#[derive(Clone, Debug)]
pub struct Step {
    pub name: String,
    pub args: Value,
    pub now: String,
    pub done: String,
    pub sum: String,
    pub st: StepState,
    pub count: u32,
}

/// The end of a code run, as the core sent it.
#[derive(Clone, Debug)]
pub struct Packet {
    pub title: String,
    pub body: String,
    pub diff: String,
    pub before: Option<Tests>,
    pub after: Option<Tests>,
    pub scope: Vec<ScopeEntry>,
    pub outside: u32,
    pub stop: Option<String>,
    pub checkpoint: Option<String>,
    pub decided: Option<bool>,
}

/// An image in a sent message: this window's copy, or a saved chat's, read by its reference
/// (`App::chat_image`) into `Ecl::pics`.
#[derive(Clone, Debug)]
pub enum Pic {
    Live(Arc<gpui::Image>),
    Saved(String),
}

/// An image added for the next message: its reference with the core (what `agent_send` names),
/// the image its chip shows, the prompt rows it takes on the loaded model, and whether it is on
/// its way in a sent message (its chip hides until the core takes the message, and shows again
/// if the core refuses it).
pub struct Chip {
    pub reference: String,
    pub image: Arc<gpui::Image>,
    pub tokens: Option<u32>,
    pub sending: bool,
}

/// A piece of a chat.
#[derive(Clone, Debug)]
pub enum Item {
    User {
        text: String,
        files: Vec<String>,
        images: Vec<Pic>,
    },
    Steps(Vec<Step>),
    Text {
        text: String,
        cut: bool,
    },
    Notice(String),
    Saved {
        receipt: Receipt,
        line: Option<String>,
        again: bool,
    },
    Packet(Box<Packet>),
    Error(String),
    /// The helpers a large request was split into.
    Crew(crate::crew::Crew),
    /// The model's reasoning for the reply after it, folded until a click opens it.
    // ponytail: shown live only; a saved chat keeps the answer, not its reasoning (`saved_items`).
    Reasoning {
        text: String,
        open: bool,
    },
}

/// Marks the chat's newest packet decided, or undecided with None.
fn set_decided(t: &mut Thread, yes: Option<bool>) {
    if let Some(Item::Packet(p)) = t
        .items
        .iter_mut()
        .rev()
        .find(|i| matches!(i, Item::Packet(_)))
    {
        p.decided = yes;
    }
}

/// What a chat waits on the user for.
#[derive(Clone, Debug)]
pub enum Ask {
    Confirm {
        name: String,
        args: Value,
    },
    Code {
        name: String,
        args: Value,
        path: Option<String>,
        folder: Option<String>,
    },
    Packet,
}

/// The page view a cite opened: the page's text, with the quote on it highlighted.
pub struct DocView {
    /// The page ("report.pdf, p. 3").
    pub title: String,
    pub page: String,
    /// The bytes of `page` to highlight.
    pub hit: Option<std::ops::Range<usize>>,
}

/// A reply's numbers, for the side panel.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub tok_s: f64,
    pub ttft_ms: f64,
    pub reused: f64,
}

/// The chat tree's size at the top of a chat.
const SUN_BIG: f32 = 92.;
/// The pinned banner's height.
const BANNER_H: f32 = 44.;
/// How a failed start's message begins (`Ecl::failed`), so its screen is not a GPU fault's.
const NO_START: &str = "Larch could not start its engine";

/// The app colour fading out over `h` px below a pinned tree, eased in three steps so text slides
/// under it rather than meeting a line.
fn soft_edge(th: &Theme, h: f32) -> Div {
    let step = |a: f32, b: f32, f: f32| {
        div().h(u(h * f)).bg(gpui::linear_gradient(
            180.,
            gpui::linear_color_stop(fade(th.app, a), 0.),
            gpui::linear_color_stop(fade(th.app, b), 1.),
        ))
    };
    div()
        .child(step(1., 0.82, 0.25))
        .child(step(0.82, 0.35, 0.35))
        .child(step(0.35, 0., 0.4))
}

/// How the chat's tree draws: above the messages, or pinned at the top as a one-line banner once
/// they scroll it away.
#[derive(Clone, Copy, PartialEq)]
enum SunForm {
    Big,
    Banner,
}

/// A chat this window has open or has heard from.
pub struct Thread {
    pub title: String,
    pub items: Vec<Item>,
    pub live: bool,
    pub ask: Option<Ask>,
    pub files: Vec<String>,
    pub repo: Option<PathBuf>,
    pub progress: Option<String>,
    /// The chip of the chat's pinned documents and the last answer's first word.
    pub docs: logic::Docs,
    /// The cites of each answer (by its item), checked against the chat's documents.
    pub cites: HashMap<usize, Vec<logic::CiteLink>>,
    pub stats: Option<Stats>,
    pub task: Option<Task>,
    pub updated: u64,
    pub loaded: bool,
    pub scroll: ScrollHandle,
}

impl Thread {
    /// A chat with nothing in it yet.
    pub fn new(title: &str) -> Thread {
        Thread {
            title: title.to_owned(),
            items: Vec::new(),
            live: false,
            ask: None,
            files: Vec::new(),
            repo: None,
            progress: None,
            docs: logic::Docs::default(),
            cites: HashMap::new(),
            stats: None,
            task: None,
            updated: now_ms(),
            loaded: false,
            scroll: ScrollHandle::new(),
        }
    }

    /// The newest group of steps, wherever it is: a memory receipt can land between a call and its
    /// result, and a new group there would leave the call waiting for good.
    fn last_steps(&mut self) -> Option<&mut Vec<Step>> {
        self.items.iter_mut().rev().find_map(|i| match i {
            Item::Steps(s) => Some(s),
            _ => None,
        })
    }

    fn steps(&mut self) -> &mut Vec<Step> {
        if !matches!(self.items.last(), Some(Item::Steps(_))) {
            self.items.push(Item::Steps(Vec::new()));
        }
        match self.items.last_mut() {
            Some(Item::Steps(s)) => s,
            _ => unreachable!("just pushed"),
        }
    }

    /// Where the live reply starts: the item after the user's last message.
    fn reply(&self) -> usize {
        (self.items.iter())
            .rposition(|i| matches!(i, Item::User { .. }))
            .map_or(0, |at| at + 1)
    }

    /// Marks the step of the call `name` that just returned `summary` done, failed or declined.
    fn tool_result(&mut self, name: String, summary: String) {
        // The step may sit before what its call showed since: its helpers, its readers' text.
        let reply = self.reply();
        let found = (self.items[reply..].iter_mut().rev()).find_map(|it| match it {
            Item::Steps(steps) => {
                let i = steps.iter().rposition(|s| {
                    s.name == name
                        && matches!(s.st, StepState::Run | StepState::Wait | StepState::Stop)
                })?;
                Some((steps, i))
            }
            _ => None,
        });
        let Some((steps, i)) = found else { return };
        let s = &mut steps[i];
        (s.st, s.sum) = if summary == "declined" {
            (StepState::No, "you said not now".into())
        } else if summary.starts_with("failed") {
            (StepState::Fail, summary)
        } else {
            (StepState::Ok, logic::tidy_sum(&name, &summary))
        };
        // The same call with the same result again is one step with a count.
        if i + 1 == steps.len() && i > 0 {
            let (a, b) = steps.split_at_mut(i);
            let p = &mut a[i - 1];
            if p.name == b[0].name && p.st == b[0].st && p.sum == b[0].sum && p.args == b[0].args {
                p.count += 1;
                steps.pop();
            }
        }
    }

    /// Adds the reply's next words to the text it is writing, or starts one.
    fn delta(&mut self, text: String) {
        match self.items.last_mut() {
            Some(Item::Text { text: s, .. }) => s.push_str(&text),
            _ => self.items.push(Item::Text { text, cut: false }),
        }
    }

    /// Adds the next piece of a step's reasoning, which comes before its words, or starts one.
    fn reasoning(&mut self, text: String) {
        match self.items.last_mut() {
            Some(Item::Reasoning { text: s, .. }) => s.push_str(&text),
            // A block's leading blank lines start nothing, so an empty block shows no row.
            _ if text.trim().is_empty() => {}
            _ => self.items.push(Item::Reasoning { text, open: false }),
        }
    }

    /// The reply's helpers as they now stand: the first list starts its crew.
    fn set_helpers(&mut self, helpers: Vec<crate::crew::Helper>) {
        let reply = self.reply();
        let crew = self.items[reply..].iter_mut().rev().find_map(|i| match i {
            Item::Crew(c) => Some(c),
            _ => None,
        });
        match crew {
            // An empty list only starts a turn (`Agent::helpers`): it never ends this reply's.
            _ if helpers.is_empty() => {}
            Some(c) => c.helpers = helpers,
            // Before text already streaming (the agent can send its first words before the
            // helpers that started them), so the next words still join that text.
            None => {
                let at = self.items.len()
                    - usize::from(matches!(self.items.last(), Some(Item::Text { .. })));
                self.items
                    .insert(at, Item::Crew(crate::crew::Crew::new(helpers)));
            }
        }
    }

    /// Ends what still runs once the turn has ended, so nothing spins after it.
    fn stop_open(&mut self) {
        for it in &mut self.items {
            match it {
                Item::Steps(steps) => {
                    let open = steps
                        .iter_mut()
                        .filter(|s| matches!(s.st, StepState::Run | StepState::Wait));
                    open.for_each(|s| s.st = StepState::Stop);
                }
                // A cancelled turn's helpers never say they ended.
                Item::Crew(c) => {
                    let open = c.helpers.iter_mut().filter(|h| crate::crew::open(h));
                    open.for_each(|h| h.st = StepState::Stop);
                }
                _ => {}
            }
        }
    }

    /// The live reply's helpers while any is still waiting or working, whatever the reply shows
    /// after them: the tree leads them until the last one ends.
    pub fn crew(&self) -> Option<&crate::crew::Crew> {
        (self.items[self.reply()..].iter().rev())
            .find_map(|i| match i {
                Item::Crew(c) => Some(c),
                _ => None,
            })
            .filter(|c| self.live && c.helpers.iter().any(crate::crew::open))
    }

    /// What the chat's tree says now.
    pub fn mood(&self) -> Mood {
        if self.ask.is_some() {
            return Mood::Need;
        }
        if !self.live {
            return Mood::Ready;
        }
        if self.crew().is_some() {
            return Mood::Work;
        }
        match self.items.last() {
            Some(Item::Steps(s)) if s.last().is_some_and(|s| s.st == StepState::Run) => Mood::Work,
            Some(Item::Text { .. }) => Mood::Write,
            _ => Mood::Think,
        }
    }

    /// What the chat's crest says: its [`Thread::mood`], or `idle`, the engine's state when it is
    /// loading or has no model, while the chat waits on nothing (it said "Ready" while a model
    /// loaded).
    pub fn crest(&self, idle: Option<Mood>) -> Mood {
        match self.mood() {
            Mood::Ready => idle.unwrap_or(Mood::Ready),
            m => m,
        }
    }

    /// The question the chat waits on, in words.
    pub fn ask_line(&self) -> Option<(String, String)> {
        Some(match self.ask.as_ref()? {
            Ask::Confirm { name, args } | Ask::Code { name, args, .. } => {
                let target = args
                    .get("path")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| {
                        args.get("command")
                            .and_then(Value::as_str)
                            .map(|c| logic::clip(c, 60))
                    });
                let q = match name.as_str() {
                    "write_file" => format!(
                        "Save {}?",
                        target.as_deref().map_or("this file", logic::base_name)
                    ),
                    "edit" => format!(
                        "Edit {}?",
                        target.as_deref().map_or("this file", logic::base_name)
                    ),
                    "bash" => format!("Run {}?", target.unwrap_or_default()),
                    "remember" => "Remember this?".into(),
                    "forget" => "Forget this?".into(),
                    // An MCP server's tool, <server>__<tool>: who runs it.
                    n if crate::mcp::outside_tool(n).is_some() => {
                        let (server, tool) = crate::mcp::outside_tool(n).unwrap_or_default();
                        format!(
                            "Let {} run {}?",
                            server.replace('_', " "),
                            tool.replace('_', " ")
                        )
                    }
                    n => format!("Let it run {}?", n.replace('_', " ")),
                };
                (q, self.title.clone())
            }
            Ask::Packet => ("Keep the change it made?".into(), self.title.clone()),
        })
    }

    /// Whether `ask_line` shows all that Allow would send, so Home or the popover may answer it:
    /// not an outside tool's call (its arguments show on the chat's card only), a command cut to
    /// fit the line, or a change, whose No undoes it.
    pub fn quick_answer(&self) -> bool {
        match self.ask.as_ref() {
            Some(Ask::Confirm { name, args } | Ask::Code { name, args, .. }) => {
                crate::mcp::outside_tool(name).is_none()
                    && (args.get("command").and_then(Value::as_str))
                        .is_none_or(|c| c.chars().count() <= 60 && !c.contains('\n'))
            }
            _ => false,
        }
    }
}

/// A download in flight.
#[derive(Clone, Debug)]
pub struct Dl {
    pub repo: String,
    pub file: String,
    pub got: u64,
    pub total: u64,
    pub bps: f64,
}

/// What a toast's button does.
#[derive(Clone)]
pub enum Act {
    Restore(Gone),
    Undo(Receipt),
    Load(String),
    DownloadAnyway(String, String),
    /// Make the faster version of the model at this path.
    Faster(String),
    /// Switch the model at this path to its faster version.
    UseFaster(String),
}

/// A short message at the bottom, with at most one button.
pub struct Toast {
    pub text: String,
    pub act: Option<(String, Act)>,
    pub at: Instant,
}

impl Toast {
    /// How long it stays up: 5 s, 9 with a button, and a second per 15 characters of a long one
    /// (an error read whole, not cut off mid-sentence).
    pub fn lasts(&self) -> Duration {
        let base = if self.act.is_some() { 9 } else { 5 };
        Duration::from_secs(base.max(self.text.chars().count() as u64 / 15))
    }
}

/// The MCP tools page: installed servers, the MCP Registry's search, an open install form, and
/// sign-ins in progress.
#[derive(Default)]
pub struct Mcp {
    pub servers: Vec<McpServerStatus>,
    pub found: Vec<crate::mcp::Card>,
    pub cursor: Option<String>,
    pub loading: bool,
    pub searched: bool,
    pub err: Option<String>,
    /// The result card whose install form is open, its fields (input name, field) and the trust tick.
    pub open: Option<usize>,
    pub answers: Vec<(String, Entity<TextInput>)>,
    pub trust: bool,
    pub custom_open: bool,
    pub custom_trust: bool,
    /// "Show tools": each server's listed tools, or why they could not be listed.
    pub tested: HashMap<String, Result<Vec<McpToolInfo>, String>>,
    pub busy: HashSet<String>,
    /// A sign-in started: its page and host, and whether the browser was opened.
    pub signing: HashMap<String, (McpSignIn, bool)>,
    /// The server whose "Trust in tasks" waits for the user to confirm what it unlocks.
    pub trusting: Option<String>,
}

/// Hugging Face search and what is known of each repo.
#[derive(Default)]
pub struct Hf {
    pub sort: usize,
    pub results: Vec<HfModel>,
    pub loading: bool,
    pub searched: bool,
    pub files: HashMap<String, Result<Vec<HfFile>, String>>,
    pub info: HashMap<(String, String), RemoteModel>,
    /// Headers that could not be read, so their rows say why instead of reading forever.
    pub bad: HashMap<(String, String), String>,
    pub open: Option<String>,
    pub fits: bool,
    pub runs: bool,
    pub use_: Option<&'static str>,
    pub size: Option<&'static str>,
    /// Headers to read, oldest first, and how many are being read.
    pub queue: VecDeque<(String, String)>,
    pub reading: usize,
    /// Bumped by each search: an older one that answers last is dropped, not shown under the
    /// newer one's sort.
    pub searches: u64,
}

/// The window.
pub struct Ecl {
    pub core: Option<Arc<larch_app::App>>,
    pub tasks: crate::tasks_model::Model,
    pub task_goal: Entity<TextInput>,
    /// The composer's steps, one field each, in order.
    pub task_steps: Vec<Entity<TextInput>>,
    /// The composer's "done when" criteria, one per line.
    pub task_done: Entity<TextInput>,
    pub task_planning: String,
    pub task_request: u64,
    pub task_loading: bool,
    /// An action asked for a refresh while one was running; run another when it lands.
    pub task_dirty: bool,
    /// Runs with a command unsettled, so a second click is ignored.
    pub task_busy: crate::tasks_model::InFlight,
    pub task_poll: u64,
    /// `tasks::watch` runs: started at the first snapshot, so a quiet app hosts no service for it.
    pub task_watch: bool,
    pub task_pending: Option<(crate::tasks_model::Draft, String, String)>,
    /// The new task repeats every day at `task_time`.
    pub task_repeat: bool,
    pub task_time: Entity<TextInput>,
    /// Each schedule's rule, for the week calendar; the model holds no service types.
    pub task_rules:
        BTreeMap<String, (larch_task::schedule::Schedule, larch_task::schedule::Policy)>,
    /// The weekdays a new schedule runs on (`Schedule::Calendar::days`).
    pub task_days: u8,
    /// A field to focus on the next frame: a step Enter just added, made outside a window call.
    pub focus_next: Option<Entity<TextInput>>,
    /// The schedule the composer is editing, as it was when opened.
    pub task_editing: Option<crate::tasks_model::ScheduledRecord>,
    pub data: PathBuf,
    pub prefs: Prefs,
    /// The copy Larch Setup installed, which it updates; None for a build from source or the demo.
    pub install: Option<crate::update::Install>,
    /// Larch Setup's last answer to "is a newer release out?", and when it came.
    pub found: Option<crate::update::Found>,
    pub checked: Option<Instant>,
    /// Why the last check failed, for Settings; a background check never raises a toast.
    pub check_err: Option<String>,
    pub checking: bool,
    /// The sidebar's update card was put off until the next launch.
    pub update_later: bool,
    /// Larch Setup reopened this app after an update that did not install and was undone; said
    /// until dismissed, in place of the offer.
    pub update_failed: bool,
    pub os_dark: bool,
    pub os_reduce: bool,
    /// The page column's width in `u` units, as of this frame. gpui has no media queries: a
    /// narrow window (down to its 880 px minimum) drops the chat's side panel and grid columns.
    pub room: f32,
    pub page: Page,
    pub mtab: MTab,
    pub stab: STab,
    pub st: AppState,
    pub models: Vec<ModelEntry>,
    pub chats: Vec<ChatEntry>,
    pub threads: HashMap<String, Thread>,
    pub open: Option<String>,
    /// The next message's images, in the order they were added.
    pub chips: Vec<Chip>,
    /// The composer's draft: New Chat starts another, so an image still being read for the one it
    /// left does not join it.
    pub draft: u64,
    /// What the composer held for each place it left (None is Home), back when that place opens.
    drafts: HashMap<Option<String>, Held>,
    /// Half a device pixel, which `mid` moves a glyph down by (see there).
    pub half: Pixels,
    /// Saved chats' images by reference: None once one could not be read; absent while it is.
    pub pics: HashMap<String, Option<Arc<gpui::Image>>>,
    pub attach_later: Vec<String>,
    pub repo_later: Option<PathBuf>,
    pub chip: Option<&'static str>,
    pub composer: Entity<TextInput>,
    pub search: Entity<TextInput>,
    pub fact_in: Entity<TextInput>,
    pub edit_in: Entity<TextInput>,
    pub name_in: Entity<TextInput>,
    pub port_in: Entity<TextInput>,
    pub mcp_in: Entity<TextInput>,
    pub knob_in: Entity<TextInput>,
    /// The MCP tools page: the registry search, "add your own" (name, command or URL), and the
    /// address a sign-in's browser ended on.
    pub mcp: Mcp,
    pub mcpq_in: Entity<TextInput>,
    pub mcp_name_in: Entity<TextInput>,
    pub mcp_target_in: Entity<TextInput>,
    pub mcp_paste_in: Entity<TextInput>,
    pub knobs: Vec<Knob>,
    pub ledger: logic::Ledger,
    pub samples: VecDeque<(u64, f64)>,
    /// Where the pointer is across the throughput chart (0 left, 1 right), and the chart's box as
    /// last painted, which the pointer is measured against.
    pub chart_at: Option<f32>,
    pub chart_box: std::rc::Rc<std::cell::Cell<Option<gpui::Bounds<gpui::Pixels>>>>,
    pub last: Option<EngineSample>,
    pub editing_knob: Option<String>,
    pub facts: Vec<Fact>,
    pub mem_on: bool,
    /// The page view a cite opened, over the chat.
    pub docview: Option<DocView>,
    pub editing: Option<(String, String)>,
    pub hf: Hf,
    pub dls: BTreeMap<String, Dl>,
    /// Faster versions being made, by the original's path: how far (0 to 1) and seconds so far.
    pub fasters: BTreeMap<String, (f64, f64)>,
    pub toast: Option<Toast>,
    pub failed: Option<String>,
    /// A model's speed (`Prefs::tok_s`) changed since prefs were last saved: saved as a reply ends,
    /// not each second while it runs, nor lost at quit as before.
    tok_s_dirty: bool,
    /// The model being loaded and since when, to time it.
    pub loading: Option<(String, Instant)>,
    /// The launch screen while the last model loads again: its name and the expected load in ms.
    pub boot: Option<(String, f64)>,
    /// The launch has yet to load the last model (once, after the first model list).
    pub autoload: bool,
    pub popover: bool,
    /// The composer's Attach menu (Files or Folder) is open.
    pub attach_menu: bool,
    pub pop_ix: usize,
    pub armed: Option<String>,
    /// The chat whose "..." menu is open, and in which list (`chat_menu`'s `place`): a chat can
    /// show in Recent and on Chats at once, and only the row it opened from owns it.
    pub menu: Option<(&'static str, String)>,
    /// The chat whose name is being edited in its row, in `rename_in`, and in which list.
    pub renaming: Option<(&'static str, String)>,
    pub rename_in: Entity<TextInput>,
    /// Bumped by each rename: a chat list read before it is stale (`take_chats`).
    pub chats_rev: u64,
    pub sun: Tree,
    pub snip: usize,
    pub server: Option<Result<(u16, String), String>>,
    pub focus: FocusHandle,
    /// The welcome or the tour stop on screen (tour.rs).
    pub tour: Option<crate::tour::At>,
    /// The tour card's focus, for its keys.
    pub tour_focus: FocusHandle,
    /// The next frame moves the focus to the welcome's name field or the tour card.
    pub tour_refocus: bool,
    /// The open thread's text and what the mouse selected in it.
    pub sel: Selection,
    /// The chat's right-click menu while open: its window point and the item under it.
    pub ctx: Option<(Point<Pixels>, Option<usize>)>,
    shown: Page,
    _subs: Vec<Subscription>,
}

type Res<T> = Result<T, String>;

/// What Settings and the sidebar say after Larch Setup undid an update that did not install.
pub const FAILED_UPDATE: &str = "The last update did not install and was undone. This version still \
                                 runs, and your chats are kept.";

/// What a restart would stop, as "a reply and 2 task runs"; None when nothing is going.
fn going(replies: usize, runs: usize) -> Option<String> {
    let count = |n: usize, one: &str, many: &str| match n {
        0 => None,
        1 => Some(one.to_owned()),
        n => Some(format!("{n} {many}")),
    };
    let parts: Vec<String> = [
        count(replies, "a reply", "replies"),
        count(runs, "a task run", "task runs"),
    ]
    .into_iter()
    .flatten()
    .collect();
    (!parts.is_empty()).then(|| parts.join(" and "))
}

impl Ecl {
    /// The window over `core` (None: the demo), reading the core's events from `rx`.
    pub fn new(
        core: Res<Option<Arc<larch_app::App>>>,
        installed: bool,
        update_failed: bool,
        rx: UnboundedReceiver<Event>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let (core, startup_err) = match core {
            Ok(c) => (c, None),
            Err(e) => (None, Some(e)),
        };
        let demo = core.is_none() && startup_err.is_none();
        // A quit stops the task service, so a step mid-effect settles rather than being cut off.
        // Bounded: past 2 s the quit goes on, and the next start recovers the step. App-scoped,
        // not this view's: a title-bar close drops the view before the quit observers run.
        if let Some(core) = core.clone() {
            App::on_app_quit(cx, move |_| {
                let (done, stopped) = std::sync::mpsc::channel();
                let core = core.clone();
                std::thread::spawn(move || drop(done.send(core.stop_tasks())));
                match stopped.recv_timeout(Duration::from_secs(2)) {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => eprintln!("cannot stop the tasks: {e}"),
                    Err(_) => {
                        eprintln!("the tasks did not stop in 2 s; the next start recovers them")
                    }
                }
                async {}
            })
            .detach();
        }
        let data =
            larch_app::handoff::data_dir().unwrap_or_else(|_| std::env::temp_dir().join("larch"));
        let (prefs, prefs_err) = if demo {
            (Prefs::default(), None)
        } else {
            Prefs::load(&data)
        };
        let th = Theme::light(prefs.light);
        let field = |cx: &mut Context<Self>, ph: &str, multi: bool| {
            cx.new(|cx| TextInput::new(cx, ph, multi, if multi { 10 } else { 1 }, th))
        };
        let composer = field(
            cx,
            "Ask anything, attach papers, or point me at a repo…",
            true,
        );
        let search = field(cx, "Search Hugging Face", false);
        let task_goal = field(
            cx,
            "Compare the four MoE papers and tell me which routing holds up",
            false,
        );
        let task_done = field(
            cx,
            "Names the routing that holds up and cites each paper",
            true,
        );
        let task_time = field(cx, "7:45 am", false);
        let fact_in = field(cx, "Something to remember, in one sentence", false);
        let edit_in = field(cx, "", false);
        let rename_in = field(cx, "Chat name", false);
        let name_in = field(cx, "Your name", false);
        let port_in = field(cx, "8080", false);
        let mcp_in = field(cx, "8090", false);
        let knob_in = field(cx, "", false);
        let mcpq_in = field(
            cx,
            "Search the MCP Registry, for example time or github",
            false,
        );
        let mcp_name_in = field(cx, "My notes", false);
        let mcp_target_in = field(
            cx,
            "uvx mcp-server-time   or   https://example.com/mcp",
            false,
        );
        let mcp_paste_in = field(cx, "The address the browser ended on", false);
        name_in.update(cx, |i, cx| i.set_text(&prefs.name, cx));
        port_in.update(cx, |i, cx| i.set_text("8080", cx));
        mcp_in.update(cx, |i, cx| i.set_text("8090", cx));
        let subs = vec![
            cx.subscribe_in(&composer, window, |s, _, _: &Submitted, w, cx| {
                s.send(w, cx)
            }),
            cx.subscribe(&composer, |s, _, e: &PastedImage, cx| s.pasted(e, cx)),
            cx.subscribe(&composer, |s, _, e: &PastedFiles, cx| {
                s.add_image_files(e.0.clone(), cx)
            }),
            cx.subscribe(&search, |s, _, _: &Submitted, cx| s.hf_search(cx)),
            cx.subscribe(&mcpq_in, |s, _, _: &Submitted, cx| s.mcp_search(false, cx)),
            cx.subscribe(&fact_in, |s, _, _: &Submitted, cx| s.remember(cx)),
            cx.subscribe(&edit_in, |s, _, _: &Submitted, cx| s.save_edit(cx)),
            cx.subscribe(&rename_in, |s, _, _: &Submitted, cx| s.save_rename(cx)),
            // The same field asks the name in the welcome, where Enter answers it.
            cx.subscribe(&name_in, |s, _, _: &Submitted, cx| {
                if s.tour == Some(crate::tour::At::Welcome) {
                    crate::tour::next(s, cx)
                } else {
                    s.save_name(cx)
                }
            }),
            cx.subscribe(&knob_in, |s, _, _: &Submitted, cx| s.set_knob(cx)),
            cx.subscribe_in(&task_done, window, |_, input, _: &Submitted, w, cx| {
                input.update(cx, |i, cx| {
                    gpui::EntityInputHandler::replace_text_in_range(i, None, "\n", w, cx)
                })
            }),
            cx.observe_window_appearance(window, |s, w, cx| {
                s.os_dark = matches!(
                    w.appearance(),
                    WindowAppearance::Dark | WindowAppearance::VibrantDark
                );
                cx.notify();
            }),
        ];
        let install = (installed && !demo)
            .then(|| larch_fetch::install_root().ok())
            .flatten()
            .and_then(|root| crate::update::Install::read(&root));
        let mut s = Ecl {
            install,
            found: None,
            checked: None,
            check_err: None,
            checking: false,
            update_later: false,
            update_failed,
            core,
            tasks: crate::tasks_model::Model::default(),
            task_goal,
            task_steps: Vec::new(),
            task_done,
            task_planning: "initial".into(),
            task_request: 0,
            task_loading: false,
            task_dirty: false,
            task_busy: Default::default(),
            task_poll: 0,
            task_watch: false,
            task_pending: None,
            task_repeat: false,
            task_time,
            task_rules: BTreeMap::new(),
            task_days: larch_task::schedule::EVERY_DAY,
            focus_next: None,
            task_editing: None,
            data,
            prefs,
            os_dark: matches!(
                window.appearance(),
                WindowAppearance::Dark | WindowAppearance::VibrantDark
            ),
            os_reduce: logic::os_reduce_motion(),
            room: f32::MAX,
            page: Page::Home,
            mtab: MTab::Library,
            stab: STab::Appearance,
            st: AppState {
                health: Health::Empty,
                model: None,
                device: String::new(),
                data_dir: String::new(),
                gpu_total_bytes: None,
                gpu_free_bytes: None,
                agent_folder: None,
                libraries: Vec::new(),
                hf_cache: None,
                docs_keep_ready: true,
                tasks_web: false,
                code_mode: false,
                vision: false,
                vision_error: None,
            },
            models: Vec::new(),
            chats: Vec::new(),
            threads: HashMap::new(),
            open: None,
            chips: Vec::new(),
            draft: 0,
            drafts: HashMap::new(),
            half: px(0.),
            pics: HashMap::new(),
            attach_later: Vec::new(),
            repo_later: None,
            chip: None,
            composer,
            search,
            fact_in,
            edit_in,
            rename_in,
            menu: None,
            renaming: None,
            chats_rev: 0,
            name_in,
            port_in,
            mcp_in,
            knob_in,
            mcp: Mcp::default(),
            mcpq_in,
            mcp_name_in,
            mcp_target_in,
            mcp_paste_in,
            knobs: Vec::new(),
            ledger: logic::Ledger::default(),
            samples: VecDeque::new(),
            chart_at: None,
            chart_box: Default::default(),
            last: None,
            editing_knob: None,
            facts: Vec::new(),
            mem_on: true,
            docview: None,
            editing: None,
            hf: Hf::default(),
            dls: BTreeMap::new(),
            fasters: BTreeMap::new(),
            toast: None,
            loading: None,
            boot: None,
            autoload: !demo,
            failed: startup_err.map(|e| format!("{NO_START}: {e}")),
            tok_s_dirty: false,
            popover: false,
            attach_menu: false,
            pop_ix: 0,
            armed: None,
            sun: Tree::new(Mood::Rest),
            snip: 0,
            server: None,
            focus: cx.focus_handle(),
            tour: None,
            tour_focus: cx.focus_handle(),
            tour_refocus: false,
            sel: Selection::default(),
            ctx: None,
            shown: Page::Home,
            _subs: subs,
        };
        s.tasks.reviewed = s.prefs.reviewed.clone();
        if let Some(e) = prefs_err {
            s.say(e);
        }
        if s.core.is_some()
            && let Some(why) = larch_app::unused_gpu()
        {
            s.say(format!("Models run on the CPU, not your GPU: {why}."));
        }
        // A new install is welcomed once the core runs, so its answer can be saved. `--demo` has
        // no core and saves nothing; Settings › General replays the tour there.
        if s.core.is_some() && s.prefs.welcomed < crate::tour::TOUR {
            crate::tour::welcome(&mut s);
        }
        if demo {
            crate::demo::seed(&mut s);
        } else {
            s.refresh(cx);
            s.load_memory(cx);
            // From the start, not from the Tasks page's first visit: until then the workflow
            // service was never hosted, so schedules did not run and nothing told the user.
            s.task_watch = true;
            crate::tasks::watch(cx);
            s.call(
                cx,
                |c| c.list_downloads(),
                |s, d, _| {
                    for d in d.into_iter().filter(|d| d.state == "running") {
                        s.dls.insert(
                            d.id,
                            Dl {
                                repo: d.repo,
                                file: d.file,
                                got: d.received_bytes,
                                total: d.total_bytes,
                                bps: 0.,
                            },
                        );
                    }
                },
            );
        }
        if s.install.is_some() {
            // A while after launch, so the check never competes with loading the model; then daily.
            cx.spawn(async move |this, cx| {
                cx.background_executor()
                    .timer(Duration::from_secs(30))
                    .await;
                while this
                    .update(cx, |s, cx| {
                        if s.prefs.updates {
                            s.check_updates(cx)
                        }
                    })
                    .is_ok()
                {
                    cx.background_executor().timer(crate::update::EVERY).await;
                }
            })
            .detach();
        }
        cx.spawn(async move |this, cx| {
            let mut rx = rx;
            while let Some(e) = rx.next().await {
                if this.update(cx, |s, cx| s.on_event(e, cx)).is_err() {
                    break;
                }
            }
        })
        .detach();
        window.focus(&s.composer.focus_handle(cx));
        s
    }

    /// The colours in force.
    pub fn theme(&self) -> Theme {
        let dark = match self.prefs.mode {
            Mode::System => self.os_dark,
            Mode::Light => false,
            Mode::Dark => true,
        };
        if dark {
            Theme::dark(self.prefs.dark)
        } else {
            Theme::light(self.prefs.light)
        }
    }

    /// The tree holds still: Animation off, Motion Reduced or Off, or System with the OS asking.
    pub fn still(&self) -> bool {
        !self.prefs.animate
            || match self.prefs.motion {
                Motion::System => self.os_reduce,
                Motion::Full => false,
                Motion::Reduced | Motion::Off => true,
            }
    }

    /// Shows `text` in a toast.
    pub fn say(&mut self, text: impl Into<String>) {
        self.set_toast(Some(Toast {
            text: text.into(),
            act: None,
            at: Instant::now(),
        }));
    }

    /// A toast with a button.
    pub fn say_act(&mut self, text: impl Into<String>, label: &str, act: Act) {
        self.set_toast(Some(Toast {
            text: text.into(),
            act: Some((label.into(), act)),
            at: Instant::now(),
        }));
    }

    /// `t` in place of the toast; a selection in the old one goes with it, or the next would
    /// show it and stay up holding it.
    pub fn set_toast(&mut self, t: Option<Toast>) {
        if self.sel.holds(select::TOAST) {
            self.sel.clear();
        }
        self.toast = t;
    }

    /// Checks the cites of chat `id`'s answers against its documents, each answer once, for the
    /// buttons under it. The documents are read where the agent reads them, off the UI thread.
    fn check_cites(&mut self, id: String, cx: &mut Context<Self>) {
        let Some(t) = self.threads.get(&id) else {
            return;
        };
        let texts: Vec<(usize, String)> = (t.items.iter().enumerate())
            .filter_map(|(i, it)| match it {
                Item::Text { text, .. } if text.contains('[') && !t.cites.contains_key(&i) => {
                    Some((i, text.clone()))
                }
                _ => None,
            })
            .collect();
        if texts.is_empty() {
            return;
        }
        let chat = id.clone();
        self.try_call(
            cx,
            move |c| {
                let docs = c.documents(&chat)?;
                Ok(texts
                    .into_iter()
                    .map(|(i, text)| {
                        let links = logic::cite_links(&text, &docs);
                        (i, text, links)
                    })
                    .collect::<Vec<_>>())
            },
            move |s, got, cx| {
                let Some(t) = s.threads.get_mut(&id) else {
                    return;
                };
                for (i, text, links) in got {
                    // The chat may have been reloaded since: only the answer it was made from.
                    if matches!(t.items.get(i), Some(Item::Text { text: now, .. }) if *now == text)
                    {
                        t.cites.insert(i, links);
                    }
                }
                cx.notify();
            },
            // An answer with no documents to check against shows no buttons, which is not an error.
            |_, _, _| {},
        );
    }

    /// The page view over the chat: the cited page, the quote highlighted.
    fn docview_el(&self, th: &Theme, cx: &mut Context<Self>) -> Option<Stateful<Div>> {
        let v = self.docview.as_ref()?;
        let th2 = *th;
        let shut = cx.listener(|s, _, _, cx| {
            s.docview = None;
            cx.notify()
        });
        Some(
            div()
                .id("docview-back")
                .absolute()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .bg(fade(th.ink, 0.35))
                .occlude()
                .on_click(cx.listener(|s, _, _, cx| {
                    s.docview = None;
                    cx.notify()
                }))
                .child(
                    div()
                        .id("docview")
                        .w(u(760.))
                        .max_w_full()
                        .h(u(560.))
                        .max_h_full()
                        .flex()
                        .flex_col()
                        .rounded(u(16.))
                        .bg(th.card)
                        .border_1()
                        .border_color(th.line)
                        .shadow(shadow(th))
                        .occlude()
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .px(u(20.))
                                .py(u(14.))
                                .border_b_1()
                                .border_color(th.line)
                                .child(
                                    div()
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .child(v.title.clone()),
                                )
                                .child(
                                    div()
                                        .id("docview-x")
                                        .cursor_pointer()
                                        .child(icon("x", 12., th.ink3))
                                        .on_click(shut),
                                ),
                        )
                        .child(
                            div()
                                .id("docview-rows")
                                .flex_1()
                                .min_h_0()
                                .p(u(20.))
                                .overflow_y_scroll()
                                .text_size(u(14.))
                                .line_height(u(22.))
                                .child(StyledText::new(v.page.clone()).with_runs(page_runs(
                                    &th2,
                                    &v.page,
                                    v.hit.clone(),
                                ))),
                        ),
                ),
        )
    }

    /// Asks Larch Setup whether a newer release is out. A failure shows in Settings, not as a
    /// toast, so a computer offline never nags.
    pub fn check_updates(&mut self, cx: &mut Context<Self>) {
        let Some(install) = self.install.clone().filter(|_| !self.checking) else {
            return;
        };
        self.checking = true;
        let task = cx.background_spawn(async move { install.check() });
        cx.spawn(async move |this, cx| {
            let r = task.await;
            this.update(cx, |s, cx| {
                s.checking = false;
                s.checked = Some(Instant::now());
                match r {
                    Ok(f) => {
                        s.found = Some(f);
                        s.check_err = None;
                    }
                    Err(e) => s.check_err = Some(e),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// The newer release to offer, if any: never while pinned.
    pub fn update_offer(&self) -> Option<String> {
        crate::update::offer(self.install.as_ref()?, self.found.as_ref()?)
    }

    /// Hands over to Larch Setup, which installs once this app has quit and then opens it again.
    /// Quitting stops replies and task runs, so it asks first while any is going.
    pub fn restart_to_update(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let replies = self.threads.values().filter(|t| t.live).count();
        let runs = self.tasks.running();
        let Some(going) = going(replies, runs) else {
            return self.update_now(cx);
        };
        let answer = window.prompt(
            PromptLevel::Warning,
            "Restart Larch to update?",
            Some(&format!("Restarting now stops {going}.")),
            &["Restart", "Not now"],
            cx,
        );
        cx.spawn(async move |this, cx| {
            if answer.await == Ok(0) {
                this.update(cx, |s, cx| s.update_now(cx)).ok();
            }
        })
        .detach();
    }

    fn update_now(&mut self, cx: &mut Context<Self>) {
        let Some(install) = &self.install else {
            return;
        };
        match install.update() {
            Ok(()) => cx.quit(),
            Err(e) => self.say(e),
        }
    }

    /// Runs `f` on the core off the UI thread, then `then` with its result; an error is shown as
    /// it is. In the demo nothing runs and a toast says so.
    pub fn call<T: Send + 'static>(
        &mut self,
        cx: &mut Context<Self>,
        f: impl FnOnce(&larch_app::App) -> Res<T> + Send + 'static,
        then: impl FnOnce(&mut Self, T, &mut Context<Self>) + 'static,
    ) {
        self.try_call(cx, f, then, |s, e, _| s.say(e));
    }

    /// `call`, with `fail` handling the error in place of the toast.
    pub fn try_call<T: Send + 'static>(
        &mut self,
        cx: &mut Context<Self>,
        f: impl FnOnce(&larch_app::App) -> Res<T> + Send + 'static,
        then: impl FnOnce(&mut Self, T, &mut Context<Self>) + 'static,
        fail: impl FnOnce(&mut Self, String, &mut Context<Self>) + 'static,
    ) {
        let Some(core) = self.core.clone() else {
            // The demo refuses as the core would, so whatever the caller set up is put back.
            fail(
                self,
                "This is the demo, so nothing runs. Start Larch without --demo to use your models."
                    .into(),
                cx,
            );
            cx.notify();
            return;
        };
        let task = cx.background_spawn(async move { f(&core) });
        cx.spawn(async move |this, cx| {
            let r = task.await;
            this.update(cx, |s, cx| {
                match r {
                    Ok(v) => then(s, v, cx),
                    Err(e) => fail(s, e, cx),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Reads the chat list again.
    pub fn reload_chats(&mut self, cx: &mut Context<Self>) {
        let rev = self.chats_rev;
        self.call(
            cx,
            |c| c.list_chats(),
            move |s, c, cx| s.take_chats(c, rev, cx),
        );
    }

    /// Shows `chats`, read at `rev`, and the open threads' names from it. A list read before a
    /// rename is read again instead: the rename's own read can come before a chat made with this
    /// one (`open_as_chat`).
    pub fn take_chats(&mut self, chats: Vec<ChatEntry>, rev: u64, cx: &mut Context<Self>) {
        if rev != self.chats_rev {
            self.reload_chats(cx);
            return;
        }
        for c in &chats {
            if let Some(t) = self.threads.get_mut(&c.id) {
                t.title = c.title.clone();
            }
        }
        self.chats = chats;
    }

    /// Reads the app state and the model list again.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let rev = self.chats_rev;
        self.call(
            cx,
            |c| Ok((c.app_state(), c.list_models()?, c.list_chats()?)),
            move |s, (st, models, chats), cx| {
                let loaded = st.model.is_some() && st.model != s.st.model;
                if st.health == Health::Failed {
                    s.failed
                        .get_or_insert_with(|| "The GPU stopped responding.".into());
                }
                s.st = st;
                s.models = models;
                s.take_chats(chats, rev, cx);
                if loaded {
                    // Whatever loaded it, the next launch loads it again.
                    if s.prefs.last_model != s.st.model {
                        s.prefs.last_model = s.st.model.clone();
                        s.save_prefs();
                    }
                    s.load_knobs(cx);
                }
                if std::mem::take(&mut s.autoload) {
                    s.load_last(cx);
                }
            },
        );
    }

    pub fn load_knobs(&mut self, cx: &mut Context<Self>) {
        self.call(cx, |c| c.knobs(), |s, k, _| s.knobs = k);
    }

    /// Reads memory again.
    pub fn load_memory(&mut self, cx: &mut Context<Self>) {
        self.call(
            cx,
            |c| Ok((c.memory_list()?, c.memory_enabled())),
            |s, (f, on), _| {
                s.facts = f;
                s.mem_on = on;
            },
        );
    }

    fn on_event(&mut self, e: Event, cx: &mut Context<Self>) {
        match e {
            Event::Chat { id, event } => self.on_chat(id, event, cx),
            Event::Sample(sm) => {
                let t = now_ms();
                if self.samples.back().is_none_or(|(at, _)| t >= at + 1000) {
                    self.samples.push_back((t, sm.tok_s));
                }
                while self
                    .samples
                    .front()
                    .is_some_and(|(at, _)| t.saturating_sub(*at) > 600_000)
                {
                    self.samples.pop_front();
                }
                if sm.tok_s > 0.
                    && let Some(m) = self.st.model.clone()
                {
                    let best = self.prefs.tok_s.entry(m).or_default();
                    if (*best - sm.tok_s).abs() > 0.5 && sm.running > 0 {
                        *best = sm.tok_s;
                        self.tok_s_dirty = true;
                    }
                }
                self.last = Some(sm);
            }
            Event::Config(c) => {
                if let Some(k) = self.knobs.iter_mut().find(|k| k.name == c.knob) {
                    k.value = c.new;
                }
                self.ledger.changes.insert(0, c);
            }
            Event::Failed(f) => self.failed = Some(f.message),
            Event::Health(h) => {
                self.st.health = h;
                match h {
                    // The old engine is gone; the next one numbers its changes afresh. Its load is
                    // timed from here, not from the click: the old engine's shutdown is no part
                    // of the next launch's load.
                    Health::Loading => {
                        self.ledger.new_engine();
                        if let Some(l) = &mut self.loading {
                            l.1 = Instant::now();
                        }
                    }
                    Health::Failed => {
                        self.failed
                            .get_or_insert_with(|| "The GPU stopped responding.".into());
                    }
                    Health::Ready => {
                        // The next launch loads this model again, and times its bar by this load
                        // averaged with the ones before: one load from a cold disk, or one from a
                        // warm cache, never halves or doubles the next bar alone.
                        if let Some((path, at)) = self.loading.take() {
                            let ms = at.elapsed().as_millis() as u64;
                            let had = self.prefs.load_ms.get(&path);
                            let ms = had.map_or(ms, |&had| (had + ms) / 2);
                            self.prefs.load_ms.insert(path.clone(), ms);
                            self.prefs.last_model = Some(path);
                            self.save_prefs();
                        }
                        // The bar fills before the launch screen goes, never leaving mid-way.
                        if self.boot.is_some() {
                            cx.spawn(async move |this, cx| {
                                cx.background_executor().timer(BOOT_FULL).await;
                                this.update(cx, |s, cx| {
                                    if s.st.health == Health::Ready {
                                        s.boot = None;
                                        cx.notify();
                                    }
                                })
                                .ok();
                            })
                            .detach();
                        }
                        self.refresh(cx);
                        self.load_knobs(cx);
                    }
                    Health::Empty => {
                        self.loading = None;
                        self.boot = None;
                    }
                }
            }
            Event::DownloadProgress(p) => {
                let d = self.dls.entry(p.id).or_insert(Dl {
                    repo: p.repo,
                    file: p.file,
                    got: 0,
                    total: 0,
                    bps: 0.,
                });
                d.got = p.received_bytes;
                d.total = p.total_bytes;
                d.bps = if d.bps > 0. {
                    d.bps * 0.7 + p.bytes_per_sec * 0.3
                } else {
                    p.bytes_per_sec
                };
            }
            Event::DownloadDone(p) => {
                self.dls.remove(&p.id);
                if let Some(Ok(files)) = self.hf.files.get_mut(&p.repo) {
                    files
                        .iter_mut()
                        .filter(|f| f.file == p.file)
                        .for_each(|f| f.downloaded = true);
                }
                let note = p.note.map(|n| format!(" {n}")).unwrap_or_default();
                // The first model on this computer has nothing to stand in for: it loads at once,
                // as Home's first-model card promised.
                let first = self.models.is_empty()
                    && self.st.model.is_none()
                    && self.st.health != Health::Loading;
                if first {
                    if !note.is_empty() {
                        self.say(note.trim_start().to_owned());
                    }
                    self.load_model(p.path.clone(), cx);
                } else {
                    self.say_act(
                        format!("{} is downloaded.{note}", logic::base_name(&p.file)),
                        "Use it",
                        Act::Load(p.path.clone()),
                    );
                }
                // Where a faster version runs better on this computer, it is offered right away.
                let (path, name) = (p.path.clone(), logic::base_name(&p.file).to_owned());
                self.try_call(
                    cx,
                    move |c| Ok(c.faster_for(&path).map(|f| (path, f))),
                    move |s, offer, _| {
                        if let Some((path, _)) =
                            offer.filter(|(_, f)| f.copy.is_none() && !f.building)
                        {
                            s.say_act(
                                format!(
                                    "{name} is downloaded. It can run faster on this computer."
                                ),
                                "Make it faster",
                                Act::Faster(path),
                            );
                        }
                    },
                    |_, _, _| {},
                );
                self.refresh(cx);
                self.notify_os(
                    "Download finished",
                    &format!("{} is ready to use.", logic::base_name(&p.file)),
                );
            }
            Event::DownloadFailed(p) => {
                self.dls.remove(&p.id);
                if p.cancelled {
                    self.say("Download cancelled.");
                } else {
                    self.say_act(p.message, "Try again", Act::DownloadAnyway(p.repo, p.file));
                }
            }
            Event::FasterProgress(p) => {
                self.fasters.insert(p.path, (p.fraction, p.seconds));
            }
            Event::FasterDone(p) => {
                self.fasters.remove(&p.path);
                let name = self.listed_name(&p.path);
                // Chatting went on on the original meanwhile: it moves to the faster one now. A
                // reply being written refuses the switch, and the toast's button tries again.
                if self.st.model.as_deref() == Some(p.path.as_str()) {
                    self.use_faster(p.path.clone(), true, cx);
                } else {
                    self.say_act(
                        format!("The faster version of {name} is ready."),
                        "Use it",
                        Act::UseFaster(p.path.clone()),
                    );
                }
                self.notify_os(
                    "Faster version ready",
                    &format!("{name} now runs faster on this computer."),
                );
                self.refresh(cx);
            }
            Event::FasterFailed(p) => {
                self.fasters.remove(&p.path);
                self.say(if p.cancelled {
                    "Stopped making the faster version.".to_owned()
                } else {
                    p.message
                });
                self.refresh(cx);
            }
            Event::Allowed(_) => {}
            // A server connected, failed, signed in or was turned off.
            Event::Mcp(_) => self.mcp_refresh(cx),
        }
        cx.notify();
    }

    fn on_chat(&mut self, id: String, e: ChatEvent, cx: &mut Context<Self>) {
        // A chat's name comes after its answer ended, perhaps while the user scrolls back
        // through it: it changes the header and the list only, never the thread's scroll or
        // order, and a chat this window does not have (deleted just now) gets no thread from it.
        match e {
            // The name is read back from disk, not taken from the event: one prepared before a
            // rename can arrive after it, and the saved name is the user's.
            ChatEvent::Title { .. } => {
                self.reload_chats(cx);
                return;
            }
            // Said on its own, not in the thread, whose next reply may already be streaming.
            ChatEvent::TitleFailed { message } => {
                self.say(message);
                return;
            }
            _ => {}
        }
        let mut ask_now = None;
        let t = self
            .threads
            .entry(id.clone())
            .or_insert_with(|| Thread::new("Chat"));
        t.updated = now_ms();
        match e {
            ChatEvent::Delta { text } => {
                t.progress = None;
                t.live = true;
                t.delta(text);
            }
            ChatEvent::Done {
                finish,
                prompt_tokens,
                cached_tokens,
                ttft_ms,
                tok_s,
                ..
            } => {
                t.live = false;
                t.progress = None;
                t.stop_open();
                t.docs.ended();
                if finish == "cancelled"
                    && let Some(Item::Text { cut, .. }) = t.items.last_mut()
                {
                    *cut = true;
                }
                if tok_s > 0. {
                    t.stats = Some(Stats {
                        tok_s,
                        ttft_ms,
                        reused: if prompt_tokens > 0 {
                            cached_tokens as f64 / prompt_tokens as f64
                        } else {
                            0.
                        },
                    });
                }
                self.sun.set(Mood::Done);
                self.reload_chats(cx);
                self.check_cites(id, cx);
                if std::mem::take(&mut self.tok_s_dirty) {
                    self.save_prefs();
                }
            }
            // In place of `Done` (a GPU fault, a turn with no answer, a failed code run): its open
            // steps and helpers stop with it, or they showed as running for good.
            ChatEvent::Error { message } => {
                t.live = false;
                t.progress = None;
                t.stop_open();
                t.docs.ended();
                t.items.push(Item::Error(message));
            }
            ChatEvent::ToolCall { name, args } => {
                t.progress = None;
                t.live = true;
                let (now, done) = logic::describe(&name, &args);
                t.steps().push(Step {
                    name,
                    args,
                    now,
                    done,
                    sum: String::new(),
                    st: StepState::Run,
                    count: 1,
                });
            }
            ChatEvent::ToolResult { name, summary } => t.tool_result(name, summary),
            ChatEvent::Confirm { name, args } => {
                if let Some(s) =
                    (t.last_steps().and_then(|s| s.last_mut())).filter(|s| s.st == StepState::Run)
                {
                    s.st = StepState::Wait;
                }
                t.ask = Some(Ask::Confirm { name, args });
                ask_now = t.ask_line();
            }
            ChatEvent::CodeAsk {
                name,
                args,
                path,
                folder,
            } => {
                t.ask = Some(Ask::Code {
                    name,
                    args,
                    path,
                    folder,
                });
                ask_now = t.ask_line();
            }
            ChatEvent::Notice { text } => t.items.push(Item::Notice(text)),
            ChatEvent::Thinking { text } => {
                t.progress = None;
                t.live = true;
                t.reasoning(text);
            }
            ChatEvent::MemorySaved { receipt } => {
                t.items.push(Item::Saved {
                    receipt,
                    line: None,
                    again: false,
                });
                self.load_memory(cx);
            }
            ChatEvent::Packet {
                title,
                body,
                diff,
                tests_before,
                tests_after,
                scope,
                outside,
                stop,
                checkpoint,
                ..
            } => {
                t.items.push(Item::Packet(Box::new(Packet {
                    title,
                    body,
                    diff,
                    before: tests_before,
                    after: tests_after,
                    scope,
                    outside,
                    stop,
                    checkpoint,
                    decided: None,
                })));
                t.ask = Some(Ask::Packet);
                ask_now = t.ask_line();
            }
            ChatEvent::Progress { text } => t.progress = Some(text),
            ChatEvent::Documents { state, text } => t.docs.apply(&state, &text),
            // This reply's helpers, as they stand: the tree spawns them on the first word.
            ChatEvent::Helpers { helpers } => t.set_helpers(
                helpers
                    .into_iter()
                    .map(crate::crew::Helper::from_chat)
                    .collect(),
            ),
            // Taken above: they touch no thread.
            ChatEvent::Title { .. } | ChatEvent::TitleFailed { .. } => {}
        }
        if let Some((q, _)) = ask_now {
            self.needs_you(&q);
        }
    }

    /// An ask arrived: the OS notification and the chime, as Settings > Assistant says.
    fn needs_you(&mut self, q: &str) {
        self.notify_os("Larch needs you", q);
        if self.prefs.sound {
            chime();
        }
    }

    /// The OS notification, when Settings > Assistant allows it.
    pub(crate) fn notify_os(&self, title: &str, body: &str) {
        if !self.prefs.notify || self.core.is_none() {
            return;
        }
        notify(title, body);
    }

    /// Opens chat `id`, reading its saved messages the first time.
    pub fn open_chat(&mut self, id: String, cx: &mut Context<Self>) {
        self.page = Page::Chat;
        self.set_open(Some(id.clone()), cx);
        self.sel.clear();
        self.ctx = None;
        self.popover = false;
        let title = self
            .chats
            .iter()
            .find(|c| c.id == id)
            .map_or("Chat".into(), |c| c.title.clone());
        let t = self
            .threads
            .entry(id.clone())
            .or_insert_with(|| Thread::new(&title));
        if t.loaded || self.core.is_none() {
            return;
        }
        t.loaded = true;
        let cid = id.clone();
        let failed = id.clone();
        self.try_call(
            cx,
            move |c| Ok((c.open_chat(&cid)?, c.attachments(&cid)?)),
            move |s, (msgs, files), cx| {
                let Some(t) = s.threads.get_mut(&id) else {
                    return;
                };
                let live = std::mem::take(&mut t.items);
                t.items = saved_items(&msgs);
                t.items.extend(live);
                t.files = files;
                t.scroll.scroll_to_bottom();
                s.load_pics(&id, cx);
                s.check_cites(id, cx);
            },
            // Read again at the next open, not left empty for the rest of the session.
            move |s, e, _| {
                s.say(e);
                if let Some(t) = s.threads.get_mut(&failed) {
                    t.loaded = false;
                }
            },
        );
    }

    /// Reads the images chat `id`'s saved messages show that this window has not read yet.
    fn load_pics(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some(t) = self.threads.get(id) else {
            return;
        };
        let refs: Vec<String> = (t.items.iter())
            .flat_map(|i| match i {
                Item::User { images, .. } => images.as_slice(),
                _ => &[][..],
            })
            .filter_map(|p| match p {
                Pic::Saved(r) if !self.pics.contains_key(r) => Some(r.clone()),
                _ => None,
            })
            .collect();
        if refs.is_empty() {
            return;
        }
        let cid = id.to_owned();
        self.try_call(
            cx,
            move |c| {
                Ok((refs.into_iter())
                    .map(|r| {
                        let image = (c.chat_image(&cid, &r).ok())
                            .and_then(|b| images::sniff(&b).map(|f| (f, b)))
                            .and_then(|(f, b)| images::preview(f, b).ok());
                        (r, image)
                    })
                    .collect::<Vec<_>>())
            },
            |s, read, _| s.pics.extend(read),
            |_, _, _| {},
        );
    }

    /// A paste in the composer that held no text: its image joins the next message. When the
    /// window read none, the OS clipboard is asked (a Windows bitmap, or Windows under WSL).
    fn pasted(&mut self, e: &PastedImage, cx: &mut Context<Self>) {
        if let Some(image) = &e.0 {
            let bytes = image.bytes().to_vec();
            self.add_image(move || Ok(bytes), cx);
        } else {
            let read = cx.background_spawn(async { images::os_clipboard_png() });
            let draft = self.draft;
            cx.spawn(async move |this, cx| {
                if let Some(png) = read.await {
                    this.update(cx, |s, cx| {
                        if s.draft == draft {
                            s.add_image(move || Ok(png), cx)
                        }
                    })
                    .ok();
                }
            })
            .detach();
        }
    }

    /// Image files dropped on the composer or picked with Attach join the next message.
    pub fn add_image_files(&mut self, paths: Vec<PathBuf>, cx: &mut Context<Self>) {
        for p in paths {
            self.add_image(move || images::read_file(&p), cx);
        }
    }

    /// Adds the image `read` gives to the next message: checked here, converted when its format
    /// is one larch does not decode, then taken by the core (`App::image_add`), which encodes it
    /// while the user types. A refusal is a toast; the chip shows once the core has it.
    fn add_image(
        &mut self,
        read: impl FnOnce() -> Result<Vec<u8>, String> + Send + 'static,
        cx: &mut Context<Self>,
    ) {
        let waiting = self.chips.iter().filter(|c| !c.sending).count();
        let (vision, why) = (self.st.vision, self.st.vision_error.clone());
        if let Some(no) = images::refusal(waiting, 0, vision, why.as_deref()) {
            self.say(no);
            cx.notify();
            return;
        }
        let draft = self.draft;
        self.call(
            cx,
            move |c| {
                let (bytes, format) = images::readable(read()?)?;
                if let Some(no) = images::refusal(0, bytes.len(), true, None) {
                    return Err(no);
                }
                let image = images::preview(format, bytes.clone())?;
                Ok((c.image_add(&bytes)?, image))
            },
            move |s, (info, image), _| {
                // Older unsent images this one pushed out of the core's memory for drafts go.
                let before = s.chips.len();
                (s.chips).retain(|c| c.sending || !info.evicted.contains(&c.reference));
                let gone = before - s.chips.len();
                if gone > 0 {
                    s.say(format!(
                        "{gone} earlier image{} dropped to make room; add again to send.",
                        if gone == 1 { " was" } else { "s were" }
                    ));
                }
                // Read for a draft New Chat left: it does not join this one.
                if s.draft != draft {
                    return;
                }
                if s.chips
                    .iter()
                    .any(|c| c.reference == info.reference && !c.sending)
                {
                    return;
                }
                let waiting = s.chips.iter().filter(|c| !c.sending).count();
                if let Some(no) = images::refusal(waiting, 0, true, None) {
                    s.say(no);
                    return;
                }
                s.chips.push(Chip {
                    reference: info.reference,
                    image,
                    tokens: info.tokens,
                    sending: false,
                });
            },
        );
    }

    /// The core refused the message carrying `refs`: their chips are back, to send again.
    fn images_unsent(&mut self, refs: &[String]) {
        for c in &mut self.chips {
            if refs.contains(&c.reference) {
                c.sending = false;
            }
        }
    }

    /// A new, empty chat in the composer.
    pub fn new_chat(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.set_open(None, cx);
        self.page = Page::Home;
        self.chips.clear();
        self.draft += 1;
        self.attach_later.clear();
        self.repo_later = None;
        self.chip = None;
        window.focus(&self.composer.focus_handle(cx));
        cx.notify();
    }

    /// Sends the composer's text: to the open chat, or to a new one.
    pub fn send(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self
            .open
            .as_ref()
            .and_then(|id| self.threads.get(id))
            .is_some_and(|t| t.live)
        {
            return;
        }
        let typed = self.composer.read(cx).text.trim().to_owned();
        // The message's images: their chips hide from now until the core takes or refuses it.
        let (refs, pics): (Vec<String>, Vec<Pic>) = (self.chips.iter())
            .filter(|c| !c.sending)
            .map(|c| (c.reference.clone(), Pic::Live(c.image.clone())))
            .unzip();
        if typed.is_empty() && refs.is_empty() {
            return;
        }
        let chip = self.chip.take();
        // Everything the send consumes, put back whole if the core refuses it.
        let draft = Draft {
            typed: typed.clone(),
            chip,
            attach: std::mem::take(&mut self.attach_later),
            repo: self.repo_later.clone(),
        };
        let thread = self.open.as_ref().and_then(|id| self.threads.get(id));
        let task = logic::send_task(
            chip,
            &typed,
            (thread.and_then(|t| t.repo.as_ref()))
                .or(self.repo_later.as_ref())
                .is_some(),
            thread.is_some_and(|t| !t.files.is_empty()) || !draft.attach.is_empty(),
        );
        let folder = self
            .open
            .as_ref()
            .and_then(|id| self.threads.get(id))
            .and_then(|t| t.repo.clone())
            .or(self.repo_later.clone())
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let text = match task {
            Some(t) => match logic::prompt(t, &typed, &folder) {
                Ok(p) => p,
                Err(e) => {
                    self.say(e);
                    self.redraft(draft, cx);
                    cx.notify();
                    return;
                }
            },
            None => typed.clone(),
        };
        self.composer.update(cx, |i, cx| {
            i.take(cx);
        });
        let open = self.open.clone();
        let repo = open
            .as_ref()
            .and_then(|id| self.threads.get(id))
            .and_then(|t| t.repo.clone())
            .or(self.repo_later.clone());
        let intent = match chip {
            Some("Code") if repo.is_some() => Some(Intent::Change),
            Some("Code") => Some(Intent::Code),
            Some("Write") => Some(Intent::Chat),
            _ => None,
        };
        let user = Item::User {
            text: typed.clone(),
            files: Vec::new(),
            images: pics,
        };
        let attach = draft.attach.clone();
        for c in &mut self.chips {
            if refs.contains(&c.reference) {
                c.sending = true;
            }
        }
        if let Some(id) = open {
            let t = self
                .threads
                .get_mut(&id)
                .expect("an open chat has a thread");
            t.items.push(user);
            t.docs.first_word = None;
            t.live = true;
            t.task = task.or(t.task);
            t.scroll.scroll_to_bottom();
            let (failed, shown) = (id.clone(), id.clone());
            let (sent, unsent) = (refs.clone(), refs.clone());
            self.try_call(
                cx,
                move |c| {
                    let mut files = None;
                    for p in &attach {
                        files = Some(c.attach(&id, p)?.attached);
                    }
                    c.agent_send(&id, &text, repo.as_deref(), intent, &refs)?;
                    Ok(files)
                },
                move |s, files, cx| {
                    if let (Some(t), Some(files)) = (s.threads.get_mut(&shown), files) {
                        t.files = files;
                    }
                    s.sent(&sent, cx)
                },
                move |s, e, cx| s.send_failed(&failed, draft, &unsent, e, cx),
            );
        } else {
            if self.core.is_none() {
                self.images_unsent(&refs);
                self.say("This is the demo, so nothing runs. Start Larch without --demo to use your models.");
                self.redraft(draft, cx);
                cx.notify();
                return;
            }
            let title = logic::clip(
                if typed.is_empty() {
                    "Image"
                } else {
                    &draft.typed
                },
                40,
            );
            self.repo_later = None;
            let retry = draft.clone();
            let chat_refs = refs.clone();
            self.try_call(
                cx,
                |c| c.new_chat(None),
                move |s, id, cx| {
                    let mut t = s.threads.remove(&id).unwrap_or_else(|| Thread::new(&title));
                    t.title = title;
                    t.loaded = true;
                    t.repo = repo.clone();
                    t.task = task;
                    t.items.insert(0, user);
                    t.live = true;
                    s.threads.insert(id.clone(), t);
                    s.set_open(Some(id.clone()), cx);
                    s.sel.clear();
                    s.page = Page::Chat;
                    let (failed, shown) = (id.clone(), id.clone());
                    let (sent, unsent) = (refs.clone(), refs.clone());
                    s.try_call(
                        cx,
                        move |c| {
                            let mut files = None;
                            for p in &attach {
                                files = Some(c.attach(&id, p)?.attached);
                            }
                            c.agent_send(&id, &text, repo.as_deref(), intent, &refs)?;
                            Ok(files)
                        },
                        // The chat lists what this send attached, as an attach in it does.
                        move |s, files, cx| {
                            if let (Some(t), Some(files)) = (s.threads.get_mut(&shown), files) {
                                t.files = files;
                            }
                            s.sent(&sent, cx)
                        },
                        move |s, e, cx| s.first_send_failed(failed, retry, &unsent, e, cx),
                    );
                },
                move |s, e, cx| {
                    s.say(e);
                    s.images_unsent(&chat_refs);
                    s.redraft(draft, cx);
                },
            );
        }
        window.focus(&self.composer.focus_handle(cx));
        cx.notify();
    }

    /// The core took the message carrying `refs`: their chips go (one added meanwhile stays), and
    /// the list shows the chat while it answers, not only once the answer ends.
    fn sent(&mut self, refs: &[String], cx: &mut Context<Self>) {
        (self.chips).retain(|c| !(c.sending && refs.contains(&c.reference)));
        self.reload_chats(cx);
    }

    /// A message the core refused: the chat stops waiting, says why, and the draft and its images
    /// are back in the composer to send again.
    fn send_failed(
        &mut self,
        id: &str,
        draft: Draft,
        refs: &[String],
        e: String,
        cx: &mut Context<Self>,
    ) {
        if let Some(t) = self.threads.get_mut(id) {
            t.live = false;
            t.items.push(Item::Error(e));
        }
        self.images_unsent(refs);
        self.redraft(draft, cx);
    }

    /// A new chat's first message was refused, so the chat holds nothing: it is removed rather
    /// than listed for good as an empty "New chat", and the draft goes back in Home's composer
    /// with the reason.
    fn first_send_failed(
        &mut self,
        id: String,
        draft: Draft,
        refs: &[String],
        e: String,
        cx: &mut Context<Self>,
    ) {
        self.threads.remove(&id);
        if self.open.as_deref() == Some(id.as_str()) {
            self.go(Page::Home, cx);
        }
        self.images_unsent(refs);
        self.say(e);
        self.redraft(draft, cx);
        self.delete_chat(id, cx);
    }

    /// Opens `to` (None is Home) under the composer: what was typed for the place it leaves waits
    /// for it, and `to` gets its own draft back, so one chat's half-typed message never follows
    /// into another. Images still sending stay with their send.
    fn set_open(&mut self, to: Option<String>, cx: &mut Context<Self>) {
        if self.open == to {
            return;
        }
        let (images, sending) = std::mem::take(&mut self.chips)
            .into_iter()
            .partition(|c| !c.sending);
        self.chips = sending;
        let left = Held {
            draft: Draft {
                typed: self.composer.update(cx, |i, cx| i.take(cx)),
                chip: self.chip.take(),
                attach: std::mem::take(&mut self.attach_later),
                repo: self.repo_later.take(),
            },
            images,
        };
        let from = std::mem::replace(&mut self.open, to);
        if left.draft != Draft::default() || !left.images.is_empty() {
            self.drafts.insert(from, left);
        }
        // An image still being read was for the place just left.
        self.draft += 1;
        if let Some(h) = self.drafts.remove(&self.open) {
            self.composer
                .update(cx, |i, cx| i.set_text(&h.draft.typed, cx));
            self.chip = h.draft.chip;
            self.attach_later = h.draft.attach;
            self.repo_later = h.draft.repo;
            self.chips.extend(h.images);
        }
    }

    /// Puts a refused send's draft back in the composer (`Draft::back_into`).
    fn redraft(&mut self, d: Draft, cx: &mut Context<Self>) {
        let now = Draft {
            typed: self.composer.read(cx).text.clone(),
            chip: self.chip.take(),
            attach: std::mem::take(&mut self.attach_later),
            repo: self.repo_later.take(),
        };
        let shown = now.typed.clone();
        let back = d.back_into(now);
        if back.typed != shown {
            self.composer
                .update(cx, |i, cx| i.set_text(&back.typed, cx));
        }
        self.chip = back.chip;
        self.attach_later = back.attach;
        self.repo_later = back.repo;
    }

    fn stop_reply(&mut self, cx: &mut Context<Self>) {
        if let Some(id) = self
            .open
            .clone()
            .filter(|id| self.threads.get(id).is_some_and(|t| t.live))
        {
            self.call(cx, move |c| c.cancel(&id), |_, _, _| {});
        }
    }

    /// Answers chat `id`'s ask: `yes` for a Confirm or a Packet, or a code-mode decision.
    pub fn answer(
        &mut self,
        id: String,
        yes: bool,
        decision: Option<Decision>,
        cx: &mut Context<Self>,
    ) {
        let Some(t) = self.threads.get_mut(&id) else {
            return;
        };
        let Some(ask) = t.ask.take() else { return };
        t.live = true;
        if matches!(ask, Ask::Packet) {
            set_decided(t, Some(yes));
        }
        let ask_now = ask.clone();
        let back = id.clone();
        // Refused: the ask comes back unanswered, so the user can answer again.
        let fail = move |s: &mut Self, e: String, _: &mut Context<Self>| {
            if let Some(t) = s.threads.get_mut(&back) {
                t.live = false;
                if matches!(ask, Ask::Packet) {
                    set_decided(t, None);
                }
                t.ask = Some(ask);
            }
            s.say(e);
        };
        match ask_now {
            Ask::Confirm { .. } => {
                self.try_call(cx, move |c| c.agent_confirm(&id, yes), |_, _, _| {}, fail)
            }
            Ask::Code { .. } => {
                let d = decision.unwrap_or(if yes { Decision::Once } else { Decision::No });
                self.try_call(cx, move |c| c.code_answer(&id, d), |_, _, _| {}, fail)
            }
            Ask::Packet => self.try_call(cx, move |c| c.code_decide(&id, yes), |_, _, _| {}, fail),
        }
        cx.notify();
    }

    fn remember(&mut self, cx: &mut Context<Self>) {
        let text = self.fact_in.read(cx).text.trim().to_owned();
        if text.is_empty() {
            return;
        }
        self.fact_in.update(cx, |i, cx| {
            i.take(cx);
        });
        self.call(
            cx,
            move |c| c.memory_add(&text),
            |s, r, cx| {
                s.say_act(
                    format!("I'll remember: {}", logic::fact_sentence(&r.text)),
                    "Undo",
                    Act::Undo(r),
                );
                s.load_memory(cx);
            },
        );
    }

    fn save_edit(&mut self, cx: &mut Context<Self>) {
        let Some((slug, expected)) = self.editing.take() else {
            return;
        };
        let text = self.edit_in.read(cx).text.trim().to_owned();
        self.call(
            cx,
            move |c| c.memory_update(&slug, &expected, &text),
            |s, changed, cx| {
                if changed {
                    s.say("Saved.");
                }
                s.load_memory(cx);
            },
        );
        cx.notify();
    }

    fn save_name(&mut self, cx: &mut Context<Self>) {
        self.prefs.name = crate::tour::clean_name(&self.name_in.read(cx).text);
        self.save_prefs();
        self.say("Saved.");
        cx.notify();
    }

    fn set_knob(&mut self, cx: &mut Context<Self>) {
        let Some(name) = self.editing_knob.take() else {
            return;
        };
        let typed = self.knob_in.read(cx).text.trim().to_owned();
        let Ok(v) = typed.parse::<f64>() else {
            self.say(format!("{typed} is not a number."));
            cx.notify();
            return;
        };
        self.call(
            cx,
            move |c| c.knob_set(&name, v),
            |s, r, cx| {
                s.say(match r {
                    KnobResult::Applied { .. } => "Applied.".to_owned(),
                    KnobResult::Pending { reason } => format!("Queued: {reason}"),
                    KnobResult::Rejected { reason } => format!("Not applied: {reason}"),
                });
                s.load_knobs(cx);
            },
        );
    }

    /// Saves the window settings; a failure is shown.
    pub fn save_prefs(&mut self) {
        if self.core.is_none() {
            return;
        }
        if let Err(e) = self.prefs.save(&self.data) {
            self.say(e);
        }
    }

    /// Runs a toast's button.
    pub fn act(&mut self, a: Act, cx: &mut Context<Self>) {
        self.set_toast(None);
        match a {
            Act::Restore(g) => self.call(
                cx,
                move |c| c.memory_restore(&g),
                |s, _, cx| {
                    s.say("It's back.");
                    s.load_memory(cx)
                },
            ),
            Act::Undo(r) => self.call_undo(r, cx),
            Act::Load(p) => self.load_model(p, cx),
            Act::DownloadAnyway(repo, file) => self.download(repo, file, true, cx),
            Act::Faster(p) => self.make_faster(p, cx),
            Act::UseFaster(p) => self.use_faster(p, true, cx),
        }
    }

    /// Undoes a memory save; the outcome replaces the toast or the chat's receipt line.
    pub fn call_undo(&mut self, r: Receipt, cx: &mut Context<Self>) {
        let Some(core) = self.core.clone() else {
            self.say("This is the demo, so nothing runs.");
            return;
        };
        let slug = r.slug.clone();
        let task = cx.background_spawn(async move { core.memory_undo(&r) });
        cx.spawn(async move |this, cx| {
            let err = task.await.err();
            this.update(cx, |s, cx| {
                let (line, again) = logic::undo_outcome(err.as_deref());
                let mut shown = false;
                for t in s.threads.values_mut() {
                    for i in &mut t.items {
                        if let Item::Saved {
                            receipt,
                            line: l,
                            again: a,
                        } = i
                            && receipt.slug == slug
                        {
                            *l = Some(line.clone());
                            *a = again;
                            shown = true;
                        }
                    }
                }
                if !shown {
                    s.say(line);
                }
                s.load_memory(cx);
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// At launch, loads the model loaded last time behind the launch screen, when it is still in a
    /// library, runs, and nothing is loaded yet.
    fn load_last(&mut self, cx: &mut Context<Self>) {
        if self.st.model.is_some() || self.st.health == Health::Loading {
            return;
        }
        if let Some((path, name, expect)) = relaunch(&self.prefs, &self.models) {
            self.boot = Some((name, expect));
            self.load_model(path, cx);
        } else if let Some(last) = &self.prefs.last_model {
            self.say(format!(
                "{} is no longer in a library, so nothing was loaded.",
                logic::base_name(last)
            ));
        }
    }

    /// A listed model's name (its faster version's too), else its file's.
    fn listed_name(&self, path: &str) -> String {
        listing(&self.models, path).map_or(logic::base_name(path).to_owned(), |m| m.name.clone())
    }

    /// Starts making the faster version of the model at `path`; its card shows how far it got.
    pub fn make_faster(&mut self, path: String, cx: &mut Context<Self>) {
        let p = path.clone();
        self.call(
            cx,
            move |c| c.build_faster(&p),
            move |s, _, cx| {
                s.fasters.entry(path).or_insert((0., 0.));
                s.refresh(cx);
            },
        );
    }

    /// Runs the model at `path` as its faster version (`on`) or as the original. A faster version
    /// that does not load is deleted and the original loaded again; the toast says so.
    pub fn use_faster(&mut self, path: String, on: bool, cx: &mut Context<Self>) {
        let target = match on {
            true => (self.models.iter().find(|m| m.path == path))
                .and_then(|m| m.faster.as_ref()?.copy.clone())
                .unwrap_or_else(|| path.clone()),
            false => path.clone(),
        };
        self.say(match on {
            true => "Switching to the faster version",
            false => "Switching to the original",
        });
        let was = self.st.health;
        self.loading = Some((target, Instant::now()));
        self.st.health = Health::Loading;
        self.try_call(
            cx,
            move |c| c.use_faster(&path, on),
            |s, _, cx| s.refresh(cx),
            move |s, e, cx| {
                s.say(e);
                s.loading = None;
                s.st.health = was;
                s.refresh(cx);
            },
        );
    }

    /// Loads the model at `path`.
    pub fn load_model(&mut self, path: String, cx: &mut Context<Self>) {
        let name = self
            .models
            .iter()
            .find(|m| m.path == path)
            .map_or(logic::base_name(&path).to_owned(), |m| m.name.clone());
        if self.boot.is_none() {
            self.say(format!("Loading {name}"));
        }
        let was = self.st.health;
        self.loading = Some((path.clone(), Instant::now()));
        self.st.health = Health::Loading;
        // The ledger starts afresh at `Health::Loading`, once the old engine is gone: a refused
        // load leaves it running, with its changes and their Undo.
        self.try_call(
            cx,
            move |c| c.load_model(&path),
            |s, _, cx| s.refresh(cx),
            // A refusal (a reply being written) comes before any health event: without this the
            // window says "Loading a model" until the next load.
            move |s, e, cx| {
                s.say(e);
                s.loading = None;
                s.st.health = was;
                s.refresh(cx);
            },
        );
    }

    /// Downloads `file` from `repo`, after reading its header so nothing Larch cannot run is
    /// downloaded; `anyway` skips the fit check.
    pub fn download(&mut self, repo: String, file: String, anyway: bool, cx: &mut Context<Self>) {
        let lib = self
            .prefs
            .download_to
            .clone()
            .filter(|l| self.st.libraries.contains(l));
        let have = self.gpu_room();
        let ctx = self
            .knobs
            .iter()
            .find(|k| k.name == "max_context")
            .map_or(8192., |k| k.value);
        self.call(
            cx,
            move |c| {
                let info = c.hf_inspect(&repo, &file)?;
                if !info.supported {
                    return Ok(Err(format!(
                        "{}: {}, so it was not downloaded.",
                        logic::base_name(&file),
                        cannot_run(&info)
                    )));
                }
                let need = floor_bytes(&info, ctx);
                if !anyway && have.is_some_and(|h| need > h) {
                    return Ok(Err(format!(
                        "{} needs about {} of GPU memory; this GPU has {} free.|{repo}|{file}",
                        logic::base_name(&file),
                        logic::gb(need),
                        logic::gb(have.unwrap_or_default())
                    )));
                }
                c.download_model(&repo, &file, lib.as_deref())
                    .map(|_| Ok(()))
            },
            // The row comes from the core's first progress event, sent before the download can
            // end: added here, it would come after a quick failure and stay "Connecting" for good.
            |s, r, _| {
                if let Err(e) = r {
                    match e.split('|').collect::<Vec<_>>()[..] {
                        [msg, repo, file] => s.say_act(
                            msg,
                            "Download anyway",
                            Act::DownloadAnyway(repo.into(), file.into()),
                        ),
                        _ => s.say(e),
                    }
                }
            },
        );
    }

    /// The GPU's memory, measured, when it is known.
    pub fn gpu_bytes(&self) -> Option<u64> {
        self.st
            .gpu_total_bytes
            .or(self.last.as_ref().and_then(|s| s.gpu_total_bytes))
    }

    /// The room a model has: what other programs left free on the GPU (the loader's figure), else
    /// its total. The total let a 7.5 GB model through on a card that had 7.2 GB free.
    pub fn gpu_room(&self) -> Option<u64> {
        self.st.gpu_free_bytes.or_else(|| self.gpu_bytes())
    }

    /// Searches Hugging Face for the search field's text.
    /// The installed MCP servers and their state; closes sign-in panels that ended.
    pub fn mcp_refresh(&mut self, cx: &mut Context<Self>) {
        self.call(
            cx,
            |c| Ok(c.mcp_servers()),
            |s, list, cx| {
                // Kept while the sign-in waits: one that failed left a panel whose paste had nowhere
                // to go, and hid Sign in.
                s.mcp
                    .signing
                    .retain(|id, _| list.iter().any(|x| &x.id == id && x.signing));
                s.mcp.servers = list;
                cx.notify();
            },
        );
    }

    /// Searches the MCP Registry; an empty search starts with the servers that need no sign-in.
    pub fn mcp_search(&mut self, more: bool, cx: &mut Context<Self>) {
        let q = self.mcpq_in.read(cx).text.trim().to_owned();
        let cursor = if more { self.mcp.cursor.clone() } else { None };
        self.mcp.loading = true;
        self.mcp.searched = true;
        if self.mcp.servers.is_empty() {
            self.mcp_refresh(cx);
        }
        self.try_call(
            cx,
            move |c| c.mcp_search(&q, cursor.as_deref()),
            move |s, r, cx| {
                s.mcp.loading = false;
                s.mcp.err = None;
                let cards: Vec<crate::mcp::Card> = r.servers.iter().map(crate::mcp::card).collect();
                let empty = s.mcpq_in.read(cx).text.trim().is_empty();
                if more {
                    s.mcp.found.extend(cards);
                } else {
                    s.mcp.found = if empty {
                        crate::mcp::suggested().into_iter().chain(cards).collect()
                    } else {
                        cards
                    };
                    s.mcp_close();
                }
                s.mcp.cursor = r.next_cursor;
                cx.notify();
            },
            |s, e, cx| {
                s.mcp.loading = false;
                s.mcp.err = Some(e);
                if s.mcp.found.is_empty() {
                    s.mcp.found = crate::mcp::suggested();
                }
                cx.notify();
            },
        );
        cx.notify();
    }

    /// Opens a result's install form: one field per value it asks for, secrets masked.
    pub fn mcp_open(&mut self, i: usize, cx: &mut Context<Self>) {
        let th = self.theme();
        let inputs = (self.mcp.found.get(i).and_then(|c| c.option()))
            .map(|o| o.inputs.clone())
            .unwrap_or_default();
        self.mcp.answers = inputs
            .into_iter()
            .map(|inp| {
                let ph = if inp.description.is_empty() {
                    inp.name.clone()
                } else {
                    inp.description.clone()
                };
                let f = cx.new(|cx| {
                    let mut t = TextInput::new(cx, &ph, false, 1, th);
                    t.secret = inp.secret;
                    t
                });
                f.update(cx, |t, cx| t.set_text(&inp.value, cx));
                (inp.name, f)
            })
            .collect();
        self.mcp.open = Some(i);
        self.mcp.trust = false;
    }

    pub fn mcp_close(&mut self) {
        self.mcp.open = None;
        self.mcp.answers.clear();
        self.mcp.trust = false;
    }

    /// The open form's answers, by input name.
    pub fn mcp_answers(&self, cx: &App) -> BTreeMap<String, String> {
        (self.mcp.answers.iter())
            .map(|(n, f)| (n.clone(), f.read(cx).text.clone()))
            .collect()
    }

    fn mcp_taken(&self) -> Vec<String> {
        self.mcp.servers.iter().map(|s| s.id.clone()).collect()
    }

    /// Installs the open result, once the user ticked "I trust this server".
    pub fn mcp_install(&mut self, cx: &mut Context<Self>) {
        let Some(card) = self.mcp.open.and_then(|i| self.mcp.found.get(i)).cloned() else {
            return;
        };
        let Some(option) = card.option().cloned() else {
            return;
        };
        if !self.mcp.trust {
            return self.say("Tick “I trust this server” to install it.");
        }
        match crate::mcp::plan(&card, &option, &self.mcp_answers(cx), &self.mcp_taken()) {
            Err(e) => self.say(e.join(" ")),
            Ok(server) => {
                let name = server.name.clone();
                self.mcp_close();
                self.mcp_add(
                    server,
                    format!(
                        "{name} is installed and on. Its tools join new chats once it connects."
                    ),
                    cx,
                );
            }
        }
    }

    /// Adds the "add your own" server, once ticked as trusted.
    pub fn mcp_add_custom(&mut self, cx: &mut Context<Self>) {
        if !self.mcp.custom_trust {
            return self.say("Tick “I trust this server” to add it.");
        }
        let (name, target) = (
            self.mcp_name_in.read(cx).text.clone(),
            self.mcp_target_in.read(cx).text.clone(),
        );
        match crate::mcp::custom(&name, &target, &self.mcp_taken()) {
            Err(e) => self.say(e.join(" ")),
            Ok(server) => {
                self.mcp_name_in.update(cx, |i, cx| i.set_text("", cx));
                self.mcp_target_in.update(cx, |i, cx| i.set_text("", cx));
                self.mcp.custom_trust = false;
                let name = server.name.clone();
                self.mcp_add(server, format!("{name} is added and on."), cx);
            }
        }
    }

    fn mcp_add(&mut self, server: McpServer, done: String, cx: &mut Context<Self>) {
        self.call(
            cx,
            move |c| c.mcp_add(server),
            move |s, (), cx| {
                s.say(done);
                s.mcp_refresh(cx);
            },
        );
    }

    /// Runs `f` on the core for server `id`, marked busy meanwhile, then refreshes.
    fn mcp_act(
        &mut self,
        id: String,
        cx: &mut Context<Self>,
        f: impl FnOnce(&larch_app::App, &str) -> Res<()> + Send + 'static,
    ) {
        self.mcp.busy.insert(id.clone());
        let (id2, id3) = (id.clone(), id.clone());
        self.try_call(
            cx,
            move |c| f(c, &id2),
            move |s, (), cx| {
                s.mcp.busy.remove(&id);
                s.mcp_refresh(cx);
            },
            // Its own flag only: another server still connecting keeps its "Connecting…".
            move |s, e, cx| {
                s.mcp.busy.remove(&id3);
                s.say(e);
                s.mcp_refresh(cx);
            },
        );
    }

    pub fn mcp_set_enabled(&mut self, id: String, on: bool, cx: &mut Context<Self>) {
        self.mcp_act(id, cx, move |c, id| c.mcp_set_enabled(id, on));
    }

    pub fn mcp_set_trusted(&mut self, id: String, on: bool, cx: &mut Context<Self>) {
        self.mcp_act(id, cx, move |c, id| c.mcp_set_trusted(id, on));
    }

    pub fn mcp_remove(&mut self, id: String, cx: &mut Context<Self>) {
        self.mcp.tested.remove(&id);
        self.mcp.signing.remove(&id);
        // A server installed again under this id asks for trust afresh.
        if self.mcp.trusting.as_ref() == Some(&id) {
            self.mcp.trusting = None;
        }
        self.mcp_act(id, cx, |c, id| c.mcp_remove(id));
    }

    /// "Show tools": connects if needed and lists the tools; pressed again, hides them.
    pub fn mcp_test(&mut self, id: String, cx: &mut Context<Self>) {
        if self.mcp.tested.remove(&id).is_some() {
            return;
        }
        self.mcp.busy.insert(id.clone());
        let (a, b) = (id.clone(), id.clone());
        self.try_call(
            cx,
            move |c| c.mcp_test(&a),
            move |s, tools, cx| {
                s.mcp.busy.remove(&b);
                s.mcp.tested.insert(b, Ok(tools));
                s.mcp_refresh(cx);
            },
            move |s, e, cx| {
                s.mcp.busy.remove(&id);
                s.mcp.tested.insert(id, Err(e));
                s.mcp_refresh(cx);
            },
        );
    }

    /// Starts a sign-in: the card first says which site it opens.
    pub fn mcp_sign_in(&mut self, id: String, cx: &mut Context<Self>) {
        self.mcp.busy.insert(id.clone());
        let (a, b) = (id.clone(), id.clone());
        self.try_call(
            cx,
            move |c| c.mcp_sign_in(&a),
            move |s, r, cx| {
                s.mcp.busy.remove(&b);
                s.mcp.signing.insert(b, (r, false));
                cx.notify();
            },
            move |s, e, cx| {
                s.mcp.busy.remove(&id);
                s.say(e);
                cx.notify();
            },
        );
    }

    /// Opens the sign-in page in the browser (best effort: the link stays on the card).
    pub fn mcp_open_sign(&mut self, id: &str, cx: &mut Context<Self>) {
        let Some((r, opened)) = self.mcp.signing.get_mut(id) else {
            return;
        };
        *opened = true;
        if logic::web_url(&r.url).is_some() {
            cx.open_url(&r.url);
        }
        let host = r.host.clone();
        self.say(format!("Finish signing in at {host} in your browser."));
    }

    /// Finishes a waiting sign-in with the address the browser ended on.
    pub fn mcp_finish_sign(&mut self, id: String, cx: &mut Context<Self>) {
        let url = self.mcp_paste_in.read(cx).text.trim().to_owned();
        self.mcp_paste_in.update(cx, |i, cx| i.set_text("", cx));
        self.call(
            cx,
            move |c| c.mcp_sign_in_finish(&id, &url),
            |s, (), _| s.say("Finishing the sign-in…"),
        );
    }

    pub fn mcp_sign_out(&mut self, id: String, cx: &mut Context<Self>) {
        self.mcp.signing.remove(&id);
        self.mcp_act(id, cx, |c, id| c.mcp_sign_out(id));
    }

    /// Closes a sign-in panel; a sign-in the server already had stays.
    pub fn mcp_cancel_sign_in(&mut self, id: String, cx: &mut Context<Self>) {
        self.mcp.signing.remove(&id);
        self.mcp_act(id, cx, |c, id| c.mcp_cancel_sign_in(id));
    }

    pub fn hf_search(&mut self, cx: &mut Context<Self>) {
        let q = self.search.read(cx).text.trim().to_owned();
        let sort = ["downloads", "likes", "lastModified"][self.hf.sort];
        self.hf.loading = true;
        self.hf.searches += 1;
        let search = self.hf.searches;
        self.try_call(
            cx,
            move |c| c.hf_search(&q, sort, 24),
            move |s, r, cx| {
                if s.hf.searches != search {
                    return;
                }
                s.hf.loading = false;
                s.hf.searched = true;
                // A listing that failed is asked for again: kept, it hid the repo's files all
                // session.
                s.hf.files.retain(|_, f| f.is_ok());
                let repos: Vec<String> = r
                    .iter()
                    .map(|m| m.repo.clone())
                    .filter(|r| !s.hf.files.contains_key(r))
                    .collect();
                // The last search's headers are no longer on screen.
                s.hf.queue.clear();
                s.hf.results = r;
                for repo in repos {
                    s.hf_files(repo, cx);
                }
            },
            // Offline, "Searching Hugging Face…" stayed up until a later search worked.
            move |s, e, cx| {
                if s.hf.searches != search {
                    return;
                }
                s.hf.loading = false;
                s.say(e);
                cx.notify();
            },
        );
        cx.notify();
    }

    /// Lists `repo`'s files, then reads the header of the one its row offers. The list lands first
    /// (one small request), so a row has its file and size while its header (a few MB) comes.
    pub fn hf_files(&mut self, repo: String, cx: &mut Context<Self>) {
        let Some(core) = self.core.clone() else {
            return;
        };
        let task = cx.background_spawn(async move {
            let files = core.hf_files(&repo);
            (repo, files)
        });
        cx.spawn(async move |this, cx| {
            let (repo, files) = task.await;
            this.update(cx, |s, cx| {
                let file = files
                    .as_ref()
                    .ok()
                    .and_then(|f| pick_file(f, s.gpu_room()))
                    .map(|f| f.file.clone());
                s.hf.files.insert(repo.clone(), files);
                if let Some(file) = file {
                    s.hf.queue.push_back((repo, file));
                    s.read_headers(cx);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Reads queued headers, a few at a time: each is a few MB, and every call the app makes waits
    /// for a free background thread, so 24 at once stalled the next search and slowed a download.
    /// Rows meanwhile go by what the search and file list say.
    fn read_headers(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.core.clone() else {
            return;
        };
        // ponytail: 4 readers, 1 while a download runs; measure before raising.
        let most = if self.dls.is_empty() { 4 } else { 1 };
        while self.hf.reading < most {
            let Some(key) = self.hf.queue.pop_front() else {
                return;
            };
            let shown = self.hf.results.iter().any(|m| m.repo == key.0);
            if !shown || self.hf.info.contains_key(&key) || self.hf.bad.contains_key(&key) {
                continue;
            }
            self.hf.reading += 1;
            let core = core.clone();
            let task = cx.background_spawn(async move {
                let info = core.hf_inspect(&key.0, &key.1);
                (key, info)
            });
            cx.spawn(async move |this, cx| {
                let (key, info) = task.await;
                this.update(cx, |s, cx| {
                    s.hf.reading -= 1;
                    match info {
                        Ok(i) => s.hf.info.insert(key, i).map(drop),
                        Err(e) => s.hf.bad.insert(key, e).map(drop),
                    };
                    s.read_headers(cx);
                    cx.notify();
                })
                .ok();
            })
            .detach();
        }
    }

    fn mood(&mut self, window: &Window, cx: &App) -> Mood {
        let m = if self.failed.is_some() {
            Mood::Err
        } else if self.st.health == Health::Loading {
            Mood::Load
        } else if self.needs().next().is_some() {
            Mood::Need
        } else if let Some(t) = self
            .open
            .as_ref()
            .and_then(|id| self.threads.get(id))
            .filter(|t| t.live)
        {
            t.mood()
        } else if self.sun.mood == Mood::Done
            && self.sun.since.elapsed() < Duration::from_millis(1150)
        {
            Mood::Done
        } else if !self.composer.read(cx).text.is_empty()
            && self.composer.focus_handle(cx).is_focused(window)
        {
            Mood::Listen
        } else if self.st.model.is_none() && self.core.is_some() {
            Mood::Rest
        } else {
            Mood::Ready
        };
        self.sun.set(m);
        m
    }

    /// The engine's state when no chat can be answered now: loading a model, or without one.
    fn idle_mood(&self) -> Option<Mood> {
        if self.st.health == Health::Loading {
            Some(Mood::Load)
        } else if self.st.model.is_none() && self.core.is_some() {
            Some(Mood::Rest)
        } else {
            None
        }
    }

    /// Every chat waiting on the user, newest first.
    pub fn needs(&self) -> impl Iterator<Item = (&String, &Thread)> {
        let mut v: Vec<_> = self
            .threads
            .iter()
            .filter(|(_, t)| t.ask.is_some())
            .collect();
        v.sort_by_key(|(_, t)| std::cmp::Reverse(t.updated));
        v.into_iter()
    }

    /// How many things need the user.
    pub fn need_count(&self) -> usize {
        self.needs().count()
    }

    /// The assistant's status line: its state and, while it writes, the speed.
    pub fn status_line(&self, m: Mood) -> String {
        match m {
            Mood::Need => format!("Needs you · {}", self.need_count()),
            Mood::Write => match &self.last {
                Some(s) if s.tok_s > 0. => format!("Writing · {:.0} tok/s", s.tok_s),
                _ => "Writing".into(),
            },
            Mood::Err => "Problem".into(),
            Mood::Load => "Loading a model".into(),
            Mood::Rest if self.st.model.is_none() => "No model loaded".into(),
            m => m.label().into(),
        }
    }

    /// Where the model stands, from the engine's health: a switch keeps the old model's name in
    /// `st.model` until the new one is up, so a name alone does not mean ready.
    fn model_state(&self) -> ModelState {
        ModelState::of(
            self.st.health,
            self.failed.is_some(),
            self.st.model.is_some(),
        )
    }

    /// The running model's name.
    pub fn model_name(&self) -> Option<String> {
        Some(self.listed_name(self.st.model.as_ref()?))
    }

    /// Keeps papers for the next message, which attaches them before it goes; `tell`, in an open
    /// chat, says so, as nothing else shows they wait.
    fn attach_next(&mut self, paths: Vec<String>, tell: bool) {
        for p in paths {
            if !self.attach_later.contains(&p) {
                self.attach_later.push(p);
            }
        }
        if tell {
            self.say("Attaches with your next message");
        }
    }

    /// Takes `path` off what is attached: the open chat's own (`saved`) comes off its saved list,
    /// one waiting for the next message just stops waiting.
    fn unattach(&mut self, path: String, saved: bool, cx: &mut Context<Self>) {
        let Some(id) = self.open.clone().filter(|_| saved) else {
            self.attach_later.retain(|p| *p != path);
            return;
        };
        self.call(
            cx,
            move |c| Ok((c.detach(&id, &path)?, id)),
            |s, (files, id), _| {
                if let Some(t) = s.threads.get_mut(&id) {
                    t.files = files;
                }
            },
        );
    }

    /// Runs the system picker (with picker.rs's Linux stand-ins) and hands `then` its paths, or
    /// `None` when no picker could open, which `say`s why first. A cancel calls nothing.
    pub fn pick(
        &mut self,
        o: PathPromptOptions,
        then: impl FnOnce(&mut Ecl, Option<Vec<PathBuf>>, &mut Context<Ecl>) + 'static,
        cx: &mut Context<Self>,
    ) {
        picker::pick(cx, o, then);
    }

    /// Opens the picker for attachments: files, or folders when `folder`.
    fn pick_attach(&mut self, folder: bool, cx: &mut Context<Self>) {
        // One picker cannot take both on Windows (FOS_PICKFOLDERS) or in the Linux portal:
        // allowing folders allows nothing else. So Attach asks which, the same on every OS.
        let o = PathPromptOptions {
            files: !folder,
            directories: folder,
            multiple: true,
            prompt: Some("Attach".into()),
        };
        self.pick(
            o,
            |s, paths, cx| {
                let Some(paths) = paths else { return };
                // Images join the next message; the rest, papers too, attach to the chat.
                let (pictures, other) = logic::attach_split(paths);
                s.add_image_files(pictures, cx);
                let open = s.open.clone();
                let live = (open.as_ref().and_then(|id| s.threads.get(id))).is_some_and(|t| t.live);
                match open {
                    _ if other.is_empty() => {}
                    Some(id) if !logic::attach_waits(live, None) => {
                        let later = other.clone();
                        s.try_call(
                            cx,
                            move |c| {
                                let mut last = None;
                                for p in &other {
                                    last = Some(c.attach(&id, p)?);
                                }
                                Ok((id, last))
                            },
                            |s, (id, a), _| {
                                if let (Some(t), Some(a)) = (s.threads.get_mut(&id), a) {
                                    t.files = a.attached;
                                    s.say(a.line);
                                }
                            },
                            move |s, e, _| {
                                if logic::attach_waits(false, Some(&e)) {
                                    s.attach_next(later, true);
                                } else {
                                    s.say(e);
                                }
                            },
                        );
                    }
                    open => s.attach_next(other, open.is_some()),
                }
            },
            cx,
        );
    }

    /// The Code chip. On, it goes off and lets its folder go. Off, it goes on with the folder the
    /// chat already works in, or with the one the "Work here" picker gives; a cancelled pick
    /// leaves it off, and a picker that cannot open turns it on with no folder (a code answer).
    fn code_chip(&mut self, cx: &mut Context<Self>) {
        let folder = match self.open.as_ref().and_then(|id| self.threads.get_mut(id)) {
            Some(t) => &mut t.repo,
            None => &mut self.repo_later,
        };
        if self.chip == Some("Code") {
            self.chip = None;
            *folder = None;
            return;
        }
        if folder.is_some() {
            self.chip = Some("Code");
            return;
        }
        let o = PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Work here".into()),
        };
        self.pick(
            o,
            |s, p, _| {
                let dir = match p.map(|mut p| p.pop()) {
                    Some(None) => return,
                    Some(dir) => dir,
                    None => None,
                };
                match s.open.as_ref().and_then(|id| s.threads.get_mut(id)) {
                    Some(t) => t.repo = dir,
                    None => s.repo_later = dir,
                }
                s.chip = Some("Code");
            },
            cx,
        );
    }
}

/// A place's waiting draft and its images.
struct Held {
    draft: Draft,
    images: Vec<Chip>,
}

/// What one send took from the composer.
#[derive(Clone, Debug, Default, PartialEq)]
struct Draft {
    typed: String,
    chip: Option<&'static str>,
    attach: Vec<String>,
    repo: Option<PathBuf>,
}

impl Draft {
    /// The composer once refused draft `self` comes back to it as `now`: each part returns unless
    /// something new was put there since, and none does under newly typed text.
    fn back_into(self, now: Draft) -> Draft {
        // A send the prompt refused never took the text, so the composer still holds it.
        if !now.typed.is_empty() && now.typed.trim() != self.typed {
            return now;
        }
        Draft {
            typed: self.typed,
            chip: now.chip.or(self.chip),
            attach: if now.attach.is_empty() {
                self.attach
            } else {
                now.attach
            },
            repo: now.repo.or(self.repo),
        }
    }
}

/// Bytes a model needs to run one chat, as the engine's fit counts them: its weights, a KV pool
/// of which 85 % (the default kv_pool_fraction) holds `ctx` tokens, and its 512 MiB headroom.
// ponytail: leaves out the engine's scratch and graph buffers (a few hundred MB); use the engine's
// fit_estimate once #123 makes it public.
pub fn need_bytes(info: &RemoteModel, ctx: f64) -> u64 {
    let ctx = info.context_length.map_or(ctx, |c| ctx.min(c as f64));
    info.weights_bytes
        + (info.kv_bytes_per_token.unwrap_or(0) as f64 * ctx / 0.85) as u64
        + (logic::GIB / 2.) as u64
}

/// The least of `need_bytes` the GPU must hold: an MoE's routed experts can stay in host memory
/// (the engine's host expert layers), so only the rest has to fit.
// ponytail: host RAM is not checked against the experts left there; check it when a machine
// short of RAM shows "Fits" for a model it cannot load.
pub fn floor_bytes(info: &RemoteModel, ctx: f64) -> u64 {
    need_bytes(info, ctx).saturating_sub(info.expert_bytes)
}

/// Why the engine cannot run a file: the tensor types it cannot load, else its architecture.
pub fn cannot_run(info: &RemoteModel) -> String {
    match (info.unloadable.is_empty(), info.arch.is_empty()) {
        (false, _) => format!(
            "Larch can't load {} tensors yet",
            info.unloadable.join(", ")
        ),
        (true, true) => "Larch can't run this file".to_owned(),
        (true, false) => format!("Larch can't run {} models yet", info.arch),
    }
}

/// How long the launch screen shows its full bar once the model is ready.
const BOOT_FULL: Duration = Duration::from_millis(350);

/// The file quants the row prefers, best-liked first.
const QUANTS: [&str; 10] = [
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q5_K_S", "Q6_K", "Q8_0", "Q3_K_L", "Q3_K_M", "Q3_K_S",
];

/// A file quant, as its name spells it, made of tensor types the engine loads (gguf.rs's
/// GGML_TYPES): Q2_K, Q4_1, Q5_1, BF16 and the IQ, TQ and MXFP4 quants are not. Judged by name so
/// the Hub filter and the picker answer before a header arrives; the header then has the last word.
pub fn loads(quant: &str) -> bool {
    [
        "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "Q5_0", "F16", "F32",
    ]
    .iter()
    .any(|p| quant.starts_with(p))
}

/// The listed model whose file is `path`, or whose faster version is: that version is not listed,
/// and its original's card stands for it.
pub fn listing<'a>(models: &'a [ModelEntry], path: &str) -> Option<&'a ModelEntry> {
    let copy = |m: &ModelEntry| m.faster.as_ref().and_then(|f| f.copy.as_deref()) == Some(path);
    models.iter().find(|m| m.path == path || copy(m))
}

/// The model a launch loads again (path, name, expected ms): the last one loaded, while it is still
/// in a library and runs. The expected time is its last load's, else about 1 GB/s from disk plus
/// the engine's start.
pub fn relaunch(prefs: &Prefs, models: &[ModelEntry]) -> Option<(String, String, f64)> {
    let last = prefs.last_model.as_ref()?;
    let m = listing(models, last).filter(|m| m.supported)?;
    let expect = prefs
        .load_ms
        .get(last)
        .map_or(m.size_bytes as f64 / 1e6 + 3000., |&ms| ms as f64);
    Some((last.clone(), m.name.clone(), expect))
}

/// The file a repo's row offers: the best-liked quant that fits in `have` bytes, single files
/// before split sets, else the smallest. Only a repo with no file the engine `loads` offers another
/// quant: a Q2_K that fits is no use when the engine cannot load it. A `draft` file is a
/// speculative-decoding helper (an MTP layer or a small drafter), never the model the repo is
/// named for, so it comes last.
pub fn pick_file(files: &[HfFile], have: Option<u64>) -> Option<&HfFile> {
    let draft = |f: &HfFile| f.file.to_ascii_lowercase().contains("draft");
    let at = |f: &HfFile| QUANTS.iter().position(|q| q.eq_ignore_ascii_case(&f.quant));
    let known = files.iter().any(|f| loads(&f.quant));
    let rank = |f: &HfFile| {
        (if draft(f) { 1000 } else { 0 })
            + (if f.parts.len() > 1 { 100 } else { 0 })
            + at(f).unwrap_or(99)
    };
    let mut c: Vec<&HfFile> = files.iter().filter(|f| !known || loads(&f.quant)).collect();
    c.sort_by_key(|f| (rank(f), f.size_bytes));
    c.iter()
        .find(|f| !draft(f) && have.is_none_or(|h| f.size_bytes + logic::GIB as u64 <= h))
        .or_else(|| c.iter().min_by_key(|f| (draft(f), f.size_bytes)))
        .copied()
}

/// A saved chat as the window shows it; tool results fold into steps.
pub fn saved_items(msgs: &[Message]) -> Vec<Item> {
    let mut out: Vec<Item> = Vec::new();
    for m in msgs {
        // The loop's own messages (larch-agent's `context::MARK`), such as a retry's or the call
        // cap's, are neither the user's nor a step: the turn goes on.
        if matches!(m.role, Role::User | Role::Tool) && m.content.starts_with("[agent]") {
            continue;
        }
        match m.role {
            Role::System => {}
            Role::User => {
                let (text, files) =
                    logic::paper_message(&m.content).unwrap_or((m.content.clone(), Vec::new()));
                let images = m.images.iter().cloned().map(Pic::Saved).collect();
                out.push(Item::User {
                    text,
                    files,
                    images,
                });
            }
            Role::Tool => {
                let line = logic::saved_step(&m.content);
                let step = Step {
                    name: "tool".into(),
                    args: Value::Null,
                    now: line.clone(),
                    done: line,
                    sum: String::new(),
                    st: StepState::Ok,
                    count: 1,
                };
                match out.last_mut() {
                    Some(Item::Steps(s)) => s.push(step),
                    _ => out.push(Item::Steps(vec![step])),
                }
            }
            Role::Assistant => {
                let text = logic::reply_words(&m.content);
                if !text.is_empty() || m.truncated {
                    out.push(Item::Text {
                        text,
                        cut: m.truncated,
                    });
                }
            }
        }
    }
    out
}

fn notify(title: &str, body: &str) {
    // The text goes in the environment or as one argument, never through a shell.
    let r = if cfg!(target_os = "macos") {
        std::process::Command::new("osascript")
            .args(["-e", "on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run", title, body])
            .spawn()
    } else if cfg!(windows) {
        no_window(&mut std::process::Command::new("powershell"))
            .args(["-NoProfile", "-Command", "$t=[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime]::GetTemplateContent(1);$x=$t.GetElementsByTagName('text');$x.Item(0).AppendChild($t.CreateTextNode($env:ECL_T))|Out-Null;$x.Item(1).AppendChild($t.CreateTextNode($env:ECL_B))|Out-Null;[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Larch').Show([Windows.UI.Notifications.ToastNotification]::new($t))"])
            .env("ECL_T", title)
            .env("ECL_B", body)
            .spawn()
    } else {
        std::process::Command::new("notify-send")
            .args(["-a", "Larch", title, body])
            .spawn()
    };
    if let Err(e) = r {
        eprintln!("cannot show a notification: {e}");
    }
}

fn chime() {
    let r = if cfg!(target_os = "macos") {
        std::process::Command::new("afplay")
            .arg("/System/Library/Sounds/Glass.aiff")
            .spawn()
    } else if cfg!(windows) {
        no_window(&mut std::process::Command::new("powershell"))
            .args([
                "-NoProfile",
                "-Command",
                "[System.Media.SystemSounds]::Asterisk.Play()",
            ])
            .spawn()
    } else {
        std::process::Command::new("canberra-gtk-play")
            .args(["-i", "message-new-instant"])
            .spawn()
    };
    if let Err(e) = r {
        eprintln!("cannot play the chime: {e}");
    }
}

/// `el` with the mouse selecting the text it registered (`Selection`): a press on text takes the
/// keys from the composer, so Ctrl+C copies the selection. One per place, never nested.
pub fn selects<E: InteractiveElement>(el: E, cx: &mut Context<Ecl>) -> E {
    el.on_mouse_down(
        MouseButton::Left,
        cx.listener(|s, e: &MouseDownEvent, w, cx| {
            if s.sel.down(e.position, e.click_count, e.modifiers.shift) {
                w.focus(&s.focus);
            }
            cx.notify();
        }),
    )
    .on_mouse_move(cx.listener(|s, e: &MouseMoveEvent, _, cx| {
        if s.sel.dragging && e.pressed_button == Some(MouseButton::Left) {
            s.sel.drag(e.position);
            cx.notify();
        }
    }))
    .on_mouse_up(
        MouseButton::Left,
        cx.listener(|s, _: &MouseUpEvent, _, _| s.sel.dragging = false),
    )
    .on_mouse_up_out(
        MouseButton::Left,
        cx.listener(|s, _: &MouseUpEvent, _, _| s.sel.dragging = false),
    )
}

/// A Ghost `button` on the dark card (`theme::NOTE`): no fill, the card's ink and a faint outline.
pub fn on_note(b: Stateful<Div>) -> Stateful<Div> {
    let ink = gpui::rgb(NOTE_INK);
    b.bg(gpui::transparent_black())
        .text_color(ink)
        .border_color(fade(ink, 0.3))
}

/// Reply Markdown, thread item `item`: paragraphs, lists, headings, code with a Copy button, all
/// selectable.
pub fn markdown(th: &Theme, sel: &Selection, item: usize, text: &str) -> Div {
    let th = *th;
    let id = format!("m{item}");
    let bg = fade(th.accent, 0.25);
    div().flex().flex_col().gap(u(10.)).children(
        logic::md_blocks(text)
            .into_iter()
            .enumerate()
            .map(move |(i, b)| -> AnyElement {
                match b {
                    Block::Heading(l, t) => {
                        let runs = select::plain(&t, serif_font(), th.ink.into());
                        serif_box(if l <= 2 { 26. } else { 21. })
                            .mt(u(4.))
                            .child(sel.text((item, i, 0), t, runs, bg))
                            .into_any_element()
                    }
                    Block::Para(lines) => div()
                        .text_size(u(15.))
                        .line_height(u(24.))
                        .flex()
                        .flex_col()
                        .children(lines.iter().enumerate().map(|(n, l)| {
                            let (t, runs) = rich(&th, l);
                            sel.text((item, i, n), t, runs, bg)
                        }))
                        .into_any_element(),
                    Block::List { ordered, items } => {
                        // As wide as the longest number: "10." in an 18 px column wrapped its dot
                        // onto a line of its own.
                        let digits = items.len().to_string().len() as f32;
                        let mark = if ordered { 10. + 8.5 * digits } else { 18. };
                        div()
                            .flex()
                            .flex_col()
                            .gap(u(4.))
                            .text_size(u(15.))
                            .line_height(u(24.))
                            .children(items.iter().enumerate().map(|(n, it)| {
                                div()
                                    .flex()
                                    .gap(u(8.))
                                    .child(div().flex_none().w(u(mark)).text_color(th.ink3).child(
                                        if ordered {
                                            format!("{}.", n + 1)
                                        } else {
                                            "•".into()
                                        },
                                    ))
                                    .child({
                                        let (t, runs) = rich(&th, it);
                                        sel.text((item, i, n), t, runs, bg).flex_1().min_w_0()
                                    })
                            }))
                            .into_any_element()
                    }
                    Block::Code { lang, body } => {
                        let copy = body.clone();
                        div()
                            .rounded(u(10.))
                            .bg(th.code_bg)
                            .text_color(th.code_ink)
                            .flex()
                            .flex_col()
                            .child(
                                div()
                                    .flex()
                                    .justify_between()
                                    .px(u(14.))
                                    .pt(u(8.))
                                    .text_size(u(11.5))
                                    .text_color(fade(th.code_ink, 0.6))
                                    .child(if lang.is_empty() {
                                        "code".to_owned()
                                    } else {
                                        lang
                                    })
                                    .child(
                                        div()
                                            .id(gpui::ElementId::Name(
                                                format!("{id}-copy-{i}").into(),
                                            ))
                                            .cursor_pointer()
                                            .flex()
                                            .gap(u(4.))
                                            .items_center()
                                            .child(icon("copy", 12., fade(th.code_ink, 0.6)))
                                            .child("Copy")
                                            .on_click(move |_, _, cx| {
                                                cx.write_to_clipboard(ClipboardItem::new_string(
                                                    copy.clone(),
                                                ))
                                            }),
                                    ),
                            )
                            .child(
                                div()
                                    .px(u(14.))
                                    .py(u(10.))
                                    .text_size(u(12.5))
                                    .line_height(u(19.))
                                    .child({
                                        let t = body.trim_end_matches('\n').to_owned();
                                        let mono = font("JetBrains Mono");
                                        let runs = select::plain(&t, mono, th.code_ink.into());
                                        sel.text((item, i, 0), t, runs, bg)
                                    }),
                            )
                            .into_any_element()
                    }
                }
            }),
    )
}

/// The runs of a cited page's text: the quote's bytes `hit` on the accent, the rest as it is.
fn page_runs(th: &Theme, page: &str, hit: Option<std::ops::Range<usize>>) -> Vec<TextRun> {
    let hit = hit.filter(|h| page.is_char_boundary(h.start) && page.is_char_boundary(h.end));
    let (from, to) = hit.map_or((0, 0), |h| (h.start, h.end));
    [(0, from, false), (from, to, true), (to, page.len(), false)]
        .into_iter()
        .filter(|(a, z, _)| z > a)
        .map(|(a, z, on)| TextRun {
            len: z - a,
            font: font("Figtree"),
            color: th.ink.into(),
            background_color: on.then(|| fade(th.accent, 0.3)),
            underline: None,
            strikethrough: None,
        })
        .collect()
}

/// One line of reply text and the runs of its bold, italic and code spans.
pub fn rich(th: &Theme, line: &str) -> (String, Vec<TextRun>) {
    let mut text = String::new();
    let mut runs = Vec::new();
    for (s, k) in logic::spans(line) {
        let mut f = font(if k == Span::Code {
            "JetBrains Mono"
        } else {
            "Figtree"
        });
        match k {
            Span::Bold => f.weight = FontWeight::SEMIBOLD,
            Span::Italic => f.style = gpui::FontStyle::Italic,
            _ => {}
        }
        runs.push(TextRun {
            len: s.len(),
            font: f,
            color: if k == Span::Code {
                th.ink2.into()
            } else {
                th.ink.into()
            },
            background_color: (k == Span::Code).then(|| th.sunk.into()),
            underline: None,
            strikethrough: None,
        });
        text.push_str(&s);
    }
    (text, runs)
}

/// The shadow under a raised card, a popover or a dialog, by theme.
pub fn shadow(th: &Theme) -> Vec<BoxShadow> {
    vec![BoxShadow {
        color: if th.dark {
            gpui::black().opacity(0.5)
        } else {
            fade(th.ink, 0.12)
        },
        offset: point(px(0.), px(8.)),
        blur_radius: px(28.),
        spread_radius: px(0.),
    }]
}

/// Where the model stands, for Home's line and the sidebar's model card.
#[derive(Clone, Copy, PartialEq)]
enum ModelState {
    Empty,
    Loading,
    Ready,
    Stopped,
}

impl ModelState {
    /// The state for engine `health`, whether the engine `failed`, and whether a model is named.
    fn of(health: Health, failed: bool, named: bool) -> ModelState {
        match health {
            _ if failed => ModelState::Stopped,
            Health::Failed => ModelState::Stopped,
            Health::Loading => ModelState::Loading,
            Health::Ready if named => ModelState::Ready,
            _ => ModelState::Empty,
        }
    }
}

#[cfg(test)]
#[test]
fn a_model_switch_reads_as_loading_while_the_old_name_stays() {
    assert!(ModelState::of(Health::Loading, false, true) == ModelState::Loading);
    assert!(ModelState::of(Health::Ready, true, true) == ModelState::Stopped);
    assert!(ModelState::of(Health::Ready, false, true) == ModelState::Ready);
}

/// The sidebar's places: page, print icon, name. Engine sits under Models as This computer, the
/// MCP and Connect pages share Tools, and Settings is at the foot.
const NAV: &[(Page, &str, &str)] = &[
    (Page::Home, "home", "Home"),
    (Page::Chats, "chat", "Chats"),
    (Page::Tasks, "tasks", "Tasks"),
    (Page::Models, "models", "Models"),
    (Page::Mcp, "tools", "Tools"),
];

impl Page {
    /// The sidebar place this page belongs to.
    fn place(self) -> Page {
        match self {
            Page::Chat => Page::Chats,
            Page::Engine => Page::Models,
            Page::Connect => Page::Mcp,
            p => p,
        }
    }
}

impl Ecl {
    pub fn go(&mut self, p: Page, cx: &mut Context<Self>) {
        self.page = p;
        // Another page's pieces would take the old keys.
        self.sel.clear();
        self.task_poll = self.task_poll.wrapping_add(1);
        (self.popover, self.attach_menu) = (false, false);
        self.armed = None;
        // Off the chat, the composer starts a new one.
        if p != Page::Chat {
            self.set_open(None, cx);
        }
        if self.core.is_none() {
            cx.notify();
            return;
        }
        if p == Page::Tasks {
            crate::tasks::refresh(self, cx);
            crate::tasks::poll(self.task_poll, cx);
        }
        if p == Page::Settings && self.stab == STab::Memory {
            self.load_memory(cx);
        }
        // A page's first search goes online (the MCP Registry, Hugging Face). The tour only looks
        // around, so while it is on the search waits for the user's own visit or search; Tools
        // still lists the servers set up on this computer.
        let touring = self.tour.is_some();
        if p == Page::Mcp {
            if self.mcp.searched || touring {
                self.mcp_refresh(cx);
            } else {
                self.mcp_search(false, cx);
            }
        }
        if p == Page::Models
            && self.mtab == MTab::Discover
            && !self.hf.searched
            && self.core.is_some()
            && !touring
        {
            self.hf_search(cx);
        }
        cx.notify();
    }

    fn sidebar(&self, th: &Theme, cx: &mut Context<Self>) -> Div {
        let lotus = format!(
            "brand/lotus-{:02X}{:02X}{:02X}.svg",
            (th.side.r * 255.) as u8,
            (th.side.g * 255.) as u8,
            (th.side.b * 255.) as u8
        );
        let cur = self.page.place();
        let th2 = *th;
        let recent: Vec<ChatEntry> =
            logic::arrange(&self.chats, &self.prefs.pinned, &self.prefs.archived)
                .0
                .into_iter()
                .take(8)
                .cloned()
                .collect();
        let place = |p: Page,
                     ic: &'static str,
                     label: &'static str,
                     badge: Option<AnyElement>,
                     cx: &mut Context<Self>| {
            let on = p == cur;
            div()
                .id(gpui::ElementId::Name(format!("nav-{label}").into()))
                .flex()
                .items_center()
                .gap(u(11.))
                .px(u(10.))
                .h(u(36.))
                .rounded(u(9.))
                .cursor_pointer()
                .text_color(th2.ink2)
                .when(on, |d| {
                    d.bg(th2.cur)
                        .text_color(th2.ink)
                        .font_weight(FontWeight::SEMIBOLD)
                })
                .when(!on, |d| d.hover(move |s| s.bg(fade(th2.cur, 0.6))))
                .on_click(cx.listener(move |s, _, _, cx| s.go(p, cx)))
                .child(picon(&th2, ic, 20., on))
                .child(div().flex_1().text_size(u(14.)).child(label))
                .children(badge)
                .relative()
                .children(crate::tour::marks(
                    self,
                    crate::tour::Spot::Nav(p),
                    &th2,
                    cx,
                ))
        };
        div()
            .w(u(SIDEBAR))
            .flex_none()
            .h_full()
            .bg(th.side)
            .border_r_1()
            .border_color(th.line)
            .px(u(14.))
            .py(u(18.))
            .flex()
            .flex_col()
            .gap(u(2.))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(u(10.))
                    .px(u(4.))
                    .pb(u(16.))
                    // The logo is the idle tree and never moves; the assistant's state shows in the chat.
                    .child(tree(Tree::new(Mood::Ready), 34., 225., true, *th, th.side))
                    .child(serif("Larch", 25.).line_height(u(26.))),
            )
            .child(
                div()
                    .id("new-chat")
                    .flex()
                    .items_center()
                    .gap(u(10.))
                    .px(u(12.))
                    .h(u(38.))
                    .mb(u(14.))
                    .rounded(u(10.))
                    .border_1()
                    .border_color(th.line)
                    .bg(th.card)
                    .cursor_pointer()
                    .hover(move |s| s.bg(th2.sunk))
                    .on_click(cx.listener(|s, _, w, cx| s.new_chat(w, cx)))
                    .relative()
                    .children(crate::tour::marks(self, crate::tour::Spot::NewChat, th, cx))
                    .child(picon(th, "new-chat", 18., false))
                    .child(
                        div()
                            .text_size(u(14.))
                            .font_weight(FontWeight::MEDIUM)
                            .child("New chat"),
                    ),
            )
            .children(NAV.iter().map(|&(p, ic, label)| {
                let badge: Option<AnyElement> = match p {
                    Page::Home if self.need_count() > 0 => Some(
                        div()
                            .min_w(u(18.))
                            .h(u(18.))
                            .px(u(5.))
                            .flex()
                            .items_center()
                            .justify_center()
                            .rounded_full()
                            .bg(th.ember)
                            // Dark themes' ember is light: white on it read at 2.5:1.
                            .text_color(if th.dark {
                                th.page
                            } else {
                                gpui::white().into()
                            })
                            .text_size(u(11.))
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(mid(self.half, self.need_count().to_string()))
                            .into_any_element(),
                    ),
                    // Every running download, as one share of their bytes.
                    Page::Models if !self.dls.is_empty() => {
                        let (got, total) = self
                            .dls
                            .values()
                            .fold((0, 0), |(g, t), d| (g + d.got, t + d.total));
                        Some(
                            div()
                                .flex()
                                .items_center()
                                .gap(u(4.))
                                .text_size(u(11.5))
                                .text_color(th.ink3)
                                .child(icon("down", 12., th.ink3))
                                .child(if total > 0 {
                                    format!("{:.0} %", got as f64 / total as f64 * 100.)
                                } else {
                                    "…".into()
                                })
                                .into_any_element(),
                        )
                    }
                    _ => None,
                };
                place(p, ic, label, badge, cx)
            }))
            .when(!recent.is_empty(), |d| {
                d.child(
                    eyebrow(th, "Recent")
                        .text_color(th.ink3)
                        .px(u(10.))
                        .pt(u(20.))
                        .pb(u(6.)),
                )
            })
            .child(
                div()
                    .id("recent")
                    .flex_1()
                    .overflow_y_scroll()
                    .flex()
                    .flex_col()
                    .children(recent.into_iter().map(|c| {
                        let id = c.id.clone();
                        let on =
                            self.open.as_deref() == Some(id.as_str()) && self.page == Page::Chat;
                        let needs = self.threads.get(&id).is_some_and(|t| t.ask.is_some());
                        let live = self.threads.get(&id).is_some_and(|t| t.live);
                        div()
                            .id(gpui::ElementId::Name(format!("recent-{id}").into()))
                            .group("recent-row")
                            .relative()
                            .flex()
                            .items_center()
                            .gap(u(8.))
                            .px(u(10.))
                            .py(u(6.))
                            .rounded(u(8.))
                            .text_size(u(13.5))
                            .text_color(th2.ink2)
                            .cursor_pointer()
                            .when(on, |d| d.bg(th2.cur).text_color(th2.ink))
                            .hover(move |s| s.bg(fade(th2.cur, 0.6)))
                            .on_mouse_down(
                                MouseButton::Right,
                                self.chat_menu_on_right(&c.id, "recent", cx),
                            )
                            .on_click(cx.listener(move |s, _, _, cx| {
                                s.open_chat(id.clone(), cx);
                                cx.notify()
                            }))
                            .child(self.chat_name(&th2, &c, "recent", 13.5, cx))
                            .when(needs || live, |d| {
                                d.child(div().size(u(7.)).rounded_full().bg(if needs {
                                    th2.ember
                                } else {
                                    th2.sun
                                }))
                            })
                            .child(self.chat_menu(&th2, &c.id, "recent", cx))
                    })),
            )
            .children(self.update_card(th, cx))
            .child(self.status_card(th, cx))
            .child(place(Page::Settings, "settings", "Settings", None, cx))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(u(8.))
                    .mt(u(10.))
                    .pt(u(12.))
                    .px(u(6.))
                    .border_t_1()
                    .border_color(th.line)
                    .child(img(SharedString::from(lotus)).size(u(18.)))
                    .child(
                        div()
                            .text_size(u(12.5))
                            .text_color(th.ink3)
                            .child("Arkaya Labs"),
                    ),
            )
    }

    /// The sidebar's quiet note that a newer release is ready, above the model card; never a
    /// dialog.
    fn update_card(&self, th: &Theme, cx: &mut Context<Self>) -> Option<Div> {
        if self.update_failed {
            return Some(
                div()
                    .flex()
                    .flex_col()
                    .gap(u(8.))
                    .mt(u(10.))
                    .p(u(12.))
                    .rounded(u(12.))
                    .border_1()
                    .border_color(th.line)
                    .bg(th.card)
                    .child(
                        div()
                            .text_size(u(13.))
                            .font_weight(FontWeight::MEDIUM)
                            .child("The update did not install"),
                    )
                    .child(
                        div()
                            .text_size(u(12.))
                            .text_color(th.ink3)
                            .child(FAILED_UPDATE),
                    )
                    .child(
                        div()
                            .id("update-failed-ok")
                            .text_size(u(12.5))
                            .text_color(th.ink2)
                            .underline()
                            .cursor_pointer()
                            .on_click(cx.listener(|s, _, _, cx| {
                                s.update_failed = false;
                                s.update_later = true;
                                cx.notify();
                            }))
                            .child("Dismiss"),
                    ),
            );
        }
        let version = self.update_offer().filter(|_| !self.update_later)?;
        let notes = crate::update::notes(&version);
        Some(
            div()
                .flex()
                .flex_col()
                .gap(u(8.))
                .mt(u(10.))
                .p(u(12.))
                .rounded(u(12.))
                .border_1()
                .border_color(th.line)
                .bg(th.card)
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap(u(8.))
                        .text_size(u(13.))
                        .font_weight(FontWeight::MEDIUM)
                        .child(icon("up", 14., th.accent))
                        .child(format!("Larch {version} is ready")),
                )
                .child(
                    div()
                        .text_size(u(12.))
                        .text_color(th.ink3)
                        .child("Larch restarts to install it. Your chats are kept."),
                )
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap(u(12.))
                        .child(button(
                            th,
                            "update-restart",
                            "Restart to update",
                            Btn::Primary,
                            cx.listener(|s, _, w, cx| s.restart_to_update(w, cx)),
                        ))
                        .child(
                            div()
                                .id("update-notes")
                                .text_size(u(12.5))
                                .text_color(th.ink2)
                                .underline()
                                .cursor_pointer()
                                .on_click(move |_, _, cx| cx.open_url(&notes))
                                .child("What's new"),
                        ),
                ),
        )
    }

    /// The sidebar's model card: what runs, how full the GPU is, and how fast it last wrote.
    /// Opens This computer (Engine) for advanced users, else Models.
    fn status_card(&self, th: &Theme, cx: &mut Context<Self>) -> Stateful<Div> {
        let th2 = *th;
        let to = if self.prefs.advanced {
            Page::Engine
        } else {
            Page::Models
        };
        let card = div()
            .id("status-card")
            .flex()
            .flex_col()
            .gap(u(8.))
            .mt(u(10.))
            .mb(u(6.))
            .p(u(12.))
            .rounded(u(12.))
            .border_1()
            .border_color(th.line)
            .bg(th.card)
            .cursor_pointer()
            .hover(move |s| s.bg(th2.sunk))
            .on_click(cx.listener(move |s, _, _, cx| s.go(to, cx)));
        // While a switch loads, `st.model` still names the old model: the state comes first.
        let state = self.model_state();
        let Some(name) = self.model_name().filter(|_| state == ModelState::Ready) else {
            let (head, note) = match state {
                ModelState::Loading => ("Loading a model", "It is ready in a moment"),
                ModelState::Stopped => ("The model stopped", "Load it again in Models"),
                _ => ("No model loaded", "Pick one in Models"),
            };
            return card
                .child(
                    div()
                        .text_size(u(13.))
                        .font_weight(FontWeight::MEDIUM)
                        .child(head),
                )
                .child(div().text_size(u(12.)).text_color(th.ink3).child(note));
        };
        // The same figures as Home's This computer card: weights in green, chats in gold.
        let total = self.gpu_bytes().filter(|t| *t > 0);
        let last = self.last.as_ref();
        let weights = last.map_or(0, |s| s.weights_bytes);
        let chats = last.map_or(0, |s| s.kv_total_bytes);
        let used = last
            .and_then(|s| s.gpu_used_bytes)
            .unwrap_or(weights + chats);
        let speed = last
            .filter(|s| s.tok_s > 0.)
            .map(|s| format!("{:.0} tok/s", s.tok_s));
        let mem = total.map(|t| {
            format!(
                "{} of {} used",
                logic::gb(used).trim_end_matches(" GB"),
                logic::gb(t).replace(".0 GB", " GB")
            )
        });
        let line2 = match (speed, mem) {
            (Some(v), Some(m)) => format!("{v} · {m}"),
            (v, m) => v.or(m).unwrap_or_else(|| "Ready".into()),
        };
        let share = |b: u64, t: u64| gpui::relative((b as f32 / t as f32).clamp(0., 1.));
        card.child(
            div()
                .flex()
                .items_center()
                .gap(u(8.))
                .text_size(u(13.))
                .font_weight(FontWeight::MEDIUM)
                .child(div().flex_none().size(u(8.)).rounded_full().bg(th.sun))
                .child(div().min_w_0().truncate().child(name)),
        )
        .when_some(total, |d, t| {
            d.child(
                div()
                    .h(u(5.))
                    .flex()
                    .rounded_full()
                    .overflow_hidden()
                    .bg(th.sunk)
                    .child(div().h_full().w(share(weights, t)).bg(th.accent))
                    .child(div().h_full().w(share(chats, t)).bg(th.sun)),
            )
        })
        .child(
            div()
                .text_size(u(12.))
                .text_color(th.ink3)
                .truncate()
                .child(line2),
        )
    }

    fn topbar(
        &self,
        th: &Theme,
        title: AnyElement,
        extra: Vec<AnyElement>,
        m: Mood,
        cx: &mut Context<Self>,
    ) -> Div {
        let th2 = *th;
        // A chat shows the assistant large at its top, so the chip would only repeat it.
        let chip = self.page != Page::Chat && (self.prefs.advanced || self.page != Page::Home);
        // A chat's or a task's title truncates; a page's name keeps its width and the model and
        // GPU pills give way ("Home" broke a letter a line beside them).
        let shrinks = matches!(self.page, Page::Chat | Page::Tasks);
        let narrow = self.room < 760.;
        div()
            .h(u(56.))
            .flex_none()
            .flex()
            .items_center()
            .gap(u(10.))
            .px(u(28.))
            .border_b_1()
            .border_color(th.line)
            .child(div().flex_1().when(shrinks, |d| d.min_w_0()).child(title))
            .children(extra)
            .when(chip, |d| {
                d.child(
                    div()
                        .id("assistant-chip")
                        .flex()
                        .items_center()
                        .gap(u(8.))
                        .flex_none()
                        .pl(u(8.))
                        .pr(u(if narrow { 8. } else { 14. }))
                        .py(u(4.))
                        .rounded_full()
                        .border_1()
                        .border_color(th.line)
                        .bg(th.card)
                        .cursor_pointer()
                        .hover(move |s| s.bg(th2.sunk))
                        .on_click(cx.listener(|s, _, _, cx| {
                            s.popover = !s.popover;
                            s.pop_ix = 0;
                            cx.notify()
                        }))
                        .when(self.prefs.sun || narrow, |d| {
                            d.child(tree(self.sun, 26., 180., self.still(), *th, th.card))
                        })
                        // A narrow page keeps the tree alone, so the tabs beside it keep their room.
                        // The words keep a word space apart, not the tree's 8.
                        .when(!narrow, |d| {
                            d.child(
                                div()
                                    .flex()
                                    .gap(u(4.))
                                    .text_size(u(13.))
                                    .child(
                                        div().font_weight(FontWeight::SEMIBOLD).child("Assistant"),
                                    )
                                    .child(
                                        div()
                                            .text_color(th.ink2)
                                            .child(format!("· {}", self.status_line(m))),
                                    ),
                            )
                        }),
                )
            })
    }

    fn popover(&self, th: &Theme, cx: &mut Context<Self>) -> Div {
        let needs: Vec<(String, (String, String), bool)> = self
            .needs()
            .filter_map(|(id, t)| t.ask_line().map(|l| (id.clone(), l, t.quick_answer())))
            .collect();
        let total = needs.len();
        let ix = self.pop_ix.min(total.saturating_sub(1));
        let ink = gpui::rgb(NOTE_INK);
        let pop = |t: String| self.sel.label(select::POP, t);
        let mut body = div().flex().flex_col().gap(u(8.));
        if total == 0 {
            body = body
                .child(eyebrow(th, "Assistant").text_color(gpui::rgb(NOTE_EYEBROW)))
                .child(
                    div()
                        .text_size(u(15.))
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(pop(self.status_line(self.sun.mood))),
                )
                .child(
                    div()
                        .text_size(u(13.))
                        .text_color(fade(ink, 0.7))
                        .child(pop("Nothing needs you right now.".into())),
                );
        } else {
            let (id, (q, chat), quick) = needs[ix].clone();
            let (yes_id, no_id, open_id) = (id.clone(), id.clone(), id.clone());
            body = body
                .child(
                    eyebrow(th, format!("Needs you · {} of {total}", ix + 1))
                        .text_color(gpui::rgb(NOTE_EYEBROW)),
                )
                .child(
                    div()
                        .text_size(u(16.))
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(pop(q)),
                )
                .child(
                    div()
                        .text_size(u(13.))
                        .text_color(fade(ink, 0.7))
                        .child(pop(chat)),
                )
                .child(
                    div()
                        .flex()
                        .gap(u(8.))
                        .pt(u(4.))
                        // An outside call's arguments, a long command or a change show on the
                        // chat's card only: Allow here sent what the user never saw, and Not now
                        // on a change undid it.
                        .when(quick, |d| {
                            d.child(button(
                                th,
                                "pop-yes",
                                "Allow",
                                Btn::Light,
                                cx.listener(move |s, _, _, cx| {
                                    s.answer(yes_id.clone(), true, None, cx)
                                }),
                            ))
                        })
                        .child(on_note(button(
                            th,
                            "pop-open",
                            "Open chat",
                            Btn::Ghost,
                            cx.listener(move |s, _, _, cx| s.open_chat(open_id.clone(), cx)),
                        )))
                        .when(quick, |d| {
                            d.child(on_note(button(
                                th,
                                "pop-no",
                                "Not now",
                                Btn::Ghost,
                                cx.listener(move |s, _, _, cx| {
                                    s.answer(no_id.clone(), false, None, cx)
                                }),
                            )))
                        })
                        // In the same row: on a row of its own it read as a stray button.
                        .when(total > 1, |d| {
                            d.child(div().flex_1()).child(
                                button(
                                    th,
                                    "pop-next",
                                    "Next",
                                    Btn::Ghost,
                                    cx.listener(move |s, _, _, cx| {
                                        s.pop_ix = (s.pop_ix + 1) % total;
                                        cx.notify()
                                    }),
                                )
                                .bg(gpui::transparent_black())
                                .text_color(fade(ink, 0.7))
                                .border_color(gpui::transparent_black()),
                            )
                        }),
                );
        }
        let el = div()
            .absolute()
            .top(u(62.))
            .right(u(24.))
            .w(u(400.))
            .p(u(20.))
            .rounded(u(16.))
            .bg(gpui::rgb(NOTE))
            .text_color(ink)
            .shadow(shadow(th))
            .occlude()
            .child(body);
        selects(el, cx)
    }

    fn toast_el(&self, th: &Theme, cx: &mut Context<Self>) -> Option<Div> {
        // Its text selected, it waits for the copy.
        let t = (self.toast.as_ref())
            .filter(|t| t.at.elapsed() < t.lasts() || self.sel.holds(select::TOAST))?;
        let act = t.act.clone();
        Some(
            div()
                .absolute()
                .bottom(u(24.))
                .left(u(0.))
                .right(u(0.))
                .flex()
                .justify_center()
                .child(selects(
                    div()
                        .flex()
                        .items_center()
                        .gap(u(14.))
                        .max_w(u(640.))
                        .px(u(18.))
                        .py(u(10.))
                        .rounded(u(12.))
                        .bg(gpui::rgb(NOTE))
                        .text_color(gpui::rgb(NOTE_INK))
                        .text_size(u(13.5))
                        .shadow(shadow(th))
                        .occlude()
                        // min_w 0: a long path has no space to wrap at, and would stretch the toast.
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .child(self.sel.label(select::TOAST, t.text.clone())),
                        )
                        .children(act.map(|(label, a)| {
                            div()
                                .id("toast-act")
                                .cursor_pointer()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(gpui::rgb(0xb5dcc6))
                                .child(label)
                                .on_click(cx.listener(move |s, _, _, cx| {
                                    s.act(a.clone(), cx);
                                    cx.notify()
                                }))
                        }))
                        .child(
                            div()
                                .id("toast-x")
                                .cursor_pointer()
                                .child(icon("x", 12., gpui::rgb(0xbdb5a6)))
                                .on_click(cx.listener(|s, _, _, cx| {
                                    s.set_toast(None);
                                    cx.notify()
                                })),
                        ),
                    cx,
                )),
        )
    }

    /// The launch screen: the assistant working in the middle and a bar timed by the last load,
    /// held short of full until the engine says Ready.
    fn boot_el(&self, th: &Theme, window: &mut Window, cx: &mut Context<Self>) -> Option<Div> {
        let (name, expect) = self.boot.clone()?;
        let since = self
            .loading
            .as_ref()
            .map_or(Duration::ZERO, |l| l.1.elapsed());
        let frac = if self.st.health == Health::Ready {
            1.
        } else {
            (since.as_millis() as f64 / expect.max(1.)).min(0.95) as f32
        };
        // The bar moves between the tree's frames too, when the tree holds still.
        window.request_animation_frame();
        Some(
            div()
                .absolute()
                .size_full()
                .bg(th.page)
                .flex()
                .flex_col()
                .items_center()
                .justify_center()
                .gap(u(18.))
                .occlude()
                .child(tree(self.sun, 150., 225., self.still(), *th, th.page))
                .child(serif(format!("Loading {name}"), 30.))
                .child(
                    div()
                        .w(u(360.))
                        .child(bar(th, &[(frac, th.sun.into())], 6.)),
                )
                .child(
                    div()
                        .text_size(u(13.))
                        .text_color(th.ink3)
                        .child(format!("{:.0} %", frac * 100.)),
                )
                .child(button(
                    th,
                    "boot-skip",
                    "Use Larch meanwhile",
                    Btn::Ghost,
                    cx.listener(|s, _, _, cx| {
                        s.boot = None;
                        cx.notify()
                    }),
                )),
        )
    }

    fn failed_el(&self, th: &Theme, msg: &str, cx: &mut Context<Self>) -> Div {
        // A start that failed is no GPU fault: a missing library or a bad data folder read as
        // "The GPU stopped", and a restart alone would not fix it.
        let (title, hint) = match msg.starts_with(NO_START) {
            true => (
                "Larch could not start",
                "Restart once what the message names is fixed. Your chats are saved.",
            ),
            false => (
                "The GPU stopped",
                "Only a fresh start can use it again. Your chats are saved.",
            ),
        };
        let alert = |t: &str| self.sel.label(select::ALERT, t.to_owned());
        let el = div()
            .absolute()
            .size_full()
            .bg(fade(th.page, 0.85))
            .flex()
            .items_center()
            .justify_center()
            .occlude()
            .child(
                card(th)
                    .w(u(460.))
                    .p(u(28.))
                    .gap(u(14.))
                    .shadow(shadow(th))
                    .child(tree(self.sun, 64., 225., true, *th, th.card))
                    .child(serif_box(30.).child(alert(title)))
                    .child(
                        div()
                            .text_size(u(14.))
                            .text_color(th.ink2)
                            .child(alert(msg)),
                    )
                    .child(
                        div()
                            .text_size(u(13.))
                            .text_color(th.ink3)
                            .child(alert(hint)),
                    )
                    .child(
                        div()
                            .flex()
                            .gap(u(8.))
                            .child(button(
                                th,
                                "restart",
                                "Restart Larch",
                                Btn::Primary,
                                |_, _, cx| cx.restart(),
                            ))
                            .child(button(
                                th,
                                "dismiss-failed",
                                "Not now",
                                Btn::Ghost,
                                cx.listener(|s, _, _, cx| {
                                    s.failed = None;
                                    cx.notify()
                                }),
                            )),
                    ),
            );
        selects(el, cx)
    }

    /// The composer's Attach button and its menu (Files or Folder), which opens upward: in a chat
    /// the composer sits at the window's foot, where a menu below would be pushed over the button.
    fn attach_el(&self, th: &Theme, cx: &mut Context<Self>) -> Div {
        use crate::context_menu::{card, row};
        let open = self.attach_menu;
        let pick = |id, label, folder| {
            row(th, id, label, true).on_click(cx.listener(move |s, _, _, cx| {
                cx.stop_propagation();
                s.attach_menu = false;
                cx.notify();
                s.pick_attach(folder, cx)
            }))
        };
        div()
            .relative()
            .flex_none()
            // The button only opens the menu, on mouse down with this frame's `open`: the card's
            // mouse-down-out closes it first and redraws, so a click toggle would reopen it.
            .child(
                pill(th, "c-attach", "+ Attach", open, |_, _, _| {}).on_mouse_down(
                    MouseButton::Left,
                    cx.listener(move |s, _, _, cx| {
                        if !open {
                            s.attach_menu = true;
                            cx.notify()
                        }
                    }),
                ),
            )
            .when(open, |d| {
                let menu = card(th)
                    .on_mouse_down_out(cx.listener(|s, _, _, cx| {
                        s.attach_menu = false;
                        cx.notify()
                    }))
                    .child(pick("attach-files", "Files…", false))
                    .child(pick("attach-folder", "Folder…", true));
                d.child(
                    div().absolute().bottom_full().left_0().mb(u(6.)).child(
                        gpui::deferred(
                            gpui::anchored()
                                .anchor(gpui::Corner::BottomLeft)
                                .snap_to_window_with_margin(px(8.))
                                .child(menu),
                        )
                        .with_priority(1),
                    ),
                )
            })
    }

    /// The composer: the field, its chips and the send or stop button.
    pub fn composer_el(&self, th: &Theme, cx: &mut Context<Self>) -> Stateful<Div> {
        let live = self
            .open
            .as_ref()
            .and_then(|id| self.threads.get(id))
            .is_some_and(|t| t.live);
        let repo = self
            .open
            .as_ref()
            .and_then(|id| self.threads.get(id))
            .and_then(|t| t.repo.clone())
            .or(self.repo_later.clone());
        let chips: &[&'static str] = if self.prefs.advanced {
            &["Research", "Code"]
        } else {
            &["Research", "Write"]
        };
        // Each attached file or folder, with the × that takes it off: the chat's own (`true`),
        // then those waiting for the next message.
        let attached: Vec<(String, bool)> =
            (self.open.as_ref().and_then(|id| self.threads.get(id)))
                .map_or(&[][..], |t| &t.files[..])
                .iter()
                .map(|f| (f.clone(), true))
                .chain(self.attach_later.iter().map(|f| (f.clone(), false)))
                .collect();
        // The documents' state as the agent last reported it, whatever brought the documents
        // (attached files, a folder, a link in a message).
        let docs_chip = (self.open.as_ref().and_then(|id| self.threads.get(id)))
            .and_then(|t| t.docs.chip.clone().map(|c| (c, t.docs.slower)));
        let placeholder = if self.open.is_some() {
            "Reply…"
        } else if self.prefs.advanced {
            "Ask anything, attach papers, or point me at a repo…"
        } else {
            "What would you like help with?"
        };
        self.composer
            .update(cx, |i, _| i.set_placeholder(placeholder));
        let focus = self.composer.focus_handle(cx);
        let (th2, accent) = (*th, th.accent);
        // The next message's images; one in a message on its way is in the thread already.
        let waiting: Vec<&Chip> = self.chips.iter().filter(|c| !c.sending).collect();
        div()
            .id("composer")
            .relative()
            .children(crate::tour::marks(
                self,
                crate::tour::Spot::Composer,
                th,
                cx,
            ))
            .flex()
            .flex_col()
            .gap(u(12.))
            .p(u(16.))
            .pb(u(12.))
            .rounded(u(16.))
            .bg(th.card)
            .border_1()
            .border_color(th.line)
            .border_b_2()
            .cursor_text()
            .on_click(move |_, w, _| w.focus(&focus))
            // Image files dropped here join the message; other files attach with + Attach.
            .drag_over::<ExternalPaths>(move |st, _, _, _| st.border_color(accent))
            .on_drop(cx.listener(|s, paths: &ExternalPaths, _, cx| {
                let (pictures, rest): (Vec<PathBuf>, Vec<PathBuf>) =
                    (paths.paths().iter().cloned()).partition(|p| images::image_path(p));
                if pictures.is_empty() && !rest.is_empty() {
                    s.say("Only images can be dropped here. Attach other files with + Attach.");
                }
                s.add_image_files(pictures, cx);
                cx.notify();
            }))
            .when(!waiting.is_empty(), |d| {
                d.child(
                    div()
                        .flex()
                        .flex_wrap()
                        .gap(u(8.))
                        .children(waiting.iter().map(|c| {
                            let gone = c.reference.clone();
                            div()
                                .flex()
                                .flex_col()
                                .items_center()
                                .gap(u(2.))
                                .child(
                                    div()
                                        .relative()
                                        .size(u(64.))
                                        .rounded(u(10.))
                                        .overflow_hidden()
                                        .border_1()
                                        .border_color(th2.line)
                                        .bg(th2.sunk)
                                        .child(
                                            img(c.image.clone())
                                                .size_full()
                                                .object_fit(ObjectFit::Cover),
                                        )
                                        .child(
                                            div()
                                                .id(SharedString::from(format!("img-rm-{gone}")))
                                                .absolute()
                                                .top(u(3.))
                                                .right(u(3.))
                                                .size(u(20.))
                                                .rounded_full()
                                                .flex()
                                                .items_center()
                                                .justify_center()
                                                .bg(fade(th2.ink, 0.65))
                                                .cursor_pointer()
                                                .child(icon("x", 12., th2.app))
                                                .on_click(cx.listener(move |s, _, _, cx| {
                                                    (s.chips).retain(|c| {
                                                        c.sending || c.reference != gone
                                                    });
                                                    cx.notify()
                                                })),
                                        ),
                                )
                                .children(c.tokens.map(|n| {
                                    div()
                                        .text_size(u(11.))
                                        .text_color(th2.ink3)
                                        .child(format!("{n} tokens"))
                                }))
                        })),
                )
            })
            // What is attached, each with its ×, over the text it goes with.
            .when(!attached.is_empty(), |d| {
                d.child(
                    div()
                        .flex()
                        .flex_wrap()
                        .gap(u(8.))
                        .children(attached.into_iter().map(|(f, saved)| {
                            let name = logic::base_name(&f).to_owned();
                            div()
                                .id(SharedString::from(format!("att-{f}")))
                                .flex()
                                .items_center()
                                .gap(u(6.))
                                .pl(u(12.))
                                .pr(u(6.))
                                .h(u(34.))
                                .rounded_full()
                                .bg(th2.sunk)
                                .text_size(u(13.))
                                .text_color(th2.ink2)
                                .child(div().max_w(u(220.)).truncate().child(name))
                                .child(
                                    div()
                                        .id(SharedString::from(format!("att-rm-{f}")))
                                        .size(u(22.))
                                        .rounded_full()
                                        .flex()
                                        .items_center()
                                        .justify_center()
                                        .group("att-rm")
                                        .cursor_pointer()
                                        .hover(move |d| d.bg(fade(th2.ink3, 0.18)))
                                        .child(
                                            icon("x", 12., th2.ink3)
                                                .group_hover("att-rm", move |d| {
                                                    d.text_color(th2.ink)
                                                }),
                                        )
                                        .on_click(cx.listener(move |s, _, _, cx| {
                                            s.unattach(f.clone(), saved, cx);
                                            cx.notify()
                                        })),
                                )
                        })),
                )
            })
            .child(
                div()
                    .text_size(u(15.))
                    .line_height(u(22.))
                    .text_color(th.ink)
                    .child(self.composer.clone()),
            )
            // The chips wrap in their own box, so the send button keeps its place at the right.
            .child(
                div()
                    .flex()
                    .items_end()
                    .gap(u(8.))
                    .child(
                        div()
                            .flex()
                            .flex_1()
                            .flex_wrap()
                            .items_center()
                            .gap(u(8.))
                            .child(self.attach_el(th, cx))
                            .when_some(docs_chip, |d, (c, slower)| {
                                d.child(tag(th, c)).when_some(slower, |d, l| {
                                    d.child(div().text_size(u(12.5)).text_color(th.ink3).child(l))
                                })
                            })
                            .children(chips.iter().map(|c| {
                                let c = *c;
                                // Code names the folder it works in.
                                let label = match (c, &repo) {
                                    ("Code", Some(r)) => format!(
                                        "Code · {}",
                                        logic::base_name(&r.display().to_string())
                                    ),
                                    _ => c.to_owned(),
                                };
                                pill(
                                    th,
                                    format!("c-{c}"),
                                    label,
                                    self.chip == Some(c),
                                    cx.listener(move |s, _, _, cx| {
                                        if c == "Code" {
                                            s.code_chip(cx)
                                        } else {
                                            s.chip = if s.chip == Some(c) { None } else { Some(c) };
                                        }
                                        cx.notify()
                                    }),
                                )
                            })),
                    )
                    .child(
                        div()
                            .id("send")
                            .flex_none()
                            .size(u(40.))
                            .rounded_full()
                            .flex()
                            .items_center()
                            .justify_center()
                            .cursor_pointer()
                            .bg(if live { th.ink } else { th.accent })
                            .hover(|s| s.opacity(0.9))
                            .on_click(cx.listener(move |s, _, w, cx| {
                                if live {
                                    s.stop_reply(cx)
                                } else {
                                    s.send(w, cx)
                                }
                            }))
                            .child(icon(
                                if live { "stop" } else { "up" },
                                16.,
                                if live { th.app } else { th.on_accent },
                            )),
                    ),
            )
    }

    fn home(&self, th: &Theme, cx: &mut Context<Self>) -> Div {
        let hour = chrono_hour();
        // Behind the welcome, the name is still being asked.
        let name = match self.tour {
            Some(crate::tour::At::Welcome) => "",
            _ => self.prefs.name.trim(),
        };
        let greet = if name.is_empty() {
            format!("{}.", logic::greeting(hour))
        } else {
            format!("{}, {name}.", logic::greeting(hour))
        };
        // After the first listing (it spends `autoload`; the demo brings its own models), an
        // empty list means nothing is on this computer to load yet.
        let first = (self.core.is_some() && !self.autoload && self.models.is_empty())
            .then(|| crate::pages::first_model(self, th, cx));
        let needs = self.needs_card(th, cx);
        let cont = self.continue_card(th, cx);
        let still = self.still();
        let sub = match (self.model_state(), self.model_name()) {
            (ModelState::Loading, _) => "Loading a model…".into(),
            (ModelState::Stopped, _) => "The model stopped. Load it again from Models.".into(),
            (ModelState::Ready, Some(m)) if self.st.device.is_empty() => format!("{m} is ready."),
            (ModelState::Ready, Some(m)) => format!("{m} is ready on your {}.", self.st.device),
            _ if first.is_some() => "Download a model to start. It runs on this computer.".into(),
            _ => "Pick a model in Models to start.".into(),
        };
        // One column, read top to bottom: who is here, the box to ask in, what waits on you,
        // then what to pick up again.
        div()
            .flex()
            .flex_col()
            .items_center()
            .px(u(40.))
            .pt(u(52.))
            .child(
                div()
                    .w_full()
                    .max_w(u(900.))
                    .flex()
                    .flex_col()
                    .gap(u(26.))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap(u(22.))
                            .when(self.prefs.sun, |d| {
                                d.child(tree(self.sun, 84., 225., still, *th, th.app))
                            })
                            .child(
                                div()
                                    .min_w_0()
                                    .flex()
                                    .flex_col()
                                    .gap(u(6.))
                                    .child(serif(greet, 46.))
                                    .child(div().text_size(u(15.)).text_color(th.ink2).child(sub)),
                            ),
                    )
                    .child(self.composer_el(th, cx))
                    .children(first)
                    .children(needs)
                    .child(
                        div()
                            .flex()
                            .gap(u(16.))
                            .children(cont.map(|c| c.flex_1().min_w_0()))
                            .when(self.prefs.advanced, |d| {
                                d.child(self.computer_card(th).flex_1().min_w_0())
                            }),
                    ),
            )
    }

    fn needs_card(&self, th: &Theme, cx: &mut Context<Self>) -> Option<Div> {
        let th2 = *th;
        let items: Vec<(String, String, String, bool, bool)> = self
            .needs()
            .filter_map(|(id, t)| {
                let code = matches!(t.ask, Some(Ask::Code { .. }));
                t.ask_line()
                    .map(|(q, c)| (id.clone(), q, c, code, t.quick_answer()))
            })
            .collect();
        // Shown when empty too, so Home always says whether anything waits on the user.
        if items.is_empty() {
            return Some(
                card(th).child(eyebrow(th, "Needs you")).child(
                    div()
                        .text_size(u(14.5))
                        .text_color(th.ink2)
                        .child("Nothing needs you."),
                ),
            );
        }
        let row = |d: Div| {
            d.flex()
                .flex_wrap()
                .items_center()
                .gap_x(u(16.))
                .gap_y(u(8.))
                .p(u(14.))
                .pl(u(16.))
                .rounded(u(10.))
                .bg(th2.sunk)
                .border_l(u(3.))
                .border_color(th2.ember)
        };
        Some(
            card(th)
                .child(eyebrow(
                    th,
                    if self.prefs.advanced {
                        format!("Needs you · {}", items.len())
                    } else {
                        "Needs you".into()
                    },
                ))
                .children(items.into_iter().take(3).map(|(id, q, chat, code, quick)| {
                    let (a, b, c, o) = (id.clone(), id.clone(), id.clone(), id.clone());
                    row(div())
                        .child(
                            div()
                                .flex_1()
                                .min_w(u(240.))
                                .flex()
                                .flex_col()
                                .gap(u(3.))
                                .child(
                                    div()
                                        .text_size(u(14.5))
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .child(q),
                                )
                                .child(div().text_size(u(12.5)).text_color(th2.ink3).child(chat)),
                        )
                        // What only the chat's card shows is answered there.
                        .when(!quick, |d| {
                            d.child(div().flex().child(button(
                                &th2,
                                format!("n-open-{id}"),
                                "Open chat",
                                Btn::Primary,
                                cx.listener(move |s, _, _, cx| s.open_chat(o.clone(), cx)),
                            )))
                        })
                        .when(quick, |d| {
                            d.child(
                                div()
                                    .flex()
                                    .flex_wrap()
                                    .gap(u(8.))
                                    .child(button(
                                        &th2,
                                        format!("n-yes-{id}"),
                                        if code { "Allow once" } else { "Allow" },
                                        Btn::Primary,
                                        cx.listener(move |s, _, _, cx| {
                                            s.answer(a.clone(), true, None, cx)
                                        }),
                                    ))
                                    .when(code, |d| {
                                        d.child(button(
                                            &th2,
                                            format!("n-folder-{id}"),
                                            "Always in this folder",
                                            Btn::Outline,
                                            cx.listener(move |s, _, _, cx| {
                                                s.answer(
                                                    c.clone(),
                                                    true,
                                                    Some(Decision::Folder),
                                                    cx,
                                                )
                                            }),
                                        ))
                                    })
                                    .child(button(
                                        &th2,
                                        format!("n-no-{id}"),
                                        "No",
                                        Btn::Ghost,
                                        cx.listener(move |s, _, _, cx| {
                                            s.answer(b.clone(), false, None, cx)
                                        }),
                                    )),
                            )
                        })
                })),
        )
    }

    fn continue_card(&self, th: &Theme, cx: &mut Context<Self>) -> Option<Div> {
        let c = self.chats.first()?;
        let id = c.id.clone();
        let last = self.threads.get(&c.id).and_then(|t| {
            t.items.iter().rev().find_map(|i| match i {
                Item::Text { text, .. } => logic::md_blocks(text).into_iter().find_map(|b| {
                    if let Block::Para(l) = b {
                        Some(
                            logic::spans(&l.join(" "))
                                .into_iter()
                                .map(|(s, _)| s)
                                .collect::<String>(),
                        )
                    } else {
                        None
                    }
                }),
                _ => None,
            })
        });
        Some(
            card(th)
                .child(eyebrow(th, "Continue"))
                .child(serif(c.title.clone(), 25.))
                .children(last.filter(|_| self.prefs.advanced).map(|l| {
                    lines(3., 21.)
                        .text_size(u(14.))
                        .text_color(th.ink2)
                        .child(format!("“…{}”", logic::clip(&l, 140)))
                }))
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap(u(12.))
                        .child(button(
                            th,
                            "continue",
                            "Open chat",
                            Btn::Primary,
                            cx.listener(move |s, _, _, cx| s.open_chat(id.clone(), cx)),
                        ))
                        .child(
                            div()
                                .text_size(u(13.))
                                .text_color(th.ink3)
                                .child(logic::ago(c.updated_at, now_ms())),
                        ),
                ),
        )
    }

    /// This computer's health and GPU memory.
    pub fn computer_card(&self, th: &Theme) -> Div {
        let (title, color) = match (self.failed.is_some(), self.st.health) {
            (true, _) | (_, Health::Failed) => ("Stopped", th.lose),
            (_, Health::Loading) => ("Loading a model", th.ink),
            (_, Health::Ready) => ("Healthy", th.ink),
            (_, Health::Empty) => ("No model loaded", th.ink2),
        };
        let total = self.gpu_bytes();
        let s = self.last.as_ref();
        let weights = s.map_or(0, |s| s.weights_bytes);
        let chats = s.map_or(0, |s| s.kv_total_bytes);
        let used = s.and_then(|s| s.gpu_used_bytes).unwrap_or(weights + chats);
        card(th)
            .child(eyebrow(th, "This computer"))
            .child(serif(title, 27.).text_color(color))
            .child(
                div()
                    .text_size(u(13.5))
                    .text_color(th.ink2)
                    .child(match total {
                        Some(t) => format!(
                            "GPU memory · {} of {}",
                            logic::gb(used).trim_end_matches(" GB"),
                            logic::gb(t)
                        ),
                        None if self.st.device.is_empty() => "Running on the CPU".into(),
                        None => self.st.device.clone(),
                    }),
            )
            .when_some(total, |d, t| {
                let f = |b: u64| b as f32 / t as f32;
                d.child(bar(
                    th,
                    &[(f(weights), th.accent.into()), (f(chats), th.sun.into())],
                    8.,
                ))
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .gap(u(14.))
                        .text_size(u(12.5))
                        .text_color(th.ink2)
                        .child(legend(
                            th.accent,
                            format!("Weights {:.1}", weights as f64 / logic::GIB),
                        ))
                        .child(legend(
                            th.sun,
                            format!("Chats {:.1}", chats as f64 / logic::GIB),
                        ))
                        .child(legend(
                            th.sunk,
                            format!("Free {:.1}", t.saturating_sub(used) as f64 / logic::GIB),
                        )),
                )
            })
    }

    fn chats_page(&self, th: &Theme, cx: &mut Context<Self>) -> Div {
        let th2 = *th;
        let (shown, archived) =
            logic::arrange(&self.chats, &self.prefs.pinned, &self.prefs.archived);
        let row = |c: &ChatEntry, cx: &mut Context<Self>| {
            let id = c.id.clone();
            div()
                .id(gpui::ElementId::Name(format!("chat-{}", c.id).into()))
                .flex()
                .items_center()
                .gap(u(14.))
                .px(u(16.))
                .py(u(12.))
                .rounded(u(10.))
                .cursor_pointer()
                .hover(move |s| s.bg(th2.sunk))
                .on_mouse_down(
                    MouseButton::Right,
                    self.chat_menu_on_right(&c.id, "chats", cx),
                )
                .on_click(cx.listener(move |s, _, _, cx| s.open_chat(id.clone(), cx)))
                .child(icon("chat", 16., th2.ink3))
                .child(self.chat_name(&th2, c, "chats", 14.5, cx))
                .child(div().text_size(u(12.5)).text_color(th2.ink3).child(format!(
                    "{} · {}",
                    if c.turns == 1 {
                        "1 turn".into()
                    } else {
                        format!("{} turns", c.turns)
                    },
                    logic::ago(c.updated_at, now_ms())
                )))
                .child(self.chat_menu(&th2, &c.id, "chats", cx))
        };
        let mut list = card(th).mx(u(40.)).mt(u(32.)).p(u(8.)).gap(u(0.));
        if self.chats.is_empty() {
            list = list.p(u(24.)).child(
                div()
                    .text_color(th.ink2)
                    .child("No chats yet. Ask something on Home to start one."),
            );
        }
        for c in shown {
            list = list.child(row(c, cx));
        }
        if !archived.is_empty() {
            list = list.child(eyebrow(th, "Archived").px(u(16.)).pt(u(18.)).pb(u(4.)));
            for c in archived {
                list = list.child(row(c, cx).opacity(0.75));
            }
        }
        list
    }

    /// A chat's name in its row: a pin before a pinned one's, and the field while it is renamed.
    fn chat_name(
        &self,
        th: &Theme,
        c: &ChatEntry,
        place: &'static str,
        size: f32,
        cx: &mut Context<Self>,
    ) -> Div {
        let pinned = self.prefs.pinned.contains(&c.id);
        div()
            .flex_1()
            .min_w_0()
            .flex()
            .items_center()
            .gap(u(6.))
            .when(pinned, |d| d.child(icon("pin", 12., th.ink3)))
            .child(if self.renaming == Some((place, c.id.clone())) {
                div()
                    .id("rename")
                    .flex_1()
                    .min_w_0()
                    .px(u(8.))
                    .py(u(2.))
                    .rounded(u(6.))
                    .border_1()
                    .border_color(th.accent)
                    .bg(th.card)
                    .text_size(u(size))
                    // A click in the field stays there; one anywhere else keeps the new name.
                    .on_click(|_, _, cx| cx.stop_propagation())
                    .on_mouse_down_out(cx.listener(|s, _, _, cx| s.save_rename(cx)))
                    .child(self.rename_in.clone())
                    .into_any_element()
            } else {
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .text_size(u(size))
                    .child(c.title.clone())
                    .into_any_element()
            })
    }

    /// The "..." on a chat's row and, once pressed, its menu: pin, rename, archive, delete.
    /// `place`: the list the row is in; in "recent" the "..." shows only on the row's hover.
    fn chat_menu(&self, th: &Theme, id: &str, place: &'static str, cx: &mut Context<Self>) -> Div {
        let th2 = *th;
        let group = (place == "recent").then_some("recent-row");
        let open = self.menu == Some((place, id.to_owned()));
        let pinned = self.prefs.pinned.iter().any(|p| p == id);
        let archived = self.prefs.archived.iter().any(|p| p == id);
        let armed = self.armed.as_deref() == Some(id);
        let item = |key: &str, ic: &'static str, label: &str, danger: bool, f: Box<MenuAct>| {
            let c = if danger { th2.lose } else { th2.ink };
            div()
                .id(gpui::ElementId::Name(format!("{key}-{id}").into()))
                .flex()
                .items_center()
                .gap(u(10.))
                .px(u(10.))
                .py(u(7.))
                .rounded(u(7.))
                .text_size(u(13.5))
                .text_color(c)
                .cursor_pointer()
                .hover(move |s| s.bg(th2.sunk))
                .child(icon(ic, 15., c))
                .child(label.to_owned())
                .on_click(cx.listener(move |s, _, w, cx| {
                    cx.stop_propagation();
                    f(s, w, cx);
                    cx.notify()
                }))
        };
        let (a, b, c, d) = (id.to_owned(), id.to_owned(), id.to_owned(), id.to_owned());
        let menu = div()
            .id(gpui::ElementId::Name(format!("menu-{id}").into()))
            .occlude()
            .w(u(200.))
            .mt(u(4.))
            .p(u(5.))
            .flex()
            .flex_col()
            .rounded(u(10.))
            .bg(th.card)
            .border_1()
            .border_color(th.line)
            .shadow_lg()
            .on_mouse_down_out(cx.listener(|s, _, _, cx| {
                s.menu = None;
                s.armed = None;
                cx.notify()
            }))
            .child(item(
                "pin",
                "pin",
                if pinned { "Unpin" } else { "Pin" },
                false,
                Box::new(move |s, _, _| {
                    s.menu = None;
                    s.toggle_pin(&a)
                }),
            ))
            .child(item(
                "rename",
                "rename",
                "Rename",
                false,
                Box::new(move |s, w, cx| {
                    s.menu = None;
                    s.start_rename(place, b.clone(), w, cx)
                }),
            ))
            .child(item(
                "archive",
                "archive",
                if archived { "Unarchive" } else { "Archive" },
                false,
                Box::new(move |s, _, _| {
                    s.menu = None;
                    s.toggle_archive(&c)
                }),
            ))
            .child(item(
                "delete",
                "trash",
                if armed {
                    "Click again to delete"
                } else {
                    "Delete"
                },
                true,
                Box::new(move |s, _, cx| {
                    if s.armed.as_deref() != Some(d.as_str()) {
                        s.armed = Some(d.clone());
                        return;
                    }
                    (s.armed, s.menu) = (None, None);
                    s.delete_chat(d.clone(), cx)
                }),
            ));
        let idc = id.to_owned();
        div()
            .relative()
            .flex_none()
            // On hover only, it lies over the name's end instead of taking its room.
            .when(group.is_some(), |d| d.absolute().right(u(4.)))
            .child(
                div()
                    .id(gpui::ElementId::Name(format!("more-{id}").into()))
                    .when(group.is_some(), |d| d.bg(th.cur))
                    .size(u(26.))
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded(u(7.))
                    .cursor_pointer()
                    .when_some(group.filter(|_| !open), |d, g| {
                        d.opacity(0.).group_hover(g, |s| s.opacity(1.))
                    })
                    .when(open, |d| d.bg(th.sunk))
                    .hover(move |s| s.bg(th2.sunk))
                    .child(icon("more", 16., th.ink2))
                    // Opens on mouse down, as "+ Attach" does: the menu's mouse-down-out closes it.
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |s, _, _, cx| {
                            if !open {
                                s.armed = None;
                                s.menu = Some((place, idc.clone()));
                                cx.notify()
                            }
                        }),
                    )
                    // The row's own click opens the chat.
                    .on_click(|_, _, cx| cx.stop_propagation()),
            )
            .when(open, |d| {
                d.child(
                    div().absolute().top_full().right_0().child(
                        gpui::deferred(
                            gpui::anchored()
                                .anchor(gpui::Corner::TopRight)
                                .snap_to_window_with_margin(px(8.))
                                .child(menu),
                        )
                        .with_priority(1),
                    ),
                )
            })
    }

    /// Right-clicking a chat's row in `place` opens the row's "..." menu, as the "..." does. A row
    /// whose name is being edited keeps the text field's own right-click menu.
    fn chat_menu_on_right(
        &self,
        id: &str,
        place: &'static str,
        cx: &mut Context<Self>,
    ) -> impl Fn(&MouseDownEvent, &mut Window, &mut App) + 'static {
        let id = id.to_owned();
        cx.listener(move |s, _: &MouseDownEvent, _, cx| {
            if s.renaming
                .as_ref()
                .is_some_and(|(p, r)| *p == place && *r == id)
            {
                return;
            }
            cx.stop_propagation();
            s.armed = None;
            s.menu = Some((place, id.clone()));
            cx.notify()
        })
    }

    /// The chat's right-click menu at `at`: Copy the selection, and Copy message for the message
    /// at `item` (a question or a reply).
    fn ctx_menu(
        &self,
        th: &Theme,
        at: Point<Pixels>,
        item: Option<usize>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        use crate::context_menu::{card, row};
        let picked = self.sel.copied();
        let message = item
            .and_then(|i| self.threads.get(self.open.as_ref()?)?.items.get(i))
            .and_then(|it| match it {
                Item::User { text, .. }
                | Item::Text { text, .. }
                | Item::Reasoning { text, .. } => Some(text.clone()),
                _ => None,
            });
        // Each row copies its text; a row with none is greyed.
        let copy = |id, label, text: Option<String>| {
            row(th, id, label, text.is_some()).when_some(text, |d, t| {
                d.on_click(cx.listener(move |s, _, _, cx| {
                    cx.stop_propagation();
                    cx.write_to_clipboard(ClipboardItem::new_string(t.clone()));
                    s.ctx = None;
                    cx.notify()
                }))
            })
        };
        let rows = [
            Some(copy("chat-copy", "Copy", picked)),
            (message.is_some()).then(|| copy("chat-copy-message", "Copy message", message)),
        ];
        let card = card(th)
            .on_mouse_down_out(cx.listener(|s, _, _, cx| {
                s.ctx = None;
                cx.notify()
            }))
            .children(rows.into_iter().flatten());
        crate::context_menu::at(at, card)
    }

    /// Pins chat `id` to the top of the lists (out of the archive), or unpins it.
    fn toggle_pin(&mut self, id: &str) {
        toggle(&mut self.prefs.pinned, id);
        self.prefs.archived.retain(|a| a != id);
        self.save_prefs();
    }

    /// Archives chat `id`, kept but out of Recent, or brings it back.
    fn toggle_archive(&mut self, id: &str) {
        toggle(&mut self.prefs.archived, id);
        self.prefs.pinned.retain(|a| a != id);
        self.save_prefs();
    }

    fn start_rename(
        &mut self,
        place: &'static str,
        id: String,
        w: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let title = self
            .chats
            .iter()
            .find(|c| c.id == id)
            .map_or(String::new(), |c| c.title.clone());
        self.rename_in.update(cx, |i, cx| i.set_text(&title, cx));
        w.focus(&self.rename_in.focus_handle(cx));
        self.renaming = Some((place, id));
    }

    /// The name in `rename_in` becomes the chat's; an empty or unchanged one changes nothing.
    fn save_rename(&mut self, cx: &mut Context<Self>) {
        let Some((_, id)) = self.renaming.take() else {
            return;
        };
        let title = self.rename_in.read(cx).text.trim().to_owned();
        cx.notify();
        if title.is_empty() || self.chats.iter().any(|c| c.id == id && c.title == title) {
            return;
        }
        self.call(
            cx,
            move |c| c.rename_chat(&id, &title),
            |s, (), cx| {
                s.chats_rev += 1;
                s.reload_chats(cx);
            },
        );
    }

    fn delete_chat(&mut self, id: String, cx: &mut Context<Self>) {
        self.call(
            cx,
            move |c| c.delete_chat(&id).map(|()| id),
            |s, id, cx| {
                s.threads.remove(&id);
                s.chats.retain(|c| c.id != id);
                if s.open.as_deref() == Some(id.as_str()) {
                    s.set_open(None, cx);
                    // The chat page has nothing to show without it: an empty page, no composer.
                    if s.page == Page::Chat {
                        s.go(Page::Home, cx);
                    }
                }
                s.drafts.remove(&Some(id.clone()));
                let before = s.prefs.pinned.len() + s.prefs.archived.len();
                s.prefs.pinned.retain(|p| *p != id);
                s.prefs.archived.retain(|p| *p != id);
                if s.prefs.pinned.len() + s.prefs.archived.len() != before {
                    s.save_prefs();
                }
            },
        );
    }

    fn chat_page(&self, th: &Theme, cx: &mut Context<Self>) -> Div {
        let Some(id) = self.open.clone() else {
            return div();
        };
        let Some(t) = self.threads.get(&id) else {
            return div();
        };
        // Decided each frame, before layout, so the view keeps up with any growth (text, steps, a
        // rewrap), not only the events that bring text; a reader who scrolled back keeps their place.
        if logic::at_end(
            f32::from(t.scroll.offset().y),
            f32::from(t.scroll.max_offset().height),
        ) {
            t.scroll.scroll_to_bottom();
        }
        let bg = fade(th.accent, 0.25);
        let th2 = *th;
        let mut col = div()
            .flex()
            .flex_col()
            .gap(u(18.))
            .px(u(40.))
            .pt(u(12.))
            .pb(u(20.))
            // At most 1000 wide by shrinking from a 1000 basis in the row below, not by
            // `w_full().max_w(..)`: taffy measures such a child's height at the unclamped width,
            // so on a chat wider than 1000 the text wrapped to fewer lines than it drew, the
            // scroll range came out short and a long chat's end stayed under the composer.
            .flex_basis(u(1000.))
            .min_w_0();
        // The tree opens the chat and scrolls with it: pinned at full size, it cut off the messages
        // under it. Once it scrolls out of sight, a banner stays at the top (`chat_sun`).
        // Kept small, the banner stays pinned from the start and the chat begins below it.
        let small = self.prefs.sun_small;
        if self.prefs.sun && small {
            col = col.pt(u(BANNER_H));
        } else if self.prefs.sun {
            col = col.child(self.chat_sun(th, t, SunForm::Big));
        }
        // ponytail: where the big tree ends as a constant (padding + 92 px); measure it if that layout changes.
        let away = self.prefs.sun && (small || t.scroll.offset().y < px(-110.));
        let crest = away
            && !small
            && logic::crest_pinned(
                t.live,
                f32::from(t.scroll.offset().y),
                f32::from(t.scroll.max_offset().height),
            );
        let n = t.items.len();
        // The reply whose sources the side panel lists: only it leaves its list out of the chat.
        let panel = (self.shows_side_panel())
            .then(|| t.items.iter().rposition(|i| matches!(i, Item::Text { .. })))
            .flatten();
        for (i, it) in t.items.iter().enumerate() {
            let last = i + 1 == n;
            // A reply's helpers read as its steps; the tree above shows them working.
            let as_steps;
            let it = match it {
                Item::Crew(c) => {
                    as_steps = Item::Steps(crate::crew::steps(c));
                    &as_steps
                }
                it => it,
            };
            col = col.child(match it {
                Item::User {
                    text,
                    files,
                    images,
                } => div()
                    .flex()
                    .flex_col()
                    .items_end()
                    .gap(u(8.))
                    .when(!text.is_empty(), |d| {
                        d.child(
                            div()
                                .max_w(u(620.))
                                .px(u(18.))
                                .py(u(11.))
                                .rounded(u(14.))
                                .bg(th2.sunk)
                                .text_size(u(15.))
                                .line_height(u(22.))
                                .child(self.sel.text(
                                    (i, 0, 0),
                                    text.clone(),
                                    select::plain(text, font("Figtree"), th2.ink.into()),
                                    bg,
                                )),
                        )
                    })
                    .when(!images.is_empty(), |d| {
                        d.child(
                            div()
                                .flex()
                                .flex_wrap()
                                .justify_end()
                                .gap(u(6.))
                                .children(images.iter().map(|p| self.pic_el(&th2, p))),
                        )
                    })
                    .when(!files.is_empty(), |d| {
                        d.child(
                            div()
                                .w_full()
                                .flex()
                                .flex_wrap()
                                .justify_end()
                                .gap(u(8.))
                                .children(files.iter().map(|f| {
                                    tag(&th2, logic::base_name(f).to_owned())
                                        .max_w_full()
                                        .truncate()
                                        .font_family("Figtree")
                                        .text_size(u(13.))
                                        .px(u(12.))
                                        .py(u(5.))
                                        .border_1()
                                        .border_color(th2.line)
                                        .bg(th2.card)
                                })),
                        )
                    }),
                Item::Steps(steps) => {
                    let head = i == 0
                        || !matches!(
                            t.items[i - 1],
                            Item::Steps(_) | Item::Text { .. } | Item::Reasoning { .. }
                        )
                        || matches!(t.items[i - 1], Item::User { .. });
                    div()
                        .flex()
                        .flex_col()
                        .gap(u(8.))
                        .when(head, |d| d.child(self.who(th, t, last)))
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .gap(u(5.))
                                .ml(u(4.))
                                .pl(u(16.))
                                .border_l_2()
                                .border_color(th.line)
                                .children(
                                    (steps.iter().enumerate())
                                        .map(|(n, s)| step_row(&th2, &self.sel, (i, n), s)),
                                )
                                .when(last && t.progress.is_some(), |d| {
                                    d.child(div().text_size(u(13.5)).text_color(th2.ink3).child(
                                        self.sel.styled(
                                            (i, steps.len(), 0),
                                            t.progress.clone().unwrap_or_default(),
                                        ),
                                    ))
                                }),
                        )
                }
                Item::Text { text, cut } => {
                    let head = i == 0 || matches!(t.items[i - 1], Item::User { .. });
                    let copy = text.clone();
                    // The answer stays as written; each cite it makes is a button below it.
                    let cited = t.cites.get(&i).cloned().unwrap_or_default();
                    div()
                        .flex()
                        .flex_col()
                        .gap(u(10.))
                        .when(head, |d| d.child(self.who(th, t, last)))
                        .child(if panel == Some(i) {
                            markdown(th, &self.sel, i, &logic::without_sources(text))
                        } else {
                            markdown(th, &self.sel, i, text)
                        })
                        .when(!cited.is_empty(), |d| {
                            d.child(div().flex().flex_wrap().gap(u(8.)).children(
                                cited.into_iter().enumerate().map(|(n, c)| {
                                    let (th2, missing) = (*th, c.missing);
                                    pill(
                                        th,
                                        format!("cite-{i}-{n}"),
                                        c.label.clone(),
                                        false,
                                        cx.listener(move |s, _, _, cx| {
                                            s.docview = Some(DocView {
                                                title: c.title.clone(),
                                                page: c.page.clone(),
                                                hit: c.hit.clone(),
                                            });
                                            cx.notify();
                                        }),
                                    )
                                    .when(missing, |d| {
                                        d.text_color(th2.lose).border_color(th2.lose)
                                    })
                                }),
                            ))
                        })
                        .child(
                            div()
                                .flex()
                                .gap(u(12.))
                                .text_size(u(12.5))
                                .text_color(th.ink3)
                                .when(*cut, |d| d.child("Stopped"))
                                .when_some(
                                    t.docs.first_word.clone().filter(|_| last && !t.live),
                                    |d, w| d.child(w),
                                )
                                .when(!(last && t.live), |d| {
                                    d.child(
                                        div()
                                            .id(gpui::ElementId::Name(format!("copy-{i}").into()))
                                            .flex()
                                            .items_center()
                                            .gap(u(4.))
                                            .cursor_pointer()
                                            .child(icon("copy", 12., th2.ink3))
                                            .child("Copy")
                                            .on_click(move |_, _, cx| {
                                                cx.write_to_clipboard(ClipboardItem::new_string(
                                                    copy.clone(),
                                                ))
                                            }),
                                    )
                                }),
                        )
                }
                // The app speaking, not the agent: a gold rule and tint, no bubble.
                Item::Notice(n) => div()
                    .text_size(u(13.))
                    .text_color(th.ink2)
                    .px(u(12.))
                    .py(u(7.))
                    .border_l_2()
                    .border_color(th.gold)
                    .bg(fade(th.gold, 0.10))
                    .child(self.sel.styled((i, 0, 0), n.clone())),
                Item::Error(e) => div()
                    .text_size(u(13.5))
                    .text_color(th.lose)
                    .px(u(14.))
                    .py(u(8.))
                    .rounded(u(9.))
                    .border_1()
                    .border_color(fade(th.lose, 0.4))
                    .child(self.sel.styled((i, 0, 0), e.clone())),
                Item::Saved {
                    receipt,
                    line,
                    again,
                } => {
                    let r = receipt.clone();
                    div()
                        .flex()
                        .items_center()
                        .gap(u(10.))
                        .text_size(u(13.))
                        .text_color(th.ink2)
                        .child(icon("memory", 14., th.ink3))
                        // min_w 0: a long fact ran past the window instead of wrapping.
                        .child(
                            self.sel
                                .styled(
                                    (i, 0, 0),
                                    format!(
                                        "{} “{}”",
                                        if receipt.old.is_some() {
                                            "Updated in memory:"
                                        } else {
                                            "Saved to memory:"
                                        },
                                        logic::fact_sentence(&receipt.text)
                                    ),
                                )
                                .flex_1()
                                .min_w_0(),
                        )
                        .when_some(line.clone(), |d, l| {
                            d.child(self.sel.styled((i, 0, 1), l).min_w_0().text_color(th2.ink3))
                        })
                        .when(line.is_none() || *again, |d| {
                            d.child(
                                div()
                                    .id(gpui::ElementId::Name(format!("undo-{i}").into()))
                                    .cursor_pointer()
                                    .text_color(th2.accent)
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .child("Undo")
                                    .on_click(
                                        cx.listener(move |s, _, _, cx| s.call_undo(r.clone(), cx)),
                                    ),
                            )
                        })
                }
                Item::Packet(p) => self.packet_el(th, &id, i, p, cx),
                // Apart from the answer, in muted type: open while the model thinks, so the wait
                // shows its work, then one line until a click opens it.
                Item::Reasoning { text, open } => {
                    let head = i == 0 || matches!(t.items[i - 1], Item::User { .. });
                    let thinking = last && t.live;
                    let (open, tid) = (*open || thinking, id.clone());
                    div()
                        .flex()
                        .flex_col()
                        .gap(u(8.))
                        .when(head, |d| d.child(self.who(th, t, last)))
                        .child(
                            div()
                                .id(gpui::ElementId::Name(format!("reason-{i}").into()))
                                .flex()
                                .items_center()
                                .gap(u(6.))
                                .text_size(u(13.))
                                .text_color(th.ink3)
                                .cursor_pointer()
                                .child(icon(if open { "up" } else { "down" }, 12., th.ink3))
                                .child(if thinking { "Thinking…" } else { "Reasoning" })
                                .on_click(cx.listener(move |s, _, _, cx| {
                                    if let Some(Item::Reasoning { open, .. }) =
                                        (s.threads.get_mut(&tid)).and_then(|t| t.items.get_mut(i))
                                    {
                                        *open = !*open;
                                    }
                                    cx.notify();
                                })),
                        )
                        .when(open, |d| {
                            d.child(
                                div()
                                    .ml(u(4.))
                                    .pl(u(16.))
                                    .border_l_2()
                                    .border_color(th.line)
                                    .text_size(u(13.))
                                    .line_height(u(20.))
                                    .text_color(th.ink3)
                                    .child(self.sel.styled((i, 0, 0), text.trim().to_owned())),
                            )
                        })
                }
                Item::Crew(_) => unreachable!("read as steps above"),
            });
        }
        let ask = t
            .ask
            .clone()
            .filter(|a| !matches!(a, Ask::Packet))
            .map(|a| self.ask_card(th, &id, t, &a, cx));
        div()
            .flex()
            .size_full()
            .min_h_0()
            .child(
                div()
                    .relative()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .min_h_0()
                    .child(selects(
                        div()
                            .id("thread")
                            .flex_1()
                            .min_h_0()
                            .overflow_y_scroll()
                            .track_scroll(&t.scroll)
                            // A scroll repaints the list, not this view: `away` needs a render.
                            .on_scroll_wheel(cx.listener(|_, _, _, cx| cx.notify()))
                            // The selection stays: the menu copies it, or the message under it.
                            .on_mouse_down(
                                MouseButton::Right,
                                cx.listener(|s, e: &MouseDownEvent, _, cx| {
                                    s.ctx = Some((e.position, s.sel.item_at(e.position)));
                                    cx.notify();
                                }),
                            )
                            .flex()
                            .justify_center()
                            .items_start()
                            .child(col),
                        cx,
                    ))
                    // Once the tree scrolls away, a banner holds its place.
                    .when(away && !crest, |d| {
                        d.child(
                            div()
                                .absolute()
                                .top_0()
                                .left_0()
                                .right_0()
                                .child(self.chat_sun(th, t, SunForm::Banner))
                                .child(soft_edge(th, 28.))
                                .child(self.sun_size_toggle(th, BANNER_H, cx)),
                        )
                    })
                    // Solid behind the crest however tall its helpers make it, then a short fade
                    // into the text.
                    .when(crest, |d| {
                        d.child(
                            div()
                                .absolute()
                                .top_0()
                                .left_0()
                                .right_0()
                                .child(
                                    div()
                                        .flex()
                                        .justify_center()
                                        .bg(th.app)
                                        .child(self.chat_sun(th, t, SunForm::Big)),
                                )
                                .child(soft_edge(th, 64.))
                                .child(self.sun_size_toggle(th, 44., cx)),
                        )
                    })
                    // Over the tree's banner or crest, level with its chevron.
                    .children(self.side_panel_toggle(
                        th,
                        if away && !crest { BANNER_H } else { 44. },
                        cx,
                    ))
                    // Capped at 1000 like the thread's column, and for the same reason: an ask
                    // or a tall composer under `max_w` was measured at the unclamped width.
                    .child(
                        div().flex().justify_center().px(u(40.)).pb(u(20.)).child(
                            div()
                                .flex_basis(u(1000.))
                                .min_w_0()
                                .flex()
                                .flex_col()
                                .gap(u(14.))
                                .children(ask.map(|a| a.w_full()))
                                .child(div().w_full().child(self.composer_el(th, cx))),
                        ),
                    ),
            )
            // Beside a narrow window's conversation the panel left it a third of the page.
            .when(self.shows_side_panel(), |d| {
                d.child(self.side_panel(th, t, cx))
            })
    }

    /// Whether a chat shows the Advanced side panel: Advanced is on, the panel is not hidden
    /// (`prefs.side_panel`) and the window has room for it.
    fn shows_side_panel(&self) -> bool {
        self.prefs.advanced && self.prefs.side_panel && self.room >= SIDE_PANEL_ROOM
    }

    /// Shows or hides the Advanced side panel, as its button, Settings and M-Alt-B do.
    fn toggle_side_panel(&mut self, cx: &mut Context<Self>) {
        if !self.prefs.advanced {
            return;
        }
        self.prefs.side_panel = !self.prefs.side_panel;
        self.save_prefs();
        cx.notify()
    }

    /// The chat's corner button that shows or hides the side panel, left of the tree's chevron
    /// and at its height (`top`); only with Advanced on and room for the panel.
    fn side_panel_toggle(
        &self,
        th: &Theme,
        top: f32,
        cx: &mut Context<Self>,
    ) -> Option<Stateful<Div>> {
        if !self.prefs.advanced || self.room < SIDE_PANEL_ROOM {
            return None;
        }
        let th2 = *th;
        let on = self.prefs.side_panel;
        Some(
            div()
                .id("side-panel-toggle")
                .absolute()
                .top(u(top / 2. - 13.))
                .right(u(48.))
                .size(u(26.))
                .flex()
                .items_center()
                .justify_center()
                .rounded(u(13.))
                .cursor_pointer()
                .when(on, |d| d.bg(th.sunk))
                .hover(move |s| s.bg(th2.sunk))
                .child(icon("panel", 15., if on { th.ink } else { th.ink2 }))
                .on_click(cx.listener(|s, _, _, cx| {
                    cx.stop_propagation();
                    s.toggle_side_panel(cx)
                })),
        )
    }

    /// The pinned tree's corner chevron: up keeps it a small banner, down shows it full size at the
    /// chat's top and while a run streams. It sits `top` px down, centred on the banner's line or
    /// the crest's first row.
    fn sun_size_toggle(&self, th: &Theme, top: f32, cx: &mut Context<Self>) -> Stateful<Div> {
        let small = self.prefs.sun_small;
        let th2 = *th;
        div()
            .id("sun-size")
            .absolute()
            .top(u(top / 2. - 13.))
            .right(u(16.))
            .size(u(26.))
            .flex()
            .items_center()
            .justify_center()
            .rounded(u(13.))
            .cursor_pointer()
            .hover(move |s| s.bg(th2.sunk))
            .child(icon(if small { "down" } else { "shrink" }, 14., th.ink2))
            .on_click(cx.listener(|s, _, _, cx| {
                cx.stop_propagation();
                s.prefs.sun_small = !s.prefs.sun_small;
                s.save_prefs();
                cx.notify()
            }))
    }

    /// The assistant, large and centred above the chat, so its state reads at a glance: the tree,
    /// what it is doing and, while a tool runs, which one, in the `form` the scroll calls for.
    fn chat_sun(&self, th: &Theme, t: &Thread, form: SunForm) -> Div {
        let small = form != SunForm::Big;
        let m = t.crest(self.idle_mood());
        // The shared tree already animates this state; another state gets its own.
        let s = if self.sun.mood == m {
            self.sun
        } else {
            Tree::new(m)
        };
        let mut status = match m {
            Mood::Need => "Needs your answer".to_owned(),
            Mood::Write | Mood::Load | Mood::Rest => self.status_line(m),
            m => m.label().to_owned(),
        };
        let mut doing = t
            .items
            .last()
            .and_then(|i| match i {
                Item::Steps(s) => s.last().filter(|s| s.st == StepState::Run),
                _ => None,
            })
            .map(|s| s.now.clone())
            .filter(|n| t.live && !n.is_empty());
        let still = self.still() || !(t.live || t.ask.is_some());
        // While helpers run, the tree spawns and leads them, and says what it waits on.
        let crew = t.crew();
        if let Some(c) = crew {
            (status, doing) = crate::crew::status(c, still);
        }
        let size = match form {
            SunForm::Big => SUN_BIG,
            SunForm::Banner => 30.,
        };
        // The banner keeps one line: the helpers show in the crest, its status names them.
        let mark = match crew.filter(|_| form != SunForm::Banner) {
            Some(c) => crate::crew::crest(th, s, c, size, still),
            None => tree(s, size, 225., still, *th, th.app).into_any_element(),
        };
        let status = div()
            .flex_none()
            .text_size(u(if small { 14. } else { 15. }))
            .font_weight(FontWeight::SEMIBOLD)
            .text_color(if m == Mood::Need { th.eyebrow } else { th.ink })
            .child(status);
        let doing = doing.map(|d| {
            div()
                .min_w_0()
                .max_w(u(520.))
                .truncate()
                .text_size(u(if small { 12.5 } else { 13. }))
                .text_color(th.ink3)
                .child(d)
        });
        if form == SunForm::Banner {
            // A banner across the chat's top: one line, edge to edge, solid so text never shows
            // through it; its soft edge below fades into the text.
            return div()
                .flex()
                .items_center()
                .justify_center()
                .gap(u(10.))
                .w_full()
                .h(u(BANNER_H))
                .px(u(20.))
                .bg(th.app)
                .child(mark)
                .child(status)
                .children(doing);
        }
        div()
            .flex_none()
            .flex()
            .flex_col()
            .items_center()
            .gap(u(6.))
            .pt(u(8.))
            .pb(u(6.))
            .child(mark)
            .child(status)
            .children(doing)
    }

    /// A sent image in the thread, at most 220 by 160; a saved chat's shows once it is read.
    fn pic_el(&self, th: &Theme, p: &Pic) -> AnyElement {
        let image = match p {
            Pic::Live(i) => Some(Some(i.clone())),
            Pic::Saved(r) => self.pics.get(r).cloned(),
        };
        match image {
            Some(Some(i)) => img(i)
                .max_w(u(220.))
                .max_h(u(160.))
                .rounded(u(8.))
                .object_fit(ObjectFit::Contain)
                .into_any_element(),
            Some(None) => div()
                .text_size(u(12.))
                .text_color(th.ink3)
                .child("image unavailable")
                .into_any_element(),
            None => div()
                .w(u(220.))
                .h(u(120.))
                .rounded(u(8.))
                .bg(th.sunk)
                .into_any_element(),
        }
    }

    fn who(&self, th: &Theme, t: &Thread, last: bool) -> Div {
        let live = last && t.live;
        let what = match t.task {
            Some(Task::Research) => " · research",
            Some(Task::Web) => " · web",
            Some(Task::Docs) => " · documents",
            None => "",
        };
        div()
            .flex()
            .items_center()
            .gap(u(8.))
            .text_size(u(13.5))
            .text_color(th.ink3)
            .when(self.prefs.sun, |d| {
                d.child(tree(
                    if live {
                        self.sun
                    } else {
                        Tree::new(Mood::Ready)
                    },
                    22.,
                    180.,
                    // The tree above the chat moves; this one marks who wrote.
                    true,
                    *th,
                    th.app,
                ))
            })
            .child(format!("Assistant{what}"))
            .when(live, |d| {
                d.child(
                    div()
                        .text_color(th.ink2)
                        .child(format!("· {}", t.mood().label())),
                )
            })
    }

    fn ask_card(&self, th: &Theme, id: &str, t: &Thread, a: &Ask, cx: &mut Context<Self>) -> Div {
        let (q, _) = t.ask_line().unwrap_or_default();
        let (name, args, path, folder) = match a {
            Ask::Confirm { name, args } => (name.clone(), args.clone(), None, None),
            Ask::Code {
                name,
                args,
                path,
                folder,
            } => (name.clone(), args.clone(), path.clone(), folder.clone()),
            Ask::Packet => unreachable!("packets show their own card"),
        };
        // An MCP tool: every argument it is sent, whole, before the path or command a built-in
        // shows, so Allow never sends what the card left out.
        let lines = crate::mcp::call_lines(&name, &args);
        let outside = !lines.is_empty();
        let target = path
            .or_else(|| {
                args.get("path")
                    .or(args.get("command"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .filter(|_| !outside)
            .unwrap_or_default();
        let body = args
            .get("text")
            .or(args.get("content"))
            .or(args.get("new_text"))
            .and_then(Value::as_str)
            .map(|t| t.lines().count())
            .filter(|_| !outside);
        let code = matches!(a, Ask::Code { .. });
        let (yes, no, always) = (id.to_owned(), id.to_owned(), id.to_owned());
        let th2 = *th;
        let ask = |t: String| self.sel.label(select::ASK, t);
        let el = card(th)
            .p(u(24.))
            .shadow(shadow(th))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(u(10.))
                    .when(self.prefs.sun, |d| {
                        d.child(tree(self.sun, 30., 180., self.still(), *th, th.card))
                    })
                    .child(eyebrow(th, "Needs you")),
            )
            .child(serif_box(27.).child(ask(q)))
            .child(
                div()
                    .rounded(u(10.))
                    .bg(th.code_bg)
                    .text_color(th.code_ink)
                    .px(u(16.))
                    .py(u(12.))
                    .font_family("JetBrains Mono")
                    .text_size(u(12.5))
                    .line_height(u(19.))
                    .child(
                        div()
                            .flex()
                            .gap(u(16.))
                            .child(ask(name.clone()))
                            .child(ask(target.clone()).min_w_0()),
                    )
                    .when_some(body, |d, n| {
                        d.child(
                            div()
                                .pl(u(16. + 8. * name.len() as f32))
                                .child(ask(format!("+ {n} lines"))),
                        )
                    })
                    .when(outside, |d| {
                        d.child(
                            div()
                                .id(gpui::ElementId::Name(format!("ask-args-{id}").into()))
                                .max_h(u(240.))
                                .overflow_y_scroll()
                                .mt(u(6.))
                                .children(lines.into_iter().map(ask)),
                        )
                    }),
            )
            .child(
                div()
                    .flex()
                    .gap(u(8.))
                    .child(button(
                        th,
                        "ask-yes",
                        if name == "write_file" {
                            "Save"
                        } else {
                            "Allow"
                        },
                        Btn::Primary,
                        cx.listener(move |s, _, _, cx| s.answer(yes.clone(), true, None, cx)),
                    ))
                    .when(code, |d| {
                        d.child(button(
                            &th2,
                            "ask-always",
                            folder.as_ref().map_or("Always for this file".into(), |f| {
                                format!("Always in {f}")
                            }),
                            Btn::Outline,
                            cx.listener(move |s, _, _, cx| {
                                let dec = if matches!(
                                    s.threads.get(&always).and_then(|t| t.ask.as_ref()),
                                    Some(Ask::Code {
                                        folder: Some(_),
                                        ..
                                    })
                                ) {
                                    Decision::Folder
                                } else {
                                    Decision::File
                                };
                                s.answer(always.clone(), true, Some(dec), cx)
                            }),
                        ))
                    })
                    .child(button(
                        th,
                        "ask-no",
                        "No",
                        Btn::Ghost,
                        cx.listener(move |s, _, _, cx| s.answer(no.clone(), false, None, cx)),
                    )),
            );
        selects(el, cx)
    }

    /// The thread's item `i`: a code task's change, its words selectable.
    fn packet_el(&self, th: &Theme, id: &str, i: usize, p: &Packet, cx: &mut Context<Self>) -> Div {
        let sel = &self.sel;
        let th2 = *th;
        let tests = |t: &Option<Tests>| {
            t.as_ref().map_or("no tests".into(), |t| {
                if t.failed == 0 {
                    format!("{} passed", t.passed)
                } else {
                    format!("{} failed, {} passed", t.failed, t.passed)
                }
            })
        };
        let (yes, no) = (id.to_owned(), id.to_owned());
        let in_place = p.checkpoint.is_some();
        card(th)
            .child(eyebrow(th, "Change ready"))
            .child(serif_box(25.).child(sel.styled((i, 0, 0), p.title.clone())))
            .when(!p.body.is_empty(), |d| {
                d.child(
                    div()
                        .text_size(u(14.))
                        .text_color(th2.ink2)
                        .child(sel.styled((i, 1, 0), p.body.clone())),
                )
            })
            .child(
                div()
                    .flex()
                    .gap(u(18.))
                    .text_size(u(13.))
                    .text_color(th.ink2)
                    .flex_wrap()
                    .child(sel.styled((i, 2, 0), format!("Tests before: {}", tests(&p.before))))
                    .child(sel.styled((i, 2, 1), format!("after: {}", tests(&p.after))))
                    .child(sel.styled(
                        (i, 2, 2),
                        format!(
                            "{} file{}{}",
                            p.scope.len(),
                            if p.scope.len() == 1 { "" } else { "s" },
                            if p.outside > 0 {
                                format!(", {} outside what you allowed", p.outside)
                            } else {
                                String::new()
                            }
                        ),
                    )),
            )
            .when_some(
                p.stop
                    .clone()
                    .filter(|s| !s.starts_with("the model said it is done")),
                |d, s| {
                    d.child(
                        div()
                            .text_size(u(13.))
                            .text_color(th2.ink3)
                            .child(sel.styled((i, 3, 0), logic::stop_line(&s))),
                    )
                },
            )
            .child(
                div()
                    .id(gpui::ElementId::Name(format!("diff-{id}").into()))
                    .max_h(u(320.))
                    .overflow_y_scroll()
                    .rounded(u(10.))
                    .bg(th.code_bg)
                    .px(u(14.))
                    .py(u(10.))
                    .font_family("JetBrains Mono")
                    .text_size(u(12.))
                    .line_height(u(18.))
                    .children(
                        (logic::diff_lines(&p.diff, 400).into_iter().enumerate()).map(
                            move |(n, (k, l))| {
                                use logic::DiffKind::*;
                                let c: Hsla = match k {
                                    File => th2.sun.into(),
                                    Hunk => fade(th2.code_ink, 0.5),
                                    Add => gpui::rgb(0x8fd19e).into(),
                                    Del => gpui::rgb(0xf29ab8).into(),
                                    Same => fade(th2.code_ink, 0.85),
                                };
                                div()
                                    .text_color(c)
                                    .when(k == File, |d| {
                                        d.pt(u(6.)).font_weight(FontWeight::SEMIBOLD)
                                    })
                                    .child(sel.styled(
                                        (i, 4, n),
                                        if l.is_empty() { " ".into() } else { l },
                                    ))
                            },
                        ),
                    ),
            )
            .child(match p.decided {
                Some(true) => div()
                    .text_size(u(13.5))
                    .text_color(th.keep)
                    .child(if in_place { "Kept." } else { "Merged." }),
                Some(false) => div()
                    .text_size(u(13.5))
                    .text_color(th.ink3)
                    .child(if in_place { "Undone." } else { "Discarded." }),
                None => div()
                    .flex()
                    .gap(u(8.))
                    .child(button(
                        th,
                        format!("keep-{id}"),
                        if in_place { "Keep it" } else { "Merge it" },
                        Btn::Primary,
                        cx.listener(move |s, _, _, cx| s.answer(yes.clone(), true, None, cx)),
                    ))
                    .child(button(
                        th,
                        format!("drop-{id}"),
                        if in_place { "Undo it" } else { "Discard" },
                        Btn::Ghost,
                        cx.listener(move |s, _, _, cx| s.answer(no.clone(), false, None, cx)),
                    )),
            })
    }

    /// Scrolls the open chat so heading block `b` of item `i` sits just under the banner.
    fn jump(&mut self, (i, b): (usize, usize), w: &Window) {
        let Some(t) = self.open.as_ref().and_then(|id| self.threads.get(id)) else {
            return;
        };
        let Some(top) = self.sel.top((i, b, 0)) else {
            return;
        };
        let gap = u(if self.prefs.sun { BANNER_H + 16. } else { 12. }).to_pixels(w.rem_size());
        let o = t.scroll.offset();
        let mut y = o.y - (top - t.scroll.bounds().top()) + gap;
        // A jump is reading back: one that lands at a live chat's end would follow the run again
        // and pin the full tree over the heading, so it stops a pixel short and keeps the banner.
        let max = t.scroll.max_offset().height;
        if self.prefs.sun
            && !self.prefs.sun_small
            && logic::crest_pinned(t.live, f32::from(y), f32::from(max))
        {
            y = px(2.) - max;
        }
        t.scroll.set_offset(point(o.x, y));
    }

    fn side_panel(&self, th: &Theme, t: &Thread, cx: &mut Context<Self>) -> Stateful<Div> {
        let last = t.items.iter().rposition(|i| matches!(i, Item::Text { .. }));
        let text = match last.map(|i| &t.items[i]) {
            Some(Item::Text { text, .. }) => text.as_str(),
            _ => "",
        };
        let a = logic::parse_answer(text);
        // Beside this panel the chat shows the reply without its sources list, so its blocks are
        // counted the same way.
        let heads = logic::heading_blocks(&logic::without_sources(text), &a.sections);
        let stats = t.stats.or_else(|| {
            t.live.then(|| Stats {
                tok_s: self.last.as_ref().map_or(0., |s| s.tok_s),
                ..Default::default()
            })
        });
        let mem: Vec<&Fact> = self.facts.iter().take(3).collect();
        let th2 = *th;
        // Sources and sections stay links: a drag across one would open it on release.
        let sel = &self.sel;
        let side = |t: String| sel.label(select::SIDE, t);
        let head = |t: &str| eyebrow_box(&th2).child(side(t.to_uppercase()));
        let panel = div()
            .id("side")
            .w(u(300.))
            .flex_none()
            .h_full()
            .overflow_y_scroll()
            .bg(th.side)
            .border_l_1()
            .border_color(th.line)
            .p(u(20.))
            .flex()
            .flex_col()
            .gap(u(12.))
            .when_some(stats, |d, s| {
                let stat = |v: String, l: &str| {
                    div()
                        .flex_1()
                        .flex()
                        .flex_col()
                        .pt(u(8.))
                        .border_t_1()
                        .border_color(th2.line)
                        .child(serif_box(30.).child(side(v)))
                        .child(
                            div()
                                .text_size(u(12.))
                                .text_color(th2.ink3)
                                .child(side(l.to_owned())),
                        )
                };
                d.child(head("This reply")).child(
                    div()
                        .flex()
                        .gap(u(12.))
                        .child(stat(format!("{:.0}", s.tok_s), "tok/s"))
                        .when(s.ttft_ms > 0., |d| {
                            d.child(stat(format!("{:.0}", s.ttft_ms), "ms first token"))
                        })
                        .when(s.ttft_ms > 0., |d| {
                            d.child(stat(format!("{:.0} %", s.reused * 100.), "reused"))
                        }),
                )
            })
            .when(!a.sources.is_empty() || !t.files.is_empty(), |d| {
                d.child(head("Sources").mt(u(10.)))
                    .children(a.sources.iter().map(|s| {
                        let url = s.url.clone();
                        div()
                            .id(gpui::ElementId::Name(format!("src-{}", s.n).into()))
                            .flex()
                            .gap(u(8.))
                            .py(u(7.))
                            .border_b_1()
                            .border_color(th2.line)
                            .text_size(u(13.))
                            .cursor_pointer()
                            .on_click(move |_, _, cx| cx.open_url(&url))
                            .child(
                                div()
                                    .text_color(th2.accent)
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .child(s.n.to_string()),
                            )
                            .child(
                                div()
                                    .flex_1()
                                    .min_w_0()
                                    .flex()
                                    .flex_col()
                                    .child(s.title.clone())
                                    .when(!s.quote.is_empty(), |d| {
                                        d.child(
                                            lines(2., 19.).text_color(th2.ink3).child(format!(
                                                "“{}”",
                                                logic::plain_links(&s.quote)
                                            )),
                                        )
                                    }),
                            )
                    }))
                    .children(t.files.iter().map(|f| {
                        div()
                            .py(u(7.))
                            .border_b_1()
                            .border_color(th2.line)
                            .text_size(u(13.))
                            .child(side(logic::base_name(f).to_owned()))
                    }))
            })
            .when(!a.papers.is_empty(), |d| {
                d.child(head("Papers read").mt(u(10.)))
                    .children(a.papers.iter().map(|p| {
                        div()
                            .flex()
                            .gap(u(8.))
                            .text_size(u(13.))
                            .child(
                                div()
                                    .text_color(th2.accent)
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .child(side(p.n.to_string())),
                            )
                            .child(
                                div()
                                    .min_w_0()
                                    .truncate()
                                    .child(side(logic::base_name(&p.file).to_owned())),
                            )
                            .when(p.cut, |d| {
                                d.child(side("· key parts".into()).text_color(th2.ink3))
                            })
                            .when(
                                a.sections
                                    .get(p.section)
                                    .is_some_and(|s| s.text.contains("Not read:")),
                                |d| d.child(side("· parts not read".into()).text_color(th2.lose)),
                            )
                    }))
            })
            .when(a.sections.len() > 1, |d| {
                d.child(head("In this answer").mt(u(10.))).children(
                    a.sections.iter().zip(heads).enumerate().map(|(n, (s, b))| {
                        div()
                            .id(gpui::ElementId::Name(format!("sec-{n}").into()))
                            .pl(u(10. * s.level.saturating_sub(1) as f32))
                            .text_size(u(13.))
                            .text_color(th2.ink2)
                            .when_some(last.zip(b), |d, key| {
                                d.cursor_pointer()
                                    .hover(move |s| s.text_color(th2.ink))
                                    .on_click(cx.listener(move |s, _, w, cx| {
                                        s.jump(key, w);
                                        cx.notify()
                                    }))
                            })
                            .child(s.title.clone())
                    }),
                )
            })
            .when(!a.warnings.is_empty(), |d| {
                d.child(head("Could not verify").mt(u(10.)))
                    .children(a.warnings.iter().map(|w| {
                        div()
                            .text_size(u(12.5))
                            .text_color(th2.ink2)
                            .child(side(w.clone()))
                    }))
            })
            .when(!mem.is_empty(), |d| {
                d.child(head("Memory").mt(u(10.)))
                    .children(mem.into_iter().map(|f| {
                        div()
                            .text_size(u(13.))
                            .text_color(th2.ink3)
                            .child(side(format!(
                                "“{}”",
                                logic::clip(logic::fact_sentence(&f.text), 60)
                            )))
                    }))
            });
        selects(panel, cx)
    }
}

fn legend(c: Rgba, t: String) -> Div {
    div()
        .flex()
        .items_center()
        .gap(u(5.))
        .child(div().size(u(8.)).rounded(u(2.)).bg(c))
        .child(t)
}

/// Step `n` of the thread's item `i`, its words selectable.
fn step_row(th: &Theme, sel: &Selection, (i, n): (usize, usize), s: &Step) -> Div {
    let c: Hsla = match s.st {
        StepState::Run => th.sun.into(),
        StepState::Wait => th.accent.into(),
        StepState::Ok => th.keep.into(),
        StepState::Fail => th.lose.into(),
        StepState::No | StepState::Stop => th.ink3.into(),
    };
    // Drawn, not ●, ✓ and ✕: Figtree has none of them, and fallback fonts set them off centre.
    let mark = match s.st {
        StepState::Run | StepState::Wait => {
            div().size(u(7.)).rounded_full().bg(c).into_any_element()
        }
        StepState::Ok => icon("check", 12., c).into_any_element(),
        StepState::Fail => icon("x", 12., c).into_any_element(),
        StepState::No | StepState::Stop => div().child("–").into_any_element(),
    };
    let text = if matches!(s.st, StepState::Run | StepState::Wait | StepState::Fail) {
        &s.now
    } else {
        &s.done
    };
    div()
        .flex()
        .items_center()
        .gap(u(8.))
        .text_size(u(14.))
        .text_color(th.ink2)
        .child(
            div()
                .flex_none()
                .w(u(12.))
                .flex()
                .justify_center()
                .text_color(c)
                .text_size(u(12.))
                .child(mark),
        )
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_wrap()
                .gap_x(u(8.))
                .child(sel.styled((i, n, 0), text.clone()).min_w_0())
                .when(s.count > 1, |d| {
                    d.child(
                        sel.styled((i, n, 1), format!("×{}", s.count))
                            .text_color(th.ink3),
                    )
                })
                // min_w 0: a failure's JSON has no space to wrap at, and ran past the window.
                .when(!s.sum.is_empty(), |d| {
                    d.child(
                        (sel.styled((i, n, 2), format!("· {}", s.sum)))
                            .min_w_0()
                            .text_color(th.ink3),
                    )
                })
                .when(s.st == StepState::Wait, |d| {
                    d.child(
                        sel.styled((i, n, 3), "· waiting for your OK")
                            .text_color(th.accent),
                    )
                }),
        )
}

/// The local hour, from the OS clock and zone.
fn chrono_hour() -> u32 {
    // The zone offset is the OS's; without a time crate, `date` reads it.
    static HOUR: std::sync::OnceLock<(Instant, u32)> = std::sync::OnceLock::new();
    let (at, h) = *HOUR.get_or_init(|| {
        let out = if cfg!(windows) {
            no_window(&mut std::process::Command::new("powershell"))
                .args(["-NoProfile", "-Command", "(Get-Date).Hour"])
                .output()
        } else {
            std::process::Command::new("date").arg("+%H").output()
        };
        let h = out
            .ok()
            .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
            .unwrap_or(((now_ms() / 3_600_000) % 24) as u32);
        (Instant::now(), h)
    });
    (h + (at.elapsed().as_secs() / 3600) as u32) % 24
}

impl Focusable for Ecl {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus.clone()
    }
}

impl Render for Ecl {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let th = self.theme();
        if let Some(i) = self.focus_next.take() {
            window.focus(&i.focus_handle(cx));
        }
        // Every place registers its text again as it renders.
        self.sel.restart(fade(th.accent, 0.25));
        // Every field follows the theme, not only the composer.
        for i in [
            &self.composer,
            &self.task_goal,
            &self.task_done,
            &self.task_time,
            &self.search,
            &self.fact_in,
            &self.edit_in,
            &self.rename_in,
            &self.name_in,
            &self.port_in,
            &self.mcp_in,
            &self.knob_in,
            &self.mcpq_in,
            &self.mcp_name_in,
            &self.mcp_target_in,
            &self.mcp_paste_in,
        ] {
            i.update(cx, |i, _| i.theme = th);
        }
        for (_, i) in &self.mcp.answers {
            i.update(cx, |i, _| i.theme = th);
        }
        for i in &self.task_steps {
            i.update(cx, |i, _| i.theme = th);
        }
        self.half = px(if cfg!(any(target_os = "windows", target_os = "linux")) {
            0.5 / window.scale_factor()
        } else {
            0.
        });
        window.set_rem_size(px(
            self.prefs.text_size as f32 * if self.prefs.compact { 0.94 } else { 1. }
        ));
        // `u` is a 14th of the rem; the sidebar is `SIDEBAR` of them.
        self.room =
            f32::from(window.viewport_size().width) * 14. / f32::from(window.rem_size()) - SIDEBAR;
        if self.shown != self.page {
            self.shown = self.page;
            // A page without the composer takes the focus back, so the shortcuts keep working.
            match self.page {
                Page::Home | Page::Chat => window.focus(&self.composer.focus_handle(cx)),
                _ => window.focus(&self.focus),
            }
        }
        // After the page's own focus: the welcome's field or the tour card takes the keys.
        if std::mem::take(&mut self.tour_refocus) {
            match self.tour {
                Some(crate::tour::At::Welcome) => window.focus(&self.name_in.focus_handle(cx)),
                Some(crate::tour::At::Stop(_)) => window.focus(&self.tour_focus),
                None => {}
            }
        }
        let m = self.mood(window, cx);
        if self
            .toast
            .as_ref()
            .is_some_and(|t| t.at.elapsed() < t.lasts())
        {
            // Toasts fade on their own: a frame a second is enough to take one down.
            cx.spawn(async move |this, cx| {
                cx.background_executor().timer(Duration::from_secs(1)).await;
                this.update(cx, |_, cx| cx.notify()).ok();
            })
            .detach();
        }
        let title: AnyElement = match self.page {
            Page::Chat => div()
                .text_size(u(15.))
                .font_weight(FontWeight::SEMIBOLD)
                .truncate()
                .child(
                    self.open
                        .as_ref()
                        .and_then(|id| self.threads.get(id))
                        .map_or("Chat".into(), |t| t.title.clone()),
                )
                .into_any_element(),
            Page::Models | Page::Engine => {
                crate::pages::models_tabs(self, &th, cx).into_any_element()
            }
            Page::Mcp | Page::Connect => crate::pages::tools_tabs(self, &th, cx).into_any_element(),
            Page::Tasks => crate::tasks::title(self, &th),
            Page::Settings => head(&th, "settings", "Settings").into_any_element(),
            p => NAV
                .iter()
                .find(|n| n.0 == p)
                .map_or(div(), |n| head(&th, n.1, n.2))
                .into_any_element(),
        };
        let mut extra: Vec<AnyElement> = Vec::new();
        // The sidebar's card says the same, so a narrow page drops these before they cut.
        if self.page == Page::Home && self.prefs.advanced && self.room >= 760. {
            if let Some(name) = self.model_name() {
                let quant = self
                    .models
                    .iter()
                    .find(|x| Some(&x.path) == self.st.model.as_ref())
                    .map(|x| x.quant.clone())
                    .unwrap_or_default();
                extra.push(
                    div()
                        .min_w_0()
                        .flex()
                        .items_center()
                        .gap(u(8.))
                        .px(u(14.))
                        .py(u(6.))
                        .rounded_full()
                        .border_1()
                        .border_color(th.line)
                        .bg(th.card)
                        .text_size(u(13.))
                        .child(div().flex_none().size(u(10.)).rounded_full().bg(th.sun))
                        .child(
                            div()
                                .min_w_0()
                                .truncate()
                                .child(format!("{name} · {quant}")),
                        )
                        .into_any_element(),
                );
            }
            if let Some(t) = self.gpu_bytes() {
                extra.push(
                    div()
                        .min_w_0()
                        .truncate()
                        .px(u(14.))
                        .py(u(6.))
                        .rounded_full()
                        .border_1()
                        .border_color(th.line)
                        .bg(th.card)
                        .text_size(u(13.))
                        .child(format!(
                            "{} · {}",
                            if self.st.device.is_empty() {
                                "GPU".into()
                            } else {
                                self.st.device.clone()
                            },
                            logic::gb(t).replace(".0 GB", " GB")
                        ))
                        .into_any_element(),
                );
            }
        }
        if self.page == Page::Chat
            && let Some(t) = self
                .open
                .as_ref()
                .and_then(|id| self.threads.get(id))
                .and_then(|t| t.task)
        {
            extra.push(
                tag(&th, format!("{t:?}"))
                    .font_family("Figtree")
                    .text_size(u(13.))
                    .px(u(14.))
                    .py(u(6.))
                    .bg(th.card)
                    .border_1()
                    .border_color(th.line)
                    .into_any_element(),
            );
        }
        if self.page == Page::Tasks {
            extra.extend(crate::tasks::chips(self, &th, cx));
        }
        let body: AnyElement = match self.page {
            Page::Home => div()
                .id("home")
                .size_full()
                .overflow_y_scroll()
                .pb(u(40.))
                .child(self.home(&th, cx))
                .into_any_element(),
            Page::Chats => div()
                .id("chats")
                .size_full()
                .overflow_y_scroll()
                .pb(u(40.))
                .child(self.chats_page(&th, cx))
                .into_any_element(),
            Page::Chat => self.chat_page(&th, cx).into_any_element(),
            Page::Tasks => crate::tasks::page(self, &th, cx).into_any_element(),
            Page::Models => crate::pages::models(self, &th, cx).into_any_element(),
            Page::Engine => crate::pages::engine(self, &th, cx).into_any_element(),
            Page::Mcp => crate::pages::mcp(self, &th, cx).into_any_element(),
            Page::Connect => crate::pages::connect(self, &th, cx).into_any_element(),
            Page::Settings => crate::pages::settings(self, &th, cx).into_any_element(),
        };
        let failed = self.failed.clone();
        div()
            .key_context("Ecl")
            .track_focus(&self.focus)
            .on_action(cx.listener(|s, _: &NewChat, w, cx| s.new_chat(w, cx)))
            .on_action(cx.listener(|s, _: &GoHome, _, cx| s.go(Page::Home, cx)))
            .on_action(cx.listener(|s, _: &GoChats, _, cx| s.go(Page::Chats, cx)))
            .on_action(cx.listener(|s, _: &GoTasks, _, cx| s.go(Page::Tasks, cx)))
            .on_action(cx.listener(|s, _: &GoModels, _, cx| s.go(Page::Models, cx)))
            .on_action(cx.listener(|s, _: &GoTools, _, cx| s.go(Page::Mcp, cx)))
            // A field's own Copy runs while it has the keys; this one, the chat's selection.
            .on_action(cx.listener(|s, _: &crate::input::Copy, _, cx| {
                if let Some(t) = s.sel.copied() {
                    cx.write_to_clipboard(ClipboardItem::new_string(t))
                }
            }))
            .on_action(cx.listener(|s, _: &GoSettings, _, cx| s.go(Page::Settings, cx)))
            .on_action(cx.listener(|s, _: &StopReply, _, cx| s.stop_reply(cx)))
            .on_action(cx.listener(|s, _: &ToggleSidePanel, _, cx| s.toggle_side_panel(cx)))
            .on_action(
                cx.listener(|s, _: &FocusComposer, w, cx| w.focus(&s.composer.focus_handle(cx))),
            )
            .on_action(cx.listener(|s, _: &Dismiss, _, cx| {
                if s.tour.is_some() {
                    crate::tour::end(s, false, cx);
                }
                (s.popover, s.attach_menu) = (false, false);
                s.docview = None;
                s.armed = None;
                s.editing = None;
                s.editing_knob = None;
                (s.menu, s.renaming, s.ctx) = (None, None, None);
                cx.notify()
            }))
            .relative()
            .size_full()
            .flex()
            .bg(th.app)
            .text_color(th.ink)
            .font_family("Figtree")
            .text_size(u(14.))
            .child(self.sidebar(&th, cx))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .flex()
                    .flex_col()
                    .child(self.topbar(&th, title, extra, m, cx))
                    // A flex column, not a block: taffy 0.9 lays a block's children out for real
                    // even when it only measures the block. A late measure here at content width
                    // outlived the real pass (served from cache), so the composer, ask cards and
                    // chat text kept that measure's width: the composer narrow and at the left.
                    .child(div().flex_1().min_h_0().flex().flex_col().child(body)),
            )
            .when(self.popover, |d| d.child(self.popover(&th, cx)))
            .children(self.docview_el(&th, cx))
            .when_some(self.ctx, |d, (at, item)| {
                d.child(self.ctx_menu(&th, at, item, cx))
            })
            .children(self.boot_el(&th, window, cx))
            .children(crate::tour::welcome_el(self, &th, cx))
            .children(self.toast_el(&th, cx))
            .when_some(failed, |d, f| d.child(self.failed_el(&th, &f, cx)))
    }
}

/// Adds `id` to `v`, or takes it out if there.
fn toggle(v: &mut Vec<String>, id: &str) {
    match v.iter().position(|x| x == id) {
        Some(i) => {
            v.remove(i);
        }
        None => v.push(id.to_owned()),
    }
}

/// What a chat menu item does.
type MenuAct = dyn Fn(&mut Ecl, &mut Window, &mut Context<Ecl>);

#[cfg(test)]
mod tests {
    use super::*;

    /// The tree leads the reply's helpers while any still runs, even once text follows them.
    #[test]
    fn the_sun_leads_helpers_until_the_last_one_ends() {
        let helper = |st| crate::crew::Helper {
            task: "a.pdf".into(),
            now: String::new(),
            st,
            frac: None,
        };
        let mut t = Thread::new("Chat");
        t.live = true;
        t.items = vec![
            Item::User {
                text: "go".into(),
                files: vec![],
                images: vec![],
            },
            Item::Crew(crate::crew::Crew::new(vec![
                helper(StepState::Ok),
                helper(StepState::No),
            ])),
            Item::Text {
                text: "So far".into(),
                cut: false,
            },
        ];
        assert!(t.crew().is_some());
        assert_eq!(t.mood(), Mood::Work);
        if let Item::Crew(c) = &mut t.items[1] {
            c.helpers[1].st = StepState::Ok;
        }
        assert!(t.crew().is_none());
        assert_eq!(t.mood(), Mood::Write);
    }

    /// An idle chat's crest says what the engine does while it loads or has no model; a chat at
    /// work says its own state.
    #[test]
    fn an_idle_chats_crest_says_the_engine_loads() {
        let mut t = Thread::new("Chat");
        assert_eq!(t.crest(None), Mood::Ready);
        assert_eq!(t.crest(Some(Mood::Load)), Mood::Load);
        assert_eq!(t.crest(Some(Mood::Rest)), Mood::Rest);
        t.live = true;
        assert_eq!(t.crest(Some(Mood::Load)), Mood::Think);
    }

    /// Home and the popover answer only what their line shows whole: a file write or a short
    /// command, not an outside call's arguments, a command cut to fit, or a change to keep.
    #[test]
    fn only_an_ask_its_line_shows_whole_is_answered_off_its_card() {
        let ask = |a: Ask| {
            let mut t = Thread::new("Chat");
            t.ask = Some(a);
            t.quick_answer()
        };
        let confirm = |name: &str, args: Value| Ask::Confirm {
            name: name.into(),
            args,
        };
        assert!(ask(confirm(
            "write_file",
            serde_json::json!({"path": "a.md"})
        )));
        assert!(ask(confirm(
            "bash",
            serde_json::json!({"command": "ls -la"})
        )));
        assert!(!ask(confirm(
            "bash",
            serde_json::json!({"command": "x".repeat(61)})
        )));
        assert!(!ask(confirm(
            "bash",
            serde_json::json!({"command": "a\nb"})
        )));
        assert!(!ask(confirm(
            "mail__send",
            serde_json::json!({"to": "someone@example.com"})
        )));
        assert!(!ask(Ask::Packet));
    }

    fn asked() -> Item {
        Item::User {
            text: "go".into(),
            files: vec![],
            images: vec![],
        }
    }

    fn crew_of(st: &[StepState]) -> Item {
        let helper = |&st| crate::crew::Helper {
            task: "a.pdf".into(),
            now: String::new(),
            st,
            frac: None,
        };
        Item::Crew(crate::crew::Crew::new(st.iter().map(helper).collect()))
    }

    fn states(it: &Item) -> Vec<StepState> {
        match it {
            Item::Crew(c) => c.helpers.iter().map(|h| h.st).collect(),
            Item::Steps(s) => s.iter().map(|s| s.st).collect(),
            _ => vec![],
        }
    }

    /// A call's result finds its step behind the helpers the call started: done, never left
    /// running for the turn's end to mark stopped.
    #[test]
    fn a_tool_result_finds_its_step_behind_the_replys_helpers() {
        let mut t = Thread::new("Chat");
        t.live = true;
        let step = Step {
            name: "read_papers".into(),
            args: serde_json::json!({"papers": ["a.pdf"]}),
            now: "Reading a.pdf".into(),
            done: "Read a.pdf".into(),
            sum: String::new(),
            st: StepState::Run,
            count: 1,
        };
        t.items = vec![asked(), Item::Steps(vec![step]), crew_of(&[StepState::Ok])];
        t.tool_result("read_papers".into(), "1 paper".into());
        assert_eq!(states(&t.items[1]), [StepState::Ok]);
    }

    /// A turn that ends (or is cancelled) stops its helpers still waiting or working; one that
    /// failed or finished keeps its state.
    #[test]
    fn a_turns_end_stops_its_open_helpers() {
        use StepState::*;
        let mut t = Thread::new("Chat");
        t.live = true;
        t.items = vec![asked(), crew_of(&[Fail, Run, No, Ok])];
        t.stop_open();
        assert_eq!(states(&t.items[1]), [Fail, Stop, Stop, Ok]);
    }

    /// An empty list keeps the reply's helpers: only a new turn's start says none.
    #[test]
    fn an_empty_helper_list_keeps_the_replys_crew() {
        let mut t = Thread::new("Chat");
        t.live = true;
        t.items = vec![asked(), crew_of(&[StepState::Run])];
        t.set_helpers(Vec::new());
        assert_eq!(states(&t.items[1]), [StepState::Run]);
    }

    /// Helpers first heard while the reply's text streams go before it: the text stays one block.
    #[test]
    fn a_first_helper_list_keeps_the_streaming_text_whole() {
        let mut t = Thread::new("Chat");
        t.live = true;
        let helpers = |st| match crew_of(&[st]) {
            Item::Crew(c) => c.helpers,
            _ => unreachable!("crew_of makes a crew"),
        };
        t.items = vec![asked()];
        t.delta("So *far".into());
        t.set_helpers(helpers(StepState::Run));
        t.delta(" so* good".into());
        t.set_helpers(helpers(StepState::Ok));
        assert!(
            matches!(&t.items[..], [Item::User { .. }, Item::Crew(_), Item::Text { text, .. }]
            if text == "So *far so* good")
        );
        assert_eq!(states(&t.items[1]), [StepState::Ok]);
    }

    /// A step's reasoning streams into one block above the answer it leads to, which stays one
    /// block too.
    #[test]
    fn reasoning_sits_above_its_answer() {
        let mut t = Thread::new("Chat");
        t.items = vec![asked()];
        t.reasoning("\n".into());
        t.reasoning("Wait, re-read".into());
        t.reasoning(" the table.".into());
        t.delta("PA26 and".into());
        t.delta(" PA27.".into());
        assert!(matches!(&t.items[..], [
            Item::User { .. },
            Item::Reasoning { text: r, open: false },
            Item::Text { text, .. },
        ] if r == "Wait, re-read the table." && text == "PA26 and PA27."));
    }

    /// A restart to update asks only while something is going, and names what it would stop.
    #[test]
    fn a_restart_to_update_names_what_it_would_stop() {
        assert_eq!(going(0, 0), None);
        assert_eq!(going(1, 0).as_deref(), Some("a reply"));
        assert_eq!(going(0, 3).as_deref(), Some("3 task runs"));
        assert_eq!(going(2, 1).as_deref(), Some("2 replies and a task run"));
    }

    /// An MCP tool's confirm names its server and tool; a built-in's keeps its own question.
    #[test]
    fn an_mcp_tool_confirm_names_its_server() {
        let mut t = Thread::new("Plan Friday");
        t.ask = Some(Ask::Confirm {
            name: "google_calendar__create_event".into(),
            args: serde_json::json!({"title": "Lunch with Sam"}),
        });
        assert_eq!(
            t.ask_line().unwrap().0,
            "Let google calendar run create event?"
        );
        t.ask = Some(Ask::Confirm {
            name: "write_file".into(),
            args: serde_json::json!({"path": "/tmp/notes.md"}),
        });
        assert_eq!(t.ask_line().unwrap().0, "Save notes.md?");
    }

    fn file(q: &str, gb: f64, parts: usize) -> HfFile {
        HfFile {
            file: format!("m-{q}.gguf"),
            parts: vec!["x".into(); parts],
            size_bytes: (gb * logic::GIB) as u64,
            quant: q.into(),
            downloaded: false,
            mmproj: None,
        }
    }

    /// A launch loads the last model again only while it is listed and runs, timed by its last load.
    #[test]
    fn a_launch_reloads_the_last_model_that_still_runs() {
        let entry = |path: &str, supported| ModelEntry {
            name: path.into(),
            arch: "llama".into(),
            path: path.into(),
            quant: "Q4_K_M".into(),
            size_bytes: 5_000_000_000,
            supported,
            fit: Fit::Unknown,
            source: String::new(),
            library: String::new(),
            vision: false,
            faster: None,
        };
        let mut p = Prefs::default();
        let models = [entry("a.gguf", true), entry("b.gguf", false)];
        assert_eq!(relaunch(&p, &models), None);
        p.last_model = Some("a.gguf".into());
        assert_eq!(
            relaunch(&p, &models),
            Some(("a.gguf".into(), "a.gguf".into(), 8000.))
        );
        p.load_ms.insert("a.gguf".into(), 4200);
        assert_eq!(relaunch(&p, &models).unwrap().2, 4200.);
        p.last_model = Some("b.gguf".into());
        assert_eq!(relaunch(&p, &models), None);
        p.last_model = Some("gone.gguf".into());
        assert_eq!(relaunch(&p, &models), None);
        // A faster version, which is not listed, loads again through its original's entry.
        let mut a = entry("a.gguf", true);
        a.faster = Some(Faster {
            copy: Some("a.faster.gguf".into()),
            building: false,
        });
        p.last_model = Some("a.faster.gguf".into());
        assert_eq!(
            listing(std::slice::from_ref(&a), "a.faster.gguf").map(|m| &m.path),
            Some(&a.path)
        );
        assert_eq!(
            relaunch(&p, &[a]),
            Some(("a.faster.gguf".into(), "a.gguf".into(), 8000.))
        );
    }

    /// The row offers the best-liked quant that fits, single files first, else the smallest; never a
    /// quant the engine cannot load while the repo has one it can.
    #[test]
    fn a_row_offers_the_quant_that_fits() {
        let f = [
            file("Q8_0", 8., 1),
            file("Q4_K_M", 4.5, 1),
            file("Q4_K_M", 4.4, 2),
            file("Q2_K", 3., 1),
        ];
        assert_eq!(
            pick_file(&f, Some((16. * logic::GIB) as u64))
                .unwrap()
                .size_bytes,
            f[1].size_bytes
        );
        // The Q2_K fits but cannot load: the smallest quant that loads is offered (and shows Too big).
        assert_eq!(
            pick_file(&f, Some((3.5 * logic::GIB) as u64))
                .unwrap()
                .size_bytes,
            f[2].size_bytes
        );
        // A repo with only quants the engine cannot load still names one, which its header refuses.
        assert_eq!(pick_file(&f[3..], None).unwrap().quant, "Q2_K");
        // A loadable quant outside QUANTS (a UD K_XL) still beats a Q2_K that fits.
        let x = [file("Q4_K_XL", 4.6, 1), file("Q2_K", 3., 1)];
        let have = Some((3.5 * logic::GIB) as u64);
        assert_eq!(pick_file(&x, have).unwrap().quant, "Q4_K_XL");
        assert!(
            !["Q2_K", "IQ4_XS", "BF16", "Q4_1", "MXFP4"]
                .iter()
                .any(|q| loads(q))
        );
        assert_eq!(
            pick_file(&f, Some((9.5 * logic::GIB) as u64))
                .unwrap()
                .quant,
            "Q4_K_M"
        );
    }

    /// A 27B repo's 1.6 GB `-draft-Q4_0` MTP file fits a 16 GB card where the model does not; the
    /// row still offers the model.
    #[test]
    fn a_row_never_offers_a_draft_file() {
        let mut d = file("Q4_0", 1.6, 1);
        d.file = "Qwen3.8-27B-draft-Q4_0.gguf".into();
        let f = [d, file("Q4_K_M", 16.5, 1)];
        let have = Some((16. * logic::GIB) as u64);
        assert_eq!(pick_file(&f, have).unwrap().quant, "Q4_K_M");
        assert_eq!(pick_file(&f[..1], have).unwrap().quant, "Q4_0");
    }

    /// A send refused with no model loaded gives back its mode, attachment and folder, so the
    /// retry is the same request; newer text or a newer attachment stays.
    #[test]
    fn a_refused_send_gives_its_whole_draft_back() {
        let sent = Draft {
            typed: "Summarize this paper".into(),
            chip: Some("Research"),
            attach: vec!["/notes".into()],
            repo: Some("/repo".into()),
        };
        assert_eq!(sent.clone().back_into(Draft::default()), sent);
        let refused_by_prompt = Draft {
            typed: " Summarize this paper\n".into(),
            ..Draft::default()
        };
        assert_eq!(sent.clone().back_into(refused_by_prompt), sent);
        let newer = Draft {
            typed: "something else".into(),
            ..Draft::default()
        };
        assert_eq!(sent.clone().back_into(newer.clone()), newer);
        let attached = Draft {
            attach: vec!["/b.pdf".into()],
            ..Draft::default()
        };
        assert_eq!(sent.clone().back_into(attached).attach, ["/b.pdf"]);
    }

    /// Saved tool results fold into one run of steps; a paper request shows as its question.
    #[test]
    fn saved_chats_read_as_the_live_ones() {
        let m = |role, c: &str| Message {
            role,
            content: c.into(),
            truncated: false,
            images: Vec::new(),
        };
        // As the paper chips of an older version asked.
        let req = r#"Use read_papers with files ["/a.pdf"] and focus "Compare". Keep each paper separate and show any reading limits or errors."#;
        let items = saved_items(&[
            m(Role::System, "s"),
            m(Role::User, req),
            m(Role::Tool, "read 1\nmore"),
            m(Role::Tool, "read 2"),
            m(Role::Assistant, "<tool_call>\n{\"name\": \"x\"}"),
            m(Role::User, "[agent] Answer now."),
            m(Role::Assistant, "Done."),
        ]);
        assert!(
            matches!(&items[0], Item::User { text, files, .. } if text == "Compare" && files.len() == 1)
        );
        assert!(matches!(&items[1], Item::Steps(s) if s.len() == 2 && s[0].done == "2 lines"));
        // A reply of call markup alone and the loop's own message do not show.
        assert!(matches!(&items[2], Item::Text { text, .. } if text == "Done."));
        assert_eq!(items.len(), 3);
    }

    /// A saved message's images come back by reference, in order, for `chat_image` to read; a
    /// message of images alone has no words.
    #[test]
    fn saved_images_come_back_by_reference() {
        let (a, b) = ("a".repeat(64), "b".repeat(64));
        let items = saved_items(&[Message {
            role: Role::User,
            content: String::new(),
            truncated: false,
            images: vec![a.clone(), b.clone()],
        }]);
        let Item::User { text, images, .. } = &items[0] else {
            panic!("not a user message")
        };
        let refs: Vec<&str> = (images.iter())
            .map(|p| match p {
                Pic::Saved(r) => r.as_str(),
                Pic::Live(_) => panic!("a saved image is read by reference"),
            })
            .collect();
        assert_eq!((text.as_str(), refs), ("", vec![a.as_str(), b.as_str()]));
    }

    /// One chat needs 8 GB of weights, its KV cache at the context, and half a GB.
    #[test]
    fn need_bytes_counts_weights_cache_and_buffers() {
        let i = RemoteModel {
            arch: "qwen3".into(),
            supported: true,
            quant: "Q4_K_M".into(),
            weights_bytes: 8 << 30,
            params: 0,
            kv_bytes_per_token: Some(1024),
            context_length: Some(4096),
            expert_bytes: 6 << 30,
            unloadable: vec![],
        };
        let need = (8 << 30) + (4096. * 1024. / 0.85) as u64 + (1 << 29);
        assert_eq!(need_bytes(&i, 8192.), need);
        // The 6 GB of routed experts can stay in host memory.
        assert_eq!(floor_bytes(&i, 8192.), need - (6 << 30));
    }
}
