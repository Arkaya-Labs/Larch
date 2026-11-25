//! Presentation state for the native task center; the service remains authoritative.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CapabilitySet {
    pub authoring: bool,
    pub explicit_start: bool,
    pub controls: bool,
    pub retry_child: bool,
    pub execution: bool,
    pub planning: bool,
    pub schedules: bool,
    pub decisions: bool,
    pub autonomous_application: bool,
    /// Schedules and settled runs can be deleted.
    pub deletes: bool,
}

pub fn can_author(caps: CapabilitySet) -> bool {
    caps.authoring && caps.explicit_start
}

/// Which service call starts a paused workflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// Its own steps, each a model step.
    Steps,
    /// Its goal, which the service's planner turns into steps.
    Goal,
}

/// Why Start is refused, shown as the card's hint and said again if it is clicked.
pub const TOO_OLD: &str = "This task service is too old to start tasks.";
/// See [`TOO_OLD`].
pub const NO_MODEL: &str = "Load a model to start this task.";
/// See [`TOO_OLD`].
pub const REPEATING: &str = "The app cannot start a task with a repeating step.";
/// A start the workflow's revision overtook; the page refreshes after it.
pub const CHANGED: &str = "This task changed; refreshed.";

/// The service's refusal of a start, with a stale one said as the refresh that follows it.
pub fn start_error(error: String) -> String {
    match error.as_str() {
        "stale or inactive start request" | "revision conflict" => CHANGED.into(),
        _ => error,
    }
}

/// The task center's three lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tab {
    /// Running, paused or waiting on the user.
    #[default]
    Now,
    Scheduled,
    /// Finished, failed or stopped.
    Done,
}

/// One step of a run's graph as the canvas draws it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepNode {
    pub id: String,
    pub depends_on: Vec<String>,
    /// What the step is: "step", "join", "each", "repeat" or "check".
    pub kind: &'static str,
}

/// Where one step stands, read from the run's committed state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepState {
    Done,
    Running,
    /// Waiting on the user's approval.
    NeedsYou,
    /// Its last attempt failed and the run waits for a retry.
    Failed,
    Paused,
    /// A condition took it off the path.
    Skipped,
    /// The run ended before it finished.
    Stopped,
    Upcoming,
}

/// How one occurrence of a schedule shows on the week calendar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockState {
    Ran,
    Stopped,
    /// Its run has not finished yet.
    Running,
    Upcoming,
    /// Upcoming, but the schedule is off.
    Paused,
    /// Past, with no run: missed while off, asleep or skipped by the overlap policy.
    Missed,
}

/// An occurrence at `at` with the run it started, if any (both in unix seconds).
pub fn block_state(enabled: bool, at: i64, now: i64, run_status: Option<&str>) -> BlockState {
    match run_status {
        Some("succeeded") => BlockState::Ran,
        Some("failed" | "cancelled") => BlockState::Stopped,
        Some(_) => BlockState::Running,
        None if at <= now => BlockState::Missed,
        None if enabled => BlockState::Upcoming,
        None => BlockState::Paused,
    }
}

/// How the Scheduled tab lays out its occurrences.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CalView {
    Day,
    #[default]
    Week,
    /// The week's occurrences as rows, day by day.
    List,
}

/// One attempt's wall time (unix milliseconds) and tokens, as the service recorded it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attempt {
    pub step: String,
    pub started_ms: i64,
    pub finished_ms: i64,
    /// Prompt and completion tokens, when the driver counted them.
    pub tokens: Option<(u64, u64)>,
}

/// A step's attempts summed: what its card and the detail panel say.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepTiming {
    pub attempts: usize,
    pub millis: u64,
    pub prompt: u64,
    pub completion: u64,
    /// Every attempt was counted, so the token sums are whole.
    pub counted: bool,
}

impl StepTiming {
    /// Output tokens per second over the step's time; None without whole counts or any time.
    pub fn rate(&self) -> Option<f64> {
        (self.counted && self.millis > 0)
            .then(|| self.completion as f64 * 1000. / self.millis as f64)
    }
}

/// Each step's attempts summed: a retry and the items of a repeating step add up.
pub fn timings(attempts: &[Attempt]) -> std::collections::BTreeMap<String, StepTiming> {
    let mut out = std::collections::BTreeMap::<String, StepTiming>::new();
    for a in attempts {
        let t = out.entry(a.step.clone()).or_insert(StepTiming {
            counted: true,
            ..Default::default()
        });
        t.attempts += 1;
        // A clock stepped back mid-step reads as no time, never as a negative one.
        t.millis += u64::try_from(a.finished_ms - a.started_ms).unwrap_or(0);
        match a.tokens {
            Some((prompt, completion)) => {
                t.prompt += prompt;
                t.completion += completion;
            }
            None => t.counted = false,
        }
    }
    out
}

/// A step's time as the card says it: "0.4 s", "38 s", "2 min 5 s", "1 h 3 min".
pub fn duration(millis: u64) -> String {
    let secs = millis / 1000;
    match secs {
        0..10 => format!("{:.1} s", millis as f64 / 1000.),
        10..60 => format!("{secs} s"),
        60..3600 if secs.is_multiple_of(60) => format!("{} min", secs / 60),
        60..3600 => format!("{} min {} s", secs / 60, secs % 60),
        _ => format!("{} h {} min", secs / 3600, secs % 3600 / 60),
    }
}

/// The section the service put a run in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Category {
    #[default]
    Running,
    NeedsYou,
    History,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub id: String,
    pub revision: u64,
    pub status: String,
    pub goal: String,
    /// The current graph's steps, in the service's order.
    pub nodes: Vec<StepNode>,
    pub has_initial_graph: bool,
    /// Its steps are ones the app can start (no repeating step); false without steps.
    pub startable_steps: bool,
    pub planning: String,
    pub scope: Vec<String>,
    pub acceptance: Vec<String>,
    pub result: Option<String>,
    /// The section the service put the run in.
    pub category: Category,
    pub recovery: Option<ChildRecovery>,
    /// The action waiting for the user's approval, bound to its exact digest.
    pub decision: Option<Decision>,
    /// Steps a condition skipped; shown so a skipped branch is not mistaken for a failure.
    pub skipped: Vec<String>,
    /// Why the last attempt failed, as the service committed it.
    pub attempt_failure: Option<String>,
    /// An action's outcome is still unknown; the service refuses Resume until it settles.
    pub unresolved_effects: bool,
    pub started: Vec<String>,
    pub completed: Vec<String>,
    /// Finished steps' results, as the card cut them.
    pub results: std::collections::BTreeMap<String, String>,
    /// The schedule that started this run and its occurrence, in unix seconds.
    pub origin: Option<(String, i64)>,
}

/// One pending approval as the service reported it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub request_id: String,
    pub step: String,
    pub action_digest: String,
    pub scope: String,
    pub expires_at: u64,
}

impl Decision {
    /// Seconds left to answer; `None` once the service no longer takes one.
    pub fn expires_in(&self, now_secs: u64) -> Option<u64> {
        self.expires_at
            .checked_sub(now_secs)
            .filter(|left| *left > 0)
    }
}

/// The user's answer to one pending decision, carrying the digest they were shown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionAnswer {
    pub run_id: String,
    pub decision: Decision,
    pub approve: bool,
    pub command_id: String,
    pub expected_revision: u64,
}

/// An answer only for a run that is waiting on this decision, on a service that takes answers.
pub fn decision_answer(
    record: &Record,
    approve: bool,
    command_id: &str,
    caps: CapabilitySet,
    now_secs: u64,
) -> Result<DecisionAnswer, String> {
    let decision = record
        .decision
        .as_ref()
        .filter(|_| caps.decisions && record.status == "waiting" && !command_id.trim().is_empty())
        .ok_or("Approvals are unavailable.")?;
    if decision.expires_in(now_secs).is_none() {
        return Err("This approval request expired; the step will ask again.".into());
    }
    Ok(DecisionAnswer {
        run_id: record.id.clone(),
        decision: decision.clone(),
        approve,
        command_id: command_id.into(),
        expected_revision: record.revision,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChildRecovery {
    pub child_id: String,
    pub kind: String,
    pub reason: Option<String>,
}

/// Explain only a committed child failure on a parent that is still waiting.
pub fn child_recovery_explanation(
    parent_status: &str,
    child_id: &str,
    kind: &str,
    reason: Option<&str>,
) -> Option<String> {
    let child_id = child_id.trim();
    if parent_status != "waiting" || child_id.is_empty() {
        return None;
    }
    let action = match kind {
        "failed" | "failed_attempt" => "failed",
        "cancelled" => "was cancelled",
        _ => return None,
    };
    let mut explanation = format!("Child task {child_id} {action}");
    if let Some(reason) = reason.map(str::trim).filter(|reason| !reason.is_empty()) {
        explanation.push_str(": ");
        explanation.push_str(reason);
    }
    if !explanation.ends_with('.') && !explanation.ends_with('!') && !explanation.ends_with('?') {
        explanation.push('.');
    }
    explanation.push_str(" This task is waiting.");
    Some(explanation)
}

impl Record {
    pub fn recovery_explanation(&self) -> Option<String> {
        let recovery = self.recovery.as_ref()?;
        child_recovery_explanation(
            &self.status,
            &recovery.child_id,
            &recovery.kind,
            recovery.reason.as_deref(),
        )
    }

    /// A failed attempt left the run waiting on nothing else, so Resume retries it. Approvals,
    /// child failures and unsettled effects have their own paths; a saved run is paused, not here.
    pub fn can_retry(&self) -> bool {
        self.status == "waiting"
            && self.attempt_failure.is_some()
            && self.decision.is_none()
            && self.recovery.is_none()
            && !self.unresolved_effects
    }

    /// The core pauses a ready run, or one waiting on a decision or a child; a failed attempt's
    /// wait is neither, and the card carries no plain child wait, so Pause is not offered there.
    pub fn can_pause(&self) -> bool {
        self.status == "ready"
            || (self.status == "waiting" && (self.decision.is_some() || self.recovery.is_some()))
    }

    /// Where step `id` stands.
    pub fn step_state(&self, id: &str) -> StepState {
        let has = |list: &[String]| list.iter().any(|s| s == id);
        let terminal = matches!(self.status.as_str(), "succeeded" | "failed" | "cancelled");
        if has(&self.completed) {
            StepState::Done
        } else if has(&self.skipped) {
            StepState::Skipped
        } else if terminal {
            StepState::Stopped
        } else if self.status == "waiting" && self.decision.as_ref().is_some_and(|d| d.step == id) {
            StepState::NeedsYou
        } else if !has(&self.started) {
            StepState::Upcoming
        } else if self.status == "paused" {
            StepState::Paused
        } else if self.status == "waiting" && self.attempt_failure.is_some() {
            StepState::Failed
        } else {
            StepState::Running
        }
    }

    /// How many of step `id`'s dependencies have not settled.
    pub fn waits_on(&self, id: &str) -> usize {
        self.nodes.iter().find(|n| n.id == id).map_or(0, |n| {
            n.depends_on
                .iter()
                .filter(|d| !matches!(self.step_state(d), StepState::Done | StepState::Skipped))
                .count()
        })
    }

    /// The steps by depth: each row holds the steps whose longest chain of dependencies has that
    /// many steps, as indices into `nodes`. A dependency the graph does not hold is ignored.
    pub fn layers(&self) -> Vec<Vec<usize>> {
        let mut depth = vec![0usize; self.nodes.len()];
        // The service validated the graph acyclic, so `len` passes settle every depth.
        for _ in 0..self.nodes.len() {
            let mut moved = false;
            for (i, node) in self.nodes.iter().enumerate() {
                let want = node
                    .depends_on
                    .iter()
                    .filter_map(|d| self.nodes.iter().position(|n| &n.id == d))
                    .map(|j| depth[j] + 1)
                    .max()
                    .unwrap_or(0);
                if want > depth[i] {
                    depth[i] = want;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        let mut layers = vec![Vec::new(); depth.iter().max().map_or(0, |d| d + 1)];
        for (i, d) in depth.into_iter().enumerate() {
            layers[d].push(i);
        }
        layers
    }

    /// Every step's state in canvas order, for the list's progress bar.
    pub fn progress(&self) -> Vec<StepState> {
        self.layers()
            .concat()
            .into_iter()
            .map(|i| self.step_state(&self.nodes[i].id))
            .collect()
    }

    /// The run's state in a few words: "Running · step 3 of 5", "Needs you", "Done".
    pub fn headline(&self) -> String {
        match self.status.as_str() {
            "ready" => "Starting".into(),
            "running" => {
                let progress = self.progress();
                match progress.iter().position(|s| *s == StepState::Running) {
                    Some(i) => format!("Running · step {} of {}", i + 1, progress.len()),
                    None => "Running".into(),
                }
            }
            "paused" => "Paused".into(),
            "waiting" if self.decision.is_some() || self.recovery.is_some() => "Needs you".into(),
            "waiting" if self.attempt_failure.is_some() => "Step failed".into(),
            "waiting" => "Waiting".into(),
            "succeeded" => "Done".into(),
            "failed" => "Failed".into(),
            "cancelled" => "Stopped".into(),
            other => other.into(),
        }
    }

    /// The step the detail panel opens on: `picked` while it is one of this run's, else the one
    /// that needs attention, the one running, the last finished, or the first.
    pub fn focus_step(&self, picked: Option<&str>) -> Option<&str> {
        if let Some(node) = picked.and_then(|p| self.nodes.iter().find(|n| n.id == p)) {
            return Some(&node.id);
        }
        let order: Vec<&str> = self
            .layers()
            .concat()
            .into_iter()
            .map(|i| self.nodes[i].id.as_str())
            .collect();
        let find = |want: &[StepState]| {
            order
                .iter()
                .copied()
                .find(|id| want.contains(&self.step_state(id)))
        };
        find(&[StepState::NeedsYou, StepState::Failed])
            .or_else(|| find(&[StepState::Running, StepState::Paused]))
            .or_else(|| {
                order
                    .iter()
                    .rev()
                    .copied()
                    .find(|id| self.step_state(id) == StepState::Done)
            })
            .or_else(|| order.first().copied())
    }
}

/// The revision an edit opened on `opened` saves on, given the schedule as listed `now`: the
/// newer one when only a run it started moved it (a firing advances the revision and the next
/// due), so an edit open across a run is not refused as stale; `None` when anything else changed,
/// someone's edit the save must not overwrite.
pub fn edit_revision(opened: &ScheduledRecord, now: &ScheduledRecord) -> Option<u64> {
    let fired = ScheduledRecord {
        revision: opened.revision,
        next_due: opened.next_due.clone(),
        ..now.clone()
    };
    (fired == *opened).then_some(now.revision)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledRecord {
    pub id: String,
    pub revision: u64,
    pub goal: String,
    pub steps: Vec<String>,
    pub planning: String,
    pub scope: Vec<String>,
    pub acceptance: Vec<String>,
    pub schedule: String,
    pub next_due: Option<String>,
    pub enabled: bool,
    /// The policy in words: what a missed occurrence and an overlap do.
    pub catch_up: String,
    pub overlap: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Draft {
    pub goal: String,
    pub steps: Vec<String>,
    pub planning: String,
    /// What the run must meet before it is done, one criterion per line; checked once it settles.
    pub done_when: Vec<String>,
}

impl Draft {
    pub fn new(goal: &str, steps: Vec<String>, planning: &str) -> Result<Self, String> {
        let goal = goal.trim();
        if goal.is_empty() {
            return Err("Enter a goal.".into());
        }
        if !matches!(planning, "disabled" | "initial" | "adaptive") {
            return Err("Unknown planning policy.".into());
        }
        let steps: Vec<_> = steps
            .into_iter()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        let mut unique = std::collections::BTreeSet::new();
        if steps.iter().any(|s| !unique.insert(s.clone())) {
            return Err("Step identifiers must be unique.".into());
        }
        if steps.is_empty() && planning == "disabled" {
            return Err("Add starting steps or choose a planning policy.".into());
        }
        Ok(Self {
            goal: goal.into(),
            steps,
            planning: planning.into(),
            done_when: vec![],
        })
    }

    /// The draft with its "done when" criteria, one per non-empty line.
    pub fn done_when(self, lines: &str) -> Self {
        let done_when = (lines.lines().map(str::trim))
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();
        Self { done_when, ..self }
    }
}

/// A time of day as people type it ("7:45", "07:45", "7:45 pm", "6pm", "18:30"), as (hour, minute).
pub fn parse_clock(text: &str) -> Result<(u32, u32), String> {
    // The field's "7:45 am" is a placeholder: left empty, it said `"" is not a time of day`.
    let bad = || match text.trim() {
        "" => "Set the time of day it runs, such as 7:45 am or 18:30.".to_owned(),
        t => format!("\"{t}\" is not a time of day; try 7:45 am or 18:30."),
    };
    let lower = text.trim().to_ascii_lowercase();
    let (digits, half) = match lower.strip_suffix("am").or_else(|| lower.strip_suffix("a")) {
        Some(rest) => (rest.trim(), Some(false)),
        None => match lower.strip_suffix("pm").or_else(|| lower.strip_suffix("p")) {
            Some(rest) => (rest.trim(), Some(true)),
            None => (lower.as_str(), None),
        },
    };
    let (h, m) = digits.split_once(':').unwrap_or((digits, "0"));
    let (h, m): (u32, u32) = (h.parse().map_err(|_| bad())?, m.parse().map_err(|_| bad())?);
    let h = match half {
        Some(_) if !(1..=12).contains(&h) => return Err(bad()),
        Some(pm) => h % 12 + if pm { 12 } else { 0 },
        None => h,
    };
    if h > 23 || m > 59 {
        return Err(bad());
    }
    Ok((h, m))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Pause,
    Resume,
    Cancel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlRequest {
    pub run_id: String,
    pub command_id: String,
    pub expected_revision: u64,
    pub control: Control,
}

pub fn command(
    record: &Record,
    command_id: &str,
    control: Control,
    caps: CapabilitySet,
) -> Result<ControlRequest, String> {
    if !caps.controls {
        return Err("Task controls are unavailable.".into());
    }
    if command_id.trim().is_empty() {
        return Err("Command ID is required.".into());
    }
    Ok(ControlRequest {
        run_id: record.id.clone(),
        command_id: command_id.into(),
        expected_revision: record.revision,
        control,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetryChildRequest {
    pub run_id: String,
    pub child_id: String,
    pub command_id: String,
    pub expected_revision: u64,
}

/// Only a visible, committed child failure authorizes offering a retry.
pub fn retry_child_request(
    record: &Record,
    command_id: &str,
    caps: CapabilitySet,
) -> Result<RetryChildRequest, String> {
    if !caps.controls
        || !caps.retry_child
        || command_id.trim().is_empty()
        || record.recovery_explanation().is_none()
    {
        return Err("Child retry is unavailable.".into());
    }
    let recovery = record
        .recovery
        .as_ref()
        .ok_or("Child retry is unavailable.")?;
    Ok(RetryChildRequest {
        run_id: record.id.clone(),
        child_id: recovery.child_id.clone(),
        command_id: command_id.into(),
        expected_revision: record.revision,
    })
}

pub trait Host {
    fn snapshot(&self) -> Result<Vec<Record>, String>;
}

#[derive(Default)]
pub struct Model {
    records: Vec<Record>,
    schedules: Vec<ScheduledRecord>,
    cursors: std::collections::BTreeMap<String, u64>,
    /// Each run's latest committed events, newest last.
    recent: std::collections::BTreeMap<String, Vec<(u64, String)>>,
    /// Runs whose committed history shows they left the paused state they were saved in.
    started: std::collections::BTreeSet<String>,
    /// Runs whose event history did not read cleanly; only their own controls are off.
    run_errors: std::collections::BTreeMap<String, String>,
    error: Option<String>,
    /// The service answered with revisions older than this view holds, as after a backup restore.
    behind: bool,
    loaded: bool,
    pub tab: Tab,
    /// The run or schedule open in the center; None opens the tab's first.
    pub selected: Option<String>,
    /// The step open in the detail panel.
    pub step: Option<String>,
    /// The new-task composer is open in the center.
    pub composing: bool,
    /// The week the calendar shows, in weeks from this one.
    pub week: i32,
    /// The day the Day view shows, in days from today.
    pub day: i32,
    pub view: CalView,
    /// Each run's attempts, as last read.
    attempts: std::collections::BTreeMap<String, Vec<Attempt>>,
    /// Each run's step instructions by step id, as last read.
    names: std::collections::BTreeMap<String, std::collections::BTreeMap<String, String>>,
    pub capabilities: CapabilitySet,
    pub connected: bool,
    /// Finished runs the user filed as done; None until the first snapshot seeds it, so runs that
    /// ended before review existed go straight to Done.
    // ponytail: never pruned, ~40 bytes a run; prune to the snapshot's ids if the prefs file grows.
    pub reviewed: Option<std::collections::BTreeSet<String>>,
}

/// How many committed events a run keeps for its activity list.
pub const RECENT_EVENTS: usize = 6;

impl Model {
    /// What to tell the user, as a title and the run's goal, when `records` replaces this view: a
    /// run that newly needs them, and one that newly failed. Nothing on the first load, so a start
    /// does not repeat what was already shown.
    pub fn alerts(&self, records: &[Record]) -> Vec<(&'static str, String)> {
        if !self.loaded {
            return Vec::new();
        }
        let was = |id: &str| self.records.iter().find(|r| r.id == id);
        records
            .iter()
            .filter_map(|new| {
                let old = was(&new.id);
                let needs = |r: &Record| r.category == Category::NeedsYou;
                let failed = |r: &Record| r.status == "failed";
                if needs(new) && !old.is_some_and(needs) {
                    Some(("A task needs you", new.goal.clone()))
                } else if failed(new) && !old.is_some_and(failed) {
                    Some(("A task stopped on an error", new.goal.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn replace(&mut self, records: Vec<Record>) {
        self.records = records;
        // A run gone from the service takes its history with it.
        let live: std::collections::BTreeSet<&str> =
            self.records.iter().map(|r| r.id.as_str()).collect();
        self.cursors.retain(|id, _| live.contains(id.as_str()));
        self.recent.retain(|id, _| live.contains(id.as_str()));
        self.started.retain(|id| live.contains(id.as_str()));
        self.run_errors.retain(|id, _| live.contains(id.as_str()));
        self.attempts.retain(|id, _| live.contains(id.as_str()));
        self.names.retain(|id, _| live.contains(id.as_str()));
        if self.reviewed.is_none() {
            self.reviewed = Some(
                self.records
                    .iter()
                    .filter(|r| r.category == Category::History)
                    .map(|r| r.id.clone())
                    .collect(),
            );
        }
        self.error = None;
        self.behind = false;
        self.loaded = true;
        self.connected = true;
    }

    /// Forget everything read so far, so the next snapshot is taken whole. Only the user asks for
    /// this, after the service fell behind this view; an ordinary older reply is still refused.
    pub fn resync(&mut self) {
        *self = Self {
            tab: self.tab,
            selected: self.selected.take(),
            step: self.step.take(),
            composing: self.composing,
            week: self.week,
            day: self.day,
            view: self.view,
            reviewed: self.reviewed.take(),
            ..Self::default()
        };
    }

    /// The service holds older revisions than this view, so only a resync can show them.
    pub fn behind(&self) -> bool {
        self.behind
    }

    /// Only a resync can clear what this view holds: the service fell behind it, or a run's
    /// history did not read cleanly.
    pub fn needs_resync(&self) -> bool {
        self.behind || !self.run_errors.is_empty()
    }

    /// A snapshot has been shown at least once.
    pub fn loaded(&self) -> bool {
        self.loaded
    }

    /// Why this run's controls are off, when its event history did not read cleanly.
    pub fn run_error(&self, run_id: &str) -> Option<&str> {
        self.run_errors.get(run_id).map(String::as_str)
    }

    /// A run's attempts, read again after its revision moved.
    pub fn set_attempts(&mut self, run_id: &str, attempts: Vec<Attempt>) {
        self.attempts.insert(run_id.to_owned(), attempts);
    }

    /// A run's step instructions, read again after its revision moved.
    pub fn set_names(&mut self, run_id: &str, names: std::collections::BTreeMap<String, String>) {
        self.names.insert(run_id.to_owned(), names);
    }

    /// What a step does, as its card names it: its instruction, else its id.
    pub fn name<'a>(&'a self, run_id: &str, step: &'a str) -> &'a str {
        (self.names.get(run_id))
            .and_then(|n| n.get(step))
            .map_or(step, String::as_str)
    }

    /// A run's steps' time and tokens, by step.
    pub fn timings(&self, run_id: &str) -> std::collections::BTreeMap<String, StepTiming> {
        self.attempts
            .get(run_id)
            .map_or_else(Default::default, |a| timings(a))
    }

    pub fn replace_schedules(&mut self, schedules: Vec<ScheduledRecord>) {
        self.schedules = schedules;
    }

    pub fn schedules(&self) -> &[ScheduledRecord] {
        &self.schedules
    }

    /// How many runs are running now.
    pub fn running(&self) -> usize {
        (self.records.iter())
            .filter(|r| r.category == Category::Running)
            .count()
    }

    /// The runs a tab lists, in the service's order.
    pub fn in_tab(&self, tab: Tab) -> Vec<&Record> {
        self.records
            .iter()
            .filter(|r| match tab {
                Tab::Now => {
                    matches!(r.category, Category::Running | Category::NeedsYou)
                        || self.to_review(r)
                }
                Tab::Done => r.category == Category::History && !self.to_review(r),
                Tab::Scheduled => false,
            })
            .collect()
    }

    /// A run that ended on its own and waits on Now until the user files it as done.
    pub fn to_review(&self, r: &Record) -> bool {
        matches!(r.status.as_str(), "succeeded" | "failed")
            && self.reviewed.as_ref().is_some_and(|s| !s.contains(&r.id))
    }

    /// A run's headline, saying so while a finished run waits on the user.
    pub fn headline(&self, r: &Record) -> String {
        if self.to_review(r) && r.status == "succeeded" {
            "Ready for review".into()
        } else {
            r.headline()
        }
    }

    /// File a finished run under Done.
    pub fn mark_done(&mut self, run_id: &str) {
        self.reviewed
            .get_or_insert_with(Default::default)
            .insert(run_id.to_owned());
    }

    /// Whether the user may delete run `r`: it is settled, no effect of it waits to be checked,
    /// and no saved schedule holds it (a schedule's runs can go once the schedule has).
    pub fn can_delete(&self, r: &Record) -> bool {
        self.capabilities.deletes
            && matches!(r.status.as_str(), "succeeded" | "failed" | "cancelled")
            && !r.unresolved_effects
            && (r.origin.as_ref()).is_none_or(|(id, _)| self.schedules.iter().all(|s| &s.id != id))
    }

    /// Forget a deleted run at once, so it neither lingers until the next snapshot nor shows
    /// for review again, and leaves the Done list it was filed in.
    pub fn forget(&mut self, run_id: &str) {
        self.records.retain(|r| r.id != run_id);
        if let Some(reviewed) = self.reviewed.as_mut() {
            reviewed.remove(run_id);
        }
    }

    /// The count on a tab's badge.
    pub fn count(&self, tab: Tab) -> usize {
        match tab {
            Tab::Scheduled => self.schedules.len(),
            _ => self.in_tab(tab).len(),
        }
    }

    /// The run open in the center: the selected one while its tab lists it, else the tab's first.
    pub fn current(&self) -> Option<&Record> {
        let listed = self.in_tab(self.tab);
        listed
            .iter()
            .find(|r| Some(&r.id) == self.selected.as_ref())
            .or(listed.first())
            .copied()
    }

    /// The schedule open in the center, chosen as [`Model::current`] chooses a run.
    pub fn current_schedule(&self) -> Option<&ScheduledRecord> {
        self.schedules
            .iter()
            .find(|s| Some(&s.id) == self.selected.as_ref())
            .or(self.schedules.first())
    }

    /// The runs a schedule started, newest occurrence first.
    pub fn runs_of(&self, schedule_id: &str) -> Vec<&Record> {
        let mut runs: Vec<&Record> = self
            .records
            .iter()
            .filter(|r| r.origin.as_ref().is_some_and(|(id, _)| id == schedule_id))
            .collect();
        runs.sort_by_key(|r| std::cmp::Reverse(r.origin.as_ref().map(|(_, at)| *at)));
        runs
    }

    /// Apply a committed event page atomically; an incomplete page never advances the cursor.
    /// A gap is kept against that run alone, so the other runs keep their controls.
    pub fn apply_events(&mut self, run_id: &str, events: &[(u64, String)]) -> Result<(), String> {
        let mut cursor = self.cursor(run_id);
        let mut seen = Vec::new();
        let mut started = false;
        for (revision, kind) in events {
            if *revision <= cursor {
                continue;
            }
            if cursor.checked_add(1) != Some(*revision) {
                let error = "This run's event history has a gap, so its steps are hidden and                              it cannot be started from here.";
                self.run_errors.insert(run_id.to_owned(), error.into());
                return Err(error.into());
            }
            cursor = *revision;
            // A run saved paused commits nothing until it starts, except a grant revocation or a
            // backup restore. ponytail: inferred from history; use the service's own flag once
            // TaskCard carries one.
            started |= !matches!(kind.as_str(), "grant_revoked" | "restored_paused");
            seen.push((*revision, kind.clone()));
        }
        self.run_errors.remove(run_id);
        self.cursors.insert(run_id.to_owned(), cursor);
        if started {
            self.started.insert(run_id.to_owned());
        }
        if !seen.is_empty() {
            let recent = self.recent.entry(run_id.to_owned()).or_default();
            recent.extend(seen);
            let extra = recent.len().saturating_sub(RECENT_EVENTS);
            recent.drain(..extra);
        }
        Ok(())
    }

    pub fn cursor(&self, run_id: &str) -> u64 {
        self.cursors.get(run_id).copied().unwrap_or(0)
    }

    pub fn cursors(&self) -> std::collections::BTreeMap<String, u64> {
        self.cursors.clone()
    }

    /// The run's latest committed events, newest last.
    pub fn recent(&self, run_id: &str) -> &[(u64, String)] {
        self.recent.get(run_id).map_or(&[], Vec::as_slice)
    }

    /// A paused run that has started: Resume returns it to the state it paused from. A run saved
    /// paused has never started and needs an explicit start instead.
    pub fn user_paused(&self, record: &Record) -> bool {
        record.status == "paused" && self.started.contains(&record.id)
    }

    /// The first start a card offers: None without one to offer (not paused, paused by the user,
    /// which Resume answers, or no controls), else the call that starts it or why it cannot. A
    /// run whose history has a gap offers none: whether it ever started is read from that history,
    /// and a Start on a run that had would be refused. Pause and Stop go by its status.
    pub fn start(&self, record: &Record) -> Option<Result<Start, &'static str>> {
        let caps = self.capabilities;
        if !caps.controls
            || record.status != "paused"
            || self.user_paused(record)
            || self.run_errors.contains_key(&record.id)
        {
            return None;
        }
        // Steps are model steps too, and the service names no activity kinds: every host with a
        // model has a planner, so planning stands for running model steps.
        let runs_models = caps.execution && caps.planning;
        Some(if !caps.explicit_start {
            Err(TOO_OLD)
        } else if !runs_models {
            Err(NO_MODEL)
        } else if !record.has_initial_graph {
            Ok(Start::Goal)
        } else if record.startable_steps {
            Ok(Start::Steps)
        } else {
            Err(REPEATING)
        })
    }

    /// Take the host's snapshot unless it is older than this view, with what it tells the user
    /// (`alerts`): a refused one tells nothing, or every poll would repeat it.
    pub fn refresh(&mut self, host: &impl Host) -> Result<Vec<(&'static str, String)>, String> {
        match host.snapshot() {
            Ok(records) => {
                let stale = records.iter().any(|new| {
                    self.cursor(&new.id) > new.revision
                        || self
                            .records
                            .iter()
                            .any(|old| old.id == new.id && old.revision > new.revision)
                });
                if stale {
                    let error = "The service holds older task revisions than this view, as after \
                                 restoring a backup. Reload from the service to show them."
                        .to_owned();
                    self.error = Some(error.clone());
                    self.behind = true;
                    self.connected = false;
                    return Err(error);
                }
                let alerts = self.alerts(&records);
                self.replace(records);
                Ok(alerts)
            }
            Err(error) => {
                self.error = Some(error.clone());
                self.connected = false;
                Err(error)
            }
        }
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

/// The retry identity of the most recently submitted workflow draft.
#[derive(Default)]
pub struct PendingSubmission(Option<(Draft, String, String)>);

impl PendingSubmission {
    pub fn from_entry(entry: Option<(Draft, String, String)>) -> Self {
        Self(entry)
    }

    pub fn into_entry(self) -> Option<(Draft, String, String)> {
        self.0
    }

    pub fn begin(&mut self, draft: &Draft, run_id: &str, command_id: &str) -> (String, String) {
        match &self.0 {
            Some((old, run, command)) if old == draft => (run.clone(), command.clone()),
            _ => {
                let run = run_id.to_owned();
                let command = command_id.to_owned();
                self.0 = Some((draft.clone(), run.clone(), command.clone()));
                (run, command)
            }
        }
    }

    pub fn complete(&mut self, run_id: &str, command_id: &str) -> bool {
        let current = self
            .0
            .as_ref()
            .is_some_and(|(_, run, command)| run == run_id && command == command_id);
        if current {
            self.0 = None;
        }
        current
    }

    pub fn fail(&mut self, run_id: &str, command_id: &str) -> bool {
        self.0
            .as_ref()
            .is_some_and(|(_, run, command)| run == run_id && command == command_id)
    }
}

/// Runs with a command unsettled: a second click would carry the same revision and only conflict.
#[derive(Default)]
pub struct InFlight(std::collections::BTreeMap<String, u64>);

impl InFlight {
    /// Claim the run for one command; false while its last command is unsettled.
    pub fn begin(&mut self, run_id: &str) -> bool {
        if self.0.contains_key(run_id) {
            return false;
        }
        self.0.insert(run_id.to_owned(), u64::MAX);
        true
    }

    /// The command answered while refresh `request` was the latest; the run stays claimed until a
    /// later refresh shows its new revision.
    pub fn answered(&mut self, run_id: &str, request: u64) {
        if let Some(after) = self.0.get_mut(run_id) {
            *after = request;
        }
    }

    /// Refresh `request` landed, settling every command answered before it began.
    pub fn refreshed(&mut self, request: u64) {
        self.0.retain(|_, after| *after >= request);
    }

    /// The run has a command unsettled.
    pub fn contains(&self, run_id: &str) -> bool {
        self.0.contains_key(run_id)
    }
}
