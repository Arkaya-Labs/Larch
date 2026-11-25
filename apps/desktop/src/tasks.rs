//! Native workflow center. All mutations go through the service client.
use crate::input::{Submitted, TextInput};
use crate::tasks_model::{
    Attempt, BlockState, CalView, CapabilitySet, Category, ChildRecovery, Control, Decision, Draft,
    Host, PendingSubmission, Record, ScheduledRecord, Start, StepNode, StepState, Tab, duration,
};
use crate::theme::{Theme, fade};
use crate::tree::{Mood, Tree, tree};
use crate::ui::{Btn, Ecl, button, card, eyebrow, head, mono, now_ms, serif, u};
use chrono::{DateTime, Datelike, Local, NaiveDate, NaiveTime, TimeZone, Timelike, Utc};
use gpui::{
    AnyElement, AppContext, Context, Div, Entity, Focusable, FontWeight, Hsla, InteractiveElement,
    IntoElement, ParentElement, Pixels, Rgba, Stateful, StatefulInteractiveElement, Styled, div,
    prelude::FluentBuilder,
};
use larch_app::task::{
    PlanningPolicy, ServiceCapabilities, TASK_API_VERSION, TaskCard, TaskCommand, TaskControl,
    TaskSection,
};
use larch_task::graph::NodeKind;
use larch_task::schedule::{CatchUp, Overlap, Schedule};
use larch_task::types::RunStatus;
use std::sync::atomic::{AtomicU64, Ordering};

struct Snapshot(Result<Vec<Record>, String>);
impl Host for Snapshot {
    fn snapshot(&self) -> Result<Vec<Record>, String> {
        self.0.clone()
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);
fn id() -> String {
    format!(
        "desktop-{}-{}",
        now_ms(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
}
/// Events read per request, the most the service pages.
const EVENT_PAGE: u32 = 999;
/// Occurrences one schedule may put on a week.
// ponytail: a schedule more often than every ~3 h fills its week at this cap; draw it as a band
// when one shows up.
const WEEK_BLOCKS: usize = 64;

/// The model's planning key: one exhaustive match, so a new policy fails to compile here.
fn planning(policy: &PlanningPolicy) -> String {
    match policy {
        PlanningPolicy::Disabled => "disabled",
        PlanningPolicy::Initial => "initial",
        PlanningPolicy::Adaptive => "adaptive",
    }
    .into()
}

/// The policy a draft's planning key names; Draft::new admits only these keys.
fn policy(key: &str) -> Result<PlanningPolicy, String> {
    match key {
        "disabled" => Ok(PlanningPolicy::Disabled),
        "initial" => Ok(PlanningPolicy::Initial),
        "adaptive" => Ok(PlanningPolicy::Adaptive),
        _ => Err("Unknown planning policy.".into()),
    }
}

/// The model's status key: one exhaustive match, so a renamed or new status fails to compile here.
fn status(status: &RunStatus) -> String {
    match status {
        RunStatus::Ready => "ready",
        RunStatus::Running => "running",
        RunStatus::Paused => "paused",
        RunStatus::Waiting => "waiting",
        RunStatus::Succeeded => "succeeded",
        RunStatus::Failed => "failed",
        RunStatus::Cancelled => "cancelled",
    }
    .into()
}

/// A step's kind as the canvas labels it: one exhaustive match, so a new kind fails to compile here.
fn kind(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::Activity => "step",
        NodeKind::Join => "join",
        NodeKind::Map { .. } => "each",
        NodeKind::Loop { .. } => "repeat",
        NodeKind::Condition { .. } => "check",
    }
}

/// Every service capability by name, so one added later must be mapped before this compiles.
pub(crate) fn capabilities(caps: ServiceCapabilities) -> CapabilitySet {
    // The desktop tells the user of runs itself (`Model::alerts`), from its own snapshots.
    // activity_kinds: the start gate reads `planning`, which every host with a model offers.
    let ServiceCapabilities {
        protocol_version: _,
        notifications: _,
        activity_kinds: _,
        step_stats: _,
        schedule_updates: _,
        // The client pages the list itself where the host can.
        list_pages: _,
        authoring,
        explicit_start,
        controls,
        retry_child,
        execution,
        planning,
        schedules,
        decisions,
        autonomous_application,
        deletes,
    } = caps;
    CapabilitySet {
        authoring,
        explicit_start,
        controls,
        retry_child,
        execution,
        planning,
        schedules,
        decisions,
        autonomous_application,
        deletes,
    }
}

/// A service card as the model shows it; the section is the service's, not re-derived here.
pub(crate) fn project(card: TaskCard) -> Record {
    let nodes = card.graph.as_ref().map_or_else(Vec::new, |g| {
        g.nodes
            .iter()
            .map(|n| StepNode {
                id: n.id.clone(),
                depends_on: n.depends_on.clone(),
                kind: kind(&n.kind),
            })
            .collect()
    });
    let planning = planning(&card.definition.planning_policy);
    let category = match card.section {
        TaskSection::Running => Category::Running,
        TaskSection::NeedsYou => Category::NeedsYou,
        TaskSection::History => Category::History,
    };
    let recovery = card.child_failure.map(|failure| ChildRecovery {
        child_id: failure.child_id,
        kind: match failure.kind {
            larch_app::task::ChildFailureKind::FailedAttempt => "failed_attempt",
            larch_app::task::ChildFailureKind::Failed => "failed",
            larch_app::task::ChildFailureKind::Cancelled => "cancelled",
        }
        .to_owned(),
        reason: failure.reason,
    });
    Record {
        id: card.run_id,
        revision: card.revision,
        status: status(&card.status),
        category,
        goal: card.definition.goal,
        nodes,
        has_initial_graph: card.definition.initial_graph.is_some(),
        startable_steps: card
            .graph
            .as_ref()
            .is_some_and(|g| larch_app::task::step_activities(g).is_ok()),
        planning,
        scope: card.definition.grants,
        acceptance: card.definition.acceptance,
        result: card.result,
        recovery,
        decision: card.pending_decision.map(|d| Decision {
            request_id: d.request_id,
            step: d.step,
            action_digest: d.action_digest,
            scope: d.scope,
            expires_at: d.expires_at,
        }),
        skipped: card.skipped_steps,
        // Shown and retried as a failed step: without this the run waits forever with only Stop.
        attempt_failure: card.attempt_failure.or_else(|| {
            (card.interrupted)
                .then(|| "The step was cut off when Larch closed or the model changed.".into())
        }),
        unresolved_effects: card.unresolved_effects,
        started: card.started_steps,
        completed: card.completed_steps,
        results: card.step_results,
        origin: card
            .schedule_origin
            .and_then(|(id, at)| Some((id, DateTime::parse_from_rfc3339(&at).ok()?.timestamp()))),
    }
}

/// A time of day as the design writes it: "7:45 am", "6 pm".
pub(crate) fn clock(t: NaiveTime) -> String {
    t.format(if t.minute() == 0 {
        "%-I %P"
    } else {
        "%-I:%M %P"
    })
    .to_string()
}

/// "Every 2 h": an interval in the largest unit that divides it.
fn every(seconds: u32) -> String {
    match seconds {
        86_400 => "Every day".into(),
        s if s % 86_400 == 0 => format!("Every {} days", s / 86_400),
        s if s % 3_600 == 0 => format!("Every {} h", s / 3_600),
        s if s % 60 == 0 => format!("Every {} min", s / 60),
        s => format!("Every {s} s"),
    }
}

/// A rule in a few words, for the list.
pub(crate) fn rule_label(rule: &Schedule) -> String {
    match rule {
        Schedule::Calendar { time, days, .. } => {
            format!("{} · {}", larch_app::task::weekdays(*days), clock(*time))
        }
        Schedule::Interval { anchor, seconds } => format!(
            "{} · from {}",
            every(*seconds),
            clock(anchor.with_timezone(&Local).time())
        ),
        Schedule::Once { at } => {
            let at = at.with_timezone(&Local);
            format!("Once · {} {}", at.format("%a %b %-d"), clock(at.time()))
        }
    }
}

/// A rule as the detail panel says it.
fn rule_sentence(rule: &Schedule) -> String {
    match rule {
        Schedule::Calendar {
            time,
            timezone,
            days,
            ..
        } => format!(
            "{} at {} ({timezone})",
            larch_app::task::weekdays(*days),
            clock(*time)
        ),
        Schedule::Interval { seconds, .. } => every(*seconds),
        Schedule::Once { at } => {
            let at = at.with_timezone(&Local);
            format!(
                "Once, on {} at {}",
                at.format("%a %b %-d"),
                clock(at.time())
            )
        }
    }
}

/// How long until `at`, in the largest whole unit.
fn until(at: DateTime<Utc>, now: DateTime<Utc>) -> String {
    match (at - now).num_minutes() {
        m if m < 1 => "now".into(),
        m if m < 60 => format!("in {m} min"),
        m if m < 48 * 60 => format!("in {} h", m / 60),
        m => format!("in {} days", m / (24 * 60)),
    }
}

/// When the schedule next runs, or that it is off.
fn next_label(item: &ScheduledRecord) -> String {
    if !item.enabled {
        return "off".into();
    }
    item.next_due
        .as_deref()
        .and_then(|d| DateTime::parse_from_rfc3339(d).ok())
        .map_or("nothing due".into(), |d| {
            format!("next {}", until(d.with_timezone(&Utc), Utc::now()))
        })
}

pub fn refresh(s: &mut Ecl, cx: &mut Context<Ecl>) {
    // The refresh already running may have read the service before what asked for this one.
    if s.task_loading {
        s.task_dirty = true;
        return;
    }
    s.task_loading = true;
    s.task_dirty = false;
    s.task_request += 1;
    let request = s.task_request;
    let cursors = s.tasks.cursors();
    s.try_call(
        cx,
        move |app| {
            let service = app.task_service()?;
            let (caps, cards) = service.snapshot()?;
            let schedules = if caps.schedules {
                service.schedules()?
            } else {
                Vec::new()
            };
            let mut pages = Vec::new();
            // Only runs that moved past their cursor, read to the end so a long history is whole at once.
            for card in &cards {
                let mut cursor = cursors.get(&card.run_id).copied().unwrap_or(0);
                let mut events = Vec::new();
                while cursor < card.revision {
                    let page = service.events(&card.run_id, cursor, EVENT_PAGE)?;
                    if page.next_cursor <= cursor {
                        break;
                    }
                    cursor = page.next_cursor;
                    events.extend(page.events.into_iter().map(|e| (e.revision, e.kind)));
                    if !page.has_more {
                        break;
                    }
                }
                if !events.is_empty() {
                    // Its steps' times change only when its revision moves.
                    let attempts = service
                        .step_stats(&card.run_id)?
                        .into_iter()
                        .map(|a| Attempt {
                            step: a.node_id,
                            started_ms: a.started_at.timestamp_millis(),
                            finished_ms: a.finished_at.timestamp_millis(),
                            tokens: a.tokens,
                        })
                        .collect();
                    // A planner names its steps s1, s2…: the card shows what each one does instead.
                    let names = service.step_instructions(&card.run_id)?;
                    pages.push((card.run_id.clone(), events, attempts, names));
                }
            }
            Ok((caps, cards, schedules, pages))
        },
        move |s, (caps, cards, schedules, pages), cx| {
            if request != s.task_request {
                return;
            }
            s.task_loading = false;
            s.task_busy.refreshed(request);
            if !std::mem::replace(&mut s.task_watch, true) {
                watch(cx);
            }
            s.tasks.capabilities = capabilities(caps);
            let alerts = s
                .tasks
                .refresh(&Snapshot(Ok(cards.into_iter().map(project).collect())));
            if alerts.is_err() {
                s.tasks.capabilities = CapabilitySet::default();
            } else {
                save_reviewed(s);
                for (run_id, events, attempts, names) in pages {
                    // A gap stays on that run's card and turns off its controls alone.
                    let _ = s.tasks.apply_events(&run_id, &events);
                    s.tasks.set_attempts(&run_id, attempts);
                    s.tasks.set_names(&run_id, names);
                }
                s.task_rules = schedules
                    .iter()
                    .map(|item| (item.id.clone(), (item.rule.clone(), item.policy)))
                    .collect();
                s.tasks
                    .replace_schedules(schedules.into_iter().map(scheduled).collect());
            }
            // A run waiting on the user or stopped overnight is seen without the Tasks page open.
            for (title, goal) in alerts.unwrap_or_default() {
                s.notify_os(title, &goal);
            }
            if s.task_dirty {
                refresh(s, cx);
            }
        },
        move |s, error, cx| {
            if request != s.task_request {
                return;
            }
            s.task_loading = false;
            s.task_busy.refreshed(request);
            s.tasks.capabilities = CapabilitySet::default();
            let _ = s.tasks.refresh(&Snapshot(Err(error)));
            if s.task_dirty {
                refresh(s, cx);
            }
        },
    );
}

/// A schedule as the Scheduled tab shows it.
fn scheduled(item: larch_app::task::ScheduledCard) -> ScheduledRecord {
    let steps = (item.definition.initial_graph.as_ref())
        .map_or_else(Vec::new, |g| g.nodes.iter().map(|n| n.id.clone()).collect());
    let planning = planning(&item.definition.planning_policy);
    let (catch_up, overlap) = policy_labels(&item.policy);
    ScheduledRecord {
        id: item.id,
        revision: item.revision,
        goal: item.definition.goal,
        steps,
        planning,
        scope: item.definition.grants,
        acceptance: item.definition.acceptance,
        schedule: rule_label(&item.rule),
        next_due: item.next_due,
        enabled: item.enabled,
        catch_up,
        overlap,
    }
}

/// What a missed occurrence and an overlapping one do, in words: one exhaustive match each.
pub(crate) fn policy_labels(policy: &larch_task::schedule::Policy) -> (String, String) {
    let catch_up = match policy.catch_up {
        CatchUp::Latest => "Run the latest one".into(),
        CatchUp::All { max } => format!("Run each one, up to {max}"),
    };
    let overlap = match policy.overlap {
        Overlap::Skip => "Skip this one".into(),
        Overlap::Queue { max_queued } => format!("Queue it, up to {max_queued}"),
        Overlap::BufferLatest => "Run the newest after it".into(),
        Overlap::Parallel { max_active } => format!("Run side by side, up to {max_active}"),
    };
    (catch_up, overlap)
}

/// Start over from the service after it fell behind this view, as after a backup restore.
fn resync(s: &mut Ecl, cx: &mut Context<Ecl>) {
    s.tasks.resync();
    // The reply in flight was read against the old cursors; the new request makes it ignored.
    s.task_loading = false;
    refresh(s, cx);
}

/// Send one command for a run, then refresh; clicks on that run are ignored until the refresh
/// after the reply shows its new revision.
fn send(
    s: &mut Ecl,
    run_id: String,
    cx: &mut Context<Ecl>,
    call: impl FnOnce(&larch_app::App) -> Result<(), String> + Send + 'static,
) {
    send_then(s, run_id, cx, call, |_| ());
}

/// [`send`], running `then` once the service has taken the command.
fn send_then(
    s: &mut Ecl,
    run_id: String,
    cx: &mut Context<Ecl>,
    call: impl FnOnce(&larch_app::App) -> Result<(), String> + Send + 'static,
    then: impl FnOnce(&mut Ecl) + 'static,
) {
    if !s.task_busy.begin(&run_id) {
        return;
    }
    let failed = run_id.clone();
    s.try_call(
        cx,
        call,
        move |s, (), cx| {
            s.task_busy.answered(&run_id, s.task_request);
            then(s);
            refresh(s, cx);
        },
        move |s, error, cx| {
            s.task_busy.answered(&failed, s.task_request);
            s.say(error);
            refresh(s, cx);
        },
    );
}

/// Reconnect by polling authoritative service snapshots while this page is open.
pub fn poll(generation: u64, cx: &mut Context<Ecl>) {
    cx.spawn(async move |this, cx| {
        cx.background_executor()
            .timer(std::time::Duration::from_secs(3))
            .await;
        this.update(cx, |s, cx| {
            if s.page == crate::ui::Page::Tasks && s.task_poll == generation {
                refresh(s, cx);
                poll(generation, cx);
            }
        })
        .ok();
    })
    .detach();
}

/// Refresh every 15 s on any page, so a run that needs the user or failed is told of with the
/// Tasks page closed. `refresh` runs one at a time, so this and `poll` never stack, and the model
/// alone remembers what was told.
pub(crate) fn watch(cx: &mut Context<Ecl>) {
    cx.spawn(async move |this, cx| {
        loop {
            cx.background_executor()
                .timer(std::time::Duration::from_secs(15))
                .await;
            if this.update(cx, refresh).is_err() {
                break;
            }
        }
    })
    .detach();
}

/// The composer's draft as its fields read now.
fn composed(s: &Ecl, cx: &Context<Ecl>) -> Result<Draft, String> {
    // A field holds one step; one pasted with line breaks holds one per line.
    let steps = (s.task_steps.iter())
        .flat_map(|i| {
            i.read(cx)
                .text
                .lines()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .collect();
    Draft::new(&s.task_goal.read(cx).text, steps, &s.task_planning)
        .map(|d| d.done_when(&s.task_done.read(cx).text))
}

fn submit(s: &mut Ecl, cx: &mut Context<Ecl>) {
    if !crate::tasks_model::can_author(s.tasks.capabilities) {
        s.say("Saving tasks needs a service with explicit start support.");
        return;
    }
    let draft = match composed(s, cx) {
        Ok(draft) => draft,
        Err(error) => {
            s.say(error);
            return;
        }
    };
    if s.task_repeat || s.task_editing.is_some() {
        return save_schedule(s, draft, cx);
    }
    let mut pending = PendingSubmission::from_entry(s.task_pending.take());
    let (run_id, command_id) = pending.begin(&draft, &id(), &id());
    s.task_pending = pending.into_entry();
    let success_run = run_id.clone();
    let success_command = command_id.clone();
    let failure_run = run_id.clone();
    let failure_command = command_id.clone();
    let saved = draft.clone();
    s.try_call(
        cx,
        move |app| {
            app.task_service()?.create_intent(
                &command_id,
                run_id,
                draft.goal,
                draft.steps,
                draft.done_when,
                policy(&draft.planning)?,
            )?;
            Ok(())
        },
        move |s, (), cx| {
            let mut pending = PendingSubmission::from_entry(s.task_pending.take());
            let current = pending.complete(&success_run, &success_command);
            s.task_pending = pending.into_entry();
            if current {
                s.say("Task saved. Press Start when you're ready.");
                s.tasks.tab = Tab::Now;
                s.tasks.selected = Some(success_run.clone());
                s.tasks.step = None;
                s.tasks.composing = false;
                // A second click would save a second run; text edited since the save stays.
                if composed(s, cx).as_ref() == Ok(&saved) {
                    for i in [&s.task_goal, &s.task_done] {
                        i.update(cx, |i, cx| i.set_text("", cx));
                    }
                    s.task_steps.clear();
                }
            }
            refresh(s, cx);
        },
        move |s, error, _| {
            let mut pending = PendingSubmission::from_entry(s.task_pending.take());
            let current = pending.fail(&failure_run, &failure_command);
            s.task_pending = pending.into_entry();
            if current {
                s.say(error);
            }
        },
    );
}

/// A calendar rule at `time` on `days`. An edited one keeps its zone, start and rule revision,
/// so a computer moved to another zone never shifts it; a new one starts today in `here`.
fn calendar_rule(
    kept: Option<&Schedule>,
    here: Option<chrono_tz::Tz>,
    time: NaiveTime,
    days: u8,
) -> Option<Schedule> {
    let (start, timezone, rule_revision) = match kept {
        Some(Schedule::Calendar {
            start,
            timezone,
            rule_revision,
            ..
        }) => (*start, *timezone, rule_revision.clone()),
        _ => {
            let here = here?;
            (
                Utc::now().with_timezone(&here).date_naive(),
                here,
                chrono_tz::IANA_TZDB_VERSION.into(),
            )
        }
    };
    Some(Schedule::Calendar {
        start,
        time,
        timezone,
        rule_revision,
        days,
    })
}

/// Save the draft to repeat on the composer's days and time, in this computer's time zone; or,
/// editing a schedule, replace its goal and steps, and its days and time when it repeats by them.
fn save_schedule(s: &mut Ecl, draft: Draft, cx: &mut Context<Ecl>) {
    if !s.tasks.capabilities.schedules {
        s.say("Repeating tasks need a service with schedules.");
        return;
    }
    if draft.steps.is_empty() {
        s.say("Add steps to repeat a task: a repeating task runs the same steps each time.");
        return;
    }
    let opened = s.task_editing.clone();
    let editing = (opened.as_ref()).map(|e| (e.id.clone(), e.revision));
    let kept = editing
        .as_ref()
        .and_then(|(id, _)| s.task_rules.get(id))
        .cloned();
    if editing.is_some() && kept.is_none() {
        s.say("This schedule is gone.");
        return;
    }
    let schedule = match kept.as_ref().map(|(rule, _)| rule) {
        // The composer sets days and a time; any other rule is kept as it is.
        Some(rule @ (Schedule::Once { .. } | Schedule::Interval { .. })) => rule.clone(),
        _ => {
            if s.task_days == 0 {
                s.say("Pick at least one day.");
                return;
            }
            let (h, m) = match crate::tasks_model::parse_clock(&s.task_time.read(cx).text) {
                Ok(hm) => hm,
                Err(error) => {
                    s.say(error);
                    return;
                }
            };
            let here = iana_time_zone::get_timezone()
                .ok()
                .and_then(|tz| tz.parse::<chrono_tz::Tz>().ok());
            let Some(time) = NaiveTime::from_hms_opt(h, m, 0) else {
                return;
            };
            match calendar_rule(kept.as_ref().map(|(rule, _)| rule), here, time, s.task_days) {
                Some(rule) => rule,
                None => {
                    s.say("This computer's time zone is not one the scheduler knows, so a time of day cannot be saved.");
                    return;
                }
            }
        }
    };
    // The composer's own key, or the schedule's, so a second click waits for the first save.
    let key = editing.as_ref().map_or_else(
        || "schedule:new".to_owned(),
        |(id, _)| format!("schedule:{id}"),
    );
    if !s.task_busy.begin(&key) {
        return;
    }
    let schedule_id = editing.as_ref().map_or_else(id, |(id, _)| id.clone());
    let (saved, selected, failed, edited) = (
        draft.clone(),
        schedule_id.clone(),
        key.clone(),
        editing.is_some(),
    );
    s.try_call(
        cx,
        move |app| {
            let service = app.task_service()?;
            let planning = policy(&draft.planning)?;
            match (editing, kept) {
                (Some((id, revision)), Some((_, policy))) => {
                    let update = |revision| {
                        service.update_schedule(
                            &id,
                            revision,
                            schedule.clone(),
                            policy,
                            draft.goal.clone(),
                            draft.steps.clone(),
                            draft.done_when.clone(),
                            planning.clone(),
                        )
                    };
                    match update(revision) {
                        // A run the schedule started since the edit opened moved only its
                        // revision: the edit saves on the schedule as it is now.
                        Err(e) if e.contains("stale revision") => {
                            let now = (service.schedules()?.into_iter())
                                .find(|c| c.id == id)
                                .map(scheduled);
                            let fired = (opened.as_ref().zip(now.as_ref()))
                                .and_then(|(o, n)| crate::tasks_model::edit_revision(o, n));
                            fired.map_or(Err(e), update)
                        }
                        saved => saved,
                    }
                    .map(drop)
                    .map_err(|e| {
                        if e.contains("stale revision") {
                            "This schedule changed since you opened it; refreshed. Edit it again."
                                .into()
                        } else {
                            e
                        }
                    })
                }
                _ => service.create_schedule(
                    &schedule_id,
                    schedule,
                    draft.goal,
                    draft.steps,
                    draft.done_when,
                    planning,
                ),
            }
        },
        move |s, (), cx| {
            s.task_busy.answered(&key, s.task_request);
            s.say(if edited {
                "Schedule saved. Runs it already started keep their old steps."
            } else {
                "Schedule saved, off. Turn it on when you're ready."
            });
            s.tasks.tab = Tab::Scheduled;
            s.tasks.selected = Some(selected);
            s.tasks.composing = false;
            s.task_editing = None;
            // Text edited since the save stays.
            if composed(s, cx).as_ref() == Ok(&saved) {
                for i in [&s.task_goal, &s.task_done] {
                    i.update(cx, |i, cx| i.set_text("", cx));
                }
                s.task_steps.clear();
            }
            refresh(s, cx);
        },
        move |s, error, cx| {
            s.task_busy.answered(&failed, s.task_request);
            s.say(error);
            refresh(s, cx);
        },
    );
}

/// A step's field in the builder, holding `text`.
/// Enter in it adds the next step under it and moves there; Shift+Enter breaks the line.
fn step_field(th: Theme, text: &str, cx: &mut Context<Ecl>) -> Entity<TextInput> {
    let i = cx.new(|cx| TextInput::new(cx, "What should this step do?", true, 4, th));
    i.update(cx, |i, cx| i.set_text(text, cx));
    // Lives as long as the field: a removed step takes it with it.
    cx.subscribe(&i, |s, from, _: &Submitted, cx| {
        let at = (s.task_steps.iter())
            .position(|x| x == &from)
            .map_or(s.task_steps.len(), |p| p + 1);
        let next = step_field(s.theme(), "", cx);
        s.task_steps.insert(at, next.clone());
        s.focus_next = Some(next);
        cx.notify();
    })
    .detach();
    i
}

/// Open the composer on a goal and its steps: a run's, saved as a new task, or a schedule's
/// (`editing`), saved in place.
fn edit(
    s: &mut Ecl,
    goal: &str,
    steps: &[String],
    planning: &str,
    done_when: &[String],
    editing: Option<ScheduledRecord>,
    cx: &mut Context<Ecl>,
) {
    s.task_goal.update(cx, |i, cx| i.set_text(goal, cx));
    let th = s.theme();
    s.task_steps = steps.iter().map(|t| step_field(th, t, cx)).collect();
    s.task_done
        .update(cx, |i, cx| i.set_text(&done_when.join("\n"), cx));
    s.task_planning = planning.to_owned();
    if let Some(Schedule::Calendar { time, days, .. }) = editing
        .as_ref()
        .and_then(|e| s.task_rules.get(&e.id))
        .map(|(rule, _)| rule)
    {
        let text = clock(*time);
        s.task_time.update(cx, |i, cx| i.set_text(&text, cx));
        s.task_days = *days;
    }
    s.task_repeat = editing.is_some();
    s.task_editing = editing;
    s.tasks.composing = true;
    cx.notify();
}

/// Open the run's goal and results as a chat, to ask about it or carry it on.
fn open_as_chat(s: &mut Ecl, run_id: String, cx: &mut Context<Ecl>) {
    let rev = s.chats_rev;
    s.try_call(
        cx,
        move |app| {
            let chat = app.chat_from_task(&run_id)?;
            Ok((chat, app.list_chats()?))
        },
        move |s, (chat, chats), cx| {
            s.take_chats(chats, rev, cx);
            s.open_chat(chat, cx);
        },
        |s, error, _| s.say(error),
    );
}

/// Turn a schedule on or off at the revision shown.
fn toggle_schedule(s: &mut Ecl, item: ScheduledRecord, cx: &mut Context<Ecl>) {
    send(s, format!("schedule:{}", item.id), cx, move |app| {
        app.task_service()?
            .enable_schedule(&item.id, item.revision, !item.enabled)?;
        Ok(())
    });
}

/// Delete a schedule at the revision shown; the runs it started stay.
fn delete_schedule(s: &mut Ecl, item: ScheduledRecord, cx: &mut Context<Ecl>) {
    send(s, format!("schedule:{}", item.id), cx, move |app| {
        app.task_service()?.delete_schedule(&item.id, item.revision)
    });
}

/// Delete a settled run with its steps and results.
fn delete_run(s: &mut Ecl, run_id: String, cx: &mut Context<Ecl>) {
    let gone = run_id.clone();
    send_then(
        s,
        run_id.clone(),
        cx,
        move |app| app.task_service()?.delete_run(&run_id),
        move |s| {
            s.tasks.forget(&gone);
            save_reviewed(s);
        },
    );
}

/// A Delete that takes a second click, as a chat's does: what it deletes cannot come back.
fn delete_button(
    s: &Ecl,
    th: &Theme,
    id: String,
    cx: &mut Context<Ecl>,
    delete: impl Fn(&mut Ecl, &mut Context<Ecl>) + 'static,
) -> impl IntoElement {
    let key = format!("delete:{id}");
    let armed = s.armed.as_deref() == Some(key.as_str());
    button(
        th,
        key.clone(),
        if armed {
            "Click again to delete"
        } else {
            "Delete"
        },
        Btn::Danger,
        cx.listener(move |s, _, _, cx| {
            if s.armed.as_deref() == Some(key.as_str()) {
                s.armed = None;
                delete(s, cx);
            } else {
                s.armed = Some(key.clone());
            }
            cx.notify();
        }),
    )
}

/// Keep the saved review list in step with the view's, so a restart shows the same Now.
fn save_reviewed(s: &mut Ecl) {
    if s.tasks.reviewed != s.prefs.reviewed {
        s.prefs.reviewed = s.tasks.reviewed.clone();
        s.save_prefs();
    }
}

/// File a finished run under Done.
fn mark_done(s: &mut Ecl, run_id: &str) {
    s.tasks.mark_done(run_id);
    save_reviewed(s);
}

/// Run a schedule's task once now: a new run of its definition, started as Start would.
fn run_now(s: &mut Ecl, schedule_id: String, cx: &mut Context<Ecl>) {
    let key = format!("schedule:{schedule_id}");
    let read = move |service: &larch_app::task::ServiceClient| {
        Ok(service
            .schedules()?
            .into_iter()
            .find(|c| c.id == schedule_id)
            .ok_or("This schedule is gone.")?
            .definition)
    };
    start_new(s, key, cx, read, |_| ());
}

/// Run a finished run's task again as a new run; the old one is filed under Done once the new
/// one has started, so a failed Run again leaves it waiting for review.
fn run_again(s: &mut Ecl, run_id: String, cx: &mut Context<Ecl>) {
    let (old, done) = (run_id.clone(), run_id.clone());
    let read = move |service: &larch_app::task::ServiceClient| {
        Ok(service.run(&old)?.ok_or("This task is gone.")?.definition)
    };
    start_new(s, run_id, cx, read, move |s| mark_done(s, &done));
}

/// A new run of the definition `read` returns, started as Start would, and opened on Now; `then`
/// runs once it has started.
fn start_new(
    s: &mut Ecl,
    key: String,
    cx: &mut Context<Ecl>,
    read: impl FnOnce(
        &larch_app::task::ServiceClient,
    ) -> Result<larch_task::types::WorkflowDefinition, String>
    + Send
    + 'static,
    then: impl FnOnce(&mut Ecl) + 'static,
) {
    let run_id = id();
    let (create, begin) = (id(), id());
    let shown = run_id.clone();
    let call = move |app: &larch_app::App| {
        let service = app.task_service()?;
        let definition = read(&service)?;
        let steps = definition.initial_graph.is_some();
        let reply = service.create_run(&create, run_id.clone(), definition)?;
        if steps {
            service.start_steps(&run_id, &begin, reply.revision)
        } else {
            service.start_goal(&run_id, &begin, reply.revision)
        }
        .map_err(crate::tasks_model::start_error)?;
        Ok(())
    };
    send_then(s, key, cx, call, then);
    s.tasks.tab = Tab::Now;
    s.tasks.selected = Some(shown);
    s.tasks.step = None;
}

fn start(s: &mut Ecl, record: Record, cx: &mut Context<Ecl>) {
    let kind = match s.tasks.start(&record) {
        Some(Ok(kind)) => kind,
        Some(Err(why)) => {
            s.say(why);
            return;
        }
        None => return,
    };
    let command_id = id();
    send(s, record.id.clone(), cx, move |app| {
        let service = app.task_service()?;
        match kind {
            Start::Steps => service.start_steps(&record.id, &command_id, record.revision),
            Start::Goal => service.start_goal(&record.id, &command_id, record.revision),
        }
        .map_err(crate::tasks_model::start_error)?;
        Ok(())
    });
}

fn retry_child(s: &mut Ecl, record: Record, cx: &mut Context<Ecl>) {
    let request =
        match crate::tasks_model::retry_child_request(&record, &id(), s.tasks.capabilities) {
            Ok(request) => request,
            Err(error) => {
                s.say(error);
                return;
            }
        };
    send(s, request.run_id.clone(), cx, move |app| {
        app.task_service()?.retry_child(
            &request.run_id,
            &request.child_id,
            &request.command_id,
            request.expected_revision,
        )?;
        Ok(())
    });
}

fn answer(s: &mut Ecl, record: Record, approve: bool, cx: &mut Context<Ecl>) {
    let now = now_ms() / 1000;
    let answer = match crate::tasks_model::decision_answer(
        &record,
        approve,
        &id(),
        s.tasks.capabilities,
        now,
    ) {
        Ok(answer) => answer,
        Err(error) => {
            s.say(error);
            return;
        }
    };
    send(s, answer.run_id.clone(), cx, move |app| {
        let d = answer.decision;
        let pending = larch_app::task::PendingDecision {
            request_id: d.request_id,
            step: d.step,
            action_digest: d.action_digest,
            scope: d.scope,
            expires_at: d.expires_at,
        };
        app.task_service()?.respond_decision(
            &answer.run_id,
            &pending,
            answer.approve,
            &answer.command_id,
            answer.expected_revision,
        )?;
        Ok(())
    });
}

fn control(s: &mut Ecl, record: Record, action: Control, cx: &mut Context<Ecl>) {
    let request = match crate::tasks_model::command(&record, &id(), action, s.tasks.capabilities) {
        Ok(c) => c,
        Err(error) => {
            s.say(error);
            return;
        }
    };
    let control = match request.control {
        Control::Pause => TaskControl::Pause,
        Control::Resume => TaskControl::Resume,
        Control::Cancel => TaskControl::Cancel,
    };
    let command = TaskCommand {
        version: TASK_API_VERSION,
        run_id: request.run_id.clone(),
        command_id: request.command_id,
        expected_revision: request.expected_revision,
        control,
    };
    send(s, request.run_id, cx, move |app| {
        app.task_service()?.control(command)?;
        Ok(())
    });
}

/// A committed event in words; events carry no step, so these stay general.
fn event_label(kind: &str) -> String {
    match kind {
        "attempt_started" => "A step started".into(),
        "attempt_finished" => "A step finished".into(),
        "attempt_failed" => "A step failed".into(),
        "attempt_retrying" => "A step failed and will try again".into(),
        "acceptance_recorded" => "A done-when check passed".into(),
        "acceptance_failed" => "A done-when check failed".into(),
        "decision_required" => "Asked for your approval".into(),
        "decision_approved" => "You approved".into(),
        "decision_rejected" => "You said no".into(),
        "graph_changed" => "The plan changed".into(),
        "run_succeeded" => "Finished".into(),
        "cancelled" => "Stopped".into(),
        other => {
            let words = other.replace('_', " ");
            let mut chars = words.chars();
            chars
                .next()
                .map_or_else(String::new, |c| c.to_uppercase().chain(chars).collect())
        }
    }
}

fn eid(parts: std::fmt::Arguments) -> gpui::ElementId {
    gpui::ElementId::Name(parts.to_string().into())
}

/// Gold text on a gold wash: the sun darkened toward the ink so it reads.
fn gold_ink(th: &Theme) -> Hsla {
    crate::theme::mix(th.sun, th.ink, 0.45).into()
}

/// A step's colour: green done, gold running, ember for you, red failed, quiet otherwise.
fn tone(th: &Theme, state: StepState) -> Hsla {
    match state {
        StepState::Done => th.keep.into(),
        StepState::Running => th.sun.into(),
        StepState::NeedsYou => th.ember.into(),
        StepState::Failed | StepState::Stopped => th.lose.into(),
        StepState::Paused | StepState::Skipped | StepState::Upcoming => th.ink3.into(),
    }
}

/// A small rounded status chip: a dot, then the words.
fn chip(dot: Option<Hsla>, text: impl Into<gpui::SharedString>, color: Hsla, bg: Hsla) -> Div {
    div()
        .flex()
        .flex_none()
        .items_center()
        .gap(u(7.))
        .px(u(11.))
        .py(u(4.))
        .rounded_full()
        .bg(bg)
        .text_size(u(13.))
        .text_color(color)
        .font_weight(FontWeight::MEDIUM)
        .when_some(dot, |d, c| d.child(div().size(u(7.)).rounded_full().bg(c)))
        .child(text.into())
}

/// A count badge, its number centred in the circle.
fn badge(th: &Theme, n: usize, on: bool, half: Pixels) -> Div {
    div()
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .size(u(22.))
        .rounded_full()
        .bg(if on {
            Hsla::from(th.accent)
        } else {
            fade(th.ink3, 0.18)
        })
        .text_color(if on { th.on_accent } else { th.ink2 })
        .text_size(u(12.))
        .line_height(u(22.))
        .font_weight(FontWeight::SEMIBOLD)
        .child(crate::ui::mid(half, n.to_string()))
}

/// The breadcrumb in the top bar.
pub fn title(s: &Ecl, th: &Theme) -> AnyElement {
    let sub = if s.tasks.composing {
        Some(
            if s.task_editing.is_some() {
                "Edit schedule"
            } else {
                "New task"
            }
            .to_string(),
        )
    } else if s.tasks.tab == Tab::Scheduled {
        Some("Scheduled".into())
    } else {
        // A run's goal heads the page under it, so the bar leaves it out.
        None
    };
    div()
        .flex()
        .items_center()
        .gap(u(8.))
        .min_w_0()
        .text_size(u(15.))
        .child(head(th, "tasks", "Tasks"))
        .when_some(sub, |d, sub| {
            d.child(
                div()
                    .min_w_0()
                    .truncate()
                    .text_color(th.ink2)
                    .child(format!("/ {sub}")),
            )
        })
        .into_any_element()
}

/// The top bar's run status and controls, or the schedules' summary.
pub fn chips(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Vec<AnyElement> {
    let mut out = Vec::new();
    if s.tasks.composing {
        return out;
    }
    let pill = |text: String| {
        div()
            .flex_none()
            .px(u(14.))
            .py(u(6.))
            .rounded_full()
            .border_1()
            .border_color(th.line)
            .bg(th.card)
            .text_size(u(13.))
            .child(text)
            .into_any_element()
    };
    if s.tasks.tab == Tab::Scheduled {
        let schedules = s.tasks.schedules();
        if schedules.is_empty() {
            return out;
        }
        let on = schedules.iter().filter(|x| x.enabled).count();
        out.push(pill(format!("{on} on · {} off", schedules.len() - on)));
        let next = schedules
            .iter()
            .filter(|x| x.enabled)
            .filter_map(|x| {
                Some((
                    DateTime::parse_from_rfc3339(x.next_due.as_deref()?)
                        .ok()?
                        .with_timezone(&Utc),
                    x,
                ))
            })
            .min_by_key(|(at, _)| *at);
        if let Some((at, x)) = next {
            out.push(pill(format!(
                "Next · {} {}",
                short(&x.goal, 22),
                until(at, Utc::now())
            )));
        }
        return out;
    }
    let Some(record) = s.tasks.current() else {
        return out;
    };
    let (dot, color, bg): (Hsla, Hsla, Hsla) = match record.status.as_str() {
        "running" | "ready" => (th.sun.into(), gold_ink(th), fade(th.sun, 0.18)),
        "waiting" if record.attempt_failure.is_some() && record.decision.is_none() => {
            (th.lose.into(), th.lose.into(), fade(th.lose, 0.1))
        }
        // Ember is for "needs you" only, as on its step card.
        "waiting" if record.decision.is_some() || record.recovery.is_some() => {
            (th.ember.into(), th.ember.into(), fade(th.ember, 0.12))
        }
        "waiting" => (th.accent.into(), th.accent.into(), fade(th.accent, 0.12)),
        "failed" | "cancelled" => (th.lose.into(), th.lose.into(), fade(th.lose, 0.1)),
        "succeeded" => (th.keep.into(), th.keep.into(), fade(th.keep, 0.12)),
        _ => (th.ink3.into(), th.ink2.into(), th.sunk.into()),
    };
    out.push(
        chip(Some(dot), s.tasks.headline(record), color, bg)
            .py(u(6.))
            .px(u(13.))
            .into_any_element(),
    );
    if s.task_busy.contains(&record.id) {
        out.push(
            div()
                .text_size(u(13.))
                .text_color(th.ink3)
                .child("Sending…")
                .into_any_element(),
        );
        return out;
    }
    // Beside the run's state, where Stop is: a finished run goes from here.
    if s.tasks.can_delete(record) {
        let id = record.id.clone();
        out.push(
            delete_button(s, th, record.id.clone(), cx, move |s, cx| {
                delete_run(s, id.clone(), cx)
            })
            .into_any_element(),
        );
    }
    if !s.tasks.capabilities.controls {
        return out;
    }
    if let Some(Ok(_)) = s.tasks.start(record) {
        let r = record.clone();
        out.push(
            button(
                th,
                "task-start",
                "Start",
                Btn::Primary,
                cx.listener(move |s, _, _, cx| start(s, r.clone(), cx)),
            )
            .into_any_element(),
        );
    }
    let action = if record.can_pause() {
        Some((Control::Pause, "Pause"))
    } else if s.tasks.user_paused(record) {
        Some((Control::Resume, "Resume"))
    } else {
        None
    };
    if let Some((action, label)) = action {
        let r = record.clone();
        out.push(
            button(
                th,
                format!("task-{label}"),
                label,
                Btn::Ghost,
                cx.listener(move |s, _, _, cx| control(s, r.clone(), action, cx)),
            )
            .into_any_element(),
        );
    }
    if !matches!(record.status.as_str(), "succeeded" | "failed" | "cancelled") {
        let r = record.clone();
        out.push(
            button(
                th,
                "task-stop",
                "Stop",
                Btn::Ghost,
                cx.listener(move |s, _, _, cx| control(s, r.clone(), Control::Cancel, cx)),
            )
            .into_any_element(),
        );
    }
    out
}

/// The page width, in `u`, the list (330), a 440 wide center and the detail (360) need side by
/// side; a narrower page narrows the list and puts the detail under the center.
const PANES: f32 = 1130.;

/// The first `n` characters of `text`, with an ellipsis when cut.
fn short(text: &str, n: usize) -> String {
    if text.chars().count() <= n {
        return text.into();
    }
    format!(
        "{}…",
        text.chars().take(n - 1).collect::<String>().trim_end()
    )
}

pub fn page(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let center: AnyElement = if s.tasks.composing {
        composer(s, th, cx).into_any_element()
    } else if s.tasks.tab == Tab::Scheduled {
        calendar(s, th, cx).into_any_element()
    } else if let Some(record) = s.tasks.current() {
        canvas(s, record, th, cx).into_any_element()
    } else {
        empty(
            th,
            if s.tasks.tab == Tab::Done {
                "Nothing finished yet."
            } else {
                "Nothing running."
            },
            "Start one with + New task: give it a goal, and steps if you have them.",
        )
        .into_any_element()
    };
    let detail: Option<AnyElement> = if s.tasks.composing {
        None
    } else if s.tasks.tab == Tab::Scheduled {
        s.tasks
            .current_schedule()
            .map(|x| schedule_detail(s, x, th, cx).into_any_element())
    } else {
        s.tasks
            .current()
            .map(|r| step_detail(s, r, th, cx).into_any_element())
    };
    // Beside the list and the detail, a narrow page's center was a sliver (the week's days ran
    // together; at the window's smallest, nothing): there the detail goes under the center.
    let (beside, below) = match s.room >= PANES {
        true => (detail, None),
        false => (None, detail),
    };
    let page = div()
        .size_full()
        .flex()
        .overflow_hidden()
        .child(list(s, th, cx))
        .child(
            div()
                .id("task-center")
                .flex_1()
                .min_w_0()
                .h_full()
                .overflow_y_scroll()
                .px(u(40.))
                .py(u(30.))
                .child(center)
                .when_some(below, |d, detail| {
                    d.child(
                        div()
                            .mt(u(32.))
                            .pt(u(24.))
                            .border_t_1()
                            .border_color(th.line)
                            .child(detail),
                    )
                }),
        )
        .when_some(beside, |d, detail| {
            d.child(
                div()
                    .id("task-detail")
                    .w(u(360.))
                    .flex_none()
                    .h_full()
                    .overflow_y_scroll()
                    .bg(th.side)
                    .border_l_1()
                    .border_color(th.line)
                    .px(u(28.))
                    .py(u(30.))
                    .child(detail),
            )
        });
    // A run's result and a step's output are the chat's Markdown, selected as a chat's text is.
    crate::ui::selects(page, cx)
}

fn empty(th: &Theme, head: &str, body: &str) -> Div {
    div()
        .flex()
        .flex_col()
        .gap(u(10.))
        .pt(u(60.))
        .items_center()
        .child(serif(head.to_owned(), 28.))
        .child(
            div()
                .text_color(th.ink2)
                .max_w(u(420.))
                .text_center()
                .child(body.to_owned()),
        )
}

/// The left column: New task, the tabs and the selected tab's list.
fn list(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Stateful<Div> {
    let th2 = *th;
    let mut col = div()
        .id("task-list")
        .w(u(if s.room >= PANES { 330. } else { 280. }))
        .flex_none()
        .h_full()
        .overflow_y_scroll()
        .border_r_1()
        .border_color(th.line)
        .px(u(16.))
        .py(u(24.))
        .flex()
        .flex_col()
        .gap(u(6.));
    col = col.child(
        div()
            .id("task-new")
            .flex()
            .items_center()
            .justify_center()
            .h(u(50.))
            .rounded(u(12.))
            .bg(th.accent)
            .text_color(th.on_accent)
            .text_size(u(15.))
            .font_weight(FontWeight::SEMIBOLD)
            .cursor_pointer()
            .hover(move |d| d.bg(fade(th2.accent, 0.9)))
            .on_click(cx.listener(|s, _, _, cx| {
                s.tasks.composing = true;
                s.task_editing = None;
                cx.notify();
            }))
            .child("+ New task"),
    );
    let mut tabs = div()
        .flex()
        .gap(u(2.))
        .p(u(4.))
        .mt(u(14.))
        .mb(u(10.))
        .rounded(u(12.))
        .bg(th.sunk);
    for (tab, label) in [
        (Tab::Now, "Now"),
        (Tab::Scheduled, "Scheduled"),
        (Tab::Done, "Done"),
    ] {
        let on = s.tasks.tab == tab && !s.tasks.composing;
        let n = s.tasks.count(tab);
        tabs = tabs.child(
            div()
                .id(label)
                .flex_auto()
                .px(u(6.))
                .flex()
                .items_center()
                .justify_center()
                .gap(u(5.))
                .h(u(38.))
                .rounded(u(9.))
                .when(on, |d| d.bg(th.card).font_weight(FontWeight::SEMIBOLD))
                .text_color(if on { th.ink } else { th.ink2 })
                .text_size(u(14.))
                .cursor_pointer()
                .on_click(cx.listener(move |s, _, _, cx| {
                    s.tasks.tab = tab;
                    s.tasks.selected = None;
                    s.tasks.step = None;
                    s.tasks.composing = false;
                    cx.notify();
                }))
                .child(label)
                .when(tab != Tab::Done && n > 0, |d| {
                    d.child(badge(th, n, on, s.half))
                }),
        );
    }
    col = col.child(tabs);
    match s.tasks.tab {
        Tab::Scheduled => {
            let selected = if s.tasks.composing {
                None
            } else {
                s.tasks.current_schedule().map(|x| x.id.clone())
            };
            for item in s.tasks.schedules() {
                let on = selected.as_deref() == Some(item.id.as_str());
                let id = item.id.clone();
                let runs = s.tasks.runs_of(&item.id);
                let mut marks = div().flex().gap(u(5.)).mt(u(8.));
                for run in runs.iter().take(5).rev() {
                    let color: Hsla = match run.status.as_str() {
                        "succeeded" => th.keep.into(),
                        "failed" | "cancelled" => th.lose.into(),
                        _ => th.sun.into(),
                    };
                    // Figtree has no ✓, and a fallback font's tick sat off centre: it is drawn.
                    let mark = match run.status.as_str() {
                        "succeeded" => crate::ui::icon("check", 13., color).into_any_element(),
                        "failed" | "cancelled" => crate::ui::mid(s.half, "!").into_any_element(),
                        _ => div()
                            .size(u(7.))
                            .rounded_full()
                            .bg(color)
                            .into_any_element(),
                    };
                    marks = marks.child(
                        div()
                            .flex()
                            .items_center()
                            .justify_center()
                            .size(u(26.))
                            .rounded(u(7.))
                            .bg(color.opacity(0.16))
                            .text_color(color)
                            .text_size(u(13.))
                            .line_height(u(26.))
                            .font_weight(FontWeight::BOLD)
                            .child(mark),
                    );
                }
                col = col.child(
                    row(
                        th,
                        on,
                        eid(format_args!("sched-{}", item.id)),
                        cx.listener(move |s, _, _, cx| {
                            s.tasks.selected = Some(id.clone());
                            s.tasks.composing = false;
                            cx.notify();
                        }),
                    )
                    .child(
                        crate::ui::lines(2., 22.)
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(item.goal.clone()),
                    )
                    .child(div().text_size(u(13.)).text_color(th.ink2).child(format!(
                        "{} · {}",
                        item.schedule,
                        next_label(item)
                    )))
                    .when(!runs.is_empty(), |d| d.child(marks)),
                );
            }
            col = col.child(
                div()
                    .px(u(14.))
                    .pt(u(14.))
                    .text_size(u(13.))
                    .text_color(th.ink3)
                    .child(if s.tasks.capabilities.schedules {
                        "Tip: in New task, choose Repeats to make a schedule."
                    } else {
                        "Scheduling is unavailable on this service."
                    }),
            );
        }
        tab => {
            let selected = if s.tasks.composing {
                None
            } else {
                s.tasks.current().map(|r| r.id.clone())
            };
            let records = s.tasks.in_tab(tab);
            if !records.is_empty() {
                col = col.child(
                    div()
                        .px(u(14.))
                        .pb(u(4.))
                        .child(eyebrow(th, if tab == Tab::Now { "Now" } else { "Earlier" })),
                );
            }
            for record in records {
                let on = selected.as_deref() == Some(record.id.as_str());
                let id = record.id.clone();
                let mut item = row(
                    th,
                    on,
                    eid(format_args!("run-{}", record.id)),
                    cx.listener(move |s, _, _, cx| {
                        s.tasks.selected = Some(id.clone());
                        s.tasks.step = None;
                        s.tasks.composing = false;
                        cx.notify();
                    }),
                )
                .child(
                    crate::ui::lines(2., 22.)
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(record.goal.clone()),
                );
                let progress = record.progress();
                let needs_you = record.status == "waiting"
                    && (record.decision.is_some() || record.recovery.is_some());
                item = if needs_you {
                    item.child(div().flex().mt(u(6.)).child(chip(
                        None,
                        "Needs you",
                        th.ember.into(),
                        fade(th.ember, 0.12),
                    )))
                } else if s.tasks.to_review(record) && record.status == "succeeded" {
                    item.child(div().flex().mt(u(6.)).child(chip(
                        None,
                        "Ready for review",
                        th.keep.into(),
                        fade(th.keep, 0.12),
                    )))
                } else if tab == Tab::Done || progress.is_empty() {
                    let color: Hsla = match record.status.as_str() {
                        "failed" | "cancelled" => th.lose.into(),
                        _ => th.ink2.into(),
                    };
                    item.child(
                        div()
                            .text_size(u(13.))
                            .text_color(color)
                            .child(record.headline()),
                    )
                } else {
                    item.child(div().flex().gap(u(4.)).mt(u(8.)).children(
                        progress.into_iter().map(|state| {
                            let c = match state {
                                StepState::Upcoming | StepState::Paused => fade(th.ink3, 0.22),
                                other => tone(th, other).opacity(0.8),
                            };
                            div().flex_1().h(u(6.)).rounded_full().bg(c)
                        }),
                    ))
                };
                col = col.child(item);
            }
        }
    }
    if s.tasks.count(s.tasks.tab) == 0 {
        col = col.child(
            div()
                .px(u(14.))
                .text_size(u(13.))
                .text_color(th.ink3)
                .child(match s.tasks.tab {
                    Tab::Now => "Nothing running right now.",
                    Tab::Scheduled => "No schedules yet.",
                    Tab::Done => "Finished tasks show up here.",
                }),
        );
    }
    col.child(div().flex_1()).child(status_line(s, th, cx))
}

/// One list row; `on` marks the open one.
fn row(
    th: &Theme,
    on: bool,
    id: gpui::ElementId,
    click: impl Fn(&gpui::ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> Stateful<Div> {
    let th2 = *th;
    div()
        .id(id)
        .flex()
        .flex_col()
        .gap(u(2.))
        .px(u(14.))
        .py(u(12.))
        .rounded(u(10.))
        .text_size(u(15.))
        .cursor_pointer()
        .when(on, |d| d.bg(fade(th.accent, 0.12)))
        .when(!on, |d| d.hover(move |d| d.bg(fade(th2.ink3, 0.08))))
        .on_click(click)
}

/// The service connection, quietly at the foot of the list.
fn status_line(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let banner = if s.tasks.behind() {
        "The service fell behind this view · showing the last snapshot"
    } else if !s.tasks.connected && s.tasks.error().is_none() {
        "Connecting to the task service…"
    } else if !s.tasks.connected && s.tasks.loaded() {
        "Service unavailable · showing the last snapshot"
    } else if !s.tasks.connected {
        "Task service unavailable"
    } else if !s.tasks.capabilities.execution {
        "Connected · tasks can be saved; load a model to run them"
    } else {
        ""
    };
    let mut out = div()
        .flex()
        .flex_col()
        .gap(u(4.))
        .px(u(14.))
        .pt(u(12.))
        .text_size(u(12.5))
        .text_color(th.ink3)
        .child(banner);
    if let Some(error) = s.tasks.error() {
        out = out.child(div().child(error.to_owned()));
    }
    if s.tasks.needs_resync() {
        out = out.child(
            div()
                .id("task-resync")
                .cursor_pointer()
                .text_color(th.accent)
                .on_click(cx.listener(|s, _, _, cx| resync(s, cx)))
                .child("Reload from the service"),
        );
    }
    out
}

/// A step card's width; the spine between rows runs down its middle.
const STEP_W: f32 = 300.;

/// A step card's width: `STEP_W`, or what a narrow page's center has left beside the list (280),
/// its padding (80) and the board's (32), so the card is not cut at the board's edge.
fn step_w(s: &Ecl) -> f32 {
    if s.room >= PANES {
        STEP_W
    } else {
        (s.room - 392.).clamp(180., STEP_W)
    }
}

/// The run canvas: the goal, then its steps in rows by depth.
fn canvas(s: &Ecl, record: &Record, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let mut out = div()
        .flex()
        .flex_col()
        .gap(u(14.))
        .max_w(u(900.))
        .child(eyebrow(th, "Goal"))
        .child(serif(
            record.goal.clone(),
            if s.room >= PANES { 36. } else { 28. },
        ));
    if let Some(error) = s.tasks.run_error(&record.id) {
        return out.child(div().text_color(th.ink3).child(error.to_owned()));
    }
    if let Some(Err(why)) = s.tasks.start(record) {
        out = out.child(div().text_color(th.ink2).child(why));
    }
    let focus = record
        .focus_step(s.tasks.step.as_deref())
        .map(str::to_owned);
    let mut board = div()
        .flex()
        .flex_col()
        .items_center()
        .p(u(if s.room >= PANES { 28. } else { 16. }))
        .mt(u(8.))
        .rounded(u(16.))
        .border_1()
        .border_color(th.line)
        .bg(th.sunk);
    let layers = record.layers();
    let order = layers.concat();
    if layers.is_empty() {
        board = board.child(
            div()
                .text_color(th.ink2)
                .child(match record.status.as_str() {
                    "paused" => "No steps yet: the agent plans them when it starts.",
                    "succeeded" | "failed" | "cancelled" => "This task had no steps.",
                    _ => "Planning the steps…",
                }),
        );
    }
    let settled = |layer: &Vec<usize>| {
        layer.iter().all(|i| {
            matches!(
                record.step_state(&record.nodes[*i].id),
                StepState::Done | StepState::Skipped
            )
        })
    };
    // A wire is solid once the row above it settled, dashed while it has not.
    let wire = |d: Div, done: bool| {
        d.border_color(if done {
            Hsla::from(th.keep)
        } else {
            fade(th.ink3, 0.5)
        })
        .when(!done, |d| d.border_dashed())
    };
    let drop = |h: f32, done: bool| wire(div().w(u(0.)).h(u(h)).border_l_2(), done);
    // Parallel steps hang from a bar across their centres, and join one under them.
    let bar = |done: bool| {
        wire(
            div()
                .absolute()
                .left(u(step_w(s) / 2.))
                .right(u(step_w(s) / 2.))
                .h(u(0.))
                .border_t_2(),
            done,
        )
    };
    for (depth, layer) in layers.iter().enumerate() {
        let above = (depth > 0).then(|| settled(&layers[depth - 1]));
        let below = (depth + 1 < layers.len()).then(|| settled(layer));
        let fan = layer.len() > 1;
        if let Some(done) = above {
            board = board.child(drop(14., done));
        }
        let row = div()
            .relative()
            .flex()
            .gap(u(14.))
            .when_some(above.filter(|_| fan), |d, done| d.child(bar(done).top_0()))
            .when_some(below.filter(|_| fan), |d, done| {
                d.child(bar(done).bottom_0())
            })
            .children(layer.iter().map(|i| {
                let node = &record.nodes[*i];
                let n = order.iter().position(|j| j == i).map_or(0, |p| p + 1);
                div()
                    .flex()
                    .flex_col()
                    .items_center()
                    .when_some(above.filter(|_| fan), |d, done| d.child(drop(12., done)))
                    .child(step_card(
                        s,
                        record,
                        node,
                        n,
                        focus.as_deref() == Some(node.id.as_str()),
                        th,
                        cx,
                    ))
                    .when_some(below.filter(|_| fan), |d, done| d.child(drop(12., done)))
            }));
        // A wide row scrolls sideways rather than wrapping away from its wires.
        board = board.child(
            div()
                .id(eid(format_args!("layer-{depth}")))
                .max_w_full()
                .overflow_x_scroll()
                .child(row),
        );
    }
    out = out.child(board);
    // Before the run ends this is the planner's answer, or a failure the step already shows.
    let ended = match record.status.as_str() {
        "succeeded" => Some("Result"),
        "failed" => Some("Why it stopped"),
        _ => None,
    };
    if let (Some(result), Some(title)) = (&record.result, ended) {
        // As a chat's reply: its Markdown read as "**Red:**", and none of it could be copied.
        out = out.child(
            card(th)
                .mt(u(8.))
                .child(eyebrow(th, title))
                .child(crate::ui::markdown(th, &s.sel, RESULT_ITEM, result)),
        );
    }
    out
}

/// The selection keys of a run's result and of a step's output, as the Tasks page uses the chat's
/// Markdown: past `select::PLACES`, each a place of its own, which a drag stays in.
const RESULT_ITEM: usize = usize::MAX;
const STEP_ITEM: usize = usize::MAX - 1;

/// "Step 3", or "Step 3 · join" for a step that is not a plain one.
fn step_name(n: usize, kind: &str) -> String {
    if kind == "step" {
        format!("Step {n}")
    } else {
        format!("Step {n} · {kind}")
    }
}

fn step_card(
    s: &Ecl,
    record: &Record,
    node: &StepNode,
    n: usize,
    open: bool,
    th: &Theme,
    cx: &mut Context<Ecl>,
) -> Stateful<Div> {
    let timing = s.tasks.timings(&record.id).get(&node.id).copied();
    let state = record.step_state(&node.id);
    let label: String = match state {
        StepState::Done => {
            timing.map_or("Done".into(), |t| format!("Done · {}", duration(t.millis)))
        }
        StepState::Running => "Running".into(),
        StepState::NeedsYou => "Needs you".into(),
        StepState::Failed => "Failed · retry".into(),
        StepState::Paused => "Paused".into(),
        StepState::Skipped => "Skipped".into(),
        StepState::Stopped => "Not run".into(),
        StepState::Upcoming => match record.waits_on(&node.id) {
            0 => "Next".into(),
            1 => "Waits on 1 step".into(),
            n => format!("Waits on {n} steps"),
        },
    };
    let color = tone(th, state);
    let quiet = matches!(
        state,
        StepState::Upcoming | StepState::Stopped | StepState::Skipped | StepState::Paused
    );
    let border: Hsla = if state == StepState::Running {
        th.sun.into()
    } else if open {
        th.accent.into()
    } else {
        th.line.into()
    };
    let (text, wash) = if quiet {
        (Hsla::from(th.ink2), fade(th.ink3, 0.12))
    } else if state == StepState::Running {
        (gold_ink(th), fade(th.sun, 0.18))
    } else {
        (color, color.opacity(0.14))
    };
    let id = node.id.clone();
    let (icon, kind_tone) = match node.kind {
        "step" => ("larch", Tone::Larch),
        "check" => ("page", Tone::Decide),
        _ => ("tasks", Tone::Tool),
    };
    // A running step shows the tree at work in its tile.
    let mark = if state == StepState::Running {
        let bg = tint(th, kind_tone);
        div()
            .size(u(38.))
            .flex_none()
            .rounded(u(9.))
            .flex()
            .items_center()
            .justify_center()
            .bg(bg)
            .child(tree(Tree::new(Mood::Work), 30., 180., s.still(), *th, bg))
    } else {
        tile(th, icon, kind_tone)
    };
    div()
        .id(eid(format_args!("step-{}", node.id)))
        .w(u(step_w(s)))
        .flex()
        .flex_col()
        .gap(u(10.))
        .p(u(14.))
        .rounded(u(13.))
        .bg(if quiet {
            fade(th.card, 0.6)
        } else {
            th.card.into()
        })
        .border_color(border)
        .map(|d| {
            if open || state == StepState::Running {
                d.border_2()
            } else {
                d.border_1()
            }
        })
        .when(quiet && !open, |d| d.border_dashed())
        .when(!quiet, |d| d.shadow_sm())
        .when(open && state == StepState::Running, |d| d.shadow_md())
        .cursor_pointer()
        .on_click(cx.listener(move |s, _, _, cx| {
            s.tasks.step = Some(id.clone());
            cx.notify();
        }))
        .child(
            div()
                .flex()
                .items_center()
                .gap(u(11.))
                .when(quiet, |d| d.opacity(0.75))
                .child(mark)
                .child(
                    div()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap(u(2.))
                        .child(eyebrow(th, step_name(n, node.kind)).text_color(th.ink3))
                        .child(
                            div()
                                .text_size(u(14.))
                                .line_height(u(19.))
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(short(s.tasks.name(&record.id, &node.id), 72)),
                        ),
                ),
        )
        .child(div().flex().child(chip(
            (state == StepState::Running).then_some(color),
            label,
            text,
            wash,
        )))
}

/// The right panel for a run: the open step, what it needs, and the run's activity.
fn step_detail(s: &Ecl, record: &Record, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let mut out = div().flex().flex_col().gap(u(12.));
    let order: Vec<usize> = record.layers().concat();
    if let Some(step) = record.focus_step(s.tasks.step.as_deref()) {
        let n = order
            .iter()
            .position(|i| record.nodes[*i].id == step)
            .map_or(0, |p| p + 1);
        let node = record.nodes.iter().find(|x| x.id == step);
        out = out
            .child(eyebrow(th, step_name(n, node.map_or("step", |x| x.kind))))
            .child(serif(s.tasks.name(&record.id, step).to_owned(), 22.));
        if let Some(node) = node.filter(|x| !x.depends_on.is_empty()) {
            let number = |id: &String| {
                order
                    .iter()
                    .position(|i| &record.nodes[*i].id == id)
                    .map_or_else(|| id.clone(), |p| (p + 1).to_string())
            };
            let uses = node
                .depends_on
                .iter()
                .map(number)
                .collect::<Vec<_>>()
                .join(", ");
            let noun = if node.depends_on.len() == 1 {
                "step"
            } else {
                "steps"
            };
            out = out.child(
                div()
                    .text_size(u(13.5))
                    .text_color(th.ink2)
                    .child(format!("Uses {noun} {uses}")),
            );
        }
        if let Some(t) = s.tasks.timings(&record.id).get(step) {
            let tries = if t.attempts > 1 {
                format!(" · {} tries", t.attempts)
            } else {
                String::new()
            };
            let mut stats = div().flex().flex_wrap().gap(u(8.)).mt(u(2.)).child(chip(
                None,
                format!("Took {}{tries}", duration(t.millis)),
                th.ink2.into(),
                th.sunk.into(),
            ));
            if t.counted {
                stats = stats.child(chip(
                    None,
                    format!(
                        "{} in · {} out",
                        thousands(t.prompt),
                        thousands(t.completion)
                    ),
                    th.ink2.into(),
                    th.sunk.into(),
                ));
            }
            if let Some(rate) = t.rate() {
                stats = stats.child(chip(
                    None,
                    format!("{rate:.1} tok/s out"),
                    th.ink2.into(),
                    th.sunk.into(),
                ));
            }
            out = out.child(stats);
        }
        let body = record
            .results
            .get(step)
            .cloned()
            .or(match record.step_state(step) {
                StepState::Running => Some("Working on it…".into()),
                StepState::Upcoming => Some("Not started yet.".into()),
                StepState::Skipped => Some("A condition took this step off the path.".into()),
                StepState::Stopped => Some("The task ended before this step ran.".into()),
                _ => None,
            });
        if let Some(body) = body {
            out = out.child(
                div()
                    .p(u(16.))
                    .mt(u(4.))
                    .rounded(u(12.))
                    .bg(th.card)
                    .border_1()
                    .border_color(th.line)
                    .child(crate::ui::markdown(th, &s.sel, STEP_ITEM, &body)),
            );
        }
    }
    let now = now_ms() / 1000;
    if let Some(decision) = record
        .decision
        .clone()
        .filter(|_| record.status == "waiting")
    {
        let left = decision.expires_in(now);
        out = out
            .child(div().mt(u(8.)).child(eyebrow(th, "Needs you")))
            .child(div().text_size(u(15.)).child(format!(
                "\"{}\" asks to {}.",
                short(s.tasks.name(&record.id, &decision.step), 90),
                decision.scope
            )))
            .child(
                div()
                    .text_size(u(13.))
                    .text_color(th.ink2)
                    .child(match left {
                        Some(left) => format!("Answer within {} min.", left.div_ceil(60)),
                        None => "This request expired; the step will ask again.".into(),
                    }),
            )
            .child(
                mono(format!("Approves exactly {}", decision.action_digest)).text_color(th.ink3),
            );
        if s.tasks.capabilities.decisions && left.is_some() && !s.task_busy.contains(&record.id) {
            let (yes, no) = (record.clone(), record.clone());
            out = out.child(
                div()
                    .flex()
                    .gap(u(10.))
                    .child(button(
                        th,
                        "task-approve",
                        "Approve",
                        Btn::Primary,
                        cx.listener(move |s, _, _, cx| answer(s, yes.clone(), true, cx)),
                    ))
                    .child(button(
                        th,
                        "task-reject",
                        "Reject",
                        Btn::Ghost,
                        cx.listener(move |s, _, _, cx| answer(s, no.clone(), false, cx)),
                    )),
            );
        }
    }
    if let Some(explanation) = record.recovery_explanation() {
        out = out.child(
            div()
                .mt(u(8.))
                .text_size(u(14.))
                .text_color(th.ink2)
                .child(explanation),
        );
        if s.tasks.capabilities.controls && s.tasks.capabilities.retry_child {
            let r = record.clone();
            out = out
                .child(div().flex().child(button(
                    th,
                    "task-retry-child",
                    "Retry step",
                    Btn::Primary,
                    cx.listener(move |s, _, _, cx| retry_child(s, r.clone(), cx)),
                )))
                .child(
                    div()
                        .text_size(u(12.5))
                        .text_color(th.ink3)
                        .child("Retrying runs the waiting step again and may create a new child."),
                );
        }
    }
    if let Some(reason) = record
        .attempt_failure
        .as_ref()
        .filter(|_| record.status == "waiting" && record.recovery.is_none())
    {
        out = out.child(
            div()
                .mt(u(8.))
                .text_size(u(14.))
                .text_color(th.lose)
                .child(format!("The last attempt failed: {reason}")),
        );
        if record.unresolved_effects {
            out = out.child(
                div()
                    .text_size(u(13.))
                    .text_color(th.ink3)
                    .child("An action's outcome is still being checked; retry once it settles."),
            );
        }
        if s.tasks.capabilities.controls && record.can_retry() {
            let r = record.clone();
            out = out.child(div().flex().child(button(
                th,
                "task-retry",
                "Retry",
                Btn::Primary,
                cx.listener(move |s, _, _, cx| control(s, r.clone(), Control::Resume, cx)),
            )));
        }
    }
    let ended = matches!(record.status.as_str(), "succeeded" | "failed" | "cancelled");
    let unstarted = record.status == "paused" && !s.tasks.user_paused(record);
    // In the canvas's order: a step's text is its instruction, and a planner's ids are not.
    let steps: Vec<String> = order
        .iter()
        .map(|i| &record.nodes[*i])
        .filter(|n| n.kind == "step")
        .map(|n| s.tasks.name(&record.id, &n.id).to_owned())
        .collect();
    let can_edit = (ended || unstarted) && crate::tasks_model::can_author(s.tasks.capabilities);
    let can_chat = ended && (record.result.is_some() || !record.results.is_empty());
    let can_review = s.tasks.to_review(record);
    let can_rerun = ended
        && crate::tasks_model::can_author(s.tasks.capabilities)
        && s.tasks.capabilities.execution;
    if can_edit || can_chat || can_review || can_rerun {
        let (r, c) = (record.clone(), record.id.clone());
        let (done, again) = (record.id.clone(), record.id.clone());
        out = out.child(
            div()
                .flex()
                .flex_wrap()
                .gap(u(10.))
                .mt(u(10.))
                .when(can_review, |d| {
                    d.child(button(
                        th,
                        "task-done",
                        "Mark done",
                        Btn::Primary,
                        cx.listener(move |s, _, _, cx| {
                            mark_done(s, &done);
                            cx.notify();
                        }),
                    ))
                })
                .when(can_rerun, |d| {
                    d.child(button(
                        th,
                        "task-again",
                        "Run again",
                        Btn::Outline,
                        cx.listener(move |s, _, _, cx| run_again(s, again.clone(), cx)),
                    ))
                })
                .when(can_edit, |d| {
                    d.child(button(
                        th,
                        "task-edit",
                        "Edit steps",
                        Btn::Outline,
                        cx.listener(move |s, _, _, cx| {
                            edit(s, &r.goal, &steps, &r.planning, &r.acceptance, None, cx);
                            s.say("Change the steps, then save: it runs as a new task.");
                        }),
                    ))
                })
                .when(can_chat, |d| {
                    d.child(button(
                        th,
                        "task-chat",
                        "Open as chat",
                        Btn::Outline,
                        cx.listener(move |s, _, _, cx| open_as_chat(s, c.clone(), cx)),
                    ))
                }),
        );
    }
    let recent = s.tasks.recent(&record.id);
    if !recent.is_empty() {
        out = out.child(div().mt(u(14.)).child(eyebrow(th, "Activity")));
        for (revision, kind) in recent.iter().rev() {
            out = out.child(
                div()
                    .flex()
                    .gap(u(14.))
                    .text_size(u(14.))
                    .child(mono(format!("#{revision}")).w(u(40.)).text_color(th.ink3))
                    .child(event_label(kind)),
            );
        }
    }
    let mut about = vec![format!(
        "Planning: {}",
        policy(&record.planning).map_or("unknown", |p| larch_app::task::planning_label(&p))
    )];
    let name = |id: &str| (s.mcp.servers.iter().find(|x| x.id == id)).map(|x| x.name.clone());
    about.push(format!(
        "Access: {}",
        crate::logic::access(&record.scope, name, s.st.tasks_web)
    ));
    if !record.acceptance.is_empty() {
        about.push(format!("Done when: {}", record.acceptance.join(", ")));
    }
    out.child(div().mt(u(14.)).child(eyebrow(th, "About")))
        .children(
            about
                .into_iter()
                .map(|line| div().text_size(u(13.5)).text_color(th.ink2).child(line)),
        )
}

/// 12,345: a count with its thousands marked.
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// One occurrence on the calendar; `day` counts from the first day shown.
struct Block {
    schedule: String,
    goal: String,
    day: usize,
    minute: u32,
    state: BlockState,
}

/// Every occurrence of every schedule on the `days` days from `first`, oldest first per schedule.
fn blocks(s: &Ecl, first: NaiveDate, days: i64) -> Vec<Block> {
    let (Some(from), Some(to)) = (
        day_start(&Local, first),
        day_start(&Local, first + chrono::Duration::days(days)),
    ) else {
        return Vec::new();
    };
    let now = Utc::now().timestamp();
    let mut blocks = Vec::new();
    for item in s.tasks.schedules() {
        let Some((rule, _)) = s.task_rules.get(&item.id) else {
            continue;
        };
        let runs = s.tasks.runs_of(&item.id);
        let ran: Vec<i64> = runs
            .iter()
            .filter_map(|r| r.origin.as_ref().map(|(_, t)| *t))
            .collect();
        for at in occurrences(rule, &ran, from, to) {
            let Some(local) = DateTime::from_timestamp(at, 0).map(|t| t.with_timezone(&Local))
            else {
                continue;
            };
            let run = runs
                .iter()
                .find(|r| r.origin.as_ref().is_some_and(|(_, t)| *t == at));
            blocks.push(Block {
                schedule: item.id.clone(),
                goal: item.goal.clone(),
                day: (local.date_naive() - first).num_days().clamp(0, days - 1) as usize,
                minute: local.hour() * 60 + local.minute(),
                state: crate::tasks_model::block_state(
                    item.enabled,
                    at,
                    now,
                    run.map(|r| r.status.as_str()),
                ),
            });
        }
    }
    blocks
}

/// When `day` starts in `tz`: its midnight, or its first minute after a clock change skipped
/// midnight (Chile, Paraguay and Cuba move their clocks at 00:00), which drew no blocks at all.
fn day_start<Tz: TimeZone>(tz: &Tz, day: NaiveDate) -> Option<DateTime<Utc>> {
    (0..24 * 60).find_map(|m| {
        let at = day.and_time(NaiveTime::MIN) + chrono::Duration::minutes(m);
        tz.from_local_datetime(&at)
            .earliest()
            .map(|t| t.with_timezone(&Utc))
    })
}

/// The instants in `[from, to)` the rule gives, and those its runs started at (`ran`), oldest
/// first: a run from before an edit changed the rule still happened, so it stays drawn.
fn occurrences(rule: &Schedule, ran: &[i64], from: DateTime<Utc>, to: DateTime<Utc>) -> Vec<i64> {
    let mut out: Vec<i64> = ran
        .iter()
        .copied()
        .filter(|t| (from.timestamp()..to.timestamp()).contains(t))
        .collect();
    let mut at = from;
    for _ in 0..WEEK_BLOCKS {
        let Ok(Some(next)) = rule.next_at_or_after(at) else {
            break;
        };
        if next >= to {
            break;
        }
        out.push(next.timestamp());
        at = next + chrono::Duration::seconds(1);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// A block's wash, text, edge and whether its edge is dashed (still to come).
fn block_look(th: &Theme, state: BlockState) -> (Hsla, Hsla, Hsla, bool) {
    match state {
        BlockState::Ran => (fade(th.keep, 0.14), th.keep.into(), th.keep.into(), false),
        BlockState::Stopped => (fade(th.lose, 0.12), th.lose.into(), th.lose.into(), false),
        BlockState::Running => (fade(th.sun, 0.2), gold_ink(th), th.sun.into(), false),
        BlockState::Upcoming => (
            gpui::transparent_black(),
            th.accent.into(),
            th.accent.into(),
            true,
        ),
        BlockState::Paused => (
            gpui::transparent_black(),
            th.ink3.into(),
            fade(th.ink3, 0.6),
            true,
        ),
        BlockState::Missed => (
            fade(th.ink3, 0.1),
            th.ink3.into(),
            fade(th.ink3, 0.3),
            false,
        ),
    }
}

/// A block's state in the legend's words.
fn block_word(state: BlockState) -> &'static str {
    match state {
        BlockState::Ran => "Ran",
        BlockState::Stopped => "Stopped",
        BlockState::Running => "Running",
        BlockState::Upcoming => "Coming up",
        BlockState::Paused => "Off",
        BlockState::Missed => "No run",
    }
}

/// The Scheduled tab's center: the occurrences of every schedule on one day, a week, or a week as a list.
fn calendar(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let today = Local::now().date_naive();
    let view = s.tasks.view;
    let monday = today - chrono::Duration::days(i64::from(today.weekday().num_days_from_monday()))
        + chrono::Duration::weeks(i64::from(s.tasks.week));
    let (first, days) = match view {
        CalView::Day => (today + chrono::Duration::days(i64::from(s.tasks.day)), 1),
        _ => (monday, 7),
    };
    let last_day = first + chrono::Duration::days(days - 1);
    let (heading, range) = if view == CalView::Day {
        let heading = match s.tasks.day {
            0 => "Today",
            1 => "Tomorrow",
            -1 => "Yesterday",
            _ => "Day",
        };
        (heading, first.format("%A, %b %-d").to_string())
    } else {
        let heading = match s.tasks.week {
            0 => "This week",
            -1 => "Last week",
            1 => "Next week",
            _ => "Week of",
        };
        let range = if first.month() == last_day.month() {
            format!("{} – {}", first.format("%b %-d"), last_day.day())
        } else {
            format!("{} – {}", first.format("%b %-d"), last_day.format("%b %-d"))
        };
        (heading, range)
    };
    let nav = |id: &'static str, label: &'static str, delta: i32| {
        button(
            th,
            id,
            label,
            Btn::Ghost,
            cx.listener(move |s, _, _, cx| {
                let offset = if s.tasks.view == CalView::Day {
                    &mut s.tasks.day
                } else {
                    &mut s.tasks.week
                };
                *offset = if delta == 0 { 0 } else { *offset + delta };
                cx.notify();
            }),
        )
    };
    let mut views = div().flex().gap(u(2.)).p(u(3.)).rounded(u(10.)).bg(th.sunk);
    for (v, label) in [
        (CalView::Day, "Day"),
        (CalView::Week, "Week"),
        (CalView::List, "List"),
    ] {
        let on = view == v;
        views = views.child(
            div()
                .id(eid(format_args!("view-{label}")))
                .px(u(14.))
                .py(u(6.))
                .rounded(u(8.))
                .text_size(u(13.5))
                .cursor_pointer()
                .when(on, |d| d.bg(th.card).font_weight(FontWeight::SEMIBOLD))
                .text_color(if on { th.ink } else { th.ink2 })
                .on_click(cx.listener(move |s, _, _, cx| {
                    s.tasks.view = v;
                    cx.notify();
                }))
                .child(label),
        );
    }
    let blocks = blocks(s, first, days);
    let mut out = div().flex().flex_col().gap(u(16.)).max_w(u(900.)).child(
        div()
            .flex()
            .flex_wrap()
            .items_end()
            .gap(u(12.))
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .gap(u(6.))
                    .child(eyebrow(th, heading))
                    .child(serif(range, 36.)),
            )
            .child(views)
            .child(
                div()
                    .flex()
                    .gap(u(8.))
                    .child(nav("week-prev", "‹", -1))
                    .child(nav("week-today", "Today", 0))
                    .child(nav("week-next", "›", 1)),
            ),
    );
    let open = s.tasks.current_schedule().map(|x| x.id.clone());
    out = out.child(if view == CalView::List {
        agenda(&blocks, first, open.as_deref(), th, cx)
    } else {
        grid(&blocks, first, days, open.as_deref(), th, cx)
    });
    let key = |color: Hsla, dashed: bool, label: &'static str| {
        div()
            .flex()
            .items_center()
            .gap(u(6.))
            .child(div().size(u(11.)).rounded(u(3.)).map(|d| {
                if dashed {
                    d.border_1().border_dashed().border_color(color)
                } else {
                    d.bg(color)
                }
            }))
            .child(label)
    };
    if view != CalView::List {
        out = out.child(
            div()
                .flex()
                .flex_wrap()
                .gap(u(18.))
                .text_size(u(13.))
                .text_color(th.ink2)
                .child(key(th.keep.into(), false, "Ran"))
                .child(key(th.lose.into(), false, "Stopped"))
                .child(key(th.sun.into(), false, "Running"))
                .child(key(th.accent.into(), true, "Coming up"))
                .child(key(th.ink3.into(), true, "Off"))
                .child(key(fade(th.ink3, 0.3), false, "No run")),
        );
    }
    if s.tasks.schedules().is_empty() {
        out = out.child(
            div()
                .text_color(th.ink2)
                .child("No schedules yet. In New task, choose Repeats, the days and a time."),
        );
    }
    out
}

/// The calendar's time grid: one column a day, each occurrence a block at its time.
fn grid(
    blocks: &[Block],
    first_day: NaiveDate,
    days: i64,
    open: Option<&str>,
    th: &Theme,
    cx: &mut Context<Ecl>,
) -> Div {
    const HOUR: f32 = 36.;
    let first = blocks
        .iter()
        .map(|b| b.minute / 60)
        .min()
        .unwrap_or(8)
        .saturating_sub(1)
        .min(8);
    let last = blocks
        .iter()
        .map(|b| b.minute / 60 + 2)
        .max()
        .unwrap_or(20)
        .clamp(first + 8, 24);
    let hours = (last - first) as f32;
    let now = Local::now();
    let today_col = usize::try_from((now.date_naive() - first_day).num_days())
        .ok()
        .filter(|d| (*d as i64) < days);
    let mut head = div()
        .flex()
        .border_b_1()
        .border_color(th.line)
        .child(div().w(u(56.)).flex_none());
    let mut axis = div().w(u(56.)).flex_none().relative().h_full();
    for h in (first..last).step_by(2).skip(1) {
        let label = match h {
            12 => "12p".to_string(),
            h if h < 12 => format!("{h}a"),
            h => format!("{}p", h - 12),
        };
        axis = axis.child(
            div()
                .absolute()
                .top(u((h - first) as f32 * HOUR - 8.))
                .right(u(10.))
                .text_size(u(12.))
                .text_color(th.ink3)
                .child(label),
        );
    }
    let mut grid = div().flex().h(u(hours * HOUR)).child(axis);
    for day in 0..days as usize {
        let date = first_day + chrono::Duration::days(day as i64);
        let shaded = today_col == Some(day);
        head = head.child(
            div()
                .flex_1()
                .flex()
                .flex_col()
                .items_center()
                .py(u(12.))
                .gap(u(2.))
                .when(shaded, |d| d.bg(fade(th.ink3, 0.08)))
                .child(
                    div()
                        .text_size(u(13.))
                        .text_color(th.ink2)
                        .child(date.format("%a").to_string()),
                )
                .child(serif(date.day().to_string(), 24.).text_color(if shaded {
                    Hsla::from(th.accent)
                } else {
                    th.ink.into()
                })),
        );
        let mut col = div()
            .flex_1()
            .relative()
            .h_full()
            .border_l_1()
            .border_color(th.line)
            .when(shaded, |d| d.bg(fade(th.ink3, 0.08)));
        for h in (first..last).step_by(2).skip(1) {
            col = col.child(
                div()
                    .absolute()
                    .top(u((h - first) as f32 * HOUR))
                    .left_0()
                    .right_0()
                    .h(u(0.))
                    .border_t_1()
                    .border_color(fade(th.line, 0.7)),
            );
        }
        if shaded {
            let minute = now.hour() * 60 + now.minute();
            if minute >= first * 60 && minute < last * 60 {
                let top = (minute as f32 / 60. - first as f32) * HOUR;
                col = col
                    .child(
                        div()
                            .absolute()
                            .top(u(top))
                            .left_0()
                            .right_0()
                            .h(u(2.))
                            .bg(th.sun),
                    )
                    .child(
                        div()
                            .absolute()
                            .top(u(top - 4.))
                            .left(u(-5.))
                            .size(u(10.))
                            .rounded_full()
                            .bg(th.sun),
                    );
            }
        }
        for (n, block) in blocks.iter().enumerate().filter(|(_, b)| b.day == day) {
            let (bg, fg, edge, dashed) = block_look(th, block.state);
            let id = block.schedule.clone();
            let picked = open == Some(block.schedule.as_str());
            let time = NaiveTime::from_hms_opt(block.minute / 60, block.minute % 60, 0)
                .map_or_else(String::new, clock);
            // A week column fits about the goal's first word; the day view has room for all of it.
            let label = if days == 1 {
                block.goal.clone()
            } else {
                block
                    .goal
                    .split([' ', ':'])
                    .next()
                    .unwrap_or_default()
                    .to_owned()
            };
            col = col.child(
                div()
                    .id(eid(format_args!("block-{day}-{n}")))
                    .absolute()
                    .top(u((block.minute as f32 / 60. - first as f32) * HOUR + 2.))
                    .left(u(5.))
                    .right(u(5.))
                    .h(u(46.))
                    .px(u(8.))
                    .py(u(5.))
                    .rounded(u(7.))
                    .bg(bg)
                    .text_color(fg)
                    .overflow_hidden()
                    .cursor_pointer()
                    .map(|d| {
                        if dashed {
                            d.border_1().border_dashed().border_color(edge)
                        } else {
                            d.border_l_4().border_color(edge)
                        }
                    })
                    .when(picked, |d| d.shadow_sm())
                    .on_click(cx.listener(move |s, _, _, cx| {
                        s.tasks.selected = Some(id.clone());
                        cx.notify();
                    }))
                    .child(
                        div()
                            .text_size(u(12.5))
                            .font_weight(FontWeight::SEMIBOLD)
                            .whitespace_nowrap()
                            .overflow_hidden()
                            .child(label),
                    )
                    .child(div().text_size(u(11.5)).opacity(0.85).child(if days == 1 {
                        format!("{time} · {}", block_word(block.state))
                    } else {
                        time
                    })),
            );
        }
        grid = grid.child(col);
    }
    div()
        .rounded(u(16.))
        .border_1()
        .border_color(th.line)
        .bg(th.card)
        .overflow_hidden()
        .child(head)
        .child(grid)
}

/// The List view: the week's occurrences as rows under their day, in time order.
fn agenda(
    blocks: &[Block],
    first_day: NaiveDate,
    open: Option<&str>,
    th: &Theme,
    cx: &mut Context<Ecl>,
) -> Div {
    let today = Local::now().date_naive();
    let mut out = div()
        .flex()
        .flex_col()
        .rounded(u(16.))
        .border_1()
        .border_color(th.line)
        .bg(th.card)
        .overflow_hidden();
    let mut any = false;
    for day in 0..7usize {
        let mut rows: Vec<(usize, &Block)> = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.day == day)
            .collect();
        if rows.is_empty() {
            continue;
        }
        any = true;
        rows.sort_by_key(|(_, b)| b.minute);
        let date = first_day + chrono::Duration::days(day as i64);
        out = out.child(
            div()
                .px(u(20.))
                .py(u(10.))
                .bg(th.sunk)
                .text_size(u(13.))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(if date == today {
                    Hsla::from(th.accent)
                } else {
                    th.ink2.into()
                })
                .child(if date == today {
                    format!("Today · {}", date.format("%a %b %-d"))
                } else {
                    date.format("%A, %b %-d").to_string()
                }),
        );
        for (n, block) in rows {
            let (bg, fg, _, _) = block_look(th, block.state);
            let id = block.schedule.clone();
            let time = NaiveTime::from_hms_opt(block.minute / 60, block.minute % 60, 0)
                .map_or_else(String::new, clock);
            out = out.child(
                div()
                    .id(eid(format_args!("agenda-{day}-{n}")))
                    .flex()
                    .items_center()
                    .gap(u(16.))
                    .px(u(20.))
                    .py(u(12.))
                    .border_t_1()
                    .border_color(th.line)
                    .cursor_pointer()
                    .text_size(u(14.5))
                    .when(open == Some(block.schedule.as_str()), |d| {
                        d.bg(fade(th.accent, 0.08))
                    })
                    .on_click(cx.listener(move |s, _, _, cx| {
                        s.tasks.selected = Some(id.clone());
                        cx.notify();
                    }))
                    .child(mono(time).w(u(80.)).flex_none().text_color(th.ink2))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .truncate()
                            .child(block.goal.clone()),
                    )
                    .child(chip(
                        None,
                        block_word(block.state),
                        fg,
                        if bg.a == 0. { fade(th.ink3, 0.1) } else { bg },
                    )),
            );
        }
    }
    if !any {
        out = out.child(
            div()
                .p(u(20.))
                .text_color(th.ink2)
                .child("Nothing scheduled this week."),
        );
    }
    out
}

/// The right panel for a schedule: its rule, policy, last runs and controls.
fn schedule_detail(s: &Ecl, item: &ScheduledRecord, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let sentence = s
        .task_rules
        .get(&item.id)
        .map_or_else(|| item.schedule.clone(), |(rule, _)| rule_sentence(rule));
    let line = |label: &str, value: AnyElement| {
        div()
            .flex()
            .items_center()
            .gap(u(12.))
            .py(u(12.))
            .border_b_1()
            .border_color(th.line)
            .text_size(u(14.))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .text_color(th.ink2)
                    .child(label.to_owned()),
            )
            .child(value)
    };
    let value = |text: String| {
        div()
            .font_weight(FontWeight::SEMIBOLD)
            .max_w(u(170.))
            .text_right()
            .child(text)
            .into_any_element()
    };
    let busy = s.task_busy.contains(&format!("schedule:{}", item.id));
    let live = s.tasks.capabilities.schedules && !busy;
    let toggle = {
        let on = item.enabled;
        let x = item.clone();
        div()
            .id("schedule-toggle")
            .flex()
            .items_center()
            .gap(u(10.))
            .cursor_pointer()
            .when(live, |d| {
                d.on_click(cx.listener(move |s, _, _, cx| toggle_schedule(s, x.clone(), cx)))
            })
            .child(
                div()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(if on { "On" } else { "Off" }),
            )
            .child(
                div()
                    .w(u(42.))
                    .h(u(24.))
                    .rounded_full()
                    .p(u(3.))
                    .flex()
                    .when(on, |d| d.justify_end())
                    .bg(if on {
                        Hsla::from(th.accent)
                    } else {
                        fade(th.ink3, 0.35)
                    })
                    .child(div().size(u(18.)).rounded_full().bg(gpui::white())),
            )
            .into_any_element()
    };
    let mut out = div()
        .flex()
        .flex_col()
        .gap(u(4.))
        .child(eyebrow(th, "Repeats"))
        .child(serif(item.goal.clone(), 26.).mb(u(10.)))
        .child(serif(sentence, 22.).text_color(th.accent).mb(u(10.)))
        .child(line("Status", toggle))
        .child(line(
            "Next",
            value(next_label(item).trim_start_matches("next ").to_string()),
        ))
        .child(line(
            "If the computer was asleep",
            value(item.catch_up.clone()),
        ))
        .child(line(
            "If the last run is still going",
            value(item.overlap.clone()),
        ))
        .child(line(
            "Steps",
            value(match item.steps.len() {
                0 => "Planned each time".into(),
                1 => "1 step".into(),
                n => format!("{n} steps"),
            }),
        ));
    let runs = s.tasks.runs_of(&item.id);
    if !runs.is_empty() {
        out = out.child(div().mt(u(16.)).mb(u(4.)).child(eyebrow(th, "Last 5 runs")));
        for run in runs.into_iter().take(5) {
            let when = run
                .origin
                .as_ref()
                .and_then(|(_, at)| Utc.timestamp_opt(*at, 0).single())
                .map_or_else(String::new, |t| {
                    t.with_timezone(&Local).format("%a").to_string()
                });
            let color: Hsla = match run.status.as_str() {
                "failed" | "cancelled" => th.lose.into(),
                "succeeded" => th.ink.into(),
                _ => th.ink2.into(),
            };
            let (id, tab) = (
                run.id.clone(),
                if run.category == Category::History && !s.tasks.to_review(run) {
                    Tab::Done
                } else {
                    Tab::Now
                },
            );
            out = out.child(
                div()
                    .id(eid(format_args!("sched-run-{}", run.id)))
                    .flex()
                    .gap(u(14.))
                    .py(u(5.))
                    .text_size(u(14.))
                    .cursor_pointer()
                    .on_click(cx.listener(move |s, _, _, cx| {
                        s.tasks.tab = tab;
                        s.tasks.selected = Some(id.clone());
                        s.tasks.step = None;
                        cx.notify();
                    }))
                    .child(mono(when).w(u(40.)).text_color(th.ink3))
                    .child(div().text_color(color).child(s.tasks.headline(run))),
            );
        }
    }
    if live {
        let run = item.id.clone();
        let (x, gone) = (item.clone(), item.clone());
        let e = item.clone();
        out = out.child(
            div()
                .flex()
                .flex_wrap()
                .gap(u(10.))
                .mt(u(18.))
                .when(
                    crate::tasks_model::can_author(s.tasks.capabilities)
                        && s.tasks.capabilities.execution,
                    |d| {
                        d.child(button(
                            th,
                            "schedule-run",
                            "Run now",
                            Btn::Primary,
                            cx.listener(move |s, _, _, cx| run_now(s, run.clone(), cx)),
                        ))
                    },
                )
                .when(crate::tasks_model::can_author(s.tasks.capabilities), |d| {
                    d.child(button(
                        th,
                        "schedule-edit",
                        "Edit",
                        Btn::Outline,
                        cx.listener(move |s, _, _, cx| {
                            edit(
                                s,
                                &e.goal,
                                &e.steps,
                                &e.planning,
                                &e.acceptance,
                                Some(e.clone()),
                                cx,
                            )
                        }),
                    ))
                })
                .child(button(
                    th,
                    "schedule-pause",
                    if item.enabled { "Turn off" } else { "Turn on" },
                    Btn::Ghost,
                    cx.listener(move |s, _, _, cx| toggle_schedule(s, x.clone(), cx)),
                ))
                .when(s.tasks.capabilities.deletes, |d| {
                    d.child(delete_button(s, th, item.id.clone(), cx, move |s, cx| {
                        delete_schedule(s, gone.clone(), cx)
                    }))
                }),
        );
    } else if busy {
        out = out.child(div().mt(u(18.)).text_color(th.ink3).child("Sending…"));
    }
    out
}

/// The center while composing: a goal, optional steps, how it plans and whether it repeats.
fn composer(s: &Ecl, th: &Theme, cx: &mut Context<Ecl>) -> Div {
    let choice = |id: &'static str, label: &'static str, on: bool, pick: fn(&mut Ecl)| {
        crate::ui::pill(
            th,
            id,
            label,
            on,
            cx.listener(move |s, _, _, cx| {
                pick(s);
                cx.notify();
            }),
        )
    };
    let can_author = crate::tasks_model::can_author(s.tasks.capabilities);
    let editing = s.task_editing.is_some();
    // A schedule the composer cannot set (an interval, a one-off) keeps its own rule.
    let fixed = s
        .task_editing
        .as_ref()
        .and_then(|e| s.task_rules.get(&e.id))
        .map(|(rule, _)| rule)
        .filter(|rule| !matches!(rule, Schedule::Calendar { .. }));
    let field = |label: &str, input: AnyElement| {
        div()
            .flex()
            .flex_col()
            .gap(u(8.))
            .child(
                div()
                    .text_size(u(13.5))
                    .text_color(th.ink2)
                    .child(label.to_owned()),
            )
            .child(input)
    };
    let repeat: AnyElement = match fixed {
        Some(rule) => div()
            .text_color(th.ink2)
            .child(format!(
                "{} · edit the goal and steps here; the rule stays.",
                rule_sentence(rule)
            ))
            .into_any_element(),
        None => {
            let mut days = div().flex().flex_wrap().items_center().gap(u(6.));
            for (i, name) in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                .into_iter()
                .enumerate()
            {
                days = days.child(crate::ui::pill(
                    th,
                    format!("day-{name}"),
                    name,
                    s.task_days & (1 << i) != 0,
                    cx.listener(move |s, _, _, cx| {
                        s.task_days ^= 1 << i;
                        cx.notify();
                    }),
                ));
            }
            let summary = if s.task_days == 0 {
                "Pick at least one day.".to_owned()
            } else {
                larch_app::task::weekdays(s.task_days)
            };
            div()
                .flex()
                .flex_col()
                .gap(u(12.))
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .items_center()
                        .gap(u(8.))
                        .when(!editing, |d| {
                            d.child(choice("repeat-once", "Just once", !s.task_repeat, |s| {
                                s.task_repeat = false
                            }))
                        })
                        .when(s.tasks.capabilities.schedules, |d| {
                            d.child(choice("repeat-days", "Repeats", s.task_repeat, |s| {
                                s.task_repeat = true
                            }))
                        })
                        .when(s.task_repeat, |d| {
                            d.child(div().text_color(th.ink2).child("at"))
                                .child(div().w(u(120.)).child(boxed(th, s.task_time.clone())))
                        }),
                )
                .when(s.task_repeat, |d| {
                    d.child(
                        days.child(div().w(u(10.)))
                            .child(choice(
                                "days-every",
                                "Every day",
                                s.task_days == larch_task::schedule::EVERY_DAY,
                                |s| s.task_days = larch_task::schedule::EVERY_DAY,
                            ))
                            .child(choice(
                                "days-weekdays",
                                "Weekdays",
                                s.task_days == 0b001_1111,
                                |s| s.task_days = 0b001_1111,
                            )),
                    )
                    .child(
                        div()
                            .text_size(u(13.))
                            .text_color(if s.task_days == 0 {
                                Hsla::from(th.lose)
                            } else {
                                th.ink2.into()
                            })
                            .child(summary),
                    )
                })
                .into_any_element()
        }
    };
    let save = if editing {
        "Save changes"
    } else if s.task_repeat {
        "Save schedule"
    } else {
        "Save task"
    };
    let hint = if !can_author {
        "Saving needs a task service with explicit start."
    } else if editing {
        "Runs it already started keep their old steps."
    } else if s.task_repeat {
        "A schedule saves off; turn it on in Scheduled."
    } else {
        "It saves paused; press Start when you're ready."
    };
    let th2 = *th;
    let when = match fixed {
        Some(rule) => rule_sentence(rule),
        None if s.task_repeat && s.task_days != 0 => {
            let days = larch_app::task::weekdays(s.task_days);
            match s.task_time.read(cx).text.trim() {
                "" => days,
                at => format!("{days} at {at}"),
            }
        }
        None if s.task_repeat => "Repeats".into(),
        None => "When you press Start".into(),
    };
    let steps = s.task_steps.len();
    // Ask Larch: the only step a task runs today. Branches, file triggers and notices need the
    // task service to run them first, so the builder offers none.
    let mut flow = div()
        .flex()
        .flex_col()
        .items_center()
        .child(node(th, "clock", Tone::Start, "Start", &when, repeat))
        .child(wire(th, None))
        .child(node(
            th,
            "larch",
            Tone::Larch,
            "Goal",
            "What the whole task is for",
            boxed(th, s.task_goal.clone()),
        ));
    for (i, input) in s.task_steps.iter().enumerate() {
        flow = flow.child(wire(th, Some(add_here(th, i, cx)))).child(
            node(
                th,
                "larch",
                Tone::Larch,
                &format!("Step {} · Ask Larch", i + 1),
                "Larch does this, with what the steps before it found",
                boxed(th, input.clone()),
            )
            .child(
                div()
                    .id(eid(format_args!("step-remove-{i}")))
                    .absolute()
                    .top(u(14.))
                    .right(u(14.))
                    .p(u(4.))
                    .rounded(u(6.))
                    .cursor_pointer()
                    .hover(move |d| d.bg(th2.sunk))
                    .on_click(cx.listener(move |s, _, _, cx| {
                        if i < s.task_steps.len() {
                            s.task_steps.remove(i);
                        }
                        cx.notify();
                    }))
                    .child(crate::ui::icon("x", 14., th.ink3)),
            ),
        );
    }
    flow = flow
        .child(wire(th, Some(add_here(th, steps, cx))))
        .when(steps == 0, |d| {
            d.child(div().text_size(u(13.)).text_color(th.ink3).pb(u(6.)).child(
                if s.task_planning == "disabled" {
                    "Just my steps needs at least one step. Press + to add it."
                } else {
                    "No steps: Larch plans them from the goal. Press + to write your own."
                },
            ))
            .child(wire(th, None))
        })
        .child(node(
            th,
            "page",
            Tone::Decide,
            "Check before it finishes",
            "Optional · one thing per line",
            boxed(th, s.task_done.clone()),
        ));
    let options = card(th)
        .gap(u(16.))
        .p(u(22.))
        .child(field(
            "Planning",
            div()
                .flex()
                .flex_wrap()
                .gap(u(8.))
                .child(choice(
                    "plan-initial",
                    "Plan first",
                    s.task_planning == "initial",
                    |s| s.task_planning = "initial".into(),
                ))
                .child(choice(
                    "plan-adaptive",
                    "Revise as it goes",
                    s.task_planning == "adaptive",
                    |s| s.task_planning = "adaptive".into(),
                ))
                .child(choice(
                    "plan-disabled",
                    "Just my steps",
                    s.task_planning == "disabled",
                    |s| s.task_planning = "disabled".into(),
                ))
                .into_any_element(),
        ))
        .child(field(
            "Web · a task saved while this is on may search and read pages, until it is off",
            {
                let on = s.st.tasks_web;
                crate::ui::pill(
                    th,
                    "task-web",
                    if on {
                        "✓ Let tasks use the web"
                    } else {
                        "Let tasks use the web"
                    },
                    on,
                    cx.listener(move |s, _, _, cx| {
                        s.call(
                            cx,
                            move |c| c.set_tasks_web(!on),
                            move |s, _, cx| {
                                s.st.tasks_web = !on;
                                cx.notify();
                            },
                        )
                    }),
                )
                .into_any_element()
            },
        ));
    div()
        .flex()
        .flex_col()
        .gap(u(18.))
        .max_w(u(680.))
        .mx_auto()
        .child(eyebrow(
            th,
            if editing { "Edit schedule" } else { "New task" },
        ))
        .child(serif("What should it do?", 36.))
        .child(
            div()
                .p(u(24.))
                .rounded(u(16.))
                .border_1()
                .border_color(th.line)
                .bg(th.sunk)
                .child(flow),
        )
        .child(options)
        .child(
            div()
                .flex()
                .items_center()
                .gap(u(10.))
                .when(can_author, |d| {
                    d.child(button(
                        th,
                        "task-save",
                        save,
                        Btn::Primary,
                        cx.listener(|s, _, _, cx| submit(s, cx)),
                    ))
                })
                .child(button(
                    th,
                    "task-cancel-new",
                    "Cancel",
                    Btn::Ghost,
                    cx.listener(|s, _, _, cx| {
                        s.tasks.composing = false;
                        s.task_editing = None;
                        cx.notify();
                    }),
                ))
                .child(
                    div()
                        .min_w_0()
                        .text_size(u(13.))
                        .text_color(th.ink3)
                        .child(hint),
                ),
        )
}

/// What kind of work a flow node does, for its tile's tint.
#[derive(Clone, Copy)]
enum Tone {
    Start,
    Larch,
    Tool,
    Decide,
}

/// The design's pale tints, which never clash with the status colours (gold running, ember needs
/// you, green done).
fn tint(th: &Theme, t: Tone) -> Rgba {
    let (light, dark) = match t {
        Tone::Start => (0xEBD6A6, 0x5A4A26),
        Tone::Larch => (0xF2CDAE, 0x5C3A2A),
        Tone::Tool => (0xD3DCC1, 0x34452F),
        Tone::Decide => (0xDDD3E8, 0x41385A),
    };
    gpui::rgb(if th.dark { dark } else { light })
}

/// A flow node's tile: its print icon on its tint.
fn tile(th: &Theme, icon: &str, t: Tone) -> Div {
    div()
        .size(u(38.))
        .flex_none()
        .rounded(u(9.))
        .flex()
        .items_center()
        .justify_center()
        .bg(tint(th, t))
        .child(crate::ui::print(icon, 20., th.card, th.ink))
}

/// A node of the builder: tile, name and what it does, then its field.
fn node(th: &Theme, icon: &str, t: Tone, title: &str, sub: &str, body: AnyElement) -> Div {
    div()
        .relative()
        .w_full()
        .flex()
        .flex_col()
        .gap(u(12.))
        .p(u(14.))
        .rounded(u(13.))
        .border_1()
        .border_color(th.line)
        .bg(th.card)
        .shadow_sm()
        .child(
            div()
                .flex()
                .items_center()
                .gap(u(11.))
                .pr(u(28.))
                .child(tile(th, icon, t))
                // Its own width, so a narrow page wraps the line rather than cutting it.
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .flex_col()
                        .gap(u(2.))
                        .child(
                            div()
                                .text_size(u(14.))
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(title.to_owned()),
                        )
                        .child(
                            div()
                                .text_size(u(12.5))
                                .text_color(th.ink3)
                                .child(sub.to_owned()),
                        ),
                ),
        )
        .child(body)
}

/// A node's text field, in its well.
fn boxed(th: &Theme, input: Entity<TextInput>) -> AnyElement {
    div()
        .px(u(12.))
        .py(u(9.))
        .rounded(u(9.))
        .border_1()
        .border_color(th.line)
        .bg(th.app)
        .child(input)
        .into_any_element()
}

/// The wire between two nodes, with a + on it when `plus` is given.
fn wire(th: &Theme, plus: Option<Stateful<Div>>) -> Div {
    let line = || {
        div()
            .w(u(0.))
            .h(u(14.))
            .border_l_2()
            .border_color(fade(th.ink3, 0.5))
    };
    div()
        .flex()
        .flex_col()
        .items_center()
        .child(line())
        .children(plus)
        .child(line())
}

/// The + that adds a step at `at`, focused and ready to type in.
fn add_here(th: &Theme, at: usize, cx: &Context<Ecl>) -> Stateful<Div> {
    let th2 = *th;
    div()
        .id(eid(format_args!("step-add-{at}")))
        .size(u(24.))
        .flex()
        .items_center()
        .justify_center()
        .rounded_full()
        .border_1()
        .border_color(th.line)
        .bg(th.card)
        .group("step-add")
        .cursor_pointer()
        .hover(move |d| d.bg(th2.accent).border_color(th2.accent))
        .on_click(cx.listener(move |s, _, w, cx| {
            let i = step_field(s.theme(), "", cx);
            w.focus(&i.focus_handle(cx));
            s.task_steps.insert(at.min(s.task_steps.len()), i);
            cx.notify();
        }))
        .child(
            crate::ui::icon("plus", 12., th.ink2)
                .group_hover("step-add", move |d| d.text_color(th2.on_accent)),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_day_whose_midnight_a_clock_change_skips_starts_at_its_first_minute() {
        let santiago: chrono_tz::Tz = "America/Santiago".parse().unwrap();
        let first = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let midnight = |d: NaiveDate| santiago.from_local_datetime(&d.and_time(NaiveTime::MIN));
        let skipped = (0..366)
            .map(|i| first + chrono::Duration::days(i))
            .find(|d| midnight(*d).earliest().is_none())
            .expect("Santiago moves its clocks at midnight");
        let start = day_start(&santiago, skipped)
            .unwrap()
            .with_timezone(&santiago);
        assert_eq!(
            (start.date_naive(), start.hour(), start.minute()),
            (skipped, 1, 0)
        );
        let day = NaiveDate::from_ymd_opt(2026, 11, 2).unwrap();
        let start = day_start(&santiago, day).unwrap().with_timezone(&santiago);
        assert_eq!((start.date_naive(), start.time()), (day, NaiveTime::MIN));
    }

    #[test]
    fn an_edited_calendar_keeps_its_zone_and_start_on_a_computer_moved_elsewhere() {
        let york: chrono_tz::Tz = "America/New_York".parse().unwrap();
        let start = NaiveDate::from_ymd_opt(2026, 11, 2).unwrap();
        let saved = Schedule::Calendar {
            start,
            time: NaiveTime::from_hms_opt(7, 45, 0).unwrap(),
            timezone: york,
            rule_revision: "2025b".into(),
            days: 0b001_1111,
        };
        let angeles = "America/Los_Angeles".parse().ok();
        let eight = NaiveTime::from_hms_opt(8, 0, 0).unwrap();
        assert_eq!(
            calendar_rule(Some(&saved), angeles, eight, 0b000_0001),
            Some(Schedule::Calendar {
                start,
                time: eight,
                timezone: york,
                rule_revision: "2025b".into(),
                days: 0b000_0001
            })
        );
        // A new rule takes this computer's zone, and needs one.
        assert!(
            matches!(calendar_rule(None, angeles, eight, 1), Some(Schedule::Calendar { timezone, .. }) if Some(timezone) == angeles)
        );
        assert_eq!(calendar_rule(None, None, eight, 1), None);
    }

    #[test]
    fn runs_from_before_an_edit_stay_on_the_calendar() {
        let utc: chrono_tz::Tz = "UTC".parse().unwrap();
        let day = |d| NaiveDate::from_ymd_opt(2026, 11, d).unwrap();
        let at = |d, h| day(d).and_hms_opt(h, 0, 0).unwrap().and_utc();
        // Edited on Friday the 13th: from then, daily at 10:00. It ran Wednesday and Thursday at 9.
        let rule = Schedule::Calendar {
            start: day(13),
            time: NaiveTime::from_hms_opt(10, 0, 0).unwrap(),
            timezone: utc,
            rule_revision: "2025b".into(),
            days: larch_task::schedule::EVERY_DAY,
        };
        let ran = [
            at(11, 9).timestamp(),
            at(12, 9).timestamp(),
            at(13, 10).timestamp(),
        ];
        assert_eq!(
            occurrences(&rule, &ran, at(9, 0), at(16, 0)),
            [at(11, 9), at(12, 9), at(13, 10), at(14, 10), at(15, 10)].map(|t| t.timestamp())
        );
    }
}
