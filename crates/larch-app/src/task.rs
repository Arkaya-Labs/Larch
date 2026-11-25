//! Task-facing projections over the shared workflow model.
//!
//! The projections run nothing: the one task service on the data folder, which the app hosts
//! (`Hosted`) unless another process already runs it, supplies committed states and events
//! from the durable task store.
pub use larch_task::types::{ChildFailure, ChildFailureKind, PlanningPolicy};
use larch_task::types::{EffectStatus, Event, RunState, RunStatus, WorkflowDefinition};
use serde::{Deserialize, Serialize};

/// Stable wire version for this app-facing projection.
pub const TASK_API_VERSION: u32 = 1;

/// A task center section. Scheduling is supplied by the scheduler, not inferred
/// from a run's status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TaskSection {
    /// Ready, running or paused.
    Running,
    /// Waiting on the user: a decision, a failed attempt or a child to recover.
    NeedsYou,
    /// Succeeded, failed or cancelled.
    History,
}

/// One run as the task center shows it, projected from its committed state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskCard {
    pub version: u32,
    pub run_id: String,
    pub revision: u64,
    pub section: TaskSection,
    pub status: RunStatus,
    /// The exact definition pinned to the run, including its goal, optional
    /// initial steps, planning policy, grants, budgets and acceptance.
    pub definition: WorkflowDefinition,
    /// Current validated graph, including approved revisions.
    pub graph: Option<larch_task::graph::Graph>,
    pub result: Option<String>,
    /// Explicitly committed child failure; absent for generic waits.
    #[serde(default)]
    pub child_failure: Option<ChildFailure>,
    /// The exact action waiting for the user's approval, if any.
    #[serde(default)]
    pub pending_decision: Option<PendingDecision>,
    /// Steps a false condition or a skipped dependency took off the path.
    #[serde(default)]
    pub skipped_steps: Vec<String>,
    /// Why the last attempt failed; the core keeps it until the next attempt starts.
    #[serde(default)]
    pub attempt_failure: Option<String>,
    /// The last attempt was cut off (a restart, a model load) with its outcome unknown, so no
    /// failure: the run waits on nothing, and Resume runs the step again.
    #[serde(default)]
    pub interrupted: bool,
    /// An effect is still intended or uncertain, so the core refuses Resume.
    #[serde(default)]
    pub unresolved_effects: bool,
    /// Steps that have started, finished ones included.
    #[serde(default)]
    pub started_steps: Vec<String>,
    /// Steps that finished.
    #[serde(default)]
    pub completed_steps: Vec<String>,
    /// Each finished step's result, cut to its first [`RESULT_PREVIEW`] characters.
    #[serde(default)]
    pub step_results: std::collections::BTreeMap<String, String>,
    /// The schedule and occurrence (RFC 3339) this run was started for.
    #[serde(default)]
    pub schedule_origin: Option<(String, String)>,
}

/// The most of a step's result a card carries: enough to read, not a whole document per step.
pub const RESULT_PREVIEW: usize = 2000;

/// What the user approves: the action digest is shown and answered verbatim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingDecision {
    pub request_id: String,
    pub step: String,
    pub action_digest: String,
    pub scope: String,
    pub expires_at: u64,
}

impl From<&RunState> for TaskCard {
    fn from(run: &RunState) -> Self {
        let section = match run.status {
            RunStatus::Ready | RunStatus::Running | RunStatus::Paused => TaskSection::Running,
            RunStatus::Waiting => TaskSection::NeedsYou,
            RunStatus::Succeeded | RunStatus::Failed | RunStatus::Cancelled => TaskSection::History,
        };
        Self {
            version: TASK_API_VERSION,
            run_id: run.id.clone(),
            revision: run.revision,
            section,
            status: run.status.clone(),
            definition: run.definition.clone(),
            graph: run.graph.clone(),
            result: run.result.clone(),
            child_failure: run.child_failure.clone(),
            pending_decision: run.waiting_decision.as_ref().and_then(|id| {
                let d = run.decisions.get(id)?;
                Some(PendingDecision {
                    request_id: id.clone(),
                    step: d.node_id.clone(),
                    action_digest: d.action_digest.clone(),
                    scope: d.scope.clone(),
                    expires_at: d.expires_at,
                })
            }),
            skipped_steps: run.skipped_nodes.iter().cloned().collect(),
            attempt_failure: run.attempt_failure.clone(),
            interrupted: run.status == RunStatus::Waiting
                && run.attempt_failure.is_none()
                && run.waiting_child.is_none()
                && run.waiting_decision.is_none()
                && run.child_failure.is_none(),
            unresolved_effects: run
                .effects
                .values()
                .any(|e| matches!(e.status, EffectStatus::Intent | EffectStatus::Uncertain)),
            started_steps: run.started_nodes.iter().cloned().collect(),
            completed_steps: run.completed_nodes.iter().cloned().collect(),
            step_results: run
                .node_results
                .iter()
                .map(|(id, result)| (id.clone(), result.chars().take(RESULT_PREVIEW).collect()))
                .collect(),
            schedule_origin: larch_task::schedule::run_origin(&run.id)
                .map(|(id, at)| (id, at.to_rfc3339())),
        }
    }
}

/// The UI can request only run controls here. Driver and effect commands remain
/// internal to the runtime; exposing them would bypass executor mediation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TaskControl {
    Pause,
    Resume,
    Cancel,
}

/// A run control fenced by the revision the user saw.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskCommand {
    pub version: u32,
    pub run_id: String,
    pub command_id: String,
    pub expected_revision: u64,
    pub control: TaskControl,
}

/// A per-run cursor is the last committed event revision seen by the client.
/// The client must retain the cursor only after applying its event page.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskEventPage {
    pub version: u32,
    pub run_id: String,
    pub events: Vec<Event>,
    pub next_cursor: u64,
    pub has_more: bool,
}

/// A goal-only draft may explicitly request planning; predefined and adaptive
/// drafts retain the same definition rather than introducing another schema.
pub fn planning_label(policy: &PlanningPolicy) -> &'static str {
    match policy {
        PlanningPolicy::Disabled => "No planning",
        PlanningPolicy::Initial => "Plan before starting",
        PlanningPolicy::Adaptive => "Plan and revise at allowed boundaries",
    }
}

use larch_task::store::SqliteStore;
use larch_task::types::{Command, CommandEnvelope, CommandReply};

/// Saves paused drafts to, and reads committed records from, the service's durable store. It has
/// no start, control or answer: those go through [`ServiceClient`], so the service's start checks
/// (planner, execution, allowed commands) cannot be skipped.
pub struct TaskApi {
    store: SqliteStore,
}

impl TaskApi {
    /// Wraps an open store.
    pub fn new(store: SqliteStore) -> Self {
        Self { store }
    }

    /// Saves a new run intent paused, using the same run and command IDs on retry.
    /// A service/executor is required to actually run it, and only after an explicit start.
    pub fn create(
        &mut self,
        version: u32,
        command_id: &str,
        run_id: String,
        definition: WorkflowDefinition,
    ) -> Result<CommandReply, String> {
        check_version(version)?;
        let state = RunState::new(run_id, definition)?;
        // The background service shares this database; a Ready intent would be planned and run.
        self.store.create_paused_idempotent(command_id, &state)
    }

    /// The run's card, or None for an unknown run.
    pub fn inspect(&self, run_id: &str) -> Result<Option<TaskCard>, String> {
        Ok(self.store.get(run_id)?.as_ref().map(TaskCard::from))
    }

    /// Every run's card.
    pub fn list(&self) -> Result<Vec<TaskCard>, String> {
        Ok(self.store.list()?.iter().map(TaskCard::from).collect())
    }

    /// Replay committed events after the last cursor, in revision order (`event_page`).
    pub fn events(&self, run_id: &str, cursor: u64, limit: u32) -> Result<TaskEventPage, String> {
        check_limit(limit)?;
        let run = self.store.get(run_id)?.ok_or("unknown run")?;
        if cursor > run.revision {
            return Err("event cursor is ahead of run revision".into());
        }
        let events = self.store.events_after(run_id, cursor, limit + 1)?;
        Ok(event_page(run_id, cursor, limit, events))
    }
}

/// The service answers at most 1000 events, and a page asks for one more than its limit.
fn check_limit(limit: u32) -> Result<(), String> {
    if limit == 0 || limit > 999 {
        return Err("event page limit must be between 1 and 999".into());
    }
    Ok(())
}

/// A page of up to `limit` events from `limit + 1` fetched: the extra one only reports `has_more`.
fn event_page(run_id: &str, cursor: u64, limit: u32, mut events: Vec<Event>) -> TaskEventPage {
    let has_more = events.len() > limit as usize;
    events.truncate(limit as usize);
    let next_cursor = events.last().map_or(cursor, |event| event.revision);
    TaskEventPage {
        version: TASK_API_VERSION,
        run_id: run_id.into(),
        events,
        next_cursor,
        has_more,
    }
}

fn check_version(version: u32) -> Result<(), String> {
    if version != TASK_API_VERSION {
        return Err(format!("unsupported task API version {version}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "larch-task-app-{}-{}.sqlite",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }
    fn definition() -> WorkflowDefinition {
        WorkflowDefinition {
            version: 1,
            goal: "Summarize notes".into(),
            initial_graph: None,
            planning_policy: PlanningPolicy::Initial,
            grants: vec![],
            budgets: BTreeMap::new(),
            acceptance: vec!["Cite sources".into()],
        }
    }
    fn command(id: &str, revision: u64, control: TaskControl) -> TaskCommand {
        TaskCommand {
            version: 1,
            run_id: "r1".into(),
            command_id: id.into(),
            expected_revision: revision,
            control,
        }
    }

    #[test]
    fn weekdays_read_as_people_say_them() {
        assert_eq!(super::weekdays(0x7f), "Every day");
        assert_eq!(super::weekdays(0b001_1111), "Weekdays");
        assert_eq!(super::weekdays(0b110_0000), "Weekends");
        assert_eq!(super::weekdays(0b001_0101), "Mon, Wed, Fri");
    }

    #[test]
    fn wire_contract_keeps_one_workflow_definition() {
        let state = RunState::new("r1".into(), definition()).unwrap();
        let value = serde_json::to_value(TaskCard::from(&state)).unwrap();
        assert_eq!(value["version"], 1);
        assert_eq!(value["runId"], "r1");
        assert_eq!(value["section"], "running");
        assert_eq!(value["definition"]["goal"], "Summarize notes");
        assert_eq!(value["definition"]["planning_policy"], "initial");
        let back: TaskCard = serde_json::from_value(value).unwrap();
        assert_eq!(back.definition, state.definition);
        let request = serde_json::to_value(command("c1", 0, TaskControl::Pause)).unwrap();
        assert_eq!(request["expectedRevision"], 0);
        assert_eq!(request["commandId"], "c1");
    }

    #[test]
    fn a_pending_decision_reaches_the_card_with_its_exact_digest() {
        let mut state = RunState::new("r1".into(), definition()).unwrap();
        state.status = RunStatus::Waiting;
        state.waiting_decision = Some("apply:1".into());
        state.decisions.insert(
            "apply:1".into(),
            larch_task::types::Decision {
                node_id: "deliver".into(),
                action_digest: "sha256:abc".into(),
                scope: "repo".into(),
                expires_at: 9,
                grant_generation: 0,
                approved: None,
            },
        );
        let card = TaskCard::from(&state);
        assert_eq!(card.section, TaskSection::NeedsYou);
        let pending = card.pending_decision.unwrap();
        assert_eq!(
            (pending.step.as_str(), pending.action_digest.as_str()),
            ("deliver", "sha256:abc")
        );
        state.waiting_decision = None;
        assert!(TaskCard::from(&state).pending_decision.is_none());
    }

    #[test]
    fn the_card_carries_step_progress_a_cut_result_and_the_schedule_that_started_it() {
        let mut state = RunState::new(
            "schedule:2:am:2024-01-01T07:45:00.000000000Z".into(),
            definition(),
        )
        .unwrap();
        state.started_nodes.extend(["a".to_owned(), "b".to_owned()]);
        state.completed_nodes.insert("a".into());
        state
            .node_results
            .insert("a".into(), "x".repeat(RESULT_PREVIEW + 5));
        let card = TaskCard::from(&state);
        assert_eq!(
            (card.started_steps, card.completed_steps),
            (vec!["a".into(), "b".into()], vec!["a".into()])
        );
        assert_eq!(card.step_results["a"].chars().count(), RESULT_PREVIEW);
        assert_eq!(
            card.schedule_origin,
            Some(("am".into(), "2024-01-01T07:45:00+00:00".into()))
        );
        assert_eq!(
            TaskCard::from(&RunState::new("r1".into(), definition()).unwrap()).schedule_origin,
            None
        );
    }

    #[test]
    fn only_intended_or_uncertain_effects_block_resume_on_the_card() {
        let mut state = RunState::new("r1".into(), definition()).unwrap();
        let effect = |status| larch_task::types::Effect {
            attempt_id: "a1".into(),
            operation_digest: "sha256:op".into(),
            status,
        };
        state.effects.insert(
            "done".into(),
            effect(EffectStatus::Completed {
                receipt: "r".into(),
            }),
        );
        state
            .effects
            .insert("gone".into(), effect(EffectStatus::ReconciledAbsent));
        assert!(!TaskCard::from(&state).unresolved_effects);
        for status in [EffectStatus::Intent, EffectStatus::Uncertain] {
            state.effects.insert("open".into(), effect(status));
            assert!(TaskCard::from(&state).unresolved_effects);
        }
    }

    #[test]
    fn predefined_and_planned_definitions_share_the_same_contract() {
        let mut predefined = definition();
        predefined.initial_graph = Some(larch_task::graph::Graph::default());
        predefined.planning_policy = PlanningPolicy::Disabled;
        let goal_only = TaskCard::from(&RunState::new("planned".into(), definition()).unwrap());
        let steps = TaskCard::from(&RunState::new("steps".into(), predefined).unwrap());
        for card in [goal_only, steps] {
            let json = serde_json::to_value(&card).unwrap();
            assert!(json["definition"].get("initial_graph").is_some());
            assert!(json["definition"].get("planning_policy").is_some());
            assert_eq!(serde_json::from_value::<TaskCard>(json).unwrap(), card);
        }
    }

    #[test]
    fn service_client_does_not_fall_back_to_a_local_store_when_unavailable() {
        let root = path().with_extension("missing-service");
        assert!(ServiceClient::connect(&root).is_err());
        assert!(!root.exists());
    }

    #[test]
    fn reconnect_replays_committed_events_and_rejects_stale_commands() {
        let db = path();
        {
            let mut api = TaskApi::new(SqliteStore::open(&db).unwrap());
            let first = api.create(1, "create", "r1".into(), definition()).unwrap();
            assert_eq!(first.status, RunStatus::Paused);
            assert_eq!(
                first,
                api.create(1, "create", "r1".into(), definition()).unwrap()
            );
            let mut changed = definition();
            changed.goal = "A different goal".into();
            assert!(api.create(1, "create", "r1".into(), changed).is_err());
            assert!(api.create(2, "new", "r2".into(), definition()).is_err());
            // The service's controls, on the store: TaskApi itself cannot control a run.
            let control = |store: &mut SqliteStore, id: &str, revision: u64, command: Command| {
                let id = id.into();
                store.execute(
                    "r1",
                    &CommandEnvelope {
                        id,
                        expected_revision: revision,
                        command,
                    },
                )
            };
            let reply = control(&mut api.store, "start", 0, Command::Resume).unwrap();
            assert_eq!(
                reply,
                control(&mut api.store, "start", 0, Command::Resume).unwrap()
            );
            assert!(control(&mut api.store, "stale", 0, Command::Cancel).is_err());
            control(&mut api.store, "pause", 1, Command::Pause).unwrap();
        }
        {
            let mut api = TaskApi::new(SqliteStore::open(&db).unwrap());
            assert_eq!(
                api.create(1, "create", "r1".into(), definition())
                    .unwrap()
                    .revision,
                0
            );
            assert_eq!(api.list().unwrap().len(), 1);
            assert_eq!(api.inspect("r1").unwrap().unwrap().revision, 2);
            let first = api.events("r1", 0, 1).unwrap();
            assert!(first.has_more);
            assert_eq!(first.next_cursor, 1);
            let second = api.events("r1", first.next_cursor, 1).unwrap();
            assert!(!second.has_more);
            assert_eq!(second.next_cursor, 2);
            assert!(api.events("r1", 3, 1).is_err());
            assert!(api.events("missing", 0, 1).is_err());
        }
        let _ = std::fs::remove_file(db);
    }

    #[test]
    fn an_older_services_capabilities_offer_no_stats_or_schedule_edits() {
        let old = r#"{"protocol_version":1,"authoring":true,"controls":true,"execution":true,
            "schedules":true,"decisions":true,"autonomous_application":false}"#;
        let caps: ServiceCapabilities = serde_json::from_str(old).unwrap();
        assert!(!caps.step_stats && !caps.schedule_updates);
    }

    #[test]
    fn a_model_step_answers_in_half_the_context_without_thinking() {
        let p = step_params(8192);
        assert_eq!(
            (p.max_tokens, p.thinking, p.thinking_budget),
            (4096, Some(false), None)
        );
    }

    #[test]
    fn steps_are_model_activities_and_control_nodes_have_none() {
        use larch_task::graph::{Graph, Node, NodeKind};
        let node = |id: &str, kind: NodeKind| Node {
            when: None,
            optional: Vec::new(),
            id: id.into(),
            depends_on: vec![],
            kind,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        };
        let condition = NodeKind::Condition {
            input: "a".into(),
            pointer: "/ok".into(),
            equals: serde_json::Value::Bool(true),
        };
        let mut graph = Graph {
            nodes: vec![
                node("Read the notes", NodeKind::Activity),
                node("both", NodeKind::Join),
                node("ok", condition),
            ],
        };
        let steps = step_activities(&graph).unwrap();
        assert_eq!(steps.keys().collect::<Vec<_>>(), ["Read the notes"]);
        let step = &steps["Read the notes"];
        assert_eq!(step.kind, "model");
        assert_eq!(step.payload["instruction"], "Read the notes");
        let mut plan = larch_task::execution::ExecutionPlan {
            graph_revision: 0,
            activities: steps.clone(),
        };
        plan.activities.insert(
            "s2".into(),
            larch_task::execution::ActivityDescriptor {
                kind: "script".into(),
                payload: serde_json::json!({ "code": "x" }),
            },
        );
        assert_eq!(
            step_instructions(&plan),
            BTreeMap::from([("Read the notes".into(), "Read the notes".into())])
        );
        for repeating in [
            NodeKind::Loop { max_iterations: 2 },
            NodeKind::Map { max_items: 2 },
        ] {
            graph.nodes.push(node("again", repeating));
            assert!(step_activities(&graph).is_err());
            graph.nodes.pop();
        }
    }

    #[test]
    fn a_hosted_service_never_takes_another_owners_for_its_own() {
        let dir = std::env::temp_dir().join(format!(
            "larch-task-host-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let other = {
            let dir = dir.clone();
            std::thread::spawn(move || larch_agent::runtime::service::serve(&dir))
        };
        let deadline = std::time::Instant::now() + HOSTING;
        while larch_agent::runtime::ipc::Client::connect(&dir).is_err() {
            assert!(
                std::time::Instant::now() < deadline,
                "the other service did not start"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let mut hosted = Hosted::start(&dir, None, std::sync::Arc::new(Vec::new)).unwrap();
        let error = hosted.ready().unwrap_err();
        assert!(error.contains("already owned"), "{error}");
        drop(hosted);
        let client = larch_agent::runtime::ipc::Client::connect(&dir).expect("left running");
        client.shutdown().unwrap();
        other.join().unwrap().unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_service_another_host_started_after_ours_is_never_taken_for_ours() {
        let dir = std::env::temp_dir().join(format!(
            "larch-task-host-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // Ours is held before it tries the folder's lock, as a thread the scheduler delays is.
        let (release, held) = std::sync::mpsc::channel::<()>();
        let (_tell, told) = std::sync::mpsc::channel();
        let mut hosted = Hosted {
            data: dir.clone(),
            told,
            token: None,
            thread: Some(std::thread::spawn(move || {
                let _ = held.recv();
                Err("already owned".into())
            })),
        };
        let other = {
            let dir = dir.clone();
            std::thread::spawn(move || larch_agent::runtime::service::serve(&dir))
        };
        let deadline = std::time::Instant::now() + HOSTING;
        while larch_agent::runtime::ipc::Client::connect(&dir).is_err() {
            assert!(
                std::time::Instant::now() < deadline,
                "the other service did not start"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(!hosted.answered(), "another service's token is not ours");
        let later = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            drop(release);
        });
        hosted.stop().unwrap();
        later.join().unwrap();
        let client = larch_agent::runtime::ipc::Client::connect(&dir).expect("left running");
        client.shutdown().unwrap();
        other.join().unwrap().unwrap();
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// How long the hosted service has to answer once started, and to end once asked to.
const HOSTING: std::time::Duration = std::time::Duration::from_secs(10);

/// The task service the app hosts on a thread of its own, on its loaded engine when it has one,
/// so tasks run in the app's one process and share the engine's seats with chat. It acts only on
/// the service it started, which it tells from another one by the token its own thread reports.
pub(crate) struct Hosted {
    data: std::path::PathBuf,
    /// The token this service's thread reports once it owns the folder and listens.
    told: std::sync::mpsc::Receiver<String>,
    /// This service's token, once its thread reported it.
    token: Option<String>,
    thread: Option<std::thread::JoinHandle<Result<(), String>>>,
}

/// The token of the service that owns `data`'s task service, if one wrote it.
fn service_token(data: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(data.join("runtime-service").join("token")).ok()
}

impl Hosted {
    /// Starts the service on `data`; with `larch`, model steps and planning run on it.
    pub(crate) fn start(
        data: &std::path::Path,
        larch: Option<larch::Larch>,
        tools: larch_agent::runtime::driver::StepToolSource,
    ) -> Result<Self, String> {
        let dir = data.to_owned();
        let (tell, told) = std::sync::mpsc::channel();
        // A send fails only once this `Hosted` is gone, and nobody is left to tell then.
        let ready: larch_agent::runtime::service::Ready = Box::new(move |token: &str| {
            let _ = tell.send(token.to_owned());
        });
        let thread = std::thread::Builder::new()
            .name("larch-task-host".into())
            .spawn(move || match larch {
                Some(larch) => {
                    let max_context =
                        larch::knob::spec("max_context").map_err(|e| e.to_string())?;
                    let context =
                        larch.knobs().effective().values[max_context.id as usize] as usize;
                    let seats = larch_agent::runtime::inference::SeatAdmission::of(&larch);
                    let params = step_params(context);
                    // One model step at a time: the service runs its steps on one worker.
                    let tool_ctx = larch_agent::tool::ToolCtx {
                        allowed: Vec::new(),
                        data_dir: dir.clone(),
                        source: "workflow".into(),
                    };
                    larch_agent::runtime::service::serve_with_tools(
                        &dir,
                        larch,
                        seats,
                        params,
                        context,
                        1,
                        Some((tools, tool_ctx)),
                        ready,
                    )
                }
                None => larch_agent::runtime::service::serve_notify(&dir, ready),
            })
            .map_err(|e| format!("cannot start the task service: {e}"))?;
        Ok(Self {
            data: data.to_owned(),
            told,
            token: None,
            thread: Some(thread),
        })
    }

    /// The service thread has not ended.
    pub(crate) fn alive(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }

    /// Learns this service's token once its own thread reports it; true then.
    fn answered(&mut self) -> bool {
        if self.token.is_none() {
            self.token = self.told.try_recv().ok();
        }
        self.token.is_some()
    }

    /// Waits for this service to answer; its startup error if it ended instead (another process
    /// already owns the data folder's service, say).
    pub(crate) fn ready(&mut self) -> Result<(), String> {
        let deadline = std::time::Instant::now() + HOSTING;
        loop {
            if !self.alive() {
                return match self.thread.take().map(std::thread::JoinHandle::join) {
                    Some(Ok(Err(error))) => Err(error),
                    Some(Err(_)) => Err("the task service stopped unexpectedly".into()),
                    _ => Err("the task service stopped".into()),
                };
            }
            if self.answered() {
                return Ok(());
            }
            if std::time::Instant::now() > deadline {
                return Err("the task service did not start in time".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    /// Asks this service, and no other, to shut down, and waits for it to end; refused when it
    /// has not ended in time, and it keeps its engine meanwhile.
    pub(crate) fn stop(&mut self) -> Result<(), String> {
        let deadline = std::time::Instant::now() + HOSTING;
        let mut asked = false;
        while self.alive() {
            if !asked
                && self.answered()
                && service_token(&self.data) == self.token
                && let Ok(client) = larch_agent::runtime::ipc::Client::connect(&self.data)
            {
                asked = client.shutdown().is_ok();
            }
            if std::time::Instant::now() > deadline {
                return Err("The task service is still stopping: try again in a moment.".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        // How it ended does not matter once it has: it was asked to.
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        Ok(())
    }
}

impl Drop for Hosted {
    fn drop(&mut self) {
        // The app is going: a service that does not end in time is left to the process's exit.
        let _ = self.stop();
    }
}

/// A model step's parameters on a load of `context` tokens: half of it for the answer (the knob
/// holds the context the load got), so the prompt has the other half. Thinking is off, as for
/// the planner: the block is hidden, unbudgeted it ran a short pitch to 4,096 tokens, and a budget
/// would need its room reserved on top of the answer's.
fn step_params(context: usize) -> larch::GenParams {
    larch::GenParams {
        max_tokens: context as u32 / 2,
        thinking: Some(false),
        class: larch_engine::Class::Background,
        ..Default::default()
    }
}

/// Why `start_steps` refuses a service without model steps.
pub const NO_MODEL_STEPS: &str = "Load a model to start this task.";

/// Each step's instruction in `plan`, by step id; a step without a text one is left out.
pub fn step_instructions(
    plan: &larch_task::execution::ExecutionPlan,
) -> std::collections::BTreeMap<String, String> {
    (plan.activities.iter())
        .filter_map(|(id, a)| {
            Some((
                id.clone(),
                a.payload.get("instruction")?.as_str()?.to_owned(),
            ))
        })
        .collect()
}

/// A model step for each of `graph`'s activity nodes, whose instruction is the node's id: the
/// step's text, as `create_intent` saves it. Join and condition nodes settle in the core without
/// one; a repeating node is refused, since its activity must answer in a shape a step does not.
pub fn step_activities(
    graph: &larch_task::graph::Graph,
) -> Result<std::collections::BTreeMap<String, larch_task::execution::ActivityDescriptor>, String> {
    use larch_task::graph::NodeKind;
    let mut activities = std::collections::BTreeMap::new();
    for node in &graph.nodes {
        match node.kind {
            NodeKind::Activity => {
                let step = larch_task::execution::ActivityDescriptor {
                    kind: "model".into(),
                    payload: serde_json::json!({ "instruction": node.id }),
                };
                activities.insert(node.id.clone(), step);
            }
            NodeKind::Join | NodeKind::Condition { .. } => {}
            NodeKind::Map { .. } | NodeKind::Loop { .. } => {
                return Err("The app cannot start a task with a repeating step.".into());
            }
        }
    }
    Ok(activities)
}

/// A definition whose steps run in the order given, each after the one before, done once it
/// meets each of `acceptance`.
fn steps_definition(
    goal: String,
    steps: Vec<String>,
    acceptance: Vec<String>,
    planning_policy: PlanningPolicy,
    grants: Vec<String>,
) -> Result<WorkflowDefinition, String> {
    use larch_task::graph::{Graph, Node, NodeKind};
    // Named here: the core's graph check reports only an invalid graph.
    if let Some((i, step)) = steps
        .iter()
        .enumerate()
        .find(|(i, s)| steps[..*i].contains(s))
    {
        return Err(format!(
            "Step {} repeats \"{step}\": give each step its own name.",
            i + 1
        ));
    }
    let nodes = steps
        .iter()
        .enumerate()
        .map(|(i, id)| Node {
            when: None,
            optional: Vec::new(),
            id: id.clone(),
            depends_on: if i == 0 {
                vec![]
            } else {
                vec![steps[i - 1].clone()]
            },
            kind: NodeKind::Activity,
            inputs: Default::default(),
            outputs: Default::default(),
        })
        .collect();
    Ok(WorkflowDefinition {
        version: larch_task::types::SCHEMA_VERSION,
        goal,
        initial_graph: if steps.is_empty() {
            None
        } else {
            Some(Graph { nodes })
        },
        planning_policy,
        grants,
        budgets: Default::default(),
        acceptance,
    })
}

/// Authenticated client for the data folder's one task service; it never opens the database.
/// It holds that service's token, so after a model load restarts the hosted service, get a new
/// one from `App::task_service`.
pub struct ServiceClient {
    client: larch_agent::runtime::ipc::Client,
    /// Read once per connection: a restarted service has a new token, so this client fails then.
    capabilities: ServiceCapabilities,
    /// What a workflow this client saves may use (`with_grants`); none unless told.
    grants: Vec<String>,
}

/// What the connected service offers; the service's own type, so a field it adds is never lost.
pub use larch_task::host::Capabilities as ServiceCapabilities;

/// A saved schedule as the task center shows it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScheduledCard {
    pub id: String,
    pub revision: u64,
    pub definition: WorkflowDefinition,
    pub schedule: String,
    /// The rule itself, so the task center can lay its occurrences out on a calendar.
    pub rule: larch_task::schedule::Schedule,
    pub policy: larch_task::schedule::Policy,
    pub next_due: Option<String>,
    pub enabled: bool,
}

/// The weekdays of a calendar schedule (`Schedule::Calendar::days`) as a person says them.
pub fn weekdays(days: u8) -> String {
    const NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    match days {
        larch_task::schedule::EVERY_DAY => "Every day".into(),
        0b001_1111 => "Weekdays".into(),
        0b110_0000 => "Weekends".into(),
        _ => (0..7)
            .filter(|i| days & (1 << i) != 0)
            .map(|i| NAMES[i])
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// Why a service without `deletes` refuses a delete: it outlived an update of the app.
const TOO_OLD_TO_DELETE: &str =
    "This task service is older than the app: restart the app to delete tasks.";

/// Refuses an action the service does not offer before sending it: an older service would answer
/// with a parse error instead.
fn offered(offered: bool, refusal: &str) -> Result<(), String> {
    if offered { Ok(()) } else { Err(refusal.into()) }
}

impl ServiceClient {
    /// Connects to the data folder's running service; never starts one or opens its database.
    // ponytail: two capability round trips, as ipc::Client::connect discards the one it makes;
    // one once it keeps them.
    pub fn connect(data_dir: &std::path::Path) -> Result<Self, String> {
        let client = larch_agent::runtime::ipc::Client::connect(data_dir)?;
        let capabilities = client.capabilities()?;
        Ok(Self {
            client,
            capabilities,
            grants: Vec::new(),
        })
    }

    /// The grants every workflow this client saves takes, such as `mcp.<server id>`: its model
    /// steps may call those servers' read tools.
    pub fn with_grants(self, grants: Vec<String>) -> Self {
        Self { grants, ..self }
    }

    /// The service's capabilities and every run's card.
    pub fn snapshot(&self) -> Result<(ServiceCapabilities, Vec<TaskCard>), String> {
        let cards = self.client.list()?.iter().map(TaskCard::from).collect();
        Ok((self.capabilities.clone(), cards))
    }

    /// Read committed schedules; saving or listing does not enable them.
    pub fn schedules(&self) -> Result<Vec<ScheduledCard>, String> {
        use larch_task::schedule::Schedule;
        Ok(self
            .client
            .list_schedules()?
            .into_iter()
            .map(|record| {
                let schedule = match &record.schedule {
                    Schedule::Once { at } => format!("Once at {}", at.to_rfc3339()),
                    Schedule::Interval { anchor, seconds } => {
                        format!("Every {seconds} seconds from {}", anchor.to_rfc3339())
                    }
                    Schedule::Calendar {
                        time,
                        timezone,
                        days,
                        ..
                    } => format!("{} at {time} ({timezone})", weekdays(*days)),
                };
                ScheduledCard {
                    id: record.id,
                    revision: record.revision,
                    definition: record.definition,
                    schedule,
                    rule: record.schedule,
                    policy: record.policy,
                    next_due: record.next_due.map(|time| time.to_rfc3339()),
                    enabled: record.enabled,
                }
            })
            .collect())
    }

    /// Persist a run intent in the service, paused: it runs only after an explicit start. The
    /// steps run in the order given, each after the one before, as the task center's draft lists them;
    /// it is done once it meets each of `acceptance`.
    pub fn create_intent(
        &self,
        command_id: &str,
        run_id: String,
        goal: String,
        steps: Vec<String>,
        acceptance: Vec<String>,
        planning_policy: PlanningPolicy,
    ) -> Result<CommandReply, String> {
        self.create_run(
            command_id,
            run_id,
            steps_definition(
                goal,
                steps,
                acceptance,
                planning_policy,
                self.grants.clone(),
            )?,
        )
    }

    /// Persist a run of `definition` in the service, paused, as "Run now" does with a schedule's.
    pub fn create_run(
        &self,
        command_id: &str,
        run_id: String,
        definition: WorkflowDefinition,
    ) -> Result<CommandReply, String> {
        offered(
            self.capabilities.authoring,
            "Saving tasks is unavailable on this service.",
        )?;
        self.client
            .create(command_id, &RunState::new(run_id, definition)?)
    }

    /// Save `steps` to repeat on `schedule`, off until enabled; each occurrence runs them as model
    /// steps under the default policy, done once it meets each of `acceptance`.
    pub fn create_schedule(
        &self,
        id: &str,
        schedule: larch_task::schedule::Schedule,
        goal: String,
        steps: Vec<String>,
        acceptance: Vec<String>,
        planning_policy: PlanningPolicy,
    ) -> Result<(), String> {
        offered(
            self.capabilities.schedules && self.capabilities.authoring,
            "Schedules are unavailable on this service.",
        )?;
        let definition = steps_definition(
            goal,
            steps,
            acceptance,
            planning_policy,
            self.grants.clone(),
        )?;
        let graph = definition
            .initial_graph
            .as_ref()
            .ok_or("Add steps to repeat a task.")?;
        let plan = larch_task::execution::ExecutionPlan {
            graph_revision: 0,
            activities: step_activities(graph)?,
        };
        self.client.create_schedule(
            id,
            schedule,
            larch_task::schedule::Policy::default(),
            definition,
            plan,
            chrono::Utc::now(),
        )
    }

    /// Replace a saved schedule's rule, goal, steps and acceptance, fenced by the revision the user saw; its
    /// policy and on/off state stay. Runs it already started keep their steps.
    #[allow(clippy::too_many_arguments)]
    pub fn update_schedule(
        &self,
        id: &str,
        expected_revision: u64,
        schedule: larch_task::schedule::Schedule,
        policy: larch_task::schedule::Policy,
        goal: String,
        steps: Vec<String>,
        acceptance: Vec<String>,
        planning_policy: PlanningPolicy,
    ) -> Result<u64, String> {
        offered(
            self.capabilities.schedules && self.capabilities.authoring,
            "Schedules are unavailable on this service.",
        )?;
        offered(
            self.capabilities.schedule_updates,
            "This task service is older than the app: restart the app to edit schedules.",
        )?;
        // The composer edits the goal, steps, acceptance and planning only: the saved grants and
        // budgets carry over, read at the revision the user edited, which the update fences.
        let saved = (self.client.list_schedules()?.into_iter())
            .find(|s| s.id == id && s.revision == expected_revision)
            .ok_or("unknown schedule or stale revision")?
            .definition;
        let definition = WorkflowDefinition {
            grants: saved.grants,
            budgets: saved.budgets,
            ..steps_definition(goal, steps, acceptance, planning_policy, Vec::new())?
        };
        let graph = definition
            .initial_graph
            .as_ref()
            .ok_or("Add steps to repeat a task.")?;
        let plan = larch_task::execution::ExecutionPlan {
            graph_revision: 0,
            activities: step_activities(graph)?,
        };
        self.client
            .update_schedule(id, expected_revision, schedule, policy, definition, plan)
    }

    /// Delete a saved schedule, fenced by the revision the user saw; refused while a run it
    /// started is unfinished. The runs it started stay.
    pub fn delete_schedule(&self, id: &str, expected_revision: u64) -> Result<(), String> {
        offered(self.capabilities.deletes, TOO_OLD_TO_DELETE)?;
        self.client.delete_schedule(id, expected_revision)
    }

    /// Delete a settled run, with its steps and results.
    pub fn delete_run(&self, run_id: &str) -> Result<(), String> {
        offered(self.capabilities.deletes, TOO_OLD_TO_DELETE)?;
        self.client.delete_run(run_id)
    }

    /// The run itself, results whole.
    pub fn run(&self, run_id: &str) -> Result<Option<RunState>, String> {
        self.client.get(run_id)
    }

    /// Each attempt's wall time and tokens, oldest first: what a step's line shows.
    pub fn step_stats(&self, run_id: &str) -> Result<Vec<larch_task::execution::StepStat>, String> {
        // An older service answers neither this nor `step_instructions`: its runs show without.
        if !self.capabilities.step_stats {
            return Ok(Vec::new());
        }
        self.client.step_stats(run_id)
    }

    /// What each of the run's steps does, by step id: a planned step's id is only a short name.
    pub fn step_instructions(
        &self,
        run_id: &str,
    ) -> Result<std::collections::BTreeMap<String, String>, String> {
        if !self.capabilities.step_stats {
            return Ok(Default::default());
        }
        Ok(self
            .client
            .plan(run_id)?
            .map_or_else(Default::default, |p| step_instructions(&p)))
    }

    /// Turn a saved schedule on or off, fenced by the revision the user saw.
    pub fn enable_schedule(
        &self,
        id: &str,
        expected_revision: u64,
        enabled: bool,
    ) -> Result<u64, String> {
        offered(
            self.capabilities.schedules,
            "Schedules are unavailable on this service.",
        )?;
        self.client.enable_schedule(id, expected_revision, enabled)
    }

    /// Retry a settled failed child through the revision-fenced service command.
    pub fn retry_child(
        &self,
        run_id: &str,
        child_id: &str,
        command_id: &str,
        expected_revision: u64,
    ) -> Result<CommandReply, String> {
        offered(
            self.capabilities.retry_child,
            "Child retry is unavailable on this service.",
        )?;
        self.client.control(
            run_id,
            &CommandEnvelope {
                id: command_id.into(),
                expected_revision,
                command: Command::RetryChild {
                    child_id: child_id.into(),
                },
            },
        )
    }

    /// Approve or reject the pending action whose digest the user was shown.
    pub fn respond_decision(
        &self,
        run_id: &str,
        decision: &PendingDecision,
        approve: bool,
        command_id: &str,
        expected_revision: u64,
    ) -> Result<CommandReply, String> {
        offered(
            self.capabilities.decisions,
            "Approvals are unavailable on this service.",
        )?;
        self.client.respond_decision(
            run_id,
            command_id,
            expected_revision,
            &decision.request_id,
            &decision.action_digest,
            approve,
        )
    }

    /// Explicitly activate or resume a goal-only workflow through the service planner.
    pub fn start_goal(
        &self,
        run_id: &str,
        command_id: &str,
        expected_revision: u64,
    ) -> Result<CommandReply, String> {
        offered(
            self.capabilities.explicit_start,
            "Starting tasks is unavailable on this service.",
        )?;
        self.client
            .start_goal(run_id, command_id, expected_revision)
    }

    /// Start a workflow's own steps (`step_activities`) through the revision-fenced service
    /// command; refused on a service that runs no model steps, since an installed plan is final.
    pub fn start_steps(
        &self,
        run_id: &str,
        command_id: &str,
        expected_revision: u64,
    ) -> Result<CommandReply, String> {
        // The service names no activity kinds, and every host with a model has a planner too.
        let caps = self.client.capabilities()?;
        if !caps.execution || !caps.planning {
            return Err(NO_MODEL_STEPS.into());
        }
        let state = self.client.get(run_id)?.ok_or("unknown task")?;
        let graph = state.graph.as_ref().ok_or("this task has no steps")?;
        let plan = larch_task::execution::ExecutionPlan {
            graph_revision: state.graph_revision,
            activities: step_activities(graph)?,
        };
        self.client
            .start_plan(run_id, &plan, command_id, expected_revision)
    }

    /// Pause, resume or cancel a run through the service's allowed controls.
    pub fn control(&self, request: TaskCommand) -> Result<CommandReply, String> {
        check_version(request.version)?;
        offered(
            self.capabilities.controls,
            "Run controls are unavailable on this service.",
        )?;
        let command = match request.control {
            TaskControl::Pause => Command::Pause,
            TaskControl::Resume => Command::Resume,
            TaskControl::Cancel => Command::Cancel,
        };
        self.client.control(
            &request.run_id,
            &CommandEnvelope {
                id: request.command_id,
                expected_revision: request.expected_revision,
                command,
            },
        )
    }

    /// Replay committed events after the last cursor, paged as [`TaskApi::events`] pages them.
    pub fn events(&self, run_id: &str, cursor: u64, limit: u32) -> Result<TaskEventPage, String> {
        check_limit(limit)?;
        let events = self.client.events(run_id, cursor, limit + 1)?;
        Ok(event_page(run_id, cursor, limit, events))
    }
}
