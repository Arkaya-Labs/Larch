//! Persisted activity plans and bounded, externally executed activity steps.
use crate::{
    runtime::ActorHandle,
    store::SqliteStore,
    types::{Attempt, Command, CommandEnvelope, RunStatus},
};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActivityDescriptor {
    pub kind: String,
    pub payload: serde_json::Value,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub graph_revision: u64,
    pub activities: BTreeMap<String, ActivityDescriptor>,
}
/// An execution-origin child admission, fenced to its live parent attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChildSpawnRequest {
    pub parent_id: String,
    pub expected_revision: u64,
    pub command_id: String,
    pub parent_attempt_id: String,
    pub parent_epoch: u64,
    pub expected_graph_revision: u64,
    pub child: crate::types::RunState,
    pub plan: ExecutionPlan,
}

#[derive(Clone)]
pub struct ActivityRequest {
    pub root_id: String,
    pub run_id: String,
    pub node_id: String,
    pub attempt_id: String,
    pub epoch: u64,
    pub graph_revision: u64,
    pub descriptor: ActivityDescriptor,
    pub inputs: BTreeMap<String, String>,
    pub max_tokens: Option<u64>,
    pub cancellation: Cancellation,
    /// The user's approvals for this node that are still valid; an effect binds one by digest.
    pub approvals: BTreeMap<String, crate::types::Decision>,
}
/// An exact action the driver must not perform until the user approves it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionRequest {
    pub request_id: String,
    pub action_digest: String,
    pub scope: String,
    pub expires_at: u64,
}
/// Unix seconds from the service clock; decisions and approvals compare against it.
pub fn now_secs() -> Result<u64, String> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|e| e.to_string())
}
/// Unix milliseconds from the service clock; retry backoffs compare against it.
fn now_millis() -> Result<u64, String> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .map_err(|e| e.to_string())
}
/// When a retry asked to wait `delay` from now may start, in unix milliseconds.
fn retry_not_before(delay: std::time::Duration) -> Result<u64, String> {
    // Rounded up, plus the millisecond the truncated clock may hide: never shorter than asked.
    let wait = delay
        .as_micros()
        .div_ceil(1000)
        .min(MAX_RETRY_DELAY_MS.into()) as u64;
    Ok(now_millis()? + 1 + wait)
}
/// The longest backoff a retry waits; a driver asking for more gets this.
// ponytail: 1 h ceiling on a retry backoff, raise it when a driver needs to wait longer
const MAX_RETRY_DELAY_MS: u64 = 3_600_000;
/// Trusted, measured usage for an atomic budgeted child handoff.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BudgetedChildSpawnRequest {
    pub spawn: ChildSpawnRequest,
    pub used: u64,
    pub usage_receipt: String,
}
/// Cooperative cancellation observes durable attempt ownership; errors fail closed.
#[derive(Clone)]
pub struct Cancellation {
    actor: ActorHandle,
    run_id: String,
    attempt_id: String,
    epoch: u64,
}
impl Cancellation {
    pub fn is_cancelled(&self) -> bool {
        self.actor
            .get(self.run_id.clone())
            .ok()
            .flatten()
            .is_none_or(|state| {
                state.status != RunStatus::Running
                    || state.attempt.as_ref().is_none_or(|attempt| {
                        attempt.id != self.attempt_id || attempt.epoch != self.epoch
                    })
            })
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct ActivityOutcome {
    pub output: String,
}
/// Wall time and tokens of one attempt, stamped by the executor around the driver call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepStat {
    pub attempt_id: String,
    pub node_id: String,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub finished_at: chrono::DateTime<chrono::Utc>,
    /// Prompt and completion tokens, when the driver counts them.
    pub tokens: Option<(u64, u64)>,
}
/// A driver must reject unknown kinds and mediate all external effects.
pub trait Driver: Send + Sync {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String>;
    /// The activity kinds this driver runs. None (the default) does not enumerate them and
    /// admits any non-empty kind; Some lets the service refuse a plan before it is stored.
    fn registered_kinds(&self) -> Option<BTreeSet<String>> {
        None
    }
    /// Trusted admission estimate and shared capacity. None refuses budgeted dispatch.
    fn suspended_child(
        &self,
        _request: &ActivityRequest,
        _outcome: &Result<ActivityOutcome, String>,
    ) -> Option<String> {
        None
    }
    /// Return a proposed child for trusted, receipt-backed budgeted handoff.
    fn budgeted_child(
        &self,
        _request: &ActivityRequest,
        _outcome: &Result<ActivityOutcome, String>,
    ) -> Option<ChildSpawnRequest> {
        None
    }
    /// An action the activity prepared but must not perform without the user's approval.
    fn decision_request(
        &self,
        _request: &ActivityRequest,
        _outcome: &Result<ActivityOutcome, String>,
    ) -> Option<DecisionRequest> {
        None
    }
    fn enforces_budget(&self) -> bool {
        false
    }
    fn reservation(&self, _request: &ActivityRequest) -> Option<(u64, usize)> {
        None
    }
    /// Measured usage on both success and failure. None retains the reservation.
    fn measured_usage(
        &self,
        _request: &ActivityRequest,
        _outcome: &Result<ActivityOutcome, String>,
    ) -> Option<u64> {
        None
    }
    /// Prompt and completion tokens the attempt used, for display only; called once per
    /// attempt, budgeted or not. None when the driver does not count tokens.
    fn step_tokens(&self, _request: &ActivityRequest) -> Option<(u64, u64)> {
        None
    }
    /// How long to wait before the node runs again after this failed attempt, its `failures`-th
    /// in a row before it (at most an hour); None parks the run for the user, as every failure did
    /// before.
    fn retry_delay(
        &self,
        _request: &ActivityRequest,
        _reason: &str,
        _failures: u32,
    ) -> Option<std::time::Duration> {
        None
    }
    /// Checks the settled run `state` against its acceptance `criterion`: the evidence that it
    /// holds, or why not, which parks the run for the user. None: no check here, and the run
    /// waits for one (a `RecordAcceptance` command, say).
    fn accept(
        &self,
        _state: &crate::types::RunState,
        _criterion: &str,
    ) -> Option<Result<String, String>> {
        None
    }
    /// Trusted evidence for post-cancellation usage reconciliation.
    fn usage_receipt(
        &self,
        _request: &ActivityRequest,
        _outcome: &Result<ActivityOutcome, String>,
    ) -> Option<String> {
        None
    }
}

pub(crate) fn migrate(db: &Connection) -> Result<(), String> {
    db.execute_batch("BEGIN IMMEDIATE")
        .map_err(|e| e.to_string())?;
    let result = (|| {
        let version: u32 = db
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .map_err(|e| e.to_string())?;
        if version > crate::store::STORE_VERSION {
            return Err(format!("unsupported store version {version}"));
        }
        if version < 3 {
            db.execute_batch("CREATE TABLE execution_plans (run_id TEXT PRIMARY KEY REFERENCES runs(id), plan TEXT NOT NULL); CREATE TABLE notification_receipts (outbox_id TEXT PRIMARY KEY REFERENCES task_outbox(id), receipt TEXT NOT NULL); CREATE TABLE schedule_plans (schedule_id TEXT PRIMARY KEY REFERENCES schedules(id), plan TEXT NOT NULL); CREATE TABLE budget_receipts (permit_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), used INTEGER NOT NULL, receipt TEXT NOT NULL); PRAGMA user_version=3;")
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => db.execute_batch("COMMIT").map_err(|e| e.to_string()),
        Err(e) => {
            let _ = db.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}
/// Refuse a plan naming a kind outside `kinds`: an installed plan is immutable, so it would
/// fail every attempt and could never be corrected.
pub fn check_plan_kinds(kinds: &BTreeSet<String>, plan: &ExecutionPlan) -> Result<(), String> {
    match plan
        .activities
        .values()
        .find(|item| !kinds.contains(&item.kind))
    {
        Some(item) => Err(format!(
            "activity kind {} is not registered on this service",
            item.kind
        )),
        None => Ok(()),
    }
}
pub fn validate_plan_for_definition(
    definition: &crate::types::WorkflowDefinition,
    plan: &ExecutionPlan,
) -> Result<(), String> {
    definition.validate()?;
    let graph = definition
        .initial_graph
        .as_ref()
        .ok_or("schedule requires an initial graph")?;
    if plan.graph_revision != 0 {
        return Err("schedule plan requires initial graph revision".into());
    }
    validate_plan_graph(graph, plan)
}
fn validate_plan_graph(graph: &crate::graph::Graph, plan: &ExecutionPlan) -> Result<(), String> {
    crate::graph::validate_graph(graph)?;
    let mut activities = graph.nodes.iter().filter(|node| node.kind.runs_activity());
    if graph.nodes.is_empty()
        || activities.clone().count() != plan.activities.len()
        || activities.any(|node| !plan.activities.contains_key(&node.id))
    {
        return Err("plan must cover the graph exactly".into());
    }
    if plan
        .activities
        .values()
        .any(|item| item.kind.trim().is_empty())
    {
        return Err("empty activity kind".into());
    }
    Ok(())
}
impl SqliteStore {
    pub fn install_execution_plan(
        &mut self,
        run_id: &str,
        plan: &ExecutionPlan,
    ) -> Result<(), String> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| e.to_string())?;
        let old: Option<String> = tx
            .query_row(
                "SELECT plan FROM execution_plans WHERE run_id=?1",
                [run_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        let encoded = serde_json::to_string(plan).map_err(|e| e.to_string())?;
        if let Some(old) = old {
            return if old == encoded {
                Ok(())
            } else {
                Err("execution plan is immutable".into())
            };
        }
        let json: String = tx
            .query_row("SELECT state FROM runs WHERE id=?1", [run_id], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let state: crate::types::RunState =
            serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if !matches!(state.status, RunStatus::Ready | RunStatus::Paused)
            || state.attempt.is_some()
            || state.graph_revision != plan.graph_revision
        {
            return Err("run is not ready for this plan".into());
        }
        let graph = state.graph.as_ref().ok_or("missing graph")?;
        validate_plan_graph(graph, plan)?;
        tx.execute(
            "INSERT INTO execution_plans(run_id,plan) VALUES (?1,?2)",
            params![run_id, encoded],
        )
        .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())
    }
    /// Commit the planner's initial graph and its immutable activity plan together.
    pub fn install_initial_plan(
        &mut self,
        run_id: &str,
        command: &CommandEnvelope,
        plan: &ExecutionPlan,
    ) -> Result<crate::types::CommandReply, String> {
        let Command::PatchGraph {
            graph,
            expected_graph_revision,
            ..
        } = &command.command
        else {
            return Err("expected initial graph patch".into());
        };
        if expected_graph_revision.checked_add(1) != Some(plan.graph_revision) {
            return Err("plan graph revision mismatch".into());
        }
        validate_plan_graph(graph, plan)?;
        let request = serde_json::to_string(&(command, plan)).map_err(|e| e.to_string())?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| e.to_string())?;
        let old: Option<(String, String)> = tx
            .query_row(
                "SELECT request,reply FROM commands WHERE run_id=?1 AND id=?2",
                params![run_id, command.id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        if let Some((old, reply)) = old {
            if old != request {
                return Err("command id reused".into());
            }
            return serde_json::from_str(&reply).map_err(|e| e.to_string());
        }
        let json: String = tx
            .query_row("SELECT state FROM runs WHERE id=?1", [run_id], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let state: crate::types::RunState =
            serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if state.graph.is_some() {
            return Err("initial graph already exists".into());
        }
        let (next, event, reply) = crate::reducer::apply(&state, command)?;
        let revision = i64::try_from(next.revision).map_err(|_| "integer exceeds SQLite range")?;
        tx.execute(
            "UPDATE runs SET revision=?1,state=?2 WHERE id=?3",
            params![
                revision,
                serde_json::to_string(&next).map_err(|e| e.to_string())?,
                run_id
            ],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO events(run_id,revision,event) VALUES (?1,?2,?3)",
            params![
                run_id,
                revision,
                serde_json::to_string(&event).map_err(|e| e.to_string())?
            ],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO commands(run_id,id,request,reply) VALUES (?1,?2,?3,?4)",
            params![
                run_id,
                command.id,
                request,
                serde_json::to_string(&reply).map_err(|e| e.to_string())?
            ],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO execution_plans(run_id,plan) VALUES (?1,?2)",
            params![
                run_id,
                serde_json::to_string(plan).map_err(|e| e.to_string())?
            ],
        )
        .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(reply)
    }
    /// Install a predefined plan and start the paused run in one transaction.
    pub fn start_plan(
        &mut self,
        run_id: &str,
        command_id: &str,
        revision: u64,
        plan: &ExecutionPlan,
    ) -> Result<crate::types::CommandReply, String> {
        if command_id.is_empty() {
            return Err("command id is empty".into());
        }
        let command = CommandEnvelope {
            id: command_id.into(),
            expected_revision: revision,
            command: Command::Resume,
        };
        let request = serde_json::to_string(&(&command, plan)).map_err(|e| e.to_string())?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| e.to_string())?;
        let old: Option<(String, String)> = tx
            .query_row(
                "SELECT request,reply FROM commands WHERE run_id=?1 AND id=?2",
                params![run_id, command_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        if let Some((old, reply)) = old {
            if old != request {
                return Err("command id reused".into());
            }
            return serde_json::from_str(&reply).map_err(|e| e.to_string());
        }
        let json: String = tx
            .query_row("SELECT state FROM runs WHERE id=?1", [run_id], |r| r.get(0))
            .map_err(|e| e.to_string())?;
        let state: crate::types::RunState =
            serde_json::from_str(&json).map_err(|e| e.to_string())?;
        if state.status != RunStatus::Paused
            || state.revision != revision
            || state.graph_revision != plan.graph_revision
        {
            return Err("stale or inactive start request".into());
        }
        let graph = state.graph.as_ref().ok_or("missing graph")?;
        validate_plan_graph(graph, plan)?;
        let encoded = serde_json::to_string(plan).map_err(|e| e.to_string())?;
        let existing: Option<String> = tx
            .query_row(
                "SELECT plan FROM execution_plans WHERE run_id=?1",
                [run_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        if existing.as_ref().is_some_and(|value| value != &encoded) {
            return Err("execution plan is immutable".into());
        }
        let (next, event, reply) = crate::reducer::apply(&state, &command)?;
        let next_revision =
            i64::try_from(next.revision).map_err(|_| "integer exceeds SQLite range")?;
        if existing.is_none() {
            tx.execute(
                "INSERT INTO execution_plans(run_id,plan) VALUES (?1,?2)",
                params![run_id, encoded],
            )
            .map_err(|e| e.to_string())?;
        }
        tx.execute(
            "UPDATE runs SET revision=?1,state=?2 WHERE id=?3",
            params![
                next_revision,
                serde_json::to_string(&next).map_err(|e| e.to_string())?,
                run_id
            ],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO events(run_id,revision,event) VALUES (?1,?2,?3)",
            params![
                run_id,
                next_revision,
                serde_json::to_string(&event).map_err(|e| e.to_string())?
            ],
        )
        .map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO commands(run_id,id,request,reply) VALUES (?1,?2,?3,?4)",
            params![
                run_id,
                command_id,
                request,
                serde_json::to_string(&reply).map_err(|e| e.to_string())?
            ],
        )
        .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(reply)
    }
    pub fn execution_plan(&self, run_id: &str) -> Result<Option<ExecutionPlan>, String> {
        let json: Option<String> = self
            .connection
            .query_row(
                "SELECT plan FROM execution_plans WHERE run_id=?1",
                [run_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        json.map(|value| serde_json::from_str(&value).map_err(|e| e.to_string()))
            .transpose()
    }
    pub fn root_id(&self, run_id: &str) -> Result<String, String> {
        self.connection
            .query_row(
                "SELECT parent_id FROM children WHERE child_id=?1",
                [run_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(|e| e.to_string())
            .map(|parent: Option<String>| parent.unwrap_or_else(|| run_id.to_owned()))
    }
}

pub struct Executor {
    actor: ActorHandle,
    driver: Arc<dyn Driver>,
}
impl Executor {
    pub fn new(actor: ActorHandle, driver: Arc<dyn Driver>) -> Self {
        Self { actor, driver }
    }
    pub fn install_plan(&self, run_id: &str, plan: ExecutionPlan) -> Result<(), String> {
        if let Some(kinds) = self.driver.registered_kinds() {
            check_plan_kinds(&kinds, &plan)?;
        }
        self.actor.install_execution_plan(run_id.to_owned(), plan)
    }
    /// Execute at most one activity or finish one ready run. No activity runs on the store thread.
    /// An error after the attempt is claimed releases it, so the run never stays Running.
    pub fn tick(&self, run_id: &str) -> Result<bool, String> {
        let mut claim = None;
        match (self.tick_claimed(run_id, &mut claim), claim) {
            (Err(error), Some((attempt_id, epoch, budgeted))) => Err(
                match self.release(run_id, &attempt_id, epoch, budgeted, &error) {
                    Ok(()) => error,
                    Err(release) => format!("{error}; releasing {attempt_id} failed: {release}"),
                },
            ),
            (outcome, _) => outcome,
        }
    }
    /// An unbudgeted attempt fails with the error as its reason; a budgeted one recovers to
    /// Waiting with its reservation retained, because its usage is unknown.
    fn release(
        &self,
        run_id: &str,
        attempt_id: &str,
        epoch: u64,
        budgeted: bool,
        error: &str,
    ) -> Result<(), String> {
        let latest = self.actor.get(run_id.to_owned())?.ok_or("unknown run")?;
        if !holds(&latest, attempt_id, epoch) {
            return Ok(());
        }
        let command = if budgeted {
            Command::Recover
        } else {
            Command::FailAttempt {
                attempt_id: attempt_id.to_owned(),
                epoch,
                reason: error.to_owned(),
            }
        };
        self.fenced(run_id, attempt_id, epoch, latest.revision, |revision| {
            self.actor.execute(
                run_id.to_owned(),
                CommandEnvelope {
                    id: format!("release:{attempt_id}"),
                    expected_revision: revision,
                    command: command.clone(),
                },
            )
        })
        .map(drop)
    }
    /// Submit a command for the live attempt at the run's latest revision. A concurrent
    /// revocation moves the revision but leaves the attempt Running; the reducer re-checks
    /// attempt and epoch in the same transaction, so a retry never repeats the activity.
    fn fenced<T>(
        &self,
        run_id: &str,
        attempt_id: &str,
        epoch: u64,
        mut revision: u64,
        submit: impl Fn(u64) -> Result<T, String>,
    ) -> Result<T, String> {
        loop {
            let error = match submit(revision) {
                Ok(value) => return Ok(value),
                Err(error) => error,
            };
            let latest = self.actor.get(run_id.to_owned())?.ok_or("unknown run")?;
            if latest.revision == revision || !holds(&latest, attempt_id, epoch) {
                return Err(error);
            }
            revision = latest.revision;
        }
    }
    fn tick_claimed(
        &self,
        run_id: &str,
        claim: &mut Option<(String, u64, bool)>,
    ) -> Result<bool, String> {
        let scheduled = self.actor.scheduled_start_for_run(run_id.to_owned())?;
        let state = self.actor.get(run_id.to_owned())?.ok_or("unknown run")?;
        if state.status == RunStatus::Waiting
            && let Some(wait) = &state.waiting_child
        {
            if state.child_failure.is_some() {
                self.actor
                    .check_child_wait(run_id.to_owned(), wait.child_id.clone())?;
                return Ok(false);
            }
            let Some(child) = self
                .actor
                .poll_waited_child(run_id.to_owned(), wait.child_id.clone())?
            else {
                return Ok(false);
            };
            if child.status != RunStatus::Succeeded {
                let (kind, reason) = match child.status {
                    RunStatus::Waiting => (
                        crate::types::ChildFailureKind::FailedAttempt,
                        Some(
                            child
                                .attempt_failure
                                .ok_or("missing child failure evidence")?,
                        ),
                    ),
                    RunStatus::Failed => (crate::types::ChildFailureKind::Failed, child.result),
                    RunStatus::Cancelled => {
                        (crate::types::ChildFailureKind::Cancelled, child.result)
                    }
                    _ => return Err("child has no settled outcome".into()),
                };
                self.actor.execute(
                    run_id.to_owned(),
                    CommandEnvelope {
                        id: format!("child-failure:{}:{}", wait.child_id, state.revision),
                        expected_revision: state.revision,
                        command: Command::RecordChildFailure {
                            child_id: wait.child_id.clone(),
                            kind,
                            reason,
                        },
                    },
                )?;
                return Ok(true);
            }
            let result = child.result.ok_or("child result missing")?;
            self.actor.execute(
                run_id.to_owned(),
                CommandEnvelope {
                    id: format!("wake:{}:{}", wait.child_id, state.revision),
                    expected_revision: state.revision,
                    command: Command::WakeChild {
                        child_id: wait.child_id.clone(),
                        result,
                    },
                },
            )?;
            return Ok(true);
        }
        if state.status == RunStatus::Waiting
            && let Some(request_id) = &state.waiting_decision
        {
            let decision = state.decisions.get(request_id).ok_or("unknown decision")?;
            let now = now_secs()?;
            if state.live(decision, now) {
                return Ok(false);
            }
            self.actor.execute(
                run_id.to_owned(),
                CommandEnvelope {
                    id: format!("retire:{request_id}:{}", state.revision),
                    expected_revision: state.revision,
                    command: Command::RetireDecision {
                        request_id: request_id.clone(),
                        now,
                    },
                },
            )?;
            return Ok(true);
        }
        if state.status != RunStatus::Ready {
            return Ok(false);
        }
        // A retried node waits out its backoff; the run stays Ready, so a restart keeps the wait.
        // A wait longer than any backoff means the wall clock stepped back: the retry is due now.
        if let Some(retry) = &state.retry
            && (1..=MAX_RETRY_DELAY_MS + 1)
                .contains(&retry.not_before_ms.saturating_sub(now_millis()?))
        {
            return Ok(false);
        }
        let graph = state.graph.as_ref().ok_or("missing graph")?;
        let plan = self
            .actor
            .execution_plan(run_id.to_owned())?
            .ok_or("missing execution plan")?;
        if plan.graph_revision != state.graph_revision {
            return Err("stale execution plan".into());
        }
        let node = graph.nodes.iter().find(|node| {
            !state.settled(&node.id) && node.depends_on.iter().all(|id| state.settled(id))
        });
        let Some(node) = node else {
            if !state.graph_settled() {
                return Ok(false);
            }
            let open = (state.definition.acceptance.iter())
                .find(|check| !state.acceptance_receipts.contains_key(*check));
            let command = match open {
                None => Command::CompleteRun {
                    result: state.result.clone().unwrap_or_default(),
                },
                Some(criterion) => match self.driver.accept(&state, criterion) {
                    None => return Ok(false),
                    Some(Ok(receipt)) if !receipt.trim().is_empty() => Command::RecordAcceptance {
                        criterion: criterion.clone(),
                        receipt,
                    },
                    // The reducer refuses blank text: a blank verdict still parks the run, saying
                    // what is missing, so the host never asks the same check every tick.
                    Some(verdict) => Command::RejectAcceptance {
                        criterion: criterion.clone(),
                        reason: match verdict {
                            Err(reason) if !reason.trim().is_empty() => reason,
                            Err(_) => "the check gave no reason".into(),
                            Ok(_) => "the check gave no receipt".into(),
                        },
                    },
                },
            };
            self.actor.execute(
                run_id.to_owned(),
                CommandEnvelope {
                    id: format!("finish:{}", state.revision),
                    expected_revision: state.revision,
                    command,
                },
            )?;
            return Ok(true);
        };
        let items = match &node.kind {
            crate::graph::NodeKind::Map { .. } => crate::reducer::map_items(&state, node).ok(),
            _ => None,
        };
        if !node.kind.runs_activity()
            || crate::reducer::must_skip(&state, node)
            || (matches!(&node.kind, crate::graph::NodeKind::Map { .. })
                && items.as_ref().is_none_or(Vec::is_empty))
        {
            self.actor.execute(
                run_id.to_owned(),
                CommandEnvelope {
                    id: format!("settle:{}:{}", node.id, state.revision),
                    expected_revision: state.revision,
                    command: Command::SettleNode {
                        node_id: node.id.clone(),
                    },
                },
            )?;
            return Ok(true);
        }
        let iteration = match &node.kind {
            crate::graph::NodeKind::Activity => None,
            _ => Some(
                u32::try_from(state.partials.get(&node.id).map_or(0, Vec::len))
                    .map_err(|_| "iteration overflow")?,
            ),
        };
        let descriptor = plan
            .activities
            .get(&node.id)
            .cloned()
            .ok_or("missing activity")?;
        let epoch = state
            .last_epoch
            .checked_add(1)
            .ok_or("attempt epoch overflow")?;
        let attempt_id = format!("{}:{}:{}", run_id, node.id, epoch);
        let attempt = Attempt {
            id: attempt_id.clone(),
            node_id: node.id.clone(),
            epoch,
            iteration,
        };
        let mut inputs = node
            .depends_on
            .iter()
            .map(|id| {
                state
                    .node_results
                    .get(id)
                    .cloned()
                    .map(|value| (id.clone(), value))
                    .ok_or_else(|| "missing committed dependency output".to_string())
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        for (name, input) in &node.inputs {
            let upstream = graph
                .nodes
                .iter()
                .find(|source| source.id == input.node)
                .ok_or("missing input source")?;
            if !matches!(input.schema.as_str(), "text" | "json")
                || upstream.outputs.len() != 1
                || upstream.outputs.get(&input.output) != Some(&input.schema)
            {
                return Err("unsupported typed input".into());
            }
            let value = state
                .node_results
                .get(&input.node)
                .ok_or("missing input result")?
                .clone();
            if inputs.insert(name.clone(), value).is_some() && name != &input.node {
                return Err("input name collides with dependency".into());
            }
        }
        // The item or iteration index rides beside the declared inputs, never over one.
        let extra = match (&items, iteration) {
            (Some(items), Some(i)) => Some(("item", items[i as usize].to_string())),
            (None, Some(i)) => Some(("iteration", i.to_string())),
            _ => None,
        };
        if let Some((name, value)) = extra
            && inputs.insert(name.into(), value).is_some()
        {
            return Err(format!("input name collides with the {name} slot"));
        }
        let root_id = self.actor.root_id(run_id.to_owned())?;
        let mut request = ActivityRequest {
            root_id,
            run_id: run_id.to_owned(),
            node_id: node.id.clone(),
            attempt_id: attempt_id.clone(),
            epoch,
            graph_revision: state.graph_revision,
            descriptor,
            inputs,
            max_tokens: None,
            approvals: state.approvals(&node.id, now_secs()?),
            cancellation: Cancellation {
                actor: self.actor.clone(),
                run_id: run_id.to_owned(),
                attempt_id: attempt_id.clone(),
                epoch,
            },
        };
        let budgeted = !state.definition.budgets.is_empty();
        let reservation = if budgeted {
            if !self.driver.enforces_budget() {
                return Err("driver cannot enforce budget".into());
            }
            Some(
                self.driver
                    .reservation(&request)
                    .ok_or("driver lacks budget reservation")?,
            )
        } else {
            None
        };
        request.max_tokens = reservation.map(|(tokens, _)| tokens);
        let start = CommandEnvelope {
            id: format!("start:{attempt_id}"),
            expected_revision: state.revision,
            command: Command::StartAttempt { attempt },
        };
        let (_, claimed) = match (scheduled, reservation) {
            (Some((outbox, false)), Some((reserve, capacity))) => {
                self.actor.claim_budgeted_scheduled_start_once(
                    outbox,
                    run_id.to_owned(),
                    start,
                    reserve,
                    capacity,
                )?
            }
            (Some((outbox, false)), None) => {
                self.actor.claim_scheduled_start_once(outbox, start)?
            }
            (_, Some((reserve, capacity))) => {
                self.actor
                    .dispatch_budgeted_once(run_id.to_owned(), start, reserve, capacity)?
            }
            (_, None) => self.actor.dispatch_once(run_id.to_owned(), start)?,
        };
        if !claimed {
            return Ok(false);
        }
        *claim = Some((attempt_id.clone(), epoch, budgeted));
        let started_at = chrono::Utc::now();
        let outcome = self.driver.execute(request.clone());
        // Saved before the result, in the same store, so a step never finishes without its line.
        self.actor.record_step(
            run_id.to_owned(),
            StepStat {
                attempt_id: attempt_id.clone(),
                node_id: node.id.clone(),
                started_at,
                finished_at: chrono::Utc::now(),
                tokens: self.driver.step_tokens(&request),
            },
        )?;
        let suspended_child = self.driver.suspended_child(&request, &outcome);
        let child_proposal = self.driver.budgeted_child(&request, &outcome);
        let usage = if budgeted {
            self.driver.measured_usage(&request, &outcome)
        } else {
            None
        };
        let receipt = if budgeted {
            self.driver.usage_receipt(&request, &outcome)
        } else {
            None
        };
        let current = self.actor.get(run_id.to_owned())?.ok_or("unknown run")?;
        if budgeted && (child_proposal.is_some() || suspended_child.is_some()) {
            let proposal_result = (|| {
                if outcome.is_err() {
                    return Err("failed activity cannot hand off a child".to_string());
                }
                let proposal = child_proposal.ok_or("budgeted child proposal is missing")?;
                if proposal.parent_id != run_id
                    || proposal.parent_attempt_id != attempt_id
                    || proposal.parent_epoch != epoch
                    || proposal.expected_graph_revision != state.graph_revision
                    || suspended_child
                        .as_ref()
                        .is_some_and(|id| *id != proposal.child.id)
                {
                    return Err("child proposal does not match the active attempt".into());
                }
                let used = usage.ok_or("activity usage is unknown")?;
                let evidence = receipt
                    .clone()
                    .filter(|value| !value.trim().is_empty())
                    .ok_or("activity usage receipt is missing")?;
                let child_id = proposal.child.id.clone();
                let handoff = BudgetedChildSpawnRequest {
                    spawn: proposal,
                    used,
                    usage_receipt: evidence,
                };
                if let Err(error) = self.actor.spawn_child_and_wait_budgeted(handoff.clone()) {
                    let latest = self.actor.get(run_id.to_owned())?.ok_or("unknown run")?;
                    if latest.status == RunStatus::Waiting {
                        self.actor.spawn_child_and_wait_budgeted(handoff)?;
                    } else {
                        return Err(error);
                    }
                }
                let latest = self.actor.get(run_id.to_owned())?.ok_or("unknown run")?;
                let wait = latest.waiting_child.as_ref().ok_or("parent wait missing")?;
                if latest.status != RunStatus::Waiting
                    || wait.child_id != child_id
                    || wait.origin_attempt_id.as_deref() != Some(attempt_id.as_str())
                    || wait.origin_epoch != Some(epoch)
                    || wait.graph_revision != state.graph_revision
                    || wait.node_id != node.id
                {
                    return Err("stale or mismatched child wait".into());
                }
                Ok(())
            })();
            if let Err(error) = proposal_result {
                let latest = self.actor.get(run_id.to_owned())?.ok_or("unknown run")?;
                if latest.status == RunStatus::Running
                    && latest
                        .attempt
                        .as_ref()
                        .is_some_and(|active| active.id == attempt_id && active.epoch == epoch)
                {
                    if let (Some(used), Some(receipt)) = (
                        usage,
                        receipt.clone().filter(|value| !value.trim().is_empty()),
                    ) {
                        let failure = CommandEnvelope {
                            id: format!("handoff-failed:{attempt_id}"),
                            expected_revision: latest.revision,
                            command: Command::FailAttempt {
                                attempt_id: attempt_id.clone(),
                                epoch,
                                reason: "child handoff failed".into(),
                            },
                        };
                        if let Err(first_error) = self.actor.fail_budgeted_with_receipt(
                            run_id.to_owned(),
                            failure.clone(),
                            used,
                            receipt.clone(),
                        ) && self
                            .actor
                            .fail_budgeted_with_receipt(
                                run_id.to_owned(),
                                failure,
                                used,
                                receipt.clone(),
                            )
                            .is_err()
                        {
                            let after = self.actor.get(run_id.to_owned())?.ok_or("unknown run")?;
                            if after.status == RunStatus::Running
                                && after.attempt.as_ref().is_some_and(|active| {
                                    active.id == attempt_id && active.epoch == epoch
                                })
                            {
                                let recover = CommandEnvelope {
                                    id: format!("unknown-usage:{attempt_id}"),
                                    expected_revision: after.revision,
                                    command: Command::Recover,
                                };
                                if let Err(error) = self.actor.execute(run_id.to_owned(), recover) {
                                    let observed =
                                        self.actor.get(run_id.to_owned())?.ok_or("unknown run")?;
                                    if !matches!(
                                        observed.status,
                                        RunStatus::Cancelled | RunStatus::Waiting
                                    ) || observed.attempt.is_some()
                                        || observed.waiting_child.is_some()
                                    {
                                        return Err(error);
                                    }
                                }
                                self.actor.reconcile_budgeted_usage(
                                    run_id.to_owned(),
                                    attempt_id.clone(),
                                    used,
                                    receipt,
                                )?;
                            } else if matches!(
                                after.status,
                                RunStatus::Cancelled | RunStatus::Waiting
                            ) && after.attempt.is_none()
                                && after.waiting_child.is_none()
                            {
                                self.actor.reconcile_budgeted_usage(
                                    run_id.to_owned(),
                                    attempt_id.clone(),
                                    used,
                                    receipt,
                                )?;
                            }
                            return Err(first_error);
                        }
                    } else {
                        self.actor.execute(
                            run_id.to_owned(),
                            CommandEnvelope {
                                id: format!("unknown-usage:{attempt_id}"),
                                expected_revision: latest.revision,
                                command: Command::Recover,
                            },
                        )?;
                    }
                } else if matches!(latest.status, RunStatus::Cancelled | RunStatus::Waiting)
                    && latest.attempt.is_none()
                    && latest.waiting_child.is_none()
                    && let (Some(used), Some(receipt)) = (
                        usage,
                        receipt.clone().filter(|value| !value.trim().is_empty()),
                    )
                {
                    self.actor.reconcile_budgeted_usage(
                        run_id.to_owned(),
                        attempt_id.clone(),
                        used,
                        receipt,
                    )?;
                }
                return Err(error);
            }
            return Ok(true);
        }
        if let Some(decision) = self.driver.decision_request(&request, &outcome) {
            if outcome.is_err() {
                return Err("failed activity cannot request a decision".into());
            }
            let command = Command::RequestDecision {
                attempt_id: attempt_id.clone(),
                epoch,
                request_id: decision.request_id,
                action_digest: decision.action_digest,
                scope: decision.scope,
                expires_at: decision.expires_at,
            };
            self.fenced(run_id, &attempt_id, epoch, current.revision, |revision| {
                self.actor.execute(
                    run_id.to_owned(),
                    CommandEnvelope {
                        id: format!("decision:{attempt_id}"),
                        expected_revision: revision,
                        command: command.clone(),
                    },
                )
            })?;
            return Ok(true);
        }
        if let Some(child_id) = suspended_child {
            if outcome.is_err() {
                return Err("failed activity cannot suspend on child".into());
            }
            if current.status == RunStatus::Waiting {
                let wait = current
                    .waiting_child
                    .as_ref()
                    .ok_or("parent wait missing")?;
                if wait.child_id != child_id
                    || wait.origin_attempt_id.as_deref() != Some(attempt_id.as_str())
                    || wait.origin_epoch != Some(epoch)
                    || wait.graph_revision != state.graph_revision
                    || wait.node_id != node.id
                {
                    return Err("stale or mismatched child wait".into());
                }
                return Ok(true);
            }
            let command = Command::WaitForChild {
                attempt_id: attempt_id.clone(),
                epoch,
                child_id,
            };
            self.fenced(run_id, &attempt_id, epoch, current.revision, |revision| {
                self.actor.execute(
                    run_id.to_owned(),
                    CommandEnvelope {
                        id: format!("wait:{attempt_id}"),
                        expected_revision: revision,
                        command: command.clone(),
                    },
                )
            })?;
            return Ok(true);
        }
        if budgeted && matches!(current.status, RunStatus::Cancelled | RunStatus::Waiting) {
            if let (Some(used), Some(receipt)) = (usage, receipt.clone()) {
                self.actor.reconcile_budgeted_usage(
                    run_id.to_owned(),
                    attempt_id,
                    used,
                    receipt,
                )?;
                return Ok(true);
            }
            return Err("stopped activity usage is unresolved; reservation retained".into());
        }
        if budgeted && usage.is_none() {
            self.fenced(run_id, &attempt_id, epoch, current.revision, |revision| {
                self.actor.execute(
                    run_id.to_owned(),
                    CommandEnvelope {
                        id: format!("unknown-usage:{attempt_id}"),
                        expected_revision: revision,
                        command: Command::Recover,
                    },
                )
            })?;
            return Err("activity usage is unknown; reservation retained".into());
        }
        let command = match outcome {
            Ok(output) if output.output.len() <= 1_048_576 => Command::FinishAttempt {
                attempt_id: attempt_id.clone(),
                epoch,
                result: output.output,
            },
            Ok(_) => Command::FailAttempt {
                attempt_id: attempt_id.clone(),
                epoch,
                reason: "activity output too large".into(),
            },
            Err(reason) => {
                let failures = (state.retry.as_ref())
                    .filter(|r| r.node_id == node.id)
                    .map_or(0, |r| r.failures);
                // ponytail: unbudgeted only, as RetryAttempt; budgeted failures park as before
                match (self.driver.retry_delay(&request, &reason, failures))
                    .filter(|_| usage.is_none() && state.definition.budgets.is_empty())
                {
                    Some(delay) => Command::RetryAttempt {
                        attempt_id: attempt_id.clone(),
                        epoch,
                        reason,
                        not_before_ms: retry_not_before(delay)?,
                    },
                    None => Command::FailAttempt {
                        attempt_id: attempt_id.clone(),
                        epoch,
                        reason,
                    },
                }
            }
        };
        let completion = |revision| CommandEnvelope {
            id: format!("result:{attempt_id}"),
            expected_revision: revision,
            command: command.clone(),
        };
        if let Some(used) = usage {
            if let Err(error) =
                self.fenced(run_id, &attempt_id, epoch, current.revision, |revision| {
                    self.actor
                        .settle_budgeted(run_id.to_owned(), completion(revision), used)
                })
            {
                let latest = self.actor.get(run_id.to_owned())?.ok_or("unknown run")?;
                if matches!(latest.status, RunStatus::Cancelled | RunStatus::Waiting)
                    && let Some(receipt) = receipt
                {
                    self.actor.reconcile_budgeted_usage(
                        run_id.to_owned(),
                        attempt_id,
                        used,
                        receipt,
                    )?;
                    return Ok(true);
                }
                return Err(error);
            }
        } else {
            self.fenced(run_id, &attempt_id, epoch, current.revision, |revision| {
                self.actor.execute(run_id.to_owned(), completion(revision))
            })?;
        }
        Ok(true)
    }
}
/// Whether `state` is Running the attempt `attempt_id` at `epoch`.
fn holds(state: &crate::types::RunState, attempt_id: &str, epoch: u64) -> bool {
    state.status == RunStatus::Running
        && state
            .attempt
            .as_ref()
            .is_some_and(|attempt| attempt.id == attempt_id && attempt.epoch == epoch)
}
