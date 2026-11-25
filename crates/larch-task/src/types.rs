use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::graph::Graph;

pub const SCHEMA_VERSION: u32 = 1;

/// The same definition describes both goal-only and predefined workflows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    pub version: u32,
    pub goal: String,
    #[serde(default)]
    pub initial_graph: Option<Graph>,
    #[serde(default)]
    pub planning_policy: PlanningPolicy,
    #[serde(default)]
    pub grants: Vec<String>,
    #[serde(default)]
    pub budgets: BTreeMap<String, u64>,
    #[serde(default)]
    pub acceptance: Vec<String>,
}

impl WorkflowDefinition {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != SCHEMA_VERSION {
            return Err(format!("unsupported workflow version {}", self.version));
        }
        if self.goal.trim().is_empty() {
            return Err("workflow goal is empty".into());
        }
        if let Some(graph) = &self.initial_graph {
            crate::graph::validate_graph(graph)?;
        } else if self.planning_policy == PlanningPolicy::Disabled {
            return Err("goal-only workflow requires planning".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanningPolicy {
    #[default]
    Disabled,
    Initial,
    Adaptive,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Ready,
    Running,
    Paused,
    Waiting,
    Succeeded,
    Failed,
    Cancelled,
}
impl RunStatus {
    pub fn terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub id: String,
    pub node_id: String,
    pub epoch: u64,
    /// Map item or loop iteration this attempt runs; None for a plain activity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iteration: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectStatus {
    Intent,
    Uncertain,
    ReconciledAbsent,
    Completed { receipt: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Effect {
    pub attempt_id: String,
    pub operation_digest: String,
    pub status: EffectStatus,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildWait {
    pub child_id: String,
    pub node_id: String,
    pub graph_revision: u64,
    #[serde(default)]
    pub origin_attempt_id: Option<String>,
    #[serde(default)]
    pub origin_epoch: Option<u64>,
}
/// A child failure is persisted evidence; its reason is untrusted data.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChildFailureKind {
    FailedAttempt,
    Failed,
    Cancelled,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildFailure {
    pub child_id: String,
    pub kind: ChildFailureKind,
    pub reason: Option<String>,
}
/// A user decision bound to one exact action digest. Only the user's response answers it:
/// child reports, model output and memory never do.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decision {
    pub node_id: String,
    pub action_digest: String,
    /// What the action touches, shown to the user beside the digest.
    pub scope: String,
    /// Unix seconds; a response at or after it is refused.
    pub expires_at: u64,
    /// The grant generation at request time; any revocation since, on this run or on its root
    /// (which moves its children's generation too), invalidates the decision.
    pub grant_generation: u64,
    /// The user's answer; stays None on a request retired unanswered after it lapsed.
    #[serde(default)]
    pub approved: Option<bool>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunState {
    pub version: u32,
    pub id: String,
    pub definition: WorkflowDefinition,
    pub revision: u64,
    pub status: RunStatus,
    pub attempt: Option<Attempt>,
    #[serde(default)]
    pub waiting_child: Option<ChildWait>,
    #[serde(default)]
    pub child_failure: Option<ChildFailure>,
    /// The decision request this run waits on, if any.
    #[serde(default)]
    pub waiting_decision: Option<String>,
    #[serde(default)]
    pub decisions: BTreeMap<String, Decision>,
    #[serde(default)]
    pub attempt_failure: Option<String>,
    #[serde(default)]
    pub graph_revision: u64,
    #[serde(default)]
    pub graph: Option<Graph>,
    #[serde(default)]
    pub started_nodes: BTreeSet<String>,
    #[serde(default)]
    pub completed_nodes: BTreeSet<String>,
    /// Nodes a false guard or a skipped required dependency took off the path; settled, no result.
    #[serde(default)]
    pub skipped_nodes: BTreeSet<String>,
    /// Committed per-item (map) or per-iteration (loop) results of an unfinished node.
    #[serde(default)]
    pub partials: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub node_results: BTreeMap<String, String>,
    #[serde(default)]
    pub acceptance_receipts: BTreeMap<String, String>,
    #[serde(default)]
    pub grant_generation: u64,
    pub last_epoch: u64,
    #[serde(default)]
    pub used_attempt_ids: BTreeSet<String>,
    pub effects: BTreeMap<String, Effect>,
    pub result: Option<String>,
    /// The node a failed attempt will run again, and not before when.
    #[serde(default)]
    pub retry: Option<Retry>,
}
/// A node whose attempt failed in a way its driver retries: how often, and when it may run again.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Retry {
    pub node_id: String,
    /// Failed attempts of this node in a row.
    pub failures: u32,
    /// Unix milliseconds; the executor starts no attempt before it.
    pub not_before_ms: u64,
}
impl RunState {
    /// Completed or skipped: the node will not run again in this graph revision.
    pub fn settled(&self, node: &str) -> bool {
        self.completed_nodes.contains(node) || self.skipped_nodes.contains(node)
    }
    /// Approved decisions for `node` still valid at `now` under the current grant generation.
    pub fn approvals(&self, node: &str, now: u64) -> BTreeMap<String, Decision> {
        self.decisions
            .iter()
            .filter(|(_, d)| d.node_id == node && d.approved == Some(true) && self.live(d, now))
            .map(|(id, d)| (id.clone(), d.clone()))
            .collect()
    }
    /// Not expired at `now` and requested under the current grant generation.
    pub(crate) fn live(&self, decision: &Decision, now: u64) -> bool {
        now < decision.expires_at && decision.grant_generation == self.grant_generation
    }
    /// Every node of a non-empty graph is settled.
    pub fn graph_settled(&self) -> bool {
        self.graph.as_ref().is_some_and(|graph| {
            !graph.nodes.is_empty() && graph.nodes.iter().all(|node| self.settled(&node.id))
        })
    }
    /// Exactly what `new` builds for this id and definition: nothing started, decided or
    /// recorded, whatever fields the state grows.
    pub fn is_pristine(&self) -> bool {
        Self::new(self.id.clone(), self.definition.clone()).is_ok_and(|fresh| fresh == *self)
    }
    pub fn new(id: String, definition: WorkflowDefinition) -> Result<Self, String> {
        definition.validate()?;
        if id.is_empty() {
            return Err("run id is empty".into());
        }
        let graph = definition.initial_graph.clone();
        Ok(Self {
            version: SCHEMA_VERSION,
            id,
            definition,
            graph,
            graph_revision: 0,
            started_nodes: BTreeSet::new(),
            completed_nodes: BTreeSet::new(),
            skipped_nodes: BTreeSet::new(),
            partials: BTreeMap::new(),
            node_results: BTreeMap::new(),
            acceptance_receipts: BTreeMap::new(),
            revision: 0,
            status: RunStatus::Ready,
            attempt: None,
            waiting_child: None,
            child_failure: None,
            waiting_decision: None,
            decisions: BTreeMap::new(),
            attempt_failure: None,
            last_epoch: 0,
            grant_generation: 0,
            used_attempt_ids: BTreeSet::new(),
            effects: BTreeMap::new(),
            result: None,
            retry: None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Command {
    PatchGraph {
        graph: Graph,
        expected_graph_revision: u64,
        grants: Vec<String>,
        acceptance: Vec<String>,
    },
    RevokeGrant {
        grant: String,
    },
    WaitForChild {
        attempt_id: String,
        epoch: u64,
        child_id: String,
    },
    WakeChild {
        child_id: String,
        result: String,
    },
    RecordChildFailure {
        child_id: String,
        kind: ChildFailureKind,
        reason: Option<String>,
    },
    RetryChild {
        child_id: String,
    },
    /// Suspend the active attempt until the user answers one exact action.
    RequestDecision {
        attempt_id: String,
        epoch: u64,
        request_id: String,
        action_digest: String,
        scope: String,
        expires_at: u64,
    },
    /// The user's answer; `now` is stamped by the trusted host, never by the client.
    RespondDecision {
        request_id: String,
        action_digest: String,
        approve: bool,
        now: u64,
    },
    /// Retire a pending request that expired or lost its grants, unanswered, so the node asks
    /// again; `now` is stamped by the trusted executor.
    RetireDecision {
        request_id: String,
        now: u64,
    },
    /// Settle the next ready control step (skip, join or condition) from stored results.
    SettleNode {
        node_id: String,
    },
    StartAttempt {
        attempt: Attempt,
    },
    FinishAttempt {
        attempt_id: String,
        epoch: u64,
        result: String,
    },
    FailAttempt {
        attempt_id: String,
        epoch: u64,
        reason: String,
    },
    /// A failure the driver retries: the run goes back to Ready for `not_before_ms` (unix ms),
    /// unless the attempt left an effect unresolved, which parks it as `FailAttempt` would.
    RetryAttempt {
        attempt_id: String,
        epoch: u64,
        reason: String,
        not_before_ms: u64,
    },
    RecordIntent {
        effect_id: String,
        attempt_id: String,
        epoch: u64,
        operation_digest: String,
    },
    RecordReceipt {
        effect_id: String,
        attempt_id: String,
        epoch: u64,
        receipt: String,
    },
    ReconcileEffect {
        effect_id: String,
        receipt: Option<String>,
    },
    /// A host check found the settled run does not meet `criterion`: it waits for the user,
    /// saying why, and a resume checks again.
    RejectAcceptance {
        criterion: String,
        reason: String,
    },
    RecordAcceptance {
        criterion: String,
        receipt: String,
    },
    CompleteRun {
        result: String,
    },
    Recover,
    Pause,
    Resume,
    Cancel,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandEnvelope {
    pub id: String,
    pub expected_revision: u64,
    pub command: Command,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub revision: u64,
    pub command_id: String,
    pub kind: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandReply {
    pub revision: u64,
    pub status: RunStatus,
}
