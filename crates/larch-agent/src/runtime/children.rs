//! Child commands are mediated by the workflow actor, which must persist and fence them.
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Default maximum simultaneously admitted children per root.
pub const DEFAULT_CHILD_CAP: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildRequest {
    pub child_id: String,
    pub root_id: String,
    pub graph_revision: u64,
    pub depth: u8,
    pub goal: String,
    pub grants: BTreeSet<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildResult {
    pub child_id: String,
    pub graph_revision: u64,
    pub summary: String,
    pub evidence: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChildCommand {
    Spawn {
        request: ChildRequest,
    },
    Inspect {
        child_id: String,
    },
    Await {
        child_id: String,
    },
    Message {
        child_id: String,
        message_id: String,
        text: String,
    },
    Cancel {
        child_id: String,
    },
    Collect {
        child_id: String,
        graph_revision: u64,
    },
}
/// Called by the actor while deciding admission, before committing Spawn. The actor must make
/// the count and insertion atomic, inherit the root budget, and persist commands and messages.
pub fn validate_spawn(
    request: &ChildRequest,
    parent_root: &str,
    parent_depth: u8,
    live_children: usize,
    cap: usize,
    parent_grants: &BTreeSet<String>,
) -> Result<(), &'static str> {
    if parent_root.trim().is_empty() || request.root_id != parent_root {
        return Err("child root does not match parent");
    }
    if parent_depth != 0 || request.depth != 1 {
        return Err("only depth-one children are supported");
    }
    if cap == 0 || live_children >= cap {
        return Err("child limit reached");
    }
    if request.child_id.trim().is_empty()
        || request.root_id.trim().is_empty()
        || request.goal.trim().is_empty()
    {
        return Err("missing child identity or goal");
    }
    if !request.grants.is_subset(parent_grants) {
        return Err("child authority exceeds parent");
    }
    Ok(())
}
pub fn validate_result(
    result: &ChildResult,
    child_id: &str,
    revision: u64,
) -> Result<(), &'static str> {
    if result.child_id != child_id || result.graph_revision != revision {
        return Err("stale or mismatched child result");
    }
    if result.summary.trim().is_empty()
        || result.evidence.is_empty()
        || result.evidence.iter().any(|x| x.trim().is_empty())
    {
        return Err("child result requires summary and evidence");
    }
    Ok(())
}

/// Commit a child through the single task actor. The parent run revision fences the preceding
/// snapshot against concurrent changes; the store rechecks depth, count and grant subset inside
/// its transaction. Children with budgets remain blocked until root accounting is connected.
pub fn spawn(
    actor: &larch_task::runtime::ActorHandle,
    parent_id: &str,
    parent_revision: u64,
    command_id: String,
    request: &ChildRequest,
    child: larch_task::types::RunState,
) -> Result<larch_task::types::CommandReply, String> {
    let parent = actor.get(parent_id.to_string())?.ok_or("unknown parent")?;
    if parent.revision == parent_revision && parent.graph_revision != request.graph_revision {
        return Err("stale graph revision".into());
    }
    let parent_grants = parent.definition.grants.iter().cloned().collect();
    validate_spawn(request, parent_id, 0, 0, DEFAULT_CHILD_CAP, &parent_grants)?;
    if child.id != request.child_id
        || child.definition.goal != request.goal
        || child
            .definition
            .grants
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            != request.grants
    {
        return Err("child definition differs from request".into());
    }
    actor.spawn_child(parent_id.to_string(), parent_revision, command_id, child)
}

/// Durable idempotent message. Message bodies are untrusted data, never authority.
pub fn message(
    actor: &larch_task::runtime::ActorHandle,
    parent_id: &str,
    child_id: &str,
    message_id: &str,
    text: &str,
) -> Result<(), String> {
    actor.message_child(
        parent_id.into(),
        child_id.into(),
        message_id.into(),
        text.into(),
    )
}
