use crate::types::*;

/// Apply one command without I/O. The caller commits state and event atomically.
pub fn apply(
    state: &RunState,
    envelope: &CommandEnvelope,
) -> Result<(RunState, Event, CommandReply), String> {
    apply_inner(state, envelope, false)
}
pub(crate) fn apply_budgeted(
    state: &RunState,
    envelope: &CommandEnvelope,
) -> Result<(RunState, Event, CommandReply), String> {
    apply_inner(state, envelope, true)
}
fn apply_inner(
    state: &RunState,
    envelope: &CommandEnvelope,
    budget_reserved: bool,
) -> Result<(RunState, Event, CommandReply), String> {
    if state.version != SCHEMA_VERSION {
        return Err(format!("unsupported run version {}", state.version));
    }
    if envelope.id.is_empty() {
        return Err("command id is empty".into());
    }
    if state.revision != envelope.expected_revision {
        return Err("revision conflict".into());
    }
    if state.status.terminal()
        && !(state.status == RunStatus::Cancelled
            && matches!(&envelope.command, Command::ReconcileEffect { .. }))
    {
        return Err("terminal run is immutable".into());
    }
    let mut next = state.clone();
    let kind = match &envelope.command {
        Command::PatchGraph {
            graph,
            expected_graph_revision,
            grants,
            acceptance,
        } => {
            if next.status != RunStatus::Ready {
                return Err("run is not ready for graph revision".into());
            }
            if next.graph.is_some() {
                // ponytail: only the initial plan patches the graph; Adaptive revisions need
                // validate_revision against the live graph once their boundaries are enforced
                return Err("adaptive graph boundaries are not yet enforced".into());
            }
            if next.definition.planning_policy == PlanningPolicy::Disabled {
                return Err("planning is disabled".into());
            }
            crate::graph::validate_revision(
                &crate::graph::Graph::default(),
                graph,
                *expected_graph_revision,
                next.graph_revision,
                &next.started_nodes,
                &next.definition.grants.iter().cloned().collect(),
                &grants.iter().cloned().collect(),
                &next.definition.acceptance.iter().cloned().collect(),
                &acceptance.iter().cloned().collect(),
            )?;
            next.graph_revision = next
                .graph_revision
                .checked_add(1)
                .ok_or("graph revision overflow")?;
            next.graph = Some(graph.clone());
            next.acceptance_receipts.clear();
            next.definition.grants = grants.clone();
            next.definition.acceptance = acceptance.clone();
            next.retry = None;
            "graph_changed"
        }
        Command::RevokeGrant { grant } => {
            if !next.definition.grants.contains(grant) {
                return Err("unknown grant".into());
            }
            next.definition.grants.retain(|value| value != grant);
            next.grant_generation = next
                .grant_generation
                .checked_add(1)
                .ok_or("grant generation overflow")?;
            "grant_revoked"
        }
        Command::WaitForChild {
            attempt_id,
            epoch,
            child_id,
        } => {
            check_attempt(&next, attempt_id, *epoch)?;
            if !budget_reserved && !next.definition.budgets.is_empty() {
                return Err("budgeted child suspension requires atomic settlement".into());
            }
            if next.effects.values().any(|effect| {
                effect.attempt_id == *attempt_id
                    && matches!(
                        effect.status,
                        EffectStatus::Intent | EffectStatus::Uncertain
                    )
            }) {
                return Err("attempt has unresolved effects".into());
            }
            if next.attempt.as_ref().is_some_and(|a| a.iteration.is_some()) {
                // ponytail: map items and loop iterations run inline; child handoff per item when delegation lands
                return Err("map and loop iterations cannot wait for a child".into());
            }
            let attempt = next.attempt.take().ok_or("missing attempt")?;
            next.waiting_child = Some(ChildWait {
                child_id: child_id.clone(),
                node_id: attempt.node_id,
                graph_revision: next.graph_revision,
                origin_attempt_id: Some(attempt.id),
                origin_epoch: Some(attempt.epoch),
            });
            next.status = RunStatus::Waiting;
            "child_wait_started"
        }
        Command::WakeChild { child_id, result } => {
            let wait = next.waiting_child.take().ok_or("no child wait")?;
            if next.status != RunStatus::Waiting
                || wait.child_id != *child_id
                || wait.graph_revision != next.graph_revision
            {
                return Err("stale child result".into());
            }
            next.node_results
                .insert(wait.node_id.clone(), result.clone());
            next.completed_nodes.insert(wait.node_id);
            next.result = Some(result.clone());
            next.status = RunStatus::Ready;
            "child_wait_completed"
        }
        Command::RecordChildFailure {
            child_id,
            kind,
            reason,
        } => {
            let wait = next.waiting_child.as_ref().ok_or("no child wait")?;
            if next.status != RunStatus::Waiting
                || wait.child_id != *child_id
                || next.child_failure.is_some()
            {
                return Err("child failure cannot be recorded".into());
            }
            next.child_failure = Some(ChildFailure {
                child_id: child_id.clone(),
                kind: kind.clone(),
                reason: reason.clone(),
            });
            "child_wait_failed"
        }
        Command::RetryChild { child_id } => {
            if next.status != RunStatus::Waiting
                || next
                    .waiting_child
                    .as_ref()
                    .is_none_or(|wait| wait.child_id != *child_id)
                || next
                    .child_failure
                    .as_ref()
                    .is_none_or(|failure| failure.child_id != *child_id)
                || next.effects.values().any(|effect| {
                    matches!(
                        effect.status,
                        EffectStatus::Intent | EffectStatus::Uncertain
                    )
                })
            {
                return Err("child cannot be retried".into());
            }
            next.waiting_child = None;
            next.child_failure = None;
            next.acceptance_receipts.clear();
            next.status = RunStatus::Ready;
            "child_retry_requested"
        }
        Command::RequestDecision {
            attempt_id,
            epoch,
            request_id,
            action_digest,
            scope,
            expires_at,
        } => {
            check_attempt(&next, attempt_id, *epoch)?;
            if !next.definition.budgets.is_empty() {
                // ponytail: budgeted runs settle usage atomically on suspension; add a budgeted decision handoff when a budgeted Review workflow needs one
                return Err("budgeted decision suspension requires atomic settlement".into());
            }
            if request_id.is_empty()
                || action_digest.is_empty()
                || scope.trim().is_empty()
                || next.decisions.contains_key(request_id)
            {
                return Err("invalid or duplicate decision request".into());
            }
            if next.effects.values().any(|effect| {
                effect.attempt_id == *attempt_id
                    && matches!(
                        effect.status,
                        EffectStatus::Intent | EffectStatus::Uncertain
                    )
            }) {
                return Err("attempt has unresolved effects".into());
            }
            let attempt = next.attempt.take().ok_or("missing attempt")?;
            if attempt.iteration.is_some() {
                return Err("map and loop iterations cannot wait for a decision".into());
            }
            next.decisions.insert(
                request_id.clone(),
                Decision {
                    node_id: attempt.node_id,
                    action_digest: action_digest.clone(),
                    scope: scope.clone(),
                    expires_at: *expires_at,
                    grant_generation: next.grant_generation,
                    approved: None,
                },
            );
            next.waiting_decision = Some(request_id.clone());
            next.status = RunStatus::Waiting;
            "decision_required"
        }
        Command::RespondDecision {
            request_id,
            action_digest,
            approve,
            now,
        } => {
            let decision = next.decisions.get(request_id).ok_or("unknown decision")?;
            if next.status != RunStatus::Waiting
                || next.waiting_decision.as_deref() != Some(request_id.as_str())
                || decision.approved.is_some()
            {
                return Err("decision is not pending".into());
            }
            if decision.action_digest != *action_digest {
                return Err("decision is bound to a different action".into());
            }
            if *now >= decision.expires_at {
                return Err("decision request expired".into());
            }
            if decision.grant_generation != next.grant_generation {
                return Err("grants changed since the decision was requested".into());
            }
            let node = decision.node_id.clone();
            next.decisions
                .get_mut(request_id)
                .ok_or("unknown decision")?
                .approved = Some(*approve);
            next.waiting_decision = None;
            next.status = RunStatus::Ready;
            if *approve {
                "decision_approved"
            } else {
                // A rejected action settles its node; the patch disposition is the result.
                let result = serde_json::json!({"decision": "rejected", "request_id": request_id})
                    .to_string();
                next.result = Some(result.clone());
                complete(&mut next, &node, result);
                "decision_rejected"
            }
        }
        Command::RetireDecision { request_id, now } => {
            let decision = next.decisions.get(request_id).ok_or("unknown decision")?;
            if next.status != RunStatus::Waiting
                || next.waiting_decision.as_deref() != Some(request_id.as_str())
                || decision.approved.is_some()
            {
                return Err("decision is not pending".into());
            }
            if next.live(decision, *now) {
                return Err("decision request is still valid".into());
            }
            // The row stays unanswered: it never authorizes the action and its id is never reused.
            next.waiting_decision = None;
            next.status = RunStatus::Ready;
            "decision_retired"
        }
        Command::SettleNode { node_id } => {
            if next.status != RunStatus::Ready || next.attempt.is_some() {
                return Err("run is not ready".into());
            }
            let graph = next.graph.clone().ok_or("missing graph")?;
            let node = graph
                .nodes
                .iter()
                .find(|node| node.id == *node_id)
                .ok_or("unknown graph node")?;
            if next.settled(node_id) || node.depends_on.iter().any(|id| !next.settled(id)) {
                return Err("node is not ready to settle".into());
            }
            settle(&mut next, node)?
        }
        Command::StartAttempt { attempt } => {
            if !budget_reserved && !next.definition.budgets.is_empty() {
                return Err("budget accounting is not connected".into());
            }
            if next.status != RunStatus::Ready || next.attempt.is_some() {
                return Err("run is not ready".into());
            }
            if attempt.id.is_empty()
                || attempt.node_id.is_empty()
                || attempt.epoch <= next.last_epoch
                || next.used_attempt_ids.contains(&attempt.id)
            {
                return Err("invalid attempt identity or epoch".into());
            }
            if let Some(graph) = &next.graph {
                let node = graph
                    .nodes
                    .iter()
                    .find(|node| node.id == attempt.node_id)
                    .ok_or("unknown graph node")?;
                if !node.kind.runs_activity() {
                    return Err("control nodes settle without an attempt".into());
                }
                for input in node.inputs.values() {
                    let upstream = graph
                        .nodes
                        .iter()
                        .find(|source| source.id == input.node)
                        .ok_or("missing input source")?;
                    if !matches!(input.schema.as_str(), "text" | "json")
                        || upstream.outputs.len() != 1
                        || upstream.outputs.get(&input.output) != Some(&input.schema)
                        || !next.node_results.contains_key(&input.node)
                    {
                        return Err("unsupported or missing typed input".into());
                    }
                }
                if next.settled(&node.id)
                    || node.depends_on.iter().any(|id| !next.settled(id))
                    || must_skip(&next, node)
                {
                    return Err("node is not runnable".into());
                }
                let done = next.partials.get(&node.id).map_or(0, Vec::len);
                let bound = match &node.kind {
                    crate::graph::NodeKind::Map { .. } => Some(map_items(&next, node)?.len()),
                    crate::graph::NodeKind::Loop { max_iterations } => {
                        Some(*max_iterations as usize)
                    }
                    _ => None,
                };
                let expected = bound
                    .map(|_| u32::try_from(done))
                    .transpose()
                    .map_err(|_| "iteration overflow")?;
                if attempt.iteration != expected || bound.is_some_and(|bound| done >= bound) {
                    return Err("attempt iteration does not match the node".into());
                }
                next.started_nodes.insert(node.id.clone());
            } else if attempt.node_id != "__planning__"
                || next.definition.planning_policy == PlanningPolicy::Disabled
            {
                return Err("initial planning is required".into());
            }
            next.attempt_failure = None;
            next.last_epoch = attempt.epoch;
            next.used_attempt_ids.insert(attempt.id.clone());
            next.attempt = Some(attempt.clone());
            next.status = RunStatus::Running;
            "attempt_started"
        }
        Command::FinishAttempt {
            attempt_id,
            epoch,
            result,
        } => {
            if !budget_reserved && !next.definition.budgets.is_empty() {
                return Err("budget settlement is not connected".into());
            }
            check_attempt(&next, attempt_id, *epoch)?;
            if next.effects.values().any(|e| {
                e.attempt_id == *attempt_id
                    && matches!(e.status, EffectStatus::Intent | EffectStatus::Uncertain)
            }) {
                return Err("attempt has unresolved effects".into());
            }
            next.result = Some(result.clone());
            next.status = RunStatus::Ready;
            let attempt = next.attempt.take().ok_or("missing attempt")?;
            next.retry = None;
            if attempt.node_id == "__planning__" {
                "attempt_finished"
            } else {
                finish_node(&mut next, &attempt.node_id, result)?
            }
        }
        Command::FailAttempt {
            attempt_id,
            epoch,
            reason,
        } => {
            if !budget_reserved && !next.definition.budgets.is_empty() {
                return Err("budget settlement is not connected".into());
            }
            check_attempt(&next, attempt_id, *epoch)?;
            mark_uncertain(&mut next, attempt_id);
            next.attempt = None;
            next.result = Some(reason.clone());
            next.attempt_failure = Some(reason.clone());
            next.status = RunStatus::Waiting;
            "attempt_failed"
        }
        Command::RetryAttempt {
            attempt_id,
            epoch,
            reason,
            not_before_ms,
        } => {
            // ponytail: unbudgeted only; a budgeted retry needs settle_budgeted to take this command
            if !next.definition.budgets.is_empty() {
                return Err("budgeted attempts are not retried".into());
            }
            check_attempt(&next, attempt_id, *epoch)?;
            mark_uncertain(&mut next, attempt_id);
            let attempt = next.attempt.take().ok_or("missing attempt")?;
            next.result = Some(reason.clone());
            next.attempt_failure = Some(reason.clone());
            // An effect that may have happened is the user's to settle: never run past it.
            if next
                .effects
                .values()
                .any(|e| matches!(e.status, EffectStatus::Intent | EffectStatus::Uncertain))
            {
                next.status = RunStatus::Waiting;
                "attempt_failed"
            } else {
                let failures = next
                    .retry
                    .as_ref()
                    .filter(|r| r.node_id == attempt.node_id)
                    .map_or(0, |r| r.failures);
                next.retry = Some(crate::types::Retry {
                    node_id: attempt.node_id,
                    failures: failures.checked_add(1).ok_or("retry count overflow")?,
                    not_before_ms: *not_before_ms,
                });
                next.status = RunStatus::Ready;
                "attempt_retrying"
            }
        }
        Command::RecordIntent {
            effect_id,
            attempt_id,
            epoch,
            operation_digest,
        } => {
            check_attempt(&next, attempt_id, *epoch)?;
            if effect_id.is_empty()
                || operation_digest.is_empty()
                || next.effects.contains_key(effect_id)
            {
                return Err("invalid or duplicate effect".into());
            }
            next.effects.insert(
                effect_id.clone(),
                Effect {
                    attempt_id: attempt_id.clone(),
                    operation_digest: operation_digest.clone(),
                    status: EffectStatus::Intent,
                },
            );
            "effect_intent"
        }
        Command::RecordReceipt {
            effect_id,
            attempt_id,
            epoch,
            receipt,
        } => {
            check_attempt(&next, attempt_id, *epoch)?;
            let effect = next.effects.get_mut(effect_id).ok_or("unknown effect")?;
            if effect.attempt_id != *attempt_id || effect.status != EffectStatus::Intent {
                return Err("effect is not an active intent".into());
            }
            effect.status = EffectStatus::Completed {
                receipt: receipt.clone(),
            };
            "effect_receipt"
        }
        Command::ReconcileEffect { effect_id, receipt } => {
            // Paused covers a restore, which turns an in-flight intent uncertain and pauses.
            if !matches!(
                next.status,
                RunStatus::Waiting | RunStatus::Paused | RunStatus::Cancelled
            ) {
                return Err("run is not awaiting reconciliation".into());
            }
            let effect = next.effects.get_mut(effect_id).ok_or("unknown effect")?;
            if effect.status != EffectStatus::Uncertain {
                return Err("effect is not uncertain".into());
            }
            effect.status = match receipt {
                Some(value) => EffectStatus::Completed {
                    receipt: value.clone(),
                },
                None => EffectStatus::ReconciledAbsent,
            };
            "effect_reconciled"
        }
        Command::RejectAcceptance { criterion, reason } => {
            if !next.graph_settled()
                || next.status != RunStatus::Ready
                || !next.definition.acceptance.contains(criterion)
                || next.acceptance_receipts.contains_key(criterion)
                || reason.trim().is_empty()
            {
                return Err("invalid acceptance rejection".into());
            }
            next.attempt_failure = Some(format!("not done yet ({criterion}): {reason}"));
            next.status = RunStatus::Waiting;
            "acceptance_failed"
        }
        Command::RecordAcceptance { criterion, receipt } => {
            if !next.graph_settled() {
                return Err("graph is incomplete".into());
            }
            if next.effects.values().any(|effect| {
                matches!(
                    effect.status,
                    EffectStatus::Intent | EffectStatus::Uncertain
                )
            }) {
                return Err("effects are unresolved".into());
            }
            if next.status != RunStatus::Ready
                || !next.definition.acceptance.contains(criterion)
                || receipt.trim().is_empty()
            {
                return Err("invalid acceptance receipt".into());
            }
            if next.acceptance_receipts.contains_key(criterion) {
                return Err("acceptance already recorded".into());
            }
            next.acceptance_receipts
                .insert(criterion.clone(), receipt.clone());
            // A pass settles an earlier rejection: no stale "not done yet" on a finished run.
            next.attempt_failure = None;
            "acceptance_recorded"
        }
        Command::CompleteRun { result } => {
            if next
                .definition
                .acceptance
                .iter()
                .any(|criterion| !next.acceptance_receipts.contains_key(criterion))
            {
                return Err("acceptance is incomplete".into());
            }
            if !next.graph_settled() {
                return Err("graph is incomplete".into());
            }
            if next.status != RunStatus::Ready
                || next
                    .effects
                    .values()
                    .any(|e| matches!(e.status, EffectStatus::Intent | EffectStatus::Uncertain))
            {
                return Err("run has pending work or effects".into());
            }
            next.status = RunStatus::Succeeded;
            next.result = Some(result.clone());
            next.attempt_failure = None;
            "run_succeeded"
        }
        Command::Recover => {
            if let Some(attempt) = next.attempt.take() {
                mark_uncertain(&mut next, &attempt.id);
                next.attempt_failure = None;
                next.status = RunStatus::Waiting;
            }
            "recovered"
        }
        Command::Pause => {
            if next.status != RunStatus::Ready
                && !(next.status == RunStatus::Waiting
                    && (next.waiting_child.is_some() || next.waiting_decision.is_some()))
            {
                return Err("run cannot pause in its current state".into());
            }
            next.status = RunStatus::Paused;
            "paused"
        }
        Command::Resume => {
            if !matches!(next.status, RunStatus::Paused | RunStatus::Waiting)
                || (next.status == RunStatus::Waiting
                    && (next.waiting_child.is_some() || next.waiting_decision.is_some()))
                || next
                    .effects
                    .values()
                    .any(|e| matches!(e.status, EffectStatus::Intent | EffectStatus::Uncertain))
            {
                return Err("run cannot resume with unresolved effects".into());
            }
            // A parked run the user restarts gets a fresh retry budget; a pause keeps the count
            // and the backoff, so pausing in a backoff never adds attempts.
            if next.status == RunStatus::Waiting {
                next.retry = None;
            }
            next.status = if next.waiting_child.is_some() || next.waiting_decision.is_some() {
                RunStatus::Waiting
            } else {
                RunStatus::Ready
            };
            "resumed"
        }
        Command::Cancel => {
            if let Some(attempt) = next.attempt.take() {
                mark_uncertain(&mut next, &attempt.id);
            }
            next.status = RunStatus::Cancelled;
            "cancelled"
        }
    };
    next.revision = next.revision.checked_add(1).ok_or("revision overflow")?;
    let event = Event {
        revision: next.revision,
        command_id: envelope.id.clone(),
        kind: kind.into(),
    };
    let reply = CommandReply {
        revision: next.revision,
        status: next.status.clone(),
    };
    Ok((next, event, reply))
}
fn check_attempt(state: &RunState, id: &str, epoch: u64) -> Result<(), String> {
    match &state.attempt {
        Some(a) if state.status == RunStatus::Running && a.id == id && a.epoch == epoch => Ok(()),
        _ => Err("stale attempt".into()),
    }
}
fn mark_uncertain(state: &mut RunState, attempt_id: &str) {
    for effect in state.effects.values_mut() {
        if effect.attempt_id == attempt_id && effect.status == EffectStatus::Intent {
            effect.status = EffectStatus::Uncertain;
        }
    }
}

/// A false guard, or a skipped dependency the node requires, takes it off the path.
pub(crate) fn must_skip(state: &RunState, node: &crate::graph::Node) -> bool {
    let guard_off = node.when.as_ref().is_some_and(|guard| {
        state.skipped_nodes.contains(&guard.condition)
            || state.node_results.get(&guard.condition).map(String::as_str)
                != Some(if guard.value { "true" } else { "false" })
    });
    guard_off
        || node
            .depends_on
            .iter()
            .any(|id| state.skipped_nodes.contains(id) && !node.optional.contains(id))
}
/// The items a map node runs over: its single json input, an array within the bound.
pub(crate) fn map_items(
    state: &RunState,
    node: &crate::graph::Node,
) -> Result<Vec<serde_json::Value>, String> {
    let crate::graph::NodeKind::Map { max_items } = &node.kind else {
        return Err("not a map node".into());
    };
    let input = node.inputs.values().next().ok_or("map input missing")?;
    let raw = state
        .node_results
        .get(&input.node)
        .ok_or("map input missing")?;
    match serde_json::from_str(raw) {
        Ok(serde_json::Value::Array(items)) if items.len() <= *max_items as usize => Ok(items),
        Ok(serde_json::Value::Array(items)) => Err(format!(
            "map input has {} items, over its bound of {max_items}",
            items.len()
        )),
        _ => Err("map input is not a JSON array".into()),
    }
}
/// A result as JSON: parsed when it is JSON, otherwise the text itself.
fn as_json(result: &str) -> serde_json::Value {
    serde_json::from_str(result).unwrap_or_else(|_| serde_json::Value::String(result.into()))
}
fn complete(state: &mut RunState, node: &str, result: String) {
    state.partials.remove(node);
    state.node_results.insert(node.into(), result);
    state.completed_nodes.insert(node.into());
}
/// Settle a control step from stored results; activity nodes settle here only when skipped,
/// or when a map has nothing (or nothing valid) to run over.
fn settle(state: &mut RunState, node: &crate::graph::Node) -> Result<&'static str, String> {
    use crate::graph::NodeKind;
    if must_skip(state, node) {
        state.skipped_nodes.insert(node.id.clone());
        return Ok("node_skipped");
    }
    match &node.kind {
        NodeKind::Join => {
            let joined: serde_json::Map<_, _> = node
                .depends_on
                .iter()
                .filter_map(|id| Some((id.clone(), as_json(state.node_results.get(id)?))))
                .collect();
            complete(
                state,
                &node.id,
                serde_json::Value::Object(joined).to_string(),
            );
            Ok("node_settled")
        }
        NodeKind::Condition { .. } => {
            match crate::graph::evaluate_condition(&node.kind, &state.node_results) {
                Some(value) => {
                    complete(state, &node.id, value.to_string());
                    Ok("node_settled")
                }
                None => {
                    state.result = Some(format!("condition {} is unknown", node.id));
                    state.status = RunStatus::Failed;
                    Ok("condition_unknown")
                }
            }
        }
        NodeKind::Map { .. } => match map_items(state, node) {
            Ok(items) if items.is_empty() => {
                complete(state, &node.id, "[]".into());
                Ok("node_settled")
            }
            Ok(_) => Err("node runs an activity".into()),
            Err(reason) => {
                state.result = Some(reason);
                state.status = RunStatus::Failed;
                Ok("map_input_invalid")
            }
        },
        NodeKind::Activity | NodeKind::Loop { .. } => Err("node runs an activity".into()),
    }
}
/// Commit one activity result: a plain node completes; a map item or loop iteration
/// accumulates until the node's bound or its done signal.
fn finish_node(state: &mut RunState, node_id: &str, result: &str) -> Result<&'static str, String> {
    use crate::graph::NodeKind;
    let kind = state
        .graph
        .as_ref()
        .and_then(|graph| graph.nodes.iter().find(|node| node.id == node_id))
        .map(|node| node.kind.clone())
        .ok_or("unknown graph node")?;
    match kind {
        NodeKind::Map { .. } => {
            let node = state
                .graph
                .as_ref()
                .and_then(|graph| graph.nodes.iter().find(|node| node.id == node_id))
                .cloned()
                .ok_or("unknown graph node")?;
            let total = map_items(state, &node)?.len();
            let partial = state.partials.entry(node_id.into()).or_default();
            partial.push(result.into());
            if partial.len() == total {
                let items: Vec<_> = partial.iter().map(|item| as_json(item)).collect();
                complete(state, node_id, serde_json::Value::Array(items).to_string());
                return Ok("node_completed");
            }
            Ok("iteration_finished")
        }
        NodeKind::Loop { max_iterations } => {
            let step: serde_json::Value = serde_json::from_str(result).unwrap_or_default();
            let (Some(done), Some(value)) = (
                step.get("done").and_then(|d| d.as_bool()),
                step.get("value"),
            ) else {
                state.attempt_failure =
                    Some("loop result must be {\"done\": bool, \"value\": ...}".into());
                state.status = RunStatus::Waiting;
                return Ok("attempt_failed");
            };
            if done {
                let value = value
                    .as_str()
                    .map_or_else(|| value.to_string(), str::to_owned);
                complete(state, node_id, value);
                return Ok("node_completed");
            }
            let partial = state.partials.entry(node_id.into()).or_default();
            partial.push(result.into());
            if partial.len() >= max_iterations as usize {
                state.result = Some(format!(
                    "loop {node_id} exhausted {max_iterations} iterations"
                ));
                state.status = RunStatus::Failed;
                return Ok("loop_exhausted");
            }
            Ok("iteration_finished")
        }
        _ => {
            complete(state, node_id, result.into());
            Ok("attempt_finished")
        }
    }
}
