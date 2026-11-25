//! Validation of the single workflow graph, including planner revisions.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub depends_on: Vec<String>,
    pub kind: NodeKind,
    /// Named inputs map to an upstream node and its declared output type.
    pub inputs: BTreeMap<String, Input>,
    pub outputs: BTreeMap<String, String>,
    /// Runs only when the named condition dependency settled to `value`; otherwise skipped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<Guard>,
    /// Join dependencies that may be skipped; every other dependency must succeed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub optional: Vec<String>,
}
/// A branch guard on a condition node's settled value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Guard {
    pub condition: String,
    pub value: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    pub node: String,
    pub output: String,
    pub schema: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Activity,
    /// Settles without an activity: an object of its succeeded dependencies' results.
    Join,
    /// Runs its activity once per item of its single `json` array input, in order.
    Map {
        max_items: u32,
    },
    /// Reruns its activity until it returns `{"done": true, "value": ...}`; exhausting the
    /// bound fails the node rather than returning a partial value.
    Loop {
        max_iterations: u32,
    },
    /// Settles without an activity: whether the JSON at `pointer` in dependency `input`'s
    /// result equals `equals`. A missing or unparsable value is unknown and fails the run;
    /// unknown never reads as false.
    // ponytail: no "ask the user" on unknown; add it as a decision whose answer is the value
    // when a planner emits one
    Condition {
        input: String,
        pointer: String,
        equals: serde_json::Value,
    },
}
impl NodeKind {
    /// Whether the node runs a planned activity rather than settling in the reducer.
    pub fn runs_activity(&self) -> bool {
        matches!(self, Self::Activity | Self::Map { .. } | Self::Loop { .. })
    }
}

/// Outcome of evaluating a condition on stored results.
pub fn evaluate_condition(kind: &NodeKind, results: &BTreeMap<String, String>) -> Option<bool> {
    let NodeKind::Condition {
        input,
        pointer,
        equals,
        ..
    } = kind
    else {
        return None;
    };
    let value: serde_json::Value = serde_json::from_str(results.get(input)?).ok()?;
    Some(value.pointer(pointer)? == equals)
}

/// Validates identifiers, schemas, references, bounded constructs and DAG topology.
pub fn validate_graph(graph: &Graph) -> Result<(), String> {
    let mut by_id = BTreeMap::new();
    for node in &graph.nodes {
        if node.id.trim().is_empty() || by_id.insert(node.id.as_str(), node).is_some() {
            return Err("empty or duplicate node id".into());
        }
        match &node.kind {
            NodeKind::Map { max_items: 0 } | NodeKind::Loop { max_iterations: 0 } => {
                return Err("map and loop bounds must be positive".into());
            }
            _ => {}
        }
        if node
            .outputs
            .iter()
            .any(|(name, schema)| name.trim().is_empty() || schema.trim().is_empty())
        {
            return Err("invalid output schema".into());
        }
    }
    for node in &graph.nodes {
        let mut seen = BTreeSet::new();
        for dep in &node.depends_on {
            if !by_id.contains_key(dep.as_str()) || !seen.insert(dep.as_str()) {
                return Err("missing or duplicate dependency".into());
            }
        }
        if matches!(&node.kind, NodeKind::Join) && node.depends_on.is_empty() {
            return Err("join requires dependencies".into());
        }
        if !node.optional.is_empty()
            && (!matches!(&node.kind, NodeKind::Join)
                || node.optional.iter().any(|id| !seen.contains(id.as_str())))
        {
            return Err("only a join may declare optional dependencies".into());
        }
        if let NodeKind::Condition { input, pointer, .. } = &node.kind
            && (!seen.contains(input.as_str()) || !(pointer.is_empty() || pointer.starts_with('/')))
        {
            return Err("condition must read a dependency at a JSON pointer".into());
        }
        if let Some(guard) = &node.when
            && (!seen.contains(guard.condition.as_str())
                || !matches!(
                    by_id.get(guard.condition.as_str()).map(|n| &n.kind),
                    Some(NodeKind::Condition { .. })
                ))
        {
            return Err("guard must name a condition dependency".into());
        }
        if matches!(&node.kind, NodeKind::Map { .. })
            && (node.inputs.len() != 1 || node.inputs.values().any(|i| i.schema != "json"))
        {
            return Err("map requires exactly one json input".into());
        }
        for (name, input) in &node.inputs {
            if name.trim().is_empty()
                || input.schema.trim().is_empty()
                || !seen.contains(input.node.as_str())
            {
                return Err("input must name a dependency and schema".into());
            }
            let upstream = by_id.get(input.node.as_str()).ok_or("missing input node")?;
            if upstream.outputs.get(&input.output) != Some(&input.schema) {
                return Err("input output schema mismatch".into());
            }
        }
    }
    let mut remaining: BTreeMap<&str, usize> = graph
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), n.depends_on.len()))
        .collect();
    let mut ready: Vec<&str> = remaining
        .iter()
        .filter_map(|(&id, &degree)| (degree == 0).then_some(id))
        .collect();
    let mut visited = 0;
    while let Some(id) = ready.pop() {
        visited += 1;
        for node in &graph.nodes {
            if node.depends_on.iter().any(|d| d == id) {
                let degree = remaining.get_mut(node.id.as_str()).expect("indexed node");
                *degree -= 1;
                if *degree == 0 {
                    ready.push(node.id.as_str());
                }
            }
        }
    }
    if visited != graph.nodes.len() {
        return Err("graph contains a cycle".into());
    }
    Ok(())
}

/// Call under the same transaction that compares and advances graph revision.
/// Started or completed nodes are immutable; scope and acceptance may only narrow.
#[allow(clippy::too_many_arguments)]
pub fn validate_revision(
    old: &Graph,
    new: &Graph,
    expected_revision: u64,
    actual_revision: u64,
    started: &BTreeSet<String>,
    old_grants: &BTreeSet<String>,
    new_grants: &BTreeSet<String>,
    old_acceptance: &BTreeSet<String>,
    new_acceptance: &BTreeSet<String>,
) -> Result<(), String> {
    if expected_revision != actual_revision {
        return Err("stale graph revision".into());
    }
    validate_graph(new)?;
    if !new_grants.is_subset(old_grants) {
        return Err("revision widens grants".into());
    }
    if !old_acceptance.is_subset(new_acceptance) {
        return Err("revision weakens acceptance".into());
    }
    let old_nodes: BTreeMap<_, _> = old.nodes.iter().map(|n| (&n.id, n)).collect();
    let new_nodes: BTreeMap<_, _> = new.nodes.iter().map(|n| (&n.id, n)).collect();
    for id in started {
        if !old_nodes.contains_key(id) || old_nodes.get(id) != new_nodes.get(id) {
            return Err("started history is immutable".into());
        }
    }
    Ok(())
}
