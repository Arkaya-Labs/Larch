//! Architecture 7: conditions, guarded branches, joins, bounded maps and loops run on the
//! one executor, settle from committed results, and never turn failure into empty success.
use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, Driver, ExecutionPlan, Executor,
    },
    graph::{Graph, Guard, Input, Node, NodeKind, validate_graph},
    runtime::ActorHandle,
    types::{
        Command, CommandEnvelope, PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION,
        WorkflowDefinition,
    },
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

type Calls = Vec<(String, BTreeMap<String, String>)>;
type Script = dyn Fn(&ActivityRequest) -> Result<String, String> + Send + Sync;
struct Scripted(Box<Script>, Mutex<Calls>);
impl Driver for Scripted {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        let output = (self.0)(&request)?;
        self.1
            .lock()
            .unwrap()
            .push((request.node_id.clone(), request.inputs.clone()));
        Ok(ActivityOutcome { output })
    }
}

fn node(id: &str, deps: &[&str], kind: NodeKind) -> Node {
    Node {
        id: id.into(),
        depends_on: deps.iter().map(|d| (*d).into()).collect(),
        kind,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
        when: None,
        optional: Vec::new(),
    }
}
fn json_out(mut n: Node) -> Node {
    n.outputs.insert("out".into(), "json".into());
    n
}
fn json_in(mut n: Node, from: &str) -> Node {
    n.inputs.insert(
        "list".into(),
        Input {
            node: from.into(),
            output: "out".into(),
            schema: "json".into(),
        },
    );
    n
}
fn guarded(mut n: Node, condition: &str, value: bool) -> Node {
    n.when = Some(Guard {
        condition: condition.into(),
        value,
    });
    n
}
fn condition(id: &str, input: &str, equals: serde_json::Value) -> Node {
    node(
        id,
        &[input],
        NodeKind::Condition {
            input: input.into(),
            pointer: "/changed".into(),
            equals,
        },
    )
}

/// Run a graph to quiescence; returns the final state and the activities in call order.
fn run(nodes: Vec<Node>, script: Box<Script>) -> (RunState, Calls) {
    let activities = nodes
        .iter()
        .filter(|n| n.kind.runs_activity())
        .map(|n| {
            (
                n.id.clone(),
                ActivityDescriptor {
                    kind: "scripted".into(),
                    payload: serde_json::json!({}),
                },
            )
        })
        .collect();
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "graph semantics".into(),
        initial_graph: Some(Graph { nodes }),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    actor
        .create(
            "create".into(),
            RunState::new("run".into(), definition).unwrap(),
        )
        .unwrap();
    let driver = Arc::new(Scripted(script, Mutex::new(Vec::new())));
    let executor = Executor::new(actor.clone(), driver.clone());
    executor
        .install_plan(
            "run",
            ExecutionPlan {
                graph_revision: 0,
                activities,
            },
        )
        .unwrap();
    for _ in 0..64 {
        if !executor.tick("run").unwrap() {
            break;
        }
    }
    let state = actor.get("run".into()).unwrap().unwrap();
    actor.shutdown().unwrap();
    let calls = driver.1.lock().unwrap().clone();
    (state, calls)
}

#[test]
fn a_false_condition_skips_its_branch_and_a_no_change_run_succeeds() {
    let nodes = vec![
        node("collect", &[], NodeKind::Activity),
        condition("changed", "collect", true.into()),
        guarded(
            node("repair", &["changed"], NodeKind::Activity),
            "changed",
            true,
        ),
        guarded(
            node("report", &["changed"], NodeKind::Activity),
            "changed",
            false,
        ),
        node("after_repair", &["repair"], NodeKind::Activity),
    ];
    let (state, calls) = run(
        nodes,
        Box::new(|r| {
            Ok(match r.node_id.as_str() {
                "collect" => r#"{"changed": false}"#.into(),
                other => format!("{other} done"),
            })
        }),
    );
    assert_eq!(state.status, RunStatus::Succeeded);
    let order: Vec<_> = calls.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(order, ["collect", "report"]);
    assert_eq!(state.node_results["changed"], "false");
    // A skipped node's required dependents skip too; nothing reads as an empty success.
    assert!(state.skipped_nodes.contains("repair") && state.skipped_nodes.contains("after_repair"));
    assert!(!state.node_results.contains_key("repair"));
}

#[test]
fn an_unknown_condition_fails_never_reads_as_false() {
    let nodes = vec![
        node("collect", &[], NodeKind::Activity),
        condition("changed", "collect", true.into()),
        guarded(
            node("report", &["changed"], NodeKind::Activity),
            "changed",
            false,
        ),
    ];
    for answer in ["no structured answer", r#"{"other": 1}"#] {
        let (failed, calls) = run(nodes.clone(), Box::new(move |_| Ok(answer.into())));
        assert_eq!(failed.status, RunStatus::Failed);
        assert_eq!(
            failed.result.as_deref(),
            Some("condition changed is unknown")
        );
        assert_eq!(calls.len(), 1);
    }
}

#[test]
fn a_join_requires_every_dependency_unless_it_declares_one_optional() {
    let branches = || {
        vec![
            node("collect", &[], NodeKind::Activity),
            condition("changed", "collect", true.into()),
            guarded(
                node("correctness", &["changed"], NodeKind::Activity),
                "changed",
                true,
            ),
            node("compat", &["collect"], NodeKind::Activity),
        ]
    };
    let script = || -> Box<Script> {
        Box::new(|r| {
            Ok(match r.node_id.as_str() {
                "collect" => r#"{"changed": false}"#.into(),
                "compat" => r#"{"ok": true}"#.into(),
                other => format!("{other} done"),
            })
        })
    };
    let mut strict = branches();
    strict.push(node("join", &["correctness", "compat"], NodeKind::Join));
    let (state, _) = run(strict, script());
    assert!(state.skipped_nodes.contains("join"));
    let mut partial = branches();
    let mut join = node("join", &["correctness", "compat"], NodeKind::Join);
    join.optional = vec!["correctness".into()];
    partial.push(join);
    let (state, _) = run(partial, script());
    assert_eq!(state.status, RunStatus::Succeeded);
    let joined: serde_json::Value = serde_json::from_str(&state.node_results["join"]).unwrap();
    assert_eq!(joined, serde_json::json!({"compat": {"ok": true}}));
}

#[test]
fn a_map_runs_each_item_in_order_and_refuses_inputs_over_its_bound() {
    let nodes = |max_items| {
        vec![
            json_out(node("list", &[], NodeKind::Activity)),
            json_in(node("each", &["list"], NodeKind::Map { max_items }), "list"),
        ]
    };
    let script = || -> Box<Script> {
        Box::new(|r| {
            Ok(match r.node_id.as_str() {
                "list" => r#"["a", "b", "c"]"#.into(),
                _ => format!("{{\"seen\": {}}}", r.inputs["item"]),
            })
        })
    };
    let (state, calls) = run(nodes(3), script());
    assert_eq!(state.status, RunStatus::Succeeded);
    let items: Vec<_> = calls[1..].iter().map(|(_, i)| i["item"].clone()).collect();
    assert_eq!(items, [r#""a""#, r#""b""#, r#""c""#]);
    let result: serde_json::Value = serde_json::from_str(&state.node_results["each"]).unwrap();
    assert_eq!(
        result,
        serde_json::json!([{"seen": "a"}, {"seen": "b"}, {"seen": "c"}])
    );
    assert!(state.partials.is_empty());
    let (over, calls) = run(nodes(2), script());
    assert_eq!(over.status, RunStatus::Failed);
    assert_eq!(calls.len(), 1);
}

#[test]
fn an_empty_map_succeeds_without_an_activity() {
    let nodes = vec![
        json_out(node("list", &[], NodeKind::Activity)),
        json_in(
            node("each", &["list"], NodeKind::Map { max_items: 4 }),
            "list",
        ),
    ];
    let (state, calls) = run(nodes, Box::new(|_| Ok("[]".into())));
    assert_eq!(state.status, RunStatus::Succeeded);
    assert_eq!(state.node_results["each"], "[]");
    assert_eq!(calls.len(), 1);
}

#[test]
fn a_loop_stops_on_done_and_fails_when_its_bound_runs_out() {
    let looped = |max_iterations| vec![node("refine", &[], NodeKind::Loop { max_iterations })];
    let (state, calls) = run(
        looped(5),
        Box::new(|r| {
            let i: u32 = r.inputs["iteration"].parse().unwrap();
            Ok(serde_json::json!({"done": i == 2, "value": format!("draft {i}")}).to_string())
        }),
    );
    assert_eq!(state.status, RunStatus::Succeeded);
    assert_eq!(calls.len(), 3);
    assert_eq!(state.node_results["refine"], "draft 2");
    let (exhausted, calls) = run(
        looped(2),
        Box::new(|_| Ok(r#"{"done": false, "value": null}"#.into())),
    );
    assert_eq!(exhausted.status, RunStatus::Failed);
    assert_eq!(calls.len(), 2);
}

#[test]
fn a_loop_result_without_done_waits_instead_of_finishing() {
    let (state, _) = run(
        vec![node("refine", &[], NodeKind::Loop { max_iterations: 3 })],
        Box::new(|_| Ok("plain text".into())),
    );
    assert_eq!(state.status, RunStatus::Waiting);
    assert!(state.attempt_failure.is_some() && state.partials.is_empty());
}

#[test]
fn settle_refuses_activity_nodes_and_unready_control_nodes() {
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "fences".into(),
        initial_graph: Some(Graph {
            nodes: vec![
                node("collect", &[], NodeKind::Activity),
                node("join", &["collect"], NodeKind::Join),
            ],
        }),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let state = RunState::new("run".into(), definition).unwrap();
    for id in ["collect", "join"] {
        let settle = CommandEnvelope {
            id: format!("settle-{id}"),
            expected_revision: 0,
            command: Command::SettleNode { node_id: id.into() },
        };
        assert!(larch_task::reducer::apply(&state, &settle).is_err());
    }
}

#[test]
fn validation_rejects_misused_guards_optionals_and_map_inputs() {
    let collect = node("collect", &[], NodeKind::Activity);
    let not_a_condition = guarded(node("b", &["collect"], NodeKind::Activity), "collect", true);
    assert!(
        validate_graph(&Graph {
            nodes: vec![collect.clone(), not_a_condition]
        })
        .is_err()
    );
    let mut optional_on_activity = node("b", &["collect"], NodeKind::Activity);
    optional_on_activity.optional = vec!["collect".into()];
    assert!(
        validate_graph(&Graph {
            nodes: vec![collect.clone(), optional_on_activity]
        })
        .is_err()
    );
    let untyped_map = node("each", &["collect"], NodeKind::Map { max_items: 2 });
    assert!(
        validate_graph(&Graph {
            nodes: vec![collect.clone(), untyped_map]
        })
        .is_err()
    );
    let stray = condition("c", "collect", true.into());
    let mut detached = stray.clone();
    detached.depends_on.clear();
    assert!(
        validate_graph(&Graph {
            nodes: vec![collect.clone(), detached]
        })
        .is_err()
    );
    assert!(
        validate_graph(&Graph {
            nodes: vec![collect, stray]
        })
        .is_ok()
    );
}
