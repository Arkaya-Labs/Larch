//! A delegating step spawns one depth-one child with at most the parent's grants, waits for
//! it without holding the driver, and takes its result.
use larch_agent::runtime::driver::ActivityRegistry;
use larch_task::{
    execution::{ActivityDescriptor, ExecutionPlan, Executor},
    graph::{Graph, Node, NodeKind},
    runtime::ActorHandle,
    types::{PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION, WorkflowDefinition},
};
use std::{collections::BTreeMap, sync::Arc};

fn root(actor: &ActorHandle, executor: &Executor, id: &str, payload: serde_json::Value) {
    let node = Node {
        id: "ask".into(),
        depends_on: vec![],
        kind: NodeKind::Activity,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
        when: None,
        optional: vec![],
    };
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "delegate".into(),
        initial_graph: Some(Graph { nodes: vec![node] }),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec!["read".into()],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    actor
        .create(
            format!("create-{id}"),
            RunState::new(id.into(), definition).unwrap(),
        )
        .unwrap();
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: [(
            "ask".into(),
            ActivityDescriptor {
                kind: "delegate".into(),
                payload,
            },
        )]
        .into(),
    };
    executor.install_plan(id, plan).unwrap();
}

#[test]
fn delegation_runs_one_bounded_child_and_returns_its_result() {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    let mut registry = ActivityRegistry::new();
    registry
        .register("summarize", |request| {
            Ok(format!("summary of {}", request.descriptor.payload["goal"]))
        })
        .unwrap();
    registry
        .register_delegate("delegate", actor.clone(), ["summarize".to_string()].into())
        .unwrap();
    let executor = Executor::new(actor.clone(), Arc::new(registry));
    root(
        &actor,
        &executor,
        "ok",
        serde_json::json!({"goal": "the notes", "kind": "summarize", "grants": ["read"]}),
    );
    root(
        &actor,
        &executor,
        "wider",
        serde_json::json!({"goal": "x", "kind": "summarize", "grants": ["write"]}),
    );
    root(
        &actor,
        &executor,
        "unlisted",
        serde_json::json!({"goal": "x", "kind": "delegate"}),
    );
    assert!(executor.tick("ok").unwrap());
    let waiting = actor.get("ok".into()).unwrap().unwrap();
    assert_eq!(waiting.status, RunStatus::Waiting);
    let child = waiting.waiting_child.unwrap().child_id;
    assert!(
        !executor.tick("ok").unwrap(),
        "the parent holds nothing while it waits"
    );
    while !actor.get(child.clone()).unwrap().unwrap().status.terminal() {
        assert!(executor.tick(&child).unwrap());
    }
    while !actor.get("ok".into()).unwrap().unwrap().status.terminal() {
        assert!(executor.tick("ok").unwrap());
    }
    let done = actor.get("ok".into()).unwrap().unwrap();
    assert_eq!(done.status, RunStatus::Succeeded);
    assert_eq!(done.node_results["ask"], "summary of \"the notes\"");
    for refused in ["wider", "unlisted"] {
        let _ = executor.tick(refused);
        let state = actor.get(refused.into()).unwrap().unwrap();
        assert!(state.waiting_child.is_none(), "{refused}");
        assert!(
            actor
                .get(format!("{refused}/{refused}:ask:1"))
                .unwrap()
                .is_none()
        );
    }
    actor.shutdown().unwrap();
}

#[test]
fn a_child_kind_that_delegates_or_needs_review_is_refused_at_registration() {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    let mut registry = ActivityRegistry::new();
    registry
        .register_reviewed(
            "deliver",
            60,
            |_, _| Err("unused".into()),
            |_, _, ()| Ok(String::new()),
        )
        .unwrap();
    registry
        .register_delegate("first", actor.clone(), ["summarize".to_string()].into())
        .unwrap();
    for child in ["second", "first", "deliver"] {
        assert!(
            registry
                .register_delegate("second", actor.clone(), [child.to_string()].into())
                .is_err(),
            "{child}"
        );
    }
    actor.shutdown().unwrap();
}
