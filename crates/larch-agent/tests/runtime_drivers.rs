use larch_agent::runtime::{
    children::{ChildRequest, ChildResult, validate_result, validate_spawn},
    context::ContextPacket,
    planner,
};
use larch_task::{
    graph::{Graph, Node, NodeKind},
    types::{PlanningPolicy, SCHEMA_VERSION, WorkflowDefinition},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::AtomicBool,
};

fn definition() -> WorkflowDefinition {
    WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "keep goal".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Adaptive,
        grants: vec!["read".into()],
        budgets: BTreeMap::new(),
        acceptance: vec!["check".into()],
    }
}
fn graph(id: &str) -> Graph {
    Graph {
        nodes: vec![Node {
            when: None,
            optional: Vec::new(),
            id: id.into(),
            depends_on: vec![],
            kind: NodeKind::Activity,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }],
    }
}

#[test]
fn context_preserves_authoritative_text_or_fails() {
    let def = definition();
    let empty = ContextPacket::fit(&def, &[], &[], usize::MAX, str::len).unwrap();
    let cap = empty.render().len();
    let packet = ContextPacket::fit(
        &def,
        &["artifact".into()],
        &["history".into()],
        cap,
        str::len,
    )
    .unwrap();
    assert_eq!(packet.goal, def.goal);
    assert_eq!(packet.acceptance, def.acceptance);
    assert!(packet.history.is_empty());
    assert!(packet.artifacts.is_empty());
    assert!(ContextPacket::fit(&def, &[], &[], cap - 1, str::len).is_err());
}

#[test]
fn planner_enforces_boundaries_and_immutable_started_nodes() {
    let def = definition();
    let old = graph("first");
    assert!(planner::initial(&def, old.clone(), 1).is_ok());
    let started = BTreeSet::from(["first".to_string()]);
    assert!(planner::revision(&def, &old, graph("second"), true, 1, 1, &started, 1).is_err());
    assert!(planner::revision(&def, &old, old.clone(), false, 1, 1, &started, 1).is_err());
    assert!(planner::revision(&def, &old, old.clone(), true, 0, 1, &started, 1).is_err());
    assert!(planner::revision(&def, &old, old.clone(), true, 1, 1, &started, 1).is_ok());
}

#[test]
fn child_authority_and_results_are_fenced() {
    let grants = BTreeSet::from(["read".to_string()]);
    let mut child = ChildRequest {
        child_id: "child".into(),
        root_id: "root".into(),
        graph_revision: 2,
        depth: 1,
        goal: "work".into(),
        grants: grants.clone(),
    };
    assert!(validate_spawn(&child, "root", 0, 0, 4, &grants).is_ok());
    assert!(validate_spawn(&child, "another-root", 0, 0, 4, &grants).is_err());
    child.grants.insert("write".into());
    assert!(validate_spawn(&child, "root", 0, 0, 4, &grants).is_err());
    let result = ChildResult {
        child_id: "child".into(),
        graph_revision: 2,
        summary: "done".into(),
        evidence: vec!["artifact".into()],
    };
    assert!(validate_result(&result, "child", 3).is_err());
    assert!(validate_result(&result, "child", 2).is_ok());
}

#[test]
fn shared_admission_enforces_capacity_and_releases_on_drop() {
    use larch_agent::runtime::inference::{Admission, AdmissionPermit, SharedAdmission};
    use std::sync::{Arc, Mutex};
    let dir = tempfile::tempdir().unwrap();
    let mut store = larch_task::store::SqliteStore::open(dir.path().join("tasks.sqlite")).unwrap();
    store.create_admission_root("root", 100).unwrap();
    let store = Arc::new(Mutex::new(store));
    let admission = SharedAdmission::new(store, 1, 10, "test-instance".into()).unwrap();
    let stop = AtomicBool::new(false);
    let first = admission.acquire("root", 10, &stop).unwrap();
    assert!(admission.clone().acquire("root", 10, &stop).is_err());
    first.settle(3).unwrap();
    let second = admission.acquire("root", 10, &stop).unwrap();
    drop(second);
    assert!(admission.acquire("root", 10, &stop).is_ok());
}

#[test]
fn child_adapter_fences_parent_revision_and_persists_idempotent_message() {
    use larch_agent::runtime::children::{message, spawn};
    use larch_task::{runtime::ActorHandle, types::RunState};
    let dir = tempfile::tempdir().unwrap();
    let actor = ActorHandle::start(dir.path().join("children.sqlite"), 8).unwrap();
    let mut parent_def = definition();
    parent_def.initial_graph = Some(graph("parent-step"));
    actor
        .create(
            "create".into(),
            RunState::new("root".into(), parent_def.clone()).unwrap(),
        )
        .unwrap();
    let mut child_def = parent_def;
    child_def.goal = "child goal".into();
    let child = RunState::new("child".into(), child_def).unwrap();
    let request = ChildRequest {
        child_id: "child".into(),
        root_id: "root".into(),
        graph_revision: 0,
        depth: 1,
        goal: "child goal".into(),
        grants: BTreeSet::from(["read".into()]),
    };
    assert!(spawn(&actor, "root", 1, "bad".into(), &request, child.clone()).is_err());
    spawn(&actor, "root", 0, "spawn".into(), &request, child.clone()).unwrap();
    spawn(&actor, "root", 0, "spawn".into(), &request, child).unwrap();
    message(&actor, "root", "child", "message", "data").unwrap();
    message(&actor, "root", "child", "message", "data").unwrap();
    assert!(message(&actor, "root", "child", "message", "different").is_err());
}

#[test]
fn model_seats_cancel_waiters_and_release_on_drop() {
    use larch_agent::runtime::inference::{Admission, SeatAdmission};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    let admission = SeatAdmission::new(1).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let permit = admission.acquire("root-a", 10, &stop).unwrap();
    let other = admission.clone();
    let cancelled = stop.clone();
    let waiter = std::thread::spawn(move || other.acquire("root-b", 10, &cancelled).is_err());
    stop.store(true, Ordering::Release);
    assert!(waiter.join().unwrap());
    drop(permit);
    let ready = AtomicBool::new(false);
    assert!(admission.acquire("root-b", 10, &ready).is_ok());
}
