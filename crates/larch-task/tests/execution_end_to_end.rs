use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, Driver, ExecutionPlan, Executor,
    },
    graph::{Graph, Node, NodeKind},
    runtime::ActorHandle,
    types::{PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION, WorkflowDefinition},
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

struct ScriptedDriver(Mutex<Vec<ActivityRequest>>);
impl Driver for ScriptedDriver {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        let output = match request.node_id.as_str() {
            "first" => "first result".to_owned(),
            "second" if request.inputs.get("first").map(String::as_str) == Some("first result") => {
                "second result".to_owned()
            }
            _ => return Err("unexpected activity or input".into()),
        };
        self.0.lock().unwrap().push(request);
        Ok(ActivityOutcome { output })
    }
    fn step_tokens(&self, request: &ActivityRequest) -> Option<(u64, u64)> {
        (request.node_id == "second").then_some((120, 30))
    }
}
fn definition() -> WorkflowDefinition {
    let node = |id: &str, dependencies: Vec<String>| Node {
        when: None,
        optional: Vec::new(),
        id: id.into(),
        depends_on: dependencies,
        kind: NodeKind::Activity,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
    };
    WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "execute both steps".into(),
        initial_graph: Some(Graph {
            nodes: vec![node("first", vec![]), node("second", vec!["first".into()])],
        }),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    }
}
fn plan() -> ExecutionPlan {
    let activity = || ActivityDescriptor {
        kind: "scripted".into(),
        payload: serde_json::json!({"version": 1}),
    };
    ExecutionPlan {
        graph_revision: 0,
        activities: [("first".into(), activity()), ("second".into(), activity())]
            .into_iter()
            .collect(),
    }
}
#[test]
fn persisted_graph_runs_in_order_and_passes_committed_outputs() {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    actor
        .create(
            "create".into(),
            RunState::new("run".into(), definition()).unwrap(),
        )
        .unwrap();
    let driver = Arc::new(ScriptedDriver(Mutex::new(Vec::new())));
    let executor = Executor::new(actor.clone(), driver.clone());
    executor.install_plan("run", plan()).unwrap();
    assert!(executor.tick("run").unwrap());
    assert!(executor.tick("run").unwrap());
    assert!(executor.tick("run").unwrap());
    let state = actor.get("run".into()).unwrap().unwrap();
    assert_eq!(state.status, RunStatus::Succeeded);
    assert_eq!(
        state.node_results.get("first").map(String::as_str),
        Some("first result")
    );
    assert_eq!(
        state.node_results.get("second").map(String::as_str),
        Some("second result")
    );
    let calls = driver.0.lock().unwrap();
    assert_eq!(
        calls
            .iter()
            .map(|call| call.node_id.as_str())
            .collect::<Vec<_>>(),
        vec!["first", "second"]
    );
    assert_eq!(
        calls[1].inputs.get("first").map(String::as_str),
        Some("first result")
    );
    drop(calls);
    // Each attempt leaves its wall time, and its tokens when the driver counts them.
    let stats = actor.step_stats("run".into()).unwrap();
    assert_eq!(
        stats
            .iter()
            .map(|stat| (stat.node_id.as_str(), stat.tokens))
            .collect::<Vec<_>>(),
        vec![("first", None), ("second", Some((120, 30)))]
    );
    assert!(stats.iter().all(|stat| stat.started_at <= stat.finished_at));
    actor.shutdown().unwrap();
}
#[test]
fn execution_plan_must_cover_exact_graph() {
    let actor = ActorHandle::start(":memory:", 2).unwrap();
    actor
        .create(
            "create".into(),
            RunState::new("run".into(), definition()).unwrap(),
        )
        .unwrap();
    let executor = Executor::new(
        actor.clone(),
        Arc::new(ScriptedDriver(Mutex::new(Vec::new()))),
    );
    let mut incomplete = plan();
    incomplete.activities.remove("second");
    assert!(executor.install_plan("run", incomplete).is_err());
    assert_eq!(actor.get("run".into()).unwrap().unwrap().revision, 0);
    actor.shutdown().unwrap();
}
#[test]
fn recovery_fences_old_attempt_and_retries_only_interrupted_node() {
    use larch_task::types::{Attempt, Command, CommandEnvelope};
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    actor
        .create(
            "create".into(),
            RunState::new("run".into(), definition()).unwrap(),
        )
        .unwrap();
    let driver = Arc::new(ScriptedDriver(Mutex::new(Vec::new())));
    let executor = Executor::new(actor.clone(), driver.clone());
    executor.install_plan("run", plan()).unwrap();
    let start = CommandEnvelope {
        id: "start-old".into(),
        expected_revision: 0,
        command: Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "old".into(),
                node_id: "first".into(),
                epoch: 1,
            },
        },
    };
    actor.execute("run".into(), start).unwrap();
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: "recover".into(),
                expected_revision: 1,
                command: Command::Recover,
            },
        )
        .unwrap();
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: "resume".into(),
                expected_revision: 2,
                command: Command::Resume,
            },
        )
        .unwrap();
    assert!(
        actor
            .execute(
                "run".into(),
                CommandEnvelope {
                    id: "stale".into(),
                    expected_revision: 3,
                    command: Command::FinishAttempt {
                        attempt_id: "old".into(),
                        epoch: 1,
                        result: "stale output".into(),
                    },
                },
            )
            .is_err()
    );
    assert!(executor.tick("run").unwrap());
    assert!(executor.tick("run").unwrap());
    assert!(executor.tick("run").unwrap());
    let calls = driver.0.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert!(calls[0].epoch > 1);
    assert_eq!(
        actor
            .get("run".into())
            .unwrap()
            .unwrap()
            .node_results
            .get("first")
            .map(String::as_str),
        Some("first result")
    );
    drop(calls);
    actor.shutdown().unwrap();
}
