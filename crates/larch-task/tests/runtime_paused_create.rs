use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, Driver, ExecutionPlan, Executor,
    },
    graph::{Graph, Node, NodeKind},
    runtime::ActorHandle,
    types::*,
};
use std::{collections::BTreeMap, sync::Arc};
struct Fake;
impl Driver for Fake {
    fn execute(&self, _: ActivityRequest) -> Result<ActivityOutcome, String> {
        Ok(ActivityOutcome {
            output: "done".into(),
        })
    }
}
#[test]
fn saved_run_is_paused_until_explicit_start() {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    let node = Node {
        when: None,
        optional: Vec::new(),
        id: "node".into(),
        depends_on: vec![],
        kind: NodeKind::Activity,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
    };
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "work".into(),
        initial_graph: Some(Graph { nodes: vec![node] }),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let state = RunState::new("run".into(), definition).unwrap();
    let first = actor.create_paused("save".into(), state.clone()).unwrap();
    assert_eq!(first.status, RunStatus::Paused);
    assert_eq!(actor.create_paused("save".into(), state).unwrap(), first);
    let executor = Executor::new(actor.clone(), Arc::new(Fake));
    executor
        .install_plan(
            "run",
            ExecutionPlan {
                graph_revision: 0,
                activities: [(
                    "node".into(),
                    ActivityDescriptor {
                        kind: "test".into(),
                        payload: serde_json::Value::Null,
                    },
                )]
                .into_iter()
                .collect(),
            },
        )
        .unwrap();
    assert!(!executor.tick("run").unwrap());
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: "start".into(),
                expected_revision: 0,
                command: Command::Resume,
            },
        )
        .unwrap();
    assert!(executor.tick("run").unwrap());
    assert!(executor.tick("run").unwrap());
    assert_eq!(
        actor.get("run".into()).unwrap().unwrap().status,
        RunStatus::Succeeded
    );
    actor.shutdown().unwrap();
}
#[test]
fn stale_start_plan_does_not_install_immutable_plan() {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    let node = Node {
        when: None,
        optional: Vec::new(),
        id: "node".into(),
        depends_on: vec![],
        kind: NodeKind::Activity,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
    };
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "work".into(),
        initial_graph: Some(Graph { nodes: vec![node] }),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    actor
        .create_paused(
            "save".into(),
            RunState::new("run".into(), definition).unwrap(),
        )
        .unwrap();
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: [(
            "node".into(),
            ActivityDescriptor {
                kind: "test".into(),
                payload: serde_json::Value::Null,
            },
        )]
        .into_iter()
        .collect(),
    };
    assert!(
        actor
            .start_plan("run".into(), "stale".into(), 99, plan.clone())
            .is_err()
    );
    assert!(actor.execution_plan("run".into()).unwrap().is_none());
    let first = actor
        .start_plan("run".into(), "start".into(), 0, plan.clone())
        .unwrap();
    assert_eq!(
        actor
            .start_plan("run".into(), "start".into(), 0, plan)
            .unwrap(),
        first
    );
    assert_eq!(
        actor.get("run".into()).unwrap().unwrap().status,
        RunStatus::Ready
    );
    actor.shutdown().unwrap();
}
