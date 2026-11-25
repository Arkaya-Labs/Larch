use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, ChildSpawnRequest, Driver,
        ExecutionPlan, Executor,
    },
    graph::{Graph, Node, NodeKind},
    runtime::ActorHandle,
    types::*,
};
use std::{collections::BTreeMap, sync::Arc};

fn state(id: &str) -> RunState {
    let node = Node {
        when: None,
        optional: Vec::new(),
        id: "node".into(),
        depends_on: vec![],
        kind: NodeKind::Activity,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
    };
    RunState::new(
        id.into(),
        WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "delegate".into(),
            initial_graph: Some(Graph { nodes: vec![node] }),
            planning_policy: PlanningPolicy::Disabled,
            grants: vec![],
            budgets: BTreeMap::new(),
            acceptance: vec![],
        },
    )
    .unwrap()
}
fn plan() -> ExecutionPlan {
    ExecutionPlan {
        graph_revision: 0,
        activities: [(
            "node".into(),
            ActivityDescriptor {
                kind: "child-test".into(),
                payload: serde_json::Value::Null,
            },
        )]
        .into_iter()
        .collect(),
    }
}
struct Delegate(ActorHandle);
impl Driver for Delegate {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        if request.run_id == "root" {
            let parent = self.0.get("root".into())?.ok_or("missing parent")?;
            self.0.spawn_child_and_wait(ChildSpawnRequest {
                parent_id: request.run_id.clone(),
                expected_revision: parent.revision,
                command_id: "spawn".into(),
                parent_attempt_id: request.attempt_id.clone(),
                parent_epoch: request.epoch,
                expected_graph_revision: request.graph_revision,
                child: state("child"),
                plan: plan(),
            })?;
            Ok(ActivityOutcome {
                output: String::new(),
            })
        } else {
            Ok(ActivityOutcome {
                output: "child result".into(),
            })
        }
    }
    fn suspended_child(
        &self,
        request: &ActivityRequest,
        _: &Result<ActivityOutcome, String>,
    ) -> Option<String> {
        (request.run_id == "root").then(|| "child".into())
    }
}
#[test]
fn parent_wait_releases_driver_and_wakes_only_on_settled_child() {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    actor.create("create".into(), state("root")).unwrap();
    let executor = Executor::new(actor.clone(), Arc::new(Delegate(actor.clone())));
    executor.install_plan("root", plan()).unwrap();
    assert!(executor.tick("root").unwrap());
    let waiting = actor.get("root".into()).unwrap().unwrap();
    assert_eq!(waiting.status, RunStatus::Waiting);
    assert_eq!(waiting.revision, 2);
    let wait = waiting.waiting_child.unwrap();
    assert_eq!(wait.child_id, "child");
    assert_eq!(wait.node_id, "node");
    assert_eq!(wait.graph_revision, 0);
    assert_eq!(wait.origin_attempt_id.as_deref(), Some("root:node:1"));
    assert_eq!(wait.origin_epoch, Some(1));
    assert_eq!(actor.events_after("root".into(), 0, 100).unwrap().len(), 2);
    assert_eq!(actor.execution_plan("child".into()).unwrap(), Some(plan()));
    assert!(!executor.tick("root").unwrap());
    assert!(executor.tick("child").unwrap());
    assert!(executor.tick("child").unwrap());
    assert!(executor.tick("root").unwrap());
    assert!(executor.tick("root").unwrap());
    let root = actor.get("root".into()).unwrap().unwrap();
    assert_eq!(root.status, RunStatus::Succeeded);
    assert_eq!(
        root.node_results.get("node").map(String::as_str),
        Some("child result")
    );
    actor.shutdown().unwrap();
}
