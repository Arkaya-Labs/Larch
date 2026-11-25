use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, Cancellation, Driver, ExecutionPlan,
        Executor,
    },
    graph::{Graph, Node, NodeKind},
    runtime::ActorHandle,
    types::*,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, mpsc},
    time::Duration,
};

struct Blocking {
    ready: mpsc::Sender<Cancellation>,
    release: Mutex<mpsc::Receiver<()>>,
}
impl Driver for Blocking {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        self.ready.send(request.cancellation.clone()).unwrap();
        self.release
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        Ok(ActivityOutcome {
            output: "late".into(),
        })
    }
}
#[test]
fn cancellation_is_observable_and_late_result_is_fenced() {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    let node = Node {
        when: None,
        optional: Vec::new(),
        id: "n".into(),
        depends_on: vec![],
        kind: NodeKind::Activity,
        inputs: BTreeMap::new(),
        outputs: BTreeMap::new(),
    };
    let def = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "work".into(),
        initial_graph: Some(Graph { nodes: vec![node] }),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    actor
        .create("create".into(), RunState::new("run".into(), def).unwrap())
        .unwrap();
    let (ready_send, ready_recv) = mpsc::channel();
    let (release_send, release_recv) = mpsc::channel();
    let driver = Arc::new(Blocking {
        ready: ready_send,
        release: Mutex::new(release_recv),
    });
    let executor = Executor::new(actor.clone(), driver);
    executor
        .install_plan(
            "run",
            ExecutionPlan {
                graph_revision: 0,
                activities: [(
                    "n".into(),
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
    let thread = std::thread::spawn(move || executor.tick("run"));
    let cancellation = ready_recv.recv_timeout(Duration::from_secs(5)).unwrap();
    let state = actor.get("run".into()).unwrap().unwrap();
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: "cancel".into(),
                expected_revision: state.revision,
                command: Command::Cancel,
            },
        )
        .unwrap();
    assert!(cancellation.is_cancelled());
    release_send.send(()).unwrap();
    assert!(thread.join().unwrap().is_err());
    let state = actor.get("run".into()).unwrap().unwrap();
    assert_eq!(state.status, RunStatus::Cancelled);
    assert!(state.node_results.is_empty());
    actor.shutdown().unwrap();
}
