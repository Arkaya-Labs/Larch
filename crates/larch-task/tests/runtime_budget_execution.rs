use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, Driver, ExecutionPlan, Executor,
    },
    graph::{Graph, Node, NodeKind},
    runtime::ActorHandle,
    types::*,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

struct Metered(Mutex<Vec<Option<u64>>>);
impl Driver for Metered {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        self.0.lock().unwrap().push(request.max_tokens);
        Ok(ActivityOutcome {
            output: "done".into(),
        })
    }
    fn enforces_budget(&self) -> bool {
        true
    }
    fn reservation(&self, _: &ActivityRequest) -> Option<(u64, usize)> {
        Some((8, 2))
    }
    fn measured_usage(
        &self,
        _: &ActivityRequest,
        _: &Result<ActivityOutcome, String>,
    ) -> Option<u64> {
        Some(5)
    }
}
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
            goal: "work".into(),
            initial_graph: Some(Graph { nodes: vec![node] }),
            planning_policy: PlanningPolicy::Disabled,
            grants: vec![],
            budgets: [("model_tokens".into(), 10)].into_iter().collect(),
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
                kind: "fake".into(),
                payload: serde_json::Value::Null,
            },
        )]
        .into_iter()
        .collect(),
    }
}
#[test]
fn children_share_root_budget_and_driver_receives_reserved_cap() {
    let path = std::env::temp_dir().join(format!(
        "larch-executor-budget-{}-{}.sqlite",
        std::process::id(),
        {
            // macOS clocks tick in microseconds: parallel tests in one binary can read the same instant.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            format!("{t}-{n}")
        }
    ));
    let actor = ActorHandle::start(&path, 4).unwrap();
    actor.create("create".into(), state("root")).unwrap();
    actor
        .spawn_child("root".into(), 0, "spawn-1".into(), state("child-1"))
        .unwrap();
    actor
        .spawn_child("root".into(), 1, "spawn-2".into(), state("child-2"))
        .unwrap();
    let parent = CommandEnvelope {
        id: "parent-start".into(),
        expected_revision: 2,
        command: Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "parent-attempt".into(),
                node_id: "node".into(),
                epoch: 1,
            },
        },
    };
    actor
        .dispatch_budgeted_once("root".into(), parent, 0, 2)
        .unwrap();
    // Budgeted public child-plan publication is not available yet. Seed matching
    // durable origins solely to exercise the shared Executor admission ledger.
    let db = rusqlite::Connection::open(&path).unwrap();
    for child in ["child-1", "child-2"] {
        db.execute(
            "INSERT INTO child_origins(parent_id,child_id,attempt_id,epoch,graph_revision,node_id) VALUES('root',?1,'parent-attempt',1,0,'node')",
            [child],
        ).unwrap();
    }
    drop(db);
    let driver = Arc::new(Metered(Mutex::new(Vec::new())));
    let executor = Executor::new(actor.clone(), driver.clone());
    executor.install_plan("child-1", plan()).unwrap();
    executor.install_plan("child-2", plan()).unwrap();
    assert!(executor.tick("child-1").unwrap());
    assert!(executor.tick("child-2").is_err());
    assert_eq!(*driver.0.lock().unwrap(), vec![Some(8)]);
    actor.shutdown().unwrap();
    let _ = std::fs::remove_file(path);
}
