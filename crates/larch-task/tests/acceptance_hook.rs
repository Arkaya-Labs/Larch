use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, Driver, ExecutionPlan, Executor,
    },
    graph::{Graph, Node, NodeKind},
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

/// Answers "done"; its check passes, fails or is missing as `verdict` says.
struct Checked(Mutex<Option<Result<String, String>>>);
impl Driver for Checked {
    fn execute(&self, _: ActivityRequest) -> Result<ActivityOutcome, String> {
        Ok(ActivityOutcome {
            output: "done".into(),
        })
    }
    fn accept(&self, state: &RunState, criterion: &str) -> Option<Result<String, String>> {
        assert_eq!(
            (criterion, state.result.as_deref()),
            ("cites a source", Some("done"))
        );
        self.0.lock().unwrap().clone()
    }
}

/// A one-step run that is done when it "cites a source", checked by the returned driver.
fn start(verdict: Option<Result<String, String>>) -> (ActorHandle, Executor, Arc<Checked>) {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    let node = Node {
        when: None,
        optional: Vec::new(),
        id: "work".into(),
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
        acceptance: vec!["cites a source".into()],
    };
    let state = RunState::new("run".into(), definition).unwrap();
    actor.create("create".into(), state).unwrap();
    let driver = Arc::new(Checked(Mutex::new(verdict)));
    let executor = Executor::new(actor.clone(), driver.clone());
    let descriptor = ActivityDescriptor {
        kind: "work".into(),
        payload: serde_json::Value::Null,
    };
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: [("work".into(), descriptor)].into(),
    };
    executor.install_plan("run", plan).unwrap();
    (actor, executor, driver)
}

fn settle(actor: &ActorHandle, executor: &Executor) -> RunState {
    for _ in 0..4 {
        executor.tick("run").unwrap();
    }
    actor.get("run".into()).unwrap().unwrap()
}

/// The run settled under `verdict`.
fn run(verdict: Option<Result<String, String>>) -> RunState {
    let (actor, executor, _driver) = start(verdict);
    settle(&actor, &executor)
}

#[test]
fn a_settled_run_finishes_only_once_the_hosts_check_passes() {
    let passed = run(Some(Ok("source [1] is in the journal".into())));
    assert_eq!(passed.status, RunStatus::Succeeded);
    assert_eq!(
        passed.acceptance_receipts["cites a source"],
        "source [1] is in the journal"
    );
    let failed = run(Some(Err("no source named".into())));
    assert_eq!(failed.status, RunStatus::Waiting);
    assert_eq!(
        failed.attempt_failure.as_deref(),
        Some("not done yet (cites a source): no source named")
    );
    // No check here: the run waits for one, as before.
    assert_eq!(run(None).status, RunStatus::Ready);
}

#[test]
fn a_blank_verdict_still_parks_the_run_saying_why() {
    for (verdict, why) in [
        (Ok(" ".to_string()), "the check gave no receipt"),
        (Err("  ".to_string()), "the check gave no reason"),
    ] {
        let state = run(Some(verdict));
        assert_eq!(state.status, RunStatus::Waiting);
        let failure = format!("not done yet (cites a source): {why}");
        assert_eq!(state.attempt_failure, Some(failure));
    }
}

#[test]
fn a_check_that_passes_after_a_rejection_leaves_no_failure_behind() {
    let (actor, executor, driver) = start(Some(Err("no source named".into())));
    assert_eq!(settle(&actor, &executor).status, RunStatus::Waiting);
    *driver.0.lock().unwrap() = Some(Ok("source [1]".into()));
    let revision = actor.get("run".into()).unwrap().unwrap().revision;
    let resume = CommandEnvelope {
        id: "resume".into(),
        expected_revision: revision,
        command: Command::Resume,
    };
    actor.execute("run".into(), resume).unwrap();
    let state = settle(&actor, &executor);
    assert_eq!(
        (state.status, state.attempt_failure),
        (RunStatus::Succeeded, None)
    );
}
