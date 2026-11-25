use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, Driver, ExecutionPlan, Executor,
    },
    graph::{Graph, Node, NodeKind},
    runtime::ActorHandle,
    types::{
        Attempt, Command, CommandEnvelope, PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION,
        WorkflowDefinition,
    },
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// Fails its first `fails` attempts with `reason`; retries "busy" up to twice after `delay`, never
/// "denied". With `actor`, each attempt records an effect's intent first, which a failure leaves
/// unresolved.
struct Flaky {
    fails: usize,
    reason: &'static str,
    delay: Duration,
    starts: Mutex<Vec<Instant>>,
    actor: Option<ActorHandle>,
}
impl Flaky {
    fn calls(&self) -> usize {
        self.starts.lock().unwrap().len()
    }
}
impl Driver for Flaky {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        if let Some(actor) = &self.actor {
            let revision = actor.get("run".into())?.ok_or("no run")?.revision;
            actor.execute(
                "run".into(),
                CommandEnvelope {
                    id: format!("intent:{}", request.attempt_id),
                    expected_revision: revision,
                    command: Command::RecordIntent {
                        effect_id: format!("send:{}", request.attempt_id),
                        attempt_id: request.attempt_id.clone(),
                        epoch: request.epoch,
                        operation_digest: "digest".into(),
                    },
                },
            )?;
        }
        let mut starts = self.starts.lock().unwrap();
        starts.push(Instant::now());
        match starts.len() <= self.fails {
            true => Err(self.reason.into()),
            false => Ok(ActivityOutcome {
                output: "done".into(),
            }),
        }
    }
    fn retry_delay(&self, _: &ActivityRequest, reason: &str, failures: u32) -> Option<Duration> {
        (reason == "busy" && failures < 2).then_some(self.delay)
    }
}

/// One activity node "work" with `driver` built over the new actor.
fn start(driver: impl FnOnce(&ActorHandle) -> Flaky) -> (ActorHandle, Executor, Arc<Flaky>) {
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
        acceptance: vec![],
    };
    let state = RunState::new("run".into(), definition).unwrap();
    actor.create("create".into(), state).unwrap();
    let driver = Arc::new(driver(&actor));
    let executor = Executor::new(actor.clone(), driver.clone());
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: [(
            "work".into(),
            ActivityDescriptor {
                kind: "flaky".into(),
                payload: serde_json::Value::Null,
            },
        )]
        .into(),
    };
    executor.install_plan("run", plan).unwrap();
    (actor, executor, driver)
}

fn flaky(fails: usize, reason: &'static str, delay: Duration) -> Flaky {
    Flaky {
        fails,
        reason,
        delay,
        starts: Mutex::new(vec![]),
        actor: None,
    }
}

fn run(fails: usize, reason: &'static str) -> (RunState, usize) {
    run_with(fails, reason, false)
}

fn run_with(fails: usize, reason: &'static str, effect: bool) -> (RunState, usize) {
    let (actor, executor, driver) = start(|actor| Flaky {
        actor: effect.then(|| actor.clone()),
        ..flaky(fails, reason, Duration::ZERO)
    });
    for _ in 0..8 {
        let state = actor.get("run".into()).unwrap().unwrap();
        if state.status != RunStatus::Ready && state.status != RunStatus::Running {
            break;
        }
        if !executor.tick("run").unwrap() {
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    let state = actor.get("run".into()).unwrap().unwrap();
    (state, driver.calls())
}

fn control(actor: &ActorHandle, id: &str, command: Command) {
    let revision = actor.get("run".into()).unwrap().unwrap().revision;
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: id.into(),
                expected_revision: revision,
                command,
            },
        )
        .unwrap();
}

#[test]
fn a_retryable_failure_runs_again_after_its_backoff_and_succeeds() {
    let (state, calls) = run(2, "busy");
    assert_eq!(state.status, RunStatus::Succeeded);
    assert_eq!(calls, 3);
    assert_eq!(state.retry, None);
}

#[test]
fn a_used_up_policy_or_a_final_error_parks_the_run_for_the_user() {
    let (state, calls) = run(5, "busy");
    assert_eq!((state.status, calls), (RunStatus::Waiting, 3));
    assert_eq!(state.attempt_failure.as_deref(), Some("busy"));
    let (state, calls) = run(5, "denied");
    assert_eq!((state.status, calls), (RunStatus::Waiting, 1));
}

#[test]
fn a_failure_that_may_have_sent_something_is_never_retried() {
    let (state, calls) = run_with(5, "busy", true);
    assert_eq!((state.status, calls), (RunStatus::Waiting, 1));
    assert_eq!(state.retry, None);
}

#[test]
fn the_next_attempt_never_starts_before_the_backoff_is_over() {
    // Sub-second and off the second boundary, so a whole-second clock would show it short.
    let delay = Duration::from_millis(1300);
    let (_actor, executor, driver) = start(|_| flaky(1, "busy", delay));
    let end = Instant::now() + Duration::from_secs(5);
    while driver.calls() < 2 && Instant::now() < end {
        executor.tick("run").unwrap();
        std::thread::sleep(Duration::from_millis(2));
    }
    let starts = driver.starts.lock().unwrap();
    assert_eq!(starts.len(), 2);
    assert!(
        starts[1] - starts[0] >= delay,
        "{:?}",
        starts[1] - starts[0]
    );
}

#[test]
fn pausing_in_the_backoff_keeps_the_retry_budget() {
    let (actor, executor, driver) = start(|_| flaky(9, "busy", Duration::from_secs(30)));
    for round in 0..3 {
        executor.tick("run").unwrap();
        assert_eq!(
            actor.get("run".into()).unwrap().unwrap().status,
            RunStatus::Ready
        );
        control(&actor, &format!("pause{round}"), Command::Pause);
        control(&actor, &format!("resume{round}"), Command::Resume);
    }
    let state = actor.get("run".into()).unwrap().unwrap();
    assert_eq!((driver.calls(), state.retry.unwrap().failures), (1, 1));
}

#[test]
fn a_backoff_past_any_driver_wait_is_due_after_the_clock_steps_back() {
    let (actor, executor, driver) = start(|_| flaky(0, "busy", Duration::ZERO));
    // A retry stored ten days out: what an hour's backoff looks like after the clock steps back.
    let attempt = Attempt {
        iteration: None,
        id: "first".into(),
        node_id: "work".into(),
        epoch: 1,
    };
    control(&actor, "start", Command::StartAttempt { attempt });
    let ten_days = std::time::SystemTime::now() + Duration::from_secs(10 * 86_400);
    let not_before = ten_days.duration_since(std::time::UNIX_EPOCH).unwrap();
    let retry = Command::RetryAttempt {
        attempt_id: "first".into(),
        epoch: 1,
        reason: "busy".into(),
        not_before_ms: not_before.as_millis() as u64,
    };
    control(&actor, "retry", retry);
    assert!(executor.tick("run").unwrap());
    assert_eq!(driver.calls(), 1);
    executor.tick("run").unwrap();
    assert_eq!(
        actor.get("run".into()).unwrap().unwrap().status,
        RunStatus::Succeeded
    );
}
