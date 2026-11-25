//! Review mode (architecture 8, 15, appendix A decision transaction): an activity that prepared
//! an action waits for the user's answer on its exact digest; expiry, a different digest or a
//! revocation since the request refuse it; only the host may stamp the response.
use larch_task::{
    execution::{
        ActivityDescriptor, ActivityOutcome, ActivityRequest, DecisionRequest, Driver,
        ExecutionPlan, Executor,
    },
    graph::{Graph, Node, NodeKind},
    host::{self, Action},
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

const DIGEST: &str = "sha256:patch";

/// Prepares a patch, asks for approval, and applies only under an approval of its digest.
/// The first request expires at the second field; later ones never do.
struct Reviewer(Mutex<Vec<usize>>, u64);
impl Driver for Reviewer {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        self.0.lock().unwrap().push(request.approvals.len());
        let applied = request
            .approvals
            .values()
            .any(|d| d.action_digest == DIGEST);
        Ok(ActivityOutcome {
            output: if applied { "applied" } else { "prepared" }.into(),
        })
    }
    fn decision_request(
        &self,
        request: &ActivityRequest,
        outcome: &Result<ActivityOutcome, String>,
    ) -> Option<DecisionRequest> {
        (request.approvals.is_empty() && outcome.is_ok()).then(|| DecisionRequest {
            request_id: format!("apply:{}", request.attempt_id),
            action_digest: DIGEST.into(),
            scope: "repo: src/lib.rs".into(),
            expires_at: if self.0.lock().unwrap().len() == 1 {
                self.1
            } else {
                u64::MAX
            },
        })
    }
}

fn start(grants: Vec<String>) -> (ActorHandle, Executor, Arc<Reviewer>) {
    start_expiring(grants, u64::MAX)
}
fn start_expiring(grants: Vec<String>, expires_at: u64) -> (ActorHandle, Executor, Arc<Reviewer>) {
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "review a patch".into(),
        initial_graph: Some(Graph {
            nodes: vec![Node {
                id: "deliver".into(),
                depends_on: vec![],
                kind: NodeKind::Activity,
                inputs: BTreeMap::new(),
                outputs: BTreeMap::new(),
                when: None,
                optional: vec![],
            }],
        }),
        planning_policy: PlanningPolicy::Disabled,
        grants,
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
    let driver = Arc::new(Reviewer(Mutex::new(Vec::new()), expires_at));
    let executor = Executor::new(actor.clone(), driver.clone());
    executor
        .install_plan(
            "run",
            ExecutionPlan {
                graph_revision: 0,
                activities: [(
                    "deliver".into(),
                    ActivityDescriptor {
                        kind: "code.deliver".into(),
                        payload: serde_json::json!({}),
                    },
                )]
                .into_iter()
                .collect(),
            },
        )
        .unwrap();
    (actor, executor, driver)
}
fn state(actor: &ActorHandle) -> RunState {
    actor.get("run".into()).unwrap().unwrap()
}
fn respond(actor: &ActorHandle, digest: &str, approve: bool, now: u64) -> Result<(), String> {
    let s = state(actor);
    let request_id = s.waiting_decision.clone().ok_or("nothing pending")?;
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: format!("respond:{}", s.revision),
                expected_revision: s.revision,
                command: Command::RespondDecision {
                    request_id,
                    action_digest: digest.into(),
                    approve,
                    now,
                },
            },
        )
        .map(drop)
}

#[test]
fn approval_of_the_exact_digest_reruns_the_step_with_the_approval() {
    let (actor, executor, driver) = start(vec![]);
    assert!(executor.tick("run").unwrap());
    let waiting = state(&actor);
    assert_eq!(waiting.status, RunStatus::Waiting);
    let request = waiting.waiting_decision.clone().unwrap();
    assert_eq!(waiting.decisions[&request].scope, "repo: src/lib.rs");
    assert!(
        !executor.tick("run").unwrap(),
        "a pending decision admits no work"
    );
    assert!(respond(&actor, "sha256:other", true, 0).is_err());
    respond(&actor, DIGEST, true, 0).unwrap();
    assert!(executor.tick("run").unwrap());
    assert!(executor.tick("run").unwrap());
    let done = state(&actor);
    assert_eq!(done.status, RunStatus::Succeeded);
    assert_eq!(done.node_results["deliver"], "applied");
    assert_eq!(*driver.0.lock().unwrap(), [0, 1]);
    actor.shutdown().unwrap();
}

#[test]
fn rejection_settles_the_step_without_running_it_again() {
    let (actor, executor, driver) = start(vec![]);
    executor.tick("run").unwrap();
    respond(&actor, DIGEST, false, 0).unwrap();
    assert!(executor.tick("run").unwrap());
    let done = state(&actor);
    assert_eq!(done.status, RunStatus::Succeeded);
    let result: serde_json::Value = serde_json::from_str(&done.node_results["deliver"]).unwrap();
    assert_eq!(result["decision"], "rejected");
    assert_eq!(driver.0.lock().unwrap().len(), 1);
    actor.shutdown().unwrap();
}

#[test]
fn expiry_and_revocation_since_the_request_refuse_an_answer() {
    let (actor, executor, _) = start(vec!["code.apply".into()]);
    executor.tick("run").unwrap();
    assert!(respond(&actor, DIGEST, true, u64::MAX).is_err());
    let s = state(&actor);
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: "revoke".into(),
                expected_revision: s.revision,
                command: Command::RevokeGrant {
                    grant: "code.apply".into(),
                },
            },
        )
        .unwrap();
    assert!(respond(&actor, DIGEST, true, 0).is_err());
    assert_eq!(state(&actor).status, RunStatus::Waiting);
    actor.shutdown().unwrap();
}

#[test]
fn pause_and_resume_keep_the_decision_pending() {
    let (actor, executor, _) = start(vec![]);
    executor.tick("run").unwrap();
    for (id, command) in [("pause", Command::Pause), ("resume", Command::Resume)] {
        let s = state(&actor);
        actor
            .execute(
                "run".into(),
                CommandEnvelope {
                    id: id.into(),
                    expected_revision: s.revision,
                    command,
                },
            )
            .unwrap();
    }
    let s = state(&actor);
    assert_eq!(s.status, RunStatus::Waiting);
    assert!(s.waiting_decision.is_some());
    respond(&actor, DIGEST, true, 0).unwrap();
    actor.shutdown().unwrap();
}

#[test]
fn clients_answer_only_through_the_host_which_stamps_the_time() {
    let (actor, executor, _) = start(vec![]);
    executor.tick("run").unwrap();
    let s = state(&actor);
    let request_id = s.waiting_decision.clone().unwrap();
    let forged = Action::Control {
        run_id: "run".into(),
        command: CommandEnvelope {
            id: "forged".into(),
            expected_revision: s.revision,
            command: Command::RespondDecision {
                request_id: request_id.clone(),
                action_digest: DIGEST.into(),
                approve: true,
                now: 0,
            },
        },
    };
    let (reply, _) = host::handle(&actor, forged, true, false);
    assert!(reply.error.is_some());
    let (reply, _) = host::handle(
        &actor,
        Action::RespondDecision {
            run_id: "run".into(),
            command_id: "answer".into(),
            expected_revision: s.revision,
            request_id,
            action_digest: DIGEST.into(),
            approve: true,
        },
        true,
        false,
    );
    assert!(reply.error.is_none(), "{:?}", reply.error);
    assert_eq!(state(&actor).status, RunStatus::Ready);
    actor.shutdown().unwrap();
}

fn control(actor: &ActorHandle, id: &str, command: Command) {
    let s = state(actor);
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: id.into(),
                expected_revision: s.revision,
                command,
            },
        )
        .unwrap();
}
/// A lapsed request is retired unanswered, the step asks again under a fresh id, and only the
/// fresh approval applies the action.
fn retire_and_ask_again(actor: &ActorHandle, executor: &Executor, driver: &Reviewer) {
    let stale = state(actor).waiting_decision.clone().unwrap();
    assert!(respond(actor, DIGEST, true, 2).is_err());
    assert_eq!(
        state(actor).waiting_decision.as_deref(),
        Some(stale.as_str())
    );
    // A client cannot retire it with a forged clock; only the executor stamps the time.
    let s = state(actor);
    let forged = Action::Control {
        run_id: "run".into(),
        command: CommandEnvelope {
            id: "forged-retire".into(),
            expected_revision: s.revision,
            command: Command::RetireDecision {
                request_id: stale.clone(),
                now: u64::MAX,
            },
        },
    };
    assert!(host::handle(actor, forged, true, false).0.error.is_some());
    assert!(executor.tick("run").unwrap(), "the lapsed request retires");
    let retired = state(actor);
    assert_eq!(retired.status, RunStatus::Ready);
    assert_eq!(retired.waiting_decision, None);
    assert_eq!(retired.decisions[&stale].approved, None);
    assert!(executor.tick("run").unwrap(), "the step runs again");
    let fresh = state(actor).waiting_decision.clone().unwrap();
    assert_ne!(fresh, stale);
    respond(actor, DIGEST, true, 0).unwrap();
    assert!(executor.tick("run").unwrap());
    assert!(executor.tick("run").unwrap());
    let done = state(actor);
    assert_eq!(done.status, RunStatus::Succeeded);
    assert_eq!(done.node_results["deliver"], "applied");
    assert_eq!(*driver.0.lock().unwrap(), [0, 0, 1]);
}

#[test]
fn an_expired_request_retires_and_the_step_asks_again() {
    let (actor, executor, driver) = start_expiring(vec![], 1);
    executor.tick("run").unwrap();
    retire_and_ask_again(&actor, &executor, &driver);
    actor.shutdown().unwrap();
}

#[test]
fn a_request_invalidated_by_revocation_retires_through_pause_and_resume() {
    let (actor, executor, driver) = start(vec!["code.apply".into()]);
    executor.tick("run").unwrap();
    assert!(
        !executor.tick("run").unwrap(),
        "a valid request stays pending"
    );
    control(
        &actor,
        "revoke",
        Command::RevokeGrant {
            grant: "code.apply".into(),
        },
    );
    control(&actor, "pause", Command::Pause);
    assert!(
        !executor.tick("run").unwrap(),
        "a paused run is not retired"
    );
    control(&actor, "resume", Command::Resume);
    retire_and_ask_again(&actor, &executor, &driver);
    actor.shutdown().unwrap();
}

#[test]
fn a_root_revocation_invalidates_its_childs_decisions() {
    let (actor, _, _) = start(vec!["code.apply".into(), "net".into()]);
    let mut root = state(&actor);
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: "root-start".into(),
                expected_revision: 0,
                command: Command::StartAttempt {
                    attempt: larch_task::types::Attempt {
                        id: "root-attempt".into(),
                        node_id: "deliver".into(),
                        epoch: 1,
                        iteration: None,
                    },
                },
            },
        )
        .unwrap();
    root.id = "child".into();
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: [(
            "deliver".into(),
            ActivityDescriptor {
                kind: "code.deliver".into(),
                payload: serde_json::json!({}),
            },
        )]
        .into(),
    };
    actor
        .spawn_child_with_plan(larch_task::execution::ChildSpawnRequest {
            parent_id: "run".into(),
            expected_revision: 1,
            command_id: "spawn".into(),
            parent_attempt_id: "root-attempt".into(),
            parent_epoch: 1,
            expected_graph_revision: 0,
            child: RunState::new("child".into(), root.definition).unwrap(),
            plan,
        })
        .unwrap();
    let child = || actor.get("child".into()).unwrap().unwrap();
    let revoke = |grant: &str| {
        let root = state(&actor);
        actor
            .execute(
                "run".into(),
                CommandEnvelope {
                    id: format!("revoke:{grant}"),
                    expected_revision: root.revision,
                    command: Command::RevokeGrant {
                        grant: grant.into(),
                    },
                },
            )
            .unwrap();
    };
    let answer = || {
        let waiting = child();
        actor.execute(
            "child".into(),
            CommandEnvelope {
                id: format!("respond:{}", waiting.revision),
                expected_revision: waiting.revision,
                command: Command::RespondDecision {
                    request_id: waiting.waiting_decision.clone().unwrap(),
                    action_digest: DIGEST.into(),
                    approve: true,
                    now: 0,
                },
            },
        )
    };
    let executor = Executor::new(
        actor.clone(),
        Arc::new(Reviewer(Mutex::new(Vec::new()), u64::MAX)),
    );
    assert!(executor.tick("child").unwrap());
    revoke("net");
    let pending = child();
    assert_eq!(
        pending.grant_generation, 1,
        "the revocation reached the child"
    );
    assert_eq!(pending.definition.grants, ["code.apply"]);
    assert!(
        answer().is_err(),
        "a request from before the revocation is stale"
    );
    assert!(executor.tick("child").unwrap(), "so it retires");
    assert!(executor.tick("child").unwrap(), "and the step asks again");
    answer().unwrap();
    assert_eq!(child().approvals("deliver", 0).len(), 1);
    revoke("code.apply");
    assert!(
        child().approvals("deliver", 0).is_empty(),
        "an approval given before the root's revocation no longer authorizes the child"
    );
    actor.shutdown().unwrap();
}

/// Asks for a decision the reducer refuses (an empty scope), after its activity ran.
struct BadRequest;
impl Driver for BadRequest {
    fn execute(&self, _: ActivityRequest) -> Result<ActivityOutcome, String> {
        Ok(ActivityOutcome {
            output: "prepared".into(),
        })
    }
    fn decision_request(
        &self,
        _: &ActivityRequest,
        _: &Result<ActivityOutcome, String>,
    ) -> Option<DecisionRequest> {
        Some(DecisionRequest {
            request_id: "apply".into(),
            action_digest: DIGEST.into(),
            scope: " ".into(),
            expires_at: u64::MAX,
        })
    }
}

#[test]
fn an_error_after_the_claim_fails_the_attempt_instead_of_leaving_it_running() {
    let (actor, _, _) = start(vec![]);
    let executor = Executor::new(actor.clone(), Arc::new(BadRequest));
    let error = executor.tick("run").unwrap_err();
    assert!(
        error.contains("invalid or duplicate decision request"),
        "{error}"
    );
    let failed = state(&actor);
    assert_eq!(failed.status, RunStatus::Waiting);
    assert_eq!(failed.attempt, None);
    assert_eq!(failed.attempt_failure.as_deref(), Some(error.as_str()));
    control(&actor, "resume", Command::Resume);
    assert_eq!(
        state(&actor).status,
        RunStatus::Ready,
        "the user can retry it"
    );
    actor.shutdown().unwrap();
}
