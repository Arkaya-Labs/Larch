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

fn run(id: &str) -> RunState {
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
            grants: vec!["file:read".into()],
            budgets: BTreeMap::new(),
            acceptance: vec![],
        },
    )
    .unwrap()
}

#[test]
fn revocation_fences_inflight_child_effect_claim() {
    let actor = ActorHandle::start(":memory:", 4).unwrap();
    actor.create("create".into(), run("root")).unwrap();
    actor
        .execute(
            "root".into(),
            CommandEnvelope {
                id: "root-start".into(),
                expected_revision: 0,
                command: Command::StartAttempt {
                    attempt: Attempt {
                        iteration: None,
                        id: "root-attempt".into(),
                        node_id: "node".into(),
                        epoch: 1,
                    },
                },
            },
        )
        .unwrap();
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([(
            "node".into(),
            ActivityDescriptor {
                kind: "script".into(),
                payload: serde_json::Value::Null,
            },
        )]),
    };
    actor
        .spawn_child_with_plan(ChildSpawnRequest {
            parent_id: "root".into(),
            expected_revision: 1,
            command_id: "spawn".into(),
            parent_attempt_id: "root-attempt".into(),
            parent_epoch: 1,
            expected_graph_revision: 0,
            child: run("child"),
            plan,
        })
        .unwrap();
    let (generation, grants) = actor.effective_grants("child".into()).unwrap();
    assert!(grants.contains(&"file:read".into()));
    actor
        .execute(
            "child".into(),
            CommandEnvelope {
                id: "start".into(),
                expected_revision: 0,
                command: Command::StartAttempt {
                    attempt: Attempt {
                        iteration: None,
                        id: "a".into(),
                        node_id: "node".into(),
                        epoch: 1,
                    },
                },
            },
        )
        .unwrap();
    actor
        .execute(
            "root".into(),
            CommandEnvelope {
                id: "revoke".into(),
                expected_revision: 2,
                command: Command::RevokeGrant {
                    grant: "file:read".into(),
                },
            },
        )
        .unwrap();
    let intent = CommandEnvelope {
        id: "intent".into(),
        expected_revision: 1,
        command: Command::RecordIntent {
            effect_id: "effect".into(),
            attempt_id: "a".into(),
            epoch: 1,
            operation_digest: "digest".into(),
        },
    };
    assert!(
        actor
            .reserve_granted_effect("child".into(), intent, "file:read".into(), generation)
            .is_err()
    );
    assert!(
        actor
            .get("child".into())
            .unwrap()
            .unwrap()
            .effects
            .is_empty()
    );
    let (new_generation, grants) = actor.effective_grants("child".into()).unwrap();
    assert!(new_generation > generation);
    assert!(!grants.contains(&"file:read".into()));
    actor.shutdown().unwrap();
}

/// Revokes a grant on its root while the activity runs, as the user can from another window.
struct RevokesMidway {
    actor: ActorHandle,
    budgeted: bool,
}
impl Driver for RevokesMidway {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        let root = self.actor.get(request.root_id.clone())?.ok_or("no root")?;
        self.actor.execute(
            request.root_id,
            CommandEnvelope {
                id: "revoke-midway".into(),
                expected_revision: root.revision,
                command: Command::RevokeGrant {
                    grant: "file:read".into(),
                },
            },
        )?;
        Ok(ActivityOutcome {
            output: "done".into(),
        })
    }
    fn enforces_budget(&self) -> bool {
        self.budgeted
    }
    fn reservation(&self, _: &ActivityRequest) -> Option<(u64, usize)> {
        self.budgeted.then_some((8, 1))
    }
    fn measured_usage(
        &self,
        _: &ActivityRequest,
        _: &Result<ActivityOutcome, String>,
    ) -> Option<u64> {
        self.budgeted.then_some(5)
    }
}

#[test]
fn a_finished_activity_commits_after_a_concurrent_revocation() {
    for budgeted in [false, true] {
        let actor = ActorHandle::start(":memory:", 4).unwrap();
        let mut state = run("root");
        if budgeted {
            state.definition.budgets = [("model_tokens".into(), 10)].into();
            state = RunState::new("root".into(), state.definition).unwrap();
        }
        actor.create("create".into(), state).unwrap();
        let executor = Executor::new(
            actor.clone(),
            Arc::new(RevokesMidway {
                actor: actor.clone(),
                budgeted,
            }),
        );
        executor
            .install_plan(
                "root",
                ExecutionPlan {
                    graph_revision: 0,
                    activities: BTreeMap::from([(
                        "node".into(),
                        ActivityDescriptor {
                            kind: "script".into(),
                            payload: serde_json::Value::Null,
                        },
                    )]),
                },
            )
            .unwrap();
        assert!(executor.tick("root").unwrap(), "budgeted: {budgeted}");
        let done = actor.get("root".into()).unwrap().unwrap();
        assert_eq!(done.status, RunStatus::Ready, "budgeted: {budgeted}");
        assert_eq!(done.node_results["node"], "done", "budgeted: {budgeted}");
        assert!(done.definition.grants.is_empty());
        assert!(executor.tick("root").unwrap());
        assert_eq!(
            actor.get("root".into()).unwrap().unwrap().status,
            RunStatus::Succeeded,
            "budgeted: {budgeted}"
        );
        if budgeted {
            // Capacity is one permit across roots: another root starts only if this one settled.
            let mut other = run("other");
            other.definition.budgets = [("model_tokens".into(), 10)].into();
            let other = RunState::new("other".into(), other.definition).unwrap();
            actor.create("create-other".into(), other).unwrap();
            executor
                .install_plan(
                    "other",
                    actor.execution_plan("root".into()).unwrap().unwrap(),
                )
                .unwrap();
            assert!(executor.tick("other").unwrap());
        }
        actor.shutdown().unwrap();
    }
}
