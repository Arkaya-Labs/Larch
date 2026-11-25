use larch_task::{
    graph::{Graph, Node, NodeKind},
    reducer::apply,
    types::*,
};

fn initial() -> RunState {
    RunState::new(
        "run".into(),
        WorkflowDefinition {
            version: 1,
            goal: "goal".into(),
            initial_graph: Some(Graph {
                nodes: vec![Node {
                    when: None,
                    optional: Vec::new(),
                    id: "n".into(),
                    depends_on: vec![],
                    kind: NodeKind::Activity,
                    inputs: Default::default(),
                    outputs: Default::default(),
                }],
            }),
            planning_policy: PlanningPolicy::Initial,
            grants: vec![],
            budgets: Default::default(),
            acceptance: vec![],
        },
    )
    .unwrap()
}
fn cmd(state: &RunState, id: &str, command: Command) -> RunState {
    apply(
        state,
        &CommandEnvelope {
            id: id.into(),
            expected_revision: state.revision,
            command,
        },
    )
    .unwrap()
    .0
}
#[test]
fn stale_attempts_and_unresolved_effects_are_fenced() {
    let mut s = initial();
    s = cmd(
        &s,
        "start",
        Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "a".into(),
                node_id: "n".into(),
                epoch: 1,
            },
        },
    );
    s = cmd(
        &s,
        "intent",
        Command::RecordIntent {
            effect_id: "e".into(),
            attempt_id: "a".into(),
            epoch: 1,
            operation_digest: "digest".into(),
        },
    );
    s = cmd(&s, "recover", Command::Recover);
    assert_eq!(s.effects["e"].status, EffectStatus::Uncertain);
    let stale = CommandEnvelope {
        id: "stale".into(),
        expected_revision: s.revision,
        command: Command::FinishAttempt {
            attempt_id: "a".into(),
            epoch: 1,
            result: "bad".into(),
        },
    };
    assert!(apply(&s, &stale).is_err());
    s = cmd(
        &s,
        "resolve",
        Command::ReconcileEffect {
            effect_id: "e".into(),
            receipt: Some("provider receipt".into()),
        },
    );
    s = cmd(&s, "resume", Command::Resume);
    let reuse_epoch = CommandEnvelope {
        id: "bad-start".into(),
        expected_revision: s.revision,
        command: Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "b".into(),
                node_id: "n".into(),
                epoch: 1,
            },
        },
    };
    assert!(apply(&s, &reuse_epoch).is_err());
}
#[test]
fn cancelled_run_accepts_reconciliation_but_not_new_work() {
    let s = cmd(
        &initial(),
        "start",
        Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "a".into(),
                node_id: "n".into(),
                epoch: 1,
            },
        },
    );
    let s = cmd(
        &s,
        "intent",
        Command::RecordIntent {
            effect_id: "e".into(),
            attempt_id: "a".into(),
            epoch: 1,
            operation_digest: "d".into(),
        },
    );
    let s = cmd(&s, "cancel", Command::Cancel);
    let s = cmd(
        &s,
        "resolve",
        Command::ReconcileEffect {
            effect_id: "e".into(),
            receipt: None,
        },
    );
    assert_eq!(s.status, RunStatus::Cancelled);
    assert!(
        apply(
            &s,
            &CommandEnvelope {
                id: "resume".into(),
                expected_revision: s.revision,
                command: Command::Resume,
            }
        )
        .is_err()
    );
}
#[test]
fn graph_nodes_and_acceptance_gate_completion() {
    let mut s = initial();
    s.definition.acceptance.push("check".into());
    let complete = |s: &RunState| CommandEnvelope {
        id: "done".into(),
        expected_revision: s.revision,
        command: Command::CompleteRun {
            result: "ok".into(),
        },
    };
    assert!(apply(&s, &complete(&s)).is_err());
    s = cmd(
        &s,
        "start",
        Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "a".into(),
                node_id: "n".into(),
                epoch: 1,
            },
        },
    );
    s = cmd(
        &s,
        "finish",
        Command::FinishAttempt {
            attempt_id: "a".into(),
            epoch: 1,
            result: "output".into(),
        },
    );
    assert!(apply(&s, &complete(&s)).is_err());
    s = cmd(
        &s,
        "check",
        Command::RecordAcceptance {
            criterion: "check".into(),
            receipt: "verified".into(),
        },
    );
    assert_eq!(
        apply(&s, &complete(&s)).unwrap().0.status,
        RunStatus::Succeeded
    );
}
