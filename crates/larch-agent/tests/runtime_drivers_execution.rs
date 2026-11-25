//! End-to-end driver contract: both authoring paths use the same persisted engine.
use larch_agent::runtime::driver::ActivityRegistry;
use larch_task::{
    execution::{ActivityDescriptor, ExecutionPlan, Executor},
    graph::{Graph, Input, Node, NodeKind},
    runtime::ActorHandle,
    types::{PlanningPolicy, RunState, SCHEMA_VERSION, WorkflowDefinition},
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

fn graph() -> Graph {
    Graph {
        nodes: vec![
            Node {
                when: None,
                optional: Vec::new(),
                id: "first".into(),
                depends_on: vec![],
                kind: NodeKind::Activity,
                inputs: BTreeMap::new(),
                outputs: BTreeMap::from([("text".into(), "text".into())]),
            },
            Node {
                when: None,
                optional: Vec::new(),
                id: "second".into(),
                depends_on: vec!["first".into()],
                kind: NodeKind::Activity,
                inputs: BTreeMap::from([(
                    "source".into(),
                    Input {
                        node: "first".into(),
                        output: "text".into(),
                        schema: "text".into(),
                    },
                )]),
                outputs: BTreeMap::new(),
            },
        ],
    }
}
fn definition(planned: bool) -> WorkflowDefinition {
    WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "produce a verified result".into(),
        initial_graph: (!planned).then(graph),
        planning_policy: PlanningPolicy::Initial,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    }
}

fn run(planned: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workflow.sqlite");
    let actor = ActorHandle::start(&path, 8).unwrap();
    let def = definition(planned);
    actor
        .create(
            "create".into(),
            RunState::new("run".into(), def.clone()).unwrap(),
        )
        .unwrap();
    let planned_execution = if planned {
        let scripted_model_reply = json!({"graph":graph(),"activities":{
            "first":{"kind":"script","payload":{}}, "second":{"kind":"script","payload":{}}
        }})
        .to_string();
        let allowed = ["script".to_string()].into_iter().collect();
        let mut ledger = larch_task::store::SqliteStore::open(&path).unwrap();
        ledger.create_admission_root("run", 100).unwrap();
        let admission = larch_agent::runtime::inference::SharedAdmission::new(
            Arc::new(Mutex::new(ledger)),
            1,
            100,
            "planner-test".into(),
        )
        .unwrap();
        let stop = std::sync::atomic::AtomicBool::new(false);
        let (proposal, execution) = larch_agent::runtime::planner::plan_with_backend(
            &def,
            &admission,
            "run",
            100,
            2,
            &allowed,
            &stop,
            |input, _| {
                assert_eq!(input.goal, "produce a verified result");
                Ok((scripted_model_reply, 3))
            },
        )
        .unwrap();
        larch_agent::runtime::planner::commit_initial_execution(
            &actor,
            "run",
            "plan".into(),
            0,
            &def,
            proposal,
            execution.clone(),
        )
        .unwrap();
        Some(execution)
    } else {
        None
    };
    let observed = Arc::new(Mutex::new(Vec::new()));
    let seen = observed.clone();
    let mut registry = ActivityRegistry::new();
    registry
        .register("script", move |request| {
            seen.lock()
                .unwrap()
                .push((request.node_id.clone(), request.inputs.clone()));
            Ok(if request.node_id == "first" {
                "durable evidence".into()
            } else {
                "verified".into()
            })
        })
        .unwrap();
    let executor = Executor::new(actor.clone(), Arc::new(registry));
    let execution = planned_execution.unwrap_or_else(|| ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([
            (
                "first".into(),
                ActivityDescriptor {
                    kind: "script".into(),
                    payload: json!({}),
                },
            ),
            (
                "second".into(),
                ActivityDescriptor {
                    kind: "script".into(),
                    payload: json!({}),
                },
            ),
        ]),
    });
    executor.install_plan("run", execution).unwrap();
    assert!(executor.tick("run").unwrap());
    drop(executor);
    drop(actor);
    let actor = ActorHandle::start(&path, 8).unwrap();
    let seen = observed.clone();
    let mut registry = ActivityRegistry::new();
    registry
        .register("script", move |request| {
            seen.lock()
                .unwrap()
                .push((request.node_id.clone(), request.inputs.clone()));
            Ok("verified".into())
        })
        .unwrap();
    let executor = Executor::new(actor, Arc::new(registry));
    assert!(executor.tick("run").unwrap());
    let observed = observed.lock().unwrap();
    assert_eq!(observed.len(), 2);
    assert_eq!(observed[0].0, "first");
    assert_eq!(observed[1].0, "second");
    assert_eq!(
        observed[1].1.get("source").map(String::as_str),
        Some("durable evidence")
    );
}

#[test]
fn predefined_steps_resume_with_committed_dependency() {
    run(false);
}
#[test]
fn planned_steps_resume_with_committed_dependency() {
    run(true);
}

#[test]
fn registered_tool_is_journaled_before_completion() {
    use larch_agent::{
        runtime::{
            effects::Declaration,
            permissions::{LiveGrants, Scope},
        },
        tool::{Permission, Tool, ToolCtx, ToolError, ToolSpec},
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct FakeTool(Arc<AtomicUsize>);
    impl Tool for FakeTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: "fake".into(),
                description: "fixture".into(),
                schema: json!({"type":"object"}),
                permission: Permission::Read,
                untrusted_results: true,
                max_result_tokens: 128,
                max_args_tokens: 128,
            }
        }
        fn call(&self, _: &serde_json::Value, _: &ToolCtx) -> Result<String, ToolError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok("tool evidence".into())
        }
    }
    struct FakeDeclaration;
    impl Declaration for FakeDeclaration {
        fn tool_name(&self) -> &str {
            "fake"
        }
        fn capability(&self, _: &serde_json::Value, _: &ToolCtx) -> Result<String, String> {
            Ok("read:fixture".into())
        }
        fn max_argument_bytes(&self) -> usize {
            4096
        }
        fn max_result_bytes(&self) -> usize {
            4096
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let actor = ActorHandle::start(dir.path().join("tool.sqlite"), 8).unwrap();
    let mut def = definition(false);
    def.initial_graph = Some(Graph {
        nodes: vec![graph().nodes.remove(0)],
    });
    def.grants = vec!["read:fixture".into()];
    actor
        .create("create".into(), RunState::new("run".into(), def).unwrap())
        .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let scope = Scope(["read:fixture".into()].into_iter().collect());
    let mut registry = ActivityRegistry::new();
    registry
        .register_tool(
            "tool",
            actor.clone(),
            Arc::new(FakeTool(calls.clone())),
            Arc::new(FakeDeclaration),
            ToolCtx {
                allowed: vec![],
                data_dir: dir.path().into(),
                source: "fixture".into(),
            },
            LiveGrants::new(scope.clone()),
            scope,
        )
        .unwrap();
    let executor = Executor::new(actor.clone(), Arc::new(registry));
    executor
        .install_plan(
            "run",
            ExecutionPlan {
                graph_revision: 0,
                activities: BTreeMap::from([(
                    "first".into(),
                    ActivityDescriptor {
                        kind: "tool".into(),
                        payload: json!({}),
                    },
                )]),
            },
        )
        .unwrap();
    assert!(executor.tick("run").unwrap());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let state = actor.get("run".into()).unwrap().unwrap();
    assert!(state.effects.values().any(|effect| matches!(
        effect.status,
        larch_task::types::EffectStatus::Completed { .. }
    )));
}

#[test]
fn planner_inline_answer_binds_each_step_to_its_activity() {
    let def = definition(true);
    // The shape the planner's grammar writes: each step carries its activity and instruction.
    let response = json!({"nodes": [
        {"id":"first","depends_on":[],"activity":"script","instruction":"draft"},
        {"id":"second","depends_on":["first"],"activity":"script","instruction":"refine"}
    ]})
    .to_string();
    let allowed = ["script".to_string()].into_iter().collect();
    let (proposed, plan) =
        larch_agent::runtime::planner::parse_initial_execution(&def, &response, 2, &allowed)
            .unwrap();
    assert_eq!(proposed.nodes[1].depends_on, ["first"]);
    assert_eq!(plan.activities["second"].kind, "script");
    assert_eq!(
        plan.activities["second"].payload,
        json!({"instruction": "refine"})
    );
    let denied = ["other".to_string()].into_iter().collect();
    assert!(
        larch_agent::runtime::planner::parse_initial_execution(&def, &response, 2, &denied)
            .is_err()
    );
}

#[test]
fn the_planner_binds_only_kinds_that_take_an_instruction() {
    use std::sync::atomic::AtomicBool;
    let dir = tempfile::tempdir().unwrap();
    let actor = ActorHandle::start(dir.path().join("kinds.sqlite"), 8).unwrap();
    let mut registry = ActivityRegistry::new();
    let params = larch::GenParams {
        max_tokens: 8,
        ..Default::default()
    };
    let seats = larch_agent::runtime::inference::SeatAdmission::new(1).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    registry
        .register_model(
            "model",
            actor.clone(),
            fake_larch(),
            seats,
            params,
            512,
            1,
            stop,
        )
        .unwrap();
    let children = ["model".to_string()].into_iter().collect();
    registry
        .register_delegate("delegate", actor, children)
        .unwrap();
    registry.register("script", |_| Ok("done".into())).unwrap();
    // A script's payload is its own, so the planner never binds it; a delegate step is written
    // with its own keys (`goal`, `grants`), never an instruction.
    let allowed = registry.activity_kinds();
    assert_eq!(
        allowed,
        std::collections::BTreeSet::from(["delegate".to_string(), "model".to_string()])
    );
    let def = definition(true);
    let plan = |kind: &str| {
        json!({"nodes": [{"id":"a","depends_on":[],"activity":kind,"instruction":"draft"}]})
            .to_string()
    };
    let parse = |response: &str| {
        larch_agent::runtime::planner::parse_initial_execution(&def, response, 2, &allowed)
    };
    let refused = parse(&plan("delegate")).unwrap_err();
    assert!(refused.contains("wrong keys for its kind"), "{refused}");
    let refused = parse(&plan("script")).unwrap_err();
    assert!(refused.contains("unregistered activity kind"), "{refused}");
    let (_, bound) = parse(&plan("model")).unwrap();
    assert_eq!(
        bound.activities["a"].payload,
        json!({"instruction": "draft"})
    );
}

#[test]
fn an_inline_plan_fails_with_its_own_cause() {
    let def = definition(true);
    let allowed = ["script".to_string()].into_iter().collect();
    let parse = |nodes: serde_json::Value| {
        let response = json!({ "nodes": nodes }).to_string();
        larch_agent::runtime::planner::parse_initial_execution(&def, &response, 2, &allowed)
            .unwrap_err()
    };
    let step = |id: &str, after: &[&str]| json!({"id": id, "depends_on": after, "activity": "script", "instruction": "go"});
    assert!(parse(json!([])).contains("empty graph"));
    for bad in [
        json!([step("a", &[]), step("a", &[])]),
        json!([step("a", &["missing"])]),
    ] {
        let error = parse(bad);
        assert!(!error.contains("invalid planning response"), "{error}");
    }
    // A step short of a field says which, not that the split shape's `graph` is missing.
    let error = parse(json!([{"id":"a","activity":"script","instruction":"go"}]));
    assert!(
        error.contains("depends_on") && !error.contains("graph"),
        "{error}"
    );
}

#[test]
fn step_errors_name_their_cause() {
    use larch_agent::runtime::inference::AdmittedError;
    let shown = |e: AdmittedError<String>| e.to_string();
    assert_eq!(
        shown(AdmittedError::Admission("cancelled".into())),
        "model seat unavailable: cancelled"
    );
    assert_eq!(
        shown(AdmittedError::Settle("seat counter underflow".into())),
        "model seat release failed: seat counter underflow"
    );
    let engine = shown(AdmittedError::Engine(larch::EngineError::Cancelled));
    assert!(engine.starts_with("model inference failed: "), "{engine}");
    assert_eq!(
        shown(AdmittedError::Invalid(
            "runtime requires a finite output limit"
        )),
        "runtime requires a finite output limit"
    );
}

#[test]
fn planner_proposal_requires_registered_kind_and_complete_bindings() {
    let def = definition(true);
    let response = json!({"graph": graph(), "activities": {
        "first": {"kind":"script", "payload":{}},
        "second": {"kind":"script", "payload":{}}
    }})
    .to_string();
    let allowed = ["script".to_string()].into_iter().collect();
    let (proposed, plan) =
        larch_agent::runtime::planner::parse_initial_execution(&def, &response, 2, &allowed)
            .unwrap();
    assert_eq!(proposed, graph());
    assert_eq!(plan.graph_revision, 1);
    let denied = ["other".to_string()].into_iter().collect();
    assert!(
        larch_agent::runtime::planner::parse_initial_execution(&def, &response, 2, &denied)
            .is_err()
    );
}

#[test]
fn budgeted_scripted_model_is_dispatched_and_accounted_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("budget.sqlite");
    let actor = ActorHandle::start(&path, 8).unwrap();
    let mut def = definition(false);
    def.initial_graph = Some(Graph {
        nodes: vec![graph().nodes.remove(0)],
    });
    def.budgets.insert("model_tokens".into(), 25);
    actor
        .create("create".into(), RunState::new("run".into(), def).unwrap())
        .unwrap();
    let mut registry = ActivityRegistry::new();
    registry
        .register_metered("script-model", 20, 1, |request| {
            assert_eq!(request.max_tokens, Some(20));
            Ok(("answer".into(), 7))
        })
        .unwrap();
    let executor = Executor::new(actor.clone(), Arc::new(registry));
    executor
        .install_plan(
            "run",
            ExecutionPlan {
                graph_revision: 0,
                activities: BTreeMap::from([(
                    "first".into(),
                    ActivityDescriptor {
                        kind: "script-model".into(),
                        payload: json!({}),
                    },
                )]),
            },
        )
        .unwrap();
    assert!(executor.tick("run").unwrap());
    let state = actor.get("run".into()).unwrap().unwrap();
    assert_eq!(
        state.node_results.get("first").map(String::as_str),
        Some("answer")
    );
    let mut ledger = larch_task::store::SqliteStore::open(&path).unwrap();
    // A 19-token reservation no longer fits after the 7-token successful charge.
    let request = larch_task::admission::Request {
        id: "check".into(),
        root: "run".into(),
        reserve: 19,
    };
    assert!(ledger.reserve_permit(&request, 1).is_err());
}

/// The fake model on the CPU engine, answering from the last message.
fn fake_larch() -> larch::Larch {
    fake_larch_with("{{ messages[-1].content }}")
}

/// The fake answers each prompt's last token + 1 where a grammar allows it, else the first token
/// it allows: a judge's packet ends in `}`, so it says FAIL, and a `template` ending in `O` says PASS.
fn fake_larch_with(template: &str) -> larch::Larch {
    use larch_engine::{EngineOptions, start_with};
    use larch_kernels::CPU;
    use larch_model::{ChatTemplate, FakeModel, Tokenizer, gguf::Gguf, tiny};
    let bytes = std::fs::read(tiny::TOKENIZER_GGUF).unwrap();
    let gguf = Gguf::parse(&bytes).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    let template = ChatTemplate::new(template, &tokenizer).unwrap();
    let engine = start_with(
        EngineOptions::new("fixture".into(), CPU),
        Box::new(|_, _, _| Ok(Box::new(FakeModel::new(512)))),
    )
    .unwrap();
    larch::Larch::from_engine(engine, tokenizer, template)
}

#[test]
fn larch_cpu_session_runs_through_workflow_admission() {
    use larch::GenParams;
    use std::sync::atomic::AtomicBool;
    let larch = fake_larch();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.sqlite");
    let actor = ActorHandle::start(&path, 8).unwrap();
    let mut def = definition(false);
    def.initial_graph = Some(Graph {
        nodes: vec![graph().nodes.remove(0)],
    });
    def.budgets.insert("model_tokens".into(), 1024);
    actor
        .create("create".into(), RunState::new("run".into(), def).unwrap())
        .unwrap();
    let admission = larch_agent::runtime::inference::SeatAdmission::new(1).unwrap();
    let mut registry = ActivityRegistry::new();
    registry
        .register_model(
            "model",
            actor.clone(),
            larch,
            admission,
            GenParams {
                max_tokens: 8,
                temperature: Some(0.0),
                ..Default::default()
            },
            512,
            1,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    let executor = Executor::new(actor.clone(), Arc::new(registry));
    executor
        .install_plan(
            "run",
            ExecutionPlan {
                graph_revision: 0,
                activities: BTreeMap::from([(
                    "first".into(),
                    ActivityDescriptor {
                        kind: "model".into(),
                        payload: json!({"task":"answer"}),
                    },
                )]),
            },
        )
        .unwrap();
    assert!(executor.tick("run").unwrap());
    let state = actor.get("run".into()).unwrap().unwrap();
    assert!(state.last_epoch > 0);
    // The step's line carries the model's own counts, truncated or not.
    let stats = actor.step_stats("run".into()).unwrap();
    let (prompt, completion) = stats[0].tokens.unwrap();
    assert!(
        prompt > 0 && (1..=8).contains(&completion),
        "{prompt} {completion}"
    );
    if !state.completed_nodes.contains("first") {
        assert!(
            (state.result.as_deref())
                .is_some_and(|r| r.contains("-token limit before it finished")),
            "{:?}",
            state.result
        );
    }
}

#[test]
fn a_helpers_model_step_must_cite_what_it_read() {
    use std::sync::atomic::AtomicBool;
    let dir = tempfile::tempdir().unwrap();
    let actor = ActorHandle::start(dir.path().join("helper.sqlite"), 8).unwrap();
    let mut def = definition(false);
    def.initial_graph = Some(Graph {
        nodes: vec![graph().nodes.remove(0)],
    });
    def.grants = vec!["mcp.wiki".into()];
    actor
        .create("create".into(), RunState::new("run".into(), def).unwrap())
        .unwrap();
    let mut registry = ActivityRegistry::new();
    // The fake model counts up a token at a time: room to reach end of text, never the limit.
    let params = larch::GenParams {
        max_tokens: 600,
        temperature: Some(0.0),
        ..Default::default()
    };
    let seats = larch_agent::runtime::inference::SeatAdmission::new(1).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    registry
        .register_model(
            "model",
            actor.clone(),
            fake_larch(),
            seats,
            params,
            2048,
            1,
            stop,
        )
        .unwrap();
    let children = ["model".to_string()].into_iter().collect();
    registry
        .register_delegate("delegate", actor.clone(), children)
        .unwrap();
    let executor = Executor::new(actor.clone(), Arc::new(registry));
    let payload = json!({"goal": "ask the wiki", "kind": "model", "grants": ["mcp.wiki"]});
    let activities = BTreeMap::from([(
        "first".into(),
        ActivityDescriptor {
            kind: "delegate".into(),
            payload,
        },
    )]);
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities,
    };
    executor.install_plan("run", plan).unwrap();
    executor.tick("run").unwrap();
    let helper = (actor.get("run".into()).unwrap().unwrap())
        .waiting_child
        .unwrap()
        .child_id;
    // The fake model calls no tool, so the helper's answer cites nothing and is refused.
    let _ = executor.tick(&helper);
    let helper = actor.get(helper).unwrap().unwrap();
    assert_eq!(
        helper.attempt_failure.as_deref(),
        Some("the helper cited no tool result it read")
    );
}

#[test]
fn unknown_model_usage_keeps_root_reservation_after_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("uncertain.sqlite");
    let actor = ActorHandle::start(&path, 8).unwrap();
    let mut def = definition(false);
    def.initial_graph = Some(Graph {
        nodes: vec![graph().nodes.remove(0)],
    });
    def.budgets.insert("model_tokens".into(), 20);
    actor
        .create("create".into(), RunState::new("run".into(), def).unwrap())
        .unwrap();
    let mut registry = ActivityRegistry::new();
    registry
        .register_metered("script-model", 20, 1, |_| {
            Err("model failed before usage".into())
        })
        .unwrap();
    let executor = Executor::new(actor, Arc::new(registry));
    executor
        .install_plan(
            "run",
            ExecutionPlan {
                graph_revision: 0,
                activities: BTreeMap::from([(
                    "first".into(),
                    ActivityDescriptor {
                        kind: "script-model".into(),
                        payload: json!({}),
                    },
                )]),
            },
        )
        .unwrap();
    assert!(executor.tick("run").is_err());
    let mut ledger = larch_task::store::SqliteStore::open(&path).unwrap();
    let request = larch_task::admission::Request {
        id: "other".into(),
        root: "run".into(),
        reserve: 1,
    };
    assert!(ledger.reserve_permit(&request, 1).is_err());
}

#[test]
fn committed_planner_output_recovers_without_another_model_call() {
    use larch_task::types::{Attempt, Command, CommandEnvelope};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("planner-recovery.sqlite");
    let actor = ActorHandle::start(&path, 8).unwrap();
    actor
        .create(
            "create".into(),
            RunState::new("run".into(), definition(true)).unwrap(),
        )
        .unwrap();
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: "start".into(),
                expected_revision: 0,
                command: Command::StartAttempt {
                    attempt: Attempt {
                        iteration: None,
                        id: "planning".into(),
                        node_id: "__planning__".into(),
                        epoch: 1,
                    },
                },
            },
        )
        .unwrap();
    let response = json!({"graph":graph(),"activities":{
        "first":{"kind":"script","payload":{}}, "second":{"kind":"script","payload":{}}
    }})
    .to_string();
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: "result".into(),
                expected_revision: 1,
                command: Command::FinishAttempt {
                    attempt_id: "planning".into(),
                    epoch: 1,
                    result: response,
                },
            },
        )
        .unwrap();
    drop(actor);
    let actor = ActorHandle::start(&path, 8).unwrap();
    let state = actor.get("run".into()).unwrap().unwrap();
    let allowed = ["script".to_string()].into_iter().collect();
    larch_agent::runtime::planner::commit_saved_initial(&actor, "run", &state, 2, &allowed)
        .unwrap();
    let state = actor.get("run".into()).unwrap().unwrap();
    assert_eq!(state.graph, Some(graph()));
    assert_eq!(
        actor
            .execution_plan("run".into())
            .unwrap()
            .unwrap()
            .graph_revision,
        1
    );
}

/// A delegate step a planner wrote and committed before delegation left planning, whose graph a
/// crash kept from being installed, is still installed at the next start.
#[test]
fn a_committed_delegate_plan_from_before_the_retirement_still_recovers() {
    use larch_task::types::{Attempt, Command, CommandEnvelope};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("delegate-recovery.sqlite");
    let actor = ActorHandle::start(&path, 8).unwrap();
    let run = RunState::new("run".into(), definition(true)).unwrap();
    actor.create("create".into(), run).unwrap();
    let attempt = Attempt {
        iteration: None,
        id: "planning".into(),
        node_id: "__planning__".into(),
        epoch: 1,
    };
    let response = json!({"nodes": [{"id": "ask", "depends_on": [], "activity": "delegate",
        "goal": "check the wiki", "grants": []}]})
    .to_string();
    for (id, revision, command) in [
        ("start", 0, Command::StartAttempt { attempt }),
        (
            "result",
            1,
            Command::FinishAttempt {
                attempt_id: "planning".into(),
                epoch: 1,
                result: response,
            },
        ),
    ] {
        let envelope = CommandEnvelope {
            id: id.into(),
            expected_revision: revision,
            command,
        };
        actor.execute("run".into(), envelope).unwrap();
    }
    drop(actor);
    let actor = ActorHandle::start(&path, 8).unwrap();
    let state = actor.get("run".into()).unwrap().unwrap();
    // What a host that still registers the delegate kind offers (`activity_kinds`).
    let allowed = ["model".to_string(), "delegate".to_string()].into();
    larch_agent::runtime::planner::commit_saved_initial(&actor, "run", &state, 2, &allowed)
        .unwrap();
    let plan = actor.execution_plan("run".into()).unwrap().unwrap();
    assert_eq!(plan.activities["ask"].kind, "delegate");
    assert_eq!(
        plan.activities["ask"].payload,
        json!({"goal": "check the wiki", "kind": "model", "grants": []})
    );
}

#[test]
fn a_failed_web_search_fails_its_step_and_the_run_can_resume() {
    use larch_agent::{
        runtime::{
            effects::{WebSearchDeclaration, web_search_tool},
            permissions::{LiveGrants, Scope},
        },
        tool::ToolCtx,
    };
    use larch_task::types::{Command, CommandEnvelope, EffectStatus, RunStatus};
    let dir = tempfile::tempdir().unwrap();
    // A config.toml that does not read fails the search before any provider is asked.
    std::fs::create_dir_all(dir.path().join("agent/config.toml")).unwrap();
    let actor = ActorHandle::start(dir.path().join("search.sqlite"), 8).unwrap();
    let mut def = definition(false);
    def.initial_graph = Some(Graph {
        nodes: vec![graph().nodes.remove(0)],
    });
    def.grants = vec!["network.web_search".into()];
    actor
        .create("create".into(), RunState::new("run".into(), def).unwrap())
        .unwrap();
    let scope = Scope(["network.web_search".into()].into_iter().collect());
    let mut registry = ActivityRegistry::new();
    registry
        .register_tool(
            "web_search",
            actor.clone(),
            web_search_tool(),
            Arc::new(WebSearchDeclaration),
            ToolCtx {
                allowed: vec![],
                data_dir: dir.path().into(),
                source: "fixture".into(),
            },
            LiveGrants::new(scope.clone()),
            scope,
        )
        .unwrap();
    let executor = Executor::new(actor.clone(), Arc::new(registry));
    executor
        .install_plan(
            "run",
            ExecutionPlan {
                graph_revision: 0,
                activities: BTreeMap::from([(
                    "first".into(),
                    ActivityDescriptor {
                        kind: "web_search".into(),
                        payload: json!({"query": "larch"}),
                    },
                )]),
            },
        )
        .unwrap();
    executor.tick("run").unwrap();
    let state = actor.get("run".into()).unwrap().unwrap();
    assert_eq!(state.status, RunStatus::Waiting);
    assert!(
        state
            .attempt_failure
            .as_deref()
            .is_some_and(|reason| reason.contains("does not read")),
        "{:?}",
        state.attempt_failure
    );
    assert!(
        !state.effects.is_empty()
            && state
                .effects
                .values()
                .all(|e| matches!(e.status, EffectStatus::Completed { .. })),
        "{:?}",
        state.effects
    );
    actor
        .execute(
            "run".into(),
            CommandEnvelope {
                id: "resume".into(),
                expected_revision: state.revision,
                command: Command::Resume,
            },
        )
        .unwrap();
}

#[test]
fn a_failed_plan_is_planned_afresh_and_an_unfit_goal_waits_with_its_reason() {
    use larch::{GenParams, Larch};
    use larch_agent::runtime::planner::{InitialPlanner, LarchPlanner};
    use larch_engine::{EngineOptions, start_with};
    use larch_kernels::CPU;
    use larch_model::{ChatTemplate, FakeModel, Tokenizer, gguf::Gguf, tiny};
    use larch_task::types::{Attempt, Command, CommandEnvelope, RunStatus};
    use std::sync::atomic::AtomicBool;
    let bytes = std::fs::read(tiny::TOKENIZER_GGUF).unwrap();
    let gguf = Gguf::parse(&bytes).unwrap();
    let tokenizer = Tokenizer::from_gguf(&gguf).unwrap();
    let template = ChatTemplate::new("{{ messages[-1].content }}", &tokenizer).unwrap();
    let engine = start_with(
        EngineOptions::new("fixture".into(), CPU),
        Box::new(|_, _, _| Ok(Box::new(FakeModel::new(512)))),
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let actor = ActorHandle::start(dir.path().join("replan.sqlite"), 8).unwrap();
    actor
        .create(
            "create".into(),
            RunState::new("run".into(), definition(true)).unwrap(),
        )
        .unwrap();
    // The first answer did not parse: its reason is now the run's `result`.
    let commands = [
        Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "planner:run:1".into(),
                node_id: "__planning__".into(),
                epoch: 1,
            },
        },
        Command::FailAttempt {
            attempt_id: "planner:run:1".into(),
            epoch: 1,
            reason: "invalid planning response: EOF".into(),
        },
        Command::Resume,
    ];
    for (revision, command) in commands.into_iter().enumerate() {
        actor
            .execute(
                "run".into(),
                CommandEnvelope {
                    id: format!("c{revision}"),
                    expected_revision: revision as u64,
                    command,
                },
            )
            .unwrap();
    }
    // An 8-token answer cannot fit a 4-token context: planning fails before any model call.
    let planner = LarchPlanner {
        larch: Larch::from_engine(engine, tokenizer, template),
        seats: larch_agent::runtime::inference::SeatAdmission::new(1).unwrap(),
        params: GenParams {
            max_tokens: 8,
            ..Default::default()
        },
        context_tokens: 4,
        capacity: 1,
        max_nodes: 2,
        allowed_kinds: ["model".to_string()].into_iter().collect(),
        step_tools: Arc::new(Vec::new),
        stop: Arc::new(AtomicBool::new(false)),
    };
    let error = planner.plan(&actor, "run").unwrap_err();
    assert!(error.contains("response budget exceeds context"), "{error}");
    let state = actor.get("run".into()).unwrap().unwrap();
    assert_eq!(state.status, RunStatus::Waiting);
    assert_eq!(state.last_epoch, 2);
    assert_eq!(state.attempt_failure.as_deref(), Some(error.as_str()));
    // Recorded once: the worker's next tick finds nothing to plan.
    assert!(planner.plan(&actor, "run").is_err());
    assert_eq!(
        actor.get("run".into()).unwrap().unwrap().revision,
        state.revision
    );
}

#[test]
fn a_planning_failure_lands_though_a_revocation_moved_the_revision() {
    use larch_task::types::{Attempt, Command, CommandEnvelope, RunStatus};
    let dir = tempfile::tempdir().unwrap();
    let actor = ActorHandle::start(dir.path().join("revoked.sqlite"), 8).unwrap();
    let mut def = definition(true);
    def.grants = vec!["g".into()];
    actor
        .create("create".into(), RunState::new("run".into(), def).unwrap())
        .unwrap();
    let send = |revision: u64, command| {
        actor.execute(
            "run".into(),
            CommandEnvelope {
                id: format!("c{revision}"),
                expected_revision: revision,
                command,
            },
        )
    };
    send(
        0,
        Command::StartAttempt {
            attempt: Attempt {
                iteration: None,
                id: "planner:run:1".into(),
                node_id: "__planning__".into(),
                epoch: 1,
            },
        },
    )
    .unwrap();
    // The revocation lands between the synthetic start and its failure.
    send(1, Command::RevokeGrant { grant: "g".into() }).unwrap();
    larch_agent::runtime::planner::fail_attempt(
        &actor,
        "run",
        "planner:run:1",
        1,
        1,
        "unfit goal",
        false,
    )
    .unwrap();
    let state = actor.get("run".into()).unwrap().unwrap();
    assert_eq!(state.status, RunStatus::Waiting);
    assert_eq!(state.attempt_failure.as_deref(), Some("unfit goal"));
    // A stale epoch is not the live attempt: it is refused, not retried.
    assert!(
        larch_agent::runtime::planner::fail_attempt(
            &actor,
            "run",
            "planner:run:1",
            1,
            1,
            "x",
            false
        )
        .is_err()
    );
}

#[test]
fn planner_rejects_unsupported_nodes_and_oversized_activity_payload() {
    let def = definition(true);
    let allowed = ["script".to_string()].into_iter().collect();
    let mut unsupported = graph();
    unsupported.nodes[0].kind = NodeKind::Map { max_items: 2 };
    let response = json!({"graph":unsupported,"activities":{
        "first":{"kind":"script","payload":{}}, "second":{"kind":"script","payload":{}}
    }})
    .to_string();
    assert!(
        larch_agent::runtime::planner::parse_initial_execution(&def, &response, 2, &allowed)
            .is_err()
    );
    let response = json!({"graph":graph(),"activities":{
        "first":{"kind":"script","payload":{"input":"x".repeat(65537)}},
        "second":{"kind":"script","payload":{}}
    }})
    .to_string();
    assert!(
        larch_agent::runtime::planner::parse_initial_execution(&def, &response, 2, &allowed)
            .is_err()
    );
}

/// Hands the executor's real request to the test instead of running it.
struct Probe(Mutex<Option<larch_task::execution::ActivityRequest>>);
impl larch_task::execution::Driver for Probe {
    fn execute(
        &self,
        request: larch_task::execution::ActivityRequest,
    ) -> Result<larch_task::execution::ActivityOutcome, String> {
        *self.0.lock().unwrap() = Some(request);
        Err("probe".into())
    }
}

#[test]
fn only_a_model_step_that_timed_out_retries_and_it_backs_off_twice() {
    use larch_task::execution::Driver;
    use std::{sync::atomic::AtomicBool, time::Duration};
    let actor = ActorHandle::start(":memory:", 8).unwrap();
    let mut def = definition(false);
    def.initial_graph = Some(Graph {
        nodes: vec![graph().nodes.remove(0)],
    });
    actor
        .create("create".into(), RunState::new("run".into(), def).unwrap())
        .unwrap();
    let probe = Arc::new(Probe(Mutex::new(None)));
    let executor = Executor::new(actor.clone(), probe.clone());
    let descriptor = |kind: &str| ActivityDescriptor {
        kind: kind.into(),
        payload: json!({}),
    };
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([("first".into(), descriptor("model"))]),
    };
    executor.install_plan("run", plan).unwrap();
    executor.tick("run").unwrap();
    let mut request = probe.0.lock().unwrap().take().unwrap();
    let mut registry = ActivityRegistry::new();
    let params = larch::GenParams {
        max_tokens: 8,
        ..Default::default()
    };
    let seats = larch_agent::runtime::inference::SeatAdmission::new(1).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    registry
        .register_model("model", actor, fake_larch(), seats, params, 512, 1, stop)
        .unwrap();
    let late = "the step ran past its 10-minute time limit";
    let delays: Vec<_> = (0..3)
        .map(|n| registry.retry_delay(&request, late, n))
        .collect();
    let s = Duration::from_secs;
    assert_eq!(delays, [Some(s(2)), Some(s(4)), None]);
    assert_eq!(registry.retry_delay(&request, "cancelled", 0), None);
    request.descriptor = descriptor("script");
    assert_eq!(registry.retry_delay(&request, late, 0), None);
}

/// A one-step run done when it "names a source", its answer `answer`, ticked until it settles
/// under `larch`'s judge.
fn judged(larch: larch::Larch, answer: &'static str) -> RunState {
    use std::sync::atomic::AtomicBool;
    let dir = tempfile::tempdir().unwrap();
    let actor = ActorHandle::start(dir.path().join("judged.sqlite"), 8).unwrap();
    let mut def = definition(false);
    def.initial_graph = Some(Graph {
        nodes: vec![graph().nodes.remove(0)],
    });
    def.acceptance = vec!["names a source".into()];
    actor
        .create("create".into(), RunState::new("run".into(), def).unwrap())
        .unwrap();
    let mut registry = ActivityRegistry::new();
    registry
        .register("script", move |_| Ok(answer.into()))
        .unwrap();
    let seats = larch_agent::runtime::inference::SeatAdmission::new(1).unwrap();
    let params = larch::GenParams {
        max_tokens: 256,
        temperature: Some(0.0),
        ..Default::default()
    };
    let stop = Arc::new(AtomicBool::new(false));
    registry
        .register_checks(actor.clone(), larch, seats, params, 512, stop)
        .unwrap();
    let executor = Executor::new(actor.clone(), Arc::new(registry));
    let descriptor = ActivityDescriptor {
        kind: "script".into(),
        payload: json!({}),
    };
    executor
        .install_plan(
            "run",
            ExecutionPlan {
                graph_revision: 0,
                activities: BTreeMap::from([("first".into(), descriptor)]),
            },
        )
        .unwrap();
    for _ in 0..4 {
        executor.tick("run").unwrap();
    }
    actor.get("run".into()).unwrap().unwrap()
}

/// A judge's ruling, never a failure to ask one.
fn ruled(verdict: &Result<String, String>) -> bool {
    match verdict {
        Ok(receipt) => receipt.starts_with("judge: "),
        Err(why) => !["could not run", "no verdict", "too long", "budget"]
            .iter()
            .any(|cause| why.contains(cause)),
    }
}

#[test]
fn a_run_is_done_only_when_its_helpers_cite_what_they_read_and_the_judge_rules() {
    use larch_task::{execution::Driver, types::RunStatus};
    use std::sync::atomic::AtomicBool;
    // A PASS is recorded and the run completes.
    let passed = judged(
        fake_larch_with("{{ messages[-1].content }}O"),
        "the answer, from [1]",
    );
    assert_eq!(passed.status, RunStatus::Succeeded, "{passed:?}");
    assert!(passed.acceptance_receipts["names a source"].starts_with("judge: "));
    // A FAIL parks the run for the user, saying why.
    let failed = judged(fake_larch(), "the answer, from [1]");
    assert_eq!(failed.status, RunStatus::Waiting);
    let why = failed.attempt_failure.unwrap();
    let why = why.strip_prefix("not done yet (names a source): ").unwrap();
    assert!(ruled(&Err(why.into())), "{why}");

    let dir = tempfile::tempdir().unwrap();
    let actor = ActorHandle::start(dir.path().join("checks.sqlite"), 8).unwrap();
    let mut def = definition(false);
    def.acceptance = vec!["names a source".into()];
    let mut state = RunState::new("run".into(), def).unwrap();
    actor.create("create".into(), state.clone()).unwrap();
    // A step answer that fits its own budget (half the context) fits the judge's packet too.
    let answer = "a".repeat(191);
    let larch = fake_larch();
    let tokens = larch
        .prompt_len(larch::Prompt::Text(answer.clone()), &[], None)
        .unwrap();
    assert_eq!(tokens, 192, "with its BOS");
    state.result = Some(answer);
    let mut registry = ActivityRegistry::new();
    // Unregistered, nothing is checked and the run settles as before.
    assert_eq!(registry.accept(&state, "names a source"), None);
    let params = larch::GenParams {
        max_tokens: 256,
        temperature: Some(0.0),
        ..Default::default()
    };
    let seats = larch_agent::runtime::inference::SeatAdmission::new(1).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    registry
        .register_checks(actor, larch, seats, params, 512, stop.clone())
        .unwrap();
    let verdict = registry.accept(&state, "names a source").unwrap();
    assert!(ruled(&verdict), "{verdict:?}");
    // A budgeted run is refused before the judge spends a token its budget does not see.
    let mut budgeted = state.clone();
    (budgeted.definition.budgets).insert("model_tokens".into(), 1024);
    assert_eq!(
        registry.accept(&budgeted, "names a source"),
        Some(Err(
            "a budgeted run is not checked: the check would spend outside its budget".into()
        ))
    );
    // A helper citing a read its journal lacks fails before any judge is asked.
    let cited = larch_agent::runtime::children::ChildResult {
        child_id: "run/never".into(),
        graph_revision: 0,
        summary: "made up".into(),
        evidence: vec!["read-1".into()],
    };
    let mut ungrounded = state.clone();
    (ungrounded.node_results).insert("helper".into(), serde_json::to_string(&cited).unwrap());
    assert_eq!(
        registry.accept(&ungrounded, "names a source"),
        Some(Err(
            "the helper behind helper cites a tool result its journal lacks".into()
        ))
    );
    // Shutting down is no verdict: nothing is recorded, and the next start checks again.
    stop.store(true, std::sync::atomic::Ordering::Release);
    assert_eq!(registry.accept(&state, "names a source"), None);
}

#[test]
fn a_check_waiting_for_a_seat_ends_when_its_run_is_cancelled() {
    use larch_agent::runtime::inference::Admission;
    use larch_task::{
        execution::Driver,
        types::{Command, CommandEnvelope},
    };
    use std::{sync::atomic::AtomicBool, time::Duration};
    let dir = tempfile::tempdir().unwrap();
    let actor = ActorHandle::start(dir.path().join("checks.sqlite"), 8).unwrap();
    let mut def = definition(false);
    def.acceptance = vec!["names a source".into()];
    actor
        .create("create".into(), RunState::new("run".into(), def).unwrap())
        .unwrap();
    let state = actor.get("run".into()).unwrap().unwrap();
    let seats = larch_agent::runtime::inference::SeatAdmission::new(1).unwrap();
    let mut registry = ActivityRegistry::new();
    let params = larch::GenParams {
        max_tokens: 256,
        temperature: Some(0.0),
        ..Default::default()
    };
    let stop = Arc::new(AtomicBool::new(false));
    registry
        .register_checks(
            actor.clone(),
            fake_larch(),
            seats.clone(),
            params,
            512,
            stop,
        )
        .unwrap();
    // Foreground work holds the only seat.
    let held = seats.acquire("other", 1, &AtomicBool::new(false)).unwrap();
    std::thread::scope(|s| {
        let (sent, verdict) = std::sync::mpsc::channel();
        let (registry, state) = (&registry, &state);
        s.spawn(move || sent.send(registry.accept(state, "names a source")));
        std::thread::sleep(Duration::from_millis(100));
        let envelope = CommandEnvelope {
            id: "cancel".into(),
            expected_revision: state.revision,
            command: Command::Cancel,
        };
        actor.execute("run".into(), envelope).unwrap();
        let verdict = verdict.recv_timeout(Duration::from_secs(5));
        // Free the seat either way, so a check still queued ends and the test fails, not hangs.
        drop(held);
        assert_eq!(
            verdict,
            Ok(None),
            "the cancelled run's check kept its place"
        );
    });
}
