#![cfg(unix)]
use larch_agent::runtime::{ipc::Client, service};
use larch_task::{
    execution::{ActivityDescriptor, ActivityOutcome, ActivityRequest, Driver, ExecutionPlan},
    graph::{Graph, Node, NodeKind},
    types::{PlanningPolicy, RunState, RunStatus, SCHEMA_VERSION, WorkflowDefinition},
};
use std::{
    collections::BTreeMap,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
struct Script;
impl Driver for Script {
    fn execute(&self, request: ActivityRequest) -> Result<ActivityOutcome, String> {
        if request.descriptor.kind != "script" {
            return Err("unknown driver".into());
        }
        Ok(ActivityOutcome {
            output: "done".into(),
        })
    }
}
#[test]
fn authenticated_client_starts_a_persisted_plan_and_observes_completion() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_owned();
    let service_path = root.clone();
    let server = thread::spawn(move || service::serve_with_driver(&service_path, Arc::new(Script)));
    let deadline = Instant::now() + Duration::from_secs(5);
    let client = loop {
        if let Ok(client) = Client::connect(&root) {
            break client;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(20));
    };
    let graph = Graph {
        nodes: vec![Node {
            when: None,
            optional: Vec::new(),
            id: "n".into(),
            depends_on: vec![],
            kind: NodeKind::Activity,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }],
    };
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "test".into(),
        initial_graph: Some(graph),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let state = RunState::new("r".into(), definition).unwrap();
    client.create("c", &state).unwrap();
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([(
            "n".into(),
            ActivityDescriptor {
                kind: "script".into(),
                payload: serde_json::json!({}),
            },
        )]),
    };
    assert!(client.start_plan("r", &plan, "stale-start", 7).is_err());
    assert_eq!(client.get("r").unwrap().unwrap().status, RunStatus::Paused);
    client.start_plan("r", &plan, "start", 0).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let state = client.get("r").unwrap().unwrap();
        if state.status == RunStatus::Succeeded {
            assert_eq!(state.result.as_deref(), Some("done"));
            break;
        }
        assert!(
            Instant::now() < deadline,
            "run did not complete: {:?}",
            state.status
        );
        thread::sleep(Duration::from_millis(20));
    }
    client.shutdown().unwrap();
    server.join().unwrap().unwrap();
}

#[test]
fn service_pumps_due_schedule_and_runs_one_occurrence() {
    use larch_task::schedule::{Policy, Schedule};
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_owned();
    let path = root.clone();
    let server = thread::spawn(move || service::serve_with_driver(&path, Arc::new(Script)));
    let deadline = Instant::now() + Duration::from_secs(5);
    let client = loop {
        if let Ok(client) = Client::connect(&root) {
            break client;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(20));
    };
    let graph = Graph {
        nodes: vec![Node {
            when: None,
            optional: Vec::new(),
            id: "n".into(),
            depends_on: vec![],
            kind: NodeKind::Activity,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }],
    };
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "scheduled".into(),
        initial_graph: Some(graph),
        planning_policy: PlanningPolicy::Disabled,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    let plan = ExecutionPlan {
        graph_revision: 0,
        activities: BTreeMap::from([(
            "n".into(),
            ActivityDescriptor {
                kind: "script".into(),
                payload: serde_json::json!({}),
            },
        )]),
    };
    let now = chrono::Utc::now();
    client
        .create_schedule(
            "s",
            Schedule::Once { at: now },
            Policy::default(),
            definition,
            plan,
            now,
        )
        .unwrap();
    assert!(!client.list_schedules().unwrap()[0].enabled);
    client.enable_schedule("s", 0, true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let runs = client.list().unwrap();
        if runs.len() == 1 && runs[0].status == RunStatus::Succeeded {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "scheduled occurrence did not finish"
        );
        thread::sleep(Duration::from_millis(20));
    }
    thread::sleep(Duration::from_millis(200));
    assert_eq!(client.list().unwrap().len(), 1);
    client.shutdown().unwrap();
    server.join().unwrap().unwrap();
}

struct ScriptPlanner;
impl larch_agent::runtime::planner::InitialPlanner for ScriptPlanner {
    fn plan(&self, actor: &larch_task::runtime::ActorHandle, run_id: &str) -> Result<(), String> {
        let state = actor.get(run_id.into())?.ok_or("unknown run")?;
        let graph = Graph {
            nodes: vec![Node {
                when: None,
                optional: Vec::new(),
                id: "n".into(),
                depends_on: vec![],
                kind: NodeKind::Activity,
                inputs: BTreeMap::new(),
                outputs: BTreeMap::new(),
            }],
        };
        let plan = ExecutionPlan {
            graph_revision: 1,
            activities: BTreeMap::from([(
                "n".into(),
                ActivityDescriptor {
                    kind: "script".into(),
                    payload: serde_json::json!({}),
                },
            )]),
        };
        larch_agent::runtime::planner::commit_initial_execution(
            actor,
            run_id,
            "plan".into(),
            state.revision,
            &state.definition,
            graph,
            plan,
        )?;
        Ok(())
    }
}
#[test]
fn goal_only_run_is_planned_and_executed_by_same_host() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_owned();
    let path = root.clone();
    let server = thread::spawn(move || {
        service::serve_with_planner(&path, Arc::new(Script), Arc::new(ScriptPlanner))
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    let client = loop {
        if let Ok(client) = Client::connect(&root) {
            break client;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(20));
    };
    let definition = WorkflowDefinition {
        version: SCHEMA_VERSION,
        goal: "planned".into(),
        initial_graph: None,
        planning_policy: PlanningPolicy::Initial,
        grants: vec![],
        budgets: BTreeMap::new(),
        acceptance: vec![],
    };
    client
        .create("create", &RunState::new("goal".into(), definition).unwrap())
        .unwrap();
    thread::sleep(Duration::from_millis(300));
    let dormant = client.get("goal").unwrap().unwrap();
    assert_eq!(dormant.status, RunStatus::Paused);
    assert!(dormant.graph.is_none());
    assert_eq!(dormant.revision, 0);
    client.shutdown().unwrap();
    server.join().unwrap().unwrap();
    let path = root.clone();
    let server = thread::spawn(move || {
        service::serve_with_planner(&path, Arc::new(Script), Arc::new(ScriptPlanner))
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    let client = loop {
        if let Ok(client) = Client::connect(&root) {
            break client;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(20));
    };
    thread::sleep(Duration::from_millis(200));
    let restored = client.get("goal").unwrap().unwrap();
    assert_eq!(restored.status, RunStatus::Paused);
    assert!(restored.graph.is_none());
    client
        .start_goal("goal", "start-goal", restored.revision)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let state = client.get("goal").unwrap().unwrap();
        if state.status == RunStatus::Succeeded {
            break;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(20));
    }
    client.shutdown().unwrap();
    server.join().unwrap().unwrap();
}
