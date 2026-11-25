//! App participation in the local task service and device ownership protocol.
use larch_agent::runtime::service::DeviceLease;
use larch_app::App;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "larch-app-owner-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn the_app_hosts_the_task_service_when_none_runs_and_stops_it_on_drop() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
    let service = app.task_service().unwrap();
    let (caps, cards) = service.snapshot().unwrap();
    assert!(cards.is_empty());
    // No model loaded: tasks are kept and listed, but nothing runs them.
    assert!(!caps.execution);
    assert!(dir.0.join("runtime-service/tasks.sqlite").exists());
    drop(service);
    drop(app);
    assert!(
        larch_agent::runtime::ipc::Client::connect(&dir.0).is_err(),
        "the hosted service stops with its app"
    );
}

#[test]
fn the_hosted_service_follows_the_loaded_model() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
    let model = tiny_model(&dir.0, "tiny");
    assert!(!execution(&app));
    app.load_model(&model).unwrap();
    assert!(
        execution(&app),
        "a load restarts the hosted service on the new engine"
    );
    app.load_model(&model).unwrap();
    assert!(execution(&app));
    app.delete_model(&model).unwrap();
    assert!(
        !execution(&app),
        "with the engine gone it keeps serving, without execution"
    );
}

#[test]
fn model_load_refuses_an_owned_device_and_failed_load_releases_its_lease() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
    let holder = DeviceLease::acquire(&dir.0).unwrap();
    let missing = dir.0.join("missing.gguf");
    let error = app.load_model(missing.to_str().unwrap()).unwrap_err();
    assert!(error.contains("device is already owned"), "{error}");
    drop(holder);
    let error = app.load_model(missing.to_str().unwrap()).unwrap_err();
    assert!(error.contains("Cannot load"), "{error}");
    assert!(DeviceLease::acquire(&dir.0).is_ok());
}

#[test]
fn lease_covers_a_loaded_model_and_is_held_after_unload() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
    let model = tiny_model(&dir.0, "tiny");
    app.load_model(&model).unwrap();
    assert!(DeviceLease::acquire(&dir.0).is_err());
    app.delete_model(&model).unwrap();
    assert!(DeviceLease::acquire(&dir.0).is_err());
    drop(app);
    assert!(DeviceLease::acquire(&dir.0).is_err());
}

/// The hosted service can run steps: a service on the loaded engine.
fn execution(app: &App) -> bool {
    app.task_service().unwrap().snapshot().unwrap().0.execution
}

#[test]
fn first_calls_at_once_host_one_service() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
    let clients: Vec<_> = std::thread::scope(|s| {
        let calls: Vec<_> = (0..4)
            .map(|_| s.spawn(|| app.task_service().unwrap()))
            .collect();
        calls.into_iter().map(|c| c.join().unwrap()).collect()
    });
    // A second start would have shut the first down, and its token with it.
    for client in clients {
        client.snapshot().unwrap();
    }
}

#[test]
fn a_service_asked_for_during_a_load_runs_on_the_new_engine() {
    use std::sync::atomic::AtomicBool;
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
    let model = tiny_model(&dir.0, "tiny");
    let loaded = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            while !loaded.load(Ordering::Relaxed) {
                app.task_service().unwrap();
            }
        });
        app.load_model(&model).unwrap();
        loaded.store(true, Ordering::Relaxed);
    });
    assert!(execution(&app), "a service started mid-load kept no engine");
}

#[test]
fn a_delete_beside_a_load_never_unloads_the_new_model() {
    let dir = Temp::new();
    let app = App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
    let (old, new) = (tiny_model(&dir.0, "old"), tiny_model(&dir.0, "new"));
    app.load_model(&old).unwrap();
    app.task_service().unwrap();
    std::thread::scope(|s| {
        s.spawn(|| app.delete_model(&old).unwrap());
        s.spawn(|| app.load_model(&new).unwrap());
    });
    assert_eq!(app.app_state().model.as_deref(), Some(new.as_str()));
    assert!(!std::path::Path::new(&old).exists());
    assert!(execution(&app));
}

#[test]
fn a_refused_load_or_delete_keeps_the_hosted_service_running() {
    use larch_app::{Event, contract::SendParams};
    use std::sync::{Mutex, mpsc};
    let dir = Temp::new();
    // The reply's first token blocks on `hold`, so its turn is under way for as long as the test
    // needs: a chat reply never waits for a seat, so nothing else keeps it there.
    let (hold, held) = mpsc::channel::<()>();
    let (started, on_token) = mpsc::channel();
    let gate = Mutex::new(Some((held, started)));
    let app = App::new(dir.0.clone(), larch_kernels::CPU, move |e| {
        if let Event::Chat { .. } = e {
            let first = gate.lock().unwrap().take();
            if let Some((held, started)) = first {
                started.send(()).unwrap();
                let _ = held.recv();
            }
        }
    })
    .unwrap();
    let model = tiny_model(&dir.0, "tiny");
    app.load_model(&model).unwrap();
    let client = app.task_service().unwrap();
    let chat = app.new_chat(None).unwrap();
    let params = SendParams {
        max_tokens: Some(4),
        temperature: Some(0.0),
    };
    app.send(&chat, "hello", params).unwrap();
    on_token
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the reply streams a token");
    let refused = app.load_model(&model).unwrap_err();
    assert!(refused.contains("Stop the replies"), "{refused}");
    let refused = app.delete_model(&model).unwrap_err();
    assert!(refused.contains("Stop the replies"), "{refused}");
    // The same service: a restart would have ended this client's token.
    assert!(client.snapshot().unwrap().0.execution);
    app.cancel(&chat).unwrap();
    drop(hold);
}

#[test]
fn a_failed_restart_keeps_the_model_change_and_shows_on_the_next_call() {
    use larch_app::{Event, contract::Health};
    use std::sync::{Arc, Mutex};
    let dir = Temp::new();
    // Another owner takes the folder's service while the model changes, so the restart fails.
    let taken = Arc::new(Mutex::new(None::<fs::File>));
    let lock = dir.0.join("runtime-service").join("owner.lock");
    let app = App::new(dir.0.clone(), larch_kernels::CPU, {
        let taken = taken.clone();
        move |e| {
            if let Event::Health(Health::Loading | Health::Empty) = e {
                let file = fs::File::options().write(true).open(&lock).unwrap();
                file.try_lock().unwrap();
                *taken.lock().unwrap() = Some(file);
            }
        }
    })
    .unwrap();
    let model = tiny_model(&dir.0, "tiny");
    assert!(!execution(&app));
    app.load_model(&model).unwrap();
    assert_eq!(app.app_state().model.as_deref(), Some(model.as_str()));
    let error = app.task_service().err().expect("the restart failed");
    assert!(error.contains("already owned"), "{error}");
    taken.lock().unwrap().take();
    assert!(execution(&app), "the next call hosts it on the new engine");
    app.delete_model(&model).unwrap();
    assert!(!std::path::Path::new(&model).exists());
    let error = app.task_service().err().expect("the restart failed");
    assert!(error.contains("already owned"), "{error}");
    taken.lock().unwrap().take();
    assert!(!execution(&app));
}

#[test]
fn a_killed_services_files_do_not_stop_the_app_hosting() {
    let dir = Temp::new();
    let service = dir.0.join("runtime-service");
    fs::create_dir_all(&service).unwrap();
    fs::write(service.join("token"), "stale").unwrap();
    fs::write(service.join("service.sock"), "").unwrap();
    let app = App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
    assert!(!execution(&app));
    app.stop_tasks().unwrap();
    assert!(
        larch_agent::runtime::ipc::Client::connect(&dir.0).is_err(),
        "stop_tasks ends the hosted service"
    );
}

/// The tiny model, with a byte tokenizer, as `<name>.gguf` in `dir`'s models folder.
fn tiny_model(dir: &std::path::Path, name: &str) -> String {
    use larch_model::{
        Arch,
        gguf::{self, Gguf, Value},
        tiny,
    };
    let bytes = tiny::gguf(Arch::Llama, 3);
    let g = Gguf::parse(&bytes).unwrap();
    let tensors: Vec<_> = g
        .tensors
        .iter()
        .map(|t| {
            let at = (g.data_offset + t.offset) as usize;
            (
                t.name.clone(),
                t.dims.clone(),
                t.ggml_type,
                bytes[at..at + t.bytes as usize].to_vec(),
            )
        })
        .collect();
    let mut kv: Vec<_> = g.kv.into_iter().collect();
    let tokens = (0..256).map(|b| Value::Str(format!("<0x{b:02X}>")));
    kv.extend([
        ("tokenizer.ggml.model".into(), Value::Str("llama".into())),
        ("tokenizer.ggml.tokens".into(), Value::Arr(tokens.collect())),
    ]);
    let model = dir.join("models").join(format!("{name}.gguf"));
    fs::write(&model, gguf::write(&kv, &tensors)).unwrap();
    model.to_str().unwrap().to_owned()
}

/// A background service with a planner whose single step runs `Step`.
#[cfg(unix)]
mod service {
    use larch_app::task::{TaskCard, TaskSection};
    use larch_task::{
        execution::{ActivityDescriptor, ActivityOutcome, ActivityRequest, Driver, ExecutionPlan},
        graph::{Graph, Node, NodeKind},
        host::{self, InitialPlanner},
        ipc::Client,
        runtime::ActorHandle,
        types::{Command, CommandEnvelope, PlanningPolicy, SCHEMA_VERSION, WorkflowDefinition},
    };
    use std::{
        collections::BTreeMap,
        path::{Path, PathBuf},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        thread,
        time::{Duration, Instant},
    };

    /// Fails its first `failures` executions, then succeeds; counts every call.
    pub struct Step {
        pub failures: usize,
        pub calls: Arc<AtomicUsize>,
    }
    impl Driver for Step {
        fn execute(&self, _: ActivityRequest) -> Result<ActivityOutcome, String> {
            if self.calls.fetch_add(1, Ordering::SeqCst) < self.failures {
                return Err("tool unavailable".into());
            }
            Ok(ActivityOutcome {
                output: "done".into(),
            })
        }
    }
    struct Planner(Arc<AtomicUsize>);
    impl InitialPlanner for Planner {
        fn plan(&self, actor: &ActorHandle, run_id: &str) -> Result<(), String> {
            self.0.fetch_add(1, Ordering::SeqCst);
            let state = actor.get(run_id.into())?.ok_or("unknown run")?;
            let graph = Graph {
                nodes: vec![Node {
                    when: None,
                    optional: Vec::new(),
                    id: "step".into(),
                    depends_on: vec![],
                    kind: NodeKind::Activity,
                    inputs: BTreeMap::new(),
                    outputs: BTreeMap::new(),
                }],
            };
            let command = CommandEnvelope {
                id: "plan".into(),
                expected_revision: state.revision,
                command: Command::PatchGraph {
                    graph,
                    expected_graph_revision: 0,
                    grants: state.definition.grants.clone(),
                    acceptance: state.definition.acceptance.clone(),
                },
            };
            let plan = ExecutionPlan {
                graph_revision: 1,
                activities: BTreeMap::from([(
                    "step".into(),
                    ActivityDescriptor {
                        kind: "step".into(),
                        payload: serde_json::Value::Null,
                    },
                )]),
            };
            actor.install_initial_plan(run_id.into(), command, plan)?;
            Ok(())
        }
    }

    pub fn goal() -> WorkflowDefinition {
        WorkflowDefinition {
            version: SCHEMA_VERSION,
            goal: "Summarize notes".into(),
            initial_graph: None,
            planning_policy: PlanningPolicy::Initial,
            grants: vec![],
            budgets: BTreeMap::new(),
            acceptance: vec![],
        }
    }

    /// The running host; dropping it shuts the host down and joins it.
    pub struct Service {
        dir: PathBuf,
        thread: Option<thread::JoinHandle<Result<(), String>>>,
        pub planned: Arc<AtomicUsize>,
    }
    impl Service {
        pub fn start(dir: &Path, driver: Step) -> Self {
            Self::host(dir, driver, true)
        }

        /// A service that runs activities but plans none: one with no model, as a web-search-only
        /// CLI service.
        pub fn without_planner(dir: &Path, driver: Step) -> Self {
            Self::host(dir, driver, false)
        }

        fn host(dir: &Path, driver: Step, planning: bool) -> Self {
            let planned = Arc::new(AtomicUsize::new(0));
            let (path, planner) = (dir.to_owned(), Planner(planned.clone()));
            let thread = thread::spawn(move || match planning {
                true => host::serve_with_planner(&path, Arc::new(driver), Arc::new(planner)),
                false => host::serve_with_driver(&path, Arc::new(driver)),
            });
            let deadline = Instant::now() + Duration::from_secs(8);
            while Client::connect(dir).is_err() {
                assert!(!thread.is_finished(), "host exited during startup");
                assert!(Instant::now() < deadline, "host did not start");
                thread::sleep(Duration::from_millis(20));
            }
            Self {
                dir: dir.to_owned(),
                thread: Some(thread),
                planned,
            }
        }
    }
    impl Drop for Service {
        fn drop(&mut self) {
            if let Ok(client) = Client::connect(&self.dir) {
                let _ = client.shutdown();
            }
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    /// Poll the app's service view of `run_id` until `done` holds.
    pub fn wait_for(
        app: &larch_app::App,
        run_id: &str,
        done: impl Fn(&TaskCard) -> bool,
    ) -> TaskCard {
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            let (_, cards) = app.task_service().unwrap().snapshot().unwrap();
            let card = cards.into_iter().find(|c| c.run_id == run_id).unwrap();
            if done(&card) {
                return card;
            }
            assert!(Instant::now() < deadline, "run stuck at {:?}", card.status);
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn a_task_saved_through_the_store_adapter_waits_for_an_explicit_start() {
        use larch_task::types::RunStatus;
        let dir = super::Temp::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let service = Service::start(
            &dir.0,
            Step {
                failures: 0,
                calls: calls.clone(),
            },
        );
        let app = larch_app::App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
        let mut api = app.task_api().unwrap();
        let saved = api.create(1, "save", "goal".into(), goal()).unwrap();
        assert_eq!(saved.status, RunStatus::Paused);
        // A probe started after the save runs to its end over several worker passes, each of
        // which lists every run: a Ready goal-only run would be planned and run by then.
        api.create(1, "save", "probe".into(), goal()).unwrap();
        let tasks = app.task_service().unwrap();
        tasks.start_goal("probe", "start", 0).unwrap();
        wait_for(&app, "probe", |c| c.status == RunStatus::Succeeded);
        let card = api.inspect("goal").unwrap().unwrap();
        assert_eq!((card.status, card.revision), (RunStatus::Paused, 0));
        assert!(card.graph.is_none());
        assert_eq!(card.section, TaskSection::Running);
        assert_eq!(service.planned.load(Ordering::SeqCst), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        tasks.start_goal("goal", "start", 0).unwrap();
        let done = wait_for(&app, "goal", |c| c.status == RunStatus::Succeeded);
        assert_eq!(done.result.as_deref(), Some("done"));
        assert_eq!(service.planned.load(Ordering::SeqCst), 2);
    }

    /// The service client pages events as the store adapter does, and names a repeated step.
    #[test]
    fn the_service_client_pages_events_and_names_a_repeated_step() {
        use larch_app::task::PlanningPolicy;
        let dir = super::Temp::new();
        let _service = Service::start(
            &dir.0,
            Step {
                failures: 0,
                calls: Arc::new(AtomicUsize::new(0)),
            },
        );
        let app = larch_app::App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
        let tasks = app.task_service().unwrap();
        let steps = |s: &[&str]| s.iter().map(|s| s.to_string()).collect();
        let error = tasks
            .create_intent(
                "c0",
                "r0".into(),
                "Go".into(),
                steps(&["a", "b", "a"]),
                vec![],
                PlanningPolicy::Disabled,
            )
            .unwrap_err();
        assert!(error.contains("Step 3 repeats \"a\""), "{error}");
        tasks
            .create_intent(
                "c1",
                "r1".into(),
                "Go".into(),
                vec![],
                vec![],
                PlanningPolicy::Initial,
            )
            .unwrap();
        tasks.start_goal("r1", "start", 0).unwrap();
        // Started, planned and run: several events.
        wait_for(&app, "r1", |c| {
            c.status == larch_task::types::RunStatus::Succeeded
        });
        let page = tasks.events("r1", 0, 1).unwrap();
        assert_eq!(
            (page.events.len(), page.next_cursor, page.has_more),
            (1, 1, true)
        );
        let rest = tasks.events("r1", page.next_cursor, 999).unwrap();
        assert!(!rest.has_more);
        assert_eq!(rest.next_cursor, rest.events.last().unwrap().revision);
        assert!(tasks.events("r1", 0, 1000).is_err());
    }

    /// Editing a schedule keeps the budget the composer does not show and saves its "done when".
    #[test]
    fn a_schedule_edit_keeps_its_budget_and_saves_its_done_when() {
        use larch_app::task::PlanningPolicy;
        use larch_task::schedule::{Policy, Schedule};
        let dir = super::Temp::new();
        let _service = Service::start(
            &dir.0,
            Step {
                failures: 0,
                calls: Arc::new(AtomicUsize::new(0)),
            },
        );
        let app = larch_app::App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
        let tasks = app.task_service().unwrap();
        let mut definition = goal();
        definition.planning_policy = PlanningPolicy::Disabled;
        definition.initial_graph = Some(Graph {
            nodes: vec![Node {
                when: None,
                optional: Vec::new(),
                id: "Summarize".into(),
                depends_on: vec![],
                kind: NodeKind::Activity,
                inputs: Default::default(),
                outputs: Default::default(),
            }],
        });
        definition.budgets = BTreeMap::from([("model_tokens".into(), 5000)]);
        definition.acceptance = vec!["cites every note".into()];
        let plan = ExecutionPlan {
            graph_revision: 0,
            activities: larch_app::task::step_activities(
                definition.initial_graph.as_ref().unwrap(),
            )
            .unwrap(),
        };
        let at = chrono::Utc::now() + chrono::Duration::days(1);
        let client = Client::connect(&dir.0).unwrap();
        client
            .create_schedule(
                "s",
                Schedule::Once { at },
                Policy::default(),
                definition,
                plan,
                chrono::Utc::now(),
            )
            .unwrap();
        let revision = tasks.schedules().unwrap()[0].revision;
        tasks
            .update_schedule(
                "s",
                revision,
                Schedule::Once { at },
                Policy::default(),
                "Summarize the notes briefly".into(),
                vec!["Summarize".into()],
                vec!["cites every note and its date".into()],
                PlanningPolicy::Disabled,
            )
            .unwrap();
        let saved = &tasks.schedules().unwrap()[0].definition;
        assert_eq!(saved.goal, "Summarize the notes briefly");
        assert_eq!(saved.budgets["model_tokens"], 5000);
        assert_eq!(saved.acceptance, ["cites every note and its date"]);
        // An edit from an older revision is refused before anything is sent.
        let stale = tasks
            .update_schedule(
                "s",
                revision,
                Schedule::Once { at },
                Policy::default(),
                "x".into(),
                vec!["Summarize".into()],
                vec![],
                PlanningPolicy::Disabled,
            )
            .unwrap_err();
        assert!(stale.contains("stale revision"), "{stale}");
    }

    /// A paused workflow of the steps `a`, a join, then `b`, saved through the store adapter.
    fn save_steps(app: &larch_app::App) -> u64 {
        let node = |id: &str, depends_on: &[&str], kind: NodeKind| Node {
            when: None,
            optional: Vec::new(),
            id: id.into(),
            depends_on: depends_on.iter().map(|d| d.to_string()).collect(),
            kind,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        };
        let definition = WorkflowDefinition {
            initial_graph: Some(Graph {
                nodes: vec![
                    node("a", &[], NodeKind::Activity),
                    node("joined", &["a"], NodeKind::Join),
                    node("b", &["joined"], NodeKind::Activity),
                ],
            }),
            planning_policy: PlanningPolicy::Disabled,
            ..goal()
        };
        let mut api = app.task_api().unwrap();
        api.create(1, "save", "steps".into(), definition)
            .unwrap()
            .revision
    }

    #[test]
    fn a_workflows_own_steps_start_and_run_past_a_join() {
        use larch_task::types::RunStatus;
        let dir = super::Temp::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let service = Service::start(
            &dir.0,
            Step {
                failures: 0,
                calls: calls.clone(),
            },
        );
        let app = larch_app::App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
        let revision = save_steps(&app);
        app.task_service()
            .unwrap()
            .start_steps("steps", "start", revision)
            .unwrap();
        let done = wait_for(&app, "steps", |c| c.status == RunStatus::Succeeded);
        assert_eq!(done.result.as_deref(), Some("done"));
        // One activity per step; the join settles without one.
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(service.planned.load(Ordering::SeqCst), 0);
        // Each activity left its time; this driver counts no tokens.
        let stats = app.task_service().unwrap().step_stats("steps").unwrap();
        assert_eq!(
            stats
                .iter()
                .map(|s| (s.node_id.as_str(), s.tokens))
                .collect::<Vec<_>>(),
            [("a", None), ("b", None)]
        );
        // Open as chat: the goal asks, the steps' results answer, the goal names it.
        let chat = app.chat_from_task("steps").unwrap();
        let messages = app.open_chat(&chat).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "Summarize notes");
        let answer = &messages[1].content;
        assert!(
            answer.contains("**Step 1: a**\n\ndone")
                && answer.contains("**Step 2: b**\n\ndone")
                && !answer.contains("joined"),
            "{answer}"
        );
        let listed = app.list_chats().unwrap();
        assert_eq!(listed[0].title, "Summarize notes");
    }

    #[test]
    fn steps_never_start_on_a_service_without_model_steps() {
        let dir = super::Temp::new();
        let calls = Arc::new(AtomicUsize::new(0));
        // A driver but no model, as a web-search-only service: a plan would be installed for
        // good, then fail at its first step.
        let _service = Service::without_planner(
            &dir.0,
            Step {
                failures: 0,
                calls: calls.clone(),
            },
        );
        let app = larch_app::App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
        let revision = save_steps(&app);
        let tasks = app.task_service().unwrap();
        assert!(tasks.snapshot().unwrap().0.execution);
        let refused = tasks.start_steps("steps", "start", revision).unwrap_err();
        assert_eq!(refused, larch_app::task::NO_MODEL_STEPS);
        let card = app.task_api().unwrap().inspect("steps").unwrap().unwrap();
        assert_eq!(card.revision, revision, "nothing was installed");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_failed_activity_waits_with_its_reason_and_resume_retries_it() {
        use larch_app::task::{TASK_API_VERSION, TaskCommand, TaskControl};
        use larch_task::types::RunStatus;
        let dir = super::Temp::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let _service = Service::start(
            &dir.0,
            Step {
                failures: 1,
                calls: calls.clone(),
            },
        );
        let app = larch_app::App::new(dir.0.clone(), larch_kernels::CPU, |_| {}).unwrap();
        app.task_api()
            .unwrap()
            .create(1, "save", "goal".into(), goal())
            .unwrap();
        let tasks = app.task_service().unwrap();
        tasks.start_goal("goal", "start", 0).unwrap();
        let waiting = wait_for(&app, "goal", |c| c.status == RunStatus::Waiting);
        assert_eq!(waiting.section, TaskSection::NeedsYou);
        assert_eq!(waiting.attempt_failure.as_deref(), Some("tool unavailable"));
        assert!(!waiting.unresolved_effects);
        assert!(waiting.pending_decision.is_none() && waiting.child_failure.is_none());
        let reply = tasks
            .control(TaskCommand {
                version: TASK_API_VERSION,
                run_id: "goal".into(),
                command_id: "retry".into(),
                expected_revision: waiting.revision,
                control: TaskControl::Resume,
            })
            .unwrap();
        assert_eq!(reply.status, RunStatus::Ready);
        let done = wait_for(&app, "goal", |c| c.status == RunStatus::Succeeded);
        assert_eq!(done.result.as_deref(), Some("done"));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
