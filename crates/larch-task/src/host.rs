//! Single-owner task service. An explicitly configured driver runs on one joined worker; the
//! default command exposes durable task authoring and controls without inference ownership.
use crate::{
    runtime::ActorHandle,
    types::{Command, CommandEnvelope, RunState},
};
use serde::{Deserialize, Serialize};

/// Initial graph planning performed outside the shared workflow engine.
pub trait InitialPlanner: Send + Sync {
    fn plan(&self, actor: &ActorHandle, run_id: &str) -> Result<(), String>;
}

pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_FRAME: usize = 1024 * 1024;

#[derive(Serialize, Deserialize)]
pub struct Request {
    pub version: u32,
    pub token: String,
    pub action: Action,
}
// Preserve the public protocol's existing owned payload types.
#[allow(clippy::large_enum_variant)]
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    Capabilities,
    /// The user's answer to a pending decision; the host stamps the time.
    RespondDecision {
        run_id: String,
        command_id: String,
        expected_revision: u64,
        request_id: String,
        action_digest: String,
        approve: bool,
    },
    ListSchedules,
    /// Undelivered notifications; uncertain ones were claimed and never resolved.
    Notifications {
        limit: u32,
    },
    /// Marks a notification in flight for one transport before it is shown, so a crash leaves
    /// it uncertain; replies with a `NotificationClaim`.
    ClaimNotification {
        id: String,
        transport: String,
    },
    /// The claiming transport's verdict: a receipt commits delivery, none returns it to pending.
    ResolveNotification {
        id: String,
        transport: String,
        receipt: Option<String>,
    },
    EnableSchedule {
        id: String,
        expected_revision: u64,
        enabled: bool,
    },
    /// Deletes a schedule whose runs are all settled; the runs stay.
    DeleteSchedule {
        id: String,
        expected_revision: u64,
    },
    /// Deletes a settled run that no schedule occurrence, child, pin or receipt refers to.
    DeleteRun {
        run_id: String,
    },
    Start {
        run_id: String,
        plan: crate::execution::ExecutionPlan,
        command_id: String,
        expected_revision: u64,
    },
    StartGoal {
        run_id: String,
        command_id: String,
        expected_revision: u64,
    },
    CreateSchedule {
        id: String,
        schedule: crate::schedule::Schedule,
        policy: crate::schedule::Policy,
        definition: crate::types::WorkflowDefinition,
        plan: crate::execution::ExecutionPlan,
        from: chrono::DateTime<chrono::Utc>,
    },
    /// Replaces a saved schedule's rule, policy, goal and steps; the new rule counts from now.
    UpdateSchedule {
        id: String,
        expected_revision: u64,
        schedule: crate::schedule::Schedule,
        policy: crate::schedule::Policy,
        definition: crate::types::WorkflowDefinition,
        plan: crate::execution::ExecutionPlan,
    },
    /// Each attempt's wall time and tokens, oldest first: [`STATS_PAGE`] of them from `offset`.
    StepStats {
        run_id: String,
        #[serde(default)]
        offset: usize,
    },
    /// The run's execution plan: what each step does.
    Plan {
        run_id: String,
    },
    Create {
        command_id: String,
        state: RunState,
    },
    Get {
        run_id: String,
    },
    List,
    /// The runs from `offset`, as many as fit [`LIST_BUDGET`] bytes (one at least), each step's
    /// result cut to [`LIST_PREVIEW`] characters; `Get` has them whole.
    ListPage {
        offset: usize,
    },
    Events {
        run_id: String,
        cursor: u64,
        limit: u32,
    },
    Control {
        run_id: String,
        command: CommandEnvelope,
    },
    Shutdown,
}
/// Exactly one field is present. `result` is present on success even when it is null (an
/// unknown run's `get`), so an empty or truncated envelope never reads as success.
#[derive(Serialize, Deserialize)]
pub struct Response {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present"
    )]
    pub result: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
/// A present field is `Some`, null included; only an absent one takes the default `None`.
fn present<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error> {
    serde_json::Value::deserialize(deserializer).map(Some)
}

/// The most attempts one `StepStats` answer carries: about 130 KB, well inside a frame.
pub const STATS_PAGE: usize = 1000;

/// The bytes of runs one `ListPage` answer gathers before it stops: a frame's half, so the
/// envelope and the run that crosses it still fit.
pub const LIST_BUDGET: usize = MAX_FRAME / 2;

/// The most of each step's result a `ListPage` run carries, as a card shows it.
pub const LIST_PREVIEW: usize = 2000;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    pub protocol_version: u32,
    pub authoring: bool,
    pub controls: bool,
    #[serde(default)]
    pub retry_child: bool,
    pub execution: bool,
    #[serde(default)]
    pub planning: bool,
    #[serde(default)]
    pub explicit_start: bool,
    pub schedules: bool,
    pub decisions: bool,
    /// This host raises watch notifications; the transport actions answer either way.
    #[serde(default)]
    pub notifications: bool,
    pub autonomous_application: bool,
    /// This host answers `StepStats` and `Plan`; an older one has neither.
    #[serde(default)]
    pub step_stats: bool,
    /// This host answers `UpdateSchedule`.
    #[serde(default)]
    pub schedule_updates: bool,
    /// This host answers `ListPage`; an older one sends every run whole in one `List` frame.
    #[serde(default)]
    pub list_pages: bool,
    /// This host answers `DeleteSchedule` and `DeleteRun`.
    #[serde(default)]
    pub deletes: bool,
    /// The activity kinds this service runs; None when its driver does not enumerate them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activity_kinds: Option<std::collections::BTreeSet<String>>,
}

fn response(result: Result<serde_json::Value, String>) -> Response {
    match result {
        Ok(value) => Response {
            result: Some(value),
            error: None,
        },
        Err(error) => Response {
            result: None,
            error: Some(error),
        },
    }
}

/// Accepts user controls and explicit plan activation. Clients cannot forge driver completions
/// or effect receipts; those must pass through the trusted executor and actor.
fn json<T: Serialize>(value: T) -> Result<serde_json::Value, String> {
    serde_json::to_value(value).map_err(|e| e.to_string())
}
pub fn handle(
    actor: &ActorHandle,
    action: Action,
    execution_enabled: bool,
    planning_enabled: bool,
) -> (Response, bool) {
    let result = match action {
        Action::Capabilities => json(Capabilities {
            protocol_version: PROTOCOL_VERSION,
            authoring: true,
            controls: true,
            retry_child: true,
            execution: execution_enabled,
            planning: planning_enabled,
            explicit_start: true,
            schedules: true,
            decisions: true,
            // No watch evaluator runs in this host yet, so it never raises a notification.
            notifications: false,
            autonomous_application: false,
            step_stats: true,
            schedule_updates: true,
            list_pages: true,
            deletes: true,
            // Filled in by the serving loop, which holds the driver.
            activity_kinds: None,
        }),
        Action::RespondDecision {
            run_id,
            command_id,
            expected_revision,
            request_id,
            action_digest,
            approve,
        } => crate::execution::now_secs().and_then(|now| {
            actor
                .execute(
                    run_id,
                    CommandEnvelope {
                        id: command_id,
                        expected_revision,
                        command: Command::RespondDecision {
                            request_id,
                            action_digest,
                            approve,
                            now,
                        },
                    },
                )
                .and_then(json)
        }),
        Action::ListSchedules => actor.list_schedules().and_then(json),
        Action::Notifications { limit } => actor.pending_notifications(limit).and_then(json),
        Action::ClaimNotification { id, transport } => actor
            .claim_notification(id, transport, chrono::Utc::now())
            .and_then(json),
        Action::ResolveNotification {
            id,
            transport,
            receipt,
        } => actor
            .resolve_notification(id, transport, receipt)
            .and_then(json),
        Action::EnableSchedule {
            id,
            expected_revision,
            enabled,
        } => {
            if enabled && !execution_enabled {
                Err("activity execution is not configured".into())
            } else {
                actor
                    .enable_schedule(id, expected_revision, enabled)
                    .and_then(json)
            }
        }
        Action::DeleteSchedule {
            id,
            expected_revision,
        } => actor.delete_schedule(id, expected_revision).and_then(json),
        Action::DeleteRun { run_id } => actor.delete_run(run_id).and_then(json),
        Action::CreateSchedule {
            id,
            schedule,
            policy,
            definition,
            plan,
            from,
        } => actor
            .create_schedule_with_plan(id, schedule, policy, definition, plan, from)
            .map(|_| serde_json::json!({"saved":true, "enabled":false})),
        Action::Start {
            run_id,
            plan,
            command_id,
            expected_revision,
        } => {
            if !execution_enabled {
                Err("activity execution is not configured".into())
            } else {
                actor
                    .start_plan(run_id, command_id, expected_revision, plan)
                    .and_then(json)
            }
        }
        Action::StartGoal {
            run_id,
            command_id,
            expected_revision,
        } => {
            if !planning_enabled {
                Err("goal planning is not configured".into())
            } else {
                actor
                    .execute(
                        run_id,
                        CommandEnvelope {
                            id: command_id,
                            expected_revision,
                            command: Command::Resume,
                        },
                    )
                    .and_then(json)
            }
        }
        Action::UpdateSchedule {
            id,
            expected_revision,
            schedule,
            policy,
            definition,
            plan,
        } => actor
            .update_schedule_with_plan(
                id,
                expected_revision,
                schedule,
                policy,
                definition,
                plan,
                chrono::Utc::now(),
            )
            .and_then(json),
        // ponytail: each page reads the run's every row; a store-side LIMIT/OFFSET when a run's
        // attempts reach the hundreds of thousands
        Action::StepStats { run_id, offset } => actor
            .step_stats(run_id)
            .map(|all| {
                all.into_iter()
                    .skip(offset)
                    .take(STATS_PAGE)
                    .collect::<Vec<_>>()
            })
            .and_then(json),
        Action::Plan { run_id } => actor.execution_plan(run_id).and_then(json),
        Action::Create { command_id, state } => {
            actor.create_paused(command_id, state).and_then(json)
        }
        Action::Get { run_id } => actor.get(run_id).and_then(json),
        Action::List => actor.list().and_then(json),
        // ponytail: one run whose own result passes the frame still fails here; cut `result` too
        // when a single step's answer ever comes near 1 MiB.
        Action::ListPage { offset } => actor.list().and_then(|all| {
            let (mut page, mut bytes) = (Vec::new(), 0);
            for mut run in all.into_iter().skip(offset) {
                for result in run.node_results.values_mut() {
                    if let Some((at, _)) = result.char_indices().nth(LIST_PREVIEW) {
                        result.truncate(at);
                    }
                }
                bytes += serde_json::to_vec(&run).map_err(|e| e.to_string())?.len();
                if !page.is_empty() && bytes > LIST_BUDGET {
                    break;
                }
                page.push(run);
            }
            json(page)
        }),
        Action::Events {
            run_id,
            cursor,
            limit,
        } => {
            if limit > 1000 {
                Err("event limit exceeds 1000".into())
            } else {
                actor.events_after(run_id, cursor, limit).and_then(json)
            }
        }
        Action::Control { run_id, command } => {
            // Only a run saved paused at create resumes from revision 0; Start and StartGoal
            // apply the service's gates to that first start, so a control must not bypass them.
            if matches!(command.command, Command::Resume) && command.expected_revision == 0 {
                Err("a run that never started is started with Start or StartGoal".into())
            } else if !matches!(
                &command.command,
                Command::Pause
                    | Command::Resume
                    | Command::Cancel
                    | Command::RevokeGrant { .. }
                    | Command::RetryChild { .. }
                    // The user's word that an uncertain effect did not happen; a receipt
                    // claiming it did comes only from the actor.
                    | Command::ReconcileEffect { receipt: None, .. }
            ) {
                Err("command requires the task actor or a trusted decision handler".into())
            } else {
                actor.execute(run_id, command).and_then(json)
            }
        }
        Action::Shutdown => {
            return (
                response(Ok(serde_json::json!({"shutting_down":true}))),
                true,
            );
        }
    };
    (response(result), false)
}

pub fn serve(data_dir: &std::path::Path) -> Result<(), String> {
    serve_inner(data_dir, None, None)
}
/// `serve`, calling `ready` with the token once this process owns the folder and listens.
pub fn serve_notify(data_dir: &std::path::Path, ready: Ready) -> Result<(), String> {
    serve_inner(data_dir, None, Some(ready))
}
/// `serve_with_components`, calling `ready` with the token once this process owns the folder,
/// its components are built and it listens.
pub fn serve_with_components_notify<F>(
    data_dir: &std::path::Path,
    factory: F,
    ready: Ready,
) -> Result<(), String>
where
    F: FnOnce(
            ActorHandle,
            std::sync::Arc<std::sync::atomic::AtomicBool>,
        ) -> Result<HostComponents, String>
        + 'static,
{
    serve_inner(data_dir, Some(Box::new(factory)), Some(ready))
}
/// Told the service token once the service is ready: a caller can then tell its own service
/// from one another process started on the same folder.
pub type Ready = Box<dyn FnOnce(&str) + Send>;
pub fn serve_with_driver(
    data_dir: &std::path::Path,
    driver: std::sync::Arc<dyn crate::execution::Driver>,
) -> Result<(), String> {
    serve_inner(
        data_dir,
        Some(Box::new(move |_, _| Ok((driver, None)))),
        None,
    )
}
pub fn serve_with_planner(
    data_dir: &std::path::Path,
    driver: std::sync::Arc<dyn crate::execution::Driver>,
    planner: std::sync::Arc<dyn InitialPlanner>,
) -> Result<(), String> {
    serve_inner(
        data_dir,
        Some(Box::new(move |_, _| Ok((driver, Some(planner))))),
        None,
    )
}
pub fn serve_with_factory<F>(data_dir: &std::path::Path, factory: F) -> Result<(), String>
where
    F: FnOnce(
            ActorHandle,
            std::sync::Arc<std::sync::atomic::AtomicBool>,
        ) -> Result<std::sync::Arc<dyn crate::execution::Driver>, String>
        + 'static,
{
    serve_inner(
        data_dir,
        Some(Box::new(move |actor, stop| {
            factory(actor, stop).map(|driver| (driver, None))
        })),
        None,
    )
}
pub fn serve_with_components<F>(data_dir: &std::path::Path, factory: F) -> Result<(), String>
where
    F: FnOnce(
            ActorHandle,
            std::sync::Arc<std::sync::atomic::AtomicBool>,
        ) -> Result<HostComponents, String>
        + 'static,
{
    serve_inner(data_dir, Some(Box::new(factory)), None)
}
pub type HostComponents = (
    std::sync::Arc<dyn crate::execution::Driver>,
    Option<std::sync::Arc<dyn InitialPlanner>>,
);
type DriverFactory = Box<
    dyn FnOnce(
        ActorHandle,
        std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<HostComponents, String>,
>;
fn serve_inner(
    data_dir: &std::path::Path,
    factory: Option<DriverFactory>,
    ready: Option<Ready>,
) -> Result<(), String> {
    use crate::ipc::{read_frame, write_frame};
    #[cfg(unix)]
    use std::os::unix::fs::OpenOptionsExt;
    use std::{
        fs::{self, OpenOptions},
        io::Write,
        time::Duration,
    };
    // ponytail: on Windows the data directory's inherited per-user ACL guards the token file and
    // the pipe's own DACL admits this user only; set explicit file ACLs if a shared --data-dir
    // ever ships.
    let dir = runtime_dir(data_dir)?;
    let mut owner = OpenOptions::new();
    owner.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    owner.mode(0o600);
    let owner = owner
        .open(dir.join("owner.lock"))
        .map_err(|e| e.to_string())?;
    owner
        .try_lock()
        .map_err(|e| format!("runtime data directory is already owned: {e}"))?;
    let token = random_hex()?;
    let token_path = dir.join("token");
    if token_path.exists() {
        fs::remove_file(&token_path).map_err(|e| e.to_string())?;
    }
    let mut token_file = OpenOptions::new();
    token_file.write(true).create_new(true);
    #[cfg(unix)]
    token_file.mode(0o600);
    let mut token_file = token_file.open(&token_path).map_err(|e| e.to_string())?;
    token_file
        .write_all(token.as_bytes())
        .and_then(|_| token_file.sync_all())
        .map_err(|e| e.to_string())?;
    let (mut listener, endpoint_path) = listen(&dir)?;
    struct ActorGuard(ActorHandle);
    impl Drop for ActorGuard {
        fn drop(&mut self) {
            let _ = self.0.shutdown();
        }
    }
    let actor = ActorGuard(ActorHandle::start(
        crate::store::database_path(data_dir),
        64,
    )?);
    // A process death or quit cannot leave an old lease authoritative.
    for state in actor.0.list()? {
        if state.attempt.is_some() {
            actor.0.execute(
                state.id.clone(),
                CommandEnvelope {
                    id: format!("service.recover:{}", state.revision),
                    expected_revision: state.revision,
                    command: Command::Recover,
                },
            )?;
        }
    }
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let components = match factory {
        Some(factory) => Some(factory(actor.0.clone(), stop.clone())?),
        None => None,
    };
    // Exactly one worker owns driver calls. Shutdown joins it before the actor and data lock.
    // Driver implementations must be bounded; arbitrary external tools are not registered here.
    let execution_enabled = components.is_some();
    let kinds = components
        .as_ref()
        .and_then(|(driver, _)| driver.registered_kinds());
    let planning_enabled = components
        .as_ref()
        .is_some_and(|(_, planner)| planner.is_some());
    struct WorkerGuard {
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        worker: Option<std::thread::JoinHandle<()>>,
    }
    impl Drop for WorkerGuard {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::Release);
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
    }
    let worker = if let Some((driver, planner)) = components {
        let actor = actor.0.clone();
        let stop = stop.clone();
        Some(
            std::thread::Builder::new()
                .name("larch-activity".into())
                .spawn(move || {
                    let executor = crate::execution::Executor::new(actor.clone(), driver);
                    // Each failure is reported once per subject until it changes or clears,
                    // not every 100 ms.
                    let mut failing = std::collections::BTreeMap::new();
                    let mut report = |subject: String, outcome: Result<(), String>| match outcome {
                        Ok(()) => {
                            failing.remove(&subject);
                        }
                        Err(error) => {
                            if failing.get(&subject) != Some(&error) {
                                eprintln!("larch-task: {subject}: {error}");
                                failing.insert(subject, error);
                            }
                        }
                    };
                    while !stop.load(std::sync::atomic::Ordering::Acquire) {
                        // Occurrence creation is durable and Executor atomically claims the start outbox.
                        // Notification outboxes remain pending until a separate transport reconciles them.
                        match actor.pump_due(chrono::Utc::now(), 1024) {
                            Ok(results) => {
                                for result in results {
                                    report(
                                        format!("schedule {}", result.schedule_id),
                                        result.outcome.map(drop),
                                    );
                                }
                            }
                            Err(error) => report("schedules".into(), Err(error)),
                        }
                        let runs = actor.list_active();
                        report(
                            "run list".into(),
                            runs.as_ref().map(drop).map_err(Clone::clone),
                        );
                        for run in runs.unwrap_or_default() {
                            if stop.load(std::sync::atomic::Ordering::Acquire) {
                                break;
                            }
                            let outcome = if run.status == crate::types::RunStatus::Ready {
                                if run.graph.is_none() {
                                    // ponytail: a queued goal-only occurrence is offered to the
                                    // planner each tick and refused at dispatch until its turn;
                                    // skip it here if that context fitting shows in a profile.
                                    planner
                                        .as_ref()
                                        .map_or(Ok(()), |planner| planner.plan(&actor, &run.id))
                                } else {
                                    match actor.execution_plan(run.id.clone()) {
                                        Ok(Some(_)) => executor.tick(&run.id).map(drop),
                                        other => other.map(drop),
                                    }
                                }
                            } else if run.status == crate::types::RunStatus::Waiting
                                && (run.waiting_child.is_some() && run.child_failure.is_none()
                                    || run.waiting_decision.is_some())
                            {
                                executor.tick(&run.id).map(drop)
                            } else {
                                Ok(())
                            };
                            report(format!("run {}", run.id), outcome);
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                })
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    if let Some(ready) = ready {
        ready(&token);
    }
    let worker = WorkerGuard { stop, worker };
    loop {
        // ponytail: one client at a time, each request bounded by REQUEST_DEADLINE; serve each
        // connection on its own thread when the desktop and the CLI show the stall together.
        let Some(mut stream) = accept(&mut listener)? else {
            continue;
        };
        let request = read_frame(
            &mut Deadline {
                stream: &mut stream,
                until: std::time::Instant::now() + REQUEST_DEADLINE,
            },
            MAX_FRAME,
        )
        .ok()
        .and_then(|body| serde_json::from_slice::<Request>(&body).ok());
        let (reply, stop) = match request {
            Some(request)
                if request.version == PROTOCOL_VERSION
                    && secret_eq(request.token.as_bytes(), token.as_bytes()) =>
            {
                // Plans become immutable once stored; refuse kinds the driver cannot run first.
                let refused = match (&request.action, &kinds) {
                    (
                        Action::Start { plan, .. }
                        | Action::CreateSchedule { plan, .. }
                        | Action::UpdateSchedule { plan, .. },
                        Some(kinds),
                    ) => crate::execution::check_plan_kinds(kinds, plan).err(),
                    _ => None,
                };
                // A worker that panicked runs nothing: stop offering execution and planning.
                let running = worker.worker.as_ref().is_some_and(|w| !w.is_finished());
                let capabilities = matches!(request.action, Action::Capabilities);
                match refused {
                    Some(error) => (response(Err(error)), false),
                    None => {
                        let (mut reply, stop) = handle(
                            &actor.0,
                            request.action,
                            execution_enabled && running,
                            planning_enabled && running,
                        );
                        if capabilities && let Some(answer) = reply.result.as_mut() {
                            answer["activity_kinds"] = serde_json::json!(kinds);
                        }
                        (reply, stop)
                    }
                }
            }
            _ => (
                response(Err("invalid request or authentication".into())),
                false,
            ),
        };
        // The reply gets its own budget, whatever the request left.
        if let Ok(bytes) = serde_json::to_vec(&reply)
            && set_timeout(&mut stream, REQUEST_DEADLINE).is_ok()
        {
            let _ = write_frame(&mut stream, &bytes, MAX_FRAME);
        }
        if stop {
            break;
        }
    }
    drop(worker);
    actor.0.shutdown()?;
    let _ = fs::remove_file(endpoint_path);
    let _ = fs::remove_file(token_path);
    Ok(())
}

/// 32 random bytes as hex: the service token, and on Windows the pipe's unguessable name.
fn random_hex() -> Result<String, String> {
    let mut random = [0u8; 32];
    getrandom::fill(&mut random).map_err(|e| e.to_string())?;
    Ok(random.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// The bound socket and the file that names it.
#[cfg(unix)]
fn listen(
    dir: &std::path::Path,
) -> Result<(std::os::unix::net::UnixListener, std::path::PathBuf), String> {
    let socket_path = dir.join("service.sock");
    if socket_path.exists() {
        std::fs::remove_file(&socket_path).map_err(|e| e.to_string())?;
    }
    let listener =
        std::os::unix::net::UnixListener::bind(&socket_path).map_err(|e| e.to_string())?;
    Ok((listener, socket_path))
}

/// How long a client may take to send its whole request, from accept; the reply gets as long.
const REQUEST_DEADLINE: std::time::Duration = std::time::Duration::from_secs(2);
#[cfg(unix)]
type Stream = std::os::unix::net::UnixStream;
#[cfg(windows)]
type Stream = larch_sys::pipe::PipeStream;
/// Reads sharing one deadline, so a client trickling bytes cannot hold the only server thread.
struct Deadline<'a> {
    stream: &'a mut Stream,
    until: std::time::Instant,
}
impl std::io::Read for Deadline<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self
            .until
            .saturating_duration_since(std::time::Instant::now());
        if left.is_zero() {
            return Err(std::io::ErrorKind::TimedOut.into());
        }
        set_timeout(self.stream, left)?;
        self.stream.read(buf)
    }
}
#[cfg(unix)]
fn set_timeout(stream: &mut Stream, timeout: std::time::Duration) -> std::io::Result<()> {
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))
}
#[cfg(windows)]
fn set_timeout(stream: &mut Stream, timeout: std::time::Duration) -> std::io::Result<()> {
    stream.set_timeout(timeout)
}

/// The next same-user connection; `None` skips one that failed.
#[cfg(unix)]
fn accept(listener: &mut std::os::unix::net::UnixListener) -> Result<Option<Stream>, String> {
    let stream = match listener.accept() {
        Ok((stream, _)) => stream,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::Interrupted | std::io::ErrorKind::ConnectionAborted
            ) =>
        {
            return Ok(None);
        }
        Err(error) => return Err(error.to_string()),
    };
    Ok(crate::peer::authenticate_peer(&stream)
        .is_ok()
        .then_some(stream))
}

/// A pipe under a fresh random name, written to `service.pipe` once it listens. The pipe's DACL
/// admits this user only and the name is unguessable, so no other user can squat it first. The
/// name file is renamed into place, so a client never reads half of it.
#[cfg(windows)]
fn listen(
    dir: &std::path::Path,
) -> Result<(larch_sys::pipe::PipeListener, std::path::PathBuf), String> {
    let name = format!("{}{}", crate::ipc::PIPE_PREFIX, random_hex()?);
    let listener = larch_sys::pipe::PipeListener::bind(&name).map_err(|e| e.to_string())?;
    let path = dir.join("service.pipe");
    let staged = dir.join("service.pipe.new");
    std::fs::write(&staged, &name)
        .and_then(|()| std::fs::rename(&staged, &path))
        .map_err(|e| e.to_string())?;
    Ok((listener, path))
}

/// The next connection; the pipe's DACL already admits this user only. The listener skips a
/// client that leaves early and retries its own failures, so an error here persisted.
#[cfg(windows)]
fn accept(listener: &mut larch_sys::pipe::PipeListener) -> Result<Option<Stream>, String> {
    listener
        .accept(REQUEST_DEADLINE)
        .map(Some)
        .map_err(|e| e.to_string())
}
fn secret_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |diff, (x, y)| diff | (x ^ y)) == 0
}
/// The folder's private `runtime-service` directory, created owner-only (0700 on unix) before
/// anything, the task database included, is created in it; a looser existing one is tightened.
pub fn runtime_dir(data_dir: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let dir = data_dir.join("runtime-service");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
            .map_err(|e| e.to_string())?;
    }
    Ok(dir)
}
/// Advisory ownership of one data folder by the one model-hosting service: a lock file per
/// folder, not per device. Every engine-starting entrypoint must take it on the same folder.
pub struct DeviceLease {
    _file: std::fs::File,
}
impl DeviceLease {
    /// Take the folder's lease; fails while another process holds it.
    pub fn acquire(data_dir: &std::path::Path) -> Result<Self, String> {
        let dir = runtime_dir(data_dir)?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join("device.lock"))
            .map_err(|e| e.to_string())?;
        file.try_lock()
            .map_err(|e| format!("device is already owned by another process: {e}"))?;
        Ok(Self { _file: file })
    }
}
