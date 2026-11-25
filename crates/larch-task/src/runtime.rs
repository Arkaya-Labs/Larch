use crate::{store::SqliteStore, types::*};

/// A single owner for SQLite with a bounded command queue. Drivers must run
/// outside this thread and report their results as fenced commands.
#[derive(Clone)]
pub struct ActorHandle {
    sender: std::sync::mpsc::SyncSender<ActorRequest>,
    thread: std::sync::Arc<std::sync::Mutex<Option<std::thread::JoinHandle<()>>>>,
}
enum ActorRequest {
    FailBudgetReceipt(
        String,
        CommandEnvelope,
        u64,
        String,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    SpawnAndWaitBudgeted(
        crate::execution::BudgetedChildSpawnRequest,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    PollWaited(
        String,
        String,
        std::sync::mpsc::Sender<Result<Option<RunState>, String>>,
    ),
    SpawnAndWait(
        crate::execution::ChildSpawnRequest,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    CollectWaited(
        String,
        String,
        std::sync::mpsc::Sender<Result<RunState, String>>,
    ),
    CheckChildWait(String, String, std::sync::mpsc::Sender<Result<(), String>>),
    SpawnWithPlan(
        crate::execution::ChildSpawnRequest,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    StartPlan(
        String,
        String,
        u64,
        crate::execution::ExecutionPlan,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    CreatePaused(
        String,
        RunState,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    ListSchedules(std::sync::mpsc::Sender<Result<Vec<crate::schedule::ScheduleRecord>, String>>),
    EnableSchedule(
        String,
        u64,
        bool,
        std::sync::mpsc::Sender<Result<u64, String>>,
    ),
    DeleteSchedule(String, u64, std::sync::mpsc::Sender<Result<(), String>>),
    DeleteRun(String, std::sync::mpsc::Sender<Result<(), String>>),
    UpdateWatch(
        String,
        u64,
        crate::watch::Truth,
        Option<String>,
        String,
        String,
        std::sync::mpsc::Sender<Result<crate::watch::WatchState, String>>,
    ),
    PendingNotifications(
        u32,
        std::sync::mpsc::Sender<Result<Vec<crate::watch::PendingNotification>, String>>,
    ),
    NotificationStatus(
        String,
        std::sync::mpsc::Sender<Result<crate::watch::NotificationStatus, String>>,
    ),
    ClaimNotification(
        String,
        String,
        chrono::DateTime<chrono::Utc>,
        std::sync::mpsc::Sender<Result<crate::watch::NotificationClaim, String>>,
    ),
    ResolveNotification(
        String,
        String,
        Option<String>,
        std::sync::mpsc::Sender<Result<(), String>>,
    ),
    EffectiveGrants(
        String,
        std::sync::mpsc::Sender<Result<(u64, Vec<String>), String>>,
    ),
    ReserveGranted(
        String,
        CommandEnvelope,
        String,
        u64,
        std::sync::mpsc::Sender<Result<(CommandReply, bool), String>>,
    ),
    ReconcileBudget(
        String,
        String,
        u64,
        String,
        std::sync::mpsc::Sender<Result<(), String>>,
    ),
    ScheduledOnce(
        String,
        CommandEnvelope,
        std::sync::mpsc::Sender<Result<(CommandReply, bool), String>>,
    ),
    DispatchBudgetOnce(
        String,
        CommandEnvelope,
        u64,
        usize,
        std::sync::mpsc::Sender<Result<(CommandReply, bool), String>>,
    ),
    BudgetScheduledOnce(
        String,
        String,
        CommandEnvelope,
        u64,
        usize,
        std::sync::mpsc::Sender<Result<(CommandReply, bool), String>>,
    ),
    ScheduledForRun(
        String,
        std::sync::mpsc::Sender<Result<Option<(String, bool)>, String>>,
    ),
    UpdateScheduleWithPlan(
        String,
        u64,
        crate::schedule::Schedule,
        crate::schedule::Policy,
        WorkflowDefinition,
        crate::execution::ExecutionPlan,
        chrono::DateTime<chrono::Utc>,
        std::sync::mpsc::Sender<Result<u64, String>>,
    ),
    RecordStep(
        String,
        crate::execution::StepStat,
        std::sync::mpsc::Sender<Result<(), String>>,
    ),
    StepStats(
        String,
        std::sync::mpsc::Sender<Result<Vec<crate::execution::StepStat>, String>>,
    ),
    CreateScheduleWithPlan(
        String,
        crate::schedule::Schedule,
        crate::schedule::Policy,
        WorkflowDefinition,
        crate::execution::ExecutionPlan,
        chrono::DateTime<chrono::Utc>,
        std::sync::mpsc::Sender<Result<(), String>>,
    ),
    InstallInitialPlan(
        String,
        CommandEnvelope,
        crate::execution::ExecutionPlan,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    DispatchOnce(
        String,
        CommandEnvelope,
        std::sync::mpsc::Sender<Result<(CommandReply, bool), String>>,
    ),
    DispatchBudget(
        String,
        CommandEnvelope,
        u64,
        usize,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    SettleBudget(
        String,
        CommandEnvelope,
        u64,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    ClaimScheduled(
        String,
        CommandEnvelope,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    ClaimBudgetScheduled(
        String,
        String,
        CommandEnvelope,
        u64,
        usize,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    InstallPlan(
        String,
        crate::execution::ExecutionPlan,
        std::sync::mpsc::Sender<Result<(), String>>,
    ),
    GetPlan(
        String,
        std::sync::mpsc::Sender<Result<Option<crate::execution::ExecutionPlan>, String>>,
    ),
    RootId(String, std::sync::mpsc::Sender<Result<String, String>>),
    Shutdown(std::sync::mpsc::Sender<()>),
    Create(
        String,
        RunState,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    Execute(
        String,
        CommandEnvelope,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    Get(
        String,
        std::sync::mpsc::Sender<Result<Option<RunState>, String>>,
    ),
    List(std::sync::mpsc::Sender<Result<Vec<RunState>, String>>),
    ListActive(std::sync::mpsc::Sender<Result<Vec<RunState>, String>>),
    Events(
        String,
        u64,
        u32,
        std::sync::mpsc::Sender<Result<Vec<Event>, String>>,
    ),
    Spawn(
        String,
        u64,
        String,
        RunState,
        std::sync::mpsc::Sender<Result<CommandReply, String>>,
    ),
    Message(
        String,
        String,
        String,
        String,
        std::sync::mpsc::Sender<Result<(), String>>,
    ),
    Collect(
        String,
        String,
        std::sync::mpsc::Sender<Result<RunState, String>>,
    ),
    Reserve(
        String,
        CommandEnvelope,
        std::sync::mpsc::Sender<Result<(CommandReply, bool), String>>,
    ),
    Pump(
        chrono::DateTime<chrono::Utc>,
        usize,
        std::sync::mpsc::Sender<Result<Vec<crate::schedule::PumpResult>, String>>,
    ),
    Pending(
        u32,
        std::sync::mpsc::Sender<Result<Vec<crate::schedule::OutboxEntry>, String>>,
    ),
    Ack(String, std::sync::mpsc::Sender<Result<(), String>>),
    CreateSchedule(
        String,
        crate::schedule::Schedule,
        crate::schedule::Policy,
        WorkflowDefinition,
        chrono::DateTime<chrono::Utc>,
        std::sync::mpsc::Sender<Result<(), String>>,
    ),
}
impl ActorHandle {
    pub fn start(path: impl AsRef<std::path::Path>, capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("actor queue capacity must be positive".into());
        }
        let path = path.as_ref().to_path_buf();
        let (sender, receiver) = std::sync::mpsc::sync_channel(capacity);
        let (ready_send, ready_recv) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("larch-task-store".into())
            .spawn(move || {
                let mut store = match SqliteStore::open(path) {
                    Ok(store) => {
                        let _ = ready_send.send(Ok(()));
                        store
                    }
                    Err(error) => {
                        let _ = ready_send.send(Err(error));
                        return;
                    }
                };
                while let Ok(request) = receiver.recv() {
                    match request {
                        ActorRequest::FailBudgetReceipt(run, command, used, receipt, reply) => {
                            let _ =
                                reply
                                    .send(store.fail_budgeted_with_receipt(
                                        &run, &command, used, &receipt,
                                    ));
                        }
                        ActorRequest::SpawnAndWaitBudgeted(request, reply) => {
                            let _ = reply.send(store.spawn_child_and_wait_budgeted(&request));
                        }
                        ActorRequest::PollWaited(parent, child, reply) => {
                            let _ = reply.send(store.poll_waited_child(&parent, &child));
                        }
                        ActorRequest::SpawnAndWait(request, reply) => {
                            let _ = reply.send(store.spawn_child_and_wait(&request));
                        }
                        ActorRequest::CollectWaited(parent, child, reply) => {
                            let _ = reply.send(store.collect_waited_child(&parent, &child));
                        }
                        ActorRequest::CheckChildWait(parent, child, reply) => {
                            let _ = reply.send(store.check_child_wait(&parent, &child));
                        }
                        ActorRequest::StartPlan(id, command, revision, plan, reply) => {
                            let _ = reply.send(store.start_plan(&id, &command, revision, &plan));
                        }
                        ActorRequest::CreatePaused(id, state, reply) => {
                            let _ = reply.send(store.create_paused_idempotent(&id, &state));
                        }
                        ActorRequest::ListSchedules(reply) => {
                            let _ = reply.send(store.list_schedules());
                        }
                        ActorRequest::EnableSchedule(id, revision, enabled, reply) => {
                            let _ = reply.send(store.enable_schedule(
                                &id,
                                revision,
                                enabled,
                                chrono::Utc::now(),
                            ));
                        }
                        ActorRequest::DeleteSchedule(id, revision, reply) => {
                            let _ = reply.send(store.delete_schedule(&id, revision));
                        }
                        ActorRequest::DeleteRun(id, reply) => {
                            let _ = reply.send(store.delete_run(&id));
                        }
                        ActorRequest::UpdateWatch(
                            id,
                            version,
                            truth,
                            cursor,
                            notification,
                            payload,
                            reply,
                        ) => {
                            let _ = reply.send(store.update_watch_notify(
                                &id,
                                version,
                                truth,
                                cursor,
                                &notification,
                                &payload,
                            ));
                        }
                        ActorRequest::PendingNotifications(limit, reply) => {
                            let _ = reply.send(store.pending_notifications(limit));
                        }
                        ActorRequest::NotificationStatus(id, reply) => {
                            let _ = reply.send(store.notification_status(&id));
                        }
                        ActorRequest::ClaimNotification(id, transport, now, reply) => {
                            let _ = reply.send(store.claim_notification(&id, &transport, now));
                        }
                        ActorRequest::ResolveNotification(id, transport, receipt, reply) => {
                            let _ =
                                reply.send(store.resolve_notification(&id, &transport, receipt));
                        }
                        ActorRequest::EffectiveGrants(id, reply) => {
                            let _ = reply.send(store.effective_grants(&id));
                        }
                        ActorRequest::ReserveGranted(id, command, grant, generation, reply) => {
                            let _ = reply.send(
                                store.reserve_granted_effect(&id, &command, &grant, generation),
                            );
                        }
                        ActorRequest::ReconcileBudget(run, attempt, used, receipt, reply) => {
                            let _ = reply.send(
                                store.reconcile_budgeted_usage(&run, &attempt, used, &receipt),
                            );
                        }
                        ActorRequest::ScheduledOnce(id, command, reply) => {
                            let _ = reply.send(store.claim_scheduled_start_once(&id, &command));
                        }
                        ActorRequest::DispatchBudgetOnce(id, command, reserve, capacity, reply) => {
                            let _ = reply.send(
                                store.dispatch_budgeted_once(&id, &command, reserve, capacity),
                            );
                        }
                        ActorRequest::BudgetScheduledOnce(
                            outbox,
                            id,
                            command,
                            reserve,
                            capacity,
                            reply,
                        ) => {
                            let _ = reply.send(store.claim_budgeted_scheduled_start_once(
                                &outbox, &id, &command, reserve, capacity,
                            ));
                        }
                        ActorRequest::ScheduledForRun(id, reply) => {
                            let _ = reply.send(store.scheduled_start_for_run(&id));
                        }
                        ActorRequest::CreateScheduleWithPlan(
                            id,
                            schedule,
                            policy,
                            definition,
                            plan,
                            from,
                            reply,
                        ) => {
                            let _ = reply.send(store.create_schedule_with_plan(
                                &id,
                                &schedule,
                                policy,
                                &definition,
                                &plan,
                                from,
                            ));
                        }
                        ActorRequest::UpdateScheduleWithPlan(
                            id,
                            revision,
                            schedule,
                            policy,
                            definition,
                            plan,
                            from,
                            reply,
                        ) => {
                            let _ = reply.send(store.update_schedule_with_plan(
                                &id,
                                revision,
                                &schedule,
                                policy,
                                &definition,
                                &plan,
                                from,
                            ));
                        }
                        ActorRequest::RecordStep(id, stat, reply) => {
                            let _ = reply.send(store.record_step(&id, &stat));
                        }
                        ActorRequest::StepStats(id, reply) => {
                            let _ = reply.send(store.step_stats(&id));
                        }
                        ActorRequest::InstallInitialPlan(id, command, plan, reply) => {
                            let _ = reply.send(store.install_initial_plan(&id, &command, &plan));
                        }
                        ActorRequest::DispatchOnce(id, command, reply) => {
                            let _ = reply.send(store.execute_claim(&id, &command));
                        }
                        ActorRequest::DispatchBudget(id, command, reserve, capacity, reply) => {
                            let _ = reply
                                .send(store.dispatch_budgeted(&id, &command, reserve, capacity));
                        }
                        ActorRequest::SettleBudget(id, command, used, reply) => {
                            let _ = reply.send(store.settle_budgeted(&id, &command, used));
                        }
                        ActorRequest::ClaimScheduled(id, command, reply) => {
                            let _ = reply.send(store.claim_scheduled_start(&id, &command));
                        }
                        ActorRequest::ClaimBudgetScheduled(
                            outbox,
                            run,
                            command,
                            reserve,
                            capacity,
                            reply,
                        ) => {
                            let _ = reply.send(store.claim_budgeted_scheduled_start(
                                &outbox, &run, &command, reserve, capacity,
                            ));
                        }
                        ActorRequest::InstallPlan(id, plan, reply) => {
                            let _ = reply.send(store.install_execution_plan(&id, &plan));
                        }
                        ActorRequest::GetPlan(id, reply) => {
                            let _ = reply.send(store.execution_plan(&id));
                        }
                        ActorRequest::RootId(id, reply) => {
                            let _ = reply.send(store.root_id(&id));
                        }
                        ActorRequest::Shutdown(reply) => {
                            let _ = reply.send(());
                            break;
                        }
                        ActorRequest::Create(id, state, reply) => {
                            let _ = reply.send(store.create_idempotent(&id, &state));
                        }
                        ActorRequest::Execute(id, command, reply) => {
                            let _ = reply.send(store.execute(&id, &command));
                        }
                        ActorRequest::Get(id, reply) => {
                            let _ = reply.send(store.get(&id));
                        }
                        ActorRequest::List(reply) => {
                            let _ = reply.send(store.list());
                        }
                        ActorRequest::ListActive(reply) => {
                            let _ = reply.send(store.list_active());
                        }
                        ActorRequest::Events(id, cursor, limit, reply) => {
                            let _ = reply.send(store.events_after(&id, cursor, limit));
                        }
                        ActorRequest::Spawn(parent, revision, command, child, reply) => {
                            let _ =
                                reply.send(store.spawn_child(&parent, revision, &command, &child));
                        }
                        ActorRequest::SpawnWithPlan(request, reply) => {
                            let _ = reply.send(store.spawn_child_with_plan(&request));
                        }
                        ActorRequest::Message(parent, child, id, body, reply) => {
                            let _ = reply.send(store.message_child(&parent, &child, &id, &body));
                        }
                        ActorRequest::Collect(parent, child, reply) => {
                            let _ = reply.send(store.collect_child(&parent, &child));
                        }
                        ActorRequest::Reserve(run, command, reply) => {
                            let _ = reply.send(store.reserve_effect(&run, &command));
                        }
                        ActorRequest::Pump(now, limit, reply) => {
                            let _ = reply.send(store.pump_due(now, limit));
                        }
                        ActorRequest::Pending(limit, reply) => {
                            let _ = reply.send(store.pending_outbox(limit));
                        }
                        ActorRequest::Ack(id, reply) => {
                            let _ = reply.send(store.acknowledge_outbox(&id));
                        }
                        ActorRequest::CreateSchedule(
                            id,
                            schedule,
                            policy,
                            definition,
                            from,
                            reply,
                        ) => {
                            let _ = reply.send(store.create_schedule(
                                &id,
                                &schedule,
                                policy,
                                &definition,
                                from,
                            ));
                        }
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        ready_recv.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            sender,
            thread: std::sync::Arc::new(std::sync::Mutex::new(Some(thread))),
        })
    }
    pub fn create(&self, command_id: String, state: RunState) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::Create(command_id, state, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn execute(
        &self,
        run_id: String,
        command: CommandEnvelope,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::Execute(run_id, command, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn get(&self, run_id: String) -> Result<Option<RunState>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::Get(run_id, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn events_after(
        &self,
        run_id: String,
        cursor: u64,
        limit: u32,
    ) -> Result<Vec<Event>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::Events(run_id, cursor, limit, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn spawn_child(
        &self,
        parent: String,
        revision: u64,
        command: String,
        child: RunState,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::Spawn(parent, revision, command, child, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn spawn_child_with_plan(
        &self,
        request: crate::execution::ChildSpawnRequest,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::SpawnWithPlan(request, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn message_child(
        &self,
        parent: String,
        child: String,
        id: String,
        body: String,
    ) -> Result<(), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::Message(parent, child, id, body, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn collect_child(&self, parent: String, child: String) -> Result<RunState, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::Collect(parent, child, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn reserve_effect(
        &self,
        run: String,
        command: CommandEnvelope,
    ) -> Result<(CommandReply, bool), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::Reserve(run, command, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn pump_due(
        &self,
        now: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<crate::schedule::PumpResult>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::Pump(now, limit, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn pending_outbox(&self, limit: u32) -> Result<Vec<crate::schedule::OutboxEntry>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::Pending(limit, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn acknowledge_outbox(&self, id: String) -> Result<(), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::Ack(id, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn create_schedule(
        &self,
        id: String,
        schedule: crate::schedule::Schedule,
        policy: crate::schedule::Policy,
        definition: WorkflowDefinition,
        from: chrono::DateTime<chrono::Utc>,
    ) -> Result<(), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::CreateSchedule(
                id, schedule, policy, definition, from, send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn list(&self) -> Result<Vec<RunState>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::List(send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    /// Runs that are not terminal: what the host's worker polls.
    pub fn list_active(&self) -> Result<Vec<RunState>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::ListActive(send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    /// Drain requests ahead of shutdown, stop the store owner, and join its thread.
    pub fn shutdown(&self) -> Result<(), String> {
        let mut thread = self.thread.lock().map_err(|e| e.to_string())?;
        if let Some(handle) = thread.take() {
            let (send, recv) = std::sync::mpsc::channel();
            let sent = self.sender.send(ActorRequest::Shutdown(send)).is_ok();
            let acknowledged = sent && recv.recv().is_ok();
            handle
                .join()
                .map_err(|_| "store actor panicked".to_string())?;
            if !acknowledged {
                return Err("store actor stopped before shutdown".into());
            }
        }
        Ok(())
    }

    pub fn install_execution_plan(
        &self,
        id: String,
        plan: crate::execution::ExecutionPlan,
    ) -> Result<(), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::InstallPlan(id, plan, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn execution_plan(
        &self,
        id: String,
    ) -> Result<Option<crate::execution::ExecutionPlan>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::GetPlan(id, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn root_id(&self, id: String) -> Result<String, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::RootId(id, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn dispatch_budgeted(
        &self,
        id: String,
        command: CommandEnvelope,
        reserve: u64,
        capacity: usize,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::DispatchBudget(
                id, command, reserve, capacity, send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn settle_budgeted(
        &self,
        id: String,
        command: CommandEnvelope,
        used: u64,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::SettleBudget(id, command, used, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn claim_scheduled_start(
        &self,
        id: String,
        command: CommandEnvelope,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::ClaimScheduled(id, command, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn claim_budgeted_scheduled_start(
        &self,
        outbox: String,
        run: String,
        command: CommandEnvelope,
        reserve: u64,
        capacity: usize,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::ClaimBudgetScheduled(
                outbox, run, command, reserve, capacity, send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn dispatch_once(
        &self,
        id: String,
        command: CommandEnvelope,
    ) -> Result<(CommandReply, bool), String> {
        if !matches!(&command.command, Command::StartAttempt { .. }) {
            return Err("expected start attempt".into());
        }
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::DispatchOnce(id, command, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn install_initial_plan(
        &self,
        id: String,
        command: CommandEnvelope,
        plan: crate::execution::ExecutionPlan,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::InstallInitialPlan(id, command, plan, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn scheduled_start_for_run(&self, id: String) -> Result<Option<(String, bool)>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::ScheduledForRun(id, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn create_schedule_with_plan(
        &self,
        id: String,
        schedule: crate::schedule::Schedule,
        policy: crate::schedule::Policy,
        definition: WorkflowDefinition,
        plan: crate::execution::ExecutionPlan,
        from: chrono::DateTime<chrono::Utc>,
    ) -> Result<(), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::CreateScheduleWithPlan(
                id, schedule, policy, definition, plan, from, send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    /// Replaces a schedule's rule, policy, goal and steps; see `update_schedule_with_plan`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_schedule_with_plan(
        &self,
        id: String,
        expected_revision: u64,
        schedule: crate::schedule::Schedule,
        policy: crate::schedule::Policy,
        definition: WorkflowDefinition,
        plan: crate::execution::ExecutionPlan,
        from: chrono::DateTime<chrono::Utc>,
    ) -> Result<u64, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::UpdateScheduleWithPlan(
                id,
                expected_revision,
                schedule,
                policy,
                definition,
                plan,
                from,
                send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    /// Saves one attempt's wall time and tokens.
    pub fn record_step(
        &self,
        run_id: String,
        stat: crate::execution::StepStat,
    ) -> Result<(), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::RecordStep(run_id, stat, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    /// Every recorded attempt of a run, oldest first.
    pub fn step_stats(&self, run_id: String) -> Result<Vec<crate::execution::StepStat>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::StepStats(run_id, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn claim_scheduled_start_once(
        &self,
        id: String,
        command: CommandEnvelope,
    ) -> Result<(CommandReply, bool), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::ScheduledOnce(id, command, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn dispatch_budgeted_once(
        &self,
        id: String,
        command: CommandEnvelope,
        reserve: u64,
        capacity: usize,
    ) -> Result<(CommandReply, bool), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::DispatchBudgetOnce(
                id, command, reserve, capacity, send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn claim_budgeted_scheduled_start_once(
        &self,
        outbox: String,
        id: String,
        command: CommandEnvelope,
        reserve: u64,
        capacity: usize,
    ) -> Result<(CommandReply, bool), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::BudgetScheduledOnce(
                outbox, id, command, reserve, capacity, send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn reconcile_budgeted_usage(
        &self,
        run: String,
        attempt: String,
        used: u64,
        receipt: String,
    ) -> Result<(), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::ReconcileBudget(
                run, attempt, used, receipt, send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn effective_grants(&self, id: String) -> Result<(u64, Vec<String>), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::EffectiveGrants(id, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn reserve_granted_effect(
        &self,
        id: String,
        command: CommandEnvelope,
        grant: String,
        generation: u64,
    ) -> Result<(CommandReply, bool), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::ReserveGranted(
                id, command, grant, generation, send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn update_watch_notify(
        &self,
        id: String,
        version: u64,
        truth: crate::watch::Truth,
        cursor: Option<String>,
        notification: String,
        payload: String,
    ) -> Result<crate::watch::WatchState, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::UpdateWatch(
                id,
                version,
                truth,
                cursor,
                notification,
                payload,
                send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    /// Undelivered notifications for the transport, uncertain ones included.
    pub fn pending_notifications(
        &self,
        limit: u32,
    ) -> Result<Vec<crate::watch::PendingNotification>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::PendingNotifications(limit, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn notification_status(
        &self,
        id: String,
    ) -> Result<crate::watch::NotificationStatus, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::NotificationStatus(id, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    /// Claim a notification for `transport` before showing it; see `SqliteStore::claim_notification`.
    pub fn claim_notification(
        &self,
        id: String,
        transport: String,
        now: chrono::DateTime<chrono::Utc>,
    ) -> Result<crate::watch::NotificationClaim, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::ClaimNotification(id, transport, now, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    /// Resolve the claim `transport` holds: a receipt commits delivery, none returns it to pending.
    pub fn resolve_notification(
        &self,
        id: String,
        transport: String,
        receipt: Option<String>,
    ) -> Result<(), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::ResolveNotification(
                id, transport, receipt, send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn list_schedules(&self) -> Result<Vec<crate::schedule::ScheduleRecord>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::ListSchedules(send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn enable_schedule(&self, id: String, revision: u64, enabled: bool) -> Result<u64, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::EnableSchedule(id, revision, enabled, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    /// Deletes a schedule whose runs are all settled (`SqliteStore::delete_schedule`).
    pub fn delete_schedule(&self, id: String, revision: u64) -> Result<(), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::DeleteSchedule(id, revision, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    /// Deletes a settled run nothing refers to (`SqliteStore::delete_run`).
    pub fn delete_run(&self, id: String) -> Result<(), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::DeleteRun(id, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn create_paused(
        &self,
        command_id: String,
        state: RunState,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::CreatePaused(command_id, state, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn start_plan(
        &self,
        id: String,
        command_id: String,
        revision: u64,
        plan: crate::execution::ExecutionPlan,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::StartPlan(
                id, command_id, revision, plan, send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn spawn_child_and_wait(
        &self,
        request: crate::execution::ChildSpawnRequest,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::SpawnAndWait(request, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn collect_waited_child(&self, parent: String, child: String) -> Result<RunState, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::CollectWaited(parent, child, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
    pub fn check_child_wait(&self, parent: String, child: String) -> Result<(), String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::CheckChildWait(parent, child, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn poll_waited_child(
        &self,
        parent: String,
        child: String,
    ) -> Result<Option<RunState>, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::PollWaited(parent, child, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn spawn_child_and_wait_budgeted(
        &self,
        request: crate::execution::BudgetedChildSpawnRequest,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::SpawnAndWaitBudgeted(request, send))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }

    pub fn fail_budgeted_with_receipt(
        &self,
        run: String,
        command: CommandEnvelope,
        used: u64,
        receipt: String,
    ) -> Result<CommandReply, String> {
        let (send, recv) = std::sync::mpsc::channel();
        self.sender
            .send(ActorRequest::FailBudgetReceipt(
                run, command, used, receipt, send,
            ))
            .map_err(|e| e.to_string())?;
        recv.recv().map_err(|e| e.to_string())?
    }
}
