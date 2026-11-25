use crate::{reducer, types::*};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use std::path::{Path, PathBuf};

/// Canonical database path shared by the service and local clients.
pub fn database_path(data_dir: impl AsRef<Path>) -> PathBuf {
    data_dir
        .as_ref()
        .join("runtime-service")
        .join("tasks.sqlite")
}

type Result<T> = std::result::Result<T, String>;
/// The newest schema this build reads; a newer database fails clearly instead of emptying.
pub(crate) const STORE_VERSION: u32 = 9;
fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
/// Switching to WAL needs an exclusive lock and SQLite returns BUSY without consulting the busy
/// handler, so concurrent first opens (service and CLI) retry within the same five seconds.
fn enable_wal(connection: &Connection) -> Result<()> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match connection.query_row("PRAGMA journal_mode=WAL", [], |r| r.get::<_, String>(0)) {
            // In-memory stores report "memory"; the mode was never asserted before.
            Ok(_) => return Ok(()),
            Err(rusqlite::Error::SqliteFailure(e, _))
                if e.code == rusqlite::ErrorCode::DatabaseBusy
                    && std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(std::time::Duration::from_millis(10))
            }
            Err(e) => return Err(err(e)),
        }
    }
}
fn sqlite_int(value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| "integer exceeds SQLite range".into())
}

fn migrate_v6(connection: &mut Connection) -> Result<()> {
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(err)?;
    let version: u32 = tx
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(err)?;
    if version > crate::store::STORE_VERSION {
        return Err(format!("unsupported store version {version}"));
    }
    if version < 5 {
        return Err("previous migration missing".into());
    }
    if version == 5 {
        tx.execute_batch("CREATE TABLE child_origins(parent_id TEXT NOT NULL, child_id TEXT NOT NULL PRIMARY KEY, attempt_id TEXT NOT NULL, epoch INTEGER NOT NULL, graph_revision INTEGER NOT NULL, node_id TEXT NOT NULL, FOREIGN KEY(parent_id,child_id) REFERENCES children(parent_id,child_id)); PRAGMA user_version=6;")
            .map_err(err)?;
    }
    tx.commit().map_err(err)
}

/// Spans of occurrences a schedule passed over, with the reason; the last pump time, so a clock
/// that runs back is noticed; a finished flag that keeps the unfinished-occurrence scan short;
/// and the transport that holds each notification claim, until when.
fn migrate_v8(connection: &mut Connection) -> Result<()> {
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(err)?;
    let version: u32 = tx
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(err)?;
    if version < 7 {
        return Err("previous migration missing".into());
    }
    if version == 7 {
        tx.execute_batch(
            "CREATE TABLE missed_occurrences(schedule_id TEXT NOT NULL REFERENCES schedules(id), first_at TEXT NOT NULL, last_at TEXT NOT NULL, count INTEGER NOT NULL, reason TEXT NOT NULL, PRIMARY KEY(schedule_id,first_at));
            ALTER TABLE schedules ADD COLUMN pumped_at TEXT;
            ALTER TABLE occurrences ADD COLUMN finished INTEGER NOT NULL DEFAULT 0;
            CREATE INDEX occurrences_unfinished ON occurrences(schedule_id,scheduled_at) WHERE finished=0;
            CREATE TABLE notification_claims(outbox_id TEXT PRIMARY KEY REFERENCES task_outbox(id), transport TEXT NOT NULL, expires_at TEXT NOT NULL);
            PRAGMA user_version=8;",
        )
        .map_err(err)?;
    }
    tx.commit().map_err(err)
}
/// Wall time and tokens of each attempt, for the step's line in the app.
fn migrate_v9(connection: &mut Connection) -> Result<()> {
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(err)?;
    let version: u32 = tx
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(err)?;
    if version < 8 {
        return Err("previous migration missing".into());
    }
    if version == 8 {
        tx.execute_batch(
            "CREATE TABLE step_stats(run_id TEXT NOT NULL REFERENCES runs(id), attempt_id TEXT NOT NULL, node_id TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT NOT NULL, prompt_tokens INTEGER, completion_tokens INTEGER, PRIMARY KEY(run_id,attempt_id));
            PRAGMA user_version=9;",
        )
        .map_err(err)?;
    }
    tx.commit().map_err(err)
}
fn migrate_v7(connection: &mut Connection) -> Result<()> {
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(err)?;
    let version: u32 = tx
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(err)?;
    if version > crate::store::STORE_VERSION {
        return Err(format!("unsupported store version {version}"));
    }
    if version < 6 {
        return Err("previous migration missing".into());
    }
    if version == 6 {
        tx.execute_batch("PRAGMA user_version=7;").map_err(err)?;
    }
    tx.commit().map_err(err)
}

fn has_attempt_permit(db: &Connection, run_id: &str) -> Result<bool> {
    let prefix = format!("dispatch:{}:{}:", run_id.len(), run_id);
    db.query_row(
        "SELECT EXISTS(SELECT 1 FROM admission_permits WHERE substr(id,1,length(?1))=?1)",
        [&prefix],
        |row| row.get(0),
    )
    .map_err(err)
}
fn child_failure(child: &RunState) -> Option<ChildFailure> {
    let (kind, reason) = match child.status {
        RunStatus::Waiting if child.attempt.is_none() && child.waiting_child.is_none() => (
            ChildFailureKind::FailedAttempt,
            Some(child.attempt_failure.clone()?),
        ),
        RunStatus::Failed => (ChildFailureKind::Failed, child.result.clone()),
        RunStatus::Cancelled => (ChildFailureKind::Cancelled, child.result.clone()),
        _ => return None,
    };
    Some(ChildFailure {
        child_id: child.id.clone(),
        kind,
        reason,
    })
}

/// Verify a persisted wait against the child relationship's immutable origin.
fn verify_wait_origin(
    db: &Connection,
    parent_id: &str,
    child_id: &str,
    parent: &RunState,
) -> Result<()> {
    let wait = parent
        .waiting_child
        .as_ref()
        .ok_or("parent is not waiting for a child")?;
    if parent.status != RunStatus::Waiting
        || wait.child_id != child_id
        || wait.graph_revision != parent.graph_revision
    {
        return Err("parent child wait is stale or inactive".into());
    }
    let origin: Option<(String, i64, i64, String)> = db
        .query_row(
            "SELECT attempt_id,epoch,graph_revision,node_id FROM child_origins WHERE parent_id=?1 AND child_id=?2",
            params![parent_id, child_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .optional()
        .map_err(err)?;
    let (attempt_id, epoch, graph_revision, node_id) =
        origin.ok_or("child has no trusted origin")?;
    if wait.origin_attempt_id.as_deref() != Some(attempt_id.as_str())
        || wait.origin_epoch != Some(u64::try_from(epoch).map_err(err)?)
        || wait.graph_revision != u64::try_from(graph_revision).map_err(err)?
        || wait.node_id != node_id
    {
        return Err("child origin does not match parent wait".into());
    }
    Ok(())
}
/// Reject a child start when its root is inactive or awaiting unrelated recovery.
pub(crate) fn ensure_root_dispatchable(tx: &rusqlite::Transaction<'_>, run_id: &str) -> Result<()> {
    let row: Option<(String, String)> = tx
        .query_row(
            "SELECT root.id,root.state FROM children relation JOIN runs root ON root.id=relation.parent_id WHERE relation.child_id=?1",
            [run_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(err)?;
    if let Some((root_id, json)) = row {
        let root = decode_state(&json)?;
        let origin: Option<(String, i64, i64, String)> = tx
            .query_row(
                "SELECT attempt_id,epoch,graph_revision,node_id FROM child_origins WHERE parent_id=?1 AND child_id=?2",
                params![root_id, run_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()
            .map_err(err)?;
        let (attempt_id, epoch, graph, node) = origin.ok_or("child has no trusted origin")?;
        match root.status {
            RunStatus::Running => {
                let attempt = root.attempt.as_ref().ok_or("root attempt missing")?;
                if attempt.id != attempt_id
                    || attempt.epoch != u64::try_from(epoch).map_err(err)?
                    || root.graph_revision != u64::try_from(graph).map_err(err)?
                    || attempt.node_id != node
                {
                    return Err("child origin is stale".into());
                }
            }
            RunStatus::Waiting => {
                if root.child_failure.is_some() {
                    return Err("root is awaiting a child failure decision".into());
                }
                if root.effects.values().any(|effect| {
                    matches!(
                        effect.status,
                        EffectStatus::Intent | EffectStatus::Uncertain
                    )
                }) {
                    return Err("root has unresolved effects".into());
                }
                verify_wait_origin(tx, &root_id, run_id, &root)?;
            }
            _ => return Err("child root is not dispatchable".into()),
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum SpawnMode<'a> {
    /// A child admitted without a parent-attempt fence or plan.
    Unfenced,
    Plan(&'a crate::execution::ChildSpawnRequest),
    Wait(&'a crate::execution::ChildSpawnRequest),
    Budgeted(&'a crate::execution::BudgetedChildSpawnRequest),
}

/// SQLite serializes writers; command reply, event and state share one commit.
pub struct SqliteStore {
    pub(crate) connection: Connection,
}
impl SqliteStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut connection = Connection::open(path).map_err(err)?;
        connection
            .busy_timeout(std::time::Duration::from_secs(5))
            .map_err(err)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;")
            .map_err(err)?;
        enable_wal(&connection)?;
        let version: u32 = connection
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .map_err(err)?;
        if version > crate::store::STORE_VERSION {
            return Err(format!("unsupported store version {version}"));
        }
        if version == 0 {
            let tx = connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(err)?;
            let current: u32 = tx
                .pragma_query_value(None, "user_version", |r| r.get(0))
                .map_err(err)?;
            if current == 0 {
                tx.execute_batch("CREATE TABLE runs (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, state TEXT NOT NULL);
                    CREATE TABLE events (run_id TEXT NOT NULL REFERENCES runs(id), revision INTEGER NOT NULL, event TEXT NOT NULL, PRIMARY KEY(run_id, revision));
                    CREATE TABLE commands (run_id TEXT NOT NULL REFERENCES runs(id), id TEXT NOT NULL, request TEXT NOT NULL, reply TEXT NOT NULL, PRIMARY KEY(run_id, id));
                    PRAGMA user_version=1;")
                    .map_err(err)?;
            }
            tx.commit().map_err(err)?;
        }
        if version < 2 {
            crate::schedule::migrate(&connection)?;
        }
        crate::execution::migrate(&connection)?;
        crate::schedule::migrate_v4(&connection)?;
        crate::admission::migrate_v5(&connection)?;
        migrate_v6(&mut connection)?;
        migrate_v7(&mut connection)?;
        migrate_v8(&mut connection)?;
        migrate_v9(&mut connection)?;
        Ok(Self { connection })
    }
    pub fn create(&mut self, state: &RunState) -> Result<()> {
        state.definition.validate()?;
        if !state.is_pristine() {
            return Err("invalid initial state".into());
        }
        let json = serde_json::to_string(state).map_err(err)?;
        self.connection
            .execute(
                "INSERT INTO runs(id,revision,state) VALUES (?1,0,?2)",
                params![state.id, json],
            )
            .map_err(err)?;
        Ok(())
    }
    /// Atomically creates a run and records its idempotency receipt.
    pub fn create_idempotent(
        &mut self,
        command_id: &str,
        state: &RunState,
    ) -> Result<CommandReply> {
        self.create_with_status(command_id, state, RunStatus::Ready)
    }
    pub fn create_paused_idempotent(
        &mut self,
        command_id: &str,
        state: &RunState,
    ) -> Result<CommandReply> {
        self.create_with_status(command_id, state, RunStatus::Paused)
    }
    fn create_with_status(
        &mut self,
        command_id: &str,
        state: &RunState,
        status: RunStatus,
    ) -> Result<CommandReply> {
        if command_id.is_empty() {
            return Err("command id is empty".into());
        }
        state.definition.validate()?;
        if !state.is_pristine() {
            return Err("invalid initial state".into());
        }
        let mut initial = state.clone();
        initial.status = status.clone();
        let request = serde_json::to_string(&initial).map_err(err)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(err)?;
        let previous: Option<(String, String)> = tx
            .query_row(
                "SELECT request,reply FROM commands WHERE run_id=?1 AND id=?2",
                params![state.id, command_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(err)?;
        if let Some((old, reply)) = previous {
            if old != request {
                return Err("command id reused with different request".into());
            }
            return serde_json::from_str(&reply).map_err(err);
        }
        let reply = CommandReply {
            revision: 0,
            status,
        };
        tx.execute(
            "INSERT INTO runs(id,revision,state) VALUES (?1,0,?2)",
            params![state.id, request],
        )
        .map_err(err)?;
        tx.execute(
            "INSERT INTO commands(run_id,id,request,reply) VALUES (?1,?2,?3,?4)",
            params![
                state.id,
                command_id,
                request,
                serde_json::to_string(&reply).map_err(err)?
            ],
        )
        .map_err(err)?;
        tx.commit().map_err(err)?;
        Ok(reply)
    }
    /// Admit a depth-one child and its relationship in one transaction.
    /// Budgeted delegation is refused until shared budget accounting is connected.
    pub fn spawn_child(
        &mut self,
        parent_id: &str,
        expected_revision: u64,
        command_id: &str,
        child: &RunState,
    ) -> Result<CommandReply> {
        self.spawn_child_inner(
            parent_id,
            expected_revision,
            command_id,
            child,
            SpawnMode::Unfenced,
        )
    }
    /// Atomically admit a child and its immutable plan for a live parent attempt.
    pub fn spawn_child_with_plan(
        &mut self,
        request: &crate::execution::ChildSpawnRequest,
    ) -> Result<CommandReply> {
        self.spawn_child_inner(
            &request.parent_id,
            request.expected_revision,
            &request.command_id,
            &request.child,
            SpawnMode::Plan(request),
        )
    }
    /// Admit a child, its plan and the parent wait in a single transaction.
    pub fn spawn_child_and_wait(
        &mut self,
        request: &crate::execution::ChildSpawnRequest,
    ) -> Result<CommandReply> {
        self.spawn_child_inner(
            &request.parent_id,
            request.expected_revision,
            &request.command_id,
            &request.child,
            SpawnMode::Wait(request),
        )
    }
    /// The caller must verify the usage receipt before invoking this trusted API.
    pub fn spawn_child_and_wait_budgeted(
        &mut self,
        request: &crate::execution::BudgetedChildSpawnRequest,
    ) -> Result<CommandReply> {
        let spawn = &request.spawn;
        self.spawn_child_inner(
            &spawn.parent_id,
            spawn.expected_revision,
            &spawn.command_id,
            &spawn.child,
            SpawnMode::Budgeted(request),
        )
    }
    fn spawn_child_inner(
        &mut self,
        parent_id: &str,
        expected_revision: u64,
        command_id: &str,
        child: &RunState,
        mode: SpawnMode<'_>,
    ) -> Result<CommandReply> {
        let fenced = match mode {
            SpawnMode::Unfenced => None,
            SpawnMode::Plan(request) | SpawnMode::Wait(request) => Some(request),
            SpawnMode::Budgeted(request) => Some(&request.spawn),
        };
        let suspend = matches!(mode, SpawnMode::Wait(_) | SpawnMode::Budgeted(_));
        if command_id.is_empty() {
            return Err("command id is empty".into());
        }
        child.definition.validate()?;
        if !child.is_pristine() || child.id == parent_id {
            return Err("invalid child state".into());
        }
        if let Some(request) = fenced {
            crate::execution::validate_plan_for_definition(&child.definition, &request.plan)?;
        }
        let request = match mode {
            SpawnMode::Unfenced => {
                serde_json::to_string(&("spawn_child", expected_revision, child)).map_err(err)?
            }
            SpawnMode::Plan(request) => {
                serde_json::to_string(&("spawn_child_with_plan", request)).map_err(err)?
            }
            SpawnMode::Wait(request) => {
                serde_json::to_string(&("spawn_child_and_wait", request)).map_err(err)?
            }
            SpawnMode::Budgeted(request) => {
                serde_json::to_string(&("spawn_child_and_wait_budgeted", request)).map_err(err)?
            }
        };
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(err)?;
        let previous: Option<(String, String)> = tx
            .query_row(
                "SELECT request,reply FROM commands WHERE run_id=?1 AND id=?2",
                params![parent_id, command_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(err)?;
        if let Some((old, reply)) = previous {
            if old != request {
                return Err("command id reused with different request".into());
            }
            return serde_json::from_str(&reply).map_err(err);
        }
        let json: String = tx
            .query_row("SELECT state FROM runs WHERE id=?1", [parent_id], |r| {
                r.get(0)
            })
            .optional()
            .map_err(err)?
            .ok_or("unknown parent")?;
        let mut parent = decode_state(&json)?;
        if parent.revision != expected_revision
            || !matches!(parent.status, RunStatus::Ready | RunStatus::Running)
        {
            return Err("stale or inactive parent".into());
        }
        if let Some(fence) = fenced {
            if parent.status != RunStatus::Running
                || parent.graph_revision != fence.expected_graph_revision
                || parent.attempt.as_ref().is_none_or(|attempt| {
                    attempt.id != fence.parent_attempt_id || attempt.epoch != fence.parent_epoch
                })
            {
                return Err("stale parent attempt or graph".into());
            }
            if !parent.definition.budgets.is_empty() && !matches!(mode, SpawnMode::Budgeted(_)) {
                return Err("budgeted child suspension requires atomic settlement".into());
            }
            if parent.definition.budgets.is_empty() && matches!(mode, SpawnMode::Budgeted(_)) {
                return Err("parent has no budget".into());
            }
        }
        let nested: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM children WHERE child_id=?1)",
                [parent_id],
                |r| r.get(0),
            )
            .map_err(err)?;
        let count: u32 = tx
            .query_row(
                "SELECT COUNT(*) FROM children WHERE parent_id=?1",
                [parent_id],
                |r| r.get(0),
            )
            .map_err(err)?;
        if nested || count >= 4 {
            return Err("child depth or count limit exceeded".into());
        }
        let existing_ledger: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM admission_roots WHERE id=?1) OR EXISTS(SELECT 1 FROM admission_permits WHERE root_id=?1)",
                [&child.id],
                |r| r.get(0),
            )
            .map_err(err)?;
        if existing_ledger {
            return Err("child identity has an existing admission ledger".into());
        }
        if parent.definition.budgets != child.definition.budgets
            || parent
                .definition
                .budgets
                .keys()
                .any(|key| key != "model_tokens")
        {
            return Err("child budget must match supported root budget".into());
        }
        if child
            .definition
            .grants
            .iter()
            .any(|grant| !parent.definition.grants.contains(grant))
        {
            return Err("child grants exceed parent".into());
        }
        let origin_node = parent
            .attempt
            .as_ref()
            .map(|attempt| attempt.node_id.clone());
        let (event, reply) = if suspend {
            let fence = fenced.ok_or("missing parent fence")?;
            let command = CommandEnvelope {
                id: command_id.into(),
                expected_revision,
                command: Command::WaitForChild {
                    attempt_id: fence.parent_attempt_id.clone(),
                    epoch: fence.parent_epoch,
                    child_id: child.id.clone(),
                },
            };
            let (next, event, reply) = if let SpawnMode::Budgeted(budget) = mode {
                crate::admission::settle_for_child_wait(
                    &tx,
                    parent_id,
                    &fence.parent_attempt_id,
                    budget.used,
                    &budget.usage_receipt,
                )?;
                reducer::apply_budgeted(&parent, &command)?
            } else {
                reducer::apply(&parent, &command)?
            };
            parent = next;
            (event, reply)
        } else {
            parent.revision = parent.revision.checked_add(1).ok_or("revision overflow")?;
            let event = Event {
                revision: parent.revision,
                command_id: command_id.into(),
                kind: "child_spawned".into(),
            };
            let reply = CommandReply {
                revision: parent.revision,
                status: parent.status.clone(),
            };
            (event, reply)
        };
        tx.execute(
            "INSERT INTO runs(id,revision,state) VALUES (?1,0,?2)",
            params![child.id, serde_json::to_string(child).map_err(err)?],
        )
        .map_err(err)?;
        tx.execute(
            "INSERT INTO children(parent_id,child_id) VALUES (?1,?2)",
            params![parent_id, child.id],
        )
        .map_err(err)?;
        if let Some(fence) = fenced {
            tx.execute(
                "INSERT INTO execution_plans(run_id,plan) VALUES (?1,?2)",
                params![child.id, serde_json::to_string(&fence.plan).map_err(err)?],
            )
            .map_err(err)?;
            tx.execute(
                "INSERT INTO child_origins(parent_id,child_id,attempt_id,epoch,graph_revision,node_id) VALUES (?1,?2,?3,?4,?5,?6)",
                params![
                    parent_id,
                    child.id,
                    fence.parent_attempt_id,
                    sqlite_int(fence.parent_epoch)?,
                    sqlite_int(fence.expected_graph_revision)?,
                    origin_node.ok_or("missing parent node")?
                ],
            )
            .map_err(err)?;
        }
        tx.execute(
            "UPDATE runs SET revision=?1,state=?2 WHERE id=?3",
            params![
                sqlite_int(parent.revision)?,
                serde_json::to_string(&parent).map_err(err)?,
                parent_id
            ],
        )
        .map_err(err)?;
        tx.execute(
            "INSERT INTO events(run_id,revision,event) VALUES (?1,?2,?3)",
            params![
                parent_id,
                sqlite_int(parent.revision)?,
                serde_json::to_string(&event).map_err(err)?
            ],
        )
        .map_err(err)?;
        tx.execute(
            "INSERT INTO commands(run_id,id,request,reply) VALUES (?1,?2,?3,?4)",
            params![
                parent_id,
                command_id,
                request,
                serde_json::to_string(&reply).map_err(err)?
            ],
        )
        .map_err(err)?;
        tx.commit().map_err(err)?;
        Ok(reply)
    }
    /// Durable, idempotent parent-to-child message. Reused identities must match.
    pub fn message_child(
        &mut self,
        parent_id: &str,
        child_id: &str,
        message_id: &str,
        body: &str,
    ) -> Result<()> {
        if message_id.is_empty() || body.len() > 65536 {
            return Err("invalid message".into());
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(err)?;
        let old: Option<String> = tx
            .query_row(
                "SELECT body FROM child_messages WHERE parent_id=?1 AND child_id=?2 AND message_id=?3",
                params![parent_id, child_id, message_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(err)?;
        if let Some(old) = old {
            return if old == body {
                Ok(())
            } else {
                Err("message id reused".into())
            };
        }
        let related: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM children WHERE parent_id=?1 AND child_id=?2)",
                params![parent_id, child_id],
                |r| r.get(0),
            )
            .map_err(err)?;
        if !related {
            return Err("unknown child".into());
        }
        let parent_json: String = tx
            .query_row("SELECT state FROM runs WHERE id=?1", [parent_id], |r| {
                r.get(0)
            })
            .map_err(err)?;
        if decode_state(&parent_json)?.status.terminal() {
            return Err("terminal parent".into());
        }
        tx.execute(
            "INSERT INTO child_messages(parent_id,child_id,message_id,body) VALUES (?1,?2,?3,?4)",
            params![parent_id, child_id, message_id, body],
        )
        .map_err(err)?;
        tx.commit().map_err(err)?;
        Ok(())
    }
    pub fn collect_child(&self, parent_id: &str, child_id: &str) -> Result<RunState> {
        let exists: bool = self
            .connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM children WHERE parent_id=?1 AND child_id=?2)",
                params![parent_id, child_id],
                |r| r.get(0),
            )
            .map_err(err)?;
        if !exists {
            return Err("unknown child".into());
        }
        let state = self.get(child_id)?.ok_or("unknown child")?;
        if !state.status.terminal()
            || state.effects.values().any(|effect| {
                matches!(
                    effect.status,
                    EffectStatus::Intent | EffectStatus::Uncertain
                )
            })
            || has_attempt_permit(&self.connection, child_id)?
        {
            return Err("child outcome is not settled".into());
        }
        Ok(state)
    }
    pub fn check_child_wait(&self, parent_id: &str, child_id: &str) -> Result<()> {
        let tx = self.connection.unchecked_transaction().map_err(err)?;
        let json: String = tx
            .query_row("SELECT state FROM runs WHERE id=?1", [parent_id], |r| {
                r.get(0)
            })
            .optional()
            .map_err(err)?
            .ok_or("unknown parent")?;
        let parent = decode_state(&json)?;
        verify_wait_origin(&tx, parent_id, child_id, &parent)?;
        tx.commit().map_err(err)
    }
    /// Inspect a bound child, distinguishing unfinished work from invalid state.
    pub fn poll_waited_child(&self, parent_id: &str, child_id: &str) -> Result<Option<RunState>> {
        let tx = self.connection.unchecked_transaction().map_err(err)?;
        let json: String = tx
            .query_row("SELECT state FROM runs WHERE id=?1", [parent_id], |r| {
                r.get(0)
            })
            .optional()
            .map_err(err)?
            .ok_or("unknown parent")?;
        let parent = decode_state(&json)?;
        verify_wait_origin(&tx, parent_id, child_id, &parent)?;
        let child_json: String = tx
            .query_row("SELECT state FROM runs WHERE id=?1", [child_id], |r| {
                r.get(0)
            })
            .optional()
            .map_err(err)?
            .ok_or("unknown child")?;
        let child = decode_state(&child_json)?;
        if !child.status.terminal() && child_failure(&child).is_none() {
            tx.commit().map_err(err)?;
            return Ok(None);
        }
        if child.effects.values().any(|effect| {
            matches!(
                effect.status,
                EffectStatus::Intent | EffectStatus::Uncertain
            )
        }) || has_attempt_permit(&tx, child_id)?
        {
            return Err("child has unresolved effects or usage".into());
        }
        tx.commit().map_err(err)?;
        Ok(Some(child))
    }
    pub fn collect_waited_child(&self, parent_id: &str, child_id: &str) -> Result<RunState> {
        self.poll_waited_child(parent_id, child_id)?
            .ok_or("child outcome is not settled".into())
    }
    pub fn child_messages(&self, parent_id: &str, child_id: &str) -> Result<Vec<(String, String)>> {
        let mut stmt = self.connection
            .prepare("SELECT message_id,body FROM child_messages WHERE parent_id=?1 AND child_id=?2 ORDER BY rowid")
            .map_err(err)?;
        let rows = stmt
            .query_map(params![parent_id, child_id], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(err)?;
        rows.map(|r| r.map_err(err)).collect()
    }
    pub fn get(&self, run_id: &str) -> Result<Option<RunState>> {
        let json: Option<String> = self
            .connection
            .query_row("SELECT state FROM runs WHERE id=?1", [run_id], |r| r.get(0))
            .optional()
            .map_err(err)?;
        json.map(|s| decode_state(&s)).transpose()
    }
    pub fn effective_grants(&self, run_id: &str) -> Result<(u64, Vec<String>)> {
        let row: Option<(String, Option<String>)> = self.connection
            .query_row(
                "SELECT child.state, parent.state FROM runs child LEFT JOIN children relation ON relation.child_id=child.id LEFT JOIN runs parent ON parent.id=relation.parent_id WHERE child.id=?1",
                [run_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(err)?;
        let (child_json, root_json) = row.ok_or("unknown run")?;
        let child = decode_state(&child_json)?;
        let root = match root_json {
            Some(json) => decode_state(&json)?,
            None => child.clone(),
        };
        if root.status.terminal() || child.status.terminal() {
            return Err("run stopped".into());
        }
        let grants = child
            .definition
            .grants
            .iter()
            .filter(|grant| root.definition.grants.contains(grant))
            .cloned()
            .collect();
        Ok((root.grant_generation, grants))
    }
    pub fn list(&self) -> Result<Vec<RunState>> {
        self.runs("SELECT state FROM runs ORDER BY id")
    }
    /// Runs that are not terminal, without decoding the finished ones.
    // ponytail: scans every row's JSON in SQLite; add an indexed status column past ~10k runs
    pub fn list_active(&self) -> Result<Vec<RunState>> {
        self.runs("SELECT state FROM runs WHERE json_extract(state,'$.status') NOT IN ('succeeded','failed','cancelled') ORDER BY id")
    }
    fn runs(&self, sql: &str) -> Result<Vec<RunState>> {
        let mut stmt = self.connection.prepare(sql).map_err(err)?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0)).map_err(err)?;
        rows.map(|r| decode_state(&r.map_err(err)?)).collect()
    }
    pub fn events_after(&self, run_id: &str, cursor: u64, limit: u32) -> Result<Vec<Event>> {
        let mut stmt = self.connection
            .prepare("SELECT event FROM events WHERE run_id=?1 AND revision>?2 ORDER BY revision LIMIT ?3")
            .map_err(err)?;
        let rows = stmt
            .query_map(params![run_id, sqlite_int(cursor)?, limit.min(1000)], |r| {
                r.get::<_, String>(0)
            })
            .map_err(err)?;
        rows.map(|r| serde_json::from_str(&r.map_err(err)?).map_err(err))
            .collect()
    }
    /// Every event of the run, read in pages of the 1000-event cap.
    pub fn events(&self, run_id: &str) -> Result<Vec<Event>> {
        let mut events: Vec<Event> = Vec::new();
        loop {
            let cursor = events.last().map_or(0, |event| event.revision);
            let page = self.events_after(run_id, cursor, 1000)?;
            let last = page.len() < 1000;
            events.extend(page);
            if last {
                return Ok(events);
            }
        }
    }
    /// Saves one attempt's wall time and tokens; a repeat of the same attempt replaces it.
    pub fn record_step(&mut self, run_id: &str, stat: &crate::execution::StepStat) -> Result<()> {
        let (prompt, completion) = stat.tokens.unzip();
        self.connection
            .execute(
                "INSERT OR REPLACE INTO step_stats(run_id,attempt_id,node_id,started_at,finished_at,prompt_tokens,completion_tokens) VALUES(?1,?2,?3,?4,?5,?6,?7)",
                params![
                    run_id,
                    stat.attempt_id,
                    stat.node_id,
                    stat.started_at.to_rfc3339(),
                    stat.finished_at.to_rfc3339(),
                    prompt.map(sqlite_int).transpose()?,
                    completion.map(sqlite_int).transpose()?
                ],
            )
            .map_err(err)?;
        Ok(())
    }
    /// Every recorded attempt of a run, oldest first.
    pub fn step_stats(&self, run_id: &str) -> Result<Vec<crate::execution::StepStat>> {
        let time = |text: String| {
            chrono::DateTime::parse_from_rfc3339(&text)
                .map(|at| at.with_timezone(&chrono::Utc))
                .map_err(err)
        };
        let count = |n: Option<i64>| n.map(|n| u64::try_from(n).map_err(err)).transpose();
        let mut statement = self
            .connection
            .prepare("SELECT attempt_id,node_id,started_at,finished_at,prompt_tokens,completion_tokens FROM step_stats WHERE run_id=?1 ORDER BY started_at,attempt_id")
            .map_err(err)?;
        let rows = statement
            .query_map([run_id], |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                ))
            })
            .map_err(err)?;
        rows.map(|row| {
            let (attempt_id, node_id, started, finished, prompt, completion): (
                String,
                String,
                String,
                String,
                Option<i64>,
                Option<i64>,
            ) = row.map_err(err)?;
            Ok(crate::execution::StepStat {
                attempt_id,
                node_id,
                started_at: time(started)?,
                finished_at: time(finished)?,
                tokens: count(prompt)?.zip(count(completion)?),
            })
        })
        .collect()
    }
    pub fn execute(&mut self, run_id: &str, command: &CommandEnvelope) -> Result<CommandReply> {
        self.execute_claim_guarded(run_id, command, None)
            .map(|(reply, _)| reply)
    }
    /// Returns true only to the transaction that first committed this command.
    /// A false result must never be used as permission to repeat an effect.
    pub fn reserve_effect(
        &mut self,
        run_id: &str,
        command: &CommandEnvelope,
    ) -> Result<(CommandReply, bool)> {
        if !matches!(&command.command, Command::RecordIntent { .. }) {
            return Err("expected effect intent".into());
        }
        self.execute_claim_guarded(run_id, command, None)
    }
    pub(crate) fn execute_claim(
        &mut self,
        run_id: &str,
        command: &CommandEnvelope,
    ) -> Result<(CommandReply, bool)> {
        self.execute_claim_guarded(run_id, command, None)
    }
    pub fn reserve_granted_effect(
        &mut self,
        run_id: &str,
        command: &CommandEnvelope,
        grant: &str,
        generation: u64,
    ) -> Result<(CommandReply, bool)> {
        if !matches!(&command.command, Command::RecordIntent { .. }) || grant.is_empty() {
            return Err("expected effect intent and grant".into());
        }
        self.execute_claim_guarded(run_id, command, Some((grant, generation)))
    }
    fn execute_claim_guarded(
        &mut self,
        run_id: &str,
        command: &CommandEnvelope,
        guard: Option<(&str, u64)>,
    ) -> Result<(CommandReply, bool)> {
        let request = serde_json::to_string(command).map_err(err)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(err)?;
        let previous: Option<(String, String)> = tx
            .query_row(
                "SELECT request,reply FROM commands WHERE run_id=?1 AND id=?2",
                params![run_id, command.id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(err)?;
        if let Some((old_request, reply)) = previous {
            if old_request != request {
                return Err("command id reused with different request".into());
            }
            return serde_json::from_str(&reply)
                .map(|reply| (reply, false))
                .map_err(err);
        }
        let state_json: String = tx
            .query_row("SELECT state FROM runs WHERE id=?1", [run_id], |r| r.get(0))
            .optional()
            .map_err(err)?
            .ok_or("unknown run")?;
        let state = decode_state(&state_json)?;
        let root_id: Option<String> = tx
            .query_row(
                "SELECT parent_id FROM children WHERE child_id=?1",
                [run_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(err)?;
        if matches!(&command.command, Command::RevokeGrant { .. }) && root_id.is_some() {
            return Err("only a root can revoke grants".into());
        }
        if let Some((grant, generation)) = guard {
            let root = match root_id {
                Some(id) => {
                    let json: String = tx
                        .query_row("SELECT state FROM runs WHERE id=?1", [id], |r| r.get(0))
                        .map_err(err)?;
                    decode_state(&json)?
                }
                None => state.clone(),
            };
            if root.status.terminal()
                || state.status.terminal()
                || root.grant_generation != generation
                || !root.definition.grants.iter().any(|value| value == grant)
                || !state.definition.grants.iter().any(|value| value == grant)
            {
                return Err("grant revoked, stale, or run stopped".into());
            }
        }
        match &command.command {
            Command::WaitForChild { child_id, .. }
            | Command::WakeChild { child_id, .. }
            | Command::RecordChildFailure { child_id, .. }
            | Command::RetryChild { child_id } => {
                let child_json: Option<String> = tx
                    .query_row(
                        "SELECT child.state FROM children relation JOIN runs child ON child.id=relation.child_id WHERE relation.parent_id=?1 AND relation.child_id=?2",
                        params![run_id, child_id],
                        |r| r.get(0),
                    )
                    .optional()
                    .map_err(err)?;
                let child = decode_state(&child_json.ok_or("unknown child relationship")?)?;
                match &command.command {
                    Command::WaitForChild {
                        attempt_id, epoch, ..
                    } => {
                        let origin: Option<(String, i64, i64, String)> = tx
                            .query_row(
                                "SELECT attempt_id,epoch,graph_revision,node_id FROM child_origins WHERE parent_id=?1 AND child_id=?2",
                                params![run_id, child_id],
                                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                            )
                            .optional()
                            .map_err(err)?;
                        let (origin_id, origin_epoch, origin_graph, origin_node) =
                            origin.ok_or("child has no trusted origin")?;
                        let attempt = state.attempt.as_ref().ok_or("parent attempt missing")?;
                        if attempt_id != &origin_id
                            || *epoch != u64::try_from(origin_epoch).map_err(err)?
                            || state.graph_revision != u64::try_from(origin_graph).map_err(err)?
                            || attempt.id != origin_id
                            || attempt.epoch != *epoch
                            || attempt.node_id != origin_node
                        {
                            return Err("child origin is stale".into());
                        }
                    }
                    Command::WakeChild { result, .. } => {
                        verify_wait_origin(&tx, run_id, child_id, &state)?;
                        if child.status != RunStatus::Succeeded
                            || child.effects.values().any(|effect| {
                                matches!(
                                    effect.status,
                                    EffectStatus::Intent | EffectStatus::Uncertain
                                )
                            })
                            || has_attempt_permit(&tx, child_id)?
                            || child.result.as_ref() != Some(result)
                        {
                            return Err("child result is not settled".into());
                        }
                    }
                    Command::RecordChildFailure { .. } | Command::RetryChild { .. } => {
                        verify_wait_origin(&tx, run_id, child_id, &state)?;
                        if child.effects.values().any(|effect| {
                            matches!(
                                effect.status,
                                EffectStatus::Intent | EffectStatus::Uncertain
                            )
                        }) || state.effects.values().any(|effect| {
                            matches!(
                                effect.status,
                                EffectStatus::Intent | EffectStatus::Uncertain
                            )
                        }) || has_attempt_permit(&tx, child_id)?
                            || has_attempt_permit(&tx, run_id)?
                        {
                            return Err("child failure has unresolved effects or usage".into());
                        }
                        let actual = child_failure(&child).ok_or("child has not failed")?;
                        if let Command::RecordChildFailure { kind, reason, .. } = &command.command {
                            if &actual.kind != kind || &actual.reason != reason {
                                return Err("child failure does not match evidence".into());
                            }
                        } else {
                            let recorded = state
                                .child_failure
                                .as_ref()
                                .ok_or("child failure missing")?;
                            let cancelled_after_failure = recorded.child_id == child.id
                                && recorded.kind == ChildFailureKind::FailedAttempt
                                && child.status == RunStatus::Cancelled
                                && recorded.reason == child.attempt_failure;
                            if recorded != &actual && !cancelled_after_failure {
                                return Err("child failure changed".into());
                            }
                            if !child.status.terminal() {
                                let cancellation = CommandEnvelope {
                                    id: format!("parent-retry:{run_id}:{}", command.id),
                                    expected_revision: child.revision,
                                    command: Command::Cancel,
                                };
                                let (cancelled, event, _) = reducer::apply(&child, &cancellation)?;
                                tx.execute(
                                    "UPDATE runs SET revision=?1,state=?2 WHERE id=?3",
                                    params![
                                        sqlite_int(cancelled.revision)?,
                                        serde_json::to_string(&cancelled).map_err(err)?,
                                        child_id
                                    ],
                                )
                                .map_err(err)?;
                                tx.execute(
                                    "INSERT INTO events(run_id,revision,event) VALUES (?1,?2,?3)",
                                    params![
                                        child_id,
                                        sqlite_int(event.revision)?,
                                        serde_json::to_string(&event).map_err(err)?
                                    ],
                                )
                                .map_err(err)?;
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            }
            _ => {}
        }
        if matches!(
            &command.command,
            Command::Resume | Command::StartAttempt { .. }
        ) {
            let prefix = format!("dispatch:{}:{}:", run_id.len(), run_id);
            let unresolved: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM admission_permits WHERE substr(id,1,length(?1))=?1)",
                [&prefix],
                |r| r.get(0),
            ).map_err(err)?;
            if unresolved {
                return Err("unresolved usage reservation".into());
            }
        }
        if matches!(
            &command.command,
            Command::Resume | Command::StartAttempt { .. } | Command::RecordIntent { .. }
        ) {
            ensure_root_dispatchable(&tx, run_id)?;
        }
        // Every start, planning included, waits its occurrence's turn, not only the outbox claim.
        if matches!(&command.command, Command::StartAttempt { .. }) {
            crate::schedule::ensure_occurrence_turn(&tx, run_id)?;
        }
        if matches!(&command.command, Command::CompleteRun { .. }) {
            let children: Vec<String> = {
                let mut stmt = tx
                    .prepare("SELECT child.state FROM children relation JOIN runs child ON child.id=relation.child_id WHERE relation.parent_id=?1")
                    .map_err(err)?;
                let rows = stmt.query_map([run_id], |r| r.get(0)).map_err(err)?;
                rows.collect::<std::result::Result<_, _>>().map_err(err)?
            };
            for json in children {
                let child = decode_state(&json)?;
                if !child.status.terminal()
                    || child.effects.values().any(|effect| {
                        matches!(
                            effect.status,
                            EffectStatus::Intent | EffectStatus::Uncertain
                        )
                    })
                {
                    return Err("run has active or unsettled children".into());
                }
            }
        }
        let (next, event, reply) = reducer::apply(&state, command)?;
        let changed = tx
            .execute(
                "UPDATE runs SET revision=?1,state=?2 WHERE id=?3 AND revision=?4",
                params![
                    sqlite_int(next.revision)?,
                    serde_json::to_string(&next).map_err(err)?,
                    run_id,
                    sqlite_int(state.revision)?
                ],
            )
            .map_err(err)?;
        if changed != 1 {
            return Err("revision conflict".into());
        }
        tx.execute(
            "INSERT INTO events(run_id,revision,event) VALUES (?1,?2,?3)",
            params![
                run_id,
                sqlite_int(event.revision)?,
                serde_json::to_string(&event).map_err(err)?
            ],
        )
        .map_err(err)?;
        tx.execute(
            "INSERT INTO commands(run_id,id,request,reply) VALUES (?1,?2,?3,?4)",
            params![
                run_id,
                command.id,
                request,
                serde_json::to_string(&reply).map_err(err)?
            ],
        )
        .map_err(err)?;
        // A root's cancellation and revocation reach its children in the same commit: a child's
        // decisions compare the child's own grant generation, which only this moves.
        if matches!(
            &command.command,
            Command::Cancel | Command::RevokeGrant { .. }
        ) {
            let children: Vec<String> = {
                let mut stmt = tx
                    .prepare("SELECT child_id FROM children WHERE parent_id=?1")
                    .map_err(err)?;
                let rows = stmt.query_map([run_id], |r| r.get(0)).map_err(err)?;
                rows.collect::<std::result::Result<_, _>>().map_err(err)?
            };
            for child_id in children {
                let json: String = tx
                    .query_row("SELECT state FROM runs WHERE id=?1", [&child_id], |r| {
                        r.get(0)
                    })
                    .map_err(err)?;
                let child = decode_state(&json)?;
                if child.status.terminal() {
                    continue;
                }
                let (cancelled, child_event) =
                    if let Command::RevokeGrant { grant } = &command.command {
                        let mut revoked = child.clone();
                        revoked.definition.grants.retain(|value| value != grant);
                        revoked.grant_generation = revoked
                            .grant_generation
                            .checked_add(1)
                            .ok_or("grant generation overflow")?;
                        revoked.revision =
                            revoked.revision.checked_add(1).ok_or("revision overflow")?;
                        let event = Event {
                            revision: revoked.revision,
                            command_id: format!("parent-revoke:{run_id}:{}", command.id),
                            kind: "grant_revoked".into(),
                        };
                        (revoked, event)
                    } else {
                        let cancellation = CommandEnvelope {
                            id: format!("parent-cancel:{run_id}:{}", command.id),
                            expected_revision: child.revision,
                            command: Command::Cancel,
                        };
                        let (cancelled, event, _) = reducer::apply(&child, &cancellation)?;
                        (cancelled, event)
                    };
                tx.execute(
                    "UPDATE runs SET revision=?1,state=?2 WHERE id=?3",
                    params![
                        sqlite_int(cancelled.revision)?,
                        serde_json::to_string(&cancelled).map_err(err)?,
                        child_id
                    ],
                )
                .map_err(err)?;
                tx.execute(
                    "INSERT INTO events(run_id,revision,event) VALUES (?1,?2,?3)",
                    params![
                        child_id,
                        sqlite_int(child_event.revision)?,
                        serde_json::to_string(&child_event).map_err(err)?
                    ],
                )
                .map_err(err)?;
            }
        }
        tx.commit().map_err(err)?;
        Ok((reply, true))
    }
}
fn decode_state(json: &str) -> Result<RunState> {
    let state: RunState = serde_json::from_str(json).map_err(err)?;
    if state.version != SCHEMA_VERSION {
        return Err(format!("unsupported run version {}", state.version));
    }
    state.definition.validate()?;
    Ok(state)
}

/// A verified set of content-addressed objects referenced by a database snapshot.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BackupManifest {
    pub version: u32,
    pub artifacts: Vec<String>,
}
impl SqliteStore {
    /// Pin operations and collection must run under the service's data-directory ownership lock.
    pub fn pin_artifact(&mut self, run_id: &str, digest: &str, kind: &str) -> Result<()> {
        let _guard = crate::artifacts::lock().map_err(err)?;
        self.pin_artifact_locked(run_id, digest, kind)
    }
    /// Cross-process safe pin for a known artifact root.
    pub fn pin_artifact_in(
        &mut self,
        artifacts: &crate::artifacts::ArtifactStore,
        run_id: &str,
        digest: &str,
        kind: &str,
    ) -> Result<()> {
        let _guard = crate::artifacts::lock().map_err(err)?;
        let _file_guard = artifacts.lock_filesystem().map_err(err)?;
        self.verify_artifact_owner(artifacts)?;
        artifacts.read_locked(digest).map_err(err)?;
        self.pin_artifact_locked(run_id, digest, kind)
    }
    fn pin_artifact_locked(&mut self, run_id: &str, digest: &str, kind: &str) -> Result<()> {
        if !crate::artifacts::valid_digest(digest) || kind.is_empty() {
            return Err("invalid artifact pin".into());
        }
        self.connection
            .execute(
                "INSERT OR IGNORE INTO artifact_pins(run_id,digest,kind) VALUES (?1,?2,?3)",
                params![run_id, digest, kind],
            )
            .map_err(err)?;
        Ok(())
    }
    pub fn unpin_artifact(&mut self, run_id: &str, digest: &str, kind: &str) -> Result<()> {
        let _guard = crate::artifacts::lock().map_err(err)?;
        if matches!(kind, "pending_decision" | "undo") {
            return Err("protected artifact pins require lifecycle release".into());
        }
        self.connection
            .execute(
                "DELETE FROM artifact_pins WHERE run_id=?1 AND digest=?2 AND kind=?3",
                params![run_id, digest, kind],
            )
            .map_err(err)?;
        Ok(())
    }
    /// Release an ordinary pin while holding the artifact-root process lock.
    pub fn unpin_artifact_in(
        &mut self,
        artifacts: &crate::artifacts::ArtifactStore,
        run_id: &str,
        digest: &str,
        kind: &str,
    ) -> Result<()> {
        let _guard = crate::artifacts::lock().map_err(err)?;
        let _file_guard = artifacts.lock_filesystem().map_err(err)?;
        self.verify_artifact_owner(artifacts)?;
        if matches!(kind, "pending_decision" | "undo") {
            return Err("protected artifact pins require lifecycle release".into());
        }
        self.connection
            .execute(
                "DELETE FROM artifact_pins WHERE run_id=?1 AND digest=?2 AND kind=?3",
                params![run_id, digest, kind],
            )
            .map_err(err)?;
        Ok(())
    }
    /// Publish and pin under one process-wide critical section, so collection cannot race publication.
    pub fn put_pinned(
        &mut self,
        artifacts: &crate::artifacts::ArtifactStore,
        run_id: &str,
        bytes: &[u8],
        kind: &str,
    ) -> Result<String> {
        let _guard = crate::artifacts::lock().map_err(err)?;
        let _file_guard = artifacts.lock_filesystem().map_err(err)?;
        self.check_storage_cap(artifacts, bytes)?;
        let digest = artifacts.put_locked(bytes).map_err(err)?;
        self.pin_artifact_locked(run_id, &digest, kind)?;
        Ok(digest)
    }
    pub fn collect_unpinned(
        &mut self,
        artifacts: &crate::artifacts::ArtifactStore,
    ) -> Result<Vec<String>> {
        let _guard = crate::artifacts::lock().map_err(err)?;
        let _file_guard = artifacts.lock_filesystem().map_err(err)?;
        self.verify_artifact_owner(artifacts)?;
        let mut stmt = self
            .connection
            .prepare("SELECT DISTINCT digest FROM artifact_pins")
            .map_err(err)?;
        let pins = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(err)?
            .collect::<std::result::Result<std::collections::BTreeSet<_>, _>>()
            .map_err(err)?;
        let mut removed = Vec::new();
        for entry in std::fs::read_dir(&artifacts.root).map_err(err)? {
            let entry = entry.map_err(err)?;
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if crate::artifacts::valid_digest(&name)
                && !pins.contains(&name)
                && entry.file_type().map_err(err)?.is_file()
            {
                std::fs::remove_file(entry.path()).map_err(err)?;
                removed.push(name);
            }
        }
        crate::artifacts::sync_dir(&artifacts.root).map_err(err)?;
        removed.sort();
        Ok(removed)
    }
    /// Snapshot SQLite and the references in that snapshot under the artifact lock.
    pub fn backup_bundle(
        &mut self,
        artifacts: &crate::artifacts::ArtifactStore,
        destination: &Path,
    ) -> Result<BackupManifest> {
        let _guard = crate::artifacts::lock().map_err(err)?;
        let _file_guard = artifacts.lock_filesystem().map_err(err)?;
        self.verify_artifact_owner(artifacts)?;
        if destination.exists() {
            return Err("backup destination exists".into());
        }
        let stage = destination.with_extension(format!("stage-{}", std::process::id()));
        std::fs::create_dir(&stage).map_err(err)?;
        let outcome = (|| {
            let db = stage.join("state.sqlite");
            let path = db.to_str().ok_or("non-UTF-8 backup path")?;
            self.connection
                .execute("VACUUM INTO ?1", [path])
                .map_err(err)?;
            let snapshot = Connection::open(&db).map_err(err)?;
            let mut statement = snapshot
                .prepare("SELECT DISTINCT digest FROM artifact_pins ORDER BY digest")
                .map_err(err)?;
            let digests = statement
                .query_map([], |r| r.get::<_, String>(0))
                .map_err(err)?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(err)?;
            let object_dir = stage.join("objects");
            std::fs::create_dir(&object_dir).map_err(err)?;
            for digest in &digests {
                let bytes = artifacts.read_locked(digest).map_err(err)?;
                let path = object_dir.join(digest);
                let mut file = std::fs::File::create(path).map_err(err)?;
                std::io::Write::write_all(&mut file, &bytes).map_err(err)?;
                file.sync_all().map_err(err)?;
            }
            let manifest = BackupManifest {
                version: 1,
                artifacts: digests,
            };
            let mut file = std::fs::File::create(stage.join("manifest.json")).map_err(err)?;
            std::io::Write::write_all(&mut file, &serde_json::to_vec(&manifest).map_err(err)?)
                .map_err(err)?;
            file.sync_all().map_err(err)?;
            crate::artifacts::sync_dir(&object_dir).map_err(err)?;
            crate::artifacts::sync_dir(&stage).map_err(err)?;
            // Windows refuses to rename a folder while a file in it is open.
            drop(file);
            drop(statement);
            drop(snapshot);
            std::fs::rename(&stage, destination).map_err(err)?;
            if let Some(parent) = destination.parent().filter(|p| !p.as_os_str().is_empty()) {
                crate::artifacts::sync_dir(parent).map_err(err)?;
            }
            Ok(manifest)
        })();
        if outcome.is_err() {
            let _ = std::fs::remove_dir_all(stage);
        }
        outcome
    }
    /// Stage a verified copy of the bundle; `published` is the root it will be installed as.
    fn restore_staged(
        bundle: &Path,
        destination_db: &Path,
        destination_artifacts: &Path,
        published: &Path,
    ) -> Result<()> {
        let _guard = crate::artifacts::lock().map_err(err)?;
        if destination_db.exists() || destination_artifacts.exists() {
            return Err("restore destination exists".into());
        }
        let manifest: BackupManifest =
            serde_json::from_slice(&std::fs::read(bundle.join("manifest.json")).map_err(err)?)
                .map_err(err)?;
        if manifest.version != 1 {
            return Err("unsupported backup manifest".into());
        }
        let source = Connection::open(bundle.join("state.sqlite")).map_err(err)?;
        let check: String = source
            .query_row("PRAGMA integrity_check", [], |r| r.get(0))
            .map_err(err)?;
        if check != "ok" {
            return Err("backup integrity check failed".into());
        }
        let mut stmt = source
            .prepare("SELECT DISTINCT digest FROM artifact_pins ORDER BY digest")
            .map_err(err)?;
        let pins = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(err)?;
        if pins != manifest.artifacts {
            return Err("backup manifest differs from snapshot".into());
        }
        let bundle_objects = crate::artifacts::ArtifactStore {
            root: bundle.join("objects"),
        };
        for digest in &pins {
            bundle_objects.read_locked(digest).map_err(err)?;
        }
        drop(stmt);
        drop(source);
        let staged_db = destination_db.with_extension(format!("restore-{}", std::process::id()));
        let staged_objects =
            destination_artifacts.with_extension(format!("restore-{}", std::process::id()));
        if staged_db.exists() || staged_objects.exists() {
            return Err("restore staging exists".into());
        }
        let outcome = (|| {
            std::fs::copy(bundle.join("state.sqlite"), &staged_db).map_err(err)?;
            std::fs::create_dir(&staged_objects).map_err(err)?;
            for digest in &pins {
                let bytes = bundle_objects.read_locked(digest).map_err(err)?;
                let mut file = std::fs::File::create(staged_objects.join(digest)).map_err(err)?;
                std::io::Write::write_all(&mut file, &bytes).map_err(err)?;
                file.sync_all().map_err(err)?;
            }
            let mut db = Connection::open(&staged_db).map_err(err)?;
            db.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA foreign_keys=ON;")
                .map_err(err)?;
            let tx = db
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .map_err(err)?;
            let states: Vec<String> = {
                let mut stmt = tx.prepare("SELECT state FROM runs").map_err(err)?;
                let rows = stmt.query_map([], |r| r.get(0)).map_err(err)?;
                rows.collect::<std::result::Result<_, _>>().map_err(err)?
            };
            for json in states {
                let mut state = decode_state(&json)?;
                state.definition.grants.clear();
                state.grant_generation = state
                    .grant_generation
                    .checked_add(1)
                    .ok_or("grant generation overflow")?;
                if !state.status.terminal() {
                    if let Some(attempt) = state.attempt.take() {
                        for effect in state.effects.values_mut() {
                            if effect.attempt_id == attempt.id
                                && effect.status == EffectStatus::Intent
                            {
                                effect.status = EffectStatus::Uncertain;
                            }
                        }
                    }
                    state.status = RunStatus::Paused;
                }
                state.revision = state.revision.checked_add(1).ok_or("revision overflow")?;
                let event = Event {
                    revision: state.revision,
                    command_id: format!("restore:{}", state.revision),
                    kind: "restored_paused".into(),
                };
                tx.execute(
                    "UPDATE runs SET revision=?1,state=?2 WHERE id=?3",
                    params![
                        sqlite_int(state.revision)?,
                        serde_json::to_string(&state).map_err(err)?,
                        state.id
                    ],
                )
                .map_err(err)?;
                tx.execute(
                    "INSERT INTO events(run_id,revision,event) VALUES (?1,?2,?3)",
                    params![
                        state.id,
                        sqlite_int(state.revision)?,
                        serde_json::to_string(&event).map_err(err)?
                    ],
                )
                .map_err(err)?;
            }
            tx.execute("UPDATE schedules SET enabled=0", [])
                .map_err(err)?;
            crate::admission::write_artifact_binding(
                &published.join("tasks.sqlite"),
                &staged_objects,
                &published.join("objects"),
            )?;
            tx.commit().map_err(err)?;
            drop(db);
            std::fs::OpenOptions::new()
                .write(true)
                .open(&staged_db)
                .and_then(|f| f.sync_all())
                .map_err(err)?;
            crate::artifacts::sync_dir(&staged_objects).map_err(err)?;
            std::fs::rename(&staged_objects, destination_artifacts).map_err(err)?;
            if let Err(error) = std::fs::rename(&staged_db, destination_db) {
                let _ = std::fs::remove_dir_all(destination_artifacts);
                return Err(err(error));
            }
            for parent in [destination_db.parent(), destination_artifacts.parent()]
                .into_iter()
                .flatten()
                .filter(|p| !p.as_os_str().is_empty())
            {
                crate::artifacts::sync_dir(parent).map_err(err)?;
            }
            Ok(())
        })();
        if outcome.is_err() {
            let _ = std::fs::remove_file(&staged_db);
            let _ = std::fs::remove_dir_all(&staged_objects);
        }
        outcome
    }
    /// Restore into a new single root and publish it with one directory rename. Every
    /// non-terminal run comes back paused with no grants and schedules disabled: authority
    /// never travels in a bundle, so a run that needs a grant is re-created from its exported
    /// definition. An intent in flight at backup comes back uncertain, to reconcile before
    /// Resume.
    pub fn restore_root(bundle: &Path, destination: &Path) -> Result<()> {
        if destination.exists() {
            return Err("restore destination exists".into());
        }
        let stage = destination.with_extension(format!("restore-root-{}", std::process::id()));
        std::fs::create_dir(&stage).map_err(err)?;
        let outcome = (|| {
            Self::restore_staged(
                bundle,
                &stage.join("tasks.sqlite"),
                &stage.join("objects"),
                destination,
            )?;
            crate::artifacts::sync_dir(&stage).map_err(err)?;
            std::fs::rename(&stage, destination).map_err(err)?;
            if let Some(parent) = destination
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
            {
                crate::artifacts::sync_dir(parent).map_err(err)?;
            }
            Ok(())
        })();
        if outcome.is_err() {
            let _ = std::fs::remove_dir_all(&stage);
        }
        outcome
    }
    pub fn restore_bundle(bundle: &Path, destination: &Path) -> Result<()> {
        Self::restore_root(bundle, destination)
    }
}
