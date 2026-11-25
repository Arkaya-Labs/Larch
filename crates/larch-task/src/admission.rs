//! Fair, in-memory admission bookkeeping. Persist reservations and accounting
//! alongside dispatch/completion before treating permits as durable.
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub id: String,
    pub root: String,
    pub reserve: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permit {
    pub id: String,
    pub root: String,
    pub reserved: u64,
}
#[derive(Clone, Debug, Default)]
struct Budget {
    limit: u64,
    spent: u64,
    reserved: u64,
}
#[derive(Clone, Debug)]
pub struct Admission {
    capacity: usize,
    active: BTreeMap<String, Permit>,
    budgets: BTreeMap<String, Budget>,
    queues: BTreeMap<String, VecDeque<Request>>,
    rotation: VecDeque<String>,
}
impl Admission {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            active: BTreeMap::new(),
            budgets: BTreeMap::new(),
            queues: BTreeMap::new(),
            rotation: VecDeque::new(),
        }
    }
    pub fn set_budget(&mut self, root: String, limit: u64) -> Result<(), String> {
        let budget = self.budgets.entry(root).or_default();
        if budget.spent.saturating_add(budget.reserved) > limit {
            return Err("budget below committed usage".into());
        }
        budget.limit = limit;
        Ok(())
    }
    pub fn enqueue(&mut self, request: Request) -> Result<(), String> {
        if request.id.is_empty()
            || request.root.is_empty()
            || !self.budgets.contains_key(&request.root)
        {
            return Err("unknown root or empty identity".into());
        }
        let budget = self.budgets.get(&request.root).expect("known root");
        if budget
            .spent
            .checked_add(request.reserve)
            .is_none_or(|n| n > budget.limit)
        {
            return Err("request exceeds remaining root budget".into());
        }
        if self.active.contains_key(&request.id)
            || self
                .queues
                .values()
                .any(|q| q.iter().any(|r| r.id == request.id))
        {
            return Err("duplicate request".into());
        }
        let queue = self.queues.entry(request.root.clone()).or_default();
        if queue.is_empty() {
            self.rotation.push_back(request.root.clone());
        }
        queue.push_back(request);
        Ok(())
    }
    /// Round-robin by root. Children cannot increase their root's share.
    pub fn admit(&mut self) -> Option<Permit> {
        if self.active.len() >= self.capacity {
            return None;
        }
        for _ in 0..self.rotation.len() {
            let root = self.rotation.pop_front()?;
            let queue = self.queues.get_mut(&root).expect("queued root");
            let req = queue.front().expect("nonempty queue");
            let budget = self.budgets.get_mut(&root).expect("known root");
            let allowed = budget
                .spent
                .checked_add(budget.reserved)
                .and_then(|n| n.checked_add(req.reserve))
                .is_some_and(|n| n <= budget.limit);
            if !allowed {
                self.rotation.push_back(root);
                continue;
            }
            let req = queue.pop_front().expect("nonempty queue");
            if !queue.is_empty() {
                self.rotation.push_back(root.clone());
            }
            budget.reserved += req.reserve;
            let permit = Permit {
                id: req.id,
                root,
                reserved: req.reserve,
            };
            self.active.insert(permit.id.clone(), permit.clone());
            return Some(permit);
        }
        None
    }
    /// Release the execution seat on completion, failure, cancellation or wait.
    /// `used` can exceed the reservation and is always charged.
    pub fn finish(&mut self, id: &str, used: u64) -> Result<(), String> {
        let permit = self.active.get(id).ok_or("unknown permit")?;
        let budget = self.budgets.get_mut(&permit.root).expect("known root");
        let spent = budget.spent.checked_add(used).ok_or("usage overflow")?;
        budget.reserved -= permit.reserved;
        budget.spent = spent;
        self.active.remove(id);
        Ok(())
    }
    pub fn cancel_queued(&mut self, id: &str) -> bool {
        for (root, queue) in &mut self.queues {
            if let Some(pos) = queue.iter().position(|r| r.id == id) {
                queue.remove(pos);
                if queue.is_empty() {
                    self.rotation.retain(|r| r != root);
                }
                return true;
            }
        }
        false
    }
    pub fn usage(&self, root: &str) -> Option<(u64, u64)> {
        self.budgets.get(root).map(|b| (b.spent, b.reserved))
    }
}

use crate::store::SqliteStore;
use rusqlite::{OptionalExtension, TransactionBehavior, params};
fn db_error(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn sql_integer(value: u64) -> Result<i64, String> {
    i64::try_from(value).map_err(|_| "integer exceeds SQLite range".into())
}
fn unsigned_integer(value: i64) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| "negative SQLite integer".into())
}
impl SqliteStore {
    pub fn create_admission_root(&mut self, id: &str, budget: u64) -> Result<(), String> {
        if id.is_empty() {
            return Err("empty root id".into());
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let is_child: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM children WHERE child_id=?1)",
                [id],
                |r| r.get(0),
            )
            .map_err(db_error)?;
        if is_child {
            return Err("child cannot hold an independent budget".into());
        }
        tx.execute(
            "INSERT INTO admission_roots(id,budget,spent,reserved) VALUES(?1,?2,0,0)",
            params![id, sql_integer(budget)?],
        )
        .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(())
    }
    /// Reservation and global capacity check commit together. Unknown usage remains
    /// reserved after restart until explicitly reconciled by the caller.
    pub fn reserve_permit(&mut self, request: &Request, capacity: usize) -> Result<Permit, String> {
        if request.id.is_empty() || request.id.starts_with("dispatch:") {
            return Err("invalid permit id".into());
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let is_child: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM children WHERE child_id=?1)",
                [&request.root],
                |r| r.get(0),
            )
            .map_err(db_error)?;
        if is_child {
            return Err("child cannot hold an independent budget".into());
        }
        let count: i64 = tx
            .query_row("SELECT COUNT(*) FROM admission_permits", [], |r| r.get(0))
            .map_err(db_error)?;
        if unsigned_integer(count)? >= u64::try_from(capacity).map_err(db_error)? {
            return Err("capacity exhausted".into());
        }
        let (limit, spent, reserved): (i64, i64, i64) = tx
            .query_row(
                "SELECT budget,spent,reserved FROM admission_roots WHERE id=?1",
                [&request.root],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()
            .map_err(db_error)?
            .ok_or("unknown root")?;
        let (limit, spent, reserved) = (
            unsigned_integer(limit)?,
            unsigned_integer(spent)?,
            unsigned_integer(reserved)?,
        );
        let next = reserved
            .checked_add(request.reserve)
            .ok_or("reservation overflow")?;
        if spent.checked_add(next).is_none_or(|n| n > limit) {
            return Err("budget exhausted".into());
        }
        tx.execute(
            "INSERT INTO admission_permits(id,root_id,reserved) VALUES(?1,?2,?3)",
            params![request.id, request.root, sql_integer(request.reserve)?],
        )
        .map_err(db_error)?;
        tx.execute(
            "UPDATE admission_roots SET reserved=?1 WHERE id=?2",
            params![sql_integer(next)?, request.root],
        )
        .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(Permit {
            id: request.id.clone(),
            root: request.root.clone(),
            reserved: request.reserve,
        })
    }
    pub fn settle_permit(&mut self, id: &str, used: u64) -> Result<(), String> {
        if id.starts_with("dispatch:") {
            return Err("dispatch reservation requires transactional settlement".into());
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let (root, amount): (String, i64) = tx
            .query_row(
                "SELECT root_id,reserved FROM admission_permits WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(db_error)?
            .ok_or("unknown permit")?;
        let (spent, reserved): (i64, i64) = tx
            .query_row(
                "SELECT spent,reserved FROM admission_roots WHERE id=?1",
                [&root],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(db_error)?;
        let (amount, spent, reserved) = (
            unsigned_integer(amount)?,
            unsigned_integer(spent)?,
            unsigned_integer(reserved)?,
        );
        let next_spent = spent.checked_add(used).ok_or("usage overflow")?;
        let next_reserved = reserved
            .checked_sub(amount)
            .ok_or("reservation underflow")?;
        tx.execute(
            "UPDATE admission_roots SET spent=?1,reserved=?2 WHERE id=?3",
            params![sql_integer(next_spent)?, sql_integer(next_reserved)?, root],
        )
        .map_err(db_error)?;
        tx.execute("DELETE FROM admission_permits WHERE id=?1", [id])
            .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(())
    }
}

pub(crate) fn attempt_permit(run_id: &str, attempt_id: &str) -> String {
    format!("dispatch:{}:{}:{}", run_id.len(), run_id, attempt_id)
}
fn root_for(tx: &rusqlite::Transaction<'_>, run_id: &str) -> Result<String, String> {
    let parent: Option<String> = tx
        .query_row(
            "SELECT parent_id FROM children WHERE child_id=?1",
            [run_id],
            |r| r.get(0),
        )
        .optional()
        .map_err(db_error)?;
    Ok(parent.unwrap_or_else(|| run_id.to_string()))
}

/// Settle a fenced parent attempt inside the transaction that publishes its child
/// and enters the child wait. The caller checks the parent attempt and graph fence
/// and handles exact command replay before invoking this helper.
pub(crate) fn settle_for_child_wait(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    attempt_id: &str,
    used: u64,
    verified_receipt: &str,
) -> Result<(), String> {
    if verified_receipt.trim().is_empty() {
        return Err("usage receipt is required".into());
    }
    let permit_id = attempt_permit(run_id, attempt_id);
    let (root, amount): (String, i64) = tx
        .query_row(
            "SELECT root_id,reserved FROM admission_permits WHERE id=?1",
            [&permit_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .map_err(db_error)?
        .ok_or("unknown reservation")?;
    if root != root_for(tx, run_id)? {
        return Err("reservation root mismatch".into());
    }
    let amount = unsigned_integer(amount)?;
    if used > amount {
        return Err("usage exceeds reservation".into());
    }
    let (spent, reserved): (i64, i64) = tx
        .query_row(
            "SELECT spent,reserved FROM admission_roots WHERE id=?1",
            [&root],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(db_error)?;
    let next_spent = unsigned_integer(spent)?
        .checked_add(used)
        .ok_or("usage overflow")?;
    let next_reserved = unsigned_integer(reserved)?
        .checked_sub(amount)
        .ok_or("reservation underflow")?;
    tx.execute(
        "INSERT INTO budget_receipts(permit_id,run_id,used,receipt) VALUES(?1,?2,?3,?4)",
        params![permit_id, run_id, sql_integer(used)?, verified_receipt],
    )
    .map_err(db_error)?;
    tx.execute(
        "UPDATE admission_roots SET spent=?1,reserved=?2 WHERE id=?3",
        params![sql_integer(next_spent)?, sql_integer(next_reserved)?, root],
    )
    .map_err(db_error)?;
    if tx
        .execute("DELETE FROM admission_permits WHERE id=?1", [&permit_id])
        .map_err(db_error)?
        != 1
    {
        return Err("reservation changed".into());
    }
    Ok(())
}
fn read_run(tx: &rusqlite::Transaction<'_>, id: &str) -> Result<crate::types::RunState, String> {
    let json: String = tx
        .query_row("SELECT state FROM runs WHERE id=?1", [id], |r| r.get(0))
        .optional()
        .map_err(db_error)?
        .ok_or("unknown run")?;
    let state: crate::types::RunState = serde_json::from_str(&json).map_err(db_error)?;
    if state.version != crate::types::SCHEMA_VERSION {
        return Err("unsupported run version".into());
    }
    state.definition.validate()?;
    Ok(state)
}
fn old_reply(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    command_id: &str,
    request: &str,
) -> Result<Option<crate::types::CommandReply>, String> {
    let previous: Option<(String, String)> = tx
        .query_row(
            "SELECT request,reply FROM commands WHERE run_id=?1 AND id=?2",
            params![run_id, command_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .map_err(db_error)?;
    previous
        .map(|(old, reply)| {
            if old != request {
                return Err("command id reused".into());
            }
            serde_json::from_str(&reply).map_err(db_error)
        })
        .transpose()
}
fn record_transition(
    tx: &rusqlite::Transaction<'_>,
    old: &crate::types::RunState,
    command: &crate::types::CommandEnvelope,
    request: &str,
) -> Result<crate::types::CommandReply, String> {
    let (next, event, reply) = crate::reducer::apply_budgeted(old, command)?;
    let changed = tx
        .execute(
            "UPDATE runs SET revision=?1,state=?2 WHERE id=?3 AND revision=?4",
            params![
                sql_integer(next.revision)?,
                serde_json::to_string(&next).map_err(db_error)?,
                old.id,
                sql_integer(old.revision)?
            ],
        )
        .map_err(db_error)?;
    if changed != 1 {
        return Err("run revision conflict".into());
    }
    tx.execute(
        "INSERT INTO events(run_id,revision,event) VALUES(?1,?2,?3)",
        params![
            old.id,
            sql_integer(event.revision)?,
            serde_json::to_string(&event).map_err(db_error)?
        ],
    )
    .map_err(db_error)?;
    tx.execute(
        "INSERT INTO commands(run_id,id,request,reply) VALUES(?1,?2,?3,?4)",
        params![
            old.id,
            command.id,
            request,
            serde_json::to_string(&reply).map_err(db_error)?
        ],
    )
    .map_err(db_error)?;
    Ok(reply)
}
impl SqliteStore {
    /// Record a failed fenced attempt, its verified usage and its receipt in
    /// the same transaction. Actual measured usage is charged even when it
    /// exceeds the reservation; uncertain usage must leave the permit intact.
    pub fn fail_budgeted_with_receipt(
        &mut self,
        run_id: &str,
        command: &crate::types::CommandEnvelope,
        used: u64,
        verified_receipt: &str,
    ) -> Result<crate::types::CommandReply, String> {
        let attempt_id = match &command.command {
            crate::types::Command::FailAttempt { attempt_id, .. } => attempt_id,
            _ => return Err("receipt settlement requires attempt failure".into()),
        };
        if verified_receipt.trim().is_empty() {
            return Err("usage receipt is required".into());
        }
        let request =
            serde_json::to_string(&(command, used, verified_receipt)).map_err(db_error)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        if let Some(reply) = old_reply(&tx, run_id, &command.id, &request)? {
            return Ok(reply);
        }
        let state = read_run(&tx, run_id)?;
        let permit_id = attempt_permit(run_id, attempt_id);
        let (root, amount): (String, i64) = tx
            .query_row(
                "SELECT root_id,reserved FROM admission_permits WHERE id=?1",
                [&permit_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(db_error)?
            .ok_or("unknown reservation")?;
        if root != root_for(&tx, run_id)? {
            return Err("reservation root mismatch".into());
        }
        let (spent, reserved): (i64, i64) = tx
            .query_row(
                "SELECT spent,reserved FROM admission_roots WHERE id=?1",
                [&root],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(db_error)?;
        let next_spent = unsigned_integer(spent)?
            .checked_add(used)
            .ok_or("usage overflow")?;
        let next_reserved = unsigned_integer(reserved)?
            .checked_sub(unsigned_integer(amount)?)
            .ok_or("reservation underflow")?;
        tx.execute(
            "INSERT INTO budget_receipts(permit_id,run_id,used,receipt) VALUES(?1,?2,?3,?4)",
            params![permit_id, run_id, sql_integer(used)?, verified_receipt],
        )
        .map_err(db_error)?;
        tx.execute(
            "UPDATE admission_roots SET spent=?1,reserved=?2 WHERE id=?3",
            params![sql_integer(next_spent)?, sql_integer(next_reserved)?, root],
        )
        .map_err(db_error)?;
        if tx
            .execute("DELETE FROM admission_permits WHERE id=?1", [&permit_id])
            .map_err(db_error)?
            != 1
        {
            return Err("reservation changed".into());
        }
        let reply = record_transition(&tx, &state, command, &request)?;
        tx.commit().map_err(db_error)?;
        Ok(reply)
    }

    /// The root is derived from the persisted parent relation, never the caller.
    /// Reservation, attempt, and idempotency receipt commit together.
    pub fn dispatch_budgeted(
        &mut self,
        run_id: &str,
        command: &crate::types::CommandEnvelope,
        reserve: u64,
        capacity: usize,
    ) -> Result<crate::types::CommandReply, String> {
        self.dispatch_budgeted_once(run_id, command, reserve, capacity)
            .map(|(reply, _)| reply)
    }
    pub fn dispatch_budgeted_once(
        &mut self,
        run_id: &str,
        command: &crate::types::CommandEnvelope,
        reserve: u64,
        capacity: usize,
    ) -> Result<(crate::types::CommandReply, bool), String> {
        self.dispatch_budgeted_inner(run_id, command, reserve, capacity, None)
    }
    pub fn claim_budgeted_scheduled_start(
        &mut self,
        outbox_id: &str,
        run_id: &str,
        command: &crate::types::CommandEnvelope,
        reserve: u64,
        capacity: usize,
    ) -> Result<crate::types::CommandReply, String> {
        self.claim_budgeted_scheduled_start_once(outbox_id, run_id, command, reserve, capacity)
            .map(|(reply, _)| reply)
    }
    pub fn claim_budgeted_scheduled_start_once(
        &mut self,
        outbox_id: &str,
        run_id: &str,
        command: &crate::types::CommandEnvelope,
        reserve: u64,
        capacity: usize,
    ) -> Result<(crate::types::CommandReply, bool), String> {
        self.dispatch_budgeted_inner(run_id, command, reserve, capacity, Some(outbox_id))
    }
    fn dispatch_budgeted_inner(
        &mut self,
        run_id: &str,
        command: &crate::types::CommandEnvelope,
        reserve: u64,
        capacity: usize,
        outbox_id: Option<&str>,
    ) -> Result<(crate::types::CommandReply, bool), String> {
        use crate::types::Command;
        let attempt = match &command.command {
            Command::StartAttempt { attempt } => attempt,
            _ => return Err("budgeted dispatch requires start".into()),
        };
        let request =
            serde_json::to_string(&(command, reserve, capacity, outbox_id)).map_err(db_error)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        if let Some(reply) = old_reply(&tx, run_id, &command.id, &request)? {
            return Ok((reply, false));
        }
        let permit_prefix = attempt_permit(run_id, "");
        let outstanding: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM admission_permits WHERE substr(id,1,length(?1))=?1)",
                [&permit_prefix],
                |r| r.get(0),
            )
            .map_err(db_error)?;
        if outstanding {
            return Err("run has an unresolved reservation".into());
        }
        crate::store::ensure_root_dispatchable(&tx, run_id)?;
        crate::schedule::ensure_occurrence_turn(&tx, run_id)?;
        if let Some(outbox_id) = outbox_id {
            let row: Option<(String, String, bool)> = tx
                .query_row(
                    "SELECT kind,payload,delivered FROM task_outbox WHERE id=?1",
                    [outbox_id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .optional()
                .map_err(db_error)?;
            if row != Some(("start".into(), run_id.into(), false)) {
                return Err("invalid or already claimed start outbox".into());
            }
        }
        let root_id = root_for(&tx, run_id)?;
        let state = read_run(&tx, run_id)?;
        let root = read_run(&tx, &root_id)?;
        if root.definition.budgets.len() != 1 {
            return Err("exactly one model_tokens budget is supported".into());
        }
        let budget = *root
            .definition
            .budgets
            .get("model_tokens")
            .ok_or("unsupported budget metric")?;
        if state.definition.budgets != root.definition.budgets {
            return Err("child budget must match root budget".into());
        }
        tx.execute(
            "INSERT OR IGNORE INTO admission_roots(id,budget,spent,reserved) VALUES(?1,?2,0,0)",
            params![root_id, sql_integer(budget)?],
        )
        .map_err(db_error)?;
        let (stored, spent, reserved): (i64, i64, i64) = tx
            .query_row(
                "SELECT budget,spent,reserved FROM admission_roots WHERE id=?1",
                [&root_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .map_err(db_error)?;
        let (stored, spent, reserved) = (
            unsigned_integer(stored)?,
            unsigned_integer(spent)?,
            unsigned_integer(reserved)?,
        );
        if stored != budget {
            return Err("root budget does not match ledger".into());
        }
        let next = reserved
            .checked_add(reserve)
            .ok_or("reservation overflow")?;
        if spent.checked_add(next).is_none_or(|n| n > budget) {
            return Err("budget exhausted".into());
        }
        let count: i64 = tx
            .query_row("SELECT COUNT(*) FROM admission_permits", [], |r| r.get(0))
            .map_err(db_error)?;
        if unsigned_integer(count)? >= u64::try_from(capacity).map_err(db_error)? {
            return Err("capacity exhausted".into());
        }
        tx.execute(
            "INSERT INTO admission_permits(id,root_id,reserved) VALUES(?1,?2,?3)",
            params![
                attempt_permit(run_id, &attempt.id),
                root_id,
                sql_integer(reserve)?
            ],
        )
        .map_err(db_error)?;
        tx.execute(
            "UPDATE admission_roots SET reserved=?1 WHERE id=?2",
            params![sql_integer(next)?, root_id],
        )
        .map_err(db_error)?;
        let reply = record_transition(&tx, &state, command, &request)?;
        if let Some(outbox_id) = outbox_id {
            tx.execute(
                "UPDATE task_outbox SET delivered=1 WHERE id=?1",
                [outbox_id],
            )
            .map_err(db_error)?;
        }
        tx.commit().map_err(db_error)?;
        Ok((reply, true))
    }
    /// Commit completion/failure and the actual usage together. Cancellation or
    /// crash without measured usage leaves the reservation for reconciliation.
    pub fn settle_budgeted(
        &mut self,
        run_id: &str,
        command: &crate::types::CommandEnvelope,
        used: u64,
    ) -> Result<crate::types::CommandReply, String> {
        use crate::types::Command;
        let attempt_id = match &command.command {
            Command::FinishAttempt { attempt_id, .. } | Command::FailAttempt { attempt_id, .. } => {
                attempt_id
            }
            _ => return Err("budget settlement requires completion or failure".into()),
        };
        let request = serde_json::to_string(&(command, used)).map_err(db_error)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        if let Some(reply) = old_reply(&tx, run_id, &command.id, &request)? {
            return Ok(reply);
        }
        let state = read_run(&tx, run_id)?;
        let permit_id = attempt_permit(run_id, attempt_id);
        let (root, amount): (String, i64) = tx
            .query_row(
                "SELECT root_id,reserved FROM admission_permits WHERE id=?1",
                [&permit_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(db_error)?
            .ok_or("unknown reservation")?;
        if root != root_for(&tx, run_id)? {
            return Err("reservation root mismatch".into());
        }
        let (spent, reserved): (i64, i64) = tx
            .query_row(
                "SELECT spent,reserved FROM admission_roots WHERE id=?1",
                [&root],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(db_error)?;
        let next_spent = unsigned_integer(spent)?
            .checked_add(used)
            .ok_or("usage overflow")?;
        let next_reserved = unsigned_integer(reserved)?
            .checked_sub(unsigned_integer(amount)?)
            .ok_or("reservation underflow")?;
        tx.execute(
            "UPDATE admission_roots SET spent=?1,reserved=?2 WHERE id=?3",
            params![sql_integer(next_spent)?, sql_integer(next_reserved)?, root],
        )
        .map_err(db_error)?;
        tx.execute("DELETE FROM admission_permits WHERE id=?1", [&permit_id])
            .map_err(db_error)?;
        let reply = record_transition(&tx, &state, command, &request)?;
        tx.commit().map_err(db_error)?;
        Ok(reply)
    }
}

impl SqliteStore {
    /// Reconcile a stopped attempt only after the driver has verified a usage receipt.
    /// Unknown usage must remain reserved. Do not expose this as an unauthenticated API.
    pub fn reconcile_budgeted_usage(
        &mut self,
        run_id: &str,
        attempt_id: &str,
        used: u64,
        verified_receipt: &str,
    ) -> Result<(), String> {
        use crate::types::RunStatus;
        if verified_receipt.trim().is_empty() {
            return Err("usage receipt is required".into());
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let permit_id = attempt_permit(run_id, attempt_id);
        let prior: Option<(String, i64, String)> = tx
            .query_row(
                "SELECT run_id,used,receipt FROM budget_receipts WHERE permit_id=?1",
                [&permit_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()
            .map_err(db_error)?;
        if let Some((prior_run, prior_used, prior_receipt)) = prior {
            return if prior_run == run_id
                && unsigned_integer(prior_used)? == used
                && prior_receipt == verified_receipt
            {
                Ok(())
            } else {
                Err("conflicting usage receipt".into())
            };
        }
        let state = read_run(&tx, run_id)?;
        if !matches!(state.status, RunStatus::Waiting | RunStatus::Cancelled)
            || state.attempt.is_some()
        {
            return Err("attempt must be stopped before reconciliation".into());
        }
        let (root, amount): (String, i64) = tx
            .query_row(
                "SELECT root_id,reserved FROM admission_permits WHERE id=?1",
                [&permit_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(db_error)?
            .ok_or("unknown reservation")?;
        if root != root_for(&tx, run_id)? {
            return Err("reservation root mismatch".into());
        }
        let (spent, reserved): (i64, i64) = tx
            .query_row(
                "SELECT spent,reserved FROM admission_roots WHERE id=?1",
                [&root],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(db_error)?;
        let next_spent = unsigned_integer(spent)?
            .checked_add(used)
            .ok_or("usage overflow")?;
        let next_reserved = unsigned_integer(reserved)?
            .checked_sub(unsigned_integer(amount)?)
            .ok_or("reservation underflow")?;
        tx.execute(
            "UPDATE admission_roots SET spent=?1,reserved=?2 WHERE id=?3",
            params![sql_integer(next_spent)?, sql_integer(next_reserved)?, root],
        )
        .map_err(db_error)?;
        tx.execute(
            "INSERT INTO budget_receipts(permit_id,run_id,used,receipt) VALUES(?1,?2,?3,?4)",
            params![permit_id, run_id, sql_integer(used)?, verified_receipt],
        )
        .map_err(db_error)?;
        tx.execute("DELETE FROM admission_permits WHERE id=?1", [&permit_id])
            .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(())
    }
}

/// Schema for a persisted artifact-byte cap. The caller must run all earlier migrations first.
pub(crate) fn migrate_v5(db: &rusqlite::Connection) -> Result<(), String> {
    db.execute_batch("BEGIN IMMEDIATE;").map_err(db_error)?;
    let result = (|| {
        let version: u32 = db
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .map_err(db_error)?;
        if version > crate::store::STORE_VERSION {
            return Err(format!("unsupported store version {version}"));
        }
        if version < 4 {
            return Err("previous migration missing".into());
        }
        if version == 4 {
            db.execute_batch("CREATE TABLE storage_policy(id INTEGER PRIMARY KEY CHECK(id=1), byte_cap INTEGER); INSERT INTO storage_policy(id,byte_cap) VALUES(1,NULL); PRAGMA user_version=5;")
                .map_err(db_error)?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => db.execute_batch("COMMIT;").map_err(db_error),
        Err(error) => {
            let _ = db.execute_batch("ROLLBACK;");
            Err(error)
        }
    }
}
/// Bind a managed artifact root to exactly one database, named relative to the root so the
/// binding survives moving both together. `published` is where the root is read from once
/// installed (a restore writes into a staging copy). The caller must hold the artifact
/// filesystem lock, or use a private restore staging directory.
pub(crate) fn write_artifact_binding(
    database: &std::path::Path,
    root: &std::path::Path,
    published: &std::path::Path,
) -> Result<(), String> {
    if database
        .components()
        .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return Err("database path cannot contain parent traversal".into());
    }
    let (from, to) = (canonical(published)?, canonical(database)?);
    let common = from
        .components()
        .zip(to.components())
        .take_while(|(a, b)| a == b)
        .count();
    let relative: std::path::PathBuf = from
        .components()
        .skip(common)
        .map(|_| std::path::Component::ParentDir)
        .chain(to.components().skip(common))
        .collect();
    let owner = relative.to_str().ok_or("database path is not UTF-8")?;
    let encoded = serde_json::to_vec(owner).map_err(db_error)?;
    let path = root.join(".larch-managed-cap");
    match std::fs::read(&path) {
        Ok(previous) if previous == encoded => return Ok(()),
        Ok(_) => return Err("artifact root belongs to a different database".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(db_error(error)),
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(db_error)?;
    file.write_all(&encoded).map_err(db_error)?;
    file.sync_all().map_err(db_error)?;
    crate::artifacts::sync_dir(root).map_err(db_error)?;
    Ok(())
}
/// The canonical form of a path whose tail may not exist yet.
fn canonical(path: &std::path::Path) -> Result<std::path::PathBuf, String> {
    let mut ancestor = path;
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        suffix.push(
            ancestor
                .file_name()
                .ok_or("a filename is required")?
                .to_os_string(),
        );
        ancestor = ancestor
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
    }
    let mut resolved = std::fs::canonicalize(ancestor).map_err(db_error)?;
    for component in suffix.iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}
fn database_path(db: &rusqlite::Connection) -> Result<std::path::PathBuf, String> {
    let path: String = db
        .query_row("PRAGMA database_list", [], |r| r.get(2))
        .map_err(db_error)?;
    if path.is_empty() {
        return Err("managed artifacts require a file-backed database".into());
    }
    Ok(std::path::PathBuf::from(path))
}
impl SqliteStore {
    /// Updates the cap under the same lock as artifact publication and collection.
    pub fn set_storage_cap(
        &mut self,
        artifacts: &crate::artifacts::ArtifactStore,
        bytes: u64,
    ) -> Result<(), String> {
        let _guard = crate::artifacts::lock().map_err(db_error)?;
        let _filesystem_guard = artifacts.lock_filesystem().map_err(db_error)?;
        self.verify_artifact_owner(artifacts)?;
        self.connection
            .execute(
                "UPDATE storage_policy SET byte_cap=?1 WHERE id=1",
                [sql_integer(bytes)?],
            )
            .map_err(db_error)?;
        Ok(())
    }
    /// Bind the root to this database on its first managed use, or refuse another database's
    /// root: collection deletes whatever this database has not pinned. Caller holds the
    /// artifact locks through the subsequent operation.
    pub(crate) fn verify_artifact_owner(
        &self,
        artifacts: &crate::artifacts::ArtifactStore,
    ) -> Result<(), String> {
        write_artifact_binding(
            &database_path(&self.connection)?,
            &artifacts.root,
            &artifacts.root,
        )
    }
    /// Caller holds artifacts::lock through the subsequent publication and pin.
    pub(crate) fn check_storage_cap(
        &self,
        artifacts: &crate::artifacts::ArtifactStore,
        bytes: &[u8],
    ) -> Result<(), String> {
        use sha2::{Digest, Sha256};
        let cap: Option<i64> = self
            .connection
            .query_row("SELECT byte_cap FROM storage_policy WHERE id=1", [], |r| {
                r.get(0)
            })
            .map_err(db_error)?;
        self.verify_artifact_owner(artifacts)?;
        let Some(cap) = cap else {
            return Ok(());
        };
        let cap = unsigned_integer(cap)?;
        let mut total = 0u64;
        for entry in std::fs::read_dir(&artifacts.root).map_err(db_error)? {
            let entry = entry.map_err(db_error)?;
            if entry.file_name() == ".larch-artifacts.lock"
                || entry.file_name() == ".larch-managed-cap"
            {
                continue;
            }
            let meta = std::fs::symlink_metadata(entry.path()).map_err(db_error)?;
            if !meta.file_type().is_file() {
                return Err("unexpected non-file in artifact store".into());
            }
            total = total
                .checked_add(meta.len())
                .ok_or("artifact storage size overflow")?;
        }
        let digest = format!("{:x}", Sha256::digest(bytes));
        let already_exists = artifacts.root.join(&digest).exists();
        let extra = if already_exists {
            0
        } else {
            u64::try_from(bytes.len()).map_err(db_error)?
        };
        if total.checked_add(extra).is_none_or(|size| size > cap) {
            return Err("artifact storage cap exceeded".into());
        }
        Ok(())
    }
    /// Portable workflow exports omit grants; callers must handle secrets in free text.
    pub fn export_workflow(&self, run_id: &str) -> Result<Vec<u8>, String> {
        let mut definition = self.get(run_id)?.ok_or("unknown run")?.definition;
        definition.grants.clear();
        serde_json::to_vec(&definition).map_err(db_error)
    }
    pub fn import_workflow(bytes: &[u8]) -> Result<crate::types::WorkflowDefinition, String> {
        if bytes.len() > 1_048_576 {
            return Err("workflow import exceeds size limit".into());
        }
        let mut definition: crate::types::WorkflowDefinition =
            serde_json::from_slice(bytes).map_err(db_error)?;
        definition.grants.clear();
        definition.validate()?;
        Ok(definition)
    }
    /// Reject deletion when a live run, child, occurrence, pin or receipt still refers to it.
    pub fn delete_run(&mut self, run_id: &str) -> Result<(), String> {
        let _guard = crate::artifacts::lock().map_err(db_error)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let state = read_run(&tx, run_id)?;
        if !state.status.terminal()
            || state.effects.values().any(|effect| {
                matches!(
                    effect.status,
                    crate::types::EffectStatus::Intent | crate::types::EffectStatus::Uncertain
                )
            })
        {
            return Err("run is active or has uncertain effects".into());
        }
        let referenced: bool = tx
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM artifact_pins WHERE run_id=?1 UNION ALL SELECT 1 FROM children WHERE parent_id=?1 OR child_id=?1 UNION ALL SELECT 1 FROM occurrences WHERE run_id=?1 UNION ALL SELECT 1 FROM budget_receipts WHERE run_id=?1 UNION ALL SELECT 1 FROM admission_permits WHERE root_id=?1)",
                [run_id],
                |r| r.get(0),
            )
            .map_err(db_error)?;
        if referenced {
            return Err("run is still referenced".into());
        }
        tx.execute("DELETE FROM execution_plans WHERE run_id=?1", [run_id])
            .map_err(db_error)?;
        tx.execute("DELETE FROM events WHERE run_id=?1", [run_id])
            .map_err(db_error)?;
        tx.execute("DELETE FROM step_stats WHERE run_id=?1", [run_id])
            .map_err(db_error)?;
        tx.execute("DELETE FROM commands WHERE run_id=?1", [run_id])
            .map_err(db_error)?;
        tx.execute("DELETE FROM admission_roots WHERE id=?1", [run_id])
            .map_err(db_error)?;
        tx.execute("DELETE FROM runs WHERE id=?1", [run_id])
            .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(())
    }
}
