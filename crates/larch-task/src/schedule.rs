//! Deterministic schedule calculations. Persistence and occurrence uniqueness belong
//! in the transaction that claims the schedule revision and creates the run.
use chrono::{
    DateTime, Datelike, Duration, LocalResult, NaiveDate, NaiveTime, Offset, TimeZone, Utc,
};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Schedule {
    Once {
        at: DateTime<Utc>,
    },
    Interval {
        anchor: DateTime<Utc>,
        seconds: u32,
    },
    /// Civil time in an IANA timezone on the weekdays `days` names. Rule revision must be
    /// pinned by the host.
    Calendar {
        start: NaiveDate,
        time: NaiveTime,
        timezone: Tz,
        rule_revision: String,
        /// Weekdays it runs on: bit 0 is Monday through bit 6 Sunday; saved before weekdays
        /// existed, every day.
        #[serde(default = "every_day")]
        days: u8,
    },
}
/// Every weekday in `Schedule::Calendar::days`.
pub const EVERY_DAY: u8 = 0x7f;
fn every_day() -> u8 {
    EVERY_DAY
}
/// Whether `days` holds `date`'s weekday; a mask with no day or a bit past Sunday is refused,
/// since it would never run or names no day.
fn on_day(days: u8, date: NaiveDate) -> Result<bool, String> {
    if days == 0 || days > EVERY_DAY {
        return Err("a calendar schedule needs one to seven weekdays".into());
    }
    Ok(days & (1 << date.weekday().num_days_from_monday()) != 0)
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Occurrence {
    pub schedule_id: String,
    pub scheduled_at: DateTime<Utc>,
}
/// What a due occurrence does while an earlier run of the schedule is unfinished
/// (including one waiting for the user).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Overlap {
    /// Record it missed. At most one occurrence of a due window starts: the newest.
    Skip,
    /// Queue it behind the running one; runs start one at a time, oldest first.
    Queue { max_queued: usize },
    /// Keep one buffered occurrence, replaced by the newest; it starts after the running one.
    BufferLatest,
    /// Start it beside the others, up to `max_active` unfinished runs.
    Parallel { max_active: usize },
}
impl Overlap {
    /// Whether occurrences start one at a time, oldest first.
    fn serial(self) -> bool {
        matches!(self, Self::Queue { .. } | Self::BufferLatest)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CatchUp {
    Latest,
    All { max: usize },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub overlap: Overlap,
    pub catch_up: CatchUp,
    pub freshness_seconds: u32,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            overlap: Overlap::Skip,
            catch_up: CatchUp::Latest,
            freshness_seconds: 86400,
        }
    }
}
/// The largest queue, parallel or catch-up bound a client may set: one tick never starts more.
pub const MAX_POLICY_BOUND: usize = 32;
impl Policy {
    /// Policies come from clients: a zero bound never runs and a large one is a catch-up storm.
    fn validate(&self) -> Result<(), String> {
        let bounded = |n: usize| (1..=MAX_POLICY_BOUND).contains(&n);
        let overlap = match self.overlap {
            Overlap::Queue { max_queued: n } | Overlap::Parallel { max_active: n } => bounded(n),
            Overlap::Skip | Overlap::BufferLatest => true,
        };
        let catch_up = match self.catch_up {
            CatchUp::All { max } => bounded(max),
            CatchUp::Latest => true,
        };
        if overlap && catch_up {
            Ok(())
        } else {
            Err(format!(
                "schedule queue, parallel and catch-up bounds must be 1 to {MAX_POLICY_BOUND}"
            ))
        }
    }
}
/// The UTC instant of a daily civil time. An ambiguous time takes the earlier instant; a time a
/// DST gap skips runs at the instant it names under the offset before the gap (02:30 EST is
/// 03:30 EDT), so every day has exactly one occurrence.
fn civil(timezone: &Tz, date: NaiveDate, time: NaiveTime) -> Result<DateTime<Utc>, String> {
    let local = date.and_time(time);
    Ok(match timezone.from_local_datetime(&local) {
        LocalResult::Single(value) => value.with_timezone(&Utc),
        LocalResult::Ambiguous(a, b) => a.with_timezone(&Utc).min(b.with_timezone(&Utc)),
        LocalResult::None => {
            let day_before = local
                .checked_sub_signed(Duration::days(1))
                .ok_or("calendar overflow")?;
            let before = timezone.offset_from_utc_datetime(&day_before).fix();
            local
                .checked_sub_signed(Duration::seconds(before.local_minus_utc().into()))
                .ok_or("calendar overflow")?
                .and_utc()
        }
    })
}
fn check_rules(rule_revision: &str) -> Result<(), String> {
    if rule_revision != chrono_tz::IANA_TZDB_VERSION {
        return Err(format!(
            "timezone rule revision {rule_revision} is unavailable; installed revision is {}",
            chrono_tz::IANA_TZDB_VERSION
        ));
    }
    Ok(())
}
fn after(at: DateTime<Utc>) -> Result<DateTime<Utc>, String> {
    at.checked_add_signed(Duration::nanoseconds(1))
        .ok_or_else(|| "schedule time overflow".into())
}
impl Schedule {
    /// First occurrence at or after `from`.
    pub fn next_at_or_after(&self, from: DateTime<Utc>) -> Result<Option<DateTime<Utc>>, String> {
        match self {
            Self::Once { at } => Ok((*at >= from).then_some(*at)),
            Self::Interval { anchor, seconds } => {
                if *seconds == 0 {
                    return Err("interval must be positive".into());
                }
                if from <= *anchor {
                    return Ok(Some(*anchor));
                }
                let elapsed = from.signed_duration_since(*anchor).num_seconds();
                let period = i64::from(*seconds);
                let mut steps = elapsed / period;
                let mut candidate = anchor
                    .checked_add_signed(Duration::seconds(
                        steps.checked_mul(period).ok_or("interval overflow")?,
                    ))
                    .ok_or("interval overflow")?;
                if candidate < from {
                    steps = steps.checked_add(1).ok_or("interval overflow")?;
                    candidate = anchor
                        .checked_add_signed(Duration::seconds(
                            steps.checked_mul(period).ok_or("interval overflow")?,
                        ))
                        .ok_or("interval overflow")?;
                }
                Ok(Some(candidate))
            }
            Self::Calendar {
                start,
                time,
                timezone,
                rule_revision,
                days,
            } => {
                check_rules(rule_revision)?;
                let mut date = (*start).max(from.with_timezone(timezone).date_naive());
                // Bounded search avoids a corrupt timezone/date causing an unbounded tick.
                for _ in 0..370 {
                    if on_day(*days, date)? {
                        let candidate = civil(timezone, date, *time)?;
                        if candidate >= from {
                            return Ok(Some(candidate));
                        }
                    }
                    date = date.succ_opt().ok_or("calendar overflow")?;
                }
                Err("calendar search bound exceeded".into())
            }
        }
    }
    /// Latest occurrence at or before `to`, found without visiting the ones before it.
    pub fn latest_at_or_before(&self, to: DateTime<Utc>) -> Result<Option<DateTime<Utc>>, String> {
        match self {
            Self::Once { at } => Ok((*at <= to).then_some(*at)),
            Self::Interval { anchor, seconds } => {
                if *seconds == 0 {
                    return Err("interval must be positive".into());
                }
                if to < *anchor {
                    return Ok(None);
                }
                let period = i64::from(*seconds);
                let steps = to.signed_duration_since(*anchor).num_seconds() / period;
                anchor
                    .checked_add_signed(Duration::seconds(
                        steps.checked_mul(period).ok_or("interval overflow")?,
                    ))
                    .map(Some)
                    .ok_or_else(|| "interval overflow".into())
            }
            Self::Calendar {
                start,
                time,
                timezone,
                rule_revision,
                days,
            } => {
                check_rules(rule_revision)?;
                let mut date = to.with_timezone(timezone).date_naive();
                for _ in 0..370 {
                    if date < *start {
                        return Ok(None);
                    }
                    if on_day(*days, date)? {
                        let candidate = civil(timezone, date, *time)?;
                        if candidate <= to {
                            return Ok(Some(candidate));
                        }
                    }
                    date = date.pred_opt().ok_or("calendar overflow")?;
                }
                Err("calendar search bound exceeded".into())
            }
        }
    }
    /// The occurrences in `[from, to)` as one missed span, counted without starting any.
    fn span(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        reason: &str,
    ) -> Result<Option<Missed>, String> {
        let Some(first) = self.next_at_or_after(from)?.filter(|at| *at < to) else {
            return Ok(None);
        };
        let end = to
            .checked_sub_signed(Duration::nanoseconds(1))
            .ok_or("schedule time overflow")?;
        let last = self
            .latest_at_or_before(end)?
            .ok_or("schedule span has no last occurrence")?;
        let count = match self {
            Self::Interval { seconds, .. } => {
                u64::try_from(last.signed_duration_since(first).num_seconds() / i64::from(*seconds))
                    .map_err(|_| "schedule span is negative")?
                    + 1
            }
            // ponytail: a calendar span walks its days; arithmetic if daily schedules ever get long freshness windows
            _ => {
                let (mut count, mut at) = (1, first);
                while at < last {
                    at = self
                        .next_at_or_after(after(at)?)?
                        .ok_or("schedule span has no last occurrence")?;
                    count += 1;
                }
                count
            }
        };
        Ok(Some(Missed {
            first,
            last,
            count,
            reason: reason.into(),
        }))
    }
}
/// Occurrences passed over in an earlier run's way, with no room to start or queue them.
pub const MISSED_OVERLAP: &str = "overlap";
/// Occurrences a newer one of the same due window replaced (catch-up or the buffered slot).
pub const MISSED_SUPERSEDED: &str = "superseded";
/// Occurrences older than the freshness window when the schedule was next pumped.
pub const MISSED_STALE: &str = "stale";
/// A run of consecutive occurrences passed over for one reason.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Missed {
    pub first: DateTime<Utc>,
    pub last: DateTime<Utc>,
    pub count: u64,
    pub reason: String,
}
/// What one pump does with a schedule's due window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Due {
    /// Occurrences to start, oldest first.
    pub selected: Vec<Occurrence>,
    /// Occurrences passed over for good.
    pub missed: Vec<Missed>,
    /// The cursor after this pump: occurrences at or after it are still due.
    pub next_from: DateTime<Utc>,
}
/// Occurrences to start now. `scan_limit` bounds how many one call may start.
#[allow(clippy::too_many_arguments)]
pub fn due(
    schedule: &Schedule,
    id: &str,
    from: DateTime<Utc>,
    now: DateTime<Utc>,
    policy: Policy,
    active: bool,
    queued: usize,
    scan_limit: usize,
) -> Result<Vec<Occurrence>, String> {
    Ok(due_with_missed(schedule, id, from, now, policy, active, queued, scan_limit)?.selected)
}
/// `due`, plus every occurrence passed over (as spans) and the next cursor. `queued` counts
/// unfinished runs beyond the first. Latest-style catch-up jumps straight to the newest due
/// occurrence, so a long sleep costs nothing; `CatchUp::All` starts the oldest ones that fit and
/// leaves the rest due.
#[allow(clippy::too_many_arguments)]
pub fn due_with_missed(
    schedule: &Schedule,
    id: &str,
    from: DateTime<Utc>,
    now: DateTime<Utc>,
    policy: Policy,
    active: bool,
    queued: usize,
    scan_limit: usize,
) -> Result<Due, String> {
    if id.is_empty() || scan_limit == 0 {
        return Err("invalid schedule identity or scan limit".into());
    }
    let pending = usize::from(active) + queued;
    let capacity = match policy.overlap {
        Overlap::Skip if active => 0,
        Overlap::Skip => 1,
        Overlap::Queue { max_queued } => max_queued
            .saturating_sub(queued)
            .saturating_add(usize::from(!active)),
        // The buffered slot is refilled with the newest; the store supersedes the old one.
        Overlap::BufferLatest => 1,
        Overlap::Parallel { max_active } => max_active.saturating_sub(pending),
    };
    let occurrence = |at| Occurrence {
        schedule_id: id.into(),
        scheduled_at: at,
    };
    let cutoff = now
        .checked_sub_signed(Duration::seconds(i64::from(policy.freshness_seconds)))
        .ok_or("freshness overflow")?;
    let mut missed = Vec::new();
    if from < cutoff {
        missed.extend(schedule.span(from, cutoff, MISSED_STALE)?);
    }
    let cursor = from.max(cutoff);
    let after_now = after(now)?;
    let mut selected = Vec::new();
    if capacity == 0 {
        // A full queue keeps its fresh occurrences due for a later tick instead of losing them.
        if matches!(policy.overlap, Overlap::Queue { .. }) {
            return Ok(Due {
                selected,
                missed,
                next_from: cursor,
            });
        }
        missed.extend(schedule.span(cursor, after_now, MISSED_OVERLAP)?);
        return Ok(Due {
            selected,
            missed,
            next_from: after_now,
        });
    }
    match (policy.overlap, policy.catch_up) {
        (Overlap::Skip | Overlap::BufferLatest, _) | (_, CatchUp::Latest) => {
            if let Some(at) = schedule
                .latest_at_or_before(now)?
                .filter(|at| *at >= cursor)
            {
                missed.extend(schedule.span(cursor, at, MISSED_SUPERSEDED)?);
                selected.push(occurrence(at));
            }
            Ok(Due {
                selected,
                missed,
                next_from: after_now,
            })
        }
        (_, CatchUp::All { max }) => {
            let limit = max.min(capacity).min(scan_limit);
            let mut next_from = cursor;
            while selected.len() < limit {
                let Some(at) = schedule
                    .next_at_or_after(next_from)?
                    .filter(|at| *at <= now)
                else {
                    next_from = after_now;
                    break;
                };
                selected.push(occurrence(at));
                next_from = after(at)?;
            }
            Ok(Due {
                selected,
                missed,
                next_from,
            })
        }
    }
}

use crate::{
    store::SqliteStore,
    types::{RunState, WorkflowDefinition},
};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
fn db_error(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn sql_integer(value: u64) -> Result<i64, String> {
    i64::try_from(value).map_err(|_| "integer exceeds SQLite range".into())
}
fn unsigned_integer(value: i64) -> Result<u64, String> {
    u64::try_from(value).map_err(|_| "negative SQLite integer".into())
}
/// Schema migration runs once under an exclusive SQLite write transaction.
pub(crate) fn migrate(db: &Connection) -> Result<(), String> {
    db.execute_batch("BEGIN IMMEDIATE;").map_err(db_error)?;
    let result = (|| {
        let version: u32 = db
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .map_err(db_error)?;
        if version > crate::store::STORE_VERSION {
            return Err(format!("unsupported store version {version}"));
        }
        if version >= 2 {
            return Ok(());
        }
        db.execute_batch(
            "CREATE TABLE schedules(id TEXT PRIMARY KEY, revision INTEGER NOT NULL, schedule TEXT NOT NULL, policy TEXT NOT NULL, definition TEXT NOT NULL, next_due TEXT);
      CREATE TABLE occurrences(schedule_id TEXT NOT NULL REFERENCES schedules(id), scheduled_at TEXT NOT NULL, run_id TEXT NOT NULL UNIQUE REFERENCES runs(id), PRIMARY KEY(schedule_id,scheduled_at));
      CREATE TABLE task_outbox(id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE watches(id TEXT PRIMARY KEY, state TEXT NOT NULL);
      CREATE TABLE admission_roots(id TEXT PRIMARY KEY, budget INTEGER NOT NULL, spent INTEGER NOT NULL, reserved INTEGER NOT NULL);
      CREATE TABLE admission_permits(id TEXT PRIMARY KEY, root_id TEXT NOT NULL REFERENCES admission_roots(id), reserved INTEGER NOT NULL);
      CREATE TABLE children(parent_id TEXT NOT NULL, child_id TEXT NOT NULL UNIQUE, PRIMARY KEY(parent_id,child_id), FOREIGN KEY(parent_id) REFERENCES runs(id), FOREIGN KEY(child_id) REFERENCES runs(id));
      CREATE TABLE child_messages(parent_id TEXT NOT NULL, child_id TEXT NOT NULL, message_id TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY(parent_id,child_id,message_id), FOREIGN KEY(parent_id,child_id) REFERENCES children(parent_id,child_id));
      PRAGMA user_version=2;",
        )
        .map_err(db_error)
    })();
    match result {
        Ok(()) => db.execute_batch("COMMIT;").map_err(db_error),
        Err(e) => {
            let _ = db.execute_batch("ROLLBACK;");
            Err(e)
        }
    }
}
/// Upgrade old schedules without changing their previously active behavior.
pub(crate) fn migrate_v4(db: &Connection) -> Result<(), String> {
    db.execute_batch("BEGIN IMMEDIATE;").map_err(db_error)?;
    let result = (|| {
        let version: u32 = db
            .pragma_query_value(None, "user_version", |r| r.get(0))
            .map_err(db_error)?;
        if version > crate::store::STORE_VERSION {
            return Err(format!("unsupported store version {version}"));
        }
        if version < 3 {
            return Err("previous migration missing".into());
        }
        if version == 3 {
            db.execute_batch(
                "ALTER TABLE schedules ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1; CREATE TABLE artifact_pins(run_id TEXT NOT NULL REFERENCES runs(id), digest TEXT NOT NULL, kind TEXT NOT NULL, PRIMARY KEY(run_id,digest,kind)); PRAGMA user_version=4;",
            )
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScheduleRecord {
    pub id: String,
    pub revision: u64,
    pub schedule: Schedule,
    pub policy: Policy,
    pub definition: WorkflowDefinition,
    pub next_due: Option<DateTime<Utc>>,
    pub enabled: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PumpResult {
    pub schedule_id: String,
    pub outcome: Result<Vec<Occurrence>, String>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutboxEntry {
    pub id: String,
    pub kind: String,
    pub payload: String,
}
impl SqliteStore {
    /// Saving a schedule is explicit; callers decide whether and when to pump it.
    pub fn create_schedule(
        &mut self,
        id: &str,
        schedule: &Schedule,
        policy: Policy,
        definition: &WorkflowDefinition,
        from: DateTime<Utc>,
    ) -> Result<(), String> {
        if id.is_empty() {
            return Err("empty schedule id".into());
        }
        definition.validate()?;
        policy.validate()?;
        let next = schedule
            .next_at_or_after(from)?
            .ok_or("schedule has no occurrence")?;
        self.connection
            .execute(
                "INSERT INTO schedules(id,revision,schedule,policy,definition,next_due,enabled) VALUES(?1,0,?2,?3,?4,?5,0)",
                params![
                    id,
                    serde_json::to_string(schedule).map_err(db_error)?,
                    serde_json::to_string(&policy).map_err(db_error)?,
                    serde_json::to_string(definition).map_err(db_error)?,
                    next.to_rfc3339()
                ],
            )
            .map_err(db_error)?;
        Ok(())
    }
    /// Install a validated activity template with the schedule in one transaction.
    pub fn create_schedule_with_plan(
        &mut self,
        id: &str,
        schedule: &Schedule,
        policy: Policy,
        definition: &WorkflowDefinition,
        plan: &crate::execution::ExecutionPlan,
        from: DateTime<Utc>,
    ) -> Result<(), String> {
        if id.is_empty() {
            return Err("empty schedule id".into());
        }
        definition.validate()?;
        policy.validate()?;
        crate::execution::validate_plan_for_definition(definition, plan)?;
        let next = schedule
            .next_at_or_after(from)?
            .ok_or("schedule has no occurrence")?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        tx.execute(
            "INSERT INTO schedules(id,revision,schedule,policy,definition,next_due,enabled) VALUES(?1,0,?2,?3,?4,?5,0)",
            params![
                id,
                serde_json::to_string(schedule).map_err(db_error)?,
                serde_json::to_string(&policy).map_err(db_error)?,
                serde_json::to_string(definition).map_err(db_error)?,
                next.to_rfc3339()
            ],
        )
        .map_err(db_error)?;
        tx.execute(
            "INSERT INTO schedule_plans(schedule_id,plan) VALUES(?1,?2)",
            params![id, serde_json::to_string(plan).map_err(db_error)?],
        )
        .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(())
    }
    /// Replace a schedule's rule, policy, goal and steps. Runs it already started keep the steps
    /// they started with; the new rule counts from `from`, so nothing between the last tick and
    /// the edit is owed. Returns the new revision.
    #[allow(clippy::too_many_arguments)]
    pub fn update_schedule_with_plan(
        &mut self,
        id: &str,
        expected_revision: u64,
        schedule: &Schedule,
        policy: Policy,
        definition: &WorkflowDefinition,
        plan: &crate::execution::ExecutionPlan,
        from: DateTime<Utc>,
    ) -> Result<u64, String> {
        definition.validate()?;
        policy.validate()?;
        crate::execution::validate_plan_for_definition(definition, plan)?;
        let next = schedule
            .next_at_or_after(from)?
            .ok_or("schedule has no occurrence")?;
        let revision = expected_revision
            .checked_add(1)
            .ok_or("schedule revision overflow")?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let changed = tx
            .execute(
                "UPDATE schedules SET revision=?1,schedule=?2,policy=?3,definition=?4,next_due=?5 WHERE id=?6 AND revision=?7",
                params![
                    sql_integer(revision)?,
                    serde_json::to_string(schedule).map_err(db_error)?,
                    serde_json::to_string(&policy).map_err(db_error)?,
                    serde_json::to_string(definition).map_err(db_error)?,
                    next.to_rfc3339(),
                    id,
                    sql_integer(expected_revision)?
                ],
            )
            .map_err(db_error)?;
        if changed != 1 {
            return Err("unknown schedule or stale revision".into());
        }
        tx.execute(
            "INSERT INTO schedule_plans(schedule_id,plan) VALUES(?1,?2) ON CONFLICT(schedule_id) DO UPDATE SET plan=excluded.plan",
            params![id, serde_json::to_string(plan).map_err(db_error)?],
        )
        .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(revision)
    }
    /// A tick atomically claims a schedule revision, creates stable run identities,
    /// writes start outbox entries and advances the cursor. Repeated ticks are safe.
    pub fn pump_schedule(
        &mut self,
        id: &str,
        expected_revision: u64,
        now: DateTime<Utc>,
        scan_limit: usize,
    ) -> Result<Vec<Occurrence>, String> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        #[allow(clippy::type_complexity)]
        let row: (i64, String, String, String, Option<String>, bool, Option<String>) = tx
            .query_row(
                "SELECT revision,schedule,policy,definition,next_due,enabled,pumped_at FROM schedules WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?)),
            )
            .optional()
            .map_err(db_error)?
            .ok_or("unknown schedule")?;
        if unsigned_integer(row.0)? != expected_revision {
            return Err("stale schedule revision".into());
        }
        if !row.5 {
            return Err("schedule is disabled".into());
        }
        let schedule: Schedule = serde_json::from_str(&row.1).map_err(db_error)?;
        let policy: Policy = serde_json::from_str(&row.2).map_err(db_error)?;
        let definition: WorkflowDefinition = serde_json::from_str(&row.3).map_err(db_error)?;
        let Some(next_due) = row.4 else {
            tx.commit().map_err(db_error)?;
            return Ok(Vec::new());
        };
        let stored = parse_time(&next_due)?;
        let mut cursor = stored;
        // The wall clock ran back past the last pump (a bad RTC, a manual change, a VM restore):
        // the cursor sits in a false future, so resume from now instead of staying silent.
        let rewound = row
            .6
            .as_deref()
            .map(parse_time)
            .transpose()?
            .is_some_and(|at| at > now);
        if rewound {
            if let Some(at) = schedule.next_at_or_after(now)? {
                cursor = cursor.min(at);
            }
            forget_misses_from(&tx, &schedule, id, cursor)?;
        }
        if cursor > now {
            if rewound {
                advance(&tx, id, expected_revision, Some(cursor), now)?;
            }
            tx.commit().map_err(db_error)?;
            return Ok(Vec::new());
        }
        let pending = unfinished_occurrences(&tx, id, None)?;
        let (active, queued) = (!pending.is_empty(), pending.len().saturating_sub(1));
        let due = due_with_missed(
            &schedule, id, cursor, now, policy, active, queued, scan_limit,
        )?;
        // A window computed from the cursor can reach over occurrences that ran before the clock
        // stepped back; a superseded one added below has its row and is missed all the same.
        let mut missed = without_occurrences(&tx, &schedule, id, due.missed)?;
        // An occurrence created before the clock stepped back already has its run.
        let mut selected = Vec::new();
        for occurrence in due.selected {
            let exists: bool = tx
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM occurrences WHERE schedule_id=?1 AND scheduled_at=?2)",
                    params![id, stamp(occurrence.scheduled_at)],
                    |r| r.get(0),
                )
                .map_err(db_error)?;
            if !exists {
                selected.push(occurrence);
            }
        }
        if policy.overlap == Overlap::BufferLatest && !selected.is_empty() {
            // The newest occurrence replaces every buffered one. Started means an attempt began
            // (the turn check keeps later ones from starting), not a changed revision: Pause and
            // Resume, or planning, change the revision of a run that is still waiting its turn.
            for (at, buffered) in &pending {
                if buffered.last_epoch == 0 && buffered.attempt.is_none() {
                    supersede(&tx, buffered)?;
                    let at = parse_time(at)?;
                    missed.push(Missed {
                        first: at,
                        last: at,
                        count: 1,
                        reason: MISSED_SUPERSEDED.into(),
                    });
                }
            }
        }
        record_missed(&tx, &schedule, id, missed)?;
        for occurrence in &selected {
            let stamp = stamp(occurrence.scheduled_at);
            let run_id = occurrence_run_id(id, occurrence.scheduled_at);
            let state = RunState::new(run_id.clone(), definition.clone())?;
            tx.execute(
                "INSERT INTO runs(id,revision,state) VALUES(?1,0,?2)",
                params![run_id, serde_json::to_string(&state).map_err(db_error)?],
            )
            .map_err(db_error)?;
            let plan: Option<String> = tx
                .query_row(
                    "SELECT plan FROM schedule_plans WHERE schedule_id=?1",
                    [id],
                    |r| r.get(0),
                )
                .optional()
                .map_err(db_error)?;
            if let Some(plan) = plan {
                let decoded: crate::execution::ExecutionPlan =
                    serde_json::from_str(&plan).map_err(db_error)?;
                crate::execution::validate_plan_for_definition(&definition, &decoded)?;
                tx.execute(
                    "INSERT INTO execution_plans(run_id,plan) VALUES(?1,?2)",
                    params![run_id, plan],
                )
                .map_err(db_error)?;
            }
            tx.execute(
                "INSERT INTO occurrences(schedule_id,scheduled_at,run_id) VALUES(?1,?2,?3)",
                params![id, stamp, run_id],
            )
            .map_err(db_error)?;
            tx.execute(
                "INSERT INTO task_outbox(id,kind,payload) VALUES(?1,'start',?2)",
                params![format!("start:{run_id}"), run_id],
            )
            .map_err(db_error)?;
        }
        // NULL records an exhausted one-time schedule.
        let next = schedule.next_at_or_after(due.next_from)?;
        // A full queue that keeps its cursor leaves the revision alone, so client edits never race
        // a tick that changed nothing.
        if rewound || next != Some(stored) {
            advance(&tx, id, expected_revision, next, now)?;
        }
        tx.commit().map_err(db_error)?;
        Ok(selected)
    }
    /// Tick every due schedule; each result is isolated so an invalid schedule
    /// cannot prevent another schedule from being processed. `now` is injected.
    pub fn pump_due(
        &mut self,
        now: DateTime<Utc>,
        scan_limit: usize,
    ) -> Result<Vec<PumpResult>, String> {
        let due = {
            let mut stmt = self
                .connection
                .prepare(
                    "SELECT id,revision,next_due,pumped_at FROM schedules WHERE enabled=1 ORDER BY id",
                )
                .map_err(db_error)?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, Option<String>>(2)?,
                        r.get::<_, Option<String>>(3)?,
                    ))
                })
                .map_err(db_error)?;
            let mut due = Vec::new();
            for row in rows {
                let (id, revision, next, pumped) = row.map_err(db_error)?;
                let next_due = next.as_deref().map(parse_time).transpose()?;
                let rewound = pumped.as_deref().map(parse_time).transpose()?;
                if next_due.is_some_and(|at| at <= now) || rewound.is_some_and(|at| at > now) {
                    due.push((id, unsigned_integer(revision)?));
                }
            }
            due
        };
        Ok(due
            .into_iter()
            .map(|(id, revision)| {
                let outcome = self.pump_schedule(&id, revision, now, scan_limit);
                PumpResult {
                    schedule_id: id,
                    outcome,
                }
            })
            .collect())
    }
    pub fn list_schedules(&self) -> Result<Vec<ScheduleRecord>, String> {
        let mut stmt = self
            .connection
            .prepare(
                "SELECT id,revision,schedule,policy,definition,next_due,enabled FROM schedules ORDER BY id",
            )
            .map_err(db_error)?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, Option<String>>(5)?,
                    r.get::<_, bool>(6)?,
                ))
            })
            .map_err(db_error)?;
        rows.map(|row| {
            let (id, revision, schedule, policy, definition, next, enabled) =
                row.map_err(db_error)?;
            let next_due = next
                .map(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .map(|d| d.with_timezone(&Utc))
                        .map_err(db_error)
                })
                .transpose()?;
            Ok(ScheduleRecord {
                id,
                revision: unsigned_integer(revision)?,
                schedule: serde_json::from_str(&schedule).map_err(db_error)?,
                policy: serde_json::from_str(&policy).map_err(db_error)?,
                definition: serde_json::from_str(&definition).map_err(db_error)?,
                next_due,
                enabled,
            })
        })
        .collect()
    }
    /// Turns a schedule on or off at `now`. Turned on, its next due is not before `now`: an
    /// occurrence that fell while it was off is no run, as the calendar shows it, rather than one
    /// started the moment it is turned on (a daily 7:45 turned on at 9:00 ran at 9:00). A
    /// schedule with no occurrence left keeps its own, for its catch-up policy to take.
    pub fn enable_schedule(
        &mut self,
        id: &str,
        expected_revision: u64,
        enabled: bool,
        now: DateTime<Utc>,
    ) -> Result<u64, String> {
        let next = expected_revision
            .checked_add(1)
            .ok_or("schedule revision overflow")?;
        let row: Option<(String, Option<String>)> = self
            .connection
            .query_row(
                "SELECT schedule,next_due FROM schedules WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(db_error)?;
        let mut due = row.as_ref().and_then(|r| r.1.clone());
        if let Some((schedule, Some(stored))) = row.filter(|_| enabled)
            && parse_time(&stored)? < now
        {
            let schedule: Schedule = serde_json::from_str(&schedule).map_err(db_error)?;
            if let Some(at) = schedule.next_at_or_after(now)? {
                due = Some(at.to_rfc3339());
            }
        }
        let changed = self
            .connection
            .execute(
                "UPDATE schedules SET enabled=?1,revision=?2,next_due=?3 WHERE id=?4 AND revision=?5",
                params![
                    enabled,
                    sql_integer(next)?,
                    due,
                    id,
                    sql_integer(expected_revision)?
                ],
            )
            .map_err(db_error)?;
        if changed != 1 {
            return Err("unknown schedule or stale revision".into());
        }
        Ok(next)
    }
    /// Deletes a schedule, fenced by the revision the user saw: its rule, plan, occurrences and
    /// missed spans go, so it starts nothing again. The runs it started stay as runs of their own,
    /// which `delete_run` takes once settled. Refused while one of them is unfinished: a queued
    /// occurrence still waits on the schedule for its turn (`ensure_occurrence_turn`).
    pub fn delete_schedule(&mut self, id: &str, expected_revision: u64) -> Result<(), String> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let revision: Option<i64> = tx
            .query_row("SELECT revision FROM schedules WHERE id=?1", [id], |r| {
                r.get(0)
            })
            .optional()
            .map_err(db_error)?;
        if revision.map(unsigned_integer).transpose()? != Some(expected_revision) {
            return Err("unknown schedule or stale revision".into());
        }
        if !unfinished_occurrences(&tx, id, None)?.is_empty() {
            return Err("a run of this schedule is still unfinished".into());
        }
        for table in ["occurrences", "missed_occurrences", "schedule_plans"] {
            tx.execute(&format!("DELETE FROM {table} WHERE schedule_id=?1"), [id])
                .map_err(db_error)?;
        }
        tx.execute("DELETE FROM schedules WHERE id=?1", [id])
            .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(())
    }
    pub fn schedule_revision(&self, id: &str) -> Result<Option<u64>, String> {
        let revision: Option<i64> = self
            .connection
            .query_row("SELECT revision FROM schedules WHERE id=?1", [id], |r| {
                r.get(0)
            })
            .optional()
            .map_err(db_error)?;
        revision.map(unsigned_integer).transpose()
    }
    /// Distinguish ordinary runs from scheduled occurrences awaiting a claim.
    pub fn scheduled_start_for_run(&self, run_id: &str) -> Result<Option<(String, bool)>, String> {
        self.connection
            .query_row(
                "SELECT id,delivered FROM task_outbox WHERE kind='start' AND payload=?1",
                [run_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(db_error)
    }
    pub fn pending_outbox(&self, limit: u32) -> Result<Vec<OutboxEntry>, String> {
        let mut stmt = self
            .connection
            .prepare(
                "SELECT id,kind,payload FROM task_outbox WHERE delivered=0 ORDER BY id LIMIT ?1",
            )
            .map_err(db_error)?;
        let rows = stmt
            .query_map([limit.min(1000)], |r| {
                Ok(OutboxEntry {
                    id: r.get(0)?,
                    kind: r.get(1)?,
                    payload: r.get(2)?,
                })
            })
            .map_err(db_error)?;
        rows.map(|r| r.map_err(db_error)).collect()
    }
    pub fn acknowledge_outbox(&mut self, id: &str) -> Result<(), String> {
        if self
            .connection
            .execute(
                "UPDATE task_outbox SET delivered=1 WHERE id=?1 AND kind != 'start' AND kind != 'notification'",
                [id],
            )
            .map_err(db_error)?
            != 1
        {
            return Err("unknown outbox entry".into());
        }
        Ok(())
    }
}

impl SqliteStore {
    /// Durably claim a start intent and fence it with the corresponding attempt.
    /// A crash after commit leaves the attempt for normal lease recovery; it must
    /// not be blindly reissued as a new external action.
    pub fn claim_scheduled_start(
        &mut self,
        outbox_id: &str,
        command: &crate::types::CommandEnvelope,
    ) -> Result<crate::types::CommandReply, String> {
        self.claim_scheduled_start_once(outbox_id, command)
            .map(|(reply, _)| reply)
    }
    /// Spans of occurrences this schedule passed over, newest first, with the reason.
    pub fn missed_occurrences(&self, id: &str, limit: u32) -> Result<Vec<Missed>, String> {
        let mut stmt = self
            .connection
            .prepare("SELECT first_at,last_at,count,reason FROM missed_occurrences WHERE schedule_id=?1 ORDER BY first_at DESC LIMIT ?2")
            .map_err(db_error)?;
        let rows = stmt
            .query_map(params![id, limit], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })
            .map_err(db_error)?;
        rows.map(|row| {
            let (first, last, count, reason) = row.map_err(db_error)?;
            Ok(Missed {
                first: parse_time(&first)?,
                last: parse_time(&last)?,
                count: unsigned_integer(count)?,
                reason,
            })
        })
        .collect()
    }
    pub fn claim_scheduled_start_once(
        &mut self,
        outbox_id: &str,
        command: &crate::types::CommandEnvelope,
    ) -> Result<(crate::types::CommandReply, bool), String> {
        use crate::{
            reducer,
            types::{Command, RunState, SCHEMA_VERSION},
        };
        if !matches!(&command.command, Command::StartAttempt { .. }) {
            return Err("outbox claim requires an attempt start".into());
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let (kind, run_id): (String, String) = tx
            .query_row(
                "SELECT kind,payload FROM task_outbox WHERE id=?1",
                [outbox_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(db_error)?
            .ok_or("unknown outbox entry")?;
        if kind != "start" {
            return Err("outbox entry is not a start".into());
        }
        let delivered: bool = tx
            .query_row(
                "SELECT delivered FROM task_outbox WHERE id=?1",
                [outbox_id],
                |r| r.get(0),
            )
            .map_err(db_error)?;
        let request = serde_json::to_string(command).map_err(db_error)?;
        let previous: Option<(String, String)> = tx
            .query_row(
                "SELECT request,reply FROM commands WHERE run_id=?1 AND id=?2",
                params![run_id, command.id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(db_error)?;
        if let Some((prior, reply)) = previous {
            if prior != request || !delivered {
                return Err("command id reused with different request or outbox".into());
            }
            let result = serde_json::from_str(&reply).map_err(db_error)?;
            tx.commit().map_err(db_error)?;
            return Ok((result, false));
        }
        if delivered {
            return Err("start outbox already claimed".into());
        }
        crate::store::ensure_root_dispatchable(&tx, &run_id)?;
        ensure_occurrence_turn(&tx, &run_id)?;
        let json: String = tx
            .query_row("SELECT state FROM runs WHERE id=?1", [&run_id], |r| {
                r.get(0)
            })
            .map_err(db_error)?;
        let state: RunState = serde_json::from_str(&json).map_err(db_error)?;
        if state.version != SCHEMA_VERSION {
            return Err("unsupported run version".into());
        }
        state.definition.validate()?;
        let (next, event, reply) = reducer::apply(&state, command)?;
        tx.execute(
            "UPDATE runs SET revision=?1,state=?2 WHERE id=?3 AND revision=?4",
            params![
                sql_integer(next.revision)?,
                serde_json::to_string(&next).map_err(db_error)?,
                run_id,
                sql_integer(state.revision)?
            ],
        )
        .map_err(db_error)?;
        tx.execute(
            "INSERT INTO events(run_id,revision,event) VALUES(?1,?2,?3)",
            params![
                run_id,
                sql_integer(event.revision)?,
                serde_json::to_string(&event).map_err(db_error)?
            ],
        )
        .map_err(db_error)?;
        tx.execute(
            "INSERT INTO commands(run_id,id,request,reply) VALUES(?1,?2,?3,?4)",
            params![
                run_id,
                command.id,
                request,
                serde_json::to_string(&reply).map_err(db_error)?
            ],
        )
        .map_err(db_error)?;
        tx.execute(
            "UPDATE task_outbox SET delivered=1 WHERE id=?1",
            [outbox_id],
        )
        .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok((reply, true))
    }
}

/// Queued and buffered occurrences start one at a time, oldest first: refuse a start while
/// an earlier occurrence of the same schedule is unfinished. Other runs pass.
pub(crate) fn ensure_occurrence_turn(tx: &Connection, run_id: &str) -> Result<(), String> {
    let Some((schedule_id, at, policy)): Option<(String, String, String)> = tx
        .query_row(
            "SELECT o.schedule_id,o.scheduled_at,s.policy FROM occurrences o JOIN schedules s ON s.id=o.schedule_id WHERE o.run_id=?1",
            [run_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()
        .map_err(db_error)?
    else {
        return Ok(());
    };
    let policy: Policy = serde_json::from_str(&policy).map_err(db_error)?;
    if !policy.overlap.serial() || unfinished_occurrences(tx, &schedule_id, Some(&at))?.is_empty() {
        return Ok(());
    }
    Err("an earlier occurrence of this schedule is still unfinished".into())
}
/// Unfinished runs of a schedule, oldest first, before `before` when given. A finished run never
/// changes again, so it is flagged once and the partial index skips it on every later scan.
fn unfinished_occurrences(
    tx: &Connection,
    schedule_id: &str,
    before: Option<&str>,
) -> Result<Vec<(String, RunState)>, String> {
    let rows: Vec<(String, String, String)> = {
        let mut stmt = tx
            .prepare("SELECT o.scheduled_at,o.run_id,r.state FROM occurrences o JOIN runs r ON r.id=o.run_id WHERE o.schedule_id=?1 AND o.finished=0 AND (?2 IS NULL OR o.scheduled_at<?2) ORDER BY o.scheduled_at")
            .map_err(db_error)?;
        let rows = stmt
            .query_map(params![schedule_id, before], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .map_err(db_error)?;
        rows.collect::<Result<_, _>>().map_err(db_error)?
    };
    let mut pending = Vec::new();
    for (at, run_id, state) in rows {
        let state: RunState = serde_json::from_str(&state).map_err(db_error)?;
        if state.status.terminal() {
            tx.execute(
                "UPDATE occurrences SET finished=1 WHERE run_id=?1",
                [run_id],
            )
            .map_err(db_error)?;
        } else {
            pending.push((at, state));
        }
    }
    Ok(pending)
}
fn parse_time(value: &str) -> Result<DateTime<Utc>, String> {
    Ok(DateTime::parse_from_rfc3339(value)
        .map_err(db_error)?
        .with_timezone(&Utc))
}
/// The run a schedule starts for one occurrence; [`run_origin`] reads it back.
pub fn occurrence_run_id(schedule_id: &str, at: DateTime<Utc>) -> String {
    format!("schedule:{}:{schedule_id}:{}", schedule_id.len(), stamp(at))
}
/// The schedule and occurrence a run was started for; None for a run made any other way.
pub fn run_origin(run_id: &str) -> Option<(String, DateTime<Utc>)> {
    let (len, rest) = run_id.strip_prefix("schedule:")?.split_once(':')?;
    let len: usize = len.parse().ok()?;
    let at = rest.get(len..)?.strip_prefix(':')?;
    Some((rest.get(..len)?.to_owned(), parse_time(at).ok()?))
}
/// Fixed-width, so stored instants compare as text.
fn stamp(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}
/// Move the cursor and remember when, so a clock that later runs back is noticed.
fn advance(
    tx: &Connection,
    id: &str,
    expected_revision: u64,
    next: Option<DateTime<Utc>>,
    now: DateTime<Utc>,
) -> Result<(), String> {
    let revision = expected_revision
        .checked_add(1)
        .ok_or("schedule revision overflow")?;
    tx.execute(
        "UPDATE schedules SET revision=?1,next_due=?2,pumped_at=?3 WHERE id=?4 AND revision=?5",
        params![
            sql_integer(revision)?,
            next.map(|at| at.to_rfc3339()),
            stamp(now),
            id,
            sql_integer(expected_revision)?
        ],
    )
    .map_err(db_error)?;
    Ok(())
}
/// How many missed spans a schedule keeps; older ones are dropped.
pub const MISSED_RETAINED: u32 = 256;
/// Store spans oldest first, extending the newest row when a span continues it with the same
/// reason, so a schedule stuck behind one run keeps one growing row instead of one per tick.
fn record_missed(
    tx: &Connection,
    schedule: &Schedule,
    id: &str,
    mut missed: Vec<Missed>,
) -> Result<(), String> {
    if missed.is_empty() {
        return Ok(());
    }
    missed.sort_by_key(|span| span.first);
    for span in missed {
        let newest: Option<(String, String, String)> = tx
            .query_row(
                "SELECT first_at,last_at,reason FROM missed_occurrences WHERE schedule_id=?1 ORDER BY first_at DESC LIMIT 1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()
            .map_err(db_error)?;
        let continued = match newest {
            Some((first, last, reason)) if reason == span.reason => schedule
                .next_at_or_after(after(parse_time(&last)?)?)?
                .is_some_and(|at| at == span.first)
                .then_some(first),
            _ => None,
        };
        let count = sql_integer(span.count)?;
        if let Some(first) = continued {
            tx.execute(
                "UPDATE missed_occurrences SET last_at=?1,count=count+?2 WHERE schedule_id=?3 AND first_at=?4",
                params![stamp(span.last), count, id, first],
            )
            .map_err(db_error)?;
        } else {
            tx.execute(
                "INSERT OR IGNORE INTO missed_occurrences(schedule_id,first_at,last_at,count,reason) VALUES(?1,?2,?3,?4,?5)",
                params![id, stamp(span.first), stamp(span.last), count, span.reason],
            )
            .map_err(db_error)?;
        }
    }
    tx.execute(
        "DELETE FROM missed_occurrences WHERE schedule_id=?1 AND first_at NOT IN (SELECT first_at FROM missed_occurrences WHERE schedule_id=?1 ORDER BY first_at DESC LIMIT ?2)",
        params![id, MISSED_RETAINED],
    )
    .map_err(db_error)?;
    Ok(())
}
/// `missed` with every instant that already has an occurrence cut out, a span split around it.
fn without_occurrences(
    tx: &Connection,
    schedule: &Schedule,
    id: &str,
    missed: Vec<Missed>,
) -> Result<Vec<Missed>, String> {
    let mut kept = Vec::new();
    for span in missed {
        let mut statement = tx
            .prepare("SELECT scheduled_at FROM occurrences WHERE schedule_id=?1 AND scheduled_at>=?2 AND scheduled_at<=?3 ORDER BY scheduled_at")
            .map_err(db_error)?;
        let ran = statement
            .query_map(params![id, stamp(span.first), stamp(span.last)], |r| {
                r.get::<_, String>(0)
            })
            .map_err(db_error)?
            .map(|at| parse_time(&at.map_err(db_error)?))
            .collect::<Result<Vec<_>, String>>()?;
        if ran.is_empty() {
            kept.push(span);
            continue;
        }
        let mut from = span.first;
        for at in ran {
            kept.extend(schedule.span(from, at, &span.reason)?);
            from = after(at)?;
        }
        kept.extend(schedule.span(from, after(span.last)?, &span.reason)?);
    }
    Ok(kept)
}
/// Rows written while the clock ran ahead name instants that are due again from the rewound
/// cursor `from`: drop them, and cut a span that reaches `from` back to end before it.
fn forget_misses_from(
    tx: &Connection,
    schedule: &Schedule,
    id: &str,
    from: DateTime<Utc>,
) -> Result<(), String> {
    tx.execute(
        "DELETE FROM missed_occurrences WHERE schedule_id=?1 AND first_at>=?2",
        params![id, stamp(from)],
    )
    .map_err(db_error)?;
    let straddling: Option<(String, String)> = tx
        .query_row(
            "SELECT first_at,reason FROM missed_occurrences WHERE schedule_id=?1 AND last_at>=?2",
            params![id, stamp(from)],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .map_err(db_error)?;
    if let Some((first, reason)) = straddling {
        let span = schedule
            .span(parse_time(&first)?, from, &reason)?
            .ok_or("missed span has no occurrence")?;
        tx.execute(
            "UPDATE missed_occurrences SET last_at=?1,count=?2 WHERE schedule_id=?3 AND first_at=?4",
            params![stamp(span.last), sql_integer(span.count)?, id, first],
        )
        .map_err(db_error)?;
    }
    Ok(())
}
/// Cancel a buffered occurrence that never started, recording its command and event and
/// retiring its start in the same transaction.
fn supersede(tx: &Connection, buffered: &RunState) -> Result<(), String> {
    let command = crate::types::CommandEnvelope {
        id: format!("superseded:{}", buffered.id),
        expected_revision: buffered.revision,
        command: crate::types::Command::Cancel,
    };
    let (mut next, event, reply) = crate::reducer::apply(buffered, &command)?;
    next.result = Some(MISSED_SUPERSEDED.into());
    tx.execute(
        "UPDATE runs SET revision=?1,state=?2 WHERE id=?3 AND revision=?4",
        params![
            sql_integer(next.revision)?,
            serde_json::to_string(&next).map_err(db_error)?,
            buffered.id,
            sql_integer(buffered.revision)?
        ],
    )
    .map_err(db_error)?;
    tx.execute(
        "INSERT INTO events(run_id,revision,event) VALUES(?1,?2,?3)",
        params![
            buffered.id,
            sql_integer(event.revision)?,
            serde_json::to_string(&event).map_err(db_error)?
        ],
    )
    .map_err(db_error)?;
    tx.execute(
        "INSERT INTO commands(run_id,id,request,reply) VALUES(?1,?2,?3,?4)",
        params![
            buffered.id,
            command.id,
            serde_json::to_string(&command).map_err(db_error)?,
            serde_json::to_string(&reply).map_err(db_error)?
        ],
    )
    .map_err(db_error)?;
    tx.execute(
        "UPDATE task_outbox SET delivered=1 WHERE id=?1",
        [format!("start:{}", buffered.id)],
    )
    .map_err(db_error)?;
    tx.execute(
        "UPDATE occurrences SET finished=1 WHERE run_id=?1",
        [&buffered.id],
    )
    .map_err(db_error)?;
    Ok(())
}
