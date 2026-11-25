//! A failed observation is unknown, not a negative predicate result.
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Truth {
    True,
    False,
    Unknown,
}
impl Truth {
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::False, _) | (_, Self::False) => Self::False,
            (Self::True, Self::True) => Self::True,
            _ => Self::Unknown,
        }
    }
    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, _) | (_, Self::True) => Self::True,
            (Self::False, Self::False) => Self::False,
            _ => Self::Unknown,
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WatchState {
    pub version: u64,
    pub cursor: Option<String>,
    pub last_success: Option<bool>,
}
impl WatchState {
    /// Compare-and-swap must be committed atomically with the watch outcome.
    /// Unknown/partial observations never advance the successful cursor.
    pub fn update(
        &mut self,
        expected: u64,
        value: Truth,
        successful_cursor: Option<String>,
    ) -> Result<(), String> {
        if expected != self.version {
            return Err("stale watch version".into());
        }
        let next = self
            .version
            .checked_add(1)
            .ok_or("watch version overflow")?;
        if value != Truth::Unknown {
            self.cursor = successful_cursor;
            self.last_success = Some(value == Truth::True);
        }
        self.version = next;
        Ok(())
    }
}

use crate::store::SqliteStore;
use chrono::{DateTime, Duration, Utc};
use rusqlite::{OptionalExtension, TransactionBehavior, params};
fn db_error(e: impl std::fmt::Display) -> String {
    e.to_string()
}
impl SqliteStore {
    pub fn create_watch(&mut self, id: &str) -> Result<(), String> {
        if id.is_empty() {
            return Err("empty watch id".into());
        }
        self.connection
            .execute(
                "INSERT INTO watches(id,state) VALUES(?1,?2)",
                params![
                    id,
                    serde_json::to_string(&WatchState::default()).map_err(db_error)?
                ],
            )
            .map_err(db_error)?;
        Ok(())
    }
    pub fn watch(&self, id: &str) -> Result<Option<WatchState>, String> {
        let value: Option<String> = self
            .connection
            .query_row("SELECT state FROM watches WHERE id=?1", [id], |r| r.get(0))
            .optional()
            .map_err(db_error)?;
        value
            .map(|s| serde_json::from_str(&s).map_err(db_error))
            .transpose()
    }
    /// State comparison and successful cursor advancement share a commit.
    pub fn update_watch(
        &mut self,
        id: &str,
        expected: u64,
        value: Truth,
        cursor: Option<String>,
    ) -> Result<WatchState, String> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let raw: String = tx
            .query_row("SELECT state FROM watches WHERE id=?1", [id], |r| r.get(0))
            .optional()
            .map_err(db_error)?
            .ok_or("unknown watch")?;
        let mut state: WatchState = serde_json::from_str(&raw).map_err(db_error)?;
        state.update(expected, value, cursor)?;
        tx.execute(
            "UPDATE watches SET state=?1 WHERE id=?2",
            params![serde_json::to_string(&state).map_err(db_error)?, id],
        )
        .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(state)
    }
}

/// A notification waiting for its transport; `uncertain` ones were handed out and never resolved.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingNotification {
    pub id: String,
    pub payload: String,
    pub uncertain: bool,
    /// The transport holding the claim on an uncertain notification.
    #[serde(default)]
    pub claimed_by: Option<String>,
}
/// How long a claim keeps other transports off a notification; its owner may reclaim it any time.
pub const NOTIFICATION_LEASE_SECONDS: i64 = 60;
/// What a transport learns when it claims a notification.
///
/// Policy for an uncertain notification: the store cannot know whether the OS showed it, so it
/// never resolves one by itself. The claiming transport (or, once the lease lapses, another one)
/// reclaims it with `may_repeat` set and shows it marked as a possible repeat. A notification is
/// therefore never lost, and one that may be a duplicate is labelled as such.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotificationClaim {
    /// An earlier claim never resolved: the user may already have seen this notification.
    pub may_repeat: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotificationStatus {
    Pending,
    Uncertain,
    Delivered { receipt: String },
}
impl SqliteStore {
    /// Commit predicate state, successful cursor and notification intent together.
    pub fn update_watch_notify(
        &mut self,
        id: &str,
        expected: u64,
        value: Truth,
        cursor: Option<String>,
        notification_id: &str,
        payload: &str,
    ) -> Result<WatchState, String> {
        if notification_id.is_empty() || payload.len() > 65536 {
            return Err("invalid notification".into());
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let raw: String = tx
            .query_row("SELECT state FROM watches WHERE id=?1", [id], |r| r.get(0))
            .optional()
            .map_err(db_error)?
            .ok_or("unknown watch")?;
        let mut state: WatchState = serde_json::from_str(&raw).map_err(db_error)?;
        state.update(expected, value, cursor)?;
        tx.execute(
            "UPDATE watches SET state=?1 WHERE id=?2",
            params![serde_json::to_string(&state).map_err(db_error)?, id],
        )
        .map_err(db_error)?;
        if value == Truth::True {
            tx.execute(
                "INSERT INTO task_outbox(id,kind,payload,delivered) VALUES(?1,'notification',?2,0)",
                params![notification_id, payload],
            )
            .map_err(db_error)?;
        }
        tx.commit().map_err(db_error)?;
        Ok(state)
    }
    pub fn notification_status(&self, id: &str) -> Result<NotificationStatus, String> {
        let status: Option<i64> = self
            .connection
            .query_row(
                "SELECT delivered FROM task_outbox WHERE id=?1 AND kind='notification'",
                [id],
                |r| r.get(0),
            )
            .optional()
            .map_err(db_error)?;
        match status {
            Some(0) => Ok(NotificationStatus::Pending),
            Some(2) => Ok(NotificationStatus::Uncertain),
            Some(1) => {
                let receipt: String = self
                    .connection
                    .query_row(
                        "SELECT receipt FROM notification_receipts WHERE outbox_id=?1",
                        [id],
                        |r| r.get(0),
                    )
                    .map_err(db_error)?;
                Ok(NotificationStatus::Delivered { receipt })
            }
            _ => Err("unknown notification state".into()),
        }
    }
    /// Undelivered notifications: pending ones first, then uncertain ones, each in the order they
    /// were raised, so unresolved claims never hide a new notification.
    pub fn pending_notifications(&self, limit: u32) -> Result<Vec<PendingNotification>, String> {
        let mut stmt = self
            .connection
            .prepare("SELECT o.id,o.payload,o.delivered,c.transport FROM task_outbox o LEFT JOIN notification_claims c ON c.outbox_id=o.id WHERE o.kind='notification' AND o.delivered IN (0,2) ORDER BY o.delivered,o.rowid LIMIT ?1")
            .map_err(db_error)?;
        let rows = stmt
            .query_map([limit.min(1000)], |r| {
                Ok(PendingNotification {
                    id: r.get(0)?,
                    payload: r.get(1)?,
                    uncertain: r.get::<_, i64>(2)? == 2,
                    claimed_by: r.get(3)?,
                })
            })
            .map_err(db_error)?;
        rows.map(|r| r.map_err(db_error)).collect()
    }
    /// Call before external dispatch: crash or ambiguous result remains uncertain. A pending
    /// notification is claimed outright; an uncertain one only by its owner or after its lease.
    pub fn claim_notification(
        &mut self,
        id: &str,
        transport: &str,
        now: DateTime<Utc>,
    ) -> Result<NotificationClaim, String> {
        if transport.trim().is_empty() {
            return Err("empty notification transport".into());
        }
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let status: i64 = tx
            .query_row(
                "SELECT delivered FROM task_outbox WHERE id=?1 AND kind='notification'",
                [id],
                |r| r.get(0),
            )
            .optional()
            .map_err(db_error)?
            .ok_or("unknown notification")?;
        let may_repeat = match status {
            0 => false,
            2 => {
                let claim: Option<(String, String)> = tx
                    .query_row(
                        "SELECT transport,expires_at FROM notification_claims WHERE outbox_id=?1",
                        [id],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .optional()
                    .map_err(db_error)?;
                if let Some((owner, expires)) = claim {
                    let expires = DateTime::parse_from_rfc3339(&expires).map_err(db_error)?;
                    if owner != transport && expires > now {
                        return Err("notification is claimed by another transport".into());
                    }
                }
                true
            }
            _ => return Err("notification is not pending".into()),
        };
        let expires = now
            .checked_add_signed(Duration::seconds(NOTIFICATION_LEASE_SECONDS))
            .ok_or("notification lease overflow")?;
        tx.execute(
            "INSERT OR REPLACE INTO notification_claims(outbox_id,transport,expires_at) VALUES(?1,?2,?3)",
            params![id, transport, expires.to_rfc3339()],
        )
        .map_err(db_error)?;
        tx.execute("UPDATE task_outbox SET delivered=2 WHERE id=?1", [id])
            .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(NotificationClaim { may_repeat })
    }
    /// A receipt commits delivery; explicit absence permits a retry. Only the claim's owner
    /// resolves it; until then, uncertainty remains visible.
    pub fn resolve_notification(
        &mut self,
        id: &str,
        transport: &str,
        receipt: Option<String>,
    ) -> Result<(), String> {
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(db_error)?;
        let status: i64 = tx
            .query_row(
                "SELECT delivered FROM task_outbox WHERE id=?1 AND kind='notification'",
                [id],
                |r| r.get(0),
            )
            .optional()
            .map_err(db_error)?
            .ok_or("unknown notification")?;
        if status == 1 {
            let previous: String = tx
                .query_row(
                    "SELECT receipt FROM notification_receipts WHERE outbox_id=?1",
                    [id],
                    |r| r.get(0),
                )
                .map_err(db_error)?;
            return if receipt.as_ref() == Some(&previous) {
                Ok(())
            } else {
                Err("notification already delivered with different receipt".into())
            };
        }
        if status != 2 {
            return Err("notification is not uncertain".into());
        }
        let owner: Option<String> = tx
            .query_row(
                "SELECT transport FROM notification_claims WHERE outbox_id=?1",
                [id],
                |r| r.get(0),
            )
            .optional()
            .map_err(db_error)?;
        if owner.as_deref() != Some(transport) {
            return Err("notification is claimed by another transport".into());
        }
        match receipt {
            Some(value) if !value.trim().is_empty() => {
                tx.execute(
                    "INSERT INTO notification_receipts(outbox_id,receipt) VALUES(?1,?2)",
                    params![id, value],
                )
                .map_err(db_error)?;
                tx.execute("UPDATE task_outbox SET delivered=1 WHERE id=?1", [id])
                    .map_err(db_error)?;
            }
            Some(_) => return Err("empty notification receipt".into()),
            None => {
                tx.execute("UPDATE task_outbox SET delivered=0 WHERE id=?1", [id])
                    .map_err(db_error)?;
            }
        }
        tx.execute("DELETE FROM notification_claims WHERE outbox_id=?1", [id])
            .map_err(db_error)?;
        tx.commit().map_err(db_error)?;
        Ok(())
    }
}
