//! Runtime effect admission. The host must supply a transactional, durable journal and trusted
//! declarations; the legacy agent loop remains independent until explicitly migrated.
use super::permissions::{LiveGrants, Scope};
use crate::parse::{ToolCall, validate};
use crate::tool::{self, Tool, ToolCtx};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Unknown tools are opaque by default: an unfinished invocation must never be repeated blindly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplayClass {
    Opaque,
    IdempotentWithProviderKey,
    Pure,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Intent {
    pub effect_id: String,
    pub tool: String,
    pub args_digest: String,
    pub capability: String,
    pub replay: ReplayClass,
    pub provider_key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Receipt {
    pub result: Result<String, String>,
    pub untrusted: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reservation {
    New,
    Completed(Receipt),
    Uncertain,
    Conflict,
}

/// Implementations atomically compare the entire intent and reserve its identity durably before
/// returning `New`. A prior unfinished intent returns `Uncertain` unless an explicitly declared
/// replay policy is reconciled by the host. Receipt commits must also be durable.
pub trait EffectJournal {
    fn reserve(&mut self, intent: &Intent) -> Result<Reservation, String>;
    fn receipt(&mut self, effect_id: &str, receipt: &Receipt) -> Result<(), String>;
}

/// Trusted declarations derive authority from *prepared* arguments. They must reject paths and
/// resources outside their own validated scope; they are never populated from model claims.
pub trait Declaration {
    fn tool_name(&self) -> &str;
    fn capability(&self, prepared: &Value, ctx: &ToolCtx) -> Result<String, String>;
    /// `Pure` is accepted only for a read-only tool, and makes its errors durable receipts.
    fn replay_class(&self) -> ReplayClass {
        ReplayClass::Opaque
    }
    fn max_argument_bytes(&self) -> usize;
    fn max_result_bytes(&self) -> usize;
}

#[derive(Debug, PartialEq, Eq)]
pub enum DispatchError {
    Rejected(String),
    Journal(String),
    Uncertain,
    Conflict,
}

/// Prepare, authorize and durably reserve before executing. The prepared value passed to the
/// tool is exactly the one whose digest was journaled. A failed receipt commit is uncertain.
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    journal: &mut dyn EffectJournal,
    grants: &LiveGrants,
    inherited: &Scope,
    tool: &dyn Tool,
    declaration: &dyn Declaration,
    args: &Value,
    ctx: &ToolCtx,
    effect_id: &str,
) -> Result<Receipt, DispatchError> {
    let spec = tool.spec();
    within_argument_limit(&spec, declaration, args).map_err(DispatchError::Rejected)?;
    validate(
        &ToolCall {
            name: spec.name.clone(),
            args: args.clone(),
        },
        &spec,
    )
    .map_err(|e| DispatchError::Rejected(e.to_string()))?;
    let prepared = tool
        .prepare(args, ctx)
        .map_err(|e| DispatchError::Rejected(e.to_string()))?;
    dispatch_prepared(
        journal,
        grants,
        inherited,
        tool,
        declaration,
        &prepared,
        ctx,
        effect_id,
    )
}

/// `dispatch` of arguments `Tool::prepare` already made, sent exactly as given: a reviewed call
/// sends the ones the user approved, never a second preparation that could resolve differently.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_prepared(
    journal: &mut dyn EffectJournal,
    grants: &LiveGrants,
    inherited: &Scope,
    tool: &dyn Tool,
    declaration: &dyn Declaration,
    prepared: &Value,
    ctx: &ToolCtx,
    effect_id: &str,
) -> Result<Receipt, DispatchError> {
    if effect_id.is_empty() {
        return Err(DispatchError::Rejected("empty effect identity".into()));
    }
    let spec = tool.spec();
    if declaration.tool_name() != spec.name {
        return Err(DispatchError::Rejected(
            "declaration does not match tool".into(),
        ));
    }
    let max_result = (spec.max_result_tokens as usize)
        .saturating_mul(4)
        .min(declaration.max_result_bytes());
    within_argument_limit(&spec, declaration, prepared).map_err(DispatchError::Rejected)?;
    let capability = declaration
        .capability(prepared, ctx)
        .map_err(DispatchError::Rejected)?;
    let _guard = grants
        .check(inherited, &capability)
        .map_err(|e| DispatchError::Rejected(e.into()))?;
    let replay = declaration.replay_class();
    // Only a read can be replay-safe: any other call may have changed something on its way out.
    if spec.permission != tool::Permission::Read && replay != ReplayClass::Opaque {
        return Err(DispatchError::Rejected(
            "only a read-only tool can declare replay-safe behavior".into(),
        ));
    }
    if replay == ReplayClass::IdempotentWithProviderKey {
        return Err(DispatchError::Rejected(
            "provider-key dispatch is not supported by this tool interface".into(),
        ));
    }
    let intent = Intent {
        effect_id: effect_id.into(),
        tool: spec.name,
        args_digest: tool::args_hash(prepared),
        capability,
        replay,
        provider_key: (replay == ReplayClass::IdempotentWithProviderKey)
            .then(|| effect_id.to_owned()),
    };
    match journal.reserve(&intent).map_err(DispatchError::Journal)? {
        Reservation::Completed(receipt) => return Ok(receipt),
        Reservation::Uncertain => return Err(DispatchError::Uncertain),
        Reservation::Conflict => return Err(DispatchError::Conflict),
        Reservation::New => {}
    }
    // An opaque tool's error may follow an external effect, so its intent stays unresolved; a
    // pure one changed nothing, so its error is the receipt and a retry can run it again.
    let result = match tool.call(prepared, ctx) {
        // A long result is cut, the cut stated, as a chat's is: a write's has already happened,
        // so failing it would lose the receipt and invite a second send.
        Ok(value) => Ok(tool::truncate(value, (max_result / 4) as u32)),
        Err(error) if replay == ReplayClass::Pure => Err(error.to_string()),
        Err(_) => return Err(DispatchError::Uncertain),
    };
    let receipt = Receipt {
        result,
        untrusted: spec.untrusted_results,
    };
    journal
        .receipt(effect_id, &receipt)
        .map_err(|_| DispatchError::Uncertain)?;
    Ok(receipt)
}

/// Fails when `value`, a call's arguments raw or prepared, encodes to more bytes than `spec`'s
/// budget (four a token, as `ToolSpec::max_result_tokens` is cut) or `declaration` allows.
pub fn within_argument_limit(
    spec: &tool::ToolSpec,
    declaration: &dyn Declaration,
    value: &Value,
) -> Result<(), String> {
    let max = (spec.max_args_tokens as usize)
        .saturating_mul(4)
        .min(declaration.max_argument_bytes());
    let len = serde_json::to_vec(value).map_err(|e| e.to_string())?.len();
    match len > max {
        true => Err(format!(
            "{}'s arguments take {len} bytes, over its limit of {max}",
            spec.name
        )),
        false => Ok(()),
    }
}

#[derive(Serialize, Deserialize)]
struct StoredIntent {
    attempt_id: String,
    epoch: u64,
    intent: Intent,
}
fn stored_intent(intent: &Intent, attempt_id: &str, epoch: u64) -> Result<String, String> {
    serde_json::to_string(&StoredIntent {
        attempt_id: attempt_id.into(),
        epoch,
        intent: intent.clone(),
    })
    .map_err(|e| e.to_string())
}
fn receipt_command(
    state: &larch_task::types::RunState,
    effect_id: &str,
    attempt_id: &str,
    epoch: u64,
    receipt: &Receipt,
) -> Result<Option<larch_task::types::Command>, String> {
    use larch_task::types::{Command, EffectStatus, RunStatus};
    let effect = state.effects.get(effect_id).ok_or("unknown effect")?;
    let stored: StoredIntent =
        serde_json::from_str(&effect.operation_digest).map_err(|e| e.to_string())?;
    if effect.attempt_id != attempt_id
        || stored.attempt_id != attempt_id
        || stored.epoch != epoch
        || stored.intent.effect_id != effect_id
    {
        return Err("stale effect receipt identity".into());
    }
    let encoded = serde_json::to_string(receipt).map_err(|e| e.to_string())?;
    match &effect.status {
        EffectStatus::Completed { receipt } if *receipt == encoded => Ok(None),
        EffectStatus::Intent => Ok(Some(Command::RecordReceipt {
            effect_id: effect_id.into(),
            attempt_id: attempt_id.into(),
            epoch,
            receipt: encoded,
        })),
        EffectStatus::Uncertain
            if matches!(state.status, RunStatus::Waiting | RunStatus::Cancelled) =>
        {
            Ok(Some(Command::ReconcileEffect {
                effect_id: effect_id.into(),
                receipt: Some(encoded),
            }))
        }
        _ => Err("effect cannot accept this receipt".into()),
    }
}

/// Transactional adapter for the task store. An inserted intent is dispatched by exactly the
/// caller that won the store transaction; an existing or orphaned intent is never dispatched.
pub struct TaskJournal<'a> {
    pub store: &'a mut larch_task::store::SqliteStore,
    pub run_id: &'a str,
    pub attempt_id: &'a str,
    pub epoch: u64,
}

impl EffectJournal for TaskJournal<'_> {
    fn reserve(&mut self, intent: &Intent) -> Result<Reservation, String> {
        use larch_task::types::{Command, CommandEnvelope, EffectStatus};
        let encoded = stored_intent(intent, self.attempt_id, self.epoch)?;
        let (generation, grants) = self.store.effective_grants(self.run_id)?;
        if !grants.contains(&intent.capability) {
            return Err("capability denied or revoked".into());
        }
        let state = self.store.get(self.run_id)?.ok_or("unknown run")?;
        if let Some(existing) = state.effects.get(&intent.effect_id) {
            if existing.operation_digest != encoded || existing.attempt_id != self.attempt_id {
                return Ok(Reservation::Conflict);
            }
            return match &existing.status {
                EffectStatus::Completed { receipt } => serde_json::from_str(receipt)
                    .map(Reservation::Completed)
                    .map_err(|e| e.to_string()),
                _ => Ok(Reservation::Uncertain),
            };
        }
        let command = CommandEnvelope {
            id: format!("runtime.effect.intent:{}", intent.effect_id),
            expected_revision: state.revision,
            command: Command::RecordIntent {
                effect_id: intent.effect_id.clone(),
                attempt_id: self.attempt_id.into(),
                epoch: self.epoch,
                operation_digest: encoded,
            },
        };
        let (_, inserted) = self.store.reserve_granted_effect(
            self.run_id,
            &command,
            &intent.capability,
            generation,
        )?;
        Ok(if inserted {
            Reservation::New
        } else {
            Reservation::Uncertain
        })
    }
    fn receipt(&mut self, effect_id: &str, receipt: &Receipt) -> Result<(), String> {
        use larch_task::types::CommandEnvelope;
        for _ in 0..3 {
            let state = self.store.get(self.run_id)?.ok_or("unknown run")?;
            let Some(command) =
                receipt_command(&state, effect_id, self.attempt_id, self.epoch, receipt)?
            else {
                return Ok(());
            };
            // Include revision because a concurrent cancellation changes the required command.
            let envelope = CommandEnvelope {
                id: format!("runtime.effect.receipt:{effect_id}:{}", state.revision),
                expected_revision: state.revision,
                command,
            };
            match self.store.execute(self.run_id, &envelope) {
                Ok(_) => return Ok(()),
                Err(error) if error.contains("revision conflict") => continue,
                Err(error) => return Err(error),
            }
        }
        Err("receipt revision remained contended".into())
    }
}

/// Actor-backed variant for drivers running outside the single store-owner thread.
pub struct ActorJournal<'a> {
    pub actor: &'a larch_task::runtime::ActorHandle,
    pub run_id: &'a str,
    pub attempt_id: &'a str,
    pub epoch: u64,
    /// The grant generation the user approved the call under, which the reservation must still
    /// be at; `None` reserves under the current one.
    pub approved_generation: Option<u64>,
}
impl EffectJournal for ActorJournal<'_> {
    fn reserve(&mut self, intent: &Intent) -> Result<Reservation, String> {
        use larch_task::types::{Command, CommandEnvelope, EffectStatus};
        let encoded = stored_intent(intent, self.attempt_id, self.epoch)?;
        let (current, grants) = self.actor.effective_grants(self.run_id.into())?;
        let generation = self.approved_generation.unwrap_or(current);
        if !grants.contains(&intent.capability) {
            return Err("capability denied or revoked".into());
        }
        let state = self.actor.get(self.run_id.into())?.ok_or("unknown run")?;
        if let Some(existing) = state.effects.get(&intent.effect_id) {
            if existing.operation_digest != encoded || existing.attempt_id != self.attempt_id {
                return Ok(Reservation::Conflict);
            }
            return match &existing.status {
                EffectStatus::Completed { receipt } => serde_json::from_str(receipt)
                    .map(Reservation::Completed)
                    .map_err(|e| e.to_string()),
                _ => Ok(Reservation::Uncertain),
            };
        }
        let command = CommandEnvelope {
            id: format!("runtime.effect.intent:{}", intent.effect_id),
            expected_revision: state.revision,
            command: Command::RecordIntent {
                effect_id: intent.effect_id.clone(),
                attempt_id: self.attempt_id.into(),
                epoch: self.epoch,
                operation_digest: encoded,
            },
        };
        let (_, inserted) = self.actor.reserve_granted_effect(
            self.run_id.into(),
            command,
            intent.capability.clone(),
            generation,
        )?;
        Ok(if inserted {
            Reservation::New
        } else {
            Reservation::Uncertain
        })
    }
    fn receipt(&mut self, effect_id: &str, receipt: &Receipt) -> Result<(), String> {
        use larch_task::types::CommandEnvelope;
        for _ in 0..3 {
            let state = self.actor.get(self.run_id.into())?.ok_or("unknown run")?;
            let Some(command) =
                receipt_command(&state, effect_id, self.attempt_id, self.epoch, receipt)?
            else {
                return Ok(());
            };
            let envelope = CommandEnvelope {
                id: format!("runtime.effect.receipt:{effect_id}:{}", state.revision),
                expected_revision: state.revision,
                command,
            };
            match self.actor.execute(self.run_id.into(), envelope) {
                Ok(_) => return Ok(()),
                Err(error) if error.contains("revision conflict") => continue,
                Err(error) => return Err(error),
            }
        }
        Err("receipt revision remained contended".into())
    }
}

/// Trusted declaration for the existing search tool. The explicit capability permits sending
/// the prepared query to configured search providers; results retain untrusted provenance.
pub struct WebSearchDeclaration;
impl Declaration for WebSearchDeclaration {
    fn tool_name(&self) -> &str {
        "web_search"
    }
    fn capability(&self, _: &Value, _: &ToolCtx) -> Result<String, String> {
        Ok("network.web_search".into())
    }
    /// A search only reads (its local cache aside): a failed one (no provider answered, an
    /// unreadable config) changed nothing, so it fails its step instead of leaving an effect
    /// nobody can reconcile.
    fn replay_class(&self) -> ReplayClass {
        ReplayClass::Pure
    }
    fn max_argument_bytes(&self) -> usize {
        1024
    }
    fn max_result_bytes(&self) -> usize {
        6000
    }
}

/// A host's tool (an MCP server's, say) under one capability the run must hold, such as
/// `mcp.<server id>`. A read changed nothing, so it is `Pure` and its error a durable receipt;
/// a write is `Opaque`.
pub struct HostDeclaration {
    pub name: String,
    pub capability: String,
    pub replay: ReplayClass,
}
impl Declaration for HostDeclaration {
    fn tool_name(&self) -> &str {
        &self.name
    }
    fn capability(&self, _: &Value, _: &ToolCtx) -> Result<String, String> {
        Ok(self.capability.clone())
    }
    fn replay_class(&self) -> ReplayClass {
        self.replay
    }
    // The spec's own token budgets bound both (`dispatch`).
    fn max_argument_bytes(&self) -> usize {
        usize::MAX
    }
    fn max_result_bytes(&self) -> usize {
        usize::MAX
    }
}

/// Existing web search, available only when explicitly registered by the host.
pub fn web_search_tool() -> std::sync::Arc<dyn Tool> {
    crate::web::search::tool().into()
}

/// Existing page reader, available only when explicitly registered by the host.
pub fn fetch_page_tool() -> std::sync::Arc<dyn Tool> {
    std::sync::Arc::new(crate::web::fetch::FetchPage)
}
