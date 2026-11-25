//! A chat's documents held ready: the route they take (`route`), and for a medium set the engine's
//! pin of the Folio block (`Larch::pin`), read with its progress as `Documents` events and
//! kept on disk (`cache`) so a restart, or a chat that regains focus, imports it instead of
//! reading it again. Only the chat in focus holds a pin: the app releases the others
//! (`Held::release`).

use std::{
    io::Seek,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use larch::{EngineError, Larch, Pin, PinError, TextEvent, api::invalid};
use larch_model::chat::ToolMessage;

use super::{Doc, Documents, Route, cache, in_time, pages, route, system};
use crate::{run::AgentEvent, skill::SkillError};

/// What a call needs of its chat and the engine.
pub struct Env<'a> {
    /// The engine the documents are pinned in.
    pub larch: &'a Larch,
    /// The app's data folder.
    pub data: &'a Path,
    /// The prefix cache's namespace (`Agent`'s), the one the chat's turns read from, so a turn finds the pin.
    pub namespace: u64,
    /// Tokens of the context loaded now, which `route` sizes small documents by.
    pub context: u32,
    /// Whether pins are kept on disk (`config.toml`'s `docs_keep_ready`).
    pub keep: bool,
    /// The most bytes of pins kept (`docs_cache_bytes`); `None` is `cache::budget`'s default.
    pub budget: Option<u64>,
    /// Set by the user's Stop.
    pub stop: &'a AtomicBool,
    /// The tools a chat step shows, as `run::params` gives them. Some templates (Qwen 3.5) render
    /// them before the system message, so the pinned tokens start with them.
    pub tools: Vec<serde_json::Value>,
}

/// How a set of documents is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Plain text in the chat's system prompt; nothing is pinned.
    Small,
    /// Pinned in the engine.
    Pinned,
    /// Too long for either: read by section.
    Large,
}

/// What `Held::release` did with the pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Released {
    /// No pin was held.
    Nothing,
    /// The pin went and nothing it held is lost: its file holds it whole, so `hold` imports it
    /// back, or it had gone cold already.
    Covered,
    /// The pin went and no file holds it whole: the next question reads the set again.
    Uncovered,
    /// No file holds the pin whole and the caller asked to keep such a pin, so it stays.
    Kept,
}

/// The documents of one chat and the pin that holds them.
pub struct Held {
    /// The documents, in letter order.
    pub docs: Vec<Doc>,
    /// How they are held.
    pub kind: Kind,
    pin: Option<Pin>,
    tokens: Vec<u32>,
    key: String,
    /// The tools the pinned tokens start with.
    tools: Vec<serde_json::Value>,
    /// Tokens the pin file holds, from which the next export continues.
    sent: u32,
    /// The context the set was routed against (`Env::context`): another means route again.
    context: u32,
}

/// The tokens a turn on `docs` starts with: the Folio block as the chat template renders it, the
/// prefix two turns with other agent text and other questions share, less its last token, where
/// the tokenizer may merge the agent's text into it. Rendered with `tools`, which a real step
/// shows and some templates put first: without them no turn would share the pin.
pub fn prefix(
    larch: &Larch,
    docs: &[Doc],
    tools: &[serde_json::Value],
) -> Result<Vec<u32>, EngineError> {
    let turn = |plain: &str, question: &str| {
        let message = |role: &str, content: String| ToolMessage {
            role: role.into(),
            content,
            ..Default::default()
        };
        let messages = [
            message("system", system(docs, plain)),
            message("user", question.into()),
        ];
        larch.encode_messages(&messages, tools, Some(false))
    };
    let (a, b) = (turn("A", "Who?")?, turn("B", "When?")?);
    let same = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    Ok(a[..same.saturating_sub(1)].to_vec())
}

impl Held {
    /// `docs` routed by size, and pinned when medium: imported from the disk when it has them,
    /// else read, with `on` hearing each step. On `PINS_FULL`, `unpin_others` releases the other
    /// chats' pins and the pin is tried once more.
    pub fn open(
        env: &Env,
        docs: Vec<Doc>,
        on: &mut dyn FnMut(AgentEvent),
        unpin_others: &mut dyn FnMut(),
    ) -> Result<Held, SkillError> {
        // `route` counts the block once; the tool frame is part of what the pin holds.
        let frame = tool_frame(env);
        let count = |text: &str| {
            u32::try_from(env.larch.token_ids(text).len())
                .unwrap_or(u32::MAX)
                .saturating_add(frame)
        };
        let rate = cache::read_rate(&cache::dir(env.data), env.larch.model_path());
        let fit = |n: u32| in_time(env.larch.document_fit(n), n, rate);
        let plan = route(docs, env.context, count, fit);
        let (docs, kind) = match plan {
            Route::Small(docs) => (docs, Kind::Small),
            Route::Pinned(docs, _) => (docs, Kind::Pinned),
            Route::Large { docs, fits_pages } => {
                on(AgentEvent::Documents(Documents::FitsUpTo {
                    pages: fits_pages.unwrap_or(0),
                }));
                // Kept for the prompt's list of them (`large_note`); nothing is pinned.
                return Ok(Held::new(docs, Kind::Large, Vec::new(), env));
            }
        };
        let tokens = match kind {
            Kind::Pinned => prefix(env.larch, &docs, &env.tools).map_err(SkillError::Engine)?,
            _ => Vec::new(),
        };
        let mut held = Held::new(docs, kind, tokens, env);
        match kind {
            Kind::Pinned => held.hold(env, on, unpin_others)?,
            _ => on(AgentEvent::Documents(Documents::Ready)),
        }
        Ok(held)
    }

    fn new(docs: Vec<Doc>, kind: Kind, tokens: Vec<u32>, env: &Env) -> Held {
        let key = cache::key(&tokens, Path::new(""));
        Held {
            docs,
            kind,
            pin: None,
            tokens,
            key,
            tools: env.tools.clone(),
            sent: 0,
            context: env.context,
        }
    }

    /// Whether the pin is in the engine and live: one that went cold to give way to another
    /// request is not.
    pub fn pinned(&self, larch: &Larch) -> bool {
        self.pin.as_ref().is_some_and(|p| larch.pin_live(p))
    }

    /// The tokens a turn may add after the documents: the live pin's room, else `TURN_ROOM`.
    pub fn turn_room(&self, larch: &Larch) -> u32 {
        match &self.pin {
            Some(pin) if larch.pin_live(pin) => larch.turn_room(pin),
            _ => super::TURN_ROOM,
        }
    }

    /// Pins a medium set again after `release`, as `open` does.
    pub fn hold(
        &mut self,
        env: &Env,
        on: &mut dyn FnMut(AgentEvent),
        unpin_others: &mut dyn FnMut(),
    ) -> Result<(), SkillError> {
        if self.kind != Kind::Pinned || self.pinned(env.larch) {
            return Ok(());
        }
        // A cold pin's file is still good: dropping it sends this call down the import path.
        self.pin = None;
        // Where the file ends is what this import finds, not what an earlier session wrote: a
        // refused import deletes the file, and `read` takes `sent` as the end of a file it
        // resumes, so a stale count would have the next export write an empty one (#590 Mac
        // review).
        self.sent = 0;
        // The key names the model, which the engine's files check too.
        self.key = cache::key(&self.tokens, env.larch.model_path());
        let dir = cache::dir(env.data);
        let total = u32::try_from(self.tokens.len()).unwrap_or(u32::MAX);
        let mut full = false;
        if env.keep {
            // Only an import that finds a file reloads here; a miss leaves it to `read`.
            if dir.join(&self.key).exists() {
                making_room(env, total, on);
            }
            let tokens = self.tokens.clone();
            let loaded = cache::load(&dir, &self.key, |file, state| {
                let state = state.unwrap_or_default();
                let (mut tried, mut torn) = (false, false);
                loop {
                    let mut side = state;
                    let got = env
                        .larch
                        .import_pin(tokens.clone(), env.namespace, file, &mut side);
                    match got {
                        Ok(pin) => return Ok(Some(pin)),
                        Err(PinError::Miss) => return Ok(None),
                        // The pool is full of other chats' pins (`PINS_FULL` arrives as
                        // `NoRoom`): they give way and the file is imported once more. A second
                        // refusal keeps the file (an error, not a miss) and the set is read.
                        Err(PinError::NoRoom) if !tried => {
                            tried = true;
                            unpin_others();
                            file.rewind().map_err(PinError::Io)?;
                        }
                        // Another chat's documents reload broke the import off: the file is
                        // good, and is imported once more (#591 review).
                        Err(PinError::Engine(EngineError::InvalidRequest(
                            invalid::IMPORT_TORN,
                        ))) if !torn => {
                            torn = true;
                            file.rewind().map_err(PinError::Io)?;
                        }
                        Err(e) => return Err(e),
                    }
                }
            });
            match loaded {
                Ok(Some(pin)) => {
                    full = u32::try_from(pin.tokens().len()).is_ok_and(|n| n >= total);
                    self.sent = u32::try_from(pin.tokens().len()).unwrap_or(0);
                    self.pin = Some(pin);
                }
                Ok(None) => {}
                Err(cache::Failed::Engine(PinError::NoRoom)) => on(AgentEvent::Notice(
                    "Other chats' documents fill the memory, so these are read again.".into(),
                )),
                Err(e) => on(AgentEvent::Notice(format!("document cache: {e:?}"))),
            }
        }
        if !full && !self.read(env, on, unpin_others)? {
            // Too slow to wait for: the set is read by section from now on (`Kind::Large`).
            self.kind = Kind::Large;
            on(AgentEvent::Notice(SLOW.into()));
            // The pages a read at the rate it just kept takes `READ_SECS` for, as `route` says it.
            let rate = cache::read_rate(&dir, env.larch.model_path()).unwrap_or(0.0);
            let share = (rate * super::READ_SECS / f64::from(total.max(1))).min(1.0);
            let pages = (f64::from(pages(&self.docs)) * share) as u32;
            on(AgentEvent::Documents(Documents::FitsUpTo { pages }));
            return Ok(());
        }
        on(AgentEvent::Documents(Documents::ready(
            env.larch.document_fit(total).host_layers,
        )));
        Ok(())
    }

    /// Reads the block in, as one background request whose progress is the chip, exporting the pin
    /// every `cache::EXPORT_EVERY` tokens and once more at the end or on Stop. False when the
    /// read gave up as too slow (`too_slow`): nothing is pinned, and its rate is kept so the next
    /// set routes by it (`in_time`).
    fn read(
        &mut self,
        env: &Env,
        on: &mut dyn FnMut(AgentEvent),
        unpin_others: &mut dyn FnMut(),
    ) -> Result<bool, SkillError> {
        let mut tried = false;
        making_room(
            env,
            u32::try_from(self.tokens.len()).unwrap_or(u32::MAX),
            on,
        );
        let (pin, mut stream) = loop {
            match env.larch.pin(self.tokens.clone(), env.namespace) {
                Err(EngineError::InvalidRequest(invalid::PINS_FULL)) if !tried => {
                    tried = true;
                    unpin_others();
                }
                other => break other.map_err(SkillError::Engine)?,
            }
        };
        // A partial pin from the disk was imported before this one: its file ends where the
        // import's blocks did, and a torn tail would sit between old and new segments. This pin
        // writes the file again from the start, unless it ends with nothing past the import
        // (`export`'s `kept`); `resumed` is where that file ends until this pin's first export.
        self.pin = None;
        let mut resumed = std::mem::take(&mut self.sent);
        let docs = pages(&self.docs);
        let mut done = 0;
        let mut stopped = false;
        // The read's rate, from its first progress on: an imported head and the queue are not in it.
        let mut first: Option<(u32, Instant)> = None;
        while let Some(event) = stream.next() {
            match event {
                TextEvent::Prefilled { done: d, total: t } => {
                    done = d;
                    let (from, since) = *first.get_or_insert((d, Instant::now()));
                    if d >= t && d - from >= cache::EXPORT_EVERY {
                        // A short read is mostly fixed cost: its rate would say long reads are slow.
                        // A rate the disk refuses leaves the device alone to route the next set.
                        let secs = since.elapsed().as_secs_f64();
                        let dir = cache::dir(env.data);
                        let _ = cache::keep_read_rate(&dir, env.larch.model_path(), d - from, secs);
                    }
                    let secs = since.elapsed().as_secs_f64();
                    if too_slow(secs, d - from, t - from) {
                        stream.cancel();
                        let dir = cache::dir(env.data);
                        let _ = cache::keep_read_rate(&dir, env.larch.model_path(), d - from, secs);
                        return Ok(false);
                    }
                    on(AgentEvent::Documents(Documents::reading(d, t, docs)));
                    // A resumed read's file holds the import: the next checkpoint is past it.
                    if cache::export_due(self.sent.max(resumed), d, t, false) {
                        let kept = std::mem::take(&mut resumed);
                        self.export(env, &pin, if d >= t { kept } else { 0 }, on);
                    }
                }
                TextEvent::Error(e) => return Err(SkillError::Engine(e)),
                TextEvent::Done { .. } => break,
                TextEvent::Text { .. } | TextEvent::Thinking { .. } => {}
            }
            if env.stop.load(Ordering::Relaxed) {
                stream.cancel();
                stopped = true;
                break;
            }
        }
        // The last checkpoint of a read that was stopped, or of one that ended.
        if done > 0 || !stopped {
            self.export(env, &pin, if stopped { 0 } else { resumed }, on);
        }
        if stopped {
            return Err(SkillError::Engine(EngineError::Cancelled));
        }
        if env.keep
            && let Ok(free) = cache::free(&cache::dir(env.data))
        {
            let _ = cache::evict(&cache::dir(env.data), cache::budget(env.budget, free));
        }
        self.pin = Some(pin);
        Ok(true)
    }

    /// Appends what `pin` has past the file's end to the pin file, with its state sidecar. `kept`,
    /// when not 0: `pin` finished reading over an import whose file holds `kept` tokens. True
    /// when the file then holds all `pin` can export.
    fn export(&mut self, env: &Env, pin: &Pin, kept: u32, on: &mut dyn FnMut(AgentEvent)) -> bool {
        if !env.keep {
            return false;
        }
        // The pin's whole blocks end where the file does (its tail under a block, which no file
        // holds): the file already is this pin's, header to state. The state export only names
        // the chain's end, without copying KV off the device. A pin with more writes the file again.
        if kept > 0 && env.larch.export_pin_state(pin, &mut std::io::sink()).ok() == Some(kept) {
            self.sent = kept;
            return true;
        }
        let dir = cache::dir(env.data);
        if self.sent == 0 {
            // A file left by another read of this key would take a second header.
            let _ = std::fs::remove_file(dir.join(&self.key));
        }
        let (from, mut to) = (self.sent, self.sent);
        let wrote = cache::append(&dir, &self.key, |file| {
            to = env.larch.export_pin(pin, file, from)?;
            let mut state = Vec::new();
            // The state covers a checkpoint the file must reach first.
            if env.larch.export_pin_state(pin, &mut state)? > to {
                to = env.larch.export_pin(pin, file, to)?;
            }
            Ok::<_, PinError>(state)
        });
        match wrote {
            Ok(true) => {
                self.sent = to;
                let hashes: Vec<String> = self.docs.iter().map(|d| d.sha256.clone()).collect();
                if let Err(e) = cache::tag(&dir, &self.key, &hashes) {
                    on(AgentEvent::Notice(format!("document cache: {e}")));
                }
                true
            }
            // No room on the volume, or nothing yet to export: only the memory holds the pin.
            Ok(false) | Err(cache::Failed::Engine(PinError::Miss)) => false,
            Err(e) => {
                on(AgentEvent::Notice(format!("document cache: {e:?}")));
                false
            }
        }
    }

    /// Exports the pin and lets it go: the chat has lost focus, or sat idle. `hold` imports it
    /// again. With `covered_only` (the app's idle release) a pin no file holds whole stays:
    /// "Keep documents ready" is off, or the volume has no room, and letting it go would cost
    /// the next question a full read (#591 review).
    pub fn release(
        &mut self,
        env: &Env,
        on: &mut dyn FnMut(AgentEvent),
        covered_only: bool,
    ) -> Released {
        let Some(pin) = self.pin.take() else {
            return Released::Nothing;
        };
        // A cold pin's blocks went to another request: there is nothing to write or to lose.
        if !env.larch.pin_live(&pin) || self.export(env, &pin, 0, on) {
            Released::Covered
        } else if covered_only {
            self.pin = Some(pin);
            Released::Kept
        } else {
            Released::Uncovered
        }
    }

    /// Whether `docs` are the documents this holds, by what each was read from, pinned under the
    /// same tools and routed against the same context: a changed list changes the pinned tokens,
    /// and a changed context may change the route (#590 F7), so the set is routed and read again.
    /// A large set counts too, so it is not loaded, counted and routed before every question
    /// (#590 F11).
    pub fn same(&self, docs: &[Doc], env: &Env) -> bool {
        let key = |d: &Doc| (d.index, d.sha256.clone());
        self.context == env.context
            && self.tools == env.tools
            && (self.docs.iter().map(key).collect::<Vec<_>>()
                == docs.iter().map(key).collect::<Vec<_>>())
    }
}

/// What the chat hears when a read gives up (`too_slow`).
const SLOW: &str = "These documents read too slowly on this computer to wait for, so questions \
search them by section instead.";

/// Seconds into a first read before its rate is trusted: the first chunks hold the queue and
/// the first launches.
const SLOW_AFTER: f64 = 15.0;

/// Whether a first read that took `secs` for `done` of its `total` tokens would take more than
/// twice `READ_SECS` in all: `in_time` routes such a set to sections once a rate is kept, and this
/// is that rule for the read that measures the rate (Windows 5070 Ti, 23 Nov: a 121-page read at
/// ~85 tok/s, the host experts prefilled on the CPU, 20 minutes). The early rate is the fastest
/// (attention grows with the read), so the projection errs short. A read that is done never is:
/// it is pinned, however long its last chunk took.
fn too_slow(secs: f64, done: u32, total: u32) -> bool {
    secs >= SLOW_AFTER
        && done > 0
        && done < total
        && secs * f64::from(total) / f64::from(done) > 2.0 * super::READ_SECS
}

/// The tokens `env.tools` add to a prompt, as the template renders them.
fn tool_frame(env: &Env) -> u32 {
    let system = [ToolMessage {
        role: "system".into(),
        ..Default::default()
    }];
    let len = |tools: &[serde_json::Value]| {
        env.larch
            .encode_messages(&system, tools, Some(false))
            // A failed render fails `prefix` next, which reports it.
            .map_or(0, |t| t.len())
    };
    u32::try_from(len(&env.tools).saturating_sub(len(&[]))).unwrap_or(u32::MAX)
}

/// Tells the chip the engine is about to reload for `tokens`, before the call that blocks on it.
fn making_room(env: &Env, tokens: u32, on: &mut dyn FnMut(AgentEvent)) {
    if env.larch.pin_reloads(tokens) {
        on(AgentEvent::Documents(Documents::MakingRoom {
            secs: env.larch.load_secs(),
        }));
    }
}

#[cfg(test)]
mod tests {
    /// A read gives up only once its rate is trusted and only when it projects past twice
    /// READ_SECS (85 tok/s over 90k tokens does, 1,000 tok/s over 110k does not), and never once
    /// it is done.
    #[test]
    fn only_a_read_projected_past_twice_the_read_time_gives_up() {
        use super::too_slow;
        assert!(!too_slow(10.0, 850, 90_000), "too early to trust");
        assert!(too_slow(20.0, 1_700, 90_000));
        assert!(!too_slow(20.0, 20_000, 110_000));
        assert!(!too_slow(20.0, 0, 110_000));
        assert!(
            !too_slow(3_600.0, 90_000, 90_000),
            "the last progress: the read is done"
        );
    }
}
