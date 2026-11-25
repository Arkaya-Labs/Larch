//! Pinned documents and their `.pin` files (architecture section 8; format in
//! `larch_engine`'s `pinfile`). Unstable: may change before 1.0.

use std::io::{Read, Write};

pub use larch_engine::PinError;
use larch_engine::PinHandle;

use crate::Larch;
use crate::session::KV_BLOCK_TOKENS;
use larch_engine::TURN_ROOM;

/// A document kept in the prefix cache ([`Larch::pin`]); dropping it lets the cache evict it.
/// Unstable: may change before 1.0.
pub struct Pin {
    pub(crate) hold: PinHandle,
    tokens: Vec<u32>,
}

impl std::fmt::Debug for Pin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pin({} tokens)", self.tokens.len())
    }
}

impl Pin {
    pub(crate) fn new(hold: PinHandle, tokens: Vec<u32>) -> Self {
        Self { hold, tokens }
    }

    /// The pinned prompt.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

impl Larch {
    /// Whether `pin` still holds its document: false once it went cold to give way to a request
    /// that needed its blocks (a pin never fails or holds up another request). A question on a
    /// cold pin's document reads it again; the pin file is still good. Unstable: may change before
    /// 1.0.
    pub fn pin_live(&self, pin: &Pin) -> bool {
        self.runtime.engine.pin_live(&pin.hold)
    }

    /// The tokens a turn on `pin` may add after the document: what the context and the KV pool
    /// leave beside the pinned blocks, never under [`TURN_ROOM`] while the pin is live (the plan
    /// kept that much); 0 for a cold pin. Unstable: may change before 1.0.
    pub fn turn_room(&self, pin: &Pin) -> u32 {
        let engine = &self.runtime.engine;
        if !engine.pin_live(&pin.hold) {
            return 0;
        }
        let context = engine.deployment().ctx_max;
        let context = context.saturating_sub(u32::try_from(pin.tokens.len()).unwrap_or(u32::MAX));
        let pool = engine.prefix_cache_stats().map_or(0, |s| {
            let free = s.total_blocks.saturating_sub(s.pinned_blocks);
            u32::try_from(free).map_or(u32::MAX, |b| b.saturating_mul(KV_BLOCK_TOKENS))
        });
        context.min(pool).max(TURN_ROOM)
    }

    /// Appends `pin`'s segments from token `from` (0 for a new file, else what the last call
    /// returned) to `w`, with the header first when `from` is 0, up to its last checkpoint (while
    /// its read is under way, its last 8,192-token mark; a hybrid's state is kept at each);
    /// returns the token the file now ends at, which is `from` when nothing is new. Nothing at
    /// all to export is [`PinError::Miss`]. Only the new segments leave the device: a caller
    /// follows a read by calling this as the progress events come, appending to one file.
    /// Unstable: may change before 1.0.
    // ponytail: the engine thread copies a segment's layer between its steps (about 0.1 s a layer
    // at 8,192 tokens); slice by block ranges if a chat's first token shows it.
    pub fn export_pin(&self, pin: &Pin, w: &mut dyn Write, from: u32) -> Result<u32, PinError> {
        self.runtime.engine.export_pin(&pin.hold, w, from)
    }

    /// Writes `pin`'s recurrent state sidecar to `w` and returns the tokens it covers (a dense
    /// model's is empty, but still names them). Write it to a temporary file and rename it over
    /// the old one, and only once the pin file holds at least that many tokens: call
    /// [`Self::export_pin`] first, and again if this returns more than it did. Unstable: may
    /// change before 1.0.
    pub fn export_pin_state(&self, pin: &Pin, w: &mut dyn Write) -> Result<u32, PinError> {
        self.runtime.engine.export_pin_state(&pin.hold, w)
    }

    /// Loads a pin file and its state sidecar into the cache of `namespace` and pins `tokens` up
    /// to the whole blocks the sidecar covers (a torn trailing record is ignored; a hybrid's
    /// missing record is `Miss`, as its state belongs to the full count). A document longer
    /// than that is read in by [`Self::pin`] after this, which computes only the rest. The files
    /// must be this engine's (model file, build and kernel ABI, KV layout and dtype, state size,
    /// no image rows) with every checksum good and the ids matching `tokens`; else `Miss`,
    /// nothing published. Reloads once for the documents plan like [`Self::pin`], after the
    /// headers check out. Unstable: may change before 1.0.
    pub fn import_pin(
        &self,
        tokens: Vec<u32>,
        namespace: u64,
        pin: &mut dyn Read,
        state: &mut dyn Read,
    ) -> Result<Pin, PinError> {
        self.require_prefix_cache()?;
        let engine = &self.runtime.engine;
        let n = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        let (hold, done) = engine.import_pin(&tokens, namespace, pin, state, &mut || {
            self.prepare_document(n)
        })?;
        Ok(Pin::new(hold, tokens[..done].to_vec()))
    }
}
