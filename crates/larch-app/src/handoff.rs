//! Stand-ins for the two functions PR #123 (F4) makes public, with that branch's exact signatures.
//! After it merges this file goes, and the app imports `larch::chat::data_dir` and
//! `larch_engine::{FitEstimate, fit_estimate}` instead.

use std::{env, path::PathBuf};

use anyhow::{Context, Result};
use larch_engine::EngineOptions;
use larch_records::LoadError;

/// The per-user data directory (architecture 14): `larch` under %LOCALAPPDATA%, ~/Library/Application
/// Support, $XDG_DATA_HOME or ~/.local/share. Transcripts live in its `chat/` subdirectory.
// A copy of chat.rs's private `data_dir`, which #123 only makes `pub`.
pub fn data_dir() -> Result<PathBuf> {
    #[cfg(windows)]
    let base = env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let base = env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"));
    #[cfg(all(unix, not(target_os = "macos")))]
    let base = env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute()) // the XDG spec ignores empty and relative values
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")));
    Ok(base
        .context("set --data-dir: no per-user data directory")?
        .join("larch"))
}

/// The memory a model would take on its device, before it is loaded (9.10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FitEstimate {
    /// The GGUF's tensor bytes.
    pub weights_bytes: u64,
    /// The KV pool a load would allocate.
    pub kv_bytes: u64,
    /// Full `max_context` conversations that pool holds.
    pub max_seqs_fit: u32,
}

/// Always "not checked" until #123 lands: the fit plan is the engine's, so the app never guesses
/// it, and every model shows `Fit::Unknown` meanwhile.
pub fn fit_estimate(_opts: &EngineOptions) -> Result<FitEstimate, LoadError> {
    Err(LoadError::Backend(0))
}
