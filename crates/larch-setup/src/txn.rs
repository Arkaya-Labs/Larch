//! Durable, exclusively locked installation transactions and idempotent recovery.

#[path = "txn/engine.rs"]
mod engine;
#[path = "txn/fs.rs"]
mod filesystem;
pub use filesystem::{atomic_write, create_dir_all, sync_parent};

use crate::manifest::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};

/// The machine choice recorded when components were selected.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Detection {
    /// The selected backend flavour.
    pub flavour: String,
    /// The selected GPU's display name, when applicable.
    pub gpu: Option<String>,
    /// The reason for the choice, including slow CPU fallback.
    pub why: String,
}

/// A component's committed version, destination and owned files.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Component {
    /// Its semantic release version.
    pub version: String,
    /// Its backend flavour.
    pub flavour: String,
    /// Its installed directory.
    pub dest: PathBuf,
    /// Relative owned paths, hashes and sizes.
    pub files: Vec<(String, String, u64)>,
}

/// An idempotent integration with the user's operating-system settings.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Effect {
    /// The integration implemented by the hooks.
    pub kind: String,
    /// The exact value or path setup owns.
    pub value: String,
    /// Whether installation committed this integration.
    pub applied: bool,
}

/// A model file setup downloaded, independent of component rollback.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Model {
    /// Its installed path.
    pub file: String,
    /// Its source repository.
    pub repo: String,
    /// Its immutable revision.
    pub revision: String,
    /// Its verified content hash.
    pub sha256: String,
}

/// The last fully committed installation state.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Receipt {
    /// The receipt format, currently one.
    pub schema: u32,
    /// The canonical installation root.
    pub root: PathBuf,
    /// The release channel.
    pub channel: String,
    /// An update pin, cleared explicitly by unpin.
    pub pinned: Option<String>,
    /// The committed hardware selection.
    pub detect: Detection,
    /// Installed components indexed by name.
    pub components: BTreeMap<String, Component>,
    /// Installed runtime links as destination/source pairs.
    pub links: Vec<(PathBuf, PathBuf)>,
    /// Committed OS integrations.
    pub effects: Vec<Effect>,
    /// Model files created by setup.
    pub models: Vec<Model>,
    /// The last committed transaction ID.
    pub last_txn: String,
}

impl Receipt {
    /// An empty installation state rooted at the chosen destination.
    pub fn empty(root: PathBuf, channel: String) -> Self {
        Self {
            schema: 1,
            root,
            channel,
            pinned: None,
            detect: Detection::default(),
            components: BTreeMap::new(),
            links: Vec::new(),
            effects: Vec::new(),
            models: Vec::new(),
            last_txn: String::new(),
        }
    }
}

/// Whether an OS integration is being installed or removed.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// Apply the owned integration.
    Apply,
    /// Remove the owned integration.
    Remove,
}

/// One reversible mutation or a verification barrier.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Step {
    /// Atomically move an owned path.
    Rename {
        /// Existing source.
        from: PathBuf,
        /// Destination on the same filesystem.
        to: PathBuf,
    },
    /// Link a runtime into a component, falling back to a durable copy.
    Link {
        /// Runtime source.
        from: PathBuf,
        /// Component destination.
        to: PathBuf,
    },
    /// Change an idempotent OS integration.
    Effect {
        /// Integration type.
        kind: String,
        /// Owned value.
        value: String,
        /// Desired operation.
        action: Action,
    },
    /// Verify a changed executable before committing its receipt.
    SelfTest {
        /// Component to verify.
        component: String,
    },
    /// Move one recorded file or link to transaction trash.
    Delete {
        /// Recorded file or link, never a directory.
        path: PathBuf,
    },
}

/// Pre-step observations make no-ops and partially applied effects reversible.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Before {
    /// Whether a move source existed before the operation.
    #[serde(default)]
    pub source_existed: bool,
    /// A pre-existing destination must never be overwritten.
    #[serde(default)]
    pub destination_existed: bool,
    /// Whether an effect was present before its operation.
    pub effect_present: Option<bool>,
    /// Exact external state when presence alone cannot preserve formatting or ordering.
    #[serde(default)]
    pub effect_state: Option<String>,
    /// Whether a link already existed before its operation.
    pub link_existed: bool,
}

/// Durable progress; only the prepared step at done can be uncertain.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Journal {
    /// Unique transaction identity, also used by setup's self-test children.
    pub txn: String,
    /// SHA-256 of the exact starting receipt bytes, or the empty string's bytes.
    pub from_receipt: String,
    /// The receipt committed only after all steps complete.
    pub to: Receipt,
    /// Ordered reversible steps.
    pub steps: Vec<Step>,
    /// Number of completed, fsynced steps.
    pub done: usize,
    /// The step whose pre-state was fsynced before its mutation.
    #[serde(default)]
    pub prepared: Option<usize>,
    /// Persisted before undo so a crash during rollback continues backward.
    #[serde(default)]
    pub rolling_back: bool,
    /// Before-state for each prepared step, including completed ones.
    #[serde(default)]
    pub before: BTreeMap<usize, Before>,
    /// Directories absent before the transaction, removed if rollback leaves them empty.
    #[serde(default)]
    pub created_dirs: Vec<PathBuf>,
}

impl Journal {
    /// Starts an unexecuted plan against an exact committed receipt digest.
    pub fn new(txn: String, from_receipt: String, mut to: Receipt, steps: Vec<Step>) -> Self {
        to.last_txn = txn.clone();
        Self {
            txn,
            from_receipt,
            to,
            steps,
            done: 0,
            prepared: None,
            rolling_back: false,
            before: BTreeMap::new(),
            created_dirs: Vec::new(),
        }
    }
}

/// The platform hooks; tests use filesystem-backed fakes instead of real user settings.
pub trait Hooks {
    /// Apply an integration idempotently.
    fn apply(&mut self, kind: &str, value: &str) -> Result<()>;
    /// Remove only the integration setup owns.
    fn remove(&mut self, kind: &str, value: &str) -> Result<()>;
    /// Observe the exact integration's current state.
    fn present(&mut self, kind: &str, value: &str) -> Result<bool>;
    /// Captures external state that needs more than a presence bit for exact rollback.
    fn snapshot(&mut self, _kind: &str, _value: &str) -> Result<Option<String>> {
        Ok(None)
    }
    /// Restores a captured state; presence-only hooks never need this operation.
    fn restore(&mut self, _kind: &str, _value: &str, _state: &str) -> Result<()> {
        Err(Error::Refused(
            "this hook cannot restore a captured effect state".into(),
        ))
    }
    /// Verify a new component, returning failures that require rollback.
    fn self_test(&mut self, component: &str) -> Result<()>;
}

/// The recovery direction taken while holding the exclusive installation lock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recovery {
    /// No transaction was pending.
    None,
    /// The target receipt was committed.
    Forward,
    /// The exact starting receipt remains committed.
    Back,
}

/// Exclusive ownership of an installation's mutations until this value is dropped.
pub struct Transaction {
    root: PathBuf,
    _lock: File,
}
