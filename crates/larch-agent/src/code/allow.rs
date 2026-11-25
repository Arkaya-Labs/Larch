//! What the user lets code mode change unasked in a folder it edits in place: files, folders, or
//! everything ("silent"), kept per folder in `<data>/agent/allowed.json` across runs. Anything
//! else the agent would write asks first, and the user's answer may widen what is allowed.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool::ToolError;

/// One folder's allowances.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allowed {
    /// Everything runs unasked: each edit, new file and command.
    #[serde(default)]
    pub silent: bool,
    /// Files, relative to the folder, `/` between names.
    #[serde(default)]
    pub files: BTreeSet<String>,
    /// Folders whose files, at any depth, are allowed; "." is the whole folder. Not commands:
    /// one can write any file, so only `silent` lets them run unasked.
    #[serde(default)]
    pub folders: BTreeSet<String>,
}

impl Allowed {
    /// Whether a write to `path` (relative, `/` between names) runs unasked; a command, which
    /// names no path, does only in silence.
    pub fn covers(&self, path: Option<&str>) -> bool {
        let under = |dir: &String| {
            let p = path.unwrap_or_default();
            dir == "."
                || p.strip_prefix(dir.as_str())
                    .is_some_and(|r| r.starts_with('/'))
        };
        self.silent
            || path.is_some_and(|p| self.files.contains(p) || self.folders.iter().any(under))
    }

    /// Widens the allowances as `decision` says for `ask`: its file, its file's folder, or
    /// everything; whether anything changed, so the caller saves.
    pub fn take(&mut self, ask: &Ask, decision: Decision) -> bool {
        match (decision, &ask.path) {
            (Decision::All, _) => !std::mem::replace(&mut self.silent, true),
            (Decision::File, Some(path)) => self.files.insert(path.clone()),
            (Decision::Folder, Some(_)) => self.folders.insert(ask.folder().into()),
            _ => false,
        }
    }
}

/// A write code mode asks the user to allow before it runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ask {
    /// The tool: edit, write_file or bash.
    pub name: String,
    /// Its arguments, as the call made them.
    pub args: Value,
    /// The file it writes, relative to the folder, `/` between names; None for a command.
    pub path: Option<String>,
}

impl Ask {
    /// The folder `Decision::Folder` allows: the file's own, "." at the folder's top.
    pub fn folder(&self) -> &str {
        let path = self.path.as_deref().unwrap_or_default();
        path.rsplit_once('/').map_or(".", |(dir, _)| dir)
    }
}

/// The user's answer to an `Ask`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// This call only.
    Once,
    /// This call, and every later write to its file.
    File,
    /// This call, and every later write in its file's folder.
    Folder,
    /// This call, and everything after it in this folder: silent.
    All,
    /// Not this call; the model hears so and goes on.
    No,
    /// Not this call, and the run ends.
    Stop,
}

/// The file every folder's allowances live in.
fn file(data_dir: &Path) -> PathBuf {
    data_dir.join("agent").join("allowed.json")
}

/// Every folder's allowances; none before the first is kept.
fn all(data_dir: &Path) -> Result<BTreeMap<String, Allowed>, ToolError> {
    let path = file(data_dir);
    let io = |e| ToolError::Io(path.display().to_string(), e);
    match fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text)
            .map_err(|e| ToolError::Invalid(format!("'{}': {e}", path.display()))),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(BTreeMap::new()),
        Err(e) => Err(io(e)),
    }
}

/// `folder`'s allowances, by its real path.
pub fn load(data_dir: &Path, folder: &Path) -> Result<Allowed, ToolError> {
    Ok(all(data_dir)?
        .remove(&folder.display().to_string())
        .unwrap_or_default())
}

/// Keeps `allowed` as `folder`'s, the others as they were; none at all forgets the folder.
pub fn save(data_dir: &Path, folder: &Path, allowed: &Allowed) -> Result<(), ToolError> {
    let mut every = all(data_dir)?;
    let key = folder.display().to_string();
    match *allowed == Allowed::default() {
        true => every.remove(&key),
        false => every.insert(key, allowed.clone()),
    };
    let path = file(data_dir);
    let io = |e| ToolError::Io(path.display().to_string(), e);
    fs::create_dir_all(path.parent().expect("a file in a folder")).map_err(io)?;
    let text = serde_json::to_string_pretty(&every).expect("allowances serialize");
    crate::write_atomic(&path, &text, true).map_err(io)
}
