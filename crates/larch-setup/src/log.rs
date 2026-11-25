//! Bounded append-only installer diagnostics with rotation and credential redaction.
use crate::{
    manifest::{Error, Result},
    txn::{create_dir_all, sync_parent},
};
use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};
/// Maximum size of each retained log before rotation.
pub const MAX_BYTES: u64 = 1024 * 1024;
/// Removes credentials and control characters before either logging or printing.
pub fn clean(text: &str, token: Option<&str>) -> String {
    let text = if let Some(token) = token.filter(|t| !t.is_empty()) {
        text.replace(token, "[redacted]")
    } else {
        text.into()
    };
    let mut line = String::new();
    for c in text.chars() {
        if line.len() + c.len_utf8() > 64 * 1024 {
            line.push_str(" [truncated]");
            break;
        }
        line.push(if c.is_control() { ' ' } else { c });
    }
    line
}
/// One installation's serialized diagnostic writer; tokens remain in memory only.
pub struct Log {
    root: PathBuf,
    token: Option<String>,
}
fn regular(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(m) => {
            if !m.is_file() {
                return Err(Error::Refused(
                    "setup log path is not a regular file".into(),
                ));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if m.nlink() != 1 {
                    return Err(Error::Refused("setup log has another hard link".into()));
                }
            }
            Ok(())
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}
impl Log {
    /// Uses the private-release token only to redact accidental error text.
    pub fn new(root: &Path) -> Self {
        Self::with_token(root, env::var("LARCH_RELEASES_TOKEN").ok())
    }
    /// Uses an explicit credential for clients and tests without changing the environment.
    pub fn with_token(root: &Path, token: Option<String>) -> Self {
        Self {
            root: root.into(),
            token,
        }
    }
    /// Appends and syncs one sanitized line, retaining at most three older logs.
    pub fn line(&self, text: &str) -> Result<()> {
        create_dir_all(&self.root)?;
        regular(&self.root.join("setup.log.lock"))?;
        let guard = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.root.join("setup.log.lock"))?;
        guard.lock()?;
        let path = self.root.join("setup.log");
        for p in std::iter::once(path.clone())
            .chain((1..=3).map(|n| self.root.join(format!("setup.log.{n}"))))
        {
            regular(&p)?;
        }
        let text = clean(text, self.token.as_deref());
        let size = match fs::metadata(&path) {
            Ok(m) => m.len(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => 0,
            Err(e) => return Err(e.into()),
        };
        if size + text.len() as u64 + 1 > MAX_BYTES {
            let oldest = self.root.join("setup.log.3");
            match fs::remove_file(oldest) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
                _ => {}
            }
            for n in (1..=2).rev() {
                let from = self.root.join(format!("setup.log.{n}"));
                match fs::rename(from, self.root.join(format!("setup.log.{}", n + 1))) {
                    Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
                    _ => {}
                }
            }
            match fs::rename(&path, self.root.join("setup.log.1")) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
                _ => {}
            }
            sync_parent(&path)?;
        }
        let mut file = File::options().create(true).append(true).open(&path)?;
        writeln!(file, "{text}")?;
        file.sync_all()?;
        sync_parent(&path)?;
        Ok(())
    }
    /// Prints exactly one actionable failure line even when the log itself is unavailable.
    pub fn failure(&self, what: &str, state: &str, next: &str) {
        let mut line = clean(
            &format!("{what}; state: {state}; next: {next}"),
            self.token.as_deref(),
        );
        if let Err(error) = self.line(&line) {
            line.push_str(&clean(
                &format!("; setup log unavailable: {error}"),
                self.token.as_deref(),
            ));
        }
        eprintln!("{line}");
    }
}
