use crate::{
    cli::files,
    detect,
    effects::Roots,
    fetch::{Client, PointerDocument},
    manifest::{self, Error, Result},
    plan::{self, Request},
    txn,
};
use serde::{Deserialize, Serialize};
use std::{fs, fs::OpenOptions};
/// A verified channel change and the archives needed by installed components.
#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateCheck {
    /// Newer release, absent when no upgrade is offered.
    pub version: Option<String>,
    /// Compressed bytes for changed components and runtimes.
    pub download_size: u64,
    /// Names of components whose archives would change.
    pub components: Vec<String>,
}
/// Checks signed channel metadata with ETags; never changes receipt.json or installed programs.
pub fn check() -> Result<UpdateCheck> {
    let root = larch_fetch::install_root().map_err(|e| Error::Refused(e.to_string()))?;
    let _guard = files::shared(&root)?;
    let root = fs::canonicalize(root)?;
    let lock_path = root.join("check-cache.lock");
    if fs::symlink_metadata(&lock_path).is_ok_and(|m| !m.is_file()) {
        return Err(Error::Refused("check cache lock is not a file".into()));
    }
    let cache_guard = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock_path)?;
    cache_guard.lock()?;
    let receipt =
        files::receipt(&root)?.ok_or_else(|| Error::Refused("nothing is installed".into()))?;
    let cached = match fs::read(root.join("check-cache.json")) {
        Ok(b) => Some(serde_json::from_slice::<PointerDocument>(&b)?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };
    let mut client = Client::from_env()?;
    let head = client.pointer(&receipt.channel, cached.as_ref())?;
    txn::atomic_write(&root.join("check-cache.json"), &serde_json::to_vec(&head)?)?;
    let current = files::version(&receipt)?;
    if current.as_ref().is_some_and(|v| {
        manifest::compare_versions(&head.pointer.version, v).is_ok_and(|o| !o.is_gt())
    }) {
        return Ok(UpdateCheck {
            version: None,
            download_size: 0,
            components: vec![],
        });
    }
    let release = client.release_from_head(&receipt.channel, None, head)?;
    let machine = detect::detect_machine()?;
    let choice = detect::choose(&machine, &release.manifest, None)?;
    let roots = Roots::from_env()?;
    let paths = files::snapshot(&root, &roots, &machine.target, "check", machine.os)?;
    let mut unpinned = receipt;
    unpinned.pinned = None;
    let plan = plan::plan(
        Some(&unpinned),
        &release.manifest,
        &Request::Update,
        &choice,
        &paths,
        machine.os,
    )?;
    Ok(UpdateCheck {
        version: Some(release.manifest.version),
        download_size: plan.download_size,
        components: plan.downloads.into_iter().map(|d| d.0).collect(),
    })
}
