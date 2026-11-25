use super::*;
use crate::plan::Previous;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io,
    path::{Path, PathBuf},
};
/// Reads committed state without creating installation or user directories.
pub(crate) fn receipt(root: &Path) -> Result<Option<Receipt>> {
    match fs::read(root.join("receipt.json")) {
        Ok(b) => Ok(Some(serde_json::from_slice(&b)?)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}
/// Captures existing paths without following directory symlinks.
pub(crate) fn snapshot(
    root: &Path,
    roots: &Roots,
    target: &str,
    txn: &str,
    os: Os,
) -> Result<Paths> {
    let mut existing = BTreeSet::new();
    let mut pending = vec![root.to_owned()];
    if os == Os::Macos {
        pending.push(roots.home.join("Applications/Larch.app"));
        pending.push(roots.home.join("Applications/.larch-staging"));
    }
    while let Some(p) = pending.pop() {
        match fs::symlink_metadata(&p) {
            Ok(m) => {
                existing.insert(p.clone());
                if m.is_dir() {
                    for e in fs::read_dir(p)? {
                        pending.push(e?.path());
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    let mut previous = BTreeMap::new();
    for name in ["cli", "app", "setup"] {
        let p = root.join(".previous").join(format!("{name}.json"));
        if existing.contains(&p) {
            let before: Previous = serde_json::from_slice(&fs::read(p)?)?;
            previous.insert(name.into(), before);
        }
    }
    Ok(Paths {
        root: root.into(),
        home: roots.home.clone(),
        target: target.into(),
        txn: txn.into(),
        existing,
        previous,
    })
}
/// Finds the newest committed component release for channel-level downgrade checks.
pub(crate) fn version(receipt: &Receipt) -> Result<Option<String>> {
    let mut newest: Option<String> = None;
    for c in receipt.components.values() {
        manifest::accept_version(None, &c.version, true)?;
        if match &newest {
            Some(old) => manifest::compare_versions(&c.version, old)?.is_gt(),
            None => true,
        } {
            newest = Some(c.version.clone());
        }
    }
    Ok(newest)
}
fn regular(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|m| m.is_file())
}
/// Reuses only an archive with the same canonical name, under the installation lock.
pub(crate) fn resume(t: &Transaction, dest: &Path, a: &manifest::Artifact) -> Result<()> {
    if dest.exists() || larch_fetch::part_path(dest).exists() {
        return Ok(());
    }
    let base = t.root().join(".staging");
    if !base.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(&base)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() || entry.path() == dest.parent().unwrap() {
            continue;
        }
        let old = entry.path().join(dest.file_name().unwrap());
        t.check_path(&old)?;
        if regular(&old)
            && fs::metadata(&old)?.len() == a.size
            && larch_fetch::sha256_file(&old)?.eq_ignore_ascii_case(&a.sha256)
        {
            fs::rename(&old, dest)?;
            txn::sync_parent(&old)?;
            txn::sync_parent(dest)?;
            return Ok(());
        }
        let part = larch_fetch::part_path(&old);
        let mut etag = part.as_os_str().to_owned();
        etag.push(".etag");
        let etag = PathBuf::from(etag);
        if regular(&part) && regular(&etag) {
            let to = larch_fetch::part_path(dest);
            let mut tag = to.as_os_str().to_owned();
            tag.push(".etag");
            fs::rename(&part, &to)?;
            fs::rename(&etag, PathBuf::from(tag))?;
            txn::sync_parent(&part)?;
            txn::sync_parent(&to)?;
            return Ok(());
        }
    }
    Ok(())
}
/// Removes empty directories only; foreign files and symlinks survive.
pub(crate) fn empty_dirs(root: &Path) -> Result<()> {
    if !fs::symlink_metadata(root).is_ok_and(|m| m.is_dir()) {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            empty_dirs(&entry.path())?;
        }
    }
    match fs::remove_dir(root) {
        Ok(()) => Ok(()),
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::DirectoryNotEmpty
            ) =>
        {
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}
/// Reports every remaining non-directory without following a symlink.
pub(crate) fn leftovers(root: &Path) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            leftovers(&entry.path())?;
        } else {
            eprintln!("left user or unrecorded file: {}", entry.path().display());
        }
    }
    Ok(())
}
/// Acquires an existing lock shared for read-only cache and status work.
pub(crate) fn shared(root: &Path) -> Result<File> {
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(root.join(".lock"))?;
    f.lock_shared()?;
    Ok(f)
}
/// Cleans known staging downloads after commit, leaving anything unrecognized.
pub(crate) fn clean_downloads(paths: &[PathBuf]) -> Result<()> {
    for p in paths {
        for path in [p.clone(), larch_fetch::part_path(p)] {
            match fs::remove_file(&path) {
                Ok(()) => txn::sync_parent(&path)?,
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        let mut tag = larch_fetch::part_path(p).into_os_string();
        tag.push(".etag");
        match fs::remove_file(PathBuf::from(tag)) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
