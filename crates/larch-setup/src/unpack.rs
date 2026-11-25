//! Archive extraction and recovery verification share one complete, bounded inventory.
use crate::{
    manifest::{self, Artifact, Error, Result},
    txn::{Journal, Step},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io,
    path::{Path, PathBuf},
};
/// Domain separation distinguishes symlink metadata from same-sized regular file contents.
pub const SYMLINK_DOMAIN: &[u8] = b"symlink\0";
fn refused(reason: impl Into<String>) -> Error {
    Error::Refused(reason.into())
}
fn directories(files: &BTreeMap<String, (String, u64)>) -> BTreeSet<String> {
    files
        .keys()
        .flat_map(|name| {
            Path::new(name)
                .ancestors()
                .skip(1)
                .filter_map(|p| p.to_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.replace('\\', "/"))
                .collect::<Vec<_>>()
        })
        .collect()
}
fn name(bytes: &[u8]) -> Result<String> {
    let text = std::str::from_utf8(bytes).map_err(|_| refused("archive path is not UTF-8"))?;
    Ok(manifest::relative_path(text)?
        .to_string_lossy()
        .replace('\\', "/"))
}
fn parent_dirs(root: &Path, relative: &Path) -> Result<()> {
    let mut path = root.to_owned();
    if let Some(parent) = relative.parent() {
        for part in parent.components() {
            path.push(part);
            match fs::symlink_metadata(&path) {
                Ok(m) if m.is_dir() => {}
                Ok(_) => {
                    return Err(refused(format!(
                        "archive parent is not a directory: {}",
                        path.display()
                    )));
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => fs::create_dir(&path)?,
                Err(e) => return Err(e.into()),
            }
        }
    }
    Ok(())
}
/// Canonicalizes a relative link and proves that its resolution cannot escape the component.
pub fn link_target(link: &str, target: &str) -> Result<(String, PathBuf)> {
    if target.is_empty()
        || target.starts_with('/')
        || target.contains(['\\', ':'])
        || target.chars().any(char::is_control)
    {
        return Err(refused(format!("unsafe link target {target:?}")));
    }
    let mut normalized: Vec<&str> = Vec::new();
    let mut named = false;
    for p in target.split('/') {
        match p {
            "" | "." => {}
            ".." if named => {
                // A named segment might be another symlink; lexical cancellation cannot prove safety.
                return Err(refused(
                    "link target has ambiguous traversal through a named directory",
                ));
            }
            ".." => normalized.push(p),
            _ => {
                named = true;
                manifest::relative_path(p)?;
                normalized.push(p);
            }
        }
    }
    let normalized = if normalized.is_empty() {
        ".".into()
    } else {
        normalized.join("/")
    };
    let mut resolved: Vec<_> = link.split('/').collect();
    resolved.pop();
    for p in normalized.split('/') {
        match p {
            "." => {}
            ".." => {
                if resolved.pop().is_none() {
                    return Err(refused(format!("link {link:?} escapes its component")));
                }
            }
            _ => resolved.push(p),
        }
    }
    let resolved = PathBuf::from(resolved.join("/"));
    Ok((normalized, resolved))
}
/// Hashes normalized symlink metadata, keeping its size equal to the target text length.
pub fn symlink_hash(target: &str) -> String {
    let mut bytes = SYMLINK_DOMAIN.to_vec();
    bytes.extend_from_slice(target.as_bytes());
    larch_fetch::sha256_bytes(&bytes)
}
fn symlink(target: &str, to: &Path, directory: bool) -> Result<()> {
    #[cfg(unix)]
    {
        let _ = directory;
        std::os::unix::fs::symlink(target, to)?;
    }
    #[cfg(windows)]
    {
        if directory {
            std::os::windows::fs::symlink_dir(target, to)?;
        } else {
            std::os::windows::fs::symlink_file(target, to)?;
        }
    }
    Ok(())
}
/// Checks every file, link, size and hash, rejecting extras and symlinks outside the root.
pub fn verify_tree(root: &Path, files: &[(String, String, u64)]) -> Result<()> {
    verify(root, files, &BTreeSet::new())
}
fn verify(root: &Path, files: &[(String, String, u64)], extras: &BTreeSet<PathBuf>) -> Result<()> {
    if !fs::symlink_metadata(root)?.is_dir() {
        return Err(refused("component root is not a directory"));
    }
    let canonical = fs::canonicalize(root)?;
    let expected = manifest::inventory(files)?;
    let dirs = directories(&expected);
    let mut seen = BTreeSet::new();
    let mut pending = vec![root.to_owned()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            let rel = path
                .strip_prefix(root)
                .map_err(|_| refused("component path escapes root"))?;
            let text = rel
                .to_str()
                .ok_or_else(|| refused("component path is not UTF-8"))?
                .replace('\\', "/");
            let text = name(text.as_bytes())?;
            let meta = fs::symlink_metadata(&path)?;
            if extras.contains(&path) {
                if meta.is_file() {
                    continue;
                }
                return Err(refused("recorded runtime link is not a file"));
            }
            if meta.is_dir() {
                if !dirs.contains(&text) {
                    return Err(refused(format!("unlisted component directory: {text}")));
                }
                pending.push(path);
                continue;
            }
            let (hash, size) = expected
                .get(&text)
                .ok_or_else(|| refused(format!("unlisted component file: {text}")))?;
            let (actual, bytes) = if meta.is_symlink() {
                let target = fs::read_link(&path)?;
                let target = target
                    .to_str()
                    .ok_or_else(|| refused("symlink target is not UTF-8"))?
                    .replace('\\', "/");
                let (normalized, _) = link_target(&text, &target)?;
                let resolved = fs::canonicalize(&path)?;
                if !resolved.starts_with(&canonical) {
                    return Err(refused(format!("symlink {text} escapes its component")));
                }
                (symlink_hash(&normalized), normalized.len() as u64)
            } else if meta.is_file() {
                if meta.len() != *size {
                    return Err(refused(format!("component size mismatch: {text}")));
                }
                (larch_fetch::sha256_file(&path)?, meta.len())
            } else {
                return Err(refused(format!("unsupported component file type: {text}")));
            };
            if bytes != *size || !actual.eq_ignore_ascii_case(hash) {
                return Err(refused(format!(
                    "component size or SHA-256 mismatch: {text}"
                )));
            }
            seen.insert(text);
        }
    }
    if seen.len() != expected.len() {
        let missing = expected.keys().find(|p| !seen.contains(*p)).unwrap();
        return Err(refused(format!("component file is missing: {missing}")));
    }
    Ok(())
}
fn sync_tree(root: &Path, files: &[(String, String, u64)]) -> Result<()> {
    let expected = manifest::inventory(files)?;
    for path in expected.keys() {
        let path = root.join(path);
        if fs::symlink_metadata(&path)?.is_file() {
            File::options()
                .read(true)
                .write(true)
                .open(path)?
                .sync_all()?;
        }
    }
    #[cfg(unix)]
    {
        let mut dirs: Vec<_> = directories(&expected)
            .into_iter()
            .map(|p| root.join(p))
            .collect();
        dirs.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
        dirs.push(root.to_owned());
        for dir in dirs {
            File::open(dir)?.sync_all()?;
        }
    }
    Ok(())
}

/// Verifies the compressed archive, then creates only listed files and internal links.
pub fn unpack(archive: &Path, artifact: &Artifact, dest: &Path) -> Result<()> {
    let expected = manifest::inventory(&artifact.files)?;
    if fs::metadata(archive)?.len() != artifact.size
        || !larch_fetch::sha256_file(archive)?.eq_ignore_ascii_case(&artifact.sha256)
    {
        return Err(refused("archive size or SHA-256 mismatch"));
    }
    match fs::symlink_metadata(dest) {
        Ok(m) if m.is_dir() => {
            if fs::read_dir(dest)?.next().is_some() {
                verify_tree(dest, &artifact.files)?;
                sync_tree(dest, &artifact.files)?;
                return Ok(());
            }
        }
        Ok(_) => return Err(refused("staging destination is not a directory")),
        Err(e) if e.kind() == io::ErrorKind::NotFound => crate::txn::create_dir_all(dest)?,
        Err(e) => return Err(e.into()),
    }
    let result = (|| -> Result<()> {
        let mut reader = zstd::stream::read::Decoder::new(File::open(archive)?)?;
        // ponytail: 128 MiB windows and 16 KiB metadata/entry; raise only for a reviewed release that needs more.
        reader.window_log_max(27)?;
        let allowed_dirs = directories(&expected);
        let payload = expected.values().try_fold(0u64, |sum, (_, size)| {
            sum.checked_add(*size)
                .ok_or_else(|| refused("unpacked size overflow"))
        })?;
        let overhead = (expected.len() as u64)
            .checked_add(allowed_dirs.len() as u64)
            .and_then(|n| n.checked_add(2))
            .and_then(|n| n.checked_mul(16 * 1024))
            .ok_or_else(|| refused("archive metadata size overflow"))?;
        let limit = payload
            .checked_add(overhead)
            .ok_or_else(|| refused("archive size overflow"))?;
        let mut tar = tar::Archive::new(std::io::Read::take(reader, limit));
        let mut seen = BTreeSet::new();
        let mut seen_dirs = BTreeSet::new();
        let mut symbolic = Vec::new();
        let mut hard = Vec::new();
        for entry in tar.entries()? {
            let mut entry = entry?;
            let kind = entry.header().entry_type();
            let raw = entry.path_bytes();
            let raw =
                std::str::from_utf8(&raw).map_err(|_| refused("archive path is not UTF-8"))?;
            if kind.is_dir() && matches!(raw, "." | "./") {
                if !seen_dirs.insert(String::new()) {
                    return Err(refused("duplicate archive root directory"));
                }
                continue;
            }
            let path = name(if kind.is_dir() {
                raw.trim_end_matches('/').as_bytes()
            } else {
                raw.as_bytes()
            })?;
            let relative = Path::new(&path);
            if kind.is_dir() {
                if !allowed_dirs.contains(&path) || !seen_dirs.insert(path.clone()) {
                    return Err(refused(format!(
                        "unlisted or duplicate archive directory: {path}"
                    )));
                }
                parent_dirs(dest, relative)?;
                let to = dest.join(relative);
                match fs::symlink_metadata(&to) {
                    Ok(m) if m.is_dir() => {}
                    Ok(_) => return Err(refused("archive directory collides with a file")),
                    Err(e) if e.kind() == io::ErrorKind::NotFound => fs::create_dir(to)?,
                    Err(e) => return Err(e.into()),
                }
                continue;
            }
            let (hash, size) = expected
                .get(&path)
                .ok_or_else(|| refused(format!("archive contains an extra file: {path}")))?;
            if !seen.insert(path.clone()) {
                return Err(refused(format!("duplicate archive file: {path}")));
            }
            parent_dirs(dest, relative)?;
            if kind.is_file() {
                if entry.size() != *size {
                    return Err(refused(format!("archive file size mismatch: {path}")));
                }
                let mut file = File::options()
                    .create_new(true)
                    .write(true)
                    .open(dest.join(relative))?;
                if io::copy(&mut entry, &mut file)? != *size {
                    return Err(refused(format!("short archive file: {path}")));
                }
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    file.set_permissions(fs::Permissions::from_mode(
                        (entry.header().mode()? & 0o755) | 0o600,
                    ))?;
                }
                file.sync_all()?;
            } else if kind.is_symlink() {
                let target = entry
                    .link_name_bytes()
                    .ok_or_else(|| refused("symlink has no target"))?;
                let target = std::str::from_utf8(&target)
                    .map_err(|_| refused("symlink target is not UTF-8"))?;
                let (target, resolved) = link_target(&path, target)?;
                if target.len() as u64 != *size || !symlink_hash(&target).eq_ignore_ascii_case(hash)
                {
                    return Err(refused(format!("symlink metadata mismatch: {path}")));
                }
                symbolic.push((path, target, resolved));
            } else if kind.is_hard_link() {
                let target = entry
                    .link_name_bytes()
                    .ok_or_else(|| refused("hard link has no target"))?;
                hard.push((path, name(&target)?));
            } else {
                return Err(refused(format!("unsupported archive entry type: {path}")));
            }
        }
        if tar.into_inner().limit() == 0 {
            return Err(refused("decompressed archive exceeds its inventory bounds"));
        }
        while !hard.is_empty() {
            let before = hard.len();
            let mut next = Vec::new();
            for (path, target) in hard {
                let target = dest.join(target);
                if fs::symlink_metadata(&target).is_ok_and(|m| m.is_file()) {
                    fs::hard_link(target, dest.join(path))?;
                } else {
                    next.push((
                        path,
                        target
                            .strip_prefix(dest)
                            .unwrap()
                            .to_string_lossy()
                            .replace('\\', "/"),
                    ));
                }
            }
            if next.len() == before {
                return Err(refused(
                    "hard link target is missing, cyclic or not a regular file",
                ));
            }
            hard = next;
        }
        let canonical = fs::canonicalize(dest)?;
        while !symbolic.is_empty() {
            let before = symbolic.len();
            let mut next = Vec::new();
            for (path, target, resolved) in symbolic {
                let resolved = dest.join(&resolved);
                match fs::canonicalize(&resolved) {
                    Ok(real) => {
                        if !real.starts_with(&canonical) {
                            return Err(refused("symlink target escapes its component"));
                        }
                        symlink(&target, &dest.join(path), real.is_dir())?;
                    }
                    Err(e) if e.kind() == io::ErrorKind::NotFound => next.push((
                        path,
                        target,
                        resolved.strip_prefix(dest).unwrap().to_owned(),
                    )),
                    Err(e) => return Err(e.into()),
                }
            }
            if next.len() == before {
                return Err(refused("symlink target is missing or cyclic"));
            }
            symbolic = next;
        }
        verify_tree(dest, &artifact.files)?;
        sync_tree(dest, &artifact.files)
    })();
    if let Err(error) = result {
        if let Err(cleanup) = fs::remove_dir_all(dest) {
            return Err(refused(format!(
                "{error}; staging cleanup failed: {cleanup}"
            )));
        }
        return Err(error);
    }
    Ok(())
}
/// Validates changed components wherever a prepared directory swap currently holds them.
pub fn staged_ok(journal: &Journal) -> Result<bool> {
    for c in journal.to.components.values() {
        let sources: Vec<_> = journal
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Rename { from, to } if to == &c.dest => Some(from),
                _ => None,
            })
            .collect();
        if sources.is_empty() {
            continue;
        }
        let extras: BTreeSet<_> = journal
            .to
            .links
            .iter()
            .filter(|(p, _)| p.starts_with(&c.dest))
            .map(|(p, _)| p.clone())
            .collect();
        let valid = std::iter::once(&c.dest).chain(sources).any(|root| {
            let mapped: BTreeSet<_> = extras
                .iter()
                .filter_map(|p| p.strip_prefix(&c.dest).ok().map(|rel| root.join(rel)))
                .collect();
            verify(root, &c.files, &mapped).is_ok()
        });
        if !valid {
            return Ok(false);
        }
    }
    for (index, step) in journal.steps.iter().enumerate() {
        if index >= journal.done {
            break;
        }
        if let Step::Link { from, to } = step
            && (!fs::metadata(to).is_ok_and(|m| m.is_file())
                || !larch_fetch::sha256_file(from)
                    .and_then(|hash| larch_fetch::sha256_file(to).map(|other| hash == other))
                    .unwrap_or(false))
        {
            return Ok(false);
        }
    }
    Ok(true)
}
