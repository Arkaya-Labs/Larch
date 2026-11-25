use super::*;
use std::time::{Duration, Instant};

/// Atomically writes synced metadata; an interrupted temp is never a committed receipt.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let reserved = matches!(
        path.file_name().and_then(|s| s.to_str()),
        Some("receipt.json" | "journal.json")
    );
    let (temp, mut file) = loop {
        let mut name = path.as_os_str().to_owned();
        if reserved {
            name.push(".tmp");
        } else {
            name.push(format!(
                ".larch-{}-{}.tmp",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        }
        let temp = PathBuf::from(name);
        if reserved {
            match fs::remove_file(&temp) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
                _ => {}
            }
        }
        let mut options = File::options();
        options.write(true).create_new(true);
        #[cfg(unix)]
        if reserved {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&temp) {
            Ok(file) => break (temp, file),
            Err(e) if !reserved && e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    };
    let result = (|| -> Result<()> {
        match fs::metadata(path) {
            Ok(metadata) => file.set_permissions(metadata.permissions())?,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        rename_retry(&temp, path)?;
        sync_parent(path)?;
        Ok(())
    })();
    if let Err(error) = result {
        match fs::remove_file(&temp) {
            Err(cleanup) if cleanup.kind() != io::ErrorKind::NotFound => {
                return Err(Error::Refused(format!(
                    "{error}; temporary metadata cleanup failed: {cleanup}"
                )));
            }
            _ => {}
        }
        return Err(error);
    }
    Ok(())
}

/// Persists a namespace mutation on platforms with directory fsync.
pub fn sync_parent(path: &Path) -> io::Result<()> {
    #[cfg(not(unix))]
    let _ = path;
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

pub(super) fn exists(path: &Path) -> io::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

pub(super) fn mkdir(path: &Path) -> io::Result<()> {
    if path.is_dir() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        mkdir(parent)?;
    }
    match fs::create_dir(path) {
        Ok(()) => sync_parent(path),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists && path.is_dir() => Ok(()),
        Err(e) => Err(e),
    }
}

/// Creates each missing directory and persists its parent before proceeding.
pub fn create_dir_all(path: &Path) -> io::Result<()> {
    mkdir(path)
}

fn retry(mut operation: impl FnMut() -> io::Result<()>, windows: bool) -> io::Result<()> {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut pause = Duration::from_millis(20);
    loop {
        match operation() {
            Ok(()) => return Ok(()),
            Err(e)
                if Instant::now() < deadline
                    && (e.kind() == io::ErrorKind::ResourceBusy
                        || (windows && matches!(e.raw_os_error(), Some(32 | 5)))) =>
            {
                std::thread::sleep(pause.min(deadline.saturating_duration_since(Instant::now())));
                pause = (pause * 2).min(Duration::from_millis(250));
            }
            Err(e) => return Err(e),
        }
    }
}

pub(super) fn rename_retry(from: &Path, to: &Path) -> io::Result<()> {
    retry(|| fs::rename(from, to), cfg!(windows))
}

pub(super) fn same_file(from: &Path, to: &Path) -> Result<bool> {
    let (a, b) = match (fs::metadata(from), fs::metadata(to)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        (Err(e), _) | (_, Err(e)) => return Err(e.into()),
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(a.dev() == b.dev() && a.ino() == b.ino())
    }
    #[cfg(windows)]
    {
        // ponytail: use the file index once larch-sys exposes GetFileInformationByHandle.
        Ok(a.is_file()
            && b.is_file()
            && a.len() == b.len()
            && a.modified()? == b.modified()?
            && larch_fetch::sha256_file(from)? == larch_fetch::sha256_file(to)?)
    }
}

pub(super) fn existing_ancestor(path: &Path) -> Result<PathBuf> {
    for parent in path.ancestors() {
        if parent.exists() {
            return Ok(fs::canonicalize(parent)?);
        }
    }
    Err(Error::Refused(format!(
        "no existing filesystem parent for {}",
        path.display()
    )))
}

pub(super) fn same_volume(from: &Path, to: &Path) -> Result<bool> {
    let a = existing_ancestor(from)?;
    let b = existing_ancestor(to)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(fs::metadata(a)?.dev() == fs::metadata(b)?.dev())
    }
    #[cfg(windows)]
    {
        Ok(a.components().next() == b.components().next())
    }
}

pub(super) fn crash(index: usize, before: bool) {
    #[cfg(any(test, feature = "crash-hook"))]
    if std::env::var("LARCH_SETUP_CRASH_AT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        == Some(index)
        && (std::env::var("LARCH_SETUP_CRASH_MODE").as_deref() == Ok("before_journal")) == before
    {
        std::process::abort();
    }
    #[cfg(not(any(test, feature = "crash-hook")))]
    let _ = (index, before);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "linux")]
    #[test]
    fn filesystems_are_compared_before_a_rename() {
        assert!(!same_volume(Path::new("/proc/self"), &std::env::temp_dir()).unwrap());
    }
    #[test]
    fn sharing_violation_retries_twice_then_succeeds() {
        let mut calls = 0;
        retry(
            || {
                calls += 1;
                if calls <= 2 {
                    Err(io::Error::from_raw_os_error(32))
                } else {
                    Ok(())
                }
            },
            true,
        )
        .unwrap();
        assert_eq!(calls, 3);
    }
}
