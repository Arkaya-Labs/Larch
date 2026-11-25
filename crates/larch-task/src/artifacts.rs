use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
// All artifact publication, pinning, backup and collection in this process share
// this lock. A service must hold the data-directory ownership lock across processes.
static ARTIFACT_LOCK: Mutex<()> = Mutex::new(());
pub(crate) fn lock() -> std::io::Result<MutexGuard<'static, ()>> {
    ARTIFACT_LOCK
        .lock()
        .map_err(|_| std::io::Error::other("artifact lock poisoned"))
}
/// Make a rename or create inside `dir` durable. Windows has no directory flush through std
/// (opening a directory needs FILE_FLAG_BACKUP_SEMANTICS and FlushFileBuffers needs a writable
/// handle), and NTFS journals the rename itself, so there it is a no-op.
pub(crate) fn sync_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}
pub(crate) fn valid_digest(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
/// Immutable SHA-256 objects. Publication follows file sync, atomic rename and directory sync.
pub struct ArtifactStore {
    pub(crate) root: PathBuf,
}
impl ArtifactStore {
    pub fn open(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let existed = root.as_ref().exists();
        fs::create_dir_all(root.as_ref())?;
        if !existed
            && let Some(parent) = root
                .as_ref()
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
        {
            sync_dir(parent)?;
        }
        Ok(Self {
            root: root.as_ref().to_path_buf(),
        })
    }
    /// Cross-process advisory lock shared by publication, pinning, backup and collection.
    pub(crate) fn lock_filesystem(&self) -> std::io::Result<File> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(self.root.join(".larch-artifacts.lock"))?;
        fs2::FileExt::lock_exclusive(&file)?;
        Ok(file)
    }
    /// Publish an object. Use SqliteStore::put_pinned for objects referenced by a run.
    pub fn put(&self, bytes: &[u8]) -> std::io::Result<String> {
        let _guard = lock()?;
        let _file_guard = self.lock_filesystem()?;
        if self.root.join(".larch-managed-cap").exists() {
            return Err(std::io::Error::other("managed storage requires put_pinned"));
        }
        self.put_locked(bytes)
    }
    pub(crate) fn put_locked(&self, bytes: &[u8]) -> std::io::Result<String> {
        let digest = format!("{:x}", Sha256::digest(bytes));
        let target = self.root.join(&digest);
        if target.exists() {
            self.read_locked(&digest)?;
            return Ok(digest);
        }
        let temp = self.root.join(format!(
            ".stage-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed),
        ));
        let outcome = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temp, &target)?;
            sync_dir(&self.root)?;
            Ok(digest)
        })();
        if outcome.is_err() {
            let _ = fs::remove_file(&temp);
        }
        outcome
    }
    pub fn read(&self, digest: &str) -> std::io::Result<Vec<u8>> {
        let _guard = lock()?;
        let _file_guard = self.lock_filesystem()?;
        self.read_locked(digest)
    }
    pub(crate) fn read_locked(&self, digest: &str) -> std::io::Result<Vec<u8>> {
        if !valid_digest(digest) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid artifact digest",
            ));
        }
        let mut bytes = Vec::new();
        File::open(self.root.join(digest))?.read_to_end(&mut bytes)?;
        if format!("{:x}", Sha256::digest(&bytes)) != digest {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "artifact checksum mismatch",
            ));
        }
        Ok(bytes)
    }
}
