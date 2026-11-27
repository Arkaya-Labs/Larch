//! Engine-free HTTP transport and per-user installation and model storage.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(any(windows, target_os = "macos"))]
use std::sync::Arc;
use std::{
    collections::BTreeMap,
    env, fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Condvar, Mutex, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc,
    },
    time::Duration,
};

const CHUNK: u64 = 4 << 20;

/// Whether a canonical executable belongs to an installation and must hold its lock.
pub fn needs_install_lock(
    exe: &Path,
    root: &Path,
    app_dest: Option<&Path>,
    env_txn: Option<&str>,
    journal_txn: Option<&str>,
) -> bool {
    let installed = exe.starts_with(root) || app_dest.is_some_and(|app| exe.starts_with(app));
    installed && !env_txn.is_some_and(|value| !value.is_empty() && Some(value) == journal_txn)
}

/// Holds the existing installation lock shared until the returned file is dropped.
pub fn hold_install_lock() -> io::Result<Option<File>> {
    let exe = fs::canonicalize(env::current_exe()?)?;
    let root = install_root().map_err(io::Error::other)?;
    let root = match fs::canonicalize(root) {
        Ok(root) => root,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    #[cfg(target_os = "macos")]
    let app_dest = env::var_os("HOME")
        .map(|home| PathBuf::from(home).join("Applications/Larch.app"))
        .map(fs::canonicalize)
        .transpose()
        .or_else(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                Ok(None)
            } else {
                Err(e)
            }
        })?;
    #[cfg(not(target_os = "macos"))]
    let app_dest: Option<PathBuf> = None;
    if !needs_install_lock(&exe, &root, app_dest.as_deref(), None, None) {
        return Ok(None);
    }
    let lock = match OpenOptions::new()
        .read(true)
        .write(true)
        .open(root.join(".lock"))
    {
        Ok(lock) => lock,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    // A malformed journal cannot grant a bypass; setup owns its recovery and error report.
    let journal_txn = match fs::read(root.join("journal.json")) {
        Ok(bytes) => serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| v.get("txn").and_then(|v| v.as_str()).map(str::to_owned)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => return Err(e),
    };
    let env_txn = env::var("LARCH_SETUP_SELFTEST").ok();
    if !needs_install_lock(
        &exe,
        &root,
        app_dest.as_deref(),
        env_txn.as_deref(),
        journal_txn.as_deref(),
    ) {
        return Ok(None);
    }
    match lock.try_lock_shared() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => {
            eprintln!("larch: an install or update is running; waiting for it to finish");
            lock.lock_shared()?;
        }
        Err(std::fs::TryLockError::Error(e)) => return Err(e),
    }
    Ok(Some(lock))
}

/// Nothing personal in the User-Agent; ureq follows redirects (resolve goes to a CDN). The OS's
/// trusted certificates and HTTPS_PROXY are used, so a machine behind a company proxy or an
/// HTTPS-scanning antivirus reaches the server as its browser does. ureq ignores NO_PROXY, so a server
/// on this machine (the tests' one) is never sent through the proxy.
pub fn agent(endpoint: &str) -> ureq::Agent {
    builder(endpoint).build()
}

/// [`agent`], answering a redirect with the redirect itself: the Hub's own answer to a resolve
/// carries headers (`x-repo-commit`) the CDN's does not.
pub fn agent_unfollowed(endpoint: &str) -> ureq::Agent {
    builder(endpoint).redirects(0).build()
}

fn builder(endpoint: &str) -> ureq::AgentBuilder {
    let host = endpoint
        .split("://")
        .nth(1)
        .unwrap_or(endpoint)
        .split(['/', '?'])
        .next()
        .unwrap_or("");
    let local =
        host.starts_with("127.") || host.starts_with("localhost") || host.starts_with("[::1]");
    let builder = ureq::AgentBuilder::new()
        .try_proxy_from_env(!local)
        .redirect_auth_headers(ureq::RedirectAuthHeaders::Never)
        .user_agent(concat!("larch/", env!("CARGO_PKG_VERSION")))
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(60))
        // A parallel download keeps PARALLEL connections busy; one idle connection per host (the
        // default) would make most range requests a new TLS handshake.
        .max_idle_connections_per_host(PARALLEL);
    // The OS's TLS where it has one ureq can use (see Cargo.toml); rustls if it cannot start.
    #[cfg(any(windows, target_os = "macos"))]
    let builder = match native_tls::TlsConnector::new() {
        Ok(tls) => builder.tls_connector(Arc::new(tls)),
        Err(_) => builder,
    };
    builder
}

/// The Hugging Face hub cache, where huggingface_hub, its CLI and the tools built on it keep
/// models, found as huggingface_hub finds it: `HF_HUB_CACHE`, the older `HUGGINGFACE_HUB_CACHE`,
/// `HF_HOME/hub`, `XDG_CACHE_HOME/huggingface/hub`, then `~/.cache/huggingface/hub` on every OS.
/// A leading `~` is the home folder. A relative value, which would resolve against whatever folder
/// the process started in, is passed over. None with no home folder to start from.
pub fn hf_cache() -> Option<PathBuf> {
    hf_cache_in(|k| env::var_os(k))
}

/// [`hf_cache`] with the environment `env`.
fn hf_cache_in(env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Option<PathBuf> {
    // A HOME set on Windows (MSYS2, Emacs) is not the one huggingface_hub's `~` reads.
    let home = env(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    let var = |k: &str| {
        let p = PathBuf::from(env(k).filter(|v| !v.is_empty())?);
        let p = match p.strip_prefix("~") {
            Ok(rest) => home.as_ref()?.join(rest),
            Err(_) => p,
        };
        p.is_absolute().then_some(p)
    };
    var("HF_HUB_CACHE")
        .or_else(|| var("HUGGINGFACE_HUB_CACHE"))
        .or_else(|| Some(var("HF_HOME")?.join("hub")))
        .or_else(|| Some(var("XDG_CACHE_HOME")?.join("huggingface").join("hub")))
        .or_else(|| {
            Some(
                home.as_ref()?
                    .join(".cache")
                    .join("huggingface")
                    .join("hub"),
            )
        })
}

/// The files named `name` in snapshots of the Hugging Face cache (`models--*/snapshots/*/`, or a
/// folder in one, where a split quant sits), one for each repo that has one: from the commit the
/// repo's `refs/main` names when that has it. How a command line runs a model the app downloaded,
/// by its file name. More than one is for the user to choose between (ggml-org and others name
/// their files alike), by path.
pub fn hf_cache_files(name: &str) -> Vec<PathBuf> {
    hf_cache().map_or_else(Vec::new, |cache| cache_files(&cache, name))
}

/// [`hf_cache_files`] in the cache `cache`.
fn cache_files(cache: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for repo in fs::read_dir(cache).into_iter().flatten().flatten() {
        let repo = repo.path();
        let is_repo =
            (repo.file_name().and_then(|n| n.to_str())).is_some_and(|n| n.starts_with("models--"));
        if !is_repo {
            continue;
        }
        let main = fs::read_to_string(repo.join("refs").join("main")).unwrap_or_default();
        let mut here = Vec::new();
        for snapshot in fs::read_dir(repo.join("snapshots"))
            .into_iter()
            .flatten()
            .flatten()
        {
            let snapshot = snapshot.path();
            let older = snapshot.file_name() != Some(main.trim().as_ref());
            let folders = (fs::read_dir(&snapshot).into_iter().flatten().flatten())
                .map(|e| e.path())
                .filter(|p| p.is_dir());
            for dir in std::iter::once(snapshot.clone()).chain(folders) {
                let file = dir.join(name);
                if file.is_file() {
                    here.push((older, file));
                }
            }
        }
        here.sort();
        found.extend(here.into_iter().next().map(|(_, file)| file));
    }
    found.sort();
    found
}

/// A remote file as a `Read`, fetched in lazy 4 MiB range requests. A server that
/// ignores ranges sends the whole file at the first request; it is read as far as needed.
pub struct Ranged<F> {
    get: F,
    source: String,
    what: String,
    at: u64,
    /// End of the range `body` holds.
    end: u64,
    len: u64,
    body: Option<Box<dyn Read + Send + Sync>>,
    /// Why a read failed, for the user rather than the GGUF parser's truncation error.
    error: Option<String>,
}

impl<F: FnMut(String) -> io::Result<ureq::Response>> Ranged<F> {
    /// Opens a remote file using the caller's ranged request and error vocabulary.
    pub fn new(get: F, source: &str, what: String) -> io::Result<Self> {
        let mut file = Self {
            get,
            source: source.into(),
            what,
            at: 0,
            end: 0,
            len: CHUNK,
            body: None,
            error: None,
        };
        file.chunk()?;
        Ok(file)
    }

    /// The size reported by the server.
    pub fn size(&self) -> u64 {
        self.len
    }

    /// The underlying read error, rather than a parser's truncation error.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    fn chunk(&mut self) -> io::Result<()> {
        let last = self.at.saturating_add(CHUNK).min(self.len) - 1;
        let resp = (self.get)(format!("bytes={}-{last}", self.at))?;
        if resp.status() == 206 {
            let (_, end, len) = resp
                .header("content-range")
                .and_then(content_range)
                .filter(|&(start, ..)| start == self.at)
                .ok_or_else(|| {
                    io::Error::other(format!(
                        "{} sent the wrong part of {}.",
                        self.source, self.what
                    ))
                })?;
            (self.end, self.len) = (end + 1, len);
        } else if self.at == 0 {
            self.len = resp
                .header("content-length")
                .and_then(|l| l.parse().ok())
                .ok_or_else(|| {
                    io::Error::other(format!("{} sent no size for {}.", self.source, self.what))
                })?;
            self.end = self.len;
        } else {
            return Err(io::Error::other(format!(
                "{} cannot send parts of {}.",
                self.source, self.what
            )));
        }
        self.body = Some(resp.into_reader());
        Ok(())
    }

    fn fill(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.at >= self.len || buf.is_empty() {
            return Ok(0);
        }
        if self.at >= self.end {
            self.chunk()?;
        }
        let want = buf.len().min((self.end - self.at) as usize);
        let n = self
            .body
            .as_mut()
            .expect("a chunk is open")
            .read(&mut buf[..want])?;
        if n == 0 {
            return Err(io::Error::other(format!(
                "{} stopped sending {}.",
                self.source, self.what
            )));
        }
        self.at += n as u64;
        Ok(n)
    }
}

impl<F: FnMut(String) -> io::Result<ureq::Response>> Read for Ranged<F> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.fill(buf)
            .inspect_err(|e| self.error = Some(e.to_string()))
    }
}

/// `bytes <start>-<end>/<len>` → (start, end, len).
fn content_range(v: &str) -> Option<(u64, u64, u64)> {
    let (range, len) = v.strip_prefix("bytes ")?.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    let (start, end, len) = (start.parse().ok()?, end.parse().ok()?, len.parse().ok()?);
    (start <= end && end < len).then_some((start, end, len))
}

/// One GET with bounded retries; credentials are stripped on every redirect by [`agent`].
pub fn get(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    headers: &[(&str, &str)],
) -> Result<ureq::Response, Box<ureq::Error>> {
    let waits = [500, 1500, 4000].map(Duration::from_millis);
    let mut attempt = 0;
    loop {
        let mut req = agent.get(url);
        if let Some(token) = token.filter(|t| !t.is_empty()) {
            req = req.set("Authorization", &format!("Bearer {token}"));
        }
        for &(name, value) in headers {
            req = req.set(name, value);
        }
        let after = match req.call() {
            Err(ureq::Error::Status(code, r))
                if attempt < waits.len() && (code == 429 || code >= 500) =>
            {
                r.header("retry-after")
                    .and_then(|s| s.trim().parse().ok())
                    .map(Duration::from_secs)
            }
            Err(ureq::Error::Transport(_)) if attempt < waits.len() => None,
            r => return r.map_err(Box::new),
        };
        std::thread::sleep(after.unwrap_or(waits[attempt]).min(Duration::from_secs(5)));
        attempt += 1;
    }
}

/// Why a resumable download did not become a verified destination file.
#[derive(Debug)]
pub enum DownloadError {
    /// The request was refused or could not connect.
    Http(Box<ureq::Error>),
    /// The partial file or its validator could not be written.
    Write(io::Error),
    /// The response stopped while streaming.
    Read(io::Error),
    /// The server returned a different byte range or total size.
    Range,
    /// The response exceeded the signed size.
    Oversize,
    /// The response ended before its expected size.
    Short,
    /// The expected hash is invalid or the bytes do not match it.
    Checksum,
    /// The progress callback requested cancellation.
    Cancelled,
    /// The verified partial could not be renamed into place.
    Save(io::Error),
}

impl fmt::Display for DownloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(e) => write!(f, "download request failed: {e}"),
            Self::Write(e) => write!(f, "cannot write partial download: {e}"),
            Self::Read(e) => write!(f, "download stopped: {e}; try again to resume"),
            Self::Range => f.write_str("server returned the wrong range or size"),
            Self::Oversize => f.write_str("download exceeded its expected size; partial deleted"),
            Self::Short => f.write_str("download stopped early; try again to resume"),
            Self::Checksum => f.write_str("download checksum mismatch; partial deleted"),
            Self::Cancelled => f.write_str("download cancelled"),
            Self::Save(e) => write!(f, "cannot save verified download: {e}"),
        }
    }
}

impl std::error::Error for DownloadError {}

/// The resumable sibling of a destination, preserving non-Unicode paths: `<dest>.part`, or for a
/// blob of the Hugging Face cache (`blobs/<sha256>`) huggingface_hub's own `<sha256>.incomplete`,
/// so either tool resumes, or clears, what the other left.
pub fn part_path(dest: &Path) -> PathBuf {
    let blob = dest.parent().and_then(Path::file_name) == Some("blobs".as_ref());
    let mut path = dest.as_os_str().to_owned();
    path.push(if blob { ".incomplete" } else { ".part" });
    path.into()
}

fn etag_path(dest: &Path) -> PathBuf {
    let mut path = part_path(dest).into_os_string();
    path.push(".etag");
    path.into()
}

/// Deletes a partial and its ETag; missing files are already removed.
pub fn remove_partial(dest: &Path) -> io::Result<()> {
    for path in [part_path(dest), etag_path(dest)] {
        match fs::remove_file(path) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
            _ => {}
        }
    }
    Ok(())
}

/// Lowercase hexadecimal, shared with callers that record verified file hashes.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Hashes small authenticated metadata without a second hashing implementation.
pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn regular_partial(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if !metadata.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "partial download path is not a regular file",
                ));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if metadata.nlink() > 1 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "partial download has another hard link",
                    ));
                }
            }
            Ok(())
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// Streams a file's SHA-256 without loading it into memory.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut hash = Sha256::new();
    hash_into(File::open(path)?, &mut hash)?;
    Ok(hex(&hash.finalize()))
}

fn hash_into(mut reader: impl Read, hash: &mut Sha256) -> io::Result<()> {
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            return Ok(());
        }
        hash.update(&buf[..n]);
    }
}

/// Resumes, verifies size and SHA-256, and only then renames into place, over [`PARALLEL`]
/// connections as [`download_parallel`] does. The callback receives (bytes present, bytes fetched
/// this run); false cancels.
pub fn download(
    url: &str,
    token: Option<&str>,
    dest: &Path,
    sha256: &str,
    size: u64,
    progress: impl FnMut(u64, u64) -> bool,
) -> Result<String, DownloadError> {
    let agent = agent(url);
    // Where `url` redirected (a CDN's signed URL): the ranges after the first go there straight,
    // since a round trip to `url` per 8 MiB piece costs about as much as the piece. It takes no
    // token, and one the CDN refuses (it expired) sends the next range via `url` again.
    let direct = Mutex::new(None::<String>);
    let lock = || direct.lock().unwrap_or_else(PoisonError::into_inner);
    download_parallel(
        dest,
        Some(sha256),
        size,
        |start, end, etag| {
            let range = match end {
                Some(end) => format!("bytes={start}-{end}"),
                None => format!("bytes={start}-"),
            };
            let mut headers = vec![("Range", range.as_str())];
            if let Some(etag) = etag {
                headers.push(("If-Range", etag));
            }
            let cdn = lock().clone();
            if let Some(cdn) = cdn {
                match get(&agent, &cdn, None, &headers) {
                    Err(e) if matches!(*e, ureq::Error::Status(401 | 403 | 404 | 410, _)) => {
                        *lock() = None;
                    }
                    r => return r,
                }
            }
            let response = get(&agent, url, token, &headers)?;
            if response.get_url() != url {
                *lock() = Some(response.get_url().to_owned());
            }
            Ok(response)
        },
        progress,
    )
}

/// The same transport for callers with request-specific headers and optional upstream hashes.
pub fn download_with(
    dest: &Path,
    expected: Option<&str>,
    size: u64,
    mut request: impl FnMut(u64, Option<&str>) -> Result<ureq::Response, Box<ureq::Error>>,
    mut progress: impl FnMut(u64, u64) -> bool,
) -> Result<String, DownloadError> {
    use DownloadError::*;
    let (mut file, mut have, mut hash) = open_partial(dest, expected, size)?;
    let mut fetched = 0;
    if !progress(have, fetched) {
        return Err(Cancelled);
    }
    if have < size {
        let response = request(have, read_etag(dest)?.as_deref()).map_err(Http)?;
        if response.status() == 206 {
            let range = response.header("content-range").and_then(content_range);
            if range.is_none_or(|(start, _, total)| start != have || total != size) {
                return Err(Range);
            }
        } else if response.status() == 200 {
            file.set_len(0).map_err(Write)?;
            file.rewind().map_err(Write)?;
            have = 0;
            hash = Sha256::new();
            if !progress(have, fetched) {
                return Err(Cancelled);
            }
        } else {
            return Err(Range);
        }
        save_etag(dest, response.header("etag"), response.status() == 200)?;
        let mut body = response.into_reader();
        let mut buf = vec![0; 256 << 10];
        loop {
            if !progress(have, fetched) {
                return Err(Cancelled);
            }
            let n = body.read(&mut buf).map_err(Read)?;
            if n == 0 {
                break;
            }
            if n as u64 > size - have {
                drop(file);
                let _ = remove_partial(dest);
                return Err(Oversize);
            }
            file.write_all(&buf[..n]).map_err(Write)?;
            hash.update(&buf[..n]);
            have += n as u64;
            fetched += n as u64;
        }
    }
    finish(dest, expected, size, file, have, hash)
}

/// Connections a parallel download keeps busy.
// ponytail: fixed at 8, as most Hub clients use; a knob when a link shows more helps.
pub const PARALLEL: usize = 8;

/// Bytes per range request of [`download_parallel`].
const PIECE: u64 = 8 << 20;

/// Pieces fetched ahead of the one being written: the most a download holds in memory.
const AHEAD: u64 = PARALLEL as u64 + 4;

/// [`download_with`] over [`PARALLEL`] connections: what is left past the `.part` file is fetched
/// as 8 MiB ranges and written in order, so the `.part` file stays a prefix that resumes and
/// hashes as before. `request(start, end, etag)` asks for bytes `start..=end` (`end` None: to the
/// end of the file). Less than two pieces left, or a server that ignores ranges, takes one stream.
pub fn download_parallel(
    dest: &Path,
    expected: Option<&str>,
    size: u64,
    request: impl Fn(u64, Option<u64>, Option<&str>) -> Result<ureq::Response, Box<ureq::Error>> + Sync,
    mut progress: impl FnMut(u64, u64) -> bool,
) -> Result<String, DownloadError> {
    use DownloadError::*;
    let one = |progress| {
        download_with(
            dest,
            expected,
            size,
            |at, etag| request(at, None, etag),
            progress,
        )
    };
    let present = fs::metadata(part_path(dest)).map_or(0, |m| m.len());
    if size.saturating_sub(if present > size { 0 } else { present }) < 2 * PIECE {
        return one(&mut progress);
    }
    let (mut file, mut have, mut hash) = open_partial(dest, expected, size)?;
    if !progress(have, 0) {
        return Err(Cancelled);
    }
    // The first piece says whether the server serves ranges and fixes the validator the rest send.
    let stored = read_etag(dest)?;
    let first = request(have, Some(have + PIECE - 1), stored.as_deref()).map_err(Http)?;
    if first.status() != 206 {
        drop((first, file));
        return one(&mut progress);
    }
    let etag = first.header("etag").map(str::to_owned).or(stored);
    let fetched = AtomicU64::new(0);
    let halt = AtomicBool::new(false);
    // Read on its own thread so Cancel and progress hold for this piece as for the rest.
    let (tx, rx) = mpsc::channel();
    let piece = std::thread::scope(|scope| {
        scope.spawn(|| tx.send(read_piece(first, have, PIECE, size, &halt, &fetched)));
        loop {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(piece) => break piece,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if !halt.load(Ordering::Relaxed)
                        && !progress(have, fetched.load(Ordering::Relaxed))
                    {
                        halt.store(true, Ordering::Relaxed);
                    }
                }
                // The reader panicked; the scope raises its panic as it ends.
                Err(mpsc::RecvTimeoutError::Disconnected) => break Err(Cancelled),
            }
        }
    })?;
    save_etag(dest, etag.as_deref(), false)?;
    file.write_all(&piece).map_err(Write)?;
    hash.update(&piece);
    have += PIECE;

    struct Line {
        /// The next piece a connection takes.
        next: u64,
        /// Bytes written, in order.
        written: u64,
        done: BTreeMap<u64, Result<Vec<u8>, DownloadError>>,
    }
    let line = Mutex::new(Line {
        next: have,
        written: have,
        done: BTreeMap::new(),
    });
    let wake = Condvar::new();
    let lock = || line.lock().unwrap_or_else(PoisonError::into_inner);
    let outcome = std::thread::scope(|scope| {
        for _ in 0..PARALLEL {
            scope.spawn(|| {
                loop {
                    let start = {
                        let mut l = lock();
                        while l.next < size
                            && l.next >= l.written + AHEAD * PIECE
                            && !halt.load(Ordering::Relaxed)
                        {
                            l = wake.wait(l).unwrap_or_else(PoisonError::into_inner);
                        }
                        if l.next >= size || halt.load(Ordering::Relaxed) {
                            return;
                        }
                        let start = l.next;
                        l.next = (start + PIECE).min(size);
                        start
                    };
                    let len = PIECE.min(size - start);
                    let got =
                        fetch_piece(&request, start, len, size, etag.as_deref(), &halt, &fetched);
                    lock().done.insert(start, got);
                    wake.notify_all();
                }
            });
        }
        // The writer: each piece in order as it lands, progress at least every 100 ms.
        let outcome = loop {
            if have == size {
                break Ok(());
            }
            let next = {
                let mut l = lock();
                if !l.done.contains_key(&have) {
                    l = wake
                        .wait_timeout(l, Duration::from_millis(100))
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
                l.done.remove(&have)
            };
            if !progress(have, fetched.load(Ordering::Relaxed)) {
                break Err(Cancelled);
            }
            let bytes = match next {
                None => continue,
                Some(Err(e)) => break Err(e),
                Some(Ok(bytes)) => bytes,
            };
            if let Err(e) = file.write_all(&bytes) {
                break Err(Write(e));
            }
            hash.update(&bytes);
            have += bytes.len() as u64;
            lock().written = have;
            wake.notify_all();
        };
        // Under the line's lock: a worker between its check of `halt` and its wait would otherwise
        // miss this wake-up and sleep for good, and the scope would never join.
        {
            let _line = lock();
            halt.store(outcome.is_err(), Ordering::Relaxed);
        }
        wake.notify_all();
        outcome
    });
    outcome?;
    finish(dest, expected, size, file, have, hash)
}
/// The `.part` file of `dest` opened for writing at its end, its length and the hash of those bytes.
fn open_partial(
    dest: &Path,
    expected: Option<&str>,
    size: u64,
) -> Result<(File, u64, Sha256), DownloadError> {
    use DownloadError::*;
    if expected.is_some_and(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit())) {
        return Err(Checksum);
    }
    let path = part_path(dest);
    regular_partial(&path).map_err(Write)?;
    regular_partial(&etag_path(dest)).map_err(Write)?;
    // Write, not append: Windows refuses set_len on an append-only handle.
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(Write)?;
    let mut have = file.seek(SeekFrom::End(0)).map_err(Write)?;
    if have > size {
        file.set_len(0).map_err(Write)?;
        file.rewind().map_err(Write)?;
        have = 0;
    }
    let mut hash = Sha256::new();
    if have > 0 {
        hash_into(File::open(&path).map_err(Write)?.take(have), &mut hash).map_err(Write)?;
    }
    Ok((file, have, hash))
}

fn read_etag(dest: &Path) -> Result<Option<String>, DownloadError> {
    match fs::read_to_string(etag_path(dest)) {
        Ok(etag) => Ok(Some(etag)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(DownloadError::Write(e)),
    }
}

/// Keeps the validator a resume sends; a whole-file answer without one drops the old one.
fn save_etag(dest: &Path, etag: Option<&str>, whole: bool) -> Result<(), DownloadError> {
    use DownloadError::Write;
    if let Some(etag) = etag {
        let mut validator = File::create(etag_path(dest)).map_err(Write)?;
        validator.write_all(etag.as_bytes()).map_err(Write)?;
        validator.sync_all().map_err(Write)?;
    } else if whole {
        match fs::remove_file(etag_path(dest)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(Write(e)),
            _ => {}
        }
    }
    Ok(())
}

/// Checks the finished `.part` file's size and hash, then renames it to `dest`.
fn finish(
    dest: &Path,
    expected: Option<&str>,
    size: u64,
    file: File,
    have: u64,
    hash: Sha256,
) -> Result<String, DownloadError> {
    use DownloadError::*;
    file.sync_all().map_err(Write)?;
    drop(file);
    if have != size {
        return Err(Short);
    }
    let digest = hex(&hash.finalize());
    if expected.is_some_and(|want| !digest.eq_ignore_ascii_case(want)) {
        let _ = remove_partial(dest);
        return Err(Checksum);
    }
    fs::rename(part_path(dest), dest).map_err(Save)?;
    // A leftover validator is harmless; the verified destination is already committed.
    let _ = fs::remove_file(etag_path(dest));
    Ok(digest)
}

/// Bytes `start..start + len` of a `size`-byte file, tried three times when the body breaks off.
fn fetch_piece(
    request: &impl Fn(u64, Option<u64>, Option<&str>) -> Result<ureq::Response, Box<ureq::Error>>,
    start: u64,
    len: u64,
    size: u64,
    etag: Option<&str>,
    halt: &AtomicBool,
    fetched: &AtomicU64,
) -> Result<Vec<u8>, DownloadError> {
    let mut tries = 0;
    loop {
        let got = request(start, Some(start + len - 1), etag)
            .map_err(DownloadError::Http)
            .and_then(|r| read_piece(r, start, len, size, halt, fetched));
        match got {
            Err(DownloadError::Read(_) | DownloadError::Short) if tries < 2 => tries += 1,
            got => return got,
        }
    }
}

/// The body of a 206 for exactly `start..start + len` of a `size`-byte file.
fn read_piece(
    response: ureq::Response,
    start: u64,
    len: u64,
    size: u64,
    halt: &AtomicBool,
    fetched: &AtomicU64,
) -> Result<Vec<u8>, DownloadError> {
    use DownloadError::*;
    // A 200 here is the whole file: it changed since the first piece (If-Range), or ranges stopped.
    let range = response.header("content-range").and_then(content_range);
    if response.status() != 206 || range != Some((start, start + len - 1, size)) {
        return Err(Range);
    }
    let mut body = response.into_reader().take(len + 1);
    let mut out = Vec::with_capacity(len as usize);
    let mut buf = vec![0; 256 << 10];
    loop {
        if halt.load(Ordering::Relaxed) {
            return Err(Cancelled);
        }
        let n = body.read(&mut buf).map_err(Read)?;
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
        fetched.fetch_add(n as u64, Ordering::Relaxed);
    }
    match (out.len() as u64).cmp(&len) {
        std::cmp::Ordering::Greater => Err(Range),
        std::cmp::Ordering::Less => Err(Short),
        std::cmp::Ordering::Equal => Ok(out),
    }
}

fn user_data_base() -> Option<PathBuf> {
    #[cfg(windows)]
    let base = env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(target_os = "macos")]
    let base = env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support"));
    #[cfg(all(unix, not(target_os = "macos")))]
    let base = env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|dir| dir.is_absolute())
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")));
    base
}

/// The per-user data directory, with the CLI's platform rules and error text.
pub fn data_dir() -> Result<PathBuf> {
    Ok(user_data_base()
        .context("set --data-dir: no per-user data directory")?
        .join("larch"))
}

/// The per-user installer destination, overridden by LARCH_INSTALL_ROOT for support and tests.
pub fn install_root() -> Result<PathBuf> {
    if let Some(root) = env::var_os("LARCH_INSTALL_ROOT") {
        return Ok(root.into());
    }
    let base = user_data_base().context("no per-user install directory; set LARCH_INSTALL_ROOT")?;
    #[cfg(windows)]
    let root = base.join("Programs/larch");
    #[cfg(not(windows))]
    let root = base.join("larch-install");
    Ok(root)
}

/// One downloaded model file and the source and hash that identify it.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct IndexEntry {
    /// The installed file path, including files in a user-selected library.
    pub file: String,
    /// The source repository.
    pub repo: String,
    /// The source revision used by the download.
    pub revision: String,
    /// The verified file SHA-256.
    pub sha256: String,
    /// The file size in bytes.
    pub size: u64,
    /// An optional catalog alias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias: Option<String>,
}

fn index_lock(data: &Path) -> io::Result<File> {
    let models = data.join("models");
    fs::create_dir_all(&models)?;
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(models.join(".index.lock"))
}

fn read_index(data: &Path) -> Result<Vec<IndexEntry>> {
    match fs::read(data.join("models/index.json")) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e.into()),
    }
}

/// Reads the model index while holding its shared filesystem lock.
pub fn index_read(data: &Path) -> Result<Vec<IndexEntry>> {
    let lock = index_lock(data)?;
    lock.lock_shared()?;
    read_index(data)
}

/// Replaces a model entry under an exclusive lock and atomically commits a synced index.
pub fn index_add(data: &Path, entry: IndexEntry) -> Result<()> {
    update_index(data, |entries| {
        entries.retain(|e| e.file != entry.file);
        entries.push(entry);
    })
}

/// Removes an entry under the same exclusive lock used by concurrent downloads.
pub fn index_remove(data: &Path, file: &str) -> Result<()> {
    update_index(data, |entries| entries.retain(|e| e.file != file))
}

fn update_index(data: &Path, change: impl FnOnce(&mut Vec<IndexEntry>)) -> Result<()> {
    let lock = index_lock(data)?;
    lock.lock()?;
    let mut entries = read_index(data)?;
    change(&mut entries);
    let models = data.join("models");
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let (temp, mut file) = loop {
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let temp = models.join(format!(".index-{}-{n}.tmp", std::process::id()));
        match File::options().create_new(true).write(true).open(&temp) {
            Ok(file) => break (temp, file),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    };
    let bytes = serde_json::to_vec(&entries)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(temp, models.join("index.json"))?;
    #[cfg(unix)]
    File::open(models)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// huggingface_hub's order of variables, `~` expanded, a relative value passed over.
    #[test]
    fn hf_cache_is_found_as_huggingface_hub_finds_it() {
        let home = if cfg!(windows) {
            "C:\\Users\\u"
        } else {
            "/home/u"
        };
        let home_var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        let find = |vars: &[(&str, &str)]| {
            hf_cache_in(|k| {
                (k == home_var)
                    .then_some(home)
                    .or_else(|| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| *v))
                    .map(Into::into)
            })
        };
        let at = |p: &str| Some(Path::new(home).join(p));
        assert_eq!(find(&[]), at(".cache/huggingface/hub"));
        assert_eq!(find(&[("XDG_CACHE_HOME", home)]), at("huggingface/hub"));
        assert_eq!(
            find(&[("HF_HOME", "~/hf"), ("XDG_CACHE_HOME", home)]),
            at("hf/hub")
        );
        assert_eq!(
            find(&[("HUGGINGFACE_HUB_CACHE", "~/old"), ("HF_HOME", "~/hf")]),
            at("old")
        );
        assert_eq!(
            find(&[
                ("HF_HUB_CACHE", "~/new"),
                ("HUGGINGFACE_HUB_CACHE", "~/old")
            ]),
            at("new")
        );
        assert_eq!(
            find(&[("HF_HUB_CACHE", "relative/hub")]),
            at(".cache/huggingface/hub")
        );
    }

    /// A file in a snapshot, or a folder of one, the commit `refs/main` names first; one per repo.
    #[test]
    fn a_cache_file_is_found_by_name() {
        let cache = env::temp_dir().join(format!("larch-hf-cache-file-{}", std::process::id()));
        let repo = cache.join("models--o--n");
        let (old, new) = (repo.join("snapshots/aaaa"), repo.join("snapshots/bbbb/Q4"));
        for dir in [&old, &new, &repo.join("refs")] {
            fs::create_dir_all(dir).unwrap();
        }
        fs::write(old.join("m.gguf"), b"old").unwrap();
        fs::write(new.join("m.gguf"), b"new").unwrap();
        assert_eq!(cache_files(&cache, "m.gguf"), [old.join("m.gguf")]);
        fs::write(repo.join("refs/main"), "bbbb").unwrap();
        assert_eq!(cache_files(&cache, "m.gguf"), [new.join("m.gguf")]);
        assert!(cache_files(&cache, "other.gguf").is_empty());
        // Another repo's file of the same name: both, for the user to pick by path.
        let other = cache.join("models--p--q/snapshots/cccc");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("m.gguf"), b"other").unwrap();
        assert_eq!(
            cache_files(&cache, "m.gguf"),
            [new.join("m.gguf"), other.join("m.gguf")]
        );
        fs::remove_dir_all(cache).unwrap();
    }

    #[test]
    fn a_cache_blob_resumes_under_huggingface_hubs_name() {
        let blob = Path::new("hub/models--o--n/blobs/abc");
        assert_eq!(
            part_path(blob),
            Path::new("hub/models--o--n/blobs/abc.incomplete")
        );
        assert_eq!(
            part_path(Path::new("models/m.gguf")),
            Path::new("models/m.gguf.part")
        );
    }
}
