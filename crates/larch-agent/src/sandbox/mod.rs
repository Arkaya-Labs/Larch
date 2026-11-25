//! Runs one shell command confined to a worktree: no network, file writes only inside the
//! worktree and a private tmp, CPU, memory and process caps, and its whole process group killed
//! on timeout. Linux confines with unshare(1), Landlock, seccomp and prlimit(1); macOS with
//! sandbox-exec and ulimit; Windows with an AppContainer and a Job Object.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
use linux as os;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos as os;
#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::{run, setup};

use std::{
    collections::VecDeque,
    env, fmt,
    io::{self, ErrorKind, Read},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};
#[cfg(unix)]
use std::{
    os::unix::{fs::MetadataExt, process::CommandExt},
    process::{Child, Command, ExitStatus, Stdio},
    time::Instant,
};

/// How often a running command is checked for its exit.
#[cfg(unix)]
const POLL: Duration = Duration::from_millis(20);
/// How long the output readers may take to reach EOF once the group is killed.
const GRACE: Duration = Duration::from_secs(1);

/// What one command may use.
#[derive(Clone, Debug)]
pub struct Limits {
    /// Wall-clock time before the whole process group is killed.
    pub timeout: Duration,
    /// Bytes kept of stdout, and of stderr: the last ones, where a test run's summary is; the
    /// ones before are read and dropped.
    pub max_output_bytes: usize,
    /// CPU seconds per process (RLIMIT_CPU).
    pub cpu_s: u64,
    /// Address space per process, in bytes (RLIMIT_AS).
    pub mem_bytes: u64,
    /// Processes and threads the command may have at once (RLIMIT_NPROC).
    pub max_procs: u32,
    /// Ends the command as `timeout` does once it is set: the user's Stop of a code run.
    pub stop: Option<Arc<AtomicBool>>,
    /// None: the command writes anywhere in the worktree. Some: the worktree is read-only but
    /// for these folders of it (a read-only run's build output); one that does not exist, or
    /// whose real path is outside the worktree, is left out.
    pub writable: Option<Vec<PathBuf>>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            max_output_bytes: 1 << 20,
            cpu_s: 120,
            mem_bytes: 4 << 30,
            // Threads count too, so a parallel build needs room.
            max_procs: 256,
            stop: None,
            writable: None,
        }
    }
}

/// How a command ended and what it printed.
#[derive(Debug)]
pub struct Output {
    /// Its exit code; None when a signal ended it, the timeout's kill included.
    pub exit: Option<i32>,
    /// Its stdout, at most its last `max_output_bytes`.
    pub stdout: Vec<u8>,
    /// Its stderr, at most its last `max_output_bytes`.
    pub stderr: Vec<u8>,
    /// It ran past `timeout` and was killed.
    pub timed_out: bool,
    /// `Limits::stop` was set while it ran, and it was killed.
    pub stopped: bool,
    /// stdout or stderr went past `max_output_bytes`, or could not be read to its end.
    pub truncated: bool,
}

/// Why a command did not run, or its end could not be seen.
#[derive(Debug)]
pub enum SandboxError {
    /// This host cannot confine it: a tool or kernel feature is missing, or we run as root. The
    /// command never ran.
    Unavailable(String),
    /// Starting, waiting for or killing it failed.
    Io(io::Error),
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(why) => write!(f, "the sandbox is unavailable: {why}"),
            Self::Io(e) => write!(f, "the sandbox failed: {e}"),
        }
    }
}

impl std::error::Error for SandboxError {}

impl From<io::Error> for SandboxError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Runs `cmd` with `sh -c` in `worktree`, confined, within `limits`. The environment holds only
/// PATH, LANG, CARGO_HOME and RUSTUP_HOME from ours, CARGO_NET_OFFLINE, and HOME, TMP and TMPDIR
/// set to a private tmp deleted afterwards.
#[cfg(unix)]
pub fn run(cmd: &str, worktree: &Path, limits: &Limits) -> Result<Output, SandboxError> {
    // Real paths: sandbox-exec matches them, and /var on macOS is a link.
    let worktree = worktree.canonicalize()?;
    let tmp_dir = tempfile::tempdir()?;
    let tmp = tmp_dir.path().canonicalize()?;
    // A new directory belongs to our effective uid.
    if tmp.metadata()?.uid() == 0 {
        return Err(SandboxError::Unavailable(
            "refusing to run as root: the kernel does not apply the process cap to uid 0".into(),
        ));
    }
    let writable = writable(&worktree, &tmp, limits);
    probe(os::command("true", &writable, limits))?;
    let mut command = os::command(cmd, &writable, limits);
    command
        .env_clear()
        .envs(env::var_os("PATH").map(|p| ("PATH", p)))
        .envs(env::var_os("LANG").map(|l| ("LANG", l)))
        .envs(toolchains())
        // Offline at once, rather than after a failed fetch of the index.
        .env("CARGO_NET_OFFLINE", "true")
        .env("HOME", &tmp)
        .env("TMP", &tmp)
        // TMP is what was asked for; TMPDIR is what mktemp, cargo and Python read.
        .env("TMPDIR", &tmp)
        .current_dir(&worktree)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    let mut child = os::spawn(&mut command, &writable)?;
    let stdout = drain(child.stdout.take(), limits.max_output_bytes);
    let stderr = drain(child.stderr.take(), limits.max_output_bytes);

    let ended = wait_until(&mut child, Instant::now() + limits.timeout, limits);
    // Also after a clean exit: what it left running in the background would hold the pipes open.
    let killed = kill_group(child.id());
    if killed.is_err() {
        // At least the direct child, so the wait below returns.
        let _ = child.kill();
    }
    let ended = ended?;
    let stopped = ended.is_none() && stopped(limits);
    let (status, timed_out) = match ended {
        Some(status) => (status, false),
        None => (child.wait()?, !stopped),
    };
    killed?;

    let (stdout, cut_out) = collect(stdout)?;
    let (stderr, cut_err) = collect(stderr)?;
    Ok(Output {
        exit: status.code(),
        stdout,
        stderr,
        timed_out,
        stopped,
        truncated: cut_out || cut_err,
    })
}

/// Where a command in `worktree` may write: the private `tmp`, and the worktree, or only the
/// folders of it `Limits::writable` names (`only`).
#[cfg(unix)]
fn writable(worktree: &Path, tmp: &Path, limits: &Limits) -> Vec<PathBuf> {
    let mut writable = vec![tmp.to_path_buf()];
    writable.extend(only(worktree, limits).unwrap_or_else(|| vec![worktree.to_path_buf()]));
    writable
}

/// The folders of `worktree` that `Limits::writable` names, as real paths: each one an existing
/// folder whose real path is the worktree's own joined with its name, so a link (`build -> src`,
/// `node_modules -> .git`, or one outside) is never one; never `.git`. None when the command may
/// write all of the worktree.
pub(super) fn only(worktree: &Path, limits: &Limits) -> Option<Vec<PathBuf>> {
    let dirs = limits.writable.as_ref()?;
    let Ok(real) = worktree.canonicalize() else {
        return Some(Vec::new());
    };
    let dirs = (dirs.iter())
        .filter(|d| {
            d.components()
                .all(|c| matches!(c, std::path::Component::Normal(n) if n != ".git"))
        })
        .filter_map(|d| {
            let named = real.join(d);
            (named.canonicalize().ok()).filter(|r| *r == named && r.is_dir())
        });
    Some(dirs.collect())
}

/// Whether `limits.stop` is set.
fn stopped(limits: &Limits) -> bool {
    (limits.stop.as_ref()).is_some_and(|stop| stop.load(Ordering::Relaxed))
}

/// CARGO_HOME and RUSTUP_HOME, which the private HOME would leave empty: rustup's proxies would
/// find no toolchain and cargo no registry. As cargo and rustup find them (the variable, else
/// under our HOME), when the folder exists; Landlock leaves them read-only.
fn toolchains() -> impl Iterator<Item = (&'static str, PathBuf)> {
    let home = env::home_dir();
    [("CARGO_HOME", ".cargo"), ("RUSTUP_HOME", ".rustup")]
        .into_iter()
        .filter_map(move |(var, dir)| {
            let path =
                (env::var_os(var).map(PathBuf::from)).or_else(|| Some(home.as_ref()?.join(dir)));
            path.filter(|p| p.is_dir()).map(|p| (var, p))
        })
}

/// Output read so far, and whether any was dropped before it.
type Kept = Arc<Mutex<(VecDeque<u8>, bool)>>;

/// Reads `from` on a thread of its own, so a full pipe never stalls the child: keeps the last
/// `cap` bytes and drops the ones before. The receiver gets the read's end.
fn drain(
    from: Option<impl Read + Send + 'static>,
    cap: usize,
) -> (Kept, mpsc::Receiver<io::Result<()>>) {
    let kept: Kept = Arc::default();
    let (tx, rx) = mpsc::channel();
    let shared = Arc::clone(&kept);
    thread::spawn(move || {
        let mut from = from.expect("stdout and stderr are piped");
        let mut buf = [0u8; 8192];
        let read = loop {
            match from.read(&mut buf) {
                Ok(0) => break Ok(()),
                Ok(n) => {
                    let mut kept = shared.lock().unwrap_or_else(PoisonError::into_inner);
                    kept.0.extend(&buf[..n]);
                    let over = kept.0.len().saturating_sub(cap);
                    kept.0.drain(..over);
                    kept.1 |= over > 0;
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => break Err(e),
            }
        };
        // Fails only once `run` has stopped waiting for it.
        let _ = tx.send(read);
    });
    (kept, rx)
}

/// What `drain` kept, once its pipe reached EOF or the grace passed; past the grace it is flagged
/// as cut.
// ponytail: on macOS a process that leaves the group (setsid) outlives the kill and holds the
// pipe; what it prints past the grace is lost. ceiling: a command that daemonises. upgrade: a
// launchd job per command if a tool does. Linux kills the whole pid namespace.
fn collect((kept, done): (Kept, mpsc::Receiver<io::Result<()>>)) -> io::Result<(Vec<u8>, bool)> {
    let open = match done.recv_timeout(GRACE) {
        Ok(read) => read.map(|()| false)?,
        Err(_) => true,
    };
    let mut kept = kept.lock().unwrap_or_else(PoisonError::into_inner);
    Ok((std::mem::take(&mut kept.0).into(), kept.1 || open))
}

/// Runs `probe`, the confinement's tools around `true`: a failure of theirs (missing, namespaces
/// off, a limit above our hard limit) refuses here, never reads as the command's own exit.
#[cfg(unix)]
fn probe(mut probe: Command) -> Result<(), SandboxError> {
    let program = probe.get_program().to_string_lossy().into_owned();
    match probe.stdin(Stdio::null()).output() {
        Err(e) if e.kind() == ErrorKind::NotFound => Err(SandboxError::Unavailable(format!(
            "{program} is not installed"
        ))),
        Err(e) => Err(SandboxError::Unavailable(format!(
            "cannot run {program}: {e}"
        ))),
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => {
            let why = String::from_utf8_lossy(&out.stderr).trim().to_string();
            #[cfg(target_os = "linux")]
            let why = why + os::userns_hint();
            Err(SandboxError::Unavailable(format!(
                "{program} failed: {why}"
            )))
        }
    }
}

/// The child's exit status, or None once `deadline` passes or `limits.stop` is set first.
#[cfg(unix)]
fn wait_until(
    child: &mut Child,
    deadline: Instant,
    limits: &Limits,
) -> io::Result<Option<ExitStatus>> {
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline || stopped(limits) {
            return Ok(None);
        }
        thread::sleep(POLL);
    }
}

/// SIGKILLs every process left in the group `pgid`.
#[cfg(unix)]
fn kill_group(pgid: u32) -> io::Result<()> {
    // ponytail: kill(1), a copy of the private `kill` in crates/larch/src/supervise.rs, another
    // lane's file: std has no killpg and unsafe is forbidden. upgrade: call it once it is public,
    // or rustix's kill_process_group for both. A failing status is the normal "nothing left".
    Command::new("kill")
        .args(["-KILL", "--", &format!("-{pgid}")])
        .stderr(Stdio::null())
        .status()
        .map(drop)
}
