//! Windows: an AppContainer with no capabilities, so no network, and writes only where its SID is
//! granted, the worktree (or, for a read-only run, the folders of it it may write, the rest only
//! read) and the private tmp; a Job Object caps each process's memory and CPU
//! time and how many run at once, and kills the whole tree on timeout or once the command ends.

use std::{
    env,
    ffi::OsString,
    fs, io,
    path::{self, Path, PathBuf},
    process::Command,
    sync::{Mutex, OnceLock, PoisonError},
};

use larch_sys::appcontainer::{self, Child, Job, Profile};

use super::{GRACE, Limits, Output, SandboxError, collect, drain, only, stopped, toolchains};
use crate::tool::{no_window, plain};

/// How often a running command is checked for a Stop.
const STOP_POLL: std::time::Duration = std::time::Duration::from_millis(100);

/// Runs `cmd` with `cmd.exe /d /s /c` in `worktree`, confined, within `limits`. The environment
/// holds only PATH, SystemRoot, LANG, CARGO_HOME and RUSTUP_HOME from ours, MSVC's LIB, LIBPATH,
/// INCLUDE and PATH, CARGO_NET_OFFLINE, and HOME, USERPROFILE and LOCALAPPDATA set to a private
/// tmp deleted afterwards; CreateProcess points TEMP and TMP at `Packages\<profile>\AC\Temp`
/// beneath it. Unavailable until `setup` has run once from an administrator prompt.
// ponytail: AppContainer only; decision 19's opt-in fallback (a restricted token, this Job Object
// and a WFP rule, which needs admin) is not built. ceiling: a node child with piped stdio hangs
// to the timeout (node --test, jest's workers; Node 20 and 24 alike), as libuv names its pipes
// outside `\\.\pipe\LOCAL\`, the one prefix an AppContainer may create, and retries the denial
// forever. upgrade: JavaScript tests in code mode on Windows.
pub fn run(cmd: &str, worktree: &Path, limits: &Limits) -> Result<Output, SandboxError> {
    // cmd.exe refuses a \\?\ working directory, which canonicalize gives callers.
    let worktree = plain(path::absolute(worktree)?);
    let tmp_dir = tempfile::tempdir()?;
    let tmp = tmp_dir.path();
    let (name, profile) = container()?;
    fs::create_dir_all(tmp.join("Packages").join(name).join(r"AC\Temp"))?;
    // The tmp's grant goes with the tmp.
    grant(profile, tmp, "M")?;
    let granted = Granted::new(profile, &worktree, limits)?;
    let env = environment(tmp);

    let (probe, _job) = start("exit 0", &worktree, &env, limits, profile)
        .map_err(|e| SandboxError::Unavailable(format!("cannot start cmd.exe in it: {e}")))?;
    match probe.process.wait(limits.timeout)? {
        Some(0) => {}
        ended => {
            return Err(SandboxError::Unavailable(format!(
                "`cmd /c exit 0` in the AppContainer ended with {ended:?}"
            )));
        }
    }

    let (
        Child {
            process,
            stdout,
            stderr,
        },
        job,
    ) = start(cmd, &worktree, &env, limits, profile)?;
    let stdout = drain(Some(stdout), limits.max_output_bytes);
    let stderr = drain(Some(stderr), limits.max_output_bytes);
    // In slices, so a Stop (`Limits::stop`) ends it within one.
    let deadline = std::time::Instant::now() + limits.timeout;
    let ended = loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        let slice = left.min(STOP_POLL);
        match process.wait(slice)? {
            Some(code) => break Some(code),
            None if slice == left || stopped(limits) => break None,
            None => {}
        }
    };
    let stopped = ended.is_none() && stopped(limits);
    // Also after a clean exit: what it left running in the background would hold the pipes open.
    drop(job);
    let (exit, timed_out) = match ended {
        Some(code) => (Some(code as i32), false),
        None => {
            // The kill is asynchronous; its exit code is the kill's, not the command's.
            process.wait(GRACE)?;
            (None, !stopped)
        }
    };
    granted.revoke()?;

    let (stdout, cut_out) = collect(stdout)?;
    let (stderr, cut_err) = collect(stderr)?;
    Ok(Output {
        exit,
        stdout,
        stderr,
        timed_out,
        stopped,
        truncated: cut_out || cut_err,
    })
}

/// The worktree's grant to the profile, revoked by `revoke` or, on an early return, on drop.
// ponytail: granting and revoking walk every file in the worktree, twice per command. ceiling: a
// node_modules or target/ of tens of thousands of files adds seconds. upgrade: one grant per
// worktree, made and revoked where the code task adds and removes it, once a run shows that.
struct Granted<'a> {
    profile: &'a Profile,
    /// The worktree, then the folders of it a read-only run may write.
    paths: Vec<PathBuf>,
    revoked: bool,
}

impl<'a> Granted<'a> {
    /// Grants `worktree` to the profile: all of it, or, with `Limits::writable`, reading it and
    /// writing only the folders that names (`only`).
    fn new(profile: &'a Profile, worktree: &Path, limits: &Limits) -> Result<Self, SandboxError> {
        let mut granted = Self {
            profile,
            paths: vec![worktree.to_path_buf()],
            revoked: false,
        };
        let Some(dirs) = only(worktree, limits) else {
            grant(profile, worktree, "M")?;
            return Ok(granted);
        };
        grant(profile, worktree, "RX")?;
        for dir in dirs {
            // icacls takes no \\?\ path, which canonicalize gives.
            let dir = plain(dir);
            granted.paths.push(dir.clone());
            grant(profile, &dir, "M")?;
        }
        Ok(granted)
    }

    /// Takes the profile's entries off the worktree and every file beneath it.
    fn revoke(mut self) -> Result<(), SandboxError> {
        self.revoked = true;
        self.remove()
            .map_err(|why| SandboxError::Io(io::Error::other(why)))
    }

    /// The folder's own entry and each one's beneath it, the last first: every one is tried, a
    /// folder the command deleted (`rd /s /q node_modules`) has nothing left to take off, and
    /// the errors of the others come back together.
    fn remove(&self) -> Result<(), String> {
        let sid = format!("*{}", self.profile.sid());
        let failed: Vec<String> = (self.paths.iter().rev())
            .filter(|p| p.exists())
            .filter_map(|p| icacls(p, &["/remove:g", &sid]).err())
            .collect();
        match failed.is_empty() {
            true => Ok(()),
            false => Err(failed.join("; ")),
        }
    }
}

impl Drop for Granted<'_> {
    fn drop(&mut self) {
        if !self.revoked {
            // The run's own error is already on its way out; a leftover entry grants only this
            // profile, which nothing but the sandbox runs as, and the next run's grant reuses it.
            let _ = self.remove();
        }
    }
}

/// Gives the profile `rights` in `path`, M (read, write and delete) or RX (read), inherited by
/// everything beneath it, and lets it pass through the folders above it.
fn grant(profile: &Profile, path: &Path, rights: &str) -> Result<(), SandboxError> {
    grant_ancestors(profile, path)?;
    // :r replaces the profile's own entry, so one a run left behind grants no more than this.
    icacls(
        path,
        &["/grant:r", &format!("*{}:(OI)(CI){rights}", profile.sid())],
    )
    .map(drop)
    .map_err(|why| {
        SandboxError::Unavailable(format!(
            "cannot grant {} to the AppContainer: {why}",
            path.display()
        ))
    })
}

/// What `larch-agent sandbox-setup` does, once per computer from an administrator prompt: lets
/// the profile through the folders above the user's profile folder, which git and cmd.exe need
/// and only an admin may change, and read the toolchains on PATH outside it (Program
/// Files\nodejs, whose installer takes ALL APPLICATION PACKAGES off it). The folders it changed.
pub fn setup() -> Result<Vec<PathBuf>, SandboxError> {
    let name = profile_name()?;
    let profile = appcontainer::profile(&name).map_err(|e| {
        SandboxError::Unavailable(format!(
            "cannot create the AppContainer profile {name}: {e}"
        ))
    })?;
    let home = env::home_dir()
        .ok_or_else(|| SandboxError::Unavailable("no user profile folder".into()))?;
    let mut changed = Vec::new();
    for dir in home.ancestors().skip(1) {
        appcontainer::let_through(&profile, dir).map_err(|e| {
            SandboxError::Unavailable(format!(
                "cannot let the AppContainer through {} (run this as an administrator): {e}",
                dir.display()
            ))
        })?;
        changed.push(dir.to_owned());
    }
    changed.extend(grant_toolchains(&profile, false)?);
    Ok(changed)
}

/// Whether `setup` has run: the folders above the user's let the profile through.
fn is_set_up(profile: &Profile) -> Result<bool, SandboxError> {
    let Some(home) = env::home_dir() else {
        return Ok(true);
    };
    for dir in home.ancestors().skip(1) {
        let acl = icacls(dir, &[]).map_err(SandboxError::Unavailable)?;
        if !acl.contains(profile.sid()) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Lets the profile read and run the toolchains, beneath the user's profile folder when `home`
/// (whose ACLs grant AppContainers nothing), else outside it (the admin's to grant): CARGO_HOME,
/// RUSTUP_HOME and the folders of the first python.exe and node.exe on PATH. Once per folder:
/// the entry stays, files added later inherit it, and a grant on a toolchain's 100k files takes
/// minutes. The folders it changed.
// ponytail: the entries outlive larch. upgrade: the uninstaller revokes them.
fn grant_toolchains(profile: &Profile, home: bool) -> Result<Vec<PathBuf>, SandboxError> {
    let Some(user) = env::home_dir() else {
        return Ok(Vec::new());
    };
    let path = env::var_os("PATH").unwrap_or_default();
    let on_path = ["python.exe", "node.exe"]
        .into_iter()
        .filter_map(|exe| env::split_paths(&path).find(|dir| dir.join(exe).is_file()));
    let dirs = toolchains().map(|(_, dir)| dir).chain(on_path);
    let mut changed = Vec::new();
    // Never WindowsApps: its aliases start Store apps, which run as their own package.
    for dir in dirs.filter(|dir| dir.starts_with(&user) == home && !dir.ends_with("WindowsApps")) {
        let unavailable = |why| {
            SandboxError::Unavailable(format!(
                "cannot grant {} to the AppContainer: {why}",
                dir.display()
            ))
        };
        let acl = icacls(&dir, &[]).map_err(unavailable)?;
        if !acl.contains(profile.sid()) {
            icacls(&dir, &["/grant", &format!("*{}:(OI)(CI)RX", profile.sid())])
                .map_err(unavailable)?;
            changed.push(dir);
        }
    }
    Ok(changed)
}

/// Lets the profile traverse and read the attributes of each folder above `path`, that folder
/// only, but those above the user's profile folder, which `setup` let through as an
/// administrator: git and cmd.exe check every component of a path, and fail inside a folder
/// whose parents grant AppContainers nothing. No listing, no reading; the entries stay, as the
/// toolchains' do, and writing one again changes nothing. A folder the user may not change the
/// entries of (a drive's top other than C:) is refused by name.
fn grant_ancestors(profile: &Profile, path: &Path) -> Result<(), SandboxError> {
    let home = env::home_dir();
    let above_home = |dir: &Path| {
        home.as_ref()
            .is_some_and(|h| h.starts_with(dir) && h != dir)
    };
    for dir in path.ancestors().skip(1).filter(|dir| !above_home(dir)) {
        appcontainer::let_through(profile, dir).map_err(|e| {
            SandboxError::Unavailable(format!(
                "cannot let the AppContainer through {}: {e}",
                dir.display()
            ))
        })?;
    }
    Ok(())
}

/// Runs icacls on `path`: what it printed, or its error text when it fails.
fn icacls(path: &Path, args: &[&str]) -> Result<String, String> {
    let out = no_window(&mut Command::new("icacls"))
        .arg(path)
        .args(args)
        .output()
        .map_err(|e| format!("cannot run icacls: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout) + String::from_utf8_lossy(&out.stderr);
    if out.status.success() {
        return Ok(text.into_owned());
    }
    Err(text.trim().to_owned())
}

/// The profile's name and SID, with the toolchains granted to it: once per process, as two
/// CreateAppContainerProfile calls at once on one name fail, and a failure is tried again on the
/// next run.
fn container() -> Result<&'static (String, Profile), SandboxError> {
    static CONTAINER: OnceLock<(String, Profile)> = OnceLock::new();
    static MAKING: Mutex<()> = Mutex::new(());
    let _one = MAKING.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(container) = CONTAINER.get() {
        return Ok(container);
    }
    let name = profile_name()?;
    let profile = appcontainer::profile(&name).map_err(|e| {
        SandboxError::Unavailable(format!(
            "cannot create the AppContainer profile {name}: {e}"
        ))
    })?;
    if !is_set_up(&profile)? {
        return Err(SandboxError::Unavailable(
            "git and cmd.exe cannot run in the AppContainer until `larch-agent sandbox-setup` \
             has run once from an administrator prompt"
                .into(),
        ));
    }
    grant_toolchains(&profile, true)?;
    Ok(CONTAINER.get_or_init(|| (name, profile)))
}

/// The profile for this user's data directory: the same name, and so the same SID, every run.
fn profile_name() -> Result<String, SandboxError> {
    let dir = larch::chat::data_dir().map_err(|e| SandboxError::Unavailable(e.to_string()))?;
    // FNV-1a: stable across Rust releases, unlike DefaultHasher, so the SID stays.
    let hash = dir
        .as_os_str()
        .as_encoded_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
    Ok(format!("larch.sandbox.{hash:016x}"))
}

/// What the command sees: ours trimmed to what a toolchain needs, MSVC's linker environment, the
/// private tmp as its home.
fn environment(tmp: &Path) -> Vec<(OsString, OsString)> {
    // Found out here: rustc's own search for Visual Studio fails inside the container, and
    // outside a developer prompt ours has none of it. Inside one the finder returns nothing and
    // ours already holds it. Once per process: where its COM lookup fails it starts vswhere,
    // with a console window of its own.
    static MSVC: OnceLock<Vec<(OsString, OsString)>> = OnceLock::new();
    let msvc = MSVC.get_or_init(|| {
        find_msvc_tools::find_tool(env::consts::ARCH, "link.exe")
            .map(|tool| tool.env().into_iter().cloned().collect())
            .unwrap_or_default()
    });
    let get = |var: &str| {
        let found = msvc.iter().find(|(name, _)| name.eq_ignore_ascii_case(var));
        found
            .map(|(_, value)| value.clone())
            .or_else(|| env::var_os(var))
    };
    // LANG is not Windows' own, but tools ported from Unix read it.
    let ours = ["PATH", "SystemRoot", "LANG", "LIB", "LIBPATH", "INCLUDE"]
        .into_iter()
        .filter_map(|var| Some((var.into(), get(var)?)));
    let toolchains = toolchains().map(|(var, path)| (var.into(), path.into()));
    // CreateProcess refuses an AppContainer whose environment has no LOCALAPPDATA.
    let private = ["HOME", "USERPROFILE", "LOCALAPPDATA"].map(|var| (var.into(), tmp.into()));
    // Offline at once, rather than after a failed fetch of the index.
    let offline = [("CARGO_NET_OFFLINE".into(), "true".into())];
    let mut env: Vec<_> = ours
        .chain(toolchains)
        .chain(private)
        .chain(offline)
        .collect();
    // The block CreateProcess takes is sorted, case-insensitively.
    env.sort_by_key(|(var, _): &(OsString, OsString)| var.to_ascii_uppercase());
    env
}

/// `cmd` under cmd.exe in the AppContainer, inside a Job Object of its own with `limits`.
fn start(
    cmd: &str,
    worktree: &Path,
    env: &[(OsString, OsString)],
    limits: &Limits,
    profile: &Profile,
) -> io::Result<(Child, Job)> {
    let job = appcontainer::job(limits.mem_bytes, limits.max_procs, limits.cpu_s)?;
    let system = env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
    let program = Path::new(&system).join(r"System32\cmd.exe");
    // /s: cmd strips the outer quotes and runs the rest as typed; /d: no AutoRun.
    let line = format!("cmd.exe /d /s /c \"{cmd}\"");
    let child = appcontainer::spawn(profile, &job, &program, line.as_ref(), env, worktree)?;
    Ok((child, job))
}
