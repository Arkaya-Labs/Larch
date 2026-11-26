//! macOS: sandbox-exec denies the network, AppleEvents and every write outside the worktree (or
//! the folders of it a read-only run may write) and the private tmp; ulimit sets the caps.

use std::{
    path::PathBuf,
    process::{Child, Command},
};

use super::{Limits, SandboxError};

/// The sandbox-exec profile, before the folders it may write (`profile`); the paths come in as
/// parameters, so none needs quoting.
// ponytail: Mach services stay reachable, so a launchd job submitted through launchctl (or a copy
// of it in the worktree) would run unconfined. ceiling: a tool command that talks to launchd.
// upgrade: (deny mach-lookup) with an allow-list, measured on the M5 runner.
const PROFILE: &str = r#"(version 1)
(allow default)
(deny network*)
(deny appleevent-send)
(deny file-write*)
(allow file-write* (literal "/dev/null"))"#;

/// `PROFILE` with a write allowed beneath each of `n` folders, the parameters W0 to W`n-1`.
fn profile(n: usize) -> String {
    let subpaths: String = (0..n)
        .map(|i| format!(" (subpath (param \"W{i}\"))"))
        .collect();
    format!("{PROFILE}\n(allow file-write*{subpaths})")
}

/// `cmd` under ulimit and sandbox-exec, writing only beneath `writable` (`super::writable`), one
/// at least. RLIMIT_NPROC counts every process of our uid on macOS, so the cap is `max_procs`
/// above the uid's other processes (`others`); the command's own shell is one of its
/// `max_procs`, as on Linux. A ulimit the shell refuses stops it before `cmd` runs. Without a
/// count, the command is a shell that prints why and fails, so the probe reports it.
// ponytail: no memory cap: xnu does not enforce RLIMIT_AS. ceiling: the timeout and CPU cap bound
// a runaway. upgrade: a jetsam limit through launchd if a tool command exhausts memory.
// ponytail: a process of our uid that ends during the run gives the command its place.
// ceiling: the uid's own churn. upgrade: a per-run uid, which needs a privileged helper.
pub(super) fn command(cmd: &str, writable: &[PathBuf], limits: &Limits) -> Command {
    let others = match others() {
        Ok(n) => n,
        Err(why) => {
            let mut refusal = Command::new("/bin/sh");
            refusal.args(["-c", "echo \"$0\" >&2; exit 1", &why]);
            return refusal;
        }
    };
    // $0 is `cmd`, $1 the profile, and the folders from $2 on.
    let params: String = (0..writable.len())
        .map(|i| format!(" -D W{i}=\"${{{}}}\"", i + 2))
        .collect();
    let script = format!(
        "ulimit -t {} && ulimit -u {} && exec sandbox-exec{params} -p \"$1\" /bin/sh -c \"$0\"",
        limits.cpu_s,
        others + limits.max_procs as usize,
    );
    let mut command = Command::new("/bin/sh");
    command
        .args(["-c", &script, cmd])
        .arg(profile(writable.len()))
        .args(writable);
    command
}

/// P_ADOPTPERSONA in a process's flags (xnu's bsd/sys/proc.h; `ps` shows them in hex): it took
/// on our uid through a persona, as the app extensions launchd starts do, and RLIMIT_NPROC still
/// counts it against the uid that started it, not ours.
const ADOPTED_PERSONA: u64 = 0x0400_0000;

/// The processes RLIMIT_NPROC counts against our real uid, less the `ps` that lists them: `ps -A`
/// shows every process with its real uid and flags, and our own pid's row says which uid is ours.
/// Counted here rather than in the command's shell, where the count also took in the shell's
/// own `ps`, `wc` and subshell, each of which then left the command a place; the persona rows
/// left it one each too.
fn others() -> Result<usize, String> {
    let out = Command::new("/bin/ps")
        .args(["-A", "-o", "ruid=,flags=,pid="])
        .output()
        .map_err(|e| format!("cannot run ps to count this user's processes: {e}"))?;
    if !out.status.success() {
        let why = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "ps could not count this user's processes: {}",
            why.trim()
        ));
    }
    let listed = String::from_utf8_lossy(&out.stdout);
    let mut rows = Vec::new();
    for row in listed.lines().filter(|row| !row.trim().is_empty()) {
        let columns: Vec<&str> = row.split_whitespace().collect();
        let &[uid, flags, pid] = columns.as_slice() else {
            return Err(format!(
                "ps listed a row that is not a uid, flags and pid: {row:?}"
            ));
        };
        let flags = u64::from_str_radix(flags, 16)
            .map_err(|_| format!("ps listed flags that are not hex: {row:?}"))?;
        rows.push((uid, flags, pid));
    }
    let me = std::process::id().to_string();
    let uid = (rows.iter())
        .find_map(|&(uid, _, pid)| (pid == me).then_some(uid))
        .ok_or("ps did not list this process, so its user's processes cannot be counted")?;
    let counted = (rows.iter())
        .filter(|&&(u, flags, _)| u == uid && flags & ADOPTED_PERSONA == 0)
        .count();
    Ok(counted.saturating_sub(1))
}

/// Spawns `command`; sandbox-exec applies the profile in the child.
pub(super) fn spawn(command: &mut Command, _writable: &[PathBuf]) -> Result<Child, SandboxError> {
    Ok(command.spawn()?)
}
