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
/// above what the uid runs now. A ulimit the shell refuses stops it before `cmd` runs.
// ponytail: no memory cap: xnu does not enforce RLIMIT_AS. ceiling: the timeout and CPU cap bound
// a runaway. upgrade: a jetsam limit through launchd if a tool command exhausts memory.
pub(super) fn command(cmd: &str, writable: &[PathBuf], limits: &Limits) -> Command {
    // $0 is `cmd`, $1 the profile, and the folders from $2 on.
    let params: String = (0..writable.len())
        .map(|i| format!(" -D W{i}=\"${{{}}}\"", i + 2))
        .collect();
    let script = format!(
        "ulimit -t {} && ulimit -u $(($(ps -o pid= -U \"$(id -u)\" | wc -l) + {})) && exec sandbox-exec{params} -p \"$1\" /bin/sh -c \"$0\"",
        limits.cpu_s, limits.max_procs,
    );
    let mut command = Command::new("/bin/sh");
    command
        .args(["-c", &script, cmd])
        .arg(profile(writable.len()))
        .args(writable);
    command
}

/// Spawns `command`; sandbox-exec applies the profile in the child.
pub(super) fn spawn(command: &mut Command, _writable: &[PathBuf]) -> Result<Child, SandboxError> {
    Ok(command.spawn()?)
}
