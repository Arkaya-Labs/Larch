//! Restarting supervisor (architecture 10.3): runs a child, restarts it when it dies, and kills
//! whatever the child started.

use std::{
    env,
    ffi::OsString,
    io::{self, BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    process::{self, Child, ChildStderr, Command, ExitStatus, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use serde_json::json;

/// Exit code of an engine fault (architecture 9.11): restart at once.
const FAULT_EXIT: i32 = 3;
/// A child stderr line that marks its first Ready; the log's `ready_ms` runs from spawn to it and its
/// `recovery_ms` ends at it.
// ponytail: a stderr line stands in for the ring's Ready HealthEvent. ceiling: chat and bench must
// print it. upgrade: read the HealthEvent from the ring once the supervisor maps it.
pub const READY_LINE: &str = "larch: ready";

/// Set by `run` for its child: the address of the supervisor's guard listener.
const ADDR_VAR: &str = "LARCH_SUPERVISOR_ADDR";
/// How long a child waits to reach its supervisor's guard before giving up.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Delay before a restart after the 1st, 2nd, ... consecutive early death; the last repeats.
const BACKOFF_MS: [u64; 5] = [0, 250, 500, 1000, 2000];
/// A run at least this long counts as healthy and resets the backoff.
const HEALTHY: Duration = Duration::from_secs(60);
/// How long a run's stderr may stay open after the child exits. Only a leftover process the
/// cleanup could not reach keeps it open longer, and the next run must not wait for that.
const STDERR_GRACE: Duration = Duration::from_secs(1);

/// Delay before the next start, given how the last run ended and whether it reached Ready.
/// `deaths` counts consecutive early deaths, plain ones then faults.
fn next_delay(
    deaths: &mut [usize; 2],
    code: Option<i32>,
    uptime: Duration,
    ready: bool,
) -> Duration {
    if uptime >= HEALTHY {
        *deaths = [0; 2];
    }
    // Exit 3 restarts at once, as specified, and does not advance the plain backoff. Faults keep a
    // count of their own, so a child that faults on every start (a lost device, a broken driver)
    // backs off from its second fault in a row and does not spin. A run that reached Ready starts
    // that count afresh, so a child that faults while it serves restarts at once every time.
    if ready {
        deaths[1] = 0;
    }
    let fault = code == Some(FAULT_EXIT);
    let count = &mut deaths[usize::from(fault)];
    let ms = BACKOFF_MS[(*count).min(BACKOFF_MS.len() - 1)];
    *count += 1;
    Duration::from_millis(ms)
}

/// What `run` waits on while a child runs.
enum Event {
    /// The child printed `READY_LINE`, this long after its spawn.
    Ready(Duration),
    /// The child exited: its status, and the wall-clock time and instant when `wait()` returned.
    Exited(io::Result<ExitStatus>, SystemTime, Instant),
}

/// Copies the child's stderr to ours and sends `Event::Ready` on the first `READY_LINE`; returns
/// its last non-empty line.
fn tee_stderr(
    stderr: ChildStderr,
    start: Instant,
    ready_tx: &mpsc::Sender<Event>,
) -> Option<String> {
    let (mut last, mut ready) = (None, false);
    let (mut reader, mut line, mut ours) = (BufReader::new(stderr), Vec::new(), io::stderr());
    // Capped, so a child that never prints a newline cannot grow `line` without bound.
    // A read error means the pipe is gone, the same as EOF for this purpose.
    while (&mut reader)
        .take(1 << 16)
        .read_until(b'\n', &mut line)
        .is_ok_and(|n| n > 0)
    {
        // Our own stderr failing must not stop us draining the child's, or the child blocks on it.
        let _ = ours.write_all(&line);
        let text = String::from_utf8_lossy(&line);
        let text = text.trim();
        if !text.is_empty() {
            if !ready && text == READY_LINE {
                ready = true;
                // Fails only once `run` has returned, when nobody wants it.
                let _ = ready_tx.send(Event::Ready(start.elapsed()));
            }
            last = Some(text.to_owned());
        }
        line.clear();
    }
    last
}

/// What stops everything the child started when it drops: after the child exits, or on any early
/// return from `run`. On Windows the job handle's drop does it.
#[cfg(unix)]
struct Guard(Option<u32>); // the child's own process group, when it has one
#[cfg(windows)]
struct Guard {
    _job: win32job::Job,
}

impl Guard {
    /// Starts `cmd` under the guard: its own process group on Unix, a Job Object on Windows.
    fn spawn(cmd: &mut Command) -> Result<(Child, Guard)> {
        #[cfg(unix)]
        {
            use std::io::IsTerminal;
            // On a terminal the child stays in our foreground group: in a group of its own it would
            // be a background job, stopped (SIGTTIN/SIGTTOU) on its first read or mode change.
            // ponytail: no group kill for a child on a terminal. ceiling: a grandchild of an
            // interactive child can outlive it. upgrade: hand it the terminal with tcsetpgrp if
            // chat ever starts processes.
            let own_group = !(io::stdin().is_terminal() || io::stdout().is_terminal());
            if own_group {
                std::os::unix::process::CommandExt::process_group(cmd, 0);
            }
            let child = cmd.spawn()?;
            let group = own_group.then(|| child.id());
            Ok((child, Guard(group)))
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            let mut limits = win32job::ExtendedLimitInfo::new();
            limits.limit_kill_on_job_close();
            let job = win32job::Job::create_with_limit_info(&limits)?;
            let mut child = cmd.spawn()?;
            // ponytail: assigned after spawn, so a grandchild started in the first instants escapes.
            // ceiling: the window is microseconds. upgrade: CREATE_SUSPENDED if the R5 kill test
            // leaks GPU memory.
            if let Err(e) = job.assign_process(child.as_raw_handle() as isize) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e.into());
            }
            Ok((child, Guard { _job: job }))
        }
    }
}

/// SIGKILLs `target`, a pid or `-<pgid>`; false when kill(1) fails or cannot be run.
#[cfg(unix)]
fn kill(target: &str) -> bool {
    // ponytail: kill(1), because std has no killpg and unsafe is forbidden here. ceiling: one
    // process spawn per restart. upgrade: rustix's kill_process_group if kill is missing.
    // Non-zero status is the normal "nothing left to kill".
    let killed = Command::new("kill")
        .args(["-KILL", "--", target])
        .stderr(Stdio::null())
        .status();
    if let Err(e) = &killed {
        let _ = writeln!(io::stderr(), "larch supervise: cannot run kill(1): {e}");
    }
    killed.is_ok_and(|s| s.success())
}

#[cfg(unix)]
impl Drop for Guard {
    /// Kills what is left of the child's process group.
    fn drop(&mut self) {
        if let Some(group) = self.0 {
            kill(&format!("-{group}"));
        }
    }
}

/// Kills and reaps the direct child `pid` on an early return while it runs: a child on a terminal
/// has no group for the guard to kill. Windows needs none: the Job Object's drop kills the whole
/// tree, the direct child included.
#[cfg(unix)]
fn end_child(pid: u32, events: &mpsc::Receiver<Event>) {
    // An `Exited` already sent means the pid is reaped and may be reused: no kill.
    if events.try_iter().any(|e| matches!(e, Event::Exited(..))) {
        return;
    }
    // Window: the waiter's wait() to its send; sequential Linux pids reuse one only after a wrap.
    if kill(&pid.to_string()) {
        while !matches!(events.recv(), Ok(Event::Exited(..)) | Err(_)) {}
    }
}

/// `t` in unix nanoseconds.
fn unix_ns(t: SystemTime) -> Result<u64> {
    let ns = t
        .duration_since(UNIX_EPOCH)
        .context("clock before 1970")?
        .as_nanos();
    u64::try_from(ns).context("clock past 2554")
}

/// Writes the ready line of the run restart `n` started, Ready at `ready`: `recovery_ms` from
/// `died`, the exit it restarted after. The first run (`died` None) gets none.
fn write_ready(log: &mut impl Write, n: u64, died: Option<Instant>, ready: Instant) -> Result<()> {
    let Some(died) = died else {
        return Ok(());
    };
    let ms = ready.saturating_duration_since(died).as_millis() as u64;
    Ok(writeln!(
        log,
        "{}",
        json!({ "restart": n, "recovery_ms": ms })
    )?)
}

/// Ties this process to the supervisor that started it, if any (`LARCH_SUPERVISOR_ADDR` is set):
/// connects to the supervisor's guard and exits as soon as that connection ends, so a supervised
/// `larch` never outlives its supervisor. `main` calls it before anything else.
pub fn exit_with_supervisor() -> Result<()> {
    let Some(raw) = env::var_os(ADDR_VAR) else {
        return Ok(());
    };
    let addr: SocketAddr = raw
        .to_str()
        .and_then(|a| a.parse().ok())
        .with_context(|| format!("{ADDR_VAR} is not an address: {raw:?}"))?;
    let mut conn = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
        .with_context(|| format!("cannot reach the supervisor at {addr}"))?;
    thread::Builder::new()
        .name("supervisor-guard".into())
        .spawn(move || {
            // The supervisor never accepts or writes, so a read only returns once it is gone.
            let mut byte = [0u8; 1];
            let _ = conn.read(&mut byte);
            process::exit(1);
        })?;
    Ok(())
}

/// Runs `command` until it exits 0, restarting it each time it dies. Errors if the command cannot be
/// started or `log` cannot be written, after the guard has cleaned up as at the end of any run.
///
/// `log` gets two kinds of JSON line, paired by `restart`, n for the nth restart. A death line for
/// each run that dies and is restarted, written once the run is cleaned up, before the backoff
/// sleep and the respawn: `restart`, `exit` (null when a signal killed it), `stderr_tail` (its last
/// non-empty stderr line), `exited_ns` (when `wait()` returned, unix nanoseconds), `ready_ms` (its
/// spawn to `READY_LINE`, null if it never printed it) and `backoff_ms`. A ready line,
/// `{"restart": n, "recovery_ms": N}`, when the run restart n started first prints `READY_LINE`:
/// from the death's `wait()` returning to that Ready, backoff included. The first run gets no
/// ready line, and a restarted run that dies before Ready gets none either. A clean exit 0 ends
/// `run` with no line. `stderr_tail` stands in for the fault from the last `HealthEvent`, if
/// flushed (architecture 10.3), until the supervisor reads the child's ring.
///
/// The child gets `LARCH_SUPERVISOR_ADDR` (see `exit_with_supervisor`). After exit 3 it also gets
/// `LARCH_RESTARTED_AT`, the `exited_ns` of the faulted run. On Unix only a child that calls
/// `exit_with_supervisor` (every `larch` command does, from `main`) dies when the supervisor is
/// killed; any other command's tree is cleaned up when that command exits, or by the Job Object on
/// Windows.
pub fn run(command: &[OsString], log: &mut impl Write) -> Result<()> {
    let (program, args) = command.split_first().context("no command to supervise")?;
    // ponytail: a loopback listener whose queued connections stand in for the inherited pipe: std
    // cannot hand a child an extra handle without unsafe. The OS resets them when this process
    // dies. ceiling: needs loopback networking. upgrade: an inherited pipe if a host
    // blocks loopback.
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    listener.set_nonblocking(true)?;
    let addr = listener.local_addr()?.to_string();
    let (mut failures, mut restarted_at): ([usize; 2], Option<u64>) = ([0; 2], None);
    // The restart that started this run (0 for the first run) and the exit it restarted after.
    let (mut restart, mut died): (u64, Option<Instant>) = (0, None);
    loop {
        let mut cmd = Command::new(program);
        cmd.args(args).env(ADDR_VAR, &addr).stderr(Stdio::piped());
        match restarted_at {
            Some(ns) => cmd.env("LARCH_RESTARTED_AT", ns.to_string()),
            None => cmd.env_remove("LARCH_RESTARTED_AT"),
        };
        let start = Instant::now();
        let (mut child, guard) =
            Guard::spawn(&mut cmd).with_context(|| format!("cannot start {program:?}"))?;
        #[cfg(unix)]
        let pid = child.id();
        let stderr = child.stderr.take().context("child stderr is not piped")?;
        let (tx, events) = mpsc::channel();
        let (tee_tx, tee) = mpsc::channel();
        let ready_tx = tx.clone();
        thread::spawn(move || tee_tx.send(tee_stderr(stderr, start, &ready_tx)));
        // Timed here, so neither our log writes nor the cleanup below delay it, and the next
        // child's recovery_ms leaves none of the downtime out.
        thread::spawn(move || {
            let status = child.wait();
            tx.send(Event::Exited(status, SystemTime::now(), Instant::now()))
        });

        let mut ready_ms = None;
        let (status, exited, at) = loop {
            match events.recv().context("the child waiter died")? {
                Event::Ready(d) => {
                    ready_ms = Some(d);
                    write_ready(log, restart, died, start + d).inspect_err(|_| {
                        #[cfg(unix)]
                        end_child(pid, &events);
                    })?;
                }
                Event::Exited(status, exited, at) => break (status?, exited, at),
            }
        };
        let exited_ns = unix_ns(exited)?;
        // Before reading the tee's result: a surviving grandchild holds the stderr pipe open.
        drop(guard);
        // Each child leaves its connection in the accept queue; a full queue would refuse the next.
        // Accepting and dropping also ends any guarded process the child left behind. An accept
        // error still takes one entry off the queue, so the queue cannot grow across runs.
        while listener.accept().is_ok() {}
        // Past the grace the tee keeps copying in the background and this run's tail is unknown.
        // ponytail: a reader blocked by an escaped grandchild lives as long as that process.
        // ceiling: one thread and one pipe per escapee, and only a child that leaves its process
        // group makes one (larch never does). upgrade: kill the session or the cgroup if a
        // supervised command starts daemons.
        let stderr_tail = tee.recv_timeout(STDERR_GRACE).unwrap_or_default();
        // A Ready printed just before the exit can lose the race to it; the tee sent it before returning.
        for event in events.try_iter() {
            if let Event::Ready(d) = event {
                ready_ms = Some(d);
                write_ready(log, restart, died, start + d)?;
            }
        }

        if status.success() {
            return Ok(());
        }
        let code = status.code();
        let delay = next_delay(
            &mut failures,
            code,
            at.duration_since(start),
            ready_ms.is_some(),
        );
        restarted_at = (code == Some(FAULT_EXIT)).then_some(exited_ns);
        restart += 1;
        died = Some(at);
        let death = json!({
            "restart": restart,
            "exit": code,
            "stderr_tail": stderr_tail,
            "exited_ns": exited_ns,
            "ready_ms": ready_ms.map(|d| d.as_millis() as u64),
            "backoff_ms": delay.as_millis() as u64,
        });
        writeln!(log, "{death}")?;
        thread::sleep(delay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_steps_then_caps_then_resets_after_a_healthy_run() {
        let ms = |deaths: &mut [usize; 2], code, secs| {
            next_delay(deaths, code, Duration::from_secs(secs), false).as_millis()
        };
        let mut f = [0; 2];
        let steps: Vec<_> = (0..7).map(|_| ms(&mut f, Some(1), 1)).collect();
        assert_eq!(steps, [0, 250, 500, 1000, 2000, 2000, 2000]);
        assert_eq!(
            ms(&mut f, Some(3), 1),
            0,
            "exit 3 restarts at once and does not advance the backoff"
        );
        let faults: Vec<_> = (0..3).map(|_| ms(&mut f, Some(3), 1)).collect();
        assert_eq!(faults, [250, 500, 1000], "faults in a row back off");
        let served = |deaths: &mut [usize; 2]| {
            next_delay(deaths, Some(3), Duration::from_secs(1), true).as_millis()
        };
        assert_eq!(served(&mut f), 0, "a fault after Ready restarts at once");
        assert_eq!(served(&mut f), 0, "and again, however many came before");
        assert_eq!(
            ms(&mut f, Some(3), 1),
            250,
            "a fault before Ready backs off"
        );
        assert_eq!(ms(&mut f, Some(1), 59), 2000, "59 s is not healthy");
        assert_eq!(ms(&mut f, None, 60), 0, "60 s healthy resets the backoff");
        assert_eq!(ms(&mut f, Some(1), 1), 250);
    }
}
