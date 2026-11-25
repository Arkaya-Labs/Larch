//! Supervisor tests. The fake child is this test binary run again: `fake_child` does nothing unless
//! its arguments (after the test name) say which behaviour to play.

use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use larch::supervise::{READY_LINE, exit_with_supervisor, run};

fn ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

/// Plays `play` with `args`: the command line a supervisor test hands to the supervisor.
fn fake(play: &str, dir: &Path, args: &[&str]) -> Vec<OsString> {
    let exe = env::current_exe().unwrap().into_os_string();
    let mut cmd = vec![
        exe,
        "--exact".into(),
        "--nocapture".into(),
        "fake_child".into(),
    ];
    cmd.extend(
        [play, dir.to_str().unwrap()]
            .iter()
            .chain(args)
            .map(Into::into),
    );
    cmd
}

fn temp(name: &str) -> PathBuf {
    let dir = env::temp_dir().join(format!("larch-supervise-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(20));
    }
}

/// A process that has exited but is not reaped yet (a zombie) counts as gone.
fn alive(pid: u32) -> bool {
    #[cfg(unix)]
    let out = Command::new("ps")
        .args(["-o", "stat=", "-p", &pid.to_string()])
        .output()
        .unwrap();
    #[cfg(windows)]
    let out = Command::new("tasklist")
        .args(["/NH", "/FI", &format!("PID eq {pid}")])
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&out.stdout);
    let out = out.trim();
    #[cfg(unix)]
    return !out.is_empty() && !out.starts_with('Z');
    #[cfg(windows)]
    return out.contains(&pid.to_string());
}

/// Behaviours the supervisor tests ask of the fake child. A plain run of the test does nothing.
#[test]
fn fake_child() {
    let args: Vec<String> = env::args()
        .skip_while(|a| a != "fake_child")
        .skip(1)
        .collect();
    let [play, dir, rest @ ..] = &args[..] else {
        return;
    };
    let dir = Path::new(dir);
    match play.as_str() {
        // seq <codes> <delay_ms>: on run n, arm the guard as `larch` does, report Ready unless
        // codes[n] ends in `q`, wait, print two lines, record the exit time and exit codes[n].
        "seq" => {
            exit_with_supervisor().unwrap();
            let runs = dir.join("runs");
            let mut log = fs::read_to_string(&runs).unwrap_or_default();
            let n = log.lines().count();
            let restarted = env::var("LARCH_RESTARTED_AT").unwrap_or("-".into());
            log += &format!("{} {restarted}\n", ns());
            fs::write(&runs, log).unwrap();
            // Past the end of the list, stop the sequence instead of looping forever.
            let code = rest[0].split(',').nth(n).unwrap_or("0");
            if !code.ends_with('q') {
                eprintln!("{READY_LINE}");
            }
            thread::sleep(Duration::from_millis(rest[1].parse().unwrap()));
            eprintln!("fault in run {n}");
            eprintln!("shutdown of run {n}");
            fs::write(dir.join(format!("exit-{n}")), ns().to_string()).unwrap();
            std::process::exit(code.trim_end_matches('q').parse().unwrap());
        }
        // grandchild: start a long sleeper that shares our stderr, then exit 0 at once.
        "grandchild" => {
            #[cfg(unix)]
            let mut sleeper = Command::new("sleep");
            #[cfg(unix)]
            sleeper.arg("60");
            #[cfg(windows)]
            let mut sleeper = Command::new("ping");
            #[cfg(windows)]
            sleeper.args(["-n", "60", "127.0.0.1"]);
            // Left running on purpose: the supervisor, not this test, must clean it up.
            #[allow(clippy::zombie_processes)]
            let sleeper = sleeper.stdout(Stdio::null()).spawn().unwrap();
            fs::write(dir.join("pid"), sleeper.id().to_string()).unwrap();
        }
        // guard: arm the guard as `larch` does, record our pid and our parent's, then idle.
        "guard" => {
            exit_with_supervisor().unwrap();
            #[cfg(unix)]
            let pids = format!(
                "{} {}",
                std::process::id(),
                std::os::unix::process::parent_id()
            );
            #[cfg(windows)]
            let pids = std::process::id().to_string();
            fs::write(dir.join("pids.tmp"), pids).unwrap();
            fs::rename(dir.join("pids.tmp"), dir.join("pids")).unwrap();
            thread::sleep(Duration::from_secs(60));
        }
        // hold: run 0 exits 3 at once; run 1 waits for <dir>/go, reports Ready, then waits for
        // <dir>/release and exits 0.
        "hold" => {
            exit_with_supervisor().unwrap();
            let runs = dir.join("runs");
            let n = fs::read_to_string(&runs)
                .unwrap_or_default()
                .lines()
                .count();
            fs::write(&runs, "run\n".repeat(n + 1)).unwrap();
            if n == 0 {
                std::process::exit(3);
            }
            let wait = |file| {
                while !dir.join(file).exists() {
                    thread::sleep(Duration::from_millis(10));
                }
            };
            wait("go");
            eprintln!("{READY_LINE}");
            wait("release");
        }
        // vanish: move our own executable out of the way and exit 1, so no restart can start.
        "vanish" => {
            fs::rename(env::current_exe().unwrap(), dir.join("gone")).unwrap();
            std::process::exit(1);
        }
        other => panic!("unknown play {other}"),
    }
}

/// Parses a JSON-lines log.
fn parse(log: &[u8]) -> Vec<serde_json::Value> {
    String::from_utf8(log.to_vec())
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// Splits a log into its death lines (`exit`), numbered `restart` 1, 2, ... in order, and its ready
/// lines (`recovery_ms`), each exactly a `restart` and a `recovery_ms` number.
fn split(log: Vec<u8>) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    let (deaths, readies): (Vec<_>, Vec<_>) = parse(&log)
        .into_iter()
        .partition(|l| l.get("exit").is_some());
    for (n, d) in deaths.iter().enumerate() {
        assert_eq!(d["restart"], n as u64 + 1, "{d}");
    }
    for r in &readies {
        assert!(
            r["restart"].as_u64().is_some()
                && r["recovery_ms"].as_u64().is_some()
                && r.as_object().unwrap().len() == 2,
            "{r}"
        );
    }
    (deaths, readies)
}

/// The `restart` of each line.
fn restarts(lines: &[serde_json::Value]) -> Vec<u64> {
    lines
        .iter()
        .map(|l| l["restart"].as_u64().unwrap())
        .collect()
}

#[test]
fn restarts_back_off_and_exit_3_restarts_at_once() {
    let dir = temp("seq");
    let mut log = Vec::new();
    let began = Instant::now();
    // Three plain failures, one fault, then a clean exit.
    run(&fake("seq", &dir, &["1,1,1,3,0", "10"]), &mut log).unwrap();
    let total = began.elapsed();

    let (lines, readies) = split(log);
    let backoffs: Vec<_> = lines
        .iter()
        .map(|l| l["backoff_ms"].as_u64().unwrap())
        .collect();
    assert_eq!(backoffs, [0, 250, 500, 0]);
    let exits: Vec<_> = lines.iter().map(|l| l["exit"].as_i64().unwrap()).collect();
    assert_eq!(exits, [1, 1, 1, 3]);
    // The last line, whatever it says: not a fault report.
    assert_eq!(lines[3]["stderr_tail"], "shutdown of run 3");
    assert!(
        lines.iter().all(|l| l["ready_ms"].as_u64().is_some()),
        "every dead run printed Ready: {lines:?}"
    );
    assert_eq!(restarts(&readies), [1, 2, 3, 4], "none for the first run");
    // Recovery runs from the exit, so it covers the backoff sleep too.
    for (r, ms) in readies.iter().zip(&backoffs) {
        assert!(
            r["recovery_ms"].as_u64().unwrap() >= *ms,
            "{r}, backoff {ms}"
        );
    }

    let runs = fs::read_to_string(dir.join("runs")).unwrap();
    let runs: Vec<(u128, &str)> = runs
        .lines()
        .map(|l| {
            l.split_once(' ')
                .map(|(t, r)| (t.parse().unwrap(), r))
                .unwrap()
        })
        .collect();
    assert_eq!(runs.len(), 5);
    // Each restart waited at least its backoff after the previous child started and slept.
    for (i, ms) in backoffs.iter().enumerate() {
        let gap = Duration::from_nanos((runs[i + 1].0 - runs[i].0) as u64);
        assert!(gap >= Duration::from_millis(10 + ms), "gap {i} was {gap:?}");
    }
    assert!(total >= Duration::from_millis(750));
    // LARCH_RESTARTED_AT arrives after the fault and only then, and is a time before the restart.
    let restarted: Vec<_> = runs.iter().map(|r| r.1).collect();
    assert_eq!(&restarted[..4], ["-"; 4]);
    // The faulted child's exit, observed by the supervisor: after the child's last act, before
    // the next start.
    let at: u128 = restarted[4].parse().unwrap();
    assert_eq!(
        lines[3]["exited_ns"].as_u64().unwrap() as u128,
        at,
        "exited_ns is the LARCH_RESTARTED_AT the next child got"
    );
    let exited: u128 = fs::read_to_string(dir.join("exit-3"))
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        exited <= at && at <= runs[4].0,
        "{exited} <= {at} <= {}",
        runs[4].0
    );
}

/// A `Write` that the test can read while `run` is still going.
#[derive(Clone, Default)]
struct SharedLog(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
impl std::io::Write for SharedLog {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The death line goes out as soon as the run has died, before the restarted child is Ready; the
/// ready line goes out at that Ready, while the child still runs.
#[test]
fn the_death_line_comes_at_the_death_and_the_ready_line_at_ready() {
    let dir = temp("hold");
    let log = SharedLog::default();
    let mut writer = log.clone();
    let cmd = fake("hold", &dir, &[]);
    let sup = thread::spawn(move || run(&cmd, &mut writer));
    // `writeln!` reaches the log in pieces: parse whole lines only.
    let lines = || {
        let log = log.0.lock().unwrap();
        parse(&log[..log.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1)])
    };
    // Run 1 waits for `go` before it prints Ready.
    wait_for("the death line", || !lines().is_empty());
    let death = &lines()[0];
    assert_eq!(death["restart"], 1, "{death}");
    assert_eq!(death["exit"], 3, "{death}");
    assert!(death["exited_ns"].as_u64().is_some(), "{death}");
    assert_eq!(
        death.get("ready_ms"),
        Some(&serde_json::Value::Null),
        "{death}"
    );
    fs::write(dir.join("go"), "").unwrap();
    wait_for("the ready line", || lines().len() == 2);
    assert!(!sup.is_finished(), "run 1 is still up");
    let ready = &lines()[1];
    assert_eq!(ready["restart"], 1, "{ready}");
    assert!(ready["recovery_ms"].as_u64().is_some(), "{ready}");
    fs::write(dir.join("release"), "").unwrap();
    sup.join().unwrap().unwrap();
    assert_eq!(lines().len(), 2, "{:?}", lines());
}

/// A child that prints Ready and exits straight away still gets its ready line: the race between
/// the stderr reader and the exit must not drop it.
#[test]
fn ready_is_kept_when_the_child_exits_right_after_it() {
    let dir = temp("quick");
    let mut log = Vec::new();
    // Delay 0: run 1 prints Ready, then exits 3 immediately, 40 times over.
    let codes = format!("{}0", "3,".repeat(40));
    run(&fake("seq", &dir, &[&codes, "0"]), &mut log).unwrap();
    let (lines, readies) = split(log);
    assert_eq!(lines.len(), 40);
    assert!(
        lines.iter().all(|l| l["ready_ms"].as_u64().is_some()),
        "{lines:?}"
    );
    assert_eq!(restarts(&readies), (1..=40).collect::<Vec<_>>());
}

/// A restarted run that dies before Ready gets no ready line, and its death line has no `ready_ms`.
#[test]
fn a_run_that_dies_before_ready_has_no_ready_line() {
    let dir = temp("unready");
    let mut log = Vec::new();
    // Run 0 is Ready and exits 3; run 1 exits 1 without Ready; run 2 is Ready and exits 0.
    run(&fake("seq", &dir, &["3,1q,0", "10"]), &mut log).unwrap();
    let (lines, readies) = split(log);
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(lines[0]["ready_ms"].as_u64().is_some(), "{lines:?}");
    assert_eq!(
        lines[1].get("ready_ms"),
        Some(&serde_json::Value::Null),
        "{lines:?}"
    );
    assert_eq!(restarts(&readies), [2], "{readies:?}");
}

/// A restart that cannot be started still leaves the line about the run that died.
#[test]
fn a_failed_restart_still_logs_the_run_that_died() {
    let dir = temp("vanish");
    // A copy, so the play can move it away without touching the test binary itself.
    let exe = dir.join(env::current_exe().unwrap().file_name().unwrap());
    fs::copy(env::current_exe().unwrap(), &exe).unwrap();
    // A child another test forks while the copy is written inherits its write handle until it
    // execs, and Linux refuses to run a file open for writing (ETXTBSY). Wait for those to go.
    wait_for("the copy to be runnable", || {
        Command::new(&exe)
            .arg("--list")
            .stdout(Stdio::null())
            .status()
            .is_ok()
    });
    let mut cmd = fake("vanish", &dir, &[]);
    cmd[0] = exe.into();
    let mut log = Vec::new();
    assert!(run(&cmd, &mut log).is_err(), "the restart cannot start");
    let (lines, readies) = split(log);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_eq!(lines[0]["exit"], 1, "{lines:?}");
    assert!(readies.is_empty(), "a death line only: {readies:?}");
    // The copy of the test binary is tens of MB.
    let _ = fs::remove_dir_all(&dir);
}

/// The real `larch` binary supervising `play`, detached from any terminal.
fn supervise(play: &str, dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_larch"));
    cmd.args(["supervise", "--"]).args(fake(play, dir, &[]));
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    cmd
}

#[test]
fn child_tree_dies_with_the_child() {
    let dir = temp("tree");
    let began = Instant::now();
    assert!(supervise("grandchild", &dir).status().unwrap().success());
    // The sleeper holds the child's stderr open for 60 s unless the supervisor killed it first.
    assert!(began.elapsed() < Duration::from_secs(30));
    let pid = fs::read_to_string(dir.join("pid"))
        .unwrap()
        .parse()
        .unwrap();
    assert!(!alive(pid), "grandchild {pid} survived");
}

/// A real `larch` child (here an inner `larch supervise`) exits when its supervisor is killed,
/// and that takes down the inner one's own guarded child in turn.
#[test]
fn a_larch_child_dies_with_its_supervisor() {
    let dir = temp("guard");
    let mut outer = Command::new(env!("CARGO_BIN_EXE_larch"));
    outer
        .args([
            "supervise",
            "--",
            env!("CARGO_BIN_EXE_larch"),
            "supervise",
            "--",
        ])
        .args(fake("guard", &dir, &[]))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut outer = outer.spawn().unwrap();
    wait_for("the fake to start", || dir.join("pids").exists());
    let pids = fs::read_to_string(dir.join("pids")).unwrap();
    let mut pids = pids.split(' ').map(|p| p.parse::<u32>().unwrap());
    let fake = pids.next().unwrap();
    assert!(alive(fake));
    // Unix only: SIGKILL leaves no cleanup to the outer supervisor, so only the guard can end
    // `inner`. On Windows the outer Job Object ends both.
    #[cfg(unix)]
    let inner = pids.next().unwrap();
    #[cfg(unix)]
    assert!(alive(inner), "the fake's parent is the inner larch");
    outer.kill().unwrap();
    outer.wait().unwrap();
    #[cfg(unix)]
    wait_for("the inner larch to exit", || !alive(inner));
    wait_for("the fake to exit", || !alive(fake));
}

/// From a terminal, the supervised child can read it. In a process group of its own it would be a
/// background job, stopped by SIGTTIN on that read.
#[cfg(target_os = "linux")]
#[test]
fn a_child_on_a_terminal_can_read_it() {
    use std::io::Write;
    let line = format!("'{}' supervise -- head -n1", env!("CARGO_BIN_EXE_larch"));
    // script(1) runs the line on a new pseudo-terminal, fed from our pipe.
    let mut script = Command::new("script")
        .args(["-qec", &line, "/dev/null"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    script.stdin.take().unwrap().write_all(b"hello\n").unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let status = loop {
        if let Some(status) = script.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            script.kill().unwrap();
            script.wait().unwrap();
            panic!("the supervised child never finished reading the terminal");
        }
        thread::sleep(Duration::from_millis(20));
    };
    assert!(status.success());
}

#[test]
fn the_guard_stays_connectable_across_many_restarts() {
    let dir = temp("many");
    let codes = format!("{}0", "3,".repeat(140));
    let mut log = Vec::new();
    run(&fake("seq", &dir, &[&codes, "0"]), &mut log).unwrap();
    // A child that cannot reach the guard panics and exits 101 instead of 3.
    let log = String::from_utf8(log).unwrap();
    assert_eq!(
        log.lines().filter(|l| l.contains("\"exit\":3")).count(),
        140,
        "{log}"
    );
}

/// A leftover the group kill cannot reach (here `setsid` makes it a new session) keeps the child's
/// stderr open; the supervisor must still finish the run instead of waiting for it.
#[cfg(target_os = "linux")]
#[test]
fn an_escaped_grandchild_does_not_stall_the_supervisor() {
    let dir = temp("escaped");
    let pid_file = dir.join("pid");
    // The parent exits 0 only once the escapee has left its process group and recorded its pid.
    let script = format!(
        "setsid sh -c 'echo $$ > {0}.tmp && mv {0}.tmp {0} && exec sleep 60' & \
         while [ ! -s {0} ]; do sleep 0.01; done; exit 0",
        pid_file.display()
    );
    let began = Instant::now();
    run(&["sh".into(), "-c".into(), script.into()], &mut Vec::new()).unwrap();
    let took = began.elapsed();
    let pid = fs::read_to_string(&pid_file).unwrap();
    Command::new("kill")
        .args(["-KILL", pid.trim()])
        .status()
        .unwrap();
    assert!(took < Duration::from_secs(10), "took {took:?}");
}

/// A log that takes one line and then fails: a FIFO whose only reader is `head -n1`, so once that
/// is reaped the next write fails with EPIPE. Not /dev/full: the death line must get through.
#[cfg(target_os = "linux")]
fn one_line_log(dir: &Path) -> (PathBuf, std::process::Child) {
    let fifo = dir.join("log");
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    // A timeout, so a supervisor that never opens the FIFO fails the test instead of hanging it.
    let head = Command::new("timeout")
        .args(["20", "head", "-n1"])
        .arg(&fifo)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    (fifo, head)
}

/// A child whose restart is Ready only on <dir>/go: run 0 exits 3, run 1 records its pid, waits
/// for `go`, prints Ready and sleeps. It ignores SIGHUP: on a terminal, the hangup at the end of
/// script's session would kill it otherwise, and only the supervisor may.
#[cfg(target_os = "linux")]
fn ready_after_restart(dir: &Path, sleeper: bool) -> String {
    let d = dir.display();
    // The sleeper is in the child's group but is not the child: only the group kill ends it.
    let bg = if sleeper {
        format!("sleep 60 & echo $! > {d}/bg; ")
    } else {
        String::new()
    };
    format!(
        "trap \"\" HUP; if [ ! -e {d}/ran ]; then touch {d}/ran; exit 3; fi; {bg}\
         echo $$ > {d}/pid.tmp && mv {d}/pid.tmp {d}/pid; \
         while [ ! -e {d}/go ]; do sleep 0.01; done; echo \"{READY_LINE}\" >&2; exec sleep 60"
    )
}

/// Waits for `head` to take the death line and exit, then lets the restarted child be Ready.
#[cfg(target_os = "linux")]
fn fail_the_ready_line(head: std::process::Child, dir: &Path) {
    let out = head.wait_with_output().unwrap();
    let death = String::from_utf8_lossy(&out.stdout);
    assert!(death.contains("\"exit\":3"), "{death}");
    fs::write(dir.join("go"), "").unwrap();
}

/// A ready-line write that fails ends `run` with an error, and the child's process group dies
/// with it instead of outliving the supervisor.
#[cfg(target_os = "linux")]
#[test]
fn a_log_write_error_at_ready_kills_the_child_group() {
    let dir = temp("logfail");
    let (fifo, head) = one_line_log(&dir);
    // The log is the supervisor's stderr. No guard armed: it would end the child itself once the
    // supervisor is gone, and only the group kill may do that here.
    let log = fs::OpenOptions::new().write(true).open(&fifo).unwrap();
    let mut sup = Command::new(env!("CARGO_BIN_EXE_larch"))
        .args([
            "supervise",
            "--",
            "sh",
            "-c",
            &ready_after_restart(&dir, true),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log)
        .spawn()
        .unwrap();
    fail_the_ready_line(head, &dir);
    assert!(
        !sup.wait().unwrap().success(),
        "run returned the write error"
    );
    for f in ["pid", "bg"] {
        let pid = fs::read_to_string(dir.join(f))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        wait_for(
            &format!("the child's {f} to die with its supervisor"),
            || !alive(pid),
        );
    }
}

/// On a terminal there is no process group to kill, so a ready-line write error must end the
/// direct child itself before `run` returns.
#[cfg(target_os = "linux")]
#[test]
fn a_log_write_error_at_ready_kills_a_child_on_a_terminal() {
    let dir = temp("logfail-tty");
    let (fifo, head) = one_line_log(&dir);
    let line = format!(
        "'{}' supervise -- sh -c '{}' 2>{}",
        env!("CARGO_BIN_EXE_larch"),
        ready_after_restart(&dir, false),
        fifo.display()
    );
    // script(1) gives the supervisor a pseudo-terminal for stdin and stdout, so no process group.
    let mut script = Command::new("script")
        .args(["-qec", &line, "/dev/null"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    fail_the_ready_line(head, &dir);
    assert!(
        !script.wait().unwrap().success(),
        "run returned the write error"
    );
    let pid = fs::read_to_string(dir.join("pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    wait_for("the child to die with its supervisor", || !alive(pid));
}

/// A log whose ready line takes a second to write.
struct SlowReady(Vec<u8>);
impl std::io::Write for SlowReady {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        if b.windows(11).any(|w| w == b"recovery_ms") {
            thread::sleep(Duration::from_secs(1));
        }
        self.0.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// `exited_ns` is when the child exited, not when the supervisor got through its log writes.
#[test]
fn a_slow_ready_log_does_not_delay_the_exit_time() {
    let dir = temp("slowlog");
    let mut log = SlowReady(Vec::new());
    // Run 1 prints Ready and exits 3 at once, while its ready line is still being written.
    run(&fake("seq", &dir, &["3,3,0", "0"]), &mut log).unwrap();
    let (lines, _) = split(log.0);
    let exited = lines[1]["exited_ns"].as_u64().unwrap() as u128;
    let exit_1: u128 = fs::read_to_string(dir.join("exit-1"))
        .unwrap()
        .parse()
        .unwrap();
    let late = Duration::from_nanos((exited - exit_1) as u64);
    assert!(late < Duration::from_millis(300), "exited_ns {late:?} late");
}
