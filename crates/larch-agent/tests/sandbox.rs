//! The sandbox's gates: no network, no host UNIX socket, no write outside the worktree (for a
//! read-only run, outside its build folders), no ownership or xattr change, the process cap, the
//! timeout's kill, a clean environment and the output cap; on Windows also the toolchains of
//! decision 19's matrix, network off. Each test prints why and passes when this host cannot
//! sandbox (no unshare, no Landlock, root, or no AppContainer), unless LARCH_SANDBOX_REQUIRED=1
//! makes that a failure.
#![cfg(any(target_os = "linux", target_os = "macos", windows))]

use std::{env, path::Path};
#[cfg(unix)]
use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    os::unix::net::UnixListener,
    process::Command,
    sync::{PoisonError, RwLock, RwLockReadGuard},
    thread,
    time::{Duration, Instant},
};

use larch_agent::sandbox::{Limits, Output, SandboxError, run};

/// The fork bomb runs alone (`write`); every other test that starts a process holds `shared`.
/// On macOS the process cap counts every process of our uid (`sandbox::macos`), so another
/// test's process that ended during the bomb's run gave it its place.
#[cfg(unix)]
static TURN: RwLock<()> = RwLock::new(());

/// A turn beside the other tests but not the fork bomb.
#[cfg(unix)]
fn shared() -> RwLockReadGuard<'static, ()> {
    TURN.read().unwrap_or_else(PoisonError::into_inner)
}

/// `cmd`'s result in the sandbox, or None, with the reason printed, when this host cannot sandbox.
fn sandboxed(cmd: &str, worktree: &Path, limits: &Limits) -> Option<Output> {
    match run(cmd, worktree, limits) {
        Err(SandboxError::Unavailable(why)) => {
            assert!(
                env::var_os("LARCH_SANDBOX_REQUIRED").is_none_or(|v| v != "1"),
                "LARCH_SANDBOX_REQUIRED=1 but the sandbox is unavailable: {why}"
            );
            println!("skipped: {why}");
            None
        }
        result => Some(result.expect("the sandbox runs the command")),
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[cfg(unix)]
#[test]
fn a_tcp_connect_fails() {
    let _turn = shared();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let served = thread::spawn(move || {
        let mut accepted = 0;
        for conn in listener.incoming() {
            let mut conn = conn.unwrap();
            let mut request = [0u8; 1024];
            let n = conn.read(&mut request).unwrap();
            if request[..n].starts_with(b"STOP") {
                return accepted;
            }
            accepted += 1;
            let _ = conn.write_all(b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\nok");
        }
        accepted
    });
    let curl = format!("curl -sS --max-time 5 http://127.0.0.1:{port}/");
    // The control: the same command outside the sandbox reaches the listener.
    match Command::new("sh").args(["-c", &curl]).output() {
        Ok(out) if out.status.success() => {}
        out => {
            println!("skipped: curl cannot reach a local listener outside the sandbox: {out:?}");
            return;
        }
    }
    let worktree = tempfile::tempdir().unwrap();
    let Some(out) = sandboxed(&curl, worktree.path(), &Limits::default()) else {
        return;
    };
    std::net::TcpStream::connect(("127.0.0.1", port))
        .unwrap()
        .write_all(b"STOP")
        .unwrap();
    assert_ne!(out.exit, Some(0), "curl connected: {}", text(&out.stderr));
    assert_eq!(served.join().unwrap(), 1, "only the control connected");
}

#[cfg(unix)]
#[test]
fn a_unix_socket_outside_cannot_be_reached() {
    let _turn = shared();
    let outside = tempfile::tempdir().unwrap();
    let path = outside.path().join("host.sock");
    let listener = UnixListener::bind(&path).unwrap();
    listener.set_nonblocking(true).unwrap();
    let curl = format!(
        "curl -sS --max-time 1 --unix-socket '{}' http://x/",
        path.display()
    );
    // The control: outside the sandbox curl connects (and times out on the silent listener).
    let _ = Command::new("sh").args(["-c", &curl]).output();
    if listener.accept().is_err() {
        println!("skipped: curl cannot reach a UNIX socket outside the sandbox");
        return;
    }
    let worktree = tempfile::tempdir().unwrap();
    let Some(out) = sandboxed(&curl, worktree.path(), &Limits::default()) else {
        return;
    };
    assert_ne!(out.exit, Some(0), "{}", text(&out.stderr));
    assert!(
        listener.accept().is_err(),
        "the sandbox reached the host socket"
    );
}

/// A command cannot rename its own worktree from the parent: the parent is not a write root.
#[cfg(unix)]
#[test]
fn the_worktree_cannot_be_renamed_from_its_parent() {
    let _turn = shared();
    let parent = tempfile::tempdir().unwrap();
    let worktree = parent.path().join("0");
    fs::create_dir(&worktree).unwrap();
    let Some(out) = sandboxed("cd .. && mv 0 x", &worktree, &Limits::default()) else {
        return;
    };
    assert_ne!(out.exit, Some(0), "mv succeeded");
    assert!(worktree.is_dir() && !parent.path().join("x").exists());
}

#[cfg(target_os = "linux")]
#[test]
fn a_datagram_socket_outside_cannot_be_reached() {
    let _turn = shared();
    let outside = tempfile::tempdir().unwrap();
    let path = outside.path().join("host.dgram");
    let host = std::os::unix::net::UnixDatagram::bind(&path).unwrap();
    host.set_nonblocking(true).unwrap();
    // A datagram socketpair may sendto any pathname socket.
    let send = format!(
        "python3 -c \"import socket; a, _ = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM); \
         a.sendto(b'escape', '{}')\"",
        path.display()
    );
    let control = Command::new("sh").args(["-c", &send]).output();
    if host.recv(&mut [0u8; 16]).is_err() {
        println!("skipped: the send fails outside the sandbox too: {control:?}");
        return;
    }
    let worktree = tempfile::tempdir().unwrap();
    let Some(out) = sandboxed(&send, worktree.path(), &Limits::default()) else {
        return;
    };
    assert_ne!(out.exit, Some(0), "{}", text(&out.stderr));
    assert!(
        host.recv(&mut [0u8; 16]).is_err(),
        "the sandbox reached the host socket"
    );
}

#[cfg(unix)]
#[test]
fn a_write_outside_the_worktree_fails_and_inside_works() {
    let _turn = shared();
    let (worktree, outside) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let escaped = outside.path().join("escaped");
    let cmd = format!(
        "echo in > made; echo out > '{}'; cat made",
        escaped.display()
    );
    let Some(out) = sandboxed(&cmd, worktree.path(), &Limits::default()) else {
        return;
    };
    assert!(!escaped.exists(), "wrote outside the worktree");
    assert_eq!(
        fs::read_to_string(worktree.path().join("made")).unwrap(),
        "in\n"
    );
    assert_eq!(text(&out.stdout), "in\n", "cwd is the worktree");
    assert!(!out.truncated && !out.timed_out);
}

/// A read-only run (`Limits::writable`) writes its build folders and its tmp, and nothing else
/// of the worktree: no new file, no change, no delete, no new folder, no rename of a build
/// folder, no write through a build folder that links outside or to the project's own folders
/// (its sources, its .git), and never to .git itself.
#[cfg(unix)]
#[test]
fn a_read_only_run_writes_only_its_build_folders() {
    let _turn = shared();
    let (worktree, outside) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let w = worktree.path();
    fs::write(w.join("src.txt"), "mine\n").unwrap();
    fs::create_dir(w.join("target")).unwrap();
    std::os::unix::fs::symlink(outside.path(), w.join("node_modules")).unwrap();
    // Links to the project's own folders are no build folder either.
    fs::create_dir_all(w.join("src")).unwrap();
    fs::write(w.join("src/main.py"), "mine\n").unwrap();
    std::os::unix::fs::symlink("src", w.join("dist")).unwrap();
    fs::create_dir_all(w.join(".git")).unwrap();
    fs::write(w.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
    std::os::unix::fs::symlink(".git", w.join(".venv")).unwrap();
    let limits = Limits {
        writable: Some(
            ["target", "build", "node_modules", "dist", ".venv", ".git"]
                .map(Into::into)
                .into(),
        ),
        ..Limits::default()
    };
    // Each line prints its name when the write went through.
    let cmd = "echo built > target/out && echo target\n\
               echo t > \"$TMPDIR/t\" && echo tmp\n\
               echo x > new.txt 2>/dev/null && echo new\n\
               echo x >> src.txt 2>/dev/null && echo append\n\
               rm -f src.txt 2>/dev/null; [ -e src.txt ] || echo rm\n\
               mkdir build 2>/dev/null && echo mkdir\n\
               mv target moved 2>/dev/null && echo mv\n\
               echo x > node_modules/x 2>/dev/null && echo link\n\
               echo x > dist/main.py 2>/dev/null && echo src\n\
               echo x > .venv/HEAD 2>/dev/null && echo git\n\
               echo x > .git/HEAD 2>/dev/null && echo dotgit\n";
    let Some(out) = sandboxed(cmd, w, &limits) else {
        return;
    };
    assert_eq!(text(&out.stdout), "target\ntmp\n", "{}", text(&out.stderr));
    assert_eq!(fs::read_to_string(w.join("src.txt")).unwrap(), "mine\n");
    assert_eq!(fs::read_to_string(w.join("target/out")).unwrap(), "built\n");
    assert!(!w.join("new.txt").exists() && !w.join("build").exists());
    assert!(!outside.path().join("x").exists(), "wrote through the link");
    assert_eq!(fs::read_to_string(w.join("src/main.py")).unwrap(), "mine\n");
    assert_eq!(
        fs::read_to_string(w.join(".git/HEAD")).unwrap(),
        "ref: refs/heads/main\n"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn ownership_and_xattr_changes_fail() {
    let _turn = shared();
    let worktree = tempfile::tempdir().unwrap();
    fs::write(worktree.path().join("f"), "").unwrap();
    // Both are no-ops that still make the syscall, so the control proves only that the call
    // itself works here.
    for call in [
        "os.chown('f', -1, -1)",
        "os.setxattr('f', 'user.larch', b'1')",
    ] {
        let cmd = format!("python3 -c \"import os; {call}\"");
        let control = Command::new("sh")
            .args(["-c", &cmd])
            .current_dir(worktree.path())
            .output();
        if !control.as_ref().is_ok_and(|o| o.status.success()) {
            println!("skipped {call}: it fails outside the sandbox too: {control:?}");
            continue;
        }
        let Some(out) = sandboxed(&cmd, worktree.path(), &Limits::default()) else {
            return;
        };
        assert_ne!(out.exit, Some(0), "{call} ran in the sandbox");
        assert!(
            text(&out.stderr).contains("PermissionError"),
            "{call}: {}",
            text(&out.stderr)
        );
    }
}

#[cfg(unix)]
#[test]
fn a_fork_bomb_hits_the_process_cap() {
    let _alone = TURN.write().unwrap_or_else(PoisonError::into_inner);
    let worktree = tempfile::tempdir().unwrap();
    let limits = Limits {
        max_procs: 16,
        timeout: Duration::from_secs(20),
        ..Limits::default()
    };
    // Bounded at 100, so a broken cap cannot take the test host down with it.
    let bomb = "i=0; while [ $i -lt 100 ]; do sleep 30 & echo $!; i=$((i+1)); done";
    let Some(out) = sandboxed(bomb, worktree.path(), &limits) else {
        return;
    };
    let started = text(&out.stdout).lines().count();
    assert!(
        started < limits.max_procs as usize,
        "{started} processes started under a cap of {}; stderr: {}",
        limits.max_procs,
        text(&out.stderr)
    );
    assert!(!out.timed_out, "the shell stops at the cap");
}

/// A Stop of the run (`Limits::stop`) ends a running command within a poll, long before its
/// timeout, and reads as stopped, not timed out.
#[cfg(unix)]
#[test]
fn a_stop_ends_a_running_command_before_its_timeout() {
    let _turn = shared();
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    let worktree = tempfile::tempdir().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let limits = Limits {
        timeout: Duration::from_secs(60),
        stop: Some(stop.clone()),
        ..Limits::default()
    };
    let user = thread::spawn(move || {
        thread::sleep(Duration::from_millis(300));
        stop.store(true, Ordering::Relaxed);
    });
    let start = Instant::now();
    let Some(out) = sandboxed("sleep 30", worktree.path(), &limits) else {
        return;
    };
    user.join().unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "{:?}",
        start.elapsed()
    );
    assert!(out.stopped && !out.timed_out, "{out:?}");
    assert_eq!(out.exit, None);
}

#[cfg(unix)]
#[test]
fn a_sleep_past_the_timeout_kills_the_whole_group() {
    let _turn = shared();
    let worktree = tempfile::tempdir().unwrap();
    let limits = Limits {
        timeout: Duration::from_secs(1),
        ..Limits::default()
    };
    // Each process writes its file once it outlives the kill by 2 s: one in the group, one that
    // left it, and on Linux the shell itself, which leaves it too. macOS has no setsid(1), and
    // nothing there kills a process that left the group.
    let (cmd, orphans): (&str, &[&str]) = if cfg!(target_os = "linux") {
        (
            "(sleep 3; echo > grouped) & setsid sh -c 'sleep 3; echo > escaped' & \
             exec setsid sh -c 'sleep 3; echo > leader'",
            &["grouped", "escaped", "leader"],
        )
    } else {
        ("(sleep 3; echo > grouped) & sleep 300", &["grouped"])
    };
    let start = Instant::now();
    let Some(out) = sandboxed(cmd, worktree.path(), &limits) else {
        return;
    };
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "{:?}",
        start.elapsed()
    );
    assert!(out.timed_out);
    assert_eq!(out.exit, None, "killed by a signal");
    thread::sleep(Duration::from_secs(4));
    for orphan in orphans {
        assert!(
            !worktree.path().join(orphan).exists(),
            "the {orphan} sleep survived"
        );
    }
}

/// Set on the copy of this test binary that `the_env_holds_no_key_or_token` starts.
#[cfg(unix)]
const SECRET: &str = "LARCH_SANDBOX_TEST_KEY";

#[cfg(unix)]
#[test]
fn the_env_holds_no_key_or_token() {
    let _turn = shared();
    // Our own environment cannot be changed without unsafe, so a copy of this binary runs the
    // check with a key and a token in its environment.
    if env::var_os(SECRET).is_none() {
        let copy = Command::new(env::current_exe().unwrap())
            .args(["--exact", "the_env_holds_no_key_or_token", "--nocapture"])
            .env(SECRET, "leak")
            .env("LARCH_SANDBOX_TEST_TOKEN", "leak")
            .output()
            .unwrap();
        print!("{}", text(&copy.stdout));
        assert!(copy.status.success(), "{}", text(&copy.stderr));
        return;
    }
    let worktree = tempfile::tempdir().unwrap();
    let Some(out) = sandboxed("env", worktree.path(), &Limits::default()) else {
        return;
    };
    let names: Vec<String> = text(&out.stdout)
        .lines()
        .filter_map(|l| l.split_once('=').map(|(name, _)| name.to_owned()))
        .collect();
    assert!(names.iter().any(|n| n == "HOME"), "{names:?}");
    for name in &names {
        assert!(
            !name.ends_with("_KEY") && !name.ends_with("_TOKEN"),
            "{name} leaked"
        );
        // The shell adds PWD, SHLVL and _ itself; the toolchains' folders are there when they
        // exist.
        let allowed = [
            "PATH",
            "HOME",
            "LANG",
            "TMP",
            "TMPDIR",
            "CARGO_HOME",
            "RUSTUP_HOME",
            "CARGO_NET_OFFLINE",
            "PWD",
            "SHLVL",
            "_",
        ];
        assert!(allowed.contains(&name.as_str()), "{name} leaked: {names:?}");
    }
}

#[cfg(unix)]
#[test]
fn output_past_the_cap_is_truncated_and_flagged() {
    let _turn = shared();
    let worktree = tempfile::tempdir().unwrap();
    let limits = Limits {
        max_output_bytes: 1000,
        ..Limits::default()
    };
    let Some(out) = sandboxed("head -c 5000 /dev/zero", worktree.path(), &limits) else {
        return;
    };
    assert_eq!(out.stdout.len(), 1000);
    assert!(out.truncated);
    assert_eq!(out.exit, Some(0));
    // The end is what is kept: a test run's summary is there.
    let Some(out) = sandboxed("seq 1 2000", worktree.path(), &limits) else {
        return;
    };
    assert_eq!(out.stdout.len(), 1000);
    assert!(
        out.stdout.ends_with(b"\n1999\n2000\n"),
        "{}",
        text(&out.stdout)
    );
    assert!(out.truncated);
    let Some(out) = sandboxed("head -c 1000 /dev/zero", worktree.path(), &limits) else {
        return;
    };
    assert_eq!(out.stdout.len(), 1000);
    assert!(!out.truncated, "exactly the cap is not truncated");
}

#[cfg(windows)]
mod windows {
    use std::{
        fs,
        process::Command,
        thread,
        time::{Duration, Instant},
    };

    use super::*;

    /// `cmd` under cmd.exe outside the sandbox, in `dir`: the controls.
    fn host(cmd: &str, dir: &Path) -> std::io::Result<std::process::Output> {
        Command::new("cmd")
            .args(["/d", "/c", cmd])
            .current_dir(dir)
            .output()
    }

    #[test]
    fn a_tcp_connect_fails() {
        // Stop: a failed Connect is otherwise a non-terminating error, and the exit stays 0.
        let connect = "powershell -NoProfile -Command \"$ErrorActionPreference = 'Stop'; (New-Object Net.Sockets.TcpClient).Connect('1.1.1.1', 443); 'connected'\"";
        let worktree = tempfile::tempdir().unwrap();
        // The control: outside the sandbox this host reaches 1.1.1.1.
        match host(connect, worktree.path()) {
            Ok(out) if out.status.success() => {}
            out => {
                println!("skipped: 1.1.1.1:443 is unreachable outside the sandbox: {out:?}");
                return;
            }
        }
        let Some(out) = sandboxed(connect, worktree.path(), &Limits::default()) else {
            return;
        };
        assert_ne!(out.exit, Some(0), "connected: {}", text(&out.stdout));
        assert!(!text(&out.stdout).contains("connected"));
    }

    #[test]
    fn a_write_outside_the_worktree_or_to_the_profile_fails_and_inside_works() {
        let (worktree, outside) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let escaped = outside.path().join("escaped");
        let profile = env::home_dir().unwrap().join("larch-sandbox-escaped");
        // No space before &: echo would keep it.
        let cmd = format!(
            "echo in>made& echo out>\"{}\"& echo out>\"{}\"& type made",
            escaped.display(),
            profile.display()
        );
        let Some(out) = sandboxed(&cmd, worktree.path(), &Limits::default()) else {
            return;
        };
        assert!(!escaped.exists(), "wrote outside the worktree");
        assert!(!profile.exists(), "wrote to the user's profile folder");
        assert_eq!(
            fs::read_to_string(worktree.path().join("made")).unwrap(),
            "in\r\n"
        );
        assert_eq!(text(&out.stdout), "in\r\n", "cwd is the worktree");
        assert!(!out.truncated && !out.timed_out);
    }

    /// Set on the copy of this test binary that `the_env_holds_no_key_or_token` starts.
    const SECRET: &str = "LARCH_SANDBOX_TEST_KEY";

    #[test]
    fn the_env_holds_no_key_or_token() {
        // Our own environment cannot be changed without unsafe, so a copy of this binary runs the
        // check with a key and a token in its environment.
        if env::var_os(SECRET).is_none() {
            let copy = Command::new(env::current_exe().unwrap())
                .args([
                    "--exact",
                    "windows::the_env_holds_no_key_or_token",
                    "--nocapture",
                ])
                .env(SECRET, "leak")
                .env("LARCH_SANDBOX_TEST_TOKEN", "leak")
                .output()
                .unwrap();
            print!("{}", text(&copy.stdout));
            assert!(copy.status.success(), "{}", text(&copy.stderr));
            return;
        }
        let worktree = tempfile::tempdir().unwrap();
        let Some(out) = sandboxed("set", worktree.path(), &Limits::default()) else {
            return;
        };
        let vars: Vec<(String, String)> = text(&out.stdout)
            .lines()
            .filter_map(|l| {
                let (name, value) = l.split_once('=')?;
                Some((name.to_uppercase(), value.to_owned()))
            })
            .collect();
        let get = |name: &str| {
            vars.iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.as_str())
        };
        let profile = get("USERPROFILE").expect("USERPROFILE is set");
        assert_ne!(
            Path::new(profile),
            env::home_dir().unwrap(),
            "the user's own"
        );
        assert_eq!(get("HOME"), Some(profile));
        // The private tmp: CreateProcess points TEMP at a folder beneath it.
        let temp = get("TEMP").expect("TEMP is set");
        assert!(temp.starts_with(profile), "{temp} is not beneath {profile}");
        // cmd.exe adds COMSPEC, PATHEXT and PROMPT itself; the toolchains' folders are there
        // when they exist.
        let allowed = [
            "PATH",
            "SYSTEMROOT",
            "LANG",
            "CARGO_HOME",
            "RUSTUP_HOME",
            "CARGO_NET_OFFLINE",
            // MSVC's, for the linker.
            "LIB",
            "LIBPATH",
            "INCLUDE",
            "HOME",
            "USERPROFILE",
            "LOCALAPPDATA",
            "TEMP",
            "TMP",
            "COMSPEC",
            "PATHEXT",
            "PROMPT",
        ];
        for (name, _) in &vars {
            assert!(
                !name.ends_with("_KEY") && !name.ends_with("_TOKEN"),
                "{name} leaked"
            );
            assert!(allowed.contains(&name.as_str()), "{name} leaked: {vars:?}");
        }
    }

    /// The ids of the processes, other than this query's, whose command line holds `mark`.
    fn marked(mark: &str) -> Vec<String> {
        let query = format!(
            "Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*{mark}*' -and $_.ProcessId -ne $PID }} | ForEach-Object ProcessId"
        );
        let out = Command::new("powershell")
            .args(["-NoProfile", "-Command", &query])
            .output()
            .unwrap();
        text(&out.stdout)
            .split_whitespace()
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn a_timeout_kills_the_whole_process_tree() {
        let worktree = tempfile::tempdir().unwrap();
        let limits = Limits {
            timeout: Duration::from_secs(15),
            ..Limits::default()
        };
        // Unique to this run, so nothing else matches; a comment at each level. timeout.exe
        // refuses a stdin that is not a console, so the leaf sleeps in PowerShell.
        let mark = format!("larch-tree-{}", std::process::id());
        let cmd = format!(
            "cmd /d /c \"cmd /d /c powershell -NoProfile -Command Start-Sleep 60 #{mark}\" & rem {mark}"
        );
        let run = {
            let worktree = worktree.path().to_owned();
            thread::spawn(move || sandboxed(&cmd, &worktree, &limits))
        };
        // The shell, the nested shell and the sleep.
        let deadline = Instant::now() + Duration::from_secs(14);
        while marked(&mark).len() < 3 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(200));
        }
        let up = marked(&mark);
        let Some(out) = run.join().unwrap() else {
            return;
        };
        assert!(up.len() >= 3, "the tree never started: {up:?}");
        let killed = Instant::now();
        assert!(out.timed_out);
        assert_eq!(out.exit, None);
        let left = marked(&mark);
        let looked = killed.elapsed();
        assert!(left.is_empty(), "left running after the kill: {left:?}");
        // The query's snapshot falls within it; PowerShell's start is most of it.
        assert!(looked < Duration::from_secs(5), "{looked:?}");
    }

    #[test]
    fn output_past_the_cap_is_truncated_and_flagged() {
        let worktree = tempfile::tempdir().unwrap();
        let limits = Limits {
            max_output_bytes: 1000,
            ..Limits::default()
        };
        let count = "for /l %i in (1,1,2000) do @echo %i";
        let Some(out) = sandboxed(count, worktree.path(), &limits) else {
            return;
        };
        assert_eq!(out.stdout.len(), 1000);
        // The end is what is kept: a test run's summary is there.
        assert!(
            out.stdout.ends_with(b"\r\n1999\r\n2000\r\n"),
            "{}",
            text(&out.stdout)
        );
        assert!(out.truncated);
        assert_eq!(out.exit, Some(0));
        // 200 lines of `abc` and CRLF: exactly the cap.
        let exact = "for /l %i in (1,1,200) do @echo abc";
        let Some(out) = sandboxed(exact, worktree.path(), &limits) else {
            return;
        };
        assert_eq!(out.stdout.len(), 1000);
        assert!(!out.truncated, "exactly the cap is not truncated");
    }

    /// A repo's setup outside the sandbox; its error text when it fails.
    type Setup = fn(&Path) -> Result<(), String>;

    /// Decision 19's matrix: each toolchain's test command passes in the sandbox, network off, in
    /// a repo of its own. A tool this host lacks is skipped.
    #[test]
    fn the_toolchains_pass_their_tests_offline() {
        let matrix: [(&str, &str, &str, Setup); 4] = [
            ("cargo", "cargo --version", "cargo test", cargo_repo),
            ("npm", "npm --version", "npm test", npm_repo),
            ("pytest", "pytest --version", "pytest", pytest_repo),
            // The host alone runs nothing: an SDK is what `dotnet test` needs.
            (
                "dotnet",
                "dotnet --list-sdks | findstr .",
                "dotnet test",
                dotnet_repo,
            ),
        ];
        let mut failed = Vec::new();
        for (tool, installed, test, setup) in matrix {
            let repo = tempfile::tempdir().unwrap();
            if !host(installed, repo.path()).is_ok_and(|o| o.status.success()) {
                println!("skipped: {tool} not installed");
                continue;
            }
            if let Err(why) = setup(repo.path()) {
                failed.push(format!("{tool}'s repo: {why}"));
                continue;
            }
            let limits = Limits {
                timeout: Duration::from_secs(300),
                ..Limits::default()
            };
            let Some(out) = sandboxed(test, repo.path(), &limits) else {
                return;
            };
            if out.exit == Some(0) {
                println!("passed: {test}");
            } else {
                failed.push(format!(
                    "{test} ended with {:?}:\n{}\n{}",
                    out.exit,
                    text(&out.stdout),
                    text(&out.stderr)
                ));
            }
        }
        assert!(failed.is_empty(), "{}", failed.join("\n\n"));
    }

    /// `cmd` on the host in `dir`, or what it printed as the error.
    fn on_host(cmd: &str, dir: &Path) -> Result<(), String> {
        match host(cmd, dir) {
            Ok(out) if out.status.success() => Ok(()),
            out => Err(format!("{cmd}: {out:?}")),
        }
    }

    /// A one-function crate with a dependency, fetched on the host.
    fn cargo_repo(dir: &Path) -> Result<(), String> {
        fs::create_dir(dir.join("src")).map_err(|e| e.to_string())?;
        fs::write(
            dir.join("Cargo.toml"),
            "[package]\nname = \"one\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n\
             [dependencies]\nstrsim = \"0.11\"\n",
        )
        .map_err(|e| e.to_string())?;
        fs::write(
            dir.join("src/lib.rs"),
            "pub fn one() -> f64 {\n    strsim::jaro(\"a\", \"a\")\n}\n\n\
             #[test]\nfn it_is_one() {\n    assert_eq!(one(), 1.0);\n}\n",
        )
        .map_err(|e| e.to_string())?;
        on_host("cargo fetch", dir)
    }

    fn npm_repo(dir: &Path) -> Result<(), String> {
        let package = r#"{"name": "one", "version": "1.0.0", "scripts": {"test": "node -e \"process.exit(0)\""}}"#;
        fs::write(dir.join("package.json"), package).map_err(|e| e.to_string())
    }

    fn pytest_repo(dir: &Path) -> Result<(), String> {
        fs::write(
            dir.join("test_one.py"),
            "def test_one():\n    assert 1 == 1\n",
        )
        .map_err(|e| e.to_string())
    }

    /// An MSTest project, its packages restored on the host.
    fn dotnet_repo(dir: &Path) -> Result<(), String> {
        on_host("dotnet new mstest --name one --output .", dir)?;
        on_host("dotnet restore", dir)
    }
}
