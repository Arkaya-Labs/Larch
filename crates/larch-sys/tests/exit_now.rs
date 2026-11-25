//! `exit_now` skips the teardown that `std::process::exit` runs (#73). The child arms a hook that
//! writes a marker and then blocks, as a wedged GPU runtime teardown would: a C `atexit` handler on
//! Unix, a process-detach TLS callback on Windows (std's exit is `ExitProcess`, which runs DLL detach
//! but not the CRT's `atexit` table).
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};
use std::{env, fs, thread};

const CHILD: &str = "LARCH_EXIT_NOW_CHILD";
const MARK: &str = "LARCH_EXIT_NOW_MARK";

fn block() {
    let _ = fs::write(env::var_os(MARK).unwrap(), "ran");
    thread::sleep(Duration::from_secs(60));
}

#[cfg(unix)]
fn arm() {
    unsafe extern "C" {
        fn atexit(handler: extern "C" fn()) -> std::ffi::c_int;
    }
    extern "C" fn handler() {
        block();
    }
    // SAFETY: the handler is a plain `extern "C" fn` that lives for the whole program.
    assert_eq!(unsafe { atexit(handler) }, 0);
}

#[cfg(windows)]
static ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The loader calls every `.CRT$XL*` entry on process detach, as it calls a DLL's entry point.
#[cfg(windows)]
#[used]
#[unsafe(link_section = ".CRT$XLM")]
static ON_DETACH: unsafe extern "system" fn(*mut std::ffi::c_void, u32, *mut std::ffi::c_void) =
    on_detach;

#[cfg(windows)]
unsafe extern "system" fn on_detach(
    _: *mut std::ffi::c_void,
    reason: u32,
    _: *mut std::ffi::c_void,
) {
    const DLL_PROCESS_DETACH: u32 = 0;
    // Armed only in the child, so the test process itself exits normally.
    if reason == DLL_PROCESS_DETACH && ARMED.load(std::sync::atomic::Ordering::Relaxed) {
        block();
    }
}

#[cfg(windows)]
fn arm() {
    ARMED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// The child: arms the hook, then exits with 3 the way `CHILD` names.
#[test]
fn child() {
    let Ok(how) = env::var(CHILD) else {
        return;
    };
    arm();
    match how.as_str() {
        "std" => std::process::exit(3),
        "now" => larch_sys::exit_now(3),
        other => panic!("unknown exit {other}"),
    }
}

fn spawn(how: &str, mark: &Path) -> Child {
    Command::new(env::current_exe().unwrap())
        .args(["--exact", "child", "--nocapture", "--test-threads=1"])
        .env(CHILD, how)
        .env(MARK, mark)
        .stdout(Stdio::null())
        .spawn()
        .unwrap()
}

/// The child's status, or None if it still runs after `limit`.
fn wait(child: &mut Child, limit: Duration) -> Option<ExitStatus> {
    let end = Instant::now() + limit;
    while Instant::now() < end {
        if let Some(status) = child.try_wait().unwrap() {
            return Some(status);
        }
        thread::sleep(Duration::from_millis(10));
    }
    None
}

fn mark(name: &str) -> PathBuf {
    let p = env::temp_dir().join(format!("larch-exit-now-{name}-{}", std::process::id()));
    let _ = fs::remove_file(&p);
    p
}

/// Kills `child` (blocked in its hook for 60 s otherwise) and removes the markers, then panics.
fn fail(child: &mut Child, marks: &[&Path], msg: &str) -> ! {
    let _ = child.kill();
    let _ = child.wait();
    for m in marks {
        let _ = fs::remove_file(m);
    }
    panic!("{msg}")
}

#[test]
fn exit_now_skips_a_blocking_exit_handler() {
    // Control: std's exit runs the hook, which then blocks the exit.
    let std_mark = mark("std");
    let mut c = spawn("std", &std_mark);
    let end = Instant::now() + Duration::from_secs(20);
    while !std_mark.exists() {
        if Instant::now() >= end {
            fail(
                &mut c,
                &[&std_mark],
                "std::process::exit never ran the hook",
            );
        }
        thread::sleep(Duration::from_millis(10));
    }
    if wait(&mut c, Duration::from_millis(500)).is_some() {
        fail(&mut c, &[&std_mark], "the hook did not block std's exit");
    }
    c.kill().unwrap();
    c.wait().unwrap();

    let now_mark = mark("now");
    let mut c = spawn("now", &now_mark);
    let Some(status) = wait(&mut c, Duration::from_secs(20)) else {
        fail(
            &mut c,
            &[&std_mark, &now_mark],
            "exit_now waited on the hook",
        );
    };
    let ran = now_mark.exists();
    let _ = fs::remove_file(&std_mark);
    let _ = fs::remove_file(&now_mark);
    assert_eq!(status.code(), Some(3), "{status:?}");
    assert!(!ran, "exit_now ran the hook");
}
