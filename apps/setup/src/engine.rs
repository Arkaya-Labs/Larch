//! larch-setup, run as a child of this same program: one command at a time with `--json --yes`.
//! Every line it prints is an event; the last one is the command's result, and a failure ends with
//! its `error` line. Every decision stays in larch-setup; the window only shows them.

use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use serde_json::Value;

/// Set on the child: there this program is larch-setup itself, with no window.
pub const ENGINE: &str = "LARCH_SETUP_ENGINE";

/// What a running command sends the window.
pub enum Msg {
    /// One progress line: `download`, `unpack`, `apply` or `gpu_failed`.
    Event(Value),
    /// The command's result, or why it failed.
    Done(Result<Value, Failure>),
}

/// A failed command, as the window shows it.
#[derive(Clone, Debug)]
pub struct Failure {
    /// larch-setup's own message, or what the child left on stderr.
    pub message: String,
    /// Larch is running, so setup could not change it.
    pub busy: bool,
}

impl Failure {
    fn new(message: impl ToString) -> Failure {
        Failure {
            message: message.to_string(),
            busy: false,
        }
    }
}

/// The commands running now, so quitting the window stops them: larch-setup's recovery undoes a
/// transaction cut short the next time it runs.
static RUNNING: Mutex<Vec<Arc<Mutex<Child>>>> = Mutex::new(Vec::new());

/// Whether a command is running now.
pub fn busy() -> bool {
    !RUNNING.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
}

/// Stops every running command; called as the window closes.
pub fn stop_all() {
    let running = std::mem::take(&mut *RUNNING.lock().unwrap_or_else(|e| e.into_inner()));
    for child in running {
        // Best effort: a child that already exited has nothing to stop.
        let _ = child.lock().unwrap_or_else(|e| e.into_inner()).kill();
    }
}

/// Runs `larch-setup <args> --json --yes` to its end on this thread, sending each line as it
/// arrives. `send` returns false once nobody listens, and the command is then stopped.
pub fn run(args: Vec<String>, send: impl Fn(Msg) -> bool) {
    let result = run_inner(args, &send);
    send(Msg::Done(result));
}

fn run_inner(args: Vec<String>, send: &impl Fn(Msg) -> bool) -> Result<Value, Failure> {
    let mut command = Command::new(std::env::current_exe().map_err(Failure::new)?);
    command
        .env(ENGINE, "1")
        .args(&args)
        .args(["--json", "--yes"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: the engine is a console program inside a windowed one.
        command.creation_flags(0x0800_0000);
    }
    let mut child = command.spawn().map_err(Failure::new)?;
    let stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let child = Arc::new(Mutex::new(child));
    RUNNING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(child.clone());
    let log = std::thread::spawn(move || {
        let mut s = String::new();
        stderr.read_to_string(&mut s).map(|_| s)
    });
    let read = (|| {
        let mut last = Value::Null;
        for line in BufReader::new(stdout).lines() {
            let line = line.map_err(Failure::new)?;
            last = serde_json::from_str(&line)
                .map_err(|e| Failure::new(format!("setup printed {line:?}: {e}")))?;
            if !send(Msg::Event(last.clone())) {
                return Err(Failure::new("the window closed"));
            }
        }
        Ok(last)
    })();
    let lock = || child.lock().unwrap_or_else(|e| e.into_inner());
    if read.is_err() {
        // setup must not go on installing behind a window that stopped listening.
        let _ = lock().kill();
    }
    let status = lock().wait().map_err(Failure::new);
    RUNNING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|c| !Arc::ptr_eq(c, &child));
    let last = read?;
    let status = status?;
    let log = log
        .join()
        .map_err(|_| Failure::new("the setup log reader stopped"))?
        .map_err(Failure::new)?;
    if status.success() {
        Ok(last)
    } else if last["event"] == "error" {
        Err(Failure {
            message: last["message"].as_str().unwrap_or_default().to_owned(),
            busy: last["busy"].as_bool().unwrap_or(false),
        })
    } else {
        let line = log.lines().last().map(String::from);
        Err(Failure::new(
            line.unwrap_or_else(|| format!("setup exited with {status}")),
        ))
    }
}
