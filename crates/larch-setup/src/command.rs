//! Bounded platform-tool execution without a shell or unsafe process manipulation.
use crate::manifest::{Error, Result};
use std::{
    io::{self, Read},
    process::{Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};
/// A missing tool and an unfinished tool are distinct from a reported failure.
#[derive(Debug)]
pub enum Output {
    /// The executable is not installed.
    Missing,
    /// The whole invocation exceeded its deadline, including output collection.
    TimedOut,
    /// The tool exited and its bounded output was collected.
    Finished {
        /// Whether the process exited successfully.
        success: bool,
        /// UTF-8 standard output.
        stdout: String,
        /// UTF-8 standard error.
        stderr: String,
    },
}
fn reader(pipe: impl Read + Send + 'static) -> mpsc::Receiver<io::Result<String>> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = pipe
            .take(8 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .and_then(|_| {
                if bytes.len() > 8 * 1024 * 1024 {
                    return Err(io::Error::other("tool output exceeds 8 MiB"));
                }
                String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            });
        let _ = tx.send(result);
    });
    rx
}
/// Runs one program with a deadline; only that child is terminated on timeout.
pub fn run(command: &mut Command, timeout: Duration) -> Result<Output> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: under the installer window this process has no console, so a console
        // tool (PowerShell, the self-tested larch) would otherwise flash one of its own.
        command.creation_flags(0x0800_0000);
    }
    let mut child = match command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Output::Missing),
        Err(e) => return Err(e.into()),
    };
    let stdout = reader(child.stdout.take().unwrap());
    let stderr = reader(child.stderr.take().unwrap());
    let deadline = Instant::now() + timeout;
    let mut status = None;
    let mut out = None;
    let mut err = None;
    loop {
        if status.is_none() {
            status = child.try_wait()?;
        }
        if out.is_none() {
            match stdout.try_recv() {
                Ok(v) => out = Some(v),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(e) => return Err(Error::Refused(format!("tool stdout reader failed: {e}"))),
            }
        }
        if err.is_none() {
            match stderr.try_recv() {
                Ok(v) => err = Some(v),
                Err(mpsc::TryRecvError::Empty) => {}
                Err(e) => return Err(Error::Refused(format!("tool stderr reader failed: {e}"))),
            }
        }
        if let (Some(status), Some(out), Some(err)) = (status, out.as_ref(), err.as_ref()) {
            return Ok(Output::Finished {
                success: status.success(),
                stdout: out
                    .as_ref()
                    .map_err(|e| io::Error::new(e.kind(), e.to_string()))?
                    .clone(),
                stderr: err
                    .as_ref()
                    .map_err(|e| io::Error::new(e.kind(), e.to_string()))?
                    .clone(),
            });
        }
        if Instant::now() >= deadline {
            if status.is_none() {
                if let Err(error) = child.kill()
                    && child.try_wait()?.is_none()
                {
                    return Err(error.into());
                }
                child.wait()?;
            }
            return Ok(Output::TimedOut);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
/// Returns successful output, treating unavailable or timed-out optional tools as absent.
pub fn text(program: &str, args: &[&str]) -> Result<Option<String>> {
    match run(Command::new(program).args(args), Duration::from_secs(10))? {
        Output::Finished {
            success: true,
            stdout,
            ..
        } => Ok(Some(stdout)),
        _ => Ok(None),
    }
}
