//! Executable acceptance checks distinguish explicit failures from unfinished GPU initialization.
use crate::{
    command::{self, Output},
    effects::{Effects, Roots},
    log::Log,
    manifest::{self, Error, Result},
    txn::{Component, Hooks, Receipt},
};
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};
/// Self-test deadlines; tests supply short limits through the same API.
#[derive(Clone, Copy)]
pub struct Limits {
    /// CLI version deadline.
    pub version: Duration,
    /// Normal CLI/app check deadline.
    pub normal: Duration,
    /// First CUDA/HIP initialization deadline.
    pub first_gpu: Duration,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            version: Duration::from_secs(20),
            normal: Duration::from_secs(30),
            first_gpu: Duration::from_secs(180),
        }
    }
}
/// A timeout commits with a visible warning; it is never reported as a passed check.
#[derive(Debug, PartialEq, Eq)]
pub enum Verification {
    /// The requested contract was observed.
    Verified,
    /// The tool did not finish; repair or a CPU install can retry it.
    Unverified(String),
}
/// The read-only CLI JSON contract supplied by the facade lane (`larch info --json`).
#[derive(Deserialize)]
pub struct Info {
    /// The backend that opened a device: the GPU's, else `cpu`.
    pub backend: String,
    /// Enumerated devices, the CPU reference among them, with an explicit usability flag.
    #[serde(default)]
    pub devices: Vec<Device>,
    /// The GPU runtime libraries the process has loaded.
    #[serde(default)]
    pub loaded_libraries: Vec<Library>,
}
/// One enumerated device's acceptance state.
#[derive(Deserialize)]
pub struct Device {
    /// Whether the device opened.
    pub usable: bool,
    /// Its backend: `cpu` for the reference, which is always usable.
    #[serde(default)]
    pub backend: String,
}
/// One loaded runtime library.
#[derive(Deserialize)]
pub struct Library {
    /// Where the loader found it.
    pub path: PathBuf,
}
fn invocation(
    exe: &Path,
    args: &[&str],
    txn: &str,
    roots: &Roots,
    timeout: Duration,
) -> Result<Output> {
    let mut command = Command::new(exe);
    command
        .args(args)
        .env("LARCH_SETUP_SELFTEST", txn)
        .env_remove("LARCH_RELEASES_TOKEN");
    roots.isolate(&mut command);
    command::run(&mut command, timeout)
}
fn output(result: Output, what: &str) -> Result<Option<String>> {
    match result {
        Output::Finished {
            success: true,
            stdout,
            ..
        } => Ok(Some(stdout)),
        Output::TimedOut => Ok(None),
        Output::Missing => Err(Error::Refused(format!("{what} executable is missing"))),
        Output::Finished { stderr, .. } => Err(Error::Refused(format!(
            "{what} failed: {}",
            stderr.trim().replace(['\r', '\n'], " ")
        ))),
    }
}
fn unfinished(name: &str) -> Verification {
    Verification::Unverified(format!(
        "{name} self-test timed out; the GPU check did not finish; run repair or install the cpu flavour"
    ))
}
/// Runs a component's self-test without waiting on setup's own exclusive lock.
pub fn component(
    name: &str,
    c: &Component,
    root: &Path,
    txn: &str,
    roots: &Roots,
    first_gpu: bool,
    limits: Limits,
) -> Result<Verification> {
    let gpu = matches!(c.flavour.as_str(), "cuda" | "hip");
    let timeout = if first_gpu && gpu {
        limits.first_gpu
    } else {
        limits.normal
    };
    let exe = |name: &str| {
        c.dest.join(if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.into()
        })
    };
    if name == "app" {
        let app = if cfg!(target_os = "macos") {
            c.dest.join("Contents/MacOS/larch-desktop")
        } else {
            exe("larch-desktop")
        };
        return Ok(
            if output(
                invocation(&app, &["--self-test"], txn, roots, timeout)?,
                "desktop self-test",
            )?
            .is_some()
            {
                Verification::Verified
            } else {
                unfinished(name)
            },
        );
    }
    if name != "cli" {
        return Err(Error::Refused(format!(
            "no self-test is defined for component {name}"
        )));
    }
    let Some(version) = output(
        invocation(&exe("larch"), &["--version"], txn, roots, limits.version)?,
        "CLI version check",
    )?
    else {
        return Ok(unfinished(name));
    };
    let words: Vec<_> = version.split_whitespace().collect();
    if words.len() < 2
        || words[0] != "larch"
        || manifest::compare_versions(words[1], &c.version)? != std::cmp::Ordering::Equal
    {
        return Err(Error::Refused(format!(
            "CLI version mismatch: expected {}",
            c.version
        )));
    }
    let Some(json) = output(
        invocation(&exe("larch"), &["info", "--json"], txn, roots, timeout)?,
        "CLI device check",
    )?
    else {
        return Ok(unfinished(name));
    };
    let info: Info = serde_json::from_str(&json).map_err(|e| {
        Error::Refused(format!(
            "CLI info --json does not satisfy the self-test contract: {e}"
        ))
    })?;
    // First, so a GPU build that opened none (it reports backend `cpu`) gets this message. The CPU
    // reference is always usable, so only a device of the flavour's own backend counts.
    let opened = |d: &Device| d.usable && d.backend == c.flavour;
    if gpu && !info.devices.iter().any(opened) {
        return Err(Error::Refused("no usable device; the GPU install was refused; install the cpu flavour as a new transaction".into()));
    }
    if info.backend.to_ascii_lowercase() != c.flavour {
        return Err(Error::Refused(format!(
            "CLI backend mismatch: expected {}",
            c.flavour
        )));
    }
    if gpu {
        // ponytail: only Linux reports loaded libraries (info.rs reads /proc/self/maps); require
        // them on Windows once larch-sys names loaded modules with GetModuleFileNameW (E5). Its
        // loader searches the exe's folder, where setup links the runtime, first.
        if cfg!(target_os = "linux") && info.loaded_libraries.is_empty() {
            return Err(Error::Refused(
                "GPU check reported no loaded runtime libraries".into(),
            ));
        }
        let root = fs::canonicalize(root)?;
        for Library { path: library } in info.loaded_libraries {
            if !library.is_absolute() || !fs::canonicalize(&library)?.starts_with(&root) {
                return Err(Error::Refused(format!(
                    "runtime library resolved outside the install root: {}",
                    library.display()
                )));
            }
        }
    }
    Ok(Verification::Verified)
}
/// The real transaction hooks combine OS integrations and executable checks.
pub struct SystemHooks {
    /// OS integrations and warnings, available to the command-line coordinator.
    pub effects: Effects,
    receipt: Receipt,
    previous: Option<Receipt>,
    txn: String,
    roots: Roots,
    log: Log,
    limits: Limits,
    unverified: Vec<String>,
    failed_gpu: bool,
}
impl SystemHooks {
    /// Binds checks to the planned receipt and exact journal identity.
    pub fn new(
        receipt: Receipt,
        previous: Option<Receipt>,
        txn: String,
        roots: Roots,
        effects: Effects,
        limits: Limits,
    ) -> Self {
        let log = Log::new(&receipt.root);
        Self {
            effects,
            receipt,
            previous,
            txn,
            roots,
            log,
            limits,
            unverified: Vec::new(),
            failed_gpu: false,
        }
    }
    /// Returns unfinished checks without describing them as successful verification.
    pub fn unverified(&self) -> &[String] {
        &self.unverified
    }
    /// Whether an explicit CUDA/HIP acceptance failure warrants a new CPU transaction.
    pub fn failed_gpu(&self) -> bool {
        self.failed_gpu
    }
    /// Records and prints deferred integration warnings after transaction recovery or commit.
    pub fn report_warnings(&mut self) -> Result<()> {
        for warning in self.effects.take_messages() {
            self.log.line(&warning)?;
            eprintln!("{warning}");
        }
        for warning in &self.unverified {
            eprintln!("{warning}");
        }
        Ok(())
    }
}
impl Hooks for SystemHooks {
    fn apply(&mut self, kind: &str, value: &str) -> Result<()> {
        self.effects.apply(kind, value)
    }
    fn remove(&mut self, kind: &str, value: &str) -> Result<()> {
        self.effects.remove(kind, value)
    }
    fn present(&mut self, kind: &str, value: &str) -> Result<bool> {
        self.effects.present(kind, value)
    }
    fn snapshot(&mut self, kind: &str, value: &str) -> Result<Option<String>> {
        self.effects.snapshot(kind, value)
    }
    fn restore(&mut self, kind: &str, value: &str, state: &str) -> Result<()> {
        self.effects.restore(kind, value, state)
    }
    fn self_test(&mut self, name: &str) -> Result<()> {
        let c = self.receipt.components.get(name).ok_or_else(|| {
            Error::Refused(format!(
                "self-test component {name} is absent from the target receipt"
            ))
        })?;
        let first = self
            .previous
            .as_ref()
            .and_then(|r| r.components.get(name))
            .is_none_or(|old| {
                old.version != c.version || old.flavour != c.flavour || old.files != c.files
            });
        let result = component(
            name,
            c,
            &self.receipt.root,
            &self.txn,
            &self.roots,
            first,
            self.limits,
        );
        if result.is_err() {
            self.failed_gpu = matches!(c.flavour.as_str(), "cuda" | "hip");
        }
        match result? {
            Verification::Verified => self
                .log
                .line(&format!("{name} {} self-test verified", c.version))?,
            Verification::Unverified(warning) => {
                self.log.line(&format!(
                    "Unverified: {warning}; state: install may commit; next: repair or install cpu"
                ))?;
                self.unverified.push(warning);
            }
        }
        Ok(())
    }
}
