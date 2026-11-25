//! Updates go through the installed larch-setup: the app asks it what is new, shows the answer and
//! hands over to it on a click. The app downloads and installs nothing itself, so the release key,
//! the transaction log and the rollback stay larch-setup's alone.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use serde::Deserialize;

/// How often an open app asks again.
pub const EVERY: Duration = Duration::from_secs(24 * 60 * 60);

/// What `larch-setup update --check --json` prints: a newer release, if there is one.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct Found {
    /// The newer version; None when this one is the newest.
    pub version: Option<String>,
    /// Compressed bytes the update downloads.
    pub download_size: u64,
    /// The installed parts the update replaces.
    pub components: Vec<String>,
}

/// The installed copy of Larch this app belongs to.
#[derive(Clone, Debug, PartialEq)]
pub struct Install {
    /// Its larch-setup program.
    pub setup: PathBuf,
    /// Its release channel, as setup recorded it.
    pub channel: String,
    /// Its build: cuda, hip, metal or cpu.
    pub flavour: String,
    /// The version it is pinned to, which turns update offers off.
    pub pinned: Option<String>,
}

impl Install {
    /// The installation under `root`, read from setup's receipt; None when setup did not install
    /// this app (a build from source) or the receipt does not read.
    pub fn read(root: &Path) -> Option<Install> {
        let receipt: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("receipt.json")).ok()?).ok()?;
        let setup = root.join("setup").join(if cfg!(windows) {
            "larch-setup.exe"
        } else {
            "larch-setup"
        });
        let text = |k: &str| receipt.get(k).and_then(|v| v.as_str()).map(str::to_owned);
        Some(Install {
            setup: setup.is_file().then_some(setup)?,
            channel: text("channel")?,
            flavour: receipt
                .pointer("/detect/flavour")
                .and_then(|v| v.as_str())
                .unwrap_or("cpu")
                .to_owned(),
            pinned: text("pinned"),
        })
    }

    /// Asks setup whether a newer release is out. It checks the signed channel head and manifest
    /// and changes nothing; the request carries no identifiers.
    pub fn check(&self) -> Result<Found, String> {
        let out = quiet(Command::new(&self.setup).args(["update", "--check", "--json"]))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| format!("Could not run Larch Setup: {e}"))?;
        if !out.status.success() {
            // Under --json the last line is setup's own `error` event.
            let text = String::from_utf8_lossy(&out.stdout);
            let why = text
                .lines()
                .rev()
                .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                .find_map(|v| v["message"].as_str().map(str::to_owned))
                .unwrap_or_else(|| String::from_utf8_lossy(&out.stderr).trim().to_owned());
            return Err(format!("Could not check for updates: {why}"));
        }
        parse(&String::from_utf8_lossy(&out.stdout))
    }

    /// Starts setup's update, which waits for this app to quit, installs, and opens the app again
    /// (or opens it with the failure when the update rolled back). The caller then quits.
    pub fn update(&self) -> Result<(), String> {
        // Nothing reads its output once this app quits, so none is kept: a closed pipe would stop
        // it mid-update.
        quiet(Command::new(&self.setup).args(["update", "--relaunch-app", "--yes"]))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map(drop)
            .map_err(|e| format!("Could not start the update: {e}"))
    }

    /// Opens the Larch Setup window, to add or remove parts, repair or uninstall.
    pub fn open_setup(&self) -> Result<(), String> {
        let dir = self.setup.parent().unwrap_or(Path::new("."));
        let mut c = if cfg!(target_os = "macos") {
            let mut c = Command::new("open");
            c.arg(dir.join("Larch Setup.app"));
            c
        } else {
            Command::new(dir.join(if cfg!(windows) {
                "larch-installer.exe"
            } else {
                "larch-installer"
            }))
        };
        quiet(&mut c)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map(drop)
            .map_err(|e| format!("Could not open Larch Setup: {e}"))
    }
}

/// No console window on Windows and no terminal input.
fn quiet(c: &mut Command) -> &mut Command {
    larch_agent::tool::no_window(c).stdin(Stdio::null())
}

/// The last line `update --check --json` printed: its result.
fn parse(stdout: &str) -> Result<Found, String> {
    let line = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("");
    serde_json::from_str(line).map_err(|e| format!("Larch Setup printed {line:?}: {e}"))
}

/// The newer version worth offering: none while pinned, or when setup found nothing newer.
pub fn offer(install: &Install, found: &Found) -> Option<String> {
    found
        .version
        .clone()
        .filter(|_| install.pinned.is_none() && !found.components.is_empty())
}

/// A download's size as the installer window says it: "157 MB", "1.4 GB".
pub fn size(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.1} GB", b as f64 / 1e9)
    } else {
        format!("{} MB", (b as f64 / 1e6).round().max((b > 0) as u8 as f64))
    }
}

/// The release notes of a version.
pub fn notes(version: &str) -> String {
    format!("https://github.com/Arkaya-Labs/Larch/releases/tag/v{version}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn install(dir: &Path) -> Install {
        Install::read(dir).unwrap()
    }

    /// A receipt with setup beside it reads, a pin turns offers off, and setup's answer parses.
    #[test]
    fn an_installed_copy_reads_and_a_pin_turns_offers_off() {
        let d = std::env::temp_dir().join(format!("larch-update-{}", std::process::id()));
        let setup = d.join("setup");
        std::fs::create_dir_all(&setup).unwrap();
        std::fs::write(
            d.join("receipt.json"),
            r#"{"channel":"stable","pinned":null,"detect":{"flavour":"cuda"}}"#,
        )
        .unwrap();
        // No setup program: a receipt alone is not an install the app can update.
        assert_eq!(Install::read(&d), None);
        let exe = setup.join(if cfg!(windows) {
            "larch-setup.exe"
        } else {
            "larch-setup"
        });
        std::fs::write(&exe, b"").unwrap();
        let i = install(&d);
        assert_eq!(
            (i.channel.as_str(), i.flavour.as_str(), i.pinned.as_deref()),
            ("stable", "cuda", None)
        );
        let found = parse("{\"event\":\"x\"}\n{\"version\":\"0.2.0\",\"download_size\":157000000,\"components\":[\"app\",\"cli\"]}\n").unwrap();
        assert_eq!(offer(&i, &found).as_deref(), Some("0.2.0"));
        let none = parse(r#"{"version":null,"download_size":0,"components":[]}"#).unwrap();
        assert_eq!(offer(&i, &none), None);
        let pinned = Install {
            pinned: Some("0.1.0".into()),
            ..i
        };
        assert_eq!(offer(&pinned, &found), None);
        assert!(parse("not json").is_err());
        assert_eq!(
            (size(found.download_size), size(1_400_000_000), size(0)),
            ("157 MB".into(), "1.4 GB".into(), "0 MB".into())
        );
        std::fs::remove_dir_all(&d).unwrap();
    }
}
