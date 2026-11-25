use super::*;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
/// The installer window's folder in the temporary directory; the window deletes it when it quits.
const WINDOW_DIR: &str = "larch-installer-";
/// How long the window has to fail before the command reports it open. A window that cannot
/// start (no display, no graphics driver) exits within this; raise it if slow machines miss one.
const START: Duration = Duration::from_secs(3);
/// The lines of the window's log a failure shows.
const TAIL: usize = 5;
/// Opens the installer window from this release's signed setup package, so the one-line install
/// shows the same window as the download: the archive's hash and inventory are checked as for
/// an install, and nothing is installed until the window says so.
pub(super) fn open(args: &Args, channel: &str, roots: &Roots) -> Result<()> {
    let machine = detect::detect_machine()?;
    let mut client = Client::from_env()?;
    let release = client.release(channel, None)?;
    if args.bootstrap {
        client.bootstrap(&release.head, &machine.target, &env::current_exe()?)?;
    }
    let artifact = release
        .manifest
        .components
        .get("setup")
        .and_then(|m| m.get(&format!("{}-cpu", machine.target)))
        .ok_or_else(|| Error::Refused("release has no setup archive for this target".into()))?;
    let exe = executable(&artifact.files, machine.os)?;
    let dir = env::temp_dir().join(format!("{WINDOW_DIR}{}", id()));
    fs::create_dir(&dir)?;
    let opened = (|| -> Result<()> {
        let archive = dir.join("setup.tar.zst");
        client.download(
            artifact,
            &archive,
            progress(args.json, "setup", artifact.size),
        )?;
        let dest = dir.join("setup");
        unpack::unpack(&archive, artifact, &dest)?;
        fs::remove_file(&archive)?;
        // Kept in the folder, which the window deletes when it quits.
        let log = dir.join("window.log");
        let mut command = Command::new(dest.join(exe));
        command
            .env("LARCH_SETUP_CHANNEL", channel)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(fs::File::create(&log)?);
        roots.isolate(&mut command);
        // Its own process group, so closing the terminal that ran the install command leaves the
        // window open.
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        let mut child = command.spawn().map_err(|e| {
            Error::Refused(format!(
                "cannot open the installer window: {e}; run the install command with --terminal"
            ))
        })?;
        let start = Instant::now();
        while start.elapsed() < START {
            if let Some(status) = child.try_wait()? {
                if status.success() {
                    break;
                }
                let text = String::from_utf8_lossy(&fs::read(&log)?).into_owned();
                let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
                let tail = lines[lines.len().saturating_sub(TAIL)..].join("\n");
                return Err(Error::Refused(format!(
                    "the installer window did not start ({status}){}{tail}\nrun the install command \
                     with --terminal",
                    if tail.is_empty() { "" } else { ":\n" }
                )));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Ok(())
    })();
    if opened.is_err() {
        // The first error is the one to report; a folder left behind is only temporary files.
        let _ = fs::remove_dir_all(&dir);
    }
    opened?;
    eprintln!("Larch Setup is open in its own window.");
    Ok(())
}
/// The window's program in the setup package: `larch-installer` beside larch-setup, or the one
/// program in the macOS bundle.
fn executable(files: &[(String, String, u64)], os: Os) -> Result<&str> {
    let mut found = files.iter().map(|f| f.0.as_str()).filter(|p| match os {
        Os::Macos => p
            .strip_prefix("Larch Setup.app/Contents/MacOS/")
            .is_some_and(|name| !name.is_empty() && !name.contains('/')),
        Os::Windows => *p == "larch-installer.exe",
        Os::Linux => *p == "larch-installer",
    });
    match (found.next(), found.next()) {
        (Some(p), None) => Ok(p),
        _ => Err(Error::Refused(
            "the setup package has no installer window for this computer".into(),
        )),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn files(paths: &[&str]) -> Vec<(String, String, u64)> {
        paths
            .iter()
            .map(|p| ((*p).to_owned(), "0".repeat(64), 1))
            .collect()
    }
    /// Each system finds its window, and a package without exactly one is refused.
    #[test]
    fn the_window_is_found_in_each_systems_package() {
        let linux = files(&["larch-setup", "larch-installer"]);
        assert_eq!(executable(&linux, Os::Linux).unwrap(), "larch-installer");
        let windows = files(&["larch-setup.exe", "larch-installer.exe"]);
        assert_eq!(
            executable(&windows, Os::Windows).unwrap(),
            "larch-installer.exe"
        );
        let mac = files(&[
            "larch-setup",
            "Larch Setup.app/Contents/Info.plist",
            "Larch Setup.app/Contents/MacOS/larch-installer",
            "Larch Setup.app/Contents/Resources/icon.icns",
        ]);
        assert_eq!(
            executable(&mac, Os::Macos).unwrap(),
            "Larch Setup.app/Contents/MacOS/larch-installer"
        );
        assert!(executable(&files(&["larch-setup"]), Os::Linux).is_err());
        assert!(executable(&linux, Os::Macos).is_err());
        let two = files(&[
            "Larch Setup.app/Contents/MacOS/a",
            "Larch Setup.app/Contents/MacOS/b",
        ]);
        assert!(executable(&two, Os::Macos).is_err());
    }
}
