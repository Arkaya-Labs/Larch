use super::*;
use std::process::{Child, Command};
/// Runs from a stable copy before a transaction replaces setup itself; on Windows returns the
/// copy for the caller to wait on once it has released the lock, unless this is the installed
/// setup, which exits so its folder can be swapped.
pub(super) fn restart(
    t: &Transaction,
    txn: &str,
    os: Os,
    install: Option<&super::Command>,
    args: &Args,
) -> Result<Option<Child>> {
    let dir = t.root().join(".run");
    t.check_path(&dir)?;
    txn::create_dir_all(&dir)?;
    let exe = dir.join(format!(
        "larch-setup-{txn}{}",
        if os == Os::Windows { ".exe" } else { "" }
    ));
    t.check_path(&exe)?;
    let current = env::current_exe()?;
    let bytes = fs::read(&current)?;
    txn::atomic_write(&exe, &bytes)?;
    fs::set_permissions(&exe, fs::metadata(&current)?.permissions())?;
    fs::File::options()
        .read(true)
        .write(true)
        .open(&exe)?
        .sync_all()?;
    let mut command = Command::new(&exe);
    if let Some(super::Command::Install {
        cli,
        app,
        flavour,
        channel,
        version,
    }) = install
    {
        command.args(["install", "--interactive-install", "--channel", channel]);
        if *cli {
            command.arg("--cli");
        }
        if *app {
            command.arg("--app");
        }
        if let Some(f) = flavour {
            command.args(["--flavour", f]);
        }
        if let Some(v) = version {
            command.args(["--version", v]);
        }
        if args.bootstrap {
            command.arg("--bootstrap");
        }
        if args.yes {
            command.arg("--yes");
        }
        if args.json {
            command.arg("--json");
        }
    } else {
        command.args(env::args_os().skip(1));
    }
    command.args(["--txn-child", "--txn-id", txn]);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        Err(command.exec().into())
    }
    #[cfg(windows)]
    {
        let child = command.spawn()?;
        let installed = fs::canonicalize(&current)?.starts_with(t.root());
        Ok((!installed).then_some(child))
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err(Error::Refused("unsupported OS".into()))
    }
}
/// Waits for the copy `restart` started and exits with its status, so callers such as
/// install.ps1 see the transaction's result.
pub(super) fn forward(child: Option<Child>) -> Result<()> {
    if let Some(mut child) = child {
        let status = child.wait()?;
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
    }
    Ok(())
}
/// Removes only older setup copies, never a directory recursively.
pub(super) fn clean_old(t: &Transaction) -> Result<()> {
    let dir = t.root().join(".run");
    t.check_path(&dir.join("scope"))?;
    let current = fs::canonicalize(env::current_exe()?)?;
    let entries = match fs::read_dir(&dir) {
        Ok(v) => v,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    for entry in entries {
        let e = entry?;
        let name = e.file_name().to_string_lossy().into_owned();
        let stem = name.strip_prefix("larch-setup-").and_then(|s| {
            if cfg!(windows) {
                s.strip_suffix(".exe")
            } else {
                Some(s)
            }
        });
        if stem.is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit() || b == b'-'))
            && e.file_type()?.is_file()
            && fs::canonicalize(e.path())? != current
        {
            match fs::remove_file(e.path()) {
                Ok(()) => txn::sync_parent(&e.path())?,
                // A copy still running (its window left open on Windows) cannot be deleted; it
                // goes on a later run, and must not stop this one. Logged, as an uninstall has no
                // later run and a Unix permission error is a real one.
                Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => {
                    t.log(&format!("left {}: {err}", e.path().display()));
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
    Ok(())
}
/// Deletes the running copy on Unix or schedules one-file Windows cleanup after exit.
pub(super) fn finish_uninstall(t: &Transaction) -> Result<()> {
    clean_old(t)?;
    let exe = fs::canonicalize(env::current_exe()?)?;
    let dir = t.root().join(".run");
    if exe.parent() != Some(dir.as_path()) {
        return Ok(());
    }
    #[cfg(unix)]
    {
        fs::remove_file(exe)?;
        txn::sync_parent(&dir.join("removed"))?;
        files::empty_dirs(&dir)?;
    }
    #[cfg(windows)]
    {
        // Values go through environment variables, never command interpolation. No recursive rmdir.
        let script = "ping -n 3 127.0.0.1 >nul & del /q \"%LARCH_RUN_FILE%\" & rmdir \"%LARCH_RUN_DIR%\" & rmdir \"%LARCH_ROOT%\"";
        use std::os::windows::process::CommandExt;
        Command::new("cmd.exe")
            .args(["/d", "/v:off", "/c", script])
            .env("LARCH_RUN_FILE", exe)
            .env("LARCH_RUN_DIR", dir)
            .env("LARCH_ROOT", t.root())
            .creation_flags(0x00000008 | 0x08000000)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
    }
    Ok(())
}
/// Relaunches the app after releasing the setup lock, including a transaction error marker.
pub(super) fn launch(app: &txn::Component, error: Option<&str>, roots: &Roots) -> Result<()> {
    let exe = if cfg!(target_os = "macos") {
        app.dest.join("Contents/MacOS/larch-desktop")
    } else {
        app.dest.join(if cfg!(windows) {
            "larch-desktop.exe"
        } else {
            "larch-desktop"
        })
    };
    let mut c = Command::new(exe);
    if let Some(txn) = error {
        c.args(["--setup-error", txn]);
    }
    roots.isolate(&mut c);
    c.env_remove("LARCH_RELEASES_TOKEN").spawn()?;
    Ok(())
}
