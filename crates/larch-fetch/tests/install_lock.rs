//! Installed copies hold the setup lock; developer copies and matched self-tests do not.

use larch_fetch::needs_install_lock;
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
};

#[test]
fn decision_covers_paths_and_journal_bypass() {
    let root = Path::new("/install");
    let exe = root.join("cli/larch");
    let app = Path::new("/Applications/Larch.app");
    assert!(needs_install_lock(&exe, root, None, None, None));
    assert!(!needs_install_lock(
        Path::new("/install-copy/larch"),
        root,
        None,
        None,
        None
    ));
    assert!(needs_install_lock(
        &app.join("Contents/MacOS/larch-desktop"),
        root,
        Some(app),
        None,
        None
    ));
    assert!(!needs_install_lock(
        &exe,
        root,
        None,
        Some("txn"),
        Some("txn")
    ));
    assert!(needs_install_lock(
        &exe,
        root,
        None,
        Some("stray"),
        Some("txn")
    ));
    assert!(needs_install_lock(&exe, root, None, Some("stray"), None));
}

#[test]
fn lock_holder_child() {
    if std::env::var_os("INSTALL_LOCK_CHILD").is_none() {
        return;
    }
    let lock = larch_fetch::hold_install_lock().unwrap();
    assert!(lock.is_some());
    println!("locked");
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    drop(lock);
}

#[test]
fn installed_process_prevents_an_exclusive_lock() {
    let root = std::env::temp_dir().join(format!("larch-install-lock-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let exe = root.join(if cfg!(windows) {
        "holder.exe"
    } else {
        "holder"
    });
    fs::copy(std::env::current_exe().unwrap(), &exe).unwrap();
    let file = File::create(root.join(".lock")).unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&exe, root.join("link")).unwrap();
        let canonical = fs::canonicalize(root.join("link")).unwrap();
        assert!(needs_install_lock(
            &canonical,
            &fs::canonicalize(&root).unwrap(),
            None,
            None,
            None
        ));
    }
    let mut child = Command::new(&exe)
        .args(["--exact", "lock_holder_child", "--nocapture"])
        .env("INSTALL_LOCK_CHILD", "1")
        .env("LARCH_INSTALL_ROOT", &root)
        .env_remove("LARCH_SETUP_SELFTEST")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    loop {
        line.clear();
        assert_ne!(output.read_line(&mut line).unwrap(), 0);
        if line.trim() == "locked" {
            break;
        }
    }
    assert!(matches!(
        file.try_lock(),
        Err(std::fs::TryLockError::WouldBlock)
    ));
    writeln!(child.stdin.take().unwrap(), "exit").unwrap();
    assert!(child.wait().unwrap().success());
    file.try_lock().unwrap();
    drop(file);
    fs::remove_dir_all(root).unwrap();
}
