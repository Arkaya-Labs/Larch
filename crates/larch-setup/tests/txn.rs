//! Crash recovery is checked against complete old and new filesystem snapshots.
use larch_setup::{
    manifest::{Error, Result},
    txn::*,
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "larch-txn-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(fs::canonicalize(p).unwrap())
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
struct Fake {
    root: PathBuf,
    fail: bool,
}
impl Hooks for Fake {
    fn apply(&mut self, _: &str, _: &str) -> Result<()> {
        fs::write(self.root.join("effect"), "owned")?;
        Ok(())
    }
    fn remove(&mut self, _: &str, _: &str) -> Result<()> {
        match fs::remove_file(self.root.join("effect")) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
            _ => {}
        }
        Ok(())
    }
    fn present(&mut self, _: &str, _: &str) -> Result<bool> {
        Ok(self.root.join("effect").exists())
    }
    fn snapshot(&mut self, _: &str, _: &str) -> Result<Option<String>> {
        match fs::read_to_string(self.root.join("effect")) {
            Ok(v) => Ok(Some(v)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    fn restore(&mut self, _: &str, _: &str, state: &str) -> Result<()> {
        fs::write(self.root.join("effect"), state)?;
        Ok(())
    }
    fn self_test(&mut self, _: &str) -> Result<()> {
        if self.fail {
            Err(Error::Refused("self-test deliberately failed".into()))
        } else {
            Ok(())
        }
    }
}
fn write(root: &Path, path: &str, contents: &str) {
    let p = root.join(path);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, contents).unwrap();
}
fn initial(root: &Path, uninstall: bool) {
    write(root, "cli/larch", "old");
    write(root, ".previous/cli/larch", "older");
    write(root, ".staging/t/cli/larch", "new");
    write(root, "runtime/dll", "runtime");
    write(root, "user-file", "untouched");
    write(
        root,
        "effect",
        "owned;%USERPROFILE%\\bin;;original-position;",
    );
    let receipt = Receipt::empty(root.into(), "stable".into());
    atomic_write(
        &root.join("receipt.json"),
        &serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
    if !uninstall {
        fs::remove_file(root.join("effect")).unwrap();
    }
}
fn journal(t: &Transaction, uninstall: bool) -> Journal {
    let root = t.root();
    let mut to = t.receipt().unwrap().unwrap();
    to.channel = "new".into();
    let steps = if uninstall {
        vec![
            Step::Effect {
                kind: "path".into(),
                value: "owned".into(),
                action: Action::Remove,
            },
            Step::Delete {
                path: root.join("cli/larch"),
            },
            Step::Delete {
                path: root.join("runtime/dll"),
            },
        ]
    } else {
        vec![
            Step::Rename {
                from: root.join(".previous/cli"),
                to: root.join(".trash/t/old"),
            },
            Step::Rename {
                from: root.join("cli"),
                to: root.join(".previous/cli"),
            },
            Step::Rename {
                from: root.join(".staging/t/cli"),
                to: root.join("cli"),
            },
            Step::Link {
                from: root.join("runtime/dll"),
                to: root.join("cli/dll"),
            },
            Step::Effect {
                kind: "path".into(),
                value: "owned".into(),
                action: Action::Apply,
            },
            Step::SelfTest {
                component: "cli".into(),
            },
        ]
    };
    Journal::new("t".into(), t.receipt_digest().unwrap(), to, steps)
}
fn snapshot(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, p: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for e in fs::read_dir(p).unwrap() {
            let p = e.unwrap().path();
            let rel = p.strip_prefix(root).unwrap().to_string_lossy().into_owned();
            if matches!(rel.as_str(), ".lock" | "setup.log") {
                continue;
            }
            if p.is_dir() {
                out.insert(format!("{rel}/"), Vec::new());
                walk(root, &p, out);
            } else {
                let b = fs::read(&p).unwrap();
                let b = if rel == "receipt.json" {
                    String::from_utf8(b)
                        .unwrap()
                        .replace(&root.to_string_lossy().replace('\\', "\\\\"), "ROOT")
                        .into_bytes()
                } else {
                    b
                };
                out.insert(rel, b);
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}
#[test]
fn crash_child() {
    let Some(root) = std::env::var_os("TXN_CHILD_ROOT") else {
        return;
    };
    let root = PathBuf::from(root);
    let t = Transaction::acquire(&root).unwrap();
    let uninstall = std::env::var_os("TXN_UNINSTALL").is_some();
    let mut h = Fake { root, fail: false };
    t.run(journal(&t, uninstall), &mut h).unwrap();
}
#[test]
fn every_step_before_and_after_journal_recovers_forward_or_back() {
    for uninstall in [false, true] {
        let oracle = Temp::new();
        initial(&oracle.0, uninstall);
        let t = Transaction::acquire(&oracle.0).unwrap();
        let mut hooks = Fake {
            root: oracle.0.clone(),
            fail: false,
        };
        let j = journal(&t, uninstall);
        let count = j.steps.len();
        t.run(j, &mut hooks).unwrap();
        let new = snapshot(&oracle.0);
        drop(t);
        for k in 0..count {
            for mode in ["before_journal", "after_journal"] {
                for forward in [true, false] {
                    let dir = Temp::new();
                    initial(&dir.0, uninstall);
                    let old = snapshot(&dir.0);
                    let mut cmd = Command::new(std::env::current_exe().unwrap());
                    cmd.args(["--exact", "crash_child"])
                        .env("TXN_CHILD_ROOT", &dir.0)
                        .env("LARCH_SETUP_CRASH_AT", k.to_string())
                        .env("LARCH_SETUP_CRASH_MODE", mode);
                    if uninstall {
                        cmd.env("TXN_UNINSTALL", "1");
                    }
                    let status = cmd.output().unwrap().status;
                    assert!(!status.success(), "hook did not crash: {k} {mode}");
                    let t = Transaction::acquire(&dir.0).unwrap();
                    let mut h = Fake {
                        root: dir.0.clone(),
                        fail: false,
                    };
                    let direction = t.recover(&mut h, |_| Ok(forward)).unwrap();
                    assert_eq!(
                        direction,
                        if forward {
                            Recovery::Forward
                        } else {
                            Recovery::Back
                        }
                    );
                    assert_eq!(
                        snapshot(&dir.0),
                        if forward { new.clone() } else { old },
                        "uninstall={uninstall}, k={k}, {mode}, forward={forward}"
                    );
                    assert_eq!(
                        t.recover(&mut h, |_| panic!("no journal")).unwrap(),
                        Recovery::None
                    );
                }
            }
        }
    }
}
#[test]
fn failure_rolls_back_and_lock_is_exclusive() {
    let dir = Temp::new();
    initial(&dir.0, false);
    let old = snapshot(&dir.0);
    let t = Transaction::acquire(&dir.0).unwrap();
    assert!(matches!(Transaction::acquire(&dir.0), Err(Error::Busy)));
    let mut h = Fake {
        root: dir.0.clone(),
        fail: true,
    };
    let err = t.run(journal(&t, false), &mut h).unwrap_err().to_string();
    assert!(err.contains("step 5") && err.contains("deliberately failed"));
    assert_eq!(snapshot(&dir.0), old);
}
#[test]
fn no_journal_preserves_staging_previous_and_truncated_temp() {
    let dir = Temp::new();
    initial(&dir.0, false);
    write(&dir.0, "receipt.json.tmp", "{truncated");
    let old = snapshot(&dir.0);
    let t = Transaction::acquire(&dir.0).unwrap();
    let mut h = Fake {
        root: dir.0.clone(),
        fail: false,
    };
    assert_eq!(
        t.recover(&mut h, |_| panic!("must not check staging"))
            .unwrap(),
        Recovery::None
    );
    assert_eq!(snapshot(&dir.0), old);
    t.run(journal(&t, false), &mut h).unwrap();
    assert!(!dir.0.join("receipt.json.tmp").exists());
}
#[test]
fn receipt_mismatch_changes_nothing_but_commit_window_is_finalized() {
    let dir = Temp::new();
    initial(&dir.0, false);
    let t = Transaction::acquire(&dir.0).unwrap();
    let mut j = journal(&t, false);
    j.from_receipt = "wrong".into();
    atomic_write(
        &dir.0.join("journal.json"),
        &serde_json::to_vec(&j).unwrap(),
    )
    .unwrap();
    let old = snapshot(&dir.0);
    let mut h = Fake {
        root: dir.0.clone(),
        fail: false,
    };
    assert!(matches!(
        t.recover(&mut h, |_| Ok(true)),
        Err(Error::StateChangedOutside)
    ));
    assert_eq!(snapshot(&dir.0), old);
    fs::remove_file(dir.0.join("journal.json")).unwrap();
    j = journal(&t, false);
    t.run(j.clone(), &mut h).unwrap();
    let new = snapshot(&dir.0);
    j.done = j.steps.len();
    atomic_write(
        &dir.0.join("journal.json"),
        &serde_json::to_vec(&j).unwrap(),
    )
    .unwrap();
    assert_eq!(
        t.recover(&mut h, |_| panic!("committed")).unwrap(),
        Recovery::Forward
    );
    assert_eq!(snapshot(&dir.0), new);
}

#[cfg(unix)]
#[test]
fn symlink_parent_cannot_escape_transaction_root() {
    let dir = Temp::new();
    let outside = Temp::new();
    initial(&dir.0, false);
    std::os::unix::fs::symlink(&outside.0, dir.0.join("escape")).unwrap();
    let t = Transaction::acquire(&dir.0).unwrap();
    let mut j = journal(&t, false);
    j.steps = vec![Step::Rename {
        from: dir.0.join("cli/larch"),
        to: dir.0.join("escape/stolen"),
    }];
    let mut h = Fake {
        root: dir.0.clone(),
        fail: false,
    };
    assert!(
        t.run(j, &mut h)
            .unwrap_err()
            .to_string()
            .contains("escapes")
    );
    assert!(dir.0.join("cli/larch").exists());
    assert!(!outside.0.join("stolen").exists());
    assert!(!dir.0.join("journal.json").exists());
}

#[test]
fn move_destinations_are_never_overwritten() {
    for delete in [true, false] {
        let dir = Temp::new();
        initial(&dir.0, false);
        write(&dir.0, ".trash/t/0", "foreign");
        let t = Transaction::acquire(&dir.0).unwrap();
        let mut j = journal(&t, false);
        j.steps = vec![if delete {
            Step::Delete {
                path: dir.0.join("cli/larch"),
            }
        } else {
            Step::Rename {
                from: dir.0.join("cli/larch"),
                to: dir.0.join(".trash/t/0"),
            }
        }];
        let old = snapshot(&dir.0);
        let mut h = Fake {
            root: dir.0.clone(),
            fail: false,
        };
        assert!(
            t.run(j, &mut h)
                .unwrap_err()
                .to_string()
                .contains("destination already exists")
        );
        assert_eq!(snapshot(&dir.0), old);
    }
}
