//! Tiny executable fixtures exercise the protocol without an engine or GPU.
#![cfg(unix)]
use larch_setup::{
    effects::Roots,
    selftest::{self, Limits, Verification},
    txn::Component,
};
use serde_json::json;
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::PathBuf,
    sync::{
        Mutex, MutexGuard,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
/// One test at a time: a fork in one test inherits another's still-open script, and exec then
/// fails with ETXTBSY ("Text file busy").
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "larch-selftest-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn fixture(t: &Temp, flavour: &str, version: &str, info: serde_json::Value) -> (Component, Roots) {
    let dest = t.0.join("cli");
    fs::create_dir_all(&dest).unwrap();
    fs::write(dest.join("version"), format!("larch {version} (test)\n")).unwrap();
    fs::write(dest.join("info.json"), info.to_string()).unwrap();
    fs::write(dest.join("larch"),"#!/bin/sh\n[ \"$LARCH_SETUP_SELFTEST\" = test-txn ] || exit 9\n[ -z \"${LARCH_RELEASES_TOKEN+x}\" ] || exit 10\ncase \"$HOME\" in */home) ;; *) exit 11;; esac\ncase \"$1\" in --version) exec cat \"$(dirname \"$0\")/version\";; info) exec cat \"$(dirname \"$0\")/info.json\";; *) exit 12;; esac\n").unwrap();
    fs::set_permissions(dest.join("larch"), fs::Permissions::from_mode(0o755)).unwrap();
    (
        Component {
            version: "1.2.3".into(),
            flavour: flavour.into(),
            dest,
            files: vec![],
        },
        Roots::in_test_root(&t.0).unwrap(),
    )
}
fn run(t: &Temp, c: &Component, r: &Roots) -> larch_setup::manifest::Result<Verification> {
    selftest::component(
        "cli",
        c,
        &t.0,
        "test-txn",
        r,
        true,
        Limits {
            version: Duration::from_secs(2),
            normal: Duration::from_secs(2),
            first_gpu: Duration::from_millis(50),
        },
    )
}
#[test]
fn correct_cpu_and_app_pass_wrong_version_or_backend_fail() {
    let _serial = serial();
    let t = Temp::new();
    let (c, r) = fixture(&t, "cpu", "1.2.3", json!({"backend":"cpu"}));
    assert_eq!(run(&t, &c, &r).unwrap(), Verification::Verified);
    fs::write(c.dest.join("version"), "larch 1.2.2").unwrap();
    assert!(
        run(&t, &c, &r)
            .unwrap_err()
            .to_string()
            .contains("version mismatch")
    );
    fs::write(c.dest.join("version"), "larch 1.2.3").unwrap();
    fs::write(c.dest.join("info.json"), r#"{"backend":"cuda"}"#).unwrap();
    assert!(run(&t, &c, &r).is_err());
    let app = c.dest.join(if cfg!(target_os = "macos") {
        "Contents/MacOS/larch-desktop"
    } else {
        "larch-desktop"
    });
    fs::create_dir_all(app.parent().unwrap()).unwrap();
    fs::write(&app, "#!/bin/sh\n[ \"$1\" = --self-test ]\n").unwrap();
    fs::set_permissions(&app, fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(
        selftest::component("app", &c, &t.0, "test-txn", &r, false, Limits::default()).unwrap(),
        Verification::Verified
    );
}
#[test]
fn explicit_no_device_fails_timeout_is_unverified_and_runtime_must_be_inside_root() {
    let _serial = serial();
    let t = Temp::new();
    let (c, r) = fixture(
        &t,
        "cuda",
        "1.2.3",
        json!({"backend":"cuda","devices":[{"usable":false,"backend":"unknown"}]}),
    );
    assert!(
        run(&t, &c, &r)
            .unwrap_err()
            .to_string()
            .contains("no usable device")
    );
    // A CUDA build whose GPU did not open reports the CPU backend, and the CPU reference as usable.
    fs::write(
        c.dest.join("info.json"),
        json!({"backend":"cpu","devices":[{"usable":true,"backend":"cpu"},{"usable":false,"backend":"unknown"}]})
            .to_string(),
    )
    .unwrap();
    assert!(
        run(&t, &c, &r)
            .unwrap_err()
            .to_string()
            .contains("no usable device")
    );
    let runtime = t.0.join("runtime.so");
    fs::write(&runtime, "fixture").unwrap();
    fs::write(
        c.dest.join("info.json"),
        json!({"backend":"cuda","devices":[{"usable":true,"backend":"cuda"}],"loaded_libraries":[{"name":"runtime.so","path":runtime}]})
            .to_string(),
    )
    .unwrap();
    assert_eq!(run(&t, &c, &r).unwrap(), Verification::Verified);
    let outside = Temp::new();
    let lib = outside.0.join("outside.so");
    fs::write(&lib, "fixture").unwrap();
    let escape = t.0.join("escape.so");
    symlink(lib, &escape).unwrap();
    fs::write(
        c.dest.join("info.json"),
        json!({"backend":"cuda","devices":[{"usable":true,"backend":"cuda"}],"loaded_libraries":[{"name":"escape.so","path":escape}]})
            .to_string(),
    )
    .unwrap();
    assert!(run(&t, &c, &r).unwrap_err().to_string().contains("outside"));
    fs::write(
        c.dest.join("larch"),
        "#!/bin/sh\nif [ \"$1\" = --version ]; then echo 'larch 1.2.3'; else exec sleep 10; fi\n",
    )
    .unwrap();
    assert!(matches!(
        run(&t, &c, &r).unwrap(),
        Verification::Unverified(_)
    ));
}
#[test]
fn credentials_child() {
    if std::env::var_os("SELFTEST_ENV_CHILD").is_none() {
        return;
    }
    let t = Temp::new();
    let (c, r) = fixture(&t, "cpu", "1.2.3", json!({"backend":"cpu"}));
    assert_eq!(run(&t, &c, &r).unwrap(), Verification::Verified);
}
#[test]
fn credentials_are_removed_and_transaction_identity_is_set() {
    let _serial = serial();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "credentials_child"])
        .env("SELFTEST_ENV_CHILD", "1")
        .env("LARCH_RELEASES_TOKEN", "not-for-the-child")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
}
