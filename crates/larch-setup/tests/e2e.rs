//! The real command coordinator runs only against generated signatures and isolated fake programs.
#[path = "support/release.rs"]
mod release;
#[path = "support/mod.rs"]
mod support;
use larch_setup::{
    effects::Roots,
    manifest::Artifact,
    txn::{Receipt, Transaction},
};
use minisign::KeyPair;
use release::*;
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
    process::{Command, Output, Stdio},
    sync::{OnceLock, atomic::Ordering},
};
/// The flavour detection picks on this OS: macOS has no `cpu` flavour (architecture 4).
const FLAVOUR: &str = if cfg!(target_os = "macos") {
    "metal"
} else {
    "cpu"
};
fn program() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(||{let t=Temp::new();let src=t.0.join("fake.rs");fs::write(&src,r#"
use std::{env,fs,io::{self,Write}};
fn main(){let arg=env::args().nth(1).unwrap_or_default();if arg.is_empty() && env::var_os("LARCH_TEST_WINDOW_FAILS").is_some() {eprintln!("cannot open a display");std::process::exit(3);}if arg.is_empty() || arg=="--setup-error" {fs::write(std::path::Path::new(&env::var("LARCH_INSTALL_ROOT").unwrap()).join("relaunch-marker"),&arg).unwrap();return;}if arg=="--hold" {let lock=fs::File::options().read(true).write(true).open(std::path::Path::new(&env::var("LARCH_INSTALL_ROOT").unwrap()).join(".lock")).unwrap();lock.lock_shared().unwrap();println!("ready");io::stdout().flush().unwrap();let mut s=String::new();io::stdin().read_line(&mut s).unwrap();return;}assert!(env::var("LARCH_SETUP_SELFTEST").is_ok());assert!(env::var("LARCH_RELEASES_TOKEN").is_err());match arg.as_str(){"--version"=>{let p=env::current_exe().unwrap();println!("larch {}",fs::read_to_string(p.parent().unwrap().join("version")).unwrap().trim());},"info"=>println!("{{\"backend\":\"FLAVOUR\"}}"),"--self-test"=>{},_=>std::process::exit(2)}}
"#.replace("FLAVOUR",FLAVOUR)).unwrap();let exe=t.0.join(if cfg!(windows){"fake.exe"}else{"fake"});let status=Command::new("rustc").args(["--edition=2024","-C","strip=symbols","-O"]).arg(&src).arg("-o").arg(&exe).status().unwrap();assert!(status.success());fs::read(exe).unwrap()})
}
struct Fixture {
    temp: Temp,
    root: PathBuf,
    roots: Roots,
    server: Server,
    key: KeyPair,
    keys_path: PathBuf,
    target: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = Temp::new();
        let root = temp.0.join("install");
        let roots = Roots::in_test_root(&temp.0).unwrap();
        let server = Server::new();
        let (key, keys) = signing();
        let keys_path = temp.0.join("keys.json");
        fs::write(&keys_path, serde_json::to_vec(&keys).unwrap()).unwrap();
        let target = if cfg!(windows) {
            "x86_64-pc-windows-msvc"
        } else if cfg!(target_os = "macos") {
            "aarch64-apple-darwin"
        } else {
            "x86_64-unknown-linux-gnu"
        }
        .into();
        Self {
            temp,
            root,
            roots,
            server,
            key,
            keys_path,
            target,
        }
    }
    fn command(&self, args: &[&str]) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_larch-setup"));
        c.args(args)
            .env("LARCH_INSTALL_ROOT", &self.root)
            .env("LARCH_SETUP_TEST_ROOTS", &self.temp.0)
            .env("LARCH_SETUP_TEST_KEYS", &self.keys_path)
            .env("LARCH_SETUP_RELEASES_URL", &self.server.base)
            // A tester's machine sets both (candidate channel, read token); the fixture serves stable.
            .env_remove("LARCH_RELEASES_TOKEN")
            .env_remove("LARCH_SETUP_CHANNEL")
            .env("SHELL", "/bin/bash");
        self.roots.isolate(&mut c);
        c
    }
    fn run(&self, args: &[&str]) -> Output {
        self.command(args).output().unwrap()
    }
    fn good(&self, args: &[&str]) -> Output {
        let o = self.run(args);
        assert!(
            o.status.success(),
            "{args:?}\n{}\n{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        );
        o
    }
    fn receipt(&self) -> Receipt {
        serde_json::from_slice(&fs::read(self.root.join("receipt.json")).unwrap()).unwrap()
    }
    fn artifact(&self, name: &str, version: &str) -> Artifact {
        let exe = if cfg!(windows) { ".exe" } else { "" };
        let cli = format!("larch{exe}");
        let agent = format!("larch-agent{exe}");
        let app = if cfg!(target_os = "macos") {
            "Contents/MacOS/larch-desktop".into()
        } else {
            format!("larch-desktop{exe}")
        };
        let setup = format!("larch-setup{exe}");
        let entries = match name {
            "cli" => vec![
                (cli.as_str(), program(), b'0', None),
                (agent.as_str(), program(), b'0', None),
                ("version", version.as_bytes(), b'0', None),
            ],
            "app" => vec![(app.as_str(), program(), b'0', None)],
            _ => vec![(setup.as_str(), program(), b'0', None)],
        };
        let bytes = archive(&entries);
        let flavour = if name == "setup" { "cpu" } else { FLAVOUR };
        let path = format!("/{name}-{}-{flavour}-{version}.tar.zst", self.target);
        self.server.put(&path, bytes.clone());
        artifact(
            format!("{}{path}", self.server.base),
            &bytes,
            entries.iter().map(|(n, b, _, _)| file(n, b)).collect(),
        )
    }
    fn publish(&self, version: &str, app: &Artifact, setup: &Artifact) -> serde_json::Value {
        let cli = self.artifact("cli", version);
        let mut m = serde_json::to_value(support::manifest()).unwrap();
        m["version"] = version.into();
        m["flavours"] = serde_json::json!({self.target.clone():[FLAVOUR]});
        m["gated"] = serde_json::json!([FLAVOUR]);
        // The Windows CI runner is Windows 10; the floor itself is tested in detect.rs.
        m["floors"]["windows_min_build"] = 0.into();
        m["components"] = serde_json::json!({"cli":{format!("{}-{FLAVOUR}",self.target):cli},"app":{format!("{}-{FLAVOUR}",self.target):app},"setup":{format!("{}-cpu",self.target):setup}});
        self.server.publish(&self.key, version, m.clone());
        m
    }
    fn initial(&self) -> (Artifact, Artifact) {
        let app = self.artifact("app", "1.2.3");
        let setup = self.artifact("setup", "1.2.3");
        self.publish("1.2.3", &app, &setup);
        (app, setup)
    }
}
#[test]
fn install_cli_both_update_only_cli_rollback_modify_and_preserve_user_files() {
    let f = Fixture::new();
    let (app, setup) = f.initial();
    f.good(&["install", "--cli", "--flavour", FLAVOUR, "--yes"]);
    assert!(f.receipt().components.contains_key("cli"));
    assert!(!f.receipt().components.contains_key("app"));
    f.good(&["modify", "--add", "app", "--yes"]);
    let app_before = f.receipt().components["app"].clone();
    let app_file = app_before.dest.join(&app_before.files[0].0);
    let app_bytes = fs::read(&app_file).unwrap();
    f.publish("1.2.4", &app, &setup);
    let before = fs::read(f.root.join("receipt.json")).unwrap();
    let check = f.good(&["update", "--check", "--json"]);
    let value: serde_json::Value = serde_json::from_slice(&check.stdout).unwrap();
    assert_eq!(value["version"], "1.2.4");
    assert_eq!(value["components"], serde_json::json!(["cli"]));
    assert_eq!(fs::read(f.root.join("receipt.json")).unwrap(), before);
    f.good(&["update", "--check"]);
    assert!(
        f.server
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|r| r.to_ascii_lowercase().contains("if-none-match:"))
    );
    f.good(&["update", "--yes"]);
    assert_eq!(f.receipt().components["cli"].version, "1.2.4");
    assert_eq!(f.receipt().components["app"], app_before);
    assert_eq!(fs::read(app_file).unwrap(), app_bytes);
    f.good(&["rollback", "cli", "--yes"]);
    assert_eq!(f.receipt().components["cli"].version, "1.2.3");
    f.good(&["modify", "--remove", "app", "--yes"]);
    assert!(!f.receipt().components.contains_key("app"));
    let a = f.root.join("cli").join("user.txt");
    let b = f.root.join("user.txt");
    fs::write(&a, "user-cli").unwrap();
    fs::write(&b, "user-root").unwrap();
    let out = f.good(&["uninstall", "--yes"]);
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(stderr.contains(a.to_str().unwrap()));
    assert!(stderr.contains(b.to_str().unwrap()));
    assert_eq!(fs::read_to_string(a).unwrap(), "user-cli");
    assert_eq!(fs::read_to_string(b).unwrap(), "user-root");
    assert!(!f.root.join("receipt.json").exists());
    assert!(!f.root.join(".lock").exists());
}
/// The installer window's protocol: every stdout line under --json is one JSON value.
fn lines(o: &Output) -> Vec<serde_json::Value> {
    o.stdout
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect()
}
#[test]
fn dry_run_changes_nothing_and_json_streams_progress_errors_and_the_receipt() {
    let f = Fixture::new();
    f.initial();
    let install = [
        "install",
        "--cli",
        "--app",
        "--flavour",
        FLAVOUR,
        "--yes",
        "--json",
    ];
    let plan = &lines(&f.good(&[&install[..], &["--dry-run"]].concat()))[0];
    assert_eq!(plan["flavour"], FLAVOUR);
    assert_eq!(plan["version"], "1.2.3");
    let parts: Vec<_> = plan["components"].as_object().unwrap().keys().collect();
    assert_eq!(parts, ["app", "cli", "setup"]);
    assert!(plan["required"].as_u64().unwrap() > plan["download"].as_u64().unwrap());
    assert!(!f.root.join("receipt.json").exists());
    f.server.cut.store(true, Ordering::Relaxed);
    let fail = f.run(&install);
    assert!(!fail.status.success());
    let error = lines(&fail).pop().unwrap();
    assert_eq!(
        (&error["event"], &error["busy"]),
        (&"error".into(), &false.into())
    );
    let out = lines(&f.good(&install));
    assert!(
        out.iter()
            .any(|l| l["event"] == "download" && l["done"] == l["total"])
    );
    assert!(out.iter().any(|l| l["event"] == "apply"));
    let receipt: Receipt = serde_json::from_value(out.last().unwrap().clone()).unwrap();
    assert_eq!(receipt, f.receipt());
}
#[test]
fn pin_downgrade_and_a_cut_download_are_safe_and_resumable() {
    let f = Fixture::new();
    let (app, setup) = f.initial();
    f.server.cut.store(true, Ordering::Relaxed);
    let fail = f.run(&["install", "--cli", "--app", "--flavour", FLAVOUR, "--yes"]);
    assert!(!fail.status.success());
    assert!(!f.root.join("receipt.json").exists());
    f.good(&["install", "--cli", "--app", "--flavour", FLAVOUR, "--yes"]);
    assert!(
        f.server
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|r| r.to_ascii_lowercase().contains("if-range:"))
    );
    f.publish("1.2.4", &app, &setup);
    f.good(&["update", "--yes"]);
    f.good(&["pin", "1.2.4", "--yes"]);
    let pinned = fs::read(f.root.join("receipt.json")).unwrap();
    assert!(!f.run(&["update", "--yes"]).status.success());
    assert_eq!(fs::read(f.root.join("receipt.json")).unwrap(), pinned);
    f.good(&["unpin", "--yes"]);
    f.publish("1.2.3", &app, &setup);
    let before = fs::read(f.root.join("receipt.json")).unwrap();
    assert!(!f.run(&["update", "--yes"]).status.success());
    assert_eq!(fs::read(f.root.join("receipt.json")).unwrap(), before);
    f.good(&[
        "install",
        "--cli",
        "--version",
        "1.2.3",
        "--flavour",
        FLAVOUR,
        "--yes",
    ]);
    assert_eq!(f.receipt().components["cli"].version, "1.2.3");
}
#[test]
fn locks_refuse_without_writes_and_name_the_shared_holder() {
    let f = Fixture::new();
    f.initial();
    f.good(&["install", "--cli", "--flavour", FLAVOUR, "--yes"]);
    let before = fs::read(f.root.join("receipt.json")).unwrap();
    let log = fs::read(f.root.join("setup.log")).unwrap();
    let guard = Transaction::acquire(&f.root).unwrap();
    let out = f.run(&["repair", "--yes"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("another larch-setup is running"));
    assert_eq!(fs::read(f.root.join("setup.log")).unwrap(), log);
    drop(guard);
    let exe = f
        .root
        .join("cli")
        .join(if cfg!(windows) { "larch.exe" } else { "larch" });
    let mut holder = Command::new(exe)
        .arg("--hold")
        .env("LARCH_INSTALL_ROOT", &f.root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(holder.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    assert_eq!(line.trim(), "ready");
    let out = f.run(&["update", "--yes"]);
    assert!(!out.status.success());
    let text = String::from_utf8_lossy(&out.stderr);
    assert!(text.contains("larch") && text.contains(&holder.id().to_string()));
    assert_eq!(fs::read(f.root.join("receipt.json")).unwrap(), before);
    assert_eq!(fs::read(f.root.join("setup.log")).unwrap(), log);
    writeln!(holder.stdin.take().unwrap(), "exit").unwrap();
    assert!(holder.wait().unwrap().success());
}
#[test]
fn purge_removes_known_data_and_asks_about_unknown_files() {
    let f = Fixture::new();
    f.initial();
    f.good(&["install", "--cli", "--app", "--flavour", FLAVOUR, "--yes"]);
    let data = f.roots.data_dir();
    fs::create_dir_all(data.join("models")).unwrap();
    let model = data.join("models/m3.gguf");
    fs::write(&model, "fixture").unwrap();
    larch_fetch::index_add(
        &data,
        larch_fetch::IndexEntry {
            file: model.to_string_lossy().into_owned(),
            repo: "org/repo".into(),
            revision: "a".repeat(40),
            sha256: larch_fetch::sha256_bytes(b"fixture"),
            size: 7,
            alias: Some("m3".into()),
        },
    )
    .unwrap();
    for name in ["chat", "agent", "memory", "logs"] {
        fs::create_dir_all(data.join(name)).unwrap();
        fs::write(data.join(name).join("known"), "fixture").unwrap();
    }
    let unknown = data.join("user.txt");
    fs::write(&unknown, "keep").unwrap();
    let mut child = f
        .command(&["uninstall", "--purge", "--yes"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.take().unwrap(), "n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("Unrecognized data file"));
    assert!(!model.exists());
    for name in ["chat", "agent", "memory", "logs"] {
        assert!(!data.join(name).exists());
    }
    assert_eq!(fs::read_to_string(unknown).unwrap(), "keep");
    assert!(larch_fetch::index_read(&data).unwrap().is_empty());
}

#[test]
fn explicit_selftest_failure_restores_components_receipt_and_offline_manifest() {
    let f = Fixture::new();
    let (app, setup) = f.initial();
    f.good(&["install", "--yes", "--flavour", FLAVOUR]);
    let receipt = fs::read(f.root.join("receipt.json")).unwrap();
    let manifest = fs::read(f.root.join("manifest.json")).unwrap();
    let old_cli = fs::read(f.root.join("cli/version")).unwrap();
    let mut m = f.publish("1.2.4", &app, &setup);
    let exe = if cfg!(windows) { "larch.exe" } else { "larch" };
    let agent = if cfg!(windows) {
        "larch-agent.exe"
    } else {
        "larch-agent"
    };
    let entries = [
        (exe, program(), b'0', None),
        (agent, program(), b'0', None),
        ("version", &b"9.9.9"[..], b'0', None),
    ];
    let bytes = archive(&entries);
    let path = format!("/cli-{}-{FLAVOUR}-1.2.4.tar.zst", f.target);
    f.server.put(&path, bytes.clone());
    let a = artifact(
        format!("{}{path}", f.server.base),
        &bytes,
        entries.iter().map(|(n, b, _, _)| file(n, b)).collect(),
    );
    m["components"]["cli"][format!("{}-{FLAVOUR}", f.target)] = serde_json::to_value(a).unwrap();
    f.server.publish(&f.key, "1.2.4", m);
    let out = f.run(&["update", "--yes"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("version mismatch"));
    assert_eq!(fs::read(f.root.join("receipt.json")).unwrap(), receipt);
    assert_eq!(fs::read(f.root.join("manifest.json")).unwrap(), manifest);
    assert_eq!(fs::read(f.root.join("cli/version")).unwrap(), old_cli);
    assert!(!f.root.join("journal.json").exists());
}
#[test]
fn interactive_model_failure_happens_after_commit_and_yes_keeps_data() {
    let f = Fixture::new();
    let (app, setup) = f.initial();
    let mut m = f.publish("1.2.3", &app, &setup);
    m["models"] = serde_json::json!([{"display":"m3","repo":"org/repo","file":"m3.gguf","revision":"","sha256":"a".repeat(64),"size":1,"min_mem_gb":0,"license_url":"https://example.invalid/license"}]);
    f.server.publish(&f.key, "1.2.3", m);
    let mut child = f
        .command(&[])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    write!(child.stdin.take().unwrap(), "cli\ny\n1\ny\n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(!out.status.success());
    let text = String::from_utf8_lossy(&out.stderr);
    assert!(
        text.contains("components committed; model download failed"),
        "{text}"
    );
    assert!(text.contains("https://example.invalid/license"));
    assert!(f.receipt().components.contains_key("cli"));
    assert!(!f.receipt().components.contains_key("app"));
    let data = f.roots.data_dir();
    fs::create_dir_all(data.join("chat")).unwrap();
    fs::write(data.join("chat/user"), "keep").unwrap();
    f.good(&["uninstall", "--yes"]);
    assert_eq!(fs::read_to_string(data.join("chat/user")).unwrap(), "keep");
}
#[test]
fn a_double_click_takes_its_channel_from_larch_setup_channel() {
    let f = Fixture::new();
    f.initial();
    let mut child = f
        .command(&[])
        .env("LARCH_SETUP_CHANNEL", "candidate")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    write!(child.stdin.take().unwrap(), "cli\ny\n").unwrap();
    assert!(!child.wait_with_output().unwrap().status.success());
    let requests = f.server.requests.lock().unwrap();
    assert!(
        requests
            .iter()
            .any(|r| r.contains("/channels/candidate.json"))
    );
    assert!(!requests.iter().any(|r| r.contains("/channels/stable.json")));
}
#[test]
fn relaunch_waits_for_the_app_shared_lock_then_starts_the_app() {
    let f = Fixture::new();
    let (app, setup) = f.initial();
    f.good(&["install", "--app", "--cli", "--flavour", FLAVOUR, "--yes"]);
    f.publish("1.2.4", &app, &setup);
    let c = &f.receipt().components["app"];
    let exe = c.dest.join(&c.files[0].0);
    let mut holder = Command::new(exe)
        .arg("--hold")
        .env("LARCH_INSTALL_ROOT", &f.root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(holder.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    assert_eq!(line.trim(), "ready");
    let mut update = f
        .command(&["update", "--relaunch-app", "--yes"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert!(update.try_wait().unwrap().is_none());
    writeln!(holder.stdin.take().unwrap(), "exit").unwrap();
    assert!(holder.wait().unwrap().success());
    let out = update.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(f.root.join("relaunch-marker").exists());
    assert_eq!(f.receipt().components["cli"].version, "1.2.4");
}

#[test]
fn uninstall_uses_signed_offline_cache_even_when_install_floors_are_no_longer_met() {
    let f = Fixture::new();
    let (app, setup) = f.initial();
    let mut m = f.publish("1.2.3", &app, &setup);
    f.good(&["install", "--cli", "--yes", "--flavour", FLAVOUR]);
    m["floors"]["glibc_min"] = "999.0".into();
    m["floors"]["macos_min"] = "999.0".into();
    m["floors"]["windows_min_build"] = u32::MAX.into();
    let bytes = serde_json::to_vec(&m).unwrap();
    fs::write(f.root.join("manifest.json"), &bytes).unwrap();
    fs::write(
        f.root.join("manifest.minisig"),
        sign(&f.key, &bytes, "1.2.3"),
    )
    .unwrap();
    f.server.files.lock().unwrap().clear();
    let count = f.server.requests.lock().unwrap().len();
    f.good(&["uninstall", "--yes"]);
    assert_eq!(f.server.requests.lock().unwrap().len(), count);
    assert!(!f.root.join("receipt.json").exists());
}

#[test]
fn a_second_setup_process_exits_busy_while_the_first_fetches() {
    let f = Fixture::new();
    let (app, setup) = f.initial();
    f.good(&["install", "--cli", "--yes", "--flavour", FLAVOUR]);
    f.publish("1.2.4", &app, &setup);
    let server_files = f.server.files.lock().unwrap();
    let spawn = || {
        f.command(&["update", "--yes"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };
    let mut first = spawn();
    // A freshly built binary can take seconds to start on Windows (its first scan).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match Transaction::acquire(&f.root) {
            Err(larch_setup::manifest::Error::Busy) => break,
            Ok(lock) => drop(lock),
            Err(e) => panic!("{e}"),
        }
        // The probe holds the lock for a moment; a first setup that tried it then exits busy.
        if first.try_wait().unwrap().is_some() {
            let out = first.wait_with_output().unwrap();
            let err = String::from_utf8_lossy(&out.stderr);
            assert!(err.contains("another larch-setup is running"), "{err}");
            first = spawn();
        }
        assert!(
            std::time::Instant::now() < deadline,
            "first setup did not take the lock"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let receipt = fs::read(f.root.join("receipt.json")).unwrap();
    let log = fs::read(f.root.join("setup.log")).unwrap();
    let second = f.run(&["update", "--yes"]);
    assert!(!second.status.success());
    assert!(String::from_utf8_lossy(&second.stderr).contains("another larch-setup is running"));
    assert_eq!(fs::read(f.root.join("receipt.json")).unwrap(), receipt);
    assert_eq!(fs::read(f.root.join("setup.log")).unwrap(), log);
    drop(server_files);
    let out = first.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(f.receipt().components["cli"].version, "1.2.4");
}
#[test]
fn window_opens_the_installer_from_the_signed_setup_package_and_installs_nothing() {
    let f = Fixture::new();
    let exe = if cfg!(windows) { ".exe" } else { "" };
    let setup_name = format!("larch-setup{exe}");
    let window = if cfg!(target_os = "macos") {
        "Larch Setup.app/Contents/MacOS/larch-installer".to_owned()
    } else {
        format!("larch-installer{exe}")
    };
    let entries = [
        (setup_name.as_str(), program(), b'0', None),
        (window.as_str(), program(), b'0', None),
    ];
    let bytes = archive(&entries);
    let path = format!("/setup-{}-cpu-1.2.3.tar.zst", f.target);
    f.server.put(&path, bytes.clone());
    let setup = artifact(
        format!("{}{path}", f.server.base),
        &bytes,
        entries.iter().map(|(n, b, _, _)| file(n, b)).collect(),
    );
    let app = f.artifact("app", "1.2.3");
    f.publish("1.2.3", &app, &setup);
    // The fake window, started with no arguments, writes its marker into the install root.
    fs::create_dir_all(&f.root).unwrap();
    let temp = f.temp.0.join("tmp");
    fs::create_dir(&temp).unwrap();
    let o = f
        .command(&["window"])
        .env("TMPDIR", &temp)
        .env("TEMP", &temp)
        .env("TMP", &temp)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let marker = f.root.join("relaunch-marker");
    let start = std::time::Instant::now();
    while !marker.exists() && start.elapsed() < std::time::Duration::from_secs(20) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(marker.exists(), "the window never started");
    let dirs: Vec<PathBuf> = fs::read_dir(&temp)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    assert_eq!(dirs.len(), 1, "{dirs:?}");
    assert!(dirs[0].join("setup").join(&window).is_file());
    assert!(!dirs[0].join("setup.tar.zst").exists());
    assert!(!f.root.join("receipt.json").exists());
    fs::remove_dir_all(&dirs[0]).unwrap();
    // A window that exits at once with a failure is reported with its log, and leaves no folder.
    let o = f
        .command(&["window"])
        .env("TMPDIR", &temp)
        .env("TEMP", &temp)
        .env("TMP", &temp)
        .env("LARCH_TEST_WINDOW_FAILS", "1")
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success());
    assert!(err.contains("did not start"), "{err}");
    assert!(err.contains("cannot open a display"), "{err}");
    assert!(err.contains("--terminal"), "{err}");
    assert_eq!(fs::read_dir(&temp).unwrap().count(), 0);
    // A package without the window is refused, and leaves no folder behind.
    let bare = f.artifact("setup", "1.2.4");
    f.publish("1.2.4", &app, &bare);
    let o = f
        .command(&["window"])
        .env("TMPDIR", &temp)
        .env("TEMP", &temp)
        .env("TMP", &temp)
        .output()
        .unwrap();
    assert!(!o.status.success());
    assert!(
        String::from_utf8_lossy(&o.stderr).contains("no installer window"),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert_eq!(fs::read_dir(&temp).unwrap().count(), 0);
}
