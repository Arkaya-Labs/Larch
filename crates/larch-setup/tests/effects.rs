//! User integrations are exercised only in isolated directories; registry tests are native-only.
use larch_setup::{detect::Os, effects::*};
use serde_json::json;
use std::{
    ffi::OsStr,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
#[cfg(target_os = "linux")]
use std::{
    io::{BufRead, BufReader},
    process::Stdio,
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "larch-effects-{}-{} O'Brien",
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
fn val(root: &Path, key: &str, name: &str) -> String {
    json!({"root":root,key:name}).to_string()
}
fn isolated(temp: &Temp, os: Os, shell: &str, path: &OsStr) -> (Roots, Effects) {
    let roots = Roots::in_test_root(&temp.0).unwrap();
    let effects = Effects::new(roots.clone(), os, shell, path);
    (roots, effects)
}
#[test]
fn windows_path_preserves_unexpanded_values_empty_entries_and_other_bytes() {
    let entries = vec![
        r"C:\Program Files\larch\cli".into(),
        r"C:\Program Files\larch\setup".into(),
    ];
    for raw in [
        "",
        r"%USERPROFILE%\bin",
        r";%USERPROFILE%\bin;;C:\existing;",
        r"%LOCALAPPDATA%\tools;C:\existing",
    ] {
        let added = path_add(raw, &entries);
        assert_eq!(path_add(&added, &entries), added);
        assert_eq!(path_remove(&added, &entries), raw);
    }
    let raw = format!(
        r"pre;%USERPROFILE%\bin;;{};post;{};",
        entries[0], entries[1]
    );
    assert_eq!(path_remove(&raw, &entries), r"pre;%USERPROFILE%\bin;;post;");
    let case = entries[0].to_ascii_uppercase();
    assert_eq!(path_add(&case, &[entries[0].clone()]), case);
    assert_eq!(path_remove(&case, &[entries[0].clone()]), "");
}
#[cfg(unix)]
#[test]
fn user_links_are_idempotent_and_foreign_targets_survive() {
    let temp = Temp::new();
    let (roots, mut e) = isolated(&temp, Os::Linux, "/bin/bash", OsStr::new("/usr/bin"));
    let install = temp.0.join("install");
    for name in ["larch", "larch-agent", "larch-setup"] {
        let v = val(&install, "name", name);
        assert!(!e.present("path", &v).unwrap());
        for _ in 0..2 {
            e.apply("path", &v).unwrap();
            assert!(e.present("path", &v).unwrap());
        }
        assert_eq!(
            fs::read_link(roots.bin().join(name)).unwrap(),
            install
                .join(if name == "larch-setup" {
                    "setup"
                } else {
                    "cli"
                })
                .join(name)
        );
        for _ in 0..2 {
            e.remove("path", &v).unwrap();
            assert!(!e.present("path", &v).unwrap());
        }
    }
    let v = val(&install, "name", "larch");
    fs::write(roots.bin().join("larch"), "user file").unwrap();
    e.apply("path", &v).unwrap();
    e.remove("path", &v).unwrap();
    assert_eq!(
        fs::read_to_string(roots.bin().join("larch")).unwrap(),
        "user file"
    );
    assert!(
        e.take_messages()
            .iter()
            .any(|s| s.contains("not our symlink"))
    );
    fs::remove_file(roots.bin().join("larch")).unwrap();
    std::os::unix::fs::symlink(temp.0.join("foreign"), roots.bin().join("larch")).unwrap();
    e.apply("path", &v).unwrap();
    e.remove("path", &v).unwrap();
    assert_eq!(
        fs::read_link(roots.bin().join("larch")).unwrap(),
        temp.0.join("foreign")
    );
}
#[cfg(unix)]
#[test]
fn shell_blocks_preserve_edits_file_permissions_and_unowned_temp_files() {
    use std::os::unix::fs::PermissionsExt;
    for (shell, names) in [
        ("bash", vec!["bashrc", "profile"]),
        ("zsh", vec!["zshenv"]),
        ("fish", vec!["fish"]),
    ] {
        let temp = Temp::new();
        let (roots, mut e) = isolated(
            &temp,
            Os::Linux,
            &format!("/bin/{shell}"),
            OsStr::new("/usr/bin"),
        );
        fs::create_dir_all(&roots.home).unwrap();
        for name in names {
            let file = match name {
                "bashrc" => roots.home.join(".bashrc"),
                "profile" => roots.home.join(".profile"),
                "zshenv" => roots.home.join(".zshenv"),
                _ => roots.config.join("fish/conf.d/larch.fish"),
            };
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(&file, "# existing without newline").unwrap();
            fs::set_permissions(&file, fs::Permissions::from_mode(0o600)).unwrap();
            let mut legacy = file.as_os_str().to_owned();
            legacy.push(".tmp");
            fs::write(Path::new(&legacy), "editor state").unwrap();
            let v = val(&temp.0.join("install"), "file", name);
            for _ in 0..2 {
                e.apply("shell", &v).unwrap();
                assert!(e.present("shell", &v).unwrap());
            }
            let installed = fs::read_to_string(&file).unwrap();
            assert_eq!(installed.matches("# >>> larch-setup >>>").count(), 1);
            assert_eq!(
                fs::metadata(&file).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(
                fs::read_to_string(Path::new(&legacy)).unwrap(),
                "editor state"
            );
            e.remove("shell", &v).unwrap();
            e.remove("shell", &v).unwrap();
            assert_eq!(
                fs::read_to_string(&file).unwrap(),
                "# existing without newline"
            );
            e.apply("shell", &v).unwrap();
            let edited = fs::read_to_string(&file)
                .unwrap()
                .replace("# <<< larch-setup <<<", "# my changed end");
            fs::write(&file, &edited).unwrap();
            e.remove("shell", &v).unwrap();
            e.apply("shell", &v).unwrap();
            assert_eq!(fs::read_to_string(&file).unwrap(), edited);
            assert!(
                e.take_messages()
                    .iter()
                    .any(|s| s.contains("edited shell block"))
            );
        }
    }
}
#[cfg(unix)]
#[test]
fn shell_condition_mac_always_and_foreign_symlinks_are_preserved() {
    let temp = Temp::new();
    let roots = Roots::in_test_root(&temp.0).unwrap();
    let path = std::env::join_paths([roots.bin()]).unwrap();
    let mut e = Effects::new(roots.clone(), Os::Linux, "/bin/zsh", &path);
    let v = val(&temp.0.join("install"), "file", "zshenv");
    e.apply("shell", &v).unwrap();
    assert!(!roots.home.join(".zshenv").exists());
    let mut e = Effects::new(roots.clone(), Os::Macos, "/bin/zsh", &path);
    e.apply("shell", &v).unwrap();
    assert!(e.present("shell", &v).unwrap());
    e.remove("shell", &v).unwrap();
    let foreign = temp.0.join("dotfiles");
    fs::write(&foreign, "keep").unwrap();
    fs::remove_file(roots.home.join(".zshenv")).unwrap();
    std::os::unix::fs::symlink(&foreign, roots.home.join(".zshenv")).unwrap();
    e.apply("shell", &v).unwrap();
    e.remove("shell", &v).unwrap();
    assert!(
        fs::symlink_metadata(roots.home.join(".zshenv"))
            .unwrap()
            .is_symlink()
    );
    assert_eq!(fs::read_to_string(foreign).unwrap(), "keep");
}
#[test]
fn linux_launcher_quotes_exec_and_copies_only_the_existing_app_icon() {
    let temp = Temp::new();
    let (roots, mut e) = isolated(&temp, Os::Linux, "", OsStr::new("/usr/bin"));
    let install = temp.0.join("install");
    for file in ["entry", "icon", "setup"] {
        let v = val(&install, "file", file);
        for _ in 0..2 {
            e.apply("launcher", &v).unwrap();
            assert!(e.present("launcher", &v).unwrap());
        }
    }
    let text = fs::read_to_string(roots.desktop_entry()).unwrap();
    assert!(text.contains(&format!(
        "Exec={}\n",
        desktop_quote(&install.join("app/larch-desktop").to_string_lossy())
    )));
    assert!(text.contains("Name=Larch\n"));
    let text = fs::read_to_string(roots.setup_entry()).unwrap();
    assert!(text.contains(&format!(
        "Exec={}\n",
        desktop_quote(&install.join("setup/larch-installer").to_string_lossy())
    )));
    assert!(text.contains("Name=Larch Setup\n"));
    assert_eq!(fs::read(roots.desktop_icon()).unwrap(), ICON);
    assert_eq!(desktop_quote(r#"a\b"$`%"#), r#""a\\\\b\\"\\$\\`%%""#);
    for file in ["entry", "icon", "setup"] {
        let v = val(&install, "file", file);
        e.remove("launcher", &v).unwrap();
        e.remove("launcher", &v).unwrap();
        assert!(!e.present("launcher", &v).unwrap());
    }
    fs::write(roots.desktop_entry(), "foreign launcher").unwrap();
    let v = val(&install, "file", "entry");
    e.apply("launcher", &v).unwrap();
    e.remove("launcher", &v).unwrap();
    assert_eq!(
        fs::read_to_string(roots.desktop_entry()).unwrap(),
        "foreign launcher"
    );
}
#[cfg(unix)]
#[test]
fn mac_setup_launcher_links_the_window_into_applications_and_keeps_a_foreign_one() {
    let temp = Temp::new();
    let (roots, mut e) = isolated(&temp, Os::Macos, "", OsStr::new("/usr/bin"));
    let install = temp.0.join("install");
    let v = val(&install, "file", "setup");
    let link = roots.home.join("Applications/Larch Setup.app");
    for _ in 0..2 {
        e.apply("launcher", &v).unwrap();
        assert!(e.present("launcher", &v).unwrap());
    }
    assert_eq!(
        fs::read_link(&link).unwrap(),
        install.join("setup/Larch Setup.app")
    );
    e.remove("launcher", &v).unwrap();
    assert!(!e.present("launcher", &v).unwrap());
    assert!(fs::symlink_metadata(&link).is_err());
    fs::create_dir_all(&link).unwrap();
    e.apply("launcher", &v).unwrap();
    e.remove("launcher", &v).unwrap();
    assert!(link.is_dir());
    assert_eq!(e.take_messages().len(), 2);
}
#[test]
fn environment_test_root_child() {
    let Some(root) = std::env::var_os("EFFECTS_ROOT_CHILD") else {
        return;
    };
    let roots = Roots::from_env().unwrap();
    let root = PathBuf::from(root);
    for p in [
        &roots.home,
        &roots.data,
        &roots.config,
        &roots.cache,
        &roots.appdata,
        &roots.localappdata,
        &roots.start_menu,
    ] {
        assert!(p.starts_with(&root));
    }
    assert!(
        roots
            .registry_prefix
            .unwrap()
            .starts_with("Software\\LarchSetupTests\\")
    );
}
#[test]
fn test_root_override_redirects_every_user_location() {
    let temp = Temp::new();
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "environment_test_root_child"])
        .env("EFFECTS_ROOT_CHILD", &temp.0)
        .env("LARCH_SETUP_TEST_ROOTS", &temp.0)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
#[test]
fn holder_child() {
    if std::env::var_os("EFFECTS_HOLDER_CHILD").is_none() {
        return;
    }
    println!("ready");
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
}
#[cfg(target_os = "linux")]
#[test]
fn holders_name_an_installed_image_without_signalling_it() {
    let temp = Temp::new();
    let exe = temp.0.join("holder");
    fs::copy(std::env::current_exe().unwrap(), &exe).unwrap();
    let spawn = || {
        Command::new(&exe)
            .args(["--exact", "holder_child", "--nocapture"])
            .env("EFFECTS_HOLDER_CHILD", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
    };
    // A child another test forks while the copy is written inherits its write handle until it
    // execs, and Linux refuses to run a file open for writing (ETXTBSY). Retry until it has.
    let mut child = (0..200)
        .find_map(|_| match spawn() {
            Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy => {
                std::thread::sleep(std::time::Duration::from_millis(10));
                None
            }
            r => Some(r.unwrap()),
        })
        .expect("the copy stayed busy for 2 s");
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    loop {
        line.clear();
        assert_ne!(out.read_line(&mut line).unwrap(), 0);
        if line.trim() == "ready" {
            break;
        }
    }
    let got = holders(&temp.0, Os::Linux).unwrap();
    assert!(
        got.iter()
            .any(|p| p.pid == child.id() && p.name == "holder")
    );
    assert!(child.try_wait().unwrap().is_none());
    writeln!(child.stdin.take().unwrap(), "exit").unwrap();
    assert!(child.wait().unwrap().success());
}

#[test]
fn test_roots_isolate_child_environments_without_global_mutation() {
    let temp = Temp::new();
    let roots = Roots::in_test_root(&temp.0).unwrap();
    let mut command = Command::new("unused");
    roots.isolate(&mut command);
    let fields: Vec<_> = command.get_envs().collect();
    assert_eq!(fields.len(), 7);
    assert!(
        fields
            .iter()
            .all(|(_, value)| value.is_some_and(|v| Path::new(v).starts_with(&temp.0)))
    );
}
