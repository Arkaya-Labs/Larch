//! Generated signatures and local HTTP keep archive trust checks independent of real releases.
#[path = "support/release.rs"]
mod release;
use larch_setup::{fetch::Client, unpack};
use release::*;
use std::{fs, sync::atomic::Ordering};
#[path = "support/mod.rs"]
mod support;
#[test]
fn signed_fetch_resume_etag_and_inventory() {
    let temp = Temp::new();
    let server = Server::new();
    let (key, keys) = signing();
    let mut client = Client::new(Some(server.base.clone()), None, keys).unwrap();
    let body = archive(&[("bin/tool", b"safe", b'0', None)]);
    let a = artifact(
        format!("{}/file.tar.zst", server.base),
        &body,
        vec![file("bin/tool", b"safe")],
    );
    server.put("/file.tar.zst", body.clone());
    let mut m = serde_json::to_value(support::manifest()).unwrap();
    m["components"] = serde_json::json!({"cli":{"x86_64-unknown-linux-gnu-cpu":a}});
    server.publish(&key, "1.2.3", m);
    let r = client.release("stable", None).unwrap();
    assert_eq!(r.manifest.version, "1.2.3");
    let cached = client.pointer("stable", Some(&r.head)).unwrap();
    assert_eq!(cached.body, r.head.body);
    let mut forged = r.head.clone();
    forged.body.push(b' ');
    assert!(client.pointer("stable", Some(&forged)).is_err());
    let path = temp.0.join("file.tar.zst");
    server.cut.store(true, Ordering::Relaxed);
    assert!(client.download(&a, &path, |_, _| true).is_err());
    assert!(!path.exists());
    assert_eq!(
        fs::metadata(larch_fetch::part_path(&path)).unwrap().len(),
        body.len() as u64 / 2
    );
    client.download(&a, &path, |_, _| true).unwrap();
    unpack::unpack(&path, &a, &temp.0.join("out")).unwrap();
    assert_eq!(fs::read(temp.0.join("out/bin/tool")).unwrap(), b"safe");
    unpack::verify_tree(&temp.0.join("out"), &a.files).unwrap();
    assert!(
        server
            .requests
            .lock()
            .unwrap()
            .iter()
            .any(|r| r.contains("If-Range: \"test\"") || r.contains("if-range: \"test\""))
    );
    let mut tampered = body;
    tampered[0] ^= 1;
    server.put("/file.tar.zst", tampered);
    assert!(
        client
            .download(&a, &temp.0.join("bad"), |_, _| true)
            .is_err()
    );
}
#[test]
fn a_missing_release_without_a_token_names_the_token() {
    let server = Server::new();
    let (_, keys) = signing();
    let hint = |token: Option<&str>| {
        let mut client = Client::new(
            Some(server.base.clone()),
            token.map(Into::into),
            keys.clone(),
        )
        .unwrap();
        let err = client.release("stable", None).err().unwrap().to_string();
        assert!(err.contains("HTTP 404"), "{err}");
        err.contains("LARCH_RELEASES_TOKEN")
    };
    assert!(hint(None));
    assert!(!hint(Some("read-token")));
}
#[test]
fn validly_hashed_archives_still_reject_unsafe_paths_links_and_extras() {
    let temp = Temp::new();
    for (i, entries) in [
        vec![("../x", &b"bad"[..], b'0', None)],
        vec![("/x", &b"bad"[..], b'0', None)],
        vec![
            ("tool", &b"safe"[..], b'0', None),
            ("extra", &b"bad"[..], b'0', None),
        ],
        vec![("tool", &b""[..], b'2', Some("../../outside"))],
        vec![("tool", &b""[..], b'3', None)],
        vec![
            ("tool", &b"safe"[..], b'0', None),
            ("tool", &b"safe"[..], b'0', None),
        ],
    ]
    .iter()
    .enumerate()
    {
        let bytes = archive(entries);
        let a = artifact("unused".into(), &bytes, vec![file("tool", b"safe")]);
        let path = temp.0.join(format!("{i}.tar.zst"));
        fs::write(&path, bytes).unwrap();
        assert!(
            unpack::unpack(&path, &a, &temp.0.join(format!("out{i}"))).is_err(),
            "case {i}"
        );
        assert!(!temp.0.join(format!("out{i}")).exists());
    }
    assert!(!temp.0.join("x").exists());
    assert!(unpack::link_target("d/link", "x/../../escape").is_err());
    assert!(unpack::link_target("link", "../escape").is_err());
}
#[cfg(unix)]
#[test]
fn internal_symlinks_and_forward_hardlinks_verify_without_weakening_inventory() {
    let temp = Temp::new();
    let bytes = archive(&[
        ("hard", b"", b'1', Some("d/file")),
        ("link", b"", b'2', Some("d/file")),
        ("d/", b"", b'5', None),
        ("d/file", b"payload", b'0', None),
    ]);
    let mut files = vec![file("d/file", b"payload"), file("hard", b"payload")];
    files.push(("link".into(), unpack::symlink_hash("d/file"), 6));
    let a = artifact("unused".into(), &bytes, files);
    let path = temp.0.join("a");
    fs::write(&path, bytes).unwrap();
    let dest = temp.0.join("out");
    unpack::unpack(&path, &a, &dest).unwrap();
    unpack::unpack(&path, &a, &dest).unwrap();
    assert_eq!(
        fs::read_link(dest.join("link")).unwrap(),
        std::path::Path::new("d/file")
    );
    fs::write(dest.join("hard"), b"changed").unwrap();
    assert!(unpack::verify_tree(&dest, &a.files).is_err());
}

#[test]
fn private_release_uses_contents_and_asset_api_and_bootstrap_uses_signed_bytes() {
    let temp = Temp::new();
    let server = Server::new();
    let (key, keys) = signing();
    let bytes = serde_json::to_vec(&support::manifest()).unwrap();
    let download = "https://github.com/Arkaya-Labs/Larch/releases/download/v1.2.3/manifest.json";
    let exe = temp.0.join("setup");
    fs::write(&exe, b"fixture").unwrap();
    let pointer=serde_json::to_vec(&serde_json::json!({"schema":1,"channel":"stable","version":"1.2.3","manifest_url":download,"manifest_sha256":larch_fetch::sha256_bytes(&bytes),"setup":{"test-target":{"url":"unused","sha256":larch_fetch::sha256_bytes(b"fixture"),"size":7}}})).unwrap();
    server.put("/api/contents/stable.json?ref=channels", pointer.clone());
    server.put(
        "/api/contents/stable.json.minisig?ref=channels",
        sign(&key, &pointer, "1.2.3"),
    );
    server.put("/api/releases/tags/v1.2.3",serde_json::to_vec(&serde_json::json!({"assets":[{"name":"manifest.json","url":"https://api.github.com/repos/Arkaya-Labs/Larch/releases/assets/1"},{"name":"manifest.json.minisig","url":"https://api.github.com/repos/Arkaya-Labs/Larch/releases/assets/2"}]})).unwrap());
    server.put("/api/releases/assets/1", bytes.clone());
    server.put("/api/releases/assets/2", sign(&key, &bytes, "1.2.3"));
    let mut client = Client::new(
        Some(server.base.clone()),
        Some("fixture-secret".into()),
        keys,
    )
    .unwrap();
    let mut r = client.release("stable", None).unwrap();
    r.head.pointer.setup.clear();
    client.bootstrap(&r.head, "test-target", &exe).unwrap();
    fs::write(&exe, b"changed").unwrap();
    assert!(client.bootstrap(&r.head, "test-target", &exe).is_err());
    r.head.body.push(b' ');
    assert!(client.bootstrap(&r.head, "test-target", &exe).is_err());
    let requests = server.requests.lock().unwrap();
    assert!(requests.iter().all(|r| {
        r.to_ascii_lowercase()
            .contains("authorization: bearer fixture-secret")
    }));
    assert!(
        requests
            .iter()
            .filter(|r| r.contains("/api/releases/assets/"))
            .all(|r| r
                .to_ascii_lowercase()
                .contains("accept: application/octet-stream"))
    );
}

fn python(script: &str) -> Option<std::process::Command> {
    // The scripts need tomllib (3.11+); macOS ships python3 3.9.
    match std::process::Command::new("python3")
        .args([
            "-c",
            "import sys; sys.exit(3 if sys.version_info < (3, 11) else 0)",
        ])
        .output()
    {
        Ok(out) if out.status.code() == Some(3) => {
            eprintln!("NOT RUN: python3 {script}; python3 is older than 3.11");
            return None;
        }
        // Windows' python3.exe without Python is the Store alias stub, which exits 9009.
        Ok(out) if cfg!(windows) && out.status.code() == Some(9009) => {
            eprintln!("NOT RUN: python3 {script}; python3 is only the Store alias");
            return None;
        }
        Ok(out) => assert!(
            out.status.success(),
            "python3 exists but cannot start: {} {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("NOT RUN: python3 {script}; python3 is not installed");
            return None;
        }
        Err(e) => panic!("python3 probe: {e}"),
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut command = std::process::Command::new("python3");
    command
        .arg(root.join("scripts").join(script))
        .env("PYTHONDONTWRITEBYTECODE", "1");
    Some(command)
}

#[test]
fn release_script_produces_a_manifest_accepted_by_rust_and_never_gates_unsigned_builds() {
    // The script runs on the Linux release host (release.yml installs zstd); CI's Windows runner
    // has no zstd.
    if std::process::Command::new("zstd")
        .arg("--version")
        .output()
        .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
    {
        eprintln!("NOT RUN: release-manifest.py; zstd is not installed");
        return;
    }
    let Some(mut command) = python("release-manifest.py") else {
        return;
    };
    let temp = Temp::new();
    let archives = temp.0.join("archives");
    fs::create_dir(&archives).unwrap();
    let bytes = archive(&[
        ("d/file", b"fixture", b'0', None),
        ("link", b"", b'2', Some("d/file")),
    ]);
    let name = "cli-x86_64-unknown-linux-gnu-cpu-1.2.3.tar.zst";
    fs::write(archives.join(name), &bytes).unwrap();
    let dest = temp.0.join("manifest.json");
    command
        .args(["--version", "1.2.3", "--archives"])
        .arg(&archives)
        .arg("--output")
        .arg(&dest);
    let out = command.output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let manifest = larch_setup::manifest::parse_manifest(&fs::read(&dest).unwrap()).unwrap();
    assert!(manifest.gated.is_empty());
    // The rows of bench/models.toml whose `use` names "catalog", in file order, and no others.
    let toml = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/models.toml");
    let toml = fs::read_to_string(toml).unwrap();
    let catalog: Vec<&str> = toml
        .split("\n[")
        .filter(|row| {
            row.lines()
                .any(|l| l.starts_with("use = ") && l.contains("\"catalog\""))
        })
        .filter_map(|row| row.lines().find_map(|l| l.strip_prefix("display = ")))
        .map(|display| display.trim_matches('"'))
        .collect();
    assert!(!catalog.is_empty());
    let displays: Vec<&str> = manifest.models.iter().map(|m| m.display.as_str()).collect();
    assert_eq!(displays, catalog);
    let a = &manifest.components["cli"]["x86_64-unknown-linux-gnu-cpu"];
    assert_eq!(a.sha256, larch_fetch::sha256_bytes(&bytes));
    assert_eq!(a.size, bytes.len() as u64);
    assert_eq!(
        a.files,
        vec![
            file("d/file", b"fixture"),
            ("link".into(), unpack::symlink_hash("d/file"), 6)
        ]
    );
    #[cfg(unix)]
    {
        unpack::unpack(&archives.join(name), a, &temp.0.join("out")).unwrap();
    }
    let out = command
        .arg("--upload")
        .env_remove("RELEASES_WRITE")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("native audit reports are required"));
    let committed = fs::read(&dest).unwrap();
    for entries in [
        vec![("../outside", &b"unsafe"[..], b'0', None)],
        vec![
            ("D/a", &b"x"[..], b'0', None),
            ("d/b", &b"y"[..], b'0', None),
        ],
        vec![("\u{0085}control", &b"x"[..], b'0', None)],
        vec![("contiguous", &b"x"[..], b'7', None)],
    ] {
        fs::write(archives.join(name), archive(&entries)).unwrap();
        let mut bad = python("release-manifest.py").unwrap();
        let out = bad
            .args(["--version", "1.2.3", "--archives"])
            .arg(&archives)
            .arg("--output")
            .arg(&dest)
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert_eq!(fs::read(&dest).unwrap(), committed);
    }
    assert!(!temp.0.join("outside").exists());
}

#[test]
fn human_signing_refuses_a_missing_or_unconfirmed_gate_record_before_any_remote_access() {
    let Some(mut command) = python("release-sign") else {
        return;
    };
    let temp = Temp::new();
    let record = temp.0.join("1.2.3.md");
    command
        .args(["publish", "--version", "1.2.3", "--gate-record"])
        .arg(&record)
        .env_remove("RELEASES_WRITE");
    let out = command.output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("gate record is required"));
    fs::write(&record, "version=1.2.3\nchannel=stable\n").unwrap();
    let out = command.output().unwrap();
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr)
            .contains("every gate needs exactly one recorded answer")
    );
}
