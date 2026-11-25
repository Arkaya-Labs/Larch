//! Filesystem and HTTP boundaries, using subprocess environments and local servers only.

use larch_fetch::{
    DownloadError, IndexEntry, download, download_parallel, hex, index_add, index_read, part_path,
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "larch-fetch-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn child_command(mode: &str) -> Command {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", "child", "--nocapture"])
        .env("LARCH_FETCH_CHILD", mode)
        .stdout(Stdio::null());
    cmd
}

#[test]
fn child() {
    match std::env::var("LARCH_FETCH_CHILD").as_deref() {
        Ok("directories") => {
            assert_eq!(
                larch_fetch::data_dir().unwrap(),
                PathBuf::from(std::env::var_os("EXPECT_DATA").unwrap())
            );
            assert_eq!(
                larch_fetch::install_root().unwrap(),
                PathBuf::from(std::env::var_os("EXPECT_ROOT").unwrap())
            );
        }
        Ok("missing") => {
            assert_eq!(
                larch_fetch::data_dir().unwrap_err().to_string(),
                "set --data-dir: no per-user data directory"
            );
        }
        Ok("add") => {
            let data = PathBuf::from(std::env::var_os("INDEX_DATA").unwrap());
            let worker = std::env::var("INDEX_WORKER").unwrap();
            for i in 0..50 {
                index_add(&data, entry(format!("{worker}-{i}"))).unwrap();
            }
        }
        _ => {}
    }
}

#[test]
fn per_user_directories_and_overrides() {
    let dir = Temp::new();
    #[cfg(windows)]
    let (data, root) = (
        dir.0.join("local/larch"),
        dir.0.join("local/Programs/larch"),
    );
    #[cfg(target_os = "macos")]
    let (data, root) = (
        dir.0.join("Library/Application Support/larch"),
        dir.0.join("Library/Application Support/larch-install"),
    );
    #[cfg(all(unix, not(target_os = "macos")))]
    let (data, root) = (dir.0.join("xdg/larch"), dir.0.join("xdg/larch-install"));
    let mut cmd = child_command("directories");
    cmd.env("HOME", &dir.0)
        .env("LOCALAPPDATA", dir.0.join("local"))
        .env("XDG_DATA_HOME", dir.0.join("xdg"))
        .env_remove("LARCH_INSTALL_ROOT")
        .env("EXPECT_DATA", &data)
        .env("EXPECT_ROOT", &root);
    assert!(cmd.status().unwrap().success());
    assert!(
        cmd.env("LARCH_INSTALL_ROOT", dir.0.join("override"))
            .env("EXPECT_ROOT", dir.0.join("override"))
            .status()
            .unwrap()
            .success()
    );
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        cmd.env_remove("LARCH_INSTALL_ROOT")
            .env("EXPECT_DATA", dir.0.join(".local/share/larch"))
            .env("EXPECT_ROOT", dir.0.join(".local/share/larch-install"));
        for xdg in ["relative/path", ""] {
            assert!(cmd.env("XDG_DATA_HOME", xdg).status().unwrap().success());
        }
    }
    assert!(
        child_command("missing")
            .env_remove("HOME")
            .env_remove("LOCALAPPDATA")
            .env_remove("XDG_DATA_HOME")
            .status()
            .unwrap()
            .success()
    );
}

fn entry(file: String) -> IndexEntry {
    IndexEntry {
        file,
        repo: "org/model".into(),
        revision: "revision".into(),
        sha256: "a".repeat(64),
        size: 7,
        alias: None,
    }
}

#[test]
fn concurrent_processes_keep_every_index_entry_and_replace_by_file() {
    let dir = Temp::new();
    let mut a = child_command("add")
        .env("INDEX_DATA", &dir.0)
        .env("INDEX_WORKER", "a")
        .spawn()
        .unwrap();
    let mut b = child_command("add")
        .env("INDEX_DATA", &dir.0)
        .env("INDEX_WORKER", "b")
        .spawn()
        .unwrap();
    assert!(a.wait().unwrap().success());
    assert!(b.wait().unwrap().success());
    let entries = index_read(&dir.0).unwrap();
    assert_eq!(entries.len(), 100);
    for worker in ["a", "b"] {
        for i in 0..50 {
            assert!(entries.contains(&entry(format!("{worker}-{i}"))));
        }
    }
    let mut replacement = entry("a-0".into());
    replacement.size = 99;
    index_add(&dir.0, replacement.clone()).unwrap();
    let entries = index_read(&dir.0).unwrap();
    assert_eq!(entries.len(), 100);
    assert!(entries.contains(&replacement));
    fs::write(dir.0.join("models/index.json"), "broken").unwrap();
    assert!(index_add(&dir.0, entry("new".into())).is_err());
    assert_eq!(
        fs::read(dir.0.join("models/index.json")).unwrap(),
        b"broken"
    );
}

fn request(stream: &TcpStream) -> String {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    let mut reader = BufReader::new(stream);
    let mut request = String::new();
    loop {
        let mut line = String::new();
        assert_ne!(reader.read_line(&mut line).unwrap(), 0);
        request.push_str(&line);
        if line == "\r\n" {
            return request.to_ascii_lowercase();
        }
    }
}

#[test]
fn a_half_download_resumes_with_if_range_and_verifies_all_bytes() {
    let dir = Temp::new();
    let dest = dir.0.join("artifact");
    fs::write(&dest, b"old installation").unwrap();
    let data: Vec<u8> = (0..512 * 1024).map(|i| (i % 251) as u8).collect();
    let hash = hex(&Sha256::digest(&data));
    let server = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/file", server.local_addr().unwrap());
    let served = data.clone();
    let worker = thread::spawn(move || {
        for i in 0..2 {
            let (mut stream, _) = server.accept().unwrap();
            let req = request(&stream);
            let start = if i == 0 { 0 } else { served.len() / 2 };
            assert!(req.contains(&format!("range: bytes={start}-\r\n")));
            if i == 1 {
                assert!(req.contains("if-range: \"v1\"\r\n"));
            }
            write!(stream, "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nETag: \"v1\"\r\nConnection: close\r\n\r\n", served.len()-start, start, served.len()-1, served.len()).unwrap();
            let end = if i == 0 {
                served.len() / 2
            } else {
                served.len()
            };
            stream.write_all(&served[start..end]).unwrap();
        }
    });
    assert!(download(&url, None, &dest, &hash, data.len() as u64, |_, _| true).is_err());
    assert_eq!(fs::read(&dest).unwrap(), b"old installation");
    assert_eq!(fs::read(part_path(&dest)).unwrap(), data[..data.len() / 2]);
    let mut resumed = false;
    download(&url, None, &dest, &hash, data.len() as u64, |have, got| {
        resumed |= have == (data.len() / 2) as u64 && got == 0;
        true
    })
    .unwrap();
    assert!(resumed);
    assert_eq!(fs::read(&dest).unwrap(), data);
    assert!(!part_path(&dest).exists());
    worker.join().unwrap();
}

#[test]
fn cancel_holds_while_the_first_range_trickles_in() {
    let dir = Temp::new();
    let dest = dir.0.join("model.gguf");
    let size: u64 = 32 << 20;
    let server = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/file", server.local_addr().unwrap());
    // The first 8 MiB range at 80 KiB/s: about 100 s if Cancel waited for it.
    thread::spawn(move || {
        let (mut stream, _) = server.accept().unwrap();
        assert!(request(&stream).contains(&format!("range: bytes=0-{}\r\n", (8 << 20) - 1)));
        write!(stream, "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes 0-{}/{size}\r\nConnection: close\r\n\r\n", 8 << 20, (8 << 20) - 1).unwrap();
        while stream.write_all(&[7; 4096]).is_ok() {
            thread::sleep(std::time::Duration::from_millis(50));
        }
    });
    let request = |start: u64, end: Option<u64>, _: Option<&str>| {
        let end = end.map_or(String::new(), |e| e.to_string());
        ureq::get(&url)
            .set("Range", &format!("bytes={start}-{end}"))
            .call()
            .map_err(Box::new)
    };
    let at = std::time::Instant::now();
    let mut moved = false;
    let r = download_parallel(&dest, None, size, request, |_, got| {
        moved |= got > 0;
        !moved
    });
    assert!(matches!(r, Err(DownloadError::Cancelled)), "{r:?}");
    assert!(moved, "progress never saw the first range's bytes");
    assert!(at.elapsed() < std::time::Duration::from_secs(10));
    assert!(!dest.exists());
}

#[test]
fn redirect_to_another_port_never_gets_the_bearer_token() {
    let dir = Temp::new();
    let origin = TcpListener::bind("127.0.0.1:0").unwrap();
    let cdn = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/file", origin.local_addr().unwrap());
    let redirected = format!("http://{}/file", cdn.local_addr().unwrap());
    let a = thread::spawn(move || {
        let (mut stream, _) = origin.accept().unwrap();
        assert!(request(&stream).contains("authorization: bearer test-token\r\n"));
        write!(stream, "HTTP/1.1 302 Found\r\nLocation: {redirected}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
    });
    let b = thread::spawn(move || {
        let (mut stream, _) = cdn.accept().unwrap();
        assert!(!request(&stream).contains("authorization:"));
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\ndata")
            .unwrap();
    });
    download(
        &url,
        Some("test-token"),
        &dir.0.join("artifact"),
        &hex(&Sha256::digest(b"data")),
        4,
        |_, _| true,
    )
    .unwrap();
    a.join().unwrap();
    b.join().unwrap();
}

#[test]
fn bad_sizes_ranges_and_hashes_never_replace_the_destination() {
    for (headers, body, size, hash) in [
        (
            "200 OK\r\nContent-Length: 4",
            "data",
            3,
            hex(&Sha256::digest(b"dat")),
        ),
        (
            "200 OK\r\nContent-Length: 4",
            "data",
            4,
            hex(&Sha256::digest(b"nope")),
        ),
        (
            "206 Partial Content\r\nContent-Length: 4\r\nContent-Range: bytes 4-3/4",
            "data",
            4,
            hex(&Sha256::digest(b"data")),
        ),
    ] {
        let dir = Temp::new();
        let dest = dir.0.join("artifact");
        fs::write(&dest, b"old").unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/file", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            request(&stream);
            write!(
                stream,
                "HTTP/1.1 {headers}\r\nConnection: close\r\n\r\n{body}"
            )
            .unwrap();
        });
        assert!(download(&url, None, &dest, &hash, size, |_, _| true).is_err());
        assert_eq!(fs::read(dest).unwrap(), b"old");
        server.join().unwrap();
    }
}

#[cfg(unix)]
#[test]
fn partial_files_never_follow_symlinks_or_modify_hardlinked_user_files() {
    use std::os::unix::fs::symlink;
    for (etag, hard) in [(false, false), (true, false), (false, true), (true, true)] {
        let dir = Temp::new();
        let dest = dir.0.join("artifact");
        let user = dir.0.join("user");
        fs::write(&user, "keep").unwrap();
        let mut path = part_path(&dest);
        if etag {
            let mut name = path.into_os_string();
            name.push(".etag");
            path = PathBuf::from(name);
        }
        if hard {
            fs::hard_link(&user, &path).unwrap();
        } else {
            symlink(&user, &path).unwrap();
        }
        assert!(
            download(
                "http://127.0.0.1:1/not-contacted",
                None,
                &dest,
                &"a".repeat(64),
                4,
                |_, _| true
            )
            .is_err()
        );
        assert_eq!(fs::read_to_string(user).unwrap(), "keep");
    }
}
