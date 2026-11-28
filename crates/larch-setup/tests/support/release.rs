use larch_setup::manifest::{Artifact, KeySet};
use minisign::KeyPair;
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Cursor, Write},
    net::TcpListener,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
/// A uniquely owned scratch directory removed when its test finishes.
pub struct Temp(
    /// Absolute scratch location.
    pub PathBuf,
);
impl Temp {
    /// Creates a new isolated fixture.
    pub fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "larch-release-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        // Long names, as in a real profile: TEMP can be an 8.3 short path (C:\Users\LONGNA~1),
        // which never matches the canonical install root setup prints. Not verbatim (\\?\), as
        // the real roots are not.
        let p = fs::canonicalize(p).unwrap();
        Self(
            p.to_str()
                .and_then(|s| s.strip_prefix(r"\\?\"))
                .map_or_else(|| p.clone(), PathBuf::from),
        )
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        // A program the test started can still be exiting (the app setup relaunched, which ends
        // once it writes its marker): Windows refuses to delete its file until it has.
        let mut tries = 0;
        while let Err(e) = fs::remove_dir_all(&self.0) {
            tries += 1;
            // A second panic while the test unwinds aborts the binary (0xc0000409 on Windows, #22)
            // and hides the first one's message, so a failed test leaves its directory instead.
            if std::thread::panicking() {
                return;
            }
            assert!(
                e.kind() == std::io::ErrorKind::PermissionDenied && tries < 50,
                "{}: {e}",
                self.0.display()
            );
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}
/// Generates fresh primary and backup keys for this test.
pub fn signing() -> (KeyPair, KeySet) {
    let p = KeyPair::generate_unencrypted_keypair().unwrap();
    let b = KeyPair::generate_unencrypted_keypair().unwrap();
    let k = KeySet {
        version: 1,
        primary: p.pk.to_base64(),
        backup: b.pk.to_base64(),
    };
    (p, k)
}
/// Signs fixture bytes with a version and channel trusted comment.
pub fn sign(p: &KeyPair, bytes: &[u8], v: &str) -> Vec<u8> {
    minisign::sign(
        Some(&p.pk),
        &p.sk,
        Cursor::new(bytes),
        Some(&format!("version={v} channel=stable gates=test")),
        Some("test only"),
    )
    .unwrap()
    .to_string()
    .into_bytes()
}
/// Builds tar headers directly so hostile paths reach the unpack boundary.
pub fn archive(entries: &[(&str, &[u8], u8, Option<&str>)]) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    for (path, bytes, kind, link) in entries {
        let mut h = tar::Header::new_gnu();
        h.set_mode(0o755);
        h.set_size(bytes.len() as u64);
        h.set_entry_type(tar::EntryType::new(*kind));
        h.as_mut_bytes()[..path.len()].copy_from_slice(path.as_bytes());
        if let Some(link) = link {
            h.set_link_name(link).unwrap();
        }
        h.set_cksum();
        builder.append(&h, *bytes).unwrap();
    }
    zstd::stream::encode_all(Cursor::new(builder.into_inner().unwrap()), 1).unwrap()
}
/// Describes a compressed fixture and its complete inventory.
pub fn artifact(url: String, bytes: &[u8], files: Vec<(String, String, u64)>) -> Artifact {
    Artifact {
        url,
        sha256: larch_fetch::sha256_bytes(bytes),
        size: bytes.len() as u64,
        files,
    }
}
/// Hashes a regular fixture file.
pub fn file(path: &str, bytes: &[u8]) -> (String, String, u64) {
    (
        path.into(),
        larch_fetch::sha256_bytes(bytes),
        bytes.len() as u64,
    )
}
/// A loopback HTTP fixture with mutable release contents.
pub struct Server {
    /// Loopback base URL.
    pub base: String,
    /// Responses indexed by exact request path.
    pub files: Arc<Mutex<BTreeMap<String, Vec<u8>>>>,
    /// Observed request headers for protocol assertions.
    pub requests: Arc<Mutex<Vec<String>>>,
    /// Cuts the next archive response halfway through.
    pub cut: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}
impl Server {
    /// Creates a new isolated fixture.
    pub fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let files = Arc::new(Mutex::new(BTreeMap::<String, Vec<u8>>::new()));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let cut = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let (f, r, c, s) = (files.clone(), requests.clone(), cut.clone(), stop.clone());
        let worker = thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(v) => v,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(e) => panic!("{e}"),
                };
                // Windows hands accepted sockets the listener's non-blocking mode.
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(&stream);
                let mut req = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                        break;
                    }
                    req.push_str(&line);
                }
                drop(reader);
                let path = req.split_whitespace().nth(1).unwrap_or("").to_owned();
                let body = f.lock().unwrap().get(&path).cloned();
                let start = req
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("range: bytes=")
                            .and_then(|v| v.trim().strip_suffix('-'))
                            .and_then(|v| v.parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                let etag = if path.ends_with("stable.json") {
                    body.as_ref()
                        .map(|b| larch_fetch::sha256_bytes(b))
                        .unwrap_or_default()
                } else {
                    "test".into()
                };
                let unchanged = req
                    .to_ascii_lowercase()
                    .contains(&format!("if-none-match: \"{etag}\""))
                    && path.ends_with("stable.json");
                r.lock().unwrap().push(req);
                if unchanged {
                    write!(
                        stream,
                        "HTTP/1.1 304 Not Modified\r\nETag: \"{etag}\"\r\nConnection: close\r\n\r\n"
                    )
                    .unwrap();
                    continue;
                }
                if let Some(body) = body {
                    if start > 0 {
                        write!(
                            stream,
                            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/{}\r\n",
                            start,
                            body.len() - 1,
                            body.len()
                        )
                        .unwrap();
                    } else {
                        write!(stream, "HTTP/1.1 200 OK\r\n").unwrap();
                    }
                    write!(
                        stream,
                        "Content-Length: {}\r\nETag: \"{etag}\"\r\nConnection: close\r\n\r\n",
                        body.len() - start
                    )
                    .unwrap();
                    let end = if path.ends_with(".tar.zst") && c.swap(false, Ordering::Relaxed) {
                        body.len() / 2
                    } else {
                        body.len()
                    };
                    stream.write_all(&body[start..end]).unwrap();
                } else {
                    write!(
                        stream,
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )
                    .unwrap();
                }
            }
        });
        Self {
            base,
            files,
            requests,
            cut,
            stop,
            worker: Some(worker),
        }
    }
    /// Replaces one served resource.
    pub fn put(&self, path: &str, bytes: Vec<u8>) {
        self.files.lock().unwrap().insert(path.into(), bytes);
    }
    /// Publishes a signed fixture pointer and manifest.
    pub fn publish(&self, p: &KeyPair, version: &str, manifest: serde_json::Value) {
        let bytes = serde_json::to_vec(&manifest).unwrap();
        let path = format!("/releases/download/v{version}/manifest.json");
        self.put(&path, bytes.clone());
        self.put(&format!("{path}.minisig"), sign(p, &bytes, version));
        let pointer=serde_json::to_vec(&json!({"schema":1,"channel":"stable","version":version,"manifest_url":format!("{}{path}",self.base),"manifest_sha256":larch_fetch::sha256_bytes(&bytes),"setup":{}})).unwrap();
        self.put("/channels/stable.json", pointer.clone());
        self.put("/channels/stable.json.minisig", sign(p, &pointer, version));
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let worker = self.worker.take().unwrap().join();
        // A test that panicked holding `files` poisons it and ends the worker; that is not news.
        if !std::thread::panicking() {
            worker.unwrap();
        }
    }
}
