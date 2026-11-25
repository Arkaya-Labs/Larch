//! The Hugging Face commands against a local HTTP server: no network.

use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering, Ordering::SeqCst},
        mpsc::{Receiver, channel},
    },
    time::{Duration, Instant},
};

use larch_app::{App, Event, contract::*};
use larch_kernels::CPU;
use larch_model::{
    Arch,
    gguf::{Gguf, Value, write},
    tiny,
};
use serde_json::json;
use sha2::{Digest, Sha256};

const DEADLINE: Duration = Duration::from_secs(30);
const AUTH: &str = "This model needs a Hugging Face account: accept its license on huggingface.co and set HF_TOKEN.";
static NEXT: AtomicU64 = AtomicU64::new(0);
/// A projector beside a model turns its images on, except on macOS, where the app reads none
/// (`mmproj_beside`).
const READS_IMAGES: bool = !cfg!(target_os = "macos");

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "larch-hf-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// What the server answers for one path (with its query).
#[derive(Clone, Default)]
struct Route {
    status: u16,
    body: Vec<u8>,
    headers: Vec<(String, String)>,
    /// Answers `Range` with 206; otherwise always the whole body with 200.
    ranges: bool,
    /// A request from byte 0 gets only this many bytes before the connection closes.
    cut: Option<usize>,
    /// Sends 64 KiB every 20 ms.
    slow: bool,
    /// Answers this many requests with 503 first.
    busy: Arc<AtomicUsize>,
}

fn file(body: Vec<u8>) -> Route {
    Route {
        status: 200,
        body,
        ranges: true,
        ..Route::default()
    }
}

fn status(status: u16) -> Route {
    Route {
        status,
        ..Route::default()
    }
}

/// Every request's path and `Range` header.
type Log = Mutex<Vec<(String, Option<String>)>>;

/// A local Hub; `log` has every request's path and `Range` header.
struct Hub {
    base: String,
    routes: Arc<Mutex<HashMap<String, Route>>>,
    log: Arc<Log>,
    /// Requests being answered now, and the most at once.
    live: Arc<(AtomicUsize, AtomicUsize)>,
}

impl Hub {
    fn start() -> Hub {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let hub = Hub {
            base,
            routes: Arc::default(),
            log: Arc::default(),
            live: Arc::default(),
        };
        let (routes, log, live) = (hub.routes.clone(), hub.log.clone(), hub.live.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let (routes, log, live) = (routes.clone(), log.clone(), live.clone());
                std::thread::spawn(move || {
                    let now = live.0.fetch_add(1, SeqCst) + 1;
                    live.1.fetch_max(now, SeqCst);
                    serve(stream.unwrap(), &routes, &log);
                    live.0.fetch_sub(1, SeqCst);
                });
            }
        });
        hub
    }

    fn route(&self, path: &str, route: Route) {
        self.routes.lock().unwrap().insert(path.into(), route);
    }

    /// The repo's tree listing over two pages, at `main` and at [`COMMIT`].
    fn tree(&self, repo: &str, page1: serde_json::Value, page2: serde_json::Value) {
        for revision in ["main", COMMIT] {
            let path = format!("/api/models/{repo}/tree/{revision}?recursive=true");
            let next = format!("{path}&cursor=2");
            let mut first = file(page1.to_string().into_bytes());
            first.headers.push((
                "Link".into(),
                format!("<{}{next}>; rel=\"next\"", self.base),
            ));
            self.route(&path, first);
            self.route(&next, file(page2.to_string().into_bytes()));
        }
    }

    /// `name` of `repo` as the Hub serves it: resolving `main` names [`COMMIT`], and the file is
    /// there at that commit too. Returns the commit's path, which a cache download fetches from.
    fn serve(&self, repo: &str, name: &str, route: Route) -> String {
        let mut main = route.clone();
        main.headers.push(("X-Repo-Commit".into(), COMMIT.into()));
        self.route(&format!("/{repo}/resolve/main/{name}"), main);
        let at = format!("/{repo}/resolve/{COMMIT}/{name}");
        self.route(&at, route);
        at
    }

    fn ranges(&self, path: &str) -> Vec<Option<String>> {
        let log = self.log.lock().unwrap();
        log.iter()
            .filter(|(p, _)| p == path)
            .map(|(_, r)| r.clone())
            .collect()
    }
}

fn serve(mut stream: TcpStream, routes: &Mutex<HashMap<String, Route>>, log: &Log) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let path = line.split(' ').nth(1).unwrap_or_default().to_owned();
    let mut range = None;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).unwrap() == 0 || h == "\r\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':')
            && k.eq_ignore_ascii_case("range")
        {
            range = Some(v.trim().to_owned());
        }
    }
    log.lock().unwrap().push((path.clone(), range.clone()));
    let route = routes
        .lock()
        .unwrap()
        .get(&path)
        .cloned()
        .unwrap_or(status(404));
    let len = route.body.len();
    let busy = route
        .busy
        .fetch_update(SeqCst, SeqCst, |n| n.checked_sub(1))
        .is_ok();
    let route = if busy { status(503) } else { route };
    let (mut code, mut body, mut headers) = (route.status, &route.body[..], route.headers.clone());
    let mut start = 0;
    if let Some(r) = range.as_deref().and_then(|r| r.strip_prefix("bytes="))
        && route.ranges
        && code == 200
    {
        let (a, b) = r.split_once('-').unwrap();
        start = a.parse::<usize>().unwrap();
        let end = b.parse::<usize>().map_or(len, |b| (b + 1).min(len));
        code = 206;
        body = &route.body[start..end];
        headers.push((
            "Content-Range".into(),
            format!("bytes {start}-{}/{len}", end - 1),
        ));
    }
    let mut head = format!(
        "HTTP/1.1 {code} X\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        head += &format!("{k}: {v}\r\n");
    }
    head += "\r\n";
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    if let Some(cut) = route.cut.filter(|_| start == 0) {
        body = &body[..cut];
    }
    // The client may hang up (a header read, a cancel); that is not an error here.
    for chunk in body.chunks(64 << 10) {
        if stream.write_all(chunk).is_err() {
            return;
        }
        if route.slow {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

fn app(hub: &Hub) -> (App, Receiver<Event>, Temp) {
    let dir = Temp::new();
    let (tx, rx) = channel();
    let app = App::new(dir.0.clone(), CPU, move |e| {
        let _ = tx.send(e);
    })
    .unwrap();
    app.set_hf_endpoint(&hub.base);
    // Only this test's folders: not the models other apps keep on the machine running the tests.
    app.set_model_dirs(Vec::new());
    (app, rx, dir)
}

/// A tree entry, an LFS file when `sha` is given.
fn entry(path: &str, body: &[u8], sha: Option<String>) -> serde_json::Value {
    match sha {
        Some(oid) => json!({"type": "file", "path": path, "size": 134, "oid": "x",
            "lfs": {"oid": oid, "size": body.len(), "pointerSize": 134}}),
        None => json!({"type": "file", "path": path, "size": body.len(), "oid": "x"}),
    }
}

fn sha(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The next download event with id `id`, skipping progress.
fn end(rx: &Receiver<Event>, id: &str) -> Event {
    let deadline = Instant::now() + DEADLINE;
    loop {
        match rx.recv_timeout(deadline - Instant::now()).unwrap() {
            Event::DownloadDone(d) if d.id == id => return Event::DownloadDone(d),
            Event::DownloadFailed(f) if f.id == id => return Event::DownloadFailed(f),
            _ => {}
        }
    }
}

/// Deterministic bytes.
fn bytes(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761).to_le_bytes()[1] ^ seed)
        .collect()
}

#[test]
fn files_are_paged_grouped_and_sorted() {
    let hub = Hub::start();
    let (app, _rx, dir) = app(&hub);
    let (a, b, c, d) = (bytes(300, 1), bytes(100, 2), bytes(120, 3), bytes(50, 4));
    let pj = tiny::Projector::new(64, 9).gguf();
    let p = pj.len() as u64;
    hub.tree(
        "org/model",
        json!([
            entry("model-Q8_0.gguf", &a, Some(sha(&a))),
            entry("README.md", b"hi", None),
            {"type": "directory", "path": "sub", "size": 0, "oid": "x"},
            entry("sub/model-Q4_K_M-00002-of-00002.gguf", &c, Some(sha(&c))),
            entry("mmproj-model-f16.gguf", &pj, Some(sha(&pj))),
        ]),
        json!([
            entry("sub/model-Q4_K_M-00001-of-00002.gguf", &b, Some(sha(&b))),
            entry("model-F16.gguf", &d, None),
            // A split set with a part missing is not offered.
            entry("model-Q2_K-00001-of-00002.gguf", &b, None),
        ]),
    );
    hub.route(
        "/org/model/resolve/main/mmproj-model-f16.gguf",
        file(pj.clone()),
    );
    fs::write(dir.0.join("models/model-F16.gguf"), &d).unwrap();
    // The repo's vision projector (one the app loads, by its header) goes with every quant of the
    // model, and counts in its size; a model is downloaded only with it.
    let files = app.hf_files("org/model").unwrap();
    assert!(!files[0].downloaded);
    fs::write(dir.0.join("models/mmproj-model-f16.gguf"), &pj).unwrap();
    let files = app.hf_files("org/model").unwrap();
    let split = [
        "sub/model-Q4_K_M-00001-of-00002.gguf".to_owned(),
        "sub/model-Q4_K_M-00002-of-00002.gguf".to_owned(),
    ];
    let mmproj = Some("mmproj-model-f16.gguf".to_owned());
    assert_eq!(
        files,
        [
            HfFile {
                file: "model-F16.gguf".into(),
                parts: vec!["model-F16.gguf".into()],
                size_bytes: 50 + p,
                quant: "F16".into(),
                downloaded: true,
                mmproj: mmproj.clone(),
            },
            HfFile {
                file: split[0].clone(),
                parts: split.to_vec(),
                size_bytes: 220 + p,
                quant: "Q4_K_M".into(),
                downloaded: false,
                mmproj: mmproj.clone(),
            },
            HfFile {
                file: "model-Q8_0.gguf".into(),
                parts: vec!["model-Q8_0.gguf".into()],
                size_bytes: 300 + p,
                quant: "Q8_0".into(),
                downloaded: false,
                mmproj,
            },
        ]
    );
    assert_eq!(
        app.hf_files("org/none").unwrap_err(),
        "Hugging Face has no such model: org/none."
    );
    assert!(app.hf_files("../etc").is_err());
}

/// The tiny model with a header over one 4 MiB range request (a big tokenizer-like string).
fn big_header_gguf() -> Vec<u8> {
    let bytes = tiny::gguf(Arch::Llama, 3);
    let g = Gguf::parse(&bytes).unwrap();
    let tensors: Vec<_> = g
        .tensors
        .iter()
        .map(|t| {
            let at = (g.data_offset + t.offset) as usize;
            let data = bytes[at..at + t.bytes as usize].to_vec();
            (t.name.clone(), t.dims.clone(), t.ggml_type, data)
        })
        .collect();
    let mut kv: Vec<_> = g.kv.into_iter().collect();
    kv.push(("test.padding".into(), Value::Str("x".repeat(5 << 20))));
    write(&kv, &tensors)
}

#[test]
fn inspect_reads_the_header_with_ranges() {
    let hub = Hub::start();
    let (app, _rx, dir) = app(&hub);
    let model = big_header_gguf();
    hub.tree(
        "org/model",
        json!([entry("model-Q4_K_M.gguf", &model, Some(sha(&model)))]),
        json!([]),
    );
    hub.route(
        "/org/model/resolve/main/model-Q4_K_M.gguf",
        file(model.clone()),
    );
    let m = app.hf_inspect("org/model", "model-Q4_K_M.gguf").unwrap();

    // The same file read locally, as list_models reads it.
    fs::write(dir.0.join("models/local.gguf"), &model).unwrap();
    let local = &app.list_models().unwrap()[0];
    let g = Gguf::parse(&model).unwrap();
    assert_eq!(
        m,
        RemoteModel {
            arch: "llama".into(),
            supported: true,
            quant: local.quant.clone(),
            weights_bytes: g.tensors.iter().map(|t| t.bytes).sum(),
            params: g
                .tensors
                .iter()
                .map(|t| t.dims.iter().product::<u64>())
                .sum(),
            kv_bytes_per_token: Some(
                2 * u64::from(tiny::LAYERS) * tiny::KV_HEADS * tiny::HEAD_DIM * 2
            ),
            context_length: Some(tiny::CTX.into()),
            expert_bytes: 0,
            unloadable: vec![],
        }
    );
    assert!(!m.quant.is_empty() && m.weights_bytes > 0);
    // Two 4 MiB ranges, then cached.
    let path = "/org/model/resolve/main/model-Q4_K_M.gguf";
    assert_eq!(
        hub.ranges(path),
        [
            Some("bytes=0-4194303".to_owned()),
            Some(format!("bytes=4194304-{}", (8 << 20).min(model.len()) - 1))
        ]
    );
    assert_eq!(app.hf_inspect("org/model", "model-Q4_K_M.gguf").unwrap(), m);
    assert_eq!(hub.ranges(path).len(), 2);
}

#[test]
fn download_verifies_and_lists_the_model() {
    let hub = Hub::start();
    let (app, rx, dir) = app(&hub);
    let model = tiny::gguf(Arch::Llama, 3);
    hub.tree(
        "org/model",
        json!([entry("model-Q4_K_M.gguf", &model, Some(sha(&model)))]),
        json!([]),
    );
    // resolve redirects to a CDN, as the Hub does.
    let mut redirect = status(302);
    redirect.headers.push((
        "Location".into(),
        format!("{}/cdn/model-Q4_K_M.gguf", hub.base),
    ));
    hub.route("/org/model/resolve/main/model-Q4_K_M.gguf", redirect);
    hub.route("/cdn/model-Q4_K_M.gguf", file(model.clone()));
    app.hf_files("org/model").unwrap();
    let id = app
        .download_model("org/model", "model-Q4_K_M.gguf", None)
        .unwrap();
    let path = dir.0.join("models").join("model-Q4_K_M.gguf");
    assert_eq!(
        end(&rx, &id),
        Event::DownloadDone(DownloadDone {
            id: id.clone(),
            repo: "org/model".into(),
            file: "model-Q4_K_M.gguf".into(),
            path: path.to_string_lossy().into(),
            note: None,
        })
    );
    assert_eq!(fs::read(&path).unwrap(), model);
    assert!(!dir.0.join("models/model-Q4_K_M.gguf.part").exists());
    assert_eq!(
        hub.ranges("/cdn/model-Q4_K_M.gguf"),
        [Some("bytes=0-".to_owned())]
    );
    assert_eq!(
        app.list_downloads().unwrap(),
        [Download {
            id,
            repo: "org/model".into(),
            file: "model-Q4_K_M.gguf".into(),
            received_bytes: model.len() as u64,
            total_bytes: model.len() as u64,
            state: "done".into(),
        }]
    );
    let models = app.list_models().unwrap();
    assert_eq!(models.len(), 1);
    assert!(models[0].supported);
    assert!(app.hf_files("org/model").unwrap()[0].downloaded);
    assert_eq!(
        app.download_model("org/model", "model-Q4_K_M.gguf", None)
            .unwrap_err(),
        "model-Q4_K_M.gguf is already downloaded."
    );
}

/// A big file comes as 8 MiB ranges over several connections, resolve asked once and the rest
/// straight from the CDN, written in order after the part already there.
#[test]
fn a_big_download_fetches_pieces_in_parallel_from_the_cdn() {
    let hub = Hub::start();
    let (app, rx, dir) = app(&hub);
    let a = bytes(40 << 20, 3);
    hub.tree(
        "org/model",
        json!([entry("model-Q8_0.gguf", &a, Some(sha(&a)))]),
        json!([]),
    );
    let mut redirect = status(302);
    redirect.headers.push((
        "Location".into(),
        format!("{}/cdn/model-Q8_0.gguf", hub.base),
    ));
    hub.route("/org/model/resolve/main/model-Q8_0.gguf", redirect);
    let mut cdn = file(a.clone());
    cdn.headers.push(("ETag".into(), "\"v1\"".into()));
    // 3.2 MB/s a connection: the pieces overlap.
    cdn.slow = true;
    hub.route("/cdn/model-Q8_0.gguf", cdn);
    let have = 3 << 20;
    fs::write(dir.0.join("models/model-Q8_0.gguf.part"), &a[..have]).unwrap();
    app.hf_files("org/model").unwrap();
    let id = app
        .download_model("org/model", "model-Q8_0.gguf", None)
        .unwrap();
    assert!(matches!(end(&rx, &id), Event::DownloadDone(_)));
    assert_eq!(fs::read(dir.0.join("models/model-Q8_0.gguf")).unwrap(), a);
    assert_eq!(
        hub.ranges("/org/model/resolve/main/model-Q8_0.gguf").len(),
        1
    );
    let mut got = hub.ranges("/cdn/model-Q8_0.gguf");
    got.sort_by_key(|r| {
        let r = r.as_deref().unwrap();
        r["bytes=".len()..r.find('-').unwrap()]
            .parse::<usize>()
            .unwrap()
    });
    let piece = 8 << 20;
    let want: Vec<_> = (have..a.len())
        .step_by(piece)
        .map(|s| Some(format!("bytes={s}-{}", (s + piece).min(a.len()) - 1)))
        .collect();
    assert_eq!(got, want);
    assert!(hub.live.1.load(SeqCst) >= 4, "{:?}", hub.live);
}

#[test]
fn a_vision_model_downloads_with_its_projector() {
    let hub = Hub::start();
    let (app, rx, dir) = app(&hub);
    let model = tiny::gguf(Arch::Llama, 3);
    let projector = tiny::Projector::new(64, 9).gguf();
    // ggml-org's naming: a projector whose name does not start as the model's; a bf16 one the loader
    // would refuse is passed over.
    hub.tree(
        "org/vl",
        json!([
            entry("gemma-3-4b-it-Q4_K_M.gguf", &model, Some(sha(&model))),
            entry("mmproj-model-f16.gguf", &projector, Some(sha(&projector))),
            entry("mmproj-model-bf16.gguf", &projector, Some(sha(&projector))),
        ]),
        json!([]),
    );
    hub.route(
        "/org/vl/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
        file(model.clone()),
    );
    hub.route(
        "/org/vl/resolve/main/mmproj-model-f16.gguf",
        file(projector.clone()),
    );
    let files = app.hf_files("org/vl").unwrap();
    assert_eq!(files[0].mmproj.as_deref(), Some("mmproj-model-f16.gguf"));
    assert_eq!(files[0].size_bytes, (model.len() + projector.len()) as u64);
    let id = app
        .download_model("org/vl", "gemma-3-4b-it-Q4_K_M.gguf", None)
        .unwrap();
    assert!(matches!(end(&rx, &id), Event::DownloadDone(_)));
    // Named for its model, so the app finds it beside the model in a shared folder.
    let local = dir.0.join("models/mmproj-gemma-3-4b-it-f16.gguf");
    assert_eq!(fs::read(&local).unwrap(), projector);
    assert!(app.hf_files("org/vl").unwrap()[0].downloaded);
    let models = app.list_models().unwrap();
    assert_eq!(models.len(), 1, "a projector is not a model");
    assert_eq!(models[0].vision, READS_IMAGES);
}

/// A repo's projector of a kind the app does not load (its header is no InternVL projector's) is
/// not offered: the model downloads alone and is not said to read images.
#[test]
fn a_projector_the_app_does_not_load_is_not_offered() {
    let hub = Hub::start();
    let (app, _rx, _dir) = app(&hub);
    let model = tiny::gguf(Arch::Llama, 3);
    let other = tiny::gguf(Arch::Llama, 5);
    hub.tree(
        "org/other",
        json!([
            entry("model-Q4_K_M.gguf", &model, Some(sha(&model))),
            entry("mmproj-model-f16.gguf", &other, Some(sha(&other))),
        ]),
        json!([]),
    );
    hub.route(
        "/org/other/resolve/main/mmproj-model-f16.gguf",
        file(other.clone()),
    );
    let files = app.hf_files("org/other").unwrap();
    assert_eq!(files[0].mmproj, None);
    assert_eq!(files[0].size_bytes, model.len() as u64);
}

#[test]
fn a_broken_download_keeps_its_part_and_resumes() {
    let hub = Hub::start();
    let (app, rx, dir) = app(&hub);
    let (a, b) = (bytes(3 << 20, 7), bytes(1 << 20, 9));
    hub.tree(
        "org/model",
        json!([
            entry("model-Q8_0.gguf", &a, Some(sha(&a))),
            entry("model-Q6_K.gguf", &b, Some(sha(&b))),
        ]),
        json!([]),
    );
    let mut cut = file(a.clone());
    cut.cut = Some(1 << 20);
    hub.route("/org/model/resolve/main/model-Q8_0.gguf", cut);
    let id = app
        .download_model("org/model", "model-Q8_0.gguf", None)
        .unwrap();
    let Event::DownloadFailed(f) = end(&rx, &id) else {
        panic!("not failed")
    };
    assert!(
        !f.cancelled && f.message.contains("Try again"),
        "{}",
        f.message
    );
    let part = dir.0.join("models/model-Q8_0.gguf.part");
    assert_eq!(fs::read(&part).unwrap(), a[..1 << 20]);
    assert_eq!(app.list_downloads().unwrap()[0].state, "failed");

    // The retry asks for the rest only, and the hash covers the bytes already there.
    let id = app
        .download_model("org/model", "model-Q8_0.gguf", None)
        .unwrap();
    assert!(matches!(end(&rx, &id), Event::DownloadDone(_)));
    assert_eq!(
        hub.ranges("/org/model/resolve/main/model-Q8_0.gguf"),
        [
            Some("bytes=0-".to_owned()),
            Some("bytes=1048576-".to_owned())
        ]
    );
    assert_eq!(fs::read(dir.0.join("models/model-Q8_0.gguf")).unwrap(), a);
    assert!(!part.exists());

    // A server that ignores the range sends the whole file: the part starts over.
    fs::write(dir.0.join("models/model-Q6_K.gguf.part"), bytes(1000, 0)).unwrap();
    let mut whole = file(b.clone());
    whole.ranges = false;
    hub.route("/org/model/resolve/main/model-Q6_K.gguf", whole);
    let id = app
        .download_model("org/model", "model-Q6_K.gguf", None)
        .unwrap();
    assert!(matches!(end(&rx, &id), Event::DownloadDone(_)));
    assert_eq!(fs::read(dir.0.join("models/model-Q6_K.gguf")).unwrap(), b);
}

/// A download lands in the library asked for, counts as downloaded in every library, and an unknown
/// library is refused before anything is fetched.
#[test]
fn download_goes_to_the_chosen_library() {
    let hub = Hub::start();
    let (app, rx, dir) = app(&hub);
    let model = tiny::gguf(Arch::Llama, 3);
    hub.tree(
        "org/model",
        json!([entry("model-Q4_K_M.gguf", &model, Some(sha(&model)))]),
        json!([]),
    );
    hub.route(
        "/org/model/resolve/main/model-Q4_K_M.gguf",
        file(model.clone()),
    );
    let lib = dir.0.join("D").join("Models");
    let libs = app.add_library(lib.to_str().unwrap()).unwrap();
    assert!(
        app.download_model("org/model", "model-Q4_K_M.gguf", Some("/not/a/library"))
            .is_err()
    );
    let id = app
        .download_model("org/model", "model-Q4_K_M.gguf", Some(&libs[1]))
        .unwrap();
    assert!(matches!(end(&rx, &id), Event::DownloadDone(_)));
    assert_eq!(fs::read(lib.join("model-Q4_K_M.gguf")).unwrap(), model);
    assert!(!dir.0.join("models/model-Q4_K_M.gguf").exists());
    assert!(app.hf_files("org/model").unwrap()[0].downloaded);
    assert!(
        app.download_model("org/model", "model-Q4_K_M.gguf", None)
            .is_err()
    );
}

const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

/// An app whose Hugging Face cache is a temporary folder; returns the cache's folder for org/model.
fn cache_app(hub: &Hub) -> (App, Receiver<Event>, Temp, PathBuf) {
    let (app, rx, dir) = app(hub);
    let cache = dir.0.join("hf-hub");
    app.set_model_dirs(vec![("Hugging Face".into(), cache.clone(), 4)]);
    (app, rx, dir, cache.join("models--org--model"))
}

fn done(rx: &Receiver<Event>, id: &str) -> DownloadDone {
    match end(rx, id) {
        Event::DownloadDone(d) => d,
        e => panic!("{e:?}"),
    }
}

/// With no library asked for, a download goes into the Hugging Face cache as huggingface_hub lays
/// it out (a blob named by its hash, linked from the snapshot of the commit resolving `main` named,
/// fetched at that commit, then `refs/main`), lists from there, counts as downloaded, and deletes
/// with its blob.
#[test]
fn a_download_goes_into_the_hugging_face_cache() {
    let hub = Hub::start();
    let (app, rx, dir, repo) = cache_app(&hub);
    let model = tiny::gguf(Arch::Llama, 3);
    let hash = sha(&model);
    let name = "Q4_K_M/model-Q4_K_M.gguf";
    hub.tree(
        "org/model",
        json!([entry(name, &model, Some(hash.clone()))]),
        json!([]),
    );
    let at = hub.serve("org/model", name, file(model.clone()));
    let id = app.download_model("org/model", name, None).unwrap();
    let path = repo.join("snapshots").join(COMMIT).join(name);
    let d = done(&rx, &id);
    assert_eq!((d.path.as_str(), d.note), (&*path.to_string_lossy(), None));
    assert_eq!(fs::read(&path).unwrap(), model);
    #[cfg(unix)]
    assert_eq!(
        fs::read_link(&path).unwrap(),
        PathBuf::from(format!("../../../blobs/{hash}"))
    );
    assert_eq!(fs::read_to_string(repo.join("refs/main")).unwrap(), COMMIT);
    // huggingface_hub's lock, and nothing else left beside the blob.
    let locks = repo.with_file_name(".locks").join("models--org--model");
    assert!(locks.join(format!("{hash}.lock")).exists());
    assert_eq!(fs::read_dir(repo.join("blobs")).unwrap().count(), 1);
    assert_eq!(hub.ranges(&at), [Some("bytes=0-".to_owned())]);
    assert!(!dir.0.join("models/model-Q4_K_M.gguf").exists());
    let models = app.list_models().unwrap();
    assert_eq!(
        models
            .iter()
            .map(|m| (m.source.as_str(), m.path.as_str()))
            .collect::<Vec<_>>(),
        [("Hugging Face", d.path.as_str())]
    );
    assert!(app.hf_files("org/model").unwrap()[0].downloaded);
    assert_eq!(
        app.download_model("org/model", name, None).unwrap_err(),
        "Q4_K_M/model-Q4_K_M.gguf is already downloaded."
    );
    // Nor into a library: the model is on this machine.
    let id = app.download_model("org/model", name, Some(&app.libraries()[0]));
    assert!(id.unwrap_err().contains("already downloaded"));
    app.delete_model(&d.path).unwrap();
    assert!(fs::symlink_metadata(&path).is_err());
    assert!(!repo.join("blobs").join(&hash).exists());
    assert!(app.list_models().unwrap().is_empty());
    assert!(!app.hf_files("org/model").unwrap()[0].downloaded);
}

/// A model another tool put in the cache is not downloaded again; a blob already there (a download
/// stopped before its link) is hashed and only linked; one of the right size but damaged is
/// fetched again; a link to an older revision's blob is not the file.
#[test]
fn what_the_cache_has_is_not_downloaded_again() {
    let hub = Hub::start();
    let (app, rx, _dir, repo) = cache_app(&hub);
    let (a, b, c) = (bytes(1000, 1), bytes(2000, 2), bytes(3000, 3));
    hub.tree(
        "org/model",
        json!([
            entry("model-Q8_0.gguf", &a, Some(sha(&a))),
            entry("model-Q4_0.gguf", &b, Some(sha(&b))),
            entry("model-Q2_K.gguf", &c, Some(sha(&c))),
        ]),
        json!([]),
    );
    let older = repo.join("snapshots/fedcba9876543210fedcba9876543210fedcba98");
    fs::create_dir_all(&older).unwrap();
    fs::write(older.join("model-Q8_0.gguf"), &a).unwrap();
    let blobs = repo.join("blobs");
    fs::create_dir_all(&blobs).unwrap();
    fs::write(blobs.join(sha(&b)), &b).unwrap();
    fs::write(blobs.join(sha(&c)), bytes(3000, 9)).unwrap();
    #[cfg(unix)]
    {
        // The older revision's Q2_K: the same size, another blob.
        fs::write(blobs.join(sha(b"old")), bytes(3000, 8)).unwrap();
        let old = format!("../../blobs/{}", sha(b"old"));
        std::os::unix::fs::symlink(old, older.join("model-Q2_K.gguf")).unwrap();
    }
    let files = app.hf_files("org/model").unwrap();
    let got: Vec<_> = files
        .iter()
        .map(|f| (f.file.as_str(), f.downloaded))
        .collect();
    assert_eq!(
        got,
        [
            ("model-Q8_0.gguf", true),
            ("model-Q4_0.gguf", false),
            ("model-Q2_K.gguf", false)
        ]
    );
    assert!(
        app.download_model("org/model", "model-Q8_0.gguf", None)
            .unwrap_err()
            .contains("already downloaded")
    );
    let snapshot = repo.join("snapshots").join(COMMIT);
    for (name, body, fetched) in [
        ("model-Q4_0.gguf", &b, false),
        ("model-Q2_K.gguf", &c, true),
    ] {
        let at = hub.serve("org/model", name, file(body.clone()));
        let id = app.download_model("org/model", name, None).unwrap();
        done(&rx, &id);
        assert_eq!(&fs::read(snapshot.join(name)).unwrap(), body, "{name}");
        assert_eq!(!hub.ranges(&at).is_empty(), fetched, "{name}");
    }
}

/// Cancel deletes a split model's finished parts from the cache, links and blobs, and the part
/// being fetched; `refs/main` is not moved for a download that did not finish.
#[test]
fn cancel_leaves_nothing_in_the_cache() {
    let hub = Hub::start();
    let (app, rx, _dir, repo) = cache_app(&hub);
    let (a, b) = (bytes(1000, 1), bytes(8 << 20, 2));
    let (one, two) = ("m-00001-of-00002.gguf", "m-00002-of-00002.gguf");
    hub.tree(
        "org/model",
        json!([entry(one, &a, Some(sha(&a))), entry(two, &b, Some(sha(&b)))]),
        json!([]),
    );
    hub.serve("org/model", one, file(a.clone()));
    let mut slow = file(b);
    slow.slow = true;
    hub.serve("org/model", two, slow);
    let id = app.download_model("org/model", one, None).unwrap();
    let deadline = Instant::now() + DEADLINE;
    // Past the first part: it is linked, and the second is under way.
    while !matches!(rx.recv_timeout(deadline - Instant::now()).unwrap(),
        Event::DownloadProgress(p) if p.received_bytes > a.len() as u64)
    {}
    assert!(repo.join("snapshots").join(COMMIT).join(one).exists());
    app.cancel_download(&id).unwrap();
    let Event::DownloadFailed(f) = end(&rx, &id) else {
        panic!("not failed")
    };
    assert!(f.cancelled);
    let left = |d: &str| fs::read_dir(repo.join(d)).unwrap().count();
    assert_eq!(
        (left("blobs"), left(&format!("snapshots/{COMMIT}"))),
        (0, 0)
    );
    assert!(!repo.join("refs/main").exists());
}

/// A blob another download holds the lock of (huggingface_hub's, or another Larch's) is waited
/// for, and a cancel meanwhile leaves that download's partial alone.
#[test]
fn a_locked_blob_is_waited_for_and_its_partial_kept() {
    let hub = Hub::start();
    let (app, rx, _dir, repo) = cache_app(&hub);
    let a = bytes(1000, 1);
    hub.tree(
        "org/model",
        json!([entry("model-Q8_0.gguf", &a, Some(sha(&a)))]),
        json!([]),
    );
    hub.serve("org/model", "model-Q8_0.gguf", file(a.clone()));
    let locks = repo.with_file_name(".locks").join("models--org--model");
    fs::create_dir_all(&locks).unwrap();
    let held = fs::File::create(locks.join(format!("{}.lock", sha(&a)))).unwrap();
    held.lock().unwrap();
    let partial = repo.join("blobs").join(format!("{}.incomplete", sha(&a)));
    fs::create_dir_all(partial.parent().unwrap()).unwrap();
    fs::write(&partial, &a[..100]).unwrap();
    let id = app
        .download_model("org/model", "model-Q8_0.gguf", None)
        .unwrap();
    std::thread::sleep(Duration::from_millis(500));
    app.cancel_download(&id).unwrap();
    let Event::DownloadFailed(f) = end(&rx, &id) else {
        panic!("not failed")
    };
    assert!(f.cancelled);
    assert_eq!(fs::read(&partial).unwrap(), &a[..100]);
}

/// A cache Larch cannot write (a file where its folder should be; on Linux often root's, from
/// `sudo pip`), or a Hub that names no commit (a mirror), sends the download to Larch's own
/// library, and its end says why.
#[test]
fn a_download_the_cache_cannot_take_goes_to_the_library() {
    let hub = Hub::start();
    let a = bytes(1000, 1);
    hub.tree(
        "org/model",
        json!([entry("model-Q8_0.gguf", &a, Some(sha(&a)))]),
        json!([]),
    );
    hub.route("/org/model/resolve/main/model-Q8_0.gguf", file(a.clone()));
    let (app, rx, dir) = app(&hub);
    fs::write(dir.0.join("hf-hub"), b"").unwrap();
    app.set_model_dirs(vec![("Hugging Face".into(), dir.0.join("hf-hub"), 4)]);
    let id = app
        .download_model("org/model", "model-Q8_0.gguf", None)
        .unwrap();
    let d = done(&rx, &id);
    assert_eq!(
        d.path,
        dir.0
            .join("models")
            .join("model-Q8_0.gguf")
            .to_string_lossy()
    );
    assert!(
        d.note
            .unwrap()
            .starts_with("Larch cannot write to the Hugging Face cache")
    );
    let (app, rx, dir, repo) = cache_app(&hub);
    let id = app
        .download_model("org/model", "model-Q8_0.gguf", None)
        .unwrap();
    let d = done(&rx, &id);
    assert_eq!(
        d.path,
        dir.0
            .join("models")
            .join("model-Q8_0.gguf")
            .to_string_lossy()
    );
    assert!(d.note.unwrap().contains("named no commit"));
    assert!(!repo.exists());
}

/// In the cache a projector keeps its repo name (ggml-org's `mmproj-model-f16.gguf`), and the
/// model beside it still reads images.
#[test]
fn a_vision_model_in_the_cache_finds_its_projector() {
    let hub = Hub::start();
    let (app, rx, _dir, repo) = cache_app(&hub);
    let model = tiny::gguf(Arch::Llama, 3);
    let projector = tiny::Projector::new(64, 9).gguf();
    hub.tree(
        "org/model",
        json!([
            entry("gemma-3-4b-it-Q4_K_M.gguf", &model, Some(sha(&model))),
            entry("mmproj-model-f16.gguf", &projector, Some(sha(&projector))),
        ]),
        json!([]),
    );
    hub.serve(
        "org/model",
        "gemma-3-4b-it-Q4_K_M.gguf",
        file(model.clone()),
    );
    hub.serve(
        "org/model",
        "mmproj-model-f16.gguf",
        file(projector.clone()),
    );
    let id = app
        .download_model("org/model", "gemma-3-4b-it-Q4_K_M.gguf", None)
        .unwrap();
    done(&rx, &id);
    let snapshot = repo.join("snapshots").join(COMMIT);
    assert_eq!(
        fs::read(snapshot.join("mmproj-model-f16.gguf")).unwrap(),
        projector
    );
    let models = app.list_models().unwrap();
    assert_eq!(models.len(), 1, "a projector is not a model");
    assert_eq!(models[0].vision, READS_IMAGES);
    // Deleting the model takes its projector, links and blobs.
    app.delete_model(&models[0].path).unwrap();
    assert_eq!(fs::read_dir(repo.join("blobs")).unwrap().count(), 0);
}

/// A split quant sits in a folder of the snapshot (unsloth's `UD-Q4_K_XL/`) and the repo's
/// projector at the snapshot's root: the model still reads images, and deleting it takes the
/// projector too.
#[test]
fn a_split_vision_model_in_a_snapshot_folder_finds_the_root_projector() {
    let hub = Hub::start();
    let (app, rx, _dir, repo) = cache_app(&hub);
    let (one, two) = (tiny::gguf(Arch::Llama, 3), bytes(1000, 4));
    let projector = tiny::Projector::new(64, 9).gguf();
    let first = "Q4_K_M/gemma-3-4b-it-Q4_K_M-00001-of-00002.gguf";
    let second = "Q4_K_M/gemma-3-4b-it-Q4_K_M-00002-of-00002.gguf";
    hub.tree(
        "org/model",
        json!([
            entry(first, &one, Some(sha(&one))),
            entry(second, &two, Some(sha(&two))),
            entry("mmproj-F16.gguf", &projector, Some(sha(&projector))),
        ]),
        json!([]),
    );
    hub.serve("org/model", first, file(one));
    hub.serve("org/model", second, file(two));
    hub.serve("org/model", "mmproj-F16.gguf", file(projector));
    assert_eq!(
        app.hf_files("org/model").unwrap()[0].mmproj.as_deref(),
        Some("mmproj-F16.gguf")
    );
    let id = app.download_model("org/model", first, None).unwrap();
    done(&rx, &id);
    let models = app.list_models().unwrap();
    assert_eq!(models.len(), 1, "{models:?}");
    assert_eq!(models[0].vision, READS_IMAGES);
    app.delete_model(&models[0].path).unwrap();
    assert_eq!(fs::read_dir(repo.join("blobs")).unwrap().count(), 0);
}

#[test]
fn a_checksum_mismatch_deletes_the_part() {
    let hub = Hub::start();
    let (app, rx, dir) = app(&hub);
    let a = bytes(1 << 20, 5);
    hub.tree(
        "org/model",
        json!([entry("model-Q8_0.gguf", &a, Some(sha(b"other")))]),
        json!([]),
    );
    hub.route("/org/model/resolve/main/model-Q8_0.gguf", file(a));
    let id = app
        .download_model("org/model", "model-Q8_0.gguf", None)
        .unwrap();
    let Event::DownloadFailed(f) = end(&rx, &id) else {
        panic!("not failed")
    };
    assert_eq!(
        f.message,
        "org/model/model-Q8_0.gguf does not match its checksum on Hugging Face; the download was deleted. Try again."
    );
    assert!(!f.cancelled);
    assert!(!dir.0.join("models/model-Q8_0.gguf.part").exists());
    assert!(!dir.0.join("models/model-Q8_0.gguf").exists());
}

#[test]
fn cancel_deletes_the_part() {
    let hub = Hub::start();
    let (app, rx, dir) = app(&hub);
    let a = bytes(8 << 20, 6);
    hub.tree(
        "org/model",
        json!([entry("model-Q8_0.gguf", &a, Some(sha(&a)))]),
        json!([]),
    );
    let mut slow = file(a);
    slow.slow = true;
    hub.route("/org/model/resolve/main/model-Q8_0.gguf", slow);
    let id = app
        .download_model("org/model", "model-Q8_0.gguf", None)
        .unwrap();
    assert_eq!(
        app.download_model("org/model", "model-Q8_0.gguf", None)
            .unwrap_err(),
        "model-Q8_0.gguf is already downloading."
    );
    let deadline = Instant::now() + DEADLINE;
    let progress = || loop {
        if let Event::DownloadProgress(p) = rx.recv_timeout(deadline - Instant::now()).unwrap() {
            break p;
        }
    };
    // The first is sent before the download starts, so a window has its row before any end.
    let first = progress();
    assert_eq!((first.id.as_str(), first.received_bytes), (id.as_str(), 0));
    let p = progress();
    assert_eq!((p.id.as_str(), p.total_bytes), (id.as_str(), 8 << 20));
    assert!(p.received_bytes > 0 && p.bytes_per_sec > 0.0);
    let part = dir.0.join("models/model-Q8_0.gguf.part");
    assert!(part.exists());
    app.cancel_download(&id).unwrap();
    let Event::DownloadFailed(f) = end(&rx, &id) else {
        panic!("not failed")
    };
    assert!(f.cancelled);
    assert!(!part.exists());
    assert_eq!(app.list_downloads().unwrap()[0].state, "cancelled");
    assert!(app.cancel_download("nope").is_err());
}

#[test]
fn missing_and_gated_files_say_why() {
    let hub = Hub::start();
    let (app, rx, _dir) = app(&hub);
    let a = bytes(1000, 1);
    hub.tree(
        "org/model",
        json!([
            entry("model-Q8_0.gguf", &a, Some(sha(&a))),
            entry("model-Q4_0.gguf", &a, Some(sha(&a))),
        ]),
        json!([]),
    );
    hub.route("/org/model/resolve/main/model-Q8_0.gguf", status(401));
    // model-Q4_0.gguf is listed but resolves to 404.
    assert_eq!(
        app.hf_inspect("org/model", "model-Q4_0.gguf").unwrap_err(),
        "Hugging Face has no such file: org/model/model-Q4_0.gguf."
    );
    assert_eq!(
        app.hf_inspect("org/model", "model-Q2_K.gguf").unwrap_err(),
        "Hugging Face has no such file: org/model/model-Q2_K.gguf."
    );
    assert_eq!(
        app.hf_inspect("org/model", "model-Q8_0.gguf").unwrap_err(),
        AUTH
    );
    let id = app
        .download_model("org/model", "model-Q8_0.gguf", None)
        .unwrap();
    let Event::DownloadFailed(f) = end(&rx, &id) else {
        panic!("not failed")
    };
    assert_eq!(f.message, AUTH);
}

#[test]
fn search_lists_repos_and_rides_out_a_busy_hub() {
    let hub = Hub::start();
    let (app, _rx, _dir) = app(&hub);
    let path = "/api/models?filter=gguf&sort=downloads&direction=-1&limit=24&expand%5B%5D=downloads&expand%5B%5D=tags&expand%5B%5D=gguf&expand%5B%5D=pipeline_tag&search=qwen%20coder%2B";
    // An image-text-to-text repo (Qwen3.5's) is a chat model; an embedding repo is not.
    let mut found = file(
        json!([{"id": "org/qwen-coder-GGUF", "downloads": 7, "tags": ["gguf", "qwen2"],
                "pipeline_tag": "image-text-to-text",
                "gguf": {"total": 7_615_616_512u64, "architecture": "qwen2", "context_length": 32768}},
               {"id": "e/embed-GGUF", "pipeline_tag": "feature-extraction", "gguf": {"architecture": "bert"}},
               {"id": "a/b", "gguf": {"architecture": "clip"}}])
            .to_string()
            .into_bytes(),
    );
    found.busy = Arc::new(AtomicUsize::new(1));
    hub.route(path, found);
    let got = app.hf_search(" qwen coder+ ", "downloads", 24).unwrap();
    assert_eq!(
        got,
        [
            // Its pipeline is image-text-to-text, which its tags now say: the app knows it reads
            // images before listing its files.
            HfModel {
                repo: "org/qwen-coder-GGUF".into(),
                downloads: 7,
                tags: vec!["gguf".into(), "qwen2".into(), "image-text-to-text".into()],
                arch: "qwen2".into(),
                params: Some(7_615_616_512),
                context_length: Some(32768),
                arch_runs: true,
            },
            HfModel {
                repo: "a/b".into(),
                downloads: 0,
                tags: vec![],
                arch: "clip".into(),
                params: None,
                context_length: None,
                arch_runs: false,
            },
        ]
    );
    // The 503, then the answer.
    assert_eq!(hub.ranges(path).len(), 2);
    assert!(app.hf_search("x", "name", 5).is_err());
}

/// "moe" is a filter, not a name word: the Hub is asked for "Qwen", and an `-A3B` repo is kept
/// where a dense one is not.
#[test]
fn a_moe_search_keeps_the_experts_models() {
    let hub = Hub::start();
    let (app, _rx, _dir) = app(&hub);
    let path = "/api/models?filter=gguf&sort=downloads&direction=-1&limit=100&expand%5B%5D=downloads&expand%5B%5D=tags&expand%5B%5D=gguf&expand%5B%5D=pipeline_tag&search=Qwen";
    hub.route(
        path,
        file(
            json!([{"id": "org/Qwen3-8B-GGUF", "gguf": {"architecture": "qwen3"}},
                   {"id": "org/Qwen3.5-35B-A3B-GGUF", "gguf": {"architecture": "qwen35moe"}},
                   {"id": "org/Qwen3-30B-A3B-GGUF"}])
            .to_string()
            .into_bytes(),
        ),
    );
    let got = app.hf_search("Qwen MoE GGUF", "downloads", 1).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].repo, "org/Qwen3.5-35B-A3B-GGUF");
}
