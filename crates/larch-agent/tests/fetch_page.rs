//! fetch_page and fetch_many against a server on this machine, and the extraction goldens on the
//! saved pages in fixtures/pages.

use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use larch_agent::tool::{Permission, Registry, ToolCtx};
use larch_agent::web::fetch::{self, MAX_BYTES, Page};
use serde_json::{Value, json};
use url::Url;

/// A route's status, extra headers and body.
type Reply = (u16, Vec<(&'static str, String)>, Vec<u8>);

/// An HTTP/1.1 server on 127.0.0.1 that answers each path from `routes` (404 otherwise) and
/// logs each request's path and time.
struct Server {
    base: String,
    log: Arc<Mutex<Vec<(String, Instant)>>>,
}

impl Server {
    fn new(routes: Vec<(&'static str, Reply)>) -> Server {
        fetch::allow_this_machine();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let routes: Arc<HashMap<_, _>> = Arc::new(routes.into_iter().collect());
        let log = Arc::new(Mutex::new(Vec::new()));
        let seen = log.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let (routes, seen) = (routes.clone(), seen.clone());
                thread::spawn(move || answer(stream.unwrap(), &routes, &seen));
            }
        });
        Server { base, log }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn paths(&self) -> Vec<String> {
        self.log
            .lock()
            .unwrap()
            .iter()
            .map(|(p, _)| p.clone())
            .collect()
    }
}

fn answer(
    mut stream: TcpStream,
    routes: &HashMap<&'static str, Reply>,
    log: &Mutex<Vec<(String, Instant)>>,
) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut first = String::new();
    reader.read_line(&mut first).unwrap();
    let path = first.split(' ').nth(1).unwrap_or("/").to_string();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
            break;
        }
    }
    log.lock().unwrap().push((path.clone(), Instant::now()));
    if path == "/hang" {
        thread::sleep(Duration::from_secs(60));
        return;
    }
    let (status, headers, body) =
        routes
            .get(path.as_str())
            .cloned()
            .unwrap_or((404, vec![], b"not found".to_vec()));
    let mut head = format!(
        "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&body);
}

fn html(body: &str) -> Reply {
    (
        200,
        vec![("Content-Type", "text/html; charset=utf-8".into())],
        body.as_bytes().to_vec(),
    )
}

fn robots(body: &str) -> Reply {
    (
        200,
        vec![("Content-Type", "text/plain".into())],
        body.as_bytes().to_vec(),
    )
}

fn redirect(to: &str) -> Reply {
    (302, vec![("Location", to.into())], Vec::new())
}

const ARTICLE: &str = "<html><head><title>A page</title></head><body><nav>menu</nav>\
    <article><p>The main text of the page, long enough, with a <a href='/b'>link</a>.</p></article></body></html>";

fn data() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn one(url: &str, dir: &Path) -> Result<Page, String> {
    let mut pages = fetch::fetch_many(&[url], MAX_BYTES, dir);
    pages.pop().unwrap().map_err(|e| e.to_string())
}

#[test]
fn fetch_page_is_a_read_whose_results_are_untrusted_and_returns_the_page() {
    let server = Server::new(vec![("/a", html(ARTICLE))]);
    let dir = data();
    let ctx = ToolCtx {
        allowed: vec![],
        data_dir: dir.path().into(),
        source: "test".into(),
    };
    let registry = Registry::builtin();
    let tool = registry.get("fetch_page").unwrap();
    let spec = tool.spec();
    assert_eq!(spec.permission, Permission::Read);
    assert!(spec.untrusted_results);
    let result = tool.call(&json!({"url": server.url("/a")}), &ctx).unwrap();
    let page: Value = serde_json::from_str(&result).unwrap();
    assert_eq!(page["url"], server.url("/a"));
    assert_eq!(page["final_url"], server.url("/a"));
    assert_eq!(page["title"], "A page");
    assert_eq!(
        page["text"],
        format!(
            "The main text of the page, long enough, with a [link]({}).",
            server.url("/b")
        )
    );
    assert_eq!(page["bytes"], ARTICLE.len());
    assert_eq!(page["truncated"], false);
    let sha = page["sha256"].as_str().unwrap();
    assert_eq!(sha.len(), 64);
    let at = page["fetched_at"].as_str().unwrap();
    assert!(at.len() == 20 && at.ends_with('Z'), "{at}");
    // The cache keeps the page under its URL's folder, by its content hash.
    let cached: Vec<_> = fs::read_dir(dir.path().join("agent/web"))
        .unwrap()
        .flat_map(|d| fs::read_dir(d.unwrap().path()).unwrap())
        .map(|f| f.unwrap().path())
        .collect();
    assert_eq!(cached.len(), 1);
    assert_eq!(
        cached[0].file_name().unwrap(),
        format!("{sha}.json").as_str()
    );
    let saved: Value = serde_json::from_slice(&fs::read(&cached[0]).unwrap()).unwrap();
    assert_eq!(saved, page);
    // robots.txt was asked for first, once.
    assert_eq!(server.paths(), ["/robots.txt", "/a"]);
}

#[test]
fn robots_txt_is_honoured_for_larch_and_everyone() {
    let server = Server::new(vec![
        (
            "/robots.txt",
            robots(
                "User-agent: *\nDisallow: /\n\nUser-agent: Larch\nDisallow: /private\nAllow: /private/ok\n",
            ),
        ),
        ("/open", html(ARTICLE)),
        ("/private/x", html(ARTICLE)),
        ("/private/ok", html(ARTICLE)),
    ]);
    let dir = data();
    let got = fetch::fetch_many(
        &[
            &server.url("/open"),
            &server.url("/private/x"),
            &server.url("/private/ok"),
        ],
        MAX_BYTES,
        dir.path(),
    );
    assert!(got[0].is_ok(), "{:?}", got[0]);
    let refused = got[1].as_ref().unwrap_err().to_string();
    assert!(refused.contains("robots.txt"), "{refused}");
    assert!(got[2].is_ok(), "{:?}", got[2]);
    assert!(!server.paths().contains(&"/private/x".to_string()));
}

#[test]
fn a_robots_txt_that_errors_refuses_the_fetch_says_so_and_is_read_again_and_a_missing_one_allows_it()
 {
    let down = Server::new(vec![
        ("/robots.txt", (503, vec![], Vec::new())),
        ("/a", html(ARTICLE)),
    ]);
    let err = one(&down.url("/a"), data().path()).unwrap_err();
    assert!(
        err.contains("127.0.0.1's robots.txt could not be read: ") && err.contains("HTTP 503"),
        "{err}"
    );
    assert_eq!(down.paths(), ["/robots.txt"]);
    // The next fetch asks again rather than taking the failure for a Disallow.
    assert!(one(&down.url("/a"), data().path()).is_err());
    assert_eq!(down.paths(), ["/robots.txt", "/robots.txt"]);
    let missing = Server::new(vec![("/a", html(ARTICLE))]);
    assert!(one(&missing.url("/a"), data().path()).is_ok());
}

#[test]
fn one_host_gets_one_request_a_second_and_hosts_run_in_parallel() {
    let a = Server::new(vec![("/1", html(ARTICLE)), ("/2", html(ARTICLE))]);
    let b = Server::new(vec![("/1", html(ARTICLE))]);
    let dir = data();
    let start = Instant::now();
    let got = fetch::fetch_many(
        &[&a.url("/1"), &a.url("/2"), &b.url("/1")],
        MAX_BYTES,
        dir.path(),
    );
    // a's robots.txt, /1 and /2 take three turns a second apart, so the last starts 2 s in or
    // later. The server's own clock cannot show the gaps: it adds the runner's scheduling delays.
    assert!(
        start.elapsed() >= Duration::from_secs(2),
        "{:?}",
        start.elapsed()
    );
    assert!(got.iter().all(Result::is_ok), "{got:?}");
    let times = |s: &Server| {
        s.log
            .lock()
            .unwrap()
            .iter()
            .map(|(_, t)| *t)
            .collect::<Vec<_>>()
    };
    let (ta, tb) = (times(&a), times(&b));
    assert_eq!(ta.len(), 3);
    // b's requests do not wait behind a's.
    assert!(tb[0] < ta[2]);
}

#[test]
fn redirects_stay_on_the_host_or_go_to_https() {
    let server = Server::new(vec![
        ("/old", redirect("/new")),
        ("/new", html(ARTICLE)),
        ("/loop", redirect("/loop")),
    ]);
    let dir = data();
    let page = one(&server.url("/old"), dir.path()).unwrap();
    assert_eq!(page.url, server.url("/old"));
    assert_eq!(page.final_url, server.url("/new"));
    // The same server under another host name, over plain http.
    let other = server.url("/new").replace("127.0.0.1", "localhost");
    let away = Server::new(vec![("/go", redirect(&other))]);
    let err = one(&away.url("/go"), dir.path()).unwrap_err();
    assert!(err.contains("redirect"), "{err}");
    let err = one(&server.url("/loop"), dir.path()).unwrap_err();
    assert!(err.contains("redirects"), "{err}");
}

#[test]
fn a_redirect_to_a_private_address_is_refused() {
    let server = Server::new(vec![("/go", redirect("https://10.0.0.1/admin"))]);
    let err = one(&server.url("/go"), data().path()).unwrap_err();
    assert!(
        err.contains("10.0.0.1") && err.contains("never opens"),
        "{err}"
    );
}

#[test]
fn fetch_seen_refuses_the_one_unseen_url_fetches_the_rest_and_sees_their_links() {
    let server = Server::new(vec![("/a", html(ARTICLE)), ("/c", html(ARTICLE))]);
    let mut seen = fetch::Seen::default();
    seen.read(&format!("{} and {}", server.url("/a"), server.url("/c")));
    seen.untrusted = true;
    let bad = server.url("/a?d=secret");
    let mut each = Vec::new();
    let got = fetch::fetch_seen(
        &[&server.url("/a"), &bad, &server.url("/c")],
        MAX_BYTES,
        data().path(),
        &mut seen,
        |r| each.push(r.is_ok()),
    );
    assert!(got[0].is_ok() && got[2].is_ok(), "{got:?}");
    // Each result as it comes, in the URLs' order, the refusal in its place.
    assert_eq!(each, [true, false, true]);
    let why = got[1].as_ref().unwrap_err().to_string();
    assert!(
        why.contains(&bad) && why.contains("carry data out"),
        "{why}"
    );
    assert!(!server.paths().contains(&"/a?d=secret".to_string()));
    // ARTICLE's link, now that its page was fetched.
    assert_eq!(seen.refuses(&server.url("/b")), None);
}

#[test]
fn a_body_over_max_bytes_is_cut_and_marked_truncated() {
    let long = format!("<p>{}</p>", "word, ".repeat(400));
    let server = Server::new(vec![("/big", html(&long))]);
    let mut got = fetch::fetch_many(&[&server.url("/big")], 100, data().path());
    let page = got.pop().unwrap().unwrap();
    assert_eq!(page.bytes, 100);
    assert!(page.truncated);
}

#[test]
fn a_document_link_reads_past_a_pages_limit_keeps_its_pdf_and_says_why_it_was_refused() {
    let paper =
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/paper-01.pdf"))
            .unwrap();
    let filing = format!("<p>{}</p>", "revenue ".repeat(MAX_BYTES / 8 + 1000));
    let server = Server::new(vec![
        (
            "/paper",
            (
                200,
                vec![("Content-Type", "application/pdf".into())],
                paper.clone(),
            ),
        ),
        ("/filing", html(&filing)),
        ("/paywall", (403, vec![], b"Subscribe".to_vec())),
    ]);
    let dir = data();
    let urls = [
        server.url("/paper"),
        server.url("/filing"),
        server.url("/paywall"),
    ];
    let urls: Vec<&str> = urls.iter().map(String::as_str).collect();
    let got = fetch::fetch_documents(&urls, dir.path(), &mut fetch::Seen::default(), true);
    let pdf = got[0].as_ref().unwrap();
    assert_eq!(pdf.text, "Attention Is Not All You Need");
    let blob = dir.path().join("agent/docs/blobs").join(&pdf.sha256);
    assert_eq!(fs::read(blob).unwrap(), paper);
    // With follow-ups off, no PDF's bytes are kept; a blob that cannot be written fails no read.
    let off = data();
    let read = |data: &Path, blobs: bool| {
        let got = fetch::fetch_documents(&urls[..1], data, &mut fetch::Seen::default(), blobs);
        got.into_iter().next().unwrap().unwrap()
    };
    assert!(read(off.path(), false).pdf);
    assert!(!off.path().join("agent/docs").exists());
    fs::create_dir_all(off.path().join("agent/docs")).unwrap();
    fs::write(off.path().join("agent/docs/blobs"), "").unwrap();
    assert_eq!(read(off.path(), true).text, pdf.text);
    // An EDGAR filing's HTML runs past a web page's limit: a document link reads it whole.
    let filing = got[1].as_ref().unwrap();
    assert!(
        !filing.truncated && filing.bytes > MAX_BYTES as u64,
        "{}",
        filing.bytes
    );
    let refused = got[2].as_ref().unwrap_err().to_string();
    assert!(refused.contains("HTTP 403: the site refused"), "{refused}");
    // A search's fetch keeps a page's limit and no PDF's bytes.
    let dir = data();
    let page = one(&server.url("/filing"), dir.path()).unwrap();
    assert!(page.truncated);
    one(&server.url("/paper"), dir.path()).unwrap();
    assert!(!dir.path().join("agent/docs").exists());
}

#[test]
fn a_page_with_no_content_type_is_sniffed_and_a_runaway_title_is_cut() {
    let bare = (
        200,
        vec![],
        b"<!doctype html><title>Bare</title><p>Text of the bare page, long enough.</p>".to_vec(),
    );
    let runaway = html(&format!("<title>{}", "word ".repeat(10_000)));
    // A Latin-1 robots.txt that allows everything.
    let latin1 = (
        200,
        vec![],
        b"# \xc6\nUser-agent: *\nDisallow: /nothing\n".to_vec(),
    );
    let server = Server::new(vec![
        ("/robots.txt", latin1),
        ("/bare", bare),
        ("/runaway", runaway),
    ]);
    let page = one(&server.url("/bare"), data().path()).unwrap();
    assert_eq!(
        (page.title.as_str(), page.text.as_str()),
        ("Bare", "Text of the bare page, long enough.")
    );
    let dir = data();
    let ctx = ToolCtx {
        allowed: vec![],
        data_dir: dir.path().into(),
        source: "test".into(),
    };
    let registry = Registry::builtin();
    let result = (registry.get("fetch_page").unwrap())
        .call(&json!({"url": server.url("/runaway")}), &ctx)
        .unwrap();
    let page: Value = serde_json::from_str(&result).unwrap();
    assert!(page["title"].as_str().unwrap().len() <= 1000);
    assert_eq!(page["truncated"], true);
}

#[test]
fn every_fetch_caps_the_title_not_only_fetch_page() {
    let server = Server::new(vec![(
        "/long",
        // 5,001 bytes: byte 1000 falls inside an "é".
        html(&format!("<title>a{}</title><p>Body.</p>", "é".repeat(2500))),
    )]);
    let page = one(&server.url("/long"), data().path()).unwrap();
    assert!(page.title.len() <= 1000, "{}", page.title.len());
    assert!(page.title.starts_with("aé"));
    assert!(page.truncated);
}

#[test]
fn control_characters_are_stripped_but_newlines_and_tabs_kept() {
    let dirty = "Bell\u{7} \u{1b}[31mred\u{7f} next\u{85} line \u{9b}2J";
    let server = Server::new(vec![
        (
            "/html",
            html(&format!("<title>{dirty}</title><p>{dirty}</p>")),
        ),
        (
            "/text",
            (
                200,
                vec![("Content-Type", "text/plain".into())],
                format!("{dirty}\n\tindented").into_bytes(),
            ),
        ),
    ]);
    let clean = "Bell [31mred next line 2J";
    let page = one(&server.url("/html"), data().path()).unwrap();
    assert_eq!((page.title.as_str(), page.text.as_str()), (clean, clean));
    let page = one(&server.url("/text"), data().path()).unwrap();
    assert_eq!(page.text, format!("{clean}\n\tindented"));
}

#[test]
fn a_bare_carriage_return_still_ends_a_line() {
    let base = Url::parse("http://a.example/").unwrap();
    let (_, text) = fetch::convert("text/plain", b"a\rb\rc\r\nd", &base).unwrap();
    assert_eq!(text.lines().collect::<Vec<_>>(), ["a", "b", "c", "d"]);
}

#[test]
fn a_pdf_is_read_by_the_paper_reader_and_a_broken_one_is_an_error() {
    let kind = || vec![("Content-Type", "application/pdf".into())];
    let paper =
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/paper-01.pdf"))
            .unwrap();
    let server = Server::new(vec![
        ("/paper", (200, kind(), paper)),
        ("/broken", (200, kind(), b"%PDF-1.4\n".to_vec())),
    ]);
    let page = one(&server.url("/paper"), data().path()).unwrap();
    assert_eq!(page.title, "Attention Is Not All You Need");
    assert_eq!(page.text, "Attention Is Not All You Need");
    let err = one(&server.url("/broken"), data().path()).unwrap_err();
    assert!(err.contains("PDF could not be read"), "{err}");
}

/// `pdf` with a comment of `pad` bytes after its header, its offsets moved to match: a PDF past
/// fetch's page limit.
fn padded(pdf: &[u8], pad: usize) -> Vec<u8> {
    let mut lines = pdf.split_inclusive(|&b| b == b'\n');
    let mut out = lines.next().unwrap().to_vec();
    out.extend(b"%");
    out.extend(vec![b'a'; pad]);
    out.push(b'\n');
    let moved = |n: &[u8]| {
        std::str::from_utf8(n)
            .unwrap()
            .trim()
            .parse::<usize>()
            .unwrap()
            + pad
            + 2
    };
    let mut after_startxref = false;
    for line in lines {
        let bare = line.trim_ascii_end();
        if bare.len() == 18 && bare.ends_with(b" n") && bare[10] == b' ' {
            out.extend(format!("{:010}", moved(&bare[..10])).as_bytes());
            out.extend(&line[10..]);
        } else if after_startxref {
            out.extend(format!("{}\n", moved(bare)).as_bytes());
        } else {
            out.extend(line);
        }
        after_startxref = bare == b"startxref";
    }
    out
}

#[test]
fn a_pdf_past_the_page_limit_is_read_whole() {
    // The arXiv PDF 2610.39302 is 3.2 MB: cut at 2 MiB it had no cross reference table, and the
    // reader said "invalid start value" (5 Nov 2026).
    let paper =
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/paper-01.pdf"))
            .unwrap();
    let big = padded(&paper, fetch::MAX_BYTES + (1 << 20));
    let kind = vec![("Content-Type", "application/pdf".into())];
    let server = Server::new(vec![("/big", (200, kind, big))]);
    let page = one(&server.url("/big"), data().path()).unwrap();
    assert_eq!(page.text, "Attention Is Not All You Need");
    assert!(!page.truncated);
}

#[test]
fn only_web_urls_are_fetched() {
    for url in ["ftp://example.org/x", "file:///etc/passwd", "not a url"] {
        assert!(one(url, data().path()).is_err(), "{url}");
    }
}

#[test]
fn a_server_that_never_answers_times_out_after_ten_seconds() {
    let server = Server::new(vec![]);
    let start = Instant::now();
    // robots.txt is a 404, then /hang stalls for a minute.
    let err = one(&server.url("/hang"), data().path()).unwrap_err();
    let took = start.elapsed();
    assert!(err.ends_with("/hang: no answer within 10 s"), "{err}");
    // A busy runner adds seconds; a 30 s timeout would still fail.
    assert!(
        took >= Duration::from_secs(10) && took < Duration::from_secs(25),
        "{took:?}"
    );
}

#[derive(serde::Deserialize)]
struct Golden {
    file: String,
    url: String,
    title: Option<String>,
    start: Option<String>,
    error: Option<String>,
}

#[test]
fn goldens_title_and_first_200_characters_of_30_saved_pages() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages");
    let goldens: Vec<Golden> =
        serde_json::from_slice(&fs::read(dir.join("goldens.json")).unwrap()).unwrap();
    assert_eq!(goldens.len(), 30);
    let mut wrong = Vec::new();
    for g in &goldens {
        let body = fs::read(dir.join(&g.file)).unwrap();
        let kind = if g.file.ends_with(".pdf") {
            "application/pdf"
        } else {
            "text/html"
        };
        let got = fetch::convert(kind, &body, &Url::parse(&g.url).unwrap());
        let got = match got {
            Ok((title, text)) => {
                let start: String = text.chars().take(200).collect();
                (Some(title), Some(start), None)
            }
            Err(e) => (None, None, Some(e.to_string())),
        };
        if got != (g.title.clone(), g.start.clone(), g.error.clone()) {
            wrong.push(format!(
                "{}:\n  want {:?}\n  got  {got:?}",
                g.file,
                (&g.title, &g.start, &g.error)
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
