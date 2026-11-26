//! `fetch_page` and `fetch_many`: a page over HTTPS, as polite as a crawler (robots.txt, one
//! request a second per host), its main text as Markdown, and a copy on disk by content hash so a
//! quote can be found again offline.

use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{self, Read, Write},
    net::{IpAddr, SocketAddr, ToSocketAddrs},
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{
        Arc, LazyLock, Mutex, PoisonError,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use url::{Host, Url};

use crate::memory::dates;
use crate::tool::{
    ARGS_TOKENS, Permission, Tool, ToolCtx, ToolError, ToolSpec, str_arg, whole_arg,
};
use crate::{folio::source, skills::read_papers};

/// The most of a page's body read; the rest is cut and the page marked `truncated`. A PDF reads
/// up to `PDF_BYTES` whatever the caller's limit.
pub const MAX_BYTES: usize = 2 << 20;

/// The most of a PDF a fetch reads: a whole paper, which is useless cut.
pub const PDF_BYTES: usize = 32 << 20;
/// The text `fetch_page` hands the model; `fetch_many` and the cache keep it whole.
const TEXT_BYTES: usize = 16_000;
/// The longest title a page keeps: an unclosed `<title>` runs to the end of the page.
const TITLE_BYTES: usize = 1000;
/// The longest one request may take, body included.
// ponytail: one timeout for the whole body, so a 32 MiB PDF needs ~27 Mbit/s; give PDFs their
// own when a slow link shows it
const TIMEOUT: Duration = Duration::from_secs(10);
/// Redirects followed before giving up.
const MAX_HOPS: usize = 10;
/// robots.txt read; RFC 9309 asks for at least 500 KiB.
const ROBOTS_BYTES: u64 = 512 << 10;
/// The name robots.txt groups are matched against.
const AGENT: &str = "larch";

/// A fetched page, as `fetch_page` returns it and the cache keeps it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Page {
    /// The URL asked for.
    pub url: String,
    /// The URL after redirects and the https upgrade.
    pub final_url: String,
    /// When the body came, `%Y-%m-%dT%H:%M:%SZ`.
    pub fetched_at: String,
    /// SHA-256 of the body as read, hex.
    pub sha256: String,
    /// The `<title>`, or a PDF's first line; empty for plain text.
    pub title: String,
    /// The main text as Markdown, links as `[text](url)`.
    pub text: String,
    /// Bytes of body read.
    pub bytes: u64,
    /// The body passed `max_bytes`, the title was cut, or `fetch_page` cut the text.
    pub truncated: bool,
    /// The day the page says it was published, `YYYY-MM-DD` (`published`); `None` when it does
    /// not say, and in a copy saved before the field.
    #[serde(default)]
    pub published: Option<String>,
    /// The body is a PDF, whose text has a form feed between pages.
    #[serde(default)]
    pub pdf: bool,
}

/// A host's rules from its robots.txt, or why it could not be read and when that read ended.
type Robots = Result<Vec<(bool, String)>, (String, Instant)>;

/// A host's turn and robots.txt, shared by every fetch in the process.
struct Site {
    next: Mutex<Instant>,
    robots: Mutex<Option<Robots>>,
}

/// A score a page lays out as one cell a line ("Aston Villa", "2", "1", "Manchester City" on
/// mancity.com/results) on one line, "Aston Villa 2-1 Manchester City": split, the reader at no
/// or little thinking took the line after the higher number for the winner (sport-mancity, 1 Nov).
pub fn scores_joined(text: &str) -> std::borrow::Cow<'_, str> {
    // A team is a short line of letters only: "$", "Rated" with a lone number after, or a third
    // number (tennis sets) is no score.
    static SPLIT: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?m)^([ \t]*)(\p{L}[\p{L} .'&-]{0,39}?)[ \t]*\n[ \t]*(\d{1,3})[ \t]*\n[ \t]*(\d{1,3})[ \t]*\n[ \t]*(\p{L}[\p{L} .'&-]{0,39}?)[ \t]*$")
            .expect("a valid pattern")
    });
    static NUMBER: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\A\n[ \t]*\d+[ \t]*(?:\n|\z)").expect("a valid pattern"));
    SPLIT.replace_all(text, |c: &regex::Captures| {
        let all = c.get(0).expect("the whole match");
        if NUMBER.is_match(&text[all.end()..]) {
            all.as_str().to_owned()
        } else {
            format!("{}{} {}-{} {}", &c[1], &c[2], &c[3], &c[4], &c[5])
        }
    })
}

static SITES: LazyLock<Mutex<HashMap<String, Arc<Site>>>> = LazyLock::new(Default::default);

// ponytail: our own agents, since larch_fetch::agent follows redirects itself and has no total
// timeout; the OS's TLS where ureq can use it, as larch_fetch's. Use larch_fetch's once it takes
// both (a1).
/// The proxy from the environment, and a direct agent for this machine, which ureq would send
/// through the proxy as it ignores NO_PROXY. A direct agent resolves through `public`, so a name
/// that turns private between the check and the connection (DNS rebinding) is refused too.
// ponytail: through a proxy the proxy resolves the name, so only `refused` before each hop guards
// it: a name that does not resolve here, or a rebinding after the check, gets through (owner's
// call, 26 Oct); pin the checked address (CONNECT by IP) if the proxy is ever not the user's own
static AGENTS: LazyLock<[ureq::Agent; 2]> = LazyLock::new(|| {
    [true, false].map(|proxy| {
        let agent = ureq::AgentBuilder::new()
            .try_proxy_from_env(false)
            .redirects(0)
            .timeout(TIMEOUT)
            .user_agent(concat!("larch/", env!("CARGO_PKG_VERSION")));
        // rustls trusts only the roots Windows has installed, and Windows installs most on first
        // use; its own verifier fetches them, as a browser on it does.
        #[cfg(any(windows, target_os = "macos"))]
        let agent = match native_tls::TlsConnector::new() {
            Ok(tls) => agent.tls_connector(Arc::new(tls)),
            Err(_) => agent,
        };
        match PROXY.clone().filter(|_| proxy) {
            Some(p) => agent.proxy(p),
            None => agent.resolver(public),
        }
        .build()
    })
});

/// The proxy ureq's `try_proxy_from_env` would take, read once here so `refused` knows the
/// first agent has one.
static PROXY: LazyLock<Option<ureq::Proxy>> = LazyLock::new(|| {
    [
        "ALL_PROXY",
        "all_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ]
    .into_iter()
    .find_map(|var| ureq::Proxy::new(std::env::var(var).ok()?).ok())
});

/// Lets fetches reach this machine's own (loopback) addresses: set only by Rust code, never by
/// a setting or a variable a page could influence.
static THIS_MACHINE: AtomicBool = AtomicBool::new(false);

/// Lets fetches reach loopback addresses, for tests whose server runs on 127.0.0.1.
#[doc(hidden)]
pub fn allow_this_machine() {
    THIS_MACHINE.store(true, Ordering::Relaxed);
}

/// What `ip` is, when it is this machine or a network behind it that a page must not make
/// fetch_page reach: loopback, private (RFC 1918, fc00::/7), carrier-grade NAT (100.64/10,
/// where some clouds keep their metadata service), link-local or unspecified.
fn private(ip: IpAddr) -> Option<&'static str> {
    match ip.to_canonical() {
        ip if ip.is_loopback() => (!THIS_MACHINE.load(Ordering::Relaxed)).then_some("this machine"),
        ip if ip.is_unspecified() => Some("the unspecified address"),
        IpAddr::V4(v4) if v4.is_private() => Some("a private address"),
        // Ipv4Addr::is_shared, unstable.
        IpAddr::V4(v4) if v4.octets()[0] == 100 && v4.octets()[1] & 0xc0 == 64 => {
            Some("a carrier-grade NAT address")
        }
        IpAddr::V4(v4) if v4.is_link_local() => Some("a link-local address"),
        IpAddr::V6(v6) if v6.is_unique_local() => Some("a private address"),
        IpAddr::V6(v6) if v6.is_unicast_link_local() => Some("a link-local address"),
        _ => None,
    }
}

/// The addresses `netloc` (`host:port`) resolves to, or why not one is fetched: any address that
/// is `private` refuses them all, as a connection may go to any of them, and a name that does not
/// resolve here cannot be checked.
fn public(netloc: &str) -> io::Result<Vec<SocketAddr>> {
    let unchecked = |e: io::Error| {
        let why = format!(
            "{netloc} does not resolve on this computer ({e}), so it cannot be checked for a private address"
        );
        io::Error::new(e.kind(), why)
    };
    checked(
        netloc,
        netloc.to_socket_addrs().map_err(unchecked)?.collect(),
    )
}

/// `addrs`, which `netloc` resolved to, unless one is `private`.
fn checked(netloc: &str, addrs: Vec<SocketAddr>) -> io::Result<Vec<SocketAddr>> {
    match addrs
        .iter()
        .find_map(|a| private(a.ip()).map(|what| (a.ip(), what)))
    {
        Some((ip, what)) => Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("{netloc} resolves to {ip}, {what}, which fetch_page never opens"),
        )),
        None => Ok(addrs),
    }
}

/// Why `url`'s host is not fetched, checked before each hop. Direct, every address it resolves
/// to (`public`). Through a proxy (`proxied`), which resolves the name itself, a name that does
/// not resolve here goes to the proxy; an address, `localhost` and `*.localhost` never do.
fn refused(url: &Url, proxied: bool) -> Result<(), ToolError> {
    let netloc = format!(
        "{}:{}",
        url.host_str().unwrap_or_default(),
        url.port_or_known_default().unwrap_or_default()
    );
    let refuse = |e: io::Error| ToolError::Fetch(format!("{url}: {e}"));
    match url.host() {
        Some(Host::Domain(name)) if proxied => {
            let name = name.trim_end_matches('.');
            if name.eq_ignore_ascii_case("localhost") || name.ends_with(".localhost") {
                return Err(ToolError::Fetch(format!(
                    "{url}: {name} is this machine, which fetch_page never opens"
                )));
            }
            match netloc.to_socket_addrs() {
                Ok(addrs) => checked(&netloc, addrs.collect()).map(drop).map_err(refuse),
                Err(_) => Ok(()),
            }
        }
        _ => public(&netloc).map(drop).map_err(refuse),
    }
}

/// The pages at `urls`, in order, fetched at once: one thread per URL, hosts in parallel, each
/// host one request a second. Each page read is saved under `data_dir/agent/web`.
// ponytail: a thread per URL; a pool when the research skill passes lists of 32 or more
pub fn fetch_many(
    urls: &[&str],
    max_bytes: usize,
    data_dir: &Path,
) -> Vec<Result<Page, ToolError>> {
    fetch_each(urls, Want::page(max_bytes), data_dir, false, |_| {})
}

/// What one fetch reads: up to `bytes` of a body, and, for a caller that reads documents with
/// follow-ups on, a PDF's bytes kept as its blob (`source::keep`).
#[derive(Clone, Copy)]
struct Want {
    bytes: usize,
    blobs: bool,
}

impl Want {
    /// A web page for a search or a question: `max_bytes` of it, a PDF `PDF_BYTES`.
    fn page(max_bytes: usize) -> Want {
        Want {
            bytes: max_bytes,
            blobs: false,
        }
    }
}

/// `fetch_many`, which gives `each` each result as soon as it and the ones before it are in,
/// while the fetches after it still run; with `arrival`, as soon as it is in.
fn fetch_each(
    urls: &[&str],
    want: Want,
    data_dir: &Path,
    arrival: bool,
    mut each: impl FnMut(&Result<Page, ToolError>),
) -> Vec<Result<Page, ToolError>> {
    thread::scope(|s| {
        let (tx, rx) = mpsc::channel();
        for (i, url) in urls.iter().enumerate() {
            let tx = tx.clone();
            s.spawn(move || {
                let got = panic::catch_unwind(|| fetch(url, want, data_dir))
                    .unwrap_or_else(|_| Err(ToolError::Fetch("the fetch panicked".into())));
                let _ = tx.send((i, got));
            });
        }
        drop(tx);
        let mut got: Vec<Option<Result<Page, ToolError>>> = urls.iter().map(|_| None).collect();
        let mut next = 0;
        for (i, page) in rx {
            if arrival {
                each(&page);
            }
            got[i] = Some(page);
            while !arrival && let Some(Some(page)) = got.get(next) {
                each(page);
                next += 1;
            }
        }
        (got.into_iter())
            .map(|p| p.expect("every fetch sends its result"))
            .collect()
    })
}

/// Fetches that each run on a thread of their own and are read as they finish, in any order. The
/// ones still running when it is dropped run on unread, so a slow page holds up nothing (a web
/// answer with `web_answer::race`); each still saves its page under `data_dir/agent/web`.
pub struct Race {
    tx: mpsc::Sender<(usize, Result<Page, ToolError>)>,
    rx: mpsc::Receiver<(usize, Result<Page, ToolError>)>,
    live: usize,
}

impl Default for Race {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel();
        Self { tx, rx, live: 0 }
    }
}

impl Race {
    /// Starts the fetch of `url`, which `next` gives back with `tag`.
    pub fn start(&mut self, tag: usize, url: &str, max_bytes: usize, data_dir: &Path) {
        let (tx, url, data_dir) = (self.tx.clone(), url.to_owned(), data_dir.to_owned());
        self.live += 1;
        thread::spawn(move || {
            let got = panic::catch_unwind(AssertUnwindSafe(|| {
                fetch(&url, Want::page(max_bytes), &data_dir)
            }))
            .unwrap_or_else(|_| Err(ToolError::Fetch(format!("{url}: the fetch panicked"))));
            let _ = tx.send((tag, got));
        });
    }

    /// The fetches started and not yet read.
    pub fn live(&self) -> usize {
        self.live
    }

    /// The next fetch to finish, with the tag it was started with, or none in `wait`.
    pub fn next(&mut self, wait: Duration) -> Option<(usize, Result<Page, ToolError>)> {
        let got = (self.live > 0).then(|| self.rx.recv_timeout(wait).ok())??;
        self.live -= 1;
        Some(got)
    }
}

fn fetch(asked: &str, want: Want, data_dir: &Path) -> Result<Page, ToolError> {
    let max_bytes = want.bytes;
    let parsed = Url::parse(asked).map_err(|e| ToolError::Invalid(format!("'{asked}': {e}")))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host().is_none() {
        return Err(ToolError::Invalid(format!(
            "'{asked}' is not an http or https URL"
        )));
    }
    let mut url = upgrade(parsed);
    for _ in 0..=MAX_HOPS {
        // Before robots.txt, which would be read from that address too; `get` sends anything but
        // http through the first agent, the proxied one when there is a proxy.
        refused(&url, url.scheme() != "http" && PROXY.is_some())?;
        allowed(&url)?;
        let resp = get(&url)?;
        let status = resp.status();
        if (300..400).contains(&status) {
            let to = (resp.header("location"))
                .and_then(|to| url.join(to).ok())
                .ok_or_else(|| {
                    ToolError::Fetch(format!("{url}: HTTP {status} with no Location"))
                })?;
            if !hop_allowed(&url, &to) {
                return Err(ToolError::Fetch(format!(
                    "{url}: redirect to {to} refused: another host over plain http"
                )));
            }
            url = upgrade(to);
            continue;
        }
        if status >= 400 {
            return Err(ToolError::Fetch(refusal(&url, status)));
        }
        // Not `content_type()`, which reads a missing header as text/plain: that page is sniffed.
        let kind = (resp
            .header("content-type")
            .and_then(|v| v.split(';').next()))
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_default();
        let mut body = Vec::new();
        let mut reader = resp.into_reader();
        let failed = |e: io::Error| ToolError::Fetch(format!("{url}: {e}"));
        (reader.by_ref().take(max_bytes as u64 + 1))
            .read_to_end(&mut body)
            .map_err(failed)?;
        // A PDF is read whole up to PDF_BYTES: a cut one has no cross reference table, which
        // sits at its end ("invalid start value" on a 3.2 MB arXiv paper, 5 Nov 2026).
        let pdf = is_pdf(&kind, &body);
        let limit = if pdf {
            max_bytes.max(PDF_BYTES)
        } else {
            max_bytes
        };
        (reader.take((limit + 1 - body.len()) as u64))
            .read_to_end(&mut body)
            .map_err(failed)?;
        let truncated = body.len() > limit;
        if pdf && truncated {
            return Err(ToolError::Fetch(format!(
                "{url}: the PDF is over {} MiB, the largest read",
                limit >> 20
            )));
        }
        body.truncate(limit);
        let (mut title, text) = convert(&kind, &body, &url)?;
        let sha256 = larch::bench::hex(&larch::bench::sha256(&body));
        // A link can change or vanish: a document's original stays for OCR and a reopen. One
        // that cannot be kept fails no read; a follow-up that needs it says it is not kept.
        if want.blobs && pdf {
            let _ = source::keep(data_dir, &sha256, &body);
        }
        // Cut here, not only in `fetch_page`: research and web_answer put it in prompts and reports.
        let long_title = title.len() > TITLE_BYTES;
        title.truncate(title.floor_char_boundary(TITLE_BYTES));
        // A PDF's bytes hold no page date, and lossy text of 32 MiB is no small string.
        let published = match pdf {
            true => published("", url.as_str()),
            false => published(&String::from_utf8_lossy(&body), url.as_str()),
        };
        let sha256 = larch::bench::hex(&larch::bench::sha256(&body));
        // A link can change or vanish: a document's original stays for OCR and a reopen. One
        // that cannot be kept fails no read; a follow-up that needs it says it is not kept.
        if want.blobs && pdf {
            let _ = source::keep(data_dir, &sha256, &body);
        }
        let page = Page {
            url: asked.into(),
            final_url: url.into(),
            fetched_at: larch::bench::utc_now(),
            sha256,
            title,
            text,
            bytes: body.len() as u64,
            truncated: truncated || long_title,
            published,
            pdf,
        };
        save(&page, data_dir)?;
        return Ok(page);
    }
    Err(ToolError::Fetch(format!(
        "{asked}: more than {MAX_HOPS} redirects"
    )))
}

/// `url`'s answer `status` as the model hears it: a refusal in one plain sentence that says
/// what the site wants, so the model can say so or look elsewhere.
fn refusal(url: &Url, status: u16) -> String {
    let host = url.host_str().unwrap_or_default();
    match status {
        // EDGAR asks automated tools for a contact email in their user agent; this app sends
        // no contact details in any request.
        403 if host == "sec.gov" || host.ends_with(".sec.gov") => format!(
            "{url}: HTTP 403: sec.gov serves automated tools only when they send a contact \
             email, which this app never sends; the user can open the filing in a browser and \
             attach it as a file"
        ),
        401 | 402 | 403 | 451 => format!(
            "{url}: HTTP {status}: the site refused to send the page, as a site does behind a \
             sign-in or a paywall, or to automated readers; the user can open it in a browser \
             and attach it as a file"
        ),
        _ => format!("{url}: HTTP {status}"),
    }
}

/// Words below which a page read as a document is likely not one but a sign-in, a paywall or a
/// viewer.
const THIN_WORDS: usize = 300;

/// Hosts that show a document only in their own viewer, whose page holds no document text.
const VIEWERS: [&str; 2] = ["docsend.com", "scribd.com"];

/// What the model should hear about `page`, a link `fetch_documents` read, when it may not be
/// the document: a viewer's page, or one with so few words that it is likely a sign-in, a
/// paywall or a viewer. None for a PDF, however short it is.
pub fn note(page: &Page) -> Option<String> {
    let url = Url::parse(&page.final_url).ok()?;
    let host = url.host_str()?.trim_start_matches("www.");
    if let Some(viewer) = VIEWERS
        .iter()
        .find(|v| host == **v || host.ends_with(&format!(".{v}")))
    {
        return Some(format!(
            "{viewer} shows documents only in its own viewer, so this text is likely the \
             viewer's page, not the document; the user can download the file and attach it."
        ));
    }
    if page.pdf {
        return None;
    }
    let words = page.text.split_whitespace().count();
    (words < THIN_WORDS).then(|| {
        format!(
            "The page has only {words} words of text, so it may be a sign-in, paywall or \
             viewer page rather than the document."
        )
    })
}

/// The day `body` says it was published, `YYYY-MM-DD`: an `article:published_time` or
/// `datePublished` (a meta tag or JSON-LD), a `<time datetime>`, else a date in `url`'s path
/// (`/2026/9/19/`).
pub fn published(body: &str, url: &str) -> Option<String> {
    static META: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r#"(?i)(?:article:published_time|datePublished|"pubdate"|name="date")["']?[^>]{0,40}?(?:content=|:)\s*["'](\d{4}-\d{2}-\d{2})"#,
        )
        .expect("a valid pattern")
    });
    static TIME: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"(?i)<time[^>]*\sdatetime=["'](\d{4}-\d{2}-\d{2})"#).expect("a valid pattern")
    });
    static PATH: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"/(20\d\d)[/-](\d{1,2})[/-](\d{1,2})(?:[/-]|$)").expect("a valid pattern")
    });
    let day = |y: &str, m: &str, d: &str| {
        let (y, m, d) = (y.parse().ok()?, m.parse().ok()?, d.parse().ok()?);
        dates::day(y, m, d).map(|_| format!("{y}-{m:02}-{d:02}"))
    };
    let iso = |re: &Regex, text: &str| {
        let c = re.captures(text)?;
        dates::iso(&c[1]).map(|_| c[1].to_owned())
    };
    (iso(&META, body).or_else(|| iso(&TIME, body)))
        .or_else(|| PATH.captures(url).and_then(|c| day(&c[1], &c[2], &c[3])))
}

/// The URLs a task has seen: in the user's messages, in web_search results and in fetched pages
/// (their links and final URLs). Once untrusted content has entered the task, fetch_page and
/// `fetch_seen` open only these, so a page cannot make the agent put data in a URL of its own.
#[derive(Clone, Debug, Default)]
pub struct Seen {
    /// Untrusted content has entered the task; before it, every URL opens.
    pub untrusted: bool,
    urls: HashSet<String>,
}

impl Seen {
    /// Adds each http and https URL written in `text`, and each word that starts "www." as an
    /// https one.
    pub fn read(&mut self, text: &str) {
        self.urls.extend(urls_in(text).filter_map(|u| normal(&u)));
    }

    /// Whether no URL is seen yet.
    pub fn is_empty(&self) -> bool {
        self.urls.is_empty()
    }

    /// Why `url` may not be fetched without the user's yes, or `None` when it may.
    pub fn refuses(&self, url: &str) -> Option<String> {
        let seen = normal(url).is_some_and(|u| self.urls.contains(&u));
        (self.untrusted && !seen).then(|| {
            format!(
                "'{url}' is not in the user's messages, a web_search result or a link of a page \
                 fetched in this chat; after untrusted content only those are fetched, so that a \
                 page cannot make the agent carry data out in a URL"
            )
        })
    }
}

/// `url` as the provenance check compares it: scheme and host lowercased and the default port
/// dropped (as `Url` parses), the fragment dropped; path and query exact, as data leaks there.
fn normal(url: &str) -> Option<String> {
    let mut url = Url::parse(url).ok()?;
    url.set_fragment(None);
    Some(url.into())
}

/// The http and https URLs written in `text`, and each word that starts "www." as an https one,
/// as a user writes a link. Each runs to whitespace, a quote, `<`, `>`, `]`, a backslash (JSON's
/// escapes) or a `)` it did not open, less a final `.`, `,`, `;`, `:`, `!`, `?` or `'` of the
/// sentence around it.
/// The longest URL taken from page text, as browsers and servers commonly cap them.
const MAX_URL: usize = 2048;

fn urls_in(text: &str) -> impl Iterator<Item = String> {
    let starts = text.match_indices("http").chain(text.match_indices("www."));
    starts.filter_map(|(at, _)| {
        // Read to `MAX_URL` at most: a page of `http://http://…` read each one to its end, a
        // quadratic scan of minutes on a 2 MB page.
        let rest = &text[at..];
        let rest = &rest[..rest.floor_char_boundary(MAX_URL + 1)];
        let www = rest.starts_with("www.");
        // The "www." of "https://www.a.org" is that URL's own.
        let word = (text[..at].chars().next_back())
            .is_none_or(|c| c.is_whitespace() || "(\"'<`[".contains(c));
        let url = rest.starts_with("http://") || rest.starts_with("https://");
        if !(url || www && word) {
            return None;
        }
        let mut depth = 0usize;
        let end = rest.find(|c: char| match c {
            '(' => {
                depth += 1;
                false
            }
            ')' if depth == 0 => true,
            ')' => {
                depth -= 1;
                false
            }
            c => c.is_whitespace() || matches!(c, '"' | '<' | '>' | ']' | '\\' | '`'),
        });
        // A run with no end within `MAX_URL` is no link.
        let end = end.or((rest.len() <= MAX_URL).then_some(rest.len()))?;
        let url = rest[..end].trim_end_matches(['.', ',', ';', ':', '!', '?', '\'']);
        Some(match www {
            true => format!("https://{url}"),
            false => url.to_owned(),
        })
    })
}

/// `fetch_many` for a skill that holds its task's `Seen`: a URL `seen` refuses is not fetched and
/// gets the reason, the others are fetched, and each page's final URL and links become seen.
/// `each` gets each result as `fetch_each` gives them, in the order of `urls`.
pub fn fetch_seen(
    urls: &[&str],
    max_bytes: usize,
    data_dir: &Path,
    seen: &mut Seen,
    each: impl FnMut(&Result<Page, ToolError>),
) -> Vec<Result<Page, ToolError>> {
    fetch_seen_as(urls, max_bytes, data_dir, seen, false, each)
}

/// `fetch_seen`, whose `each` gets, with `arrival`, the refusals first and then each page as it
/// comes in, whatever its place in `urls`.
pub fn fetch_seen_as(
    urls: &[&str],
    max_bytes: usize,
    data_dir: &Path,
    seen: &mut Seen,
    arrival: bool,
    each: impl FnMut(&Result<Page, ToolError>),
) -> Vec<Result<Page, ToolError>> {
    seen_each(urls, Want::page(max_bytes), data_dir, seen, arrival, each)
}

/// `fetch_seen` for read_papers and Folio, which read each link as a document: any body up to
/// `PDF_BYTES`, not a web page's `MAX_BYTES` (an EDGAR filing's HTML runs past it), and with
/// `blobs` (follow-ups on) a PDF's bytes kept as its blob, since the link can change or vanish.
pub fn fetch_documents(
    urls: &[&str],
    data_dir: &Path,
    seen: &mut Seen,
    blobs: bool,
) -> Vec<Result<Page, ToolError>> {
    let want = Want {
        bytes: PDF_BYTES,
        blobs,
    };
    seen_each(urls, want, data_dir, seen, false, |_| {})
}

/// `fetch_seen_as` for what `want` names.
fn seen_each(
    urls: &[&str],
    want: Want,
    data_dir: &Path,
    seen: &mut Seen,
    arrival: bool,
    mut each: impl FnMut(&Result<Page, ToolError>),
) -> Vec<Result<Page, ToolError>> {
    let why: Vec<Option<String>> = urls.iter().map(|url| seen.refuses(url)).collect();
    let open: Vec<&str> = (urls.iter().zip(&why))
        .filter(|(_, why)| why.is_none())
        .map(|(url, _)| *url)
        .collect();
    // The refusals up to the next URL fetched, which wait for no fetch.
    let refused = |left: &mut std::iter::Peekable<std::vec::IntoIter<Option<String>>>,
                   each: &mut dyn FnMut(&Result<Page, ToolError>)| {
        while let Some(Some(why)) = left.next_if(Option::is_some) {
            each(&Err(ToolError::Fetch(why)));
        }
    };
    let mut left = why.clone().into_iter().peekable();
    match arrival {
        // Every refusal first: the fetches' order is not the URLs'.
        true => (why.iter().flatten()).for_each(|why| each(&Err(ToolError::Fetch(why.clone())))),
        false => refused(&mut left, &mut each),
    }
    let fetched = fetch_each(&open, want, data_dir, arrival, |got| {
        if arrival {
            return each(got);
        }
        left.next();
        each(got);
        refused(&mut left, &mut each);
    });
    let mut pages = fetched.into_iter();
    let got: Vec<_> = (why.into_iter())
        .map(|why| match why {
            Some(why) => Err(ToolError::Fetch(why)),
            None => pages.next().expect("one result per URL fetched"),
        })
        .collect();
    for page in got.iter().flatten() {
        seen.read(&page.final_url);
        seen.read(&page.text);
    }
    got
}

/// The title and Markdown text of a body of type `kind` from `base`, without control characters.
pub fn convert(kind: &str, body: &[u8], base: &Url) -> Result<(String, String), ToolError> {
    // A page's ESC and C1 sequences would garble the prompts and the terminal. A bare CR is a
    // line end (old Mac text), so it becomes one before the strip.
    // A PDF's form feeds are its page breaks, which a follow-up cites by.
    let printable = |s: String, feeds: bool| -> String {
        (s.replace("\r\n", "\n").replace('\r', "\n").chars())
            .filter(|&c| !c.is_control() || matches!(c, '\n' | '\t') || (feeds && c == '\u{c}'))
            .collect()
    };
    let pdf = is_pdf(kind, body);
    readable(kind, body, base).map(|(title, text)| (printable(title, false), printable(text, pdf)))
}

/// Whether a body of type `kind` that starts with `body` is a PDF.
fn is_pdf(kind: &str, body: &[u8]) -> bool {
    kind == "application/pdf" || body.starts_with(b"%PDF-")
}

/// `convert`'s title and text, controls kept.
fn readable(kind: &str, body: &[u8], base: &Url) -> Result<(String, String), ToolError> {
    if is_pdf(kind, body) {
        return pdf(body);
    }
    // ponytail: UTF-8 only, others read lossily; honour the charset when such pages show up
    let text = String::from_utf8_lossy(body);
    let head = text
        .get(..text.floor_char_boundary(1024))
        .unwrap_or_default()
        .to_ascii_lowercase();
    if kind.contains("html")
        || (!kind.starts_with("text/")
            && (head.contains("<html") || head.contains("<!doctype html")))
    {
        Ok(super::extract::html(&text, base))
    } else if kind.starts_with("text/") {
        Ok((String::new(), text.into_owned()))
    } else {
        Err(ToolError::Fetch(format!(
            "{base}: {kind} is not a page fetch_page can read"
        )))
    }
}

/// A PDF's first line, as its title, and its text, by read_papers' reader (pdftotext when it is
/// installed, else the built-in one), which reads a file: each page tidied, with a form feed
/// between pages, so a page with no text stays one.
fn pdf(body: &[u8]) -> Result<(String, String), ToolError> {
    let failed = |e: String| ToolError::Fetch(format!("pdf: {e}"));
    let file = (tempfile::Builder::new().suffix(".pdf").tempfile())
        .and_then(|mut f| f.write_all(body).map(|()| f))
        .map_err(|e| failed(e.to_string()))?;
    let text = read_papers::text_of(file.path(), &AtomicBool::new(false)).map_err(failed)?;
    // Each page tidied alone, a form feed between them, so a page with no text stays one.
    let pages: Vec<String> = (source::between(&text).split('\u{c}'))
        .map(read_papers::tidy)
        .collect();
    let text = pages.join("\u{c}").trim_end_matches('\n').to_string();
    let title = text.lines().next().unwrap_or_default().to_string();
    Ok((title, text))
}

/// `url` over https, unless it is this machine, which has no certificate to show.
fn upgrade(mut url: Url) -> Url {
    let local = match url.host() {
        Some(Host::Ipv4(ip)) => IpAddr::V4(ip).is_loopback(),
        Some(Host::Ipv6(ip)) => ip.is_loopback(),
        Some(Host::Domain(name)) => name.eq_ignore_ascii_case("localhost"),
        None => false,
    };
    if url.scheme() == "http" && !local {
        url.set_scheme("https")
            .expect("http and https are both special schemes");
    }
    url
}

/// A redirect stays on its host, or goes over https.
fn hop_allowed(from: &Url, to: &Url) -> bool {
    to.scheme() == "https" || (to.scheme() == "http" && to.host() == from.host())
}

/// One GET in its host's turn; a status error is a response, for the caller to judge.
fn get(url: &Url) -> Result<ureq::Response, ToolError> {
    wait_turn(&site(url));
    let local = url.scheme() == "http";
    match AGENTS[usize::from(local)].request_url("GET", url).call() {
        Ok(resp) | Err(ureq::Error::Status(_, resp)) => Ok(resp),
        // One sentence for a timeout on every OS (Windows says only "os error 10060"); any
        // other error is ureq's, which names the URL already.
        Err(e) if timed_out(&e) => Err(ToolError::Fetch(format!(
            "{url}: no answer within {} s",
            TIMEOUT.as_secs()
        ))),
        Err(e) => Err(ToolError::Fetch(e.to_string())),
    }
}

/// Whether `e` comes from an I/O timeout: `TimedOut`, or the `WouldBlock` a socket read past
/// its timeout gives on Unix.
fn timed_out(e: &ureq::Error) -> bool {
    std::iter::successors(Some(e as &dyn std::error::Error), |e| e.source()).any(|e| {
        e.downcast_ref::<io::Error>().is_some_and(|e| {
            matches!(
                e.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            )
        })
    })
}

fn site(url: &Url) -> Arc<Site> {
    let mut sites = SITES.lock().unwrap_or_else(PoisonError::into_inner);
    let key = url.origin().ascii_serialization();
    (sites.entry(key))
        .or_insert_with(|| {
            Arc::new(Site {
                next: Mutex::new(Instant::now()),
                robots: Mutex::new(None),
            })
        })
        .clone()
}

/// Sleeps until the host's next free second and takes it.
fn wait_turn(site: &Site) {
    let slot = {
        let mut next = site.next.lock().unwrap_or_else(PoisonError::into_inner);
        let slot = (*next).max(Instant::now());
        *next = slot + Duration::from_secs(1);
        slot
    };
    thread::sleep(slot.saturating_duration_since(Instant::now()));
}

/// Whether the host's robots.txt lets us fetch `url`, or why not. It is read once per process;
/// one that could not be read refuses this fetch (RFC 9309: unreachable allows nothing) and is
/// read again by the next (arXiv's papers were refused as disallowed, which its robots.txt never
/// said: 14 Nov 2026).
fn allowed(url: &Url) -> Result<(), ToolError> {
    let asked = Instant::now();
    let site = site(url);
    let mut robots = site.robots.lock().unwrap_or_else(PoisonError::into_inner);
    // Fetches that waited on a read that failed share it rather than each wait out a timeout.
    let fresh = match &*robots {
        Some(Ok(_)) => true,
        Some(Err((_, at))) => *at >= asked,
        None => false,
    };
    if !fresh {
        *robots = Some(robots_of(url).map_err(|e| (e, Instant::now())));
    }
    let path = match url.query() {
        Some(q) => format!("{}?{q}", url.path()),
        None => url.path().into(),
    };
    match robots.as_ref().expect("read above") {
        Ok(rules) if robots_allow(rules, &path) => Ok(()),
        Ok(_) => Err(ToolError::Fetch(format!(
            "{url}: robots.txt does not allow it"
        ))),
        Err((why, _)) => Err(ToolError::Fetch(format!(
            "{url}: {}'s robots.txt could not be read: {why}",
            url.host_str().unwrap_or_default()
        ))),
    }
}

/// The rules of `url`'s host's robots.txt: none when it is missing (4xx), why not when it is
/// unreachable (5xx, no answer, a body cut off).
fn robots_of(url: &Url) -> Result<Vec<(bool, String)>, String> {
    let mut robots = url.clone();
    robots.set_path("/robots.txt");
    robots.set_query(None);
    robots.set_fragment(None);
    // ponytail: a redirected robots.txt is read as missing; follow it if hosts show up that redirect it
    let resp = get(&robots).map_err(|e| e.to_string())?;
    match resp.status() {
        200..300 => {
            let mut bytes = Vec::new();
            (resp.into_reader().take(ROBOTS_BYTES))
                .read_to_end(&mut bytes)
                .map_err(|e| format!("{robots}: {e}"))?;
            Ok(robots_rules(&String::from_utf8_lossy(&bytes)))
        }
        300..500 => Ok(Vec::new()),
        status => Err(format!("{robots}: HTTP {status}")),
    }
}

/// The Allow (`true`) and Disallow rules of robots.txt's groups for `AGENT`, or for `*` when none
/// names it (RFC 9309: groups naming the same agent merge).
fn robots_rules(text: &str) -> Vec<(bool, String)> {
    let (mut ours, mut anyone) = (Vec::new(), Vec::new());
    let mut agents: Vec<String> = Vec::new();
    let (mut in_rules, mut named) = (false, false);
    for line in text.trim_start_matches('\u{feff}').lines() {
        let line = line.split('#').next().unwrap_or_default();
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let (key, value) = (key.trim().to_ascii_lowercase(), value.trim());
        match key.as_str() {
            "user-agent" => {
                if in_rules {
                    agents.clear();
                    in_rules = false;
                }
                agents.push(value.to_ascii_lowercase());
                named |= agents.last().is_some_and(|a| a == AGENT);
            }
            "allow" | "disallow" => {
                in_rules = true;
                if value.is_empty() {
                    continue;
                }
                let rule = (key == "allow", value.to_string());
                if agents.iter().any(|a| a == AGENT) {
                    ours.push(rule.clone());
                }
                if agents.iter().any(|a| a == "*") {
                    anyone.push(rule);
                }
            }
            _ => {}
        }
    }
    if named { ours } else { anyone }
}

/// The longest rule matching `path` decides; Allow wins a tie, and no match allows.
// ponytail: no percent-encoding normalisation of rules or path; add when a site's rules use it
fn robots_allow(rules: &[(bool, String)], path: &str) -> bool {
    (rules.iter())
        .filter(|(_, pattern)| robots_match(pattern, path))
        .max_by_key(|(allow, pattern)| (pattern.len(), *allow))
        .is_none_or(|(allow, _)| *allow)
}

/// RFC 9309 matching: `*` is any run of characters, a final `$` ends the path.
fn robots_match(pattern: &str, path: &str) -> bool {
    let (pattern, anchored) = match pattern.strip_suffix('$') {
        Some(p) => (p, true),
        None => (pattern, false),
    };
    let mut parts = pattern.split('*');
    let first = parts.next().unwrap_or_default();
    let Some(mut rest) = path.strip_prefix(first) else {
        return false;
    };
    let parts: Vec<&str> = parts.collect();
    for (i, part) in parts.iter().enumerate() {
        if anchored && i + 1 == parts.len() {
            return rest.ends_with(part);
        }
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    !anchored || rest.is_empty()
}

/// Where the page asked for at `url` whose body hashed to `sha256` is saved:
/// `agent/web/<URL hash>/<content hash>.json`.
pub(crate) fn saved_at(url: &str, sha256: &str, data_dir: &Path) -> PathBuf {
    let key = larch::bench::hex(&larch::bench::sha256(url.as_bytes()));
    let dir = data_dir.join("agent").join("web").join(&key[..16]);
    dir.join(format!("{sha256}.json"))
}

/// The copy `fetch_many` saved of the page asked for at `url` whose body hashed to `sha256`.
pub fn cached(url: &str, sha256: &str, data_dir: &Path) -> Result<Page, ToolError> {
    let path = saved_at(url, sha256, data_dir);
    let failed = |e: io::Error| ToolError::Io(path.display().to_string(), e);
    let json = fs::read(&path).map_err(failed)?;
    serde_json::from_slice(&json).map_err(|e| failed(e.into()))
}

/// Writes `page` to `saved_at`, whole, through a temporary file.
pub(crate) fn save(page: &Page, data_dir: &Path) -> Result<(), ToolError> {
    let path = saved_at(&page.url, &page.sha256, data_dir);
    let dir = path.parent().expect("a file in the URL's folder");
    let failed = |e: io::Error| ToolError::Io(path.display().to_string(), e);
    fs::create_dir_all(dir).map_err(failed)?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir).map_err(failed)?;
    let json = serde_json::to_vec(page).map_err(|e| failed(e.into()))?;
    tmp.write_all(&json).map_err(failed)?;
    tmp.persist(&path).map_err(|e| failed(e.error))?;
    Ok(())
}

/// The `fetch_page` tool.
pub(crate) struct FetchPage;

impl Tool for FetchPage {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "fetch_page".into(),
            description:
                "Fetch a web page by URL; returns its title and main text as Markdown, links kept."
                    .into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_bytes": {"type": "integer", "minimum": 1, "maximum": MAX_BYTES}
                },
                "required": ["url"],
                "additionalProperties": false
            }),
            permission: Permission::Read,
            untrusted_results: true,
            // Never cut by `tool::run`, which would break the JSON: `call` cuts the text.
            max_result_tokens: u32::MAX,
            max_args_tokens: ARGS_TOKENS,
        }
    }

    /// The page as JSON, its text cut to `TEXT_BYTES`.
    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        let url = str_arg(args, "url")?;
        let max = whole_arg(args, "max_bytes").map_or(MAX_BYTES, |n| (n as usize).min(MAX_BYTES));
        let mut page =
            (fetch_many(&[url], max, &ctx.data_dir).pop()).expect("one URL, one result")?;
        if page.text.len() > TEXT_BYTES {
            page.text
                .truncate(page.text.floor_char_boundary(TEXT_BYTES));
            page.truncated = true;
        }
        serde_json::to_string(&page).map_err(|e| ToolError::Fetch(e.to_string()))
    }

    fn opens<'a>(&self, args: &'a Value) -> Option<&'a str> {
        args["url"].as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_score_laid_out_one_cell_a_line_reads_as_one_line() {
        let page =
            "1. Sat 24 Oct\n   Aston Villa\n   2\n   1\n   Manchester City\n   - [Report](x)\n";
        assert_eq!(
            scores_joined(page),
            "1. Sat 24 Oct\n   Aston Villa 2-1 Manchester City\n   - [Report](x)\n"
        );
        // One number alone, a year, or three numbers in a row are no score.
        for other in [
            "Gate\n4\nOpen",
            "Year\n2026\n10\nTotal",
            "a\n1\n2\n3\nb",
            "Sinner\n6\n6\nMedvedev\n4\n3",
            "$\n12\n99\nUSD",
            "Price: $\n12\n99\nUSD",
            "Page 1 of 20\n3\n4\nNext",
        ] {
            assert_eq!(scores_joined(other), other, "{other:?}");
        }
    }

    #[test]
    fn a_page_says_when_it_was_published_or_its_address_does() {
        let sep19 = Some("2026-09-19".to_owned());
        let utsa = "https://texaslonghorns.com/news/2026/9/19/no-1-1-football-downs-utsa-30-6";
        assert_eq!(published("<html></html>", utsa), sep19);
        let other = "https://a.example/x";
        let meta = r#"<meta property="article:published_time" content="2026-09-19T22:10:00Z">"#;
        assert_eq!(published(meta, other), sep19);
        let ld = r#"<script type="application/ld+json">{"datePublished": "2026-09-19"}</script>"#;
        assert_eq!(published(ld, other), sep19);
        let time = r#"<p>Posted <time class="t" datetime="2026-09-19T08:00">Sat</time></p>"#;
        assert_eq!(published(time, other), sep19);
        // The page's own date wins over its address's.
        assert_eq!(published(meta, "https://a.example/2020/01/02/x"), sep19);
        // No such day, a year alone, no date at all.
        assert_eq!(published("", "https://a.example/2026/13/40/x"), None);
        assert_eq!(published("", "https://a.example/news/2026/x"), None);
        assert_eq!(published("<p>Saturday</p>", other), None);
    }

    fn url(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    #[test]
    fn a_refusal_says_in_plain_words_what_the_site_wants() {
        let edgar = refusal(&url("https://www.sec.gov/Archives/x.htm"), 403);
        assert!(
            edgar.contains("contact email, which this app never sends"),
            "{edgar}"
        );
        let paywall = refusal(&url("https://a.example/x"), 401);
        assert!(paywall.contains("HTTP 401: the site refused"), "{paywall}");
        assert_eq!(
            refusal(&url("https://a.example/x"), 404),
            "https://a.example/x: HTTP 404"
        );
    }

    #[test]
    fn a_viewer_or_a_page_of_few_words_is_named_as_maybe_not_the_document() {
        let page = |at: &str, text: String| Page {
            url: at.into(),
            final_url: at.into(),
            fetched_at: String::new(),
            sha256: "ab".into(),
            title: String::new(),
            text,
            bytes: 0,
            truncated: false,
            published: None,
            pdf: false,
        };
        let deck = page("https://docsend.com/view/abc", "word ".repeat(400));
        let said = note(&deck).unwrap();
        assert!(
            said.starts_with("docsend.com shows documents only in its own viewer"),
            "{said}"
        );
        let scribd = page("https://www.scribd.com/document/1/x", "word ".repeat(400));
        assert!(note(&scribd).unwrap().starts_with("scribd.com"));
        let thin = page("https://a.example/x", "Subscribe to read the rest.".into());
        let said = note(&thin).unwrap();
        assert!(
            said.starts_with("The page has only 5 words of text"),
            "{said}"
        );
        assert_eq!(
            note(&page("https://a.example/x", "word ".repeat(300))),
            None
        );
        // A short PDF is a document.
        assert_eq!(note(&Page { pdf: true, ..thin }), None);
    }

    #[test]
    fn a_web_pdf_keeps_its_page_breaks_and_an_empty_page() {
        let pdf = source::tests::pdf(&["Sparse experts", "", "Appendix A"]);
        let (title, text) = convert("application/pdf", &pdf, &url("https://a.example/p")).unwrap();
        assert_eq!(title, "Sparse experts");
        assert_eq!(text, "Sparse experts\n\u{c}\u{c}Appendix A");
    }

    #[test]
    fn http_becomes_https_except_on_this_machine() {
        assert_eq!(
            upgrade(url("http://example.org/a?b")).as_str(),
            "https://example.org/a?b"
        );
        for local in ["http://127.0.0.1:8/", "http://localhost/", "http://[::1]/"] {
            assert_eq!(upgrade(url(local)).as_str(), local);
        }
    }

    #[test]
    fn a_redirect_may_change_host_only_to_https() {
        let from = url("https://a.org/x");
        assert!(hop_allowed(&from, &url("https://b.org/y")));
        assert!(hop_allowed(&from, &url("http://a.org/y")));
        assert!(!hop_allowed(&from, &url("http://b.org/y")));
        assert!(!hop_allowed(&from, &url("ftp://a.org/y")));
    }

    #[test]
    fn robots_rules_pick_our_group_and_the_longest_match() {
        let text = "User-agent: *\nDisallow: /\n\n# ours\nUser-agent: other\nUser-agent: Larch\n\
                    Disallow: /a # a comment\nAllow: /a/b\nDisallow: /*.pdf$\nDisallow:\n";
        let rules = robots_rules(text);
        assert_eq!(rules.len(), 3);
        assert!(robots_allow(&rules, "/"));
        assert!(!robots_allow(&rules, "/a/c"));
        assert!(robots_allow(&rules, "/a/b/c"));
        assert!(!robots_allow(&rules, "/x/y.pdf"));
        assert!(robots_allow(&rules, "/x/y.pdf?v=1"));
        // A group for everyone, when none names us.
        let rules = robots_rules("User-agent: *\nDisallow: /private\n");
        assert!(!robots_allow(&rules, "/private/x") && robots_allow(&rules, "/public"));
        // A byte-order mark is not part of the first key.
        let rules = robots_rules("\u{feff}User-agent: *\nDisallow: /private\n");
        assert!(!robots_allow(&rules, "/private/x"));
        // Allow wins a tie.
        assert!(robots_allow(
            &[(false, "/p".into()), (true, "/p".into())],
            "/p"
        ));
    }

    #[test]
    fn each_turn_on_a_host_comes_a_second_after_the_last() {
        let site = Site {
            next: Mutex::new(Instant::now()),
            robots: Mutex::new(None),
        };
        let start = Instant::now();
        for n in 0..3 {
            wait_turn(&site);
            assert!(
                start.elapsed() >= Duration::from_secs(n),
                "turn {n}: {:?}",
                start.elapsed()
            );
        }
    }

    /// A run past `MAX_URL` with no end is no link, and a page of them is scanned in linear time:
    /// each start reads at most `MAX_URL`, not to the page's end.
    #[test]
    fn a_run_longer_than_a_url_is_no_link() {
        let long = format!("https://a.org/{}", "x".repeat(MAX_URL));
        assert_eq!(urls_in(&long).count(), 0);
        let fits = format!("https://a.org/{} next", "x".repeat(100));
        assert_eq!(
            urls_in(&fits).collect::<Vec<_>>(),
            [&fits[..fits.len() - 5]]
        );
        // Only the starts within `MAX_URL` of the page's end reach an end: the text's.
        // 60,000 starts read at most 123M chars, about 4 s in a debug build; read to the page's
        // end, they would read 100 times that. 300,000 starts took over 20 s in a debug build on
        // a hosted CI Mac.
        let hostile = "http://".repeat(60_000);
        let started = std::time::Instant::now();
        let n = urls_in(&hostile).count();
        assert!(n <= MAX_URL / "http://".len() + 1, "{n}");
        assert!(started.elapsed() < std::time::Duration::from_secs(20));
    }

    #[test]
    fn urls_are_found_in_prose_markdown_and_json() {
        let text = "See https://a.org/x. Or [https://a.org/y](https://a.org/y), \
            [w](https://w.org/wiki/Rust_(language)) and {\"url\":\"https://b.org/z?q=1\"}\\nhttp://c.org";
        assert_eq!(
            urls_in(text).collect::<Vec<_>>(),
            [
                "https://a.org/x",
                "https://a.org/y",
                "https://a.org/y",
                "https://w.org/wiki/Rust_(language)",
                "https://b.org/z?q=1",
                "http://c.org"
            ]
        );
    }

    #[test]
    fn a_www_link_a_user_writes_is_seen_as_its_https_url() {
        let mut seen = Seen {
            untrusted: true,
            ..Seen::default()
        };
        seen.read("summarize www.paulgraham.com/startupideas.html, please");
        let url = "https://www.paulgraham.com/startupideas.html";
        assert_eq!(seen.refuses(url), None);
        // The "www." of a URL written whole is that URL's own.
        seen.read("see http://www.a.org/x");
        assert!(seen.refuses("https://www.a.org/x").is_some());
    }

    #[test]
    fn after_untrusted_content_only_a_seen_url_opens_compared_normalised() {
        let mut seen = Seen::default();
        seen.read("look at https://Example.ORG:443/a?b=1#top");
        assert_eq!(seen.refuses("https://evil.example/?d=secret"), None);
        seen.untrusted = true;
        assert_eq!(seen.refuses("https://example.org/a?b=1"), None);
        assert_eq!(seen.refuses("HTTPS://EXAMPLE.org:443/a?b=1#other"), None);
        for unseen in [
            "https://example.org/a?b=1&d=secret",
            "https://example.org/a",
            "https://example.org/a?b=2",
            "http://example.org/a?b=1",
            "https://evil.example/?d=secret",
            "not a url",
        ] {
            let why = seen.refuses(unseen).expect(unseen);
            assert!(
                why.contains(unseen) && why.contains("carry data out"),
                "{why}"
            );
        }
    }

    // This binary never calls `allow_this_machine`, so loopback is refused here.
    #[test]
    fn this_machine_and_its_networks_are_refused_with_the_reason() {
        let dir = tempfile::tempdir().unwrap();
        let urls = [
            "http://127.0.0.1:9/",
            "http://[::1]:9/",
            "http://10.1.2.3/",
            "http://192.168.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::ffff:10.0.0.1]/",
            "http://[fd00::1]/",
            "http://[fe80::1]/",
            "http://0.0.0.0/",
            "http://100.64.0.1/",
            "http://100.127.255.255/",
            // A name, not an address: it resolves to loopback.
            "http://localhost:9/",
        ];
        for (url, got) in urls.iter().zip(fetch_many(&urls, MAX_BYTES, dir.path())) {
            let e = got.expect_err(url).to_string();
            assert!(e.contains("which fetch_page never opens"), "{url}: {e}");
        }
    }

    #[test]
    fn carrier_grade_nat_is_private_and_its_neighbours_are_not() {
        let ip = |s: &str| s.parse::<IpAddr>().unwrap();
        assert!(private(ip("100.100.100.200")).is_some());
        assert!(private(ip("100.63.255.255")).is_none());
        assert!(private(ip("100.128.0.1")).is_none());
    }

    #[test]
    fn through_a_proxy_a_name_that_does_not_resolve_goes_to_it_but_addresses_and_localhost_do_not()
    {
        let check = |s: &str, proxied| refused(&url(s), proxied);
        // RFC 2606's .invalid never resolves here: the proxy may resolve it, a direct fetch cannot.
        assert!(check("https://iris-proxy-test.invalid/", true).is_ok());
        assert!(check("https://iris-proxy-test.invalid/", false).is_err());
        for local in [
            "https://10.0.0.1/",
            "https://[::1]/",
            "https://100.64.1.1/",
            "https://169.254.169.254/",
            "https://localhost/",
            "https://LocalHost./",
            "https://a.b.localhost/",
        ] {
            let e = check(local, true).expect_err(local).to_string();
            assert!(e.contains("never opens"), "{local}: {e}");
        }
    }

    #[test]
    fn the_resolver_refuses_a_name_with_any_private_address() {
        let e = public("localhost:80").unwrap_err().to_string();
        assert!(e.contains("this machine"), "{e}");
        assert!(public("93.184.215.14:443").is_ok());
        // RFC 2606's .invalid never resolves: unchecked, so refused.
        let e = public("larch.invalid:443").unwrap_err().to_string();
        assert!(e.contains("cannot be checked"), "{e}");
    }

    #[test]
    fn robots_wildcards_match_runs_and_the_end() {
        assert!(robots_match("/a*c", "/abbc/d"));
        assert!(robots_match("/a*c$", "/abbc"));
        assert!(!robots_match("/a*c$", "/abbcd"));
        assert!(robots_match("/a$", "/a") && !robots_match("/a$", "/ab"));
        assert!(robots_match("*", "/anything"));
        assert!(!robots_match("/b", "/a"));
    }
}
