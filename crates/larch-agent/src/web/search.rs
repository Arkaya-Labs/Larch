//! `web_search`: ranked results from the first provider in the chain that answers.

use std::{
    fmt,
    fs::{self, OpenOptions},
    hash::{BuildHasher, Hasher, RandomState},
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, sleep},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    config,
    memory::dates,
    skills::web_answer,
    tool::{Builtin, Tool, ToolCtx, ToolError, str_arg, whole_arg},
};

/// A search provider. Keyed ones come first, as a key moves its provider ahead of the free ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    /// Brave Search API (JSON, key).
    Brave,
    /// Serper's Google results (JSON, key).
    Serper,
    /// Tavily (JSON, key).
    Tavily,
    /// Exa (JSON, key).
    Exa,
    /// html.duckduckgo.com.
    DuckDuckGoHtml,
    /// lite.duckduckgo.com.
    DuckDuckGoLite,
    /// mojeek.com.
    Mojeek,
}

/// One result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hit {
    /// The page's title.
    pub title: String,
    /// The page's address.
    pub url: String,
    /// The text around the match, as the provider gives it.
    pub snippet: String,
}

impl Provider {
    /// The chain's order.
    const ALL: [Provider; 7] = [
        Self::Brave,
        Self::Serper,
        Self::Tavily,
        Self::Exa,
        Self::DuckDuckGoHtml,
        Self::DuckDuckGoLite,
        Self::Mojeek,
    ];

    /// The provider whose config.toml key is `name` (`brave_key` and so on).
    pub fn for_key(name: &str) -> Option<Provider> {
        Self::ALL[..4]
            .iter()
            .copied()
            .find(|p| name == format!("{}_key", format!("{p:?}").to_lowercase()))
    }

    fn url(self) -> String {
        match self {
            Self::Brave => "https://api.search.brave.com/res/v1/web/search",
            Self::Serper => "https://google.serper.dev/search",
            Self::Tavily => "https://api.tavily.com/search",
            Self::Exa => "https://api.exa.ai/search",
            Self::DuckDuckGoHtml => "https://html.duckduckgo.com/html/",
            Self::DuckDuckGoLite => "https://lite.duckduckgo.com/lite/",
            Self::Mojeek => "https://www.mojeek.com/search",
        }
        .into()
    }
}

/// A provider's key from config.toml. Its `Debug` hides it, and it only ever goes in a request
/// header, never a URL, which ureq's errors quote.
#[derive(Clone, PartialEq, Eq)]
pub struct Key(pub(crate) String);

impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Key(hidden)")
    }
}

/// Who is asking, and where to read why; never a browser's.
const USER_AGENT: &str = concat!(
    "larch-iris/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/Arkaya-Labs/Larch)"
);
/// Results a call returns when it does not say.
const N: u64 = 5;
const MAX_N: u64 = 10;
/// A snippet is cut at this many bytes: Tavily's and Exa's run to paragraphs.
const SNIPPET: usize = 300;
const CACHE_FOR: Duration = Duration::from_secs(24 * 60 * 60);
/// A search for this week's results (`recent`) is cached this long: its answer changes by the hour.
const RECENT_FOR: Duration = Duration::from_secs(60 * 60);
const DOWN_FOR: Duration = Duration::from_secs(15 * 60);
/// The least time between two requests of one install, before jitter.
const GAP: Duration = Duration::from_secs(1);
const JITTER_MS: u64 = 250;
/// A provider that has not answered in this long is skipped for this call.
const TIMEOUT: Duration = Duration::from_secs(10);

/// The `web_search` tool.
pub(crate) fn tool() -> Box<dyn Tool> {
    Builtin::untrusted(
        "web_search",
        "Search the web; returns ranked results, each with its title, url and snippet.",
        1500,
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "n": {"type": "integer", "minimum": 1, "maximum": MAX_N},
                "recent": {"type": "string", "enum": ["week"]}
            },
            "required": ["query"]
        }),
        |args, ctx| call_at(args, ctx, Provider::url),
    )
}

/// A web_search call with each provider at `url`'s address for it.
fn call_at(
    args: &Value,
    ctx: &ToolCtx,
    url: impl Fn(Provider) -> String,
) -> Result<String, ToolError> {
    let query = str_arg(args, "query")?;
    let n = whole_arg(args, "n").unwrap_or(N).min(MAX_N) as usize;
    let recent = args["recent"] == "week";
    // Only where: the reason can quote the line, and the line can hold a key.
    let config = config::load(&ctx.data_dir).map_err(|e| {
        let at = match e.line {
            0 => e.path.display().to_string(),
            line => format!("{}:{line}", e.path.display()),
        };
        ToolError::Search(format!(
            "{at} does not read, so web_search cannot see its keys; fix it in Settings"
        ))
    })?;
    let dir = ctx.data_dir.join("agent").join("search");
    let mut hits = search(&dir, query, &config.search_keys, recent, url)?;
    hits.truncate(n);
    Ok(serde_json::to_string(&hits).expect("strings serialize"))
}

/// The results for `query` from the first provider in the chain that has them cached or answers
/// with some: the keyed ones, then the free ones, each tried once. One provider's "no results"
/// moves on to the next, and is never cached, so it cannot hide a page for a day. `url` gives a
/// provider's address. With `recent`, each provider is asked for the past week's results only,
/// and the result is cached for `RECENT_FOR`; Mojeek, which has no such filter, is skipped. Each search leaves a line in `log.jsonl`.
/// With `web_answer::race`, the providers no cache answers are asked two at a time (`chain`).
fn search(
    dir: &Path,
    query: &str,
    keys: &[(Provider, Key)],
    recent: bool,
    url: impl Fn(Provider) -> String,
) -> Result<Vec<Hit>, ToolError> {
    chain(dir, query, keys, recent, url, web_answer::race())
}

/// `search`, with `race` given rather than read from the environment.
/// Starts loading the OS's trusted certificates on a thread of its own, once per process. ureq
/// loads them the first time any HTTP agent is built, and every other request waits for that
/// load: on macOS, which reads them from the keychain, it took 3.5-5 s on the M5 CI runner
/// (5 Nov), so a chat's first search paid it. A chat calls this when it opens, well before its
/// first search.
pub fn warm() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = thread::Builder::new()
            .name("larch-tls-warm".into())
            .spawn(|| drop(larch_fetch::agent("https://example.com")));
    });
}

fn chain(
    dir: &Path,
    query: &str,
    keys: &[(Provider, Key)],
    recent: bool,
    url: impl Fn(Provider) -> String,
    race: bool,
) -> Result<Vec<Hit>, ToolError> {
    let io = |what: &str| {
        let what = format!("search {what}");
        move |e| ToolError::Io(what, e)
    };
    let (mut reasons, mut empty) = (Vec::new(), false);
    // The free providers take turns by query, so a research round's dozen queries do not all
    // reach DuckDuckGo, which rate-limits them (a user's Mac, 27 Oct). The turn and the cache read
    // the query in lower case with single spaces, so "What is" and "what  is" are one search.
    let (keyed, free) = Provider::ALL.split_at(4);
    let mut same = query
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let turn = same.bytes().map(usize::from).sum::<usize>() % free.len();
    if recent {
        same.push_str("\0week");
    }
    let age = if recent { RECENT_FOR } else { CACHE_FOR };
    let order = keyed.iter().chain(&free[turn..]).chain(&free[..turn]);
    // With `race`, the providers left after the cache reads are asked two at a time.
    let mut ready = Vec::new();
    for &p in order {
        let key = keys.iter().find(|(k, _)| *k == p).map(|(_, k)| k);
        if key.is_none() && keyed.contains(&p) {
            continue;
        }
        if recent && p == Provider::Mojeek {
            reasons.push(format!("{p:?}: no filter for the week"));
            continue;
        }
        let name = format!("{p:?}");
        let cached = dir.join("cache").join(cache_name(&name, &same));
        // An empty list cached before empties stopped being cached reads as not cached.
        if let Some(hits) = (fresh(&cached, age).filter(|_| keeps(p)))
            .and_then(|t| serde_json::from_str::<Vec<Hit>>(&t).ok())
            .filter(|hits| !hits.is_empty())
        {
            note(dir, query, recent, Some(&name), "cached", &hits, &reasons).map_err(io("log"))?;
            return Ok(hits);
        }
        let down = dir.join("down").join(&name);
        if let Some(reason) = fresh(&down, DOWN_FOR) {
            reasons.push(format!("{name}: {reason}"));
            continue;
        }
        // A key is paid for: it is asked alone, before any free provider, as without the race.
        if race && !keyed.contains(&p) {
            ready.push((p, key));
            continue;
        }
        throttle(dir, "last").map_err(io("throttle"))?;
        let got = fetch(p, &url(p), query, recent, key)
            .and_then(|body| parse(p, &body).map_err(Fail::Down));
        let mut state = (&mut reasons, &mut empty);
        if let Some(hits) = settle(dir, (query, &same, recent), p, got, &mut state)? {
            return Ok(hits);
        }
    }
    // The first good list of two providers asked at once is the answer; the slower one's request
    // runs on, unread. Each provider keeps its own gap, so the two do not wait for each other.
    // DuckDuckGo's two pages are one site: a pair is never both of them while Mojeek can take
    // the second place, and they share one gap.
    let ddg = |p: Provider| matches!(p, Provider::DuckDuckGoHtml | Provider::DuckDuckGoLite);
    if ready.len() == 3 && ddg(ready[0].0) && ddg(ready[1].0) {
        ready.swap(1, 2);
    }
    for pair in ready.chunks(2) {
        let (tx, rx) = mpsc::channel();
        // Set once a list is kept: a request still waiting on its gap is then never sent.
        let won = Arc::new(AtomicBool::new(false));
        for &(p, key) in pair {
            let (tx, url, query, key, dir, won) = (
                tx.clone(),
                url(p),
                query.to_owned(),
                key.cloned(),
                dir.to_owned(),
                won.clone(),
            );
            let gap = match ddg(p) {
                true => "last-DuckDuckGo".to_owned(),
                false => format!("last-{p:?}"),
            };
            thread::spawn(move || {
                let first = || Err(Fail::Skip("another provider answered first".into()));
                // A request skipped before its gap takes no turn of the gap.
                let got = match won.load(Ordering::Relaxed) {
                    true => first(),
                    false => throttle(&dir, &gap)
                        .map_err(|e| Fail::Skip(format!("throttle: {e}")))
                        .and_then(|()| match won.load(Ordering::Relaxed) {
                            true => first(),
                            false => fetch(p, &url, &query, recent, key.as_ref()),
                        }),
                };
                let got = got.and_then(|body| parse(p, &body).map_err(Fail::Down));
                // Each provider marks its own failure, as no one may read it once a list is
                // kept: a rate limit then rests it like any other.
                if let Err(Fail::Down(reason)) = &got {
                    let down = dir.join("down");
                    let _ = fs::create_dir_all(&down)
                        .and_then(|()| fs::write(down.join(format!("{p:?}")), reason));
                }
                let _ = tx.send((p, got));
            });
        }
        drop(tx);
        for (p, got) in rx {
            let mut state = (&mut reasons, &mut empty);
            if let Some(hits) = settle(dir, (query, &same, recent), p, got, &mut state)? {
                won.store(true, Ordering::Relaxed);
                return Ok(hits);
            }
        }
    }
    // Some provider answered and none had a result: the web has none, which is not a failure.
    let status = if empty { "empty" } else { "failed" };
    note(dir, query, recent, None, status, &[], &reasons).map_err(io("log"))?;
    if empty {
        return Ok(Vec::new());
    }
    let list = reasons.join("; ");
    let limited = reasons.iter().any(|r| r.contains(": rate-limited"));
    Err(ToolError::Search(match (limited, keys.is_empty()) {
        (true, true) => {
            format!("search is rate-limited; add a key in Settings to lift it ({list})")
        }
        (true, false) => format!("search is rate-limited ({list})"),
        (false, _) => format!("search failed ({list})"),
    }))
}

/// Whether `p`'s results may be cached: Brave's API terms (3(b)(i)) allow only transient storage.
fn keeps(p: Provider) -> bool {
    p != Provider::Brave
}

/// What one provider's answer `got` to `query` (`same` as the cache reads it, `recent` for the
/// week) gives: the hits, cached and logged; or none, with `state`'s reasons told why (and
/// whether it answered with no results), and a provider that failed marked down.
fn settle(
    dir: &Path,
    (query, same, recent): (&str, &str, bool),
    p: Provider,
    got: Result<Vec<Hit>, Fail>,
    (reasons, empty): &mut (&mut Vec<String>, &mut bool),
) -> Result<Option<Vec<Hit>>, ToolError> {
    let io = |what: &str| {
        let what = format!("search {what}");
        move |e| ToolError::Io(what, e)
    };
    let name = format!("{p:?}");
    match got {
        Ok(hits) if hits.is_empty() => {
            **empty = true;
            reasons.push(format!("{name}: no results"));
        }
        Ok(hits) => {
            if keeps(p) {
                let cached = dir.join("cache").join(cache_name(&name, same));
                fs::create_dir_all(dir.join("cache")).map_err(io("cache"))?;
                let json = serde_json::to_string(&hits).expect("strings serialize");
                fs::write(&cached, json).map_err(io("cache"))?;
                for old in fs::read_dir(dir.join("cache")).map_err(io("cache"))? {
                    let old = old.map_err(io("cache"))?.path();
                    if fresh(&old, CACHE_FOR).is_none() {
                        fs::remove_file(old).map_err(io("cache"))?;
                    }
                }
            }
            note(dir, query, recent, Some(&name), "ok", &hits, reasons).map_err(io("log"))?;
            return Ok(Some(hits));
        }
        Err(Fail::Down(reason)) => {
            fs::create_dir_all(dir.join("down")).map_err(io("state"))?;
            fs::write(dir.join("down").join(&name), &reason).map_err(io("state"))?;
            reasons.push(format!("{name}: {reason}"));
        }
        Err(Fail::Skip(reason)) => reasons.push(format!("{name}: {reason}")),
    }
    Ok(None)
}

/// The day seven days before today (UTC) as Exa's start date takes it.
fn week_ago() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let day = (now.as_secs() / 86_400) as i64 - 7;
    let (y, m) = dates::year_month(day);
    let d = day - dates::day(y, m, 1).expect("a month's first day") + 1;
    format!("{y:04}-{m:02}-{d:02}T00:00:00.000Z")
}

/// Why a provider gave no results: `Down` marks it down for `DOWN_FOR`; `Skip` fails this call
/// only, as no connection or a refused key may be fixed before the next.
enum Fail {
    Down(String),
    Skip(String),
}

/// One request to `p` at `url`, never retried: a refusal moves the search to the next provider.
// ponytail: ureq strips only Authorization on a redirect, so a keyed provider that redirects sends
// X-API-KEY on; give larch_fetch::agent a no-redirect option when one is seen redirecting
/// With `recent`, the request asks for the past week's results, in the provider's own words.
fn fetch(
    p: Provider,
    url: &str,
    query: &str,
    recent: bool,
    key: Option<&Key>,
) -> Result<String, Fail> {
    let agent = larch_fetch::agent(url);
    let key = key.map_or("", |k| &k.0);
    let (mut request, mut body) = match p {
        Provider::Brave => (
            (agent.get(url).set("X-Subscription-Token", key))
                .set("Accept", "application/json")
                .query("q", query)
                .query("count", &MAX_N.to_string()),
            None,
        ),
        Provider::Serper => (
            agent.post(url).set("X-API-KEY", key),
            Some(json!({"q": query, "num": MAX_N})),
        ),
        Provider::Tavily => (
            agent
                .post(url)
                .set("Authorization", &format!("Bearer {key}")),
            Some(json!({"query": query, "max_results": MAX_N})),
        ),
        Provider::Exa => (
            agent.post(url).set("x-api-key", key),
            Some(json!({"query": query, "numResults": MAX_N,
                "contents": {"text": {"maxCharacters": SNIPPET}}})),
        ),
        Provider::DuckDuckGoHtml | Provider::DuckDuckGoLite | Provider::Mojeek => {
            (agent.get(url).query("q", query), None)
        }
    };
    if recent {
        match (p, &mut body) {
            (Provider::Brave, _) => request = request.query("freshness", "pw"),
            (Provider::DuckDuckGoHtml | Provider::DuckDuckGoLite, _) => {
                request = request.query("df", "w")
            }
            (Provider::Serper, Some(b)) => b["tbs"] = "qdr:w".into(),
            (Provider::Tavily, Some(b)) => b["time_range"] = "week".into(),
            (Provider::Exa, Some(b)) => b["startPublishedDate"] = week_ago().into(),
            _ => {}
        }
    }
    let request = request.set("User-Agent", USER_AGENT).timeout(TIMEOUT);
    let reply = match body {
        Some(body) => {
            (request.set("Content-Type", "application/json")).send_string(&body.to_string())
        }
        None => request.call(),
    };
    match reply {
        // DuckDuckGo answers a client it suspects with 202 and a challenge page.
        Ok(r) if r.status() == 202 => Err(Fail::Down("rate-limited (HTTP 202 challenge)".into())),
        Ok(r) => (r.into_string()).map_err(|e| Fail::Down(format!("unreadable page: {e}"))),
        Err(ureq::Error::Status(code @ (401 | 403), _)) if !key.is_empty() => Err(Fail::Skip(
            format!("refused (HTTP {code}); check the key in Settings"),
        )),
        // A free provider's 403 is its bot block: a key is the way past it too.
        Err(ureq::Error::Status(code @ (403 | 429), _)) => {
            Err(Fail::Down(format!("rate-limited (HTTP {code})")))
        }
        Err(ureq::Error::Status(code, _)) => Err(Fail::Down(format!("HTTP {code}"))),
        Err(ureq::Error::Transport(e)) => Err(Fail::Skip(e.kind().to_string())),
    }
}

/// Appends one search to `dir`'s `log.jsonl`: the query, whether it asked for the week, the
/// provider that answered, the status
/// (ok, cached, empty, failed), the results' URLs and why each provider before it did not answer.
/// It is what tells "found in one chat, not in the next" apart: another provider, or none.
// ponytail: a line a search, never trimmed, about 3 MB at 10,000 searches; rotate it when a log
// that size shows up
fn note(
    dir: &Path,
    query: &str,
    recent: bool,
    provider: Option<&str>,
    status: &str,
    hits: &[Hit],
    reasons: &[String],
) -> io::Result<()> {
    let at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let urls: Vec<&str> = hits.iter().map(|h| h.url.as_str()).collect();
    let line = json!({"at": at, "query": query, "recent": recent, "provider": provider, "status": status,
        "urls": urls, "tried": reasons});
    fs::create_dir_all(dir)?;
    // One write a line, so the lines of searches that run at once do not interleave.
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("log.jsonl"))?;
    log.write_all(format!("{line}\n").as_bytes())
}

/// The text of `path` when it was written less than `age` ago.
fn fresh(path: &Path, age: Duration) -> Option<String> {
    let written = fs::metadata(path).and_then(|m| m.modified()).ok()?;
    (written.elapsed().ok()? < age).then(|| fs::read_to_string(path).ok())?
}

/// The cache file of `query` from `provider`.
fn cache_name(provider: &str, query: &str) -> String {
    let key = format!("{provider}\0{query}");
    format!(
        "{}.json",
        larch::bench::hex(&larch::bench::sha256(key.as_bytes()))
    )
}

/// Waits until `GAP` plus jitter has passed since the last request recorded in `dir`'s `file`, in
/// any process, and records this one: `last` is the whole install's, `last-<provider>` one
/// provider's (`race`).
fn throttle(dir: &Path, file: &str) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut last = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join(file))?;
    last.lock()?;
    let mut text = String::new();
    last.read_to_string(&mut text)?;
    let now = || {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
    };
    let jitter = RandomState::new().build_hasher().finish() % JITTER_MS;
    let next = Duration::from_millis(text.trim().parse().unwrap_or(0) + jitter) + GAP;
    // A clock set back would put `next` far ahead: never wait past one full gap.
    sleep(
        next.saturating_sub(now())
            .min(GAP + Duration::from_millis(JITTER_MS)),
    );
    last.set_len(0)?;
    last.seek(SeekFrom::Start(0))?;
    write!(last, "{}", now().as_millis())
}

/// Each HTML provider's result link (href, title), the snippet after it, and the words of its page
/// with no results.
// ponytail: the no-results words are each site's as of Oct 2026; a site that rewords them reads as
// a changed layout (down 15 min) until its golden is refreshed
static HTML: LazyLock<[(Regex, Regex, &str); 3]> = LazyLock::new(|| {
    let re = |p: &str| Regex::new(p).expect("a valid pattern");
    [
        (
            re(r#"(?s)class="result__a"\s+href="([^"]*)"[^>]*>(.*?)</a>"#),
            re(r#"(?s)class="result__snippet"[^>]*>(.*?)</a>"#),
            r#"class="no-results""#,
        ),
        (
            re(r#"(?s)href="([^"]*)"\s+class=['"]result-link['"][^>]*>(.*?)</a>"#),
            re(r#"(?s)class=['"]result-snippet['"][^>]*>(.*?)</td>"#),
            "No results.",
        ),
        (
            re(r#"(?s)<a class="title" href="([^"]*)"[^>]*>(.*?)</a>"#),
            re(r#"(?s)<p class="s">(.*?)</p>"#),
            "No pages found",
        ),
    ]
});

/// The results `body` holds, in the provider's order, ads left out; an error says what changed.
fn parse(p: Provider, body: &str) -> Result<Vec<Hit>, String> {
    let changed = |what: &str| format!("the results page changed layout: {what}");
    let (list, keys) = match p {
        Provider::DuckDuckGoHtml | Provider::DuckDuckGoLite | Provider::Mojeek => {
            let (link, snippet, none) = &HTML[p as usize - Provider::DuckDuckGoHtml as usize];
            let links: Vec<_> = link.captures_iter(body).collect();
            if links.is_empty() && !body.contains(none) {
                return Err(changed("no result links and no no-results notice"));
            }
            let hits = links.iter().enumerate().map(|(i, c)| {
                let end = links
                    .get(i + 1)
                    .map_or(body.len(), |n| n.get_match().start());
                let rest = &body[c.get_match().end()..end];
                Hit {
                    title: text(&c[2]),
                    url: target(&c[1]),
                    snippet: snippet.captures(rest).map_or(String::new(), |s| cut(&s[1])),
                }
            });
            return Ok(hits
                .filter(|h| !h.url.contains("duckduckgo.com/y.js"))
                .collect());
        }
        Provider::Brave => ("/web/results", ["title", "url", "description"]),
        Provider::Serper => ("/organic", ["title", "link", "snippet"]),
        Provider::Tavily => ("/results", ["title", "url", "content"]),
        Provider::Exa => ("/results", ["title", "url", "text"]),
    };
    let json: Value = serde_json::from_str(body).map_err(|e| changed(&e.to_string()))?;
    let list = json.pointer(list).and_then(Value::as_array);
    let field = |hit: &Value, key| hit[key].as_str().unwrap_or_default().to_owned();
    Ok(list
        .ok_or_else(|| changed("no results list"))?
        .iter()
        .map(|hit| Hit {
            title: text(&field(hit, keys[0])),
            url: field(hit, keys[1]),
            snippet: cut(&field(hit, keys[2])),
        })
        .collect())
}

/// `html` as text, cut to `SNIPPET` bytes.
fn cut(html: &str) -> String {
    let mut text = text(html);
    text.truncate(text.floor_char_boundary(SNIPPET));
    text
}

/// HTML as plain text: tags dropped, entities decoded, runs of white space one space.
fn text(html: &str) -> String {
    static TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]*>").expect("valid"));
    static ENTITY: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"&(#[xX][0-9a-fA-F]+|#[0-9]+|[a-z]+);").expect("valid"));
    let plain = TAG.replace_all(html, "");
    let decoded = ENTITY.replace_all(&plain, |c: &regex::Captures| {
        let name = &c[1];
        let code = match name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => name.strip_prefix('#').and_then(|d| d.parse().ok()),
        };
        let named = match name {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some(' '),
            "rsaquo" => Some('\u{203a}'),
            _ => None,
        };
        match named.or(code.and_then(char::from_u32)) {
            Some(c) => c.to_string(),
            None => c[0].to_owned(),
        }
    });
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The address a result link leads to: DuckDuckGo's `/l/?uddg=` redirect unwrapped, and a
/// scheme-relative `//host` made https.
fn target(href: &str) -> String {
    let href = href.replace("&amp;", "&");
    let uddg = (href.split_once("duckduckgo.com/l/?"))
        .and_then(|(_, query)| query.split('&').find_map(|kv| kv.strip_prefix("uddg=")));
    let Some(encoded) = uddg else {
        return match href.strip_prefix("//") {
            Some(rest) => format!("https://{rest}"),
            None => href,
        };
    };
    let encoded = encoded.as_bytes();
    let mut out = Vec::with_capacity(encoded.len());
    let mut i = 0;
    while i < encoded.len() {
        let hex = encoded
            .get(i + 1..i + 3)
            .and_then(|h| std::str::from_utf8(h).ok());
        match hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
            Some(byte) if encoded[i] == b'%' => {
                out.push(byte);
                i += 3;
            }
            _ => {
                out.push(encoded[i]);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    // The queries here start the free providers at DuckDuckGo HTML: their bytes sum to a multiple
    // of three (`free_engines_take_turns` checks the turns).
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use super::*;
    use crate::{
        audit::{Audit, AuditEvent},
        tool::Permission,
    };

    /// `search` one provider at a time, as with `RACE_VAR` at 0: the race is on by default, and
    /// these tests check the chain's order (the race tests call `chain` themselves).
    fn search(
        dir: &Path,
        query: &str,
        keys: &[(Provider, Key)],
        recent: bool,
        url: impl Fn(Provider) -> String,
    ) -> Result<Vec<Hit>, ToolError> {
        chain(dir, query, keys, recent, url, false)
    }

    fn fixture(name: &str) -> String {
        let path = format!(
            "{}/tests/fixtures/search/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(path).unwrap()
    }

    const BOOK: &str = "https://doc.rust-lang.org/book/ch04-02-references-and-borrowing.html";
    const BOOK_TITLE: &str = "References and Borrowing - The Rust Programming Language";

    #[test]
    fn golden_duckduckgo_html() {
        let hits = parse(Provider::DuckDuckGoHtml, &fixture("duckduckgo_html.html")).unwrap();
        // The ad is dropped; the redirect is unwrapped; tags and entities are text.
        assert_eq!(
            hits,
            [
                Hit {
                    title: BOOK_TITLE.into(),
                    url: BOOK.into(),
                    snippet: "A reference is like a pointer in that it's an address we can \
                        follow; the borrow checker makes sure references are always valid & \
                        never dangle."
                        .into(),
                },
                Hit {
                    title: "Introducing the Rust borrow checker - LogRocket Blog".into(),
                    url: "https://blog.logrocket.com/introducing-rust-borrow-checker/".into(),
                    snippet: "The borrow checker is the part of the compiler that enforces \
                        ownership rules."
                        .into(),
                },
                Hit {
                    title: "What is the borrow checker? - Stack Overflow".into(),
                    url: "https://stackoverflow.com/questions/47506/what-is-the-borrow-checker?q=1&r=2"
                        .into(),
                    snippet: "Rust's compiler tracks lifetimes of \"borrows\" <at compile time>."
                        .into(),
                },
            ]
        );
    }

    #[test]
    fn golden_duckduckgo_lite() {
        let hits = parse(Provider::DuckDuckGoLite, &fixture("duckduckgo_lite.html")).unwrap();
        assert_eq!(
            hits,
            [
                Hit {
                    title: BOOK_TITLE.into(),
                    url: BOOK.into(),
                    snippet: "A reference is like a pointer; the borrow checker makes sure \
                        references are always valid."
                        .into(),
                },
                Hit {
                    title: "Rust without a redirect".into(),
                    url: "https://without.redirect.example/rust".into(),
                    snippet: "Lite links some results directly.".into(),
                },
            ]
        );
    }

    #[test]
    fn golden_mojeek() {
        let hits = parse(Provider::Mojeek, &fixture("mojeek.html")).unwrap();
        assert_eq!(
            hits,
            [
                Hit {
                    title: BOOK_TITLE.into(),
                    url: BOOK.into(),
                    snippet: "A reference is like a pointer; the borrow checker makes sure \
                        references are always valid."
                        .into(),
                },
                Hit {
                    title: "MIR borrow check - Rust Compiler Development Guide".into(),
                    url: "https://rustc-dev-guide.rust-lang.org/borrow_check.html".into(),
                    snippet: "The borrow check is Rust's \"secret sauce\".".into(),
                },
            ]
        );
    }

    #[test]
    fn golden_json_providers() {
        for (p, file) in [
            (Provider::Brave, "brave.json"),
            (Provider::Serper, "serper.json"),
            (Provider::Tavily, "tavily.json"),
            (Provider::Exa, "exa.json"),
        ] {
            let hits = parse(p, &fixture(file)).unwrap();
            assert_eq!(
                hits,
                [
                    Hit {
                        title: BOOK_TITLE.into(),
                        url: BOOK.into(),
                        snippet: "A reference is like a pointer; the borrow checker makes sure \
                            references are always valid."
                            .into(),
                    },
                    Hit {
                        title: "MIR borrow check".into(),
                        url: "https://rustc-dev-guide.rust-lang.org/borrow_check.html".into(),
                        snippet: "The borrow check is Rust's secret sauce.".into(),
                    },
                ],
                "{p:?}"
            );
        }
    }

    #[test]
    fn a_changed_layout_fails_loudly() {
        let page = fixture("changed_layout.html");
        for p in [
            Provider::DuckDuckGoHtml,
            Provider::DuckDuckGoLite,
            Provider::Mojeek,
        ] {
            let e = parse(p, &page).unwrap_err();
            assert!(e.contains("changed layout"), "{p:?}: {e}");
        }
        for p in [
            Provider::Brave,
            Provider::Serper,
            Provider::Tavily,
            Provider::Exa,
        ] {
            let e = parse(p, r#"{"data": []}"#).unwrap_err();
            assert!(e.contains("changed layout"), "{p:?}: {e}");
            assert!(parse(p, "<html>").is_err(), "{p:?}");
        }
    }

    /// A server on 127.0.0.1 that answers each connection with the next of `replies`, then
    /// refuses, and keeps every request it read.
    struct Server {
        port: u16,
        seen: Arc<Mutex<Vec<String>>>,
    }

    fn serve(replies: Vec<(u16, String)>) -> Server {
        serve_after(replies, Duration::ZERO)
    }

    /// `serve`, each reply sent `delay` after its request is read.
    fn serve_after(replies: Vec<(u16, String)>, delay: Duration) -> Server {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for (status, body) in replies {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = Vec::new();
                let mut buf = [0; 4096];
                // The head, then as much body as Content-Length says.
                while let Ok(n @ 1..) = stream.read(&mut buf) {
                    request.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&request).to_lowercase();
                    if let Some(head) = text.find("\r\n\r\n") {
                        let len = text
                            .split("content-length:")
                            .nth(1)
                            .and_then(|l| l.lines().next()?.trim().parse().ok())
                            .unwrap_or(0);
                        if request.len() >= head + 4 + len {
                            break;
                        }
                    }
                }
                log.lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&request).into_owned());
                std::thread::sleep(delay);
                let reply = format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(reply.as_bytes()).unwrap();
            }
        });
        Server { port, seen }
    }

    impl Server {
        /// Each provider at its own path on this server.
        fn url(&self) -> impl Fn(Provider) -> String {
            let port = self.port;
            move |p| format!("http://127.0.0.1:{port}/{p:?}")
        }

        /// The paths requested, in order.
        fn paths(&self) -> Vec<String> {
            let seen = self.seen.lock().unwrap();
            let path = |r: &String| r.split(' ').nth(1).unwrap_or_default().to_owned();
            seen.iter()
                .map(path)
                .map(|p| p[1..].split('?').next().unwrap().into())
                .collect()
        }
    }

    /// DuckDuckGo HTML at `html`'s address, the other providers at `rest`'s.
    fn split(html: &Server, rest: &Server) -> impl Fn(Provider) -> String {
        let (html, rest) = (html.url(), rest.url());
        move |p| match p {
            Provider::DuckDuckGoHtml => html(p),
            _ => rest(p),
        }
    }

    #[test]
    fn a_race_keeps_the_first_good_list() {
        let dir = tempfile::tempdir().unwrap();
        let slow = serve_after(
            vec![(200, fixture("duckduckgo_html.html"))],
            Duration::from_secs(3),
        );
        let fast = serve(vec![(200, fixture("mojeek.html"))]);
        // The race is timed, not the process's first certificate load, which every request
        // waits for: 3.5-5 s on the M5 runner, where this failed 2 runs in 3.
        drop(larch_fetch::agent("https://example.com"));
        let start = Instant::now();
        let hits = chain(dir.path(), "rust", &[], false, split(&slow, &fast), true).unwrap();
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "{:?}",
            start.elapsed()
        );
        assert_eq!(
            hits,
            parse(Provider::Mojeek, &fixture("mojeek.html")).unwrap()
        );
        // DuckDuckGo HTML is paired with Mojeek, not with DuckDuckGo Lite, its own site.
        assert_eq!(fast.paths(), ["Mojeek"]);
        let log = fs::read_to_string(dir.path().join("log.jsonl")).unwrap();
        assert!(log.contains(r#""provider":"Mojeek""#), "{log}");
    }

    #[test]
    fn a_race_whose_faster_provider_fails_waits_for_the_other() {
        let dir = tempfile::tempdir().unwrap();
        let slow = serve_after(
            vec![(200, fixture("duckduckgo_html.html"))],
            Duration::from_millis(500),
        );
        let fast = serve(vec![(429, String::new())]);
        let hits = chain(dir.path(), "rust", &[], false, split(&slow, &fast), true).unwrap();
        assert_eq!(
            hits,
            parse(Provider::DuckDuckGoHtml, &fixture("duckduckgo_html.html")).unwrap()
        );
        // The failure still marks its provider down; DuckDuckGo Lite, the third, is never asked.
        assert!(dir.path().join("down/Mojeek").exists());
        assert_eq!(fast.paths(), ["Mojeek"]);
    }

    #[test]
    fn the_slower_providers_rate_limit_is_marked_after_the_race() {
        let dir = tempfile::tempdir().unwrap();
        let fast = serve(vec![(200, fixture("duckduckgo_html.html"))]);
        let slow = serve_after(vec![(429, String::new())], Duration::from_millis(300));
        chain(dir.path(), "rust", &[], false, split(&fast, &slow), true).unwrap();
        let down = dir.path().join("down/Mojeek");
        let start = Instant::now();
        while !down.exists() && start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(down.exists());
    }

    #[test]
    fn a_race_tries_the_next_pair_when_both_fail() {
        let dir = tempfile::tempdir().unwrap();
        let html = serve(vec![(503, String::new())]);
        let rest = serve(vec![
            (429, String::new()),
            (200, fixture("duckduckgo_lite.html")),
        ]);
        let hits = chain(dir.path(), "rust", &[], false, split(&html, &rest), true).unwrap();
        assert_eq!(
            hits,
            parse(Provider::DuckDuckGoLite, &fixture("duckduckgo_lite.html")).unwrap()
        );
        assert_eq!(rest.paths(), ["Mojeek", "DuckDuckGoLite"]);
    }

    #[test]
    fn a_race_where_no_provider_has_results_is_an_empty_list() {
        let dir = tempfile::tempdir().unwrap();
        let html = serve(vec![(200, NO_RESULTS.into())]);
        let rest = serve(vec![
            (200, "No pages found".into()),
            (200, "<td>No results.</td>".into()),
        ]);
        let hits = chain(dir.path(), "rust", &[], false, split(&html, &rest), true).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn a_race_asks_a_key_alone_before_any_free_cache() {
        let dir = tempfile::tempdir().unwrap();
        // A free provider's list, cached by an earlier search.
        let free = serve(vec![(200, fixture("duckduckgo_html.html"))]);
        let idle = serve(vec![]);
        chain(dir.path(), "rust", &[], false, split(&free, &idle), true).unwrap();
        let paid = serve(vec![(200, fixture("serper.json"))]);
        let keys = [(Provider::Serper, Key("k".into()))];
        let hits = chain(dir.path(), "rust", &keys, false, paid.url(), true).unwrap();
        assert_eq!(
            hits,
            parse(Provider::Serper, &fixture("serper.json")).unwrap()
        );
        assert_eq!(paid.paths(), ["Serper"]);
    }

    const RATE_LIMITED: &str = "search is rate-limited; add a key in Settings to lift it";

    #[test]
    fn a_429_marks_the_provider_down_and_the_next_one_answers() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![
            (429, String::new()),
            (200, fixture("duckduckgo_lite.html")),
            (200, fixture("duckduckgo_lite.html")),
        ]);
        let hits = search(
            dir.path(),
            "the rust borrow checker",
            &[],
            false,
            server.url(),
        )
        .unwrap();
        assert_eq!(hits[1].title, "Rust without a redirect");
        assert_eq!(server.paths(), ["DuckDuckGoHtml", "DuckDuckGoLite"]);
        // Still down for the next query: it goes straight to the next provider.
        search(dir.path(), "mir rust", &[], false, server.url()).unwrap();
        let paths = ["DuckDuckGoHtml", "DuckDuckGoLite", "DuckDuckGoLite"];
        assert_eq!(server.paths(), paths);
    }

    #[test]
    fn a_challenge_or_a_changed_page_marks_down_too_and_all_down_says_add_a_key() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![
            (202, "<html>challenge</html>".into()),
            (200, fixture("changed_layout.html")),
            (429, String::new()),
        ]);
        let e = search(dir.path(), "x", &[], false, server.url())
            .unwrap_err()
            .to_string();
        assert!(e.starts_with(RATE_LIMITED), "{e}");
        assert!(e.contains("DuckDuckGoHtml: rate-limited"), "{e}");
        assert!(
            e.contains("DuckDuckGoLite: the results page changed layout"),
            "{e}"
        );
        // Every provider is down: nothing is sent, and the reasons come back from disk.
        let again = search(dir.path(), "x", &[], false, server.url())
            .unwrap_err()
            .to_string();
        assert_eq!(again, e);
        assert_eq!(server.paths().len(), 3);
        assert_eq!(log(dir.path())[1]["status"], "failed");
    }

    /// The lines of `dir`'s search log.
    fn log(dir: &Path) -> Vec<Value> {
        let text = fs::read_to_string(dir.join("log.jsonl")).unwrap();
        text.lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    const NO_RESULTS: &str = r#"<div class="no-results">No results.</div>"#;

    #[test]
    fn no_results_from_one_provider_asks_the_next_and_is_never_cached() {
        let dir = tempfile::tempdir().unwrap();
        // An empty list cached before this rule is not read as an answer.
        let cache = dir.path().join("cache");
        fs::create_dir_all(&cache).unwrap();
        fs::write(cache.join(cache_name("DuckDuckGoHtml", "m8")), "[]").unwrap();
        let server = serve(vec![
            (200, NO_RESULTS.into()),
            (200, fixture("duckduckgo_lite.html")),
            (200, NO_RESULTS.into()),
        ]);
        let hits = search(dir.path(), "m8", &[], false, server.url()).unwrap();
        assert_eq!(hits[1].title, "Rust without a redirect");
        // Again: the empty provider is asked again, and the next one's results come from the cache.
        assert_eq!(
            search(dir.path(), "m8", &[], false, server.url()).unwrap(),
            hits
        );
        let paths = ["DuckDuckGoHtml", "DuckDuckGoLite", "DuckDuckGoHtml"];
        assert_eq!(server.paths(), paths);
        let log = log(dir.path());
        let field = |i: usize, key: &str| log[i][key].clone();
        assert_eq!(field(0, "query"), "m8");
        assert_eq!(field(0, "provider"), "DuckDuckGoLite");
        assert_eq!(field(0, "status"), "ok");
        assert_eq!(field(0, "urls")[1], "https://without.redirect.example/rust");
        assert_eq!(field(0, "tried"), json!(["DuckDuckGoHtml: no results"]));
        assert_eq!(field(1, "status"), "cached");
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn no_results_from_every_provider_is_an_empty_list_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![
            (200, NO_RESULTS.into()),
            (429, String::new()),
            (200, "<p>No pages found matching:</p>".into()),
        ]);
        assert_eq!(
            search(dir.path(), "x", &[], false, server.url()).unwrap(),
            []
        );
        assert!(!dir.path().join("cache").exists());
        let log = log(dir.path());
        assert_eq!(log[0]["status"], "empty");
        assert_eq!(log[0]["provider"], Value::Null);
        let tried = [
            "DuckDuckGoHtml: no results",
            "DuckDuckGoLite: rate-limited (HTTP 429)",
            "Mojeek: no results",
        ];
        assert_eq!(log[0]["tried"], json!(tried));
    }

    #[test]
    fn a_cached_query_sends_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![(200, fixture("duckduckgo_html.html"))]);
        let first = search(dir.path(), "rust", &[], false, server.url()).unwrap();
        assert_eq!(first.len(), 3);
        let second = search(dir.path(), "rust", &[], false, server.url()).unwrap();
        assert_eq!(first, second);
        assert_eq!(server.paths().len(), 1);
    }

    #[test]
    fn braves_results_are_never_cached() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![
            (200, fixture("brave.json")),
            (200, fixture("brave.json")),
        ]);
        let keys = [(Provider::Brave, Key("k".into()))];
        let first = search(dir.path(), "rust", &keys, false, server.url()).unwrap();
        assert_eq!(
            search(dir.path(), "rust", &keys, false, server.url()).unwrap(),
            first
        );
        assert_eq!(server.paths(), ["Brave", "Brave"]);
        assert!(!dir.path().join("cache").exists());
    }

    #[test]
    fn a_key_moves_its_provider_first_in_a_header_with_the_larch_user_agent() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![(200, fixture("serper.json"))]);
        let keys = [(Provider::Serper, Key("sk-serper-123".into()))];
        let hits = search(dir.path(), "rust & café", &keys, false, server.url()).unwrap();
        assert_eq!(hits[0].url, BOOK);
        assert_eq!(server.paths(), ["Serper"]);
        let request = server.seen.lock().unwrap()[0].clone();
        assert!(request.starts_with("POST "), "{request}");
        let lower = request.to_lowercase();
        assert!(
            lower.contains("\r\nx-api-key: sk-serper-123\r\n"),
            "{request}"
        );
        assert!(
            lower.contains(&format!(
                "\r\nuser-agent: {}\r\n",
                USER_AGENT.to_lowercase()
            )),
            "{request}"
        );
        assert!(
            USER_AGENT.starts_with("larch-iris/")
                && USER_AGENT.ends_with(" (+https://github.com/Arkaya-Labs/Larch)")
        );
        assert!(request.contains(r#""q":"rust & café""#), "{request}");
    }

    #[test]
    fn free_engines_take_turns() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![
            (200, fixture("duckduckgo_html.html")),
            (200, fixture("duckduckgo_lite.html")),
            (200, fixture("mojeek.html")),
        ]);
        // Their bytes sum to 0, 1 and 2 more than a multiple of three.
        for q in ["rust", "one", "q"] {
            assert!(
                !search(dir.path(), q, &[], false, server.url())
                    .unwrap()
                    .is_empty()
            );
        }
        let first = ["DuckDuckGoHtml", "DuckDuckGoLite", "Mojeek"];
        assert_eq!(server.paths(), first);
    }

    #[test]
    fn a_free_provider_gets_the_query_in_the_url() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![(200, fixture("duckduckgo_html.html"))]);
        search(dir.path(), "rust & cafés", &[], false, server.url()).unwrap();
        let request = server.seen.lock().unwrap()[0].clone();
        assert!(
            request.starts_with("GET /DuckDuckGoHtml?q=rust+%26+caf%C3%A9s "),
            "{request}"
        );
    }

    #[test]
    fn a_query_in_another_case_or_spacing_is_the_same_search() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![(200, fixture("duckduckgo_html.html"))]);
        search(dir.path(), "What is  Rust", &[], false, server.url()).unwrap();
        // Cached: the same provider's file, so nothing is sent.
        search(dir.path(), "what is rust", &[], false, server.url()).unwrap();
        assert_eq!(server.paths(), ["DuckDuckGoHtml"]);
        assert_eq!(
            cache_name("DuckDuckGoHtml", "what is rust"),
            fs::read_dir(dir.path().join("cache"))
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .file_name()
                .into_string()
                .unwrap()
        );
    }

    #[test]
    fn a_recent_search_asks_for_the_week_and_is_cached_apart() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![
            (200, fixture("duckduckgo_html.html")),
            (200, fixture("duckduckgo_html.html")),
        ]);
        search(dir.path(), "rust", &[], false, server.url()).unwrap();
        search(dir.path(), "rust", &[], true, server.url()).unwrap();
        let seen = server.seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2, "the recent search is not the cached one");
        assert!(
            seen[1].starts_with("GET /DuckDuckGoHtml?q=rust&df=w "),
            "{}",
            seen[1]
        );
        assert!(!seen[0].contains("df="), "{}", seen[0]);
        let log = fs::read_to_string(dir.path().join("log.jsonl")).unwrap();
        let recent: Vec<bool> = log
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["recent"] == true)
            .collect();
        assert_eq!(
            recent,
            [false, true],
            "the log says which search asked for the week"
        );
    }

    #[test]
    fn a_recent_search_asks_each_keyed_provider_for_the_week_in_its_words() {
        let start = format!(r#""startPublishedDate":"{}""#, week_ago());
        for (p, file, filter) in [
            (Provider::Brave, "brave.json", "&freshness=pw "),
            (Provider::Serper, "serper.json", r#""tbs":"qdr:w""#),
            (Provider::Tavily, "tavily.json", r#""time_range":"week""#),
            (Provider::Exa, "exa.json", &start),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let server = serve(vec![(200, fixture(file))]);
            let keys = [(p, Key("k".into()))];
            search(dir.path(), "rust", &keys, true, server.url()).unwrap();
            let request = server.seen.lock().unwrap()[0].clone();
            assert!(request.contains(filter), "{p:?}: {request}");
        }
    }

    #[test]
    fn a_recent_search_skips_mojeek_which_cannot_ask_for_the_week() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![
            (200, r#"<div class="no-results">No results.</div>"#.into()),
            (200, "<td>No results.</td>".into()),
        ]);
        let hits = search(dir.path(), "rust", &[], true, server.url()).unwrap();
        assert!(hits.is_empty());
        let mut asked = server.paths();
        asked.sort();
        assert_eq!(asked, ["DuckDuckGoHtml", "DuckDuckGoLite"]);
    }

    #[test]
    fn requests_from_one_install_are_a_second_apart() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![
            (200, fixture("duckduckgo_html.html")),
            (200, fixture("duckduckgo_html.html")),
        ]);
        let start = Instant::now();
        search(dir.path(), "first", &[], false, server.url()).unwrap();
        search(dir.path(), "second", &[], false, server.url()).unwrap();
        assert!(
            start.elapsed() >= Duration::from_secs(1),
            "{:?}",
            start.elapsed()
        );
    }

    /// web_search with every provider on `server`.
    struct OnServer(Server);

    impl Tool for OnServer {
        fn spec(&self) -> crate::tool::ToolSpec {
            tool().spec()
        }

        fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
            call_at(args, ctx, self.0.url())
        }
    }

    /// Every file under `dir` but config.toml, as text.
    fn files(dir: &Path) -> Vec<(std::path::PathBuf, String)> {
        let mut out = Vec::new();
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                out.extend(files(&path));
            } else if path.file_name().unwrap() != "config.toml" {
                out.push((
                    path.clone(),
                    String::from_utf8_lossy(&fs::read(&path).unwrap()).into(),
                ));
            }
        }
        out
    }

    #[test]
    fn no_key_reaches_a_result_an_error_the_audit_log_the_disk_or_a_url() {
        let secrets = [
            "SECRET-brave-1",
            "SECRET-serper-2",
            "SECRET-tavily-3",
            "SECRET-exa-4",
        ];
        let config = format!(
            "brave_key = \"{}\"\nserper_key = \"{}\"\ntavily_key = \"{}\"\nexa_key = \"{}\"\n",
            secrets[0], secrets[1], secrets[2], secrets[3]
        );
        let leaks = |text: &str| secrets.iter().any(|s| text.contains(s));
        let run_in = |dir: &Path, replies: Vec<(u16, String)>| {
            fs::create_dir_all(dir.join("agent")).unwrap();
            fs::write(dir.join("agent/config.toml"), &config).unwrap();
            let ctx = ToolCtx {
                allowed: vec![],
                data_dir: dir.into(),
                source: "test".into(),
            };
            let tool = OnServer(serve(replies));
            let args = json!({"query": "rust", "n": 2});
            let got = crate::tool::run(&tool, &args, &ctx, &Audit::new(dir), AuditEvent::default());
            let requests = tool.0.seen.lock().unwrap().clone();
            (got.map_err(|e| e.to_string()), requests)
        };
        // Each keyed provider fails its own way, then every free one: the error names them all.
        let failing = tempfile::tempdir().unwrap();
        let replies = vec![
            (401, "bad key".into()),
            (429, String::new()),
            (500, "oops".into()),
            (200, "<html>not json</html>".into()),
            (429, String::new()),
            (202, String::new()),
        ];
        let (got, requests) = run_in(failing.path(), replies);
        let e = got.unwrap_err();
        assert!(e.starts_with("search is rate-limited ("), "{e}");
        assert!(
            e.contains("Brave: refused (HTTP 401); check the key in Settings"),
            "{e}"
        );
        assert!(!leaks(&e), "{e}");
        // The keys went in headers only: never in a request line, which errors quote.
        assert_eq!(requests.len(), 6);
        for (request, secret) in requests.iter().zip(secrets) {
            assert!(request.contains(secret), "{request}");
            assert!(!leaks(request.lines().next().unwrap()), "{request}");
        }
        // Again, with every provider down: the reasons come back from disk.
        let (again, _) = run_in(failing.path(), vec![]);
        assert!(!leaks(&again.unwrap_err()));
        let ok = tempfile::tempdir().unwrap();
        let (hits, _) = run_in(ok.path(), vec![(200, fixture("brave.json"))]);
        let hits = hits.unwrap();
        assert!(hits.starts_with(r#"[{"title":"References and Borrowing"#) && !leaks(&hits));
        assert_eq!(serde_json::from_str::<Vec<Hit>>(&hits).unwrap().len(), 2);
        // Nothing the agent wrote holds a key: the audit log, the cache, the down marks.
        for dir in [failing.path(), ok.path()] {
            let written = files(dir);
            assert!(
                written
                    .iter()
                    .any(|(p, _)| p.starts_with(dir.join("agent/log")))
            );
            for (path, text) in written {
                assert!(!leaks(&text), "{}: {text}", path.display());
            }
        }
        assert!(!leaks(&tool().spec().function().to_string()));
    }

    #[test]
    fn a_refused_key_is_not_marked_down_so_a_fixed_key_works_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![
            (401, String::new()),
            (200, fixture("duckduckgo_html.html")),
            (200, fixture("brave.json")),
        ]);
        let keys = [(Provider::Brave, Key("typo".into()))];
        search(dir.path(), "first", &keys, false, server.url()).unwrap();
        let keys = [(Provider::Brave, Key("fixed".into()))];
        let hits = search(dir.path(), "second", &keys, false, server.url()).unwrap();
        assert_eq!(hits[1].title, "MIR borrow check");
        assert_eq!(server.paths(), ["Brave", "DuckDuckGoHtml", "Brave"]);
    }

    #[test]
    fn a_403_from_a_free_provider_reads_as_rate_limited() {
        let dir = tempfile::tempdir().unwrap();
        let server = serve(vec![
            (403, String::new()),
            (200, fixture("duckduckgo_lite.html")),
        ]);
        search(dir.path(), "x", &[], false, server.url()).unwrap();
        let reason = fs::read_to_string(dir.path().join("down/DuckDuckGoHtml")).unwrap();
        assert_eq!(reason, "rate-limited (HTTP 403)");
    }

    #[test]
    fn only_duckduckgos_redirect_is_unwrapped() {
        let direct = "https://example.com/page?xuddg=1&id=5";
        assert_eq!(target(direct), direct);
        assert_eq!(
            target("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example%2F&rut=1"),
            "https://a.example/"
        );
    }

    #[test]
    fn caching_a_result_removes_the_stale_ones() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        fs::create_dir_all(&cache).unwrap();
        let (stale, young) = (cache.join("stale.json"), cache.join("young.json"));
        for file in [&stale, &young] {
            fs::write(file, "[]").unwrap();
        }
        let old = SystemTime::now() - CACHE_FOR - Duration::from_secs(60);
        fs::File::options()
            .write(true)
            .open(&stale)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let server = serve(vec![(200, fixture("duckduckgo_html.html"))]);
        search(dir.path(), "x", &[], false, server.url()).unwrap();
        assert!(!stale.exists() && young.exists());
    }

    #[test]
    fn a_config_that_does_not_read_names_its_line_but_never_quotes_it() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("agent")).unwrap();
        let config = "git_tools = true\nbrave_key \"SECRET-1\"\n";
        fs::write(dir.path().join("agent/config.toml"), config).unwrap();
        let ctx = ToolCtx {
            allowed: vec![],
            data_dir: dir.path().into(),
            source: "test".into(),
        };
        let server = serve(vec![]);
        let e = call_at(&json!({"query": "x"}), &ctx, server.url())
            .unwrap_err()
            .to_string();
        assert!(e.contains("config.toml:2"), "{e}");
        assert!(!e.contains("SECRET"), "{e}");
    }

    #[test]
    fn web_search_is_a_read_with_untrusted_results_and_at_most_10_results() {
        let spec = tool().spec();
        assert_eq!(spec.name, "web_search");
        assert_eq!(spec.permission, Permission::Read);
        assert!(spec.untrusted_results);
        assert_eq!(spec.schema["properties"]["n"]["maximum"], 10);
        assert_eq!(spec.schema["required"], serde_json::json!(["query"]));
    }

    #[test]
    fn a_page_without_results_is_no_results_not_a_layout_change() {
        let empty = [
            (
                Provider::DuckDuckGoHtml,
                r#"<div class="no-results">No results.</div>"#,
            ),
            (Provider::DuckDuckGoLite, "<td>No results.</td>"),
            (Provider::Mojeek, "<p>No pages found matching:</p>"),
            (Provider::Brave, r#"{"web": {"results": []}}"#),
            (Provider::Serper, r#"{"organic": []}"#),
            (Provider::Tavily, r#"{"results": []}"#),
            (Provider::Exa, r#"{"results": []}"#),
        ];
        for (p, body) in empty {
            assert_eq!(parse(p, body), Ok(vec![]), "{p:?}");
        }
    }
}
