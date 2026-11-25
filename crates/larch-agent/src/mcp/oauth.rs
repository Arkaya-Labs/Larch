//! OAuth sign-in for remote MCP servers (MCP authorization, revision 2025-11-25), as a public
//! native client: discovery from the server's 401 (RFC 9728 protected resource metadata, then
//! RFC 8414 / OpenID authorization server metadata), dynamic client registration (RFC 7591), the
//! authorization code flow with PKCE S256 and a loopback redirect (RFC 8252), resource indicators
//! (RFC 8707), and refresh behind one lock ([`Auth`]). No UI here: the app opens [`authorize_url`]
//! in the browser and waits on [`Callback::wait`] (or a pasted final URL).
//!
//! Tokens never appear in an error, a log line or a `Debug` print.

use std::io::{BufReader, Write};
use std::net::{IpAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::mpsc::Receiver;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const TIMEOUT: Duration = Duration::from_secs(20);

/// Why a sign-in could not go on; the sentence is shown to the user.
#[derive(Clone, Debug, PartialEq)]
pub struct OAuthError(pub String);

impl std::fmt::Display for OAuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for OAuthError {}

fn err(s: impl Into<String>) -> OAuthError {
    OAuthError(s.into())
}

/// What a 401's (or a scope-short 403's) `WWW-Authenticate: Bearer …` says.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Challenge {
    pub resource_metadata: Option<String>,
    pub scope: Option<String>,
    pub error: Option<String>,
}

/// The Bearer challenge's parameters, or `None` when the header is not a Bearer challenge.
pub fn parse_challenge(header: &str) -> Option<Challenge> {
    let h = header.trim();
    let rest = (h.get(..6)?.eq_ignore_ascii_case("bearer")).then(|| &h[6..])?;
    let mut c = Challenge::default();
    let re = regex::Regex::new(r#"([A-Za-z_]+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^\s,]+))"#).ok()?;
    for m in re.captures_iter(rest) {
        let value = m
            .get(2)
            .or(m.get(3))
            .map(|v| v.as_str().replace("\\\"", "\""));
        match m[1].to_ascii_lowercase().as_str() {
            "resource_metadata" => c.resource_metadata = value,
            "scope" => c.scope = value,
            "error" => c.error = value,
            _ => {}
        }
    }
    Some(c)
}

/// RFC 9728 protected resource metadata, the part a sign-in reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Resource {
    /// The canonical resource the tokens are for (RFC 8707 `resource`).
    pub resource: String,
    pub authorization_servers: Vec<String>,
    pub scopes: Vec<String>,
}

/// RFC 8414 authorization server metadata, the part a sign-in reads.
#[derive(Clone, Debug, PartialEq)]
pub struct AuthServer {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub registration_endpoint: Option<String>,
}

/// An MCP server URL as the tokens name it (RFC 8707): lower-case scheme and host, no default
/// port, no query (a registry URL can carry a key), no fragment, no trailing `/`.
pub fn canonical(url: &str) -> Result<String, OAuthError> {
    let mut u = url::Url::parse(url).map_err(|e| err(format!("'{url}' is not a URL: {e}")))?;
    u.set_query(None);
    u.set_fragment(None);
    let mut s: String = u.into();
    while s.ends_with('/') && s.matches('/').count() > 3 {
        s.pop();
    }
    Ok(s.trim_end_matches('/').to_owned())
}

/// `url` split as origin (`https://host[:port]`) and path (no trailing `/`, `""` for the root).
fn origin_path(url: &str) -> Result<(String, String), OAuthError> {
    let u = url::Url::parse(url).map_err(|e| err(format!("'{url}' is not a URL: {e}")))?;
    let origin = u.origin().ascii_serialization();
    if origin == "null" {
        return Err(err(format!("'{url}' has no origin")));
    }
    Ok((origin, u.path().trim_end_matches('/').to_owned()))
}

/// Where the MCP server's resource metadata may be: the 401's hint first, then the well-known
/// URL with the server's path, then at the root.
pub fn resource_metadata_urls(
    mcp_url: &str,
    hint: Option<&str>,
) -> Result<Vec<String>, OAuthError> {
    let (origin, path) = origin_path(mcp_url)?;
    let mut urls: Vec<String> = hint.map(str::to_owned).into_iter().collect();
    if !path.is_empty() {
        urls.push(format!(
            "{origin}/.well-known/oauth-protected-resource{path}"
        ));
    }
    urls.push(format!("{origin}/.well-known/oauth-protected-resource"));
    urls.dedup();
    Ok(urls)
}

/// Where an issuer's metadata may be: the RFC 8414 and OpenID path-inserted forms, the OpenID
/// appended form, and last the appended RFC 8414 form some servers use (time.b77.ai's issuer
/// answers only there).
pub fn auth_server_metadata_urls(issuer: &str) -> Result<Vec<String>, OAuthError> {
    let (origin, path) = origin_path(issuer)?;
    let mut urls = vec![
        format!("{origin}/.well-known/oauth-authorization-server{path}"),
        format!("{origin}/.well-known/openid-configuration{path}"),
    ];
    if !path.is_empty() {
        urls.push(format!("{origin}{path}/.well-known/openid-configuration"));
        urls.push(format!(
            "{origin}{path}/.well-known/oauth-authorization-server"
        ));
    }
    Ok(urls)
}

/// Every sign-in request: no redirect is followed, so a request reaches only the address
/// [`allowed`] checked.
fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .redirects(0)
        .timeout(TIMEOUT)
        .build()
}

fn get_json(url: &str) -> Option<Value> {
    let resp = (agent().get(url).set("Accept", "application/json").call()).ok()?;
    serde_json::from_str(&resp.into_string().ok()?).ok()
}

/// How far into this computer's networks an address is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Reach {
    Public,
    /// A private, link-local, shared or otherwise reserved address.
    Private,
    /// This computer.
    Local,
}

fn reach_of(ip: IpAddr) -> Reach {
    match ip {
        IpAddr::V4(a) => {
            let [o0, o1, ..] = a.octets();
            if a.is_loopback() || a.is_unspecified() {
                Reach::Local
            } else if a.is_private()
                || a.is_link_local()
                || a.is_broadcast()
                || a.is_documentation()
                || a.is_multicast()
                || o0 == 0
                || o0 >= 240
                // 100.64.0.0/10, carrier-grade NAT.
                || (o0 == 100 && (o1 & 0xc0) == 64)
            {
                Reach::Private
            } else {
                Reach::Public
            }
        }
        IpAddr::V6(a) => match a.to_ipv4_mapped() {
            Some(v4) => reach_of(IpAddr::V4(v4)),
            None if a.is_loopback() || a.is_unspecified() => Reach::Local,
            // fc00::/7 unique local, fe80::/10 link-local, multicast.
            None if (a.segments()[0] & 0xfe00) == 0xfc00
                || (a.segments()[0] & 0xffc0) == 0xfe80
                || a.is_multicast() =>
            {
                Reach::Private
            }
            None => Reach::Public,
        },
    }
}

/// The innermost reach of `url`'s host: its address, or every address its name resolves to; `None`
/// when it has none.
fn reach(url: &url::Url) -> Option<Reach> {
    let port = url.port_or_known_default()?;
    match url.host()? {
        url::Host::Ipv4(a) => Some(reach_of(a.into())),
        url::Host::Ipv6(a) => Some(reach_of(a.into())),
        url::Host::Domain(d) if d.eq_ignore_ascii_case("localhost") => Some(Reach::Local),
        url::Host::Domain(d) => ((d, port).to_socket_addrs().ok()?)
            .map(|a| reach_of(a.ip()))
            .max(),
    }
}

/// Whether a sign-in for the MCP server at `mcp_url` may send a request to `url`: https (http only
/// on this computer, for a server on this computer), to an address no further in than the MCP
/// server's own. A public server's sign-in, its metadata hints included, reaches public addresses
/// only; a server on this computer or a private network may use those (MCP security best
/// practices, SSRF).
pub fn allowed(url: &str, mcp_url: &str) -> bool {
    let (Ok(u), Ok(m)) = (url::Url::parse(url), url::Url::parse(mcp_url)) else {
        return false;
    };
    let scheme = u.scheme() == "https" || (u.scheme() == "http" && safe_endpoint(url));
    scheme
        && match (reach(&u), reach(&m)) {
            (Some(to), Some(limit)) => {
                to <= limit && (u.scheme() == "https" || limit == Reach::Local)
            }
            _ => false,
        }
}

/// An endpoint a sign-in may use at all: https, or http on this computer (tests and local
/// servers). [`allowed`] also checks where it leads.
pub fn safe_endpoint(url: &str) -> bool {
    url::Url::parse(url).is_ok_and(|u| {
        u.scheme() == "https"
            || (u.scheme() == "http"
                && matches!(u.host_str(), Some("127.0.0.1" | "localhost" | "[::1]")))
    })
}

/// The challenge an MCP server sends without a token: one `initialize` POST, its 401's header.
pub fn probe(mcp_url: &str) -> Challenge {
    let body = serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": super::client::PROTOCOL, "capabilities": {},
        "clientInfo": {"name": "larch", "version": env!("CARGO_PKG_VERSION")}}});
    let resp = agent()
        .post(mcp_url)
        .set("Content-Type", "application/json")
        .set("Accept", "application/json, text/event-stream")
        .send_string(&body.to_string());
    match resp {
        Err(ureq::Error::Status(_, r)) => (r.header("WWW-Authenticate"))
            .and_then(parse_challenge)
            .unwrap_or_default(),
        _ => Challenge::default(),
    }
}

/// The resource metadata URLs a sign-in fetches for `mcp_url`, in order: those [`allowed`] for it.
/// A hint in the challenge that leads elsewhere (another network, this computer) is never fetched.
fn metadata_to_try(mcp_url: &str, challenge: &Challenge) -> Result<Vec<String>, OAuthError> {
    let urls = resource_metadata_urls(mcp_url, challenge.resource_metadata.as_deref())?;
    Ok(urls.into_iter().filter(|u| allowed(u, mcp_url)).collect())
}

/// Finds the MCP server's authorization server: its resource metadata (which must name this
/// server), then the first authorization server whose metadata loads, names itself as that
/// issuer and offers PKCE S256.
pub fn discover(
    mcp_url: &str,
    challenge: &Challenge,
) -> Result<(Resource, AuthServer), OAuthError> {
    let mine = canonical(mcp_url)?;
    let mut meta = None;
    for url in metadata_to_try(mcp_url, challenge)? {
        if let Some(v) = get_json(&url).filter(|v| v["authorization_servers"].is_array()) {
            meta = Some(v);
            break;
        }
    }
    let meta = meta.ok_or_else(|| {
        err("the server needs a sign-in but publishes no sign-in details (OAuth protected resource metadata)")
    })?;
    let strings = |v: &Value| -> Vec<String> {
        (v.as_array().into_iter().flatten())
            .filter_map(|s| s.as_str().map(str::to_owned))
            .collect()
    };
    // RFC 9728 §3.3: the metadata must be this server's; its resource is this URL or a prefix of it.
    let resource = match meta["resource"].as_str() {
        Some(r) => {
            let r = canonical(r)?;
            let (ro, rp) = origin_path(&r)?;
            let (mo, mp) = origin_path(&mine)?;
            if ro != mo || !(mp == rp || mp.starts_with(&format!("{rp}/"))) {
                return Err(err(format!(
                    "the server's sign-in details are for '{r}', not this server: refused"
                )));
            }
            r
        }
        None => mine.clone(),
    };
    let resource = Resource {
        resource,
        authorization_servers: strings(&meta["authorization_servers"]),
        scopes: strings(&meta["scopes_supported"]),
    };
    for issuer in &resource.authorization_servers {
        if !allowed(issuer, mcp_url) {
            continue;
        }
        for url in auth_server_metadata_urls(issuer)? {
            if !allowed(&url, mcp_url) {
                continue;
            }
            let Some(m) = get_json(&url) else { continue };
            // RFC 8414 §3.3: the metadata names the issuer it was fetched for.
            if m["issuer"].as_str() != Some(issuer.as_str()) {
                continue;
            }
            let (Some(auth), Some(token)) = (
                m["authorization_endpoint"].as_str(),
                m["token_endpoint"].as_str(),
            ) else {
                continue;
            };
            let s256 = (m["code_challenge_methods_supported"].as_array())
                .is_some_and(|a| a.iter().any(|x| x == "S256"));
            if !s256 {
                return Err(err(
                    "the server's sign-in does not offer PKCE (S256), which MCP requires",
                ));
            }
            if !allowed(auth, mcp_url) || !allowed(token, mcp_url) {
                return Err(err(
                    "the server's sign-in pages are not https, or are on another network than the server: refused",
                ));
            }
            return Ok((
                resource.clone(),
                AuthServer {
                    issuer: issuer.clone(),
                    authorization_endpoint: auth.to_owned(),
                    token_endpoint: token.to_owned(),
                    registration_endpoint: (m["registration_endpoint"].as_str())
                        .filter(|r| allowed(r, mcp_url))
                        .map(str::to_owned),
                },
            ));
        }
    }
    Err(err(
        "the server's sign-in service could not be found (no authorization server metadata)",
    ))
}

/// The client a sign-in registered (RFC 7591), and how it authenticates at the token endpoint.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Registered {
    pub client_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    /// `none`, `client_secret_post` or `client_secret_basic` (RFC 7591's default with a secret).
    #[serde(default = "none_method")]
    pub auth_method: String,
    /// The loopback redirect URI it was registered with.
    pub redirect_uri: String,
}

fn none_method() -> String {
    "none".into()
}

impl std::fmt::Debug for Registered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Registered {{ client_id: {:?}, client_secret: {}, auth_method: {:?} }}",
            self.client_id,
            if self.client_secret.is_some() {
                "<hidden>"
            } else {
                "None"
            },
            self.auth_method
        )
    }
}

/// Registers Larch as a public native client with this redirect URI.
pub fn register(server: &AuthServer, redirect_uri: &str) -> Result<Registered, OAuthError> {
    let Some(endpoint) = &server.registration_endpoint else {
        return Err(err(
            "the server's sign-in needs an app registered in advance, which Larch cannot do yet (no dynamic client registration)",
        ));
    };
    let body = serde_json::json!({
        "client_name": "Larch",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    });
    let resp = agent()
        .post(endpoint)
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| {
            err(format!(
                "registering with the server's sign-in failed: {}",
                short(e)
            ))
        })?;
    let v: Value = serde_json::from_str(&resp.into_string().unwrap_or_default())
        .map_err(|_| err("the sign-in registration answer is not JSON"))?;
    let client_id = (v["client_id"].as_str())
        .ok_or_else(|| err("the sign-in registration gave no client id"))?;
    let secret = v["client_secret"].as_str().map(str::to_owned);
    let method = match (v["token_endpoint_auth_method"].as_str(), &secret) {
        (Some(m), _) => m.to_owned(),
        (None, Some(_)) => "client_secret_basic".into(),
        (None, None) => "none".into(),
    };
    Ok(Registered {
        client_id: client_id.to_owned(),
        client_secret: secret,
        auth_method: method,
        redirect_uri: redirect_uri.to_owned(),
    })
}

/// A ureq error without its response body (which may echo what was sent).
fn short(e: ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, _) => format!("HTTP {code}"),
        ureq::Error::Transport(t) => t.kind().to_string(),
    }
}

/// `n` random bytes as base64url, for a PKCE verifier or a `state`.
pub fn random_token(n: usize) -> String {
    let mut bytes = vec![0u8; n];
    getrandom::fill(&mut bytes).expect("the OS random source");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// The PKCE S256 challenge of a verifier.
pub fn challenge_of(verifier: &str) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

/// The browser's sign-in page for this attempt.
pub fn authorize_url(
    server: &AuthServer,
    client: &Registered,
    verifier: &str,
    state: &str,
    scope: Option<&str>,
    resource: &str,
) -> Result<String, OAuthError> {
    let mut u = url::Url::parse(&server.authorization_endpoint).map_err(|e| err(e.to_string()))?;
    {
        let mut q = u.query_pairs_mut();
        q.append_pair("response_type", "code")
            .append_pair("client_id", &client.client_id)
            .append_pair("redirect_uri", &client.redirect_uri)
            .append_pair("code_challenge", &challenge_of(verifier))
            .append_pair("code_challenge_method", "S256")
            .append_pair("state", state)
            .append_pair("resource", resource);
        if let Some(s) = scope.filter(|s| !s.is_empty()) {
            q.append_pair("scope", s);
        }
    }
    Ok(u.into())
}

/// What a callback (or a pasted final URL) carried.
enum Returned {
    Code(String),
    Refused(String),
    /// Not this sign-in's: wrong path or state. The wait goes on.
    Other,
}

/// The code in a callback query, if its `state` is this sign-in's.
fn returned(query: &str, state: &str) -> Returned {
    let params: Vec<(String, String)> = url::form_urlencoded::parse(query.as_bytes())
        .into_owned()
        .collect();
    let get = |k: &str| params.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
    if get("state").as_deref() != Some(state) {
        return Returned::Other;
    }
    if let Some(e) = get("error") {
        let why: String = get("error_description")
            .unwrap_or(e)
            .chars()
            .take(200)
            .collect();
        return Returned::Refused(why);
    }
    match get("code") {
        Some(code) => Returned::Code(code),
        None => Returned::Other,
    }
}

/// The code in the address the browser ended on, pasted by the user when the redirect could not
/// reach this computer (a WSL or firewall setup); `None` when it is not this sign-in's.
pub fn code_from_url(url: &str, state: &str) -> Result<Option<String>, OAuthError> {
    let u = url::Url::parse(url.trim())
        .map_err(|_| err("that is not the address the browser ended on"))?;
    match returned(u.query().unwrap_or_default(), state) {
        Returned::Code(c) => Ok(Some(c)),
        Returned::Refused(why) => Err(err(format!("the sign-in was refused: {why}"))),
        Returned::Other => Ok(None),
    }
}

/// The loopback listener the browser returns to: `127.0.0.1` only, one path, a size limit.
pub struct Callback {
    listener: TcpListener,
    port: u16,
}

const PAGE: &str = "<!doctype html><meta charset=utf-8><title>Larch</title><body style=\"font-family:sans-serif;padding:2rem\"><h2>%</h2><p>You can close this tab and go back to Larch.</p>";

impl Callback {
    /// Listens on `port` if it is free (a registered redirect is reused), else on any port.
    pub fn bind(port: Option<u16>) -> std::io::Result<Callback> {
        let listener = (port.and_then(|p| TcpListener::bind(("127.0.0.1", p)).ok()))
            .map_or_else(|| TcpListener::bind("127.0.0.1:0"), Ok)?;
        let port = listener.local_addr()?.port();
        listener.set_nonblocking(true)?;
        Ok(Callback { listener, port })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn redirect_uri(&self) -> String {
        format!("http://127.0.0.1:{}/callback", self.port)
    }

    /// Waits for this sign-in's code: the browser's `GET /callback?code=…&state=…`, or a final URL
    /// the user pasted (`pasted`). Requests with another path or state are answered and ignored,
    /// so no other program on this computer can end the sign-in. Stops when `cancelled` says so.
    pub fn wait(
        &self,
        state: &str,
        within: Duration,
        pasted: &Receiver<String>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<String, OAuthError> {
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            if cancelled() {
                return Err(err("the sign-in was cancelled"));
            }
            while let Ok(url) = pasted.try_recv() {
                if let Some(code) = code_from_url(&url, state)? {
                    return Ok(code);
                }
            }
            match self.listener.accept() {
                Ok((stream, _)) => match answer(stream, state) {
                    Returned::Code(c) => return Ok(c),
                    Returned::Refused(why) => {
                        return Err(err(format!("the sign-in was refused: {why}")));
                    }
                    Returned::Other => {}
                },
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => return Err(err(format!("the sign-in listener failed: {e}"))),
            }
        }
        Err(err("the sign-in was not finished in the browser in time"))
    }
}

/// One request on the listener, its whole head read before the reply.
fn answer(mut stream: TcpStream, state: &str) -> Returned {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let head = super::http::read_head(&mut BufReader::new(&stream))
        .ok()
        .flatten();
    let (method, target) = head.map(|h| (h.method, h.path)).unwrap_or_default();
    let got = match (method.as_str(), target.strip_prefix("/callback?")) {
        ("GET", Some(query)) => returned(query, state),
        _ => Returned::Other,
    };
    let (status, title) = match &got {
        Returned::Code(_) => ("200 OK", "Signed in"),
        Returned::Refused(_) => ("200 OK", "Sign-in was not completed"),
        Returned::Other => ("404 Not Found", "Not found"),
    };
    let body = PAGE.replace('%', title);
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    got
}

/// A server's sign-in as it is kept (`<data>/mcp-auth.json`, owner-only): enough to refresh.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Grant {
    pub client: Registered,
    pub token_endpoint: String,
    /// The resource the tokens were issued for (their audience): the canonical MCP URL, or a
    /// same-origin parent of it that the server's metadata names.
    pub resource: String,
    /// The canonical MCP URL this sign-in was for, when it differs from `resource`; the tokens are
    /// sent to that URL only, never to other paths under a parent resource.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Unix seconds; `None` when the server gave no lifetime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

impl std::fmt::Debug for Grant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Grant {{ resource: {:?}, expires_at: {:?}, tokens: <hidden> }}",
            self.resource, self.expires_at
        )
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl Grant {
    /// The access token is past (or within a minute of) its lifetime.
    pub fn expired(&self) -> bool {
        self.expires_at.is_some_and(|t| now() + 60 >= t)
    }
}

/// Why a refresh failed: the grant is gone (sign in again), or the service could not be reached
/// (keep the tokens and try later).
#[derive(Clone, Debug, PartialEq)]
pub enum RefreshError {
    SignInAgain(String),
    Transient(String),
}

fn token_request(
    endpoint: &str,
    client: &Registered,
    form: &[(&str, &str)],
) -> Result<Value, RefreshError> {
    let mut form: Vec<(&str, &str)> = form.to_vec();
    let mut req = agent().post(endpoint).set("Accept", "application/json");
    match (client.auth_method.as_str(), &client.client_secret) {
        ("client_secret_basic", Some(secret)) => {
            let enc =
                |s: &str| url::form_urlencoded::byte_serialize(s.as_bytes()).collect::<String>();
            let pair = format!("{}:{}", enc(&client.client_id), enc(secret));
            let basic = base64::engine::general_purpose::STANDARD.encode(pair);
            req = req.set("Authorization", &format!("Basic {basic}"));
        }
        ("client_secret_post", Some(secret)) => {
            form.push(("client_id", &client.client_id));
            form.push(("client_secret", secret));
        }
        _ => form.push(("client_id", &client.client_id)),
    }
    let text = match req.send_form(&form) {
        Ok(r) => r.into_string().unwrap_or_default(),
        Err(ureq::Error::Status(code, r)) => {
            // OAuth errors carry no secrets: their code and a short description are safe to show.
            let v: Value =
                serde_json::from_str(&r.into_string().unwrap_or_default()).unwrap_or_default();
            let code_name = v["error"].as_str().unwrap_or_default().to_owned();
            let why: String = (v["error_description"].as_str().unwrap_or(&code_name))
                .chars()
                .take(200)
                .collect();
            let msg = format!("the sign-in service refused (HTTP {code}): {why}");
            return Err(match code_name.as_str() {
                "invalid_grant" | "invalid_client" | "unauthorized_client" => {
                    RefreshError::SignInAgain(msg)
                }
                // Busy or slow is not refused: signing in again would delete a grant that works.
                _ if matches!(code, 408 | 429) => RefreshError::Transient(msg),
                _ if (400..500).contains(&code) => RefreshError::SignInAgain(msg),
                _ => RefreshError::Transient(msg),
            });
        }
        Err(e) => {
            return Err(RefreshError::Transient(format!(
                "the sign-in service could not be reached: {}",
                short(e)
            )));
        }
    };
    let v: Value = serde_json::from_str(&text).map_err(|_| {
        RefreshError::Transient("the sign-in service's token answer is not JSON".into())
    })?;
    if v["token_type"]
        .as_str()
        .is_some_and(|t| !t.eq_ignore_ascii_case("bearer"))
    {
        return Err(RefreshError::SignInAgain(format!(
            "the sign-in service gave a '{}' token, not a bearer token",
            v["token_type"]
        )));
    }
    Ok(v)
}

fn grant_from(
    v: &Value,
    client: &Registered,
    endpoint: &str,
    resource: &str,
    old: Option<&Grant>,
) -> Result<Grant, RefreshError> {
    let access = (v["access_token"].as_str()).ok_or_else(|| {
        RefreshError::SignInAgain("the sign-in service gave no access token".into())
    })?;
    Ok(Grant {
        client: client.clone(),
        token_endpoint: endpoint.to_owned(),
        resource: resource.to_owned(),
        access_token: access.to_owned(),
        // A refresh answer may leave the refresh token out: the old one stays good then.
        refresh_token: (v["refresh_token"].as_str().map(str::to_owned))
            .or_else(|| old.and_then(|g| g.refresh_token.clone())),
        expires_at: v["expires_in"].as_u64().map(|s| now() + s),
        scope: (v["scope"].as_str().map(str::to_owned))
            .or_else(|| old.and_then(|g| g.scope.clone())),
        endpoint: old.and_then(|g| g.endpoint.clone()),
    })
}

impl Grant {
    /// This grant bound to the MCP URL it was signed in for (`endpoint`).
    pub fn for_endpoint(self, mcp_url: &str) -> Result<Grant, OAuthError> {
        let c = canonical(mcp_url)?;
        Ok(Grant {
            endpoint: (c != self.resource).then_some(c),
            ..self
        })
    }

    /// Whether the tokens go to the canonical URL `c`: the URL the sign-in was for.
    fn binds(&self, c: &str) -> bool {
        self.endpoint.as_deref().unwrap_or(&self.resource) == c
    }
}

/// Trades the browser's code for tokens.
pub fn exchange(
    server: &AuthServer,
    client: &Registered,
    code: &str,
    verifier: &str,
    resource: &str,
) -> Result<Grant, OAuthError> {
    let form = [
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", client.redirect_uri.as_str()),
        ("code_verifier", verifier),
        ("resource", resource),
    ];
    let v = token_request(&server.token_endpoint, client, &form).map_err(|e| match e {
        RefreshError::SignInAgain(m) | RefreshError::Transient(m) => err(m),
    })?;
    grant_from(&v, client, &server.token_endpoint, resource, None).map_err(|e| match e {
        RefreshError::SignInAgain(m) | RefreshError::Transient(m) => err(m),
    })
}

/// A new access token from the refresh token.
pub fn refresh(grant: &Grant) -> Result<Grant, RefreshError> {
    let rt = (grant.refresh_token.as_deref())
        .ok_or_else(|| RefreshError::SignInAgain("the sign-in has expired".into()))?;
    let server = grant.endpoint.as_deref().unwrap_or(&grant.resource);
    if !allowed(&grant.token_endpoint, server) {
        // A name that resolves to nothing is offline, not refused: ending the sign-in there threw
        // the refresh token away whenever the app woke before the network.
        let resolves = |u: &str| url::Url::parse(u).ok().as_ref().and_then(reach).is_some();
        return Err(match resolves(&grant.token_endpoint) && resolves(server) {
            true => RefreshError::SignInAgain(
                "the saved sign-in's token address is not allowed for this server".into(),
            ),
            false => RefreshError::Transient("the sign-in service cannot be reached".into()),
        });
    }
    let form = [
        ("grant_type", "refresh_token"),
        ("refresh_token", rt),
        ("resource", grant.resource.as_str()),
    ];
    let v = token_request(&grant.token_endpoint, &grant.client, &form)?;
    grant_from(
        &v,
        &grant.client,
        &grant.token_endpoint,
        &grant.resource,
        Some(grant),
    )
}

/// Keeps a new grant, or forgets it (`None`); `Err` says why it could not be kept.
pub type SaveGrant = Box<dyn Fn(Option<&Grant>) -> Result<(), String> + Send + Sync>;

/// One server's sign-in as its connection uses it: the grant (read on every request, so a refresh
/// keeps the session) and a refresh that runs once however many calls find the token stale.
pub struct Auth {
    grant: Mutex<Option<Grant>>,
    /// Held through a refresh, so concurrent calls never send a rotated refresh token twice.
    refreshing: Mutex<()>,
    /// The sign-in's generation, held while a change commits (memory, then the save). Every change
    /// bumps it, so a refresh or a sign-in begun before a sign-out can never commit after it.
    generation: Mutex<u64>,
    /// Keeps a new grant (the app writes mcp-auth.json).
    save: SaveGrant,
}

impl Auth {
    pub fn new(grant: Option<Grant>, save: SaveGrant) -> Auth {
        Auth {
            grant: Mutex::new(grant),
            refreshing: Mutex::default(),
            generation: Mutex::default(),
            save,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Grant>> {
        self.grant.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn generation_lock(&self) -> std::sync::MutexGuard<'_, u64> {
        self.generation
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// The sign-in's generation now: a sign-in reads it when it starts and commits with
    /// [`Auth::set_if`].
    pub fn generation(&self) -> u64 {
        *self.generation_lock()
    }

    /// Signed in for this URL: the URL the grant was signed in for, in canonical form.
    pub fn signed_in_for(&self, url: &str) -> bool {
        let c = canonical(url).ok();
        self.lock()
            .as_ref()
            .is_some_and(|g| c.as_deref().is_some_and(|c| g.binds(c)))
    }

    /// Sets the grant, or forgets it (`None`), whatever is under way: a refresh or a sign-in begun
    /// before is dropped. `Err` when it could not be saved; the change holds for this session.
    pub fn set(&self, grant: Option<Grant>) -> Result<(), String> {
        let mut generation = self.generation_lock();
        *generation += 1;
        *self.lock() = grant.clone();
        (self.save)(grant.as_ref())
    }

    /// Sets the grant only if nothing changed the sign-in since `generation`
    /// ([`Auth::generation`]): `Ok(false)` when a sign-out or another sign-in came first, and then
    /// nothing is kept.
    pub fn set_if(&self, generation: u64, grant: Option<Grant>) -> Result<bool, String> {
        let mut now = self.generation_lock();
        if *now != generation {
            return Ok(false);
        }
        *now += 1;
        *self.lock() = grant.clone();
        (self.save)(grant.as_ref()).map(|()| true)
    }

    /// The bearer token for a request to `url`, refreshed first when it is stale; `None` when not
    /// signed in for that URL (a token is never sent to another server, or another path).
    pub fn token_for(&self, url: &str) -> Option<String> {
        let c = canonical(url).ok()?;
        let g = self.lock().clone()?;
        if !g.binds(&c) {
            return None;
        }
        if g.expired() && g.refresh_token.is_some() {
            return match self.refresh_after(&g.access_token) {
                Ok(fresh) => Some(fresh),
                // Sent with none, the request's 401 reads as signed out; the old token's 401
                // tries the refresh again and ends as a connection error.
                Err(RefreshError::Transient(_)) => Some(g.access_token),
                Err(RefreshError::SignInAgain(_)) => None,
            };
        }
        Some(g.access_token)
    }

    /// After a 401 for `stale`: a fresh token. Another call may have refreshed already; then that
    /// token is used and nothing is sent. `SignInAgain` clears the grant. A sign-out during the
    /// refresh wins: its result is dropped.
    pub fn refresh_after(&self, stale: &str) -> Result<String, RefreshError> {
        let _one = self
            .refreshing
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let generation = self.generation();
        let current = self
            .lock()
            .clone()
            .ok_or_else(|| RefreshError::SignInAgain("not signed in".into()))?;
        if current.access_token != stale {
            return Ok(current.access_token);
        }
        match refresh(&current) {
            Ok(g) => {
                let token = g.access_token.clone();
                match self.set_if(generation, Some(g)) {
                    Ok(true) => Ok(token),
                    // Not saved: the new token still holds for this session.
                    Err(_) => Ok(token),
                    Ok(false) => Err(RefreshError::SignInAgain(
                        "signed out while the sign-in was being refreshed".into(),
                    )),
                }
            }
            Err(RefreshError::SignInAgain(m)) => {
                let _ = self.set_if(generation, None);
                Err(RefreshError::SignInAgain(m))
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    #[test]
    fn a_bearer_challenge_gives_its_metadata_and_scope() {
        let h = r#"Bearer error="invalid_token", error_description="Bearer token required", resource_metadata="https://time.b77.ai/.well-known/oauth-protected-resource", scope="mcp""#;
        let c = parse_challenge(h).unwrap();
        assert_eq!(
            c.resource_metadata.as_deref(),
            Some("https://time.b77.ai/.well-known/oauth-protected-resource")
        );
        assert_eq!(c.scope.as_deref(), Some("mcp"));
        assert_eq!(c.error.as_deref(), Some("invalid_token"));
        assert!(parse_challenge("Basic realm=x").is_none());
        assert!(parse_challenge("bearer scope=\"a b\"").is_some());
        // Byte 6 inside a character: not a Bearer challenge, and no slice across it.
        assert!(parse_challenge("abcd\u{e9}\u{e9}x").is_none());
    }

    #[test]
    fn urls_are_canonical_and_metadata_urls_follow_both_path_orders() {
        assert_eq!(
            canonical("HTTPS://Time.B77.ai:443/mcp/?key=1#x").unwrap(),
            "https://time.b77.ai/mcp"
        );
        assert_eq!(
            canonical("https://example.com/").unwrap(),
            "https://example.com"
        );
        assert_eq!(
            resource_metadata_urls("https://time.b77.ai/mcp", None).unwrap(),
            [
                "https://time.b77.ai/.well-known/oauth-protected-resource/mcp",
                "https://time.b77.ai/.well-known/oauth-protected-resource"
            ]
        );
        assert_eq!(
            auth_server_metadata_urls("https://api.useruid.com/p/b77").unwrap(),
            [
                "https://api.useruid.com/.well-known/oauth-authorization-server/p/b77",
                "https://api.useruid.com/.well-known/openid-configuration/p/b77",
                "https://api.useruid.com/p/b77/.well-known/openid-configuration",
                "https://api.useruid.com/p/b77/.well-known/oauth-authorization-server",
            ]
        );
    }

    #[test]
    fn pkce_is_s256_of_a_random_verifier() {
        let v = random_token(32);
        assert_eq!(v.len(), 43);
        assert_ne!(v, random_token(32));
        // BASE64URL(SHA256(verifier)), as Python's hashlib and base64 compute it for this verifier.
        assert_eq!(
            challenge_of("dBjftJeZ4CVP-mJ92K1ccA4QaatQ_ZhLuEU3-6NAANP0M8TC_iEgMa8M4w7"),
            "A9r28CdvNJNie17SAJvwIa5F4Zs1DCrYvsTe_Xx0Xnk"
        );
    }

    fn client() -> Registered {
        Registered {
            client_id: "abc".into(),
            client_secret: Some("s3cret".into()),
            auth_method: "client_secret_post".into(),
            redirect_uri: "http://127.0.0.1:5000/callback".into(),
        }
    }

    #[test]
    fn the_authorize_url_carries_pkce_state_scope_and_resource() {
        let s = AuthServer {
            issuer: "https://as.example".into(),
            authorization_endpoint: "https://as.example/authorize".into(),
            token_endpoint: "https://as.example/token".into(),
            registration_endpoint: None,
        };
        let c = client();
        let url = authorize_url(
            &s,
            &c,
            "verifier",
            "st",
            Some("mcp"),
            "https://mcp.example/mcp",
        )
        .unwrap();
        for part in [
            "response_type=code",
            "client_id=abc",
            "code_challenge_method=S256",
            "state=st",
            "scope=mcp",
            "resource=https%3A%2F%2Fmcp.example%2Fmcp",
            "redirect_uri=http%3A%2F%2F127.0.0.1%3A5000%2Fcallback",
        ] {
            assert!(url.contains(part), "{part} in {url}");
        }
        assert!(!url.contains("s3cret"));
        assert!(!format!("{c:?}").contains("s3cret"));
    }

    fn get(port: u16, path: &str) -> String {
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        write!(
            s,
            "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nUser-Agent: test\r\n\r\n"
        )
        .unwrap();
        let mut out = String::new();
        let _ = s.read_to_string(&mut out);
        out
    }

    #[test]
    fn the_callback_takes_only_its_own_state_and_path() {
        let (_tx, none) = std::sync::mpsc::channel::<String>();
        let never = || false;
        let cb = Callback::bind(None).unwrap();
        let port = cb.port();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            assert!(get(port, "/favicon.ico").starts_with("HTTP/1.1 404"));
            // Another sign-in's state is ignored, not fatal.
            assert!(get(port, "/callback?code=x&state=other").starts_with("HTTP/1.1 404"));
            let ok = get(port, "/callback?code=the-code&state=s1");
            assert!(
                ok.starts_with("HTTP/1.1 200") && ok.contains("no-store"),
                "{ok}"
            );
        });
        assert_eq!(
            cb.wait("s1", Duration::from_secs(10), &none, &never)
                .unwrap(),
            "the-code"
        );
        t.join().unwrap();
        let cb = Callback::bind(None).unwrap();
        let port = cb.port();
        std::thread::spawn(move || get(port, "/callback?error=access_denied&state=s1"));
        assert!(
            cb.wait("s1", Duration::from_secs(10), &none, &never)
                .unwrap_err()
                .0
                .contains("refused")
        );
        assert!(
            Callback::bind(None)
                .unwrap()
                .wait("s", Duration::from_millis(300), &none, &never)
                .unwrap_err()
                .0
                .contains("in time")
        );
        assert!(
            Callback::bind(None)
                .unwrap()
                .wait("s", Duration::from_secs(5), &none, &|| true)
                .unwrap_err()
                .0
                .contains("cancelled")
        );
    }

    #[test]
    fn a_pasted_final_url_finishes_the_sign_in() {
        let (tx, pasted) = std::sync::mpsc::channel::<String>();
        tx.send("http://127.0.0.1:1/callback?code=c9&state=other".into())
            .unwrap();
        tx.send("http://127.0.0.1:1/callback?code=c7&state=s1".into())
            .unwrap();
        let cb = Callback::bind(None).unwrap();
        assert_eq!(
            cb.wait("s1", Duration::from_secs(5), &pasted, &|| false)
                .unwrap(),
            "c7"
        );
        assert!(code_from_url("not a url", "s1").is_err());
    }

    #[test]
    fn a_free_registered_port_is_reused() {
        let first = Callback::bind(None).unwrap();
        let port = first.port();
        drop(first);
        assert_eq!(Callback::bind(Some(port)).unwrap().port(), port);
        let held = Callback::bind(None).unwrap();
        assert_ne!(
            Callback::bind(Some(held.port())).unwrap().port(),
            held.port()
        );
    }

    #[test]
    fn only_https_or_this_computer_is_a_safe_endpoint() {
        assert!(safe_endpoint("https://as.example/token"));
        assert!(safe_endpoint("http://127.0.0.1:9000/token"));
        assert!(!safe_endpoint("http://as.example/token"));
        assert!(!safe_endpoint("javascript:alert(1)"));
    }

    fn grant(resource: &str, expires_at: Option<u64>) -> Grant {
        Grant {
            client: client(),
            token_endpoint: "https://as/token".into(),
            resource: resource.into(),
            access_token: "AT-secret".into(),
            refresh_token: Some("RT-secret".into()),
            expires_at,
            scope: None,
            endpoint: None,
        }
    }

    #[test]
    fn a_grant_hides_its_tokens_knows_when_it_expires_and_stays_with_its_server() {
        let g = grant("https://mcp.example/mcp", Some(0));
        let d = format!("{g:?}");
        assert!(!d.contains("AT-secret") && !d.contains("RT-secret"), "{d}");
        assert!(g.expired());
        let fresh = grant("https://mcp.example/mcp", None);
        assert!(!fresh.expired());
        let auth = Auth::new(Some(fresh), Box::new(|_| Ok(())));
        assert_eq!(
            auth.token_for("https://MCP.example/mcp/").as_deref(),
            Some("AT-secret")
        );
        assert_eq!(auth.token_for("https://other.example/mcp"), None);
        assert!(auth.signed_in_for("https://mcp.example/mcp"));
    }

    /// Offline, the token address resolves to nothing: the refresh waits (`Transient`), so the
    /// sign-in is kept; an address that resolves and is still not allowed ends it.
    #[test]
    fn a_refresh_whose_host_resolves_to_nothing_keeps_the_sign_in() {
        // `.invalid` never resolves (RFC 6761).
        let mut g = grant("https://mcp.larch-test.invalid/mcp", Some(0));
        g.token_endpoint = "https://as.larch-test.invalid/token".into();
        assert!(matches!(refresh(&g), Err(RefreshError::Transient(_))));
        // A public server's sign-in that names a private token address.
        let mut g = grant("https://8.8.8.8/mcp", Some(0));
        g.token_endpoint = "https://10.0.0.1/token".into();
        assert!(matches!(refresh(&g), Err(RefreshError::SignInAgain(_))));
    }

    #[test]
    fn a_sign_in_reaches_no_further_in_than_its_server() {
        // A public server: public https only; never this computer or a private network.
        let public = "https://8.8.8.8/mcp";
        assert!(allowed(
            "https://8.8.4.4/.well-known/oauth-protected-resource",
            public
        ));
        assert!(!allowed("http://127.0.0.1:9876/admin", public));
        assert!(!allowed("https://127.0.0.1:9876/admin", public));
        assert!(!allowed("https://localhost/admin", public));
        assert!(!allowed("https://10.0.0.5/as", public));
        assert!(!allowed("https://192.168.1.1/as", public));
        assert!(!allowed("https://169.254.169.254/latest", public));
        assert!(!allowed("https://100.64.0.1/as", public));
        assert!(!allowed("https://[::1]/as", public));
        assert!(!allowed("https://[fd00::1]/as", public));
        assert!(!allowed("https://[::ffff:127.0.0.1]/as", public));
        assert!(!allowed("http://8.8.4.4/as", public));
        // A server on this computer may use this computer, over http too (local development).
        let local = "http://127.0.0.1:8123/mcp";
        assert!(allowed("http://127.0.0.1:9000/as", local));
        assert!(allowed("https://8.8.4.4/as", local));
        // A server on a private network: that network, over https; not this computer.
        let private = "https://192.168.1.5/mcp";
        assert!(allowed("https://192.168.1.6/as", private));
        assert!(!allowed("https://127.0.0.1/as", private));
        assert!(!allowed("http://192.168.1.6/as", private));
    }

    #[test]
    fn a_metadata_hint_to_this_computer_from_a_public_server_is_never_fetched() {
        let challenge = Challenge {
            resource_metadata: Some("http://127.0.0.1:9876/admin".into()),
            ..Challenge::default()
        };
        let urls = metadata_to_try("https://8.8.8.8/mcp", &challenge).unwrap();
        assert!(!urls.is_empty());
        assert!(
            urls.iter().all(|u| u.starts_with("https://8.8.8.8/")),
            "{urls:?}"
        );
        // The same hint from a server on this computer is its own: fetched first.
        let urls = metadata_to_try("http://127.0.0.1:8123/mcp", &challenge).unwrap();
        assert_eq!(urls[0], "http://127.0.0.1:9876/admin");
    }

    #[test]
    fn a_grant_for_a_parent_resource_is_sent_to_its_own_endpoint_only() {
        let g = grant("https://service.example/api", None)
            .for_endpoint("https://service.example/api/mcp")
            .unwrap();
        assert_eq!(
            g.endpoint.as_deref(),
            Some("https://service.example/api/mcp")
        );
        let auth = Auth::new(Some(g), Box::new(|_| Ok(())));
        assert!(auth.signed_in_for("https://service.example/api/mcp"));
        assert_eq!(
            auth.token_for("https://service.example/api/mcp").as_deref(),
            Some("AT-secret")
        );
        // Not to the parent, nor to another path under it.
        assert_eq!(auth.token_for("https://service.example/api"), None);
        assert_eq!(auth.token_for("https://service.example/api/admin"), None);
        // A grant for the URL itself keeps no separate endpoint.
        let same = grant("https://mcp.example/mcp", None)
            .for_endpoint("https://mcp.example/mcp")
            .unwrap();
        assert_eq!(same.endpoint, None);
    }

    #[test]
    fn a_sign_out_wins_over_a_sign_in_begun_before_it() {
        let saved: std::sync::Arc<Mutex<Vec<bool>>> = std::sync::Arc::default();
        let s2 = saved.clone();
        let auth = Auth::new(
            None,
            Box::new(move |g| {
                s2.lock().unwrap().push(g.is_some());
                Ok(())
            }),
        );
        let began = auth.generation();
        auth.set(None).unwrap();
        let late = grant("https://mcp.example/mcp", None);
        assert!(!auth.set_if(began, Some(late.clone())).unwrap());
        assert!(!auth.signed_in_for("https://mcp.example/mcp"));
        assert_eq!(*saved.lock().unwrap(), [false]);
        // One begun after commits.
        assert!(auth.set_if(auth.generation(), Some(late)).unwrap());
        assert!(auth.signed_in_for("https://mcp.example/mcp"));
    }

    #[test]
    fn a_sign_out_during_a_refresh_stays_signed_out() {
        // A token endpoint on this computer that answers only when told to.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (asked_tx, asked) = std::sync::mpsc::channel();
        let (go_tx, go) = std::sync::mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            // The whole request, headers and form: a close with bytes unread resets the
            // connection on Windows, and the refresh then read as a network failure.
            let mut r = std::io::BufReader::new(stream.try_clone().unwrap());
            let mut len = 0;
            loop {
                let mut line = String::new();
                std::io::BufRead::read_line(&mut r, &mut line).unwrap();
                let line = line.trim_end().to_ascii_lowercase();
                if line.is_empty() {
                    break;
                }
                if let Some(v) = line.strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
            }
            r.read_exact(&mut vec![0; len]).unwrap();
            asked_tx.send(()).unwrap();
            go.recv().unwrap();
            let body = r#"{"access_token":"AT-new","token_type":"bearer","expires_in":3600}"#;
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        });
        let mut g = grant(&format!("http://127.0.0.1:{port}/mcp"), Some(0));
        g.token_endpoint = format!("http://127.0.0.1:{port}/token");
        let saved: std::sync::Arc<Mutex<Vec<bool>>> = std::sync::Arc::default();
        let s2 = saved.clone();
        let auth = std::sync::Arc::new(Auth::new(
            Some(g),
            Box::new(move |g| {
                s2.lock().unwrap().push(g.is_some());
                Ok(())
            }),
        ));
        let a2 = auth.clone();
        let refreshing = std::thread::spawn(move || a2.refresh_after("AT-secret"));
        // The refresh has sent its request; the user signs out before the answer comes.
        asked.recv().unwrap();
        auth.set(None).unwrap();
        go_tx.send(()).unwrap();
        server.join().unwrap();
        assert!(matches!(
            refreshing.join().unwrap(),
            Err(RefreshError::SignInAgain(_))
        ));
        assert!(!auth.signed_in_for(&format!("http://127.0.0.1:{port}/mcp")));
        // Saved: the sign-out only, never the refreshed grant after it.
        assert_eq!(*saved.lock().unwrap(), [false]);
    }

    /// A token endpoint that is busy (429) keeps the sign-in, and the request keeps its token.
    #[test]
    fn a_busy_token_endpoint_keeps_the_sign_in() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = stream.read(&mut [0u8; 4096]);
            let body = r#"{"error":"slow_down"}"#;
            let _ = write!(
                stream,
                "HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        });
        let url = format!("http://127.0.0.1:{port}/mcp");
        let mut g = grant(&url, Some(0));
        g.token_endpoint = format!("http://127.0.0.1:{port}/token");
        let auth = Auth::new(Some(g), Box::new(|_| Ok(())));
        assert_eq!(auth.token_for(&url).as_deref(), Some("AT-secret"));
        server.join().unwrap();
        assert!(auth.signed_in_for(&url));
    }
}
