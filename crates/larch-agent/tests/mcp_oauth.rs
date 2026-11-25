//! OAuth sign-in against a fake MCP server and authorization server on 127.0.0.1, shaped like
//! time.b77.ai (metadata at the appended path, dynamic registration, PKCE S256, rotating refresh
//! tokens): the whole sign-in, a refresh shared by parallel calls, a reused refresh token, and the
//! two kinds of 403.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use larch_agent::mcp::client::{McpError, Server, Transport};
use larch_agent::mcp::oauth::{self, Auth, Callback, Grant};
use larch_agent::tool::{Tool, ToolCtx};
use serde_json::{Value, json};

#[derive(Default)]
struct Fake {
    base: String,
    valid: BTreeSet<String>,
    refresh: BTreeMap<String, bool>, // refresh token -> used
    code: Option<(String, String, String, String)>, // code, challenge, redirect, resource
    refreshes: usize,
    issued: usize,
    mode: &'static str, // "", "scope", "forbid"
}

struct Req {
    method: String,
    path: String,
    headers: BTreeMap<String, String>,
    body: String,
}

fn read(stream: &TcpStream) -> Option<Req> {
    let mut r = BufReader::new(stream);
    let mut line = String::new();
    r.read_line(&mut line).ok()?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next()?.to_owned(), parts.next()?.to_owned());
    let mut headers = BTreeMap::new();
    loop {
        let mut h = String::new();
        r.read_line(&mut h).ok()?;
        let h = h.trim();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_owned());
        }
    }
    let n: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0; n];
    r.read_exact(&mut body).ok()?;
    Some(Req {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).into(),
    })
}

fn reply(mut s: &TcpStream, status: &str, extra: &[(&str, String)], body: &str) {
    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    let _ = write!(s, "{head}\r\n{body}");
}

fn form(body: &str) -> BTreeMap<String, String> {
    url::form_urlencoded::parse(body.as_bytes())
        .into_owned()
        .collect()
}

fn issue(f: &mut Fake) -> Value {
    f.issued += 1;
    let (at, rt) = (format!("at-{}", f.issued), format!("rt-{}", f.issued));
    f.valid.insert(at.clone());
    f.refresh.insert(rt.clone(), false);
    json!({"access_token": at, "refresh_token": rt, "token_type": "Bearer", "expires_in": 3600})
}

fn serve(fake: Arc<Mutex<Fake>>, s: TcpStream) {
    let Some(req) = read(&s) else { return };
    let mut f = fake.lock().unwrap();
    let base = f.base.clone();
    let json_h = || vec![("Content-Type", "application/json".to_string())];
    match (
        req.method.as_str(),
        req.path.split('?').next().unwrap_or(""),
    ) {
        ("POST", "/mcp") => {
            let token = req
                .headers
                .get("authorization")
                .and_then(|a| a.strip_prefix("Bearer "))
                .map(str::to_owned);
            let ok = token.as_ref().is_some_and(|t| f.valid.contains(t));
            if ok && f.mode == "forbid" {
                return reply(&s, "403 Forbidden", &[], "");
            }
            if ok && f.mode == "scope" {
                let c = r#"Bearer error="insufficient_scope", scope="mcp admin""#.to_string();
                return reply(&s, "403 Forbidden", &[("WWW-Authenticate", c)], "");
            }
            if !ok {
                let c = format!(
                    r#"Bearer error="invalid_token", resource_metadata="{base}/.well-known/oauth-protected-resource/mcp", scope="mcp""#
                );
                return reply(&s, "401 Unauthorized", &[("WWW-Authenticate", c)], "");
            }
            let m: Value = serde_json::from_str(&req.body).unwrap_or_default();
            let id = m["id"].clone();
            let result = match m["method"].as_str().unwrap_or_default() {
                "initialize" => {
                    json!({"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake"}})
                }
                "tools/list" => {
                    json!({"tools": [{"name": "whoami", "description": "Who is signed in.", "inputSchema": {"type": "object"}}]})
                }
                "tools/call" => json!({"content": [{"type": "text", "text": "you are signed in"}]}),
                _ => return reply(&s, "202 Accepted", &[], ""),
            };
            reply(
                &s,
                "200 OK",
                &json_h(),
                &json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string(),
            );
        }
        ("GET", "/.well-known/oauth-protected-resource/mcp") => {
            let b = json!({"resource": format!("{base}/mcp"), "authorization_servers": [format!("{base}/as")], "scopes_supported": ["mcp"]});
            reply(&s, "200 OK", &json_h(), &b.to_string());
        }
        // Only the appended form, as time.b77.ai's issuer answers.
        ("GET", "/as/.well-known/oauth-authorization-server") => {
            let b = json!({"issuer": format!("{base}/as"), "authorization_endpoint": format!("{base}/as/authorize"),
                "token_endpoint": format!("{base}/as/token"), "registration_endpoint": format!("{base}/as/register"),
                "code_challenge_methods_supported": ["S256"]});
            reply(&s, "200 OK", &json_h(), &b.to_string());
        }
        ("POST", "/as/register") => {
            reply(&s, "201 Created", &json_h(), r#"{"client_id":"client-1"}"#)
        }
        ("GET", "/as/authorize") => {
            let q = form(req.path.split_once('?').map_or("", |(_, q)| q));
            assert_eq!(q["code_challenge_method"], "S256");
            assert_eq!(q["client_id"], "client-1");
            f.code = Some((
                "code-1".into(),
                q["code_challenge"].clone(),
                q["redirect_uri"].clone(),
                q["resource"].clone(),
            ));
            let to = format!("{}?code=code-1&state={}", q["redirect_uri"], q["state"]);
            reply(&s, "302 Found", &[("Location", to)], "");
        }
        ("POST", "/as/token") => {
            let p = form(&req.body);
            let bad = |s: &TcpStream| {
                reply(
                    s,
                    "400 Bad Request",
                    &json_h(),
                    r#"{"error":"invalid_grant"}"#,
                )
            };
            match p["grant_type"].as_str() {
                "authorization_code" => {
                    let Some((code, challenge, redirect, resource)) = f.code.take() else {
                        return bad(&s);
                    };
                    let ok = p["code"] == code
                        && oauth::challenge_of(&p["code_verifier"]) == challenge
                        && p["redirect_uri"] == redirect
                        && p["resource"] == resource
                        && resource == format!("{base}/mcp");
                    if !ok {
                        return bad(&s);
                    }
                    let b = issue(&mut f).to_string();
                    reply(&s, "200 OK", &json_h(), &b);
                }
                "refresh_token" => {
                    let rt = p["refresh_token"].clone();
                    match f.refresh.get(&rt) {
                        Some(false) => {
                            f.refresh.insert(rt, true);
                            f.refreshes += 1;
                            let b = issue(&mut f).to_string();
                            reply(&s, "200 OK", &json_h(), &b);
                        }
                        _ => bad(&s),
                    }
                }
                _ => bad(&s),
            }
        }
        _ => reply(&s, "404 Not Found", &[], ""),
    }
}

fn fake() -> (Arc<Mutex<Fake>>, String) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
    let fake = Arc::new(Mutex::new(Fake {
        base: base.clone(),
        ..Fake::default()
    }));
    let f = fake.clone();
    std::thread::spawn(move || {
        for s in listener.incoming().flatten() {
            let f = f.clone();
            std::thread::spawn(move || serve(f, s));
        }
    });
    (fake, base)
}

/// The browser: open the sign-in page, follow its redirect to the loopback callback.
fn browse(url: String) {
    let agent = ureq::AgentBuilder::new().redirects(0).build();
    let r = agent.get(&url).call().unwrap();
    assert_eq!(r.status(), 302);
    let to = r.header("Location").unwrap().to_owned();
    let page = ureq::get(&to).call().unwrap().into_string().unwrap();
    assert!(page.contains("Signed in"), "{page}");
}

fn ctx() -> ToolCtx {
    ToolCtx {
        allowed: vec![],
        data_dir: std::env::temp_dir(),
        source: "mcp_oauth".into(),
    }
}

#[test]
fn a_server_behind_oauth_signs_in_refreshes_once_and_asks_again_when_the_grant_is_gone() {
    let (fake, base) = fake();
    let url = format!("{base}/mcp");
    let saves = Arc::new(AtomicUsize::new(0));
    let s2 = saves.clone();
    let auth = Arc::new(Auth::new(
        None,
        Box::new(move |_| {
            s2.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }),
    ));
    let server = Server::with_auth(
        "fake",
        Transport::Http {
            url: url.clone(),
            headers: BTreeMap::new(),
        },
        None,
        Some(auth.clone()),
    );
    let hooked = Arc::new(AtomicUsize::new(0));
    let h2 = hooked.clone();
    server.set_on_needs_auth(Arc::new(move |_| {
        h2.fetch_add(1, Ordering::SeqCst);
    }));

    // Not signed in: the server's 401 names its sign-in details.
    match server.connect() {
        Err(McpError::NeedsAuth {
            challenge,
            scope_short: false,
        }) => {
            assert_eq!(
                challenge.resource_metadata.unwrap(),
                format!("{base}/.well-known/oauth-protected-resource/mcp")
            );
        }
        other => panic!("{:?}", other.map(|t| t.len())),
    }

    // The sign-in, as the app runs it.
    let challenge = oauth::probe(&url);
    let (resource, server_meta) = oauth::discover(&url, &challenge).unwrap();
    assert_eq!(resource.resource, url);
    assert_eq!(server_meta.issuer, format!("{base}/as"));
    let cb = Callback::bind(None).unwrap();
    let client = oauth::register(&server_meta, &cb.redirect_uri()).unwrap();
    let (verifier, state) = (oauth::random_token(32), oauth::random_token(16));
    let page = oauth::authorize_url(
        &server_meta,
        &client,
        &verifier,
        &state,
        challenge.scope.as_deref(),
        &resource.resource,
    )
    .unwrap();
    let browser = std::thread::spawn(move || browse(page));
    let (_tx, pasted) = std::sync::mpsc::channel();
    let code = cb
        .wait(&state, Duration::from_secs(10), &pasted, &|| false)
        .unwrap();
    browser.join().unwrap();
    let grant =
        oauth::exchange(&server_meta, &client, &code, &verifier, &resource.resource).unwrap();
    auth.set(Some(grant)).unwrap();
    server.reconnect();
    let tools = server.connect().unwrap();
    assert_eq!(tools[0].name, "whoami");
    let tool = server.chat_tools().into_iter().next().unwrap();
    assert_eq!(tool.call(&json!({}), &ctx()).unwrap(), "you are signed in");

    // The access token expires on the server: eight calls at once, one refresh.
    fake.lock().unwrap().valid.clear();
    let tool = Arc::new(tool);
    let calls: Vec<_> = (0..8)
        .map(|_| {
            let t = tool.clone();
            std::thread::spawn(move || t.call(&json!({}), &ctx()))
        })
        .collect();
    for c in calls {
        assert_eq!(c.join().unwrap().unwrap(), "you are signed in");
    }
    assert_eq!(fake.lock().unwrap().refreshes, 1);

    // A refresh token the server already rotated: the grant is gone, the server asks again.
    let used = fake
        .lock()
        .unwrap()
        .refresh
        .iter()
        .find(|(_, used)| **used)
        .map(|(k, _)| k.clone())
        .unwrap();
    let mut g: Grant =
        serde_json::from_str(&serde_json::to_string(&stale_grant(&auth, &url)).unwrap()).unwrap();
    g.refresh_token = Some(used);
    g.access_token = "at-expired".into();
    auth.set(Some(g)).unwrap();
    fake.lock().unwrap().valid.clear();
    let e = tool.call(&json!({}), &ctx()).unwrap_err().to_string();
    assert!(e.contains("needs a sign-in"), "{e}");
    assert!(!auth.signed_in_for(&url));
    assert!(hooked.load(Ordering::SeqCst) >= 1);
    assert!(saves.load(Ordering::SeqCst) >= 3);
}

/// A copy of the current grant (its fields, through serde), to plant a stale one.
fn stale_grant(auth: &Auth, url: &str) -> Grant {
    assert!(auth.signed_in_for(url));
    let token = auth.token_for(url).unwrap();
    Grant {
        endpoint: None,
        client: oauth::Registered {
            client_id: "client-1".into(),
            client_secret: None,
            auth_method: "none".into(),
            redirect_uri: "http://127.0.0.1:1/callback".into(),
        },
        token_endpoint: url.replace("/mcp", "/as/token"),
        resource: url.into(),
        access_token: token,
        refresh_token: None,
        expires_at: None,
        scope: None,
    }
}

#[test]
fn a_scope_short_403_asks_for_more_and_a_bare_403_is_a_refusal() {
    let (fake, base) = fake();
    let url = format!("{base}/mcp");
    let grant = {
        let mut f = fake.lock().unwrap();
        let t = issue(&mut f);
        Grant {
            endpoint: None,
            client: oauth::Registered {
                client_id: "client-1".into(),
                client_secret: None,
                auth_method: "none".into(),
                redirect_uri: "http://127.0.0.1:1/callback".into(),
            },
            token_endpoint: format!("{base}/as/token"),
            resource: url.clone(),
            access_token: t["access_token"].as_str().unwrap().into(),
            refresh_token: None,
            expires_at: None,
            scope: None,
        }
    };
    let auth = Arc::new(Auth::new(Some(grant), Box::new(|_| Ok(()))));
    let server = Server::with_auth(
        "fake",
        Transport::Http {
            url,
            headers: BTreeMap::new(),
        },
        None,
        Some(auth),
    );
    server.connect().unwrap();
    fake.lock().unwrap().mode = "scope";
    assert!(matches!(
        server.call("whoami", &json!({})),
        Err(McpError::NeedsAuth {
            scope_short: true,
            ..
        })
    ));
    fake.lock().unwrap().mode = "forbid";
    assert!(matches!(
        server.call("whoami", &json!({})),
        Err(McpError::Forbidden)
    ));
}
