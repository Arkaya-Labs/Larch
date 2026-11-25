//! An MCP client (revision 2025-11-25): connects to an outside server over stdio or Streamable
//! HTTP, lists its tools and calls them. Each tool joins a chat as an [`McpTool`]
//! (`AgentOptions::outside`), named `<server>__<tool>`.
//!
//! One request runs at a time per connection; a call waits for the one before it.

use std::collections::BTreeMap;
use std::fmt;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::oauth::{Auth, Challenge, RefreshError};
use crate::tool::{Permission, Tool, ToolCtx, ToolError, ToolSpec};

/// The revision this client asks for; [`VERSIONS`] and [`FIRST`] are the ones it accepts back.
pub const PROTOCOL: &str = "2025-11-25";
use super::VERSIONS;
/// The first revision, which stdio servers of early SDKs and older images still answer in: their
/// tools list and call in the same shape, and carry no annotations, so every one asks first.
const FIRST: &str = "2024-11-05";
/// How long `initialize` and `tools/list` may take.
const QUICK: Duration = Duration::from_secs(30);
/// How long a program's `initialize` may take: on its first start npx and uvx install the server
/// before it answers, which ran past `QUICK` and left the card failed. A program that exits fails
/// at once, whatever the wait.
const START: Duration = Duration::from_secs(300);
/// How long one `tools/call` may take.
const CALL: Duration = Duration::from_secs(120);
/// Tools taken from one server: more would crowd the chat's tool list (phase 0: each extra tool
/// makes wrong picks more likely and the first token later).
pub const MAX_TOOLS: usize = 64;

/// How to reach a server.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Transport {
    /// A program on this computer, spoken to over its stdin and stdout.
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
        /// Positions in `args` of secret values (a registry package's `isSecret` argument): the
        /// program gets them; every view of the server masks them.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        secret_args: Vec<usize>,
    },
    /// A Streamable HTTP endpoint.
    Http {
        url: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
    },
}

/// Why a server could not be reached or used.
#[derive(Debug, Clone, PartialEq)]
pub enum McpError {
    /// The program would not start.
    Spawn(String),
    /// The connection broke: the program exited, or the HTTP request failed.
    Io(String),
    /// No answer in time.
    Timeout(&'static str),
    /// An answer that is not MCP.
    Protocol(String),
    /// The server answered with a JSON-RPC error.
    Server { code: i64, message: String },
    /// The server wants a sign-in (a 401, or a 403 for a scope the token lacks): its challenge.
    NeedsAuth {
        challenge: Challenge,
        scope_short: bool,
    },
    /// A 403 that is not about scope: the account may not use this server.
    Forbidden,
    /// A 404 for a request in a session (Streamable HTTP): the server forgot the session and ran
    /// nothing, so a new one may send it again.
    Expired,
}

impl fmt::Display for McpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(e) => write!(f, "the server would not start: {e}"),
            Self::Io(e) => write!(f, "the connection to the server failed: {e}"),
            Self::Timeout(what) => write!(f, "the server did not answer {what} in time"),
            Self::Protocol(e) => write!(f, "the server's answer is not MCP: {e}"),
            Self::Server { code, message } => write!(f, "the server refused ({code}): {message}"),
            Self::NeedsAuth {
                scope_short: true, ..
            } => f.write_str(
                "this server needs a sign-in with more access: press Sign in on its card",
            ),
            Self::NeedsAuth { .. } => {
                f.write_str("this server needs a sign-in: press Sign in on its card")
            }
            Self::Forbidden => f.write_str("HTTP 403: the server refused this account (forbidden)"),
            Self::Expired => f.write_str("the server ended the session"),
        }
    }
}

impl std::error::Error for McpError {}

/// A tool as the server lists it.
#[derive(Clone, Debug, PartialEq)]
pub struct RemoteTool {
    /// The server's own name for it.
    pub name: String,
    /// What the server says it does.
    pub description: String,
    /// JSON Schema of its arguments.
    pub schema: Value,
    /// The server marks it `readOnlyHint` and not `openWorldHint`: it only reads, and nothing
    /// outside. Only the server says so, so anything less asks before it runs.
    pub read_only: bool,
}

/// A connected server.
pub struct Client {
    conn: Mutex<Conn>,
    /// `serverInfo.name` from `initialize`.
    pub server_name: String,
    /// The revision both sides speak.
    pub protocol: String,
    /// `instructions` from `initialize`, if any.
    pub instructions: Option<String>,
}

enum Conn {
    Stdio(StdioConn),
    Http(HttpConn),
}

struct StdioConn {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    next: u64,
    /// The Job Object the program runs in: closed with this process, it ends the program and
    /// what it started however the app ends; `Drop`'s taskkill runs only on an orderly close.
    #[cfg(windows)]
    _job: Option<win32job::Job>,
}

struct HttpConn {
    url: String,
    headers: BTreeMap<String, String>,
    session: Option<String>,
    protocol: Option<String>,
    next: u64,
    /// The sign-in, read on every request; `None` when the user set an Authorization header.
    auth: Option<Arc<Auth>>,
}

impl Client {
    /// Starts or reaches the server, then runs `initialize` and `notifications/initialized`.
    /// `stderr_log` collects a stdio server's stderr (otherwise it is dropped).
    pub fn connect(transport: &Transport, stderr_log: Option<PathBuf>) -> Result<Client, McpError> {
        Client::connect_with(transport, stderr_log, None)
    }

    /// [`Client::connect`] with a sign-in for an HTTP server ([`Auth`]); an `Authorization` header
    /// the user set wins, and the sign-in is not used.
    pub fn connect_with(
        transport: &Transport,
        stderr_log: Option<PathBuf>,
        auth: Option<Arc<Auth>>,
    ) -> Result<Client, McpError> {
        let conn = match transport {
            Transport::Stdio {
                command, args, env, ..
            } => Conn::Stdio(StdioConn::spawn(command, args, env, stderr_log)?),
            Transport::Http { url, headers } => {
                if !(url.starts_with("https://") || url.starts_with("http://")) {
                    return Err(McpError::Spawn(format!("'{url}' is not an http(s) URL")));
                }
                let own = headers
                    .keys()
                    .any(|k| k.eq_ignore_ascii_case("authorization"));
                Conn::Http(HttpConn {
                    url: url.clone(),
                    headers: headers.clone(),
                    session: None,
                    protocol: None,
                    next: 0,
                    auth: auth.filter(|_| !own),
                })
            }
        };
        let conn = Mutex::new(conn);
        let init = json!({
            "protocolVersion": PROTOCOL,
            "capabilities": {},
            "clientInfo": {"name": "larch", "version": env!("CARGO_PKG_VERSION")},
        });
        let wait = match transport {
            Transport::Stdio { .. } => START,
            Transport::Http { .. } => QUICK,
        };
        let result = request(&conn, "initialize", init, wait)?;
        let protocol = result["protocolVersion"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        if !(VERSIONS.contains(&protocol.as_str()) || protocol == FIRST) {
            return Err(McpError::Protocol(format!(
                "it speaks revision '{protocol}', not one of {VERSIONS:?} or {FIRST}"
            )));
        }
        if let Conn::Http(h) = &mut *lock(&conn) {
            h.protocol = Some(protocol.clone());
        }
        notify(&conn, "notifications/initialized")?;
        Ok(Client {
            conn,
            server_name: result["serverInfo"]["name"]
                .as_str()
                .unwrap_or("server")
                .to_owned(),
            protocol,
            instructions: result["instructions"].as_str().map(str::to_owned),
        })
    }

    /// Every tool the server lists (following `nextCursor`), at most [`MAX_TOOLS`].
    pub fn tools(&self) -> Result<Vec<RemoteTool>, McpError> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = match &cursor {
                Some(c) => json!({"cursor": c}),
                None => json!({}),
            };
            let page = request(&self.conn, "tools/list", params, QUICK)?;
            let Some(list) = page["tools"].as_array() else {
                return Err(McpError::Protocol("tools/list has no tools array".into()));
            };
            for t in list {
                let Some(name) = t["name"].as_str() else {
                    continue;
                };
                tools.push(RemoteTool {
                    name: name.to_owned(),
                    description: t["description"]
                        .as_str()
                        .or(t["title"].as_str())
                        .unwrap_or(name)
                        .to_owned(),
                    schema: t["inputSchema"].clone(),
                    read_only: t["annotations"]["readOnlyHint"] == json!(true)
                        && t["annotations"]["openWorldHint"] == json!(false),
                });
            }
            cursor = page["nextCursor"].as_str().map(str::to_owned);
            if cursor.is_none() || tools.len() >= MAX_TOOLS {
                break;
            }
        }
        tools.truncate(MAX_TOOLS);
        Ok(tools)
    }

    /// Runs `name` with `args`: the result's text content, or the server's error text as `Err`
    /// when it marks the result `isError`.
    pub fn call(&self, name: &str, args: &Value) -> Result<Result<String, String>, McpError> {
        let result = request(
            &self.conn,
            "tools/call",
            json!({"name": name, "arguments": args}),
            CALL,
        )?;
        let text = content_text(&result);
        Ok(match result["isError"] == json!(true) {
            true => Err(text),
            false => Ok(text),
        })
    }
}

/// The program goes with its connection, however that ends: a closed client, or a start that
/// failed (an unsupported revision, an `initialize` that timed out) before there was a client.
impl Drop for StdioConn {
    fn drop(&mut self) {
        // npx and uvx start the server as their child: the whole group goes, or it outlives us.
        kill_tree(self.child.id());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The server's whole process group, which `shell` started it as. `kill -SIG -- -pgid` is the
/// POSIX form for a group, which Linux's procps and macOS's BSD `kill` both take.
#[cfg(unix)]
fn kill_tree(pid: u32) {
    let _ = Command::new("kill")
        .args(["-KILL", "--", &format!("-{pid}")])
        .stderr(Stdio::null())
        .status();
}

#[cfg(windows)]
fn kill_tree(pid: u32) {
    use std::os::windows::process::CommandExt;
    let _ = Command::new("taskkill")
        .args(["/T", "/F", "/PID", &pid.to_string()])
        .creation_flags(0x0800_0000)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

/// The text of a `tools/call` result: its text blocks in order; other blocks are named, and a
/// result with only `structuredContent` is that JSON.
fn content_text(result: &Value) -> String {
    let mut parts = Vec::new();
    for block in result["content"].as_array().into_iter().flatten() {
        match block["type"].as_str() {
            Some("text") => parts.push(block["text"].as_str().unwrap_or_default().to_owned()),
            Some("resource") => parts.push(
                (block["resource"]["text"].as_str())
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("[resource {}]", block["resource"]["uri"])),
            ),
            Some("resource_link") => parts.push(format!("[link {}]", block["uri"])),
            Some(other) => parts.push(format!("[{other} content omitted]")),
            None => {}
        }
    }
    if parts.is_empty()
        && let Some(s) = result.get("structuredContent")
    {
        return s.to_string();
    }
    parts.join("\n")
}

fn lock(conn: &Mutex<Conn>) -> std::sync::MutexGuard<'_, Conn> {
    conn.lock().unwrap_or_else(PoisonError::into_inner)
}

fn request(
    conn: &Mutex<Conn>,
    method: &str,
    params: Value,
    wait: Duration,
) -> Result<Value, McpError> {
    let mut c = lock(conn);
    let reply = match &mut *c {
        Conn::Stdio(s) => s.request(method, params, wait)?,
        Conn::Http(h) => h.request(method, params, wait)?,
    };
    if let Some(e) = reply.get("error") {
        return Err(McpError::Server {
            code: e["code"].as_i64().unwrap_or(0),
            message: e["message"].as_str().unwrap_or("no message").to_owned(),
        });
    }
    Ok(reply["result"].clone())
}

fn notify(conn: &Mutex<Conn>, method: &str) -> Result<(), McpError> {
    let message = json!({"jsonrpc": "2.0", "method": method});
    match &mut *lock(conn) {
        Conn::Stdio(s) => s.send(&message),
        Conn::Http(h) => h.post(&message, QUICK).map(|_| ()),
    }
}

impl StdioConn {
    fn spawn(
        command: &str,
        args: &[String],
        env: &BTreeMap<String, String>,
        stderr_log: Option<PathBuf>,
    ) -> Result<StdioConn, McpError> {
        let mut cmd = shell(command, args);
        cmd.envs(env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(
                match stderr_log.and_then(|p| std::fs::File::create(p).ok()) {
                    Some(f) => Stdio::from(f),
                    None => Stdio::null(),
                },
            );
        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::Spawn(format!("{command}: {e}")))?;
        // Without a job the program still ends on an orderly close (`Drop`), as it did before.
        #[cfg(windows)]
        let job = {
            use std::os::windows::io::AsRawHandle;
            let mut info = win32job::ExtendedLimitInfo::new();
            info.limit_kill_on_job_close();
            win32job::Job::create_with_limit_info(&info)
                .and_then(|job| {
                    job.assign_process(child.as_raw_handle() as isize)
                        .map(|()| job)
                })
                .ok()
        };
        let stdin = child
            .stdin
            .take()
            .ok_or(McpError::Spawn("no stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or(McpError::Spawn("no stdout".into()))?;
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            // Bytes, not `lines()`: one log line in the console's code page (a path with é under
            // cp1252) would end the reader, and with it every later request.
            for line in BufReader::new(stdout).split(b'\n') {
                let Ok(line) = line else { break };
                let line = String::from_utf8_lossy(&line);
                if tx.send(line.trim_end_matches('\r').to_owned()).is_err() {
                    break;
                }
            }
        });
        Ok(StdioConn {
            child,
            stdin,
            lines,
            next: 0,
            #[cfg(windows)]
            _job: job,
        })
    }

    fn send(&mut self, message: &Value) -> Result<(), McpError> {
        let mut line = message.to_string();
        line.push('\n');
        (self.stdin.write_all(line.as_bytes()))
            .and_then(|()| self.stdin.flush())
            .map_err(|e| McpError::Io(e.to_string()))
    }

    fn request(&mut self, method: &str, params: Value, wait: Duration) -> Result<Value, McpError> {
        self.next += 1;
        let id = self.next;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        let deadline = Instant::now() + wait;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let line = match self.lines.recv_timeout(left) {
                Ok(line) => line,
                Err(RecvTimeoutError::Timeout) => {
                    return Err(McpError::Timeout(static_name(method)));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let status = self.child.try_wait().ok().flatten();
                    return Err(McpError::Io(match status {
                        Some(s) => format!("the program exited ({s})"),
                        None => "the program closed its output".into(),
                    }));
                }
            };
            let Ok(message) = serde_json::from_str::<Value>(line.trim()) else {
                continue; // a log line a careless server printed to stdout
            };
            if message["id"] == json!(id) && message.get("method").is_none() {
                return Ok(message);
            }
            // A request from the server: answer ping, refuse the rest (no sampling, roots or
            // elicitation in this client). Notifications are ignored.
            if let (Some(m), Some(rid)) = (message["method"].as_str(), message.get("id")) {
                let reply = match m {
                    "ping" => json!({"jsonrpc": "2.0", "id": rid, "result": {}}),
                    _ => json!({"jsonrpc": "2.0", "id": rid,
                        "error": {"code": -32601, "message": format!("{m} is not supported")}}),
                };
                self.send(&reply)?;
            }
        }
    }
}

impl HttpConn {
    fn request(&mut self, method: &str, params: Value, wait: Duration) -> Result<Value, McpError> {
        self.next += 1;
        let id = self.next;
        let reply = self.post(
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}),
            wait,
        )?;
        reply.ok_or_else(|| McpError::Protocol(format!("no reply to {method}")))
    }

    /// POSTs one message; the JSON-RPC reply carrying its id, if it has one.
    fn post(&mut self, message: &Value, wait: Duration) -> Result<Option<Value>, McpError> {
        let token = self.auth.as_ref().and_then(|a| a.token_for(&self.url));
        match self.post_with(message, wait, token.as_deref()) {
            // A token that went stale between the check and the request: refresh once (one
            // refresh for every call that finds it stale), then try again.
            Err(McpError::NeedsAuth {
                scope_short: false, ..
            }) if token.is_some() => {
                let auth = self.auth.clone().expect("a token came from it");
                match auth.refresh_after(token.as_deref().unwrap_or_default()) {
                    Ok(fresh) => self.post_with(message, wait, Some(&fresh)),
                    Err(RefreshError::SignInAgain(_)) => Err(McpError::NeedsAuth {
                        challenge: Challenge::default(),
                        scope_short: false,
                    }),
                    Err(RefreshError::Transient(m)) => Err(McpError::Io(m)),
                }
            }
            other => other,
        }
    }

    fn post_with(
        &mut self,
        message: &Value,
        wait: Duration,
        token: Option<&str>,
    ) -> Result<Option<Value>, McpError> {
        // Redirects are not followed: ureq hands back a POST's 307 as the answer and turns a
        // 302 into a GET, and either read as "not MCP"; the error names the address instead.
        let agent = ureq::AgentBuilder::new().redirects(0).build();
        let mut req = (agent.post(&self.url))
            .timeout(wait)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json, text/event-stream");
        for (k, v) in &self.headers {
            req = req.set(k, v);
        }
        if let Some(s) = &self.session {
            req = req.set("Mcp-Session-Id", s);
        }
        if let Some(p) = &self.protocol {
            req = req.set("MCP-Protocol-Version", p);
        }
        if let Some(t) = token {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        let resp = match req.send_string(&message.to_string()) {
            Ok(r) => r,
            // A sign-in is needed (401), or more of one (403 insufficient_scope); a bare 403 is a
            // refusal no sign-in fixes.
            Err(ureq::Error::Status(code @ (401 | 403), r)) => {
                let challenge = r
                    .header("WWW-Authenticate")
                    .and_then(super::oauth::parse_challenge);
                let scope_short = challenge
                    .as_ref()
                    .is_some_and(|c| c.error.as_deref() == Some("insufficient_scope"));
                return Err(match (code, challenge) {
                    (401, c) => McpError::NeedsAuth {
                        challenge: c.unwrap_or_default(),
                        scope_short: false,
                    },
                    (_, Some(c)) if scope_short => McpError::NeedsAuth {
                        challenge: c,
                        scope_short: true,
                    },
                    _ => McpError::Forbidden,
                });
            }
            Err(ureq::Error::Status(404, _)) if self.session.is_some() => {
                self.session = None;
                return Err(McpError::Expired);
            }
            Err(ureq::Error::Status(code, r)) => {
                let body = r.into_string().unwrap_or_default();
                return Err(McpError::Io(format!(
                    "HTTP {code}: {}",
                    body.chars().take(200).collect::<String>()
                )));
            }
            Err(e) => return Err(McpError::Io(e.to_string())),
        };
        if (300..400).contains(&resp.status()) {
            let to = resp.header("Location").unwrap_or("another address");
            return Err(McpError::Io(format!(
                "HTTP {}: the server sends this address on to {to}; use that one",
                resp.status()
            )));
        }
        if let Some(s) = resp.header("Mcp-Session-Id") {
            self.session = Some(s.to_owned());
        }
        let Some(id) = message.get("id").cloned() else {
            return Ok(None); // a notification: 202 and no body
        };
        let sse = resp.content_type() == "text/event-stream";
        let mut body = String::new();
        if !sse {
            resp.into_reader()
                .take(16 << 20)
                .read_to_string(&mut body)
                .map_err(|e| McpError::Io(e.to_string()))?;
            return serde_json::from_str(&body)
                .map(Some)
                .map_err(|e| McpError::Protocol(e.to_string()));
        }
        // An event stream: the reply is the `data:` event whose JSON carries our id and no method;
        // the server's own requests (a ping) count their ids from 1 too.
        let ours = |v: &Value| v["id"] == id && v.get("method").is_none();
        let mut data = String::new();
        for line in BufReader::new(resp.into_reader().take(16 << 20)).lines() {
            let line = line.map_err(|e| McpError::Io(e.to_string()))?;
            if let Some(d) = line.strip_prefix("data:") {
                data.push_str(d.trim_start());
            } else if line.is_empty() && !data.is_empty() {
                if let Ok(v) = serde_json::from_str::<Value>(&data)
                    && ours(&v)
                {
                    return Ok(Some(v));
                }
                data.clear();
            }
        }
        match serde_json::from_str::<Value>(&data) {
            Ok(v) if ours(&v) => Ok(Some(v)),
            _ => Err(McpError::Protocol(
                "the event stream ended without a reply".into(),
            )),
        }
    }
}

/// `command args` as a process in its own group (so [`kill_tree`] reaches its children). On
/// Windows `npx` and `uvx` are `.cmd` scripts `CreateProcess` cannot start by bare name: the name
/// is found on PATH with PATHEXT and started directly, so Rust quotes the arguments for a batch
/// file (a `cmd /C` line would not, and a registry's arguments could carry commands).
fn shell(command: &str, args: &[String]) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new(windows_program(command));
        c.args(args);
        c.creation_flags(0x0800_0000 | 0x0000_0200); // CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        c
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let mut c = Command::new(command);
        c.args(args).process_group(0);
        c
    }
}

#[cfg(windows)]
fn windows_program(command: &str) -> PathBuf {
    let path = std::path::Path::new(command);
    if path.extension().is_some() || path.components().count() > 1 {
        return path.into();
    }
    let exts = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into());
    for dir in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        for ext in exts.split(';').filter(|e| !e.is_empty()) {
            let candidate = dir.join(format!("{command}{}", ext.to_ascii_lowercase()));
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    path.into()
}

fn static_name(method: &str) -> &'static str {
    match method {
        "initialize" => "initialize",
        "tools/list" => "tools/list",
        "tools/call" => "a tool call",
        _ => "a request",
    }
}

/// `<server>__<tool>` as a tool name models handle: lower-case letters, digits and `_`, at most
/// 64 characters.
pub fn tool_name(server: &str, tool: &str) -> String {
    fn clean(s: &str) -> String {
        let mut out = String::new();
        for c in s.chars() {
            let c = c.to_ascii_lowercase();
            let c = if c.is_ascii_alphanumeric() { c } else { '_' };
            if !(c == '_' && out.ends_with('_')) {
                out.push(c);
            }
        }
        out.trim_matches('_').to_owned()
    }
    let mut name = format!("{}__{}", clean(server), clean(tool));
    name.truncate(64);
    name
}

/// One installed server as every chat shares it: its last tool list, and a connection made on
/// demand and made again after it breaks. Turned off, its tools in chats already open fail.
pub struct Server {
    /// The id the tools' names start with.
    pub id: String,
    transport: Transport,
    log: Option<PathBuf>,
    on: std::sync::atomic::AtomicBool,
    client: Mutex<Option<Arc<Client>>>,
    tools: Mutex<Vec<RemoteTool>>,
    auth: Option<Arc<Auth>>,
    /// Told when a call ends needing a sign-in (the app shows it on the server's card).
    on_needs_auth: Mutex<Option<NeedsAuthHook>>,
}

type NeedsAuthHook = Arc<dyn Fn(&McpError) + Send + Sync>;

impl Server {
    pub fn new(id: &str, transport: Transport, log: Option<PathBuf>) -> Arc<Server> {
        Server::with_auth(id, transport, log, None)
    }

    /// A server whose HTTP requests carry this sign-in's token.
    pub fn with_auth(
        id: &str,
        transport: Transport,
        log: Option<PathBuf>,
        auth: Option<Arc<Auth>>,
    ) -> Arc<Server> {
        Arc::new(Server {
            id: id.into(),
            transport,
            log,
            on: std::sync::atomic::AtomicBool::new(true),
            client: Mutex::default(),
            tools: Mutex::default(),
            auth,
            on_needs_auth: Mutex::default(),
        })
    }

    /// The sign-in its requests use, if any.
    pub fn auth(&self) -> Option<&Arc<Auth>> {
        self.auth.as_ref()
    }

    pub fn set_on_needs_auth(&self, hook: NeedsAuthHook) {
        *self
            .on_needs_auth
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(hook);
    }

    /// Drops the connection, so the next call connects again (after a sign-in or a sign-out).
    pub fn reconnect(&self) {
        self.client
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
    }

    /// Connects if it is not connected, and lists the tools again.
    pub fn connect(&self) -> Result<Vec<RemoteTool>, McpError> {
        use std::sync::atomic::Ordering;
        self.on.store(true, Ordering::Relaxed);
        let mut client = self.client()?;
        let mut listed = client.tools();
        // A session the server forgot (Streamable HTTP's 404): one new session lists them.
        if matches!(listed, Err(McpError::Expired)) {
            self.reconnect();
            client = self.client()?;
            listed = client.tools();
        }
        let tools = match listed {
            Ok(t) => t,
            Err(e) => {
                *self.client.lock().unwrap_or_else(PoisonError::into_inner) = None;
                return Err(e);
            }
        };
        *self.tools.lock().unwrap_or_else(PoisonError::into_inner) = tools.clone();
        Ok(tools)
    }

    /// The tools from the last [`Server::connect`]; none before one.
    pub fn tools(&self) -> Vec<RemoteTool> {
        self.tools
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Turns it off: the connection closes (its program stops) and calls fail until a connect.
    pub fn close(&self) {
        self.on.store(false, std::sync::atomic::Ordering::Relaxed);
        self.client
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
    }

    fn client(&self) -> Result<Arc<Client>, McpError> {
        let mut slot = self.client.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(c) = &*slot {
            return Ok(c.clone());
        }
        let c = Arc::new(Client::connect_with(
            &self.transport,
            self.log.clone(),
            self.auth.clone(),
        )?);
        *slot = Some(c.clone());
        Ok(c)
    }

    /// Runs one tool. A broken connection is dropped, so the next call starts it again.
    pub fn call(&self, name: &str, args: &Value) -> Result<Result<String, String>, McpError> {
        if !self.on.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(McpError::Io(format!(
                "the server '{}' is turned off",
                self.id
            )));
        }
        let mut out = self.client().and_then(|client| client.call(name, args));
        // The server forgot the session and ran nothing: the call goes once more on a new one.
        if matches!(out, Err(McpError::Expired)) {
            self.reconnect();
            out = self.client().and_then(|client| client.call(name, args));
        }
        if matches!(
            out,
            Err(McpError::Io(_)
                | McpError::Timeout(_)
                | McpError::NeedsAuth { .. }
                | McpError::Expired)
        ) {
            self.reconnect();
        }
        // With no lock held: the app's hook takes its own.
        if let Err(e @ McpError::NeedsAuth { .. }) = &out {
            let hook = self
                .on_needs_auth
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            if let Some(h) = hook {
                h(e);
            }
        }
        out
    }

    /// The chat's tools from the last tool list, one [`McpTool`] each.
    pub fn chat_tools(self: &Arc<Self>) -> Vec<McpTool> {
        let tools = self.tools();
        let needs = needs(&tools);
        (tools.into_iter().zip(needs))
            .map(|(remote, needs)| McpTool {
                needs: (needs.iter()).map(|n| tool_name(&self.id, n)).collect(),
                ..McpTool::new(self.clone(), remote)
            })
            .collect()
    }
}

/// For each of a server's `tools`, the others its text names (its description or schema):
/// Notion's create-view says "Use fetch first to get database_id". A name counts as its words
/// without the leading ones every name shares, so "query_data_sources" names
/// `notion-query-data-sources`.
fn needs(tools: &[RemoteTool]) -> Vec<Vec<String>> {
    fn words(s: &str) -> Vec<String> {
        (s.split(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect()
    }
    let names: Vec<Vec<String>> = tools.iter().map(|t| words(&t.name)).collect();
    // Every name keeps at least one word of its own.
    let shared = (0..)
        .take_while(|&i| {
            names.len() > 1 && (names.iter()).all(|n| n.len() > i + 1 && n[i] == names[0][i])
        })
        .count();
    (tools.iter().enumerate())
        .map(|(i, t)| {
            let text = words(&format!("{} {}", t.description, t.schema));
            (names.iter().enumerate())
                .filter(|&(j, n)| j != i && n.len() > shared)
                .filter(|(_, n)| text.windows(n.len() - shared).any(|w| w == &n[shared..]))
                .map(|(j, _)| tools[j].name.clone())
                .collect()
        })
        .collect()
}

/// One outside tool as a chat sees it, named `<server id>__<tool>`.
pub struct McpTool {
    server: Arc<Server>,
    remote: String,
    spec: ToolSpec,
    /// Names a bare repository (`repo::find`; tests stand in).
    find: fn(&str) -> Option<String>,
    /// Its server's tools its text names (`needs`), as the chat names them.
    needs: Vec<String>,
}

/// The spec a chat sees for `remote`, a tool of the server `server_id`.
pub fn spec(server_id: &str, remote: &RemoteTool) -> ToolSpec {
    // The server's schema as it is, with `type: object` when it leaves the type out.
    let mut schema = match remote.schema {
        Value::Object(_) => remote.schema.clone(),
        _ => json!({"type": "object", "properties": {}}),
    };
    if let Some(o) = schema.as_object_mut() {
        o.entry("type").or_insert_with(|| "object".into());
    }
    // The compiled form: the purpose sentence, no examples, parameters cut short. With
    // retrieval it took Qwen3 1.7B from 53.5% to 62.6% right on 99 MCP rows (toolcall-opt
    // items 6 and 7).
    let description = super::describe::compile(&remote.description, super::describe::MAX_CHARS);
    let schema = super::describe::compile_schema(&schema);
    ToolSpec {
        name: tool_name(server_id, &remote.name),
        description,
        schema,
        // Unless the server says it only reads, and nothing outside, a call asks as any
        // outside write does.
        permission: match remote.read_only {
            true => Permission::Read,
            false => Permission::WriteExternal,
        },
        untrusted_results: true,
        max_result_tokens: 2048,
        max_args_tokens: 512,
    }
}

impl McpTool {
    pub fn new(server: Arc<Server>, remote: RemoteTool) -> McpTool {
        McpTool {
            spec: spec(&server.id, &remote),
            server,
            remote: remote.name,
            find: super::repo::find,
            needs: Vec::new(),
        }
    }
}

impl Tool for McpTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    /// The arguments as the call sends them, once: a bare repository name the server wants as
    /// owner/repo made one (`repo`). A write is asked about first, and nothing leaves the computer
    /// before the yes, so its name is made one only from a name a read looked up already; a read
    /// runs unasked, and looks it up (GitHub's search) here.
    fn prepare(&self, args: &Value, _ctx: &ToolCtx) -> Result<Value, ToolError> {
        let find = match self.spec.permission {
            Permission::Read => self.find,
            _ => super::repo::cached,
        };
        Ok(super::repo::resolve(&self.spec.schema, args, find).unwrap_or_else(|| args.clone()))
    }

    fn needs(&self) -> Vec<String> {
        self.needs.clone()
    }

    /// Sends `args` as they are: the ones prepared, allowed and audited, never resolved again.
    fn call(&self, args: &Value, _ctx: &ToolCtx) -> Result<String, ToolError> {
        match self.server.call(&self.remote, args) {
            Ok(Ok(text)) => Ok(text),
            Ok(Err(text)) => Err(ToolError::Outside(text)),
            Err(e) => Err(ToolError::Outside(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Streamable HTTP server on loopback: each `initialize` opens a new session, and a request
    /// in any other session than the newest gets 404, as after the server restarted. Counts the
    /// `initialize`s.
    fn sessions_server(
        forget: Arc<Mutex<bool>>,
        protocol: &'static str,
    ) -> (String, Arc<Mutex<u32>>) {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let inits = Arc::new(Mutex::new(0u32));
        let count = inits.clone();
        std::thread::spawn(move || {
            // The newest session's number; a forget skips past every one handed out.
            let mut newest = 0u32;
            for mut s in listener.incoming().flatten() {
                let mut r = BufReader::new(s.try_clone().unwrap());
                let (mut len, mut session) = (0, None);
                loop {
                    let mut line = String::new();
                    r.read_line(&mut line).unwrap();
                    let line = line.trim_end();
                    if line.is_empty() {
                        break;
                    }
                    let (k, v) = line.split_once(':').unwrap_or((line, ""));
                    match k.to_ascii_lowercase().as_str() {
                        "content-length" => len = v.trim().parse().unwrap(),
                        "mcp-session-id" => session = Some(v.trim().to_owned()),
                        _ => {}
                    }
                }
                let mut body = vec![0; len];
                r.read_exact(&mut body).unwrap();
                let msg: Value = serde_json::from_slice(&body).unwrap();
                if std::mem::take(&mut *forget.lock().unwrap()) {
                    newest += 100;
                }
                let live = format!("s{newest}");
                let (status, header, result) = match msg["method"].as_str() {
                    Some("initialize") => {
                        newest += 1;
                        *count.lock().unwrap() += 1;
                        let info = json!({"protocolVersion": protocol, "capabilities": {},
                            "serverInfo": {"name": "fake"}});
                        ("200 OK", format!("s{newest}"), Some(info))
                    }
                    _ if session.as_deref() != Some(live.as_str()) => ("404 Not Found", live, None),
                    _ if msg.get("id").is_none() => ("202 Accepted", live, None),
                    Some("tools/list") => (
                        "200 OK",
                        live,
                        Some(
                            json!({"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}),
                        ),
                    ),
                    _ => (
                        "200 OK",
                        live,
                        Some(json!({"content": [{"type": "text", "text": "echoed"}]})),
                    ),
                };
                let body = result
                    .map(|r| json!({"jsonrpc": "2.0", "id": msg["id"], "result": r}).to_string())
                    .unwrap_or_default();
                let _ = write!(
                    s,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nMcp-Session-Id: {header}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        (url, inits)
    }

    /// A call in a session the server forgot (its 404) runs once more in a new session, rather
    /// than failing to the model; listing the tools does too.
    #[test]
    fn a_forgotten_session_is_started_again_and_the_call_runs() {
        let forget = Arc::new(Mutex::new(false));
        let (url, inits) = sessions_server(forget.clone(), PROTOCOL);
        let transport = Transport::Http {
            url,
            headers: BTreeMap::new(),
        };
        let server = Server::with_auth("fake", transport, None, None);
        assert_eq!(server.connect().unwrap().len(), 1);
        *forget.lock().unwrap() = true;
        assert_eq!(
            server.call("echo", &json!({})).unwrap(),
            Ok("echoed".into())
        );
        *forget.lock().unwrap() = true;
        assert_eq!(server.connect().unwrap().len(), 1);
        assert_eq!(*inits.lock().unwrap(), 3);
    }

    /// A server that redirects its endpoint (`/mcp` to `/mcp/`) is not followed into a parse error:
    /// the error names where it points.
    #[test]
    fn a_redirect_names_the_address_it_points_to() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        let to = format!("{url}/");
        let moved = to.clone();
        std::thread::spawn(move || {
            for mut s in listener.incoming().flatten() {
                // The whole request is read: a close with bytes unread resets the connection on
                // Windows, before the client reads the answer.
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut len = 0;
                loop {
                    let mut line = String::new();
                    r.read_line(&mut line).unwrap();
                    let line = line.trim_end().to_ascii_lowercase();
                    if line.is_empty() {
                        break;
                    }
                    if let Some(v) = line.strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap();
                    }
                }
                r.read_exact(&mut vec![0; len]).unwrap();
                let _ = write!(
                    s,
                    "HTTP/1.1 307 Temporary Redirect\r\nLocation: {moved}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
            }
        });
        let transport = Transport::Http {
            url,
            headers: BTreeMap::new(),
        };
        match Client::connect(&transport, None) {
            Err(McpError::Io(m)) => assert!(m.contains("307") && m.contains(&to), "{m}"),
            other => panic!("{:?}", other.map(|c| c.server_name)),
        }
    }

    /// A server that answers in the first revision is used; one in a revision never published is
    /// refused, naming it.
    #[test]
    fn the_first_revision_is_spoken_and_an_unknown_one_refused() {
        let connect = |protocol| {
            let (url, _) = sessions_server(Arc::default(), protocol);
            let transport = Transport::Http {
                url,
                headers: BTreeMap::new(),
            };
            Server::with_auth("fake", transport, None, None).connect()
        };
        assert_eq!(connect(FIRST).unwrap().len(), 1);
        match connect("1999-01-01") {
            Err(McpError::Protocol(m)) => assert!(m.contains("1999-01-01"), "{m}"),
            other => panic!("{:?}", other.map(|t| t.len())),
        }
    }

    #[test]
    fn tool_names_are_safe_and_short() {
        assert_eq!(
            tool_name("Time", "get_current_time"),
            "time__get_current_time"
        );
        assert_eq!(
            tool_name("ai.b77/time", "Get Time!"),
            "ai_b77_time__get_time"
        );
        assert_eq!(tool_name("x", &"a".repeat(100)).len(), 64);
    }

    #[test]
    fn result_text_joins_text_blocks_and_names_others() {
        let r = json!({"content": [{"type": "text", "text": "a"}, {"type": "image", "data": "…"},
            {"type": "text", "text": "b"}]});
        assert_eq!(content_text(&r), "a\n[image content omitted]\nb");
        let s = json!({"content": [], "structuredContent": {"n": 1}});
        assert_eq!(content_text(&s), r#"{"n":1}"#);
    }

    #[test]
    fn a_program_that_does_not_exist_fails_to_spawn() {
        let t = Transport::Stdio {
            command: "/nonexistent/mcp-server".into(),
            args: vec![],
            env: BTreeMap::new(),
            secret_args: Vec::new(),
        };
        assert!(matches!(Client::connect(&t, None), Err(McpError::Spawn(_))));
    }

    #[test]
    fn transport_serializes_with_a_type_tag() {
        let t = Transport::Http {
            url: "https://x/mcp".into(),
            headers: BTreeMap::new(),
        };
        let v = serde_json::to_value(&t).unwrap();
        assert_eq!(v["type"], "http");
        assert_eq!(serde_json::from_value::<Transport>(v).unwrap(), t);
    }

    /// A tool needs the siblings its description or schema names, by their words past the shared
    /// prefix, whatever the separators; never itself.
    #[test]
    fn a_tool_needs_the_siblings_its_text_names() {
        let tool = |name: &str, description: &str| RemoteTool {
            name: name.into(),
            description: description.into(),
            schema: json!({"type": "object", "properties": {}}),
            read_only: false,
        };
        let mut tools = vec![
            tool("notion-search", "Search pages and databases."),
            tool(
                "notion-fetch",
                "Retrieves a page. Use query_data_sources for rows.",
            ),
            tool("notion-query-data-sources", "Query data source rows."),
            tool(
                "notion-create-view",
                "Create a new view. Use fetch first to get database_id.",
            ),
            tool("notion-create-pages", "Creates pages."),
        ];
        tools[4].schema = json!({"type": "object", "properties": {"parent": {"type": "string",
            "description": "A page found with search."}}});
        let got = needs(&tools);
        assert_eq!(got[0], Vec::<String>::new(), "search names no sibling");
        assert_eq!(got[1], ["notion-query-data-sources"]);
        assert_eq!(
            got[2],
            Vec::<String>::new(),
            "\"query\" alone is not a name"
        );
        assert_eq!(got[3], ["notion-fetch"]);
        assert_eq!(got[4], ["notion-search"]);
        // One tool, or names with nothing in common: whole names.
        assert_eq!(needs(&tools[..1]), [Vec::<String>::new()]);
        let other = [
            tool("lookup", "Find an id."),
            tool("write", "Call lookup first."),
        ];
        assert_eq!(needs(&other)[1], ["lookup"]);
    }

    fn repo_tool(read_only: bool, find: fn(&str) -> Option<String>) -> McpTool {
        let server = Server::with_auth(
            "wiki",
            Transport::Http {
                url: "http://127.0.0.1:9/mcp".into(),
                headers: BTreeMap::new(),
            },
            None,
            None,
        );
        let remote = RemoteTool {
            name: "ask".into(),
            description: "Ask about a repository".into(),
            schema: json!({"type": "object", "properties": {"repoName": {"type": "string",
                "description": "GitHub repository: owner/repo (e.g. \"facebook/react\")"}}}),
            read_only,
        };
        McpTool {
            find,
            ..McpTool::new(server, remote)
        }
    }

    fn ctx() -> ToolCtx {
        ToolCtx {
            allowed: vec![],
            data_dir: std::env::temp_dir(),
            source: "test".into(),
        }
    }

    /// A write is asked about first: preparing it sends nothing anywhere (no search for a bare
    /// repository name), so a no leaves nothing sent. A read resolves its name as it is prepared.
    #[test]
    fn preparing_a_write_searches_for_nothing_and_a_read_resolves_once() {
        let args = json!({"repoName": "ollama"});
        let write = repo_tool(false, |n| panic!("searched GitHub for {n} before the yes"));
        assert_eq!(write.prepare(&args, &ctx()).unwrap(), args);
        let read = repo_tool(true, |n| Some(format!("{n}/{n}")));
        assert_eq!(
            read.prepare(&args, &ctx()).unwrap(),
            json!({"repoName": "ollama/ollama"})
        );
    }

    /// The call sends what it is given, the arguments the user allowed and the audit holds: it
    /// never resolves them again, so a search that now succeeds cannot change them.
    #[test]
    fn a_call_sends_its_arguments_unresolved() {
        let tool = repo_tool(false, |n| panic!("searched GitHub for {n} at the call"));
        // No server listens: the call fails, having sent `ollama` as it was.
        assert!(tool.call(&json!({"repoName": "ollama"}), &ctx()).is_err());
    }
}
