//! The MCP servers the user installed from the app's MCP tools page: kept in `<data>/mcp.json`,
//! their sign-ins in `<data>/mcp-auth.json` (both readable by the user alone: they can hold keys
//! and tokens), connected in the background, and their tools handed to each chat as it opens
//! (`Hub::chat_tools`). A remote server behind OAuth is signed in to with `Hub::sign_in`.

use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Sender},
    },
    time::Duration,
};

use larch_agent::{
    mcp::{
        client::{self, McpError, RemoteTool, Server, Transport},
        oauth::{self, Auth, Callback, Grant, Registered},
    },
    runtime::driver::{StepTool, StepToolSource},
    tool::{Tool, ToolCtx, ToolError, ToolSpec},
};
use serde::{Deserialize, Serialize};

use crate::contract::{McpChanged, McpSearch, McpServer, McpServerStatus, McpSignIn, McpToolInfo};
use crate::{Emit, Event};

/// The official MCP Registry's server list.
const REGISTRY: &str = "https://registry.modelcontextprotocol.io/v0/servers";
/// Servers one user can install.
const MAX_SERVERS: usize = 32;
const MASK: &str = "••••";
/// How long a sign-in waits for the browser.
const SIGN_IN_WAIT: Duration = Duration::from_secs(300);

#[derive(Clone, Debug, PartialEq)]
enum State {
    Off,
    Connecting,
    Connected,
    NeedsAuth(String),
    Error(String),
}

struct Entry {
    saved: McpServer,
    live: Arc<Server>,
    state: State,
    /// Bumped by a new sign-in, a removal or a turn-off: an older sign-in's result is dropped.
    attempt: u64,
    /// Where a pasted final URL goes while a sign-in waits.
    paste: Option<Sender<String>>,
    /// The scope a call's 403 (`insufficient_scope`) asked for: the next sign-in asks for it.
    step_up: Option<String>,
    /// Its install is in mcp.json, so a grant naming it outlives this session (`grants`).
    persisted: bool,
}

/// `<data>/mcp-auth.json`: each server's sign-in, and the client each authorization server
/// registered (reused while its loopback port is free).
#[derive(Default, Serialize, Deserialize)]
struct AuthFile {
    #[serde(default)]
    grants: BTreeMap<String, Grant>,
    #[serde(default)]
    clients: BTreeMap<String, Registered>,
}

/// The installed servers and their connections.
pub struct Hub {
    file: PathBuf,
    auth_file: Arc<(PathBuf, Mutex<AuthFile>)>,
    logs: PathBuf,
    emit: Emit,
    entries: Arc<Mutex<Vec<Entry>>>,
    /// Changes whenever the tools a new chat would get may change; an idle chat opened at another
    /// version reopens (`App::open`).
    version: Arc<AtomicU64>,
}

/// `path`'s JSON, or the default when there is none. One that cannot be read or parsed (a hand
/// edit, a newer version's format) is moved to `<file>.bad`: the next save writes only what
/// loaded and would end every server or sign-in in it.
fn read_saved<T: serde::de::DeserializeOwned + Default>(path: &Path) -> T {
    let got = fs::read(path)
        .map_err(|e| e.kind() == std::io::ErrorKind::NotFound)
        .and_then(|b| serde_json::from_slice(&b).map_err(|_| false));
    match got {
        Ok(v) => v,
        Err(true) => T::default(),
        Err(false) => {
            // A file that cannot be moved cannot be replaced either: the save fails, the file stays.
            let _ = fs::rename(path, path.with_extension("json.bad"));
            T::default()
        }
    }
}

/// Writes `json` to `path` readable by the user alone, through a synced temporary file.
fn write_private(path: &Path, json: &[u8]) -> Result<(), String> {
    let temp = path.with_extension("json.tmp");
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    (opts.open(&temp))
        .and_then(|mut f| {
            f.write_all(json)?;
            f.sync_all()
        })
        .and_then(|()| fs::rename(&temp, path))
        .map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// A step tool that runs only while its server (id and install) is still on and trusted, so
/// untrusting a server stops a step's calls at once rather than at the next step. Otherwise it
/// runs as its tool.
struct Trusted {
    tool: Box<dyn Tool>,
    entries: std::sync::Weak<Mutex<Vec<Entry>>>,
    id: String,
    install: String,
}

impl Tool for Trusted {
    fn spec(&self) -> ToolSpec {
        self.tool.spec()
    }

    fn call(&self, args: &serde_json::Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        let entries = self.entries.upgrade();
        let live = entries.as_ref().is_some_and(|entries| {
            (lock(entries).iter()).any(|e| {
                e.saved.id == self.id
                    && e.saved.install == self.install
                    && e.saved.trusted
                    && e.saved.enabled
            })
        });
        if !live {
            return Err(ToolError::Outside(
                "this server is no longer trusted in tasks".into(),
            ));
        }
        self.tool.call(args, ctx)
    }

    fn prepare(
        &self,
        args: &serde_json::Value,
        ctx: &ToolCtx,
    ) -> Result<serde_json::Value, ToolError> {
        self.tool.prepare(args, ctx)
    }

    fn opens<'a>(&self, args: &'a serde_json::Value) -> Option<&'a str> {
        self.tool.opens(args)
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A connection found its sign-in gone or short: the card says so, and a 403 for a scope the token
/// lacks (`insufficient_scope`) keeps that scope for the next sign-in to ask for (step-up).
fn needs_auth(entries: &Mutex<Vec<Entry>>, id: &str, e: &McpError) {
    if let Some(entry) = lock(entries).iter_mut().find(|x| x.saved.id == id) {
        entry.state = State::NeedsAuth(e.to_string());
        if let McpError::NeedsAuth {
            challenge,
            scope_short: true,
        } = e
        {
            entry.step_up = challenge.scope.clone().or(entry.step_up.take());
        }
    }
}

/// The capability a workflow run holds to call a server's read tools: its id and install.
fn capability(s: &McpServer) -> String {
    format!("mcp.{}.{}", s.id, s.install)
}

/// A new install's identity: 128 random bits, so no two installs share one, even two that
/// `Hub::load` names within one tick of a coarse clock.
fn new_install() -> String {
    oauth::random_token(16)
}

/// The user set an Authorization header: their own token, so no sign-in is offered or used.
fn own_auth(t: &Transport) -> bool {
    matches!(t, Transport::Http { headers, .. } if headers.keys().any(|k| k.eq_ignore_ascii_case("authorization")))
}

impl Hub {
    /// Reads `<data>/mcp.json` and `<data>/mcp-auth.json`, and starts connecting every server that
    /// is on.
    pub fn load(data: &Path, emit: Emit) -> Hub {
        let file = data.join("mcp.json");
        let saved: Vec<McpServer> = read_saved(&file);
        let auth_path = data.join("mcp-auth.json");
        let auth: AuthFile = read_saved(&auth_path);
        let hub = Hub {
            entries: Arc::new(Mutex::new(Vec::new())),
            auth_file: Arc::new((auth_path, Mutex::new(auth))),
            file,
            logs: data.join("mcp-logs"),
            emit,
            version: Arc::new(AtomicU64::new(0)),
        };
        // Saved before installs had an identity: give each one now, so its grants name it.
        let named = saved.iter().any(|s| s.install.is_empty());
        for mut s in saved {
            let persisted = !s.install.is_empty();
            if !persisted {
                s.install = new_install();
            }
            let live = hub.server(&s);
            lock(&hub.entries).push(Entry {
                state: State::Off,
                saved: s,
                live,
                attempt: 0,
                paste: None,
                step_up: None,
                persisted,
            });
        }
        if named {
            // Best effort: unsaved, the next load names them again, so `grants` gives them none
            // until a save lands.
            let _ = hub.save(&mut lock(&hub.entries));
        }
        let on: Vec<String> = (lock(&hub.entries).iter())
            .filter(|e| e.saved.enabled)
            .map(|e| e.saved.id.clone())
            .collect();
        for id in on {
            hub.connect_later(&id);
        }
        hub
    }

    /// The version a chat opened at ([`Hub`]'s `version`).
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    fn bump(&self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    fn server(&self, s: &McpServer) -> Arc<Server> {
        let _ = fs::create_dir_all(&self.logs);
        let log = Some(self.logs.join(format!("{}.log", s.id)));
        let auth = match &s.transport {
            Transport::Http { .. } if !own_auth(&s.transport) => {
                let grant = lock(&self.auth_file.1).grants.get(&s.id).cloned();
                let (store, id) = (self.auth_file.clone(), s.id.clone());
                // Called by `Auth` as it commits a sign-in, a refresh or a sign-out; its error says
                // the file still holds the old sign-in.
                let save: oauth::SaveGrant = Box::new(move |g: Option<&Grant>| {
                    let mut f = lock(&store.1);
                    match g {
                        Some(g) => f.grants.insert(id.clone(), g.clone()),
                        None => f.grants.remove(&id),
                    };
                    let json = serde_json::to_vec_pretty(&*f).map_err(|e| e.to_string())?;
                    write_private(&store.0, &json)
                });
                Some(Arc::new(Auth::new(grant, save)))
            }
            _ => None,
        };
        let live = Server::with_auth(&s.id, s.transport.clone(), log, auth);
        // A call in a chat that finds the sign-in gone shows on the card. The entries are held
        // weakly: they own this server, so a strong hold would keep both alive past the hub.
        let (entries, emit, version, id) = (
            Arc::downgrade(&self.entries),
            self.emit.clone(),
            self.version.clone(),
            s.id.clone(),
        );
        live.set_on_needs_auth(Arc::new(move |e: &McpError| {
            let Some(entries) = entries.upgrade() else {
                return;
            };
            needs_auth(&entries, &id, e);
            version.fetch_add(1, Ordering::Relaxed);
            emit(Event::Mcp(McpChanged { id: id.clone() }));
        }));
        live
    }

    /// The servers, with every env and header value masked.
    pub fn list(&self) -> Vec<McpServerStatus> {
        lock(&self.entries).iter().map(status).collect()
    }

    /// Installs a server, untrusted and under a new install identity; on, it starts connecting.
    pub fn add(&self, mut server: McpServer) -> Result<(), String> {
        check(&server)?;
        // Only `set_trusted` trusts, and no grant saved for an earlier install reaches this one.
        server.trusted = false;
        server.install = new_install();
        {
            let mut entries = lock(&self.entries);
            if entries.iter().any(|e| e.saved.id == server.id) {
                return Err(format!(
                    "a server named '{}' is installed already",
                    server.id
                ));
            }
            if entries.len() >= MAX_SERVERS {
                return Err(format!(
                    "{MAX_SERVERS} servers are installed: remove one first"
                ));
            }
            let live = self.server(&server);
            entries.push(Entry {
                saved: server.clone(),
                live,
                state: State::Off,
                attempt: 0,
                paste: None,
                step_up: None,
                persisted: false,
            });
            // Unsaved, it is not installed: a retry would find it "installed already".
            if let Err(e) = self.save(&mut entries) {
                entries.pop();
                return Err(e);
            }
        }
        if server.enabled {
            self.connect_later(&server.id);
        }
        Ok(())
    }

    /// Uninstalls a server: its connection closes, its program stops, its sign-in is forgotten.
    pub fn remove(&self, id: &str) -> Result<(), String> {
        let live = {
            let mut entries = lock(&self.entries);
            let at = (entries.iter().position(|e| e.saved.id == id)).ok_or_else(|| unknown(id))?;
            let e = entries.remove(at);
            // Unsaved, it is still installed and running: it comes back at the next start.
            if let Err(err) = self.save(&mut entries) {
                entries.insert(at, e);
                return Err(err);
            }
            e.live
        };
        live.close();
        let forgotten = live.auth().map_or(Ok(()), |a| a.set(None));
        self.bump();
        forgotten.map_err(|e| {
            format!("the server is removed, but its saved sign-in could not be deleted ({e}): it stays in mcp-auth.json until the file can be written")
        })
    }

    /// Trusts a server's read-only labels for workflows, or stops: the next step takes it.
    pub fn set_trusted(&self, id: &str, on: bool) -> Result<(), String> {
        let mut entries = lock(&self.entries);
        let e = (entries.iter_mut().find(|e| e.saved.id == id)).ok_or_else(|| unknown(id))?;
        e.saved.trusted = on;
        self.save(&mut entries)?;
        drop(entries);
        (self.emit)(Event::Mcp(McpChanged { id: id.into() }));
        Ok(())
    }

    /// Turns a server on (it connects) or off (it closes; chats that hold its tools get an error).
    pub fn set_enabled(&self, id: &str, on: bool) -> Result<(), String> {
        let live = {
            let mut entries = lock(&self.entries);
            let at = (entries.iter().position(|e| e.saved.id == id)).ok_or_else(|| unknown(id))?;
            let was = std::mem::replace(&mut entries[at].saved.enabled, on);
            // Unsaved, the switch has not moved: an Off card over a running server would lie.
            if let Err(err) = self.save(&mut entries) {
                entries[at].saved.enabled = was;
                return Err(err);
            }
            let e = &mut entries[at];
            if !on {
                e.state = State::Off;
                e.attempt += 1;
                e.paste = None;
            }
            e.live.clone()
        };
        if on {
            self.connect_later(id);
        } else {
            live.close();
            self.bump();
        }
        Ok(())
    }

    /// Connects now (if it is not) and lists the tools.
    pub fn test(&self, id: &str) -> Result<Vec<McpToolInfo>, String> {
        let live = self.live(id)?;
        self.set_state(id, State::Connecting);
        let got = live.connect();
        // Testing a server that is off does not leave it running.
        let on = lock(&self.entries)
            .iter()
            .any(|e| e.saved.id == id && e.saved.enabled);
        match &got {
            Ok(_) if on => self.set_state(id, State::Connected),
            Ok(_) => {
                live.close();
                self.set_state(id, State::Off);
            }
            Err(e @ McpError::NeedsAuth { .. }) => {
                self.set_state(id, State::NeedsAuth(e.to_string()))
            }
            Err(e) => self.set_state(id, State::Error(e.to_string())),
        }
        self.bump();
        got.map(|tools| tools.iter().map(|t| info(id, t)).collect())
            .map_err(|e| e.to_string())
    }

    /// Starts a sign-in: finds the server's sign-in service, registers with it if needed, and
    /// returns the page to open in the browser (and its host, shown first). The rest runs on a
    /// thread: the browser's return (or a pasted final URL, `sign_in_finish`), the token, the
    /// connection, `mcp:status`.
    pub fn sign_in(&self, id: &str) -> Result<McpSignIn, String> {
        let (url, live, attempt, step_up) = {
            let mut entries = lock(&self.entries);
            let e = (entries.iter_mut().find(|e| e.saved.id == id)).ok_or_else(|| unknown(id))?;
            let Transport::Http { url, .. } = &e.saved.transport else {
                return Err("only a remote (URL) server signs in; a program on this computer signs in its own way".into());
            };
            if own_auth(&e.saved.transport) {
                return Err("this server uses the Authorization header you set; remove it to sign in instead".into());
            }
            e.attempt += 1;
            (url.clone(), e.live.clone(), e.attempt, e.step_up.clone())
        };
        let auth = live.auth().cloned().ok_or("this server has no sign-in")?;
        // A sign-out from here on wins: this sign-in commits only at this generation.
        let generation = auth.generation();
        let challenge = oauth::probe(&url);
        let (resource, server) = oauth::discover(&url, &challenge).map_err(|e| e.to_string())?;
        // A registered client is reused while its loopback port is free; else register again.
        let cached = lock(&self.auth_file.1).clients.get(&server.issuer).cloned();
        let port = cached
            .as_ref()
            .and_then(|c| url::Url::parse(&c.redirect_uri).ok()?.port());
        let callback =
            Callback::bind(port).map_err(|e| format!("cannot listen for the sign-in: {e}"))?;
        let reused = cached.filter(|c| c.redirect_uri == callback.redirect_uri());
        let client = match reused.clone() {
            Some(c) => c,
            None => {
                let c = oauth::register(&server, &callback.redirect_uri())
                    .map_err(|e| e.to_string())?;
                let mut f = lock(&self.auth_file.1);
                f.clients.insert(server.issuer.clone(), c.clone());
                if let Ok(json) = serde_json::to_vec_pretty(&*f) {
                    write_private(&self.auth_file.0, &json)?;
                }
                c
            }
        };
        // The scope a call was refused for (step-up), else the server's challenge, else its metadata.
        let scope = (step_up.or_else(|| challenge.scope.clone()))
            .or_else(|| (!resource.scopes.is_empty()).then(|| resource.scopes.join(" ")));
        let verifier = oauth::random_token(32);
        let state = oauth::random_token(16);
        let page = oauth::authorize_url(
            &server,
            &client,
            &verifier,
            &state,
            scope.as_deref(),
            &resource.resource,
        )
        .map_err(|e| e.to_string())?;
        let host = url::Url::parse(&page)
            .ok()
            .and_then(|u| u.host_str().map(str::to_owned))
            .unwrap_or_default();
        let (tx, pasted) = mpsc::channel();
        if let Some(e) = lock(&self.entries).iter_mut().find(|e| e.saved.id == id) {
            e.paste = Some(tx);
            e.state = State::NeedsAuth(format!("finish signing in at {host} in your browser"));
        }
        (self.emit)(Event::Mcp(McpChanged { id: id.into() }));
        let (entries, emit, version, id, auth_file) = (
            self.entries.clone(),
            self.emit.clone(),
            self.version.clone(),
            id.to_owned(),
            self.auth_file.clone(),
        );
        std::thread::spawn(move || {
            let current = || {
                lock(&entries)
                    .iter()
                    .any(|e| e.saved.id == id && e.attempt == attempt)
            };
            let done =
                (callback.wait(&state, SIGN_IN_WAIT, &pasted, &|| !current())).and_then(|code| {
                    oauth::exchange(&server, &client, &code, &verifier, &resource.resource)
                });
            if !current() {
                return; // a newer sign-in, a removal or a turn-off took over
            }
            let grant = done.and_then(|g| {
                Grant {
                    scope: scope.clone(),
                    ..g
                }
                .for_endpoint(&url)
            });
            let state = match grant {
                Ok(grant) => {
                    // Kept only if no sign-out (or newer sign-in) came since this one began, nor a
                    // cancel since the check above; a save that fails still leaves it signed in
                    // for this session.
                    let kept = commit_sign_in(&entries, &id, attempt, || {
                        !matches!(auth.set_if(generation, Some(grant)), Ok(false))
                    });
                    if !kept {
                        return;
                    }
                    live.reconnect();
                    match live.connect() {
                        Ok(_) => State::Connected,
                        Err(e @ McpError::NeedsAuth { .. }) => State::NeedsAuth(e.to_string()),
                        Err(e) => State::Error(e.to_string()),
                    }
                }
                Err(e) => {
                    // A reused registration the sign-in service may have forgotten (a restart,
                    // an expired secret): dropped, so the next sign-in registers again rather
                    // than fail the same way until mcp-auth.json is edited by hand.
                    if let Some(c) = &reused {
                        let mut f = lock(&auth_file.1);
                        if (f.clients.get(&server.issuer))
                            .is_some_and(|x| x.client_id == c.client_id)
                        {
                            f.clients.remove(&server.issuer);
                            // Unsaved, it is gone for this session all the same, and the next
                            // sign-in's registration writes the file again.
                            if let Ok(json) = serde_json::to_vec_pretty(&*f) {
                                let _ = write_private(&auth_file.0, &json);
                            }
                        }
                    }
                    State::NeedsAuth(e.to_string())
                }
            };
            if let Some(e) = lock(&entries)
                .iter_mut()
                .find(|e| e.saved.id == id && e.attempt == attempt)
            {
                if state == State::Connected {
                    e.step_up = None;
                }
                e.state = state;
                e.paste = None;
            }
            version.fetch_add(1, Ordering::Relaxed);
            emit(Event::Mcp(McpChanged { id }));
        });
        Ok(McpSignIn { url: page, host })
    }

    /// Finishes a waiting sign-in with the address the browser ended on (when its return could
    /// not reach this computer).
    pub fn sign_in_finish(&self, id: &str, url: &str) -> Result<(), String> {
        // An address that does not parse is refused here, so the sign-in still waits for the
        // right one: in the sign-in it ended the whole wait.
        if url::Url::parse(url.trim()).is_err() {
            return Err(
                "that is not the address the browser ended on: copy all of it, from https://"
                    .into(),
            );
        }
        let entries = lock(&self.entries);
        let e = (entries.iter().find(|e| e.saved.id == id)).ok_or_else(|| unknown(id))?;
        let tx = e
            .paste
            .as_ref()
            .ok_or("no sign-in is waiting for this server: press Sign in first")?;
        tx.send(url.to_owned())
            .map_err(|_| "the sign-in already ended".to_owned())
    }

    /// Forgets the server's sign-in; its next call needs a new one. A sign-in or a refresh under
    /// way is dropped. `Err` when the saved sign-in could not be deleted: it is gone for this
    /// session, but would come back at the next start.
    pub fn sign_out(&self, id: &str) -> Result<(), String> {
        let live = self.live(id)?;
        // The attempt first: a sign-in waiting for the browser stops before it commits.
        if let Some(e) = lock(&self.entries).iter_mut().find(|e| e.saved.id == id) {
            e.attempt += 1;
            e.paste = None;
            e.step_up = None;
            e.state = State::NeedsAuth("signed out".into());
        }
        let forgotten = live.auth().map_or(Ok(()), |a| a.set(None));
        live.reconnect();
        self.bump();
        (self.emit)(Event::Mcp(McpChanged { id: id.into() }));
        forgotten.map_err(|e| {
            format!("signed out for now, but the saved sign-in could not be deleted ({e}): it comes back when Larch restarts, until the file can be written")
        })
    }

    /// Stops a sign-in still waiting for the browser. Unlike [`Hub::sign_out`] the sign-in the
    /// server has stays (a scope step-up that is given up keeps the token that works), and the
    /// card shows the connection as it is now.
    pub fn cancel_sign_in(&self, id: &str) -> Result<(), String> {
        let mut entries = lock(&self.entries);
        let e = (entries.iter_mut().find(|e| e.saved.id == id)).ok_or_else(|| unknown(id))?;
        e.attempt += 1;
        e.paste = None;
        drop(entries);
        self.connect_later(id);
        (self.emit)(Event::Mcp(McpChanged { id: id.into() }));
        Ok(())
    }

    /// The tools a new chat takes: those of every server that is on and connected, but those no
    /// chat can take (`larch_agent::left_out`), which the server's card names instead: every
    /// chat began on a notice of Notion's three.
    pub fn chat_tools(&self) -> Vec<Box<dyn Tool>> {
        let entries = lock(&self.entries);
        (entries.iter())
            .filter(|e| e.saved.enabled && e.state == State::Connected)
            .flat_map(|e| e.live.chat_tools())
            .filter(|t| larch_agent::left_out(&t.spec()).is_none())
            .map(|t| Box::new(t) as Box<dyn Tool>)
            .collect()
    }

    /// The grants a workflow saved now takes: one per server that is on and trusted
    /// (`capability`), so trusting a server later never widens a workflow saved before. None for
    /// an install not yet saved, which the next start would replace.
    pub fn grants(&self) -> Vec<String> {
        (lock(&self.entries).iter())
            .filter(|e| e.saved.enabled && e.saved.trusted && e.persisted)
            .map(|e| capability(&e.saved))
            .collect()
    }

    /// Model steps' tools: each step takes those of every trusted server on and connected as it
    /// starts, and calls the read-only ones its run's grants hold. A server labels its own tools
    /// read-only, so an untrusted one's never run unattended.
    pub fn step_tools(&self) -> StepToolSource {
        let entries = self.entries.clone();
        Arc::new(move || {
            (lock(&entries).iter())
                .filter(|e| e.saved.trusted && e.saved.enabled && e.state == State::Connected)
                .flat_map(|e| {
                    let capability = capability(&e.saved);
                    let (entries, id, install) = (
                        Arc::downgrade(&entries),
                        e.saved.id.clone(),
                        e.saved.install.clone(),
                    );
                    (e.live.chat_tools().into_iter()).map(move |t| StepTool {
                        tool: Arc::new(Trusted {
                            tool: Box::new(t),
                            entries: entries.clone(),
                            id: id.clone(),
                            install: install.clone(),
                        }),
                        capability: capability.clone(),
                    })
                })
                .collect()
        })
    }

    fn live(&self, id: &str) -> Result<Arc<Server>, String> {
        (lock(&self.entries).iter().find(|e| e.saved.id == id))
            .map(|e| e.live.clone())
            .ok_or_else(|| unknown(id))
    }

    fn set_state(&self, id: &str, state: State) {
        if let Some(e) = lock(&self.entries).iter_mut().find(|e| e.saved.id == id) {
            e.state = state;
        }
        (self.emit)(Event::Mcp(McpChanged { id: id.into() }));
    }

    /// Connects on a thread of its own: a program may take a while to start (uvx downloads it
    /// the first time), and the UI waits on nothing.
    fn connect_later(&self, id: &str) {
        let Ok(live) = self.live(id) else { return };
        self.set_state(id, State::Connecting);
        let (entries, emit, version, id) = (
            self.entries.clone(),
            self.emit.clone(),
            self.version.clone(),
            id.to_owned(),
        );
        std::thread::spawn(move || {
            let got = live.connect();
            let mut list = lock(&entries);
            match list.iter_mut().find(|e| e.saved.id == id) {
                // Turned off or removed while it connected: the connection goes.
                Some(e) if !e.saved.enabled => {
                    e.state = State::Off;
                    drop(list);
                    live.close();
                }
                Some(e) => {
                    e.state = match got {
                        Ok(_) => State::Connected,
                        Err(err @ McpError::NeedsAuth { .. }) => State::NeedsAuth(err.to_string()),
                        Err(err) => State::Error(err.to_string()),
                    };
                    drop(list);
                }
                None => {
                    drop(list);
                    live.close();
                }
            }
            version.fetch_add(1, Ordering::Relaxed);
            emit(Event::Mcp(McpChanged { id }));
        });
    }

    fn save(&self, entries: &mut [Entry]) -> Result<(), String> {
        let saved: Vec<&McpServer> = entries.iter().map(|e| &e.saved).collect();
        let json = serde_json::to_vec_pretty(&saved).map_err(|e| e.to_string())?;
        write_private(&self.file, &json)?;
        entries.iter_mut().for_each(|e| e.persisted = true);
        Ok(())
    }
}

/// Runs a finished sign-in's `commit` only while `attempt` is still the entry's, checked and
/// committed under the lock a cancel, removal or turn-off takes to move the attempt: one that came
/// after the sign-in's last check would otherwise still keep its grant.
fn commit_sign_in(
    entries: &Mutex<Vec<Entry>>,
    id: &str,
    attempt: u64,
    commit: impl FnOnce() -> bool,
) -> bool {
    let held = lock(entries);
    held.iter()
        .any(|e| e.saved.id == id && e.attempt == attempt)
        && commit()
}

/// A page of the MCP Registry's servers matching `query`: the latest version of each.
pub fn search(query: &str, cursor: Option<&str>) -> Result<McpSearch, String> {
    let mut req = ureq::get(REGISTRY)
        .timeout(Duration::from_secs(20))
        .query("limit", "30");
    if !query.trim().is_empty() {
        req = req.query("search", query.trim());
    }
    if let Some(c) = cursor {
        req = req.query("cursor", c);
    }
    let text = (req.call())
        .map_err(|e| format!("the MCP Registry did not answer: {e}"))?
        .into_string()
        .map_err(|e| format!("the MCP Registry's answer could not be read: {e}"))?;
    let body: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("the MCP Registry's answer is not JSON: {e}"))?;
    let servers = (body["servers"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter())
    .filter(|e| {
        let latest = &e["_meta"]["io.modelcontextprotocol.registry/official"]["isLatest"];
        latest.is_null() || *latest == serde_json::Value::Bool(true)
    })
    .collect();
    Ok(McpSearch {
        servers,
        next_cursor: body["metadata"]["nextCursor"].as_str().map(str::to_owned),
    })
}

fn unknown(id: &str) -> String {
    format!("no MCP server named '{id}' is installed")
}

/// Refuses a server the hub could not run: an id the tools cannot be named after, an empty
/// command, a URL that is not http(s).
fn check(s: &McpServer) -> Result<(), String> {
    let id_ok = !s.id.is_empty()
        && s.id.len() <= 32
        && (s.id.chars()).all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if !id_ok {
        return Err(format!("'{}' is not a server id: use a-z, 0-9 and -", s.id));
    }
    match &s.transport {
        Transport::Stdio { command, .. } if command.trim().is_empty() => {
            Err("the server has no command to start it".into())
        }
        Transport::Http { url, .. }
            if !(url.starts_with("https://") || url.starts_with("http://")) =>
        {
            Err(format!("'{url}' is not an http(s) URL"))
        }
        _ => Ok(()),
    }
}

fn info(server: &str, t: &RemoteTool) -> McpToolInfo {
    McpToolInfo {
        name: t.name.clone(),
        description: t.description.clone(),
        read_only: t.read_only,
        left_out: larch_agent::left_out(&client::spec(server, t)).map(|why| why.to_string()),
    }
}

fn status(e: &Entry) -> McpServerStatus {
    let mut transport = e.saved.transport.clone();
    match &mut transport {
        Transport::Stdio {
            env,
            args,
            secret_args,
            ..
        } => {
            env.values_mut().for_each(|v| *v = MASK.into());
            for &at in secret_args.iter() {
                if let Some(a) = args.get_mut(at) {
                    *a = MASK.into();
                }
            }
        }
        Transport::Http { headers, .. } => headers.values_mut().for_each(|v| *v = MASK.into()),
    }
    let (state, error) = match &e.state {
        _ if !e.saved.enabled => ("off", None),
        State::Off => ("off", None),
        State::Connecting => ("connecting", None),
        State::Connected => ("connected", None),
        State::NeedsAuth(m) => ("needs-sign-in", Some(m.clone())),
        State::Error(m) => ("error", Some(m.clone())),
    };
    let (signed_in, can_sign_in) = match (&e.saved.transport, e.live.auth()) {
        (Transport::Http { url, .. }, Some(a)) => (a.signed_in_for(url), true),
        _ => (false, false),
    };
    McpServerStatus {
        id: e.saved.id.clone(),
        name: e.saved.name.clone(),
        source: e.saved.source.clone(),
        transport,
        enabled: e.saved.enabled,
        trusted: e.saved.trusted,
        status: state.into(),
        tools: match state {
            "connected" => (e.live.tools().iter())
                .map(|t| info(&e.saved.id, t))
                .collect(),
            _ => Vec::new(),
        },
        error,
        signed_in,
        can_sign_in,
        signing: e.paste.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// A fresh folder under the system's temp folder, as the crate's other tests make them.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("larch-mcp-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn hub(dir: &Path) -> (Hub, Arc<Mutex<Vec<String>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        let emit: Emit = Arc::new(move |e: Event| s.lock().unwrap().push(e.name()));
        (Hub::load(dir, emit), seen)
    }

    fn stdio(id: &str, command: &str, enabled: bool) -> McpServer {
        McpServer {
            id: id.into(),
            name: id.into(),
            source: "custom".into(),
            transport: Transport::Stdio {
                command: command.into(),
                args: vec![],
                env: BTreeMap::from([("API_KEY".to_string(), "secret".to_string())]),
                secret_args: Vec::new(),
            },
            enabled,
            trusted: false,
            install: String::new(),
        }
    }

    #[test]
    fn an_unreadable_mcp_json_is_kept_aside_not_saved_over() {
        let dir = scratch("unreadable");
        let theirs = r#"[{"id":"a","transport":{"type":"sse"}}]"#;
        fs::write(dir.join("mcp.json"), theirs).unwrap();
        let (h, _) = hub(&dir);
        h.add(stdio("b", "x", false)).unwrap();
        assert_eq!(
            fs::read_to_string(dir.join("mcp.json.bad")).unwrap(),
            theirs
        );
    }

    #[test]
    fn a_switch_that_could_not_be_saved_does_not_move() {
        let dir = scratch("unsaved");
        let (h, _) = hub(&dir);
        h.add(stdio("a", "/nonexistent/server", true)).unwrap();
        fs::remove_file(dir.join("mcp.json")).unwrap();
        fs::create_dir(dir.join("mcp.json")).unwrap();
        assert!(h.set_enabled("a", false).is_err());
        assert!(h.list()[0].enabled);
        assert!(h.remove("a").is_err());
        assert_eq!(h.list().len(), 1);
    }

    #[test]
    fn servers_persist_masked_and_owner_only() {
        let dir = scratch("persist");
        let (h, _) = hub(&dir);
        h.add(stdio("notes", "/nonexistent/server", false)).unwrap();
        assert!(
            h.add(stdio("notes", "x", false))
                .unwrap_err()
                .contains("installed already")
        );
        assert!(
            h.add(stdio("Bad Id", "x", false))
                .unwrap_err()
                .contains("not a server id")
        );
        let listed = h.list();
        assert_eq!(listed[0].status, "off");
        assert!(!listed[0].can_sign_in);
        match &listed[0].transport {
            Transport::Stdio { env, .. } => assert_eq!(env["API_KEY"], MASK),
            t => panic!("{t:?}"),
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(dir.join("mcp.json"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let (again, _) = hub(&dir);
        assert_eq!(again.list().len(), 1);
        assert!(
            fs::read_to_string(dir.join("mcp.json"))
                .unwrap()
                .contains("secret")
        );
        again.remove("notes").unwrap();
        assert!(again.list().is_empty());
        assert!(again.remove("notes").unwrap_err().contains("no MCP server"));
    }

    /// A stdio server with one read-only tool, `search`, answering "found".
    fn notes_server(dir: &Path) -> McpServer {
        let script = dir.join("notes.py");
        fs::write(
            &script,
            r#"import json, sys
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m:
        continue
    r = {}
    if m["method"] == "initialize":
        r = {"protocolVersion": m["params"]["protocolVersion"], "capabilities": {"tools": {}}, "serverInfo": {"name": "notes", "version": "1"}}
    elif m["method"] == "tools/list":
        r = {"tools": [{"name": "search", "description": "Search the notes.", "inputSchema": {"type": "object", "properties": {}}, "annotations": {"readOnlyHint": True, "openWorldHint": False}}]}
    elif m["method"] == "tools/call":
        r = {"content": [{"type": "text", "text": "found"}]}
    print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}), flush=True)
"#,
        )
        .unwrap();
        // Windows' Python has no python3 outside the Store's aliases.
        let python = if cfg!(windows) { "python" } else { "python3" };
        let mut s = stdio("notes", python, true);
        if let Transport::Stdio { args, .. } = &mut s.transport {
            args.push(script.display().to_string());
        }
        s
    }

    fn connected(h: &Hub, id: &str) {
        for _ in 0..500 {
            if h.list()
                .iter()
                .any(|s| s.id == id && s.status == "connected")
            {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("{:?}", h.list());
    }

    /// A workflow is granted a server only while it is on and trusted, under that install's own
    /// identity; its steps call the server's read tools only while it stays on and trusted.
    #[test]
    fn a_workflow_reaches_a_server_only_while_it_is_on_and_trusted() {
        let dir = scratch("grants");
        let (h, _) = hub(&dir);
        let mut asked_trust = notes_server(&dir);
        // A caller cannot install a server trusted: only `set_trusted` trusts.
        asked_trust.trusted = true;
        h.add(asked_trust).unwrap();
        h.add(stdio("mail", "/nonexistent/server", false)).unwrap();
        connected(&h, "notes");
        assert!(h.list().iter().all(|s| !s.trusted));
        assert!(h.grants().is_empty());
        assert!((h.step_tools())().is_empty());

        h.set_trusted("notes", true).unwrap();
        let [grant] = &h.grants()[..] else {
            panic!("{:?}", h.grants())
        };
        assert!(grant.starts_with("mcp.notes."), "{grant}");
        let tools = (h.step_tools())();
        let [step] = &tools[..] else {
            panic!("{} tools", tools.len())
        };
        assert_eq!(&step.capability, grant);
        let ctx = ToolCtx {
            allowed: Vec::new(),
            data_dir: dir.clone(),
            source: "workflow".into(),
        };
        assert!(
            step.tool
                .call(&serde_json::json!({}), &ctx)
                .unwrap()
                .contains("found")
        );

        // Untrusted, a step that holds the tool can no longer call it, and the next gets none.
        h.set_trusted("notes", false).unwrap();
        assert!(step.tool.call(&serde_json::json!({}), &ctx).is_err());
        assert!((h.step_tools())().is_empty());
        h.set_trusted("notes", true).unwrap();
        h.set_enabled("notes", false).unwrap();
        assert!((h.step_tools())().is_empty() && h.grants().is_empty());

        // Trust is saved with the server, and the install keeps its identity.
        drop(h);
        let (h, _) = hub(&dir);
        assert!(h.list().iter().find(|s| s.id == "notes").unwrap().trusted);
        h.set_enabled("notes", true).unwrap();
        assert_eq!(h.grants(), std::slice::from_ref(grant));

        // Removed and installed again under the same id, it is another install: an earlier
        // workflow's grant does not reach it.
        h.remove("notes").unwrap();
        h.add(notes_server(&dir)).unwrap();
        h.set_trusted("notes", true).unwrap();
        let again = h.grants();
        assert!(again.len() == 1 && &again[0] != grant, "{again:?}");
    }

    /// A step tool runs as the server's tool itself, its argument fixes (`prepare`) included:
    /// the step's effect prepares through the wrapper.
    #[test]
    fn a_trusted_tool_prepares_arguments_as_its_tool_does() {
        struct Fixes;
        impl Tool for Fixes {
            fn spec(&self) -> ToolSpec {
                unreachable!("only prepare is asked")
            }
            fn call(&self, _: &serde_json::Value, _: &ToolCtx) -> Result<String, ToolError> {
                unreachable!("only prepare is asked")
            }
            fn prepare(
                &self,
                _: &serde_json::Value,
                _: &ToolCtx,
            ) -> Result<serde_json::Value, ToolError> {
                Ok(serde_json::json!({"repo": "owner/repo"}))
            }
        }
        let trusted = Trusted {
            tool: Box::new(Fixes),
            entries: std::sync::Weak::new(),
            id: "notes".into(),
            install: "one".into(),
        };
        let ctx = ToolCtx {
            allowed: Vec::new(),
            data_dir: scratch("prepare"),
            source: "workflow".into(),
        };
        let args = serde_json::json!({"repo": "repo"});
        assert_eq!(
            trusted.prepare(&args, &ctx).unwrap(),
            Fixes.prepare(&args, &ctx).unwrap()
        );
    }

    /// Old saves named at one load get installs of their own, and two servers saved under one
    /// install are still told apart: untrusting one stops its step's tool, not the other's.
    #[test]
    fn servers_saved_under_one_install_are_told_apart() {
        let dir = scratch("one-install");
        let mut notes = notes_server(&dir);
        notes.trusted = true;
        notes.install = "same".into();
        let other = McpServer {
            id: "other".into(),
            ..notes.clone()
        };
        let old = [
            stdio("old-a", "/nonexistent/server", false),
            stdio("old-b", "/nonexistent/server", false),
        ];
        let saved = [notes, other, old[0].clone(), old[1].clone()];
        fs::write(dir.join("mcp.json"), serde_json::to_vec(&saved).unwrap()).unwrap();
        let (h, _) = hub(&dir);
        let installs: Vec<String> = (lock(&h.entries).iter())
            .map(|e| e.saved.install.clone())
            .collect();
        assert!(
            !installs[2].is_empty() && installs[2] != installs[3],
            "{installs:?}"
        );
        connected(&h, "notes");
        connected(&h, "other");
        let tools = (h.step_tools())();
        let held = |id: &str| {
            let at = (tools.iter()).position(|t| t.capability == format!("mcp.{id}.same"));
            tools[at.unwrap_or_else(|| panic!("no {id} tool"))]
                .tool
                .clone()
        };
        let (notes, other) = (held("notes"), held("other"));
        let ctx = ToolCtx {
            allowed: Vec::new(),
            data_dir: dir.clone(),
            source: "workflow".into(),
        };
        h.set_trusted("notes", false).unwrap();
        assert!(notes.call(&serde_json::json!({}), &ctx).is_err());
        assert!(
            (other.call(&serde_json::json!({}), &ctx).unwrap()).contains("found"),
            "the other server stays trusted"
        );
    }

    /// An install named at load whose save failed gets no grant: the next start names it again,
    /// so a grant for this one would reach nothing. Once a save lands, it does.
    #[test]
    fn an_install_that_could_not_be_saved_gets_no_grant() {
        let dir = scratch("unsaved-install");
        let mut mail = stdio("mail", "/nonexistent/server", true);
        mail.trusted = true;
        fs::write(dir.join("mcp.json"), serde_json::to_vec(&[mail]).unwrap()).unwrap();
        // mcp.json cannot be replaced: a folder stands where its temporary file goes.
        fs::create_dir(dir.join("mcp.json.tmp")).unwrap();
        let (h, _) = hub(&dir);
        assert!(h.grants().is_empty(), "{:?}", h.grants());
        fs::remove_dir(dir.join("mcp.json.tmp")).unwrap();
        h.set_trusted("mail", true).unwrap();
        assert_eq!(h.grants().len(), 1);
    }

    #[test]
    fn a_server_that_cannot_start_shows_its_error_and_gives_no_tools() {
        let dir = scratch("broken");
        let (h, seen) = hub(&dir);
        h.add(stdio("broken", "/nonexistent/server", false))
            .unwrap();
        let v0 = h.version();
        h.set_enabled("broken", true).unwrap();
        for _ in 0..100 {
            if h.list()[0].status == "error" {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let s = &h.list()[0];
        assert_eq!(s.status, "error");
        assert!(s.error.as_deref().unwrap().contains("would not start"));
        assert!(h.chat_tools().is_empty());
        assert!(h.test("broken").is_err());
        assert!(h.version() > v0);
        assert!(seen.lock().unwrap().iter().all(|n| n == "mcp:status"));
        assert!(h.sign_in("broken").unwrap_err().contains("only a remote"));
        h.set_enabled("broken", false).unwrap();
        assert_eq!(h.list()[0].status, "off");
    }

    /// A pasted text that is no address is refused before it reaches a sign-in, which it ended; a
    /// card says whether a sign-in waits, so its panel closes once none does.
    #[test]
    fn a_paste_that_is_no_address_is_refused_and_no_sign_in_waits() {
        let dir = scratch("paste");
        let (h, _) = hub(&dir);
        h.add(remote("notes")).unwrap();
        assert!(!h.list()[0].signing);
        let e = h
            .sign_in_finish("notes", "app.example/cb?code=1")
            .unwrap_err();
        assert!(e.contains("copy all of it"), "{e}");
        let e = h
            .sign_in_finish("notes", "https://app.example/cb?code=1")
            .unwrap_err();
        assert!(e.contains("press Sign in first"), "{e}");
    }

    #[test]
    fn a_server_with_its_own_authorization_header_never_signs_in() {
        let dir = scratch("ownauth");
        let (h, _) = hub(&dir);
        let s = McpServer {
            id: "keyed".into(),
            name: "keyed".into(),
            source: "custom".into(),
            transport: Transport::Http {
                url: "http://127.0.0.1:9/mcp".into(),
                headers: BTreeMap::from([("authorization".to_string(), "Bearer mine".to_string())]),
            },
            enabled: false,
            trusted: false,
            install: String::new(),
        };
        h.add(s).unwrap();
        assert!(!h.list()[0].can_sign_in);
        assert!(
            h.sign_in("keyed")
                .unwrap_err()
                .contains("Authorization header")
        );
    }

    fn remote(id: &str) -> McpServer {
        McpServer {
            id: id.into(),
            name: id.into(),
            source: "custom".into(),
            transport: Transport::Http {
                url: "http://127.0.0.1:9/mcp".into(),
                headers: BTreeMap::new(),
            },
            enabled: false,
            trusted: false,
            install: String::new(),
        }
    }

    fn signed_in(h: &Hub, id: &str) {
        let grant = Grant {
            client: Registered {
                client_id: "c".into(),
                client_secret: None,
                auth_method: "none".into(),
                redirect_uri: "http://127.0.0.1:1/cb".into(),
            },
            token_endpoint: "http://127.0.0.1:9/token".into(),
            resource: "http://127.0.0.1:9/mcp".into(),
            access_token: "AT".into(),
            refresh_token: None,
            expires_at: None,
            scope: None,
            endpoint: None,
        };
        lock(&h.auth_file.1).grants.insert(id.into(), grant);
    }

    /// A sign-out whose saved sign-in cannot be deleted says so, instead of returning as if done:
    /// the next start would read it back.
    #[test]
    fn a_sign_out_that_cannot_delete_the_saved_sign_in_says_so() {
        let dir = scratch("signout-fail");
        let (h, _) = hub(&dir);
        signed_in(&h, "svc");
        h.add(remote("svc")).unwrap();
        // mcp-auth.json cannot be replaced: a folder stands in its place.
        let _ = fs::remove_file(dir.join("mcp-auth.json"));
        fs::create_dir_all(dir.join("mcp-auth.json")).unwrap();
        let e = h.sign_out("svc").unwrap_err();
        assert!(e.contains("could not be deleted"), "{e}");
        // Gone for this session all the same.
        let live = h.live("svc").unwrap();
        assert!(!live.auth().unwrap().signed_in_for("http://127.0.0.1:9/mcp"));
        // Removing it says so too.
        assert!(
            h.remove("svc")
                .unwrap_err()
                .contains("could not be deleted")
        );
    }

    /// The hub's servers do not keep the hub's entries alive: dropping the hub drops them, so each
    /// connection's teardown runs (its program stops).
    #[test]
    fn dropping_the_hub_drops_its_servers() {
        let dir = scratch("drop");
        let (h, _) = hub(&dir);
        h.add(remote("svc")).unwrap();
        h.add(stdio("local", "true", false)).unwrap();
        let weak: Vec<_> = lock(&h.entries)
            .iter()
            .map(|e| Arc::downgrade(&e.live))
            .collect();
        drop(h);
        assert!(weak.iter().all(|w| w.upgrade().is_none()));
    }

    /// A call refused for a scope the token lacks (403 insufficient_scope) keeps that scope for
    /// the next sign-in, which asks for it.
    #[test]
    fn a_403_for_a_missing_scope_is_asked_for_at_the_next_sign_in() {
        let dir = scratch("stepup");
        let (h, _) = hub(&dir);
        h.add(remote("svc")).unwrap();
        needs_auth(
            &h.entries,
            "svc",
            &McpError::NeedsAuth {
                challenge: oauth::Challenge {
                    scope: Some("mcp admin".into()),
                    ..oauth::Challenge::default()
                },
                scope_short: true,
            },
        );
        assert_eq!(lock(&h.entries)[0].step_up.as_deref(), Some("mcp admin"));
    }

    /// A secret package argument is masked in the server's status, as env values are.
    #[test]
    fn a_secret_argument_is_masked_in_the_status() {
        let dir = scratch("secret-arg");
        let (h, _) = hub(&dir);
        let mut s = stdio("keyed", "npx", false);
        if let Transport::Stdio {
            args, secret_args, ..
        } = &mut s.transport
        {
            *args = vec!["pkg".into(), "--api-key".into(), "sk-synthetic".into()];
            *secret_args = vec![2];
        }
        h.add(s).unwrap();
        let Transport::Stdio { args, .. } = &h.list()[0].transport else {
            panic!("stdio");
        };
        assert_eq!(args, &["pkg", "--api-key", MASK]);
    }

    #[test]
    fn a_cancelled_sign_in_never_commits_after_the_cancel_returns() {
        let dir = scratch("cancel-commit");
        let (h, _) = hub(&dir);
        h.add(remote("r")).unwrap();
        let attempt = |h: &Hub| {
            let entries = lock(&h.entries);
            entries.iter().find(|e| e.saved.id == "r").unwrap().attempt
        };
        // Cancelled after the sign-in's last check of its attempt: the grant is never set.
        let began = attempt(&h);
        h.cancel_sign_in("r").unwrap();
        assert!(!commit_sign_in(&h.entries, "r", began, || unreachable!()));
        // A cancel while the grant commits waits for it: nothing commits after it returns.
        let began = attempt(&h);
        let (inside, committing) = mpsc::channel();
        let (go, wait) = mpsc::channel::<()>();
        let entries = &h.entries;
        std::thread::scope(|s| {
            let commit = s.spawn(move || {
                commit_sign_in(entries, "r", began, || {
                    inside.send(()).unwrap();
                    wait.recv().unwrap();
                    true
                })
            });
            committing.recv().unwrap();
            let cancel = s.spawn(|| h.cancel_sign_in("r"));
            std::thread::sleep(Duration::from_millis(100));
            let waited = !cancel.is_finished();
            // The commit ends before anything is asserted, so a failure fails rather than hangs.
            go.send(()).unwrap();
            assert!(commit.join().unwrap());
            cancel.join().unwrap().unwrap();
            assert!(waited, "the cancel waits for the commit under way");
        });
        assert!(!commit_sign_in(&h.entries, "r", began, || unreachable!()));
    }
}
