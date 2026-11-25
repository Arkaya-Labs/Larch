//! Iris as an MCP server (revision 2025-11-25): a client such as Claude Code hands it work that runs
//! on this computer's GPU, through the entries of a catalogue the caller passes. Iris's rules hold
//! on every call, whatever the client's model asks: a Read entry runs; a WriteLocal entry runs only
//! inside the write folder the server was started with; WriteExternal and Spend entries are never
//! offered, since Iris asks its own user for those. JSON-RPC over stdio (`Server::stdio`), or
//! Streamable HTTP on 127.0.0.1 behind a bearer token (`Server::http`). Long calls report progress
//! and may run as MCP tasks.

pub mod client;
pub(crate) mod describe;
mod http;
pub mod oauth;
mod repo;

use std::{
    collections::HashMap,
    fs,
    io::{self, BufRead, Write},
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use serde_json::{Value, json};

pub use self::http::token;
use crate::{
    audit::{Audit, AuditEvent, EventKind},
    parse::{ToolCall, validate},
    tool::{self, ARGS_TOKENS, Permission, ToolCtx, ToolSpec},
};

/// The revision this server speaks, then the older ones whose clients it answers in theirs: their
/// tool calls have the same shape.
pub(crate) const VERSIONS: [&str; 3] = ["2025-11-25", "2025-06-18", "2025-03-26"];

/// A task's longest life in milliseconds, and the life of one whose client names none.
const TASK_TTL_MS: u64 = 3_600_000;

/// How often a client should poll a task, in milliseconds.
const POLL_MS: u64 = 1000;

const INSTRUCTIONS: &str = "Iris runs these tools on this computer's GPU, over the user's files. \
    Calls run side by side, so hand it many reads at once. Changes outside this computer and \
    spending are not offered here: Iris asks its own user for those.";

/// What an entry returns: text for the model, and optional structured content.
pub struct Output {
    /// The text the client's model reads.
    pub text: String,
    /// JSON for the client itself (`structuredContent`).
    pub structured: Option<Value>,
}

/// Runs an entry on arguments that passed its schema; an error is the message the model gets.
pub type Run = Box<dyn Fn(&Value, &Ctx) -> Result<Output, String> + Send + Sync>;

/// One tool in the catalogue the caller passes.
pub struct Entry {
    /// The name the client calls.
    pub name: String,
    /// What the client's model reads to choose it.
    pub description: String,
    /// JSON Schema of the arguments object. A WriteLocal entry's names a required string `path`,
    /// the only place it writes: the server makes it a real path inside the write folder.
    pub schema: Value,
    /// The risk class, which decides whether the entry is offered at all.
    pub permission: Permission,
    /// The work.
    pub run: Run,
}

/// How the server runs.
#[derive(Debug)]
pub struct Settings {
    /// The one folder WriteLocal entries may write in; without it they are neither listed nor run.
    pub write_folder: Option<PathBuf>,
    /// Results are cut at four bytes per token of this (`tool::truncate`), with the cut stated.
    pub max_result_tokens: u32,
    /// Calls that run at once; the rest wait in order.
    pub workers: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            write_folder: None,
            // 32 KB of text.
            max_result_tokens: 8192,
            workers: 4,
        }
    }
}

/// Where a message to the client goes.
type Out = Arc<dyn Fn(Value) + Send + Sync>;

/// A running call's side of its request: whether it was cancelled, and its progress.
pub struct Ctx {
    stop: Arc<AtomicBool>,
    progress: Option<(Value, Out)>,
    task: Option<String>,
    last: Mutex<f64>,
}

impl Ctx {
    /// Whether the client cancelled the call; the work should stop where it can.
    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Tells the client how far the call is, when it asked to know. A value that does not rise
    /// above the last one, or comes after a cancel, is not sent: the protocol forbids both.
    pub fn progress(&self, done: f64, total: Option<f64>, message: &str) {
        let Some((token, out)) = &self.progress else {
            return;
        };
        let mut last = lock(&self.last);
        if done <= *last || self.stopped() {
            return;
        }
        *last = done;
        let mut params = json!({"progressToken": token, "progress": done, "message": message});
        if let Some(total) = total {
            params["total"] = total.into();
        }
        if let Some(task) = &self.task {
            params["_meta"] = related(task);
        }
        out(json!({"jsonrpc": "2.0", "method": "notifications/progress", "params": params}));
    }
}

/// An MCP server over one catalogue.
pub struct Server {
    shared: Arc<Shared>,
    jobs: mpsc::Sender<Job>,
    workers: Vec<JoinHandle<()>>,
}

type Job = Box<dyn FnOnce() + Send>;

struct Shared {
    entries: Vec<Entry>,
    settings: Settings,
    data_dir: PathBuf,
    audit: Audit,
    /// The client's name from `initialize`, for the audit log.
    client: Mutex<String>,
    /// The stop flags of calls not yet answered, by request ID.
    // ponytail: request IDs and the client name are server-wide, so one HTTP server serves one
    // client; key them by MCP-Session-Id when two clients share one
    running: Mutex<HashMap<String, Arc<AtomicBool>>>,
    tasks: Mutex<Tasks>,
    /// Signalled when a task ends.
    ended: Condvar,
}

#[derive(Default)]
struct Tasks {
    made: u64,
    all: HashMap<String, Task>,
}

struct Task {
    status: &'static str,
    message: Option<String>,
    created: String,
    updated: String,
    ttl: u64,
    born: Instant,
    stop: Arc<AtomicBool>,
    /// The call's result, once the task has ended.
    result: Option<Value>,
}

impl Task {
    fn json(&self, id: &str) -> Value {
        let mut task = json!({"taskId": id, "status": self.status, "createdAt": self.created,
            "lastUpdatedAt": self.updated, "ttl": self.ttl, "pollInterval": POLL_MS});
        if let Some(message) = &self.message {
            task["statusMessage"] = message.as_str().into();
        }
        task
    }

    fn end(&mut self, status: &'static str, result: Value) {
        self.status = status;
        self.updated = larch::bench::utc_now();
        self.result = Some(result);
    }
}

/// Where a call's result goes: the request's answer, or its task.
enum Reply {
    Answer(Value, Out),
    Task(String),
}

impl Server {
    /// A server over `entries`, logging to `data_dir`'s audit log. Fails when the write folder
    /// does not exist, or a WriteLocal entry takes no required string `path`.
    pub fn new(entries: Vec<Entry>, mut settings: Settings, data_dir: &Path) -> io::Result<Server> {
        if let Some(e) = (entries.iter())
            .find(|e| e.permission == Permission::WriteLocal && !takes_path(&e.schema))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{}: a WriteLocal entry takes a required string 'path', which the server keeps inside the write folder",
                    e.name
                ),
            ));
        }
        settings.write_folder = settings.write_folder.map(fs::canonicalize).transpose()?;
        let (jobs, queue) = mpsc::channel::<Job>();
        let queue = Arc::new(Mutex::new(queue));
        let workers = (0..settings.workers.max(1))
            .map(|_| {
                let queue = queue.clone();
                thread::spawn(move || {
                    loop {
                        // Its own statement, so the guard drops before the job runs; in a
                        // `while let` it would live through the body and run one job at a time.
                        let job = lock(&queue).recv();
                        let Ok(job) = job else { break };
                        job();
                    }
                })
            })
            .collect();
        let shared = Shared {
            entries,
            settings,
            data_dir: data_dir.into(),
            audit: Audit::new(data_dir),
            client: Mutex::new("unknown".into()),
            running: Mutex::default(),
            tasks: Mutex::default(),
            ended: Condvar::new(),
        };
        Ok(Server {
            shared: Arc::new(shared),
            jobs,
            workers,
        })
    }

    /// Serves newline-delimited JSON-RPC until `input` ends, then lets the calls still running
    /// finish and answer.
    pub fn stdio(self, input: impl BufRead, output: impl Write + Send + 'static) -> io::Result<()> {
        let output = Mutex::new(output);
        let out: Out = Arc::new(move |message| {
            let mut output = lock(&output);
            // A client that left cannot be told; the loop below ends at its end of input.
            let _ = writeln!(output, "{message}").and_then(|()| output.flush());
        });
        // A read error ends the loop, not the calls already running: they still log and answer.
        let read = input.split(b'\n').try_for_each(|line| {
            match String::from_utf8(line?) {
                Ok(line) if line.trim().is_empty() => {}
                Ok(line) => self.handle(&line, &out),
                Err(_) => out(error(
                    Value::Null,
                    -32700,
                    "parse error: a message is UTF-8",
                )),
            }
            Ok(())
        });
        drop(self.jobs);
        for worker in self.workers {
            worker
                .join()
                .map_err(|_| io::Error::other("an MCP worker panicked"))?;
        }
        read
    }

    /// Answers one message through `out`; a call answers later, from a worker.
    fn handle(&self, line: &str, out: &Out) {
        let message: Value = match serde_json::from_str(line) {
            Ok(message) => message,
            Err(e) => return out(error(Value::Null, -32700, &format!("parse error: {e}"))),
        };
        if !message.is_object() {
            return out(error(Value::Null, -32600, "a message is one JSON object"));
        }
        // A response needs nothing: this server sends no requests.
        let Some(method) = message["method"].as_str() else {
            return;
        };
        let params = &message["params"];
        let Some(id) = message.get("id").cloned() else {
            if method == "notifications/cancelled"
                && let Some(stop) = lock(&self.shared.running).get(&params["requestId"].to_string())
            {
                stop.store(true, Ordering::Relaxed);
            }
            return;
        };
        let shared = &self.shared;
        let result = match method {
            "initialize" => Ok(shared.initialize(params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(shared.list()),
            "tools/call" => return self.call(id, params, out),
            "tasks/get" => shared.with_task(params, |id, task| Ok(task.json(id))),
            "tasks/cancel" => shared.with_task(params, |id, task| {
                if task.result.is_some() {
                    return Err((
                        -32602,
                        format!("cannot cancel {id}: it is already {}", task.status),
                    ));
                }
                task.stop.store(true, Ordering::Relaxed);
                task.message = Some("cancelled by the client".into());
                task.end(
                    "cancelled",
                    tool_result(Err("the task was cancelled".into())),
                );
                shared.ended.notify_all();
                Ok(task.json(id))
            }),
            "tasks/result" => {
                let (shared, out) = (shared.clone(), out.clone());
                let task = params["taskId"].as_str().unwrap_or_default().to_owned();
                // ponytail: one thread per waiting tasks/result; wait on a shared one if clients
                // ever hold hundreds open
                thread::spawn(move || out(answer(id, shared.wait(&task))));
                return;
            }
            _ => Err((-32601, format!("method not found: {method}"))),
        };
        out(answer(id, result));
    }

    /// Checks a call against the policy and the schema, then queues it for a worker.
    fn call(&self, id: Value, params: &Value, out: &Out) {
        let shared = &self.shared;
        let Some(name) = params["name"].as_str() else {
            return out(error(id, -32602, "tools/call needs params.name"));
        };
        let Some(index) = shared.entries.iter().position(|e| e.name == name) else {
            return out(error(id, -32602, &format!("unknown tool: {name}")));
        };
        let args = match &params["arguments"] {
            Value::Null => json!({}),
            args @ Value::Object(_) => args.clone(),
            _ => return out(error(id, -32602, "tools/call arguments must be an object")),
        };
        let stop = Arc::new(AtomicBool::new(false));
        let to = match params.get("task") {
            Some(task) => {
                let ttl = task["ttl"].as_u64().unwrap_or(TASK_TTL_MS).min(TASK_TTL_MS);
                let (task, json) = shared.new_task(ttl, stop.clone());
                out(answer(id, Ok(json!({ "task": json }))));
                Reply::Task(task)
            }
            None => Reply::Answer(id, out.clone()),
        };
        let entry = &shared.entries[index];
        if let Err(refusal) = shared.admit(entry, &args) {
            return shared.refuse(entry, &args, &to, refusal);
        }
        if let Reply::Answer(id, _) = &to {
            lock(&shared.running).insert(id.to_string(), stop.clone());
        }
        let progress = params["_meta"]
            .get("progressToken")
            .map(|token| (token.clone(), out.clone()));
        let shared = shared.clone();
        let job = move || shared.run(index, &args, to, stop, progress);
        // Workers end only when `stdio` drops the sender, after its last message.
        (self.jobs.send(Box::new(job))).expect("the workers outlive the server");
    }
}

impl Shared {
    fn initialize(&self, params: &Value) -> Value {
        let asked = params["protocolVersion"].as_str().unwrap_or_default();
        let version = VERSIONS.into_iter().find(|v| *v == asked);
        *lock(&self.client) = params["clientInfo"]["name"]
            .as_str()
            .unwrap_or("unknown")
            .into();
        json!({
            "protocolVersion": version.unwrap_or(VERSIONS[0]),
            "capabilities": {"tools": {}, "tasks": {"cancel": {}, "requests": {"tools": {"call": {}}}}},
            "serverInfo": {"name": "iris", "version": env!("CARGO_PKG_VERSION")},
            "instructions": INSTRUCTIONS,
        })
    }

    fn list(&self) -> Value {
        let tools: Vec<Value> = (self.entries.iter())
            .filter(|e| {
                e.permission == Permission::Read
                    || e.permission == Permission::WriteLocal
                        && self.settings.write_folder.is_some()
            })
            .map(|e| {
                json!({"name": e.name, "description": e.description, "inputSchema": e.schema,
                    "execution": {"taskSupport": "optional"}})
            })
            .collect();
        json!({ "tools": tools })
    }

    /// Whether the policy and the entry's schema allow the call.
    fn admit(&self, entry: &Entry, args: &Value) -> Result<(), String> {
        let name = &entry.name;
        match (entry.permission, &self.settings.write_folder) {
            (Permission::WriteExternal, _) => {
                return Err(format!(
                    "{name} changes something outside this computer; Iris asks its own user for that, so it is not offered over MCP"
                ));
            }
            (Permission::Spend, _) => {
                return Err(format!(
                    "{name} spends money; Iris asks its own user for that, so it is not offered over MCP"
                ));
            }
            (Permission::WriteLocal, None) => {
                return Err(format!(
                    "{name} changes files, and Iris was started without a write folder"
                ));
            }
            (Permission::WriteLocal, Some(_)) | (Permission::Read, _) => {}
        }
        let spec = ToolSpec {
            name: name.clone(),
            description: entry.description.clone(),
            schema: entry.schema.clone(),
            permission: entry.permission,
            untrusted_results: false,
            max_result_tokens: self.settings.max_result_tokens,
            max_args_tokens: ARGS_TOKENS,
        };
        let call = ToolCall {
            name: name.clone(),
            args: args.clone(),
        };
        validate(&call, &spec).map_err(|e| e.to_string())
    }

    /// The arguments the entry runs on: a WriteLocal call's `path` made the real path inside the
    /// write folder it resolves to. Called on the worker, just before the run, so a link made
    /// while the call waited is seen.
    fn place(&self, entry: &Entry, args: &Value) -> Result<Value, String> {
        let mut args = args.clone();
        if let Some(folder) = (self.settings.write_folder.as_ref())
            .filter(|_| entry.permission == Permission::WriteLocal)
        {
            let ctx = ToolCtx {
                allowed: vec![folder.clone()],
                data_dir: self.data_dir.clone(),
                source: String::new(),
            };
            let path = args["path"].as_str().unwrap_or_default();
            let real = ctx.resolve(path, true).map_err(|e| e.to_string())?;
            args["path"] = real
                .to_str()
                .ok_or_else(|| format!("'{path}' resolves to a path that is not Unicode"))?
                .into();
        }
        Ok(args)
    }

    /// Runs an admitted call on a worker, between its intent and outcome lines, and sends or keeps
    /// its result. `args` are the client's, which the log hashes.
    fn run(
        &self,
        index: usize,
        args: &Value,
        to: Reply,
        stop: Arc<AtomicBool>,
        progress: Option<(Value, Out)>,
    ) {
        let entry = &self.entries[index];
        let log = |kind, bytes, error: Option<&str>| self.log(entry, args, kind, bytes, error);
        if stop.load(Ordering::Relaxed) {
            let logged = log(EventKind::Outcome, None, Some("cancelled before it ran"));
            return self.finish(&to, tool_result(unlogged(Err("cancelled".into()), logged)));
        }
        let admitted = match self.place(entry, args) {
            Ok(admitted) => admitted,
            Err(refusal) => return self.refuse(entry, args, &to, refusal),
        };
        if let Err(e) = log(EventKind::Intent, None, None) {
            let refusal =
                format!("the audit log could not record this call, so it did not run: {e}");
            return self.finish(&to, tool_result(Err(refusal)));
        }
        let ctx = Ctx {
            stop: stop.clone(),
            progress,
            task: match &to {
                Reply::Task(task) => Some(task.clone()),
                Reply::Answer(..) => None,
            },
            last: Mutex::new(f64::NEG_INFINITY),
        };
        // A panic must still end in an outcome line and an answer, and must not take a worker.
        let result = catch_unwind(AssertUnwindSafe(|| (entry.run)(&admitted, &ctx)))
            .unwrap_or_else(|panic| {
                let why = (panic.downcast_ref::<&str>().copied())
                    .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or_default();
                Err(format!("{} panicked: {why}", entry.name))
            });
        let logged = match &result {
            _ if stop.load(Ordering::Relaxed) => log(EventKind::Outcome, None, Some("cancelled")),
            Ok(output) => log(EventKind::Outcome, Some(output.text.len() as u64), None),
            Err(e) => log(EventKind::Outcome, None, Some(e)),
        };
        let result = self.cap(unlogged(result, logged));
        self.finish(&to, result);
    }

    /// Answers a call that will not run; one outcome line says why.
    fn refuse(&self, entry: &Entry, args: &Value, to: &Reply, refusal: String) {
        let logged = self.log(entry, args, EventKind::Outcome, None, Some(&refusal));
        self.finish(to, tool_result(unlogged(Err(refusal), logged)));
    }

    /// The result as the client gets it: the text cut to the setting, and structured content
    /// past the same cap dropped, with a line that says so.
    fn cap(&self, result: Result<Output, String>) -> Value {
        let max = self.settings.max_result_tokens;
        let result = result.map(|Output { text, structured }| {
            let bytes = structured.as_ref().map_or(0, |s| s.to_string().len());
            let (text, structured) = if bytes > max as usize * 4 {
                let note = format!("[structured content of {bytes} bytes dropped: past the cap]");
                (format!("{note}\n{text}"), None)
            } else {
                (text, structured)
            };
            Output {
                text: tool::truncate(text, max),
                structured,
            }
        });
        tool_result(result.map_err(|e| tool::truncate(e, max)))
    }

    /// Sends the result as the request's answer, unless the client cancelled it, or ends its task,
    /// unless the client cancelled that.
    fn finish(&self, to: &Reply, result: Value) {
        match to {
            Reply::Answer(id, out) => {
                let stop = lock(&self.running).remove(&id.to_string());
                if !stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
                    out(answer(id.clone(), Ok(result)));
                }
            }
            Reply::Task(id) => {
                let mut tasks = lock(&self.tasks);
                if let Some(task) = tasks.all.get_mut(id)
                    && task.result.is_none()
                {
                    let failed = result["isError"] == true;
                    task.end(if failed { "failed" } else { "completed" }, result);
                    self.ended.notify_all();
                }
            }
        }
    }

    fn new_task(&self, ttl: u64, stop: Arc<AtomicBool>) -> (String, Value) {
        let mut tasks = lock(&self.tasks);
        // A task past its TTL goes once it has ended: a client may still wait on a working one.
        // ponytail: swept when the next task starts; sweep on a timer if a quiet server's
        // finished results ever weigh
        (tasks.all)
            .retain(|_, t| t.result.is_none() || t.born.elapsed() < Duration::from_millis(t.ttl));
        tasks.made += 1;
        let id = format!("task-{}", tasks.made);
        let now = larch::bench::utc_now();
        let task = Task {
            status: "working",
            message: None,
            created: now.clone(),
            updated: now,
            ttl,
            born: Instant::now(),
            stop,
            result: None,
        };
        let json = task.json(&id);
        tasks.all.insert(id.clone(), task);
        (id, json)
    }

    fn with_task(
        &self,
        params: &Value,
        f: impl FnOnce(&str, &mut Task) -> Result<Value, (i64, String)>,
    ) -> Result<Value, (i64, String)> {
        let id = params["taskId"].as_str().unwrap_or_default();
        match lock(&self.tasks).all.get_mut(id) {
            Some(task) => f(id, task),
            None => Err((-32602, format!("no task {id}"))),
        }
    }

    /// The task's result once it has ended, as `tasks/result` returns it.
    fn wait(&self, id: &str) -> Result<Value, (i64, String)> {
        let mut tasks = lock(&self.tasks);
        loop {
            match tasks.all.get(id) {
                None => return Err((-32602, format!("no task {id}"))),
                Some(Task {
                    result: Some(result),
                    ..
                }) => {
                    let mut result = result.clone();
                    result["_meta"] = related(id);
                    return Ok(result);
                }
                Some(_) => {
                    tasks = self
                        .ended
                        .wait(tasks)
                        .unwrap_or_else(PoisonError::into_inner)
                }
            }
        }
    }

    fn log(
        &self,
        entry: &Entry,
        args: &Value,
        kind: EventKind,
        result_bytes: Option<u64>,
        error: Option<&str>,
    ) -> io::Result<()> {
        let client = lock(&self.client).clone();
        self.audit.append(AuditEvent {
            kind,
            tool: Some(entry.name.clone()),
            args_hash: Some(tool::args_hash(args)),
            permission: Some(entry.permission),
            result_bytes,
            error: error.map(Into::into),
            // AuditEvent has no field for these yet; it is another lane's file.
            detail: Some(json!({"created_by": "mcp", "client": client})),
            ..AuditEvent::default()
        })
    }
}

/// The result, or an error that adds the outcome line the log could not take: the call happened
/// (or failed) all the same, and the client must not be told otherwise.
fn unlogged(result: Result<Output, String>, logged: io::Result<()>) -> Result<Output, String> {
    let Err(e) = logged else {
        return result;
    };
    let said = match result {
        Ok(output) => output.text,
        Err(e) => e,
    };
    Err(format!(
        "{said}\nthe audit log could not record this call's outcome: {e}"
    ))
}

fn takes_path(schema: &Value) -> bool {
    schema["properties"]["path"]["type"] == "string"
        && (schema["required"].as_array()).is_some_and(|r| r.iter().any(|k| k == "path"))
}

fn tool_result(result: Result<Output, String>) -> Value {
    match result {
        Ok(output) => {
            let mut result =
                json!({"content": [{"type": "text", "text": output.text}], "isError": false});
            if let Some(structured) = output.structured {
                result["structuredContent"] = structured;
            }
            result
        }
        Err(e) => json!({"content": [{"type": "text", "text": e}], "isError": true}),
    }
}

fn related(task: &str) -> Value {
    json!({"io.modelcontextprotocol/related-task": {"taskId": task}})
}

fn answer(id: Value, result: Result<Value, (i64, String)>) -> Value {
    match result {
        Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
        Err((code, message)) => error(id, code, &message),
    }
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests;
