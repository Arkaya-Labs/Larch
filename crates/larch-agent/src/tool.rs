//! The `Tool` trait, its spec, the registry, the folder check every path goes through, and `run`,
//! which audits a call and cuts its result.

use std::{
    ffi::OsStr,
    fmt, fs, io,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    audit::{Audit, AuditEvent, EventKind},
    memory::MemoryError,
    parse::{ToolCall, ValidateError, validate},
};

/// A tool's risk class: what a call can change, which decides whether it waits for the user.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    /// Reads only.
    Read,
    /// Changes a file or the memory on this computer. Audit logs before the four classes say
    /// `write`.
    #[serde(alias = "write")]
    WriteLocal,
    /// Changes something outside this computer.
    WriteExternal,
    /// Spends the user's money.
    Spend,
}

impl Permission {
    /// Whether a call of this class waits for the user's yes, when the user's setting would let
    /// it run unasked (`allowed`) and the chat has read an untrusted result (`untrusted`): a
    /// spend always asks, and after an untrusted result every write does too.
    pub fn asks(self, allowed: bool, untrusted: bool) -> bool {
        match self {
            Self::Read => false,
            Self::WriteLocal | Self::WriteExternal => untrusted || !allowed,
            Self::Spend => true,
        }
    }
}

/// What the model sees of a tool, plus how much of its result it gets back.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolSpec {
    /// The name the model calls.
    pub name: String,
    /// One sentence the model reads to choose the tool.
    pub description: String,
    /// JSON Schema of the arguments object.
    pub schema: Value,
    /// The risk class.
    pub permission: Permission,
    /// Its results may carry text from outside the user's control, which makes the chat's later
    /// writes and spends ask.
    pub untrusted_results: bool,
    /// Results are cut at four bytes per token of this budget.
    pub max_result_tokens: u32,
    /// The tokens a call's arguments may take: the room a step that may call the tool leaves
    /// after its thinking budget.
    pub max_args_tokens: u32,
}

/// `max_args_tokens` of a tool whose arguments are a path or a few words.
pub const ARGS_TOKENS: u32 = 256;

impl ToolSpec {
    /// The spec in the OpenAI function shape the chat templates read (`GenParams.tools`).
    pub fn function(&self) -> Value {
        serde_json::json!({"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.schema}})
    }
}

/// A tool failure; `Display` is the message the model gets back.
#[derive(Debug)]
pub enum ToolError {
    /// The path resolves outside every allowed folder, or starts with `~name`.
    Denied(String),
    /// A file-system error on the named path.
    Io(String, io::Error),
    /// Arguments that pass the schema but make no sense together.
    Invalid(String),
    /// git exited with an error; its stderr.
    Git(String),
    /// A memory tool failed or refused.
    Memory(MemoryError),
    /// A skill run as a tool failed: a model turn, its search index or its log.
    Skill(String),
    /// A web page could not be fetched or read; the sentence says why.
    Fetch(String),
    /// No search provider gave results; the reason of each.
    Search(String),
    /// A write ran and gave this result or error, but its outcome line could not be logged.
    Unlogged(Result<String, String>, io::Error),
    /// An outside (MCP) tool failed or its server could not be reached; what it said.
    Outside(String),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Denied(path) if other_home(path) => write!(
                f,
                "access denied: '{path}' names another user's home folder; only ~ and ~/ are expanded, and a file of that name is './{path}'"
            ),
            Self::Denied(path) => {
                write!(f, "access denied: '{path}' is outside the allowed folders")
            }
            Self::Io(path, e) => write!(f, "'{path}': {e}"),
            Self::Invalid(sentence) | Self::Fetch(sentence) => f.write_str(sentence),
            Self::Git(stderr) => write!(f, "git failed: {stderr}"),
            Self::Memory(e) => e.fmt(f),
            Self::Skill(e) => write!(f, "the skill failed: {e}"),
            Self::Search(reasons) => f.write_str(reasons),
            Self::Outside(e) => write!(f, "the outside tool failed: {e}"),
            Self::Unlogged(Ok(result) | Err(result), e) => {
                write!(
                    f,
                    "{result}\nthe audit log could not record this write: {e}"
                )
            }
        }
    }
}

impl std::error::Error for ToolError {}

impl From<MemoryError> for ToolError {
    fn from(e: MemoryError) -> Self {
        Self::Memory(e)
    }
}

/// What a tool may touch: the allowed folders and the engine's data folder.
#[derive(Clone, Debug)]
pub struct ToolCtx {
    /// Folders the file and git tools may use; relative paths start at the first.
    pub allowed: Vec<PathBuf>,
    /// The engine's per-user data folder (`chat.rs`'s `data_dir()` rule), holding `agent/`.
    pub data_dir: PathBuf,
    /// Where a fact `remember` saves came from: the chat ID and turn.
    pub source: String,
}

impl ToolCtx {
    /// `path` canonicalized, if it lies inside an allowed folder; `new` also accepts a file that
    /// does not exist yet in an existing folder, and only inside an allowed folder: an allowed
    /// file is read only. Checked before any read or write of the path.
    pub fn resolve(&self, path: &str, new: bool) -> Result<PathBuf, ToolError> {
        self.resolve_in(path, new, false)
    }

    /// `resolve` of a new file in folders that may not exist yet either, below the nearest real
    /// one, each missing one a plain name: code mode's write_file makes them.
    pub fn resolve_deep(&self, path: &str) -> Result<PathBuf, ToolError> {
        self.resolve_in(path, true, true)
    }

    // ponytail: lexical canonicalize check; port the descriptor walk from the orchestrator's writable_root_containment when the terminal tool lands
    fn resolve_in(&self, path: &str, new: bool, deep: bool) -> Result<PathBuf, ToolError> {
        let denied = || ToolError::Denied(path.into());
        let expanded = expand_home(path)?;
        let joined = if expanded.is_relative() {
            // When the first folder is gone, a relative path fails; it never moves to the next.
            let first = self.allowed.first().ok_or_else(denied)?;
            let root = first_folder(first)?;
            // "." is the first attachment itself, an attached file too, below which nothing is:
            // "file.pdf/." failed "Not a directory" (Mac, 24 Nov).
            let joined = match expanded.components().all(|c| c == Component::CurDir) {
                true => root.clone(),
                false => root.join(&expanded),
            };
            // The system prompt shows the folder by its path, so a model writes "demo/a.md" (or
            // "demo") for demo's a.md; that is the folder itself unless it holds a "demo" too.
            match root.file_name().map(|name| expanded.strip_prefix(name)) {
                Some(Ok(rest)) if fs::symlink_metadata(&joined).is_err() => root.join(rest),
                _ => joined,
            }
        } else {
            // A model writes "/src/lib.rs" for the project's src/lib.rs: an absolute path outside
            // the folders whose top folder the first folder has is read from it, the same on
            // every machine; any other stays as given.
            // ponytail: Windows' rooted "\\src" takes the relative branch, unmapped; add there
            // when a Windows run shows it
            let first = self.allowed.first().map(|f| first_folder(f)).transpose()?;
            let inside = (self.allowed.iter()).any(|a| expanded.starts_with(a))
                || first.as_ref().is_some_and(|f| expanded.starts_with(f));
            let rest: PathBuf = expanded.components().skip(1).collect();
            let top = rest.components().next();
            match (first, top) {
                (Some(first), Some(top)) if !inside && first.join(top).exists() => first.join(rest),
                _ => expanded,
            }
        };
        let roots: Vec<PathBuf> = self
            .allowed
            .iter()
            .filter_map(|a| fs::canonicalize(a).ok())
            .collect();
        // An error names the problem only inside the folders; outside them, a path that does not
        // exist is denied like one that does, so the model cannot probe what is there. The folded
        // path and the nearest real folder above it must both be inside: a link can lead out.
        let failed = |e| {
            let folded = lexical(&joined);
            let near = joined
                .ancestors()
                .skip(1)
                .find_map(|a| fs::canonicalize(a).ok());
            let inside = roots
                .iter()
                .chain(&self.allowed)
                .any(|root| folded.starts_with(root))
                && near.is_some_and(|near| roots.iter().any(|root| near.starts_with(root)));
            if inside {
                ToolError::Io(path.into(), e)
            } else {
                denied()
            }
        };
        let real = match fs::canonicalize(&joined) {
            Ok(real) => real,
            Err(e) if new && e.kind() == io::ErrorKind::NotFound => {
                // A link here that canonicalize could not follow dangles; writing would follow it.
                if fs::symlink_metadata(&joined).is_ok() {
                    return Err(denied());
                }
                let (Some(parent), Some(name)) = (joined.parent(), joined.file_name()) else {
                    return Err(denied());
                };
                match fs::canonicalize(parent) {
                    Ok(real) => real.join(name),
                    Err(e) if !deep || fs::symlink_metadata(parent).is_ok() => {
                        return Err(failed(e));
                    }
                    // The missing folders below the nearest that exists, each a plain name: a
                    // link that dangles would be followed, and `..` has no file name.
                    Err(_) => {
                        let mut missing = vec![name];
                        let mut at = parent;
                        let real = loop {
                            if fs::symlink_metadata(at).is_ok() {
                                break fs::canonicalize(at).map_err(|_| denied())?;
                            }
                            missing.push(at.file_name().ok_or_else(denied)?);
                            at = at.parent().ok_or_else(denied)?;
                        };
                        missing
                            .iter()
                            .rev()
                            .fold(real, |path, name| path.join(name))
                    }
                }
            }
            Err(e) => return Err(failed(e)),
        };
        // An attached file is read only: a write (`new`) goes in an attached folder.
        if (roots.iter()).any(|root| real.starts_with(root) && (!new || root.is_dir())) {
            Ok(real)
        } else {
            Err(denied())
        }
    }
}

/// The variable that names the home folder `~` stands for.
const HOME: &str = if cfg!(windows) { "USERPROFILE" } else { "HOME" };

/// `path` with a leading `~` or `~/` made the home folder; `~name`, another user's home, is
/// denied rather than read as a folder of that name, and any other `~` is part of a name.
pub fn expand_home(path: &str) -> Result<PathBuf, ToolError> {
    expand_tilde(path, std::env::var_os(HOME).as_deref())
}

fn expand_tilde(path: &str, home: Option<&OsStr>) -> Result<PathBuf, ToolError> {
    let Some(rest) = path
        .strip_prefix('~')
        .filter(|rest| rest.is_empty() || rest.starts_with(std::path::is_separator))
    else {
        if other_home(path) {
            return Err(ToolError::Denied(path.into()));
        }
        return Ok(path.into());
    };
    // An empty or relative value would put the path in the first allowed folder.
    let home = home
        .map(Path::new)
        .filter(|h| h.is_absolute())
        .ok_or_else(|| {
            ToolError::Invalid(format!(
                "'{path}' starts with ~, but {HOME} names no folder; give the full path"
            ))
        })?;
    Ok(home.join(rest.trim_start_matches(std::path::is_separator)))
}

/// Whether `path` starts with `~name`, the home folder of the user `name`: a name of the
/// letters, digits, `.`, `_` and `-` that user names take, so `~$report.docx` is a file.
fn other_home(path: &str) -> bool {
    path.strip_prefix('~').is_some_and(|rest| {
        let name = rest
            .split(std::path::is_separator)
            .next()
            .unwrap_or_default();
        !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    })
}

/// The first allowed folder's real path: where relative paths start and the git tools run.
pub(crate) fn first_folder(first: &Path) -> Result<PathBuf, ToolError> {
    fs::canonicalize(first).map_err(|e| ToolError::Io(first.display().to_string(), e))
}

/// `command` with no console window of its own. The desktop app has no console to share, so on
/// Windows every console program it starts (git, icacls, a checker) would open, and close, one.
pub fn no_window(command: &mut std::process::Command) -> &mut std::process::Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW.
        command.creation_flags(0x0800_0000);
    }
    command
}

/// The path without the `\\?\` prefix of Windows' canonical paths, which git does not take and
/// the user does not write. A path that is not Unicode stays as it is: read lossily, it would name
/// another file.
pub fn plain(path: PathBuf) -> PathBuf {
    let Some(shown) = path.to_str() else {
        return path;
    };
    let plain = match shown.strip_prefix(r"\\?\UNC\") {
        Some(share) => Some(PathBuf::from(format!(r"\\{share}"))),
        None => shown.strip_prefix(r"\\?\").map(PathBuf::from),
    };
    plain.unwrap_or(path)
}

/// `path` with `.` and `..` folded away, without touching the disk.
// ponytail: hand-rolled; Path::normalize_lexically once it is stable
fn lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            part => out.push(part),
        }
    }
    out
}

/// A tool the model can call.
pub trait Tool: Send + Sync {
    /// Name, description, schema, permission and result budget.
    fn spec(&self) -> ToolSpec;
    /// Runs the tool on arguments that passed `validate`.
    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError>;
    /// Runs the checks `call` would fail, before the user is asked to allow a write, and returns
    /// the arguments as the call will run them: a path made the one it resolves to.
    fn prepare(&self, args: &Value, _ctx: &ToolCtx) -> Result<Value, ToolError> {
        Ok(args.clone())
    }
    /// The web URL the call opens, which after untrusted content must be one the chat has seen
    /// (`fetch::Seen`), or it asks.
    fn opens<'a>(&self, _args: &'a Value) -> Option<&'a str> {
        None
    }
    /// The tools a step that shows this one shows with it, by the chat's names: those its own
    /// text says to call first (an outside server's lookups).
    fn needs(&self) -> Vec<String> {
        Vec::new()
    }
}

type Call = fn(&Value, &ToolCtx) -> Result<String, ToolError>;
type Prepare = fn(&Value, &ToolCtx) -> Result<Value, ToolError>;

/// A built-in tool: its spec and plain functions.
pub(crate) struct Builtin {
    spec: ToolSpec,
    call: Call,
    prepare: Prepare,
}

impl Builtin {
    /// A built-in tool, boxed for the registry.
    pub fn boxed(
        name: &str,
        description: &str,
        permission: Permission,
        max_result_tokens: u32,
        schema: Value,
        call: Call,
    ) -> Box<dyn Tool> {
        Box::new(Self::new(
            name,
            description,
            permission,
            max_result_tokens,
            schema,
            call,
        ))
    }

    /// A built-in write, whose `prepare` runs before the user is asked.
    pub fn write(
        name: &str,
        description: &str,
        max_result_tokens: u32,
        max_args_tokens: u32,
        schema: Value,
        prepare: Prepare,
        call: Call,
    ) -> Box<dyn Tool> {
        let write = Permission::WriteLocal;
        let mut tool = Self::new(name, description, write, max_result_tokens, schema, call);
        tool.spec.max_args_tokens = max_args_tokens;
        Box::new(Builtin { prepare, ..tool })
    }

    /// A built-in read whose results carry text from outside the user's control.
    pub fn untrusted(
        name: &str,
        description: &str,
        max_result_tokens: u32,
        schema: Value,
        call: Call,
    ) -> Box<dyn Tool> {
        let read = Permission::Read;
        let mut tool = Self::new(name, description, read, max_result_tokens, schema, call);
        tool.spec.untrusted_results = true;
        Box::new(tool)
    }

    fn new(
        name: &str,
        description: &str,
        permission: Permission,
        max_result_tokens: u32,
        schema: Value,
        call: Call,
    ) -> Builtin {
        // A misspelled optional argument is refused with the names, not silently ignored.
        let mut schema = schema;
        schema["additionalProperties"] = Value::Bool(false);
        let spec = ToolSpec {
            name: name.into(),
            description: description.into(),
            schema,
            permission,
            untrusted_results: false,
            max_result_tokens,
            max_args_tokens: ARGS_TOKENS,
        };
        Builtin {
            spec,
            call,
            prepare: |args, _| Ok(args.clone()),
        }
    }
}

impl Tool for Builtin {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        (self.call)(args, ctx)
    }

    fn prepare(&self, args: &Value, ctx: &ToolCtx) -> Result<Value, ToolError> {
        (self.prepare)(args, ctx)
    }
}

/// A tool under another name (`Registry::renamed`); it runs as itself.
struct Renamed {
    tool: Box<dyn Tool>,
    spec: ToolSpec,
}

impl Tool for Renamed {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        self.tool.call(args, ctx)
    }

    fn prepare(&self, args: &Value, ctx: &ToolCtx) -> Result<Value, ToolError> {
        self.tool.prepare(args, ctx)
    }

    fn opens<'a>(&self, args: &'a Value) -> Option<&'a str> {
        self.tool.opens(args)
    }
}

/// A string argument the schema made required.
pub(crate) fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str, ToolError> {
    args[key]
        .as_str()
        .ok_or_else(|| ToolError::Invalid(format!("missing required argument '{key}'")))
}

/// A whole-number argument the schema checked. JSON Schema's `integer` also passes `3.0`, which
/// `as_u64` refuses; a float past `u64::MAX` saturates.
pub(crate) fn whole_arg(args: &Value, key: &str) -> Option<u64> {
    let value = &args[key];
    value.as_u64().or_else(|| value.as_f64().map(|n| n as u64))
}

/// The tools by name; a skill's allowlist decides which ones the model sees.
pub struct Registry {
    tools: Vec<Box<dyn Tool>>,
    /// Every tool's schema also takes `note` (`Registry::noted`).
    note: bool,
}

impl Registry {
    /// The built-in tools: files, git, memory, web_search, fetch_page, and ask_docs, web_answer,
    /// read_papers and research, which run only in a chat (`Turn` runs them with the chat's engine).
    pub fn builtin() -> Registry {
        let mut tools = crate::tools::files::tools();
        tools.extend(crate::tools::git::tools());
        tools.extend(crate::tools::code::tools());
        tools.extend(crate::tools::memory::tools());
        tools.push(crate::web::search::tool());
        tools.push(Box::new(crate::web::fetch::FetchPage));
        tools.push(Box::new(crate::skills::ask_docs::DocsTool {
            engine: None,
            stop: Default::default(),
        }));
        tools.push(Box::new(crate::skills::web_answer::WebTool {
            engine: None,
            seen: Default::default(),
            stop: Default::default(),
            progress: None,
        }));
        tools.push(Box::new(crate::skills::read_papers::PapersTool {
            engine: None,
            seen: Default::default(),
            stop: Default::default(),
            show: None,
            followup: None,
        }));
        tools.push(Box::new(crate::skills::research::ResearchTool {
            engine: None,
            seen: Default::default(),
            stop: Default::default(),
            plan_only: false,
            show: None,
            progress: None,
        }));
        Registry { tools, note: false }
    }

    /// The tools with each name, and each word of a description, as `rename` gives it: the bench's
    /// `--rename-tools`, which shows the model names that mean nothing.
    pub(crate) fn renamed(self, rename: fn(&str) -> String) -> Registry {
        let tools = (self.tools.into_iter())
            .map(|tool| {
                let mut spec = tool.spec();
                spec.name = rename(&spec.name);
                spec.description = rename(&spec.description);
                Box::new(Renamed { tool, spec }) as Box<dyn Tool>
            })
            .collect();
        Registry {
            tools,
            note: self.note,
        }
    }

    /// The registry whose every tool also takes an optional `note`, one line on why the call is
    /// made, which code mode shows the user live and strips before the call is shown again; the
    /// tools never read it.
    pub(crate) fn noted(self) -> Registry {
        Registry { note: true, ..self }
    }

    /// `tool`'s spec, with `note` when the registry is `noted`.
    fn spec_of(&self, tool: &dyn Tool) -> ToolSpec {
        let mut spec = tool.spec();
        if self.note {
            let why = "one short line: why this call";
            spec.schema["properties"]["note"] =
                serde_json::json!({"type": "string", "description": why});
        }
        spec
    }

    /// Puts `tool` in place of the tool of its name, or adds it: a code task's `run_tests`
    /// (`CodeTask::run_tests`), a test's tool that fails as no built-in can be made to, or its
    /// fake of a tool another lane ships.
    pub fn replace(&mut self, tool: Box<dyn Tool>) {
        let name = tool.spec().name;
        match self.tools.iter_mut().find(|t| t.spec().name == name) {
            Some(old) => *old = tool,
            None => self.tools.push(tool),
        }
    }

    /// The tool named `name`, whether or not an allowlist includes it.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .map(|t| &**t)
            .find(|t| t.spec().name == name)
    }

    /// The specs of the tools in `allow`, in registry order.
    pub fn specs(&self, allow: &[&str]) -> Vec<ToolSpec> {
        self.tools
            .iter()
            .map(|t| self.spec_of(&**t))
            .filter(|s| allow.contains(&s.name.as_str()))
            .collect()
    }

    /// `call` with each argument its tool's schema types `integer` and the model wrote as a
    /// number in a string (`"3"`) made that number; any other argument, and a call to a tool
    /// the registry lacks, stay as they are for `resolve` to judge.
    pub fn coerce(&self, call: &ToolCall) -> ToolCall {
        let mut call = call.clone();
        let Some(tool) = self.get(&call.name) else {
            return call;
        };
        let spec = self.spec_of(tool);
        let Some(props) = spec.schema["properties"].as_object() else {
            return call;
        };
        let Some(args) = call.args.as_object_mut() else {
            return call;
        };
        // A note never fails a call: one that is not a string is dropped.
        if self.note && args.get("note").is_some_and(|n| !n.is_string()) {
            args.remove("note");
        }
        for (key, value) in args.iter_mut() {
            let integer = props.get(key).is_some_and(|p| p["type"] == "integer");
            let number = value.as_str().and_then(|text| {
                (text.parse::<i64>().map(Value::from))
                    .or_else(|_| text.parse::<u64>().map(Value::from))
                    .ok()
            });
            if let Some(number) = number.filter(|_| integer) {
                *value = number;
            }
        }
        call
    }

    /// The tool `call` names, if `allow` includes it and the arguments fit its schema.
    pub fn resolve(&self, call: &ToolCall, allow: &[&str]) -> Result<&dyn Tool, ValidateError> {
        let tool = self
            .get(&call.name)
            .filter(|_| allow.contains(&call.name.as_str()))
            .ok_or_else(|| ValidateError::UnknownTool {
                name: call.name.clone(),
                available: self.specs(allow).into_iter().map(|s| s.name).collect(),
            })?;
        validate(call, &self.spec_of(tool))?;
        Ok(tool)
    }
}

/// The names the system prompt lists at the top of each allowed folder: the folder's real path
/// (`plain`), and its names, a folder's with `/`.
pub type Clues = Vec<(PathBuf, Vec<String>)>;

/// The tools whose `path` must exist, so a near miss of a listed name is the name meant.
const NEAR_TOOLS: [&str; 2] = ["list_dir", "read_file"];

/// Edits (insertions, deletions, substitutions) a near miss is from the name it meant.
const NEAR: usize = 2;

/// The `path` a `list_dir` or `read_file` call meant, when nothing is at its own: the path with its
/// last part replaced by the one name `clues` lists for its folder within `NEAR` edits of it. A
/// path that exists, that the folder check denies, or that is near two names or none, gets `None`,
/// and the tool's own error goes back.
// ponytail: only the last part, against the folders' top-level names; correct the parts before it
// too when the bench shows misspelled subfolders
pub fn near_path(call: &ToolCall, ctx: &ToolCtx, clues: &Clues) -> Option<String> {
    if !NEAR_TOOLS.contains(&call.name.as_str()) {
        return None;
    }
    let path = call.args.get("path")?.as_str()?;
    match ctx.resolve(path, false) {
        Err(ToolError::Io(_, e)) if e.kind() == io::ErrorKind::NotFound => {}
        _ => return None,
    }
    let trimmed = path.trim_end_matches(std::path::is_separator);
    // `a/.` names `a` but does not end with it: such a path is not a near miss.
    let name = Path::new(trimmed).file_name()?.to_str()?;
    if !trimmed.ends_with(name) {
        return None;
    }
    let parent = &trimmed[..trimmed.len() - name.len()];
    let folder = plain(
        ctx.resolve(if parent.is_empty() { "." } else { parent }, false)
            .ok()?,
    );
    let (_, names) = clues.iter().find(|(f, _)| *f == folder)?;
    let mut near = (names.iter())
        .map(|n| n.trim_end_matches('/'))
        .filter(|n| *n != name && strsim::levenshtein(n, name) <= NEAR);
    let meant = near.next()?;
    near.next().is_none().then(|| format!("{parent}{meant}"))
}

/// `text` cut to `max_result_tokens * 4` bytes, at the last character boundary at or below that,
/// with a `... N more lines` tail.
// ponytail: bytes stand in for tokens (4 per token); count with the tokenizer when the bench shows
// results cut far from their token budget
pub fn truncate(text: String, max_result_tokens: u32) -> String {
    let cap = (max_result_tokens as usize).saturating_mul(4);
    if text.len() <= cap {
        return text;
    }
    let cut = text.floor_char_boundary(cap);
    // A cut just before a line's end leaves that line whole: only the lines after it are more.
    let rest = &text[cut..];
    let rest = rest
        .strip_prefix("\r\n")
        .or_else(|| rest.strip_prefix('\n'))
        .unwrap_or(rest);
    let more = rest.lines().count();
    // Only the last line's end was past the cap: the text is at most 2 bytes over it, shorter
    // than any tail.
    if more == 0 {
        return text;
    }
    let kept = &text[..cut];
    let newline = if kept.is_empty() || kept.ends_with('\n') {
        ""
    } else {
        "\n"
    };
    format!("{kept}{newline}... {more} more lines")
}

/// The audit log's `args_hash`: SHA-256 of the arguments' JSON with keys sorted, so one call has
/// one hash in whatever order the model wrote them.
pub(crate) fn args_hash(args: &Value) -> String {
    let mut sorted = args.clone();
    sorted.sort_all_objects();
    larch::bench::hex(&larch::bench::sha256(sorted.to_string().as_bytes()))
}

/// Runs a validated call and audits it: a write gets an intent line before it runs and an
/// outcome line after, a read one line. The caller fills `event`'s session, chat, skill and
/// confirmation. A result comes back cut to the tool's `max_result_tokens`, except a local
/// write's, which names what it changed and so is never cut.
pub fn run(
    tool: &dyn Tool,
    args: &Value,
    ctx: &ToolCtx,
    audit: &Audit,
    event: AuditEvent,
) -> Result<String, ToolError> {
    let spec = tool.spec();
    let mut event = AuditEvent {
        tool: Some(spec.name.clone()),
        args_hash: Some(args_hash(args)),
        permission: Some(spec.permission),
        ..event
    };
    let audit_failed = |e| ToolError::Io("audit log".into(), e);
    let write = spec.permission != Permission::Read;
    if write {
        audit
            .append(AuditEvent {
                kind: EventKind::Intent,
                ..event.clone()
            })
            .map_err(audit_failed)?;
    }
    let result = tool.call(args, ctx);
    event.kind = if write {
        EventKind::Outcome
    } else {
        EventKind::Call
    };
    match &result {
        Ok(text) => event.result_bytes = Some(text.len() as u64),
        Err(e) => event.error = Some(e.to_string()),
    }
    let result = if spec.permission == Permission::WriteLocal {
        result
    } else {
        result.map(|text| truncate(text, spec.max_result_tokens))
    };
    match audit.append(event) {
        Ok(()) => result,
        // The write happened (or failed) all the same; the model must not be told otherwise.
        Err(e) if write => Err(ToolError::Unlogged(
            result.map_err(|err| err.to_string()),
            e,
        )),
        Err(e) => Err(audit_failed(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "/name/a.txt" is the first folder's name/a.txt when the folder has a "name"; a path whose
    /// top folder it lacks stays as given. Not on Windows, where "/name" has no drive and is not
    /// mapped (`resolve_in`).
    #[cfg(not(windows))]
    #[test]
    fn an_absolute_path_whose_top_the_project_has_is_the_projects() {
        let tmp = tempfile::tempdir().unwrap();
        let top = "larch-no-such-top";
        fs::create_dir(tmp.path().join(top)).unwrap();
        fs::write(tmp.path().join(top).join("a.txt"), "a").unwrap();
        let ctx = ToolCtx {
            allowed: vec![tmp.path().into()],
            data_dir: tmp.path().into(),
            source: String::new(),
        };
        let real = tmp.path().canonicalize().unwrap().join(top).join("a.txt");
        assert_eq!(ctx.resolve(&format!("/{top}/a.txt"), false).unwrap(), real);
        assert!(ctx.resolve("/etc/hostname", false).is_err());
        assert!(ctx.resolve("/larch-not-here/a.txt", true).is_err());
    }

    /// A noted registry takes a string note on any tool, and drops one of another type instead
    /// of failing the call.
    #[test]
    fn a_note_never_fails_a_call() {
        let registry = Registry::builtin().noted();
        let call = |note: Value| ToolCall {
            name: "view".into(),
            args: serde_json::json!({"path": "a.py", "note": note}),
        };
        let fine = call("why".into());
        assert!(registry.resolve(&registry.coerce(&fine), &["view"]).is_ok());
        let odd = registry.coerce(&call(5.into()));
        assert_eq!(odd.args, serde_json::json!({"path": "a.py"}));
        assert!(registry.resolve(&odd, &["view"]).is_ok());
    }

    #[test]
    fn plain_drops_the_verbatim_prefix_windows_adds() {
        let p = |s: &str| plain(PathBuf::from(s));
        assert_eq!(p(r"\\?\C:\work\repo"), PathBuf::from(r"C:\work\repo"));
        assert_eq!(
            p(r"\\?\UNC\server\share\repo"),
            PathBuf::from(r"\\server\share\repo")
        );
        assert_eq!(p("/home/u/repo"), PathBuf::from("/home/u/repo"));
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn plain_keeps_a_path_that_is_not_unicode_as_it_is() {
        // Read through U+FFFD, it would name another file.
        #[cfg(unix)]
        let odd = {
            use std::os::unix::ffi::OsStringExt;
            std::ffi::OsString::from_vec(b"\\\\?\\C:\\caf\xe9".to_vec())
        };
        #[cfg(windows)]
        let odd = {
            use std::os::windows::ffi::OsStringExt;
            let mut wide: Vec<u16> = r"\\?\C:\caf".encode_utf16().collect();
            wide.push(0xD800);
            std::ffi::OsString::from_wide(&wide)
        };
        assert_eq!(plain(PathBuf::from(&odd)), PathBuf::from(odd));
    }

    #[test]
    fn a_tilde_without_a_home_folder_fails_instead_of_turning_relative() {
        for home in [None, Some(""), Some("relative/home")] {
            let got = expand_tilde("~/x.md", home.map(OsStr::new));
            assert!(
                matches!(got, Err(ToolError::Invalid(_))),
                "{home:?}: {got:?}"
            );
        }
        let home = if cfg!(windows) {
            r"C:\Users\u"
        } else {
            "/home/u"
        };
        let got = expand_tilde("~/notes/x.md", Some(OsStr::new(home))).unwrap();
        assert_eq!(got, Path::new(home).join("notes/x.md"));
        assert_eq!(
            expand_tilde("notes/~x.md", None).unwrap(),
            Path::new("notes/~x.md")
        );
    }
}
