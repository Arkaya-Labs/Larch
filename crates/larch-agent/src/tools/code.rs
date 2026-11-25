//! Code mode's `view`, `search`, `edit`, `replace_symbol` and `write_file`: numbered windows of a
//! file or one of its symbols, a regex over the folder that skips what .gitignore ignores, a
//! one-match replace, a by-name replace of a symbol's lines and a whole-file write, each undone by
//! a newly failing check; `run_tests` and `bash`, which run commands only through `sandbox::run`.

use std::{
    collections::{BTreeMap, HashSet},
    ffi::OsStr,
    fs, io,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, atomic::AtomicBool},
};

use regex::Regex;
use serde_json::{Value, json};

use crate::{
    code::{
        self,
        map::{self, Symbol},
        task::{CodeTask, FAILED_IMPORT, TestRun},
    },
    skills::ask_docs::rank_words,
    tool::{
        ARGS_TOKENS, Builtin, Permission, Tool, ToolCtx, ToolError, ToolSpec, first_folder, plain,
        str_arg, truncate, whole_arg,
    },
    tools::files::{
        MAX_READ, allowed_target, deep_target, prepare_deep, prepare_write, read_window, write_at,
    },
};

/// The most lines one `view` shows.
const VIEW_LINES: u64 = 400;

/// The most matches one `search` lists.
const MAX_HITS: usize = 100;

/// `view`'s and `lookup`'s result budget in a chat, and their most in a code run (`readers`);
/// each fits its lines in it, so its hint is never cut off.
pub(crate) const VIEW_TOKENS: u32 = 4000;

/// `search`'s result budget, fitted as `view`'s is.
const SEARCH_TOKENS: u32 = 2000;

/// Room a fitted result keeps for the hint lines after it.
const HINT_BYTES: usize = 128;

/// The lines of a file a refused edit shows (`holds`): a screen around its old_text, not a view.
const HOLDS_LINES: usize = 40;

/// `holds`'s budget, for a file of long lines.
const HOLDS_TOKENS: u32 = 600;

/// The tokens of a file's symbols a refused `replace_symbol` or `view` lists.
const SYMBOLS_TOKENS: u32 = 400;

/// What a call that changes nothing is told, a refused no-op edit and a repeated call alike.
pub(crate) const NO_PROGRESS: &str =
    "This makes no progress: check the result, or change your approach.";

/// What a refused edit's error says when its text is in the file already.
pub(crate) const ALREADY: &str = "already in the file";

/// The tokens of a checker's error an undone edit returns.
const CHECK_TOKENS: u32 = 500;

/// The bytes of a matching line `search` shows: a minified file's line would fill the result.
const HIT_BYTES: usize = 300;

/// A command's wall-clock seconds, the sandbox's default: `bash` may ask for less.
// ponytail: run_tests gets 120 s too; give it the task's remaining budget when a cold build in a
// fresh worktree outruns it
const TIMEOUT_S: u64 = 120;

/// `bash`'s result budget; a longer output keeps its start and its end.
const BASH_TOKENS: u32 = 2000;

/// The tokens of failing tests' names `run_tests` lists.
pub(crate) const NAMES_TOKENS: u32 = 250;

/// The bytes of the output's end a `TestRun` keeps.
const TAIL_BYTES: usize = 4000;

/// `run_tests`' result budget: its status line, `NAMES_TOKENS` of names and the tail.
const RUN_TOKENS: u32 = 1300;

pub(crate) fn tools() -> Vec<Box<dyn Tool>> {
    let path = json!({"type": "string", "description": "File path"});
    let text = |description: &str| json!({"type": "string", "description": description});
    let [view, lookup] = readers(VIEW_TOKENS);
    vec![
        view,
        Builtin::boxed(
            "search",
            "Lists the lines that match a regex as path:line: text, in the files under path \
             (the folder when left out) that .gitignore does not ignore.",
            Permission::Read,
            SEARCH_TOKENS,
            json!({
                "type": "object",
                "properties": {
                    "pattern": text("Regular expression, in Rust regex syntax"),
                    "path": text("Folder or file to search")
                },
                "required": ["pattern"]
            }),
            search,
        ),
        lookup,
        Builtin::write(
            "edit",
            "Replaces old_text, which must occur exactly once in the file, with new_text; an \
             edit that breaks the file's syntax check is undone.",
            200,
            // Two snippets of a few lines each.
            1024,
            json!({
                "type": "object",
                "properties": {
                    "path": path,
                    "old_text": {"type": "string", "minLength": 1,
                                 "description": "Text to replace, as it is in the file"},
                    "new_text": text("Text to put in its place")
                },
                "required": ["path", "old_text", "new_text"]
            }),
            prepare_edit,
            edit,
        ),
        Builtin::write(
            "replace_symbol",
            "Replaces a whole function, method, class or other named block, from its first line \
             to its last, with new_text; an edit that breaks the file's syntax check is undone.",
            200,
            // A whole function of a few dozen lines.
            2048,
            json!({
                "type": "object",
                "properties": {
                    "path": path,
                    "name": {"type": "string", "minLength": 1, "description":
                        "Its name, qualified when the bare one is in more than one block: \
                         Board.dropDisc, Foo::bar, impl Foo, .cell:hover, an HTML element's id"},
                    "new_text": text("Its whole new text, header to last line")
                },
                "required": ["path", "name", "new_text"]
            }),
            prepare_replace,
            replace_symbol,
        ),
        Box::new(Bash {
            stop: None,
            tokens: BASH_TOKENS,
            read_only: false,
        }),
    ]
}

/// `view` and `lookup`, their results fitted to `tokens`: the chat's `VIEW_TOKENS`, a code
/// run's what its room keeps whole of a step (`Room::view_tokens`).
pub(crate) fn readers(tokens: u32) -> [Box<dyn Tool>; 2] {
    let path = json!({"type": "string", "description": "File path"});
    let whole = json!({"type": "integer", "minimum": 1});
    let text = |description: &str| json!({"type": "string", "description": description});
    [
        Reader::boxed(
            "view",
            "Shows a file's lines numbered, from line start (1 when left out), at most 400; \
             or, given symbol, the whole function, method, class or block of that name.",
            tokens,
            // No count: a small model fills it with 10 on every call and views from 1 again, so it
            // never sees line 11.
            json!({
                "type": "object",
                "properties": {
                    "path": path,
                    "start": whole,
                    "symbol": text("Name of a function, method, class or block to show whole")
                },
                "required": ["path"]
            }),
            view,
        ),
        Reader::boxed(
            "lookup",
            "Finds the code that matters for some names or words: the named blocks (functions, \
             classes, CSS rules, HTML elements) that match best, each whole, with the blocks it \
             uses (the element a script reads, the rule for a class, a function it calls).",
            tokens,
            json!({
                "type": "object",
                "properties": {
                    "query": text("Names or words of what to find, such as: dropDisc animation")
                },
                "required": ["query"]
            }),
            lookup,
        ),
    ]
}

/// A read whose result fits its spec's `max_result_tokens`, which it is given: `readers`.
struct Reader {
    spec: ToolSpec,
    read: fn(&Value, &ToolCtx, u32) -> Result<String, ToolError>,
}

impl Reader {
    /// A read named `name`, boxed for the registry.
    fn boxed(
        name: &str,
        description: &str,
        tokens: u32,
        mut schema: Value,
        read: fn(&Value, &ToolCtx, u32) -> Result<String, ToolError>,
    ) -> Box<dyn Tool> {
        schema["additionalProperties"] = Value::Bool(false);
        let spec = ToolSpec {
            name: name.into(),
            description: description.into(),
            schema,
            permission: Permission::Read,
            untrusted_results: false,
            max_result_tokens: tokens,
            max_args_tokens: ARGS_TOKENS,
        };
        Box::new(Reader { spec, read })
    }
}

impl Tool for Reader {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        (self.read)(args, ctx, self.spec.max_result_tokens)
    }
}

/// Code mode's write_file, for each attempt's registry (`Registry::replace`): the model names the
/// file and what it holds, and the loop has the file's text written from that as a plain answer
/// (`code::loop`'s `body`), which it adds as `text` before the call runs. It makes the folders a
/// new file goes in, inside the allowed folder, and keeps a text only as `write_whole` says.
pub(crate) fn write_file() -> Box<dyn Tool> {
    let path = json!({"type": "string", "description": "File path"});
    let about = json!({"type": "string", "minLength": 1, "description":
        "Everything the file must hold: its sections and parts, and the names, classes and ids \
         it shares with the other files"});
    Builtin::write(
        "write_file",
        "Writes a file whole, from what about says it holds, replacing what it held, and makes \
         the folders it goes in.",
        100,
        // A path and a paragraph; the text is written apart from the call.
        512,
        json!({
            "type": "object",
            "properties": {"path": path, "about": about},
            "required": ["path", "about"]
        }),
        prepare_whole,
        write_whole,
    )
}

/// write_file's write, after `prepare_whole`'s checks again: an attempt in a worktree asks no
/// one, so nothing ran them before. Then the check `edit` runs (`check`): a text that fails it,
/// for a file that passed it or was not there, is undone, the file as it was, or gone with the
/// folders made for it, and the checker's words go back as an undone edit's do.
fn write_whole(args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    prepare_whole(args, ctx)?;
    let (path, text) = (str_arg(args, "path")?, str_arg(args, "text")?);
    let real = deep_target(path, ctx)?;
    let io = |e| ToolError::Io(path.into(), e);
    // What a failing text is undone to: the file's text, or none for a new file. A file that is
    // not text, or fails its check already, keeps the write, as an edit to it does.
    let undo = match fs::read(&real) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Some(None),
        Err(e) => return Err(io(e)),
        Ok(bytes) => match String::from_utf8(bytes) {
            Ok(old) if check(&real, &old).map_err(io)?.is_none() => Some(Some(old)),
            _ => None,
        },
    };
    // Deepest first, as remove_dir must take them.
    let made: Vec<PathBuf> = (real.ancestors().skip(1))
        .take_while(|dir| !dir.exists())
        .map(Path::to_path_buf)
        .collect();
    let wrote = write_at(path, &real, text)?;
    let Some(undo) = undo else {
        return Ok(wrote);
    };
    let error = match check(&real, text) {
        Ok(None) => return Ok(wrote),
        Ok(Some(error)) => error,
        // The text must not stay unchecked: undo it, then report why.
        Err(e) => format!("the check could not run: {e}"),
    };
    match undo {
        Some(old) => crate::write_atomic(&real, &old, true).map_err(io)?,
        None => {
            fs::remove_file(&real).map_err(io)?;
            for dir in &made {
                // Empty but for the file, unless something else went in since.
                let _ = fs::remove_dir(dir);
            }
        }
    }
    Err(ToolError::Invalid(format!(
        "the write was rolled back: its text makes '{path}' fail its check:\n{}\n{}",
        truncate(error.trim_end().into(), CHECK_TOKENS),
        around(&real, text, &error)
    )))
}

/// `prepare_deep`, refusing a text the file there already holds, or one that would replace most
/// of it, or a whole page with a fragment of one: a small model writes the few lines it means to
/// add as the whole file, and the rest is gone.
fn prepare_whole(args: &Value, ctx: &ToolCtx) -> Result<Value, ToolError> {
    let args = prepare_deep(args, ctx)?;
    let path = str_arg(&args, "path")?;
    let new = str_arg(&args, "text")?;
    let old = fs::read_to_string(path);
    // A small model's writer copies the file it was shown; "wrote" would read as the change made.
    if old.as_deref().is_ok_and(|old| old == new) {
        return Err(ToolError::Invalid(format!(
            "this is the text '{path}' already holds: the write would change nothing; make the \
             change the request asks for, with edit on the lines it changes"
        )));
    }
    let old = old.unwrap_or_default();
    if old.len() > 200 && new.len() < old.len() / 2 {
        return Err(ToolError::Invalid(format!(
            "this would replace the {} bytes of '{path}' with {}, deleting most of it: change \
             part of it with edit, or write the whole file",
            old.len(),
            new.len()
        )));
    }
    let gone = dropped(path, &old, new, str_arg(&args, "about").unwrap_or_default());
    if !gone.is_empty() {
        return Err(ToolError::Invalid(format!(
            "this would take {} out of '{path}', which its about does not name: keep them, or \
             name each one the change removes",
            gone.join(", ")
        )));
    }
    let page = path.to_ascii_lowercase();
    let body = |text: &str| text.to_ascii_lowercase().contains("<body");
    if (page.ends_with(".html") || page.ends_with(".htm")) && body(&old) && !body(new) {
        return Err(ToolError::Invalid(format!(
            "'{path}' is a whole page and this text is a fragment of one, with no <body>: write \
             the whole page with this in it, or change part of it with edit"
        )));
    }
    Ok(args)
}

/// The top-level named blocks of `old`, the file at `path`, (`map::symbols`) that a whole new
/// text `new` no longer has and `named`, what the rewrite was asked for, does not name: a
/// writer that drops a function it was not asked to is cut short or summing up, and the rest of
/// the file would be gone. Every whole-file rewrite, a planned one too, checks this.
pub(crate) fn dropped(path: &str, old: &str, new: &str, named: &str) -> Vec<String> {
    let before = map::symbols(path, old);
    let now: HashSet<String> = map::symbols(path, new)
        .into_iter()
        .map(|s| s.name)
        .collect();
    let inside =
        |s: &Symbol| (before.iter()).any(|o| o.start <= s.start && s.end <= o.end && o != s);
    let mut gone: Vec<String> = (before.iter())
        .filter(|s| !inside(s) && !now.contains(&s.name))
        .map(|s| s.name.clone())
        .filter(|n| !named.contains(n.trim_start_matches(['.', '#'])))
        .collect();
    gone.dedup();
    gone
}

/// The lines numbered from `first`, one `n: line` each.
pub(crate) fn numbered<'a>(lines: impl Iterator<Item = &'a str>, first: usize) -> Vec<String> {
    (lines.enumerate())
        .map(|(i, line)| format!("{}: {line}", first + i))
        .collect()
}

/// `lines` joined while they fit in `tokens` (4 bytes each, as `tool::truncate` counts) less
/// `HINT_BYTES`, and how many did. A first line longer than that is cut to it, so a call always
/// shows something.
// ponytail: the rest of a cut first line is never shown; split long lines when a minified file
// needs reading
fn fit(lines: &[String], tokens: u32) -> (String, usize) {
    let cap = tokens as usize * 4 - HINT_BYTES;
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        if out.len() + line.len() + 1 > cap {
            if i == 0 {
                return (line[..line.floor_char_boundary(cap)].into(), 1);
            }
            return (out, i);
        }
        if i > 0 {
            out.push('\n');
        }
        out += line;
    }
    (out, lines.len())
}

/// `notes` after `out`, one per line.
fn with_notes(mut out: String, notes: &[String]) -> String {
    for note in notes {
        if !out.is_empty() {
            out.push('\n');
        }
        out += note;
    }
    out
}

fn view(args: &Value, ctx: &ToolCtx, tokens: u32) -> Result<String, ToolError> {
    // A small model views a folder to see what is in it, and "not a file" alone sends it back
    // to view the folder again.
    let path = str_arg(args, "path")?;
    if ctx.resolve(path, false).is_ok_and(|p| p.is_dir()) {
        return Err(ToolError::Invalid(format!(
            "'{path}' is a folder: the repository map lists its files; view one of them"
        )));
    }
    if let Some(name) = args["symbol"].as_str() {
        let text = read_window(&json!({"path": path}), ctx)?.0.concat();
        let symbols = map::symbols(path, &text);
        let s = symbol(path, &symbols, name)?;
        let span = text.lines().skip(s.start_line - 1);
        let lines = numbered(span.take(s.end_line + 1 - s.start_line), s.start_line);
        let (shown, taken) = fit(&lines, tokens);
        return Ok(match taken == lines.len() {
            true => shown,
            false => with_notes(
                shown,
                &[format!(
                    "... more lines; view from start {}",
                    s.start_line + taken
                )],
            ),
        });
    }
    let start = whole_arg(args, "start").unwrap_or(1).max(1);
    let start = match start {
        1 => 1,
        start => around_block(path, ctx, start, tokens)?,
    };
    // One line past the window says whether more follow; read_window runs the path, size and
    // type checks, and leaves the cut to `fit`, so read_file's own tail is never a line here.
    let end = start.saturating_add(VIEW_LINES);
    let window = json!({"path": args["path"], "start_line": start, "end_line": end});
    let text = read_window(&window, ctx)?.0.concat();
    let lines = numbered(text.lines().take(VIEW_LINES as usize), start as usize);
    if lines.is_empty() && start > 1 {
        let path = str_arg(args, "path")?;
        return Err(ToolError::Invalid(format!(
            "start {start} is past the end of '{path}'"
        )));
    }
    let (shown, taken) = fit(&lines, tokens);
    if taken == lines.len() && text.lines().nth(VIEW_LINES as usize).is_none() {
        return Ok(shown);
    }
    let next = start + taken as u64;
    Ok(with_notes(
        shown,
        &[format!("... more lines; view from start {next}")],
    ))
}

/// Where a view from line `start` of the file at `path` starts instead: the first line of the
/// outermost mapped block `start` is inside of whose lines all fit `tokens`, so a function is
/// read whole, not in windows that cut it; `start` when none does.
fn around_block(path: &str, ctx: &ToolCtx, start: u64, tokens: u32) -> Result<u64, ToolError> {
    let text = read_window(&json!({"path": path}), ctx)?.0.concat();
    let lines: Vec<&str> = text.lines().collect();
    let symbols = map::symbols(path, &text);
    let at = start as usize;
    let mut inside: Vec<&Symbol> = (symbols.iter())
        .filter(|s| s.start_line < at && at <= s.end_line && s.end_line <= lines.len())
        .collect();
    inside.sort_by_key(|s| s.start_line);
    let fits = |s: &&&Symbol| {
        let block = numbered(
            lines[s.start_line - 1..s.end_line].iter().copied(),
            s.start_line,
        );
        block.len() as u64 <= VIEW_LINES && fit(&block, tokens).1 == block.len()
    };
    Ok(inside
        .iter()
        .find(fits)
        .map_or(start, |s| s.start_line as u64))
}

/// The folder's blocks that match `query` best, each whole and numbered, then the blocks it
/// uses (`code::lookup`), as far as `tokens` goes.
fn lookup(args: &Value, ctx: &ToolCtx, tokens: u32) -> Result<String, ToolError> {
    let query = str_arg(args, "query")?;
    // A query of no words matches nothing: said before any file is read.
    if rank_words(query).is_empty() {
        return Err(ToolError::Invalid(
            "query has no words: give names or words of the code to find".into(),
        ));
    }
    let first = (ctx.allowed.first()).ok_or_else(|| ToolError::Denied(".".into()))?;
    let parts = code::lookup::lookup(&first_folder(first)?, query)?;
    if parts.is_empty() {
        return Ok(format!(
            "no block matches '{query}': search takes a regex over every line"
        ));
    }
    let (out, taken) = fit(&parts, tokens);
    Ok(match taken < parts.len() {
        true => with_notes(
            out,
            &["... more blocks; lookup fewer words, or view one".into()],
        ),
        false => out,
    })
}

fn search(args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let pattern = str_arg(args, "pattern")?;
    // A small model writes a JavaScript literal, /background:/g, whose slashes are in no line:
    // the pattern is what is between them, and its i flag ignores case.
    let literal = Regex::new(r"^/(.+)/([a-z]*)$").expect("a valid regex");
    let pattern = match literal.captures(pattern) {
        Some(c) if c[2].contains('i') => format!("(?i){}", &c[1]),
        Some(c) => c[1].to_string(),
        None => pattern.to_string(),
    };
    let re = Regex::new(&pattern)
        .map_err(|e| ToolError::Invalid(format!("pattern is not a valid regex: {e}")))?;
    let path = args["path"].as_str().unwrap_or(".");
    let real = ctx.resolve(path, false)?;
    let io = |e| ToolError::Io(path.into(), e);
    let roots: Vec<PathBuf> = (ctx.allowed.iter())
        .filter_map(|a| fs::canonicalize(a).ok())
        .collect();
    // The .gitignore files from the allowed folder down to the searched one apply, as git reads them.
    let root = (roots.iter())
        .filter(|r| real.starts_with(r))
        .max_by_key(|r| r.as_os_str().len())
        .ok_or_else(|| ToolError::Denied(path.into()))?;
    let mut search = Search {
        re,
        rules: Vec::new(),
        hits: Vec::new(),
        unread: 0,
        shown_from: roots.first().cloned(),
    };
    let above: Vec<&Path> = (real.ancestors().skip(1))
        .take_while(|a| a.starts_with(root))
        .collect();
    for dir in above.into_iter().rev() {
        load_gitignore(dir, &mut search.rules).map_err(io)?;
    }
    let meta = fs::metadata(&real).map_err(io)?;
    if meta.is_dir() {
        search.walk(&real).map_err(io)?;
    } else if meta.is_file() {
        search.grep(&real).map_err(io)?;
    } else {
        // A FIFO or a device would block or never end, as read_file says.
        return Err(ToolError::Invalid(format!("'{path}' is not a file")));
    }
    let hits = &search.hits[..search.hits.len().min(MAX_HITS)];
    let (out, taken) = fit(hits, SEARCH_TOKENS);
    let mut notes = Vec::new();
    // An empty result reads as a failed call, which a small model makes again and again.
    if search.hits.is_empty() {
        notes.push(format!("no line matches '{pattern}'"));
    }
    if taken < search.hits.len() {
        notes.push("... more matches; narrow the pattern or the path".into());
    }
    if search.unread > 0 {
        notes.push(format!(
            "... {} files or folders could not be read",
            search.unread
        ));
    }
    Ok(with_notes(out, &notes))
}

/// One `.gitignore` line, matched against paths relative to its file's folder.
pub(crate) struct Rule {
    base: PathBuf,
    glob: Regex,
    negate: bool,
    dir_only: bool,
}

/// A search in progress: its regex, the rules of the folders it is in, and what it found.
struct Search {
    re: Regex,
    rules: Vec<Rule>,
    hits: Vec<String>,
    /// Files and folders below the searched one that failed to read; the rest is still searched.
    unread: usize,
    /// The first allowed folder, which relative paths start at: hits inside it show relative.
    shown_from: Option<PathBuf>,
}

impl Search {
    /// Fails only when `dir` itself cannot be read; an entry below it that fails is counted in
    /// `unread`, as ripgrep skips it, so one unreadable file loses no other file's hits.
    fn walk(&mut self, dir: &Path) -> io::Result<()> {
        let outer = self.rules.len();
        load_gitignore(dir, &mut self.rules)?;
        let mut entries = Vec::new();
        for entry in fs::read_dir(dir)? {
            match entry {
                Ok(entry) => entries.push(entry),
                Err(_) => self.unread += 1,
            }
        }
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            if self.hits.len() > MAX_HITS {
                break;
            }
            // A link is never followed: it can lead out of the allowed folder.
            let (Ok(ty), path) = (entry.file_type(), entry.path()) else {
                self.unread += 1;
                continue;
            };
            if entry.file_name() == ".git" || ignored(&self.rules, &path, ty.is_dir()) {
                continue;
            }
            let read = if ty.is_dir() {
                self.walk(&path)
            } else if ty.is_file() {
                self.grep(&path)
            } else {
                Ok(())
            };
            if read.is_err() {
                self.unread += 1;
            }
        }
        self.rules.truncate(outer);
        Ok(())
    }

    /// The matching lines of a text file; a file over `MAX_READ` or not UTF-8 is not text.
    fn grep(&mut self, file: &Path) -> io::Result<()> {
        if fs::metadata(file)?.len() > MAX_READ {
            return Ok(());
        }
        let Ok(text) = String::from_utf8(fs::read(file)?) else {
            return Ok(());
        };
        let shown = (self.shown_from.as_deref())
            .and_then(|from| file.strip_prefix(from).ok())
            .map_or_else(|| plain(file.into()), PathBuf::from);
        let shown = shown.display().to_string();
        // git's separator, as the repository map shows paths: a `\` the model copies into a
        // call's JSON is an escape.
        #[cfg(windows)]
        let shown = shown.replace('\\', "/");
        for (i, line) in text.lines().enumerate() {
            if self.hits.len() > MAX_HITS {
                break;
            }
            if self.re.is_match(line) {
                let line = &line[..line.floor_char_boundary(HIT_BYTES)];
                self.hits.push(format!("{shown}:{}: {line}", i + 1));
            }
        }
        Ok(())
    }
}

/// Whether `rules` ignore `path`, a folder when `dir`: the last rule that matches decides, as in
/// git; a folder's own rules come after its parents'.
pub(crate) fn ignored(rules: &[Rule], path: &Path, dir: bool) -> bool {
    let matches = |r: &&Rule| {
        (dir || !r.dir_only)
            && path
                .strip_prefix(&r.base)
                .is_ok_and(|rel| r.glob.is_match(&rel.to_string_lossy().replace('\\', "/")))
    };
    rules.iter().rev().find(matches).is_some_and(|r| !r.negate)
}

/// `dir`'s `.gitignore` rules onto `rules`; a folder without one adds none.
// ponytail: .gitignore files only; read .git/info/exclude and core.excludesFile when a user's
// search lists what only those ignore
pub(crate) fn load_gitignore(dir: &Path, rules: &mut Vec<Rule>) -> io::Result<()> {
    let bytes = match fs::read(dir.join(".gitignore")) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        bytes => bytes?,
    };
    // Lossy, as git reads bytes: a Latin-1 name in one rule leaves the other rules working.
    for line in String::from_utf8_lossy(&bytes).lines().map(str::trim_end) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (negate, line) = line.strip_prefix('!').map_or((false, line), |l| (true, l));
        let (dir_only, line) = line.strip_suffix('/').map_or((false, line), |l| (true, l));
        // A slash anywhere but the end ties the pattern to this folder; otherwise any depth.
        let anchored = line.contains('/');
        // A pattern that does not translate (an unclosed `[`) is dropped: git ignores nothing
        // with it either.
        let Some(body) = glob_regex(line.strip_prefix('/').unwrap_or(line)) else {
            continue;
        };
        let re = match anchored {
            true => format!("^{body}$"),
            false => format!("^(?:.*/)?{body}$"),
        };
        if let Ok(glob) = Regex::new(&re) {
            let base = dir.into();
            rules.push(Rule {
                base,
                glob,
                negate,
                dir_only,
            });
        }
    }
    Ok(())
}

/// A gitignore glob as a regex body: `*`, `?` and `[...]` stay inside one name, `**` crosses
/// folders; `None` for an unclosed `[`.
fn glob_regex(glob: &str) -> Option<String> {
    let mut out = String::new();
    let mut chars = glob.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' if chars.next_if_eq(&'*').is_some() => {
                out += match chars.next_if_eq(&'/') {
                    Some(_) => "(?:.*/)?",
                    None => ".*",
                }
            }
            '*' => out += "[^/]*",
            '?' => out += "[^/]",
            '[' => {
                let negate = chars.next_if(|&c| c == '!' || c == '^').is_some();
                let mut class = String::from(if negate { "^" } else { "" });
                let mut closed = false;
                for c in chars.by_ref() {
                    match c {
                        ']' => {
                            closed = true;
                            break;
                        }
                        '-' => class.push('-'),
                        c => class += &regex::escape(&c.to_string()),
                    }
                }
                if !closed {
                    return None;
                }
                // Intersected with [^/]: a bracket never matches a folder separator in git.
                out += &format!("[[{class}]&&[^/]]");
            }
            '\\' => out += &regex::escape(&chars.next().map(String::from).unwrap_or_default()),
            c => out += &regex::escape(&c.to_string()),
        }
    }
    Some(out)
}

/// The real path, the file's text, the text after the edit, and the line the match starts on.
type Edited = (PathBuf, String, String, usize);

/// The edit `args` ask for, refused unless it changes the text and `old_text` occurs exactly once.
fn edited(args: &Value, ctx: &ToolCtx) -> Result<Edited, ToolError> {
    let path = str_arg(args, "path")?;
    let (old, new) = (str_arg(args, "old_text")?, str_arg(args, "new_text")?);
    let (real, text) = source(path, ctx)?;
    // Overlapping matches count too (`}}` is twice in `}}}`), which match_indices skips.
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(i) = text[from..].find(old).filter(|_| found.len() < 3) {
        let at = from + i;
        found.push(at);
        from = at + text[at..].chars().next().map_or(1, char::len_utf8);
    }
    let line = |at: usize| text[..at].matches('\n').count() + 1;
    // A small model sends the same edit again, or one for a line it already changed: the file as
    // it is now is what its next edit needs.
    // The same edit sent again after it was made: its new_text is there, its old_text gone. Its
    // first line must be one of the file's whole, so a short new_text is no match by chance.
    let whole = |l: &str| text.lines().any(|t| t.trim() == l.trim());
    let first = new.lines().find(|l| !l.trim().is_empty());
    let made = found.is_empty() && first.is_some_and(whole) && text.contains(new);
    if old == new || found.is_empty() {
        let why = match (old == new, made) {
            (true, _) => format!("new_text is old_text, so the text is {ALREADY}. {NO_PROGRESS}"),
            (false, true) => format!(
                "the change is {ALREADY}: new_text is in '{path}' and old_text is not. \
                 {NO_PROGRESS}"
            ),
            (false, false) => format!(
                "old_text has 0 matches in '{path}': it is not in the file now; copy it from \
                 the text below{}",
                // A function rewritten whole needs no text copied, only its name.
                match map::symbols(path, &text).is_empty() {
                    true => "",
                    false => ", or replace a whole function or class by name with replace_symbol",
                }
            ),
        };
        // Where old_text was: the whole line its first line, trimmed, still is, else any match.
        let first = old.lines().map(str::trim).find(|l| !l.is_empty());
        let near = |first: &str| {
            (text.lines().position(|l| l.trim() == first))
                .map(|i| i + 1)
                .or_else(|| text.find(first).map(line))
        };
        let at = match found.first() {
            Some(&at) => Some(line(at)),
            None if made => text.find(new).map(line),
            None => first.and_then(near),
        };
        let from = at.map_or(1, |n| n.saturating_sub(HOLDS_LINES / 2).max(1));
        return Err(ToolError::Invalid(format!(
            "{why}\n{}",
            holds(path, &text, from)
        )));
    }
    let [at] = found[..] else {
        let n = match found.len() {
            3 => "3 or more".into(),
            n => n.to_string(),
        };
        return Err(ToolError::Invalid(format!(
            "old_text has {n} matches in '{path}', not 1; give it exactly as the file has it, \
             with enough lines around it to match once"
        )));
    };
    let after = format!("{}{new}{}", &text[..at], &text[at + old.len()..]);
    let line = line(at);
    Ok((real, text, after, line))
}

/// The real path edits write `path` at, and the file's text, refused over `MAX_READ`.
fn source(path: &str, ctx: &ToolCtx) -> Result<(PathBuf, String), ToolError> {
    let real = allowed_target(path, ctx)?;
    let io = |e| ToolError::Io(path.into(), e);
    let len = fs::metadata(&real).map_err(io)?.len();
    if len > MAX_READ {
        return Err(ToolError::Invalid(format!(
            "'{path}' is {len} bytes; edit edits files up to {MAX_READ} bytes"
        )));
    }
    let text = fs::read_to_string(&real).map_err(io)?;
    Ok((real, text))
}

/// The one symbol of `symbols`, the file at `path`'s, that `name` names (`map::find`), or why
/// none is, with the names to give instead.
fn symbol<'a>(
    path: &str,
    symbols: &'a [map::Symbol],
    name: &str,
) -> Result<&'a map::Symbol, ToolError> {
    map::find(symbols, name).map_err(|found| {
        ToolError::Invalid(match found.len() {
            _ if symbols.is_empty() => format!(
                "'{path}' has no function, class or other named block the map reads; change it \
                 with edit"
            ),
            0 => format!(
                "no symbol '{name}' in '{path}'; it has, as kind, name and lines:\n{}",
                map::outline(&symbols.iter().collect::<Vec<_>>(), SYMBOLS_TOKENS)
            ),
            n => format!(
                "'{name}' names {n} symbols in '{path}'; give one as listed, with its kind or \
                 its lines:\n{}",
                map::outline(&found, SYMBOLS_TOKENS)
            ),
        })
    })
}

/// The replace `args` ask for, as `edited` gives an edit's, and what the result says it did:
/// the symbol `name` names replaced with `new_text`, its lines whole, or only its own bytes
/// where it shares a line with other code (`<p>Score: <span id="s">0</span></p>`); and whether
/// it keeps every other symbol (`others_kept`).
fn symbol_edited(
    args: &Value,
    ctx: &ToolCtx,
) -> Result<(Edited, String, Result<(), ToolError>), ToolError> {
    let path = str_arg(args, "path")?;
    let (name, new) = (str_arg(args, "name")?, str_arg(args, "new_text")?);
    let (real, text) = source(path, ctx)?;
    let symbols = map::symbols(path, &text);
    let s = symbol(path, &symbols, name)?;
    let line_start = text[..s.start].rfind('\n').map_or(0, |n| n + 1);
    let line_end = text[s.end..]
        .find('\n')
        .map_or(text.len(), |n| s.end + n + 1);
    let alone_from = text[line_start..s.start].trim().is_empty();
    let alone_to = text[s.end..line_end].trim().is_empty();
    let from = if alone_from { line_start } else { s.start };
    let to = if alone_to { line_end } else { s.end };
    let mut new = new.to_string();
    if !alone_from {
        new = new.trim_start().into();
    }
    if !alone_to {
        new = new.trim_end().into();
    } else if text[..to].ends_with('\n') && !new.ends_with('\n') {
        // Whole lines out, whole lines in: the line after it stays its own.
        new.push('\n');
    }
    if text[from..to] == new {
        return Err(ToolError::Invalid(format!(
            "new_text is {} {} as it is, so the text is {ALREADY}. {NO_PROGRESS}",
            s.kind, s.name
        )));
    }
    let after = format!("{}{new}{}", &text[..from], &text[to..]);
    let kept = others_kept(path, &text, &symbols, s, &after, from..from + new.len());
    let done = format!(
        "replaced {} {} in {path}, lines {}-{}, with lines {}-{}",
        s.kind,
        s.name,
        s.start_line,
        s.end_line,
        s.start_line,
        s.start_line + new.lines().count().max(1) - 1
    );
    Ok(((real, text, after, s.start_line), done, kept))
}

/// Refuses a replace of `s`, one of `symbols` of `text` (the file at `path`), whose result
/// `after` holds the new text at `new`, when it changes more than `s`: mapped again, every other
/// symbol must be there as it was, and none may cross the new text's bounds. A span the map got
/// wrong, in any language, would otherwise eat or split the block beside it. The blocks inside
/// `s` (in its bytes, and on its first line or indented deeper) and the ones around it may
/// change.
fn others_kept(
    path: &str,
    text: &str,
    symbols: &[Symbol],
    s: &Symbol,
    after: &str,
    new: std::ops::Range<usize>,
) -> Result<(), ToolError> {
    let indent = |t: &str, at: usize| {
        let line = &t[t[..at].rfind('\n').map_or(0, |n| n + 1)..];
        line.len() - line.trim_start_matches([' ', '\t']).len()
    };
    let inner = |o: &Symbol| {
        s.start <= o.start
            && o.end <= s.end
            && (o.start_line == s.start_line || indent(text, o.start) > indent(text, s.start))
    };
    let around = |o: &Symbol| o.start <= s.start && s.end <= o.end && o != s;
    let was: Vec<(&str, &str)> = (symbols.iter())
        .filter(|o| !inner(o) && !around(o))
        .map(|o| (o.name.as_str(), &text[o.start..o.end]))
        .collect();
    let outer: Vec<&str> = (symbols.iter().filter(|o| around(o)))
        .map(|o| o.name.as_str())
        .collect();
    let mapped = map::symbols(path, after);
    let mut now = Vec::new();
    let mut crossed = Vec::new();
    // A block the new text ends its line in (a Python class whose last method it is) ends
    // before that line's newline: it holds the text up to there.
    let end = new.start + after[new.clone()].trim_end_matches(['\r', '\n']).len();
    for o in &mapped {
        let inside = new.start <= o.start && o.end <= new.end;
        let holds = o.start <= new.start && end <= o.end && outer.contains(&o.name.as_str());
        if inside || holds {
            continue;
        }
        if o.start < new.end && new.start < o.end {
            crossed.push(o.name.as_str());
        }
        now.push((o.name.as_str(), &after[o.start..o.end]));
    }
    let mut changed: Vec<&str> = (was.iter().filter(|w| !now.contains(w)))
        .chain(now.iter().filter(|n| !was.contains(n)))
        .map(|(name, _)| *name)
        .chain(crossed)
        .collect();
    changed.sort_unstable();
    changed.dedup();
    if changed.is_empty() {
        return Ok(());
    }
    Err(ToolError::Invalid(format!(
        "replacing {} {} would change {} too: change it with edit",
        s.kind,
        s.name,
        changed.join(", ")
    )))
}

/// `prepare_write`'s path checks, then the lookup: a refused replace never reaches the y/n.
/// Whether it keeps the other symbols is told after the file's check (`replace_symbol`).
fn prepare_replace(args: &Value, ctx: &ToolCtx) -> Result<Value, ToolError> {
    let args = prepare_write(args, ctx)?;
    let _ = symbol_edited(&args, ctx)?;
    Ok(args)
}

/// The replace, then `others_kept`'s word once the file's check passed: a new text that breaks
/// the file maps its neighbours wrong too, and the checker says why better.
fn replace_symbol(args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let path = str_arg(args, "path")?;
    let (edited, done, kept) = symbol_edited(args, ctx)?;
    let (real, before) = (edited.0.clone(), edited.1.clone());
    let done = write_checked(path, edited, done)?;
    if let Err(e) = kept {
        crate::write_atomic(&real, &before, true).map_err(|e| ToolError::Io(path.into(), e))?;
        return Err(e);
    }
    Ok(done)
}

/// `text`, the file at `path`, numbered from line `from`, `HOLDS_LINES` as far as `HOLDS_TOKENS`
/// goes.
fn holds(path: &str, text: &str, from: usize) -> String {
    let lines = numbered(text.lines().skip(from - 1).take(HOLDS_LINES), from);
    let (shown, taken) = fit(&lines, HOLDS_TOKENS);
    let notes = match text.lines().nth(from - 1 + taken).is_some() {
        true => vec![format!("... more lines; view from start {}", from + taken)],
        false => vec![],
    };
    with_notes(format!("'{path}' holds now:\n{shown}"), &notes)
}

/// `prepare_write`'s path checks, then the match: a refused edit never reaches the y/n.
fn prepare_edit(args: &Value, ctx: &ToolCtx) -> Result<Value, ToolError> {
    let args = prepare_write(args, ctx)?;
    edited(&args, ctx)?;
    Ok(args)
}

fn edit(args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
    let path = str_arg(args, "path")?;
    let edited = edited(args, ctx)?;
    let done = format!("replaced 1 match in {path} at line {}", edited.3);
    write_checked(path, edited, done)
}

/// Writes an edit's text, then undoes it if it makes the file fail a check it passed, with the
/// checker's words; `done` is the result when it stays.
fn write_checked(path: &str, edited: Edited, done: String) -> Result<String, ToolError> {
    let (real, before, after, _) = edited;
    let io = |e| ToolError::Io(path.into(), e);
    let was_failing = check(&real, &before).map_err(io)?.is_some();
    crate::write_atomic(&real, &after, true).map_err(io)?;
    let failed = match check(&real, &after) {
        Ok(failed) => failed.filter(|_| !was_failing),
        // The edit must not stay unchecked: undo it, then report why.
        Err(e) => Some(format!("the check could not run: {e}")),
    };
    let Some(error) = failed else {
        return Ok(done);
    };
    crate::write_atomic(&real, &before, true).map_err(io)?;
    Err(ToolError::Invalid(format!(
        "the edit was rolled back: it makes '{path}' fail its check:\n{}\n{}",
        // A local write's result is never cut (`tool::run`), so the checker's words are cut here.
        truncate(error.trim_end().into(), CHECK_TOKENS),
        around(&real, &after, &error)
    )))
}

/// The 10 numbered lines of `text` around the line `error` names, or none when it names none.
fn around(real: &Path, text: &str, error: &str) -> String {
    let name = real
        .file_name()
        .map(OsStr::to_string_lossy)
        .unwrap_or_default();
    let Some(n) = error_line(&name, error) else {
        return String::new();
    };
    let first = n.saturating_sub(5).max(1);
    // Each line cut as search cuts a hit: a minified file's one line would be the whole file.
    let lines = (text.lines().skip(first - 1).take(10))
        .map(|line| &line[..line.floor_char_boundary(HIT_BYTES)]);
    numbered(lines, first).join("\n")
}

/// The line a checker's `error` names in the file called `name`: Python's `line N`, serde's
/// `at line N`, node's `path:N`, rustfmt's `<stdin>:N:col`; None when it names none.
pub(crate) fn error_line(name: &str, error: &str) -> Option<usize> {
    let at = format!(r"line (\d+)|(?:{}|<stdin>):(\d+)", regex::escape(name));
    let caps = Regex::new(&at).ok()?.captures(error)?;
    caps.get(1).or(caps.get(2))?.as_str().parse().ok()
}

/// Why `text`, the file at `real`, fails its extension's check, or `None`: it passes, its
/// extension has no check, or the checker is not installed.
fn check(real: &Path, text: &str) -> io::Result<Option<String>> {
    let ext = real.extension().and_then(OsStr::to_str).unwrap_or_default();
    if ext == "json" {
        return Ok(serde_json::from_str::<Value>(text)
            .err()
            .map(|e| e.to_string()));
    }
    let programs: &[&str] = match ext {
        "py" => &["python3", "python"],
        "rs" => &["rustfmt"],
        "js" | "mjs" => &["node"],
        _ => return Ok(None),
    };
    // A folder of its own, so no checker reads the user's folder as its working directory, and
    // Python leaves its .pyc there.
    let tmp = tempfile::tempdir()?;
    let prefix = format!("pycache_prefix={}", tmp.path().display());
    // Python puts a `\\?\` path under the prefix as it is, which no file can be named: every
    // file would fail its check, so none would ever be checked.
    let shown = plain(real.into());
    let file = shown.as_os_str();
    let edition = &rust_edition(real);
    let args: Vec<&OsStr> = match ext {
        // -I: neither the folder nor PYTHON* variables put a module of theirs before py_compile.
        "py" => ["-I", "-X", &prefix, "-m", "py_compile"]
            .map(OsStr::new)
            .into_iter()
            .chain([file])
            .collect(),
        // The text on stdin: a path would make rustfmt load the files `mod x;` names too.
        "rs" => ["--check", "--edition", edition].map(OsStr::new).into(),
        _ => vec![OsStr::new("--check"), file],
    };
    for program in programs {
        let child = crate::tool::no_window(&mut Command::new(program))
            .args(&args)
            .current_dir(tmp.path())
            .stdin(if ext == "rs" {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match child {
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            child => child?,
        };
        if let Some(mut stdin) = child.stdin.take() {
            // rustfmt reads all of stdin before it writes, so this cannot block on its output.
            stdin.write_all(text.as_bytes())?;
        }
        let out = child.wait_with_output()?;
        // A failure with nothing on stderr is formatting only: rustfmt --check's diff is stdout.
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        // 9009 is Windows' "not found": `python3` there can be the Store's alias, which says
        // "Python was not found" and runs no Python, so its words are no verdict on the file.
        if out.status.code() == Some(9009) {
            continue;
        }
        return Ok((!out.status.success() && !stderr.trim().is_empty()).then_some(stderr));
    }
    Ok(None)
}

/// The edition of the crate `real` is in: no single one parses every crate, since 2024 reserves
/// `gen` (`rng.gen()`) and 2021 refuses let chains. The nearest Cargo.toml's `edition = "..."`,
/// past manifests that inherit it (`edition.workspace = true` or `edition = { workspace = true }`);
/// a package without one is 2015, as Cargo reads it.
// ponytail: a line match, not a TOML parse; parse when a manifest writes edition another way
fn rust_edition(real: &Path) -> String {
    let edition = Regex::new(r#"(?m)^\s*edition\s*=\s*"(\d{4})""#).expect("a valid regex");
    let inherits = Regex::new(r"(?m)^\s*edition\s*(?:\.\s*workspace|=\s*\{[^}]*workspace)")
        .expect("a valid regex");
    for dir in real.ancestors().skip(1) {
        let Ok(manifest) = fs::read_to_string(dir.join("Cargo.toml")) else {
            continue;
        };
        if let Some(caps) = edition.captures(&manifest) {
            return caps[1].into();
        }
        if manifest.contains("[package]") && !inherits.is_match(&manifest) {
            return "2015".into();
        }
    }
    // A file in no crate, such as a script: the edition `cargo new` gives.
    "2024".into()
}

/// A command's stdout and stderr as one text, and how it ended as one line.
// Only the sandbox makes one.
#[cfg_attr(
    not(any(target_os = "linux", target_os = "macos", windows)),
    allow(dead_code)
)]
struct Ran {
    text: String,
    exit: Option<i32>,
    status: String,
    /// Killed at its timeout or by the user's stop: it did not end on its own.
    cut: bool,
}

/// The folders at a project's top a read-only run's commands may still write, where they exist
/// already: build output, caches and virtual environments, none of them the user's code.
// ponytail: top-level names only. ceiling: a build folder nested in a package (a monorepo's
// packages/x/target) stays read-only, so that build fails and says so. upgrade: the folders the
// project's .gitignore ignores, if a run shows one.
pub(crate) const BUILD_DIRS: &[&str] = &[
    "target",
    "build",
    "dist",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    ".gradle",
    ".next",
    "_build",
    "cmake-build-debug",
    "cmake-build-release",
    ".zig-cache",
    "zig-cache",
];

/// Runs `cmd` in `worktree` through `sandbox::run`, the only way a tool runs a command: no
/// network, writes only inside the worktree, killed after `timeout_s`, or once `stop` is set.
/// A `read_only` run's command (`CodeTask::read_only`) writes only the worktree's `BUILD_DIRS`
/// and its private tmp, where a cargo build goes when the project has no target/ yet.
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn sandboxed(
    cmd: &str,
    worktree: &Path,
    timeout_s: u64,
    stop: Option<&Arc<AtomicBool>>,
    read_only: bool,
) -> Result<Ran, ToolError> {
    use crate::sandbox::{Limits, run};
    let limits = Limits {
        timeout: std::time::Duration::from_secs(timeout_s),
        stop: stop.cloned(),
        writable: read_only.then(|| BUILD_DIRS.iter().map(PathBuf::from).collect()),
        ..Limits::default()
    };
    // stderr into stdout, so the two read in the order they were written; no .pyc beside the
    // sources, which `CodeTask::packet` would commit with everything else.
    // ponytail: cargo's target/ lands in the worktree too, and is committed where .gitignore
    // lacks /target; point CARGO_TARGET_DIR at the private tmp when such a repo shows up
    // Pipefail too, so `pytest | tail` fails with pytest: in sh when it has it, else in bash, which
    // runs the command quoted; a shell with neither runs it without.
    #[cfg(unix)]
    let target = match read_only {
        true => "[ -d target ] || export CARGO_TARGET_DIR=\"$TMPDIR/target\"\n",
        false => "",
    };
    #[cfg(unix)]
    let script = format!(
        "exec 2>&1\nexport PYTHONDONTWRITEBYTECODE=1\n{target}\
         if (set -o pipefail) 2>/dev/null; then set -o pipefail\n\
         elif command -v bash >/dev/null; then exec bash -o pipefail -c '{}'\nfi\n{cmd}",
        cmd.replace('\'', r"'\''")
    );
    // ponytail: cmd.exe groups only what parses inside ( ): an unbalanced ) or a second line in
    // `cmd` breaks it. upgrade: a script file in the private tmp if a model's commands do that.
    // cmd.exe ends with 1 after a command it does not have, though its errorlevel is 9009, as
    // run_tests needs to read it: the bare `exit` ends with the errorlevel, which a trailing
    // builtin (`echo`) does not reset, so a failure before it is no longer hidden either.
    // ponytail: cmd.exe has no pipefail, so a pipe's status is its last command's; a PowerShell
    // shell if a Windows run hides a failure behind one.
    #[cfg(windows)]
    let target = match read_only {
        true => r"(if not exist target\ set CARGO_TARGET_DIR=%TMP%\target)& ",
        false => "",
    };
    #[cfg(windows)]
    let script = format!(
        "set PYTHONDONTWRITEBYTECODE=1& {target}({cmd}) 2>&1& \
         (if errorlevel 9009 if not errorlevel 9010 exit 9009)& exit"
    );
    let io = |e| ToolError::Io(worktree.join(".git").display().to_string(), e);
    let guard = git_guard(worktree).map_err(io)?;
    let out = run(&script, worktree, &limits).map_err(|e| ToolError::Invalid(e.to_string()));
    // Every process of the command is gone by now, so nothing writes the files put back.
    let put_back = guard
        .map(|before| git_restore(worktree, before))
        .transpose()
        .map_err(io)?;
    let out = out?;
    let mut status = match out.exit {
        _ if out.stopped => "killed: the user stopped the run".into(),
        _ if out.timed_out => format!("killed after {timeout_s} s, its timeout"),
        Some(code) => format!("exit code {code}"),
        // RLIMIT_CPU's SIGXCPU, or the kernel's SIGKILL at the memory cap.
        None => "killed by a signal".into(),
    };
    // The cap or a pipe held open past the end: `Output::truncated` does not say which.
    if out.truncated {
        status += "; the output was cut";
    }
    let mut text =
        (String::from_utf8_lossy(&out.stdout) + String::from_utf8_lossy(&out.stderr)).into_owned();
    if let Some(paths) = put_back.filter(|p| !p.is_empty()) {
        let paths = paths.join(", ");
        text += &format!(
            "\nthe command changed what git runs or reads as config outside the sandbox, so \
             it was put back as it was: {paths}"
        );
    }
    Ok(Ran {
        text,
        exit: out.exit,
        status,
        cut: out.timed_out || out.stopped,
    })
}

/// A file of `.git` as `git_guard` found it: its bytes, or, for a link, what it names. A link is
/// never read or written through: the command may point it outside the sandbox.
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
#[derive(Debug, PartialEq)]
enum Kept {
    File(Vec<u8>),
    Link(PathBuf),
}

/// The files of `dir/.git` git runs or reads as its config, and their bytes, and every link
/// there: in a folder done in place a command can write its repository, and git run outside the
/// sandbox later would run what it planted (a hook, a `core.fsmonitor`, a link to either).
/// Empty when `.git` is missing, so one the command makes is seen; None when `.git` is a
/// worktree's file, naming a folder the sandbox cannot write.
// ponytail: a repository nested deeper than `dir/.git` (not a submodule) is not guarded; walk for
// them if a project vendors one
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn git_guard(dir: &Path) -> io::Result<Option<BTreeMap<PathBuf, Kept>>> {
    let git = dir.join(".git");
    match fs::symlink_metadata(&git) {
        Ok(meta) if meta.is_dir() => {}
        Ok(_) => return Ok(None),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Some(BTreeMap::new())),
        Err(e) => return Err(e),
    }
    let mut kept = BTreeMap::new();
    let mut dirs = vec![git.clone()];
    while let Some(at) = dirs.pop() {
        for entry in fs::read_dir(&at)? {
            let entry = entry?;
            let path = entry.path();
            let rel = path.strip_prefix(&git).expect("under .git");
            let name = entry.file_name();
            let kind = entry.file_type()?;
            // Objects, refs and logs are data git never runs; they are most of the bytes.
            if kind.is_dir() {
                if !matches!(name.to_str(), Some("objects" | "refs" | "logs" | "lfs")) {
                    dirs.push(path);
                }
                continue;
            }
            let hook = rel.components().any(|c| c.as_os_str() == "hooks");
            let config = matches!(name.to_str(), Some("config" | "config.worktree"));
            if kind.is_symlink() {
                kept.insert(rel.to_path_buf(), Kept::Link(fs::read_link(&path)?));
            } else if hook || config || rel == Path::new("info").join("attributes") {
                kept.insert(rel.to_path_buf(), Kept::File(fs::read(&path)?));
            }
        }
    }
    Ok(Some(kept))
}

/// Puts `dir/.git` back as `git_guard` found it: each file or link it changed or made goes,
/// then each one kept comes back; a whole `.git` made where there was none goes, and a file or
/// link made where `.git` was a folder goes for a folder of the kept files. The paths put back,
/// from `dir`.
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
fn git_restore(dir: &Path, before: BTreeMap<PathBuf, Kept>) -> io::Result<Vec<String>> {
    #[cfg(unix)]
    use std::os::unix::fs::symlink as link;
    #[cfg(windows)]
    use std::os::windows::fs::symlink_file as link;
    let git = dir.join(".git");
    let shown = |rel: &Path| Path::new(".git").join(rel).display().to_string();
    let now = match git_guard(dir)? {
        Some(now) if !before.is_empty() => now,
        // No `.git` before, or a file or link where its folder was: what is there goes.
        _ => {
            match fs::symlink_metadata(&git) {
                Ok(meta) if meta.is_dir() => fs::remove_dir_all(&git)?,
                Ok(_) => fs::remove_file(&git)?,
                Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(vec![]),
                Err(e) => return Err(e),
            }
            if before.is_empty() {
                return Ok(vec![".git".into()]);
            }
            BTreeMap::new()
        }
    };
    let mut put = Vec::new();
    for (rel, is) in &now {
        if before.get(rel) != Some(is) {
            let path = git.join(rel);
            // A link to a folder is a directory to Windows.
            fs::remove_file(&path).or_else(|_| fs::remove_dir(&path))?;
            put.push(shown(rel));
        }
    }
    for (rel, was) in &before {
        if now.get(rel) != Some(was) {
            let path = git.join(rel);
            fs::create_dir_all(path.parent().expect("under .git"))?;
            match was {
                Kept::File(bytes) => fs::write(&path, bytes)?,
                Kept::Link(to) => link(to, &path)?,
            }
            if !now.contains_key(rel) {
                put.push(shown(rel));
            }
        }
    }
    Ok(put)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn sandboxed(
    _: &str,
    _: &Path,
    _: u64,
    _: Option<&Arc<AtomicBool>>,
    _: bool,
) -> Result<Ran, ToolError> {
    Err(ToolError::Invalid(
        "commands run only in the sandbox, which this OS has none of".into(),
    ))
}

/// `bash`: one shell command in the first allowed folder. Its results are untrusted: the
/// sandbox lets it read any file the user can, inside the allowed folders or not.
/// `stop` ends its running command: the user's Stop of the code run it is in. Its output is
/// fitted to `tokens`.
struct Bash {
    stop: Option<Arc<AtomicBool>>,
    tokens: u32,
    read_only: bool,
}

/// Code mode's bash, for each attempt's registry (`Registry::replace`): the chat's, whose
/// command a Stop of the run (`stop`) ends, its output fitted to a view's room (`tokens`,
/// `Room::view_tokens`) up to `BASH_TOKENS`, so a `cat` of a whole file cannot fill a small
/// context. A `read_only` run's (`CodeTask::read_only`) writes only build output (`sandboxed`).
pub(crate) fn bash(stop: Option<Arc<AtomicBool>>, tokens: u32, read_only: bool) -> Box<dyn Tool> {
    Box::new(Bash {
        stop,
        tokens: tokens.min(BASH_TOKENS),
        read_only,
    })
}

impl Tool for Bash {
    fn spec(&self) -> ToolSpec {
        let writes = match self.read_only {
            true => {
                "the folder's build output folders that exist (such as target or \
                     node_modules) and $TMPDIR"
            }
            false => "inside the folder",
        };
        ToolSpec {
            name: "bash".into(),
            // The sandbox runs cmd.exe on Windows: told sh, the model writes printf and $VARs.
            description: format!(
                "Runs one shell command with {} in the folder, without network; it can read any \
                 file but write only {writes}, and is killed after timeout seconds (120 when \
                 left out). Returns its exit code and output.",
                if cfg!(windows) {
                    "cmd.exe (Windows, not sh)"
                } else {
                    "sh"
                }
            ),
            schema: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1,
                                "description": "Shell command"},
                    "timeout": {"type": "integer", "minimum": 1, "maximum": TIMEOUT_S,
                                "description": "Seconds before it is killed"}
                },
                "required": ["command"],
                "additionalProperties": false
            }),
            permission: Permission::WriteLocal,
            untrusted_results: true,
            max_result_tokens: self.tokens,
            // A command with a here-document of a few lines.
            max_args_tokens: 1024,
        }
    }

    fn call(&self, args: &Value, ctx: &ToolCtx) -> Result<String, ToolError> {
        let first = (ctx.allowed.first()).ok_or_else(|| ToolError::Denied(".".into()))?;
        let timeout = whole_arg(args, "timeout").unwrap_or(TIMEOUT_S);
        let folder = first_folder(first)?;
        let ran = sandboxed(
            str_arg(args, "command")?,
            &folder,
            timeout,
            self.stop.as_ref(),
            self.read_only,
        )?;
        // A local write's result is never cut (`tool::run`), so the output is fitted here.
        Ok(fitted(&ran.status, &ran.text, self.tokens))
    }
}

/// A command's result: its `status` line, then its output `text` fitted to `tokens`, its start
/// and end kept, with a line on how much was cut.
fn fitted(status: &str, text: &str, tokens: u32) -> String {
    let lines = text.lines().count();
    let note = |shown: usize| {
        format!(
            "\n[output cut at {}: {shown} of {lines} lines shown; view a file, or grep or sed -n a part]",
            crate::skills::read_papers::LEFT_OUT,
        )
    };
    // The status and the longest note come out of the cap too, so the whole result fits it.
    let reserve = (status.len() + 1 + note(lines).len())
        .div_ceil(crate::skills::read_papers::TOKEN_BYTES) as u32;
    let (shown, cut) = crate::skills::read_papers::fit(text, tokens.saturating_sub(reserve));
    let note = match cut {
        true => note(shown.lines().count().saturating_sub(1)),
        false => String::new(),
    };
    format!("{status}\n{}{note}", shown.trim_end())
        .trim_end()
        .into()
}

/// Runs `cmd`, a task's test command, in `worktree` until it ends or `stop` is set: the run, and
/// the text `run_tests` returns, whose first line says how the command ended; an error when the
/// shell has no such command. With `user`, the user's checkout beside the worktree, the worktree
/// goes first on Python's import path (`python_first`), and a run that still loaded a module
/// from `user` (`LOADED`) ran its code, not the worktree's (an editable install): `UNCOMPARED`,
/// never a pass. A `read_only` run's command writes only build output (`sandboxed`).
// ponytail: a worktree under the user's checkout (a data folder inside it) reads as the user's
// code, so each run there is uncompared; skip the worktree's own paths when a user has one
pub(crate) fn run_tests(
    cmd: &str,
    worktree: &Path,
    user: Option<&Path>,
    stop: Option<&Arc<AtomicBool>>,
    read_only: bool,
) -> Result<(TestRun, String), ToolError> {
    let wrapped = match user {
        Some(user) => python_first(cmd, worktree, &[user])
            .map_err(|e| ToolError::Io(worktree.join(SITE).display().to_string(), e))?,
        None => cmd.to_string(),
    };
    let ran = sandboxed(&wrapped, worktree, TIMEOUT_S, stop, read_only)?;
    // The shell found no such command (sh's 127, cmd.exe's 9009): no test ran, and none will
    // until the user installs it, so the run ends here rather than after its budget.
    if matches!(ran.exit, Some(127 | 9009)) {
        return Err(ToolError::Invalid(format!(
            "the tests cannot run: `{cmd}` names a command this computer does not have ({}): {}; \
             install it, or name the test command",
            ran.status,
            truncate(ran.text.trim().into(), NAMES_TOKENS)
        )));
    }
    let run = TestRun::parse(&ran.text, ran.exit);
    let note = match user.filter(|u| named(&ran.text, &[u]).is_some()) {
        Some(u) => format!(
            "{UNCOMPARED}: the run loaded modules from {}, the user's checkout, so it ran that code, \
             not the worktree's (an editable install?)",
            u.display()
        ),
        None => String::new(),
    };
    let out = report(&ran.status, &run, &note);
    Ok((run, out))
}

/// `run_tests`' text for `run`, which ended as `status` says: its counts, the failing tests'
/// names, `note`'s line when there is one, and the output's end.
fn report(status: &str, run: &TestRun, note: &str) -> String {
    let mut out = format!("{status}: {} passed, {} failed", run.passed, run.failed);
    // Before the names, which run to the blank line (`r#loop::verdict`).
    if !note.is_empty() {
        out = format!("{out}\n{note}");
    }
    if !run.failing.is_empty() {
        out += "\nfailing:\n";
        out += &truncate(run.failing.join("\n"), NAMES_TOKENS);
    }
    format!("{out}\n\n{}", run.tail)
}

/// The note `run_tests`' text gives failures it could not compare with the base's: they may be
/// the base's own, so they neither pass nor block a done (`uncompared`).
pub(crate) const UNCOMPARED: &str = "could not compare with the base";

/// Whether `shown`, `run_tests`' text, says its failures could not be compared (`UNCOMPARED`).
pub(crate) fn uncompared(shown: &str) -> bool {
    noted(shown, UNCOMPARED)
}

/// The note `run_tests`' text gives failures the base has too (`TestRun::shared`), before their
/// names: not the change's, and never a pass.
pub(crate) const AS_BEFORE: &str = "still failing, as before your change:";

/// Whether `shown`, `run_tests`' text, names failures the base has too (`AS_BEFORE`).
pub(crate) fn as_before(shown: &str) -> bool {
    noted(shown, AS_BEFORE)
}

/// Whether a line of `shown`'s head starts with `note`: its lines before the output's end only,
/// so the output cannot say it for the run.
fn noted(shown: &str, note: &str) -> bool {
    let head = shown.split("\n\n").next().unwrap_or_default();
    head.lines().any(|l| l.starts_with(note))
}

/// `after`, the tests after a change, whose text is `shown`, against `base`, the same tests with
/// the change's sources as the base has them, or why there is no such run. Three outcomes: every
/// failure new, kept as it is; some the base has too by name, which move to `TestRun::shared`
/// with the `AS_BEFORE` note; or no comparison (no base run, one that names no failing test, or
/// an `after` that names none while the base fails), kept whole with the `UNCOMPARED` note.
pub(crate) fn against(
    base: Result<TestRun, String>,
    mut after: TestRun,
    shown: &str,
) -> (TestRun, String) {
    let first = shown.lines().next().unwrap_or_default();
    let status = first.rsplit_once(": ").map_or(first, |(s, _)| s);
    let why = match &base {
        Err(why) => why.clone(),
        Ok(b) if b.none_ran() => "no test ran there".into(),
        Ok(b) if b.failed > 0 && b.failing.is_empty() => {
            "it fails there without naming a test (a crash, or a suite that never started)".into()
        }
        Ok(b) if b.failed > 0 && after.failing.is_empty() => {
            "this run names no failing test".into()
        }
        Ok(base) => {
            let (old, new): (Vec<String>, _) =
                (after.failing.iter().cloned()).partition(|n| base.failing.contains(n));
            if old.is_empty() {
                return (after, shown.into());
            }
            (after.shared, after.failing) = (old, new);
            after.failed = after.failed.saturating_sub(after.shared.len() as u32);
            let names = truncate(after.shared.join(", "), NAMES_TOKENS);
            let out = report(status, &after, &format!("{AS_BEFORE} {names}"));
            return (after, out);
        }
    };
    let note = format!("{UNCOMPARED}: {why}; these failures may be the base's own");
    let out = report(status, &after, &note);
    (after, out)
}

/// Runs `cmd`, a build check (`CodeTask::check_cmds`), in `worktree`, bounded as any command is:
/// whether it passed, None when it could not say (killed at a time limit, ours or `timeout`'s
/// 124, a signal, a command this computer lacks, or no sandbox to run it in), and its status
/// line and output cut to fit, or why it did not run; and the files it says failed
/// (`FAILED_IMPORT`), read from the whole output, so none is lost in the cut. A `read_only`
/// run's check writes only build output (`sandboxed`).
pub(crate) fn build_check(
    cmd: &str,
    worktree: &Path,
    stop: Option<&Arc<AtomicBool>>,
    read_only: bool,
) -> Result<Judged, ToolError> {
    match sandboxed(cmd, worktree, TIMEOUT_S, stop, read_only) {
        Ok(ran) => Ok(judged(ran)),
        Err(e @ ToolError::Invalid(_)) => Ok((None, e.to_string(), Vec::new())),
        Err(e) => Err(e),
    }
}

/// A build check's result (`build_check`): whether it passed, its text, the files it failed.
pub(crate) type Judged = (Option<bool>, String, Vec<String>);

/// `build_check`'s result of `ran`.
fn judged(ran: Ran) -> Judged {
    let passed = match ran.exit {
        _ if ran.cut => None,
        Some(0) => Some(true),
        Some(124 | 127 | 9009) | None => None,
        Some(_) => Some(false),
    };
    let failed = (ran.text.lines())
        .filter_map(|l| l.strip_prefix(FAILED_IMPORT))
        .map(Into::into)
        .collect();
    let (shown, _) = crate::skills::read_papers::fit(&ran.text, BASH_TOKENS);
    let shown = format!("{}\n{shown}", ran.status).trim_end().into();
    (passed, shown, failed)
}

/// Python's `sitecustomize` for a base run (`on_base`): at its end it names each module it
/// loaded from `FOLDERS`, the folders the change is in, which a test's own traceback rarely does.
/// An installed package is never the project's code, even in a virtualenv inside the folder.
const LOADED: &str = "import atexit, os, sys\nFOLDERS = {folders}\nSKIP = tuple(os.path.join(p, '') \
for p in {sys.prefix, sys.base_prefix, sys.exec_prefix})\ndef named():\n for m in \
list(sys.modules.values()):\n  f = getattr(m, '__file__', None) or ''\n  if f.startswith(FOLDERS) \
and not f.startswith(SKIP) and 'site-packages' not in f and 'dist-packages' not in f: \
print('loaded ' + f, file=sys.stderr)\natexit.register(named)\n";

/// The folder of a run's `sitecustomize` (`python_first`), inside the folder it runs in.
const SITE: &str = ".larch-site";

/// `cmd` with `dir/SITE`, `dir` and `dir/src` first on Python's import path, ahead of an
/// editable install's, and `LOADED` in that `SITE` naming each module loaded from `folders`.
/// `SITE` ignores itself, so git never sees it in the attempt's diff.
fn python_first(cmd: &str, dir: &Path, folders: &[&Path]) -> io::Result<String> {
    let site = dir.join(SITE);
    let named: Vec<String> = folders.iter().map(inside).collect();
    let loaded = LOADED.replace("{folders}", &format!("tuple({})", json!(named)));
    fs::create_dir_all(&site)?;
    fs::write(site.join(".gitignore"), "*\n")?;
    fs::write(site.join("sitecustomize.py"), loaded)?;
    let path = [site, dir.into(), dir.join("src")].map(|p| p.display().to_string());
    #[cfg(unix)]
    let cmd = {
        let quoted = format!("'{}'", path.join(":").replace('\'', r"'\''"));
        format!("export PYTHONPATH={quoted}\n{cmd}")
    };
    #[cfg(windows)]
    let cmd = format!("set \"PYTHONPATH={}\"& {cmd}", path.join(";"));
    Ok(cmd)
}

/// `folder` as a prefix of the paths inside it.
fn inside(folder: &&Path) -> String {
    format!("{}{}", folder.display(), std::path::MAIN_SEPARATOR)
}

/// The first of `folders` that a module `LOADED` names in `text`, a run's output, is in.
fn named<'a>(text: &str, folders: &[&'a Path]) -> Option<&'a Path> {
    let loaded: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("loaded "))
        .collect();
    (folders.iter())
        .find(|f| loaded.iter().any(|l| l.starts_with(&inside(f))))
        .copied()
}

/// Runs `cmd` in `checkout`, a checkout of the base (`CodeTask::at_base`), with it and its
/// `src/` first on Python's import path, ahead of an editable install's (`python_first`): the
/// run, or why it cannot stand for the base. That is a run that did not end on its own or could not start, and
/// one that loaded a Python module from one of `folders`, where the change is, as `LOADED` names
/// it: it ran the change's code (an editable install, a .pth file).
// ponytail: LOADED hides a sitecustomize of the project's own; chain to it when one shows up
fn on_base(
    cmd: &str,
    checkout: &Path,
    folders: &[&Path],
    stop: Option<&Arc<AtomicBool>>,
) -> Result<Ran, String> {
    let cmd = python_first(cmd, checkout, folders)
        .map_err(|e| format!("cannot write the base run's sitecustomize: {e}"))?;
    // The base's own checkout, made for this run: the user's folder is not in it.
    let ran = sandboxed(&cmd, checkout, TIMEOUT_S, stop, false).map_err(|e| e.to_string())?;
    if ran.cut || matches!(ran.exit, Some(127 | 9009)) {
        return Err(format!(
            "the base run did not end on its own: {}",
            ran.status
        ));
    }
    match named(&ran.text, folders) {
        Some(f) => Err(format!(
            "the base run loaded modules from {}, where the change is, so it ran the change's code \
             (an editable install?)",
            f.display()
        )),
        None => Ok(ran),
    }
}

/// `run_tests`' run of `cmd` on the base (`on_base`), or why there is none.
pub(crate) fn base_tests(
    cmd: &str,
    checkout: &Path,
    folders: &[&Path],
    stop: Option<&Arc<AtomicBool>>,
) -> Result<TestRun, String> {
    on_base(cmd, checkout, folders, stop).map(|ran| TestRun::parse(&ran.text, ran.exit))
}

/// `build_check` of `cmd` on the base (`on_base`), or why there is none.
pub(crate) fn base_check(
    cmd: &str,
    checkout: &Path,
    folders: &[&Path],
    stop: Option<&Arc<AtomicBool>>,
) -> Result<Judged, String> {
    on_base(cmd, checkout, folders, stop).map(judged)
}

/// `run_tests`: a code task's tests, in the worktree of the attempt it was made for
/// (`CodeTask::run_tests`).
pub(crate) struct RunTests(pub CodeTask, pub Option<Arc<AtomicBool>>);

impl Tool for RunTests {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "run_tests".into(),
            description: "Runs the task's tests in its worktree, without network; returns how \
                          many passed and failed, the failing tests' names and the output's end. \
                          `tests` picks which (test files or the runner's labels); else those \
                          near the change."
                .into(),
            schema: json!({
                "type": "object",
                "properties": {"tests": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "string"},
                    ],
                }},
                "additionalProperties": false,
            }),
            permission: Permission::WriteLocal,
            untrusted_results: false,
            max_result_tokens: RUN_TOKENS,
            max_args_tokens: ARGS_TOKENS,
        }
    }

    fn call(&self, args: &Value, _: &ToolCtx) -> Result<String, ToolError> {
        let tests = labels(args)?;
        self.0
            .test_these(self.1.as_ref(), true, &tests)
            .map(|(_, shown)| shown)
    }
}

/// The tests a run_tests call names in `args` (one string, or a list), each checked to be a
/// label in the worktree that the shell reads as one word: none for none named.
pub(crate) fn labels(args: &Value) -> Result<Vec<String>, ToolError> {
    let tests = match &args["tests"] {
        Value::Null => Vec::new(),
        Value::String(t) => vec![t.clone()],
        Value::Array(all) => (all.iter())
            .map(|t| t.as_str().map(String::from))
            .collect::<Option<_>>()
            .ok_or_else(|| ToolError::Invalid("tests must be strings".into()))?,
        _ => {
            return Err(ToolError::Invalid(
                "tests must be a string or a list of them".into(),
            ));
        }
    };
    for t in &tests {
        let word = |c: char| c.is_ascii_alphanumeric() || "/._-:[],=+@".contains(c);
        let why = if t.is_empty() || !t.chars().all(word) {
            "it may hold only letters, digits and / . _ - : [ ] , = + @"
        } else if t.starts_with('/') || t.as_bytes().get(1) == Some(&b':') {
            "it is an absolute path; name it from the folder's top"
        } else if t.split('/').any(|part| part == "..") {
            "it leaves the folder (`..`)"
        } else if t.starts_with('-') {
            "it is an option, not a test"
        } else {
            continue;
        };
        return Err(ToolError::Invalid(format!(
            "the test '{t}' is refused: {why}"
        )));
    }
    Ok(tests)
}

/// Passed and failed counts, and the failing tests' names.
type Counts = (u32, u32, Vec<String>);

impl TestRun {
    /// `output` read as cargo test, pytest, jest, unittest or node --test print it, and its end.
    /// Output none of them prints counts as one test, which passed when `exit` is 0.
    pub fn parse(output: &str, exit: Option<i32>) -> TestRun {
        let (passed, mut failed, mut failing) = (cargo_tests(output))
            .or_else(|| pytest(output))
            .or_else(|| jest(output))
            .or_else(|| unittest(output))
            .or_else(|| node(output))
            .unwrap_or(((exit == Some(0)).into(), 0, vec![]));
        // A failed run with no failure counted (a crash, a suite that never ran, a summary the
        // timeout cut off) is one failure, so it never reads as a pass.
        if exit != Some(0) && failed == 0 {
            failed = 1;
        }
        // jest lists each failure again under "Summary of all failing tests".
        let mut seen = HashSet::new();
        failing.retain(|name| seen.insert(name.clone()));
        let tail = &output[output.ceil_char_boundary(output.len().saturating_sub(TAIL_BYTES))..];
        TestRun {
            passed,
            failed,
            failing,
            shared: Vec::new(),
            tail: tail.into(),
        }
    }
}

/// A count in test output; one past `u32` saturates rather than reads as none.
fn count(digits: &str) -> u32 {
    digits.parse().unwrap_or(u32::MAX)
}

/// The passed and failed counts in a list like `2 failed, 4 passed, 1 skipped`; `fails` are the
/// words that count as failures.
fn tally(list: &str, fails: &[&str]) -> (u32, u32) {
    let (mut passed, mut failed) = (0u32, 0u32);
    for (n, word) in list.split(", ").filter_map(|part| part.split_once(' ')) {
        if word == "passed" {
            passed = passed.saturating_add(count(n));
        } else if fails.contains(&word) {
            failed = failed.saturating_add(count(n));
        }
    }
    (passed, failed)
}

/// cargo test: each test binary's `test result:` line, summed, and the names the `failures:`
/// block right above it lists. The patterns read CRLF too (`R`), as a Windows run prints it.
fn cargo_tests(out: &str) -> Option<Counts> {
    let result =
        Regex::new(r"(?m)^test result: \w+\. (\d+) passed; (\d+) failed").expect("a valid regex");
    let mut sums: Option<(u32, u32)> = None;
    for caps in result.captures_iter(out) {
        let (passed, failed) = sums.get_or_insert((0, 0));
        *passed = passed.saturating_add(count(&caps[1]));
        *failed = failed.saturating_add(count(&caps[2]));
    }
    let (passed, failed) = sums?;
    // Only that block: a test's captured output may print a `failures:` line of its own.
    let listed = Regex::new(r"(?mR)^failures:\r?\n((?:    .*\r?\n)+)\r?\ntest result: ")
        .expect("a valid regex");
    let failing = (listed.captures_iter(out))
        .flat_map(|caps| caps.get(1).map_or("", |m| m.as_str()).lines())
        .map(|name| name.trim_start().into())
        .collect();
    Some((passed, failed, failing))
}

/// pytest: its summary line, between `=` rules or, under -q, the output's last line, errors
/// counted as failures; and the FAILED and ERROR names of its short test summary.
fn pytest(out: &str) -> Option<Counts> {
    // `2 failed, 4 passed, 1 error in 0.03s`, `(0:01:02)` after a long run, `seconds` before
    // pytest 6.
    const COUNTS: &str = r"((?:\d+ \w+, )*\d+ \w+) in [\d.]+(?:s| seconds)(?: \([\d:]+\))?";
    let ruled = Regex::new(&format!(r"(?mR)^=+ {COUNTS} =+$")).expect("a valid regex");
    let quiet = Regex::new(&format!("^{COUNTS}$")).expect("a valid regex");
    let last = (out.lines().rev())
        .find(|line| !line.trim().is_empty())
        .unwrap_or_default();
    let caps = (ruled.captures_iter(out).last()).or_else(|| quiet.captures(last))?;
    let (passed, failed) = tally(&caps[1], &["failed", "error", "errors"]);
    // Below the short summary's rule only: a captured log line starts with ERROR too.
    let summary = out
        .rfind("short test summary info")
        .map_or("", |i| &out[i..]);
    let named = Regex::new(r"(?mR)^(?:FAILED|ERROR) (.+?)(?: - .*)?$").expect("a valid regex");
    let failing = named.captures_iter(summary).map(|c| c[1].into()).collect();
    Some((passed, failed, failing))
}

/// jest: its last `Tests:` line, and the `●` title of each failure.
fn jest(out: &str) -> Option<Counts> {
    let tests = Regex::new(r"(?m)^Tests:\s+(?:(.+), )?\d+ total").expect("a valid regex");
    let caps = tests.captures_iter(out).last()?;
    let (passed, failed) = tally(caps.get(1).map_or("", |m| m.as_str()), &["failed"]);
    let titles = Regex::new(r"(?mR)^  ● (.+)$").expect("a valid regex");
    let failing = titles.captures_iter(out).map(|c| c[1].into()).collect();
    Some((passed, failed, failing))
}

/// Python's unittest: its last `Ran N tests` line and the `OK` or `FAILED (failures=1, ...)`
/// under it, errors and unexpected successes counted as failures, skips as neither; and the
/// title under each `===` rule, `FAIL: test_x (module.Class.test_x)`.
fn unittest(out: &str) -> Option<Counts> {
    let ran = Regex::new(r"(?mR)^Ran (\d+) tests? in [\d.]+s\s+^(?:OK|FAILED)(?: \(([^)]*)\))?$")
        .expect("a valid regex");
    let caps = ran.captures_iter(out).last()?;
    let (mut failed, mut neither) = (0u32, 0u32);
    for (key, n) in
        (caps.get(2).map_or("", |m| m.as_str()).split(", ")).filter_map(|part| part.split_once('='))
    {
        match key {
            "failures" | "errors" | "unexpected successes" => {
                failed = failed.saturating_add(count(n))
            }
            _ => neither = neither.saturating_add(count(n)),
        }
    }
    let passed = count(&caps[1])
        .saturating_sub(failed)
        .saturating_sub(neither);
    let titles = Regex::new(r"(?mR)^=+\r?\n(?:FAIL|ERROR): (.+)$").expect("a valid regex");
    let failing = titles.captures_iter(out).map(|c| c[1].into()).collect();
    Some((passed, failed, failing))
}

/// node --test: its last `pass`, `fail` and `cancelled` counts, after `# ` in TAP or `ℹ ` in
/// the spec reporter, a cancelled test (one past its timeout) counted as failed; and each
/// failing title, TAP's `not ok 2 - title` or spec's `✖ title (2.5ms)`, nested ones too.
fn node(out: &str) -> Option<Counts> {
    let last = |key: &str| {
        let re = Regex::new(&format!(r"(?mR)^(?:#|ℹ) +{key} +(\d+)$")).expect("a valid regex");
        re.captures_iter(out).last().map(|c| count(&c[1]))
    };
    let (passed, failed) = (last("pass")?, last("fail")?);
    let failed = failed.saturating_add(last("cancelled").unwrap_or(0));
    let titles = Regex::new(r"(?mR)^\s*(?:not ok \d+ - (.+?)(?: # .*)?|✖ (.+?) \([\d.]+m?s\))$")
        .expect("a valid regex");
    let failing = (titles.captures_iter(out))
        .filter_map(|c| c.get(1).or(c.get(2)).map(|m| m.as_str().into()))
        .collect();
    Some((passed, failed, failing))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read-only run's command writes an existing build folder but no file of the project,
    /// and a cargo build of a project with no target/ yet goes to the private tmp.
    #[cfg(unix)]
    #[test]
    fn a_read_only_command_writes_only_build_output() {
        let dir = tempfile::tempdir().unwrap();
        let cmd = "echo x > mine.txt 2>/dev/null && echo wrote; echo \"$CARGO_TARGET_DIR\"";
        let ran = match sandboxed(cmd, dir.path(), 30, None, true) {
            Err(ToolError::Invalid(why)) if why.contains("unavailable") => {
                return println!("skipped: {why}");
            }
            ran => ran.unwrap(),
        };
        assert!(!dir.path().join("mine.txt").exists(), "{}", ran.text);
        assert!(
            ran.text.trim().ends_with("/target") && !ran.text.contains("wrote"),
            "{}",
            ran.text
        );
        fs::create_dir(dir.path().join("target")).unwrap();
        let cmd = "echo x > target/out && echo \"[$CARGO_TARGET_DIR]\"";
        let ran = sandboxed(cmd, dir.path(), 30, None, true).unwrap();
        assert_eq!(ran.text, "[]\n");
        // A change run writes the project as before.
        let ran = sandboxed("echo x > mine.txt", dir.path(), 30, None, false).unwrap();
        assert!(dir.path().join("mine.txt").exists(), "{}", ran.text);
    }

    /// Code mode's bash output is fitted to the room's view with a line on what was cut, so a
    /// `cat` of a whole file cannot fill a small context.
    #[test]
    fn a_long_output_is_fitted_and_says_what_it_cut() {
        let text: String = (1..=5000).map(|i| format!("{i}\n")).collect();
        let out = fitted("exit code 0", &text, 500);
        assert!(out.len() <= 500 * 4 + 200, "{}: {out}", out.len());
        assert!(out.starts_with("exit code 0\n1\n2\n"), "{out}");
        assert!(
            out.contains("\n4999\n5000\n[output cut at [...]: "),
            "{out}"
        );
        assert!(out.ends_with(" of 5000 lines shown; view a file, or grep or sed -n a part]"));
        assert_eq!(fitted("exit code 0", "1\n2\n", 500), "exit code 0\n1\n2");
    }

    /// run_tests takes one test or a list, from the folder's top, each one word to the shell;
    /// an absolute path, a `..`, an option or a shell metacharacter is refused.
    #[test]
    fn named_tests_are_labels_in_the_folder() {
        let ok =
            labels(&json!({"tests": ["tests/test_x.py::T::test_a", "migrations.test_writer"]}));
        assert_eq!(ok.unwrap().len(), 2);
        assert_eq!(
            labels(&json!({"tests": "tests/test_x.py"})).unwrap().len(),
            1
        );
        assert!(labels(&json!({})).unwrap().is_empty());
        for bad in [
            "/etc/passwd",
            "C:x",
            "../x.py",
            "a/../../b",
            "-p evil",
            "x; rm -rf .",
            "$(id)",
            "a b",
            "",
        ] {
            let e = labels(&json!({"tests": [bad]})).unwrap_err().to_string();
            assert!(e.contains("is refused"), "{bad}: {e}");
        }
        assert!(labels(&json!({"tests": [1]})).is_err());
    }

    /// Failures the base has too are not the change's, by name only; a base that cannot be
    /// compared (no run, a crash, no test) leaves every failure, said to be uncompared.
    #[test]
    fn failures_the_base_has_too_are_not_the_changes() {
        let run = |failed: u32, failing: &[&str]| TestRun {
            passed: 3,
            failed,
            failing: failing.iter().map(|n| n.to_string()).collect(),
            shared: Vec::new(),
            tail: "end".into(),
        };
        let shown = "exit code 1: 3 passed, 2 failed\nfailing:\nold\nnew\n\nend";
        let (after, out) = against(Ok(run(1, &["old"])), run(2, &["old", "new"]), shown);
        let names = |n: &[&str]| n.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        assert_eq!(
            (after.failed, after.failing, after.shared),
            (1, names(&["new"]), names(&["old"]))
        );
        assert!(
            out.starts_with(
                "exit code 1: 3 passed, 1 failed\nstill failing, as before your change: old\n\
                 failing:\nnew\n\n"
            ),
            "{out}"
        );
        assert!(!uncompared(&out) && as_before(&out), "{out}");
        // A base that passes leaves every failure, compared.
        let (after, out) = against(Ok(run(0, &[])), run(2, &["old", "new"]), shown);
        assert_eq!((after.failed, out.as_str()), (2, shown));
        // A run that names no failure (a crash) is never subtracted, the base's crash or not.
        let none = TestRun {
            passed: 0,
            ..run(0, &[])
        };
        for (base, after) in [
            (Ok(run(1, &[])), run(1, &[])),
            (Ok(run(1, &[])), run(1, &["new"])),
            (Ok(run(1, &["old"])), run(1, &[])),
            (Ok(none), run(1, &["new"])),
            (Err("the checkout failed".into()), run(1, &["new"])),
        ] {
            let (kept, out) = against(base, after.clone(), shown);
            assert_eq!(kept, after);
            assert!(uncompared(&out), "{out}");
        }
        // Only the note says so, not the output's end.
        assert!(!uncompared(&format!(
            "{shown}\n{UNCOMPARED}: told by a test"
        )));
    }

    /// A replace changes its block and what is inside it; one whose span the map got wrong is
    /// refused: too long, it would eat the block after it; too short, its old tail would join
    /// the new text. A class's method is replaced inside the class, which may change.
    #[test]
    fn a_replace_that_changes_another_block_is_refused() {
        let path = "a.js";
        let text = "const f = () => 1\nconst g = () => 2\nconst h = () => 3\n";
        let symbols = map::symbols(path, text);
        let replace = |s: &Symbol, from: usize, to: usize, new: &str| {
            // The map as it would be with `s`'s span in place of f's.
            let symbols: Vec<Symbol> = std::iter::once(s.clone())
                .chain(symbols[1..].iter().cloned())
                .collect();
            let after = format!("{}{new}{}", &text[..from], &text[to..]);
            others_kept(path, text, &symbols, s, &after, from..from + new.len())
        };
        let new = "const f = () => 10\n";
        let g = symbols[1].start;
        assert!(replace(&symbols[0], 0, g, new).is_ok());
        // f as a map that ran on into g would have it.
        let long = Symbol {
            end: symbols[1].end,
            end_line: 2,
            ..symbols[0].clone()
        };
        let err = replace(&long, 0, symbols[2].start, new).unwrap_err();
        assert!(err.to_string().contains("would change g too"), "{err}");
        // f as a map that stopped short of its `> 1` would leave it to join the new text.
        let short = Symbol {
            end: 14,
            ..symbols[0].clone()
        };
        let err = replace(&short, 0, 14, "const f = (a) =").unwrap_err();
        assert!(err.to_string().contains("would change f too"), "{err}");
        let class = "class A {\n  run() {\n    return 1;\n  }\n}\nfunction b() {\n}\n";
        let symbols = map::symbols(path, class);
        let run = &symbols[1];
        let new = "  run() {\n    return 2;\n  }\n";
        let from = class.find("  run").unwrap();
        let to = class.find("}\nfunction").unwrap();
        let after = format!("{}{new}{}", &class[..from], &class[to..]);
        assert!(others_kept(path, class, &symbols, run, &after, from..from + new.len()).is_ok());
    }

    /// A rewrite that drops a top-level block its about does not name is caught; one it names,
    /// or a block kept, is not.
    #[test]
    fn a_rewrite_that_drops_an_unnamed_block_is_caught() {
        let old = "function a() {\n  return 1;\n}\nfunction b() {\n  return 2;\n}\n";
        let new = "function a() {\n  return 3;\n}\n";
        assert_eq!(dropped("s.js", old, new, "a returns 3"), ["b"]);
        assert!(dropped("s.js", old, new, "a returns 3, and b goes").is_empty());
        assert!(dropped("s.js", old, old, "").is_empty());
        assert!(dropped("s.txt", "x\n", "", "").is_empty());
    }
}
