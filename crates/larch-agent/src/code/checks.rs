//! Cross-file checks after a write: references between a project's files that cannot resolve,
//! each told as the line the model fixes. Precision over recall: whatever might resolve in a
//! way these regexes cannot see (a framework, a star import, a global) is left alone.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    ops::Range,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex, OnceLock},
    time::SystemTime,
};

use regex::Regex;

use super::{r#loop::page_names, map::js_code};
use crate::tools::code::{Rule, ignored, load_gitignore};

/// Code files read past this many leave the ids and classes unchecked: an element's may sit in
/// an unread one. The changed files are read first, so they are always checked.
pub(crate) const MAX_FILES: usize = 5000;
/// Folder entries listed past this many stop the walk the same way: a folder of data.
const MAX_ENTRIES: usize = 50_000;
/// A larger file is generated or data, not code a model wrote.
pub(crate) const MAX_BYTES: u64 = 1 << 20;
/// Server templates, whose markup gives elements ids and classes as a page's does.
const TEMPLATES: &[&str] = &[
    "ejs",
    "hbs",
    "handlebars",
    "mustache",
    "njk",
    "j2",
    "jinja",
    "jinja2",
    "twig",
    "php",
    "erb",
    "liquid",
];
/// The kinds of file the checks read, besides `TEMPLATES`.
const READ: &[&str] = &[
    "html", "htm", "css", "js", "mjs", "cjs", "jsx", "ts", "tsx", "mts", "cts", "vue", "svelte",
    "py", "rs",
];
/// Folders of dependencies and build output, never the project's own references. A folder with
/// a `pyvenv.cfg` is a virtualenv whatever its name.
pub(crate) const SKIP: &[&str] = &[
    "node_modules",
    "target",
    "__pycache__",
    "venv",
    "env",
    "dist",
    "build",
];
/// Scripts that import and export.
const JS: &[&str] = &["js", "mjs", "cjs", "jsx", "ts", "tsx", "mts", "cts"];
/// Files whose markup gives elements ids and classes, besides `TEMPLATES`.
const MARKUP: &[&str] = &["html", "htm", "jsx", "tsx", "vue", "svelte"];
/// Python that makes names where no regex sees them.
const PY_UNSEEN: &[&str] = &[
    "import *",
    "__getattr__",
    "globals()",
    "exec(",
    "__builtins__",
    "builtins.",
    "setattr(",
];
/// Python's callable builtins, and keywords a bracket may follow.
const PY_KNOWN: &str = "ArithmeticError AssertionError AttributeError BaseException BaseExceptionGroup \
    BlockingIOError BrokenPipeError BufferError BytesWarning ChildProcessError \
    ConnectionAbortedError ConnectionError ConnectionRefusedError ConnectionResetError \
    DeprecationWarning EOFError EncodingWarning EnvironmentError Exception ExceptionGroup \
    FileExistsError FileNotFoundError FloatingPointError FutureWarning GeneratorExit IOError \
    ImportError ImportWarning IndentationError IndexError InterruptedError IsADirectoryError \
    KeyError KeyboardInterrupt LookupError MemoryError ModuleNotFoundError NameError \
    NotADirectoryError NotImplementedError OSError OverflowError PendingDeprecationWarning \
    PermissionError ProcessLookupError PythonFinalizationError RecursionError ReferenceError \
    ResourceWarning RuntimeError RuntimeWarning StopAsyncIteration StopIteration SyntaxError \
    SyntaxWarning SystemError SystemExit TabError TimeoutError TypeError UnboundLocalError \
    UnicodeDecodeError UnicodeEncodeError UnicodeError UnicodeTranslateError UnicodeWarning \
    UserWarning ValueError Warning ZeroDivisionError __import__ abs aiter all anext any \
    ascii bin bool breakpoint bytearray bytes callable chr classmethod compile complex \
    copyright credits delattr dict dir divmod enumerate eval exec exit filter float format \
    frozenset getattr globals hasattr hash help hex id input int isinstance issubclass iter \
    len license list locals map max memoryview min next object oct open ord pow print \
    property quit range repr reversed round set setattr slice sorted staticmethod str sum \
    super tuple type vars zip _ if elif while for and or not in is return yield assert await \
    del with except raise match case lambda else from import as async";

/// When a file last changed and its size: a text read at the same stamp is read again from
/// memory.
pub(crate) type Stamp = (Option<SystemTime>, u64);

/// The folders whose texts are kept, the last checked last: the attempts of the runs going on.
const KEPT: usize = 8;

/// Each folder's texts the checks read, with the stamp each was read at: a writing reply's
/// check reads again only the files changed since the last one.
// ponytail: holds what one check holds at its peak, for the run; cap its bytes when a big
// repository's run shows the memory
static TEXTS: Mutex<Vec<(PathBuf, Texts)>> = Mutex::new(Vec::new());

/// A folder's texts by path, each with the stamp it was read at.
type Texts = BTreeMap<String, (Stamp, Arc<str>)>;

/// Drops the texts kept of `dir`, once the run in it ended.
pub(crate) fn forget(dir: &Path) {
    let mut kept = TEXTS.lock().unwrap_or_else(|e| e.into_inner());
    kept.retain(|(d, _)| d != dir);
}

/// One reference that cannot resolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// The file the reference is in, relative to the folder, with `/` between names.
    pub file: String,
    /// Its line, from 1.
    pub line: usize,
    /// What is wrong and its fix, in one line, after the file and the line.
    pub message: String,
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} line {} {}", self.file, self.line, self.message)
    }
}

/// The references in `dir` that cannot resolve and are in a `changed` file or point at one:
/// HTML and CSS links to missing files, ids and classes a script looks up that no markup or
/// script makes, JS/TS relative imports of missing files or exports, Python imports of missing
/// local modules or names, Rust `mod x;` without its file, and JS/Python calls of a name the
/// file never defines or imports.
pub fn check(dir: &Path, changed: &BTreeSet<String>) -> Vec<Finding> {
    if changed.is_empty() {
        return Vec::new();
    }
    let (mut listed, complete) = listed(dir, readable);
    // The changed files too, so a project past the bounds still checks them.
    for c in changed.iter().filter(|c| readable(c)) {
        if let Some(m) =
            (fs::metadata(dir.join(c)).ok()).filter(|m| m.is_file() && m.len() <= MAX_BYTES)
        {
            listed.insert(c.clone(), (m.modified().ok(), m.len()));
        }
    }
    let was = {
        let mut kept = TEXTS.lock().unwrap_or_else(|e| e.into_inner());
        (kept.iter().position(|(d, _)| d == dir)).map_or_else(BTreeMap::new, |at| kept.remove(at).1)
    };
    let texts: Texts = (listed.into_iter())
        .filter_map(|(path, stamp)| match was.get(&path) {
            // A write may keep the size within the clock's tick: what changed is read again.
            Some((at, text)) if *at == stamp && !changed.contains(&path) => {
                Some((path, (stamp, text.clone())))
            }
            _ => {
                let text = fs::read_to_string(dir.join(&path)).ok()?;
                Some((path, (stamp, text.into())))
            }
        })
        .collect();
    let files = (texts.iter())
        .map(|(p, (_, t))| (p.clone(), t.clone()))
        .collect();
    {
        let mut kept = TEXTS.lock().unwrap_or_else(|e| e.into_inner());
        kept.push((dir.to_path_buf(), texts));
        if kept.len() > KEPT {
            kept.remove(0);
        }
    }
    // A file that names a changed one may import it, link it or look up its elements.
    let mut stems = BTreeSet::new();
    for c in changed {
        let stem = name(c).split('.').next().unwrap_or("");
        match stem {
            "__init__" | "index" | "mod" => stems.insert(name(parent(c))),
            _ => stems.insert(stem),
        };
    }
    stems.remove("");
    let p = Project {
        dir,
        files,
        complete,
        changed,
        stems,
        names: OnceLock::new(),
    };
    let mut out = Vec::new();
    links(&p, &mut out);
    lookups(&p, &mut out);
    imports(&p, &mut out);
    js_calls(&p, &mut out);
    python(&p, &mut out);
    rust_mods(&p, &mut out);
    out
}

/// The folder, its files' texts by path, whether every one was read, and the write.
struct Project<'a> {
    dir: &'a Path,
    files: BTreeMap<String, Arc<str>>,
    complete: bool,
    changed: &'a BTreeSet<String>,
    /// The names of the changed files, which a file naming one may refer to.
    stems: BTreeSet<&'a str>,
    /// The name of every file of the project, listed at the first link that is not there.
    names: OnceLock<BTreeSet<String>>,
}

impl Project<'_> {
    /// Adds `message` at `line` of `file` when `file` or one of `related` changed.
    fn push(
        &self,
        out: &mut Vec<Finding>,
        file: &str,
        related: &[&str],
        line: usize,
        message: String,
    ) {
        if self.changed.contains(file) || related.iter().any(|r| self.changed.contains(*r)) {
            out.push(Finding {
                file: file.into(),
                line,
                message,
            });
        }
    }

    /// The files of the `kinds`, with their texts.
    fn of<'s>(&'s self, kinds: &'s [&str]) -> impl Iterator<Item = (&'s str, &'s str)> {
        (self.files.iter())
            .filter(|(f, _)| kinds.contains(&ext(f)))
            .map(|(f, t)| (f.as_str(), &**t))
    }

    /// The files of the `kinds` a finding of the write may be in: changed, or naming a changed
    /// one. The rest is never checked, only read for what it defines.
    fn subjects<'s>(&'s self, kinds: &'s [&str]) -> impl Iterator<Item = (&'s str, &'s str)> {
        (self.of(kinds)).filter(|(f, t)| self.subject(f, t))
    }

    /// Whether `file`, with `text`, changed or names a changed file.
    fn subject(&self, file: &str, text: &str) -> bool {
        self.changed.contains(file) || self.stems.iter().any(|s| text.contains(s))
    }

    /// Whether a file named `file` is anywhere in the project, as `listed` walks it.
    fn named(&self, file: &str) -> bool {
        let names = self.names.get_or_init(|| {
            let all = listed(self.dir, |_| true).0;
            all.keys().map(|p| name(p).to_string()).collect()
        });
        names.contains(file)
    }

    fn is_file(&self, path: &str) -> bool {
        self.files.contains_key(path) || self.dir.join(path).is_file()
    }

    fn text(&self, path: &str) -> Option<Cow<'_, str>> {
        match self.files.get(path) {
            Some(t) => Some(Cow::Borrowed(&**t)),
            None => fs::read_to_string(self.dir.join(path)).ok().map(Cow::Owned),
        }
    }

    /// The names in the folder `path` of the project.
    fn names_in(&self, path: &str) -> Vec<String> {
        let Ok(list) = fs::read_dir(self.dir.join(path)) else {
            return Vec::new();
        };
        (list.flatten())
            .filter_map(|e| e.file_name().to_str().map(String::from))
            .collect()
    }
}

fn readable(path: &str) -> bool {
    READ.contains(&ext(path)) || TEMPLATES.contains(&ext(path))
}

/// The files under `dir` a project's checks and lookups read, `keep` saying which by path, with
/// their stamps, and whether the walk saw every one: each folder's names in order, without a
/// hidden, dependency or build folder (`SKIP`), a virtualenv, what .gitignore ignores, or a
/// file over `MAX_BYTES`; at most `MAX_FILES` kept and `MAX_ENTRIES` listed.
pub(crate) fn listed(dir: &Path, keep: fn(&str) -> bool) -> (BTreeMap<String, Stamp>, bool) {
    let mut walk = Walk {
        files: BTreeMap::new(),
        entries: 0,
        complete: true,
        rules: Vec::new(),
        keep,
    };
    walk.go(dir, "");
    (walk.files, walk.complete)
}

/// A walk of a project (`listed`): what it kept, the entries it listed, whether it saw all,
/// and the .gitignore rules of the folders it is in.
struct Walk {
    files: BTreeMap<String, Stamp>,
    entries: usize,
    complete: bool,
    rules: Vec<Rule>,
    keep: fn(&str) -> bool,
}

impl Walk {
    fn go(&mut self, dir: &Path, rel: &str) {
        let at = dir.join(rel);
        let Ok(list) = fs::read_dir(&at) else {
            return;
        };
        let outer = self.rules.len();
        // A .gitignore that cannot be read ignores nothing, as no file.
        let _ = load_gitignore(&at, &mut self.rules);
        let mut list: Vec<fs::DirEntry> = list.flatten().collect();
        list.sort_by_key(fs::DirEntry::file_name);
        for e in list {
            self.entries += 1;
            if self.entries > MAX_ENTRIES {
                self.complete = false;
            }
            if !self.complete {
                break;
            }
            let (Some(name), Ok(ty)) = (e.file_name().to_str().map(String::from), e.file_type())
            else {
                continue;
            };
            if name.starts_with('.')
                || SKIP.contains(&name.as_str())
                || ignored(&self.rules, &e.path(), ty.is_dir())
            {
                continue;
            }
            let path = under(rel, &name);
            if ty.is_dir() && !e.path().join("pyvenv.cfg").exists() {
                self.go(dir, &path);
            }
            let Ok(meta) = e.metadata() else { continue };
            if !ty.is_file() || !(self.keep)(&path) || meta.len() > MAX_BYTES {
                continue;
            }
            if self.files.len() >= MAX_FILES {
                self.complete = false;
                break;
            }
            self.files.insert(path, (meta.modified().ok(), meta.len()));
        }
        self.rules.truncate(outer);
    }
}

fn ext(path: &str) -> &str {
    name(path).rsplit_once('.').map_or("", |(_, e)| e)
}

fn name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(d, _)| d)
}

/// `name` inside the project folder `dir`, "" being the top.
fn under(dir: &str, name: &str) -> String {
    match dir {
        "" => name.into(),
        _ => format!("{dir}/{name}"),
    }
}

/// `rel` from the folder of `file`, as a project path; None when it leaves the project.
fn join(file: &str, rel: &str) -> Option<String> {
    let mut parts: Vec<&str> = parent(file).split('/').filter(|s| !s.is_empty()).collect();
    for c in rel.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            c => parts.push(c),
        }
    }
    Some(parts.join("/"))
}

/// The relative path a script in `from` imports `to` by.
fn relative(from: &str, to: &str) -> String {
    let a: Vec<&str> = parent(from).split('/').filter(|s| !s.is_empty()).collect();
    let b: Vec<&str> = to.split('/').collect();
    let common = (a.iter().zip(&b[..b.len() - 1]))
        .take_while(|(x, y)| x == y)
        .count();
    let rest = b[common..].join("/");
    let rest = (rest.strip_suffix(".ts").or(rest.strip_suffix(".tsx"))).unwrap_or(&rest);
    match a.len() - common {
        0 => format!("./{rest}"),
        up => format!("{}{rest}", "../".repeat(up)),
    }
}

/// The line, from 1, of byte `at` of `text`.
fn line(text: &str, at: usize) -> usize {
    text[..at].matches('\n').count() + 1
}

/// The one of `among` most like `name`, when one is close.
fn near<'a>(name: &str, among: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let score = |c: &str| match c.contains(name) || name.contains(c) {
        true => strsim::normalized_levenshtein(name, c).max(0.75),
        false => strsim::normalized_levenshtein(name, c),
    };
    (among.into_iter())
        .filter(|c| *c != name && !c.is_empty())
        .map(|c| (score(c), c))
        .filter(|(s, _)| *s >= 0.5)
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, c)| c)
}

/// Where `word` stands in `text` as a whole name, `wordy` saying which characters make one.
fn words<'t>(
    text: &'t str,
    word: &'t str,
    wordy: fn(char) -> bool,
) -> impl Iterator<Item = usize> + 't {
    text.match_indices(word)
        .map(|(i, _)| i)
        .filter(move |&i| !text[..i].ends_with(wordy) && !text[i + word.len()..].starts_with(wordy))
}

fn js_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '$'
}

fn py_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn css_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-'
}

macro_rules! re {
    ($name:ident, $re:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($re).expect("a valid regex"));
    };
}

re!(HTML_LINK, r#"(?i)\b(?:href|src)\s*=\s*["']([^"'#?]+)"#);
re!(
    CSS_LINK,
    r#"url\(\s*["']?([^"')#?]+)|@import\s+["']([^"'#?]+)"#
);
re!(COMMENT, r"(?s)<!--.*?-->|/\*.*?\*/");

/// The byte ranges of `text` inside an HTML or a CSS comment.
fn comments(text: &str) -> Vec<Range<usize>> {
    COMMENT.find_iter(text).map(|m| m.range()).collect()
}

/// Files a page or a stylesheet links that are not there.
fn links(p: &Project, out: &mut Vec<Finding>) {
    for (file, text) in p.subjects(&["html", "htm", "css"]) {
        let re = match ext(file) {
            "css" => &CSS_LINK,
            _ => &HTML_LINK,
        };
        let hidden = comments(text);
        for m in re.captures_iter(text).filter_map(|c| c.get(1).or(c.get(2))) {
            let target = m.as_str().trim();
            // A URL, a path from the server's root, a template, or a route such as "about".
            let odd = target.contains([':', '{', '$', '<', ' ', '\\'])
                || target.starts_with('/')
                || target.starts_with("www.");
            if odd || !name(target).contains('.') || hidden.iter().any(|r| r.contains(&m.start())) {
                continue;
            }
            let Some(path) = join(file, target) else {
                continue;
            };
            // A server, a build or a generator may put it there: missing only when no file of
            // its name is anywhere in the project.
            if p.dir.join(&path).exists() || p.named(name(target)) {
                continue;
            }
            // Told only that it is missing, a small model edits the link again and again.
            let mut message =
                format!("links {target}, which does not exist: write it with write_file {path}");
            let siblings = p.names_in(parent(&path));
            if let Some(n) = near(name(target), siblings.iter().map(String::as_str)) {
                message += &format!(", or link {}, which exists", under(parent(target), n));
            }
            p.push(out, file, &[&path], line(text, m.start()), message);
        }
    }
}

re!(
    REMOTE_SCRIPT,
    r#"(?is)<script\b[^>]*\bsrc\s*=\s*["'](?:https?:)?//"#
);
re!(GIVES_ID, r#"\bid['"]?\s*[=,:]\s*\{?\\?["'`]?([\w-]+)"#);
// An id whole: `#tab-${n}` looks up no one id.
re!(
    ID_LOOKUP,
    r#"getElementById\(\s*["'`]([\w-]+)["'`]\s*\)|querySelector(?:All)?\(\s*["'`]#([\w-]+)["'`\s.:\[>+~,)]"#
);
re!(
    CLASS_LOOKUP,
    r#"querySelector(?:All)?\(\s*["'`]\.([\w-]+)["'`]\s*\)|getElementsByClassName\(\s*["'`]([\w-]+)["'`]\s*\)"#
);

/// Ids and classes a script looks up that no markup or script gives an element. A project with
/// no markup, read only in part, or with a page that loads a script from the web, which may make
/// its own, is left alone.
fn lookups(p: &Project, out: &mut Vec<Finding>) {
    let markup = |f: &str| MARKUP.contains(&ext(f)) || TEMPLATES.contains(&ext(f));
    let remote = p
        .of(&["html", "htm"])
        .any(|(_, t)| REMOTE_SCRIPT.is_match(t));
    if !p.complete || !p.files.keys().any(|f| markup(f)) || remote {
        return;
    }
    // A page that changed may drop an id any script looks up.
    let all_look = p.changed.iter().any(|c| markup(c));
    let lookers: Vec<(&str, &str)> = (p
        .of(&["js", "mjs", "cjs", "jsx", "ts", "tsx", "html", "htm"]))
    .filter(|(f, t)| all_look || p.subject(f, t))
    .collect();
    if lookers.is_empty() {
        return;
    }
    let pages: Vec<String> = p.of(&["html", "htm"]).map(|(f, _)| f.into()).collect();
    let related: Vec<&str> = pages.iter().map(String::as_str).collect();
    let code: Vec<&str> = (p.files.iter())
        .filter(|(f, _)| ext(f) != "css")
        .map(|(_, t)| &**t)
        .collect();
    let given: BTreeSet<&str> = (code.iter())
        .flat_map(|t| GIVES_ID.captures_iter(t))
        .filter_map(|c| c.get(1).map(|m| m.as_str()))
        .collect();
    let by_id: BTreeMap<&str, usize> = (code.iter())
        .flat_map(|t| ID_LOOKUP.captures_iter(t))
        .filter_map(|c| c.get(1).map(|m| m.as_str()))
        .fold(BTreeMap::new(), |mut n, c| {
            *n.entry(c).or_default() += 1;
            n
        });
    // A value in quotes, not a class's: class="board" gives no id.
    let quoted = |w: &str| -> usize {
        let each = |t: &&str| {
            let at = ['\'', '"', '`'].map(|q| {
                t.match_indices(&format!("{q}{w}{q}"))
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>()
            });
            at.concat()
                .into_iter()
                .filter(|&i| {
                    let before = t[..i]
                        .trim_end_matches(|c: char| c.is_whitespace() || c == '=' || c == ':');
                    !before.ends_with("class") && !before.ends_with("className")
                })
                .count()
        };
        code.iter().map(each).sum()
    };
    // Each test only when the cheaper ones before it fail: the last ones read every file.
    let has_id = |id: &str| {
        // A prefix and a number, such as 'cell-' + i.
        let prefixed =
            || (given.iter()).any(|g| g.len() >= 2 && id.len() > g.len() && id.starts_with(g));
        // A name and a suffix, such as `${p}-panel`.
        let suffixed = || {
            id.find('-').is_some_and(|i| {
                let tail = &id[i..];
                let quoted = ["}", "'", "\""].map(|q| format!("{q}{tail}"));
                code.iter()
                    .any(|t| quoted.iter().any(|q| t.contains(q.as_str())))
            })
        };
        // A value in the scripts, such as a list of tab names an element takes its id from.
        given.contains(id)
            || prefixed()
            || suffixed()
            || quoted(id) > by_id.get(id).copied().unwrap_or(0)
    };
    let by_class: BTreeMap<&str, usize> = (code.iter())
        .flat_map(|t| CLASS_LOOKUP.captures_iter(t))
        .filter_map(|c| c.get(1).or(c.get(2)).map(|m| m.as_str()))
        .fold(BTreeMap::new(), |mut n, c| {
            *n.entry(c).or_default() += 1;
            n
        });
    let has_class = |c: &str| {
        let uses: usize = code.iter().map(|t| words(t, c, css_word).count()).sum();
        // A class made as a prefix and a number, such as 'player-' + n.
        let made = c.rfind('-').is_some_and(|i| {
            let stem = &c[..=i];
            (code.iter()).any(|t| {
                ["'", "\"", "${"]
                    .iter()
                    .any(|q| t.contains(&format!("{stem}{q}")))
            })
        });
        made || uses > by_class.get(c).copied().unwrap_or(0)
    };
    // The pages' names, read once each, at the first finding that names them.
    let (mut ids_of, classes) = (BTreeMap::new(), OnceLock::new());
    for (file, text) in lookers {
        let mut told = BTreeSet::new();
        // A lookup in a comment, the script's or the page's, looks nothing up.
        let (code, hidden) = (js_code(text), comments(text));
        let commented =
            |at: usize| code.as_bytes()[at] == b' ' || hidden.iter().any(|r| r.contains(&at));
        let looked = |re: &'static Regex| {
            (re.captures_iter(text)).filter(|c| !commented(c.get(0).expect("the match").start()))
        };
        for c in looked(&ID_LOOKUP) {
            let m = c.get(1).or(c.get(2)).expect("a group");
            let id = m.as_str();
            let joined = text[c.get(0).expect("the match").end()..]
                .trim_start()
                .starts_with('+');
            if joined || id.ends_with(['-', '_']) || has_id(id) || !told.insert(id) {
                continue;
            }
            // The page that loads the script, as the browser sees it.
            let page = match ext(file) {
                "html" | "htm" => file,
                _ => (pages.iter())
                    .find(|pg| p.files.get(*pg).is_some_and(|t| t.contains(name(file))))
                    .or(pages.first())
                    .map_or(file, String::as_str),
            };
            let ids = &ids_of
                .entry(page)
                .or_insert_with(|| page_names(p.dir, &[page.into()]))
                .1;
            let has = match ids.is_empty() {
                true => format!("{page} has no ids"),
                false => format!(
                    "the ids {page} has: {}",
                    Vec::from_iter(ids.iter().cloned()).join(", ")
                ),
            };
            // The lookup finds nothing, and the script stops at the first use of it.
            let message = format!(
                "looks up #{id}, which no element of {page} has: give the element it means \
                 id=\"{id}\", or use one of {has}"
            );
            p.push(out, file, &related, line(text, m.start()), message);
        }
        for c in looked(&CLASS_LOOKUP) {
            let m = c.get(1).or(c.get(2)).expect("a group");
            let class = m.as_str();
            if has_class(class) || !told.insert(class) {
                continue;
            }
            let classes = &classes.get_or_init(|| page_names(p.dir, &pages)).0;
            let hint = near(class, classes.iter().map(String::as_str))
                .map_or(String::new(), |n| format!("; the pages use .{n}"));
            let message = format!(
                "looks up .{class}, but no page or script gives an element class=\"{class}\"\
                 {hint}: look up a class that is there, or add it to the element"
            );
            p.push(out, file, &related, line(text, m.start()), message);
        }
    }
}

re!(
    IMPORT,
    r#"(?m)^[ \t]*(import|export)\s+(?:type\s+)?((?:[\w$]+\s*,\s*)?(?:\{[^}]*\}|\*(?:\s+as\s+[\w$]+)?|[\w$]+))\s*from\s*["'](\.\.?/[^"'?]*)[^"']*["']"#
);
re!(
    BARE_IMPORT,
    r#"\b(?:(import)\s*["'](\.\.?/[^"'?]*)[^"']*["']|(require|import)\s*\(\s*["'](\.\.?/[^"'?]*)[^"']*["']\s*\))"#
);
re!(
    DECLARED,
    r"\bexport\s+(?:declare\s+)?(?:async\s+)?(?:abstract\s+)?(?:function\s*\*?\s*|(?:class|const\s+enum|const|let|var|enum|interface|type|namespace)\s+)([\w$]+)"
);
re!(EXPORT_LIST, r"\bexport\s+(?:type\s+)?\{([^}]*)\}");
re!(EXPORT_STAR_AS, r"\bexport\s*\*\s*as\s+([\w$]+)");
re!(
    EXPORTS_OPEN,
    r"\bexport\s*\*\s*from|module\.exports|\bexports\.|\bexport\s+(?:const|let|var)\s*[\{\[]|\bexport\s*="
);
re!(EXPORT_DEFAULT, r"\bexport\s+default\b");

/// What a JS/TS module exports: its names, whether it has a default, and whether it may export
/// more than these regexes see (`export *`, CommonJS). `code` is its text without comments
/// and strings.
struct Exports {
    code: String,
    names: BTreeSet<String>,
    default: bool,
    open: bool,
}

impl Exports {
    fn of(text: &str) -> Exports {
        let code = js_code(text);
        let mut names: BTreeSet<String> = (DECLARED.captures_iter(&code))
            .chain(EXPORT_STAR_AS.captures_iter(&code))
            .map(|c| c[1].to_string())
            .collect();
        for c in EXPORT_LIST.captures_iter(&code) {
            for item in c[1].split(',') {
                let item = item.trim().trim_start_matches("type ");
                if let Some(n) = item.split_whitespace().last() {
                    names.insert(n.to_string());
                }
            }
        }
        Exports {
            default: EXPORT_DEFAULT.is_match(&code) || names.remove("default"),
            open: EXPORTS_OPEN.is_match(&code),
            names,
            code,
        }
    }

    /// Whether it exports `n`, also on an export line the regexes do not parse, such as
    /// `export const a = 1, b = 2`.
    fn has(&self, n: &str) -> bool {
        let line = Regex::new(&format!(r"(?m)^\s*export\b.*\b{}\b", regex::escape(n)));
        self.names.contains(n) || line.is_ok_and(|l| l.is_match(&self.code))
    }
}

/// The file the relative import `base`, a project path, resolves to, as bundlers and Node do.
fn script(p: &Project, base: &str) -> Option<String> {
    const EXTS: &[&str] = &[
        "ts", "tsx", "js", "jsx", "mjs", "cjs", "mts", "cts", "json", "vue", "svelte", "d.ts",
    ];
    let mut all = vec![base.to_string()];
    all.extend(EXTS.iter().map(|e| format!("{base}.{e}")));
    all.extend(EXTS.iter().map(|e| under(base, &format!("index.{e}"))));
    // TypeScript imports ./x.js meaning ./x.ts.
    if let Some(stem) = ["js", "jsx", "mjs", "cjs"]
        .iter()
        .find_map(|e| base.strip_suffix(&format!(".{e}")))
    {
        all.extend(
            ["ts", "tsx", "mts", "cts"]
                .iter()
                .map(|e| format!("{stem}.{e}")),
        );
    }
    all.into_iter().find(|c| p.is_file(c))
}

/// JS/TS relative imports of files that are not there, and of names their modules do not export.
fn imports(p: &Project, out: &mut Vec<Finding>) {
    let mut seen: BTreeMap<String, Option<Exports>> = BTreeMap::new();
    for (file, text) in p.subjects(JS) {
        // An import in a comment or a string is none: its keyword is code only in `code`.
        let code = js_code(text);
        let live = |m: regex::Match| code.get(m.range()) == Some(m.as_str());
        let specs = (IMPORT.captures_iter(text))
            .filter(|c| live(c.get(1).expect("a group")))
            .map(|c| (c.get(3).expect("a group"), Some(c)))
            .chain(
                (BARE_IMPORT.captures_iter(text))
                    .filter(|c| live(c.get(1).or(c.get(3)).expect("a group")))
                    .map(|c| (c.get(2).or(c.get(4)).expect("a group"), None)),
            );
        for (spec, clause) in specs {
            let at = line(text, spec.start());
            let spec = spec.as_str();
            let Some(base) = join(file, spec) else {
                continue;
            };
            let Some(target) = script(p, &base) else {
                // A path the model can write: the importer's kind, never a bare name.
                let kind = ext(file);
                let want = match (spec.ends_with('/'), name(&base).contains('.')) {
                    (true, _) => under(&base, &format!("index.{kind}")),
                    (false, true) => base.clone(),
                    (false, false) => format!("{base}.{kind}"),
                };
                let siblings = p.names_in(parent(&base));
                let stems = (siblings.iter().map(String::as_str))
                    .chain(siblings.iter().map(|s| s.split('.').next().unwrap_or(s)));
                let message = match near(name(&base), stems) {
                    Some(n) => format!(
                        "imports {spec}, but there is no such file; {} exists: import that, or \
                         write {want}",
                        under(parent(spec), n)
                    ),
                    None => format!("imports {spec}, but there is no such file: write {want}"),
                };
                p.push(out, file, &[&base, &want], at, message);
                continue;
            };
            let Some(clause) = clause else {
                continue;
            };
            let checked = JS.contains(&ext(&target)) && !target.ends_with(".d.ts");
            let e = (seen.entry(target.clone()))
                .or_insert_with(|| p.text(&target).filter(|_| checked).map(|t| Exports::of(&t)));
            let Some(e) = e.as_ref().filter(|e| !e.open) else {
                continue;
            };
            let has = {
                let mut has: Vec<&str> = e.names.iter().map(String::as_str).collect();
                has.truncate(12);
                match (has.is_empty(), e.default) {
                    (true, false) => "nothing".to_string(),
                    (true, true) => "only a default".to_string(),
                    (false, true) => format!("{} and a default", has.join(", ")),
                    (false, false) => has.join(", "),
                }
            };
            // The names, without the comments between them.
            let words = &code[clause.get(2).expect("a group").range()];
            let (head, braces) = match words.find('{') {
                Some(i) => (&words[..i], Some(words[i + 1..].trim_end_matches('}'))),
                None => (words, None),
            };
            let head = head.trim().trim_end_matches(',').trim();
            let mut wanted: Vec<&str> = (braces.into_iter().flat_map(|b| b.split(',')))
                .filter_map(|i| {
                    let i = i.trim().trim_start_matches("type ");
                    i.split_whitespace().next()
                })
                .collect();
            if &clause[1] == "import" && !head.is_empty() && !head.starts_with('*') {
                wanted.push("default");
            }
            for want in wanted {
                let message = match want {
                    "default" if e.default => continue,
                    "default" => format!(
                        "imports a default export from {spec}, but {target} has no export \
                         default; it exports {has}: import those in braces, or add export \
                         default to {target}"
                    ),
                    n if e.has(n) => continue,
                    n => {
                        let bare = Regex::new(&format!(
                            r"\b(?:function\s*\*?\s*|(?:class|const|let|var|enum|interface|type)\s+){}\b",
                            regex::escape(n)
                        ))
                        .expect("a valid regex");
                        match bare.is_match(&e.code) {
                            true => format!(
                                "imports {n} from {spec}, but {target} declares {n} without \
                                 export: add export to its declaration"
                            ),
                            false => format!(
                                "imports {n} from {spec}, but {target} does not export it; it \
                                 exports {has}"
                            ),
                        }
                    }
                };
                p.push(out, file, &[&target], at, message);
            }
        }
    }
}

re!(MODULE, r"(?m)^\s*(?:import|export)\b|\brequire\s*\(");
re!(USES_IMPORT, r"(?m)^\s*import\b");
re!(
    EXPORTED_FN,
    r"\bexport\s+(?:async\s+)?(?:function\s*\*?\s*|(?:class|const|let|var)\s+)([\w$]+)|\b(?:module\.)?exports\.([\w$]+)\s*="
);

/// What an occurrence of a name is: a call by its bare name, or a definition or a value.
#[derive(PartialEq)]
enum Use {
    Call,
    Defines,
    /// A member (`a.f()`) or a method's own definition (`f() {`), which defines no bare name.
    Other,
}

/// What the name `word` at byte `at` of `code` is; `def` lists the keywords that define one.
fn use_at(code: &str, at: usize, word: &str, def: &[&str]) -> Use {
    // A `\` ends a line that goes on, as in Python's `x.\` before `decode()`.
    let before = code[..at].trim_end_matches(|c: char| c.is_whitespace() || c == '\\');
    let after = code[at + word.len()..].trim_start();
    if before.ends_with('.') {
        return Use::Other;
    }
    let prev = before.rsplit(|c: char| !js_word(c)).next().unwrap_or("");
    if def.contains(&prev) || !after.starts_with('(') {
        return Use::Defines;
    }
    // Past the brackets, a body or a type marks a method being defined, not a call.
    let mut depth = 0;
    for (i, c) in after.char_indices() {
        match c {
            '(' => depth += 1,
            ')' if depth == 1 => {
                let next = after[i + 1..].trim_start();
                return match next.starts_with('{') || next.starts_with(':') {
                    true => Use::Other,
                    false => Use::Call,
                };
            }
            ')' => depth -= 1,
            _ => {}
        }
    }
    Use::Call
}

/// The first bare call of `word` in `code`, when the file calls it and never defines, imports or
/// takes it as a value.
fn undefined_call(code: &str, word: &str, def: &[&str], wordy: fn(char) -> bool) -> Option<usize> {
    let uses: Vec<(usize, Use)> = (words(code, word, wordy))
        .map(|i| (i, use_at(code, i, word, def)))
        .collect();
    if uses.iter().any(|(_, u)| *u == Use::Defines) {
        return None;
    }
    uses.into_iter()
        .find(|(_, u)| *u == Use::Call)
        .map(|(i, _)| i)
}

/// Calls in a changed JS/TS module of a function another module of the project exports, which it
/// never imports. Classic scripts share their globals, which `loop`'s page checks cover.
fn js_calls(p: &Project, out: &mut Vec<Finding>) {
    let modules: Vec<(&str, &str)> = (p.of(JS))
        .filter(|(f, t)| p.changed.contains(*f) && MODULE.is_match(t))
        .collect();
    if modules.is_empty() {
        return;
    }
    let mut exported: BTreeMap<&str, &str> = BTreeMap::new();
    for (file, text) in p.of(JS) {
        for c in EXPORTED_FN.captures_iter(text) {
            let n = c.get(1).or(c.get(2)).expect("a group").as_str();
            exported.entry(n).or_insert(file);
        }
    }
    let global = |n: &str| {
        let (window, this) = (format!("window.{n}"), format!("globalThis.{n}"));
        p.files
            .values()
            .any(|t| t.contains(&window) || t.contains(&this))
    };
    re!(CALL, r"([A-Za-z_$][\w$]*)\s*\(");
    const DEF: &[&str] = &["function", "class"];
    for (file, text) in modules {
        let code = js_code(text);
        let called: BTreeSet<&str> = (CALL.captures_iter(&code))
            .map(|c| c.get(1).expect("a group").as_str())
            .collect();
        for n in called {
            let Some(&from) = exported.get(n).filter(|f| **f != file) else {
                continue;
            };
            let Some(at) = undefined_call(&code, n, DEF, js_word).filter(|_| !global(n)) else {
                continue;
            };
            let rel = relative(file, from);
            let add = match USES_IMPORT.is_match(text) {
                true => format!("import {{ {n} }} from '{rel}'"),
                false => format!("const {{ {n} }} = require('{rel}')"),
            };
            let message = format!("calls {n}() but never imports it; {from} exports it: add {add}");
            p.push(out, file, &[from], line(text, at), message);
        }
    }
}

/// `py` without its comments and with its strings left empty, lines kept.
fn py_code(py: &str) -> String {
    let b: Vec<char> = py.chars().collect();
    let (mut out, mut i) = (String::with_capacity(py.len()), 0);
    while i < b.len() {
        let c = b[i];
        if c == '#' {
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c != '"' && c != '\'' {
            out.push(c);
            i += 1;
            continue;
        }
        let three = |i: usize| i + 2 < b.len() && b[i] == c && b[i + 1] == c && b[i + 2] == c;
        let q = if three(i) { 3 } else { 1 };
        i += q;
        while i < b.len() {
            match b[i] {
                '\\' => i += 1,
                d if d == c && (q == 1 || three(i)) => {
                    i += q;
                    break;
                }
                '\n' if q == 1 => break,
                '\n' => out.push('\n'),
                _ => {}
            }
            i += 1;
        }
        out.push_str("\"\"");
    }
    out
}

re!(
    PY_FROM,
    r"(?m)^[ \t]*from[ \t]+(\.*)([\w.]*)[ \t]+import[ \t]+(\([^)]*\)|[^\n;]*)"
);
re!(
    PY_DEFINES,
    r"(?m)^(?:async[ \t]+)?(?:def|class)[ \t]+(\w+)|^(\w+)[ \t]*(?::[^=\n]*)?="
);
re!(PY_TOP_DEF, r"(?m)^(?:async[ \t]+)?(?:def|class)[ \t]+(\w+)");

/// Where a Python import's module is in the project.
enum Py {
    /// Not a local module, or one whose names cannot be read: the standard library, an
    /// installed one, or a compiled one (`.so`, `.pyx`, `.pyi`).
    Outside,
    /// A local one that is not there, at this path.
    Missing(String),
    /// Each `.py` file or package folder it may be.
    Found(Vec<String>),
}

/// The module `module`, after `dots` leading dots, that `file` imports from.
fn py_module(p: &Project, file: &str, dots: usize, module: &str) -> Py {
    let parts: Vec<&str> = module.split('.').filter(|s| !s.is_empty()).collect();
    // Ok(None): there, but compiled.
    let find = |root: &str| -> Result<Option<String>, String> {
        let path = parts.iter().fold(root.to_string(), |r, s| under(&r, s));
        let compiled = || {
            let prefix = format!("{}.", name(&path));
            p.names_in(parent(&path))
                .iter()
                .any(|n| n.starts_with(&prefix))
        };
        match (parts.is_empty(), p.is_file(&format!("{path}.py"))) {
            (true, _) => Ok(Some(path)),
            (_, true) => Ok(Some(format!("{path}.py"))),
            _ if p.dir.join(&path).is_dir() => Ok(Some(path)),
            _ if compiled() => Ok(None),
            _ => Err(format!("{path}.py")),
        }
    };
    if dots > 0 {
        let mut base = parent(file);
        for _ in 1..dots {
            if base.is_empty() {
                return Py::Outside;
            }
            base = parent(base);
        }
        return match find(base) {
            Ok(Some(path)) => Py::Found(vec![path]),
            Ok(None) => Py::Outside,
            Err(path) => Py::Missing(path),
        };
    }
    // Python looks beside the script, at the top, where tests run from, and in src; never in a
    // folder above the script, nor inside a package, whose folder is no root.
    let mut roots = vec![parent(file), "", "src"];
    roots.dedup();
    roots.retain(|r| !p.is_file(&under(r, "__init__.py")));
    let first = parts.first().copied().unwrap_or("");
    let local = |r: &&&str| {
        let at = under(r, first);
        p.is_file(&format!("{at}.py")) || p.is_file(&format!("{at}/__init__.py"))
    };
    let tried: Vec<Result<Option<String>, String>> =
        roots.iter().filter(local).map(|r| find(r)).collect();
    if tried.iter().any(|t| matches!(t, Ok(None))) {
        return Py::Outside;
    }
    let found: Vec<String> = tried
        .iter()
        .filter_map(|t| t.clone().ok().flatten())
        .collect();
    match tried.into_iter().next() {
        None => Py::Outside,
        Some(Err(path)) if found.is_empty() => Py::Missing(path),
        _ => Py::Found(found),
    }
}

/// Whether the Python module at `path`, a `.py` file or a package folder, has `want`, or may
/// make it in a way the regexes cannot see.
fn py_has(p: &Project, path: &str, want: &str) -> bool {
    let (text, package) = match path.ends_with(".py") {
        true => (p.text(path).unwrap_or_default(), false),
        false => (
            p.text(&under(path, "__init__.py")).unwrap_or_default(),
            true,
        ),
    };
    let sub = under(path, want);
    (package && (p.is_file(&format!("{sub}.py")) || p.dir.join(&sub).is_dir()))
        || PY_UNSEEN.iter().any(|u| text.contains(u))
        || words(&text, want, py_word).next().is_some()
}

/// Python imports of local modules that are not there or of names they do not define, and calls
/// of names a file never defines, imports or takes, which are no builtin. An import indented in
/// a file that catches ImportError may be meant to fail.
fn python(p: &Project, out: &mut Vec<Finding>) {
    for (file, text) in p.subjects(&["py"]) {
        let code = py_code(text);
        let optional = text.contains("ImportError") || text.contains("ModuleNotFoundError");
        for c in PY_FROM.captures_iter(&code) {
            let whole = c.get(0).expect("the match");
            if optional && whole.as_str().starts_with([' ', '\t']) {
                continue;
            }
            let at = line(&code, whole.start());
            let (dots, module) = (c[1].len(), &c[2]);
            let shown = format!("{}{module}", &c[1]);
            let found = match py_module(p, file, dots, module) {
                Py::Outside => continue,
                Py::Missing(path) => {
                    let message = format!(
                        "imports from {shown}, but there is no {path}: write it, or fix the import"
                    );
                    p.push(out, file, &[&path], at, message);
                    continue;
                }
                Py::Found(found) => found,
            };
            let names = (c[3].split(','))
                .filter_map(|n| {
                    let n = n.trim_matches(|c: char| c.is_whitespace() || "()\\".contains(c));
                    n.split_whitespace().next()
                })
                .filter(|n| *n != "*");
            for want in names {
                if found.iter().any(|f| py_has(p, f, want)) {
                    continue;
                }
                let target = match found[0].ends_with(".py") {
                    true => found[0].clone(),
                    false => under(&found[0], "__init__.py"),
                };
                let target_text = p.text(&target).unwrap_or_default();
                let defined = (PY_DEFINES.captures_iter(&target_text))
                    .filter_map(|c| c.get(1).or(c.get(2)).map(|m| m.as_str()));
                let hint =
                    near(want, defined).map_or(String::new(), |n| format!("; did you mean {n}?"));
                let message =
                    format!("imports {want} from {shown}, but {target} does not define it{hint}");
                p.push(out, file, &[&target], at, message);
            }
        }
    }
    let mut defs: BTreeMap<&str, &str> = BTreeMap::new();
    for (file, text) in p.of(&["py"]) {
        for c in PY_TOP_DEF.captures_iter(text) {
            defs.entry(c.get(1).expect("a group").as_str())
                .or_insert(file);
        }
    }
    re!(CALL, r"([A-Za-z_]\w*)\s*\(");
    const DEF: &[&str] = &["def", "class"];
    for (file, text) in p.of(&["py"]).filter(|(f, _)| p.changed.contains(*f)) {
        if PY_UNSEEN.iter().any(|u| text.contains(u)) {
            continue;
        }
        let code = py_code(text);
        let called: BTreeSet<&str> = (CALL.captures_iter(&code))
            .map(|c| c.get(1).expect("a group").as_str())
            .filter(|n| !PY_KNOWN.split_whitespace().any(|k| k == *n))
            .collect();
        for n in called {
            let Some(at) = undefined_call(&code, n, DEF, py_word) else {
                continue;
            };
            let message = match defs.get(n).filter(|f| **f != file) {
                Some(from) => {
                    let m = (from.trim_end_matches(".py").trim_end_matches("/__init__"))
                        .replace('/', ".");
                    format!(
                        "calls {n}() but never defines or imports it; {from} defines it: add from {m} import {n}"
                    )
                }
                None => {
                    format!("calls {n}() but never defines or imports it: define it, or import it")
                }
            };
            p.push(out, file, &[], line(&code, at), message);
        }
    }
}

re!(
    RUST_MOD,
    r"(?m)^(?:pub(?:\([^)]*\))?[ \t]+)?mod[ \t]+(?:r#)?(\w+)[ \t]*;"
);

/// Rust `mod x;` at the top of a file with no file for it. A `#[path]`, or a `#[cfg]` other than
/// test that may leave it out of the build, is left alone.
fn rust_mods(p: &Project, out: &mut Vec<Finding>) {
    for (file, text) in p.subjects(&["rs"]) {
        for c in RUST_MOD.captures_iter(text) {
            let start = c.get(0).expect("the match").start();
            // What stands between the item before and this one: its attributes and comments.
            let from = text[..start].rfind([';', '{', '}']).map_or(0, |i| i + 1);
            let attrs = &text[from..start];
            let cfg = attrs.contains("cfg(") && !attrs.contains("#[cfg(test)]");
            // Inside a block, or inside a string such as a test's, it is no file's module.
            let inside = text[..start].matches('{').count() > text[..start].matches('}').count();
            if inside || cfg || attrs.contains("path") || attrs.contains('"') {
                continue;
            }
            let (m, dir) = (&c[1], parent(file));
            let stem = name(file).trim_end_matches(".rs");
            let nested = under(dir, stem);
            let all = [
                under(dir, &format!("{m}.rs")),
                under(dir, &format!("{m}/mod.rs")),
                under(&nested, &format!("{m}.rs")),
                under(&nested, &format!("{m}/mod.rs")),
            ];
            if all.iter().any(|a| p.is_file(a)) {
                continue;
            }
            let root = ["main", "lib", "mod", "build"].contains(&stem)
                || ["bin", "examples", "tests", "benches"].contains(&name(dir));
            let want = if root { &all[0] } else { &all[2] };
            let message = format!(
                "declares mod {m}; but {want} does not exist: write {want}, or remove the line"
            );
            p.push(out, file, &[want], line(text, start), message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `check` finds in a folder of `files` after a write of `changed`, every file when empty.
    fn found(files: &[(&str, &str)], changed: &[&str]) -> Vec<Finding> {
        let dir = tempfile::tempdir().expect("a temp dir");
        for (name, text) in files {
            let path = dir.path().join(name);
            fs::create_dir_all(path.parent().expect("a parent")).expect("a folder");
            fs::write(path, text).expect("a file");
        }
        let changed: BTreeSet<String> = match changed.is_empty() {
            true => files.iter().map(|(n, _)| n.to_string()).collect(),
            false => changed.iter().map(|n| n.to_string()).collect(),
        };
        check(dir.path(), &changed)
    }

    fn one(files: &[(&str, &str)]) -> Finding {
        let mut f = found(files, &[]);
        assert_eq!(f.len(), 1, "{f:#?}");
        f.remove(0)
    }

    const PAGE: &str = r##"<!doctype html>
<html>
<head>
  <link rel="stylesheet" href="style.css">
  <link rel="icon" href="https://example.com/icon.png">
</head>
<body>
  <h1 id="title">Connect four</h1>
  <div id="game-board" class="board"></div>
  <button id="reset" class="btn primary" onclick="reset()">Reset</button>
  <a href="#rules">Rules</a>
  <a href="about">About</a>
  <script src="script.js"></script>
</body>
</html>
"##;

    const STYLE: &str = r#"body { background: url("img/bg.png"); }
.board { display: grid; }
.cell { width: 3rem; }
.cell.red { background: red; }
#title { font-size: 2rem; }
@media (max-width: 600px) { .board { gap: 2px; } }
"#;

    const SCRIPT: &str = r#"const board = document.getElementById('game-board');
const title = document.querySelector('#title');
for (let i = 0; i < 42; i++) {
  const cell = document.createElement('div');
  cell.className = 'cell';
  cell.id = 'cell-' + i;
  board.appendChild(cell);
}
const first = document.getElementById('cell-0');
const panel = document.createElement('div');
panel.setAttribute('id', 'score');
document.body.appendChild(panel);
document.getElementById('score').textContent = '0';
function reset() {
  document.querySelectorAll('.cell').forEach(c => c.classList.remove('red'));
  document.querySelectorAll('.red').forEach(c => c.remove());
  title.textContent = `Turn ${1}`;
}
"#;

    fn site(script: &str) -> Vec<(&'static str, String)> {
        vec![
            ("index.html", PAGE.into()),
            ("style.css", STYLE.into()),
            ("script.js", script.into()),
            ("img/bg.png", String::new()),
        ]
    }

    fn files<'a>(owned: &'a [(&'static str, String)]) -> Vec<(&'static str, &'a str)> {
        owned.iter().map(|(n, t)| (*n, t.as_str())).collect()
    }

    #[test]
    fn a_correct_site_has_no_finding() {
        assert_eq!(found(&files(&site(SCRIPT)), &[]), vec![]);
    }

    #[test]
    fn a_script_looking_up_a_missing_id_names_the_near_one() {
        let script = SCRIPT.replace("getElementById('game-board')", "getElementById('board')");
        let f = one(&files(&site(&script)));
        assert_eq!((f.file.as_str(), f.line), ("script.js", 1));
        assert!(f.message.contains("#board"), "{}", f.message);
        assert!(
            f.message.contains("the ids index.html has: game-board"),
            "{}",
            f.message
        );
    }

    #[test]
    fn a_script_looking_up_a_missing_class_is_found() {
        let script = SCRIPT.replace("querySelectorAll('.cell')", "querySelectorAll('.cells')");
        let f = one(&files(&site(&script)));
        assert_eq!((f.file.as_str(), f.line), ("script.js", 15));
        assert!(f.message.contains(".cells"), "{}", f.message);
    }

    #[test]
    fn a_missing_linked_file_is_found_in_html_and_css() {
        let mut owned = site(SCRIPT);
        owned[0].1 = PAGE.replace("style.css", "styles.css");
        let f = one(&files(&owned));
        assert_eq!((f.file.as_str(), f.line), ("index.html", 4));
        assert!(f.message.contains("styles.css"), "{}", f.message);
        assert!(
            f.message.contains("or link style.css, which exists"),
            "{}",
            f.message
        );
        owned[0].1 = PAGE.into();
        owned.pop();
        let f = one(&files(&owned));
        assert_eq!((f.file.as_str(), f.line), ("style.css", 1));
        assert!(f.message.contains("img/bg.png"), "{}", f.message);
    }

    /// A lookup in a commented-out line looks nothing up, a whole-line `//` or a `/* */` block;
    /// the same lookup in code is still found, on its own line.
    #[test]
    fn a_lookup_in_a_comment_is_no_lookup() {
        let commented = format!(
            "{SCRIPT}// const old = document.getElementById('legacy');\n/*\n  document.querySelector('.gone');\n*/\n"
        );
        assert_eq!(found(&files(&site(&commented)), &[]), vec![]);
        let live = format!("{commented}const old = document.getElementById('legacy');\n");
        let f = one(&files(&site(&live)));
        assert_eq!((f.file.as_str(), f.line), ("script.js", 23));
        assert!(f.message.contains("#legacy"), "{}", f.message);
    }

    /// An id an element takes from a list in the script, looked up with querySelector, is
    /// there: the list holds it in quotes, and only getElementById's lookups count against that.
    #[test]
    fn an_id_from_a_list_looked_up_by_query_selector_is_there() {
        let script = format!(
            "{SCRIPT}const panels = ['home'];\npanels.forEach(p => {{ const d = document.createElement('div'); d.id = p; document.body.appendChild(d); }});\ndocument.querySelector('#home');\n"
        );
        assert_eq!(found(&files(&site(&script)), &[]), vec![]);
    }

    /// A server's template links from its app's root: app/templates/index.html's
    /// static/style.css is app/static/style.css.
    #[test]
    fn a_template_links_from_its_apps_root() {
        let page = PAGE
            .replace("style.css", "static/style.css")
            .replace("script.js", "static/script.js");
        let owned = vec![
            ("app/templates/index.html", page),
            (
                "app/static/style.css",
                STYLE.replace("img/bg.png", "bg.png"),
            ),
            ("app/static/script.js", SCRIPT.to_string()),
            ("app/static/bg.png", String::new()),
        ];
        assert_eq!(found(&files(&owned), &[]), vec![]);
    }

    /// A page that loads a framework from the web may make its ids at run time.
    #[test]
    fn a_page_with_a_remote_script_is_left_to_it() {
        let mut owned = site(&SCRIPT.replace("'game-board'", "'board'"));
        owned[0].1 = PAGE.replace(
            "<script src=\"script.js\">",
            "<script src=\"https://cdn.example.com/lib.js\"></script>\n<script src=\"script.js\">",
        );
        assert_eq!(found(&files(&owned), &[]), vec![]);
    }

    const MATH: &str = r#"export function add(a, b) { return a + b; }
export const PI = 3.14;
const square = (x) => x * x;
export { square as sq };
export default function mean(xs) { return xs.reduce(add, 0) / xs.length; }
"#;

    const MAIN: &str = r#"import mean, { add, PI, sq } from './math.js';
import { greet } from "./lib";
import * as all from './math.js';
import config from './config.json';
import { Shape } from './shape.js';
const fs = require('fs');
const legacy = require('./legacy');

class View {
  add(x) { return x; }
  render() { return greet(add(1, 2)); }
}
console.log(mean([PI]), sq(2), all.add(1, 2), config, legacy.run(), new Shape(), fs);
"#;

    fn node(main: &str) -> Vec<(&'static str, String)> {
        vec![
            ("src/main.js", main.into()),
            ("src/math.js", MATH.into()),
            (
                "src/lib/index.js",
                "export function greet(n) { return `hi ${n}`; }\n".into(),
            ),
            ("src/config.json", "{}\n".into()),
            ("src/shape.ts", "export class Shape {}\n".into()),
            (
                "src/legacy.js",
                "module.exports = { run() { return 1; } };\n".into(),
            ),
            ("package.json", "{\"type\": \"module\"}\n".into()),
        ]
    }

    #[test]
    fn a_correct_node_module_set_has_no_finding() {
        assert_eq!(found(&files(&node(MAIN)), &[]), vec![]);
    }

    #[test]
    fn an_import_of_a_missing_file_names_the_near_one() {
        let f = one(&files(&node(
            &MAIN.replace("'./math.js';\nimport {", "'./maths.js';\nimport {"),
        )));
        assert_eq!((f.file.as_str(), f.line), ("src/main.js", 1));
        assert!(f.message.contains("./maths.js"), "{}", f.message);
        assert!(f.message.contains("./math.js"), "{}", f.message);
    }

    #[test]
    fn an_import_of_a_missing_export_names_what_is_exported() {
        let f = one(&files(&node(&MAIN.replace("PI, sq }", "PI, sq, sub }"))));
        assert_eq!((f.file.as_str(), f.line), ("src/main.js", 1));
        assert!(f.message.contains("sub"), "{}", f.message);
        assert!(f.message.contains("add"), "{}", f.message);
    }

    #[test]
    fn a_default_import_of_a_module_without_one_is_found() {
        let main = MAIN.replace("import { Shape }", "import Shape");
        let f = one(&files(&node(&main)));
        assert_eq!((f.file.as_str(), f.line), ("src/main.js", 5));
        assert!(f.message.contains("default"), "{}", f.message);
    }

    #[test]
    fn a_call_to_a_function_another_module_exports_but_never_imported_is_found() {
        let main = MAIN.replace("import { greet } from \"./lib\";\n", "");
        let f = one(&files(&node(&main)));
        assert_eq!((f.file.as_str(), f.line), ("src/main.js", 10));
        assert!(f.message.contains("greet"), "{}", f.message);
        assert!(f.message.contains("./lib"), "{}", f.message);
    }

    /// Only the files a write changed, and what reads them, are checked.
    #[test]
    fn a_mistake_outside_the_write_is_left_alone() {
        let owned = node(&MAIN.replace("PI, sq }", "PI, sq, sub }"));
        assert_eq!(found(&files(&owned), &["src/shape.ts"]), vec![]);
        assert_eq!(found(&files(&owned), &["src/math.js"]).len(), 1);
    }

    const CORE: &str = r#""""The core of the package."""
from dataclasses import dataclass
import os

LIMIT: int = 10


@dataclass
class Board:
    cells: list

    def drop(self, column):
        return column % LIMIT


def winner(board, check=None):
    check = check or (lambda b: None)
    return check(board)
"#;

    const CLI: &str = r#"import sys
from typing import List
from . import core
from .core import Board, winner, LIMIT
from pkg.util import (
    shout,  # loud
    quiet as q,
)


def main(argv: List[str], out=print) -> int:
    board = Board(cells=[0] * LIMIT)
    handlers = {"win": winner}
    for name, fn in handlers.items():
        out(name, fn(board))
    f = lambda x: len(str(x))
    out(shout(q("hi")), f(sys.argv), core.winner(board), sorted([3, 1]))
    text = " ".join(argv).\\
        lower()
    raise SystemExit(main(sys.argv) if False else 0)
"#;

    fn package(cli: &str) -> Vec<(&'static str, String)> {
        vec![
            ("pkg/__init__.py", "from .core import Board\n".into()),
            ("pkg/core.py", CORE.into()),
            ("pkg/util.py", "def shout(s):\n    return s.upper()\n\n\ndef quiet(s):\n    return s.lower()\n".into()),
            ("pkg/cli.py", cli.into()),
            ("tests/test_core.py", "from pkg.core import winner\n\n\ndef test_winner():\n    assert winner([]) is None\n".into()),
            // Inside a package, `io` is the standard library's, not this one.
            ("pkg/io.py", "def dump(x):\n    return x\n".into()),
            ("pkg/sub/__init__.py", "globals().update(_load())\n".into()),
            ("pkg/sub/text.py", "from io import StringIO\nfrom . import made\n".into()),
        ]
    }

    #[test]
    fn a_correct_python_package_has_no_finding() {
        assert_eq!(found(&files(&package(CLI)), &[]), vec![]);
    }

    #[test]
    fn a_python_import_of_an_undefined_name_names_the_near_one() {
        let f = one(&files(&package(
            &CLI.replace("Board, winner", "Board, winer"),
        )));
        assert_eq!((f.file.as_str(), f.line), ("pkg/cli.py", 4));
        assert!(f.message.contains("winer"), "{}", f.message);
        assert!(f.message.contains("winner"), "{}", f.message);
    }

    #[test]
    fn a_python_import_of_a_missing_local_module_is_found() {
        let f = one(&files(&package(
            &CLI.replace("from .core import", "from .cores import"),
        )));
        assert_eq!((f.file.as_str(), f.line), ("pkg/cli.py", 4));
        assert!(f.message.contains(".cores"), "{}", f.message);
        let f = one(&files(&package(
            &CLI.replace("    quiet as q,", "    quiet as q,\n    whisper,"),
        )));
        assert_eq!((f.file.as_str(), f.line), ("pkg/cli.py", 5));
        assert!(f.message.contains("whisper"), "{}", f.message);
    }

    #[test]
    fn a_python_call_to_a_name_defined_nowhere_in_its_file_is_found() {
        let cli = CLI.replace("out(shout(q(\"hi\"))", "out(shout(q(loud(\"hi\")))");
        let mut owned = package(&cli);
        owned.push(("pkg/talk.py", "def loud(s):\n    return s + '!'\n".into()));
        let f = one(&files(&owned));
        assert_eq!((f.file.as_str(), f.line), ("pkg/cli.py", 17));
        assert!(f.message.contains("loud"), "{}", f.message);
        assert!(
            f.message.contains("from pkg.talk import loud"),
            "{}",
            f.message
        );
    }

    const LIB: &str = r#"//! A crate.
pub mod board;
mod r#loop;
pub(crate) mod net;
#[path = "elsewhere.rs"]
mod moved;
#[cfg(windows)]
mod win;

mod inline {
    pub fn f() {}
}

#[cfg(test)]
mod tests {
    const SOURCE: &str = "
mod gone;
";
}
"#;

    fn krate(lib: &str) -> Vec<(&'static str, String)> {
        vec![
            ("src/lib.rs", lib.into()),
            ("src/board.rs", "mod cell;\n".into()),
            ("src/board/cell.rs", "\n".into()),
            ("src/loop.rs", "\n".into()),
            ("src/net/mod.rs", "\n".into()),
            ("src/elsewhere.rs", "\n".into()),
            ("src/bin/tool.rs", "mod args;\nfn main() {}\n".into()),
            ("src/bin/args.rs", "\n".into()),
        ]
    }

    #[test]
    fn a_correct_rust_crate_has_no_finding() {
        assert_eq!(found(&files(&krate(LIB)), &[]), vec![]);
    }

    #[test]
    fn a_rust_mod_without_its_file_is_found() {
        let f = one(&files(&krate(
            &LIB.replace("mod net;", "mod net;\nmod gone;"),
        )));
        assert_eq!((f.file.as_str(), f.line), ("src/lib.rs", 5));
        assert!(f.message.contains("src/gone.rs"), "{}", f.message);
    }

    /// A package at the top of the folder keeps its names in ./__init__.py.
    #[test]
    fn a_package_at_the_top_resolves_its_init() {
        let files = [
            ("__init__.py", "from .events import Policy\n"),
            (
                "events.py",
                "class Policy:\n    pass\n\n\ndef get():\n    from . import Policy\n    return Policy()\n",
            ),
        ];
        assert_eq!(found(&files, &[]), vec![]);
        let f = one(&[
            ("__init__.py", ""),
            ("events.py", "def get():\n    from . import Policy\n"),
        ]);
        assert!(
            f.message.contains("__init__.py does not define"),
            "{}",
            f.message
        );
    }

    fn clean(files: &[(&str, &str)]) {
        assert_eq!(found(files, &[]), vec![]);
    }

    #[test]
    fn a_lookup_by_template_or_concatenated_prefix_is_no_finding() {
        clean(&[
            (
                "index.html",
                "<body><section id=\"tab-1\"></section><section id=\"tab-2\"></section><script src=\"app.js\"></script></body>",
            ),
            (
                "app.js",
                "function show(n) {\n  document.querySelector(`#tab-${n}`).hidden = false;\n  document.querySelector('#tab-' + n).focus();\n}\nshow(1);\n",
            ),
        ]);
    }

    #[test]
    fn an_id_from_data_or_a_suffix_template_is_no_finding() {
        clean(&[
            (
                "index.html",
                "<body><script src=\"app.js\"></script></body>",
            ),
            (
                "app.js",
                "const tabs = ['home', 'settings'];\ntabs.forEach(t => { const d = document.createElement('div'); d.id = t; document.body.append(d); });\ndocument.getElementById('home').classList.add('active');\nfor (const p of ['left']) document.body.innerHTML += `<div id=\"${p}-panel\"></div>`;\ndocument.getElementById('left-panel').remove();\n",
            ),
        ]);
    }

    #[test]
    fn markup_in_a_server_template_gives_ids() {
        clean(&[
            ("public/index.html", "<body><h1 id=\"t\"></h1></body>"),
            ("views/home.ejs", "<div id=\"cart\"></div>"),
            (
                "public/js/app.js",
                "document.getElementById('cart').textContent = '0';\n",
            ),
        ]);
    }

    #[test]
    fn an_optional_or_compiled_python_module_is_no_finding() {
        clean(&[
            (
                "proj/settings/__init__.py",
                "from .base import *\ntry:\n    from .local import *\nexcept ImportError:\n    pass\n",
            ),
            ("proj/settings/base.py", "DEBUG = False\n"),
        ]);
        clean(&[
            ("pkg/__init__.py", "from ._speedups import fast\n"),
            ("pkg/_speedups.pyx", "def fast():\n    pass\n"),
        ]);
    }

    #[test]
    fn an_import_in_a_comment_or_a_string_is_no_finding() {
        clean(&[(
            "a.js",
            "// import { old } from './old';\n/* import { b } from './gone'; */\nexport const tpl = `import { App } from './App';`;\nconst s = \"require('./nope')\";\n",
        )]);
    }

    #[test]
    fn a_comment_in_import_braces_and_every_export_form_are_no_finding() {
        clean(&[
            (
                "m.js",
                "export function *gen() {}\nexport const a = 1, b = 2;\nexport let c, d;\n",
            ),
            (
                "a.js",
                "import {\n  a, // the a\n  b, /* and b */\n  gen,\n  d,\n} from './m.js';\nconsole.log(a, b, gen, d);\n",
            ),
        ]);
    }

    #[test]
    fn a_folder_above_the_script_does_not_shadow_the_standard_library() {
        clean(&[
            ("app/email.py", "def send():\n    pass\n"),
            (
                "app/handlers/notify.py",
                "from email.mime.text import MIMEText\nfrom email import message_from_string\n",
            ),
            ("tools/json.py", "def load():\n    pass\n"),
            ("tools/sub/x.py", "from json import dumps\n"),
        ]);
    }

    /// A link the server or a build resolves elsewhere is no finding while a file of its name is
    /// somewhere in the project: a template's `static/` path, or a page under `src/` linking
    /// what a build copies beside it.
    #[test]
    fn a_link_to_a_name_the_project_has_elsewhere_is_no_finding() {
        clean(&[
            (
                "templates/index.html",
                "<link rel=\"stylesheet\" href=\"static/style.css\"><script src=\"app.js\"></script>",
            ),
            ("static/style.css", "body { color: red; }\n"),
            ("src/app.js", "console.log(1);\n"),
        ]);
    }

    /// A lookup in a comment, the script's or the page's, looks nothing up; a selector of an id
    /// a script gives an element from its data is there.
    #[test]
    fn a_lookup_in_a_comment_or_of_an_id_from_data_is_no_finding() {
        clean(&[
            (
                "index.html",
                "<div id=\"app\"></div><!-- document.getElementById('old') -->\n<script src=\"a.js\"></script>",
            ),
            (
                "a.js",
                "// document.getElementById('gone').remove();\n/* querySelector('.nope') */\n\
                 const app = document.getElementById('app');\nconst panels = [\"home\", \"about\"];\n\
                 for (const p of panels) { const d = document.createElement('div'); d.id = p; app.appendChild(d); }\n\
                 document.querySelector(\"#home\").hidden = false;\n",
            ),
        ]);
    }

    #[test]
    fn a_commented_link_or_a_bare_host_is_no_finding() {
        clean(&[
            (
                "index.html",
                "<head><!-- <link rel=\"stylesheet\" href=\"old.css\"> --><link rel=\"stylesheet\" href=\"style.css\"></head><body><a href=\"www.example.com\">x</a></body>",
            ),
            (
                "style.css",
                "/* background: url(old.png); */\nbody { color: red; }\n",
            ),
        ]);
    }

    #[test]
    fn python_on_one_line_or_in_a_docstring_is_no_finding() {
        clean(&[
            ("pkg/__init__.py", ""),
            ("pkg/b.py", "def f():\n    pass\n"),
            ("pkg/c.py", "def g():\n    pass\n"),
            (
                "pkg/a.py",
                "\"\"\"Usage:\n    from .widgets import Gauge\n\"\"\"\nfrom .b import f; from .c import g\n",
            ),
        ]);
    }

    #[test]
    fn a_rust_mod_under_a_long_cfg_or_in_a_string_is_no_finding() {
        clean(&[(
            "src/lib.rs",
            "#[cfg(all(\n    feature = \"gpu\",\n))]\nmod gpu;\n\npub const S: &str = r\"\nmod generated;\n\";\n",
        )]);
    }

    /// A path the model can write: with the importer's kind, never a bare name.
    #[test]
    fn a_missing_import_is_told_as_a_file_with_its_kind() {
        let f = one(&[(
            "src/main.ts",
            "import { x } from './greet';\nconsole.log(x);\n",
        )]);
        assert!(f.message.ends_with("write src/greet.ts"), "{}", f.message);
        let f = one(&[("src/main.js", "import x from './';\nconsole.log(x);\n")]);
        assert!(f.message.ends_with("write src/index.js"), "{}", f.message);
    }

    /// What .gitignore ignores is not read, so a vendored folder never fills the bounds; each
    /// folder is read in name order; a second check reads a changed file again.
    #[test]
    fn a_walk_skips_what_gitignore_ignores_and_rereads_a_change() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let make = |name: &str, text: &str| {
            let path = dir.path().join(name);
            fs::create_dir_all(path.parent().expect("a parent")).expect("a folder");
            fs::write(path, text).expect("a file");
        };
        make(".gitignore", "vendor/\n*.gen.js\n");
        make("vendor/lib.js", "");
        make("b.js", "");
        make("a.gen.js", "");
        make("a.js", "");
        let (files, complete) = listed(dir.path(), readable);
        assert!(complete);
        assert_eq!(files.keys().collect::<Vec<_>>(), ["a.js", "b.js"]);
        make(
            "index.html",
            "<div id=\"x\"></div><script src=\"a.js\"></script>",
        );
        make("a.js", "document.getElementById('x');\n");
        let changed = BTreeSet::from(["a.js".to_string()]);
        assert_eq!(check(dir.path(), &changed), vec![]);
        make("a.js", "document.getElementById('y');\n");
        assert_eq!(check(dir.path(), &changed).len(), 1);
        forget(dir.path());
    }

    /// A virtualenv is no project code, and a project past the bounds still checks the changed
    /// files.
    #[test]
    fn a_virtualenv_is_skipped_and_a_large_project_still_checks_its_changes() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let make = |name: &str, text: &str| {
            let path = dir.path().join(name);
            fs::create_dir_all(path.parent().expect("a parent")).expect("a folder");
            fs::write(path, text).expect("a file");
        };
        make("env/pyvenv.cfg", "home = /usr\n");
        make("myenv/pyvenv.cfg", "home = /usr\n");
        make("env/lib/x.py", "x = 1\n");
        make("myenv/lib/y.py", "y = 1\n");
        make("app.py", "print(1)\n");
        let (files, _) = listed(dir.path(), readable);
        assert_eq!(files.keys().collect::<Vec<_>>(), ["app.py"]);
        for n in 0..=MAX_FILES {
            make(&format!("gen/{n}.js"), "");
        }
        make(
            "main.js",
            "import { a } from './gone.js';\nconsole.log(a);\n",
        );
        let changed = BTreeSet::from(["main.js".to_string()]);
        let f = check(dir.path(), &changed);
        assert_eq!(f.len(), 1, "{f:#?}");
        assert_eq!((f[0].file.as_str(), f[0].line), ("main.js", 1));
    }
}
