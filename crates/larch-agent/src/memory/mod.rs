//! File memory under `<data_dir>/agent/memory/`: `USER.md` and `MEMORY.md` go into every system
//! prompt, one fact per file in `facts/`, append-only notes in `daily/`. The files are the source
//! of truth; `index.sqlite` is derived from them.

pub mod capture;
pub(crate) mod dates;
mod index;

use std::{
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};

use larch::GenParams;
use serde::{Deserialize, Serialize};

pub(crate) use index::fts_query;
pub use index::{Hit, RECALL_HITS, Recall, rebuild, recall, recall_with, today};

use crate::{parse::Family, write_atomic};

/// The day of a `YYYY-MM-DD` date, in days since 1970: a `Recall`'s `today`.
pub fn day(date: &str) -> Option<i64> {
    dates::iso(date).filter(|_| date.len() == 10)
}

/// Where an index row comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// A line of a fact in `facts/`.
    Fact,
    /// A line of a daily note.
    Daily,
    /// A message of a chat transcript.
    Transcript,
    /// A key phrase a model turn extracted from a daily note line, pointing at that line.
    Key,
}

impl Source {
    /// The name the index and the recall tool use.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Daily => "daily",
            Self::Transcript => "transcript",
            Self::Key => "key",
        }
    }

    /// The source `name` names.
    pub fn named(name: &str) -> Option<Source> {
        ALL.into_iter().find(|s| s.as_str() == name)
    }
}

/// What automatic recall reads: the curated facts, the daily notes and their keys, never a
/// transcript, which holds what was said and not what is true.
pub const AUTO: [Source; 3] = [Source::Fact, Source::Daily, Source::Key];
/// Every source.
pub const ALL: [Source; 4] = [Source::Fact, Source::Daily, Source::Transcript, Source::Key];

/// Lines of `USER.md` in the system prompt.
pub const USER_LINES: usize = 30;
/// Bytes of `MEMORY.md` lines in the system prompt, one line per fact: with `USER.md`, near
/// 1,500 tokens. The newest lines that fit go in; the store itself has no cap.
pub const MEMORY_BYTES: usize = 6 * 1024;
/// The fact types the front matter allows.
pub const TYPES: [&str; 4] = ["user", "project", "preference", "reference"];
// A whole fact of two or three sentences fits; the full text is in its file.
const SENTENCE_BYTES: usize = 240;
// Six English words fit; text without spaces (Chinese, a long token) is one word, and file names
// stop at 255 bytes.
const SLUG_BYTES: usize = 48;

/// A memory failure; `Display` is the message the model gets back.
#[derive(Debug)]
pub enum MemoryError {
    /// No fact by that slug (or the slug is not a plain name).
    NoFact(String),
    /// A type outside `TYPES`.
    BadType(String),
    /// The text has no letters or digits to name it by.
    Empty,
    /// The same fact is in memory already, by this slug.
    Known(String),
    /// A key turn's reply held no key phrase; the reply's start.
    NoKeys(String),
    /// The fact is not as a receipt saved it (edited, saved again or forgotten since), so `undo`
    /// leaves it.
    Changed(String),
    /// The fact is not as the edit found it (a capture or another window changed it since), so
    /// `edit` leaves it.
    Edited(String),
    /// An edit's text holds a line break or the user's words a capture keeps (`Said:`).
    NotOneSentence,
    /// Memory is off (`set_enabled`).
    Off,
    /// A file could not be read or written.
    Io(io::Error),
    /// The index could not be read or written.
    Index(rusqlite::Error),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoFact(slug) => write!(f, "no fact named '{slug}'"),
            Self::BadType(kind) => write!(f, "type '{kind}' is not one of {}", TYPES.join(", ")),
            Self::Empty => f.write_str("the fact has no words"),
            Self::Known(slug) => write!(f, "already in memory as [{slug}]; nothing to save"),
            Self::NoKeys(reply) => write!(f, "no key phrases in the reply '{reply}'"),
            Self::Changed(slug) => write!(
                f,
                "the fact [{slug}] has changed since it was saved, so it was not undone"
            ),
            Self::Edited(slug) => write!(
                f,
                "the fact [{slug}] changed since you opened it, so your edit was not saved; \
                 open it again to edit what it says now"
            ),
            Self::NotOneSentence => {
                f.write_str("a fact is one sentence on one line, without a 'Said:' part")
            }
            Self::Off => f.write_str("memory is off: the user turned it off in the app"),
            Self::Io(e) => write!(f, "memory: {e}"),
            Self::Index(e) => write!(f, "memory index: {e}"),
        }
    }
}

impl std::error::Error for MemoryError {}

impl From<io::Error> for MemoryError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<rusqlite::Error> for MemoryError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Index(e)
    }
}

fn dir(data_dir: &Path) -> PathBuf {
    data_dir.join("agent").join("memory")
}

/// Whether memory is on: no `agent/memory/off` file. Off, a chat has no memory block, no memory
/// tools, no capture and no daily notes.
pub fn enabled(data_dir: &Path) -> bool {
    !dir(data_dir).join("off").exists()
}

/// `MemoryError::Off` when memory is off.
pub(crate) fn check_on(data_dir: &Path) -> Result<(), MemoryError> {
    if enabled(data_dir) {
        Ok(())
    } else {
        Err(MemoryError::Off)
    }
}

/// Turns memory on or off, for every chat on the data folder from its next turn (`rev`).
/// Turning it on again captures nothing said while it was off.
pub fn set_enabled(data_dir: &Path, on: bool) -> Result<(), MemoryError> {
    let _lock = lock(data_dir)?;
    let flag = dir(data_dir).join("off");
    if on == enabled(data_dir) {
        return Ok(());
    }
    match on {
        true => fs::remove_file(&flag)?,
        false => fs::write(
            &flag,
            "memory is off; delete this file or switch it on in the app\n",
        )?,
    }
    bump(data_dir)
}

/// The memory's revision: `agent/memory/.rev`, a counter that goes up on a delete, an edit, an
/// undo and an on/off, but not on an add. An open chat reads it at each turn start, one small
/// read, and rebuilds its prompt only after it moved (and only when its memory block changed).
/// 0 when there is none.
pub fn rev(data_dir: &Path) -> u64 {
    (fs::read_to_string(dir(data_dir).join(".rev")).ok())
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(0)
}

/// Moves `rev` on. Called under the lock.
fn bump(data_dir: &Path) -> Result<(), MemoryError> {
    let path = dir(data_dir).join(".rev");
    Ok(write_atomic(
        &path,
        &format!("{}\n", rev(data_dir) + 1),
        true,
    )?)
}

/// Holds `agent/memory/.lock` until dropped, so chats that share the data folder take turns at
/// `MEMORY.md`, the facts and the index; the lock ends with its process, so a crash leaves none.
fn lock(data_dir: &Path) -> io::Result<File> {
    fs::create_dir_all(dir(data_dir))?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(dir(data_dir).join(".lock"))?;
    file.lock()?;
    Ok(file)
}

fn fact_path(data_dir: &Path, slug: &str) -> PathBuf {
    dir(data_dir).join("facts").join(format!("{slug}.md"))
}

/// A slug is a file name of lower-case letters, digits and dashes, as `slug` makes it; anything
/// else is never a path, and never another case of a slug.
fn is_slug(slug: &str) -> bool {
    !slug.is_empty()
        && slug
            .chars()
            .all(|c| c == '-' || c.is_alphanumeric() && c.to_lowercase().eq([c]))
}

/// What `remember` checks before it reads the memory: a known type, and words to name the fact
/// by. Returns the slug they make.
pub(crate) fn check_fact(text: &str, kind: &str) -> Result<String, MemoryError> {
    if !TYPES.contains(&kind) {
        return Err(MemoryError::BadType(kind.into()));
    }
    let slug = slug(text.trim());
    if slug.is_empty() {
        return Err(MemoryError::Empty);
    }
    Ok(slug)
}

/// The slug of the fact in memory whose text is `text` and type `kind`, so saving it again asks
/// the user nothing. A fact without its `MEMORY.md` line is not known: `remember` adds it back.
pub(crate) fn known(
    data_dir: &Path,
    text: &str,
    kind: &str,
) -> Result<Option<String>, MemoryError> {
    let Some(slug) = index::near_duplicate(data_dir, text.trim())? else {
        return Ok(None);
    };
    let old = read_lossy(&fact_path(data_dir, &slug))?;
    let index = read_lossy(&dir(data_dir).join("MEMORY.md"))?;
    let index = index.lines().filter(|l| !l.trim().is_empty());
    // Only a line the prompt shows: a fact outside the budget is saved again, which moves its
    // line to the end, into the prompt.
    let same = has_front_matter(&old)
        && body(&old) == text.trim()
        && front_matter(&old).contains(&("type", kind))
        && shown(index.map(String::from).collect())
            .iter()
            .any(|l| line_slug(l) == Some(&slug));
    Ok(same.then_some(slug))
}

/// What `forget` checks before it reads the memory: `slug` could name a fact.
pub(crate) fn check_slug(slug: &str) -> Result<(), MemoryError> {
    if is_slug(slug) {
        Ok(())
    } else {
        Err(MemoryError::NoFact(slug.into()))
    }
}

/// The first six words of `text`, lowercased and joined by dashes, within `SLUG_BYTES`.
pub(crate) fn slug(text: &str) -> String {
    let words: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .map(|w| {
            w.chars()
                .flat_map(char::to_lowercase)
                .filter(|c| c.is_alphanumeric())
                .collect()
        })
        .filter(|w: &String| !w.is_empty())
        .take(6)
        .collect();
    let slug = words.join("-");
    let cut = slug.floor_char_boundary(SLUG_BYTES);
    slug[..cut].trim_end_matches('-').to_owned()
}

/// `slug`, or `slug-2`, `slug-3`... when another fact holds it.
fn free_slug(data_dir: &Path, slug: &str) -> String {
    std::iter::once(slug.to_owned())
        .chain((2..).map(|n| format!("{slug}-{n}")))
        .find(|s| !fact_path(data_dir, s).exists())
        .expect("an unbounded range has a free slug")
}

/// `text` on one line, cut at a word before `SENTENCE_BYTES`.
fn sentence(text: &str) -> String {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.len() <= SENTENCE_BYTES {
        return line;
    }
    let cut = line.floor_char_boundary(SENTENCE_BYTES);
    let cut = line[..cut].rfind(' ').unwrap_or(cut);
    format!("{}...", &line[..cut])
}

/// The slug of a `MEMORY.md` line `- [slug] sentence`; some Windows editors put a byte-order
/// mark in front of the first.
fn line_slug(line: &str) -> Option<&str> {
    line.trim_start_matches('\u{feff}')
        .strip_prefix("- [")?
        .split_once(']')
        .map(|(slug, _)| slug)
}

/// The `key: value` pairs of a fact file's front matter.
fn front_matter(text: &str) -> Vec<(&str, &str)> {
    let mut lines = text.lines();
    if lines.next() != Some("---") {
        return Vec::new();
    }
    lines
        .take_while(|l| *l != "---")
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim(), v.trim()))
        .collect()
}

/// Writes a memory file where `fs::write` would: through a link (MEMORY.md kept with the user's
/// dotfiles, say), also to a file the link names that does not exist yet, so the link stays.
fn write_through(path: &Path, text: &str) -> io::Result<()> {
    let mut path = path.to_owned();
    // The kernel's own bound: a loop of links fails instead of spinning.
    for _ in 0..40 {
        match fs::symlink_metadata(&path) {
            Ok(meta) if meta.is_symlink() => {
                let to = fs::read_link(&path)?;
                path = path
                    .parent()
                    .map_or_else(|| to.clone(), |dir| dir.join(&to));
            }
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
            _ => return write_atomic(&path, text, true),
        }
    }
    Err(io::Error::other("too many levels of links"))
}

fn joined(lines: &[impl AsRef<str>]) -> String {
    lines.iter().map(|l| format!("{}\n", l.as_ref())).collect()
}

fn read_lossy(path: &Path) -> io::Result<String> {
    match fs::read(path) {
        Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(String::new()),
        Err(e) => Err(e),
    }
}

/// `MEMORY.md`'s text, for a write that keeps its other lines: one that is not UTF-8 is refused,
/// as those lines would come back with U+FFFD in place of their bytes. A leading byte order mark
/// is dropped, so it never ends up in front of another line when lines move.
fn memory_md_text(path: &Path) -> io::Result<String> {
    match fs::read_to_string(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(String::new()),
        Err(e) if e.kind() == io::ErrorKind::InvalidData => Err(io::Error::new(
            e.kind(),
            "MEMORY.md is not UTF-8 text; ask the user to save it as UTF-8",
        )),
        read => read.map(|t| t.strip_prefix('\u{feff}').map_or(t.clone(), String::from)),
    }
}

/// `USER.md` within `USER_LINES` non-blank lines, then the newest `MEMORY.md` lines that fit
/// `MEMORY_BYTES`, in file order, each under its heading; the index ends with how to name a fact
/// to `forget`, which models otherwise give a path. A file that is missing, cannot be read or
/// holds no lines is left out, so the prompt still opens.
pub fn system_prefix(data_dir: &Path) -> String {
    system_prefix_fitting(data_dir, |_| true)
}

/// `system_prefix` within what `fits` takes, as the history drops its oldest exchanges first:
/// the oldest `MEMORY.md` lines go first, then, with none left, the last `USER.md` lines; empty
/// when not even one `USER.md` line fits. For a context the block would fill: a chat whose
/// prompt leaves its first step no room fails every turn.
pub fn system_prefix_fitting(data_dir: &Path, fits: impl Fn(&str) -> bool) -> String {
    if !enabled(data_dir) {
        return String::new();
    }
    let lines = |name: &str| {
        let text = read_lossy(&dir(data_dir).join(name)).unwrap_or_default();
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(String::from)
            .collect::<Vec<_>>()
    };
    let mut user = lines("USER.md");
    user.truncate(USER_LINES);
    let index = shown(lines("MEMORY.md"));
    let block = |user: &[String], index: &[String]| {
        let part = |lines: &[String], heading: &str, after: &str| {
            (!lines.is_empty()).then(|| format!("{heading}\n{}{after}", lines.join("\n")))
        };
        let forget = "\n`forget` takes the slug in brackets, not a path.";
        let parts = [
            part(user, "## About the user", ""),
            part(index, "## Memory index", forget),
        ];
        parts.into_iter().flatten().collect::<Vec<_>>().join("\n\n")
    };
    let whole = block(&user, &index);
    if fits(&whole) {
        return whole;
    }
    // The most of `n` lines, kept by `keep`, that fit: the block only grows with a line.
    let most = |n: usize, keep: &dyn Fn(usize) -> String| {
        let (mut lo, mut hi) = (0, n);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            match fits(&keep(mid)) {
                true => lo = mid,
                false => hi = mid - 1,
            }
        }
        keep(lo)
    };
    let newest = |k: usize| block(&user, &index[index.len() - k..]);
    if fits(&newest(0)) {
        return most(index.len(), &newest);
    }
    let first = |k: usize| block(&user[..k], &[]);
    match !user.is_empty() && fits(&first(1)) {
        true => most(user.len(), &first),
        false => String::new(),
    }
}

/// The newest of `lines` (`MEMORY.md`'s non-blank lines) that fit `MEMORY_BYTES`, in file
/// order: what the system prompt shows. New and updated facts go last, so the newest are last.
fn shown(lines: Vec<String>) -> Vec<String> {
    let mut index = Vec::new();
    let mut left = MEMORY_BYTES;
    for line in lines.into_iter().rev() {
        if line.len() < left {
            left -= line.len() + 1;
            index.push(line);
        }
    }
    index.reverse();
    index
}

/// A daily note line: its day, `YYYY-MM-DD`, its line in that day's file, and its text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Noted {
    /// The day, which names the file.
    pub day: String,
    /// The line, counted from 1.
    pub line: u32,
    /// The line as the file holds it.
    pub text: String,
}

/// Appends one line to today's `daily/YYYY-MM-DD.md` (UTC), prefixed with the time, and returns
/// where it went: under the lock, so keys of two chats' notes point at the right lines. Memory
/// off writes nothing (`None`), checked under the lock, so no worker notes what recall would find
/// once memory is on again.
pub fn note(data_dir: &Path, line: &str) -> io::Result<Option<Noted>> {
    let now = larch::bench::utc_now();
    let _lock = lock(data_dir)?;
    if !enabled(data_dir) {
        return Ok(None);
    }
    let daily = dir(data_dir).join("daily");
    fs::create_dir_all(&daily)?;
    let path = daily.join(format!("{}.md", &now[..10]));
    let old = read_lossy(&path)?;
    let text = format!(
        "- {}Z {}",
        &now[11..16],
        line.split_whitespace().collect::<Vec<_>>().join(" ")
    );
    // A last line without its end (an edit by hand) is ended, so this one is a line of its own.
    let open = !old.is_empty() && !old.ends_with('\n');
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    file.write_all(format!("{}{text}\n", if open { "\n" } else { "" }).as_bytes())?;
    Ok(Some(Noted {
        day: now[..10].to_owned(),
        line: u32::try_from(old.lines().count() + 1).unwrap_or(u32::MAX),
        text,
    }))
}

/// Every line of the daily notes, oldest day first: what a bench extracts keys from.
pub fn notes(data_dir: &Path) -> io::Result<Vec<Noted>> {
    let mut days: Vec<String> = Vec::new();
    match fs::read_dir(dir(data_dir).join("daily")) {
        Ok(entries) => {
            for entry in entries {
                let name = entry?.file_name();
                let day = name.to_str().and_then(|n| n.strip_suffix(".md"));
                days.extend(day.filter(|d| self::day(d).is_some()).map(String::from));
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    days.sort();
    let mut notes = Vec::new();
    for day in days {
        let text = read_lossy(&dir(data_dir).join("daily").join(format!("{day}.md")))?;
        for (i, line) in text.lines().enumerate() {
            if !line.trim().is_empty() {
                notes.push(Noted {
                    day: day.clone(),
                    line: u32::try_from(i + 1).unwrap_or(u32::MAX),
                    text: line.trim().to_owned(),
                });
            }
        }
    }
    Ok(notes)
}

/// The notice for an answer that claims a save no `remember` or `write_file` call made.
pub(crate) const NOTHING_SAVED: &str =
    "nothing was saved: the answer says so, but no remember or write_file call ran";

/// The notice for an answer that claims a save when its `remember` or `write_file` calls failed:
/// a failed call may have written part of its text, so it is not called unsaved.
pub(crate) const SAVE_FAILED: &str = "the answer says it saved, but its remember or write_file \
call failed; whether anything was saved is not known";

/// What an answer says when it claims a save, as whole words; `. ` marks a sentence's start, so
/// "Noted." is a claim and "as noted above" is not. Weaker models claim saves they never make.
const SAVE_CLAIMS: [&str; 16] = [
    "i've saved",
    "i have saved",
    "i saved",
    "saved that",
    "saved it",
    "saved to memory",
    "saved in memory",
    "i'll remember",
    "i will remember",
    "i've remembered",
    "i've noted",
    "i have noted",
    ". noted",
    "i've made a note",
    "i'll make a note",
    "added to memory",
];

/// Whether `text` holds one of `SAVE_CLAIMS`, case and the apostrophe's form aside.
pub(crate) fn claims_a_save(text: &str) -> bool {
    let mut words = String::from(" . ");
    for c in text.to_lowercase().chars() {
        match c {
            '\u{2019}' | '\'' => words.push('\''),
            '.' | '!' | '?' | ';' | ':' | '\n' => words.push_str(" . "),
            c if c.is_alphanumeric() => words.push(c),
            _ => words.push(' '),
        }
    }
    let words = format!(
        " {} ",
        words.split_whitespace().collect::<Vec<_>>().join(" ")
    );
    SAVE_CLAIMS
        .iter()
        .any(|claim| words.contains(&format!(" {claim} ")))
}

/// Key phrases one note line gets at most.
pub const MAX_KEYS: usize = 3;
/// Words a key phrase has at most: a longer line is a summary, and summaries as keys made an 8B
/// model's recall worse than no keys at all.
const KEY_WORDS: usize = 6;
const KEY_BYTES: usize = 80;

/// The key turn's system prompt.
pub const KEY_SYSTEM: &str = "You index a personal assistant's notes for search. Reply with up to \
three short key phrases, two to five words each, that someone might search for to find the note: \
names, things, places, amounts. One phrase per line, nothing else.";

/// The key turn's parameters, its user message being the note line: a worker step's
/// (`run::Kind::Worker`: greedy, no tools, thinking off where the family thinks, so 64 tokens are
/// all phrases), seeded as every worker request is, 64 tokens, in the data folder's cache
/// `namespace`.
pub fn key_params(family: Option<Family>, namespace: u64) -> GenParams {
    let worker = crate::run::params(
        &crate::tool::Registry::builtin(),
        &[],
        family,
        crate::run::Kind::Worker,
        Default::default(),
        namespace,
    );
    GenParams {
        max_tokens: 64,
        seed: Some(0),
        ..worker
    }
}

/// The key phrases of a key turn's reply, at most `MAX_KEYS`: one per line or per `, `, list
/// marks, quotes and a closing stop taken off. A heading (`Keys:`), a fence, markup, a call and a
/// line of more than `KEY_WORDS` words are no keys; a `cut` reply loses its unfinished last line.
/// No key at all is an error, so a turn that said nothing useful reaches the log.
pub fn parse_keys(reply: &str, cut: bool) -> Result<Vec<String>, MemoryError> {
    let after_think = reply.rsplit("</think>").next().unwrap_or(reply);
    let mut lines: Vec<&str> = after_think.lines().collect();
    if cut {
        lines.pop();
    }
    let mut keys: Vec<String> = Vec::new();
    // ", " and not ",": "1,850 USD" is one key.
    for part in lines
        .iter()
        .flat_map(|l| l.split("; ").flat_map(|p| p.split(", ")))
    {
        let mut key = part.trim();
        if key.starts_with("```") {
            continue;
        }
        for mark in ["- [ ] ", "- [x] ", "- ", "* ", "• "] {
            key = key.strip_prefix(mark).unwrap_or(key);
        }
        let digits = key.find(|c: char| !c.is_ascii_digit()).unwrap_or(0);
        if digits > 0 && (key[digits..].starts_with(". ") || key[digits..].starts_with(") ")) {
            key = &key[digits + 1..];
        }
        let key = key
            .trim()
            .trim_matches(|c| matches!(c, '"' | '\'' | '`' | '*'))
            .trim_end_matches(['.', '!'])
            .trim();
        let words = key.split_whitespace().count();
        if key.is_empty()
            || key.ends_with(':')
            || key.starts_with(['{', '<', '[', '`'])
            || key.eq_ignore_ascii_case("none")
            || words > KEY_WORDS
            || keys.iter().any(|k| k.eq_ignore_ascii_case(key))
        {
            continue;
        }
        keys.push(
            key[..key.floor_char_boundary(KEY_BYTES)]
                .trim_end()
                .to_owned(),
        );
        if keys.len() == MAX_KEYS {
            break;
        }
    }
    if keys.is_empty() {
        return Err(MemoryError::NoKeys(sentence(reply)));
    }
    Ok(keys)
}

/// Saves `keys` as the key rows of the note line `noted`, in `keys/<day>.txt`, one
/// `<line> <phrase>` per line, in place of any that line had. The note itself is left as it is.
pub fn save_keys(data_dir: &Path, noted: &Noted, keys: &[String]) -> Result<(), MemoryError> {
    let _lock = lock(data_dir)?;
    let folder = dir(data_dir).join("keys");
    fs::create_dir_all(&folder)?;
    let path = folder.join(format!("{}.txt", noted.day));
    let at = format!("{} ", noted.line);
    let mut lines: Vec<String> = read_lossy(&path)?
        .lines()
        .filter(|l| !l.starts_with(&at))
        .map(String::from)
        .collect();
    lines.extend(keys.iter().map(|k| {
        let k = k.split_whitespace().collect::<Vec<_>>().join(" ");
        format!("{at}{k}")
    }));
    Ok(write_through(&path, &joined(&lines))?)
}

/// What `remember` saved: the fact's slug, and the text it had before when it was updated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Saved {
    /// The fact's slug.
    pub slug: String,
    /// The text the fact had before, when a near-duplicate was updated.
    pub old: Option<String>,
    /// Its type and source before, when it was updated.
    pub was: Option<Was>,
    /// Its `created`, which an update keeps.
    pub created: String,
}

/// A fact's type and source before an update, which `undo` puts back.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Was {
    /// One of `TYPES`.
    pub kind: String,
    /// Where it came from.
    pub source: String,
}

impl Saved {
    /// What `remember` and friends return, from the fact's old file (`None` for a new fact).
    fn of(slug: String, old: Option<&str>, created: String) -> Saved {
        Saved {
            old: old.map(body),
            was: old.map(|f| Was {
                kind: meta(f, "type").unwrap_or("user").to_owned(),
                source: meta(f, "source").unwrap_or("the desktop app").to_owned(),
            }),
            created,
            slug,
        }
    }

    /// The receipt of this save of `text`.
    pub fn receipt(self, text: &str) -> Receipt {
        Receipt {
            slug: self.slug,
            text: text.trim().to_owned(),
            old: self.old,
            was: self.was,
            created: self.created,
        }
    }
}

/// A fact in memory, as `list` reads it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Fact {
    /// The slug, which names its file.
    pub slug: String,
    /// The whole text.
    pub text: String,
    /// One of `TYPES`.
    pub kind: String,
    /// When it was first saved.
    pub created: String,
    /// Where it came from: a chat ID and turn, or the desktop app.
    pub source: String,
}

/// Saves `text` as a fact of type `kind`. A near-duplicate fact (the same words, or each text
/// covering most of the other's BM25 weight) is updated instead of a new one being added; a
/// different fact that holds the slug moves this one to `slug-2`. `source` names where the fact
/// came from (the chat ID and turn).
pub fn remember(
    data_dir: &Path,
    text: &str,
    kind: &str,
    source: &str,
) -> Result<Saved, MemoryError> {
    let new = check_fact(text, kind)?;
    let text = text.trim();
    let _lock = lock(data_dir)?;
    let (slug, old) = match index::near_duplicate(data_dir, text)? {
        Some(slug) => {
            let old = read_lossy(&fact_path(data_dir, &slug))?;
            (slug, Some(old))
        }
        None => (free_slug(data_dir, &new), None),
    };
    let file = old.as_deref().unwrap_or_default();
    let keep = |key, or: &str| meta(file, key).unwrap_or(or).to_owned();
    let created = keep("created", &larch::bench::utc_now());
    let confidence = keep("confidence", "medium");
    write_fact(data_dir, &slug, kind, &created, source, &confidence, text)?;
    // A replace is an edit: open chats drop the old line at their next turn. An add keeps them.
    if old.is_some() {
        bump(data_dir)?;
    }
    Ok(Saved::of(slug, old.as_deref(), created))
}

/// Saves a captured `text` (a sentence, then `Said:` and the user's) as a fact of type `kind`:
/// in place of a fact whose sentence has the same words however it names the user
/// (`capture::fact_words`), else as a new fact. Never on BM25: the
/// capture model saw the facts like it and chose not to replace one, and a wrong merge loses a
/// fact where a missed one leaves a duplicate line.
pub(crate) fn save_captured(
    data_dir: &Path,
    text: &str,
    kind: &str,
    source: &str,
) -> Result<Saved, MemoryError> {
    let text = text.trim();
    // Named by its sentence, not the user's words after it.
    let new = check_fact(capture::sentence_of(text), kind)?;
    let sentence_of = |t: &str| capture::sentence_of(t).to_owned();
    let words = capture::fact_words(&sentence_of(text));
    let _lock = lock(data_dir)?;
    // Under the lock: a switch to off between the capture's check and here saves nothing.
    check_on(data_dir)?;
    let same = list(data_dir)?
        .into_iter()
        .find(|f| capture::fact_words(&sentence_of(&f.text)) == words);
    let (slug, old) = match same {
        Some(f) => {
            let old = read_lossy(&fact_path(data_dir, &f.slug))?;
            (f.slug, Some(old))
        }
        None => (free_slug(data_dir, &new), None),
    };
    let file = old.as_deref().unwrap_or_default();
    let keep = |key, or: &str| meta(file, key).unwrap_or(or).to_owned();
    let created = keep("created", &larch::bench::utc_now());
    write_fact(data_dir, &slug, kind, &created, source, "medium", text)?;
    // Its `MEMORY.md` line changed (punctuation, case): open chats show the new one.
    if old
        .as_deref()
        .is_some_and(|o| sentence_of(&body(o)) != sentence_of(text))
    {
        bump(data_dir)?;
    }
    Ok(Saved::of(slug, old.as_deref(), created))
}

/// Puts `text` in place of the fact `slug`'s, keeping its slug and front matter, and returns the
/// old text.
pub fn update(data_dir: &Path, slug: &str, text: &str) -> Result<String, MemoryError> {
    one_sentence(text)?;
    let saved = updated(data_dir, slug, text, None)?;
    Ok(saved.old.unwrap_or_default())
}

/// An edit's text: one line, and none of the `Said:` a capture adds after the sentence, which
/// would put the user's words in every system prompt.
fn one_sentence(text: &str) -> Result<(), MemoryError> {
    match text.trim().contains(['\n', '\r']) || text.contains(capture::SAID.trim_start()) {
        true => Err(MemoryError::NotOneSentence),
        false => Ok(()),
    }
}

/// The app's Edit: puts `sentence` in place of the fact `slug`'s sentence, keeping the user's
/// words a capture saved after it (`Said:`), its slug and front matter. Only while the fact's
/// text is still `expected`, the text the edit started from, checked under the lock: else
/// `MemoryError::Edited`, so an edit never overwrites a correction saved since. The same
/// sentence writes nothing and moves no `rev`: `Ok(false)`.
pub fn edit(
    data_dir: &Path,
    slug: &str,
    expected: &str,
    sentence: &str,
) -> Result<bool, MemoryError> {
    check_slug(slug)?;
    one_sentence(sentence)?;
    let sentence = sentence.trim();
    if self::slug(sentence).is_empty() {
        return Err(MemoryError::Empty);
    }
    let _lock = lock(data_dir)?;
    if !listed(&dir(data_dir).join("facts"), &format!("{slug}.md"))? {
        return Err(MemoryError::Edited(slug.into()));
    }
    let file = read_lossy(&fact_path(data_dir, slug))?;
    let now = body(&file);
    if now.trim() != expected.trim() {
        return Err(MemoryError::Edited(slug.into()));
    }
    if capture::sentence_of(&now).trim() == sentence {
        return Ok(false);
    }
    let text = match now.split_once(capture::SAID) {
        Some((_, said)) => format!("{sentence}{}{said}", capture::SAID),
        None => sentence.to_owned(),
    };
    let keep = |key, or: &str| meta(&file, key).unwrap_or(or).to_owned();
    write_fact(
        data_dir,
        slug,
        &keep("type", "user"),
        &keep("created", &larch::bench::utc_now()),
        &keep("source", "the desktop app"),
        &keep("confidence", "medium"),
        &text,
    )?;
    bump(data_dir)?;
    Ok(true)
}

/// A fact as `forget_kept` removed it: all `restore` needs to put it back as it was.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gone {
    /// Its slug.
    pub slug: String,
    /// Its text, as the file held it.
    pub text: String,
    /// Its type.
    pub kind: String,
    /// Where it came from.
    pub source: String,
    /// When it was first saved.
    pub created: String,
    /// Its confidence.
    pub confidence: String,
}

/// `forget`, returning the fact it removed, for an Undo (`restore`).
pub fn forget_kept(data_dir: &Path, slug: &str) -> Result<Gone, MemoryError> {
    check_slug(slug)?;
    let _lock = lock(data_dir)?;
    let file = read_lossy(&fact_path(data_dir, slug))?;
    let get = |key, or: &str| meta(&file, key).unwrap_or(or).to_owned();
    let gone = Gone {
        slug: slug.to_owned(),
        text: body(&file),
        kind: get("type", "user"),
        source: get("source", "the desktop app"),
        created: get("created", &larch::bench::utc_now()),
        confidence: get("confidence", "medium"),
    };
    forget_locked(data_dir, slug)?;
    Ok(gone)
}

/// Undoes a `forget_kept`: the fact comes back under its slug, with its type, source and
/// `created`. Refused (`Known`) when a fact holds the slug again.
pub fn restore(data_dir: &Path, gone: &Gone) -> Result<(), MemoryError> {
    check_slug(&gone.slug)?;
    check_fact(capture::sentence_of(&gone.text), &gone.kind)?;
    let _lock = lock(data_dir)?;
    if listed(&dir(data_dir).join("facts"), &format!("{}.md", gone.slug))? {
        return Err(MemoryError::Known(gone.slug.clone()));
    }
    write_fact(
        data_dir,
        &gone.slug,
        &gone.kind,
        &gone.created,
        &gone.source,
        &gone.confidence,
        &gone.text,
    )
}

/// `update`, with what it replaced; with `source`, the fact now comes from there (a capture's
/// correction), its type kept, and memory must be on. Either way it moves `rev`, so open chats
/// show the edit.
pub(crate) fn updated(
    data_dir: &Path,
    slug: &str,
    text: &str,
    source: Option<&str>,
) -> Result<Saved, MemoryError> {
    check_slug(slug)?;
    let text = text.trim();
    if self::slug(text).is_empty() {
        return Err(MemoryError::Empty);
    }
    let _lock = lock(data_dir)?;
    if source.is_some() {
        check_on(data_dir)?;
    }
    // Byte for byte, as `forget` names a fact.
    if !listed(&dir(data_dir).join("facts"), &format!("{slug}.md"))? {
        return Err(MemoryError::NoFact(slug.into()));
    }
    let old = read_lossy(&fact_path(data_dir, slug))?;
    // A hand-written file has no front matter: it gets `remember`'s defaults.
    let keep = |key, or: &str| meta(&old, key).unwrap_or(or).to_owned();
    let created = keep("created", &larch::bench::utc_now());
    write_fact(
        data_dir,
        slug,
        &keep("type", "user"),
        &created,
        &source.map_or_else(|| keep("source", "the desktop app"), str::to_owned),
        &keep("confidence", "medium"),
        text,
    )?;
    bump(data_dir)?;
    Ok(Saved::of(slug.to_owned(), Some(&old), created))
}

/// The facts with a `MEMORY.md` line and a file, in `MEMORY.md` order.
pub fn list(data_dir: &Path) -> Result<Vec<Fact>, MemoryError> {
    let index = read_lossy(&dir(data_dir).join("MEMORY.md"))?;
    let mut facts: Vec<Fact> = Vec::new();
    for slug in index.lines().filter_map(line_slug).filter(|s| is_slug(s)) {
        if facts.iter().any(|f| f.slug == slug) {
            continue;
        }
        // A missing file reads empty: no fact, or one a `forget` deleted while this read. A file
        // that cannot be read (a folder by that name, no permission) is skipped too, so one bad
        // file does not hide every fact.
        let Ok(file) = read_lossy(&fact_path(data_dir, slug)) else {
            continue;
        };
        if file.trim().is_empty() {
            continue;
        }
        let get = |key| meta(&file, key).unwrap_or_default().to_owned();
        facts.push(Fact {
            slug: slug.to_owned(),
            text: body(&file),
            kind: get("type"),
            created: get("created"),
            source: get("source"),
        });
    }
    Ok(facts)
}

/// The value of `key` in a fact file's front matter.
fn meta<'a>(file: &'a str, key: &str) -> Option<&'a str> {
    front_matter(file)
        .into_iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}

/// Whether a fact file opens with front matter, as `front_matter` reads it.
fn has_front_matter(file: &str) -> bool {
    file.lines().next() == Some("---")
}

/// A fact file's text, after its front matter: after the second line that is `---`, read line by
/// line as `front_matter` reads it, so `\r\n` line ends count too. A file without front matter
/// is all text.
fn body(file: &str) -> String {
    let mut fences = 0;
    let mut at = 0;
    if has_front_matter(file) {
        for line in file.split_inclusive('\n') {
            at += line.len();
            if line.trim_end_matches(['\r', '\n']) == "---" {
                fences += 1;
                if fences == 2 {
                    break;
                }
            }
        }
    }
    let text = if fences == 2 { &file[at..] } else { file };
    text.trim().to_owned()
}

/// Writes the fact file `slug`, then its `MEMORY.md` line at the end, in place of its old line
/// if it has one (a fact with none: a new fact, or a crash between the two writes). Called under
/// the lock.
fn write_fact(
    data_dir: &Path,
    slug: &str,
    kind: &str,
    created: &str,
    source: &str,
    confidence: &str,
    text: &str,
) -> Result<(), MemoryError> {
    let memory_md = dir(data_dir).join("MEMORY.md");
    let mut lines: Vec<String> = memory_md_text(&memory_md)?
        .lines()
        .map(String::from)
        .collect();
    fs::create_dir_all(dir(data_dir).join("facts"))?;
    write_through(
        &fact_path(data_dir, slug),
        &format!(
            "---\ntype: {kind}\ncreated: {created}\nsource: {source}\nconfidence: {confidence}\n---\n{text}\n"
        ),
    )?;
    // A captured fact's line is its sentence; the user's own (`Said:`) stays in the file.
    let shown = text
        .split_once(capture::SAID)
        .map_or(text, |(fact, _)| fact);
    let line = format!("- [{slug}] {}", sentence(shown));
    // Last, in place of its old line: an update counts as newest, so the prompt shows it.
    lines.retain(|l| line_slug(l) != Some(slug));
    lines.push(line);
    Ok(write_through(&memory_md, &joined(&lines))?)
}

/// A save the user can undo: what a chat's `remember` saved, as the app shows it ("Saved: ... ·
/// Undo"), and what `undo` puts back.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Receipt {
    /// The fact's slug.
    pub slug: String,
    /// The text saved, as the fact file holds it.
    pub text: String,
    /// The text the fact had before, when the save updated it; `None` for a new fact.
    pub old: Option<String>,
    /// Its type and source before, when the save updated it.
    pub was: Option<Was>,
    /// Its `created` after the save: a fact forgotten and saved again under the slug has
    /// another.
    pub created: String,
}

/// Undoes the save `receipt` names: a new fact is forgotten, an updated one gets its old text,
/// type and source back. Only while the fact is as the save left it, checked under the lock:
/// after an edit, a later save or a forget it refuses (`MemoryError::Changed`), so an Undo never
/// loses a change made after it.
pub fn undo(data_dir: &Path, receipt: &Receipt) -> Result<(), MemoryError> {
    let slug = receipt.slug.as_str();
    check_slug(slug)?;
    let _lock = lock(data_dir)?;
    let changed = || MemoryError::Changed(slug.into());
    if !listed(&dir(data_dir).join("facts"), &format!("{slug}.md"))? {
        return Err(changed());
    }
    let file = read_lossy(&fact_path(data_dir, slug))?;
    let same = has_front_matter(&file)
        && body(&file) == receipt.text.trim()
        && meta(&file, "created") == Some(receipt.created.as_str());
    if !same {
        return Err(changed());
    }
    match (&receipt.old, &receipt.was) {
        (None, _) => forget_locked(data_dir, slug),
        (Some(old), was) => {
            let keep = |key, or: &str| meta(&file, key).unwrap_or(or).to_owned();
            let (kind, source) = match was {
                Some(w) => (w.kind.clone(), w.source.clone()),
                None => (keep("type", "user"), keep("source", "the desktop app")),
            };
            let confidence = keep("confidence", "medium");
            write_fact(
                data_dir,
                slug,
                &kind,
                &receipt.created,
                &source,
                &confidence,
                old,
            )?;
            bump(data_dir)
        }
    }
}

/// Deletes the fact file and its `MEMORY.md` line.
pub fn forget(data_dir: &Path, slug: &str) -> Result<(), MemoryError> {
    check_slug(slug)?;
    let _lock = lock(data_dir)?;
    forget_locked(data_dir, slug)
}

/// `forget`, under the lock.
fn forget_locked(data_dir: &Path, slug: &str) -> Result<(), MemoryError> {
    let memory_md = dir(data_dir).join("MEMORY.md");
    let text = memory_md_text(&memory_md)?;
    let kept: Vec<&str> = text
        .lines()
        .filter(|l| line_slug(l) != Some(slug))
        .collect();
    let line = kept.len() < text.lines().count();
    // A fact is deleted only for a name that is its own byte for byte, its MEMORY.md line or,
    // for a fact without a line, its listed file name: APFS and NTFS open one file for names
    // that differ in bytes ('ſ' is 's' there). The delete itself goes through the path, as HFS+
    // lists names in another form (decomposed) than the one written.
    if !line
        && (!listed(&dir(data_dir).join("facts"), &format!("{slug}.md"))?
            || a_line_names_its_file(data_dir, &text, slug)?)
    {
        return Err(MemoryError::NoFact(slug.into()));
    }
    match fs::remove_file(fact_path(data_dir, slug)) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
        _ => {}
    }
    if line {
        write_through(&memory_md, &joined(&kept))?;
    }
    bump(data_dir)
}

/// Whether a `MEMORY.md` line in `text` names the fact file of the listed `slug` by other bytes:
/// HFS+ lists the decomposed form of a composed slug, and deleting through it keeps the line.
#[cfg_attr(not(unix), allow(unused_variables))]
fn a_line_names_its_file(data_dir: &Path, text: &str, slug: &str) -> io::Result<bool> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let id = |m: &fs::Metadata| (m.dev(), m.ino());
        let file = id(&fs::metadata(fact_path(data_dir, slug))?);
        for other in text.lines().filter_map(line_slug).filter(|s| is_slug(s)) {
            match fs::metadata(fact_path(data_dir, other)) {
                Ok(m) if id(&m) == file => return Ok(true),
                Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e),
                _ => {}
            }
        }
    }
    // NTFS lists every name as it was written.
    Ok(false)
}

/// Whether `folder` lists `name`, byte for byte.
fn listed(folder: &Path, name: &str) -> io::Result<bool> {
    let entries = match fs::read_dir(folder) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e),
    };
    for entry in entries {
        if entry?.file_name() == name {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_slug_has_no_upper_case() {
        // On a case-insensitive file system an upper-case slug opens the fact's file, while
        // MEMORY.md holds its line under the lower-case one.
        assert!(super::is_slug("the-user-drinks-green-tea-2"));
        assert!(super::is_slug("柏林的用户喜欢写代码"));
        assert!(!super::is_slug("THE-USER-DRINKS-GREEN-TEA"));
        assert!(!super::is_slug("the-User"));
    }
}
