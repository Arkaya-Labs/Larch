//! `index.sqlite`: one FTS5 row per line of `facts/`, `daily/` and the engine's `chat/<id>.jsonl`
//! transcripts (read, never written), plus one per key phrase in `keys/`, ranked by BM25. Each
//! row carries its source and, for facts, notes and keys, its day. Derived: deleting it loses
//! nothing. Memory writes never touch it: what they change is newer than the index, so the next
//! recall rebuilds it from the names on disk, and a transcript that cannot be read never blocks a
//! write.

use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
};

use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde_json::Value;

use super::{ALL, MemoryError, Source, dates, dir, front_matter, is_slug, lock, read_lossy};

/// One recall result: where the line is, the words around the match, and its BM25 score.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Hit {
    /// The file, relative to the data folder, with `/` separators.
    pub path: String,
    /// The line in that file, counted from 1.
    pub line: u32,
    /// Up to 16 words around the match; a key's hit has the whole note line it points at.
    pub snippet: String,
    /// BM25, higher is better.
    pub score: f64,
    /// The row that matched.
    pub source: Source,
}

/// Lines a recall returns when config.toml does not say: the research behind R6 found that
/// models of 8B lose answers when more retrieved text fills the prompt.
pub const RECALL_HITS: usize = 5;

/// What a recall reads and how much it returns.
#[derive(Clone, Copy, Debug)]
pub struct Recall<'a> {
    /// The sources whose rows it reads.
    pub sources: &'a [Source],
    /// Lines it returns at most.
    pub hits: usize,
    /// The day a date phrase counts from (`today()`, or a bench's fixed day).
    pub today: i64,
}

/// Today in days since 1970, UTC, as the daily notes are named.
pub fn today() -> i64 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    (secs / 86_400) as i64
}

/// A fact is a near-duplicate when each text covers at least this share of the other's BM25
/// weight: the new text's score against the whole fact over the fact's own words' score, and the
/// same the other way round. One way only, a longer fact that contains an old one scores 1.00 and
/// would replace it. Measured both ways on tables of facts and daily notes of 10 and 310 rows
/// (lower share): a contradiction 0.82, a reordered rephrasing 0.78 to 0.80, a heavy rephrasing
/// (three new words of eight) 0.53 to 0.55, a different fact sharing words 0.35 to 0.49, a longer
/// fact containing the old one 0.27 to 0.45, one line of a two-line fact 0.41. Raw BM25 tripled
/// between the two sizes, so no fixed raw score separates them. Set above 0.55: a merge loses the
/// old text, a missed merge only leaves a duplicate line.
// ponytail: word overlap judges sameness; add a decision model when the bench shows false or
// missed merges
const DUP_SHARE: f64 = 0.7;

/// The folders the index reads, relative to the data folder, the file extension in each and the
/// source of its rows.
const SOURCES: [(&str, &str, Source); 4] = [
    ("agent/memory/facts", "md", Source::Fact),
    ("agent/memory/daily", "md", Source::Daily),
    ("agent/memory/keys", "txt", Source::Key),
    ("chat", "jsonl", Source::Transcript),
];

type Folders<'a> = [(&'a str, &'a str, Source)];

fn index_path(data_dir: &Path) -> PathBuf {
    dir(data_dir).join("index.sqlite")
}

/// Every file in `folders`, sorted, as (path relative to the data folder, path on disk, source).
fn sources(data_dir: &Path, folders: &Folders) -> io::Result<Vec<(String, PathBuf, Source)>> {
    let mut files = Vec::new();
    for &(folder, ext, source) in folders {
        let entries = match fs::read_dir(data_dir.join(folder)) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e),
        };
        for entry in entries {
            let path = entry?.path();
            let name = path.file_name().and_then(|n| n.to_str());
            if let Some(name) = name.filter(|_| path.extension().is_some_and(|e| e == ext)) {
                files.push((format!("{folder}/{name}"), path, source));
            }
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

/// Whether the index is missing or a source file or folder changed since it was written; a
/// folder's time moves when a file in it is added or deleted. The same time counts as newer:
/// file times come from a coarse clock (one tick is milliseconds on Linux), so a note written
/// right after an index update carries the index's exact time.
// ponytail: a whole rebuild when any file is newer; refresh per file when recall latency shows it
fn stale(data_dir: &Path) -> Result<bool, MemoryError> {
    let Ok(built) = fs::metadata(index_path(data_dir)).and_then(|m| m.modified()) else {
        return Ok(true);
    };
    let folders = SOURCES.iter().map(|(folder, ..)| data_dir.join(folder));
    let files = sources(data_dir, &SOURCES)?
        .into_iter()
        .map(|(_, path, _)| path);
    if folders.chain(files).any(|p| {
        fs::metadata(p)
            .and_then(|m| m.modified())
            .is_ok_and(|t| t >= built)
    }) {
        return Ok(true);
    }
    // A folder that went leaves no time behind, and one put back keeps its old time, so the
    // index keeps the list of folders it was built from. An index whose list or columns cannot be
    // read (one built before it had them) is built again: it is derived from the files.
    let conn = Connection::open(index_path(data_dir))?;
    if conn
        .prepare("SELECT source, day, note FROM lines LIMIT 0")
        .is_err()
    {
        return Ok(true);
    }
    let Ok(built_from) = conn
        .prepare("SELECT name FROM folders ORDER BY rowid")
        .and_then(|mut names| {
            names
                .query_map([], |r| r.get(0))?
                .collect::<Result<Vec<String>, _>>()
        })
    else {
        return Ok(true);
    };
    Ok(built_from != present(data_dir)?)
}

/// The source folders that exist, in `SOURCES` order.
fn present(data_dir: &Path) -> io::Result<Vec<String>> {
    let mut folders = Vec::new();
    for (folder, ..) in SOURCES {
        match fs::metadata(data_dir.join(folder)) {
            Ok(_) => folders.push(folder.to_owned()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    Ok(folders)
}

/// A new `lines` table in `conn`, filled from the files in `folders`.
fn fill(conn: &mut Connection, data_dir: &Path, folders: &Folders) -> Result<(), MemoryError> {
    conn.execute_batch(
        "CREATE VIRTUAL TABLE lines USING fts5(text, path UNINDEXED, line UNINDEXED,
         source UNINDEXED, day UNINDEXED, note UNINDEXED)",
    )?;
    let tx = conn.transaction()?;
    for (rel, file, source) in sources(data_dir, folders)? {
        add(&tx, &rel, &file, source)?;
    }
    Ok(tx.commit()?)
}

/// Deletes the index and builds it again from the files. It is built under a temporary name and
/// renamed, so a build that fails never looks like a fresh index.
pub fn rebuild(data_dir: &Path) -> Result<(), MemoryError> {
    let _lock = lock(data_dir)?;
    build(data_dir)
}

/// `rebuild` for a caller that holds the lock.
fn build(data_dir: &Path) -> Result<(), MemoryError> {
    let path = index_path(data_dir);
    let tmp = path.with_extension("sqlite.tmp");
    // A journal a crash left would be replayed into the new file of the same name and corrupt it.
    let journal = |p: &Path| {
        let mut name = p.as_os_str().to_owned();
        name.push("-journal");
        PathBuf::from(name)
    };
    for p in [journal(&path), journal(&tmp), path.clone(), tmp.clone()] {
        match fs::remove_file(&p) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => return Err(e.into()),
            _ => {}
        }
    }
    fs::create_dir_all(dir(data_dir))?;
    // The index dates from the start of its build, by the file system's clock that dates the
    // sources: a line written to a file the build has read is then newer than the index.
    let started = File::create(&tmp)?.metadata()?.modified()?;
    let folders = present(data_dir)?;
    let mut conn = Connection::open(&tmp)?;
    fill(&mut conn, data_dir, &SOURCES)?;
    conn.execute_batch("CREATE TABLE folders (name TEXT)")?;
    for folder in folders {
        conn.execute("INSERT INTO folders (name) VALUES (?1)", [folder])?;
    }
    conn.close().map_err(|(_, e)| e)?;
    File::options()
        .write(true)
        .open(&tmp)?
        .set_modified(started)?;
    Ok(fs::rename(&tmp, &path)?)
}

/// One row per line: a fact's body lines, dated by its `created`; every daily line, dated by the
/// file's name; a key, `<line> <phrase>`, as a row of the note line it points at, dated the same
/// and holding that line's text; a transcript line's message texts, undated.
// ponytail: transcript rows carry no day, so a dated recall leaves them out; date them by the
// chat ID's start time when the recall tool's date questions miss chats
// ponytail: a key names its note line by number, as notes are only appended to; a hand edit that
// removes a line moves later keys onto other lines; keep the line's hash with its keys when
// edited notes show up
fn add(conn: &Connection, rel: &str, file: &Path, source: Source) -> Result<(), MemoryError> {
    let bytes = match fs::read(file) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        // A running chat holds its transcript locked, and Windows locks block reads
        // (ERROR_LOCK_VIOLATION); the transcript is indexed after its next change rather than
        // failing every recall.
        Err(e) if source == Source::Transcript && cfg!(windows) && e.raw_os_error() == Some(33) => {
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };
    let text = String::from_utf8_lossy(&bytes);
    let stem = Path::new(rel).file_stem().and_then(|s| s.to_str());
    let (body, day) = match source {
        Source::Fact => {
            let created = front_matter(&text)
                .into_iter()
                .find(|(k, _)| *k == "created");
            (body_start(&text), created.and_then(|(_, v)| dates::iso(v)))
        }
        Source::Daily | Source::Key => (0, stem.and_then(dates::iso)),
        Source::Transcript => (0, None),
    };
    let note = format!("agent/memory/daily/{}.md", stem.unwrap_or_default());
    let noted = match source {
        // keys/<day>.txt sits next to daily/<day>.md.
        Source::Key => match file.parent().and_then(Path::parent) {
            Some(memory) => read_lossy(
                &memory
                    .join("daily")
                    .join(format!("{}.md", stem.unwrap_or_default())),
            )?,
            None => String::new(),
        },
        _ => String::new(),
    };
    let noted: Vec<&str> = noted.lines().collect();
    let mut insert = conn.prepare(
        "INSERT INTO lines (text, path, line, source, day, note) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )?;
    for (i, line) in text.lines().enumerate().skip(body) {
        let (row, path, at, value) = match source {
            Source::Transcript => (transcript_text(line), rel, i + 1, None),
            // A line that is no key (a hand edit), or a key whose line is gone, points at nothing.
            Source::Key => {
                let Some((at, key)) = line.trim().split_once(' ') else {
                    continue;
                };
                let Some((at, value)) = at
                    .parse::<usize>()
                    .ok()
                    .and_then(|at| Some((at, noted.get(at.checked_sub(1)?)?.trim())))
                else {
                    continue;
                };
                (key.trim().to_owned(), note.as_str(), at, Some(value))
            }
            Source::Fact | Source::Daily => (line.trim().to_owned(), rel, i + 1, None),
        };
        if !row.is_empty() {
            insert.execute(params![row, path, at as i64, source.as_str(), day, value])?;
        }
    }
    Ok(())
}

/// The 0-based line where a fact's body starts, after its `---` front matter.
fn body_start(text: &str) -> usize {
    let mut lines = text.lines();
    if lines.next() != Some("---") {
        return 0;
    }
    lines.position(|l| l == "---").map_or(0, |i| i + 2)
}

/// The message texts of one transcript line: every `content` string in it but the system
/// prompt's, which holds MEMORY.md and would copy every fact into each chat, and a tool's result
/// or a call's arguments, which are what a tool saw rather than what was said and would rank a
/// file read once above the facts. Reading only the texts keeps F4's record fields out of the
/// snippets and lets its format change around them.
fn transcript_text(line: &str) -> String {
    fn contents<'a>(value: &'a Value, out: &mut Vec<&'a str>) {
        match value {
            Value::Object(o)
                if matches!(
                    o.get("role").and_then(Value::as_str),
                    Some("system" | "tool")
                ) => {}
            Value::Object(o) => o.iter().for_each(|(key, v)| match (key.as_str(), v) {
                ("content", Value::String(s)) => out.push(s),
                ("tool_calls", _) => {}
                _ => contents(v, out),
            }),
            Value::Array(a) => a.iter().for_each(|v| contents(v, out)),
            _ => {}
        }
    }
    // A torn last line (a crash mid-write) parses as nothing.
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return String::new();
    };
    let mut out = Vec::new();
    contents(&value, &mut out);
    out.join(" ")
}

/// Each word of `text` quoted and joined by OR, so FTS5 syntax in a query stays plain words.
pub(crate) fn fts_query(text: &str) -> String {
    let words: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| format!("\"{w}\""))
        .collect();
    words.join(" OR ")
}

/// The best `RECALL_HITS` lines for `query` over every source, dated from today.
pub fn recall(data_dir: &Path, query: &str) -> Result<Vec<Hit>, MemoryError> {
    let how = Recall {
        sources: &ALL,
        hits: RECALL_HITS,
        today: today(),
    };
    recall_with(data_dir, query, how)
}

/// The best `how.hits` lines for `query` in the rows of `how.sources`, rebuilding the index first
/// when it is missing or older than the files. A date phrase at the query's start or end keeps
/// the rows of its days and leaves the search; a phrase alone returns those days' lines, newest
/// first. A note line found through its keys and its own words counts once.
pub fn recall_with(data_dir: &Path, query: &str, how: Recall) -> Result<Vec<Hit>, MemoryError> {
    if how.hits == 0 || fts_query(query).is_empty() {
        return Ok(Vec::new());
    }
    // The read holds the lock too: another chat's rebuild deletes the index before it renames the
    // new one, and opening the missing file would make an empty one.
    let _lock = lock(data_dir)?;
    if stale(data_dir)? {
        build(data_dir)?;
    }
    let conn = Connection::open(index_path(data_dir))?;
    let (range, words) = dates::split(query, how.today, |event| fact_day(&conn, event, how.today))?;
    let (from, to) = range.unzip();
    let sources: String = how
        .sources
        .iter()
        .map(|s| format!(" {} ", s.as_str()))
        .collect();
    let words = fts_query(&words);
    let mut hits: Vec<Hit> = Vec::new();
    let mut take = |sql: &str, params: &[&dyn rusqlite::ToSql]| -> Result<(), MemoryError> {
        let mut stmt = conn.prepare(sql)?;
        let mut rows = stmt.query(params)?;
        while hits.len() < how.hits
            && let Some(r) = rows.next()?
        {
            let (path, line): (String, u32) = (r.get(0)?, r.get(1)?);
            let Some(source) = Source::named(&r.get::<_, String>(4)?) else {
                continue;
            };
            if !hits.iter().any(|h| h.path == path && h.line == line) {
                let (snippet, score) = (r.get(2)?, r.get(3)?);
                hits.push(Hit {
                    path,
                    line,
                    snippet,
                    score,
                    source,
                });
            }
        }
        Ok(())
    };
    // Ties go by path and line, so a rebuilt index returns the same order. A key's snippet is
    // its note line, the value.
    if !words.is_empty() {
        take(
            "SELECT path, line, coalesce(note, snippet(lines, 0, '', '', '...', 16)),
             -bm25(lines), source FROM lines WHERE lines MATCH ?1
             AND instr(?2, ' ' || source || ' ') > 0 AND (?3 IS NULL OR day BETWEEN ?3 AND ?4)
             ORDER BY rank, path, line",
            params![words, sources, from, to],
        )?;
    }
    // The other words of a dated question often name nothing in its days ("what did I do
    // yesterday"): their lines, newest first, fill the hits the words left.
    if range.is_some() {
        take(
            "SELECT path, line, text, 0.0, source FROM lines
             WHERE instr(?1, ' ' || source || ' ') > 0 AND source != 'key'
             AND day BETWEEN ?2 AND ?3 ORDER BY day DESC, path, line",
            params![sources, from, to],
        )?;
    }
    Ok(hits)
}

/// The first day a fact's body names, of the facts with the lines that best match `event` ("the
/// Lisbon trip"): what `before` and `after` an event count from. The whole body, as the event's
/// name and its date may be on different lines; the front matter is not in the index. A day
/// written without its year is the one nearest the fact's own date, so a fact saved in September
/// about "3 October" means the coming one.
fn fact_day(conn: &Connection, event: &str, today: i64) -> rusqlite::Result<Option<i64>> {
    // Every word but the articles and possessives: "the Porto trip" is not "the Lisbon trip".
    let skip = ["the", "a", "an", "my", "our", "your", "his", "her", "their"];
    let words: Vec<&str> = event.split(' ').filter(|w| !skip.contains(w)).collect();
    let words = fts_query(&words.join(" ")).replace(" OR ", " AND ");
    if words.is_empty() {
        return Ok(None);
    }
    let mut best = conn.prepare(
        "SELECT path, day FROM lines WHERE lines MATCH ?1 AND source = 'fact'
         ORDER BY rank, path, line LIMIT 5",
    )?;
    let mut paths: Vec<(String, Option<i64>)> = Vec::new();
    for row in best.query_map([words], |r| Ok((r.get(0)?, r.get(1)?)))? {
        let row = row?;
        if !paths.iter().any(|(p, _)| *p == row.0) {
            paths.push(row);
        }
    }
    let mut body =
        conn.prepare("SELECT text FROM lines WHERE path = ?1 AND source = 'fact' ORDER BY line")?;
    for (path, day) in paths {
        let lines = body.query_map([path], |r| r.get::<_, String>(0))?;
        let text = lines.collect::<rusqlite::Result<Vec<_>>>()?.join("\n");
        // Half a year on, the latest such month not after it is the nearest one.
        if let Some(found) = dates::first_day(&text, day.unwrap_or(today) + 182) {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

/// Facts scored as near-duplicates of a new text, in the order of their best line: the best
/// line alone may belong to another fact that shares the words.
const CANDIDATES: usize = 5;

/// The slug of the first of the `CANDIDATES` facts with the best lines for `text` that the text
/// and each covers `DUP_SHARE` of the other. Scored in memory on facts and daily notes, the table
/// `DUP_SHARE` was measured on: the engine's transcripts need not be readable for a write.
pub(super) fn near_duplicate(data_dir: &Path, text: &str) -> Result<Option<String>, MemoryError> {
    let query = fts_query(text);
    if query.is_empty() {
        return Ok(None);
    }
    let mut conn = Connection::open_in_memory()?;
    fill(&mut conn, data_dir, &SOURCES[..2])?;
    let mut facts: Vec<(String, String)> = Vec::new();
    {
        let mut lines = conn.prepare(
            "SELECT path FROM lines
             WHERE lines MATCH ?1 AND source = 'fact' ORDER BY rank, path",
        )?;
        for path in lines.query_map([&query], |r| r.get::<_, String>(0))? {
            let path = path?;
            // A file no slug names (a note the user keeps there) is no fact to update.
            // ponytail: the slug is the name the folder lists; HFS+ lists an accented name
            // decomposed, which is no slug, so that fact is added again instead of updated;
            // compose listed names when a data folder on HFS+ shows up
            let slug = Path::new(&path).file_stem().and_then(|s| s.to_str());
            let Some(slug) = slug.filter(|s| is_slug(s)).map(String::from) else {
                continue;
            };
            if !facts.iter().any(|(p, _)| *p == path) {
                facts.push((path, slug));
            }
            if facts.len() == CANDIDATES {
                break;
            }
        }
    }
    for (path, slug) in facts {
        // Rolled back on drop, so each candidate is scored on the same table.
        let tx = conn.transaction()?;
        if share(&tx, text, &path)? >= DUP_SHARE {
            return Ok(Some(slug));
        }
    }
    Ok(None)
}

/// The smaller of the shares that `text` and the fact at `path` each cover of the other's BM25
/// weight.
fn share(conn: &Connection, text: &str, path: &str) -> rusqlite::Result<f64> {
    // The whole fact as one row in place of its lines: an update replaces every line, so one
    // line of text must not cover a fact of several. The new text gets a row too, so both are
    // scored on one table, and so does an empty text: BM25 weighs a word in 1 row of 2 at zero,
    // so in a new memory every word would weigh the same and a fact sharing most words would
    // match. With fewer than about three other fact or note lines, a word in both texts still
    // weighs next to nothing, so only the same words match: a contradiction or a rephrasing is
    // then added, not merged, the side a wrong decision should fall on.
    let old = conn
        .prepare("SELECT text FROM lines WHERE path = ?1 ORDER BY line")?
        .query_map([path], |r| r.get::<_, String>(0))?
        .collect::<Result<Vec<_>, _>>()?
        .join("\n");
    conn.execute("DELETE FROM lines WHERE path = ?1", [path])?;
    conn.execute(
        "INSERT INTO lines (text, path, line) VALUES (?1, '', 0), (?2, ?3, 0), ('', '-', 0)",
        [text, &old, path],
    )?;
    let score = |words: &str, path: &str| -> rusqlite::Result<f64> {
        let score = conn.query_row(
            "SELECT -bm25(lines) FROM lines WHERE lines MATCH ?1 AND path = ?2",
            params![fts_query(words), path],
            |r| r.get(0),
        );
        Ok(score.optional()?.unwrap_or(0.0))
    };
    let covers_old = score(text, path)? / score(&old, path)?;
    let covers_new = score(&old, "")? / score(text, "")?;
    Ok(covers_old.min(covers_new))
}
