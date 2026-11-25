//! The lookup index: the named blocks (`map::symbols`) of a folder's source files in an in-memory
//! FTS5 table, ranked by BM25 as research ranks a document's sections. One index is kept per
//! folder (each attempt's worktree) while its run lasts (`forget`) and brought up to date with the
//! files changed since its last use, so a step's lookup reads only what changed.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::{
        LazyLock, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use regex::Regex;
use rusqlite::{Connection, params};

use crate::{
    code::{
        checks::{self, Stamp},
        map::{self, Symbol},
    },
    memory::fts_query,
    skills::ask_docs::rank_words,
    tool::ToolError,
    tools::code::numbered,
};

/// The blocks a lookup shows whole, best first.
const HITS: usize = 5;

/// The blocks `lookup_areas` ranks: an area's best is rarely past them.
// ponytail: a fixed cap; rank more when a big repo's asks show an area with hits left out
const RANKED: usize = 1000;

/// The blocks a hit uses that a lookup shows after it.
const USES: usize = 5;

/// A name more blocks than this define says nothing of which one a call means: none is shown.
const MANY: usize = 2;

/// The folders whose indexes are kept, the last used: the attempts of the runs going on.
const KEPT: usize = 8;

/// The indexes kept, the last used last.
static INDEXES: Mutex<Vec<(PathBuf, Index)>> = Mutex::new(Vec::new());

/// A call, `name(`, after the name it is called on when it has one: `cache.get(`, `Foo::new(`.
static CALL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:([A-Za-z_$][\w$]*)\s*(?:\.|::)\s*)?([A-Za-z_$][\w$]*)\s*\(")
        .expect("a valid regex")
});

/// A string literal, whose words may be the ids and classes a script or a page names.
static STRING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"["'`]([^"'`\n]{1,200})["'`]"#).expect("a valid regex"));

/// A class or id in a CSS selector.
static SELECTOR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[.#]([A-Za-z_][\w-]*)").expect("a valid regex"));

/// A lower letter or digit before a capital: the two words of a camelCase name.
static CAMEL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([a-z0-9])([A-Z])").expect("a valid regex"));

/// One file of an index: when it was read, its blocks and the row of its first. Its text is
/// read again to show a block, so the index holds no file twice.
struct File {
    stamp: Stamp,
    symbols: Vec<Symbol>,
    first: i64,
}

/// A folder's blocks: the folder, the FTS5 table of them, and each file's.
struct Index {
    dir: PathBuf,
    db: Connection,
    files: BTreeMap<String, File>,
    /// The file of each block row, by its first row: a row's file is the last one at or before it.
    rows: BTreeMap<i64, String>,
    next: i64,
}

/// A block: its file, relative to the folder, and its index among the file's symbols.
type Block = (String, usize);

impl Index {
    /// An empty index of `dir`, its table in memory.
    fn new(dir: PathBuf) -> rusqlite::Result<Index> {
        let db = Connection::open_in_memory()?;
        db.execute_batch("CREATE VIRTUAL TABLE blocks USING fts5(text)")?;
        Ok(Index {
            dir,
            db,
            files: BTreeMap::new(),
            rows: BTreeMap::new(),
            next: 0,
        })
    }

    /// Reads again each file of its folder the map reads that changed since, and drops the
    /// gone; `stop` ends it between files, the files not read yet left for the next update.
    fn update(&mut self, stop: &AtomicBool) -> rusqlite::Result<()> {
        let (listed, _) = checks::listed(&self.dir, map::reads);
        let tx = self.db.transaction()?;
        let unrow = |tx: &rusqlite::Transaction, f: &File| {
            let rows = f.first..f.first + f.symbols.len() as i64;
            rows.into_iter().try_for_each(|r| {
                tx.execute("DELETE FROM blocks WHERE rowid = ?1", [r])
                    .map(|_| ())
            })
        };
        let gone: Vec<String> = (self.files.keys())
            .filter(|f| !listed.contains_key(*f))
            .cloned()
            .collect();
        for f in gone {
            let file = self.files.remove(&f).expect("listed above");
            unrow(&tx, &file)?;
            if !file.symbols.is_empty() {
                self.rows.remove(&file.first);
            }
        }
        for (rel, stamp) in listed {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            if self.files.get(&rel).is_some_and(|f| f.stamp == stamp) {
                continue;
            }
            if let Some(old) = self.files.remove(&rel) {
                unrow(&tx, &old)?;
                if !old.symbols.is_empty() {
                    self.rows.remove(&old.first);
                }
            }
            let Ok(text) = fs::read_to_string(self.dir.join(&rel)) else {
                continue;
            };
            let symbols = map::symbols(&rel, &text);
            let first = self.next;
            for (i, s) in symbols.iter().enumerate() {
                // The file, kind and name head the block's text, the name's words apart too:
                // "drop disc" finds dropDisc.
                let words = CAMEL.replace_all(&s.name, "$1 $2");
                let row = format!(
                    "{rel} {} {} {words}\n{}",
                    s.kind,
                    s.name,
                    &text[s.start..s.end]
                );
                tx.execute(
                    "INSERT INTO blocks (rowid, text) VALUES (?1, ?2)",
                    params![first + i as i64, row],
                )?;
            }
            self.next += symbols.len() as i64;
            let file = File {
                stamp,
                symbols,
                first,
            };
            // A file of no block takes no row, and the next file's first is the same.
            if !file.symbols.is_empty() {
                self.rows.insert(first, rel.clone());
            }
            self.files.insert(rel, file);
        }
        tx.commit()
    }

    /// The `top` blocks that match the words of `query` best, best first; none for a query
    /// without words.
    fn rank(&self, query: &str, top: usize) -> rusqlite::Result<Vec<Block>> {
        let q = fts_query(&rank_words(query).join(" "));
        if q.is_empty() {
            return Ok(Vec::new());
        }
        // Ties go by row, so a folder ranks the same way every time. FTS5's rank is BM25 negated.
        let mut best = self.db.prepare(
            "SELECT rowid FROM blocks WHERE blocks MATCH ?1 ORDER BY rank, rowid LIMIT ?2",
        )?;
        let rows: Vec<i64> = best
            .query_map(params![q, top as i64], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let of = |row: i64| {
            let (first, name) = self.rows.range(..=row).next_back()?;
            let i = (row - first) as usize;
            (i < self.files[name].symbols.len()).then(|| (name.clone(), i))
        };
        Ok(rows.into_iter().filter_map(of).collect())
    }

    /// The text of `file` as it is now; empty when it cannot be read.
    fn text(&self, file: &str) -> String {
        fs::read_to_string(self.dir.join(file)).unwrap_or_default()
    }

    /// The symbol `block` is.
    fn symbol(&self, (file, i): &Block) -> &Symbol {
        &self.files[file].symbols[*i]
    }

    /// Whether block `a` lies inside block `b`, or is it.
    fn inside(&self, a: &Block, b: &Block) -> bool {
        let (sa, sb) = (self.symbol(a), self.symbol(b));
        a.0 == b.0 && sb.start <= sa.start && sa.end <= sb.end
    }

    /// `block` whole and numbered, after its file, kind, name and lines.
    fn shown(&self, block: &Block) -> String {
        let (text, s) = (self.text(&block.0), self.symbol(block));
        let span = text.lines().skip(s.start_line - 1);
        let lines = numbered(span.take(s.end_line + 1 - s.start_line), s.start_line);
        format!(
            "{}, {} {}, lines {}-{}:\n{}",
            block.0,
            s.kind,
            s.name,
            s.start_line,
            s.end_line,
            lines.join("\n")
        )
    }

    /// Every block by the last part of its name (`cache.get` is a `get`), elements and CSS rules
    /// left out, and the elements and CSS rules by each id and class they name: built once a
    /// lookup, so `uses` reads the blocks a name has instead of every block for each call.
    fn names(&self) -> Names {
        let mut names = Names::default();
        for (f, file) in &self.files {
            for (i, s) in file.symbols.iter().enumerate() {
                let block = (f.clone(), i);
                let marks: Vec<&str> = match s.kind {
                    "element" => vec![s.name.as_str()],
                    "rule" | "at-rule" => (SELECTOR.captures_iter(&s.name))
                        .map(|m| m.get(1).expect("a group").as_str())
                        .collect(),
                    k if k.ends_with("rule") => continue,
                    _ => {
                        let last = s.name.rsplit(['.', ':']).find(|p| !p.is_empty());
                        if let Some(last) = last {
                            names.calls.entry(last.to_owned()).or_default().push(block);
                        }
                        continue;
                    }
                };
                for mark in marks {
                    let these = names.marks.entry(mark.to_owned()).or_default();
                    if these.last() != Some(&block) {
                        these.push(block.clone());
                    }
                }
            }
        }
        names
    }

    /// The blocks `hit` uses, its own file's first, up to `USES`: a function or method it calls,
    /// by the name it calls it on when it has one (`cache.get(` is a `get` of a `cache` type or
    /// module, `self.get(` one of its own file), and the elements and CSS rules whose ids and
    /// classes its strings name (`names`). A name more than `MANY` blocks have is left out.
    fn uses(&self, hit: &Block, names: &Names) -> Vec<Block> {
        let s = self.symbol(hit);
        let text = self.text(&hit.0);
        let code = text.get(s.start..s.end).unwrap_or_default();
        let apart = |b: &&Block| !self.inside(b, hit) && !self.inside(hit, b);
        let parts = |b: &Block| -> Vec<String> {
            (self.symbol(b).name.split(['.', ':']))
                .filter(|p| !p.is_empty())
                .map(String::from)
                .collect()
        };
        let stem = |b: &Block| {
            let name = b.0.rsplit('/').next().unwrap_or_default();
            name.split('.').next().unwrap_or_default().to_string()
        };
        let mut found: Vec<Block> = Vec::new();
        let mut take = |these: Vec<&Block>| {
            if these.len() <= MANY {
                let new: Vec<Block> = these
                    .into_iter()
                    .filter(|b| !found.contains(b))
                    .cloned()
                    .collect();
                found.extend(new);
            }
        };
        let none = Vec::new();
        for c in CALL.captures_iter(code) {
            let (on, name) = (c.get(1).map(|m| m.as_str()), &c[2]);
            let named = names.calls.get(name).unwrap_or(&none).iter().filter(apart);
            let these: Vec<&Block> = match on {
                None => named.collect(),
                Some("this" | "self" | "Self" | "super") => {
                    named.filter(|b| b.0 == hit.0).collect()
                }
                Some(on) => named
                    .filter(|b| {
                        let p = parts(b);
                        let of = p.len().checked_sub(2).map(|i| p[i].as_str());
                        of.is_some_and(|o| o.eq_ignore_ascii_case(on))
                            || stem(b).eq_ignore_ascii_case(on)
                    })
                    .collect(),
            };
            take(these);
        }
        for c in STRING.captures_iter(code) {
            for word in c[1].split_whitespace() {
                let word = word.trim_start_matches(['.', '#']);
                let marked = names.marks.get(word).unwrap_or(&none);
                take(marked.iter().filter(apart).collect());
            }
        }
        found.sort_by_key(|b| b.0 != hit.0);
        found.truncate(USES);
        found
    }
}

/// The blocks by name a lookup's `uses` reads (`Index::names`).
#[derive(Default)]
struct Names {
    calls: HashMap<String, Vec<Block>>,
    marks: HashMap<String, Vec<Block>>,
}

/// Runs `f` on the index of `dir`, brought up to date first as far as `stop` lets it.
fn with<T>(
    dir: &Path,
    stop: &AtomicBool,
    f: impl FnOnce(&Index) -> rusqlite::Result<T>,
) -> Result<T, ToolError> {
    let failed = |e: rusqlite::Error| ToolError::Invalid(format!("the lookup index failed: {e}"));
    let dir = fs::canonicalize(dir).map_err(|e| ToolError::Io(dir.display().to_string(), e))?;
    let taken = {
        let mut kept = INDEXES.lock().unwrap_or_else(|e| e.into_inner());
        (kept.iter().position(|(d, _)| *d == dir)).map(|at| kept.remove(at).1)
    };
    let mut index = match taken {
        Some(index) => index,
        None => Index::new(dir.clone()).map_err(failed)?,
    };
    // Outside the lock: another folder's lookup or run end never waits for this build.
    let got = index.update(stop).and_then(|()| f(&index)).map_err(failed);
    let mut kept = INDEXES.lock().unwrap_or_else(|e| e.into_inner());
    kept.push((dir, index));
    if kept.len() > KEPT {
        kept.remove(0);
    }
    got
}

/// Drops the index of `dir`, once the run in it ended: the next run reads its files afresh.
pub(crate) fn forget(dir: &Path) {
    let Ok(dir) = fs::canonicalize(dir) else {
        return;
    };
    let mut kept = INDEXES.lock().unwrap_or_else(|e| e.into_inner());
    kept.retain(|(d, _)| *d != dir);
}

/// What `lookup` shows for `query` in `dir`: the `HITS` blocks that match it best, each whole,
/// then the blocks it uses (`Index::uses`) marked `used: `; every block once.
pub(crate) fn lookup(dir: &Path, query: &str) -> Result<Vec<String>, ToolError> {
    with(dir, &AtomicBool::new(false), |index| {
        Ok(shown(index, index.rank(query, HITS)?, &mut None))
    })
}

/// The `n` areas of `dir` whose blocks match `query` best, best first: each area's index and
/// what `lookup` shows of its `HITS` best blocks among the files `area` puts in it. An area with
/// no match is none of them. At most `RANKED` blocks are ranked; `stop` ends the showing with
/// the areas shown so far.
pub(crate) fn lookup_areas(
    dir: &Path,
    query: &str,
    n: usize,
    area: impl Fn(&str) -> Option<usize>,
    stop: &AtomicBool,
) -> Result<Vec<(usize, Vec<String>)>, ToolError> {
    with(dir, stop, |index| {
        let mut best: Vec<(usize, Vec<Block>)> = Vec::new();
        for block in index.rank(query, RANKED)? {
            let Some(k) = area(&block.0) else { continue };
            match best.iter().position(|(a, _)| *a == k) {
                Some(at) if best[at].1.len() < HITS => best[at].1.push(block),
                None if best.len() < n => best.push((k, vec![block])),
                _ => {}
            }
            if best.len() == n && best.iter().all(|(_, h)| h.len() == HITS) {
                break;
            }
        }
        let (mut out, mut names) = (Vec::new(), None);
        for (k, hits) in best {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            out.push((k, shown(index, hits, &mut names)));
        }
        Ok(out)
    })
}

/// `hits` whole, each followed by the blocks it uses, from `names`, built at the first hit (`Index::uses`) marked `used: `; every block once.
fn shown(index: &Index, hits: Vec<Block>, names: &mut Option<Names>) -> Vec<String> {
    let mut seen: Vec<Block> = Vec::new();
    let mut parts = Vec::new();
    for hit in hits {
        if seen
            .iter()
            .any(|s| index.inside(&hit, s) || index.inside(s, &hit))
        {
            continue;
        }
        parts.push(index.shown(&hit));
        let names = names.get_or_insert_with(|| index.names());
        for used in index.uses(&hit, names) {
            if !seen.contains(&used) {
                parts.push(format!("used: {}", index.shown(&used)));
                seen.push(used);
            }
        }
        seen.push(hit);
    }
    parts
}

/// The files of `dir` whose blocks match `query` best, best first, at most `n`; of the files
/// read before `stop`, when it is set.
pub(crate) fn closest(
    dir: &Path,
    query: &str,
    n: usize,
    stop: &AtomicBool,
) -> Result<Vec<String>, ToolError> {
    with(dir, stop, |index| {
        let mut files: Vec<String> = Vec::new();
        for (file, _) in index.rank(query, n * HITS)? {
            if !files.contains(&file) && files.len() < n {
                files.push(file);
            }
        }
        Ok(files)
    })
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, thread, time::Duration};

    use super::*;

    /// A build with Stop set reads no file, and the next goes on from there; a run's end drops
    /// its folder's index.
    #[test]
    fn a_stopped_build_reads_nothing_and_a_forgotten_index_is_gone() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("a.rs"), "fn alpha() {}\n").unwrap();
        let kept = || {
            let dir = fs::canonicalize(tmp.path()).unwrap();
            let kept = INDEXES.lock().unwrap_or_else(|e| e.into_inner());
            (kept.iter().find(|(d, _)| *d == dir)).map(|(_, i)| i.files.len())
        };
        let stopped = closest(tmp.path(), "alpha", 3, &AtomicBool::new(true)).unwrap();
        assert!(stopped.is_empty(), "{stopped:?}");
        assert_eq!(kept(), Some(0));
        let found = closest(tmp.path(), "alpha", 3, &AtomicBool::new(false)).unwrap();
        assert_eq!(found, ["a.rs"]);
        forget(tmp.path());
        assert_eq!(kept(), None);
    }

    /// A folder's lookup runs outside the lock: another folder's run ends while it goes on.
    #[test]
    fn a_lookup_holds_no_lock_another_folder_waits_on() {
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let ((inside, entered), (go, wait)) = (mpsc::channel(), mpsc::channel::<()>());
        let (a, b) = (a.path(), b.path());
        thread::scope(|s| {
            s.spawn(move || {
                with(a, &AtomicBool::new(false), |_| {
                    inside.send(()).unwrap();
                    wait.recv().unwrap();
                    Ok(())
                })
            });
            entered.recv().unwrap();
            let (done, ended) = mpsc::channel();
            s.spawn(move || {
                forget(b);
                done.send(()).unwrap();
            });
            let waited = ended.recv_timeout(Duration::from_secs(5));
            go.send(()).unwrap();
            assert!(waited.is_ok(), "forget waited for another folder's lookup");
        });
    }
}
