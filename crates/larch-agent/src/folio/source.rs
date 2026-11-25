//! A document's text with its pages, its source record, and what is kept to reopen it.

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        LazyLock, Once,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use larch::bench::{hex, sha256};
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::{Doc, Page, Rep, ocr};
use crate::{skills::read_papers, web::fetch};

/// The extractors' version, in every extract cache key: a change to the text they give bumps it,
/// so older entries are not used. 2 keeps a form feed after each page, and a page with no text.
pub const EXTRACTOR: u32 = 2;

/// The version of a web page's text, which fetch's Markdown gives and its cache does not version.
pub const FETCHED: u32 = 1;

/// Days a linked PDF's blob is kept after a chat last used it.
const BLOB_DAYS: u64 = 30;

/// What `text` says of a PDF the extract cache holds as one every reader failed on.
const UNREADABLE: &str = "the PDF could not be read; the file itself may be fine, as this app's \
readers refuse some PDFs that viewers open";

/// `text` as a document: its pages cut at form feeds, a page with no text kept as an empty one,
/// and each page's printed label when it shows one. Each page is tidied (`read_papers::tidy`);
/// the text keeps no form feed.
pub fn doc(index: usize, name: &str, sha256: &str, rep: Rep, text: String) -> Doc {
    // Printed page numbers are a PDF's, not a web page's or a note's.
    let printed = matches!(rep, Rep::Pdf | Rep::Ocr);
    let mut all = String::with_capacity(text.len());
    let pages = (text.split('\u{c}'))
        .map(|page| {
            let page = read_papers::tidy(page);
            let start = all.len();
            all.push_str(&page);
            Page {
                start,
                end: all.len(),
                label: label(&page).filter(|_| printed),
            }
        })
        .collect();
    Doc {
        index,
        name: name.into(),
        sha256: sha256.into(),
        rep,
        text: all,
        pages,
    }
}

/// The label `page` prints: its last, else its first, line when that is only a page number or a
/// roman numeral, as written or as "Page 12", "12 of 130", "- iv -" or "F-3".
// ponytail: a page that ends on a table's lone number takes it as its label; check each label
// against its neighbours' when a citation shows a wrong page
fn label(page: &str) -> Option<String> {
    static PRINTED: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"^(?:[-–—]\s*)?(?:(?i:page|pg\.?|p\.)\s*)?([A-Z]-\d{1,4}|\d{1,4}|[ivxlcdm]{1,8})(?:\s*(?i:of|/)\s*\d{1,5})?(?:\s*[-–—])?$",
        )
        .expect("a valid pattern")
    });
    // A word of roman letters is a numeral only when it is a well-formed one: "mix" is, "dim" is
    // not.
    static ROMAN: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$")
            .expect("a valid pattern")
    });
    let mut lines = page.lines().map(str::trim).filter(|l| !l.is_empty());
    let first = lines.next()?;
    [lines.next_back(), Some(first)]
        .into_iter()
        .flatten()
        .find_map(|line| {
            let label = PRINTED.captures(line)?.get(1)?.as_str();
            let roman = label.starts_with(|c: char| c.is_ascii_lowercase());
            (!roman || ROMAN.is_match(label)).then(|| label.to_owned())
        })
}

/// `raw`, an extractor's text with a form feed after each page (`read_papers::text_of`), with
/// one between pages instead, as `doc` cuts it.
pub fn between(raw: &str) -> &str {
    raw.strip_suffix('\u{c}').unwrap_or(raw)
}

/// A local file's size and change time when it was read, which key its cached text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stamp {
    /// Its size in bytes.
    pub size: u64,
    /// Its change time, in ns since the Unix epoch.
    pub modified: u64,
}

impl Stamp {
    /// `meta`'s stamp; none where the system gives no change time.
    pub fn of(meta: &fs::Metadata) -> Option<Stamp> {
        let modified = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
        Some(Stamp {
            size: meta.len(),
            modified: u64::try_from(modified.as_nanos()).ok()?,
        })
    }
}

/// Where the extract cache keeps the text of `file` with `stamp`: `<data>/agent/extract/`, by a
/// hash of its path, its stamp and `EXTRACTOR`, so a changed file or an older extractor misses.
fn cached_at(data_dir: &Path, file: &Path, stamp: Stamp) -> PathBuf {
    let key = format!(
        "{}\0{}\0{}\0{EXTRACTOR}",
        file.display(),
        stamp.size,
        stamp.modified
    );
    (data_dir.join("agent/extract")).join(format!("{}.txt", hex(&sha256(key.as_bytes()))))
}

/// The cached text of `file` with `stamp`, as its extractor gave it (a form feed after each PDF
/// page); empty for a PDF that every reader failed on, which ask_docs caches so it is tried once.
pub fn cached(data_dir: &Path, file: &Path, stamp: Stamp) -> Option<String> {
    fs::read_to_string(cached_at(data_dir, file, stamp)).ok()
}

/// Keeps `text`, the text of `file` with `stamp`, in the extract cache, whole or not at all.
pub fn save(data_dir: &Path, file: &Path, stamp: Stamp, text: &str) -> io::Result<()> {
    let path = cached_at(data_dir, file, stamp);
    fs::create_dir_all(path.parent().expect("a file in agent/extract"))?;
    crate::write_atomic(&path, text, true)
}

/// A local file's text as `text` read it, and what its source record entry keeps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extract {
    /// The text: a PDF's with a form feed between pages (`between`), any other file's as it is.
    pub text: String,
    /// `Rep::Pdf` for a PDF, else `Rep::Text`.
    pub rep: Rep,
    /// The SHA-256 of the file's bytes, lowercase hex.
    pub sha256: String,
    /// Its size and change time when it was read.
    pub stamp: Stamp,
    /// Some page's text was read from its pictures (`Rep::Ocr`).
    pub ocr: bool,
}

/// The text of `file`, as read_papers reads it (`read_papers::text_of`): from the extract cache
/// under `data_dir` when this extractor read the file before, else read and cached. With
/// `pictures`, a PDF's thin pages get the text this machine's OCR reads from their pictures
/// (`with_pictures`), and `pictures` hears each page. `stop` ends an extraction. A failure is not
/// cached, so it is tried at each read and says why; a PDF that ask_docs cached as unreadable
/// stays so until the file changes.
pub fn text(
    data_dir: &Path,
    file: &Path,
    stop: &AtomicBool,
    pictures: Option<&dyn Fn(&str)>,
) -> Result<Extract, String> {
    let meta = fs::metadata(file).map_err(|e| e.to_string())?;
    let stamp = Stamp::of(&meta).ok_or_else(|| "the file has no change time".to_owned())?;
    let bytes = read_papers::bytes_of(file)?;
    let rep = match read_papers::is_pdf(file, &bytes) {
        true => Rep::Pdf,
        false => Rep::Text,
    };
    let text = match cached(data_dir, file, stamp) {
        // Every page of a PDF read ends with a form feed, so only a failure is empty.
        Some(text) if rep == Rep::Pdf && text.is_empty() => return Err(UNREADABLE.into()),
        Some(text) => text,
        None => {
            let text = read_papers::text_in(file, &bytes, stop)?;
            // A cache that cannot be written only costs the next read another extraction.
            let _ = save(data_dir, file, stamp, &text);
            text
        }
    };
    let sha = hex(&sha256(&bytes));
    let (text, ocr) = match (rep, pictures) {
        // The extract cache keeps the text layer alone, as ask_docs reads it; the OCR cache keeps
        // each page's pictures' text beside it.
        (Rep::Pdf, Some(progress)) if ocr::available() => {
            with_pictures(data_dir, Some(file), &sha, between(&text), stop, progress)?
        }
        (Rep::Pdf, _) => (between(&text).to_owned(), false),
        _ => (text, false),
    };
    Ok(Extract {
        text,
        rep,
        sha256: sha,
        stamp,
        ocr,
    })
}

/// `text`, a PDF's with a form feed between pages, with the text OCR reads from the pictures of
/// each page whose text layer is too thin (`ocr::needs_ocr`) after that layer, marked as read
/// from a picture, so no page moves; and whether a page got some. The pictures are `file`'s, or
/// only the OCR cache's when it is not there as it was read. `progress` hears each page as it is
/// read. A page OCR fails on keeps its text layer; the failure is the file's only on `stop`, or
/// when no page has text without it.
// ponytail: one page at a time; OCR pages at once when the Mac's scanned contract shows a
// summary waiting on it
fn with_pictures(
    data_dir: &Path,
    file: Option<&Path>,
    sha256: &str,
    text: &str,
    stop: &AtomicBool,
    progress: &dyn Fn(&str),
) -> Result<(String, bool), String> {
    let mut pages: Vec<String> = text.split('\u{c}').map(str::to_owned).collect();
    let thin: Vec<usize> = (0..pages.len())
        .filter(|&i| ocr::needs_ocr(&pages[i]))
        .collect();
    let (mut any, mut failed) = (false, None);
    for (n, &i) in thin.iter().enumerate() {
        if stop.load(Ordering::Relaxed) {
            return Err(larch::EngineError::Cancelled.to_string());
        }
        progress(&format!(
            "Reading scanned pages: {} of {}",
            n + 1,
            thin.len()
        ));
        match ocr::read(file, sha256, i, data_dir, stop) {
            Ok(read) if read.lines.is_empty() => {}
            Ok(read) => {
                let own = pages[i].trim_end();
                pages[i] = match own.trim().is_empty() {
                    true => format!("{}\n", read.marked()),
                    false => format!("{own}\n{}\n", read.marked()),
                };
                any = true;
            }
            Err(e) if stop.load(Ordering::Relaxed) => return Err(e),
            Err(e) => failed = failed.or(Some(e)),
        }
    }
    let text = pages.join("\u{c}");
    match failed {
        Some(e) if text.trim().is_empty() => Err(e),
        _ => Ok((text, any)),
    }
}

/// The text of `page`, a PDF link read with follow-ups on, with the text OCR reads from the
/// pictures of its thin pages, from its blob, as `text` reads a file's (`with_pictures`); and
/// whether a page got some.
pub fn link_pictures(
    data_dir: &Path,
    page: &fetch::Page,
    stop: &AtomicBool,
    progress: &dyn Fn(&str),
) -> Result<(String, bool), String> {
    let blob = blob(data_dir, &page.sha256);
    let original = Some(blob.as_path()).filter(|b| b.is_file());
    with_pictures(data_dir, original, &page.sha256, &page.text, stop, progress)
}

/// One document a chat read, as the chat's source record keeps it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// Its letter in the chat ("A"), which `add` gives it.
    pub letter: String,
    /// The file name or URL, as the user gave it.
    pub name: String,
    /// The local file's real path, or the URL read (`fetch::Page::url`).
    pub source: String,
    /// The SHA-256 of the original bytes, lowercase hex.
    pub sha256: String,
    /// Each representation read, with the version of what made it.
    pub reps: Vec<(Rep, u32)>,
    /// Its page count.
    pub pages: usize,
    /// A local file's size and change time when it was read, which find its cached text; none
    /// for a link.
    pub stamp: Option<Stamp>,
}

impl Entry {
    /// The entry for the local file at `path`, which the user named `name`, as `text` read it.
    pub fn file(name: &str, path: &Path, read: &Extract) -> Entry {
        Entry {
            letter: String::new(),
            name: name.into(),
            source: path.display().to_string(),
            sha256: read.sha256.clone(),
            reps: [(read.rep, EXTRACTOR)]
                .into_iter()
                .chain(read.ocr.then_some((Rep::Ocr, ocr::VERSION)))
                .collect(),
            pages: pages(&read.text),
            stamp: Some(read.stamp),
        }
    }

    /// The entry for the link the user named `name`, read as `page` by `fetch::fetch_documents`,
    /// a PDF or a web page; with `pictures` when some page's text was read from its pictures
    /// (`link_pictures`).
    pub fn link(name: &str, page: &fetch::Page, pictures: bool) -> Entry {
        let rep = match page.pdf {
            true => (Rep::Pdf, EXTRACTOR),
            false => (Rep::Html, FETCHED),
        };
        Entry {
            letter: String::new(),
            name: name.into(),
            source: page.url.clone(),
            sha256: page.sha256.clone(),
            reps: [rep]
                .into_iter()
                .chain(pictures.then_some((Rep::Ocr, ocr::VERSION)))
                .collect(),
            pages: pages(&page.text),
            stamp: None,
        }
    }
}

/// The pages of `text`, which has a form feed between pages.
fn pages(text: &str) -> usize {
    text.matches('\u{c}').count() + 1
}

/// Where chat `chat_id`'s source record is kept: `<data>/agent/docs/<chat id>.json`.
pub fn record_at(data_dir: &Path, chat_id: &str) -> PathBuf {
    data_dir.join("agent/docs").join(format!("{chat_id}.json"))
}

/// Chat `chat_id`'s source record: its documents in letter order, entry `i` the document with
/// index `i`; empty when the chat has read none.
pub fn load(data_dir: &Path, chat_id: &str) -> io::Result<Vec<Entry>> {
    match fs::read(record_at(data_dir, chat_id)) {
        Ok(json) => serde_json::from_slice(&json).map_err(io::Error::other),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

/// Adds `entry` to chat `chat_id`'s source record, or puts it in the place, and the letter, of
/// the entry with its source, which the chat read again; the document's index.
pub fn add(data_dir: &Path, chat_id: &str, mut entry: Entry) -> io::Result<usize> {
    let mut docs = load(data_dir, chat_id)?;
    let index = (docs.iter().position(|d| d.source == entry.source)).unwrap_or(docs.len());
    entry.letter = doc(index, "", "", Rep::Text, String::new()).letter();
    match docs.get_mut(index) {
        Some(old) => *old = entry,
        None => docs.push(entry),
    }
    let path = record_at(data_dir, chat_id);
    fs::create_dir_all(path.parent().expect("a file in agent/docs"))?;
    let json = serde_json::to_string_pretty(&docs).map_err(io::Error::other)?;
    crate::write_atomic(&path, &json, true)?;
    Ok(index)
}

/// Deletes what the extract cache keeps of the file at `path`: its text and the text OCR read from
/// its pictures, for each time a chat's source record says it read it. The records themselves
/// stay, so a document keeps its letter. Returns the SHA-256 of each snapshot that was read.
pub fn forget(data_dir: &Path, path: &Path) -> io::Result<Vec<String>> {
    let Ok(records) = fs::read_dir(data_dir.join("agent/docs")) else {
        return Ok(Vec::new());
    };
    let source = path.display().to_string();
    let mut shas = Vec::new();
    for record in records.flatten() {
        let Ok(json) = fs::read(record.path()) else {
            continue;
        };
        // A blob folder or a record being rewritten is no record; the next call sees it whole.
        let Ok(entries) = serde_json::from_slice::<Vec<Entry>>(&json) else {
            continue;
        };
        for e in entries.iter().filter(|e| e.source == source) {
            if let Some(stamp) = e.stamp {
                remove(&cached_at(data_dir, path, stamp))?;
            }
            shas.push(e.sha256.clone());
        }
    }
    shas.sort();
    shas.dedup();
    if let Ok(extracts) = fs::read_dir(data_dir.join("agent/extract")) {
        for f in extracts.flatten() {
            let name = f.file_name().to_string_lossy().into_owned();
            if shas.iter().any(|h| name.starts_with(&format!("{h}.ocr"))) {
                remove(&f.path())?;
            }
        }
    }
    Ok(shas)
}

/// Deletes chat `chat_id`'s source record and its plan record (`cache::plan`), with the chat, and
/// what it kept of the links it read: their pin files and a linked PDF's bytes, unless another
/// chat's record still names the same snapshot (#590 F10). Attached files go through `forget`.
pub fn forget_chat(data_dir: &Path, chat_id: &str) -> io::Result<()> {
    let dir = super::cache::dir(data_dir);
    super::cache::keep_plan(&dir, chat_id, None)?;
    let links: Vec<String> = (load(data_dir, chat_id).unwrap_or_default().into_iter())
        .filter(|e| e.stamp.is_none())
        .map(|e| e.sha256)
        .collect();
    remove(&record_at(data_dir, chat_id))?;
    let mut named = Vec::new();
    if let Ok(records) = fs::read_dir(data_dir.join("agent/docs")) {
        for record in records.flatten() {
            if let Ok(entries) =
                serde_json::from_slice::<Vec<Entry>>(&fs::read(record.path()).unwrap_or_default())
            {
                named.extend(entries.into_iter().map(|e| e.sha256));
            }
        }
    }
    let gone: Vec<String> = links.into_iter().filter(|h| !named.contains(h)).collect();
    for sha in &gone {
        remove(&blob(data_dir, sha))?;
    }
    super::cache::forget(&dir, &gone).map(|_| ())
}

/// Deletes the file at `path`; a file that is already gone is deleted.
fn remove(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// A document reopened from its source record entry, and what its reader should hear about its
/// original.
#[derive(Clone, Debug)]
pub struct Opened {
    /// The document as it was read.
    pub doc: Doc,
    /// Why its original cannot be read again (moved, changed, no longer kept): its text still
    /// answers, its pictures cannot be seen.
    pub note: Option<String>,
}

/// Document `index` of a chat's source record, `entry`, opened as the exact snapshot that was
/// read: from the extract or the fetch cache, or extracted again from the original bytes when
/// they are there unchanged; never from the network. `stop` ends an extraction. The error says
/// plainly why it cannot be opened.
pub fn open(
    data_dir: &Path,
    index: usize,
    entry: &Entry,
    stop: &AtomicBool,
) -> Result<Opened, String> {
    let rep = entry.reps.first().map_or(Rep::Text, |r| r.0);
    let (text, note) = match entry.stamp {
        Some(stamp) => local(data_dir, entry, rep, stamp, stop)?,
        None => linked(data_dir, entry, rep, stop)?,
    };
    Ok(Opened {
        doc: doc(index, &entry.name, &entry.sha256, rep, text),
        note,
    })
}

/// A local file's original as it is now, against what its entry read.
enum Now {
    /// There, with the bytes that were read.
    Same,
    /// There, with other bytes.
    Changed,
    /// Not there, or not readable.
    Gone,
}

/// The file at `path` as it is now, against `e`, which read it with `stamp`.
fn now(path: &Path, e: &Entry, stamp: Stamp) -> Now {
    match fs::metadata(path) {
        Err(_) => Now::Gone,
        Ok(m) if Stamp::of(&m) == Some(stamp) => Now::Same,
        // Touched, or copied back: its bytes say whether it is the file that was read.
        Ok(_) => match read_papers::bytes_of(path) {
            Ok(bytes) if hex(&sha256(&bytes)) == e.sha256 => Now::Same,
            Ok(_) => Now::Changed,
            Err(_) => Now::Gone,
        },
    }
}

/// Where `entry`'s original bytes are, as they were read: its local file while unchanged, or a
/// linked PDF's kept blob; none when it moved, changed or is no longer kept.
pub fn original(data_dir: &Path, entry: &Entry) -> Option<PathBuf> {
    match entry.stamp {
        Some(stamp) => {
            let path = PathBuf::from(&entry.source);
            matches!(now(&path, entry, stamp), Now::Same).then_some(path)
        }
        None => Some(blob(data_dir, &entry.sha256)).filter(|b| b.is_file()),
    }
}

/// The text of `e`, a local file read with `stamp` as `rep`, and a note when the file moved or
/// changed since.
fn local(
    data_dir: &Path,
    e: &Entry,
    rep: Rep,
    stamp: Stamp,
    stop: &AtomicBool,
) -> Result<(String, Option<String>), String> {
    let path = Path::new(&e.source);
    let now = now(path, e, stamp);
    // Read with its pictures' text, which the OCR cache gives again by the hash; only the file
    // that was read may fill a page the cache misses.
    let pictures = e.reps.iter().any(|r| r.0 == Rep::Ocr);
    let original = matches!(now, Now::Same).then_some(path);
    let quiet: &dyn Fn(&str) = &|_| {};
    let cached = match (cached(data_dir, path, stamp)).filter(|t| rep != Rep::Pdf || !t.is_empty())
    {
        Some(t) if rep == Rep::Pdf && pictures => Some(
            (with_pictures(data_dir, original, &e.sha256, between(&t), stop, quiet))
                .map_err(|err| format!("{} cannot be reopened: {err}", e.name))?
                .0,
        ),
        Some(t) if rep == Rep::Pdf => Some(between(&t).to_owned()),
        t => t,
    };
    let name = &e.name;
    match (cached, now) {
        (Some(text), Now::Same) => Ok((text, None)),
        (None, Now::Same) => {
            text(data_dir, path, stop, pictures.then_some(quiet)).map(|t| (t.text, None))
        }
        (Some(text), Now::Gone) => Ok((
            text,
            Some(format!(
                "{name} is no longer where it was read, so this is its text as it was read; \
                 its pictures cannot be seen."
            )),
        )),
        (Some(text), Now::Changed) => Ok((
            text,
            Some(format!(
                "{name} has changed since it was read, so this is its text as it was read then; \
                 its pictures cannot be seen."
            )),
        )),
        (None, Now::Gone) => Err(format!(
            "{name} cannot be reopened: it is no longer where it was read, and its text is not \
             kept; the user can attach it again."
        )),
        (None, Now::Changed) => Err(format!(
            "{name} cannot be reopened as it was read: it has changed since, and the text read \
             then is not kept; read it again."
        )),
    }
}

/// The text of `e`, a link read as `rep`: from the fetch cache, else from its blob, with its
/// pictures' text when it was read with it; a note when a PDF's blob is no longer kept.
fn linked(
    data_dir: &Path,
    e: &Entry,
    rep: Rep,
    stop: &AtomicBool,
) -> Result<(String, Option<String>), String> {
    let blob = blob(data_dir, &e.sha256);
    // Used now, so `prune` keeps it another `BLOB_DAYS`.
    let kept = touch(&blob).is_ok();
    let (text, note) = match fetch::cached(&e.source, &e.sha256, data_dir) {
        Ok(page) => {
            let note = (rep == Rep::Pdf && !kept).then(|| {
                format!(
                    "The original PDF of {} is no longer kept, so this is its text as it was \
                     read; its pictures cannot be seen.",
                    e.name
                )
            });
            (page.text, note)
        }
        Err(_) if kept => (
            between(&read_papers::text_of(&blob, stop)?).to_owned(),
            None,
        ),
        Err(_) => {
            return Err(format!(
                "{} cannot be reopened: neither the text read from the link nor its original is \
                 kept; read the link again.",
                e.name
            ));
        }
    };
    if !e.reps.iter().any(|r| r.0 == Rep::Ocr) {
        return Ok((text, note));
    }
    // As `local` reads a file's: the OCR cache gives it again by the hash.
    let original = kept.then_some(blob.as_path());
    let (text, _) = (with_pictures(data_dir, original, &e.sha256, &text, stop, &|_| {}))
        .map_err(|err| format!("{} cannot be reopened: {err}", e.name))?;
    Ok((text, note))
}

/// Where a linked PDF's original bytes are kept: `<data>/agent/docs/blobs/<sha256>`.
pub fn blob(data_dir: &Path, sha256: &str) -> PathBuf {
    data_dir.join("agent/docs/blobs").join(sha256)
}

/// Keeps `bytes`, a linked PDF's whose SHA-256 is `sha256`, as its blob, whole or not at all; a
/// blob already kept is marked used now instead.
pub fn keep(data_dir: &Path, sha256: &str, bytes: &[u8]) -> io::Result<()> {
    let path = blob(data_dir, sha256);
    if touch(&path).is_ok() {
        return Ok(());
    }
    let dir = path.parent().expect("a file in blobs");
    fs::create_dir_all(dir)?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
    tmp.write_all(bytes)?;
    tmp.persist(&path).map_err(|e| e.error)?;
    Ok(())
}

/// Marks the blob at `path` used now, by its change time, which `prune` reads.
fn touch(path: &Path) -> io::Result<()> {
    fs::File::options()
        .write(true)
        .open(path)?
        .set_modified(SystemTime::now())
}

/// Deletes the blobs no chat used in `BLOB_DAYS` days, at most once a process, so read_papers
/// calls it as it starts work at no cost after the first.
pub fn prune(data_dir: &Path) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let days = Duration::from_secs(BLOB_DAYS * 24 * 60 * 60);
        if let Some(before) = SystemTime::now().checked_sub(days) {
            prune_before(data_dir, before);
        }
    });
}

/// Deletes the blobs last used before `before`. A blob that cannot be deleted costs only its
/// disk, and is tried again at the next start.
fn prune_before(data_dir: &Path, before: SystemTime) {
    // No folder: no link was read yet.
    let Ok(blobs) = fs::read_dir(data_dir.join("agent/docs/blobs")) else {
        return;
    };
    for blob in blobs.flatten() {
        let old = (blob.metadata().and_then(|m| m.modified())).is_ok_and(|t| t < before);
        if old {
            let _ = fs::remove_file(blob.path());
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::folio::cite;

    /// A PDF with one page for each of `pages`, each line of a page in Helvetica, a page with no
    /// text an empty one, with its cross-reference table's offsets right.
    pub(crate) fn pdf(pages: &[&str]) -> Vec<u8> {
        let kids: Vec<String> = (0..pages.len())
            .map(|i| format!("{} 0 R", 4 + 2 * i))
            .collect();
        let mut objects = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
            format!(
                "<< /Type /Pages /Kids [{}] /Count {} >>",
                kids.join(" "),
                pages.len()
            ),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_owned(),
        ];
        for (i, page) in pages.iter().enumerate() {
            let stream: String = (page.lines().enumerate())
                .map(|(j, line)| format!("BT /F1 12 Tf 72 {} Td ({line}) Tj ET\n", 720 - 40 * j))
                .collect();
            objects.push(format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {} 0 R \
                 /Resources << /Font << /F1 3 0 R >> >> >>",
                5 + 2 * i
            ));
            objects.push(format!(
                "<< /Length {} >>\nstream\n{stream}\nendstream",
                stream.len()
            ));
        }
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, o) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend(format!("{} 0 obj\n{o}\nendobj\n", i + 1).bytes());
        }
        let xref = out.len();
        out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
        for o in offsets {
            out.extend(format!("{o:010} 00000 n \n").bytes());
        }
        out.extend(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .bytes(),
        );
        out
    }

    /// Three pages, the middle one blank, then the appendix after the references.
    const PAGES: [&str; 3] = [
        "Sparse experts route tokens\nii",
        "",
        "References\nAppendix A proofs\n3 of 3",
    ];

    fn no_stop() -> AtomicBool {
        AtomicBool::new(false)
    }

    #[test]
    fn pages_are_cut_at_form_feeds_empty_ones_kept_and_their_labels_found() {
        let text = "Title\nText\n12\n\u{c}\u{c}iv\nPreface\n\u{c}Page 3 of 9\nBody\n\u{c}\
                    Body\n- 4 -\n\u{c}12 of 130\n\u{c}F-3\n\u{c}Total\n42 of\n\u{c}civil\n\u{c}mix\n";
        let d = doc(1, "a.pdf", "ab", Rep::Pdf, text.into());
        assert!(!d.text.contains('\u{c}'));
        let labels: Vec<_> = d.pages.iter().map(|p| p.label.as_deref()).collect();
        assert_eq!(
            labels,
            [
                Some("12"),
                None,
                Some("iv"),
                Some("3"),
                Some("4"),
                Some("12"),
                Some("F-3"),
                None,
                None,
                Some("mix"),
            ]
        );
        // The empty page is an empty range where the next page starts, so no byte is on it.
        assert_eq!(d.pages[1].start, d.pages[1].end);
        assert_eq!(
            (d.page_of(d.pages[1].start), d.pages[0].end),
            (2, d.pages[2].start)
        );
        assert_eq!(&d.text[d.pages[2].start..d.pages[2].end], "iv\nPreface\n");
        // A text with no form feed is one page, and only a PDF's pages print labels.
        let note = doc(0, "a.md", "", Rep::Text, "Notes\n7\n".into());
        assert_eq!(note.pages.len(), 1);
        assert_eq!(note.pages[0].label, None);
    }

    #[test]
    fn the_built_in_reader_puts_a_form_feed_after_each_page_and_keeps_an_empty_one() {
        let raw = read_papers::pdf_text(&pdf(&PAGES)).unwrap();
        let d = doc(0, "p.pdf", "", Rep::Pdf, between(&raw).to_owned());
        assert_eq!(d.pages.len(), 3, "{raw:?}");
        assert_eq!(d.pages[1].start, d.pages[1].end);
        assert_eq!(d.pages[0].label.as_deref(), Some("ii"));
        assert!(d.text.contains("Appendix A proofs"), "{}", d.text);
    }

    #[test]
    fn a_pdf_is_read_with_its_pages_and_appendix_and_an_old_cache_entry_is_not_used() {
        let tmp = tempfile::tempdir().unwrap();
        let (data, file) = (tmp.path().join("data"), tmp.path().join("p.pdf"));
        fs::write(&file, pdf(&PAGES)).unwrap();
        let stamp = Stamp::of(&fs::metadata(&file).unwrap()).unwrap();
        // The cache's entry from before the extractor's version, which dropped page breaks.
        let old = format!("{}\0{}\0{}", file.display(), stamp.size, stamp.modified);
        let old =
            (data.join("agent/extract")).join(format!("{}.txt", hex(&sha256(old.as_bytes()))));
        fs::create_dir_all(old.parent().unwrap()).unwrap();
        fs::write(&old, "Sparse experts route tokens ii References").unwrap();
        let read = text(&data, &file, &no_stop(), None).unwrap();
        assert_eq!(read.rep, Rep::Pdf);
        assert_eq!(read.sha256, hex(&sha256(&fs::read(&file).unwrap())));
        let d = doc(0, "p.pdf", &read.sha256, read.rep, read.text.clone());
        assert_eq!(d.pages.len(), 3, "{:?}", read.text);
        assert_eq!(d.pages[1].start, d.pages[1].end);
        assert_eq!(
            d.pages
                .iter()
                .map(|p| p.label.as_deref())
                .collect::<Vec<_>>(),
            [Some("ii"), None, Some("3")],
            "{:?}",
            read.text
        );
        assert!(d.text.contains("Appendix A proofs"), "{}", d.text);
        // Its own entry is the one read next.
        fs::write(cached_at(&data, &file, stamp), "From the cache\n\u{c}").unwrap();
        assert_eq!(
            text(&data, &file, &no_stop(), None).unwrap().text,
            "From the cache\n"
        );
    }

    /// A scan (page 1, a picture only), a text layer with a table (page 2), and a caption in the
    /// text layer over a picture of a chart's labels (page 3): pdftoppm's 150 dpi renders of a
    /// text PDF, made 1-bit.
    pub(crate) fn scanned() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pages/scanned.pdf")
    }

    #[test]
    fn a_scanned_page_is_read_from_its_picture_on_its_own_page_and_kept_for_the_reopen() {
        if !ocr::available() {
            println!("skipped: {}", ocr::UNAVAILABLE);
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let (data, file) = (tmp.path().join("data"), tmp.path().join("scanned.pdf"));
        fs::copy(scanned(), &file).unwrap();
        let stop = no_stop();
        let heard = std::cell::RefCell::new(Vec::new());
        let progress = |line: &str| heard.borrow_mut().push(line.to_owned());
        let read = text(&data, &file, &stop, Some(&progress)).unwrap();
        assert_eq!(
            *heard.borrow(),
            [
                "Reading scanned pages: 1 of 2",
                "Reading scanned pages: 2 of 2"
            ]
        );
        let d = doc(0, "scanned.pdf", &read.sha256, read.rep, read.text.clone());
        assert_eq!(d.pages.len(), 3, "{}", d.text);
        let page = |i: usize| &d.text[d.pages[i].start..d.pages[i].end];
        let scan = format!("{}\nMASTER SERVICES AGREEMENT\n", ocr::mark(0));
        assert!(page(0).starts_with(&scan), "{}", page(0));
        assert!(page(0).contains("The warranty lasts seven years from delivery."));
        // A good text layer is not read again; a thin one keeps its own words first.
        assert!(page(1).starts_with("Section 4 covers") && !page(1).contains("[text read"));
        let caption = format!("Figure 3. Support requests by year\n{}\n", ocr::mark(2));
        assert!(page(2).starts_with(&caption), "{}", page(2));
        assert!(page(2).contains("1,730"), "{}", page(2));
        // The check matches a quote from the picture on its own page.
        let quote = "\"The warranty lasts seven years from delivery.\" [p. 1]";
        let checked = cite::check(quote, std::slice::from_ref(&d), &[]);
        assert_eq!((checked.matched, checked.notes), (1, Vec::<String>::new()));
        // Without OCR, the text layer alone, as before: the scan is an empty page.
        let plain = text(&data, &file, &stop, None).unwrap();
        assert!(
            !plain.ocr && plain.text.starts_with('\u{c}'),
            "{:?}",
            plain.text
        );
        // The source record names the OCR, and a reopen gives the text that was read.
        let entry = Entry::file("scanned.pdf", &file, &read);
        assert_eq!(
            entry.reps,
            [(Rep::Pdf, EXTRACTOR), (Rep::Ocr, ocr::VERSION)]
        );
        assert_eq!(open(&data, 0, &entry, &stop).unwrap().doc.text, d.text);
        // A second read is the OCR cache's: a changed entry there is what it gives, and a reopen
        // of the moved file gives it too.
        let cache = (data.join("agent/extract")).join(format!(
            "{}.ocr{}.p0.tsv",
            read.sha256,
            ocr::VERSION
        ));
        fs::write(&cache, "72\t70\t300\t81\tFrom the cache\n").unwrap();
        let cached = format!("{}\nFrom the cache\n\u{c}", ocr::mark(0));
        let again = text(&data, &file, &stop, Some(&|_| {})).unwrap();
        assert!(again.text.starts_with(&cached), "{:?}", again.text);
        fs::rename(&file, tmp.path().join("moved.pdf")).unwrap();
        let opened = open(&data, 0, &entry, &stop).unwrap();
        assert!(opened.doc.text.starts_with(&cached[..cached.len() - 1]));
        assert!(opened.note.unwrap().contains("no longer where it was read"));
        assert_eq!(original(&data, &entry), None);
    }

    #[test]
    fn a_reopened_document_is_the_snapshot_read_and_says_when_the_file_moved_or_changed() {
        let tmp = tempfile::tempdir().unwrap();
        let (data, file) = (tmp.path().join("data"), tmp.path().join("p.pdf"));
        fs::write(&file, pdf(&PAGES)).unwrap();
        let stop = no_stop();
        let read = text(&data, &file, &stop, None).unwrap();
        let entry = Entry::file("p.pdf", &file, &read);
        assert_eq!(add(&data, "chat-1", entry.clone()).unwrap(), 0);
        let other = Entry {
            source: "/elsewhere/q.pdf".into(),
            ..entry.clone()
        };
        assert_eq!(add(&data, "chat-1", other).unwrap(), 1);
        // Read again, it keeps its place and its letter.
        assert_eq!(add(&data, "chat-1", entry.clone()).unwrap(), 0);
        let record = load(&data, "chat-1").unwrap();
        assert_eq!(
            (
                record.len(),
                record[0].letter.as_str(),
                record[1].letter.as_str()
            ),
            (2, "A", "B")
        );
        assert_eq!((record[0].pages, &record[0].sha256), (3, &read.sha256));
        assert!(load(&data, "chat-2").unwrap().is_empty());

        let read_then = doc(0, "p.pdf", &read.sha256, Rep::Pdf, read.text.clone());
        let opened = open(&data, 0, &record[0], &stop).unwrap();
        assert_eq!(
            (&opened.doc.text, &opened.doc.pages),
            (&read_then.text, &read_then.pages)
        );
        assert_eq!(
            (opened.doc.sha256.as_str(), opened.note),
            (read.sha256.as_str(), None)
        );

        // Moved: the text read is still there to answer, and the reader hears why.
        let moved = tmp.path().join("moved.pdf");
        fs::rename(&file, &moved).unwrap();
        let opened = open(&data, 0, &record[0], &stop).unwrap();
        assert_eq!(opened.doc.text, read_then.text);
        assert!(opened.note.unwrap().contains("no longer where it was read"));
        // Changed in place: the text read then, and a note that says so.
        fs::write(&file, pdf(&["Other words"])).unwrap();
        let opened = open(&data, 0, &record[0], &stop).unwrap();
        assert_eq!(opened.doc.text, read_then.text);
        assert!(opened.note.unwrap().contains("has changed since"));
        // With no text kept, a moved or a changed file cannot be reopened, and an unchanged one
        // is read again.
        fs::remove_dir_all(data.join("agent/extract")).unwrap();
        let err = open(&data, 0, &record[0], &stop).unwrap_err();
        assert!(err.contains("cannot be reopened as it was read"), "{err}");
        fs::remove_file(&file).unwrap();
        let err = open(&data, 0, &record[0], &stop).unwrap_err();
        assert!(err.contains("no longer where it was read"), "{err}");
        fs::rename(&moved, &file).unwrap();
        let opened = open(&data, 0, &record[0], &stop).unwrap();
        assert_eq!((opened.doc.text, opened.note), (read_then.text, None));
    }

    #[test]
    fn a_link_reopens_from_the_fetch_cache_or_its_blob_and_never_from_the_network() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path();
        let bytes = pdf(&PAGES);
        let sha = hex(&sha256(&bytes));
        // A host that never resolves: a reopen that went to the network would fail.
        let url = "https://example.invalid/p.pdf";
        let page = fetch::Page {
            url: url.into(),
            final_url: url.into(),
            fetched_at: "2026-11-13T00:00:00Z".into(),
            sha256: sha.clone(),
            title: "Sparse experts route tokens".into(),
            text:
                "Sparse experts route tokens\nii\n\u{c}\u{c}References\nAppendix A proofs\n3 of 3"
                    .into(),
            bytes: bytes.len() as u64,
            truncated: false,
            published: None,
            pdf: true,
        };
        fetch::save(&page, data).unwrap();
        keep(data, &sha, &bytes).unwrap();
        let entry = Entry::link(url, &page, false);
        assert_eq!(
            (entry.reps.as_slice(), entry.pages),
            (&[(Rep::Pdf, EXTRACTOR)][..], 3)
        );
        let stop = no_stop();
        let opened = open(data, 0, &entry, &stop).unwrap();
        assert_eq!((opened.doc.pages.len(), opened.note), (3, None));
        assert!(opened.doc.text.contains("Appendix A proofs"));
        // The saved text gone, the original's text is read again.
        fs::remove_file(fetch::saved_at(url, &sha, data)).unwrap();
        let again = open(data, 0, &entry, &stop).unwrap();
        assert_eq!((again.doc.pages.len(), again.note), (3, None));
        assert_eq!(again.doc.pages[1].start, again.doc.pages[1].end);
        // The original gone, the saved text answers and the reader hears it cannot see more.
        fetch::save(&page, data).unwrap();
        fs::remove_file(blob(data, &sha)).unwrap();
        let opened = open(data, 0, &entry, &stop).unwrap();
        assert!(opened.doc.text.contains("Appendix A proofs"));
        assert!(opened.note.unwrap().contains("no longer kept"));
        fs::remove_file(fetch::saved_at(url, &sha, data)).unwrap();
        let err = open(data, 0, &entry, &stop).unwrap_err();
        assert!(err.contains("cannot be reopened"), "{err}");
    }

    #[test]
    fn a_blob_is_kept_while_used_and_deleted_a_month_after() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path();
        keep(data, "old", b"%PDF-old").unwrap();
        keep(data, "new", b"%PDF-new").unwrap();
        let month = SystemTime::now() - Duration::from_secs(31 * 24 * 60 * 60);
        for name in ["old", "new"] {
            (fs::File::options().write(true).open(blob(data, name)))
                .and_then(|f| f.set_modified(month))
                .unwrap();
        }
        // Read again, a blob is used now.
        keep(data, "new", b"%PDF-new").unwrap();
        prune_before(
            data,
            SystemTime::now() - Duration::from_secs(30 * 24 * 60 * 60),
        );
        assert!(!blob(data, "old").exists());
        assert_eq!(fs::read(blob(data, "new")).unwrap(), b"%PDF-new");
    }
}
